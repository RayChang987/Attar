"""Netlist edits on a Design: cells and nets added or removed, pins connected or moved between nets, buffers inserted,
cells cloned; debuffer() removes the buffers and double inversions a flow put in. Design.edit() returns an Edit;
Design.debuffer() runs debuffer; Design.buffers() / Design.inverters() list the library cells for buffering.

    with d.edit() as e:
        b, n = e.insert_buffer('net5476', loads, 'BUFx2_ASAP7_75t_R')
        c, n2 = e.clone('_1234_', loads2)
    naive.legalize(d)                      # the new cells sit where they were put, not yet on the rows

Edits are collected and applied together when the `with` block ends (or by e.apply()):
  - new cells and nets are appended (cell indices N.., net indices E..): no existing cell or net index moves, so what
    refers to cells (the SDC's ports and clock sources, IO terminals, snapshots, a user's cell lists) stays valid;
  - pins stay in net order (the CSR n_ptr): new pins are appended and all pins put back in net order by one stable
    sort, so pin indices change; e.pin_map[old] is the new index of a pin of before the edit (and of a pin handle
    connect() returned);
  - the timer is released (rebuilt by the next timing()); a TimingResult of before the edit is stale and says so
    when asked for a path or pin slacks;
  - a new cell is placed where given (insert_buffer: centered on its loads; clone: on the original), not legalized;
  - a removed cell stays as a tombstone (flags ASIDE | REMOVED: engines, metrics and export skip it; its pins are
    gone, so no index moves); a net an edit empties is flagged N_REMOVED and not exported.
Masters the design does not have yet come from its cell library (as Design.resize). Clock nets are not edited: clocks
are ideal (pre-CTS), so a buffer there would mean nothing to the timer."""
import numpy as np
import torch

from .io.distill import ASIDE, D_IN, D_OUT, FIXED, N_CLOCK, N_REMOVED, PLACED, REMOVED, TERMINAL, pin_dir_counts


def _memo(L, kind):
    """A dict kept on a Library object: what is worked out from its (immutable) tables once, e.g. a master's parsed
    functions. A library read again (attar.fetch(force=True)) is a new object, so its memo starts empty."""
    return L.__dict__.setdefault('_netlist_' + kind, {})


def _lib_function(d, master):
    """{output pin: (variables, evaluator)} of a master from the design's library chain ({} when it has none).
    Parsed once per library and master (the result is shared: do not modify it)."""
    from .io.liberty import parse_function
    from .timing.sizing import lib_cell
    L, _ = lib_cell(d, master)
    if L is None:
        return {}
    memo = _memo(L, 'function')
    if master not in memo:
        out = {}
        for pn, f in (L.meta['functions'].get(master) or {}).items():
            try:
                out[pn] = parse_function(f)
            except Exception:  # noqa: BLE001  (an unparsable function: not a buffer, not an inverter)
                pass
        memo[master] = out
    return memo[master]


def _signal_pins(d, master):
    """The number of signal pins (LEF, USE SIGNAL) a library cell has, or None when the library lacks it."""
    from .timing.sizing import lib_cell
    L, c = lib_cell(d, master)
    if L is None:
        return None
    ptr = int(L.a['l_pin_ptr'][c])
    return sum(1 for j in range(len(L.pins[c])) if L.a['lp_lef'][ptr + j] and L.a['lp_use'][ptr + j] == 0)


def _single_input(d, master):
    """(input pin, output pin, inverting) when master is a one-input, one-output cell (two signal pins: not a
    tri-state buffer with its enable) that passes its input (a buffer) or inverts it (an inverter); None otherwise.
    Worked out once per library and master."""
    from .timing.sizing import lib_cell
    L, _ = lib_cell(d, master)
    if L is None:
        return _single_input_of(d, master)
    memo = _memo(L, 'single_input')
    if master not in memo:
        memo[master] = _single_input_of(d, master)
    return memo[master]


def _single_input_of(d, master):
    fs = _lib_function(d, master)
    if len(fs) != 1 or _signal_pins(d, master) != 2:
        return None
    (out, (vs, f)), = fs.items()
    if len(vs) != 1:
        return None
    v = vs[0]
    from .timing.sizing import lib_cell
    L, c = lib_cell(d, master)
    if v == out or v not in L.pins[c]:                     # an internal state (a register's IQN), not an input pin
        return None
    t = (f({v: 0}), f({v: 1}))
    if t == (0, 1):
        return v, out, False
    if t == (1, 0):
        return v, out, True
    return None


def library_cells(d, inverting):
    """Buffers (inverting=False) or inverters of the design's library chain with LEF geometry and Liberty data,
    without dont_use cells and sequential ones, smallest area first. Kept on the design for the library chain it was
    worked out for (insert_buffer asks for it on every call)."""
    from .timing.sizing import library
    chain = library(d)
    key = (bool(inverting), tuple(id(L) for L in chain))
    hit = getattr(d, '_library_cells', None)
    if hit is not None and key in hit:
        return list(hit[key])
    out, seen = [], set()
    for L in chain:
        skip = set(L.meta.get('dont_use', [])) | set(L.meta.get('sequentials', {}) or {})
        for c, name in enumerate(L.meta['cells']):
            if name in seen or name in skip or L.a['lc_file'][c] < 0 or not L.a['lc_lef'][c]:
                continue
            seen.add(name)
            s = _single_input(d, name)
            if s is not None and s[2] == inverting:
                w, h = L.a['lc_size'][c]
                out.append((float(w) * float(h), name))
    res = [n for _, n in sorted(out)]
    if hit is None:
        hit = d._library_cells = {}
    hit[key] = res
    return list(res)


class Edit:
    """A batch of netlist edits on a Design (see the module doc). Cells, nets and pins are given by index or by name
    ('inst', 'net', 'inst/PIN'); indices of new cells and nets are returned at once and stay valid after apply()."""

    def __init__(self, d):
        d.require('netlist_edit')
        self.d = d
        a = d.raw
        self.N0, self.P0, self.E0 = len(a['c_master']), len(a['p_cell']), len(a['n_ptr']) - 1
        self.cells = []        # (master index, xy lower-left um, orient, die, name)
        self.nets = []         # names
        self.pins = []         # (cell, master pin index (global), direction, net)
        self.moves = {}        # pin (index of before, or a new pin's handle) -> net
        self.dropped = set()   # pins of cells removed in this edit
        self.removed = set()   # cells removed in this edit
        self.pin_map = None    # after apply(): old pin index -> new (-1: the pin is gone)
        self._net_id = None
        self._names = set(d.names['cells']); self._net_names = None
        self._joined = {}      # net -> pins that joined it in this edit (some may have left again: checked on use)
        self._clk = None
        self._default_buffer = None
        self._applied = False
        self._pnet = d.p_net.cpu().numpy(); self._nptr = a['n_ptr'].astype(np.int64)   # host copies for lookups
        self._cpins = None     # cell -> its pins, built on first use

    # -- context manager ----------------------------------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.apply()
        return False

    # -- lookups ------------------------------------------------------------------------------------------------------
    def cell(self, c):
        from .timing.sizing import cell_index
        if isinstance(c, str):
            for k, (_, _, _, _, nm) in enumerate(self.cells):
                if nm == c:
                    return self.N0 + k
        return cell_index(self.d, c)

    def net(self, k):
        if isinstance(k, str):
            if self._net_id is None:
                self._net_id = {n: i for i, n in enumerate(self.d.names['nets'])}
                self._net_id.update({n: self.E0 + i for i, n in enumerate(self.nets)})
            if k not in self._net_id:
                raise KeyError('no net %r' % k)
            return self._net_id[k]
        k = int(k)
        if not 0 <= k < self.E0 + len(self.nets):
            raise IndexError('net %d out of range' % k)
        return k

    def pin(self, p):
        """A pin by index (of before the edit, or a handle connect() returned) or by name 'inst/PIN'."""
        if isinstance(p, str):
            inst, _, pn = p.rpartition('/')
            c = self.cell(inst)
            d = self.d
            mp = d.names['master_pins'][int(d.raw['c_master'][c])] if c < self.N0 else None
            if mp is None or pn not in mp:
                raise KeyError('no pin %r' % p)
            k = int(d.raw['m_pin_ptr'][int(d.raw['c_master'][c])]) + mp.index(pn)
            hit = np.nonzero((d.raw['p_cell'] == c) & (d.raw['p_mpin'] == k))[0]
            if not len(hit):
                raise KeyError('pin %r is not connected' % p)
            return int(hit[0])
        return int(p)

    def _pin_net(self, p):
        if p in self.moves:
            return self.moves[p]
        if p >= self.P0:
            return self.pins[p - self.P0][3]
        return int(self._pnet[p])

    def cell_pins(self, c):
        """The pins of a cell of before the edit (indices of before)."""
        if self._cpins is None:
            pc = self.d.raw['p_cell'].astype(np.int64)
            o = np.argsort(pc, kind='stable')
            self._cpins = (o, np.searchsorted(pc[o], np.arange(self.N0 + 1)))
        o, st = self._cpins
        return o[st[c]:st[c + 1]]

    def _live(self, c):
        """c, unless it was removed (now or by an earlier edit)."""
        if c in self.removed or (c < self.N0 and self.d.raw['c_flags'][c] & REMOVED):
            raise ValueError('cell %s was removed' % self.d.names['cells'][c] if c < self.N0 else c)
        return c

    def _pin_dir(self, p):
        return self.pins[p - self.P0][2] if p >= self.P0 else int(self.d.raw['p_dir'][p])

    def clock_nets(self):
        """Nets the edit leaves alone: those the source marks USE CLOCK and those driven by a clock source port."""
        if self._clk is not None:
            return self._clk
        import os
        d = self.d
        out = set(np.nonzero(d.raw['n_flags'] & N_CLOCK)[0].tolist())
        tj = os.path.join(d.path, 'timing.json')
        if os.path.exists(tj):
            clocks = d._timing_info().get('clocks', {})     # with Design.set_timing's changes
            src = {int(c) for ck in clocks.values() for c in ck.get('sources', [])}
            pc = d.raw['p_cell']
            out |= {int(d.p_net[p]) for p in np.nonzero(np.isin(pc, list(src)))[0]}
        self._clk = out
        return out

    def _unique(self, base, taken):
        """base_<k>, the first k not taken (add_cell / add_net take it)."""
        k = 0
        while '%s_%d' % (base, k) in taken:
            k += 1
        return '%s_%d' % (base, k)

    def default_buffer(self):
        """The buffer insert_buffer takes by default: the smallest buffer the design already uses (its flow's threshold
        voltage and cell choice), else the library's smallest (d.buffers()[0])."""
        if self._default_buffer is None:          # the design does not change inside an edit: worked out once
            d = self.d
            bufs = d.buffers()
            used = set(d.names['masters'][m] for m in np.unique(d.raw['c_master']) if m >= 0)
            self._default_buffer = (next((b for b in bufs if b in used), bufs[0] if bufs else None),)
        return self._default_buffer[0]

    # -- primitives ---------------------------------------------------------------------------------------------------
    def add_cell(self, master, xy, orient=0, die=0, name=None):
        """A new cell of `master` (a name; from the library when the design lacks it) at lower-left xy (um).
        Returns its index."""
        self._open()
        from .timing.sizing import add_master
        d = self.d
        ms = d.names['masters']
        m = ms.index(master) if master in ms else add_master(d, master)
        if name is None:
            name = self._unique('attar_cell', self._names)
        elif name in self._names:
            raise ValueError('a cell named %r exists' % name)
        self._names.add(name)
        self.cells.append((m, (float(xy[0]), float(xy[1])), int(orient), int(die), name))
        return self.N0 + len(self.cells) - 1

    def add_net(self, name=None):
        """A new net (no pins yet). Returns its index."""
        self._open()
        if self._net_names is None:
            self._net_names = set(self.d.names['nets'])
        if name is None:
            name = self._unique('attar_net', self._net_names)
        elif name in self._net_names:
            raise ValueError('a net named %r exists' % name)
        self._net_names.add(name)
        self.nets.append(name)
        if self._net_id is not None:
            self._net_id[name] = self.E0 + len(self.nets) - 1
        return self.E0 + len(self.nets) - 1

    def connect(self, cell, pin, net):
        """Connect pin `pin` (a master pin name) of a cell added in this edit to a net. Returns the pin's handle."""
        self._open()
        c, k = self._live(self.cell(cell)), self.net(net)
        if c < self.N0:
            raise ValueError('connect() takes a cell added in this edit; move() moves the pins of existing cells')
        d = self.d
        m = self.cells[c - self.N0][0]
        mp = d.names['master_pins'][m]
        if pin not in mp:
            raise KeyError('%s has no pin %r (pins: %s)' % (d.names['masters'][m], pin, ', '.join(mp)))
        g = int(d.raw['m_pin_ptr'][m]) + mp.index(pin)
        if any(q[0] == c and q[1] == g for q in self.pins):
            raise ValueError('pin %s of cell %d is connected already' % (pin, c))
        lef = int(d.raw['m_pin_dir'][g]) if 'm_pin_dir' in d.raw else -1
        fs = _lib_function(d, d.names['masters'][m])
        dr = D_OUT if (pin in fs or lef == D_OUT) else D_IN
        self.pins.append((c, g, dr, k))
        self._joined.setdefault(k, set()).add(self.P0 + len(self.pins) - 1)
        return self.P0 + len(self.pins) - 1

    def move(self, pins, net):
        """Move pins (indices, handles or 'inst/PIN' names) to a net."""
        self._open()
        k = self.net(net)
        for p in ([pins] if isinstance(pins, (int, np.integer, str)) else pins):
            p = self.pin(p)
            if p in self.dropped:
                raise ValueError('pin %d belongs to a removed cell' % p)
            self.moves[p] = k
            self._joined.setdefault(k, set()).add(p)

    def remove_cell(self, cell):
        """Remove a cell of before the edit: its pins leave their nets; the cell stays as a tombstone (flags ASIDE |
        REMOVED: engines, metrics and export skip it), so no index moves. IO pins and fixed cells stay."""
        self._open()
        c = self._live(self.cell(cell))
        if c >= self.N0:
            raise ValueError('remove_cell() takes a cell of before the edit')
        if self.d.raw['c_flags'][c] & (TERMINAL | FIXED):
            raise ValueError('%s is an IO pin or fixed: it stays' % self.d.names['cells'][c])
        self.removed.add(c)
        self.dropped.update(self.cell_pins(c).tolist())

    # -- transformations ------------------------------------------------------------------------------------------------
    def insert_buffer(self, net, loads=None, master=None, xy=None, name=None):
        """Put a buffer between a net's driver and some of its loads: the buffer's input joins the net, its output
        drives a new net with the loads (pins; default: all the net's sinks). master: a buffer of the library (default
        e.default_buffer(): the smallest one the design uses); xy: its lower-left (default: centered on the loads).
        Returns (cell, new net)."""
        self._open()
        d = self.d
        k = self.net(net)
        if k in self.clock_nets():
            raise ValueError('net %s is a clock net: clocks are ideal (pre-CTS), buffering them is not modeled'
                             % self._net_name(k))
        on = self._net_pins(k)
        if loads is None:
            loads = [p for p in on if self._pin_dir(p) != D_OUT]
        else:
            loads = [self.pin(p) for p in ([loads] if isinstance(loads, (int, np.integer, str)) else loads)]
        if not loads:
            raise ValueError('no loads to buffer on net %s' % self._net_name(k))
        bad = [p for p in loads if p not in on or self._pin_dir(p) == D_OUT]
        if bad:
            raise ValueError('pins %s are not loads of net %s' % (bad[:5], self._net_name(k)))
        master = master or self.default_buffer()
        if master is None:
            raise ValueError('the design\'s library has no buffer')
        s = _single_input(d, master)
        if s is None or s[2]:
            raise ValueError('%s is not a buffer (one input passed to one output)' % master)
        size = self._master_size(master)
        if xy is None:
            pp = d.pin_pos()[torch.as_tensor([p for p in loads if p < self.P0] or [0], device=d.device)]
            xy = (pp.mean(0) - size / 2).tolist()
        die = int(d.raw['c_die'][int(d.raw['p_cell'][loads[0]])]) if 'c_die' in d.raw and loads[0] < self.P0 else 0
        c = self.add_cell(master, xy, die=die, name=name or self._unique('attar_buf', self._names))
        n2 = self.add_net()
        self.connect(c, s[0], k)
        self.connect(c, s[1], n2)
        self.move(loads, n2)
        return c, n2

    def clone(self, cell, loads, xy=None, name=None):
        """A copy of a one-output cell that takes over some of its loads: the clone's inputs join the original's input
        nets, its output drives a new net with the loads (pins on the original's output net). xy: the clone's
        lower-left (default: the original's). Returns (clone, new net)."""
        self._open()
        d = self.d
        c0 = self._live(self.cell(cell))
        if c0 >= self.N0:
            raise ValueError('clone() takes a cell of before the edit')
        a = d.raw
        pins = self.cell_pins(c0)
        outs = [p for p in pins if self._pin_dir(p) == D_OUT]
        if len(outs) != 1:
            raise ValueError('%s has %d outputs; clone() takes one-output cells' % (d.names['cells'][c0], len(outs)))
        k = self._pin_net(outs[0])
        if k in self.clock_nets():
            raise ValueError('%s drives a clock net: clocks are ideal (pre-CTS)' % d.names['cells'][c0])
        loads = [self.pin(p) for p in ([loads] if isinstance(loads, (int, np.integer, str)) else loads)]
        on = self._net_pins(k)
        bad = [p for p in loads if p not in on or self._pin_dir(p) == D_OUT]
        if not loads or bad:
            raise ValueError('pins %s are not loads of the output net of %s' % (bad[:5] or '[]', d.names['cells'][c0]))
        m = int(a['c_master'][c0])
        if xy is None:
            xy = d.pos[c0].tolist()
        c = self.add_cell(d.names['masters'][m], xy, orient=int(d.orient[c0]), die=int(d.cell_die[c0]),
                          name=name or self._unique(d.names['cells'][c0] + '_clone', self._names))
        mp = d.names['master_pins'][m]; base = int(a['m_pin_ptr'][m])
        n2 = self.add_net()
        for p in pins:
            pn = mp[int(a['p_mpin'][p]) - base]
            self.connect(c, pn, n2 if p == outs[0] else self._pin_net(p))
        self.move(loads, n2)
        return c, n2

    # -- helpers --------------------------------------------------------------------------------------------------------
    def _open(self):
        if self._applied:
            raise RuntimeError('this edit was applied; start another with d.edit()')

    def _net_name(self, k):
        return self.d.names['nets'][k] if k < self.E0 else self.nets[k - self.E0]

    def _net_pins(self, k):
        """Pins on net k as the edit stands (indices of before, and new pins' handles)."""
        if k < self.E0:
            a, b = int(self._nptr[k]), int(self._nptr[k + 1])
            on = {p for p in range(a, b) if self.moves.get(p, k) == k and p not in self.dropped}
        else:
            on = set()
        return on | {p for p in self._joined.get(k, ()) if self._pin_net(p) == k and p not in self.dropped}

    def _master_size(self, master):
        from .timing.sizing import lib_cell
        d = self.d
        ms = d.names['masters']
        if master in ms:
            return torch.as_tensor(d.raw['m_size'][ms.index(master)].astype(np.float64) * d.scale, device=d.device,
                                   dtype=torch.float32)
        L, c = lib_cell(d, master)
        if L is None:
            raise KeyError('%s: not in the design\'s library' % master)
        q = [round(float(v) * d.dbu) * d.scale for v in L.a['lc_size'][c]]
        return torch.tensor(q, device=d.device, dtype=torch.float32)

    # -- apply ----------------------------------------------------------------------------------------------------------
    def apply(self):
        """Write the edits into the design (see the module doc). Returns self (pin_map set)."""
        self._open()
        self._applied = True
        d = self.d
        if not (self.cells or self.nets or self.pins or self.moves or self.removed):
            self.pin_map = np.arange(self.P0)
            return self
        a, nm, dev = d.raw, d.names, d.device
        T = lambda x, dt=None: torch.as_tensor(np.asarray(x), device=dev, dtype=dt)  # noqa: E731
        cat = lambda key, v: np.concatenate([a[key], np.asarray(v, a[key].dtype)])  # noqa: E731
        # cells: appended to the IR arrays and the design's tensors; placed where given
        if self.cells:
            m = np.array([c[0] for c in self.cells], np.int64)
            xy = np.array([c[1] for c in self.cells], np.float64)
            o = np.array([c[2] for c in self.cells], np.int64); die = np.array([c[3] for c in self.cells], np.int64)
            dxy = np.round(xy * d.dbu)
            a['c_master'] = cat('c_master', m); a['c_xy'] = cat('c_xy', dxy); a['c_orient'] = cat('c_orient', o)
            a['c_flags'] = cat('c_flags', np.full(len(m), PLACED))
            if 'c_die' in a:
                a['c_die'] = cat('c_die', die)
            for key in [k for k in a if k == 'ref_xy' or (k.startswith('snap_') and k.endswith('_xy'))]:
                a[key] = cat(key, dxy); a[key[:-3] + '_orient'] = cat(key[:-3] + '_orient', o)
            msize = a['m_size'][m].astype(np.float64) * d.scale
            d.flags = torch.cat([d.flags, T(np.full(len(m), PLACED), d.flags.dtype)])
            d.orient = torch.cat([d.orient, T(o, d.orient.dtype)])
            d._pos = torch.cat([d._pos, T(xy, d._pos.dtype)])
            d.size_n = torch.cat([d.size_n, T(msize, d.size_n.dtype)])
            d.cell_die = torch.cat([d.cell_die, T(die, d.cell_die.dtype)])
            d.placed = torch.cat([d.placed, torch.ones(len(m), dtype=torch.bool, device=dev)])
            d._refresh_movable()
            for k, snap in list(d.snapshots.items()):
                if isinstance(snap[0], np.ndarray):
                    d.snapshots[k] = (np.concatenate([snap[0], dxy.astype(snap[0].dtype)]),
                                      np.concatenate([snap[1], o.astype(snap[1].dtype)]))
                else:
                    d.snapshots[k] = (torch.cat([snap[0], T(xy, snap[0].dtype)]),
                                      torch.cat([snap[1], T(o, snap[1].dtype)]), snap[2])
            nm['cells'].extend(c[4] for c in self.cells)
            if getattr(d, '_cell_id', None) is not None:
                d._cell_id.update({c[4]: self.N0 + i for i, c in enumerate(self.cells)})
        # nets: appended (weight 1, no flags)
        if self.nets:
            a['n_weight'] = cat('n_weight', np.ones(len(self.nets))); a['n_flags'] = cat('n_flags', np.zeros(len(self.nets)))
            d.n_weight = torch.cat([d.n_weight, torch.ones(len(self.nets), dtype=d.n_weight.dtype, device=dev)])
            d.n_flags = torch.cat([d.n_flags, torch.zeros(len(self.nets), dtype=d.n_flags.dtype, device=dev)])
            nm['nets'].extend(self.nets)
        E = self.E0 + len(self.nets)
        # removed cells: tombstones (ASIDE | REMOVED), not placed, no pins (dropped below)
        if self.removed:
            rc = np.array(sorted(self.removed), np.int64)
            a['c_flags'] = a['c_flags'].copy(); a['c_flags'][rc] |= ASIDE | REMOVED
            if 'tie_cell' in a:                          # their pins on supply nets go too
                keep = ~np.isin(a['tie_cell'], rc)
                for k in ('tie_cell', 'tie_mpin', 'tie_val'):
                    a[k] = a[k][keep]
            d.flags[T(rc)] |= ASIDE | REMOVED
            d.placed[T(rc)] = False
            d._refresh_movable()
        # pins: new ones appended, moved ones re-netted, dropped ones gone, all in net order by one stable sort
        P1 = self.P0 + len(self.pins)
        net = np.concatenate([self._pnet, np.array([q[3] for q in self.pins], np.int64)])
        for p, k in self.moves.items():
            net[p] = k
        keep = np.ones(P1, bool)
        if self.dropped:
            keep[np.fromiter(self.dropped, np.int64, len(self.dropped))] = False
        idx = np.nonzero(keep)[0]
        order = idx[np.argsort(net[idx], kind='stable')]
        self.pin_map = np.full(P1, -1, np.int64); self.pin_map[order] = np.arange(len(order))
        pc = np.array([q[0] for q in self.pins], np.int64); pm = np.array([q[1] for q in self.pins], np.int64)
        pd = np.array([q[2] for q in self.pins], np.int64)
        a['p_cell'] = cat('p_cell', pc)[order]; a['p_mpin'] = cat('p_mpin', pm)[order]
        if 'p_dir' in a:
            a['p_dir'] = cat('p_dir', pd)[order]
        deg_old = np.concatenate([np.diff(self._nptr), np.zeros(len(self.nets), np.int64)])
        deg_new = np.bincount(net[order], minlength=E)
        a['n_ptr'] = np.concatenate([[0], np.cumsum(deg_new)]).astype(a['n_ptr'].dtype)
        emptied = np.nonzero((deg_new == 0) & (deg_old > 0))[0]            # nets the edit emptied: N_REMOVED
        if len(emptied):
            a['n_flags'] = a['n_flags'].copy(); a['n_flags'][emptied] |= N_REMOVED
            d.n_flags[T(emptied)] |= N_REMOVED
        # the new pins' centers relative to their master's lower-left (as Design.__init__)
        if len(self.pins):
            if 'pin_offsets' in d.facts:
                box = a['m_pin_box'][pm].astype(np.float64) * d.scale
                ctr = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], -1)
            else:
                ctr = (a['m_size'][a['c_master'][pc]].astype(np.float64) * d.scale) / 2
        else:
            ctr = np.zeros((0, 2))
        ot = T(order)
        d.p_cell = torch.cat([d.p_cell, T(pc, d.p_cell.dtype)])[ot]
        d.pin_dir = torch.cat([d.pin_dir, T(pd, d.pin_dir.dtype)])[ot]
        d._p_ctr_n = torch.cat([d._p_ctr_n, T(ctr, d._p_ctr_n.dtype)])[ot]
        d._p_term = torch.cat([d._p_term, torch.zeros(len(self.pins), dtype=torch.bool, device=dev)])[ot]
        d.n_ptr = T(a['n_ptr'].astype(np.int64))
        deg = d.n_ptr[1:] - d.n_ptr[:-1]
        d.p_net = torch.repeat_interleave(torch.arange(E, device=dev), deg)
        d.n_mask = deg >= 2
        d.update_orient()
        # the manifest's counts follow the netlist
        c = d.manifest.setdefault('counts', {})
        for key, n in (('cells', len(self.cells) - len(self.removed)), ('movable', len(self.cells) - len(self.removed)),
                       ('nets', len(self.nets) - len(emptied)), ('pins', len(self.pins) - len(self.dropped))):
            if key in c:
                c[key] += n
        if 'p_dir' in a and 'pin_direction' in d.manifest:
            d.manifest['pin_direction'] = pin_dir_counts(a['p_dir'], a['n_ptr'])
        d.manifest['netlist_edits'] = d.manifest.get('netlist_edits', 0) + 1
        d._netlist_changed('the netlist was edited')
        return self


def check(d):
    """Problems of the netlist as it stands (a list of strings; empty: consistent): the CSR over the pins, index
    ranges, every cell pin on a pin of its cell's master, names and tensors in step with the arrays."""
    a = d.raw
    out = []
    n_ptr = a['n_ptr'].astype(np.int64)
    if n_ptr[0] != 0 or n_ptr[-1] != len(a['p_cell']) or (np.diff(n_ptr) < 0).any():
        out.append('n_ptr is not a CSR over the pins')
    N = len(a['c_master'])
    if (a['p_cell'] < 0).any() or (a['p_cell'] >= N).any():
        out.append('a pin on a cell out of range')
    term = (a['c_flags'][a['p_cell']] & TERMINAL) != 0
    cm = a['c_master'][a['p_cell']]
    k = a['p_mpin'].astype(np.int64)
    ok = term | ((k >= a['m_pin_ptr'][np.maximum(cm, 0)]) & (k < a['m_pin_ptr'][np.maximum(cm, 0) + 1]))
    if not ok.all():
        out.append('%d pins not on a pin of their cell\'s master' % int((~ok).sum()))
    for t in ('p_cell', 'p_mpin', 'p_dir'):
        if t in a and len(a[t]) != len(a['p_cell']):
            out.append('%s has %d entries, not %d' % (t, len(a[t]), len(a['p_cell'])))
    if len(d.names['cells']) != N or len(d.names['nets']) != len(n_ptr) - 1:
        out.append('names out of step with the arrays')
    if len(d.pos) != N or len(d.p_net) != len(a['p_cell']):
        out.append('tensors out of step with the arrays')
    return out


def debuffer(d, buffers=True, inverters=True):
    """Remove the netlist's buffers and double inversions, as OpenROAD's remove_buffers does for buffers (rsz): a
    buffer's output net joins its input net and the buffer goes; an inverter whose input an inverter drives has its
    output net join that inverter's input net and goes, and an inverter left without loads goes too (repeated: an even
    chain of inverters goes whole, an odd one keeps one). The joined net keeps the name of the one with an IO pin, else
    of the one with the remaining driver. Left alone: cells on clock nets (clocks are ideal), fixed cells, nets with
    several or no drivers, and joins that would put two IO pins on one net (a feedthrough).
    Returns counts: {'buffers', 'inverters', 'nets_joined', 'kept_ports', 'kept_clock'}."""
    e = Edit(d)
    a = d.raw
    E = e.E0
    pc = a['p_cell'].astype(np.int64); pdir = a['p_dir'].astype(np.int64); pnet = e._pnet; nptr = e._nptr
    flags = a['c_flags']; term_pin = (flags[pc] & TERMINAL) != 0
    # per master: (inverting, input pin, output pin) of buffers / inverters
    kind = {}
    for m, name in enumerate(d.names['masters']):
        s = _single_input(d, name)
        if s is not None and (inverters if s[2] else buffers):
            mp = d.names['master_pins'][m]
            if s[0] in mp and s[1] in mp:
                base = int(a['m_pin_ptr'][m])
                kind[m] = (s[2], base + mp.index(s[0]), base + mp.index(s[1]))
    n = {'buffers': 0, 'inverters': 0, 'nets_joined': 0, 'kept_ports': 0, 'kept_clock': 0}
    if not kind:
        e.apply()
        return n
    clk = e.clock_nets()
    is_drv = pdir == D_OUT
    ndrv = np.bincount(pnet[is_drv], minlength=E)
    drv_cell = np.full(E, -1, np.int64); drv_cell[pnet[is_drv]] = np.where(term_pin[is_drv], -2, pc[is_drv])
    sink = ~is_drv
    loads = np.bincount(pnet[sink], minlength=E).astype(np.int64)                # live sinks per group (root)
    ports = np.bincount(pnet[term_pin], minlength=E).astype(np.int64)            # IO pins per group (root)
    parent = np.arange(E)

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    def join(keep, other):
        """Group `other` into group `keep` (roots); keep's driver stays."""
        parent[other] = keep
        loads[keep] += loads[other]; ports[keep] += ports[other]

    # candidate cells: (cell, inverting, input net, output net)
    cm = a['c_master'].astype(np.int64)
    cand = {}
    for c in np.nonzero(np.isin(cm, list(kind)) & ((flags & (FIXED | TERMINAL | REMOVED)) == 0))[0].tolist():
        inv, ki, ko = kind[int(cm[c])]
        ps = e.cell_pins(c)
        pi = [p for p in ps if a['p_mpin'][p] == ki]; po = [p for p in ps if a['p_mpin'][p] == ko]
        if len(pi) != 1 or len(po) != 1:
            continue
        ni, no = int(pnet[pi[0]]), int(pnet[po[0]])
        if ni in clk or no in clk:
            n['kept_clock'] += 1
            continue
        if ni == no or ndrv[ni] != 1 or ndrv[no] != 1 or drv_cell[no] != c:
            continue
        cand[c] = (inv, ni, no)
    removed = set()

    def remove(c):
        removed.add(c)
        loads[find(cand[c][1])] -= 1                                            # its input was a load

    # buffers: the output group joins the input group
    for c, (inv, ni, no) in sorted(cand.items()):
        if inv:
            continue
        gi, go = find(ni), find(no)
        if ports[gi] and ports[go]:
            n['kept_ports'] += 1
            continue
        remove(c); join(gi, go); n['buffers'] += 1
    # inverters: a double inversion goes; an inverter left without loads goes; until nothing changes
    invs = sorted(c for c, v in cand.items() if v[0])
    drives = {}                                                                 # group -> the inverter driving it
    for c in invs:
        drives[find(cand[c][2])] = c
    kept = set()
    changed = True
    while changed:
        changed = False
        for c2 in invs:
            if c2 in removed:
                continue
            _, ni2, no2 = cand[c2]
            g1 = find(ni2)
            c1 = drives.get(g1)
            if c1 is None or c1 in removed or c1 == c2:
                continue
            base, g2 = find(cand[c1][1]), find(no2)
            if base == g2:
                continue
            if ports[base] and ports[g2]:
                kept.add(c2)
                continue
            remove(c2); join(base, g2); n['inverters'] += 1; changed = True
            drives.pop(g2, None)
            if loads[find(g1)] == 0:                                            # c1 drives nothing any more
                remove(c1); n['inverters'] += 1
                drives.pop(find(g1), None)
    n['kept_ports'] += len(kept - removed)
    # the edit: removed cells go; every other net of a group moves onto the group's surviving net
    for c in sorted(removed):
        e.remove_cell(c)
    groups = {}
    for k in {k for c in cand for k in cand[c][1:]}:
        groups.setdefault(find(k), []).append(k)
    for _r, members in sorted(groups.items()):
        members = sorted(members)
        if len(members) < 2:
            continue
        live = [k for k in members if any(p not in e.dropped for p in range(nptr[k], nptr[k + 1]))]
        if not live:
            continue
        with_port = [k for k in live if (term_pin[nptr[k]:nptr[k + 1]]).any()]
        with_drv = [k for k in live if any(is_drv[p] and p not in e.dropped for p in range(nptr[k], nptr[k + 1]))]
        keep = (with_port or with_drv or sorted(live))[0]
        for k in members:
            if k == keep:
                continue
            ps = [p for p in range(nptr[k], nptr[k + 1]) if p not in e.dropped]
            if ps:
                e.move(ps, keep); n['nets_joined'] += 1
    e.apply()
    return n
