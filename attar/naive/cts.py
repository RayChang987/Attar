"""Naive clock tree synthesis: buffered clock trees for a placed design, balanced on Attar's propagated-clock timer.

For every net of an ideal clock's network (attar.timing.sta: from the clock source through nets and buffers or gates
up to the register clock pins) whose loads are many or far apart:
  0. buffers    the clock buffers as OpenROAD infers them (Resizer::inferClockBufferList: *CLKBUF*, else *BUF*, without
                dont_use; ASAP7: BUFx*, not the HB* hold buffers); the max wire length of each (Resizer's
                findMaxWireLength: where splitting a wire with a second buffer is faster) and the repeater (the
                least delay per length at its own max wire length).
  1. clusters   the net's register loads and its loads that continue the clock network (clock gates) get trees of
                their own. Loads are cut into groups of at most leaf_size whose half-perimeter is at most max_diameter
                (TritonCTS's default sink clustering diameter, never above the max wire length) by recursive median
                bisection along the longer side (deterministic); each group gets a buffer at its loads' centroid
                (moved out of fixed macros). The buffers are clustered the same way, at most `fanout` per group, level by
                level, until one root buffer is left; the net's driver drives it.
  2. sizes      each buffer is the smallest clock buffer whose output transition at its estimated load (the pins'
                capacitance plus clock wire RC on the group's half perimeter) stays under target_slew.
  3. netlist    the buffers and their nets (USE CLOCK) go into the netlist (attar.netlist.Edit); tree edges longer than
                the max wire length get repeaters, as many on every edge into a level as its longest edge needs (each
                branch keeps the same number of stages). The new buffers alone are legalized.
  4. timing     the clocks become propagated (Design.propagate_clocks): the timer times the trees.
  5. pad        register trees of a clock that the timer finds earlier than its latest by more than half a stage
                (registers on the clock's own net next to gated ones) get a chain of the smallest clock buffer at their
                root, its length from the measured delay per stage (_pad, as TritonCTS's latency balancing).
  6. balance    one level per round from the root down: each register-tree buffer resized so that its subtree's clock
                pins meet the clock's common latency at that level (a change predicted from the measured delays and
                the Liberty tables, including its driver's delay and slew change; _balance); the round with the
                smallest spread is kept.
Returns statistics: buffers per level, repeaters, pads, latency, skew (the spread of the register clock pins'
arrival), slew limit violations, seconds. Export with d.export (the nets carry USE CLOCK); a flow then times it with
set_propagated_clock.
"""
import math
import time

import numpy as np
import torch

from ..io.distill import N_CLOCK


def _bisect(xy, idx, cap, max_span=float('inf')):
    """Balanced groups of at most cap points whose bounding box half-perimeter is at most max_span: recursive median
    split along the longer side. idx (n,) -> [arrays]."""
    out, stack = [], [idx]
    while stack:
        g = stack.pop()
        p = xy[g]
        span = p.max(0) - p.min(0)
        if len(g) <= cap and (len(g) < 2 or float(span.sum()) <= max_span):
            out.append(g)
            continue
        ax = int(np.argmax(span))
        # split so that both halves fill whole groups where possible (balanced leaf counts)
        n_groups = max(2, math.ceil(len(g) / cap))
        left = (n_groups // 2) * len(g) // n_groups
        o = np.argsort(p[:, ax], kind='stable')
        stack.append(g[o[left:]]); stack.append(g[o[:left]])
    return out


def _buffer_table(d, masters):
    """Per buffer master: (input pin, output pin, input capacitance, max capacitance, (slew, load) -> worst output
    transition, (slews, loads) -> rise delays, (slews, loads) -> worst output transitions) from the library chain's
    Liberty tables."""
    from ..netlist import _single_input
    from ..timing.sizing import lib_cell
    from ..timing.sta import nldm
    out = {}
    for m in masters:
        s = _single_input(d, m)
        if s is None or s[2]:
            continue
        L, c = lib_cell(d, m)
        A = L.a
        pid = L.pid[c]
        if s[0] not in pid or s[1] not in pid:
            continue
        kin, kout = pid[s[0]], pid[s[1]]
        arcs = [int(a) for a in L.arcs_of[c] if A['la_from'][a] == kin and A['la_to'][a] == kout]
        if not arcs:
            continue
        a = arcs[0]
        cin = float(A['lp_cap'][kin].max())
        cmax = float(A['lp_maxcap'][kout]) or float('inf')
        ix1, ix2, val, n = (torch.as_tensor(A[k][a]) for k in ('la_ix1', 'la_ix2', 'la_val', 'la_n'))

        def look(sl, load, slots, ix1=ix1, ix2=ix2, val=val, n=n):
            x = torch.as_tensor(sl, dtype=torch.float32).reshape(-1)
            y = torch.as_tensor(load, dtype=torch.float32).reshape(-1)
            y = y.expand_as(x) if len(y) == 1 else y; x = x.expand_as(y) if len(x) == 1 else x
            v = [nldm(ix1[k:k + 1].expand(len(x), -1), ix2[k:k + 1].expand(len(x), -1), val[k:k + 1].expand(len(x), -1, -1),
                      n[k:k + 1].long().expand(len(x), -1), x, y) for k in slots if int(n[k, 0]) > 0]
            return torch.stack(v).amax(0) if v else torch.zeros(len(x))
        slew = lambda sl, load, look=look: float(look([sl], [load], (2, 3))[0])  # noqa: E731
        delay = lambda sl, load, look=look: look(sl, load, (0,))                 # noqa: E731  (rise: clock edge)
        slews = lambda sl, load, look=look: look(sl, load, (2, 3))               # noqa: E731  (vectors)
        out[m] = (s[0], s[1], cin, cmax, slew, delay, slews)
    return out


def _design_buffers(d):
    """The library's buffers from the Liberty file of the buffer the design uses most (its threshold-voltage flavor),
    smallest first; all the library's buffers when the design has none."""
    from ..timing.sizing import lib_cell
    a, names = d.raw, d.names
    bufs = d.buffers()
    used = {names['masters'][int(m)]: int(k) for m, k in zip(*np.unique(a['c_master'][a['c_master'] >= 0],
                                                                        return_counts=True))}
    have = [(used[b], b) for b in bufs if b in used]
    if not have:
        return bufs
    L0, c0 = lib_cell(d, max(have)[1])
    out = []
    for b in bufs:
        L, c = lib_cell(d, b)
        if L is L0 and L.a['lc_file'][c] == L0.a['lc_file'][c0]:
            out.append(b)
    return out or bufs


def _clock_buffers(d):
    """The clock buffers, as OpenROAD's Resizer::inferClockBufferList with no user setting: of the design's buffers
    (its threshold-voltage flavor, no dont_use), those named *CLKBUF*, else those named *BUF* (case-insensitive),
    else all. (Its first two rules, Liberty is_clock_cell and a LEF USE CLOCK input, are not in Attar's library
    data; neither ASAP7 nor Nangate45 sets them.) ASAP7: the BUFx* cells, not the HB* hold buffers."""
    bufs = _design_buffers(d)
    for pat in ('CLKBUF', 'BUF'):
        sel = [b for b in bufs if pat in b.upper()]
        if sel:
            return sel
    return bufs


def _wire_delay(table, m, length, cin, r, c, in_slew):
    """Buffer m driving a wire of length (um, clock RC r kohm/um, c fF/um) into a load cin: the Liberty delay at the
    total capacitance plus the wire's Elmore delay (ps)."""
    return float(table[m][5](in_slew, c * length + cin)[0]) + r * length * (c * length / 2 + cin)


def _max_wire_length(table, m, r, c, in_slew):
    """The wire length (um) beyond which splitting it in half with a second m is faster, as OpenROAD's
    Resizer::findMaxWireLength (splitWireDelayDiff = delay(L) - 2 delay(L / 2), binary search to 1%), with the
    delays of _wire_delay (input slew: the target slew, as Resizer::cellWireDelay)."""
    cin = table[m][2]
    diff = lambda L: _wire_delay(table, m, L, cin, r, c, in_slew) - 2 * _wire_delay(table, m, L / 2, cin, r, c, in_slew)  # noqa: E731
    if r <= 0 or c <= 0:
        return float('inf')
    lo, hi = 0.0, 10.0
    while diff(hi) < 0:
        lo, hi = hi, hi * 2
        if hi > 1e6:
            return float('inf')
    while hi - lo > 0.01 * hi:
        mid = (lo + hi) / 2
        if diff(mid) < 0:
            lo = mid
        else:
            hi = mid
    return lo


def _free_spot(d):
    """A function xy -> xy that moves a point inside a fixed macro (or outside the rows' extent) to the nearest point
    just outside it (a buffer placed there legalizes nearby; the legalizer may otherwise move it far)."""
    from ..io.distill import MACRO
    from ..rules.rails import base_rows
    _hb, ys, rx0, rx1, _sw, _ro = base_rows(d)
    lo = np.array([float(rx0.min()), float(ys.min())]); hi = np.array([float(rx1.max()), float(ys.max())])
    m = (~d.movable & ((d.flags & MACRO) != 0)).cpu().numpy()
    box = np.concatenate([d.pos.detach().cpu().numpy()[m], (d.pos + d.size).detach().cpu().numpy()[m]], 1).astype(np.float64)

    def f(xy):
        p = np.clip(np.asarray(xy, np.float64), lo, hi)
        for _ in range(8):
            inside = np.nonzero((p[0] > box[:, 0]) & (p[0] < box[:, 2]) & (p[1] > box[:, 1]) & (p[1] < box[:, 3]))[0]
            if not len(inside):
                break
            b = box[inside[0]]
            cand = [(b[0] - 1e-3, p[1]), (b[2] + 1e-3, p[1]), (p[0], b[1] - 1e-3), (p[0], b[3] + 1e-3)]
            cand = [np.clip(np.array(c), lo, hi) for c in cand]
            p = min(cand, key=lambda c: abs(c[0] - p[0]) + abs(c[1] - p[1]) + 1e9 * float(
                ((c[0] > box[:, 0]) & (c[0] < box[:, 2]) & (c[1] > box[:, 1]) & (c[1] < box[:, 3])).any()))
        return p
    return f


def _pick(table, order, load, target_slew, in_slew=20.0):
    """The smallest buffer (order: small first) meeting target_slew and its max capacitance at load, else the largest."""
    for m in order:
        cmax, slew = table[m][3], table[m][4]
        if load <= cmax and slew(in_slew, load) <= target_slew:
            return m
    return order[-1]


def cts(d, leaf_size=20, fanout=8, target_slew=None, buffers=None, balance_rounds=None, tol=2.0, min_sinks=2,
        max_diameter=50.0, verbose=True):
    """Build buffered clock trees (see the module doc) in place. buffers: the buffer masters to use (default: the
    clock buffers, _clock_buffers); target_slew: ps (default: a tenth of the library's default max transition, else 30);
    max_diameter: um, the largest half-perimeter of a leaf's sinks (TritonCTS's default sink clustering diameter; never
    above the max wire length). Returns statistics."""
    d.require('netlist_edit')
    t0 = time.time()
    res0 = d.timing()
    T = res0.timer
    a, names = d.raw, d.names
    pdir = a['p_dir'].astype(np.int64)
    n_ptr = a['n_ptr'].astype(np.int64)
    rc = T.info.get('rc') or {}
    r_clk, c_clk = (rc.get('clock') or rc.get('signal') or [0.0, 0.0])
    if target_slew is None:                         # a tenth of the library's default max transition (ASAP7: 32 ps)
        lims = [float(x) for x in np.unique(d.library()[-1].a['lc_dlim'][:, 0]) if np.isfinite(x)] \
            if 'lc_dlim' in d.library()[-1].a else []
        target_slew = 0.1 * min(lims) if lims else 30.0
    # the clock nets to build: driven by an ideal clock network pin, with loads
    drv = T.net_driver
    nets = [k for k in range(len(n_ptr) - 1) if drv[k] >= 0 and T.iclk[drv[k]]
            and n_ptr[k + 1] - n_ptr[k] - 1 >= min_sinks]
    stats = {'nets': 0, 'buffers': 0, 'levels': []}
    order = list(buffers) if buffers is not None else _clock_buffers(d)
    table = _buffer_table(d, order)
    order = [m for m in order if m in table]
    if not order:
        raise ValueError('no usable buffer in the library')
    # repeaters for long tree edges (the clock source to the root, a parent to a far child): the buffer with the
    # least delay per length at its own max wire length (OpenROAD's repair_clock_nets buffers these wires too)
    lmax = {m: _max_wire_length(table, m, r_clk, c_clk, target_slew) for m in order}
    rep = min(order, key=lambda m: _wire_delay(table, m, lmax[m], table[m][2], r_clk, c_clk, target_slew) / lmax[m]
              if math.isfinite(lmax[m]) and lmax[m] > 0 else float('inf'))
    rep_len = lmax[rep]
    stats['repeater'] = {'master': rep, 'spacing_um': rep_len}
    pin_xy = d.pin_pos().detach().cpu().numpy().astype(np.float64)
    free = _free_spot(d)
    cap = T.pin_cap.max(1).values.cpu().numpy().astype(np.float64)
    plan = []                     # (net, [(level, buffer xy, master, children: pins or nodes, load)], root, kind)

    def build(loads, size):
        nodes = []                                    # (level, xy, master, children (pins or node ids), load)
        groups = _bisect(pin_xy, loads, size, min(max_diameter, rep_len))
        cur = []
        for g in groups:
            xy = pin_xy[g].mean(0)
            hp = float((pin_xy[g].max(0) - pin_xy[g].min(0)).sum())
            load = float(cap[g].sum()) + c_clk * hp
            nodes.append((0, xy, _pick(table, order, load, target_slew), ('pins', g), load))
            cur.append(len(nodes) - 1)
        level = 1
        while len(cur) > 1:
            cxy = np.stack([nodes[i][1] for i in cur])
            grp = _bisect(cxy, np.arange(len(cur)), fanout)
            nxt = []
            for g in grp:
                kids = [cur[i] for i in g]
                xy = cxy[g].mean(0)
                hp = float((cxy[g].max(0) - cxy[g].min(0)).sum())
                load = float(sum(table[nodes[i][2]][2] for i in kids)) + c_clk * hp
                nodes.append((level, xy, _pick(table, order, load, target_slew), ('nodes', kids), load))
                nxt.append(len(nodes) - 1)
            cur = nxt
            level += 1
        return [(lv, free(xy), m, kids, ld) for lv, xy, m, kids, ld in nodes], cur[0], level
    # a clock net's loads that continue the clock network (clock gates, clock logic) get a tree of their own: the
    # registers' tree can then be delayed (_pad) without delaying the gated clocks behind them
    ccell = np.zeros(len(a['c_master']), bool)
    ccell[T.p_cell_np[(pdir == 1) & (T.clk_of >= 0)]] = True
    for k in nets:
        pins = np.arange(n_ptr[k], n_ptr[k + 1])
        loads = pins[(pdir[pins] != 1) & (pins != drv[k])]
        thru = ccell[T.p_cell_np[loads]]
        for part, kind, size in ((loads[~thru], 'seq', leaf_size), (loads[thru], 'thru', leaf_size)):
            if len(part) < min_sinks:
                continue
            nodes, root, level = build(part, size)
            plan.append((k, nodes, root, kind))
            stats['nets'] += 1
            stats['buffers'] += len(nodes)
            stats['levels'].append(level)
    if not plan:
        return dict(stats, seconds=time.time() - t0, note='no clock net to build')
    # netlist edit: a buffer per node; the net's driver keeps the original net, which now drives the root buffer
    new_cells, new_nets, leaf_cells = [], [], []
    tree = []                                         # (cell, level, child cells, clock, kind) per tree buffer
    domains = []                                      # (root cell, leaf cells, clock, net) per register tree
    net_names = set(names['nets'])
    stats['repeaters'] = 0

    def n_rep(src_xy, dst_xy):
        L = float(abs(dst_xy[0] - src_xy[0]) + abs(dst_xy[1] - src_xy[1]))
        return 0 if not math.isfinite(rep_len) or L <= rep_len else math.ceil(L / rep_len) - 1

    def link(e, src_xy, dst_xy, net, n):
        """The net a buffer at dst_xy takes its input from, driven from src_xy on net: net itself, or past n
        repeaters evenly spaced on the Manhattan path (x first)."""
        dx, dy = float(dst_xy[0] - src_xy[0]), float(dst_xy[1] - src_xy[1])
        L = abs(dx) + abs(dy)
        if n <= 0:
            return net
        for j in range(1, n + 1):
            t = L * j / (n + 1)
            xy = free((src_xy[0] + math.copysign(t, dx), src_xy[1]) if t <= abs(dx)
                      else (dst_xy[0], src_xy[1] + math.copysign(t - abs(dx), dy)))
            c = e.add_cell(rep, (float(xy[0]), float(xy[1])), name=e._unique('attar_cts_rep_', e._names))
            o = e.add_net(name=e._unique('attar_cts_net_', e._net_names or net_names))
            e.connect(c, table[rep][0], net)
            e.connect(c, table[rep][1], o)
            new_cells.append(c); new_nets.append(o)
            stats['repeaters'] += 1
            net = o
        return net
    with d.edit() as e:
        for k, nodes, root, kind in plan:
            cid, nid = {}, {}
            for i, (lev, xy, m, _kids, _) in enumerate(nodes):
                c = e.add_cell(m, (float(xy[0]), float(xy[1])), name=e._unique('attar_cts_%d_' % lev, e._names))
                n = e.add_net(name=e._unique('attar_cts_net_', e._net_names or set(names['nets'])))
                e.connect(c, table[m][1], n)
                cid[i], nid[i] = c, n
                new_cells.append(c); new_nets.append(n)
                if lev == 0:
                    leaf_cells.append(c)
            # level-synchronous repeaters: every edge into a level gets as many as its longest edge needs, so that
            # every branch keeps the same number of stages (the balance resizes buffers, it cannot remove a stage)
            per_lev = {}
            for i, (lev, _xy, _m, kids, _) in enumerate(nodes):
                if kids[0] == 'nodes':
                    for j in kids[1]:
                        per_lev[lev - 1] = max(per_lev.get(lev - 1, 0), n_rep(nodes[i][1], nodes[j][1]))
            for i, (_lev, _xy, _m, kids, _) in enumerate(nodes):
                if kids[0] == 'pins':
                    e.move([int(p) for p in kids[1]], nid[i])
                else:
                    for j in kids[1]:
                        e.connect(cid[j], table[nodes[j][2]][0],
                                  link(e, nodes[i][1], nodes[j][1], nid[i], per_lev.get(nodes[j][0], 0)))
            e.connect(cid[root], table[nodes[root][2]][0],
                      link(e, pin_xy[drv[k]], nodes[root][1], k, n_rep(pin_xy[drv[k]], nodes[root][1])))
            ck = int(T.clk_of[drv[k]])
            for i, (lev, _xy, _m, kids, _) in enumerate(nodes):
                tree.append((cid[i], lev, [cid[j] for j in kids[1]] if kids[0] == 'nodes' else [], ck, kind))
            if kind == 'seq':
                domains.append((cid[root], [cid[i] for i, nd in enumerate(nodes) if nd[0] == 0], ck, k))
    # the new nets are clock nets (USE CLOCK: OpenROAD's estimate_parasitics gives the non-leaf ones clock RC)
    na = names.setdefault('net_attrs', {'default': '', 'except': {}})
    for n in new_nets:
        a['n_flags'][n] |= N_CLOCK
        na['except'][str(n)] = ' + USE CLOCK'
    d.n_flags[torch.as_tensor(new_nets, device=d.device)] |= N_CLOCK
    # buffer centers -> lower-left, then legalize the new buffers only
    nc = torch.as_tensor(new_cells, device=d.device)
    pos = d.pos.clone()
    pos[nc] = pos[nc] - d.size[nc] / 2
    d.pos = pos
    stats['legalize'] = _legalize_new(d, nc, verbose=False)
    d.propagate_clocks()
    stats['pads'] = _pad(d, domains, table, order[0], target_slew, stats)
    # balance (_balance): one level per round from the root down, each tree buffer resized so that its subtree's
    # clock pins meet the level's common latency; re-time after each level; keep the round with the smallest spread
    levels = sorted({lv for _, lv, _, _, kind in tree if kind == 'seq'}, reverse=True)
    rounds = len(levels) if balance_rounds is None else balance_rounds
    tcells = np.asarray([c for c, _, _, _, _ in tree], np.int64)
    hist, best, li = [], None, 0
    for rnd in range(rounds + 1):
        res = d.timing(crpr=False)                          # clock latencies only
        lat, skew, _ = _latencies(d, res, [])
        hist.append({'round': rnd, 'latency_max': lat[1], 'latency_min': lat[0], 'skew': skew})
        if best is None or skew < best[0]:
            best = (skew, [names['masters'][int(m)] for m in d.raw['c_master'][tcells]])
        if rnd == rounds or skew <= tol:
            break
        cells = []
        while not cells and li < len(levels) * (1 + rounds):
            cells, masters = _balance(d, res, tree, levels[li % len(levels)], table, order, target_slew)
            li += 1
        if not cells:
            break
        d.resize(cells, masters, check=False)
        d.propagate_clocks()
    now = [names['masters'][int(m)] for m in d.raw['c_master'][tcells]]
    if best is not None and now != best[1]:                  # keep the round with the smallest spread
        d.resize(list(tcells), best[1], check=False)
        d.propagate_clocks()
        hist.append({'round': 'best', 'skew': best[0]})
    res = d.timing()
    drv_ = res.drv()
    lat, skew, _ = _latencies(d, res, [])
    stats.update({'balance': hist, 'latency': lat[1], 'latency_min': lat[0], 'skew': skew,
                  'wns': res.wns, 'tns': res.tns, 'whs': res.whs, 'slew_violations': drv_.slew_violations,
                  'cap_violations': drv_.cap_violations, 'seconds': time.time() - t0})
    if verbose:
        print('cts: %d nets, %d buffers (levels %s), latency %.1f ps, skew %.1f ps, WNS %.1f TNS %.1f, %.1fs'
              % (stats['nets'], stats['buffers'], sorted(set(stats['levels'])), stats['latency'], stats['skew'],
                 res.wns, res.tns, stats['seconds']))
    return stats


def _balance(d, res, tree, level, table, order, target_slew):
    """New sizes for the tree buffers of one level. Per buffer, the clock arrival range (min, max) of the pins its
    subtree reaches (the timer's, rising, latest over tags) and its midpoint; a size s moves the whole subtree by
    delay(s) - delay(current), the Liberty rise delays at the buffer's input slew and output net's load as timed (a
    change predicted from the measured delay, not the table's absolute value). Sizes whose output transition at that
    load meets target_slew and their max capacitance only (the current one always). Per clock, the target is the
    common midpoint the level's buffers can reach (the current median, clipped into the intersection of their
    reachable ranges; the middle of the gap when they do not intersect); each buffer takes the size nearest it.
    Register trees only (not the branches into clock gates), grouped by clock across its nets.
    Returns (cells, masters) to resize."""
    T = res.timer
    a, names = d.raw, d.names
    cells = np.asarray([c for c, _, _, _, _ in tree], np.int64)
    n_ptr = a['n_ptr'].astype(np.int64)
    # each tree buffer's input and output pin
    pins = np.nonzero(np.isin(a['p_cell'], cells))[0]
    pdir = a['p_dir'][pins]
    pin_in = dict(zip(a['p_cell'][pins[pdir != 1]].tolist(), pins[pdir != 1].tolist()))
    pin_out = dict(zip(a['p_cell'][pins[pdir == 1]].tolist(), pins[pdir == 1].tolist()))
    arr = T.amax[:, 0].amax(-1)                                         # (P,) rise, latest over tags
    pnet = d.p_net.cpu().numpy()
    rng = {}
    for c, _lv, kids, _, _ in sorted(tree, key=lambda x: x[1]):         # leaves first
        if kids:
            lo = min(rng[k][0] for k in kids if k in rng) if any(k in rng for k in kids) else np.nan
            hi = max(rng[k][1] for k in kids if k in rng) if any(k in rng for k in kids) else np.nan
        else:
            n = int(pnet[pin_out[c]])
            ld = np.arange(n_ptr[n], n_ptr[n + 1])
            ld = ld[ld != pin_out[c]]
            v = arr[torch.as_tensor(ld, device=arr.device)]
            v = v[torch.isfinite(v)]
            lo, hi = (float(v.min()), float(v.max())) if len(v) else (np.nan, np.nan)
        rng[c] = (lo, hi)
    sel = [(c, k) for c, lv, _, k, kind in tree if lv == level and kind == 'seq' and np.isfinite(rng[c][0])]
    if len(sel) < 2:
        return [], []
    E = len(n_ptr) - 1
    pit = (T.pi[:, 0, 0] + T.pi[:, 0, 2]) if T.pi is not None else torch.zeros(E, device=d.device)
    cs = np.asarray([c for c, _ in sel]); nets = np.asarray([k for _, k in sel])
    pi_ = torch.as_tensor([pin_in[c] for c in cs], device=d.device)
    po = torch.as_tensor([pin_out[c] for c in cs], device=d.device)
    sin = T.smax[pi_, 0].cpu().numpy()
    load = pit[d.p_net[po]].cpu().numpy()
    mid = np.array([(rng[c][0] + rng[c][1]) / 2 for c in cs])
    cur = [names['masters'][int(a['c_master'][c])] for c in cs]
    dl = np.stack([table[m][5](sin, load).numpy() for m in order], 1)              # (n, sizes)
    ok = np.stack([(load <= table[m][3]) & (table[m][6](sin, load).numpy() <= target_slew) for m in order], 1)
    ci = np.array([order.index(m) if m in order else -1 for m in cur])
    known = ci >= 0
    ok[np.arange(len(cs))[known], ci[known]] = True
    ok &= known[:, None]
    est = mid[:, None] + dl - dl[np.arange(len(cs)), np.maximum(ci, 0)][:, None]
    est = np.where(ok, est, np.nan)
    # the buffers' input capacitance loads their driver (the parent or a repeater): a resize moves the driver's delay,
    # and with it every sibling on that net (the driver's Liberty delay at its own input slew and its net's load +
    # the change in input capacitance); picks iterate to a fixed point
    cin = np.array([table[m][2] for m in order])
    in_net = pnet[pi_.cpu().numpy()]
    dpin = T.net_driver[in_net]
    dcell = T.p_cell_np[np.maximum(dpin, 0)]
    dm = [names['masters'][int(a['c_master'][c])] if p_ >= 0 and a['p_mpin'][p_] >= 0 else None
          for c, p_ in zip(dcell, dpin)]
    has_d = np.array([m in table for m in dm])
    din = {}
    for c in np.unique(dcell[has_d]):
        q = np.nonzero(a['p_cell'] == c)[0]
        q = q[a['p_dir'][q] != 1]
        din[int(c)] = int(q[0]) if len(q) else -1
    d_sin = np.array([float(T.smax[din[int(c)], 0]) if h and din.get(int(c), -1) >= 0 else 0.0 for c, h in zip(dcell, has_d)])
    d_load = pit[torch.as_tensor(in_net, device=d.device)].cpu().numpy()
    un, inv = np.unique(in_net, return_inverse=True)
    lo0 = np.array([rng[c][0] for c in cs]); hi0 = np.array([rng[c][1] for c in cs])
    rows_ = np.arange(len(cs))
    groups = [(m, np.nonzero(np.array([x == m for x in dm]) & has_d)[0]) for m in set(x for x, h in zip(dm, has_d) if h)]

    def shift_of(pk):
        """For picks pk: per buffer, its driver's delay change from the children's input capacitance, and the
        estimates (n, sizes) with the driver's output slew change in the buffer's own delay."""
        dC = np.bincount(inv, weights=cin[pk] - cin[np.maximum(ci, 0)], minlength=len(un))[inv]
        sh = np.zeros(len(cs)); ds = np.zeros(len(cs))
        for m, g in groups:
            sh[g] = (table[m][5](d_sin[g], d_load[g] + dC[g]) - table[m][5](d_sin[g], d_load[g])).numpy()
            ds[g] = (table[m][6](d_sin[g], d_load[g] + dC[g]) - table[m][6](d_sin[g], d_load[g])).numpy()
        dl2 = np.stack([table[m][5](sin + ds, load).numpy() for m in order], 1)
        e = mid[:, None] + dl2 - dl[rows_, np.maximum(ci, 0)][:, None] + sh[:, None]
        return np.where(ok, e, np.nan)

    def spread_of(pk):
        """Predicted (spread of the level's subtrees, worst over clocks; total distance of their midpoints from their
        clock's median): the first decides, the second breaks ties (a move inside one tree that leaves a clock's
        extremes as they are)."""
        mv = shift_of(pk)[rows_, pk] - mid
        sp = max(float((hi0 + mv)[nets == k].max() - (lo0 + mv)[nets == k].min()) for k in np.unique(nets))
        m2 = mid + mv
        dev = sum(float(np.abs(m2[nets == k] - np.median(m2[nets == k])).sum()) for k in np.unique(nets))
        return sp, dev
    pick = np.maximum(ci, 0).copy()
    best_pk, best_sp = pick.copy(), spread_of(pick)
    for _ in range(6):
        e2 = shift_of(pick)
        new = pick.copy()
        for k in np.unique(nets):
            g = np.nonzero((nets == k) & known)[0]
            if len(g) < 2:
                continue
            lo, hi = np.nanmin(e2[g], 1), np.nanmax(e2[g], 1)
            a_, b_ = lo.max(), hi.min()
            t = float(np.clip(np.median(mid[g]), a_, b_)) if a_ <= b_ else (a_ + b_) / 2
            new[g] = np.nanargmin(np.abs(e2[g] - t), 1)
        sp = spread_of(new)
        if sp[0] < best_sp[0] - 1e-6 or (sp[0] <= best_sp[0] + 1e-6 and sp[1] < best_sp[1] - 1e-6):
            best_pk, best_sp = new.copy(), sp
        if np.array_equal(new, pick):
            break
        pick = new
    pick = best_pk
    out_c, out_m = [], []
    for c, j, m in zip(cs, pick, cur):
        if order[j] != m:
            out_c.append(int(c)); out_m.append(order[j])
    return out_c, out_m


def _pad(d, domains, table, pad, target_slew, stats, passes=3):
    """Latency padding across the register trees of a clock (as TritonCTS's latency balancing inserts delay buffers):
    behind clock gates, registers sit stages deeper than those on the clock's own net. Per clock, the target is its
    latest register tree (the latest arrival at its registers); a tree earlier by more than half a stage gets a chain
    of `pad` buffers at its root, round(gap / stage) of them. The first stage is pad's Liberty delay driving its own
    input at its own output transition; then, re-timed, each chain's measured delay per stage, adding or removing
    pads (up to `passes` times). Returns the number of pad buffers."""
    if len(domains) < 2:
        return 0
    names = d.names
    cin = table[pad][2]
    s = target_slew
    for _ in range(8):
        s = table[pad][4](s, cin)
    stage0 = float(table[pad][5]([s], [cin])[0])
    chains = {root: [] for root, _, _, _ in domains}            # root -> [(pad cell, its output net)]
    stage = {root: stage0 for root, _, _, _ in domains}
    last, orig = {}, {}                                         # root -> its input net before any pad
    for _ in range(passes):
        res = d.timing(crpr=False)
        T = res.timer
        n_ptr = d.raw['n_ptr'].astype(np.int64)
        pnet = d.p_net.cpu().numpy()
        arr = T.amax[:, 0].amax(-1).cpu().numpy()
        leaves = sorted({c for _, lv, _, _ in domains for c in lv})
        pins = np.nonzero(np.isin(d.raw['p_cell'], leaves))[0]
        po = pins[d.raw['p_dir'][pins] == 1]
        out = dict(zip(d.raw['p_cell'][po].tolist(), po.tolist()))
        late = {}
        for root, lv, _ck, _k in domains:
            v = []
            for c in lv:
                n = int(pnet[out[c]])
                ld = np.arange(n_ptr[n], n_ptr[n + 1])
                v.append(arr[ld[ld != out[c]]])
            v = np.concatenate(v) if v else np.zeros(0)
            v = v[np.isfinite(v)]
            late[root] = float(v.max()) if len(v) else np.nan
        for root, (L0, n0) in last.items():                      # measured delay per pad stage
            dn = len(chains[root]) - n0
            if dn and np.isfinite(late[root]) and np.isfinite(L0) and (late[root] - L0) / dn > 0:
                stage[root] = (late[root] - L0) / dn
        target = {}
        for root, _, ck, _ in domains:
            if np.isfinite(late[root]):
                target[ck] = max(target.get(ck, -np.inf), late[root])
        delta = {}
        for root, _, ck, _ in domains:
            if np.isfinite(late[root]):
                g = target[ck] - late[root]
                k = int(round(g / stage[root]))
                k = max(k, -len(chains[root]))
                if k and abs(g) > stage[root] / 2:
                    delta[root] = k
        if not delta:
            break
        last = {root: (late[root], len(chains[root])) for root in delta}
        new_cells, new_nets = [], []
        with d.edit() as e:
            for root, k in delta.items():
                q = e.cell_pins(root)
                p_in = int(q[d.raw['p_dir'][q] != 1][0])
                ch = chains[root]
                orig.setdefault(root, int(pnet[p_in]))
                if k > 0:
                    prev = ch[-1][1] if ch else orig[root]
                    xy = (d.pos[root] + d.size[root] / 2).tolist()
                    for _ in range(k):
                        c = e.add_cell(pad, xy, name=e._unique('attar_cts_pad_', e._names))
                        o = e.add_net(name=e._unique('attar_cts_net_', e._net_names or set(names['nets'])))
                        e.connect(c, table[pad][0], prev)
                        e.connect(c, table[pad][1], o)
                        new_cells.append(c); new_nets.append(o)
                        ch.append((c, o))
                        prev = o
                    e.move([p_in], prev)
                else:
                    for c, _o in ch[k:]:
                        e.remove_cell(c)
                    del ch[k:]
                    e.move([p_in], ch[-1][1] if ch else orig[root])
        if new_nets:
            na = names.setdefault('net_attrs', {'default': '', 'except': {}})
            for n in new_nets:
                d.raw['n_flags'][n] |= N_CLOCK
                na['except'][str(n)] = ' + USE CLOCK'
            d.n_flags[torch.as_tensor(new_nets, device=d.device)] |= N_CLOCK
            nc = torch.as_tensor(new_cells, device=d.device)
            pos = d.pos.clone()
            pos[nc] = pos[nc] - d.size[nc] / 2
            d.pos = pos
            stats['pad_legalize'] = _legalize_new(d, nc, verbose=False)
        d.propagate_clocks()
    return sum(len(c) for c in chains.values())


def _latencies(d, res, leaf_cells):
    """(min, max) clock arrival over the register clock pins (rising), their spread, and per leaf buffer the mean
    arrival of its loads."""
    T = res.timer
    if not len(T.k_lib):
        return (0.0, 0.0), 0.0, {}
    ck = torch.unique(T.k_fr)
    arr = T.amax[ck, 0].amax(1)                      # latest rise over tags
    ok = torch.isfinite(arr)
    ck, arr = ck[ok], arr[ok]
    lat = (float(arr.min()), float(arr.max()))
    per = {}
    if leaf_cells:
        drv_cell = torch.as_tensor(T.p_cell_np, device=d.device)[torch.as_tensor(T.net_driver, device=d.device)
                                                                    .clamp(min=0)[d.p_net[ck]]]
        lc = torch.as_tensor(leaf_cells, device=d.device)
        for c in leaf_cells:
            m = drv_cell == c
            if bool(m.any()):
                per[c] = float(arr[m].mean())
        del lc
    return lat, lat[1] - lat[0], per


def _legalize_new(d, cells, verbose=False):
    """Legalize the cells given (new buffers) with the rest of the placement as obstacles (attar.naive.lg)."""
    from .lg import legalize
    keep = d.movable.clone()
    try:
        m = torch.zeros_like(keep)
        m[cells] = True
        d.movable = m & keep
        st = legalize(d, verbose=verbose, improve=False)
    finally:
        d.movable = keep
    return {k: v for k, v in st.items() if isinstance(v, (int, float, str, bool))}
