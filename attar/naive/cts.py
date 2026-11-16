"""Naive clock tree synthesis: buffered clock trees for a placed design, balanced on Attar's propagated-clock timer.

For every net of an ideal clock's network (attar.timing.sta: from the clock source through nets and buffers or gates
up to the register clock pins) whose loads are many or far apart:
  1. clusters   the loads are cut into balanced groups of at most leaf_size by recursive median bisection along the
                longer side of their bounding box (geometric, deterministic, on the GPU); each group gets a buffer at its
                loads' centroid. The buffers are clustered the same way, at most `fanout` per group, level by level,
                until one root buffer is left; the net's driver drives it.
  2. sizes      each buffer is the smallest of the library's buffers whose output transition at its estimated load
                (the pins' capacitance plus clock wire RC on the group's half perimeter) stays under target_slew, from
                the Liberty tables.
  3. netlist    the buffers and their nets (USE CLOCK) go into the netlist (attar.netlist.Edit); the new buffers alone
                are legalized (the rest of the placement is kept as obstacles).
  4. timing     the clocks become propagated (Design.propagate_clocks): the timer times the trees.
  5. balance    a few rounds of resizing the leaf buffers: those whose loads' clock arrival is early by more than tol
                get the next smaller size, late ones the next larger (one timing per round).
Returns statistics: buffers per level, latency, skew (the spread of the register clock pins' arrival), slew limit
violations on the clock nets, seconds. Export with d.export (the nets carry USE CLOCK); a flow then times it with
set_propagated_clock.
"""
import math
import time

import numpy as np
import torch

from ..io.distill import N_CLOCK


def _bisect(xy, idx, cap):
    """Balanced groups of at most cap points: recursive median split along the longer side. idx (n,) -> [arrays]."""
    out, stack = [], [idx]
    while stack:
        g = stack.pop()
        if len(g) <= cap:
            out.append(g)
            continue
        p = xy[g]
        span = p.max(0) - p.min(0)
        ax = int(np.argmax(span))
        # split so that both halves fill whole groups where possible (balanced leaf counts)
        n_groups = math.ceil(len(g) / cap)
        left = (n_groups // 2) * len(g) // n_groups
        o = np.argsort(p[:, ax], kind='stable')
        stack.append(g[o[left:]]); stack.append(g[o[:left]])
    return out


def _buffer_table(d, masters):
    """Per buffer master: (input pin, output pin, input capacitance, max capacitance, a function (slew, load) ->
    worst output transition) from the library chain's Liberty tables."""
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
        out[m] = (s[0], s[1], cin, cmax, slew, delay)
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


def _pick(table, order, load, target_slew, in_slew=20.0):
    """The smallest buffer (order: small first) meeting target_slew and its max capacitance at load, else the largest."""
    for m in order:
        _, _, _, cmax, slew, _ = table[m]
        if load <= cmax and slew(in_slew, load) <= target_slew:
            return m
    return order[-1]


def cts(d, leaf_size=20, fanout=8, target_slew=None, buffers=None, balance_rounds=4, tol=2.0, min_sinks=2,
        verbose=True):
    """Build buffered clock trees (see the module doc) in place. buffers: the buffer masters to use (default: the
    library's, d.buffers()); target_slew: ps (default: a quarter of the library's default max transition, else 80).
    Returns statistics."""
    d.require('netlist_edit')
    t0 = time.time()
    res0 = d.timing()
    T = res0.timer
    a, names = d.raw, d.names
    pdir = a['p_dir'].astype(np.int64)
    n_ptr = a['n_ptr'].astype(np.int64)
    rc = T.info.get('rc') or {}
    r_clk, c_clk = (rc.get('clock') or rc.get('signal') or [0.0, 0.0])
    if target_slew is None:                         # a tenth of the library's default max transition
        lims = [float(x) for x in np.unique(d.library()[-1].a['lc_dlim'][:, 0]) if np.isfinite(x)] \
            if 'lc_dlim' in d.library()[-1].a else []
        target_slew = 0.1 * min(lims) if lims else 30.0
    # the clock nets to build: driven by an ideal clock network pin, with loads
    drv = T.net_driver
    nets = [k for k in range(len(n_ptr) - 1) if drv[k] >= 0 and T.iclk[drv[k]]
            and n_ptr[k + 1] - n_ptr[k] - 1 >= min_sinks]
    order = list(buffers) if buffers is not None else _design_buffers(d)
    table = _buffer_table(d, order)
    order = [m for m in order if m in table]
    if not order:
        raise ValueError('no usable buffer in the library')
    pin_xy = d.pin_pos().detach().cpu().numpy().astype(np.float64)
    cap = T.pin_cap.max(1).values.cpu().numpy().astype(np.float64)
    plan = []                                         # (net, [(level, buffer xy, master, children: pins or nodes)])
    stats = {'nets': 0, 'buffers': 0, 'levels': []}
    for k in nets:
        pins = np.arange(n_ptr[k], n_ptr[k + 1])
        loads = pins[(pdir[pins] != 1) & (pins != drv[k])]
        if len(loads) < min_sinks:
            continue
        nodes = []                                    # (level, xy, master, children (pins or node ids), load)
        groups = _bisect(pin_xy, loads, leaf_size)
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
        plan.append((k, nodes, cur[0]))
        stats['nets'] += 1
        stats['buffers'] += len(nodes)
        stats['levels'].append(level)
    if not plan:
        return dict(stats, seconds=time.time() - t0, note='no clock net to build')
    # netlist edit: a buffer per node; the net's driver keeps the original net, which now drives the root buffer
    new_cells, new_nets, leaf_cells = [], [], []
    with d.edit() as e:
        for k, nodes, root in plan:
            cid, nid = {}, {}
            for i, (lev, xy, m, _kids, _) in enumerate(nodes):
                c = e.add_cell(m, (float(xy[0]), float(xy[1])), name=e._unique('attar_cts_%d_' % lev, e._names))
                n = e.add_net(name=e._unique('attar_cts_net_', e._net_names or set(names['nets'])))
                e.connect(c, table[m][1], n)
                cid[i], nid[i] = c, n
                new_cells.append(c); new_nets.append(n)
                if lev == 0:
                    leaf_cells.append(c)
            for i, (_lev, _xy, _m, kids, _) in enumerate(nodes):
                if kids[0] == 'pins':
                    e.move([int(p) for p in kids[1]], nid[i])
                else:
                    for j in kids[1]:
                        e.connect(cid[j], table[nodes[j][2]][0], nid[i])
            e.connect(cid[root], table[nodes[root][2]][0], k)
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
    # balance: per leaf buffer, its latency as a function of its size (its input arrival and slew from the timer, the
    # buffer's Liberty delay at its load, the wire part as now); every leaf takes the size closest to the slowest
    # leaf's best; re-time and repeat (input capacitances change the upper levels a little)
    hist, best = [], None
    leaf_t = np.asarray(leaf_cells, np.int64)
    for rnd in range(balance_rounds + 1):
        res = d.timing(crpr=False)                          # clock latencies only
        lat, skew, per_leaf = _latencies(d, res, leaf_cells)
        hist.append({'round': rnd, 'latency_max': lat[1], 'latency_min': lat[0], 'skew': skew})
        if best is None or skew < best[0]:
            best = (skew, [names['masters'][int(m)] for m in a['c_master'][leaf_t]])
        if rnd == balance_rounds or skew <= tol or not per_leaf:
            break
        cells, masters = _balance(d, res, per_leaf, table, order)
        if not cells:
            break
        d.resize(cells, masters, check=False)
        d.propagate_clocks()
    now = [names['masters'][int(m)] for m in a['c_master'][leaf_t]]
    if best is not None and now != best[1]:                  # keep the round with the smallest spread
        d.resize(list(leaf_t), best[1], check=False)
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


def _balance(d, res, per_leaf, table, order):
    """New sizes of the leaf buffers: est(leaf, size) = input arrival + Liberty delay(input slew, load) + the leaf's
    wire part now; target = the largest over leaves of their smallest estimate; each leaf the size nearest it."""
    T = res.timer
    a, names = d.raw, d.names
    pin_of = {}
    for c in per_leaf:
        for p in np.nonzero(a['p_cell'] == c)[0]:
            pin_of.setdefault(c, {})[names['master_pins'][int(a['c_master'][c])][int(a['p_mpin'][p]) -
                                     int(a['m_pin_ptr'][int(a['c_master'][c])])]] = int(p)
    E = len(a['n_ptr']) - 1
    pit = (T.pi[:, 0, 0] + T.pi[:, 0, 2]) if T.pi is not None else torch.zeros(E, device=d.device)
    cells = list(per_leaf)
    rows = []
    for c in cells:
        m = names['masters'][int(a['c_master'][c])]
        pi_, po = pin_of[c][table[m][0]], pin_of[c][table[m][1]]
        ain = float(T.amax[pi_, 0].amax()); sin = float(T.smax[pi_, 0]); aout = float(T.amax[po, 0].amax())
        load = float(pit[int(d.p_net[po])])
        rows.append((ain, sin, load, per_leaf[c] - aout))
    ain, sin, load, wire = (np.array(x) for x in zip(*rows))
    est = np.stack([ain + table[m][5](sin, load).numpy() + wire for m in order], 1)     # (leaves, sizes)
    target = est.min(1).max()
    pick = np.abs(est - target).argmin(1)
    out_c, out_m = [], []
    for c, j in zip(cells, pick):
        if names['masters'][int(a['c_master'][c])] != order[j]:
            out_c.append(c); out_m.append(order[j])
    return out_c, out_m


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
