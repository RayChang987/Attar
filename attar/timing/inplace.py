"""The timer updated in place after a netlist edit (Design.edit, attar.netlist.Edit.apply), as OpenSTA updates its
graph on network edits (makeInstanceAfter, connectPinAfter, disconnectPinBefore, deleteInstanceBefore) instead of
building it again: every pin-indexed array follows the pins' new numbering (Edit.pin_map), the new cells' arcs join
the instance arcs in a new timer's order, the nets get their net arcs again, the levels and the per-level stages are
made again from those, and the last run's arrivals, slews and delays are carried over. The next run then re-times the
cone of the edited nets only (attar.timing.incremental), bit for bit what a new timer gives.

Not covered (update returns why, and Design builds a new timer): global-routing parasitics, latches, pin clock
latencies, path delay exceptions, combinational loops, masters the timer has no tables for, new or removed cells that
are not combinational, and edits that touch a clock network, a constant net, a net an input port drives or a net with
several drivers."""
import numpy as np
import torch

from ..io.distill import D_OUT, N_CLOCK
from . import ir as timing_mod
from .ir import T_COMB, T_TRISTATE

INF = float('inf')


def touched_nets(e):
    """The nets an applied edit changed: new nets, the nets new pins joined, both nets of a moved pin, the nets of
    the removed cells' pins."""
    tn = set(range(e.E0, e.E0 + len(e.nets)))
    tn.update(int(q[3]) for q in e.pins)
    for p, k in e.moves.items():
        tn.add(int(k))
        tn.add(int(e._pnet[p]) if p < e.P0 else int(e.pins[p - e.P0][3]))
    tn.update(int(e._pnet[p]) for p in e.dropped)
    return np.array(sorted(tn), np.int64)


def _comb_master(t, m, funcs):
    """A master whose arcs are all combinational and whose outputs depend on an input (not a tie cell)."""
    ty = t['a_type'][t['a_master'] == m]
    if not np.isin(ty, (T_COMB, T_TRISTATE)).all():
        return False
    from ..io.liberty import parse_function
    fs = funcs.get(str(m)) or funcs.get(m) or {}
    return all(parse_function(f)[0] for f in fs.values())


def lib_map(T, t):
    """Old Liberty arc -> the new arrays' (ir.arrays lists the arcs of each used master as one block, in master
    order, so a master that comes into use or goes out of use shifts the blocks after it); -1 where the master's block
    is gone or not the same size. None: the same arrays."""
    om, nm = T._a_master, t['a_master'].astype(np.int64)
    if np.array_equal(om, nm):
        return None
    out = np.full(len(om), -1, np.int64)
    m1, s1, c1 = np.unique(om, return_index=True, return_counts=True)
    m2, s2, c2 = np.unique(nm, return_index=True, return_counts=True)
    blk = {int(m): (int(s), int(c)) for m, s, c in zip(m2, s2, c2)}
    for m, s, c in zip(m1.tolist(), s1.tolist(), c1.tolist()):
        b = blk.get(m)
        if b is not None and b[1] == c:
            out[s:s + c] = np.arange(b[0], b[0] + c)
    return out


def unsupported(T, e, tn, t, funcs, lmap):
    """Why the timer cannot follow the applied edit e in place ('' when it can). tn: touched_nets(e); t, funcs: the
    new arrays and functions (ir.arrays); lmap: lib_map(T, t)."""
    d = T.d
    stale = T.__dict__.get('_heavy_stale', False)          # (the levels and stages dropped: update_light)
    if T.parasitics != 'placement':
        return 'global-routing parasitics'
    if T._lq_n or (not stale and (len(T._lq_c) or T._arc_flags['ldq'].any())):
        return 'latches'
    if T._lat_at or T._ideal_pin_lat is not None or T.info.get('pin_latency'):
        return 'pin clock latencies'
    ex_cells = {int(c) for x in T.ex for c in list(x['from']) + list(x['to'])}
    if ex_cells & (set(e.removed) | {int(T.p_cell_np[p]) for p in e.moves if p < e.P0}):
        return 'a cell a path delay exception names'
    if len(T.loop_edges[0]) or len(T.loop_edges[1]) or T.n_cycle_edges:
        return 'combinational loops'
    a = d.raw
    cm = a['c_master']
    for m in {int(c[0]) for c in e.cells}:
        if not _comb_master(t, m, funcs):
            return 'a new cell that is not combinational (%s)' % d.names['masters'][m]
    if e.removed and stale:                         # (their masters' arcs)
        if not all(np.isin(t['a_type'][t['a_master'] == int(cm[c])], (T_COMB, T_TRISTATE)).all() for c in e.removed):
            return 'a removed cell that is not combinational'
    elif e.removed:                                 # (their arcs as the timer has them; a tie cell: a constant net)
        gone = np.isin(T._arc_cell, np.fromiter(e.removed, np.int64))
        if not np.isin(T._arcs_all[3][gone], (T_COMB, T_TRISTATE)).all():
            return 'a removed cell that is not combinational'
    if lmap is not None and not stale:              # (stale: a master's block goes only with its last instance)
        lib_id, fr, to = T._arcs_all[:3]
        keep = (e.pin_map[fr] >= 0) & (e.pin_map[to] >= 0)
        if (lmap[lib_id[keep]] < 0).any():
            return 'the Liberty arcs of a master changed'
    # the touched nets before the edit (the timer's numbering)
    E0 = e.E0
    old = tn[tn < E0]
    if (T.const_net[old] >= 0).any():
        return 'a constant net'
    if T.net_iclk[old].any() or T.net_clkrc[old].any() or (a['n_flags'][tn] & N_CLOCK).any():
        return 'a clock net'
    nptr = e._nptr
    if len(old):
        cnt = nptr[old + 1] - nptr[old]
        pins = np.repeat(nptr[old] - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt) + np.arange(int(cnt.sum()))
        if (T.clk_of[pins] >= 0).any():
            return 'a clock network'
        drv = T.net_driver[old]
        if ((drv < 0) & (cnt > 0)).any():
            return 'a net with several or no drivers'
        if (cm[T.p_cell_np[drv[drv >= 0]]] < 0).any():              # (IO pins have no master)
            return 'a net an input port drives'
    # after the edit (the design's numbering)
    nptr = a['n_ptr']
    pdir = a['p_dir'] if 'p_dir' in a else None
    for k in tn.tolist():
        ps = np.arange(int(nptr[k]), int(nptr[k + 1]))
        if not len(ps):
            continue
        dr = ps[pdir[ps] == D_OUT] if pdir is not None else ps[:0]
        if len(dr) != 1:
            return 'a net with several or no drivers'
        if a['p_mpin'][dr[0]] < 0:
            return 'a net an input port drives'
    return ''


def update(T, e):
    """Follow the applied edit e in place. Returns '' when done, else why not: the caller then drops the timer (it
    is unchanged, except after an edit that closed a combinational loop, found once the levels are made). With the
    CPU engine (a timed GPU timer; made here on the first edit) the update is update_light's, else every structure
    is made again (the levels and stages included)."""
    tn = touched_nets(e)
    t, functions, arc_when = timing_mod.arrays(T.d)
    lmap = lib_map(T, t)
    why = unsupported(T, e, tn, t, functions, lmap)
    if why:
        return why
    stale = T.__dict__.get('_heavy_stale', False)
    why = light_ok(T)
    if not why:
        T._in_update = True
        try:
            why = update_light(T, e, tn, t, functions, arc_when, lmap)
        finally:
            T.__dict__.pop('_in_update', None)
        return why
    if stale:                                       # (no levels to update: a new timer)
        return why
    return _update_full(T, e, tn, t, functions, arc_when, lmap)


def light_ok(T):
    """Why the CPU engine cannot follow edits for this timer ('' when it can: it is then made here if needed)."""
    if T.dev.type != 'cuda' or T.triton is None or not T.fused or T.precision != 'double':
        return 'no fused float64 GPU kernels'
    if not T.__dict__.get('_timed'):
        return 'not timed yet'
    if bool(T.prop_clk.any()):
        return 'propagated clocks (CRPR)'
    if T.__dict__.get('_cpu') is None:
        if T.__dict__.get('_heavy_stale') or T.__dict__.get('_cpu_failed'):
            return 'no CPU engine'
        inc = T._inc_get()
        if not inc.ok:
            return inc.why
        from .cpu_inc import CpuEngine
        try:
            T._cpu = CpuEngine(inc)
        except OSError:
            T._cpu_failed = True
            return 'no CPU engine (no compiler)'
    return ''


def _update_full(T, e, tn, t, functions, arc_when, lmap):
    d, dev = T.d, T.dev
    a = d.raw
    P0, E0 = e.P0, e.E0
    pm = np.asarray(e.pin_map, np.int64)
    pmo = pm[:P0]                                           # old pin -> new pin (-1: gone)
    p_cell = a['p_cell'].astype(np.int64); p_mpin = a['p_mpin'].astype(np.int64)
    P = len(p_cell)
    E = len(a['n_ptr']) - 1
    pdir = a['p_dir'].astype(np.int64) if 'p_dir' in a else np.full(P, 3)
    p_net = d.p_net.cpu().numpy()
    src = np.full(P, -1, np.int64)                          # new pin -> its handle (old pin, or P0 + new pin)
    live = np.nonzero(pm >= 0)[0]
    src[pm[live]] = live
    was = src < P0                                          # pins of before the edit
    so = np.where(was, src, 0)
    newp = np.nonzero(~was)[0]
    so_t = torch.as_tensor(so, device=dev); was_t = torch.as_tensor(was, device=dev)

    def rows(x, fill):                                      # a pin-indexed array in the new numbering
        if isinstance(x, torch.Tensor):
            y = x[so_t]
            y[~was_t] = fill
            return y
        y = x[so]
        y[~was] = fill
        return y

    old = _snapshot(T)
    # -- the Liberty tables, when the used masters changed (Liberty arcs renumbered by lmap below)
    info = T.info
    tmin = T._min_arrays(d, t) if T.lmin else None
    if lmap is not None:
        info['functions'], info['arc_when'] = functions, arc_when
        T._set_tables(t, tmin)
        if T.host is not None:
            T.host_tables = T.host.Tables(T.t_ix1, T.t_ix2, T.t_val, T.t_n)
        if T.triton is not None:
            T.t_n32 = T.t_n.to(torch.int32).contiguous()
    # -- pins: Liberty data, via resistance; clock network flags (the edit leaves clocks alone) -------------------------
    cap_n, capmin_n, ctp_n, sthr_n = T._pin_library(t, info, p_cell[newp], p_mpin[newp])
    if tmin is not None:
        capmin_n = T._pin_library(tmin, info, p_cell[newp], p_mpin[newp])[1]
    rv_n = T._pin_vias(d, info, info.get('rc') or {}, p_cell[newp], p_mpin[newp], np.zeros(len(newp), bool))
    nt = torch.as_tensor(newp, device=dev)

    def pin_data(x, new):
        y = rows(x, 0)
        y[nt] = torch.as_tensor(new, device=dev, dtype=y.dtype)
        return y
    T.pin_cap, T.pin_cap_min = pin_data(T.pin_cap, cap_n), pin_data(T.pin_cap_min, capmin_n)
    T.pin_ctp, T.sink_thr = pin_data(T.pin_ctp, ctp_n), pin_data(T.sink_thr, sthr_n)
    T.pin_rvia = pin_data(T.pin_rvia, rv_n)
    T.clk_of, T.par = rows(T.clk_of, -1), rows(T.par, 0)
    T.pclk, T.iclk = rows(T.pclk, False), rows(T.iclk, False)
    T.P, T.p_cell_np, T.pin_net = P, p_cell, d.p_net
    # -- instance arcs: the removed cells' go, the kept ones take the new pin numbers, the new cells' join in a new
    # timer's order (Liberty arc, then instance)
    lib_id, fr, to, atype, asense_lib = T._arcs_all
    keep = (pmo[fr] >= 0) & (pmo[to] >= 0)
    ki = np.nonzero(keep)[0]
    if lmap is not None:
        lib_id = lmap[lib_id]
    am = t['a_master'].astype(np.int64)
    a_from, a_to = t['a_from'].astype(np.int64), t['a_to'].astype(np.int64)
    pin_at = {(int(p_cell[p]), int(p_mpin[p])): int(p) for p in newp}
    nl, nf, ntt, nc = [], [], [], []
    for i, cell in enumerate(range(e.N0, e.N0 + len(e.cells))):
        m = int(e.cells[i][0])
        for lid in np.nonzero(am == m)[0].tolist():
            f, g = pin_at.get((cell, int(a_from[lid]))), pin_at.get((cell, int(a_to[lid])))
            if f is not None and g is not None:
                nl.append(lid); nf.append(f); ntt.append(g); nc.append(cell)
    NC = len(a['c_master'])
    nl, nf, ntt, nc = (np.asarray(x, np.int64) for x in (nl, nf, ntt, nc))
    o = np.argsort(nl * NC + nc, kind='stable')
    nl, nf, ntt, nc = nl[o], nf[o], ntt[o], nc[o]
    key_old = lib_id[ki] * NC + T._arc_cell[ki]
    at = np.searchsorted(key_old, nl * NC + nc, side='right')
    ins = lambda x, v: np.insert(x[ki], at, v)               # noqa: E731
    lib2, fr2, to2 = ins(lib_id, nl), np.insert(pmo[fr[ki]], at, nf), np.insert(pmo[to[ki]], at, ntt)
    atype2, asl2 = ins(atype, t['a_type'][nl].astype(np.int64)), ins(asense_lib, t['a_sense'][nl].astype(np.int64))
    amap = np.full(len(lib_id), -1, np.int64)               # old arc -> new arc
    amap[ki] = np.arange(len(ki)) + np.searchsorted(at, np.arange(len(ki)), side='right')
    f = T._arc_flags
    ones, zeros = np.ones(len(nl), bool), np.zeros(len(nl), bool)
    T._arc_flags = {'ck': ins(f['ck'], ones), 'arr': ins(f['arr'], ones), 'launch': ins(f['launch'], zeros),
                    'chk': ins(f['chk'], zeros), 'asense': ins(f['asense'], t['a_sense'][nl].astype(np.int64)),
                    'ldq': ins(f['ldq'], zeros), 'lq': ins(f['lq'], zeros)}
    T.cond_off = ins(T.cond_off, zeros)
    T._arcs_all = (lib2, fr2, to2, atype2, asl2)
    T._arc_cell = p_cell[fr2]
    # -- nets
    drv, n_fr, n_to = T._net_arcs(pdir, p_net, E)
    T.net_driver = drv
    T.const_net = np.concatenate([T.const_net, np.full(E - E0, -1, np.int64)])
    T.net_clkrc = np.concatenate([T.net_clkrc, np.zeros(E - E0, bool)])
    T.net_iclk = np.zeros(E, bool)
    T.net_iclk[drv >= 0] = T.iclk[drv[drv >= 0]]
    drv_t = torch.as_tensor(drv, device=dev)
    T.drv_nets = torch.nonzero(drv_t >= 0).flatten(); T.drv_pins = drv_t[T.drv_nets]
    # -- ports and constraints: pin numbers
    T._psrc = pmo[T._psrc]
    T.in_delay = [(int(pmo[p]), ci, x) for p, ci, x in T.in_delay]
    T.out_delay = [(int(pmo[p]), ci, x) for p, ci, x in T.out_delay]
    same = T.in_slew_min is T.in_slew
    T.in_slew = {int(pmo[p]): v for p, v in T.in_slew.items()}
    T.in_slew_min = T.in_slew if same else {int(pmo[p]): v for p, v in T.in_slew_min.items()}
    T.ex_ports = {int(pmo[p]) for p in T.ex_ports}
    # -- the graph, its levels and stages; the wire model; the fused kernels' arrays
    T._set_graph(n_fr, n_to, p_net, np.zeros(len(n_to), bool))
    if T.n_cycle_edges:                                 # (a new timer breaks it where OpenSTA does: _loop_edges)
        return 'the edit closed a combinational loop'
    lim = list(T.limitations)
    T.__dict__.pop('_reach', None)                      # (made again from the new levels by _wire_setup)
    T._prepare_static()
    T._big = None                                       # (the old one's pins: _wire_setup makes it again)
    loc = timing_mod.pin_locations(d)[p_mpin[newp]] if len(newp) else np.zeros((0, 2))
    lok = np.isfinite(loc).all(1)
    loc_n, loc_ok = rows(T._loc_n, 0.0), rows(T._loc_ok, False)
    loc_n[nt] = torch.as_tensor(np.nan_to_num(loc), dtype=loc_n.dtype, device=dev)
    loc_ok[nt] = torch.as_tensor(lok, device=dev)
    T._wire_setup(locs=(loc_n, loc_ok))
    T.limitations = lim + [m for m in T.limitations if m not in lim]
    if T.triton is not None:
        T._prepare_triton()
    for k in ('_graph', '_ws', '_inc', '_cpu', '_crpr', '_power', '_load_seg', '_mp_names', '_sink_drv', '_g_out',
              '_pos_buf'):
        T.__dict__[k] = None
    _carry(T, old, rows, amap, tn, n_fr, n_to, pmo)
    T._gen = T.__dict__.get('_gen', 0) + 1
    return ''


def _snapshot(T):
    """What the last run left that the next one reuses: arrivals, slews, wire delays per load, the stages' recorded
    delays with each entry's key and each lane's net arc, the wire model's cache, the port wires."""
    s = {k: T.__dict__.get(k) for k in ('amax', 'amin', 'smax', 'smin', '_wb', '_rec', '_w', '_pa_w', 'pa')}
    s['keys'], s['lanes'] = _entry_keys(T)
    s['sel'] = {k: v.copy() for k, v in T._sel.items()}
    s['n_fr'], s['n_to'] = T.n_fr, T.n_to
    s['timed'] = T.__dict__.get('_timed', False) and s['_rec'] is not None and len(s['_rec']) == len(T.tri_lev)
    return s


def _entry_keys(T):
    """Per fused stage entry (Timer.tri_lev, in run order) a key that names it across timers: c-selection arc * 4 +
    output transition * 2 + input transition, or for a launch -((l-selection arc * 2 + output transition) * NT + tag)
    - 1; per lane (entry, net arc). Selection indices are the timer's (mapped by the caller)."""
    keys, lanes = [], []
    if getattr(T, 'tri_lev', None) is None:
        return None, None
    off = 0
    for lv, (kind, src) in zip(T.tri_lev, T._tri_src):
        lib, fi, t_, orf, net, t0, tag, ea, en, s0, arr = lv
        srt = torch.as_tensor(src, device=T.dev)
        if kind == 'c':
            keys.append(srt * 4 + orf.long() * 2 + (fi.long() & 1))
        else:                                       # (a launch arc has one entry per tag: path delay exceptions)
            keys.append(-((srt * 2 + orf.long()) * T.NT + tag.long()) - 1)
        lanes.append(torch.stack([ea.long() + off, en.long()], 1))
        off += len(lib)
    return keys, lanes


def _sel_map(old_sel, amap, flag):
    """Old selection index -> new selection index (-1: gone), through the arc map."""
    rank = np.cumsum(flag) - 1
    na = amap[old_sel]
    return np.where(na >= 0, rank[np.maximum(na, 0)], -1)


def _carry(T, old, rows, amap, tn, n_fr, n_to, pmo):
    """Carry the last run's results into the new numbering (Timer state for the incremental re-timing): arrivals and
    slews per pin, wire delays per load row, the stages' recorded delays per entry and lane (matched by key), the port
    wires, the wire model's cache; the edited nets are marked for wires() to redo."""
    dev, P, NT = T.dev, T.P, T.NT
    if old['amax'] is None or not old['timed']:
        T._timed = False
        return
    T.amax, T.amin = rows(old['amax'], -INF), rows(old['amin'], INF)
    T.smax, T.smin = rows(old['smax'], 0.0), rows(old['smin'], 0.0)
    T.__dict__.pop('_load', None); T.__dict__.pop('_load_cw', None)   # (summed when read, after the next run)
    so_rows = lambda x, fill: rows(x[:len(x) // 2 * 2].view(-1, 2), fill).reshape(-1)  # noqa: E731
    T._wb = (so_rows(old['_wb'][0], -INF), so_rows(old['_wb'][1], INF))
    # net arcs: old -> new (the same driver and sink, renumbered)
    A_new = len(n_to)
    arc_of_sink = np.full(P, -1, np.int64)
    arc_of_sink[n_to] = np.arange(A_new)
    of, ot = old['n_fr'].cpu().numpy(), old['n_to'].cpu().numpy()
    nt_ = pmo[ot]
    na = np.where(nt_ >= 0, arc_of_sink[np.maximum(nt_, 0)], -1)
    ok = (na >= 0) & (pmo[of] == n_fr[np.maximum(na, 0)])
    nmap = np.where(ok, na, -1)
    # stage entries and lanes, by key
    c_map = _sel_map(old['sel']['c'], amap, T._arc_flags['ck'])
    l_map = _sel_map(old['sel']['l'], amap, T._arc_flags['launch'])
    ok_keys, ok_lanes = old['keys'], old['lanes']
    cm_t, lm_t, nm_t = (torch.as_tensor(x, device=dev) for x in (c_map, l_map, nmap))

    def remap_key(k):
        c = k >= 0
        out = torch.full_like(k, -(1 << 62))
        kc = k[c]
        nc = cm_t[kc // 4] if len(kc) else kc
        out[c] = torch.where(nc >= 0, nc * 4 + kc % 4, torch.full_like(kc, -(1 << 62)))
        kl = -k[~c] - 1
        tg, kl = kl % NT, kl // NT
        nl = lm_t[kl // 2] if len(kl) else kl
        out[~c] = torch.where(nl >= 0, -((nl * 2 + kl % 2) * NT + tg) - 1, torch.full_like(kl, -(1 << 62)))
        return out
    k_old = remap_key(torch.cat(ok_keys)) if ok_keys else torch.zeros(0, dtype=torch.long, device=dev)
    k_new_l, l_new_l = _entry_keys(T)
    k_new = torch.cat(k_new_l) if k_new_l else torch.zeros(0, dtype=torch.long, device=dev)
    rec_old = old['_rec']
    d_old = [torch.cat([r[j] for r in rec_old]) if rec_old else torch.zeros(0, device=dev) for j in range(4)]
    so_, oi = torch.sort(k_old)
    pos = torch.searchsorted(so_, k_new).clamp(max=max(len(so_) - 1, 0))
    hit = (so_[pos] == k_new) if len(so_) else torch.zeros(len(k_new), dtype=torch.bool, device=dev)
    ematch = torch.where(hit, oi[pos] if len(oi) else pos, torch.full_like(pos, -1))   # new entry -> old entry
    # lanes: (new entry, new net arc) keys
    o2n = torch.full((len(k_old),), -1, dtype=torch.long, device=dev)
    o2n[ematch[ematch >= 0]] = torch.nonzero(ematch >= 0).flatten()
    lo = torch.cat(ok_lanes) if ok_lanes else torch.zeros(0, 2, dtype=torch.long, device=dev)
    ln = torch.cat(l_new_l) if l_new_l else torch.zeros(0, 2, dtype=torch.long, device=dev)
    A1 = max(A_new, 1)
    lo_e = o2n[lo[:, 0]] if len(lo) else lo[:, 0]
    lo_a = nm_t[lo[:, 1]] if len(lo) else lo[:, 1]
    lk_old = torch.where((lo_e >= 0) & (lo_a >= 0), lo_e * A1 + lo_a, torch.full_like(lo_e, -1))
    lk_new = ln[:, 0] * A1 + ln[:, 1]
    so2, oi2 = torch.sort(lk_old)
    pos2 = torch.searchsorted(so2, lk_new).clamp(max=max(len(so2) - 1, 0))
    hit2 = (so2[pos2] == lk_new) & (lk_new >= 0) if len(so2) else torch.zeros(len(lk_new), dtype=torch.bool, device=dev)
    lmatch = torch.where(hit2, oi2[pos2] if len(oi2) else pos2, torch.full_like(pos2, -1))
    T._rec = []
    eo = lo_ = 0
    for lv in T.tri_lev:
        N, M = len(lv[0]), len(lv[7])
        em, lm = ematch[eo:eo + N], lmatch[lo_:lo_ + M]
        r = []
        for j, (mt, n) in enumerate(((em, N), (lm, M), (em, N), (lm, M))):
            v = torch.zeros(n, device=dev)
            g = mt >= 0
            if len(d_old[j]) and bool(g.any()):
                v[g] = d_old[j][mt[g]].to(v.dtype)
            r.append(v)
        T._rec.append(tuple(r))
        eo += N; lo_ += M
    # port wires: per port net arc
    pa_old = old['pa'].cpu().numpy() if old['pa'] is not None else np.zeros(0, np.int64)
    pa_new = T.pa.cpu().numpy()
    T._pa_w = []
    if old['_pa_w']:
        dst = np.searchsorted(pa_new, nmap[pa_old])
        okp = (nmap[pa_old] >= 0) & (dst < len(pa_new))
        okp[okp] &= pa_new[dst[okp]] == nmap[pa_old][okp]
        dt_, st_ = torch.as_tensor(dst[okp], device=dev), torch.as_tensor(np.nonzero(okp)[0], device=dev)
        for rf, wx, wn in old['_pa_w']:
            x, n = torch.zeros(len(pa_new), device=dev), torch.zeros(len(pa_new), device=dev)
            x[dt_] = wx[st_].to(x.dtype); n[dt_] = wn[st_].to(n.dtype)
            T._pa_w.append((rf, x, n))
    # the wire model's cache: pins renumbered (new pins: unknown, so their nets are redone), nets appended; the edited
    # nets are redone too (a net that lost a pin has none that moved)
    w = old['_w']
    if w is not None:
        E = T.ev.E
        grow = lambda x: torch.cat([x, torch.zeros(E - len(x), *x.shape[1:], dtype=x.dtype, device=dev)])  # noqa: E731
        T._w = {'p': rows(w['p'], float('nan')), 'caps': rows(w['caps'], float('nan')),
                'rv': rows(w['rv'], float('nan'))}
        for key in ('main', 'clk'):
            v = w[key]
            T._w[key] = None if v is None else (rows(v[0], 0.0), grow(v[1]), grow(v[2]))
    force = torch.zeros(T.ev.E, dtype=torch.bool, device=dev)
    force[torch.as_tensor(tn, device=dev)] = True
    T._force_dirty = force
    T._timed = True


def update_light(T, e, tn, t, functions, arc_when, lmap):
    """Follow the applied edit e with what the next run needs, as OpenSTA updates its graph: the pin-indexed arrays
    (renumbered: Edit.pin_map), the nets, the checks' pins, the wire model, the device state; the CPU engine adds the
    new cells' arcs and the edited nets' loads under its own pin handles (attar.timing.cpu_inc). The levels and
    stages (sta.HEAVY) are dropped: reading them rebuilds the timer (Timer._materialize), and so does a run the
    engine cannot take (Timer._run_light). Returns '' (done) or why not (before any change: the caller drops the
    timer)."""
    from .sta import HEAVY
    d, dev = T.d, T.dev
    a = d.raw
    C = T._cpu
    P0, E0 = e.P0, e.E0
    pm = np.asarray(e.pin_map, np.int64)
    pmo = pm[:P0]                                           # old pin -> new pin (-1: gone)
    p_cell = a['p_cell'].astype(np.int64); p_mpin = a['p_mpin'].astype(np.int64)
    P = len(p_cell)
    E = len(a['n_ptr']) - 1
    pdir = a['p_dir'] if 'p_dir' in a else np.full(P, 3)
    p_net = d.p_net.cpu().numpy()
    src = np.full(P, -1, np.int64)                          # new pin -> its handle in the edit (old pin, or P0 + j)
    live = np.nonzero(pm >= 0)[0]
    src[pm[live]] = live
    was = src < P0
    so = np.where(was, src, 0)
    newp = np.nonzero(~was)[0]
    drv_t, n_fr_t, n_to_t = T._net_arcs_dev(d.pin_dir, d.p_net, E)
    drv = drv_t.cpu().numpy()
    # reachability (which nets get the exact host tree): the new pins from their nets' drivers (new cells are
    # combinational); an edit that changes an old pin's would change trees elsewhere
    reach_old = T._reachable()
    reach = np.zeros(P, bool)
    reach[was] = reach_old[so[was]]
    cells_new = sorted({int(p_cell[p]) for p in newp})
    for _ in range(len(cells_new) + 1):                     # (new cells feeding new cells: in order)
        for c in cells_new:
            ps = newp[p_cell[newp] == c]
            ins = ps[pdir[ps] != D_OUT]
            r = bool(np.any([reach[drv[p_net[p]]] if drv[p_net[p]] >= 0 else False for p in ins]))
            for p in ps:
                reach[p] = r if pdir[p] == D_OUT else (reach[drv[p_net[p]]] if drv[p_net[p]] >= 0 else False)
    for p, k in e.moves.items():                            # an old pin on another net: same reachability
        q = int(pm[p])
        if p < P0 and q >= 0 and pdir[q] != D_OUT and drv[k] >= 0 and reach[drv[k]] != reach_old[p]:
            return 'a pin moved between nets of different reachability'
    in_sync = C.ver is not None and C.ver == T.__dict__.get('_state_ver', 0)
    so_t = torch.as_tensor(so, device=dev); was_t = torch.as_tensor(was, device=dev)

    def rows(x, fill):                                      # a pin-indexed array in the new numbering
        if isinstance(x, torch.Tensor):
            y = x[so_t]
            y[~was_t] = fill
            return y
        y = x[so]
        y[~was] = fill
        return y
    old_state = {k: T.__dict__.get(k) for k in ('amax', 'amin', 'smax', 'smin', '_wb', '_w')}
    # -- the library tables (the used masters changed: Liberty arcs renumbered by lmap)
    info = T.info
    tmin = T._min_arrays(d, t) if T.lmin else None
    if lmap is not None:
        info['functions'], info['arc_when'] = functions, arc_when
        T._set_tables(t, tmin)
        if T.host is not None:
            T.host_tables = T.host.Tables(T.t_ix1, T.t_ix2, T.t_val, T.t_n)
        if T.triton is not None:
            T.t_n32 = T.t_n.to(torch.int32).contiguous()
        if len(T.k_lib):
            T.k_lib = torch.as_tensor(lmap, device=dev)[T.k_lib]
    # -- pins
    cap_n, capmin_n, ctp_n, sthr_n = T._pin_library(t, info, p_cell[newp], p_mpin[newp])
    if tmin is not None:
        capmin_n = T._pin_library(tmin, info, p_cell[newp], p_mpin[newp])[1]
    rv_n = T._pin_vias(d, info, info.get('rc') or {}, p_cell[newp], p_mpin[newp], np.zeros(len(newp), bool))
    nt = torch.as_tensor(newp, device=dev)

    def pin_data(x, new):
        y = rows(x, 0)
        y[nt] = torch.as_tensor(new, device=dev, dtype=y.dtype)
        return y
    T.pin_cap, T.pin_cap_min = pin_data(T.pin_cap, cap_n), pin_data(T.pin_cap_min, capmin_n)
    T.pin_ctp, T.sink_thr = pin_data(T.pin_ctp, ctp_n), pin_data(T.sink_thr, sthr_n)
    T.pin_rvia = pin_data(T.pin_rvia, rv_n)
    T.clk_of, T.par = rows(T.clk_of, -1), rows(T.par, 0)
    T.pclk, T.iclk = rows(T.pclk, False), rows(T.iclk, False)
    T.P, T.p_cell_np, T.pin_net = P, p_cell, d.p_net
    T._reach = reach
    # -- nets
    T.net_driver = drv
    T.const_net = np.concatenate([T.const_net, np.full(E - E0, -1, np.int64)])
    T.net_clkrc = np.concatenate([T.net_clkrc, np.zeros(E - E0, bool)])
    T.net_iclk = np.zeros(E, bool)
    T.net_iclk[drv >= 0] = T.iclk[drv[drv >= 0]]
    T.drv_nets = torch.nonzero(drv_t >= 0).flatten(); T.drv_pins = drv_t[T.drv_nets]
    T.n_fr, T.n_to = n_fr_t, n_to_t
    T.n_net = d.p_net[n_to_t]
    pnet = np.zeros(E, bool)
    pnet[drv >= 0] = p_mpin[drv[drv >= 0]] < 0
    T._port_net = torch.as_tensor(pnet, device=dev)          # nets an input port drives (the GPU's port wires)
    # -- ports, constraints and checks: pin numbers
    pmo_t = torch.as_tensor(pmo, device=dev)
    T._psrc = pmo[T._psrc]
    T.in_delay = [(int(pmo[p]), ci, x) for p, ci, x in T.in_delay]
    T.out_delay = [(int(pmo[p]), ci, x) for p, ci, x in T.out_delay]
    same = T.in_slew_min is T.in_slew
    T.in_slew = {int(pmo[p]): v for p, v in T.in_slew.items()}
    T.in_slew_min = T.in_slew if same else {int(pmo[p]): v for p, v in T.in_slew_min.items()}
    T.ex_ports = {int(pmo[p]) for p in T.ex_ports}
    if len(T.k_lib):
        T.k_fr, T.k_to = pmo_t[T.k_fr], pmo_t[T.k_to]
    T.op_su, T.op_ho, T.op_ex = ((pmo_t[o[0]],) + tuple(o[1:]) if len(o[0]) else o for o in (T.op_su, T.op_ho, T.op_ex))
    # -- the wire model
    lim = list(T.limitations)
    T._big = None
    loc = timing_mod.pin_locations(d)[p_mpin[newp]] if len(newp) else np.zeros((0, 2))
    lok = np.isfinite(loc).all(1)
    loc_n, loc_ok = rows(T._loc_n, 0.0), rows(T._loc_ok, False)
    loc_n[nt] = torch.as_tensor(np.nan_to_num(loc), dtype=loc_n.dtype, device=dev)
    loc_ok[nt] = torch.as_tensor(lok, device=dev)
    T._wire_setup(locs=(loc_n, loc_ok))
    T.limitations = lim + [m for m in T.limitations if m not in lim]
    # -- the levels and stages go (sta.HEAVY); so do the CUDA graph and what was made over the old pins
    for k in HEAVY:
        T.__dict__.pop(k, None)
    T._heavy_stale = True
    for k in ('_graph', '_ws', '_inc', '_crpr', '_power', '_load_seg', '_mp_names', '_sink_drv', '_g_out',
              '_pos_buf', '_load', '_load_cw'):
        T.__dict__.pop(k, None)
    T._graph = None; T._ws = None; T._inc = None
    # -- the device state in the new numbering; the edited nets' wires redone at the next run
    T.amax, T.amin = rows(old_state['amax'], -INF), rows(old_state['amin'], INF)
    T.smax, T.smin = rows(old_state['smax'], 0.0), rows(old_state['smin'], 0.0)
    so_rows = lambda x, fill: rows(x[:len(x) // 2 * 2].view(-1, 2), fill).reshape(-1)  # noqa: E731
    T._wb = (so_rows(old_state['_wb'][0], -INF), so_rows(old_state['_wb'][1], INF))
    w = old_state['_w']
    if w is not None:
        grow = lambda x: torch.cat([x, torch.zeros(E - len(x), *x.shape[1:], dtype=x.dtype, device=dev)])  # noqa: E731
        T._w = {'p': rows(w['p'], float('nan')), 'caps': rows(w['caps'], float('nan')),
                'rv': rows(w['rv'], float('nan'))}
        for key in ('main', 'clk'):
            v = w[key]
            T._w[key] = None if v is None else (rows(v[0], 0.0), grow(v[1]), grow(v[2]))
    force = torch.zeros(E, dtype=torch.bool, device=dev)
    force[torch.as_tensor(tn, device=dev)] = True
    T._force_dirty = force
    T._timed = True
    T._gen = T.__dict__.get('_gen', 0) + 1
    # -- the CPU engine
    why = C.edit(e, T, t, lmap, tn, src, newp)
    if why:                                                 # (the next run rebuilds the timer)
        T._cpu = None
    elif in_sync:
        T._state_ver = T.__dict__.get('_state_ver', 0) + 1
        C.ver = T._state_ver
    return ''
