"""Static timing analysis on the GPU: the 'sta' capability (needs the 'timing' fact, see attar.timing).

Graph: a node per pin and transition (rise, fall); net arcs (driver -> sink) and cell arcs (one per Liberty timing arc
and instance). Clocks are ideal (pre-CTS): a register clock pin sees its clock's edge plus the ideal latency, slew 0
(set_clock_transition is not modeled). Arrivals are kept per tag (launching clock, launching edge), so setup and hold
use OpenSTA's edge relationships between any two clocks. Delays: NLDM tables, bilinear with linear extrapolation
(OpenSTA's rule), on the net's total load (no effective capacitance). Wires (parasitics='placement'): net capacitance
= Steiner length x c per um; a sink's delay is the Elmore delay of a wire of its Manhattan distance from the driver
(r, c per um: the RC script's set_wire_rc -signal), and its slew sqrt(s^2 + (ln 9 x delay)^2).
Not modeled: path exceptions (set_false_path, set_max_delay, ...), clock gating checks, recovery / removal, latches.
"""
import json
import math
import os
import warnings

import numpy as np
import torch

from . import metrics as M
from . import st
from .distill import D_IN, D_INOUT, D_OUT
from .timing import (S_NEG, S_NON, S_POS, T_CLEAR, T_COMB, T_FALL_EDGE, T_HOLD_F, T_HOLD_R, T_PRESET, T_RISE_EDGE,
                     T_SETUP_F, T_SETUP_R, T_TRISTATE)

warnings.filterwarnings('ignore', message='index_reduce')     # torch marks index_reduce beta; it is what we need
LN9 = math.log(9.0)
INF = float('inf')


def nldm(ix1, ix2, val, n, x, y):
    """Table lookup for E arcs: ix1, ix2 (E, K) axes, val (E, K, K), n (E, 2) sizes, x / y (E,) -> (E,).
    Bilinear inside, linear extrapolation outside (from the end segment), constant along a size-1 axis."""
    E, K = ix1.shape
    dev = x.device
    out = []
    idx = []
    for ax, v, nn in ((ix1, x, n[:, 0].long()), (ix2, y, n[:, 1].long())):
        if K > 2:
            j = torch.arange(1, K - 1, device=dev)
            i0 = ((v[:, None] >= ax[:, 1:K - 1]) & (j[None] < nn[:, None] - 1)).sum(1)
        else:
            i0 = torch.zeros(E, dtype=torch.long, device=dev)
        i1 = torch.minimum(i0 + 1, (nn - 1).clamp(min=0))
        a0 = ax.gather(1, i0[:, None])[:, 0]; a1 = ax.gather(1, i1[:, None])[:, 0]
        den = a1 - a0
        t = torch.where((i1 > i0) & (den != 0), (v - a0) / torch.where(den != 0, den, torch.ones_like(den)), torch.zeros_like(v))
        idx.append((i0, i1, t))
    (i0, i1, tx), (j0, j1, ty) = idx
    flat = val.reshape(E, K * K)
    g = lambda i, j: flat.gather(1, (i * K + j)[:, None])[:, 0]
    return (g(i0, j0) * (1 - tx) * (1 - ty) + g(i1, j0) * tx * (1 - ty) + g(i0, j1) * (1 - tx) * ty + g(i1, j1) * tx * ty)


def _relationships(clocks):
    """Setup and hold relationships between every (launch clock, edge) and (capture clock, edge), OpenSTA style:
    setup = the smallest positive capture - launch over the common period; hold = setup capture - capture period
    - launch, the largest over the launches. Times relative to the launch edge (ps). Shape (2C, 2C) each."""
    C = len(clocks)
    su = np.full((2 * C, 2 * C), np.nan); ho = np.full((2 * C, 2 * C), np.nan)
    for a, ca in enumerate(clocks):
        for b, cb in enumerate(clocks):
            Ta, Tb = ca['period'], cb['period']
            # common period (rounded to 1e-3 ps; at most 1000 launches)
            ia, ib = int(round(Ta * 1000)), int(round(Tb * 1000))
            L = ia * ib // math.gcd(ia, ib) if ia > 0 and ib > 0 else ia
            n_launch = max(1, min(1000, L // max(ia, 1)))
            for ea in (0, 1):
                for eb in (0, 1):
                    best_s, best_h = INF, -INF
                    for m in range(n_launch):
                        tl = ca['waveform'][ea] + m * Ta
                        tc0 = cb['waveform'][eb]
                        k = math.floor((tl - tc0) / Tb) + 1          # first capture edge strictly after tl
                        tc = tc0 + k * Tb
                        if tc - tl < 1e-6:
                            tc += Tb
                        s = tc - tl
                        if s < best_s - 1e-9:
                            best_s = s
                        h = tc - Tb - tl
                        if h > best_h:
                            best_h = h
                    su[2 * a + ea, 2 * b + eb] = best_s
                    ho[2 * a + ea, 2 * b + eb] = best_h
    return su, ho


STATE = ('valid', 'k0', 'k1', 'k2', 'k3', 'k4', 'p1', 'p2', 't0', 'dt', 'vo_delay', 'ub', 'drvr_slew', 'vth', 'vl', 'vh',
         'derate')


def arc_eval(ix1, ix2, val, tn, lib, orf, slew, ctot, c_far, rpi, thr, steps=None):
    """OpenSTA's dmp_ceff_elmore gate delay for a batch of cell arcs (attar.dmp.gate): gate delay, driver slew and the
    driver model each load's wire delay needs, packed as STATE columns. ctot: the driven net's total capacitance for
    this analysis (max or min pin caps), (c_far, rpi): its pi model's far capacitance and resistance (the near one is
    ctot - c_far). thr: the driver library's (output threshold, slew lower, slew upper, slew derate)."""
    from . import dmp
    tab = (ix1[lib, orf], ix2[lib, orf], val[lib, orf], tn[lib, orf],
           ix1[lib, 2 + orf], ix2[lib, 2 + orf], val[lib, 2 + orf], tn[lib, 2 + orf])
    c_far = torch.minimum(c_far, ctot)
    delay, dslew, _, stt = dmp.gate(tab, slew, ctot, (ctot - c_far).clamp(min=0), rpi, c_far,
                                    thr[:, 0], thr[:, 1], thr[:, 2], thr[:, 3], steps or dmp.EAGER)
    return delay, dslew, torch.stack([stt[k].to(delay.dtype) for k in STATE], 1)


def load_eval(state, elmore, rise, lthr, steps=None):
    """Wire delay and load slew of the loads of a batch of arcs (attar.dmp.load), at the loads' thresholds
    (DelayCalcBase::thresholdAdjust). state: the arcs' STATE rows, one per load; lthr: the loads' (input threshold,
    slew lower, slew upper, slew derate)."""
    from . import dmp
    st_ = {k: state[:, i] for i, k in enumerate(STATE)}
    st_['valid'] = st_['valid'] > 0.5
    w, sl = dmp.load(st_, elmore, steps or dmp.EAGER)
    return dmp.threshold_adjust(w, sl, rise, st_['vth'], st_['vl'], st_['vh'], st_['derate'],
                                lthr[:, 0], lthr[:, 1], lthr[:, 2], lthr[:, 3])


def compile_fn(fn):
    """torch.compile with dynamic shapes (Triton fuses a step into a few kernels); fn itself if unavailable."""
    try:
        import logging
        import triton  # noqa: F401
        import torch._inductor.config as icfg
        icfg.fx_graph_cache = True                       # reuse compiled kernels across processes
        logging.getLogger('torch.fx.experimental.symbolic_shapes').setLevel(logging.ERROR)
        return torch.compile(fn, dynamic=True)
    except Exception:  # noqa: BLE001
        return fn


TREE_BUCKETS = ((2, 2), (3, 3), (4, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256))
TREE_CHUNK = 4_000_000       # D^2 entries per batch (the tree's matrices are (2D)^2)


def tree_plan(ptr, deg, drv_local, chunk=TREE_CHUNK):
    """The placement-independent part of tree_elmore: per degree bucket and chunk, the nets and their pins with the
    driver first (padding: copies of the driver); the nets above the largest bucket. No host sync at run time."""
    dev = deg.device
    has = drv_local >= 0
    buckets, tri = [], []
    for dlo, dhi in TREE_BUCKETS:
        allnets = torch.nonzero(has & (deg >= dlo) & (deg <= dhi)).flatten()
        D = 1 << (dhi - 1).bit_length()                            # a power of two (padding: invalid pins)
        step = max(1, chunk // (D * D))
        k = torch.arange(D, device=dev)
        for c0 in range(0, len(allnets), step):
            nets = allnets[c0:c0 + step]
            dl = drv_local[nets][:, None]
            dg = deg[nets][:, None]
            loc = torch.where(k[None] == 0, dl, torch.where(k[None] <= dl, k[None] - 1, k[None]))
            loc = torch.where(k[None] < dg, loc, dl)
            real = (k[None] < dg).expand(len(nets), D)
            idx = ptr[nets][:, None] + loc
            ri = torch.nonzero(real.flatten()).flatten()           # static: no boolean indexing at run time
            buckets.append((D, nets, idx, real, torch.arange(len(nets), device=dev), idx.flatten()[ri], ri))
            tri.append((idx.contiguous(), real.to(torch.int8).contiguous(), nets.contiguous()))
    big = torch.nonzero(has & (deg > TREE_BUCKETS[-1][1])).flatten()
    plan = {'buckets': buckets, 'tri': tri, 'big': big if len(big) else None}
    if len(big):
        net_of = torch.repeat_interleave(torch.arange(len(deg), device=dev), deg)
        plan['pin_big'] = torch.isin(net_of, big)
        plan['net_of'] = net_of
        plan['drv_pin'] = (ptr[net_of] + drv_local[net_of]).clamp(0, int(ptr[-1]) - 1)
    return plan


SIDE_D = 32                  # tree buckets from this degree run on a side stream (see tree_elmore)
_SIDE = {}


def _side_stream(dev):
    if dev not in _SIDE:
        _SIDE[dev] = torch.cuda.Stream(device=dev)
    return _SIDE[dev]


def tree_elmore(p, plan, n_nets, cap, r, c, length, dbu, rcut=0.0):
    """Elmore delay from each net's driver to every pin and the driver's pi model, on OpenROAD's estimate-parasitics
    tree (attar.steiner.pd_tree: Prim-Dijkstra alpha 0.3 on nearest neighbors, steinerized; dbu coordinates), each
    branch a pi segment of wire RC r, c per um, every pin behind rcut (OpenROAD's average cut resistance), the driver
    included (EstimateParasitics::estimateWireParasiticSteiner, parasiticNodeConnectPins). cap: pin capacitance (P,).
    Nets above the largest bucket: lumped on length (r x distance x half the net's C).
    cap may hold V variants (P, V) (rise / fall, max / min pin capacitances: OpenSTA reduces per rise / fall and
    min / max). Returns per pin and variant the Elmore delay (P, V) in ps, per net and variant the pi model (E, V, 3)
    near C / R / far C, and per net the tree's wire length in um (length where no tree was built)."""
    from .steiner import pd_tree
    rc_bucket = None
    if p.is_cuda:
        try:
            from .steiner_triton import rc_bucket
        except Exception:  # noqa: BLE001  (no Triton: the PyTorch reference)
            pass
    dev = p.device
    if cap.dim() == 1:
        cap = cap[:, None]
    V = cap.shape[1]
    out = torch.zeros(len(p), V, device=dev)
    pi = torch.zeros(n_nets, V, 3, device=dev)
    tlen = length.clone()
    if rc_bucket is not None:                     # one Triton launch per bucket: tree, moments, pi, scattered outputs
        pc, cc = p.float().contiguous(), cap.float().contiguous()
        # the high-degree buckets (few nets, one long program each) on a side stream, beside the small ones
        cur = torch.cuda.current_stream(dev)
        side = _side_stream(dev)
        side.wait_stream(cur)
        for idx, real, nets in plan['tri']:
            if len(nets):
                with torch.cuda.stream(side if idx.shape[1] >= SIDE_D else cur):
                    rc_bucket(pc, idx, real, cc, nets, out, pi, tlen, 0.3, dbu, r, c, rcut)
        cur.wait_stream(side)
    for D, nets, idx, real, rows, rpins, ri in (plan['buckets'] if rc_bucket is None else ()):
        n = len(nets)
        N = 2 * D
        q = torch.round(p[idx] * dbu)
        xy, parent, alive = pd_tree(q, real)
        hasp = parent >= 0
        par = parent.clamp(min=0)
        el = torch.where(hasp, (xy - torch.gather(xy, 1, par[:, :, None].expand(-1, -1, 2))).abs().sum(-1), torch.zeros_like(xy[..., 0])) / dbu
        tlen[nets] = el.sum(1)
        V = cap.shape[1]
        cp = torch.cat([torch.where(real[:, :, None], cap[idx], torch.zeros_like(cap[idx])), torch.zeros(n, N - D, V, device=dev)], 1)
        # the tree as matrices: parent one-hot Pm (child row -> parent column), ancestor-or-self M by doubling
        Pm = torch.nn.functional.one_hot(par, N).float() * hasp[:, :, None].float()
        M = torch.eye(N, device=dev).expand(n, N, N) + Pm
        reach = 1
        while reach < N:
            M = (torch.bmm(M, M) > 0).float(); reach *= 2
        # point capacitance: half of every incident wire (pi segments); pins behind rcut at their points
        half = c * el / 2
        cw = (half + torch.bmm(Pm.transpose(1, 2), half[:, :, None])[:, :, 0])[:, :, None]   # (n, N, 1)
        cn = cw + cp                                                       # (n, N, V) per pin-cap variant
        Re = (r * el)[:, :, None]                                          # resistance of each point's parent wire
        sub = lambda x: torch.bmm(M.transpose(1, 2), x)                    # subtree sums
        path = lambda x: torch.bmm(M, x)                                   # root-to-point sums
        y1 = cn.sum(1)                                                     # (n, V)
        m1 = rcut * y1[:, None, :] + path(Re * sub(cn))                    # Elmore at the points
        m1p = m1 + rcut * cp                                               # ... at the pins
        qq = cw * m1 + cp * m1p
        m2 = rcut * qq.sum(1, keepdim=True) + path(Re * sub(qq))
        m2p = m2 + rcut * cp * m1p
        # the driver's pi model from the driving point admittance moments (O'Brien-Savarino), per variant
        y2 = -qq.sum(1); y3 = (cw * m2 + cp * m2p).sum(1)
        ok = (y3 > 1e-12) & (y2 < -1e-12)
        c2 = torch.where(ok, y2 * y2 / torch.where(ok, y3, torch.ones_like(y3)), torch.zeros_like(y1))
        rr = torch.where(ok, -y3 * y3 / torch.where(ok, y2 ** 3, -torch.ones_like(y2)), torch.zeros_like(y1))
        c2 = torch.minimum(c2, y1)
        pi[nets] = torch.stack([y1 - c2, rr, c2], -1)
        out[rpins] = m1p[:, :D].reshape(n * D, V)[ri]
    if plan['big'] is not None:
        big, net_of = plan['big'], plan['net_of']
        dd = (p - p[plan['drv_pin']]).abs().sum(1)
        ctot = torch.zeros(n_nets, V, device=dev).index_add_(0, net_of, cap) + c * length[:, None]
        out = torch.where(plan['pin_big'][:, None], r * dd[:, None] * ctot[net_of] / 2, out)
        pi[big] = torch.stack([ctot[big], torch.zeros_like(ctot[big]), torch.zeros_like(ctot[big])], -1)
    return out, pi, tlen


class Timer:
    """Build once per design (graph, levels, constraints); run() times a placement."""

    def __init__(self, d, parasitics='placement'):
        d.require('sta')
        self.d = d
        self.dev = dev = d.device
        self.parasitics = parasitics
        t, _ = st.load(os.path.join(d.path, 'timing.safetensors'))
        self.info = info = json.load(open(os.path.join(d.path, 'timing.json')))
        a = d.raw
        T = lambda x, dt=None: torch.as_tensor(np.asarray(x), device=dev, dtype=dt)
        p_cell = a['p_cell'].astype(np.int64); p_mpin = a['p_mpin'].astype(np.int64)
        P = len(p_cell); self.P = P
        n_mp = len(t['mp_cap'])
        pdir = a['p_dir'].astype(np.int64) if 'p_dir' in a else np.full(P, 3)
        p_net = d.p_net.cpu().numpy(); n_ptr = a['n_ptr'].astype(np.int64)
        c_master = a['c_master'].astype(np.int64)
        m_ptr = a['m_pin_ptr'].astype(np.int64)

        # pin capacitance (rise, fall): Liberty for cell pins, set_load for output ports
        cap = np.zeros((P, 2), np.float32)
        ok = p_mpin >= 0
        cap[ok] = t['mp_cap'][p_mpin[ok]]
        cap_min = cap.copy()
        if 'mp_cap_min' in t:
            cap_min[ok] = t['mp_cap_min'][p_mpin[ok]]
        for c, v in info.get('load', {}).items():
            cap[p_cell == int(c)] = v; cap_min[p_cell == int(c)] = v
        self.pin_cap, self.pin_cap_min = T(cap), T(cap_min)
        # macro clock pins: Liberty min / max_clock_tree_path; OpenSTA's ideal-clock capture time is the clock edge minus
        # the min value for setup checks and minus the max value for hold checks (launch unchanged)
        ctp = np.zeros((P, 2, 2), np.float32)
        if 'mp_ctp' in t:
            ctp[ok] = t['mp_ctp'][p_mpin[ok]]
        self.pin_ctp = T(ctp)

        # cell arcs: one per Liberty arc and instance of its master
        keys = np.where(ok, p_cell * n_mp + p_mpin, -1)
        order = np.argsort(keys); skeys = keys[order]
        def pin_of(cell, mp):
            k = cell * n_mp + mp
            i = np.searchsorted(skeys, k)
            i = np.minimum(i, len(skeys) - 1)
            return np.where(skeys[i] == k, order[i], -1)
        am = t['a_master'].astype(np.int64)
        cm_order = np.argsort(c_master, kind='stable')
        cnt = np.bincount(c_master[c_master >= 0], minlength=len(m_ptr) - 1)
        start = np.concatenate([[0], np.cumsum(cnt)])[:-1] + int((c_master < 0).sum())
        reps = cnt[am]
        lib_id = np.repeat(np.arange(len(am)), reps)
        within = np.arange(len(lib_id)) - np.repeat(np.cumsum(reps) - reps, reps)
        icell = cm_order[start[am[lib_id]] + within]
        fr = pin_of(icell, t['a_from'].astype(np.int64)[lib_id]); to = pin_of(icell, t['a_to'].astype(np.int64)[lib_id])
        keep = (fr >= 0) & (to >= 0)
        lib_id, fr, to = lib_id[keep], fr[keep], to[keep]
        atype = t['a_type'][lib_id].astype(np.int64); asense = t['a_sense'][lib_id].astype(np.int64)
        # constants from tie cells: arcs into a constant output, or from an input that cannot change it, are off
        off, asense = self._constants(d, info, t, p_cell, p_mpin, pdir, p_net, n_ptr, c_master, fr, to, atype, lib_id,
                                      asense)
        keep = ~off
        lib_id, fr, to, atype, asense = lib_id[keep], fr[keep], to[keep], atype[keep], asense[keep]

        # nets: one driver, its sinks
        drv = np.full(len(n_ptr) - 1, -1, np.int64)
        is_drv = pdir == D_OUT
        nd = np.bincount(p_net[is_drv], minlength=len(n_ptr) - 1)
        one = nd == 1
        drv[p_net[is_drv & one[p_net]]] = np.nonzero(is_drv & one[p_net])[0]
        sink = (pdir == D_IN) | (pdir == D_INOUT) | (pdir == 3)
        sink &= ~is_drv & (drv[p_net] >= 0)
        n_to = np.nonzero(sink)[0]; n_fr = drv[p_net[n_to]]
        self.net_driver = drv

        # clock network: from the clock source ports through nets and combinational arcs, up to register clock pins
        clocks = list(info['clocks'].items())
        self.clock_names = [k for k, _ in clocks]
        C = len(clocks)
        seq_clk = np.zeros(P, bool)
        seq_clk[fr[np.isin(atype, (T_RISE_EDGE, T_FALL_EDGE, T_SETUP_R, T_HOLD_R, T_SETUP_F, T_HOLD_F))]] = True
        clk_of = np.full(P, -1, np.int64); par = np.zeros(P, np.int64)
        comb = np.isin(atype, (T_COMB, T_TRISTATE))
        e_fr = np.concatenate([n_fr, fr[comb]]); e_to = np.concatenate([n_to, to[comb]])
        e_inv = np.concatenate([np.zeros(len(n_fr), np.int64), (asense[comb] == S_NEG).astype(np.int64)])
        eo = np.argsort(e_fr, kind='stable'); e_fr, e_to, e_inv = e_fr[eo], e_to[eo], e_inv[eo]
        e_ptr = np.searchsorted(e_fr, np.arange(P + 1))
        frontier = []
        for ci, (_, c) in enumerate(clocks):
            for cell in c['sources']:
                for p in np.nonzero(p_cell == cell)[0]:
                    if clk_of[p] < 0:
                        clk_of[p] = ci; frontier.append(p)
        frontier = np.asarray(frontier, np.int64)
        while len(frontier):
            f = frontier[~seq_clk[frontier]]
            if not len(f):
                break
            lo, hi = e_ptr[f], e_ptr[f + 1]
            rep = hi - lo
            src = np.repeat(f, rep)
            ei = np.repeat(lo - np.cumsum(rep) + rep, rep) + np.arange(rep.sum()) if rep.sum() else np.zeros(0, np.int64)
            nxt = e_to[ei]
            new = clk_of[nxt] < 0
            nxt, src, ei = nxt[new], src[new], ei[new]
            nxt, first = np.unique(nxt, return_index=True)
            clk_of[nxt] = clk_of[src[first]]; par[nxt] = par[src[first]] ^ e_inv[ei[first]]
            frontier = nxt
        clk_net = clk_of >= 0
        self.clk_of, self.par = clk_of, par

        # data graph: net and combinational arcs not into the clock network; launch arcs from clock pins
        nk = ~clk_net[n_to] & ~clk_net[n_fr]
        ck = comb & ~clk_net[to] & ~clk_net[fr]
        launch = np.isin(atype, (T_RISE_EDGE, T_FALL_EDGE)) & (clk_of[fr] >= 0) & ~clk_net[to]
        self.n_fr, self.n_to = T(n_fr[nk]), T(n_to[nk])
        self.n_net = T(p_net[n_to[nk]])
        sel = lambda m: (T(lib_id[m]), T(fr[m]), T(to[m]), T(asense[m]), T(atype[m]))
        self.c_lib, self.c_fr, self.c_to, self.c_sense, self.c_type = sel(ck)
        self.l_lib, self.l_fr, self.l_to, self.l_sense, self.l_type = sel(launch)
        chk = np.isin(atype, (T_SETUP_R, T_HOLD_R, T_SETUP_F, T_HOLD_F)) & (clk_of[fr] >= 0)
        self.k_lib, self.k_fr, self.k_to, self.k_sense, self.k_type = sel(chk)

        # levels (Kahn) over net + combinational arcs
        g_fr = np.concatenate([n_fr[nk], fr[ck]]); g_to = np.concatenate([n_to[nk], to[ck]])
        lev, dropped = self._levelize(P, g_fr, g_to)
        self.n_cycle_edges = dropped
        self.level = lev
        nl = lev[n_to[nk]]; cl = lev[to[ck]]
        self.n_lev = T(nl); self.c_lev = T(cl)
        self.max_level = int(max(nl.max(initial=0), cl.max(initial=0)))
        self.n_by_lev = self._group(nl); self.c_by_lev = self._group(cl)

        # library tables on the device
        self.t_ix1, self.t_ix2 = T(t['a_ix1']), T(t['a_ix2'])
        # thresholds: drivers per arc and output transition; loads per pin (ports: the first library read)
        dt_ = info.get('default_thresholds') or {'input': (.5, .5), 'slew_lower': (.2, .2), 'slew_upper': (.8, .8),
                                                  'slew_derate': 1.0}
        dflt = np.array([[dt_['input'][i], dt_['slew_lower'][i], dt_['slew_upper'][i], dt_['slew_derate']] for i in (0, 1)],
                        np.float32)
        self.t_thr = T(t['a_thr']) if 'a_thr' in t else T(np.tile(dflt, (len(t['a_master']), 1, 1)))
        sthr = np.tile(dflt, (P, 1, 1))
        if 'mp_thr' in t:
            sthr[ok] = t['mp_thr'][p_mpin[ok]]
        self.sink_thr = T(sthr)
        self.t_val, self.t_n = T(t['a_val']), T(t['a_n'].astype(np.int64))

        # clocks and constraints
        self.clocks = [c for _, c in clocks]
        su, ho = _relationships(self.clocks) if C else (np.zeros((0, 0)), np.zeros((0, 0)))
        excl = np.zeros((C, C), bool)
        cidx = {k: i for i, k in enumerate(self.clock_names)}
        for g in info.get('clock_groups', []):
            gs = [[cidx[n] for n in grp if n in cidx] for grp in g['groups']]
            for i, ga in enumerate(gs):
                for j, gb in enumerate(gs):
                    if i != j:
                        for x in ga:
                            for y in gb:
                                excl[x, y] = True
        su_t = np.repeat(np.repeat(excl, 2, 0), 2, 1)
        su[su_t] = np.nan; ho[su_t] = np.nan
        self.rel_setup, self.rel_hold = T(su, torch.float32), T(ho, torch.float32)
        self.wf = T([c['waveform'] for c in self.clocks] or np.zeros((0, 2)), torch.float32)
        self.lat = T([c.get('latency', 0.0) for c in self.clocks] or np.zeros(0), torch.float32)
        self.unc_s = T([c.get('uncertainty_setup', c.get('uncertainty', 0.0)) for c in self.clocks] or np.zeros(0), torch.float32)
        self.unc_h = T([c.get('uncertainty_hold', c.get('uncertainty', 0.0)) for c in self.clocks] or np.zeros(0), torch.float32)
        io_mask = np.isin(p_cell, a['io_cell']) if 'io_cell' in a else np.zeros(len(p_cell), bool)
        port_pin = dict(zip(p_cell[io_mask].tolist(), np.nonzero(io_mask)[0].tolist()))
        self.in_delay = [(port_pin[e['cell']], cidx.get(e['clock'], -1), e) for e in info.get('input_delay', [])
                         if e['cell'] in port_pin and e['clock'] in cidx and not clk_net[port_pin[e['cell']]]]
        self.out_delay = [(port_pin[e['cell']], cidx.get(e['clock'], -1), e) for e in info.get('output_delay', [])
                          if e['cell'] in port_pin and e['clock'] in cidx]
        self.in_slew = {port_pin[int(c)]: v for c, v in info.get('input_transition', {}).items() if int(c) in port_pin}
        rc = info.get('rc') or {}
        self.r_sig, self.c_sig = (rc.get('signal') or [0.0, 0.0])
        self.cut_res = float(rc.get('cut_res', 0.0))
        self.p_cell_np = p_cell
        self.pi = None
        self.pin_net = d.p_net
        drv_t = torch.as_tensor(drv, device=dev)
        self.drv_nets = torch.nonzero(drv_t >= 0).flatten(); self.drv_pins = drv_t[self.drv_nets]
        self._prepare_static()
        self._wire_setup()
        self._graph = None
        self.use_graph = self.dev.type == 'cuda'
        from . import dmp
        self.steps = dmp.EAGER
        self.triton = None
        self.fused = True                       # with Triton: one gate + one load kernel per level (False: per-op path)
        if self.dev.type == 'cuda':
            try:
                from . import dmp_triton
                self.triton = dmp_triton
                self.t_n32 = self.t_n.to(torch.int32).contiguous()
                self._prepare_triton()
            except Exception:  # noqa: BLE001  (no Triton: the PyTorch reference path)
                self.triton = None
        self._mp_names = None; self._sink_drv = None

    def _constants(self, d, info, t, p_cell, p_mpin, pdir, p_net, n_ptr, c_master, fr, to, atype, lib_id, asense):
        """Constant propagation (OpenSTA's sim): nets driven by tie cells, then every output whose function is
        constant given its constant inputs. Then on the instances with constant inputs, OpenSTA's Sim::findDisabledEdges:
        an arc is off when the output's function, under the constants, does not depend on its input (functionSense
        none); its sense becomes the function's (positive / negative unate) where that is narrower than the arc's; a
        conditional arc is off when its `when` is 0 under the constants, the unconditional (default) arc between the
        same pins when another one's `when` is 1 (Sim::isDisabledCond). Returns (arcs to disable, arc senses).
        Sets self.const_net (net -> 0 / 1 / -1)."""
        from itertools import product
        from .liberty import parse_function
        E = len(n_ptr) - 1
        self.const_net = cn = np.full(E, -1, np.int64)
        off = np.zeros(len(fr), bool)
        asense = asense.copy()
        fdefs = info.get('functions') or {}
        if not fdefs:
            return off, asense
        names = [x for ps in d.names['master_pins'] for x in ps]
        funcs = {int(m): {pn: parse_function(f) for pn, f in fs.items()} for m, fs in fdefs.items()}
        o = np.argsort(p_cell, kind='stable'); cptr = np.searchsorted(p_cell[o], np.arange(len(c_master) + 1))
        pins_of = lambda c: o[cptr[c]:cptr[c + 1]]
        def evaluate(c):
            """Constant outputs of instance c: {pin: value}, and {input pin name: its constant}."""
            pins = {names[p_mpin[p]]: p for p in pins_of(c) if p_mpin[p] >= 0}
            known = {n: int(cn[p_net[p]]) for n, p in pins.items() if cn[p_net[p]] >= 0 and pdir[p] != D_OUT}
            out = {}
            for pn, (vs, f) in funcs.get(int(c_master[c]), {}).items():
                if pn not in pins:
                    continue
                unk = [v for v in vs if v not in known]
                if len(unk) > 12:
                    continue
                vals = {f(dict(known, **dict(zip(unk, bits)))) for bits in product((0, 1), repeat=len(unk))}
                if len(vals) == 1:
                    out[pins[pn]] = vals.pop()
            return out, known, pins
        # seeds: tie cells (an output function without variables)
        queue = []
        for m, fs in funcs.items():
            if all(not vs for vs, _ in fs.values()):
                for c in np.nonzero(c_master == m)[0]:
                    res, _, _ = evaluate(c)
                    for p, v in res.items():
                        if cn[p_net[p]] < 0:
                            cn[p_net[p]] = v; queue.append(p_net[p])
        touched = set()
        while queue:
            n = queue.pop()
            for p in range(n_ptr[n], n_ptr[n + 1]):
                if pdir[p] == D_OUT or p_mpin[p] < 0:
                    continue
                c = int(p_cell[p]); touched.add(c)
                res, _, _ = evaluate(c)
                for q, v in res.items():
                    if cn[p_net[q]] < 0:
                        cn[p_net[q]] = v; queue.append(p_net[q])
        if not touched:
            return off, asense
        # conditional arcs: `when` per Liberty arc, and per (master, from, to) the conditions of its arcs
        whens = {int(k): parse_function(v) for k, v in (info.get('arc_when') or {}).items()}
        conds = {}
        for k, w in whens.items():
            conds.setdefault((int(t['a_master'][k]), int(t['a_from'][k]), int(t['a_to'][k])), []).append(w)
        def value(expr, known):
            """An expression under the constants: 0 / 1, or -1 when it depends on a non-constant pin."""
            vs, f = expr
            unk = [v for v in vs if v not in known]
            if len(unk) > 12:
                return -1
            vals = {f(dict(known, **dict(zip(unk, bits)))) for bits in product((0, 1), repeat=len(unk))}
            return vals.pop() if len(vals) == 1 else -1
        # arcs of the touched instances
        comb = np.isin(atype, (T_COMB, T_TRISTATE))
        acell = p_cell[fr]
        cand = np.nonzero(comb & np.isin(acell, list(touched)))[0]
        cache = {}
        n_sense = 0
        for a in cand:
            c = int(acell[a])
            if c not in cache:
                cache[c] = evaluate(c)
            res, known, pins = cache[c]
            y, x = to[a], fr[a]
            if cn[p_net[y]] >= 0 or y in res:
                off[a] = True; continue
            xn = names[p_mpin[x]]
            if xn in known:
                off[a] = True; continue
            fn = funcs.get(int(c_master[c]), {}).get(names[p_mpin[y]])
            if fn is not None:
                vs, f = fn
                unk = [v for v in vs if v not in known and v != xn]
                if len(unk) <= 12:
                    inc = dec = True
                    for bits in product((0, 1), repeat=len(unk)):
                        e = dict(known, **dict(zip(unk, bits)))
                        f0, f1 = f(dict(e, **{xn: 0})), f(dict(e, **{xn: 1}))
                        inc &= f1 >= f0; dec &= f1 <= f0
                    if inc and dec:                                # functionSense none: no dependence
                        off[a] = True; continue
                    if inc != dec:                                 # positive / negative unate under the constants
                        s_sim = S_POS if inc else S_NEG
                        if asense[a] == S_NON:
                            asense[a] = s_sim; n_sense += 1
                        elif asense[a] != s_sim:                   # the arc's transitions are all excluded
                            off[a] = True; continue
            # Sim::isDisabledCond
            k = int(lib_id[a])
            if k in whens:
                off[a] = value(whens[k], known) == 0
            else:
                key = (int(t['a_master'][k]), int(t['a_from'][k]), int(t['a_to'][k]))
                off[a] = any(value(w, known) == 1 for w in conds.get(key, ()))
        self.n_const_nets = int((cn >= 0).sum()); self.n_disabled_arcs = int(off.sum()); self.n_sim_sense = n_sense
        return off, asense

    @staticmethod
    def _levelize(P, fr, to):
        """Longest-path levels of a DAG (Kahn by layers). Edges inside cycles are dropped (count returned)."""
        dropped = 0
        while True:
            indeg = np.bincount(to, minlength=P)
            o = np.argsort(fr, kind='stable'); f2, t2 = fr[o], to[o]
            ptr = np.searchsorted(f2, np.arange(P + 1))
            lev = np.zeros(P, np.int64)
            done = np.zeros(P, bool)
            cur = np.nonzero(indeg == 0)[0]
            k = 0
            while len(cur):
                done[cur] = True; lev[cur] = k
                lo, hi = ptr[cur], ptr[cur + 1]
                rep = hi - lo
                if rep.sum() == 0:
                    break
                ei = np.repeat(lo - np.cumsum(rep) + rep, rep) + np.arange(rep.sum())
                nx = t2[ei]
                np.subtract.at(indeg, nx, 1)
                cand = np.unique(nx)
                cur = cand[indeg[cand] == 0]
                k += 1
            if done.all():
                return lev, dropped
            bad = ~done[to] & ~done[fr]
            dropped += int(bad.sum())
            fr, to = fr[~bad], to[~bad]

    def _group(self, lev):
        if not len(lev):
            return [None] * (self.max_level + 1)
        o = np.argsort(lev, kind='stable')
        b = np.searchsorted(lev[o], np.arange(self.max_level + 2))
        return [torch.as_tensor(o[b[k]:b[k + 1]], device=self.dev) for k in range(self.max_level + 1)]

    # -- wires ------------------------------------------------------------------------------------------------------
    def _wire_setup(self):
        """Placement-independent parts of the wire model: the Evaluator's pin offsets, Steiner (RMST) buckets and the
        tree plan, so that wires() has no host sync (CUDA-graph capturable)."""
        from .evaluator import Evaluator
        ev = self.ev = Evaluator(self.d)
        drv = torch.as_tensor(self.net_driver, device=self.dev)
        drv_local = torch.where(drv >= 0, drv - ev.ptr[:-1], torch.full_like(drv, -1))
        self.plan = tree_plan(ev.ptr, ev.deg, drv_local)
        self.big_steiner = ev.deg > M.RMST_MAX_DEGREE

    def steiner(self, p):
        """Wire length of the nets without an estimate-parasitics tree (above the largest bucket, or no driver):
        min(single trunk, star), HPWL up to 3 pins. tree_elmore replaces it on every net it builds a tree for."""
        ev = self.ev
        lo, hi = M.net_bbox(p, ev.net, ev.E)
        s = torch.minimum(M.stt(p, ev.net, ev.deg, ev.ptr, ev.E, lo, hi), M.star(p, ev.net, ev.deg, ev.E))
        L = torch.where(ev.deg <= 3, (hi - lo).sum(1), s)
        return torch.where(ev.deg >= 2, L, torch.zeros_like(L))

    def wires(self, pos=None):
        """Placement parasitics: per net the wire capacitance, per net arc the Elmore delay (ps)."""
        p = self.ev._pins(pos)
        L = self.steiner(p)
        caps = torch.cat([self.pin_cap, self.pin_cap_min], 1)            # rise / fall max, rise / fall min
        elm_pin, self.pi, L = tree_elmore(p, self.plan, self.ev.E, caps, self.r_sig, self.c_sig, L,
                                          self.d.dbu, rcut=self.cut_res)
        return L * self.c_sig, elm_pin[self.n_to]                       # (A, 4): Elmore per variant

    def _lookup(self, lib, slot, x, y):
        return nldm(self.t_ix1[lib, slot], self.t_ix2[lib, slot], self.t_val[lib, slot], self.t_n[lib, slot], x, y)

    # -- static preparation (independent of the placement) -----------------------------------------------------------
    def _prepare_static(self):
        dev, P = self.dev, self.P
        C = len(self.clocks)
        # set_max_delay -from / -to: one extra tag each (paths from its startpoints, checked at its endpoints only)
        self.ex = [e for e in self.info.get('path_delays', []) if e['kind'] == 'max' and e['from'] and e['to']]
        X = len(self.ex)
        self.NT = NT = max(1, 2 * C + X)
        cell_of = self.p_cell_np
        ex_from = [set(e['from']) for e in self.ex]; ex_to = [set(e['to']) for e in self.ex]
        T = lambda x, dt=torch.long: torch.as_tensor(np.asarray(x), device=dev, dtype=dt)
        # input ports: arrival entries (flat index into (P, 2, NT)) and slews
        ix_max, v_max, ix_min, v_min, sl_p, sl_v = [], [], [], [], [], []
        wf = self.wf.cpu().numpy(); lat = self.lat.cpu().numpy()
        for p, ci, e in self.in_delay:
            edge = 1 if e['clock_fall'] else 0
            base = float(wf[ci, edge]) + (0.0 if e['network_latency_included'] else float(lat[ci]))
            rfs = [0, 1] if not (e['rise'] or e['fall']) else ([0] if e['rise'] else [1])
            for rf in rfs:
                k = (p * 2 + rf) * NT + 2 * ci + edge
                if not e['min']:
                    ix_max.append(k); v_max.append(base + e['value'])
                if not e['max']:
                    ix_min.append(k); v_min.append(base + e['value'])
            sl_p.append(p); sl_v.append(self.in_slew.get(p, 0.0))
        self.in_ix_max, self.in_v_max = T(ix_max), T(v_max, torch.float32)
        self.in_ix_min, self.in_v_min = T(ix_min), T(v_min, torch.float32)
        self.in_sl_p, self.in_sl_v = T(sl_p), T(sl_v, torch.float32)
        if NT > 2 * C:                                  # exception tags: no clock relationship (NaN: not checked)
            pad = lambda m: torch.nn.functional.pad(m, (0, NT - m.shape[1], 0, NT - m.shape[0]), value=float('nan'))
            self.rel_setup, self.rel_hold = pad(self.rel_setup), pad(self.rel_hold)
        # exception startpoints: input ports (arrival = their input delay, else 0)
        in_d = {}
        for p, ci, e in self.in_delay:
            if not e['min']:
                in_d[p] = max(in_d.get(p, -INF), e['value'])
        ex_ix, ex_v = [], []
        tp = np.nonzero(self.d.raw['p_mpin'] < 0)[0]
        port_pins = dict(zip(cell_of[tp].tolist(), tp.tolist()))
        for i, e in enumerate(self.ex):
            for c in ex_from[i]:
                p = port_pins.get(c)
                if p is not None and self.clk_of[p] < 0:
                    for rf in (0, 1):
                        ex_ix.append((p * 2 + rf) * NT + 2 * C + i); ex_v.append(in_d.get(p, 0.0))
        self.in_ix_max = torch.cat([self.in_ix_max, T(ex_ix)]); self.in_v_max = torch.cat([self.in_v_max, T(ex_v, torch.float32)])
        tn = self.t_n.cpu().numpy()
        # launch arcs: (arc, output transition) where the table exists; their tag and clock arrival
        clk_of, par = torch.as_tensor(self.clk_of, device=dev), torch.as_tensor(self.par, device=dev)
        if len(self.l_lib):
            ll = self.l_lib.cpu().numpy()
            a_i, o_i = [], []
            for orf in (0, 1):
                m = np.nonzero(tn[ll, orf, 0] > 0)[0]
                a_i.append(m); o_i.append(np.full(len(m), orf))
            a_i, o_i = T(np.concatenate(a_i)), T(np.concatenate(o_i))
            ck = clk_of[self.l_fr[a_i]]
            edge = (self.l_type[a_i] == T_FALL_EDGE).long() ^ par[self.l_fr[a_i]]
            lc = [(self.l_lib[a_i], self.l_to[a_i], o_i, 2 * ck + edge, self.wf[ck, edge] + self.lat[ck])]
            lcell = cell_of[self.l_fr[a_i].cpu().numpy()]
            for i, e in enumerate(self.ex):
                m = T(np.nonzero(np.isin(lcell, list(ex_from[i])))[0])
                if len(m):
                    t0 = self.wf[ck[m], edge[m]] + (0.0 if e['ignore_clock_latency'] else self.lat[ck[m]])
                    lc.append((self.l_lib[a_i][m], self.l_to[a_i][m], o_i[m], torch.full_like(m, 2 * C + i), t0))
            self.lc = tuple(torch.cat([x[j] for x in lc]) for j in range(5))
        else:
            self.lc = None
        # data net arcs grouped by driver: a driver's arcs expand to its loads (OpenSTA: gate delay + wire delay per arc)
        nf_np = self.n_fr.cpu().numpy()
        na_o = np.argsort(nf_np, kind='stable'); na_ptr = np.searchsorted(nf_np[na_o], np.arange(P + 1))

        def expand(drv):
            """For arcs into driver pins drv (numpy): (arc index, net arc index) for every load of the driver."""
            lo, hi = na_ptr[drv], na_ptr[drv + 1]
            rep_ = hi - lo
            ai = np.repeat(np.arange(len(drv)), rep_)
            ni = na_o[np.repeat(lo - np.cumsum(rep_) + rep_, rep_) + np.arange(rep_.sum())] if rep_.sum() else np.zeros(0, np.int64)
            return T(ai), T(ni)
        # nets driven by input ports (no gate: DelayCalcBase::dspfWireDelaySlew)
        port_drv = self.d.raw['p_mpin'][nf_np] < 0
        self.pa = T(np.nonzero(port_drv)[0])
        # cell arcs per level: (arc, output transition, input transition) allowed by the sense, with a table
        cl, cs, clev = self.c_lib.cpu().numpy(), self.c_sense.cpu().numpy(), self.c_lev.cpu().numpy()
        cto = self.c_to.cpu().numpy()
        combos = []
        for orf in (0, 1):
            for irf in (0, 1):
                ok = (tn[cl, orf, 0] > 0) & ((cs == S_NON) | ((cs == S_POS) & (irf == orf)) | ((cs == S_NEG) & (irf != orf)))
                m = np.nonzero(ok)[0]
                combos.append(np.stack([m, np.full(len(m), orf), np.full(len(m), irf)], 1))
        cb = np.concatenate(combos) if combos else np.zeros((0, 3), np.int64)
        o = np.argsort(clev[cb[:, 0]], kind='stable'); cb = cb[o]
        b = np.searchsorted(clev[cb[:, 0]], np.arange(self.max_level + 2))
        self.cb_lev = []
        for k in range(self.max_level + 1):
            x = cb[b[k]:b[k + 1]]
            if len(x):
                a_ = T(x[:, 0])
                ea, en = expand(cto[x[:, 0]])
                self.cb_lev.append((self.c_lib[a_], self.c_fr[a_], self.c_to[a_], T(x[:, 1]), T(x[:, 2]), ea, en))
            else:
                self.cb_lev.append(None)
        if self.lc is not None:
            self.lc_ex = expand(self.lc[1].cpu().numpy())
        # register checks
        if len(self.k_lib):
            ck = clk_of[self.k_fr]
            fall_chk = (self.k_type == T_SETUP_F) | (self.k_type == T_HOLD_F)
            self.k_ctag = 2 * ck + (fall_chk.long() ^ par[self.k_fr])
            self.k_setup = (self.k_type == T_SETUP_R) | (self.k_type == T_SETUP_F)
            self.k_ck = ck
            crf = fall_chk.long()                          # the clock pin's transition at the check
            self.k_ctp_min = self.pin_ctp[self.k_fr, 0, crf]; self.k_ctp_max = self.pin_ctp[self.k_fr, 1, crf]
            kcell = cell_of[self.k_to.cpu().numpy()]; ks = self.k_setup.cpu().numpy()
            ea, et, er = [], [], []
            lat_k = self.lat[ck].cpu().numpy() if C else np.zeros(len(kcell))
            for i, e in enumerate(self.ex):
                m = np.nonzero(np.isin(kcell, list(ex_to[i])) & ks)[0]
                ea.append(m); et.append(np.full(len(m), 2 * C + i))
                er.append(e['value'] + (np.zeros(len(m)) if e['ignore_clock_latency'] else lat_k[m]))
            self.kx = (T(np.concatenate(ea)) if ea else T([]), T(np.concatenate(et)) if et else T([]),
                       T(np.concatenate(er)) if er else T([], torch.float32))
            self.kx = (self.kx[0], self.kx[1], self.kx[2].float())
        # output ports: (pin, transition, capture tag, offset) for setup and for hold
        su, ho = [], []
        unc_s, unc_h = self.unc_s.cpu().numpy(), self.unc_h.cpu().numpy()
        for p, ci, e in self.out_delay:
            edge = 1 if e['clock_fall'] else 0
            l_ = 0.0 if e['network_latency_included'] else float(lat[ci])
            rfs = [0, 1] if not (e['rise'] or e['fall']) else ([0] if e['rise'] else [1])
            for rf in rfs:
                if not e['min']:
                    su.append((p, rf, 2 * ci + edge, l_ - e['value'] - float(unc_s[ci])))
                if not e['max']:
                    ho.append((p, rf, 2 * ci + edge, l_ - e['value'] + float(unc_h[ci])))
        mk = lambda L: (T([x[0] for x in L]), T([x[1] for x in L]), T([x[2] for x in L]), T([x[3] for x in L], torch.float32))
        out_d = {}
        for p, ci, e in self.out_delay:
            if not e['min']:
                out_d[p] = max(out_d.get(p, -INF), e['value'])
        ox = []
        for i, e in enumerate(self.ex):
            for c in ex_to[i]:
                p = port_pins.get(c)
                if p is not None:
                    for rf in (0, 1):
                        ox.append((p, rf, 2 * C + i, e['value'] - out_d.get(p, 0.0)))
        self.op_su, self.op_ho = mk(su), mk(ho)
        self.op_ex = mk(ox)
        tags = torch.arange(NT, device=dev)
        self.launch_t = torch.zeros(NT, device=dev)
        if C:
            self.launch_t[:2 * C] = self.wf[tags[:2 * C] // 2, tags[:2 * C] % 2]

    def _prepare_triton(self):
        """Per level, the int32 arrays of the fused stage kernel (attar.dmp_triton.stage)."""
        i32 = lambda x: x.to(torch.int32).contiguous()  # noqa: E731
        dev = self.dev

        def lev(lib, fi, t, orf, t0, tag, ea, en):
            return (i32(lib), i32(fi), i32(t), i32(orf), i32(self.pin_net[t]), t0.float().contiguous(), i32(tag),
                    i32(ea), i32(en))
        self.tri_lev = []
        if self.lc is not None:
            lib, to, orf, tag, t0 = self.lc
            self.tri_lev.append(lev(lib, torch.full_like(lib, -1), to, orf, t0, tag, *self.lc_ex))
        for cb in self.cb_lev:
            if cb is not None:
                lib, f, t_, orf, irf, ea, en = cb
                z = torch.zeros(len(lib), device=dev)
                self.tri_lev.append(lev(lib, f * 2 + irf, t_, orf, z, torch.zeros_like(lib), ea, en))
        self.tri_tables = (self.t_ix1.contiguous(), self.t_ix2.contiguous(), self.t_val.contiguous(), self.t_n32,
                           self.t_thr.contiguous())
        self.tri_sthr = self.sink_thr.contiguous()
        self.tri_nto = i32(self.n_to)

    # -- propagation ----------------------------------------------------------------------------------------------------
    def run(self, pos=None):
        """Time the design at pos (default: the design's placement). Returns a TimingResult. On CUDA the whole
        computation is recorded once as a CUDA graph and replayed (positions copied into a static buffer)."""
        pos = self.d.pos if pos is None else pos
        if not self.use_graph:
            return TimingResult(self, *self._run_core(pos))
        with torch.cuda.device(self.dev):
            if self._graph is None:
                self._pos_buf = pos.detach().clone()
                st_ = torch.cuda.Stream(); st_.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st_):
                    for _ in range(2):
                        self._run_core(self._pos_buf)
                torch.cuda.current_stream().wait_stream(st_)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._g_out = self._run_core(self._pos_buf)
                self._graph = g
            self._pos_buf.copy_(pos)
            self._graph.replay()
            return TimingResult(self, self._g_out[0].clone(), self._g_out[1].clone())

    def _run_core(self, pos):
        dev, P, NT = self.dev, self.P, self.NT
        cw, elm = self.wires(pos) if self.parasitics == 'placement' else (None, None)
        if elm is None:
            elm = torch.zeros(len(self.n_fr), 4, device=dev)
        # loads per driver pin (net wire cap + sink pin caps), max and min pin capacitances
        net_load = torch.zeros(len(self.d.n_ptr) - 1, 2, device=dev)
        net_load.index_add_(0, self.n_net, self.pin_cap[self.n_to])
        net_min = torch.zeros_like(net_load).index_add_(0, self.n_net, self.pin_cap_min[self.n_to])
        if cw is not None:
            net_load += cw[:, None]; net_min += cw[:, None]
        load = torch.zeros(P, 2, device=dev); load[self.drv_pins] = net_load[self.drv_nets]
        load_min = torch.zeros(P, 2, device=dev); load_min[self.drv_pins] = net_min[self.drv_nets]
        self.load, self.load_min = load, load_min
        L, Lm = load.view(-1), load_min.view(-1)
        pi = self.pi if self.pi is not None else torch.zeros(len(net_load), 4, 3, device=dev)
        amax = torch.full((P * 2, NT), -INF, device=dev); amin = torch.full((P * 2, NT), INF, device=dev)
        smax = torch.zeros(P * 2, device=dev); smin = torch.full((P * 2,), INF, device=dev)
        amax.view(-1).scatter_reduce_(0, self.in_ix_max, self.in_v_max, 'amax')
        amin.view(-1).scatter_reduce_(0, self.in_ix_min, self.in_v_min, 'amin')
        for rf in (0, 1):
            smax.scatter_reduce_(0, self.in_sl_p * 2 + rf, self.in_sl_v, 'amax')
            smin.scatter_reduce_(0, self.in_sl_p * 2 + rf, self.in_sl_v, 'amin')
        from . import dmp
        # nets driven by input ports: no driver model, the wire alone (dspfWireDelaySlew + thresholdAdjust)
        if len(self.pa):
            f, t_, w4 = self.n_fr[self.pa], self.n_to[self.pa], elm[self.pa]
            for rf in (0, 1):
                lt = self.sink_thr[t_, rf]
                for A, S, red, w in ((amax, smax, 'amax', w4[:, rf]), (amin, smin, 'amin', w4[:, 2 + rf])):
                    wd, ls = dmp.input_port_load(S[f * 2 + rf], w, lt[:, 0], lt[:, 1], lt[:, 2], lt[:, 3])
                    A.index_reduce_(0, t_ * 2 + rf, A[f * 2 + rf] + wd[:, None], red)
                    S.scatter_reduce_(0, t_ * 2 + rf, ls, red)

        def stage(lib, fi, t_, orf, t0, tag, ea, en, sx, sn):
            """Arcs into drivers t_ (output transition orf): gate delays and driver slews (DMP), arrivals at the
            drivers and, through each arc's wire delays (DMP load waveform), at their loads. fi: the arcs' input
            (pin * 2 + transition), or None for launch arcs (arrival t0 into tag)."""
            ti = t_ * 2 + orf
            pv = pi[self.pin_net[t_]]                                     # (n, 4, 3)
            n = len(lib)
            rows_ = torch.arange(n, device=dev)
            px, pn = pv[rows_, orf], pv[rows_, 2 + orf]                   # this transition's max / min pi models
            thr = self.t_thr[lib, orf]
            # max and min analysis in one batch; the total load is the pi model's (OpenSTA: c1 + c2)
            c2_ = lambda v: torch.cat([v, v])  # noqa: E731
            args = (c2_(lib), c2_(orf), torch.cat([sx, sn]), torch.cat([px[:, 0] + px[:, 2], pn[:, 0] + pn[:, 2]]),
                    torch.cat([px[:, 2], pn[:, 2]]), torch.cat([px[:, 1], pn[:, 1]]), torch.cat([thr, thr]))
            if self.triton is not None:
                dd, ds, stt = self.triton.gate(self.t_ix1, self.t_ix2, self.t_val, self.t_n32, *args)
            else:
                dd, ds, stt = arc_eval(self.t_ix1, self.t_ix2, self.t_val, self.t_n, *args, self.steps)
            dmax, dmin, dsmax, dsmin, stx, stn = dd[:n], dd[n:], ds[:n], ds[n:], stt[:n], stt[n:]
            if fi is None:                                     # launch: one tag, the clock edge's arrival t0
                ax = torch.full((len(lib), NT), -INF, device=dev).scatter_(1, tag[:, None], (t0 + dmax)[:, None])
                an = torch.full((len(lib), NT), INF, device=dev).scatter_(1, tag[:, None], (t0 + dmin)[:, None])
                reach = torch.ones(len(lib), dtype=torch.bool, device=dev)
            else:
                ax, an = amax[fi] + dmax[:, None], amin[fi] + dmin[:, None]
                reach = torch.isfinite(amax[fi]).any(1) | torch.isfinite(amin[fi]).any(1)
            amax.index_reduce_(0, ti, ax, 'amax'); amin.index_reduce_(0, ti, an, 'amin')
            smax.scatter_reduce_(0, ti, torch.where(reach, dsmax, torch.zeros_like(dsmax)), 'amax')
            smin.scatter_reduce_(0, ti, torch.where(reach, dsmin, torch.full_like(dsmin, INF)), 'amin')
            if len(ea):
                sk = self.n_to[en]
                rise = orf[ea] == 0
                lt = self.sink_thr[sk, orf[ea]]
                m = len(ea)
                oe = orf[ea]
                largs = (torch.cat([stx[ea], stn[ea]]), torch.cat([elm[en, oe], elm[en, 2 + oe]]), rise.repeat(2), lt.repeat(2, 1))
                w_, l_ = self.triton.load(*largs) if self.triton is not None else load_eval(*largs, self.steps)
                wx, wn, lsx, lsn = w_[:m], w_[m:], l_[:m], l_[m:]
                si = sk * 2 + orf[ea]
                amax.index_reduce_(0, si, ax[ea] + wx[:, None], 'amax')
                amin.index_reduce_(0, si, an[ea] + wn[:, None], 'amin')
                r = reach[ea]
                smax.scatter_reduce_(0, si, torch.where(r, lsx, torch.zeros_like(lsx)), 'amax')
                smin.scatter_reduce_(0, si, torch.where(r, lsn, torch.full_like(lsn, INF)), 'amin')
        if self.triton is not None and self.fused:               # one fused kernel per level
            pic, elc = pi.contiguous(), elm.contiguous()
            for lv in self.tri_lev:
                self.triton.stage(self.tri_tables, lv, self.tri_sthr, pic, elc, self.tri_nto, amax, amin, smax, smin)
            smin = torch.where(torch.isfinite(smin), smin, smax)
            self.amax, self.amin = amax.view(P, 2, NT), amin.view(P, 2, NT)
            self.smax, self.smin = smax.view(P, 2), smin.view(P, 2)
            return self._checks()
        if self.lc is not None:
            lib, to, orf, tag, t0 = self.lc
            zero = torch.zeros_like(t0)                         # ideal clock: slew 0 at the register clock pin
            stage(lib, None, to, orf, t0, tag, *self.lc_ex, zero, zero)
        for k in range(self.max_level + 1):
            cb = self.cb_lev[k]
            if cb is not None:
                lib, f, t_, orf, irf, ea, en = cb
                fi = f * 2 + irf
                sx = smax[fi]; sn = smin[fi]; sn = torch.where(torch.isfinite(sn), sn, sx)
                stage(lib, fi, t_, orf, None, None, ea, en, sx, sn)
        smin = torch.where(torch.isfinite(smin), smin, smax)
        self.amax, self.amin = amax.view(P, 2, NT), amin.view(P, 2, NT)
        self.smax, self.smin = smax.view(P, 2), smin.view(P, 2)
        return self._checks()

    # -- checks -------------------------------------------------------------------------------------------------------------
    def _checks(self):
        dev, P = self.dev, self.P
        amax, amin, smax, smin = self.amax, self.amin, self.smax, self.smin
        lt = self.launch_t
        fin = lambda x: torch.where(torch.isnan(x) | ~torch.isfinite(x), torch.full_like(x, INF), x)
        sp = torch.full((P,), INF, device=dev); hp = torch.full((P,), INF, device=dev)
        if len(self.k_lib):
            zero = torch.zeros(len(self.k_lib), device=dev)
            for rf in (0, 1):
                ok = self.t_n[self.k_lib, rf, 0] > 0
                cmax = self._lookup(self.k_lib, rf, smax[self.k_to, rf], zero)
                cmin = self._lookup(self.k_lib, rf, smin[self.k_to, rf], zero)
                lat = self.lat[self.k_ck][:, None]
                req_s = (lt[None] + self.rel_setup[:, self.k_ctag].T + lat - cmax[:, None] - self.unc_s[self.k_ck][:, None]
                         - self.k_ctp_min[:, None])
                req_h = (lt[None] + self.rel_hold[:, self.k_ctag].T + lat + cmin[:, None] + self.unc_h[self.k_ck][:, None]
                         - self.k_ctp_max[:, None])
                s = fin(req_s - amax[self.k_to, rf]).amin(1); h = fin(amin[self.k_to, rf] - req_h).amin(1)
                sp.scatter_reduce_(0, self.k_to, torch.where(ok & self.k_setup, s, torch.full_like(s, INF)), 'amin')
                hp.scatter_reduce_(0, self.k_to, torch.where(ok & ~self.k_setup, h, torch.full_like(h, INF)), 'amin')
        p, rf, ct, off = self.op_su
        if len(p):
            s = fin(lt[None] + self.rel_setup[:, ct].T + off[:, None] - amax[p, rf]).amin(1)
            sp.scatter_reduce_(0, p, s, 'amin')
        p, rf, ct, off = self.op_ho
        if len(p):
            h = fin(amin[p, rf] - (lt[None] + self.rel_hold[:, ct].T + off[:, None])).amin(1)
            hp.scatter_reduce_(0, p, h, 'amin')
        # path delay exceptions: registers (setup time from the constrained slew) and output ports
        if len(self.k_lib) and len(self.kx[0]):
            a_, tg, rq = self.kx
            to = self.k_to[a_]
            for rf in (0, 1):
                ok = self.t_n[self.k_lib[a_], rf, 0] > 0
                cst = self._lookup(self.k_lib[a_], rf, smax[to, rf], torch.zeros(len(a_), device=dev))
                s = fin(rq - cst - self.k_ctp_min[a_] - amax[to, rf, tg])
                sp.scatter_reduce_(0, to, torch.where(ok, s, torch.full_like(s, INF)), 'amin')
        p, rf, tg, rq = self.op_ex
        if len(p):
            sp.scatter_reduce_(0, p, fin(rq - amax[p, rf, tg]), 'amin')
        return sp, hp


class TimingResult:
    """Per-endpoint setup / hold slack (ps; inf where not constrained) and their summaries."""

    def __init__(self, timer, setup, hold):
        self.timer = timer
        self.setup, self.hold = setup, hold
        ep = torch.isfinite(setup)
        self.endpoints = torch.nonzero(ep).flatten()
        s = setup[ep]
        self.wns = float(min(0.0, s.min().item())) if len(s) else 0.0
        self.worst_slack = float(s.min().item()) if len(s) else INF
        self.tns = float(s.clamp(max=0).sum().item())
        self.violations = int((s < 0).sum().item())
        h = hold[torch.isfinite(hold)]
        self.whs = float(min(0.0, h.min().item())) if len(h) else 0.0
        self.ths = float(h.clamp(max=0).sum().item())
        self.hold_violations = int((h < 0).sum().item())

    def summary(self):
        return {'wns': self.wns, 'tns': self.tns, 'worst_slack': self.worst_slack, 'setup_violations': self.violations,
                'endpoints': int(len(self.endpoints)), 'whs': self.whs, 'ths': self.ths, 'hold_violations': self.hold_violations}

    def _pin_name(self, p):
        d = self.timer.d
        n = d.names
        c = int(self.timer.p_cell_np[p]); m = int(d.raw['p_mpin'][p])
        if self.timer._mp_names is None:
            self.timer._mp_names = [x for ps in n['master_pins'] for x in ps]
        return (n['cells'][c] if m < 0 else '%s/%s' % (n['cells'][c], self.timer._mp_names[m])).replace('\\', '')

    def path(self, endpoint=None):
        """The latest-arriving path into an endpoint (default: the worst setup endpoint), traced back from the
        arrivals: at a net sink its driver; at a cell output the arc and input transition whose arrival + delay is the
        latest (gate delays recomputed with the timer's own arc_eval). Returns [(pin, 'r'|'f', arrival ps, slew ps)],
        startpoint first. Valid for the last run of this result's timer."""
        T = self.timer
        dev = T.dev
        if endpoint is None:
            p = int(torch.argmin(self.setup).item())
        elif isinstance(endpoint, str):
            p = next(q for q in self.endpoints.tolist() if self._pin_name(q) == endpoint)
        else:
            p = int(endpoint)
        arr = T.amax.max(2).values                                          # (P, 2): latest over tags
        rf = int(torch.argmax(arr[p]).item())
        sink_drv = dict(zip(T.n_to.tolist(), T.n_fr.tolist())) if T._sink_drv is None else T._sink_drv
        T._sink_drv = sink_drv
        out = []
        for _ in range(100000):
            out.append((self._pin_name(p), 'rf'[rf], float(arr[p, rf]), float(T.smax[p, rf])))
            if p in sink_drv:
                p = sink_drv[p]; continue
            ai = torch.nonzero(T.c_to == p).flatten()
            if not len(ai):
                break                                               # a startpoint (register output, input port)
            lib, f = T.c_lib[ai], T.c_fr[ai]
            best, nxt = -INF, None
            for irf in (0, 1):
                sense = T.c_sense[ai]
                use = (sense == S_NON) | ((sense == S_POS) & (irf == rf)) | ((sense == S_NEG) & (irf != rf))
                use = use & (T.t_n[lib, rf, 0] > 0)
                if not bool(use.any()):
                    continue
                orf = torch.full_like(lib, rf)
                pi = T.pi[T.pin_net[torch.full_like(lib, p)], rf] if T.pi is not None else torch.zeros(len(ai), 3, device=dev)
                dmax, _, _ = arc_eval(T.t_ix1, T.t_ix2, T.t_val, T.t_n, lib, orf, T.smax[f, irf], pi[:, 0] + pi[:, 2],
                                      pi[:, 2], pi[:, 1], T.t_thr[lib, orf])
                cand = torch.where(use, arr[f, irf] + dmax, torch.full_like(dmax, -INF))
                k = int(torch.argmax(cand).item())
                if float(cand[k]) > best:
                    best, nxt = float(cand[k]), (int(f[k]), irf)
            if nxt is None:
                break
            p, rf = nxt
        return out[::-1]

    def endpoint_slacks(self, names=True):
        """{endpoint name ('inst/PIN' or port): (setup slack, hold slack)}."""
        d = self.timer.d
        n = d.names
        cells, mp = n['cells'], [x for ps in n['master_pins'] for x in ps]
        p_mpin = d.raw['p_mpin']
        out = {}
        for p in self.endpoints.tolist():
            c = int(self.timer.p_cell_np[p])
            nm = (cells[c] if p_mpin[p] < 0 else '%s/%s' % (cells[c], mp[p_mpin[p]])).replace('\\', '')
            out[nm] = (float(self.setup[p]), float(self.hold[p]))
        return out
