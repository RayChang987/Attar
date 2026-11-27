"""Static timing analysis on the GPU: the 'sta' capability (needs the 'timing' fact, see attar.timing.ir).

Graph: a node per pin and transition (rise, fall); net arcs (driver -> sink) and cell arcs (one per Liberty timing arc
and instance). Clocks are ideal (pre-CTS): a register clock pin sees its clock's edge plus the ideal latency, slew 0
Arrivals are kept per tag (launching clock, launching edge), so setup and hold
use OpenSTA's edge relationships between any two clocks. Delays: NLDM tables, bilinear with linear extrapolation
(OpenSTA's rule), at the effective capacitance of the driver's pi model (OpenSTA's DmpCeff, attar.timing.dmp). Wires
(parasitics='placement'): OpenROAD's estimate_parasitics -placement: per net its Steiner tree (pdr::primDijkstra,
alpha 0.3), a pi segment per branch (set_wire_rc -signal), each pin at OpenROAD's location (the mean of its LEF
rectangles' centers, _pin_setup) behind its via resistance; reduced as OpenSTA does to the driver's pi model and
every sink's Elmore delay.
Path delays: set_max_delay -from / -to. Recovery / removal checks are setup / hold checks (as OpenSTA's). Latches: data
passes D -> Q while the enable is open (time borrowing, OpenSTA's Latches) in the max analysis, and a latch D's setup
required time is OpenSTA's (the opening, the arrival while borrowing, or the closing less max borrow); the min analysis
passes nothing through a latch. Combinational loops are broken where OpenSTA's levelization breaks them. Not
modeled: other path exceptions (set_false_path, ...), clock gating checks.
"""
import re
import math
import os
import sys
import warnings

import numpy as np
import torch

from .. import metrics as M
from ..io.distill import D_IN, D_INOUT, D_OUT
from . import ir as timing_mod
from .ir import (S_NEG, S_NON, S_POS, T_CLEAR, T_COMB, T_FALL_EDGE, T_HOLD_F, T_HOLD_R, T_PRESET, T_RISE_EDGE,
                     T_SETUP_F, T_SETUP_R, T_TRISTATE)

warnings.filterwarnings('ignore', message='index_reduce')     # torch marks index_reduce beta; it is what we need
LN9 = math.log(9.0)
INF = float('inf')
# the version of the timer's data for research code (TimingResult.graph, library, arrivals, parasitics, required;
# TimingData.version): raised when a field changes meaning or shape, so code built on it can check what it reads
TIMING_DATA_VERSION = 1


def nldm(ix1, ix2, val, n, x, y):
    """Table lookup for E arcs: ix1, ix2 (E, K) axes, val (E, K, K), n (E, 2) sizes, x / y (E,) -> (E,).
    Bilinear inside, linear extrapolation outside (from the end segment), constant along a size-1 axis."""
    E, K = ix1.shape
    dev = x.device
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


def _latch_relationships(clocks):
    """Latch setup cycle accounting (OpenSTA's CycleAccting, TimingRole::latchSetup) between every data (launch
    clock, edge) and latch enable (clock, open edge): the enable's opening is the one before the first closing
    (opposite) edge strictly after the launch. Returns (required, shift), (2C, 2C) each: the opening time relative to
    the launch cycle's start, and the launch cycle's offset less the enable cycle's (sourceTimeOffset -
    targetTimeOffset: moves an arrival into the enable clock's cycle 0)."""
    C = len(clocks)
    req = np.full((2 * C, 2 * C), np.nan); sh = np.full((2 * C, 2 * C), np.nan)
    for a, ca in enumerate(clocks):
        for b, cb in enumerate(clocks):
            Ta, Tb = ca['period'], cb['period']
            ia, ib = int(round(Ta * 1000)), int(round(Tb * 1000))
            L = ia * ib // math.gcd(ia, ib) if ia > 0 and ib > 0 else ia
            n_launch = max(1, min(1000, L // max(ia, 1)))
            for ea in (0, 1):
                for eb in (0, 1):
                    best = INF
                    for m in range(n_launch):
                        ts = ca['waveform'][ea] + m * Ta
                        opp0, en0 = cb['waveform'][1 - eb], cb['waveform'][eb]
                        k = math.floor((ts - opp0) / Tb) + 1                 # first closing edge strictly after
                        if opp0 + k * Tb - ts < 1e-6:
                            k += 1
                        t_opp = opp0 + k * Tb
                        t_en, cyc = en0 + k * Tb, k
                        if t_en > t_opp:
                            t_en -= Tb; cyc -= 1
                        if t_opp - ts < best - 1e-9:
                            best = t_opp - ts
                            req[2 * a + ea, 2 * b + eb] = t_en - m * Ta
                            sh[2 * a + ea, 2 * b + eb] = m * Ta - cyc * Tb
    return req, sh


STATE = ('valid', 'k0', 'k1', 'k2', 'k3', 'k4', 'p1', 'p2', 't0', 'dt', 'vo_delay', 'ub', 'drvr_slew', 'vth', 'vl', 'vh',
         'derate')


def _stack_tables(a, b):
    """Two libraries' tables (A, 4, K...) one after the other, padded to the larger K (the sizes in t_n say how much
    of each table is real)."""
    if a.shape[2:] != b.shape[2:]:
        k = [max(x, y) for x, y in zip(a.shape[2:], b.shape[2:])]
        pad = lambda v: torch.nn.functional.pad(v, [p for i in reversed(range(len(k))) for p in (0, k[i] - v.shape[2 + i])])  # noqa: E731
        a, b = pad(a), pad(b)
    return torch.cat([a, b])


def arc_eval(ix1, ix2, val, tn, lib, orf, slew, ctot, c_far, rpi, thr, steps=None):
    """OpenSTA's dmp_ceff_elmore gate delay for a batch of cell arcs (attar.timing.dmp.gate): gate delay, driver slew and the
    driver model each load's wire delay needs, packed as STATE columns. ctot: the driven net's total capacitance for
    this analysis (max or min pin caps), (c_far, rpi): its pi model's far capacitance and resistance (the near one is
    ctot - c_far). thr: the driver library's (output threshold, slew lower, slew upper, slew derate)."""
    from . import dmp
    F = torch.float64                                    # OpenSTA's precision (float32 fails on strongly resistive nets)
    dt_out = slew.dtype
    tab = (ix1[lib, orf].to(F), ix2[lib, orf].to(F), val[lib, orf].to(F), tn[lib, orf],
           ix1[lib, 2 + orf].to(F), ix2[lib, 2 + orf].to(F), val[lib, 2 + orf].to(F), tn[lib, 2 + orf])
    slew, ctot, c_far, rpi, thr = slew.to(F), ctot.to(F), c_far.to(F), rpi.to(F), thr.to(F)
    c_far = torch.minimum(c_far, ctot)
    delay, dslew, _, stt = dmp.gate(tab, slew, ctot, (ctot - c_far).clamp(min=0), rpi, c_far,
                                    thr[:, 0], thr[:, 1], thr[:, 2], thr[:, 3], steps or dmp.EAGER)
    return delay.to(dt_out), dslew.to(dt_out), torch.stack([stt[k].to(dt_out) for k in STATE], 1)


def load_eval(state, elmore, rise, lthr, steps=None):
    """Wire delay and load slew of the loads of a batch of arcs (attar.timing.dmp.load), at the loads' thresholds
    (DelayCalcBase::thresholdAdjust). state: the arcs' STATE rows, one per load; lthr: the loads' (input threshold,
    slew lower, slew upper, slew derate)."""
    from . import dmp
    F, dt_out = torch.float64, elmore.dtype
    st_ = {k: state[:, i].to(F) for i, k in enumerate(STATE)}
    st_['valid'] = st_['valid'] > 0.5
    w, sl = dmp.load(st_, elmore.to(F), steps or dmp.EAGER)
    lthr = lthr.to(F)
    w, sl = dmp.threshold_adjust(w, sl, rise, st_['vth'], st_['vl'], st_['vh'], st_['derate'],
                                 lthr[:, 0], lthr[:, 1], lthr[:, 2], lthr[:, 3])
    return w.to(dt_out), sl.to(dt_out)


_NOTICED = False


def _default_thresholds(info):
    """(rise, fall) x (input, slew lower, slew upper, slew derate) of the first library read (ports, missing data)."""
    dt_ = info.get('default_thresholds') or {'input': (.5, .5), 'slew_lower': (.2, .2), 'slew_upper': (.8, .8),
                                              'slew_derate': 1.0}
    return np.array([[dt_['input'][i], dt_['slew_lower'][i], dt_['slew_upper'][i], dt_['slew_derate']] for i in (0, 1)],
                    np.float32)


def _first_compile_notice():
    """Say so when the GPU timer's Triton kernels are not compiled on this machine yet: the first run compiles them
    (about two minutes on an RTX 3080 Ti) into Triton's cache, and later runs, in any process, reuse them."""
    global _NOTICED
    if _NOTICED:
        return
    _NOTICED = True
    probe = 'attar_cache_probe'
    try:                    # where Triton keeps its kernels, as Triton resolves it (its setting, else its default)
        from triton.runtime.cache import FileCacheManager
        made = FileCacheManager(probe).cache_dir               # Triton creates <cache>/<probe>: removed again
        cache = os.path.dirname(made)
        try:
            os.rmdir(made)
        except OSError:
            pass
    except Exception:  # noqa: BLE001
        cache = os.path.join(os.path.expanduser('~'), '.triton', 'cache')
    if not os.path.isdir(cache) or not [e for e in os.listdir(cache) if e != probe]:
        print('attar.timing: first GPU timing run on this machine: compiling its Triton kernels (a couple of minutes, '
              'once; cached in %s; a fresh machine such as a new Colab runtime compiles again unless TRITON_CACHE_DIR '
              'and TORCHINDUCTOR_CACHE_DIR point to a folder it keeps)' % cache, flush=True)


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


def tree_elmore(p, plan, n_nets, cap, r, c, length, dbu, rcut=0.0, into=None):
    """Elmore delay from each net's driver to every pin and the driver's pi model, on OpenROAD's estimate-parasitics
    tree (attar.timing.steiner.pd_tree: Prim-Dijkstra alpha 0.3 on nearest neighbors, steinerized; dbu coordinates), each
    branch a pi segment of wire RC r, c per um, every pin behind its via resistance rcut (a scalar, OpenROAD's average
    cut resistance, or per pin (P,): the vias up to the wire's layer), the driver's in series with the whole tree (EstimateParasitics::estimateWireParasiticSteiner, parasiticNodeConnectPins). cap: pin capacitance (P,).
    Nets above the largest bucket: from plan['big_buf'] (their exact trees, filled by Timer._big_update), else lumped
    on length (r x distance x half the net's C).
    cap may hold V variants (P, V) (rise / fall, max / min pin capacitances: OpenSTA reduces per rise / fall and
    min / max). Returns per pin and variant the Elmore delay (P, V) in ps, per net and variant the pi model (E, V, 3)
    near C / R / far C, and per net the tree's wire length in um (length where no tree was built).
    into: (out, pi, tlen) buffers to update in place, only the rows of the plan's nets and pins (Timer.wires
    recomputes the nets that changed with a plan of those); tlen's rows of the plan's nets must hold length."""
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
    if into is None:
        out = torch.zeros(len(p), V, device=dev)
        pi = torch.zeros(n_nets, V, 3, device=dev)
        tlen = length.clone()
    else:
        out, pi, tlen = into
    if rc_bucket is not None:                     # one Triton launch per bucket: tree, moments, pi, scattered outputs
        pc, cc = p.float().contiguous(), cap.float().contiguous()
        rv = (rcut.float() if torch.is_tensor(rcut) else torch.full((len(p),), float(rcut), device=dev)).contiguous()
        # the high-degree buckets (few nets, one long program each) on a side stream, beside the small ones
        cur = torch.cuda.current_stream(dev)
        side = _side_stream(dev)
        side.wait_stream(cur)
        for idx, real, nets in plan['tri']:
            if len(nets):
                with torch.cuda.stream(side if idx.shape[1] >= SIDE_D else cur):
                    rc_bucket(pc, idx, real, cc, nets, out, pi, tlen, 0.3, dbu, r, c, rv)
        cur.wait_stream(side)
    for D, nets, idx, real, _rows, rpins, ri in (plan['buckets'] if rc_bucket is None else ()):
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
        # point capacitance: half of every incident wire (pi segments); pins behind their vias at their points
        half = c * el / 2
        cw = (half + torch.bmm(Pm.transpose(1, 2), half[:, :, None])[:, :, 0])[:, :, None]   # (n, N, 1)
        cn = cw + cp                                                       # (n, N, V) per pin-cap variant
        Re = (r * el)[:, :, None]                                          # resistance of each point's parent wire
        sub = lambda x: torch.bmm(M.transpose(1, 2), x)                    # subtree sums
        path = lambda x: torch.bmm(M, x)                                   # root-to-point sums
        rv = rcut[idx] if torch.is_tensor(rcut) else torch.full(idx.shape, float(rcut), device=dev)
        rv = torch.cat([torch.where(real, rv, torch.zeros_like(rv)), torch.zeros(n, N - D, device=dev)], 1)[:, :, None]
        rd = rv[:, :1]                                                     # the driver's (slot 0), (n, 1, 1)
        y1 = cn.sum(1)                                                     # (n, V)
        m1 = rd * y1[:, None, :] + path(Re * sub(cn))                      # Elmore at the points
        m1p = m1 + rv * cp                                                 # ... at the pins
        qq = cw * m1 + cp * m1p
        m2 = rd * qq.sum(1, keepdim=True) + path(Re * sub(qq))
        m2p = m2 + rv * cp * m1p
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
        pb = plan['pin_big']
        out[pb] = (r * dd[:, None] * ctot[net_of] / 2)[pb]
        pi[big] = torch.stack([ctot[big], torch.zeros_like(ctot[big]), torch.zeros_like(ctot[big])], -1)
    if plan.get('big_buf') is not None:              # large data nets: their exact trees (attar.timing.steiner_big, host)
        nets_b, pins_b, elm_b, pi_b, len_b = plan['big_buf']
        out[pins_b] = elm_b
        pi[nets_b] = pi_b
        tlen[nets_b] = len_b
    return out, pi, tlen


class Timer:
    """Build once per design (graph, levels, constraints); run() times a placement."""

    def _min_arrays(self, d, t):
        """The Early library's arrays for the min analysis (timing_mod.arrays_min), checked to line up arc for arc
        with the Late ones in t; None without one (or, with a limitation, when they do not line up)."""
        tmin = timing_mod.arrays_min(d)
        if tmin is None:
            return None
        if not all(np.array_equal(t[k], tmin[k]) for k in timing_mod.ALIGN):
            self._limit('the Early library does not line up with the Late one arc for arc: hold uses the Late one')
            return None
        return tmin

    def _limit(self, msg):
        """A way this timer differs from OpenSTA on the design's inputs: warned, and listed in TimingResult.limitations
        (summary()['exact'] is False)."""
        self.limitations.append(msg)
        warnings.warn('attar.timing.sta: ' + msg, stacklevel=3)

    def release(self, why):
        """Drop everything the timer holds (tables, buffers, CUDA graphs: their GPU memory goes back before the next
        timer is built). The timer can no longer run; anything that still holds it (a TimingResult) gets `why`."""
        self.__dict__.clear()
        self._stale = why

    def resized(self, cells):
        """Update the timer in place after Design.resize swapped `cells` to other masters (attar.timing.sizing): the
        graph, its levels and the CUDA graph stay; each instance arc points to its new master's Liberty arc and every
        pin takes its new master pin's data (capacitance, thresholds, clock-tree path, via resistance, location).
        Returns False, leaving the timer unchanged, where that is not enough and a new timer must be built: a master
        the design had no tables for yet (the tables grow), or new masters whose arcs do not line up one to one with
        the old ones (pins, type, sense, when, which tables exist)."""
        if self.parasitics != 'placement' or getattr(self, '_sel', None) is None:
            return False
        d = self.d
        a = d.raw
        t, functions, arc_when = timing_mod.arrays(d)
        if len(t['a_master']) != self.n_lib:
            return False                                # new masters: their tables are not on the device
        tmin = self._min_arrays(d, t) if self.lmin else None
        lib_id, fr, to, atype, asense_lib = self._arcs_all
        cells = np.unique(np.asarray(cells, np.int64))
        p_cell = self.p_cell_np
        is_cell = np.zeros(int(p_cell.max()) + 1, bool); is_cell[cells] = True
        pos = np.nonzero(is_cell[self._arc_cell])[0]
        p_mpin = a['p_mpin'].astype(np.int64)
        c_master = a['c_master'].astype(np.int64)
        am = t['a_master'].astype(np.int64)
        a_from, a_to = t['a_from'].astype(np.int64), t['a_to'].astype(np.int64)
        when = lambda lid: arc_when.get(str(int(lid)), '')        # noqa: E731
        tab = lambda lid: tuple((t['a_n'][lid] > 0).ravel().tolist())   # noqa: E731
        # the new masters' arcs on each cell's pins, as the build makes them
        pins = np.nonzero(is_cell[p_cell] & (p_mpin >= 0))[0]
        pin_at = {(int(p_cell[p]), int(p_mpin[p])): int(p) for p in pins}
        lib_order = np.argsort(am, kind='stable')
        lib_start = np.searchsorted(am[lib_order], np.arange(int(am.max(initial=-1)) + 2))
        new = {}
        for c in cells.tolist():
            m = int(c_master[c])
            for lid in lib_order[lib_start[m]:lib_start[m + 1]].tolist():
                f, g = pin_at.get((c, int(a_from[lid]))), pin_at.get((c, int(a_to[lid])))
                if f is None or g is None:
                    continue
                key = (f, g, int(t['a_type'][lid]), int(t['a_sense'][lid]), when(lid), tab(lid))
                new.setdefault(key, []).append(lid)
        old = {}
        for i in pos.tolist():
            key = (int(fr[i]), int(to[i]), int(atype[i]), int(asense_lib[i]), when(lib_id[i]), tab(lib_id[i]))
            old.setdefault(key, []).append(i)
        if {k: len(v) for k, v in old.items()} != {k: len(v) for k, v in new.items()}:
            return False                                # the arcs do not line up: the graph would change
        lib_new = lib_id.copy()
        for k, ii in old.items():
            lib_new[ii] = new[k]
        changed = np.zeros(len(lib_id), bool)
        changed[pos[lib_new[pos] != lib_id[pos]]] = True
        # every device copy of an instance arc's Liberty arc, in place (the CUDA graph keeps reading the same tensors)
        inst = {s: self._sel[s] for s in ('c', 'l', 'k')}

        def write(tensor, src):                         # tensor[i] = new Liberty arc of instance arc src[i]
            hit = np.nonzero(changed[src])[0]
            if len(hit):
                tensor[torch.as_tensor(hit, device=self.dev)] = torch.as_tensor(lib_new[src[hit]], device=self.dev,
                                                                                 dtype=tensor.dtype)
        write(self.c_lib, inst['c']); write(self.l_lib, inst['l']); write(self.k_lib, inst['k'])
        if self.lc is not None:
            write(self.lc[0], inst['l'][self._lc_src])
        for cb, src in zip(self.cb_lev, self._cb_src):
            if cb is not None:
                write(cb[0], inst['c'][src])
        for lv, (s, src) in zip(getattr(self, 'tri_lev', ()), getattr(self, '_tri_src', ())):
            write(lv[0], inst[s][src])
        self._arcs_all = (lib_new, fr, to, atype, asense_lib)
        self.info['functions'], self.info['arc_when'] = functions, arc_when
        # the pins' Liberty data, via resistance and location
        info = self.info
        cap, cap_min, ctp, sthr = self._pin_library(t, info, p_cell, p_mpin)
        if tmin is not None:
            cap_min = self._pin_library(tmin, info, p_cell, p_mpin)[1]
        dev = self.dev
        self.pin_cap.copy_(torch.as_tensor(cap, device=dev)); self.pin_cap_min.copy_(torch.as_tensor(cap_min, device=dev))
        self.pin_ctp.copy_(torch.as_tensor(ctp, device=dev)); self.sink_thr.copy_(torch.as_tensor(sthr, device=dev))
        if len(self.k_lib):
            self.k_ctp_min.copy_(self.pin_ctp[self.k_fr, 0, self.k_crf]); self.k_ctp_max.copy_(self.pin_ctp[self.k_fr, 1, self.k_crf])
        rc = info.get('rc') or {}
        io_mask = np.isin(p_cell, a['io_cell']) if 'io_cell' in a else np.zeros(len(p_cell), bool)
        rv = self._pin_vias(d, info, rc, p_cell, p_mpin, io_mask)
        if self.net_clkrc.any() and rc.get('clock_layers'):
            rv_clk = self._pin_vias(d, info, rc, p_cell, p_mpin, io_mask, layers=rc['clock_layers'])
            rv = np.where(self.net_clkrc[d.p_net.cpu().numpy()], rv_clk, rv)
        self.pin_rvia.copy_(torch.as_tensor(rv, device=dev, dtype=self.pin_rvia.dtype))
        if self._iclk_pi is not None:
            from .drv import SegSum
            tot = SegSum(d.p_net, self.ev.E)(torch.cat([self.pin_cap, self.pin_cap_min], 1))
            self._iclk_pi.copy_(torch.stack([tot, torch.zeros_like(tot), torch.zeros_like(tot)], -1))
        b = getattr(self, '_big', None)
        if b is not None:
            pins_b = b['pins'].cpu().numpy()
            b['cap'] = np.concatenate([cap, cap_min], 1)[pins_b]
            b['rv'] = self.pin_rvia.cpu().numpy()[pins_b]
        self._pin_setup()                               # locations (the master's pin rectangles), into ev.off in place
        self._mp_names = None; self._sink_drv = None
        self._power = None                              # its functions and tables are the old masters' (OpenSTA:
        return True                                     # replaceEquivCellAfter -> Power::powerInvalid)

    def __getattr__(self, name):
        stale = self.__dict__.get('_stale')
        if stale is not None:
            raise RuntimeError('this timing result is stale: %s since; time the design again (d.timing())' % stale)
        raise AttributeError(name)

    def __init__(self, d, parasitics='placement', exact_trees=True, precision='double', routes=None, via_res=None):
        """parasitics: 'placement' (estimate_parasitics -placement: Steiner trees at the placement) or
        'global_routing' (estimate_parasitics -global_routing on `routes`, an attar.route.GlobalRoutes; attar.timing.grpar;
        via_res: per routing level l the cut resistance between l and l + 1 in kohm, where timing.json's set_layer_rc
        has no -via for it: OpenROAD then reads the LEF cut layer's; default: the routing fact's 'cut_res').
        exact_trees: every net that can carry a data arrival gets OpenROAD's tree on the host (attar.timing.steiner_big,
        an exact port including its tie-breaks); False: the GPU kernel for nets up to 256 pins (faster; its ties
        differ from OpenROAD's on a few nets in 10^4), the host for larger ones.
        precision: the GPU delay model's, 'double' (OpenSTA's; float32 fails or drifts on strongly resistive nets),
        'retry' (float32, failing lanes in float64) or 'single'."""
        d.require('sta')
        self.d = d
        self.exact_trees = exact_trees
        self.precision = precision
        self.dev = dev = d.device
        self.parasitics = parasitics
        if parasitics not in ('placement', 'global_routing'):
            raise ValueError("parasitics: 'placement' or 'global_routing', not %r" % (parasitics,))
        if parasitics == 'global_routing' and routes is None:
            raise ValueError("parasitics='global_routing' needs routes= (attar.route.GlobalRoutes)")
        self.routes, self.via_res = routes, via_res
        # timing.json with the design's in-memory changes (Design.set_timing, propagate_clocks, ideal_clocks)
        self.info = info = d._timing_info()
        # where this timer is known to differ from OpenSTA on these inputs (TimingResult.limitations)
        self.limitations = ['SDC command %s (%d) not applied' % (k, v) for k, v in sorted((info.get('unsupported') or {}).items())]
        # the cell data: from the design's library chain (bundle + local) or its timing.safetensors (attar.timing.ir.arrays)
        t, info['functions'], info['arc_when'] = timing_mod.arrays(d)
        tmin = self._min_arrays(d, t)           # the min analysis's library (an Early library), or None: the same
        libs = [x.get('name', '') for x in (info.get('sources') or {}).get('lib') or []]
        if libs and all('_late' in x.lower() for x in libs) and tmin is None:
            self.limitations.append('hold checks use the Late library (the benchmark\'s Early library is not read)')
        a = d.raw
        T = lambda x, dt=None: torch.as_tensor(np.asarray(x), device=dev, dtype=dt)
        p_cell = a['p_cell'].astype(np.int64); p_mpin = a['p_mpin'].astype(np.int64)
        P = len(p_cell); self.P = P
        n_mp = len(t['mp_cap'])
        pdir = a['p_dir'].astype(np.int64) if 'p_dir' in a else np.full(P, 3)
        p_net = d.p_net.cpu().numpy(); n_ptr = a['n_ptr'].astype(np.int64)
        c_master = a['c_master'].astype(np.int64)
        m_ptr = a['m_pin_ptr'].astype(np.int64)

        ok = p_mpin >= 0
        cap, cap_min, ctp, sthr = self._pin_library(t, info, p_cell, p_mpin)
        if tmin is not None:                    # the min analysis's pin capacitances: the Early library's
            cap_min = self._pin_library(tmin, info, p_cell, p_mpin)[1]
        self.pin_cap, self.pin_cap_min = T(cap), T(cap_min)
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
        # constants from tie cells: arcs into a constant output, or from an input that cannot change it, are off for
        # the arrival search, and the search's sense narrows (OpenSTA's sim); delay calculation (graph slews) still
        # uses every arc with its Liberty sense (GraphDelayCalc's DcalcPred ignores constants)
        asense_lib = asense.copy()
        off, asense = self._constants(d, info, t, p_cell, p_mpin, pdir, p_net, n_ptr, c_master, fr, to, atype, lib_id,
                                      asense)
        self._arcs_all = (lib_id, fr, to, atype, asense_lib)            # every instance arc (attar.timing.power)
        self._arc_cell = p_cell[fr]                                      # its instance (resized())

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
        # propagated clocks (set_propagated_clock): their network is timed (clock tree delays and slews); ideal clocks'
        # is not (an ideal latency and transition at the register clock pins)
        self.prop_clk = np.array([bool(c.get('propagated')) for _, c in clocks], bool)
        pclk = clk_net & (self.prop_clk[np.maximum(clk_of, 0)] if C else False)
        iclk = clk_net & ~pclk
        self.pclk, self.iclk = pclk, iclk

        # data graph: net and combinational arcs not into the clock network; launch arcs from clock pins. Delay
        # calculation also times the arcs the search does not use (constant-disabled arcs; register clock-to-output
        # arcs whose clock pin no clock reaches): they set slews only (c_arr False)
        # (OpenROAD's delay calculation skips arcs out of constant pins, as the search does: an output whose inputs are
        # all constant keeps slew 0; an arc from a live input into a constant output still sets its slew)
        # a propagated clock's network and launch arcs (clock pin -> output) are ordinary arcs of the search; an
        # ideal clock's network sets slews only (OpenSTA's graph slews there, with no wire parasitics: wires())
        nk = np.ones(len(n_to), bool)
        edge = np.isin(atype, (T_RISE_EDGE, T_FALL_EDGE))
        from_const = self.const_net[p_net[fr]] >= 0
        plaunch = edge & pclk[fr] & ~clk_net[to] & ~off
        ck = ((comb | (edge & (clk_of[fr] < 0))) | plaunch) & ~from_const
        # combinational loops: the edges OpenSTA's levelization disables (Levelize::findBackEdges) are not searched
        ldq = self._latch_d_to_q(d, t, lib_id, atype)
        loop_arc, loop_net = self._loop_edges(P, n_fr, n_to, fr, to, atype, ldq, p_cell, d.raw['p_mpin'])
        ck &= ~loop_arc
        self.loop_edges = (np.nonzero(loop_arc)[0], np.nonzero(loop_net)[0])
        launch = edge & iclk[fr] & ~clk_net[to] & ~off
        self.net_iclk = np.zeros(len(n_ptr) - 1, bool)
        self.net_iclk[drv >= 0] = iclk[drv[drv >= 0]]
        self.n_fr, self.n_to = T(n_fr[nk]), T(n_to[nk])
        self.n_net = T(p_net[n_to[nk]])
        sel = lambda m: (T(lib_id[m]), T(fr[m]), T(to[m]), T(asense[m]), T(atype[m]))
        self.c_lib, self.c_fr, self.c_to, self.c_sense, self.c_type = sel(ck)
        self.c_sense_lib = T(asense_lib[ck])
        # (a data pin into a propagated clock network, e.g. a clock gate's enable: its arrivals are data, not the
        # clock's (OpenSTA's tags are not clock tags there), so they do not become clock arrivals; slews only)
        d2c = ~clk_net[fr] & pclk[to]
        # latch D -> Q arcs are not ordinary arcs of the search (Latches::latchOutArrival): a latch whose enable a clock
        # reaches passes data that arrives while it is open, as data of the enable's clock (_latch_prepare: the stage
        # reads a transformed copy of D's arrivals); one whose enable no clock reaches passes nothing (slews only)
        clocked = np.zeros(len(d.raw['c_master']) + 1, bool)
        clocked[p_cell[fr[edge & clk_net[fr]]]] = True
        self._lq_c = np.nonzero((ldq & clocked[p_cell[fr]])[ck])[0]          # c-selection indices of the clocked ones
        self.c_arr = T(((comb & ~iclk[fr] & ~iclk[to] & ~d2c & ~ldq) | plaunch)[ck] & ~off[ck])   # the search's
        #                                                                                    (the rest: slews)
        self.l_lib, self.l_fr, self.l_to, self.l_sense, self.l_type = sel(launch)
        chk = np.isin(atype, (T_SETUP_R, T_HOLD_R, T_SETUP_F, T_HOLD_F)) & (clk_of[fr] >= 0) & ~off
        self.k_lib, self.k_fr, self.k_to, self.k_sense, self.k_type = sel(chk)
        # each selection's instance arcs (resized() writes new Liberty arcs through these)
        self._sel = {'c': np.nonzero(ck)[0], 'l': np.nonzero(launch)[0], 'k': np.nonzero(chk)[0]}

        # levels (Kahn) over net + combinational arcs (and the slew-only arcs)
        gn = nk & ~loop_net                              # (a disabled loop wire still loads its net)
        g_fr = np.concatenate([n_fr[gn], fr[ck]]); g_to = np.concatenate([n_to[gn], to[ck]])
        # latch D -> Q arcs: OpenSTA's levelization never follows them (Levelize::searchThru); here a cycle is broken
        # there first (with propagated clocks, a clock gating latch closes one: register clock -> Q -> ... -> latch
        # D -> Q -> gate enable -> gated clock tree -> register clock)
        lev, dropped = self._levelize(P, g_fr, g_to, prefer=np.concatenate([np.zeros(int(gn.sum()), bool), ldq[ck]]))
        self.n_cycle_edges = dropped
        self.level = lev
        psrc = np.nonzero(pclk & np.isin(p_cell, [c_ for _, c in clocks for c_ in c['sources']]))[0]
        self._g_edges = (g_fr, g_to, np.concatenate([to[launch], psrc]))   # graph and seeds (structural reachability)
        nl = lev[n_to[nk]]; cl = lev[to[ck]]
        self.n_lev = T(nl); self.c_lev = T(cl)
        self.max_level = int(max(nl.max(initial=0), cl.max(initial=0)))
        self.n_by_lev = self._group(nl); self.c_by_lev = self._group(cl)

        # library tables on the device
        self.t_ix1, self.t_ix2 = T(t['a_ix1']), T(t['a_ix2'])
        # thresholds: drivers per arc and output transition; loads per pin (ports: the first library read)
        dflt = _default_thresholds(info)
        self.t_thr = T(t['a_thr']) if 'a_thr' in t else T(np.tile(dflt, (len(t['a_master']), 1, 1)))
        self.sink_thr = T(sthr)
        self.t_val, self.t_n = T(t['a_val']), T(t['a_n'].astype(np.int64))
        # an Early library: its tables after the Late ones; the min analysis reads Liberty arc a at a + lmin (the
        # arcs line up one to one, ir.ALIGN). lmin = 0: one library for both analyses
        self.n_lib, self.lmin = len(t['a_master']), 0
        if tmin is not None:
            thr_e = T(tmin['a_thr']) if 'a_thr' in tmin else T(np.tile(dflt, (len(tmin['a_master']), 1, 1)))
            self.t_ix1, self.t_ix2, self.t_val = (_stack_tables(self.t_ix1, T(tmin['a_ix1'])),
                                                  _stack_tables(self.t_ix2, T(tmin['a_ix2'])),
                                                  _stack_tables(self.t_val, T(tmin['a_val'])))
            self.t_n = torch.cat([self.t_n, T(tmin['a_n'].astype(np.int64))])
            self.t_thr = torch.cat([self.t_thr, thr_e])
            self.lmin = self.n_lib

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
        # set_clock_transition: the ideal clock's slew at the register clock pins, rise / fall (default 0)
        self.ctr = T([c.get('transition', [0.0, 0.0]) for c in self.clocks] or np.zeros((0, 2)), torch.float32)
        self.unc_s = T([c.get('uncertainty_setup', c.get('uncertainty', 0.0)) for c in self.clocks] or np.zeros(0), torch.float32)
        self.unc_h = T([c.get('uncertainty_hold', c.get('uncertainty', 0.0)) for c in self.clocks] or np.zeros(0), torch.float32)
        io_mask = np.isin(p_cell, a['io_cell']) if 'io_cell' in a else np.zeros(len(p_cell), bool)
        port_pin = dict(zip(p_cell[io_mask].tolist(), np.nonzero(io_mask)[0].tolist()))
        self.in_delay = [(port_pin[e['cell']], cidx.get(e['clock'], -1), e) for e in info.get('input_delay', [])
                         if e['cell'] in port_pin and e['clock'] in cidx and not clk_net[port_pin[e['cell']]]]
        self.out_delay = [(port_pin[e['cell']], cidx.get(e['clock'], -1), e) for e in info.get('output_delay', [])
                          if e['cell'] in port_pin and e['clock'] in cidx]
        self.in_slew = {port_pin[int(c)]: v for c, v in info.get('input_transition', {}).items() if int(c) in port_pin}
        # min analysis (set_input_transition -min; one set without -max sets both, an unset slew is 0); an IR attached
        # before the min slews were kept has the max ones only
        self.in_slew_min = self.in_slew if 'input_transition_min' not in info else \
            {port_pin[int(c)]: v for c, v in info['input_transition_min'].items() if int(c) in port_pin}
        rc = info.get('rc') or {}
        self.r_sig, self.c_sig = (rc.get('signal') or [0.0, 0.0])
        self.cut_res = float(rc.get('cut_res', 0.0))
        # clock wire RC (set_wire_rc -clock) on the non-leaf clock nets, as OpenROAD's estimate_parasitics
        # (GlobalRouter::isNonLeafClock: a net of USE CLOCK none of whose pins is a register clock pin); the rest
        # signal RC
        self.r_clk, self.c_clk = (rc.get('clock') or [self.r_sig, self.c_sig])
        regclk = np.zeros(P, bool)
        regclk[fr[np.isin(atype, (T_RISE_EDGE, T_FALL_EDGE))]] = True
        self.net_clkrc = self._nonleaf_clock_nets(d, p_net, regclk)
        rv_sig = self._pin_vias(d, info, rc, p_cell, p_mpin, io_mask)
        if self.net_clkrc.any() and rc.get('clock_layers'):
            rv_clk = self._pin_vias(d, info, rc, p_cell, p_mpin, io_mask, layers=rc['clock_layers'])
            rv_sig = np.where(self.net_clkrc[p_net], rv_clk, rv_sig)
        self.pin_rvia = T(rv_sig, torch.float32)
        self.p_cell_np = p_cell
        self.pi = None
        self.pin_net = d.p_net
        drv_t = torch.as_tensor(drv, device=dev)
        self.drv_nets = torch.nonzero(drv_t >= 0).flatten(); self.drv_pins = drv_t[self.drv_nets]
        self._prepare_static()
        self._wire_setup()
        if parasitics == 'global_routing':
            self._gr_setup()
        self._graph = None
        self._ws = None                              # placement wires for the CUDA graph: (cw, elm, pi), static
        self.use_graph = self.dev.type == 'cuda'
        from . import dmp
        self.steps = dmp.EAGER
        self.triton = None
        self.host = None                        # on the CPU: the delay calculation in C++ (attar.timing.dmp_host)
        if self.dev.type == 'cpu':
            from . import dmp_host
            if dmp_host.available():
                self.host = dmp_host
                self.host_tables = dmp_host.Tables(self.t_ix1, self.t_ix2, self.t_val, self.t_n)
        self.fused = True                       # with Triton: one gate + one load kernel per level (False: per-op path)
        if self.dev.type == 'cuda':
            try:
                from . import dmp_triton
                self.triton = dmp_triton
                self.t_n32 = self.t_n.to(torch.int32).contiguous()
                self._prepare_triton()
            except Exception:  # noqa: BLE001  (no Triton: the PyTorch reference path)
                self.triton = None
            if self.triton is not None:
                _first_compile_notice()
        self._mp_names = None; self._sink_drv = None

    def _nonleaf_clock_nets(self, d, p_net, regclk):
        """Per net: USE CLOCK and no register clock pin on it (GlobalRouter::isNonLeafClock; LibertyPort::isRegClk:
        the from pin of a clock-to-output or latch-enable arc, regclk)."""
        from ..io.distill import N_CLOCK
        a = d.raw
        E = len(a['n_ptr']) - 1
        if 'n_flags' not in a or not ((a['n_flags'] & N_CLOCK) > 0).any():
            return np.zeros(E, bool)
        has_reg = np.zeros(E, bool)
        has_reg[p_net[regclk]] = True
        return ((a['n_flags'] & N_CLOCK) > 0) & ~has_reg

    @staticmethod
    def _pin_library(t, info, p_cell, p_mpin):
        """Per pin, from its master pin's Liberty data: capacitance (rise, fall; max and min analysis), the macro
        clock-tree path (min / max_clock_tree_path) and the input thresholds; set_load for output ports.
        Returns numpy (cap (P, 2), cap_min (P, 2), ctp (P, 2, 2), thresholds (P, 2, 4))."""
        P = len(p_cell)
        ok = p_mpin >= 0
        # pin capacitance (rise, fall): Liberty for cell pins, set_load for output ports
        cap = np.zeros((P, 2), np.float32)
        cap[ok] = t['mp_cap'][p_mpin[ok]]
        cap_min = cap.copy()
        if 'mp_cap_min' in t:
            cap_min[ok] = t['mp_cap_min'][p_mpin[ok]]
        loads = info.get('load', {})
        if loads:                                 # one lookup per pin (a mask per port was O(ports x pins))
            lv = np.full((int(p_cell.max()) + 1, 2), np.nan, np.float32)
            for c, v in loads.items():
                if int(c) < len(lv):
                    lv[int(c)] = v                # a scalar, or (rise, fall)
            hit = ~np.isnan(lv[p_cell, 0])
            cap[hit] = lv[p_cell[hit]]; cap_min[hit] = lv[p_cell[hit]]
        # macro clock pins: Liberty min / max_clock_tree_path; OpenSTA's ideal-clock capture time is the clock edge minus
        # the min value for setup checks and minus the max value for hold checks (launch unchanged)
        ctp = np.zeros((P, 2, 2), np.float32)
        if 'mp_ctp' in t:
            ctp[ok] = t['mp_ctp'][p_mpin[ok]]
        # thresholds of loads per pin (ports: the first library read)
        sthr = np.tile(_default_thresholds(info), (P, 1, 1))
        if 'mp_thr' in t:
            sthr[ok] = t['mp_thr'][p_mpin[ok]]
        return cap, cap_min, ctp, sthr

    def _pin_vias(self, d, info, rc, p_cell, p_mpin, io_mask, layers=None):
        """Per pin, the resistance between it and its net's estimated tree (EstimateParasitics::
        parasiticNodeConnectPins): with set_wire_rc on a layer and layer RC given, the vias from the pin's layer
        (a cell pin's LEF layer, a port's DEF layer) to the wire's first layer, one per cut layer, each its
        set_layer_rc -via value (at least 1e-3 ohm; insertViaResistances); else the average cut resistance of the
        routing layers for every pin (computeAverageCutResistance)."""
        P = len(p_cell)
        layers = layers or rc.get('signal_layers')
        if not layers or not (rc.get('layer') or rc.get('via')):
            return np.full(P, self.cut_res, np.float32)
        num = lambda name: int(re.sub(r'\D', '', name)) if re.search(r'\d', name or '') else None   # noqa: E731
        via = {num(v): r for v, r in rc.get('via', {}).items() if re.fullmatch(r'(?i)v(?:ia)?\d+', v)}
        tree = num(layers[0])
        def stack(k):
            if k is None or tree is None:
                return self.cut_res
            lo, hi = min(k, tree), max(k, tree)
            return sum(max(via.get(i, 0.0), 1e-6) for i in range(lo, hi)) if hi > lo else 1e-6
        names = d.names.get('layers') or []
        m_lay = d.raw['m_pin_layer'] if 'm_pin_layer' in d.raw else None
        lut = np.array([stack(num(n)) for n in names] + [self.cut_res], np.float32)   # per layer; last: unknown
        out = np.full(P, self.cut_res, np.float32)
        ok = p_mpin >= 0
        if m_lay is not None:
            ml = m_lay.astype(np.int64)[p_mpin[ok]]
            out[ok] = lut[np.where((ml >= 0) & (ml < len(names)), ml, len(names))]
        io_l = info.get('io_layer') or {}
        for p in np.nonzero(io_mask)[0]:
            l = io_l.get(str(int(p_cell[p])))
            if l is not None:
                out[p] = stack(num(l))
        return out

    def _constants(self, d, info, t, p_cell, p_mpin, pdir, p_net, n_ptr, c_master, fr, to, atype, lib_id, asense):
        """Constant propagation (OpenSTA's sim): nets driven by tie cells, then every output whose function is
        constant given its constant inputs. Then on the instances with constant inputs, OpenSTA's Sim::findDisabledEdges:
        an arc is off when the output's function, under the constants, does not depend on its input (functionSense
        none); its sense becomes the function's (positive / negative unate) where that is narrower than the arc's; a
        conditional arc is off when its `when` is 0 under the constants, the unconditional (default) arc between the
        same pins when another one's `when` is 1 (Sim::isDisabledCond). Returns (arcs to disable, arc senses).
        Sets self.const_net (net -> 0 / 1 / -1)."""
        from itertools import product
        from ..io.liberty import parse_function
        E = len(n_ptr) - 1
        self.const_net = cn = np.full(E, -1, np.int64)
        off = np.zeros(len(fr), bool)
        self.cond_off = np.zeros(len(fr), bool)          # off by a `when` condition under the constants
        asense = asense.copy()
        fdefs = info.get('functions') or {}
        if not fdefs:
            return off, asense
        names = [x for ps in d.names['master_pins'] for x in ps]
        funcs = {int(m): {pn: parse_function(f) for pn, f in fs.items()} for m, fs in fdefs.items()}
        o = np.argsort(p_cell, kind='stable'); cptr = np.searchsorted(p_cell[o], np.arange(len(c_master) + 1))
        pins_of = lambda c: o[cptr[c]:cptr[c + 1]]
        pin_memo, out_memo = {}, {}      # per instance: its pins; per (master, constant inputs): constant outputs
        ties = {}                        # per instance: {pin on a supply net: 1 / 0} (OpenSTA: a constant pin)
        def evaluate(c):
            """Constant outputs of instance c: {pin: value}, and {input pin name: its constant}."""
            pins = pin_memo.get(c)
            if pins is None:
                pins = pin_memo[c] = {names[p_mpin[p]]: p for p in pins_of(c) if p_mpin[p] >= 0}
            known = {n: int(cn[p_net[p]]) for n, p in pins.items() if cn[p_net[p]] >= 0 and pdir[p] != D_OUT}
            known.update(ties.get(c, ()))
            key = (int(c_master[c]), tuple(sorted(known.items())))
            res = out_memo.get(key)
            if res is None:
                res = {}
                for pn, (vs, f) in funcs.get(key[0], {}).items():
                    unk = [v for v in vs if v not in known]
                    if len(unk) > 12:
                        continue
                    vals = {f(dict(known, **dict(zip(unk, bits)))) for bits in product((0, 1), repeat=len(unk))}
                    if len(vals) == 1:
                        res[pn] = vals.pop()
                out_memo[key] = res
            return {pins[pn]: v for pn, v in res.items() if pn in pins}, known, pins
        # seeds: tie cells (an output function without variables), instances with pins on a supply net
        queue = []
        touched = set()
        for c, k, v in zip(*(d.raw.get(x, ()) for x in ('tie_cell', 'tie_mpin', 'tie_val'))):
            ties.setdefault(int(c), {})[names[k]] = int(v)
        for c in ties:
            res, _, _ = evaluate(c); touched.add(c)
            for p, v in res.items():
                if cn[p_net[p]] < 0:
                    cn[p_net[p]] = v; queue.append(p_net[p])
        for m, fs in funcs.items():
            if all(not vs for vs, _ in fs.values()):
                for c in np.nonzero(c_master == m)[0]:
                    res, _, _ = evaluate(c)
                    for p, v in res.items():
                        if cn[p_net[p]] < 0:
                            cn[p_net[p]] = v; queue.append(p_net[p])
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
            mk = (id(expr), tuple(sorted(known.items())))
            if mk not in val_memo:
                val_memo[mk] = _value(expr, known)
            return val_memo[mk]
        val_memo, sense_memo = {}, {}   # per (expression | master, output, input; constant inputs)
        def _value(expr, known):
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
            # Sim::isDisabledCond (kept apart: OpenROAD's delay calculation skips these arcs too, see __init__)
            k = int(lib_id[a])
            if k in whens:
                self.cond_off[a] = value(whens[k], known) == 0
            else:
                key = (int(t['a_master'][k]), int(t['a_from'][k]), int(t['a_to'][k]))
                self.cond_off[a] = any(value(w, known) == 1 for w in conds.get(key, ()))
            if cn[p_net[y]] >= 0 or y in res:
                off[a] = True; continue
            xn = names[p_mpin[x]]
            if xn in known:
                off[a] = True; continue
            fn = funcs.get(int(c_master[c]), {}).get(names[p_mpin[y]])
            if fn is not None:
                mk = (int(c_master[c]), names[p_mpin[y]], xn, tuple(sorted(known.items())))
                if mk not in sense_memo:
                    vs, f = fn
                    unk = [v for v in vs if v not in known and v != xn]
                    inc = dec = None
                    if len(unk) <= 12:
                        inc = dec = True
                        for bits in product((0, 1), repeat=len(unk)):
                            e = dict(known, **dict(zip(unk, bits)))
                            f0, f1 = f(dict(e, **{xn: 0})), f(dict(e, **{xn: 1}))
                            inc &= f1 >= f0; dec &= f1 <= f0
                    sense_memo[mk] = (inc, dec)
                inc, dec = sense_memo[mk]
                if inc is not None:
                    if inc and dec:                                # functionSense none: no dependence
                        off[a] = True; continue
                    if inc != dec:                             # positive / negative unate under the constants
                        s_sim = S_POS if inc else S_NEG
                        if asense[a] == S_NON:
                            asense[a] = s_sim; n_sense += 1
                        elif asense[a] != s_sim:               # the arc's transitions are all excluded
                            off[a] = True; continue
            off[a] = self.cond_off[a]
        self.n_const_nets = int((cn >= 0).sum()); self.n_disabled_arcs = int(off.sum()); self.n_sim_sense = n_sense
        return off, asense

    @staticmethod
    def _latch_d_to_q(d, t, lib_id, atype):
        """Per instance arc: a latch D -> Q arc, as OpenSTA's LibertyBuilder (a combinational arc into an output whose
        function is a latch's state variable, from a pin of the latch's data_in)."""
        import re
        names = d.names
        m_ptr = d.raw['m_pin_ptr'].astype(np.int64)
        fdefs = timing_mod.arrays(d)[1]
        seqs = {}
        if 'library' in d.manifest:                   # the timer's own library chain (no 'library' fact needed)
            for L in timing_mod.chain(d.path, d.manifest):
                for cn, v in (L.meta.get('sequentials') or {}).items():
                    seqs.setdefault(cn, v)
        if not seqs:                                  # timing.safetensors designs: no sequential data, no latch arcs
            return np.zeros(len(lib_id), bool)
        tok = lambda e: set(re.findall(r'[A-Za-z_][A-Za-z0-9_\[\]]*', e or ''))  # noqa: E731
        lib_dq = np.zeros(len(t['a_master']), bool)
        for a in np.nonzero(t['a_type'] == T_COMB)[0]:
            m = int(t['a_master'][a])
            sq = [q for q in (seqs.get(names['masters'][m]) or {}).get('seq', []) if q['kind'] in ('latch', 'latch_bank')]
            if not sq:
                continue
            pins = names['master_pins'][m]
            fp, tp = pins[int(t['a_from'][a]) - m_ptr[m]], pins[int(t['a_to'][a]) - m_ptr[m]]
            tf = tok((fdefs.get(str(m)) or {}).get(tp))
            lib_dq[a] = any(fp in tok(q['data']) and tf & set(q['vars']) for q in sq)
        return lib_dq[lib_id] & (atype == T_COMB)

    @staticmethod
    def _levelize(P, fr, to, prefer=None):
        """Longest-path levels of a DAG (Kahn by layers). Edges inside cycles are dropped (count returned): those
        marked prefer first (latch D -> Q arcs, as OpenSTA's levelization), the rest only if cycles remain."""
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
                cand, cnt = np.unique(nx, return_counts=True)
                indeg[cand] -= cnt                             # np.subtract.at(indeg, nx, 1), without ufunc.at
                cur = cand[indeg[cand] == 0]
                k += 1
            if done.all():
                return lev, dropped
            bad = ~done[to] & ~done[fr]
            if prefer is not None and bool((bad & prefer).any()):
                bad &= prefer
            else:
                bad = Timer._back_edges(P, fr, to, bad)
            dropped += int(bad.sum())
            fr, to = fr[~bad], to[~bad]
            if prefer is not None:
                prefer = prefer[~bad]

    @staticmethod
    def _loop_edges(P, n_fr, n_to, fr, to, atype, ldq, p_cell, p_mpin):
        """The edges OpenSTA's levelization disables to break combinational loops (Levelize::findBackEdges,
        findCycleBackEdges): (per instance arc, per net edge) masks. Its depth-first search is replayed in its order:
        the roots (no fanin) by vertex id, i.e. instance (DEF component) order then the instance's pins in LEF order,
        ports last; a vertex's out edges newest first (Graph::makeEdge prepends): a driver's wires to its net's ports
        then to its instance pins in reverse DEF order (definNet::end leaves a net's pins in DEF order, the wires are
        made in that order, the ports' after), an instance pin's arcs in reverse Liberty order. The graph: wires and every instance arc but timing checks,
        latch D -> Q and preset / clear (Levelize::searchThru). Only run when there is a cycle. (Under 100 roots
        OpenSTA sorts them by name: not replayed.)"""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        keep = ~np.isin(atype, (T_SETUP_R, T_HOLD_R, T_SETUP_F, T_HOLD_F, T_PRESET, T_CLEAR)) & ~ldq
        ia = np.nonzero(keep)[0]
        NN = len(n_fr)
        e_fr = np.concatenate([n_fr, fr[ia]]); e_to = np.concatenate([n_to, to[ia]])
        loop_arc = np.zeros(len(fr), bool); loop_net = np.zeros(NN, bool)
        g = coo_matrix((np.ones(len(e_fr)), (e_fr, e_to)), shape=(P, P)).tocsr()
        _, comp = connected_components(g, directed=True, connection='strong')
        if np.bincount(comp).max(initial=0) < 2 and not (e_fr == e_to).any():
            return loop_arc, loop_net
        # out edges in OpenSTA's iteration order
        is_port = p_mpin < 0
        k1 = np.concatenate([np.zeros(NN, np.int64), np.ones(len(ia), np.int64)])           # wires first
        k2 = np.concatenate([(~is_port[n_to]).astype(np.int64), np.zeros(len(ia), np.int64)])
        k3 = np.concatenate([-n_to.astype(np.int64), -ia.astype(np.int64)])
        o = np.lexsort((k3, k2, k1, e_fr))
        e_fr, e_to = e_fr[o], e_to[o]
        eid = np.arange(len(o))[o]                                   # the edge's index in (net edges, kept arcs)
        ptr = np.searchsorted(e_fr, np.arange(P + 1))
        rank = np.empty(P, np.int64); rank[np.lexsort((p_mpin, p_cell))] = np.arange(P)
        indeg = np.bincount(e_to, minlength=P); outdeg = np.diff(ptr)
        roots = np.nonzero((indeg == 0) & (outdeg > 0))[0]
        roots = roots[np.argsort(rank[roots], kind='stable')]
        visited = np.zeros(P, bool); on_path = np.zeros(P, bool)
        back = []
        ptr_l, to_l = ptr.tolist(), e_to.tolist()

        def dfs(stack):
            while stack:
                v, k = stack[-1]
                if k < ptr_l[v + 1]:
                    stack[-1] = (v, k + 1)
                    w = to_l[k]
                    if not visited[w]:
                        visited[w] = True; on_path[w] = True
                        stack.append((w, ptr_l[w]))
                    elif on_path[w]:
                        back.append(k)
                else:
                    stack.pop(); on_path[v] = False
        # every root is pushed (and on the path) at the start: the search starts from the last one (findBackEdges)
        for v in roots.tolist():
            visited[v] = True; on_path[v] = True
        dfs([(v, ptr_l[v]) for v in roots.tolist()])
        # cycles no root reaches (findCycleBackEdges): a search from each vertex left (its start vertex not flagged)
        left = np.nonzero(~visited)[0]
        for v in left[np.argsort(rank[left], kind='stable')].tolist():
            dfs([(v, ptr_l[v])])
        b = eid[np.asarray(back, np.int64)] if back else np.zeros(0, np.int64)
        loop_net[b[b < NN]] = True
        loop_arc[ia[b[b >= NN] - NN]] = True
        return loop_arc, loop_net

    @staticmethod
    def _back_edges(P, fr, to, cand):
        """The edges that close cycles (OpenSTA's Levelize::findBackEdges: a depth-first search, every edge to a pin
        on the current path is a disabled loop edge), searched inside each strongly connected component of the
        candidate edges cand only: the rest of the graph stays. The search starts at the component's pins entered from
        outside it (then the others), in pin order: OpenSTA's order (roots by name, the network's edge order) is not
        reproduced, so the edge cut in a loop can differ."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        ci = np.nonzero(cand)[0]
        g = coo_matrix((np.ones(len(ci)), (fr[ci], to[ci])), shape=(P, P)).tocsr()
        _, comp = connected_components(g, directed=True, connection='strong')
        size = np.bincount(comp, minlength=comp.max() + 1)
        inner = ci[(comp[fr[ci]] == comp[to[ci]]) & (size[comp[fr[ci]]] > 1)]
        o = np.argsort(fr[inner], kind='stable'); inner = inner[o]
        f2 = fr[inner]; t2 = to[inner]
        nodes = np.unique(f2)
        ptr = {int(v): (int(a), int(b)) for v, a, b in zip(nodes, np.searchsorted(f2, nodes),
                                                             np.searchsorted(f2, nodes, side='right'))}
        entered = set(to[~((comp[fr] == comp[to]) & (size[comp[to]] > 1))].tolist())
        starts = sorted(nodes.tolist(), key=lambda v: (v not in entered, v))
        visited, on_path = set(), set()
        back = np.zeros(len(fr), bool)
        for s0 in starts:
            if s0 in visited:
                continue
            visited.add(s0); on_path.add(s0)
            stack = [(s0, ptr[s0][0])]
            while stack:
                v, k = stack[-1]
                hi = ptr.get(v, (0, 0))[1]
                if k < hi:
                    stack[-1] = (v, k + 1)
                    w = int(t2[k])
                    if w not in visited:
                        visited.add(w); on_path.add(w)
                        stack.append((w, ptr.get(w, (0, 0))[0]))
                    elif w in on_path:
                        back[inner[k]] = True
                else:
                    stack.pop(); on_path.discard(v)
        return back

    def _group(self, lev):
        if not len(lev):
            return [None] * (self.max_level + 1)
        o = np.argsort(lev, kind='stable')
        b = np.searchsorted(lev[o], np.arange(self.max_level + 2))
        return [torch.as_tensor(o[b[k]:b[k + 1]], device=self.dev) for k in range(self.max_level + 1)]

    # -- wires ------------------------------------------------------------------------------------------------------
    def _wire_setup(self):
        """Placement-independent parts of the wire model: the pins' locations (_pin_setup), Steiner (RMST) buckets
        and the tree plan, so that wires() has no host sync (CUDA-graph capturable)."""
        from ..evaluator import Evaluator
        ev = self.ev = Evaluator(self.d)
        ev.off = ev.off.clone()                         # the timer's own pin offsets (_pin_setup), updated in place
        self._pin_setup()
        drv = torch.as_tensor(self.net_driver, device=self.dev)
        drv_local = torch.where(drv >= 0, drv - ev.ptr[:-1], torch.full_like(drv, -1))
        self.plan_clk = None
        if self.net_clkrc.any():                    # non-leaf clock nets: their own plan, timed with clock RC
            nc = torch.as_tensor(self.net_clkrc, device=self.dev)
            self.net_clkrc_t = nc
            self._plan_clk_drv = torch.where(nc, drv_local, torch.full_like(drv_local, -1))
            self.plan_clk = tree_plan(ev.ptr, ev.deg, self._plan_clk_drv)
            drv_local = torch.where(nc, torch.full_like(drv_local, -1), drv_local)
        self.plan = tree_plan(ev.ptr, ev.deg, drv_local)
        self._plan_drv = drv_local                  # the drivers the plans were made with (Timer.wires: a plan of
        self._w = None                              # the nets that changed); the last wires and their inputs
        self._big = None
        big = self.plan['big']
        from . import steiner_big
        if (big is not None or self.exact_trees) and steiner_big.available():
            # nets above the GPU buckets: OpenROAD's tree on the host each run (exact), into static buffers; only
            # those that can carry a data arrival (clock nets are ideal, the rest keep the lumped model: unused)
            ptr, deg = ev.ptr.cpu().numpy(), ev.deg.cpu().numpy()
            if self.exact_trees:
                bn = np.nonzero((drv_local.cpu().numpy() >= 0) & (deg >= 2))[0]
            else:
                bn = big.cpu().numpy()
            if self.dev.type == 'cpu' and self.exact_trees:
                # on the host every net with a driver gets its exact tree here (the C++ port is far faster than the
                # batched PyTorch buckets on a CPU), and the buckets keep none of them
                bn = bn[~self.net_clkrc[bn]]
                dl = drv_local.clone(); dl[torch.as_tensor(bn, device=self.dev, dtype=torch.long)] = -1
                self.plan = tree_plan(ev.ptr, ev.deg, dl)
                self._plan_drv = dl
            else:
                reach = self._reachable()
                bn = bn[reach[self.net_driver[bn]] & ~self.net_clkrc[bn]]      # (clock RC nets: plan_clk)
            big = torch.as_tensor(bn, device=self.dev)
        if big is not None and len(big) and steiner_big.available():
            cnt = deg[bn]
            pins = np.repeat(ptr[bn] - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt) + np.arange(int(cnt.sum()))
            caps = torch.cat([self.pin_cap, self.pin_cap_min], 1).cpu().numpy()
            pins_t = torch.as_tensor(pins, device=self.dev)
            self._big = {'nets': bn, 'ptr': np.concatenate([[0], np.cumsum(deg[bn])]), 'drv': drv_local.cpu().numpy()[bn],
                         'cap': caps[pins], 'rv': self.pin_rvia.cpu().numpy()[pins], 'cell': ev.cell[pins_t],
                         'off': ev.off[pins_t], 'pins': pins_t}
            self.plan['big_buf'] = (big, pins_t, torch.zeros(len(pins), 4, device=self.dev),
                                    torch.zeros(len(bn), 4, 3, device=self.dev), torch.zeros(len(bn), device=self.dev))
        elif big is not None and not steiner_big.available():
            self._limit('%d nets above %d pins use a lumped wire model (%s)'
                        % (len(big), TREE_BUCKETS[-1][1], steiner_big.error()))
        self.big_steiner = ev.deg > M.RMST_MAX_DEGREE
        self._iclk_pi = None
        if self.net_iclk.any():
            from .drv import SegSum
            ni = torch.as_tensor(self.net_iclk, device=self.dev)
            self.net_iclk_t = ni
            tot = SegSum(self.d.p_net, ev.E)(torch.cat([self.pin_cap, self.pin_cap_min], 1))     # (E, 4)
            self._iclk_pi = torch.stack([tot, torch.zeros_like(tot), torch.zeros_like(tot)], -1)

    def _gr_setup(self):
        """Global-routing parasitics: the networks of the routes (attar.timing.grpar.build), static buffers for their
        reduction (filled on the host before each run, _gr_update), the lumped pin capacitance for unrouted nets."""
        from . import grpar
        rc = self.info.get('rc') or {}
        via_res = self.via_res
        if via_res is None:                          # the cut layers' LEF resistance (routing fact), as OpenROAD falls back to
            cr = (self.d.manifest.get('routing') or {}).get('cut_res')
            via_res = {int(k): v for k, v in cr.items()} if cr else None
        self._gr_rc, self._gr_via = rc, via_res
        self._gr_net = grpar.build(self.d, self.routes, rc, via_res)
        self._gr_sig = grpar.net_signatures(self.d, self.routes)
        E, P = len(self.d.raw['n_ptr']) - 1, len(self.d.raw['p_cell'])
        dev = self.dev
        self._gr_cw64 = self._gr_net['wire_cap'].astype(np.float64)
        self._gr_cw = torch.as_tensor(self._gr_cw64, device=dev, dtype=torch.float32)
        self._gr_elm = torch.zeros(P, 4, device=dev); self._gr_pi = torch.zeros(E, 4, 3, device=dev)
        self._gr_elm64 = None                       # the reduction (float64), per pin and per net: _gr_update
        self._gr_key = None
        self.gr_loops = 0

    def _gr_update(self):
        """Reduce the routes' networks at the pins' present capacitances, into the static buffers."""
        key = self._gr_caps_key()
        if key == self._gr_key:                     # the pin capacitances have not changed since the last reduction
            return
        caps_t = torch.cat([self.pin_cap, self.pin_cap_min], 1)
        self._gr_key = key
        from . import grpar
        if self._gr_net is None:                    # patched by set_routes: the whole network again
            self._gr_net = grpar.build(self.d, self.routes, self._gr_rc, self._gr_via)
            self._gr_cw64 = self._gr_net['wire_cap'].astype(np.float64)
            self._gr_cw.copy_(torch.from_numpy(self._gr_cw64).float())
        caps = caps_t.double().cpu().numpy()
        self._gr_caps = caps
        E = len(self._gr_cw64)
        self._gr_tot = np.zeros((E, caps.shape[1])); np.add.at(self._gr_tot, np.asarray(self.d.p_net.cpu().numpy()), caps)
        elm, pi, _ = grpar.reduce(self._gr_net, np.asarray(self.net_driver), caps)
        self._gr_fill(pi, None)
        self._gr_elm64, self._gr_pi64 = elm, pi
        self._gr_loop = grpar.loop_nets(self._gr_net, len(pi))
        self.gr_loops = int(self._gr_loop.sum())
        self._gr_elm.copy_(torch.from_numpy(elm).float()); self._gr_pi.copy_(torch.from_numpy(pi).float())

    def _gr_caps_key(self):
        """Identifies the pin capacitances (resized() updates them in place: the version counters change)."""
        return (id(self.pin_cap), self.pin_cap._version, id(self.pin_cap_min), self.pin_cap_min._version)

    def _gr_fill(self, pi, nets):
        """Nets without a network (unrouted): their pins' capacitance, lumped (pi rows still NaN; nets: a mask of
        the rows to consider, or None for all)."""
        tot = self._gr_tot
        miss = np.isnan(pi[:, 0, 0])
        if nets is not None:
            miss &= nets
        pi[miss] = np.stack([tot[miss], np.zeros_like(tot[miss]), np.zeros_like(tot[miss])], -1)

    def set_routes(self, routes):
        """Time on other routes of the same netlist: only the nets whose route or pins changed (grpar.net_signatures)
        get their networks built and reduced again; every other net keeps its parasitics, which are bit for bit
        what a new timer on these routes would compute. Returns the number of nets redone."""
        from . import grpar
        if self.parasitics != 'global_routing':
            raise ValueError('set_routes: this timer uses placement parasitics')
        sig = grpar.net_signatures(self.d, routes)
        dirty = sig != self._gr_sig
        self.routes, self._gr_sig = routes, sig
        n = int(dirty.sum())
        if n == 0:
            return 0
        if self._gr_elm64 is None or self._gr_key != self._gr_caps_key():
            self._gr_net, self._gr_key = None, None      # never reduced, or the pin capacitances changed: all again
            return n
        net = grpar.build(self.d, routes, self._gr_rc, self._gr_via, nets=dirty)
        caps = self._gr_caps
        elm, pi, _ = grpar.reduce(net, np.asarray(self.net_driver), caps)
        self._gr_fill(pi, dirty)
        pins = dirty[np.asarray(self.d.p_net.cpu().numpy())]
        self._gr_elm64[pins] = elm[pins]
        self._gr_pi64[dirty] = pi[dirty]
        self._gr_cw64[dirty] = net['wire_cap'][dirty]
        self._gr_loop[dirty] = grpar.loop_nets(net, len(dirty))[dirty]
        self.gr_loops = int(self._gr_loop.sum())
        self._gr_net = None                              # no longer the whole network (_gr_update builds it if needed)
        self._gr_elm.copy_(torch.from_numpy(self._gr_elm64).float())
        self._gr_pi.copy_(torch.from_numpy(self._gr_pi64).float())
        self._gr_cw.copy_(torch.from_numpy(self._gr_cw64).float())
        return n

    def wires_gr(self):
        """Global-routing parasitics (the buffers _gr_update filled): per net the wire capacitance, per net arc the
        Elmore delay (ps). Ideal clock nets as in wires(): lumped pins, no wire."""
        pi, elm_pin, cw = self._gr_pi, self._gr_elm, self._gr_cw
        if self._iclk_pi is not None:
            ni = self.net_iclk_t
            elm_pin = torch.where(ni[self.pin_net][:, None], torch.zeros_like(elm_pin), elm_pin)
            pi = torch.where(ni[:, None, None], self._iclk_pi, pi)
            cw = torch.where(ni, torch.zeros_like(cw), cw)
        self.pi = pi
        return cw, elm_pin[self.n_to]

    def _pin_setup(self):
        """Where estimate_parasitics puts each pin: OpenROAD's dbNetwork::location, dbITerm::getAvgXY, the mean of the
        centers of the pin's LEF rectangles (not the center of their bounding box, the Evaluator's pin offset: they
        differ on pins of several rectangles, most ASAP7 pins), oriented with the cell and truncated to the database
        unit as OpenROAD's int(). From the library chain (attar.timing.ir.pin_locations); pins it does not locate (no
        chain, a bundle without lp_loc) and ports keep the Evaluator's offsets."""
        d = self.d
        loc = timing_mod.pin_locations(d)
        pm = d.raw['p_mpin'].astype(np.int64)
        ok = pm >= 0
        lp = np.full((len(pm), 2), np.nan)
        lp[ok] = loc[pm[ok]]
        ok &= np.isfinite(lp).all(1)
        self._loc_n = torch.as_tensor(np.nan_to_num(lp), dtype=torch.float64, device=self.dev)
        self._loc_ok = torch.as_tensor(ok, device=self.dev)
        self._pin_key = None
        self._sync_pins()

    def _sync_pins(self):
        """The pins' offsets from their cells' lower-left, into ev.off in place (a static input of the CUDA graph):
        again whenever the design's orientations or sizes change (Design.update_orient makes a new pin_off)."""
        d = self.d
        key = (id(d.pin_off), id(d.size))
        if key == self._pin_key:
            return
        self._pin_key = key
        from ..design import _orient_offset
        c = d.p_cell
        w, h = d.size_n[c, 0].double(), d.size_n[c, 1].double()
        o = _orient_offset(d.orient[c], self._loc_n[:, 0], self._loc_n[:, 1], w, h)
        o = torch.floor(o * d.dbu + 1e-3) / d.dbu        # OpenROAD truncates the mean; the cell origin is on the grid
        base = d.size[c] / 2 + d.pin_off                # the Evaluator's (box centers)
        self.ev.off.copy_(torch.where(self._loc_ok[:, None], o.to(base.dtype), base))
        b = getattr(self, '_big', None)
        if b is not None:
            b['off'].copy_(self.ev.off[b['pins']])

    def _reachable(self):
        """Pins a data arrival can reach (structural): from launch arcs, constrained inputs and the input ports a
        path delay exception starts from (set_max_delay -from), over net and combinational arcs, level by level."""
        g_fr, g_to, seeds = self._g_edges
        reach = np.zeros(self.P, bool)
        reach[seeds] = True
        reach[[p for p, _, _ in self.in_delay]] = True
        reach[sorted(getattr(self, 'ex_ports', ()))] = True
        o = np.argsort(self.level[g_fr], kind='stable')
        g_fr, g_to = g_fr[o], g_to[o]
        bounds = np.searchsorted(self.level[g_fr], np.arange(int(self.level.max(initial=0)) + 2))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b > a:
                reach[g_to[a:b][reach[g_fr[a:b]]]] = True       # np.logical_or.at, without ufunc.at
        return reach

    def _big_update(self, pos, nets=None):
        """The large nets' parasitics at pos, on the host (attar.timing.steiner_big), into plan['big_buf']. nets: a
        mask (E,) of the nets to redo (the others keep theirs); returns the mask of big_buf rows redone."""
        if self._big is None or self.parasitics != 'placement':
            return None
        from . import steiner_big
        b = self._big
        _, _, elm_b, pi_b, len_b = self.plan['big_buf']
        pos = self.d.pos if pos is None else pos
        if nets is None:
            q = torch.round((pos[b['cell']] + b['off']) * self.d.dbu).cpu().numpy()
            elm, pi, length = steiner_big.pd_rc(q[:, 0], q[:, 1], b['ptr'], b['drv'], b['cap'], b['rv'], self.r_sig,
                                                self.c_sig, self.d.dbu)
            elm_b.copy_(torch.from_numpy(elm)); pi_b.copy_(torch.from_numpy(pi)); len_b.copy_(torch.from_numpy(length))
            return None
        rows = nets.cpu().numpy()[b['nets']]                 # which big nets
        if not rows.any():
            return rows
        k = np.nonzero(rows)[0]
        cnt = np.diff(b['ptr'])[k]
        pin_rows = np.repeat(b['ptr'][k] - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt) + np.arange(int(cnt.sum()))
        pr = torch.as_tensor(pin_rows, device=self.dev)
        q = torch.round((pos[b['cell'][pr]] + b['off'][pr]) * self.d.dbu).cpu().numpy()
        elm, pi, length = steiner_big.pd_rc(q[:, 0], q[:, 1], np.concatenate([[0], np.cumsum(cnt)]), b['drv'][k],
                                            b['cap'][pin_rows], b['rv'][pin_rows], self.r_sig, self.c_sig, self.d.dbu)
        kt = torch.as_tensor(k, device=self.dev)
        elm_b[pr] = torch.from_numpy(elm).to(elm_b); pi_b[kt] = torch.from_numpy(pi).to(pi_b)
        len_b[kt] = torch.from_numpy(length).to(len_b)
        return rows

    def steiner(self, p):
        """Wire length of the nets without an estimate-parasitics tree (above the largest bucket, or no driver):
        min(single trunk, star), HPWL up to 3 pins. tree_elmore replaces it on every net it builds a tree for."""
        ev = self.ev
        lo, hi = M.net_bbox(p, ev.net, ev.E)
        s = torch.minimum(M.stt(p, ev.net, ev.deg, ev.ptr, ev.E, lo, hi), M.star(p, ev.net, ev.deg, ev.E))
        L = torch.where(ev.deg <= 3, (hi - lo).sum(1), s)
        return torch.where(ev.deg >= 2, L, torch.zeros_like(L))

    INCREMENTAL_WIRES = 0.3     # wires(): redo only the changed nets while they are at most this fraction of all

    def wires(self, pos=None):
        """Placement parasitics: per net the wire capacitance, per net arc the Elmore delay (ps). Only the nets with a
        pin that moved or changed (capacitance, via resistance: a resize) since the last call are computed again, the
        others keep theirs: each net's tree and reduction are its own, so the result is bit for bit a full one's."""
        p = self.ev._pins(pos)
        L = self.steiner(p)
        caps = torch.cat([self.pin_cap, self.pin_cap_min], 1)            # rise / fall max, rise / fall min
        E = self.ev.E
        w = self._w
        dirty = None
        if w is not None:
            pins = (p != w['p']).any(1) | (caps != w['caps']).any(1) | (self.pin_rvia != w['rv'])
            dirty = torch.zeros(E, dtype=torch.bool, device=self.dev)
            dirty[self.pin_net[pins]] = True
            if int(dirty.sum()) > self.INCREMENTAL_WIRES * E:
                dirty = None
        if dirty is None:                                                # every net
            self._big_update(pos)
            main = tree_elmore(p, self.plan, E, caps, self.r_sig, self.c_sig, L, self.d.dbu, rcut=self.pin_rvia)
            clk = None
            if self.plan_clk is not None:
                clk = tree_elmore(p, self.plan_clk, E, caps, self.r_clk, self.c_clk, L, self.d.dbu,
                                  rcut=self.pin_rvia)
            self._w = w = {'p': p.clone(), 'caps': caps.clone(), 'rv': self.pin_rvia.clone(), 'main': main,
                           'clk': clk}
        elif bool(dirty.any()):                                          # the nets that changed
            rows = self._big_update(pos, dirty)
            for key, drv0, r_, c_ in (('main', self._plan_drv, self.r_sig, self.c_sig),
                                       ('clk', getattr(self, '_plan_clk_drv', None), self.r_clk, self.c_clk)):
                if w[key] is None:
                    continue
                sub = tree_plan(self.ev.ptr, self.ev.deg, torch.where(dirty, drv0, torch.full_like(drv0, -1)))
                w[key][2][dirty] = L[dirty]
                tree_elmore(p, sub, E, caps, r_, c_, L, self.d.dbu, rcut=self.pin_rvia, into=w[key])
                if key == 'main' and rows is not None and rows.any():  # the large nets' host trees again
                    nets_b, pins_b, elm_b, pi_b, len_b = self.plan['big_buf']
                    rk = torch.as_tensor(rows, device=self.dev)
                    pk = rk.repeat_interleave(torch.as_tensor(np.diff(self._big['ptr']), device=self.dev))
                    out, pi_, tl = w['main']
                    out[pins_b[pk]] = elm_b[pk]; pi_[nets_b[rk]] = pi_b[rk]; tl[nets_b[rk]] = len_b[rk]
            w['p'].copy_(p); w['caps'].copy_(caps); w['rv'].copy_(self.pin_rvia)
        elm_pin, pi, L2 = w['main']
        cw = L2 * self.c_sig
        if self.plan_clk is not None:
            e2, pi2, L3 = w['clk']
            nc = self.net_clkrc_t
            elm_pin = torch.where(nc[self.pin_net][:, None], e2, elm_pin)
            pi = torch.where(nc[:, None, None], pi2, pi)
            cw = torch.where(nc, L3 * self.c_clk, cw)
        if self._iclk_pi is not None:                # ideal clock nets: no parasitics (estimate_parasitics skips
            ni = self.net_iclk_t                     # them): the pins' capacitance, lumped, no wire delay
            elm_pin = torch.where(ni[self.pin_net][:, None], torch.zeros_like(elm_pin), elm_pin)
            pi = torch.where(ni[:, None, None], self._iclk_pi, pi)
            cw = torch.where(ni, torch.zeros_like(cw), cw)
        self.pi = pi
        return cw, elm_pin[self.n_to]                                   # (A, 4): Elmore per variant

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
        ix_max, v_max, ix_min, v_min, sl_p, sl_v, sl_n = [], [], [], [], [], [], []
        wf = self.wf.cpu().numpy(); lat = self.lat.cpu().numpy()
        prop = self.prop_clk
        for p, ci, e in self.in_delay:
            edge = 1 if e['clock_fall'] else 0
            base = float(wf[ci, edge]) + (0.0 if e['network_latency_included'] or prop[ci] else float(lat[ci]))
            rfs = [0, 1] if not (e['rise'] or e['fall']) else ([0] if e['rise'] else [1])
            for rf in rfs:
                k = (p * 2 + rf) * NT + 2 * ci + edge
                if not e['min']:
                    ix_max.append(k); v_max.append(base + e['value'])
                if not e['max']:
                    ix_min.append(k); v_min.append(base + e['value'])
            sl_p.append(p); sl_v.append(self.in_slew.get(p, 0.0)); sl_n.append(self.in_slew_min.get(p, 0.0))
        # propagated clocks: the source's rise at the rising edge, its fall at the falling edge (+ source latency),
        # its slew the port's (set_input_transition)
        cell_of_pin = self.p_cell_np
        for ci, c in enumerate(self.clocks):
            if not prop[ci]:
                continue
            for p in np.nonzero(np.isin(cell_of_pin, c['sources']) & (self.d.raw['p_mpin'] < 0))[0].tolist():
                for rf in (0, 1):
                    k = (p * 2 + rf) * NT + 2 * ci + rf
                    v = float(wf[ci, rf]) + float(c.get('source_latency', 0.0))
                    ix_max.append(k); v_max.append(v); ix_min.append(k); v_min.append(v)
                sl_p.append(p); sl_v.append(self.in_slew.get(p, 0.0)); sl_n.append(self.in_slew_min.get(p, 0.0))
        seen = set(sl_p)
        for p in list(self.in_slew) + [q for q in self.in_slew_min if q not in self.in_slew]:
            if p not in seen and not self.clk_of[p] >= 0:   # set_input_transition sets a port's slew, input delay or not
                seen.add(p); sl_p.append(p); sl_v.append(self.in_slew.get(p, 0.0)); sl_n.append(self.in_slew_min.get(p, 0.0))
        self.in_ix_max, self.in_v_max = T(ix_max), T(v_max, torch.float32)
        self.in_ix_min, self.in_v_min = T(ix_min), T(v_min, torch.float32)
        self.in_sl_p, self.in_sl_v, self.in_sl_n = T(sl_p), T(sl_v, torch.float32), T(sl_n, torch.float32)
        if NT > 2 * C:                                  # exception tags: no clock relationship (NaN: not checked)
            pad = lambda m: torch.nn.functional.pad(m, (0, NT - m.shape[1], 0, NT - m.shape[0]), value=float('nan'))
            self.rel_setup, self.rel_hold = pad(self.rel_setup), pad(self.rel_hold)
        # exception startpoints: input ports (arrival = their input delay, else 0)
        in_d = {}
        for p, _ci, e in self.in_delay:
            if not e['min']:
                in_d[p] = max(in_d.get(p, -INF), e['value'])
        ex_ix, ex_v = [], []
        tp = np.nonzero(self.d.raw['p_mpin'] < 0)[0]
        port_pins = dict(zip(cell_of[tp].tolist(), tp.tolist()))
        self.ex_ports = set()                                      # where data starts too (_reachable)
        for i in range(len(self.ex)):
            for c in ex_from[i]:
                p = port_pins.get(c)
                if p is not None and self.clk_of[p] < 0:
                    self.ex_ports.add(p)
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
            cks = self.ctr[ck, (self.l_type[a_i] == T_FALL_EDGE).long()]     # the clock pin's slew
            lc = [(self.l_lib[a_i], self.l_to[a_i], o_i, 2 * ck + edge, self.wf[ck, edge] + self.lat[ck], cks)]
            lsrc = [a_i.cpu().numpy()]
            llat = [np.ones(len(a_i), bool)]                    # t0 includes the clock's latency
            lcell = cell_of[self.l_fr[a_i].cpu().numpy()]
            for i, e in enumerate(self.ex):
                m = T(np.nonzero(np.isin(lcell, list(ex_from[i])))[0])
                if len(m):
                    t0 = self.wf[ck[m], edge[m]] + (0.0 if e['ignore_clock_latency'] else self.lat[ck[m]])
                    lc.append((self.l_lib[a_i][m], self.l_to[a_i][m], o_i[m], torch.full_like(m, 2 * C + i), t0, cks[m]))
                    lsrc.append(lsrc[0][m.cpu().numpy()])
                    llat.append(np.full(len(m), not e['ignore_clock_latency']))
            self.lc = tuple(torch.cat([x[j] for x in lc]) for j in range(6))
            self._lc_src = np.concatenate(lsrc)                 # l-selection index of each launch entry
            self._lc_lat = np.concatenate(llat)
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
        csl, carr, ctype = self.c_sense_lib.cpu().numpy(), self.c_arr.cpu().numpy(), self.c_type.cpu().numpy()
        cto = self.c_to.cpu().numpy()
        allowed = lambda sense, irf, orf: ((sense == S_NON) | ((sense == S_POS) & (irf == orf))  # noqa: E731
                                           | ((sense == S_NEG) & (irf != orf)))
        combos = []
        for orf in (0, 1):
            for irf in (0, 1):
                # delay calculation: the Liberty sense (an edge arc: from its clock edge); the search: the sim's
                ok = (tn[cl, orf, 0] > 0) & np.where(np.isin(ctype, (T_COMB, T_TRISTATE)), allowed(csl, irf, orf), True)
                ok &= np.where(ctype == T_RISE_EDGE, irf == 0, np.where(ctype == T_FALL_EDGE, irf == 1, True))
                m = np.nonzero(ok)[0]
                arr = carr[m] & np.where(np.isin(ctype[m], (T_COMB, T_TRISTATE)), allowed(cs[m], irf, orf), True)
                combos.append(np.stack([m, np.full(len(m), orf), np.full(len(m), irf), arr], 1))
        cb = np.concatenate(combos) if combos else np.zeros((0, 4), np.int64)
        o = np.argsort(clev[cb[:, 0]], kind='stable'); cb = cb[o]
        b = np.searchsorted(clev[cb[:, 0]], np.arange(self.max_level + 2))
        self.cb_lev = []
        self._cb_src = []                                        # c-selection index of each entry, per level
        for k in range(self.max_level + 1):
            x = cb[b[k]:b[k + 1]]
            self._cb_src.append(x[:, 0] if len(x) else None)
            if len(x):
                a_ = T(x[:, 0])
                ea, en = expand(cto[x[:, 0]])
                self.cb_lev.append((self.c_lib[a_], self.c_fr[a_], self.c_to[a_], T(x[:, 1]), T(x[:, 2]), ea, en,
                                    T(x[:, 3].astype(bool), torch.bool)))
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
            self.k_cslew = self.ctr[ck, crf]               # its slew (ideal clock: set_clock_transition)
            self.k_ctp_min = self.pin_ctp[self.k_fr, 0, crf]; self.k_ctp_max = self.pin_ctp[self.k_fr, 1, crf]
            self.k_crf = crf
            self.k_prop = torch.as_tensor(self.prop_clk, device=dev)[ck] if C else torch.zeros_like(ck, dtype=torch.bool)
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
            self.kx_ignore_lat = T(np.concatenate([np.full(len(m), bool(e['ignore_clock_latency']))
                                                   for m, e in zip(ea, self.ex)]) if ea else np.zeros(0, bool),
                                   torch.bool)
        # output ports: (pin, transition, capture tag, offset) for setup and for hold
        su, ho = [], []
        unc_s, unc_h = self.unc_s.cpu().numpy(), self.unc_h.cpu().numpy()
        for p, ci, e in self.out_delay:
            edge = 1 if e['clock_fall'] else 0
            l_ = 0.0 if e['network_latency_included'] or prop[ci] else float(lat[ci])
            rfs = [0, 1] if not (e['rise'] or e['fall']) else ([0] if e['rise'] else [1])
            for rf in rfs:
                if not e['min']:
                    su.append((p, rf, 2 * ci + edge, l_ - e['value'] - float(unc_s[ci])))
                if not e['max']:
                    ho.append((p, rf, 2 * ci + edge, l_ - e['value'] + float(unc_h[ci])))
        mk = lambda L: (T([x[0] for x in L]), T([x[1] for x in L]), T([x[2] for x in L]), T([x[3] for x in L], torch.float32))
        out_d = {}
        for p, _ci, e in self.out_delay:
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
        self._pin_latency_setup()
        self._latch_prepare()

    def _pin_latency_setup(self):
        """Pin clock latencies (set_clock_latency on pins, timing.json 'pin_latency'), as OpenSTA's Search::mutateTag: where
        a propagated clock reaches such a pin its arrival becomes the ideal clock edge + source latency + the pin's
        latency (to_latency), and from there the clock is not propagated (no CRPR: Crpr requires propagated clock
        infos); the pin's slew stays the propagated one (ClkNetwork does not make it ideal). Each override is written
        into the arrivals after the pin's own arrival is final (the stage of its level), before any arc out of it.
        _lat_at: stage level (or 'end') -> (flat index, value) for the max and the min arrivals."""
        self._lat_pins = np.zeros(0, np.int64)
        self._lat_at = {}
        self._ideal_pin_lat = None
        pl = self.info.get('pin_latency') or []
        if not pl:
            return
        NT, dev = self.NT, self.dev
        wf = self.wf.cpu().numpy()
        cidx = {k: i for i, k in enumerate(self.clock_names)}
        src_lat = [float(c.get('source_latency', 0.0)) for c in self.clocks]
        slot, skipped = {}, 0
        ideal = np.full(self.P, np.nan)
        for e in pl:
            p = int(e['pin'])
            cis = [cidx[e['clock']]] if e.get('clock') in cidx else ([int(self.clk_of[p])] if self.clk_of[p] >= 0 else [])
            for ci in cis:
                if not self.prop_clk[ci]:
                    # an ideal clock's pin latency (OpenSTA: the ideal clock network delay at the pin is the pin's
                    # latency instead of the clock's, at launch and at capture; e.g. pre-CTS timing of an SDC with
                    # useful-skew pin latencies). Modeled for -rise and -fall, -min and -max alike
                    if sorted(e['rf']) == [0, 1] and sorted(e['mm']) == ['max', 'min']:
                        ideal[p] = float(e['value'])
                    else:
                        skipped += 1
                    continue
                for rf in e['rf']:
                    edge = rf ^ int(self.par[p])
                    v = float(wf[ci, edge]) + src_lat[ci] + float(e['value'])
                    s_ = slot.setdefault((p, rf, 2 * ci + edge), [np.nan, np.nan])
                    for mm in e['mm']:
                        s_[0 if mm == 'max' else 1] = v
        if skipped:
            self._limit('%d pin clock latencies on ideal clocks with -rise / -fall / -min / -max are not modeled'
                        % skipped)
        if np.isfinite(ideal).any():
            self._ideal_pin_lat = torch.as_tensor(ideal, device=dev, dtype=torch.float32)
            if self.lc is not None:                     # launch: the pin's latency replaces the clock's
                lib, to, orf, tag, t0, cks = self.lc
                fr = self.l_fr[torch.as_tensor(self._lc_src, device=dev)]
                v = self._ideal_pin_lat[fr]
                ok = torch.isfinite(v) & torch.as_tensor(self._lc_lat, device=dev)
                ck = torch.as_tensor(self.clk_of, device=dev)[fr].clamp(min=0)
                self.lc = (lib, to, orf, tag, torch.where(ok, t0 - self.lat[ck] + v, t0), cks)
        if not slot:
            return
        keys = np.array(list(slot), np.int64).reshape(-1, 3)
        vals = np.array(list(slot.values()), np.float64).reshape(-1, 2)
        pins = keys[:, 0]
        self._lat_pins = np.unique(pins)
        # a pin with clock pins after it (not a register clock pin): only its own arrival is overridden here
        leaf = np.zeros(self.P, bool)
        leaf[self.k_fr.cpu().numpy()] = True
        if len(self.l_fr):
            leaf[self.l_fr.cpu().numpy()] = True
        lf = self.c_fr[self.c_type == T_RISE_EDGE].cpu().numpy(), self.c_fr[self.c_type == T_FALL_EDGE].cpu().numpy()
        leaf[np.concatenate(lf)] = True
        if (~leaf[self._lat_pins]).any():
            self._limit('%d pin clock latencies on pins that are not register clock pins: the clock after them '
                        'stays propagated' % int((~leaf[self._lat_pins]).sum()))
        stage_lv = np.array([k for k in range(self.max_level + 1) if self.cb_lev[k] is not None], np.int64)
        pos = np.searchsorted(stage_lv, self.level[pins])          # the first stage at or after the pin's level
        at = np.where(pos < len(stage_lv), stage_lv[np.minimum(pos, max(len(stage_lv) - 1, 0))], -1)
        flat = (keys[:, 0] * 2 + keys[:, 1]) * NT + keys[:, 2]
        for a in np.unique(at):
            m = at == a
            mx, mn = m & np.isfinite(vals[:, 0]), m & np.isfinite(vals[:, 1])
            T_ = lambda x, dt=torch.long: torch.as_tensor(x, device=dev, dtype=dt)  # noqa: E731
            self._lat_at['end' if a < 0 else int(a)] = (T_(flat[mx]), T_(vals[mx, 0], torch.float32),
                                                         T_(flat[mn]), T_(vals[mn, 1], torch.float32))

    def _latch_prepare(self):
        """Clocked latch D -> Q arcs, as OpenSTA's Latches::latchOutArrival / latchRequired: the stage of such an arc
        reads its own rows of the arrival buffers (pseudo rows after the pins'), written just before the stage
        (_apply_latch) from D's final arrivals: per launch tag, data arriving after the enable opens (enable arrival:
        the latch setup cycle's opening + the enable clock's early latency) passes, at most max_borrow late (the enable
        pulse width less the open / close latency difference and the latch setup time), moved into the enable clock's
        cycle and carried as data of the enable's opening edge. Max analysis only (the min one passes nothing through
        a latch); no CRPR, no set_max_time_borrow. _cb_fwd: per level the stage's (input rows, arrival mask)."""
        dev, P, C = self.dev, self.P, len(self.clocks)
        self._lq_n, self._lq_at, self._lq_chk = 0, {}, None
        self._lq_sel = self._lq_en = np.zeros(0, np.int64)      # the clocked latch D -> Q arcs (c-selection), enables
        self._cb_fwd = [None if cb is None else (cb[1] * 2 + cb[4], cb[7], torch.zeros_like(cb[7]))
                        for cb in self.cb_lev]
        c = self._lq_c
        if not len(c) or not C:
            self._load_indices()
            return
        cfr, cto = self.c_fr.cpu().numpy(), self.c_to.cpu().numpy()
        _, fr, to, atype, _ = self._arcs_all
        en_of = {}                                               # Q -> (enable pin, its opening transition)
        for i in np.nonzero(np.isin(atype, (T_RISE_EDGE, T_FALL_EDGE)) & np.isin(to, cto[c]))[0]:
            en_of.setdefault(int(to[i]), (int(fr[i]), 0 if atype[i] == T_RISE_EDGE else 1))
        keep = np.array([int(cto[x]) in en_of and self.clk_of[en_of[int(cto[x])][0]] >= 0 for x in c], bool)
        c = c[keep]
        if not len(c):
            self._load_indices()
            return
        jof = {int(x): j for j, x in enumerate(c)}
        self._lq_n = len(c)
        D, Q = cfr[c], cto[c]
        E = np.array([en_of[int(q)][0] for q in Q]); orf_e = np.array([en_of[int(q)][1] for q in Q])
        self._lq_sel, self._lq_en = c.astype(np.int64), E.astype(np.int64)
        ci = self.clk_of[E]; rf_e = orf_e ^ self.par[E]
        tag_e, tag_c = 2 * ci + rf_e, 2 * ci + (1 - rf_e)
        wf = self.wf.cpu().numpy(); per = np.array([cc['period'] for cc in self.clocks])
        hi = wf[ci, 1] - wf[ci, 0]
        pulse = np.where(rf_e == 0, hi, per[ci] - hi)
        # the latch setup check (enable closing edge -> D)
        kidx = np.full(len(c), -1, np.int64)
        if len(self.k_lib):
            kk = {}
            kf, kt, ky = self.k_fr.cpu().numpy(), self.k_to.cpu().numpy(), self.k_type.cpu().numpy()
            for i in range(len(kf)):
                kk.setdefault((int(kf[i]), int(kt[i]), int(ky[i])), i)
            for j in range(len(c)):
                kidx[j] = kk.get((int(E[j]), int(D[j]), T_SETUP_R if orf_e[j] == 1 else T_SETUP_F), -1)
        req, sh = _latch_relationships(self.clocks)
        T_ = lambda x, dt=torch.long: torch.as_tensor(np.asarray(x), device=dev, dtype=dt)  # noqa: E731
        f32 = torch.float32
        # the latch D setup checks (Latches::latchRequired): required = the enable's opening if data arrives before it,
        # the arrival while borrowing, the register check's (closing edge) beyond max_borrow
        _, first = np.unique(kidx, return_index=True)
        first = first[kidx[first] >= 0]
        if len(first):
            self._lq_chk = dict(k=T_(kidx[first]), E=T_(E[first]), oe=T_(orf_e[first]), tag_e=T_(tag_e[first]),
                                ci=T_(ci[first]), prop=T_(self.prop_clk[ci[first]], torch.bool),
                                req=T_(req[:, tag_e[first]].T, f32))
        cs = self.c_sense.cpu().numpy()
        for k, cb in enumerate(self.cb_lev):
            if cb is None:
                continue
            src = self._cb_src[k]
            m = np.isin(src, c)
            if not m.any():
                continue
            irf, orf = cb[4].cpu().numpy(), cb[3].cpu().numpy()
            js = np.array([jof.get(int(x), 0) for x in src])
            fi = np.where(m, (P + js) * 2 + irf, cb[1].cpu().numpy() * 2 + irf)
            sense = cs[src]
            ok = (sense == S_NON) | ((sense == S_POS) & (irf == orf)) | ((sense == S_NEG) & (irf != orf))
            arr = np.where(m, ok, cb[7].cpu().numpy())
            self._cb_fwd[k] = (T_(fi), T_(arr, torch.bool), T_(m, torch.bool))
            jj = np.unique(js[m])                                 # the latch arcs whose rows this stage reads
            self._lq_at[k] = dict(
                rows=T_(P + jj), D=T_(D[jj]), E=T_(E[jj]), oe=T_(orf_e[jj]), ce=T_(1 - orf_e[jj]),
                tag_e=T_(tag_e[jj]), tag_c=T_(tag_c[jj]), ci=T_(ci[jj]),
                prop=T_(self.prop_clk[ci[jj]], torch.bool), pulse=T_(pulse[jj], f32),
                kidx=T_(np.maximum(kidx[jj], 0)), has_k=T_(kidx[jj] >= 0, torch.bool),
                req=T_(req[:, tag_e[jj]].T, f32), sh=T_(sh[:, tag_e[jj]].T, f32))
        self._load_indices()

    def _load_indices(self):
        """Per stage, how its loads' arrivals are made (_load_arr): OpenSTA's graph delay calculation keeps one wire
        delay per wire edge and transition, the largest over the driver's arcs (GraphDelayCalc::annotateLoadDelaySlew
        merges), but none from a latch D -> Q arc (annotateDelaysSlews); a load's arrival is its driver's + that delay.
        Per stage (lane -> its (load, transition), those rows, their drivers' rows, the lanes that count)."""
        def index(t_, orf, ea, en, nol):
            si = self.n_to[en] * 2 + orf[ea]
            u, inv = torch.unique(si, return_inverse=True)
            d = torch.zeros(len(u), dtype=torch.long, device=self.dev).scatter_(0, inv, t_[ea] * 2 + orf[ea])
            return inv, u, d, ~nol[ea]
        self._ld_lev = [None if cb is None else index(cb[2], cb[3], cb[5], cb[6], fw[2])
                        for cb, fw in zip(self.cb_lev, self._cb_fwd)]
        self._ld_lc = None
        if self.lc is not None:
            lib, to, orf, tag, t0, cks = self.lc
            self._ld_lc = index(to, orf, *self.lc_ex, torch.zeros_like(lib, dtype=torch.bool))

    def _load_arr(self, amax, amin, ld, wx, wn):
        """One stage's load arrivals (see _load_indices), the wire delays kept across stages in self._wb."""
        inv, u, d, ok = ld
        if not len(u):
            return
        wbx, wbn = self._wb
        wbx.scatter_reduce_(0, u[inv], torch.where(ok, wx, torch.full_like(wx, -INF)), 'amax')
        wbn.scatter_reduce_(0, u[inv], torch.where(ok, wn, torch.full_like(wn, INF)), 'amin')
        amax.index_reduce_(0, u, amax[d] + wbx[u][:, None], 'amax')
        amin.index_reduce_(0, u, amin[d] + wbn[u][:, None], 'amin')

    def _latch_required(self, req, rf):
        """Register check required times (n_k, NT) with the latch D checks' (see _latch_prepare): min(the closing
        edge's, max(the enable's opening, the arrival)), per data tag."""
        g = self._lq_chk
        C2 = 2 * len(self.clocks)
        E, k = g['E'], g['k']
        lt = self.launch_t
        open_lat = torch.where(g['prop'], self.amin[E, g['oe'], g['tag_e']] - lt[g['tag_e']], self.lat[g['ci']])
        en = g['req'] + (open_lat - self.unc_s[g['ci']])[:, None]                         # (n, 2C)
        a = self.amax[self.k_to[k], rf, :C2]
        r = req[k].clone()
        r[:, :C2] = torch.minimum(r[:, :C2], torch.maximum(en, a))
        out = req.clone()
        out[k] = r
        return out

    def _apply_latch(self, amax, amin, smax, smin, key):
        """Write the pseudo rows of the latch D -> Q arcs of stage `key` (in place), see _latch_prepare."""
        g = self._lq_at.get(key)
        if g is None:
            return
        NT, C2 = self.NT, 2 * len(self.clocks)
        A = amax.view(-1, 2, NT); Am = amin.view(-1, 2, NT)
        S = smax.view(-1, 2); Sm = smin.view(-1, 2)
        D, E, rows = g['D'], g['E'], g['rows']
        lt = self.launch_t
        lat_i = self.lat[g['ci']]
        open_lat = torch.where(g['prop'], Am[E, g['oe'], g['tag_e']] - lt[g['tag_e']], lat_i)
        close_lat = torch.where(g['prop'], Am[E, g['ce'], g['tag_c']] - lt[g['tag_c']], lat_i)
        if len(self.k_lib):
            kx = g['kidx']
            cslew = torch.where(g['prop'], Sm[E, g['ce']], self.k_cslew[kx])
            margin = torch.stack([self._lookup(self.k_lib[kx], rf, S[D, rf], cslew) for rf in (0, 1)], 1)
            margin = torch.where(g['has_k'][:, None] & torch.isfinite(margin), margin, torch.zeros_like(margin))
        else:
            margin = torch.zeros(len(D), 2, device=self.dev)
        en_arr = g['req'] + (open_lat - self.unc_s[g['ci']])[:, None]                    # (n, 2C)
        maxb = g['pulse'][:, None] - (open_lat - close_lat)[:, None] - margin           # (n, 2)
        a = A[D, :, :C2]                                                                 # (n, 2, 2C)
        borrow = a - en_arr[:, None, :]
        req = torch.where(borrow <= maxb[:, :, None], a, en_arr[:, None, :] + maxb[:, :, None])
        v = torch.where((borrow > 0) & torch.isfinite(a), req + g['sh'][:, None, :], torch.full_like(a, -INF))
        q = v.amax(2)                                                                    # (n, 2): per D transition
        out = torch.full((len(D), 2, NT), -INF, device=self.dev)
        out.scatter_(2, g['tag_e'][:, None, None].expand(-1, 2, 1), q[:, :, None])
        A.index_copy_(0, rows, out)
        Am.index_fill_(0, rows, INF)
        S.index_copy_(0, rows, S[D]); Sm.index_copy_(0, rows, Sm[D])

    def _apply_lat(self, amax, amin, key):
        """Write the pin clock latencies assigned to stage `key` into the arrivals (in place)."""
        o = self._lat_at.get(key)
        if o is not None:
            ix, vx, jn, vn = o
            amax.view(-1)[ix] = vx
            amin.view(-1)[jn] = vn

    def _prepare_triton(self):
        """Per level, the int32 arrays of the fused stage kernel (attar.timing.dmp_triton.stage)."""
        i32 = lambda x: x.to(torch.int32).contiguous()  # noqa: E731
        dev = self.dev

        def lev(lib, fi, t, orf, t0, tag, ea, en, s0, arr):
            return (i32(lib), i32(fi), i32(t), i32(orf), i32(self.pin_net[t]), t0.float().contiguous(), i32(tag),
                    i32(ea), i32(en), s0.float().contiguous(), i32(arr))
        self.tri_lev = []
        self._tri_src = []                                       # ('l' | 'c', selection index of each entry)
        self._tri_k = []                                         # the stage level of each entry (None: launches)
        if self.lc is not None:
            lib, to, orf, tag, t0, cks = self.lc
            self.tri_lev.append(lev(lib, torch.full_like(lib, -1), to, orf, t0, tag, *self.lc_ex, cks,
                                    torch.ones_like(lib)))
            self._tri_src.append(('l', self._lc_src))
            self._tri_k.append(None)
        for k, (cb, src) in enumerate(zip(self.cb_lev, self._cb_src)):
            if cb is not None:
                self._tri_k.append(k)
                lib, f, t_, orf, irf, ea, en, _ = cb
                fi, arr, nol = self._cb_fwd[k]
                z = torch.zeros(len(lib), device=dev)
                self.tri_lev.append(lev(lib, fi, t_, orf, z, torch.zeros_like(lib), ea, en, z,
                                        arr.to(torch.int32) | (nol.to(torch.int32) << 1)))
                self._tri_src.append(('c', src))
        self.tri_tables = (self.t_ix1.contiguous(), self.t_ix2.contiguous(), self.t_val.contiguous(), self.t_n32,
                           self.t_thr.contiguous())
        self.tri_sthr = self.sink_thr.contiguous()
        self.tri_nto = i32(self.n_to)

    # -- propagation ----------------------------------------------------------------------------------------------------
    def run(self, pos=None, crpr=True):
        """Time the design at pos (default: the design's placement). Returns a TimingResult. On CUDA the whole
        computation is recorded once as a CUDA graph and replayed (positions copied into a static buffer).
        crpr: with propagated clocks, remove clock reconvergence pessimism from register checks (OpenSTA's default,
        attar.timing.crpr; after the replay, eager); False: as OpenSTA's sta::set_crpr_enabled 0."""
        pos = self.d.pos if pos is None else pos
        self._last_pos = pos.detach().clone()          # the run's placement (TimingResult.parasitics / trees)
        self._sync_pins()
        if self.parasitics == 'global_routing':
            self._gr_update()
        if not self.use_graph:
            return self._result(*self._run_core(pos), crpr=crpr)
        with torch.cuda.device(self.dev):
            if self.parasitics == 'placement':          # eager (only the nets that changed), into static buffers
                cw, elm = self.wires(pos)
                if self._ws is None:
                    self._ws = (cw.clone(), elm.clone(), self.pi.clone())
                else:
                    self._ws[0].copy_(cw); self._ws[1].copy_(elm); self._ws[2].copy_(self.pi)
                self.pi = self._ws[2]
            if self._graph is None:
                self._pos_buf = pos.detach().clone()
                st_ = torch.cuda.Stream(); st_.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st_):                # warm-up before the capture (results dropped)
                    for _ in range(2):
                        self._run_core(self._pos_buf)
                torch.cuda.current_stream().wait_stream(st_)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._g_out = self._run_core(self._pos_buf)
                self._graph = g
            self._pos_buf.copy_(pos)
            self._graph.replay()
            return self._result(self._g_out[0].clone(), self._g_out[1].clone(), crpr=crpr)

    def _result(self, setup, hold, crpr=True):
        cr = None
        if crpr and bool(self.prop_clk.any()) and len(self.k_lib):
            if getattr(self, '_crpr', None) is None:
                from .crpr import Crpr
                self._crpr = Crpr(self)
            cr = self._crpr
            cr.uncovered = 0
            setup, hold = self._checks(crpr=cr.run())
        res = TimingResult(self, setup, hold)
        res.crpr = None if cr is None else {'entries': cr.entries, 'uncovered': cr.uncovered, 'clock_muxes': cr.n_multi}
        return res

    def _run_core(self, pos):
        dev, P, NT = self.dev, self.P, self.NT
        if self.parasitics == 'global_routing':
            cw, elm = self.wires_gr()
        elif self.use_graph:                                  # run() filled them (the graph reads static buffers)
            cw, elm = self._ws[0], self._ws[1]; self.pi = self._ws[2]
        else:
            cw, elm = self.wires(pos)
        if elm is None:
            elm = torch.zeros(len(self.n_fr), 4, device=dev)
        self._elm = elm                                       # the run's wire delays, for required()
        self._cw = cw                                         # the run's wire capacitance per net (parasitics())
        # loads per driver pin (net wire cap + sink pin caps), max and min pin capacitances
        net_load = torch.zeros(len(self.d.n_ptr) - 1, 2, device=dev)
        net_load.index_add_(0, self.n_net, self.pin_cap[self.n_to])
        net_min = torch.zeros_like(net_load).index_add_(0, self.n_net, self.pin_cap_min[self.n_to])
        if cw is not None:
            net_load += cw[:, None]; net_min += cw[:, None]
        load = torch.zeros(P, 2, device=dev); load[self.drv_pins] = net_load[self.drv_nets]
        load_min = torch.zeros(P, 2, device=dev); load_min[self.drv_pins] = net_min[self.drv_nets]
        self.load, self.load_min = load, load_min
        pi = self.pi if self.pi is not None else torch.zeros(len(net_load), 4, 3, device=dev)
        PX = (P + self._lq_n) * 2                             # + the latch D -> Q arcs' pseudo rows (_latch_prepare)
        amax = torch.full((PX, NT), -INF, device=dev); amin = torch.full((PX, NT), INF, device=dev)
        smax = torch.zeros(PX, device=dev); smin = torch.full((PX,), INF, device=dev)
        self._wb = (torch.full((PX,), -INF, device=dev), torch.full((PX,), INF, device=dev))   # wire delay per load row
        amax.view(-1).scatter_reduce_(0, self.in_ix_max, self.in_v_max, 'amax')
        amin.view(-1).scatter_reduce_(0, self.in_ix_min, self.in_v_min, 'amin')
        for rf in (0, 1):
            smax.scatter_reduce_(0, self.in_sl_p * 2 + rf, self.in_sl_v, 'amax')
            smin.scatter_reduce_(0, self.in_sl_p * 2 + rf, self.in_sl_n, 'amin')
        from . import dmp
        # nets driven by input ports: no driver model, the wire alone (dspfWireDelaySlew + thresholdAdjust)
        self._pa_w = []                                       # their wire delays [(rf, max, min)] (attar.timing.crpr)
        if len(self.pa):
            f, t_, w4 = self.n_fr[self.pa], self.n_to[self.pa], elm[self.pa]
            for rf in (0, 1):
                lt = self.sink_thr[t_, rf]
                wds = []
                for A, S, red, w in ((amax, smax, 'amax', w4[:, rf]), (amin, smin, 'amin', w4[:, 2 + rf])):
                    wd, ls = dmp.input_port_load(S[f * 2 + rf], w, lt[:, 0], lt[:, 1], lt[:, 2], lt[:, 3])
                    A.index_reduce_(0, t_ * 2 + rf, A[f * 2 + rf] + wd[:, None], red)
                    S.scatter_reduce_(0, t_ * 2 + rf, ls, red)
                    wds.append(wd)
                self._pa_w.append((rf, *wds))

        rec = []                       # per stage, the gate and wire delays, max and min (required(), attar.timing.crpr)

        def stage(lib, fi, t_, orf, t0, tag, ea, en, sx, sn, arr=None, nol=None, ld=None):
            """Arcs into drivers t_ (output transition orf): gate delays and driver slews (DMP), arrivals at the
            drivers and, through each arc's wire delays (DMP load waveform), at their loads. fi: the arcs' input
            (pin * 2 + transition), or None for launch arcs (arrival t0 into tag). arr: the arcs of the arrival search
            (the others set slews only); slews come from every arc (OpenSTA's delay calculation)."""
            ti = t_ * 2 + orf
            pv = pi[self.pin_net[t_]]                                     # (n, 4, 3)
            n = len(lib)
            rows_ = torch.arange(n, device=dev)
            px, pn = pv[rows_, orf], pv[rows_, 2 + orf]                   # this transition's max / min pi models
            lib2 = torch.cat([lib, lib + self.lmin])                     # the min analysis: the Early library's arcs
            thr = self.t_thr[lib2, torch.cat([orf, orf])]
            # max and min analysis in one batch; the total load is the pi model's (OpenSTA: c1 + c2)
            c2_ = lambda v: torch.cat([v, v])  # noqa: E731
            args = (lib2, c2_(orf), torch.cat([sx, sn]), torch.cat([px[:, 0] + px[:, 2], pn[:, 0] + pn[:, 2]]),
                    torch.cat([px[:, 2], pn[:, 2]]), torch.cat([px[:, 1], pn[:, 1]]), thr)
            if self.triton is not None:
                dd, ds, stt = self.triton.gate(self.t_ix1, self.t_ix2, self.t_val, self.t_n32, *args)
            elif self.host is not None:
                dd, ds, stt = self.host.gate(self.host_tables, *args)
            else:
                dd, ds, stt = arc_eval(self.t_ix1, self.t_ix2, self.t_val, self.t_n, *args, self.steps)
            dmax, dmin, dsmax, dsmin, stx, stn = dd[:n], dd[n:], ds[:n], ds[n:], stt[:n], stt[n:]
            if fi is None:                                     # launch: one tag, the clock edge's arrival t0
                ax = torch.full((len(lib), NT), -INF, device=dev).scatter_(1, tag[:, None], (t0 + dmax)[:, None])
                an = torch.full((len(lib), NT), INF, device=dev).scatter_(1, tag[:, None], (t0 + dmin)[:, None])
            else:
                ax, an = amax[fi] + dmax[:, None], amin[fi] + dmin[:, None]
                if arr is not None:
                    ax = torch.where(arr[:, None], ax, torch.full_like(ax, -INF))
                    an = torch.where(arr[:, None], an, torch.full_like(an, INF))
            amax.index_reduce_(0, ti, ax, 'amax'); amin.index_reduce_(0, ti, an, 'amin')
            smax.scatter_reduce_(0, ti, dsmax, 'amax')
            smin.scatter_reduce_(0, ti, dsmin, 'amin')
            if len(ea):
                sk = self.n_to[en]
                rise = orf[ea] == 0
                lt = self.sink_thr[sk, orf[ea]]
                m = len(ea)
                oe = orf[ea]
                largs = (torch.cat([stx[ea], stn[ea]]), torch.cat([elm[en, oe], elm[en, 2 + oe]]), rise.repeat(2), lt.repeat(2, 1))
                if self.triton is not None:
                    w_, l_ = self.triton.load(*largs)
                elif self.host is not None:
                    w_, l_ = self.host.load(*largs)
                else:
                    w_, l_ = load_eval(*largs, self.steps)
                wx, wn, lsx, lsn = w_[:m], w_[m:], l_[:m], l_[m:]
                si = sk * 2 + orf[ea]
                if nol is not None:                          # latch D -> Q arcs: no load slews (nor wire delays)
                    lsx = torch.where(nol[ea], torch.zeros_like(lsx), lsx)
                    lsn = torch.where(nol[ea], torch.full_like(lsn, INF), lsn)
                smax.scatter_reduce_(0, si, lsx, 'amax')
                smin.scatter_reduce_(0, si, lsn, 'amin')
                self._load_arr(amax, amin, ld, wx, wn)
            z0 = torch.zeros(0, device=dev)
            rec.append((dmax.float(), wx.float() if len(ea) else z0, dmin.float(), wn.float() if len(ea) else z0))
        if self.triton is not None and self.fused:               # one fused kernel per level
            pic, elc = pi.contiguous(), elm.contiguous()
            self._rec = []
            for lv, k in zip(self.tri_lev, self._tri_k):
                if k is not None:
                    self._apply_lat(amax, amin, k)
                    self._apply_latch(amax, amin, smax, smin, k)
                self._rec.append(self.triton.stage(self.tri_tables, lv, self.tri_sthr, pic, elc, self.tri_nto, amax,
                                                   amin, smax, smin, self.precision, lmin=self.lmin))
                self._load_arr(amax, amin, self._ld_lc if k is None else self._ld_lev[k], self._rec[-1][1],
                               self._rec[-1][3])
            self._apply_lat(amax, amin, 'end')
            amax, amin, smax, smin = amax[:P * 2], amin[:P * 2], smax[:P * 2], smin[:P * 2]
            smin = torch.where(torch.isfinite(smin), smin, smax)
            self.amax, self.amin = amax.view(P, 2, NT), amin.view(P, 2, NT)
            self.smax, self.smin = smax.view(P, 2), smin.view(P, 2)
            return self._checks()
        if self.lc is not None:
            lib, to, orf, tag, t0, cks = self.lc               # ideal clock: its transition at the register clock pin
            stage(lib, None, to, orf, t0, tag, *self.lc_ex, cks, cks, ld=self._ld_lc)
        for k in range(self.max_level + 1):
            cb = self.cb_lev[k]
            if cb is not None:
                self._apply_lat(amax, amin, k)
                self._apply_latch(amax, amin, smax, smin, k)
                lib, f, t_, orf, irf, ea, en, _ = cb
                fi, arr, nol = self._cb_fwd[k]
                sx = smax[fi]; sn = smin[fi]; sn = torch.where(torch.isfinite(sn), sn, sx)
                stage(lib, fi, t_, orf, None, None, ea, en, sx, sn, arr, nol, self._ld_lev[k])
        self._apply_lat(amax, amin, 'end')
        self._rec = rec
        amax, amin, smax, smin = amax[:P * 2], amin[:P * 2], smax[:P * 2], smin[:P * 2]
        smin = torch.where(torch.isfinite(smin), smin, smax)
        self.amax, self.amin = amax.view(P, 2, NT), amin.view(P, 2, NT)
        self.smax, self.smin = smax.view(P, 2), smin.view(P, 2)
        return self._checks()

    # -- checks -------------------------------------------------------------------------------------------------------------
    def _checks(self, crpr=None):
        """Setup and hold slack per pin (P,), inf where unchecked. crpr: an attar.timing.crpr.Crpr of this run (register
        checks of propagated clocks less the clock reconvergence pessimism, per launch clock pin)."""
        dev, P = self.dev, self.P
        amax, amin, smin = self.amax, self.amin, self.smin
        lt = self.launch_t
        fin = lambda x: torch.where(torch.isnan(x) | ~torch.isfinite(x), torch.full_like(x, INF), x)
        sp = torch.full((P,), INF, device=dev); hp = torch.full((P,), INF, device=dev)
        A = amax.reshape(P * 2, -1)
        for rows, req, rf in self._setup_required():             # setup: required time less arrival, per launch tag
            v = fin(req - A[rows])
            if crpr is not None and rf is not None:
                v = crpr.setup(rf, req, v)
            sp.scatter_reduce_(0, rows // 2, v.amin(1), 'amin')
        if len(self.k_lib):
            lat = self._cap_lat(early=False)                    # hold: the capture clock's late arrival
            cslew = self._cap_slew(self.smax)                   # (OCV: the check reads the opposite analysis's)
            for rf in (0, 1):
                ok = self.t_n[self.k_lib, rf, 0] > 0
                cmin = self._lookup(self.k_lib + self.lmin, rf, smin[self.k_to, rf], cslew)   # hold: the min library
                req_h = (lt[None] + self.rel_hold[:, self.k_ctag].T + lat + cmin[:, None] + self.unc_h[self.k_ck][:, None]
                         - self.k_ctp_max[:, None])
                h = fin(amin[self.k_to, rf] - req_h)
                if crpr is not None:
                    h = crpr.hold(rf, req_h, h)
                h = h.amin(1)
                hp.scatter_reduce_(0, self.k_to, torch.where(ok & ~self.k_setup, h, torch.full_like(h, INF)), 'amin')
        p, rf, ct, off = self.op_ho
        if len(p):
            h = fin(amin[p, rf] - (lt[None] + self.rel_hold[:, ct].T + off[:, None])).amin(1)
            hp.scatter_reduce_(0, p, h, 'amin')
        return sp, hp

    def _cap_lat(self, early):
        """Per register check (n, 1): the capture clock's latency, its ideal latency, or for a propagated clock its
        arrival at the register's clock pin (early for setup, late for hold) less the clock edge."""
        lat = self.lat[self.k_ck] if len(self.clocks) else torch.zeros(len(self.k_lib), device=self.dev)
        if bool(self.prop_clk.any()):
            A = self.amin if early else self.amax                         # (P, 2, NT)
            v = A[self.k_fr, self.k_crf, self.k_ctag] - self.launch_t[self.k_ctag]
            lat = torch.where(self.k_prop & torch.isfinite(v), v, lat)
        pl = getattr(self, '_ideal_pin_lat', None)                 # an ideal clock's pin latency (_pin_latency_setup)
        if pl is not None and len(self.k_lib):
            v = pl[self.k_fr]
            lat = torch.where(torch.isfinite(v) & ~self.k_prop, v, lat)
        return lat[:, None]

    def _cap_slew(self, slew):
        """Per register check: the clock pin's slew in the check's table lookup, the ideal clock's transition or the
        propagated clock's slew at the pin. slew: the opposite analysis's (OpenSTA's default OCV analysis,
        Scene::checkClkSlewMinMax): min for setup, max for hold."""
        if not bool(self.prop_clk.any()):
            return self.k_cslew
        return torch.where(self.k_prop, slew[self.k_fr, self.k_crf], self.k_cslew)

    def _setup_required(self):
        """Setup required times at the endpoints of the last run, per launch tag: [(rows pin * 2 + transition,
        required (n, NT), the transition of register checks (rows k_to * 2 + rf) or None)], inf where a check does not apply. Register checks (setup time from the data and clock
        slews), output delays, and path delay exceptions (their own tag). An endpoint's setup slack is its required
        time less its arrival (_checks); every other pin's goes back from here (required)."""
        dev, smax, lt, NT = self.dev, self.smax, self.launch_t, self.NT
        out = []
        if len(self.k_lib):
            lat = self._cap_lat(early=True)                     # setup: the capture clock's early arrival
            cslew = self._cap_slew(self.smin)                   # (OCV: the check reads the opposite analysis's)
            for rf in (0, 1):
                ok = self.t_n[self.k_lib, rf, 0] > 0
                cmax = self._lookup(self.k_lib, rf, smax[self.k_to, rf], cslew)
                req = (lt[None] + self.rel_setup[:, self.k_ctag].T + lat - cmax[:, None] - self.unc_s[self.k_ck][:, None]
                       - self.k_ctp_min[:, None])
                if self._lq_chk is not None:
                    req = self._latch_required(req, rf)
                out.append((self.k_to * 2 + rf, torch.where((ok & self.k_setup)[:, None], req, torch.full_like(req, INF)),
                            rf))
        p, rf, ct, off = self.op_su
        if len(p):
            out.append((p * 2 + rf, lt[None] + self.rel_setup[:, ct].T + off[:, None], None))

        def one_tag(rows, tag, v):                                  # a required time for one tag only
            req = torch.full((len(rows), NT), INF, device=dev)
            return rows, req.scatter_(1, tag[:, None], v[:, None]), None
        # path delay exceptions: registers (setup time from the constrained slew) and output ports
        if len(self.k_lib) and len(self.kx[0]):
            a_, tg, rq = self.kx
            to = self.k_to[a_]
            cs_ = self._cap_slew(self.smin)[a_]                       # the propagated clock's slew, as checks
            if bool(self.prop_clk.any()):                              # a propagated clock's latency: its arrival
                rq = torch.where(self.k_prop[a_] & ~self.kx_ignore_lat,
                                 rq - self.lat[self.k_ck[a_]] + self._cap_lat(early=True)[a_, 0], rq)
            for rf in (0, 1):
                ok = self.t_n[self.k_lib[a_], rf, 0] > 0
                cst = self._lookup(self.k_lib[a_], rf, smax[to, rf], cs_)
                v = rq - cst - self.k_ctp_min[a_]
                out.append(one_tag(to * 2 + rf, tg, torch.where(ok, v, torch.full_like(v, INF))))
        p, rf, tg, rq = self.op_ex
        if len(p):
            out.append(one_tag(p * 2 + rf, tg, rq))
        return out

    # -- required times: pin and net slack (setup) ------------------------------------------------------------------------
    def required(self):
        """Setup required times of the last run (OpenSTA's, max analysis): (r, through) with r (P * 2, NT) at every
        net sink, cell input and input port (pin * 2 + transition, per launch tag), and through (P * 2,) at every
        driver pin the worst slack of the paths through it (a load's wire delay is the largest over its driver's arcs,
        as in the forward pass). Lazy: d.timing() only keeps the gate and wire delays it computed
        (one float per arc and per load); this goes back through the levels with index reductions only."""
        dev, P, NT = self.dev, self.P, self.NT
        A = self.amax.reshape(P * 2, NT); S = self.smax.reshape(-1)
        r = torch.full((P * 2, NT), INF, device=dev)
        for rows, req, _ in self._setup_required():
            r.index_reduce_(0, rows, torch.where(torch.isnan(req), INF, req), 'amin')   # NaN: no relationship
        through = torch.full((P * 2,), INF, device=dev)
        levs = []                                   # as the forward pass ran them: launch arcs, then level by level
        if self.lc is not None:
            lib, to, orf, tag, t0, cks = self.lc
            levs.append((lib, None, to, orf, cks, (tag, t0), *self.lc_ex, None))
        for cb in self.cb_lev:
            if cb is not None:
                lib, f, t_, orf, irf, ea, en, arr = cb
                fi = f * 2 + irf
                levs.append((lib, fi, t_, orf, None, None, ea, en, arr))
        if levs:
            assert len(self._rec) == len(levs), 'the delays of the last run do not match its levels'
            cat = lambda k: torch.cat([L[k] for L in levs])  # noqa: E731
            t_, orf, en = cat(2), cat(3), cat(7)
            off = np.cumsum([0] + [len(L[0]) for L in levs])
            ea = torch.cat([L[6] + int(o) for L, o in zip(levs, off)])
            dmax = torch.cat([x[0] for x in self._rec])
            si = self.n_to[en] * 2 + orf[ea]
            wx = self._wb[0][si]                                    # the loads' wire delays (_load_arr)
            loff = np.searchsorted(ea.cpu().numpy(), off)          # the loads of each level (ea ascends)
            for i in range(len(levs) - 1, -1, -1):
                a0, a1, l0, l1 = int(off[i]), int(off[i + 1]), int(loff[i]), int(loff[i + 1])
                fi, launch = levs[i][1], levs[i][5]
                d_ = dmax[a0:a1, None]
                ra = torch.full((a1 - a0, NT), INF, device=dev)     # required at the driver, through each arc
                ra.index_reduce_(0, ea[l0:l1] - a0, r[si[l0:l1]] - wx[l0:l1, None], 'amin')
                if levs[i][8] is not None:                         # slew-only arcs carry no path
                    ra = torch.where(levs[i][8][:, None], ra, torch.full_like(ra, INF))
                if launch is None:
                    ax = A[fi] + d_
                else:                                              # launch: one tag, the clock edge's arrival
                    tag, t0 = launch
                    ax = torch.full_like(ra, -INF).scatter_(1, tag[:, None], (t0 + dmax[a0:a1])[:, None])
                through.scatter_reduce_(0, t_[a0:a1] * 2 + orf[a0:a1], (ra - ax).amin(1), 'amin')
                if fi is not None:
                    r.index_reduce_(0, fi, ra - d_, 'amin')
        if len(self.pa):                                           # nets driven by input ports: the wire alone
            from . import dmp
            f, t_, w4 = self.n_fr[self.pa], self.n_to[self.pa], self._elm[self.pa]
            for rf in (0, 1):
                lt = self.sink_thr[t_, rf]
                wd, _ = dmp.input_port_load(S[f * 2 + rf], w4[:, rf], lt[:, 0], lt[:, 1], lt[:, 2], lt[:, 3])
                r.index_reduce_(0, f * 2 + rf, r[t_ * 2 + rf] - wd[:, None], 'amin')
        return r, through

    def path_prev(self):
        """The arrival predecessor of every pin transition and tag of the last run: prev (P * 2, NT), the row (pin * 2
        + transition) whose arrival in the tag plus the step's delay makes this row's arrival in that tag, -1 at a
        startpoint. At a net sink: its driver, same transition (the wire); at a cell output: among the search arcs
        into it, the input transition with the largest arrival + the arc's delay of the last run (its recorded gate
        delay: nothing is recomputed), as the forward pass took the maximum; ties go to the rise input, then the lower
        arc index. Also wire (P * 2,) bool: the step into the row is a wire. Index reductions over the levels."""
        dev, P, NT = self.dev, self.P, self.NT
        A = self.amax.reshape(P * 2, NT)
        i32 = torch.int32                                            # (rows and keys fit: half the memory)
        prev = torch.full((P * 2, NT), -1, device=dev, dtype=i32)
        wire = torch.zeros(P * 2, device=dev, dtype=torch.bool)
        for rf in (0, 1):                                            # net sinks: their driver, through the wire
            s_, f_ = self.n_to * 2 + rf, self.n_fr * 2 + rf
            ok = torch.isfinite(A[s_]) & torch.isfinite(A[f_])
            prev[s_] = torch.where(ok, f_[:, None].to(i32), prev[s_])
            wire[s_] = True
        cbs = [(cb, src) for cb, src in zip(self.cb_lev, self._cb_src) if cb is not None]
        if cbs:
            assert len(self._rec) == len(cbs) + (self.lc is not None), \
                'the delays of the last run do not match its levels'
            recs = self._rec[1:] if self.lc is not None else self._rec
            big = 2 ** 31 - 1
            best = torch.full((P * 2 * NT,), -INF, device=dev, dtype=A.dtype)
            kmin = torch.full((P * 2 * NT,), big, device=dev, dtype=i32)
            tg = torch.arange(NT, device=dev)
            span = max(int(len(self.c_lib)), 1)
            for (cb, src), rec in zip(cbs, recs):                    # a level at a time: its arcs' outputs
                _, f, t_, orf, irf, _, _, arr = cb
                arc = torch.as_tensor(src, device=dev)
                fi, ti = f * 2 + irf, t_ * 2 + orf
                v = A[fi] + rec[0].to(A.dtype)[:, None]
                v = torch.where(arr[:, None] & torch.isfinite(A[fi]), v, torch.full_like(v, -INF))
                ix = (ti[:, None] * NT + tg).flatten()
                v = v.flatten()
                best.scatter_reduce_(0, ix, v, 'amax')
                win = (v == best[ix]) & torch.isfinite(v)
                key = torch.where(win, (irf * span + arc)[:, None].expand(-1, NT).flatten().to(i32),
                                  torch.full(ix.shape, big, device=dev, dtype=i32))
                kmin.scatter_reduce_(0, ix, key, 'amin')
                sel = win & (key == kmin[ix])
                prev.view(-1)[ix[sel]] = fi[:, None].expand(-1, NT).flatten()[sel].to(i32)
                wire[ti] = False
        return prev, wire


class TimingData:
    """A read-only bundle of the timer's data (TimingResult.graph, library, arrivals, parasitics, required):
    attributes, copies of the timer's tensors (changing them changes nothing in the timer); version:
    TIMING_DATA_VERSION."""

    def __init__(self, what, **fields):
        self._what = what
        self.version = TIMING_DATA_VERSION
        self.__dict__.update(fields)

    def __repr__(self):
        def show(v):
            if torch.is_tensor(v):
                return '%s %s' % (tuple(v.shape), str(v.dtype).replace('torch.', ''))
            if isinstance(v, (list, tuple)):
                return '%d items' % len(v)
            return type(v).__name__ if callable(v) else repr(v)
        return '<%s: %s>' % (self._what, ', '.join('%s %s' % (k, show(v)) for k, v in self.__dict__.items()
                                                   if not k.startswith('_')))


class TimingResult:
    """Per-endpoint setup / hold slack (ps; inf where not constrained) and their summaries; per-pin and per-net
    setup slack on demand (pin_slack, net_slack); the timer's graph, library tables, arrivals and wire parasitics
    (graph, library, arrivals, parasitics, trees), read only, for research code built on the timer."""

    def __init__(self, timer, setup, hold):
        self.timer = timer
        self.setup, self.hold = setup, hold
        self._pin_slack = None
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

    def pin_slack(self):
        """Setup slack of every pin (P,), ps; inf where no constrained path passes: the worst path through the pin,
        required time less arrival, as OpenSTA's pin slack (max analysis). An endpoint's is its setup slack (a latch
        D pin: also the paths through it); the pins of the worst path have the WNS. With propagated clocks these do
        not remove the clock reconvergence pessimism (the endpoints' setup slacks do: attar.timing.crpr), so a pin's
        can be below its endpoints' by up to the credit. Computed on the first call
        (Timer.required: one batched delay evaluation, then index reductions back through the levels); valid for the
        last run of this result's timer, as path()."""
        if self._pin_slack is None:
            T = self.timer
            r, through = T.required()
            s = torch.minimum((r - T.amax.reshape(T.P * 2, -1)).amin(1), through)
            self._pin_slack = s.view(T.P, 2).amin(1)
        return self._pin_slack

    def net_slack(self):
        """Setup slack of every net (E,), ps; inf where unconstrained: the worst slack of its pins, as OpenSTA's
        Sta::netSlack, by which OpenROAD's timing-driven placement weights nets."""
        s = self.pin_slack()
        d = self.timer.d
        return torch.full((len(d.n_ptr) - 1,), INF, device=s.device).scatter_reduce_(0, d.p_net, s, 'amin')

    def drv(self):
        """The electrical design rules of this run: max slew, capacitance and fanout per pin, as OpenSTA's
        report_check_types (attar.timing.drv.DrvResult)."""
        from .drv import check
        return check(self)

    @property
    def limitations(self):
        """The known gaps that affect this timing (a list of str, empty when none): SDC commands not applied, pin
        latencies not modeled, nets with a lumped wire model, a benchmark's Early library not read (hold checks on Late
        delays). Each but the last was also warned when the timer was built."""
        return list(self.timer.__dict__.get('limitations', []))

    def summary(self):
        """WNS / TNS / worst slack / violations / endpoints (setup), WHS / THS / violations (hold), and exact: False
        when a known gap of the timer or of the design's timing inputs affects these numbers (the reasons:
        limitations)."""
        return {'wns': self.wns, 'tns': self.tns, 'worst_slack': self.worst_slack, 'setup_violations': self.violations,
                'endpoints': int(len(self.endpoints)), 'whs': self.whs, 'ths': self.ths, 'hold_violations': self.hold_violations,
                'exact': not self.limitations}

    def report(self, endpoints=10, path=True, file=None):
        """Print (and return) a timing summary: WNS, TNS, violations, hold, the worst endpoints and the worst path.
        file: where to print (default stdout). Returns summary() plus 'worst_endpoints' and 'worst_path'."""
        out = file or sys.stdout
        sm = self.summary()
        name = self.timer.d.manifest.get('design', '')
        print('Timing of %s (%s clocks, parasitics: %s)'
              % (name, ', '.join(self.timer.clock_names) or 'no',
                 'placement estimate' if self.timer.parasitics == 'placement' else 'global routes'), file=out)
        for x in self.limitations:
            print('  not exact: %s' % x, file=out)
        print('  setup: WNS %.2f ps, TNS %.2f ps, %d of %d endpoints violated (worst slack %.2f ps)'
              % (sm['wns'], sm['tns'], sm['setup_violations'], sm['endpoints'], sm['worst_slack']), file=out)
        print('  hold:  WHS %.2f ps, THS %.2f ps, %d violated' % (sm['whs'], sm['ths'], sm['hold_violations']), file=out)
        worst = []
        if endpoints:
            v, i = torch.sort(self.setup)
            k = int(torch.isfinite(v).sum())
            for p in i[:min(endpoints, k)].tolist():
                worst.append((self._pin_name(p), float(self.setup[p]), float(self.hold[p])))
            print('  worst endpoints (setup slack, hold slack):', file=out)
            for nm, a, b in worst:
                print('    %-50s %9.2f %9.2f' % (nm, a, b), file=out)
        pth = self.path() if path and sm['endpoints'] else []
        if pth:
            print('  worst path (arrival, slew ps):', file=out)
            for nm, rf, a, sl in pth:
                print('    %9.2f %7.2f %s %s' % (a, sl, '^' if rf == 'r' else 'v', nm), file=out)
        sm.update({'worst_endpoints': worst, 'worst_path': pth})
        return sm

    def _pin_name(self, p):
        return self.timer.d.pin_names([int(p)])[0]

    def pin_names(self, pins):
        """Pin indices -> names ('instance/PIN' or a port, as OpenSTA prints them; Design.pin_names)."""
        return self.timer.d.pin_names(pins)

    def path(self, endpoint=None):
        """The worst path into an endpoint (default: the worst setup endpoint), as critical_paths picks it (the worst
        transition and tag, back through the latest arrivals). Returns [(pin, 'r'|'f', arrival ps, slew ps)],
        startpoint first. Valid for the last run of this result's timer."""
        T = self.timer
        if endpoint is None:
            p = int(torch.argmin(self.setup).item())
        elif isinstance(endpoint, str):
            p = next(q for q in self.endpoints.tolist() if self._pin_name(q) == endpoint)
        else:
            p = int(endpoint)
        cp = self.critical_paths([p])
        sm = T.smax.reshape(-1)[cp.rows].tolist()
        return [(nm, rf, a, s_) for (nm, rf, a), s_ in zip(cp.report(0), sm)]

    def clock_arrivals(self):
        """{register clock pin name: (arrival max rise, max fall, min rise, min fall, slew max rise, max fall,
        min rise, min fall)} (ps): a propagated clock's arrival at the pin (the latest / earliest over its tags,
        OpenSTA's arrival_max_rise ...), an ideal clock's edge plus its latency (the pin's own set_clock_latency where
        one replaces the clock's) with the clock's set_clock_transition, as the timer launches and captures with them."""
        T = self.timer
        pins = torch.unique(T.k_fr) if len(T.k_lib) else torch.zeros(0, dtype=torch.long, device=T.dev)
        ax, an = T.amax[pins].amax(2), T.amin[pins].amin(2)
        sx, sn = T.smax[pins].clone(), T.smin[pins].clone()
        if len(pins) and len(T.clocks):
            ck = torch.as_tensor(T.clk_of, device=T.dev)[pins]
            par = torch.as_tensor(T.par, device=T.dev)[pins]
            ideal = (ck >= 0) & ~torch.as_tensor(T.prop_clk, device=T.dev)[ck.clamp(min=0)]
            c = ck.clamp(min=0)
            lat = T.lat[c]
            pl = T.__dict__.get('_ideal_pin_lat')                   # an ideal clock's pin latency, where modeled
            if pl is not None:
                lat = torch.where(torch.isfinite(pl[pins]), pl[pins], lat)
            rf = torch.arange(2, device=T.dev)
            t = T.wf[c[:, None], rf[None] ^ par[:, None]] + lat[:, None]     # the pin's rise / fall: the edge it sees
            tr = T.ctr[c]
            m = ideal[:, None]
            ax, an = torch.where(m, t, ax), torch.where(m, t, an)
            sx, sn = torch.where(m, tr, sx), torch.where(m, tr, sn)
        names = self.timer.d.pin_names(pins)
        v = torch.cat([ax, an, sx, sn], 1).tolist()
        return {nm: tuple(x) for nm, x in zip(names, v)}

    def critical_paths(self, endpoints=None, violating=True, max_paths=None):
        """The worst path into each of many endpoints at once (batched on the timer's device): endpoints (pin
        indices; default: every constrained endpoint, only the violating ones when violating), the worst max_paths of
        them when given. As OpenSTA's report_checks -path_delay max picks it: at the endpoint the transition and tag
        (launch clock edge) of the worst setup slack, then back through the latest arrivals in that tag. Not modeled:
        clock reconvergence pessimism removal in the choice (with propagated clocks OpenSTA may report another
        launching register whose path has a larger CRPR credit). Returns a CriticalPaths, worst first."""
        T = self.timer
        dev, P, NT = T.dev, T.P, T.NT
        ep = self.endpoints if endpoints is None else torch.as_tensor(endpoints, device=dev, dtype=torch.long)
        sl = self.setup[ep]
        if endpoints is None and violating:
            ep, sl = ep[sl < 0], sl[sl < 0]
        o = torch.argsort(sl, stable=True)
        if max_paths is not None:
            o = o[:max_paths]
        ep, sl = ep[o], sl[o]
        A = T.amax.reshape(P * 2, NT)
        req = torch.full((P * 2, NT), INF, device=dev)
        for rows, rq, _ in T._setup_required():
            req.index_reduce_(0, rows, torch.where(torch.isnan(rq), INF, rq), 'amin')
        both = (ep[:, None] * 2 + torch.arange(2, device=dev)).flatten()          # (E * 2,): rise, fall
        slk = torch.where(torch.isfinite(A[both]) & torch.isfinite(req[both]), req[both] - A[both],
                          torch.full_like(A[both], INF)).reshape(len(ep), 2 * NT)
        late = torch.where(torch.isfinite(A[both]), A[both], torch.full_like(A[both], -INF)).reshape(len(ep), 2 * NT)
        k = torch.where(torch.isfinite(slk).any(1), torch.argmin(slk, 1), torch.argmax(late, 1))  # (unconstrained:
        node = (ep * 2 + k // NT) * NT + k % NT                                                   # the latest)
        prev, wire = T.path_prev()
        par = (prev * NT + torch.arange(NT, device=dev, dtype=prev.dtype)).flatten()   # (int32: P * 2 * NT < 2^31)
        par = torch.where(prev.flatten() >= 0, par, torch.full_like(par, -1))
        del prev
        # walk back twice: the lengths, then each path written from its end
        n = torch.zeros(len(ep), device=dev, dtype=torch.long)
        cur = node.clone()
        while len(cur):
            for _ in range(32):
                live = cur >= 0
                n += live.long()
                cur = torch.where(live, par[cur.clamp(min=0)].long(), cur)
            if not bool((cur >= 0).any()):
                break
        ptr = torch.zeros(len(ep) + 1, device=dev, dtype=torch.long)
        ptr[1:] = torch.cumsum(n, 0)
        nodes = torch.empty(int(ptr[-1]), device=dev, dtype=torch.long)
        cur, at = node.clone(), ptr[1:] - 1
        for _ in range(int(n.max()) if len(ep) else 0):
            live = cur >= 0
            nodes[at[live]] = cur[live]
            at = at - live.long()
            cur = torch.where(live, par[cur.clamp(min=0)].long(), cur)
        rows = nodes // NT
        return CriticalPaths(self, ep, sl, ptr, rows, A.reshape(-1)[nodes], wire[rows], nodes % NT)

    # -- the timer's data, read only ---------------------------------------------------------------------------------
    def graph(self):
        """The timing graph (copies on the timer's device; pins are the design's pin indices, nets its nets). Fields:
          level (P,)              each pin's level over net and combinational arcs (Kahn; the search runs level by level)
          net_driver (E,)         each net's driver pin, -1 where it has none (or several)
          n_from, n_to, n_net     net arcs (A_n,): driver pin -> sink pin, and the net
          c_from, c_to, c_lib     cell arcs (A_c,) of the delay calculation: combinational, and the launch arcs of
                                  propagated clocks; c_lib the arc's Liberty arc (library())
          c_type, c_sense         its timing type and the sense the search uses (ir.T_*, ir.S_*; a constant input
                                  narrows the Liberty sense, c_sense_lib)
          c_searched              False where the arc only sets slews (ideal clock network, constants, a data pin into
                                  a propagated clock network, latch D -> Q)
          l_from, l_to, l_lib, l_type, l_sense   launch arcs under ideal clocks: register clock pin -> output
          k_from, k_to, k_lib, k_type            setup / hold checks: register clock pin -> data pin
          launch_arc, launch_rf   the launches under ideal clocks, one per launch arc (index into l_*) and output
                                  transition with a table
          launch_tag, launch_time, launch_slew
                                  its tag (below), the time it launches at (the clock edge plus the clock's or the
                                  pin's latency, ps) and the clock pin's slew (the clock's set_clock_transition)
          tag_clock, tag_edge, tag_time (NT,)
                                  arrivals are kept per launch tag (arrivals(per_tag=True), endpoint_required()): the
                                  tag's clock (index into clock_names) and edge (0: the waveform's rise, 1: its fall),
                                  -1 for the tags of path delay exceptions; the edge's time (ps)
          n_loop (A_n,)           net arcs OpenSTA's levelization disables to break combinational loops (not searched;
                                  they still load their net)
          loop_from, loop_to, loop_lib   the cell arcs it disables (not among c_*)
          latch_arcs, latch_enable       the clocked latch D -> Q arcs (indices into c_*; not searched as plain arcs:
                                  data arriving while the latch is open passes, borrowing time, as OpenSTA's latches)
                                  and their enable pins
          clock_of (P,)           the clock (index into clock_names) whose network reaches the pin, -1 none
          clock_names, endpoints  the clocks; this result's constrained setup endpoints (pins)
          version                 TIMING_DATA_VERSION"""
        T = self.timer
        dev = T.dev
        t = lambda x: torch.as_tensor(np.asarray(x), device=dev).clone()  # noqa: E731
        i64 = lambda x: torch.as_tensor(np.asarray(x, np.int64), device=dev)  # noqa: E731
        C, NT = len(T.clock_names), T.NT
        tag_clock = torch.full((NT,), -1, dtype=torch.long, device=dev); tag_edge = tag_clock.clone()
        tag_clock[:2 * C] = torch.arange(2 * C, device=dev) // 2; tag_edge[:2 * C] = torch.arange(2 * C, device=dev) % 2
        if T.lc is not None:
            _, _, orf, tag, t0, cks = T.lc
            launch = dict(launch_arc=i64(T._lc_src), launch_rf=orf.clone(), launch_tag=tag.clone(),
                          launch_time=t0.clone(), launch_slew=cks.clone())
        else:
            z = torch.zeros(0, dtype=torch.long, device=dev)
            launch = dict(launch_arc=z, launch_rf=z.clone(), launch_tag=z.clone(),
                          launch_time=torch.zeros(0, device=dev), launch_slew=torch.zeros(0, device=dev))
        lib_id, a_fr, a_to = T._arcs_all[0], T._arcs_all[1], T._arcs_all[2]
        la, ln = T.loop_edges
        n_loop = torch.zeros(len(T.n_fr), dtype=torch.bool, device=dev); n_loop[i64(ln)] = True
        return TimingData('TimingGraph', level=t(T.level), net_driver=t(T.net_driver),
                          n_from=T.n_fr.clone(), n_to=T.n_to.clone(), n_net=T.n_net.clone(),
                          c_from=T.c_fr.clone(), c_to=T.c_to.clone(), c_lib=T.c_lib.clone(), c_type=T.c_type.clone(),
                          c_sense=T.c_sense.clone(), c_sense_lib=T.c_sense_lib.clone(), c_searched=T.c_arr.clone(),
                          l_from=T.l_fr.clone(), l_to=T.l_to.clone(), l_lib=T.l_lib.clone(), l_type=T.l_type.clone(),
                          l_sense=T.l_sense.clone(),
                          k_from=T.k_fr.clone(), k_to=T.k_to.clone(), k_lib=T.k_lib.clone(), k_type=T.k_type.clone(),
                          **launch, tag_clock=tag_clock, tag_edge=tag_edge, tag_time=T.launch_t.clone(),
                          n_loop=n_loop, loop_from=i64(a_fr[la]), loop_to=i64(a_to[la]), loop_lib=i64(lib_id[la]),
                          latch_arcs=i64(T._lq_sel), latch_enable=i64(T._lq_en),
                          clock_of=t(T.clk_of), clock_names=list(T.clock_names), endpoints=self.endpoints.clone())

    def library(self):
        """The cell library the timer looks up (copies). Per Liberty arc a (graph()'s c_lib, l_lib, k_lib):
          master (A,)             the arc's master (d.names['masters'])
          from_pin, to_pin (A,)   its related and output (or constrained) pin, as master pin indices (d.raw['m_pin_ptr']
                                  ranges; names in d.names['master_pins'])
          type, sense (A,)        ir.T_*, ir.S_*
          ix1, ix2 (A, 4, K), val (A, 4, K, K), n (A, 4, 2)
                                  NLDM tables padded to K x K, n the real sizes (0: no table); slots of a delay arc
                                  cell_rise, cell_fall, rise_transition, fall_transition on (input slew ps, load fF),
                                  of a check rise_constraint, fall_constraint on (constrained, related pin transition);
                                  values in ps (ir.DELAY_SLOTS, ir.CHECK_SLOTS)
          pin_cap (P, 4)          every pin's capacitance (fF): rise and fall for the max analysis, rise and fall for
                                  the min analysis (set_load for output ports)
          lookup(a, slot, x, y)   the timer's lookup of arcs a (tensor) in slot at x, y (bilinear, linear
                                  extrapolation outside, OpenSTA's rule)
          min_offset              0, or with an Early library for the min analysis (hold, earliest arrivals) A: the
                                  tables have 2A rows, Liberty arc a's Early tables at a + A (same arcs, same order)"""
        T = self.timer
        tt, _, _ = timing_mod.arrays(T.d)
        dev = T.dev
        t = lambda x: torch.as_tensor(np.asarray(x).astype(np.int64), device=dev)  # noqa: E731
        ix1, ix2, val, tn = T.t_ix1.clone(), T.t_ix2.clone(), T.t_val.clone(), T.t_n.clone()

        def lookup(a, slot, x, y):
            a = torch.as_tensor(a, device=dev, dtype=torch.long)
            return nldm(ix1[a, slot], ix2[a, slot], val[a, slot], tn[a, slot], x, y)
        return TimingData('TimingLibrary', master=t(tt['a_master']), from_pin=t(tt['a_from']), to_pin=t(tt['a_to']),
                          type=t(tt['a_type']), sense=t(tt['a_sense']), ix1=ix1, ix2=ix2, val=val, n=tn,
                          pin_cap=torch.cat([T.pin_cap, T.pin_cap_min], 1).clone(), lookup=lookup,
                          min_offset=T.lmin)

    def arrivals(self, per_tag=False):
        """The arrival times and slews of the last run of this result's timer, as path() (copies, ps; per pin,
        transition rise / fall):
          max, min (P, 2)         the latest / earliest arrival over the launch tags (-inf / inf: none arrives);
                                  per_tag: (P, 2, NT), per launch tag (graph()'s tag_clock, tag_edge)
          slew_max, slew_min (P, 2)
          load, load_min (P, 2)   a driver pin's total load (wire and sink pins, fF; max / min pin capacitances),
                                  0 on other pins"""
        T = self.timer
        mx, mn = (T.amax.clone(), T.amin.clone()) if per_tag else (T.amax.amax(2).clone(), T.amin.amin(2).clone())
        return TimingData('Arrivals', max=mx, min=mn, slew_max=T.smax.clone(),
                          slew_min=T.smin.clone(), load=T.load.clone(), load_min=T.load_min.clone())

    def endpoint_required(self):
        """The setup checks of the last run of this result's timer (copies, ps), one row per check and data transition:
          pin, rf (R,)            the endpoint (a register data pin, an output port, a latch D pin) and its transition
          required (R, NT)        the required time per launch tag (graph()'s tag_clock, tag_edge; inf where the
                                  check does not apply to the tag): the capture edge, the capture clock's latency or
                                  arrival, less the setup time from the data and clock slews and the uncertainty;
                                  an output delay; a path delay exception's value (its own tag); a latch's, with
                                  time borrowing
          arrival (R, NT)         the latest arrival there per tag
          slack (R,)              min over tags of required - arrival (inf: unconstrained). With propagated clocks
                                  this is before the clock reconvergence pessimism removal; the result's setup slack
                                  (per pin, the min over its rows) is after it.
        A pin can have several rows (a register check and a path delay exception)."""
        T = self.timer
        rows, req = [], []
        for r, q, _ in T._setup_required():
            rows.append(r); req.append(q)
        if rows:
            rows, req = torch.cat(rows), torch.cat(req)
        else:
            rows = torch.zeros(0, dtype=torch.long, device=T.dev); req = torch.zeros(0, T.NT, device=T.dev)
        arr = T.amax.reshape(T.P * 2, -1)[rows]
        v = req - arr
        v = torch.where(torch.isnan(v) | ~torch.isfinite(v), torch.full_like(v, INF), v)
        return TimingData('EndpointRequired', pin=rows // 2, rf=rows % 2, required=req.clone(), arrival=arr.clone(),
                          slack=v.amin(1) if len(v) else torch.zeros(0, device=T.dev))

    def stages(self):
        """The level schedule of the last run of this result's timer, with the delays it computed (copies, ps). The
        search runs stage by stage: the launch arcs of ideal clocks first (if any), then one stage per level that has
        cell arcs. A stage evaluates its cell arcs (gate delay and driver slew, OpenSTA's delay calculation at the
        driver's load and input slew), then the wire arcs leaving their output pins (wire delay and load slew).
        Returns a TimingData with
          stages                  the stages in run order, each a TimingData:
            level                 the level of its arcs' output pins (graph().level); -1 for the launch stage
            launch                True for the launch stage
            arc (N,)              each entry's arc: an index into graph()'s l_* (launch stage) or c_* (others)
            lib, from_pin, to_pin (N,)   its Liberty arc (library()), input pin (the launch stage: the register clock
                                  pin) and output pin
            from_rf, to_rf (N,)   the input and output transitions (0 rise, 1 fall; launch stage: from_rf -1)
            searched (N,)         False where the entry only sets the output slew (graph()'s c_searched, and the
                                  transitions the sense allows)
            latch (N,)            clocked latch D -> Q entries: their arrival is the latch's (time borrowing), not
                                  the input pin's arrival plus the delay, and their loads get no wire delay
            tag, time, in_slew (N,)      launch stage only (None elsewhere): the launch tag, the time it launches at
                                  and the clock pin's slew (graph()'s launch_tag, launch_time, launch_slew)
            delay_max, delay_min (N,)    the gate delay of each entry (max and min analysis)
            load_entry, net_arc, sink (M,)   the wire arcs leaving the stage: per entry, every load of its output pin;
                                  the entry (index into this stage's N), the net arc (graph()'s n_*), the load pin
            wire_max, wire_min (M,)      the wire delay of each, after this entry's driver slew
          port_net_arcs           the net arcs driven by input ports (graph()'s n_*): no stage; a port's arrival and
                                  slew come from the constraints, its loads' arrivals add the wire delay
        How arrivals follow (arrivals(per_tag=True).max, per pin, transition and launch tag): an output pin's is the
        max over its searched, non-latch entries of the input pin's arrival plus delay_max (launch stage: time plus
        delay_max, in its tag); a load's is its driver's plus the wire delay, where OpenSTA keeps one wire delay per
        load and transition, the largest over the driver's entries. Before each stage the timer also sets the
        arrivals of pins with a set_clock_latency on them and of latch outputs; those are in arrivals(), not here."""
        T = self.timer
        rec = getattr(T, '_rec', None)
        if rec is None:
            raise RuntimeError('stages(): this result\'s timer has no delays of a run')
        dev = T.dev
        c = lambda x: x.clone()  # noqa: E731
        i64 = lambda x: torch.as_tensor(np.asarray(x, np.int64), device=dev)  # noqa: E731
        out = []
        if T.lc is not None:
            lib, to, orf, tag, t0, cks = T.lc
            ea, en = T.lc_ex
            src = i64(T._lc_src)
            out.append(dict(level=-1, launch=True, arc=src, lib=c(lib), from_pin=T.l_fr[src].clone(), to_pin=c(to),
                            from_rf=torch.full_like(orf, -1), to_rf=c(orf),
                            searched=torch.ones(len(lib), dtype=torch.bool, device=dev),
                            latch=torch.zeros(len(lib), dtype=torch.bool, device=dev),
                            tag=c(tag), time=c(t0), in_slew=c(cks), load_entry=c(ea), net_arc=c(en), sink=T.n_to[en]))
        for k, (cb, src) in enumerate(zip(T.cb_lev, T._cb_src)):
            if cb is None:
                continue
            lib, f, t_, orf, irf, ea, en, _ = cb
            _, arr, nol = T._cb_fwd[k]
            out.append(dict(level=k, launch=False, arc=i64(src), lib=c(lib), from_pin=c(f), to_pin=c(t_),
                            from_rf=c(irf), to_rf=c(orf), searched=c(arr), latch=c(nol), tag=None, time=None,
                            in_slew=None, load_entry=c(ea), net_arc=c(en), sink=T.n_to[en]))
        if len(rec) != len(out):
            raise RuntimeError('stages(): the delays of the last run do not match its levels')
        st = []
        for s, (dmax, wx, dmin, wn) in zip(out, rec):
            st.append(TimingData('TimingStage', **s, delay_max=dmax.float().clone(), delay_min=dmin.float().clone(),
                                 wire_max=wx.float().clone(), wire_min=wn.float().clone()))
        return TimingData('TimingStages', stages=st, port_net_arcs=c(T.pa))

    def parasitics(self):
        """The wire parasitics of the last run of this result's timer (copies). Variants v are the pin capacitances
        the reduction used: rise, fall (max analysis), rise, fall (min analysis), as OpenSTA reduces per transition.
          kind                    'placement' (estimate_parasitics -placement) or 'global_routing'
          wire_cap (E,)           each net's wire capacitance (fF; 0 on ideal clock nets)
          pi (E, 4, 3)            each net's driver pi model per variant: near C (fF), R (kohm), far C (fF)
          elmore (A_n, 4)         each net arc's (graph() n_from -> n_to) Elmore delay per variant (ps), from the
                                  driver through the wire and the sink's via
          pin_xy (P, 2)           where the parasitics put each pin (um): OpenROAD's location (the mean of its LEF
                                  rectangles' centers) at the run's placement"""
        T = self.timer
        cw = T._cw if T._cw is not None else torch.zeros(len(T.d.n_ptr) - 1, device=T.dev)
        pi = T.pi if T.pi is not None else torch.zeros(len(cw), 4, 3, device=T.dev)
        return TimingData('Parasitics', kind=T.parasitics, wire_cap=cw.clone(), pi=pi.clone(), elmore=T._elm.clone(),
                          pin_xy=T.ev._pins(T._last_pos).clone())

    def trees(self, nets=None):
        """The Steiner trees of the placement parasitics at the run's pin locations (parasitics().pin_xy): OpenROAD's
        estimate_parasitics tree (Prim-Dijkstra, alpha 0.3, steinerized; attar.timing.steiner_big), rooted at the
        driver, as the timer reduces it on every net that can carry a data arrival. nets: which (default: every net
        with a driver and two pins or more). Returns (edges (K, 4) um as x0, y0, x1, y1, each edge's first point
        nearer the driver; net (K,)), tensors on the timer's device."""
        T = self.timer
        if T.parasitics != 'placement':
            raise ValueError('trees(): the timer uses global-routing parasitics; its wires are the routes')
        from . import steiner_big
        if not steiner_big.available():
            raise RuntimeError('attar.timing.steiner_big unavailable: %s' % steiner_big.error())
        d = T.d
        ptr = d.n_ptr.cpu().numpy().astype(np.int64)
        deg = ptr[1:] - ptr[:-1]
        drv = np.asarray(T.net_driver)
        if nets is None:
            nets = np.nonzero((drv >= 0) & (deg >= 2))[0]
        else:
            nets = np.asarray(torch.as_tensor(nets).cpu(), np.int64)
            nets = nets[(drv[nets] >= 0) & (deg[nets] >= 2)]
        cnt = deg[nets]
        pins = np.repeat(ptr[nets] - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt) + np.arange(int(cnt.sum()))
        q = torch.round(T.ev._pins(T._last_pos) * d.dbu).cpu().numpy()[pins]
        sub = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
        e, k = steiner_big.pd_edges(q[:, 0], q[:, 1], sub, alpha=0.3, drv=drv[nets] - ptr[nets])
        return (torch.as_tensor(e.astype(np.float64) / d.dbu, device=T.dev, dtype=torch.float32),
                torch.as_tensor(nets[k], device=T.dev))

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



class CriticalPaths:
    """Paths of TimingResult.critical_paths, in CSR form: path k is rows[ptr[k]:ptr[k + 1]], startpoint first, each row
    a pin transition (pin * 2 + 0 rise / 1 fall) with its arrival (ps) in the path's tag; wire[i]: the step into row
    i is a net (driver pin -> sink pin), else a cell arc (input -> output). endpoint / slack (setup, ps) per path,
    worst first."""

    def __init__(self, result, endpoint, slack, ptr, rows, arrival, wire, tag):
        self.result = result
        self.endpoint, self.slack, self.ptr, self.rows, self.arrival, self.wire, self.tag = \
            endpoint, slack, ptr, rows, arrival, wire, tag
        self.pin = rows // 2
        self.rf = rows % 2

    def __len__(self):
        return len(self.endpoint)

    def path_of(self):
        """(N,) the path index of every row."""
        return torch.repeat_interleave(torch.arange(len(self), device=self.ptr.device), self.ptr[1:] - self.ptr[:-1])

    def pin_count(self):
        """(P,) how many of the paths pass through each pin."""
        P = self.result.timer.P
        k = self.path_of()
        key = torch.unique(k * P + self.pin)                         # once per path and pin
        return torch.bincount(key % P, minlength=P)

    def edges(self):
        """The steps of the paths: (from pin, to pin, is a wire, path index), (M,) each: consecutive rows."""
        k = self.path_of()
        if len(k) < 2:
            z = torch.zeros(0, dtype=torch.long, device=self.ptr.device)
            return z, z, z.bool(), z
        same = k[1:] == k[:-1]
        return self.pin[:-1][same], self.pin[1:][same], self.wire[1:][same], k[1:][same]

    def report(self, k=0):
        """Path k as [(pin name, 'r'|'f', arrival ps)], startpoint first (as TimingResult.path)."""
        a, b = int(self.ptr[k]), int(self.ptr[k + 1])
        name = self.result._pin_name
        return [(name(int(p)), 'rf'[int(r)], float(t)) for p, r, t in
                zip(self.pin[a:b].tolist(), self.rf[a:b].tolist(), self.arrival[a:b].tolist())]