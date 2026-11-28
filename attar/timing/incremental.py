"""Cone-limited re-timing after a placement change, as OpenSTA's incremental search: the drivers of the nets whose
parasitics changed are timed again, then only the arcs whose input arrival or slew changed, level by level, until
nothing changes (OpenSTA's incremental_delay_tolerance 0). Each recomputed pin takes every arc into it, with the same
GPU kernels as a full run fed only those arcs; arrivals are maxima and minima, so the result is bit for bit a full
run's. Timer.run uses it after a full run (Timer.incremental) and after a netlist edit the timer followed in place
(attar.timing.inplace); the checks (setup, hold, CRPR) are then evaluated as after a full run. The index over the
stages stays on the device (a stage's arcs are gathered there). A small cone is swept on the CPU instead (CpuSweep:
a worklist in level order with the C++ twin of the kernels, the same bits), a larger one on the GPU.

Not covered (a full run instead): the CPU path, global-routing parasitics, latch time borrowing, pin clock latencies,
a resize (the arcs' tables change), a change on a propagated clock network (its cone is the design), and a cone over
MAX_SHARE of the arcs (a full run is then faster). Timer.run also keeps the average time of both kinds of run and
re-times incrementally only while that is the faster one."""
import ctypes
import time

import numpy as np
import torch

INF = float('inf')


class Incremental:
    MAX_SHARE = 0.05            # a cone with more of the timer's arcs than this: a full run (cheaper per arc)

    def __init__(self, T):
        self.T = T
        self.last = {}
        why = [w for w, bad in (('no fused GPU kernels', T.triton is None or not T.fused),
                                ('global-routing parasitics', T.parasitics != 'placement'),
                                ('latch time borrowing', bool(T._lq_n or T._lq_at)),
                                ('pin clock latencies', bool(T._lat_at)),
                                ('not on a GPU', T.dev.type != 'cuda')) if bad]
        self.why = '; '.join(why)
        self.ok = not why
        if self.ok:
            self._index()

    def _index(self):
        """Indices over the stages of the fused run (Timer.tri_lev, in run order): every arc (entry) with its stage,
        input row and output row (pin * 2 + transition) and the kernel's inputs; every load lane with its row; CSR
        maps from a row to the entries into it and the entries out of it, and from an entry to its lanes. Rows written
        by more than one stage, or both as an output and a load, make the timer unsupported (none in practice)."""
        T = self.T
        dev = T.dev
        P2 = T.P * 2
        self.stages = []
        fi_l, ti_l, ld_l, le_l, ei_l, ef_l, en_l = [], [], [], [], [], [], []
        off = loff = 0
        for lv in T.tri_lev:
            lib, fi, t, orf, net, t0, tag, ea, en, s0, arr = lv
            N, M = len(lib), len(ea)
            fi_l.append(fi.long())
            ti_l.append(t.long() * 2 + orf.long())
            ea_l = ea.long()
            ld_l.append(T.n_to[en.long()] * 2 + orf.long()[ea_l])
            le_l.append(ea_l + off)
            ei_l.append(torch.stack([lib, fi, t, orf, net, tag, arr], 1))
            ef_l.append(torch.stack([t0.float(), s0.float()], 1))
            en_l.append(en.to(torch.int32))
            self.stages.append((off, N, loff, M))
            off += N; loff += M
        cat = lambda xs, *shape, dt=torch.long: torch.cat(xs) if xs else torch.zeros(0, *shape, dtype=dt, device=dev)  # noqa: E731
        e_int = cat(ei_l, 7, dt=torch.int32)
        self.e_cols = [e_int[:, i].contiguous() for i in range(7)]   # (lib, fi, t, orf, net, tag, arr)
        e_flt = cat(ef_l, 2, dt=torch.float32)
        self.e_flt0, self.e_flt1 = e_flt[:, 0].contiguous(), e_flt[:, 1].contiguous()
        self.lane_en = cat(en_l, dt=torch.int32)
        self.n_ent = off
        fi, ti, self.lane_row, lane_ent = cat(fi_l), cat(ti_l), cat(ld_l), cat(le_l)
        self.ti, self.fi, self.lane_ent = ti, fi, lane_ent
        self.ent_stage = torch.repeat_interleave(torch.arange(len(self.stages), device=dev),
                                                 torch.as_tensor([s[1] for s in self.stages], device=dev,
                                                                 dtype=torch.long))
        self.lane_drv = ti[lane_ent]                    # the driver row of each lane
        self.lane_ok = (e_int[lane_ent, 6] & 2) == 0
        # per row: the stage that writes it, the first and the last stage that read it
        nst = len(self.stages)
        self.row_stage = torch.full((P2,), nst, dtype=torch.long, device=dev).scatter_reduce_(
            0, ti, self.ent_stage, 'amin')
        v = fi >= 0
        self.first_use = torch.full((P2,), 1 << 30, dtype=torch.long, device=dev).scatter_reduce_(
            0, fi[v], self.ent_stage[v], 'amin')
        self.last_use = torch.full((P2,), -1, dtype=torch.long, device=dev).scatter_reduce_(
            0, fi[v], self.ent_stage[v], 'amax')
        # who writes each row: stage entries (outputs), stage lanes (loads), input ports' wires, constraints; a row
        # with two writers (keys) has a smallest key below its largest
        w_row = torch.cat([ti, self.lane_row])
        w_key = torch.cat([self.ent_stage * 2, self.ent_stage[lane_ent] * 2 + 1])
        kmin = torch.full((P2,), 1 << 62, dtype=torch.long, device=dev).scatter_reduce_(0, w_row, w_key, 'amin')
        kmax = torch.full((P2,), -1, dtype=torch.long, device=dev).scatter_reduce_(0, w_row, w_key, 'amax')
        ext = torch.zeros(P2, dtype=torch.bool, device=dev)
        NT = T.NT
        ext[T.in_ix_max // NT] = True
        ext[T.in_ix_min // NT] = True
        ext[T.in_sl_p * 2] = True; ext[T.in_sl_p * 2 + 1] = True
        if len(T.pa):
            tp = T.n_to[T.pa]
            ext[tp * 2] = True; ext[tp * 2 + 1] = True
        if bool(((kmax >= 0) & ((kmin != kmax) | ext)).any()):
            self.ok, self.why = False, 'rows written by more than one stage'
        self.pa_net = T.n_net[T.pa] if len(T.pa) else None
        # nets of a propagated clock network: a change there moves the clock arrival at every register it reaches,
        # a cone as large as the design (a full run at once)
        drv = np.asarray(T.net_driver)
        pc = np.zeros(len(drv), bool)
        pc[drv >= 0] = np.asarray(T.pclk)[drv[drv >= 0]]
        self.pclk_net = torch.as_tensor(pc, device=T.dev) if pc.any() else None
        # the input ports' own min slews as the run leaves them before the fallback to max (the port wires read them)
        self.smin_raw = torch.full((P2,), INF, device=T.dev)
        for rf in (0, 1):
            self.smin_raw.scatter_reduce_(0, T.in_sl_p * 2 + rf, T.in_sl_n, 'amin')

    # -- one re-timing ------------------------------------------------------------------------------------------------
    def run(self, dirty, cw, elm, budget=None):
        """Re-time after wires() redid the nets in `dirty` (a mask over nets; None: every net). Updates the timer's
        arrivals, slews, wire delays, loads and recorded delays in place; False (nothing usable done): a full run.
        budget: seconds after which a full run is cheaper (the timer's average full run, doubled).
        The stages are swept in run order with two row masks on the device: rows to recompute (the dirty nets'
        drivers, and the outputs of arcs whose input changed) and rows that changed; a stage recomputes every entry
        into a marked row, then marks the rows that changed; the sweep ends at the last stage that reads one."""
        T = self.T
        dev = T.dev
        t_start = time.perf_counter()
        if dirty is None or (self.pclk_net is not None and bool((dirty & self.pclk_net).any())):
            return False
        T._elm, T._cw = elm, cw
        T._loads(cw)
        dn = torch.nonzero(dirty).flatten()
        if not len(dn):
            return True
        drv = torch.as_tensor(T.net_driver, device=dev)[dn]
        drv = drv[drv >= 0]
        P2, NT = T.P * 2, T.NT
        A, Am = T.amax.view(P2, NT), T.amin.view(P2, NT)
        S, Sm = T.smax.view(-1), T.smin.view(-1)
        mark = torch.zeros(P2, dtype=torch.int32, device=dev)   # rows to recompute
        chg = torch.zeros(P2, dtype=torch.int32, device=dev)    # rows that changed
        seeds = torch.cat([drv * 2, drv * 2 + 1])
        mark[seeds] = 1
        nst_all = len(self.stages)
        rs = self.row_stage[seeds]
        lo = torch.cat([rs, torch.full((1,), 1 << 30, dtype=torch.long, device=dev)]).min()
        hi = torch.cat([torch.where(rs < nst_all, rs, -1), torch.full((1,), -1, dtype=torch.long, device=dev)]).max()
        if self.pa_net is not None:
            k = torch.nonzero(dirty[self.pa_net]).flatten()
            if len(k):
                R, diff = self._ports(k, elm, A, Am, S, Sm)
                chg[R] |= diff.int()
                lo = torch.minimum(lo, torch.where(diff, self.first_use[R], 1 << 30).min())
                hi = torch.maximum(hi, torch.where(diff, self.last_use[R], -1).max())
        s, hi = (int(x) for x in torch.stack([lo, hi]).cpu())
        self.last = {'dirty_nets': int(len(dn)), 'stages': 0, 'entries': 0, 'full': False}
        cpu = self._cpu_get() if self.pa_net is None or not bool(dirty[self.pa_net].any()) else None
        if cpu is not None and cpu.run(dirty, dn, seeds, T.pi, elm, int(min(cpu.MAX_ENTRIES, self.MAX_SHARE * self.n_ent))):
            self.last.update(cpu=True, entries=cpu.last['entries'])
            return True
        T._state_ver = T.__dict__.get('_state_ver', 0) + 1     # (the CPU mirror is stale from here)
        work = nst = 0
        budget, time_budget = self.MAX_SHARE * self.n_ent, budget
        pi, elc = T.pi.contiguous(), elm.contiguous()
        while s <= hi:
            off, N, loff, M = self.stages[s]
            fi_s, ti_s = self.fi[off:off + N], self.ti[off:off + N]
            hit = (chg[fi_s.clamp(min=0)] & (fi_s >= 0).int())   # arcs whose input changed: their outputs again
            mark.scatter_reduce_(0, ti_s, hit, 'amax')
            em = mark[ti_s] > 0                                   # every arc into a marked row
            ents = torch.nonzero(em).flatten()
            if len(ents):
                work += len(ents); nst += 1
                self.last.update(stages=nst, entries=work)
                if work > budget or (time_budget is not None and time.perf_counter() - t_start > time_budget):
                    self.last['full'] = True
                    return False
                R, diff = self._stage(s, em, ents, A, Am, S, Sm, pi, elc)
                chg[R] |= diff.int()
                h = int(torch.where(diff, self.last_use[R], -1).max()) if len(R) else -1
                hi = max(hi, h)
            s += 1
        return True

    def _stage(self, s, em, ents, A, Am, S, Sm, pi, elc):
        """Recompute the entries `ents` (local to stage s, sorted; em: their mask) of stage s and all their loads,
        with the fused kernel; returns the rows written (outputs, loads) and which of them changed. The kernel's
        inputs are gathered on the device into new tensors (aligned: Triton compiles a kernel per pointer alignment,
        these reuse the full run's)."""
        T = self.T
        dev = T.dev
        off, N, loff, M = self.stages[s]
        le = self.lane_ent[loff:loff + M] - off                  # the stage's lanes' entries (local)
        lanes = torch.nonzero(em[le]).flatten()
        m = len(lanes)
        pos = torch.cumsum(em, 0, dtype=torch.int32) - 1         # entry -> its position among ents
        ea_s = pos[le[lanes]]
        eg, lg = ents + off, lanes + loff
        lib, fi, t, orf, net, tag, arr = (c[eg] for c in self.e_cols)
        t0, s0 = self.e_flt0[eg], self.e_flt1[eg]
        en = self.lane_en[lg]
        u = self.lane_row[lg]                                     # load rows (repeats: the same value twice)
        R = torch.cat([self.ti[eg], u])
        ok = self.lane_ok[lg]
        old = (A.index_select(0, R), Am.index_select(0, R), S.index_select(0, R), Sm.index_select(0, R))
        A.index_fill_(0, R, -INF); Am.index_fill_(0, R, INF); S.index_fill_(0, R, 0.0); Sm.index_fill_(0, R, INF)
        wbx, wbn = T._wb
        wbx.index_fill_(0, u, -INF); wbn.index_fill_(0, u, INF)
        sub = (lib, fi, t, orf, net, t0, tag, ea_s, en, s0, arr)
        dmax, wx, dmin, wn = T.triton.stage(T.tri_tables, sub, T.tri_sthr, pi, elc, T.tri_nto, A, Am, S, Sm,
                                            T.precision, lmin=T.lmin)
        if m:
            T._load_arr(A, Am, (torch.arange(m, device=dev), u, self.lane_drv[lg], ok), wx, wn)
        v, sx = Sm.index_select(0, R), S.index_select(0, R)
        Sm.index_copy_(0, R, torch.where(torch.isfinite(v), v, sx))      # Timer: smin falls back to smax
        rec = T._rec[s]
        rec[0].index_copy_(0, ents, dmax); rec[2].index_copy_(0, ents, dmin)
        if m:
            rec[1].index_copy_(0, lanes, wx); rec[3].index_copy_(0, lanes, wn)
        diff = ((A.index_select(0, R) != old[0]).any(1) | (Am.index_select(0, R) != old[1]).any(1)
                | (sx != old[2]) | (Sm.index_select(0, R) != old[3]))
        return R, diff

    def _cpu_get(self):
        """The CPU sweep (CpuSweep), made on this index's second re-timing (its host copies cost about a hundred GPU
        sweeps of a small cone: an index used once, as after a netlist edit, keeps the GPU sweep); None before, and
        where it cannot run (no compiler, a delay model in float32: only the float64 kernels have a CPU twin)."""
        self.runs = self.__dict__.get('runs', 0) + 1
        c = self.__dict__.get('cpu', False)
        if c is False and self.runs < 2:
            return None
        if c is False:
            c = None
            if self.T.precision == 'double':
                try:
                    c = CpuSweep(self)
                except OSError:                     # (no compiler: the GPU sweep only)
                    c = None
            self.cpu = c
        return c

    def _ports(self, k, elm, A, Am, S, Sm):
        """Recompute the loads of the port-driven net arcs pa[k] (Timer._run_core's input-port wires); returns the
        rows that changed."""
        T = self.T
        from . import dmp
        arcs = T.pa[k]
        f, t_, w4 = T.n_fr[arcs], T.n_to[arcs], elm[arcs]
        R = torch.unique(torch.cat([t_ * 2, t_ * 2 + 1]))
        old = (A[R], Am[R], S[R], Sm[R])
        A[R] = -INF; Am[R] = INF; S[R] = 0.0; Sm[R] = INF
        for rf in (0, 1):
            lt = T.sink_thr[t_, rf]
            wds = []
            for AA, SS, red, w, src in ((A, S, 'amax', w4[:, rf], S[f * 2 + rf]),
                                        (Am, Sm, 'amin', w4[:, 2 + rf], self.smin_raw[f * 2 + rf])):
                wd, ls = dmp.input_port_load(src, w, lt[:, 0], lt[:, 1], lt[:, 2], lt[:, 3])
                AA.index_reduce_(0, t_ * 2 + rf, AA[f * 2 + rf] + wd[:, None], red)
                SS.scatter_reduce_(0, t_ * 2 + rf, ls, red)
                wds.append(wd)
            prev = T._pa_w[rf]
            prev[1][k] = wds[0]; prev[2][k] = wds[1]
        Sm[R] = torch.where(torch.isfinite(Sm[R]), Sm[R], S[R])
        diff = (A[R] != old[0]).any(1) | (Am[R] != old[1]).any(1) | (S[R] != old[2]) | (Sm[R] != old[3])
        return R, diff


class CpuSweep:
    """The re-timing of Incremental.run on the CPU (attar/timing/csrc/inc.cpp): a worklist of driver rows in stage
    order, each taking every arc into it, with the C++ twin of the fused kernels (csrc/dmp_core.h: the same bits).
    The timer's state is mirrored on the host (copied once after each full run, then kept in step: the rows, entries
    and lanes a sweep recomputes are copied back to the device). Faster than the GPU sweep while the cone is small (no
    launch per stage); a cone over MAX_ENTRIES entries goes to the GPU sweep."""
    MAX_ENTRIES = 10000

    def __init__(self, inc):
        from . import dmp_host
        from .dmp import FIND_ROOT_MAX_ITER, NEWTON_MAX_ITER
        self.inc, T = inc, inc.T
        self.lib = dmp_host.compile_lib('inc', 'inc.cpp')
        L = self.lib
        P, I64, I = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
        L.attar_inc_new.restype = P
        L.attar_inc_new.argtypes = [I64, I64] + [P] * 19 + [P, P, P, P, I64, P, P, P, I64, I, I]
        L.attar_inc_free.argtypes = [P]
        L.attar_inc_state.argtypes = [P] * 13
        L.attar_inc_run.restype = I
        L.attar_inc_run.argtypes = [P, P, I64, I64, P, I64, P, P, I64, P, P, I64, P]
        h = lambda t, dt: np.ascontiguousarray(t.detach().cpu().numpy(), dt)  # noqa: E731
        P2, NT = T.P * 2, T.NT
        self.P2, self.NT = P2, NT
        e = [h(c, np.int32) for c in inc.e_cols]            # lib, fi, t, orf, net, tag, arr
        t0, s0 = h(inc.e_flt0, np.float32), h(inc.e_flt1, np.float32)
        ti, fi = inc.ti, inc.fi

        def csr(key, n, sel=None):
            idx = torch.arange(len(key), device=key.device) if sel is None else torch.nonzero(sel).flatten()
            k = key[idx]
            o = torch.argsort(k, stable=True)
            ptr = torch.zeros(n + 1, dtype=torch.long, device=key.device)
            ptr[1:] = torch.cumsum(torch.bincount(k, minlength=n), 0)
            return h(ptr, np.int64), h(idx[o], np.int32)
        in_ptr, in_idx = csr(ti, P2)
        out_ptr, out_idx = csr(fi.clamp(min=0), P2, fi >= 0)
        ln_ptr, ln_idx = csr(inc.lane_ent, inc.n_ent)
        lane_en, lane_row = h(inc.lane_en, np.int32), h(inc.lane_row, np.int32)
        lane_ok = h(inc.lane_ok, np.uint8)
        row_stage = h(inc.row_stage, np.int32)
        ix1, ix2, val, n32, thr = T.tri_tables
        tabs = (h(ix1, np.float32), h(ix2, np.float32), h(val, np.float32), h(n32, np.int64))
        thr_h, sthr_h, nto_h = h(thr, np.float32), h(T.tri_sthr, np.float32), h(T.tri_nto, np.int32)
        self._keep = (e, t0, s0, in_ptr, in_idx, out_ptr, out_idx, ln_ptr, ln_idx, lane_en, lane_row, lane_ok,
                      row_stage, tabs, thr_h, sthr_h, nto_h)
        ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
        self.h = L.attar_inc_new(P2, NT, *[ptr(x) for x in e], ptr(t0), ptr(s0), ptr(in_ptr), ptr(in_idx),
                                 ptr(out_ptr), ptr(out_idx), ptr(ln_ptr), ptr(ln_idx), ptr(lane_en), ptr(lane_row),
                                 ptr(lane_ok), ptr(row_stage), *[ptr(x) for x in tabs], int(ix1.shape[-1]),
                                 ptr(thr_h), ptr(sthr_h), ptr(nto_h), int(T.lmin), NEWTON_MAX_ITER,
                                 FIND_ROOT_MAX_ITER)
        self.ent_stage = h(inc.ent_stage, np.int64)
        self.lane_stage = self.ent_stage[h(inc.lane_ent, np.int64)]
        self.e_off = np.array([s[0] for s in inc.stages], np.int64)
        self.l_off = np.array([s[2] for s in inc.stages], np.int64)
        self.ver = None                                     # the timer state version the mirror holds
        self.n_lane = len(lane_en)
        self.out_rows = np.empty(P2, np.int32)
        self.out_ents = np.empty(inc.n_ent, np.int32)
        self.out_lanes = np.empty(max(self.n_lane, 1), np.int32)
        self.last = {}

    def __del__(self):
        h = self.__dict__.get('h')
        if h:
            self.lib.attar_inc_free(h)

    def _mirror(self, pi, elm):
        """Copy the timer's state to the host (after a full run)."""
        T, P2 = self.inc.T, self.P2
        cp = lambda t: np.ascontiguousarray(t.detach().reshape(-1).cpu().numpy(), np.float32)  # noqa: E731
        self.A, self.Am = cp(T.amax), cp(T.amin)
        self.S, self.Sm = cp(T.smax), cp(T.smin)
        self.wbx, self.wbn = cp(T._wb[0][:P2]), cp(T._wb[1][:P2])
        rec = T._rec
        cat = lambda j: cp(torch.cat([r[j].reshape(-1) for r in rec])) if rec else np.zeros(0, np.float32)  # noqa: E731
        self.dmax, self.wx, self.dmin, self.wn = cat(0), cat(1), cat(2), cat(3)
        self.pi, self.elm = cp(pi), cp(elm)
        self._bind()

    def _bind(self):
        ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
        self.lib.attar_inc_state(self.h, *[ptr(x) for x in (self.A, self.Am, self.S, self.Sm, self.wbx, self.wbn,
                                                             self.dmax, self.dmin, self.wx, self.wn, self.pi,
                                                             self.elm)])

    def run(self, dirty, dn, seeds, pi, elm, max_ents):
        """Re-time on the CPU from the seed rows (a device tensor) after wires() redid the nets dn. Returns True
        (done: the timer's device state updated), or False (the cone is over max_ents entries, or the index cannot
        be followed: the device state is untouched, the GPU sweep takes over)."""
        T, dev = self.inc.T, self.inc.T.dev
        ver = T.__dict__.get('_state_ver', 0)
        if self.ver != ver:
            self._mirror(pi, elm)
        else:                                               # the redone nets' pi models and wire delays
            self.pi.reshape(-1, 12)[dn.cpu().numpy()] = pi[dn].reshape(-1, 12).cpu().numpy()
            arcs = torch.nonzero(dirty[T.n_net]).flatten()
            self.elm.reshape(-1, 4)[arcs.cpu().numpy()] = elm[arcs].cpu().numpy()
        self.ver = None                                     # (until the device has the result)
        sd = np.ascontiguousarray(seeds.cpu().numpy(), np.int64)
        nr, ne, nl = ctypes.c_int64(), ctypes.c_int64(), ctypes.c_int64()
        ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
        st = self.lib.attar_inc_run(self.h, ptr(sd), len(sd), max_ents, ptr(self.out_rows), self.P2,
                                    ctypes.byref(nr), ptr(self.out_ents), len(self.out_ents), ctypes.byref(ne),
                                    ptr(self.out_lanes), len(self.out_lanes), ctypes.byref(nl))
        self.last = {'status': st, 'rows': nr.value, 'entries': ne.value, 'lanes': nl.value}
        if st:
            return False                                    # (the mirror is copied again next time)
        rows = self.out_rows[:nr.value].astype(np.int64)
        ents = self.out_ents[:ne.value].astype(np.int64)
        lanes = self.out_lanes[:nl.value].astype(np.int64)
        P2, NT = self.P2, self.NT
        # rows: arrivals, slews, wire delays per load, in one copy
        pk = np.concatenate([self.A.reshape(P2, NT)[rows], self.Am.reshape(P2, NT)[rows],
                             np.stack([self.S[rows], self.Sm[rows], self.wbx[rows], self.wbn[rows]], 1)], 1)
        pk_t = torch.from_numpy(pk).to(dev)
        r_t = torch.from_numpy(rows).to(dev)
        T.amax.view(P2, NT).index_copy_(0, r_t, pk_t[:, :NT])
        T.amin.view(P2, NT).index_copy_(0, r_t, pk_t[:, NT:2 * NT])
        T.smax.view(-1).index_copy_(0, r_t, pk_t[:, 2 * NT])
        T.smin.view(-1).index_copy_(0, r_t, pk_t[:, 2 * NT + 1])
        T._wb[0].index_copy_(0, r_t, pk_t[:, 2 * NT + 2])
        T._wb[1].index_copy_(0, r_t, pk_t[:, 2 * NT + 3])
        # recorded delays, per stage
        for idx, stg, off, src, cols in ((ents, self.ent_stage, self.e_off, (self.dmax, self.dmin), (0, 2)),
                                         (lanes, self.lane_stage, self.l_off, (self.wx, self.wn), (1, 3))):
            if not len(idx):
                continue
            s = stg[idx]
            o = np.argsort(s, kind='stable')
            idx, s = idx[o], s[o]
            vals = torch.from_numpy(np.stack([src[0][idx], src[1][idx]], 1)).to(dev)
            loc = torch.from_numpy(idx - off[s]).to(dev)
            cut = np.flatnonzero(np.diff(s)) + 1
            for a, b in zip(np.r_[0, cut], np.r_[cut, len(s)]):
                rec = T._rec[int(s[a])]
                rec[cols[0]].index_copy_(0, loc[a:b], vals[a:b, 0].to(rec[cols[0]].dtype))
                rec[cols[1]].index_copy_(0, loc[a:b], vals[a:b, 1].to(rec[cols[1]].dtype))
        self.ver = ver
        return True
