"""Cone-limited re-timing after a placement change, as OpenSTA's incremental search: the drivers of the nets whose
parasitics changed are timed again, then only the arcs whose input arrival or slew changed, level by level, until
nothing changes (OpenSTA's incremental_delay_tolerance 0). Each recomputed pin takes every arc into it, with the same
GPU kernels as a full run fed only those arcs; arrivals are maxima and minima, so the result is bit for bit a full
run's. Timer.run uses it after a full run (Timer.incremental); the checks (setup, hold, CRPR) are then evaluated as
after a full run.

Not covered (a full run instead): the CPU path, global-routing parasitics, latch time borrowing, pin clock latencies,
a resize (the arcs' tables change), a change on a propagated clock network (its cone is the design), and a cone over
MAX_SHARE of the arcs (a full run is then faster). Timer.run also keeps the average time of both kinds of run and
re-times incrementally only while that is the faster one."""
import heapq
import time

import numpy as np
import torch

INF = float('inf')


def _upload(parts, dev):
    """Copy the 1-d arrays `parts` to the device in one copy (one host sync), each starting on a 16-byte boundary
    (Triton compiles one kernel per pointer alignment: aligned views reuse the full run's kernels); returns their
    views on the device, in their own dtypes."""
    offs, o = [], 0
    for x in parts:
        offs.append(o)
        o += -(-x.nbytes // 16) * 16
    buf = np.zeros(max(o, 16), np.uint8)
    for x, a in zip(parts, offs):
        buf[a:a + x.nbytes] = np.ascontiguousarray(x).view(np.uint8)
    t = torch.from_numpy(buf).to(dev)
    return [t[a:a + x.nbytes].view(_TORCH[x.dtype]) for x, a in zip(parts, offs)]


_TORCH = {np.dtype(np.int32): torch.int32, np.dtype(np.int64): torch.int64, np.dtype(np.float32): torch.float32,
          np.dtype(np.bool_): torch.bool}


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
        """Host indices over the stages of the fused run (Timer.tri_lev, in run order): every arc (entry) with its
        stage, input row and output row (pin * 2 + transition); every load lane with its row; CSR maps from a row to
        the entries into it and the entries out of it, and from an entry to its lanes. Rows written by more than one
        stage, or both as an output and a load, make the timer unsupported (none in practice)."""
        T = self.T
        dev = T.dev
        P2 = T.P * 2
        self.stages = []
        fi_l, ti_l, ld_l, le_l, ei_l, ef_l, en_l = [], [], [], [], [], [], []
        off = loff = 0
        for lv in T.tri_lev:                            # (built on the device: sorts there are seconds faster)
            lib, fi, t, orf, net, t0, tag, ea, en, s0, arr = lv
            N, M = len(lib), len(ea)
            fi_l.append(fi.long())
            ti_l.append(t.long() * 2 + orf.long())
            ea_l = ea.long()
            ld_l.append(T.n_to[en.long()] * 2 + orf.long()[ea_l])
            le_l.append(ea_l + off)
            # the kernel's per-arc and per-lane inputs, on the host (a stage's subset goes up in one copy)
            ei_l.append(torch.stack([lib, fi, t, orf, net, tag, arr], 1))
            ef_l.append(torch.stack([t0.float(), s0.float()], 1))
            en_l.append(en.to(torch.int32))
            self.stages.append((off, N, loff, M))
            off += N; loff += M
        cat = lambda xs, *shape, dt=torch.long: torch.cat(xs) if xs else torch.zeros(0, *shape, dtype=dt, device=dev)  # noqa: E731
        e_int = cat(ei_l, 7, dt=torch.int32)
        self.e_int = e_int.cpu().numpy()
        self.e_flt = cat(ef_l, 2, dt=torch.float32).cpu().numpy()
        self.lane_en = cat(en_l, dt=torch.int32).cpu().numpy()
        self.n_ent = off
        fi, ti, lane_row, lane_ent = cat(fi_l), cat(ti_l), cat(ld_l), cat(le_l)
        ent_stage = torch.repeat_interleave(torch.arange(len(self.stages), device=dev),
                                            torch.as_tensor([s[1] for s in self.stages], device=dev, dtype=torch.long))
        rows = torch.arange(P2 + 1, device=dev)
        o = torch.sort(ti, stable=True)[1]
        to_ptr = torch.searchsorted(ti[o], rows)
        self.to_ord, self.to_ptr = o.cpu().numpy(), to_ptr.cpu().numpy()
        v = torch.nonzero(fi >= 0).flatten()
        o = v[torch.sort(fi[v], stable=True)[1]]
        self.fr_ord, self.fr_ptr = o.cpu().numpy(), torch.searchsorted(fi[o], rows).cpu().numpy()
        self.lane_ptr = torch.searchsorted(lane_ent, torch.arange(self.n_ent + 1, device=dev)).cpu().numpy()  # lanes
        self.lane_drv = ti[lane_ent].cpu().numpy()      # the driver row of each lane            ascend by entry
        self.lane_ok = ((e_int[lane_ent, 6] & 2) == 0).cpu().numpy()
        self.lane_row, self.ti, self.ent_stage = lane_row.cpu().numpy(), ti.cpu().numpy(), ent_stage.cpu().numpy()
        # who writes each row: stage entries (outputs), stage lanes (loads), input ports' wires, constraints; a row
        # with two writers (keys) has a smallest key below its largest
        w_row = torch.cat([ti, lane_row])
        w_key = torch.cat([ent_stage * 2, ent_stage[lane_ent] * 2 + 1])
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
        budget: seconds after which a full run is cheaper (the timer's average full run, doubled)."""
        T = self.T
        t_start = time.perf_counter()
        if dirty is None or (self.pclk_net is not None and bool((dirty & self.pclk_net).any())):
            return False
        T._elm, T._cw = elm, cw
        T._loads(cw)
        dn = torch.nonzero(dirty).flatten()
        if not len(dn):
            return True
        drv = torch.as_tensor(T.net_driver, device=T.dev)[dn]
        drv = drv[drv >= 0].cpu().numpy()
        seeds = np.concatenate([drv * 2, drv * 2 + 1])
        P2, NT = T.P * 2, T.NT
        A, Am = T.amax.view(P2, NT), T.amin.view(P2, NT)
        S, Sm = T.smax.view(-1), T.smin.view(-1)
        changed = []
        if self.pa_net is not None:
            k = torch.nonzero(dirty[self.pa_net]).flatten()
            if len(k):
                changed.append(self._ports(k, elm, A, Am, S, Sm))
        pending = {}
        heap = []

        def add(rows):
            """Queue the output rows `rows` for recomputation at their stage."""
            rows = np.unique(rows)
            n = self.to_ptr[rows + 1] - self.to_ptr[rows]
            rows = rows[n > 0]
            if not len(rows):
                return
            st = self.ent_stage[self.to_ord[self.to_ptr[rows]]]
            o = np.argsort(st, kind='stable')
            st, rows = st[o], rows[o]
            cut = np.flatnonzero(np.diff(st)) + 1
            for s, part in zip(st[np.concatenate([[0], cut])].tolist(), np.split(rows, cut)):
                if s not in pending:
                    pending[s] = []
                    heapq.heappush(heap, s)
                pending[s].append(part)

        def consumers(rows):
            """The output rows of the arcs out of `rows`."""
            if not len(rows):
                return
            lo, hi = self.fr_ptr[rows], self.fr_ptr[rows + 1]
            n = hi - lo
            if not n.sum():
                return
            idx = np.repeat(lo - np.cumsum(n) + n, n) + np.arange(int(n.sum()))
            add(self.ti[self.fr_ord[idx]])

        add(seeds)
        for c in changed:
            consumers(c)
        work = nst = 0
        budget, time_budget = self.MAX_SHARE * self.n_ent, budget
        self.last = {'dirty_nets': int(len(dn)), 'stages': 0, 'entries': 0, 'full': False}
        pi, elc = T.pi.contiguous(), elm.contiguous()
        while heap:
            s = heapq.heappop(heap)
            rows = np.unique(np.concatenate(pending.pop(s)))
            lo, hi = self.to_ptr[rows], self.to_ptr[rows + 1]
            n = hi - lo
            ents = np.sort(self.to_ord[np.repeat(lo - np.cumsum(n) + n, n) + np.arange(int(n.sum()))])
            work += len(ents); nst += 1
            self.last.update(stages=nst, entries=work)
            if work > budget or (time_budget is not None and time.perf_counter() - t_start > time_budget):
                self.last['full'] = True
                return False
            ch = self._stage(s, ents, rows, A, Am, S, Sm, pi, elc)
            consumers(ch)
        return True

    def _stage(self, s, ents, rows, A, Am, S, Sm, pi, elc):
        """Recompute the entries `ents` (global, sorted; every entry into `rows`) of stage s and all their loads, with
        the fused kernel; returns the rows whose arrival or slew changed. The subset's kernel inputs and indices are
        made on the host and copied up at once."""
        T = self.T
        dev = T.dev
        off, N, loff, M = self.stages[s]
        lo, hi = self.lane_ptr[ents], self.lane_ptr[ents + 1]
        n = hi - lo
        k, m = len(ents), int(n.sum())
        lanes = np.repeat(lo - np.cumsum(n) + n, n) + np.arange(m)
        u, inv = np.unique(self.lane_row[lanes], return_inverse=True)
        d = np.empty(len(u), np.int64)
        d[inv] = self.lane_drv[lanes]
        r = len(rows)
        ei = self.e_int[ents]
        i64 = lambda x: np.asarray(x, np.int64)  # noqa: E731
        (lib, fi, t, orf, net, tag, arr, ea_s, en, t0, s0, R, inv_t, d_t, e_t, l_t, ok) = _upload(
            [ei[:, i] for i in range(7)] + [np.repeat(np.arange(k, dtype=np.int32), n), self.lane_en[lanes],
                                            self.e_flt[ents, 0], self.e_flt[ents, 1],
                                            i64(np.concatenate([rows, u])), i64(inv), d, i64(ents - off),
                                            i64(lanes - loff), self.lane_ok[lanes]], dev)
        u_t = R[r:]
        old = (A.index_select(0, R), Am.index_select(0, R), S.index_select(0, R), Sm.index_select(0, R))
        A.index_fill_(0, R, -INF); Am.index_fill_(0, R, INF); S.index_fill_(0, R, 0.0); Sm.index_fill_(0, R, INF)
        wbx, wbn = T._wb
        wbx.index_fill_(0, u_t, -INF); wbn.index_fill_(0, u_t, INF)
        sub = (lib, fi, t, orf, net, t0, tag, ea_s, en, s0, arr)
        dmax, wx, dmin, wn = T.triton.stage(T.tri_tables, sub, T.tri_sthr, pi, elc, T.tri_nto, A, Am, S, Sm,
                                            T.precision, lmin=T.lmin)
        if m:
            T._load_arr(A, Am, (inv_t, u_t, d_t, ok), wx, wn)
        v, sx = Sm.index_select(0, R), S.index_select(0, R)
        Sm.index_copy_(0, R, torch.where(torch.isfinite(v), v, sx))      # Timer: smin falls back to smax
        rec = T._rec[s]
        rec[0].index_copy_(0, e_t, dmax); rec[2].index_copy_(0, e_t, dmin)
        if m:
            rec[1].index_copy_(0, l_t, wx); rec[3].index_copy_(0, l_t, wn)
        diff = ((A.index_select(0, R) != old[0]).any(1) | (Am.index_select(0, R) != old[1]).any(1)
                | (sx != old[2]) | (Sm.index_select(0, R) != old[3]))
        return R[diff].cpu().numpy()

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
        return R[diff].cpu().numpy()
