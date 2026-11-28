"""The incremental re-timing on the CPU (attar/timing/csrc/inc.cpp): a worklist of driver rows in level-key order,
each taking every arc into it, with the C++ twin of the GPU kernels (csrc/dmp_core.h: the same bits). The engine
mirrors the timer's state on the host (copied after each full run, then kept in step: the rows, entries and lanes a
sweep recomputes are copied back to the device).

It numbers pins by its own handles: the design renumbers its pins on a netlist edit (they stay sorted by net), the
engine only updates its handle <-> pin maps, and its arcs, loads and adjacency lists grow in place (base CSR plus
linked overflow nodes; removed arcs are flagged dead), so that an edit costs what it touches (CpuEngine.edit)."""
import ctypes

import numpy as np
import torch

INF = float('inf')
P_ = ctypes.c_void_p
I64 = ctypes.c_int64


class _State(ctypes.Structure):
    _fields_ = ([(k, I64) for k in ('NT', 'lmin', 'nit', 'rit', 'K')]
                + [(k, P_) for k in ('lib', 'fi', 'to', 'orf', 'net', 'tag', 'arr', 't0', 's0', 'e_alive',
                                     'l_ent', 'l_u', 'l_sk', 'l_ok', 'l_alive')]
                + [('R0', I64), ('E0', I64)]
                + [(k, P_) for k in ('in_ptr', 'out_ptr', 'ln_ptr', 'in_idx', 'out_idx', 'ln_idx', 'in_head',
                                     'out_head', 'ln_head', 'o_val', 'o_next', 'key', 'ix1', 'ix2', 'val', 'thr',
                                     'nn', 'elm', 'sthr', 'pi', 'A', 'Am', 'S', 'Sm', 'wbx', 'wbn', 'dmax', 'dmin',
                                     'wx', 'wn', 'mark', 'stamp')]
                + [('clock', ctypes.c_uint32)])


class Grow:
    """A numpy array with spare capacity along axis 0 (a: the whole buffer, n: the rows in use)."""

    def __init__(self, init, fill, extra=0.05):
        init = np.ascontiguousarray(init)
        self.fill = fill
        cap = int(len(init) * (1 + extra)) + 1024
        self.a = np.full((cap,) + init.shape[1:], fill, init.dtype)
        self.a[:len(init)] = init
        self.n = len(init)

    def reserve(self, n):
        if n > len(self.a):
            b = np.full((max(n, 2 * len(self.a)),) + self.a.shape[1:], self.fill, self.a.dtype)
            b[:len(self.a)] = self.a
            self.a = b

    def append(self, v):
        v = np.asarray(v, self.a.dtype)
        i = self.n
        self.reserve(i + len(v))
        self.a[i:i + len(v)] = v
        self.n = i + len(v)
        return np.arange(i, i + len(v))

    @property
    def v(self):
        return self.a[:self.n]


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


class CpuEngine:
    MAX_ENTRIES = 10000         # a larger cone goes to the GPU sweep (its cost per arc is lower)

    def __init__(self, inc):
        """From the GPU sweep's index (attar.timing.incremental.Incremental) of timer inc.T: handles are the pins."""
        from . import dmp_host
        from .dmp import FIND_ROOT_MAX_ITER, NEWTON_MAX_ITER
        T = inc.T
        self.T = T
        self.lib = dmp_host.compile_lib('inc', 'inc.cpp')
        L = self.lib
        L.attar_inc_run.restype = ctypes.c_int
        L.attar_inc_run.argtypes = [P_, P_, I64, I64, P_, I64, P_, P_, I64, P_, P_, I64, P_]
        h = lambda t, dt: np.ascontiguousarray(t.detach().cpu().numpy(), dt)  # noqa: E731
        P, NT = T.P, T.NT
        self.NT = NT
        self.nit, self.rit = NEWTON_MAX_ITER, FIND_ROOT_MAX_ITER
        lib, fi, to, orf, net, tag, arr = [h(c, np.int32) for c in inc.e_cols]
        ne = len(lib)
        self.E = {k: Grow(v, f) for k, v, f in (
            ('lib', lib, 0), ('fi', fi, -1), ('to', to, 0), ('orf', orf, 0), ('net', net, 0), ('tag', tag, 0),
            ('arr', arr, 0), ('t0', h(inc.e_flt0, np.float32), 0), ('s0', h(inc.e_flt1, np.float32), 0),
            ('alive', np.ones(ne, np.uint8), 0), ('head', np.full(ne, -1, np.int32), -1),
            ('dmax', np.zeros(ne, np.float32), 0), ('dmin', np.zeros(ne, np.float32), 0),
            ('gpu', np.arange(ne, dtype=np.int64), -1))}
        l_ent = h(inc.lane_ent, np.int32)
        nl = len(l_ent)
        self.L = {k: Grow(v, f) for k, v, f in (
            ('ent', l_ent, 0), ('u', h(inc.lane_row, np.int32), 0), ('sk', h(T.n_to[inc.lane_en.long()], np.int32), 0),
            ('ok', h(inc.lane_ok, np.uint8), 0), ('alive', np.ones(nl, np.uint8), 0),
            ('wx', np.zeros(nl, np.float32), 0), ('wn', np.zeros(nl, np.float32), 0),
            ('gpu', np.arange(nl, dtype=np.int64), -1))}

        def csr(key, n, sel=None):
            idx = torch.arange(len(key), device=key.device) if sel is None else torch.nonzero(sel).flatten()
            k = key[idx]
            o = torch.argsort(k, stable=True)
            ptr = torch.zeros(n + 1, dtype=torch.long, device=key.device)
            ptr[1:] = torch.cumsum(torch.bincount(k, minlength=n), 0)
            return h(ptr, np.int64), h(idx[o], np.int32)
        self.R0, self.E0 = 2 * P, ne
        self.in_ptr, self.in_idx = csr(inc.ti, 2 * P)
        self.out_ptr, self.out_idx = csr(inc.fi.clamp(min=0), 2 * P, inc.fi >= 0)
        self.ln_ptr, self.ln_idx = csr(inc.lane_ent, ne)
        # per row (handle * 2 + transition)
        key = h(inc.row_stage, np.float64)
        self.R = {k: Grow(v, f) for k, v, f in (
            ('key', key, 0.0), ('in_head', np.full(2 * P, -1, np.int32), -1),
            ('out_head', np.full(2 * P, -1, np.int32), -1), ('mark', np.zeros(2 * P, np.uint8), 0),
            ('stamp', np.zeros(2 * P, np.uint32), 0),
            ('A', np.full((2 * P, NT), -INF, np.float32), -INF), ('Am', np.full((2 * P, NT), INF, np.float32), INF),
            ('S', np.zeros(2 * P, np.float32), 0), ('Sm', np.full(2 * P, INF, np.float32), INF),
            ('wbx', np.full(2 * P, -INF, np.float32), -INF), ('wbn', np.full(2 * P, INF, np.float32), INF))}
        # per pin handle; per net
        self.H = {k: Grow(v, f) for k, v, f in (
            ('elm', np.zeros((P, 4), np.float32), 0), ('sthr', h(T.tri_sthr, np.float32).reshape(P, 8), 0))}
        self.h_of_pin = np.arange(P, dtype=np.int64)            # design pin -> handle, on the host and the device
        self.h_dev = torch.arange(P, device=T.dev)
        self._pin_dev = None                                    # handle -> design pin (-1: gone), made when read
        self.pi = Grow(h(T.pi, np.float32).reshape(-1, 12), 0)
        self.O = {'val': Grow(np.zeros(0, np.int32), 0), 'next': Grow(np.zeros(0, np.int32), -1)}
        ix1, ix2, val, n32, thr = T.tri_tables
        self.tab = (h(ix1, np.float32), h(ix2, np.float32), h(val, np.float32), h(thr, np.float32),
                    h(n32, np.int64))
        self.K, self.lmin = int(ix1.shape[-1]), int(T.lmin)
        # recorded delays on the device: each GPU entry's / lane's stage and position (Timer._rec)
        self.ent_stage = h(inc.ent_stage, np.int64)
        self.lane_stage = self.ent_stage[h(inc.lane_ent, np.int64)]
        self.e_off = np.array([s[0] for s in inc.stages], np.int64)
        self.l_off = np.array([s[2] for s in inc.stages], np.int64)
        self.rec_ok = True                                  # the device's stages are this engine's arcs
        self.st = _State()
        self.st.clock = 0
        self.ver = None                                     # the timer state version the mirror holds
        self.last = {}

    # -- the C++ view -------------------------------------------------------------------------------------------------
    def _bind(self):
        s, E, L, R, H, O = self.st, self.E, self.L, self.R, self.H, self.O
        s.NT, s.lmin, s.nit, s.rit, s.K = self.NT, self.lmin, self.nit, self.rit, self.K
        for k, a in (('lib', E['lib']), ('fi', E['fi']), ('to', E['to']), ('orf', E['orf']), ('net', E['net']),
                     ('tag', E['tag']), ('arr', E['arr']), ('t0', E['t0']), ('s0', E['s0']), ('e_alive', E['alive']),
                     ('l_ent', L['ent']), ('l_u', L['u']), ('l_sk', L['sk']), ('l_ok', L['ok']),
                     ('l_alive', L['alive']), ('in_head', R['in_head']), ('out_head', R['out_head']),
                     ('ln_head', E['head']), ('o_val', O['val']), ('o_next', O['next']), ('key', R['key']),
                     ('elm', H['elm']), ('sthr', H['sthr']), ('pi', self.pi), ('A', R['A']), ('Am', R['Am']),
                     ('S', R['S']), ('Sm', R['Sm']), ('wbx', R['wbx']), ('wbn', R['wbn']), ('dmax', E['dmax']),
                     ('dmin', E['dmin']), ('wx', L['wx']), ('wn', L['wn']), ('mark', R['mark']),
                     ('stamp', R['stamp'])):
            setattr(s, k, _ptr(a.a))
        s.R0, s.E0 = self.R0, self.E0
        for k in ('in_ptr', 'out_ptr', 'ln_ptr', 'in_idx', 'out_idx', 'ln_idx'):
            setattr(s, k, _ptr(getattr(self, k)))
        for k, a in zip(('ix1', 'ix2', 'val', 'thr', 'nn'), self.tab):
            setattr(s, k, _ptr(a))

    # -- the state ----------------------------------------------------------------------------------------------------
    def _pins(self):
        """Handle -> design pin (-1: gone), on the device."""
        if self._pin_dev is None:
            pin = torch.full((self.H['elm'].n,), -1, dtype=torch.long, device=self.T.dev)
            pin[self.h_dev] = torch.arange(len(self.h_dev), device=self.T.dev)
            self._pin_dev = pin
        return self._pin_dev

    def _mirror(self, pi, elm):
        """Copy the timer's state to the host (after a full run) into the handles' rows."""
        T, R, NT = self.T, self.R, self.NT
        pin = self._pins()
        hs = torch.nonzero(pin >= 0).flatten()
        rows_h = torch.stack([hs * 2, hs * 2 + 1], 1).reshape(-1).cpu().numpy()
        rows_d = torch.stack([pin[hs] * 2, pin[hs] * 2 + 1], 1).reshape(-1)
        g = lambda t, w: t.detach().reshape(-1, w)[rows_d].cpu().numpy()  # noqa: E731
        R['A'].a[rows_h] = g(T.amax, NT); R['Am'].a[rows_h] = g(T.amin, NT)
        R['S'].a[rows_h] = g(T.smax, 1)[:, 0]; R['Sm'].a[rows_h] = g(T.smin, 1)[:, 0]
        R['wbx'].a[rows_h] = g(T._wb[0][:T.P * 2], 1)[:, 0]; R['wbn'].a[rows_h] = g(T._wb[1][:T.P * 2], 1)[:, 0]
        if self.rec_ok and T._rec:
            cat = lambda j: torch.cat([r[j].reshape(-1) for r in T._rec]).float().cpu().numpy()  # noqa: E731
            E, L = self.E, self.L
            eg, lg = E['gpu'].v, L['gpu'].v
            for k, j, g_ in (('dmax', 0, eg), ('dmin', 2, eg)):
                E[k].a[:E[k].n][g_ >= 0] = cat(j)[g_[g_ >= 0]]
            for k, j, g_ in (('wx', 1, lg), ('wn', 3, lg)):
                L[k].a[:L[k].n][g_ >= 0] = cat(j)[g_[g_ >= 0]]
        self.pi.reserve(len(pi))
        self.pi.n = max(self.pi.n, len(pi))
        self.pi.a[:len(pi)] = pi.detach().reshape(-1, 12).cpu().numpy()
        self.H['elm'].a[self.h_dev[T.n_to].cpu().numpy()] = elm.detach().cpu().numpy()

    def run(self, dirty, dn, seeds, pi, elm, max_ents):
        """Re-time from the seed rows (design rows, a device tensor) after wires() redid the nets dn. Returns True
        (done: the timer's device state updated), or False (over max_ents entries, or an order the sweep cannot
        follow: the device state is untouched, the caller takes over)."""
        T = self.T
        ver = T.__dict__.get('_state_ver', 0)
        if self.ver != ver:
            self._mirror(pi, elm)
        else:                                               # the redone nets' pi models and wire delays
            dn_h = dn.cpu().numpy()
            self.pi.reserve(len(pi)); self.pi.n = max(self.pi.n, len(pi))
            self.pi.a[dn_h] = pi[dn].reshape(-1, 12).cpu().numpy()
            arcs = torch.nonzero(dirty[T.n_net]).flatten()
            self.H['elm'].a[self.h_of_pin[T.n_to[arcs].cpu().numpy()]] = elm[arcs].cpu().numpy()
        self.ver = None                                     # (until the device has the result)
        sd = seeds.cpu().numpy()
        sd = np.ascontiguousarray(self.h_of_pin[sd // 2] * 2 + sd % 2, np.int64)
        self._bind()
        cap_r, cap_e, cap_l = len(self.R['key'].a), len(self.E['lib'].a), len(self.L['ent'].a)
        out_r, out_e, out_l = (np.empty(min(c, max_ents * 64 + 1024), np.int32) for c in (cap_r, cap_e, cap_l))
        nr, ne, nl = I64(), I64(), I64()
        status = self.lib.attar_inc_run(ctypes.byref(self.st), _ptr(sd), len(sd), max_ents, _ptr(out_r), len(out_r),
                                        ctypes.byref(nr), _ptr(out_e), len(out_e), ctypes.byref(ne), _ptr(out_l),
                                        len(out_l), ctypes.byref(nl))
        self.last = {'status': status, 'rows': nr.value, 'entries': ne.value, 'lanes': nl.value}
        if status:
            return False                                    # (the mirror is copied again next time)
        self._push(out_r[:nr.value].astype(np.int64), out_e[:ne.value].astype(np.int64),
                   out_l[:nl.value].astype(np.int64))
        self.ver = ver
        return True

    def _push(self, rows, ents, lanes):
        """Copy the recomputed rows (arrivals, slews, wire delays per load), and the recorded delays of the
        recomputed entries and lanes while the device's stages are this engine's, to the device."""
        T, R, NT, dev = self.T, self.R, self.NT, self.T.dev
        P2 = T.P * 2
        pk = np.concatenate([R['A'].a[rows], R['Am'].a[rows],
                             np.stack([R['S'].a[rows], R['Sm'].a[rows], R['wbx'].a[rows], R['wbn'].a[rows]], 1)], 1)
        rt = torch.from_numpy(rows).to(dev)
        r_t = self._pins()[rt // 2] * 2 + rt % 2
        pk_t = torch.from_numpy(pk).to(dev)
        T.amax.view(P2, NT).index_copy_(0, r_t, pk_t[:, :NT])
        T.amin.view(P2, NT).index_copy_(0, r_t, pk_t[:, NT:2 * NT])
        T.smax.view(-1).index_copy_(0, r_t, pk_t[:, 2 * NT])
        T.smin.view(-1).index_copy_(0, r_t, pk_t[:, 2 * NT + 1])
        T._wb[0].index_copy_(0, r_t, pk_t[:, 2 * NT + 2])
        T._wb[1].index_copy_(0, r_t, pk_t[:, 2 * NT + 3])
        if not self.rec_ok:
            return
        E, L = self.E, self.L
        for idx, gpu, stg, off, src, cols in ((ents, E['gpu'].a, self.ent_stage, self.e_off,
                                               (E['dmax'].a, E['dmin'].a), (0, 2)),
                                              (lanes, L['gpu'].a, self.lane_stage, self.l_off,
                                               (L['wx'].a, L['wn'].a), (1, 3))):
            if not len(idx):
                continue
            g = gpu[idx]
            s = stg[g]
            o = np.argsort(s, kind='stable')
            idx, g, s = idx[o], g[o], s[o]
            vals = torch.from_numpy(np.stack([src[0][idx], src[1][idx]], 1)).to(dev)
            loc = torch.from_numpy(g - off[s]).to(dev)
            cut = np.flatnonzero(np.diff(s)) + 1
            for a, b in zip(np.r_[0, cut], np.r_[cut, len(s)]):
                rec = T._rec[int(s[a])]
                rec[cols[0]].index_copy_(0, loc[a:b], vals[a:b, 0].to(rec[cols[0]].dtype))
                rec[cols[1]].index_copy_(0, loc[a:b], vals[a:b, 1].to(rec[cols[1]].dtype))

    # -- netlist edits ------------------------------------------------------------------------------------------------
    def n_alive(self):
        return int(self.E['alive'].v.sum())

    def _items(self, i, n0, ptr, idx, head):
        out = idx[ptr[i]:ptr[i + 1]].tolist() if i < n0 else []
        n = int(head.a[i])
        ov, on = self.O['val'].a, self.O['next'].a
        while n >= 0:
            out.append(int(ov[n]))
            n = int(on[n])
        return out

    def _ins(self, r):
        return self._items(r, self.R0, self.in_ptr, self.in_idx, self.R['in_head'])

    def _outs(self, r):
        return self._items(r, self.R0, self.out_ptr, self.out_idx, self.R['out_head'])

    def _lanes(self, e):
        return self._items(e, self.E0, self.ln_ptr, self.ln_idx, self.E['head'])

    def _link(self, head, i, v):
        """Prepend v to list i (head: its per-row / per-entry heads)."""
        n = int(self.O['val'].append([v])[0])
        self.O['next'].append([head.a[i]])
        head.a[i] = n

    def edit(self, e, T, t, lmap, tn, remap, newp):
        """Follow the applied edit e (attar.netlist.Edit; T already in the new pin numbering, attar.timing.inplace):
        handles for the new pins, the removed cells' arcs dead, the new cells' arcs (each combinational arc per
        transition pair its sense and tables allow, as a new timer's stages), the edited nets' loads, a level key
        for each new driver row between its fanin's and its readers'. Returns '' or why not (the engine is then
        dropped)."""
        from .ir import S_NEG, S_NON, S_POS
        d = T.d
        a = d.raw
        P0 = e.P0
        pm = np.asarray(e.pin_map, np.int64)
        p_cell, p_mpin = T.p_cell_np, a['p_mpin']
        from ..netlist import NetOf
        p_net = NetOf(a['n_ptr'])                           # (a few pins' nets)
        drv = T.net_driver
        E, L, R, H = self.E, self.L, self.R, self.H
        # handles: the old pins keep theirs (remap: Edit.pin_src by runs), the new pins get new ones
        h_old = self.h_of_pin
        dead_h = h_old[np.fromiter(e.dropped, np.int64, len(e.dropped))] if e.dropped else np.zeros(0, np.int64)
        nh = len(newp)
        hs = np.arange(H['elm'].n, H['elm'].n + nh)
        h = remap(h_old, -1)
        h[newp] = hs
        hd = remap(self.h_dev, -1)
        hd[torch.as_tensor(newp, device=T.dev)] = torch.as_tensor(hs, device=T.dev)
        self.h_dev, self._pin_dev = hd, None
        H['elm'].append(np.zeros((nh, 4), np.float32))
        H['sthr'].append(T.sink_thr[torch.as_tensor(newp, device=T.dev)].reshape(nh, 8).cpu().numpy())
        h[newp] = hs
        for g in R.values():
            g.append(np.full((2 * nh,) + g.a.shape[1:], g.fill, g.a.dtype))
        self.h_of_pin = h
        if lmap is not None:                                    # the Liberty arcs renumbered; the tables grow
            lib, al = E['lib'].v, E['alive'].v.astype(bool)
            nl = lmap[lib]
            if (nl[al] < 0).any():
                return 'the Liberty arcs of a master changed'
            lib[al] = nl[al]
            g = lambda x, dt: np.ascontiguousarray(x.detach().cpu().numpy(), dt)  # noqa: E731
            self.tab = (g(T.t_ix1, np.float32), g(T.t_ix2, np.float32), g(T.t_val, np.float32),
                        g(T.t_thr, np.float32), g(T.t_n32, np.int64))
            self.K, self.lmin = int(T.t_ix1.shape[-1]), int(T.lmin)

        def kill(x):
            E['alive'].a[x] = 0
            for ln in self._lanes(x):
                L['alive'].a[ln] = 0
        # the removed cells' arcs
        for hd in dead_h.tolist():
            for r in (2 * hd, 2 * hd + 1):
                for x in self._ins(r) + self._outs(r):
                    kill(x)
        # the new cells' arcs
        nn = self.tab[4].reshape(-1, 4, 2)
        am = t['a_master'].astype(np.int64)
        a_from, a_to = t['a_from'].astype(np.int64), t['a_to'].astype(np.int64)
        a_sense = t['a_sense'].astype(np.int64)
        allowed = lambda s, irf, orf: s == S_NON or (s == S_POS and irf == orf) or (s == S_NEG and irf != orf)  # noqa: E731
        new_rows = set()
        ent = []
        for c in sorted({int(p_cell[p]) for p in newp}):
            m = int(a['c_master'][c])
            ps = newp[p_cell[newp] == c]
            pin_at = {int(p_mpin[p]): int(p) for p in ps}
            for lid in np.nonzero(am == m)[0].tolist():
                f, g_ = pin_at.get(int(a_from[lid])), pin_at.get(int(a_to[lid]))
                if f is None or g_ is None:
                    continue
                for orf in (0, 1):
                    for irf in (0, 1):
                        if nn[lid, orf, 0] > 0 and allowed(int(a_sense[lid]), irf, orf):
                            ent.append((lid, h[f] * 2 + irf, h[g_], orf, p_net[g_]))
        if ent:
            ent = np.array(ent, np.int64)
            n = len(ent)
            ids = E['lib'].append(ent[:, 0])
            for k, v in (('fi', ent[:, 1]), ('to', ent[:, 2]), ('orf', ent[:, 3]), ('net', ent[:, 4]),
                         ('tag', np.zeros(n)), ('arr', np.ones(n)), ('t0', np.zeros(n)), ('s0', np.zeros(n)),
                         ('alive', np.ones(n)), ('head', np.full(n, -1)), ('dmax', np.zeros(n)),
                         ('dmin', np.zeros(n)), ('gpu', np.full(n, -1))):
                E[k].append(v)
            for x, (_, fi, to, orf, _n) in zip(ids.tolist(), ent.tolist()):
                self._link(R['in_head'], to * 2 + orf, x)
                self._link(R['out_head'], fi, x)
                new_rows.add(to * 2 + orf)
        # the edited nets' loads: every arc into the driver, one lane per sink (Timer._net_arcs)
        from ..io.distill import D_IN, D_INOUT
        n_ptr, pdir = a['n_ptr'], a['p_dir']
        for k in tn.tolist():
            dp = int(drv[k])
            if dp < 0:
                continue
            ps = np.arange(int(n_ptr[k]), int(n_ptr[k + 1]))
            pd = pdir[ps]
            sinks = ps[(pd == D_IN) | (pd == D_INOUT) | (pd == 3)]
            hd = int(h[dp])
            for o in (0, 1):
                for x in self._ins(2 * hd + o):
                    if not E['alive'].a[x]:
                        continue
                    for ln in self._lanes(x):
                        L['alive'].a[ln] = 0
                    E['net'].a[x] = k
                    if not len(sinks):
                        continue
                    ok = 0 if E['arr'].a[x] & 2 else 1
                    hk = h[sinks]
                    lids = L['ent'].append(np.full(len(sinks), x))
                    L['u'].append(hk * 2 + o); L['sk'].append(hk); L['ok'].append(np.full(len(sinks), ok))
                    L['alive'].append(np.ones(len(sinks))); L['wx'].append(np.zeros(len(sinks)))
                    L['wn'].append(np.zeros(len(sinks))); L['gpu'].append(np.full(len(sinks), -1))
                    for ln in lids.tolist():
                        self._link(E['head'], x, ln)
        self.rec_ok = False
        return self._keys(new_rows, e, pm, P0, h, p_net, drv)

    def _driver_rows(self, us, h, p_net, drv):
        """The driver rows of load rows us (their nets' drivers, same transition; -1: none)."""
        if not len(us):
            return np.zeros(0, np.int64)
        us = np.asarray(us, np.int64)
        p = self._pins()[torch.as_tensor(us // 2, device=self.T.dev)].cpu().numpy()
        dp = np.where(p >= 0, drv[p_net[np.maximum(p, 0)]], -1)
        return np.where(dp >= 0, h[np.maximum(dp, 0)] * 2 + us % 2, -1)

    def _readers(self, r):
        """The rows whose arcs read row r or one of its loads."""
        out = set()
        for x in self._outs(r):
            if self.E['alive'].a[x]:
                out.add(int(self.E['to'].a[x]) * 2 + int(self.E['orf'].a[x]))
        for x in self._ins(r):
            if not self.E['alive'].a[x]:
                continue
            for ln in self._lanes(x):
                if self.L['alive'].a[ln]:
                    for y in self._outs(int(self.L['u'].a[ln])):
                        if self.E['alive'].a[y]:
                            out.add(int(self.E['to'].a[y]) * 2 + int(self.E['orf'].a[y]))
        return out

    def _keys(self, new_rows, e, pm, P0, h, p_net, drv):
        """Level keys of the new driver rows: above every row they read, below every row that reads them (new rows
        in order, each in the middle of what is left); and an old pin moved to another net must be read only by
        rows above its new driver."""
        key = self.R['key'].a
        E = self.E
        parents, children, lb, ub = {}, {}, {}, {}
        fan = {r: [int(E['fi'].a[x]) for x in self._ins(r) if E['alive'].a[x] and E['fi'].a[x] >= 0]
               for r in new_rows}
        us = sorted({u for v in fan.values() for u in v})
        drow = dict(zip(us, self._driver_rows(us, h, p_net, drv).tolist()))
        for r in new_rows:
            ps, lo = set(), -np.inf
            for u in fan[r]:
                dr = drow[u]
                if dr < 0:
                    continue
                if dr in new_rows:
                    ps.add(dr)
                else:
                    lo = max(lo, key[dr])
            cs, hi = set(), np.inf
            for t_ in self._readers(r):
                if t_ in new_rows:
                    cs.add(t_)
                else:
                    hi = min(hi, key[t_])
            parents[r], children[r], lb[r], ub[r] = ps, cs, lo, hi
        order, seen = [], set()                                 # parents first
        pending = set(new_rows)
        while pending:
            ready = [r for r in pending if parents[r] <= seen]
            if not ready:
                return 'the edit closed a combinational loop'
            for r in sorted(ready):
                order.append(r); seen.add(r); pending.discard(r)
        for r in reversed(order):                               # the room below each row's readers, transitively
            for c in children[r]:
                ub[r] = min(ub[r], ub[c])
        for r in order:
            lo = max([lb[r]] + [key[p] for p in parents[r]])
            hi = ub[r]
            k = lo + 1.0 if not np.isfinite(hi) else (lo + (hi - lo) / 2 if np.isfinite(lo) else hi - 1.0)
            if not (lo < k < hi):
                return 'no level key between a new row\'s fanin and readers'
            key[r] = k
        for p, net in e.moves.items():                          # old pins on another net
            q = int(pm[p])
            if p >= P0 or q < 0:
                continue
            dp = int(drv[net])
            if dp < 0:
                continue
            for o in (0, 1):
                dk = key[int(h[dp]) * 2 + o]
                for x in self._outs(int(h[q]) * 2 + o):
                    if E['alive'].a[x] and not key[int(E['to'].a[x]) * 2 + int(E['orf'].a[x])] > dk:
                        return 'a moved pin\'s readers are not above its new driver'
        return ''
