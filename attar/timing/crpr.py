"""Clock reconvergence pessimism removal (CRPR), as OpenSTA's (search/Crpr.cc, Search.cc pruneCrprArrivals).

With propagated clocks, a register check compares a late launch clock path with an early capture clock path (setup;
the opposite for hold). Where the two share clock tree pins, OpenSTA credits the slack back with the late / early
arrival difference at their last common pin (CheckCrpr::findCrpr: crpr = min(source delta, target delta), each
|late - early| of that path's transition at the pin; crpr_mode same_pin). The credit depends on the launching
register, so an endpoint's slack is the worst over launch clock pins of required + credit - arrival: OpenSTA keeps a
pin's arrivals per launch clock pin (the tag's crpr clk path) and prunes those that cannot be worst (an arrival below
another's less that one's largest possible credit, the late / early difference at its own clock pin).

Here, after the timer's run (the arc and wire delays it recorded, the clock trees' arrivals), every data pin that
feeds a path keeps its (arrival, launch clock pin) entries per transition and propagated clock tag, setup (max) and
hold (min) apart: variable-length lists in one flat buffer, merged level by level (the latest per launch clock pin,
then the dominance pruning). Nothing is truncated, so the worst slack is exact. The register checks then take, per
entry, the credit at the lowest common ancestor of the launch and capture clock pins in the clock trees (binary
lifting). Paths from input ports (no clock path) get no credit.
Not modelled: generated clocks (their source paths), output delays to generated clocks, path delay exceptions
(set_max_delay: no credit), latches; at a clock mux the first clock input is the clock path."""
import numpy as np
import torch

INF = float('inf')


class _Store:
    """Entries (value, launch clock pin row or -1) per key (row * G + g): one flat buffer, each key's entries
    contiguous (start, count)."""

    def __init__(self, n_keys, dev):
        self.dev = dev
        self.start = torch.zeros(n_keys, device=dev, dtype=torch.long)
        self.cnt = torch.zeros(n_keys, device=dev, dtype=torch.long)
        self.val = torch.empty(1 << 16, device=dev)
        self.id = torch.empty(1 << 16, device=dev, dtype=torch.long)
        self.n = 0

    def put(self, key, val, cid):
        """key ascending, each key once in the store's life."""
        m = len(key)
        if not m:
            return
        if self.n + m > len(self.val):
            cap = max(2 * len(self.val), self.n + m)
            self.val = torch.cat([self.val[:self.n], torch.empty(cap - self.n, device=self.dev)])
            self.id = torch.cat([self.id[:self.n], torch.empty(cap - self.n, device=self.dev, dtype=torch.long)])
        self.val[self.n:self.n + m] = val
        self.id[self.n:self.n + m] = cid
        ku, c = torch.unique_consecutive(key, return_counts=True)
        self.start[ku] = self.n + torch.cumsum(c, 0) - c
        self.cnt[ku] = c
        self.n += m

    def get(self, key):
        """Every entry of keys key (m,): (index into key, value, launch clock pin row)."""
        c = self.cnt[key]
        rep = torch.repeat_interleave(torch.arange(len(key), device=self.dev), c)
        pos = self.start[key][rep] + torch.arange(len(rep), device=self.dev) - (torch.cumsum(c, 0) - c)[rep]
        return rep, self.val[pos], self.id[pos]


class Crpr:
    """Built once per timer (attar.timing.sta.Timer); run() after each timing run, then Timer._checks(crpr=self)."""

    def __init__(self, T):
        self.T = T
        dev, P, NT = T.dev, T.P, T.NT
        self.dev, self.P, self.R, self.NT = dev, P, 2 * P, NT
        C = len(T.clocks)
        tags = [2 * c + e for c in range(C) if T.prop_clk[c] for e in (0, 1)]
        self.G = G = len(tags)
        self.Gt = torch.as_tensor(tags, device=dev, dtype=torch.long)
        self.ar = torch.arange(G, device=dev)
        clk_net = T.clk_of >= 0
        pclk = T.pclk
        self.clk_net = clk_net
        # crprPossible: the same clock, or clocks on common source pins (create_clock -add)
        src = [set(c['sources']) for c in T.clocks]
        poss = np.array([[i == j or bool(src[i] & src[j]) for j in range(C)] for i in range(C)], bool).reshape(C, C)
        self.poss = torch.as_tensor(poss, device=dev)
        # -- clock trees: every propagated clock pin's previous pin on its clock path ----------------------------------
        n_fr, n_to = T.n_fr.cpu().numpy(), T.n_to.cpu().numpy()
        c_fr, c_to, c_arr = T.c_fr.cpu().numpy(), T.c_to.cpu().numpy(), T.c_arr.cpu().numpy()
        parent = np.full(P, -1, np.int64)
        m = pclk[n_fr] & pclk[n_to]
        parent[n_to[m]] = n_fr[m]
        m = pclk[c_fr] & pclk[c_to] & c_arr
        if m.any():
            fr_, to_ = np.unique(np.stack([c_fr[m], c_to[m]], 1), axis=0).T
        else:
            fr_ = to_ = np.zeros(0, np.int64)
        self.n_multi = int((np.bincount(to_, minlength=P) > 1).sum())     # clock muxes: the first input is kept
        o = np.arange(len(to_))[::-1]
        parent[to_[o]] = fr_[o]
        nodes = np.nonzero(pclk)[0]
        cidx = np.full(P, -1, np.int64); cidx[nodes] = np.arange(len(nodes))
        lat_pins = getattr(T, '_lat_pins', None)                 # pin clock latencies: the clock is ideal there, no
        if lat_pins is not None and len(lat_pins):                # credit (Crpr: both clock infos propagated)
            cidx[lat_pins] = -1
        idx = np.arange(len(nodes))
        up0 = np.where(parent[nodes] >= 0, cidx[np.maximum(parent[nodes], 0)], idx)
        up0 = np.where(up0 >= 0, up0, idx)
        depth = (up0 != idx).astype(np.int64)                         # pointer jumping: distance to the root
        jump = up0.copy()
        while True:
            nj = jump[jump]
            if np.array_equal(nj, jump):
                break
            depth = depth + depth[jump]
            jump = nj
        root = jump
        L = max(1, int(depth.max(initial=0)).bit_length())
        ups = [up0]
        for _ in range(1, L):
            ups.append(ups[-1][ups[-1]])
        T_ = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)  # noqa: E731
        self.up = T_(np.stack(ups)); self.depth = T_(depth); self.root = T_(root)
        self.nodes, self.cidx = T_(nodes), T_(cidx)
        self.par = T_(T.par.astype(np.int64))
        # -- the data graph, level by level as the timer ran it (its recorded delays) -----------------------------------
        # units: (arc of the level, its load or -1 for the arc's own output, the target row); launch arcs start entries
        # (a propagated clock pin into data), data arcs carry their input's; targets are net sinks and the outputs that
        # feed arcs (an adder's carry into its sum). Setup (late) and hold (early) in one pass: early keys are offset by
        # RG and their values negated, so that both merge as the latest.
        RG = self.R * G
        self.RG = RG
        cn = torch.as_tensor(clk_net, device=dev); pc = torch.as_tensor(pclk, device=dev)
        lv_raw = []
        for cb in T.cb_lev:
            if cb is None:
                continue
            lib, f, t_, orf, irf, ea, en, arr = cb
            launch = arr & pc[f] & ~cn[t_]
            data = arr & ~cn[f] & ~cn[t_]
            lv_raw.append((f * 2 + irf, t_ * 2 + orf, launch, data, ea, en))
        feeds = torch.zeros(self.R, device=dev, dtype=torch.bool)       # rows that are an arc's input
        for fi, _ti, _launch, data, _ea, _en in lv_raw:
            feeds[fi[data]] = True
        two = lambda x: torch.cat([x, x + RG])  # noqa: E731
        self.levs = [None] if T.lc is not None else []                 # ideal clocks' launches: no propagated tag
        for fi, ti, launch, data, ea, en in lv_raw:
            keep = launch | data
            sk = T.n_to[en]
            ld = torch.nonzero(keep[ea] & ~cn[sk]).flatten()
            dv = torch.nonzero(keep & feeds[ti]).flatten()
            a = torch.cat([ea[ld], dv])
            if not len(a):
                self.levs.append(None)
                continue
            w = torch.cat([ld, torch.full_like(dv, -1)])
            tgt = torch.cat([sk[ld] * 2 + (ti % 2)[ea[ld]], ti[dv]])
            fa, la = fi[a], launch[a]
            n = len(a)
            dk = torch.nonzero(~la).flatten(); lk = torch.nonzero(la).flatten()
            kG = lambda r: (r[:, None] * G + self.ar).flatten()  # noqa: E731
            d_src, d_tgt = two(kG(fa[dk])), two(kG(tgt[dk]))
            d_u = torch.cat([dk.repeat_interleave(G), dk.repeat_interleave(G) + n])     # into (late, early) units
            l_tgt = two(kG(tgt[lk]))
            l_u = torch.cat([lk.repeat_interleave(G), lk.repeat_interleave(G) + n])
            l_row = fa[lk].repeat_interleave(G).repeat(2)
            l_tag = self.Gt.repeat(len(lk)).repeat(2)
            l_neg = torch.cat([torch.zeros(len(lk) * G, device=dev, dtype=torch.bool),
                               torch.ones(len(lk) * G, device=dev, dtype=torch.bool)])
            self.levs.append((a, w, tgt, (d_src, d_tgt, d_u), (l_tgt, l_u, l_row, l_tag, l_neg)))
        # nets driven by input ports
        self.pa = None
        if len(T.pa):
            keep = ~cn[T.n_to[T.pa]]
            self.pa = (T.n_fr[T.pa], T.n_to[T.pa], keep)
        # input ports' arrivals of propagated tags (set_input_delay -clock): entries without a clock path
        gi = torch.full((NT,), -1, device=dev, dtype=torch.long)
        gi[self.Gt] = self.ar
        self.seeds = []
        for ix in (T.in_ix_max, T.in_ix_min):
            row, tag = ix // NT, ix % NT
            m = gi[tag] >= 0
            k = row[m] * G + gi[tag[m]]
            k = torch.unique(k)
            self.seeds.append(k)
        self.uncovered = 0
        self.entries = 0

    # -- propagation ------------------------------------------------------------------------------------------------------
    def run(self):
        """The entries of the timer's last run, setup (late) and hold (early)."""
        T = self.T
        A = T.amax.reshape(self.R, self.NT); B = T.amin.reshape(self.R, self.NT)
        d = (A - B).abs()
        self.D = torch.where(torch.isfinite(d), d, torch.zeros_like(d))  # late - early, per clock pin and tag
        self.S = self._propagate(A, B)
        self.entries = self.S.n
        return self

    def _propagate(self, A, B):
        T, G, RG = self.T, self.G, self.RG
        dev = self.dev
        S = _Store(2 * RG, dev)
        kl, ke = self.seeds
        S.put(torch.cat([kl, ke + RG]), torch.cat([A[kl // G, self.Gt[kl % G]], -B[ke // G, self.Gt[ke % G]]]),
              torch.full((len(kl) + len(ke),), -1, device=dev, dtype=torch.long))
        if self.pa is not None:
            f, t_, keep = self.pa
            ks, vs, ids = [], [], []
            for rf, wx, wn in T._pa_w:
                src = ((f[keep] * 2 + rf)[:, None] * G + self.ar).flatten()
                dst = ((t_[keep] * 2 + rf)[:, None] * G + self.ar).flatten()
                w2 = torch.cat([wx[keep].repeat_interleave(G), -wn[keep].repeat_interleave(G)])
                rep, v, cid = S.get(torch.cat([src, src + RG]))
                ks.append(torch.cat([dst, dst + RG])[rep]); vs.append(v + w2[rep]); ids.append(cid)
            self._merge(S, torch.cat(ks), torch.cat(vs), torch.cat(ids))
        for lv, rec in zip(self.levs, T._rec):
            if lv is None:
                continue
            a, w, trow, (d_src, d_tgt, d_u), (l_tgt, l_u, l_row, l_tag, l_neg) = lv
            ow = w >= 0
            z = torch.zeros(len(a), device=dev)
            dly = torch.cat([rec[0][a], -rec[2][a]])                     # per (late, early) unit
            # a load's wire delay: the largest of its driver's arcs (the timer's, Timer._load_arr)
            wbx, wbn = T._wb
            wir = torch.cat([torch.where(ow, wbx[trow], z), torch.where(ow, -wbn[trow], z)])
            rep, v, cid = S.get(d_src)                                   # data arcs: their input's entries
            u = d_u[rep]
            v = (v + dly[u]) + wir[u]                                    # the timer's order: (arrival + gate) + wire
            key = d_tgt[rep]
            if len(l_u):                                                 # launch arcs: the clock pin's arrival per tag
                la = torch.where(l_neg, -B[l_row, l_tag], A[l_row, l_tag])
                v = torch.cat([v, (la + dly[l_u]) + wir[l_u]])
                key = torch.cat([key, l_tgt])
                cid = torch.cat([cid, l_row])
            ok = torch.isfinite(v)
            self._merge(S, key[ok], v[ok], cid[ok])
        return S

    def _merge(self, S, key, val, cid):
        """Candidates (key = row * G + g, value, launch clock pin row or -1) into S: the latest (late) per key and
        clock pin, then those no other entry of the key dominates (an entry is never worst if another's value less
        its largest credit, the late / early difference at its own clock pin, is beyond it). Early keys hold negated
        arrivals: the same rule."""
        if not len(key):
            return
        R, G = self.R, self.G
        comb = key * (R + 1) + (cid + 1)
        u, inv = torch.unique(comb, return_inverse=True)
        ga = torch.full((len(u),), -INF, device=self.dev).scatter_reduce_(0, inv, val, 'amax')
        gk = u // (R + 1); gid = u % (R + 1) - 1
        mc = torch.where(gid >= 0, self.D[gid.clamp(min=0), self.Gt[gk % G]], torch.zeros_like(ga))
        ku, kinv = torch.unique_consecutive(gk, return_inverse=True)
        thr = torch.full((len(ku),), -INF, device=self.dev).scatter_reduce_(0, kinv, ga - mc, 'amax')
        keep = ga >= thr[kinv]
        S.put(gk[keep], ga[keep], gid[keep])

    # -- the checks -------------------------------------------------------------------------------------------------------
    def _lca(self, a, b):
        """Lowest common ancestor of clock tree nodes a, b (compact indices, same tree)."""
        up, dep = self.up, self.depth
        da, db = dep[a], dep[b]
        sw = da < db
        a, b = torch.where(sw, b, a), torch.where(sw, a, b)
        diff = (da - db).abs()
        for j in range(len(up)):
            a = torch.where(((diff >> j) & 1).bool(), up[j][a], a)
        eq = a == b
        for j in range(len(up) - 1, -1, -1):
            ua, ub = up[j][a], up[j][b]
            m = ua != ub
            a, b = torch.where(m, ua, a), torch.where(m, ub, b)
        return torch.where(eq, a, up[0][a])

    def _credit(self, k, gi, ids):
        """CRPR of entries (register check k, tag index gi, launch clock pin row ids, -1: none)."""
        T = self.T
        L = ids.clamp(min=0) // 2; lrf = ids.clamp(min=0) % 2
        Cp = T.k_fr[k]; crf = T.k_crf[k]; ctag = T.k_ctag[k]
        g = self.Gt[gi]
        la, ca = self.cidx[L], self.cidx[Cp]
        ok = (ids >= 0) & (la >= 0) & (ca >= 0) & T.k_prop[k] & self.poss[g // 2, T.k_ck[k]]
        la, ca = la.clamp(min=0), ca.clamp(min=0)
        ok &= self.root[la] == self.root[ca]
        X = self.nodes[self._lca(la, ca)]
        xs = lrf ^ self.par[L] ^ self.par[X]                           # each path's transition at the common pin
        xt = crf ^ self.par[Cp] ^ self.par[X]
        c = torch.minimum(self.D[X * 2 + xs, g], self.D[X * 2 + xt, ctag])
        return torch.where(ok, c, torch.zeros_like(c))

    def _apply(self, rf, req, v, late):
        T, G = self.T, self.G
        nk = len(T.k_to)
        keys = ((T.k_to * 2 + rf)[:, None] * G + self.ar).flatten()     # (nk * G)
        rep, val, cid = self.S.get(keys if late else keys + self.RG)
        if not late:
            val = -val
        k, gi = rep // G, rep % G
        r = req[k, self.Gt[gi]]
        cr = self._credit(k, gi, cid)
        t = (r + cr - val) if late else (val - r + cr)
        t = torch.where(torch.isnan(t), torch.full_like(t, INF), t)
        out_t = torch.full((nk * G,), INF, device=self.dev).scatter_reduce_(0, rep, t, 'amin').view(nk, G)
        top = torch.full((nk * G,), -INF if late else INF, device=self.dev).scatter_reduce_(
            0, rep, val, 'amax' if late else 'amin').view(nk, G)
        A = (T.amax if late else T.amin).reshape(self.R, self.NT)[T.k_to * 2 + rf][:, self.Gt]
        cov = (top == A) | ((top - A).abs() <= 1e-3)
        self.uncovered += int((~cov & torch.isfinite(A)).sum())
        out = v.clone()
        out[:, self.Gt] = torch.where(cov, out_t, v[:, self.Gt])
        return out

    def setup(self, rf, req, v):
        """Register setup checks of transition rf: v (nk, NT) required - arrival per tag; the propagated tags'
        columns become the min over entries of required + crpr - arrival."""
        return self._apply(rf, req, v, True)

    def hold(self, rf, req, v):
        """Register hold checks: the min over entries of arrival + crpr - required."""
        return self._apply(rf, req, v, False)
