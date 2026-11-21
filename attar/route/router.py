"""A global router on OpenROAD's gcell grid and edge capacities (attar.rules.routing.capacity), in PyTorch (any device).

1. Pins (pins_from_ir): each pin's gcell is that of its center (Design.pin_pos; GlobalRouter's on-grid position is
   the most common gcell of its connection-layer shapes' middles: the same for one-shape pins), its connection layer
   its LEF layer (an IO pin: its DEF layer, timing.json 'io_layer', else Design.pin_layers by its die edge), its
   position the lower-left of its LEF bounding box at the placement (GlobalRouter: of its last shape).
2. Topology: per net, OpenROAD's Prim-Dijkstra tree (alpha 0: a rectilinear MST, steinerized; attar.timing.steiner_big)
   over the gcells of its pins; each tree edge is a branch.
3. 2D routing (as gsta's ggr): every branch an L, then negotiated rip-up and reroute: branches over an overflowing edge,
   in hashed batches, take the cheapest Z / L pattern (HVH with any middle column, VHV with any middle row, up to
   `margin` gcells beyond their box); an edge costs 1 + history + pres x (excess with one more wire); history grows on
   overflowing edges each iteration, pres by pres_grow.
4. Layer assignment: every straight piece cut into chunks of at most la_chunk gcells, each chunk on one layer of its
   direction in the routing range, shortest pieces first in batches: the lowest layer with room along all of the
   chunk, else the one it overflows least (+ step per layer up); then rounds of rip-up and reassignment of the chunks
   over a full edge; consecutive chunks of a piece on one layer are merged back into one segment.
5. Routes: per piece a planar segment between gcell centers; at every tree point a via stack over the layers of the
   pieces and pins there; a net within one gcell: a stack from its pins' layers to the lowest routing layer.
Not a port of FastRoute: no FLUTE, no maze stage, no 3D layer-assignment DP; its numbers are checked against
OpenROAD's (tools/validate_router.py), not claimed equal.
"""
import json
import os
import time

import numpy as np
import torch

from .groutes import GlobalRoutes

_ORI = ((1, 0, 0, 1), (-1, 0, 0, -1), (0, -1, 1, 0), (0, 1, -1, 0), (-1, 0, 0, 1), (1, 0, 0, -1), (0, -1, -1, 0),
        (0, 1, 1, 0))       # attar.io.deffile.ORIENTS order: N S W E FN FS FW FE


def _orient_table():
    from ..io.deffile import ORIENTS
    m = {'N': (1, 0, 0, 1), 'S': (-1, 0, 0, -1), 'W': (0, -1, 1, 0), 'E': (0, 1, -1, 0), 'FN': (-1, 0, 0, 1),
         'FS': (1, 0, 0, -1), 'FW': (0, -1, -1, 0), 'FE': (0, 1, 1, 0)}
    return np.array([m[o] for o in ORIENTS], np.int64)


def pins_from_ir(d, g, lo, hi, pos=None):
    """Per pin: pt (P, 2) dbu, grid (P, 2) gcell index, layer (P,) routing level, from the IR (see the module doc)."""
    a, n = d.raw, d.names
    dbu = d.dbu
    P = len(a['p_cell'])
    pp = d.pin_pos(pos).detach().double().cpu().numpy()
    ctr = np.rint(pp * dbu).astype(np.int64)
    x0, y0 = g['die'][0], g['die'][1]
    tile, nx, ny = g['tile'], g['nx'], g['ny']
    gx = np.clip((ctr[:, 0] - x0) // tile, 0, nx - 1); gy = np.clip((ctr[:, 1] - y0) // tile, 0, ny - 1)
    mp = a['p_mpin'].astype(np.int64)
    cell = a['p_cell'].astype(np.int64)
    layer = np.full(P, lo, np.int64)
    cp = mp >= 0
    if 'm_pin_layer' in a:
        ml = a['m_pin_layer'].astype(np.int64)[mp[cp]]
        layer[cp] = np.where(ml >= 0, ml + 1, 1)
    # IO pins: their DEF layer
    io = np.nonzero(~cp)[0]
    if len(io):
        tj = os.path.join(d.path, 'timing.json')
        iol = {}
        if os.path.exists(tj):
            with open(tj) as f:
                iol = json.load(f).get('io_layer') or {}
        lid = {nm: i + 1 for i, nm in enumerate(n['layers'])}
        from ..io.export import pin_layers
        h, v = pin_layers(d)
        die = g['die']
        for p in io.tolist():
            nm = iol.get(str(int(cell[p])))
            if nm in lid:
                layer[p] = lid[nm]
            else:
                x, y = ctr[p]
                side = int(np.argmin([y - die[1], die[2] - x, die[3] - y, x - die[0]]))
                layer[p] = lid[v if side in (0, 2) else h]
    layer = np.minimum(layer, hi)
    # pt: the lower-left of the pin's LEF box at the placement
    pt = ctr.copy()
    if cp.any():
        P_ = (d.pos if pos is None else pos)
        cpos = np.rint(P_.detach().double().cpu().numpy() * dbu).astype(np.int64)
        om = _orient_table()[d.orient.cpu().numpy().astype(np.int64)]
        msz = a['m_size'].astype(np.int64); cm = a['c_master'].astype(np.int64)
        c = cell[cp]; mi = cm[c]
        A, B, C, D = (om[c, k] for k in range(4))
        w, h_ = msz[mi, 0], msz[mi, 1]
        ox = cpos[c, 0] - np.minimum(0, A * w + B * h_); oy = cpos[c, 1] - np.minimum(0, C * w + D * h_)
        r = a['m_pin_box'].astype(np.int64)[mp[cp]]
        xa, xb = A * r[:, 0] + B * r[:, 1], A * r[:, 2] + B * r[:, 3]
        ya, yb = C * r[:, 0] + D * r[:, 1], C * r[:, 2] + D * r[:, 3]
        pt[cp, 0] = np.minimum(xa, xb) + ox; pt[cp, 1] = np.minimum(ya, yb) + oy
    return pt, np.stack([gx, gy], 1), layer


def _pieces(seg, typ, mid):
    """The (up to) three straight pieces of each routed branch: (branch, dir 0 h / 1 v, line, a, b) with a < b."""
    ax, ay, bx, by = seg.unbind(1)
    m = mid
    t0 = typ == 0
    # type 0 (HVH): h on ay from ax to m, v on m from ay to by, h on by from m to bx
    # type 1 (VHV): v on ax from ay to m, h on m from ax to bx, v on bx from m to by
    d1 = torch.where(t0, 0, 1); l1 = torch.where(t0, ay, ax); a1 = torch.where(t0, ax, ay); b1 = m
    d2 = torch.where(t0, 1, 0); l2 = m; a2 = torch.where(t0, ay, ax); b2 = torch.where(t0, by, bx)
    d3 = torch.where(t0, 0, 1); l3 = torch.where(t0, by, bx); a3 = m; b3 = torch.where(t0, bx, by)
    br = torch.arange(len(seg), device=seg.device)
    P = torch.stack([torch.cat([br, br, br]), torch.cat([d1, d2, d3]), torch.cat([l1, l2, l3]),
                     torch.cat([a1, a2, a3]), torch.cat([b1, b2, b3])], 1)
    lo_, hi_ = torch.minimum(P[:, 3], P[:, 4]), torch.maximum(P[:, 3], P[:, 4])
    P[:, 3], P[:, 4] = lo_, hi_
    return P[lo_ < hi_]


class _Grid2D:
    """2D usage, history and cost prefix sums: horizontal edges per row [ny][nx - 1], vertical per column [nx][ny - 1]."""

    def __init__(self, ch, cv, dev):
        self.ch, self.cv = ch.to(dev).float(), cv.to(dev).float()
        self.ny, self.nx = ch.shape[0], ch.shape[1] + 1
        self.hh = torch.zeros_like(self.ch); self.hv = torch.zeros_like(self.cv)
        self.dev = dev

    def usage(self, pieces):
        nx, ny, dev = self.nx, self.ny, self.dev
        uh = torch.zeros(ny * (nx + 1), device=dev); uv = torch.zeros(nx * (ny + 1), device=dev)
        for dirv, u, L in ((0, uh, nx + 1), (1, uv, ny + 1)):
            q = pieces[pieces[:, 1] == dirv]
            one = torch.ones(len(q), device=dev)
            u.index_add_(0, q[:, 2] * L + q[:, 3], one); u.index_add_(0, q[:, 2] * L + q[:, 4], -one)
        self.uh = uh.view(ny, nx + 1).cumsum(1)[:, :nx - 1]
        self.uv = uv.view(nx, ny + 1).cumsum(1)[:, :ny - 1]

    def overflow(self):
        oh = (self.uh - self.ch).clamp(min=0); ov = (self.uv - self.cv).clamp(min=0)
        return float(torch.ceil(oh).sum() + torch.ceil(ov).sum()), float(max(oh.max(), ov.max()))

    def costs(self, pres, hist=0.0):
        if hist:
            self.hh += hist * (self.uh > self.ch); self.hv += hist * (self.uv > self.cv)
        z = lambda n: torch.zeros(n, 1, device=self.dev)  # noqa: E731
        c_h = 1 + self.hh + pres * (self.uh + 1 - self.ch).clamp(min=0)
        c_v = 1 + self.hv + pres * (self.uv + 1 - self.cv).clamp(min=0)
        self.PH = torch.cat([z(self.ny), c_h.cumsum(1)], 1)
        self.PV = torch.cat([z(self.nx), c_v.cumsum(1)], 1)
        self.OH = torch.cat([z(self.ny), (self.uh > self.ch).float().cumsum(1)], 1)
        self.OV = torch.cat([z(self.nx), (self.uv > self.cv).float().cumsum(1)], 1)

    @staticmethod
    def _span(Pf, line, a, b, L):
        lo, hi = torch.minimum(a, b), torch.maximum(a, b)
        return Pf[line * L + hi] - Pf[line * L + lo]

    def h(self, y, a, b, over=False):
        return self._span((self.OH if over else self.PH).view(-1), y, a, b, self.nx)

    def v(self, x, a, b, over=False):
        return self._span((self.OV if over else self.PV).view(-1), x, a, b, self.ny)

    def path(self, seg, typ, mid, over=False, via=0.0):
        ax, ay, bx, by = seg.unbind(1)
        # mid is a column for type 0 and a row for type 1; both costs are computed and one is kept, so each formula
        # clamps mid into its own axis (the discarded one would otherwise index past the grid when nx != ny)
        m0, m1 = mid.clamp(max=self.nx - 1), mid.clamp(max=self.ny - 1)
        c0 = self.h(ay, ax, m0, over) + self.v(m0, ay, by, over) + self.h(by, m0, bx, over)
        c1 = self.v(ax, ay, m1, over) + self.h(m1, ax, bx, over) + self.v(bx, m1, by, over)
        if via:
            c0 = c0 + via * (ay != by) * ((m0 != ax).float() + (m0 != bx).float())
            c1 = c1 + via * (ax != bx) * ((m1 != ay).float() + (m1 != by).float())
        return torch.where(typ == 0, c0, c1)

    def best(self, seg, margin, via, chunk=1 << 22):
        """The cheapest pattern of each branch: (type, mid)."""
        nB = len(seg)
        dev = self.dev
        typ = torch.zeros(nB, dtype=torch.long, device=dev); mid = seg[:, 2].clone()
        best = torch.full((nB,), float('inf'), device=dev)
        for t, (i0, i1, n) in ((0, (0, 2, self.nx)), (1, (1, 3, self.ny))):
            lo = (torch.minimum(seg[:, i0], seg[:, i1]) - margin).clamp(min=0)
            hi = (torch.maximum(seg[:, i0], seg[:, i1]) + margin).clamp(max=n - 1)
            w = hi - lo + 1
            order = torch.argsort(w)
            W_sorted = w[order]
            start = 0
            Wh = W_sorted.cpu().numpy()
            while start < nB:
                end = min(nB, start + max(1, chunk // max(1, int(Wh[start]))))
                W = int(Wh[end - 1])
                k = max(1, min(end - start, chunk // max(1, W)))
                W = int(Wh[start + k - 1])
                idx = order[start:start + k]
                mm = lo[idx, None] + torch.arange(W, device=dev)[None]
                ok = mm <= hi[idx, None]
                mm = torch.minimum(mm, hi[idx, None])
                s = seg[idx, None, :].expand(-1, W, -1).reshape(-1, 4)
                c = self.path(s, torch.full((len(s),), t, device=dev), mm.reshape(-1), via=via).view(k, W)
                c = torch.where(ok, c, torch.full_like(c, float('inf')))
                cb, j = c.min(1)
                better = cb < best[idx]
                best[idx] = torch.where(better, cb, best[idx])
                typ[idx] = torch.where(better, torch.full_like(j, t), typ[idx])
                mid[idx] = torch.where(better, mm[torch.arange(k, device=dev), j], mid[idx])
                start += k
        return typ, mid


def _hash(i, it):
    x = (i * 2654435761 + it * 97) & 0xFFFFFFFF
    x ^= x >> 16; x = (x * 0x7feb352d) & 0xFFFFFFFF; x ^= x >> 15; x = (x * 0x846ca68b) & 0xFFFFFFFF; x ^= x >> 16
    return x


def global_route(d, pos=None, adjustment=None, layers=None, iterations=30, batches=8, margin=10, via=2.0, pres=0.5,
                 pres_grow=1.2, pres_max=100.0, hist=1.0, la_batches=64, la_step=0.5, la_iterations=10, la_chunk=10, device=None, verbose=True):
    """Route every net of d at pos (default: its placement). Returns (GlobalRoutes, stats)."""
    from ..rules import routing as R
    from ..timing import steiner_big
    t0 = time.time()
    dev = torch.device(device) if device is not None else d.device
    g, h3, v3 = R.capacity(d, adjustment=adjustment, layers=layers, pos=pos)
    lo0, hi0 = g['layers']                              # 0-based layer indices of the routing range
    lo, hi = lo0 + 1, hi0 + 1                           # routing levels
    tile, nx, ny = g['tile'], g['nx'], g['ny']
    X0, Y0 = g['die'][0], g['die'][1]
    horiz = np.asarray(g['horizontal']).astype(bool)
    t_cap = time.time() - t0
    a = d.raw
    pt, gp, play = pins_from_ir(d, g, lo, hi, pos)
    n_ptr = a['n_ptr'].astype(np.int64)
    E = len(n_ptr) - 1
    p_net = np.repeat(np.arange(E), np.diff(n_ptr))
    deg = np.diff(n_ptr)
    # per net, the distinct gcells of its pins (first pin's first)
    key = p_net * (nx * ny) + gp[:, 1] * nx + gp[:, 0]
    ok = deg[p_net] >= 2
    uk, first = np.unique(key[ok], return_index=True)
    src = np.nonzero(ok)[0][first]
    o = np.lexsort((src, p_net[src]))                   # per net, in pin order
    src = src[o]
    pn = p_net[src]
    cnt = np.bincount(pn, minlength=E)
    ptr = np.concatenate([[0], np.cumsum(cnt)])
    edges, enet = steiner_big.pd_edges(gp[src, 0], gp[src, 1], ptr, alpha=0.0)
    t_tree = time.time() - t0
    # branches, left end first
    e = edges.astype(np.int64)
    sw = (e[:, 0] > e[:, 2]) | ((e[:, 0] == e[:, 2]) & (e[:, 1] > e[:, 3]))
    e[sw] = e[sw][:, [2, 3, 0, 1]]
    seg = torch.as_tensor(e, device=dev)
    B = len(seg)
    ch = torch.as_tensor(h3[lo0:hi0 + 1].sum(0), dtype=torch.float32)              # [ny][nx - 1]
    cv = torch.as_tensor(v3[lo0:hi0 + 1].sum(0).T.copy(), dtype=torch.float32)     # [nx][ny - 1]
    G = _Grid2D(ch, cv, dev)
    typ = torch.zeros(B, dtype=torch.long, device=dev); mid = seg[:, 2].clone()      # L: horizontal first
    G.usage(_pieces(seg, typ, mid)); G.costs(pres)
    ov0, _ = G.overflow()
    tot = ov0
    it = 0
    for it in range(iterations):
        if tot <= 0:
            break
        G.costs(pres, hist=hist)
        hb = torch.as_tensor(_hash(np.arange(B, dtype=np.int64), it) % batches, device=dev)
        for b in range(batches):
            over = G.path(seg, typ, mid, over=True) > 0
            rip = over & (hb == b)
            if not bool(rip.any()):
                continue
            keep = ~rip
            G.usage(_pieces(seg[keep], typ[keep], mid[keep])); G.costs(pres)
            ri = torch.nonzero(rip).flatten()
            t_, m_ = G.best(seg[ri], margin, via)
            typ[ri], mid[ri] = t_, m_
            G.usage(_pieces(seg, typ, mid)); G.costs(pres)
        pres = min(pres_max, pres * pres_grow)
        tot, mx = G.overflow()
        if verbose and (it < 3 or it % 5 == 4):
            print('  gr: iteration %d overflow %d (max %d)' % (it, tot, mx))
    t_2d = time.time() - t0
    ov2d = tot
    # layer assignment
    P0 = _pieces(seg, typ, mid)                         # (branch, dir, line, a, b)
    # chunks of at most la_chunk gcells, each on its own layer (a run may change layer on the way, as FastRoute's
    # layer assignment does); consecutive chunks of a piece on one layer are merged back into one segment
    plen0 = P0[:, 4] - P0[:, 3]
    nch = (plen0 + la_chunk - 1) // la_chunk
    pid = torch.repeat_interleave(torch.arange(len(P0), device=dev), nch)
    cix = torch.arange(len(pid), device=dev) - torch.repeat_interleave(torch.cumsum(nch, 0) - nch, nch)
    P = P0[pid].clone()
    P[:, 3] = P0[pid, 3] + cix * la_chunk
    P[:, 4] = torch.minimum(P[:, 3] + la_chunk, P0[pid, 4])
    Pl = torch.zeros(len(P), dtype=torch.long, device=dev)
    caps = {}
    use = {}
    for l in range(lo0, hi0 + 1):
        if horiz[l]:
            caps[l] = torch.as_tensor(h3[l], dtype=torch.float32, device=dev)                    # [ny][nx - 1]
        else:
            caps[l] = torch.as_tensor(v3[l].T.copy(), dtype=torch.float32, device=dev)           # [nx][ny - 1]
        use[l] = torch.zeros_like(caps[l])
    lay_d = {0: [l for l in range(lo0, hi0 + 1) if horiz[l]], 1: [l for l in range(lo0, hi0 + 1) if not horiz[l]]}
    def span(prefix, q, Lw):
        return prefix[q[:, 2] * Lw + q[:, 4]] - prefix[q[:, 2] * Lw + q[:, 3]]

    def add(sel, sign):
        """Add (sign 1) or remove (-1) the usage of pieces sel on their layers."""
        q = P[sel]; ls = Pl[sel]
        for l in use:
            m_ = ls == l
            if not bool(m_.any()):
                continue
            qm = q[m_]
            c = caps[l]; Lw = c.shape[1] + 1
            dd = torch.zeros(c.shape[0] * Lw, device=dev)
            one = torch.full((len(qm),), float(sign), device=dev)
            dd.index_add_(0, qm[:, 2] * Lw + qm[:, 3], one); dd.index_add_(0, qm[:, 2] * Lw + qm[:, 4], -one)
            use[l] += dd.view(c.shape[0], Lw).cumsum(1)[:, :-1]

    def assign(sel):
        """Each piece of sel on the lowest layer of its direction with room along all of it, else the one it
        overflows least (+ la_step per layer up), against the present usage; then its usage added."""
        q = P[sel]
        for dirv in (0, 1):
            s2 = torch.nonzero(q[:, 1] == dirv).flatten()
            if not len(s2) or not lay_d[dirv]:
                continue
            qq = q[s2]
            full_l, cost_l = [], []
            for k, l in enumerate(lay_d[dirv]):
                c, u = caps[l], use[l]
                Lw = c.shape[1] + 1
                z = torch.zeros(c.shape[0], 1, device=dev)
                e1 = u + 1 - c
                full_l.append(span(torch.cat([z, (e1 > 0).float().cumsum(1)], 1).view(-1), qq, Lw))
                cost_l.append(span(torch.cat([z, e1.clamp(min=0).cumsum(1)], 1).view(-1), qq, Lw) + la_step * k)
            F = torch.stack(full_l, 1); Cc = torch.stack(cost_l, 1)
            free = F == 0
            pick = torch.where(free.any(1), free.float().argmax(1), Cc.argmin(1))
            Pl[sel[s2]] = torch.as_tensor(lay_d[dirv], device=dev)[pick]
        add(sel, 1)

    def overflowing():
        """Per piece: does it cross an edge of its layer used beyond capacity."""
        out = torch.zeros(len(P), dtype=torch.bool, device=dev)
        for l in use:
            m_ = torch.nonzero(Pl == l).flatten()
            if not len(m_):
                continue
            c = caps[l]; Lw = c.shape[1] + 1
            z = torch.zeros(c.shape[0], 1, device=dev)
            po = torch.cat([z, (use[l] > c).float().cumsum(1)], 1).view(-1)
            out[m_] = span(po, P[m_], Lw) > 0
        return out

    length = plen0[pid]                                 # shortest pieces first (a chunk ranks as its piece)
    order = torch.argsort(length, stable=True)
    nb = max(1, min(la_batches, len(P)))
    for bi in range(nb):
        sel = order[bi * len(P) // nb:(bi + 1) * len(P) // nb]
        if len(sel):
            assign(sel)
    la_rounds = 0
    hp = torch.as_tensor(_hash(np.arange(len(P), dtype=np.int64), 7) % 8, device=dev)
    for la_rounds in range(la_iterations):
        ov = overflowing()
        if not bool(ov.any()):
            break
        for b in range(8):                              # rip up and reassign in hashed batches
            sel = torch.nonzero(ov & (hp == (b + la_rounds) % 8)).flatten()
            if len(sel):
                add(sel, -1)
                assign(sel)
    ov3d = sum(float(torch.ceil((use[l] - caps[l]).clamp(min=0)).sum()) for l in use)
    t_la = time.time() - t0
    # merge each piece's consecutive chunks on one layer
    new = torch.ones(len(P), dtype=torch.bool, device=dev)
    new[1:] = (pid[1:] != pid[:-1]) | (Pl[1:] != Pl[:-1])
    run = torch.cumsum(new.long(), 0) - 1
    first = torch.nonzero(new).flatten()
    R_ = P[first].clone()
    R_[:, 4] = torch.zeros(len(first), dtype=torch.long, device=dev).scatter_reduce(0, run, P[:, 4], 'amax',
                                                                                     include_self=False)
    P, Pl = R_, Pl[first]
    # 3D segments (gcell indices, routing levels), then dbu
    Pn = P.cpu().numpy(); Ln = Pl.cpu().numpy() + 1
    br_net = enet[Pn[:, 0]]
    hz = Pn[:, 1] == 0
    sx0 = np.where(hz, Pn[:, 3], Pn[:, 2]); sy0 = np.where(hz, Pn[:, 2], Pn[:, 3])
    sx1 = np.where(hz, Pn[:, 4], Pn[:, 2]); sy1 = np.where(hz, Pn[:, 2], Pn[:, 4])
    planar = np.stack([sx0, sy0, Ln, sx1, sy1, Ln], 1)
    # via stacks at every point where pieces end or pins are: per (net, gx, gy) the layer span
    ev_net = np.concatenate([br_net, br_net, p_net[ok]])
    ev_x = np.concatenate([sx0, sx1, gp[ok, 0]]); ev_y = np.concatenate([sy0, sy1, gp[ok, 1]])
    ev_l = np.concatenate([Ln, Ln, play[ok]])
    routed = np.zeros(E, bool); routed[br_net] = True
    # a net within one gcell: its pins up to the lowest routing layer
    loc = ok & ~routed[p_net]
    ev_net = np.concatenate([ev_net, p_net[loc]]); ev_x = np.concatenate([ev_x, gp[loc, 0]])
    ev_y = np.concatenate([ev_y, gp[loc, 1]]); ev_l = np.concatenate([ev_l, np.full(int(loc.sum()), lo)])
    k3 = (ev_net * nx + ev_x) * ny + ev_y
    uk3, inv = np.unique(k3, return_inverse=True)
    lmin = np.full(len(uk3), 10 ** 9); lmax = np.full(len(uk3), -1)
    np.minimum.at(lmin, inv, ev_l); np.maximum.at(lmax, inv, ev_l)
    vn = uk3 // (nx * ny); vx = (uk3 // ny) % nx; vy = uk3 % ny
    nv = np.maximum(lmax - lmin, 0)                    # vias in each stack
    rep = np.repeat(np.arange(len(uk3)), nv)
    step = np.arange(len(rep)) - np.repeat(np.cumsum(nv) - nv, nv)
    l0 = lmin[rep] + step
    vias = np.stack([vx[rep], vy[rep], l0, vx[rep], vy[rep], l0 + 1], 1)
    segs = np.concatenate([planar, vias])
    snet = np.concatenate([br_net, vn[rep]])
    o = np.argsort(snet, kind='stable')
    segs, snet = segs[o], snet[o]
    dbu_xy = lambda v, o_: v * tile + tile // 2 + o_  # noqa: E731
    out = segs.copy()
    out[:, 0] = dbu_xy(segs[:, 0], X0); out[:, 3] = dbu_xy(segs[:, 3], X0)
    out[:, 1] = dbu_xy(segs[:, 1], Y0); out[:, 4] = dbu_xy(segs[:, 4], Y0)
    gxy = np.stack([dbu_xy(gp[:, 0], X0), dbu_xy(gp[:, 1], Y0)], 1)
    rt = GlobalRoutes(d, snet, out, pin_pt=pt, pin_grid=gxy, pin_layer=play, min_layer=lo,
                      source='attar.route.global_route')
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)
    stats = {'wirelength_um': rt.wirelength(tile), 'vias': rt.vias(), 'branches': B, 'nets_routed': int(routed.sum()),
             'overflow_initial': ov0, 'overflow_2d': ov2d, 'overflow_3d': ov3d, 'iterations': it, 'la_rounds': la_rounds,
             'seconds': time.time() - t0, 'seconds_capacity': t_cap, 'seconds_tree': t_tree - t_cap,
             'seconds_2d': t_2d - t_tree, 'seconds_layers': t_la - t_2d, 'tile': tile, 'grid': (nx, ny)}
    if verbose:
        print('  gr: %d nets, %d branches, WL %d um, %d vias, overflow 2D %d / 3D %d, %.2fs'
              % (stats['nets_routed'], B, stats['wirelength_um'], stats['vias'], ov2d, ov3d, stats['seconds']))
    return rt, stats
