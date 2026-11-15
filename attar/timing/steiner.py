"""OpenROAD's estimate-parasitics Steiner tree (stt::SteinerTreeBuilder::makeSteinerTree, alpha 0.3 ->
pdr::primDijkstra, OpenROAD src/stt/src/pdr/src/pd.cpp and neighbors.cpp), batched over nets of one degree bucket.

  1. nearest neighbors: two points are neighbors when the closed box spanned by them holds no other point
     (getNearestNeighbors' sweep); coincident points link to the first one at their location;
  2. Prim-Dijkstra over those edges from the driver: weight = edge length + alpha x the parent's path length, heap
     order (weight, parent id, node id), a node's entry replaced when a new weight is <= (buildSpanningTree);
  3. steinerize: repeatedly the node and pair of its edges whose corner point saves the most wire, a new Steiner
     point there (or the edge's other end if it is that point), edges rewired (steinerize, best_steiner_for_node);
  4. degree-4 nodes are not split (splitDegree4Nodes adds zero-length edges: electrically nothing).
Node ids follow OpenROAD's input order: pins sorted by (x, y) (EstimateParasitics::makeSteinerTree). Coordinates in
integer dbu like OpenROAD's, so distance ties break the same way. The tree is rooted at the driver: parent pointers.
"""
import torch

BIG = 1 << 40


# Steinerize rounds and incident edges per node: measured on ASAP7 designs, the PD trees' nodes have at most 6 edges
# (the nearest-neighbor graph limits stars) and steinerizing D/2 rounds gives the full result's length within 1e-5.
MAX_DEG = 8


def pd_tree(q, valid, alpha=0.3, rounds=None, max_deg=MAX_DEG, chunk=8_000_000):
    """q: (n, D, 2) pin locations in dbu (float, integral values), slot 0 the driver; valid: (n, D) real pins.
    Returns (xy (n, N, 2), parent (n, N) with -1 at the root and at unused nodes, alive (n, N)), N = 2D: pins in
    slots 0..D-1 (as given), Steiner points after them."""
    n, D, _ = q.shape
    dev = q.device
    N = 2 * D
    k = torch.arange(D, device=dev)
    # node ids: rank in (x, y) order (OpenROAD sorts the pins by location; equal locations by input order here)
    BIGF = 1e18
    key = torch.where(valid, q[..., 0] * 4e9 + q[..., 1], torch.full_like(q[..., 0], BIGF))
    order = torch.argsort(key * D + k[None].float(), dim=1)               # ties by slot
    rank = torch.empty_like(order).scatter_(1, order, k[None].expand(n, D).contiguous())
    # coincident points: the first one (lowest rank) at a location represents the others
    same = (q[:, :, None, :] == q[:, None, :, :]).all(-1) & valid[:, :, None] & valid[:, None, :]
    # the slot (column) with the lowest rank among coincident points
    rep_slot = torch.where(same, rank[:, None, :], torch.full_like(rank[:, None, :].expand(n, D, D), D)).argmin(2)
    uniq = valid & (rep_slot == k[None])
    # 1. neighbors among unique points: empty closed box
    lo = torch.minimum(q[:, :, None, :], q[:, None, :, :]); hi = torch.maximum(q[:, :, None, :], q[:, None, :, :])
    nb = torch.zeros(n, D, D, dtype=torch.bool, device=dev)
    ib = max(1, min(D, chunk // max(1, D * D)))                            # rows i per block
    step = max(1, chunk // max(1, ib * D * D))
    for c0 in range(0, n, step):
        sl = slice(c0, c0 + step)
        pk = q[sl][:, None, None, :, :]                                   # (m, 1, 1, D, 2)
        for i0 in range(0, D, ib):
            ii = slice(i0, i0 + ib)
            inside = ((pk >= lo[sl, ii][:, :, :, None, :]) & (pk <= hi[sl, ii][:, :, :, None, :])).all(-1)   # (m, ib, D, D)
            inside = inside & uniq[sl][:, None, None, :]
            inside = inside & (k[None, None, None, :] != k[ii][None, :, None, None]) & (k[None, None, None, :] != k[None, None, :, None])
            nb[sl, ii] = ~inside.any(-1)
    nb = nb & uniq[:, :, None] & uniq[:, None, :] & (k[None, :, None] != k[None, None, :])
    dup = valid & ~uniq                                                   # duplicate -> its representative
    nb = nb | (dup[:, :, None] & (rep_slot[:, :, None] == k[None, None, :]))
    nb = nb | nb.transpose(1, 2)
    dist = (q[:, :, None, :] - q[:, None, :, :]).abs().sum(-1)
    # 2. Prim-Dijkstra from the driver (slot 0)
    INF = float('inf')
    rows = torch.arange(n, device=dev)
    keyw = torch.full((n, D), INF, device=dev); keyw[:, 0] = 0.0
    pl = torch.zeros(n, D, device=dev)
    par = torch.full((n, D), -1, dtype=torch.long, device=dev)
    visited = ~valid
    for _ in range(D):
        cand = ~visited & torch.isfinite(keyw)
        w = torch.where(cand, keyw, torch.full_like(keyw, INF))
        m = w.min(1, keepdim=True).values
        cand = cand & (w == m)
        prank = torch.where(par >= 0, torch.gather(rank, 1, par.clamp(min=0)), torch.full_like(par, -1))
        pr = torch.where(cand, prank, torch.full_like(prank, BIG))
        cand = cand & (pr == pr.min(1, keepdim=True).values)
        j = torch.where(cand, rank, torch.full_like(rank, BIG)).argmin(1)
        has = cand.any(1)
        visited = visited | (torch.nn.functional.one_hot(j, D).bool() & has[:, None])
        wj = dist[rows, j, :] + alpha * pl[rows, j][:, None]
        upd = has[:, None] & nb[rows, j, :] & ~visited & (wj <= keyw)
        keyw = torch.where(upd, wj, keyw)
        pl = torch.where(upd, dist[rows, j, :] + pl[rows, j][:, None], pl)
        par = torch.where(upd, j[:, None].expand(n, D), par)
    # tree on N nodes: pins then Steiner points
    xy = torch.cat([q, torch.zeros(n, N - D, 2, device=dev)], 1)
    parent = torch.cat([par, torch.full((n, N - D), -1, dtype=torch.long, device=dev)], 1)
    alive = torch.cat([valid, torch.zeros(n, N - D, dtype=torch.bool, device=dev)], 1)
    nxt = torch.full((n,), D, dtype=torch.long, device=dev)               # next free Steiner slot
    # 3. steinerize
    K = torch.arange(N, device=dev)
    for _ in range(rounds if rounds is not None else max(1, D // 2)):
        # incident nodes of every node: its parent and its children, up to max_deg
        child = (parent[:, None, :] == K[None, :, None]) & alive[:, None, :]          # (n, v, c)
        inc = child | (torch.nn.functional.one_hot(parent.clamp(min=0), N).bool() & (parent >= 0)[:, :, None])
        inc = inc & alive[:, :, None]
        Dg = min(max_deg, N)
        opp = torch.where(inc, K[None, None, :].expand(n, N, N), torch.full_like(parent[:, :, None].expand(n, N, N), N)).sort(2).values
        opp = opp[:, :, :Dg]                                                # (n, v, Dg) incident node ids (N: none)
        okp = opp < N
        oxy = torch.gather(xy, 1, opp.clamp(max=N - 1).reshape(n, -1)[:, :, None].expand(-1, -1, 2)).reshape(n, N, Dg, 2)
        v = xy[:, :, None, None, :]
        a, b = oxy[:, :, :, None, :], oxy[:, :, None, :, :]
        mn, mx = torch.minimum(a, b), torch.maximum(a, b)
        sp = torch.where(mn > v, mn, torch.where(mx < v, mx, v.expand_as(mn)))       # best_steiner_for_node
        gain = (sp - v).abs().sum(-1)                                                  # (n, v, Dg, Dg)
        pair_ok = okp[:, :, :, None] & okp[:, :, None, :] & torch.triu(torch.ones(Dg, Dg, dtype=torch.bool, device=dev), 1)
        gain = torch.where(pair_ok & alive[:, :, None, None], gain, torch.full_like(gain, -1))
        g = gain.reshape(n, N, -1)
        bi = g.argmax(2)                                                                # first best pair per node
        bg = torch.gather(g, 2, bi[:, :, None])[:, :, 0]
        bsp = torch.gather(sp.reshape(n, N, -1, 2), 2, bi[:, :, None, None].expand(-1, -1, 1, 2))[:, :, 0]
        # heap order: gain, then the Steiner point's x, then y (largest first)
        sc = bg.double() * 1e30 + bsp[..., 0].double() * 1e15 + bsp[..., 1].double()
        sc = torch.where(bg > 0, sc, torch.full_like(sc, -1e300))
        vn = sc.argmax(1)
        act = (bg[rows, vn] > 0) & (nxt < N)
        pi_ = bi[rows, vn]
        o1 = opp[rows, vn, pi_ // Dg].clamp(max=N - 1); o2 = opp[rows, vn, pi_ % Dg].clamp(max=N - 1)
        s = bsp[rows, vn]
        is1 = (xy[rows, o1] == s).all(1); is2 = (xy[rows, o2] == s).all(1) & ~is1
        new = act & ~is1 & ~is2
        sid = torch.where(new, nxt.clamp(max=N - 1), torch.where(is1, o1, o2))
        p1 = parent[rows, vn] == o1                                    # edge1 is v's parent edge
        p2 = parent[rows, vn] == o2
        P = parent.clone()
        def setp(node, val, mask):
            cur = P[rows, node]
            P[rows, node] = torch.where(mask, val, cur)
        # new Steiner point s: both children -> s child of v; one parent edge -> s between v and that parent
        both_child = new & ~p1 & ~p2
        setp(o1, sid, both_child | (new & p2)); setp(o2, sid, both_child | (new & p1))
        setp(sid, vn, both_child)
        setp(sid, torch.where(p1, o1, o2), new & (p1 | p2))
        setp(vn, sid, new & (p1 | p2))
        # s is o1 (or o2): the other edge moves from v to it
        r1 = act & is1; r2 = act & is2
        setp(o2, o1, r1 & ~p2)                                         # o2 a child of v: now a child of o1
        setp(o1, o2, r1 & p2); setp(vn, o1, r1 & p2)                   # o2 v's parent: o2 - o1 - v
        setp(o1, o2, r2 & ~p1)
        setp(o2, o1, r2 & p1); setp(vn, o2, r2 & p1)
        parent = P
        xy = xy.clone()
        xy[rows, sid] = torch.where(new[:, None], s, xy[rows, sid])
        alive = alive.clone(); alive[rows, sid] = alive[rows, sid] | new
        nxt = nxt + new.long()
    return xy, parent, alive
