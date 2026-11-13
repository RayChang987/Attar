"""attar.steiner.pd_tree as a Triton kernel: one program per net, the whole Prim-Dijkstra + steinerize in one launch per
degree bucket. Same algorithm and tie-breaks as attar.steiner (which stays the reference and the CPU path)."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .steiner import MAX_DEG


@triton.jit
def _pd_body(b, x, y, valid, q_p, scr_p, pscr_p, alpha, D: tl.constexpr, N: tl.constexpr, ROUNDS: tl.constexpr,
             MAXDEG: tl.constexpr):
    """Net b's tree: pins x, y, valid in lanes 0..D-1 (also at q_p, (D, 2) per net). Returns per node (lanes 0..N-1:
    pins, then Steiner points) X, Y, parent P (-1: root or unused), alive A."""
    lane = tl.arange(0, D)
    # rank in (x, y, slot) order among valid points; the representative slot of coincident points
    rank = tl.zeros([D], dtype=tl.int32)
    rep = lane
    reprank = tl.where(valid, D * 4, D * 4)
    for j in range(D):
        xj = tl.sum(tl.where(lane == j, x, 0.0))
        yj = tl.sum(tl.where(lane == j, y, 0.0))
        vj = tl.sum(tl.where(lane == j, valid.to(tl.int32), 0)) != 0
        before = vj & ((xj < x) | ((xj == x) & (yj < y)) | ((xj == x) & (yj == y) & (j < lane)))
        rank += before.to(tl.int32)
    for j in range(D):
        xj = tl.sum(tl.where(lane == j, x, 0.0))
        yj = tl.sum(tl.where(lane == j, y, 0.0))
        vj = tl.sum(tl.where(lane == j, valid.to(tl.int32), 0)) != 0
        rj = tl.sum(tl.where(lane == j, rank, 0))
        same = vj & valid & (xj == x) & (yj == y)
        better = same & (rj < reprank)
        rep = tl.where(better, j, rep)
        reprank = tl.where(better, rj, reprank)
    uniq = valid & (rep == lane)
    # per-pin scratch: rank, unique flag (scalar and gather loads in the loops below)
    pb = b * D * 2
    tl.store(pscr_p + pb + lane, rank)
    tl.store(pscr_p + pb + D + lane, uniq.to(tl.int32))
    tl.debug_barrier()
    # Prim-Dijkstra from slot 0 over nearest neighbors computed on the fly
    INF = float('inf')
    key = tl.where(lane == 0, 0.0, INF)
    pl = tl.zeros([D], dtype=tl.float32)
    par = tl.full([D], -1, tl.int32)
    visited = ~valid
    for _ in range(D):
        cand = ~visited & (key < INF)
        w = tl.where(cand, key, INF)
        m = tl.min(w, 0)
        cand = cand & (w == m)
        parrank = tl.where(par >= 0, tl.load(pscr_p + pb + tl.maximum(par, 0)), -1)
        pr = tl.where(cand, parrank, 1 << 30)
        cand = cand & (pr == tl.min(pr, 0))
        rr = tl.where(cand, rank, 1 << 30)
        mr = tl.min(rr, 0)
        has = mr < (1 << 30)
        j = tl.sum(tl.where(cand & (rank == mr), lane, 0))
        visited = visited | ((lane == j) & has)
        xj = tl.sum(tl.where(lane == j, x, 0.0))
        yj = tl.sum(tl.where(lane == j, y, 0.0))
        uj = tl.sum(tl.where(lane == j, uniq.to(tl.int32), 0)) != 0
        repj = tl.sum(tl.where(lane == j, rep, 0))
        plj = tl.sum(tl.where(lane == j, pl, 0.0))
        # neighbors of j: empty closed box among unique points; duplicates link to their representative
        lox = tl.minimum(x, xj); hix = tl.maximum(x, xj); loy = tl.minimum(y, yj); hiy = tl.maximum(y, yj)
        blocked = lane < 0
        for k in range(D):
            xk = tl.load(q_p + (b * D + k) * 2)
            yk = tl.load(q_p + (b * D + k) * 2 + 1)
            uk = tl.load(pscr_p + pb + D + k) != 0
            blocked = blocked | (uk & (k != j) & (lane != k) & (xk >= lox) & (xk <= hix) & (yk >= loy) & (yk <= hiy))
        nb = uj & uniq & (lane != j) & ~blocked
        nb = nb | (~uj & valid & (lane == repj)) | (valid & ~uniq & (rep == j))
        d = tl.abs(x - xj) + tl.abs(y - yj)
        wj = d + alpha * plj
        upd = has & nb & ~visited & (wj <= key)
        key = tl.where(upd, wj, key)
        pl = tl.where(upd, d + plj, pl)
        par = tl.where(upd, j, par)
    # nodes: pins 0..D-1, Steiner points D..N-1 (kept in this program's scratch: parent, x, y)
    nl = tl.arange(0, N)
    base = b * N * 3
    X = tl.where(nl < D, tl.load(q_p + (b * D + tl.minimum(nl, D - 1)) * 2), 0.0)
    Y = tl.where(nl < D, tl.load(q_p + (b * D + tl.minimum(nl, D - 1)) * 2 + 1), 0.0)
    P = tl.full([N], -1, tl.int32)
    A = nl < 0
    for k in range(D):
        pk = tl.sum(tl.where(lane == k, par, 0))
        vk = tl.sum(tl.where(lane == k, valid.to(tl.int32), 0)) != 0
        P = tl.where(nl == k, pk, P)
        A = A | ((nl == k) & vk)
    nxt = D
    for _ in range(ROUNDS):
        # incident nodes of every node v (lanes): its parent and its children, up to MAXDEG slots
        tl.store(scr_p + base + nl, P)
        tl.store(scr_p + base + N + nl, X.to(tl.int32))
        tl.store(scr_p + base + 2 * N + nl, Y.to(tl.int32))
        tl.debug_barrier()
        cnt = tl.zeros([N], dtype=tl.int32)
        s0 = tl.full([N], -1, tl.int32); s1 = tl.full([N], -1, tl.int32); s2 = tl.full([N], -1, tl.int32)
        s3 = tl.full([N], -1, tl.int32); s4 = tl.full([N], -1, tl.int32); s5 = tl.full([N], -1, tl.int32)
        s6 = tl.full([N], -1, tl.int32); s7 = tl.full([N], -1, tl.int32)
        hp = (P >= 0) & A
        s0 = tl.where(hp, P, s0)
        cnt = hp.to(tl.int32)
        for c in range(N):
            pc = tl.load(scr_p + base + c)
            ac = (c < D) | (c < nxt)
            isc = A & (pc == nl) & ac & (pc >= 0)
            s0 = tl.where(isc & (cnt == 0), c, s0); s1 = tl.where(isc & (cnt == 1), c, s1)
            s2 = tl.where(isc & (cnt == 2), c, s2); s3 = tl.where(isc & (cnt == 3), c, s3)
            s4 = tl.where(isc & (cnt == 4), c, s4); s5 = tl.where(isc & (cnt == 5), c, s5)
            s6 = tl.where(isc & (cnt == 6), c, s6); s7 = tl.where(isc & (cnt == 7), c, s7)
            cnt += isc.to(tl.int32)
        # best pair per node
        bg = tl.full([N], -1.0, tl.float32)
        bx = X; by = Y
        b1 = tl.full([N], -1, tl.int32); b2 = tl.full([N], -1, tl.int32)
        for i1 in tl.static_range(MAXDEG):
            o1 = tl.where(i1 == 0, s0, tl.where(i1 == 1, s1, tl.where(i1 == 2, s2, tl.where(i1 == 3, s3, tl.where(i1 == 4, s4, tl.where(i1 == 5, s5, tl.where(i1 == 6, s6, s7)))))))
            x1 = tl.load(scr_p + base + N + tl.maximum(o1, 0)).to(tl.float32)
            y1 = tl.load(scr_p + base + 2 * N + tl.maximum(o1, 0)).to(tl.float32)
            for i2 in tl.static_range(i1 + 1, MAXDEG):
                o2 = tl.where(i2 == 0, s0, tl.where(i2 == 1, s1, tl.where(i2 == 2, s2, tl.where(i2 == 3, s3, tl.where(i2 == 4, s4, tl.where(i2 == 5, s5, tl.where(i2 == 6, s6, s7)))))))
                x2 = tl.load(scr_p + base + N + tl.maximum(o2, 0)).to(tl.float32)
                y2 = tl.load(scr_p + base + 2 * N + tl.maximum(o2, 0)).to(tl.float32)
                mnx = tl.minimum(x1, x2); mxx = tl.maximum(x1, x2); mny = tl.minimum(y1, y2); mxy = tl.maximum(y1, y2)
                sx = tl.where(mnx > X, mnx, tl.where(mxx < X, mxx, X))
                sy = tl.where(mny > Y, mny, tl.where(mxy < Y, mxy, Y))
                g = tl.abs(sx - X) + tl.abs(sy - Y)
                ok = (o1 >= 0) & (o2 >= 0) & A
                better = ok & (g > bg)
                bg = tl.where(better, g, bg); bx = tl.where(better, sx, bx); by = tl.where(better, sy, by)
                b1 = tl.where(better, o1, b1); b2 = tl.where(better, o2, b2)
        # the heap's best: gain, then x, then y (largest first)
        cand = bg > 0
        mg = tl.max(tl.where(cand, bg, -1.0), 0)
        cand = cand & (bg == mg)
        mx_ = tl.max(tl.where(cand, bx, -3.0e38), 0)
        cand = cand & (bx == mx_)
        my_ = tl.max(tl.where(cand, by, -3.0e38), 0)
        cand = cand & (by == my_)
        v = tl.min(tl.where(cand, nl, N), 0)
        act = (mg > 0) & (v < N) & (nxt < N)
        o1 = tl.sum(tl.where(nl == v, b1, 0)); o2 = tl.sum(tl.where(nl == v, b2, 0))
        sx = tl.sum(tl.where(nl == v, bx, 0.0)); sy = tl.sum(tl.where(nl == v, by, 0.0))
        pv = tl.sum(tl.where(nl == v, P, 0))
        x1 = tl.sum(tl.where(nl == o1, X, 0.0)); y1 = tl.sum(tl.where(nl == o1, Y, 0.0))
        x2 = tl.sum(tl.where(nl == o2, X, 0.0)); y2 = tl.sum(tl.where(nl == o2, Y, 0.0))
        is1 = (x1 == sx) & (y1 == sy)
        is2 = (x2 == sx) & (y2 == sy) & ~is1
        new = act & ~is1 & ~is2
        sid = tl.where(new, nxt, tl.where(is1, o1, o2))
        p1 = pv == o1
        p2 = pv == o2
        both = new & ~p1 & ~p2
        # rewire (attar.steiner.pd_tree, parent pointers)
        Pn = P
        Pn = tl.where((nl == o1) & (both | (new & p2)), sid, Pn)
        Pn = tl.where((nl == o2) & (both | (new & p1)), sid, Pn)
        Pn = tl.where((nl == sid) & both, v, Pn)
        Pn = tl.where((nl == sid) & new & (p1 | p2), tl.where(p1, o1, o2), Pn)
        Pn = tl.where((nl == v) & new & (p1 | p2), sid, Pn)
        r1 = act & is1
        r2 = act & is2
        Pn = tl.where((nl == o2) & r1 & ~p2, o1, Pn)
        Pn = tl.where((nl == o1) & r1 & p2, o2, Pn)
        Pn = tl.where((nl == v) & r1 & p2, o1, Pn)
        Pn = tl.where((nl == o1) & r2 & ~p1, o2, Pn)
        Pn = tl.where((nl == o2) & r2 & p1, o1, Pn)
        Pn = tl.where((nl == v) & r2 & p1, o2, Pn)
        P = Pn
        X = tl.where((nl == sid) & new, sx, X)
        Y = tl.where((nl == sid) & new, sy, Y)
        A = A | ((nl == sid) & new)
        nxt = nxt + new.to(tl.int32)
        tl.debug_barrier()
    return X, Y, tl.where(A, P, -1), A


@triton.jit
def _pd_kernel(q_p, valid_p, xy_o, par_o, alive_o, scr_p, pscr_p, alpha, D: tl.constexpr, N: tl.constexpr,
               ROUNDS: tl.constexpr, MAXDEG: tl.constexpr):
    b = tl.program_id(0)
    lane = tl.arange(0, D)
    x = tl.load(q_p + (b * D + lane) * 2)
    y = tl.load(q_p + (b * D + lane) * 2 + 1)
    valid = tl.load(valid_p + b * D + lane) != 0
    X, Y, P, A = _pd_body(b, x, y, valid, q_p, scr_p, pscr_p, alpha, D, N, ROUNDS, MAXDEG)
    nl = tl.arange(0, N)
    tl.store(xy_o + (b * N + nl) * 2, X)
    tl.store(xy_o + (b * N + nl) * 2 + 1, Y)
    tl.store(par_o + b * N + nl, P)
    tl.store(alive_o + b * N + nl, A.to(tl.int8))


@triton.jit
def _gather(f_p, P, V: tl.constexpr, nl, vv, other):
    """(N, V) values at each node's parent (other at the root) from scratch f_p (N, V)."""
    return tl.load(f_p + tl.maximum(P, 0)[:, None] * V + vv[None, :], mask=(P >= 0)[:, None], other=other,
                   volatile=True)


@triton.jit
def _path(T, P, f_p, i_p, nl, vv, V: tl.constexpr, LOGN: tl.constexpr):
    """Root-to-node sums of T (N, V) along parent pointers P: pointer doubling, LOGN rounds."""
    anc = P
    for _ in tl.static_range(LOGN):
        tl.store(f_p + nl[:, None] * V + vv[None, :], T)
        tl.store(i_p + nl, anc)
        tl.debug_barrier()
        T = T + _gather(f_p, anc, V, nl, vv, 0.0)
        anc = tl.where(anc >= 0, tl.load(i_p + tl.maximum(anc, 0), volatile=True), -1)
        tl.debug_barrier()
    return T


@triton.jit
def _subtree(T, P, depth, maxd, f_p, nl, vv, V: tl.constexpr):
    """Subtree sums of T (N, V): deepest level first, each node adds its sum into its parent's."""
    off = nl[:, None] * V + vv[None, :]
    tl.store(f_p + off, T)
    tl.debug_barrier()
    up = tl.maximum(P, 0)[:, None] * V + vv[None, :]
    for k in range(maxd):
        m = ((depth == maxd - k) & (P >= 0))[:, None] & (vv[None, :] >= 0)
        sv = tl.load(f_p + off, mask=m, other=0.0, volatile=True)
        tl.atomic_add(f_p + up, sv, mask=m)
        tl.debug_barrier()
    return tl.load(f_p + off, volatile=True)


@triton.jit
def _rc_kernel(p_p, idx_p, real_p, cap_p, nets_p, elm_o, pi_o, len_o, q_p, scr_p, pscr_p, f_p, i_p, rv_p,
               alpha, dbu, r, c, D: tl.constexpr, N: tl.constexpr, ROUNDS: tl.constexpr,
               MAXDEG: tl.constexpr, V: tl.constexpr, LOGN: tl.constexpr):
    """attar.sta.tree_elmore for one net per program: pin locations (dbu, rounded), the PD tree, its RC moments with
    every pin behind its via resistance rv (the driver's in series with the whole tree), the driving point pi model (O'Brien-Savarino); writes the pins' Elmore delays (P, V), the net's
    pi (E, V, 3) and tree length (E,)."""
    b = tl.program_id(0)
    lane = tl.arange(0, D)
    pin = tl.load(idx_p + b * D + lane)
    valid = tl.load(real_p + b * D + lane) != 0
    x = libdevice.rint(tl.load(p_p + pin * 2) * dbu)
    y = libdevice.rint(tl.load(p_p + pin * 2 + 1) * dbu)
    tl.store(q_p + (b * D + lane) * 2, x)
    tl.store(q_p + (b * D + lane) * 2 + 1, y)
    tl.debug_barrier()
    X, Y, P, A = _pd_body(b, x, y, valid, q_p, scr_p, pscr_p, alpha, D, N, ROUNDS, MAXDEG)
    nl = tl.arange(0, N)
    vv = tl.arange(0, V)
    fb = f_p + b * N * V
    ib = i_p + b * N
    # branch lengths (um): each node's wire to its parent
    tl.store(fb + nl, X)
    tl.store(fb + N + nl, Y)
    tl.debug_barrier()
    px = tl.load(fb + tl.maximum(P, 0), volatile=True)
    py = tl.load(fb + N + tl.maximum(P, 0), volatile=True)
    el = tl.where(P >= 0, (tl.abs(X - px) + tl.abs(Y - py)) / dbu, 0.0)
    tl.store(len_o + tl.load(nets_p + b), tl.sum(el, 0))
    tl.debug_barrier()
    # depth (edges to the root), for the subtree sums
    one = tl.where(P >= 0, 1.0, 0.0)[:, None] + tl.zeros([N, V], tl.float32)
    depth = tl.max(_path(one, P, fb, ib, nl, vv, V, LOGN), 1).to(tl.int32)
    maxd = tl.max(depth, 0)
    # node capacitance: half of every incident wire; pins (lanes < D) carry their pin capacitance behind rcut
    half = c * el / 2
    tl.store(fb + nl, half * 0.0)
    tl.debug_barrier()
    tl.atomic_add(fb + tl.maximum(P, 0), half, mask=P >= 0)
    tl.debug_barrier()
    cw = (half + tl.load(fb + nl, volatile=True))[:, None]
    tl.debug_barrier()
    isp = (nl < D) & A
    pinn = tl.load(idx_p + b * D + tl.minimum(nl, D - 1), mask=isp, other=0)
    cp = tl.load(cap_p + pinn[:, None] * V + vv[None, :], mask=isp[:, None] & (vv[None, :] >= 0), other=0.0)
    cn = cw + cp
    rv = tl.load(rv_p + pinn, mask=isp, other=0.0)[:, None]          # pins' via resistance; the driver: slot 0
    rd = tl.sum(tl.where(nl == 0, tl.load(rv_p + pinn, mask=isp, other=0.0), 0.0), 0)
    Re = (r * el)[:, None]
    y1 = tl.sum(cn, 0)
    m1 = rd * y1[None, :] + _path(Re * _subtree(cn, P, depth, maxd, fb, nl, vv, V), P, fb, ib, nl, vv, V, LOGN)
    m1p = m1 + rv * cp
    qq = cw * m1 + cp * m1p
    m2 = rd * tl.sum(qq, 0)[None, :] + _path(Re * _subtree(qq, P, depth, maxd, fb, nl, vv, V), P, fb, ib, nl, vv,
                                              V, LOGN)
    m2p = m2 + rv * cp * m1p
    y2 = -tl.sum(qq, 0)
    y3 = tl.sum(cw * m2 + cp * m2p, 0)
    ok = (y3 > 1e-12) & (y2 < -1e-12)
    c2 = tl.where(ok, y2 * y2 / tl.where(ok, y3, 1.0), 0.0)
    rr = tl.where(ok, -y3 * y3 / tl.where(ok, y2 * y2 * y2, -1.0), 0.0)
    c2 = tl.minimum(c2, y1)
    net = tl.load(nets_p + b)
    pb = pi_o + net * V * 3 + vv * 3
    tl.store(pb, y1 - c2)
    tl.store(pb + 1, rr)
    tl.store(pb + 2, c2)
    tl.store(elm_o + pinn[:, None] * V + vv[None, :], m1p, mask=isp[:, None] & (vv[None, :] >= 0))


def pd_tree(q, valid, alpha=0.3, rounds=None, max_deg=MAX_DEG):
    """attar.steiner.pd_tree on the GPU. Returns (xy, parent, alive) like it."""
    n, D, _ = q.shape
    D2 = max(2, triton.next_power_of_2(D))
    dev = q.device
    if D2 != D:                                    # pad to a power of two with invalid points at the driver
        q = torch.cat([q, q[:, :1].expand(n, D2 - D, 2)], 1)
        valid = torch.cat([valid, torch.zeros(n, D2 - D, dtype=torch.bool, device=dev)], 1)
    N = 2 * D2
    xy = torch.empty(n, N, 2, device=dev); par = torch.empty(n, N, dtype=torch.int32, device=dev)
    alive = torch.empty(n, N, dtype=torch.int8, device=dev)
    scr = torch.empty(n * N * 3, dtype=torch.int32, device=dev)
    pscr = torch.empty(n * D2 * 2, dtype=torch.int32, device=dev)
    R = rounds if rounds is not None else max(1, D // 2)
    if n:
        _pd_kernel[(n,)](q.contiguous().float(), valid.to(torch.int8).contiguous(), xy, par, alive, scr, pscr, alpha,
                         D=D2, N=N, ROUNDS=R, MAXDEG=max_deg)
    if D2 != D:                                    # back to the caller's layout: pins 0..D-1, Steiner after
        keep = torch.cat([torch.arange(D, device=dev), torch.arange(D2, N, device=dev)])
        remap = torch.full((N,), -1, dtype=torch.long, device=dev); remap[keep] = torch.arange(len(keep), device=dev)
        p = par.long()
        p = torch.where(p >= 0, remap[p.clamp(min=0)], p)
        xy, par, alive = xy[:, keep], p[:, keep], alive[:, keep]
        Nn = 2 * D
        xy, par, alive = xy[:, :Nn], par[:, :Nn], alive[:, :Nn]
    return xy, par.long(), alive.bool()


def rc_bucket(p, idx, real, cap, nets, elm, pi, tlen, alpha, dbu, r, c, rvia, rounds=None, max_deg=MAX_DEG,
              num_warps=None):
    """One degree bucket of attar.sta.tree_elmore in one launch: idx (n, D) pin ids (D a power of two), real (n, D),
    rvia (P,) the pins' via resistances; writes elm (P, V), pi (E, V, 3), tlen (E,) in place."""
    n, D = idx.shape
    assert D == triton.next_power_of_2(D) and D >= 2
    N = 2 * D
    V = cap.shape[1]
    dev = p.device
    q = torch.empty(n * D * 2, device=dev)
    scr = torch.empty(n * N * 3, dtype=torch.int32, device=dev)
    pscr = torch.empty(n * D * 2, dtype=torch.int32, device=dev)
    f = torch.empty(n * N * max(V, 2), device=dev)
    i = torch.empty(n * N, dtype=torch.int32, device=dev)
    R = rounds if rounds is not None else max(1, D // 2)
    _rc_kernel[(n,)](p, idx, real, cap, nets, elm, pi, tlen, q, scr, pscr, f, i, rvia, alpha, float(dbu), float(r),
                     float(c), D=D, N=N, ROUNDS=R, MAXDEG=max_deg, V=V, LOGN=(N - 1).bit_length(),
                     num_warps=num_warps or max(1, min(4, N * V // 128)))
