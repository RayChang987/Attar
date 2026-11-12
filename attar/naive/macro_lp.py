"""Macro legalization by constraint graphs and linear programming.

Every pair of macros must be apart along x or along y. Starting from a (global) placement with overlaps:

1. Directions. Each pair is separated along the axis where it is already further apart (the larger gap, so the
   smaller move), in the order of its centers on that axis. This gives a horizontal and a vertical constraint graph,
   both acyclic (edges follow a strict order of the centers), together covering every pair.
2. Fit. On each axis, the earliest start of every macro is the longest path to it; the graph fits in the region iff
   no macro must start later than its upper bound. While one axis does not fit, an edge of its critical path moves
   to the other axis, where it fits (the lower macro ends before the latest start of the upper one); the edge with
   the largest gap on the other axis goes first.
3. Positions. Each axis is a linear program on the grid: minimize the total distance to the start positions subject
   to x_j >= x_i + w_i (edges), lo <= x_i <= hi - w_i. Its constraint matrix is a network matrix and the data are
   integers (grid units), so the simplex vertex the solver returns is on the grid. Only the edges of pairs that
   actually meet on the other axis go into the LP; edges found violated afterwards are added and the LP solved again.

When no fitting graphs are found (step 2 is a local search; fitting is NP-hard in general), the graphs that overflow
the least are kept and the region grows by what they need: the macros never overlap, but may stick out of the core.

Solver: HiGHS (scipy.optimize.linprog, dual simplex), on the CPU: the LPs have a few unknowns per macro.
"""
import math
import time

import numpy as np


def _longest(adj, order, k, ub):
    """Earliest starts (longest paths, >= 0) and latest starts (<= ub) on a DAG given as a dense bool matrix
    adj[i, j] (i before j, start_j >= start_i + k_i), topological order `order`."""
    n = len(k)
    est = np.zeros(n)
    for j in order:
        p = adj[:, j]
        if p.any():
            est[j] = max(0.0, float((est[p] + k[p]).max()))
    lst = ub.astype(np.float64).copy()
    for i in order[::-1]:
        s = adj[i]
        if s.any():
            lst[i] = min(lst[i], float((lst[s] - k[i]).min()))
    return est, lst


def _critical_edges(adj, est, k, v):
    """Edges (i, j) of a longest path ending at v."""
    path = []
    while True:
        p = np.nonzero(adj[:, v])[0]
        if len(p) == 0:
            return path
        u = int(p[np.argmax(est[p] + k[p])])
        if est[u] + k[u] < est[v] - 1e-9 and est[v] > 0:   # v starts at 0 or at a bound, not pushed by u
            return path
        path.append((u, v))
        v = u


def _solve_axis(x0, k, ub, edges, w):
    """min sum w_i |x_i - x0_i| s.t. x_j - x_i >= k_i for (i, j) in edges, 0 <= x_i <= ub_i. Variables (x, t)."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    n = len(x0)
    E = len(edges)
    rows, cols, vals, rhs = [], [], [], []
    # x_i - x_j <= -k_i
    for r, (i, j) in enumerate(edges):
        rows += [r, r]; cols += [i, j]; vals += [1.0, -1.0]; rhs.append(-float(k[i]))
    # x_i - t_i <= x0_i ; -x_i - t_i <= -x0_i
    for i in range(n):
        rows += [E + 2 * i, E + 2 * i, E + 2 * i + 1, E + 2 * i + 1]
        cols += [i, n + i, i, n + i]
        vals += [1.0, -1.0, -1.0, -1.0]
        rhs += [float(x0[i]), -float(x0[i])]
    A = coo_matrix((vals, (rows, cols)), shape=(E + 2 * n, 2 * n)).tocsr()
    c = np.concatenate([np.zeros(n), w])
    bounds = [(0.0, float(ub[i])) for i in range(n)] + [(0.0, None)] * n
    r = linprog(c, A_ub=A, b_ub=np.array(rhs), bounds=bounds, method='highs-ds')
    if r.status != 0:
        return None
    return np.round(r.x[:n])


def lp_legalize(q0, size_g, lo, hi, halo, stx, sty, q_dir=None, max_flips=None, max_rounds=20):
    """Positions near q0 (n, 2) for boxes size_g (n, 2), at least `halo` apart, on the grid lo + k * step, inside
    [lo, hi] as far as possible. torch tensors in, (torch positions, info) out.
    The result never has overlaps. It fits [lo, hi] (info['fits']) when the search of step 2 finds constraint graphs
    that fit; otherwise the graphs that overflow the least are kept, the region grows up and to the right by just
    what they need, and info['overflow'] says by how much (x, y, in the units of q0).
    q_dir: positions to read the pair directions from (default q0), e.g. the result of another legalizer: when it
    has no overlaps its constraint graphs fit as they are, and the LP only moves the macros back towards q0."""
    import torch
    t0 = time.time()
    dev, dt = q0.device, q0.dtype
    q = q0.detach().double().cpu().numpy()
    sz = size_g.detach().double().cpu().numpy()
    lo_ = lo.detach().double().cpu().numpy(); hi_ = hi.detach().double().cpu().numpy()
    n = len(q)
    step = np.array([stx, sty], dtype=np.float64)
    # grid units relative to lo; k: separation (box + halo), ub: last start
    k = np.ceil((sz + halo) / step - 1e-6)
    ub = np.maximum(np.floor((hi_ - lo_ - sz) / step + 1e-6), 0)    # a macro larger than the region: overflows
    x0 = np.clip((q - lo_) / step, 0, ub)
    qd = q if q_dir is None else q_dir.detach().double().cpu().numpy()
    xd = np.clip((qd - lo_) / step, 0, ub)
    c = xd + k / 2                                                 # centers, grid units: they set the directions
    # gaps along each axis (negative: overlap), in units of the region size so both axes compare
    span = np.maximum(ub.max(0) + k.max(0), 1.0)
    d = np.abs(c[:, None, :] - c[None, :, :]) - (k[:, None, :] + k[None, :, :]) / 2
    g = d / span
    iu = np.triu(np.ones((n, n), bool), 1)
    horiz = iu & (g[..., 0] >= g[..., 1])
    vert = iu & ~horiz
    adj = []
    for a, sel in ((0, horiz), (1, vert)):
        before = (c[:, None, a] < c[None, :, a]) | ((c[:, None, a] == c[None, :, a]) & (np.arange(n)[:, None] < np.arange(n)[None, :]))
        m = np.zeros((n, n), bool)
        m[sel & before] = True
        m[(sel & ~before).T] = True
        adj.append(m)
    order = [np.lexsort((np.arange(n), c[:, a])) for a in (0, 1)]

    def state():
        est, lst = [None, None], [None, None]
        for a in (0, 1):
            est[a], lst[a] = _longest(adj[a], order[a], k[:, a], ub[:, a])
        over = [float(np.maximum(est[a] - ub[:, a], 0).sum()) / span[a] for a in (0, 1)]   # every late macro counts
        return est, lst, over

    # local search over the edge directions: move one edge of a critical path to the other axis, the move that
    # leaves the least total overflow (both axes evaluated exactly); a pair moves at most 3 times
    flips = 0
    moved = {}
    max_flips = max_flips if max_flips is not None else 20 * n

    def _best_move(cands):
        best = None
        for a, i, j in cands:
            if moved.get((min(i, j), max(i, j)), 0) >= 3:
                continue
            b = 1 - a
            lower, upper = (i, j) if (c[i, b], i) < (c[j, b], j) else (j, i)
            adj[a][i, j] = False; adj[b][lower, upper] = True
            e2, l2, o2 = state()
            adj[a][i, j] = True; adj[b][lower, upper] = False
            key = (sum(o2), -d[i, j, b])
            if best is None or key < best[0]:
                best = (key, a, i, j, lower, upper, (e2, l2, o2))
        return best

    est, lst, over = state()
    why = None
    while sum(over) > 0:
        if flips >= max_flips:
            why = 'no fit after %d moves' % flips
            break
        best = None
        for wide in (False, True):
            cands = set()
            for a in (0, 1):
                if over[a] > 0:
                    late = np.nonzero(est[a] > ub[:, a] + 1e-9)[0]
                    ends = late if wide else [int(np.argmax(est[a] - ub[:, a]))]
                    for v in ends:
                        cands |= {(a, i, j) for i, j in _critical_edges(adj[a], est[a], k[:, a], int(v))}
            best = _best_move(cands)
            if best is not None and best[0][0] < sum(over) - 1e-12:
                break
            best = None
        if best is None:
            why = 'no move lowers the overflow'
            break
        _, a, i, j, lower, upper, st = best
        adj[a][i, j] = False; adj[1 - a][lower, upper] = True
        moved[(min(i, j), max(i, j))] = moved.get((min(i, j), max(i, j)), 0) + 1
        est, lst, over = st
        flips += 1
    # best effort: the region grows on each axis to what the longest chain needs (then every macro has room)
    fits = sum(over) == 0 and bool((sz / step <= np.floor((hi_ - lo_) / step + 1e-6) + 1e-6).all())
    if not fits:
        for a in (0, 1):
            ub[:, a] = np.maximum(ub[:, a], (est[a] + k[:, a]).max() - k[:, a])
    # LPs, with lazily added edges: first the pairs that meet on the other axis at the start; then the edge of every
    # pair found overlapping, until none is. All edges together fit (above), so this ends; the last round, if
    # reached, takes them all.
    w = np.ones(n)
    E = [np.argwhere(adj[a]) for a in (0, 1)]
    use = []
    for a in (0, 1):
        lo_b, hi_b = xd[:, 1 - a], xd[:, 1 - a] + k[:, 1 - a]
        e = E[a]
        use.append(np.minimum(hi_b[e[:, 0]], hi_b[e[:, 1]]) > np.maximum(lo_b[e[:, 0]], lo_b[e[:, 1]]))
    out = x0.copy()
    rounds = 0
    for rnd in range(max_rounds + 1):
        if rnd == max_rounds:
            use = [np.ones(len(E[a]), bool) for a in (0, 1)]
        for a in (0, 1):
            x = _solve_axis(x0[:, a], k[:, a], ub[:, a], [tuple(e) for e in E[a][use[a]]], w)
            if x is None:                                   # cannot happen: the graphs fit the (grown) region
                raise RuntimeError('macro LP of axis %s infeasible' % 'xy'[a])
            out[:, a] = x
        rounds += 1
        sep = np.zeros((n, n), bool)
        for a in (0, 1):
            sep |= out[None, :, a] >= out[:, None, a] + k[:, None, a] - 1e-6        # j after i
            sep |= out[:, None, a] >= out[None, :, a] + k[None, :, a] - 1e-6        # i after j
        bad = iu & ~sep
        if not bad.any():
            break
        bad = bad | bad.T
        added = 0
        for a in (0, 1):
            m = bad[E[a][:, 0], E[a][:, 1]] & ~use[a]
            use[a] |= m; added += int(m.sum())
        if not added:                                     # cannot happen: every pair has an edge
            raise RuntimeError('macro LP: overlaps with every edge in the LP')
    pos = lo_ + out * step
    overflow = np.maximum(pos + sz - hi_, 0).max(0)
    info = {'fits': fits, 'why': why, 'overflow': tuple(float(v) for v in overflow), 'flips': flips, 'lp_rounds': rounds, 'lp_edges': int(use[0].sum() + use[1].sum()),
            'displacement': float(np.abs(pos - q).sum(1).mean()), 'seconds': time.time() - t0}
    return torch.as_tensor(pos, device=dev, dtype=dt), info

