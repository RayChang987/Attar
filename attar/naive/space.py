"""Free-spot search on the site grid of the base rows (int32 occupancy prefix sums), used by the legalizer to
place cells that did not fit (fill) and to pull cells back towards their GP positions (improve). Every accepted
move is overlap-free, so legality is kept."""
import torch


class _Space:
    """The site grid of the base rows in global sites, for free-spot searches (int32 throughout: bigblue4 is
    2.7k rows x 32k sites)."""

    def __init__(self, ys, rx0, nsites, roff_rows, sw, hb, fixed_blocks, dev):
        self.ys, self.R, self.sw, self.hb, self.dev = ys, len(ys), sw, hb, dev
        self.gx0 = rx0.min()
        self.S = int((roff_rows + nsites.double()).max().item()) + 2
        fr, fa, fb = fixed_blocks
        self.fr = fr
        self.fa = (fa + roff_rows[fr]).long(); self.fb = (fb + roff_rows[fr]).long()
        self.ext_lo = roff_rows.long(); self.ext_hi = (roff_rows + nsites.double()).long()

    def replicas(self, x, r0, k, w):
        dev = self.dev
        rep = torch.repeat_interleave(torch.arange(len(r0), device=dev), k)
        rows = r0[rep] + (torch.arange(len(rep), device=dev) - torch.repeat_interleave(torch.cumsum(k, 0) - k, k))
        return rep, rows, x[rep], x[rep] + w[rep]

    def prefix(self, x, r, k, w):
        """Prefix sums (R, S+1) of 'site taken': out-of-row sites, fixed objects, the given cells."""
        R, S, dev = self.R, self.S, self.dev
        _, rr, a, b = self.replicas(x, r, k, w)
        rows = torch.cat([self.fr, rr, torch.arange(R, device=dev), torch.arange(R, device=dev)])
        lo = torch.cat([self.fa, a, torch.zeros(R, dtype=torch.long, device=dev), self.ext_hi]).clamp(0, S)
        hi = torch.cat([self.fb, b, self.ext_lo, torch.full((R,), S, dtype=torch.long, device=dev)]).clamp(0, S)
        D = torch.zeros(R * (S + 1), dtype=torch.int32, device=dev)
        one = torch.ones(len(rows), dtype=torch.int32, device=dev)
        D.index_add_(0, rows * (S + 1) + lo, one)
        D.index_add_(0, rows * (S + 1) + hi, -one)
        D = D.view(R, S + 1).cumsum(1, dtype=torch.int32)
        taken = (D[:, :S] > 0).to(torch.int32)
        del D
        P = torch.zeros(R, S + 1, dtype=torch.int32, device=dev)
        P[:, 1:] = taken.cumsum(1, dtype=torch.int32)
        return P

    def best_free(self, P, gx, gy, rg, k, w, allowed, cur_x, cur_r, dr, dx, budget=16_000_000):
        """For each cell: the free spot (rows rg +- dr allowed by its rails, sites round(gx) +- dx) closest to (gx, gy).
        cur_x / cur_r: the cell's own current footprint, not counted as taken (None for cells not placed).
        Returns (cost, x, r); cost inf if nothing is free in the window."""
        R, S, dev, ys, sw = self.R, self.S, self.dev, self.ys, self.sw
        n = len(gx)
        rd = torch.arange(-dr, dr + 1, device=dev); xd = torch.arange(-dx, dx + 1, device=dev)
        bc = torch.full((n,), float('inf'), device=dev, dtype=torch.float64)
        bx = torch.zeros(n, dtype=torch.long, device=dev); br = torch.zeros(n, dtype=torch.long, device=dev)
        # memory stays within `budget` elements however wide the window: cells in chunks, and when one cell's window
        # alone is too large, its rows in blocks (the best spot so far is kept across blocks)
        chunk = max(1, budget // (len(rd) * len(xd)))
        rblock = len(rd) if chunk > 1 else max(1, budget // len(xd))
        kmax = int(k.max().item()) if n else 0                           # one sync per call, not per chunk and block
        for c0 in range(0, n, chunk):
            c = slice(c0, min(n, c0 + chunk))
            m = c.stop - c.start
            for b0 in range(0, len(rd), rblock):
                rdb = rd[b0:b0 + rblock]
                r = (rg[c][:, None, None] + rdb[None, :, None]).expand(-1, -1, len(xd))
                x = (gx[c].round().long()[:, None, None] + xd[None, None, :]).expand(-1, len(rdb), -1)
                kk = k[c][:, None, None]; ww = w[c][:, None, None]
                ok = (r >= 0) & (r + kk <= R) & (x >= 0) & (x + ww <= S)
                rc = r.clamp(0, R - 1)
                ok &= allowed[0][allowed[1][c][:, None, None], rc]            # (group table, group of each cell)
                xc = x.clamp(0, S - 1); xe = (x + ww).clamp(0, S)
                busy = torch.zeros(x.shape, dtype=torch.int32, device=dev)
                for j in range(kmax):
                    rj = (rc + j).clamp(max=R - 1)
                    use = j < kk
                    t = P[rj, xe] - P[rj, xc]
                    if cur_x is not None:
                        same = (rj >= cur_r[c][:, None, None]) & (rj < (cur_r[c] + k[c])[:, None, None])
                        own = (torch.minimum(xe, (cur_x[c] + w[c])[:, None, None]) - torch.maximum(xc, cur_x[c][:, None, None])).clamp(min=0)
                        t = t - torch.where(same, own, torch.zeros_like(own)).to(torch.int32)
                    busy += torch.where(use, t, torch.zeros_like(t))
                ok &= busy == 0
                cost = (x.double() - gx[c][:, None, None]).abs() * sw + (ys[rc] - gy[c][:, None, None]).abs().double()
                cost = torch.where(ok, cost, torch.full_like(cost, float('inf'))).view(m, -1)
                bi = cost.argmin(1)
                cb = cost.gather(1, bi[:, None]).squeeze(1)
                better = cb < bc[c]
                bc[c] = torch.where(better, cb, bc[c])
                bx[c] = torch.where(better, x.reshape(m, -1).gather(1, bi[:, None]).squeeze(1), bx[c])
                br[c] = torch.where(better, rc.reshape(m, -1).gather(1, bi[:, None]).squeeze(1), br[c])
        return bc, bx, br

    def free_runs(self, P):
        """The maximal runs of free sites in every row: keys row * (S + 1) + start (sorted), starts, ends (excl.)."""
        S = self.S
        free = (P[:, 1:] - P[:, :-1]) == 0                                  # (R, S)
        edge = torch.zeros(self.R, S + 2, dtype=torch.int8, device=self.dev)
        edge[:, 1:S + 1] = free.to(torch.int8)
        dd = edge[:, 1:] - edge[:, :-1]                                     # +1 at a run's start, -1 after its end
        sr, sa = torch.nonzero(dd == 1, as_tuple=True)
        _, sb = torch.nonzero(dd == -1, as_tuple=True)                      # same row-major order: one end per start
        return sr * (S + 1) + sa, sr, sa, sb

    def best_free_1row(self, runs, gx, gy, w, allowed, budget=4_000_000):
        """best_free for single-row cells without a window: the nearest free spot over all rows the rails allow, from
        the free runs of every row (two candidates per row: the last run long enough for the cell starting at or left
        of round(gx), and the next one; a run nearer than those is impossible). Same cost as best_free; returns
        (cost, x, r), cost inf if nothing is free. All widths at once: the runs are listed once per distinct width
        they can hold, under the key (width class, row, start)."""
        R, S, dev, ys, sw = self.R, self.S, self.dev, self.ys, self.sw
        n = len(gx)
        bc = torch.full((n,), float('inf'), device=dev, dtype=torch.float64)
        bx = torch.zeros(n, dtype=torch.long, device=dev); br = torch.zeros(n, dtype=torch.long, device=dev)
        keys, rr, ra, rb = runs
        if n == 0 or len(keys) == 0:
            return bc, bx, br
        wv, wc = torch.unique(w, return_inverse=True)                       # width classes
        fits = (rb - ra)[None, :] >= wv[:, None]                            # (W, runs)
        ci, ri = torch.nonzero(fits, as_tuple=True)                         # class-major, then key order: sorted
        if len(ci) == 0:
            return bc, bx, br
        span = R * (S + 1)
        K = ci * span + keys[ri]
        rw, aw, bw = rr[ri], ra[ri], rb[ri]
        xr = gx.round().long().clamp(0, S)
        rows = torch.arange(R, device=dev)
        chunk = max(1, budget // R)
        for c0 in range(0, n, chunk):
            c = torch.arange(c0, min(n, c0 + chunk), device=dev)
            q = wc[c][:, None] * span + rows[None, :] * (S + 1) + xr[c][:, None]      # (m, R)
            i = torch.searchsorted(K, q, right=True) - 1                    # last fitting run at or before round(gx)
            ia = i.clamp(min=0); ib = (i + 1).clamp(max=len(K) - 1)
            cls = wc[c][:, None] * span
            okA = (i >= 0) & (K[ia] >= cls) & (rw[ia] == rows[None, :])
            okB = (i + 1 < len(K)) & (K[ib] < cls + span) & (rw[ib] == rows[None, :])
            ww = w[c][:, None]
            xA = torch.minimum(xr[c][:, None], bw[ia] - ww)
            xB = aw[ib]
            al = allowed[0][allowed[1][c]]                                    # (m, R) rows the rails allow
            dy = (ys[None, :] - gy[c][:, None]).abs().double()
            inf = torch.full_like(dy, float('inf'))
            cA = torch.where(okA & al, (xA.double() - gx[c][:, None]).abs() * sw + dy, inf)
            cB = torch.where(okB & al, (xB.double() - gx[c][:, None]).abs() * sw + dy, inf)
            cost = torch.stack([cA, cB], 2).view(len(c), -1)                  # (m, R * 2): row-major, A before B
            bi = cost.argmin(1)
            bc[c] = cost.gather(1, bi[:, None]).squeeze(1)
            r = bi // 2
            bx[c] = torch.where((bi % 2 == 0)[:, None], xA.gather(1, r[:, None]), xB.gather(1, r[:, None])).squeeze(1)
            br[c] = r
        return bc, bx, br

    def resolve(self, x, r, k, w, gain, prio=None):
        """Accept a set of mutually non-overlapping proposals: in every row, overlapping proposals form a cluster
        and only its largest gain survives; a multi-row proposal must survive in all its rows. prio: among equal
        gains, the larger wins (then the lower index)."""
        dev = self.dev
        n = len(x)
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=dev)
        rep, rows, a, b = self.replicas(x, r, k, w)
        o = torch.argsort(rows.double() * (self.S + 1) + a.double())
        rep, rows, a, b = rep[o], rows[o], a[o], b[o]
        off = rows.double() * (self.S + 1) * 4
        run = torch.cummax(off + b.double(), 0).values - off
        newc = torch.ones_like(rows, dtype=torch.bool)
        newc[1:] = (rows[1:] != rows[:-1]) | (a[1:].double() >= run[:-1])
        cl = torch.cumsum(newc.long(), 0) - 1
        # an exact order of the proposals: larger gain, then larger prio, then lower index (stable sorts, last key
        # first); each cluster keeps its first
        order = torch.arange(n, device=dev)
        if prio is not None:
            order = order[torch.argsort(-prio[order].double(), stable=True)]
        order = order[torch.argsort(-gain[order].double(), stable=True)]
        rank = torch.empty(n, dtype=torch.long, device=dev); rank[order] = torch.arange(n, device=dev)
        score = -rank[rep]
        cmax = torch.full((int(cl[-1]) + 1,), -n - 1, device=dev, dtype=torch.long).scatter_reduce(0, cl, score, 'amax')
        lost = torch.zeros(n, dtype=torch.bool, device=dev)
        lost[rep[score < cmax[cl]]] = True
        return ~lost


def _nearest_row(ys, gy):
    R = len(ys)
    if R == 1:
        return torch.zeros_like(gy, dtype=torch.long)
    i = torch.searchsorted(ys, gy).clamp(1, R - 1)
    return torch.where((gy - ys[i - 1]).abs() <= (ys[i] - gy).abs(), i - 1, i)


def _fill(space, cur_x, cur_r, placed, gx, gy, k, w, allowed, stats=None, max_rounds=10_000):
    """Place the cells not placed yet (placed == False) in the nearest free spots around their GP positions.
    Every cell first searches a window of rows and sites around its position. A single-row cell that sees no free
    spot there searches every row at once from then on (free runs, no window); a taller one doubles its own window,
    up to the whole chip. (The window used to grow for every cell at once, so a few cells far from free space made
    every cell search the whole chip, one by one: 1,250 s for 2,038 cells on ispd19_test8.) Conflicting picks resolve
    to one per cluster per round, so rounds continue while they make progress; it stops when a round places nothing
    and no window can grow. Keeps everything legal."""
    rg = _nearest_row(space.ys, gy)
    dev = cur_x.device
    n = len(gx)
    dr = torch.full((n,), 4, dtype=torch.long, device=dev)
    dx = torch.full((n,), 32, dtype=torch.long, device=dev)
    far = torch.zeros(n, dtype=torch.bool, device=dev)                  # single-row cells past their first window
    one = k == 1
    rounds = 0
    for rounds in range(1, max_rounds + 1):                       # noqa: B007  (read after the loop)
        todo = torch.nonzero(~placed).flatten()
        if len(todo) == 0:
            break
        pl = torch.nonzero(placed).flatten()
        P = space.prefix(cur_x[pl], cur_r[pl], k[pl], w[pl])
        cost = torch.full((len(todo),), float('inf'), dtype=torch.float64, device=dev)
        bx = torch.zeros(len(todo), dtype=torch.long, device=dev); br = torch.zeros_like(bx)
        fr = far[todo]
        if bool(fr.any()):
            s1 = torch.nonzero(fr).flatten(); c = todo[s1]
            cost[s1], bx[s1], br[s1] = space.best_free_1row(space.free_runs(P), gx[c], gy[c], w[c],
                                                            (allowed[0], allowed[1][c]))
        if not bool(fr.all()):                                       # the others: one call per window size
            sm = torch.nonzero(~fr).flatten()
            win = dr[todo[sm]] * (space.S + 1) + dx[todo[sm]]
            for v in torch.unique(win).tolist():
                s2 = sm[win == v]; c = todo[s2]
                cost[s2], bx[s2], br[s2] = space.best_free(P, gx[c], gy[c], rg[c], k[c], w[c], (allowed[0], allowed[1][c]),
                                                          None, None, v // (space.S + 1), v % (space.S + 1))
        del P
        found = torch.isfinite(cost)
        # equal distance: the taller, then the wider cell first (OpenDP CellPlaceOrderLess: multi-row, then area)
        kt, wt = k[todo][found], w[todo][found]
        acc = space.resolve(bx[found], br[found], kt, wt, -cost[found], prio=kt * (int(w.max()) + 1) + wt)
        idx = todo[found][acc]
        cur_x[idx] = bx[found][acc]; cur_r[idx] = br[found][acc]; placed[idx] = True
        # a cell that saw no free spot looks wider next round: everywhere if single-row, else twice the window
        miss = todo[~found & ~far[todo]]
        far[miss[one[miss]]] = True
        grow = miss[~one[miss] & ((dr[miss] < space.R) | (dx[miss] < space.S))]
        dr[grow] = (dr[grow] * 2).clamp(max=space.R); dx[grow] = (dx[grow] * 2).clamp(max=space.S)
        if len(idx) == 0 and len(grow) == 0 and not bool(one[miss].any()):   # no progress, nowhere new to look
            break
    if stats is not None:
        stats['fill_rounds'] = rounds
        stats['unplaced'] = int((~placed).sum())
    return cur_x, cur_r, placed


def _improve(space, cur_x, cur_r, gx, gy, k, w, allowed, rounds=30, dr=3, dx=16, stats=None):
    """Parallel displacement reduction on a legal placement (keeps it legal): every cell far from its GP position
    looks for a free spot nearer to it (its own footprint does not count as taken); overlapping proposals are
    resolved by largest gain (see _Space.resolve). Accepted moves overlap neither each other nor anyone's current
    place."""
    rg = _nearest_row(space.ys, gy)
    n = len(gx)
    moved = 0
    it = 0
    for it in range(rounds):                                          # noqa: B007  (read after the loop)
        P = space.prefix(cur_x, cur_r, k, w)
        cur_cost = (cur_x.double() - gx).abs() * space.sw + (space.ys[cur_r] - gy).abs().double()
        cand = torch.nonzero(cur_cost > 0.5 * space.sw).flatten()
        cost, bx, br = space.best_free(P, gx[cand], gy[cand], rg[cand], k[cand], w[cand], (allowed[0], allowed[1][cand]),
                                       cur_x[cand], cur_r[cand], dr, dx)
        del P
        gain = cur_cost[cand] - cost
        better = gain > 1e-9
        if not bool(better.any()):
            break
        prop = cand[better]
        acc = space.resolve(bx[better], br[better], k[prop], w[prop], gain[better])
        idx = prop[acc]
        cur_x[idx] = bx[better][acc]; cur_r[idx] = br[better][acc]
        moved += len(idx)
        if len(idx) < max(1, n // 2000):
            break
    if stats is not None:
        stats['improve_rounds'] = it + 1
        stats['improve_moves'] = stats.get('improve_moves', 0) + moved
    return cur_x, cur_r
