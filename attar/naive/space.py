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
                for j in range(int(k[c].max().item())):
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

    def resolve(self, x, r, k, w, gain):
        """Accept a set of mutually non-overlapping proposals: in every row, overlapping proposals form a cluster
        and only its largest gain survives; a multi-row proposal must survive in all its rows."""
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
        score = gain[rep] * 1e6 - rep.double() * 1e-6
        cmax = torch.full((int(cl[-1]) + 1,), -float('inf'), device=dev, dtype=torch.float64).scatter_reduce(0, cl, score, 'amax')
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
    """Place the cells not placed yet (placed == False) in the nearest free spots around their GP positions; the
    window grows each round until the whole chip is covered. Conflicting picks resolve to one per cluster per round,
    so rounds continue while they make progress; it stops when a round over the whole chip places nothing.
    Keeps everything legal."""
    rg = _nearest_row(space.ys, gy)
    dr, dx = 4, 32
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        todo = torch.nonzero(~placed).flatten()
        if len(todo) == 0:
            break
        pl = torch.nonzero(placed).flatten()
        P = space.prefix(cur_x[pl], cur_r[pl], k[pl], w[pl])
        cost, bx, br = space.best_free(P, gx[todo], gy[todo], rg[todo], k[todo], w[todo], (allowed[0], allowed[1][todo]),
                                       None, None, dr, dx)
        del P
        found = torch.isfinite(cost)
        acc = space.resolve(bx[found], br[found], k[todo][found], w[todo][found], -cost[found])
        idx = todo[found][acc]
        cur_x[idx] = bx[found][acc]; cur_r[idx] = br[found][acc]; placed[idx] = True
        whole = dr >= space.R and dx >= space.S
        if not bool(found.all()):                                    # someone saw no free spot: look wider
            dr = min(space.R, dr * 2); dx = min(space.S, dx * 2)
        if len(idx) == 0 and whole:                                  # the whole chip and no progress: give up
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
    for it in range(rounds):
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
