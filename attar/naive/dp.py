"""Naive detailed placement: wirelength improvement of a legal placement that keeps it legal, after OpenROAD's
improve_placement (dpl Optdp.cpp: the dpo script "mis; gs; vs; ro; default", then cell flipping). Four of its
passes are ported, batched:

  ro  local reordering (dpo DetailedReorderer): in every row segment, windows of `window` consecutive single-row
      cells; every permutation of a window is laid out between its neighbours, the spare sites spread over the cells
      as dpo does (each cell gets floor(spare / n) sites, half right, half left, while they fit), and the cheapest
      permutation wins if it beats the current order. dpo slides the window by one cell and never lets it reach a
      run's last cell (i + w <= last); the same windows are used here.
  gs  global swap (dpo DetailedGlobalSwap, its wirelength pass): every single-row cell's optimal region is the median
      box of its nets' bounding boxes without it (dpo getRange); a cell already inside it stays. Otherwise its target
      is the box centre on the nearest row: it moves into the gap of the target segment there if the gap holds it
      (dpo tryMove1/2), else it swaps with the cell at the target (dpo trySwap1, non-adjacent case: each cell goes,
      site-aligned and as near its target as fits, into the other's free span). Kept if the HPWL goes down.

  vs  vertical swap (dpo DetailedVerticalSwap): as gs, with the target one or two rows up or down towards the box.
  flip  cell flipping (dpo DetailedOrient::flipCells): a cell mirrored left to right where its master allows it,
      if that shortens its nets in x.

The objective is dpo's: HPWL (x for ro, x + y for gs) of the nets with 2..99 pins (skipNetsLargerThanThis_ = 100),
here weighted by the nets' weights (1 unless the design says otherwise). dpo works one candidate at a time; here
candidates are evaluated in parallel and applied in batches whose candidates share no cell and no costed net (and
whose reorder windows are not neighbours), so every applied gain is the exact gain: the result is that of a
sequential run in some order. Moves keep rows, sites, rails (the orientation a row needs, as the legalizer sets it,
a cell's left-right mirroring kept), row gaps, and, with the 'pdn' fact, signal pins clear of the PDN.

Not ported (yet): dpo's neighbour shifting when a gap is too small (tryMove pushes cells; here such a move is not
made), adjacent swaps (dpo shift()), independent-set matching (mis), the random-move pass, the
congestion-aware second global-swap pass, and multi-row cells (fixed here, as dpo's ro / gs skip them too).
"""
import itertools
import time

import torch

from ..io.distill import ASIDE, MACRO, TERMINAL
from ..rules import pdn as PDN
from ..rules import pin_access as PA
from ..rules import rails as RL
from ..rules.rails import base_rows as _base_rows
from .rowops import BIG, _locate, _segments
from ..gpu import explain_oom

SKIP_NETS = 100               # dpo skipNetsLargerThanThis_: nets with this many pins or more are not costed
MIRROR = {0: 4, 5: 1}         # N -> FN, FS -> S: a cell mirrored left-right stays mirrored where it goes


class _DP:
    """The placement as rows of single-row cells in free segments, and the net cost of candidate moves."""

    def __init__(self, d, pin_access=True):
        self.d = d
        dev = self.dev = d.device
        hb, ys, rx0, rx1, sw, rorient = _base_rows(d)
        self.R = len(ys)
        self.hb, self.ys, self.rx0, self.rorient = float(hb), ys, rx0.double(), rorient
        self.sw = torch.full((self.R,), float(sw), dtype=torch.float64, device=dev)       # per row (one site width)
        self.nsites = ((rx1 - rx0) / sw).round()
        flags = d.flags
        self.k = (d.size[:, 1] / hb).round().clamp(min=1).long()
        std = (flags & (MACRO | TERMINAL | ASIDE)) == 0
        self.cells = torch.nonzero(d.movable & std & (self.k == 1)).flatten()         # what moves
        # segments: the rows' free stretches between fixed objects, multi-row cells and row gaps (site units, per row)
        blk = torch.nonzero((~d.movable | (self.k > 1) | ~std) & ((flags & (TERMINAL | ASIDE)) == 0)).flatten()
        p = d.pos.double()
        rlo, rhi = RL.rows_spanned(ys, hb, d.pos[blk, 1], d.pos[blk, 1] + d.size[blk, 1])
        rlo = rlo.clamp(max=self.R)
        n = (rhi - rlo).clamp(min=0)
        rep = torch.repeat_interleave(torch.arange(len(blk), device=dev), n)
        rows = rlo[rep] + (torch.arange(len(rep), device=dev) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n))
        i = blk[rep]
        a = ((p[i, 0] - self.rx0[rows]) / self.sw[rows] + 1e-3).floor()
        b = ((p[i, 0] + d.size[i, 0].double() - self.rx0[rows]) / self.sw[rows] - 1e-3).ceil()
        g_r, g_x0, g_x1 = RL.row_gaps(d, ys)
        ga = ((g_x0.double() - self.rx0[g_r]) / self.sw[g_r] + 1e-3).floor()
        gb = ((g_x1.double() - self.rx0[g_r]) / self.sw[g_r] - 1e-3).ceil()
        self.seg_row, self.s0, self.s1 = _segments(torch.cat([rows, g_r]), torch.cat([a, ga]), torch.cat([b, gb]),
                                                   self.R, self.nsites)
        # nets costed (dpo): 2..99 pins; pins by cell
        deg = d.n_ptr[1:] - d.n_ptr[:-1]
        self.net_ok = (deg >= 2) & (deg < SKIP_NETS)
        self.deg = deg
        self.w_net = d.n_weight.double()
        order = torch.argsort(d.p_cell, stable=True)
        self.pin_by_cell = order
        self.cell_ptr = torch.searchsorted(d.p_cell[order], torch.arange(len(d.pos) + 1, device=dev))
        # rails and PDN
        self.RI = RL.info(d, (ys, rorient))
        self.pdn = PDN.Index(d, dev) if 'pdn' in d.facts else None
        if self.pdn is not None and self.pdn.n == 0:
            self.pdn = None
        self.pa = PA.Checker(d, dev) if pin_access and 'pdn' in d.facts and 'tracks' in d.facts else None
        if self.pa is not None and self.pa.n == 0:
            self.pa = None
        self.pdn_check = self.pdn is not None or self.pa is not None
        if self.pdn_check:
            self.pin_rel, self.pin_layer = PDN.pin_shapes(d, dev)
        self.refresh()

    # -- state ----------------------------------------------------------------------------------------------------------
    def refresh(self):
        """Rows, sites and segments of the movable single-row cells, from d.pos; cells sorted by (segment, x)."""
        d, c = self.d, self.cells
        y = d.pos[c, 1].double()
        r = torch.searchsorted(self.ys.double(), y - 0.25 * self.hb).clamp(max=self.R - 1)
        x = ((d.pos[c, 0].double() - self.rx0[r]) / self.sw[r]).round()
        w = (d.size[c, 0].double() / self.sw[r]).round()
        seg = _locate(self.seg_row, self.s0, self.s1, r, x, w)
        if bool((seg < 0).any()):
            raise RuntimeError('%d single-row cells are outside the free row segments: legalize first' % int((seg < 0).sum()))
        o = torch.argsort(seg.double() * BIG + x)
        self.row, self.x, self.w, self.seg = r, x, w, seg
        self.sorted = o                                    # positions into self.cells, by (segment, x)
        self.rank = torch.empty_like(o); self.rank[o] = torch.arange(len(o), device=self.dev)
        sg = seg[o]
        first = torch.ones_like(sg, dtype=torch.bool); first[1:] = sg[1:] != sg[:-1]
        idx = torch.arange(len(sg), device=self.dev)
        start = torch.cummax(torch.where(first, idx, torch.zeros_like(idx)), 0).values
        last = torch.ones_like(first); last[:-1] = first[1:]
        end = torch.flip(torch.cummin(torch.flip(torch.where(last, idx, torch.full_like(idx, len(sg))), [0]), 0).values, [0])
        self.t = idx - start                               # position in its segment, by x
        self.n_in_seg = end - start + 1

    def pin_pos_with(self, pins, cell_of_pin, sub_x, sub_y, sub_o):
        """Pin positions (P, 2) float64; pins of moved cells (sub_* given per pin, NaN where not moved) at the new
        lower-left and orientation."""
        d = self.d
        cur = d.pos[cell_of_pin].double() + d.size[cell_of_pin].double() / 2 + d.pin_off[pins].double()
        moved = ~torch.isnan(sub_x)
        if not bool(moved.any()):
            return cur
        from ..design import _orient_offset
        mp = pins[moved]; mc = cell_of_pin[moved]
        w, h = d.size_n[mc, 0], d.size_n[mc, 1]
        off = _orient_offset(sub_o[moved], d._p_ctr_n[mp, 0], d._p_ctr_n[mp, 1], w, h).double()
        new = torch.stack([sub_x[moved], sub_y[moved]], 1) + off
        cur[moved] = new
        return cur

    def cost(self, mv_cells, mv_x, mv_y, mv_o, xonly):
        """Cost of C candidates: mv_cells (C, M) cell ids (-1: none), their lower-left (C, M) um and orientation
        (C, M). Cost = sum over the costed nets of the moved cells (each once per candidate) of the net's weighted
        HPWL (x only if xonly) with the moved cells at the given places, every other pin where it is. -> (C,)"""
        d, dev = self.d, self.dev
        C, M = mv_cells.shape
        # (candidate, net) pairs from the moved cells' pins
        valid = mv_cells >= 0
        cc = mv_cells.clamp(min=0)
        cnt = torch.where(valid, self.cell_ptr[cc + 1] - self.cell_ptr[cc], torch.zeros_like(cc)).reshape(-1)
        slot = torch.repeat_interleave(torch.arange(C * M, device=dev), cnt)
        off = torch.arange(len(slot), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        pin = self.pin_by_cell[self.cell_ptr[cc.reshape(-1)[slot]] + off]
        net = d.p_net[pin]
        keep = self.net_ok[net]
        cand = slot[keep] // M; net = net[keep]
        if len(cand) == 0:
            return torch.zeros(C, dtype=torch.float64, device=dev)
        E = len(self.deg)
        key = torch.unique(cand * E + net)
        pc, pn = key // E, key % E
        # every pin of each pair's net
        dg = self.deg[pn]
        rep = torch.repeat_interleave(torch.arange(len(key), device=dev), dg)
        po = torch.arange(len(rep), device=dev) - torch.repeat_interleave(torch.cumsum(dg, 0) - dg, dg)
        pins = d.n_ptr[pn][rep] + po
        pcell = d.p_cell[pins]
        # is the pin's cell moved in this candidate? which slot?
        mvc = mv_cells[pc[rep]]                                          # (P, M)
        hit = mvc == pcell[:, None]
        anyhit = hit.any(1)
        j = hit.long().argmax(1)
        ci = pc[rep]
        nan = torch.full((len(pins),), float('nan'), dtype=torch.float64, device=dev)
        sx = torch.where(anyhit, mv_x[ci, j].double(), nan)
        sy = torch.where(anyhit, mv_y[ci, j].double(), nan)
        so = torch.where(anyhit, mv_o[ci, j], torch.zeros_like(j))
        P = self.pin_pos_with(pins, pcell, sx, sy, so)
        lo = torch.full((len(key), 2), float('inf'), dtype=torch.float64, device=dev).scatter_reduce(
            0, rep[:, None].expand(-1, 2), P, 'amin')
        hi = torch.full((len(key), 2), -float('inf'), dtype=torch.float64, device=dev).scatter_reduce(
            0, rep[:, None].expand(-1, 2), P, 'amax')
        span = hi - lo
        wl = span[:, 0] if xonly else span.sum(1)
        return torch.zeros(C, dtype=torch.float64, device=dev).index_add_(0, pc, wl * self.w_net[pn])

    def nets_of(self, mv_cells):
        """(candidate, costed net) pairs of the candidates' cells, unique."""
        d, dev = self.d, self.dev
        C, M = mv_cells.shape
        valid = mv_cells >= 0
        cc = mv_cells.clamp(min=0)
        cnt = torch.where(valid, self.cell_ptr[cc + 1] - self.cell_ptr[cc], torch.zeros_like(cc)).reshape(-1)
        slot = torch.repeat_interleave(torch.arange(C * M, device=dev), cnt)
        off = torch.arange(len(slot), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        pin = self.pin_by_cell[self.cell_ptr[cc.reshape(-1)[slot]] + off]
        net = d.p_net[pin]
        keep = self.net_ok[net]
        E = len(self.deg)
        key = torch.unique((slot[keep] // M) * E + net[keep])
        return key // E, key % E

    def independent(self, mv_cells, gain):
        """A subset of candidates with no shared cell and no shared costed net, larger gains first. -> bool (C,)"""
        dev = self.dev
        C = len(mv_cells)
        if C == 0:
            return torch.zeros(0, dtype=torch.bool, device=dev)
        order = torch.argsort(-gain, stable=True)
        rank = torch.empty(C, dtype=torch.long, device=dev); rank[order] = torch.arange(C, device=dev)
        pc, pn = self.nets_of(mv_cells)
        # owners: per net and per cell, the best-ranked candidate touching it
        n_own = torch.full((len(self.deg),), C, dtype=torch.long, device=dev).scatter_reduce(0, pn, rank[pc], 'amin')
        cm = mv_cells.reshape(-1); cr = rank[:, None].expand_as(mv_cells).reshape(-1)
        ok = cm >= 0
        c_own = torch.full((len(self.d.pos),), C, dtype=torch.long, device=dev).scatter_reduce(0, cm[ok], cr[ok], 'amin')
        win = torch.ones(C, dtype=torch.bool, device=dev)
        lost_n = n_own[pn] != rank[pc]
        win[pc[lost_n]] = False
        lost_c = torch.zeros(len(cm), dtype=torch.bool, device=dev)
        lost_c[ok] = c_own[cm[ok]] != cr[ok]
        win[torch.arange(C, device=dev).repeat_interleave(mv_cells.shape[1])[lost_c]] = False
        return win

    def pdn_ok(self, cells, x, y, o):
        """Bool per entry: cell at lower-left (x, y), orientation o, has no signal pin on a PDN pin blockage (OpenROAD
        dpl's rule) and, with pin_access, makes none unreachable past the PDN that is reachable where the cell stands
        (attar.rules.pin_access)."""
        ok = torch.ones(len(cells), dtype=torch.bool, device=self.dev)
        if not self.pdn_check or len(cells) == 0:
            return ok
        d = self.d
        cnt = self.cell_ptr[cells + 1] - self.cell_ptr[cells]
        owner = torch.repeat_interleave(torch.arange(len(cells), device=self.dev), cnt)
        off = torch.arange(len(owner), device=self.dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        pins = self.pin_by_cell[self.cell_ptr[cells][owner] + off]
        lay = self.pin_layer[pins]
        k = lay >= 0
        pins, owner, lay = pins[k], owner[k], lay[k]
        c = cells[owner]
        pb = PDN.place_boxes(self.pin_rel[pins], d.size[c, 0].double(), d.size[c, 1].double(), o[owner], x[owner], y[owner])
        hit = self.pdn.hits(pb, lay) if self.pdn is not None else torch.zeros(len(pb), dtype=torch.bool, device=self.dev)
        if self.pa is not None:          # a pin made unreachable; one that is unreachable where the cell stands may stay so
            now = PDN.place_boxes(self.pin_rel[pins], d.size[c, 0].double(), d.size[c, 1].double(), d.orient[c],
                                  d.pos[c, 0].double(), d.pos[c, 1].double())
            hit |= self.pa.blocked(pb, lay)[0] & ~self.pa.blocked(now, lay)[0]
        ok[owner[hit]] = False
        return ok

    def orient_at(self, cells, rows):
        """The orientation single-row cells get in these rows (as the legalizer sets it: the row's, or the one the
        rails need), keeping a left-right mirroring the cell has now."""
        d, RI = self.d, self.RI
        row_o = self.rorient[rows].long()
        if RI['known']:
            cb, ct = RI['cell_bot'][cells], RI['cell_top'][cells]
            rb, rt = RI['row_bot'][rows], RI['row_top'][rows]
            keep = RL._eq(cb, rb) & RL._eq(ct, rt)
            flip = RL._eq(ct, rb) & RL._eq(cb, rt)
            keep, flip = keep | ~flip, flip | ~keep                     # single-row cells fit one way or the other
            o = torch.where(keep & ~flip, torch.zeros_like(row_o), row_o)
            o = torch.where(flip & ~keep, torch.full_like(row_o, 5), o)
        else:
            o = row_o
        cur = d.orient[cells].long()
        mir = (cur == 4) | (cur == 1)
        m = torch.full_like(o, -1)
        for a, b in MIRROR.items():
            m = torch.where(o == a, torch.full_like(o, b), m)
        return torch.where(mir & (m >= 0), m, o)

    def apply(self, cells, x_um, y_um, o):
        d = self.d
        pos = d.pos.clone(); pos[cells, 0] = x_um.to(pos.dtype); pos[cells, 1] = y_um.to(pos.dtype)
        orient = d.orient.clone(); orient[cells] = o.to(orient.dtype)
        d.pos = pos
        if bool((orient != d.orient).any()):
            d.orient = orient; d.update_orient()


# -- local reordering ------------------------------------------------------------------------------------------------
def _reorder_pass(S, W, stats, chunk=200_000):
    """One dpo reorder pass over every window (phases of non-neighbour windows; batches without shared nets)."""
    d, dev = S.d, S.dev
    perms = torch.tensor(list(itertools.permutations(range(W))), device=dev)          # (P, W)
    P = len(perms)
    gained = 0.0
    applied = 0
    for phase in range(W + 1):
        S.refresh()
        t, n = S.t, S.n_in_seg
        # window starts (sorted index j): dpo's i + w <= last index of the run, with stride w + 1 by phase
        start = torch.nonzero((t + W <= n - 1) & (t % (W + 1) == phase)).flatten()
        todo = start
        rounds = 0
        while len(todo):
            rounds += 1
            S.refresh()
            j = todo
            idx = j[:, None] + torch.arange(W, device=dev)[None, :]                   # sorted indices (Wn, W)
            ci = S.sorted[idx]                                                         # into S.cells
            cells = S.cells[ci]
            sg = S.seg[ci[:, 0]]
            row = S.row[ci[:, 0]]
            sw = S.sw[row]
            xs, ws = S.x[ci], S.w[ci]
            prev_ok = S.t[j] > 0
            pj = S.sorted[(j - 1).clamp(min=0)]
            L = torch.where(prev_ok, S.x[pj] + S.w[pj], S.s0[sg])
            nx = S.sorted[j + W]
            Rl = S.x[nx]
            tw = ws.sum(1)
            space = Rl - L - tw
            okw = space >= 0
            # dpo spacing: floor(spare / n) sites per cell, right half (floor) then left, while it fits
            per = (space / W).floor().clamp(min=0)
            rgt = (per / 2).floor(); lft = per - rgt
            pad_l = torch.zeros_like(ws); pad_r = torch.zeros_like(ws)
            tot = tw.clone()
            for i in range(W):
                fit_r = tot + rgt < Rl - L
                pad_r[:, i] = torch.where(fit_r, rgt, 0.0); tot = tot + pad_r[:, i]
                fit_l = tot + lft < Rl - L
                pad_l[:, i] = torch.where(fit_l, lft, 0.0); tot = tot + pad_l[:, i]
            # positions of each permutation (in sites, row-relative)
            pw = ws[:, perms]; pl = pad_l[:, perms]; pr = pad_r[:, perms]                 # (Wn, P, W)
            adv = pl + pw + pr
            xpos = L[:, None, None] + torch.cumsum(adv, 2) - pw - pr                   # left edge of each placed cell
            newx = torch.empty_like(xpos)
            newx.scatter_(2, perms[None].expand(len(j), -1, -1), xpos)                 # by original window slot
            Wn = len(j)
            mv_cells = cells[:, None, :].expand(-1, P, -1).reshape(Wn * P, W)
            x_um = (S.rx0[row][:, None, None] + newx * sw[:, None, None]).reshape(Wn * P, W)
            y_um = S.ys.double()[row][:, None, None].expand(-1, P, W).reshape(Wn * P, W)
            o = d.orient[cells].long()[:, None, :].expand(-1, P, -1).reshape(Wn * P, W)
            cost = torch.empty(Wn * P, dtype=torch.float64, device=dev)
            for c0 in range(0, Wn * P, chunk):
                sl = slice(c0, min(Wn * P, c0 + chunk))
                cost[sl] = S.cost(mv_cells[sl], x_um[sl], y_um[sl], o[sl], xonly=True)
            cost = cost.view(Wn, P)
            if S.pdn_check:
                ok = S.pdn_ok(mv_cells.reshape(-1), x_um.reshape(-1), y_um.reshape(-1), o.reshape(-1)).view(Wn, P, W).all(2)
                cost = torch.where(ok, cost, torch.full_like(cost, float('inf')))
            cur = torch.empty(Wn, dtype=torch.float64, device=dev)
            cx_um = S.rx0[row][:, None] + xs * sw[:, None]
            for c0 in range(0, Wn, chunk):
                sl = slice(c0, min(Wn, c0 + chunk))
                cur[sl] = S.cost(cells[sl], cx_um[sl], S.ys.double()[row][sl, None].expand(-1, W), d.orient[cells[sl]].long(), xonly=True)
            best = cost.argmin(1)
            bc = cost.gather(1, best[:, None]).squeeze(1)
            gain = cur - bc
            better = okw & (gain > 1e-9)
            sel = torch.zeros(Wn, dtype=torch.bool, device=dev)
            if bool(better.any()):
                b_idx = torch.nonzero(better).flatten()
                win = S.independent(cells[b_idx], gain[b_idx])
                sel[b_idx[win]] = True
            if bool(sel.any()):
                s = torch.nonzero(sel).flatten()
                nxs = newx[s, best[s]]                                                  # (k, W) sites
                cs = cells[s].reshape(-1)
                S.apply(cs, (S.rx0[row[s]][:, None] + nxs * sw[s][:, None]).reshape(-1),
                        S.ys.double()[row[s]][:, None].expand(-1, W).reshape(-1), d.orient[cs].long())
                gained += float(gain[s].sum()); applied += len(s)
            # windows not improving are done; improving ones that lost a conflict wait for the next batch
            todo = j[better & ~sel]
        stats['ro_batches'] = stats.get('ro_batches', 0) + rounds
    stats['ro_windows_applied'] = stats.get('ro_windows_applied', 0) + applied
    return gained


# -- global swap -----------------------------------------------------------------------------------------------------
def _span_winners(rows, a, b, cid, gain, C):
    """Claims (row, [a, b)) of C candidates (cid: the candidate of each claim): in each row, claims that overlap
    (transitively) form a cluster won by the largest gain (then the lower index); a candidate must win every claim."""
    dev = rows.device
    if len(rows) == 0:
        return torch.ones(C, dtype=torch.bool, device=dev)
    order = torch.argsort(-gain, stable=True)
    rank = torch.empty(C, dtype=torch.long, device=dev); rank[order] = torch.arange(C, device=dev)
    o = torch.argsort(rows.double() * BIG + a.double())
    r, a, b, cid = rows[o], a[o].double(), b[o].double(), cid[o]
    off = r.double() * BIG * 4
    run = torch.cummax(off + b, 0).values - off
    newc = torch.ones_like(r, dtype=torch.bool)
    newc[1:] = (r[1:] != r[:-1]) | (a[1:] >= run[:-1])
    cl = torch.cumsum(newc.long(), 0) - 1
    best = torch.full((int(cl[-1]) + 1,), C, dtype=torch.long, device=dev).scatter_reduce(0, cl, rank[cid], 'amin')
    win = torch.ones(C, dtype=torch.bool, device=dev)
    win[cid[rank[cid] != best[cl]]] = False
    return win

def _optimal_region(S, cells):
    """dpo getRange: per cell the median box (for its centre) of its costed nets' bounding boxes without it.
    -> (ok, xlo, xhi, ylo, yhi) um."""
    d, dev = S.d, S.dev
    n = len(cells)
    cnt = S.cell_ptr[cells + 1] - S.cell_ptr[cells]
    owner = torch.repeat_interleave(torch.arange(n, device=dev), cnt)
    off = torch.arange(len(owner), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    pin = S.pin_by_cell[S.cell_ptr[cells][owner] + off]
    net = d.p_net[pin]
    keep = S.net_ok[net]
    pin, owner, net = pin[keep], owner[keep], net[keep]
    # every other pin of each (cell pin, net)
    dg = S.deg[net]
    rep = torch.repeat_interleave(torch.arange(len(pin), device=dev), dg)
    po = torch.arange(len(rep), device=dev) - torch.repeat_interleave(torch.cumsum(dg, 0) - dg, dg)
    other = d.n_ptr[net][rep] + po
    oc = d.p_cell[other]
    mine = oc == cells[owner[rep]]
    pp = d.pin_pos().double()
    q = pp[other]
    inf = torch.full_like(q, float('inf'))
    lo = torch.full((len(pin), 2), float('inf'), dtype=torch.float64, device=dev).scatter_reduce(
        0, rep[:, None].expand(-1, 2), torch.where(mine[:, None], inf, q), 'amin')
    hi = torch.full((len(pin), 2), -float('inf'), dtype=torch.float64, device=dev).scatter_reduce(
        0, rep[:, None].expand(-1, 2), torch.where(mine[:, None], -inf, q), 'amax')
    has = torch.isfinite(lo[:, 0])
    # interval for the cell centre: minus the pin's offset from the centre; clamped to the chip
    po_ = d.pin_off[pin].double()
    cmin = torch.stack([S.rx0.min(), S.ys.double().min()])
    cmax = torch.stack([(S.rx0 + S.nsites.double() * S.sw).max(), S.ys.double().max() + S.hb])
    lo = torch.minimum(torch.maximum(lo - po_, cmin), cmax); hi = torch.maximum(torch.minimum(hi - po_, cmax), cmin)
    pin_ok = has
    owner = owner[pin_ok]; lo = lo[pin_ok]; hi = hi[pin_ok]
    # median: per cell, sort its 2t values; the box is [v[t-1], v[t]] of each axis (t = number of nets)
    t = torch.bincount(owner, minlength=n)
    ok = t >= 1                                                    # dpo: at least two points (one net)
    out = []
    for ax in (0, 1):
        v = torch.cat([lo[:, ax], hi[:, ax]]); ow = torch.cat([owner, owner])
        o = torch.argsort(ow.double() * 0 + v, stable=True)
        o = o[torch.argsort(ow[o], stable=True)]
        vs = v[o]
        st = torch.cumsum(2 * t, 0) - 2 * t
        mid = t                                                    # dpo: mid = (2t) >> 1 = t
        a = vs[(st + mid - 1).clamp(min=0, max=max(len(vs) - 1, 0))] if len(vs) else torch.zeros(n, dtype=torch.float64, device=dev)
        b = vs[(st + mid).clamp(max=max(len(vs) - 1, 0))] if len(vs) else torch.zeros(n, dtype=torch.float64, device=dev)
        out += [a, b]
    return ok, out[0], out[1], out[2], out[3]


def _global_swap_pass(S, stats, max_batches=20, vertical=False, gen=None):
    """One dpo global-swap pass (wirelength objective): optimal-region moves and swaps, applied in batches.
    vertical: dpo's vertical swap instead (DetailedVerticalSwap::generate): the target row is one or two rows (at
    random) up or down from the cell's, towards the optimal box (dpo compares the box's target bottom with the cell's
    centre), the target x the box centre as in the global swap."""
    d, dev = S.d, S.dev
    tag = 'vs' if vertical else 'gs'
    S.refresh()
    todo = torch.arange(len(S.cells), device=dev)                  # into S.cells
    gained, moves, swaps = 0.0, 0, 0
    for _ in range(max_batches):
        if len(todo) == 0:
            break
        S.refresh()
        cells = S.cells[todo]
        ok, xlo, xhi, ylo, yhi = _optimal_region(S, cells)
        w_um = d.size[cells, 0].double(); h_um = d.size[cells, 1].double()
        cx = d.pos[cells, 0].double() + w_um / 2; cy = d.pos[cells, 1].double() + h_um / 2
        inside = (cx >= xlo) & (cx <= xhi) & (cy >= ylo) & (cy <= yhi)
        act = ok & ~inside
        a = torch.nonzero(act).flatten()
        if len(a) == 0:
            break
        ti = todo[a]; c = cells[a]
        xj = torch.floor(0.5 * (xlo[a] + xhi[a]) - 0.5 * w_um[a])
        yj = torch.floor(0.5 * (ylo[a] + yhi[a]) - 0.5 * h_um[a])
        ysd = S.ys.double()
        if vertical:
            # dpo: a row in ri+1 .. ri+2 (or ri-2 .. ri-1) at random, each end clamped to the rows
            step = 1 + torch.randint(0, 2, (len(a),), generator=gen).to(dev)
            up = yj > cy[a]
            rj = torch.where(up, torch.clamp(S.row[ti] + step, max=S.R - 1), torch.clamp(S.row[ti] - step, min=0))
        else:
            rj = torch.searchsorted(ysd, yj).clamp(1, S.R - 1) if S.R > 1 else torch.zeros_like(a)
            if S.R > 1:
                rj = torch.where((yj - ysd[rj - 1]).abs() <= (ysd[rj] - yj).abs(), rj - 1, rj)
        xs_t = ((xj - S.rx0[rj]) / S.sw[rj])                                            # target left, row sites
        # target segment: the one of row rj whose [s0, s1] holds xj
        key = S.seg_row.double() * BIG + S.s0
        sj = (torch.searchsorted(key, rj.double() * BIG + xs_t, right=True) - 1).clamp(0, max(len(S.seg_row) - 1, 0))
        insg = (S.seg_row[sj] == rj) & (xs_t >= S.s0[sj]) & (xs_t <= S.s1[sj])
        wi = S.w[ti]; xi = S.x[ti]; si = S.seg[ti]; ri = S.row[ti]
        # neighbours at the target, in the sorted (segment, x) order: first cell of sj with x >= xj (dpo lower_bound)
        sk = S.seg[S.sorted].double() * BIG + S.x[S.sorted]
        pos_r = torch.searchsorted(sk, sj.double() * BIG + xs_t)
        Nn = len(S.sorted)
        r_ok = (pos_r < Nn)
        rr = S.sorted[pos_r.clamp(max=Nn - 1)]
        r_ok &= S.seg[rr] == sj
        l_pos = pos_r - 1
        l_ok = l_pos >= 0
        ll = S.sorted[l_pos.clamp(min=0)]
        l_ok &= S.seg[ll] == sj
        # tryMove (1: other segment, 2: same segment): the gap between the neighbours at the target, without the
        # cell itself (same segment: skip it if it is a neighbour)
        self_r = r_ok & (rr == ti); self_l = l_ok & (ll == ti)
        rr2 = S.sorted[(pos_r + 1).clamp(max=Nn - 1)]
        r2_ok = (pos_r + 1 < Nn) & (S.seg[rr2] == sj)
        ll2 = S.sorted[(l_pos - 1).clamp(min=0)]
        l2_ok = (l_pos - 1 >= 0) & (S.seg[ll2] == sj)
        R_c = torch.where(self_r, rr2, rr); R_ok = torch.where(self_r, r2_ok, r_ok)
        L_c = torch.where(self_l, ll2, ll); L_ok = torch.where(self_l, l2_ok, l_ok)
        lx = torch.where(L_ok, S.x[L_c] + S.w[L_c], S.s0[sj])
        rx = torch.where(R_ok, S.x[R_c], S.s1[sj])
        mx = torch.minimum(torch.maximum(xs_t, lx), rx - wi).floor()                   # alignPos (site grid)
        mx = torch.where(mx < lx, mx + 1, mx)
        can_move = insg & (mx >= lx) & (mx + wi <= rx)
        # trySwap1 (non-adjacent): partner = first cell of sj at or right of xj, else the last one of sj
        part = torch.where(r_ok, rr, torch.where(l_ok, ll, torch.full_like(rr, -1)))
        has_p = insg & (part >= 0) & (part != ti)
        pc = part.clamp(min=0)
        adjacent = (S.seg[pc] == si) & ((S.rank[pc] - S.rank[ti]).abs() == 1)
        # free span of each slot (neighbours of the cell, excluding itself)
        def span(cidx):
            rk = S.rank[cidx]
            pv = S.sorted[(rk - 1).clamp(min=0)]; pv_ok = (rk > 0) & (S.seg[pv] == S.seg[cidx])
            nv = S.sorted[(rk + 1).clamp(max=Nn - 1)]; nv_ok = (rk + 1 < Nn) & (S.seg[nv] == S.seg[cidx])
            lo_ = torch.where(pv_ok, S.x[pv] + S.w[pv], S.s0[S.seg[cidx]])
            hi_ = torch.where(nv_ok, S.x[nv], S.s1[S.seg[cidx]])
            return lo_, hi_
        li, hi_i = span(ti); lj, hj = span(pc)
        wj = S.w[pc]
        # ndj into ndi's slot as near xi, ndi into ndj's slot as near xj
        xj_new = torch.minimum(torch.maximum(xi, li), hi_i - wj).floor()
        xi_new = torch.minimum(torch.maximum(xs_t, lj), hj - wi).floor()
        xi_new = torch.where(xi_new < lj, xi_new + 1, xi_new)
        can_swap = has_p & ~adjacent & (wj <= hi_i - li) & (wi <= hj - lj) & (xj_new >= li) & (xi_new >= lj) \
            & (xj_new + wj <= hi_i) & (xi_new + wi <= hj)
        use_move = can_move
        use_swap = ~can_move & can_swap
        cand = use_move | use_swap
        k = torch.nonzero(cand).flatten()
        if len(k) == 0:
            break
        # candidates: (cell, partner) with new places
        c1 = c[k]; t1 = ti[k]
        r1 = rj[k]
        x1 = torch.where(use_move[k], mx[k], xi_new[k])
        o1 = S.orient_at(c1, r1)
        p2 = pc[k]
        c2 = torch.where(use_swap[k], S.cells[p2], torch.full_like(c1, -1))
        x2 = xj_new[k]
        r2 = ri[k]
        o2 = torch.where(c2 >= 0, S.orient_at(c2.clamp(min=0), r2), torch.zeros_like(o1))
        mv_cells = torch.stack([c1, c2], 1)
        X = torch.stack([S.rx0[r1] + x1 * S.sw[r1], S.rx0[r2] + x2 * S.sw[r2]], 1)
        Y = torch.stack([ysd[r1], ysd[r2]], 1)
        O = torch.stack([o1, o2], 1)
        new = S.cost(mv_cells, X, Y, O, xonly=False)
        cur_X = torch.stack([d.pos[c1, 0].double(), d.pos[c2.clamp(min=0), 0].double()], 1)
        cur_Y = torch.stack([d.pos[c1, 1].double(), d.pos[c2.clamp(min=0), 1].double()], 1)
        cur_O = torch.stack([d.orient[c1].long(), d.orient[c2.clamp(min=0)].long()], 1)
        old = S.cost(mv_cells, cur_X, cur_Y, cur_O, xonly=False)
        gain = old - new
        good = gain > 1e-9
        if S.pdn_check:                          # only the improving candidates need the PDN check
            gi = torch.nonzero(good).flatten()
            ok = S.pdn_ok(c1[gi], X[gi, 0], Y[gi, 0], o1[gi])
            sw = c2[gi] >= 0
            gs_ = gi[sw]
            ok[sw] &= S.pdn_ok(c2[gs_], X[gs_, 1], Y[gs_, 1], o2[gs_])
            good[gi] = ok
        g = torch.nonzero(good).flatten()
        if len(g) == 0:
            break
        win = S.independent(mv_cells[g], gain[g])
        # space: a move may fill its target gap, a swap both cells' free spans; overlapping claims in a row keep the
        # largest gain (a candidate must win all its claims)
        is_sw = c2[g] >= 0
        ca_ = torch.cat([torch.where(is_sw, li[k][g], lx[k][g]), lj[k][g][is_sw]])
        cb_ = torch.cat([torch.where(is_sw, hi_i[k][g], rx[k][g]), hj[k][g][is_sw]])
        cr_ = torch.cat([torch.where(is_sw, ri[k][g], r1[g]), r1[g][is_sw]])
        cid = torch.cat([torch.arange(len(g), device=dev), torch.nonzero(is_sw).flatten()])
        win &= _span_winners(cr_, ca_, cb_, cid, gain[g], len(g))
        gm = g[win]
        if len(gm) == 0:
            break
        sw_ = c2[gm] >= 0
        cells_ap = torch.cat([c1[gm], c2[gm][sw_]])
        S.apply(cells_ap, torch.cat([X[gm, 0], X[gm, 1][sw_]]), torch.cat([Y[gm, 0], Y[gm, 1][sw_]]),
                torch.cat([O[gm, 0], O[gm, 1][sw_]]))
        gained += float(gain[gm].sum()); swaps += int(sw_.sum()); moves += int((~sw_).sum())
        # cells with an improving candidate that lost a conflict try again with the new placement
        done_t = set(t1[gm].tolist())
        todo = torch.tensor([x for x in t1[g].tolist() if x not in done_t], dtype=torch.long, device=dev)
    stats[tag + '_moves'] = stats.get(tag + '_moves', 0) + moves
    stats[tag + '_swaps'] = stats.get(tag + '_swaps', 0) + swaps
    return gained


# -- cell flipping ---------------------------------------------------------------------------------------------------
def _flip_pass(S, stats, max_batches=50):
    """dpo cell flipping (DetailedOrient::flipCells): a single-row cell is mirrored left to right (N <-> FN,
    FS <-> S) where its master allows it (LEF SYMMETRY Y) and the x HPWL of its costed nets goes down; its place
    does not change. Batches without shared nets, as the other passes."""
    d, dev = S.d, S.dev
    if 'm_sym' not in d.raw:
        return 0.0
    msym = torch.as_tensor(d.raw['m_sym'].astype('int64'), device=dev)
    cm = torch.as_tensor(d.raw['c_master'].astype('int64'), device=dev)
    c = S.cells
    o = d.orient[c].long()
    can = ((msym[cm[c].clamp(min=0)] & 2) != 0) & ((o == 0) | (o == 4) | (o == 5) | (o == 1))
    todo = c[can]
    gained, flips = 0.0, 0
    flipmap = torch.tensor([4, 5, 2, 3, 0, 1, 6, 7], device=dev)
    for _ in range(max_batches):
        if len(todo) == 0:
            break
        o = d.orient[todo].long()
        no = flipmap[o]
        x = d.pos[todo, 0].double()[:, None]
        y = d.pos[todo, 1].double()[:, None]
        new = S.cost(todo[:, None], x, y, no[:, None], xonly=True)
        old = S.cost(todo[:, None], x, y, o[:, None], xonly=True)
        gain = old - new
        good = gain > 1e-9
        if S.pdn_check:
            good &= S.pdn_ok(todo, x[:, 0], y[:, 0], no)
        g = torch.nonzero(good).flatten()
        if len(g) == 0:
            break
        win = S.independent(todo[g, None], gain[g])
        gm = g[win]
        S.apply(todo[gm], d.pos[todo[gm], 0].double(), d.pos[todo[gm], 1].double(), no[gm])
        gained += float(gain[gm].sum())
        flips += len(gm)
        todo = todo[g[~win]]
    stats['flip_cells'] = stats.get('flip_cells', 0) + flips
    return gained


@explain_oom('naive.detailed_place')
def detailed_place(d, ops=('gs', 'vs', 'ro', 'flip'), passes=10, tol=0.01, window=3, seed=1, verbose=True,
                   snapshot='naive_dp', pin_access=True):
    """Improve the wirelength of d's legal placement, keeping it legal (module docstring). ops: the dpo passes to run,
    in order ('gs' global swap, 'vs' vertical swap, 'ro' local reordering, 'flip' cell flipping: dpo's script order
    without mis and the random improver); each runs up to `passes` times and stops when a pass changes the HPWL by tol
    or less (dpo: -p 10 -t 0.005, the tolerance raised to 0.01 inside; flipping runs once, as in dpo). window: cells
    per reordering window (dpo: 2..4, default 3). seed: the vertical swap's random row choice (dpo: 1). pin_access
    (facts 'pdn' and 'tracks'): no move makes a signal pin unreachable past the PDN (Evaluator.pin_access);
    False: only OpenROAD dpl's PDN rule, as up to 1.0.7. Updates d.pos / d.orient and saves a snapshot; returns
    statistics."""
    d.require('detailed_place')
    from ..evaluator import Evaluator
    ev = Evaluator(d)
    L = ev.legality()
    if not L['legal']:
        raise RuntimeError('detailed_place needs a legal placement (%s): run naive.legalize first'
                           % {k: v for k, v in L.items() if v and k not in ('cells', 'legal')})
    t0 = time.time()
    W = min(4, max(2, int(window)))
    tol = max(float(tol), 0.01)
    S = _DP(d, pin_access=pin_access)
    gen = torch.Generator().manual_seed(int(seed))
    pos0 = d.pos.clone()
    h0 = float(ev.hpwl())
    stats = {'hpwl_before': h0, 'passes': {}}
    for op in ops:
        n_pass = 0
        for _ in range(passes):
            last = float(ev.hpwl())
            if op == 'ro':
                stats['ro_gain'] = stats.get('ro_gain', 0.0) + _reorder_pass(S, W, stats)
            elif op in ('gs', 'vs'):
                stats[op + '_gain'] = stats.get(op + '_gain', 0.0) + _global_swap_pass(S, stats, vertical=op == 'vs',
                                                                                     gen=gen)
            elif op == 'flip':
                stats['flip_gain'] = stats.get('flip_gain', 0.0) + _flip_pass(S, stats)
                n_pass += 1
                break
            else:
                raise ValueError('unknown detailed placement pass %r (known: gs, vs, ro, flip)' % op)
            n_pass += 1
            cur = float(ev.hpwl())
            if last == 0 or abs(cur - last) / last <= tol:
                break
        stats['passes'][op] = n_pass
    h1 = float(ev.hpwl())
    L = ev.legality()
    d.snapshot(snapshot)
    disp = ev.displacement(d.pos, pos0)
    stats.update({'hpwl_after': h1, 'hpwl_change_pct': 100 * (h1 / h0 - 1) if h0 else 0.0, 'legal': L['legal'],
                  'displacement': disp, 'seconds': time.time() - t0})
    if not L['legal']:
        raise RuntimeError('detailed placement left the placement illegal: %s (a bug)' % L)
    if verbose:
        print('  dp: HPWL %.6g -> %.6g (%+.2f%%), %d cells moved, mean displacement %.3f, %.2fs'
              % (h0, h1, stats['hpwl_change_pct'], disp['moved'], disp['mean'], stats['seconds']))
    return stats
