"""Naive legalization: tensorized, multi-row-height aware. Every cell is k = height / row height rows tall, and every
height class goes through the same path (k = 1 is the special case), tallest first.

  bands      for height k, a band is k consecutive base rows starting at a multiple of k (keeps even-height cells on
             one rail parity); its free segments are the intersection of its rows' free stretches, all in site units
  obstacles  fixed objects, then the cells of the classes already legalized
  per class  (1) vertical balancing: the chip is cut into columns; in each column the bands' capacities form one
                 line and the cells, ordered by y, are spread on it by isotonic regression (PAVA): no band of the
                 column overfull, least vertical movement;
             (2) each cell goes to the segment of its band nearest its x; overfull segments spill their farthest
                 cells to nearby segments (candidates over bands x columns, evaluated in parallel);
             (3) inside each segment, isotonic regression of x in GP order (the Abacus objective), rounded to sites.
Every step is a batched tensor operation; no loop over cells or rows. Not optimal in displacement (no cell
reordering, no detailed placement). Movable macros are not handled.
"""
import time

import numpy as np
import torch

from .. import rails as RL
from ..distill import ASIDE, MACRO, TERMINAL
from ..rails import base_rows as _base_rows
from .rowops import BIG, _first, _locate, _pack, _segment_broadcast, _segments
from .. import pdn as PDN
from .space import _Space, _fill, _improve


def _solve_class(gx, gy, w, NB, band_y, band_h, roff, seg_row, s0, s1, sw, col_cells, max_rounds, window, dev, stats):
    """Place one height class on its NB bands. gx: target x (global sites), gy: target bottom y, w: width (sites).
    Returns (band, x in band sites)."""
    n = len(gx)
    if n == 0:
        return (torch.zeros(0, dtype=torch.long, device=dev), torch.zeros(0, dtype=torch.float64, device=dev),
                torch.zeros(0, dtype=torch.bool, device=dev))
    gs0 = s0 + roff[seg_row]; gs1 = s1 + roff[seg_row]
    # (1) vertical balancing in columns
    Wc = float(max(16.0, col_cells * w.mean().item()))
    C = int((gs1.max() / Wc).ceil().item()) + 1
    c0 = (gs0 / Wc).floor().long(); c1 = ((gs1 - 1e-9) / Wc).floor().long()
    ncol = (c1 - c0 + 1).clamp(min=0)
    rep = torch.repeat_interleave(torch.arange(len(gs0), device=dev), ncol)
    col = c0[rep] + (torch.arange(len(rep), device=dev) - torch.repeat_interleave(torch.cumsum(ncol, 0) - ncol, ncol))
    ov = torch.minimum(gs1[rep], (col + 1).double() * Wc) - torch.maximum(gs0[rep], col.double() * Wc)
    cap = torch.zeros(NB * C, device=dev, dtype=torch.float64).index_add_(0, seg_row[rep] * C + col, ov.clamp(min=0)).view(NB, C) * 0.97
    cc = ((gx + w / 2) / Wc).floor().long().clamp(0, C - 1)
    # an overfull column keeps its cells: its band capacities are scaled up to the demand (the spill step then
    # moves the excess sideways) instead of the whole column being clamped to its first band
    demand = torch.zeros(C, device=dev, dtype=torch.float64).index_add_(0, cc, w)
    cap = cap * (demand / cap.sum(0).clamp(min=1e-9)).clamp(min=1.0)[None, :]
    cum = torch.cat([torch.zeros(1, C, device=dev, dtype=torch.float64), torch.cumsum(cap, 0)], 0)   # (NB+1, C)
    i = torch.searchsorted(band_y, gy).clamp(1, NB - 1) if NB > 1 else torch.zeros_like(gy, dtype=torch.long)
    if NB > 1:
        i = torch.where((gy - band_y[i - 1]).abs() <= (band_y[i] - gy).abs(), i - 1, i)
    rb = i.long()
    fr = ((gy - band_y[rb]) / band_h + 0.5).double().clamp(0, 1)
    V = cum[rb, cc] + fr * cap[rb, cc] - w / 2
    line = cum[NB]
    V = _pack(cc, V, w, torch.zeros(C, device=dev, dtype=torch.float64), line)
    key = (cum.T + torch.arange(C, device=dev, dtype=torch.float64)[:, None] * (line.max() + 1)).reshape(-1)
    band = (torch.searchsorted(key, V + w / 2 + cc.double() * (line.max() + 1), right=True) - 1 - cc * (NB + 1)).clamp(0, NB - 1)
    # (2) spread along each band: its segments joined end to end form one line of free sites (gaps skipped);
    # isotonic regression on that line, then back to (segment, x). A cell crossing a segment end goes to the next one.
    x = (gx - roff[band]).clamp(min=0)
    capS0 = s1 - s0
    S = len(seg_row)
    if S:
        fs = _first(seg_row)
        cs = torch.cumsum(capS0, 0) - capS0
        C0 = cs - _segment_broadcast(fs, cs)                       # free sites before each segment in its band
        bandcap = torch.zeros(NB, device=dev, dtype=torch.float64).index_add_(0, seg_row, capS0)
        key = seg_row.double() * BIG + s0
        j = (torch.searchsorted(key, band.double() * BIG + x, right=True) - 1).clamp(0, S - 1)
        firstseg = torch.searchsorted(key, band.double() * BIG, right=False).clamp(0, S - 1)
        j = torch.where(seg_row[j] == band, j, firstseg)
        has = seg_row[j] == band
        inside = x < s1[j]
        nxt = (j + 1).clamp(max=S - 1)
        nxt_ok = (seg_row[nxt] == band) & (nxt != j)
        to_next = ~inside & nxt_ok & ((s0[nxt] - x) < (x - s1[j]))
        U = torch.where(inside, C0[j] + (x - s0[j]).clamp(min=0),
                        torch.where(to_next, C0[nxt], C0[j] + capS0[j] - w))
        Up = _pack(band, U, w, torch.zeros(NB, device=dev, dtype=torch.float64), bandcap)
        key2 = seg_row.double() * BIG + C0
        j2 = (torch.searchsorted(key2, band.double() * BIG + Up, right=True) - 1).clamp(0, S - 1)
        j2 = torch.where(seg_row[j2] == band, j2, firstseg)
        xn = s0[j2] + (Up - C0[j2])
        cross = xn + w > s1[j2]
        n2 = (j2 + 1).clamp(max=S - 1)
        go_next = cross & (seg_row[n2] == band) & (n2 != j2)
        x = torch.where(has, torch.where(go_next, s0[n2], torch.minimum(xn, s1[j2] - w)), x)
    seg = _locate(seg_row, s0, s1, band, x, w)
    for _ in range(NB):
        miss = seg < 0
        if not bool(miss.any()):
            break
        band = torch.where(miss, (band + 1) % NB, band)
        x = torch.where(miss, (gx - roff[band]).clamp(min=0), x)
        seg = torch.where(miss, _locate(seg_row, s0, s1, band, x, w), seg)
    nofit = seg < 0                    # wider than every free segment of every band: left for the fill step
    if bool(nofit.any()):
        w = torch.where(nofit, torch.zeros_like(w), w)
        seg = seg.clamp(min=0)
    capS = s1 - s0
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        load = torch.zeros(S, device=dev, dtype=torch.float64).index_add_(0, seg, w)
        over = load - capS
        if not bool((over > 1e-9).any()):
            break
        far = (gx - roff[band] - (s0[seg] + s1[seg]) / 2).abs()
        o = torch.argsort(seg.double() * BIG - far)
        sg = seg[o]
        fst = _first(sg)
        cw = torch.cumsum(w[o], 0)
        before = cw - w[o] - _segment_broadcast(fst, cw - w[o])
        evict = torch.zeros_like(seg, dtype=torch.bool)
        evict[o] = (over[sg] > 1e-9) & (before < over[sg])
        ev = torch.nonzero(evict).flatten()
        span = window * (1 + rounds // 10)
        dr = torch.arange(-span, span + 1, device=dev)
        kx = 2 * (1 + rounds // 5)
        dxs = torch.arange(-kx, kx + 1, device=dev, dtype=torch.float64) * Wc
        cb = (band[ev][:, None, None] + dr[None, :, None]).clamp(0, NB - 1).expand(-1, -1, len(dxs)).reshape(len(ev), -1)
        n_ev, n_c = cb.shape
        cx = ((gx[ev][:, None, None] + dxs[None, None, :]).expand(-1, len(dr), -1).reshape(n_ev, -1) - roff[cb]).clamp(min=0)
        cs = _locate(seg_row, s0, s1, cb.reshape(-1), cx.reshape(-1), w[ev].repeat_interleave(n_c)).view(n_ev, n_c)
        csc = cs.clamp(min=0)
        spare = torch.where(cs >= 0, (capS - load)[csc], torch.full(cs.shape, -1.0, device=dev, dtype=torch.float64))
        fits = (spare >= w[ev][:, None]) & (cs != seg[ev][:, None])
        xin = torch.minimum(torch.maximum(cx, s0[csc]), s1[csc] - w[ev][:, None])
        dist = (band_y[cb] - gy[ev][:, None]).abs().double() + (xin + roff[cb] - gx[ev][:, None]).abs() * sw
        dist = torch.where(fits, dist, torch.full_like(dist, float('inf')))
        best = dist.argmin(1)
        bd = dist.gather(1, best[:, None]).squeeze(1)
        ok = torch.isfinite(bd)
        tgt = cs.gather(1, best[:, None]).squeeze(1)
        oo = torch.argsort(torch.where(ok, tgt, torch.full_like(tgt, S)).double() * BIG + torch.where(ok, bd, torch.zeros_like(bd)))
        tg = tgt[oo]; wv = w[ev][oo]
        fs = _first(tg)
        cumw = torch.cumsum(wv, 0); cumw = cumw - _segment_broadcast(fs, cumw - wv)
        accept = torch.zeros_like(ok); accept[oo] = cumw <= (capS - load)[tg.clamp(min=0)]
        go = ok & accept
        gi = ev[go]
        seg[gi] = tgt[go]; band[gi] = seg_row[tgt[go]]
        x[gi] = xin.gather(1, best[:, None]).squeeze(1)[go]
    stats['rounds_balance'] = max(stats.get('rounds_balance', 0), rounds)
    load = torch.zeros(S, device=dev, dtype=torch.float64).index_add_(0, seg, w)
    over = load - capS
    stats['overfull_segments'] = stats.get('overfull_segments', 0) + int((over > 1e-9).sum())
    # cells that still do not fit (the last ones in x of an overfull segment) are left for the fill step
    o = torch.argsort(seg.double() * BIG - x)
    sg = seg[o]
    fst = _first(sg)
    cw = torch.cumsum(w[o], 0)
    before = cw - w[o] - _segment_broadcast(fst, cw - w[o])
    out = torch.zeros_like(seg, dtype=torch.bool)
    out[o] = (over[sg] > 1e-9) & (before < over[sg])
    out |= nofit
    # (3) isotonic packing inside segments
    xo = torch.zeros_like(x)
    keep = ~out
    xo[keep] = _pack(seg[keep], x[keep], w[keep], s0, s1)
    return band, xo, out


def legalize(d, max_rounds=5, window=8, col_cells=8.0, improve=True, passes=3, verbose=True, snapshot='naive_lg',
             macro_halo=0.0):
    """Legalize the movable standard cells of d in place (d.pos, d.orient). Returns statistics."""
    d.require('legalize')
    t0 = time.time()
    dev = d.device
    hb, ys, rx0, rx1, sw, rorient = _base_rows(d)
    R = len(ys)
    nsites = ((rx1 - rx0) / sw).round()
    flags = d.flags
    if bool((d.movable & ((flags & MACRO) != 0)).any()):
        raise NotImplementedError('movable macros: Naive legalization handles standard cells only')
    mv = torch.nonzero(d.movable).flatten()
    pos0 = d.pos.clone()
    size = d.size
    k_all = (size[:, 1] / hb).round().clamp(min=1).long()
    gx0 = rx0.min()
    roff_rows = ((rx0 - gx0) / sw).round().double()
    pos = d.pos.clone()
    obst = torch.nonzero(~d.movable & ((flags & (TERMINAL | ASIDE)) == 0)).flatten()
    stats = {'classes': {}}

    # gaps between row segments (rows cut around macros) block like obstacles
    g_r, g_x0, g_x1 = RL.row_gaps(d, ys)
    gap_rows = g_r
    gap_a = ((g_x0 - rx0[g_r]) / sw + 1e-3).floor().double()
    gap_b = ((g_x1 - rx0[g_r]) / sw - 1e-3).ceil().double()

    def blocked_rows(ids):
        """(base row, [a, b) in row sites) of every base row each object crosses, plus the row gaps."""
        if len(ids) == 0:
            return gap_rows, gap_a, gap_b
        # macro_halo: fixed macros block a band (um) around them too
        hh = macro_halo * ((flags[ids] & MACRO) != 0).double() if macro_halo else torch.zeros(len(ids), dtype=torch.float64, device=dev)
        y0, y1 = pos[ids, 1] - hh, pos[ids, 1] + size[ids, 1] + hh
        rlo, rhi = RL.rows_spanned(ys, hb, y0, y1)        # every row the object overlaps, even partly
        rlo = rlo.clamp(max=R)
        n = (rhi - rlo).clamp(min=0)
        rep = torch.repeat_interleave(torch.arange(len(ids), device=dev), n)
        rows = rlo[rep] + (torch.arange(len(rep), device=dev) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n))
        i = ids[rep]
        # tolerant rounding: float32 positions on the site grid must not grow an obstacle by a site
        a = ((pos[i, 0] - hh[rep] - rx0[rows]) / sw + 1e-3).floor()
        b = ((pos[i, 0] + size[i, 0] + hh[rep] - rx0[rows]) / sw - 1e-3).ceil()
        return torch.cat([rows, gap_rows]), torch.cat([a.double(), gap_a]), torch.cat([b.double(), gap_b])

    # rail parity (OpenROAD dpl rules): cells are grouped by height and (bottom, top) rails; each group gets the
    # bands whose start rows it is power-compatible with
    RI = RL.info(d, (ys, rorient))
    k_mv = k_all[mv]
    sig_b = RI['cell_bot'][mv] if RI['known'] else torch.zeros_like(k_mv)
    sig_t = RI['cell_top'][mv] if RI['known'] else torch.zeros_like(k_mv)
    groups = torch.unique(torch.stack([k_mv, sig_b, sig_t], 1), dim=0).tolist()
    groups.sort(key=lambda g: -g[0])                                 # tallest first
    keep_ok = torch.zeros(len(mv), dtype=torch.bool, device=dev)
    flip_ok = torch.zeros(len(mv), dtype=torch.bool, device=dev)
    placed = torch.ones(len(mv), dtype=torch.bool, device=dev)
    for k, cb, ct in groups:
        sel = (k_mv == k) & (sig_b == cb) & (sig_t == ct)
        cls = mv[sel]
        kk = torch.tensor([k], device=dev)
        keep, flip = RL.compatible(torch.tensor([cb], device=dev), torch.tensor([ct], device=dev), kk,
                                   RI['row_bot'], RI['row_top'])
        allowed = (keep | flip)[0]
        # disjoint bands of k rows at the offset with the most compatible starts
        best = max(range(k), key=lambda o: int(allowed[o::k][:max(0, (R - o) // k)].sum())) if k > 1 else 0
        band_rows = torch.arange(best, R - k + 1, k, device=dev)
        band_rows = band_rows[allowed[band_rows]]
        NB = len(band_rows)
        if NB == 0:
            raise RuntimeError('no rows are power-compatible with %d-row cells (rails %s/%s)' % (k, cb, ct))
        band_y = ys[band_rows]
        roff_band = roff_rows[band_rows]
        row2band = torch.full((R,), -1, dtype=torch.long, device=dev)
        for j in range(k):
            row2band[band_rows + j] = torch.arange(NB, device=dev)
        # band blocks: the blocks of its k rows (shifted to the band's x origin); band extent: where all its rows exist
        br, ba, bb = blocked_rows(obst)
        bband = row2band[br]
        inb = bband >= 0
        bband = bband[inb]
        shift = roff_rows[br[inb]] - roff_band[bband]
        rows_k = band_rows[:, None] + torch.arange(k, device=dev)[None, :]          # (NB, k)
        lo_k = (roff_rows[rows_k] - roff_band[:, None]).max(1).values
        hi_k = (roff_rows[rows_k] + nsites[rows_k].double() - roff_band[:, None]).min(1).values
        nb_lines = torch.cat([bband, torch.arange(NB, device=dev)])
        nb_a = torch.cat([ba[inb] + shift, torch.zeros(NB, device=dev, dtype=torch.float64)])
        nb_b = torch.cat([bb[inb] + shift, lo_k])
        seg_row, s0, s1 = _segments(nb_lines, nb_a, nb_b, NB, hi_k)
        gx = ((pos0[cls, 0] - gx0) / sw).double()
        wsites = (size[cls, 0] / sw).ceil().double()
        st = {}
        band, xb, unfit = _solve_class(gx, pos0[cls, 1], wsites, NB, band_y, hb * k, roff_band, seg_row, s0, s1, sw,
                                       col_cells, max_rounds, window, dev, st)
        pos[cls, 0] = (rx0[band_rows[band]] + xb * sw).float()
        pos[cls, 1] = band_y[band]
        placed[sel] = ~unfit
        obst = torch.cat([obst, cls[~unfit]])                       # cells that did not fit are not obstacles yet
        st['unfit'] = int(unfit.sum())
        stats['classes']['%dx(%d,%d)' % (k, cb, ct) if RI['known'] else int(k)] = {'cells': len(cls), 'bands': NB, **st}

    # rows each cell's rails allow (any start row when rails are unknown)
    # (G, R) table per (height, rails) group plus the group of each cell: not a cells x rows matrix (bigblue4 would
    # need 5.8 GB for that)
    gtab = torch.zeros(len(groups), R, dtype=torch.bool, device=dev)
    gid = torch.zeros(len(mv), dtype=torch.long, device=dev)
    for gi, (k, cb, ct) in enumerate(groups):
        sel = (k_mv == k) & (sig_b == cb) & (sig_t == ct)
        kp, fp = RL.compatible(torch.tensor([cb], device=dev), torch.tensor([ct], device=dev), torch.tensor([k], device=dev),
                               RI['row_bot'], RI['row_top'])
        gtab[gi] = (kp | fp)[0]
        gid[sel] = gi
    allowed_rows = (gtab, gid)
    fixed = torch.nonzero(~d.movable & ((flags & (TERMINAL | ASIDE)) == 0)).flatten()
    space = _Space(ys, rx0, nsites, roff_rows, sw, hb, blocked_rows(fixed), dev)
    wg = (size[mv, 0] / sw).round().long()
    gxs = ((pos0[mv, 0] - gx0) / sw).double()
    gy = pos0[mv, 1]
    cur_x = ((pos[mv, 0] - gx0) / sw).round().long()
    cur_r = torch.searchsorted(ys, pos[mv, 1] - 0.25 * hb).clamp(max=R - 1)
    # cells that did not fit their class: nearest free spot anywhere (window grows until found)
    if not bool(placed.all()):
        cur_x, cur_r, placed = _fill(space, cur_x, cur_r, placed, gxs, gy, k_mv, wg, allowed_rows, stats=stats)
    if improve:
        for rnd in range(passes):
            cur_x, cur_r = _improve(space, cur_x, cur_r, gxs, gy, k_mv, wg, allowed_rows, stats=stats)
            if rnd == passes - 1:
                break
            # re-pack the single-row cells of every row segment towards their GP x (rows kept, order may change)
            one = k_mv == 1
            if bool(one.any()):
                pos[mv, 0] = (gx0 + cur_x.double() * sw).float(); pos[mv, 1] = ys[cur_r]
                br, ba, bb = blocked_rows(torch.cat([fixed, mv[~one]]))
                seg_row, s0, s1 = _segments(br, ba, bb, R, nsites)
                c1 = mv[one]; r1 = cur_r[one]
                x1 = ((pos[c1, 0] - rx0[r1]) / sw).round().double()
                w1 = (size[c1, 0] / sw).ceil().double()
                sg = _locate(seg_row, s0, s1, r1, x1, w1)
                if bool((sg < 0).any()):
                    continue                                          # never pack a cell outside its segment
                px = _pack(sg, ((pos0[c1, 0] - rx0[r1]) / sw).double(), w1, s0, s1)
                cur_x[one] = (px + roff_rows[r1]).round().long()
    pos[mv, 0] = (gx0 + cur_x.double() * sw).float()
    pos[mv, 1] = ys[cur_r]
    start_row = cur_r
    for k, cb, ct in groups:
        sel = (k_mv == k) & (sig_b == cb) & (sig_t == ct)
        kp, fp = RL.compatible(torch.tensor([cb], device=dev), torch.tensor([ct], device=dev), torch.tensor([k], device=dev),
                               RI['row_bot'], RI['row_top'])
        keep_ok[sel] = kp[0, cur_r[sel]]; flip_ok[sel] = fp[0, cur_r[sel]]

    # orientation: the row's when either way fits (or rails unknown), otherwise the one the rails need
    orient = d.orient.clone()
    if d.row_orient is not None:
        row_o = rorient[start_row]
        o = torch.where(keep_ok & ~flip_ok, torch.zeros_like(row_o), row_o)
        o = torch.where(flip_ok & ~keep_ok, torch.full_like(row_o, 5), o)
        orient[mv] = o if RI['known'] else torch.where(k_mv % 2 == 1, row_o, orient[mv])
    d.pos = pos
    d.orient = orient
    d.update_orient()
    if 'pdn' in d.facts:            # signal pins clear of the PDN's via metal (fact 'pdn'): a GPU row DP, then the few left
        stats['pdn_dp'] = _pdn_dp(d, space, mv, cur_x, cur_r, k_mv, wg, gx0, sw)
        stats['pdn'] = _pdn_repair(d, space, mv, cur_x, cur_r, k_mv, wg, allowed_rows, rorient, ys, gx0, sw, placed)
    d.snapshot(snapshot)
    pos = d.pos
    disp = (pos[mv] - pos0[mv]).abs().sum(1)
    if not bool(placed.all()):
        bad = mv[~placed]
        kinds = sorted(set(zip(k_mv[~placed].tolist(), (size[bad, 0] / sw).round().long().tolist(),
                               sig_b[~placed].tolist(), sig_t[~placed].tolist())))
        raise RuntimeError('%d cells found no free spot (rows x sites, rails): %s; e.g. %s'
                           % (len(bad), kinds[:6], [d.names['cells'][i] for i in bad[:3].tolist()]))
    stats.update({'seconds': time.time() - t0, 'mean_disp': disp.mean().item(), 'max_disp': disp.max().item(),
                  'cells': len(mv), 'overfull_segments': sum(c.get('overfull_segments', 0) for c in stats['classes'].values())})
    if verbose:
        print('  lg: %d cells in height classes %s, mean displacement %.3f, max %.3f, %.2fs'
              % (len(mv), {k: v['cells'] for k, v in stats['classes'].items()}, stats['mean_disp'], stats['max_disp'],
                 stats['seconds']))
    return stats



W_PA = 20.0                    # pin-access penalty of the PDN-aware DP: a cell with a signal pin on a PDN pin blockage


def _pdn_dp(d, space, mv, cur_x, cur_r, k_mv, wg, gx0, sw, S=8, w_pa=W_PA, w_disp=0.05, w_mirror=1e-3):
    """Signal pins clear of the PDN (attar.pdn) by a row-wise dynamic program on the GPU, after the PDN-aware detailed
    placement refinement of pa_pdn (Xplace pinaccess/dp_refine.py): every single-row movable cell may shift by
    -S..S sites and be mirrored left to right (N <-> FN, FS <-> S: the same rails; only where the LEF SYMMETRY allows
    it), at a cost

        c_i(s, o) = w_pa [a signal pin overlaps a pin blockage] + w_disp |s| + w_mirror [mirrored],

    and the cells of every chain (consecutive single-row cells between fixed objects, multi-row cells and row ends)
    keep their order without overlap: min sum_i c_i(s_i, o_i) s.t. x_{i-1} + w_{i-1} + s_{i-1} <= x_i + s_i, solved
    exactly, all chains at once, one step per position in the chain (prefix minima: no pairwise term). Updates
    d.pos, d.orient and cur_x; returns counts. Cells it cannot clear are left to _pdn_repair."""
    dev = d.device
    index = PDN.Index(d, dev)
    if index.n == 0:
        return {'cells': 0}
    t0 = time.time()
    R = space.R
    xl, rl, kl, wl = cur_x.tolist(), cur_r.tolist(), k_mv.tolist(), wg.tolist()
    # rows: [start, end, mv index of a single-row cell or -1]: every other object is an obstacle
    rows = [[] for _ in range(R)]
    for j in range(len(xl)):
        for t in range(kl[j]):
            rows[rl[j] + t].append((xl[j], xl[j] + wl[j], j if kl[j] == 1 else -1))
    for r, a, b in zip(space.fr.tolist(), space.fa.tolist(), space.fb.tolist()):
        rows[r].append((a, b, -1))
    lo_r, hi_r = space.ext_lo.tolist(), space.ext_hi.tolist()
    chains = []                                          # (lo, hi, [mv indices]) in sites
    for r in range(R):
        items = sorted(rows[r])
        lo, cur = lo_r[r], []
        for a, b, j in items:
            if j < 0:
                if cur:
                    chains.append((lo, min(a, hi_r[r]), cur)); cur = []
                lo = max(lo, b)
            else:
                cur.append(j)
        if cur:
            chains.append((lo, hi_r[r], cur))
    if not chains:
        return {'cells': 0}
    cells = [j for _, _, c in chains for j in c]
    n = len(cells)
    K = 2 * S + 1
    sv = torch.arange(-S, S + 1, device=dev, dtype=torch.float64)
    jt = torch.tensor(cells, device=dev)
    ci = mv[jt]                                          # cell ids
    orient = d.orient.clone()
    o0 = orient[ci]
    mirror = torch.tensor([4, 5, 2, 3, 0, 1, 6, 7], device=dev)[o0]           # N<->FN, S<->FS
    mirror = torch.where((o0 == 0) | (o0 == 1) | (o0 == 4) | (o0 == 5), mirror, o0)
    msym = torch.as_tensor(d.raw['m_sym'].astype('int64'), device=dev)[torch.as_tensor(d.raw['c_master'].astype('int64'), device=dev)[ci].clamp(min=0)] \
        if 'm_sym' in d.raw else torch.zeros(n, dtype=torch.long, device=dev)
    can_mirror = ((msym & 2) != 0) & (mirror != o0)
    # costs (n, K, 2): displacement, mirroring, blocked pins
    C = (w_disp * sv.abs()).view(1, K, 1).expand(n, K, 2).clone()
    C[:, :, 1] += w_mirror
    C[:, :, 1] = torch.where(can_mirror.view(-1, 1), C[:, :, 1], torch.full_like(C[:, :, 1], float('inf')))
    rel, layer = PDN.pin_shapes(d, dev)
    order = torch.argsort(d.p_cell)
    lo_pin = torch.searchsorted(d.p_cell[order], torch.arange(len(d.pos) + 1, device=dev))
    cnt = lo_pin[ci + 1] - lo_pin[ci]
    owner = torch.repeat_interleave(torch.arange(n, device=dev), cnt)
    pins = order[torch.arange(int(cnt.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt) + lo_pin[ci][owner]]
    keep = layer[pins] >= 0
    pins, owner = pins[keep], owner[keep]
    # only the cells near a blockage: the box of the cell grown by S sites, on its pins' layers
    c_ = ci[owner]
    x0 = d.pos[c_, 0].double() - S * float(sw); x1 = d.pos[c_, 0].double() + d.size[c_, 0].double() + S * float(sw)
    y0 = d.pos[c_, 1].double(); y1 = y0 + d.size[c_, 1].double()
    near = index.hits(torch.stack([x0, y0, x1, y1], 1), layer[pins])
    sel = torch.zeros(n, dtype=torch.bool, device=dev); sel[owner[near]] = True
    pk = sel[owner]
    pins, owner = pins[pk], owner[pk]
    blocked = torch.zeros(n, K, 2, dtype=torch.bool, device=dev)
    if len(pins):
        c_ = ci[owner]
        w, h = d.size[c_, 0].double(), d.size[c_, 1].double()
        for o in (0, 1):
            oo = o0[owner] if o == 0 else mirror[owner]
            for b0 in range(0, K, 1):
                bx = d.pos[c_, 0].double() + sv[b0] * float(sw)
                pb = PDN.place_boxes(rel[pins], w, h, oo, bx, d.pos[c_, 1].double())
                hit = index.hits(pb, layer[pins])
                blocked[:, b0, o].index_put_((owner[hit],), torch.ones(int(hit.sum()), dtype=torch.bool, device=dev), accumulate=False)
    C += w_pa * blocked.double()
    n_blocked0 = int(blocked[:, S, 0].sum())
    # the DP, chains packed longest first: step j runs on the chains longer than j (a prefix of that order)
    lens = [len(c) for _, _, c in chains]
    nc = len(chains); maxl = max(lens)
    rank = sorted(range(nc), key=lambda q: -lens[q])
    act = [0] * (maxl + 1)
    for q in rank:
        act[lens[q] - 1] += 1
    for j in range(maxl - 2, -1, -1):
        act[j] += act[j + 1]
    off = [0] * (maxl + 1)
    for j in range(maxl):
        off[j + 1] = off[j] + act[j]
    pos_of = {}
    k_ = 0
    for q, (_, _, c) in enumerate(chains):
        for t_, j in enumerate(c):
            pos_of[(q, t_)] = k_; k_ += 1                  # index into cells / C
    fc = [0] * off[maxl]; fg = [0.0] * off[maxl]
    lo_g = [0.0] * nc; hi_g = [0.0] * nc
    for r_, q in enumerate(rank):
        lo, hi, c = chains[q]
        for t_, j in enumerate(c):
            f = off[t_] + r_
            fc[f] = pos_of[(q, t_)]
            fg[f] = float(xl[j] - (xl[c[t_ - 1]] + wl[c[t_ - 1]])) if t_ else 0.0
        lo_g[r_] = float(xl[c[0]] - lo); hi_g[r_] = float(hi - xl[c[-1]] - wl[c[-1]])
    fc = torch.tensor(fc, device=dev); fg = torch.tensor(fg, device=dev, dtype=torch.float64)
    lo_g = torch.tensor(lo_g, device=dev, dtype=torch.float64); hi_g = torch.tensor(hi_g, device=dev, dtype=torch.float64)
    INF = float('inf')
    ptr = torch.empty(off[maxl], K, dtype=torch.long, device=dev)        # best (s', o') of the previous cell, per s
    Dfin = torch.empty(nc, K, 2, dtype=torch.float64, device=dev)
    D = None
    ar = torch.arange(K, device=dev)
    for j in range(maxl):
        A, f0 = act[j], off[j]
        Cj = C[fc[f0:f0 + A]]
        if j == 0:
            Dn = torch.where((sv.view(1, K) >= -lo_g.view(-1, 1)).unsqueeze(2), Cj, torch.full_like(Cj, INF))
        else:
            D = D[:A]
            Dmin, oarg = D.min(dim=2)                                       # (A, K') over o'
            cm, carg = torch.cummin(Dmin, dim=1)                            # best s' <= index
            g = fg[f0:f0 + A].round().long()
            idx = (ar.view(1, K) + g.view(-1, 1))                           # s' <= s + g
            ok = idx >= 0
            idx = idx.clamp(0, K - 1)
            M = torch.where(ok, cm.gather(1, idx), torch.full((A, K), INF, dtype=torch.float64, device=dev))
            sp = carg.gather(1, idx)
            ptr[f0:f0 + A] = sp * 2 + oarg.gather(1, sp)
            Dn = Cj + M.unsqueeze(2)
        e0 = act[j + 1]
        if e0 < A:                                                          # chains ending here: the right bound
            Dn[e0:] = torch.where((sv.view(1, K) > hi_g[e0:A].view(-1, 1)).unsqueeze(2), torch.full_like(Dn[e0:], INF), Dn[e0:])
            Dfin[e0:A] = Dn[e0:]
        D = Dn
    best, state = Dfin.view(nc, 2 * K).min(dim=1)
    feasible = torch.isfinite(best)
    sel_state = torch.empty(off[maxl], dtype=torch.long, device=dev)
    for j in range(maxl - 1, -1, -1):
        A, f0 = act[j], off[j]
        sel_state[f0:f0 + A] = state[:A]
        if j:
            state[:A] = ptr[f0:f0 + A].gather(1, (state[:A] // 2).unsqueeze(1)).squeeze(1)
    rank_of_f = torch.empty(off[maxl], dtype=torch.long, device=dev)
    for j in range(maxl):
        rank_of_f[off[j]:off[j] + act[j]] = torch.arange(act[j], device=dev)
    okf = feasible[rank_of_f]
    s_new = torch.where(okf, sel_state // 2, torch.full_like(sel_state, S)) - S
    o_new = torch.where(okf, sel_state % 2, torch.zeros_like(sel_state))
    shift = torch.zeros(n, dtype=torch.long, device=dev); mir = torch.zeros(n, dtype=torch.long, device=dev)
    shift[fc] = s_new; mir[fc] = o_new
    after_blocked = int(blocked[torch.arange(n, device=dev), shift + S, mir].sum())
    pos = d.pos.clone()
    pos[ci, 0] = pos[ci, 0] + shift.to(pos.dtype) * float(sw)
    orient[ci] = torch.where(mir == 1, mirror, o0)
    cur_x[jt] = cur_x[jt] + shift
    d.pos = pos; d.orient = orient; d.update_orient()
    # then every cell still blocked (multi-row ones included) mirrored where it stands, all at once: the place does
    # not change, so neither does legality; it stays mirrored where that clears it
    conf = PDN.conflicts(d, index=index) & d.movable
    cc = torch.nonzero(conf).flatten()
    n_mir2 = 0
    if len(cc):
        oc = d.orient[cc]
        mc = torch.tensor([4, 5, 2, 3, 0, 1, 6, 7], device=dev)[oc]
        msym_c = torch.as_tensor(d.raw['m_sym'].astype('int64'), device=dev)[torch.as_tensor(d.raw['c_master'].astype('int64'), device=dev)[cc].clamp(min=0)] \
            if 'm_sym' in d.raw else torch.zeros(len(cc), dtype=torch.long, device=dev)
        okm = ((msym_c & 2) != 0) & ((oc == 0) | (oc == 1) | (oc == 4) | (oc == 5))
        cnt = lo_pin[cc + 1] - lo_pin[cc]
        own = torch.repeat_interleave(torch.arange(len(cc), device=dev), cnt)
        pp = order[torch.arange(int(cnt.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt) + lo_pin[cc][own]]
        kp = layer[pp] >= 0
        pp, own = pp[kp], own[kp]
        c_ = cc[own]
        pb = PDN.place_boxes(rel[pp], d.size[c_, 0].double(), d.size[c_, 1].double(), mc[own], d.pos[c_, 0].double(), d.pos[c_, 1].double())
        hit = index.hits(pb, layer[pp])
        still = torch.zeros(len(cc), dtype=torch.bool, device=dev); still[own[hit]] = True
        flip = okm & ~still
        orient = d.orient.clone(); orient[cc[flip]] = mc[flip]
        d.orient = orient; d.update_orient()
        n_mir2 = int(flip.sum())
    return {'cells': n, 'near_pdn': int(sel.sum()), 'blocked_before': n_blocked0, 'blocked_after': after_blocked,
            'shifted': int((shift != 0).sum()), 'mirrored': int((mir == 1).sum()), 'infeasible_chains': int((~feasible).sum()),
            'mirrored_in_place': n_mir2, 'seconds': round(time.time() - t0, 2)}


MIRROR_Y = {0: 4, 4: 0, 5: 1, 1: 5}       # N <-> FN, FS <-> S: the same rails, pins mirrored left to right


def _pdn_repair(d, space, mv, cur_x, cur_r, k_mv, wg, allowed_rows, rorient, ys, gx0, sw, placed, max_shift=48, max_window=256):
    """Cells whose signal pins overlap PDN pin blockages (attar.pdn) move off them, keeping legality:
    1. a shift along their row by the fewest sites, pushing the single-row neighbors it runs into until the row's
       whitespace absorbs it (nothing fixed, multi-row or outside the row segment is pushed); every moved cell must
       be clear of the blockages;
    2. otherwise the nearest free spot in rows of the same orientation that the rails allow (the window doubles).
    Updates d.pos; returns counts."""
    dev = d.device
    index = PDN.Index(d, dev)
    if index.n == 0:
        return {'conflicts': 0}
    rel, layer = PDN.pin_shapes(d, dev)
    conf = PDN.conflicts(d, index=index)[mv]
    n0 = int(conf.sum())
    if n0 == 0:
        return {'conflicts': 0}
    R, S = space.R, space.S
    order = torch.argsort(d.p_cell)
    cell_sorted = d.p_cell[order]
    lo_pin = torch.searchsorted(cell_sorted, torch.arange(len(d.pos) + 1, device=dev))
    orient = d.orient
    pos = d.pos.clone()
    pos_np = pos.double().cpu().numpy(); gx0f, swf = float(gx0), float(sw)
    cx_ = cur_x.tolist(); cr_ = cur_r.tolist(); kk = k_mv.tolist(); ww = wg.tolist(); mvl = mv.tolist()

    rel_np, lay_np = rel.cpu().numpy(), layer.cpu().numpy()
    size_np = d.size.double().cpu().numpy(); orient_np = orient.cpu().numpy().copy()
    order_np, lo_np = order.cpu().numpy(), lo_pin.cpu().numpy()
    cache = {}

    def pins_of(c):
        """(master-relative boxes (n, 4), layers (n,)) of cell c's signal pins, cached."""
        if c not in cache:
            p = order_np[lo_np[c]:lo_np[c + 1]]
            p = p[lay_np[p] >= 0]
            cache[c] = (rel_np[p], lay_np[p])
        return cache[c]

    def boxes_at(c, x, o=None):
        """Cell c's signal pin boxes with its lower-left at site x, in its row, orientation o (default: its own)."""
        rb, _ = cache[c] if c in cache else pins_of(c)
        w, h = size_np[c]; o = orient_np[c] if o is None else o
        x0, y0, x1, y1 = rb[:, 0], rb[:, 1], rb[:, 2], rb[:, 3]
        if o in (1, 4):
            x0, x1 = w - x1, w - x0
        if o in (1, 5):
            y0, y1 = h - y1, h - y0
        X, Y = gx0f + x * swf, float(pos_np[c, 1])
        return np.stack([X + x0, Y + y0, X + x1, Y + y1], 1)

    # per base row: the objects in it, sorted: [start, end, mv index of a cell, or -1 for anything fixed]
    rows = [[] for _ in range(R)]
    for j in range(len(mvl)):
        for t in range(kk[j]):
            rows[cr_[j] + t].append([cx_[j], cx_[j] + ww[j], j])
    for r, a, b in zip(space.fr.tolist(), space.fa.tolist(), space.fb.tolist()):
        rows[r].append([a, b, -1])
    ext_lo, ext_hi = space.ext_lo.tolist(), space.ext_hi.tolist()
    # site occupancy (rows x sites): every cell, fixed object, row gap and out-of-row site; each move is checked
    # against it before it is made, so a move can never overlap anything or leave the rows
    occ = (space.prefix(cur_x, cur_r, k_mv, wg)).diff(dim=1).cpu().numpy().astype(np.int16)

    def footprint(moved, xs, sign):
        for m, x in zip(moved, xs):
            for r in range(cr_[m], cr_[m] + kk[m]):
                occ[r, x:x + ww[m]] += sign

    def fits(moved, xs):
        """Would the moved cells, at xs, sit on free sites only (nothing else, not each other, inside the rows)?"""
        old = [cx_[m] for m in moved]
        footprint(moved, old, -1)
        ok = all(0 <= x and x + ww[m] <= S for m, x in zip(moved, xs))
        if ok:
            footprint(moved, xs, 1)
            ok = all(occ[r, x:x + ww[m]].max() == 1 for m, x in zip(moved, xs) for r in range(cr_[m], cr_[m] + kk[m]))
            footprint(moved, xs, -1)
        footprint(moved, old, 1)
        return ok

    def clear_many(cands):
        """Indices, in order, of the candidates (moved, xs, orientation of the first moved cell or None) whose moved
        cells are all clear of the blockages."""
        boxes, lays, owner = [], [], []
        for i, (moved, xs, o0) in enumerate(cands):
            for n, (m, x) in enumerate(zip(moved, xs)):
                c = mvl[m]
                rb, lb = pins_of(c)
                if len(rb):
                    boxes.append(boxes_at(c, x, o0 if n == 0 else None)); lays.append(lb); owner.append(np.full(len(rb), i))
        bad = np.zeros(len(cands), bool)
        if boxes:
            hit = index.hits(torch.as_tensor(np.concatenate(boxes), device=dev), torch.as_tensor(np.concatenate(lays), device=dev))
            bad[np.concatenate(owner)[hit.cpu().numpy()]] = True
        return np.nonzero(~bad)[0].tolist()

    for r in range(R):
        rows[r].append([-10**9, ext_lo[r], -1]); rows[r].append([ext_hi[r], 10**9, -1])
        rows[r].sort()
    slot = {}                                            # (mv index, row) -> position in rows[row]

    def reslot(r):
        for i, it in enumerate(rows[r]):
            if it[2] >= 0:
                slot[(it[2], r)] = i
    for r in range(R):
        reslot(r)

    def shift(j, delta, limit=64):
        """(moved mv indices, their new starts) when cell j moves by delta sites along its rows and every cell it
        runs into is pushed along (in all of that cell's rows, transitively); None if a push reaches anything fixed or
        the row's end, or more than `limit` cells would move."""
        new = {j: cx_[j] + delta}
        work = [j]
        while work:
            m = work.pop()
            a, b = new[m], new[m] + ww[m]
            for r in range(cr_[m], cr_[m] + kk[m]):
                row = rows[r]; i = slot[(m, r)]
                nbrs = row[i + 1:] if delta > 0 else reversed(row[:i])
                for it in nbrs:
                    o = it[2]
                    oa = new.get(o, it[0]) if o >= 0 else it[0]
                    ob = oa + (it[1] - it[0])
                    if (delta > 0 and oa >= b) or (delta < 0 and ob <= a):
                        break
                    if o < 0:
                        return None
                    want = b if delta > 0 else a - (it[1] - it[0])
                    if o not in new or (want > new[o] if delta > 0 else want < new[o]):
                        new[o] = want; work.append(o)
                        if len(new) > limit:
                            return None
        moved = list(new)
        return moved, [new[m] for m in moved]

    shifted = farther = mirrored = 0; failed = []
    new_orient = d.orient.clone()
    m_sym = (d.raw['m_sym'][d.raw['c_master'].clip(0)] if 'm_sym' in d.raw else np.zeros(len(d.pos), np.uint8))
    todo = torch.nonzero(conf).flatten().tolist()
    # a second pass for the few the first could not move: shifts 4x as far, pushing up to 4x as many cells
    for ms, lim, ncand in ((max_shift, 64, 24), (4 * max_shift, 256, 96)):
        failed_j = []
        for j in todo:
            done = False
            c = mvl[j]
            om = MIRROR_Y.get(int(orient_np[c]))
            if om is not None and not (m_sym[c] & 2):        # the master may not be mirrored about the y axis
                om = None
            cands = [([j], [cx_[j]], om)] if om is not None else []      # mirrored where it is, then shifts, fewest sites
            for mag in range(1, ms + 1):
                for delta in (mag, -mag):
                    s_ = shift(j, delta, lim)
                    if s_ is not None:
                        cands.append(s_ + (None,))
                        if om is not None:
                            cands.append(s_ + (om,))
                if len(cands) >= ncand:
                    break
            g = next((i for i in (clear_many(cands) if cands else []) if fits(*cands[i][:2])), None)   # occupancy: the guard
            if g is not None:
                moved, xs, o0 = cands[g]
                footprint(moved, [cx_[m] for m in moved], -1); footprint(moved, xs, 1)
                for m, x in zip(moved, xs):
                    for r in range(cr_[m], cr_[m] + kk[m]):
                        it = rows[r][slot[(m, r)]]; it[0], it[1] = x, x + ww[m]
                    cx_[m] = x; pos[mvl[m], 0] = gx0 + x * sw
                if o0 is not None:
                    orient_np[c] = o0; new_orient[c] = o0; mirrored += 1
                done = True; shifted += 1
            if not done:                                         # 2. a free spot elsewhere
                found = _free_clear_spot(d, j, mvl, cx_, cr_, kk, ww, rows, allowed_rows, rorient, ys, gx0, sw, R, S, pos,
                                         pins_of, rel, layer, orient, index, max_window * (ms // max_shift),
                                         orients=[int(orient_np[c])] + ([om] if om is not None else []))
                if found is not None:                     # occupancy check, with the cell in its new rows
                    r, x, o_new = found
                    for t in range(kk[j]):
                        occ[cr_[j] + t, cx_[j]:cx_[j] + ww[j]] -= 1
                    if all(not occ[r + t, x:x + ww[j]].any() for t in range(kk[j])):
                        for t in range(kk[j]):
                            occ[r + t, x:x + ww[j]] += 1
                    else:
                        for t in range(kk[j]):
                            occ[cr_[j] + t, cx_[j]:cx_[j] + ww[j]] += 1
                        found = None
                if found is None:
                    failed_j.append(j)
                else:
                    r, x, o_new = found
                    if o_new != int(orient_np[c]):
                        orient_np[c] = o_new; new_orient[c] = o_new; mirrored += 1
                    for t in range(kk[j]):                       # leave the old rows, enter the new ones
                        ro = cr_[j] + t
                        rows[ro] = [it for it in rows[ro] if it[2] != j]; reslot(ro)
                    for t in range(kk[j]):
                        rows[r + t].append([x, x + ww[j], j]); rows[r + t].sort(); reslot(r + t)
                    cx_[j] = x; cr_[j] = r; pos[mvl[j], 0] = gx0 + x * sw; pos[mvl[j], 1] = ys[r]; farther += 1
        todo = failed_j
        if not todo:
            break
    failed = [mvl[j] for j in todo]
    cur_x[:] = torch.tensor(cx_, device=dev); cur_r[:] = torch.tensor(cr_, device=dev)
    d.pos = pos
    d.orient = new_orient
    d.update_orient()
    left = int(PDN.conflicts(d, index=index)[mv].sum())
    return {'conflicts': n0, 'shifted': shifted, 'mirrored': mirrored, 'moved_farther': farther, 'no_spot': len(failed),
            'left': left}


def _free_clear_spot(d, j, mvl, cx_, cr_, kk, ww, rows, allowed_rows, rorient, ys, gx0, sw, R, S, pos, pins_of, rel,
                     layer, orient, index, max_window, orients=None):
    """(row, start site, orientation) of the nearest free spot for cell j in rows of its orientation that its rails
    allow, where its pins are clear of the blockages in one of `orients` (default: its own; tried in order); None if
    none within max_window sites."""
    dev = d.device
    gtab, gid = allowed_rows
    c = mvl[j]; k, w, x0, r0 = kk[j], ww[j], cx_[j], cr_[j]
    ok_rows = (gtab[gid[j]] & (rorient == rorient[r0])).clone()
    ok_rows[R - k + 1:] = False
    rb, lb = pins_of(c)
    rb_t = torch.as_tensor(rb, device=dev); lb_t = torch.as_tensor(lb, device=dev)
    win = 8
    while win <= max_window:
        cands = []
        for r in range(max(0, r0 - win // 8), min(R - k + 1, r0 + win // 8 + 1)):
            if not bool(ok_rows[r]):
                continue
            # free intervals common to rows r .. r+k-1 (the cell's own footprint counts as free)
            frees = None
            for t in range(k):
                its = [it for it in rows[r + t] if it[2] != j]
                f, last = [], -10**9
                for a, b, _ in its:
                    if a > last:
                        f.append((last, a))
                    last = max(last, b)
                frees = f if frees is None else [(max(a, c_), min(b, e)) for a, b in frees for c_, e in f if max(a, c_) < min(b, e)]
            for a, b in frees or []:
                lo_, hi_ = max(a, x0 - win), min(b - w, x0 + win)
                for x in range(lo_, hi_ + 1):
                    cands.append((abs(x - x0) * sw + abs(float(ys[r]) - float(ys[r0])), r, x))
        cands.sort()
        for b0 in range(0, len(cands), 256):
            batch = cands[b0:b0 + 256]
            if len(rb) == 0:
                return batch[0][1], batch[0][2], int(orient[c])
            m, n = len(batch), len(rb)
            bx = torch.tensor([gx0 + x * sw for _, _, x in batch], device=dev, dtype=torch.float64)
            by = torch.tensor([float(ys[r]) for _, r, _ in batch], device=dev, dtype=torch.float64)
            best = None
            for o in (orients or [int(orient[c])]):
                pb = PDN.place_boxes(rb_t.repeat(m, 1), d.size[c, 0].double(), d.size[c, 1].double(),
                                     torch.full((m * n,), o, device=dev, dtype=orient.dtype),
                                     bx.repeat_interleave(n), by.repeat_interleave(n))
                hit = index.hits(pb, lb_t.repeat(m)).view(m, n).any(1)
                g = torch.nonzero(~hit).flatten()
                if len(g) and (best is None or int(g[0]) < best[0]):
                    best = (int(g[0]), o)
            if best is not None:
                _, r, x = batch[best[0]]
                return r, x, best[1]
        win *= 2
    return None
