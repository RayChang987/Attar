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

import torch

from .. import rails as RL
from ..distill import ASIDE, MACRO, TERMINAL
from ..rails import base_rows as _base_rows
from .rowops import BIG, _first, _locate, _pack, _segment_broadcast, _segments
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


def legalize(d, max_rounds=5, window=8, col_cells=8.0, improve=True, passes=3, verbose=True, snapshot='naive_lg'):
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
        y0, y1 = pos[ids, 1], pos[ids, 1] + size[ids, 1]
        rlo, rhi = RL.rows_spanned(ys, hb, y0, y1)        # every row the object overlaps, even partly
        rlo = rlo.clamp(max=R)
        n = (rhi - rlo).clamp(min=0)
        rep = torch.repeat_interleave(torch.arange(len(ids), device=dev), n)
        rows = rlo[rep] + (torch.arange(len(rep), device=dev) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n))
        i = ids[rep]
        # tolerant rounding: float32 positions on the site grid must not grow an obstacle by a site
        a = ((pos[i, 0] - rx0[rows]) / sw + 1e-3).floor()
        b = ((pos[i, 0] + size[i, 0] - rx0[rows]) / sw - 1e-3).ceil()
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
    d.snapshot(snapshot)
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
