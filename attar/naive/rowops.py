"""Row primitives for legalization, all batched tensor operations on 1D lines (rows, bands) in site units:
segments of free sites, locating a cell's segment, and least-displacement packing (parallel PAVA)."""
import torch

BIG = 1e7   # segment / row offset for segmented scans; values stay exact in float64


def _first(v):
    f = torch.ones_like(v, dtype=torch.bool)
    f[1:] = v[1:] != v[:-1]
    return f


def _segment_broadcast(first, v):
    """For each element, the value v at the first element of its run (runs are contiguous)."""
    idx = torch.arange(len(v), device=v.device)
    return v[torch.cummax(torch.where(first, idx, torch.zeros_like(idx)), 0).values]


def _segment_total(first, w):
    idx = torch.arange(len(w), device=w.device)
    start = torch.cummax(torch.where(first, idx, torch.zeros_like(idx)), 0).values
    cs = torch.cumsum(w, 0)
    last = torch.ones_like(first); last[:-1] = first[1:]
    end = torch.flip(torch.cummin(torch.flip(torch.where(last, idx, torch.full_like(idx, len(w))), [0]), 0).values, [0])
    return cs[end] - (cs[start] - w[start])


def _segments(rows_of, a, b, R, nsites):
    """Free segments of R lines (site units [0, nsites[r])) given blocked intervals (line, [a, b)), sorted by (line, s0)."""
    dev = nsites.device
    o = torch.argsort(rows_of.double() * BIG + a.double())
    r, a, b = rows_of[o], a[o].double(), b[o].double()
    off = r.double() * BIG
    run = torch.cummax(off + b, 0).values - off                       # merged blocked end so far, per line
    first = _first(r)
    prev_end = torch.cat([torch.zeros(1, device=dev, dtype=run.dtype), run[:-1]])
    prev_end = torch.where(first, torch.zeros_like(prev_end), prev_end)
    last = torch.ones_like(first); last[:-1] = r[1:] != r[:-1]
    has = torch.zeros(R, dtype=torch.bool, device=dev); has[r] = True
    free = torch.nonzero(~has).flatten()
    sr = torch.cat([r, r[last], free])
    s0 = torch.cat([prev_end, run[last], torch.zeros(len(free), device=dev, dtype=torch.float64)]).clamp(min=0)
    s1 = torch.cat([a, nsites[r[last]].double(), nsites[free].double()])
    s1 = torch.minimum(s1, nsites[sr].double())
    keep = s1 > s0
    sr, s0, s1 = sr[keep], s0[keep], s1[keep]
    o = torch.argsort(sr.double() * BIG + s0)
    return sr[o], s0[o], s1[o]


def _locate(seg_row, seg_s0, seg_s1, row, x, w):
    """For each cell: the segment of line `row` containing [x, x+w), else the nearer neighbour that fits (or -1)."""
    if len(seg_row) == 0:
        return torch.full_like(row, -1)
    key = seg_row.double() * BIG + seg_s0
    i = torch.searchsorted(key, row.double() * BIG + x, right=True) - 1
    S = len(seg_row)
    cand = torch.stack([i.clamp(0, S - 1), (i + 1).clamp(0, S - 1)], 1)
    same = seg_row[cand] == row[:, None]
    lo = seg_s0[cand]; hi = seg_s1[cand] - w[:, None]
    dist = torch.where(hi >= lo, (lo - x[:, None]).clamp(min=0) + (x[:, None] - hi).clamp(min=0),
                       torch.full_like(lo, float('inf')))
    dist = torch.where(same, dist, torch.full_like(dist, float('inf')))
    best = dist.argmin(1)
    seg = cand.gather(1, best[:, None]).squeeze(1)
    ok = torch.isfinite(dist.gather(1, best[:, None]).squeeze(1))
    return torch.where(ok, seg, torch.full_like(seg, -1))


def _pack(seg, x, w, seg_s0, seg_s1, max_rounds=10000):
    """Inside each segment, in order of x, the positions closest to x (least squares) without overlap and inside the
    segment: isotonic regression of z = x - (widths before), by a parallel pool-adjacent-violators (each round merges
    all adjacent violating blocks), clamped to the segment, rounded to whole sites (monotone, so still legal)."""
    if len(seg) == 0:
        return x.double()
    o = torch.argsort(seg.double() * BIG + x.double())
    sg, xs, ws = seg[o], x[o].double(), w[o].double()
    lo = seg_s0[sg]; hi = seg_s1[sg]
    first = _first(sg)
    cw = torch.cumsum(ws, 0)
    pre = cw - ws - _segment_broadcast(first, cw - ws)
    tot = _segment_total(first, ws)
    y = xs - pre
    start = torch.ones_like(first)                                    # every element starts as its own block
    for _ in range(max_rounds):
        blk = torch.cumsum(start.long(), 0) - 1
        nb = int(blk[-1].item()) + 1
        ssum = torch.zeros(nb, device=y.device, dtype=y.dtype).index_add_(0, blk, y)
        cnt = torch.zeros(nb, device=y.device, dtype=y.dtype).index_add_(0, blk, torch.ones_like(y))
        m = (ssum / cnt)[blk]
        viol = torch.zeros_like(start)
        viol[1:] = start[1:] & ~first[1:] & (m[:-1] > m[1:] + 1e-9)
        if not bool(viol.any()):
            break
        start = start & ~viol
    z = torch.minimum(torch.maximum(m.round(), lo), hi - tot)
    out = torch.empty(len(sg), device=x.device, dtype=torch.float64)
    out[o] = z + pre
    return out
