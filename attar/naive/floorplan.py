"""Naive floorplanning engines: die sizing, IO pin placement and macro placement, tensorized.

make_die      A die for a target utilization and aspect ratio: core = std-cell area / utilization + macro area, on
              the site / row grid of the design's rows (same site, row height, row-orientation pattern and
              die-to-core margins), rows filling the core.
place_io      The die boundary minus the corners becomes one line (bottom, right, top, left). Each IO pin targets the
              point of that line nearest the centroid of the cells it connects to (with a placement), or its slot in
              netlist-name order (without). Isotonic packing on the line (the legalizer's parallel PAVA) spreads the
              pins onto slots `pitch` apart with the least movement; slots map back to (x, y).
place_macros  Mixed-size global placement (macros move with the cells, attar.naive.gp), then macro legalization on
              the macro grid (Design.macro_grid: positions that put the macro pins on their routing tracks), with a
              halo between macros and to the core edge. Legalizers: 'sweep' (sweep line from the center out, keeps
              the GP order), 'push' (all pairs at once, pushed apart in parallel), 'auto' (both, the better one).
              Macros still too close go, largest first, to the nearest free spot of an occupancy grid (2D prefix
              sums give every free position at once). Then frozen. legalizer='lp': constraint graphs + LP
              (attar/naive/macro_lp.py).
bstar         (place_macros(packing=True)) B*-tree packing + annealing, block-level designs: attar/naive/bstar.py.
All write their answer into the Design (floorplan.die / rows / floorplan.io / floorplan.macros become facts).
attar/naive/flow.py runs them in order (floorplan) and places the cells (place).
"""
import math
import time

import torch

from ..distill import ASIDE, MACRO, PHYSICAL, TERMINAL
from ..kernels import overlapping as _overlapping
from .rowops import _pack


def _perimeter(die, margin):
    """The four sides minus the corners as one line: per side (origin, direction, length) and cumulative offsets."""
    x0, y0, x1, y1 = die
    W, H = x1 - x0, y1 - y0
    sides = [((x0 + margin, y0), (1.0, 0.0), W - 2 * margin),     # bottom, left to right
             ((x1, y0 + margin), (0.0, 1.0), H - 2 * margin),     # right, bottom to top
             ((x1 - margin, y1), (-1.0, 0.0), W - 2 * margin),    # top, right to left
             ((x0, y1 - margin), (0.0, -1.0), H - 2 * margin)]    # left, top to bottom
    off = [0.0]
    for s in sides:
        off.append(off[-1] + max(s[2], 0.0))
    return sides, off


def place_io(d, pitch=None, margin=None, mode='auto', verbose=True):
    """Place the IO pins of d on the die boundary. mode: 'auto' (net-aware when the cells are placed, else in name
    order), 'nets', 'order'. pitch: pin spacing (default 4 sites, shrunk if the boundary is too short)."""
    d.require('io_place')
    t0 = time.time()
    dev = d.device
    io = d.io
    T = len(io)
    if T == 0:
        d.facts['floorplan.io'] = 'naive:place_io (no IO pins)'
        return {'pins': 0}
    die = [float(v) for v in d.die_box[0]]
    sw = float(d.row_site_w.min()) if d.row_site_w is not None else 1.0
    pitch = pitch or 4 * sw
    margin = margin if margin is not None else max(4 * pitch, 0.01 * min(die[2] - die[0], die[3] - die[1]))
    sides, off = _perimeter(die, margin)
    L = off[-1]
    if T * pitch > L:                                   # too many pins for the pitch: tighten it
        pitch = L / (T * 1.02)

    use_nets = (mode == 'nets') or (mode == 'auto' and 'placement' in d.present)
    if use_nets:
        # centroid of the other (non-IO) pins on each IO pin's nets
        term = (d.flags[d.p_cell] & TERMINAL) != 0
        pp = d.pin_pos()
        E = len(d.n_weight)
        s = torch.zeros(E, 2, device=dev).index_add_(0, d.p_net[~term], pp[~term])
        c = torch.zeros(E, device=dev).index_add_(0, d.p_net[~term], torch.ones(int((~term).sum()), device=dev))
        net_c = s / c.clamp(min=1)[:, None]
        has = c > 0
        tp = torch.nonzero(term).flatten()
        cell_of = d.p_cell[tp]
        io_idx = torch.full((len(d.flags),), -1, dtype=torch.long, device=dev); io_idx[io] = torch.arange(T, device=dev)
        k = io_idx[cell_of]
        nets = d.p_net[tp]
        ok = has[nets] & (k >= 0)
        tgt = torch.zeros(T, 2, device=dev).index_add_(0, k[ok], net_c[nets[ok]])
        cnt = torch.zeros(T, device=dev).index_add_(0, k[ok], torch.ones(int(ok.sum()), device=dev))
        center = torch.tensor([(die[0] + die[2]) / 2, (die[1] + die[3]) / 2], device=dev)
        tgt = torch.where((cnt > 0)[:, None], tgt / cnt.clamp(min=1)[:, None], center)
        # nearest side, coordinate along it
        dist = torch.stack([tgt[:, 1] - die[1], die[2] - tgt[:, 0], die[3] - tgt[:, 1], tgt[:, 0] - die[0]], 1)
        side = dist.argmin(1)
        along = torch.stack([tgt[:, 0] - (die[0] + margin), tgt[:, 1] - (die[1] + margin),
                             (die[2] - margin) - tgt[:, 0], (die[3] - margin) - tgt[:, 1]], 1).gather(1, side[:, None]).squeeze(1)
        lens = torch.tensor([s_[2] for s_ in sides], device=dev)
        U = torch.tensor(off[:4], device=dev)[side] + along.clamp(min=0).minimum(lens[side])
    else:
        # netlist-name order (keeps buses together), evenly spaced around the boundary
        names = d.names['cells']
        order = sorted(range(T), key=lambda i: names[int(io[i])])
        rank = torch.empty(T, dtype=torch.long, device=dev)
        rank[torch.tensor(order, device=dev)] = torch.arange(T, device=dev)
        U = (rank.double() + 0.5) * (L / T)

    # isotonic packing on the line, in slots of `pitch`
    slots = math.floor(L / pitch)
    u = _pack(torch.zeros(T, dtype=torch.long, device=dev), (U.double() / pitch - 0.5),
              torch.ones(T, dtype=torch.float64, device=dev),
              torch.zeros(1, dtype=torch.float64, device=dev), torch.full((1,), float(slots), dtype=torch.float64, device=dev))
    pos_line = (u + 0.5) * pitch
    offs = torch.tensor(off, device=dev, dtype=torch.float64)
    side = (torch.searchsorted(offs, pos_line, right=True) - 1).clamp(0, 3)
    along = pos_line - offs[side]
    org = torch.tensor([s_[0] for s_ in sides], device=dev, dtype=torch.float64)
    dirv = torch.tensor([s_[1] for s_ in sides], device=dev, dtype=torch.float64)
    xy = org[side] + dirv[side] * along[:, None]
    moved = (xy.float() - d.pos[io]).abs().sum(1) if bool(d.placed[io].all()) else None
    d.set_io(xy.float(), 'naive:place_io(%s)' % ('nets' if use_nets else 'order'))
    res = {'pins': T, 'pitch': pitch, 'mode': 'nets' if use_nets else 'order', 'seconds': time.time() - t0,
           'mean_move': moved.mean().item() if moved is not None else None}
    if verbose:
        print('  io: %d pins on the boundary (%s), pitch %.3g, %.3fs' % (T, res['mode'], pitch, res['seconds']))
    return res


def _legalize_macros(pos, size, movable, lo, hi, halo, iters=400):
    """Remove macro overlaps (with a halo) by parallel pairwise pushes; fixed macros do not move. pos: lower-left."""
    M = len(pos)
    p = pos.clone()
    eye = torch.eye(M, dtype=torch.bool, device=pos.device)
    for it in range(iters):
        c = p + size / 2
        dc = c[:, None, :] - c[None, :, :]                        # (M, M, 2) i - j
        ov = (size[:, None, :] + size[None, :, :]) / 2 + halo - dc.abs()
        over = (ov > 1e-6).all(2) & ~eye
        if not bool(over.any()):
            return p, it
        ax = (ov[..., 0] <= ov[..., 1])                           # push along the smaller overlap
        sgn = torch.where(dc >= 0, torch.ones_like(dc), -torch.ones_like(dc))
        tie = (dc == 0)
        sgn = torch.where(tie, torch.where(torch.arange(M, device=pos.device)[:, None, None] < torch.arange(M, device=pos.device)[None, :, None],
                                           -torch.ones_like(dc), torch.ones_like(dc)), sgn)
        push = torch.zeros(M, M, 2, device=pos.device)
        amount = torch.where(over, torch.where(ax, ov[..., 0], ov[..., 1]), torch.zeros_like(ov[..., 0]))
        # a fixed neighbour takes no share: the movable one moves the whole overlap
        share = torch.where(movable[None, :], 0.5, 1.0) * movable[:, None]
        push[..., 0] = torch.where(ax, amount * share * sgn[..., 0], torch.zeros_like(amount))
        push[..., 1] = torch.where(~ax, amount * share * sgn[..., 1], torch.zeros_like(amount))
        p = p + push.sum(1) * 0.6
        p = torch.maximum(torch.minimum(p, hi - size), lo)
        p = torch.where(movable[:, None], p, pos)
    return p, iters


def _sweep_1d(p, size, lo, hi, halo, gx, gy, gx0, max_bins=4096):
    """Sweep-line (Tetris-style) macro legalization along x, from the core's vertical center line outwards.
    Macros are taken by distance from the center line; the right half moves only right, the left half only left.
    The placed macros form one blob per y bin, kept as two contours: its left and right edges (FL, FR, one value
    per bin of height gy). For a macro, every y start is evaluated at once: a sliding-window max / min of the
    contours (max_pool1d) gives how far it must move out at that y to clear the blob; the start with the least
    displacement (x push + y move) inside the core wins. x snaps to the site grid (gx from gx0), y to the bins.
    Returns positions and the macros that fit nowhere (left at their position)."""
    dev = p.device
    yb0 = float(lo[1]); H = float(hi[1]) - yb0
    gy = gy * max(1, math.ceil(H / gy / max_bins))
    NY = max(1, int(H // gy + 1e-6))
    FR = torch.full((NY,), -float('inf'), device=dev)
    FL = torch.full((NY,), float('inf'), device=dev)
    # exact occupancy (bins of gy x gxc): a macro whose GP spot is free at some y stays there (holes are used)
    xb0, W = float(lo[0]), float(hi[0]) - float(lo[0])
    gxc = gx * max(1, math.ceil(W / gx / max_bins))
    NX = max(1, int(W // gxc + 1e-6) + 1)
    occ = torch.zeros(NY, NX, device=dev)
    mid = (float(lo[0]) + float(hi[0])) / 2
    cx = p[:, 0] + size[:, 0] / 2
    order = torch.argsort((cx - mid).abs()).tolist()
    out = p.clone(); failed = []
    hb = math.ceil(halo / gy - 1e-6) if halo > 0 else 0
    ys = yb0 + torch.arange(NY, device=dev, dtype=torch.float32) * gy
    snap_d = lambda v: gx0 + torch.floor((v - gx0) / gx + 1e-4) * gx
    snap_u = lambda v: gx0 + torch.ceil((v - gx0) / gx - 1e-4) * gx
    for i in order:
        w, h = float(size[i, 0]), float(size[i, 1])
        kh = max(1, math.ceil(h / gy - 1e-6))
        S = NY - kh + 1
        if S <= 0:
            failed.append(i); continue
        right = float(cx[i]) >= mid
        x0 = gx0 + round((float(p[i, 0]) - gx0) / gx) * gx
        win_r = torch.nn.functional.max_pool1d(FR[None, None], kh, 1)[0, 0]          # (S,) max right edge
        win_l = -torch.nn.functional.max_pool1d(-FL[None, None], kh, 1)[0, 0]       # (S,) min left edge
        hit = (x0 < win_r) & (x0 + w > win_l)
        c0 = max(0, min(NX, math.floor((x0 - halo - xb0) / gxc + 1e-4)))
        c1 = max(c0, min(NX, math.ceil((x0 + w + halo - xb0) / gxc - 1e-4)))
        rs = torch.cat([torch.zeros(1, device=dev), occ[:, c0:c1].sum(1).cumsum(0)])
        hit = hit & ((rs[kh:] - rs[:-kh]) > 0)                                       # really blocked at that y
        if right:
            x = torch.where(hit, snap_u(win_r), torch.full_like(win_r, x0))
            ok = x + w <= float(hi[0]) + 1e-4
        else:
            x = torch.where(hit, snap_d(win_l - w), torch.full_like(win_l, x0))
            ok = x >= float(lo[0]) - 1e-4
        cost = (x - float(p[i, 0])).abs() + (ys[:S] - float(p[i, 1])).abs()
        cost = torch.where(ok, cost, torch.full_like(cost, float('inf')))
        k = int(cost.argmin())
        if not bool(torch.isfinite(cost[k])):
            failed.append(i); continue
        xi, yi = float(x[k]), float(ys[k])
        out[i, 0], out[i, 1] = xi, yi
        a, b = max(0, k - hb), min(NY, k + kh + hb)
        FR[a:b] = torch.clamp(FR[a:b], min=xi + w + halo)
        FL[a:b] = torch.clamp(FL[a:b], max=xi - halo)
        occ[a:b, max(0, math.floor((xi - xb0) / gxc + 1e-4)):max(0, math.ceil((xi + w - xb0) / gxc - 1e-4))] = 1
    return out, failed


def _sweep_legalize(p, size, lo, hi, halo, sw, hb, x0, y0):
    """Sweep along x and along y (the transposed problem); keep the one with fewer misses, then less movement."""
    px, fx = _sweep_1d(p, size, lo, hi, halo, sw, hb, x0)
    T = lambda t: t.flip(-1)
    py, fy = _sweep_1d(T(p), T(size), T(lo), T(hi), halo, hb, sw, y0)
    py = T(py)
    dx, dy = (px - p).abs().sum(), (py - p).abs().sum()
    if (len(fx), float(dx)) <= (len(fy), float(dy)):
        return px, fx, 'x'
    return py, fy, 'y'


def _grid_place(p, size, todo, lo, hi, halo, gx, gy, max_cells=2048):
    """Move the macros in `todo` (largest first) to the nearest free spot of an occupancy grid over the region
    [lo, hi] (lo on the grid) with pitch (gx, gy) (multiples of the macro grid, so the result stays on it). The
    other macros, grown by the halo, are occupied. Returns positions and the macros that found no spot."""
    dev = p.device
    cx0, cy0 = float(lo[0]), float(lo[1]); cx1, cy1 = float(hi[0]), float(hi[1])
    gx = gx * max(1, math.ceil((cx1 - cx0) / gx / max_cells))
    gy = gy * max(1, math.ceil((cy1 - cy0) / gy / max_cells))
    NX, NY = max(0, int((cx1 - cx0) // gx + 1e-6)), max(0, int((cy1 - cy0) // gy + 1e-6))
    occ = torch.zeros(NY, NX, device=dev, dtype=torch.int32)
    e = 1e-4

    def mark(i, v):
        x0 = max(0, math.floor((float(p[i, 0]) - halo - cx0) / gx + e)); x1 = min(NX, math.ceil((float(p[i, 0] + size[i, 0]) + halo - cx0) / gx - e))
        y0 = max(0, math.floor((float(p[i, 1]) - halo - cy0) / gy + e)); y1 = min(NY, math.ceil((float(p[i, 1] + size[i, 1]) + halo - cy0) / gy - e))
        if x1 > x0 and y1 > y0:
            occ[y0:y1, x0:x1] += v
    p = p.clone()
    for i in torch.nonzero(~todo).flatten().tolist():
        mark(i, 1)
    order = torch.nonzero(todo).flatten()
    order = order[torch.argsort(-size[order].prod(1))].tolist()
    jj = torch.arange(NX, device=dev, dtype=torch.float32) * gx + cx0
    ii = torch.arange(NY, device=dev, dtype=torch.float32) * gy + cy0
    failed = []
    for i in order:
        kw, kh = math.ceil(float(size[i, 0]) / gx - e), math.ceil(float(size[i, 1]) / gy - e)
        if kw > NX or kh > NY:
            failed.append(i); continue
        I = torch.zeros(NY + 1, NX + 1, device=dev, dtype=torch.int32)
        I[1:, 1:] = occ.cumsum(0).cumsum(1)
        win = I[kh:, kw:] - I[:-kh, kw:] - I[kh:, :-kw] + I[:-kh, :-kw]          # (NY-kh+1, NX-kw+1)
        free = win == 0
        if not bool(free.any()):
            failed.append(i); continue
        dx = (jj[:NX - kw + 1] - p[i, 0]).abs(); dy = (ii[:NY - kh + 1] - p[i, 1]).abs()
        dist = torch.where(free, dy[:, None] + dx[None, :], torch.full_like(win, float('inf'), dtype=torch.float32))
        k = int(dist.argmin())
        r, c = divmod(k, NX - kw + 1)
        p[i, 0] = cx0 + c * gx; p[i, 1] = cy0 + r * gy
        mark(i, 1)
    return p, failed


def place_macros(d, halo=None, target_density=0.9, legalizer='auto', packing=False, packing_args=None, verbose=True,
                 **gp_args):
    """Place the macros of d: mixed-size global placement, then macro legalization (halo between macros and to the
    core edge) on the macro grid (Design.macro_grid: pins on their routing tracks), frozen. The standard cells keep their mixed-size positions; run
    attar.naive.global_place afterwards for a placement around the fixed macros.
    legalizer: 'sweep' (sweep line from the center out, keeps the GP order), 'push' (parallel pairwise push),
    'lp' (the pair directions of the better of sweep / push, then positions by linear programming, as close to the
    GP as the constraint graphs allow: attar/naive/macro_lp.py), 'auto' (= 'lp').
    'lp' never leaves overlaps; it fits the macros into the core when it can, and otherwise lets them stick out by as
    little as it found (res['fits_outline'], res['outside_core']). sweep and push stay in the core and may leave
    overlaps (res['overlapping_pairs']).
    packing=True: no GP; the macros are packed by a B*-tree and simulated annealing (attar/naive/bstar.py): into the
    core first, then for HPWL. Only for block-level designs (no rows, nothing movable but the macros). Never
    overlapping; sticks out of the core when it cannot fit. packing_args: moves (per macro and phase, default 200),
    seed."""
    if packing:
        return _pack_macros(d, halo=halo, verbose=verbose, **(packing_args or {}))
    d.require('macro_place')
    from .gp import global_place
    t0 = time.time()
    dev = d.device
    mac = d.macros
    if len(mac) == 0:
        d.fix_macros('naive:place_macros (no macros)')
        return {'macros': 0}
    if 'floorplan.macros' in d.facts:                 # already placed: re-place them
        d.release_macros()
    mixed = d.movable.clone()
    gp = global_place(d, target_density=target_density, verbose=False, movable=mixed, **gp_args)
    mixed_pos = d.pos[d.macros].clone()                 # where the mixed-size GP put the macros
    hb = float(d.row_h.min()) if d.row_h is not None else 1.0 / d.dbu
    sw = float(d.row_site_w.min()) if d.row_site_w is not None else 1.0 / d.dbu
    if halo is None:
        # a channel between macros, and to the core edge, for SRAM-like macros among standard cells; none for
        # block-level floorplans (no rows, or blocks only a few rows tall), where it would eat the white space
        small = float(d.size[mac].min(1).values.median()) < 10 * hb
        halo = 0.0 if d.row_h is None or small else max(2 * hb, 4 * sw)
    core = d.core_box[0]
    size = d.size[mac]
    movable_m = torch.ones(len(mac), dtype=torch.bool, device=dev)
    # the macro grid (Design.macro_grid): legal lower-left corners are off + k * step, which puts the macro pins on
    # their routing tracks. Change of variables: q = p - off lives on the plain grid k * step, for a box grown by off
    # on its low side (less than one step), so every legalizer below works on one common grid.
    step, off = d.macro_grid(mac)
    stx, sty = float(step[0]), float(step[1])
    size_g, gp_g = size + off, mixed_pos - off
    first = lambda v, st: math.ceil(v / st - 1e-6) * st                     # first grid point >= v

    def bounds(h):
        """lower-left of q (on the grid) and the bound for q + size_g: the halo also to the core edge"""
        lo = torch.tensor([first(float(core[0]) + h, stx), first(float(core[1]) + h, sty)], device=dev)
        return lo, core[2:] - h

    def snap(q, lo, hi):
        st = step.to(q.dtype)
        q = (q / st).round() * st
        top = torch.floor((hi - size_g) / st + 1e-6) * st
        return torch.minimum(torch.maximum(q, lo), top)

    def legalize(method, h):
        """-> (positions, label, iterations, macros sent to the grid, macros with no spot), for a halo h."""
        it = 0
        lo, hi = bounds(h)
        if method == 'sweep':
            q, _, axis = _sweep_legalize(gp_g.float(), size_g, lo, hi, h, stx, sty, 0.0, 0.0)
            label = 'sweep-%s' % axis
        elif method == 'lp':
            # directions from the better of sweep / push (+ free grid spots); positions by LP, towards the GP;
            # never overlapping, in the core if the constraint graphs could be made to fit
            base = {m: legalize(m, h) for m in ('sweep', 'push')}
            bm = min(base, key=lambda m: score(base[m][0], h))
            from .macro_lp import lp_legalize
            q, info = lp_legalize(gp_g, size_g, lo, hi, h, stx, sty, q_dir=base[bm][0] - off)
            fit_info[h] = info
            label = 'lp on %s (%d flips, %d LP rounds, %.2fs%s)' % (
                base[bm][1], info['flips'], info['lp_rounds'], info['seconds'],
                '' if info['fits'] else '; does not fit the core: %s, sticks out by %.4g x %.4g' % ((info['why'],) + info['overflow']))
            return q + off, label, 0, 0, []
        elif method == 'push':
            q, it = _legalize_macros(gp_g, size_g, movable_m, lo, hi, h)
            q = snap(q, lo, hi)               # then undo what snapping re-created
            q, it2 = _legalize_macros(q, size_g, movable_m, lo, hi, h + max(stx, sty))
            q = snap(q, lo, hi); it += it2; label = 'push'
        else:
            raise ValueError('legalizer: auto, sweep, push or lp, not %r' % method)
        bad = _overlapping(q + off, size, h - 1e-3).any(1)
        failed = []
        if bool(bad.any()):                   # dense: nearest free spot on the grid, largest first
            q, failed = _grid_place(q, size_g, bad, lo, hi, h, stx, sty)
        return q + off, label, it, int(bad.sum()), failed

    def score(p, h):
        """(pairs closer than the halo, HPWL with the macros there and the cells where the mixed-size GP put them)"""
        q = d.pos.clone(); q[mac] = p
        return int(_overlapping(p, size, h - 1e-3).sum()) // 2, float(d.hpwl(q))

    methods = ('lp',) if legalizer == 'auto' else (legalizer,)
    fit_info = {}
    # a halo larger than the base channel (given by the caller) is halved until every pair keeps it (and, for 'lp',
    # until the macros fit the core)
    base = min(halo, max(2 * hb, 4 * sw)) if d.row_h is not None else halo
    h = halo
    while True:
        tried = {m: legalize(m, h) for m in methods}
        best = min(tried, key=lambda m: score(tried[m][0], h))
        ok = score(tried[best][0], h)[0] == 0 and (best != 'lp' or fit_info[h]['fits'])
        if ok or h <= base + 1e-9:
            break
        h = max(base, h / 2)
    res_halo = h
    p, how, it, moved, failed = tried[best]
    if len(tried) > 1:
        how = '%s (auto: %s)' % (how, ', '.join('%s %d close pairs, HPWL %.4g' % ((k,) + score(v[0], h)) for k, v in tried.items()))
    it2 = 0
    pos = d.pos.clone(); pos[mac] = p
    d._pos = pos
    outside = int(((p < core[:2] - 1e-3) | (p + size > core[2:] + 1e-3)).any(1).sum())
    d.fix_macros('naive:place_macros' + (' (%d stick out of the core)' % outside if outside else ''))
    ov = _overlapping(p, size)
    res = {'macros': len(mac), 'legalizer': how, 'halo': res_halo, 'gp_iterations': gp['iterations'], 'legalize_iterations': it + it2,
           'displacement': float((p - mixed_pos).abs().sum(1).mean()),
           'grid': d.macro_grid_source, 'grid_placed': moved, 'no_spot': len(failed), 'overlapping_pairs': int(ov.sum()) // 2,
           'outside_core': outside, 'fits_outline': outside == 0, 'seconds': time.time() - t0}
    if verbose:
        print('  macros: %d placed (mixed-size GP %d it, %s legalization, mean move %.3g%s), %d overlapping pairs, '
              '%s, %.2fs'
              % (len(mac), gp['iterations'], how, res['displacement'], ', %d to free grid spots' % moved if moved else '',
                 res['overlapping_pairs'], '%d outside the core' % outside if outside else 'all in the core', res['seconds']))
    return res


def _pack_macros(d, halo=None, moves=200, seed=0, verbose=True):
    """place_macros(packing=True): B*-tree packing of the macros in the core (attar.naive.bstar)."""
    import numpy as np
    from .bstar import anneal
    if 'rows' in d.facts:
        raise ValueError("place_macros(packing=True) is for block-level floorplans; %s has rows (%s): pack without "
                         "packing (GP + LP legalization)" % (d.manifest.get('design'), d.facts['rows']))
    d.require('macro_place')
    t0 = time.time()
    mac = d.macros
    if len(mac) == 0:
        d.fix_macros('naive:place_macros (no macros)')
        return {'macros': 0}
    if 'floorplan.macros' in d.facts:
        d.release_macros()
    others = d.movable.clone(); others[mac] = False
    if bool(others.any()):
        raise ValueError('place_macros(packing=True): %d movable cells are not macros; packing places macros only'
                         % int(others.sum()))
    h = float(halo or 0.0)
    core = d.core_box[0].tolist()
    step, off = d.macro_grid(mac)
    st = step.double().cpu().numpy(); offn = off.double().cpu().numpy()
    size = d.size[mac].double().cpu().numpy()
    size_g = size + offn
    first = lambda v, s: math.ceil(v / s - 1e-6) * s
    lo = np.array([first(core[0] + h, st[0]), first(core[1] + h, st[1])])
    hi = np.array(core[2:]) - h
    k = np.ceil((size_g + h) / st - 1e-6).astype(np.int64)
    ub = np.floor((hi - lo - size_g) / st + 1e-6).astype(np.int64)
    width = int(np.floor((hi[0] - lo[0]) / st[0] + 1e-6))
    # HPWL on the CPU: pins of the nets that count, in net order; macro pins move with their macro, the rest are fixed
    keep = d.n_mask[d.p_net]
    p_net = d.p_net[keep].cpu().numpy(); p_cell = d.p_cell[keep]
    loc = torch.full((len(d.size),), -1, dtype=torch.long, device=d.device); loc[mac] = torch.arange(len(mac), device=d.device)
    blk = loc[p_cell].cpu().numpy()
    rel = (d.size[p_cell] / 2 + d.pin_off[keep]).double().cpu().numpy()        # from the cell's lower-left
    fixed = (d.pos[p_cell].double().cpu().numpy() + rel)
    starts = np.flatnonzero(np.r_[True, p_net[1:] != p_net[:-1]])
    w = d.n_weight.double().cpu().numpy()[p_net[starts]]
    mv = blk >= 0; b = blk[mv]; relm = rel[mv]

    def lower_left(x, y):
        return np.stack([lo[0] + x * st[0], lo[1] + y * st[1]], 1) + offn

    def hpwl(x, y):
        p = lower_left(x, y)
        pin = fixed.copy()
        pin[mv] = p[b] + relm
        span = (np.maximum.reduceat(pin, starts) - np.minimum.reduceat(pin, starts)).sum(1)
        return float((span * w).sum())

    x, y, info = anneal(k[:, 0], k[:, 1], ub[:, 0], ub[:, 1], width, hpwl=hpwl, moves=moves, seed=seed, verbose=verbose)
    p = torch.as_tensor(lower_left(x, y), device=d.device, dtype=d.pos.dtype)
    pos = d.pos.clone(); pos[mac] = p
    d._pos = pos
    sz = d.size[mac]
    cb = d.core_box[0]
    outside = int(((p < cb[:2] - 1e-3) | (p + sz > cb[2:] + 1e-3)).any(1).sum())
    d.fix_macros('naive:place_macros(packing)' + (' (%d stick out of the core)' % outside if outside else ''))
    res = {'macros': len(mac), 'legalizer': 'b*-tree packing', 'halo': h, 'moves': info['moves'],
           'overlapping_pairs': int(_overlapping(p, sz).sum()) // 2, 'outside_core': outside,
           'fits_outline': outside == 0, 'hpwl': float(d.hpwl(d.pos)), 'seconds': time.time() - t0}
    if verbose:
        print('  macros: %d packed (b*-tree), %d overlapping pairs, %s, HPWL %.4g, %.2fs'
              % (len(mac), res['overlapping_pairs'], '%d outside the core' % outside if outside else 'all in the core',
                 res['hpwl'], res['seconds']))
    return res


def make_die(d, utilization, aspect_ratio=1.0, verbose=True):
    """A new die (one die) for `utilization` = std-cell area / (core - macro area) and `aspect_ratio` = core height /
    width. With rows, the site, row height, row orientations and die-to-core margins come from them and new rows fill
    the core; without rows (block-level floorplans) the die is the core and has no rows. A design with no standard
    cells (only blocks) reads `utilization` as block area / core area."""
    d.require('die_size')
    if d.die_box is not None and d.num_dies != 1:
        raise NotImplementedError('make_die: one die only (this design has %d)' % d.num_dies)
    if not 0 < utilization < 1:
        raise ValueError('utilization must be in (0, 1), got %r' % utilization)
    has_rows = d.row_h is not None
    if has_rows:
        hb = float(d.row_h.min())
        base = torch.nonzero(d.row_h <= hb + 1e-6).flatten()
        sw = float(d.row_site_w[base].min())
        by_y = base[torch.argsort(d.row_xy[base, 1])]
        ys = d.row_xy[by_y, 1]
        nxt = torch.nonzero(ys > ys[0] + 1e-6).flatten()       # rows come in segments: the next distinct y
        if d.row_orient is not None and len(nxt):
            pattern = [int(d.row_orient[by_y[0]]), int(d.row_orient[by_y[nxt[0]]])]
        else:
            pattern = [0, 5]                                    # N, FS alternating
    else:
        hb = sw = 1.0 / d.dbu                                    # snap to the database grid only
    if d.die_box is not None:
        die0, core0 = d.die_box[0].tolist(), d.core_box[0].tolist()
    else:
        die0 = core0 = [0.0, 0.0, 0.0, 0.0]
    ml, mb = core0[0] - die0[0], core0[1] - die0[1]
    mr, mt = die0[2] - core0[2], die0[3] - core0[3]
    area = d.size_n.prod(1)
    mac = (d.flags & MACRO) != 0
    # cells without connections (fill, tap cells released from the old die) are set aside by unplace: not counted
    std = (d.flags & (MACRO | TERMINAL | ASIDE | PHYSICAL)) == 0
    std_area, mac_area = float(area[std].sum()), float(area[mac].sum())
    A = std_area / utilization + mac_area if std_area > 0 else mac_area / utilization
    W = math.sqrt(A / aspect_ratio)
    H = A / W
    if bool(mac.any()):                                         # the biggest macro must fit, with some room
        W = max(W, 1.1 * float(d.size_n[mac, 0].max()))
        H = max(H, 1.1 * float(d.size_n[mac, 1].max()))
    n_sites, n_rows = math.ceil(W / sw), math.ceil(H / hb)
    x0, y0 = die0[0] + ml, die0[1] + mb
    die = [die0[0], die0[1], x0 + n_sites * sw + mr, y0 + n_rows * hb + mt]
    rows = None
    if has_rows:
        ys = [y0 + r * hb for r in range(n_rows)]
        rows = {'xy': [[x0, y] for y in ys], 'h': [hb] * n_rows, 'site_w': [sw] * n_rows, 'sites': [n_sites] * n_rows,
                'orient': [pattern[r % 2] for r in range(n_rows)]}
    released = d.set_floorplan([die], rows, 'naive:make_die(util=%g, aspect=%g)' % (utilization, aspect_ratio))
    core = d.core_box[0].tolist()
    fill = float(d.utilization()[0]) if std_area > 0 else mac_area / ((core[2] - core[0]) * (core[3] - core[1]))
    if verbose:
        print('  die: %.4g x %.4g, core %.4g x %.4g%s, %s %.1f%%%s'
              % (die[2] - die[0], die[3] - die[1], core[2] - core[0], core[3] - core[1],
                 ' (%d rows x %d sites)' % (n_rows, n_sites) if has_rows else ' (no rows)',
                 'utilization' if std_area > 0 else 'block area', 100 * fill,
                 ', %d fixed cells released' % released if released else ''))
    return {'die': die, 'rows': n_rows if has_rows else 0, 'sites': n_sites if has_rows else 0, 'utilization': fill,
            'released': released}
