"""Naive global placement: ePlace-style electrostatic placement (after ePlace, DREAMPlace and Xplace), pure PyTorch.

  objective   WA wirelength + lambda * electrostatic energy of the cell density
  density     cell area spread exactly over bins (attar.kernels.Grid.accumulate); cells smaller than sqrt(2) bins
              are expanded to that size with proportionally lower density, as in ePlace
  field       Poisson equation solved with the DCT, written as matrix products with cos/sin bases (cuBLAS)
  gradient    WA by autograd; density gradient = -(charge density) x integral of the field over the cell
              (attar.kernels.Grid.integrate), exact for the piecewise-constant field
  optimizer   Nesterov with Lipschitz step estimation and backtracking; per-cell preconditioner
  schedule    lambda grows by mu each iteration (DREAMPlace rule), WA gamma follows the overflow, stop at overflow
              below `stop_overflow`
Fillers occupy the white space up to `target_density`. Fixed cells and macros are static charge. Movable macros are
treated as big cells (no macro legalization here).
"""
import math
import time

import torch

from ..distill import ASIDE, MACRO, TERMINAL
from .. import metrics as M
from ..kernels import Grid, default_bins


def _graph(fn, example):
    """Capture fn(x) as a CUDA graph; returns f(x) that copies x into the static input and replays."""
    static = example.detach().clone()
    dev = example.device                           # capture on the tensors' GPU, not the current default one
    with torch.cuda.device(dev):
        s = torch.cuda.Stream(dev); s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            for _ in range(2):
                fn(static)
        torch.cuda.current_stream(dev).wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):     # torch's default capture stream is created once, on one GPU
            out = fn(static)

    def run(x):
        static.copy_(x)
        with torch.cuda.device(dev):
            graph.replay()
        return out
    return run



# DREAMPlace's routability defaults (dreamplace/params.json), RUDY branch
ROUTE_DEFAULTS = {'node_area_adjust_overflow': 0.15, 'max_num_area_adjust': 3, 'route_opt_adjust_exponent': 2.0,
                  'max_route_opt_adjust_rate': 2.0, 'area_adjust_stop_ratio': 0.01, 'route_area_adjust_stop_ratio': 0.01,
                  'routing_layers': None}


def _track_capacity(d, g, ra):
    """Routing tracks per um (horizontal, vertical) of every bin: 1/pitch summed over the signal routing layers of
    each direction (ra['routing_layers'] = (lowest, highest) layer names; default: the slow loop's own, recorded in
    manifest['routing'], else all but the lowest), times the share the router may use (manifest['routing']
    ['adjustment'] held back), less the layers up to the macros' highest pin layer over the part of a bin macros
    cover."""
    import os
    from .. import st
    tp, _ = st.load(os.path.join(d.path, 'tech_profile.safetensors'))
    names = d.names['layers']
    rt = d.manifest.get('routing', {})                     # the slow loop's own routing resources, when recorded
    rng = ra.get('routing_layers') or rt.get('layers')
    lo_l, hi_l = (names.index(rng[0]), names.index(rng[1])) if rng else (1, len(names) - 1)
    keep = 1.0 - float(rt.get('adjustment', 0.0))          # share of a layer's tracks the router may use
    pitch = tp['layer_pitch'].astype('float64') / d.dbu
    dirs = tp['layer_dir']
    layers = range(lo_l, hi_l + 1)
    dev = d.device
    # macros: up to their highest signal-pin layer they block routing over their area
    mac = d.macros
    top = -1
    if len(mac):
        mp = d.raw['m_pin_layer']; ptr = d.raw['m_pin_ptr']
        for m in set(d.raw['c_master'][mac.cpu().numpy()].tolist()):
            if m >= 0 and ptr[m + 1] > ptr[m]:
                top = max(top, int(mp[ptr[m]:ptr[m + 1]].max()))
    cov = torch.zeros(g.nx, g.ny, device=dev)
    if len(mac) and top >= 0:
        p = d.pos[mac]; s = d.size[mac]
        cov = (g.accumulate(p[:, 0], p[:, 1], p[:, 0] + s[:, 0], p[:, 1] + s[:, 1], torch.ones(len(mac), device=dev))
               / g.bin_area).clamp(max=1.0)
    cap = []
    for horizontal in (True, False):
        free = keep * sum(1.0 / pitch[l] for l in layers if (dirs[l] == 0) == horizontal and pitch[l] > 0 and l > top)
        blocked = keep * sum(1.0 / pitch[l] for l in layers if (dirs[l] == 0) == horizontal and pitch[l] > 0 and l <= top)
        cap.append(free + blocked * (1 - cov))
    return cap[0] if torch.is_tensor(cap[0]) else torch.full((g.nx, g.ny), cap[0], device=dev), \
        cap[1] if torch.is_tensor(cap[1]) else torch.full((g.nx, g.ny), cap[1], device=dev)

class Electrostatics:
    """Density map -> potential and field on an nx x ny grid, via DCT bases as matrices."""

    def __init__(self, grid):
        g = grid
        dev = g.device
        self.g = g
        ix = torch.arange(g.nx, device=dev, dtype=torch.float64) + 0.5
        iy = torch.arange(g.ny, device=dev, dtype=torch.float64) + 0.5
        u = torch.arange(g.nx, device=dev, dtype=torch.float64)
        v = torch.arange(g.ny, device=dev, dtype=torch.float64)
        self.Cx = torch.cos(math.pi * u[:, None] * ix[None, :] / g.nx).float()      # (u, i)
        self.Cy = torch.cos(math.pi * v[:, None] * iy[None, :] / g.ny).float()
        self.Sx = torch.sin(math.pi * u[:, None] * ix[None, :] / g.nx).float()
        self.Sy = torch.sin(math.pi * v[:, None] * iy[None, :] / g.ny).float()
        wx = (math.pi * u / (g.nx * g.bw)).float()                                  # physical frequencies
        wy = (math.pi * v / (g.ny * g.bh)).float()
        w2 = wx[:, None] ** 2 + wy[None, :] ** 2
        w2[0, 0] = 1.0
        inv = 1.0 / w2
        inv[0, 0] = 0.0                                                             # no DC term
        norm = torch.full((g.nx, g.ny), 4.0 / (g.nx * g.ny), device=dev)
        norm[0, :] *= 0.5; norm[:, 0] *= 0.5
        self.P = norm * inv                                  # potential coefficients
        self.Px = norm * inv * wx[:, None]                   # field x
        self.Py = norm * inv * wy[None, :]                   # field y

    def __call__(self, rho):
        a = self.Cx @ rho @ self.Cy.T                        # DCT-II coefficients (unnormalised)
        psi = self.Cx.T @ (a * self.P) @ self.Cy
        ex = self.Sx.T @ (a * self.Px) @ self.Cy
        ey = self.Cx.T @ (a * self.Py) @ self.Sy
        return psi, ex, ey


def _diverged(hist, best, window=50, threshold=0.05):
    """DREAMPlace's divergence check (NonLinearPlace.check_divergence): once the overflow has been below 0.25, the
    last `window` iterations average an HPWL more than 1.2 x threshold above the best solution's and an overflow more
    than threshold above the best overflow."""
    if best[0] > 0.25 or len(hist['overflow']) < window:
        return False
    wl = sum(hist['hpwl'][-window:]) / window
    ov = sum(hist['overflow'][-window:]) / window
    return (wl - best[1]) / best[1] > threshold * 1.2 and (ov - best[0]) / max(best[0], 1e-9) > threshold


def global_place(d, target_density=0.9, stop_overflow=0.07, max_iters=2000, bins=None, seed=0,
                 verbose=True, log_every=50, snapshot='naive_gp', graphs=True, movable=None, macro_halo=0.0,
                 routability=False, route_args=None):
    """Place the movable cells of d. Sets d.pos (the placement becomes a fact) and saves a snapshot.
    Returns a dict with the trajectory (iterations, HPWL, overflow, time)."""
    from ..evaluator import Evaluator
    if movable is None:
        d.require('global_place')
    dev = d.device
    t_start = time.time()
    torch.manual_seed(seed)
    if movable is None:
        util = float(d.utilization()[0])
        if util > target_density:           # cells cannot spread thinner than they are (ISPD 2019 test7-10: 96%)
            if verbose:
                print('  gp: utilization %.3f > target density %.3f: target raised to it' % (util, target_density))
            target_density = util
    core = d.core_box[0]
    cx0, cy0, cx1, cy1 = [float(v) for v in core]
    W, H = cx1 - cx0, cy1 - cy0

    mvmask = d.movable if movable is None else movable
    mv = torch.nonzero(mvmask).flatten()
    fx = torch.nonzero(~mvmask & ((d.flags & (TERMINAL | ASIDE)) == 0)).flatten()
    size = d.size
    msize = size[mv]
    marea = msize.prod(1)
    mov_area = marea.sum()

    # fixed objects: static density (clipped to the bin), free area
    n_bins = bins or default_bins(len(mv))
    g = Grid(cx0, cy0, cx1, cy1, n_bins, n_bins, dev)
    fix_map = torch.zeros(g.nx, g.ny, device=dev)
    if len(fx):
        p = d.pos[fx]
        # macro_halo: a keep-out band (um) around every fixed macro, counted as taken
        h = (macro_halo * ((d.flags[fx] & MACRO) != 0).float())[:, None] if macro_halo else torch.zeros(len(fx), 1, device=dev)
        x0 = (p[:, 0:1] - h).clamp(min=cx0)[:, 0]; y0 = (p[:, 1:2] - h).clamp(min=cy0)[:, 0]
        x1 = (p[:, 0:1] + size[fx, 0:1] + h).clamp(max=cx1)[:, 0]; y1 = (p[:, 1:2] + size[fx, 1:2] + h).clamp(max=cy1)[:, 0]
        fix_map = g.accumulate(x0, y0, x1, y1, torch.ones(len(fx), device=dev)).clamp(max=g.bin_area)
    # area without rows (e.g. macro halos and channels where the flow cut the rows) cannot take cells either
    if d.row_box is not None:
        hb = d.row_h.min()
        base = d.row_h <= hb + 1e-6
        rb = d.row_box[base]
        rows_cov = g.accumulate(rb[:, 0], rb[:, 1], rb[:, 2], rb[:, 3], torch.ones(len(rb), device=dev)).clamp(max=g.bin_area)
        fix_map = torch.maximum(fix_map, g.bin_area - rows_cov).clamp(max=g.bin_area)
    free_area = (g.bin_area - fix_map).sum()

    # fillers: average standard-cell size, fill up to the target density
    std = (d.flags[mv] & MACRO) == 0
    fsize = msize[std].mean(0) if std.any() else msize.mean(0)
    filler_area = (target_density * free_area - mov_area).clamp(min=0)
    n_fill = int((filler_area / fsize.prod()).item())
    nodes_size = torch.cat([msize, fsize[None].expand(n_fill, 2)], 0)
    n_mov = len(mv)

    # expanded sizes for the density (ePlace local smoothing)
    min_sz = torch.tensor([g.bw, g.bh], device=dev) * math.sqrt(2)
    esize = torch.maximum(nodes_size, min_sz)
    ratio = nodes_size.prod(1) / esize.prod(1)               # density per unit area of the expanded cell
    td = torch.full((), float(target_density), device=dev)  # target density: a tensor, raised by cell inflation
    static = fix_map * target_density                         # fixed charge, scaled like Xplace

    es = Electrostatics(g)
    ev = Evaluator(d)
    cell, net, E, wmask = ev.cell, ev.net, ev.E, ev.wmask
    off = ev.off = ev.off.clone()                         # pin offsets from the lower-left: follow cell inflation
    pins = torch.zeros(len(d.flags), device=dev).index_add_(0, d.p_cell, torch.ones_like(d.p_cell, dtype=torch.float32))
    npins = torch.cat([pins[mv], torch.zeros(n_fill, device=dev)])
    area_all = nodes_size.prod(1)

    # initial positions (lower-left): movable cells around the core center, fillers spread uniformly
    ctr = torch.tensor([(cx0 + cx1) / 2, (cy0 + cy1) / 2], device=dev)
    x = torch.empty(n_mov + n_fill, 2, device=dev)
    x[:n_mov] = ctr + torch.randn(n_mov, 2, device=dev) * torch.tensor([W, H], device=dev) * 0.001 - msize / 2
    x[n_mov:] = torch.rand(n_fill, 2, device=dev) * torch.tensor([W, H], device=dev) + torch.tensor([cx0, cy0], device=dev) - fsize / 2
    lo = torch.tensor([cx0, cy0], device=dev)
    hi = torch.tensor([cx1, cy1], device=dev) - nodes_size

    base = d.pos.detach().clone()
    gamma_base = 2.0 * (g.bw + g.bh)
    lam = torch.zeros((), device=dev)                    # tensors, so CUDA graphs see the updates
    gamma = torch.full((), gamma_base * 10.0, device=dev)
    static_rho = static / g.bin_area
    half_e = esize / 2
    mov_area_t = mov_area.clone()

    def full_pos(xm):
        return base.index_copy(0, mv, xm)

    def obj_grad(xa):
        """objective, preconditioned gradient (forward only: analytic WA and density gradients)"""
        xa = torch.maximum(torch.minimum(xa, hi), lo)
        p = full_pos(xa[:n_mov])[cell] + off
        wl, gp = M.wa_value_grad(p, net, E, gamma, wmask)
        gwl = torch.zeros(len(base), 2, device=dev).index_add_(0, cell, gp)[mv]
        cen = xa + nodes_size / 2
        c = g.corners(cen[:, 0] - half_e[:, 0], cen[:, 1] - half_e[:, 1], cen[:, 0] + half_e[:, 0], cen[:, 1] + half_e[:, 1])
        rho = g.accumulate_c(c, ratio) / g.bin_area + static_rho
        psi, ex, ey = es(rho)
        energy = 0.5 * (rho * psi).sum() * g.bin_area
        gd = -g.integrate_c(torch.stack([ex, ey]), c) * ratio[:, None]
        grad = lam * gd
        grad[:n_mov] += gwl
        pre = (npins + lam * area_all).clamp(min=1.0)
        return wl + lam * energy, grad / pre[:, None], gwl, gd

    def metrics(ua):
        hp = ev.hpwl(full_pos(ua[:n_mov]))
        cen = ua[:n_mov] + nodes_size[:n_mov] / 2
        c = g.corners(cen[:, 0] - half_e[:n_mov, 0], cen[:, 1] - half_e[:n_mov, 1], cen[:, 0] + half_e[:n_mov, 0], cen[:, 1] + half_e[:n_mov, 1])
        m = g.accumulate_c(c, ratio[:n_mov])
        ov = (m + fix_map * td - td * g.bin_area).clamp(min=0).sum() / mov_area_t
        return torch.stack([hp, ov])

    # initial lambda: ePlace / DREAMPlace ratio of gradient norms
    _, _, gwl0, gd0 = obj_grad(x)
    lam.fill_((gwl0.abs().sum() / gd0.abs().sum().clamp(min=1e-30) * 8e-5).item())

    # CUDA graphs: one replay per objective evaluation instead of ~100 kernel launches
    og, mt = obj_grad, metrics
    if dev.type == 'cuda' and graphs:
        og = _graph(lambda xa: obj_grad(xa)[:2], x)
        mt = _graph(metrics, x)
    else:
        og = lambda xa: obj_grad(xa)[:2]

    # WA gamma from the overflow, from the start (DREAMPlace): a fixed initial value jumps at the first update when
    # the initial overflow is far from 1 (ispd2006_hb/ibm02: 0.58), and the Lipschitz estimate then collapses
    gamma_of = lambda ov: gamma_base * 10 ** ((min(max(ov, 0.0), 1.0) - 0.1) * 20 / 9 - 1)
    gamma.fill_(gamma_of(float(metrics(x)[1])))

    # routability (DREAMPlace adjust_node_area, RUDY branch): when the overflow first falls below
    # node_area_adjust_overflow, cells in bins whose routing utilization (RUDY over track capacity) is high grow by
    # clamp(util^exponent, 1/max_rate, max_rate) of their area, within 10% of the whitespace; the target density
    # follows; at most max_num_area_adjust times, until the growth is below area_adjust_stop_ratio
    ra = dict(ROUTE_DEFAULTS, **(route_args or {}), active=routability, done=0, rounds=[])
    if routability:
        cap_h, cap_v = _track_capacity(d, g, ra)
        ra['capacity_h'], ra['capacity_v'] = float(cap_h.mean()), float(cap_v.mean())
        total_place = free_area                                  # area cells may use
        std_mov = torch.zeros(n_mov + n_fill, dtype=torch.bool, device=dev); std_mov[:n_mov] = std
        dl = torch.zeros(len(base), 2, device=dev)

    def inflate(ua):
        """Grow the cells in congested bins (in place: sizes, density terms, pin offsets, positions). True if any grew."""
        p = full_pos(ua[:n_mov])[cell] + off
        lo_, hi_ = M.net_bbox(p, net, E)
        ok = ev.deg >= 2
        lo_, hi_, w_ = lo_[ok], hi_[ok], wmask[ok]
        sz = torch.maximum(hi_ - lo_, torch.tensor([g.bw, g.bh], device=dev))
        c_ = (lo_ + hi_) / 2; lo_, hi_ = c_ - sz / 2, c_ + sz / 2
        dem_h = g.accumulate(lo_[:, 0], lo_[:, 1], hi_[:, 0], hi_[:, 1], w_ / sz[:, 1]) / g.bin_area   # horizontal tracks/um
        dem_v = g.accumulate(lo_[:, 0], lo_[:, 1], hi_[:, 0], hi_[:, 1], w_ / sz[:, 0]) / g.bin_area
        util = torch.maximum(dem_h / cap_h.clamp(min=1e-9), dem_v / cap_v.clamp(min=1e-9))
        adj = util.pow(ra['route_opt_adjust_exponent']).clamp(1 / ra['max_route_opt_adjust_rate'], ra['max_route_opt_adjust_rate'])
        cen = ua + nodes_size / 2
        cc = g.corners(cen[:, 0] - nodes_size[:, 0] / 2, cen[:, 1] - nodes_size[:, 1] / 2,
                       cen[:, 0] + nodes_size[:, 0] / 2, cen[:, 1] + nodes_size[:, 1] / 2)
        route_area = g.integrate_c(torch.stack([adj, adj]), cc)[:, 0]          # sum over bins of overlap x adj
        old = nodes_size.prod(1)
        inc = torch.where(std_mov, (route_area - old).clamp(min=0), torch.zeros_like(old))
        old_mov = old[:n_mov].sum(); inc_sum = inc.sum()
        ratio_inc = float(inc_sum / old_mov)
        ra['rounds'].append({'iteration': len(hist['iter']), 'max_util': float(util.max()), 'growth': ratio_inc})
        if ratio_inc < ra['area_adjust_stop_ratio'] or ratio_inc < ra['route_area_adjust_stop_ratio']:
            ra['active'] = False
            return False
        budget = min(0.1 * float(total_place - old_mov), float(total_place - old_mov))
        scale = budget / float(inc_sum)
        if scale <= 0:
            ra['active'] = False
            return False
        inc = inc * min(1.0, scale)
        k = ((old + inc) / old).sqrt()
        new_size = nodes_size * k[:, None]
        new_mov = new_size[:n_mov].prod(1).sum()
        fill_old = old[n_mov:].sum()
        if n_fill and new_mov + fill_old > total_place * td:     # fillers give the room back
            fr = ((total_place * td - new_mov).clamp(min=0) / fill_old).sqrt()
            new_size[n_mov:] = new_size[n_mov:] * fr
        delta = (new_size - nodes_size) / 2                      # keep centers: lower-left moves by -delta
        ua.sub_(delta)
        dl.zero_(); dl[mv] = delta[:n_mov]
        off.add_(dl[cell])                                       # pins keep their absolute place
        nodes_size.copy_(new_size)
        esize.copy_(torch.maximum(nodes_size, min_sz)); half_e.copy_(esize / 2)
        ratio.copy_(nodes_size.prod(1) / esize.prod(1)); area_all.copy_(nodes_size.prod(1))
        hi.copy_(torch.tensor([cx1, cy1], device=dev) - nodes_size)
        mov_area_t.copy_(nodes_size[:n_mov].prod(1).sum())
        td.copy_(((nodes_size.prod(1).sum()) / total_place).clamp(max=1.0))
        static_rho.copy_(fix_map * td / g.bin_area)
        ra['done'] += 1
        ra['rounds'][-1].update(scale=min(1.0, scale), target_density=float(td))
        return True

    # Nesterov (ePlace Algorithm 2) with Lipschitz step estimation and backtracking
    u = x.clone()
    v = x.clone()
    f_v, g_v = [t.clone() for t in og(v)]
    v1 = torch.maximum(torch.minimum(v - 0.01 * min(g.bw, g.bh) * g_v / g_v.norm().clamp(min=1e-30) * math.sqrt(len(v)), hi), lo)
    _, g_v1 = [t.clone() for t in og(v1)]
    alpha = ((v - v1).norm() / (g_v - g_v1).norm().clamp(min=1e-30)).item()
    a_k = 1.0
    hist = {'iter': [], 'hpwl': [], 'overflow': [], 'lambda': [], 'gamma': []}
    prev_hpwl = None
    best = None                                              # (overflow, HPWL, iteration, positions)
    diverged = False
    it = 0
    for it in range(max_iters):
        a_k1 = (1 + math.sqrt(4 * a_k * a_k + 1)) / 2
        beta = (a_k - 1) / a_k1
        alpha0 = alpha
        for tries in range(10):                               # backtracking on the Lipschitz estimate
            u1 = torch.maximum(torch.minimum(v - alpha * g_v, hi), lo)
            v_new = torch.maximum(torch.minimum(u1 + beta * (u1 - u), hi), lo)
            f_new, g_new = [t.clone() for t in og(v_new)]
            alpha_new = ((v_new - v).norm() / (g_new - g_v).norm().clamp(min=1e-30)).item()
            if alpha_new > 0.95 * alpha:
                break
            alpha = alpha_new
        else:
            # no step accepted: g_v predates this iteration's lambda / gamma update; take it again here and step once
            g_v = og(v)[1].clone()
            alpha = alpha0
            u1 = torch.maximum(torch.minimum(v - alpha * g_v, hi), lo)
            v_new = torch.maximum(torch.minimum(u1 + beta * (u1 - u), hi), lo)
            f_new, g_new = [t.clone() for t in og(v_new)]
            alpha_new = ((v_new - v).norm() / (g_new - g_v).norm().clamp(min=1e-30)).item()
        u, v, g_v, alpha, a_k = u1, v_new, g_new, alpha_new, a_k1

        # schedule (one host sync for HPWL and overflow)
        hp, ov = mt(u).tolist()
        if prev_hpwl is not None:
            dh = hp - prev_hpwl
            mu = 1.05 * max(0.9999 ** it, 0.98) if dh < 0 else 1.05 * min(max(1.05 ** (-dh / 350000), 0.95), 1.05)
            lam.mul_(mu)
        prev_hpwl = hp
        gamma.fill_(gamma_of(ov))
        hist['iter'].append(it); hist['hpwl'].append(hp); hist['overflow'].append(ov)
        if verbose and (it % log_every == 0):
            print('  gp iter %4d  HPWL %.4e  overflow %.3f  lambda %.2e  gamma %.3g' % (it, hp, ov, lam.item(), gamma.item()))
        if routability and ra['active'] and ov < ra['node_area_adjust_overflow'] and ra['done'] < ra['max_num_area_adjust']:
            if inflate(u):                                   # cells grew: restart the momentum from u
                u, v = u.clone(), u.clone()
                f_v, g_v = [t.clone() for t in og(v)]
                a_k = 1.0
                best = None
                prev_hpwl = None
                continue
        if best is None or ov < best[0]:
            best = (ov, hp, it, u.clone())
        if ov < stop_overflow and it > 20:
            break
        if _diverged(hist, best):                          # roll back to the best solution (DREAMPlace)
            diverged = True
            break

    if dev.type == 'cuda':
        torch.cuda.synchronize()
    if best is not None and best[0] < hist['overflow'][-1] - 1e-9:
        u = best[3]
        hist['restored'] = best[2]
    if routability:                                          # back to the true sizes, centers kept
        u = u.clone(); u[:n_mov] += (nodes_size[:n_mov] - msize) / 2
    if movable is None:
        d.pos = full_pos(u[:n_mov]).detach()
        d.snapshot(snapshot)
    else:                                          # mixed-size use (macro placer): positions only, no state change
        d._pos = full_pos(u[:n_mov]).detach()
    k = hist.get('restored', -1)
    res = {'iterations': it + 1, 'hpwl': hist['hpwl'][k], 'overflow': hist['overflow'][k], 'diverged': diverged,
           'restored_iteration': hist.get('restored'),
           'seconds': time.time() - t_start, 'bins': (g.nx, g.ny), 'fillers': n_fill, 'history': hist}
    if routability:
        res['routability'] = {k: v for k, v in ra.items() if k in ('done', 'rounds', 'capacity_h', 'capacity_v')}
    if verbose:
        print('  gp done: %d iterations, HPWL %.4e, overflow %.3f, %.2fs (%.2f ms/iter), %d bins, %d fillers'
              % (res['iterations'], res['hpwl'], res['overflow'], res['seconds'], 1e3 * res['seconds'] / res['iterations'],
                 g.nx, n_fill))
    return res
