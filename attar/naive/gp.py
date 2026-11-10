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
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn(static)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(static)

    def run(x):
        static.copy_(x)
        graph.replay()
        return out
    return run


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
                 verbose=True, log_every=50, snapshot='naive_gp', graphs=True, movable=None):
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
        fix_map = g.accumulate(p[:, 0], p[:, 1], p[:, 0] + size[fx, 0], p[:, 1] + size[fx, 1],
                               torch.ones(len(fx), device=dev)).clamp(max=g.bin_area)
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
    static = fix_map * target_density                         # fixed charge, scaled like Xplace

    es = Electrostatics(g)
    ev = Evaluator(d)
    cell, off, net, E, wmask = ev.cell, ev.off, ev.net, ev.E, ev.wmask
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
        cen = ua[:n_mov] + msize / 2
        c = g.corners(cen[:, 0] - half_e[:n_mov, 0], cen[:, 1] - half_e[:n_mov, 1], cen[:, 0] + half_e[:n_mov, 0], cen[:, 1] + half_e[:n_mov, 1])
        m = g.accumulate_c(c, ratio[:n_mov])
        ov = (m + fix_map * target_density - target_density * g.bin_area).clamp(min=0).sum() / mov_area
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
    if movable is None:
        d.pos = full_pos(u[:n_mov]).detach()
        d.snapshot(snapshot)
    else:                                          # mixed-size use (macro placer): positions only, no state change
        d._pos = full_pos(u[:n_mov]).detach()
    k = hist.get('restored', -1)
    res = {'iterations': it + 1, 'hpwl': hist['hpwl'][k], 'overflow': hist['overflow'][k], 'diverged': diverged,
           'restored_iteration': hist.get('restored'),
           'seconds': time.time() - t_start, 'bins': (g.nx, g.ny), 'fillers': n_fill, 'history': hist}
    if verbose:
        print('  gp done: %d iterations, HPWL %.4e, overflow %.3f, %.2fs (%.2f ms/iter), %d bins, %d fillers'
              % (res['iterations'], res['hpwl'], res['overflow'], res['seconds'], 1e3 * res['seconds'] / res['iterations'],
                 g.nx, n_fill))
    return res
