"""Design visualization: die, core, rows, standard cells, macros, IO pins (matplotlib, headless).

One light theme in the logo's colors (docs/images/logo.png, THEME): standard cells in its orange (rectangles, or an
area-density image along its yellow -> coral gradient for large designs), macros in its cream with a gold edge, ink in
its espresso, IO pins by direction (input gold, output rust, inout plum), thin outlines for die / core / rows, no
axes: a scale bar and a title block with the design's numbers instead. Block-level floorplans also show the connections
between macros and IO pins: one ink, width and opacity by connection strength (nets between the two, clique model).
Tile maps (plot(tiles=...)): global-routing overflow, pin-access penalty, cell density."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from .distill import ASIDE, FIXED, MACRO, PHYSICAL, TERMINAL  # noqa: E402

# the logo's colors (docs/images/logo.png): its yellow -> amber -> orange -> coral gradient for cells and heat, cream
# (the crossbar) and gold for macros, espresso (the background) for ink; status green / red only for legality
THEME = dict(surface='#fffaf3', text='#25150f', text2='#6b4f40', muted='#a8907e', die='#3c2318', core='#d6b896',
             row='#f4e8d8', cell='#f87c3b', density=('#fffaf3', '#ffe2a8', '#ffc656', '#ff9a3c', '#f15d3a'),
             fixed='#d6c2a8', macro='#ffeccb', macro_edge='#c19443', io=('#c19443', '#d34f34', '#7a4e8c', '#a8907e'),
             ok='#4f7a28', bad='#b8392a', ok_bg='#e9efdc', bad_bg='#f9e0d8', net='#3c2318',
             load=(('#ffe2a8', '#ffc656', '#f8a040'), ('#c4402c', '#7a2a18', '#3c2318')),
             penalty=('#ff9a3c', '#f15d3a', '#c4402c', '#7a2a18'))
IO_NAMES = ('input', 'output', 'inout', 'feedthru')


def _rects(xy, wh):
    x0, y0 = xy[:, 0], xy[:, 1]; x1, y1 = x0 + wh[:, 0], y0 + wh[:, 1]
    return np.stack([np.stack([x0, y0], 1), np.stack([x1, y0], 1), np.stack([x1, y1], 1), np.stack([x0, y1], 1)], 1)


def _legal_status(d, pos, own):
    """(True / False / None when there is nothing to judge, the problems, mask of the macros outside their core):
    Design.floorplan_check of what is drawn; with the design's own placement on rows, Evaluator.legality too."""
    import torch
    judged = len(d.macros) > 0 and ('floorplan.macros' in d.facts or not own) and 'floorplan.die' in d.facts
    issues, out = [], np.zeros(len(pos), bool)
    if judged:
        issues, o = d.floorplan_check(None if own else torch.as_tensor(pos, device=d.device, dtype=d.pos.dtype))
        out = o.cpu().numpy()
    if own and d.has('placement') and 'rows' in d.facts and bool(d.movable.any()):
        from .evaluator import Evaluator
        L = Evaluator(d).legality()
        judged = True
        if not L['legal']:
            bad = sum(v for k, v in L.items() if k not in ('cells', 'legal'))
            issues.append('%s cell problems' % '{:,}'.format(bad))
    return (None if not judged else not issues), issues, out


def _connections(d, pos, size, nodes, max_lines, max_degree=32):
    """The strongest connections between the cells in `nodes` (macros, IO pins): for every net, each pair of its
    nodes gets 1 / (k - 1), k its number of nodes (clique model); nets of more than max_degree nodes are left out.
    -> (segments (L, 2, 2), strengths (L,) in (0, 1]), strongest last."""
    pc = d.p_cell.cpu().numpy(); pn = d.p_net.cpu().numpy()
    keep = nodes[pc]
    N = len(nodes)
    key = np.unique(pn[keep].astype(np.int64) * N + pc[keep])         # (net, node) once each, sorted by net
    net, cell = key // N, key % N
    cut = np.flatnonzero(np.r_[True, net[1:] != net[:-1], True])
    w = {}
    for a, b in zip(cut[:-1], cut[1:]):
        k = b - a
        if 2 <= k <= max_degree:
            c = cell[a:b]
            for i in range(k):
                for j in range(i + 1, k):
                    e = (int(c[i]), int(c[j]))
                    w[e] = w.get(e, 0.0) + 1.0 / (k - 1)
    if not w:
        return np.zeros((0, 2, 2)), np.zeros(0)
    E = sorted(w.items(), key=lambda kv: kv[1])[-max_lines:]
    ij = np.array([e for e, _ in E]); s = np.array([v for _, v in E])
    ctr = pos + size / 2
    return np.stack([ctr[ij[:, 0]], ctr[ij[:, 1]]], 1), s / s.max()


def _unit(u):
    return 'µm' if u in ('um', 'micron', 'microns') else u


def _num(v):
    return '{:,.0f}'.format(v) if abs(v) >= 1000 else '%.4g' % v


def _scale_bar(ax, box, unit, t):
    """A scale bar of a round length about a fifth of the die width, under the die."""
    W = box[2] - box[0]
    raw = W / 5
    p = 10 ** np.floor(np.log10(raw))
    L = max(v * p for v in (1, 2, 5) if v * p <= raw)
    x0, y = box[0], box[1] - 0.035 * (box[3] - box[1])
    ax.plot([x0, x0 + L], [y, y], color=t['text2'], lw=1.6, solid_capstyle='butt', zorder=9, clip_on=False)
    for x in (x0, x0 + L):                               # end ticks
        ax.plot([x, x], [y - 0.006 * W, y + 0.006 * W], color=t['text2'], lw=1.0, zorder=9, clip_on=False)
    ax.text(x0 + L + 0.014 * W, y, '%g %s' % (L, _unit(unit)), va='center', ha='left', fontsize=8, color=t['text2'],
            clip_on=False)


def _cell_density(ax, d, t, die_box, pos, size, sel):
    """The standard cells in `sel` as an area-density image over the die: surface -> cell color, empty bins
    transparent. A bin spans at least two rows (finer bins alias the row structure into stripes), at most 640 per
    side."""
    Wd, Hd = die_box[2] - die_box[0], die_box[3] - die_box[1]
    hb = float(d.row_h.min()) if d.row_h is not None else max(Wd, Hd) / 640
    bw = max(max(Wd, Hd) / 640, 2 * hb)
    bins = (max(16, int(round(Wd / bw))), max(16, int(round(Hd / bw))))
    xy = pos[sel] + size[sel] / 2
    img, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=[[die_box[0], die_box[2]], [die_box[1], die_box[3]]],
                               weights=size[sel].prod(1))
    dens = np.clip(img / ((Wd / bins[0]) * (Hd / bins[1])), 0, 1).T

    def box(a):                                          # 3x3 box blur: the shape of the placement, not bin noise
        k = np.pad(a, 1, mode='edge')
        return sum(k[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9
    occ = box((dens > 0).astype(float))
    dens = box(dens) / np.maximum(occ, 1e-9)             # density where there are cells; single empty bins close up
    cmap = LinearSegmentedColormap.from_list('attar_cells', list(t['density']))
    rgba = cmap(dens)
    rgba[..., 3] = np.clip(occ * 2 - 0.25, 0, 1)          # soft edges, no pinholes
    ax.imshow(rgba, origin='lower', extent=[die_box[0], die_box[2], die_box[1], die_box[3]], interpolation='bilinear',
              zorder=2)


def _draw_cells(ax, d, t, s, max_cells):
    """Standard cells: movable ones as rectangles (an area-density image above max_cells), fixed ones in their own
    color. Returns legend handles."""
    legend = []
    std = s.on & ~s.term & ~s.macro & s.placed
    fixed_std = std & ((s.flags & (FIXED | PHYSICAL)) != 0)
    mov_std = std & ~fixed_std
    if int(mov_std.sum()) > max_cells:
        _cell_density(ax, d, t, s.die_box, s.pos, s.size, mov_std)
        legend.append(Patch(fc=t['cell'], ec='none', label='cell density'))
    elif mov_std.any():
        thin = int(mov_std.sum()) > 3000
        ax.add_collection(PolyCollection(_rects(s.pos[mov_std], s.size[mov_std]), facecolors=t['cell'],
                                         edgecolors='none' if thin else t['surface'], lw=0 if thin else 0.25,
                                         alpha=0.9, zorder=2))
        legend.append(Patch(fc=t['cell'], ec='none', label='cells'))
    if fixed_std.any():
        ax.add_collection(PolyCollection(_rects(s.pos[fixed_std], s.size[fixed_std]), facecolors=t['fixed'],
                                         edgecolors='none', zorder=2))
        legend.append(Patch(fc=t['fixed'], ec='none', label='fixed cells'))
    return legend


def _draw_macros(ax, d, t, s, nets, max_nets, labels):
    """Macros (outlined in the 'bad' color when outside their core), their names when they fit, and for block-level
    designs the strongest connections between macros and IO pins. Returns legend handles."""
    legend = []
    msel = s.on & s.macro & s.placed
    n_std_on = int((s.on & ~s.term & ~s.macro).sum())
    if (nets if nets is not None else n_std_on == 0) and msel.any():
        io_nodes = np.zeros(len(s.flags), bool)
        if d.io is not None and len(d.io):
            io_nodes[d.io.cpu().numpy()] = True
        cap = max_nets or int(min(400, max(40, 2 * msel.sum())))
        seg, strength = _connections(d, s.pos, s.size, (msel | (io_nodes & s.placed & s.on)), cap)
        if len(seg):
            from matplotlib.colors import to_rgba
            rgba = np.array([to_rgba(t['net'], 0.12 + 0.55 * v) for v in strength])
            ax.add_collection(LineCollection(seg, colors=rgba, linewidths=0.35 + 1.6 * strength, capstyle='round',
                                             zorder=3.5))
            legend.append(Line2D([], [], color=t['net'], lw=1.2, alpha=0.6, label='connections'))
    if msel.any():
        ax.add_collection(PolyCollection(_rects(s.pos[msel], s.size[msel]), facecolors=t['macro'],
                                         edgecolors=[t['bad'] if o else t['macro_edge'] for o in s.outside[msel]],
                                         lw=[1.2 if o else 0.6 for o in s.outside[msel]], zorder=3))
        legend.append(Patch(fc=t['macro'], ec=t['macro_edge'], lw=0.6, label='macros'))
        if labels if labels is not None else msel.sum() <= 40:
            names = d.names['cells']
            for i in np.nonzero(msel)[0]:
                if s.size[i, 0] < 0.08 * s.W:            # too narrow for its name
                    continue
                parts = [x for x in names[i].replace('/', '.').split('.') if x]
                ax.text(s.pos[i, 0] + s.size[i, 0] / 2, s.pos[i, 1] + s.size[i, 1] / 2, '.'.join(parts[-2:])[-24:],
                        ha='center', va='center', fontsize=5.5, color=t['text2'], zorder=4, clip_on=True)
    return legend


def _draw_io(ax, d, t, s):
    """Placed IO pins of the die, one color per direction. Returns (legend handles, number drawn)."""
    legend = []
    io_all = d.io.cpu().numpy() if d.io is not None else np.zeros(0, np.int64)
    sel = s.placed[io_all] & s.on[io_all] if len(io_all) else np.zeros(0, bool)      # unplaced IO pins are not drawn
    io = io_all[sel]
    io_dir = d.io_dir.cpu().numpy()[sel] if len(io_all) else np.zeros(0, np.int64)
    ms = max(2.0, min(14.0, 2500.0 / max(1, len(io))))
    for k in range(4):
        s_ = io[io_dir == k]
        if len(s_):
            ax.scatter(s.pos[s_, 0], s.pos[s_, 1], s=ms, c=t['io'][k], marker='o', zorder=7, linewidths=0)
            legend.append(Line2D([], [], ls='none', marker='o', ms=5, mfc=t['io'][k], mec='none', label='IO %s' % IO_NAMES[k]))
    return legend, len(io)


# -- tiles: a value per tile over the die, drawn in place of the cells ------------------------------------------------
TILES = {
    'gr_overflow': ('global-routing overflow', {'netlist', 'pin_offsets', 'placement', 'floorplan.die', 'tracks', 'routing'}),
    'pin_access': ('pin-access penalty', {'netlist', 'cell_geometry', 'pin_offsets', 'placement', 'pdn'}),
    'density': ('cell density', {'cell_geometry', 'placement', 'floorplan.die'}),
}
LOAD_MAX = 1.5          # load maps (demand / capacity) saturate here: the same colors for every design


def _gp_grid(d, bins, n_mov):
    """Global placement's bin grid over the core (attar.naive.gp: default_bins(#movable) per side), halved until a bin
    spans at least two rows: finer bins alias the rows into stripes. bins: given per side (int or (nx, ny))."""
    from .kernels import Grid, default_bins
    n = bins
    if n is None:
        n = default_bins(n_mov)
        core_h = float(d.core_box[0, 3] - d.core_box[0, 1])
        hb = float(d.row_h.min()) if d.row_h is not None and len(d.row_h) else 0.0
        while n > 16 and core_h / n < 2 * hb:
            n //= 2
    nx, ny = (n, n) if isinstance(n, int) else n
    return Grid.over_box(d.core_box[0].tolist(), nx, ny, d.device)


def _edges(g):
    return g.x0 + g.bw * np.arange(g.nx + 1), g.y0 + g.bh * np.arange(g.ny + 1)


def tiles_density(d, pos=None, bins=None):
    """Cell density as global placement defines it (attar.naive.gp): per bin, the movable cell area over the free area
    (bin minus fixed objects and area without rows), on GP's grid (_gp_grid). The overflow at target density 1: the
    area above the free area, summed over the bins, over the movable area.
    -> (map [ny][nx], NaN where no free area; x edges; y edges; stats)."""
    import torch
    p = d.pos if pos is None else pos
    mv = d.movable
    fx = ~mv & ((d.flags & (TERMINAL | ASIDE)) == 0)
    g = _gp_grid(d, bins, int(mv.sum()))
    sz = d.size
    one = lambda m: torch.ones(int(m.sum()), device=d.device)  # noqa: E731
    mov = g.accumulate(p[mv, 0], p[mv, 1], p[mv, 0] + sz[mv, 0], p[mv, 1] + sz[mv, 1], one(mv))
    fix = g.accumulate(p[fx, 0], p[fx, 1], p[fx, 0] + sz[fx, 0], p[fx, 1] + sz[fx, 1], one(fx)).clamp(max=g.bin_area)
    if d.row_box is not None:                            # no rows: no cells either (as in global placement)
        rb = d.row_box[d.row_h <= d.row_h.min() + 1e-6]
        cov = g.accumulate(rb[:, 0], rb[:, 1], rb[:, 2], rb[:, 3], torch.ones(len(rb), device=d.device)).clamp(max=g.bin_area)
        fix = torch.maximum(fix, g.bin_area - cov)
    free = (g.bin_area - fix).clamp(min=0)
    m = torch.where(free > 1e-3 * g.bin_area, mov / free.clamp(min=1e-12), torch.full_like(mov, float('nan')))
    m = m.double().cpu().numpy().T
    ok = np.isfinite(m)
    ovf = float((mov - free).clamp(min=0).sum() / sz[mv].prod(1).sum().clamp(min=1e-12))
    note = ('overflow %.3f' % ovf, 'densest bin %.2f  ·  %d × %d bins' % (float(m[ok].max(initial=0)), g.nx, g.ny))
    return m, *_edges(g), note


def tiles_pin_access(d, pos=None, bins=None):
    """Pin-access penalty of the PDN-aware detailed placement (attar.naive.lg._pdn_dp): w_pa for every cell with a
    signal pin on a PDN pin blockage of its layer (attar.pdn.conflicts), summed per bin of the global placement grid.
    -> (map [ny][nx], NaN where 0; x edges; y edges; stats)."""
    import torch
    from . import pdn as PDN
    from .naive.lg import W_PA
    p = d.pos if pos is None else pos
    conf = PDN.conflicts(d, pos=p) & ((d.flags & (TERMINAL | ASIDE)) == 0)
    g = _gp_grid(d, bins, int(d.movable.sum()))
    c = torch.nonzero(conf).flatten()
    ctr = p[c] + d.size[c] / 2
    ix = ((ctr[:, 0] - g.x0) / g.bw).floor().long().clamp(0, g.nx - 1)
    iy = ((ctr[:, 1] - g.y0) / g.bh).floor().long().clamp(0, g.ny - 1)
    m = torch.zeros(g.ny * g.nx, dtype=torch.float64, device=d.device)
    m.index_add_(0, iy * g.nx + ix, torch.full((len(c),), W_PA, dtype=torch.float64, device=d.device))
    m = m.view(g.ny, g.nx).cpu().numpy()
    note = ('{:,.0f}'.format(W_PA * len(c)),
            '%s cells with a signal pin on the PDN  ·  w_pa %g per cell  ·  %d × %d bins'
            % ('{:,}'.format(len(c)), W_PA, g.nx, g.ny))
    return np.where(m > 0, m, np.nan), *_edges(g), note


def tiles_gr_overflow(d, pos=None, adjustment=None):
    """Global-routing load on OpenROAD's gcells (d.gr_overflow: RUDY demand against the gcell-edge capacities): per
    gcell, the worst demand / capacity of its edges, both directions (> 1: an overflowing edge).
    -> (map [ny][nx]; x edges; y edges; stats)."""
    r = d.gr_overflow(pos, adjustment=adjustment, maps=True)
    dh, dv, ch, cv = r['demand_h'], r['demand_v'], r['cap_h'], r['cap_v']
    ny, nx = dh.shape
    eh = (dh[:, :-1] + dh[:, 1:]) / 2; ev = (dv[:-1, :] + dv[1:, :]) / 2       # as attar.routing.edge_overflow
    ratio = lambda e, c: np.where(c > 0, e / np.maximum(c, 1e-9), np.where(e > 0, np.inf, 0.0))  # noqa: E731
    uh, uv = ratio(eh, ch), ratio(ev, cv)
    m = np.zeros((ny, nx))
    if nx > 1:
        m[:, :-1] = np.maximum(m[:, :-1], uh); m[:, 1:] = np.maximum(m[:, 1:], uh)
    if ny > 1:
        m[:-1, :] = np.maximum(m[:-1, :], uv); m[1:, :] = np.maximum(m[1:, :], uv)
    # gcell rectangles as OpenROAD's (GlobalRouter::getGCellRect): the last row and column reach the die edge
    die = d.die_box[0].tolist(); tile = r['tile_um']
    xe = die[0] + tile * np.arange(nx + 1); ye = die[1] + tile * np.arange(ny + 1)
    xe[-1], ye[-1] = die[2], die[3]
    note = ('GROV %.2f%%' % r['grov_%'], 'H %.2f%%  ·  V %.2f%%  ·  %s tracks over capacity  ·  %d × %d gcells'
            % (r['grov_h_%'], r['grov_v_%'], '{:,.0f}'.format(r['total_overflow']), nx, ny))
    return m, xe, ye, note


def _load_cmap(t, vmax):
    """Demand / capacity: up to 1 the logo's cream -> yellow -> amber, above 1 a step to rust deepening to espresso at
    vmax (the step at capacity: a lightness and hue change that survives color-vision deficiency)."""
    k = 1.0 / vmax
    lo, hi = t['load']
    stops = [(0.0, t['surface'])] + [(k * (i + 1) / len(lo), c) for i, c in enumerate(lo)]
    stops += [(k + 1e-6, hi[0])] + [(k + (1 - k) * (i + 1) / (len(hi) - 1), c) for i, c in enumerate(hi[1:])]
    return LinearSegmentedColormap.from_list('attar_load', stops)


def _color_bar(fig, ax, t, mesh, caption, ticks, labels, mark=None):
    """A slim color bar right beside the drawing: caption on top, no frame, no tick marks; mark: a hairline across
    the bar at that value (capacity)."""
    cax = ax.inset_axes([1.03, 0.1, 0.022, 0.6])
    cb = fig.colorbar(mesh, cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks(ticks)
    cb.set_ticklabels(labels)
    cb.ax.tick_params(which='both', labelsize=7.5, colors=t['text2'], length=0, pad=4)
    cb.ax.minorticks_off()
    if mark is not None:
        cb.ax.axhline(mark, color=t['text'], lw=0.9)
    cb.ax.text(0, 1.04, caption, transform=cb.ax.transAxes, ha='left', va='bottom', fontsize=8, color=t['text'])
    return cb


def _draw_tiles(fig, ax, d, t, kind, pos, bins, adjustment):
    """The tile map of `kind` over the die (in place of the cells) and its color bar beside the drawing.
    Returns (name, headline, details) for the title block."""
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
    from .facts import MissingFacts
    if kind not in TILES:
        raise ValueError('tiles must be one of %s, got %r' % (', '.join(TILES), kind))
    label, need = TILES[kind]
    miss = need - d.present - (set() if pos is None else {'placement'})
    if miss:
        raise MissingFacts(d.manifest.get('design', d.path), 'plot(tiles=%r)' % kind, miss)
    load_ticks = ([0, 0.5, 1.0, LOAD_MAX], ['0', '0.5', '1.0', '%g+' % LOAD_MAX])
    if kind in ('gr_overflow', 'density'):
        m, xe, ye, note = tiles_gr_overflow(d, pos, adjustment) if kind == 'gr_overflow' else tiles_density(d, pos, bins)
        cmap, norm = _load_cmap(t, LOAD_MAX), Normalize(0, LOAD_MAX)
        bar = (('edge load', 'demand / capacity') if kind == 'gr_overflow' else ('density', 'cell / free area'))
        bar = (bar[0], *load_ticks, 1.0)
    else:
        from .naive.lg import W_PA
        m, xe, ye, note = tiles_pin_access(d, pos, bins)
        top = int(round(float(np.nanmax(m)) / W_PA)) if np.isfinite(m).any() else 1
        ramp = LinearSegmentedColormap.from_list('attar_pa', list(t['penalty']))
        if top <= 6:                                     # few cells per tile: one step per cell
            cmap = ListedColormap([ramp(i / max(top - 1, 1)) for i in range(top)])
            norm = BoundaryNorm(W_PA * (np.arange(top + 1) + 0.5), top)
            ticks = list(W_PA * np.arange(1, top + 1))
        else:
            cmap, norm = ramp, Normalize(W_PA, W_PA * top)
            ticks = list(np.linspace(W_PA, W_PA * top, 4))
        bar = ('penalty', ticks, [_num(v) for v in ticks], None)
    cmap.set_bad((0, 0, 0, 0))
    mesh = ax.pcolormesh(xe, ye, np.ma.masked_invalid(np.minimum(m, 1e9)), cmap=cmap, norm=norm, shading='flat',
                         edgecolors='none', antialiased=False, zorder=2, rasterized=True)
    _color_bar(fig, ax, t, mesh, bar[0], bar[1], bar[2], bar[3])
    return (label,) + note


def _text_run(fig, x, y, parts, renderer):
    """Texts side by side on one line from (x, y) (figure fraction, top aligned): parts = (text, style dict)."""
    for text, kw in parts:
        if not text:
            continue
        tx = fig.text(x, y, text, ha='left', va='top', **kw)
        bb = tx.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        x = bb.x1


def _title_block(fig, ax, d, t, s, die, n_io, title, status, issues, note=None):
    """The design's name, its numbers under it, a 'legal' / 'not legal' badge with the problems on the right, the
    tile map's numbers when there is one, and a note when unplaced cells are left out."""
    c = d.manifest['counts']
    hidden = int((s.on & ~s.placed & ~s.term & ~s.macro).sum())
    if hidden:
        ax.text(0.5, 0.5, '%s unplaced cells not drawn' % '{:,}'.format(hidden), transform=ax.transAxes, ha='center',
                va='center', fontsize=10, color=t['muted'], zorder=8)
    n_std = int((s.on & ~s.term & ~s.macro).sum())
    core = s.core
    core_area = (core[2] - core[0]) * (core[3] - core[1])
    fill = ('utilization %.0f%%' % (100 * d.utilization()[die].item()) if n_std else
            'block area %.0f%%' % (100 * float(s.size[s.on & s.macro].prod(1).sum()) / core_area))
    u = _unit(s.unit)
    stats = ['%s cells' % '{:,}'.format(c['cells'])]
    if c['macros']:
        stats.append('%s macros' % '{:,}'.format(c['macros']))
    if n_io:
        stats.append('%s IO pins' % '{:,}'.format(n_io))
    die_box = s.die_box
    stats += ['%s × %s %s' % (_num(die_box[2] - die_box[0]), _num(die_box[3] - die_box[1]), u), fill]
    if s.own and 'hpwl' in d.capabilities():
        stats.append('HPWL %s %s' % (_num(float(d.hpwl())), u))
    head = title if title is not None else d.manifest.get('design', '')
    L = 0.045
    fig.text(L, 0.972, head, ha='left', va='top', fontsize=15, fontweight='bold', color=t['text'])
    fig.text(L, 0.932, '  ·  '.join(stats), ha='left', va='top', fontsize=8.5, color=t['text2'])
    if note:
        name, headline, details = note
        _text_run(fig, L, 0.904, [(name + '   ', dict(fontsize=8.5, color=t['text2'])),
                                  (headline, dict(fontsize=8.5, fontweight='bold', color=t['text'])),
                                  ('   ' + details, dict(fontsize=8.5, color=t['text2']))], fig.canvas.get_renderer())
    if status is not None:
        fg, bg = (t['ok'], t['ok_bg']) if status else (t['bad'], t['bad_bg'])
        fig.text(1 - L, 0.968, ('✓  legal' if status else '✗  not legal'), ha='right', va='top', fontsize=10,
                 fontweight='bold', color=fg, bbox=dict(boxstyle='round,pad=0.45,rounding_size=0.8', fc=bg, ec='none'))
        if issues:
            fig.text(1 - L, 0.926, ', '.join(issues), ha='right', va='top', fontsize=8, color=fg)


def plot(d, path=None, die=0, pos=None, max_cells=30_000, rows=None, labels=None, title=None, ax=None, dpi=150,
         nets=None, max_nets=None, tiles=None, bins=None, adjustment=None):
    """Draw one die of design d. pos: positions to draw (default d.pos). Above max_cells standard cells, the cells are
    drawn as an area-density image instead of rectangles. rows: draw the row outlines (default: when <= 300 rows).
    labels: macro names (default: when <= 40 macros). Colors: the logo's (THEME). nets: draw the connections between
    macros and IO pins (default: block-level designs, i.e. no standard cells), the max_nets strongest (default: 2 per
    macro, 40 to 400). tiles: a tile map in place of the cells, with its color bar:
      'gr_overflow'  global-routing load on OpenROAD's gcells: per gcell, the worst RUDY demand / capacity of its
                     edges (d.gr_overflow; adjustment as there); above 1 in red
      'pin_access'   the pin-access penalty of the PDN-aware detailed placement: w_pa per cell with a signal pin on
                     a PDN pin blockage, summed per tile (needs the 'pdn' fact)
      'density'      cell density as global placement sees it: movable area / free area per bin; above 1 in red
    bins: the tile grid of 'pin_access' and 'density' per side (default: global placement's, ~sqrt(#movable), halved
    until a bin spans two rows).
    Returns the matplotlib figure; saved to path if given."""
    from types import SimpleNamespace
    import torch
    d.require('plot')
    t = THEME
    own = pos is None                                    # drawing the design's own placement (HPWL applies)
    pos_t = None if own else torch.as_tensor(pos, device=d.device, dtype=d.pos.dtype)
    placed = d.placed.cpu().numpy() if pos is None else np.ones(len(d.flags), bool)  # caller's positions count as placed
    pos = (d.pos if pos is None else pos).detach().float().cpu().numpy()
    size = d.size.detach().cpu().numpy()
    flags = d.flags.cpu().numpy(); on = d.cell_die.cpu().numpy() == die
    die_box = d.die_box[die].tolist(); core = d.core_box[die].tolist()
    term = (flags & (TERMINAL | ASIDE)) != 0; macro = (flags & MACRO) != 0
    status, issues, outside = _legal_status(d, pos, own)
    outside = outside & on
    # the view: the die, grown to whatever is drawn outside it (macros a legalizer could not fit, ...)
    drawn = on & placed & ~((flags & ASIDE) != 0)
    view = list(die_box)
    if drawn.any():
        view = [min(view[0], float(pos[drawn, 0].min())), min(view[1], float(pos[drawn, 1].min())),
                max(view[2], float((pos[drawn, 0] + size[drawn, 0]).max())), max(view[3], float((pos[drawn, 1] + size[drawn, 1]).max()))]
    W, H = view[2] - view[0], view[3] - view[1]
    s = SimpleNamespace(own=own, placed=placed, pos=pos, size=size, flags=flags, on=on, term=term, macro=macro,
                        outside=outside, die_box=die_box, core=core, W=W, unit=d.units)
    if ax is None:
        fs = 8.0
        fig, ax = plt.subplots(figsize=(fs * min(1.6, max(0.6, W / H)) * (1.12 if tiles else 1), fs * 1.06), dpi=dpi)
    else:
        fig = ax.figure
    fig.patch.set_facecolor(t['surface']); ax.set_facecolor(t['surface'])

    # outlines: rows, core, die (in the 'bad' color when something sticks out of the core)
    rsel = (d.row_die == die).cpu().numpy() if d.row_box is not None else np.zeros(0, bool)
    if rsel.any() and (rows if rows is not None else (rsel.sum() <= 300 and not tiles)):   # rows under tiles: stripes
        rb = d.row_box.cpu().numpy()[rsel]
        ax.add_collection(PolyCollection(_rects(rb[:, :2], rb[:, 2:] - rb[:, :2]), facecolors='none', edgecolors=t['row'],
                                         lw=0.4, zorder=1))
    ax.add_patch(Rectangle(core[:2], core[2] - core[0], core[3] - core[1], fill=False, ec=t['core'], ls=(0, (3, 3)),
                           lw=0.7, zorder=5))
    ax.add_patch(Rectangle(die_box[:2], die_box[2] - die_box[0], die_box[3] - die_box[1], fill=False,
                           ec=t['bad'] if outside.any() else t['die'], lw=1.4 if outside.any() else 1.0, zorder=6))
    note = _draw_tiles(fig, ax, d, t, tiles, pos_t, bins, adjustment) if tiles else None
    legend = [] if tiles else _draw_cells(ax, d, t, s, max_cells)
    legend += _draw_macros(ax, d, t, s, nets, max_nets, labels)
    io_legend, n_io = _draw_io(ax, d, t, s)
    legend += io_legend

    pad = 0.02 * max(W, H)
    ax.set_xlim(view[0] - pad, view[2] + pad); ax.set_ylim(view[1] - 0.07 * H, view[3] + pad)
    ax.set_aspect('equal'); ax.axis('off')
    _scale_bar(ax, view, s.unit, t)
    if legend:                                           # on the scale bar's line, right-aligned under the drawing
        ax.legend(handles=legend, loc='center right', bbox_to_anchor=(view[2], view[1] - 0.035 * H),
                  bbox_transform=ax.transData, ncol=min(len(legend), 6), frameon=False, fontsize=8,
                  labelcolor=t['text2'], handlelength=1.2, columnspacing=1.4, borderaxespad=0, borderpad=0)
    _title_block(fig, ax, d, t, s, die, n_io, title, status, issues, note)
    fig.subplots_adjust(left=0.045, right=0.875 if tiles else 0.955, top=0.875 if tiles else 0.895, bottom=0.02)
    if path:
        fig.savefig(path, dpi=dpi, facecolor=t['surface'])
        plt.close(fig)
    return fig
