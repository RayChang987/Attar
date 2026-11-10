"""Design visualization: die, core, rows, standard cells, macros, IO pins (matplotlib, headless).

Two themes share one set of roles; 'light' (the default) is warm: standard cells in terracotta (rectangles, or an
area-density image from the paper surface to deep brick for large designs), macros in a neutral sand fill so the
cells stand out, IO pins by direction (input amber, output crimson, inout plum), thin outlines for die / core / rows, no axes: a scale
bar and a title block with the design's numbers instead."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from .distill import ASIDE, FIXED, MACRO, PHYSICAL, TERMINAL  # noqa: E402

THEMES = {
    # warm light: paper surface, terracotta cells, sand macros
    'light': dict(surface='#fbf8f3', text='#2b211c', text2='#6b5d52', muted='#a39486', die='#4a3b32', core='#cdbfae',
                  row='#efe7dc', cell='#d9622b', density=('#fbf8f3', '#f4c3a3', '#e07a45', '#c2501f'), fixed='#c9bba8', macro='#ecdfc8',
                  macro_edge='#a8946f', io=('#d99a1e', '#b8323f', '#7a4e8c', '#a39486')),
    'dark': dict(surface='#141413', text='#f0efec', text2='#c3c2b7', muted='#8a8984', die='#d6d4cc', core='#5c5b56',
                 row='#232321', cell='#3987e5', density=('#141413', '#3987e5', '#9cc7ff'), fixed='#5c5b56', macro='#3a3833',
                 macro_edge='#8f8a7d', io=('#199e70', '#d95926', '#9085e9', '#8a8984')),
}
IO_NAMES = ('input', 'output', 'inout', 'feedthru')


def _rects(xy, wh):
    x0, y0 = xy[:, 0], xy[:, 1]; x1, y1 = x0 + wh[:, 0], y0 + wh[:, 1]
    return np.stack([np.stack([x0, y0], 1), np.stack([x1, y0], 1), np.stack([x1, y1], 1), np.stack([x0, y1], 1)], 1)


def _scale_bar(ax, box, unit, t):
    """A scale bar of a round length about a fifth of the die width, under the die."""
    W = box[2] - box[0]
    raw = W / 5
    p = 10 ** np.floor(np.log10(raw))
    L = max(v * p for v in (1, 2, 5) if v * p <= raw)
    x0, y = box[0], box[1] - 0.035 * (box[3] - box[1])
    ax.plot([x0, x0 + L], [y, y], color=t['text2'], lw=1.6, solid_capstyle='butt', zorder=9, clip_on=False)
    ax.text(x0 + L + 0.012 * W, y, '%g %s' % (L, unit), va='center', ha='left', fontsize=8, color=t['text2'], clip_on=False)


def plot(d, path=None, die=0, pos=None, max_cells=30_000, rows=None, labels=None, title=None, ax=None, dpi=150,
         theme='light'):
    """Draw one die of design d. pos: positions to draw (default d.pos). Above max_cells standard cells, the cells are
    drawn as an area-density image instead of rectangles. rows: draw the row outlines (default: when <= 300 rows).
    labels: macro names (default: when <= 40 macros). theme: 'light' or 'dark'. Returns the matplotlib figure; saved
    to path if given."""
    d.require('plot')
    t = THEMES[theme]
    own = pos is None                                    # drawing the design's own placement (HPWL applies)
    placed = d.placed.cpu().numpy() if pos is None else np.ones(len(d.flags), bool)  # caller's positions count as placed
    pos = (d.pos if pos is None else pos).detach().float().cpu().numpy()
    size = d.size.detach().cpu().numpy()
    flags = d.flags.cpu().numpy(); on = d.cell_die.cpu().numpy() == die
    unit = d.units

    die_box = d.die_box[die].tolist(); core = d.core_box[die].tolist()
    W, H = die_box[2] - die_box[0], die_box[3] - die_box[1]
    if ax is None:
        s = 8.0
        fig, ax = plt.subplots(figsize=(s * min(1.6, max(0.6, W / H)), s * 1.04), dpi=dpi)
    else:
        fig = ax.figure
    fig.patch.set_facecolor(t['surface']); ax.set_facecolor(t['surface'])

    rsel = (d.row_die == die).cpu().numpy() if d.row_box is not None else np.zeros(0, bool)
    if rsel.any() and (rows if rows is not None else rsel.sum() <= 300):
        rb = d.row_box.cpu().numpy()[rsel]
        ax.add_collection(PolyCollection(_rects(rb[:, :2], rb[:, 2:] - rb[:, :2]), facecolors='none', edgecolors=t['row'],
                                         lw=0.4, zorder=1))
    ax.add_patch(Rectangle(core[:2], core[2] - core[0], core[3] - core[1], fill=False, ec=t['core'], ls=(0, (3, 3)),
                           lw=0.7, zorder=5))
    ax.add_patch(Rectangle(die_box[:2], W, H, fill=False, ec=t['die'], lw=1.0, zorder=6))

    term = (flags & (TERMINAL | ASIDE)) != 0; macro = (flags & MACRO) != 0
    std = on & ~term & ~macro & placed
    fixed_std = std & ((flags & (FIXED | PHYSICAL)) != 0)
    mov_std = std & ~fixed_std
    legend = []
    if int(mov_std.sum()) > max_cells:
        # area density of the standard cells, surface -> blue -> deep blue; empty bins transparent
        # a bin spans at least two rows (finer bins alias the row structure into stripes), at most 640 per side
        hb = float(d.row_h.min()) if d.row_h is not None else max(W, H) / 640
        bw = max(max(W, H) / 640, 2 * hb)
        bins = (max(16, int(round(W / bw))), max(16, int(round(H / bw))))
        xy = pos[mov_std] + size[mov_std] / 2
        img, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=[[die_box[0], die_box[2]], [die_box[1], die_box[3]]],
                                   weights=size[mov_std].prod(1))
        dens = np.clip(img / ((W / bins[0]) * (H / bins[1])), 0, 1).T
        def box(a):                                      # 3x3 box blur: the shape of the placement, not bin noise
            k = np.pad(a, 1, mode='edge')
            return sum(k[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9
        occ = box((dens > 0).astype(float))
        dens = box(dens) / np.maximum(occ, 1e-9)         # density where there are cells; single empty bins close up
        cmap = LinearSegmentedColormap.from_list('attar_cells', list(t['density']))
        rgba = cmap(dens)
        rgba[..., 3] = np.clip(occ * 2 - 0.25, 0, 1)      # soft edges, no pinholes
        ax.imshow(rgba, origin='lower', extent=[die_box[0], die_box[2], die_box[1], die_box[3]], interpolation='bilinear',
                  zorder=2)
        legend.append(Patch(fc=t['cell'], ec='none', label='cell density'))
    elif mov_std.any():
        thin = int(mov_std.sum()) > 3000
        ax.add_collection(PolyCollection(_rects(pos[mov_std], size[mov_std]), facecolors=t['cell'],
                                         edgecolors='none' if thin else t['surface'], lw=0 if thin else 0.25,
                                         alpha=0.9, zorder=2))
        legend.append(Patch(fc=t['cell'], ec='none', label='cells'))
    if fixed_std.any():
        ax.add_collection(PolyCollection(_rects(pos[fixed_std], size[fixed_std]), facecolors=t['fixed'], edgecolors='none',
                                         zorder=2))
        legend.append(Patch(fc=t['fixed'], ec='none', label='fixed cells'))
    msel = on & macro & placed
    if msel.any():
        ax.add_collection(PolyCollection(_rects(pos[msel], size[msel]), facecolors=t['macro'], edgecolors=t['macro_edge'],
                                         lw=0.6, zorder=3))
        legend.append(Patch(fc=t['macro'], ec=t['macro_edge'], lw=0.6, label='macros'))
        if labels if labels is not None else msel.sum() <= 40:
            names = d.names['cells']
            for i in np.nonzero(msel)[0]:
                if size[i, 0] < 0.08 * W:                # too narrow for its name
                    continue
                parts = [x for x in names[i].replace('/', '.').split('.') if x]
                ax.text(pos[i, 0] + size[i, 0] / 2, pos[i, 1] + size[i, 1] / 2, '.'.join(parts[-2:])[-24:],
                        ha='center', va='center', fontsize=5, color=t['text2'], zorder=4, clip_on=True)
    io_all = d.io.cpu().numpy() if d.io is not None else np.zeros(0, np.int64)
    sel = placed[io_all] & on[io_all] if len(io_all) else np.zeros(0, bool)      # unplaced IO pins are not drawn
    io = io_all[sel]
    io_dir = d.io_dir.cpu().numpy()[sel] if len(io_all) else np.zeros(0, np.int64)
    ms = max(2.0, min(14.0, 2500.0 / max(1, len(io))))
    for k in range(4):
        s_ = io[io_dir == k]
        if len(s_):
            ax.scatter(pos[s_, 0], pos[s_, 1], s=ms, c=t['io'][k], marker='o', zorder=7, linewidths=0)
            legend.append(Line2D([], [], ls='none', marker='o', ms=5, mfc=t['io'][k], mec='none', label='IO %s' % IO_NAMES[k]))

    pad = 0.02 * max(W, H)
    ax.set_xlim(die_box[0] - pad, die_box[2] + pad); ax.set_ylim(die_box[1] - 0.07 * H, die_box[3] + pad)
    ax.set_aspect('equal'); ax.axis('off')
    _scale_bar(ax, die_box, unit, t)
    if legend:                                           # on the scale bar's line, right-aligned under the die
        ax.legend(handles=legend, loc='center right', bbox_to_anchor=(die_box[2], die_box[1] - 0.035 * H),
                  bbox_transform=ax.transData, ncol=min(len(legend), 6), frameon=False, fontsize=8,
                  labelcolor=t['text2'], handlelength=1.2, columnspacing=1.4, borderaxespad=0, borderpad=0)

    c = d.manifest['counts']
    hidden = int((on & ~placed & ~term & ~macro).sum())
    if hidden:
        ax.text(0.5, 0.5, '%s unplaced cells not drawn' % '{:,}'.format(hidden), transform=ax.transAxes, ha='center',
                va='center', fontsize=10, color=t['muted'], zorder=8)
    n_std = int((on & ~term & ~macro).sum())
    core_area = (core[2] - core[0]) * (core[3] - core[1])
    fill = ('utilization %.0f%%' % (100 * d.utilization()[die].item()) if n_std else
            'block area %.0f%%' % (100 * float(size[on & macro].prod(1).sum()) / core_area))
    stats = ['%s cells' % '{:,}'.format(c['cells'])]
    if c['macros']:
        stats.append('%s macros' % '{:,}'.format(c['macros']))
    if len(io):
        stats.append('%s IO pins' % '{:,}'.format(len(io)))
    num = lambda v: '{:,.0f}'.format(v) if abs(v) >= 1000 else '%.4g' % v
    stats += ['%s x %s %s' % (num(W), num(H), unit), fill]
    if own and 'hpwl' in d.capabilities():
        stats.append('HPWL %s %s' % (num(float(d.hpwl())), unit))
    head = title if title is not None else d.manifest.get('design', '')
    fig.text(0.04, 0.975, head, ha='left', va='top', fontsize=13, fontweight='bold', color=t['text'])
    fig.text(0.04, 0.938, '   '.join(stats), ha='left', va='top', fontsize=8.5, color=t['text2'])
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.02)
    if path:
        fig.savefig(path, dpi=dpi, facecolor=t['surface'])
        plt.close(fig)
    return fig
