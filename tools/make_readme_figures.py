#!/usr/bin/env python3
"""Figures for README.md (docs/images):
  proxy_agreement.png               how often routed WL moves the same way as HPWL, by HPWL gap (pa_pdn, spec 7.1)
  speed_vs_openroad.png             timing in the placement loop, Attar GPU vs OpenROAD kept warm
                                    (docs/validation/speed_vs_openroad.md)
  gallery/*.png                     what the Evaluator sees, on designs placed by attar.naive (run with
                                    --gallery [name ...]: all, or those figures, e.g. --gallery eval_grov; needs the
                                    zoo and a GPU)"""
import io
import contextlib
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
IMG = os.path.join(ROOT, 'docs', 'images')

# pa_pdn: 951 routed Innovus runs (NS3K), all placement pairs within a design group, binned by their HPWL gap
GAPS = ['<0.1%', '0.1-0.25%', '0.25-0.5%', '0.5-1%', '1-2%', '2-5%', '>5%']
AGREE = [55, 69, 74, 84, 86, 97.5, 99.8]
THEME = {'light': dict(surface='#fbf8f3', text='#2b211c', text2='#6b5d52', grid='#ece3d6', series='#d9622b',
                       base='#b9a993', muted='#a39486')}

# Timing in the placement loop, Attar (GPU) vs OpenROAD kept warm in one session, same machine (RTX 4090, 32 cores,
# 8 threads each), median of 3 rounds (docs/validation/timing_routing.md, "Inside the placement loop"):
# design, cells, GP update (every cell moved) OpenROAD s / Attar s, DP round (1% moved + legalized) OpenROAD s / Attar s
LOOP = [('netcard_fast', 298384, 7.249, 0.3431, 5.190, 0.2421),
        ('superblue18', 744947, 13.011, 0.5609, 11.519, 0.3131),
        ('superblue1', 1159346, 19.568, 0.8311, 16.223, 0.4692),
        ('superblue7', 1865884, 33.606, 1.2228, 30.582, 0.5662)]


def speed_vs_openroad(mode):
    """Seconds per timing update inside global placement, OpenROAD vs Attar, one row per design (largest at the
    bottom), and the speedups of a GP update and of a DP round in two columns on the right."""
    t = THEME[mode]
    W, H = 9.2, 4.9
    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor(t['surface'])
    L, B, Wm, Hm = 0.175, 0.19, 0.53, 0.58                  # the bar axes, in figure fractions
    ax = fig.add_axes([L, B, Wm, Hm])
    sp = fig.add_axes([L + Wm + 0.03, B, 1 - (L + Wm + 0.03) - 0.02, Hm], sharey=ax)
    n, h, gap, xmax = len(LOOP), 0.30, 0.05, 40.0
    ax.set_xlim(0, xmax); ax.set_ylim(-0.55, n - 0.45)
    sp.set_xlim(0, 2)
    asp = (n / (Hm * H)) / (xmax / (Wm * W))               # y data units per x data unit, on paper
    rad = (h / 2) / asp                                    # a radius of half the bar height, in x units
    for a in (ax, sp):
        a.set_facecolor(t['surface'])
        for s_ in a.spines.values():
            s_.set_visible(False)
        a.tick_params(length=0)
    for i, (name, cells, or_gp, at_gp, or_dp, at_dp) in enumerate(LOOP):
        y = n - 1 - i
        for yy, v, col, who in ((y + (h + gap) / 2, or_gp, t['base'], 'OpenROAD'),
                                (y - (h + gap) / 2, at_gp, t['series'], 'Attar (GPU)')):
            # one rounded patch from x = -rad: the axes clip its left end square at the baseline
            ax.add_patch(FancyBboxPatch((-rad, yy - h / 2), v + rad, h, boxstyle='round,pad=0,rounding_size=%g' % rad,
                                        mutation_aspect=asp, fc=col, ec='none', clip_on=True))
            lab = ('%.1f s' % v) if v >= 1 else ('%.2f s' % v)
            att = who.startswith('Attar')
            ax.text(v + 0.5, yy, lab, va='center', ha='left', fontsize=9.5, color=t['text'] if att else t['text2'],
                    fontweight='bold' if att else 'normal')
            if i == 0:                                      # direct series labels on the first row
                ax.text(v + 4.6, yy, who, va='center', ha='left', fontsize=9, color=t['series'] if att else t['text2'])
        for x_, s_ in ((0.5, or_gp / at_gp), (1.5, or_dp / at_dp)):
            sp.text(x_, y, '%.0f×' % s_, ha='center', va='center', fontsize=18, color=t['series'], fontweight='bold')
    ax.set_yticks(range(n))
    ax.set_yticklabels(['%s\n%.2fM cells' % (r[0], r[1] / 1e6) for r in LOOP][::-1], fontsize=10, color=t['text'],
                       linespacing=1.5)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xticklabels(['0', '10 s', '20 s', '30 s', '40 s'], fontsize=9, color=t['text2'])
    ax.xaxis.grid(True, color=t['grid'], lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('seconds per timing update in global placement (every cell moved)', fontsize=9.5,
                  color=t['text2'], labelpad=7)
    sp.set_xticks([0.5, 1.5])
    sp.set_xticklabels(['GP update', 'DP round'], fontsize=9.5, color=t['text2'])
    sp.xaxis.set_ticks_position('top')
    sp.tick_params(labelleft=False, pad=8)
    sp.text(1.0, n - 0.45 + 0.62, 'Attar is faster by', ha='center', va='bottom', fontsize=9.5, color=t['text'])
    fig.text(0.025, 0.94, 'Timing inside the placement loop: 21-54× faster than OpenROAD', ha='left', va='center',
             fontsize=14, color=t['text'])
    fig.text(0.025, 0.885, 'placement parasitics + setup WNS / TNS after every move; OpenROAD kept warm in its session', ha='left', va='center', fontsize=9.5, color=t['text2'])
    fig.text(0.025, 0.045, 'One machine, 8 CPU threads each; Attar 1.0.6, OpenROAD (ORFS 26Q4); median of 3 '
             'rounds.\nDP round: 1% of the cells moved and legalized.  TNS agrees within 0.13%.', ha='left',
             va='center', fontsize=8, color=t['muted'], linespacing=1.5)
    out = os.path.join(IMG, 'speed_vs_openroad.png')
    fig.savefig(out, facecolor=t['surface']); plt.close(fig)
    print('wrote', out)


def proxy_agreement(mode):
    t = THEME[mode]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(t['surface']); ax.set_facecolor(t['surface'])
    w = 0.32
    for i, v in enumerate(AGREE):                       # thin columns, rounded data end, square at the baseline
        r = 0.06
        ax.add_patch(FancyBboxPatch((i - w / 2, 0), w, v, boxstyle='round,pad=0,rounding_size=%g' % r,
                                    mutation_aspect=100 / 1.2, fc=t['series'], ec='none'))
        ax.add_patch(Rectangle((i - w / 2, 0), w, min(v, 10), fc=t['series'], ec='none'))
        ax.text(i, v + 1.5, ('%g%%' % v), ha='center', va='bottom', fontsize=9, color=t['text'])
    ax.axhline(50, color=t['text2'], lw=1, ls=(0, (4, 3)))
    ax.text(0.5, 51.5, 'coin flip', ha='center', va='bottom', fontsize=8.5, color=t['text2'])
    ax.set_xlim(-0.6, len(GAPS) - 0.4); ax.set_ylim(0, 108)
    ax.set_xticks(range(len(GAPS))); ax.set_xticklabels(GAPS, fontsize=9, color=t['text2'])
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9, color=t['text2'])
    ax.yaxis.grid(True, color=t['grid'], lw=0.6); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel('HPWL difference between two placements of the same design', fontsize=9.5, color=t['text2'], labelpad=8)
    ax.set_title('Routed wirelength moves the same way as HPWL', loc='left', fontsize=12, color=t['text'], pad=22)
    ax.text(0, 1.02, 'share of placement pairs, 951 routed runs (pa_pdn, NS3K + Innovus)', transform=ax.transAxes,
            fontsize=9, color=t['text2'])
    fig.tight_layout()
    out = os.path.join(IMG, 'proxy_agreement.png')
    fig.savefig(out, facecolor=t['surface']); plt.close(fig)
    print('wrote', out)


def gallery(only=()):
    """What the Evaluator sees, one view per figure (attar.viz tiles), from real runs of attar.naive:
      eval_grov         cva6 (ASAP7) placed: global-routing load on OpenROAD's gcells (d.gr_overflow)
      eval_pin_access   ariane136 (NS3K) after global placement: cells with a signal pin on the PDN
      eval_placed       the same after legalization: legal, rails and PDN included
      fp_n300           GSRC n300 floorplanned by B*-tree packing: legality and the macro / IO connections"""
    sys.path.insert(0, ROOT)
    import time
    import attar
    from attar import naive
    out = os.path.join(IMG, 'gallery')
    os.makedirs(out, exist_ok=True)

    def quiet(fn):
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            r = fn()
        return r, time.time() - t

    def save(d, fn, title, **kw):
        d.plot(os.path.join(out, fn + '.png'), title=title, dpi=160, **kw).clf()
        print('wrote', fn, '|', title)

    if not only or 'eval_grov' in only:
        d = attar.load('after_fp/orfs_asap7/cva6', 'cuda')
        _, sec = quiet(d.place)
        g = d.gr_overflow()
        save(d, 'eval_grov', 'cva6    global-routing overflow after placement (%.1f s)' % sec, tiles='gr_overflow')
        print('   GROV %.2f%%' % g['grov_%'])
        del d
    if not only or {'eval_pin_access', 'eval_placed'} & set(only):
        d = attar.load('after_fp/ns3k_pl_u80/ariane136', 'cuda')
        ev = attar.Evaluator(d)
        _, sec = quiet(lambda: naive.global_place(d))
        save(d, 'eval_pin_access', 'ariane136    pin access after global placement (%.1f s)' % sec, tiles='pin_access')
        _, sec = quiet(lambda: naive.legalize(d))
        lg = ev.legality()
        save(d, 'eval_placed', 'ariane136    legalized (%.1f s)' % sec)
        print('   legality', lg)
        del d
    if not only or 'fp_n300' in only:
        d = attar.load('before_fp/gsrc_hard/n300', 'cuda')
        _, sec = quiet(lambda: d.floorplan(utilization=0.85, packing=True))
        save(d, 'fp_n300', 'n300    floorplan by B*-tree packing (%.1f s)' % sec)
        print('   legal', d.floorplan_legal(), 'HPWL %.4g' % d.hpwl().item())
        del d

if __name__ == '__main__':
    os.makedirs(IMG, exist_ok=True)
    proxy_agreement('light')
    speed_vs_openroad('light')
    if '--gallery' in sys.argv:
        gallery(sys.argv[sys.argv.index('--gallery') + 1:])
