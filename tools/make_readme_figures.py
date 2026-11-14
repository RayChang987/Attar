#!/usr/bin/env python3
"""Figures for README.md (docs/images):
  proxy_agreement.png               how often routed WL moves the same way as HPWL, by HPWL gap (pa_pdn, spec 7.1)
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
THEME = {'light': dict(surface='#fbf8f3', text='#2b211c', text2='#6b5d52', grid='#ece3d6', series='#d9622b')}


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
    if '--gallery' in sys.argv:
        gallery(sys.argv[sys.argv.index('--gallery') + 1:])
