#!/usr/bin/env python3
"""ggr.cu against OpenROAD's global route, per design of orfs_runs/grtcorr (made by orfs_grt.py).

Per design and setting (ORFS's adjustment; with --stress also the raised one):
  capacity: ggr_attar's (from LEF/DEF) against OpenROAD's dbGCellGrid capacity, per layer and edge;
  usage:    per gcell edge, 2D (layers summed per direction) and per layer: Pearson, Spearman, total ratio;
  hotspots: the gcell edges OpenROAD fills to >= 90% of capacity (2D): how many ggr also puts there (recall),
            and of ggr's, how many OpenROAD has (precision);
  overflow: total 2D overflow both ways.
usage: compare.py [design ...] [--json out.json]"""
import argparse
import glob
import json
import os
import sys

import numpy as np

RUNS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'orfs_runs', 'grtcorr'))


def read_segs(path, meta):
    """OpenROAD's write_global_route_segments -> usage [L][ny][nx] (nets per unit gcell edge and layer) and
    unit edges per net. A wire on a horizontal layer from gcell x=a to b uses edges a..b-1 of its row."""
    L, nx, ny, tile, x0, y0 = meta['layers'], meta['nx'], meta['ny'], meta['tile'], meta['x0'], meta['y0']
    use = np.zeros((L, ny, nx), np.float64)
    per_net = {}
    name, edges = None, set()
    for line in open(path):
        t = line.split()
        if not t or t[0] == '(':
            continue
        if t[0] == ')':
            for (l, d, x, y) in edges:
                use[l, y, x] += 1
            per_net[name] = len(edges)
            edges = set(); continue
        if len(t) == 1:
            name = t[0]; continue
        x1, y1, l1, x2, y2, l2 = int(t[0]), int(t[1]), t[2], int(t[3]), int(t[4]), t[5]
        if l1 != l2:
            continue
        l = int(l1[1:]) - 1
        a = (min(x1, x2) - x0) // tile, (min(y1, y2) - y0) // tile
        b = (max(x1, x2) - x0) // tile, (max(y1, y2) - y0) // tile
        if a[1] == b[1]:
            edges |= {(l, 0, x, a[1]) for x in range(a[0], b[0])}
        else:
            edges |= {(l, 1, a[0], y) for y in range(a[1], b[1])}
    return use, per_net


def rank(a):
    r = np.empty(len(a)); r[np.argsort(a, kind='stable')] = np.arange(len(a))
    # average ties
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, r); return (sums / cnt)[inv]


def corr(a, b):
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float('nan'), float('nan')
    return float(np.corrcoef(a, b)[0, 1]), float(np.corrcoef(rank(a), rank(b))[0, 1])


def one(d, tag_or, tag_g, summ):
    dst = os.path.join(RUNS, d)
    meta = json.load(open(os.path.join(dst, tag_g + '.json')))
    L, nx, ny = meta['layers'], meta['nx'], meta['ny']
    shp = (L, ny, nx)
    gcap = np.fromfile(os.path.join(dst, tag_g + '.cap.i32'), np.int32).reshape(shp).astype(np.float64)
    guse = np.fromfile(os.path.join(dst, tag_g + '.use.i32'), np.int32).reshape(shp).astype(np.float64)
    ouse, onet = read_segs(os.path.join(dst, tag_or + '.segs'), meta)
    ocap = gcap   # ggr_attar's capacity; per layer it is checked against OpenROAD's resource table below
    lo, hi = meta['min_layer'] - 1, meta['max_layer']
    out = {'grid': [nx, ny]}
    rep = summ[tag_or]['layers']
    names = ['M%d' % (l + 1) for l in range(lo, hi)]
    out['cap_layers_equal'] = sum(int(gcap[l].sum()) == rep.get('M%d' % (l + 1), [None])[0] for l in range(lo, hi))
    out['cap_layers'] = len(names)
    out['cap_total_ratio'] = float(gcap[lo:hi].sum() / max(1, sum(rep[n][0] for n in names if n in rep)))
    # the report's demand per layer against our count of the segments (a check of read_segs)
    out['demand_report_vs_segs'] = [[rep.get(n, [0, 0])[1], int(ouse[l].sum())] for l, n in zip(range(lo, hi), names)]
    # per net unit edges
    gn = {}
    for line in open(os.path.join(dst, tag_g + '.nets.tsv')):
        t = line.rstrip('\n').split('\t')
        if t[0] != 'net':
            gn[t[0]] = int(t[3])
    common = [n for n in gn if n in onet and (gn[n] or onet[n])]
    a = np.array([gn[n] for n in common], float); b = np.array([onet[n] for n in common], float)
    out['net_pearson'], out['net_spearman'] = corr(a, b)
    out['net_exact_frac'] = float((a == b).mean()) if len(a) else None
    out['wl_ratio'] = float(a.sum() / max(1, b.sum()))
    # usage per direction (2D): sum layers of each direction
    hz = np.array([c == 'H' for c in meta['ldir']])
    sel = np.zeros(L, bool); sel[lo:hi] = True
    g2 = np.stack([guse[sel & hz].sum(0), guse[sel & ~hz].sum(0)]).ravel()
    o2 = np.stack([ouse[sel & hz].sum(0), ouse[sel & ~hz].sum(0)]).ravel()
    c2 = np.stack([ocap[sel & hz].sum(0), ocap[sel & ~hz].sum(0)]).ravel()
    m = (c2 > 0) | (o2 > 0) | (g2 > 0)
    out['use2d_pearson'], out['use2d_spearman'] = corr(g2[m], o2[m])
    out['use_total_ratio'] = float(g2.sum() / max(1, o2.sum()))
    # per layer (3D)
    g3, o3 = guse[lo:hi].ravel(), ouse[lo:hi].ravel()
    m3 = (ocap[lo:hi].ravel() > 0) | (o3 > 0) | (g3 > 0)
    out['use3d_pearson'], out['use3d_spearman'] = corr(g3[m3], o3[m3])
    out['layer_share_or'] = [round(float(ouse[l].sum() / max(1, ouse[lo:hi].sum())), 3) for l in range(lo, hi)]
    out['layer_share_ggr'] = [round(float(guse[l].sum() / max(1, guse[lo:hi].sum())), 3) for l in range(lo, hi)]
    # hotspots (2D utilization against OpenROAD's capacity)
    hot_o = (c2 > 0) & (o2 >= 0.9 * c2)
    hot_g = (c2 > 0) & (g2 >= 0.9 * c2)
    out['hot_or'], out['hot_ggr'] = int(hot_o.sum()), int(hot_g.sum())
    out['hot_recall'] = float((hot_o & hot_g).sum() / hot_o.sum()) if hot_o.sum() else None
    out['hot_precision'] = float((hot_o & hot_g).sum() / hot_g.sum()) if hot_g.sum() else None
    out['ovf2d_or'] = int(np.maximum(0, o2 - c2).sum())
    out['ovf2d_ggr'] = int(np.maximum(0, g2 - c2).sum())
    out['ovf3d_or'] = int(np.maximum(0, ouse[lo:hi] - ocap[lo:hi]).sum())
    out['ovf3d_ggr'] = int(np.maximum(0, guse[lo:hi] - ocap[lo:hi]).sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='*')
    ap.add_argument('--json')
    a = ap.parse_args()
    designs = a.designs or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(RUNS + '/*/summary.json'))
    rows = {}
    for d in designs:
        s = json.load(open(os.path.join(RUNS, d, 'summary.json')))
        rows[d] = {'summary': s, 'fresh': one(d, 'fresh', 'ggr', s)}
        if 'stress' in s:
            rows[d]['stress'] = one(d, 'stress', 'ggr_stress', s)
    hdr = '%-20s %-7s %9s %5s | %6s %6s %6s %6s | %6s %6s | %5s %5s %5s | %8s %8s | %7s %7s'
    print(hdr % ('design', 'set', 'gcells', 'capEq', 'use2dP', 'use2dS', 'use3dP', 'useR', 'netP', 'netS', 'hotO', 'rec', 'prec',
                 'ovfOR', 'ovfGGR', 'OR s', 'ggr s'))
    for d, r in rows.items():
        s = r['summary']
        for tag in ('fresh', 'stress'):
            if tag not in r:
                continue
            x = r[tag]
            if 'error' in x:
                print('%-20s %-7s %s' % (d, tag, x['error'])); continue
            gk = 'ggr' if tag == 'fresh' else 'ggr_stress'
            f = lambda v: '-' if v is None else '%.3f' % v  # noqa: E731
            print(hdr % (d, tag, x['grid'][0] * x['grid'][1], '%d/%d' % (x['cap_layers_equal'], x['cap_layers']),
                         f(x['use2d_pearson']), f(x['use2d_spearman']), f(x['use3d_pearson']), f(x['use_total_ratio']),
                         f(x['net_pearson']), f(x['net_spearman']),
                         x['hot_or'], f(x['hot_recall']), f(x['hot_precision']), x['ovf2d_or'], x['ovf2d_ggr'],
                         '%.2f' % s[tag]['gr_s'], '%.2f' % s[gk]['route_s']))
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
