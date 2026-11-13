#!/usr/bin/env python3
"""Attar's global-routing congestion (d.gr_overflow: RUDY on OpenROAD's gcell capacities) against ORFS's global route.

Per design of orfs_runs/grtcorr (experiments/ggr_corr/orfs_grt.py: what ORFS routed, grt.def, and OpenROAD's routes
of it with ORFS's layer adjustment, 'fresh', and with every layer's adjustment raised to 0.7, 'stress'):
  attar:  grt.def distilled (orfs_runs/gr/_ir/<design>) with the 'routing' fact from the run's LEFs; d.gr_overflow at
          the scenario's adjustment;
  truth:  OpenROAD's capacity per layer (the Resource column of its congestion report) and its routed 2D usage per
          gcell edge (write_global_route_segments) against that capacity, and the total overflow it reported.
Reports per scenario: capacity agreement with OpenROAD (total and worst layer), Attar's GROV (RUDY) and ORFS's (routed),
the per-edge Pearson of RUDY demand with routed usage, and how many of ORFS's 5% most utilized edges Attar's 5% has.
usage: validate_gr.py [design ...] [--json out]"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import attar  # noqa: E402
from attar import routing as R  # noqa: E402
from orfs_roundtrip import RUNS, base_dir, inputs  # noqa: E402

GRT = os.path.join(RUNS, 'grtcorr')
WORK = os.path.join(RUNS, 'gr', '_ir')


def orfs_tables(design):
    """OpenROAD's congestion reports in the run log, in order (fresh, stress): ({layer: resource}, total overflow)."""
    out, cur = [], None
    for line in open(os.path.join(GRT, design, 'openroad.log')):
        t = line.split()
        if t[:2] == ['Layer', 'Resource']:
            cur = {}; continue
        if cur is not None and t and t[0] == 'Total':
            out.append((cur, int(t[-1]))); cur = None
        elif cur is not None and len(t) >= 3 and t[1].isdigit():
            cur[t[0]] = int(t[1])
    return out


def routed_usage(path, g, L):
    """OpenROAD's write_global_route_segments -> usage per layer and gcell edge: h [L][ny][nx-1], v [L][ny-1][nx]."""
    tile, nx, ny = g['tile'], g['nx'], g['ny']
    x0, y0 = g['die'][0], g['die'][1]
    h = np.zeros((L, ny, nx - 1)); v = np.zeros((L, ny - 1, nx))
    edges = set()
    for line in open(path):
        t = line.split()
        if t[:1] == [')']:                                   # a net's end: each unit edge it uses counts once
            for (l, d, x, y) in edges:
                (h if d == 0 else v)[l, y, x] += 1
            edges = set(); continue
        if not t or t[0] == '(' or len(t) == 1:
            continue
        x1, y1, l1, x2, y2, l2 = int(t[0]), int(t[1]), t[2], int(t[3]), int(t[4]), t[5]
        if l1 != l2:
            continue
        l = int(l1[1:]) - 1
        ax, ay = (min(x1, x2) - x0) // tile, (min(y1, y2) - y0) // tile
        bx, by = (max(x1, x2) - x0) // tile, (max(y1, y2) - y0) // tile
        if ay == by:
            edges |= {(l, 0, x, ay) for x in range(ax, bx) if x < nx - 1}
        else:
            edges |= {(l, 1, ax, y) for y in range(ay, by) if y < ny - 1}
    return h, v


def ir_for(design, routing):
    ir = os.path.join(WORK, design)
    lefs, _ = inputs(base_dir(design))
    if not os.path.exists(os.path.join(ir, 'manifest.json')):
        attar.distill(os.path.join(GRT, design, 'grt.def'), lefs, ir)
    if not os.path.exists(os.path.join(ir, 'routing.safetensors')):
        R.attach(ir, lefs, layers=routing['layers'], adjustment=routing['adjustment'])
    return ir


def one(design, dev):
    summ = json.load(open(os.path.join(GRT, design, 'summary.json')))
    routing = summ['routing']
    d = attar.load(ir_for(design, routing), dev)
    tables = orfs_tables(design)
    out = {}
    for (tag, adj), (resource, orfs_total) in zip((('fresh', routing['adjustment']), ('stress', 0.7)), tables):
        seg = os.path.join(GRT, design, tag + '.segs')
        if not os.path.exists(seg):
            continue
        t = time.time()
        r = d.gr_overflow(adjustment=adj, maps=True)
        sec = time.time() - t
        g = R.grid(d)
        names = d.names['layers']
        h3, v3 = r['cap3_h'], r['cap3_v']
        ours = {names[l]: int(h3[l].sum() + v3[l].sum()) for l in range(len(names))}
        cap_tot = sum(ours[k] for k in resource); ref_tot = sum(resource.values())
        cap_worst = max((abs(ours[k] - v) / v for k, v in resource.items() if v), default=0.0)
        uh, uv = routed_usage(seg, g, len(names))
        ch, cv = r['cap_h'], r['cap_v']
        truth = R.edge_overflow(uh.sum(0), uv.sum(0), ch, cv)
        dh = (r['demand_h'][:, :-1] + r['demand_h'][:, 1:]) / 2; dv = (r['demand_v'][:-1] + r['demand_v'][1:]) / 2
        dem = np.concatenate([dh.ravel(), dv.ravel()]); use = np.concatenate([uh.sum(0).ravel(), uv.sum(0).ravel()])
        cap = np.concatenate([ch.ravel(), cv.ravel()])
        ok = cap > 0
        pear = float(np.corrcoef(dem[ok], use[ok])[0, 1])
        k = max(1, int(ok.sum() * 0.05))
        idx = np.flatnonzero(ok)
        top = lambda x: set(idx[np.argsort(-x[idx] / cap[idx], kind='stable')[:k]])  # noqa: E731
        recall = len(top(use) & top(dem)) / k
        out[tag] = {'adjustment': adj, 'capacity_err_%': 100.0 * (cap_tot - ref_tot) / ref_tot,
                    'capacity_worst_layer_%': 100.0 * cap_worst, 'attar_grov_%': r['grov_%'],
                    'orfs_grov_%': truth['grov_%'], 'orfs_total_overflow': orfs_total,
                    'attar_total_overflow': r['total_overflow'], 'pearson': pear, 'hot5_recall': recall,
                    'attar_s': sec}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='*')
    ap.add_argument('--json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    designs = a.designs or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(GRT + '/*/summary.json'))
    hdr = '%-20s %-6s %4s | %8s %8s | %9s %9s %8s | %7s %6s | %6s'
    print(hdr % ('design', 'set', 'adj', 'cap err', 'cap worst', 'Attar GROV', 'ORFS GROV', 'ORFS ovfl', 'pearson',
                 'hot5', 'sec'))
    rows = {}
    for dn in designs:
        rows[dn] = one(dn, dev)
        for tag, r in rows[dn].items():
            print(hdr % (dn, tag, '%.2f' % r['adjustment'], '%.3f%%' % r['capacity_err_%'],
                         '%.3f%%' % r['capacity_worst_layer_%'], '%.2f%%' % r['attar_grov_%'], '%.2f%%' % r['orfs_grov_%'],
                         r['orfs_total_overflow'], '%.3f' % r['pearson'], '%.0f%%' % (100 * r['hot5_recall']),
                         '%.2f' % r['attar_s']), flush=True)
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
