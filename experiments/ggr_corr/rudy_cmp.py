#!/usr/bin/env python3
"""RUDY against ggr.cu, both against OpenROAD's global route, on the designs of orfs_runs/grtcorr.

The same DEF (grt.def: what ORFS routed) is read into Attar; RUDY is computed on OpenROAD's gcell grid
(tile and origin from ggr_attar's .json, so the bins are the gcells exactly), split by direction: a net's
horizontal demand density is 1 / bbox height over its bbox, the vertical one 1 / bbox width (bbox at least one
gcell each way). Per gcell and direction, against OpenROAD's 2D usage (from its segments):
  Pearson and Spearman of the demand maps,
  hotspots: the 5% of gcell edges with the highest utilization (usage / capacity, the same capacity for all three)
  by OpenROAD, and how many of them each predictor's top 5% recovers.
Only designs whose capacity equals OpenROAD's on every layer are meaningful for ggr (no macros).
usage: rudy_cmp.py [design ...] [--json out]"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools')); sys.path.insert(0, HERE)
import attar  # noqa: E402
from attar import metrics as M  # noqa: E402
from attar.kernels import Grid  # noqa: E402
from compare import RUNS, corr, read_segs  # noqa: E402
from orfs_roundtrip import base_dir, inputs  # noqa: E402

SCRATCH = os.environ.get('GRTCORR_IR', '/tmp/claude-1000/-home-jiarui-attar/6a69536c-081b-4c8f-949e-0bc97f7756a0/scratchpad/grtcorr_ir')


def rudy_hv(d, meta):
    """Directional RUDY on the gcell grid: (H, V) maps [ny][nx] in gcell-edge units (tracks' worth of wire per gcell)."""
    ev = attar.Evaluator(d)
    ev._prepare()
    s = 1.0 / meta['dbu']
    t = meta['tile'] * s
    x0, y0, nx, ny = meta['x0'] * s, meta['y0'] * s, meta['nx'], meta['ny']
    g = Grid.over_box([x0, y0, x0 + nx * t, y0 + ny * t], nx, ny, d.pos.device)
    p = ev._pins(None)
    lo, hi = M.net_bbox(p, ev.net, ev.E)
    ok = ev.deg >= 2
    lo, hi, w = lo[ok], hi[ok], ev.wmask[ok]
    c = (lo + hi) / 2
    sz = torch.maximum(hi - lo, torch.tensor([t, t], device=p.device))
    lo, hi = c - sz / 2, c + sz / 2
    # wire length per unit area by direction, times the gcell area / gcell length = nets crossing a gcell
    mh = g.accumulate(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1], w / sz[:, 1]) / g.bin_area * t
    mv = g.accumulate(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1], w / sz[:, 0]) / g.bin_area * t
    return mh.cpu().numpy().T, mv.cpu().numpy().T   # kernels.Grid maps are [nx][ny]


def load(dname, dev):
    ir = os.path.join(SCRATCH, dname)
    if not os.path.exists(os.path.join(ir, 'manifest.json')):
        lefs, _ = inputs(base_dir(dname))
        attar.distill(os.path.join(RUNS, dname, 'grt.def'), lefs, ir)
    return attar.load(ir, dev)


def topk_recall(truth, pred, valid, frac=0.05):
    idx = np.flatnonzero(valid)
    k = max(1, int(len(idx) * frac))
    t = set(idx[np.argsort(-truth[idx], kind='stable')[:k]])
    p = set(idx[np.argsort(-pred[idx], kind='stable')[:k]])
    return len(t & p) / k


def one(dname, dev):
    dst = os.path.join(RUNS, dname)
    out = {}
    d = load(dname, dev)
    for tag, gtag in (('fresh', 'ggr'), ('stress', 'ggr_stress')):
        meta = json.load(open(os.path.join(dst, gtag + '.json')))
        L, nx, ny = meta['layers'], meta['nx'], meta['ny']
        cap = np.fromfile(os.path.join(dst, gtag + '.cap.i32'), np.int32).reshape(L, ny, nx).astype(float)
        guse = np.fromfile(os.path.join(dst, gtag + '.use2d.i32'), np.int32).reshape(2, ny, nx).astype(float)
        ouse3, _ = read_segs(os.path.join(dst, tag + '.segs'), meta)
        hz = np.array([c == 'H' for c in meta['ldir']]); sel = np.zeros(L, bool); sel[meta['min_layer'] - 1:meta['max_layer']] = True
        ouse = np.stack([ouse3[sel & hz].sum(0), ouse3[sel & ~hz].sum(0)])
        c2 = np.stack([cap[sel & hz].sum(0), cap[sel & ~hz].sum(0)])
        if tag == 'fresh':
            rh, rv = rudy_hv(d, meta)
        rudy = np.stack([rh, rv])
        valid = (c2 > 0).ravel()
        o, g, r, c = ouse.ravel(), guse.ravel(), rudy.ravel(), c2.ravel()
        res = {}
        for name, pred in (('ggr', g), ('rudy', r)):
            pp, ps = corr(pred[valid], o[valid])
            res[name] = {'pearson': pp, 'spearman': ps,
                         'hot5_recall': topk_recall(o / np.maximum(c, 1), pred / np.maximum(c, 1), valid),
                         'hot1_recall': topk_recall(o / np.maximum(c, 1), pred / np.maximum(c, 1), valid, 0.01)}
        res['rudy_total_ratio'] = float(r[valid].sum() / max(1, o[valid].sum()))
        out[tag] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='*')
    ap.add_argument('--json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    designs = a.designs or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(RUNS + '/*/summary.json'))
    rows = {}
    hdr = '%-20s %-6s | %7s %7s %7s %7s | %7s %7s %7s %7s | %6s'
    print(hdr % ('design', 'set', 'ggr P', 'ggr S', 'ggr h5', 'ggr h1', 'rudy P', 'rudy S', 'rudy h5', 'rudy h1', 'rudyR'))
    for dn in designs:
        rows[dn] = one(dn, dev)
        for tag, r in rows[dn].items():
            print(hdr % (dn, tag, *('%.3f' % r['ggr'][k] for k in ('pearson', 'spearman', 'hot5_recall', 'hot1_recall')),
                         *('%.3f' % r['rudy'][k] for k in ('pearson', 'spearman', 'hot5_recall', 'hot1_recall')),
                         '%.2f' % r['rudy_total_ratio']), flush=True)
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
