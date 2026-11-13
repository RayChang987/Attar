#!/usr/bin/env python3
"""Placement ranking: across several placements of one design, which placement-stage predictor (HPWL, RUDY, ggr.cu)
orders them as ORFS's global routing does?

Per design (zoo after_fp/orfs_asap7, read only) and placement variant:
  orfs        ORFS's own global placement (the zoo reference)
  perturb     it with 10% of the cells moved by up to +-3 sites
  attar       Attar's place() at ORFS's target density; attar_dense at 0.95; attar_sparse at 0.6;
  attar_route Attar's place() with routability (DREAMPlace cell inflation)
predictors on the placement handed to ORFS: HPWL, RUDY (top 1% / 5% bins), ggr WL (ORFS adjustment) and ggr
overflow (2D and 3D, adjustment 0.7);
truth: ORFS continues from it (3_4 .. 5_1_grt, its own make): ORFS's GR wirelength; then on 5_1_grt.odb OpenROAD
global_route with ORFS's settings (wirelength) and with adjustment 0.7 (overflow).
usage: placement_rank.py <design> ... [--par 6] [--cores 3]"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools')); sys.path.insert(0, HERE)
import attar  # noqa: E402
import orfs_roundtrip as R  # noqa: E402
from orfs_grt import GGR, dump  # noqa: E402

OUT = os.path.join(R.RUNS, 'grtrank')
STRESS = 0.7
VARIANTS = ['orfs', 'perturb', 'attar', 'attar_dense', 'attar_sparse', 'attar_route']


def make_placement(d, v, dev):
    x = attar.load(os.path.join(ROOT, 'zoo/after_fp/orfs_asap7', d), dev)
    if v in ('orfs', 'perturb'):
        x.restore('reference')
        if v == 'perturb':
            R.move(x, 0.10, 3, 0)
    else:
        kw = {'attar': {}, 'attar_dense': {'target_density': 0.95}, 'attar_sparse': {'target_density': 0.6},
              'attar_route': {'gp_args': {'routability': True}}}[v]
        x.place(verbose=False, **kw)
    return x


def ggr_run(def_path, prefix, routing, lefs, adj):
    tech = [l for l in lefs if 'tech' in os.path.basename(l)][0]
    lo, hi = (int(s[1:]) for s in routing['layers'])
    args = [GGR, def_path, prefix, '--tech', tech, '--min', str(lo), '--max', str(hi), '--adj', str(adj)]
    args += sum((['--lef', l] for l in lefs if l != tech), [])
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    return json.load(open(prefix + '.json'))


def predictors(d, v, dst, dev, routing, lefs):
    x = make_placement(d, v, dev)
    out_def = os.path.join(dst, v + '.def')
    x.export(out_def)
    ev = attar.Evaluator(x)
    r = ev.rudy()
    g = ggr_run(out_def, os.path.join(dst, v + '_ggr'), routing, lefs, routing['adjustment'])
    gs = ggr_run(out_def, os.path.join(dst, v + '_ggrs'), routing, lefs, STRESS)
    return out_def, {'hpwl': ev.hpwl().item(), 'rudy_top1': r['top1%'], 'rudy_top5': r['top5%'], 'rudy_max': r['max'],
                     'ggr_wl': g['wirelength_gcells'], 'ggr_ovf2d': g['overflow_2d'],
                     'ggrs_ovf2d': gs['overflow_2d'], 'ggrs_ovf3d': gs['overflow_3d'],
                     'ggr_s': g['setup_s'] + g['route_s']}


def truth(d, v, dst, out_def, cores, routing):
    base = R.base_dir(d)
    rt = os.path.join(OUT, d, '_rt')
    os.makedirs(rt, exist_ok=True)
    work, _, t3 = R._inject(d, base, rt, v, out_def, 0.0, '3_3_place_gp')
    t = time.time()
    ok, _ = R.finish(d, work, cores, 'grt')
    r = R.flow_dir(work)
    if not ok or os.stat(os.path.join(r, '3_3_place_gp.odb')).st_mtime_ns != t3:
        raise RuntimeError('ORFS failed or redid global placement (%s)' % work)
    m = json.load(open(os.path.join(R.flow_dir(work, 'logs'), '5_1_grt.json')))
    sub = os.path.join(dst, v + '_or'); os.makedirs(sub, exist_ok=True)
    res = dump(d, r, sub, routing, STRESS, cores)
    for f in ('grt.def', 'fresh.segs', 'stress.segs'):
        p = os.path.join(sub, f)
        os.path.exists(p) and os.remove(p)
    shutil.rmtree(work, ignore_errors=True)
    return {'orfs_grt_wl': m.get('globalroute__global_route__wirelength'), 'or_wl': res['fresh']['wl_um'],
            'or_ovf': res['fresh']['overflow'][2] if res['fresh']['overflow'] else None,
            'or_stress_ovf': res['stress']['overflow'][2] if res['stress']['overflow'] else None,
            'flow_s': round(time.time() - t)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--par', type=int, default=6)
    ap.add_argument('--cores', type=int, default=3)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    jobs = []
    for d in a.designs:
        dst = os.path.join(OUT, d); os.makedirs(dst, exist_ok=True)
        routing = json.load(open(os.path.join(ROOT, 'zoo/after_fp/orfs_asap7', d, 'manifest.json')))['routing']
        lefs, _ = R.inputs(R.base_dir(d))
        for v in VARIANTS:
            f = os.path.join(dst, v + '.json')
            if os.path.exists(f):
                continue
            out_def, pr = predictors(d, v, dst, dev, routing, lefs)   # GPU work, one at a time
            print('%s %s predictors %s' % (d, v, {k: round(val, 3) for k, val in pr.items()}), flush=True)
            jobs.append((d, v, dst, out_def, pr, routing))

    def run(j):
        d, v, dst, out_def, pr, routing = j
        try:
            tr = truth(d, v, dst, out_def, a.cores, routing)
            json.dump({'design': d, 'variant': v, 'pred': pr, 'truth': tr}, open(os.path.join(dst, v + '.json'), 'w'), indent=1)
            print('%s %s truth %s' % (d, v, tr), flush=True)
        except Exception as e:  # noqa: BLE001
            print('%s %s FAILED %s' % (d, v, str(e)[-500:]), flush=True)
    with ThreadPoolExecutor(a.par) as P:
        list(P.map(run, jobs))


if __name__ == '__main__':
    main()
