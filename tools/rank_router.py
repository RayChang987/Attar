#!/usr/bin/env python3
"""Does Attar's global router rank placements as OpenROAD's global router does? (The README's HPWL claim, for GR.)

Per design (zoo after_fp/orfs_<platform>, same netlist throughout), placements:
  orfs        ORFS's global placement (the zoo reference, as distilled; not legalized)
  perturb     it with 10% of the cells moved by up to +-3 rows / 30 sites
  td<t>_s<k>  Attar's place() (global placement + legalization) at target density t, GP seed k
Truth: OpenROAD (ORFS image) reads the design's LEFs and the DEF Attar exports for the placement, routes with the
flow's settings (tools/gr_truth.py: platform fastroute.tcl, routing layers from the manifest, no pin_access,
-congestion_iterations 30 -allow_congestion) and reports its wirelength and overflow. Predictors: HPWL, Attar's
global_route wirelength. Pairwise agreement over all pairs of placements of a design: does the predictor order the
pair as OpenROAD's wirelength does, binned by how far apart OpenROAD puts them.
Cached in orfs_runs/gr_rank/<run key>/. usage: rank_router.py <design | platform/design> ... [--device cuda:1]"""
import argparse
import json
import os
import re
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import attar  # noqa: E402
from attar import route  # noqa: E402
from gr_truth import same_adjustments, CLK_LAYER, layer_adjustments  # noqa: E402
from orfs_roundtrip import PLATFORMS, RUNS, base_dir, inputs, orfs, split, w  # noqa: E402

OUT = os.path.join(RUNS, 'gr_rank')
VARIANTS = ['orfs', 'perturb', 'td0.60_s0', 'td0.75_s0', 'td0.75_s1', 'td0.90_s0', 'td0.95_s0']
BINS = [(0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 10)]


def placement(d, v, seed=0):
    """Set d's placement to variant v."""
    d.load_reference()
    if v == 'orfs':
        return
    if v == 'perturb':
        g = torch.Generator().manual_seed(seed)
        mv = torch.nonzero(d.movable).flatten().cpu()
        k = mv[torch.randperm(len(mv), generator=g)[:len(mv) // 10]]
        dx = (torch.randint(-30, 31, (len(k),), generator=g) * float(d.row_site_w.min()))
        dy = (torch.randint(-3, 4, (len(k),), generator=g) * float(d.row_h.min()))
        p = d.pos.clone()
        p[k.to(p.device), 0] += dx.to(p.device); p[k.to(p.device), 1] += dy.to(p.device)
        core = d.core_box[0]
        p[:, 0] = torch.maximum(torch.minimum(p[:, 0], core[2] - d.size[:, 0]), core[0])
        p[:, 1] = torch.maximum(torch.minimum(p[:, 1], core[3] - d.size[:, 1]), core[1])
        d.pos = torch.where(d.movable[:, None], p, d.pos)
        return
    m = re.match(r'td([\d.]+)_s(\d+)', v)
    d.unplace()
    d.place(verbose=False, target_density=float(m.group(1)), gp_args={'seed': int(m.group(2))})


def openroad_gr(key, def_path, out_json):
    if os.path.exists(out_json):
        return json.load(open(out_json))
    plat, design = split(key)
    lefs = inputs(base_dir(key))[0]
    d = attar.load(os.path.join(attar.paths.zoo_dir(), 'after_fp', 'orfs_' + plat, design), 'cpu')
    r = d.manifest['routing']
    tcl = ['read_lef %s' % w(l) for l in lefs] + ['read_def %s' % w(def_path),
           'set ::env(MIN_ROUTING_LAYER) %s' % r['layers'][0], 'set ::env(MAX_ROUTING_LAYER) %s' % r['layers'][1],
           'set ::env(MIN_CLK_ROUTING_LAYER) %s' % CLK_LAYER[plat],
           'set ::env(ROUTING_LAYER_ADJUSTMENT) %g' % r['adjustment'],
           'source %s%s/fastroute.tcl' % (PLATFORMS, plat),
           'set t0 [clock milliseconds]', 'global_route -congestion_iterations 30 -allow_congestion -verbose',
           'puts "GR_DONE [expr {([clock milliseconds] - $t0) / 1000.0}]"']
    path = os.path.join(RUNS, 'tmp_grrank_%s.tcl' % os.path.basename(out_json).replace('.', '_'))
    with open(path, 'w') as f:
        f.write('\n'.join(tcl) + '\nexit\n')
    try:
        log = orfs('/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad -no_init -no_splash -threads 4 %s' % w(path))
    finally:
        os.remove(path)
    g = re.findall(r'GR_DONE ([\d.]+)', log)
    wl = re.findall(r'Total wirelength:\s*([\d.]+)\s*um', log)
    ov = re.findall(r'Total\s+\d+\s+\d+\s+[\d.]+%\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)', log)
    if not g or not wl:
        raise RuntimeError(log[-3000:])
    res = {'wl': float(wl[-1]), 'overflow': sum(int(v) for v in ov[-1]) if ov else None, 'seconds': float(g[-1])}
    json.dump(res, open(out_json, 'w'))
    return res


def agreement(rows):
    """Per bin of OpenROAD's relative WL difference: pairs, and the share each predictor orders as OpenROAD does."""
    out = {b: {'pairs': 0, 'hpwl': 0, 'attar_gr': 0} for b in BINS}
    for vs in rows.values():
        ks = list(vs)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = vs[ks[i]], vs[ks[j]]
                t = a['or_wl'] - b['or_wl']
                rel = abs(t) / min(a['or_wl'], b['or_wl'])
                if t == 0:
                    continue
                for lo, hi in BINS:
                    if lo <= rel < hi:
                        o = out[(lo, hi)]
                        o['pairs'] += 1
                        o['hpwl'] += (a['hpwl'] - b['hpwl']) * t > 0
                        o['attar_gr'] += (a['attar_wl'] - b['attar_wl']) * t > 0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--json')
    a = ap.parse_args()
    rows = {}
    for key in a.designs:
        plat, design = split(key)
        out = os.path.join(OUT, key); os.makedirs(out, exist_ok=True)
        d = attar.load(os.path.join(attar.paths.zoo_dir(), 'after_fp', 'orfs_' + plat, design), a.device)
        rows[key] = {}
        for v in VARIANTS:
            cache = os.path.join(out, v + '.attar.json')
            dp = os.path.join(out, v + '.def')
            if os.path.exists(cache) and os.path.exists(dp):
                r = json.load(open(cache))
            else:
                placement(d, v)
                d.export(dp)
                oj = os.path.join(out, v + '.or.json')        # a new placement (GP is not bitwise reproducible):
                if os.path.exists(oj):                         # OpenROAD routes it again
                    os.remove(oj)
                hp = float(d.hpwl())
                mr = d.manifest['routing']                     # the zoo's adjustments are what OpenROAD routes with
                assert same_adjustments(mr, layer_adjustments(d, plat))
                route.global_route(d, verbose=False)            # warm-up
                rt, st = route.global_route(d, verbose=False)
                r = {'hpwl': hp, 'attar_wl': st['wirelength_um'], 'attar_overflow': st['overflow_3d'],
                     'attar_s': st['seconds']}
                json.dump(r, open(cache, 'w'))
            o = openroad_gr(key, dp, os.path.join(out, v + '.or.json'))
            r.update({'or_wl': o['wl'], 'or_overflow': o['overflow'], 'or_s': o['seconds']})
            rows[key][v] = r
            print('%-22s %-10s HPWL %12.0f  attar GR %9d (ovfl %5d, %.2fs)  OpenROAD GR %9d (ovfl %5s, %.2fs)  %+.2f%%'
                  % (key, v, r['hpwl'], r['attar_wl'], r['attar_overflow'], r['attar_s'], r['or_wl'], r['or_overflow'],
                     r['or_s'], 100 * (r['attar_wl'] - r['or_wl']) / r['or_wl']), flush=True)
    ag = agreement(rows)
    print('\nOpenROAD GR WL difference | pairs | HPWL agrees | Attar GR agrees')
    for (lo, hi), o in ag.items():
        n = max(1, o['pairs'])
        print('%5.0f%% - %5.0f%% | %5d | %5.1f%% | %5.1f%%' % (100 * lo, 100 * hi, o['pairs'], 100 * o['hpwl'] / n,
                                                              100 * o['attar_gr'] / n))
    if a.json:
        json.dump({'rows': rows, 'agreement': {'%g-%g' % k: v for k, v in ag.items()}}, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
