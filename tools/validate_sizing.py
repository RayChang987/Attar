#!/usr/bin/env python3
"""Sizing against OpenROAD: the same cell swaps in Attar (d.resize, the new masters from the zoo's library bundle) and in
OpenROAD (replace_cell, the new masters from the bundle's Liberty / LEF files), then both time the design.

Per ORFS zoo design (after_fp/orfs_<platform>, at its reference placement = ORFS 3_3_place_gp):
  swaps:  a seeded random fraction of the cells whose equivalence class has other members, each to a random other
          member (drive strength, threshold-voltage flavor);
  attar:  d.resize(...) then d.timing();
  truth:  OpenROAD reads the run's 3_3_place_gp.odb, the bundle's LEF libraries (the masters the swaps need), the
          design's Liberty files then the bundle's, replace_cell for every swap, the SDC and setRC.tcl, runs
          estimate_parasitics -placement and dumps every endpoint's setup / hold slack.
Reports what tools/validate_sta.py reports, plus the number of swaps.
usage: validate_sizing.py <design> ... [--platform asap7] [--fraction 0.2] [--seed 0] [--json out]"""
import argparse
import json
import os
import random
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import attar  # noqa: E402
from attar import liberty  # noqa: E402
from orfs_roundtrip import RUNS, base_dir, flow_dir, orfs, run_key, w  # noqa: E402
import validate_sta as VS  # noqa: E402

DUMP_TCL = VS.DUMP_TCL


def swaps_for(d, fraction, seed):
    """[(cell name, new master)]: a seeded random fraction of the cells with alternatives in their class."""
    rng = random.Random(seed)
    L = d.library()[-1]
    cand = []
    for c, m in enumerate(d.raw['c_master']):
        if m < 0:
            continue
        eq = [x for x in d.equivalents(c, dont_use=True) if x in L.cid and L.a['lc_lef'][L.cid[x]]]
        cur = d.names['masters'][m]
        others = [x for x in eq if x != cur]
        if others:
            cand.append((c, others))
    rng.shuffle(cand)
    pick = cand[:max(1, int(len(cand) * fraction))]
    return [(d.names['cells'][c], rng.choice(others)) for c, others in pick]


def truth(design, plat, swaps, libs, bundle_libs, bundle_lefs, sdc, rc, out_dir):
    tsv = os.path.join(out_dir, 'truth.tsv')
    swap_tcl = os.path.join(out_dir, 'swaps.tcl')
    with open(swap_tcl, 'w') as f:
        for c, m in swaps:
            f.write('replace_cell {%s} %s\n' % (c, m))
    odb = os.path.join(flow_dir(base_dir(run_key(plat, design))), '3_3_place_gp.odb')
    tcl = [DUMP_TCL, 'read_db %s' % w(odb)]
    tcl += ['read_lef -library %s' % w(p) for p in bundle_lefs]
    tcl += ['read_liberty %s' % w(p) for p in libs + [b for b in bundle_libs if b not in libs]]
    tcl += ['source %s' % w(swap_tcl), 'read_sdc %s' % w(sdc), 'source %s' % w(rc), 'estimate_parasitics -placement',
            'dump_ep %s' % w(tsv)]
    path = os.path.join(RUNS, 'tmp_sizing_%s_%s.tcl' % (plat, design))
    open(path, 'w').write('\n'.join(tcl) + '\nexit\n')
    try:
        log = orfs('/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad -no_init -no_splash %s 2>&1' % w(path))
    finally:
        os.remove(path)
    if not os.path.exists(tsv):
        raise RuntimeError(log[-3000:])
    tu = liberty.read(libs[0])['time_unit_ps']
    ep = {}
    for line in open(tsv):
        t = line.split()
        if len(t) == 3:
            try:
                ep[t[0]] = (float(t[1]) * tu, float(t[2]) * tu)
            except ValueError:
                pass
    return ep


def bundle_sources(name, plat):
    """The bundle's Liberty and LEF files (recorded by name in its library.json), found in orfs_runs/platform_<plat>."""
    meta = json.load(open(os.path.join(attar.paths.zoo_dir(), 'libraries', name, 'library.json')))
    root = os.path.join(RUNS, 'platform_' + plat)
    where = {}
    for dp, _, files in os.walk(root):
        for f in files:
            where.setdefault(f, os.path.join(dp, f))
    find = lambda n: where[n] if n in where else VS.platform_file(plat, 'lib', n)  # noqa: E731
    return ([find(r['name']) for r in meta['sources']['lib']],
            [where[r['name']] for r in meta['sources'].get('lef', []) if r['name'] in where])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--platform', default='asap7')
    ap.add_argument('--fraction', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    bundle = {'asap7': 'asap7_FF', 'nangate45': 'nangate45_typical'}[a.platform]
    blibs, blefs = bundle_sources(bundle, a.platform)
    blefs = [p for p in blefs if 'tech' not in os.path.basename(p)]      # the ODB has its technology
    rows = {}
    hdr = '%-20s %6s %6s %6s | %8s %7s %7s | %9s %9s %6s | %10s %10s %6s | %7s %6s'
    print(hdr % ('design', 'swaps', 'eps', 'match', 'pearson', 'mae', 'bias', 'worst', 'worst ref', 'WNS%', 'TNS',
                 'TNS ref', 'TNS%', 'holdP', 'holdE'))
    for design in a.designs:
        src = os.path.join(attar.paths.zoo_dir(), 'after_fp', 'orfs_' + a.platform, design)
        d = attar.load(src, dev)
        d.restore('reference')
        libs = VS.lib_paths(d, a.platform, design)
        sdc, rc = VS.sdc_path(design, a.platform), VS.platform_file(a.platform, 'setRC.tcl')
        swaps = swaps_for(d, a.fraction, a.seed)
        out = os.path.join(RUNS, 'sizing', a.platform, design); os.makedirs(out, exist_ok=True)
        t = time.time()
        d.resize([c for c, _ in swaps], [m for _, m in swaps])
        ours = d.timing().endpoint_slacks()
        t_attar = time.time() - t
        ref = truth(design, a.platform, swaps, libs, blibs, blefs, sdc, rc, out)
        r, *_ = VS.compare(ours, ref)
        r.update({'swaps': len(swaps), 'attar_s': t_attar})
        rows[design] = r
        f = lambda v, p='%.3f': '-' if v is None else p % v  # noqa: E731
        print(hdr % (design, len(swaps), r['endpoints_ref'], r['matched'], f(r['setup_pearson'], '%.4f'),
                     f(r['setup_mae'], '%.2f'), f(r['setup_bias'], '%+.2f'), f(r['worst_ours'], '%.2f'),
                     f(r['worst_ref'], '%.2f'), f(r['wns_err_pct'], '%.2f'), f(r['tns_ours'], '%.1f'),
                     f(r['tns_ref'], '%.1f'), f(r['tns_err_pct'], '%.2f'), f(r['hold_pearson'], '%.3f'),
                     f(r['hold_mae'], '%.2f')), flush=True)
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
