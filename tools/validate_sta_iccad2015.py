#!/usr/bin/env python3
"""Attar's timer against OpenSTA (OpenROAD in the ORFS image) on the ICCAD 2015 superblue designs, at the contest's
given placement.

  truth: OpenROAD reads the design's LEF, its Late Liberty, the DEF Attar exports (source frame + IR: the given
         placement and the Verilog's nets), the SDC and the contest wire RC (tools/distill_timing.iccad2015_inputs),
         runs estimate_parasitics -placement and dumps every endpoint's setup and hold slack. Cached in
         orfs_runs/sta_iccad2015/<design>/.
  attar: the zoo design (timing attached by tools/distill_timing.py iccad2015), timed at the same placement.
Both read the Late library only (Attar models one library per arc), so hold is compared on Late delays too.
usage: validate_sta_iccad2015.py <design> ... [--json out] [--device cuda:1]"""
import argparse
import json
import os
import shutil
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import attar  # noqa: E402
from distill_timing import ICCAD15, iccad2015_inputs  # noqa: E402
from orfs_roundtrip import RUNS  # noqa: E402
from validate_sta import compare  # noqa: E402
import validate_sta_ns3k as VN  # noqa: E402

INF = float('inf')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--work', default=os.path.join(RUNS, 'sta_iccad2015'))
    ap.add_argument('--json')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    dev = a.device
    rows = {}
    hdr = '%-12s %7s %7s | %8s %7s %7s | %9s %9s %6s | %11s %11s %6s | %5s | %7s %6s | %7s %6s'
    print(hdr % ('design', 'eps', 'match', 'pearson', 'mae', 'bias', 'worst', 'worst ref', 'WNS%', 'TNS', 'TNS ref',
                 'TNS%', 'top1', 'holdP', 'holdE', 'sta ms', 'ref s'))
    for design in a.designs:
        d = attar.load(os.path.join(attar.paths.zoo_dir(), 'after_fp', 'iccad2015', design), dev)
        d.require('sta')
        out = os.path.join(a.work, design); os.makedirs(out, exist_ok=True)
        libs, sdc, rc = iccad2015_inputs(design)
        lefs = [os.path.join(ICCAD15, design, design + '.lef')]
        # the container sees orfs_runs only: the inputs are linked into the work folder
        inp = os.path.join(out, 'inputs'); os.makedirs(inp, exist_ok=True)

        def here(p):
            q = os.path.join(inp, os.path.basename(p))
            if not os.path.exists(q):
                try:
                    os.link(p, q)
                except OSError:
                    shutil.copy(p, q)
            return q
        libs, sdc, rc, lefs = [here(p) for p in libs], here(sdc), here(rc), [here(p) for p in lefs]
        def_path = os.path.join(out, 'placed.def')
        if not os.path.exists(def_path):
            d.export(def_path)
            # superblue DEFs have TRACKS on metal7-9, which their LEF does not define: OpenROAD's DEF parser stops
            # on them. Dropped in this copy only (the zoo's frame keeps the source verbatim).
            layers = {l.split()[1] for l in open(lefs[0]) if l.startswith('LAYER ')}
            txt = [l for l in open(def_path) if not (l.startswith('TRACKS ') and l.split()[-2] not in layers)]
            open(def_path, 'w').writelines(txt)
        ref, meta = VN.truth(design, out, lefs, libs, def_path, sdc, rc)
        sync = torch.cuda.synchronize if dev.startswith('cuda') else (lambda: None)
        from attar.timing.sta import Timer
        T = Timer(d); res = T.run(); sync()
        sec = INF
        for _ in range(3):
            t = time.time(); res = T.run(); sync(); sec = min(sec, time.time() - t)
        r, *_ = compare(res.endpoint_slacks(), ref)
        r.update({'attar_s': sec, 'ref_s': meta['seconds'], 'ref_worst_reported': meta['worst_slack'],
                  'unsupported_sdc': json.load(open(os.path.join(d.path, 'timing.json')))['unsupported']})
        rows[design] = r
        f = lambda v, p='%.3f': '-' if v is None else p % v  # noqa: E731
        print(hdr % (design, r['endpoints_ref'], r['matched'], f(r['setup_pearson'], '%.4f'), f(r['setup_mae'], '%.2f'),
                     f(r['setup_bias'], '%+.2f'), f(r['worst_ours'], '%.2f'), f(r['worst_ref'], '%.2f'),
                     f(r['wns_err_pct'], '%.2f'), f(r['tns_ours'], '%.1f'), f(r['tns_ref'], '%.1f'),
                     f(r['tns_err_pct'], '%.2f'), '%.0f%%' % (100 * r['top1_overlap']), f(r['hold_pearson'], '%.3f'),
                     f(r['hold_mae'], '%.2f'), f(1000 * sec, '%.1f'), f(meta['seconds'], '%.2f')), flush=True)
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
