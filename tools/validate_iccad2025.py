#!/usr/bin/env python3
"""Attar against the ICCAD 2025 contest's own evaluation (problem C, openRoad_eval_script/eval_def.tcl), at each
design's given placement: OpenROAD reads every ASAP7/LIB/*nldm*.lib (in the script's glob order), the tech LEF, the
LEFs, the design's DEF and SDC, the contest's setRC.tcl, runs estimate_parasitics -placement and reports TNS, WNS and
power; here it also dumps every endpoint's setup / hold slack. Attar times the zoo design (after_fp/iccad2025) at the
same placement. Cached in orfs_runs/iccad25_eval/<design>/.
usage: validate_iccad2025.py [design ...] [--device cuda:1] [--json out]"""
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
from orfs_roundtrip import RUNS, orfs, w  # noqa: E402
from validate_sta import DUMP_TCL, compare  # noqa: E402

B = os.path.join(ROOT, 'benchmarks', 'placement', 'iccad2025_c')
WORK = os.path.join(RUNS, 'iccad25_eval')
DESIGNS = ['ac97_top', 'aes', 'aes_cipher_top', 'ariane', 'des', 'des3', 'fpu', 'mc_top', 'mempool_tile_wrap',
           'netcard_fast', 'NV_NVDLA_partition_c', 'pci_bridge32', 'tv80s']


def inputs(design):
    """The contest's files, linked into the work folder (the container sees orfs_runs only)."""
    asap7 = os.path.join(WORK, 'ASAP7')
    if not os.path.isdir(asap7):
        shutil.copytree(os.path.join(B, 'ASAP7'), asap7)
    src = next(os.path.join(B, s, design) for s in ('ICCAD25_testcases', 'hidden')
               if os.path.isdir(os.path.join(B, s, design)))
    out = os.path.join(WORK, design)
    os.makedirs(out, exist_ok=True)
    for ext in ('def', 'sdc', 'lef'):
        f = os.path.join(src, '%s.%s' % (design, ext))
        if os.path.exists(f) and not os.path.exists(os.path.join(out, os.path.basename(f))):
            shutil.copy(f, out)
    return out


def truth(design):
    out = inputs(design)
    meta = os.path.join(out, 'truth.json')
    if not os.path.exists(meta):
        d = w(out)
        tcl = [DUMP_TCL,
               'foreach libFile [glob "%s/ASAP7/LIB/*nldm*.lib"] { read_liberty $libFile }' % w(WORK),
               'read_lef %s/ASAP7/techlef/asap7_tech_1x_201209.lef' % w(WORK),
               'foreach lef [glob "%s/ASAP7/LEF/*.lef"] { read_lef $lef }' % w(WORK),
               'if {[file exists %s/%s.lef]} { read_lef %s/%s.lef }' % (d, design, d, design),
               'read_def %s/%s.def' % (d, design), 'read_sdc %s/%s.sdc' % (d, design),
               'source %s/ASAP7/setRC.tcl' % w(WORK),
               'set t0 [clock milliseconds]', 'estimate_parasitics -placement',
               'set wns [sta::worst_slack -max]', 'set tns [sta::total_negative_slack -max]',
               'puts "EVAL_DONE [expr {([clock milliseconds] - $t0) / 1000.0}] $wns $tns"',
               'report_power -digits 6', 'dump_ep %s/endpoints.tsv' % d]
        path = os.path.join(RUNS, 'tmp_iccad25_%s.tcl' % design)
        open(path, 'w').write('\n'.join(tcl) + '\nexit\n')
        try:
            log = orfs('/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad -no_init -no_splash %s' % w(path))
        finally:
            os.remove(path)
        done = [ln for ln in log.splitlines() if ln.startswith('EVAL_DONE')]
        tot = [ln.split() for ln in log.splitlines() if ln.startswith('Total ')]
        if not done or not tot:
            raise RuntimeError(log[-3000:])
        _, sec, wns, tns = done[-1].split()
        json.dump({'seconds': float(sec), 'wns': float(wns), 'tns': float(tns),
                   'power': [float(x) for x in tot[-1][1:5]]}, open(meta, 'w'))   # internal, switching, leakage, total
    m = json.load(open(meta))
    ep = {}
    for ln in open(os.path.join(out, 'endpoints.tsv')):
        t = ln.split()
        if len(t) == 3:
            try:
                ep[t[0]] = (float(t[1]), float(t[2]))                      # the Liberty time unit is ps
            except ValueError:
                pass
    return ep, m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='*')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--json')
    a = ap.parse_args()
    rows = {}
    hdr = '%-22s %6s %6s | %8s %7s | %9s %9s %6s | %11s %11s %6s | %7s %6s | %7s %7s %7s'
    print(hdr % ('design', 'eps', 'match', 'pearson', 'mae', 'WNS', 'WNS ref', 'WNS%', 'TNS', 'TNS ref', 'TNS%',
                 'holdP', 'holdE', 'Pint%', 'Psw%', 'Pleak%'))
    for design in a.designs or DESIGNS:
        ref, meta = truth(design)
        d = attar.load('after_fp/iccad2025/' + design, a.device)
        t = time.time()
        res = d.timing()
        sec = time.time() - t
        r, *_ = compare(res.endpoint_slacks(), ref)
        s = res.summary()
        pw = [float(x) for x in d.power().total]
        perr = [100.0 * (x - y) / y if y else 0.0 for x, y in zip(pw, meta['power'][:3])]
        r.update({'wns': s['wns'], 'tns': s['tns'], 'wns_ref': meta['wns'], 'tns_ref': meta['tns'], 'power': pw,
                  'power_ref': meta['power'], 'power_err_pct': perr, 'attar_s': sec, 'ref_s': meta['seconds']})
        rows[design] = r
        del d, res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pct = lambda x, y: 100.0 * abs(x - y) / abs(y) if y else 0.0  # noqa: E731
        f = lambda v, p='%.3f': '-' if v is None else p % v  # noqa: E731
        print(hdr % (design, r['endpoints_ref'], r['matched'], f(r['setup_pearson'], '%.4f'), f(r['setup_mae'], '%.2f'),
                     '%.2f' % s['wns'], '%.2f' % meta['wns'], '%.2f' % pct(s['wns'], meta['wns']),
                     '%.1f' % s['tns'], '%.1f' % meta['tns'], '%.2f' % pct(s['tns'], meta['tns']),
                     f(r['hold_pearson'], '%.3f'), f(r['hold_mae'], '%.2f'), *['%+.2f' % e for e in perr]), flush=True)
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
