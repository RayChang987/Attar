#!/usr/bin/env python3
"""Attar's propagated-clock timing against OpenSTA, on ORFS's own clock trees (TritonCTS, the 4_cts stage).

Per design (an ORFS run carried through CTS by tools/orfs_run_cts.sh, orfs_runs/cts/<platform>_<design>_base):
  distill: 4_cts.odb -> DEF -> a scratch Attar design (orfs_runs/cts/_ir/<platform>_<design>: the clock buffers and
           the flow's CTS repairs are in its netlist, the CTS placement its 'reference'), timing attached with the
           run's Liberty files, 4_cts.sdc (set_propagated_clock) and the platform's setRC.tcl.
  truth:   OpenROAD reads 4_cts.odb, the Liberty files, 4_cts.sdc and setRC.tcl, runs estimate_parasitics -placement,
           and dumps every endpoint's setup / hold slack, every register clock pin's clock arrival (max / min, rise /
           fall) and slew, WNS / TNS with CRPR on (OpenSTA's default) and off, report_clock_skew. Cached in
           orfs_runs/cts/_truth/<platform>_<design>/.
  attar:   d.restore('reference'); d.timing(); clock arrivals at the register clock pins.
usage: validate_cts.py <design> ... [--platform asap7] [--json out] [--truth-only]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import attar  # noqa: E402
import validate_sta as VS  # noqa: E402
from orfs_roundtrip import RUNS, flow_dir, inputs, openroad, routing_resources, run_key, w  # noqa: E402

CTS = os.path.join(RUNS, 'cts')
BUNDLE = {'asap7': 'asap7_FF', 'nangate45': 'nangate45_typical'}

DUMP_TCL = r'''
proc dump_ep {path} {
  set f [open $path w]
  foreach p [all_registers -data_pins] {
    puts $f "[get_full_name $p] [get_property $p slack_max] [get_property $p slack_min]"
  }
  foreach p [all_outputs] {
    puts $f "[get_full_name $p] [get_property $p slack_max] [get_property $p slack_min]"
  }
  close $f
}
proc dump_ck {path} {
  set f [open $path w]
  foreach p [all_registers -clock_pins] {
    puts $f "[get_full_name $p] [get_property $p arrival_max_rise] [get_property $p arrival_max_fall] [get_property $p arrival_min_rise] [get_property $p arrival_min_fall] [get_property $p slew_max_rise] [get_property $p slew_max_fall] [get_property $p slew_min_rise] [get_property $p slew_min_fall]"
  }
  close $f
}
'''


def run_dir(plat, design):
    return os.path.join(CTS, '%s_%s_base' % (plat, design))


def distill(plat, design):
    """The scratch Attar design of the run's 4_cts stage (made once)."""
    base = run_dir(plat, design)
    res = flow_dir(base)
    ir = os.path.join(CTS, '_ir', '%s_%s' % (plat, design))
    if os.path.exists(os.path.join(ir, 'timing.json')):
        return ir
    rt = os.path.join(CTS, '_rt', '%s_%s' % (plat, design)); os.makedirs(rt, exist_ok=True)
    src = os.path.join(rt, '4_cts.def')
    if not os.path.exists(src):
        openroad('read_db %s\nwrite_def %s' % (w(res + '/4_cts.odb'), w(src)), 'cts_export_%s_%s' % (plat, design))
    lefs, libs = inputs(VS.base_dir(run_key(plat, design)))
    attar.distill(src, lefs, ir, lib_paths=libs, reference='orfs 4_cts (TritonCTS)',
                  extra={'suite': 'orfs/' + plat, 'orfs_stage': '4_cts',
                         'routing': routing_resources(run_key(plat, design))})
    from attar.timing import ir as timing
    bundle = os.path.join(attar.paths.zoo_dir(), 'libraries', BUNDLE[plat])
    timing.attach(ir, libs, os.path.join(res, '4_cts.sdc'), VS.platform_file(plat, 'setRC.tcl'), bundle=bundle)
    return ir


def truth(plat, design):
    base = run_dir(plat, design)
    res = flow_dir(base)
    out = os.path.join(CTS, '_truth', '%s_%s' % (plat, design)); os.makedirs(out, exist_ok=True)
    f_ep, f_ck, f_meta, f_skew, f_ep0 = (os.path.join(out, x) for x in ('ep.tsv', 'ck.tsv', 'meta.json', 'skew.txt',
                                                                         'ep_nocrpr.tsv'))
    _, libs = inputs(VS.base_dir(run_key(plat, design)))
    if not all(os.path.exists(f) for f in (f_ep, f_ck, f_meta, f_ep0)):
        tcl = [DUMP_TCL] + ['read_liberty %s' % w(l) for l in libs] + [
            'read_db %s' % w(res + '/4_cts.odb'), 'read_sdc %s' % w(res + '/4_cts.sdc'),
            'source %s' % w(VS.platform_file(plat, 'setRC.tcl')), 'set t0 [clock milliseconds]',
            'estimate_parasitics -placement',
            'set wns [sta::worst_slack -max]; set tns [sta::total_negative_slack -max]',
            'set whs [sta::worst_slack -min]',
            'puts "STA_DONE [expr {([clock milliseconds] - $t0) / 1000.0}] $wns $tns $whs"',
            'dump_ep %s' % w(f_ep), 'dump_ck %s' % w(f_ck),
            'report_clock_skew -setup -digits 4 > %s' % w(f_skew),
            'sta::set_crpr_enabled 0',
            'puts "NOCRPR [sta::worst_slack -max] [sta::total_negative_slack -max] [sta::worst_slack -min]"',
            'dump_ep %s' % w(os.path.join(out, 'ep_nocrpr.tsv'))]
        log = openroad('\n'.join(tcl), 'cts_truth_%s_%s' % (plat, design))
        line = [l for l in log.splitlines() if l.startswith('STA_DONE')]
        nocrpr = [l for l in log.splitlines() if l.startswith('NOCRPR')]
        if not line:
            raise RuntimeError(log[-3000:])
        _, sec, wns, tns, whs = line[-1].split()
        _, wns0, tns0, whs0 = nocrpr[-1].split()
        json.dump({'seconds': float(sec), 'wns': float(wns), 'tns': float(tns), 'whs': float(whs),
                   'wns_nocrpr': float(wns0), 'tns_nocrpr': float(tns0), 'whs_nocrpr': float(whs0)},
                  open(f_meta, 'w'))
    from attar.io import liberty
    tu = liberty.read(libs[0])['time_unit_ps']
    def read_ep(f):
        ep = {}
        for line in open(f):
            t = line.split()
            if len(t) == 3:
                try:
                    ep[t[0].replace('\\', '')] = (float(t[1]) * tu, float(t[2]) * tu)
                except ValueError:
                    pass
        return ep
    ep = read_ep(f_ep)
    meta_ep0 = read_ep(f_ep0)
    ck = {}
    for line in open(f_ck):
        t = line.split()
        if len(t) == 9:
            try:
                ck[t[0].replace('\\', '')] = tuple(float(x) * tu for x in t[1:])
            except ValueError:
                pass
    meta = json.load(open(f_meta))
    for k in ('wns', 'tns', 'whs', 'wns_nocrpr', 'tns_nocrpr', 'whs_nocrpr'):
        meta[k] *= tu
    meta['skew_report'] = open(f_skew).read() if os.path.exists(f_skew) else ''
    meta['ep_nocrpr'] = meta_ep0
    return ep, ck, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--platform', default='asap7')
    ap.add_argument('--json')
    ap.add_argument('--truth-only', action='store_true')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = {}
    for design in a.designs:
        ep, ck, meta = truth(a.platform, design)
        print('%-12s OpenSTA: WNS %.2f TNS %.1f WHS %.2f | CRPR off: WNS %.2f TNS %.1f WHS %.2f | %d endpoints, %d clock pins'
              % (design, meta['wns'], meta['tns'], meta['whs'], meta['wns_nocrpr'], meta['tns_nocrpr'],
                 meta['whs_nocrpr'], len(ep), len(ck)), flush=True)
        if a.truth_only:
            continue
        ir = distill(a.platform, design)
        d = attar.load(ir, dev)
        d.restore('reference')
        sync = torch.cuda.synchronize if dev == 'cuda' else (lambda: None)
        res = d.timing(); sync()
        t = time.time(); res = d.timing(); sync(); sec = time.time() - t
        ours = res.endpoint_slacks()
        r, *_ = VS.compare(ours, ep)
        r0, *_ = VS.compare(ours, meta.pop('ep_nocrpr'))
        r['nocrpr'] = {k: r0[k] for k in ('setup_mae', 'hold_mae', 'worst_ours', 'worst_ref', 'tns_ours', 'tns_ref',
                                          'wns_err_pct', 'tns_err_pct', 'matched')}
        r.update({'attar_s': sec, 'ref': meta})
        ca = res.clock_arrivals() if hasattr(res, 'clock_arrivals') else {}
        common = [k for k in ck if k in ca]
        if common:
            A = np.array([ca[k] for k in common]); B = np.array([ck[k] for k in common])
            fin = np.isfinite(A) & np.isfinite(B) & (np.abs(B) < 1e20)
            e = np.abs(A - B)[fin]
            r.update({'ck_matched': len(common), 'ck_arrival_mae': float(e.mean()) if len(e) else None,
                      'ck_arrival_maxe': float(e.max()) if len(e) else None})
        rows[design] = r
        n0 = r['nocrpr']
        print('   CRPR off: WNS %.2f vs %.2f TNS %.1f vs %.1f | setup MAE %.4f ps, hold MAE %.4f ps (%d endpoints)' % (
            min(n0['worst_ours'], 0), min(n0['worst_ref'], 0), n0['tns_ours'], n0['tns_ref'], n0['setup_mae'],
            n0['hold_mae'] or 0, n0['matched']))
        print('   attar: WNS %.2f vs %.2f (%.2f%%) TNS %.1f vs %.1f | setup MAE %.3f ps, hold MAE %s | clock pins %s '
              'arrival MAE %s max %s | %.1f ms' % (
                  min(r['worst_ours'], 0), min(r['worst_ref'], 0), r['wns_err_pct'] or 0, r['tns_ours'], r['tns_ref'],
                  r['setup_mae'], r['hold_mae'], r.get('ck_matched'), r.get('ck_arrival_mae'),
                  r.get('ck_arrival_maxe'), 1000 * sec), flush=True)
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
