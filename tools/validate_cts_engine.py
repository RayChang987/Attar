#!/usr/bin/env python3
"""Attar's clock tree synthesis (attar.naive.cts) on ORFS designs: agreement with OpenSTA on its trees, and quality
against TritonCTS on the same placement.

Per design (an ORFS run carried through CTS by tools/orfs_run_cts.sh, orfs_runs/cts/<platform>_<design>_base):
  start:     3_5_place_dp.odb (the placement CTS starts from) distilled into a scratch design with 3_place.sdc
             (tools/validate_cts.distill).
  attar:     attar.naive.cts, then Attar's own timing (propagated clocks), clock arrivals and power; the result is
             exported (DEF) with an SDC that adds set_propagated_clock to 3_place.sdc.
  agreement: OpenROAD reads that DEF, the SDC and setRC.tcl, estimate_parasitics -placement: WNS / TNS / WHS (CRPR
             off, Attar's model, and on, OpenSTA's default), every endpoint's slack (CRPR off), every register clock
             pin's arrival, report_clock_skew, report_power.
  baseline:  OpenROAD reads 3_5_place_dp.odb and runs ORFS's CTS steps (repair_clock_inverters, clock_tree_synthesis
             -sink_clustering_enable -repair_clock_nets, set_propagated_clock, detailed_placement; no repair_timing):
             the same metrics, its clock buffers and run time.
usage: validate_cts_engine.py <design> ... [--platform asap7] [--json out]"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import attar  # noqa: E402
import attar.naive as N  # noqa: E402
import validate_cts as VC  # noqa: E402
import validate_sta as VS  # noqa: E402
from orfs_roundtrip import design_config, flow_dir, inputs, orfs, openroad, run_key, w  # noqa: E402

REPORT_TCL = r'''
proc dump_all {dir} {
  set f [open $dir/ep.tsv w]
  foreach p [concat [all_registers -data_pins] [all_outputs]] {
    puts $f "[get_full_name $p] [get_property $p slack_max] [get_property $p slack_min]"
  }
  close $f
  set f [open $dir/ck.tsv w]
  foreach p [all_registers -clock_pins] {
    puts $f "[get_full_name $p] [get_property $p arrival_max_rise] [get_property $p arrival_min_rise]"
  }
  close $f
}
proc metrics {tag} {
  puts "M $tag [sta::worst_slack -max] [sta::total_negative_slack -max] [sta::worst_slack -min]"
}
'''


def _metrics(log, tu):
    out = {}
    for line in log.splitlines():
        if line.startswith('M '):
            t = line.split()
            out[t[1]] = {'wns': min(float(t[2]), 0) * tu, 'tns': float(t[3]) * tu, 'whs': min(float(t[4]), 0) * tu}
    sk = re.findall(r'^\s*(-?[\d.]+)\s+setup skew', log, re.M)
    if sk:
        out['skew_setup'] = float(sk[-1]) * tu
    pw = re.findall(r'^(Clock|Total)\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)', log, re.M)
    for g, *v in pw:
        out['power_' + g.lower()] = [float(x) for x in v]
    return out


def _read(dirn, tu):
    ep, ck = {}, {}
    for line in open(os.path.join(dirn, 'ep.tsv')):
        t = line.split()
        if len(t) == 3:
            try:
                ep[t[0].replace('\\', '')] = (float(t[1]) * tu, float(t[2]) * tu)
            except ValueError:
                pass
    for line in open(os.path.join(dirn, 'ck.tsv')):
        t = line.split()
        if len(t) == 3:
            try:
                ck[t[0].replace('\\', '')] = (float(t[1]) * tu, float(t[2]) * tu)
            except ValueError:
                pass
    return ep, ck


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--platform', default='asap7')
    ap.add_argument('--json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    plat = a.platform
    rows = {}
    rc = VS.platform_file(plat, 'setRC.tcl')
    for design in a.designs:
        base = VC.run_dir(plat, design)
        res_dir = flow_dir(base)
        lefs, libs = inputs(VS.base_dir(run_key(plat, design)))
        from attar.io import liberty
        tu = liberty.read(libs[0])['time_unit_ps']
        ir = VC.distill(plat, design, '3_5_place_dp', '3_place.sdc')
        out = os.path.join(VC.CTS, '_eng', '%s_%s' % (plat, design)); os.makedirs(out, exist_ok=True)
        # -- Attar ------------------------------------------------------------------------------------------------
        d = attar.load(ir, dev)
        d.restore('reference')
        t = time.time()
        st = N.cts(d, verbose=False)
        if dev == 'cuda':
            torch.cuda.synchronize()
        sec = time.time() - t
        r = d.timing()
        ours_ep = r.endpoint_slacks()
        ours_ck = {k: (v[0], v[2]) for k, v in r.clock_arrivals().items()}
        pw = d.power()
        def_path = os.path.join(out, 'attar_cts.def')
        d.export(def_path)
        sdc = os.path.join(out, 'attar_cts.sdc')
        clocks = [k for k, c in json.load(open(os.path.join(ir, 'timing.json')))['clocks'].items() if c.get('sources')]
        open(sdc, 'w').write(open(os.path.join(res_dir, '3_place.sdc')).read()
                             + '\nset_propagated_clock [get_clocks {%s}]\n' % ' '.join(clocks))
        routing = (attar.load(ir, 'cpu').manifest.get('routing') or {}).get('layers')
        tcl = [REPORT_TCL] + ['read_lef %s' % w(l) for l in lefs] + ['read_liberty %s' % w(l) for l in libs] + [
            'read_def %s' % w(def_path)] + (['set_routing_layers -signal %s-%s' % tuple(routing)] if routing else []) + [
            'read_sdc %s' % w(sdc), 'source %s' % w(rc), 'estimate_parasitics -placement', 'metrics crpr_on',
            'report_clock_skew -setup -digits 4', 'report_power -digits 6',
            'sta::set_crpr_enabled 0', 'metrics crpr_off', 'dump_all %s' % w(out)]
        log = openroad('\n'.join(tcl), 'cts_eng_%s' % design)
        ref = _metrics(log, tu)
        ref_ep, ref_ck = _read(out, tu)
        cr, *_ = VS.compare(ours_ep, ref_ep)
        common = [k for k in ref_ck if k in ours_ck]
        cke = np.abs(np.array([ours_ck[k] for k in common]) - np.array([ref_ck[k] for k in common]))
        # -- TritonCTS baseline -----------------------------------------------------------------------------------
        bdir = os.path.join(VC.CTS, '_eng', '%s_%s_triton' % (plat, design)); os.makedirs(bdir, exist_ok=True)
        bmeta = os.path.join(bdir, 'meta.json')
        if not os.path.exists(bmeta):
            dont = orfs('make --no-print-directory DESIGN_CONFIG=%s print-DONT_USE_CELLS' % design_config(
                run_key(plat, design))).split(':', 1)[-1].strip()
            tcl = [REPORT_TCL] + ['read_liberty %s' % w(l) for l in libs] + [
                'read_db %s' % w(res_dir + '/3_5_place_dp.odb'), 'read_sdc %s' % w(res_dir + '/3_place.sdc'),
                'source %s' % w(rc), 'set_dont_use {%s}' % dont, 'set t0 [clock milliseconds]',
                'repair_clock_inverters', 'clock_tree_synthesis -sink_clustering_enable -repair_clock_nets',
                'set_propagated_clock [all_clocks]', 'estimate_parasitics -placement', 'detailed_placement',
                'puts "CTS_S [expr {([clock milliseconds] - $t0) / 1000.0}]"', 'estimate_parasitics -placement',
                'metrics crpr_on', 'report_clock_skew -setup -digits 4', 'report_power -digits 6',
                'puts "NBUF [llength [get_cells clkbuf*]] [llength [get_cells clkload*]]"',
                'sta::set_crpr_enabled 0', 'metrics crpr_off', 'dump_all %s' % w(bdir)]
            blog = openroad('\n'.join(tcl), 'cts_triton_%s' % design)
            bm = _metrics(blog, tu)
            bm['seconds'] = float(re.findall(r'CTS_S (\S+)', blog)[-1])
            bm['buffers'] = [int(x) for x in re.findall(r'NBUF (\d+) (\d+)', blog)[-1]]
            json.dump(bm, open(bmeta, 'w'))
        bm = json.load(open(bmeta))
        _, bck = _read(bdir, tu)
        bl = np.array([v[0] for v in bck.values()])
        row = {'attar': {'buffers': st['buffers'], 'latency': st['latency'], 'spread': st['skew'],
                         'wns': r.wns, 'tns': r.tns, 'whs': r.whs, 'seconds': sec,
                         'power_clock': pw.groups['Clock'], 'power_total': pw.total},
               'agreement': {'opensta': ref, 'endpoints': cr['matched'], 'setup_mae': cr['setup_mae'],
                             'hold_mae': cr['hold_mae'], 'ck_pins': len(common),
                             'ck_mae': float(cke.mean()) if len(cke) else None,
                             'ck_max': float(cke.max()) if len(cke) else None},
               'triton': dict(bm, latency=float(bl.max()), spread=float(bl.max() - bl.min()))}
        rows[design] = row
        o, g, b = row['attar'], ref, bm
        print('%-10s attar: %d bufs, latency %.1f spread %.1f, WNS %.2f TNS %.1f WHS %.2f, clock power %.3e W, %.1fs'
              % (design, o['buffers'], o['latency'], o['spread'], o['wns'], o['tns'], o['whs'], sum(o['power_clock']),
                 o['seconds']))
        print('           OpenSTA on it (CRPR off): WNS %.2f TNS %.1f WHS %.2f | endpoints %d setup MAE %.4f hold MAE %.4f '
              '| clock pins %d arrival MAE %.4f max %.4f | CRPR on: WNS %.2f TNS %.1f, skew %.2f, clock power %s'
              % (g['crpr_off']['wns'], g['crpr_off']['tns'], g['crpr_off']['whs'], cr['matched'], cr['setup_mae'],
                 cr['hold_mae'] or 0, len(common), row['agreement']['ck_mae'] or 0, row['agreement']['ck_max'] or 0,
                 g['crpr_on']['wns'], g['crpr_on']['tns'], g.get('skew_setup', float('nan')),
                 g.get('power_clock')))
        print('           TritonCTS: %s bufs, latency %.1f spread %.1f, CRPR on: WNS %.2f TNS %.1f WHS %.2f, skew %.2f, '
              'clock power %s, %.1fs' % (b['buffers'], row['triton']['latency'], row['triton']['spread'],
                                        b['crpr_on']['wns'], b['crpr_on']['tns'], b['crpr_on']['whs'],
                                        b.get('skew_setup', float('nan')), b.get('power_clock'), b['seconds']),
              flush=True)
        del d, r, pw
        if dev == 'cuda':
            torch.cuda.empty_cache()
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
