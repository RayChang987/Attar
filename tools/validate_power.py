#!/usr/bin/env python3
"""Attar's power analysis (attar.timing.power) against OpenSTA's report_power in OpenROAD, on the same netlist,
placement and parasitics estimate.

Per zoo design of after_fp/orfs_<platform> (its reference placement is ORFS's 3_3_place_gp):
  truth: OpenROAD (the ORFS image) reads the run's 3_3_place_gp.odb, Liberty, SDC and setRC.tcl, runs
         estimate_parasitics -placement, then report_power (the design's groups), report_power -instances (every
         instance, JSON) and every pin's activity (get_property activity: density per s, duty, origin). Cached in
         orfs_runs/power/<platform>/<design>/.
  attar: d.restore('reference'); d.power().
Reports: the group totals of both, per-instance total / internal / switching / leakage errors, per-pin activity
density and duty errors.
usage: validate_power.py <design> ... [--platform asap7] [--json out] [--verbose]"""
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
import validate_sta as VS  # noqa: E402
from orfs_roundtrip import RUNS, base_dir, flow_dir, openroad, run_key, w  # noqa: E402

DUMP_TCL = r'''
proc dump_act {path} {
  set f [open $path w]
  foreach p [concat [get_pins -hierarchical *] [all_inputs] [all_outputs]] {
    puts $f "[get_full_name $p] [get_property $p activity]"
  }
  close $f
}
'''


def truth(design, plat, libs, sdc, rc):
    out = os.path.join(RUNS, 'power', plat, design)
    os.makedirs(out, exist_ok=True)
    files = [os.path.join(out, f) for f in ('design.txt', 'insts.json', 'act.txt')]
    if not all(os.path.exists(f) for f in files):
        odb = os.path.join(flow_dir(base_dir(run_key(plat, design))), '3_3_place_gp.odb')
        tcl = [DUMP_TCL] + ['read_liberty %s' % w(l) for l in libs] + [
            'read_db %s' % w(odb), 'read_sdc %s' % w(sdc), 'source %s' % w(rc), 'estimate_parasitics -placement',
            'set t0 [clock milliseconds]',
            'report_power -digits 6 > %s' % w(files[0]),
            'puts "POWER_DONE [expr {([clock milliseconds] - $t0) / 1000.0}]"',
            'report_power -instances [get_cells *] -format json -digits 8 > %s' % w(files[1]),
            'dump_act %s' % w(files[2])]
        log = openroad('\n'.join(tcl), 'pwr_' + design)
        if 'POWER_DONE' not in log:
            raise RuntimeError(log[-3000:])
        open(os.path.join(out, 'seconds'), 'w').write(log.split('POWER_DONE')[-1].split()[0])
    groups = {}
    for line in open(files[0]):
        m = re.match(r'^(Sequential|Combinational|Clock|Macro|Pad|Total)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', line)
        if m:
            groups[m.group(1)] = tuple(float(m.group(i)) for i in range(2, 6))
    text = open(files[1]).read().replace('\\', '\\\\')        # OpenSTA leaves escaped names' backslashes raw
    insts = {r['name'].replace('\\', ''): (r['internal'], r['switching'], r['leakage'], r['total'])
             for r in json.loads(text)}
    act = {}
    for line in open(files[2]):
        t = line.split()
        if len(t) >= 4:
            try:
                act[t[0].replace('\\', '')] = (float(t[1]), float(t[2]), t[3])
            except ValueError:
                pass
    sec = float(open(os.path.join(out, 'seconds')).read())
    return groups, insts, act, sec


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--platform', default='asap7')
    ap.add_argument('--json')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = {}
    for design in a.designs:
        rc = VS.platform_file(a.platform, 'setRC.tcl')
        d = attar.load(os.path.join(attar.paths.zoo_dir(), 'after_fp', 'orfs_' + a.platform, design), dev)
        libs = VS.lib_paths(d, a.platform, design)
        sdc = VS.sdc_path(design, a.platform)
        groups, insts, act, sec_ref = truth(design, a.platform, libs, sdc, rc)
        d.restore('reference')
        sync = torch.cuda.synchronize if dev == 'cuda' else (lambda: None)
        t = time.time(); res = d.power(); sync(); first = time.time() - t
        t = time.time(); res = d.power(); sync(); sec = time.time() - t
        ours = res.instances()
        common = [k for k in insts if k in ours]
        A = np.array([ours[k] for k in common]); B = np.array([insts[k] for k in common])
        r = {'instances_ref': len(insts), 'instances_ours': len(ours), 'matched': len(common),
             'attar_s': sec, 'attar_first_s': first, 'ref_s': sec_ref, 'passes': d._timer._power.n_passes}
        tot_ref = groups.get('Total'); tot = res.total
        r['total_ours'] = list(tot) + [sum(tot)]; r['total_ref'] = list(tot_ref) if tot_ref else None
        for i, k in enumerate(('internal', 'switching', 'leakage', 'total')):
            e = np.abs(A[:, i] - B[:, i])
            r['%s_err_pct' % k] = 100 * rel(A[:, i].sum(), B[:, i].sum())
            r['%s_inst_mae_rel' % k] = float(e.sum() / max(np.abs(B[:, i]).sum(), 1e-30))
        r['groups_ours'] = {g: list(v) for g, v in res.groups.items()}
        r['groups_ref'] = groups
        # activities: per pin density (1/s) and duty
        T = d._timer
        tr = d.timing()
        pa = {}
        fd, fu = (x.cpu().numpy() for x in res.activity[:2])
        for p in range(T.P):
            pa[tr._pin_name(p)] = (fd[p] * 1e12, fu[p])
        ck = [k for k in act if k in pa]
        ad = np.array([pa[k][0] for k in ck]); bd = np.array([act[k][0] for k in ck])
        au = np.array([pa[k][1] for k in ck]); bu = np.array([act[k][1] for k in ck])
        rd = np.abs(ad - bd) / np.maximum(np.abs(bd), 1.0)
        r.update({'act_pins': len(ck), 'act_density_rel_mean': float(rd.mean()), 'act_density_rel_max': float(rd.max()),
                  'act_duty_mae': float(np.abs(au - bu).mean()), 'act_duty_max': float(np.abs(au - bu).max()),
                  'act_bad_1pct': int((rd > 0.01).sum())})
        rows[design] = r
        print('%-18s total %.4e vs %.4e (%.2f%%) | int %.2f%% sw %.2f%% leak %.2f%% | inst rel-MAE %.4f | act dens '
              'mean %.4f max %.3f (>1%%: %d of %d) duty mae %.4f | %.3fs (first %.1fs, %d passes) vs %.2fs' % (
                  design, sum(tot), tot_ref[3] if tot_ref else float('nan'), r['total_err_pct'], r['internal_err_pct'],
                  r['switching_err_pct'], r['leakage_err_pct'], r['total_inst_mae_rel'], r['act_density_rel_mean'],
                  r['act_density_rel_max'], r['act_bad_1pct'], r['act_pins'], r['act_duty_mae'], sec, first,
                  r['passes'], sec_ref), flush=True)
        if a.verbose:
            bad = np.argsort(-rd)[:8]
            for i in bad:
                print('   act %-50s ours %.4e %.3f ref %.4e %.3f %s' % (ck[i], ad[i], au[i], bd[i], bu[i], act[ck[i]][2]))
            for g in groups:
                print('   %-14s ours %s ref %s' % (g, ['%.4e' % x for x in (res.groups.get(g) or res.total)],
                                                  ['%.4e' % x for x in groups[g]]))
        del d, res, T, tr                         # the timer and power graphs hold GPU memory pools
        import gc
        gc.collect()
        if dev == 'cuda':
            torch.cuda.empty_cache()
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
