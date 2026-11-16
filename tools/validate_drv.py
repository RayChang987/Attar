#!/usr/bin/env python3
"""Attar's design rule checks (attar.timing.drv: max slew, capacitance, fanout) against OpenSTA in OpenROAD, on the same
netlist, placement and parasitics estimate.

Per zoo design of after_fp/orfs_<platform> (its reference placement is ORFS's 3_3_place_gp):
  truth: OpenROAD (the ORFS image) reads the run's 3_3_place_gp.odb, the Liberty files, the SDC and setRC.tcl, runs
         estimate_parasitics -placement and report_check_types -max_slew -max_capacitance -max_fanout for every
         checked pin (-max_count, -digits 6): pin, limit, value, slack. Cached in orfs_runs/drv/<platform>/<design>/.
  attar: d.restore('reference'); d.drv().
Reports per rule: pins checked by each, pins matched, mean / max |slack error|, the violations of each and how many
agree.
usage: validate_drv.py <design> ... [--platform asap7] [--json out]"""
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

RULES = ('slew', 'capacitance', 'fanout')
# (OpenSTA prints a fanout slack of -1 as an empty field: then slack = limit - value)
ROW = re.compile(r'^(\S+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)?\s*\((MET|VIOLATED)\)')


def parse_checks(text):
    """report_check_types text -> {rule: {pin: (limit, value, slack)}} (pin names without backslashes)."""
    out = {r: {} for r in RULES}
    cur = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('max slew'):
            cur = 'slew'
        elif s.startswith('max capacitance'):
            cur = 'capacitance'
        elif s.startswith('max fanout'):
            cur = 'fanout'
        elif cur:
            m = ROW.match(s)
            if m:
                lim, val = float(m.group(2)), float(m.group(3))
                out[cur][m.group(1).replace('\\', '')] = (lim, val, float(m.group(4)) if m.group(4) else lim - val)
    return out


def truth(design, plat, libs, sdc, rc, def_path=None, lefs=None, routing=None, tag=''):
    """OpenSTA's checks of every checked pin, in the first Liberty file's units converted to ps / fF."""
    out_dir = os.path.join(RUNS, 'drv', plat, design + tag)
    os.makedirs(out_dir, exist_ok=True)
    txt = os.path.join(out_dir, 'checks.txt')
    if def_path or not os.path.exists(txt):
        if def_path:
            load = ['read_lef %s' % w(l) for l in lefs] + ['read_liberty %s' % w(l) for l in libs] + [
                'read_def %s' % w(def_path)] + (['set_routing_layers -signal %s-%s' % tuple(routing)] if routing else [])
        else:
            odb = os.path.join(flow_dir(base_dir(run_key(plat, design))), '3_3_place_gp.odb')
            load = ['read_liberty %s' % w(l) for l in libs] + ['read_db %s' % w(odb)]
        tcl = load + ['read_sdc %s' % w(sdc), 'source %s' % w(rc), 'estimate_parasitics -placement',
                      'set t0 [clock milliseconds]',
                      'report_check_types -max_slew -max_capacitance -max_fanout -max_count 100000000 -digits 6 '
                      '> %s' % w(txt),
                      'puts "DRV_DONE [expr {([clock milliseconds] - $t0) / 1000.0}]"']
        log = openroad('\n'.join(tcl), 'drv_' + design)
        if 'DRV_DONE' not in log:
            raise RuntimeError(log[-3000:])
        open(os.path.join(out_dir, 'seconds'), 'w').write(log.split('DRV_DONE')[-1].split()[0])
    from attar.io import liberty
    L = liberty.read(libs[0])
    tu, cu = L['time_unit_ps'], L['cap_unit_ff']
    ref = parse_checks(open(txt).read())
    scale = {'slew': tu, 'capacitance': cu, 'fanout': 1.0}
    ref = {r: {p: tuple(v * scale[r] for v in x) for p, x in ref[r].items()} for r in RULES}
    sec = float(open(os.path.join(out_dir, 'seconds')).read()) if os.path.exists(os.path.join(out_dir, 'seconds')) else None
    return ref, sec


def ours(x, res):
    """{rule: {pin: (limit, value, slack)}} of the pins Attar checks."""
    out = {}
    for r, (v, lim, s) in {'slew': (x.slew, x.slew_limit, x.slew_slack),
                           'capacitance': (x.cap, x.cap_limit, x.cap_slack),
                           'fanout': (x.fanout, x.fanout_limit, x.fanout_slack)}.items():
        idx = torch.nonzero(torch.isfinite(s)).flatten().tolist()
        v, lim, s = v.cpu().numpy(), lim.cpu().numpy(), s.cpu().numpy()
        out[r] = {res._pin_name(p): (float(lim[p]), float(v[p]), float(s[p])) for p in idx}
    return out


def compare(o, ref):
    row = {}
    for r in RULES:
        a, b = o[r], ref[r]
        common = [k for k in b if k in a]
        e = np.array([abs(a[k][2] - b[k][2]) for k in common]) if common else np.zeros(0)
        le = np.array([abs(a[k][0] - b[k][0]) for k in common]) if common else np.zeros(0)
        va = {k for k, x in a.items() if x[2] < 0}; vb = {k for k, x in b.items() if x[2] < 0}
        rel = np.array([abs(a[k][1] - b[k][1]) / max(abs(b[k][1]), 1e-9) for k in common]) if common else np.zeros(0)
        row[r] = {'ref': len(b), 'ours': len(a), 'matched': len(common),
                  'only_ref': len(set(b) - set(a)), 'only_ours': len(set(a) - set(b)),
                  'slack_mae': float(e.mean()) if len(e) else None, 'slack_maxe': float(e.max()) if len(e) else None,
                  'value_rel_max': float(rel.max()) if len(rel) else None,
                  'limit_maxe': float(le.max()) if len(le) else None,
                  'viol_ref': len(vb), 'viol_ours': len(va), 'viol_both': len(va & vb),
                  'worst_ref': min((x[2] for x in b.values()), default=None),
                  'worst_ours': min((x[2] for x in a.values()), default=None),
                  'examples_only_ref': sorted(set(b) - set(a))[:3], 'examples_only_ours': sorted(set(a) - set(b))[:3],
                  'worst_err': sorted(((abs(a[k][2] - b[k][2]), k, a[k], b[k]) for k in common), reverse=True)[:2]}
    return row


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
        ref, sec_ref = truth(design, a.platform, libs, sdc, rc)
        d.restore('reference')
        res = d.timing()
        sync = torch.cuda.synchronize if dev == 'cuda' else (lambda: None)
        x = res.drv(); sync()
        t = time.time(); x = res.drv(); sync(); sec = time.time() - t
        r = compare(ours(x, res), ref)
        r['attar_s'], r['ref_s'] = sec, sec_ref
        rows[design] = r
        print('%-18s %s' % (design, ' | '.join(
            '%s %d/%d matched, slack err mean %s max %s, viol %d/%d (both %d)' % (
                k, r[k]['matched'], r[k]['ref'], '-' if r[k]['slack_mae'] is None else '%.4f' % r[k]['slack_mae'],
                '-' if r[k]['slack_maxe'] is None else '%.3f' % r[k]['slack_maxe'], r[k]['viol_ours'],
                r[k]['viol_ref'], r[k]['viol_both']) for k in RULES)), flush=True)
        if a.verbose:
            for k in RULES:
                print('   %s only_ref %d %s only_ours %d %s worst %s' % (k, r[k]['only_ref'], r[k]['examples_only_ref'],
                      r[k]['only_ours'], r[k]['examples_only_ours'], r[k]['worst_err']))
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
