#!/usr/bin/env python3
"""Attar's timer (attar.sta) against OpenSTA in OpenROAD, on the same netlist and placement.

Per zoo design of after_fp/orfs_<platform> (its reference placement is ORFS's 3_3_place_gp):
  truth: OpenROAD (the ORFS image) reads the run's 3_3_place_gp.odb, the Liberty files the design was distilled with,
         its SDC (2_floorplan.sdc, what ORFS's write_sdc made) and the platform's setRC.tcl, runs
         estimate_parasitics -placement and dumps every endpoint's setup and hold slack (register data pins, output
         ports), WNS and TNS. Cached in orfs_runs/sta/<design>/.
  attar: the design with timing attached (attar.timing.attach into a scratch copy of the IR unless it has it), timed
         at its reference placement.
Reports per design: endpoints matched, setup / hold slack Pearson, mean |error|, WNS and TNS of both, and how many of
OpenSTA's worst 1% endpoints Attar also puts in its worst 1%.
--pins: also every pin's setup slack (OpenSTA's slack_max, cached in pins.tsv) against TimingResult.pin_slack: pins
matched, Pearson, mean and max |error|, the same over the pins with negative slack, and the seconds pin_slack takes.
usage: validate_sta.py <design> ... [--platform asap7] [--work DIR] [--json out] [--pins]"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import attar  # noqa: E402
from attar import timing  # noqa: E402
from orfs_roundtrip import RUNS, base_dir, flow_dir, orfs, run_key, w  # noqa: E402

PLATFORMS = '/OpenROAD-flow-scripts/flow/platforms/'

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
'''


DUMP_PINS_TCL = r'''
proc dump_pins {path} {
  set f [open $path w]
  foreach p [concat [get_pins -hierarchical *] [all_inputs] [all_outputs]] {
    puts $f "[get_full_name $p] [get_property $p slack_max]"
  }
  close $f
}
'''


def platform_file(plat, *rel):
    """A platform file on the host (orfs_runs/platform_<plat>/...), copied out of the ORFS image once."""
    h = os.path.join(RUNS, 'platform_' + plat, *rel)
    if not os.path.exists(h):
        os.makedirs(os.path.dirname(h), exist_ok=True)
        orfs('cp %s %s' % (PLATFORMS + plat + '/' + '/'.join(rel), w(h)))
    return h


def lib_paths(d, plat, design):
    """The Liberty files the design was distilled with (by name, from the manifest), as host paths: the platform's,
    else the run's own (e.g. memories ORFS generated during the flow)."""
    from orfs_roundtrip import inputs
    run_libs = {os.path.basename(p): p for p in inputs(base_dir(run_key(plat, design)))[1]}
    out = []
    for ref in d.manifest['source'].get('lib', []):
        for sub in ('lib/NLDM', 'lib', 'lib/CCS'):
            h = os.path.join(RUNS, 'platform_' + plat, sub, ref['name'])
            if os.path.exists(h):
                out.append(h); break
        else:
            if ref['name'] not in run_libs:
                raise FileNotFoundError(ref['name'])
            out.append(run_libs[ref['name']])
    return out


def sdc_path(design, plat):
    base = base_dir(run_key(plat, design))
    return os.path.join(flow_dir(base), '2_floorplan.sdc')


def truth(design, plat, libs, sdc, rc):
    """OpenSTA's endpoint slacks: {name: (setup, hold)}, plus wns / tns / seconds."""
    # cached per design (asap7, as before) or per platform / design
    out_dir = os.path.join(RUNS, 'sta', design) if plat == 'asap7' else os.path.join(RUNS, 'sta', plat, design)
    os.makedirs(out_dir, exist_ok=True)
    tsv = os.path.join(out_dir, 'truth.tsv'); meta = os.path.join(out_dir, 'truth.json')
    if not (os.path.exists(tsv) and os.path.exists(meta)):
        odb = os.path.join(flow_dir(base_dir(run_key(plat, design))), '3_3_place_gp.odb')
        tcl = [DUMP_TCL] + ['read_liberty %s' % w(l) for l in libs] + [
            'read_db %s' % w(odb), 'read_sdc %s' % w(sdc), 'source %s' % w(rc),
            'set t0 [clock milliseconds]', 'estimate_parasitics -placement',
            'set wns [sta::worst_slack -max]', 'set tns [sta::total_negative_slack -max]',
            'puts "STA_DONE [expr {([clock milliseconds] - $t0) / 1000.0}] $wns $tns"',
            'dump_ep %s' % w(tsv)]
        path = os.path.join(RUNS, 'tmp_sta_%s.tcl' % design)
        open(path, 'w').write('\n'.join(tcl) + '\nexit\n')
        try:
            log = orfs('/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad -no_init -no_splash %s' % w(path))
        finally:
            os.remove(path)
        line = [l for l in log.splitlines() if l.startswith('STA_DONE')]
        if not line:
            raise RuntimeError(log[-3000:])
        _, sec, wns, tns = line[-1].split()
        json.dump({'seconds': float(sec), 'worst_slack': float(wns), 'tns': float(tns)}, open(meta, 'w'))
    # OpenSTA reports in the first Liberty file's time unit (ps for ASAP7, ns for Nangate45): to ps
    from attar import liberty
    tu = liberty.read(libs[0])['time_unit_ps']
    ep = {}
    for l in open(tsv):
        t = l.split()
        if len(t) == 3:
            try:
                ep[t[0]] = (float(t[1]) * tu, float(t[2]) * tu)
            except ValueError:
                pass
    m = json.load(open(meta))
    m['worst_slack'] *= tu; m['tns'] *= tu
    return ep, m


def truth_pins(design, plat, libs, sdc, rc):
    """OpenSTA's setup slack of every pin and port: {name (no backslashes): slack ps}, inf where unconstrained."""
    out_dir = os.path.join(RUNS, 'sta', design) if plat == 'asap7' else os.path.join(RUNS, 'sta', plat, design)
    os.makedirs(out_dir, exist_ok=True)
    tsv = os.path.join(out_dir, 'pins.tsv')
    if not os.path.exists(tsv):
        odb = os.path.join(flow_dir(base_dir(run_key(plat, design))), '3_3_place_gp.odb')
        tcl = [DUMP_PINS_TCL] + ['read_liberty %s' % w(l) for l in libs] + [
            'read_db %s' % w(odb), 'read_sdc %s' % w(sdc), 'source %s' % w(rc), 'estimate_parasitics -placement',
            'dump_pins %s' % w(tsv), 'puts PINS_DONE']
        path = os.path.join(RUNS, 'tmp_sta_pins_%s.tcl' % design)
        open(path, 'w').write('\n'.join(tcl) + '\nexit\n')
        try:
            log = orfs('/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad -no_init -no_splash %s' % w(path))
        finally:
            os.remove(path)
        if 'PINS_DONE' not in log:
            raise RuntimeError(log[-3000:])
    from attar import liberty
    tu = liberty.read(libs[0])['time_unit_ps']
    pins = {}
    for l in open(tsv):
        t = l.rsplit(None, 1)
        if len(t) == 2:
            try:
                v = float(t[1]) * tu
            except ValueError:
                continue
            pins[t[0].replace('\\', '')] = v if abs(v) < 1e20 * tu else INF
    return pins


def compare_pins(res, ref):
    """TimingResult.pin_slack against OpenSTA's pin slacks (by name), on the pins both constrain."""
    sync = torch.cuda.synchronize if res.setup.is_cuda else (lambda: None)
    sync(); t = time.time(); s = res.pin_slack(); sync(); sec = time.time() - t
    s = s.cpu().numpy()
    ours = {res._pin_name(p): float(s[p]) for p in np.nonzero(np.isfinite(s))[0]}
    common = [k for k in ref if k in ours and np.isfinite(ref[k])]
    a = np.array([ours[k] for k in common]); b = np.array([ref[k] for k in common])
    neg = b < 0
    e = np.abs(a - b)
    return {'pins_ref': int(sum(np.isfinite(v) for v in ref.values())), 'pins_ours': len(ours), 'pins_matched': len(common),
            'pin_pearson': float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else None,
            'pin_mae': float(e.mean()) if len(a) else None, 'pin_maxe': float(e.max()) if len(a) else None,
            'neg_pins': int(neg.sum()), 'neg_mae': float(e[neg].mean()) if neg.any() else None,
            'neg_maxe': float(e[neg].max()) if neg.any() else None, 'pin_slack_s': sec}


def attar_design(design, plat, libs, sdc, rc, work, dev):
    src = os.path.join(attar.paths.zoo_dir(), 'after_fp', 'orfs_' + plat, design)
    d = attar.load(src, dev)
    if 'timing' not in d.facts:
        dst = os.path.join(work, design if plat == 'asap7' else plat + '_' + design)
        tj = os.path.join(dst, 'timing.json')
        if not os.path.exists(tj) or 'arc_when' not in json.load(open(tj)):   # absent, or attached by an older attar
            shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
            timing.attach(dst, libs, sdc, rc)
        d = attar.load(dst, dev)
    return d


INF = float('inf')


def pct(ours, ref):
    """|ours - ref| as a percentage of |ref| (None when ref is 0: no violation to compare against)."""
    return float(100 * abs(ours - ref) / abs(ref)) if ref else None


def compare(ours, ref, frac=0.01):
    common = [k for k in ref if k in ours and np.isfinite(ref[k][0]) and np.isfinite(ours[k][0]) and abs(ref[k][0]) < 1e20]
    a = np.array([ours[k][0] for k in common]); b = np.array([ref[k][0] for k in common])
    hk = [k for k in common if np.isfinite(ref[k][1]) and np.isfinite(ours[k][1]) and abs(ref[k][1]) < 1e20]
    ha = np.array([ours[k][1] for k in hk]); hb = np.array([ref[k][1] for k in hk])
    n = max(1, int(len(common) * frac))
    top = len(set(np.argsort(a)[:n]) & set(np.argsort(b)[:n])) / n
    r = {'endpoints_ref': len(ref), 'endpoints_ours': len(ours), 'matched': len(common),
         'setup_pearson': float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else None,
         'setup_mae': float(np.abs(a - b).mean()) if len(a) else None,
         'setup_bias': float((a - b).mean()) if len(a) else None,
         'worst_ours': float(a.min()) if len(a) else None, 'worst_ref': float(b.min()) if len(b) else None,
         'tns_ours': float(np.minimum(a, 0).sum()), 'tns_ref': float(np.minimum(b, 0).sum()),
         'top1_overlap': top,
         'wns_err_pct': pct(min(a.min(), 0), min(b.min(), 0)) if len(a) else None,
         'tns_err_pct': pct(np.minimum(a, 0).sum(), np.minimum(b, 0).sum()),
         'hold_pearson': float(np.corrcoef(ha, hb)[0, 1]) if len(ha) > 2 else None,
         'hold_mae': float(np.abs(ha - hb).mean()) if len(ha) else None}
    return r, common, a, b


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--platform', default='asap7')
    ap.add_argument('--work', default=os.path.join(RUNS, 'sta', '_ir'))
    ap.add_argument('--json')
    ap.add_argument('--pins', action='store_true', help="also every pin's setup slack (TimingResult.pin_slack)")
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = {}
    hdr = '%-20s %6s %6s | %8s %7s %7s | %9s %9s %6s | %10s %10s %6s | %5s | %7s %6s | %6s %7s %6s'
    print(hdr % ('design', 'eps', 'match', 'pearson', 'mae', 'bias', 'worst', 'worst ref', 'WNS%', 'TNS', 'TNS ref', 'TNS%', 'top1',
                 'holdP', 'holdE', 'init s', 'sta ms', 'ref s'))
    for design in a.designs:
        rc = platform_file(a.platform, 'setRC.tcl')
        src = attar.load(os.path.join(attar.paths.zoo_dir(), 'after_fp', 'orfs_' + a.platform, design), 'cpu')
        libs = lib_paths(src, a.platform, design)
        sdc = sdc_path(design, a.platform)
        ref, meta = truth(design, a.platform, libs, sdc, rc)
        d = attar_design(design, a.platform, libs, sdc, rc, a.work, dev)
        d.restore('reference')
        from attar.sta import Timer
        sync = torch.cuda.synchronize if dev == 'cuda' else (lambda: None)
        t = time.time(); T = Timer(d); res = T.run(); sync(); build = time.time() - t   # build + first run (capture)
        sec = INF
        for _ in range(5):                                           # a placement's timing: best of 5
            t = time.time(); res = T.run(); sync(); sec = min(sec, time.time() - t)
        ours = res.endpoint_slacks()
        r, *_ = compare(ours, ref)
        r.update({'attar_s': sec, 'attar_build_s': build, 'ref_s': meta['seconds'],
                  'ref_worst_reported': meta['worst_slack']})
        rows[design] = r
        f = lambda v, p='%.3f': '-' if v is None else p % v  # noqa: E731
        print(hdr % (design, r['endpoints_ref'], r['matched'], f(r['setup_pearson'], '%.4f'), f(r['setup_mae'], '%.2f'),
                     f(r['setup_bias'], '%+.2f'), f(r['worst_ours'], '%.2f'), f(r['worst_ref'], '%.2f'),
                     f(r['wns_err_pct'], '%.2f'), f(r['tns_ours'], '%.1f'), f(r['tns_ref'], '%.1f'),
                     f(r['tns_err_pct'], '%.2f'), '%.0f%%' % (100 * r['top1_overlap']),
                     f(r['hold_pearson'], '%.3f'), f(r['hold_mae'], '%.2f'), f(build, '%.1f'), f(1000 * sec, '%.1f'),
                     f(meta['seconds'], '%.2f')), flush=True)
        if a.pins:
            r.update(compare_pins(res, truth_pins(design, a.platform, libs, sdc, rc)))
            print('  pins: %d of %d matched, pearson %s, mae %s ps (max %s); %d negative: mae %s ps (max %s); '
                  'pin_slack %.1f ms' % (r['pins_matched'], r['pins_ref'], f(r['pin_pearson'], '%.5f'),
                                         f(r['pin_mae'], '%.3f'), f(r['pin_maxe'], '%.2f'), r['neg_pins'],
                                         f(r['neg_mae'], '%.3f'), f(r['neg_maxe'], '%.2f'), 1000 * r['pin_slack_s']),
                  flush=True)
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
