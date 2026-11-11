#!/usr/bin/env python3
"""Hand an Attar floorplan to ORFS at the floorplan stage (2_1) and let ORFS do the rest (spec 7.2).

The DEF written by Design.export (from a design distilled at ORFS 2_1) carries the die, rows, IO pins and macros
an engine chose. In a copy of the ORFS run, an OpenROAD script applies them to 2_1_floorplan.odb
(initialize_floorplan + the platform's make_tracks when the die changed, place_pin per IO pin, macros FIRM at their
location) and saves the result as both 2_1_floorplan.odb and 2_2_floorplan_macro.odb, the latter newer, so make
skips ORFS's macro placer. place_pin leaves the pins FIXED, so ORFS's place_pins (3_2) and global_placement
-place_ios (3_3) find nothing to place; kept() checks the final DEF. ORFS continues with tapcells, PDN, global
placement, resizing, CTS and routing; Attar's cell placement, if any, is not used.

usage: orfs_handoff.py <design> <run dir under orfs_runs/, e.g. riscv32i_base> <exported DEF> [tag]
                       [--cores 5] [--make-args "LEC_CHECK=0"] [--keep]
       --make-args: extra make variables (use the same as the baseline); --keep keeps the ORFS run copy
       (orfs_runs/rt/<design>_fp/<tag>). The summary and the DEF go to orfs_runs/results/<design>/handoff_<tag>.*"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from attar.deffile import read_def  # noqa: E402
from orfs_roundtrip import RUNS, design_config, metrics, openroad, orfs, w  # noqa: E402

PLATFORM = '/OpenROAD-flow-scripts/flow/platforms/asap7'
SITE = 'asap7sc7p5t'
ORIENT = {'N': 'R0', 'S': 'R180', 'W': 'R90', 'E': 'R270', 'FN': 'MY', 'FS': 'MX', 'FW': 'MYR90', 'FE': 'MXR90'}


def def_pins(path):
    """{pin: (layer, (x0, y0, x1, y1) absolute)} of the placed IO pins of a DEF (signal pins only)."""
    out, buf, on = {}, [], False
    for line in open(path, errors='replace'):
        if line.startswith('PINS'):
            on = True; continue
        if line.startswith('END PINS'):
            break
        if not on:
            continue
        buf.append(line.strip())
        if line.rstrip().endswith(';'):
            s = ' '.join(buf); buf = []
            if 'USE POWER' in s or 'USE GROUND' in s:
                continue
            m = re.search(r'\+\s*LAYER\s+(\S+)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)', s)
            p = re.search(r'\+\s*(PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*(\w+)', s)
            if m and p:
                a, b, c, e = (int(v) for v in m.groups()[1:]); x, y = int(p.group(2)), int(p.group(3))
                if p.group(4) != 'N':
                    raise NotImplementedError('pin orientation %s' % p.group(4))
                out[s.split()[1]] = (m.group(1), (x + a, y + b, x + c, y + e))
    return out


def apply_tcl(src_def, out_def, odb_in, odb_out):
    """OpenROAD script applying the floorplan of out_def (vs the 2_1 source DEF src_def) to odb_in."""
    s, o = read_def(src_def), read_def(out_def)
    dbu = o['dbu']
    um = lambda v: '%.4f' % (v / dbu)
    L = ['read_db %s' % odb_in]
    if tuple(o['die']) != tuple(s['die']) or [r[1:4] for r in o['rows']] != [r[1:4] for r in s['rows']]:
        rx0 = min(r[1] for r in o['rows']); ry0 = min(r[2] for r in o['rows'])
        rx1 = max(r[1] + r[4] * r[6] for r in o['rows'])
        site_h = (sorted({r[2] for r in o['rows']})[1] - sorted({r[2] for r in o['rows']})[0]) if len(o['rows']) > 1 else 0
        ry1 = max(r[2] for r in o['rows']) + site_h
        L.append('initialize_floorplan -die_area {%s} -core_area {%s} -site %s' % (
            ' '.join(um(v) for v in o['die']), ' '.join(um(v) for v in (rx0, ry0, rx1, ry1)), SITE))
        L.append('source %s/openRoad/make_tracks.tcl' % PLATFORM)
    pins = def_pins(out_def)
    for name, (layer, (x0, y0, x1, y1)) in sorted(pins.items()):
        L.append('place_pin -pin_name {%s} -layer %s -location {%s %s} -pin_size {%s %s}' % (
            name, layer, um((x0 + x1) / 2), um((y0 + y1) / 2), um(x1 - x0), um(y1 - y0)))
    src_master = {c[0]: c[1] for c in s['comps']}
    macros = [c for c in o['comps'] if c[2] == 'FIXED' and src_master.get(c[0]) and c[0] in
              {k for k, st in ((c2[0], c2[2]) for c2 in s['comps']) if st != 'FIXED'}]
    L.append('set block [ord::get_db_block]')
    for name, master, st, x, y, orient in macros:
        L.append('set i [$block findInst {%s}]; $i setOrient %s; $i setLocation %d %d; $i setPlacementStatus FIRM'
                 % (name, ORIENT[orient], x, y))
    L.append('puts "attar: [llength [$block getRows]] rows, die [[$block getDieArea] dx] x [[$block getDieArea] dy]"')
    L.append('write_db %s' % odb_out)
    return '\n'.join(L), {'pins': len(pins), 'macros': len(macros), 'die_changed': len(L) > 0 and 'initialize_floorplan' in L[1]}


def prepare(d, base, out_def, tag):
    work = os.path.join(RUNS, 'rt', '%s_fp' % d, tag)
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(base, work, symlinks=True)
    res = os.path.join(work, 'results/asap7', d, 'base')
    for sub in ('results', 'logs', 'reports', 'objects'):
        for f in glob.glob(os.path.join(work, sub, 'asap7', d, 'base', '*')):
            b = os.path.basename(f)
            if b[:1] in '3456' or b.startswith(('2_2', '2_3', '2_4', '2_floorplan', 'route', 'final', 'cts', 'drt', 'congestion', 'rtlmp')):
                os.remove(f) if os.path.isfile(f) or os.path.islink(f) else shutil.rmtree(f)
    src_def = os.path.join(RUNS, 'rt', d, '2_1_floorplan.def')
    od = os.path.join(work, 'attar_floorplan.def'); shutil.copy(out_def, od)
    tcl, info = apply_tcl(src_def, od, w(os.path.join(base, 'results/asap7', d, 'base', '2_1_floorplan.odb')),
                          w(os.path.join(res, '2_1_floorplan.odb')))
    open(os.path.join(work, 'attar_apply.tcl'), 'w').write(tcl)
    log = openroad(tcl, '%s_%s_apply' % (d, tag))
    info['openroad'] = [l for l in log.splitlines() if l.startswith('attar:') or 'ERROR' in l or 'WARNING' in l][:20]
    shutil.copy(os.path.join(res, '2_1_floorplan.odb'), os.path.join(res, '2_2_floorplan_macro.odb'))
    open(os.path.join(res, '2_2_floorplan_macro.tcl'), 'w').write('# macros placed by Attar (attar_apply.tcl)\n')
    t = time.time_ns()
    os.utime(os.path.join(res, '2_1_floorplan.odb'), ns=(t, t))
    for f in ('2_2_floorplan_macro.odb', '2_2_floorplan_macro.tcl'):
        os.utime(os.path.join(res, f), ns=(t + 10**9, t + 10**9))
    return work, info


def run(d, work, cores, make_args=''):
    t = time.time()
    try:
        orfs('make NUM_CORES=%d DESIGN_CONFIG=%s WORK_HOME=%s %s > %s 2>&1'
             % (cores, design_config(d), w(work), make_args, w(os.path.join(work, 'make.log'))))
        ok = True
    except RuntimeError:
        ok = False
    return ok, round(time.time() - t)


def kept(d, work, out_def):
    """Did ORFS keep Attar's macros and IO pins? Compare the final DEF with the hand-off DEF."""
    fin = os.path.join(work, 'results/asap7', d, 'base', '6_final.def')
    if not os.path.exists(fin):
        return None
    f, o = read_def(fin), read_def(out_def)
    fm = {c[0]: (c[3], c[4]) for c in f['comps']}
    om = [c for c in o['comps'] if c[2] == 'FIXED']
    moved_m = sum(1 for c in om if c[0] in fm and fm[c[0]] != (c[3], c[4]))
    fp = {p[0]: (p[3], p[4]) for p in f['pins']}
    op = {p[0]: (p[3], p[4]) for p in o['pins'] if p[2] != 'UNPLACED'}
    moved_p = sum(1 for k, v in op.items() if k in fp and abs(fp[k][0] - v[0]) + abs(fp[k][1] - v[1]) > 200)
    return {'macros_moved': moved_m, 'pins_moved_over_0.2um': moved_p}


def gpl_hpwl(work, d):
    log = os.path.join(work, 'logs/asap7', d, 'base', '3_3_place_gp.log')
    if os.path.exists(log):
        for line in open(log):
            if 'Final HPWL' in line:
                return float(line.split()[-1])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('design'); ap.add_argument('run', help='ORFS run dir under orfs_runs/')
    ap.add_argument('out_def', help='the DEF written by Design.export'); ap.add_argument('tag', nargs='?')
    ap.add_argument('--cores', type=int, default=5)
    ap.add_argument('--make-args', default='', help='extra make variables, e.g. LEC_CHECK=0')
    ap.add_argument('--keep', action='store_true', help='keep the ORFS run copy (a full ORFS run, up to GBs)')
    a = ap.parse_args()
    d, base, out_def = a.design, os.path.join(RUNS, a.run), os.path.abspath(a.out_def)
    tag = a.tag or os.path.splitext(os.path.basename(out_def))[0]
    work, info = prepare(d, base, out_def, tag)
    print('prepared %s: %s' % (work, info), flush=True)
    ok, sec = run(d, work, a.cores, a.make_args)
    rows = [('orfs floorplan', dict(metrics(base, d), gp_hpwl=gpl_hpwl(base, d))),
            ('attar floorplan', dict(metrics(work, d), gp_hpwl=gpl_hpwl(work, d), ok=ok, flow_s=sec, kept=kept(d, work, out_def)))]
    keys = ['gp_hpwl', 'dp_hpwl_est', 'grt_wl', 'routed_wl', 'drc', 'wns', 'tns']
    print('\n%-16s' % 'run' + ''.join('%13s' % k for k in keys))
    for name, m in rows:
        print('%-16s' % name + ''.join('%13s' % ('%.2f' % m[k] if isinstance(m.get(k), float) else m.get(k)) for k in keys))
    print('ok %s, %ds, kept: %s' % (ok, sec, rows[1][1].get('kept')))
    res = os.path.join(RUNS, 'results', d); os.makedirs(res, exist_ok=True)
    json.dump(rows, open(os.path.join(res, 'handoff_%s.summary.json' % tag), 'w'), indent=1)
    shutil.copy(out_def, os.path.join(res, 'handoff_%s.def' % tag))
    if not a.keep:                                    # a run copy is a full ORFS run (up to GBs)
        shutil.rmtree(work, ignore_errors=True)
    print('summary: %s' % os.path.join(res, 'handoff_%s.summary.json' % tag))
