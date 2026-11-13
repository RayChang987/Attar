#!/usr/bin/env python3
"""Reference global routes for the ggr.cu correlation study: ORFS (its own make, from the finished 3_3 base run)
through CTS and 5_1_grt, then on 5_1_grt.odb:
  - write_def (the input ggr_attar routes: the very netlist and placement ORFS routed),
  - a fresh global_route with ORFS's settings (platform fastroute.tcl, pin_access, -congestion_iterations 30):
    its segments (write_global_route_segments), the per-layer resources and demand of its congestion report,
    wirelength and run time. (dbGCellGrid's capacity / usage is not FastRoute's edge state: raw tracks, and
    blocked tracks counted as usage; so the segments are the reference.)
Then ggr_attar on the same DEF with the same layer range and adjustment. The run copy is deleted afterwards; the
dumps stay in orfs_runs/grtcorr/<design>/.

usage: orfs_grt.py <design> ... [--par 4] [--cores 4] [--stress ADJ] [--keep]
--stress ADJ: also route both ways with every layer's adjustment raised to ADJ (a congested variant)."""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from orfs_roundtrip import RUNS, base_dir, design_config, flow_dir, inputs, orfs, w  # noqa: E402

GGR = os.path.join(HERE, 'ggr_attar')
OUT = os.path.join(RUNS, 'grtcorr')



def make_run(d, work, cores):
    """Copy the 3_3 base run and let ORFS's make continue to 5_1_grt (its own flow, nothing injected)."""
    base = base_dir(d)
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(base, work, symlinks=True)
    r = flow_dir(work)
    for sub in ('results', 'logs', 'reports'):
        for f in glob.glob(os.path.join(work, sub, '*', '*', 'base', '*')):
            b = os.path.basename(f)
            if b[:1] in '3456' and not b.startswith(('3_1', '3_2', '3_3')) or b.startswith(('route', 'final', 'cts', 'drt', 'congestion')):
                os.remove(f) if os.path.isfile(f) else shutil.rmtree(f)
    # the same timestamp fixes as orfs_roundtrip._inject: make must not redo synthesis or global placement
    rtlil, y2 = os.path.join(r, '1_1_yosys_canonicalize.rtlil'), os.path.join(r, '1_2_yosys.v')
    if not os.path.exists(rtlil) and os.path.exists(y2):
        open(rtlil, 'w').close()
        t0 = os.stat(y2).st_mtime_ns - 10**9
        os.utime(rtlil, ns=(t0, t0))
    p32 = r + '/3_2_place_iop.odb'
    if not os.path.exists(p32):
        shutil.copy(r + '/3_1_place_gp_skip_io.odb', p32)
    t1 = os.stat(r + '/3_1_place_gp_skip_io.odb').st_mtime_ns + 10**9
    os.utime(p32, ns=(t1, t1))
    t3 = time.time_ns(); os.utime(r + '/3_3_place_gp.odb', ns=(t3, t3))
    t = time.time()
    orfs('make NUM_CORES=%d DESIGN_CONFIG=%s WORK_HOME=%s grt > %s 2>&1'
         % (cores, design_config(d), w(work), w(os.path.join(work, 'make.log'))))
    if os.stat(r + '/3_3_place_gp.odb').st_mtime_ns != t3:
        raise RuntimeError('%s: make redid global placement' % d)
    return r, round(time.time() - t)


def dump(d, r, dst, routing, stress, cores):
    """write_def, ORFS's congestion map, and fresh global routes (ORFS settings; with --stress also adj=stress)."""
    lo, hi = routing['layers']
    tcl = ['read_db %s' % w(r + '/5_1_grt.odb'), 'write_def %s' % w(dst + '/grt.def')]
    for tag, adj in [('fresh', routing['adjustment'])] + ([('stress', stress)] if stress else []):
        tcl += ['set ::env(MIN_ROUTING_LAYER) %s' % lo, 'set ::env(MAX_ROUTING_LAYER) %s' % hi,
                'set ::env(MIN_CLK_ROUTING_LAYER) M4', 'set ::env(ROUTING_LAYER_ADJUSTMENT) %g' % adj,
                'source /OpenROAD-flow-scripts/flow/platforms/asap7/fastroute.tcl',
                'pin_access',
                'set t0 [clock milliseconds]',
                'global_route -congestion_iterations 30 -allow_congestion -verbose',
                'puts "GR_DONE %s [expr {([clock milliseconds] - $t0) / 1000.0}]"' % tag,
                'write_global_route_segments %s' % w(dst + '/%s.segs' % tag)]
    # one script per output dir: several runs of one design may dump at once
    path = os.path.join(RUNS, 'tmp_grtcorr_%s.tcl' % os.path.relpath(dst, RUNS).replace('/', '_'))
    with open(path, 'w') as f:
        f.write('\n'.join(tcl) + '\nexit\n')
    try:
        log = orfs('/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad -no_init -no_splash -threads %d %s'
                   % (cores, w(path)))
    finally:
        os.remove(path)
    open(os.path.join(dst, 'openroad.log'), 'w').write(log)
    res = {}
    # each global_route prints its own report; split the log at GR_DONE markers
    for part, tag, sec in re.findall(r'(.*?)GR_DONE (\w+) ([\d.]+)', log, re.S):
        wl = re.findall(r'Total wirelength:\s*([\d.]+)\s*um', part)
        ov = re.findall(r'Total\s+\d+\s+\d+\s+[\d.]+%\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)', part)
        rep = part[part.rfind('Final congestion report'):]
        layers = {l: [int(c), int(dm)] for l, c, dm in re.findall(r'^(\w+)\s+(\d+)\s+(\d+)\s+[\d.]+%', rep, re.M) if l != 'Total'}
        res[tag] = {'gr_s': float(sec), 'wl_um': float(wl[-1]) if wl else None,
                    'overflow': [int(v) for v in ov[-1]] if ov else None, 'layers': layers}
    return res


def ggr(d, dst, routing, lefs, stress):
    tech = [l for l in lefs if 'tech' in os.path.basename(l)][0]
    args = ['--tech', tech] + sum((['--lef', l] for l in lefs if l != tech), [])
    lo, hi = (int(v[1:]) for v in routing['layers'])
    res = {}
    for tag, adj in [('ggr', routing['adjustment'])] + ([('ggr_stress', stress)] if stress else []):
        t = time.time()
        p = subprocess.run([GGR, dst + '/grt.def', dst + '/' + tag, '--min', str(lo), '--max', str(hi), '--adj', str(adj)] + args,
                           capture_output=True, text=True)
        open(dst + '/%s.log' % tag, 'w').write(p.stdout + p.stderr)
        if p.returncode:
            raise RuntimeError('%s %s: %s' % (d, tag, p.stderr[-2000:]))
        res[tag] = json.load(open(dst + '/%s.json' % tag)); res[tag]['wall_s'] = round(time.time() - t, 2)
    return res


def one(d, a):
    dst = os.path.join(OUT, d)
    os.makedirs(dst, exist_ok=True)
    summ = os.path.join(dst, 'summary.json')
    if os.path.exists(summ) and not a.redo:
        return json.load(open(summ))
    work = os.path.join(OUT, '_work', d)
    m = json.load(open(os.path.join(ROOT, 'zoo/after_fp/orfs_asap7', d, 'manifest.json')))
    routing = m['routing']
    t = time.time()
    try:
        r, make_s = make_run(d, work, a.cores)
        orfs_metrics = json.load(open(os.path.join(flow_dir(work, 'logs'), '5_1_grt.json')))
        res = {'design': d, 'routing': routing, 'make_s': make_s,
               'orfs_grt_wl': orfs_metrics.get('globalroute__global_route__wirelength')}
        res.update(dump(d, r, dst, routing, a.stress, a.cores))
        lefs, _ = inputs(base_dir(d))
        res.update(ggr(d, dst, routing, lefs, a.stress))
        res['ok'] = True
    except Exception as e:  # noqa: BLE001
        res = {'design': d, 'ok': False, 'error': str(e)[-3000:]}
    res['total_s'] = round(time.time() - t)
    if res.get('ok'):
        json.dump(res, open(summ, 'w'), indent=1)
    if not a.keep:
        if os.path.exists(os.path.join(work, 'make.log')):
            shutil.copy(os.path.join(work, 'make.log'), os.path.join(dst, 'make.log'))
        shutil.rmtree(work, ignore_errors=True)
    print('%s: %s in %ds' % (d, 'ok' if res.get('ok') else 'FAILED ' + res.get('error', '')[-300:], res['total_s']), flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--par', type=int, default=4)
    ap.add_argument('--cores', type=int, default=4)
    ap.add_argument('--stress', type=float, default=None)
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--redo', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with ThreadPoolExecutor(a.par) as P:
        list(P.map(lambda d: one(d, a), a.designs))
