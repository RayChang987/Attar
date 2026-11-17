#!/usr/bin/env python3
"""The before_fp hand-off check (docs/validation/fp_handoff.md) for a list of ORFS designs: per design, ORFS's own
floorplan (baseline) and Attar's naive floorplan at a loose and a tight utilization (0.8x and 1.15x ORFS's), each
handed to ORFS with tools/orfs_handoff.py --until 3_5_place_dp. Floorplans are made first (one GPU process), then the
ORFS runs go in parallel lanes. Results: orfs_runs/results/<run key>/handoff_<tag>_3_5_place_dp.json, as
orfs_handoff.py writes them; floorplan statistics in orfs_runs/handoff_sweep/<run key>/floorplans.json.

usage: fp_handoff_sweep.py <design | platform/design> ... [--lanes 3] [--cores 4] [--device cuda:0]
                           [--make-args "LEC_CHECK=0"] [--skip-floorplan]"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import attar  # noqa: E402
from orfs_roundtrip import RUNS, base_dir, split  # noqa: E402

SETTINGS = (('loose', 0.8), ('tight', 1.15))


def zoo_name(key):
    p, d = split(key)
    return 'before_fp/orfs_%s/%s' % (p, d)


def floorplans(key, device):
    out = os.path.join(RUNS, 'handoff_sweep', key); os.makedirs(out, exist_ok=True)
    stats = {}
    for tag, f in SETTINGS:
        d = attar.load(zoo_name(key), device=device)
        u0 = float(d.utilization()[0])
        t = time.time()
        r = d.floorplan(utilization=u0 * f, verbose=False)
        legal, problems = d.floorplan_legal()
        path = os.path.join(out, 'attar_%s.def' % tag)
        d.export(path)
        d.plot(os.path.join(out, 'attar_%s.png' % tag))
        mac = r.get('macros') or {}
        stats[tag] = {'orfs_utilization': u0, 'utilization': float(d.utilization()[0]), 'die_um': [float(d.die_w), float(d.die_h)],
                      'halo': mac.get('halo'), 'legal': legal, 'problems': problems, 'def': path,
                      'seconds': time.time() - t}
        print('%-26s %-5s util %.3f -> %.3f, die %.1f x %.1f, halo %s, legal %s %s' % (
            key, tag, u0, stats[tag]['utilization'], float(d.die_w), float(d.die_h), mac.get('halo'), legal, problems or ''), flush=True)
        del d
    json.dump(stats, open(os.path.join(out, 'floorplans.json'), 'w'), indent=1)
    return stats


def handoff(key, def_path, tag, cores, make_args):
    run = os.path.relpath(base_dir(key), RUNS)
    cmd = [sys.executable, os.path.join(ROOT, 'tools', 'orfs_handoff.py'), key, run, def_path, tag,
           '--cores', str(cores), '--until', '3_5_place_dp']
    if make_args:
        cmd += ['--make-args', make_args]
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    res = os.path.join(RUNS, 'results', key, 'handoff_%s_3_5_place_dp.json' % tag)
    r = json.load(open(res)) if os.path.exists(res) else {'ok': False, 'error': p.stdout[-1500:] + p.stderr[-1500:]}
    print('%-26s %-8s ok %s check_placement %s drv %s kept %s  %.0fs' % (
        key, tag, r.get('ok'), r.get('check_placement'), r.get('drv'), r.get('kept'), time.time() - t), flush=True)
    return key, tag, r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--lanes', type=int, default=3)
    ap.add_argument('--cores', type=int, default=4)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--make-args', default='')
    ap.add_argument('--skip-floorplan', action='store_true', help='reuse orfs_runs/handoff_sweep/<key>/attar_*.def')
    a = ap.parse_args()
    jobs = []
    for key in a.designs:
        out = os.path.join(RUNS, 'handoff_sweep', key)
        if not a.skip_floorplan:
            floorplans(key, a.device)
        jobs.append((key, 'baseline', 'baseline'))
        jobs += [(key, os.path.join(out, 'attar_%s.def' % tag), tag) for tag, _ in SETTINGS]
    with ThreadPoolExecutor(a.lanes) as ex:
        list(ex.map(lambda j: handoff(j[0], j[1], j[2], a.cores, a.make_args), jobs))


if __name__ == '__main__':
    main()
