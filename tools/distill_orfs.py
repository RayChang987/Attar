#!/usr/bin/env python3
"""Distill ORFS runs into the zoo: 3_3_place_gp -> after_fp/orfs_<platform> (placement problems), 2_1_floorplan ->
before_fp/orfs_<platform> (floorplan problems).

usage: distill_orfs.py <design | platform/design> ... [--stage 3_3_place_gp | 2_1_floorplan] [--reuse]
A design without a platform is asap7's. Run directory: orfs_runs/<design>_base, orfs_runs/<platform>/<design>_base
(orfs_run_to_gp.sh). The stage's DEF is written to orfs_runs/rt/<run key>/<stage>.def (needs docker); --reuse
distills an existing one again without docker."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.orfs_roundtrip import STAGES, base_dir, export_and_distill, rt_dir  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
ap.add_argument('designs', nargs='+')
ap.add_argument('--stage', default='3_3_place_gp', choices=STAGES)
ap.add_argument('--reuse', action='store_true', help='distill orfs_runs/rt/<design>/<stage>.def again if it exists')
a = ap.parse_args()
for d in a.designs:
    t = time.time()
    ir, m, lefs = export_and_distill(d, base_dir(d), rt_dir(d), a.stage, reuse=a.reuse)
    c = m['counts']
    print('%-24s %-14s %7d cells %4d macros %7d nets  (%d LEFs)  %.1fs'
          % (d, a.stage, c['cells'], c['macros'], c['nets'], len(lefs), time.time() - t))
