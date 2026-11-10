#!/usr/bin/env python3
"""Distill ORFS asap7 runs (finished up to 3_3_place_gp) into zoo/orfs_asap7. usage: distill_orfs.py <design> ...
Run directory: orfs_runs/<design>_base (run_to_gp.sh or run_baseline.sh)."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.orfs_roundtrip import RUNS, export_and_distill  # noqa: E402

for d in sys.argv[1:]:
    rt = os.path.join(RUNS, 'rt', d); os.makedirs(rt, exist_ok=True)
    t = time.time()
    ir, m, lefs = export_and_distill(d, os.path.join(RUNS, d + '_base'), rt)
    c = m['counts']
    print('%-14s %7d cells %4d macros %7d nets  (%d LEFs)  %.1fs' % (d, c['cells'], c['macros'], c['nets'], len(lefs), time.time() - t))
