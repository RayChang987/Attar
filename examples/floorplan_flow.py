#!/usr/bin/env python3
"""From an ORFS floorplan (IO pins and macros unplaced) to a legal placement with the Naive engines only.
Each step's answer becomes a fact that unlocks the next step.

usage: python3 examples/floorplan_flow.py [zoo name, default before_fp/orfs_asap7/tinyRocket]"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import attar  # noqa: E402
from attar import naive  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else 'before_fp/orfs_asap7/tinyRocket'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
d = attar.load(name, dev)
t0 = time.time()


def step(label, fn):
    r = fn()
    print('%-34s can do: %s' % (label, ', '.join(d.capabilities())))
    return r


print('%-34s can do: %s' % ('loaded (IO pins, macros unplaced)', ', '.join(d.capabilities())))
step('1. place_io (name order)', lambda: naive.place_io(d, verbose=False))
step('2. place_macros (mixed-size GP)', lambda: naive.place_macros(d, verbose=False))
step('3. global_place', lambda: naive.global_place(d, verbose=False))
step('4. place_io again (net-aware)', lambda: naive.place_io(d, verbose=False))
step('5. global_place', lambda: naive.global_place(d, verbose=False))
step('6. legalize', lambda: naive.legalize(d, verbose=False))
ev = attar.Evaluator(d)
print('\nHPWL %.4g, legal %s, %.1fs in all' % (ev.hpwl().item(), ev.legality()['legal'], time.time() - t0))
print('facts:', {k: v for k, v in d.facts.items() if k.startswith('floorplan')})
print('not available any more:', getattr(d, 'dropped', {}))
png = os.path.join(HERE, 'floorplan_flow_%s.png' % d.manifest['design'])
d.plot(png)
print('figure:', png)
