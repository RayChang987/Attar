#!/usr/bin/env python3
"""Attar in a few lines: load a placement problem, place it, look at it, write it back.

usage: python3 examples/quickstart_placement.py [design, default after_fp/orfs_asap7/aes]
The design comes with its floorplan and PDN (an ORFS run at global placement): only the cells are to place. The
result is written as examples/out_<design>.def and can be handed back to ORFS to route."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import attar  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else 'after_fp/orfs_asap7/aes'
out = os.path.dirname(os.path.abspath(__file__))
tag = '_'.join(name.split('/')[-2:])                 # suite_design

# load a design: netlist, die, IO pins, macros, rows and PDN as the flow made them; the cells are unplaced
d = attar.load(name, device='cuda')
d.status()
d.plot(os.path.join(out, tag + '_floorplan.png'))      # what is given: die, macros, IO pins, PDN

# place the cells: global placement + legalization (at the flow's own target density, rails and PDN respected)
d.place()
d.plot(os.path.join(out, tag + '_placed.png'))

d.status()
print(attar.Evaluator(d).evaluate(metrics=('hpwl', 'steiner_wirelength', 'legality')))

# write it as a DEF (the source's PDN, pins and fixed cells kept as they were) and hand it to ORFS to route
if 'export' in d.capabilities():
    out_def = os.path.join(out, 'out_%s.def' % tag)
    d.export(out_def)
    print('wrote', out_def)
    if 'orfs_handoff' in d.capabilities():
        print('hand it to ORFS (minutes):\n  ' + d.handoff_command(out_def))
