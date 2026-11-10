#!/usr/bin/env python3
"""Attar in a few lines: load a design, floorplan it, place it, look at it.

usage: python3 examples/quickstart.py [design, default orfs_asap7_2_1/aes] [utilization, default 0.6]"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import attar  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else 'orfs_asap7_2_1/aes'
util = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
out = os.path.dirname(os.path.abspath(__file__))
tag = name.replace('/', '_')

# load a design: a netlist on a floorplan from ORFS 2_1 (IO pins not placed yet)
d = attar.load(name, device='cuda')
d.status()

# floorplan: a die for the utilization, IO pins, macros
d.floorplan(utilization=util)
d.plot(os.path.join(out, tag + '_floorplan.png'))

# place the cells: global placement + legalization
d.place()
d.plot(os.path.join(out, tag + '_placed.png'))

d.status()
print(attar.Evaluator(d).evaluate(metrics=('hpwl', 'steiner_wirelength', 'legality')))

# write it back to DEF; designs from an ORFS floorplan (2_1) can be handed to ORFS to route
if 'export' in d.capabilities():
    out_def = os.path.join(out, 'out_%s.def' % tag)
    d.export(out_def)
    print('wrote', out_def)
    if 'orfs_floorplan_handoff' in d.capabilities() or 'orfs_handoff' in d.capabilities():
        print('hand off to ORFS (minutes):\n  ' + d.handoff_command(out_def))
