#!/usr/bin/env python3
"""Attar in a few lines, floorplan mode: load a floorplan problem, make the floorplan, look at it, write it out.

usage: python3 examples/quickstart_floorplan.py [design, default before_fp/orfs_asap7/aes] [utilization, default 0.6]
The design comes with its netlist only (die initialized, IO pins and macros unplaced, no PDN): the die for a given
utilization, the IO pins and the macros are the answer. Attar draws no power grid: the result is written as
examples/out_<design>_fp.def and handed to ORFS, which draws the PDN and places."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import attar  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else 'before_fp/gsrc_hard/n200'
util = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
out = os.path.dirname(os.path.abspath(__file__))
tag = '_'.join(name.split('/')[-2:])                 # suite_design

# load a design: the netlist and the cell library; the die, IO pins and macros are still to decide
d = attar.load(name, device='cuda')
d.status()

# the floorplan: a new die for the utilization, then the IO pins and the macros (on their routing tracks): GP, then
# a legalizer that never overlaps them; block-level designs (no rows) can pack them instead: d.floorplan(..., packing=True)
d.floorplan(utilization=util)
print('floorplan legal:', d.floorplan_legal())
d.plot(os.path.join(out, tag + '_fp.png'))             # die, macros, IO pins
d.status()

# write it as a DEF and hand it to ORFS, which draws the PDN, places and goes on
if 'export' in d.capabilities():
    out_def = os.path.join(out, 'out_%s_fp.def' % tag)
    d.export(out_def)
    print('wrote', out_def)
    if 'orfs_floorplan_handoff' in d.capabilities():
        print('hand it to ORFS (minutes):\n  ' + d.handoff_command(out_def))
