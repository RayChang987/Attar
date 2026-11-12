#!/usr/bin/env python3
"""Attar in a few lines, floorplan mode: load a floorplan problem, make the floorplan, look at it, write it out.

usage: python3 examples/quickstart_floorplan.py [design, default before_fp/gsrc_hard/n100] [utilization, default 0.85;
       '-' keeps the source's die]
The design comes with its netlist only: the die for a given utilization, the IO pins and the macros are the answer.
The macros never overlap; when they do not all fit in the core some stick out, and the floorplan is not legal (the
plot says so). Two ways to place them: GP + LP legalization (default), and, for block-level designs (no rows), B*-tree
packing. Designs from ORFS (before_fp/orfs_asap7/...) are exported as examples/out_<design>_fp.def for ORFS, which draws
the PDN and places."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import attar  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else 'before_fp/gsrc_hard/n100'
util = None if len(sys.argv) > 2 and sys.argv[2] == '-' else float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
out = os.path.dirname(os.path.abspath(__file__))
tag = '_'.join(name.split('/')[-2:])                 # suite_design

# load a design: the netlist (and, from ORFS, the cell library); the die, IO pins and macros are still to decide
d = attar.load(name, device='cuda')
d.status()

# the floorplan: a die for the utilization, then the IO pins and the macros: GP, then a legalizer that never overlaps
d.floorplan(utilization=util)
print('GP + LP:', d.floorplan_legal())
d.plot(os.path.join(out, tag + '_fp.png'))             # outline, macros, IO pins; legal or not, top right
d.status()

# block-level designs: pack the macros with a B*-tree instead (into the core first, then for HPWL)
if 'rows' not in d.facts and len(d.macros):
    p = attar.load(name, device='cuda')
    p.floorplan(utilization=util, packing=True)
    print('packing:', p.floorplan_legal())
    p.plot(os.path.join(out, tag + '_fp_packed.png'))

# from ORFS: write it as a DEF and hand it to ORFS, which draws the PDN, places and goes on
if 'export' in d.capabilities():
    out_def = os.path.join(out, 'out_%s_fp.def' % tag)
    d.export(out_def)
    print('wrote', out_def)
    if 'orfs_floorplan_handoff' in d.capabilities() and d.floorplan_legal()[0]:
        print('hand it to ORFS (minutes):\n  ' + d.handoff_command(out_def))
