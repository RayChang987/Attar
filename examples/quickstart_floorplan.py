#!/usr/bin/env python3
"""Attar quickstart, floorplan mode: load a floorplan problem, make the floorplan, look at it, write it out.

usage: python3 examples/quickstart_floorplan.py [design, default before_fp/gsrc_hard/n100]
                                                [utilization, default 0.85; '-' keeps the source's die]
Only the netlist is given: the die (for the utilization), the IO pins and the macros are the answer."""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))
import attar  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else 'before_fp/gsrc_hard/n100'
util = None if sys.argv[2:3] == ['-'] else float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
tag = '_'.join(name.split('/')[-2:])                   # <suite>_<design>

d = attar.load(name, device='cuda')                    # the netlist; die, IO pins and macros are still open
d.status()

d.floorplan(utilization=util)                          # die, IO pins, macros: GP, then a legalizer that never overlaps
print('GP + LP legal:', d.floorplan_legal())
d.plot(os.path.join(here, tag + '_fp.png'))

if 'rows' not in d.facts and len(d.macros):            # block-level designs: B*-tree packing instead
    p = attar.load(name, device='cuda')
    p.floorplan(utilization=util, packing=True)
    print('packing legal:', p.floorplan_legal())
    p.plot(os.path.join(here, tag + '_fp_packed.png'))

if 'export' in d.capabilities():                       # designs from ORFS: a DEF for ORFS to place and route
    d.export(os.path.join(here, 'out_%s_fp.def' % tag))
