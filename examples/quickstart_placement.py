#!/usr/bin/env python3
"""Attar quickstart, placement mode: load a placement problem, place it, evaluate and time it, write it out.

usage: python3 examples/quickstart_placement.py [design, default after_fp/orfs_asap7/aes]
The design comes with its floorplan (die, IO pins, macros, rows, PDN); only the cells are to place."""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))
import attar  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else 'after_fp/orfs_asap7/aes'
tag = '_'.join(name.split('/')[-2:])                   # <suite>_<design>

d = attar.load(name, device='cuda')                    # downloads the design on first use; the cells are unplaced
d.status()                                             # what the design has and what it can do
d.plot(os.path.join(here, tag + '_floorplan.png'))

d.place()                                              # global placement + legalization
d.plot(os.path.join(here, tag + '_placed.png'))
print(attar.Evaluator(d).evaluate(metrics=('hpwl', 'steiner_wirelength', 'legality')))

if 'sta' in d.capabilities():                          # designs with Liberty + SDC (ORFS, NS3K, ICCAD 2025)
    d.timing_report(endpoints=5, path=False)

if 'gr_overflow' in d.capabilities():                  # LEF/DEF designs: global-routing overflow estimate (RUDY)
    print('GROV %.2f%% of gcell edges overflow' % d.gr_overflow()['grov_%'])

if 'export' in d.capabilities():                       # LEF/DEF designs: the placement in the source DEF's frame
    d.export(os.path.join(here, 'out_%s.def' % tag))
