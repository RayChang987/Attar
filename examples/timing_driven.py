#!/usr/bin/env python3
"""Timing-driven placement in a few lines: place for wirelength, weight the nets by their slack the way OpenROAD's
timing-driven global placement does (gpl: the worst 10% of the nets, weights 1 to 5 by criticality), place again.

usage: python3 examples/timing_driven.py [design, default after_fp/orfs_asap7/cva6]"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))
import attar  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else 'after_fp/orfs_asap7/cva6'
tag = '_'.join(name.split('/')[-2:])                   # <suite>_<design>

d = attar.load(name, device='cuda')
d.place()                                              # wirelength only
d.plot(os.path.join(here, tag + '_wirelength.png'))
d.timing_report(endpoints=0, path=False)

s = d.timing().net_slack()                             # every net's setup slack, as OpenSTA's Sta::netSlack
v = s[s.isfinite()].sort().values
v = v[:len(v) // 10]                                   # OpenROAD gpl: the worst 10% of the nets
lo, hi = v[len(v) // 50], v[-1]                        # its weight ramp (anchored at 0.2% of the nets)
w = d.n_weight
d.n_weight = w * (1 + 4 * ((hi - s) / (hi - lo)).clamp(0, 1))   # weight 1 .. 5 by criticality
d.place(verbose=False)                                 # timing-driven
d.n_weight = w                                         # back to the netlist's weights (HPWL, plots)

d.timing_report(endpoints=0, path=False)
d.plot(os.path.join(here, tag + '_timing_driven.png'))
