#!/usr/bin/env python3
"""Plot a distilled design. usage: plot_design.py <zoo name> <out.png> [--ref] [--die N]
--ref: draw the reference placement kept at distillation (e.g. the ORFS GP result)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import attar  # noqa: E402
from attar.viz import plot  # noqa: E402

name, out = sys.argv[1], sys.argv[2]
d = attar.load(name, 'cuda')
if '--ref' in sys.argv:
    d.load_reference()
die = int(sys.argv[sys.argv.index('--die') + 1]) if '--die' in sys.argv else 0
plot(d, out, die=die)
print('wrote', out)
