#!/usr/bin/env python3
"""Distill the pa_pdn ORFS designs (i3dPDK_NS3K, Innovus floorplans) into zoo/ns3k/<design>.

Source per design: the Xplace input DEF of one run (floorplan + netlist, cells unplaced, macros fixed),
with the LEF list that run used. usage: distill_pa_pdn.py [design ...]"""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import attar  # noqa: E402

HOME = os.path.expanduser('~')
OUT = HOME + '/i3dPDK_NS3K/innovus_enablement/pnr/out'
ENA = HOME + '/i3dPDK_NS3K/innovus_enablement'
ZOO = os.path.join(os.path.dirname(__file__), '..', 'zoo', 'ns3k')


def source(design):
    """An in.def with a NETS section (some macro-design runs wrote the DEF without it), and its LEF list."""
    for f in sorted(glob.glob('%s/%s/NS3K_5T/*/xplace/*.in.def' % (OUT, design))):
        with open(f, errors='replace') as fh:
            if any(l.startswith('NETS ') for l in fh):
                lefs = [os.path.join(HOME, l.strip()) for l in open(f[:-4] + '.lefs') if l.strip()]
                return f, lefs
    return None, None


def libs(lefs):
    out = glob.glob(HOME + '/i3dPDK_NS3K/FE/*.lib')
    for l in lefs:
        m = os.path.splitext(os.path.basename(l))[0]
        out += glob.glob('%s/macros/%s.lib' % (ENA, m))
    return out


def size(paths):
    return sum(os.path.getsize(p) for p in paths)


if __name__ == '__main__':
    designs = sys.argv[1:] or sorted(os.path.basename(d) for d in glob.glob(OUT + '/*') if glob.glob(d + '/NS3K_5T/*/xplace/*.in.def'))
    print('%-14s %9s %9s %7s %6s %8s %8s %8s %6s' % ('design', 'cells', 'nets', 'macros', 'sec', 'src MB', 'IR MB', 'ratio', 'skip'))
    for d in designs:
        f, lefs = source(d)
        if not f:
            print('%-14s no in.def with NETS' % d); continue
        lb = libs(lefs)
        t = time.time()
        m = attar.distill(f, lefs, os.path.join(ZOO, d), lib_paths=lb,
                          extra={'suite': 'pa_pdn/ns3k', 'geometry_scale': 4, 'slow_loop': 'innovus 21.13 (pa_pdn flow_fast.tcl)'})
        ir = glob.glob(os.path.join(ZOO, d, '*'))
        src = size([f] + lefs + lb)
        c = m['counts']
        print('%-14s %9d %9d %7d %6.1f %8.1f %8.2f %7.0fx %6d' % (d, c['cells'], c['nets'], c['macros'], time.time() - t,
              src / 1e6, size(ir) / 1e6, src / size(ir), c['skipped_pin_refs']), flush=True)
