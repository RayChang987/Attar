#!/usr/bin/env python3
"""Distill the pa_pdn designs (i3dPDK_NS3K, Innovus floorplans) into zoo/ns3k/<design>, and the routed bp_quad DEF
into zoo/stress/bp_quad.

Source per design: the Xplace input DEF of one run (floorplan + netlist, cells unplaced, macros fixed), with the LEF
list that run used. usage: distill_pa_pdn.py [design ...] [--ns3k ~/i3dPDK_NS3K] [--no-stress]"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import attar  # noqa: E402
from attar.paths import zoo_dir  # noqa: E402

STRESS = 'innovus_enablement/pnr/out/bp_quad/NS3K_5T/ns3k_pl_u06_cFdpP_fast/bsg_chip.def'


def source(kit, design):
    """An in.def with a NETS section (some macro-design runs wrote the DEF without it), and its LEF list (the .lefs
    file lists them relative to the folder that holds the kit)."""
    out = os.path.join(kit, 'innovus_enablement/pnr/out')
    for f in sorted(glob.glob('%s/%s/NS3K_5T/*/xplace/*.in.def' % (out, design))):
        with open(f, errors='replace') as fh:
            if any(l.startswith('NETS ') for l in fh):
                lefs = [os.path.join(os.path.dirname(kit), l.strip()) for l in open(f[:-4] + '.lefs') if l.strip()]
                return f, lefs
    return None, None


def libs(kit, lefs):
    out = glob.glob(kit + '/FE/*.lib')
    for l in lefs:
        m = os.path.splitext(os.path.basename(l))[0]
        out += glob.glob('%s/innovus_enablement/macros/%s.lib' % (kit, m))
    return out


def size(paths):
    return sum(os.path.getsize(p) for p in paths)


def report(d, m, t, src, ir):
    c = m['counts']
    print('%-14s %9d %9d %7d %6.1f %8.1f %8.2f %7.0fx %6d' % (d, c['cells'], c['nets'], c['macros'], time.time() - t,
          size(src) / 1e6, size(ir) / 1e6, size(src) / size(ir), c['skipped_pin_refs']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='*')
    ap.add_argument('--ns3k', default='~/i3dPDK_NS3K', help='the i3dPDK_NS3K kit folder')
    ap.add_argument('--no-stress', action='store_true', help='skip stress/bp_quad')
    a = ap.parse_args()
    kit = os.path.abspath(os.path.expanduser(a.ns3k))
    out = os.path.join(kit, 'innovus_enablement/pnr/out')
    designs = a.designs or sorted(os.path.basename(d) for d in glob.glob(out + '/*') if glob.glob(d + '/NS3K_5T/*/xplace/*.in.def'))
    print('%-14s %9s %9s %7s %6s %8s %8s %8s %6s' % ('design', 'cells', 'nets', 'macros', 'sec', 'src MB', 'IR MB', 'ratio', 'skip'))
    for d in designs:
        f, lefs = source(kit, d)
        if not f:
            print('%-14s no in.def with NETS' % d); continue
        lb = libs(kit, lefs)
        t = time.time()
        ir = os.path.join(zoo_dir(), 'ns3k', d)
        m = attar.distill(f, lefs, ir, lib_paths=lb,
                          extra={'suite': 'pa_pdn/ns3k', 'geometry_scale': 4, 'flow': 'innovus 21.13 (pa_pdn flow_fast.tcl)'})
        report(d, m, t, [f] + lefs + lb, glob.glob(ir + '/*'))
    if not a.no_stress and not a.designs:
        f = os.path.join(kit, STRESS)
        ena = os.path.join(kit, 'innovus_enablement/lef_x4')
        lefs = [ena + '/i3dPDK_NS3K_5T_tech_unidir_dir.lef', ena + '/NS3K_5T_cells.lef'] + sorted(glob.glob(ena + '/fakeram*.lef'))
        t = time.time()
        ir = os.path.join(zoo_dir(), 'stress', 'bp_quad')
        m = attar.distill(f, lefs, ir, extra={'suite': 'stress (pa_pdn routed DEF)', 'geometry_scale': 4})
        report('bp_quad', m, t, [f] + lefs, glob.glob(ir + '/*'))
