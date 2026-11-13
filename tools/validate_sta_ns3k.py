#!/usr/bin/env python3
"""Attar's timer (attar.sta) against OpenSTA in OpenROAD on NS3K designs (zoo after_fp/ns3k), same inputs on both.

The NS3K zoo designs are floorplans (no placement), so per design:
  inputs: the kit's x4 LEFs (tech, cells, the design's macros), its Liberty files (NS3K 5T cells, fakeram macros), the
          SDC the kit's Innovus flow wrote for it (pnr/out/<design>/.../<top>.sdc) without set_driving_cell (its
          BUF_X1 is not an NS3K cell: OpenSTA stops read_sdc there) and with remove_from_collection as OpenSTA's
          delete_from_list, and an RC script from the tech LEF (rc_tcl);
  attar:  a copy of the zoo IR with timing attached, placed by d.place() (positions and orientations kept in <work>/<design>/pos.pt), exported
          as DEF and timed;
  truth:  OpenROAD (the ORFS image) reads the LEFs, Liberty, the exported DEF, the SDC and the RC script, runs
          estimate_parasitics -placement and dumps every endpoint's setup / hold slack (cached).
Reports what tools/validate_sta.py reports.
usage: validate_sta_ns3k.py <design> ... [--work DIR] [--json out]"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import attar  # noqa: E402
from attar import timing  # noqa: E402
from orfs_roundtrip import RUNS, orfs, w  # noqa: E402
from validate_sta import DUMP_TCL, INF, compare  # noqa: E402

KIT = os.path.expanduser('~/pa_pdn/i3dPDK_NS3K')
LEF_DIR = os.path.join(KIT, 'innovus_enablement', 'lef_x4')
LIBS = [os.path.join(KIT, 'FE', 'i3dPDK_NS3K_5T_0V7_27C.lib'), os.path.join(KIT, 'innovus_enablement', 'macros')]
PNR_OUT = os.path.expanduser('~/i3dPDK_NS3K/innovus_enablement/pnr/out')
TECH_LEF = 'i3dPDK_NS3K_5T_tech_unidir_dir.lef'
ROUTING = ('M2', 'M11')            # the kit's flow: setDesignMode -bottomRoutingLayer M2 -topRoutingLayer M11
SIGNAL, CLOCK = 'M3', 'M5'         # set_wire_rc layers, as ORFS platforms do (nangate45: metal3 / metal5)


def rc_tcl(tech_lef, out):
    """An OpenROAD RC script from the tech LEF: per routing layer R = RPERSQ / WIDTH and C = CPERSQDIST x WIDTH +
    2 x EDGECAPACITANCE per um (LEF's wire model), per cut layer its RESISTANCE; in the Liberty's units (kohm, fF).
    The x4 LEF's coefficients are already scaled so that its 4x longer, 4x wider wires keep their real R and C."""
    layers, cur = [], None
    for line in open(tech_lef):
        t = line.split()
        if not t:
            continue
        if t[0] == 'LAYER':
            cur = {'name': t[1]}; layers.append(cur)
        elif cur is not None:
            if t[0] == 'END' and t[1:2] == [cur['name']]:
                cur = None
            elif t[0] == 'TYPE':
                cur['type'] = t[1]
            elif t[0] == 'WIDTH':
                cur['width'] = float(t[1])
            elif t[:2] == ['RESISTANCE', 'RPERSQ']:
                cur['rpersq'] = float(t[2])
            elif t[0] == 'RESISTANCE':
                cur['res'] = float(t[1])
            elif t[:2] == ['CAPACITANCE', 'CPERSQDIST']:
                cur['cpersq'] = float(t[2])
            elif t[0] == 'EDGECAPACITANCE':
                cur['edge'] = float(t[1])
    lines = []
    for L in layers:
        if L.get('type') == 'ROUTING' and 'rpersq' in L:
            r = L['rpersq'] / L['width'] / 1000.0
            c = (L.get('cpersq', 0.0) * L['width'] + 2 * L.get('edge', 0.0)) * 1000.0
            lines.append('set_layer_rc -layer %s -resistance %.6e -capacitance %.6e' % (L['name'], r, c))
    for L in layers:
        if L.get('type') == 'CUT' and 'res' in L:
            lines.append('set_layer_rc -via %s -resistance %.6e' % (L['name'], L['res'] / 1000.0))
    lines += ['set_wire_rc -signal -layer %s' % SIGNAL, 'set_wire_rc -clock -layer %s' % CLOCK,
              'set_routing_layers -signal %s-%s' % ROUTING]
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return out


def flow_sdc(design, top, out):
    """The SDC the kit's flow wrote for this design, for OpenSTA: without set_driving_cell (BUF_X1 is not an NS3K cell)
    and with remove_from_collection (Innovus) as delete_from_list (OpenSTA)."""
    cands = sorted(glob.glob(os.path.join(PNR_OUT, design, '*', '*', top + '.sdc')))
    if not cands:
        raise FileNotFoundError('no flow SDC for %s' % design)
    text = ''.join(l for l in open(cands[0]) if not l.startswith('set_driving_cell'))
    text = text.replace('remove_from_collection', 'delete_from_list')    # OpenSTA's name for it
    with open(out, 'w') as f:
        f.write(text)
    return out


def inputs(src, work):
    """Copy the design's LEFs and Liberty files under work/inputs (visible in the container)."""
    d = os.path.join(work, 'inputs'); os.makedirs(d, exist_ok=True)
    def put(p):
        q = os.path.join(d, os.path.basename(p))
        if not os.path.exists(q):
            shutil.copy(p, q)
        return q
    lefs = [put(os.path.join(LEF_DIR, ref['name'])) for ref in src.manifest['source']['lef']]
    libs = []
    for ref in src.manifest['source'].get('lib', []):
        p = LIBS[0] if os.path.basename(LIBS[0]) == ref['name'] else os.path.join(LIBS[1], ref['name'])
        libs.append(put(p))
    return lefs, libs


def truth(design, out_dir, lefs, libs, def_path, sdc, rc):
    tsv = os.path.join(out_dir, 'truth.tsv'); meta = os.path.join(out_dir, 'truth.json')
    if not (os.path.exists(tsv) and os.path.exists(meta)):
        tcl = [DUMP_TCL] + ['read_lef %s' % w(l) for l in lefs] + ['read_liberty %s' % w(l) for l in libs] + [
            'read_def %s' % w(def_path), 'read_sdc %s' % w(sdc), 'source %s' % w(rc),
            'set t0 [clock milliseconds]', 'estimate_parasitics -placement',
            'set wns [sta::worst_slack -max]', 'set tns [sta::total_negative_slack -max]',
            'puts "STA_DONE [expr {([clock milliseconds] - $t0) / 1000.0}] $wns $tns"', 'dump_ep %s' % w(tsv)]
        path = os.path.join(RUNS, 'tmp_sta_ns3k_%s.tcl' % design)
        open(path, 'w').write('\n'.join(tcl) + '\nexit\n')
        try:
            log = orfs('/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad -no_init -no_splash %s' % w(path))
        finally:
            os.remove(path)
        line = [l for l in log.splitlines() if l.startswith('STA_DONE')]
        if not line:
            raise RuntimeError(log[-3000:])
        _, sec, wns, tns = line[-1].split()
        json.dump({'seconds': float(sec), 'worst_slack': float(wns), 'tns': float(tns)}, open(meta, 'w'))
    # OpenSTA reports in the first Liberty file's time unit (ns for NS3K): to ps
    from attar import liberty
    tu = liberty.read(libs[0])['time_unit_ps']
    ep = {}
    for l in open(tsv):
        t = l.split()
        if len(t) == 3:
            try:
                ep[t[0]] = (float(t[1]) * tu, float(t[2]) * tu)
            except ValueError:
                pass
    m = json.load(open(meta))
    m['worst_slack'] *= tu; m['tns'] *= tu
    return ep, m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--work', default=os.path.join(RUNS, 'sta_ns3k'))
    ap.add_argument('--json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(os.path.join(a.work, 'inputs'), exist_ok=True)
    rc = rc_tcl(os.path.join(LEF_DIR, TECH_LEF), os.path.join(a.work, 'inputs', 'setRC_ns3k.tcl'))
    rows = {}
    hdr = '%-14s %6s %6s | %8s %7s %7s | %9s %9s %6s | %10s %10s %6s | %5s | %7s %6s | %7s %6s'
    print(hdr % ('design', 'eps', 'match', 'pearson', 'mae', 'bias', 'worst', 'worst ref', 'WNS%', 'TNS', 'TNS ref',
                 'TNS%', 'top1', 'holdP', 'holdE', 'sta ms', 'ref s'))
    for design in a.designs:
        zsrc = os.path.join(attar.paths.zoo_dir(), 'after_fp', 'ns3k', design)
        src = attar.load(zsrc, 'cpu')
        out = os.path.join(a.work, design); os.makedirs(out, exist_ok=True)
        lefs, libs = inputs(src, a.work)
        top = os.path.splitext(src.manifest['source']['def']['name'])[0].replace('.in', '')
        sdc = flow_sdc(design, top, os.path.join(out, top + '.sdc'))
        ir = os.path.join(a.work, '_ir', design)
        tj = os.path.join(ir, 'timing.json')
        if not os.path.exists(tj) or 'arc_when' not in json.load(open(tj)):
            shutil.rmtree(ir, ignore_errors=True); shutil.copytree(zsrc, ir)
            timing.attach(ir, libs, sdc, rc)
        d = attar.load(ir, dev)
        pos_pt, def_path = os.path.join(out, 'pos.pt'), os.path.join(out, 'placed.def')
        if os.path.exists(pos_pt):                                   # the placement: positions and orientations
            sv = torch.load(pos_pt, weights_only=True)
            d.pos = sv['pos'].to(d.pos.device); d.orient = sv['orient'].to(d.orient.device); d.update_orient()
        else:
            d.place(verbose=False)
            torch.save({'pos': d.pos.cpu(), 'orient': d.orient.cpu()}, pos_pt)
        if not os.path.exists(def_path):
            d.export(def_path)
        ref, meta = truth(design, out, lefs, libs, def_path, sdc, rc)
        sync = torch.cuda.synchronize if dev == 'cuda' else (lambda: None)
        from attar.sta import Timer
        T = Timer(d); res = T.run(); sync()
        sec = INF
        for _ in range(5):
            t = time.time(); res = T.run(); sync(); sec = min(sec, time.time() - t)
        r, *_ = compare(res.endpoint_slacks(), ref)
        r.update({'attar_s': sec, 'ref_s': meta['seconds'], 'ref_worst_reported': meta['worst_slack'],
                  'top': top, 'unsupported_sdc': json.load(open(tj))['unsupported']})
        rows[design] = r
        f = lambda v, p='%.3f': '-' if v is None else p % v  # noqa: E731
        print(hdr % (design, r['endpoints_ref'], r['matched'], f(r['setup_pearson'], '%.4f'), f(r['setup_mae'], '%.2f'),
                     f(r['setup_bias'], '%+.2f'), f(r['worst_ours'], '%.2f'), f(r['worst_ref'], '%.2f'),
                     f(r['wns_err_pct'], '%.2f'), f(r['tns_ours'], '%.1f'), f(r['tns_ref'], '%.1f'),
                     f(r['tns_err_pct'], '%.2f'), '%.0f%%' % (100 * r['top1_overlap']), f(r['hold_pearson'], '%.3f'),
                     f(r['hold_mae'], '%.2f'), f(1000 * sec, '%.1f'), f(meta['seconds'], '%.2f')), flush=True)
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
