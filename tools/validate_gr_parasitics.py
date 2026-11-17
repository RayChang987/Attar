#!/usr/bin/env python3
"""Attar's global-routing parasitics (Timer(parasitics='global_routing'), attar.timing.grpar) against OpenROAD /
OpenSTA on OpenROAD's own routes.

Per design (orfs_<platform>, its reference placement = ORFS 3_3): tools/gr_truth.py routes it in OpenROAD and dumps
the routes, the pins' on-grid positions, the parasitics (SPEF) and every endpoint's slacks after
estimate_parasitics -global_routing. Attar reads the same routes (attar.route.read_segments), puts the pins where
GlobalRouter does from their shapes (attar.route.pins_from_shapes; checked against write_pin_locations), builds and
reduces the networks and times the design. Reports: pins on the same gcell, per-net wire capacitance and resistance
vs the SPEF, endpoint setup / hold slacks vs OpenSTA (as tools/validate_sta.py).
usage: validate_gr_parasitics.py <design | platform/design> ... [--json out] [--device cpu]"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import attar  # noqa: E402
from attar import route  # noqa: E402
from attar.rules import routing as R  # noqa: E402
from gr_truth import endpoints, truth  # noqa: E402
from orfs_roundtrip import base_dir, inputs, split  # noqa: E402
from validate_sta import compare  # noqa: E402


def lef_cut_res(lefs, layer_names):
    """{routing level l: the LEF RESISTANCE (per cut) of the cut layer between l and l + 1, in kohm}: OpenROAD's
    getCutLayerRes fallback when set_layer_rc has no -via (Nangate45)."""
    out, order, cur, typ = {}, [], None, None
    for p in lefs:
        for line in open(p, errors='replace'):
            t = line.split()
            if not t:
                continue
            if t[0] == 'LAYER' and len(t) > 1:
                cur, typ = t[1], None
            elif t[0] == 'TYPE' and cur:
                typ = t[1]
                order.append((cur, typ))
            elif t[0] == 'RESISTANCE' and typ == 'CUT' and cur:
                out[cur] = float(t[1].rstrip(';')) / 1000.0
            elif t[0] == 'END' and len(t) > 1 and t[1] == cur:
                cur = None
    lv = {n: i + 1 for i, n in enumerate(layer_names)}
    res, last = {}, None
    for name, typ in order:                    # a cut layer sits between the routing layers around it
        if typ == 'ROUTING':
            last = lv.get(name)
        elif typ == 'CUT' and last is not None and name in out:
            res[last] = out[name]
    return res


def spef_nets(path):
    """{net: (total wire cap fF, sum of resistances kohm)} of an OpenROAD SPEF (units FF / KOHM)."""
    out = {}
    net, cap, rs, sec = None, 0.0, 0.0, None
    with open(path) as f:
        for line in f:
            if line.startswith('*D_NET'):
                t = line.split(); net, cap, rs, sec = t[1].replace('\\', ''), float(t[2]), 0.0, None
            elif line.startswith('*RES'):
                sec = 'r'
            elif line.startswith('*CAP') or line.startswith('*CONN'):
                sec = None
            elif line.startswith('*END'):
                out[net] = (cap, rs); net = None
            elif sec == 'r' and net is not None:
                t = line.split()
                if len(t) == 4:
                    rs += float(t[3])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--json')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--truth-pins', action='store_true', help="use write_pin_locations' on-grid positions")
    a = ap.parse_args()
    rows = {}
    hdr = '%-22s %7s %6s | %6s %8s %8s | %7s %7s | %8s %7s %7s | %9s %9s %6s | %10s %10s %6s | %6s %6s | %6s'
    print(hdr % ('design', 'pins', 'grid%', 'loops', 'Ccorr', 'Cerr%', 'Rcorr', 'Rerr%', 'pearson', 'mae', 'max',
                 'worst', 'worst ref', 'WNS%', 'TNS', 'TNS ref', 'TNS%', 'holdP', 'holdE', 'ms'))
    for key in a.designs:
        plat, design = split(key)
        out = truth(key)
        d = attar.load(os.path.join(attar.paths.zoo_dir(), 'after_fp', 'orfs_' + plat, design), a.device)
        d.load_reference()
        rt = route.read_segments(d, os.path.join(out, 'routes.segs'))
        names = d.names['layers']
        lv = {n: i + 1 for i, n in enumerate(names)}
        g = R.grid(d)
        horiz = {i + 1: bool(h) for i, h in enumerate(g['horizontal'])}
        tracks = {}
        for i in range(len(names)):
            init, pitch, cnt = R._layer_tracks(d.raw, i, horiz[i + 1])
            if cnt:
                tracks[i + 1] = (init, pitch)
        hi = lv[d.manifest['routing']['layers'][1]]
        route.pins_from_shapes(d, rt, route.read_shapes(os.path.join(out, 'shapes.txt')), hi, g['tile'], tracks, horiz)
        tg, found = route.read_pin_locations(d, os.path.join(out, 'pins.loc'))
        same = (tg == rt.pin_grid).all(1) & found
        grid_pct = 100.0 * same.sum() / max(1, found.sum())
        if a.truth_pins:
            rt.pin_grid = np.where(found[:, None], tg, rt.pin_grid)
        lefs = inputs(base_dir(key))[0]
        info = json.load(open(os.path.join(d.path, 'timing.json')))
        if not (info.get('rc') or {}).get('via'):                    # the zoo's cut_res must be the LEF's
            zc = {int(k): v for k, v in (d.manifest['routing'].get('cut_res') or {}).items()}
            assert zc == lef_cut_res(lefs, names), (zc, lef_cut_res(lefs, names))
        from attar.timing.sta import Timer
        T = Timer(d, parasitics='global_routing', routes=rt)
        res = T.run()
        sec = float('inf')
        for _ in range(3):
            t = time.time(); res = T.run(); sec = min(sec, time.time() - t)
        # per-net wire C and R vs the SPEF
        sp = spef_nets(os.path.join(out, 'gr.spef'))
        net = T._gr_net
        nid = {n.replace('\\', ''): i for i, n in enumerate(d.names['nets'])}
        rsum = np.bincount(net['node_net'][net['res_a']], net['res_r'], minlength=len(nid))
        ks = [k for k in sp if k in nid]
        oc = np.array([net['wire_cap'][nid[k]] for k in ks]); rc_ = np.array([sp[k][0] for k in ks])
        orr = np.array([rsum[nid[k]] for k in ks]); rr = np.array([sp[k][1] for k in ks])
        r, *_ = compare(res.endpoint_slacks(), endpoints(out))
        r.update({'pins_on_same_gcell_pct': grid_pct, 'loops': T.gr_loops,
                  'cap_pearson': float(np.corrcoef(oc, rc_)[0, 1]), 'cap_err_pct': float(100 * abs(oc.sum() - rc_.sum()) / rc_.sum()),
                  'res_pearson': float(np.corrcoef(orr, rr)[0, 1]), 'res_err_pct': float(100 * abs(orr.sum() - rr.sum()) / rr.sum()),
                  'nets_cap_off_1pct': int((np.abs(oc - rc_) > 0.01 * np.maximum(rc_, 1e-9)).sum()),
                  'nets': len(ks), 'attar_s': sec, 'truth': json.load(open(os.path.join(out, 'truth.json')))})
        rows[key] = r
        f = lambda v, p='%.3f': '-' if v is None else p % v  # noqa: E731
        print(hdr % (key, found.sum(), f(grid_pct, '%.2f'), r['loops'], f(r['cap_pearson'], '%.5f'), f(r['cap_err_pct'], '%.3f'),
                     f(r['res_pearson'], '%.4f'), f(r['res_err_pct'], '%.3f'), f(r['setup_pearson'], '%.5f'),
                     f(r['setup_mae'], '%.3f'), f(r['setup_maxe'], '%.2f'), f(r['worst_ours'], '%.2f'), f(r['worst_ref'], '%.2f'),
                     f(r['wns_err_pct'], '%.2f'), f(r['tns_ours'], '%.1f'), f(r['tns_ref'], '%.1f'), f(r['tns_err_pct'], '%.2f'),
                     f(r['hold_pearson'], '%.3f'), f(r['hold_mae'], '%.2f'), f(1000 * sec, '%.0f')), flush=True)
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
