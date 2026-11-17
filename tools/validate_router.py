#!/usr/bin/env python3
"""Attar's global router (attar.route.global_route) against OpenROAD's (grt / FastRoute) on the same netlist and
placement: the zoo design at its reference placement (ORFS 3_3), OpenROAD routed by tools/gr_truth.py (the flow's
layer settings, no pin_access).

Reports per design: total wirelength as grt counts it (each segment + one gcell) and its error; per-net wirelength
(Pearson, median and 90th percentile |error| over nets OpenROAD routes beyond one gcell); total overflow; via count;
run time (Attar's on the device given, OpenROAD's global_route); and timing on the routes: WNS / TNS / endpoint setup
slack of Attar's timer on Attar's routes against OpenSTA's on OpenROAD's routes (estimate_parasitics
-global_routing), next to Attar's timer on OpenROAD's routes (the parasitics alone: tools/validate_gr_parasitics.py).
usage: validate_router.py <design | platform/design> ... [--device cuda:1] [--json out]"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import attar  # noqa: E402
from attar import route  # noqa: E402
from attar.rules import routing as R  # noqa: E402
from gr_truth import same_adjustments, endpoints, layer_adjustments, truth  # noqa: E402
from orfs_roundtrip import split  # noqa: E402
from validate_sta import compare  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--json')
    a = ap.parse_args()
    rows = {}
    hdr = '%-22s %8s %8s %6s | %6s %6s %6s | %6s %6s | %8s %8s | %6s %6s | %8s %8s %8s | %8s %8s'
    print(hdr % ('design', 'WL', 'WL ref', 'WL%', 'netP', 'med%', 'p90%', 'ovfl', 'ref', 'vias', 'vias ref', 's', 'ref s',
                 'WNS', 'WNS ref', 'WNS or', 'slackP', 'mae'))
    for key in a.designs:
        plat, design = split(key)
        out = truth(key)
        tj = json.load(open(os.path.join(out, 'truth.json')))
        d = attar.load(os.path.join(attar.paths.zoo_dir(), 'after_fp', 'orfs_' + plat, design), a.device)
        d.load_reference()
        adj = layer_adjustments(d, plat)                             # what OpenROAD routed with: the zoo must say the same
        mr = d.manifest['routing']
        assert same_adjustments(mr, adj), (mr, adj)
        route.global_route(d, verbose=False)                         # warm-up (kernels, the tree library)
        rt, st = route.global_route(d, verbose=False)
        g = R.grid(d)
        ref = route.read_segments(d, os.path.join(out, 'routes.segs'))
        w_a = rt.wirelength(g['tile'], per_net=True); w_r = ref.wirelength(g['tile'], per_net=True)
        sel = w_r > 2 * g['tile'] / d.dbu
        rel = np.abs(w_a[sel] - w_r[sel]) / w_r[sel]
        # timing: Attar's timer on Attar's routes, and on OpenROAD's (the parasitics alone)
        from attar.timing.sta import Timer                           # via resistance: setRC's, else the zoo's LEF cut_res
        res_a = Timer(d, parasitics='global_routing', routes=rt).run()
        names = d.names['layers']; lv = {n: i + 1 for i, n in enumerate(names)}
        horiz = {i + 1: bool(h) for i, h in enumerate(g['horizontal'])}
        tracks = {}
        for i in range(len(names)):
            i0, p0, c0 = R._layer_tracks(d.raw, i, horiz[i + 1])
            if c0:
                tracks[i + 1] = (i0, p0)
        route.pins_from_shapes(d, ref, route.read_shapes(os.path.join(out, 'shapes.txt')),
                               lv[d.manifest['routing']['layers'][1]], g['tile'], tracks, horiz)
        res_o = Timer(d, parasitics='global_routing', routes=ref).run()
        ep = endpoints(out)
        r_a, *_ = compare(res_a.endpoint_slacks(), ep)
        r_o, *_ = compare(res_o.endpoint_slacks(), ep)
        ov_ref = sum(tj['overflow'] or [0])
        r = {'wl': st['wirelength_um'], 'wl_ref': tj['gr_wl_um'],
             'wl_err_pct': 100.0 * (st['wirelength_um'] - tj['gr_wl_um']) / tj['gr_wl_um'],
             'net_pearson': float(np.corrcoef(w_a[sel], w_r[sel])[0, 1]), 'net_med_err_pct': 100 * float(np.median(rel)),
             'net_p90_err_pct': 100 * float(np.percentile(rel, 90)), 'nets_compared': int(sel.sum()),
             'overflow_3d': st['overflow_3d'], 'overflow_2d': st['overflow_2d'], 'overflow_ref': tj['overflow'],
             'vias': st['vias'], 'vias_ref': ref.vias(), 'seconds': st['seconds'], 'seconds_ref': tj['gr_s'],
             'wns_attar_routes': r_a['worst_ours'], 'wns_ref': r_a['worst_ref'], 'wns_or_routes': r_o['worst_ours'],
             'tns_attar_routes': r_a['tns_ours'], 'tns_ref': r_a['tns_ref'],
             'slack_pearson': r_a['setup_pearson'], 'slack_mae': r_a['setup_mae'],
             'or_routes_slack_mae': r_o['setup_mae'], 'stats': st}
        rows[key] = r
        print(hdr % (key, r['wl'], int(r['wl_ref']), '%+.2f' % r['wl_err_pct'], '%.3f' % r['net_pearson'],
                     '%.1f' % r['net_med_err_pct'], '%.1f' % r['net_p90_err_pct'], int(r['overflow_3d']), ov_ref,
                     r['vias'], r['vias_ref'], '%.2f' % r['seconds'], '%.2f' % r['seconds_ref'],
                     '%.1f' % r['wns_attar_routes'], '%.1f' % r['wns_ref'], '%.1f' % r['wns_or_routes'],
                     '%.4f' % r['slack_pearson'], '%.2f' % r['slack_mae']), flush=True)
        del d, res_a, res_o
        if a.device.startswith('cuda'):
            torch.cuda.empty_cache()
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
