#!/usr/bin/env python3
"""Naive incremental legalization and detailed placement against OpenROAD, on the same placements.

Per design (a LEF/DEF design of the zoo; LEFs found under --lef-dir as validate_openroad.py does):
  A  Attar places it (naive GP + LG) and exports it: OpenROAD check_placement and HPWL on A.
  B  OpenROAD's own wirelength improvement of A (improve_placement, OpenROAD dpo): HPWL change, check_placement,
     seconds.
  C  Attar's naive.detailed_place of A: HPWL change, Attar legality, then check_placement and HPWL in OpenROAD on its
     DEF (OpenROAD's HPWL must equal Attar's).
  D  incremental legalization: 1% of A's cells moved by up to 20 sites and 2 rows, then naive.legalize(cells=...):
     Attar legality, displacement, HPWL change against A, cells moved that were not named; check_placement.
usage: validate_dp.py <zoo design> ... --lef-dir DIR [--allow-one-site-gaps] [--device cpu] [--cores 8]
                     [--json out.json]"""
import argparse
import contextlib
import io
import json
import os
import re
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import torch  # noqa: E402
import attar  # noqa: E402
from attar import naive  # noqa: E402
import validate_openroad as VO  # noqa: E402
from validate_orfs import HPWL_TCL  # noqa: E402

WORK = VO.WORK


def improve(def_path, lefs, lef_dir, tag, cores):
    """OpenROAD improve_placement on a legal DEF: HPWL before / after, check_placement after, seconds."""
    dst = os.path.join(WORK, os.path.basename(def_path))
    tcl = ['read_lef %s' % l for l in lefs] + ['read_def /work/check/%s' % os.path.basename(dst), HPWL_TCL,
           'set t0 [clock milliseconds]', 'improve_placement',
           'puts "ATTAR_IMPROVE_S [expr {([clock milliseconds] - $t0) / 1000.0}]"',
           'if {[catch {check_placement -verbose} err]} { puts "ATTAR_CHECK FAIL $err" } else { puts "ATTAR_CHECK PASS" }',
           HPWL_TCL]
    log = VO.openroad('\n'.join(tcl), tag, lef_dir, cores)
    h = [float(v) for v in re.findall(r'ATTAR_HPWL ([\d.eE+-]+)', log)]
    s = re.search(r'ATTAR_IMPROVE_S ([\d.]+)', log)
    res = {'hpwl_before': h[0] if h else None, 'hpwl_after': h[1] if len(h) > 1 else None,
           'legal': 'ATTAR_CHECK PASS' in log, 'seconds': float(s.group(1)) if s else None}
    if len(h) < 2:
        res['log_tail'] = log[-1500:]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--lef-dir', required=True)
    ap.add_argument('--allow-one-site-gaps', action='store_true')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--cores', type=int, default=8)
    ap.add_argument('--json')
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    rows = []
    for name in a.designs:
        d = attar.load(name, a.device)
        lefs = VO.find_lefs(d, a.lef_dir)
        if a.allow_one_site_gaps:
            lefs = lefs + [VO.one_site_master(d, [os.path.join(a.lef_dir, l[len('/lefs/'):]) for l in lefs])]
        tag = name.replace('/', '_')
        ev = attar.Evaluator(d)
        r = {'design': name, 'cells': d.manifest['counts']['movable']}
        with contextlib.redirect_stdout(io.StringIO()):
            d.place()
        posA, oriA = d.pos.clone(), d.orient.clone()
        hA = float(d.hpwl())
        r['A'] = {'attar_legal': ev.legality()['legal'], 'attar_hpwl': hA}
        d.export(os.path.join(WORK, tag + '_A.def'))
        okA = lambda c: c['legal'] or (a.allow_one_site_gaps and c['legal_but_one_site_gaps'])  # noqa: E731
        cA = VO.check(d, os.path.join(WORK, tag + '_A.def'), lefs, a.lef_dir, tag + '_A', a.cores)
        r['A'].update({'openroad_legal': okA(cA), 'openroad_hpwl': cA['hpwl']})
        # B: OpenROAD improve_placement of A
        d.export(os.path.join(WORK, tag + '_B.def'))
        b = improve(os.path.join(WORK, tag + '_B.def'), lefs, a.lef_dir, tag + '_B', a.cores)
        os.remove(os.path.join(WORK, tag + '_B.def'))
        if b['hpwl_before'] and b['hpwl_after']:
            b['hpwl_change_pct'] = 100 * (b['hpwl_after'] / b['hpwl_before'] - 1)
        r['B'] = b
        # C: Attar detailed placement of A
        st = naive.detailed_place(d, verbose=False)
        d.export(os.path.join(WORK, tag + '_C.def'))
        cC = VO.check(d, os.path.join(WORK, tag + '_C.def'), lefs, a.lef_dir, tag + '_C', a.cores)
        hC = float(d.hpwl())
        r['C'] = {'hpwl_change_pct': st['hpwl_change_pct'], 'attar_legal': st['legal'], 'seconds': st['seconds'],
                  'openroad_legal': okA(cC), 'openroad_hpwl_rel_diff': hC / cC['hpwl'] - 1 if cC['hpwl'] else None,
                  'gs_moves': st.get('gs_moves', 0), 'gs_swaps': st.get('gs_swaps', 0),
                  'ro_windows': st.get('ro_windows_applied', 0), 'passes': st['passes'],
                  'displacement_mean': st['displacement']['mean']}
        # D: incremental legalization after moving 1% of A's cells
        d.pos = posA.clone(); d.orient = oriA.clone(); d.update_orient()
        g = torch.Generator().manual_seed(0)
        mv = torch.nonzero(d.movable).flatten().cpu()
        pick = mv[torch.randperm(len(mv), generator=g)[:max(1, len(mv) // 100)]].to(d.device)
        sw, hb = float(d.row_site_w.min()), float(d.row_h.min())
        q = posA.clone()
        q[pick, 0] += torch.randint(-20, 21, (len(pick),), generator=g).to(q.device).float() * sw
        q[pick, 1] += torch.randint(-2, 3, (len(pick),), generator=g).to(q.device).float() * hb
        d.pos = q.clone()
        t = time.time()
        st = naive.legalize(d, cells=pick, verbose=False)
        sec = time.time() - t
        named = torch.zeros(len(q), dtype=torch.bool, device=q.device); named[pick] = True
        moved = (d.pos - q).abs().sum(1) > 0
        d.export(os.path.join(WORK, tag + '_D.def'))
        cD = VO.check(d, os.path.join(WORK, tag + '_D.def'), lefs, a.lef_dir, tag + '_D', a.cores)
        dis = ev.displacement(d.pos, q, cells=pick)
        r['D'] = {'named': len(pick), 'mode': st['mode'], 'attar_legal': ev.legality()['legal'], 'openroad_legal': okA(cD),
                  'hpwl_vs_A_pct': 100 * (float(d.hpwl()) / hA - 1), 'disp_mean': dis['mean'], 'disp_max': dis['max'],
                  'unnamed_moved': int((moved & ~named).sum()), 'seconds': sec}
        rows.append(r)
        print(json.dumps(r), flush=True)
        del d
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
