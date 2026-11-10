#!/usr/bin/env python3
"""Does Attar's Evaluator agree with OpenROAD? Two checks on the ORFS asap7 designs (zoo/orfs_asap7, from 3_3):

hpwl      Attar's HPWL of ORFS's own global placement (the 'reference' snapshot) vs the HPWL OpenROAD computes from its
          database for the same DEF (pin shape centers, signal nets). The 'Final HPWL' that global_placement prints in
          3_3_place_gp.log is kept for reference: it is measured before port buffering, so it differs slightly.
legality  Attar places the design (naive GP + LG), its checker says legal; the DEF it writes back goes to OpenROAD's
          check_placement. Negative control: 1% of the cells moved by half a site; both must reject it.

usage: validate_orfs.py [design ...]   (default: all of zoo/orfs_asap7) -> docs/validation/orfs_check.{json,md}"""
import io
import contextlib
import json
import os
import re
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import torch  # noqa: E402
import attar  # noqa: E402
from orfs_roundtrip import RUNS, gpl_final_hpwl, inputs, openroad, w  # noqa: E402

OUT = os.path.join(ROOT, 'docs', 'validation')


def check_placement(design, def_path, base, tag):
    """OpenROAD check_placement on a DEF: (passed, number of violations reported)."""
    lefs, _ = inputs(base)
    work = os.path.join(RUNS, 'check'); os.makedirs(work, exist_ok=True)
    dst = os.path.join(work, '%s_%s.def' % (design, tag)); shutil.copy(def_path, dst)
    tcl = '\n'.join(['read_lef %s' % w(l) for l in lefs] + ['read_def %s' % w(dst),
                     'if {[catch {check_placement -verbose} err]} { puts "ATTAR_CHECK FAIL $err" } else { puts "ATTAR_CHECK PASS" }'])
    try:
        log = openroad(tcl, 'check_%s_%s' % (design, tag))
    finally:
        os.remove(dst)
    return 'ATTAR_CHECK PASS' in log, log


HPWL_TCL = """
set block [ord::get_db_block]
set dbu [[ord::get_db_tech] getDbUnitsPerMicron]
set tot 0.0
foreach net [$block getNets] {
  if {[$net getSigType] in {POWER GROUND}} { continue }
  set xs {}; set ys {}
  foreach it [$net getITerms] {
    set b [$it getBBox]
    lappend xs [expr {([$b xMin] + [$b xMax]) / 2.0}]; lappend ys [expr {([$b yMin] + [$b yMax]) / 2.0}]
  }
  foreach bt [$net getBTerms] {
    set b [$bt getBBox]
    lappend xs [expr {([$b xMin] + [$b xMax]) / 2.0}]; lappend ys [expr {([$b yMin] + [$b yMax]) / 2.0}]
  }
  if {[llength $xs] > 1} {
    set xs [lsort -real $xs]; set ys [lsort -real $ys]
    set tot [expr {$tot + [lindex $xs end] - [lindex $xs 0] + [lindex $ys end] - [lindex $ys 0]}]
  }
}
puts "ATTAR_HPWL [expr {$tot / $dbu}]"
"""


def openroad_hpwl(design, def_path, base, tag):
    """HPWL computed inside OpenROAD from its database: per signal net, the bounding box of the centers of the pins'
    shape bounding boxes (dbITerm / dbBTerm getBBox), the definition Attar uses."""
    lefs, _ = inputs(base)
    work = os.path.join(RUNS, 'check'); os.makedirs(work, exist_ok=True)
    dst = os.path.join(work, '%s_%s_hpwl.def' % (design, tag)); shutil.copy(def_path, dst)
    try:
        log = openroad('\n'.join(['read_lef %s' % w(l) for l in lefs] + ['read_def %s' % w(dst), HPWL_TCL]),
                       'hpwl_%s_%s' % (design, tag))
    finally:
        os.remove(dst)
    return float(re.search(r'ATTAR_HPWL ([\d.eE+-]+)', log).group(1))


def main(designs):
    rows = []
    for n in designs:
        name = n.split('/')[1]
        base = os.path.join(RUNS, name + '_base')
        src = os.path.join(RUNS, 'rt', name, '3_3_place_gp.def')     # the DEF it was distilled from: the template
        d = attar.load(n, 'cuda')
        d.restore('reference')                            # ORFS's own global placement
        hp_attar = d.hpwl().item()
        hp_gpl = gpl_final_hpwl(base, name)
        tmp0 = os.path.join(RUNS, 'check', '%s_ref.def' % name); os.makedirs(os.path.dirname(tmp0), exist_ok=True)
        d.export(src, tmp0)
        hp_orfs = openroad_hpwl(name, tmp0, base, 'ref')
        os.remove(tmp0)
        # legality: naive place, Attar's checker, OpenROAD's checker
        d = attar.load(n, 'cuda'); d.unplace()
        with contextlib.redirect_stdout(io.StringIO()):
            d.place()
        ok_attar = attar.Evaluator(d).legality()['legal']
        tmp = os.path.join(RUNS, 'check', '%s_attar.def' % name); os.makedirs(os.path.dirname(tmp), exist_ok=True)
        d.export(src, tmp)
        ok_or, _ = check_placement(name, tmp, base, 'legal')
        # negative control: 1% of the movable cells half a site to the right
        g = torch.Generator().manual_seed(0)
        mv = torch.nonzero(d.movable).flatten().cpu()
        pick = mv[torch.randperm(len(mv), generator=g)[:max(1, len(mv) // 100)]].to(d.device)
        d.pos[pick, 0] += float(d.row_site_w.min()) / 2
        bad_attar = not attar.Evaluator(d).legality()['legal']
        d.export(src, tmp)
        ok_or_bad, _ = check_placement(name, tmp, base, 'broken')
        os.remove(tmp)
        r = {'design': name, 'cells': d.manifest['counts']['cells'], 'hpwl_attar': hp_attar, 'hpwl_openroad': hp_orfs,
             'hpwl_rel_diff': hp_attar / hp_orfs - 1, 'hpwl_gpl_log': hp_gpl,
             'legal_attar': ok_attar, 'legal_openroad': ok_or,
             'broken_rejected_attar': bad_attar, 'broken_rejected_openroad': not ok_or_bad}
        rows.append(r)
        print('%-14s %7d cells  HPWL attar %.3f openroad-db %.3f (%+.5f%%; gpl log %s)  legal: attar %s openroad %s  broken rejected: attar %s openroad %s'
              % (name, r['cells'], hp_attar, hp_orfs, 100 * r['hpwl_rel_diff'], hp_gpl, ok_attar, ok_or, bad_attar, not ok_or_bad), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, 'orfs_check.json'), 'w'), indent=1)
    with open(os.path.join(OUT, 'orfs_check.md'), 'w') as f:
        f.write('| design | cells | HPWL, Attar (um) | HPWL, OpenROAD database (um) | difference | Attar: legal | OpenROAD check_placement | broken placement rejected (Attar / OpenROAD) |\n')
        f.write('| --- | ---: | ---: | ---: | ---: | :---: | :---: | :---: |\n')
        for r in rows:
            f.write('| %s | %d | %.2f | %.2f | %s | %s | %s | %s / %s |\n' % (
                r['design'], r['cells'], r['hpwl_attar'], r['hpwl_openroad'], '%+.5f%%' % (100 * r['hpwl_rel_diff']),
                'yes' if r['legal_attar'] else 'no', 'pass' if r['legal_openroad'] else 'FAIL',
                'yes' if r['broken_rejected_attar'] else 'no', 'yes' if r['broken_rejected_openroad'] else 'no'))


if __name__ == '__main__':
    main(sys.argv[1:] or attar.zoo('orfs_asap7'))
