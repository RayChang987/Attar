#!/usr/bin/env python3
"""Hand Attar's DEF to OpenROAD and check it there, for designs with no ORFS platform (e.g. i3dPDK_NS3K).

Per design (a LEF/DEF design of the zoo):
  1. Attar places it (naive GP + LG) and exports it (Design.export: the source frame + the cells and nets).
  2. OpenROAD reads the LEFs and the DEF: instance, net and IO counts must equal the design's.
  3. check_placement must pass, and OpenROAD's HPWL (pin shape centers, signal nets) must equal Attar's. Violations are
     counted per category; one-site gaps are reported apart: OpenROAD forbids them when the library has no
     one-site filler (as NS3K), a rule the NS3K reference flow (Innovus) does not follow either.
  4. --grt: global routing (set_routing_layers --layers) must finish; its wirelength and overflow are reported.
  5. Negative control: 1% of the cells moved by half a site; check_placement must reject it.

usage: validate_openroad.py <zoo design> ... --lef-dir DIR [--layers M2-M7] [--grt] [--allow-one-site-gaps]
                          [--cores 8] [--json out.json]
LEFs are found under --lef-dir by the name and sha256 the manifest records. Runs OpenROAD from the ORFS docker
image, with orfs_runs/ and the LEF dir mounted."""
import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import torch  # noqa: E402
import attar  # noqa: E402
from orfs_roundtrip import IMAGE, RUNS  # noqa: E402
from validate_orfs import HPWL_TCL  # noqa: E402

WORK = os.path.join(RUNS, 'check')


def find_lefs(d, lef_dir):
    """The design's LEF files under lef_dir (by name, sha256-checked), as container paths under /lefs."""
    index = {}
    for dp, _, fs in os.walk(lef_dir):
        for f in fs:
            if f.endswith('.lef'):
                index.setdefault(f, []).append(os.path.join(dp, f))
    out = []
    for ref in d.manifest['source']['lef']:
        for p in index.get(ref['name'], []):
            if hashlib.sha256(open(p, 'rb').read()).hexdigest() == ref['sha256']:
                out.append('/lefs/' + os.path.relpath(p, lef_dir)); break
        else:
            raise FileNotFoundError('LEF %s (sha256 %s...) not under %s' % (ref['name'], ref['sha256'][:12], lef_dir))
    return out


def one_site_master(d, lefs_host):
    """A LEF with one dummy one-site-wide CORE SPACER master on the design's row site. OpenROAD forbids one-site gaps
    exactly when the library has no such master (odb::hasOneSiteMaster; -disallow_one_site_gaps is deprecated), so
    reading it turns that check off. Used for libraries without fillers whose own flow allows the gaps (NS3K)."""
    import gzip
    from attar.lef import read_lef
    with gzip.open(os.path.join(d.path, 'frame.def.gz'), 'rt') as f:
        site = next(l.split()[2] for l in f if l.startswith('ROW '))
    _, tech = read_lef(lefs_host)
    w, h = tech['sites'][site]
    path = os.path.join(WORK, 'attar_fill1_%s.lef' % site)
    with open(path, 'w') as f:
        f.write('VERSION 5.8 ;\nMACRO ATTAR_FILL1\n  CLASS CORE SPACER ;\n  ORIGIN 0 0 ;\n  SIZE %g BY %g ;\n'
                '  SYMMETRY X Y ;\n  SITE %s ;\nEND ATTAR_FILL1\nEND LIBRARY\n' % (w, h, site))
    return '/work/check/' + os.path.basename(path)


def openroad(tcl, tag, lef_dir, cores):
    path = os.path.join(WORK, 'tmp_%s.tcl' % tag)
    with open(path, 'w') as f:
        f.write(tcl + '\nexit\n')
    try:
        r = subprocess.run(['docker', 'run', '--rm', '-u', '%d:%d' % (os.getuid(), os.getgid()), '-v', RUNS + ':/work',
                            '-v', os.path.abspath(lef_dir) + ':/lefs:ro', IMAGE, 'bash', '-c',
                            '/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad -no_init -no_splash -threads %d /work/check/%s'
                            % (cores, os.path.basename(path))], capture_output=True, text=True)
        return r.stdout + r.stderr
    finally:
        os.remove(path)


def check(d, def_path, lefs, lef_dir, tag, cores, grt=None):
    """OpenROAD on an exported DEF: counts, check_placement, HPWL and (grt: a layer range) global routing."""
    dst = os.path.join(WORK, os.path.basename(def_path))
    if os.path.abspath(def_path) != dst:
        os.replace(def_path, dst)
    tcl = ['read_lef %s' % l for l in lefs] + ['read_def /work/check/%s' % os.path.basename(dst),
           'set b [ord::get_db_block]',
           'puts "ATTAR_COUNTS [llength [$b getInsts]] [llength [$b getNets]] [llength [$b getBTerms]]"',
           'if {[catch {check_placement -verbose -report_file_name /work/check/%s.dpl.json} err]} { puts "ATTAR_CHECK FAIL $err" } else { puts "ATTAR_CHECK PASS" }' % tag,
           HPWL_TCL]
    if grt:
        tcl += ['set_routing_layers -signal %s' % grt,
                'if {[catch {global_route -allow_congestion -verbose} err]} { puts "ATTAR_GRT FAIL $err" } else { puts "ATTAR_GRT PASS" }']
    t = time.time()
    log = openroad('\n'.join(tcl), tag, lef_dir, cores)
    os.remove(dst)
    m = re.search(r'ATTAR_COUNTS (\d+) (\d+) (\d+)', log)
    rep = os.path.join(WORK, '%s.dpl.json' % tag)
    cats = {}
    if os.path.exists(rep):                      # violations per check_placement category
        cats = {k: len(v.get('violations', [])) for k, v in json.load(open(rep))['DPL']['category'].items()}
        os.remove(rep)
    res = {'openroad_s': round(time.time() - t), 'read': bool(m), 'legal': 'ATTAR_CHECK PASS' in log,
           'counts': [int(v) for v in m.groups()] if m else None, 'violations': cats,
           'legal_but_one_site_gaps': 'ATTAR_CHECK PASS' in log or (bool(cats) and set(k for k, v in cats.items() if v) <= {'One_site_gap_failures'})}
    h = re.search(r'ATTAR_HPWL ([\d.eE+-]+)', log)
    res['hpwl'] = float(h.group(1)) if h else None
    if grt:
        res['grt'] = 'ATTAR_GRT PASS' in log
        wl = re.findall(r'Total wirelength:\s*([\d.]+)\s*um', log) or re.findall(r'[Ww]irelength[^\d]*([\d.]+)', log)
        res['grt_wl'] = float(wl[-1]) if wl else None
        ov = re.findall(r'Total\s+(\d+)\s+\d+\s+[\d.]+%\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)', log)
        res['grt_overflow'] = int(ov[-1][1]) if ov else None
    if not m or (grt and not res.get('grt')):
        res['log_tail'] = log[-1500:]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--lef-dir', required=True)
    ap.add_argument('--layers', default=None, help='signal routing layers for --grt, e.g. M2-M7')
    ap.add_argument('--grt', action='store_true')
    ap.add_argument('--cores', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--allow-one-site-gaps', action='store_true',
                    help='read a dummy one-site master so that check_placement allows one-site gaps (see one_site_master)')
    ap.add_argument('--json')
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    rows = []
    for name in a.designs:
        d = attar.load(name, a.device)
        lefs = find_lefs(d, a.lef_dir)
        if a.allow_one_site_gaps:
            lefs = lefs + [one_site_master(d, [os.path.join(a.lef_dir, l[len('/lefs/'):]) for l in lefs])]
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            d.place()
        ev = attar.Evaluator(d)
        r = {'design': name, 'cells': d.manifest['counts']['cells'], 'nets': d.manifest['counts']['nets'],
             'terminals': d.manifest['counts']['terminals'], 'attar_legal': ev.legality()['legal'],
             'attar_hpwl': d.hpwl().item(), 'place_s': round(time.time() - t, 1)}
        tag = name.replace('/', '_')
        t = time.time(); d.export(os.path.join(WORK, tag + '.def')); r['export_s'] = round(time.time() - t, 1)
        r.update(check(d, os.path.join(WORK, tag + '.def'), lefs, a.lef_dir, tag, a.cores, a.layers if a.grt else None))
        # OpenROAD counts the PG pins of the DEF's PINS too (ORFS DEFs have VDD/VSS there); the IR skips them
        pg = d.manifest['counts'].get('pg_pins_skipped', 0)
        r['counts_match'] = r['counts'] is not None and r['counts'][0] == len(d.pos) - (len(d.io) if d.io is not None else 0) \
            and r['counts'][2] == r['terminals'] + pg
        r['hpwl_rel_diff'] = r['attar_hpwl'] / r['hpwl'] - 1 if r.get('hpwl') else None
        # negative control: 1% of the movable cells half a site to the right
        g = torch.Generator().manual_seed(0)
        mv = torch.nonzero(d.movable).flatten().cpu()
        pick = mv[torch.randperm(len(mv), generator=g)[:max(1, len(mv) // 100)]].to(d.device)
        d.pos[pick, 0] += float(d.row_site_w.min()) / 2
        r['broken_rejected_attar'] = not attar.Evaluator(d).legality()['legal']
        d.export(os.path.join(WORK, tag + '_broken.def'))
        b = check(d, os.path.join(WORK, tag + '_broken.def'), lefs, a.lef_dir, tag + '_b', a.cores)
        r['broken_rejected_openroad'] = not b['legal_but_one_site_gaps']
        rows.append(r)
        print('%-22s %8d cells  read %s counts %s  legal attar %s openroad %s (but one-site gaps: %s; %s)  HPWL diff %s  grt %s (WL %s, overflow %s)  broken rejected %s/%s  (place %.0fs, export %.0fs, openroad %ds)'
              % (name, r['cells'], r['read'], r['counts_match'], r['attar_legal'], r['legal'], r['legal_but_one_site_gaps'],
                 ', '.join('%s %d' % (k.replace('_failures', ''), v) for k, v in r['violations'].items() if v) or 'none',
                 '%+.5f%%' % (100 * r['hpwl_rel_diff']) if r['hpwl_rel_diff'] is not None else '-', r.get('grt', '-'),
                 r.get('grt_wl'), r.get('grt_overflow'), r['broken_rejected_attar'], r['broken_rejected_openroad'],
                 r['place_s'], r['export_s'], r['openroad_s']), flush=True)
        del d; torch.cuda.empty_cache()
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
