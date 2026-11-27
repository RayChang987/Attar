#!/usr/bin/env python3
"""Validate Evaluator.pin_access on routed Innovus runs of i3dPDK_NS3K (the pa_pdn archive).

Per run: distill the final routed DEF (written right after routeDesign, so its placement is the one the router got,
with the PDN), compute Evaluator.pin_access per pin, and read rpt/drc.rpt (verify_drc). Every M1/M2 marker goes to the
nearest pin of its regular net within 0.5 um (LEF x4 units); a pin is "failing" when a signal-to-PG marker lands on
it. Reports, per design and PDN style, the blocked pins against the failing ones (precision, recall) and the Kendall
tau over each design's runs between the blocked-pin count and the signal-to-PG marker count; OpenROAD dpl's PDN rule
(legality()['pdn_blocked']) alongside.

usage: validate_pin_access.py SCRATCH [--ns3k ~/i3dPDK_NS3K] [--jobs 8] [--report docs/validation/pin_access.md]
       [design ...]
SCRATCH keeps one JSON per run (a rerun skips the runs already there) and the IRs while they are measured."""
import argparse
import collections
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ATTR_R = 0.5          # um: a marker belongs to a pin of its net within this distance
ENT = re.compile(r'^(\w+): \( ([^)]*) \) (.*?)\s*\( (\w+) \)\s*$')
BND = re.compile(r'Bounds : \( ([-\d.]+), ([-\d.]+) \) \( ([-\d.]+), ([-\d.]+) \)')


def parse_drc(path):
    """Innovus verify_drc entries: [{kind, layer, nets (regular), special (a PG shape involved), box (um)}]."""
    out, cur = [], []
    for line in open(path, errors='replace'):
        if line.startswith('#'):
            continue
        if line.startswith('Bounds'):
            txt = ' '.join(cur); cur = []
            m, b = ENT.match(txt), BND.match(line.strip())
            if m and b:
                kind, _, objs, layer = m.groups()
                out.append({'kind': kind, 'layer': layer, 'nets': re.findall(r'Regular Wire of Net (\S+)', objs),
                            'special': 'Special' in objs, 'box': [float(v) for v in b.groups()]})
        elif line.strip():
            cur.append(line.strip())
    return out


def runs(ns3k, designs):
    """Run folders with a final routed DEF and a DRC report: pnr/out/<design>/NS3K_5T/<tag>."""
    out = []
    for r in sorted(glob.glob(os.path.join(ns3k, 'innovus_enablement/pnr/out/*/NS3K_5T/*'))):
        design = r.split(os.sep)[-3]
        if designs and design not in designs or not os.path.exists(os.path.join(r, 'run_params.json')):
            continue
        top = json.load(open(os.path.join(r, 'run_params.json')))['top']
        if os.path.exists(os.path.join(r, top + '.def')) and os.path.exists(os.path.join(r, 'rpt', 'drc.rpt')):
            out.append(r)
    return out


def lefs_of(run, ns3k, top):
    lf = os.path.join(run, 'xplace', top + '.in.lefs')
    if os.path.exists(lf):              # the .lefs file lists them relative to the folder that holds the kit
        return [os.path.join(os.path.dirname(ns3k), l.strip()) for l in open(lf) if l.strip()]
    L = os.path.join(ns3k, 'innovus_enablement', 'lef_x4')
    return [os.path.join(L, 'i3dPDK_NS3K_5T_tech_unidir_dir.lef'), os.path.join(L, 'NS3K_5T_cells.lef')] + \
        sorted(glob.glob(os.path.join(L, 'fakeram*.lef')))


def measure(run, out_json, scratch, ns3k):
    """One run -> out_json (see the module doc)."""
    sys.path.insert(0, ROOT)
    import numpy as np
    import attar
    from attar.io.distill import TERMINAL
    rp = json.load(open(os.path.join(run, 'run_params.json')))
    ir = os.path.join(scratch, os.path.basename(out_json) + '.ir')
    try:
        t0 = time.time()
        attar.distill(os.path.join(run, rp['top'] + '.def'), lefs_of(run, ns3k, rp['top']), ir)
        d = attar.load(ir, device='cpu')
        ev = attar.Evaluator(d)
        t1 = time.time()
        r = ev.pin_access(per_pin=True)
        t_pa = time.time() - t1
        bp = r.pop('pins').numpy()
        std = ((d.flags & TERMINAL) == 0).numpy()
        pp, pc = d.pin_pos().numpy(), d.p_cell.numpy()
        n_ptr = d.raw['n_ptr'].astype(np.int64)
        nid = {n.replace('\\', ''): i for i, n in enumerate(d.names['nets'])}
        fail_pg = np.zeros(len(pc), bool)
        mk = collections.Counter()
        for e in parse_drc(os.path.join(run, 'rpt', 'drc.rpt')):
            if e['layer'] not in ('M1', 'M2'):
                mk['other_layers'] += 1
                continue
            k = 'pg' if e['special'] else 'ss'
            mk[k] += 1
            cx, cy = (e['box'][0] + e['box'][2]) / 2, (e['box'][1] + e['box'][3]) / 2
            best = None
            for n in e['nets']:
                i = nid.get(n)
                if i is None:
                    continue
                ps = np.arange(n_ptr[i], n_ptr[i + 1]); ps = ps[std[pc[ps]]]
                if len(ps):
                    dd = np.hypot(pp[ps, 0] - cx, pp[ps, 1] - cy); j = int(dd.argmin())
                    if dd[j] <= ATTR_R and (best is None or dd[j] < best[0]):
                        best = (dd[j], ps[j])
            if best is not None:
                mk[k + '_attributed'] += 1
                if k == 'pg':
                    fail_pg[best[1]] = True
        res = {'run': os.path.basename(run), 'design': rp['design'], 'pg_style': rp.get('pg_style'),
               'metric': r, 'markers': dict(mk), 'tp': int((bp & fail_pg).sum()), 'fp': int((bp & ~fail_pg).sum()),
               'fn': int((~bp & fail_pg).sum()), 'dpl_pdn_blocked': int(ev.legality()['pdn_blocked']),
               'cells': int(std.sum()), 'seconds_pin_access': t_pa, 'seconds': time.time() - t0}
        json.dump(res, open(out_json, 'w'))
    finally:
        shutil.rmtree(ir, ignore_errors=True)


def report(scratch):
    """Markdown tables from the run JSONs in scratch."""
    import numpy as np
    from scipy.stats import kendalltau
    J = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(scratch, '*.json')))]
    by = collections.defaultdict(list)
    for r in J:
        by[(r['design'], r['pg_style'])].append(r)
    S = lambda rs, k: sum(r[k] for r in rs)  # noqa: E731
    M = lambda rs, k: sum(r['markers'].get(k, 0) for r in rs)  # noqa: E731
    lines = ['| design | PDN | runs | signal-to-PG M1/M2 markers | pins with one | blocked pins | TP | FP | FN | '
             'Kendall tau (blocked vs markers) | dpl rule: cells |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for (dn, pg), rs in sorted(by.items()):
        a = [r['metric']['blocked_pins'] for r in rs]; b = [r['markers'].get('pg', 0) for r in rs]
        tau = kendalltau(a, b)[0] if len(rs) > 2 and len(set(a)) > 1 else float('nan')
        lines.append('| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            dn, pg, len(rs), '{:,}'.format(M(rs, 'pg')), '{:,}'.format(S(rs, 'tp') + S(rs, 'fn')),
            '{:,}'.format(sum(a)), '{:,}'.format(S(rs, 'tp')), S(rs, 'fp'), S(rs, 'fn'),
            '—' if np.isnan(tau) else '%.2f' % tau, S(rs, 'dpl_pdn_blocked')))
    tp, fp, fn = S(J, 'tp'), S(J, 'fp'), S(J, 'fn')
    head = ('%d runs, %s signal pins. Blocked pins against the pins with a signal-to-PG M1/M2 marker: TP %s, FP %d, '
            'FN %d (precision %.4f, recall %.4f). OpenROAD dpl\'s PDN rule flags %d cells in all. M1/M2 markers: %s '
            'signal-to-PG, %s signal-to-signal (not modelled). pin_access: median %.2f s per run (CPU, 2 threads).'
            % (len(J), '{:,}'.format(sum(r['metric']['signal_pins'] for r in J)), '{:,}'.format(tp), fp, fn,
               tp / max(tp + fp, 1), tp / max(tp + fn, 1), S(J, 'dpl_pdn_blocked'), '{:,}'.format(M(J, 'pg')),
               '{:,}'.format(M(J, 'ss')), float(np.median([r['seconds_pin_access'] for r in J]))))
    return head, '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('scratch')
    ap.add_argument('designs', nargs='*')
    ap.add_argument('--ns3k', default='~/i3dPDK_NS3K')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--report', help='write the tables into this markdown file (between the RESULTS markers)')
    ap.add_argument('--one', nargs=2, metavar=('RUN', 'OUT'), help=argparse.SUPPRESS)
    a = ap.parse_args()
    ns3k = os.path.abspath(os.path.expanduser(a.ns3k))
    os.makedirs(a.scratch, exist_ok=True)
    if a.one:
        import torch
        torch.set_num_threads(2)
        return measure(a.one[0], a.one[1], a.scratch, ns3k)
    todo = []
    for r in runs(ns3k, a.designs):
        out = os.path.join(a.scratch, '%s__%s.json' % (r.split(os.sep)[-3], os.path.basename(r)))
        if not os.path.exists(out):
            todo.append((r, out))
    print('%d runs to measure' % len(todo), flush=True)
    procs, failed = [], []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    for job in todo + [None]:                         # None: wait for the last ones
        r, out = job if job else (None, None)
        while len(procs) >= a.jobs or (r is None and procs):
            for p in procs[:]:
                if p[0].poll() is not None:
                    procs.remove(p)
                    if p[0].returncode:
                        failed.append(p[1])
            time.sleep(0.2)
        if r is not None:
            log = open(out + '.log', 'w')
            procs.append((subprocess.Popen([sys.executable, '-W', 'ignore', __file__, a.scratch, '--ns3k', ns3k,
                                            '--one', r, out], stdout=log, stderr=log, env=env), r))
    if failed:
        print('%d runs failed (logs next to their JSON): %s' % (len(failed), failed[:5]))
    head, table = report(a.scratch)
    print(head); print(table)
    if a.report:
        txt = open(a.report).read()
        i, j = txt.index('<!-- RESULTS -->'), txt.index('<!-- /RESULTS -->')
        open(a.report, 'w').write(txt[:i] + '<!-- RESULTS -->\n' + head + '\n\n' + table + '\n' + txt[j:])


if __name__ == '__main__':
    main()
