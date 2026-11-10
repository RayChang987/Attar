#!/usr/bin/env python3
"""Summarize orfs_runs/results/<design>/roundtrip.summary.json: identity check, jitter noise floor, perturbation.
usage: rt_summary.py design ..."""
import json
import os
import sys

RT = os.path.join(os.path.dirname(__file__), '..', 'orfs_runs', 'results')
K = ['routed_wl', 'grt_wl', 'dp_hpwl_est', 'wns', 'tns']


def rel(a, b):
    return 100.0 * (a - b) / abs(b) if b else float('nan')


print('%-7s %-9s %8s | %s' % ('design', 'run', 'dHPWL%', ' | '.join('%14s' % ('d' + k + '%') for k in K)))
for d in sys.argv[1:]:
    rows = dict(json.load(open(os.path.join(RT, d, 'roundtrip.summary.json'))))
    b = rows['orfs (baseline)']
    same = all(rows['identity'].get(k) == b.get(k) for k in K)
    print('%-7s identity  bit-identical to baseline: %s' % (d, same))
    for name in sorted(r for r in rows if r.startswith(('jitter', 'perturb'))):
        m = rows[name]
        if not m.get('ok', True):
            print('%-7s %-9s FAILED' % (d, name)); continue
        print('%-7s %-9s %8.3f | %s' % (d, name, rel(m['hpwl_attar'], b['hpwl_attar']),
                                       ' | '.join('%14.3f' % rel(m[k], b[k]) for k in K)))
    js = [rows[n] for n in rows if n.startswith('jitter') and rows[n].get('ok', True)]
    if js:
        rng = lambda k: (min([b[k]] + [m[k] for m in js]), max([b[k]] + [m[k] for m in js]))
        print('%-7s noise range over baseline + %d jitters: %s' % (d, len(js), '; '.join(
            '%s %.3g%%' % (k, 100.0 * (rng(k)[1] - rng(k)[0]) / abs(b[k])) for k in K)))
