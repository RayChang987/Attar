#!/usr/bin/env python3
"""Naive GP + LG over zoo designs: legality (incl. rail parity), HPWL, displacement, time, against the kept reference.
usage: bench_naive.py [out.json] [zoo name ...]   (default: every design in the zoo)"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import attar  # noqa: E402
from attar import naive  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.json') else None
names = [a for a in sys.argv[1:] if not a.endswith('.json')] or attar.zoo()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
rows = []
print('%-26s %9s | %7s %6s | %6s %7s %7s | %-9s | %s' % ('design', 'cells', 'GP s', 'it', 'LG s', 'GP->LG', 'vs ref', 'legal', 'reference'))
for n in names:
    try:
        d = attar.load(n, dev)
    except Exception as e:                       # noqa: BLE001
        print('%-26s load failed: %s' % (n, e)); continue
    if 'global_place' not in d.capabilities() or 'rows' not in d.facts:
        print('%-26s skipped: can do %s' % (n, d.capabilities())); continue
    ref_name = next((s for s in ('reference', 'dreamplace_lg') if s in d.snapshots), None)
    ref = attar.load(n, dev).restore(ref_name).hpwl().item() if ref_name else None
    try:
        g = naive.global_place(d, verbose=False)
        ev = attar.Evaluator(d)
        h0 = ev.hpwl().item()
        st = naive.legalize(d, verbose=False)
        h1 = ev.hpwl().item()
        lg = ev.legality()
    except Exception as e:                       # noqa: BLE001
        print('%-26s FAILED: %s' % (n, str(e).split('\n')[0])); rows.append({'design': n, 'error': str(e)}); continue
    r = {'design': n, 'cells': d.manifest['counts']['movable'], 'gp_s': g['seconds'], 'gp_iters': g['iterations'],
         'gp_hpwl': h0, 'lg_s': st['seconds'], 'lg_hpwl': h1, 'mean_disp': st['mean_disp'], 'legality': lg,
         'reference': ref_name, 'ref_hpwl': ref}
    rows.append(r)
    bad = {k: v for k, v in lg.items() if k not in ('cells', 'legal') and v}
    print('%-26s %9d | %7.2f %6d | %6.2f %+6.1f%% %7s | %-9s | %s' % (
        n, r['cells'], g['seconds'], g['iterations'], st['seconds'], 100 * (h1 / h0 - 1),
        '%+.1f%%' % (100 * (h1 / ref - 1)) if ref else '-', 'legal' if lg['legal'] else 'ILLEGAL', ref_name or '-')
        + ('  %s' % bad if bad else ''), flush=True)
    del d, ev
    if dev == 'cuda':
        torch.cuda.empty_cache()
if out:
    with open(out, 'w') as f:
        json.dump(rows, f, indent=1)
