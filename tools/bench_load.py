#!/usr/bin/env python3
"""Load-speed benchmark: disk read, GPU load, names, HPWL, WA forward+backward, GPU memory.
usage: bench_load.py <zoo name> ..."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import attar  # noqa: E402
from attar import st  # noqa: E402
from attar.design import ZOO  # noqa: E402

dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def sync():
    if dev == 'cuda':
        torch.cuda.synchronize()


def timed(f, n=1):
    sync(); t = time.time()
    for _ in range(n):
        r = f()
    sync()
    return r, (time.time() - t) / n


torch.zeros(1, device=dev); sync()  # CUDA context outside the timings
print('%-22s %9s %8s %7s %7s %7s %7s %8s %9s %8s %8s' % ('design', 'cells', 'nets', 'macros', 'IR MB', 'read s', 'load s',
                                                        'names s', 'HPWL ms', 'WA f+b ms', 'GPU MB'))
for name in sys.argv[1:]:
    path = os.path.join(ZOO, name)
    size = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)) / 1e6
    _, t_read = timed(lambda: st.load(os.path.join(path, 'core.safetensors')))
    if dev == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    d, t_load = timed(lambda: attar.load(name, dev))
    _, t_names = timed(lambda: d.names)
    if d.snapshots:                                   # a kept placement: the reference, or e.g. DREAMPlace's
        d.restore('reference' if 'reference' in d.snapshots else next(iter(d.snapshots)))
    d.hpwl(); _, t_hp = timed(lambda: d.hpwl(), 20)
    p = d.pos.clone().requires_grad_(True)

    def wa():
        p.grad = None
        d.wa_wirelength(p, 0.5).backward()
    wa(); _, t_wa = timed(wa, 10)
    mem = torch.cuda.max_memory_allocated() / 1e6 if dev == 'cuda' else 0
    c = d.manifest['counts']
    print('%-22s %9d %8d %7d %7.1f %7.2f %7.2f %8.2f %9.2f %8.2f %8.0f' % (name, c['cells'], c['nets'], c['macros'], size,
          t_read, t_load, t_names, 1e3 * t_hp, 1e3 * t_wa, mem), flush=True)
    del d, p
    if dev == 'cuda':
        torch.cuda.empty_cache()
