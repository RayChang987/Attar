#!/usr/bin/env python3
"""HPWL speed of the Evaluator modes. usage: bench_hpwl.py [--compile] <zoo name> ..."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import attar  # noqa: E402

dev = 'cuda'
use_compile = '--compile' in sys.argv


def timed(f, n=50):
    f(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        r = f()
    torch.cuda.synchronize()
    return r, 1e3 * (time.time() - t) / n


torch.zeros(1, device=dev)
print('%-22s %9s | %9s %9s %9s %9s %18s %9s' % ('design', 'pins', 'Design', 'Evaluator', 'fast', 'compile', 'batch per plc (B)', 'GB/s eff'))
for name in [a for a in sys.argv[1:] if not a.startswith('--')]:
    d = attar.load(name, dev)
    if d.snapshots:
        d.restore('reference' if 'reference' in d.snapshots else next(iter(d.snapshots)))
    P = len(d.p_cell)
    ref, t_d = timed(lambda: d.hpwl())
    ev = attar.Evaluator(d)
    r1, t_e = timed(lambda: ev.hpwl())
    fe = attar.Evaluator(d, fast=True)
    r2, t_f = timed(lambda: fe.hpwl())
    t_c = float('nan')
    if use_compile:
        ce = attar.Evaluator(d, compile=True)
        r3, t_c = timed(lambda: ce.hpwl())
        assert abs(r3.item() / ref.item() - 1) < 1e-5
    B = max(1, min(64, int(2e8 // (P * 48))))        # keep the batch within a few GB
    pb = d.pos[None].expand(B, -1, -1).contiguous()
    rb, t_b = timed(lambda: ev.hpwl_batch(pb), 5)
    for r in (r1, r2, rb[0]):
        assert abs(r.item() / ref.item() - 1) < 1e-5, (name, r.item(), ref.item())
    best = min(t_e, t_f, t_c if t_c == t_c else 1e9)
    gbs = P * 40 / (best * 1e-3) / 1e9               # ~40 bytes moved per pin (gather, offsets, index, scatter)
    print('%-22s %9d | %7.3fms %7.3fms %7.3fms %7.3fms %11.4fms (%3d) %9.0f' % (name, P, t_d, t_e, t_f, t_c, t_b / B, B, gbs), flush=True)
    del d, ev, fe, pb
    torch.cuda.empty_cache()
