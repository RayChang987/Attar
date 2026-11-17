"""The timer on the CPU: the delay calculation's loops stop early there (attar.timing.dmp.HOST_EARLY_EXIT) and every
net with a driver gets its exact host tree; both must leave the results unchanged (bitwise)."""
import unittest

import torch

from common import SMALL, attar
from attar.timing import dmp
from attar.timing.sta import Timer


def _cpu(name):
    try:
        d = attar.load(name, 'cpu', download=False)
    except FileNotFoundError as e:
        raise unittest.SkipTest(str(e)) from None
    if 'reference' in d.snapshots:
        d.load_reference()
    return d


class HostEarlyExitTest(unittest.TestCase):
    def test_same_arrivals_and_slews_as_the_fixed_loops(self):
        d = _cpu(SMALL)
        T = Timer(d)
        old = dmp.HOST_EARLY_EXIT
        try:
            out = {}
            for flag in (False, True):
                dmp.HOST_EARLY_EXIT = flag
                r = T.run()
                out[flag] = ([t.clone() for t in (T.amax, T.amin, T.smax, T.smin)], r.endpoint_slacks())
        finally:
            dmp.HOST_EARLY_EXIT = old
        for a, b in zip(out[False][0], out[True][0]):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(out[False][1], out[True][1])

    def test_iterate_stops_on_frozen_elements_only(self):
        # a step that halves the live elements' x until they fall below 1: the early exit must equal the fixed loop
        def step(x, done, k):
            x = torch.where(done, x, x / k)
            return x, done | (x < 1)
        x0 = torch.tensor([1000.0, 3.0, 0.5, 64.0], dtype=torch.float64)
        done0 = x0 < 1
        fixed = x0, done0
        for _ in range(20):
            fixed = step(*fixed, 2.0)
        early = dmp._iterate(step, (x0, done0), (2.0,), 20, lambda st: ~st[1])
        self.assertTrue(torch.equal(fixed[0], early[0]) and torch.equal(fixed[1], early[1]))
