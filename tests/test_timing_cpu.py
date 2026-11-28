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


class ResizedInPlaceTest(unittest.TestCase):
    def test_in_place_update_equals_a_new_timer(self):
        import numpy as np
        d = _cpu('after_fp/orfs_asap7/aes')
        d.timing()
        p0 = d.power()
        have = set(d.names['masters'])
        cells, masters = [], []
        for c in np.random.default_rng(3).permutation(len(d.names['cells'])).tolist():
            if not d.movable[c]:
                continue
            cur = d.names['masters'][int(d.raw['c_master'][c])]
            eq = [m for m in d.equivalents(c, dont_use=True) if m in have and m != cur]
            if eq:
                cells.append(c); masters.append(eq[0])
            if len(cells) >= 200:
                break
        T0 = d._timer
        d.resize(cells, masters)
        self.assertIs(d._timer, T0)                       # updated, not released
        r1 = d.timing()
        F = Timer(d); r2 = F.run()
        for name in ('amax', 'amin', 'smax', 'smin', 'pin_cap', 'pin_cap_min', 'sink_thr', 'pin_ctp', 'pin_rvia'):
            self.assertTrue(torch.equal(getattr(T0, name), getattr(F, name)), name)
        self.assertEqual(r1.endpoint_slacks(), r2.endpoint_slacks())
        # power: the new masters' functions and tables (OpenSTA: replaceEquivCellAfter -> Power::powerInvalid)
        p1 = d.power()
        d._timer = F; F._power = None
        p2 = d.power()
        self.assertEqual(p1.summary(), p2.summary())
        self.assertNotEqual(p0.summary(), p1.summary())

    def test_a_new_master_releases_the_timer(self):
        d = _cpu('after_fp/orfs_asap7/aes')
        d.timing()
        have = set(d.names['masters'])
        for c in range(len(d.names['cells'])):
            eq = [m for m in d.equivalents(c, dont_use=True) if m not in have]
            if eq:
                d.resize([c], [eq[0]])
                self.assertIsNone(d._timer)
                return
        self.skipTest('no master outside the design')


class HostDelayCalcTest(unittest.TestCase):
    def test_cpp_equals_the_pytorch_reference(self):
        from attar.timing import dmp_host
        if not dmp_host.available():
            self.skipTest('no C++ delay calculation: %s' % dmp_host.error())
        d = _cpu('after_fp/orfs_asap7/aes')
        T = Timer(d)
        self.assertIsNotNone(T.host)
        host, T.host = T.host, None
        r0 = T.run()
        ref = [t.clone() for t in (T.amax, T.amin, T.smax, T.smin)]
        T.host = host
        r1 = T.run()
        for a, b in zip(ref, (T.amax, T.amin, T.smax, T.smin)):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(r0.endpoint_slacks(), r1.endpoint_slacks())


class HostActivityTest(unittest.TestCase):
    def test_cpp_activities_equal_the_pytorch_reference(self):
        """Power's activity propagation on the CPU in C++ (attar.timing.activity_host) against the PyTorch path (its
        dense sweep, and its sweep over the enqueued pins: bitwise the same); float32 sums in another order."""
        from attar.timing import activity_host
        if not activity_host.available():
            self.skipTest('no C++ activities: %s' % activity_host.error())
        d = _cpu('after_fp/orfs_asap7/aes')
        d.power()
        P = d.timing().timer._power
        self.assertIsNotNone(P.X.host)
        res = {}
        for name, host, frontier in (('dense', None, False), ('frontier', None, True), ('c++', activity_host, True)):
            P.X.host, P.frontier = host, frontier
            if hasattr(P, '_G'):
                del P._G
            r = d.power()
            res[name] = (r, P.n_passes)
        a, b, c = (res[k][0] for k in ('dense', 'frontier', 'c++'))
        for x, y in zip(a.activity, b.activity):
            self.assertTrue(torch.equal(x, y))
        self.assertEqual(res['dense'][1], res['c++'][1])
        for x, y in zip(a.total, c.total):
            self.assertLess(abs(float(x) - float(y)), 1e-6 * abs(float(x)))


class SameAsTheGpuTest(unittest.TestCase):
    """The CPU timer (C++ delay calculation, csrc/dmp_core.h) and the GPU's fused kernels give the same bits."""

    def test_arrivals_slews_and_slacks(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('no GPU')
        out = {}
        for dev in ('cpu', 'cuda'):
            try:
                d = attar.load(SMALL, dev, download=False)
            except FileNotFoundError as e:
                raise unittest.SkipTest(str(e)) from None
            if 'reference' in d.snapshots:
                d.load_reference()
            T = Timer(d)
            r = T.run()
            out[dev] = [t.cpu() for t in (T.amax, T.amin, T.smax, T.smin, T._wb[0], T._wb[1], r.setup, r.hold)]
        for a, b in zip(out['cpu'], out['cuda']):
            self.assertTrue(bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all()))
