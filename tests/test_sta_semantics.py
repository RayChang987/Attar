"""OpenSTA semantics the timer replays on small synthetic inputs: where levelization breaks a combinational loop
(Levelize::findBackEdges) and the latch setup cycle accounting (CycleAccting, TimingRole::latchSetup)."""
import unittest

import numpy as np
import torch

from common import attar  # (puts the checkout on the path)
from attar.timing.ir import T_COMB
from attar.timing.sta import Timer, _latch_relationships


class LoopEdgesTest(unittest.TestCase):
    def test_the_back_edge_of_the_search_from_the_root(self):
        # a NAND (A, B -> Y) and two inverters in a ring, B fed by an input port: the search goes port -> B -> Y ->
        # inv -> inv -> A, and A -> Y closes the loop (Y is on the path): that arc is disabled, nothing else
        # pins: 0 nand/A, 1 nand/B, 2 nand/Y, 3 inv1/A, 4 inv1/Y, 5 inv2/A, 6 inv2/Y, 7 the port
        p_cell = np.array([0, 0, 0, 1, 1, 2, 2, 3])
        p_mpin = np.array([0, 1, 2, 3, 4, 5, 6, -1])
        n_fr, n_to = np.array([7, 2, 4, 6]), np.array([1, 3, 5, 0])
        fr, to = np.array([0, 1, 3, 5]), np.array([2, 2, 4, 6])
        atype = np.full(4, T_COMB)
        arc, net = Timer._loop_edges(8, n_fr, n_to, fr, to, atype, np.zeros(4, bool), p_cell, p_mpin)
        self.assertEqual(arc.tolist(), [True, False, False, False])
        self.assertFalse(net.any())

    def test_no_loop_no_work(self):
        p_cell = np.array([0, 0, 1]); p_mpin = np.array([0, 1, -1])
        arc, net = Timer._loop_edges(3, np.array([2]), np.array([0]), np.array([0]), np.array([1]),
                                     np.array([T_COMB]), np.zeros(1, bool), p_cell, p_mpin)
        self.assertFalse(arc.any() or net.any())


class LatchAccountingTest(unittest.TestCase):
    def test_one_clock(self):
        # period 10, high from 0 to 5; tags: 0 the rising edge, 1 the falling one
        req, sh = _latch_relationships([{'period': 10.0, 'waveform': [0.0, 5.0]}])
        # launched at 0 into a latch open while high: it opens at 0 (closes at 5), same cycle
        self.assertEqual((req[0, 0], sh[0, 0]), (0.0, 0.0))
        # launched at 5: the next closing is at 15, the opening before it at 10 (the enable's cycle 1)
        self.assertEqual((req[1, 0], sh[1, 0]), (10.0, -10.0))
        # launched at 0 into a latch open while low (opens at 5, closes at 10)
        self.assertEqual((req[0, 1], sh[0, 1]), (5.0, 0.0))



class IdealPinLatencyTest(unittest.TestCase):
    """set_clock_latency on register clock pins of an ideal clock (pre-CTS timing of an SDC with useful-skew pin
    latencies, e.g. the ICCAD 2025 contest SDCs without set_propagated_clock). OpenSTA: the ideal clock network delay
    at such a pin is the pin's latency instead of the clock's latency l0, at launch and at capture. With latency v on
    every register clock pin, register-to-register slacks do not move and an output port's moves by l0 - v (launch
    earlier, the port's capture keeps l0); with v on one register only, its data pin's slack moves by v - l0.
    The latencies are given in memory (Design.set_timing)."""

    def test_ideal_pin_latency(self):
        import warnings
        from common import SMALL, need
        d = need(SMALL)
        d.restore('reference')
        r0 = d.timing()
        T = r0.timer
        self.assertFalse(bool(T.prop_clk.any()))            # gcd's SDC: ideal clocks
        regs = sorted(set(T.k_fr.cpu().numpy().tolist()))
        cks = {int(T.clk_of[p]) for p in regs}
        self.assertEqual(len(cks), 1)
        ck = T.clock_names[cks.pop()]
        c = T.info['clocks'][ck]
        l0, v = float(c.get('latency', 0.0)), 10.0
        self.assertGreater(abs(l0 - v), 1.0)
        ep = r0.endpoints
        s0 = r0.setup[ep].cpu().numpy()
        kto, kfr, par = T.k_to.cpu().numpy(), T.k_fr.cpu().numpy(), np.asarray(T.par)
        reg_ep = np.isin(ep.cpu().numpy(), kto)            # (T is released once the settings change)

        def with_pins(pins, val):
            d.set_timing(pin_latency=[{'pin': p, 'clock': ck, 'value': val, 'rf': [0, 1], 'mm': ['max', 'min']}
                                      for p in pins])
            with warnings.catch_warnings():
                warnings.filterwarnings('error', message='.*pin clock latencies.*')     # no 'not modeled'
                r = d.timing()
            self.assertTrue(r.summary()['exact'])
            return r, r.setup[ep].cpu().numpy() - s0

        np.testing.assert_allclose(with_pins(regs, l0)[1], 0, atol=1e-4)     # the clock's own latency: no change
        dl = with_pins(regs, v)[1]
        np.testing.assert_allclose(dl[reg_ep], 0, atol=1e-4)
        moved = np.abs(dl[~reg_ep]) > 1e-4
        self.assertTrue(moved.any())
        np.testing.assert_allclose(dl[~reg_ep][moved], l0 - v, atol=1e-4)
        reg = regs[0]
        r1, d1 = with_pins([reg], v)
        own = np.isin(ep.cpu().numpy(), kto[kfr == reg])
        np.testing.assert_allclose(d1[own], v - l0, atol=1e-4)
        other = d1[~own][np.abs(d1[~own]) > 1e-4]
        np.testing.assert_allclose(np.abs(other), abs(l0 - v), atol=1e-4)    # paths launched from reg only
        # the register's clock arrival is the edge plus its own latency, the others' the clock's
        ca = r1.clock_arrivals()
        name = d.pin_names([reg])[0]
        self.assertAlmostEqual(ca[name][0], c['waveform'][int(par[reg])] + v, places=3)
        other_reg = d.pin_names([regs[1]])[0]
        self.assertAlmostEqual(ca[other_reg][0], c['waveform'][int(par[regs[1]])] + l0, places=3)
        d.reset_timing()
        self.assertTrue(np.allclose(d.timing().setup[ep].cpu().numpy(), s0))



class EarlyLibraryTest(unittest.TestCase):
    """An Early library for the min analysis (OpenSTA's read_liberty -min; ICCAD 2015's): its tables after the Late
    ones, read by the min lanes at Liberty arc + lmin and by hold checks. Here the Early arrays are given in memory
    (attar.timing.ir.arrays_min patched): a copy of the Late ones changes nothing, halved delays move hold only."""

    def _time(self, early):
        from unittest import mock
        from attar.timing import ir
        from common import SMALL, need
        d = need(SMALL)
        d.restore('reference')
        with mock.patch.object(ir, 'arrays_min', lambda d_: early(ir.arrays(d_)[0])):
            r = d.timing()
            return r, r.timer.lmin

    def test_early_library(self):
        import copy
        r0, l0 = self._time(lambda t: None)
        self.assertEqual(l0, 0)
        same, l1 = self._time(lambda t: copy.deepcopy(t))
        self.assertEqual(l1, len(r0.library().master))
        for a, b in ((r0.setup, same.setup), (r0.hold, same.hold), (r0.timer.amin, same.timer.amin)):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(same.library().min_offset, l1)

        def halved(t):
            e = copy.deepcopy(t)
            e['a_val'] = e['a_val'] * 0.5
            return e
        fast, _ = self._time(halved)
        self.assertTrue(torch.equal(r0.setup, fast.setup))                  # the max analysis: the Late library
        ok = torch.isfinite(r0.hold) & torch.isfinite(fast.hold)
        self.assertTrue(bool((fast.hold[ok] != r0.hold[ok]).any()))
        a0, a1 = r0.timer.amin, fast.timer.amin
        m = torch.isfinite(a0) & torch.isfinite(a1) & (a0 > 0)
        self.assertTrue(bool((a1[m] <= a0[m] + 1e-3).all()) and bool((a1[m] < a0[m]).any()))   # earlier
        self.assertTrue(fast.summary()['exact'])

    def test_a_library_that_does_not_line_up_is_not_used(self):
        import copy

        def other(t):
            e = copy.deepcopy(t)
            e['a_type'] = e['a_type'].copy(); e['a_type'][0] += 1
            return e
        r, lmin = self._time(other)
        self.assertEqual(lmin, 0)
        self.assertIn('Early library does not line up', ' '.join(r.limitations))


if __name__ == '__main__':
    unittest.main()


class IncrementalWiresTest(unittest.TestCase):
    """Timer.wires redoes only the nets with a pin that moved or changed: the timing equals a new timer's, bit for bit,
    on the CPU (no CUDA graph, host trees) and on the GPU (wires outside the graph, static buffers)."""

    def check(self, dev):
        from common import need, MEDIUM
        need(MEDIUM)
        d = attar.load(MEDIUM, device=dev, download=False)
        d.timing()
        T = d._timer
        same = lambda a, b: torch.equal(a.setup, b.setup) and torch.equal(a.hold, b.hold)  # noqa: E731
        mv = torch.nonzero(d.movable).flatten()
        sel = mv[torch.randperm(len(mv), generator=torch.Generator().manual_seed(0))[:len(mv) // 100].to(mv.device)]
        p = d.pos.clone(); p[sel] += 1.0; d.pos = p
        r = d.timing()
        self.assertIs(d._timer, T)
        self.assertTrue(same(r, Timer(d).run()))
        r2 = d.timing()                                             # nothing changed: the same result
        self.assertTrue(same(r2, r))
        cells = d.names['cells']
        i = next(i for i in range(len(cells)) if bool(d.movable[i]) and len(d.equivalents(i)) > 1)
        cur = d.names['masters'][int(d.raw['c_master'][i])]
        d.resize(i, next(m for m in d.equivalents(i) if m != cur))   # capacitances and pin locations change
        self.assertTrue(same(d.timing(), Timer(d).run()))
        T.INCREMENTAL_WIRES = 0.0                                   # past the threshold: every net again
        p = d.pos.clone(); p[sel] -= 1.0; d.pos = p
        self.assertTrue(same(d.timing(), Timer(d).run()))

    def test_cpu(self):
        self.check('cpu')

    def test_gpu(self):
        if not torch.cuda.is_available():
            self.skipTest('no GPU')
        self.check('cuda')
