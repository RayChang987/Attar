"""OpenSTA semantics the timer replays on small synthetic inputs: where levelization breaks a combinational loop
(Levelize::findBackEdges) and the latch setup cycle accounting (CycleAccting, TimingRole::latchSetup)."""
import unittest

import numpy as np

from common import attar  # noqa: F401  (puts the checkout on the path)
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


if __name__ == '__main__':
    unittest.main()
