"""Hooks for research code: global placement's extra objective term and callback (attar.naive.global_place), and the
batched worst paths (TimingResult.critical_paths). On the CPU, where placement is deterministic."""
import unittest

import torch

from common import SMALL, attar
from attar import naive


def _cpu(name):
    try:
        return attar.load(name, 'cpu', download=False)
    except FileNotFoundError as e:
        raise unittest.SkipTest(str(e)) from None


class GlobalPlaceHooksTest(unittest.TestCase):
    ARGS = dict(max_iters=80, verbose=False, graphs=False, seed=0, snapshot='hook_test')

    def test_a_zero_term_changes_nothing(self):
        d = _cpu(SMALL)
        naive.global_place(d, **self.ARGS)
        a = d.pos.clone()
        naive.global_place(d, extra=lambda pos: (0.0, torch.zeros_like(pos)), **self.ARGS)
        self.assertTrue(torch.equal(a, d.pos))

    def test_a_pull_term_moves_its_cells(self):
        d = _cpu(SMALL)
        mv = torch.nonzero(d.movable).flatten()
        pick = mv[: len(mv) // 4]
        target = d.core_box[0][:2].float()                       # the core's lower-left corner

        def dist(pos):
            return float((pos[pick] - target).norm(dim=1).mean())
        naive.global_place(d, **self.ARGS)
        base = dist(d.pos)

        def pull(pos):
            g = torch.zeros_like(pos)
            delta = pos[pick] - target
            g[pick] = 0.4 * delta                                   # 0.2 x squared distance
            return float(0.2 * (delta ** 2).sum()), g
        naive.global_place(d, extra=pull, **self.ARGS)
        self.assertLess(dist(d.pos), 0.5 * base)

    def test_the_callback_sees_every_kth_iteration_and_can_stop(self):
        d = _cpu(SMALL)
        seen = []

        def cb(it, pos):
            seen.append(it)
            self.assertEqual(tuple(pos.shape), tuple(d.pos.shape))
            return it >= 30
        r = naive.global_place(d, callback=cb, callback_every=10, **self.ARGS)
        self.assertEqual(seen, [0, 10, 20, 30])
        self.assertEqual(r['iterations'], 31)


class CriticalPathsTest(unittest.TestCase):
    def setUp(self):
        self.d = _cpu('after_fp/iccad2025/des3')
        self.r = self.d.timing()

    def test_paths_are_connected_and_end_at_their_endpoints(self):
        r, d = self.r, self.d
        cp = r.critical_paths()
        self.assertEqual(len(cp), r.violations)
        self.assertTrue(bool((cp.slack[1:] >= cp.slack[:-1]).all()))                 # worst first
        p_net, p_cell = d.p_net.cpu(), d.p_cell.cpu()
        for k in range(len(cp)):
            a, b = int(cp.ptr[k]), int(cp.ptr[k + 1])
            self.assertEqual(int(cp.pin[b - 1]), int(cp.endpoint[k]))
            pins, wire = cp.pin[a:b], cp.wire[a:b]
            for i in range(1, b - a):
                if wire[i]:                                                          # a net: one net
                    self.assertEqual(int(p_net[pins[i - 1]]), int(p_net[pins[i]]))
                else:                                                                # an arc: one cell
                    self.assertEqual(int(p_cell[pins[i - 1]]), int(p_cell[pins[i]]))
            self.assertTrue(bool((cp.arrival[a + 1:b] >= cp.arrival[a:b - 1] - 1e-3).all()))

    def test_path_is_the_batched_path(self):
        cp = self.r.critical_paths(max_paths=3)
        for k in range(len(cp)):
            one = self.r.path(int(cp.endpoint[k]))
            self.assertEqual([(n, rf) for n, rf, _, _ in one], [(n, rf) for n, rf, _ in cp.report(k)])

    def test_pin_count_counts_paths(self):
        cp = self.r.critical_paths()
        cnt = cp.pin_count()
        k = cp.path_of()
        ref = torch.zeros_like(cnt)
        for p in set(cp.pin.tolist()):
            ref[p] = len(set(k[cp.pin == p].tolist()))
        self.assertTrue(torch.equal(cnt, ref))
        frm, to, wire, kk = cp.edges()
        self.assertEqual(len(frm), len(cp.rows) - len(cp))


if __name__ == '__main__':
    unittest.main()
