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

    def test_the_callback_can_take_the_iterations_metrics(self):
        d = _cpu(SMALL)
        got = {}

        def cb(it, pos, info):
            got[it] = info
        r = naive.global_place(d, callback=cb, callback_every=10, **self.ARGS)
        h = r['history']
        self.assertEqual(sorted(got), list(range(0, r['iterations'], 10)))
        for it, info in got.items():
            self.assertEqual(set(info), {'hpwl', 'overflow', 'lambda', 'gamma'})
            self.assertEqual(info['overflow'], h['overflow'][it])
            self.assertEqual(info['hpwl'], h['hpwl'][it])


class NetWeightsTest(unittest.TestCase):
    """Net weights changed during global placement reach the placer (the documented hook for net-weighted, e.g.
    timing-driven, placement), and a placement whose weights do not change stays exactly what it was."""
    ARGS = dict(max_iters=600, verbose=False, graphs=False, seed=0, snapshot='weights_test')   # to convergence: GP
    # returns its best-overflow iterate, which on a short run can predate the weight change

    def setUp(self):
        self.d = _cpu(SMALL)
        naive.global_place(self.d, **self.ARGS)
        ev = attar.Evaluator(self.d)
        hp = ev.hpwl_per_net()
        deg = self.d.n_ptr[1:] - self.d.n_ptr[:-1]
        cand = torch.nonzero(deg >= 2).flatten()
        self.heavy = cand[torch.argsort(hp[cand], descending=True)[:10]]    # the longest nets of a plain placement

    def _place(self, change=None, at=30, **kw):
        d = self.d
        d.n_weight.fill_(1.0)

        def cb(it, pos):
            if change is not None and it == at:
                change(d)
        naive.global_place(d, callback=cb, **kw, **self.ARGS)
        p = d.pos.clone()
        d.n_weight.fill_(1.0)
        return p

    def _heavy_hpwl(self, pos):
        return float(self.d.hpwl(pos, per_net=True, weighted=False)[self.heavy].sum())

    def test_unchanged_weights_change_nothing(self):
        base = self._place()
        self.assertTrue(torch.equal(base, self._place(change=lambda d: None)))
        self.assertTrue(torch.equal(base, self._place(precond='pins')))

    def test_weights_set_in_place_pull_their_nets_in(self):
        base = self._place()

        def up(d):
            d.n_weight[self.heavy] = 20.0
        moved = self._place(change=up)
        self.assertFalse(torch.equal(base, moved))
        self.assertLess(self._heavy_hpwl(moved), 0.8 * self._heavy_hpwl(base))

    def test_a_new_weight_tensor_is_seen_too(self):
        def up_inplace(d):
            d.n_weight[self.heavy] = 20.0

        def up_new(d):
            w = d.n_weight.clone(); w[self.heavy] = 20.0
            d.n_weight = w
        self.assertTrue(torch.equal(self._place(change=up_inplace), self._place(change=up_new)))

    def test_the_preconditioner_follows_the_weights(self):
        def up(d):
            d.n_weight[self.heavy] = 20.0
        self.assertFalse(torch.equal(self._place(change=up), self._place(change=up, precond='pins')))

    def test_the_evaluator_follows_the_weights(self):
        d = self.d
        d.n_weight.fill_(1.0)
        ev = attar.Evaluator(d)
        h1 = float(ev.hpwl())
        d.n_weight.mul_(2.0)
        self.assertAlmostEqual(float(ev.hpwl()), 2 * h1, delta=1e-4 * h1)
        d.n_weight = torch.ones_like(d.n_weight)
        self.assertAlmostEqual(float(ev.hpwl()), h1, delta=1e-4 * h1)
        d.n_weight[self.heavy] = 3.0
        self.assertAlmostEqual(float(ev.hpwl()), float(d.hpwl()), delta=1e-4 * h1)
        self.assertAlmostEqual(float(d.hpwl(weighted=False)), h1, delta=1e-4 * h1)
        d.n_weight.fill_(1.0)

    def test_bad_precond_is_refused(self):
        with self.assertRaises(ValueError):
            naive.global_place(self.d, precond='area', **self.ARGS)


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


class TimingDataTest(unittest.TestCase):
    """The timer's graph, library, arrivals and parasitics, read only (TimingResult.graph / library / arrivals /
    parasitics / trees): consistent with each other and with what the timer computed."""

    @classmethod
    def setUpClass(cls):
        cls.d = _cpu('after_fp/iccad2025/des3')
        cls.r = cls.d.timing()

    def test_graph_arcs_join_their_pins(self):
        g, d = self.r.graph(), self.d
        p_net, p_cell = d.p_net, d.p_cell
        self.assertTrue(torch.equal(p_net[g.n_from], g.n_net) and torch.equal(p_net[g.n_to], g.n_net))
        self.assertTrue(torch.equal(g.net_driver[g.n_net], g.n_from))
        for f, t in ((g.c_from, g.c_to), (g.l_from, g.l_to), (g.k_from, g.k_to)):
            self.assertTrue(torch.equal(p_cell[f], p_cell[t]))
        # the search order: a searched arc's head is on a later level than its tail
        self.assertTrue(bool((g.level[g.n_to] > g.level[g.n_from]).all()))
        s = g.c_searched
        self.assertTrue(bool((g.level[g.c_to[s]] > g.level[g.c_from[s]]).all()))
        self.assertTrue(torch.equal(g.endpoints, self.r.endpoints))
        self.assertEqual(len(g.clock_names), len(self.d.timing().timer.clock_names))

    def test_library_lookup_hits_its_breakpoints(self):
        lib, g = self.r.library(), self.r.graph()
        a = torch.unique(g.c_lib)[:50]
        for slot in range(4):
            ok = (lib.n[a, slot] > 0).all(1)
            aa = a[ok]
            if not len(aa):
                continue
            i = (lib.n[aa, slot, 0] - 1) // 2; j = (lib.n[aa, slot, 1] - 1) // 2
            x = lib.ix1[aa, slot].gather(1, i[:, None].long())[:, 0]
            y = lib.ix2[aa, slot].gather(1, j[:, None].long())[:, 0]
            want = lib.val[aa, slot][torch.arange(len(aa)), i.long(), j.long()]
            self.assertTrue(torch.allclose(lib.lookup(aa, slot, x, y), want))
        self.assertLess(int(lib.master.max()), len(self.d.names['masters']))
        c_master = torch.as_tensor(self.d.raw['c_master'].astype('int64'))
        self.assertTrue(torch.equal(lib.master[g.c_lib], c_master[self.d.p_cell[g.c_from]]))     # the arc's instance

    def test_arrivals_bound_the_critical_paths(self):
        a = self.r.arrivals()
        cp = self.r.critical_paths()
        if not len(cp):
            self.skipTest('no violations')
        rf = cp.rows % 2
        self.assertTrue(bool((cp.arrival <= a.max[cp.pin, rf] + 1e-3).all()))
        last = cp.ptr[1:] - 1                                   # the endpoint rows: the worst arrival there
        self.assertTrue(bool((a.max[cp.pin[last]].amax(1) >= cp.arrival[last] - 1e-3).all()))

    def test_loads_are_wire_plus_pins(self):
        a, w, g, lib = self.r.arrivals(), self.r.parasitics(), self.r.graph(), self.r.library()
        net_pins = torch.zeros(len(w.wire_cap), 2, dtype=a.load.dtype).index_add_(0, g.n_net, lib.pin_cap[g.n_to, :2])
        drv = g.net_driver >= 0
        nets = torch.nonzero(drv).flatten()
        want = w.wire_cap[nets, None] + net_pins[nets]
        self.assertTrue(torch.allclose(a.load[g.net_driver[nets]], want, rtol=1e-5, atol=1e-4))
        # the pi model holds the whole net: near + far = wire + pins (max analysis, rise), where a tree was reduced
        tot = w.pi[nets, 0, 0] + w.pi[nets, 0, 2]
        has = w.wire_cap[nets] > 0
        self.assertTrue(torch.allclose(tot[has], want[has, 0], rtol=1e-4, atol=1e-4))

    def test_trees_are_the_wires_the_timer_used(self):
        w, T = self.r.parasitics(), self.r.timer
        e, net = self.r.trees()
        length = torch.zeros(len(w.wire_cap), dtype=torch.float64).index_add_(
            0, net, (e[:, 2:] - e[:, :2]).abs().sum(1).double())
        nets = torch.unique(net)
        sig = ~torch.as_tensor(T.net_clkrc)[nets] & (w.wire_cap[nets] > 0)
        nets = nets[sig]
        self.assertGreater(len(nets), 100)
        self.assertTrue(torch.allclose(length[nets].float() * T.c_sig, w.wire_cap[nets], rtol=1e-4, atol=1e-5))
        # each tree starts at its driver
        first = torch.ones(len(net), dtype=torch.bool); first[1:] = net[1:] != net[:-1]
        drv_xy = w.pin_xy[self.r.graph().net_driver[net[first]]]
        self.assertTrue(torch.allclose(e[first, :2], drv_xy, atol=2.0 / self.d.dbu))
        e2, n2 = self.r.trees(nets[:5])
        self.assertTrue(set(n2.tolist()) <= set(nets[:5].tolist()))

    def test_copies_leave_the_timer_alone(self):
        g = self.r.graph()
        g.level.zero_(); g.n_from.zero_()
        g2 = self.r.graph()
        self.assertGreater(int(g2.level.max()), 0)
        self.assertIn('TimingGraph', repr(g2))


if __name__ == '__main__':
    unittest.main()
