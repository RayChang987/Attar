"""Global routes, their parasitics (attar.timing.grpar) and the global router (attar.route).
The OpenROAD comparisons need orfs_runs/gr_par/<design> (tools/gr_truth.py) and skip without it."""
import json
import os
import unittest

import numpy as np

from common import MEDIUM, RUNS, need

from attar import route
from attar.rules import routing as R
from attar.timing import grpar

GCD = 'after_fp/orfs_asap7/gcd'
TRUTH = os.path.join(RUNS, 'gr_par', 'gcd')


def chain_net(n_nets, k, r, c, cap):
    """n_nets copies of a driver pin - k wire nodes - load pin chain (resistor r between neighbors, node cap c)."""
    per = k + 2                                          # driver pin, k nodes, load pin
    N = n_nets * per
    base = np.arange(n_nets)[:, None] * per
    a_ = (base + np.arange(per - 1)[None]).ravel(); b_ = a_ + 1
    node_cap = np.tile(np.r_[0.0, np.full(k, c), 0.0], n_nets)
    P = 2 * n_nets
    pin_node = np.empty(P, np.int64); pin_node[0::2] = base[:, 0]; pin_node[1::2] = base[:, 0] + per - 1
    net = {'node_net': np.repeat(np.arange(n_nets), per), 'node_cap': node_cap, 'res_a': a_, 'res_b': b_,
           'res_r': np.full(len(a_), r), 'pin_node': pin_node, 'wire_cap': np.full(n_nets, k * c)}
    drv = np.arange(n_nets) * 2
    caps = np.zeros((P, 1)); caps[1::2] = cap
    return net, drv, caps, N


class ReduceTest(unittest.TestCase):
    def test_chain(self):
        """Elmore at the load = sum over resistors of r x downstream C; pi moments of a uniform chain."""
        k, r, c, cap = 5, 0.1, 2.0, 3.0
        net, drv, caps, _ = chain_net(1, k, r, c, cap)
        elm, pi, loops = grpar.reduce(net, drv, caps)
        down = [k * c + cap - i * c for i in range(k)] + [cap]           # downstream of each resistor
        self.assertAlmostEqual(elm[1, 0], r * sum(down), places=9)
        self.assertEqual(loops, 0)
        self.assertAlmostEqual(pi[0, 0, 0] + pi[0, 0, 2], k * c + cap, places=9)   # near + far = total C

    def test_random_trees_against_a_recursive_reference(self):
        """Branching trees of many depths (one deep enough that a pass per depth over all nodes was the bottleneck),
        and a net without a driver: Elmore delay at every pin and the pi moments, against a per-net recursion."""
        rng = np.random.default_rng(1)
        node_net, node_cap, ra, rb, rr, pin_node, drv, sink_pins = [], [], [], [], [], [], [], []
        N = 0
        trees = []                                            # (first node, parent list) per net
        for e, n in enumerate((1, 2, 7, 40, 600, 3)):
            parent = [-1] + [int(rng.integers(max(0, k - 3), k)) for k in range(1, n)]   # deep, some branching
            if e == 4:
                parent = [-1] + list(range(n - 1))                                       # a 600-node chain
            trees.append((N, parent))
            node_net += [e] * n
            node_cap += rng.uniform(0.1, 2.0, n).tolist()
            for k in range(1, n):
                ra.append(N + parent[k]); rb.append(N + k); rr.append(float(rng.uniform(0.01, 0.5)))
            pin_node.append(N); drv.append(len(pin_node) - 1 if e != 5 else -1)          # net 5: no driver
            for k in rng.choice(n, size=min(n, 3), replace=False):
                pin_node.append(N + int(k)); sink_pins.append(len(pin_node) - 1)
            N += n
        E = len(trees)
        net = {'node_net': np.array(node_net), 'node_cap': np.array(node_cap), 'res_a': np.array(ra, np.int64),
               'res_b': np.array(rb, np.int64), 'res_r': np.array(rr), 'pin_node': np.array(pin_node, np.int64),
               'wire_cap': np.bincount(node_net, node_cap, minlength=E)}
        P = len(pin_node)
        caps = rng.uniform(0.0, 1.0, (P, 2))
        elm, pi, loops = grpar.reduce(net, np.array(drv, np.int64), caps)
        self.assertEqual(loops, 0)
        res = {(a, b): r for a, b, r in zip(ra, rb, rr)}
        for e, (first, parent) in enumerate(trees):
            n = len(parent)
            cap = np.repeat(np.array(node_cap[first:first + n])[:, None], 2, 1)
            for p in range(P):
                if first <= pin_node[p] < first + n:
                    cap[pin_node[p] - first] += caps[p]
            kids = [[] for _ in range(n)]
            for k in range(1, n):
                kids[parent[k]].append(k)
            y = np.zeros((n, 3, 2))                            # downstream moments, children first
            for k in range(n - 1, -1, -1):
                y[k, 0] = cap[k]
                for ch in kids[k]:
                    r = res[(first + k, first + ch)]
                    a1, a2, a3 = y[ch]
                    y[k, 0] += a1; y[k, 1] += a2 - r * a1 * a1; y[k, 2] += a3 - 2 * r * a1 * a2 + r * r * a1 ** 3
            el = np.zeros((n, 2))
            for k in range(1, n):
                el[k] = el[parent[k]] + res[(first + parent[k], first + k)] * y[k, 0]
            for p in range(P):
                if first <= pin_node[p] < first + n:
                    want = el[pin_node[p] - first] if drv[e] >= 0 else 0.0
                    np.testing.assert_allclose(elm[p], want, rtol=1e-10, atol=1e-12)
            if drv[e] >= 0:
                Y1, Y2, Y3 = y[0]
                c1 = np.where(Y3 == 0, 0.0, Y2 * Y2 / np.where(Y3 == 0, 1.0, Y3))
                np.testing.assert_allclose(pi[e, :, 0] + pi[e, :, 2], Y1, rtol=1e-10)
                np.testing.assert_allclose(pi[e, :, 2], c1, rtol=1e-10)
            else:
                self.assertTrue(np.isnan(pi[e]).all())

    def test_many_nodes(self):
        """More nodes than int32 edge keys hold (regression: scipy's int32 BFS order overflowed them)."""
        net, drv, caps, N = chain_net(8000, 6, 0.05, 1.0, 2.0)
        self.assertGreater(N, 50000)
        elm, _, _ = grpar.reduce(net, drv, caps)
        one, d1, c1, _ = chain_net(1, 6, 0.05, 1.0, 2.0)
        e1, _, _ = grpar.reduce(one, d1, c1)
        np.testing.assert_allclose(elm[1::2, 0], e1[1, 0], rtol=1e-12)


class OpenROADRoutesTest(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(os.path.join(TRUTH, 'truth.json')):
            self.skipTest('no OpenROAD routes for gcd (tools/gr_truth.py gcd)')
        self.d = need(GCD)
        self.d.load_reference()
        self.t = json.load(open(os.path.join(TRUTH, 'truth.json')))

    def routes(self):
        d = self.d
        rt = route.read_segments(d, os.path.join(TRUTH, 'routes.segs'))
        names = d.names['layers']
        g = R.grid(d)
        horiz = {i + 1: bool(h) for i, h in enumerate(g['horizontal'])}
        tracks = {}
        for i in range(len(names)):
            a, p, n = R._layer_tracks(d.raw, i, horiz[i + 1])
            if n:
                tracks[i + 1] = (a, p)
        hi = names.index(d.manifest['routing']['layers'][1]) + 1
        route.pins_from_shapes(d, rt, route.read_shapes(os.path.join(TRUTH, 'shapes.txt')), hi, g['tile'], tracks, horiz)
        return rt, g

    def test_wirelength_and_pins(self):
        rt, g = self.routes()
        self.assertEqual(rt.wirelength(g['tile']), int(self.t['gr_wl_um']))
        tg, found = route.read_pin_locations(self.d, os.path.join(TRUTH, 'pins.loc'))
        self.assertTrue(((tg == rt.pin_grid).all(1) | ~found).all())

    def test_slacks(self):
        from attar.timing.sta import Timer
        rt, _ = self.routes()
        res = Timer(self.d, parasitics='global_routing', routes=rt).run()
        tu = self.t['time_unit_ps']
        ref = {}
        for line in open(os.path.join(TRUTH, 'endpoints.tsv')):
            t = line.split()
            try:
                ref[t[0]] = (float(t[1]) * tu, float(t[2]) * tu)
            except (ValueError, IndexError):
                pass
        ours = res.endpoint_slacks()
        common = [k for k in ref if k in ours and abs(ref[k][0]) < 1e20 and np.isfinite(ours[k][0])]
        self.assertGreater(len(common), 0)
        err = max(abs(ours[k][0] - ref[k][0]) for k in common)
        self.assertLess(err, 0.1)


class RouterTest(unittest.TestCase):
    def test_pattern_costs_on_a_non_square_grid(self):
        """Both pattern costs are computed for every branch; a mid on its own axis must not index past the other one."""
        import torch
        from attar.route.router import _Grid2D
        for nx, ny in ((7, 3), (3, 7)):
            G = _Grid2D(torch.ones(ny, nx - 1), torch.ones(nx, ny - 1), 'cpu')
            seg = torch.tensor([[0, 0, nx - 1, ny - 1]])
            G.usage(torch.zeros(0, 5, dtype=torch.long)); G.costs(0.5)
            for t, m in ((0, nx - 1), (1, ny - 1)):
                c = G.path(seg, torch.tensor([t]), torch.tensor([m]), via=2.0)
                self.assertEqual(c.item(), (nx - 1) + (ny - 1) + 2.0)   # unit costs plus one bend's two vias
            typ, mid = G.best(seg, margin=2, via=2.0)
            self.assertEqual(G.path(seg, typ, mid, via=2.0).item(), (nx - 1) + (ny - 1) + 2.0)

    def test_new_routes_redo_only_the_changed_nets(self):
        """d.timing(routes=other) on the kept timer (Timer.set_routes) equals a new timer on those routes, bit for
        bit, redoing only the nets whose route or pins changed; and back to the first routes, the first result."""
        import torch
        from attar.timing.sta import Timer
        d = need(MEDIUM)
        d.load_reference()
        r1, _ = route.global_route(d, verbose=False)
        t1 = d.timing(routes=r1)
        s1 = (t1.setup.clone(), t1.hold.clone())
        T = d._gr_timer[1]
        mv = torch.nonzero(d.movable).flatten()
        sel = mv[torch.randperm(len(mv), generator=torch.Generator().manual_seed(0))[:len(mv) // 50].to(mv.device)]
        p = d.pos.clone(); p[sel] += 1.5; d.pos = p
        r2, _ = route.global_route(d, verbose=False)
        t2 = d.timing(routes=r2)
        self.assertIs(d._gr_timer[1], T)                              # the same timer, patched
        fresh = Timer(d, parasitics='global_routing', routes=r2).run()
        self.assertTrue(torch.equal(t2.setup, fresh.setup) and torch.equal(t2.hold, fresh.hold))
        self.assertEqual(T.gr_loops, fresh.timer.gr_loops)
        E = len(d.n_ptr) - 1
        self.assertLess(T.set_routes(r1), E)                          # back: only the nets that differ
        self.assertEqual(T.set_routes(r1), 0)
        t3 = T.run()
        self.assertTrue(torch.equal(t3.setup, s1[0]) and torch.equal(t3.hold, s1[1]))

    def test_incremental_reroute(self):
        """global_route(previous=...): only the nets with a pin in another gcell (or layer) are routed again; every
        other net keeps its segments as they were; every net is routed; other parameters: a full route."""
        import torch
        d = need(MEDIUM)
        d.load_reference()
        r1, s1 = route.global_route(d, verbose=False)
        mv = torch.nonzero(d.movable).flatten()
        sel = mv[torch.randperm(len(mv), generator=torch.Generator().manual_seed(1))[:len(mv) // 100].to(mv.device)]
        p = d.pos.clone(); p[sel] += 4.0; d.pos = p
        r2, s2 = route.global_route(d, verbose=False, previous=r1)
        self.assertTrue(s2['incremental'])
        n = d.raw['n_ptr']
        p_net = np.repeat(np.arange(len(n) - 1), np.diff(n))
        moved = (r1.pin_grid != r2.pin_grid).any(1) | (r1.pin_layer != r2.pin_layer)
        changed = np.zeros(len(n) - 1, bool); changed[p_net[moved]] = True
        self.assertEqual(s2['nets_rerouted'], int(changed.sum()))
        self.assertGreater(s2['nets_rerouted'], 0)
        for e in np.nonzero(~changed)[0][::97]:                       # a sample of the others: segments unchanged
            self.assertTrue(np.array_equal(r1.seg[r1.seg_net == e], r2.seg[r2.seg_net == e]))
        full, _ = route.global_route(d, verbose=False)
        self.assertTrue(np.array_equal(r2.routed(), full.routed()))   # the same nets routed
        _, s3 = route.global_route(d, verbose=False, previous=r1, margin=5)
        self.assertFalse(s3['incremental'])                            # other parameters: a full route

    def test_gcd(self):
        d = need(GCD)
        d.load_reference()
        rt, st = route.global_route(d, verbose=False)
        n = d.raw['n_ptr']
        self.assertGreater(st['nets_routed'], 0)
        self.assertEqual(st['overflow_2d'], 0)
        net = grpar.build(d, rt, json.load(open(os.path.join(d.path, 'timing.json')))['rc'])
        self.assertEqual(net['pins_missing'], 0)                  # every pin of a routed net reaches its route
        multi = np.diff(n) >= 2
        self.assertTrue(rt.routed()[multi].mean() > 0.99)
        if os.path.exists(os.path.join(TRUTH, 'truth.json')):
            ref = json.load(open(os.path.join(TRUTH, 'truth.json')))['gr_wl_um']
            self.assertLess(abs(st['wirelength_um'] - ref) / ref, 0.1)


if __name__ == '__main__':
    unittest.main()
