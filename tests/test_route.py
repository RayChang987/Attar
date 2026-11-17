"""Global routes, their parasitics (attar.timing.grpar) and the global router (attar.route).
The OpenROAD comparisons need orfs_runs/gr_par/<design> (tools/gr_truth.py) and skip without it."""
import json
import os
import unittest

import numpy as np

from common import RUNS, need

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
