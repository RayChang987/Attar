"""Incremental legalization, detailed placement and the displacement metric (attar/naive/lg.py, attar/naive/dp.py)."""
import unittest

import torch

from common import attar  # noqa: F401  (puts the checkout on the path)
from attar import naive
from attar.naive.dp import SKIP_NETS
from common import SMALL, need


def _placed(name):
    d = need(name)
    d.place(verbose=False)
    return d


def _costed_hpwl(d, xonly):
    """Weighted HPWL of the nets dpo costs (2..99 pins), float64."""
    pp = d.pin_pos().double()
    E = len(d.n_weight)
    ix = d.p_net[:, None].expand(-1, 2)
    lo = torch.full((E, 2), float('inf'), dtype=torch.float64, device=pp.device).scatter_reduce(0, ix, pp, 'amin')
    hi = torch.full((E, 2), -float('inf'), dtype=torch.float64, device=pp.device).scatter_reduce(0, ix, pp, 'amax')
    deg = d.n_ptr[1:] - d.n_ptr[:-1]
    ok = (deg >= 2) & (deg < SKIP_NETS)
    sp = hi - lo
    v = sp[:, 0] if xonly else sp.sum(1)
    return float((v * d.n_weight.double())[ok].sum())


class TestIncrementalLegalize(unittest.TestCase):
    def test_legal_input_unchanged(self):
        d = _placed(SMALL)
        p0, o0 = d.pos.clone(), d.orient.clone()
        st = naive.legalize(d, verbose=False)
        self.assertEqual(st['mode'], 'incremental')
        self.assertEqual(st['moved_cells'], 0)
        self.assertTrue(torch.equal(d.pos, p0))
        self.assertTrue(torch.equal(d.orient, o0))

    def test_named_cells_only(self):
        d = _placed(SMALL)
        p0 = d.pos.clone()
        g = torch.Generator().manual_seed(0)
        mv = torch.nonzero(d.movable).flatten().cpu()
        pick = mv[torch.randperm(len(mv), generator=g)[:20]].to(d.device)
        q = p0.clone()
        q[pick, 0] += float(d.row_site_w.min()) * 7
        q[pick, 1] += float(d.row_h.min())
        d.pos = q.clone()
        st = naive.legalize(d, cells=pick, verbose=False)
        self.assertEqual(st['mode'], 'incremental')
        ev = attar.Evaluator(d)
        self.assertTrue(ev.legality()['legal'])
        named = torch.zeros(len(q), dtype=torch.bool, device=q.device); named[pick] = True
        moved = (d.pos - q).abs().sum(1) > 0
        self.assertEqual(int((moved & ~named).sum()), 0)            # only the named cells move

    def test_global_on_unplaced(self):
        d = need(SMALL)
        naive.global_place(d, verbose=False)
        st = naive.legalize(d, verbose=False)
        self.assertEqual(st['mode'], 'global')
        self.assertTrue(attar.Evaluator(d).legality()['legal'])


class TestDisplacement(unittest.TestCase):
    def test_values(self):
        d = _placed(SMALL)
        ev = attar.Evaluator(d)
        q = d.pos.clone()
        c = torch.nonzero(d.movable).flatten()[:3]
        q[c[0], 0] += 1.0; q[c[1], 1] -= 2.0; q[c[2]] += torch.tensor([0.5, 0.5], device=q.device)
        r = ev.displacement(q)
        self.assertAlmostEqual(r['total'], 4.0, places=5)
        self.assertAlmostEqual(r['max'], 2.0, places=5)
        self.assertEqual(r['moved'], 3)
        self.assertEqual(r['cells'], int(d.movable.sum()))


class TestDetailedPlace(unittest.TestCase):
    def test_legal_and_exact_gains(self):
        d = _placed(SMALL)
        for op, xonly in (('gs', False), ('vs', False), ('ro', True), ('flip', True)):
            d.restore('naive_lg')
            before = _costed_hpwl(d, xonly)
            st = naive.detailed_place(d, ops=(op,), verbose=False)
            self.assertTrue(st['legal'])
            self.assertTrue(attar.Evaluator(d).legality()['legal'])
            # batches share no costed net: the claimed gain is the measured one
            self.assertAlmostEqual(before - _costed_hpwl(d, xonly), st.get(op + '_gain', 0.0), delta=1e-3)
            self.assertLessEqual(st['hpwl_after'], st['hpwl_before'] + 1e-6)

    def test_needs_legal_input(self):
        d = need(SMALL)
        naive.global_place(d, verbose=False)
        with self.assertRaises(RuntimeError):
            naive.detailed_place(d, verbose=False)

    def test_pdn_heavy_design_in_little_memory(self):
        # ispd19_test4: M2 PG stripes a whole chip high over the cells' pins. The PDN index used to make every bin
        # the whole chip, and global swap checked the PDN for every candidate: out of memory at 24 GB (GPU) and 98 GB
        # (CPU). Now about 1.2 GB.
        d = need('after_fp/ispd2019/test4')
        self.assertTrue(d.place(verbose=False)['legal'])
        if d.device.type == 'cuda':
            torch.cuda.synchronize(d.device); torch.cuda.reset_peak_memory_stats(d.device)
        st = naive.detailed_place(d, verbose=False)
        self.assertTrue(st['legal'])
        self.assertLess(st['hpwl_after'], st['hpwl_before'])
        if d.device.type == 'cuda':
            self.assertLess(torch.cuda.max_memory_allocated(d.device), 4 * 2**30)


class TestPdnIndex(unittest.TestCase):
    def test_hits_equal_brute_force_for_any_box_size(self):
        # long stripes next to small vias; query boxes from 0.01 to 50 um, larger and smaller than the bins (the
        # index used to look only in the bins of a box's corners and missed blockages under a wide box's middle)
        from attar.rules import pdn
        d = need(SMALL)
        dbu = d.dbu
        g = torch.Generator().manual_seed(0)
        n = 400
        x = torch.randint(0, 200 * dbu, (n,), generator=g); y = torch.randint(0, 200 * dbu, (n,), generator=g)
        w = torch.randint(1, dbu // 2, (n,), generator=g); h = torch.randint(1, dbu // 2, (n,), generator=g)
        h[:40] = 150 * dbu                                                    # stripes
        L = torch.randint(0, 3, (n,), generator=g)
        m = 20000
        c = torch.rand(m, 2, generator=g, dtype=torch.float64) * 220 - 10
        s = 10 ** (torch.rand(m, 2, generator=g, dtype=torch.float64) * 3.7 - 2)
        boxes = torch.cat([c - s / 2, c + s / 2], 1)
        lay = torch.randint(0, 3, (m,), generator=g)
        for k in (0, 40):                                                    # with and without the stripes
            bl = torch.stack([L, x, y, x + w, y + h], 1)[k:].numpy()
            ix = pdn.Index(d, d.device, blockages=bl)
            R = torch.as_tensor(bl[:, 1:], dtype=torch.float64) / dbu
            truth = ((boxes[:, None, 0] < R[None, :, 2]) & (R[None, :, 0] < boxes[:, None, 2]) &
                     (boxes[:, None, 1] < R[None, :, 3]) & (R[None, :, 1] < boxes[:, None, 3]) &
                     (lay[:, None] == torch.as_tensor(bl[:, 0])[None])).any(1)
            self.assertGreater(int(truth.sum()), 1000)
            self.assertTrue(torch.equal(ix.hits(boxes.to(d.device), lay.to(d.device)).cpu(), truth))
            ix.BUDGET = 1000                                                 # the same in many small slices
            self.assertTrue(torch.equal(ix.hits(boxes.to(d.device), lay.to(d.device)).cpu(), truth))


if __name__ == '__main__':
    unittest.main()


class TestAbacus(unittest.TestCase):
    """naive.legalize(method='abacus'): the C++ Abacus on synthetic segments, and legal results on real designs."""

    def setUp(self):
        from attar.naive import abacus
        if not abacus.available():
            self.skipTest('no g++: %s' % abacus.error())

    def test_cluster_collapse_and_obstacles(self):
        from attar.naive.abacus import solve_class
        f = lambda *v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
        # one band, two segments [0, 10) and [20, 40) (an obstacle in between); three cells of width 2 all aiming at
        # x = 30 (their left edges): one cluster at the mean of 30, 30 - 2, 30 - 4, i.e. from 28
        band, x, unfit = solve_class(f(30, 30, 30), f(0, 0, 0), f(2, 2, 2), f(0.0), f(0.0), torch.tensor([0, 0]),
                                     f(0, 20), f(10, 40), 1.0)
        self.assertFalse(bool(unfit.any()))
        self.assertEqual(sorted(x.tolist()), [28.0, 30.0, 32.0])
        # a cell aiming inside the obstacle goes to the nearer segment's edge; one too wide for both is unfit
        band, x, unfit = solve_class(f(13, 17, 5), f(0, 0, 0), f(2, 2, 50), f(0.0), f(0.0), torch.tensor([0, 0]),
                                     f(0, 20), f(10, 40), 1.0)
        self.assertEqual(x[:2].tolist(), [8.0, 20.0])
        self.assertEqual(unfit.tolist(), [False, False, True])
        # two bands: the nearer in y wins unless its segment is full
        band, x, unfit = solve_class(f(1, 1, 1), f(0, 0, 0), f(4, 4, 4), f(0.0, 1.0), f(0.0, 0.0),
                                     torch.tensor([0, 1]), f(0, 0), f(8, 8), 1.0)
        self.assertEqual(sorted(band.tolist()), [0, 0, 1])

    def test_legal_and_reproducible(self):
        from attar import Evaluator
        for name in (SMALL, 'after_fp/orfs_asap7/riscv32i'):
            d = need(name)
            naive.global_place(d, verbose=False, seed=0, max_iters=300)
            d.snapshot('gp_abacus_test')
            st = naive.legalize(d, verbose=False, method='abacus')
            self.assertEqual(st['method'], 'abacus')
            self.assertTrue(Evaluator(d).legality()['legal'], name)
            p1 = d.pos.clone()
            d.restore('gp_abacus_test')
            self.assertEqual(naive.legalize(d, verbose=False)['method'], 'abacus')     # the default
            self.assertTrue(torch.equal(d.pos, p1))
        with self.assertRaises(ValueError):
            naive.legalize(d, verbose=False, method='nope')


class TestDeterministicGlobalPlace(unittest.TestCase):
    """global_place(deterministic=True) on the GPU: the same placement twice; kernels.cumsum is a cumsum."""

    def test_same_twice(self):
        if not torch.cuda.is_available():
            self.skipTest('no GPU')
        from attar import kernels
        x = torch.rand(5, 300, device='cuda', dtype=torch.float64)
        with kernels.deterministic_algorithms():
            self.assertTrue(torch.allclose(kernels.cumsum(x, 1), x.cumsum(1)))
            self.assertTrue(torch.allclose(kernels.cumsum(x, 0), x.cumsum(0)))
        self.assertFalse(torch.are_deterministic_algorithms_enabled())          # restored
        out = []
        for _ in range(2):
            d = attar.load(SMALL, 'cuda', download=False)
            naive.global_place(d, verbose=False, seed=0, deterministic=True)
            out.append(d.pos.clone())
        self.assertTrue(torch.equal(out[0], out[1]))
