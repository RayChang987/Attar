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


if __name__ == '__main__':
    unittest.main()
