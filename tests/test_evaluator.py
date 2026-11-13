"""Evaluator metrics, rectangle kernels, the Bookshelf reader."""
import unittest

import numpy as np  # noqa: F401
import torch  # noqa: F401

from common import DEV, SMALL, MEDIUM, BOOKSHELF, need, bookshelf_hpwl_numpy, attar, _find


class WATest(unittest.TestCase):
    def test_wa_approaches_hpwl_and_has_gradients(self):
        d = need(MEDIUM).restore('reference')
        h = d.hpwl().item()
        p = d.pos.clone().requires_grad_(True)
        wa = d.wa_wirelength(p, gamma=0.05)
        wa.backward()
        self.assertLess(abs(wa.item() - h) / h, 0.02)
        self.assertGreater(abs(wa.item() - h), 0)           # smooth, so slightly below HPWL
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad[d.movable].abs().sum()), 0)

class BookshelfTest(unittest.TestCase):
    def test_hpwl_matches_an_independent_computation_on_the_raw_files(self):
        d = need(BOOKSHELF).restore('dreamplace_lg')
        snap = _find(d.manifest['snapshots']['dreamplace_lg'])
        ref = bookshelf_hpwl_numpy(_find(d.manifest['source']['aux']), snap)
        self.assertAlmostEqual(d.hpwl().item() / ref, 1.0, delta=1e-5)

    def test_bookshelf_is_tier_c_and_starts_unplaced(self):
        d = need(BOOKSHELF)
        self.assertEqual(d.tier(), 'C')
        self.assertEqual(d.missing('hpwl'), ['placement'])
        self.assertIn('global_place', d.capabilities())
        self.assertEqual(d.facts['floorplan.die'], 'bookshelf(rows+fixed)')
        self.assertIn('dreamplace_lg', d.snapshots)

class EvaluatorTest(unittest.TestCase):
    def test_all_modes_agree_with_design_hpwl(self):
        for name, snap in ((MEDIUM, 'reference'), (BOOKSHELF, 'dreamplace_lg')):
            d = need(name).restore(snap)
            ref = d.hpwl().item()
            ev = attar.Evaluator(d)
            self.assertAlmostEqual(ev.hpwl().item() / ref, 1.0, delta=1e-6)
            self.assertAlmostEqual(ev.hpwl_per_net().sum().item() / ref, 1.0, delta=1e-5)
            pb = torch.stack([d.pos, d.pos + 1.0])
            hb = ev.hpwl_batch(pb)
            self.assertAlmostEqual(hb[0].item() / ref, 1.0, delta=1e-6)
            if DEV == 'cuda':
                fast = attar.Evaluator(d, fast=True)
                self.assertAlmostEqual(fast.hpwl().item() / ref, 1.0, delta=1e-6)
                p2 = d.pos.clone(); p2[d.movable] += 0.5
                self.assertAlmostEqual(fast.hpwl(p2).item(), ev.hpwl(p2).item(), delta=1e-3 * ref)  # graph sees new input

    def test_metrics_follow_facts(self):
        d = need(MEDIUM)
        ev = attar.Evaluator(d)
        self.assertEqual(ev.available(), [])              # unplaced: nothing to measure yet
        self.assertEqual(ev.missing('hpwl'), ['placement'])
        self.assertIn('(not implemented yet)', ev.missing('gr_wirelength'))
        self.assertEqual(ev.missing('gr_overflow'), ['placement'])
        self.assertEqual(ev.missing('rudy'), ['placement'])
        d.restore('reference')
        self.assertEqual(ev.available(), ['density_overflow', 'gr_overflow', 'hpwl', 'legality', 'rudy', 'star_wirelength',
                                          'steiner_wirelength', 'timing'])
        r = ev.evaluate()
        self.assertIn('hpwl', r); self.assertEqual(r['fidelity'], 'F1')
        self.assertIn('wns', ev.evaluate(metrics=('timing',))['timing'])
        with self.assertRaises(NotImplementedError):
            ev.evaluate(metrics=('gr_wirelength',))

    def test_orientation_change_is_picked_up(self):
        d = need(SMALL).restore('reference')
        ev = attar.Evaluator(d)
        ev.hpwl()
        mv = torch.nonzero(d.movable).flatten()[:20]
        d.orient[mv] = 4; d.update_orient()             # flip some cells: pin offsets change
        h0 = d.hpwl().item()
        self.assertAlmostEqual(ev.hpwl().item() / h0, 1.0, delta=1e-6)

class KernelTest(unittest.TestCase):
    def test_rectangle_accumulate_and_integrate_match_brute_force(self):
        from attar.kernels import Grid
        g = Grid(0, 0, 10, 8, 7, 5, DEV)
        gen = torch.Generator().manual_seed(0)
        n = 200
        x0 = (torch.rand(n, generator=gen) * 12 - 1).to(DEV); y0 = (torch.rand(n, generator=gen) * 10 - 1).to(DEV)
        x1 = x0 + (torch.rand(n, generator=gen) * 4).to(DEV); y1 = y0 + (torch.rand(n, generator=gen) * 4).to(DEV)
        v = torch.rand(n, generator=gen).to(DEV)
        X = np.linspace(0, 10, 8); Y = np.linspace(0, 8, 6)
        ox = np.clip(np.minimum(x1.cpu().numpy()[:, None], X[1:]) - np.maximum(x0.cpu().numpy()[:, None], X[:-1]), 0, None)
        oy = np.clip(np.minimum(y1.cpu().numpy()[:, None], Y[1:]) - np.maximum(y0.cpu().numpy()[:, None], Y[:-1]), 0, None)
        ref = np.einsum('n,ni,nj->ij', v.cpu().numpy(), ox, oy)
        c = g.corners(x0, y0, x1, y1)
        np.testing.assert_allclose(g.accumulate_c(c, v).cpu().numpy(), ref, atol=1e-4)
        F = torch.rand(2, 7, 5, generator=gen).to(DEV)
        refI = np.einsum('ni,nj,kij->nk', ox, oy, F.cpu().numpy())
        np.testing.assert_allclose(g.integrate_c(F, c).cpu().numpy(), refI, atol=1e-4)

class MetricsTest(unittest.TestCase):
    def test_rmst_matches_brute_force_prim(self):
        from attar import metrics as Mx
        gen = torch.Generator().manual_seed(1)
        degs = torch.randint(4, 40, (60,), generator=gen)
        p = torch.rand(int(degs.sum()), 2, generator=gen) * 100
        ptr = torch.cat([torch.zeros(1, dtype=torch.long), degs.cumsum(0)])
        r = Mx.rmst(p.to(DEV), degs.to(DEV), ptr.to(DEV), len(degs)).cpu()
        for i in range(len(degs)):
            q = p[ptr[i]:ptr[i + 1]]
            dist = (q[:, None] - q[None]).abs().sum(-1)
            inn = [0]; tot = 0.0; key = dist[0].clone(); key[0] = float('inf')
            for _ in range(len(q) - 1):
                j = int(key.argmin()); tot += key[j].item(); inn.append(j)
                key = torch.minimum(key, dist[j]); key[inn] = float('inf')
            self.assertAlmostEqual(r[i].item() / tot, 1.0, delta=1e-5)

    def test_wirelength_models_are_ordered(self):
        d = need(MEDIUM).restore('reference')
        ev = attar.Evaluator(d)
        h, st, sr = ev.hpwl_per_net(), ev.steiner_wirelength(per_net=True), ev.star_wirelength(per_net=True)
        self.assertTrue(bool((st >= h - 1e-3).all()))               # RSMT >= HPWL
        self.assertTrue(bool((st <= sr + 1e-3).all()))              # the estimate never exceeds the star tree
        small = (ev.deg >= 2) & (ev.deg <= 3)
        self.assertTrue(torch.allclose(st[small], h[small], atol=1e-3))  # exact for <= 3 pins

    def test_wa_gradient_matches_autograd(self):
        from attar import metrics as Mx
        d = need(MEDIUM).restore('reference'); ev = attar.Evaluator(d)
        pos = d.pos.clone().requires_grad_(True)
        v_ref = d.wa_wirelength(pos, gamma=0.5); g_ref, = torch.autograd.grad(v_ref, pos)
        v, gp = Mx.wa_value_grad(pos.detach()[ev.cell] + ev.off, ev.net, ev.E, 0.5, ev.wmask)
        g = torch.zeros_like(pos).index_add_(0, ev.cell, gp)
        self.assertAlmostEqual(v.item() / v_ref.item(), 1.0, delta=1e-5)
        self.assertLess((g - g_ref).abs().max().item(), 1e-3 * g_ref.abs().max().item())

    def test_rudy_integrates_to_the_clamped_bounding_boxes(self):
        d = need(MEDIUM).restore('reference'); ev = attar.Evaluator(d)
        r = ev.rudy(bins=32)
        g = ev.grid(32, d.die_box[0])
        p = d.pos[ev.cell] + ev.off
        from attar import metrics as Mx
        lo, hi = Mx.net_bbox(p, ev.net, ev.E)
        ok = ev.deg >= 2
        sz = torch.maximum(hi - lo, torch.tensor([g.bw, g.bh], device=d.device))[ok]
        self.assertAlmostEqual((r['map'].sum() * g.bin_area).item() / sz.sum().item(), 1.0, delta=1e-3)

    def test_legality_checker(self):
        good = need(BOOKSHELF).restore('dreamplace_lg')
        self.assertTrue(attar.Evaluator(good).legality()['legal'])
        bad = need(MEDIUM).restore('reference')                       # ORFS global placement: not legal yet
        self.assertFalse(attar.Evaluator(bad).legality()['legal'])



class RowGapTest(unittest.TestCase):
    def test_a_cell_in_a_row_gap_is_outside_the_rows(self):
        from attar.rails import base_rows, row_gaps
        d = need('after_fp/orfs_asap7/cva6')                    # rows cut around the macros: gaps between row segments
        _, ys, *_ = base_rows(d)
        g_r, g_x0, g_x1 = row_gaps(d, ys)
        self.assertGreater(len(g_r), 0)
        d.restore('reference')
        ev = attar.Evaluator(d)
        before = ev.legality()['outside_rows']
        i = int(torch.nonzero(d.movable & (d.size[:, 1] <= float(d.row_h.min()) + 1e-6)).flatten()[0])
        k = int(torch.argmax(g_x1 - g_x0))             # the widest gap
        sw = float(d.row_site_w.min())
        d.pos[i, 0] = g_x0[k] + sw * round(float(g_x1[k] - g_x0[k]) / sw / 2)
        d.pos[i, 1] = ys[g_r[k]]
        self.assertEqual(attar.Evaluator(d).legality()['outside_rows'], before + 1)

if __name__ == '__main__':
    unittest.main()
