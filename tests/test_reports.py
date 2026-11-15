"""Error messages that name what was meant, and the reports behind them."""
import io
import unittest

import torch

from common import SMALL, MEDIUM, need, attar


class FriendlyErrorsTest(unittest.TestCase):
    def test_unknown_names_suggest(self):
        d = need(SMALL)
        with self.assertRaisesRegex(KeyError, "did you mean 'legalize'"):
            d.require('legalise')
        with self.assertRaisesRegex(KeyError, 'unknown capability'):
            d.missing('nothing_like_it')
        with self.assertRaisesRegex(KeyError, "did you mean 'netlist'"):
            d.has('netlst')
        with self.assertRaisesRegex(KeyError, "did you mean 'hpwl'"):
            attar.Evaluator(d).missing('hpwll')

    def test_evaluate_takes_one_name(self):
        d = need(SMALL).restore('reference')
        ev = attar.Evaluator(d)
        self.assertEqual(ev.evaluate(metrics='hpwl')['hpwl'], ev.evaluate(metrics=('hpwl',))['hpwl'])
        self.assertIn('Evaluator of', repr(ev))
        self.assertTrue(all(not ln.endswith(' ') for ln in ev.status().splitlines()))


class ReportsTest(unittest.TestCase):
    def test_timing_report_is_the_result_report(self):
        d = need(MEDIUM).restore('reference')
        if 'sta' not in d.capabilities():
            raise unittest.SkipTest('no timing fact')
        a, b = io.StringIO(), io.StringIO()
        s1 = d.timing_report(endpoints=3, path=False, file=a)
        s2 = d.timing(None).report(endpoints=3, path=False, file=b)
        self.assertEqual(a.getvalue(), b.getvalue())
        self.assertEqual(s1['wns'], s2['wns'])
        self.assertEqual(len(s1['worst_endpoints']), 3)

    def test_status_lines_have_no_trailing_blanks(self):
        s = need(MEDIUM).status(show=False)
        self.assertTrue(all(not ln.endswith(' ') for ln in s.splitlines()))
        from attar.status import styled
        self.assertEqual(styled(s, io.StringIO()), s)            # not a terminal: unchanged

    def test_resize_round_trip(self):
        d = need(MEDIUM).restore('reference')
        if 'resize' not in d.capabilities():
            raise unittest.SkipTest('no library fact')
        cells = d.names['cells']
        i = next(i for i in range(len(cells)) if len(d.equivalents(i)) > 1)
        cur = d.names['masters'][int(d.raw['c_master'][i])]
        alt = next(m for m in d.equivalents(i) if m != cur)
        size0, off0, wns0 = d.size.clone(), d.pin_off.clone(), d.timing().wns
        self.assertEqual(d.resize(cells[i], alt), [cur])
        self.assertEqual(d.resize([i], [cur]), [alt])               # back: same geometry and timing
        self.assertTrue(bool((d.size == size0).all()) and bool((d.pin_off == off0).all()))
        self.assertAlmostEqual(d.timing().wns, wns0, places=6)
        with self.assertRaises(ValueError):
            d.resize(i, 'NOT_A_CELL')

    def test_equivalents_keep_clock_edge(self):
        """Sequentials must match (OpenSTA's equivCellSequentials): a rising-edge flop never resizes to a falling-edge
        one, nor a high latch to a low one, nor a 2-bit flop bank to a 4-bit one."""
        d = need(MEDIUM)
        if 'resize' not in d.capabilities():
            raise unittest.SkipTest('no library fact')
        from attar.timing import sizing
        L, _ = sizing.lib_cell(d, 'DFFHQNx1_ASAP7_75t_R')
        if L is None:
            raise unittest.SkipTest('not an ASAP7 library')
        fam = lambda c: {m.split('_')[0].rstrip('0123456789').rstrip('x') for m in L.class_of(c)}  # noqa: E731
        self.assertEqual(fam('DFFHQNx1_ASAP7_75t_R'), {'DFFHQN'})
        self.assertEqual(fam('DHLx1_ASAP7_75t_R'), {'DHL'})
        self.assertEqual(fam('SDFHx1_ASAP7_75t_R'), {'SDFH'})
        self.assertNotIn('DFFHQNH2V2Xx1_ASAP7_75t_R', L.class_of('DFFHQNV2Xx1_ASAP7_75t_R'))
        self.assertIn('ICGx2_ASAP7_75t_R', L.class_of('ICGx1_ASAP7_75t_R'))   # same statetable: still one class



class SlackTest(unittest.TestCase):
    def test_pin_and_net_slack_agree_with_the_endpoints(self):
        """Required times go back from the endpoints: an endpoint's pin slack is its setup slack, the worst pin and
        the worst net have the WNS, and a net's slack is its worst pin's."""
        import torch
        d = need(MEDIUM).restore('reference')
        if 'sta' not in d.capabilities():
            raise unittest.SkipTest('no timing fact')
        r = d.timing()
        s = r.pin_slack()
        self.assertEqual(s.shape, (len(d.p_net),))
        ep = r.endpoints
        self.assertTrue(bool((s[ep] == r.setup[ep]).all()))
        self.assertAlmostEqual(float(s.min()), r.worst_slack, delta=1e-3)
        n = r.net_slack()
        self.assertEqual(len(n), len(d.n_ptr) - 1)
        self.assertAlmostEqual(float(n.min()), r.worst_slack, delta=1e-3)
        k = int(torch.argmin(n))
        self.assertEqual(float(n[k]), float(s[d.p_net == k].min()))
        self.assertTrue(bool(torch.isinf(s).any()) and not bool(torch.isnan(s).any()))   # unconstrained: inf
        d.pos = d.pos + 0.0                                        # a new run: its own slacks
        self.assertIsNot(d.timing().pin_slack(), s)


class DelayCalcTest(unittest.TestCase):
    def test_extrapolated_negative_slew_clips_to_zero(self):
        # OpenSTA GateTableModel::gateDelay: a table slew extrapolated below 0 (a huge input slew off an unbuffered
        # net) is 0; a negative one used to drive min slews and arrivals negative downstream (hold off by ~60 ns)
        import torch
        from attar.timing import dmp
        ax1 = torch.tensor([[10.0, 20.0]]); ax2 = torch.tensor([[1.0, 2.0]]); n = torch.tensor([[2, 2]])
        delay = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]); slew = torch.tensor([[[30.0, 32.0], [20.0, 22.0]]])
        tab = (ax1, ax2, delay, n, ax1, ax2, slew, n)
        one = lambda v: torch.tensor([v])  # noqa: E731
        for s_in, want in ((15.0, 26.0), (100.0, 0.0)):                       # inside the table; far beyond it
            _, s, _, _ = dmp.gate(tab, one(s_in), one(1.5), one(0.0), one(0.0), one(1.5), one(0.5), one(0.2),
                                  one(0.8), one(1.0))
            self.assertAlmostEqual(float(s), want, places=4)



class PinLocationTest(unittest.TestCase):
    """The timer puts a pin where OpenROAD's estimate_parasitics does: dbITerm::getAvgXY, the mean of the centers of
    its LEF rectangles, truncated to the database unit (the Evaluator keeps their bounding box's center)."""

    def test_lef_pin_location_is_the_mean_of_its_rectangles(self):
        import os
        import tempfile
        from attar.io.lef import read_lef
        lef = ('MACRO X\n CLASS CORE ;\n SIZE 1 BY 1 ;\n PIN A\n  DIRECTION INPUT ;\n  PORT\n   LAYER M1 ;\n'
               '    RECT 0 0 0.1 1 ;\n    RECT 0 0.9 1 1 ;\n  END\n END A\nEND X\n')
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, 'x.lef')
            with open(f, 'w') as fh:
                fh.write(lef)
            m = read_lef([f])[0]['X']
        self.assertEqual(m.pins['A'][0], (0.0, 0.0, 1.0, 1.0))
        self.assertAlmostEqual(m.pin_loc['A'][0], 0.275)
        self.assertAlmostEqual(m.pin_loc['A'][1], 0.725)

    def test_timer_pins_are_openroads(self):
        import numpy as np
        d = need(SMALL).restore('reference')
        d.timing()
        t = d._timer
        loc = attar.timing.ir.pin_locations(d)
        pm = d.raw['p_mpin'].astype(np.int64)
        c = d.raw['p_cell'].astype(np.int64)
        o = d.orient.cpu().numpy()[c]
        ok = (pm >= 0) & np.isin(o, (0, 5))                       # N, FS (the rows' orientations)
        ok[ok] &= np.isfinite(loc[pm[ok]]).all(1)
        self.assertGreater(ok.sum(), 100)
        lx, ly = loc[pm[ok], 0], loc[pm[ok], 1]
        h = d.size_n.cpu().numpy().astype(np.float64)[c[ok], 1]
        ly = np.where(o[ok] == 5, h - ly, ly)
        ll = np.round(d.pos.cpu().numpy().astype(np.float64)[c[ok]] * d.dbu)
        want = ll + np.floor(np.stack([lx, ly], 1) * d.dbu + 1e-3)
        got = torch.round(t.ev._pins(None) * d.dbu).cpu().numpy()[ok]
        self.assertTrue(np.array_equal(got, want))
        box = torch.round(attar.Evaluator(d)._pins(None) * d.dbu).cpu().numpy()[ok]
        self.assertTrue((box != got).any())                     # multi-rectangle ASAP7 pins

    def test_timer_follows_orientation_changes(self):
        from attar.timing.sta import Timer
        d = need(SMALL).restore('reference')
        d.timing()
        d.orient = torch.where(d.orient == 0, torch.full_like(d.orient, 4), d.orient)   # N -> FN
        d.update_orient()
        r = d.timing()
        fresh = Timer(d)
        self.assertTrue(torch.equal(d._timer.ev.off, fresh.ev.off))
        self.assertTrue(torch.equal(r.setup, fresh.run().setup))


if __name__ == '__main__':
    unittest.main()
