"""Error messages that name what was meant, and the reports behind them."""
import io
import unittest

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
        from attar import sizing
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


if __name__ == '__main__':
    unittest.main()
