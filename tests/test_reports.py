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


if __name__ == '__main__':
    unittest.main()
