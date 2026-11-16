"""Design rules (attar.timing.drv), power (attar.timing.power), the Liberty and SDC data they read, and propagated
clocks. Agreement with OpenSTA is checked by tools/validate_drv.py, validate_power.py and validate_cts.py; these tests
check the plumbing on the zoo's small designs."""
import math
import os
import unittest

import numpy as np
import torch

from common import SMALL, need

from attar.io import liberty, sdc


class LibertyTest(unittest.TestCase):
    def test_unterminated_attribute_before_group(self):
        """'area : 0.2' without ';' on its own line, then a group: both read (ASAP7 writes this)."""
        lib = liberty.parse('library (x) { cell (c) {\narea : 0.2\npg_pin (VDD) { pg_type : primary_power; } } }',
                            liberty.SKIP - liberty.POWER_GROUPS)
        c = lib.sub('cell')[0]
        self.assertEqual(c.attrs['area'], '0.2')
        self.assertEqual([g.kind for g in c.groups], ['pg_pin'])

    def test_power_groups(self):
        text = '''library (t) { time_unit : "1ps"; capacitive_load_unit (1, ff); leakage_power_unit : "1pW";
          voltage_unit : "1V"; voltage_map (VDD, 0.7); default_max_transition : 100; default_fanout_load : 1;
          power_lut_template (p2) { variable_1 : input_transition_time; variable_2 : total_output_net_capacitance;
            index_1 ("1, 2"); index_2 ("1, 2"); }
          cell (INV) { area : 1; pg_pin (VDD) { pg_type : primary_power; voltage_name : VDD; }
            leakage_power () { value : 500; when : "A"; related_pg_pin : VDD; }
            pin (A) { direction : input; capacitance : 1; max_transition : 50; }
            pin (Y) { direction : output; function : "!A"; max_capacitance : 10; related_power_pin : VDD;
              internal_power () { related_pin : "A"; rise_power (p2) { values ("1, 2", "3, 4"); }
                                  fall_power (p2) { values ("5, 6", "7, 8"); } } } } }'''
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.lib', delete=False) as f:
            f.write(text)
        try:
            L = liberty.read(f.name, power=True)
        finally:
            os.remove(f.name)
        c = L['cells']['INV']
        self.assertEqual(L['limits']['default_max_transition'], 100.0)
        self.assertEqual(L['voltage_map'], {'VDD': 0.7})
        self.assertEqual(c['pins']['A']['max_transition'], 50.0)
        self.assertEqual(c['pins']['Y']['max_cap'], 10.0)
        self.assertAlmostEqual(c['leakage_groups'][0]['value'], 0.5)          # pW -> nW
        g = c['internal_power'][0]
        self.assertEqual((g['pin'], g['related']), ('Y', 'A'))
        self.assertEqual(g['tables']['fall'][2][1][0], 7.0)                   # fJ


class SdcTest(unittest.TestCase):
    def test_limits_and_propagated(self):
        r = sdc.read('create_clock -name clk -period 100 [get_ports clk]\nset_max_fanout 8 [current_design]\n'
                     'set_max_transition 50 [get_ports a]\nset_propagated_clock [get_clocks clk]\n',
                     ['clk', 'a'], [], 'top', is_text=True)
        self.assertEqual(r['limits']['max_fanout']['design'], 8.0)
        self.assertEqual(r['limits']['max_transition']['ports'], {'a': 50.0})
        self.assertTrue(r['clocks']['clk']['propagated'])


class DrvPowerTest(unittest.TestCase):
    def setUp(self):
        self.d = need(SMALL)
        self.d.restore('reference')

    def test_drv(self):
        x = self.d.drv()
        s = x.summary()
        self.assertGreater(s['slew_pins'], 0)
        self.assertGreater(s['cap_pins'], 0)
        ok = torch.isfinite(x.slew_slack)
        self.assertTrue(bool((x.slew[ok] >= 0).all()))
        self.assertTrue(torch.allclose(x.slew_slack[ok], (x.slew_limit - x.slew)[ok]))

    def test_power(self):
        a = self.d.power()
        b = self.d.power()                                  # deterministic
        self.assertGreater(a.total[0], 0)
        self.assertGreater(a.total[1], 0)
        self.assertGreater(a.total[2], 0)
        self.assertEqual(a.summary(), b.summary())
        g = sum(sum(v) for v in a.groups.values())
        self.assertAlmostEqual(g, sum(a.total), places=12)

    def test_propagated_without_tree(self):
        """A propagated clock whose source drives the registers directly: no tree, latency = the wire only."""
        T = self.d.timing().timer
        self.assertEqual(int(T.pclk.sum()), 0)              # gcd's 3_3 SDC: ideal clocks
        self.assertEqual(int(T.iclk.sum()), int((T.clk_of >= 0).sum()))


class CtsTest(unittest.TestCase):
    def test_cts_gcd(self):
        import attar.naive as N
        from attar.io.distill import N_CLOCK
        d = need(SMALL)
        d.restore('reference')
        N.legalize(d, verbose=False)
        n0 = len(d.names['cells'])
        st = N.cts(d, verbose=False)
        self.assertGreater(st['buffers'], 0)
        self.assertEqual(len(d.names['cells']), n0 + st['buffers'])
        r = d.timing()
        T = r.timer
        real = [i for i, c in enumerate(T.clocks) if c['sources']]      # virtual clocks have no tree: ideal
        self.assertTrue(bool(T.prop_clk[real].all()))
        ca = r.clock_arrivals()
        self.assertTrue(all(v[0] > 0 and v[0] < float('inf') for v in ca.values()))     # timed through the tree
        new = [k for k, n in enumerate(d.names['nets']) if n.startswith('attar_cts_net')]
        self.assertTrue(all(d.raw['n_flags'][k] & N_CLOCK for k in new))
        # CRPR (attar.timing.crpr): every check covered by the entries, a credit between 0 and the largest late /
        # early difference at a clock pin, deterministic
        self.assertIsNotNone(r.crpr)
        self.assertEqual(r.crpr['uncovered'], 0)
        r0 = d.timing(crpr=False)
        self.assertIsNone(r0.crpr)
        cap = float(T._crpr.D.view(T.P, 2, -1)[torch.as_tensor(T.pclk, device=T.dev)].max())
        for a, b in ((r.setup, r0.setup), (r.hold, r0.hold)):
            ok = torch.isfinite(b)
            self.assertTrue(bool(torch.equal(torch.isfinite(a), ok)))
            self.assertGreaterEqual(float((a[ok] - b[ok]).min()), -1e-4)
            self.assertLessEqual(float((a[ok] - b[ok]).max()), cap + 1e-4)
        r2 = d.timing()
        self.assertTrue(torch.equal(r.setup, r2.setup) and torch.equal(r.hold, r2.hold))
        # the tree's buffers are OpenROAD's inferred clock buffers (ASAP7: BUFx*, not the HB* hold buffers)
        import importlib
        C = importlib.import_module('attar.naive.cts')
        cb = C._clock_buffers(d)
        self.assertTrue(cb and all('BUF' in b.upper() for b in cb))
        used = {d.names['masters'][int(d.raw['c_master'][c])] for c, n in enumerate(d.names['cells'])
                if n.startswith('attar_cts_')}
        self.assertTrue(used <= set(cb))
        self.assertTrue(math.isfinite(st['repeater']['spacing_um']) and st['repeater']['spacing_um'] > 0)

    def test_bisect_span(self):
        import importlib
        C = importlib.import_module('attar.naive.cts')
        rng = np.random.default_rng(0)
        xy = rng.uniform(0, 400, (500, 2))
        groups = C._bisect(xy, np.arange(500), 20, 50.0)
        self.assertEqual(sorted(np.concatenate(groups).tolist()), list(range(500)))
        for g in groups:
            self.assertLessEqual(len(g), 20)
            if len(g) > 1:
                self.assertLessEqual(float((xy[g].max(0) - xy[g].min(0)).sum()), 50.0)


class LevelizeTest(unittest.TestCase):
    def test_cycle_broken_at_latch(self):
        """A cycle a -> b -> c -> a whose c -> a edge is a latch D -> Q arc: that edge is dropped, levels follow the
        rest (OpenSTA's levelization never follows latch D -> Q arcs)."""
        from attar.timing.sta import Timer
        fr = np.array([0, 1, 2]); to = np.array([1, 2, 0])
        lev, dropped = Timer._levelize(3, fr, to, prefer=np.array([False, False, True]))
        self.assertEqual(dropped, 1)
        self.assertEqual(lev.tolist(), [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
