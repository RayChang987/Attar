"""Front-end readers: each hands over only the facts its format has (spec 5.1). Uses the zoo built by
tools/distill_benchmarks.py; a missing design skips its test."""
import unittest

import torch

from common import need, attar  # noqa: F401
from attar.distill import FIXED


def _overlaps(d):
    m = d.macros
    p, s = d.pos[m], d.size[m]
    c = p + s / 2
    ov = ((s[:, None] + s[None]) / 2 - (c[:, None] - c[None]).abs() > 1e-6).all(2)
    ov.fill_diagonal_(False)
    return int(ov.sum()) // 2


class LefDefTest(unittest.TestCase):
    def test_fence_regions_are_counted_not_claimed(self):
        d = need('ispd2015/mgc_des_perf_a')
        self.assertEqual(d.manifest['not_stored'].get('fence_regions'), 4)
        self.assertNotIn('constraints', d.facts)
        self.assertAlmostEqual(d.manifest['target_density'], 0.429)      # placement.constraints: 42.9%

    def test_placed_macros_are_frozen(self):
        d = need('ispd2024/ariane133_51')                                  # macros + PLACED in the DEF
        m = d.macros
        self.assertGreater(len(m), 0)
        self.assertTrue(bool(((d.flags[m] & FIXED) != 0).all()))
        self.assertIn('floorplan.macros', d.facts)
        self.assertIn('hpwl', d.capabilities())                            # the routing benchmark comes placed

    def test_asap7_contest_design_has_libraries_but_no_timing_yet(self):
        d = need('iccad2025/aes')
        self.assertEqual(d.manifest['fidelity'], 'F1')
        self.assertNotIn('timing', d.facts)
        self.assertIn('sdc', d.manifest['unread_sources'])
        # the contest's aes input has every cell in N, also in FS rows: the rail check sees it
        self.assertGreater(attar.Evaluator(d).legality()['wrong_rail'], 0)
        self.assertTrue(attar.Evaluator(need('iccad2025/des')).legality()['legal'])


class BookshelfTest(unittest.TestCase):
    def test_old_scl_and_irrational_block_sizes(self):
        d = need('ispd2006_hb/ibm01')                                     # Siteorient : N; sqrt(area) soft blocks
        self.assertEqual(d.dbu, 1000)
        self.assertGreater(d.manifest['not_stored']['soft_blocks'], 0)
        self.assertNotIn('floorplan.macros', d.facts)                      # movable hard blocks: macro placement
        self.assertIn('macro_place', d.capabilities())

    def test_pads_outside_the_rows_are_io(self):
        d = need('gsrc_place/p1Unit_AR1.0_WS02')
        self.assertGreater(len(d.io), 0)
        self.assertLess(float(d.utilization()[0]), 1.0)


class FloorplanFormatTest(unittest.TestCase):
    def test_yal_blocks_get_placed_inside_the_outline(self):
        d = need('mcnc/ami33')
        self.assertEqual(d.missing('global_place'), ['floorplan.macros'])
        b = d.die_box[0]; p = d.io_pos
        edge = torch.stack([p[:, 0] - b[0], b[2] - p[:, 0], p[:, 1] - b[1], b[3] - p[:, 1]], 1).min(1).values
        self.assertLess(float(edge.abs().max()), 1e-3)                     # pads on the parent outline
        d.floorplan(verbose=False)
        self.assertEqual(_overlaps(d), 0)
        m = d.macros
        self.assertTrue(bool((d.pos[m] >= b[:2] - 1e-3).all() and (d.pos[m] + d.size[m] <= b[2:] + 1e-3).all()))

    def test_hard_blocks_without_a_die_get_one(self):
        d = need('gsrc_hard/n100')
        self.assertNotIn('floorplan.die', d.facts)
        self.assertEqual(d.capabilities(), ['die_size', 'wirelength_model'])
        d.floorplan(utilization=0.8, verbose=False)
        self.assertEqual(_overlaps(d), 0)
        self.assertIn('hpwl', d.capabilities())

    def test_soft_blocks_have_no_geometry(self):
        d = need('gsrc_soft/n10')
        self.assertNotIn('cell_geometry', d.facts)
        self.assertEqual(d.manifest['not_stored']['soft_blocks'], 10)
        self.assertEqual(d.raw['soft_area'].shape[0], len(d.flags))


if __name__ == '__main__':
    unittest.main()
