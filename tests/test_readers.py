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
        d = need('after_fp/ispd2015/mgc_des_perf_a')
        self.assertEqual(d.manifest['not_stored'].get('fence_regions'), 4)
        self.assertNotIn('constraints', d.facts)
        self.assertAlmostEqual(d.manifest['target_density'], 0.429)      # placement.constraints: 42.9%

    def test_placed_macros_are_frozen(self):
        d = need('after_fp/ispd2024/ariane133_51')                                  # macros + PLACED in the DEF
        m = d.macros
        self.assertGreater(len(m), 0)
        self.assertTrue(bool(((d.flags[m] & FIXED) != 0).all()))
        self.assertIn('floorplan.macros', d.facts)
        self.assertIn('hpwl', d.capabilities())                            # the routing benchmark comes placed

    def test_asap7_contest_design_has_libraries_but_no_timing_yet(self):
        d = need('after_fp/iccad2025/aes')
        self.assertEqual(d.manifest['fidelity'], 'F1')
        self.assertNotIn('timing', d.facts)
        self.assertIn('sdc', d.manifest['unread_sources'])
        # the contest's aes input has every cell in N, also in FS rows: the rail check sees it
        self.assertGreater(attar.Evaluator(d).legality()['wrong_rail'], 0)
        self.assertTrue(attar.Evaluator(need('after_fp/iccad2025/des')).legality()['legal'])


class PinDirectionTest(unittest.TestCase):
    def test_lef_def_every_net_has_one_driver(self):
        import torch
        from attar.distill import D_IN, D_OUT
        d = need('after_fp/orfs_asap7/gcd')
        self.assertEqual(d.facts['pin_direction'], 'lef+def')
        drv = torch.zeros(len(d.n_weight), dtype=torch.int64, device=d.device)
        drv.index_add_(0, d.p_net, (d.pin_dir == D_OUT).long())
        deg = torch.bincount(d.p_net, minlength=len(d.n_weight))
        self.assertTrue(bool((drv[deg >= 2] == 1).all()))                 # gcd: one driver per net
        io = torch.isin(d.p_cell, d.io)
        port = d.io_dir[torch.searchsorted(d.io, d.p_cell[io])]
        self.assertTrue(bool((d.pin_dir[io][port == 0] == D_OUT).all()))   # an INPUT port drives its net
        self.assertTrue(bool((d.pin_dir[io][port == 1] == D_IN).all()))

    def test_lef_attributes_on_the_pin_line(self):
        import os
        import tempfile
        from attar.lef import read_lef
        lef = ('MACRO X\n  CLASS CORE ;\n  SIZE 1 BY 1 ;\n  PIN o DIRECTION OUTPUT ;\n    PORT\n      LAYER M1 ;\n'
               '      RECT 0 0 0.1 0.1 ;\n    END\n  END o\n  PIN a\n    DIRECTION INPUT ;\n    USE SIGNAL ;\n'
               '  END a\n  PIN VDD DIRECTION INOUT ; USE POWER ;\n  END VDD\nEND X\n')
        with tempfile.TemporaryDirectory() as t:
            with open(os.path.join(t, 'x.lef'), 'w') as f:
                f.write(lef)
            m = read_lef([os.path.join(t, 'x.lef')])[0]['X']
        self.assertEqual(m.pin_dir, {'o': 'OUTPUT', 'a': 'INPUT', 'VDD': 'INOUT'})   # ISPD 2015 writes them inline
        self.assertEqual(m.pins['VDD'][1], 'POWER')

    def test_net_use_among_other_attributes(self):
        import io
        from attar.deffile import _net_attrs, _read_nets
        for text, use in (('- a ( x A ) ( y B ) + SOURCE TIMING + USE CLOCK ;', 'CLOCK'),   # Innovus writes these inline
                          ('- a\n  ( x A )\n  + ROUTED M1 ( 1 2 ) ( 3 * )\n  + USE CLOCK\n ;', 'CLOCK'),
                          ('- a ( x A ) + ROUTED M1 ( 1 2 ) ( 3 * ) ;', 'SIGNAL')):
            n = []
            _read_nets(io.StringIO(text + '\nEND NETS\n'), n)
            self.assertEqual(n[0][2], use, text)
        self.assertEqual(_net_attrs('- a ( x A ) + SOURCE TIMING + ROUTED M1 ( 1 2 ) NEW M2 ( 1 2 ) + USE CLOCK ;'),
                         ' + SOURCE TIMING + USE CLOCK')                 # the wiring is dropped, the rest kept

    def test_pg_geometry_from_vias_and_special_nets(self):
        from attar.deffile import PG_HWIRE, PG_VIA, def_vias, pg_shapes
        text = ('VIAS 1 ;\n- V01 + VIARULE V01R + CUTSIZE 80 80 + LAYERS M0 VIA0 M1 + CUTSPACING 80 80\n'
                ' + ENCLOSURE 40 24 40 24 + ROWCOL 1 5 ;\nEND VIAS\n'
                'SPECIALNETS 1 ;\n- VDD ( * VDD )\n  + ROUTED M0 96 + SHAPE FOLLOWPIN ( 0 1000 ) ( 5000 * )\n'
                '    NEW M0 96 + SHAPE COREWIRE ( 2000 1000 ) V01\n  + USE POWER ;\nEND SPECIALNETS\n')
        self.assertEqual(def_vias(text)['V01'], [('M0', (-400, -64, 400, 64)), ('M1', (-400, -64, 400, 64))])
        sh = pg_shapes(text, {}, 2000)
        self.assertIn(('M0', (-48, 952, 5048, 1048), PG_HWIRE, 'VDD'), sh)       # square ends: half a width out
        self.assertIn(('M1', (1600, 936, 2400, 1064), PG_VIA, 'VDD'), sh)

    def test_bookshelf_states_directions_and_blocks_do_not(self):
        b = need('after_fp/ispd2005/adaptec1')
        self.assertEqual(b.facts['pin_direction'], 'bookshelf')
        self.assertGreater(b.manifest['pin_direction']['drivers'], 0)
        for name in ('before_fp/mcnc/ami33', 'before_fp/gsrc_hard/n10'):                     # every pin 'B': no direction stated
            m = need(name)
            self.assertNotIn('pin_direction', m.facts)
            self.assertIn('pin_direction', m.missing('sta'))


class BookshelfTest(unittest.TestCase):
    def test_old_scl_and_irrational_block_sizes(self):
        d = need('before_fp/ispd2006_hb/ibm01')                                     # Siteorient : N; sqrt(area) soft blocks
        self.assertEqual(d.dbu, 1000)
        self.assertGreater(d.manifest['not_stored']['soft_blocks'], 0)
        self.assertNotIn('floorplan.macros', d.facts)                      # movable hard blocks: macro placement
        self.assertIn('macro_place', d.capabilities())

    def test_pads_outside_the_rows_are_io(self):
        d = need('after_fp/gsrc_place/p1Unit_AR1.0_WS02')
        self.assertGreater(len(d.io), 0)
        self.assertLess(float(d.utilization()[0]), 1.0)


class FloorplanFormatTest(unittest.TestCase):
    def test_yal_blocks_get_placed_inside_the_outline(self):
        d = need('before_fp/mcnc/ami33')
        self.assertEqual(d.missing('global_place'), ['floorplan.macros'])
        b = d.die_box[0]; p = d.io_pos
        edge = torch.stack([p[:, 0] - b[0], b[2] - p[:, 0], p[:, 1] - b[1], b[3] - p[:, 1]], 1).min(1).values
        self.assertLess(float(edge.abs().max()), 1e-3)                     # pads on the parent outline
        d.floorplan(verbose=False)
        self.assertEqual(_overlaps(d), 0)
        m = d.macros
        self.assertTrue(bool((d.pos[m] >= b[:2] - 1e-3).all() and (d.pos[m] + d.size[m] <= b[2:] + 1e-3).all()))

    def test_hard_blocks_without_a_die_get_one(self):
        d = need('before_fp/gsrc_hard/n100')
        self.assertNotIn('floorplan.die', d.facts)
        self.assertEqual(d.capabilities(), ['die_size', 'wirelength_model'])
        d.floorplan(utilization=0.8, verbose=False)
        self.assertEqual(_overlaps(d), 0)
        self.assertIn('hpwl', d.capabilities())

    def test_soft_blocks_have_no_geometry(self):
        d = need('before_fp/gsrc_soft/n10')
        self.assertNotIn('cell_geometry', d.facts)
        self.assertEqual(d.manifest['not_stored']['soft_blocks'], 10)
        self.assertEqual(d.raw['soft_area'].shape[0], len(d.flags))


if __name__ == '__main__':
    unittest.main()
