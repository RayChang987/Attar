"""Naive engines: global placement, legalization, rail parity."""
import tempfile  # noqa: F401
import unittest

import torch  # noqa: F401

from common import DEV, SMALL, MEDIUM, MACRO_DESIGNS, BOOKSHELF, need, attar, source_def, source_lefs


class NaiveTest(unittest.TestCase):
    def test_pava_pack_matches_sequential_abacus(self):
        from attar.naive.lg import _pack
        import random
        random.seed(3)
        for util in (0.3, 0.7, 0.98):
            for _ in range(20):
                cap = random.randint(50, 300); ws = []
                while sum(ws) < util * cap:
                    ws.append(random.randint(1, 8))
                ws = ws[:-1] or [1]
                xs = sorted(random.uniform(-5, cap + 5) for _ in ws)
                pre = [sum(ws[:i]) for i in range(len(ws))]
                st = []
                for v in [xs[i] - pre[i] for i in range(len(xs))]:
                    st.append([v, 1])
                    while len(st) > 1 and st[-2][0] / st[-2][1] > st[-1][0] / st[-1][1]:
                        a = st.pop(); st[-1][0] += a[0]; st[-1][1] += a[1]
                z = [s_ / c for s_, c in st for _ in range(c)]
                ref = [min(max(round(v), 0), cap - sum(ws)) + pre[i] for i, v in enumerate(z)]
                out = _pack(torch.zeros(len(ws), dtype=torch.long), torch.tensor(xs, dtype=torch.float64),
                            torch.tensor(ws, dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64),
                            torch.tensor([float(cap)], dtype=torch.float64))
                self.assertEqual(out.tolist(), [float(v) for v in ref])

    def test_global_place_then_legalize_is_legal(self):
        from attar import naive
        for name in (SMALL, MACRO_DESIGNS[0]):
            d = need(name)
            self.assertNotIn('hpwl', d.capabilities())
            r = naive.global_place(d, verbose=False)
            self.assertIn('legalize', d.capabilities())                # placement is now a fact
            self.assertLess(r['overflow'], 0.1)
            naive.legalize(d, verbose=False)
            self.assertTrue(attar.Evaluator(d).legality()['legal'], name)
            self.assertIn('naive_lg', d.snapshots)

    def test_multi_row_height_cells_are_legalized(self):
        from attar import naive
        try:
            d = attar.load('after_fp/ns3k/serv', DEV, download=False)               # NS3K: 2-row flip-flops among 1-row cells
        except FileNotFoundError:
            raise unittest.SkipTest('no ns3k zoo')
        naive.global_place(d, verbose=False)
        st = naive.legalize(d, verbose=False)
        self.assertTrue(any(str(c).startswith('2') for c in st['classes']))
        self.assertTrue(attar.Evaluator(d).legality()['legal'])

class RailTest(unittest.TestCase):
    def test_rails_follow_openroad_rules(self):
        from attar import rails as RL
        d = need(MEDIUM)                                          # ASAP7: VSS at the bottom of single-row cells
        I = RL.info(d)
        self.assertTrue(I['known'])
        self.assertEqual(I['ref'], (RL.VSS, RL.VDD))
        _, ys, _, _, _, orient = RL.base_rows(d)
        n_rows = orient == 0
        self.assertTrue(bool((I['row_bot'][n_rows] == RL.VSS).all()))      # R0 rows: reference rails
        self.assertTrue(bool((I['row_bot'][~n_rows] == RL.VDD).all()))     # flipped rows: swapped
        b = need(BOOKSHELF)
        self.assertFalse(RL.info(b)['known'])                     # Bookshelf: no rails, no constraint

    def test_even_height_cells_need_matching_rows(self):
        from attar import rails as RL
        rb = torch.tensor([RL.VDD, RL.VSS, RL.VDD, RL.VSS]); rt = torch.tensor([RL.VSS, RL.VDD, RL.VSS, RL.VDD])
        keep, flip = RL.compatible(torch.tensor([RL.VDD]), torch.tensor([RL.VDD]), torch.tensor([2]), rb, rt)
        self.assertEqual((keep | flip)[0].tolist(), [True, False, True, False])   # VDD/VDD 2-row cell: VDD rows only
        keep, flip = RL.compatible(torch.tensor([RL.VSS]), torch.tensor([RL.VDD]), torch.tensor([1]), rb, rt)
        self.assertEqual((keep | flip)[0].tolist(), [True, True, True, True])      # single-row: anywhere, flipped

    def test_checker_catches_wrong_parity_and_lg_respects_it(self):
        from attar import naive
        try:
            d = attar.load('after_fp/ns3k/aes', DEV, download=False)
        except FileNotFoundError:
            raise unittest.SkipTest('no ns3k zoo')
        naive.global_place(d, verbose=False)
        naive.legalize(d, verbose=False)
        lg = attar.Evaluator(d).legality()
        self.assertTrue(lg['legal'], lg)
        hb = d.row_h.min()
        two = torch.nonzero(d.movable & ((d.size[:, 1] / hb).round() == 2)).flatten()[:20]
        d.pos[two, 1] += hb                                        # one row up: wrong rails
        self.assertEqual(attar.Evaluator(d).legality()['wrong_rail'], 20)

    def test_rows_spanned_counts_partial_overlaps(self):
        from attar.rails import rows_spanned
        ys = torch.arange(10, dtype=torch.float32) * 0.27
        lo, hi = rows_spanned(ys, 0.27, torch.tensor([0.27 * 2 + 0.2]), torch.tensor([0.27 * 5 + 0.01]))
        self.assertEqual((lo.item(), hi.item()), (2, 6))           # bottom and top rows partly covered count
        y0 = torch.tensor([0.27 * 3], dtype=torch.float32)
        lo, hi = rows_spanned(ys, 0.27, y0, y0 + 0.54)            # an on-grid 2-row cell: exactly 2 rows
        self.assertEqual((lo.item(), hi.item()), (3, 5))



class FloorplanTest(unittest.TestCase):
    def test_lp_macro_legalizer_removes_overlaps_left_by_push(self):
        from attar import naive
        from attar.naive.floorplan import _overlapping
        d = need('before_fp/gsrc_hard/n50')            # at 0.8, push leaves 5 overlapping pairs here
        naive.make_die(d, 0.8, verbose=False)
        naive.place_io(d, mode='order', verbose=False)
        r = naive.place_macros(d, legalizer='lp', verbose=False)
        mac = d.macros; p = d.pos[mac]; core = d.core_box[0]
        self.assertTrue(r['legalizer'].startswith('lp on'), r['legalizer'])
        self.assertEqual(int(_overlapping(p, d.size[mac]).sum()), 0)
        self.assertTrue(bool(((p >= core[:2] - 1e-3) & (p + d.size[mac] <= core[2:] + 1e-3)).all()))

    def test_macro_placement_never_overlaps_and_floorplan_legal_says_when_it_does_not_fit(self):
        from attar import naive
        from attar.naive.floorplan import _overlapping
        for packing in (False, True):
            for util in (0.85, 0.97):                 # 0.97: blocks this size cannot all fit
                d = need('before_fp/gsrc_hard/n30')
                self.assertEqual(d.floorplan_legal(), (False, ['no die']))
                r = d.floorplan(utilization=util, packing=packing, packing_args={'moves': 50}, verbose=False)['macros']
                mac = d.macros; p = d.pos[mac]; core = d.core_box[0]
                self.assertEqual(int(_overlapping(p, d.size[mac]).sum()), 0)
                out = int(((p < core[:2] - 1e-3) | (p + d.size[mac] > core[2:] + 1e-3)).any(1).sum())
                self.assertEqual((r['outside_core'], r['fits_outline']), (out, out == 0))
                ok, issues = d.floorplan_legal()
                self.assertEqual(ok, out == 0, issues)
                self.assertIn('floorplan', d.status(show=False))
            self.assertFalse(r['fits_outline'])

    def test_packing_fits_where_gp_and_lp_do_not_and_needs_a_block_level_design(self):
        from attar import naive
        d = need('before_fp/gsrc_hard/n30')
        naive.make_die(d, 0.85, verbose=False)
        naive.place_io(d, mode='order', verbose=False)
        r = naive.place_macros(d, packing=True, verbose=False)
        self.assertTrue(r['fits_outline'])
        self.assertEqual(r['overlapping_pairs'], 0)
        self.assertTrue(d.floorplan_legal()[0])
        with self.assertRaises(ValueError):             # rows: packing is for block-level floorplans
            naive.place_macros(need('before_fp/orfs_asap7/tinyRocket'), packing=True)

    def test_b_star_tree_packs_without_overlaps(self):
        import random
        import numpy as np
        from attar.naive.bstar import shelf
        rng = random.Random(1)
        kx = np.array([rng.randint(1, 9) for _ in range(40)]); ky = np.array([rng.randint(1, 9) for _ in range(40)])
        t = shelf(kx, ky, 30)
        for _ in range(300):
            t2 = t.copy()
            if (t2.swap(rng) if rng.random() < 0.5 else t2.move(rng)):
                t = t2
            x, y = t.pack(kx, ky)
            sep = (x[:, None] >= x[None] + kx[None]) | (x[None] >= x[:, None] + kx[:, None]) | \
                  (y[:, None] >= y[None] + ky[None]) | (y[None] >= y[:, None] + ky[:, None])
            np.fill_diagonal(sep, True)
            self.assertTrue(sep.all())
            seen = sorted(int(v) for v in t.blk)
            self.assertEqual(seen, list(range(40)))

    def test_io_placement_unlocks_global_placement(self):
        from attar import naive
        try:
            d = attar.load('before_fp/orfs_asap7/aes', DEV, download=False)        # ORFS floorplan: IO pins unplaced
        except FileNotFoundError:
            raise unittest.SkipTest('no before_fp zoo')
        self.assertEqual(d.missing('global_place'), ['floorplan.io'])  # aes has no macros: only the IO pins are missing
        naive.place_io(d, verbose=False)
        self.assertIn('global_place', d.capabilities())
        self.assertIn('export', d.capabilities())                # export writes the IO pins
        b = d.die_box[0]
        p = d.io_pos
        edge = torch.stack([p[:, 0] - b[0], b[2] - p[:, 0], p[:, 1] - b[1], b[3] - p[:, 1]], 1).min(1).values
        self.assertLess(float(edge.abs().max()), 1e-3)             # every pin on the boundary
        o = torch.argsort(p[:, 0] * 1e4 + p[:, 1])
        q = p[o]
        same = (q[1:] - q[:-1]).abs().sum(1) < 1e-6
        self.assertFalse(bool(same.any()))                          # no two pins on one spot

    def test_macro_placement_then_cells_is_legal(self):
        from attar import naive
        try:
            d = attar.load('before_fp/orfs_asap7/riscv32i', DEV, download=False)         # ORFS floorplan: IO pins and macros unplaced
        except FileNotFoundError:
            raise unittest.SkipTest('no 2_1 zoo')
        self.assertEqual(d.capabilities(), ['die_size', 'export', 'io_place', 'plot', 'resize', 'sta', 'wirelength_model'])
        naive.place_io(d, verbose=False)
        self.assertIn('macro_place', d.capabilities())
        self.assertNotIn('global_place', d.capabilities())          # macros still unplaced
        r = naive.place_macros(d, verbose=False)
        self.assertEqual(r['overlapping_pairs'], 0)
        self.assertIn('global_place', d.capabilities())
        naive.global_place(d, verbose=False)
        naive.legalize(d, verbose=False)
        self.assertTrue(attar.Evaluator(d).legality()['legal'])
        m = d.macros
        core = d.core_box[0]
        self.assertTrue(bool((d.pos[m] >= core[:2] - 1e-3).all() and (d.pos[m] + d.size[m] <= core[2:] + 1e-3).all()))

    def test_after_fp_design_keeps_its_die(self):
        d = need(SMALL)                                  # gcd after ORFS GP: die, rows and PDN are the problem
        self.assertNotIn('die_size', d.capabilities())
        self.assertIn('the floorplan is given', d.blocked('die_size'))
        from attar.facts import MissingFacts
        with self.assertRaises(MissingFacts):
            d.floorplan(utilization=0.5, verbose=False)

    def test_new_die_for_a_utilization_then_place_is_legal(self):
        d = need('before_fp/orfs_asap7/aes')             # ORFS floorplan (2_1): tap cells placed, IO pins not
        d.floorplan(utilization=0.5, aspect_ratio=1.5, verbose=False)
        self.assertAlmostEqual(float(d.utilization()[0]), 0.5, delta=0.02)
        core = d.core_box[0]
        self.assertAlmostEqual(float((core[3] - core[1]) / (core[2] - core[0])), 1.5, delta=0.1)
        self.assertTrue(d.facts['floorplan.die'].startswith('naive:make_die'))
        self.assertIn('floorplan.io', d.facts); self.assertIn('floorplan.macros', d.facts)
        self.assertNotIn('placement', d.present)          # the floorplan leaves the cells to the placer
        self.assertIn('export', d.capabilities())         # export rewrites DIEAREA, ROWs, TRACKS
        self.assertIn('orfs_floorplan_handoff', d.capabilities())   # a new die still goes back to ORFS 2_1
        self.assertEqual(d.snapshots, {})
        r = d.place(verbose=False)
        self.assertTrue(r['legal'])                       # legality includes rail parity on the new rows
        b = d.die_box[0]; p = d.io_pos
        edge = torch.stack([p[:, 0] - b[0], b[2] - p[:, 0], p[:, 1] - b[1], b[3] - p[:, 1]], 1).min(1).values
        self.assertLess(float(edge.abs().max()), 1e-3)

    def test_naive_floorplan_writes_back_and_redistills(self):
        try:
            d = attar.load('before_fp/orfs_asap7/riscv32i', DEV, download=False)
        except FileNotFoundError:
            raise unittest.SkipTest('no 2_1 zoo')
        d.floorplan(utilization=0.5, verbose=False)
        d.place(verbose=False)
        self.assertIn('slow_loop', d.facts)                     # ORFS 2_1: the floorplan can be handed off
        self.assertIn('orfs_floorplan_handoff', d.capabilities())
        lefs = source_lefs(d)
        with tempfile.TemporaryDirectory() as t:
            r = d.export(t + '/fp.def')                         # die, rows, IO pins, macros onto the frame
            self.assertEqual((r['pins'], r['die']), (len(d.io), 1))
            attar.distill(t + '/fp.def', lefs, t + '/ir')
            e = attar.load(t + '/ir', DEV, download=False)
        self.assertEqual(e.manifest['io_unplaced'], 0); self.assertEqual(e.manifest['macros_unplaced'], 0)
        self.assertTrue(torch.allclose(e.die_box, d.die_box, atol=1e-3))
        self.assertEqual(len(e.row_h), len(d.row_h))
        self.assertAlmostEqual(e.hpwl().item(), d.hpwl().item(), delta=1e-3 * d.hpwl().item())   # pins sit 1/2 shape inside
        self.assertTrue(attar.Evaluator(e).legality()['legal'])

    def test_macro_pins_land_on_their_tracks(self):
        from attar import naive
        from attar.lef import read_lef
        try:
            d = attar.load('before_fp/orfs_asap7/riscv32i', DEV, download=False)
        except FileNotFoundError:
            raise unittest.SkipTest('no 2_1 zoo')
        naive.place_io(d, verbose=False)
        r = naive.place_macros(d, verbose=False)
        lefs, _ = read_lef(source_lefs(d))
        self.assertEqual(r['grid'], 'tracks')
        start, step = d.tracks()[('Y', 'M4')]
        for i in d.macros.tolist():
            M = lefs[d.names['masters'][d.raw['c_master'][i]]]
            y = round(float(d.pos[i, 1]) * d.dbu)
            for pn, (b, use) in M.pins.items():
                if use == 'SIGNAL' and M.pin_layer.get(pn) == 'M4':      # horizontal: on a Y track
                    self.assertEqual((y + round((b[1] + b[3]) / 2 * d.dbu) - start) % step, 0, pn)

    def test_sweep_line_macro_legalization(self):
        from attar import naive
        d = need('before_fp/mcnc/ami49')
        r = naive.place_macros(d, legalizer='sweep', verbose=False)
        self.assertEqual(r['overlapping_pairs'], 0)
        self.assertTrue(r['legalizer'].startswith('sweep'))
        m = d.macros; b = d.core_box[0]
        self.assertTrue(bool((d.pos[m] >= b[:2] - 1e-3).all() and (d.pos[m] + d.size[m] <= b[2:] + 1e-3).all()))

    def test_floorplan_places_only_what_is_missing(self):
        d = need(SMALL)                                  # IO pins and macros come placed with the source
        io0 = d.io_pos.clone()
        d.floorplan(verbose=False)
        self.assertTrue(torch.equal(d.io_pos, io0))
        self.assertIn('export', d.capabilities())

    def test_pins_keep_clear_of_the_pdn(self):
        from attar import naive, pdn
        d = need('after_fp/ns3k/aes')                      # NS3K: PG via stacks from the M0 rails cross the M1 pin layer
        self.assertIn('pdn', d.facts)
        naive.global_place(d, verbose=False)
        st = naive.legalize(d, verbose=False)
        dp = st['pdn_dp']
        self.assertGreater(dp['blocked_before'] + dp['mirrored_in_place'] + st['pdn']['conflicts'], 0)   # there was work
        self.assertEqual(st['pdn'].get('left', 0), 0)
        r = attar.Evaluator(d).legality()
        self.assertEqual(r['pdn_blocked'], 0)
        self.assertTrue(r['legal'])
        a = need(MACRO_DESIGNS[0])               # ASAP7: the M1-M2 vias sit inside the M1 rails, nothing blocks M1
        self.assertGreater(len(pdn.pin_blockages(a)), 0)
        a.restore('reference')
        self.assertEqual(int(pdn.conflicts(a).sum()), 0)

    def test_target_density_is_the_flows_own(self):
        d = need(SMALL)                                         # ORFS: the density its own global placement used
        self.assertEqual(d.manifest.get('target_density_source'), 'orfs 3_3 global_place (GPL-0023)')
        r = d.place(verbose=False)
        self.assertEqual(r['target_density'], d.manifest['target_density'])
        self.assertTrue(r['legal'])

    def test_macro_halo_keeps_cells_away(self):
        d = need(MACRO_DESIGNS[0])
        r = d.place(macro_halo=3.0, verbose=False)
        self.assertTrue(r['legal'])
        m = d.macros
        c = d.pos[d.movable]; s = d.size[d.movable]
        m0, m1 = d.pos[m], d.pos[m] + d.size[m]
        gap_x = torch.maximum(m0[None, :, 0] - (c[:, None, 0] + s[:, None, 0]), c[:, None, 0] - m1[None, :, 0])
        gap_y = torch.maximum(m0[None, :, 1] - (c[:, None, 1] + s[:, None, 1]), c[:, None, 1] - m1[None, :, 1])
        self.assertGreaterEqual(float(torch.maximum(gap_x, gap_y).min()), 3.0 - 1e-3)

    def test_routability_driven_global_placement(self):
        from attar import naive
        d = need(MEDIUM)
        gp = naive.global_place(d, routability=True, verbose=False)
        ra = gp['routability']
        self.assertGreater(ra['capacity_h'], 0); self.assertGreater(ra['capacity_v'], 0)
        self.assertGreaterEqual(len(ra['rounds']), 1)               # congestion was looked at
        naive.legalize(d, verbose=False)
        self.assertTrue(attar.Evaluator(d).legality()['legal'])

    def test_pdn_repair_mirrors_only_symmetric_masters(self):
        from attar import naive
        d = need('after_fp/ns3k/aes')
        o0 = d.orient.clone()
        naive.global_place(d, verbose=False)
        st = naive.legalize(d, verbose=False)
        self.assertGreater(st['pdn_dp']['mirrored'] + st['pdn_dp']['mirrored_in_place'] + st['pdn'].get('mirrored', 0), 0)
        flipped = torch.nonzero(torch.isin(d.orient, torch.tensor([1, 4], device=d.device))).flatten().cpu().numpy()
        sym = d.raw['m_sym'][d.raw['c_master'][flipped]]
        self.assertTrue(bool(((sym & 2) != 0).all()))               # FN / S only where the LEF allows SYMMETRY Y
        del o0

    def test_runs_on_a_second_gpu(self):
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest('one GPU')
        d = attar.load(MEDIUM, 'cuda:1', download=False)    # CUDA graphs are captured on the design's GPU
        self.assertTrue(d.place(verbose=False)['legal'])
        self.assertAlmostEqual(attar.Evaluator(d, fast=True).hpwl().item(), d.hpwl().item(), delta=1e-2)

    def test_macro_grid_from_the_ir_alone(self):
        from attar.deffile import read_def
        from attar.lef import read_lef
        done = 0
        for name in ('before_fp/orfs_asap7/riscv32i', 'before_fp/orfs_asap7/tinyRocket'):
            try:
                d = attar.load(name, DEV, download=False)
            except FileNotFoundError:
                continue
            mac = d.macros
            step, off = d.macro_grid(mac)                  # reads the IR only
            self.assertEqual(d.macro_grid_source, 'tracks')
            # independently from the raw LEF and DEF: on each axis, the offset puts as many signal pins of the
            # macro on a preferred-direction track of their layer as any offset can
            lefs, tech = read_lef(source_lefs(d))
            horiz = {l['name']: l.get('direction') == 'HORIZONTAL' for l in tech['layers']}
            tr = {(a, l): (s0, sp) for a, s0, _, sp, l in read_def(source_def(d))['tracks']}
            u = d.dbu
            for k, i in enumerate(mac.tolist()):
                M = lefs[d.names['masters'][d.raw['c_master'][i]]]
                for ax in (0, 1):
                    res = {}
                    for pn, (b, use) in M.pins.items():
                        l = M.pin_layer.get(pn)
                        if use != 'SIGNAL' or l not in horiz or horiz[l] != (ax == 1) or ('XY'[ax], l) not in tr:
                            continue
                        s0, sp = tr[('XY'[ax], l)]
                        c = (round(b[ax] * u) + round(b[ax + 2] * u)) // 2
                        res[(sp, (s0 - c) % sp)] = res.get((sp, (s0 - c) % sp), 0) + 1
                    if not res:
                        continue
                    o = round(float(off[k, ax]) * u)
                    got = sum(n for (sp, r), n in res.items() if (o - r) % sp == 0)
                    self.assertEqual(got, max(res.values()), (name, d.names['cells'][i], ax))
                    done += 1
        if not done:
            raise unittest.SkipTest('no 2_1 zoo')

if __name__ == '__main__':
    unittest.main()
