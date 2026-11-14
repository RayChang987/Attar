"""Design, facts and capabilities, dies, macros, write-back, status, plots."""
import os  # noqa: F401
import tempfile  # noqa: F401
import unittest

import numpy as np  # noqa: F401
import torch  # noqa: F401

from common import DEV, SMALL, MEDIUM, MACRO_DESIGNS, BOOKSHELF, need, gpl_final_hpwl, attar, source_def, source_lefs
from attar import st, _slow_loop  # noqa: F401
from attar.deffile import read_def  # noqa: F401
from attar.design import _orient_offset  # noqa: F401
from attar.distill import FIXED, MACRO, TERMINAL  # noqa: F401
from attar.facts import MissingFacts  # noqa: F401
from attar.lef import read_lef  # noqa: F401


class SafetensorsTest(unittest.TestCase):
    def test_roundtrip(self):
        t = {'i32': np.arange(12, dtype=np.int32).reshape(3, 4), 'u8': np.array([1, 2, 255], np.uint8),
             'f32': np.linspace(0, 1, 5, dtype=np.float32), 'i64': np.array([-1, 2 ** 40], np.int64),
             'empty': np.zeros((0, 4), np.int32)}
        with tempfile.TemporaryDirectory() as d:
            st.save(d + '/x.safetensors', t, {'k': 'v'})
            back, meta = st.load(d + '/x.safetensors')
        self.assertEqual(meta, {'k': 'v'})
        for k in t:
            self.assertEqual(back[k].dtype, t[k].dtype)
            np.testing.assert_array_equal(back[k], t[k])

class OrientationTest(unittest.TestCase):
    def test_offsets_stay_inside_the_placed_box(self):
        # a pin at (px, py) of a w x h master must land inside the oriented box for every orientation
        w, h, px, py = 3.0, 1.0, 0.5, 0.25
        n = 8
        o = torch.arange(n)
        off = _orient_offset(o, torch.full((n,), px), torch.full((n,), py), torch.full((n,), w), torch.full((n,), h))
        for i in range(n):
            rot = i in (2, 3, 6, 7)
            bw, bh = (h, w) if rot else (w, h)
            self.assertTrue(0 <= off[i, 0] <= bw and 0 <= off[i, 1] <= bh, (i, off[i]))
        self.assertEqual(off[0].tolist(), [px, py])            # N
        self.assertEqual(off[1].tolist(), [w - px, h - py])    # S
        self.assertEqual(off[4].tolist(), [w - px, py])        # FN

class DesignTest(unittest.TestCase):
    def test_zoo_listing_and_load(self):
        names = attar.zoo('after_fp/orfs_asap7')
        self.assertIn(SMALL, names)
        d = need(SMALL)
        self.assertTrue(d.has('netlist') and d.has('rows') and d.has('cell_library') and d.has('tracks'))
        self.assertTrue(d.has('timing') and d.has('library') and d.has('routing'))
        self.assertEqual(d.facts['pin_offsets'], 'lef')
        self.assertEqual(d.manifest['fidelity'], 'F1')

    def test_cells_start_unplaced_with_a_reference(self):
        d = need(SMALL)
        self.assertEqual(int((d.pos[d.movable] != 0).sum()), 0)    # engines start without coordinates
        d.restore('reference')
        self.assertGreater(int((d.pos[d.movable] != 0).sum()), 0)

    def test_hpwl_matches_an_independent_numpy_computation(self):
        d = need(MEDIUM).restore('reference')
        src = read_def(source_def(d))
        macros, _ = read_lef(source_lefs(d))
        dbu = src['dbu']
        comp = {c[0]: c for c in src['comps']}
        io = {p[0]: (p[3], p[4]) for p in src['pins']}
        total = 0
        for name, conns, use in src['nets']:
            if use in ('POWER', 'GROUND'):
                continue
            xs, ys = [], []
            for c, p in conns:
                if c == 'PIN':
                    x, y = io[p]
                else:
                    _, m, _, cx, cy, o = comp[c]
                    mac = macros[m]
                    b = mac.pins[p][0]
                    px, py = (b[0] + b[2]) / 2 * dbu, (b[1] + b[3]) / 2 * dbu
                    w, h = mac.w * dbu, mac.h * dbu
                    ox, oy = {'N': (px, py), 'S': (w - px, h - py), 'FN': (w - px, py), 'FS': (px, h - py)}[o]
                    x, y = cx + ox, cy + oy
                xs.append(x); ys.append(y)
            if len(xs) > 1:
                total += max(xs) - min(xs) + max(ys) - min(ys)
        self.assertAlmostEqual(d.hpwl().item(), total / dbu, delta=1e-4 * total / dbu)

    def test_hpwl_accepts_explicit_positions(self):
        d = need(SMALL).restore('reference')
        p = d.pos.clone()
        self.assertAlmostEqual(d.hpwl(p).item(), d.hpwl().item(), places=3)
        p[d.movable] += 1.0  # moving every movable cell by the same amount changes only nets with fixed pins
        self.assertNotAlmostEqual(d.hpwl(p).item(), d.hpwl().item(), places=3)

class CapabilityTest(unittest.TestCase):
    def test_what_a_design_can_do_follows_from_its_facts(self):
        d = need(MEDIUM)                               # unplaced: the ORFS GP is kept as the 'reference' snapshot
        self.assertIn('global_place', d.capabilities())
        self.assertNotIn('hpwl', d.capabilities())
        self.assertEqual(d.missing('hpwl'), ['placement'])
        self.assertEqual(d.missing('global_route'), ['placement'])
        self.assertEqual(d.missing('sta'), [])
        with self.assertRaises(MissingFacts) as e:
            d.hpwl()
        self.assertIn("restore('reference')", str(e.exception))
        d.restore('reference')
        for c in ('hpwl', 'place', 'legalize', 'detailed_place', 'export', 'gr_overflow'):
            self.assertIn(c, d.capabilities())

    def test_explicit_positions_need_no_placement_state(self):
        d = need(MEDIUM)
        ref = attar.load(MEDIUM, DEV, download=False).restore('reference')
        self.assertAlmostEqual(d.hpwl(ref.pos).item(), ref.hpwl().item(), delta=1e-3)
        with tempfile.TemporaryDirectory() as t:       # unplaced: the cells Attar owns go out UNPLACED, the rest as is
            d.export(t + '/fp.def')
            out, src = read_def(t + '/fp.def'), read_def(source_def(d))
        owned = {c[0] for c in src['comps'] if c[2] != 'FIXED'}
        self.assertEqual({c[2] for c in out['comps'] if c[0] in owned}, {'UNPLACED'})
        self.assertEqual(sorted(c for c in out['comps'] if c[0] not in owned), sorted(c for c in src['comps'] if c[0] not in owned))
        self.assertEqual((out['pins'], out['die'], out['rows']), (src['pins'], src['die'], src['rows']))

    def test_missing_facts_are_not_filled_in(self):
        d = need(SMALL)
        d.facts.pop('rows')                            # as if the source had no rows
        self.assertEqual(d.missing('legalize'), ['placement', 'rows'])
        self.assertIn('global_place', d.capabilities())

    def test_snapshots_save_and_restore_solution_state(self):
        d = need(SMALL).restore('reference')
        h0 = d.hpwl().item()
        d.snapshot('mine')
        d.pos[d.movable] += 0.5
        self.assertNotAlmostEqual(d.hpwl().item(), h0, places=3)
        d.restore('mine')
        self.assertAlmostEqual(d.hpwl().item(), h0, places=4)

class DieTest(unittest.TestCase):
    def test_2d_design_is_one_die_matching_the_def(self):
        d = need(MEDIUM)
        src = read_def(source_def(d))
        self.assertEqual(d.num_dies, 1)
        np.testing.assert_allclose(d.die_box[0].cpu().numpy(), [v / src['dbu'] for v in src['die']], atol=1e-4)
        self.assertEqual(int(d.cell_die.max()), 0)
        self.assertEqual(int(d.row_die.max()), 0)
        self.assertEqual(len(d.manifest['dies']), d.num_dies)

    def test_core_and_rows_inside_the_die(self):
        d = need(MEDIUM)
        die, core = d.die_box[0], d.core_box[0]
        self.assertTrue(bool((core[:2] >= die[:2]).all() and (core[2:] <= die[2:]).all()))
        self.assertTrue(bool((d.row_box[:, :2] >= core[:2] - 1e-4).all() and (d.row_box[:, 2:] <= core[2:] + 1e-4).all()))
        u = d.utilization()[0].item()
        self.assertTrue(0.05 < u < 1.0, u)

    def test_io_pins_are_signal_pins_on_the_die_edge(self):
        d = need(MEDIUM)
        self.assertEqual(len(d.io), d.manifest['counts']['terminals'])
        self.assertTrue(bool(((d.flags[d.io] & TERMINAL) != 0).all()))
        self.assertEqual(int((d.io_dir == 2).sum()), 0)        # no INOUT: VDD/VSS are not IO
        b, p = d.die_box[0], d.io_pos
        edge = torch.stack([p[:, 0] - b[0], b[2] - p[:, 0], p[:, 1] - b[1], b[3] - p[:, 1]], 1).min(1).values
        self.assertLess(float(edge.max()), 0.1)

class MacroTest(unittest.TestCase):
    def test_macros_are_fixed_blocks_with_their_pins_inside(self):
        for name in MACRO_DESIGNS:
            d = need(name).restore('reference')
            mac = torch.nonzero((d.flags & MACRO) != 0).flatten()
            self.assertGreater(len(mac), 0, name)
            self.assertTrue(bool(((d.flags[mac] & FIXED) != 0).all()), name)
            on = torch.isin(d.p_cell, mac)
            p = d.pin_pos()[on]; c = d.p_cell[on]
            lo, hi = d.pos[c] - 1e-3, d.pos[c] + d.size[c] + 1e-3
            inside = ((p >= lo) & (p <= hi)).all(1)
            # pins may only stick out where the LEF itself puts them outside the outline (reported at distillation)
            warned = any('outside' in w for w in d.manifest.get('warnings', []))
            if warned:
                self.assertLess(int((~inside).sum()), 0.1 * len(inside), name)
            else:
                self.assertTrue(bool(inside.all()), name)

    def test_hpwl_matches_openroad_global_placement(self):
        for name in MACRO_DESIGNS + [MEDIUM]:
            ref = gpl_final_hpwl(name)
            if ref is None:
                continue
            h = need(name).restore('reference').hpwl().item()
            self.assertLess(abs(h - ref) / ref, 0.01, (name, h, ref))

class VizTest(unittest.TestCase):
    def test_plot_writes_a_png(self):
        from attar.viz import plot
        for name in (SMALL, MACRO_DESIGNS[0]):
            d = need(name).restore('reference')
            with tempfile.TemporaryDirectory() as t:
                plot(d, t + '/x.png')
                self.assertGreater(os.path.getsize(t + '/x.png'), 10000)

    def test_unplaced_design_plots_without_cells_at_the_origin(self):
        from attar.viz import plot
        d = need(SMALL)                      # cells unplaced: drawn as a note, not as rectangles at (0, 0)
        fig = plot(d)
        self.assertTrue(any('unplaced' in x.get_text() for x in fig.axes[0].texts))

    def test_block_level_floorplan_shows_connections_and_legality(self):
        from matplotlib.collections import LineCollection
        from attar.viz import _connections, plot
        import numpy as np
        d = need('before_fp/mcnc/ami33')
        d.floorplan(verbose=False)
        fig = plot(d)
        lines = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)]
        self.assertEqual(len(lines), 1)
        self.assertLessEqual(len(lines[0].get_segments()), 2 * len(d.macros) if 2 * len(d.macros) > 40 else 40)
        self.assertTrue(any(x.get_text().split()[-1:] == ['legal'] and 'not' not in x.get_text() for x in fig.texts))
        fig = plot(d, nets=False)
        self.assertFalse(any(isinstance(c, LineCollection) for c in fig.axes[0].collections))
        nodes = np.zeros(len(d.size), bool); nodes[d.macros.cpu().numpy()] = True
        seg, st = _connections(d, d.pos.cpu().numpy(), d.size.cpu().numpy(), nodes, 10)
        self.assertEqual((len(seg), float(st.max())), (10, 1.0))
        self.assertTrue((np.diff(st) >= 0).all())                     # strongest last

    def test_tiles_gr_overflow_and_density(self):
        import numpy as np
        from attar.viz import tiles_density, tiles_gr_overflow
        d = need(SMALL).restore('reference')
        r = d.gr_overflow()
        m, xe, ye, _ = tiles_gr_overflow(d)
        self.assertEqual(m.shape, (len(ye) - 1, len(xe) - 1))
        self.assertEqual(m.shape, r['gcells'][::-1])
        self.assertAlmostEqual(float(m.max()), r['max_util'], places=6)          # the worst edge, on its gcells
        self.assertEqual((xe[-1], ye[-1]), tuple(d.die_box[0, 2:].tolist()))     # last gcells reach the die edge
        m, xe, ye, _ = tiles_density(d)
        ok = np.isfinite(m)
        self.assertGreater(ok.mean(), 0.5)
        self.assertGreater(float(m[ok].max()), 1.0)                             # ORFS's GP: bins overfull
        d.place(verbose=False)
        m, xe, ye, _ = tiles_density(d)
        self.assertLessEqual(float(np.nanmax(m)), 1.0 + 1e-3)                   # legal: no bin overfull
        with tempfile.TemporaryDirectory() as t:
            for k in ('gr_overflow', 'density'):
                d.plot(t + '/%s.png' % k, tiles=k)
                self.assertGreater(os.path.getsize(t + '/%s.png' % k), 10000)
        with self.assertRaises(ValueError):
            d.plot(tiles='thermal')

    def test_tiles_pin_access_is_the_dp_penalty(self):
        import numpy as np
        from attar import pdn as PDN
        from attar.facts import MissingFacts
        from attar.distill import ASIDE, TERMINAL
        from attar.naive.lg import W_PA
        from attar.viz import tiles_pin_access
        d = need('after_fp/ns3k/gcd')
        d.place(verbose=False)
        pos = d.pos.clone(); pos[:, 0] += 0.5 * float(d.row_site_w.min())     # off the DP's spots: pins hit the PDN
        n = int((PDN.conflicts(d, pos=pos) & ((d.flags & (TERMINAL | ASIDE)) == 0)).sum())
        self.assertGreater(n, 0)
        m, _, _, _ = tiles_pin_access(d, pos)
        self.assertAlmostEqual(float(np.nansum(m)), W_PA * n)
        self.assertEqual(float(np.nansum(tiles_pin_access(d)[0])), 0.0)           # after the DP: nothing blocked
        with self.assertRaises(MissingFacts):
            need(BOOKSHELF).plot(tiles='pin_access')                            # no PDN

class WritebackTest(unittest.TestCase):
    def test_unchanged_reference_exports_the_source_design(self):
        from attar.deffile import same_design
        for name in (SMALL, MACRO_DESIGNS[0]):          # no template: the frame and the IR are the whole design
            d = need(name).restore('reference')
            with tempfile.TemporaryDirectory() as t:
                d.export(t + '/out.def')
                self.assertEqual(same_design(read_def(source_def(d)), read_def(t + '/out.def')), '', name)

    def test_moved_and_flipped_cells_survive_a_redistill(self):
        d = need(SMALL).restore('reference')
        g = torch.Generator().manual_seed(1)
        mv = torch.nonzero(d.movable).flatten().cpu()
        pick = mv[torch.randperm(len(mv), generator=g)[:40]].to(d.device)
        site = float(d.raw['r_site_w'][0]) / d.dbu
        d.pos[pick, 0] += 2 * site
        d.orient[pick] = torch.tensor([0, 1, 4, 5], device=d.device)[torch.randint(0, 4, (len(pick),), generator=g).to(d.device)]
        d.update_orient()
        lefs = source_lefs(d)
        with tempfile.TemporaryDirectory() as t:
            d.export(t + '/moved.def')
            attar.distill(t + '/moved.def', lefs, t + '/ir')
            e = attar.load(t + '/ir', DEV, download=False)
            self.assertAlmostEqual(e.hpwl().item(), d.hpwl().item(), delta=1e-3)
            np.testing.assert_array_equal(e.orient.cpu().numpy(), d.orient.cpu().numpy())

    def test_fixed_cells_and_terminals_are_not_written(self):
        d = need(SMALL).restore('reference')
        fixed = ((d.flags & (FIXED | TERMINAL)) != 0)
        d.pos[fixed] += 5.0  # must be ignored by export
        from attar.deffile import same_design
        with tempfile.TemporaryDirectory() as t:
            d.export(t + '/out.def')
            self.assertEqual(same_design(read_def(source_def(d)), read_def(t + '/out.def')), '')

    def test_frame_keeps_the_pdn_and_drops_stale_wiring(self):
        import gzip
        d = need(SMALL)
        with gzip.open(os.path.join(d.path, 'frame.def.gz'), 'rt') as f:
            frame = f.read()
        self.assertIn('SPECIALNETS', frame)                           # the PDN is the problem: kept verbatim
        body = frame[frame.index('\nNETS ') + 1:frame.index('END NETS')].split(';')[1:]
        self.assertTrue(all('USE POWER' in x or 'USE GROUND' in x for x in body if x.strip().startswith('-')))
        with open(source_def(d)) as f:
            src = f.read()
        sp = src[src.index('SPECIALNETS'):src.index('END SPECIALNETS')]
        with tempfile.TemporaryDirectory() as t:
            d.restore('reference').export(t + '/out.def')
            with open(t + '/out.def') as f:
                self.assertIn(sp, f.read())                         # byte for byte

    def test_export_needs_a_cell_library_and_a_frame(self):
        b = need(BOOKSHELF)
        self.assertEqual(b.missing('export'), ['cell_library', 'def_frame'])
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(MissingFacts):
                b.export(t + '/x.def')
            self.assertFalse(os.path.exists(t + '/x.def'))

class StatusTest(unittest.TestCase):
    def test_status_checklist(self):
        d = need(MEDIUM)
        s = d.status(show=False)
        self.assertIn('\u2705 netlist', s)                       # ✅
        self.assertIn('\u2705 timing', s)
        self.assertIn('\u274c constraints', s)                   # ❌
        self.assertIn('\u274c placement', s)
        self.assertIn('only a placement is missing', s)
        d.restore('reference')
        s = d.status(show=False)
        self.assertIn('\u2705 placement', s)
        self.assertIn('ready: d.export', s)

    def test_slow_loop_stays_out_of_the_interface(self):
        """The hand-off to ORFS is Attar's own validation (attar._slow_loop): users get a DEF, nothing about ORFS."""
        d = need(MEDIUM)
        text = d.status(show=False) + d.summary() + repr(d)
        for word in ('slow_loop', 'slow loop', 'hand-off', 'handoff', 'tier', 'orfs_'):
            self.assertNotIn(word, text)
        self.assertFalse(hasattr(d, 'handoff_command') or hasattr(d, 'tier'))
        self.assertEqual(_slow_loop.tier(d), 'A')
        with self.assertRaises(KeyError):
            d.require('orfs_handoff')
        d.restore('reference')
        self.assertIn('orfs_roundtrip.py', _slow_loop.handoff_command(d, 'x.def'))

    def test_slow_loop_tier_b(self):
        try:
            n = attar.load('after_fp/ns3k/aes', DEV, download=False)            # NS3K: DEF write-back, no slow loop here
        except FileNotFoundError:
            raise unittest.SkipTest('no ns3k zoo')
        self.assertEqual(_slow_loop.tier(n), 'B')
        self.assertIn('slow_loop', _slow_loop.missing(n))
        with self.assertRaises(MissingFacts):
            _slow_loop.handoff_command(n, 'x.def')

    def test_plot_method(self):
        d = need(SMALL).restore('reference')
        with tempfile.TemporaryDirectory() as t:
            d.plot(t + '/p.png')
            self.assertGreater(os.path.getsize(t + '/p.png'), 10000)


if __name__ == '__main__':
    unittest.main()
