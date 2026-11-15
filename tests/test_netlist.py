"""Netlist edits (attar.netlist): buffers, clones, nets; the IR stays consistent, timing and export follow."""
import tempfile
import unittest

import numpy as np
import torch

from common import DEV, MEDIUM, need, attar, source_lefs
from attar import naive, netlist
from attar.io.deffile import read_def
from attar.io.distill import D_OUT


def _signal_net(d, e, min_loads=8):
    """A net with at least min_loads sinks, driven by a cell, not a clock net."""
    deg = (d.n_ptr[1:] - d.n_ptr[:-1]).cpu().numpy()
    clk = e.clock_nets()
    for k in np.argsort(-deg):
        k = int(k)
        pins = range(int(d.n_ptr[k]), int(d.n_ptr[k + 1]))
        drv = [p for p in pins if int(d.pin_dir[p]) == D_OUT]
        if k in clk or len(drv) != 1 or d.raw['c_flags'][int(d.raw['p_cell'][drv[0]])] & 8:
            continue
        loads = [p for p in pins if int(d.pin_dir[p]) != D_OUT]
        if len(loads) >= min_loads:
            return k, drv[0], loads
    raise unittest.SkipTest('no suitable net')


def _conn(d, p):
    """(cell, master pin name) of pin p."""
    a = d.raw
    c, m = int(a['p_cell'][p]), int(a['p_mpin'][p])
    mp = [x for ps in d.names['master_pins'] for x in ps]
    return d.names['cells'][c], mp[m] if m >= 0 else None


class NetlistEditTest(unittest.TestCase):
    def test_insert_buffer_splits_a_net(self):
        d = need(MEDIUM).restore('reference')
        r0 = d.timing()
        N, P, E = len(d.names['cells']), len(d.p_cell), len(d.names['nets'])
        e = d.edit()
        k, drv, loads = _signal_net(d, e)
        moved = loads[len(loads) // 2:]
        before = [_conn(d, p) for p in range(P)]
        with e:
            b, n2 = e.insert_buffer(k, moved)
        self.assertEqual(netlist.check(d), [])
        self.assertEqual((b, n2), (N, E))                                       # appended: nothing else moves
        self.assertEqual((len(d.names['cells']), len(d.p_cell), len(d.names['nets'])), (N + 1, P + 2, E + 1))
        self.assertEqual([_conn(d, int(e.pin_map[p])) for p in range(P)], before)   # every old pin, renumbered
        on = lambda n: {_conn(d, p) for p in range(int(d.n_ptr[n]), int(d.n_ptr[n + 1]))}  # noqa: E731
        bname = d.names['cells'][b]
        self.assertEqual(on(n2), {before[p] for p in moved} | {(bname, 'Y')})
        self.assertIn((bname, 'A'), on(k))
        self.assertTrue(on(k).isdisjoint({before[p] for p in moved}))
        self.assertEqual(d.names['masters'][int(d.raw['c_master'][b])], e.default_buffer())
        with self.assertRaises(RuntimeError):                                   # pins renumbered: stale
            r0.path()
        T = d.timing().timer
        pb = [int(e.pin_map[P]), int(e.pin_map[P + 1])]                          # the buffer's A and Y
        self.assertGreater(float(T.amax[pb[1]].max()), float(T.amax[pb[0]].max()))
        naive.legalize(d, verbose=False)                                  # the buffer goes onto the rows
        self.assertTrue(attar.Evaluator(d).legality()['legal'])
        with tempfile.TemporaryDirectory() as t:
            d.export(t + '/buf.def')
            r = read_def(t + '/buf.def')
            self.assertIn((bname, e.default_buffer()), {(c[0], c[1]) for c in r['comps']})
            nets = {n: set(map(tuple, c)) for n, c, _ in r['nets']}
            self.assertEqual(nets[d.names['nets'][n2]], on(n2))
            attar.distill(t + '/buf.def', source_lefs(d), t + '/ir')              # the DEF reads back as the design
            d2 = attar.load(t + '/ir', DEV, download=False)
            self.assertEqual(len(d2.names['cells']), len(d.names['cells']))
            self.assertAlmostEqual(d2.hpwl().item(), d.hpwl().item(), delta=1e-3)

    def test_clone_takes_some_loads(self):
        d = need(MEDIUM).restore('reference')
        e = d.edit()
        k, drv, loads = _signal_net(d, e)
        c0 = int(d.raw['p_cell'][drv])
        ins = {_conn(d, p)[1]: int(d.p_net[p]) for p in np.nonzero(d.raw['p_cell'] == c0)[0] if p != drv}
        with e:
            c, n2 = e.clone(c0, loads[:3])
        self.assertEqual(netlist.check(d), [])
        self.assertEqual(int(d.raw['c_master'][c]), int(d.raw['c_master'][c0]))
        got = {_conn(d, p)[1]: int(d.p_net[p]) for p in np.nonzero(d.raw['p_cell'] == c)[0]}
        out = [p for p in np.nonzero(d.raw['p_cell'] == c)[0] if int(d.pin_dir[p]) == D_OUT]
        self.assertEqual(len(out), 1)
        self.assertEqual(int(d.p_net[out[0]]), n2)
        self.assertEqual({k_: v for k_, v in got.items() if k_ in ins}, ins)    # inputs on the original's nets
        self.assertEqual(int(d.n_ptr[n2 + 1] - d.n_ptr[n2]), 4)                 # its output and the 3 loads
        self.assertEqual(int(d.n_ptr[k + 1] - d.n_ptr[k]), len(loads) + 1 - 3)
        self.assertTrue(np.isfinite(d.timing().wns))

    def test_refusals(self):
        d = need(MEDIUM).restore('reference')
        e = d.edit()
        clk = sorted(e.clock_nets())
        self.assertTrue(clk)
        with self.assertRaises(ValueError):
            e.insert_buffer(clk[0])                                             # clocks are ideal
        k, drv, loads = _signal_net(d, e)
        with self.assertRaises(ValueError):
            e.insert_buffer(k, [drv])                                           # the driver is no load
        elsewhere = next(p for p in range(len(d.p_cell)) if int(d.p_net[p]) != k and int(d.pin_dir[p]) != D_OUT)
        with self.assertRaises(ValueError):
            e.insert_buffer(k, [elsewhere])                                     # a load of another net
        with self.assertRaises(ValueError):
            e.insert_buffer(k, loads[:1], master=d.inverters()[0])             # not a buffer
        e.apply()
        with self.assertRaises(RuntimeError):
            e.add_net()                                                         # an applied edit is closed

    def test_names_and_an_empty_edit(self):
        d = need(MEDIUM).restore('reference')
        r0 = d.timing()
        with d.edit() as e:
            pass
        np.testing.assert_array_equal(e.pin_map, np.arange(len(d.p_cell)))
        self.assertIsNotNone(r0.path())                                         # nothing changed: still valid
        e = d.edit()
        k, drv, loads = _signal_net(d, e)
        name = '%s/%s' % _conn(d, loads[0])
        self.assertEqual(e.pin(name), loads[0])
        with e:
            b, n2 = e.insert_buffer(d.names['nets'][k], [name], name='my_buf')
        self.assertEqual(d.names['cells'][b], 'my_buf')
        self.assertEqual(int(d.n_ptr[n2 + 1] - d.n_ptr[n2]), 2)

    def test_a_moved_port_is_on_its_new_net_in_the_def(self):
        # a DEF reader connects a port by its PINS statement: export rewrites '+ NET' when an edit moved the port
        from attar.io.distill import TERMINAL
        d = need(MEDIUM).restore('reference')
        term = (d.raw['c_flags'] & TERMINAL) != 0
        e = d.edit()
        clk = e.clock_nets()
        pc, pd = d.raw['p_cell'], d.pin_dir.cpu().numpy()
        port = next(p for p in range(len(pc)) if term[pc[p]] and pd[p] != D_OUT and int(d.p_net[p]) not in clk)
        k = int(d.p_net[port])
        with e:
            b, n2 = e.insert_buffer(k, [port])
        name = d.names['cells'][int(pc[port])]
        with tempfile.TemporaryDirectory() as t:
            d.export(t + '/port.def')
            r = read_def(t + '/port.def')
        pins = {x[0]: x[1] for x in r['pins']}
        self.assertEqual(pins[name], d.names['nets'][n2])
        self.assertIn(('PIN', name), {tuple(c) for n, cs, _ in r['nets'] if n == d.names['nets'][n2] for c in cs})


def _snapshot(d):
    """The netlist as (cell, master pin) -> net, per net its driver (cell, master pin), the pins, a memo for _source."""
    a = d.raw
    pc, pm, pdir = a['p_cell'].astype(int), a['p_mpin'].astype(int), a['p_dir'].astype(int)
    pnet = d.p_net.cpu().numpy()
    net_of = {(c, m): k for c, m, k in zip(pc.tolist(), pm.tolist(), pnet.tolist())}
    drv = {k: (c, m) for c, m, k, dr in zip(pc.tolist(), pm.tolist(), pnet.tolist(), pdir.tolist()) if dr == D_OUT}
    return net_of, drv, list(zip(pc.tolist(), pm.tolist(), pdir.tolist())), {}


def _source(snap, d, pin):
    """The logic a sink sees: follow its net back through every buffer and inverter (removed or not) to the first other
    driver (a gate, a register, an IO pin): (that driver (cell, mpin), parity of the inversions on the way)."""
    memo = snap[3]
    k0 = snap[0][pin]
    if k0 not in memo:
        memo[k0] = _trace(snap, d, k0)
    return memo[k0]


_KINDS = {}   # master name -> attar.netlist._single_input


def _trace(snap, d, k):
    from attar.netlist import _single_input
    net_of, drv = snap[0], snap[1]
    par, seen = 0, set()
    while k in drv and k not in seen:
        seen.add(k)
        c, m = drv[k]
        mid = int(d.raw['c_master'][c])
        name = d.names['masters'][mid] if mid >= 0 else None
        if name not in _KINDS:
            _KINDS[name] = _single_input(d, name) if name else None
        s = _KINDS[name]
        if s is None or (c, int(d.raw['m_pin_ptr'][mid]) + d.names['master_pins'][mid].index(s[0])) not in net_of:
            return (c, m), par
        par ^= int(s[2])
        k = net_of[(c, int(d.raw['m_pin_ptr'][mid]) + d.names['master_pins'][mid].index(s[0]))]
    return None, par


class DebufferTest(unittest.TestCase):
    def test_every_load_keeps_its_logic(self):
        from attar.io.distill import REMOVED
        for name in (MEDIUM, 'after_fp/orfs_asap7/cva6'):
            d = need(name).restore('reference')
            old = _snapshot(d)
            n = d.debuffer()
            self.assertEqual(netlist.check(d), [])
            removed = set(np.nonzero(d.raw['c_flags'] & REMOVED)[0].tolist())
            self.assertEqual(len(removed), n['buffers'] + n['inverters'])
            self.assertGreater(n['buffers'], 0)
            new = _snapshot(d)
            checked = 0
            for c, m, dr in new[2]:                                  # every live sink sees the same logic as before
                if dr == D_OUT:
                    continue
                self.assertEqual(_source(new, d, (c, m)), _source(old, d, (c, m)), (name, c, m))
                checked += 1
            self.assertGreater(checked, 1000)
            self.assertTrue(np.isfinite(d.timing().wns))
            with tempfile.TemporaryDirectory() as t:                 # removed cells and emptied nets are not written
                d.export(t + '/deb.def')
                r = read_def(t + '/deb.def')
            comps = {c[0] for c in r['comps']}
            self.assertTrue(all(d.names['cells'][c] not in comps for c in removed))
            self.assertEqual(len(r['nets']), int(((d.n_ptr[1:] - d.n_ptr[:-1]) > 0).sum()))

    def test_removed_cells_are_out_of_the_way(self):
        from attar.io.distill import REMOVED
        d = need(MEDIUM).restore('reference')
        n0 = int(d.movable.sum())
        n = d.debuffer()
        self.assertEqual(int(d.movable.sum()), n0 - n['buffers'] - n['inverters'])
        rm = torch.as_tensor((d.raw['c_flags'] & REMOVED) != 0, device=d.device)
        self.assertFalse(bool(d.placed[rm].any()))
        naive.place(d, gp_args={'max_iters': 300})                  # engines skip the tombstones
        self.assertTrue(attar.Evaluator(d).legality()['legal'])
        with self.assertRaises(ValueError):
            with d.edit() as e:
                e.clone(int(torch.nonzero(rm)[0]), [])               # a removed cell takes no edits


if __name__ == '__main__':
    unittest.main()
