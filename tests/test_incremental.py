"""Incremental re-timing (attar.timing.incremental): after cells move, re-timing only the cone of the nets whose
parasitics changed gives a full run's result bit for bit, with the sweep on the CPU (CpuSweep) and on the GPU (GPU
timers only: the CPU path always runs in full). After a netlist edit the timer follows in place (attar.timing.inplace)
and gives what a new timer gives, bit for bit."""
import unittest

import numpy as np
import torch

from common import MEDIUM, attar
from attar.timing.sta import Timer


def _gpu(name):
    if not torch.cuda.is_available():
        raise unittest.SkipTest('no GPU')
    try:
        return attar.load(name, 'cuda', download=False)
    except FileNotFoundError as e:
        raise unittest.SkipTest(str(e)) from None


class IncrementalTimingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.d = _gpu(MEDIUM)
        cls.d.place(verbose=False)
        cls.d.timing(); cls.d.timing()                  # the second run replays the CUDA graph
        cls.T = cls.d._timer

    def _state(self, r):
        T = self.T
        out = [r.setup, r.hold, T.amax, T.amin, T.smax, T.smin, T._wb[0], T._wb[1], T.load, T.load_min]
        out += [x for rc in T._rec for x in rc]
        return [x.clone() for x in out]

    def _move(self, k, seed):
        g = torch.Generator().manual_seed(seed)
        mov = torch.nonzero(self.d.movable).flatten().cpu()
        cells = mov[torch.randperm(len(mov), generator=g)[:k]].to('cuda')
        p = self.d.pos.clone()
        p[cells, 0] += float(self.d.row_site_w[0]) * torch.randint(-4, 5, (k,), generator=g).float().to('cuda')
        self.d.pos = p

    def _cone_equals_a_full_run(self, sweep):
        T = self.T
        inc = T._inc_get()
        self.assertTrue(inc.ok, inc.why)
        if sweep == 'gpu':
            inc.cpu = None
        else:
            inc.__dict__.pop('cpu', None)
            inc.runs = 1                                # (the CPU sweep is made on an index's second re-timing)
            self.assertIsNotNone(inc._cpu_get())
        used = 0
        for k, seed in ((1, 1), (3, 2), (10, 3)):
            self._move(k, seed)
            T._inc_t = None                             # take the incremental path whatever the timings say
            a = self._state(self.d.timing())
            used += inc.last.get('entries', 0) > 0 and not inc.last.get('full') and \
                bool(inc.last.get('cpu')) == (sweep == 'cpu')
            T._inc_full = True
            b = self._state(self.d.timing())
            for x, y in zip(a, b):
                self.assertTrue(bool(((x == y) | (torch.isnan(x) & torch.isnan(y))).all()))
        self.assertGreater(used, 0)
        self.assertLess(inc.last['entries'], inc.n_ent)
        inc.__dict__.pop('cpu', None)

    def test_a_cone_equals_a_full_run_cpu_sweep(self):
        self._cone_equals_a_full_run('cpu')

    def test_a_cone_equals_a_full_run_gpu_sweep(self):
        self._cone_equals_a_full_run('gpu')

    def test_nothing_moved_changes_nothing(self):
        T = self.T
        a = self._state(self.d.timing())
        T._inc_t = None
        b = self._state(self.d.timing())
        for x, y in zip(a, b):
            self.assertTrue(bool(((x == y) | (torch.isnan(x) & torch.isnan(y))).all()))

    def test_z_a_resize_runs_in_full(self):           # (last: a resize can rebuild the timer)
        d, T = self.d, self.T
        d.timing()
        cm = d.raw['c_master']
        used = {d.names['masters'][int(m)] for m in set(cm[torch.nonzero(d.movable).flatten().cpu().numpy()].tolist())}
        for c in torch.nonzero(d.movable).flatten().tolist():
            cur = d.names['masters'][int(cm[c])]
            other = [m for m in d.equivalents(c) if m != cur and m in used]
            if other:
                break
        else:
            self.skipTest('no swap within the masters the design uses')
        d.resize([c], [other[0]])
        if d._timer is not T:
            self.skipTest('the resize rebuilt the timer')
        self.assertTrue(T._inc_full)
        d.timing()
        self.assertFalse(T._inc_full)


def _same(test, a, b):
    for x, y in zip(a, b):
        test.assertEqual(x.shape, y.shape)
        test.assertTrue(bool(((x == y) | (torch.isnan(x) & torch.isnan(y))).all()))


def _data_net(d, e, rng):
    """A net with 3+ pins a cell drives, off the clock network and not constant."""
    T, a = d._timer, d.raw
    clk = e.clock_nets()
    nptr = a['n_ptr']
    for k in rng.permutation(len(nptr) - 1).tolist():
        dr = T.net_driver[k]
        if k in clk or nptr[k + 1] - nptr[k] < 3 or dr < 0 or a['p_mpin'][dr] < 0 or T.const_net[k] >= 0:
            continue
        if T.clk_of[nptr[k]:nptr[k + 1]].max() < 0:
            return k
    raise unittest.SkipTest('no data net')


class InPlaceEditTest(unittest.TestCase):
    """insert_buffer, clone, and a buffer removed: the timer is not rebuilt, and its timing (the cone re-timed on the
    GPU, or a full run) is a new timer's."""

    def _check(self, d):
        T = d._timer
        self.assertEqual(d._timer_edit, '')
        r1 = d.timing()
        self.assertIs(d._timer, T)
        st = lambda r, T: [r.setup, r.hold, T.amax, T.amin, T.smax, T.smin, T.load, T.load_min]  # noqa: E731
        a = [x.clone() for x in st(r1, T)]
        F = Timer(d)
        b = st(F.run(), F)
        _same(self, a, b)
        T._inc_full = True                              # and a full run of the updated timer
        _same(self, [x.clone() for x in st(d.timing(), T)], b)
        self.assertTrue(np.array_equal(T.level, F.level))
        F.release('compared')

    def _edits(self, dev, n=3):
        try:
            d = attar.load(MEDIUM, dev, download=False).restore('reference')
        except FileNotFoundError as e:
            raise unittest.SkipTest(str(e)) from None
        d.timing(); d.timing()
        rng = np.random.default_rng(3)
        with d.edit() as e:
            k = _data_net(d, e, rng)
            loads = [p for p in sorted(e._net_pins(k)) if d.raw['p_dir'][p] != 1]
            e.insert_buffer(k, loads[:max(1, len(loads) // 2)])
        self._check(d)
        if n == 1:
            return
        with d.edit() as e:
            k = _data_net(d, e, rng)
            c = int(d.raw['p_cell'][d._timer.net_driver[k]])
            loads = [p for p in sorted(e._net_pins(k)) if d.raw['p_dir'][p] != 1]
            e.clone(c, loads[:1])
        self._check(d)
        b = len(d.names['cells']) - 2                   # the buffer of the first edit goes again
        pn = d.p_net.cpu().numpy()
        with d.edit() as e:
            ps = e.cell_pins(b)
            ni = [int(pn[p]) for p in ps if d.raw['p_dir'][p] != 1][0]
            no = [int(pn[p]) for p in ps if d.raw['p_dir'][p] == 1][0]
            e.remove_cell(b)
            e.move([p for p in e._net_pins(no) if p not in set(ps.tolist())], ni)
        self._check(d)

    def test_gpu(self):
        if not torch.cuda.is_available():
            self.skipTest('no GPU')
        self._edits('cuda')

    def test_cpu(self):                                 # (the device changes nothing else: one edit)
        self._edits('cpu', n=1)

    def test_a_clock_net_rebuilds(self):
        try:
            d = attar.load(MEDIUM, 'cpu', download=False)
        except FileNotFoundError as e:
            raise unittest.SkipTest(str(e)) from None
        d.timing()
        T = d._timer
        reg = int(T.k_fr[0])                            # a register clock pin: its cell's other pins are data
        c = int(d.raw['p_cell'][reg])
        from attar.netlist import _single_input
        with d.edit() as e:
            buf = e.default_buffer()
            a_, y_, _ = _single_input(d, buf)
            nb = e.add_cell(buf, d.pos[c].tolist())
            n2 = e.add_net()
            e.connect(nb, a_, int(d.p_net[reg]))
            e.connect(nb, y_, n2)
            e.move([reg], n2)
        self.assertNotEqual(d._timer_edit, '')
        self.assertIsNone(d._timer)
