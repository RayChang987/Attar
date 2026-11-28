"""Incremental re-timing (attar.timing.incremental): after cells move, re-timing only the cone of the nets whose
parasitics changed gives a full run's result bit for bit. GPU only (the CPU path always runs in full)."""
import unittest

import torch

from common import MEDIUM, attar


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

    def test_a_cone_equals_a_full_run(self):
        T = self.T
        inc = T._inc_get()
        self.assertTrue(inc.ok, inc.why)
        used = 0
        for k, seed in ((1, 1), (3, 2), (10, 3)):
            self._move(k, seed)
            T._inc_t = None                             # take the incremental path whatever the timings say
            a = self._state(self.d.timing())
            used += inc.last.get('entries', 0) > 0 and not inc.last.get('full')
            T._inc_full = True
            b = self._state(self.d.timing())
            for x, y in zip(a, b):
                self.assertTrue(bool(((x == y) | (torch.isnan(x) & torch.isnan(y))).all()))
        self.assertGreater(used, 0)
        self.assertLess(inc.last['entries'], inc.n_ent)

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
