"""attar.timing.power's expression activities on the CPU in C++ (attar/timing/csrc/activity.cpp): Power::evalActivity's
BDD path per expression entry, as _Exprs.activity (the PyTorch reference, kept on the GPU). The power propagation
evaluates a level's entries at a time, and on the CPU the PyTorch version is dominated by the per-operation cost
(a few hundred small operations per level and register pass). Built with g++ on first use into ~/.cache/attar and
loaded with ctypes; available() is False when that fails (the PyTorch path is kept)."""
import ctypes
import hashlib
import os
import subprocess

import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(__file__), 'csrc', 'activity.cpp')
_LIB = None
_ERR = None


def _threads():
    """PyTorch's intra-op thread count (the physical cores by default): with as many OpenMP threads as logical CPUs
    the two runtimes' spinning threads contend (a 20x slowdown measured on a 10-core / 20-thread CPU)."""
    return max(1, torch.get_num_threads())


def _build():
    global _LIB, _ERR
    if _LIB is not None or _ERR is not None:
        return _LIB
    try:
        with open(_SRC, 'rb') as f:
            src = f.read()
        tag = hashlib.sha1(src).hexdigest()[:12]
        cache = os.path.join(os.path.expanduser('~'), '.cache', 'attar')        # as paths.zoo_dir: no environment
        os.makedirs(cache, exist_ok=True)
        so = os.path.join(cache, 'activity_%s.so' % tag)
        if not os.path.exists(so):
            tmp = so + '.%d.tmp' % os.getpid()
            # no FMA contraction: the same rounding as the PyTorch path's separate operations
            subprocess.run(['g++', '-O3', '-std=c++17', '-fopenmp', '-ffp-contract=off', '-shared', '-fPIC', _SRC,
                            '-o', tmp], check=True, capture_output=True, text=True)
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        P, I64, F, I = ctypes.c_void_p, ctypes.c_int64, ctypes.c_float, ctypes.c_int
        lib.attar_activity.argtypes = [I64, I64, I64] + [P] * 11 + [I64, I64, F, F, F, F, P, P, I]
        lib.attar_activity.restype = I
        lib.attar_activity_sweep.argtypes = [I64] + [P] * 17 + [P] * 11 + [P] * 11 + [I64, I64, F, F, F, F, F, I]
        lib.attar_activity_sweep.restype = I
        _LIB = lib
    except Exception as e:  # noqa: BLE001  (no compiler: the caller falls back)
        _ERR = '%s: %s' % (type(e).__name__, getattr(e, 'stderr', '') or e)
    return _LIB


def available():
    return _build() is not None


def error():
    _build()
    return _ERR


def _c(t, dt):
    t = t if t.dtype == dt else t.to(dt)
    return t.contiguous()


def activity(g, K, rows, act, in_density, var_period, min_period, unit):
    """(duty, density) float32 of the entries rows of expression group g (an _Exprs group: kind / idx (n, W), tid,
    bank (., 2^K)), as _Exprs.activity. act: (pin duty, pin density, state duty, state density, pin launch period,
    pin is a clock pin), CPU tensors."""
    pin_u, pin_d, seq_u, seq_d, per, clk = act
    rows = _c(rows, torch.int64)
    n = len(rows)
    duty = torch.empty(n, dtype=torch.float32); dens = torch.empty(n, dtype=torch.float32)
    if n:
        kind, idx, tid = _c(g['kind'], torch.int8), _c(g['idx'], torch.int32), _c(g['tid'], torch.int32)
        bank = _c(g['bank'], torch.float32)
        f32 = [_c(x, torch.float32) for x in (pin_u, pin_d, seq_u, seq_d, per)]
        ck = _c(clk, torch.bool)
        keep = (rows, kind, idx, tid, bank, *f32, ck)                     # alive through the call
        S = len(seq_u)
        if S == 0:                                                        # (a valid pointer for the empty arrays)
            f32[2] = f32[3] = torch.zeros(1, dtype=torch.float32)
            keep = keep + (f32[2],)
        ptrs = [x.data_ptr() for x in (rows, kind, idx, tid, bank, *f32, ck)]
        err = _build().attar_activity(n, K, kind.shape[1], *ptrs,
                                      len(pin_u), S, float(in_density), float(var_period), float(min_period),
                                      float(unit), duty.data_ptr(), dens.data_ptr(), _threads())
        del keep
        if err:
            raise ValueError('attar_activity: %d variables' % K)
    return duty, dens


class Sweep:
    """Power._graphs' frontier sweep in C++: the level structures flattened once (CPU tensors kept alive here), the
    propagation state the Power's buffers."""

    def __init__(self, pw):
        X, D, B = pw.X, pw._dense_setup(), pw._buffers()
        L64 = lambda xs: torch.as_tensor(np.asarray(xs, np.int64))  # noqa: E731
        cat = lambda xs: torch.cat(xs) if xs else torch.zeros(0, dtype=torch.int64)  # noqa: E731
        ptr = lambda xs: L64(np.concatenate([[0], np.cumsum([len(x) for x in xs])]))  # noqa: E731
        lv = pw.levels
        ld = [Ld['load'][1] for Ld in lv]; dr = [Ld['load_drv'] for Ld in lv]; st = [Ld['state'][1] for Ld in lv]
        fp, fr, Ks, Ws, kd, ix, td, bk = [], [], [], [], [], [], [], []
        self._keep = []
        z64 = torch.zeros(1, dtype=torch.int64); zf = torch.zeros(1, dtype=torch.float32)
        for Ld in lv:
            if Ld['fun']:
                key, _, pins, rows = Ld['fun'][0]
                g = X.groups[key]
                t = [_c(g['kind'], torch.int8), _c(g['idx'], torch.int32), _c(g['tid'], torch.int32),
                     _c(g['bank'], torch.float32)]
                self._keep.append(t)
                fp.append(pins); fr.append(rows); Ks.append(g['K']); Ws.append(t[0].shape[1])
                kd.append(t[0].data_ptr()); ix.append(t[1].data_ptr()); td.append(t[2].data_ptr())
                bk.append(t[3].data_ptr())
            else:
                fp.append(z64[:0]); fr.append(z64[:0]); Ks.append(0); Ws.append(1)
                kd.append(z64.data_ptr()); ix.append(z64.data_ptr()); td.append(z64.data_ptr())
                bk.append(zf.data_ptr())
        es, ed = [e[0] for e in D['edges']], [e[1] for e in D['edges']]
        self.arrays = [ptr(ld), cat(ld), cat(dr), ptr(st), cat(st), ptr(fp), cat(fp), cat(fr), L64(Ks), L64(Ws),
                       L64(kd), L64(ix), L64(td), L64(bk), ptr(es), cat(es), cat(ed)]
        self.arrays = [_c(a, torch.int64) for a in self.arrays]
        self.n_levels = len(lv)
        f32 = lambda x: _c(x, torch.float32)  # noqa: E731
        self.state = [B['dens'], B['duty'], B['orig'], B['fd'], B['fu'], B['sq_d'], B['sq_u'], B['sq_o'], B['enq'],
                      B['regs'], B['changed']]
        for t in self.state:
            assert t.is_contiguous() and t.device.type == 'cpu'
        self.statics = [f32(B['cap']), f32(B['prop_cap']), f32(B['lp']), _c(pw.is_clock, torch.bool),
                        _c(pw.clk_of, torch.int64), f32(pw.clk_period), f32(pw.clk_duty),
                        _c(pw.const_pin, torch.int64), _c(pw.p_cell_t, torch.int64), _c(D['isl'], torch.bool),
                        _c(pw.fn_seq, torch.int64)]
        self.B, self.pw = B, pw
        self.zero = torch.zeros(1, dtype=torch.float32)

    def __call__(self, tol):
        """One sweep (changed cleared first, enq cleared after). cap / prop_cap / lp are re-read from the buffers."""
        pw, B = self.pw, self.B
        for i, k in enumerate(('cap', 'prop_cap', 'lp')):
            if self.statics[i].data_ptr() != B[k].data_ptr():
                self.statics[i] = _c(B[k], torch.float32)
        B['changed'].zero_()
        S = len(B['sq_d'])
        st = list(self.state)
        if S == 0:
            st[5] = st[6] = self.zero
            st[7] = torch.zeros(1, dtype=torch.int64)
        err = _build().attar_activity_sweep(self.n_levels, *[a.data_ptr() for a in self.arrays],
                                            *[t.data_ptr() for t in st], *[t.data_ptr() for t in self.statics],
                                            int(pw.T.P), S, float(tol), float(pw.X.in_density), float(pw.X.min_period),
                                            float(pw.min_period), 1.0, _threads())
        B['enq'].zero_()
        if err:
            raise ValueError('attar_activity_sweep')
