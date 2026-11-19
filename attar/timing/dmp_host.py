"""attar.timing.dmp on the CPU in C++ (attar/timing/csrc/dmp.cpp): OpenSTA's dmp_ceff_elmore gate and load delay, one
element at a time in double precision, each iteration loop stopping when its element has converged or failed, OpenMP
over the elements. The timer uses it on the CPU in place of the PyTorch reference (attar.timing.dmp), as it uses the
Triton kernels (attar.timing.dmp_triton) on the GPU; gate / load have their signatures. Built with g++ on first use into
~/.cache/attar and loaded with ctypes; available() is False when that fails (the timer then keeps the PyTorch path)."""
import ctypes
import hashlib
import os
import subprocess

import numpy as np
import torch

from .dmp import FIND_ROOT_MAX_ITER, NEWTON_MAX_ITER

_SRC = os.path.join(os.path.dirname(__file__), 'csrc', 'dmp.cpp')
_LIB = None
_ERR = None
NSTATE = 17


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
        so = os.path.join(cache, 'dmp_%s.so' % tag)
        if not os.path.exists(so):
            tmp = so + '.%d.tmp' % os.getpid()
            # no FMA contraction: the same rounding as the PyTorch and Triton paths' separate operations
            subprocess.run(['g++', '-O3', '-std=c++17', '-fopenmp', '-ffp-contract=off', '-shared', '-fPIC', _SRC,
                            '-o', tmp], check=True, capture_output=True, text=True)
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        P, I64, I = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
        lib.attar_dmp_gate.argtypes = [I64, P, P, P, P, I64, P, P, P, P, P, P, P, P, P, P, I, I, I]
        lib.attar_dmp_gate.restype = I
        lib.attar_dmp_load.argtypes = [I64, P, P, P, P, P, P, I, I]
        lib.attar_dmp_load.restype = I
        _LIB = lib
    except Exception as e:  # noqa: BLE001  (no compiler: the caller falls back)
        _ERR = '%s: %s' % (type(e).__name__, getattr(e, 'stderr', '') or e)
    return _LIB


def available():
    return _build() is not None


def error():
    _build()
    return _ERR


def _np(t, dt):
    return np.ascontiguousarray(t.detach().cpu().numpy(), dt)


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def _threads():
    """PyTorch's intra-op thread count (the physical cores by default): with as many OpenMP threads as logical CPUs
    the two runtimes' spinning threads contend (a 20x slowdown measured on a 10-core / 20-thread CPU)."""
    return max(1, torch.get_num_threads())


class Tables:
    """The library tables as the C++ reads them (float32 / int64, contiguous), made once per timer."""

    def __init__(self, ix1, ix2, val, tn):
        self.ix1, self.ix2, self.val = _np(ix1, np.float32), _np(ix2, np.float32), _np(val, np.float32)
        self.tn = _np(tn, np.int64)
        self.K = int(self.ix1.shape[-1])


def gate(tables, lib, orf, slew, ctot, c_far, rpi, thr):
    """Same as attar.timing.sta.arc_eval (tables: a Tables): (delay, driver slew, state (N, 17)), float32 tensors."""
    N = len(lib)
    delay = np.empty(N, np.float32); dslew = np.empty(N, np.float32); state = np.empty((N, NSTATE), np.float32)
    if N:
        T = tables
        args = [_np(lib, np.int64), _np(orf, np.int64), _np(slew, np.float32), _np(ctot, np.float32),
                _np(c_far, np.float32), _np(rpi, np.float32), _np(thr, np.float32)]
        _build().attar_dmp_gate(N, _ptr(T.ix1), _ptr(T.ix2), _ptr(T.val), _ptr(T.tn), T.K, *[_ptr(a) for a in args],
                                _ptr(delay), _ptr(dslew), _ptr(state), NEWTON_MAX_ITER, FIND_ROOT_MAX_ITER, _threads())
    return torch.from_numpy(delay), torch.from_numpy(dslew), torch.from_numpy(state)


def load(state, elmore, rise, lthr):
    """Same as attar.timing.sta.load_eval: (wire delay, load slew), float32 tensors."""
    N = len(elmore)
    w = np.empty(N, np.float32); s = np.empty(N, np.float32)
    if N:
        args = [_np(state, np.float32), _np(elmore, np.float32), _np(rise, np.uint8), _np(lthr, np.float32)]
        _build().attar_dmp_load(N, *[_ptr(a) for a in args], _ptr(w), _ptr(s), FIND_ROOT_MAX_ITER, _threads())
    return torch.from_numpy(w), torch.from_numpy(s)
