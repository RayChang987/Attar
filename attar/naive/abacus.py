"""Abacus legalization of one height class (attar/naive/csrc/abacus.cpp; naive.legalize(method='abacus')). The C++ is
built with g++ on first use into ~/.cache/attar and loaded with ctypes, as attar.timing.steiner_big."""
import ctypes
import hashlib
import os
import subprocess

import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(__file__), 'csrc', 'abacus.cpp')
_LIB = None
_ERR = None


def _build():
    global _LIB, _ERR
    if _LIB is not None or _ERR is not None:
        return _LIB
    try:
        with open(_SRC, 'rb') as f:
            tag = hashlib.sha1(f.read()).hexdigest()[:12]
        cache = os.path.join(os.path.expanduser('~'), '.cache', 'attar')        # as paths.zoo_dir: no environment
        os.makedirs(cache, exist_ok=True)
        so = os.path.join(cache, 'abacus_%s.so' % tag)
        if not os.path.exists(so):
            tmp = so + '.%d.tmp' % os.getpid()
            subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', _SRC, '-o', tmp], check=True,
                           capture_output=True, text=True)
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        P = ctypes.c_void_p
        lib.attar_abacus.argtypes = [ctypes.c_int, P, P, P, ctypes.c_int, P, P, ctypes.c_int, P, P, P, ctypes.c_double,
                                     P, P, P]
        lib.attar_abacus.restype = ctypes.c_int
        _LIB = lib
    except Exception as e:  # noqa: BLE001
        _ERR = '%s: %s' % (type(e).__name__, getattr(e, 'stderr', '') or e)
    return _LIB


def available():
    return _build() is not None


def error():
    _build()
    return _ERR


def solve_class(gx, gy, w, band_y, roff, seg_row, s0, s1, sw):
    """One height class on its bands (lg._solve_class's inputs): gx target x (global sites), gy target bottom y (um),
    w widths (sites); bands at band_y (ascending) with x origin roff (global sites); free segments (seg_row = band,
    [s0, s1) band sites). Returns (band, x in band sites, unfit) on gx's device."""
    lib = _build()
    if lib is None:
        raise RuntimeError("naive.legalize(method='abacus') needs g++: %s" % _ERR)
    dev = gx.device
    c = lambda t, dt: np.ascontiguousarray(t.detach().cpu().numpy().astype(dt))  # noqa: E731
    gx_, gy_, w_ = c(gx, np.float64), c(gy, np.float64), c(w, np.float64)
    by, ro = c(band_y, np.float64), c(roff, np.float64)
    sr, a, b = c(seg_row, np.int64), c(s0, np.float64), c(s1, np.float64)
    n = len(gx_)
    band = np.zeros(n, np.int64); x = np.zeros(n, np.float64); unfit = np.zeros(n, np.uint8)
    p = lambda v: v.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
    lib.attar_abacus(n, p(gx_), p(gy_), p(w_), len(by), p(by), p(ro), len(sr), p(sr), p(a), p(b), float(sw),
                     p(band), p(x), p(unfit))
    return (torch.as_tensor(band, device=dev), torch.as_tensor(x, device=dev),
            torch.as_tensor(unfit.astype(bool), device=dev))
