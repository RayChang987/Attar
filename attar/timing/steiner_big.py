"""Large nets of the timer's placement parasitics on the CPU: OpenROAD's estimate-parasitics tree for nets of any size
(pdr::primDijkstra: nearest-neighbor sweep, Prim-Dijkstra, steinerize) and the RC moments attar.timing.sta.tree_elmore
computes on the GPU for nets up to 256 pins. The C++ (attar/timing/csrc/pdtree.cpp) is built with g++ on first use into
~/.cache/attar and loaded with ctypes; available() is False when that fails (the timer then lumps large nets)."""
import ctypes
import hashlib
import os
import subprocess

import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(__file__), 'csrc', 'pdtree.cpp')
_LIB = None
_ERR = None


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
        so = os.path.join(cache, 'pdtree_%s.so' % tag)
        if not os.path.exists(so):
            tmp = so + '.%d.tmp' % os.getpid()
            subprocess.run(['g++', '-O3', '-std=c++17', '-fopenmp', '-shared', '-fPIC', _SRC,
                            '-o', tmp], check=True, capture_output=True, text=True)
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        f = lib.attar_pd_rc
        P = ctypes.c_void_p
        f.argtypes = [ctypes.c_int, P, P, P, P, ctypes.c_int, P, P, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                      ctypes.c_float, P, P, P, ctypes.c_int]
        f.restype = ctypes.c_int
        f = lib.attar_pd_edges
        f.argtypes = [ctypes.c_int, P, P, P, ctypes.c_float, P, P, P, ctypes.c_int]
        f.restype = ctypes.c_int
        _LIB = lib
    except Exception as e:  # noqa: BLE001  (no compiler: the caller falls back)
        _ERR = '%s: %s' % (type(e).__name__, getattr(e, 'stderr', '') or e)
    return _LIB


def available():
    return _build() is not None


def error():
    _build()
    return _ERR


def pd_rc(x, y, ptr, drv, cap, rv, r, c, dbu, alpha=0.3, threads=None):
    """Per net (CSR ptr over pins): x, y pin locations in dbu (int), drv the driver's index within its net, cap (M, V)
    pin capacitances, rv (M,) the pins' via resistances; r, c wire RC per um. Returns (elm (M, V) Elmore delays at the
    pins, pi (E, V, 3) near C / R / far C, length (E,) um), as attar.timing.sta.tree_elmore."""
    lib = _build()
    if lib is None:
        raise RuntimeError('attar.timing.steiner_big unavailable: %s' % _ERR)
    x = np.ascontiguousarray(x, np.int32); y = np.ascontiguousarray(y, np.int32)
    ptr = np.ascontiguousarray(ptr, np.int64); drv = np.ascontiguousarray(drv, np.int64)
    cap = np.ascontiguousarray(cap, np.float32); rv = np.ascontiguousarray(rv, np.float32)
    E, M, V = len(ptr) - 1, len(x), cap.shape[1]
    elm = np.zeros((M, V), np.float32); pi = np.zeros((E, V, 3), np.float32); length = np.zeros(E, np.float32)
    p = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
    lib.attar_pd_rc(E, p(ptr), p(drv), p(x), p(y), V, p(cap), p(rv), float(r), float(c), float(dbu), float(alpha),
                    p(elm), p(pi), p(length), int(threads or torch.get_num_threads() or 1))
    return elm, pi, length


def pd_edges(x, y, ptr, alpha=0.0, threads=None):
    """Per net (CSR ptr over points, x / y int): the edges of OpenROAD's Prim-Dijkstra tree over its points
    (steinerized; alpha 0: a rectilinear minimum spanning tree, steinerized), zero-length ones dropped. Returns
    (edges (K, 4) int32 as x0, y0, x1, y1, net (K,) int64)."""
    lib = _build()
    if lib is None:
        raise RuntimeError('attar.timing.steiner_big unavailable: %s' % _ERR)
    x = np.ascontiguousarray(x, np.int32); y = np.ascontiguousarray(y, np.int32)
    ptr = np.ascontiguousarray(ptr, np.int64)
    E = len(ptr) - 1
    cnt = np.zeros(E, np.int64)
    p = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
    th = int(threads or torch.get_num_threads() or 1)
    lib.attar_pd_edges(E, p(ptr), p(x), p(y), float(alpha), p(cnt), None, None, th)
    off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    out = np.zeros((int(off[-1]), 4), np.int32)
    lib.attar_pd_edges(E, p(ptr), p(x), p(y), float(alpha), p(cnt), p(off), p(out), th)
    return out, np.repeat(np.arange(E, dtype=np.int64), cnt)
