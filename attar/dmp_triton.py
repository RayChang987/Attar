"""attar.dmp as two Triton kernels: one thread lane per arc (gate) or per load (wire), every iteration in registers, one
launch per batch. The math is attar.dmp's, which follows OpenSTA's DmpCeff.cc line by line; attar.dmp stays the
reference (CPU, tests). float32 like the rest of the GPU timer.
"""
import torch
import triton
import triton.language as tl

from .dmp import FIND_ROOT_MAX_ITER, NEWTON_MAX_ITER

ROOT_N = FIND_ROOT_MAX_ITER
NSTATE = 17


@triton.jit
def _exp2(x):
    y = 1.0 + x / 4096.0
    for _ in tl.static_range(12):
        y = y * y
    return tl.where(x < -12.0, 0.0, y)


@triton.jit
def _nldm(ix1, ix2, val, nn, base, x, y, mask, K):
    """attar.sta.nldm for one table per lane: base = (arc * 4 + slot); bilinear, linear extrapolation."""
    n1 = tl.load(nn + base * 2, mask=mask, other=1)
    n2 = tl.load(nn + base * 2 + 1, mask=mask, other=1)
    i0 = tl.zeros_like(n1)
    j0 = tl.zeros_like(n1)
    for j in range(1, K - 1):
        a = tl.load(ix1 + base * K + j, mask=mask, other=0.0)
        b = tl.load(ix2 + base * K + j, mask=mask, other=0.0)
        i0 += ((x >= a) & (j < n1 - 1)).to(tl.int32)
        j0 += ((y >= b) & (j < n2 - 1)).to(tl.int32)
    i1 = tl.minimum(i0 + 1, tl.maximum(n1 - 1, 0))
    j1 = tl.minimum(j0 + 1, tl.maximum(n2 - 1, 0))
    a0 = tl.load(ix1 + base * K + i0, mask=mask, other=0.0); a1 = tl.load(ix1 + base * K + i1, mask=mask, other=1.0)
    b0 = tl.load(ix2 + base * K + j0, mask=mask, other=0.0); b1 = tl.load(ix2 + base * K + j1, mask=mask, other=1.0)
    da = a1 - a0
    db = b1 - b0
    tx = tl.where((i1 > i0) & (da != 0), (x - a0) / tl.where(da != 0, da, 1.0), 0.0)
    ty = tl.where((j1 > j0) & (db != 0), (y - b0) / tl.where(db != 0, db, 1.0), 0.0)
    vb = val + base * K * K
    v00 = tl.load(vb + i0 * K + j0, mask=mask, other=0.0); v10 = tl.load(vb + i1 * K + j0, mask=mask, other=0.0)
    v01 = tl.load(vb + i0 * K + j1, mask=mask, other=0.0); v11 = tl.load(vb + i1 * K + j1, mask=mask, other=0.0)
    return v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty) + v01 * (1 - tx) * ty + v11 * tx * ty


@triton.jit
def _y0(t, cl, rd):
    return t - rd * cl * (1.0 - _exp2(-t / (rd * cl)))


@triton.jit
def _y0dt(t, cl, rd):
    return 1.0 - _exp2(-t / (rd * cl))


@triton.jit
def _y0dcl(t, cl, rd):
    return rd * ((1.0 + t / (rd * cl)) * _exp2(-t / (rd * cl)) - 1)


@triton.jit
def _y(t, t0, dt, cl, rd):
    t1 = t - t0
    v = tl.where(t1 <= dt, _y0(t1, cl, rd) / dt, (_y0(t1, cl, rd) - _y0(t1 - dt, cl, rd)) / dt)
    return tl.where(t1 <= 0, 0.0, v)


@triton.jit
def _dy(t, t0, dt, cl, rd):
    t1 = t - t0
    r = t1 <= dt
    pos = t1 > 0
    a_t0 = -_y0dt(t1, cl, rd) / dt
    a_dt = -_y0(t1, cl, rd) / (dt * dt)
    a_cl = _y0dcl(t1, cl, rd) / dt
    b_t0 = -(_y0dt(t1, cl, rd) - _y0dt(t1 - dt, cl, rd)) / dt
    b_dt = -(_y0(t1, cl, rd) + _y0(t1 - dt, cl, rd)) / (dt * dt) + _y0dt(t1 - dt, cl, rd) / dt
    b_cl = (_y0dcl(t1, cl, rd) - _y0dcl(t1 - dt, cl, rd)) / dt
    return (tl.where(pos, tl.where(r, a_t0, b_t0), 0.0), tl.where(pos, tl.where(r, a_dt, b_dt), 0.0),
            tl.where(pos, tl.where(r, a_cl, b_cl), 0.0))


@triton.jit
def _gate_delays(ix1, ix2, val, nn, bd, bs, slew, ce, vth, vl, vh, der, mask, K):
    tv = _nldm(ix1, ix2, val, nn, bd, slew, ce, mask, K)
    s = _nldm(ix1, ix2, val, nn, bs, slew, ce, mask, K) * der
    return tv, tv - s * (vth - vl) / (vh - vl), s


@triton.jit
def _newton(ix1, ix2, val, nn, bd, bs, slew, ce0, rd, ctot, vth, vl, vh, der, A, B, D, p1, p2, mask,
            ORDER3: tl.constexpr, K, ITERS: tl.constexpr):
    """DmpAlg::findDriverParams + newtonRaphson (attar.dmp.newton_step), all iterations in registers."""
    tv, tl_, s = _gate_delays(ix1, ix2, val, nn, bd, bs, slew, ce0, vth, vl, vh, der, mask, K)
    dt = s / (vh - vl)
    t0 = tv + tl.log(1.0 - vth) * rd * ce0 - vth * dt
    ce = ce0
    conv = ce0 != ce0                                         # all False
    fail = conv | ~mask
    t0, dt, ce, conv, fail = _newton_iter(ix1, ix2, val, nn, bd, bs, slew, rd, ctot, vth, vl, vh, der, A, B, D, p1, p2,
                                          mask, t0, dt, ce, conv, fail, ORDER3, K, ITERS)
    # OpenSTA iterates in double. A lane that used up the iterations in float32 without failing (bad step, singular,
    # non-finite) is at float32's resolution of the 1% relative tolerance (e.g. t0 near 0): it continues in float64
    # from where it is, as OpenSTA's double iteration would converge there.
    more = mask & ~conv & ~fail
    if tl.max(more.to(tl.int32), 0) > 0:
        F = tl.float64
        t0d, dtd, ced, convd, faild = _newton_iter(
            ix1, ix2, val, nn, bd, bs, slew.to(F), rd.to(F), ctot.to(F), vth.to(F), vl.to(F), vh.to(F), der.to(F),
            A.to(F), B.to(F), D.to(F), p1.to(F), p2.to(F), more, t0.to(F), dt.to(F), ce.to(F), ~more, ~more,
            ORDER3, K, ITERS)
        t0 = tl.where(more, t0d.to(tl.float32), t0)
        dt = tl.where(more, dtd.to(tl.float32), dt)
        ce = tl.where(more, ced.to(tl.float32), ce)
        conv = conv | (more & convd)
        fail = fail | (more & faild)
    fail = fail | ~conv | (dt <= 0) | (ce < 0) | (ce > ctot + 1e-6)
    return t0, dt, ce, fail


@triton.jit
def _newton_iter(ix1, ix2, val, nn, bd, bs, slew, rd, ctot, vth, vl, vh, der, A, B, D, p1, p2, mask,
                 t0, dt, ce, conv, fail, ORDER3: tl.constexpr, K, ITERS: tl.constexpr):
    """DmpAlg::newtonRaphson's iterations from (t0, dt, ce), in the inputs' precision; converged / failed lanes stay."""
    it = 0
    while (it < ITERS) & (tl.max((~conv & ~fail).to(tl.int32), 0) > 0):  # until every lane converged or failed
        it += 1
        if ORDER3:
            bad = (ce < 0) | (ce > ctot + 1e-6) | (dt <= 0)
        else:
            bad = conv & False
        ce_s = tl.maximum(ce, 1e-9)
        tv, tl_, s = _gate_delays(ix1, ix2, val, nn, bd, bs, slew, ce_s, vth, vl, vh, der, mask, K)
        bad = bad | (s == 0)
        if ORDER3:
            dt_s = tl.where(dt <= 0, 1.0, dt)
        else:
            dt_s = tl.where(dt <= 0, (tl_ - tv) / 100, dt)
            dt = dt_s
        y50 = _y(tv, t0, dt_s, ce_s, rd) - vth
        y20 = _y(tl_, t0, dt_s, ce_s, rd) - vl
        a, b, c = _dy(tl_, t0, dt_s, ce_s, rd)
        d_, e, g = _dy(tv, t0, dt_s, ce_s, rd)
        if ORDER3:
            ct = tl.minimum(s / (vh - vl), 1.4 * dt_s)
            e1 = _exp2(-p1 * ct); e2 = _exp2(-p2 * ct); erc = _exp2(-ct / (rd * ce_s))
            ipi = (A * ct + (B / p1) * (1.0 - e1) + (D / p2) * (1.0 - e2)) / (rd * ct * dt_s)
            ice = (rd * ce_s * ct - (rd * ce_s) * (rd * ce_s) * (1.0 - erc)) / (rd * ct * dt_s)
            fi = ipi - ice
            e1d = _exp2(-p1 * dt_s); e2d = _exp2(-p2 * dt_s); erd = _exp2(-dt_s / (rd * ce_s))
            i_ = (-A * dt_s + B * dt_s * e1d - (2 * B / p1) * (1.0 - e1d) + D * dt_s * e2d - (2 * D / p2) * (1.0 - e2d)
                  + rd * ce_s * (dt_s + dt_s * erd - 2 * rd * ce_s * (1.0 - erd))) / (rd * dt_s * dt_s * dt_s)
            k = (2 * rd * ce_s - dt_s - (2 * rd * ce_s + dt_s) * erd) / (dt_s * dt_s)
            h = 0.0
            f0 = -y20; f1 = -y50; f2 = -fi
            C0 = e * k - g * i_; C1 = -(d_ * k - g * h); C2 = d_ * i_ - e * h
            det = a * C0 + b * C1 + c * C2
            sing = det == 0
            det = tl.where(sing, 1.0, det)
            q0 = (C0 * f0 - (b * k - c * i_) * f1 + (b * g - c * e) * f2) / det
            q1 = (C1 * f0 + (a * k - c * h) * f1 - (a * g - c * d_) * f2) / det
            q2 = (C2 * f0 - (a * i_ - b * h) * f1 + (a * e - b * d_) * f2) / det
            under = (tl.abs(q0) <= tl.abs(t0) * 0.01) & (tl.abs(q1) <= tl.abs(dt) * 0.01) & (tl.abs(q2) <= tl.abs(ce) * 0.01)
        else:
            det = a * e - b * d_
            sing = det == 0
            det = tl.where(sing, 1.0, det)
            q0 = (-y20 * e + y50 * b) / det
            q1 = (y20 * d_ - y50 * a) / det
            q2 = 0.0 * q0
            under = (tl.abs(q0) <= tl.abs(t0) * 0.01) & (tl.abs(q1) <= tl.abs(dt) * 0.01)
        act = ~conv & ~fail
        finite = (q0 == q0) & (q1 == q1) & (q2 == q2) & (tl.abs(q0) < 1e30) & (tl.abs(q1) < 1e30) & (tl.abs(q2) < 1e30)
        fail = fail | (act & (bad | sing | ~finite))
        act = act & ~fail
        t0 = tl.where(act, t0 + q0, t0)
        dt = tl.where(act, dt + q1, dt)
        ce = tl.where(act, ce + q2, ce)
        conv = conv | (act & under)
    return t0, dt, ce, conv, fail


@triton.jit
def _V0(t, k0, k1, k2, k3, k4, p1, p2):
    e1 = _exp2(-p1 * t); e2 = _exp2(-p2 * t)
    return k0 * (k1 + k2 * t + k3 * e1 + k4 * e2), k0 * (k2 - k3 * p1 * e1 - k4 * p2 * e2)


@triton.jit
def _Vl0(t, k0, k1, k2, k3, k4, p1, p2, p3):
    D1 = k0 * (k1 - k2 / p3)
    D3 = -p3 * k0 * k3 / (p1 - p3)
    nz = k4 != 0
    D4 = tl.where(nz, -p3 * k0 * k4 / (p2 - p3), 0.0)
    D5 = k0 * (k2 / p3 - k1 + p3 * k3 / (p1 - p3)) + tl.where(nz, k0 * p3 * k4 / (p2 - p3), 0.0)
    e1 = _exp2(-p1 * t); e2 = _exp2(-p2 * t); e3 = _exp2(-p3 * t)
    return D1 + t + D3 * e1 + D4 * e2 + D5 * e3, 1.0 - D3 * p1 * e1 - D4 * p2 * e2 - D5 * p3 * e3


@triton.jit
def _wave(t, k0, k1, k2, k3, k4, p1, p2, t0, dt, p3, LOAD: tl.constexpr):
    t1 = t - t0
    if LOAD:
        v1, d1 = _Vl0(t1, k0, k1, k2, k3, k4, p1, p2, p3)
        v2, d2 = _Vl0(t1 - dt, k0, k1, k2, k3, k4, p1, p2, p3)
    else:
        v1, d1 = _V0(t1, k0, k1, k2, k3, k4, p1, p2)
        v2, d2 = _V0(t1 - dt, k0, k1, k2, k3, k4, p1, p2)
    r = t1 <= dt
    v = tl.where(t1 <= 0, 0.0, tl.where(r, v1 / dt, (v1 - v2) / dt))
    d = tl.where(t1 <= 0, 0.0, tl.where(r, d1 / dt, (d1 - d2) / dt))
    return v, d


@triton.jit
def _find_root(x1, x2, target, k0, k1, k2, k3, k4, p1, p2, t0, dt, p3, on, LOAD: tl.constexpr, ITERS: tl.constexpr):
    """FindRoot.cc findRoot (attar.dmp.find_root) per lane; lanes with on False do not iterate (their result is
    unused). Returns (root, failed)."""
    y1, _d1 = _wave(x1, k0, k1, k2, k3, k4, p1, p2, t0, dt, p3, LOAD)
    y2, _d2 = _wave(x2, k0, k1, k2, k3, k4, p1, p2, t0, dt, p3, LOAD)
    y1 = y1 - target
    y2 = y2 - target
    bad = ((y1 > 0) & (y2 > 0)) | ((y1 < 0) & (y2 < 0))
    exact = (y1 == 0) | (y2 == 0)
    sw = y1 > 0
    a = tl.where(sw, x2, x1)
    b = tl.where(sw, x1, x2)
    root = (a + b) * 0.5
    root = tl.where(exact & ~bad, tl.where(y1 == 0, x1, x2), root)
    dx_prev = tl.abs(b - a)
    dx = dx_prev
    y, dy = _wave(root, k0, k1, k2, k3, k4, p1, p2, t0, dt, p3, LOAD)
    y = y - target
    conv = exact | bad | ~on
    it = 0
    while (it < ITERS) & (tl.max((~conv).to(tl.int32), 0) > 0):
        it += 1
        bis = ((((root - b) * dy - y) * ((root - a) * dy - y)) > 0) | (tl.abs(2.0 * y) > tl.abs(dx_prev * dy))
        nd = tl.where(bis, (b - a) * 0.5, y / tl.where(dy == 0, 1.0, dy))
        nr = tl.where(bis, a + nd, root - nd)
        act = ~conv
        dx_prev = tl.where(act, dx, dx_prev)
        dx = tl.where(act, nd, dx)
        root = tl.where(act, nr, root)
        conv = conv | (act & (tl.abs(dx) <= 0.01 * tl.abs(root)))
        y2, dy2 = _wave(root, k0, k1, k2, k3, k4, p1, p2, t0, dt, p3, LOAD)
        y2 = y2 - target
        act = ~conv
        a = tl.where(act & (y2 < 0), root, a)
        b = tl.where(act & (y2 >= 0), root, b)
        y = tl.where(act, y2, y)
        dy = tl.where(act, dy2, dy)
    return root, bad | ~conv


@triton.jit
def _gate_dev(ix1, ix2, val, nn, lib, orf, slew, ctot, c1, rpi, vth, vl, vh, der, mask,
              K, NIT: tl.constexpr, RIT: tl.constexpr):
    """attar.dmp.gate for one lane per arc: (delay, driver slew, valid, k0..k4, p1, p2, t0, dt, vo_delay, ub)."""
    c1 = tl.minimum(c1, ctot)
    c2 = tl.maximum(ctot - c1, 0.0)
    bd = lib * 4 + orf
    bs = lib * 4 + 2 + orf
    # gateModelRd and setCeffAlgorithm
    d1 = _nldm(ix1, ix2, val, nn, bd, slew, ctot, mask, K)
    d2 = _nldm(ix1, ix2, val, nn, bd, slew, ctot + 1.0, mask, K)
    rd = -tl.log(vth) * tl.abs(d1 - d2)
    cap_alg = (rd < 1e-5) | (rpi < rd * 1e-3) | (c1 == 0) | (c1 < c2 * 1e-3) | (rpi == 0)
    zc2 = ~cap_alg & (c2 < c1 * 1e-3)
    pia = ~cap_alg & ~zc2
    rd_ = tl.where(cap_alg, 1.0, rd)
    rpi_ = tl.where(cap_alg, 1.0, rpi)
    c1_ = tl.where(cap_alg | (c1 <= 0), 1.0, c1)
    c2_ = tl.where(pia, c2, 1.0)
    z1 = 1.0 / (rpi_ * c1_)
    a = rpi_ * rd_ * c1_ * c2_
    b = rd_ * (c1_ + c2_) + rpi_ * c1_
    sq = tl.sqrt(tl.maximum(b * b - 4 * a, 0.0))
    p1p = (b + sq) / (2 * a)
    p2p = (b - sq) / (2 * a)
    p1p2 = p1p * p2p
    k2p = z1 / p1p2
    k1p = (1.0 - k2p * (p1p + p2p)) / p1p2
    k4p = (k1p * p1p + k2p) / (p2p - p1p)
    k3p = -k1p - k4p
    zz = (c1_ + c2_) / (rpi_ * c1_ * c2_)
    A = zz / p1p2
    B = (zz - p1p) / (p1p * (p1p - p2p))
    D = (zz - p2p) / (p2p * (p2p - p1p))
    p1z = 1.0 / (c1_ * (rd_ + rpi_))
    k0z = p1z / z1
    k1z = (p1z - z1) / (p1z * p1z)
    k0 = tl.where(pia, 1.0 / (rd_ * c2_), k0z)
    k1 = tl.where(pia, k1p, k1z)
    k2 = tl.where(pia, k2p, 1.0 / k0z)
    k3 = tl.where(pia, k3p, -k1z)
    k4 = tl.where(pia, k4p, 0.0)
    p1 = tl.where(pia, p1p, p1z)
    p2 = tl.where(pia, p2p, 1.0)
    ct = c1 + c2
    t0a, dta, cea, fa = _newton(ix1, ix2, val, nn, bd, bs, slew, ctot, rd_, ct, vth, vl, vh, der, A, B, D, p1p, p2p,
                                mask & pia, True, K, NIT)
    t0b, dtb, ceb, fb = _newton(ix1, ix2, val, nn, bd, bs, slew, c2_, rd_, ct, vth, vl, vh, der, A, B, D, p1p, p2p,
                                mask & pia & fa, True, K, NIT)
    use_b = fa & ~fb
    t0 = tl.where(use_b, t0b, t0a)
    dt = tl.where(use_b, dtb, dta)
    cep = tl.where(use_b, ceb, cea)
    pi_ok = pia & ~(fa & fb)
    t0z, dtz, _cez, fz = _newton(ix1, ix2, val, nn, bd, bs, slew, c1_, rd_, ct, vth, vl, vh, der, A, B, D, p1p, p2p,
                                 mask & zc2, False, K, NIT)
    zc_ok = zc2 & ~fz
    t0 = tl.where(zc2, t0z, t0)
    dt = tl.where(zc2, dtz, dt)
    ceff = tl.where(pi_ok, cep, tl.where(zc2, c1, ctot))
    tab_d = _nldm(ix1, ix2, val, nn, bd, slew, tl.maximum(ceff, 1e-9), mask, K)
    tab_s = _nldm(ix1, ix2, val, nn, bs, slew, tl.maximum(ceff, 1e-9), mask, K)
    dt = tl.where(dt > 0, dt, 1.0)
    ub = tl.where(pia, t0 + dt + (c1 + c2) * (rd_ + rpi_) * 2.0, t0 + dt + c1 * (rd_ + rpi_) * 2.0)
    one = ub * 0.0 + 1.0
    on = mask & (pi_ok | zc_ok)                               # lanes that use the driver waveform
    vo_d, f1 = _find_root(t0, ub, vth, k0, k1, k2, k3, k4, p1, p2, t0, dt, one, on, False, RIT)
    tlo, f2 = _find_root(t0, vo_d, vl, k0, k1, k2, k3, k4, p1, p2, t0, dt, one, on, False, RIT)
    thi, f3 = _find_root(vo_d, ub, vh, k0, k1, k2, k3, k4, p1, p2, t0, dt, one, on, False, RIT)
    vo_fail = f1 | f2 | f3
    wave_ok = (pi_ok | zc_ok) & ~vo_fail
    delay = tl.where(zc_ok & ~vo_fail, vo_d, tab_d)
    dslew = tl.where(wave_ok, (thi - tlo) / der, tab_s)
    zc_bad = zc2 & (fz | vo_fail)
    zd = _nldm(ix1, ix2, val, nn, bd, slew, tl.maximum(c1_, 1e-9), mask, K)
    zs = _nldm(ix1, ix2, val, nn, bs, slew, tl.maximum(c1_, 1e-9), mask, K)
    delay = tl.where(zc_bad, zd, delay)
    dslew = tl.where(zc_bad, zs, dslew)
    return delay, dslew, wave_ok, k0, k1, k2, k3, k4, p1, p2, t0, dt, vo_d, ub


@triton.jit(do_not_specialize=['N', 'K'])
def dmp_gate_kernel(ix1, ix2, val, nn, lib_p, orf_p, slew_p, ctot_p, cfar_p, rpi_p, thr_p,
                    delay_o, dslew_o, state_o, N,
                    K, NIT: tl.constexpr, RIT: tl.constexpr, BLOCK: tl.constexpr):
    """attar.dmp.gate for N arcs. thr: (N, 4) output threshold, slew lower, slew upper, slew derate."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    lib = tl.load(lib_p + i, mask=mask, other=0)
    orf = tl.load(orf_p + i, mask=mask, other=0)
    slew = tl.load(slew_p + i, mask=mask, other=0.0)
    ctot = tl.load(ctot_p + i, mask=mask, other=1.0)
    c1 = tl.load(cfar_p + i, mask=mask, other=0.0)
    rpi = tl.load(rpi_p + i, mask=mask, other=0.0)
    vth = tl.load(thr_p + i * 4, mask=mask, other=0.5)
    vl = tl.load(thr_p + i * 4 + 1, mask=mask, other=0.2)
    vh = tl.load(thr_p + i * 4 + 2, mask=mask, other=0.8)
    der = tl.load(thr_p + i * 4 + 3, mask=mask, other=1.0)
    delay, dslew, wave_ok, k0, k1, k2, k3, k4, p1, p2, t0, dt, vo_d, ub = _gate_dev(
        ix1, ix2, val, nn, lib, orf, slew, ctot, c1, rpi, vth, vl, vh, der, mask, K, NIT, RIT)
    tl.store(delay_o + i, delay, mask=mask)
    tl.store(dslew_o + i, dslew, mask=mask)
    so = state_o + i * 17
    tl.store(so + 0, wave_ok.to(tl.float32), mask=mask)
    tl.store(so + 1, k0, mask=mask); tl.store(so + 2, k1, mask=mask); tl.store(so + 3, k2, mask=mask)
    tl.store(so + 4, k3, mask=mask); tl.store(so + 5, k4, mask=mask); tl.store(so + 6, p1, mask=mask)
    tl.store(so + 7, p2, mask=mask); tl.store(so + 8, t0, mask=mask); tl.store(so + 9, dt, mask=mask)
    tl.store(so + 10, vo_d, mask=mask); tl.store(so + 11, ub, mask=mask); tl.store(so + 12, dslew, mask=mask)
    tl.store(so + 13, vth, mask=mask); tl.store(so + 14, vl, mask=mask); tl.store(so + 15, vh, mask=mask)
    tl.store(so + 16, der, mask=mask)


@triton.jit
def _load_dev(valid, k0, k1, k2, k3, k4, p1, p2, t0, dt, vo_d, ub, ds, vth, vl, vh, der, elm, rise,
              lvth, lvl, lvh, lder, RIT: tl.constexpr):
    """attar.dmp.load + threshold_adjust for one lane per load: (wire delay, load slew)."""
    small = (elm == 0) | (elm < ds * 1e-3) | ~valid
    e = tl.where(small, 1.0, elm)
    p3 = 1.0 / e
    t_up = ub + e * 2.0
    on = ~small
    ld, f1 = _find_root(t0, t_up, vth, k0, k1, k2, k3, k4, p1, p2, t0, dt, p3, on, True, RIT)
    tlo, f2 = _find_root(t0, ld, vl, k0, k1, k2, k3, k4, p1, p2, t0, dt, p3, on, True, RIT)
    thi, f3 = _find_root(ld, t_up, vh, k0, k1, k2, k3, k4, p1, p2, t0, dt, p3, on, True, RIT)
    fail = f1 | f2 | f3
    delay = ld - vo_d
    slew = (thi - tlo) / der
    delay = tl.where(delay < 0, elm, delay)
    slew = tl.maximum(slew, ds)
    use = ~small & ~fail
    w = tl.where(use, delay, elm)
    s = tl.where(use, slew, ds)
    # thresholdAdjust to the load's library
    dd = vh - vl
    delta = s * ((lvth - vth) / dd)
    w = w + tl.where(rise, delta, -delta)
    s = s * (((lvh - lvl) / lder) / (dd / der))
    return w, s


@triton.jit(do_not_specialize=['N'])
def dmp_load_kernel(state_p, elm_p, rise_p, lthr_p, wire_o, slew_o, N, RIT: tl.constexpr, BLOCK: tl.constexpr):
    """attar.dmp.load + threshold_adjust for N loads (state: (N, 17) rows of their arcs)."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    so = state_p + i * 17
    valid = tl.load(so + 0, mask=mask, other=0.0) > 0.5
    k0 = tl.load(so + 1, mask=mask, other=1.0); k1 = tl.load(so + 2, mask=mask, other=0.0)
    k2 = tl.load(so + 3, mask=mask, other=0.0); k3 = tl.load(so + 4, mask=mask, other=0.0)
    k4 = tl.load(so + 5, mask=mask, other=0.0); p1 = tl.load(so + 6, mask=mask, other=1.0)
    p2 = tl.load(so + 7, mask=mask, other=1.0); t0 = tl.load(so + 8, mask=mask, other=0.0)
    dt = tl.load(so + 9, mask=mask, other=1.0); vo_d = tl.load(so + 10, mask=mask, other=0.0)
    ub = tl.load(so + 11, mask=mask, other=1.0); ds = tl.load(so + 12, mask=mask, other=0.0)
    vth = tl.load(so + 13, mask=mask, other=0.5); vl = tl.load(so + 14, mask=mask, other=0.2)
    vh = tl.load(so + 15, mask=mask, other=0.8); der = tl.load(so + 16, mask=mask, other=1.0)
    elm = tl.load(elm_p + i, mask=mask, other=0.0)
    rise = tl.load(rise_p + i, mask=mask, other=0) != 0
    lvth = tl.load(lthr_p + i * 4, mask=mask, other=0.5); lvl = tl.load(lthr_p + i * 4 + 1, mask=mask, other=0.2)
    lvh = tl.load(lthr_p + i * 4 + 2, mask=mask, other=0.8); lder = tl.load(lthr_p + i * 4 + 3, mask=mask, other=1.0)
    w, s = _load_dev(valid, k0, k1, k2, k3, k4, p1, p2, t0, dt, vo_d, ub, ds, vth, vl, vh, der, elm, rise,
                     lvth, lvl, lvh, lder, RIT)
    tl.store(wire_o + i, w, mask=mask)
    tl.store(slew_o + i, s, mask=mask)


@triton.jit(do_not_specialize=['N', 'NT', 'K'])
def stage_gate_kernel(ix1, ix2, val, nn, tthr, lib_p, fi_p, t_p, orf_p, net_p, t0_p, tag_p, pi_p,
                      amax, amin, smax, smin, st_o, N,
                      NT, K, NIT: tl.constexpr, RIT: tl.constexpr, BLOCK: tl.constexpr):
    """One level's arcs, 2N lanes (max analysis, then min): DMP gate delay and driver slew, the driver pin's arrivals
    (per tag) and slews by float atomics, and per lane the row its loads need (STATE + delay, reach) in st_o.
    fi < 0: a launch arc (arrival t0 into tag at its clock pin, slew 0)."""
    l = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = l < 2 * N
    mn = l >= N
    c = tl.where(mn, l - N, l)
    lib = tl.load(lib_p + c, mask=mask, other=0)
    fi = tl.load(fi_p + c, mask=mask, other=-1)
    t = tl.load(t_p + c, mask=mask, other=0)
    orf = tl.load(orf_p + c, mask=mask, other=0)
    net = tl.load(net_p + c, mask=mask, other=0)
    launch = fi < 0
    t0l = tl.load(t0_p + c, mask=mask, other=0.0)
    tagl = tl.load(tag_p + c, mask=mask, other=0)
    fis = tl.maximum(fi, 0)
    INF = float('inf')
    sx = tl.load(smax + fis, mask=mask & ~launch, other=0.0)
    sn = tl.load(smin + fis, mask=mask & ~launch, other=0.0)
    slew = tl.where(mn, tl.where(sn < INF, sn, sx), sx)
    pb = pi_p + (net * 4 + orf + 2 * mn.to(tl.int32)) * 3
    cn = tl.load(pb, mask=mask, other=1.0); rr = tl.load(pb + 1, mask=mask, other=0.0); cf = tl.load(pb + 2, mask=mask, other=0.0)
    tb = tthr + (lib * 2 + orf) * 4
    vth = tl.load(tb, mask=mask, other=0.5); vl = tl.load(tb + 1, mask=mask, other=0.2)
    vh = tl.load(tb + 2, mask=mask, other=0.8); der = tl.load(tb + 3, mask=mask, other=1.0)
    delay, ds, ok, k0, k1, k2, k3, k4, p1, p2, t0, dt, vo, ub = _gate_dev(
        ix1, ix2, val, nn, lib, orf, slew, cn + cf, cf, rr, vth, vl, vh, der, mask, K, NIT, RIT)
    ti = t * 2 + orf
    reach = launch
    for g in range(NT):
        ax = tl.load(amax + fis * NT + g, mask=mask & ~launch, other=-INF)
        an = tl.load(amin + fis * NT + g, mask=mask & ~launch, other=INF)
        reach = reach | (ax > -INF) | (an < INF)
        ax = tl.where(launch, tl.where(tagl == g, t0l, -INF), ax) + delay
        an = tl.where(launch, tl.where(tagl == g, t0l, INF), an) + delay
        tl.atomic_max(amax + ti * NT + g, ax, mask=mask & ~mn & (ax > -INF))
        tl.atomic_min(amin + ti * NT + g, an, mask=mask & mn & (an < INF))
    tl.atomic_max(smax + ti, ds, mask=mask & ~mn & reach)
    tl.atomic_min(smin + ti, ds, mask=mask & mn & reach)
    so = st_o + l * 20
    tl.store(so + 0, ok.to(tl.float32), mask=mask)
    tl.store(so + 1, k0, mask=mask); tl.store(so + 2, k1, mask=mask); tl.store(so + 3, k2, mask=mask)
    tl.store(so + 4, k3, mask=mask); tl.store(so + 5, k4, mask=mask); tl.store(so + 6, p1, mask=mask)
    tl.store(so + 7, p2, mask=mask); tl.store(so + 8, t0, mask=mask); tl.store(so + 9, dt, mask=mask)
    tl.store(so + 10, vo, mask=mask); tl.store(so + 11, ub, mask=mask); tl.store(so + 12, ds, mask=mask)
    tl.store(so + 13, vth, mask=mask); tl.store(so + 14, vl, mask=mask); tl.store(so + 15, vh, mask=mask)
    tl.store(so + 16, der, mask=mask); tl.store(so + 17, delay, mask=mask); tl.store(so + 18, reach.to(tl.float32), mask=mask)


@triton.jit(do_not_specialize=['N', 'M', 'NT'])
def stage_load_kernel(st_p, ea_p, en_p, fi_p, orf_p, t0_p, tag_p, nto_p, elm_p, sthr_p,
                      amax, amin, smax, smin, N, M,
                      NT, RIT: tl.constexpr, BLOCK: tl.constexpr):
    """One level's loads, 2M lanes (max analysis, then min): DMP wire delay and load slew from the arc's row of
    stage_gate_kernel, threshold adjustment, the load pin's arrivals and slews by float atomics."""
    l = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = l < 2 * M
    mn = l >= M
    e = tl.where(mn, l - M, l)
    c = tl.load(ea_p + e, mask=mask, other=0)
    a = tl.load(en_p + e, mask=mask, other=0)
    row = st_p + (c + N * mn.to(tl.int32)) * 20
    ok = tl.load(row, mask=mask, other=0.0) > 0.5
    k0 = tl.load(row + 1, mask=mask, other=1.0); k1 = tl.load(row + 2, mask=mask, other=0.0)
    k2 = tl.load(row + 3, mask=mask, other=0.0); k3 = tl.load(row + 4, mask=mask, other=0.0)
    k4 = tl.load(row + 5, mask=mask, other=0.0); p1 = tl.load(row + 6, mask=mask, other=1.0)
    p2 = tl.load(row + 7, mask=mask, other=1.0); t0 = tl.load(row + 8, mask=mask, other=0.0)
    dt = tl.load(row + 9, mask=mask, other=1.0); vo = tl.load(row + 10, mask=mask, other=0.0)
    ub = tl.load(row + 11, mask=mask, other=1.0); ds = tl.load(row + 12, mask=mask, other=0.0)
    vth = tl.load(row + 13, mask=mask, other=0.5); vl = tl.load(row + 14, mask=mask, other=0.2)
    vh = tl.load(row + 15, mask=mask, other=0.8); der = tl.load(row + 16, mask=mask, other=1.0)
    delay = tl.load(row + 17, mask=mask, other=0.0)
    reach = tl.load(row + 18, mask=mask, other=0.0) > 0.5
    orf = tl.load(orf_p + c, mask=mask, other=0)
    fi = tl.load(fi_p + c, mask=mask, other=-1)
    launch = fi < 0
    fis = tl.maximum(fi, 0)
    t0l = tl.load(t0_p + c, mask=mask, other=0.0)
    tagl = tl.load(tag_p + c, mask=mask, other=0)
    sk = tl.load(nto_p + a, mask=mask, other=0)
    elm = tl.load(elm_p + a * 4 + orf + 2 * mn.to(tl.int32), mask=mask, other=0.0)
    lb = sthr_p + (sk * 2 + orf) * 4
    lvth = tl.load(lb, mask=mask, other=0.5); lvl = tl.load(lb + 1, mask=mask, other=0.2)
    lvh = tl.load(lb + 2, mask=mask, other=0.8); lder = tl.load(lb + 3, mask=mask, other=1.0)
    w, sl = _load_dev(ok, k0, k1, k2, k3, k4, p1, p2, t0, dt, vo, ub, ds, vth, vl, vh, der, elm, orf == 0,
                      lvth, lvl, lvh, lder, RIT)
    si = sk * 2 + orf
    INF = float('inf')
    for g in range(NT):
        ax = tl.load(amax + fis * NT + g, mask=mask & ~launch & ~mn, other=-INF)
        an = tl.load(amin + fis * NT + g, mask=mask & ~launch & mn, other=INF)
        ax = tl.where(launch, tl.where(tagl == g, t0l, -INF), ax) + delay + w
        an = tl.where(launch, tl.where(tagl == g, t0l, INF), an) + delay + w
        tl.atomic_max(amax + si * NT + g, ax, mask=mask & ~mn & (ax > -INF))
        tl.atomic_min(amin + si * NT + g, an, mask=mask & mn & (an < INF))
    tl.atomic_max(smax + si, sl, mask=mask & ~mn & reach)
    tl.atomic_min(smin + si, sl, mask=mask & mn & reach)


BLOCK = 128


def gate(ix1, ix2, val, tn, lib, orf, slew, ctot, c_far, rpi, thr):
    """Same as attar.sta.arc_eval: (delay, driver slew, state (N, 17))."""
    N = len(lib)
    dev = slew.device
    delay = torch.empty(N, device=dev); dslew = torch.empty(N, device=dev); state = torch.empty(N, NSTATE, device=dev)
    if N:
        K = ix1.shape[-1]
        dmp_gate_kernel[(triton.cdiv(N, BLOCK),)](
            ix1, ix2, val, tn.to(torch.int32) if tn.dtype != torch.int32 else tn, lib.to(torch.int32), orf.to(torch.int32),
            slew.contiguous(), ctot.contiguous(), c_far.contiguous(), rpi.contiguous(), thr.contiguous(),
            delay, dslew, state, N, K=K, NIT=NEWTON_MAX_ITER, RIT=ROOT_N, BLOCK=BLOCK)
    return delay, dslew, state


def load(state, elmore, rise, lthr):
    """Same as attar.sta.load_eval: (wire delay, load slew)."""
    N = len(elmore)
    dev = elmore.device
    w = torch.empty(N, device=dev); s = torch.empty(N, device=dev)
    if N:
        dmp_load_kernel[(triton.cdiv(N, BLOCK),)](state.contiguous(), elmore.contiguous(), rise.to(torch.int32),
                                                  lthr.contiguous(), w, s, N, RIT=ROOT_N, BLOCK=BLOCK)
    return w, s


def stage(tables, lev, sthr, pi, elm, nto, amax, amin, smax, smin):
    """One level of the timer on the GPU: stage_gate_kernel then stage_load_kernel. tables: (ix1, ix2, val, n32, thr);
    lev: the level's static arrays (lib, fi, t, orf, net, t0, tag, ea, en)."""
    ix1, ix2, val, n32, thr = tables
    lib, fi, t, orf, net, t0, tag, ea, en = lev
    N, M = len(lib), len(ea)
    if not N:
        return
    NT = amax.shape[1]
    st = torch.empty(2 * N, 20, device=amax.device)
    stage_gate_kernel[(triton.cdiv(2 * N, BLOCK),)](ix1, ix2, val, n32, thr, lib, fi, t, orf, net, t0, tag, pi,
                                                    amax, amin, smax, smin, st, N,
                                                    NT=NT, K=ix1.shape[-1], NIT=NEWTON_MAX_ITER, RIT=ROOT_N, BLOCK=BLOCK)
    if M:
        stage_load_kernel[(triton.cdiv(2 * M, BLOCK),)](st, ea, en, fi, orf, t0, tag, nto, elm, sthr,
                                                        amax, amin, smax, smin, N, M, NT=NT, RIT=ROOT_N, BLOCK=BLOCK)
