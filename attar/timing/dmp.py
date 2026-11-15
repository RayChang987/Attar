"""OpenSTA's dmp_ceff_elmore delay calculation (OpenSTA dcalc/DmpCeff.cc, DmpDelayCalc.cc, FindRoot.cc), vectorized.

Dartu, Menezes and Pileggi, "Performance Computation for Precharacterized CMOS Gates with RC Loads", IEEE TCAD 15(5),
1996: the gate is a ramp source (start t0, duration dt) behind a resistance Rd; on the net's pi model (near C2, Rpi,
far C1 in OpenSTA's names) Newton-Raphson finds t0, dt and the effective capacitance Ceff that make the model match the
gate's table at Ceff. The gate delay is the table delay at Ceff; the driver slew is measured on the driver waveform
Vo(t); a load's wire delay and slew are measured on Vl(t), Vo filtered by one pole 1 / Elmore.

Each function follows its OpenSTA counterpart (named in the docstrings), including its constants (tolerances,
iteration limits, exp approximation); units are ps, fF and kohm. Tensors hold one element per arc (or load); failures
fall back exactly where OpenSTA falls back. Iterations are fixed-count loops over single steps (newton_step,
root_step) in which converged or failed elements freeze: no host sync, so the whole timer can be a CUDA graph, and
each step compiles to a few fused kernels.
"""
import torch

DRIVER_PARAM_TOL = 0.01        # DmpAlg::driver_param_tol_
VTH_TIME_TOL = 0.01            # DmpAlg::vth_time_tol_
FIND_ROOT_MAX_ITER = 20        # DmpAlg::find_root_max_iter_
NEWTON_MAX_ITER = 100          # DmpAlg::newton_raphson_max_iter_
# The PyTorch reference runs fixed iteration counts (no host sync): OpenSTA's limits. (The Triton kernels stop as soon
# as every lane of a block has converged or failed, up to the same limits.) Heavy RC loads need 20-30 Newton
# iterations; an element not converged within the limit counts as failed, OpenSTA's fallback for that case.
NEWTON_ITER = NEWTON_MAX_ITER
ROOT_ITER = FIND_ROOT_MAX_ITER
CAP_STEP = 1.0                 # gateModelRd: cap2 = cap1 + 1e-15 F = 1 fF


def exp2(x):
    """DmpCeff.cc exp2: (1 + x / 4096)^4096 by 12 squarings, 0 below -12."""
    y = 1.0 + x / 4096.0
    for _ in range(12):
        y = y * y
    return torch.where(x < -12.0, torch.zeros_like(x), y)


def nldm(ix1, ix2, val, n, x, y):
    from .sta import nldm as _n
    return _n(ix1, ix2, val, n, x, y)


# -- the lumped-load ramp response (DmpAlg::y0, y, dy) -------------------------------------------------------------------
def _y0(t, cl, rd):
    return t - rd * cl * (1.0 - exp2(-t / (rd * cl)))


def _y0dt(t, cl, rd):
    return 1.0 - exp2(-t / (rd * cl))


def _y0dcl(t, cl, rd):
    return rd * ((1.0 + t / (rd * cl)) * exp2(-t / (rd * cl)) - 1)


def _y(t, t0, dt, cl, rd):
    t1 = t - t0
    v = torch.where(t1 <= dt, _y0(t1, cl, rd) / dt, (_y0(t1, cl, rd) - _y0(t1 - dt, cl, rd)) / dt)
    return torch.where(t1 <= 0, torch.zeros_like(t1), v)


def _dy(t, t0, dt, cl, rd):
    """DmpAlg::dy: d y / d (t0, dt, cl), as OpenSTA writes it."""
    t1 = t - t0
    z = torch.zeros_like(t1)
    r = t1 <= dt
    pos = t1 > 0
    a_t0 = -_y0dt(t1, cl, rd) / dt
    a_dt = -_y0(t1, cl, rd) / (dt * dt)
    a_cl = _y0dcl(t1, cl, rd) / dt
    b_t0 = -(_y0dt(t1, cl, rd) - _y0dt(t1 - dt, cl, rd)) / dt
    b_dt = -(_y0(t1, cl, rd) + _y0(t1 - dt, cl, rd)) / (dt * dt) + _y0dt(t1 - dt, cl, rd) / dt
    b_cl = (_y0dcl(t1, cl, rd) - _y0dcl(t1 - dt, cl, rd)) / dt
    return (torch.where(pos, torch.where(r, a_t0, b_t0), z), torch.where(pos, torch.where(r, a_dt, b_dt), z),
            torch.where(pos, torch.where(r, a_cl, b_cl), z))


def _gate_delays(tab, slew_in, ce, vth, vl, vh, derate):
    """DmpAlg::gateDelays: t_vth, t_vl and the measured slew at load ce. tab: the arc's delay and slew tables."""
    d_ix1, d_ix2, d_val, d_n, s_ix1, s_ix2, s_val, s_n = tab
    tv = nldm(d_ix1, d_ix2, d_val, d_n, slew_in, ce)
    s = nldm(s_ix1, s_ix2, s_val, s_n, slew_in, ce).clamp(min=0) * derate    # GateTableModel::gateDelay clips at 0
    return tv, tv - s * (vth - vl) / (vh - vl), s


def newton_step(x, conv, fail, order3: bool, tab, slew_in, rd, ctot, vth, vl, vh, derate, A, B, D, p1, p2):
    """One iteration of DmpAlg::newtonRaphson with DmpPi::evalDmpEqns (order3: t0, dt, ceff) or
    DmpOnePole::evalDmpEqns (t0, dt at a fixed ceff). x: (n, 3). Converged or failed elements keep their x."""
    one = torch.ones_like(ctot)
    t0, dt, ce = x[:, 0], x[:, 1], x[:, 2]
    if order3:
        bad = (ce < 0) | (ce > ctot + 1e-6) | (dt <= 0)
    else:
        bad = torch.zeros_like(fail)
    ce_s = ce.clamp(min=1e-9)
    tv, tl, s = _gate_delays(tab, slew_in, ce_s, vth, vl, vh, derate)
    bad = bad | (s == 0)
    vr = vh - vl
    if order3:
        dt_s = torch.where(dt <= 0, one, dt)
    else:                                                    # DmpOnePole resets a bad dt
        dt_s = torch.where(dt <= 0, (tl - tv) / 100, dt)
        x = torch.stack([t0, dt_s, ce], 1)
    y50 = _y(tv, t0, dt_s, ce_s, rd) - vth
    y20 = _y(tl, t0, dt_s, ce_s, rd) - vl
    j20 = _dy(tl, t0, dt_s, ce_s, rd)
    j50 = _dy(tv, t0, dt_s, ce_s, rd)
    if order3:
        ct = torch.minimum(s / vr, 1.4 * dt_s)
        e1 = exp2(-p1 * ct); e2 = exp2(-p2 * ct); erc = exp2(-ct / (rd * ce_s))
        ipi = (A * ct + (B / p1) * (1.0 - e1) + (D / p2) * (1.0 - e2)) / (rd * ct * dt_s)
        ice = (rd * ce_s * ct - (rd * ce_s) ** 2 * (1.0 - erc)) / (rd * ct * dt_s)
        fi = ipi - ice
        e1d = exp2(-p1 * dt_s); e2d = exp2(-p2 * dt_s); erd = exp2(-dt_s / (rd * ce_s))
        ji_dt = (-A * dt_s + B * dt_s * e1d - (2 * B / p1) * (1.0 - e1d) + D * dt_s * e2d - (2 * D / p2) * (1.0 - e2d)
                 + rd * ce_s * (dt_s + dt_s * erd - 2 * rd * ce_s * (1.0 - erd))) / (rd * dt_s ** 3)
        ji_ce = (2 * rd * ce_s - dt_s - (2 * rd * ce_s + dt_s) * erd) / (dt_s * dt_s)
        # rows y20, y50, ipi; columns t0, dt, ceff: J p = -f by cofactors
        a, b, c = j20; d_, e, g = j50; h, i, k = torch.zeros_like(fi), ji_dt, ji_ce
        f0, f1, f2 = -y20, -y50, -fi
        C0 = e * k - g * i; C1 = -(d_ * k - g * h); C2 = d_ * i - e * h
        det = a * C0 + b * C1 + c * C2
        sing = det.abs() < 1e-300
        det = torch.where(sing, one, det)
        p0 = (C0 * f0 - (b * k - c * i) * f1 + (b * g - c * e) * f2) / det
        p1_ = (C1 * f0 + (a * k - c * h) * f1 - (a * g - c * d_) * f2) / det
        p2_ = (C2 * f0 - (a * i - b * h) * f1 + (a * e - b * d_) * f2) / det
        p = torch.stack([p0, p1_, p2_], 1)
        under = (p.abs() <= x.abs() * DRIVER_PARAM_TOL).all(1)
    else:
        a, b, _ = j20; c, d_, _ = j50
        det = a * d_ - b * c
        sing = det.abs() < 1e-300
        det = torch.where(sing, one, det)
        p = torch.stack([(-y20 * d_ + y50 * b) / det, (y20 * c - y50 * a) / det, torch.zeros_like(det)], 1)
        under = (p[:, :2].abs() <= x[:, :2].abs() * DRIVER_PARAM_TOL).all(1)
    act = ~conv & ~fail
    fail = fail | (act & (bad | sing | ~torch.isfinite(p).all(1)))
    act = act & ~fail
    x = torch.where(act[:, None], x + p, x)
    conv = conv | (act & under)
    return x, conv, fail


def _V0(t, k0, k1, k2, k3, k4, p1, p2):
    e1, e2 = exp2(-p1 * t), exp2(-p2 * t)
    return k0 * (k1 + k2 * t + k3 * e1 + k4 * e2), k0 * (k2 - k3 * p1 * e1 - k4 * p2 * e2)


def _Vl0(t, k0, k1, k2, k3, k4, p1, p2, p3):
    D1 = k0 * (k1 - k2 / p3)
    D3 = -p3 * k0 * k3 / (p1 - p3)
    nz = k4 != 0
    D4 = torch.where(nz, -p3 * k0 * k4 / (p2 - p3), torch.zeros_like(k4))
    D5 = k0 * (k2 / p3 - k1 + p3 * k3 / (p1 - p3)) + torch.where(nz, k0 * p3 * k4 / (p2 - p3), torch.zeros_like(k4))
    e1, e2, e3 = exp2(-p1 * t), exp2(-p2 * t), exp2(-p3 * t)
    return D1 + t + D3 * e1 + D4 * e2 + D5 * e3, 1.0 - D3 * p1 * e1 - D4 * p2 * e2 - D5 * p3 * e3


def wave(t, w, p3, load: bool):
    """DmpAlg::Vo (load False) or ::Vl (load True) and its derivative: the response to the ramp of duration dt starting
    at t0. w: (k0, k1, k2, k3, k4, p1, p2, t0, dt)."""
    k0, k1, k2, k3, k4, p1, p2, t0, dt = w
    t1 = t - t0
    if load:
        v1, d1 = _Vl0(t1, k0, k1, k2, k3, k4, p1, p2, p3)
        v2, d2 = _Vl0(t1 - dt, k0, k1, k2, k3, k4, p1, p2, p3)
    else:
        v1, d1 = _V0(t1, k0, k1, k2, k3, k4, p1, p2)
        v2, d2 = _V0(t1 - dt, k0, k1, k2, k3, k4, p1, p2)
    z = torch.zeros_like(t1)
    r = t1 <= dt
    v = torch.where(t1 <= 0, z, torch.where(r, v1 / dt, (v1 - v2) / dt))
    d = torch.where(t1 <= 0, z, torch.where(r, d1 / dt, (d1 - d2) / dt))
    return v, d


def root_init(x1, x2, target, w, p3, load: bool):
    """FindRoot.cc findRoot, setup: (a, b, root, dx, dx_prev, conv, bad, y, dy) with func(a) < 0."""
    y1, _ = wave(x1, w, p3, load); y2, _ = wave(x2, w, p3, load)
    y1 = y1 - target; y2 = y2 - target
    bad = ((y1 > 0) & (y2 > 0)) | ((y1 < 0) & (y2 < 0))
    exact = (y1 == 0) | (y2 == 0)
    sw = y1 > 0
    a, b = torch.where(sw, x2, x1), torch.where(sw, x1, x2)
    root = (a + b) * 0.5
    root = torch.where(exact & ~bad, torch.where(y1 == 0, x1, x2), root)
    dx_prev = (b - a).abs()
    y, dy = wave(root, w, p3, load)
    return a, b, root, dx_prev.clone(), dx_prev, exact | bad, bad, y - target, dy


def root_step(a, b, root, dx, dx_prev, conv, y, dy, target, w, p3, load: bool):
    """One iteration of FindRoot.cc findRoot (Newton-Raphson, bisecting when out of range or slow)."""
    bis = ((((root - b) * dy - y) * ((root - a) * dy - y)) > 0) | ((2.0 * y).abs() > (dx_prev * dy).abs())
    nd = torch.where(bis, (b - a) * 0.5, y / torch.where(dy == 0, torch.ones_like(dy), dy))
    nr = torch.where(bis, a + nd, root - nd)
    act = ~conv
    dx_prev = torch.where(act, dx, dx_prev)
    dx = torch.where(act, nd, dx)
    root = torch.where(act, nr, root)
    conv = conv | (act & (dx.abs() <= VTH_TIME_TOL * root.abs()))
    y2, dy2 = wave(root, w, p3, load)
    y2 = y2 - target
    act = ~conv
    a = torch.where(act & (y2 < 0), root, a)
    b = torch.where(act & (y2 >= 0), root, b)
    return a, b, root, dx, dx_prev, conv, torch.where(act, y2, y), torch.where(act, dy2, dy)


class Steps:
    """The single steps, compiled (torch.compile) or eager."""

    def __init__(self, compile_fn=None):
        c = compile_fn or (lambda f: f)
        self.newton = c(newton_step)
        self.root_init = c(root_init)
        self.root_step = c(root_step)


EAGER = Steps()


def find_root(x1, x2, target, w, p3, load, steps=EAGER):
    """FindRoot.cc findRoot for the waveform crossing wave(t) = target in [x1, x2]. Returns (root, failed)."""
    a, b, root, dx, dx_prev, conv, bad, y, dy = steps.root_init(x1, x2, target, w, p3, load)
    for _ in range(min(ROOT_ITER, FIND_ROOT_MAX_ITER)):
        a, b, root, dx, dx_prev, conv, y, dy = steps.root_step(a, b, root, dx, dx_prev, conv, y, dy, target, w, p3, load)
    return root, bad | ~conv


def gate(tab, slew_in, ctot, c2, rpi, c1, vth, vl, vh, derate, steps=EAGER):
    """DmpCeffDelayCalc::gateDelay for a batch of arcs. tab: the arcs' (delay ix1, ix2, val, n, slew ix1, ix2, val, n);
    slew_in: their input slews; ctot: the driven net's total capacitance; (c2, rpi, c1): its pi model, near / R / far;
    vth, vl, vh, derate: the driver library's output threshold, slew thresholds and slew derate.
    Returns (gate delay, driver slew, ceff, state) where state holds what its loads need (DmpAlg::loadDelaySlew)."""
    one = torch.ones_like(ctot)
    # the table's delay and slew at load c (GateTableModel::gateDelay: a negative slew, extrapolated, clipped to 0)
    look = lambda c: (nldm(*tab[:4], slew_in, c), nldm(*tab[4:], slew_in, c).clamp(min=0))  # noqa: E731
    # gateModelRd: the table's slope at the total load
    d1, _ = look(ctot); d2, _ = look(ctot + CAP_STEP)
    rd = -torch.log(vth) * (d1 - d2).abs() / CAP_STEP
    # setCeffAlgorithm (thresholds in SI there: rd < 1e-2 ohm = 1e-5 kohm)
    cap_alg = (rd < 1e-5) | (rpi < rd * 1e-3) | (c1 == 0) | (c1 < c2 * 1e-3) | (rpi == 0)
    zc2 = ~cap_alg & (c2 < c1 * 1e-3)
    pia = ~cap_alg & ~zc2
    rd_ = torch.where(cap_alg, one, rd); rpi_ = torch.where(cap_alg, one, rpi)
    c1_ = torch.where(cap_alg | (c1 <= 0), one, c1); c2_ = torch.where(pia, c2, one)
    # DmpPi::init
    z1 = 1.0 / (rpi_ * c1_)
    a = rpi_ * rd_ * c1_ * c2_
    b = rd_ * (c1_ + c2_) + rpi_ * c1_
    sq = torch.sqrt((b * b - 4 * a).clamp(min=0))
    p1p = (b + sq) / (2 * a); p2p = (b - sq) / (2 * a)
    p1p2 = p1p * p2p
    k2p = z1 / p1p2; k1p = (1.0 - k2p * (p1p + p2p)) / p1p2
    k4p = (k1p * p1p + k2p) / (p2p - p1p); k3p = -k1p - k4p
    zz = (c1_ + c2_) / (rpi_ * c1_ * c2_)
    A = zz / p1p2; B = (zz - p1p) / (p1p * (p1p - p2p)); D = (zz - p2p) / (p2p * (p2p - p1p))
    # DmpZeroC2::init (ceff = c1, the far capacitance)
    p1z = 1.0 / (c1_ * (rd_ + rpi_))
    k0z = p1z / z1; k1z = (p1z - z1) / (p1z * p1z)
    k0 = torch.where(pia, 1.0 / (rd_ * c2_), k0z); k1 = torch.where(pia, k1p, k1z)
    k2 = torch.where(pia, k2p, 1.0 / k0z); k3 = torch.where(pia, k3p, -k1z)
    k4 = torch.where(pia, k4p, torch.zeros_like(k4p))
    p1 = torch.where(pia, p1p, p1z); p2 = torch.where(pia, p2p, one)
    vr = vh - vl

    def newton(ceff0, order3, k=1):
        """DmpAlg::findDriverParams + newtonRaphson; returns t0, dt, ceff, failed. k: copies of the batch in ceff0."""
        r = (lambda v: v.repeat(k)) if k > 1 else (lambda v: v)
        tb = tuple(v.repeat(k, *([1] * (v.dim() - 1))) if k > 1 else v for v in tab)
        sl_, rd2, ct2, vth2, vl2, vh2, de2 = r(slew_in), r(rd_), r(c1 + c2), r(vth), r(vl), r(vh), r(derate)
        A2, B2, D2, p12, p22 = r(A), r(B), r(D), r(p1p), r(p2p)
        tv, _, s = _gate_delays(tb, sl_, ceff0, vth2, vl2, vh2, de2)
        dt = s / (vh2 - vl2)
        t0 = tv + torch.log(1.0 - vth2) * rd2 * ceff0 - vth2 * dt
        x = torch.stack([t0, dt, ceff0], 1)
        conv = torch.zeros_like(ceff0, dtype=torch.bool); fail = torch.zeros_like(conv)
        for _ in range(NEWTON_ITER):
            x, conv, fail = steps.newton(x, conv, fail, order3, tb, sl_, rd2, ct2, vth2, vl2, vh2, de2, A2, B2, D2, p12, p22)
        t0, dt, ce = x[:, 0], x[:, 1], x[:, 2]
        fail = fail | ~conv | (dt <= 0) | (ce < 0) | (ce > ct2 + 1e-6)
        return t0, dt, ce, fail

    newton2 = lambda c: newton(c, True, 2)  # noqa: E731

    # DmpPi::gateDelaySlew: Newton from c1 + c2, else from c2; failing both -> lumped ctot (both starts in one batch)
    n = len(ctot)
    t0ab, dtab, ceab, fab = newton2(torch.cat([ctot, c2_]))
    t0a, t0b = t0ab[:n], t0ab[n:]; dta, dtb = dtab[:n], dtab[n:]; cea, ceb = ceab[:n], ceab[n:]; fa, fb = fab[:n], fab[n:]
    use_b = fa & ~fb
    t0 = torch.where(use_b, t0b, t0a); dt = torch.where(use_b, dtb, dta); cep = torch.where(use_b, ceb, cea)
    pi_ok = pia & ~(fa & fb)
    # DmpZeroC2: order 2 at ceff = c1
    t0z, dtz, _, fz = newton(c1_.clone(), False)
    zc_ok = zc2 & ~fz
    t0 = torch.where(zc2, t0z, t0); dt = torch.where(zc2, dtz, dt)
    ceff = torch.where(pi_ok, cep, torch.where(zc2, c1, ctot))
    tab_d, tab_s = look(ceff.clamp(min=1e-9))
    dt = torch.where(dt > 0, dt, one)
    w = (k0, k1, k2, k3, k4, p1, p2, t0, dt)
    # driver waveform crossings (DmpAlg::findDriverDelaySlew)
    ub = torch.where(pia, t0 + dt + (c1 + c2) * (rd_ + rpi_) * 2.0, t0 + dt + c1 * (rd_ + rpi_) * 2.0)
    vo_d, f1 = find_root(t0, ub, vth, w, one, False, steps)
    w2 = tuple(torch.cat([v, v]) for v in w)
    lh, f23 = find_root(torch.cat([t0, vo_d]), torch.cat([vo_d, ub]), torch.cat([vl, vh]), w2, torch.cat([one, one]),
                        False, steps)
    tl, th, f2, f3 = lh[:n], lh[n:], f23[:n], f23[n:]
    vo_fail = f1 | f2 | f3
    wave_ok = (pi_ok | zc_ok) & ~vo_fail
    # delay: Pi uses the table at Ceff, ZeroC2 the waveform; slew: the waveform when it worked, else the table
    delay = torch.where(zc_ok & ~vo_fail, vo_d, tab_d)
    slew = torch.where(wave_ok, (th - tl) / derate, tab_s)
    zc_bad = zc2 & (fz | vo_fail)
    zd, zs = look(c1_.clamp(min=1e-9))
    delay = torch.where(zc_bad, zd, delay); slew = torch.where(zc_bad, zs, slew)
    state = {'valid': wave_ok, 'k0': k0, 'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4, 'p1': p1, 'p2': p2, 't0': t0,
             'dt': dt, 'vo_delay': vo_d, 'ub': ub, 'drvr_slew': slew, 'vth': vth, 'vl': vl, 'vh': vh, 'derate': derate}
    return delay, slew, ceff, state


def load(state, elmore, steps=EAGER):
    """DmpAlg::loadDelaySlew for loads of the drivers in state (one entry per load, state already gathered).
    Returns (wire delay, load slew) at the driver library's thresholds."""
    ds = state['drvr_slew']
    small = (elmore == 0) | (elmore < ds * 1e-3) | ~state['valid']
    e = torch.where(small, torch.ones_like(elmore), elmore)
    p3 = 1.0 / e
    w = tuple(state[k] for k in ('k0', 'k1', 'k2', 'k3', 'k4', 'p1', 'p2', 't0', 'dt'))
    t_lo, t_up = state['t0'], state['ub'] + e * 2.0
    ld, f1 = find_root(t_lo, t_up, state['vth'], w, p3, True, steps)
    n = len(ld)
    w2 = tuple(torch.cat([v, v]) for v in w)
    lh, f23 = find_root(torch.cat([t_lo, ld]), torch.cat([ld, t_up]), torch.cat([state['vl'], state['vh']]), w2,
                        torch.cat([p3, p3]), True, steps)
    tl, th, f2, f3 = lh[:n], lh[n:], f23[:n], f23[n:]
    fail = f1 | f2 | f3
    delay = ld - state['vo_delay']
    slew = (th - tl) / state['derate']
    delay = torch.where(delay < 0, elmore, delay)
    slew = torch.maximum(slew, ds)
    use = ~small & ~fail
    return torch.where(use, delay, elmore), torch.where(use, slew, ds)


def input_port_load(drvr_slew, elmore, vth, vl, vh, derate):
    """DelayCalcBase::dspfWireDelaySlew: a net driven by an input port (no driver model), the load's thresholds."""
    return -elmore * torch.log(1.0 - vth), drvr_slew + elmore * torch.log((1.0 - vl) / (1.0 - vh)) / derate


def threshold_adjust(wire_delay, load_slew, rise, d_vth, d_vl, d_vh, d_der, l_vth, l_vl, l_vh, l_der):
    """DelayCalcBase::thresholdAdjust: driver library thresholds -> the load library's (no-op when equal)."""
    dd = d_vh - d_vl
    delta = load_slew * ((l_vth - d_vth) / dd)
    wire_delay = wire_delay + torch.where(rise, delta, -delta)
    load_slew = load_slew * (((l_vh - l_vl) / l_der) / (dd / d_der))
    return wire_delay, load_slew
