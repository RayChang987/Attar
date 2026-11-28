// OpenSTA's dmp_ceff_elmore delay calculation (dcalc/DmpCeff.cc, FindRoot.cc) for one element: the gate delay, driver
// slew and driver waveform of an arc (gate_one), the wire delay and load slew of one of its loads (load_one).
// attar/timing/dmp_triton.py's MODE 2 kernels translated line by line, in double precision (OpenSTA's), every
// iteration loop stopping as soon as its element has converged or failed; bit for bit what those kernels and the
// PyTorch reference (attar/timing/dmp.py) give: the kernels are launched without FMA contraction and this is compiled
// with -ffp-contract=off. Delay and slew cross the interface in float32, the waveform state in float64 (as the kernels
// keep it between the gate and its loads). Included by dmp.cpp (the CPU timer's batches) and inc.cpp (the incremental re-timing on the CPU).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace attar_dmp {

constexpr int NSTATE = 17;  // attar.timing.sta.STATE: valid, k0..k4, p1, p2, t0, dt, vo_delay, ub, slew, vth, vl, vh, der


struct Tables {
  const float* ix1;   // (A, 4, K)
  const float* ix2;   // (A, 4, K)
  const float* val;   // (A, 4, K, K)
  const int64_t* nn;  // (A, 4, 2)
  int64_t K;
};

inline double exp2a(double x) {  // DmpCeff.cc exp2: (1 + x / 4096)^4096 by 12 squarings, 0 below -12
  double y = 1.0 + x / 4096.0;
  for (int i = 0; i < 12; ++i) y = y * y;
  return x < -12.0 ? 0.0 : y;
}

// attar.timing.sta.nldm for table base = arc * 4 + slot: bilinear, linear extrapolation
inline double nldm(const Tables& T, int64_t base, double x, double y) {
  const int64_t K = T.K;
  const int64_t n1 = T.nn[base * 2], n2 = T.nn[base * 2 + 1];
  const float* a = T.ix1 + base * K;
  const float* b = T.ix2 + base * K;
  int64_t i0 = 0, j0 = 0;
  for (int64_t j = 1; j < K - 1; ++j) {
    i0 += (x >= a[j]) && (j < n1 - 1);
    j0 += (y >= b[j]) && (j < n2 - 1);
  }
  const int64_t i1 = std::min(i0 + 1, std::max<int64_t>(n1 - 1, 0));
  const int64_t j1 = std::min(j0 + 1, std::max<int64_t>(n2 - 1, 0));
  const double a0 = a[i0], a1 = a[i1], b0 = b[j0], b1 = b[j1];
  const double da = a1 - a0, db = b1 - b0;
  const double tx = (i1 > i0 && da != 0) ? (x - a0) / da : 0.0;
  const double ty = (j1 > j0 && db != 0) ? (y - b0) / db : 0.0;
  const float* v = T.val + base * K * K;
  const double v00 = v[i0 * K + j0], v10 = v[i1 * K + j0], v01 = v[i0 * K + j1], v11 = v[i1 * K + j1];
  return v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty) + v01 * (1 - tx) * ty + v11 * tx * ty;
}

inline double y0f(double t, double cl, double rd) { return t - rd * cl * (1.0 - exp2a(-t / (rd * cl))); }
inline double y0dt(double t, double cl, double rd) { return 1.0 - exp2a(-t / (rd * cl)); }
inline double y0dcl(double t, double cl, double rd) {
  return rd * ((1.0 + t / (rd * cl)) * exp2a(-t / (rd * cl)) - 1);
}

inline double yf(double t, double t0, double dt, double cl, double rd) {
  const double t1 = t - t0;
  if (t1 <= 0) return 0.0;
  return t1 <= dt ? y0f(t1, cl, rd) / dt : (y0f(t1, cl, rd) - y0f(t1 - dt, cl, rd)) / dt;
}

inline void dyf(double t, double t0, double dt, double cl, double rd, double& d_t0, double& d_dt, double& d_cl) {
  const double t1 = t - t0;
  if (!(t1 > 0)) { d_t0 = d_dt = d_cl = 0.0; return; }
  if (t1 <= dt) {
    d_t0 = -y0dt(t1, cl, rd) / dt;
    d_dt = -y0f(t1, cl, rd) / (dt * dt);
    d_cl = y0dcl(t1, cl, rd) / dt;
  } else {
    d_t0 = -(y0dt(t1, cl, rd) - y0dt(t1 - dt, cl, rd)) / dt;
    d_dt = -(y0f(t1, cl, rd) + y0f(t1 - dt, cl, rd)) / (dt * dt) + y0dt(t1 - dt, cl, rd) / dt;
    d_cl = (y0dcl(t1, cl, rd) - y0dcl(t1 - dt, cl, rd)) / dt;
  }
}

struct Arc {  // one arc's table rows and thresholds
  const Tables* T;
  int64_t bd, bs;  // delay and slew tables: lib * 4 + orf, lib * 4 + 2 + orf
  double slew, vth, vl, vh, der;
};

inline void gate_delays(const Arc& g, double ce, double& tv, double& tl_, double& s) {
  tv = nldm(*g.T, g.bd, g.slew, ce);
  s = std::max(nldm(*g.T, g.bs, g.slew, ce), 0.0) * g.der;  // gateDelay clips at 0
  tl_ = tv - s * (g.vth - g.vl) / (g.vh - g.vl);
}

// DmpAlg::findDriverParams + newtonRaphson (dmp_triton._newton)
void newton(const Arc& g, double ce0, double rd, double ctot, double A, double B, double D, double p1, double p2,
            bool order3, int iters, double& t0, double& dt, double& ce, bool& failed) {
  double tv, tl_, s;
  gate_delays(g, ce0, tv, tl_, s);
  dt = s / (g.vh - g.vl);
  t0 = tv + std::log(1.0 - g.vth) * rd * ce0 - g.vth * dt;
  ce = ce0;
  bool conv = false, fail = false;
  for (int it = 0; it < iters && !conv && !fail; ++it) {
    bool bad = order3 ? ((ce < 0) || (ce > ctot + 1e-6) || (dt <= 0)) : false;
    const double ce_s = std::max(ce, 1e-9);
    gate_delays(g, ce_s, tv, tl_, s);
    bad = bad || (s == 0);
    double dt_s;
    if (order3) {
      dt_s = dt <= 0 ? 1.0 : dt;
    } else {
      dt_s = dt <= 0 ? (tl_ - tv) / 100 : dt;
      dt = dt_s;
    }
    const double y50 = yf(tv, t0, dt_s, ce_s, rd) - g.vth;
    const double y20 = yf(tl_, t0, dt_s, ce_s, rd) - g.vl;
    double a, b, c, d_, e, gg;
    dyf(tl_, t0, dt_s, ce_s, rd, a, b, c);
    dyf(tv, t0, dt_s, ce_s, rd, d_, e, gg);
    double q0, q1, q2, det;
    bool sing, under;
    if (order3) {
      const double ct = std::min(s / (g.vh - g.vl), 1.4 * dt_s);
      const double e1 = exp2a(-p1 * ct), e2 = exp2a(-p2 * ct), erc = exp2a(-ct / (rd * ce_s));
      const double ipi = (A * ct + (B / p1) * (1.0 - e1) + (D / p2) * (1.0 - e2)) / (rd * ct * dt_s);
      const double ice = (rd * ce_s * ct - (rd * ce_s) * (rd * ce_s) * (1.0 - erc)) / (rd * ct * dt_s);
      const double fi = ipi - ice;
      const double e1d = exp2a(-p1 * dt_s), e2d = exp2a(-p2 * dt_s), erd = exp2a(-dt_s / (rd * ce_s));
      const double i_ = (-A * dt_s + B * dt_s * e1d - (2 * B / p1) * (1.0 - e1d) + D * dt_s * e2d -
                         (2 * D / p2) * (1.0 - e2d) +
                         rd * ce_s * (dt_s + dt_s * erd - 2 * rd * ce_s * (1.0 - erd))) /
                        (rd * dt_s * dt_s * dt_s);
      const double k = (2 * rd * ce_s - dt_s - (2 * rd * ce_s + dt_s) * erd) / (dt_s * dt_s);
      const double h = 0.0;
      const double f0 = -y20, f1 = -y50, f2 = -fi;
      const double C0 = e * k - gg * i_, C1 = -(d_ * k - gg * h), C2 = d_ * i_ - e * h;
      det = a * C0 + b * C1 + c * C2;
      sing = det == 0;
      if (sing) det = 1.0;
      q0 = (C0 * f0 - (b * k - c * i_) * f1 + (b * gg - c * e) * f2) / det;
      q1 = (C1 * f0 + (a * k - c * h) * f1 - (a * gg - c * d_) * f2) / det;
      q2 = (C2 * f0 - (a * i_ - b * h) * f1 + (a * e - b * d_) * f2) / det;
      under = (std::fabs(q0) <= std::fabs(t0) * 0.01) && (std::fabs(q1) <= std::fabs(dt) * 0.01) &&
              (std::fabs(q2) <= std::fabs(ce) * 0.01);
    } else {
      det = a * e - b * d_;
      sing = det == 0;
      if (sing) det = 1.0;
      q0 = (-y20 * e + y50 * b) / det;
      q1 = (y20 * d_ - y50 * a) / det;
      q2 = 0.0 * q0;
      under = (std::fabs(q0) <= std::fabs(t0) * 0.01) && (std::fabs(q1) <= std::fabs(dt) * 0.01);
    }
    const bool finite = (q0 == q0) && (q1 == q1) && (q2 == q2) && (std::fabs(q0) < 1e30) &&
                        (std::fabs(q1) < 1e30) && (std::fabs(q2) < 1e30);
    if (bad || sing || !finite) { fail = true; break; }
    t0 += q0; dt += q1; ce += q2;
    conv = under;
  }
  failed = fail || !conv || (dt <= 0) || (ce < 0) || (ce > ctot + 1e-6);
}

struct Wave {  // the driver (or load) waveform's parameters
  double k0, k1, k2, k3, k4, p1, p2, t0, dt, p3;
  bool load;
};

inline void V0(const Wave& w, double t, double& v, double& d) {
  const double e1 = exp2a(-w.p1 * t), e2 = exp2a(-w.p2 * t);
  v = w.k0 * (w.k1 + w.k2 * t + w.k3 * e1 + w.k4 * e2);
  d = w.k0 * (w.k2 - w.k3 * w.p1 * e1 - w.k4 * w.p2 * e2);
}

inline void Vl0(const Wave& w, double t, double& v, double& d) {
  const double p3 = w.p3;
  const double D1 = w.k0 * (w.k1 - w.k2 / p3);
  const double D3 = -p3 * w.k0 * w.k3 / (w.p1 - p3);
  const bool nz = w.k4 != 0;
  const double D4 = nz ? -p3 * w.k0 * w.k4 / (w.p2 - p3) : 0.0;
  const double D5 = w.k0 * (w.k2 / p3 - w.k1 + p3 * w.k3 / (w.p1 - p3)) + (nz ? w.k0 * p3 * w.k4 / (w.p2 - p3) : 0.0);
  const double e1 = exp2a(-w.p1 * t), e2 = exp2a(-w.p2 * t), e3 = exp2a(-p3 * t);
  v = D1 + t + D3 * e1 + D4 * e2 + D5 * e3;
  d = 1.0 - D3 * w.p1 * e1 - D4 * w.p2 * e2 - D5 * p3 * e3;
}

inline void wave(const Wave& w, double t, double& v, double& d) {
  const double t1 = t - w.t0;
  double v1, d1, v2, d2;
  if (w.load) { Vl0(w, t1, v1, d1); Vl0(w, t1 - w.dt, v2, d2); } else { V0(w, t1, v1, d1); V0(w, t1 - w.dt, v2, d2); }
  if (t1 <= 0) { v = 0.0; d = 0.0; return; }
  if (t1 <= w.dt) { v = v1 / w.dt; d = d1 / w.dt; } else { v = (v1 - v2) / w.dt; d = (d1 - d2) / w.dt; }
}

// FindRoot.cc findRoot (dmp_triton._find_root); off: the element does not iterate (its result is unused)
double find_root(const Wave& w, double x1, double x2, double target, bool on, int iters, bool& failed) {
  double y1, y2, dd;
  wave(w, x1, y1, dd); wave(w, x2, y2, dd);
  y1 -= target; y2 -= target;
  const bool bad = ((y1 > 0) && (y2 > 0)) || ((y1 < 0) && (y2 < 0));
  const bool exact = (y1 == 0) || (y2 == 0);
  const bool sw = y1 > 0;
  double a = sw ? x2 : x1, b = sw ? x1 : x2;
  double root = (a + b) * 0.5;
  if (exact && !bad) root = y1 == 0 ? x1 : x2;
  double dx_prev = std::fabs(b - a), dx = dx_prev;
  double y, dy;
  wave(w, root, y, dy);
  y -= target;
  bool conv = exact || bad || !on;
  for (int it = 0; it < iters && !conv; ++it) {
    const bool bis = ((((root - b) * dy - y) * ((root - a) * dy - y)) > 0) || (std::fabs(2.0 * y) > std::fabs(dx_prev * dy));
    const double nd = bis ? (b - a) * 0.5 : y / (dy == 0 ? 1.0 : dy);
    const double nr = bis ? a + nd : root - nd;
    dx_prev = dx; dx = nd; root = nr;
    conv = std::fabs(dx) <= 0.01 * std::fabs(root);
    double yy, dyy;
    wave(w, root, yy, dyy);
    yy -= target;
    if (!conv) {
      if (yy < 0) a = root; else b = root;
      y = yy; dy = dyy;
    }
  }
  failed = bad || !conv;
  return root;
}


// dmp_triton._gate_dev (MODE 2) for one arc: lib, orf the Liberty arc and output transition; slew the input slew; ctot,
// cfar, rpi the driven net's pi model (total, far capacitance, resistance); thr: output threshold, slew lower, slew
// upper, slew derate. Outputs: delay, driver slew, state (NSTATE doubles).
inline void gate_one(const Tables& T, int64_t lib, int64_t orf, float slew_f, float ctot_f, float cfar_f, float rpi_f,
                     const float* thr, int nit, int rit, float& delay_o, float& dslew_o, double* so) {
  const double slew = slew_f, ctot = ctot_f, rpi = rpi_f;
  const double vth = thr[0], vl = thr[1], vh = thr[2], der = thr[3];
  double c1 = std::min<double>(cfar_f, ctot);
  const double c2 = std::max(ctot - c1, 0.0);
  const Arc g{&T, lib * 4 + orf, lib * 4 + 2 + orf, slew, vth, vl, vh, der};
  // gateModelRd and setCeffAlgorithm
  const double d1 = nldm(T, g.bd, slew, ctot), d2 = nldm(T, g.bd, slew, ctot + 1.0);
  const double rd = -std::log(vth) * std::fabs(d1 - d2);
  const bool cap_alg = (rd < 1e-5) || (rpi < rd * 1e-3) || (c1 == 0) || (c1 < c2 * 1e-3) || (rpi == 0);
  const bool zc2 = !cap_alg && (c2 < c1 * 1e-3);
  const bool pia = !cap_alg && !zc2;
  const double rd_ = cap_alg ? 1.0 : rd, rpi_ = cap_alg ? 1.0 : rpi;
  const double c1_ = (cap_alg || c1 <= 0) ? 1.0 : c1, c2_ = pia ? c2 : 1.0;
  const double z1 = 1.0 / (rpi_ * c1_);
  const double a = rpi_ * rd_ * c1_ * c2_, b = rd_ * (c1_ + c2_) + rpi_ * c1_;
  const double sq = std::sqrt(std::max(b * b - 4 * a, 0.0));
  const double p1p = (b + sq) / (2 * a), p2p = (b - sq) / (2 * a), p1p2 = p1p * p2p;
  const double k2p = z1 / p1p2, k1p = (1.0 - k2p * (p1p + p2p)) / p1p2;
  const double k4p = (k1p * p1p + k2p) / (p2p - p1p), k3p = -k1p - k4p;
  const double zz = (c1_ + c2_) / (rpi_ * c1_ * c2_);
  const double A = zz / p1p2, B = (zz - p1p) / (p1p * (p1p - p2p)), D = (zz - p2p) / (p2p * (p2p - p1p));
  const double p1z = 1.0 / (c1_ * (rd_ + rpi_)), k0z = p1z / z1, k1z = (p1z - z1) / (p1z * p1z);
  Wave w;
  w.k0 = pia ? 1.0 / (rd_ * c2_) : k0z; w.k1 = pia ? k1p : k1z; w.k2 = pia ? k2p : 1.0 / k0z;
  w.k3 = pia ? k3p : -k1z; w.k4 = pia ? k4p : 0.0; w.p1 = pia ? p1p : p1z; w.p2 = pia ? p2p : 1.0;
  w.p3 = 1.0; w.load = false;
  const double ct = c1 + c2;
  double t0 = 0, dt = 0, cep = 0;
  bool pi_ok = false, zc_ok = false, fz = true;
  if (pia) {  // DmpPi: Newton from c1 + c2, else from c2
    double t0a, dta, cea, t0b = 0, dtb = 0, ceb = 0;
    bool fa, fb = true;
    newton(g, ctot, rd_, ct, A, B, D, p1p, p2p, true, nit, t0a, dta, cea, fa);
    if (fa) newton(g, c2_, rd_, ct, A, B, D, p1p, p2p, true, nit, t0b, dtb, ceb, fb);
    const bool use_b = fa && !fb;
    t0 = use_b ? t0b : t0a; dt = use_b ? dtb : dta; cep = use_b ? ceb : cea;
    pi_ok = !(fa && fb);
  } else if (zc2) {  // DmpZeroC2: order 2 at ceff = c1
    double cez;
    newton(g, c1_, rd_, ct, A, B, D, p1p, p2p, false, nit, t0, dt, cez, fz);
    zc_ok = !fz;
  }
  const double ceff = pi_ok ? cep : (zc2 ? c1 : ctot);
  const double tab_d = nldm(T, g.bd, slew, std::max(ceff, 1e-9));
  const double tab_s = std::max(nldm(T, g.bs, slew, std::max(ceff, 1e-9)), 0.0);
  if (!(dt > 0)) dt = 1.0;
  const double ub = pia ? t0 + dt + (c1 + c2) * (rd_ + rpi_) * 2.0 : t0 + dt + c1 * (rd_ + rpi_) * 2.0;
  w.t0 = t0; w.dt = dt;
  const bool on = pi_ok || zc_ok;  // the elements that use the driver waveform
  bool f1 = true, f2 = true, f3 = true;
  double vo_d = 0, tlo = 0, thi = 0;
  if (on) {
    vo_d = find_root(w, t0, ub, vth, true, rit, f1);
    tlo = find_root(w, t0, vo_d, vl, true, rit, f2);
    thi = find_root(w, vo_d, ub, vh, true, rit, f3);
  } else {
    vo_d = (t0 + ub) * 0.5;
  }
  const bool vo_fail = f1 || f2 || f3;
  const bool wave_ok = (pi_ok || zc_ok) && !vo_fail;
  double delay = (zc_ok && !vo_fail) ? vo_d : tab_d;
  double dslew = wave_ok ? (thi - tlo) / der : tab_s;
  if (zc2 && (fz || vo_fail)) {
    delay = nldm(T, g.bd, slew, std::max(c1_, 1e-9));
    dslew = std::max(nldm(T, g.bs, slew, std::max(c1_, 1e-9)), 0.0);
  }
  delay_o = (float)delay;
  dslew_o = (float)dslew;
  so[0] = wave_ok ? 1.0 : 0.0;
  so[1] = w.k0; so[2] = w.k1; so[3] = w.k2; so[4] = w.k3; so[5] = w.k4;
  so[6] = w.p1; so[7] = w.p2; so[8] = t0; so[9] = dt; so[10] = vo_d;
  so[11] = ub; so[12] = (float)dslew; so[13] = vth; so[14] = vl; so[15] = vh;
  so[16] = der;
}

// dmp_triton._load_dev (MODE 2) + thresholdAdjust for one load: so the state of its arc (gate_one); elm its Elmore
// delay; rise its transition; lthr the load's thresholds. Outputs: wire delay, load slew.
inline void load_one(const double* so, float elm_f, bool rise, const float* lthr, int rit, float& wire_o,
                     float& slew_o) {
  const bool valid = so[0] > 0.5;
  Wave w;
  w.k0 = so[1]; w.k1 = so[2]; w.k2 = so[3]; w.k3 = so[4]; w.k4 = so[5]; w.p1 = so[6]; w.p2 = so[7];
  w.t0 = so[8]; w.dt = so[9]; w.load = true;
  const double vo_d = so[10], ub = so[11], ds = so[12], vth = so[13], vl = so[14], vh = so[15], der = so[16];
  const double elm = elm_f;
  const double lvth = lthr[0], lvl = lthr[1], lvh = lthr[2], lder = lthr[3];
  const bool small = (elm == 0) || (elm < ds * 1e-3) || !valid;
  const double e = small ? 1.0 : elm;
  w.p3 = 1.0 / e;
  const double t_up = ub + e * 2.0;
  double wd = elm, s = ds;
  if (!small) {
    bool f1, f2, f3;
    const double ld = find_root(w, w.t0, t_up, vth, true, rit, f1);
    const double tlo = find_root(w, w.t0, ld, vl, true, rit, f2);
    const double thi = find_root(w, ld, t_up, vh, true, rit, f3);
    if (!(f1 || f2 || f3)) {
      double delay = ld - vo_d;
      if (delay < 0) delay = elm;
      wd = delay;
      s = std::max((thi - tlo) / der, ds);
    }
  }
  // thresholdAdjust to the load's library
  const double dd = vh - vl;
  const double delta = s * ((lvth - vth) / dd);
  wd = wd + (rise ? delta : -delta);
  s = s * (((lvh - lvl) / lder) / (dd / der));
  wire_o = (float)wd;
  slew_o = (float)s;
}

}  // namespace attar_dmp
