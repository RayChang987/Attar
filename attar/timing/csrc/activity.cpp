// Power::evalActivity's BDD path (evalBddDuty, evalBddActivity) on the CPU, one expression entry at a time:
// attar/timing/power.py's _Exprs.activity (the PyTorch reference, used on the GPU) translated line by line, in float32
// as there. Built with g++ on first use (attar/timing/activity_host.py) and called through ctypes; OpenMP over the
// entries when there are many (the register passes evaluate a few entries per level, where threads only cost).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
constexpr int64_t R_PIN = 0, R_SEQ = 1, R_NONE = 2, R_PAD = 3, R_IN = 4, R_TIE = 5;

struct Act {
  const float* pin_u;
  const float* pin_d;
  const float* seq_u;
  const float* seq_d;
  const float* per;
  const uint8_t* clk;
  int64_t P, S;
  float in_density, var_period;  // R_IN variables' density, R_IN / R_TIE variables' period
};

// one entry: its K variables (kind / idx), its truth table tab (2^K), the clock period fallbacks
// (a level's entries are padded to its widest function: trailing R_PAD variables, constant 0, the table tiled over
// them; they change no sum (their rows weigh 0, their factors are the identity) and are left out here)
inline void entry(const Act& a, int64_t K, const int8_t* kind, const int32_t* idx, const float* tab,
                  float min_period, float unit, float* w, float* z, float* out_duty, float* out_dens) {
  while (K > 0 && kind[K - 1] == R_PAD) --K;
  const int64_t R = int64_t(1) << K;
  const float INF = std::numeric_limits<float>::infinity();
  float u[32], d[32], pr[32];
  bool ck[32], pinlike[32];
  bool bad = false;
  for (int64_t j = 0; j < K; ++j) {
    const int64_t kd = kind[j], ix = idx[j];
    const int64_t ip = std::min(ix, a.P - 1), iq = std::min(ix, std::max<int64_t>(a.S - 1, 0));
    if (kd == R_PIN) {
      u[j] = a.pin_u[ip]; d[j] = a.pin_d[ip]; pr[j] = a.per[ip];
    } else if (kd == R_SEQ) {
      u[j] = a.S ? a.seq_u[iq] : 0.0f; d[j] = a.S ? a.seq_d[iq] : 0.0f; pr[j] = INF;
    } else if (kd == R_IN) {
      u[j] = 0.0f + 0.5f; d[j] = 0.0f + a.in_density; pr[j] = 0.0f + a.var_period;
    } else if (kd == R_TIE) {
      u[j] = float(ix); d[j] = 0.0f; pr[j] = 0.0f + a.var_period;
    } else {
      u[j] = 0.0f; d[j] = 0.0f; pr[j] = INF;
    }
    ck[j] = kd == R_PIN && a.clk[ip];
    pinlike[j] = kd == R_PIN || kd == R_IN || kd == R_TIE;
    bad = bad || kd == R_NONE || ck[j];
  }
  if (K == 0) {
    *out_duty = tab[0]; *out_dens = 0.0f;
    return;
  }
  // duty: Power::evalBddDuty's Shannon expansion duty0 (1 - v) + duty1 v per variable (the last first), in double and
  // stored in float at each step (z as scratch)
  auto fold = [&](float* t) {
    for (int64_t j = K - 1; j >= 0; --j) {
      const int64_t h = int64_t(1) << j;
      const double v = u[j];
      for (int64_t r = 0; r < h; ++r) t[r] = float(double(t[r]) * (1.0 - v) + double(t[r + h]) * v);
    }
    return t[0];
  };
  for (int64_t r = 0; r < R; ++r) z[r] = tab[r];
  const float duty = fold(z);
  // change density over the period of the fastest launching clock
  float period = INF;
  for (int64_t j = 0; j < K; ++j)
    if (pinlike[j]) period = std::min(period, pr[j]);
  if (!std::isfinite(period)) period = min_period;
  if (!(period > 0.0f)) period = unit;
  for (int64_t r = 0; r < R; ++r) z[r] = 1.0f - tab[r];
  for (int64_t j = 0; j < K; ++j) {
    const float uc = std::min(std::max(u[j], 0.0f), 1.0f);
    const float half = std::min(d[j] * period, 2.0f * std::min(uc, 1.0f - uc)) / 2.0f;
    const float q00 = 1.0f - uc - half, q01 = half, q11 = uc - half;
    const int64_t b = int64_t(1) << j;
    for (int64_t r = 0; r < R; ++r) {
      if (r & b) continue;
      const float z0 = z[r], z1 = z[r | b];
      z[r] = q00 * z0 + q01 * z1;
      z[r | b] = q01 * z0 + q11 * z1;
    }
  }
  float change = 0.0f;
  for (int64_t r = 0; r < R; ++r) change += tab[r] * z[r];
  change = 2.0f * change;
  if (!bad) {
    *out_duty = duty; *out_dens = change / period;
    return;
  }
  // Boolean difference sum over the pin variables (evalBddActivity: the duty of each difference, as above)
  float dsum = 0.0f;
  for (int64_t j = 0; j < K; ++j) {
    const int64_t b = int64_t(1) << j;
    for (int64_t r = 0; r < R; ++r) w[r] = std::fabs(tab[r] - tab[r ^ b]);
    const float pdiff = fold(w);
    dsum = dsum + (pinlike[j] ? d[j] * pdiff : 0.0f);
  }
  *out_duty = duty; *out_dens = dsum;
}
}  // namespace

extern "C" {
// n entries rows of a group: kind / idx (rows of width W, the first K used), tid (the entry's table in bank (., 2^K));
// the activities of the pins and register states; out duty / density
int attar_activity(int64_t n, int64_t K, int64_t W, const int64_t* rows, const int8_t* kind, const int32_t* idx,
                   const int32_t* tid, const float* bank, const float* pin_u, const float* pin_d, const float* seq_u,
                   const float* seq_d, const float* per, const uint8_t* clk, int64_t P, int64_t S, float in_density,
                   float var_period, float min_period, float unit, float* duty, float* dens, int threads) {
  if (K > 30) return 1;
  const Act a{pin_u, pin_d, seq_u, seq_d, per, clk, P, S, in_density, var_period};
  const int64_t R = int64_t(1) << K;
  const int nt = n >= 4096 ? threads : 1;
#pragma omp parallel num_threads(nt)
  {
    std::vector<float> w(R), z(R);
#pragma omp for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
      const int64_t e = rows[i];
      entry(a, K, kind + e * W, idx + e * W, bank + tid[e] * R, min_period, unit, w.data(), z.data(), duty + i,
            dens + i);
    }
  }
  return 0;
}

// One sweep of the activity propagation over the levels (attar/timing/power.py, Power._graphs' frontier: the
// breadth-first visit of OpenSTA's PropActivityVisitor emulated level by level), the pins enqueued only. Per level:
// its loads (copy their driver's activity), register state pins (their state's), expression drivers (evaluated on the
// activities before the level's drivers are written); each set as set_check (density caps, the 1% change tolerance,
// findActivity kept in fd / fu); the fanout of the changed pins enqueued, registers whose data / clock changed flagged.
int attar_activity_sweep(
    int64_t n_levels, const int64_t* ld_ptr, const int64_t* ld_pin, const int64_t* ld_drv, const int64_t* st_ptr,
    const int64_t* st_pin, const int64_t* fn_ptr, const int64_t* fn_pin, const int64_t* fn_row,
    const int64_t* lv_K, const int64_t* lv_W, const int64_t* lv_kind, const int64_t* lv_idx, const int64_t* lv_tid,
    const int64_t* lv_bank, const int64_t* ed_ptr, const int64_t* ed_src, const int64_t* ed_dst,
    // state
    float* dens, float* duty, int64_t* orig, float* fd, float* fu, const float* sq_d, const float* sq_u,
    const int64_t* sq_o, uint8_t* enq, uint8_t* regs, uint8_t* changed,
    // statics
    const float* cap, const float* prop_cap, const float* lp, const uint8_t* is_clock, const int64_t* clk_of,
    const float* clk_period, const float* clk_duty, const int64_t* const_pin, const int64_t* p_cell,
    const uint8_t* isl, const int64_t* fn_seq, int64_t P, int64_t S, float tol, float in_density, float var_period,
    float min_period, float unit, int threads) {
  constexpr int64_t O_UNKNOWN = 0, O_PROP = 2;
  const Act a{fu, fd, sq_u, sq_d, lp, is_clock, P, S, in_density, var_period};
  auto pct = [](float v, float prev) {
    return prev == 0.0f ? float(v != 0.0f) : std::fabs(v - prev) / std::fabs(prev);
  };
  auto set = [&](int64_t p, float nd, float nu, int64_t no) -> bool {
    nd = std::min(nd, cap[p]);
    if (no == O_PROP) nd = std::min(nd, prop_cap[p]);
    const bool ch = pct(nd, dens[p]) > tol || pct(nu, duty[p]) > tol || no != orig[p];
    dens[p] = nd; duty[p] = nu; orig[p] = no;
    const bool clk = is_clock[p] && no == O_UNKNOWN;
    float d_ = clk ? 2.0f / clk_period[clk_of[p]] : nd, u_ = clk ? clk_duty[clk_of[p]] : nu;
    if (const_pin[p] >= 0) { d_ = 0.0f; u_ = float(const_pin[p]); }
    fd[p] = d_; fu[p] = u_;
    changed[p] = ch;
    return ch;
  };
  std::vector<int64_t> sel;
  std::vector<float> ou, od;
  for (int64_t L = 0; L < n_levels; ++L) {
    for (int64_t i = ld_ptr[L]; i < ld_ptr[L + 1]; ++i) {
      const int64_t p = ld_pin[i];
      if (!enq[p]) continue;
      const int64_t dr = ld_drv[i];
      if (set(p, dens[dr], duty[dr], O_PROP) && isl[p]) regs[p_cell[p]] = 1;
    }
    for (int64_t i = st_ptr[L]; i < st_ptr[L + 1]; ++i) {
      const int64_t p = st_pin[i];
      if (!enq[p]) continue;
      const int64_t s = fn_seq[p];
      set(p, sq_d[s], sq_u[s], sq_o[s]);
    }
    sel.clear();
    for (int64_t i = fn_ptr[L]; i < fn_ptr[L + 1]; ++i)
      if (enq[fn_pin[i]]) sel.push_back(i);
    const int64_t n = int64_t(sel.size());
    if (n) {
      const int64_t K = lv_K[L], W = lv_W[L], R = int64_t(1) << K;
      const int8_t* kind = reinterpret_cast<const int8_t*>(lv_kind[L]);
      const int32_t* idx = reinterpret_cast<const int32_t*>(lv_idx[L]);
      const int32_t* tid = reinterpret_cast<const int32_t*>(lv_tid[L]);
      const float* bank = reinterpret_cast<const float*>(lv_bank[L]);
      ou.resize(n); od.resize(n);
      const int nt = n >= 512 ? int(std::min<int64_t>(threads, n / 256)) : 1;
#pragma omp parallel num_threads(nt)
      {
        std::vector<float> w(R), z(R);
#pragma omp for schedule(static)
        for (int64_t q = 0; q < n; ++q) {
          const int64_t e = fn_row[sel[q]];
          entry(a, K, kind + e * W, idx + e * W, bank + tid[e] * R, min_period, unit, w.data(), z.data(), &ou[q],
                &od[q]);
        }
      }
      for (int64_t q = 0; q < n; ++q) set(fn_pin[sel[q]], od[q], ou[q], O_PROP);
    }
    for (int64_t i = ed_ptr[L]; i < ed_ptr[L + 1]; ++i)
      if (changed[ed_src[i]]) enq[ed_dst[i]] = 1;
  }
  return 0;
}
}  // extern "C"
