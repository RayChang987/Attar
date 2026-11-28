// The cone-limited re-timing on the CPU (attar/timing/incremental.py, CpuSweep): OpenSTA's incremental search order
// (a worklist of driver rows by stage) over the fused GPU run's stages, each recomputed row taking every arc into it
// with dmp_core.h's gate_one / load_one, bit for bit the GPU kernels (dmp_triton.stage_gate_kernel /
// stage_load_kernel, MODE 2) and the timer's load arrivals (Timer._load_arr); arrivals and slews are maxima and minima,
// so the result does not depend on the order. Rows are pin * 2 + transition. The arrays are the caller's (numpy, kept
// alive by it): the index of attar.timing.incremental.Incremental, host copies of the library tables, and the timer's
// state mirrored on the host (arrivals, slews, wire delays per load, recorded delays), which the caller copies back to
// the device for the rows / entries / lanes this reports.
#include <cstring>
#include <limits>
#include <queue>
#include <vector>

#include "dmp_core.h"

using namespace attar_dmp;

namespace {

constexpr float INF = std::numeric_limits<float>::infinity();

struct Engine {
  int64_t P2, NT;
  // per entry (arc and output transition of a stage): Liberty arc, input row (-1: launch), output pin, transition,
  // driven net, launch tag, flags (bit 0: arrival search, bit 1: no load annotation), launch time, launch slew
  const int32_t *lib, *fi, *to, *orf, *net, *tag, *arr;
  const float *t0, *s0;
  const int64_t *in_ptr; const int32_t *in_idx;    // row -> the entries into it
  const int64_t *out_ptr; const int32_t *out_idx;  // row -> the entries that read it
  const int64_t *ln_ptr; const int32_t *ln_idx;    // entry -> its lanes (loads)
  const int32_t *lane_en, *lane_row; const uint8_t *lane_ok;
  const int32_t *row_stage;
  Tables T; const float *thr, *sthr; const int32_t *nto; int64_t lmin; int nit, rit;
  // the state (host mirror)
  float *A, *Am, *S, *Sm, *wbx, *wbn;   // (P2, NT), (P2, NT), (P2,), (P2,), (P2,), (P2,)
  float *dmax, *dmin, *wx, *wn;         // per entry, per entry, per lane, per lane
  const float *pi, *elm;                // (E, 4, 3), (net arcs, 4)
  std::vector<uint8_t> mark;            // per row: queued
  std::vector<uint32_t> stamp;          // per row: written by the current driver row (load rows)
  uint32_t clock = 0;
};

inline bool row_changed(const Engine& E, int64_t r, const float* old) {
  const int64_t NT = E.NT;
  for (int64_t g = 0; g < NT; ++g)
    if (E.A[r * NT + g] != old[g] || E.Am[r * NT + g] != old[NT + g]) return true;
  return E.S[r] != old[2 * NT] || E.Sm[r] != old[2 * NT + 1];
}

inline void save_row(const Engine& E, int64_t r, float* old) {
  const int64_t NT = E.NT;
  for (int64_t g = 0; g < NT; ++g) { old[g] = E.A[r * NT + g]; old[NT + g] = E.Am[r * NT + g]; }
  old[2 * NT] = E.S[r]; old[2 * NT + 1] = E.Sm[r];
}

inline void reset_row(Engine& E, int64_t r) {
  const int64_t NT = E.NT;
  for (int64_t g = 0; g < NT; ++g) { E.A[r * NT + g] = -INF; E.Am[r * NT + g] = INF; }
  E.S[r] = 0.0f; E.Sm[r] = INF;
}

}  // namespace

extern "C" {

void* attar_inc_new(int64_t P2, int64_t NT, const int32_t* lib, const int32_t* fi, const int32_t* to,
                    const int32_t* orf, const int32_t* net, const int32_t* tag, const int32_t* arr, const float* t0,
                    const float* s0, const int64_t* in_ptr, const int32_t* in_idx, const int64_t* out_ptr,
                    const int32_t* out_idx, const int64_t* ln_ptr, const int32_t* ln_idx, const int32_t* lane_en,
                    const int32_t* lane_row, const uint8_t* lane_ok, const int32_t* row_stage, const float* ix1,
                    const float* ix2, const float* val, const int64_t* nn, int64_t K, const float* thr,
                    const float* sthr, const int32_t* nto, int64_t lmin, int nit, int rit) {
  Engine* E = new Engine();
  E->P2 = P2; E->NT = NT;
  E->lib = lib; E->fi = fi; E->to = to; E->orf = orf; E->net = net; E->tag = tag; E->arr = arr; E->t0 = t0;
  E->s0 = s0; E->in_ptr = in_ptr; E->in_idx = in_idx; E->out_ptr = out_ptr; E->out_idx = out_idx;
  E->ln_ptr = ln_ptr; E->ln_idx = ln_idx; E->lane_en = lane_en; E->lane_row = lane_row; E->lane_ok = lane_ok;
  E->row_stage = row_stage;
  E->T = Tables{ix1, ix2, val, nn, K};
  E->thr = thr; E->sthr = sthr; E->nto = nto; E->lmin = lmin; E->nit = nit; E->rit = rit;
  E->mark.assign(P2, 0);
  E->stamp.assign(P2, 0);
  return E;
}

void attar_inc_free(void* h) { delete static_cast<Engine*>(h); }

// the state arrays (they move when the caller reallocates them)
void attar_inc_state(void* h, float* A, float* Am, float* S, float* Sm, float* wbx, float* wbn, float* dmax,
                     float* dmin, float* wx, float* wn, const float* pi, const float* elm) {
  Engine* E = static_cast<Engine*>(h);
  E->A = A; E->Am = Am; E->S = S; E->Sm = Sm; E->wbx = wbx; E->wbn = wbn;
  E->dmax = dmax; E->dmin = dmin; E->wx = wx; E->wn = wn; E->pi = pi; E->elm = elm;
}

// Re-time from the seed rows (the drivers of the nets whose parasitics changed). Writes the rows it recomputed
// (out_rows), the entries (out_ents) and the lanes (out_lanes), each array at most its capacity; returns 0, or 1 when
// more than max_ents entries would be recomputed or an array is full (the state is then partly updated: the caller
// copies it again), or 2 when a row's consumer is not in a later stage (an index the sweep cannot follow).
int attar_inc_run(void* h, const int64_t* seeds, int64_t ns, int64_t max_ents, int32_t* out_rows, int64_t cap_rows,
                  int64_t* n_rows, int32_t* out_ents, int64_t cap_ents, int64_t* n_ents, int32_t* out_lanes,
                  int64_t cap_lanes, int64_t* n_lanes) {
  Engine& E = *static_cast<Engine*>(h);
  const int64_t NT = E.NT;
  using Item = std::pair<int32_t, int32_t>;  // (stage, row)
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> q;
  int64_t nr = 0, ne = 0, nl = 0;
  int status = 0;
  for (int64_t i = 0; i < ns; ++i) {
    const int64_t r = seeds[i];
    if (!E.mark[r] && E.in_ptr[r + 1] > E.in_ptr[r]) { E.mark[r] = 1; q.push({E.row_stage[r], (int32_t)r}); }
  }
  std::vector<int32_t> us;            // the current row's load rows
  std::vector<float> old_r(2 * NT + 2), old_u;
  double st[2][NSTATE];
  while (!q.empty()) {
    const auto [stage, r32] = q.top();
    q.pop();
    const int64_t r = r32;
    E.mark[r] = 0;
    const int64_t e0 = E.in_ptr[r], e1 = E.in_ptr[r + 1];
    if (ne + (e1 - e0) > max_ents || ne + (e1 - e0) > cap_ents) { status = 1; break; }
    // the rows this driver row writes: itself and its loads (each written by this row alone: Incremental's index)
    ++E.clock;
    us.clear();
    for (int64_t k = e0; k < e1; ++k) {
      const int64_t e = E.in_idx[k];
      for (int64_t j = E.ln_ptr[e]; j < E.ln_ptr[e + 1]; ++j) {
        const int32_t u = E.lane_row[E.ln_idx[j]];
        if (E.stamp[u] != E.clock) { E.stamp[u] = E.clock; us.push_back(u); }
      }
    }
    if (nr + 1 + (int64_t)us.size() > cap_rows) { status = 1; break; }
    save_row(E, r, old_r.data());
    old_u.resize(us.size() * (2 * NT + 2));
    for (size_t i = 0; i < us.size(); ++i) {
      save_row(E, us[i], old_u.data() + i * (2 * NT + 2));
      reset_row(E, us[i]);
      E.wbx[us[i]] = -INF; E.wbn[us[i]] = INF;
    }
    reset_row(E, r);
    for (int64_t k = e0; k < e1; ++k) {
      const int64_t e = E.in_idx[k];
      const int64_t f = E.fi[e], o = E.orf[e];
      const bool launch = f < 0;
      const bool search = (E.arr[e] & 1) != 0, nol = (E.arr[e] & 2) != 0;
      for (int mn = 0; mn < 2; ++mn) {
        const int64_t lb = E.lib[e] + (mn ? E.lmin : 0);
        float slew;
        if (launch) {
          slew = E.s0[e];
        } else {
          const float sx = E.S[f], sn = E.Sm[f];
          slew = mn ? (sn < INF ? sn : sx) : sx;
        }
        const float* pb = E.pi + (E.net[e] * 4 + o + 2 * mn) * 3;
        const float cn = pb[0], rr = pb[1], cf = pb[2];
        const float ctot = cn + cf;
        float delay, ds;
        gate_one(E.T, lb, o, slew, ctot, cf, rr, E.thr + (lb * 2 + o) * 4, E.nit, E.rit, delay, ds, st[mn]);
        (mn ? E.dmin : E.dmax)[e] = delay;
        for (int64_t g = 0; g < NT; ++g) {
          if (!mn) {
            float ax = launch ? (E.tag[e] == g ? E.t0[e] : -INF) : E.A[f * NT + g];
            ax = ax + delay;
            if (search && ax > -INF && ax > E.A[r * NT + g]) E.A[r * NT + g] = ax;
          } else {
            float an = launch ? (E.tag[e] == g ? E.t0[e] : INF) : E.Am[f * NT + g];
            an = an + delay;
            if (search && an < INF && an < E.Am[r * NT + g]) E.Am[r * NT + g] = an;
          }
        }
        if (!mn) { if (ds > E.S[r]) E.S[r] = ds; } else { if (ds < E.Sm[r]) E.Sm[r] = ds; }
      }
      for (int64_t j = E.ln_ptr[e]; j < E.ln_ptr[e + 1]; ++j) {
        const int64_t l = E.ln_idx[j];
        const int64_t u = E.lane_row[l], a = E.lane_en[l];
        const float* lthr = E.sthr + (E.nto[a] * 2 + o) * 4;
        for (int mn = 0; mn < 2; ++mn) {
          float w, s;
          load_one(st[mn], E.elm[a * 4 + o + 2 * mn], o == 0, lthr, E.rit, w, s);
          (mn ? E.wn : E.wx)[l] = w;
          if (!mn) {
            if (!nol && s > E.S[u]) E.S[u] = s;
            if (E.lane_ok[l] && w > E.wbx[u]) E.wbx[u] = w;
          } else {
            if (!nol && s < E.Sm[u]) E.Sm[u] = s;
            if (E.lane_ok[l] && w < E.wbn[u]) E.wbn[u] = w;
          }
        }
        if (nl >= cap_lanes) { status = 1; break; }
        out_lanes[nl++] = (int32_t)l;
      }
      if (status) break;
      out_ents[ne++] = (int32_t)e;
    }
    if (status) break;
    // the loads' arrivals: the driver's + the largest wire delay of its arcs (Timer._load_arr)
    for (const int32_t u : us)
      for (int64_t g = 0; g < NT; ++g) {
        const float ax = E.A[r * NT + g] + E.wbx[u], an = E.Am[r * NT + g] + E.wbn[u];
        if (ax > E.A[u * NT + g]) E.A[u * NT + g] = ax;
        if (an < E.Am[u * NT + g]) E.Am[u * NT + g] = an;
      }
    // smin falls back to smax (Timer); the changed rows queue their readers' rows
    auto finish = [&](int64_t x, const float* old) -> bool {
      if (!std::isfinite(E.Sm[x])) E.Sm[x] = E.S[x];
      out_rows[nr++] = (int32_t)x;
      if (!row_changed(E, x, old)) return true;
      for (int64_t k = E.out_ptr[x]; k < E.out_ptr[x + 1]; ++k) {
        const int64_t e = E.out_idx[k];
        const int64_t t = (int64_t)E.to[e] * 2 + E.orf[e];
        if (E.mark[t]) continue;
        if (E.row_stage[t] <= stage) return false;
        E.mark[t] = 1;
        q.push({E.row_stage[t], (int32_t)t});
      }
      return true;
    };
    bool ok = finish(r, old_r.data());
    for (size_t i = 0; ok && i < us.size(); ++i) ok = finish(us[i], old_u.data() + i * (2 * NT + 2));
    if (!ok) { status = 2; break; }
  }
  while (!q.empty()) { E.mark[q.top().second] = 0; q.pop(); }
  *n_rows = nr; *n_ents = ne; *n_lanes = nl;
  return status;
}

}  // extern "C"
