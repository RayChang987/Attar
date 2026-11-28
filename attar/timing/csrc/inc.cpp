// The cone-limited re-timing on the CPU (attar/timing/cpu_inc.py, CpuEngine): OpenSTA's incremental search order (a
// worklist of driver rows by level key) over the timer's arcs, each recomputed row taking every arc into it with
// dmp_core.h's gate_one / load_one, bit for bit the GPU kernels (dmp_triton.stage_gate_kernel / stage_load_kernel,
// MODE 2) and the timer's load arrivals (Timer._load_arr); arrivals and slews are maxima and minima, so the result does
// not depend on the order. The engine numbers pins by its own handles, which netlist edits do not renumber (a row is
// handle * 2 + transition); its arcs and adjacency lists grow in place (a base CSR plus linked overflow nodes, dead
// flags for removed arcs). Every array is the caller's (numpy, with spare capacity), passed in State on each call.
#include <cstring>
#include <limits>
#include <queue>
#include <vector>

#include "dmp_core.h"

using namespace attar_dmp;

namespace {

constexpr float INF = std::numeric_limits<float>::infinity();

}  // namespace

extern "C" {

struct State {
  int64_t NT, lmin, nit, rit, K;
  // entries (an arc and its output transition): Liberty arc, input row (-1: launch), output pin handle, transition,
  // driven net, launch tag, flags (bit 0: arrival search, bit 1: no load annotation), launch time and slew, alive
  const int32_t *lib, *fi, *to, *orf, *net, *tag, *arr;
  const float *t0, *s0;
  const uint8_t* e_alive;
  // lanes (an entry's load): its entry, load row, sink pin handle, annotates (ok), alive
  const int32_t *l_ent, *l_u, *l_sk;
  const uint8_t *l_ok, *l_alive;
  // lists: row -> entries into it, row -> entries reading it, entry -> lanes: base CSR over the first n_*0 rows /
  // entries, then a linked list of overflow nodes (head per row / entry, -1: none)
  int64_t R0, E0;
  const int64_t *in_ptr, *out_ptr, *ln_ptr;
  const int32_t *in_idx, *out_idx, *ln_idx;
  const int32_t *in_head, *out_head, *ln_head;
  const int32_t *o_val, *o_next;
  const double* key;      // per row: its level key (driver rows; a reader's key is larger)
  // library
  const float *ix1, *ix2, *val, *thr;
  const int64_t* nn;
  // per pin handle: the Elmore delays of its net arc (4), its thresholds as a sink (2 x 4); per net: pi models (12)
  const float *elm, *sthr, *pi;
  // the state
  float *A, *Am, *S, *Sm, *wbx, *wbn;   // per row (A, Am: NT each)
  float *dmax, *dmin, *wx, *wn;         // per entry, per lane
  uint8_t* mark;                        // per row: queued (all clear between calls)
  uint32_t* stamp;                      // per row: a load row of the current driver row
  uint32_t clock;
};

}  // extern "C"

namespace {

template <class F>
inline void each(int64_t i, int64_t n0, const int64_t* ptr, const int32_t* idx, const int32_t* head, const State& E,
                 F f) {
  if (i < n0)
    for (int64_t k = ptr[i]; k < ptr[i + 1]; ++k) f(idx[k]);
  for (int32_t n = head[i]; n >= 0; n = E.o_next[n]) f(E.o_val[n]);
}

inline bool row_changed(const State& E, int64_t r, const float* old) {
  const int64_t NT = E.NT;
  for (int64_t g = 0; g < NT; ++g)
    if (E.A[r * NT + g] != old[g] || E.Am[r * NT + g] != old[NT + g]) return true;
  return E.S[r] != old[2 * NT] || E.Sm[r] != old[2 * NT + 1];
}

inline void save_row(const State& E, int64_t r, float* old) {
  const int64_t NT = E.NT;
  for (int64_t g = 0; g < NT; ++g) { old[g] = E.A[r * NT + g]; old[NT + g] = E.Am[r * NT + g]; }
  old[2 * NT] = E.S[r]; old[2 * NT + 1] = E.Sm[r];
}

inline void reset_row(State& E, int64_t r) {
  const int64_t NT = E.NT;
  for (int64_t g = 0; g < NT; ++g) { E.A[r * NT + g] = -INF; E.Am[r * NT + g] = INF; }
  E.S[r] = 0.0f; E.Sm[r] = INF;
}

}  // namespace

extern "C" {

// Re-time from the seed rows (the drivers of the nets whose parasitics changed). Writes the rows it recomputed
// (out_rows), the entries (out_ents) and the lanes (out_lanes), each at most its capacity; returns 0, or 1 when more
// than max_ents entries would be recomputed or an array is full (the state is then partly updated: the caller copies
// it again), or 2 when a row's reader does not have a larger key (an order the sweep cannot follow).
int attar_inc_run(State* sp, const int64_t* seeds, int64_t ns, int64_t max_ents, int32_t* out_rows, int64_t cap_rows,
                  int64_t* n_rows, int32_t* out_ents, int64_t cap_ents, int64_t* n_ents, int32_t* out_lanes,
                  int64_t cap_lanes, int64_t* n_lanes) {
  State& E = *sp;
  const int64_t NT = E.NT;
  const Tables T{E.ix1, E.ix2, E.val, E.nn, E.K};
  using Item = std::pair<double, int64_t>;  // (key, row)
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> q;
  int64_t nr = 0, ne = 0, nl = 0;
  int status = 0;
  for (int64_t i = 0; i < ns; ++i) {
    const int64_t r = seeds[i];
    if (!E.mark[r]) { E.mark[r] = 1; q.push({E.key[r], r}); }
  }
  std::vector<int64_t> us, ents;      // the current row's load rows and entries
  std::vector<float> old_r(2 * NT + 2), old_u;
  double st[2][NSTATE];
  while (!q.empty()) {
    const auto [key, r] = q.top();
    q.pop();
    E.mark[r] = 0;
    ents.clear();
    each(r, E.R0, E.in_ptr, E.in_idx, E.in_head, E, [&](int64_t e) { if (E.e_alive[e]) ents.push_back(e); });
    if (ents.empty()) continue;
    if (ne + (int64_t)ents.size() > max_ents || ne + (int64_t)ents.size() > cap_ents) { status = 1; break; }
    // the rows this driver row writes: itself and its loads (each written by this row alone)
    ++E.clock;
    us.clear();
    for (const int64_t e : ents)
      each(e, E.E0, E.ln_ptr, E.ln_idx, E.ln_head, E, [&](int64_t l) {
        if (!E.l_alive[l]) return;
        const int32_t u = E.l_u[l];
        if (E.stamp[u] != E.clock) { E.stamp[u] = E.clock; us.push_back(u); }
      });
    if (nr + 1 + (int64_t)us.size() > cap_rows) { status = 1; break; }
    save_row(E, r, old_r.data());
    old_u.resize(us.size() * (2 * NT + 2));
    for (size_t i = 0; i < us.size(); ++i) {
      save_row(E, us[i], old_u.data() + i * (2 * NT + 2));
      reset_row(E, us[i]);
      E.wbx[us[i]] = -INF; E.wbn[us[i]] = INF;
    }
    reset_row(E, r);
    for (const int64_t e : ents) {
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
        gate_one(T, lb, o, slew, ctot, cf, rr, E.thr + (lb * 2 + o) * 4, (int)E.nit, (int)E.rit, delay, ds, st[mn]);
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
      each(e, E.E0, E.ln_ptr, E.ln_idx, E.ln_head, E, [&](int64_t l) {
        if (!E.l_alive[l] || status) return;
        const int64_t u = E.l_u[l], sk = E.l_sk[l];
        const float* lthr = E.sthr + (sk * 2 + o) * 4;
        for (int mn = 0; mn < 2; ++mn) {
          float w, s;
          load_one(st[mn], E.elm[sk * 4 + o + 2 * mn], o == 0, lthr, (int)E.rit, w, s);
          (mn ? E.wn : E.wx)[l] = w;
          if (!mn) {
            if (!nol && s > E.S[u]) E.S[u] = s;
            if (E.l_ok[l] && w > E.wbx[u]) E.wbx[u] = w;
          } else {
            if (!nol && s < E.Sm[u]) E.Sm[u] = s;
            if (E.l_ok[l] && w < E.wbn[u]) E.wbn[u] = w;
          }
        }
        if (nl >= cap_lanes) { status = 1; return; }
        out_lanes[nl++] = (int32_t)l;
      });
      if (status) break;
      out_ents[ne++] = (int32_t)e;
    }
    if (status) break;
    // the loads' arrivals: the driver's + the largest wire delay of its arcs (Timer._load_arr)
    for (const int64_t u : us)
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
      bool ok = true;
      each(x, E.R0, E.out_ptr, E.out_idx, E.out_head, E, [&](int64_t e) {
        if (!ok || !E.e_alive[e]) return;
        const int64_t t = (int64_t)E.to[e] * 2 + E.orf[e];
        if (E.mark[t]) return;
        if (!(E.key[t] > key)) { ok = false; return; }
        E.mark[t] = 1;
        q.push({E.key[t], t});
      });
      return ok;
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
