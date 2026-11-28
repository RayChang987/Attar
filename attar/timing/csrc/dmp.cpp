// The CPU timer's delay calculation batches (attar/timing/dmp_host.py): dmp_core.h's gate_one / load_one over the
// elements, OpenMP. Built with g++ on first use and called through ctypes.
#include "dmp_core.h"

using namespace attar_dmp;

extern "C" {

// dmp_triton.dmp_gate_kernel (MODE 2) for N arcs: tables (A, 4, K) ix1, ix2, (A, 4, K, K) val, (A, 4, 2) nn the
// sizes; thr (N, 4): output threshold, slew lower, slew upper, slew derate. Outputs: delay (N,), driver slew (N,),
// state (N, 17) float64 as attar.timing.sta.STATE.
int attar_dmp_gate(int64_t N, const float* ix1, const float* ix2, const float* val, const int64_t* nn, int64_t K,
                   const int64_t* lib_p, const int64_t* orf_p, const float* slew_p, const float* ctot_p,
                   const float* cfar_p, const float* rpi_p, const float* thr_p, float* delay_o, float* dslew_o,
                   double* state_o, int nit, int rit, int threads) {
  const Tables T{ix1, ix2, val, nn, K};
#pragma omp parallel for schedule(dynamic, 256) num_threads(threads)
  for (int64_t i = 0; i < N; ++i)
    gate_one(T, lib_p[i], orf_p[i], slew_p[i], ctot_p[i], cfar_p[i], rpi_p[i], thr_p + i * 4, nit, rit, delay_o[i],
             dslew_o[i], state_o + i * NSTATE);
  return 0;
}

// dmp_triton.dmp_load_kernel (MODE 2: _load_core + thresholdAdjust) for N loads: state (N, 17) float64 rows of their
// arcs, elmore (N,), rise (N,) uint8, lthr (N, 4) the loads' thresholds. Outputs wire delay (N,), load slew (N,).
int attar_dmp_load(int64_t N, const double* state, const float* elm_p, const uint8_t* rise_p, const float* lthr_p,
                   float* wire_o, float* slew_o, int rit, int threads) {
#pragma omp parallel for schedule(dynamic, 256) num_threads(threads)
  for (int64_t i = 0; i < N; ++i)
    load_one(state + i * NSTATE, elm_p[i], rise_p[i] != 0, lthr_p + i * 4, rit, wire_o[i], slew_o[i]);
  return 0;
}

}  // extern "C"
