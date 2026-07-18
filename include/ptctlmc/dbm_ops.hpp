// dbm_ops.hpp — DBM zone operations as __host__ __device__ free functions.
//
// The zone algebra lives here as plain functions over a raw int array so the exact
// same code runs on the CPU (validated by unit tests) and inside CUDA kernels. A
// zone is `raw_t d[dim*dim]`, dim = nClocks+1, row-major, entry [i*dim+j] = bound on
// x_i - x_j, encoded `raw = (value<<1) | weak` (weak=1 for '<=', 0 for '<').
#pragma once

#include <cstdint>

namespace ptctlmc {

#if defined(__CUDACC__)
#define PTCTLMC_HD __host__ __device__
#else
#define PTCTLMC_HD
#endif

using raw_t = std::int32_t;
constexpr raw_t kInf = static_cast<raw_t>(1) << 30;  // "< infinity"
constexpr raw_t kZeroLe = 1;                         // le(0): <= 0

PTCTLMC_HD inline raw_t le(std::int32_t v) { return (v << 1) | 1; }
PTCTLMC_HD inline raw_t lt(std::int32_t v) { return (v << 1); }
PTCTLMC_HD inline std::int32_t raw_bound(raw_t r) { return r >> 1; }
PTCTLMC_HD inline bool raw_is_strict(raw_t r) { return (r & 1) == 0; }

PTCTLMC_HD inline raw_t raw_add(raw_t a, raw_t b) {
  if (a >= kInf || b >= kInf) return kInf;
  const std::int32_t v = raw_bound(a) + raw_bound(b);
  const bool weak = !raw_is_strict(a) && !raw_is_strict(b);
  return weak ? le(v) : lt(v);
}

// Canonicalise (all-pairs shortest paths). O(dim^3).
PTCTLMC_HD inline void dbm_close(raw_t* d, int dim) {
  for (int k = 0; k < dim; ++k)
    for (int i = 0; i < dim; ++i) {
      const raw_t ik = d[i * dim + k];
      if (ik >= kInf) continue;
      for (int j = 0; j < dim; ++j) {
        const raw_t s = raw_add(ik, d[k * dim + j]);
        if (s < d[i * dim + j]) d[i * dim + j] = s;
      }
    }
}

PTCTLMC_HD inline bool dbm_empty(const raw_t* d, int dim) {
  for (int i = 0; i < dim; ++i)
    if (d[i * dim + i] < kZeroLe) return true;
  return false;
}

PTCTLMC_HD inline void dbm_delay(raw_t* d, int dim) {
  for (int i = 1; i < dim; ++i) d[i * dim + 0] = kInf;  // drop upper bounds
}

PTCTLMC_HD inline void dbm_reset(raw_t* d, int dim, int k) {
  for (int j = 0; j < dim; ++j) {
    d[k * dim + j] = d[0 * dim + j];
    d[j * dim + k] = d[j * dim + 0];
  }
  d[k * dim + k] = kZeroLe;
}

// Conjoin  x_a - x_b <| bound. Returns false if the zone became empty. Uses
// *incremental* close (O(dim^2)): valid because the DBM is canonical on entry and
// only the single edge (a,b) is tightened.
PTCTLMC_HD inline bool dbm_constrain(raw_t* d, int dim, int a, int b, raw_t bound) {
  if (bound >= d[a * dim + b]) return !dbm_empty(d, dim);  // not tighter
  d[a * dim + b] = bound;
  for (int i = 0; i < dim; ++i) {                          // shortest i->b via a
    const raw_t via = raw_add(d[i * dim + a], bound);
    if (via < d[i * dim + b]) d[i * dim + b] = via;
  }
  for (int i = 0; i < dim; ++i) {                          // propagate through b
    const raw_t ib = d[i * dim + b];
    if (ib >= kInf) continue;
    const raw_t* brow = d + b * dim;
    raw_t* irow = d + i * dim;
    for (int k = 0; k < dim; ++k) {
      const raw_t via = raw_add(ib, brow[k]);
      if (via < irow[k]) irow[k] = via;
    }
  }
  return !dbm_empty(d, dim);
}

PTCTLMC_HD inline void dbm_intersect(raw_t* a, const raw_t* b, int dim) {
  for (int i = 0; i < dim * dim; ++i)
    if (b[i] < a[i]) a[i] = b[i];
  dbm_close(a, dim);
}

// Extra_M extrapolation (maxc[0]=0): widen for a finite zone graph.
PTCTLMC_HD inline void dbm_extrapolate(raw_t* d, int dim, const std::int32_t* maxc) {
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j) {
      if (i == j) continue;
      raw_t e = d[i * dim + j];
      if (e >= kInf) continue;
      if (e > le(maxc[i])) d[i * dim + j] = kInf;
      else if (e < lt(-maxc[j])) d[i * dim + j] = lt(-maxc[j]);
    }
  dbm_close(d, dim);
}

// Extra_LU extrapolation (Behrmann et al.): L[i] = max lower-bound constant for
// clock i, U[j] = max upper-bound constant for clock j (both 0 for the reference
// clock 0). Strictly coarser than Extra_M => fewer symbolic states, still sound.
PTCTLMC_HD inline void dbm_extrapolate_lu(raw_t* d, int dim, const std::int32_t* L,
                                          const std::int32_t* U) {
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j) {
      if (i == j) continue;
      raw_t e = d[i * dim + j];
      if (e >= kInf) continue;
      if (e > le(L[i])) d[i * dim + j] = kInf;
      else if (e < lt(-U[j])) d[i * dim + j] = lt(-U[j]);
    }
  dbm_close(d, dim);
}

constexpr int kDbmMaxDim = 64;

// Extra_LU+ : the coarser (+) extrapolation UPPAAL uses by default. Beyond Extra_LU,
// it relaxes an entry to infinity when the lower bound of clock i (D[0][i]) already
// exceeds L_i, or (for i>0) the lower bound of clock j exceeds U_j. Sound; strictly
// fewer symbolic states than Extra_LU. Falls back to Extra_LU if dim > kDbmMaxDim.
PTCTLMC_HD inline void dbm_extrapolate_lu_plus(raw_t* d, int dim, const std::int32_t* L,
                                               const std::int32_t* U) {
  if (dim > kDbmMaxDim) { dbm_extrapolate_lu(d, dim, L, U); return; }
  raw_t row0[kDbmMaxDim];
  for (int k = 0; k < dim; ++k) row0[k] = d[k];  // D[0][k], snapshot before modifying
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j) {
      if (i == j) continue;
      const raw_t e = d[i * dim + j];
      if (e >= kInf) continue;
      if (raw_bound(e) > L[i]) { d[i * dim + j] = kInf; continue; }
      if (-raw_bound(row0[i]) > L[i]) { d[i * dim + j] = kInf; continue; }
      if (i != 0 && -raw_bound(row0[j]) > U[j]) { d[i * dim + j] = kInf; continue; }
      if (e < lt(-U[j])) d[i * dim + j] = lt(-U[j]);
    }
  dbm_close(d, dim);
}

// Diagonal-safe extrapolation: Extra_LU+ applied ONLY to the individual clock bounds
// (entries touching the reference clock 0), leaving every diagonal x_i - x_j (i,j>0)
// EXACT. Relaxing a diagonal is what makes LU/M unsound on automata with diagonal
// guards (Bouyer 2004); keeping diagonals exact means every diagonal guard is tested
// on true values (no spurious reachability), while clock relaxation still bounds the
// zone graph. Sound for TA with diagonal constraints.
PTCTLMC_HD inline void dbm_extrapolate_lu_plus_nodiag(raw_t* d, int dim, const std::int32_t* L,
                                                      const std::int32_t* U) {
  if (dim > kDbmMaxDim) return;  // caller keeps zones exact (rare)
  raw_t row0[kDbmMaxDim];
  for (int k = 0; k < dim; ++k) row0[k] = d[k];
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j) {
      if (i == j || (i > 0 && j > 0)) continue;  // skip diagonals: keep them exact
      const raw_t e = d[i * dim + j];
      if (e >= kInf) continue;
      if (raw_bound(e) > L[i]) { d[i * dim + j] = kInf; continue; }
      if (-raw_bound(row0[i]) > L[i]) { d[i * dim + j] = kInf; continue; }
      if (i != 0 && -raw_bound(row0[j]) > U[j]) { d[i * dim + j] = kInf; continue; }
      if (e < lt(-U[j])) d[i * dim + j] = lt(-U[j]);
    }
  dbm_close(d, dim);
}

// a ⊇ b  (both canonical): every valuation of b is in a.
PTCTLMC_HD inline bool dbm_includes(const raw_t* a, const raw_t* b, int dim) {
  for (int i = 0; i < dim * dim; ++i)
    if (b[i] > a[i]) return false;
  return true;
}

// [Z] ⊆ a_{≼LU}([Z']) : is zone Z simulated (LU-preorder) by stored zone Z'? This is
// UPPAAL/tchecker's abstraction-based subsumption — strictly COARSER than inclusion, so
// it prunes many more zones (the "better abstractions" of Herbreteau-Srivathsan-
// Walukiewicz, LICS'12). L[i]/U[i] = max lower/upper guard constant per clock, with
// L[0]=U[0]=0 (reference clock) MANDATORY. Line-for-line port of tchecker is_alu_le,
// brute-force validated exact against the point-wise definition. If it returns true,
// exploring Z discovers nothing new beyond Z' — Z can be discarded.
constexpr std::int32_t kAluNegInf = -(1 << 29);  // "clock has no lower/upper guard" (−∞)
PTCTLMC_HD inline bool dbm_is_alu_le(const raw_t* Z, const raw_t* Zp, int dim,
                                     const std::int32_t* L, const std::int32_t* U) {
  for (int x = 0; x < dim; ++x) {
    if (U[x] == kAluNegInf) continue;          // U_x = −∞: condition (1) unsatisfiable
    if (Z[0 * dim + x] < le(-U[x])) continue;  // (1) need Z[0][x] >= (<=, -U_x)
    for (int y = 0; y < dim; ++y) {
      if (x == y) continue;
      if (L[y] == kAluNegInf) continue;        // L_y = −∞: condition (3) unsatisfiable
      if (Zp[y * dim + x] < Z[y * dim + x] &&                       // (2) Z'[y][x] < Z[y][x]
          raw_add(Zp[y * dim + x], lt(-L[y])) < Z[0 * dim + x])     // (3)
        return false;  // witness -> Z NOT subsumed
    }
  }
  return true;
}

// If a ∪ b is EXACTLY convex, set a := their convex hull (a single zone) and return
// true; else leave a unchanged and return false. Sound (no spurious points) —
// this is what makes the reachable zone at a discrete state collapse to one zone,
// matching UPPAAL's stored-state count. Host-only (uses dim^2 scratch).
inline bool dbm_try_union(raw_t* a, const raw_t* b, int dim) {
  if (dim > kDbmMaxDim) return false;
  raw_t H[kDbmMaxDim * kDbmMaxDim], R[kDbmMaxDim * kDbmMaxDim];
  for (int k = 0; k < dim * dim; ++k) H[k] = a[k] > b[k] ? a[k] : b[k];  // convex hull (loosest)
  // For each zone z in {a,b}: every facet of z that is strictly inside H must,
  // when crossed (H ∧ ¬facet), land entirely inside the other zone.
  for (int pass = 0; pass < 2; ++pass) {
    const raw_t* z = (pass == 0) ? a : b;
    const raw_t* other = (pass == 0) ? b : a;
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j) {
        if (i == j || z[i * dim + j] >= H[i * dim + j]) continue;  // not a strict facet
        for (int k = 0; k < dim * dim; ++k) R[k] = H[k];
        const int v = raw_bound(z[i * dim + j]);
        const raw_t neg = raw_is_strict(z[i * dim + j]) ? le(-v) : lt(-v);  // ¬(x_i-x_j ~ v)
        if (neg < R[j * dim + i]) { R[j * dim + i] = neg; dbm_close(R, dim); }
        if (dbm_empty(R, dim)) continue;      // crossing region empty -> fine
        if (!dbm_includes(other, R, dim)) return false;  // region escapes the other zone
      }
  }
  for (int k = 0; k < dim * dim; ++k) a[k] = H[k];
  dbm_close(a, dim);
  return true;
}

}  // namespace ptctlmc
