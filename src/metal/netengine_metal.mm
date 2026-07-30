// netengine_metal.mm — Metal GPU successor generation for NETWORKS (stage 1).
//
// Peer of engine_metal.mm (single TA), but for a network of timed automata with shared int
// variables. One GPU thread per (frontier-state, process, location-edge) runs the exact
// successors_net pipeline for the non-diagonal / channel-free path; the host does the
// subsumption merge. The MSL DBM ops are the same line-for-line port of dbm_ops.hpp used by
// engine_metal.mm, so the GPU runs validated algebra.
//
// Only compiled when PTCTLMC_HAVE_METAL is defined (see CMake).
#ifdef PTCTLMC_HAVE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ptctlmc/dbm_ops.hpp"
#include "ptctlmc/net_gpu.hpp"
#include "ptctlmc/net_gpu_encode.hpp"
#include "ptctlmc/netreach.hpp"   // gen_ok / gen_apply — shared with the CPU path
#include "ptctlmc/network.hpp"

#ifdef PTCTLMC_HAVE_OPENMP
#include <omp.h>
#endif

namespace ptctlmc {
namespace {

// The kernels' private arrays are sized by PT_DIM -- the model's REAL dimension -- so this is
// only an eligibility bound now, not a per-thread cost. It used to be 24, which turned the GPU
// off for arbiter24 and up (24 clocks) and silently dropped them onto the CPU path. Matched to
// HDBM's kDbmMaxDim, which bounds dbm_try_union's host scratch.
constexpr int kMaxDim = 64;
constexpr int kMaxGroup = 512;  // the kernel holds no per-|G| state; this is just sanity

#define PTCTLMC_MTL_CHECK(cond, msg) \
  do { if (!(cond)) throw std::runtime_error(std::string("Metal(net): ") + (msg)); } while (0)

struct NetParams {
  std::int32_t dim, nprocs, nintvars, nclocks, nwork, qbits, ng;
};

// Engine-internal phase timers (PTCTLMC_MTLPROF). The driver's `t_expand` covers the WHOLE
// expand() call, which is host recipe enumeration + buffer staging + the kernel; those three
// have completely different fixes, so they are attributed separately here and dumped once at
// destruction. Off (and free) unless the env var is set.
struct EngProf {
  bool on = false;
  double enumerate = 0, stage = 0, kernel = 0, mstage = 0, mkernel = 0;
  long long levels = 0;
  ~EngProf() {
    if (!on) return;
    std::fprintf(stderr,
        "[engprof] expand: enumerate=%.0fms stage=%.0fms kernel=%.0fms | "
        "merge: stage=%.0fms kernel=%.0fms  (%lld levels)\n",
        enumerate, stage, kernel, mstage, mkernel, levels);
  }
};
static EngProf g_eprof;
struct EngClock {
  static double now() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
  }
};

// ---- MSL: DBM ops (port of dbm_ops.hpp) + network successor kernel -------------------
static NSString* const kNetSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

// SPECIALIZED AT PIPELINE BUILD. The host compiles this source once per network with
// -DPT_DIM/-DPT_NP/-DPT_NI/-DPT_NC set to that network's real sizes, so every private array is
// exactly as big as the model needs instead of the 24-clock worst case. It matters a lot: a
// thread's zone was `int z[24*24]` = 2304 B whatever the model, and net_merge held FOUR such
// arrays (zn, zw, plus try_union's H and R) = 9 KB of thread-local DEVICE memory per lane.
// fischer11 (dim 12) needs 576 B, a 4x cut in the spill footprint, and every loop bound becomes
// a compile-time constant so the index arithmetic strength-reduces and the small loops unroll.
// The fallbacks keep the source standalone-compilable (e.g. for a Metal debugger session).
#ifndef PT_DIM
#define PT_DIM 24
#endif
#ifndef PT_NP
#define PT_NP 64
#endif
#ifndef PT_NI
#define PT_NI 32
#endif
#ifndef PT_NC
#define PT_NC (PT_DIM - 1)
#endif
#define MAXDIM PT_DIM
#define MAXPROC PT_NP
#define MAXINT 32
constant int kInf = 1 << 30;
constant int kZeroLe = 1;

inline int r_le(int v)     { return (v << 1) | 1; }
inline int r_lt(int v)     { return (v << 1); }
inline int r_bound(int r)  { return r >> 1; }
inline bool r_strict(int r){ return (r & 1) == 0; }
inline int r_add(int a, int b) {
  if (a >= kInf || b >= kInf) return kInf;
  int v = r_bound(a) + r_bound(b);
  bool weak = !r_strict(a) && !r_strict(b);
  return weak ? r_le(v) : r_lt(v);
}
inline bool d_empty(thread const int* d, int dim) {
  for (int i = 0; i < dim; ++i) if (d[i*dim+i] < kZeroLe) return true;
  return false;
}
inline void d_delay(thread int* d, int dim) { for (int i = 1; i < dim; ++i) d[i*dim+0] = kInf; }
inline void d_reset(thread int* d, int dim, int k) {
  for (int j = 0; j < dim; ++j) { d[k*dim+j] = d[0*dim+j]; d[j*dim+k] = d[j*dim+0]; }
  d[k*dim+k] = kZeroLe;
}
inline bool d_constrain(thread int* d, int dim, int a, int b, int bound) {
  // NO-OP FAST PATH. If the bound does not tighten anything the zone is bit-for-bit unchanged,
  // so it is still non-empty -- every caller in the successor pipeline holds a non-empty zone
  // (the frontier zone is non-empty, and every step that could empty it returns false and
  // aborts). Scanning the diagonal here cost O(dim) device-memory reads for nothing, on the
  // MOST COMMON outcome: most guards and invariants are already implied by the incoming zone.
  if (bound >= d[a*dim+b]) return true;
  d[a*dim+b] = bound;
  for (int i = 0; i < dim; ++i) { int via = r_add(d[i*dim+a], bound); if (via < d[i*dim+b]) d[i*dim+b] = via; }
  for (int i = 0; i < dim; ++i) {
    int ib = d[i*dim+b]; if (ib >= kInf) continue;
    for (int k = 0; k < dim; ++k) { int via = r_add(ib, d[b*dim+k]); if (via < d[i*dim+k]) d[i*dim+k] = via; }
  }
  return !d_empty(d, dim);
}

// Combined invariant of the *target* location vector: every process's current-location
// invariant intersected in. Returns false on emptiness (mirrors all_invariants + apply_all).
inline bool apply_inv(thread int* z, int dim, int nprocs, thread const int* nlocs,
                      device const int* locbase, device const int* inv_off,
                      device const int* inv_data) {
  for (int q = 0; q < nprocs; ++q) {
    int Lq = locbase[q] + nlocs[q];
    for (int c = inv_off[Lq]; c < inv_off[Lq+1]; ++c)
      if (!d_constrain(z, dim, inv_data[3*c], inv_data[3*c+1], inv_data[3*c+2])) return false;
  }
  return true;
}

// ---- threadgroup-resident variants of the expand ops ------------------------------
// net_expand keeps its zone in `int z[MAXDIM*MAXDIM]` = 2304 B of PRIVATE array. A thread
// has on the order of 128-256 registers, so that array cannot live in registers: it spills
// to thread-local DEVICE memory, and every one of the ~2600 DBM ops per successor then runs
// at device-memory latency. Measured consequence: 100% of lanes survive to do full work on
// viking15 (so divergence is NOT the problem) yet the GPU still runs at ~0.5% of compute
// peak. These variants take the zone in THREADGROUP memory (on-chip) instead, sized by the
// real dim rather than MAXDIM. Bodies are identical to the d_* versions.
inline bool g_empty(threadgroup const int* d, int dim) {
  for (int i = 0; i < dim; ++i) if (d[i*dim+i] < kZeroLe) return true;
  return false;
}
inline void g_delay(threadgroup int* d, int dim) { for (int i = 1; i < dim; ++i) d[i*dim+0] = kInf; }
inline void g_reset(threadgroup int* d, int dim, int k) {
  for (int j = 0; j < dim; ++j) { d[k*dim+j] = d[0*dim+j]; d[j*dim+k] = d[j*dim+0]; }
  d[k*dim+k] = kZeroLe;
}
inline bool g_constrain(threadgroup int* d, int dim, int a, int b, int bound) {
  if (bound >= d[a*dim+b]) return !g_empty(d, dim);
  d[a*dim+b] = bound;
  for (int i = 0; i < dim; ++i) { int via = r_add(d[i*dim+a], bound); if (via < d[i*dim+b]) d[i*dim+b] = via; }
  for (int i = 0; i < dim; ++i) {
    int ib = d[i*dim+b]; if (ib >= kInf) continue;
    for (int k = 0; k < dim; ++k) { int via = r_add(ib, d[b*dim+k]); if (via < d[i*dim+k]) d[i*dim+k] = via; }
  }
  return !g_empty(d, dim);
}
inline bool g_apply_inv(threadgroup int* z, int dim, int nprocs, thread const int* nlocs,
                        device const int* locbase, device const int* inv_off,
                        device const int* inv_data) {
  for (int q = 0; q < nprocs; ++q) {
    int Lq = locbase[q] + nlocs[q];
    for (int c = inv_off[Lq]; c < inv_off[Lq+1]; ++c)
      if (!g_constrain(z, dim, inv_data[3*c], inv_data[3*c+1], inv_data[3*c+2])) return false;
  }
  return true;
}

// ---- ROW-DISTRIBUTED zone: one SIMD group per zone, lane i owns row i ---------------
// Third representation, added alongside the private-array and threadgroup ones. The dense
// dim x dim matrix is UPPAAL's shape and is what creates the working-set dilemma (§2.3 of
// docs/GPU-FINDINGS.md): 576 ints per thread cannot live in registers, and moving them
// on-chip collapses occupancy. Distributing ONE zone across the 32 lanes of a SIMD group
// makes each lane hold a single ROW -- dim ints, not dim*dim -- a 24x cut in per-lane
// footprint. Floyd-Warshall still needs d[k][*], which arrives by simd_shuffle
// (register-to-register, no memory). Lanes of a SIMD group run in lockstep, so the k-loop
// needs no barriers; every branch below is SIMD-uniform (emptiness folds through simd_any).
// Lanes >= dim hold no row but MUST keep executing: simd_shuffle is a lockstep operation.
inline bool rg_empty(thread const int* row, int dim, uint lane) {
  const bool e = (lane < uint(dim)) && (row[lane] < kZeroLe);   // d[i][i] is my row's slot i
  return simd_any(e);
}
inline void rg_delay(thread int* row, int dim, uint lane) {
  if (lane >= 1 && lane < uint(dim)) row[0] = kInf;             // d[i][0] = inf for i >= 1
}
inline void rg_reset(thread int* row, int dim, int k, uint lane) {
  int r0[MAXDIM];
  for (int j = 0; j < dim; ++j) r0[j] = simd_shuffle(row[j], 0);  // row 0, to every lane
  const int my0 = row[0];
  if (lane == uint(k)) { for (int j = 0; j < dim; ++j) row[j] = r0[j]; }  // d[k][*] = d[0][*]
  row[k] = (lane == uint(k)) ? kZeroLe : my0;                            // d[*][k] = d[*][0]
}
inline bool rg_constrain(thread int* row, int dim, int a, int b, int bound, uint lane) {
  const int dab = simd_shuffle(row[b], uint(a));                 // d[a][b] lives on lane a
  if (bound >= dab) return !rg_empty(row, dim, lane);
  if (lane == uint(a)) row[b] = bound;
  const int via = r_add(row[a], bound);                          // d[i][b] vs d[i][a]+bound
  if (via < row[b]) row[b] = via;
  const int ib = row[b];
  for (int k = 0; k < dim; ++k) {
    const int bk = simd_shuffle(row[k], uint(b));                // d[b][k], broadcast
    if (ib < kInf) { const int s = r_add(ib, bk); if (s < row[k]) row[k] = s; }
  }
  return !rg_empty(row, dim, lane);
}
inline bool rg_apply_inv(thread int* row, int dim, int nprocs, thread const int* nlocs,
                         device const int* locbase, device const int* inv_off,
                         device const int* inv_data, uint lane) {
  for (int q = 0; q < nprocs; ++q) {
    const int Lq = locbase[q] + nlocs[q];
    for (int c = inv_off[Lq]; c < inv_off[Lq+1]; ++c)
      if (!rg_constrain(row, dim, inv_data[3*c], inv_data[3*c+1], inv_data[3*c+2], lane)) return false;
  }
  return true;
}

// ---- merge ops: port of dbm_close / dbm_includes / dbm_is_alu_le / dbm_try_union -------
constant int kAluNegInf = -(1 << 29);
inline void d_close(thread int* d, int dim) {
#pragma clang loop unroll(disable)
  for (int k = 0; k < dim; ++k)
#pragma clang loop unroll(disable)
    for (int i = 0; i < dim; ++i) {
      int ik = d[i*dim+k]; if (ik >= kInf) continue;
      for (int j = 0; j < dim; ++j) { int s = r_add(ik, d[k*dim+j]); if (s < d[i*dim+j]) d[i*dim+j] = s; }
    }
}
// does a include b?  (b[i] <= a[i] for all i)
inline bool d_includes(thread const int* a, thread const int* b, int dim) {
  for (int i = 0; i < dim*dim; ++i) if (b[i] > a[i]) return false;
  return true;
}
// is Z subsumed by Zp under the aLU preorder?  (port of dbm_is_alu_le)
inline bool d_alu_le(thread const int* Z, thread const int* Zp, int dim,
                     device const int* L, device const int* U) {
  for (int x = 0; x < dim; ++x) {
    if (U[x] == kAluNegInf) continue;
    if (Z[0*dim+x] < r_le(-U[x])) continue;
    for (int y = 0; y < dim; ++y) {
      if (x == y) continue;
      if (L[y] == kAluNegInf) continue;
      if (Zp[y*dim+x] < Z[y*dim+x] && r_add(Zp[y*dim+x], r_lt(-L[y])) < Z[0*dim+x]) return false;
    }
  }
  return true;
}
// ---- minimal-graph analysis (device port of mindbm_analyze, dbm_ops.hpp) ----------------
// Marks exactly the NON-REDUNDANT off-diagonal constraints of a closed, non-empty DBM. The
// convex-union test only needs a crossing region per constraint of a generating set, and the
// minimal graph is one: it defines the same zone, so the redundant entries add nothing. That
// turns O(dim^2) facet tests into O(dim).
#define MG_WORDS ((MAXDIM*MAXDIM + 31) / 32)
inline int mg_set(thread uint* bits, int idx) {
  const uint m = 1u << (idx & 31);
  const int newly = (bits[idx >> 5] & m) ? 0 : 1;
  bits[idx >> 5] |= m;
  return newly;
}
inline void mg_analyze(thread const int* d, int dim, thread uint* bits) {
  for (int w = 0; w < MG_WORDS; ++w) bits[w] = 0u;
  if (dim <= 1) return;
  int first[MAXDIM], nxt[MAXDIM];
  for (int i = 0; i < dim; ++i) nxt[i] = 0;
  int nCl = 0;
  for (int i = 0; i < dim; ++i) {                    // (1) zero-cycle equivalence classes
    if (nxt[i] != 0) continue;
    first[nCl++] = i;
    int k = i;
    for (int j = i + 1; j < dim; ++j)
      if (r_bound(d[i*dim+j]) == -r_bound(d[j*dim+i])) { nxt[k] = j; k = j; }
    nxt[k] = -1;
  }
#pragma clang loop unroll(disable)
  for (int pi = 0; pi < nCl; ++pi) {                 // (2) drop edges achieved by a 2-hop path
    const int p = first[pi];
#pragma clang loop unroll(disable)
    for (int qi = 0; qi < nCl; ++qi) {
      const int q = first[qi];
      if (p == q) continue;
      const int bpq = d[p*dim+q];
      if (bpq >= kInf) continue;
      bool red = false;
      for (int ri = 0; ri < nCl; ++ri) {
        const int r = first[ri];
        if (r == p || r == q) continue;
        const int bpr = d[p*dim+r], brq = d[r*dim+q];
        if (bpr < kInf && brq < kInf && bpq >= r_add(bpr, brq)) { red = true; break; }
      }
      if (!red) mg_set(bits, p*dim + q);
    }
  }
  for (int pi = 0; pi < nCl; ++pi) {                 // (3) one concise cycle inside each class
    const int head = first[pi];
    if (nxt[head] == -1) continue;
    int a = head;
    while (nxt[a] != -1) { mg_set(bits, a*dim + nxt[a]); a = nxt[a]; }
    mg_set(bits, a*dim + head);
  }
}

// if a ∪ b is exactly convex, set a := hull and return true; else leave a and return false.
// Device mirror of dbm_try_union (dbm_ops.hpp): minimal-graph facets + incremental close, and
// no final close because the entrywise-max hull of two canonical DBMs is already canonical.
inline bool d_try_union(thread int* a, thread const int* b, int dim) {
  int H[MAXDIM*MAXDIM], R[MAXDIM*MAXDIM];
  for (int k = 0; k < dim*dim; ++k) H[k] = a[k] > b[k] ? a[k] : b[k];
  uint bits[MG_WORDS];
  for (int pass = 0; pass < 2; ++pass) {
    thread const int* z = (pass == 0) ? a : b;
    thread const int* other = (pass == 0) ? b : a;
    mg_analyze(z, dim, bits);
#pragma clang loop unroll(disable)
    for (int i = 0; i < dim; ++i)
#pragma clang loop unroll(disable)
      for (int j = 0; j < dim; ++j) {
        if (i == j || z[i*dim+j] >= H[i*dim+j]) continue;
        const int idx = i*dim + j;
        if (!(bits[idx >> 5] & (1u << (idx & 31)))) continue;
#pragma clang loop unroll(disable)
        for (int k = 0; k < dim*dim; ++k) R[k] = H[k];
        int v = r_bound(z[idx]);
        int neg = r_strict(z[idx]) ? r_le(-v) : r_lt(-v);
        if (!d_constrain(R, dim, j, i, neg)) continue;   // crossing region empty -> fine
        if (!d_includes(other, R, dim)) return false;
      }
  }
  for (int k = 0; k < dim*dim; ++k) a[k] = H[k];   // already canonical: max of two canonical DBMs
  return true;
}

// ---- gpusplit: cooperative merge primitives ---------------------------------------------
// One SIMD-group (32 lanes) processes ONE bucket's antichain: the O(cur^2) subsumption /
// convex-union loop is split across the lanes, and the current new zone `zn` lives in
// threadgroup memory shared by all lanes. These helpers are address-space-GENERIC (templated
// on the pointer type) so a threadgroup `zn` can be compared against a per-lane private `zw`
// without copying either into the other's space. Semantics are byte-identical to the
// per-thread d_* ops above; only the address spaces differ.

// Strided view of one zone: entry e at base[e*stride]. stride=1 => AoS (base=W+i*d2), stride=cap
// => SoA (base=W+i). Presenting it via operator[] lets the templated ops read either layout
// unchanged; in SoA, 32 lanes reading the same entry of consecutive zones hit consecutive
// addresses (coalesced).
struct SoaZone {
  device const int* base;
  int stride;
  int operator[](int e) const { return base[e * stride]; }
};

template <typename PZ, typename PZP>
inline bool g_alu_le(PZ Z, PZP Zp, int dim, device const int* L, device const int* U) {
  // Border (row-0) pre-filter: the y=0 witness of the full test, evaluated in O(dim) over just
  // the reference rows Z[0][*], Zp[0][*] (contiguous, coalescing-friendly). Zp tighter than Z on
  // some clock's lower bound is a real witness => not subsumed. Catches the common non-subsumed
  // case before the O(dim^2) scan; if none found we fall through to the exact test unchanged.
  if (L[0] != kAluNegInf) {
    for (int x = 1; x < dim; ++x) {
      if (U[x] == kAluNegInf) continue;
      if (Z[0*dim+x] < r_le(-U[x])) continue;
      if (Zp[0*dim+x] < Z[0*dim+x] && r_add(Zp[0*dim+x], r_lt(-L[0])) < Z[0*dim+x]) return false;
    }
  }
  for (int x = 0; x < dim; ++x) {
    if (U[x] == kAluNegInf) continue;
    if (Z[0*dim+x] < r_le(-U[x])) continue;
    for (int y = 1; y < dim; ++y) {          // y=0 already covered by the border pre-filter
      if (x == y) continue;
      if (L[y] == kAluNegInf) continue;
      if (Zp[y*dim+x] < Z[y*dim+x] && r_add(Zp[y*dim+x], r_lt(-L[y])) < Z[0*dim+x]) return false;
    }
  }
  return true;
}
template <typename PA, typename PB>
inline bool g_includes(PA a, PB b, int dim) {
  for (int i = 0; i < dim*dim; ++i) if (b[i] > a[i]) return false;
  return true;
}
// Facet test of the convex-union check for one operand `z` (its facets) against `other`.
// `H` (elementwise max hull) and scratch `R` are private. Returns false if crossing a strict
// facet of z escapes `other` (=> union not convex).
// Address-space-generic minimal-graph analysis (see mg_analyze above for the method).
template <typename PZ>
inline void g_mg_analyze(PZ d, int dim, thread uint* bits) {
  for (int w = 0; w < MG_WORDS; ++w) bits[w] = 0u;
  if (dim <= 1) return;
  int first[MAXDIM], nxt[MAXDIM];
  for (int i = 0; i < dim; ++i) nxt[i] = 0;
  int nCl = 0;
  for (int i = 0; i < dim; ++i) {
    if (nxt[i] != 0) continue;
    first[nCl++] = i;
    int k = i;
    for (int j = i + 1; j < dim; ++j)
      if (r_bound(d[i*dim+j]) == -r_bound(d[j*dim+i])) { nxt[k] = j; k = j; }
    nxt[k] = -1;
  }
  for (int pi = 0; pi < nCl; ++pi) {
    const int p = first[pi];
    for (int qi = 0; qi < nCl; ++qi) {
      const int q = first[qi];
      if (p == q) continue;
      const int bpq = d[p*dim+q];
      if (bpq >= kInf) continue;
      bool red = false;
      for (int ri = 0; ri < nCl; ++ri) {
        const int r = first[ri];
        if (r == p || r == q) continue;
        const int bpr = d[p*dim+r], brq = d[r*dim+q];
        if (bpr < kInf && brq < kInf && bpq >= r_add(bpr, brq)) { red = true; break; }
      }
      if (!red) mg_set(bits, p*dim + q);
    }
  }
  for (int pi = 0; pi < nCl; ++pi) {
    const int head = first[pi];
    if (nxt[head] == -1) continue;
    int a = head;
    while (nxt[a] != -1) { mg_set(bits, a*dim + nxt[a]); a = nxt[a]; }
    mg_set(bits, a*dim + head);
  }
}

template <typename PZ, typename POTHER>
inline bool g_facets_ok(PZ z, POTHER other, thread const int* H, int dim) {
  int R[MAXDIM*MAXDIM];
  uint bits[MG_WORDS];
  g_mg_analyze(z, dim, bits);
#pragma clang loop unroll(disable)
  for (int i = 0; i < dim; ++i)
#pragma clang loop unroll(disable)
    for (int j = 0; j < dim; ++j) {
      if (i == j || z[i*dim+j] >= H[i*dim+j]) continue;
      const int idx = i*dim + j;
      if (!(bits[idx >> 5] & (1u << (idx & 31)))) continue;   // redundant: covered by its path
#pragma clang loop unroll(disable)
      for (int k = 0; k < dim*dim; ++k) R[k] = H[k];
      int v = r_bound(z[idx]);
      int neg = r_strict(z[idx]) ? r_le(-v) : r_lt(-v);
      if (!d_constrain(R, dim, j, i, neg)) continue;          // crossing region empty -> fine
      if (!g_includes(other, R, dim)) return false;
    }
  return true;
}
// Cheap SOUND rejection before the O(dim^3) convex-union test: if the two zones' projections
// onto ANY single clock axis have a strict gap (one's upper bound is below the other's lower
// bound), their union has a hole on that axis and cannot be convex. Uses only the 2*dim border
// entries (row 0 = lower bounds, col 0 = upper bounds) -- no dim^2 scratch, no close. A missed
// non-convexity (diagonal, not axial) just falls through to the full test, so this is a pure
// necessary condition -- it never wrongly merges (and convex-union is only a compression step
// anyway, so even a false reject would cost compactness, never correctness).
template <typename PA, typename PB>
inline bool g_union_axisgap(PA a, PB b, int dim) {
  for (int i = 1; i < dim; ++i) {
    // clock i:  upper = r_bound(z[i*dim+0]) (unless kInf);  lower value = -r_bound(z[0*dim+i]).
    if (b[i*dim+0] < kInf && a[0*dim+i] < kInf &&
        -r_bound(a[0*dim+i]) > r_bound(b[i*dim+0])) return true;   // a's lower above b's upper -> gap
    if (a[i*dim+0] < kInf && b[0*dim+i] < kInf &&
        -r_bound(b[0*dim+i]) > r_bound(a[i*dim+0])) return true;   // b's lower above a's upper -> gap
  }
  return false;
}
// Is a ∪ b exactly convex?  PURE TEST (does not mutate a) so lanes can test different
// candidates in parallel; the winning union is applied cooperatively afterwards.
template <typename PA, typename PB>
inline bool g_union_convex(PA a, PB b, int dim) {
  if (g_union_axisgap(a, b, dim)) return false;      // cheap border pre-filter
  int H[MAXDIM*MAXDIM];
  for (int k = 0; k < dim*dim; ++k) H[k] = a[k] > b[k] ? a[k] : b[k];
  return g_facets_ok(a, b, H, dim) && g_facets_ok(b, a, H, dim);
}
// Cooperative all-pairs-shortest-path close of a threadgroup zone: lane L owns row L (dim<=24
// <32, so <=1 row/lane). The k-loop is sequential (row k must be final before it relaxes the
// others); row k is invariant during step k (d[k][k]=0), so lane k's writes are no-ops and no
// intra-step race exists. A barrier per step publishes each row before the next reads it.
inline void sg_close(threadgroup int* d, int dim, uint lane) {
  const bool act = (int)lane < dim;
  for (int k = 0; k < dim; ++k) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (act) {
      int ik = d[lane*dim + k];
      if (ik < kInf)
        for (int j = 0; j < dim; ++j) {
          int s = r_add(ik, d[k*dim + j]);
          if (s < d[lane*dim + j]) d[lane*dim + j] = s;
        }
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
}

// ---- SIMD-group cooperative DBM ops (zone in THREADGROUP memory) -------------
// The per-thread kernel below gives each thread its own `int z[MAXDIM*MAXDIM]`.
// Because it is indexed dynamically (z[i*dim+j]), the compiler cannot keep it in
// registers, so it lives in DEVICE-backed thread-private memory and every one of
// the O(dim^2)-O(dim^3) element accesses per successor is a device round trip --
// that, not the input read or the output write, is what makes net_expand slow
// (pipeline reflection confirms it is NOT register/occupancy limited: 1024
// threads/threadgroup).
//
// Here one SIMD GROUP (32 lanes) owns one successor and the DBM lives on-chip in
// threadgroup memory: 32x less memory per work item, so it fits, and each O(dim^2)
// loop is split across the lanes. Since dim <= MAXDIM(24) < 32, every lane owns AT
// MOST ONE row, which keeps the parallel form simple: lane L handles row L.
inline bool sg_empty(threadgroup int* d, int dim, uint lane) {
  bool e = ((int)lane < dim) && (d[lane*dim + lane] < kZeroLe);
  return simd_any(e);
}
inline void sg_reset(threadgroup int* d, int dim, int k, uint lane) {
  // Snapshot row0/col0 BEFORE writing row k / col k: the write d[0*dim+k] would
  // otherwise race the read d[0*dim+j] performed by the lane handling j == k.
  const bool act = (int)lane < dim;
  int rv = 0, cv = 0;
  if (act) { rv = d[0*dim + lane]; cv = d[lane*dim + 0]; }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (act) { d[k*dim + lane] = rv; d[lane*dim + k] = cv; }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) d[k*dim + k] = kZeroLe;
  simdgroup_barrier(mem_flags::mem_threadgroup);
}
inline void sg_delay(threadgroup int* d, int dim, uint lane) {
  if ((int)lane < dim && lane >= 1u) d[lane*dim + 0] = kInf;
  simdgroup_barrier(mem_flags::mem_threadgroup);
}
// Canonicity-restoring tighten after adding d[a][b] <= bound, in the standard form
//   d[i][j] = min(d[i][j], d[i][a] + bound + d[b][j])
// with row b snapshotted (rowb) so every lane sees the pre-update values.
inline bool sg_constrain(threadgroup int* d, threadgroup int* rowb, int dim,
                         int a, int b, int bound, uint lane) {
  if (bound >= d[a*dim + b]) return !sg_empty(d, dim, lane);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) d[a*dim + b] = bound;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const bool act = (int)lane < dim;
  if (act) {                                     // d[i][b] = min(d[i][b], d[i][a]+bound)
    int via = r_add(d[lane*dim + a], bound);
    if (via < d[lane*dim + b]) d[lane*dim + b] = via;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (act) rowb[lane] = d[b*dim + lane];         // snapshot row b
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (act) {                                     // d[i][k] = min(d[i][k], d[i][b]+rowb[k])
    int ib = d[lane*dim + b];
    if (ib < kInf)
      for (int k = 0; k < dim; ++k) {
        int via = r_add(ib, rowb[k]);
        if (via < d[lane*dim + k]) d[lane*dim + k] = via;
      }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  return !sg_empty(d, dim, lane);
}
inline bool sg_apply_inv(threadgroup int* d, threadgroup int* rowb, int dim, int nprocs,
                         threadgroup const int* nlocs, device const int* locbase,
                         device const int* inv_off, device const int* inv_data, uint lane) {
  for (int q = 0; q < nprocs; ++q) {
    int Lq = locbase[q] + nlocs[q];
    for (int c = inv_off[Lq]; c < inv_off[Lq+1]; ++c)
      if (!sg_constrain(d, rowb, dim, inv_data[3*c], inv_data[3*c+1], inv_data[3*c+2], lane))
        return false;
  }
  return true;
}

struct Params { int dim, nprocs, nintvars, nclocks, nwork, qbits, ng; };
struct MergeParams { int dim, nb, has_diag, union_on, soa; };

// One thread per successor "recipe": a participant list of (proc, global-edge) — 1 for a
// location step, 2 for a binary handshake, 1+K for a broadcast. Int guards, committed
// filtering, and the sync enumeration are done on the host (which knows the discrete state);
// the kernel is the pure zone pipeline over the participants (exactly successors_net's zone
// steps: all guards, all resets, combined invariant, delay-if-not-frozen, invariant,
// reset_inactive, emptiness) plus the discrete update (locs/ints).
//
// The INTEGER half of the discrete update is computed host-side and arrives already
// applied, per work item, in `work_ints`. It used to be replayed here from a flattened
// (var := const) table, which silently ignored NEdge::gen_update -- so a model whose
// update is an expression (viking's `L = (1 - L) % 2`) lost the assignment entirely on
// GPU and explored roughly half its state space. The host knows the source ints for each
// recipe, so it runs the exact CPU semantics (int_updates then gen_apply) and ships the
// result; that also lets a faulting update drop the recipe before it ever reaches the GPU.
kernel void net_expand(
    device const int* front_locs [[buffer(0)]],
    device const int* work_ints  [[buffer(1)]],   // [nwork*nintvars], post-update
    device const int* front_zone [[buffer(2)]],
    device const int* work_state [[buffer(3)]],
    device const int* part_off   [[buffer(4)]],
    device const int* part_cnt   [[buffer(5)]],
    device const int* part_proc  [[buffer(6)]],
    device const int* part_edge  [[buffer(7)]],
    device const int* edge_tgt   [[buffer(8)]],
    device const int* cg_off     [[buffer(9)]],
    device const int* cg_data    [[buffer(10)]],
    device const int* rs_off     [[buffer(11)]],
    device const int* rs_clk     [[buffer(12)]],
    device const int* locbase    [[buffer(13)]],
    device const int* inv_off    [[buffer(14)]],
    device const int* inv_data   [[buffer(15)]],
    device const int* frozen     [[buffer(16)]],
    device const int* active     [[buffer(17)]],
    device const int* qmask      [[buffer(18)]],
    device int*       out_valid  [[buffer(19)]],
    device ulong*     out_dk     [[buffer(20)]],   // packed discrete key (was out_locs)
    device int*       out_ints   [[buffer(21)]],
    device int*       out_zone   [[buffer(22)]],
    constant Params&  P          [[buffer(23)]],
    device const uint* act_mask  [[buffer(24)]],   // per-location active-clock bitmask
    device const int* dshift     [[buffer(25)]],   // bit offset of each discrete field in the key
    device const int* intlo      [[buffer(26)]],   // low bound of each int var (key is offset by it)
    device const int* sym_proc   [[buffer(27)]],   // [ng*nprocs] process permutation of each g
    device const int* sym_clk    [[buffer(28)]],   // [ng*dim]    induced clock permutation
    device const int* sym_ivoff  [[buffer(29)]],   // [ng*nintvars+1] CSR into sym_ivdata
    device const int* sym_ivdata [[buffer(30)]],   // value permutation of each (g, int var)
    uint w [[thread_position_in_grid]])
{
  if (w >= uint(P.nwork)) return;
  // Compile-time dim here ONLY: this kernel is a straight-line zone pipeline, so constant loop
  // bounds strength-reduce the index arithmetic with no compile-time blowup. The merge kernels
  // deliberately keep P.dim -- see the note on shader build time in the class comment.
  constexpr int dim = PT_DIM, d2 = dim * dim, nprocs = PT_NP, nintvars = PT_NI, nclocks = PT_NC;
  const uint qbits = uint(P.qbits);
  out_valid[w] = 0;
  const int si = work_state[w], po = part_off[w], pc = part_cnt[w];

  // target location vector: apply every participant's target
  int nlocs[MAXPROC];
  for (int q = 0; q < nprocs; ++q) nlocs[q] = front_locs[si*nprocs + q];
  for (int t = 0; t < pc; ++t) nlocs[part_proc[po+t]] = edge_tgt[part_edge[po+t]];

  // zone: all participants' clock guards, then all their resets
  int z[MAXDIM * MAXDIM];
  for (int k = 0; k < d2; ++k) z[k] = front_zone[si*d2 + k];
  for (int t = 0; t < pc; ++t) {
    int ge = part_edge[po+t];
    for (int c = cg_off[ge]; c < cg_off[ge+1]; ++c)
      if (!d_constrain(z, dim, cg_data[3*c], cg_data[3*c+1], cg_data[3*c+2])) return;
  }
  for (int t = 0; t < pc; ++t) {
    int ge = part_edge[po+t];
    for (int r = rs_off[ge]; r < rs_off[ge+1]; ++r) d_reset(z, dim, rs_clk[r]);
  }

  // combined invariant; delay if no process is committed/urgent; invariant again
  if (!apply_inv(z, dim, nprocs, nlocs, locbase, inv_off, inv_data)) return;
  bool frozen_any = false;
  for (int q = 0; q < nprocs; ++q) if (frozen[locbase[q] + nlocs[q]] & 3) { frozen_any = true; break; }
  if (!frozen_any) {
    d_delay(z, dim);
    if (!apply_inv(z, dim, nprocs, nlocs, locbase, inv_off, inv_data)) return;
  }

  // Reset clocks inactive at EVERY process's target location and not read by the query.
  // One bitmask load per process, OR'd together — was nprocs*nclocks separate loads out of the
  // [loc][clock] table (289 of them on arbiter17, on par with the whole rest of the pipeline).
  {
    uint amask = qbits;
    for (int q = 0; q < nprocs; ++q) amask |= act_mask[locbase[q] + nlocs[q]];
    amask = (~amask) & (((nclocks >= 31) ? 0xffffffffu : ((1u << (nclocks + 1)) - 1u)) & ~1u);
    while (amask) { const int cc = ctz(amask); amask &= amask - 1; d_reset(z, dim, cc); }
  }
  if (d_empty(z, dim)) return;

  out_valid[w] = 1;

  // ---- SYMMETRY: rewrite the successor to its orbit's canonical representative --------------
  // The network's automorphism group permutes processes, and with it the clocks and any
  // process-INDEX-valued integer (Fischer's `id`). Exploring one representative per orbit is a
  // sound reduction that no amount of parallelism can give you -- 10x fewer states on fischer10
  // and phil10 -- and it belongs here rather than on the host: it is |G| integer comparisons per
  // successor over data the thread already holds, which is exactly what a GPU lane is for. The
  // representative is the one with the lexicographically smallest (locations, integer values)
  // and, among ties, the smallest zone matrix -- identical to the host `canonicalize`.
  //
  // The zone is never materialised twice: ties are broken by comparing the two permuted matrices
  // through their inverse clock permutations (a permuted entry (a,b) is z[inv[a]][inv[b]]), and
  // only the winner is written, permuted directly into the output. So symmetry costs this kernel
  // no extra private memory -- which matters, because the private zone is what it is bound on.
  int bl[MAXPROC], bv[MAXINT];
  for (int q = 0; q < nprocs; ++q) bl[q] = nlocs[q];
  for (int v = 0; v < nintvars; ++v) bv[v] = work_ints[w*nintvars + v];
  int bestg = 0;
  if (P.ng > 1) {
    int cl[MAXPROC], cv[MAXINT];
    // Two passes over the group rather than one pass plus a tie list: a fixed-size tie array
    // would cap the group the kernel can take (and TRUNCATING it would be unsound -- two states
    // of one orbit could then pick different representatives and both be explored). Pass 1 finds
    // the smallest discrete image; pass 2 revisits the elements that achieve it and keeps the one
    // with the smallest permuted zone. No per-thread storage grows with |G|.
    for (int g = 1; g < P.ng; ++g) {
      for (int q = 0; q < nprocs; ++q) cl[sym_proc[g*nprocs + q]] = nlocs[q];
      for (int v = 0; v < nintvars; ++v) {
        const int lo = sym_ivoff[g*nintvars + v], hi = sym_ivoff[g*nintvars + v + 1];
        const int x = work_ints[w*nintvars + v];
        cv[v] = (hi > lo) ? sym_ivdata[lo + (x - intlo[v])] : x;   // empty range = identity
      }
      int c = 0;
      for (int q = 0; q < nprocs && c == 0; ++q) if (cl[q] != bl[q]) c = cl[q] < bl[q] ? -1 : 1;
      for (int v = 0; v < nintvars && c == 0; ++v) if (cv[v] != bv[v]) c = cv[v] < bv[v] ? -1 : 1;
      if (c < 0) {
        for (int q = 0; q < nprocs; ++q) bl[q] = cl[q];
        for (int v = 0; v < nintvars; ++v) bv[v] = cv[v];
      }
    }
    int invB[MAXDIM], invC[MAXDIM];
    bestg = -1;
    for (int g = 0; g < P.ng; ++g) {
      for (int q = 0; q < nprocs; ++q) cl[sym_proc[g*nprocs + q]] = nlocs[q];
      bool same = true;
      for (int q = 0; q < nprocs && same; ++q) same = (cl[q] == bl[q]);
      for (int v = 0; v < nintvars && same; ++v) {
        const int lo = sym_ivoff[g*nintvars + v], hi = sym_ivoff[g*nintvars + v + 1];
        const int x = work_ints[w*nintvars + v];
        same = ((hi > lo) ? sym_ivdata[lo + (x - intlo[v])] : x) == bv[v];
      }
      if (!same) continue;
      if (bestg < 0) { bestg = g; for (int i = 0; i < dim; ++i) invB[sym_clk[g*dim + i]] = i; continue; }
      for (int i = 0; i < dim; ++i) invC[sym_clk[g*dim + i]] = i;
      int c = 0;                       // compare the two permuted matrices without building them
      for (int a = 0; a < dim && c == 0; ++a)
        for (int b = 0; b < dim; ++b) {
          const int vg = z[invC[a]*dim + invC[b]], vb = z[invB[a]*dim + invB[b]];
          if (vg != vb) { c = vg < vb ? -1 : 1; break; }
        }
      if (c < 0) { bestg = g; for (int i = 0; i < dim; ++i) invB[i] = invC[i]; }
    }
  }

  // PACK THE DISCRETE KEY HERE. The host's bucketing pass is the level's serial section, and it
  // used to read this successor's nprocs locations plus nintvars integers back out of unified
  // memory (48 bytes on fischer12) purely to recompute this 8-byte key. Emitting the key means
  // the host reads 8 bytes per successor and the per-successor location array is not written at
  // all -- 41.6M x 13 words of kernel stores saved over a fischer12 run.
  ulong dk = 0;
  for (int q = 0; q < nprocs; ++q) dk |= ulong(uint(bl[q])) << dshift[q];
  for (int v = 0; v < nintvars; ++v) dk |= ulong(uint(bv[v] - intlo[v])) << dshift[nprocs + v];
  out_dk[w] = dk;
  if (bestg == 0) {
    for (int k = 0; k < d2; ++k) out_zone[w*d2 + k] = z[k];
  } else {   // permuting a canonical DBM by a clock permutation leaves it canonical
    device const int* clk = sym_clk + bestg*dim;
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j) out_zone[w*d2 + clk[i]*dim + clk[j]] = z[i*dim + j];
  }
}

kernel void net_expand_tg(
    device const int* front_locs [[buffer(0)]],
    device const int* work_ints  [[buffer(1)]],   // [nwork*nintvars], post-update
    device const int* front_zone [[buffer(2)]],
    device const int* work_state [[buffer(3)]],
    device const int* part_off   [[buffer(4)]],
    device const int* part_cnt   [[buffer(5)]],
    device const int* part_proc  [[buffer(6)]],
    device const int* part_edge  [[buffer(7)]],
    device const int* edge_tgt   [[buffer(8)]],
    device const int* cg_off     [[buffer(9)]],
    device const int* cg_data    [[buffer(10)]],
    device const int* rs_off     [[buffer(11)]],
    device const int* rs_clk     [[buffer(12)]],
    device const int* locbase    [[buffer(13)]],
    device const int* inv_off    [[buffer(14)]],
    device const int* inv_data   [[buffer(15)]],
    device const int* frozen     [[buffer(16)]],
    device const int* active     [[buffer(17)]],
    device const int* qmask      [[buffer(18)]],
    device int*       out_valid  [[buffer(19)]],
    device int*       out_locs   [[buffer(20)]],
    device int*       out_ints   [[buffer(21)]],
    device int*       out_zone   [[buffer(22)]],
    constant Params&  P          [[buffer(23)]],
    threadgroup int*  zbuf       [[threadgroup(0)]],
    uint w [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
  if (w >= uint(P.nwork)) return;
  const int dim = P.dim, d2 = dim * dim, nprocs = PT_NP, nintvars = PT_NI, nclocks = PT_NC;
  out_valid[w] = 0;
  const int si = work_state[w], po = part_off[w], pc = part_cnt[w];

  // target location vector: apply every participant's target
  int nlocs[MAXPROC];
  for (int q = 0; q < nprocs; ++q) nlocs[q] = front_locs[si*nprocs + q];
  for (int t = 0; t < pc; ++t) nlocs[part_proc[po+t]] = edge_tgt[part_edge[po+t]];

  // zone: all participants' clock guards, then all their resets
  // Stride is d2+1, not d2: threadgroup memory has 32 banks and d2 is typically a multiple
  // of 16 (dim 12 -> 144), which would map all 32 lanes onto 2 banks -- a 16-way conflict.
  // An odd stride walks every bank.
  threadgroup int* z = zbuf + lid * (d2 + 1);
  for (int k = 0; k < d2; ++k) z[k] = front_zone[si*d2 + k];
  for (int t = 0; t < pc; ++t) {
    int ge = part_edge[po+t];
    for (int c = cg_off[ge]; c < cg_off[ge+1]; ++c)
      if (!g_constrain(z, dim, cg_data[3*c], cg_data[3*c+1], cg_data[3*c+2])) return;
  }
  for (int t = 0; t < pc; ++t) {
    int ge = part_edge[po+t];
    for (int r = rs_off[ge]; r < rs_off[ge+1]; ++r) g_reset(z, dim, rs_clk[r]);
  }

  // combined invariant; delay if no process is committed/urgent; invariant again
  if (!g_apply_inv(z, dim, nprocs, nlocs, locbase, inv_off, inv_data)) return;
  bool frozen_any = false;
  for (int q = 0; q < nprocs; ++q) if (frozen[locbase[q] + nlocs[q]] & 3) { frozen_any = true; break; }
  if (!frozen_any) {
    g_delay(z, dim);
    if (!g_apply_inv(z, dim, nprocs, nlocs, locbase, inv_off, inv_data)) return;
  }

  // reset clocks inactive at every process's target location and not read by the query
  for (int cc = 1; cc <= nclocks; ++cc) {
    bool any = false;
    for (int q = 0; q < nprocs; ++q)
      if (active[(locbase[q] + nlocs[q]) * (nclocks + 1) + cc]) { any = true; break; }
    if (!any && !qmask[cc]) g_reset(z, dim, cc);
  }
  if (g_empty(z, dim)) return;

  out_valid[w] = 1;
  for (int q = 0; q < nprocs; ++q) out_locs[w*nprocs + q] = nlocs[q];
  // integers: already updated host-side for this recipe (see the header comment)
  for (int v = 0; v < nintvars; ++v) out_ints[w*nintvars + v] = work_ints[w*nintvars + v];
  for (int k = 0; k < d2; ++k) out_zone[w*d2 + k] = z[k];
}



#define RG_PER_TG 8
kernel void net_expand_rg(
    device const int* front_locs [[buffer(0)]],
    device const int* work_ints  [[buffer(1)]],
    device const int* front_zone [[buffer(2)]],
    device const int* work_state [[buffer(3)]],
    device const int* part_off   [[buffer(4)]],
    device const int* part_cnt   [[buffer(5)]],
    device const int* part_proc  [[buffer(6)]],
    device const int* part_edge  [[buffer(7)]],
    device const int* edge_tgt   [[buffer(8)]],
    device const int* cg_off     [[buffer(9)]],
    device const int* cg_data    [[buffer(10)]],
    device const int* rs_off     [[buffer(11)]],
    device const int* rs_clk     [[buffer(12)]],
    device const int* locbase    [[buffer(13)]],
    device const int* inv_off    [[buffer(14)]],
    device const int* inv_data   [[buffer(15)]],
    device const int* frozen     [[buffer(16)]],
    device const int* active     [[buffer(17)]],
    device const int* qmask      [[buffer(18)]],
    device int*       out_valid  [[buffer(19)]],
    device int*       out_locs   [[buffer(20)]],
    device int*       out_ints   [[buffer(21)]],
    device int*       out_zone   [[buffer(22)]],
    constant Params&  P          [[buffer(23)]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
  const uint w = tgpos * RG_PER_TG + sgid;
  if (w >= uint(P.nwork)) return;          // whole SIMD group leaves together - uniform
  const int dim = P.dim, d2 = dim * dim, nprocs = PT_NP, nintvars = PT_NI, nclocks = PT_NC;
  if (lane == 0) out_valid[w] = 0;
  const int si = work_state[w], po = part_off[w], pc = part_cnt[w];

  int nlocs[MAXPROC];
  for (int q = 0; q < nprocs; ++q) nlocs[q] = front_locs[si*nprocs + q];
  for (int t = 0; t < pc; ++t) nlocs[part_proc[po+t]] = edge_tgt[part_edge[po+t]];

  int row[MAXDIM];                          // MY row of the zone: dim ints, not dim*dim
  if (lane < uint(dim))
    for (int j = 0; j < dim; ++j) row[j] = front_zone[si*d2 + int(lane)*dim + j];

  for (int t = 0; t < pc; ++t) {
    const int ge = part_edge[po+t];
    for (int c = cg_off[ge]; c < cg_off[ge+1]; ++c)
      if (!rg_constrain(row, dim, cg_data[3*c], cg_data[3*c+1], cg_data[3*c+2], lane)) return;
  }
  for (int t = 0; t < pc; ++t) {
    const int ge = part_edge[po+t];
    for (int r = rs_off[ge]; r < rs_off[ge+1]; ++r) rg_reset(row, dim, rs_clk[r], lane);
  }

  if (!rg_apply_inv(row, dim, nprocs, nlocs, locbase, inv_off, inv_data, lane)) return;
  bool frozen_any = false;
  for (int q = 0; q < nprocs; ++q) if (frozen[locbase[q] + nlocs[q]] & 3) { frozen_any = true; break; }
  if (!frozen_any) {
    rg_delay(row, dim, lane);
    if (!rg_apply_inv(row, dim, nprocs, nlocs, locbase, inv_off, inv_data, lane)) return;
  }

  for (int cc = 1; cc <= nclocks; ++cc) {
    bool any = false;
    for (int q = 0; q < nprocs; ++q)
      if (active[(locbase[q] + nlocs[q]) * (nclocks + 1) + cc]) { any = true; break; }
    if (!any && !qmask[cc]) rg_reset(row, dim, cc, lane);
  }
  if (rg_empty(row, dim, lane)) return;

  if (lane == 0) {
    out_valid[w] = 1;
    for (int q = 0; q < nprocs; ++q) out_locs[w*nprocs + q] = nlocs[q];
    for (int v = 0; v < nintvars; ++v) out_ints[w*nintvars + v] = work_ints[w*nintvars + v];
  }
  if (lane < uint(dim))
    for (int j = 0; j < dim; ++j) out_zone[w*d2 + int(lane)*dim + j] = row[j];
}

// Cooperative variant of net_expand: ONE SIMD GROUP (32 lanes) per work item, with the
// zone held on-chip in threadgroup memory instead of a device-backed per-thread array.
// Every branch here is SIMD-group-UNIFORM (the sg_* helpers reduce through simd_any), so
// all 32 lanes take the same path and reach the same barriers; simdgroup_barrier (not
// threadgroup_barrier) is used so a group whose w is past the end can return without
// stalling its neighbours.
#define SG_PER_TG 8
kernel void net_expand_sg(
    device const int* front_locs [[buffer(0)]],
    device const int* work_ints  [[buffer(1)]],
    device const int* front_zone [[buffer(2)]],
    device const int* work_state [[buffer(3)]],
    device const int* part_off   [[buffer(4)]],
    device const int* part_cnt   [[buffer(5)]],
    device const int* part_proc  [[buffer(6)]],
    device const int* part_edge  [[buffer(7)]],
    device const int* edge_tgt   [[buffer(8)]],
    device const int* cg_off     [[buffer(9)]],
    device const int* cg_data    [[buffer(10)]],
    device const int* rs_off     [[buffer(11)]],
    device const int* rs_clk     [[buffer(12)]],
    device const int* locbase    [[buffer(13)]],
    device const int* inv_off    [[buffer(14)]],
    device const int* inv_data   [[buffer(15)]],
    device const int* frozen     [[buffer(16)]],
    device const int* active     [[buffer(17)]],
    device const int* qmask      [[buffer(18)]],
    device int*       out_valid  [[buffer(19)]],
    device int*       out_locs   [[buffer(20)]],
    device int*       out_ints   [[buffer(21)]],
    device int*       out_zone   [[buffer(22)]],
    constant Params&  P          [[buffer(23)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
  threadgroup int zbuf[SG_PER_TG * MAXDIM * MAXDIM];
  threadgroup int rbuf[SG_PER_TG * MAXDIM];
  threadgroup int lbuf[SG_PER_TG * MAXPROC];
  threadgroup int* z    = zbuf + sgid * (MAXDIM * MAXDIM);
  threadgroup int* rowb = rbuf + sgid * MAXDIM;
  threadgroup int* nlocs= lbuf + sgid * MAXPROC;

  const uint w = tgid * SG_PER_TG + sgid;
  if (w >= uint(P.nwork)) return;                 // whole group exits together
  const int dim = P.dim, d2 = dim*dim, nprocs = PT_NP, nintvars = PT_NI, nclocks = PT_NC;
  if (lane == 0) out_valid[w] = 0;
  const int si = work_state[w], po = part_off[w], pc = part_cnt[w];

  if ((int)lane < nprocs) nlocs[lane] = front_locs[si*nprocs + lane];
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) for (int t = 0; t < pc; ++t) nlocs[part_proc[po+t]] = edge_tgt[part_edge[po+t]];
  simdgroup_barrier(mem_flags::mem_threadgroup);

  for (int k = (int)lane; k < d2; k += 32) z[k] = front_zone[si*d2 + k];
  simdgroup_barrier(mem_flags::mem_threadgroup);

  for (int t = 0; t < pc; ++t) {
    int ge = part_edge[po+t];
    for (int c = cg_off[ge]; c < cg_off[ge+1]; ++c)
      if (!sg_constrain(z, rowb, dim, cg_data[3*c], cg_data[3*c+1], cg_data[3*c+2], lane)) return;
  }
  for (int t = 0; t < pc; ++t) {
    int ge = part_edge[po+t];
    for (int r = rs_off[ge]; r < rs_off[ge+1]; ++r) sg_reset(z, dim, rs_clk[r], lane);
  }

  if (!sg_apply_inv(z, rowb, dim, nprocs, nlocs, locbase, inv_off, inv_data, lane)) return;
  bool frozen_any = false;
  for (int q = 0; q < nprocs; ++q) if (frozen[locbase[q] + nlocs[q]] & 3) { frozen_any = true; break; }
  if (!frozen_any) {
    sg_delay(z, dim, lane);
    if (!sg_apply_inv(z, rowb, dim, nprocs, nlocs, locbase, inv_off, inv_data, lane)) return;
  }

  for (int cc = 1; cc <= nclocks; ++cc) {
    bool any = false;
    for (int q = 0; q < nprocs; ++q)
      if (active[(locbase[q] + nlocs[q]) * (nclocks + 1) + cc]) { any = true; break; }
    if (!any && !qmask[cc]) sg_reset(z, dim, cc, lane);
  }
  if (sg_empty(z, dim, lane)) return;

  if (lane == 0) {
    out_valid[w] = 1;
    for (int q = 0; q < nprocs; ++q) out_locs[w*nprocs + q] = nlocs[q];
    for (int v = 0; v < nintvars; ++v) out_ints[w*nintvars + v] = work_ints[w*nintvars + v];
  }
  for (int k = (int)lane; k < d2; k += 32) out_zone[w*d2 + k] = z[k];
}

// One thread per discrete-state bucket. Replicates add_to EXACTLY: for each new zone (in
// order) — drop it if aLU-subsumed by a stored zone; drop stored zones it subsumes; absorb
// any stored zone whose convex union with it is exact; then append it. Sequential within a
// bucket (small), parallel across the thousands of buckets. `work` is prefilled by the host
// with the bucket's existing antichain (isnew=0); the kernel appends survivors (isnew=1).
kernel void net_merge(
    device const int* work_off   [[buffer(0)]],  // zone-unit start of bucket b's work slice
    device const int* exist_cnt  [[buffer(1)]],
    device const int* new_off    [[buffer(2)]],  // start into new_idx
    device const int* new_cnt    [[buffer(3)]],
    device const int* oz         [[buffer(4)]],  // expand output zones (new successors, in place)
    device const int* new_idx    [[buffer(5)]],  // expand work-item index of each new zone
    device const int* Lb         [[buffer(6)]],  // [nb*dim]
    device const int* Ub         [[buffer(7)]],  // [nb*dim]
    device int*       work       [[buffer(8)]],  // in/out antichain zone pool
    device int*       work_isnew [[buffer(9)]],
    device int*       out_count  [[buffer(10)]],
    constant MergeParams& P      [[buffer(11)]],
    uint b [[thread_position_in_grid]])
{
  if (b >= uint(P.nb)) return;
  const int dim = P.dim, d2 = dim * dim;
  const int has_diag = P.has_diag;
  // ulong indexing: work_off[b]*d2 overflows 32 bits once the pool exceeds ~3.7M zones at d2=576
  // (kept in step with the same fix in netengine_cuda.cu, where a datacentre GPU reaches it).
  device int* W = work + (ulong)work_off[b] * d2;
  device int* WI = work_isnew + work_off[b];
  device const int* L = Lb + b * dim;
  device const int* U = Ub + b * dim;
  int cur = exist_cnt[b];
  int zn[MAXDIM * MAXDIM], zw[MAXDIM * MAXDIM];

  for (int t = 0; t < new_cnt[b]; ++t) {
    int widx = new_idx[new_off[b] + t];
    for (int k = 0; k < d2; ++k) zn[k] = oz[(ulong)widx * d2 + k];

    bool subsumed = false;                                   // zn subsumed by a stored zone?
    for (int i = 0; i < cur; ++i) {
      for (int k = 0; k < d2; ++k) zw[k] = W[i*d2 + k];
      if (has_diag ? d_includes(zw, zn, dim) : d_alu_le(zn, zw, dim, L, U)) { subsumed = true; break; }
    }
    if (subsumed) continue;

    int w = 0;                                               // drop stored zones subsumed by zn
    for (int i = 0; i < cur; ++i) {
      for (int k = 0; k < d2; ++k) zw[k] = W[i*d2 + k];
      bool killed = has_diag ? d_includes(zn, zw, dim) : d_alu_le(zw, zn, dim, L, U);
      if (!killed) { if (w != i) { for (int k = 0; k < d2; ++k) W[w*d2 + k] = W[i*d2 + k]; WI[w] = WI[i]; } ++w; }
    }
    cur = w;

    bool merged = true;                                      // convex-union absorb
    while (merged) {
      merged = false;
      for (int kk = 0; kk < cur; ++kk) {
        for (int k = 0; k < d2; ++k) zw[k] = W[kk*d2 + k];
        if (d_try_union(zn, zw, dim)) {                      // zn becomes the hull
          for (int i = kk; i < cur - 1; ++i) { for (int k = 0; k < d2; ++k) W[i*d2 + k] = W[(i+1)*d2 + k]; WI[i] = WI[i+1]; }
          --cur; merged = true; break;
        }
      }
    }

    for (int k = 0; k < d2; ++k) W[cur*d2 + k] = zn[k];      // append
    WI[cur] = -1; ++cur;   // -1 = 'this merge created it' (see net_gpu.hpp)
  }
  out_count[b] = cur;
}

// gpusplit: cooperative merge — ONE SIMD-GROUP (32 lanes) per bucket. Byte-identical antichain
// semantics to net_merge, but the per-bucket O(cur^2) subsumption/convex-union loop (the loop
// that a single thread grinds serially, and whose per-bucket cost varies wildly -> SIMD
// divergence in net_merge) is split across the 32 lanes, and the DBM close inside a convex
// union is done cooperatively. Wins exactly where net_merge is starved/divergent: few or
// uneven buckets with long antichains (e.g. philosophers). SG_PER_TG groups share a
// threadgroup; each group's current new zone `zn` lives in threadgroup slot [sgid].
kernel void net_merge_sg(
    device const int* work_off   [[buffer(0)]],
    device const int* exist_cnt  [[buffer(1)]],
    device const int* new_off    [[buffer(2)]],
    device const int* new_cnt    [[buffer(3)]],
    device const int* oz         [[buffer(4)]],
    device const int* new_idx    [[buffer(5)]],
    device const int* Lb         [[buffer(6)]],
    device const int* Ub         [[buffer(7)]],
    device int*       work       [[buffer(8)]],
    device int*       work_isnew [[buffer(9)]],
    device int*       out_count  [[buffer(10)]],
    constant MergeParams& P      [[buffer(11)]],
    device atomic_int* nextb     [[buffer(12)]],   // work-stealing bucket counter
    threadgroup int*  znbuf      [[threadgroup(0)]],   // SG_PER_TG * d2 ints
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
  const int dim = P.dim, d2 = dim * dim;
  const int has_diag = P.has_diag;
  threadgroup int* zn = znbuf + int(sgid) * d2;   // this SIMD-group's zn scratch

  // Persistent-thread WORK-STEALING: rather than a static tgpos->bucket map (which idles a
  // SIMD-group the moment its bucket finishes while its warp-mates grind a big one), each
  // SIMD-group repeatedly grabs the NEXT unclaimed bucket from a global atomic counter. This is
  // GPU-side dynamic scheduling — the same mechanism that lets 12 CPU cores beat the naive GPU
  // merge — so no lane idles on load imbalance. Lane 0 does the atomic; the index is broadcast.
  for (;;) {
    int b;
    if (lane == 0) b = atomic_fetch_add_explicit(nextb, 1, memory_order_relaxed);
    b = simd_broadcast(b, 0);
    if (b >= P.nb) break;

    device int* W  = work + (ulong)work_off[b] * d2;   // SoA pool: zone i entry e at W[e*cap + i]
    device int* WI = work_isnew + work_off[b];
    device const int* L = Lb + b * dim;
    device const int* U = Ub + b * dim;
    const int cap = exist_cnt[b] + new_cnt[b];         // bucket zone-capacity = SoA stride
    const int zstr = P.soa ? cap : 1;                  // per-zone entry stride (SoA vs AoS)
    int cur = exist_cnt[b];

  for (int t = 0; t < new_cnt[b]; ++t) {
    const int widx = new_idx[new_off[b] + t];
    for (int k = int(lane); k < d2; k += 32) zn[k] = oz[(ulong)widx * d2 + k];
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Phase A: is zn subsumed by an existing zone? Lanes split the antichain; simd_any reduces.
    // Read W[i] DIRECTLY from device (read-only test) — no private-memory spill of the operand.
    bool sub = false;
    for (int i = int(lane); i < cur; i += 32) {
      SoaZone zw{W + (P.soa ? (ulong)i : (ulong)i * d2), zstr};
      // zn subsumed by existing zw?  diag: zw ⊇ zn (d_includes(zw,zn)); non-diag: aLU(zn≼zw).
      if (has_diag ? g_includes(zw, zn, dim) : g_alu_le(zn, zw, dim, L, U)) sub = true;
    }
    if (simd_any(sub)) { simdgroup_barrier(mem_flags::mem_threadgroup); continue; }

    // Phase B: drop existing zones subsumed by zn, compacting W in place. Chunked
    // stream-compaction: each lane tests one zone, ballot+popcount gives its packed slot.
    int w = 0;
    for (int base = 0; base < cur; base += 32) {
      const int i = base + int(lane);
      int zw[MAXDIM * MAXDIM];
      bool keep = false;
      int isnew_i = 0;
      if (i < cur) {
        for (int k = 0; k < d2; ++k) zw[k] = W[P.soa ? (k*cap + i) : (i*d2 + k)];
        const bool kill = has_diag ? g_includes(zn, (thread const int*)zw, dim)
                                   : g_alu_le((thread const int*)zw, zn, dim, L, U);
        keep = !kill;
        isnew_i = WI[i];
      }
      const uint kb = (uint)(simd_vote::vote_t)simd_ballot(keep);
      const int dst = w + popcount(kb & ((1u << lane) - 1u));
      simdgroup_barrier(mem_flags::mem_threadgroup);   // all reads of W done before any write
      if (keep) { for (int k = 0; k < d2; ++k) W[P.soa ? (k*cap + dst) : (dst*d2 + k)] = zw[k]; WI[dst] = isnew_i; }
      w += popcount(kb);
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    cur = w;

    // Phase C: convex-union absorb. Lanes test different candidates in parallel (pure test);
    // the first that unions is applied cooperatively (hull + cooperative close), the absorbed
    // zone is shifted out, and we repeat until no candidate unions. Skippable (union_on): it is
    // a compression step only, so skipping it never affects correctness — just stores more zones.
    while (P.union_on) {
      // Bounded convex-union: only attempt to merge zn against the last `union_on` antichain
      // members (aligned down to a warp). Caps union cost at O(K*dim^3) while still catching the
      // common case (temporally-adjacent zones merge). union_on is the cap; a huge value = full.
      int cbase = (cur > P.union_on ? cur - P.union_on : 0) & ~31;
      int winner = -1;
      for (int base = cbase; base < cur && winner < 0; base += 32) {
        const int kk = base + int(lane);
        bool ok = false;
        if (kk < cur)   // read-only convex-union test straight from device (AoS/SoA strided view)
          ok = g_union_convex(zn, SoaZone{W + (P.soa ? (ulong)kk : (ulong)kk * d2), zstr}, dim);
        const uint vb = (uint)(simd_vote::vote_t)simd_ballot(ok);
        if (vb != 0u) winner = base + int(ctz(vb));
      }
      if (winner < 0) break;
      for (int k = int(lane); k < d2; k += 32) {       // zn := elementwise-max hull
        const int wv = W[P.soa ? (k*cap + winner) : (winner*d2 + k)];
        if (wv > zn[k]) zn[k] = wv;
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      // No re-close: the entrywise max of two canonical DBMs is canonical
      // (h_ij = max(a_ij,b_ij) <= max(a_ik+a_kj, b_ik+b_kj) <= h_ik + h_kj).
      for (int i = winner; i < cur - 1; ++i) {          // shift the absorbed zone out
        for (int k = int(lane); k < d2; k += 32)
          W[P.soa ? (k*cap + i) : (i*d2 + k)] = W[P.soa ? (k*cap + (i+1)) : ((i+1)*d2 + k)];
        if (lane == 0) WI[i] = WI[i+1];
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
      --cur;
    }

    for (int k = int(lane); k < d2; k += 32) W[P.soa ? (k*cap + cur) : (cur*d2 + k)] = zn[k];   // append
    if (lane == 0) WI[cur] = -1;   // -1 = 'this merge created it' (see net_gpu.hpp)
    ++cur;
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
    if (lane == 0) out_count[b] = cur;
  }  // work-stealing loop
}
)MSL";

using netgpu::NetEncoding;  // shared network encoding (net_gpu_encode.hpp)
using netgpu::encode;

class NetGpuMetal : public NetGpuExpander {
 public:
  // `net` must outlive this expander (it does: reachable_net owns it for the run).
  // Needed because gen_guard/gen_update are evaluated host-side via gen_ok/gen_apply.
  NetGpuMetal(const Network& net, NetEncoding&& en) : net_(&net), en_(std::move(en)) {
    device_ = MTLCreateSystemDefaultDevice();
    PTCTLMC_MTL_CHECK(device_ != nil, "no system default Metal device");
    // BOUNDED command-buffer pool. Each compute encoder makes the driver allocate a scratch
    // region sized by the kernel's per-thread stack times the threads it will keep in flight
    // (measured: ~5.4 MB apiece, and ~1950 of them live = 10.3 GB on fischer12 -- the real
    // occupant of the process's MALLOC_LARGE, dwarfing our own pools). The default queue keeps
    // up to 64 of those alive; we submit strictly one command buffer at a time and wait on it,
    // so a small pool is all we can ever use.
    queue_ = [device_ newCommandQueueWithMaxCommandBufferCount:2];
    NSError* err = nil;
    // Specialize the shader for THIS network: the kernels' private arrays and loop bounds are
    // then the model's real sizes, not the 24-clock / 64-process worst case (see the PT_DIM
    // block at the top of kNetSource). One extra source compile per run, already paid.
    MTLCompileOptions* copts = [[MTLCompileOptions alloc] init];
    copts.preprocessorMacros = @{
      @"PT_DIM" : @(en_.dim),
      @"PT_NP"  : @(en_.nprocs),
      @"PT_NI"  : @(en_.nintvars),
      @"PT_NC"  : @(en_.nclocks),
    };
    copts.fastMathEnabled = NO;   // integer-only kernels; keep semantics untouched
    id<MTLLibrary> lib = [device_ newLibraryWithSource:kNetSource options:copts error:&err];
    PTCTLMC_MTL_CHECK(lib != nil, std::string("shader compile failed: ") +
                                      (err ? err.localizedDescription.UTF8String : "?"));
    id<MTLFunction> fn = [lib newFunctionWithName:@"net_expand"];
    PTCTLMC_MTL_CHECK(fn != nil, "kernel 'net_expand' not found");
    pso_ = [device_ newComputePipelineStateWithFunction:fn error:&err];
    PTCTLMC_MTL_CHECK(pso_ != nil, "pipeline creation failed");
    // Cooperative (SIMD-group-per-successor) expand: opt-in via PTCTLMC_MTLSG=1 so it can
    // be A/B'd against the per-thread kernel and fall back cleanly.
    g_eprof.on = std::getenv("PTCTLMC_MTLPROF") != nullptr;
#ifdef PTCTLMC_HAVE_OPENMP
    eth_ = omp_get_max_threads();
#endif
    scr_.resize(eth_);
    b_sym_proc_ = b_sym_clk_ = b_sym_ivoff_ = b_sym_ivdata_ = ibuf(std::vector<int>{0});
    use_sg_ = std::getenv("PTCTLMC_MTLSG") != nullptr;
    use_tg_ = std::getenv("PTCTLMC_MTLTG") != nullptr;
    use_rg_ = std::getenv("PTCTLMC_MTLRG") != nullptr;
    if (use_sg_) {
      id<MTLFunction> fsg = [lib newFunctionWithName:@"net_expand_sg"];
      PTCTLMC_MTL_CHECK(fsg != nil, "kernel 'net_expand_sg' not found");
      pso_sg_ = [device_ newComputePipelineStateWithFunction:fsg error:&err];
      PTCTLMC_MTL_CHECK(pso_sg_ != nil, "cooperative pipeline creation failed");
    }
    {   // row-distributed expand (PTCTLMC_MTLRG=1)
      id<MTLFunction> frg = [lib newFunctionWithName:@"net_expand_rg"];
      if (frg) pso_rg_ = [device_ newComputePipelineStateWithFunction:frg error:&err];
    }
    {   // threadgroup-resident expand (PTCTLMC_MTLTG=1)
      id<MTLFunction> ftg = [lib newFunctionWithName:@"net_expand_tg"];
      if (ftg) pso_tg_ = [device_ newComputePipelineStateWithFunction:ftg error:&err];
    }
    id<MTLFunction> fm = [lib newFunctionWithName:@"net_merge"];
    PTCTLMC_MTL_CHECK(fm != nil, "kernel 'net_merge' not found");
    pso_merge_ = [device_ newComputePipelineStateWithFunction:fm error:&err];
    PTCTLMC_MTL_CHECK(pso_merge_ != nil, "merge pipeline creation failed");
    // gpusplit: SIMD-group-per-bucket cooperative work-stealing merge is now the DEFAULT GPU
    // merge (PTCTLMC_NOGPUSPLIT falls back to the one-thread-per-bucket net_merge for A/B).
    use_gpusplit_ = std::getenv("PTCTLMC_NOGPUSPLIT") == nullptr;
    if (use_gpusplit_) {
      id<MTLFunction> fmsg = [lib newFunctionWithName:@"net_merge_sg"];
      PTCTLMC_MTL_CHECK(fmsg != nil, "kernel 'net_merge_sg' not found");
      pso_merge_sg_ = [device_ newComputePipelineStateWithFunction:fmsg error:&err];
      PTCTLMC_MTL_CHECK(pso_merge_sg_ != nil, "merge_sg pipeline creation failed");
    }
    // Occupancy reflection (PTCTLMC_MTLOCC): maxTotalThreadsPerThreadgroup is the
    // compiler's verdict on register pressure -- if it is far below the device
    // maximum (1024), each thread is holding too much private state (here a whole
    // dim*dim DBM) and the kernel is occupancy/spill bound, not ALU bound.
    if (std::getenv("PTCTLMC_MTLOCC")) {
      std::fprintf(stderr,
          "[mtlocc] expand: maxThreads/TG=%lu execWidth=%lu tgMem=%lu B | "
          "merge: maxThreads/TG=%lu execWidth=%lu | device maxTG=%lu\n",
          (unsigned long)pso_.maxTotalThreadsPerThreadgroup,
          (unsigned long)pso_.threadExecutionWidth,
          (unsigned long)pso_.staticThreadgroupMemoryLength,
          (unsigned long)pso_merge_.maxTotalThreadsPerThreadgroup,
          (unsigned long)pso_merge_.threadExecutionWidth,
          (unsigned long)device_.maxThreadsPerThreadgroup.width);
    }
    // static (per-run) buffers
    b_edge_tgt_ = ibuf(en_.edge_tgt); b_cg_off_ = ibuf(en_.cg_off); b_cg_data_ = ibuf(en_.cg_data);
    b_rs_off_ = ibuf(en_.rs_off); b_rs_clk_ = ibuf(en_.rs_clk);
    // int guards AND int updates are host-side now (en_.ig_* / en_.iu_*): the update has
    // to be, because gen_update is an expression the kernel cannot evaluate.
    b_locbase_ = ibuf(en_.locbase); b_inv_off_ = ibuf(en_.inv_off); b_inv_data_ = ibuf(en_.inv_data);
    b_frozen_ = ibuf(en_.frozen); b_active_ = ibuf(en_.active); b_qmask_ = ibuf(en_.qmask);
    b_actmask_ = [device_ newBufferWithBytes:(en_.active_mask.empty() ? (const void*)&kZeroPad_
                                                                      : (const void*)en_.active_mask.data())
                                      length:(en_.active_mask.empty() ? sizeof(std::uint32_t)
                                              : en_.active_mask.size() * sizeof(std::uint32_t))
                                     options:MTLResourceStorageModeShared];
  }

  // Hand the CPU expander the same unified-memory buffers the GPU expand would fill,
  // so a CPU-generated successor set can be merged on the GPU with no copies.
  void stage_expand(std::size_t nwork, int np, int ni, int d2,
                    int*& valid, int*& locs, int*& ints, raw_t*& zones,
                    bool preserve) override {
    @autoreleasepool {
      const std::size_t n = nwork ? nwork : 1;
      id<MTLBuffer> b_ov = grow_keep(pb_ov_, n * sizeof(int), preserve),
                    b_ol = grow_keep(pb_ol_, n * std::max<std::size_t>(np * sizeof(int), sizeof(std::uint64_t)), preserve),
                    b_oi = grow_keep(pb_oi_, n * (ni ? ni : 1) * sizeof(int), preserve),
                    b_oz = grow_keep(pb_oz_, n * d2 * sizeof(raw_t), preserve);
      valid = static_cast<int*>(b_ov.contents);
      locs  = static_cast<int*>(b_ol.contents);
      ints  = static_cast<int*>(b_oi.contents);
      zones = static_cast<raw_t*>(b_oz.contents);
    }
  }

  // Per-thread scratch for the recipe enumerator. Held by the engine (not the loop) so a level
  // costs zero allocations after the first: the vectors keep their capacity across levels.
  struct EnumScratch {
    std::vector<std::pair<int, int>> parts;
    std::vector<long> upd, cur_long;
    std::vector<std::pair<int, std::vector<int>>> pj;
    std::vector<std::size_t> pick;
    char pad[64];   // one cache line between threads' scratch objects
  };

  // Enumerate every successor recipe of frontier state `s`, calling `sink(parts, upd)` in
  // EXACTLY the order successors_net produces them (location edges, then binary handshakes,
  // then broadcasts). Order matters: it fixes the per-bucket arrival order the convex-union
  // merge is sensitive to. ONE body, used by both the counting pass and the writing pass, so
  // the two cannot drift out of agreement about how many recipes a state has.
  template <class Sink>
  void enum_state(int s, const int* sl, const long* si, EnumScratch& sc, Sink&& sink) const {
    const int np = en_.nprocs, ni = en_.nintvars;
    auto& parts = sc.parts;
    if (en_.has_general) sc.cur_long.assign(si, si + ni);

    auto int_ok_ge = [&](int ge) {
      for (int g = en_.ig_off[ge]; g < en_.ig_off[ge + 1]; ++g) {
        const int var = en_.ig_data[3*g], op = en_.ig_data[3*g+1], val = en_.ig_data[3*g+2];
        const int x = static_cast<int>(si[var]);
        bool ok;
        switch (op) { case 0: ok = x == val; break; case 1: ok = x != val; break; case 2: ok = x < val; break;
                      case 3: ok = x <= val; break; case 4: ok = x > val; break; default: ok = x >= val; }
        if (!ok) return false;
      }
      // General guard (arithmetic / array / call), evaluated with the SAME function the
      // CPU path uses so the two backends cannot drift.
      if (en_.has_general) {
        const NEdge* e = en_.edge_src[ge];
        if (e->gen_guard && !gen_ok(*net_, *e, sc.cur_long.data(), sc.cur_long.size())) return false;
      }
      return true;
    };
    // Applies the discrete integer update exactly as successors_net does: every participant's
    // int_updates in participant order (sender then receivers), THEN every participant's
    // gen_update. A faulting or out-of-range gen_update discards the transition, so the recipe
    // is never emitted — which is why the counting pass has to run this too.
    auto emit = [&]() {
      auto& upd = sc.upd;
      upd.assign(si, si + ni);
      for (auto& pr : parts)
        for (int u = en_.iu_off[pr.second]; u < en_.iu_off[pr.second + 1]; ++u)
          upd[en_.iu_data[2*u]] = en_.iu_data[2*u + 1];
      if (en_.has_general)
        for (auto& pr : parts)
          if (!gen_apply(*net_, *en_.edge_src[pr.second], upd.data(), upd.size())) return;
      sink(parts, upd);
    };

    bool src_comm = false;
    for (int q = 0; q < np; ++q) if (en_.frozen[en_.locbase[q] + sl[q]] & 1) { src_comm = true; break; }
    for (int p = 0; p < np; ++p) {                                  // location edges
      if (src_comm && !(en_.frozen[en_.locbase[p] + sl[p]] & 1)) continue;
      for (int ge : en_.loc_edges[p][sl[p]]) {
        if (en_.edge_chan[ge] >= 0 || !int_ok_ge(ge)) continue;
        parts.assign({{p, ge}}); emit();
      }
    }
    for (int i = 0; i < np; ++i) {                                  // binary handshakes
      for (int ges : en_.loc_edges[i][sl[i]]) {
        if (en_.edge_chan[ges] < 0 || en_.edge_dir[ges] != 1 || en_.chan_bcast[en_.edge_chan[ges]]) continue;
        if (!int_ok_ge(ges)) continue;
        const bool i_comm = en_.frozen[en_.locbase[i] + sl[i]] & 1;
        for (int j = 0; j < np; ++j) {
          if (j == i) continue;
          for (int ger : en_.loc_edges[j][sl[j]]) {
            if (en_.edge_chan[ger] != en_.edge_chan[ges] || en_.edge_dir[ger] != -1) continue;
            if (src_comm && !(i_comm || (en_.frozen[en_.locbase[j] + sl[j]] & 1))) continue;
            if (!int_ok_ge(ger)) continue;
            parts.assign({{i, ges}, {j, ger}}); emit();
          }
        }
      }
    }
    for (int i = 0; i < np; ++i) {                                  // broadcasts
      for (int ges : en_.loc_edges[i][sl[i]]) {
        if (en_.edge_chan[ges] < 0 || en_.edge_dir[ges] != 1 || !en_.chan_bcast[en_.edge_chan[ges]]) continue;
        if (!int_ok_ge(ges)) continue;
        auto& pj = sc.pj; pj.clear();                     // (proc, its enabled receivers)
        for (int j = 0; j < np; ++j) {
          if (j == i) continue;
          std::vector<int> es;
          for (int ger : en_.loc_edges[j][sl[j]])
            if (en_.edge_chan[ger] == en_.edge_chan[ges] && en_.edge_dir[ger] == -1 && int_ok_ge(ger))
              es.push_back(ger);
          if (!es.empty()) pj.push_back({j, std::move(es)});
        }
        auto& pick = sc.pick; pick.assign(pj.size(), 0);
        for (;;) {
          bool part_comm = en_.frozen[en_.locbase[i] + sl[i]] & 1;
          for (auto& pr : pj) if (en_.frozen[en_.locbase[pr.first] + sl[pr.first]] & 1) part_comm = true;
          if (!src_comm || part_comm) {
            parts.assign({{i, ges}});
            for (std::size_t t = 0; t < pj.size(); ++t) parts.push_back({pj[t].first, pj[t].second[pick[t]]});
            emit();
          }
          if (pj.empty()) break;
          std::size_t k = 0;
          for (; k < pj.size(); ++k) { if (++pick[k] < pj[k].second.size()) break; pick[k] = 0; }
          if (k == pj.size()) break;
        }
      }
    }
  }

  void set_key_layout(const std::vector<int>& dshift, const std::vector<int>& intlo) override {
    b_dshift_ = ibuf(dshift);
    b_intlo_ = ibuf(intlo);
  }

  bool set_symmetry(int ng, const std::vector<int>& proc, const std::vector<int>& clk,
                    const std::vector<int>& ivoff, const std::vector<int>& ivdata) override {
    if (ng > kMaxGroup) return false;
    ng_ = ng;
    b_sym_proc_ = ibuf(proc); b_sym_clk_ = ibuf(clk);
    b_sym_ivoff_ = ibuf(ivoff); b_sym_ivdata_ = ibuf(ivdata);
    return true;
  }

  std::size_t expand(int nstates, const std::vector<int>& locs, const std::vector<long>& ints,
                     const std::vector<raw_t>& zones, const int*& out_valid, const int*& out_locs,
                     const int*& out_ints, const raw_t*& out_zones,
                     const std::uint64_t*& out_dk) override {
    const int np = en_.nprocs, ni = en_.nintvars, d2 = en_.dim * en_.dim;
    out_valid = out_locs = out_ints = nullptr;
    out_zones = nullptr; out_dk = nullptr;

    const double _t0 = g_eprof.on ? EngClock::now() : 0.0;
    // ---- recipe enumeration: PARALLEL, two-pass, straight into the GPU buffers ------------
    // This used to be one serial loop that push_back'd into six host vectors and then uploaded
    // them into six freshly-allocated MTLBuffers per level. Measured at 286 ms of a 1756 ms
    // fischer11 run (and 719 ms of arbiter17's 3537 ms) it was the largest single serial
    // section left — an Amdahl ceiling that no kernel work can get under.
    //
    // Pass 1 counts each state's recipes and participants; a prefix sum turns those into exact
    // per-state write offsets; pass 2 re-runs the identical enumeration and writes each recipe
    // at its own slot. Because every state's slot range is disjoint and assigned by index, the
    // result is BIT-IDENTICAL to the serial version's ordering — the convex-union merge's
    // order sensitivity is preserved exactly. The double enumeration costs ~1.6x the scalar
    // work but spreads it over 12 cores, and it removes the host vectors entirely: the writing
    // pass targets unified memory the kernel reads, so there is no upload copy and no
    // per-level allocation.
    const int NT = eth_;
    if (static_cast<int>(scr_.size()) < NT) scr_.resize(NT);
    cnt_rec_.resize(static_cast<std::size_t>(nstates) + 1);
    cnt_part_.resize(static_cast<std::size_t>(nstates) + 1);
    const long nst = nstates;
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(NT)
#endif
    for (long s = 0; s < nst; ++s) {
      int tid = 0;
#ifdef PTCTLMC_HAVE_OPENMP
      tid = omp_get_thread_num();
#endif
      int nr = 0, npt = 0;
      enum_state(static_cast<int>(s), &locs[static_cast<std::size_t>(s) * np],
                 ints.data() + static_cast<std::size_t>(s) * ni, scr_[tid],
                 [&](const std::vector<std::pair<int, int>>& p, const std::vector<long>&) {
                   ++nr; npt += static_cast<int>(p.size());
                 });
      cnt_rec_[s] = nr; cnt_part_[s] = npt;
    }
    // Exclusive prefix sums, in place: cnt_rec_[s] becomes state s's first work-item index and
    // cnt_rec_[nstates] the total. O(nstates) integer adds over data already in cache.
    std::size_t acc_r = 0, acc_p = 0;
    for (long s = 0; s <= nst; ++s) {
      const int r = (s < nst) ? cnt_rec_[s] : 0, p = (s < nst) ? cnt_part_[s] : 0;
      cnt_rec_[s] = static_cast<int>(acc_r); cnt_part_[s] = static_cast<int>(acc_p);
      acc_r += r; acc_p += p;
    }
    const int nwork = static_cast<int>(acc_r);
    if (nwork == 0) { if (g_eprof.on) { g_eprof.enumerate += EngClock::now() - _t0; ++g_eprof.levels; } return 0; }

    // Persistent, grow-on-demand staging buffers. Nothing is carried across levels, so a plain
    // grow() is safe (the previous level's kernel has already been waited on).
    int* ws       = iptr(pb_ws_, acc_r);
    int* part_off = iptr(pb_po_, acc_r);
    int* part_cnt = iptr(pb_pc_, acc_r);
    int* part_proc= iptr(pb_pp_, acc_p);
    int* part_edge= iptr(pb_pe_, acc_p);
    int* wints    = iptr(pb_wi_in_, acc_r * static_cast<std::size_t>(ni ? ni : 1));
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(NT)
#endif
    for (long s = 0; s < nst; ++s) {
      if (cnt_rec_[s] == cnt_rec_[s + 1]) continue;    // this state emitted nothing
      int tid = 0;
#ifdef PTCTLMC_HAVE_OPENMP
      tid = omp_get_thread_num();
#endif
      int w = cnt_rec_[s], po = cnt_part_[s];
      enum_state(static_cast<int>(s), &locs[static_cast<std::size_t>(s) * np],
                 ints.data() + static_cast<std::size_t>(s) * ni, scr_[tid],
                 [&](const std::vector<std::pair<int, int>>& p, const std::vector<long>& upd) {
                   ws[w] = static_cast<int>(s);
                   part_off[w] = po;
                   part_cnt[w] = static_cast<int>(p.size());
                   for (auto& pr : p) { part_proc[po] = pr.first; part_edge[po] = pr.second; ++po; }
                   for (int v = 0; v < ni; ++v)
                     wints[static_cast<std::size_t>(w) * ni + v] = static_cast<int>(upd[v]);
                   ++w;
                 });
    }
    if (g_eprof.on) { g_eprof.enumerate += EngClock::now() - _t0; ++g_eprof.levels; }

    @autoreleasepool {
      const double _t1 = g_eprof.on ? EngClock::now() : 0.0;
      id<MTLBuffer> b_fz = grow(pb_fz_, (zones.size() ? zones.size() : 1) * sizeof(raw_t));
      if (!zones.empty()) std::memcpy(b_fz.contents, zones.data(), zones.size() * sizeof(raw_t));
      id<MTLBuffer> b_fl = pb_fl_; {   // frontier locations: persistent, filled in place
        int* p = iptr(pb_fl_, locs.size() ? locs.size() : 1);
        if (!locs.empty()) std::memcpy(p, locs.data(), locs.size() * sizeof(int));
        b_fl = pb_fl_;
      }
      id<MTLBuffer> b_wi = pb_wi_in_;
      id<MTLBuffer> b_ws = pb_ws_, b_po = pb_po_, b_pc = pb_pc_,
                    b_pp = pb_pp_, b_pe = pb_pe_;
      id<MTLBuffer> b_ov = grow(pb_ov_, static_cast<std::size_t>(nwork) * sizeof(int)),
                    b_ol = grow(pb_ol_, static_cast<std::size_t>(nwork) * sizeof(std::uint64_t)),
                    b_oi = grow(pb_oi_, static_cast<std::size_t>(nwork) * (ni ? ni : 1) * sizeof(int)),
                    b_oz = grow(pb_oz_, static_cast<std::size_t>(nwork) * d2 * sizeof(int));
      NetParams P{en_.dim, np, ni, en_.nclocks, nwork, static_cast<std::int32_t>(en_.qmask_bits), ng_};

      if (g_eprof.on) g_eprof.stage += EngClock::now() - _t1;
      const double _t2 = g_eprof.on ? EngClock::now() : 0.0;
      id<MTLCommandBuffer> cb = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      // Threadgroup-memory budget: T threads each need d2 ints on-chip. Apple GPUs expose
      // 32 KB per threadgroup, so fall back to the spilling private-array kernel when even
      // one execution width (32 lanes) will not fit.
      const NSUInteger d2b = (static_cast<NSUInteger>(P.dim) * P.dim + 1) * sizeof(int);  // +1: bank-conflict pad
      const NSUInteger tgcap = 32768;
      NSUInteger tg_threads = 0;
      const bool use_rg = use_rg_ && pso_rg_ && P.dim <= 32;
      const bool use_tg = !use_rg && use_tg_ && pso_tg_ && d2b * 32 <= tgcap;
      if (use_tg) {
        tg_threads = (tgcap / d2b) & ~static_cast<NSUInteger>(31);   // multiple of exec width
        if (tg_threads > pso_tg_.maxTotalThreadsPerThreadgroup) tg_threads = pso_tg_.maxTotalThreadsPerThreadgroup;
        if (tg_threads > 256) tg_threads = 256;                      // more resident zones != more parallelism
      }
      [enc setComputePipelineState:(use_rg ? pso_rg_ : (use_tg ? pso_tg_ : (use_sg_ ? pso_sg_ : pso_)))];
      id<MTLBuffer> bufs[23] = {b_fl, b_wi, b_fz, b_ws, b_po, b_pc, b_pp, b_pe, b_edge_tgt_, b_cg_off_,
                                b_cg_data_, b_rs_off_, b_rs_clk_, b_locbase_,
                                b_inv_off_, b_inv_data_, b_frozen_, b_active_, b_qmask_, b_ov, b_ol, b_oi, b_oz};
      for (int i = 0; i < 23; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
      [enc setBytes:&P length:sizeof(P) atIndex:23];
      [enc setBuffer:b_actmask_ offset:0 atIndex:24];
      [enc setBuffer:b_dshift_ offset:0 atIndex:25];
      [enc setBuffer:b_intlo_ offset:0 atIndex:26];
      [enc setBuffer:b_sym_proc_ offset:0 atIndex:27];
      [enc setBuffer:b_sym_clk_ offset:0 atIndex:28];
      [enc setBuffer:b_sym_ivoff_ offset:0 atIndex:29];
      [enc setBuffer:b_sym_ivdata_ offset:0 atIndex:30];
      if (use_rg) {
        const NSUInteger kRg = 8, kTg = kRg * 32;   // one SIMD group per zone
        const NSUInteger groups = (static_cast<NSUInteger>(nwork) + kRg - 1) / kRg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(kTg, 1, 1)];
      } else if (use_tg) {
        [enc setThreadgroupMemoryLength:tg_threads * d2b atIndex:0];
        [enc dispatchThreads:MTLSizeMake(nwork, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_threads, 1, 1)];
      } else if (use_sg_) {
        // 32 lanes per successor, SG_PER_TG(8) groups per threadgroup
        const NSUInteger kSg = 8, kTg = kSg * 32;
        const NSUInteger groups = (static_cast<NSUInteger>(nwork) + kSg - 1) / kSg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(kTg, 1, 1)];
      } else {
        NSUInteger tg = pso_.maxTotalThreadsPerThreadgroup;
        if (tg > static_cast<NSUInteger>(nwork)) tg = nwork;
        [enc dispatchThreads:MTLSizeMake(nwork, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      }
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      PTCTLMC_MTL_CHECK(cb.status == MTLCommandBufferStatusCompleted, "kernel failed");
      if (g_eprof.on) g_eprof.kernel += EngClock::now() - _t2;

      out_valid = static_cast<const int*>(b_ov.contents);   // read valid rows in place
      out_dk = static_cast<const std::uint64_t*>(b_ol.contents);
      out_ints = static_cast<const int*>(b_oi.contents);
      out_zones = static_cast<const raw_t*>(b_oz.contents);
      return static_cast<std::size_t>(nwork);
    }
  }

  void stage_merge(std::size_t work_zones, int d2, raw_t*& work_buf, int*& isnew) override {
    id<MTLBuffer> bw = grow(pb_work_, (work_zones ? work_zones : 1) * d2 * sizeof(raw_t));
    id<MTLBuffer> bi = grow(pb_wi_, (work_zones ? work_zones : 1) * sizeof(int));
    std::memset(bi.contents, 0xff, (work_zones ? work_zones : 1) * sizeof(int));  // -1 = created here
    work_buf = static_cast<raw_t*>(bw.contents);   // caller fills existing directly (unified memory)
    isnew = static_cast<int*>(bi.contents);        // ...and seeds their level-new flags
  }

  void merge(int nb, int dim, bool has_diag,
             const std::vector<int>& work_off, const std::vector<int>& exist_cnt,
             const std::vector<int>& new_idx, const std::vector<int>& new_off,
             const std::vector<int>& new_cnt,
             const std::vector<std::int32_t>& L, const std::vector<std::int32_t>& U,
             std::vector<int>& out_work_off, std::vector<int>& out_count,
             const raw_t*& out_zones, const int*& out_isnew) override {
    out_work_off.clear();
    out_count.assign(nb, 0);
    out_zones = nullptr;
    out_isnew = nullptr;
    if (nb == 0) return;
    @autoreleasepool {
      const double _m0 = g_eprof.on ? EngClock::now() : 0.0;
      id<MTLBuffer> b_work = pb_work_, b_wi = pb_wi_;   // work in place; new zones via interned idx
      // Persistent staging (was: eight fresh MTLBuffers per level — see pb_ws_ &c).
      auto fill = [&](__strong id<MTLBuffer>& b, const std::vector<int>& v) {
        int* p = iptr(b, v.size() ? v.size() : 1);
        if (!v.empty()) std::memcpy(p, v.data(), v.size() * sizeof(int));
        return b;
      };
      auto fill32 = [&](__strong id<MTLBuffer>& b, const std::vector<std::int32_t>& v) {
        int* p = iptr(b, v.size() ? v.size() : 1);
        if (!v.empty()) std::memcpy(p, v.data(), v.size() * sizeof(std::int32_t));
        return b;
      };
      id<MTLBuffer> b_woff = fill(pb_moff_, work_off), b_ec = fill(pb_mec_, exist_cnt),
                    b_ni = fill(pb_mni_, new_idx), b_noff = fill(pb_mnoff_, new_off),
                    b_nc = fill(pb_mnc_, new_cnt), b_L = fill32(pb_mL_, L), b_U = fill32(pb_mU_, U);
      id<MTLBuffer> b_oc = pb_moc_; { iptr(pb_moc_, nb); b_oc = pb_moc_; }
      // Convex-union is ON by default on the GPU merge. It used to be off because the test was
      // O(dim^5) per pair and cost 60%+ of the merge; with the minimal-graph facet set and the
      // incremental close (dbm_try_union, dbm_ops.hpp) it is O(dim^3) and measures free or
      // better -- while cutting stored states 37% on csma14 and peak RSS 24% on fischer12,
      // because every zone it absorbs is a zone the next level does not expand.
      // union_on doubles as the bounded-union CAP: 0 = off, K = merge against the last K only,
      // huge = full. PTCTLMC_NOUNION=1 disables it; PTCTLMC_UNIONCAP=K for the bounded variant.
      int union_cap = 1 << 20;
      if (const char* c = std::getenv("PTCTLMC_UNIONCAP")) union_cap = std::atoi(c);
      else if (std::getenv("PTCTLMC_NOUNION")) union_cap = 0;
      MergeParamsGPU MP{dim, nb, has_diag ? 1 : 0, union_cap, soa_work_pool() ? 1 : 0};
      if (g_eprof.on) g_eprof.mstage += EngClock::now() - _m0;
      const double _m1 = g_eprof.on ? EngClock::now() : 0.0;
      id<MTLCommandBuffer> cb = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      id<MTLBuffer> bufs[11] = {b_woff, b_ec, b_noff, b_nc, pb_oz_, b_ni, b_L, b_U, b_work, b_wi, b_oc};
      if (use_gpusplit_) {
        // gpusplit: one SIMD-group (32 lanes) per bucket; SG_PER_TG groups per threadgroup.
        // zn threadgroup scratch = SG_PER_TG * d2 ints (must match SG_PER_TG in the kernel).
        constexpr NSUInteger kSg = 8, kTg = kSg * 32;
        // Work-stealing bucket counter (device atomic int), reset to 0 for this merge.
        id<MTLBuffer> b_next = newi(1);
        *static_cast<int*>(b_next.contents) = 0;
        [enc setComputePipelineState:pso_merge_sg_];
        for (int i = 0; i < 11; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
        [enc setBytes:&MP length:sizeof(MP) atIndex:11];
        [enc setBuffer:b_next offset:0 atIndex:12];
        [enc setThreadgroupMemoryLength:kSg * static_cast<NSUInteger>(dim) * dim * sizeof(int) atIndex:0];
        // FIXED GPU-saturating grid: SIMD-groups steal buckets from the atomic counter, so the
        // grid size decouples from nb. Cap so groups are long-lived (steal many) yet saturate.
        NSUInteger wantGroups = (static_cast<NSUInteger>(nb) + kSg - 1) / kSg;
        const NSUInteger groups = wantGroups < 128 ? wantGroups : 128;   // 128*8 = 1024 SIMD-groups
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTg, 1, 1)];
      } else {
        [enc setComputePipelineState:pso_merge_];
        for (int i = 0; i < 11; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
        [enc setBytes:&MP length:sizeof(MP) atIndex:11];
        NSUInteger tg = pso_merge_.maxTotalThreadsPerThreadgroup;
        if (tg > static_cast<NSUInteger>(nb)) tg = nb;
        [enc dispatchThreads:MTLSizeMake(nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      }
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      PTCTLMC_MTL_CHECK(cb.status == MTLCommandBufferStatusCompleted, "merge kernel failed");
      if (g_eprof.on) g_eprof.mkernel += EngClock::now() - _m1;
      const int* oc = static_cast<const int*>(b_oc.contents);
      out_work_off = work_off;
      for (int b = 0; b < nb; ++b) out_count[b] = oc[b];
      out_zones = static_cast<const raw_t*>(b_work.contents);
      out_isnew = static_cast<const int*>(b_wi.contents);
    }
  }

 private:
  struct MergeParamsGPU { std::int32_t dim, nb, has_diag, union_on, soa; };
  id<MTLBuffer> Lbuf(const std::vector<std::int32_t>& v) {
    const void* p = v.empty() ? (const void*)&kZeroPad_ : (const void*)v.data();
    std::size_t bytes = v.empty() ? sizeof(std::int32_t) : v.size() * sizeof(std::int32_t);
    return [device_ newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
  }
  id<MTLBuffer> ibuf(const std::vector<int>& v) {
    const void* p = v.empty() ? (const void*)&kZeroPad_ : (const void*)v.data();
    std::size_t bytes = v.empty() ? sizeof(int) : v.size() * sizeof(int);
    return [device_ newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
  }
  id<MTLBuffer> rbuf(const std::vector<raw_t>& v) {
    const void* p = v.empty() ? (const void*)&kZeroPad_ : (const void*)v.data();
    std::size_t bytes = v.empty() ? sizeof(raw_t) : v.size() * sizeof(raw_t);
    return [device_ newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
  }
  id<MTLBuffer> newi(std::size_t count) {
    return [device_ newBufferWithLength:(count ? count : 1) * sizeof(int) options:MTLResourceStorageModeShared];
  }
  // Persistent, grow-on-demand shared buffer: reused across levels so we never re-allocate
  // (and never free) the big per-level zone pools — the profiler showed that alloc + upload
  // of hundreds of MB per level was the metal bottleneck. Returns the buffer; caller writes
  // straight into `.contents` (unified memory), skipping the intermediate host vector + copy.
  // As grow(), but carries the existing bytes over to the new allocation. Only the
  // hybrid overflow path needs this; plain grow() stays copy-free.
  id<MTLBuffer> grow_keep(__strong id<MTLBuffer>& b, std::size_t bytes, bool keep) {
    if (!keep) return grow(b, bytes);
    if (bytes == 0) bytes = 16;
    if (b != nil && static_cast<std::size_t>(b.length) >= bytes) return b;
    const std::size_t want = std::max(bytes, static_cast<std::size_t>(b ? b.length * 3 / 2 : 0));
    id<MTLBuffer> nb = [device_ newBufferWithLength:want options:MTLResourceStorageModeShared];
    if (b != nil) std::memcpy(nb.contents, b.contents, std::min<std::size_t>(b.length, want));
    b = nb;
    return b;
  }

  // grow() a persistent buffer to hold `count` ints and hand back a host-writable pointer.
  int* iptr(__strong id<MTLBuffer>& b, std::size_t count) {
    return static_cast<int*>(grow(b, (count ? count : 1) * sizeof(int)).contents);
  }

  id<MTLBuffer> grow(__strong id<MTLBuffer>& b, std::size_t bytes) {
    if (bytes == 0) bytes = 16;
    if (b == nil || static_cast<std::size_t>(b.length) < bytes) {
      // Release the OLD pool before allocating the new one. Otherwise both are live across the
      // allocation and the driver reclaims the old pages lazily, so several generations of a
      // growing pool stay resident — which measured as GBs of RSS above what the buffers report.
      // Grow geometrically (>=1.5x) so a steadily-widening BFS re-allocates a handful of times
      // instead of on nearly every level.
      // Grow GEOMETRICALLY (>=1.5x). Measured: with exact sizing a steadily-widening BFS re-allocates
      // the pool on nearly every level, and each dropped generation is reclaimed by the driver only
      // lazily, so ~1 GB of stale pools stays resident. Over-allocating ~0.5 GB to make growth rare is
      // a net ~0.5 GB win (5.00 -> 4.50 GB on Fischer N11) at no cost in run time.
      const std::size_t want = std::max(bytes, static_cast<std::size_t>(b ? b.length * 3 / 2 : 0));
      // Releasing the old buffer is not enough: measured on fischer12, our live pools summed to
      // 6.35 GB while Metal had 11.26 GB MAPPED — ~5 GB of pages belonging to dropped pool
      // generations that the driver reclaims only lazily. Marking the buffer EMPTY declares its
      // contents dead so the pages come back now. Safe here because grow() does not preserve
      // contents (grow_keep copies first, then calls through to this).
      if (b != nil) { [b setPurgeableState:MTLPurgeableStateEmpty]; b = nil; }
      b = [device_ newBufferWithLength:want options:MTLResourceStorageModeShared];
    }
    return b;
  }

  const Network* net_ = nullptr;   // for gen_ok/gen_apply; outlives this object
  NetEncoding en_;
  int kZeroPad_ = 0;
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLComputePipelineState> pso_ = nil;
  id<MTLComputePipelineState> pso_merge_ = nil;
  id<MTLComputePipelineState> pso_merge_sg_ = nil;   // gpusplit cooperative merge
  bool use_gpusplit_ = false;
  id<MTLComputePipelineState> pso_sg_ = nil;   // cooperative expand (PTCTLMC_MTLSG)
  bool use_sg_ = false;
  bool use_tg_ = false;
  id<MTLComputePipelineState> pso_tg_ = nil;
  bool use_rg_ = false;
  id<MTLComputePipelineState> pso_rg_ = nil;
  // No b_iu_* here: the integer update is applied host-side (gen_update is an
  // expression, so the kernel cannot replay it from a flat table).
  id<MTLBuffer> b_edge_tgt_, b_cg_off_, b_cg_data_, b_rs_off_, b_rs_clk_,
      b_locbase_, b_inv_off_, b_inv_data_, b_frozen_, b_active_, b_qmask_, b_actmask_,
      b_dshift_, b_intlo_, b_sym_proc_, b_sym_clk_, b_sym_ivoff_, b_sym_ivdata_;
  int ng_ = 1;
  id<MTLBuffer> pb_fz_, pb_oz_, pb_ol_, pb_oi_, pb_ov_;   // persistent expand in/out zone+disc pools
  id<MTLBuffer> pb_work_, pb_wi_;                         // persistent merge work pool + isnew
  // Persistent expand INPUT staging (recipes + frontier locs + post-update ints). These used to
  // be freshly allocated MTLBuffers every level; on fischer11 the transient generations were
  // ~1.6 GB of RSS the memory profiler could not account for.
  id<MTLBuffer> pb_ws_, pb_po_, pb_pc_, pb_pp_, pb_pe_, pb_wi_in_, pb_fl_;
  // Persistent merge INPUT staging (the same problem on the merge side).
  id<MTLBuffer> pb_moff_, pb_mec_, pb_mni_, pb_mnoff_, pb_mnc_, pb_mL_, pb_mU_, pb_moc_;
  std::vector<EnumScratch> scr_;              // per-thread enumerator scratch, reused across levels
  std::vector<int> cnt_rec_, cnt_part_;       // per-state recipe/participant counts -> offsets
  int eth_ = 1;                               // enumeration thread count

 public:
  // Sum of every persistent unified-memory pool (diagnostics: PTCTLMC_MEMPROF).
  std::size_t device_bytes() const override {
    // What the DRIVER actually has mapped, not just what our live buffers add up to: a grown
    // pool's predecessor stays resident until the driver reclaims it, and that gap was several
    // GB of "unaccounted" RSS in the memory profile.
    return static_cast<std::size_t>(device_.currentAllocatedSize);
  }
  std::size_t live_buffer_bytes() const override {
    std::size_t t = 0;
    for (id<MTLBuffer> b : {pb_fz_, pb_oz_, pb_ol_, pb_oi_, pb_ov_, pb_work_, pb_wi_,
                            pb_ws_, pb_po_, pb_pc_, pb_pp_, pb_pe_, pb_wi_in_, pb_fl_,
                            pb_moff_, pb_mec_, pb_mni_, pb_mnoff_, pb_mnc_, pb_mL_, pb_mU_, pb_moc_})
      if (b != nil) t += static_cast<std::size_t>(b.length);
    return t;
  }
  // net_merge_sg can read/write the work pool transposed (SoA) for coalesced cross-lane loads.
  // Opt-in via PTCTLMC_SOA (default AoS — SoA lost in back-to-back tests, see kernel notes).
  bool soa_work_pool() const override { return use_gpusplit_ && std::getenv("PTCTLMC_SOA") != nullptr; }

 private:
};

}  // namespace

std::unique_ptr<NetGpuExpander> make_net_gpu_metal(const Network& net,
                                                   const std::vector<char>& qmask) {
  // scope: non-diagonal networks (channels + shared ints are handled). Diagonal models use
  // the CPU exact-zone fallback.
  if (netgpu::network_has_diagonal(net)) return nullptr;
  if (net.nclocks + 1 > kMaxDim) return nullptr;
  if (net.nprocs() > 64 || static_cast<int>(net.intvars.size()) > 32) return nullptr;  // MAXPROC/MAXINT
  if (MTLCreateSystemDefaultDevice() == nil) return nullptr;
  try {
    return std::make_unique<NetGpuMetal>(net, encode(net, qmask));
  } catch (const std::exception& e) {
    if (std::getenv("PTCTLMC_MTLDEBUG")) std::fprintf(stderr, "[mtldebug] %s\n", e.what());
    return nullptr;  // any Metal failure -> CPU fallback
  } catch (...) {
    return nullptr;  // any Metal failure -> CPU fallback
  }
}

}  // namespace ptctlmc

#endif  // PTCTLMC_HAVE_METAL
