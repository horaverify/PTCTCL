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
#include <cstdint>
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

namespace ptctlmc {
namespace {

constexpr int kMaxDim = 24;  // <=23 clocks (matches CUDA / single-TA engine)

#define PTCTLMC_MTL_CHECK(cond, msg) \
  do { if (!(cond)) throw std::runtime_error(std::string("Metal(net): ") + (msg)); } while (0)

struct NetParams {
  std::int32_t dim, nprocs, nintvars, nclocks, nwork;
};

// ---- MSL: DBM ops (port of dbm_ops.hpp) + network successor kernel -------------------
static NSString* const kNetSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define MAXDIM 24
#define MAXPROC 64
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
  if (bound >= d[a*dim+b]) return !d_empty(d, dim);
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
  for (int k = 0; k < dim; ++k)
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
// if a ∪ b is exactly convex, set a := hull and return true; else leave a and return false.
inline bool d_try_union(thread int* a, thread const int* b, int dim) {
  int H[MAXDIM*MAXDIM], R[MAXDIM*MAXDIM];
  for (int k = 0; k < dim*dim; ++k) H[k] = a[k] > b[k] ? a[k] : b[k];
  for (int pass = 0; pass < 2; ++pass) {
    thread const int* z = (pass == 0) ? a : b;
    thread const int* other = (pass == 0) ? b : a;
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j) {
        if (i == j || z[i*dim+j] >= H[i*dim+j]) continue;
        for (int k = 0; k < dim*dim; ++k) R[k] = H[k];
        int v = r_bound(z[i*dim+j]);
        int neg = r_strict(z[i*dim+j]) ? r_le(-v) : r_lt(-v);
        if (neg < R[j*dim+i]) { R[j*dim+i] = neg; d_close(R, dim); }
        if (d_empty(R, dim)) continue;
        if (!d_includes(other, R, dim)) return false;
      }
  }
  for (int k = 0; k < dim*dim; ++k) a[k] = H[k];
  d_close(a, dim);
  return true;
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

struct Params { int dim, nprocs, nintvars, nclocks, nwork; };
struct MergeParams { int dim, nb, has_diag; };

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
    device int*       out_locs   [[buffer(20)]],
    device int*       out_ints   [[buffer(21)]],
    device int*       out_zone   [[buffer(22)]],
    constant Params&  P          [[buffer(23)]],
    uint w [[thread_position_in_grid]])
{
  if (w >= uint(P.nwork)) return;
  const int dim = P.dim, d2 = dim * dim, nprocs = P.nprocs, nintvars = P.nintvars, nclocks = P.nclocks;
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

  // reset clocks inactive at every process's target location and not read by the query
  for (int cc = 1; cc <= nclocks; ++cc) {
    bool any = false;
    for (int q = 0; q < nprocs; ++q)
      if (active[(locbase[q] + nlocs[q]) * (nclocks + 1) + cc]) { any = true; break; }
    if (!any && !qmask[cc]) d_reset(z, dim, cc);
  }
  if (d_empty(z, dim)) return;

  out_valid[w] = 1;
  for (int q = 0; q < nprocs; ++q) out_locs[w*nprocs + q] = nlocs[q];
  // integers: already updated host-side for this recipe (see the header comment)
  for (int v = 0; v < nintvars; ++v) out_ints[w*nintvars + v] = work_ints[w*nintvars + v];
  for (int k = 0; k < d2; ++k) out_zone[w*d2 + k] = z[k];
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
  const int dim = P.dim, d2 = dim * dim, nprocs = P.nprocs, nintvars = P.nintvars, nclocks = P.nclocks;
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
  const int dim = P.dim, d2 = dim * dim, nprocs = P.nprocs, nintvars = P.nintvars, nclocks = P.nclocks;
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
  const int dim = P.dim, d2 = dim*dim, nprocs = P.nprocs, nintvars = P.nintvars, nclocks = P.nclocks;
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
  const int dim = P.dim, d2 = dim * dim, has_diag = P.has_diag;
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
    WI[cur] = 1; ++cur;
  }
  out_count[b] = cur;
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
    queue_ = [device_ newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib = [device_ newLibraryWithSource:kNetSource options:nil error:&err];
    PTCTLMC_MTL_CHECK(lib != nil, std::string("shader compile failed: ") +
                                      (err ? err.localizedDescription.UTF8String : "?"));
    id<MTLFunction> fn = [lib newFunctionWithName:@"net_expand"];
    PTCTLMC_MTL_CHECK(fn != nil, "kernel 'net_expand' not found");
    pso_ = [device_ newComputePipelineStateWithFunction:fn error:&err];
    PTCTLMC_MTL_CHECK(pso_ != nil, "pipeline creation failed");
    // Cooperative (SIMD-group-per-successor) expand: opt-in via PTCTLMC_MTLSG=1 so it can
    // be A/B'd against the per-thread kernel and fall back cleanly.
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
  }

  // Hand the CPU expander the same unified-memory buffers the GPU expand would fill,
  // so a CPU-generated successor set can be merged on the GPU with no copies.
  void stage_expand(std::size_t nwork, int np, int ni, int d2,
                    int*& valid, int*& locs, int*& ints, raw_t*& zones,
                    bool preserve) override {
    @autoreleasepool {
      const std::size_t n = nwork ? nwork : 1;
      id<MTLBuffer> b_ov = grow_keep(pb_ov_, n * sizeof(int), preserve),
                    b_ol = grow_keep(pb_ol_, n * np * sizeof(int), preserve),
                    b_oi = grow_keep(pb_oi_, n * (ni ? ni : 1) * sizeof(int), preserve),
                    b_oz = grow_keep(pb_oz_, n * d2 * sizeof(raw_t), preserve);
      valid = static_cast<int*>(b_ov.contents);
      locs  = static_cast<int*>(b_ol.contents);
      ints  = static_cast<int*>(b_oi.contents);
      zones = static_cast<raw_t*>(b_oz.contents);
    }
  }

  std::size_t expand(int nstates, const std::vector<int>& locs, const std::vector<long>& ints,
                     const std::vector<raw_t>& zones, const int*& out_valid, const int*& out_locs,
                     const int*& out_ints, const raw_t*& out_zones) override {
    const int np = en_.nprocs, ni = en_.nintvars, d2 = en_.dim * en_.dim;
    out_valid = out_locs = out_ints = nullptr;
    out_zones = nullptr;

    // source ints as int32 for guard evaluation + the GPU
    std::vector<int> ints32(static_cast<std::size_t>(nstates) * ni);
    for (std::size_t i = 0; i < ints32.size(); ++i) ints32[i] = static_cast<int>(ints[i]);

    // Source ints of the state currently being expanded, as the CPU path's IntVec.
    // Only materialised for networks that actually carry gen_guard/gen_update, so
    // Fischer and friends pay nothing for it.
    std::vector<long> cur_long;

    auto int_ok_ge = [&](int ge, const int* si) {
      for (int g = en_.ig_off[ge]; g < en_.ig_off[ge + 1]; ++g) {
        const int var = en_.ig_data[3*g], op = en_.ig_data[3*g+1], val = en_.ig_data[3*g+2];
        const int x = si[var];
        bool ok;
        switch (op) { case 0: ok = x == val; break; case 1: ok = x != val; break; case 2: ok = x < val; break;
                      case 3: ok = x <= val; break; case 4: ok = x > val; break; default: ok = x >= val; }
        if (!ok) return false;
      }
      // General guard (arithmetic / array / call), evaluated with the SAME function the
      // CPU path uses so the two backends cannot drift.
      if (en_.has_general) {
        const NEdge* e = en_.edge_src[ge];
        if (e->gen_guard && !gen_ok(*net_, *e, cur_long.data(), cur_long.size())) return false;
      }
      return true;
    };

    // enumerate successor recipes (participant lists), mirroring successors_net EXACTLY:
    // per state — location edges, then binary handshakes, then broadcasts.
    std::vector<int> ws, part_off, part_cnt, part_proc, part_edge;
    std::vector<int> wints;                  // [nwork*ni] post-update ints, one row per recipe
    std::vector<std::pair<int, int>> parts;
    std::vector<long> upd;                              // scratch, reused across recipes

    // Applies the discrete integer update exactly as successors_net does: every
    // participant's int_updates in participant order (sender then receivers), THEN every
    // participant's gen_update. A faulting or out-of-range gen_update discards the
    // transition, so the recipe is never emitted.
    auto emit = [&](int s) {
      const long* src = ints.data() + static_cast<std::size_t>(s) * ni;
      upd.assign(src, src + ni);
      for (auto& pr : parts)
        for (int u = en_.iu_off[pr.second]; u < en_.iu_off[pr.second + 1]; ++u)
          upd[en_.iu_data[2*u]] = en_.iu_data[2*u + 1];
      if (en_.has_general)
        for (auto& pr : parts)
          if (!gen_apply(*net_, *en_.edge_src[pr.second], upd.data(), upd.size())) return;

      ws.push_back(s);
      part_off.push_back(static_cast<int>(part_proc.size()));
      part_cnt.push_back(static_cast<int>(parts.size()));
      for (auto& pr : parts) { part_proc.push_back(pr.first); part_edge.push_back(pr.second); }
      for (int v = 0; v < ni; ++v) wints.push_back(static_cast<int>(upd[v]));
    };
    for (int s = 0; s < nstates; ++s) {
      const int* sl = &locs[static_cast<std::size_t>(s) * np];
      const int* si = &ints32[static_cast<std::size_t>(s) * ni];
      if (en_.has_general) {
        const long* src = ints.data() + static_cast<std::size_t>(s) * ni;
        cur_long.assign(src, src + ni);
      }
      bool src_comm = false;
      for (int q = 0; q < np; ++q) if (en_.frozen[en_.locbase[q] + sl[q]] & 1) { src_comm = true; break; }
      for (int p = 0; p < np; ++p) {                                  // location edges
        if (src_comm && !(en_.frozen[en_.locbase[p] + sl[p]] & 1)) continue;
        for (int ge : en_.loc_edges[p][sl[p]]) {
          if (en_.edge_chan[ge] >= 0 || !int_ok_ge(ge, si)) continue;
          parts.assign({{p, ge}}); emit(s);
        }
      }
      for (int i = 0; i < np; ++i) {                                  // binary handshakes
        for (int ges : en_.loc_edges[i][sl[i]]) {
          if (en_.edge_chan[ges] < 0 || en_.edge_dir[ges] != 1 || en_.chan_bcast[en_.edge_chan[ges]]) continue;
          if (!int_ok_ge(ges, si)) continue;
          const bool i_comm = en_.frozen[en_.locbase[i] + sl[i]] & 1;
          for (int j = 0; j < np; ++j) {
            if (j == i) continue;
            for (int ger : en_.loc_edges[j][sl[j]]) {
              if (en_.edge_chan[ger] != en_.edge_chan[ges] || en_.edge_dir[ger] != -1) continue;
              if (src_comm && !(i_comm || (en_.frozen[en_.locbase[j] + sl[j]] & 1))) continue;
              if (!int_ok_ge(ger, si)) continue;
              parts.assign({{i, ges}, {j, ger}}); emit(s);
            }
          }
        }
      }
      for (int i = 0; i < np; ++i) {                                  // broadcasts
        for (int ges : en_.loc_edges[i][sl[i]]) {
          if (en_.edge_chan[ges] < 0 || en_.edge_dir[ges] != 1 || !en_.chan_bcast[en_.edge_chan[ges]]) continue;
          if (!int_ok_ge(ges, si)) continue;
          std::vector<std::pair<int, std::vector<int>>> pj;  // (proc, its enabled receivers)
          for (int j = 0; j < np; ++j) {
            if (j == i) continue;
            std::vector<int> es;
            for (int ger : en_.loc_edges[j][sl[j]])
              if (en_.edge_chan[ger] == en_.edge_chan[ges] && en_.edge_dir[ger] == -1 && int_ok_ge(ger, si))
                es.push_back(ger);
            if (!es.empty()) pj.push_back({j, std::move(es)});
          }
          std::vector<std::size_t> pick(pj.size(), 0);
          for (;;) {
            bool part_comm = en_.frozen[en_.locbase[i] + sl[i]] & 1;
            for (auto& pr : pj) if (en_.frozen[en_.locbase[pr.first] + sl[pr.first]] & 1) part_comm = true;
            if (!src_comm || part_comm) {
              parts.assign({{i, ges}});
              for (std::size_t t = 0; t < pj.size(); ++t) parts.push_back({pj[t].first, pj[t].second[pick[t]]});
              emit(s);
            }
            if (pj.empty()) break;
            std::size_t k = 0;
            for (; k < pj.size(); ++k) { if (++pick[k] < pj[k].second.size()) break; pick[k] = 0; }
            if (k == pj.size()) break;
          }
        }
      }
    }
    const int nwork = static_cast<int>(ws.size());
    if (nwork == 0) return 0;

    @autoreleasepool {
      id<MTLBuffer> b_fz = grow(pb_fz_, (zones.size() ? zones.size() : 1) * sizeof(raw_t));
      if (!zones.empty()) std::memcpy(b_fz.contents, zones.data(), zones.size() * sizeof(raw_t));
      id<MTLBuffer> b_fl = ibuf(locs), b_wi = ibuf(wints);
      id<MTLBuffer> b_ws = ibuf(ws), b_po = ibuf(part_off), b_pc = ibuf(part_cnt),
                    b_pp = ibuf(part_proc), b_pe = ibuf(part_edge);
      id<MTLBuffer> b_ov = grow(pb_ov_, static_cast<std::size_t>(nwork) * sizeof(int)),
                    b_ol = grow(pb_ol_, static_cast<std::size_t>(nwork) * np * sizeof(int)),
                    b_oi = grow(pb_oi_, static_cast<std::size_t>(nwork) * (ni ? ni : 1) * sizeof(int)),
                    b_oz = grow(pb_oz_, static_cast<std::size_t>(nwork) * d2 * sizeof(int));
      NetParams P{en_.dim, np, ni, en_.nclocks, nwork};

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

      out_valid = static_cast<const int*>(b_ov.contents);   // read valid rows in place
      out_locs = static_cast<const int*>(b_ol.contents);
      out_ints = static_cast<const int*>(b_oi.contents);
      out_zones = static_cast<const raw_t*>(b_oz.contents);
      return static_cast<std::size_t>(nwork);
    }
  }

  void stage_merge(std::size_t work_zones, int d2, raw_t*& work_buf) override {
    id<MTLBuffer> bw = grow(pb_work_, (work_zones ? work_zones : 1) * d2 * sizeof(raw_t));
    id<MTLBuffer> bi = grow(pb_wi_, (work_zones ? work_zones : 1) * sizeof(int));
    std::memset(bi.contents, 0, (work_zones ? work_zones : 1) * sizeof(int));  // existing isnew = 0
    work_buf = static_cast<raw_t*>(bw.contents);   // caller fills existing directly (unified memory)
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
      id<MTLBuffer> b_work = pb_work_, b_wi = pb_wi_;   // work in place; new zones via interned idx
      id<MTLBuffer> b_woff = ibuf(work_off), b_ec = ibuf(exist_cnt), b_ni = ibuf(new_idx),
                    b_noff = ibuf(new_off), b_nc = ibuf(new_cnt), b_L = Lbuf(L), b_U = Lbuf(U), b_oc = newi(nb);
      MergeParamsGPU MP{dim, nb, has_diag ? 1 : 0};
      id<MTLCommandBuffer> cb = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso_merge_];
      id<MTLBuffer> bufs[11] = {b_woff, b_ec, b_noff, b_nc, pb_oz_, b_ni, b_L, b_U, b_work, b_wi, b_oc};
      for (int i = 0; i < 11; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
      [enc setBytes:&MP length:sizeof(MP) atIndex:11];
      NSUInteger tg = pso_merge_.maxTotalThreadsPerThreadgroup;
      if (tg > static_cast<NSUInteger>(nb)) tg = nb;
      [enc dispatchThreads:MTLSizeMake(nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      PTCTLMC_MTL_CHECK(cb.status == MTLCommandBufferStatusCompleted, "merge kernel failed");
      const int* oc = static_cast<const int*>(b_oc.contents);
      out_work_off = work_off;
      for (int b = 0; b < nb; ++b) out_count[b] = oc[b];
      out_zones = static_cast<const raw_t*>(b_work.contents);
      out_isnew = static_cast<const int*>(b_wi.contents);
    }
  }

 private:
  struct MergeParamsGPU { std::int32_t dim, nb, has_diag; };
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
      b = nil;
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
  id<MTLComputePipelineState> pso_sg_ = nil;   // cooperative expand (PTCTLMC_MTLSG)
  bool use_sg_ = false;
  bool use_tg_ = false;
  id<MTLComputePipelineState> pso_tg_ = nil;
  bool use_rg_ = false;
  id<MTLComputePipelineState> pso_rg_ = nil;
  // No b_iu_* here: the integer update is applied host-side (gen_update is an
  // expression, so the kernel cannot replay it from a flat table).
  id<MTLBuffer> b_edge_tgt_, b_cg_off_, b_cg_data_, b_rs_off_, b_rs_clk_,
      b_locbase_, b_inv_off_, b_inv_data_, b_frozen_, b_active_, b_qmask_;
  id<MTLBuffer> pb_fz_, pb_oz_, pb_ol_, pb_oi_, pb_ov_;   // persistent expand in/out zone+disc pools
  id<MTLBuffer> pb_work_, pb_wi_;                         // persistent merge work pool + isnew

 public:
  // Sum of every persistent unified-memory pool (diagnostics: PTCTLMC_MEMPROF).
  std::size_t device_bytes() const override {
    std::size_t t = 0;
    for (id<MTLBuffer> b : {pb_fz_, pb_oz_, pb_ol_, pb_oi_, pb_ov_, pb_work_, pb_wi_})
      if (b != nil) t += static_cast<std::size_t>(b.length);
    return t;
  }

 private:
};

}  // namespace

std::unique_ptr<NetGpuExpander> make_net_gpu_metal(const Network& net,
                                                   const std::vector<char>& qmask) {
  // scope: non-diagonal networks (channels + shared ints are handled). Diagonal models use
  // the CPU exact-zone fallback.
  if (netgpu::network_has_diagonal(net)) return nullptr;
  if (net.nclocks + 1 > kMaxDim) return nullptr;
  if (net.nprocs() > 64 || static_cast<int>(net.intvars.size()) > 32) return nullptr;
  if (MTLCreateSystemDefaultDevice() == nil) return nullptr;
  try {
    return std::make_unique<NetGpuMetal>(net, encode(net, qmask));
  } catch (...) {
    return nullptr;  // any Metal failure -> CPU fallback
  }
}

}  // namespace ptctlmc

#endif  // PTCTLMC_HAVE_METAL
