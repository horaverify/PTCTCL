// frontier_kernels.cu -- the frontier engine's kernels on CUDA: expand, group, reduce, merge.
//
// A port of src/metal/frontier_kernels.hpp and of HDBM's GDBM (metal/gdbm.hpp). The execution
// unit is one warp, the CUDA counterpart of a SIMD-group: 32 lanes that vote with __any_sync and
// meet at __syncwarp. Every gdbm_* function must be called by all 32 lanes of a warp in uniform
// control flow. A working DBM lives in shared memory as `int`; read-only operands live in global
// memory as int8, int16 or int32, the three stored-entry widths.
//
// The one real difference from Metal is the memory model. Lanes of a Metal SIMD-group run in
// lockstep; CUDA lanes are independently scheduled, so every cross-lane communication through
// memory ends in a __syncwarp, which the Metal source leaves implicit.
//
// Zone encoding is hdbm's: (value << 1) | weak, infinity = 1 << 30.
#include <cub/cub.cuh>

#include "frontier_cuda_api.hpp"

namespace hora {
namespace cuda {

namespace {

constexpr unsigned FULL = 0xffffffffu;
constexpr int GDBM_INF = 1 << 30;
constexpr int GDBM_ZERO_LE = 1;
constexpr int GDBM_ALU_NEG_INF = -(1 << 29);
constexpr short GDBM_NINF = 0x7ffe;
constexpr signed char GDBM_SINF = 0x7e;
constexpr int GDBM_G = 32;

using i8 = signed char;
using i16 = short;
using i32 = int;

__constant__ Dims cD;

// ---- warp primitives ---------------------------------------------------------------------------
__device__ __forceinline__ bool simd_any(bool p) { return __any_sync(FULL, p); }
__device__ __forceinline__ u32 simd_sum(u32 v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
  return v;
}
__device__ __forceinline__ u32 simd_prefix_exclusive_sum(u32 v) {
  const int lane = threadIdx.x & 31;
  u32 x = v;
  for (int o = 1; o < 32; o <<= 1) {
    const u32 y = __shfl_up_sync(FULL, x, o);
    if (lane >= o) x += y;
  }
  return x - v;
}
__device__ __forceinline__ u32 simd_broadcast_first(u32 v) { return __shfl_sync(FULL, v, 0); }
__device__ __forceinline__ void gdbm_sync() { __syncwarp(); }
__device__ __forceinline__ bool gdbm_any(bool p) { return __any_sync(FULL, p); }

// ---- encoding ----------------------------------------------------------------------------------
__device__ __forceinline__ int gdbm_le(int v) { return (v << 1) | 1; }
__device__ __forceinline__ int gdbm_lt(int v) { return v << 1; }
__device__ __forceinline__ int gdbm_bound(int r) { return r >> 1; }
__device__ __forceinline__ bool gdbm_strict(int r) { return (r & 1) == 0; }
__device__ __forceinline__ int gdbm_add(int a, int b) {
  if (a >= GDBM_INF || b >= GDBM_INF) return GDBM_INF;
  const int v = gdbm_bound(a) + gdbm_bound(b);
  return (!gdbm_strict(a) && !gdbm_strict(b)) ? gdbm_le(v) : gdbm_lt(v);
}

__device__ __forceinline__ int gdbm_widen(i32 v) { return v; }
__device__ __forceinline__ int gdbm_widen(i16 v) { return v == GDBM_NINF ? GDBM_INF : int(v); }
__device__ __forceinline__ int gdbm_widen(i8 v) { return v == GDBM_SINF ? GDBM_INF : int(v); }
template <class P> __device__ __forceinline__ int gdbm_ld(const P* p, int k) { return gdbm_widen(p[k]); }

template <class GZ> __device__ __forceinline__ GZ gdbm_narrow(int v);
template <> __device__ __forceinline__ i32 gdbm_narrow<i32>(int v) { return v; }
template <> __device__ __forceinline__ i16 gdbm_narrow<i16>(int v) { return v >= GDBM_INF ? GDBM_NINF : i16(v); }
template <> __device__ __forceinline__ i8 gdbm_narrow<i8>(int v) { return v >= GDBM_INF ? GDBM_SINF : i8(v); }
// Does a finite entry fit the storage form? Infinity always does: it has its own sentinel.
__device__ __forceinline__ bool gdbm_fits(int v, i32) { return true; }
__device__ __forceinline__ bool gdbm_fits(int v, i16) { return v >= GDBM_INF || (v > -32767 && v < int(GDBM_NINF)); }
__device__ __forceinline__ bool gdbm_fits(int v, i8) { return v >= GDBM_INF || (v > -127 && v < int(GDBM_SINF)); }

// ---- thread mapping (GDBM Algorithm 3) ----------------------------------------------------------
#define GDBM_FOR_IJ(i, j, lane, n)                                                               \
  for (int i = int(lane) / (n), j = int(lane) % (n), gdbm_q_ = GDBM_G / (n),                     \
           gdbm_r_ = GDBM_G % (n);                                                               \
       i < (n); j += gdbm_r_, i += gdbm_q_ + (j >= (n) ? 1 : 0), j -= (j >= (n) ? (n) : 0))
#define GDBM_FOR_E(e, lane, nn) for (int e = int(lane); e < (nn); e += GDBM_G)

struct gdbm_cursor { int i, j, q, r; };
__device__ __forceinline__ gdbm_cursor gdbm_begin(u32 lane, int n) {
  gdbm_cursor c; c.i = int(lane) / n; c.j = int(lane) % n; c.q = GDBM_G / n; c.r = GDBM_G % n;
  return c;
}
__device__ __forceinline__ void gdbm_step(gdbm_cursor& c, int n) {
  c.j += c.r; c.i += c.q;
  if (c.j >= n) { c.j -= n; c.i += 1; }
}

// ---- moving a DBM in and out of shared memory ---------------------------------------------------
template <class P>
__device__ __forceinline__ void gdbm_load(int* D, const P* src, int n, u32 lane) {
  GDBM_FOR_E(e, lane, n * n) D[e] = gdbm_ld(src, e);
  gdbm_sync();
}

// ---- canonical form ------------------------------------------------------------------------------
__device__ void gdbm_close(int* D, int n, u32 lane) {
  for (int k = 0; k < n; ++k) {
    GDBM_FOR_IJ(i, j, lane, n) {
      const int s = gdbm_add(D[i * n + k], D[k * n + j]);
      if (s < D[i * n + j]) D[i * n + j] = s;
    }
    gdbm_sync();
  }
}

// Conjoin x_a - x_b < bound into a canonical, non-empty D. False when the result is empty.
__device__ bool gdbm_constrain(int* D, int n, int a, int b, int bound, u32 lane) {
  const int dab = D[a * n + b], dba = D[b * n + a];
  if (bound >= dab) return true;
  if (gdbm_add(dba, bound) < GDBM_ZERO_LE) return false;
  gdbm_sync();   // every lane has read D[a][b], D[b][a]
  GDBM_FOR_IJ(i, j, lane, n) {
    const int s = gdbm_add(gdbm_add(D[i * n + a], bound), D[b * n + j]);
    if (s < D[i * n + j]) D[i * n + j] = s;
  }
  gdbm_sync();
  return true;
}

__device__ void gdbm_reset(int* D, int n, int k, u32 lane) {
  for (int j = int(lane); j < n; j += GDBM_G) {
    if (j == k) continue;
    const int r = D[0 * n + j], c = D[j * n + 0];
    D[k * n + j] = r;
    D[j * n + k] = c;
  }
  if (lane == 0) D[k * n + k] = GDBM_ZERO_LE;
  gdbm_sync();
}

__device__ void gdbm_up(int* D, int n, u32 lane) {
  for (int i = int(lane) + 1; i < n; i += GDBM_G) D[i * n + 0] = GDBM_INF;
  gdbm_sync();
}

// Extra_LU+, in hdbm::dbm_extrapolate_lu_plus's case order; row0 is n ints of scratch.
__device__ void gdbm_extrapolate_lu_plus(int* D, int n, const int* L, const int* U, int* row0, u32 lane) {
  for (int k = int(lane); k < n; k += GDBM_G) row0[k] = D[k];
  gdbm_sync();
  bool dirty = false;
  GDBM_FOR_IJ(i, j, lane, n) {
    if (i == j) continue;
    const int e = D[i * n + j];
    if (e >= GDBM_INF) continue;
    int v = e;
    if (gdbm_bound(e) > L[i]) v = GDBM_INF;
    else if (-gdbm_bound(row0[i]) > L[i]) v = GDBM_INF;
    else if (i != 0 && -gdbm_bound(row0[j]) > U[j]) v = GDBM_INF;
    else if (e < gdbm_lt(-U[j])) v = gdbm_lt(-U[j]);
    if (v != e) { D[i * n + j] = v; dirty = true; }
  }
  gdbm_sync();
  if (gdbm_any(dirty)) gdbm_close(D, n, lane);
}

// Do A and B (canonical) share a valuation? A may be shared or global, B is global.
template <class PA, class PB>
__device__ bool gdbm_intersects(const PA* A, const PB* B, int n, u32 lane) {
  const int nn = n * n;
  gdbm_cursor c = gdbm_begin(lane, n);
  for (int base = 0; base < nn; base += GDBM_G, gdbm_step(c, n)) {
    const bool cut = c.i < n &&
        gdbm_add(gdbm_ld(A, c.i * n + c.j), gdbm_ld(B, c.j * n + c.i)) < GDBM_ZERO_LE;
    if (gdbm_any(cut)) return false;
  }
  return true;
}

// ---- convex union (GDBM section 4.10, after hdbm::dbm_try_union) ---------------------------------
template <class PZ, class PO>
__device__ bool gdbm_union_facets(const PZ* Z, const PO* O, const int* H, int n, u32 lane) {
  const int nn = n * n;
  for (int base = 0; base < nn; base += GDBM_G) {
    const int e = base + int(lane);
    bool f = false;
    if (e < nn) { const int i = e / n; f = (e != i * n + i) && gdbm_ld(Z, e) < H[e]; }
    for (u32 m = __ballot_sync(FULL, f); m; m &= m - 1) {
      const int idx = base + int(__ffs(m) - 1);
      const int i = idx / n, j = idx - i * n;
      const int zij = gdbm_ld(Z, idx);
      const int v = gdbm_bound(zij);
      const int neg = gdbm_strict(zij) ? gdbm_le(-v) : gdbm_lt(-v);
      if (gdbm_add(neg, H[idx]) < GDBM_ZERO_LE) continue;
      bool escape = false;
      for (int pq = int(lane); pq < nn; pq += GDBM_G) {
        const int p = pq / n, q = pq - p * n;
        int r = H[pq];
        const int via = gdbm_add(gdbm_add(H[p * n + j], neg), H[i * n + q]);
        if (via < r) r = via;
        escape = escape || r > gdbm_ld(O, pq);
      }
      if (gdbm_any(escape)) return false;
    }
  }
  return true;
}
// If A u B is convex, A := the hull and true; else A is unchanged and false. H is n*n ints.
template <class PB>
__device__ bool gdbm_try_union(int* A, const PB* B, int* H, int n, u32 lane) {
  bool gap = false;
  for (int i = int(lane) + 1; i < n; i += GDBM_G) {
    const int ai0 = A[i * n], bi0 = gdbm_ld(B, i * n);
    gap = gap || (ai0 < GDBM_INF && gdbm_add(ai0, gdbm_ld(B, i)) < 0) ||
                 (bi0 < GDBM_INF && gdbm_add(bi0, A[i]) < 0);
  }
  if (gdbm_any(gap)) return false;
  GDBM_FOR_E(e, lane, n * n) { const int a = A[e], b = gdbm_ld(B, e); H[e] = a > b ? a : b; }
  gdbm_sync();
  if (!gdbm_union_facets(A, B, H, n, lane) || !gdbm_union_facets(B, A, H, n, lane)) return false;
  GDBM_FOR_E(e, lane, n * n) A[e] = H[e];
  gdbm_sync();
  return true;
}

// Z is simulated by Zp under a_LU, by one lane (hdbm::dbm_is_alu_le). L[0] = U[0] = 0 is required.
template <class PZ, class PZP>
__device__ bool gdbm_alu_le_1(const PZ* Z, const PZP* Zp, int n, const int* L, const int* U) {
  for (int x = 0; x < n; ++x) {
    if (U[x] == GDBM_ALU_NEG_INF) continue;
    const int z0x = gdbm_ld(Z, x);
    if (z0x < gdbm_le(-U[x])) continue;
    for (int y = 0; y < n; ++y) {
      if (y == x || L[y] == GDBM_ALU_NEG_INF) continue;
      const int zpyx = gdbm_ld(Zp, y * n + x);
      if (zpyx < gdbm_ld(Z, y * n + x) && gdbm_add(zpyx, gdbm_lt(-L[y])) < z0x) return false;
    }
  }
  return true;
}

// Z is simulated by Zp under the diagonal-free G-simulation, by one lane (hdbm::dbm_is_g_le): a_LU
// with the guards' strictness kept. Lg / Ug are raw bounds, kGNone where a clock has none, and
// Lg[0] = Ug[0] = (<=, 0).
constexpr int GDBM_GNONE = INT_MIN;
template <class PZ, class PZP>
__device__ bool gdbm_g_le_1(const PZ* Z, const PZP* Zp, int n, const int* Lg, const int* Ug) {
  for (int x = 0; x < n; ++x) {
    if (Ug[x] == GDBM_GNONE) continue;
    const int z0x = gdbm_ld(Z, x);
    if (gdbm_add(z0x, Ug[x]) < GDBM_ZERO_LE) continue;
    for (int y = 0; y < n; ++y) {
      if (y == x || Lg[y] == GDBM_GNONE) continue;
      const int ly = Lg[y];
      const int negL = gdbm_strict(ly) ? gdbm_lt(-gdbm_bound(ly)) : gdbm_le(-gdbm_bound(ly));
      const int zpyx = gdbm_ld(Zp, y * n + x);
      if (zpyx < gdbm_ld(Z, y * n + x) && gdbm_add(zpyx, negL) < z0x) return false;
    }
  }
  return true;
}
// The covering relation of the search: a_LU, or the G-simulation when the model-wide mode says so.
// Lb / Ub then hold raw bounds (see fr_bucket_bounds) instead of integer constants.
template <class PZ, class PZP>
__device__ __forceinline__ bool fr_cover_le(const PZ* Z, const PZP* Zp, int n, const int* L, const int* U) {
  return cD.gmode ? gdbm_g_le_1(Z, Zp, n, L, U) : gdbm_alu_le_1(Z, Zp, n, L, U);
}

// ---- frontier kernels -----------------------------------------------------------------------------
#define FR_TPB 256
#define FR_KERNEL __global__ __launch_bounds__(FR_TPB)

extern __shared__ __align__(16) int tgm[];

__device__ __forceinline__ u32 fr_mix(u32 x) {
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  return x ^ (x >> 16);
}
// Order-independent, so lanes fold shares and combine with a warp sum. Mirrored on the host
// (zone_hash in frontier_cuda.cpp) for the initial zone; the two must agree bit for bit.
__device__ __forceinline__ u32 fr_entry_hash(u32 e, int v) { return fr_mix(e * 0x9e3779b9u ^ u32(v)); }

// Zone z is entry (z & mask) of page z >> page_shift; no zone ever moves.
template <class GZ>
__device__ __forceinline__ GZ* zone_ptr(Pages AP, u32 z) {
  unsigned char* page = AP[z >> cD.page_shift];
  return reinterpret_cast<GZ*>(page) + u64(z & ((1u << cD.page_shift) - 1u)) * u64(cD.nn);
}

__device__ __forceinline__ u32 lower_bound64(const u64* a, u32 n, u64 k) {
  u32 lo = 0, hi = n;
  while (lo < hi) { const u32 m = (lo + hi) >> 1; if (a[m] < k) lo = m + 1; else hi = m; }
  return lo;
}
__device__ __forceinline__ u32 upper_bound64(const u64* a, u32 n, u64 k) {
  u32 lo = 0, hi = n;
  while (lo < hi) { const u32 m = (lo + hi) >> 1; if (a[m] <= k) lo = m + 1; else hi = m; }
  return lo;
}
// Largest b < n with a[b] <= k, for an ascending a with a[0] = 0.
__device__ __forceinline__ u32 segment_of(const u32* a, u32 n, u32 k) {
  u32 lo = 0, hi = n;
  while (hi - lo > 1) { const u32 m = (lo + hi) >> 1; if (a[m] <= k) lo = m; else hi = m; }
  return lo;
}

__device__ bool fr_constrain_all(int* D, const int* c, u32 from, u32 to, u32 lane) {
  const int n = cD.n;
  for (u32 k = from; k < to; ++k)
    if (!gdbm_constrain(D, n, c[3 * k], c[3 * k + 1], c[3 * k + 2], lane)) return false;
  return true;
}

// One warp per work item w = (frontier zone wf[w], transition wr[w]).
template <class GZ>
FR_KERNEL void fr_expand(ExpandP A) {
  const u32 sgs = blockDim.x >> 5, sgid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const u32 w = blockIdx.x * sgs + sgid;
  if (w >= A.nwork) return;
  const int n = cD.n, nn = cD.nn;
  int* D = tgm + sgid * (nn + n);
  const u32 r = A.wr[w];
  const u64 key = A.rkey[r];
  gdbm_load(D, zone_ptr<GZ>(A.AP, A.zref[A.fz[A.wf[w]]]), n, lane);
  bool ok = true;
  for (u32 g = A.rg_off[r]; ok && g < A.rg_off[r + 1]; ++g)
    ok = gdbm_constrain(D, n, A.rg[3 * g], A.rg[3 * g + 1], A.rg[3 * g + 2], lane);
  const u32 v0 = A.rv_off[r], v1 = A.rv_off[r + 1];
  if (ok) {
    for (u32 k = A.rr_off[r]; k < A.rr_off[r + 1]; ++k) gdbm_reset(D, n, A.rr[k], lane);
    ok = fr_constrain_all(D, A.rv, v0, v1, lane);
  }
  if (ok && !(A.rflags[r] & 1u)) {
    gdbm_up(D, n, lane);
    ok = fr_constrain_all(D, A.rv, v0, v1, lane);
  }
  if (ok) {
    for (u64 dead = A.rdead[r]; dead; dead &= dead - 1) gdbm_reset(D, n, int(__ffsll(dead) - 1), lane);
    if (A.extrap) gdbm_extrapolate_lu_plus(D, n, A.Lx, A.Ux, D + nn, lane);
    // The target is tested on every successor BEFORE coverage can prune it, on the exact zone.
    if ((A.rflags[r] & 2u) && (!A.has_tclock || gdbm_intersects(D, A.tdbm, n, lane)))
      if (lane == 0) atomicMin(&A.ctr[1], w);
    // A zone with an entry too wide for the store is never truncated: the batch is flagged
    // (ctr[4]) and the host restarts the search with wider entries.
    bool wide = false;
    GDBM_FOR_E(e, lane, nn) wide = wide || !gdbm_fits(D[e], GZ(0));
    if (gdbm_any(wide)) {
      if (lane == 0) atomicExch(&A.ctr[4], 1u);
      ok = false;
    } else {
      GZ* out = static_cast<GZ*>(A.bz) + u64(w) * nn;
      u32 h = 0;
      GDBM_FOR_E(e, lane, nn) {
        const GZ sv = gdbm_narrow<GZ>(D[e]);
        out[e] = sv;
        h += fr_entry_hash(u32(e), int(sv));
      }
      h = fr_mix(simd_sum(h));
      if (lane == 0) { A.hshw[w] = h; atomicAdd(&A.ctr[0], 1u); }
    }
  }
  if (lane == 0) { A.skey[w] = ok ? key : A.sentinel; A.sval[w] = w; }
}

// ---- exclusive scan --------------------------------------------------------------------------------
// Exclusive scan of one value per thread across the block: prefix sums within each warp, then over
// the warp totals in `sg`.
__device__ u32 tg_scan(u32 v, u32 lane, u32 wid, u32 nw, u32* sg, u32& total) {
  const u32 pre = simd_prefix_exclusive_sum(v);
  const u32 s = simd_sum(v);
  if (lane == 0) sg[wid] = s;
  __syncthreads();
  u32 off = 0, tot = 0;
  for (u32 i = 0; i < nw; ++i) { const u32 x = sg[i]; off += (i < wid) ? x : 0; tot += x; }
  __syncthreads();
  total = tot;
  return pre + off;
}
FR_KERNEL void fr_scan_block(const u32* in, u32* out, u32* sums, u32 n_in, u32 n_out, u32 tile) {
  __shared__ u32 sg[32];
  const u32 tid = threadIdx.x, blk = blockIdx.x, lane = tid & 31, wid = tid >> 5, nw = blockDim.x >> 5;
  const u32 base = blk * tile + tid * 4;
  u32 v[4], s = 0;
  for (u32 q = 0; q < 4; ++q) { const u32 i = base + q; v[q] = (i < n_in) ? in[i] : 0u; s += v[q]; }
  u32 tot;
  u32 run = tg_scan(s, lane, wid, nw, sg, tot);
  for (u32 q = 0; q < 4; ++q) { const u32 i = base + q; if (i < n_out) out[i] = run; run += v[q]; }
  if (tid == 0) sums[blk] = tot;
}
FR_KERNEL void fr_scan_add(u32* out, const u32* sums, u32 n_out, u32 tile) {
  for (u64 i = blockIdx.x * u64(blockDim.x) + threadIdx.x; i < n_out; i += u64(gridDim.x) * blockDim.x)
    out[i] += sums[i / tile];
}

// ---- grouping ---------------------------------------------------------------------------------------
FR_KERNEL void fr_heads(const u64* sk, u32* head, const u32* ctr, u32 W, u32* keep1, u32* keep) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= W) return;
  const bool valid = i < ctr[0];
  head[i] = (valid && (i == 0 || sk[i] != sk[i - 1])) ? 1u : 0u;
  if (!valid) { keep1[i] = 0u; keep[i] = 0u; }
}
FR_KERNEL void fr_buckets(const u64* sk, const u32* head, const u32* hpre, u32* bstart, u64* bkey, u32 W) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= W || !head[i]) return;
  bstart[hpre[i]] = i;
  bkey[hpre[i]] = sk[i];
}
FR_KERNEL void fr_bucket_setup(const u64* bkey, const u32* bstart, const u32* hpre, const u32* ctr,
                               const u64* base_k, const u64* delta_k, uint4* binfo, u32* rowcnt, u32 W,
                               u32 nbase, u32 ndelta) {
  const u32 b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= W) return;
  const u32 nb = hpre[W];
  if (b >= nb) { rowcnt[b] = 0; return; }
  const u64 k = bkey[b];
  const u32 s = bstart[b], e = (b + 1 < nb) ? bstart[b + 1] : ctr[0];
  const u32 bl = lower_bound64(base_k, nbase, k), bu = upper_bound64(base_k, nbase, k);
  const u32 dl = lower_bound64(delta_k, ndelta, k), du = upper_bound64(delta_k, ndelta, k);
  binfo[2 * b] = make_uint4(bl, bu - bl, dl, du - dl);
  binfo[2 * b + 1] = make_uint4(s, e - s, 0, 0);
  rowcnt[b] = (bu - bl) + (du - dl) + (e - s);
}
FR_KERNEL void fr_bucket_bounds(const u64* bkey, const u32* hpre, const int* Lloc, const int* Uloc,
                                const int* Lq, const int* Uq, const int* locbase, const int* kshift,
                                const u64* kmask, int* Lb, int* Ub, u32 W) {
  const int n = cD.n, np = cD.np;
  const u32 nb = hpre[W];
  const u64 total = u64(nb) * n;
  for (u64 gid = blockIdx.x * u64(blockDim.x) + threadIdx.x; gid < total; gid += u64(gridDim.x) * blockDim.x) {
    const u32 b = u32(gid / n), c = u32(gid - u64(b) * n);
    const u64 key = bkey[b];
    int l = Lq[c], u = Uq[c];
    for (int p = 0; p < np; ++p) {
      const int gl = locbase[p] + int((key >> kshift[p]) & kmask[p]);
      l = max(l, Lloc[gl * n + c]);
      u = max(u, Uloc[gl * n + c]);
    }
    if (c == 0) l = u = cD.gmode ? GDBM_ZERO_LE : 0;
    Lb[u64(b) * n + c] = l;
    Ub[u64(b) * n + c] = u;
  }
}

// ---- reduction -------------------------------------------------------------------------------------
__device__ __forceinline__ u32 fr_stored_entry(uint4 st, u32 t, const u32* base_a, const u32* delta_a) {
  return (t < st.y) ? base_a[st.x + t] : delta_a[st.z + t - st.y];
}

// Phase 1: each candidate (sorted position i) against its bucket's live stored zones; the small
// rows here, the large ones listed for fr_reduce_stored_big.
template <class GZ>
FR_KERNEL void fr_reduce_stored(ReduceP P) {
  const uint4* binfo = static_cast<const uint4*>(P.binfo);
  const GZ* bz = static_cast<const GZ*>(P.bz);
  const int n = cD.n, nn = cD.nn;
  const u32 nv = P.ctr[0];
  const u32 stride = gridDim.x * blockDim.x;
  for (u32 gid = blockIdx.x * blockDim.x + threadIdx.x; (gid - (gid & 31u)) < nv; gid += stride) {
    const u32 i = gid;
    const bool mine = i < nv;
    u32 b = 0;
    uint4 st = make_uint4(0, 0, 0, 0);
    if (mine) { b = P.hpre[i] + P.head[i] - 1; st = binfo[2 * b]; }
    const u32 nold = st.y + st.w;
    const bool big = mine && nold > u32(P.solo);
    if (mine && !big) {
      bool drop = false;
      if (nold) {
        const GZ* X = bz + u64(P.sval[i]) * nn;
        const int* L = P.Lb + u64(b) * n;
        const int* U = P.Ub + u64(b) * n;
        for (u32 t = 0; !drop && t < nold; ++t) {
          const u32 a = fr_stored_entry(st, t, P.base_a, P.delta_a);
          drop = P.alive[a] != 0u && fr_cover_le(X, zone_ptr<GZ>(P.AP, P.zref[a]), n, L, U);
        }
      }
      P.keep1[i] = drop ? 0u : 1u;
    }
    if (big) P.big1[atomicAdd(&P.ctr[6], 1u)] = i;
  }
}
// Phase 1, the large rows: one warp per listed candidate.
template <class GZ>
FR_KERNEL void fr_reduce_stored_big(ReduceP P) {
  const uint4* binfo = static_cast<const uint4*>(P.binfo);
  const GZ* bz = static_cast<const GZ*>(P.bz);
  const u32 sgs = blockDim.x >> 5, sgid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = cD.n, nn = cD.nn;
  int* X = tgm + sgid * nn;
  const u32 cnt = P.ctr[6];
  for (u32 k = blockIdx.x * sgs + sgid; k < cnt; k += gridDim.x * sgs) {
    const u32 i = P.big1[k];
    const u32 b = P.hpre[i] + P.head[i] - 1;
    const uint4 st = binfo[2 * b];
    const u32 nold = st.y + st.w;
    const int* L = P.Lb + u64(b) * n;
    const int* U = P.Ub + u64(b) * n;
    gdbm_load(X, bz + u64(P.sval[i]) * nn, n, lane);
    bool drop = false;
    for (u32 base = 0; !drop && base < nold; base += 32) {
      const u32 t = base + lane;
      bool d = false;
      if (t < nold) {
        const u32 a = fr_stored_entry(st, t, P.base_a, P.delta_a);
        d = P.alive[a] != 0u && fr_cover_le(X, zone_ptr<GZ>(P.AP, P.zref[a]), n, L, U);
      }
      drop = simd_any(d);
    }
    if (lane == 0) P.keep1[i] = drop ? 0u : 1u;
    gdbm_sync();   // X is reloaded by the next item
  }
}

FR_KERNEL void fr_survivors(const u32* keep1, const u32* k1pre, u32* surv, const u32* ctr, u32 W) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < W && i < ctr[0] && keep1[i]) surv[k1pre[i]] = i;
}
FR_KERNEL void fr_keep_init(const u32* keep1, u32* keep, const u32* ctr, u32 W) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < W && i < ctr[0] && !keep1[i]) keep[i] = 0u;
}
FR_KERNEL void fr_rows2(ReduceP P) {
  const uint4* binfo = static_cast<const uint4*>(P.binfo);
  const u32 b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= P.W) return;
  if (b >= P.hpre[P.W]) { P.rowcnt[b] = 0; return; }
  const uint4 st = binfo[2 * b], nw = binfo[2 * b + 1];
  const u32 ns = P.k1pre[nw.x + nw.y] - P.k1pre[nw.x];
  P.rowcnt[b] = ns ? st.y + st.w + ns : 0u;
  if (st.y + st.w == 0 && ns) atomicAdd(&P.ctr[2], 1u);
  if (ns && st.y + st.w + ns > 1) P.ulist[atomicAdd(&P.ctr[8], 1u)] = b;
}

// A stored entry dies: it leaves the passed set, and when it was the last entry naming its zone,
// the zone leaves the table (a tombstone keeps probe chains intact) and is queued for reuse.
constexpr u32 kZTomb = 0xfffffffeu;
constexpr u32 kZEmpty = 0xffffffffu;
constexpr u32 kZTag = 0x80000000u;
constexpr u32 kNoSlot = 0xffffffffu;
#define FR_CAS_TRIES 4096
__device__ void fr_kill(const ReduceP& P, u32 a) {
  P.alive[a] = 0u;
  atomicAdd(&P.ctr[3], 1u);
  const u32 z = P.zref[a];
  if (atomicSub(&P.zcnt[z], 1u) == 1u) {
    if (P.ztslot[z] != 0xffffffffu) atomicExch(&P.ztab[P.ztslot[z]], kZTomb);
    P.pend[atomicAdd(&P.ctr[5], 1u)] = z;
  }
}

template <class GZ>
__device__ __forceinline__ bool fr_row_dominated(bool stored, u32 r, const GZ* X, const GZ* Y, u32 t,
                                                 int n, const int* L, const int* U) {
  if (stored) return fr_cover_le(X, Y, n, L, U) && !fr_cover_le(Y, X, n, L, U);
  if (t == r) return false;
  return fr_cover_le(X, Y, n, L, U) && (t < r || !fr_cover_le(Y, X, n, L, U));
}

// Phase 2: rows are a bucket's stored zones, then its survivors; columns its survivors.
template <class GZ>
FR_KERNEL void fr_reduce_pairs(ReduceP P) {
  const uint4* binfo = static_cast<const uint4*>(P.binfo);
  GZ* bz = static_cast<GZ*>(P.bz);
  const int n = cD.n, nn = cD.nn;
  const u32 nrows = P.rowoff[P.W];
  const u32 stride = gridDim.x * blockDim.x;
  for (u32 gid = blockIdx.x * blockDim.x + threadIdx.x; (gid - (gid & 31u)) < nrows; gid += stride) {
    bool mine = gid < nrows, stored = false;
    u32 b = 0, a = 0, pos = 0, r = 0, s0 = 0, ns = 0;
    if (mine) {
      const u32 row = gid;
      b = segment_of(P.rowoff, P.hpre[P.W], row);
      const u32 local = row - P.rowoff[b];
      const uint4 st = binfo[2 * b], nw = binfo[2 * b + 1];
      const u32 nold = st.y + st.w;
      s0 = P.k1pre[nw.x];
      ns = P.k1pre[nw.x + nw.y] - s0;
      stored = local < nold;
      if (stored) {
        a = fr_stored_entry(st, local, P.base_a, P.delta_a);
        if (P.alive[a] == 0u) mine = false;
      } else {
        r = local - nold;
        pos = P.surv[s0 + r];
      }
    }
    const bool big = mine && ns > u32(P.solo);
    if (mine && !big) {
      const int* L = P.Lb + u64(b) * n;
      const int* U = P.Ub + u64(b) * n;
      const GZ* X = stored ? zone_ptr<GZ>(P.AP, P.zref[a]) : bz + u64(P.sval[pos]) * nn;
      bool dom = false;
      for (u32 t = 0; !dom && t < ns; ++t)
        dom = fr_row_dominated<GZ>(stored, r, X, bz + u64(P.sval[P.surv[s0 + t]]) * nn, t, n, L, U);
      if (stored) { if (dom) fr_kill(P, a); }
      else P.keep[pos] = dom ? 0u : 1u;
    }
    if (big) P.big2[atomicAdd(&P.ctr[7], 1u)] = gid;
  }
}
template <class GZ>
FR_KERNEL void fr_reduce_pairs_big(ReduceP P) {
  const uint4* binfo = static_cast<const uint4*>(P.binfo);
  const GZ* bz = static_cast<const GZ*>(P.bz);
  const u32 sgs = blockDim.x >> 5, sgid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = cD.n, nn = cD.nn;
  int* X = tgm + sgid * nn;
  const u32 cnt = P.ctr[7];
  for (u32 k = blockIdx.x * sgs + sgid; k < cnt; k += gridDim.x * sgs) {
    const u32 row = P.big2[k];
    const u32 b = segment_of(P.rowoff, P.hpre[P.W], row);
    const u32 local = row - P.rowoff[b];
    const uint4 st = binfo[2 * b], nw = binfo[2 * b + 1];
    const u32 nold = st.y + st.w;
    const u32 s0 = P.k1pre[nw.x], ns = P.k1pre[nw.x + nw.y] - s0;
    const bool stored = local < nold;
    const u32 a = stored ? fr_stored_entry(st, local, P.base_a, P.delta_a) : 0u;
    const u32 r = stored ? 0u : local - nold;
    const u32 pos = stored ? 0u : P.surv[s0 + r];
    const int* L = P.Lb + u64(b) * n;
    const int* U = P.Ub + u64(b) * n;
    gdbm_load(X, stored ? zone_ptr<GZ>(P.AP, P.zref[a]) : bz + u64(P.sval[pos]) * nn, n, lane);
    bool dom = false;
    for (u32 base = 0; !dom && base < ns; base += 32) {
      const u32 t = base + lane;
      bool d = false;
      if (t < ns) {
        const GZ* Y = bz + u64(P.sval[P.surv[s0 + t]]) * nn;
        if (stored) d = fr_cover_le(X, Y, n, L, U) && !fr_cover_le(Y, X, n, L, U);
        else if (t != r) d = fr_cover_le(X, Y, n, L, U) && (t < r || !fr_cover_le(Y, X, n, L, U));
      }
      dom = simd_any(d);
    }
    if (lane == 0) {
      if (stored) { if (dom) fr_kill(P, a); }
      else P.keep[pos] = dom ? 0u : 1u;
    }
    gdbm_sync();
  }
}

// ---- convex union -----------------------------------------------------------------------------------
// The CPU engine's merge: each new zone, in order, absorbs any zone among the newest live zones of
// its discrete state whose union with it is convex, restarting the scan after every merge.
template <class GZ>
FR_KERNEL void fr_union(ReduceP P) {
  const uint4* binfo = static_cast<const uint4*>(P.binfo);
  GZ* bz = static_cast<GZ*>(P.bz);
  const u32 sgs = blockDim.x >> 5, sgid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = cD.n, nn = cD.nn;
  const u32 scan_lim = u32(cD.merge_scan);
  int* X = tgm + sgid * (2 * nn);
  int* H = X + nn;
  const u32 cnt = *(volatile u32*)&P.ctr[8];
  for (u32 k = blockIdx.x * sgs + sgid; k < cnt; k += gridDim.x * sgs) {
    const u32 b = P.ulist[k];
    const uint4 st = binfo[2 * b], nw = binfo[2 * b + 1];
    const u32 s0 = P.k1pre[nw.x], ns = P.k1pre[nw.x + nw.y] - s0;
    for (u32 r = 0; r < ns; ++r) {
      const u32 pos = P.surv[s0 + r];
      if (simd_broadcast_first(P.keep[pos]) == 0u) continue;
      const u32 w = P.sval[pos];
      gdbm_load(X, bz + u64(w) * nn, n, lane);
      bool changed = false;
      for (bool merged = true; merged;) {
        merged = false;
        u32 seen = 0;
        for (int r2 = int(r) - 1; r2 >= 0 && seen < scan_lim && !merged; --r2) {
          const u32 p2 = P.surv[s0 + u32(r2)];
          if (simd_broadcast_first(P.keep[p2]) == 0u) continue;
          ++seen;
          if (gdbm_try_union(X, bz + u64(P.sval[p2]) * nn, H, n, lane)) {
            merged = changed = true;
            if (lane == 0) P.keep[p2] = 0u;
          }
        }
        for (u32 side = 0; side < 2 && !merged; ++side) {   // the delta, then the base
          const u32 first = side ? st.x : st.z, c = side ? st.y : st.w;
          const u32* run = side ? P.base_a : P.delta_a;
          for (int t = int(c) - 1; t >= 0 && seen < scan_lim && !merged; --t) {
            const u32 a = run[first + u32(t)];
            if (simd_broadcast_first(P.alive[a]) == 0u) continue;
            ++seen;
            if (gdbm_try_union(X, zone_ptr<GZ>(P.AP, P.zref[a]), H, n, lane)) {
              merged = changed = true;
              if (lane == 0) fr_kill(P, a);
            }
          }
        }
        __syncwarp();   // lane 0's flag writes, before rescanning
      }
      if (changed) {   // the hull replaces the candidate: store it, and hash it as stored
        if (lane == 0 && P.chg) P.chg[pos] = 1u;
        GZ* out = bz + u64(w) * nn;
        u32 h = 0;
        GDBM_FOR_E(e, lane, nn) {
          const GZ sv = gdbm_narrow<GZ>(X[e]);
          out[e] = sv;
          h += fr_entry_hash(u32(e), int(sv));
        }
        h = fr_mix(simd_sum(h));
        if (lane == 0) P.hshw[w] = h;
        __syncwarp();   // the stored hull is read back by other lanes
      }
    }
    __syncwarp();
  }
}

// ---- re-establishing the antichain after a merge -------------------------------------------------------
// A union hull is a larger zone than the candidate that was tested against the passed set, so it may
// cover zones the reduction kept: later survivors of its bucket, and stored zones the merge scan
// (bounded by merge_scan) never reached. One thread per bucket, scanning all of it with the
// single-lane covering test; a zone the hull covers is dropped. Pairs equivalent under the covering
// keep the earlier survivor, as the pairwise reduction does.
template <class GZ>
FR_KERNEL void fr_antichain(ReduceP P) {
  const uint4* binfo = static_cast<const uint4*>(P.binfo);
  GZ* bz = static_cast<GZ*>(P.bz);
  const int n = cD.n, nn = cD.nn;
  const u32 cnt = *(volatile u32*)&P.ctr[8];
  for (u32 k = blockIdx.x * blockDim.x + threadIdx.x; k < cnt; k += gridDim.x * blockDim.x) {
    const u32 b = P.ulist[k];
    const uint4 st = binfo[2 * b], nw = binfo[2 * b + 1];
    const u32 s0 = P.k1pre[nw.x], ns = P.k1pre[nw.x + nw.y] - s0;
    const int* L = P.Lb + u64(b) * n;
    const int* U = P.Ub + u64(b) * n;
    for (u32 r = 0; r < ns; ++r) {
      const u32 pos = P.surv[s0 + r];
      if (!P.chg[pos] || P.keep[pos] == 0u) continue;
      const GZ* X = bz + u64(P.sval[pos]) * nn;
      for (u32 r2 = 0; r2 < ns; ++r2) {         // the other new survivors
        if (r2 == r) continue;
        const u32 p2 = P.surv[s0 + r2];
        if (P.keep[p2] == 0u) continue;
        const GZ* Y = bz + u64(P.sval[p2]) * nn;
        if (!fr_cover_le(Y, X, n, L, U)) continue;
        if (r < r2 || !fr_cover_le(X, Y, n, L, U)) P.keep[p2] = 0u;
        else { P.keep[pos] = 0u; break; }       // equivalent to an earlier survivor
      }
      if (P.keep[pos] == 0u) continue;
      for (u32 side = 0; side < 2; ++side) {    // the stored zones: delta, then base
        const u32 first = side ? st.x : st.z, c = side ? st.y : st.w;
        const u32* run = side ? P.base_a : P.delta_a;
        for (u32 t = 0; t < c; ++t) {
          const u32 a = run[first + t];
          if (P.alive[a] == 0u) continue;
          if (fr_cover_le(zone_ptr<GZ>(P.AP, P.zref[a]), X, n, L, U)) fr_kill(P, a);
        }
      }
    }
  }
}

// ---- interning the survivors' zones -------------------------------------------------------------------
template <class GZ>
__device__ bool fr_zone_eq(const GZ* a, const GZ* b, int nn) {
  for (int e = 0; e < nn; ++e) if (a[e] != b[e]) return false;
  return true;
}
__device__ void fr_intern_fail(u32 j, u32* ctr_err, u32* rep, u32* zslot) {
  atomicAdd(ctr_err, 1u);
  rep[j] = kZTag | j;
  zslot[j] = kNoSlot;
}
template <class GZ>
FR_KERNEL void fr_intern(InternP P) {
  const u32 j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= P.W || j >= P.ctr[0] || !P.keep[j]) return;
  const int nn = cD.nn;
  const GZ* bz = static_cast<const GZ*>(P.bz);
  const u32 h = P.hsh[P.sval[j]];
  const GZ* Z = bz + u64(P.sval[j]) * nn;
  u32 s = h & P.mask;
  for (u32 probe = 0; probe <= P.mask; ++probe, s = (s + 1) & P.mask) {
    u32 v = *(volatile u32*)&P.ztab[s];
    for (u32 tries = 0; v == kZEmpty; ++tries) {   // claim it, or learn who did
      if (tries == FR_CAS_TRIES) { fr_intern_fail(j, P.ctr_err, P.rep, P.zslot); return; }
      const u32 old = atomicCAS(&P.ztab[s], kZEmpty, kZTag | j);
      if (old == kZEmpty) {
        P.rep[j] = kZTag | j;
        P.zslot[j] = s;
        return;
      }
      v = old;
    }
    if (v == kZTomb) continue;   // a dead zone's slot: probe on
    const bool mine = (v & kZTag) != 0u;
    const u32 hv = mine ? P.hsh[P.sval[v & ~kZTag]] : P.zhash[v];
    if (hv == h && fr_zone_eq<GZ>(Z, mine ? bz + u64(P.sval[v & ~kZTag]) * nn : zone_ptr<GZ>(P.AP, v), nn)) {
      P.rep[j] = v;
      return;
    }
  }
  fr_intern_fail(j, P.ctr_err, P.rep, P.zslot);
}
FR_KERNEL void fr_zwin(const u32* keep, const u32* ctr, const u32* rep, u32* win, u32 W) {
  const u32 j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= W) return;
  win[j] = (j < ctr[0] && keep[j] && rep[j] == (kZTag | j)) ? 1u : 0u;
}
FR_KERNEL void fr_winlist(const u32* win, const u32* wpre, u32* wlist, u32 W) {
  const u32 j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < W && win[j]) wlist[wpre[j]] = j;
}
template <class GZ>
FR_KERNEL void fr_zplace(ZPlaceP P) {
  const int nn = cD.nn;
  const u64 total = u64(P.wpre[P.W]) * nn;
  const GZ* bz = static_cast<const GZ*>(P.bz);
  for (u64 gid = blockIdx.x * u64(blockDim.x) + threadIdx.x; gid < total; gid += u64(gridDim.x) * blockDim.x) {
    const u32 r = u32(gid / nn), e = u32(gid - u64(r) * nn);
    const u32 j = P.wlist[r];
    const u32 z = (r < P.nfree) ? P.freel[P.nfree - 1 - r] : P.zcount + (r - P.nfree);
    zone_ptr<GZ>(P.AP, z)[e] = bz[u64(P.sval[j]) * nn + e];
    if (e == 0) {
      P.zid[j] = z;
      P.zhash[z] = P.hsh[P.sval[j]];
      P.ztslot[z] = P.zslot[j];
      P.zcnt[z] = 0u;   // place_new counts the entries naming it
      if (P.zslot[j] != kNoSlot) atomicExch(&P.ztab[P.zslot[j]], z);
    }
  }
}

// ---- placing survivors -------------------------------------------------------------------------------
FR_KERNEL void fr_place_new(const u32* keep, const u32* kpre, const u64* sk, const u32* ctr, u64* new_k,
                            u32* new_a, u32* alive, u64* fnext_k, u32* fnext_a, u32 W, u32 acount, u32 fbase,
                            const u32* rep, const u32* zid, u32* zref, u32* zcnt) {
  const u32 j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= W || j >= ctr[0] || !keep[j]) return;
  const u32 r = kpre[j], a = acount + r;
  const u32 v = rep[j];
  const u32 z = (v & kZTag) ? zid[v & ~kZTag] : v;
  zref[a] = z;
  atomicAdd(&zcnt[z], 1u);
  new_k[r] = sk[j];
  new_a[r] = a;
  alive[a] = 1u;
  fnext_k[fbase + r] = sk[j];
  fnext_a[fbase + r] = a;
}

FR_KERNEL void fr_fill_u32(u32* v, u32 n, u32 val) {
  for (u64 i = blockIdx.x * u64(blockDim.x) + threadIdx.x; i < n; i += u64(gridDim.x) * blockDim.x) v[i] = val;
}
FR_KERNEL void fr_zrebuild(const u32* zhash, u32* ztab, u32 W, u32 mask, const u32* zcnt, u32* ztslot,
                           u32* ctr_err) {
  const u32 z = blockIdx.x * blockDim.x + threadIdx.x;
  if (z >= W || zcnt[z] == 0u) return;
  u32 s = zhash[z] & mask;
  for (u32 probe = 0; probe <= mask; ++probe, s = (s + 1) & mask) {
    for (u32 tries = 0; tries < FR_CAS_TRIES; ++tries) {
      const u32 old = atomicCAS(&ztab[s], kZEmpty, z);
      if (old == kZEmpty) { ztslot[z] = s; return; }
      if (old != kZEmpty) break;   // taken: next slot
    }
  }
  ztslot[z] = kNoSlot;
  atomicAdd(ctr_err, 1u);
}

// ---- merging sorted runs --------------------------------------------------------------------------------
FR_KERNEL void fr_merge_a(const u64* ak, const u32* aa, const u64* bk, const u32* nbp, const u32* apre,
                          const u32* bpre, u64* ok, u32* oa, u32 na, u32 nb_cap, u32 filter) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= na) return;
  const u32 nb = nbp[0];
  if (filter && apre[i + 1] == apre[i]) return;
  const u32 lb = lower_bound64(bk, nb, ak[i]);
  const u32 pos = (filter ? apre[i] : i) + (filter ? bpre[lb] : lb);
  ok[pos] = ak[i];
  oa[pos] = aa[i];
}
FR_KERNEL void fr_merge_b(const u64* bk, const u32* ba, const u64* ak, const u32* nbp, const u32* apre,
                          const u32* bpre, u64* ok, u32* oa, u32 na, u32 nb_cap, u32 filter) {
  const u32 j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nb_cap || j >= nbp[0]) return;
  if (filter && bpre[j + 1] == bpre[j]) return;
  const u32 ub = upper_bound64(ak, na, bk[j]);
  const u32 pos = (filter ? apre[ub] : ub) + (filter ? bpre[j] : j);
  ok[pos] = bk[j];
  oa[pos] = ba[j];
}
FR_KERNEL void fr_alive_flags(const u32* idx, const u32* alive, u32* flag, u32 n) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) flag[i] = alive[idx[i]];
}
FR_KERNEL void fr_compact_frontier(const u64* ik, const u32* ia, const u32* flag, const u32* pre, u64* ok,
                                   u32* oa, u32 n) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || !flag[i]) return;
  ok[pre[i]] = ik[i];
  oa[pre[i]] = ia[i];
}
FR_KERNEL void fr_gc_entries(u32* base_a, const u32* zref, u32* zref2, u32* remap, u32* alive2, u32 n) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const u32 e = base_a[i];
  zref2[i] = zref[e];
  remap[e] = i;
  base_a[i] = i;
  alive2[i] = 1u;
}
FR_KERNEL void fr_remap(u32* a, const u32* remap, u32 n) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = remap[a[i]];
}
template <class GZ>
FR_KERNEL void fr_gather_zones(Pages AP, const u32* zref, const u32* idx, u32 n, GZ* out) {
  const u64 total = u64(n) * u64(cD.nn);
  for (u64 t = u64(blockIdx.x) * blockDim.x + threadIdx.x; t < total; t += u64(gridDim.x) * blockDim.x) {
    const u32 i = u32(t / u64(cD.nn)), e = u32(t % u64(cD.nn));
    out[t] = zone_ptr<GZ>(AP, zref[idx[i]])[e];
  }
}
FR_KERNEL void fr_summary(u32* ctr, const u32* kpre, const u32* wpre, const u32* rowoff, const u32* hpre, u32 W) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    ctr[16] = kpre[W]; ctr[17] = wpre[W]; ctr[18] = rowoff[W]; ctr[19] = hpre[W];
  }
}

// ---- launchers --------------------------------------------------------------------------------------
Dims g_dims{};
int g_sms = 108;
int g_smem_optin = 48 * 1024;

inline u32 blocks(u64 n, u32 cap = 1u << 24) {
  const u64 b = (n + FR_TPB - 1) / FR_TPB;
  return u32(b < 1 ? 1 : (b > cap ? cap : b));
}
inline u32 warp_blocks(u64 n, u32 sgs) {
  const u64 b = (n + sgs - 1) / sgs;
  const u64 cap = u64(g_sms) * 16;
  return u32(b < 1 ? 1 : (b > cap ? cap : b));
}
inline u32 stride_blocks(u64 n) {
  const u64 b = (n + FR_TPB - 1) / FR_TPB;
  const u64 cap = u64(g_sms) * 32;
  return u32(b < 1 ? 1 : (b > cap ? cap : b));
}
template <class K>
void allow_smem(K kernel) {
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, g_smem_optin);
}
template <class GZ>
void init_gz() {
  allow_smem(fr_expand<GZ>);
  allow_smem(fr_reduce_stored_big<GZ>);
  allow_smem(fr_reduce_pairs_big<GZ>);
  allow_smem(fr_union<GZ>);
}

#define DISPATCH_GZ(gz, call)                                   \
  do {                                                          \
    if ((gz) == 1) { using GZT = i8; call; } \
    else if ((gz) == 2) { using GZT = i16; call; }              \
    else { using GZT = i32; call; }                             \
  } while (0)

}  // namespace

void set_dims(const Dims& d) {
  g_dims = d;
  cudaMemcpyToSymbol(cD, &d, sizeof(Dims));
}

u32 max_sgs(std::size_t bytes_per_warp) {
  u32 k = 8;
  while (k > 1 && bytes_per_warp * k > std::size_t(g_smem_optin)) --k;
  return k;
}

void init_kernels(std::size_t, std::size_t) {
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&g_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  init_gz<i8>();
  init_gz<i16>();
  init_gz<i32>();
}

void expand(int gz, const ExpandP& p, u32 sgs, cudaStream_t s) {
  const std::size_t smem = std::size_t(sgs) * (g_dims.nn + g_dims.n) * 4;
  const u32 grid = u32((p.nwork + sgs - 1) / sgs);
  if (grid == 0) return;
  DISPATCH_GZ(gz, (fr_expand<GZT><<<grid, sgs * 32, smem, s>>>(p)));
}

void scan_block(const u32* in, u32* out, u32* sums, u32 n_in, u32 n_out, u32 tile, u32 nblk, cudaStream_t s) {
  fr_scan_block<<<nblk, tile / 4, 0, s>>>(in, out, sums, n_in, n_out, tile);
}
void scan_add(u32* out, const u32* sums, u32 n_out, u32 tile, cudaStream_t s) {
  fr_scan_add<<<blocks(n_out), FR_TPB, 0, s>>>(out, sums, n_out, tile);
}

std::size_t sort_pairs(void* tmp, std::size_t tmp_bytes, const u64* kin, u64* kout, const u32* vin, u32* vout,
                       u32 n, int bits, cudaStream_t s) {
  std::size_t need = tmp_bytes;
  cub::DeviceRadixSort::SortPairs(tmp, need, kin, kout, vin, vout, int(n), 0, bits, s);
  return need;
}

void heads(const u64* sk, u32* head, const u32* ctr, u32 W, u32* keep1, u32* keep, cudaStream_t s) {
  fr_heads<<<blocks(W), FR_TPB, 0, s>>>(sk, head, ctr, W, keep1, keep);
}
void buckets(const u64* sk, const u32* head, const u32* hpre, u32* bstart, u64* bkey, u32 W, cudaStream_t s) {
  fr_buckets<<<blocks(W), FR_TPB, 0, s>>>(sk, head, hpre, bstart, bkey, W);
}
void bucket_setup(const u64* bkey, const u32* bstart, const u32* hpre, const u32* ctr, const u64* base_k,
                  const u64* delta_k, void* binfo, u32* rowcnt, u32 W, u32 nbase, u32 ndelta, cudaStream_t s) {
  fr_bucket_setup<<<blocks(W), FR_TPB, 0, s>>>(bkey, bstart, hpre, ctr, base_k, delta_k,
                                               static_cast<uint4*>(binfo), rowcnt, W, nbase, ndelta);
}
void bucket_bounds(const u64* bkey, const u32* hpre, const int* Lloc, const int* Uloc, const int* Lq,
                   const int* Uq, const int* locbase, const int* kshift, const u64* kmask, int* Lb, int* Ub,
                   u32 W, cudaStream_t s) {
  fr_bucket_bounds<<<stride_blocks(u64(W) * g_dims.n), FR_TPB, 0, s>>>(bkey, hpre, Lloc, Uloc, Lq, Uq, locbase,
                                                                        kshift, kmask, Lb, Ub, W);
}

void reduce_stored(int gz, const ReduceP& p, cudaStream_t s) {
  DISPATCH_GZ(gz, (fr_reduce_stored<GZT><<<stride_blocks(p.W), FR_TPB, 0, s>>>(p)));
}
void reduce_stored_big(int gz, const ReduceP& p, u32 sgs, cudaStream_t s) {
  const std::size_t smem = std::size_t(sgs) * g_dims.nn * 4;
  DISPATCH_GZ(gz, (fr_reduce_stored_big<GZT><<<warp_blocks(p.W, sgs), sgs * 32, smem, s>>>(p)));
}
void survivors(const u32* keep1, const u32* k1pre, u32* surv, const u32* ctr, u32 W, cudaStream_t s) {
  fr_survivors<<<blocks(W), FR_TPB, 0, s>>>(keep1, k1pre, surv, ctr, W);
}
void keep_init(const u32* keep1, u32* keep, const u32* ctr, u32 W, cudaStream_t s) {
  fr_keep_init<<<blocks(W), FR_TPB, 0, s>>>(keep1, keep, ctr, W);
}
void rows2(const ReduceP& p, cudaStream_t s) { fr_rows2<<<blocks(p.W), FR_TPB, 0, s>>>(p); }
void reduce_pairs(int gz, const ReduceP& p, cudaStream_t s) {
  DISPATCH_GZ(gz, (fr_reduce_pairs<GZT><<<stride_blocks(u64(p.W) * 8), FR_TPB, 0, s>>>(p)));
}
void reduce_pairs_big(int gz, const ReduceP& p, u32 sgs, cudaStream_t s) {
  const std::size_t smem = std::size_t(sgs) * g_dims.nn * 4;
  DISPATCH_GZ(gz, (fr_reduce_pairs_big<GZT><<<warp_blocks(p.W, sgs), sgs * 32, smem, s>>>(p)));
}
void antichain(int gz, const ReduceP& p, cudaStream_t s) {
  DISPATCH_GZ(gz, (fr_antichain<GZT><<<blocks(1u << 16), FR_TPB, 0, s>>>(p)));
}
void merge_union(int gz, const ReduceP& p, u32 sgs, cudaStream_t s) {
  const std::size_t smem = std::size_t(sgs) * 2 * g_dims.nn * 4;
  DISPATCH_GZ(gz, (fr_union<GZT><<<warp_blocks(p.W, sgs), sgs * 32, smem, s>>>(p)));
}

void intern(int gz, const InternP& p, cudaStream_t s) {
  DISPATCH_GZ(gz, (fr_intern<GZT><<<blocks(p.W), FR_TPB, 0, s>>>(p)));
}
void zwin(const u32* keep, const u32* ctr, const u32* rep, u32* win, u32 W, cudaStream_t s) {
  fr_zwin<<<blocks(W), FR_TPB, 0, s>>>(keep, ctr, rep, win, W);
}
void winlist(const u32* win, const u32* wpre, u32* wlist, u32 W, cudaStream_t s) {
  fr_winlist<<<blocks(W), FR_TPB, 0, s>>>(win, wpre, wlist, W);
}
void zplace(int gz, const ZPlaceP& p, cudaStream_t s) {
  DISPATCH_GZ(gz, (fr_zplace<GZT><<<stride_blocks(u64(p.W) * g_dims.nn), FR_TPB, 0, s>>>(p)));
}
void place_new(const u32* keep, const u32* kpre, const u64* sk, const u32* ctr, u64* new_k, u32* new_a,
               u32* alive, u64* fnext_k, u32* fnext_a, u32 W, u32 acount, u32 fbase, const u32* rep,
               const u32* zid, u32* zref, u32* zcnt, cudaStream_t s) {
  fr_place_new<<<blocks(W), FR_TPB, 0, s>>>(keep, kpre, sk, ctr, new_k, new_a, alive, fnext_k, fnext_a, W,
                                            acount, fbase, rep, zid, zref, zcnt);
}
void fill_u32(u32* v, u32 n, u32 val, cudaStream_t s) { fr_fill_u32<<<stride_blocks(n), FR_TPB, 0, s>>>(v, n, val); }
void zrebuild(const u32* zhash, u32* ztab, u32 W, u32 mask, const u32* zcnt, u32* ztslot, u32* ctr_err,
              cudaStream_t s) {
  if (W == 0) return;
  fr_zrebuild<<<blocks(W), FR_TPB, 0, s>>>(zhash, ztab, W, mask, zcnt, ztslot, ctr_err);
}
void merge_a(const u64* ak, const u32* aa, const u64* bk, const u32* nbp, const u32* apre, const u32* bpre,
             u64* ok, u32* oa, u32 na, u32 nb_cap, u32 filter, cudaStream_t s) {
  if (na) fr_merge_a<<<blocks(na), FR_TPB, 0, s>>>(ak, aa, bk, nbp, apre, bpre, ok, oa, na, nb_cap, filter);
}
void merge_b(const u64* bk, const u32* ba, const u64* ak, const u32* nbp, const u32* apre, const u32* bpre,
             u64* ok, u32* oa, u32 na, u32 nb_cap, u32 filter, cudaStream_t s) {
  if (nb_cap) fr_merge_b<<<blocks(nb_cap), FR_TPB, 0, s>>>(bk, ba, ak, nbp, apre, bpre, ok, oa, na, nb_cap, filter);
}
void alive_flags(const u32* idx, const u32* alive, u32* flag, u32 n, cudaStream_t s) {
  if (n) fr_alive_flags<<<blocks(n), FR_TPB, 0, s>>>(idx, alive, flag, n);
}
void compact_frontier(const u64* ik, const u32* ia, const u32* flag, const u32* pre, u64* ok, u32* oa, u32 n,
                      cudaStream_t s) {
  if (n) fr_compact_frontier<<<blocks(n), FR_TPB, 0, s>>>(ik, ia, flag, pre, ok, oa, n);
}
void gc_entries(u32* base_a, const u32* zref, u32* zref2, u32* remap_, u32* alive2, u32 n, cudaStream_t s) {
  if (n) fr_gc_entries<<<blocks(n), FR_TPB, 0, s>>>(base_a, zref, zref2, remap_, alive2, n);
}
void remap(u32* a, const u32* remap_, u32 n, cudaStream_t s) {
  if (n) fr_remap<<<blocks(n), FR_TPB, 0, s>>>(a, remap_, n);
}
void gather_zones(int gz, Pages AP, const u32* zref, const u32* idx, u32 n, void* out, cudaStream_t s) {
  if (!n) return;
  DISPATCH_GZ(gz, (fr_gather_zones<GZT><<<stride_blocks(u64(n) * 64), FR_TPB, 0, s>>>(
                      AP, zref, idx, n, static_cast<GZT*>(out))));
}
void summary(u32* ctr, const u32* kpre, const u32* wpre, const u32* rowoff, const u32* hpre, u32 W,
             cudaStream_t s) {
  fr_summary<<<1, 32, 0, s>>>(ctr, kpre, wpre, rowoff, hpre, W);
}

}  // namespace cuda
}  // namespace hora
