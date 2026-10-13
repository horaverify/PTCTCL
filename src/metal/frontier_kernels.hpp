/// @file src/metal/frontier_kernels.hpp
/// @brief Metal kernels of the frontier engine: expand, sort, group, reduce, merge.
///
/// Compiled at run time after hdbm::metal::gdbm_source(), whose SIMD-group-cooperative zone
/// operations they call, and specialised per model through preprocessor macros:
///
///   PT_DIM   zone dimension (clocks + 1)        PT_NP    processes
///   GZ       stored-zone entry type (short/int) FR_SGS   SIMD-groups per threadgroup
///
/// One level of the search, for one batch of the frontier (see frontier_metal.mm):
///
///   fr_expand        one SIMD-group per (frontier zone, transition): the successor zone, the
///                    target test, and a sort key -- the successor's discrete key, or a sentinel
///                    above every key when the transition is disabled
///   fr_rs_*          stable LSD radix sort of (key, work item) on the GPU
///   fr_heads ..      group the sorted candidates by discrete key into buckets, and find each
///   fr_bucket_setup  bucket's stored zones in the passed set (a sorted base plus a sorted delta)
///   fr_reduce        all-pairs coverage within each bucket, one SIMD-group per row
///   fr_zhash ..      intern the survivors' zones (identical zones are stored once), give
///   fr_place_new     each survivor a passed-set entry, and merge the entries into the delta
///
/// Coverage is a_LU subsumption with the bucket's own bounds. The candidates of a bucket and
/// its stored zones are compared all-pairs; a zone is dropped iff another covers it and is not
/// covered back, or covers it equivalently and ranks earlier (stored before new, then by work
/// item). That relation is a strict order, so the survivors are exactly its maximal elements:
/// an antichain covering every candidate, whatever order the comparisons run in.
#pragma once

namespace hora {
namespace metal {

inline const char* frontier_kernels_source() {
  return R"FR_MSL(
#ifndef PT_DIM
#define PT_DIM 8
#endif
#ifndef PT_NP
#define PT_NP 4
#endif
#ifndef GZ
#define GZ short
#endif
#ifndef FR_SGS
#define FR_SGS 4
#endif
#ifndef FR_LANE_CMP
#define FR_LANE_CMP 1          // lane-per-pair comparisons in fr_reduce (small zones)
#endif

#define kN    (PT_DIM)
#define kNN   (PT_DIM * PT_DIM)
#define kNP   (PT_NP)
#define kTile 1024u                             // radix-sort and scan tile: 256 threads x 4

// ---- small helpers --------------------------------------------------------------------------
inline uint lower_bound64(device const ulong* a, uint n, ulong k) {
  uint lo = 0, hi = n;
  while (lo < hi) { const uint m = (lo + hi) >> 1; if (a[m] < k) lo = m + 1; else hi = m; }
  return lo;
}
inline uint upper_bound64(device const ulong* a, uint n, ulong k) {
  uint lo = 0, hi = n;
  while (lo < hi) { const uint m = (lo + hi) >> 1; if (a[m] <= k) lo = m + 1; else hi = m; }
  return lo;
}
// Largest b < n with a[b] <= k, for an ascending a with a[0] = 0.
inline uint segment_of(device const uint* a, uint n, uint k) {
  uint lo = 0, hi = n;
  while (hi - lo > 1) { const uint m = (lo + hi) >> 1; if (a[m] <= k) lo = m; else hi = m; }
  return lo;
}
inline int loc_of(ulong key, device const int* kshift, device const ulong* kmask, int p) {
  return int((key >> kshift[p]) & kmask[p]);
}

// Exclusive scan of one value per thread across a 256-thread threadgroup.
inline uint tg_scan256(uint v, uint lane, uint sgid, threadgroup uint* sg, thread uint& total) {
  const uint pre = simd_prefix_exclusive_sum(v);
  const uint s = simd_sum(v);
  if (lane == 0) sg[sgid] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint off = 0, tot = 0;
  for (uint i = 0; i < 8; ++i) { const uint x = sg[i]; off += (i < sgid) ? x : 0; tot += x; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total = tot;
  return pre + off;
}

// ---- expand ---------------------------------------------------------------------------------
struct ExpandArgs {
  uint  nwork;
  int   has_tclock;   // the target constrains clocks: test tdbm on every target-matching successor
  ulong qmask;        // clocks the query reads; never reset as inactive
  ulong clockbits;    // bits 1..nclocks
  ulong sentinel;     // sort key of a disabled transition: above every discrete key
  int   extrap;       // apply Extra_LU+ (global bounds Lx, Ux) to every successor
  int   pad;
};

// Conjoin the invariants of every process's location in `key`.
inline bool fr_invariants(threadgroup int* D, ulong key, device const int* kshift,
                          device const ulong* kmask, device const int* locbase,
                          device const int* inv_off, device const int* inv, uint lane) {
  for (int p = 0; p < kNP; ++p) {
    const int gl = locbase[p] + loc_of(key, kshift, kmask, p);
    for (int c = inv_off[gl]; c < inv_off[gl + 1]; ++c)
      if (!gdbm_constrain(D, kN, inv[3 * c], inv[3 * c + 1], inv[3 * c + 2], lane)) return false;
  }
  return true;
}

// One SIMD-group per work item w = (frontier zone wf[w], transition wr[w]). Mirrors the CPU
// successor pipeline exactly: every guard, every reset, the target invariant, delay unless the
// target is committed or urgent (decided on the host, flag bit 0), the invariant again, and the
// active-clock reduction. Writes the successor zone to its own slot, so the output order is the
// work order and the search is deterministic.
kernel void fr_expand(device const uint*  wf      [[buffer(0)]],
                      device const uint*  wr      [[buffer(1)]],
                      device const uint*  fz      [[buffer(2)]],
                      device const GZ*    arena   [[buffer(3)]],
                      device const ulong* rkey    [[buffer(4)]],
                      device const uint*  rflags  [[buffer(5)]],
                      device const uint*  rg_off  [[buffer(6)]],
                      device const int*   rg      [[buffer(7)]],
                      device const uint*  rr_off  [[buffer(8)]],
                      device const int*   rr      [[buffer(9)]],
                      device const int*   locbase [[buffer(10)]],
                      device const int*   inv_off [[buffer(11)]],
                      device const int*   inv     [[buffer(12)]],
                      device const ulong* actm    [[buffer(13)]],
                      device const int*   kshift  [[buffer(14)]],
                      device const ulong* kmask   [[buffer(15)]],
                      device const int*   tdbm    [[buffer(16)]],
                      device GZ*          bz      [[buffer(17)]],
                      device ulong*       skey    [[buffer(18)]],
                      device uint*        sval    [[buffer(19)]],
                      device atomic_uint* ctr     [[buffer(20)]],
                      constant ExpandArgs& A      [[buffer(21)]],
                      device const int*   Lx      [[buffer(22)]],
                      device const int*   Ux      [[buffer(23)]],
                      device const uint*  zref    [[buffer(24)]],
                      threadgroup int*    tgm     [[threadgroup(0)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint sgid [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  const uint w = tg * FR_SGS + sgid;
  if (w >= A.nwork) return;
  threadgroup int* D = tgm + sgid * (kNN + kN);   // the zone, then a row of scratch
  const uint r = wr[w];
  const ulong key = rkey[r];
  gdbm_load(D, arena + ulong(zref[fz[wf[w]]]) * kNN, kN, lane);
  bool ok = true;
  for (uint g = rg_off[r]; ok && g < rg_off[r + 1]; ++g)
    ok = gdbm_constrain(D, kN, rg[3 * g], rg[3 * g + 1], rg[3 * g + 2], lane);
  if (ok) {
    for (uint k = rr_off[r]; k < rr_off[r + 1]; ++k) gdbm_reset(D, kN, rr[k], lane);
    ok = fr_invariants(D, key, kshift, kmask, locbase, inv_off, inv, lane);
  }
  if (ok && !(rflags[r] & 1u)) {
    gdbm_up(D, kN, lane);
    ok = fr_invariants(D, key, kshift, kmask, locbase, inv_off, inv, lane);
  }
  if (ok) {
    ulong live = A.qmask;
    for (int p = 0; p < kNP; ++p) live |= actm[locbase[p] + loc_of(key, kshift, kmask, p)];
    ulong dead = ~live & A.clockbits;
    while (dead) { const int c = int(ctz(dead)); dead &= dead - 1; gdbm_reset(D, kN, c, lane); }
    if (A.extrap) gdbm_extrapolate_lu_plus(D, kN, Lx, Ux, D + kNN, lane);
    // The target is tested on every successor BEFORE coverage can prune it, and on the exact zone,
    // before the store's entry width can reject it.
    if ((rflags[r] & 2u) && (!A.has_tclock || gdbm_intersects(D, tdbm, kN, lane)))
      if (lane == 0) atomic_fetch_min_explicit(&ctr[1], w, memory_order_relaxed);
    // A zone with an entry too wide for the store is never truncated: the batch is flagged
    // (ctr[4]) and the host restarts the search with wider entries.
    bool wide = false;
    GDBM_FOR_E(e, lane, kNN) wide = wide || !gdbm_fits(D[e], GZ(0));
    if (gdbm_any(wide)) {
      if (lane == 0) atomic_store_explicit(&ctr[4], 1u, memory_order_relaxed);
      ok = false;
    } else {
      gdbm_store(bz + ulong(w) * kNN, D, kN, lane);
      if (lane == 0) atomic_fetch_add_explicit(&ctr[0], 1u, memory_order_relaxed);
    }
  }
  if (lane == 0) { skey[w] = ok ? key : A.sentinel; sval[w] = w; }
}

// ---- exclusive scan -------------------------------------------------------------------------
struct ScanArgs { uint n_in, n_out; };
// out[i] = sum of in[0..i) within this 1024-element block; sums[block] = block total. Values at
// i >= n_in read as 0, so a scan of n values over n + 1 outputs leaves the total at out[n].
kernel void fr_scan_block(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
                          device uint* sums [[buffer(2)]], constant ScanArgs& A [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint blk [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sgid [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint sg[8];
  const uint base = blk * kTile + tid * 4;
  uint v[4], s = 0;
  for (uint q = 0; q < 4; ++q) { const uint i = base + q; v[q] = (i < A.n_in) ? in[i] : 0u; s += v[q]; }
  uint tot;
  uint run = tg_scan256(s, lane, sgid, sg, tot);
  for (uint q = 0; q < 4; ++q) { const uint i = base + q; if (i < A.n_out) out[i] = run; run += v[q]; }
  if (tid == 0) sums[blk] = tot;
}
kernel void fr_scan_add(device uint* out [[buffer(0)]], device const uint* sums [[buffer(1)]],
                        constant ScanArgs& A [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < A.n_out) out[i] += sums[i / kTile];
}

// ---- radix sort: stable LSD, 8 bits a pass ----------------------------------------------------
struct RsArgs { uint n, ntiles, shift; };
kernel void fr_rs_hist(device const ulong* kin [[buffer(0)]], device uint* ghist [[buffer(1)]],
                       constant RsArgs& A [[buffer(2)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint tile [[threadgroup_position_in_grid]]) {
  threadgroup atomic_uint h[256];
  atomic_store_explicit(&h[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint q = 0; q < 4; ++q) {
    const uint i = tile * kTile + q * 256 + tid;
    if (i < A.n) atomic_fetch_add_explicit(&h[uint((kin[i] >> A.shift) & 255ul)], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ghist[tid * A.ntiles + tile] = atomic_load_explicit(&h[tid], memory_order_relaxed);
}
// Sort the tile locally by the digit (eight stable 1-bit splits), then write each element at
// its digit's global offset for this tile plus its rank among the tile's elements of that digit.
kernel void fr_rs_scatter(device const ulong* kin [[buffer(0)]], device const uint* vin [[buffer(1)]],
                          device ulong* kout [[buffer(2)]], device uint* vout [[buffer(3)]],
                          device const uint* gpre [[buffer(4)]], constant RsArgs& A [[buffer(5)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint tile [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sgid [[simdgroup_index_in_threadgroup]]) {
  threadgroup ulong k0[kTile], k1[kTile];
  threadgroup uint v0[kTile], v1[kTile];
  threadgroup uint sg[8];
  threadgroup uint dstart[256];
  threadgroup atomic_uint h[256];
  const uint base = tile * kTile;
  const uint cnt = min(kTile, A.n - base);
  for (uint q = 0; q < 4; ++q) {               // coalesced load; padding sorts last (digit 255)
    const uint i = q * 256 + tid;
    k0[i] = (i < cnt) ? kin[base + i] : ~0ul;
    v0[i] = (i < cnt) ? vin[base + i] : 0u;
  }
  atomic_store_explicit(&h[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup ulong* ks = k0; threadgroup ulong* kd = k1;
  threadgroup uint* vs = v0; threadgroup uint* vd = v1;
  for (uint b = 0; b < 8; ++b) {
    ulong kk[4]; uint vv[4]; uint bit[4]; uint zeros = 0;
    for (uint q = 0; q < 4; ++q) {
      const uint i = tid * 4 + q;
      kk[q] = ks[i]; vv[q] = vs[i];
      bit[q] = uint((kk[q] >> (A.shift + b)) & 1ul);
      zeros += 1u - bit[q];
    }
    uint Z;
    uint zb = tg_scan256(zeros, lane, sgid, sg, Z);    // zeros before this thread's items
    for (uint q = 0; q < 4; ++q) {
      const uint i = tid * 4 + q;
      const uint dst = bit[q] ? (Z + (i - zb)) : zb;
      zb += 1u - bit[q];
      kd[dst] = kk[q]; vd[dst] = vv[q];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup ulong* tk = ks; ks = kd; kd = tk;
    threadgroup uint* tv = vs; vs = vd; vd = tv;
  }
  for (uint q = 0; q < 4; ++q) {
    const uint i = tid * 4 + q;
    if (i < cnt) atomic_fetch_add_explicit(&h[uint((ks[i] >> A.shift) & 255ul)], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint tot;
  dstart[tid] = tg_scan256(atomic_load_explicit(&h[tid], memory_order_relaxed), lane, sgid, sg, tot);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint q = 0; q < 4; ++q) {
    const uint i = q * 256 + tid;
    if (i >= cnt) continue;
    const uint d = uint((ks[i] >> A.shift) & 255ul);
    const uint pos = gpre[d * A.ntiles + tile] + (i - dstart[d]);
    kout[pos] = ks[i];
    vout[pos] = vs[i];
  }
}

// ---- grouping ---------------------------------------------------------------------------------
struct GroupArgs { uint W; uint nbase; uint ndelta; };
// Also zeroes the keep flags past the valid candidates, so the reduction's prefix sums can run
// over the whole batch without the host knowing how many candidates are valid.
kernel void fr_heads(device const ulong* sk [[buffer(0)]], device uint* head [[buffer(1)]],
                     device const uint* ctr [[buffer(2)]], constant GroupArgs& A [[buffer(3)]],
                     device uint* keep1 [[buffer(4)]], device uint* keep [[buffer(5)]],
                     uint i [[thread_position_in_grid]]) {
  if (i >= A.W) return;
  const bool valid = i < ctr[0];
  head[i] = (valid && (i == 0 || sk[i] != sk[i - 1])) ? 1u : 0u;
  if (!valid) { keep1[i] = 0u; keep[i] = 0u; }
}

// Indirect dispatch arguments: threadgroups covering src[index] * mul items at per_tg a group.
struct IndirectArgs { uint index, mul, per_tg, slot; };
kernel void fr_indirect(device const uint* src [[buffer(0)]], device uint* args [[buffer(1)]],
                        constant IndirectArgs& A [[buffer(2)]]) {
  const uint n = src[A.index] * A.mul;
  args[3 * A.slot + 0] = max(1u, (n + A.per_tg - 1) / A.per_tg);
  args[3 * A.slot + 1] = 1u;
  args[3 * A.slot + 2] = 1u;
}
kernel void fr_buckets(device const ulong* sk [[buffer(0)]], device const uint* head [[buffer(1)]],
                       device const uint* hpre [[buffer(2)]], device uint* bstart [[buffer(3)]],
                       device ulong* bkey [[buffer(4)]], constant GroupArgs& A [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
  if (i >= A.W || !head[i]) return;
  bstart[hpre[i]] = i;
  bkey[hpre[i]] = sk[i];
}
// Per bucket: its stored zones (a range of the base and a range of the delta, both sorted by
// key) and its row count, stored plus new.
kernel void fr_bucket_setup(device const ulong* bkey [[buffer(0)]], device const uint* bstart [[buffer(1)]],
                            device const uint* hpre [[buffer(2)]], device const uint* ctr [[buffer(3)]],
                            device const ulong* base_k [[buffer(4)]], device const ulong* delta_k [[buffer(5)]],
                            device uint4* binfo [[buffer(6)]], device uint* rowcnt [[buffer(7)]],
                            constant GroupArgs& A [[buffer(8)]], uint b [[thread_position_in_grid]]) {
  if (b >= A.W) return;
  const uint nb = hpre[A.W];
  if (b >= nb) { rowcnt[b] = 0; return; }
  const ulong k = bkey[b];
  const uint s = bstart[b], e = (b + 1 < nb) ? bstart[b + 1] : ctr[0];
  const uint bl = lower_bound64(base_k, A.nbase, k), bu = upper_bound64(base_k, A.nbase, k);
  const uint dl = lower_bound64(delta_k, A.ndelta, k), du = upper_bound64(delta_k, A.ndelta, k);
  binfo[2 * b]     = uint4(bl, bu - bl, dl, du - dl);
  binfo[2 * b + 1] = uint4(s, e - s, 0, 0);
  rowcnt[b] = (bu - bl) + (du - dl) + (e - s);
}

// ---- all-pairs reduction, in two phases -----------------------------------------------------------
// The survivors of a bucket are the maximal elements of (stored zones + new candidates) under
// a_LU coverage, ties won by stored zones and then by the earlier candidate. Computed as:
//
//   phase 1  each new candidate against the bucket's stored zones: covered -> dropped
//   phase 2  all pairs among the phase-1 survivors, and the stored zones against them
//
// which is the same result with far fewer comparisons, because late in a search most candidates
// are covered by a stored zone. Phase 2 may ignore what phase 1 dropped: a candidate covered by a
// stored zone w covers nothing that w does not, and cannot strictly cover a stored zone (stored
// zones never cover each other). Each row stops at its first dominator.
//
// Rows of one bucket run concurrently, and a stored row may be marked dead while another row
// reads it. That cannot change any answer: a stored zone dies only to a new zone that strictly
// covers it, and that zone then covers whatever the dead one covered.

// The a_LU bounds of the bucket whose key is `key`: per-location maxima over processes, floored
// by the query's constants, (0, 0) for the reference clock.
inline void fr_bounds(ulong key, threadgroup int* L, threadgroup int* U, device const int* Lloc,
                      device const int* Uloc, device const int* Lq, device const int* Uq,
                      device const int* locbase, device const int* kshift, device const ulong* kmask,
                      uint lane) {
  for (int c = int(lane); c < kN; c += 32) {
    int l = Lq[c], u = Uq[c];
    for (int p = 0; p < kNP; ++p) {
      const int gl = locbase[p] + loc_of(key, kshift, kmask, p);
      l = max(l, Lloc[gl * kN + c]);
      u = max(u, Uloc[gl * kN + c]);
    }
    L[c] = (c == 0) ? 0 : l;
    U[c] = (c == 0) ? 0 : u;
  }
}

struct ReduceArgs { uint W; };

// Phase 1: one SIMD-group per new candidate (sorted position i).
kernel void fr_reduce_stored(device const uint*  hpre    [[buffer(0)]],
                             device const uint*  head    [[buffer(1)]],
                             device const uint4* binfo   [[buffer(2)]],
                             device const ulong* bkey    [[buffer(3)]],
                             device const uint*  base_a  [[buffer(4)]],
                             device const uint*  delta_a [[buffer(5)]],
                             device const GZ*    arena   [[buffer(6)]],
                             device const uint*  alive   [[buffer(7)]],
                             device const uint*  sval    [[buffer(8)]],
                             device const GZ*    bz      [[buffer(9)]],
                             device const int*   Lloc    [[buffer(10)]],
                             device const int*   Uloc    [[buffer(11)]],
                             device const int*   Lq      [[buffer(12)]],
                             device const int*   Uq      [[buffer(13)]],
                             device const int*   locbase [[buffer(14)]],
                             device const int*   kshift  [[buffer(15)]],
                             device const ulong* kmask   [[buffer(16)]],
                             device uint*        keep1   [[buffer(17)]],
                             device const uint*  ctr     [[buffer(18)]],
                             device const uint*  zref    [[buffer(19)]],
                             threadgroup int*    tgm     [[threadgroup(0)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint sgid [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const uint i = tg * FR_SGS + sgid;
  if (i >= ctr[0]) return;
  const uint b = hpre[i] + head[i] - 1;
  const uint4 st = binfo[2 * b];
  const uint nold = st.y + st.w;
  if (nold == 0) { if (lane == 0) keep1[i] = 1u; return; }
  threadgroup int* X = tgm + sgid * (kNN + 2 * kN);
  threadgroup int* L = X + kNN;
  threadgroup int* U = L + kN;
  fr_bounds(bkey[b], L, U, Lloc, Uloc, Lq, Uq, locbase, kshift, kmask, lane);
  gdbm_load(X, bz + ulong(sval[i]) * kNN, kN, lane);
  bool drop = false;
#if FR_LANE_CMP
  for (uint base = 0; !drop && base < nold; base += 32) {
    const uint t = base + lane;
    bool d = false;
    if (t < nold) {
      const uint a = (t < st.y) ? base_a[st.x + t] : delta_a[st.z + t - st.y];
      d = alive[a] != 0u && gdbm_alu_le_1(X, arena + ulong(zref[a]) * kNN, kN, L, U);
    }
    drop = simd_any(d);
  }
#else
  for (uint t = 0; !drop && t < nold; ++t) {
    const uint a = (t < st.y) ? base_a[st.x + t] : delta_a[st.z + t - st.y];
    if (simd_broadcast_first(alive[a]) != 0u) drop = gdbm_alu_le(X, arena + ulong(zref[a]) * kNN, kN, L, U, lane);
  }
#endif
  if (lane == 0) keep1[i] = drop ? 0u : 1u;
}

// Between the phases: the phase-1 survivors of each bucket, compacted in order, and the phase-2
// row count of each bucket (its stored zones plus its survivors).
kernel void fr_survivors(device const uint* keep1 [[buffer(0)]], device const uint* k1pre [[buffer(1)]],
                         device uint* surv [[buffer(2)]], device const uint* ctr [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
  if (i < ctr[0] && keep1[i]) surv[k1pre[i]] = i;
}
kernel void fr_rows2(device const uint4* binfo [[buffer(0)]], device const uint* hpre [[buffer(1)]],
                     device const uint* k1pre [[buffer(2)]], device uint* rowcnt [[buffer(3)]],
                     constant ReduceArgs& A [[buffer(4)]], device atomic_uint* ctr [[buffer(5)]],
                     uint b [[thread_position_in_grid]]) {
  if (b >= A.W) return;
  if (b >= hpre[A.W]) { rowcnt[b] = 0; return; }
  const uint4 st = binfo[2 * b], nw = binfo[2 * b + 1];
  const uint ns = k1pre[nw.x + nw.y] - k1pre[nw.x];
  rowcnt[b] = ns ? st.y + st.w + ns : 0u;   // no survivor: nothing new, and no stored zone can die
  if (st.y + st.w == 0 && ns) atomic_fetch_add_explicit(&ctr[2], 1u, memory_order_relaxed);
}

// Phase 2: one SIMD-group per row; rows past the device-side total exit at once.
kernel void fr_reduce_pairs(device const uint*  rowoff  [[buffer(0)]],
                            device const uint*  hpre    [[buffer(1)]],
                            device const uint4* binfo   [[buffer(2)]],
                            device const ulong* bkey    [[buffer(3)]],
                            device const uint*  base_a  [[buffer(4)]],
                            device const uint*  delta_a [[buffer(5)]],
                            device const GZ*    arena   [[buffer(6)]],
                            device uint*        alive   [[buffer(7)]],
                            device const uint*  sval    [[buffer(8)]],
                            device const GZ*    bz      [[buffer(9)]],
                            device const int*   Lloc    [[buffer(10)]],
                            device const int*   Uloc    [[buffer(11)]],
                            device const int*   Lq      [[buffer(12)]],
                            device const int*   Uq      [[buffer(13)]],
                            device const int*   locbase [[buffer(14)]],
                            device const int*   kshift  [[buffer(15)]],
                            device const ulong* kmask   [[buffer(16)]],
                            device uint*        keep    [[buffer(17)]],
                            device const uint*  k1pre   [[buffer(18)]],
                            device const uint*  surv    [[buffer(19)]],
                            constant ReduceArgs& A      [[buffer(20)]],
                            device atomic_uint* ctr     [[buffer(21)]],
                            device const uint*  zref    [[buffer(22)]],
                            threadgroup int*    tgm     [[threadgroup(0)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint sgid [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg * FR_SGS + sgid;
  if (row >= rowoff[A.W]) return;
  const uint b = segment_of(rowoff, hpre[A.W], row);
  const uint local = row - rowoff[b];
  const uint4 st = binfo[2 * b], nw = binfo[2 * b + 1];
  const uint nold = st.y + st.w;
  const uint s0 = k1pre[nw.x], ns = k1pre[nw.x + nw.y] - s0;   // this bucket's survivors
  threadgroup int* X = tgm + sgid * (kNN + 2 * kN);
  threadgroup int* L = X + kNN;
  threadgroup int* U = L + kN;
  fr_bounds(bkey[b], L, U, Lloc, Uloc, Lq, Uq, locbase, kshift, kmask, lane);

  if (local < nold) {                                   // a stored zone: dies to a strict cover
    const uint a = (local < st.y) ? base_a[st.x + local] : delta_a[st.z + local - st.y];
    if (simd_broadcast_first(alive[a]) == 0u) return;
    gdbm_load(X, arena + ulong(zref[a]) * kNN, kN, lane);
#if FR_LANE_CMP
    for (uint base = 0; base < ns; base += 32) {
      const uint t = base + lane;
      bool kill = false;
      if (t < ns) {
        device const GZ* Y = bz + ulong(sval[surv[s0 + t]]) * kNN;
        kill = gdbm_alu_le_1(X, Y, kN, L, U) && !gdbm_alu_le_1(Y, X, kN, L, U);
      }
      if (simd_any(kill)) { if (lane == 0) { alive[a] = 0u; atomic_fetch_add_explicit(&ctr[3], 1u, memory_order_relaxed); } return; }
    }
#else
    for (uint t = 0; t < ns; ++t) {
      device const GZ* Y = bz + ulong(sval[surv[s0 + t]]) * kNN;
      if (gdbm_alu_le(X, Y, kN, L, U, lane) && !gdbm_alu_le(Y, X, kN, L, U, lane)) {
        if (lane == 0) { alive[a] = 0u; atomic_fetch_add_explicit(&ctr[3], 1u, memory_order_relaxed); }
        return;
      }
    }
#endif
    return;
  }
  const uint r = local - nold;                          // survivor r of this bucket
  const uint pos = surv[s0 + r];
  gdbm_load(X, bz + ulong(sval[pos]) * kNN, kN, lane);
  bool drop = false;
#if FR_LANE_CMP
  for (uint base = 0; !drop && base < ns; base += 32) {
    const uint t = base + lane;
    bool d = false;
    if (t < ns && t != r) {
      device const GZ* Y = bz + ulong(sval[surv[s0 + t]]) * kNN;
      d = gdbm_alu_le_1(X, Y, kN, L, U) && (t < r || !gdbm_alu_le_1(Y, X, kN, L, U));
    }
    drop = simd_any(d);
  }
#else
  for (uint t = 0; !drop && t < ns; ++t) {
    if (t == r) continue;
    device const GZ* Y = bz + ulong(sval[surv[s0 + t]]) * kNN;
    if (gdbm_alu_le(X, Y, kN, L, U, lane)) drop = (t < r) || !gdbm_alu_le(Y, X, kN, L, U, lane);
  }
#endif
  if (lane == 0) keep[pos] = drop ? 0u : 1u;
}

// keep := 0 for the candidates phase 1 dropped (phase 2 writes the survivors' own entries).
kernel void fr_keep_init(device const uint* keep1 [[buffer(0)]], device uint* keep [[buffer(1)]],
                         device const uint* ctr [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < ctr[0] && !keep1[i]) keep[i] = 0u;
}

// ---- interning the survivors' zones ------------------------------------------------------------
// GDBM's DBM table (section 4.12): a zone reached in several discrete states is stored once. The
// passed set holds *entries* -- (discrete key, entry id) -- and zref[entry] names the entry's
// zone in the arena. The table maps a zone's hash to its arena slot, open addressing with linear
// probing over 32-bit slots, so a lock-free compare-and-swap claims a slot.
//
// A batch interns in three steps: hash every survivor; claim or find its zone in the table, a
// slot claimed for a zone of this very batch holding kZTag | candidate until it has an arena slot;
// give each zone that claimed a slot its arena slot, copy it, and resolve the tag.
constant uint kZEmpty = 0xffffffffu;
constant uint kZTag = 0x80000000u;

inline uint fr_mix(uint x) {
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  return x ^ (x >> 16);
}
// Order-independent, so lanes fold shares and combine with a SIMD sum. Mirrored on the host
// (zone_hash in frontier_metal.mm) for the initial zone; the two must agree bit for bit.
inline uint fr_entry_hash(uint e, int v) { return fr_mix(e * 0x9e3779b9u ^ uint(v)); }

kernel void fr_zhash(device const uint* keep [[buffer(0)]], device const uint* ctr [[buffer(1)]],
                     device const uint* sval [[buffer(2)]], device const GZ* bz [[buffer(3)]],
                     device uint* hsh [[buffer(4)]], constant uint& W [[buffer(5)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint sgid [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  const uint j = tg * FR_SGS + sgid;
  if (j >= W || j >= ctr[0] || !keep[j]) return;
  device const GZ* Z = bz + ulong(sval[j]) * kNN;
  uint h = 0;
  for (int e = int(lane); e < kNN; e += 32) h += fr_entry_hash(uint(e), int(Z[e]));
  h = fr_mix(simd_sum(h));
  if (lane == 0) hsh[j] = h;
}

inline bool fr_zone_eq(device const GZ* a, device const GZ* b) {
  for (int e = 0; e < kNN; ++e) if (a[e] != b[e]) return false;
  return true;
}
struct InternArgs { uint W; uint mask; };
kernel void fr_intern(device const uint* keep [[buffer(0)]], device const uint* ctr [[buffer(1)]],
                      device const uint* sval [[buffer(2)]], device const GZ* bz [[buffer(3)]],
                      device const GZ* arena [[buffer(4)]], device const uint* hsh [[buffer(5)]],
                      device const uint* zhash [[buffer(6)]], device atomic_uint* ztab [[buffer(7)]],
                      device uint* rep [[buffer(8)]], device uint* zslot [[buffer(9)]],
                      constant InternArgs& A [[buffer(10)]], uint j [[thread_position_in_grid]]) {
  if (j >= A.W || j >= ctr[0] || !keep[j]) return;
  const uint h = hsh[j];
  device const GZ* Z = bz + ulong(sval[j]) * kNN;
  for (uint s = h & A.mask;; s = (s + 1) & A.mask) {
    uint v = atomic_load_explicit(&ztab[s], memory_order_relaxed);
    while (v == kZEmpty) {                            // claim it, or learn who did
      uint expected = kZEmpty;
      if (atomic_compare_exchange_weak_explicit(&ztab[s], &expected, kZTag | j,
                                                memory_order_relaxed, memory_order_relaxed)) {
        rep[j] = kZTag | j;
        zslot[j] = s;
        return;
      }
      v = expected;
    }
    // Occupied: by a zone stored earlier, or by one this batch is interning. Both are already in
    // memory -- the arena from earlier batches, the candidate buffer from the expand kernel.
    const bool mine = (v & kZTag) != 0u;
    const uint hv = mine ? hsh[v & ~kZTag] : zhash[v];
    if (hv == h && fr_zone_eq(Z, mine ? bz + ulong(sval[v & ~kZTag]) * kNN : arena + ulong(v) * kNN)) {
      rep[j] = v;
      return;
    }
  }
}
// win[j] = 1 iff candidate j claimed a table slot, i.e. is the first copy of its zone.
kernel void fr_zwin(device const uint* keep [[buffer(0)]], device const uint* ctr [[buffer(1)]],
                    device const uint* rep [[buffer(2)]], device uint* win [[buffer(3)]],
                    constant uint& W [[buffer(4)]], uint j [[thread_position_in_grid]]) {
  if (j >= W) return;
  win[j] = (j < ctr[0] && keep[j] && rep[j] == (kZTag | j)) ? 1u : 0u;
}
// One thread per stored entry of a new zone: candidate buffer -> arena slot zcount + wpre[j];
// entry 0 also records the hash and resolves the table slot's tag.
struct ZPlaceArgs { uint W; uint zcount; };
kernel void fr_zplace(device const uint* win [[buffer(0)]], device const uint* wpre [[buffer(1)]],
                      device const uint* sval [[buffer(2)]], device const GZ* bz [[buffer(3)]],
                      device GZ* arena [[buffer(4)]], device const uint* hsh [[buffer(5)]],
                      device uint* zhash [[buffer(6)]], device atomic_uint* ztab [[buffer(7)]],
                      device const uint* zslot [[buffer(8)]], constant ZPlaceArgs& A [[buffer(9)]],
                      uint gid [[thread_position_in_grid]]) {
  const uint j = gid / uint(kNN), e = gid - j * uint(kNN);
  if (j >= A.W || !win[j]) return;
  const uint z = A.zcount + wpre[j];
  arena[ulong(z) * kNN + e] = bz[ulong(sval[j]) * kNN + e];
  if (e == 0) {
    zhash[z] = hsh[j];
    atomic_store_explicit(&ztab[zslot[j]], z, memory_order_relaxed);
  }
}

// ---- placing survivors ----------------------------------------------------------------------------
// Survivor j (sorted position, kept) becomes entry acount + kpre[j]: alive, its zone resolved,
// appended to the compacted new list and to the next frontier, both in key order.
struct PlaceArgs { uint W; uint acount; uint fbase; uint zcount; };
kernel void fr_place_new(device const uint* keep [[buffer(0)]], device const uint* kpre [[buffer(1)]],
                         device const ulong* sk [[buffer(2)]], device const uint* ctr [[buffer(3)]],
                         device ulong* new_k [[buffer(4)]], device uint* new_a [[buffer(5)]],
                         device uint* alive [[buffer(6)]], device ulong* fnext_k [[buffer(7)]],
                         device uint* fnext_a [[buffer(8)]], constant PlaceArgs& A [[buffer(9)]],
                         device const uint* rep [[buffer(10)]], device const uint* wpre [[buffer(11)]],
                         device uint* zref [[buffer(12)]], uint j [[thread_position_in_grid]]) {
  if (j >= A.W || j >= ctr[0] || !keep[j]) return;
  const uint r = kpre[j], a = A.acount + r;
  const uint v = rep[j];
  zref[a] = (v & kZTag) ? A.zcount + wpre[v & ~kZTag] : v;
  new_k[r] = sk[j];
  new_a[r] = a;
  alive[a] = 1u;
  fnext_k[A.fbase + r] = sk[j];
  fnext_a[A.fbase + r] = a;
}

// Table rebuild, after growth or compaction: clear, then insert every stored zone. The zones
// are distinct, so an insertion only looks for a free slot.
kernel void fr_fill_u32(device uint* v [[buffer(0)]], constant uint2& A [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
  if (i < A.x) v[i] = A.y;
}
kernel void fr_zrebuild(device const uint* zhash [[buffer(0)]], device atomic_uint* ztab [[buffer(1)]],
                        constant InternArgs& A [[buffer(2)]], uint z [[thread_position_in_grid]]) {
  if (z >= A.W) return;
  for (uint s = zhash[z] & A.mask;; s = (s + 1) & A.mask) {
    uint expected = kZEmpty;
    while (!atomic_compare_exchange_weak_explicit(&ztab[s], &expected, z, memory_order_relaxed,
                                                  memory_order_relaxed)) {
      if (expected != kZEmpty) break;               // taken: next slot
    }
    if (expected == kZEmpty) return;
  }
}

// ---- merging sorted runs ----------------------------------------------------------------------------
// out = merge(a, b) for two key-sorted runs, a's entries first among equal keys. When a filter
// prefix sum is given (`fa` / `fb` non-trivial), only entries whose flag is set are kept: an
// entry's output position is its rank among kept entries of its own run plus the number of kept
// entries of the other run that sort before it.
struct MergeArgs { uint na; uint nb_cap; uint filter; };
kernel void fr_merge_a(device const ulong* ak [[buffer(0)]], device const uint* aa [[buffer(1)]],
                       device const ulong* bk [[buffer(2)]], device const uint* nbp [[buffer(3)]],
                       device const uint* apre [[buffer(4)]], device const uint* bpre [[buffer(5)]],
                       device ulong* ok [[buffer(6)]], device uint* oa [[buffer(7)]],
                       constant MergeArgs& A [[buffer(8)]], uint i [[thread_position_in_grid]]) {
  if (i >= A.na) return;
  const uint nb = nbp[0];
  if (A.filter && apre[i + 1] == apre[i]) return;
  const uint lb = lower_bound64(bk, nb, ak[i]);
  const uint pos = (A.filter ? apre[i] : i) + (A.filter ? bpre[lb] : lb);
  ok[pos] = ak[i];
  oa[pos] = aa[i];
}
kernel void fr_merge_b(device const ulong* bk [[buffer(0)]], device const uint* ba [[buffer(1)]],
                       device const ulong* ak [[buffer(2)]], device const uint* nbp [[buffer(3)]],
                       device const uint* apre [[buffer(4)]], device const uint* bpre [[buffer(5)]],
                       device ulong* ok [[buffer(6)]], device uint* oa [[buffer(7)]],
                       constant MergeArgs& A [[buffer(8)]], uint j [[thread_position_in_grid]]) {
  if (j >= A.nb_cap || j >= nbp[0]) return;
  if (A.filter && bpre[j + 1] == bpre[j]) return;
  const uint ub = upper_bound64(ak, A.na, bk[j]);
  const uint pos = (A.filter ? apre[ub] : ub) + (A.filter ? bpre[j] : j);
  ok[pos] = bk[j];
  oa[pos] = ba[j];
}
// flag[i] = alive[idx[i]]
kernel void fr_alive_flags(device const uint* idx [[buffer(0)]], device const uint* alive [[buffer(1)]],
                           device uint* flag [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
  if (i < n) flag[i] = alive[idx[i]];
}

// ---- level end: frontier and arena compaction ------------------------------------------------------
// Compacts the frontier to its live entries (a later batch of the level may have dropped one).
struct CompactArgs { uint n; };
kernel void fr_compact_frontier(device const ulong* ik [[buffer(0)]], device const uint* ia [[buffer(1)]],
                                device const uint* flag [[buffer(2)]], device const uint* pre [[buffer(3)]],
                                device ulong* ok [[buffer(4)]], device uint* oa [[buffer(5)]],
                                constant CompactArgs& A [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if (i >= A.n || !flag[i]) return;
  ok[pre[i]] = ik[i];
  oa[pre[i]] = ia[i];
}
// Compaction, with the delta folded into the base so that the base holds every live entry:
// entries are renumbered in base order, and the arena keeps only the zones a live entry names.
kernel void fr_gc_mark(device const uint* base_a [[buffer(0)]], device const uint* zref [[buffer(1)]],
                       device uint* zmark [[buffer(2)]], constant uint& n [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
  if (i < n) zmark[zref[base_a[i]]] = 1u;
}
struct GcArgs { uint nzones; uint nbase; };
kernel void fr_gc_zcopy(device const uint* zmark [[buffer(0)]], device const uint* zpre [[buffer(1)]],
                        device const GZ* arena [[buffer(2)]], device GZ* arena2 [[buffer(3)]],
                        device const uint* zhash [[buffer(4)]], device uint* zhash2 [[buffer(5)]],
                        constant GcArgs& A [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  const uint z = gid / uint(kNN), e = gid - z * uint(kNN);
  if (z >= A.nzones || !zmark[z]) return;
  arena2[ulong(zpre[z]) * kNN + e] = arena[ulong(z) * kNN + e];
  if (e == 0) zhash2[zpre[z]] = zhash[z];
}
kernel void fr_gc_entries(device uint* base_a [[buffer(0)]], device const uint* zref [[buffer(1)]],
                          device uint* zref2 [[buffer(2)]], device const uint* zpre [[buffer(3)]],
                          device uint* remap [[buffer(4)]], device uint* alive2 [[buffer(5)]],
                          constant GcArgs& A [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if (i >= A.nbase) return;
  const uint e = base_a[i];
  zref2[i] = zpre[zref[e]];
  remap[e] = i;
  base_a[i] = i;
  alive2[i] = 1u;
}
kernel void fr_remap(device uint* a [[buffer(0)]], device const uint* remap [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) a[i] = remap[a[i]];
}
)FR_MSL";
}

}  // namespace metal
}  // namespace hora
