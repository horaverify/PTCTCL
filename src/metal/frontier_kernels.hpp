/// @file src/metal/frontier_kernels.hpp
/// @brief Metal kernels of the frontier engine: expand, sort, group, reduce, merge.
///
/// Compiled at run time after hdbm::metal::gdbm_source(), whose SIMD-group-cooperative zone
/// operations they call, and specialised per model through preprocessor macros:
///
///   PT_DIM   zone dimension (clocks + 1)        PT_NP    processes
///   GZ       stored-zone entry type (char/short/int)
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
// A kernel that runs one SIMD-group per item reads the number of SIMD-groups in its threadgroup
// from [[simdgroups_per_threadgroup]]: the host sizes each launch from that pipeline's own limits
// (FrontierMetal::launch_sgs), so the count differs between kernels and is not a constant here.


#define kN    (PT_DIM)
#define kNN   (PT_DIM * PT_DIM)
#define kNP   (PT_NP)
// The scan and the radix sort work in tiles of four elements a thread. They run at whatever
// threadgroup size the host found the pipeline supports -- a power of two from 32 to 256 -- so the
// tile is 4 * [[threads_per_threadgroup]], never a constant.

// ---- the zone arena: fixed-size pages, reached through a table of their GPU addresses ----------
// Growing the arena adds a page; no zone ever moves, so no old and new copy of the store exist at
// once. Zone z is entry (z & FR_PAGE_MASK) of page z >> FR_PAGE_SHIFT.
#ifndef FR_PAGE_SHIFT
#define FR_PAGE_SHIFT 16
#endif
#define FR_PAGE_MASK ((1u << FR_PAGE_SHIFT) - 1u)
#define FR_MAX_PAGES 4096
// npages is the number of allocated pages. A zone id past them -- only a corrupted id can be --
// is clamped to page 0 rather than followed: the arena is reached through raw GPU addresses, and
// an address outside every buffer faults or hangs the GPU, which can take the machine down. The
// clamped access stays inside the arena; HORA_GPU_CHECK catches the corruption that produced it.
struct ArenaPages { ulong npages; device GZ* p[FR_MAX_PAGES]; };
inline device GZ* zone_at(device const ArenaPages& A, uint z) {
  uint pg = z >> FR_PAGE_SHIFT;
  if (pg >= uint(A.npages)) pg = 0u;
  return A.p[pg] + ulong(z & FR_PAGE_MASK) * kNN;
}

// ---- small helpers --------------------------------------------------------------------------
// The zone hash of the interning table, over the entries as stored (GZ, sentinel for infinity).
inline uint fr_mix(uint x) {
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  return x ^ (x >> 16);
}
// Order-independent, so lanes fold shares and combine with a SIMD sum. Mirrored on the host
// (zone_hash in frontier_metal.mm) for the initial zone; the two must agree bit for bit.
inline uint fr_entry_hash(uint e, int v) { return fr_mix(e * 0x9e3779b9u ^ uint(v)); }
inline GZ fr_stored(int v) { return GZ(v >= GDBM_INF ? int(sizeof(GZ) == 1 ? int(GDBM_SINF)
                                                           : sizeof(GZ) == 2 ? int(GDBM_NINF)
                                                                             : GDBM_INF) : v); }
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

// Exclusive scan of one value per thread across the threadgroup, whatever its size: prefix sums
// within each SIMD-group, then over the SIMD-group totals (nsg of them, at most 32) in `sg`.
inline uint tg_scan(uint v, uint lane, uint sgid, uint nsg, threadgroup uint* sg, thread uint& total) {
  const uint pre = simd_prefix_exclusive_sum(v);
  const uint s = simd_sum(v);
  if (lane == 0) sg[sgid] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint off = 0, tot = 0;
  for (uint i = 0; i < nsg; ++i) { const uint x = sg[i]; off += (i < sgid) ? x : 0; tot += x; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total = tot;
  return pre + off;
}

// ---- expand ---------------------------------------------------------------------------------
struct ExpandArgs {
  uint  nwork;
  int   has_tclock;   // the target constrains clocks: test tdbm on every target-matching successor
  ulong sentinel;     // sort key of a disabled transition: above every discrete key
  int   extrap;       // apply Extra_LU+ (global bounds Lx, Ux) to every successor
  uint  prev_slot;    // frontier slot of the work item just before this batch (~0u: none)
};

// Conjoin the constraint triples c[from .. to).
inline bool fr_constrain_all(threadgroup int* D, device const int* c, uint from, uint to, uint lane) {
  for (uint k = from; k < to; ++k)
    if (!gdbm_constrain(D, kN, c[3 * k], c[3 * k + 1], c[3 * k + 2], lane)) return false;
  return true;
}

// One SIMD-group per work item w = (frontier zone wf[w], transition wr[w]). Mirrors the CPU
// successor pipeline exactly: every guard, every reset, the target invariant, delay unless the
// target is committed or urgent, the invariant again, and the active-clock reduction. What
// depends only on the target's discrete state -- frozen (flag bit 0), the invariant (rv), the dead
// clocks (rdead) -- the host derived once per transition. Writes the successor zone to its own
// slot, so the output order is the work order and the search is deterministic.
kernel void fr_expand(device const uint*  wf      [[buffer(0)]],
                      device const uint*  wr      [[buffer(1)]],
                      device const uint*  fz      [[buffer(2)]],
                      device const ArenaPages& AP [[buffer(3)]],
                      device const ulong* rkey    [[buffer(4)]],
                      device const uint*  rflags  [[buffer(5)]],
                      device const uint*  rg_off  [[buffer(6)]],
                      device const int*   rg      [[buffer(7)]],
                      device const uint*  rr_off  [[buffer(8)]],
                      device const int*   rr      [[buffer(9)]],
                      device const uint*  rv_off  [[buffer(10)]],
                      device const int*   rv      [[buffer(11)]],
                      device const ulong* rdead   [[buffer(12)]],
                      device const uint*  alive   [[buffer(13)]],
                      device const int*   tdbm    [[buffer(16)]],
                      device GZ*          bz      [[buffer(17)]],
                      device ulong*       skey    [[buffer(18)]],
                      device uint*        sval    [[buffer(19)]],
                      device atomic_uint* ctr     [[buffer(20)]],
                      constant ExpandArgs& A      [[buffer(21)]],
                      device const int*   Lx      [[buffer(22)]],
                      device const int*   Ux      [[buffer(23)]],
                      device const uint*  zref    [[buffer(24)]],
                      device uint*        hshw    [[buffer(25)]],
                      threadgroup int*    tgm     [[threadgroup(0)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint sgs  [[simdgroups_per_threadgroup]],
                      uint sgid [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  const uint w = tg * sgs + sgid;
  if (w >= A.nwork) return;
  threadgroup int* D = tgm + sgid * (kNN + kN);   // the zone, then a row of scratch
  const uint r = wr[w];
  const ulong key = rkey[r];
  // A frontier zone that died during this round -- a zone inserted in it covers it, or absorbed
  // it into a hull -- is not expanded: that zone is in the next frontier, and its successors
  // cover these. The first work item of each live frontier zone counts it as explored (ctr[10]).
  const uint slot = wf[w], entry = fz[slot];
  const bool live = alive[entry] != 0u;
  if (lane == 0 && live && slot != (w > 0 ? wf[w - 1] : A.prev_slot))
    atomic_fetch_add_explicit(&ctr[10], 1u, memory_order_relaxed);
  if (!live) {
    if (lane == 0) { skey[w] = A.sentinel; sval[w] = w; }
    return;
  }
  gdbm_load(D, zone_at(AP, zref[entry]), kN, lane);
  bool ok = true;
  for (uint g = rg_off[r]; ok && g < rg_off[r + 1]; ++g)
    ok = gdbm_constrain(D, kN, rg[3 * g], rg[3 * g + 1], rg[3 * g + 2], lane);
  const uint v0 = rv_off[r], v1 = rv_off[r + 1];
  if (ok) {
    for (uint k = rr_off[r]; k < rr_off[r + 1]; ++k) gdbm_reset(D, kN, rr[k], lane);
    ok = fr_constrain_all(D, rv, v0, v1, lane);
  }
  if (ok && !(rflags[r] & 1u)) {
    gdbm_up(D, kN, lane);
    ok = fr_constrain_all(D, rv, v0, v1, lane);
  }
  if (ok) {
    for (ulong dead = rdead[r]; dead; dead &= dead - 1) gdbm_reset(D, kN, int(ctz(dead)), lane);
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
      // Store, and hash as stored while the zone is on chip: interning needs the hash of every
      // survivor, and reading the zone back for it cost a pass over the candidate buffer.
      device GZ* out = bz + ulong(w) * kNN;
      uint h = 0;
      GDBM_FOR_E(e, lane, kNN) { const GZ sv = fr_stored(D[e]); out[e] = sv; h += fr_entry_hash(uint(e), int(sv)); }
      h = fr_mix(simd_sum(h));
      if (lane == 0) { hshw[w] = h; atomic_fetch_add_explicit(&ctr[0], 1u, memory_order_relaxed); }
    }
  }
  if (lane == 0) { skey[w] = ok ? key : A.sentinel; sval[w] = w; }
}

// ---- exclusive scan -------------------------------------------------------------------------
struct ScanArgs { uint n_in, n_out, tile; };
// out[i] = sum of in[0..i) within this block of A.tile = 4 * threads elements; sums[block] = the
// block total. Values at i >= n_in read as 0, so a scan of n values over n + 1 outputs leaves the
// total at out[n].
kernel void fr_scan_block(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
                          device uint* sums [[buffer(2)]], constant ScanArgs& A [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint blk [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sgid [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup uint sg[32];
  const uint base = blk * A.tile + tid * 4;
  uint v[4], s = 0;
  for (uint q = 0; q < 4; ++q) { const uint i = base + q; v[q] = (i < A.n_in) ? in[i] : 0u; s += v[q]; }
  uint tot;
  uint run = tg_scan(s, lane, sgid, nsg, sg, tot);
  for (uint q = 0; q < 4; ++q) { const uint i = base + q; if (i < A.n_out) out[i] = run; run += v[q]; }
  if (tid == 0) sums[blk] = tot;
}
kernel void fr_scan_add(device uint* out [[buffer(0)]], device const uint* sums [[buffer(1)]],
                        constant ScanArgs& A [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < A.n_out) out[i] += sums[i / A.tile];
}

// ---- radix sort: stable LSD, 8 bits a pass ----------------------------------------------------
// A tile is 4 * T elements for a threadgroup of T threads, T a power of two from 32 to 256 (so the
// 256 digits split evenly over the threads). Histogram and scatter run at the same T.
struct RsArgs { uint n, ntiles, shift; };
kernel void fr_rs_hist(device const ulong* kin [[buffer(0)]], device uint* ghist [[buffer(1)]],
                       constant RsArgs& A [[buffer(2)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint tile [[threadgroup_position_in_grid]],
                       uint tpt [[threads_per_threadgroup]]) {
  threadgroup atomic_uint h[256];
  for (uint d = tid; d < 256; d += tpt) atomic_store_explicit(&h[d], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint base = tile * tpt * 4;
  for (uint q = 0; q < 4; ++q) {
    const uint i = base + q * tpt + tid;
    if (i < A.n) atomic_fetch_add_explicit(&h[uint((kin[i] >> A.shift) & 255ul)], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = tid; d < 256; d += tpt)
    ghist[d * A.ntiles + tile] = atomic_load_explicit(&h[d], memory_order_relaxed);
}
// Sort the tile locally by the digit (eight stable 1-bit splits), then write each element at
// its digit's global offset for this tile plus its rank among the tile's elements of that digit.
// The tile's keys and values, twice over (the splits ping-pong), live in threadgroup memory the
// host sizes from T: kbuf holds 2 * 4T ulongs, vbuf 2 * 4T uints.
kernel void fr_rs_scatter(device const ulong* kin [[buffer(0)]], device const uint* vin [[buffer(1)]],
                          device ulong* kout [[buffer(2)]], device uint* vout [[buffer(3)]],
                          device const uint* gpre [[buffer(4)]], constant RsArgs& A [[buffer(5)]],
                          threadgroup ulong* kbuf [[threadgroup(0)]],
                          threadgroup uint* vbuf [[threadgroup(1)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint tile [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sgid [[simdgroup_index_in_threadgroup]],
                          uint tpt [[threads_per_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup uint sg[32];
  threadgroup uint dstart[256];
  threadgroup atomic_uint h[256];
  const uint T4 = tpt * 4;
  const uint base = tile * T4;
  const uint cnt = min(T4, A.n - base);
  threadgroup ulong* ks = kbuf;      threadgroup ulong* kd = kbuf + T4;
  threadgroup uint*  vs = vbuf;      threadgroup uint*  vd = vbuf + T4;
  for (uint q = 0; q < 4; ++q) {               // coalesced load; padding sorts last (digit 255)
    const uint i = q * tpt + tid;
    ks[i] = (i < cnt) ? kin[base + i] : ~0ul;
    vs[i] = (i < cnt) ? vin[base + i] : 0u;
  }
  for (uint d = tid; d < 256; d += tpt) atomic_store_explicit(&h[d], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint b = 0; b < 8; ++b) {
    ulong kk[4]; uint vv[4]; uint bit[4]; uint zeros = 0;
    for (uint q = 0; q < 4; ++q) {
      const uint i = tid * 4 + q;
      kk[q] = ks[i]; vv[q] = vs[i];
      bit[q] = uint((kk[q] >> (A.shift + b)) & 1ul);
      zeros += 1u - bit[q];
    }
    uint Z;
    uint zb = tg_scan(zeros, lane, sgid, nsg, sg, Z);  // zeros before this thread's items
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
  // Digit starts within the tile: thread t takes digits [t * dpt, (t + 1) * dpt).
  const uint dpt = 256 / tpt;
  uint mine = 0;
  for (uint k = 0; k < dpt; ++k) mine += atomic_load_explicit(&h[tid * dpt + k], memory_order_relaxed);
  uint tot;
  uint run = tg_scan(mine, lane, sgid, nsg, sg, tot);
  for (uint k = 0; k < dpt; ++k) {
    const uint d = tid * dpt + k;
    dstart[d] = run;
    run += atomic_load_explicit(&h[d], memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint q = 0; q < 4; ++q) {
    const uint i = q * tpt + tid;
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

// The a_LU bounds of every bucket, computed once per batch: per-location maxima over processes,
// floored by the query's constants, and (0, 0) for the reference clock. Stored narrow (FR_BOUND:
// short unless the model's constants need int), kBoundNone standing for kAluNegInf.
#ifndef FR_BOUND
#define FR_BOUND short
#endif
constant int kBoundNone = (sizeof(FR_BOUND) == 2) ? -32768 : (-2147483647 - 1);
struct BoundView {
  device const FR_BOUND* p;
  int operator[](int i) const { const int v = int(p[i]); return v == kBoundNone ? GDBM_ALU_NEG_INF : v; }
};
kernel void fr_bucket_bounds(device const ulong* bkey [[buffer(0)]], device const uint* hpre [[buffer(1)]],
                             device const int* Lloc [[buffer(2)]], device const int* Uloc [[buffer(3)]],
                             device const int* Lq [[buffer(4)]], device const int* Uq [[buffer(5)]],
                             device const int* locbase [[buffer(6)]], device const int* kshift [[buffer(7)]],
                             device const ulong* kmask [[buffer(8)]], device FR_BOUND* Lb [[buffer(9)]],
                             device FR_BOUND* Ub [[buffer(10)]], constant uint& W [[buffer(11)]],
                             uint gid [[thread_position_in_grid]]) {
  const uint b = gid / uint(kN), c = gid - b * uint(kN);
  if (b >= hpre[W]) return;
  const ulong key = bkey[b];
  int l = Lq[c], u = Uq[c];
  for (int p = 0; p < kNP; ++p) {
    const int gl = locbase[p] + loc_of(key, kshift, kmask, p);
    l = max(l, Lloc[gl * kN + c]);
    u = max(u, Uloc[gl * kN + c]);
  }
  if (c == 0) l = u = 0;
  Lb[ulong(b) * kN + c] = FR_BOUND(l == GDBM_ALU_NEG_INF ? kBoundNone : l);
  Ub[ulong(b) * kN + c] = FR_BOUND(u == GDBM_ALU_NEG_INF ? kBoundNone : u);
}

// cap2: big2's capacity. Phase 2 has a row per survivor AND per stored zone of each bucket, so
// its large rows can outnumber the batch (one bucket of many stored zones); the host sizes big2
// for every row it can have, and an append past cap2 is reported (ctr[9]) instead of written.
struct ReduceArgs { uint W; uint cap2; };

// Both phases split their rows by size. A row comparing against at most FR_SOLO zones is decided
// by one lane alone -- the common case wherever discrete states hold a handful of zones, and there
// a SIMD-group per row would leave most lanes idle. A larger row is appended to a list (ctr[6] or
// ctr[7] counts it) that a second kernel takes one SIMD-group per row, every lane testing a
// different zone of the row's bucket, so a vote ends the row at its first dominator. Keeping the
// large rows in a SIMD-group each, rather than one SIMD-group working through every large row of
// its 32, is what keeps buckets of thousands of zones (grc) parallel.
#ifndef FR_SOLO
#define FR_SOLO 8
#endif
inline uint fr_stored_entry(uint4 st, uint t, device const uint* base_a, device const uint* delta_a) {
  return (t < st.y) ? base_a[st.x + t] : delta_a[st.z + t - st.y];
}

// Phase 1: each candidate (sorted position i) against its bucket's live stored zones; the small
// rows here, the large ones listed for fr_reduce_stored_big.
kernel void fr_reduce_stored(device const uint*  hpre    [[buffer(0)]],
                             device const uint*  head    [[buffer(1)]],
                             device const uint4* binfo   [[buffer(2)]],
                             device const FR_BOUND* Lb   [[buffer(3)]],
                             device const uint*  base_a  [[buffer(4)]],
                             device const uint*  delta_a [[buffer(5)]],
                             device const ArenaPages& AP [[buffer(6)]],
                             device const uint*  alive   [[buffer(7)]],
                             device const uint*  sval    [[buffer(8)]],
                             device const GZ*    bz      [[buffer(9)]],
                             device const FR_BOUND* Ub   [[buffer(10)]],
                             device uint*        keep1   [[buffer(11)]],
                             device const uint*  ctr     [[buffer(12)]],
                             device const uint*  zref    [[buffer(13)]],
                             device uint*        big1    [[buffer(14)]],
                             device atomic_uint* bigc    [[buffer(15)]],
                             uint gid [[thread_position_in_grid]]) {
  // One thread per candidate: no SIMD-group cooperation here, so a plain thread grid.
  const uint i0 = gid - (gid & 31u);
  const uint lane = gid & 31u;
  const uint nv = ctr[0];
  if (i0 >= nv) return;
  const uint i = i0 + lane;
  const bool mine = i < nv;
  uint b = 0;
  uint4 st = uint4(0);
  if (mine) { b = hpre[i] + head[i] - 1; st = binfo[2 * b]; }
  const uint nold = st.y + st.w;
  const bool big = mine && nold > FR_SOLO;
  if (mine && !big) {
    bool drop = false;
    if (nold) {
      device const GZ* X = bz + ulong(sval[i]) * kNN;
      const BoundView L{Lb + ulong(b) * kN}, U{Ub + ulong(b) * kN};
      for (uint t = 0; !drop && t < nold; ++t) {
        const uint a = fr_stored_entry(st, t, base_a, delta_a);
        drop = alive[a] != 0u && gdbm_alu_le_1(X, zone_at(AP, zref[a]), kN, L, U);
      }
    }
    keep1[i] = drop ? 0u : 1u;
  }
  if (big) big1[atomic_fetch_add_explicit(&bigc[6], 1u, memory_order_relaxed)] = i;
}
// Phase 1, the large rows: one SIMD-group per listed candidate.
kernel void fr_reduce_stored_big(device const uint*  hpre    [[buffer(0)]],
                                 device const uint*  head    [[buffer(1)]],
                                 device const uint4* binfo   [[buffer(2)]],
                                 device const FR_BOUND* Lb   [[buffer(3)]],
                                 device const uint*  base_a  [[buffer(4)]],
                                 device const uint*  delta_a [[buffer(5)]],
                                 device const ArenaPages& AP [[buffer(6)]],
                                 device const uint*  alive   [[buffer(7)]],
                                 device const uint*  sval    [[buffer(8)]],
                                 device const GZ*    bz      [[buffer(9)]],
                                 device const FR_BOUND* Ub   [[buffer(10)]],
                                 device uint*        keep1   [[buffer(11)]],
                                 device const uint*  ctr     [[buffer(12)]],
                                 device const uint*  zref    [[buffer(13)]],
                                 device const uint*  big1    [[buffer(14)]],
                                 threadgroup int*    tgm     [[threadgroup(0)]],
                                 uint tg [[threadgroup_position_in_grid]],
                                 uint sgs  [[simdgroups_per_threadgroup]],
                                 uint sgid [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  const uint k = tg * sgs + sgid;
  if (k >= ctr[6]) return;
  const uint i = big1[k];
  const uint b = hpre[i] + head[i] - 1;
  const uint4 st = binfo[2 * b];
  const uint nold = st.y + st.w;
  const BoundView L{Lb + ulong(b) * kN}, U{Ub + ulong(b) * kN};
  threadgroup int* X = tgm + sgid * kNN;
  gdbm_load(X, bz + ulong(sval[i]) * kNN, kN, lane);
  bool drop = false;
  for (uint base = 0; !drop && base < nold; base += 32) {
    const uint t = base + lane;
    bool d = false;
    if (t < nold) {
      const uint a = fr_stored_entry(st, t, base_a, delta_a);
      d = alive[a] != 0u && gdbm_alu_le_1(X, zone_at(AP, zref[a]), kN, L, U);
    }
    drop = simd_any(d);
  }
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
                     device uint* ulist [[buffer(6)]], uint b [[thread_position_in_grid]]) {
  if (b >= A.W) return;
  if (b >= hpre[A.W]) { rowcnt[b] = 0; return; }
  const uint4 st = binfo[2 * b], nw = binfo[2 * b + 1];
  const uint ns = k1pre[nw.x + nw.y] - k1pre[nw.x];
  rowcnt[b] = ns ? st.y + st.w + ns : 0u;   // no survivor: nothing new, and no stored zone can die
  if (st.y + st.w == 0 && ns) atomic_fetch_add_explicit(&ctr[2], 1u, memory_order_relaxed);
  // A bucket whose survivors may merge with something: listed for the convex-union phase.
  if (ns && st.y + st.w + ns > 1) ulist[atomic_fetch_add_explicit(&ctr[8], 1u, memory_order_relaxed)] = b;
}

// A stored entry dies: it leaves the passed set, and when it was the last entry naming its zone,
// the zone leaves the table (a tombstone keeps probe chains intact) and is queued for reuse.
// Queued, not freed: a zone dying in this level may still be named by a frontier entry that a
// later batch of the level has yet to expand, so the host recycles the queue at the level's end.
constant uint kZTomb = 0xfffffffeu;
inline void fr_kill(uint a, device uint* alive, device const uint* zref, device atomic_uint* zcnt,
                    device atomic_uint* ztab, device const uint* ztslot, device uint* pend,
                    device atomic_uint* ctr) {
  alive[a] = 0u;
  atomic_fetch_add_explicit(&ctr[3], 1u, memory_order_relaxed);
  const uint z = zref[a];
  if (atomic_fetch_sub_explicit(&zcnt[z], 1u, memory_order_relaxed) == 1u) {
    if (ztslot[z] != 0xffffffffu) atomic_store_explicit(&ztab[ztslot[z]], kZTomb, memory_order_relaxed);
    pend[atomic_fetch_add_explicit(&ctr[5], 1u, memory_order_relaxed)] = z;
  }
}

// Phase 2: rows are a bucket's stored zones, then its survivors; columns its survivors. A stored
// row dies to a survivor strictly covering it; a survivor row is dropped by another covering it
// that it does not cover back, or that ranks earlier.
struct Phase2Row {
  bool mine, stored;
  uint b, a, pos, r, s0, ns;
};
inline bool fr_row_dominated(thread const Phase2Row& R, device const GZ* X, device const GZ* Y,
                             uint t, BoundView L, BoundView U) {
  if (R.stored) return gdbm_alu_le_1(X, Y, kN, L, U) && !gdbm_alu_le_1(Y, X, kN, L, U);
  if (t == R.r) return false;
  return gdbm_alu_le_1(X, Y, kN, L, U) && (t < R.r || !gdbm_alu_le_1(Y, X, kN, L, U));
}
kernel void fr_reduce_pairs(device const uint*  rowoff  [[buffer(0)]],
                            device const uint*  hpre    [[buffer(1)]],
                            device const uint4* binfo   [[buffer(2)]],
                            device const FR_BOUND* Lb   [[buffer(3)]],
                            device const uint*  base_a  [[buffer(4)]],
                            device const uint*  delta_a [[buffer(5)]],
                            device const ArenaPages& AP [[buffer(6)]],
                            device uint*        alive   [[buffer(7)]],
                            device const uint*  sval    [[buffer(8)]],
                            device const GZ*    bz      [[buffer(9)]],
                            device const FR_BOUND* Ub   [[buffer(10)]],
                            device uint*        keep    [[buffer(11)]],
                            device const uint*  k1pre   [[buffer(12)]],
                            device const uint*  surv    [[buffer(13)]],
                            constant ReduceArgs& A      [[buffer(14)]],
                            device atomic_uint* ctr     [[buffer(15)]],
                            device const uint*  zref    [[buffer(16)]],
                            device atomic_uint* zcnt    [[buffer(17)]],
                            device atomic_uint* ztab    [[buffer(18)]],
                            device const uint*  ztslot  [[buffer(19)]],
                            device uint*        pend    [[buffer(20)]],
                            device uint*        big2    [[buffer(21)]],
                            uint gid [[thread_position_in_grid]]) {
  // One thread per row: no SIMD-group cooperation here, so a plain thread grid.
  const uint r0 = gid - (gid & 31u);
  const uint lane = gid & 31u;
  const uint nrows = rowoff[A.W];
  if (r0 >= nrows) return;
  Phase2Row R;
  R.mine = r0 + lane < nrows;
  R.stored = false; R.b = R.a = R.pos = R.r = R.s0 = R.ns = 0;
  if (R.mine) {
    const uint row = r0 + lane;
    R.b = segment_of(rowoff, hpre[A.W], row);
    const uint local = row - rowoff[R.b];
    const uint4 st = binfo[2 * R.b], nw = binfo[2 * R.b + 1];
    const uint nold = st.y + st.w;
    R.s0 = k1pre[nw.x];
    R.ns = k1pre[nw.x + nw.y] - R.s0;
    R.stored = local < nold;
    if (R.stored) {
      R.a = fr_stored_entry(st, local, base_a, delta_a);
      if (alive[R.a] == 0u) R.mine = false;           // already dead: nothing to decide
    } else {
      R.r = local - nold;
      R.pos = surv[R.s0 + R.r];
    }
  }
  const bool big = R.mine && R.ns > FR_SOLO;
  if (R.mine && !big) {
    const BoundView L{Lb + ulong(R.b) * kN}, U{Ub + ulong(R.b) * kN};
    device const GZ* X = R.stored ? zone_at(AP, zref[R.a]) : bz + ulong(sval[R.pos]) * kNN;
    bool dom = false;
    for (uint t = 0; !dom && t < R.ns; ++t)
      dom = fr_row_dominated(R, X, bz + ulong(sval[surv[R.s0 + t]]) * kNN, t, L, U);
    if (R.stored) { if (dom) fr_kill(R.a, alive, zref, zcnt, ztab, ztslot, pend, ctr); }
    else keep[R.pos] = dom ? 0u : 1u;
  }
  if (big) {
    const uint k = atomic_fetch_add_explicit(&ctr[7], 1u, memory_order_relaxed);
    if (k < A.cap2) big2[k] = r0 + lane;
    else atomic_fetch_add_explicit(&ctr[9], 1u, memory_order_relaxed);
  }
}
// Phase 2, the large rows: one SIMD-group per listed row.
kernel void fr_reduce_pairs_big(device const uint*  rowoff  [[buffer(0)]],
                                device const uint*  hpre    [[buffer(1)]],
                                device const uint4* binfo   [[buffer(2)]],
                                device const FR_BOUND* Lb   [[buffer(3)]],
                                device const uint*  base_a  [[buffer(4)]],
                                device const uint*  delta_a [[buffer(5)]],
                                device const ArenaPages& AP [[buffer(6)]],
                                device uint*        alive   [[buffer(7)]],
                                device const uint*  sval    [[buffer(8)]],
                                device const GZ*    bz      [[buffer(9)]],
                                device const FR_BOUND* Ub   [[buffer(10)]],
                                device uint*        keep    [[buffer(11)]],
                                device const uint*  k1pre   [[buffer(12)]],
                                device const uint*  surv    [[buffer(13)]],
                                constant ReduceArgs& A      [[buffer(14)]],
                                device atomic_uint* ctr     [[buffer(15)]],
                                device const uint*  zref    [[buffer(16)]],
                                device atomic_uint* zcnt    [[buffer(17)]],
                                device atomic_uint* ztab    [[buffer(18)]],
                                device const uint*  ztslot  [[buffer(19)]],
                                device uint*        pend    [[buffer(20)]],
                                device const uint*  big2    [[buffer(21)]],
                                device const uint*  nbig    [[buffer(22)]],
                                threadgroup int*    tgm     [[threadgroup(0)]],
                                uint tg [[threadgroup_position_in_grid]],
                                uint sgs  [[simdgroups_per_threadgroup]],
                                uint sgid [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  const uint k = tg * sgs + sgid;
  if (k >= min(nbig[7], A.cap2)) return;
  const uint row = big2[k];
  const uint b = segment_of(rowoff, hpre[A.W], row);
  const uint local = row - rowoff[b];
  const uint4 st = binfo[2 * b], nw = binfo[2 * b + 1];
  const uint nold = st.y + st.w;
  const uint s0 = k1pre[nw.x], ns = k1pre[nw.x + nw.y] - s0;
  const bool stored = local < nold;
  const uint a = stored ? fr_stored_entry(st, local, base_a, delta_a) : 0u;
  const uint r = stored ? 0u : local - nold;
  const uint pos = stored ? 0u : surv[s0 + r];
  const BoundView L{Lb + ulong(b) * kN}, U{Ub + ulong(b) * kN};
  threadgroup int* X = tgm + sgid * kNN;
  gdbm_load(X, stored ? zone_at(AP, zref[a]) : bz + ulong(sval[pos]) * kNN, kN, lane);
  bool dom = false;
  for (uint base = 0; !dom && base < ns; base += 32) {
    const uint t = base + lane;
    bool d = false;
    if (t < ns) {
      device const GZ* Y = bz + ulong(sval[surv[s0 + t]]) * kNN;
      if (stored) d = gdbm_alu_le_1(X, Y, kN, L, U) && !gdbm_alu_le_1(Y, X, kN, L, U);
      else if (t != r) d = gdbm_alu_le_1(X, Y, kN, L, U) && (t < r || !gdbm_alu_le_1(Y, X, kN, L, U));
    }
    dom = simd_any(d);
  }
  if (lane == 0) {
    if (stored) { if (dom) fr_kill(a, alive, zref, zcnt, ztab, ztslot, pend, ctr); }
    else keep[pos] = dom ? 0u : 1u;
  }
}

// ---- convex union -----------------------------------------------------------------------------------
// The CPU engine's merge (NetPassed::add, step 4): each new zone, in order, absorbs any zone among
// the FR_MERGE_SCAN newest live zones of its discrete state whose union with it is convex,
// restarting the scan after every merge; it then stands as their hull. Merging changes what is
// stored, never what is reachable: the hull is exactly the union. One SIMD-group per listed
// bucket, its survivors in rank order. "Newest" here: this batch's earlier survivors, latest
// first, then the delta's entries and the base's, each latest first.
//
// Before it merges, a survivor is tested against the hulls the earlier survivors became, and
// dropped if one covers it: inserted one at a time (the CPU engine's canonical order), it would
// have met those hulls, not the zones phase 2 compared it with. Phases 1 and 2 already compared
// it with every stored zone and every earlier survivor as it was, so only the hulls are new.
// A survivor that became a hull is marked keep = 2 while its bucket is processed, then 1 again.
#ifndef FR_MERGE_SCAN
#define FR_MERGE_SCAN 32
#endif
#ifndef FR_HULL_COVER
#define FR_HULL_COVER 0   // the test against earlier hulls (HORA_GPU_HULL_COVER=1): off by default
#endif
kernel void fr_union(device const uint*  ulist   [[buffer(0)]],
                     device const uint4* binfo   [[buffer(1)]],
                     device const uint*  base_a  [[buffer(2)]],
                     device const uint*  delta_a [[buffer(3)]],
                     device const ArenaPages& AP [[buffer(4)]],
                     device uint*        alive   [[buffer(5)]],
                     device const uint*  sval    [[buffer(6)]],
                     device GZ*          bz      [[buffer(7)]],
                     device uint*        keep    [[buffer(8)]],
                     device const uint*  k1pre   [[buffer(9)]],
                     device const uint*  surv    [[buffer(10)]],
                     device const uint*  zref    [[buffer(11)]],
                     device atomic_uint* zcnt    [[buffer(12)]],
                     device atomic_uint* ztab    [[buffer(13)]],
                     device const uint*  ztslot  [[buffer(14)]],
                     device uint*        pend    [[buffer(15)]],
                     device atomic_uint* ctr     [[buffer(16)]],
                     device uint*        hshw    [[buffer(17)]],
                     device const FR_BOUND* Lb   [[buffer(18)]],
                     device const FR_BOUND* Ub   [[buffer(19)]],
                     threadgroup int*    tgm     [[threadgroup(0)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint sgs  [[simdgroups_per_threadgroup]],
                     uint sgid [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  const uint k = tg * sgs + sgid;
  if (k >= atomic_load_explicit(&ctr[8], memory_order_relaxed)) return;
  const uint b = ulist[k];
  const uint4 st = binfo[2 * b], nw = binfo[2 * b + 1];
  const uint s0 = k1pre[nw.x], ns = k1pre[nw.x + nw.y] - s0;
  threadgroup int* X = tgm + sgid * (2 * kNN);
  threadgroup int* H = X + kNN;
  const BoundView L{Lb + ulong(b) * kN}, U{Ub + ulong(b) * kN};
  for (uint r = 0; r < ns; ++r) {
    const uint pos = surv[s0 + r];
    if (simd_broadcast_first(keep[pos]) == 0u) continue;   // dropped, or absorbed earlier
    const uint w = sval[pos];
    // Covered by a hull an earlier survivor became? One earlier survivor a lane.
    device const GZ* Z = bz + ulong(w) * kNN;
    bool covered = false;
    for (uint base = 0; FR_HULL_COVER && base < r && !covered; base += 32u) {
      const uint r2 = base + lane;
      bool c = false;
      if (r2 < r) {
        const uint p2 = surv[s0 + r2];
        c = keep[p2] == 2u && gdbm_alu_le_1(Z, bz + ulong(sval[p2]) * kNN, kN, L, U);
      }
      covered = simd_any(c);
    }
    if (covered) {
      if (lane == 0) keep[pos] = 0u;
      simdgroup_barrier(mem_flags::mem_device);
      continue;
    }
    gdbm_load(X, bz + ulong(w) * kNN, kN, lane);
    bool changed = false;
    // Every merge removes a zone of the bucket, so there are at most as many rounds as zones.
    // Past that the absorbed zones are not being seen as gone: stop and report (ctr[11]) rather
    // than spin -- a kernel that never ends takes the whole GPU, and the machine, with it.
    const uint max_rounds = ns + st.y + st.w + 2u;
    uint rounds = 0;
    for (bool merged = true; merged;) {
      if (++rounds > max_rounds) {
        if (lane == 0) atomic_fetch_add_explicit(&ctr[11], 1u, memory_order_relaxed);
        break;
      }
      merged = false;
      uint seen = 0;
      for (int r2 = int(r) - 1; r2 >= 0 && seen < FR_MERGE_SCAN && !merged; --r2) {
        const uint p2 = surv[s0 + uint(r2)];
        if (simd_broadcast_first(keep[p2]) == 0u) continue;
        ++seen;
        if (gdbm_try_union(X, bz + ulong(sval[p2]) * kNN, H, kN, lane)) {
          merged = changed = true;
          if (lane == 0) keep[p2] = 0u;
        }
      }
      for (uint side = 0; side < 2 && !merged; ++side) {   // the delta, then the base
        const uint first = side ? st.x : st.z, cnt = side ? st.y : st.w;
        device const uint* run = side ? base_a : delta_a;
        for (int t = int(cnt) - 1; t >= 0 && seen < FR_MERGE_SCAN && !merged; --t) {
          const uint a = run[first + uint(t)];
          if (simd_broadcast_first(alive[a]) == 0u) continue;
          ++seen;
          const bool u = gdbm_try_union(X, zone_at(AP, zref[a]), H, kN, lane);
          // The lanes must agree (see GDBM_FOR_E in gdbm.hpp): a split answer kills the zone on
          // one lane and writes the hull on none. Reported as an error (ctr[13]), never acted on.
          if (simd_any(u) != simd_all(u)) {
            if (lane == 0) atomic_fetch_add_explicit(&ctr[13], 1u, memory_order_relaxed);
            continue;
          }
          if (u) {
            merged = changed = true;
            if (lane == 0) fr_kill(a, alive, zref, zcnt, ztab, ztslot, pend, ctr);
          }
        }
      }
      simdgroup_barrier(mem_flags::mem_device);           // lane 0's flag writes, before rescanning
    }
    if (changed) {   // the hull replaces the candidate: store it, and hash it as stored
      device GZ* out = bz + ulong(w) * kNN;
      uint h = 0;
      GDBM_FOR_E(e, lane, kNN) { const GZ sv = fr_stored(X[e]); out[e] = sv; h += fr_entry_hash(uint(e), int(sv)); }
      h = fr_mix(simd_sum(h));
      if (lane == 0) { hshw[w] = h; keep[pos] = 2u; }
      simdgroup_barrier(mem_flags::mem_device);           // the hull, before later survivors read it
    }
  }
  for (uint r = lane; r < ns; r += 32u) {                  // keep is a 0/1 flag again
    const uint pos = surv[s0 + r];
    if (keep[pos] == 2u) keep[pos] = 1u;
  }
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
// A probe sequence is at most the table long, and a slot's claim is retried a bounded number of
// times. The host keeps the table at most half full, so neither bound is reached unless that
// invariant is broken; if it is, the error counter ctr[9] says so and the host abandons the GPU
// search, instead of a thread spinning on a full table.
constant uint kNoSlot = 0xffffffffu;
#define FR_CAS_TRIES 4096


inline bool fr_zone_eq(device const GZ* a, device const GZ* b) {
  for (int e = 0; e < kNN; ++e) if (a[e] != b[e]) return false;
  return true;
}
struct InternArgs { uint W; uint mask; };
// An interning that found neither its zone nor a free slot (see kNoSlot): the candidate stands as
// its own zone, outside the table, and ctr[9] marks the batch unusable.
inline void fr_intern_fail(uint j, device atomic_uint* ctr_err, device uint* rep, device uint* zslot) {
  atomic_fetch_add_explicit(ctr_err, 1u, memory_order_relaxed);
  rep[j] = kZTag | j;
  zslot[j] = kNoSlot;
}
kernel void fr_intern(device const uint* keep [[buffer(0)]], device const uint* ctr [[buffer(1)]],
                      device const uint* sval [[buffer(2)]], device const GZ* bz [[buffer(3)]],
                      device const ArenaPages& AP [[buffer(4)]], device const uint* hsh [[buffer(5)]],
                      device const uint* zhash [[buffer(6)]], device atomic_uint* ztab [[buffer(7)]],
                      device uint* rep [[buffer(8)]], device uint* zslot [[buffer(9)]],
                      constant InternArgs& A [[buffer(10)]], device atomic_uint* ctr_err [[buffer(11)]],
                      uint j [[thread_position_in_grid]]) {
  if (j >= A.W || j >= ctr[0] || !keep[j]) return;
  const uint h = hsh[sval[j]];
  device const GZ* Z = bz + ulong(sval[j]) * kNN;
  uint s = h & A.mask;
  for (uint probe = 0; probe <= A.mask; ++probe, s = (s + 1) & A.mask) {
    uint v = atomic_load_explicit(&ztab[s], memory_order_relaxed);
    for (uint tries = 0; v == kZEmpty; ++tries) {     // claim it, or learn who did
      if (tries == FR_CAS_TRIES) { fr_intern_fail(j, ctr_err, rep, zslot); return; }
      uint expected = kZEmpty;
      if (atomic_compare_exchange_weak_explicit(&ztab[s], &expected, kZTag | j,
                                                memory_order_relaxed, memory_order_relaxed)) {
        rep[j] = kZTag | j;
        zslot[j] = s;
        return;
      }
      v = expected;
    }
    if (v == kZTomb) continue;                        // a dead zone's slot: probe on
    // Occupied: by a zone stored earlier, or by one this batch is interning. Both are already in
    // memory -- the arena from earlier batches, the candidate buffer from the expand kernel.
    const bool mine = (v & kZTag) != 0u;
    const uint hv = mine ? hsh[sval[v & ~kZTag]] : zhash[v];
    if (hv == h && fr_zone_eq(Z, mine ? bz + ulong(sval[v & ~kZTag]) * kNN : zone_at(AP, v))) {
      rep[j] = v;
      return;
    }
  }
  fr_intern_fail(j, ctr_err, rep, zslot);
}
// wlist[r] = the candidate that is the r-th new zone, so zplace runs over new zones only.
kernel void fr_winlist(device const uint* win [[buffer(0)]], device const uint* wpre [[buffer(1)]],
                       device uint* wlist [[buffer(2)]], constant uint& W [[buffer(3)]],
                       uint j [[thread_position_in_grid]]) {
  if (j < W && win[j]) wlist[wpre[j]] = j;
}
// win[j] = 1 iff candidate j claimed a table slot, i.e. is the first copy of its zone.
kernel void fr_zwin(device const uint* keep [[buffer(0)]], device const uint* ctr [[buffer(1)]],
                    device const uint* rep [[buffer(2)]], device uint* win [[buffer(3)]],
                    constant uint& W [[buffer(4)]], uint j [[thread_position_in_grid]]) {
  if (j >= W) return;
  win[j] = (j < ctr[0] && keep[j] && rep[j] == (kZTag | j)) ? 1u : 0u;
}
// One thread per stored entry of a new zone (rank r = gid / kNN, candidate wlist[r]). Its arena
// slot is a recycled one while the free list lasts (taken from the top, rank r getting
// freel[nfree - 1 - r]), else the next fresh one; entry 0 also records the zone's hash and table
// slot and resolves the table's tag.
struct ZPlaceArgs { uint W; uint zcount; uint nfree; };
kernel void fr_zplace(device const uint* wlist [[buffer(0)]], device const uint* wpre [[buffer(1)]],
                      device const uint* sval [[buffer(2)]], device const GZ* bz [[buffer(3)]],
                      device const ArenaPages& AP [[buffer(4)]], device const uint* hsh [[buffer(5)]],
                      device uint* zhash [[buffer(6)]], device atomic_uint* ztab [[buffer(7)]],
                      device const uint* zslot [[buffer(8)]], constant ZPlaceArgs& A [[buffer(9)]],
                      device const uint* freel [[buffer(10)]], device uint* zid [[buffer(11)]],
                      device uint* ztslot [[buffer(12)]], device uint* zcnt [[buffer(13)]],
                      uint gid [[thread_position_in_grid]]) {
  const uint r = gid / uint(kNN), e = gid - r * uint(kNN);
  if (r >= wpre[A.W]) return;
  const uint j = wlist[r];
  const uint z = (r < A.nfree) ? freel[A.nfree - 1 - r] : A.zcount + (r - A.nfree);
  zone_at(AP, z)[e] = bz[ulong(sval[j]) * kNN + e];
  if (e == 0) {
    zid[j] = z;
    zhash[z] = hsh[sval[j]];
    ztslot[z] = zslot[j];
    zcnt[z] = 0u;                                     // place_new counts the entries naming it
    if (zslot[j] != kNoSlot) atomic_store_explicit(&ztab[zslot[j]], z, memory_order_relaxed);
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
                         device const uint* rep [[buffer(10)]], device const uint* zid [[buffer(11)]],
                         device uint* zref [[buffer(12)]], device atomic_uint* zcnt [[buffer(13)]],
                         uint j [[thread_position_in_grid]]) {
  if (j >= A.W || j >= ctr[0] || !keep[j]) return;
  const uint r = kpre[j], a = A.acount + r;
  const uint v = rep[j];
  const uint z = (v & kZTag) ? zid[v & ~kZTag] : v;
  zref[a] = z;
  atomic_fetch_add_explicit(&zcnt[z], 1u, memory_order_relaxed);
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
                        constant InternArgs& A [[buffer(2)]], device const uint* zcnt [[buffer(3)]],
                        device uint* ztslot [[buffer(4)]], device atomic_uint* ctr_err [[buffer(5)]],
                        uint z [[thread_position_in_grid]]) {
  if (z >= A.W || zcnt[z] == 0u) return;              // a free or queued slot holds no zone
  uint s = zhash[z] & A.mask;
  for (uint probe = 0; probe <= A.mask; ++probe, s = (s + 1) & A.mask) {
    uint expected = kZEmpty;
    for (uint tries = 0; tries < FR_CAS_TRIES; ++tries) {
      if (atomic_compare_exchange_weak_explicit(&ztab[s], &expected, z, memory_order_relaxed,
                                                memory_order_relaxed)) { ztslot[z] = s; return; }
      if (expected != kZEmpty) break;               // taken: next slot
    }
  }
  ztslot[z] = kNoSlot;                                // a broken invariant: see kNoSlot
  atomic_fetch_add_explicit(ctr_err, 1u, memory_order_relaxed);
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
// Entry compaction, with the delta folded into the base so that the base holds every live entry:
// entries are renumbered in base order and keep their zones; `remap` records where each went.
kernel void fr_gc_entries(device uint* base_a [[buffer(0)]], device const uint* zref [[buffer(1)]],
                          device uint* zref2 [[buffer(2)]], device uint* remap [[buffer(3)]],
                          device uint* alive2 [[buffer(4)]], constant uint& n [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const uint e = base_a[i];
  zref2[i] = zref[e];
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
