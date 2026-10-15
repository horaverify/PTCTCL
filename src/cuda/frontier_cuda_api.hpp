/// @file src/cuda/frontier_cuda_api.hpp
/// @brief The boundary between the CUDA engine's host code (frontier_cuda.cpp, plain C++) and its
///        kernels (frontier_kernels.cu, nvcc): every kernel as a launcher over plain pointers.
///
/// The split keeps the engine's headers (the model, HDBM) away from nvcc, and the kernels free of
/// them. A launcher takes the stored-entry width in bytes (1, 2 or 4) and picks the matching
/// instantiation; "sgs" is the number of warps a block runs, each warp owning one item.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace hora {
namespace cuda {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Pages = unsigned char* const*;   // device array of the arena pages' addresses

/// Model-wide constants, installed once per engine into the device's constant memory.
struct Dims { int n, nn, np, page_shift, merge_scan, gmode; };   // gmode: G-simulation covers, raw bounds
void set_dims(const Dims& d);

/// Warps per block for a kernel giving each warp @p bytes_per_warp of shared memory.
u32 max_sgs(std::size_t bytes_per_warp);
/// Opt every kernel in for the largest shared-memory carve-out it may need.
void init_kernels(std::size_t zone_bytes_per_warp, std::size_t union_bytes_per_warp);

struct ExpandP {
  const u32 *wf, *wr, *fz;
  Pages AP;
  const u64* rkey;
  const u32 *rflags, *rg_off;
  const int* rg;
  const u32* rr_off;
  const int* rr;
  const u32* rv_off;
  const int* rv;
  const u64* rdead;
  const int* tdbm;
  void* bz;
  u64* skey;
  u32* sval;
  u32* ctr;
  const int *Lx, *Ux;
  const u32* zref;
  const u32* alive;        // entry liveness: a frontier zone that died this round is not expanded
  u32* hshw;
  u32 nwork, prev_slot;    // prev_slot: the frontier slot of the work item before this batch's first
  int has_tclock, extrap;
  u64 sentinel;
};
void expand(int gz, const ExpandP& p, u32 sgs, cudaStream_t s);

void scan_block(const u32* in, u32* out, u32* sums, u32 n_in, u32 n_out, u32 tile, u32 nblk, cudaStream_t s);
void scan_add(u32* out, const u32* sums, u32 n_out, u32 tile, cudaStream_t s);

/// Stable sort of (key, value) by the low @p bits bits of the key; returns the temp bytes needed
/// when @p tmp is null. Output lands in kout / vout.
std::size_t sort_pairs(void* tmp, std::size_t tmp_bytes, const u64* kin, u64* kout, const u32* vin,
                       u32* vout, u32 n, int bits, cudaStream_t s);

void heads(const u64* sk, u32* head, const u32* ctr, u32 W, u32* keep1, u32* keep, cudaStream_t s);
void buckets(const u64* sk, const u32* head, const u32* hpre, u32* bstart, u64* bkey, u32 W, cudaStream_t s);
void bucket_setup(const u64* bkey, const u32* bstart, const u32* hpre, const u32* ctr, const u64* base_k,
                  const u64* delta_k, void* binfo, u32* rowcnt, u32 W, u32 nbase, u32 ndelta, cudaStream_t s);
void bucket_bounds(const u64* bkey, const u32* hpre, const int* Lloc, const int* Uloc, const int* Lq,
                   const int* Uq, const int* locbase, const int* kshift, const u64* kmask, int* Lb, int* Ub,
                   u32 W, cudaStream_t s);

/// Everything the reduction and union kernels read or write, one struct for all of them.
struct ReduceP {
  const u32 *hpre, *head;
  const void* binfo;       // uint4 pairs
  const int *Lb, *Ub;
  const u32 *base_a, *delta_a;
  Pages AP;
  u32* alive;
  const u32* sval;
  void* bz;
  u32 *keep1, *keep;
  u32* ctr;
  const u32* zref;
  u32 *big1, *big2;
  const u32 *k1pre, *surv, *rowoff;
  u32 *zcnt, *ztab;
  const u32* ztslot;
  u32* pend;
  u32* ulist;
  u32* hshw;
  u32* rowcnt;
  u32* chg;                // per candidate: its zone was replaced by a union hull
  u32 W;
  int solo;
};
void reduce_stored(int gz, const ReduceP& p, cudaStream_t s);
void reduce_stored_big(int gz, const ReduceP& p, u32 sgs, cudaStream_t s);
void survivors(const u32* keep1, const u32* k1pre, u32* surv, const u32* ctr, u32 W, cudaStream_t s);
void keep_init(const u32* keep1, u32* keep, const u32* ctr, u32 W, cudaStream_t s);
void rows2(const ReduceP& p, cudaStream_t s);
void reduce_pairs(int gz, const ReduceP& p, cudaStream_t s);
void reduce_pairs_big(int gz, const ReduceP& p, u32 sgs, cudaStream_t s);
void merge_union(int gz, const ReduceP& p, u32 sgs, cudaStream_t s);
/// The same merge with a block per discrete state, the scan spread over its warps (same result).
void merge_union_par(int gz, const ReduceP& p, u32 sgs, cudaStream_t s);
/// After merge_union: drop every zone of a changed hull's bucket that the hull now covers, so the
/// bucket is an antichain of the covering relation again.
void antichain(int gz, const ReduceP& p, cudaStream_t s);

struct InternP {
  const u32 *keep, *ctr, *sval;
  const void* bz;
  Pages AP;
  const u32 *hsh, *zhash;
  u32 *ztab, *rep, *zslot;
  u32* ctr_err;
  u32 W, mask;
};
void intern(int gz, const InternP& p, cudaStream_t s);
void zwin(const u32* keep, const u32* ctr, const u32* rep, u32* win, u32 W, cudaStream_t s);
void winlist(const u32* win, const u32* wpre, u32* wlist, u32 W, cudaStream_t s);
struct ZPlaceP {
  const u32 *wlist, *wpre, *sval;
  const void* bz;
  Pages AP;
  const u32* hsh;
  u32 *zhash, *ztab;
  const u32* zslot;
  const u32* freel;
  u32 *zid, *ztslot, *zcnt;
  u32 W, zcount, nfree;
};
void zplace(int gz, const ZPlaceP& p, cudaStream_t s);
void place_new(const u32* keep, const u32* kpre, const u64* sk, const u32* ctr, u64* new_k, u32* new_a,
               u32* alive, u64* fnext_k, u32* fnext_a, u32 W, u32 acount, u32 fbase, const u32* rep,
               const u32* zid, u32* zref, u32* zcnt, cudaStream_t s);
void fill_u32(u32* v, u32 n, u32 val, cudaStream_t s);
void zrebuild(const u32* zhash, u32* ztab, u32 W, u32 mask, const u32* zcnt, u32* ztslot, u32* ctr_err,
              cudaStream_t s);
void merge_a(const u64* ak, const u32* aa, const u64* bk, const u32* nbp, const u32* apre, const u32* bpre,
             u64* ok, u32* oa, u32 na, u32 nb_cap, u32 filter, cudaStream_t s);
void merge_b(const u64* bk, const u32* ba, const u64* ak, const u32* nbp, const u32* apre, const u32* bpre,
             u64* ok, u32* oa, u32 na, u32 nb_cap, u32 filter, cudaStream_t s);
void alive_flags(const u32* idx, const u32* alive, u32* flag, u32 n, cudaStream_t s);
void compact_frontier(const u64* ik, const u32* ia, const u32* flag, const u32* pre, u64* ok, u32* oa, u32 n,
                      cudaStream_t s);
void gc_entries(u32* base_a, const u32* zref, u32* zref2, u32* remap, u32* alive2, u32 n, cudaStream_t s);
void remap(u32* a, const u32* remap, u32 n, cudaStream_t s);
/// Copy the stored zones of entries idx[0..n) -- through zref -- to out, nn entries of gz bytes each.
void gather_zones(int gz, Pages AP, const u32* zref, const u32* idx, u32 n, void* out, cudaStream_t s);
void summary(u32* ctr, const u32* kpre, const u32* wpre, const u32* rowoff, const u32* hpre, u32 W,
             cudaStream_t s);

}  // namespace cuda
}  // namespace hora
