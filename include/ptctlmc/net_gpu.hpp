// net_gpu.hpp — GPU successor generation for networks of timed automata.
//
// CPU-safe interface (no GPU types leak), so the CPU translation unit netreach.cpp can
// include it; the Metal implementation lives in src/metal/netengine_metal.mm and is only
// built when PTCTLMC_HAVE_METAL is defined.
//
// STAGE 1 scope: non-diagonal, channel-free networks (shared int variables only). Each
// (frontier-state, process, location-edge) is expanded on the GPU running the exact
// successors_net pipeline (int guard -> clock guard -> resets -> combined invariant ->
// delay-if-not-frozen -> combined invariant -> reset_inactive -> emptiness). The host still
// performs the aLU-subsumption + convex-union merge (the correctness oracle for stage 1a).
// The factory returns nullptr for anything out of scope (channels / diagonal constraints /
// no Metal device), so the caller cleanly falls back to the CPU path.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "ptctlmc/dbm_ops.hpp"  // raw_t

namespace ptctlmc {

struct Network;

// Storage type of the two biggest device zone buffers -- the frontier and the expand output.
// Both are only ever read or written a whole matrix at a time through a per-entry loop, so a
// narrow entry costs nothing there and halves the allocation (and the expand kernel's store
// traffic). The zone ALGEBRA is untouched: kernels and the CPU merge still work on raw_t.
// kGzInf is the sentinel for raw_t's kInf; a backend only narrows when the model's constants
// make overflow impossible (see GpuZoneIntern in netreach.cpp, same bound).
using gz_t = std::int16_t;
constexpr int kGzInf = 0x7ffe;
inline raw_t gz_load(gz_t v) { return (static_cast<int>(v) == kGzInf) ? kInf : static_cast<raw_t>(v); }
inline gz_t  gz_store(raw_t v) { return (v >= kInf) ? static_cast<gz_t>(kGzInf) : static_cast<gz_t>(v); }

class NetGpuExpander {
 public:
  virtual ~NetGpuExpander() = default;

  // Expand `nstates` frontier states. Inputs are row-major:
  //   locs[s*nprocs + p], ints[s*nintvars + v], zones[s*(dim*dim) + k].
  // An EMPTY `zones` means the frontier buffer was already filled through stage_frontier().
  // Returns the number of work items W. Outputs point INTO the expander's persistent buffers
  // (valid until the next expand/merge call): work item w is a successor iff out_valid[w],
  // with discrete state out_locs[w*nprocs..], out_ints[w*nintvars..] (int32) and zone
  // out_zones[w*(dim*dim)..] — read in place, no compaction copy.
  // `out_dk` is the successor's packed discrete key, computed on the device from the same
  // `dshift`/`intlo` layout the host uses (set_key_layout). A backend that does not compute it
  // leaves it null and the caller falls back to out_locs/out_ints.
  virtual std::size_t expand(int nstates,
                             const std::vector<int>& locs,
                             const std::vector<long>& ints,
                             const std::vector<raw_t>& zones,
                             const int*& out_valid, const int*& out_locs,
                             const int*& out_ints, const gz_t*& out_zones,
                             const std::uint64_t*& out_dk) = 0;

  // Tell the backend how the driver packs a discrete state into 64 bits, so the expand kernel can
  // emit the key itself. `dshift` is nprocs+nintvars bit offsets; `intlo` is per-int-var low bound.
  virtual void set_key_layout(const std::vector<int>& dshift, const std::vector<int>& intlo) {}

  // Stage the frontier-zone buffer for `nstates` states and hand back a host-writable pointer
  // into it (unified memory). The caller fills it and then passes an EMPTY `zones` to expand(),
  // which takes that as "already staged". The frontier itself is arena indices, so this is where
  // its zones are widened -- writing them straight here saves a whole pass over the batch.
  virtual void stage_frontier(std::size_t nstates, int d2, gz_t*& zones) = 0;


  // Merge this level's `nb` discrete-state buckets on the GPU (aLU-subsumption + convex-union,
  // exactly as the CPU add_to), one GPU thread per bucket. For bucket b:
  //   existing antichain: exist_zones[exist_off[b]*d2 .. +exist_cnt[b]*d2)  (d2 = dim*dim)
  //   new zones in order: new_zones[new_off[b]*d2 .. +new_cnt[b]*d2)
  //   aLU bounds:         L[b*dim .. b*dim+dim), U[...]  (has_diag => plain inclusion instead)
  // Outputs the final antichain per bucket: out_zones with out_off[b] the zone-unit start,
  // and out_isnew[i] = 1 for zones added THIS level (i.e. the next frontier).
  // Stage the merge work buffer (persistent, unified memory) and hand back a pointer the caller
  // fills DIRECTLY (no host vectors, no upload copy): for bucket b, write its EXISTING antichain
  // to work_buf[work_off[b]*d2 ..] (exist_cnt[b] zones). `work_zones` = total (exist+new)
  // work-pool capacity. New zones are NOT staged — the merge reads them in place from the expand
  // output by work-item index (see `new_idx` below), so successor zones are never copied.
  // `isnew` is the per-work-pool-slot "added during the current level" flag array; the caller
  // seeds it for the EXISTING zones (a zone an earlier batch of this level added is still new)
  // and the merge preserves it through compaction, flagging appended zones with 1.
  virtual void stage_merge(std::size_t work_zones, int d2, gz_t*& work_buf, int*& isnew) = 0;

  // Per-bucket subsumption + convex-union merge. Existing zones come from the staged work buffer;
  // NEW zones are read in place from the previous expand's output at the given work-item indices:
  // bucket b's new zones are expand items new_idx[new_off[b] .. +new_cnt[b]]. Outputs point INTO
  // the persistent work buffer (valid until the next expand/merge call): bucket b's final
  // antichain is out_zones[(out_work_off[b]+i)*d2 ..] for i in [0, out_count[b]), with
  // out_isnew[out_work_off[b]+i] flagging survivors — read straight from unified memory.
  virtual void merge(int nb, int dim, bool has_diag,
                     const std::vector<int>& work_off, const std::vector<int>& exist_cnt,
                     const std::vector<int>& new_idx, const std::vector<int>& new_off,
                     const std::vector<int>& new_cnt,
                     const std::vector<std::int32_t>& L, const std::vector<std::int32_t>& U,
                     std::vector<int>& out_work_off, std::vector<int>& out_count,
                     const gz_t*& out_zones, const int*& out_isnew) = 0;

  // CPU-EXPAND / GPU-MERGE hybrid. Measurements say the two phases want different
  // hardware: successor generation is branchy, serially-dependent small-matrix work
  // (the CPU wins), while the per-bucket subsumption/convex-union merge is thousands
  // of independent bucket jobs (the GPU wins — 3x over 12 OpenMP threads on Fischer).
  // On unified memory we can have both: this stages the SAME output buffers the GPU
  // expand would have produced and hands back host-writable pointers, so the CPU
  // expander fills them in place and the existing merge path runs on the GPU with
  // ZERO copies. Capacity is `nwork` work items.
  //
  // The caller cannot know the successor count before expanding, so it stages an
  // ESTIMATED capacity, fills it, and — on the rare level where the estimate was too
  // small — calls again with the true size and `preserve=true`, which keeps the bytes
  // already written so only the overflow tail has to be appended.
  virtual void stage_expand(std::size_t nwork, int nprocs, int nintvars, int d2,
                            int*& valid, int*& locs, int*& ints, gz_t*& zones,
                            bool preserve = false) = 0;

};

// Build a Metal / CUDA expander for `net`. `qmask[c]` marks clocks the query reads (kept active
// by reset_inactive). Returns nullptr if the network is out of scope (diagonal constraints) or no
// GPU is available — the caller then falls back to the CPU path. Both share the flat driver in
// netreach.cpp; only the device implementation (netengine_metal.mm / netengine_cuda.cu) differs.
std::unique_ptr<NetGpuExpander> make_net_gpu_metal(const Network& net,
                                                   const std::vector<char>& qmask);
std::unique_ptr<NetGpuExpander> make_net_gpu_cuda(const Network& net,
                                                  const std::vector<char>& qmask);

}  // namespace ptctlmc
