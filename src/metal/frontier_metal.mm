// frontier_metal.mm — the frontier-batched reachability engine on Metal.
//
// Each BFS level, the frontier is cut into chunks, and each chunk runs as one command buffer:
//
//   expand     one SIMD-group per (frontier zone, transition): the successor zone, the target
//              test, a sort key -- the successor's discrete key, or a sentinel when disabled
//   sort       stable LSD radix sort of (key, work item)
//   group      buckets of equal keys, and each bucket's stored zones in the passed set
//   reduce     candidates against stored zones, then all pairs among the survivors
//   intern     the survivors' zones into the arena, each distinct zone stored once
//   place      a passed-set entry per survivor, merged into the delta
//
// Sizes that depend on the batch's own results are read on the device (indirect dispatch), so a
// batch never stops for the host; the host enumerates the next chunk's transitions meanwhile.
//
// The passed set is a sorted *base* of (key, entry) plus a sorted *delta* receiving each batch's
// survivors, folded into the base once it outgrows a fraction of it. An entry names its zone
// through zref; zones live in a paged arena, are reference-counted, and a zone no live entry names
// is recycled. All buffers are shared memory: the host reads results where the GPU wrote them.
#ifdef HORA_HAVE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef HORA_HAVE_OPENMP
#include <omp.h>
#endif

#include "hdbm/metal/gdbm.hpp"
#include "hora/detail/engine_config.hpp"
#include "hora/detail/frontier_gpu.hpp"
#include "hora/detail/semantics.hpp"
#include "hora/zone.hpp"
#include "src/metal/frontier_kernels.hpp"

namespace hora {
namespace detail {
namespace {

constexpr int kMaxDim = 64;                   // clocks <= 63: one bit each in a 64-bit mask
constexpr std::uint32_t kNoHit = UINT32_MAX;
constexpr std::size_t kMaxPages = 4096;       // must match the kernels' FR_MAX_PAGES
constexpr std::size_t kPageBytes = std::size_t(256) << 20;
// A level is cut into about this many chunks, of at least kMinChunkSlots frontier zones each, so
// the host's enumeration of one chunk overlaps the GPU's work on the previous one.
constexpr std::size_t kChunks = 4;
constexpr std::size_t kMinChunkSlots = 16384;
// Stored zones take the narrowest entry that the model's largest constant M suggests: a byte
// while M is small, 16 bits below kShortMaxConst, else 32. Exact zones are not bounded by M in
// general, so this is a guess, not a proof: the expand kernel flags any zone that does not fit,
// and the search restarts one width up (Overflow below) rather than truncating it.
constexpr std::int32_t kByteMaxConst = 30;
constexpr std::int32_t kShortMaxConst = 4000;
struct Overflow {};   // a zone did not fit the store's entries

// The GPU search cannot continue within its memory budget. Reported to the user as an error, the
// way the benchmarks record a MEMOUT: the engine never asks the system for more than the budget.
struct OutOfMemory : std::runtime_error { using std::runtime_error::runtime_error; };

// Counters the kernels keep, in ctr_.
enum Ctr { kValid = 0, kHit = 1, kNewKeys = 2, kKilled = 3, kWide = 4, kFreed = 5, kBig1 = 6, kBig2 = 7,
           kUnion = 8, kErr = 9, kExpanded = 10, kMergeCap = 11, kLaneSplit = 13 };

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

#define FR_CHECK(c, msg) do { if (!(c)) throw std::runtime_error(std::string("Metal frontier: ") + (msg)); } while (0)

// A growable shared-memory buffer.
struct Buf {
  id<MTLBuffer> b = nil;
  template <class T> T* as() const { return static_cast<T*>(b.contents); }
  std::size_t bytes() const { return b ? static_cast<std::size_t>(b.length) : 0; }
};

class FrontierMetal {
 public:
  // `min_bits`: the narrowest stored-entry width to consider (8, 16 or 32).
  FrontierMetal(const FrontierProblem& p, int min_bits) : P_(p), net_(*p.net), min_bits_(min_bits) {}
  ~FrontierMetal() { if (profile_) report_profile(); }
  int bits() const { return gz_ * 8; }

  int run() {
    if (!setup()) return -1;
    return search();
  }

 private:
  // ---------------------------------------------------------------- set-up
  bool setup() {
    n_ = net_.nclocks + 1;
    nn_ = n_ * n_;
    np_ = net_.nprocs();
    if (n_ > kMaxDim || np_ < 1) return false;
    if (!net_.active.empty() && net_.active_mask.empty()) return false;
    dev_ = MTLCreateSystemDefaultDevice();
    if (!dev_) return false;
    q_ = [dev_ newCommandQueue];
    // Memory budget: everything this engine allocates -- zone store, passed set, frontier, batch
    // buffers, transition tables -- is counted against it (see alloc), and the engine never asks
    // for more. It is Metal's recommended working set for this device, and at most three quarters
    // of RAM: GPU buffers are the machine's own memory here. HORA_GPU_MEM_GB lowers it.
    const std::size_t ram = static_cast<std::size_t>([NSProcessInfo processInfo].physicalMemory);
    mem_cap_ = std::min<std::size_t>(static_cast<std::size_t>(dev_.recommendedMaxWorkingSetSize), ram / 4 * 3);
    if (const char* m = std::getenv("HORA_GPU_MEM_GB"))
      mem_cap_ = std::min<std::size_t>(mem_cap_, static_cast<std::size_t>(std::atof(m) * double(1ull << 30)));

    // Largest constant anywhere: decides the stored entry width.
    std::int32_t maxc = 0;
    auto see = [&](std::int32_t v) { if (v != kAluNegInf) maxc = std::max(maxc, std::abs(v)); };
    for (const auto& pl : P_.Lloc) for (const auto& l : pl) for (auto v : l) see(v);
    for (const auto& pl : P_.Uloc) for (const auto& l : pl) for (auto v : l) see(v);
    for (auto v : P_.Lq) see(v);
    for (auto v : P_.Uq) see(v);
    for (const auto& pr : net_.procs) {
      for (const auto& l : pr.locations) for (const auto& c : l.invariant) see(raw_bound(c.bound));
      for (const auto& e : pr.edges) for (const auto& c : e.clock_guard) see(raw_bound(c.bound));
    }
    for (const auto& c : P_.target->clock_cond) see(raw_bound(c.bound));
    int bits = maxc <= kByteMaxConst ? 8 : maxc < kShortMaxConst ? 16 : 32;
    if (const char* z = std::getenv("HORA_GPU_ZONES")) bits = std::atoi(z);   // 8 | 16 | 32
    bits = std::max(bits, min_bits_);
    gz_ = bits == 8 ? 1 : bits == 16 ? 2 : 4;
    // Arena pages of about kPageBytes, a power of two zones each.
    page_shift_ = 10;
    while (page_shift_ < 24 && (std::size_t(2) << page_shift_) * nn_ * gz_ <= kPageBytes) ++page_shift_;
    page_zones_ = std::size_t(1) << page_shift_;

    // One SIMD-group of the union kernel holds a zone and its hull on chip; past this the model is
    // too wide for the GPU engine (dim <= 64 always fits 32 KB).
    if (2 * nn_ * 4 > 32768) return false;

    // The reduction tests coverage a pair per lane: a row alone where its bucket is small, the
    // whole SIMD-group on one row otherwise, every lane against a different zone (FR_SOLO is the
    // cut-over). Measured alternatives, kept out:
    //   * a pair per SIMD-group (GDBM's warp form, 32 entries a step): slower at every dimension,
    //     hddi15's 47 included (14.8 s against 7.9 s) -- the a_LU test usually finds its witness
    //     within a few entries, so the group mostly votes on chunks it did not need;
    //   * a tiled reduction staging RT x CT blocks of the comparison matrix in threadgroup memory
    //     (64x64 down to 8x32): 1.5x slower on grc6, 4x on csma10. The GPU's caches already give
    //     the reuse the tiles were for, and the tiles cost barriers and idle SIMD-groups.
    solo_ = 8;
    // Convex union is on, as on the CPU; HORA_GPU_UNION=0 turns it off, for measuring.
    union_ = !(std::getenv("HORA_GPU_UNION") && std::string(std::getenv("HORA_GPU_UNION")) == "0");
    if (const char* v = std::getenv("HORA_GPU_SOLO")) solo_ = std::atoi(v);
    bound_bytes_ = maxc < 32767 ? 2 : 4;   // the a_LU bounds per bucket, stored narrow
    NSError* err = nil;
    MTLCompileOptions* o = [[MTLCompileOptions alloc] init];
    o.preprocessorMacros = @{
      @"PT_DIM" : @(n_), @"PT_NP" : @(np_),
      @"GZ" : gz_ == 1 ? @"char" : gz_ == 2 ? @"short" : @"int",
      @"FR_SOLO" : @(solo_), @"FR_PAGE_SHIFT" : @(page_shift_),
      @"FR_MERGE_SCAN" : @(int(detail::kMergeScan)),
      @"FR_HULL_COVER" : @(std::getenv("HORA_GPU_HULL_COVER") ? std::atoi(std::getenv("HORA_GPU_HULL_COVER")) : 0),
      @"FR_BOUND" : bound_bytes_ == 2 ? @"short" : @"int",
    };
    std::string src = std::string(hdbm::metal::gdbm_source()) + metal::frontier_kernels_source();
    id<MTLLibrary> lib = [dev_ newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                            options:o error:&err];
    FR_CHECK(lib != nil, std::string("kernel compile failed: ") +
                             (err ? err.localizedDescription.UTF8String : "?"));
    for (const char* name : {"fr_expand", "fr_scan_block", "fr_scan_add", "fr_rs_hist", "fr_rs_scatter",
                             "fr_heads", "fr_buckets", "fr_bucket_setup", "fr_bucket_bounds", "fr_reduce_stored", "fr_reduce_stored_big", "fr_survivors",
                             "fr_keep_init", "fr_rows2", "fr_reduce_pairs", "fr_reduce_pairs_big", "fr_union", "fr_intern",
                             "fr_zwin", "fr_winlist", "fr_zplace", "fr_place_new", "fr_merge_a", "fr_merge_b",
                             "fr_alive_flags", "fr_compact_frontier", "fr_fill_u32", "fr_zrebuild",
                             "fr_gc_entries", "fr_remap", "fr_indirect"}) {
      id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
      FR_CHECK(f != nil, std::string("missing kernel ") + name);
      id<MTLComputePipelineState> ps = [dev_ newComputePipelineStateWithFunction:f error:&err];
      FR_CHECK(ps != nil, std::string("pipeline failed: ") + name);
      // The kernels are written for 32-lane SIMD-groups, which every Apple GPU has.
      FR_CHECK(ps.threadExecutionWidth == 32 && ps.maxTotalThreadsPerThreadgroup >= 32,
               std::string("unexpected SIMD width or thread limit for ") + name);
      pso_[name] = ps;
    }
    // Launch sizes come from each pipeline's own limits (launch_tpt / launch_sgs), never a
    // constant: a pipeline's thread limit shrinks with the registers its kernel uses, and a
    // launch above it is undefined behaviour in the driver. The sort and the scan need a power of
    // two threads; the sort's tile also has to fit threadgroup memory (96 bytes a thread).
    sort_t_ = std::min(launch_pow2("fr_rs_hist"), launch_pow2("fr_rs_scatter"));
    {
      const NSUInteger room = dev_.maxThreadgroupMemoryLength -
                              pso_.at("fr_rs_scatter").staticThreadgroupMemoryLength;
      while (sort_t_ > 32 && 96 * sort_t_ > room) sort_t_ /= 2;
    }
    scan_t_ = launch_pow2("fr_scan_block");

    // Static model tables.
    std::vector<int> locbase(np_ + 1, 0);
    for (int p = 0; p < np_; ++p) locbase[p + 1] = locbase[p] + int(net_.procs[p].locations.size());
    const int nloc = locbase[np_];
    std::vector<int> Lloc(std::size_t(nloc) * n_, kAluNegInf), Uloc(std::size_t(nloc) * n_, kAluNegInf);
    for (int p = 0; p < np_; ++p)
      for (int l = 0; l < int(net_.procs[p].locations.size()); ++l) {
        const int gl = locbase[p] + l;
        for (int c = 0; c < n_; ++c) {
          if (!P_.Lloc.empty()) Lloc[std::size_t(gl) * n_ + c] = P_.Lloc[p][l][c];
          if (!P_.Uloc.empty()) Uloc[std::size_t(gl) * n_ + c] = P_.Uloc[p][l][c];
        }
      }
    std::vector<int> kshift(P_.keys.shift.begin(), P_.keys.shift.end());
    std::vector<std::uint64_t> kmask(P_.keys.mask.begin(), P_.keys.mask.end());
    std::vector<int> Lq(P_.Lq.begin(), P_.Lq.end()), Uq(P_.Uq.begin(), P_.Uq.end());

    // The target's clock part as a closed DBM; an unsatisfiable one is an empty DBM, which the
    // intersection test then never meets.
    has_tclock_ = !P_.target->clock_cond.empty();
    std::vector<int> tdbm(nn_, 0);
    {
      Dbm t = Dbm::universe(net_.nclocks);
      if (!apply_all(t, P_.target->clock_cond)) { for (int i = 0; i < n_; ++i) tdbm[i * n_ + i] = -1; }
      else std::copy(t.data().begin(), t.data().end(), tdbm.begin());
    }
    put(locbase_, locbase);
    put(Lloc_, Lloc); put(Uloc_, Uloc); put(Lq_, Lq); put(Uq_, Uq);
    put(kshift_, kshift); put(kmask_, kmask); put(tdbm_, tdbm);
    put(Lx_, std::vector<int>(P_.Lx.begin(), P_.Lx.end()));
    put(Ux_, std::vector<int>(P_.Ux.begin(), P_.Ux.end()));
    reserve(ptab_, (kMaxPages + 1) * 8, false);   // [0] = pages allocated, then their addresses
    std::memset(ptab_.b.contents, 0, (kMaxPages + 1) * 8);

    for (int c = 1; c <= net_.nclocks; ++c) {
      clockbits_ |= 1ull << c;
      if (c < int(P_.qmask.size()) && P_.qmask[c]) qbits_ |= 1ull << c;
    }
    key_bits_ = P_.key_bits;
    sentinel_ = 1ull << key_bits_;
    passes_ = (key_bits_ + 1 + 7) / 8;

    // Batch budget: the per-work-item buffers of one batch (candidate zone, two sort buffers,
    // grouping and reduction arrays, the work table) take at most a quarter of the memory budget,
    // and at most 2 GB; the passed set and zone store have the rest.
    const std::size_t budget = std::min<std::size_t>(mem_cap_ / 4, std::size_t(2) << 30);
    const std::size_t per_item = std::size_t(nn_) * gz_ + std::size_t(2 * n_) * bound_bytes_ + 140;
    wmax_ = std::max<std::size_t>(1024, budget / per_item);
    wmax_ = std::min<std::size_t>(wmax_, std::size_t(UINT32_MAX / 2));
#ifdef HORA_HAVE_OPENMP
    nthreads_ = omp_get_max_threads();
#endif
    // One sums buffer per scan level, sized once: scan() recurses with references into this
    // vector, so it must never reallocate. Eight levels cover any length (a block is >= 128).
    scan_sums_.resize(8);
    trace_ = std::getenv("HORA_GPU_TRACE") != nullptr;
    if (const char* s = std::getenv("HORA_GPU_WAIT_S")) wait_limit_ms_ = std::atof(s) * 1000;
    check_ = std::getenv("HORA_GPU_CHECK") != nullptr;
    // HORA_GPU_PROFILE: every kernel in its own command buffer, timed; a table at the end.
    profile_ = std::getenv("HORA_GPU_PROFILE") != nullptr;
    // HORA_GPU_STRESS: drive every rarely-taken path on every level -- sub-batches, many chunks,
    // a fold after every batch, entry compaction and a table rebuild whenever possible -- so the
    // small models of the differential test exercise them too. Verdicts must not change.
    stress_ = std::getenv("HORA_GPU_STRESS") != nullptr;
    if (stress_) wmax_ = 256;
    if (trace_)
      std::fprintf(stderr, "[gpu] %s | dim %d, %d procs, key %d bits, %s zones (%zu a page), "
                   "batch <= %zu items, memory budget %.1f GB, sort/scan %lu/%lu threads\n",
                   dev_.name.UTF8String, n_, np_, key_bits_,
                   gz_ == 1 ? "8-bit" : gz_ == 2 ? "16-bit" : "32-bit", page_zones_, wmax_,
                   double(mem_cap_) / double(1ull << 30), (unsigned long)sort_t_, (unsigned long)scan_t_);
    return true;
  }

  // ---------------------------------------------------------------- search
  // The recipe and work tables of one chunk of a level. Two of them alternate, so the host can
  // enumerate chunk k+1 into one while the GPU reads chunk k from the other.
  struct Chunk {
    Buf rkey, rflags, rg_off, rg, rr_off, rr, rv_off, rv, rdead, wf, wr;
    std::size_t W = 0;
  };

  int search() {
    // Initial state: zone 0, entry 0, the only entry of the base and of the frontier.
    const std::uint64_t k0 = P_.keys.encode(P_.s0.locs, P_.s0.ints);
    ensure_zones(1);
    store_zone(0, P_.s0.zone.data().data());
    reserve(zhash_, 1024 * 4, false); zhash_.as<std::uint32_t>()[0] = zone_hash(0);
    reserve(ztslot_, 1024 * 4, false);
    reserve(zcnt_, 1024 * 4, false); zcnt_.as<std::uint32_t>()[0] = 1;
    reserve(zref_, 1024 * 4, false); zref_.as<std::uint32_t>()[0] = 0;
    reserve(freel_, 1024 * 4, false); reserve(pend_, 1024 * 4, false);
    zcount_ = 1;
    reserve(ctr_, 64, false);            // the table build reports into ctr_[kErr]
    std::memset(ctr_.b.contents, 0, 64);
    ensure_table(1024);
    reserve(alive_, 1024 * 4, false); alive_.as<std::uint32_t>()[0] = 1;
    reserve(base_k_, 1024 * 8, false); reserve(base_a_, 1024 * 4, false);
    base_k_.as<std::uint64_t>()[0] = k0; base_a_.as<std::uint32_t>()[0] = 0;
    reserve(delta_k_, 1024 * 8, false); reserve(delta_a_, 1024 * 4, false);
    reserve(fk_, 1024 * 8, false); reserve(fa_, 1024 * 4, false);
    fk_.as<std::uint64_t>()[0] = k0; fa_.as<std::uint32_t>()[0] = 0;
    acount_ = 1; nbase_ = 1; ndelta_ = 0; nf_ = 1;
    reserve(ind_, 12 * 8, false);

    while (nf_ > 0) {
      if (P_.cancel && P_.cancel->load(std::memory_order_relaxed)) return 0;
      // Explored zones are counted as they are expanded (finish_batch): a frontier zone that a
      // zone inserted earlier in its round covers is skipped, not expanded.
      nf_prev_ = nf_;
      t_enum_ = t_wait_ = 0;
      lvl_w_ = lvl_valid_ = lvl_kept_ = lvl_rows_ = lvl_buckets_ = 0;
      nfn_ = 0;
      // Runs of equal keys (each batch's survivors arrive sorted, so equal keys are adjacent),
      // then chunks of whole runs. A level is cut into a few chunks so that enumerating one
      // overlaps the GPU's work on the previous one.
      const std::uint64_t* fk = fk_.as<std::uint64_t>();
      run_of_.resize(nf_);
      runs_.clear();
      for (std::size_t i = 0; i < nf_; ++i) {
        if (i == 0 || fk[i] != fk[i - 1]) runs_.push_back(i);
        run_of_[i] = std::uint32_t(runs_.size() - 1);
      }
      runs_.push_back(nf_);                         // sentinel: runs_[r + 1] ends run r
      const std::size_t nr = runs_.size() - 1;
      run_tid_.resize(nr); run_lbeg_.resize(nr); run_cnt_.resize(nr); run_beg_.resize(nr);
      const std::size_t chunks = stress_ ? 16 : kChunks;
      const std::size_t chunk_slots =
          std::max<std::size_t>(stress_ ? 1 : kMinChunkSlots, (nf_ + chunks - 1) / chunks);
      std::vector<std::size_t> cuts{0};             // chunk c = runs [cuts[c], cuts[c + 1])
      for (std::size_t r = 0; r < nr; ++r)
        if (runs_[r + 1] - runs_[cuts.back()] >= chunk_slots || r + 1 == nr) cuts.push_back(r + 1);
      prepare(cuts[0], cuts[1], chunks_[0]);
      for (std::size_t c = 0; c + 1 < cuts.size(); ++c) {
        Chunk& ch = chunks_[c & 1];
        const bool more = c + 2 < cuts.size();
        // Work beyond the batch budget runs as consecutive sub-batches of the same chunk.
        for (std::size_t w0 = 0; w0 < ch.W; w0 += wmax_) {
          const std::size_t w1 = std::min(ch.W, w0 + wmax_);
          id<MTLCommandBuffer> cb = encode_batch(ch, w0, w1 - w0);
          if (more && w1 == ch.W) prepare(cuts[c + 1], cuts[c + 2], chunks_[(c + 1) & 1]);
          const auto tw = Clock::now();
          wait(cb);
          t_wait_ += ms_since(tw);
          if (finish_batch(w1 - w0)) return 1;
        }
        if (more && ch.W == 0) prepare(cuts[c + 1], cuts[c + 2], chunks_[(c + 1) & 1]);
        if (P_.cancel && P_.cancel->load(std::memory_order_relaxed)) return 0;
      }
      const auto t_end = Clock::now();
      end_level();
      // The CPU engine's widening switch, on the same signal at the same point: once the passed
      // set averages adapt_ratio zones per discrete state, Extra_LU+ for the rest of the search.
      if (!extrap_ && ndisc_ && double(nlive_) >= P_.adapt_ratio * double(ndisc_)) {
        extrap_ = true;
        if (trace_) std::fprintf(stderr, "[gpu] Extra_LU+ on: %zu zones over %zu discrete states\n",
                                 nlive_, ndisc_);
      }
      if (trace_)
        std::fprintf(stderr, "[gpu] level %4d  frontier %9zu  work %10zu  valid %10zu  kept %9zu  "
                     "entries %10zu zones %10zu (+%zu free) buckets %8zu rows %10zu chunks %zu | "
                     "enum %7.2f wait %7.2f end %6.2f ms\n", level_, nf_prev_, lvl_w_, lvl_valid_,
                     lvl_kept_, nbase_ + ndelta_, zcount_ - nfree_, nfree_, lvl_buckets_, lvl_rows_,
                     cuts.size() - 1, t_enum_, t_wait_, ms_since(t_end));
      ++level_;
    }
    return 0;
  }

  // Enumerate the recipes of runs [ra, rb) of the frontier into `ch`, and its work items: every
  // frontier slot of those runs times every recipe of its key.
  void prepare(std::size_t ra, std::size_t rb, Chunk& ch) {
    const auto t0 = Clock::now();
    const std::uint64_t* fk = fk_.as<std::uint64_t>();
    tables_.resize(nthreads_);
    for (auto& t : tables_) t.clear();
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 16) num_threads(nthreads_)
#endif
    for (long r = long(ra); r < long(rb); ++r) {
      int tid = 0;
#ifdef HORA_HAVE_OPENMP
      tid = omp_get_thread_num();
#endif
      LocVec locs;   // not thread_local: see append_recipes
      IntVec ints;
      P_.keys.decode(fk[runs_[r]], locs, ints);
      RecipeTable& t = tables_[tid];
      const std::size_t before = t.size();
      append_recipes(net_, P_.keys, *P_.target, P_.canon, qbits_, locs, ints, t);
      run_tid_[r] = tid; run_lbeg_[r] = std::uint32_t(before); run_cnt_[r] = std::uint32_t(t.size() - before);
    }
    // Concatenate the per-thread tables straight into the chunk's shared recipe buffers.
    std::vector<std::size_t> rb0(nthreads_ + 1, 0), gb(nthreads_ + 1, 0), xb(nthreads_ + 1, 0),
        vb(nthreads_ + 1, 0);
    for (int t = 0; t < nthreads_; ++t) {
      rb0[t + 1] = rb0[t] + tables_[t].size();
      gb[t + 1] = gb[t] + tables_[t].g.size() / 3;
      xb[t + 1] = xb[t] + tables_[t].r.size();
      vb[t + 1] = vb[t] + tables_[t].v.size() / 3;
    }
    const std::size_t R = rb0[nthreads_];
    reserve(ch.rkey, std::max<std::size_t>(1, R) * 8, false);
    reserve(ch.rflags, std::max<std::size_t>(1, R) * 4, false);
    reserve(ch.rg_off, (R + 1) * 4, false);
    reserve(ch.rr_off, (R + 1) * 4, false);
    reserve(ch.rg, std::max<std::size_t>(1, gb[nthreads_]) * 12, false);
    reserve(ch.rr, std::max<std::size_t>(1, xb[nthreads_]) * 4, false);
    reserve(ch.rv_off, (R + 1) * 4, false);
    reserve(ch.rv, std::max<std::size_t>(1, vb[nthreads_]) * 12, false);
    reserve(ch.rdead, std::max<std::size_t>(1, R) * 8, false);
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel for schedule(static, 1) num_threads(nthreads_)
#endif
    for (int t = 0; t < nthreads_; ++t) {
      const RecipeTable& T = tables_[t];
      if (T.size()) {
        std::memcpy(ch.rkey.as<std::uint64_t>() + rb0[t], T.key.data(), T.size() * 8);
        std::memcpy(ch.rflags.as<std::uint32_t>() + rb0[t], T.flags.data(), T.size() * 4);
        std::memcpy(ch.rdead.as<std::uint64_t>() + rb0[t], T.dead.data(), T.size() * 8);
      }
      if (!T.g.empty()) std::memcpy(ch.rg.as<std::int32_t>() + gb[t] * 3, T.g.data(), T.g.size() * 4);
      if (!T.r.empty()) std::memcpy(ch.rr.as<std::int32_t>() + xb[t], T.r.data(), T.r.size() * 4);
      if (!T.v.empty()) std::memcpy(ch.rv.as<std::int32_t>() + vb[t] * 3, T.v.data(), T.v.size() * 4);
      std::uint32_t* go = ch.rg_off.as<std::uint32_t>();
      std::uint32_t* ro = ch.rr_off.as<std::uint32_t>();
      std::uint32_t* vo = ch.rv_off.as<std::uint32_t>();
      for (std::size_t r = 0; r < T.size(); ++r) {
        go[rb0[t] + r] = std::uint32_t(gb[t] + T.g_off[r]);
        ro[rb0[t] + r] = std::uint32_t(xb[t] + T.r_off[r]);
        vo[rb0[t] + r] = std::uint32_t(vb[t] + T.v_off[r]);
      }
    }
    ch.rg_off.as<std::uint32_t>()[R] = std::uint32_t(gb[nthreads_]);
    ch.rr_off.as<std::uint32_t>()[R] = std::uint32_t(xb[nthreads_]);
    ch.rv_off.as<std::uint32_t>()[R] = std::uint32_t(vb[nthreads_]);
    // Work items, run by run: a prefix sum over the runs, then a parallel fill.
    std::vector<std::size_t> woff(rb - ra + 1, 0);
    for (std::size_t r = ra; r < rb; ++r) {
      run_beg_[r] = std::uint32_t(rb0[run_tid_[r]] + run_lbeg_[r]);
      woff[r - ra + 1] = woff[r - ra] + (runs_[r + 1] - runs_[r]) * run_cnt_[r];
    }
    ch.W = woff.back();
    reserve(ch.wf, std::max<std::size_t>(1, ch.W) * 4, false);
    reserve(ch.wr, std::max<std::size_t>(1, ch.W) * 4, false);
    std::uint32_t* wf = ch.wf.as<std::uint32_t>();
    std::uint32_t* wr = ch.wr.as<std::uint32_t>();
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(nthreads_)
#endif
    for (long r = long(ra); r < long(rb); ++r) {
      std::size_t w = woff[r - ra];
      for (std::size_t f = runs_[r]; f < runs_[r + 1]; ++f)
        for (std::uint32_t t = 0; t < run_cnt_[r]; ++t) { wf[w] = std::uint32_t(f); wr[w] = run_beg_[r] + t; ++w; }
    }
    lvl_w_ += ch.W;
    t_enum_ += ms_since(t0);
  }

  // Encode one batch -- work items [w0, w0 + W) of chunk `ch` -- as a single command buffer and
  // commit it. Every size that depends on the batch's own results (valid candidates, reduction
  // rows) is read on the device: those dispatches are indirect, and the rest are sized by W with
  // the excess threads exiting at once. finish_batch() reads the outcome.
  id<MTLCommandBuffer> encode_batch(Chunk& ch, std::size_t w0, std::size_t W) {
    const std::uint32_t Wu = std::uint32_t(W);
    if (check_) check_before();
    reserve(bz_, W * nn_ * gz_, false);
    for (Buf* b : {&sk0_, &sk1_}) reserve(*b, W * 8, false);
    for (Buf* b : {&sv0_, &sv1_}) reserve(*b, W * 4, false);
    reserve(head_, W * 4, false); reserve(hpre_, (W + 1) * 4, false);
    reserve(bstart_, W * 4, false); reserve(bkey_, W * 8, false); reserve(binfo_, W * 32, false);
    reserve(rowcnt_, W * 4, false); reserve(rowoff_, (W + 1) * 4, false);
    reserve(Lb_, W * n_ * bound_bytes_, false); reserve(Ub_, W * n_ * bound_bytes_, false);
    // big2: phase 2's large rows, of which there is one per survivor and per stored zone of a
    // bucket -- more than W when a bucket holds many stored zones (see ReduceArgs).
    const std::size_t cap2 = W + nbase_ + ndelta_ + 1;
    reserve(big1_, W * 4, false); reserve(big2_, cap2 * 4, false); reserve(ulist_, W * 4, false);
    reserve(keep_, W * 4, false); reserve(keep1_, W * 4, false);
    reserve(k1pre_, (W + 1) * 4, false); reserve(kpre_, (W + 1) * 4, false);
    reserve(surv_, W * 4, false);
    reserve(new_k_, W * 8, false); reserve(new_a_, W * 4, false);
    // The batch may store W new zones and entries: room for them, kept contents and all. The
    // arena grows by pages, nothing copied; the per-zone and per-entry arrays are small.
    ensure_zones(zcount_ + W);
    for (Buf* b : {&zhash_, &ztslot_, &zcnt_, &pend_}) reserve(*b, (zcount_ + W) * 4, true);
    reserve(zref_, (acount_ + W) * 4, true);
    reserve(alive_, (acount_ + W) * 4, true);
    reserve(fnk_, (nfn_ + W) * 8, true);
    reserve(fna_, (nfn_ + W) * 4, true);
    reserve(delta2_k_, (ndelta_ + W) * 8, false);
    reserve(delta2_a_, (ndelta_ + W) * 4, false);
    reserve(hsh_, W * 4, false); reserve(rep_, W * 4, false); reserve(zslot_, W * 4, false);
    reserve(win_, W * 4, false); reserve(wpre_, (W + 1) * 4, false); reserve(zid_, W * 4, false);
    reserve(wlist_, W * 4, false);
    ensure_table(W);
    const std::uint32_t stile = std::uint32_t(4 * sort_t_);
    const std::uint32_t ntiles = std::uint32_t((W + stile - 1) / stile);
    reserve(ghist_, std::size_t(ntiles) * 256 * 4, false);
    std::uint32_t* ctr = ctr_.as<std::uint32_t>();
    ctr[kValid] = 0; ctr[kHit] = kNoHit; ctr[kNewKeys] = 0; ctr[kKilled] = 0; ctr[kWide] = 0;
    ctr[kBig1] = 0; ctr[kBig2] = 0; ctr[kUnion] = 0; ctr[kErr] = 0; ctr[kExpanded] = 0;
    ctr[kMergeCap] = 0; ctr[kLaneSplit] = 0;
    freed_before_ = ctr[kFreed];   // cumulative over the level

    begin();
    // ---- expand ----
    // prev_slot: the frontier slot of the last work item before this batch, so a frontier zone
    // whose transitions straddle two sub-batches is counted as explored once.
    const std::uint32_t prev_slot = w0 > 0 ? ch.wf.as<std::uint32_t>()[w0 - 1] : 0xffffffffu;
    struct { std::uint32_t nwork; std::int32_t has_tclock; std::uint64_t sentinel;
             std::int32_t extrap; std::uint32_t prev_slot; }
        ea{Wu, has_tclock_ ? 1 : 0, sentinel_, extrap_ ? 1 : 0, prev_slot};
    use("fr_expand");
    bind({&ch.wf, &ch.wr, &fa_, &ptab_, &ch.rkey, &ch.rflags, &ch.rg_off, &ch.rg, &ch.rr_off,
          &ch.rr, &ch.rv_off, &ch.rv, &ch.rdead, &alive_, nullptr, nullptr, &tdbm_, &bz_, &sk0_,
          &sv0_, &ctr_});
    [enc_ setBuffer:ch.wf.b offset:w0 * 4 atIndex:0];
    [enc_ setBuffer:ch.wr.b offset:w0 * 4 atIndex:1];
    [enc_ setBytes:&ea length:sizeof(ea) atIndex:21];
    set(22, Lx_); set(23, Ux_); set(24, zref_); set(25, hsh_);
    dispatch_sg(W, launch_sgs("fr_expand", (nn_ + n_) * 4), (nn_ + n_) * 4);
    // ---- sort by discrete key ----
    Buf* sk = &sk0_; Buf* sv = &sv0_;
    for (int pass = 0; pass < passes_; ++pass) {
      Buf* dk = (sk == &sk0_) ? &sk1_ : &sk0_;
      Buf* dv = (sv == &sv0_) ? &sv1_ : &sv0_;
      struct { std::uint32_t n, ntiles, shift; } ra{Wu, ntiles, std::uint32_t(pass * 8)};
      use("fr_rs_hist");
      bind({sk, &ghist_});
      [enc_ setBytes:&ra length:sizeof(ra) atIndex:2];
      [enc_ dispatchThreadgroups:MTLSizeMake(ntiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(sort_t_, 1, 1)];
      scan(ghist_, ghist_, ntiles * 256, ntiles * 256);
      use("fr_rs_scatter");
      bind({sk, sv, dk, dv, &ghist_});
      [enc_ setBytes:&ra length:sizeof(ra) atIndex:5];
      [enc_ setThreadgroupMemoryLength:2 * stile * 8 atIndex:0];   // keys, twice over
      [enc_ setThreadgroupMemoryLength:2 * stile * 4 atIndex:1];   // values, twice over
      [enc_ dispatchThreadgroups:MTLSizeMake(ntiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(sort_t_, 1, 1)];
      sk = dk; sv = dv;
    }
    chk_sk_ = sk; chk_sv_ = sv;
    // ---- group into buckets, and find each bucket's stored zones ----
    struct { std::uint32_t W, nbase, ndelta; } ga{Wu, std::uint32_t(nbase_), std::uint32_t(ndelta_)};
    use("fr_heads");
    bind({sk, &head_, &ctr_, nullptr, &keep1_, &keep_});
    [enc_ setBytes:&ga length:sizeof(ga) atIndex:3];
    dispatch_threads(W);
    scan(head_, hpre_, Wu, Wu + 1);
    use("fr_buckets");
    bind({sk, &head_, &hpre_, &bstart_, &bkey_});
    [enc_ setBytes:&ga length:sizeof(ga) atIndex:5];
    dispatch_threads(W);
    use("fr_bucket_setup");
    bind({&bkey_, &bstart_, &hpre_, &ctr_, &base_k_, &delta_k_, &binfo_, &rowcnt_});
    [enc_ setBytes:&ga length:sizeof(ga) atIndex:8];
    dispatch_threads(W);
    const std::uint32_t tbb = launch_tpt("fr_bucket_bounds");
    indirect(hpre_, W, std::uint32_t(n_), tbb, 3);
    use("fr_bucket_bounds");
    bind({&bkey_, &hpre_, &Lloc_, &Uloc_, &Lq_, &Uq_, &locbase_, &kshift_, &kmask_, &Lb_, &Ub_});
    [enc_ setBytes:&Wu length:4 atIndex:11];
    dispatch_indirect(3, tbb);
    // ---- reduce, phase 1: each candidate against its bucket's stored zones ----
    struct { std::uint32_t W, cap2; } rd{Wu, std::uint32_t(cap2)};
    const int tg_red = nn_ * 4;
    const std::uint32_t trs = launch_tpt("fr_reduce_stored");
    indirect(ctr_, kValid, 1, trs, 0);
    use("fr_reduce_stored");
    bind({&hpre_, &head_, &binfo_, &Lb_, &base_a_, &delta_a_, &ptab_, &alive_, sv, &bz_, &Ub_,
          &keep1_, &ctr_, &zref_, &big1_, &ctr_});
    dispatch_indirect(0, trs);
    const std::uint32_t sg1 = launch_sgs("fr_reduce_stored_big", tg_red);
    indirect(ctr_, kBig1, 1, sg1, 4);
    use("fr_reduce_stored_big");
    bind({&hpre_, &head_, &binfo_, &Lb_, &base_a_, &delta_a_, &ptab_, &alive_, sv, &bz_, &Ub_,
          &keep1_, &ctr_, &zref_, &big1_});
    dispatch_sg_indirect(4, sg1, tg_red);
    scan(keep1_, k1pre_, Wu, Wu + 1);
    use("fr_survivors");
    bind({&keep1_, &k1pre_, &surv_, &ctr_});
    dispatch_threads(W);
    use("fr_keep_init");
    bind({&keep1_, &keep_, &ctr_});
    dispatch_threads(W);
    use("fr_rows2");
    bind({&binfo_, &hpre_, &k1pre_, &rowcnt_, nullptr, &ctr_, &ulist_});
    [enc_ setBytes:&rd length:sizeof(rd) atIndex:4];
    dispatch_threads(W);
    scan(rowcnt_, rowoff_, Wu, Wu + 1);
    // ---- reduce, phase 2: survivors all-pairs, and stored zones against survivors ----
    const std::uint32_t trp = launch_tpt("fr_reduce_pairs");
    indirect(rowoff_, W, 1, trp, 1);
    use("fr_reduce_pairs");
    bind({&rowoff_, &hpre_, &binfo_, &Lb_, &base_a_, &delta_a_, &ptab_, &alive_, sv, &bz_, &Ub_,
          &keep_, &k1pre_, &surv_, nullptr, &ctr_, &zref_, &zcnt_, &ztab_, &ztslot_, &pend_, &big2_});
    [enc_ setBytes:&rd length:sizeof(rd) atIndex:14];
    dispatch_indirect(1, trp);
    const std::uint32_t sg2 = launch_sgs("fr_reduce_pairs_big", tg_red);
    indirect(ctr_, kBig2, 1, sg2, 4);
    use("fr_reduce_pairs_big");
    bind({&rowoff_, &hpre_, &binfo_, &Lb_, &base_a_, &delta_a_, &ptab_, &alive_, sv, &bz_, &Ub_,
          &keep_, &k1pre_, &surv_, nullptr, &ctr_, &zref_, &zcnt_, &ztab_, &ztslot_, &pend_, &big2_,
          &ctr_});
    [enc_ setBytes:&rd length:sizeof(rd) atIndex:14];
    dispatch_sg_indirect(4, sg2, tg_red);
    // ---- convex union, as the CPU engine merges ----
    if (union_) {
      const int tg_un = 2 * nn_ * 4;   // the zone and its hull
      const std::uint32_t sgu = launch_sgs("fr_union", tg_un);
      indirect(ctr_, kUnion, 1, sgu, 5);
      use("fr_union");
      bind({&ulist_, &binfo_, &base_a_, &delta_a_, &ptab_, &alive_, sv, &bz_, &keep_, &k1pre_, &surv_,
            &zref_, &zcnt_, &ztab_, &ztslot_, &pend_, &ctr_, &hsh_, &Lb_, &Ub_});
      dispatch_sg_indirect(5, sgu, tg_un);
    }
    // ---- intern the survivors' zones ----
    scan(keep_, kpre_, Wu, Wu + 1);
    struct { std::uint32_t W, mask; } ia{Wu, std::uint32_t(zcap_ - 1)};
    use("fr_intern");
    bind({&keep_, &ctr_, sv, &bz_, &ptab_, &hsh_, &zhash_, &ztab_, &rep_, &zslot_});
    [enc_ setBytes:&ia length:sizeof(ia) atIndex:10];
    [enc_ setBuffer:ctr_.b offset:kErr * 4 atIndex:11];
    dispatch_threads(W);
    use("fr_zwin");
    bind({&keep_, &ctr_, &rep_, &win_});
    [enc_ setBytes:&Wu length:4 atIndex:4];
    dispatch_threads(W);
    scan(win_, wpre_, Wu, Wu + 1);
    use("fr_winlist");
    bind({&win_, &wpre_, &wlist_});
    [enc_ setBytes:&Wu length:4 atIndex:3];
    dispatch_threads(W);
    struct { std::uint32_t W, zcount, nfree; } za{Wu, std::uint32_t(zcount_), std::uint32_t(nfree_)};
    const std::uint32_t tzp = launch_tpt("fr_zplace");
    indirect(wpre_, W, std::uint32_t(nn_), tzp, 2);
    use("fr_zplace");
    bind({&wlist_, &wpre_, sv, &bz_, &ptab_, &hsh_, &zhash_, &ztab_, &zslot_, nullptr, &freel_, &zid_,
          &ztslot_, &zcnt_});
    [enc_ setBytes:&za length:sizeof(za) atIndex:9];
    dispatch_indirect(2, tzp);
    // ---- place the survivors, and merge them into the delta ----
    struct { std::uint32_t W, acount, fbase, zcount; }
        pa{Wu, std::uint32_t(acount_), std::uint32_t(nfn_), std::uint32_t(zcount_)};
    use("fr_place_new");
    bind({&keep_, &kpre_, sk, &ctr_, &new_k_, &new_a_, &alive_, &fnk_, &fna_, nullptr, &rep_, &zid_,
          &zref_, &zcnt_});
    [enc_ setBytes:&pa length:sizeof(pa) atIndex:9];
    dispatch_threads(W);
    // delta' = merge(delta, survivors); the survivor count is read on the device (kpre[W]).
    merge(delta_k_, delta_a_, ndelta_, new_k_, new_a_, kpre_, W * 4, W, nullptr, nullptr,
          delta2_k_, delta2_a_);
    return submit();
  }

  // After the batch's command buffer completed: bookkeeping, and true on a target hit.
  bool finish_batch(std::size_t W) {
    const std::uint32_t* ctr = ctr_.as<std::uint32_t>();
    const std::uint32_t nvalid = ctr[kValid], hit = ctr[kHit];
    lvl_valid_ += nvalid;
    if (P_.stats) { P_.stats->generated += nvalid; P_.stats->explored += ctr[kExpanded]; }
    if (hit != kNoHit) return true;
    if (ctr[kWide]) throw Overflow{};
    FR_CHECK(ctr[kErr] == 0, "a GPU capacity bound was exceeded (full zone table, or big2 overflow)");
    FR_CHECK(ctr[kMergeCap] == 0, "convex-union merge loop exceeded its bound");
    FR_CHECK(ctr[kLaneSplit] == 0, "the lanes of a SIMD-group disagreed on a convex-union test");
    const std::uint32_t kept = kpre_.as<std::uint32_t>()[W];
    const std::uint32_t fresh = wpre_.as<std::uint32_t>()[W];   // zones not already stored
    lvl_rows_ += rowoff_.as<std::uint32_t>()[W];
    lvl_buckets_ += hpre_.as<std::uint32_t>()[W];
    lvl_kept_ += kept;
    nlive_ += kept;
    nlive_ -= ctr[kKilled];   // stored entries a survivor strictly covered
    ndisc_ += ctr[kNewKeys];  // discrete states seen for the first time
    ntomb_ += ctr[kFreed] - freed_before_;
    std::swap(delta_k_, delta2_k_); std::swap(delta_a_, delta2_a_);
    ndelta_ += kept;
    acount_ += kept;
    const std::size_t recycled = std::min<std::size_t>(fresh, nfree_);
    nfree_ -= recycled;
    zcount_ += fresh - recycled;
    nfn_ += kept;
    if (check_) check_after();
    if (ndelta_ > std::max<std::size_t>(nbase_ / 4, 1u << 16) || (stress_ && ndelta_)) fold();
    return false;
  }

  // base := merge(live base, live delta); delta := empty.
  void fold() {
    reserve(fl_a_, (nbase_ + 1) * 4, false); reserve(fl_ap_, (nbase_ + 1) * 4, false);
    reserve(fl_b_, (ndelta_ + 1) * 4, false); reserve(fl_bp_, (ndelta_ + 1) * 4, false);
    reserve(base2_k_, (nbase_ + ndelta_) * 8, false); reserve(base2_a_, (nbase_ + ndelta_) * 4, false);
    reserve(nbuf_, 16, false);
    nbuf_.as<std::uint32_t>()[0] = std::uint32_t(ndelta_);
    begin();
    alive_flags(base_a_, nbase_, fl_a_);
    scan(fl_a_, fl_ap_, std::uint32_t(nbase_), std::uint32_t(nbase_ + 1));
    alive_flags(delta_a_, ndelta_, fl_b_);
    scan(fl_b_, fl_bp_, std::uint32_t(ndelta_), std::uint32_t(ndelta_ + 1));
    merge(base_k_, base_a_, nbase_, delta_k_, delta_a_, nbuf_, 0, ndelta_, &fl_ap_, &fl_bp_,
          base2_k_, base2_a_);
    wait(submit());
    const std::size_t n = fl_ap_.as<std::uint32_t>()[nbase_] + fl_bp_.as<std::uint32_t>()[ndelta_];
    std::swap(base_k_, base2_k_); std::swap(base_a_, base2_a_);
    nbase_ = n;
    ndelta_ = 0;
  }

  void end_level() {
    // Next frontier: this level's survivors still alive (a later batch may have covered one).
    reserve(fl_a_, (nfn_ + 1) * 4, false); reserve(fl_ap_, (nfn_ + 1) * 4, false);
    reserve(fk_, std::max<std::size_t>(1, nfn_) * 8, false);
    reserve(fa_, std::max<std::size_t>(1, nfn_) * 4, false);
    if (nfn_ > 0) {
      begin();
      alive_flags(fna_, nfn_, fl_a_);
      scan(fl_a_, fl_ap_, std::uint32_t(nfn_), std::uint32_t(nfn_ + 1));
      struct { std::uint32_t n; } ca{std::uint32_t(nfn_)};
      use("fr_compact_frontier");
      bind({&fnk_, &fna_, &fl_a_, &fl_ap_, &fk_, &fa_});
      [enc_ setBytes:&ca length:sizeof(ca) atIndex:6];
      dispatch_threads(nfn_);
      wait(submit());
    }
    nf_ = nfn_ ? fl_ap_.as<std::uint32_t>()[nfn_] : 0;
    // The zones this level freed become reusable now: every frontier entry that could name one
    // has been expanded.
    std::uint32_t* ctr = ctr_.as<std::uint32_t>();
    const std::size_t npend = ctr[kFreed];
    if (npend) {
      reserve(freel_, (nfree_ + npend) * 4, true);
      std::memcpy(freel_.as<std::uint32_t>() + nfree_, pend_.b.contents, npend * 4);
      nfree_ += npend;
      ctr[kFreed] = 0;
    }
    // Entry compaction once dead entries outnumber live ones by a wide margin.
    if ((acount_ > (std::size_t(1) << 20) && acount_ > 3 * (nbase_ + ndelta_)) ||
        (stress_ && acount_ > nbase_ + ndelta_))
      gc();
  }

  // Entry compaction: entries renumbered in base order; zones stay where they are.
  void gc() {
    fold();
    const std::size_t n = nbase_;
    reserve(zref2_, std::max<std::size_t>(n, 1024) * 4, false);
    reserve(alive2_, std::max<std::size_t>(n, 1024) * 4, false);
    reserve(remap_, acount_ * 4, false);
    const std::uint32_t nu = std::uint32_t(n), fu = std::uint32_t(nf_);
    begin();
    use("fr_gc_entries");
    bind({&base_a_, &zref_, &zref2_, &remap_, &alive2_});
    [enc_ setBytes:&nu length:4 atIndex:5];
    dispatch_threads(n);
    if (nf_) {
      use("fr_remap");
      bind({&fa_, &remap_});
      [enc_ setBytes:&fu length:4 atIndex:2];
      dispatch_threads(nf_);
    }
    wait(submit());
    std::swap(zref_, zref2_); std::swap(alive_, alive2_);
    acount_ = n;
    for (Buf* b : {&zref2_, &alive2_, &remap_}) release(*b);
  }

  // ---------------------------------------------------------------- the zone store
  // Arena pages for `nz` zones; a new page is a new buffer, and its GPU address joins the table.
  void ensure_zones(std::size_t nz) {
    while (pages_.size() * page_zones_ < nz) {
      FR_CHECK(pages_.size() < kMaxPages, "zone arena exceeds its page table");
      const std::size_t bytes = page_zones_ * nn_ * gz_;
      charge_mem(bytes, "a zone page");
      id<MTLBuffer> b = [dev_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      if (!b) throw OutOfMemory("out of GPU memory: the system refused a " + mb(bytes) + " zone page");
      mem_used_ += bytes;
      ptab_.as<std::uint64_t>()[1 + pages_.size()] = b.gpuAddress;
      ptab_.as<std::uint64_t>()[0] = pages_.size() + 1;
      pages_.push_back(b);
    }
  }
  void* zone_ptr(std::size_t z) const {
    return static_cast<char*>(((id<MTLBuffer>)pages_[z >> page_shift_]).contents) +
           (z & (page_zones_ - 1)) * nn_ * gz_;
  }
  // The table holds at most half its capacity in live zones, tombstones and the next batch's
  // zones; past that it is rebuilt -- larger when live zones alone need it.
  void ensure_table(std::size_t W) {
    const std::size_t live = zcount_ - nfree_;
    if (zcap_ && (live + ntomb_ + W) * 2 <= zcap_ && !(stress_ && ntomb_)) return;
    if (!zcap_) zcap_ = 4096;
    while ((live + W) * 2 > zcap_) zcap_ *= 2;
    rebuild_table();
  }
  void rebuild_table() {
    reserve(ztab_, zcap_ * 4, false);
    begin();
    fill(ztab_, zcap_, 0xffffffffu);
    struct { std::uint32_t W, mask; } ia{std::uint32_t(zcount_), std::uint32_t(zcap_ - 1)};
    ctr_.as<std::uint32_t>()[kErr] = 0;
    use("fr_zrebuild");
    bind({&zhash_, &ztab_, nullptr, &zcnt_, &ztslot_});
    [enc_ setBytes:&ia length:sizeof(ia) atIndex:2];
    [enc_ setBuffer:ctr_.b offset:kErr * 4 atIndex:5];
    dispatch_threads(zcount_);
    wait(submit());
    FR_CHECK(ctr_.as<std::uint32_t>()[kErr] == 0, "zone table rebuild found the table full");
    ntomb_ = 0;
  }
  // ---- HORA_GPU_CHECK: a batch may remove a zone only if a zone that stays covers it ----------
  // Before each batch the live entries and their keys are recorded; after it, every entry that died
  // and every candidate that was dropped must be a_LU-covered by a live entry of its discrete state.
  // The first violations are printed with the zones involved, and the search stops.
  void check_before() {
    const std::uint32_t* al = alive_.as<std::uint32_t>();
    chk_acount_ = acount_;
    chk_alive_.assign(al, al + acount_);
    chk_key_.assign(acount_, ~0ull);
    for (std::size_t i = 0; i < nbase_; ++i) chk_key_[base_a_.as<std::uint32_t>()[i]] = base_k_.as<std::uint64_t>()[i];
    for (std::size_t i = 0; i < ndelta_; ++i) chk_key_[delta_a_.as<std::uint32_t>()[i]] = delta_k_.as<std::uint64_t>()[i];
  }
  void key_bounds(std::uint64_t key, std::vector<std::int32_t>& L, std::vector<std::int32_t>& U) const {
    const int* Ll = Lloc_.as<int>(); const int* Ul = Uloc_.as<int>();
    const int* lb = locbase_.as<int>(); const int* ks = kshift_.as<int>();
    const std::uint64_t* km = kmask_.as<std::uint64_t>();
    L.assign(Lq_.as<int>(), Lq_.as<int>() + n_);
    U.assign(Uq_.as<int>(), Uq_.as<int>() + n_);
    for (int p = 0; p < np_; ++p) {
      const int gl = lb[p] + int((key >> ks[p]) & km[p]);
      for (int c = 0; c < n_; ++c) {
        L[c] = std::max(L[c], Ll[std::size_t(gl) * n_ + c]);
        U[c] = std::max(U[c], Ul[std::size_t(gl) * n_ + c]);
      }
    }
    L[0] = U[0] = 0;
  }
  bool alu(const void* z, const void* zp, const std::int32_t* L, const std::int32_t* U) const {
    if (gz_ == 1) return dbm_is_alu_le(static_cast<const sraw_t*>(z), static_cast<const sraw_t*>(zp), n_, L, U);
    if (gz_ == 2) return dbm_is_alu_le(static_cast<const nraw_t*>(z), static_cast<const nraw_t*>(zp), n_, L, U);
    return dbm_is_alu_le(static_cast<const raw_t*>(z), static_cast<const raw_t*>(zp), n_, L, U);
  }
  void print_zone(const char* tag, const void* p) const {
    std::fprintf(stderr, "[gpu]     %s:", tag);
    for (int e = 0; e < nn_; ++e) {
      const int v = gz_ == 1 ? int(static_cast<const std::int8_t*>(p)[e])
                  : gz_ == 2 ? int(static_cast<const std::int16_t*>(p)[e])
                             : static_cast<const std::int32_t*>(p)[e];
      if (e % n_ == 0) std::fprintf(stderr, " |");
      std::fprintf(stderr, " %d", v);
    }
    std::fprintf(stderr, "\n");
  }
  // Is zone z covered by a live entry of `key`? Prints the live entries when `show`.
  bool covered_by_live(const void* z, std::uint64_t key, bool show) const {
    std::vector<std::int32_t> L, U;
    key_bounds(key, L, U);
    const std::uint32_t* al = alive_.as<std::uint32_t>();
    const std::uint32_t* zr = zref_.as<std::uint32_t>();
    bool hit = false;
    for (int side = 0; side < 2 && !hit; ++side) {
      const std::size_t cnt = side ? ndelta_ : nbase_;
      const std::uint64_t* k = (side ? delta_k_ : base_k_).as<std::uint64_t>();
      const std::uint32_t* a = (side ? delta_a_ : base_a_).as<std::uint32_t>();
      for (std::size_t i = 0; i < cnt; ++i) {
        if (k[i] != key || !al[a[i]]) continue;
        if (show) print_zone(side ? "live (delta)" : "live (base)", zone_ptr(zr[a[i]]));
        if (alu(z, zone_ptr(zr[a[i]]), L.data(), U.data())) { hit = true; if (!show) break; }
      }
    }
    return hit;
  }
  void check_after() {
    const std::uint32_t* al = alive_.as<std::uint32_t>();
    const std::uint32_t* zr = zref_.as<std::uint32_t>();
    int bad = 0;
    // Reference counts: every live entry's zone counted once per entry, nothing else.
    {
      std::vector<std::uint32_t> want(zcount_, 0);
      for (int side = 0; side < 2; ++side) {
        const std::size_t cnt = side ? ndelta_ : nbase_;
        const std::uint32_t* a = (side ? delta_a_ : base_a_).as<std::uint32_t>();
        for (std::size_t i = 0; i < cnt; ++i)
          if (al[a[i]]) {
            if (zr[a[i]] >= zcount_) { std::fprintf(stderr, "[gpu] CHECK: entry %u names zone %u >= %zu\n", a[i], zr[a[i]], zcount_); ++bad; }
            else ++want[zr[a[i]]];
          }
      }
      const std::uint32_t* zc = zcnt_.as<std::uint32_t>();
      std::vector<std::uint8_t> isfree(zcount_, 0);
      for (std::size_t i = 0; i < nfree_; ++i) isfree[freel_.as<std::uint32_t>()[i]] = 1;
      const std::uint32_t npend = ctr_.as<std::uint32_t>()[kFreed];
      for (std::uint32_t i = 0; i < npend; ++i) isfree[pend_.as<std::uint32_t>()[i]] |= 2;
      int shown = 0;
      for (std::size_t z = 0; z < zcount_; ++z) {
        if ((want[z] != zc[z] || (want[z] && isfree[z])) && shown < 6) {
          std::fprintf(stderr, "[gpu] CHECK level %d: zone %zu has %u live entries, refcount %u, %s\n",
                       level_, z, want[z], zc[z], isfree[z] == 1 ? "on the free list"
                       : isfree[z] == 2 ? "queued for reuse" : isfree[z] ? "free and queued" : "in use");
          ++shown; ++bad;
        }
      }
    }
    for (std::size_t a = 0; a < chk_acount_ && bad < 3; ++a) {
      if (!chk_alive_[a] || al[a]) continue;
      const void* z = zone_ptr(zr[a]);
      if (covered_by_live(z, chk_key_[a], false)) continue;
      ++bad;
      std::fprintf(stderr, "[gpu] CHECK level %d: stored entry %zu died uncovered\n", level_, a);
      print_zone("died", z);
      covered_by_live(z, chk_key_[a], true);
    }
    const std::uint32_t nvalid = ctr_.as<std::uint32_t>()[kValid];
    const std::uint32_t* keep = keep_.as<std::uint32_t>();
    const std::uint64_t* sk = chk_sk_->as<std::uint64_t>();
    const std::uint32_t* sv = chk_sv_->as<std::uint32_t>();
    for (std::uint32_t j = 0; j < nvalid && bad < 3; ++j) {
      if (keep[j]) continue;
      const void* z = static_cast<const char*>(bz_.b.contents) + std::size_t(sv[j]) * nn_ * gz_;
      if (covered_by_live(z, sk[j], false)) continue;
      ++bad;
      std::fprintf(stderr, "[gpu] CHECK level %d: candidate %u dropped uncovered (after phase 1: %s)\n",
                   level_, j, keep1_.as<std::uint32_t>()[j] ? "kept" : "dropped");
      print_zone("dropped", z);
      covered_by_live(z, sk[j], true);
      // Why was it dropped? Stored zones (alive before the batch) and other candidates that cover it.
      std::vector<std::int32_t> L, U;
      key_bounds(sk[j], L, U);
      for (std::size_t a = 0; a < chk_acount_; ++a) {
        if (!chk_alive_[a] || chk_key_[a] != sk[j]) continue;
        const void* y = zone_ptr(zr[a]);
        if (alu(z, y, L.data(), U.data())) {
          char tag[96];
          std::snprintf(tag, sizeof tag, "covering stored entry %zu, alive now %u", a, al[a]);
          print_zone(tag, y);
        }
      }
      for (std::uint32_t i = 0; i < nvalid; ++i) {
        if (i == j || sk[i] != sk[j]) continue;
        const void* y = static_cast<const char*>(bz_.b.contents) + std::size_t(sv[i]) * nn_ * gz_;
        if (alu(z, y, L.data(), U.data())) {
          char tag[96];
          std::snprintf(tag, sizeof tag, "covering candidate %u (as now in bz), keep %u", i, keep[i]);
          print_zone(tag, y);
        }
      }
    }
    FR_CHECK(bad == 0, "HORA_GPU_CHECK: a batch removed a zone no remaining zone covers");
  }
  // The kernels' zone hash, on the host, for zone `z` (fr_entry_hash).
  static std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
  }
  std::uint32_t zone_hash(std::size_t z) const {
    std::uint32_t h = 0;
    const void* p = zone_ptr(z);
    for (int e = 0; e < nn_; ++e) {
      const std::int32_t v = gz_ == 1 ? std::int32_t(static_cast<const std::int8_t*>(p)[e])
                           : gz_ == 2 ? std::int32_t(static_cast<const std::int16_t*>(p)[e])
                                      : static_cast<const std::int32_t*>(p)[e];
      h += mix32(std::uint32_t(e) * 0x9e3779b9u ^ std::uint32_t(v));
    }
    return mix32(h);
  }
  // The initial zone, stored on the host; throws Overflow like the kernels when it does not fit.
  void store_zone(std::size_t z, const raw_t* d) {
    for (int k = 0; k < nn_; ++k) {
      const bool fits = d[k] >= kInf || (gz_ == 1 ? (d[k] > -127 && d[k] < kSInf)
                                         : gz_ == 2 ? (d[k] > -32767 && d[k] < kNInf) : true);
      if (!fits) throw Overflow{};
    }
    void* p = zone_ptr(z);
    for (int k = 0; k < nn_; ++k) {
      if (gz_ == 1) zput(static_cast<sraw_t*>(p), k, d[k]);
      else if (gz_ == 2) zput(static_cast<nraw_t*>(p), k, d[k]);
      else static_cast<raw_t*>(p)[k] = d[k];
    }
  }

  // ---------------------------------------------------------------- encoding
  // One command buffer and one compute encoder at a time. The arena pages are named only through
  // the page table, so every encoder declares them resident.
  void begin() {
    cb_ = [q_ commandBuffer];
    enc_ = [cb_ computeCommandEncoder];
    if (!pages_.empty())
      [enc_ useResources:pages_.data() count:pages_.size()
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
  id<MTLCommandBuffer> submit() {
    [enc_ endEncoding];
    [cb_ commit];
    id<MTLCommandBuffer> cb = cb_;
    if (profile_) { wait(cb); charge(cb); }
    enc_ = nil; cb_ = nil;
    return cb;
  }
  // A timed wait: a command buffer still running after HORA_GPU_WAIT_S seconds (default 120) is
  // reported -- with the kernel that was running, under HORA_GPU_PROFILE -- and the process ends.
  // The GPU cannot be stopped from here, but the run no longer sits on a hung GPU indefinitely.
  void wait(id<MTLCommandBuffer> cb) {
    const auto t0 = Clock::now();
    while (cb.status < MTLCommandBufferStatusCompleted) {
      if (ms_since(t0) > wait_limit_ms_) {
        std::fprintf(stderr, "[gpu] command buffer still running after %.0f s (level %d, kernel %s): "
                     "aborting\n", wait_limit_ms_ / 1000, level_, kernel_ ? kernel_ : "?");
        std::fflush(stderr);
        std::_Exit(3);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    FR_CHECK(cb.status == MTLCommandBufferStatusCompleted,
             std::string("command buffer failed: ") + (cb.error ? cb.error.localizedDescription.UTF8String : "?"));
  }
  // Profiling: the kernel just finished is charged its command buffer's GPU time.
  void charge(id<MTLCommandBuffer> cb) {
    if (kernel_) { auto& p = prof_[kernel_]; p.first += (cb.GPUEndTime - cb.GPUStartTime) * 1e3; ++p.second; }
    kernel_ = nullptr;
  }
  void report_profile() const {
    std::vector<std::pair<double, std::string>> v;
    double total = 0;
    for (const auto& [k, p] : prof_) { v.push_back({p.first, k}); total += p.first; }
    std::sort(v.rbegin(), v.rend());
    std::fprintf(stderr, "[gpu] kernel profile (%.0f ms of GPU time):\n", total);
    for (const auto& [ms, k] : v)
      std::fprintf(stderr, "[gpu]   %-20s %9.1f ms  %5.1f%%  %8zu dispatches\n", k.c_str(), ms,
                   100.0 * ms / std::max(total, 1e-9), prof_.at(k).second);
  }
  void use(const char* name) {
    if (profile_ && kernel_) {   // profiling: the previous kernel gets a command buffer of its own
      wait(submit());
      begin();
    }
    kernel_ = name;
    cur_ = pso_.at(name);
    [enc_ setComputePipelineState:cur_];
  }
  // Bind buffers from index 0; a null entry leaves its index unbound (a kernel argument set
  // separately, e.g. by setBytes).
  void bind(std::initializer_list<const Buf*> bufs) {
    NSUInteger i = 0;
    for (const Buf* b : bufs) { if (b) [enc_ setBuffer:b->b offset:0 atIndex:i]; ++i; }
  }
  void set(NSUInteger index, const Buf& b) { [enc_ setBuffer:b.b offset:0 atIndex:index]; }
  // ---- launch geometry, from each pipeline's own limits
  // Threads a threadgroup of `name` may have: at most 256, within the pipeline's limit (which
  // falls as its kernel's register use grows), in whole SIMD-groups.
  std::uint32_t launch_tpt(const char* name) const {
    const NSUInteger m = std::min<NSUInteger>(256, pso_.at(name).maxTotalThreadsPerThreadgroup);
    return std::uint32_t(m & ~NSUInteger(31));   // >= 32: checked when the pipelines are made
  }
  // The largest power of two within launch_tpt (for the sort and the scan).
  std::uint32_t launch_pow2(const char* name) const {
    std::uint32_t t = 32;
    while (2 * t <= launch_tpt(name)) t *= 2;
    return t;
  }
  // SIMD-groups a threadgroup of `name` runs, each with `bytes` of threadgroup memory: at most 8,
  // within the thread limit and the threadgroup memory left beside the kernel's static arrays.
  std::uint32_t launch_sgs(const char* name, int bytes) const {
    const id<MTLComputePipelineState> ps = pso_.at(name);
    const NSUInteger room = dev_.maxThreadgroupMemoryLength - ps.staticThreadgroupMemoryLength;
    std::uint32_t k = std::min<std::uint32_t>(8, launch_tpt(name) / 32);
    while (k > 1 && tg_len(k, bytes) > room) --k;
    FR_CHECK(tg_len(k, bytes) <= room, std::string("threadgroup memory too small for ") + name);
    return k;
  }
  static NSUInteger tg_len(std::uint32_t sgs, int bytes) {
    return (NSUInteger(sgs) * NSUInteger(bytes) + 15) & ~NSUInteger(15);
  }
  void dispatch_threads(std::size_t n) {
    if (n == 0) return;
    const NSUInteger tg = std::min<NSUInteger>(256, cur_.maxTotalThreadsPerThreadgroup);
    [enc_ dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  }
  // One SIMD-group per item, `sgs` SIMD-groups a threadgroup (launch_sgs), `bytes` of
  // threadgroup memory each.
  void dispatch_sg(std::size_t items, std::uint32_t sgs, int bytes) {
    if (items == 0) return;
    [enc_ setThreadgroupMemoryLength:tg_len(sgs, bytes) atIndex:0];
    const std::size_t groups = (items + sgs - 1) / sgs;
    [enc_ dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(sgs * 32, 1, 1)];
  }
  // Indirect dispatch: slot `slot` of ind_ := threadgroups covering src[index] * mul items at
  // per_tg items a threadgroup, computed on the device.
  void indirect(const Buf& src, std::size_t index, std::uint32_t mul, std::uint32_t per_tg,
                std::uint32_t slot) {
    struct { std::uint32_t index, mul, per_tg, slot; } ia{std::uint32_t(index), mul, per_tg, slot};
    use("fr_indirect");
    bind({&src, &ind_});
    [enc_ setBytes:&ia length:sizeof(ia) atIndex:2];
    [enc_ dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  }
  // Indirect threads: slot `slot` holds the threadgroup count, `per_tg` threads each.
  void dispatch_indirect(std::uint32_t slot, std::uint32_t per_tg) {
    [enc_ dispatchThreadgroupsWithIndirectBuffer:ind_.b indirectBufferOffset:slot * 12
                           threadsPerThreadgroup:MTLSizeMake(per_tg, 1, 1)];
  }
  // The SIMD-group form: `sgs` must be the per_tg the slot was computed with.
  void dispatch_sg_indirect(std::uint32_t slot, std::uint32_t sgs, int bytes) {
    [enc_ setThreadgroupMemoryLength:tg_len(sgs, bytes) atIndex:0];
    [enc_ dispatchThreadgroupsWithIndirectBuffer:ind_.b indirectBufferOffset:slot * 12
                           threadsPerThreadgroup:MTLSizeMake(sgs * 32, 1, 1)];
  }
  // Exclusive scan of in[0 .. n_in) into out[0 .. n_out); in-place is allowed.
  void scan(const Buf& in, const Buf& out, std::uint32_t n_in, std::uint32_t n_out,
            std::size_t level = 0) {
    if (n_out == 0) return;
    FR_CHECK(level < scan_sums_.size(), "scan nests too deep");
    const std::uint32_t tile = 4 * scan_t_;   // four elements a thread
    const std::uint32_t nblk = (n_out + tile - 1) / tile;
    reserve(scan_sums_[level], std::size_t(nblk) * 4, false);
    struct { std::uint32_t n_in, n_out, tile; } sa{n_in, n_out, tile};
    use("fr_scan_block");
    bind({&in, &out, &scan_sums_[level]});
    [enc_ setBytes:&sa length:sizeof(sa) atIndex:3];
    [enc_ dispatchThreadgroups:MTLSizeMake(nblk, 1, 1) threadsPerThreadgroup:MTLSizeMake(scan_t_, 1, 1)];
    if (nblk > 1) {
      scan(scan_sums_[level], scan_sums_[level], nblk, nblk, level + 1);
      use("fr_scan_add");
      bind({&out, &scan_sums_[level]});
      [enc_ setBytes:&sa length:sizeof(sa) atIndex:2];
      dispatch_threads(n_out);
    }
  }
  void alive_flags(const Buf& idx, std::size_t n, const Buf& flag) {
    if (n == 0) return;
    const std::uint32_t nu = std::uint32_t(n);
    use("fr_alive_flags");
    bind({&idx, &alive_, &flag});
    [enc_ setBytes:&nu length:4 atIndex:3];
    dispatch_threads(n);
  }
  void fill(const Buf& b, std::size_t n, std::uint32_t v) {
    const std::uint32_t a[2] = {std::uint32_t(n), v};
    use("fr_fill_u32");
    bind({&b});
    [enc_ setBytes:a length:sizeof(a) atIndex:1];
    dispatch_threads(n);
  }
  // out = merge(a[0..na), b[0..*nb)), with *nb read on the device from `nbp` at `nb_off`; when
  // `apre` / `bpre` are given, only entries whose flag is set survive (their prefix sums).
  void merge(const Buf& ak, const Buf& aa, std::size_t na, const Buf& bk, const Buf& ba,
             const Buf& nbp, std::size_t nb_off, std::size_t nb_cap, const Buf* apre,
             const Buf* bpre, const Buf& ok, const Buf& oa) {
    struct { std::uint32_t na, nb_cap, filter; } ma{std::uint32_t(na), std::uint32_t(nb_cap), apre ? 1u : 0u};
    const Buf& ap = apre ? *apre : nbp;   // unused when not filtering, but must be bound
    const Buf& bp = bpre ? *bpre : nbp;
    for (int side = 0; side < 2; ++side) {
      const std::size_t n = side == 0 ? na : nb_cap;
      if (!n) continue;
      use(side == 0 ? "fr_merge_a" : "fr_merge_b");
      if (side == 0) bind({&ak, &aa, &bk}); else bind({&bk, &ba, &ak});
      [enc_ setBuffer:nbp.b offset:nb_off atIndex:3];
      set(4, ap); set(5, bp); set(6, ok); set(7, oa);
      [enc_ setBytes:&ma length:sizeof(ma) atIndex:8];
      dispatch_threads(n);
    }
  }

  // ---------------------------------------------------------------- buffers
  // Grow `b` to at least `bytes` (half again its size when the budget allows), keeping its
  // contents when `keep`. A kept buffer's old and new copies coexist during the copy and both
  // count; an unkept one is released first.
  void reserve(Buf& b, std::size_t bytes, bool keep) {
    bytes = std::max<std::size_t>(bytes, 16);
    if (b.b && b.bytes() >= bytes) return;
    if (!keep) release(b);
    const std::size_t limit = static_cast<std::size_t>(dev_.maxBufferLength);
    if (bytes > limit)
      throw OutOfMemory("out of GPU memory: a " + mb(bytes) + " buffer exceeds the device's " +
                        mb(limit) + " limit");
    std::size_t want = std::min(limit, std::max(bytes, b.bytes() + b.bytes() / 2));
    if (mem_used_ + want > mem_cap_) want = bytes;
    charge_mem(want, "a buffer");
    id<MTLBuffer> nb = [dev_ newBufferWithLength:want options:MTLResourceStorageModeShared];
    if (!nb) throw OutOfMemory("out of GPU memory: the system refused a " + mb(want) + " buffer");
    mem_used_ += want;
    if (b.b) std::memcpy(nb.contents, b.b.contents, b.bytes());
    release(b);
    b.b = nb;
  }
  void release(Buf& b) {
    if (!b.b) return;
    mem_used_ -= b.bytes();
    [b.b setPurgeableState:MTLPurgeableStateEmpty];
    b.b = nil;
  }
  // Fails the search when `bytes` more would pass the memory budget.
  void charge_mem(std::size_t bytes, const char* what) const {
    if (mem_used_ + bytes <= mem_cap_) return;
    throw OutOfMemory("out of GPU memory: " + std::string(what) + " of " + mb(bytes) + " would pass the " +
                      mb(mem_cap_) + " budget (" + mb(mem_used_) + " in use; HORA_GPU_MEM_GB sets it)");
  }
  static std::string mb(std::size_t bytes) { return std::to_string((bytes + (1u << 20) - 1) >> 20) + " MB"; }
  template <class T>
  void put(Buf& b, const std::vector<T>& v) {
    reserve(b, std::max<std::size_t>(1, v.size()) * sizeof(T), false);
    if (!v.empty()) std::memcpy(b.b.contents, v.data(), v.size() * sizeof(T));
  }

  const FrontierProblem& P_;
  const Network& net_;
  int n_ = 0, nn_ = 0, np_ = 0, gz_ = 2, passes_ = 1, key_bits_ = 0, nthreads_ = 1;
  std::uint32_t sort_t_ = 256, scan_t_ = 256;   // threads a threadgroup of the sort and the scan
  std::size_t mem_cap_ = 0, mem_used_ = 0;      // the memory budget, and what is allocated
  bool has_tclock_ = false;
  int solo_ = 8, bound_bytes_ = 2;
  bool union_ = true;
  int min_bits_ = 8;
  std::uint64_t clockbits_ = 0, qbits_ = 0, sentinel_ = 0;
  std::size_t wmax_ = 0;
  bool trace_ = false, stress_ = false, profile_ = false;
  double wait_limit_ms_ = 120000;
  int level_ = 0;
  std::size_t nf_prev_ = 0, lvl_w_ = 0, lvl_valid_ = 0, lvl_kept_ = 0, lvl_rows_ = 0, lvl_buckets_ = 0;
  bool check_ = false;    // HORA_GPU_CHECK: verify every removal (check_before / check_after)
  std::vector<std::uint32_t> chk_alive_;
  std::vector<std::uint64_t> chk_key_;
  std::size_t chk_acount_ = 0;
  Buf* chk_sk_ = nullptr;
  Buf* chk_sv_ = nullptr;
  double t_enum_ = 0, t_wait_ = 0;

  id<MTLDevice> dev_ = nil;
  id<MTLCommandQueue> q_ = nil;
  id<MTLCommandBuffer> cb_ = nil;
  id<MTLComputeCommandEncoder> enc_ = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pso_;
  id<MTLComputePipelineState> cur_ = nil;
  const char* kernel_ = nullptr;
  std::map<std::string, std::pair<double, std::size_t>> prof_;   // kernel -> (GPU ms, dispatches)

  // model tables
  Buf locbase_, Lloc_, Uloc_, Lq_, Uq_, kshift_, kmask_, tdbm_, Lx_, Ux_;
  // the zone store: a paged arena, and per zone its hash, table slot and reference count
  std::vector<id<MTLResource>> pages_;
  Buf ptab_;
  int page_shift_ = 16;
  std::size_t page_zones_ = 0;
  Buf zhash_, ztslot_, zcnt_, ztab_, freel_, pend_;
  std::size_t zcount_ = 0, zcap_ = 0, nfree_ = 0, ntomb_ = 0;
  std::uint32_t freed_before_ = 0;
  // passed-set entries: zone, liveness, and the base and delta runs
  Buf zref_, zref2_, alive_, alive2_, remap_;
  Buf base_k_, base_a_, base2_k_, base2_a_, delta_k_, delta_a_, delta2_k_, delta2_a_;
  std::size_t acount_ = 0, nbase_ = 0, ndelta_ = 0;
  std::size_t nlive_ = 1, ndisc_ = 1;   // live entries, distinct discrete states (the initial one)
  bool extrap_ = false;
  // frontier, and the next one as it accumulates
  Buf fk_, fa_, fnk_, fna_;
  std::size_t nf_ = 0, nfn_ = 0;
  // recipes and work, per chunk
  Chunk chunks_[2];
  std::vector<RecipeTable> tables_;
  std::vector<std::size_t> runs_;
  std::vector<std::uint32_t> run_of_, run_beg_, run_cnt_, run_lbeg_;
  std::vector<int> run_tid_;
  // one batch
  Buf ind_;                 // indirect dispatch arguments, three uints a slot
  Buf Lb_, Ub_;             // a_LU bounds of each bucket of the batch
  Buf big1_, big2_;         // reduction rows too large for one lane
  Buf ulist_;               // buckets for the convex-union phase
  Buf bz_, sk0_, sk1_, sv0_, sv1_, ghist_, head_, hpre_, bstart_, bkey_, binfo_, rowcnt_,
      rowoff_, keep_, kpre_, keep1_, k1pre_, surv_, new_k_, new_a_, ctr_, nbuf_;
  Buf hsh_, rep_, zslot_, win_, wpre_, zid_, wlist_;
  Buf fl_a_, fl_ap_, fl_b_, fl_bp_;
  std::vector<Buf> scan_sums_;
};

}  // namespace

int frontier_reach_metal(const FrontierProblem& p) {
  const ReachStats before = p.stats ? *p.stats : ReachStats{};
  int bits = 8;
  for (;;) {
    @autoreleasepool {
      FrontierMetal m(p, bits);
      try {
        return m.run();
      } catch (const Overflow&) {
        // A zone did not fit the stored entries: start over one width up. Nothing of the
        // abandoned attempt survives, including its statistics.
        const int got = m.bits();
        if (got >= 32) throw std::runtime_error("Metal frontier: a zone exceeds 32-bit entries");
        bits = got * 2;
        if (p.stats) *p.stats = before;
        if (std::getenv("HORA_GPU_TRACE"))
          std::fprintf(stderr, "[gpu] a zone exceeds %d-bit entries: restarting with %d-bit\n", got, bits);
      }
      // Any other failure -- out of memory, a failed command buffer -- propagates to the caller
      // as an error: the GPU engine does not hand a search it started to the CPU.
    }
  }
}

}  // namespace detail
}  // namespace hora

#endif  // HORA_HAVE_METAL
