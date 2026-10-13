// frontier_metal.mm — the frontier-batched reachability engine on Metal.
//
// Per BFS level, the frontier is cut into batches sized to a memory budget, and each batch runs
// as two command buffers:
//
//   A  expand (one SIMD-group per zone x transition) -> radix sort by discrete key -> buckets
//      -> each bucket's stored zones located in the passed set -> rows per bucket
//   B  all-pairs reduction -> survivors placed in the arena -> survivors merged into the delta
//
// The host reads four counters between A and B (valid candidates, the first target hit, buckets,
// rows) and sizes B's dispatches from them. At the level's end the next frontier is compacted to
// the zones still alive, and the passed set and arena are compacted when garbage dominates.
//
// The passed set is a sorted *base* of (key, arena slot) plus a sorted *delta* receiving each
// batch's survivors. A batch therefore merges into the delta, O(delta + survivors), rather than
// rewriting the whole passed set; the delta is folded into the base once it outgrows a fraction
// of it. A dropped zone is marked dead in `alive`, indexed by arena slot, and is removed when the
// runs are next merged. All buffers are shared memory: the host reads results where the GPU
// wrote them and fills the recipe and work tables in place.
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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef HORA_HAVE_OPENMP
#include <omp.h>
#endif

#include "hdbm/metal/gdbm.hpp"
#include "hora/detail/frontier_gpu.hpp"
#include "hora/detail/semantics.hpp"
#include "hora/zone.hpp"
#include "src/metal/frontier_kernels.hpp"

namespace hora {
namespace detail {
namespace {

constexpr int kMaxDim = 64;                   // clocks <= 63: one bit each in a 64-bit mask
constexpr std::uint32_t kNoHit = UINT32_MAX;
constexpr std::uint32_t kTile = 1024;         // must match the kernels' kTile
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

    // Largest constant anywhere: decides whether zones fit 16-bit storage.
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

    // SIMD-groups per threadgroup: as many as fit a 16 KB slice of threadgroup memory, so two
    // threadgroups can share a core.
    const int per_sg = (nn_ + 2 * n_) * 4;
    sgs_ = std::max(1, std::min(8, 16384 / per_sg));
    if (per_sg > 32768) return false;

    // Coverage tests in the reduction: a pair per lane, or a pair per SIMD-group (GDBM's warp
    // form, 32 entries a step). HORA_GPU_REDUCE=lane|simd overrides, for measuring.
    //
    // Measured against a tiled reduction that staged RT x CT blocks of the comparison matrix in
    // threadgroup memory (64x64 down to 8x32): the per-row kernels won on every model, by 1.5x on
    // grc6 and 4x on csma10. The GPU's caches already give the reuse the tiles were for, and the
    // tiles cost barriers and idle SIMD-groups while some rows are still scanning.
    // A pair per lane won at every dimension measured, including hddi15's 47 (7.9 s against 14.8
    // s): the a_LU test usually finds its witness within a few entries, so a SIMD-group spent on
    // one pair mostly votes on chunks it did not need.
    lane_cmp_ = true;
    if (const char* m = std::getenv("HORA_GPU_REDUCE")) lane_cmp_ = std::string(m) != "simd";
    NSError* err = nil;
    MTLCompileOptions* o = [[MTLCompileOptions alloc] init];
    o.preprocessorMacros = @{
      @"PT_DIM" : @(n_), @"PT_NP" : @(np_), @"FR_SGS" : @(sgs_),
      @"GZ" : gz_ == 1 ? @"char" : gz_ == 2 ? @"short" : @"int",
      @"FR_LANE_CMP" : @(lane_cmp_ ? 1 : 0),
    };
    std::string src = std::string(hdbm::metal::gdbm_source()) + metal::frontier_kernels_source();
    id<MTLLibrary> lib = [dev_ newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                            options:o error:&err];
    FR_CHECK(lib != nil, std::string("kernel compile failed: ") +
                             (err ? err.localizedDescription.UTF8String : "?"));
    for (const char* name : {"fr_expand", "fr_scan_block", "fr_scan_add", "fr_rs_hist", "fr_rs_scatter",
                             "fr_heads", "fr_buckets", "fr_bucket_setup", "fr_reduce_stored", "fr_survivors",
                             "fr_keep_init", "fr_rows2", "fr_reduce_pairs", "fr_zhash", "fr_intern",
                             "fr_zwin", "fr_zplace", "fr_place_new", "fr_merge_a", "fr_merge_b",
                             "fr_alive_flags", "fr_compact_frontier", "fr_fill_u32", "fr_zrebuild",
                             "fr_gc_mark", "fr_gc_zcopy", "fr_gc_entries", "fr_remap", "fr_indirect"}) {
      id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
      FR_CHECK(f != nil, std::string("missing kernel ") + name);
      id<MTLComputePipelineState> ps = [dev_ newComputePipelineStateWithFunction:f error:&err];
      FR_CHECK(ps != nil, std::string("pipeline failed: ") + name);
      pso_[name] = ps;
    }

    // Static model tables.
    std::vector<int> locbase(np_ + 1, 0);
    for (int p = 0; p < np_; ++p) locbase[p + 1] = locbase[p] + int(net_.procs[p].locations.size());
    const int nloc = locbase[np_];
    std::vector<int> inv_off(nloc + 1, 0), inv;
    std::vector<std::uint64_t> actm(nloc, ~0ull);
    std::vector<int> Lloc(std::size_t(nloc) * n_, kAluNegInf), Uloc(std::size_t(nloc) * n_, kAluNegInf);
    for (int p = 0; p < np_; ++p)
      for (int l = 0; l < int(net_.procs[p].locations.size()); ++l) {
        const int gl = locbase[p] + l;
        for (const auto& c : net_.procs[p].locations[l].invariant) {
          inv.push_back(c.i); inv.push_back(c.j); inv.push_back(c.bound);
        }
        inv_off[gl + 1] = int(inv.size() / 3);
        if (!net_.active.empty()) actm[gl] = net_.active_mask[p][l];
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
    put(locbase_, locbase); put(inv_off_, inv_off); put(inv_, inv); put(actm_, actm);
    put(Lloc_, Lloc); put(Uloc_, Uloc); put(Lq_, Lq); put(Uq_, Uq);
    put(kshift_, kshift); put(kmask_, kmask); put(tdbm_, tdbm);
    put(Lx_, std::vector<int>(P_.Lx.begin(), P_.Lx.end()));
    put(Ux_, std::vector<int>(P_.Ux.begin(), P_.Ux.end()));

    for (int c = 1; c <= net_.nclocks; ++c) {
      clockbits_ |= 1ull << c;
      if (c < int(P_.qmask.size()) && P_.qmask[c]) qbits_ |= 1ull << c;
    }
    key_bits_ = P_.key_bits;
    sentinel_ = 1ull << key_bits_;
    passes_ = (key_bits_ + 1 + 7) / 8;

    // Batch budget: the per-work-item buffers of one batch (candidate zone, two sort buffers,
    // grouping and reduction arrays, the work table) stay within an eighth of memory, capped.
    const std::size_t ram = static_cast<std::size_t>([NSProcessInfo processInfo].physicalMemory);
    const std::size_t budget = std::min<std::size_t>(ram / 8, std::size_t(2) << 30);
    const std::size_t per_item = std::size_t(nn_) * gz_ + 112;
    wmax_ = std::max<std::size_t>(1024, budget / per_item);
    wmax_ = std::min<std::size_t>(wmax_, std::size_t(UINT32_MAX / 2));
#ifdef HORA_HAVE_OPENMP
    nthreads_ = omp_get_max_threads();
#endif
    // One sums buffer per scan level, sized once: scan() recurses with references into this
    // vector, so it must never reallocate. Eight levels of 1024-way blocks cover any length.
    scan_sums_.resize(8);
    trace_ = std::getenv("HORA_GPU_TRACE") != nullptr;
    // HORA_GPU_STRESS: drive every rarely-taken path on every level -- sub-batches, many chunks,
    // a fold after every batch, arena compaction whenever there is garbage -- so the small
    // models of the differential test exercise them too. Verdicts must not change.
    stress_ = std::getenv("HORA_GPU_STRESS") != nullptr;
    if (stress_) wmax_ = 256;
    if (trace_)
      std::fprintf(stderr, "[gpu] %s | dim %d, %d procs, key %d bits, %s zones, %d SIMD-groups/tg, "
                   "batch <= %zu items, reduce: %s\n", dev_.name.UTF8String, n_, np_, key_bits_,
                   gz_ == 1 ? "8-bit" : gz_ == 2 ? "16-bit" : "32-bit", sgs_, wmax_,
                   lane_cmp_ ? "a pair per lane" : "a pair per SIMD-group");
    return true;
  }

  // ---------------------------------------------------------------- search
  // The recipe and work tables of one chunk of a level. Two of them alternate, so the host can
  // enumerate chunk k+1 into one while the GPU reads chunk k from the other.
  struct Chunk {
    Buf rkey, rflags, rg_off, rg, rr_off, rr, wf, wr;
    std::size_t W = 0;
  };

  int search() {
    // Initial state: arena slot 0, the only entry of the base and of the frontier.
    const std::uint64_t k0 = P_.keys.encode(P_.s0.locs, P_.s0.ints);
    reserve(arena_, std::size_t(nn_) * gz_ * 1024, false);
    store_zone(0, P_.s0.zone.data().data());
    reserve(zhash_, 1024 * 4, false); zhash_.as<std::uint32_t>()[0] = zone_hash(0);
    reserve(zref_, 1024 * 4, false); zref_.as<std::uint32_t>()[0] = 0;
    zcount_ = 1;
    ensure_table(1024);
    reserve(alive_, 1024 * 4, false); alive_.as<std::uint32_t>()[0] = 1;
    reserve(base_k_, 1024 * 8, false); reserve(base_a_, 1024 * 4, false);
    base_k_.as<std::uint64_t>()[0] = k0; base_a_.as<std::uint32_t>()[0] = 0;
    reserve(delta_k_, 1024 * 8, false); reserve(delta_a_, 1024 * 4, false);
    reserve(fk_, 1024 * 8, false); reserve(fa_, 1024 * 4, false);
    fk_.as<std::uint64_t>()[0] = k0; fa_.as<std::uint32_t>()[0] = 0;
    acount_ = 1; nbase_ = 1; ndelta_ = 0; nf_ = 1;
    reserve(ctr_, 64, false);
    reserve(ind_, 64, false);

    while (nf_ > 0) {
      if (P_.cancel && P_.cancel->load(std::memory_order_relaxed)) return 0;
      if (P_.stats) P_.stats->explored += nf_;
      nf_prev_ = nf_;
      t_enum_ = t_wait_ = 0;
      std::fill(std::begin(gpu_ms_), std::end(gpu_ms_), 0.0);
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
          commit_wait(cb, false);
          t_wait_ += ms_since(tw);
          if (finish_batch(cb, w1 - w0)) return 1;
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
      if (trace_) {
        std::fprintf(stderr, "[gpu] level %4d  frontier %9zu  work %10zu  valid %10zu  kept %9zu  "
                     "passed %10zu zones %10zu buckets %8zu rows %10zu chunks %zu | enum %7.2f wait %7.2f "
                     "end %6.2f ms\n", level_, nf_prev_, lvl_w_, lvl_valid_, lvl_kept_,
                     nbase_ + ndelta_, zcount_, lvl_buckets_, lvl_rows_, cuts.size() - 1, t_enum_, t_wait_,
                     ms_since(t_end));
        std::fprintf(stderr, "[gpu]            gpu: expand %7.2f sort %7.2f group %7.2f | phase1 %8.2f "
                     "phase2 %8.2f place %7.2f ms\n", gpu_ms_[0], gpu_ms_[1], gpu_ms_[2], gpu_ms_[3],
                     gpu_ms_[4], gpu_ms_[5]);
      }
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
      append_recipes(net_, P_.keys, *P_.target, P_.canon, locs, ints, t);
      run_tid_[r] = tid; run_lbeg_[r] = std::uint32_t(before); run_cnt_[r] = std::uint32_t(t.size() - before);
    }
    // Concatenate the per-thread tables straight into the chunk's shared recipe buffers.
    std::vector<std::size_t> rb0(nthreads_ + 1, 0), gb(nthreads_ + 1, 0), xb(nthreads_ + 1, 0);
    for (int t = 0; t < nthreads_; ++t) {
      rb0[t + 1] = rb0[t] + tables_[t].size();
      gb[t + 1] = gb[t] + tables_[t].g.size() / 3;
      xb[t + 1] = xb[t] + tables_[t].r.size();
    }
    const std::size_t R = rb0[nthreads_];
    reserve(ch.rkey, std::max<std::size_t>(1, R) * 8, false);
    reserve(ch.rflags, std::max<std::size_t>(1, R) * 4, false);
    reserve(ch.rg_off, (R + 1) * 4, false);
    reserve(ch.rr_off, (R + 1) * 4, false);
    reserve(ch.rg, std::max<std::size_t>(1, gb[nthreads_]) * 12, false);
    reserve(ch.rr, std::max<std::size_t>(1, xb[nthreads_]) * 4, false);
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel for schedule(static, 1) num_threads(nthreads_)
#endif
    for (int t = 0; t < nthreads_; ++t) {
      const RecipeTable& T = tables_[t];
      if (T.size()) {
        std::memcpy(ch.rkey.as<std::uint64_t>() + rb0[t], T.key.data(), T.size() * 8);
        std::memcpy(ch.rflags.as<std::uint32_t>() + rb0[t], T.flags.data(), T.size() * 4);
      }
      if (!T.g.empty()) std::memcpy(ch.rg.as<std::int32_t>() + gb[t] * 3, T.g.data(), T.g.size() * 4);
      if (!T.r.empty()) std::memcpy(ch.rr.as<std::int32_t>() + xb[t], T.r.data(), T.r.size() * 4);
      std::uint32_t* go = ch.rg_off.as<std::uint32_t>();
      std::uint32_t* ro = ch.rr_off.as<std::uint32_t>();
      for (std::size_t r = 0; r < T.size(); ++r) {
        go[rb0[t] + r] = std::uint32_t(gb[t] + T.g_off[r]);
        ro[rb0[t] + r] = std::uint32_t(xb[t] + T.r_off[r]);
      }
    }
    ch.rg_off.as<std::uint32_t>()[R] = std::uint32_t(gb[nthreads_]);
    ch.rr_off.as<std::uint32_t>()[R] = std::uint32_t(xb[nthreads_]);
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

  // Encode one batch -- work items [w0, w0 + W) of chunk `ch` -- as a single command buffer. Every
  // size that depends on the batch's own results (valid candidates, reduction rows) is read on the
  // device: those dispatches are indirect, and the rest are sized by W with the excess threads
  // exiting at once. The caller commits; finish_batch() reads the outcome.
  id<MTLCommandBuffer> encode_batch(Chunk& ch, std::size_t w0, std::size_t W) {
    const std::uint32_t Wu = std::uint32_t(W);
    reserve(bz_, W * nn_ * gz_, false);
    for (Buf* b : {&sk0_, &sk1_}) reserve(*b, W * 8, false);
    for (Buf* b : {&sv0_, &sv1_}) reserve(*b, W * 4, false);
    reserve(head_, W * 4, false); reserve(hpre_, (W + 1) * 4, false);
    reserve(bstart_, W * 4, false); reserve(bkey_, W * 8, false); reserve(binfo_, W * 32, false);
    reserve(rowcnt_, W * 4, false); reserve(rowoff_, (W + 1) * 4, false);
    reserve(keep_, W * 4, false); reserve(keep1_, W * 4, false);
    reserve(k1pre_, (W + 1) * 4, false); reserve(kpre_, (W + 1) * 4, false);
    reserve(surv_, W * 4, false);
    reserve(new_k_, W * 8, false); reserve(new_a_, W * 4, false);
    reserve(arena_, (zcount_ + W) * nn_ * gz_, true);
    reserve(zhash_, (zcount_ + W) * 4, true);
    reserve(zref_, (acount_ + W) * 4, true);
    reserve(alive_, (acount_ + W) * 4, true);
    reserve(hsh_, W * 4, false); reserve(rep_, W * 4, false); reserve(zslot_, W * 4, false);
    reserve(win_, W * 4, false); reserve(wpre_, (W + 1) * 4, false);
    ensure_table(zcount_ + W);
    reserve(fnk_, (nfn_ + W) * 8, true);
    reserve(fna_, (nfn_ + W) * 4, true);
    reserve(delta2_k_, (ndelta_ + W) * 8, false);
    reserve(delta2_a_, (ndelta_ + W) * 4, false);
    const std::uint32_t ntiles = std::uint32_t((W + kTile - 1) / kTile);
    reserve(ghist_, std::size_t(ntiles) * 256 * 4, false);
    std::uint32_t* ctr = ctr_.as<std::uint32_t>();
    ctr[0] = 0; ctr[1] = kNoHit; ctr[2] = 0; ctr[3] = 0; ctr[4] = 0;

    id<MTLCommandBuffer> cb = [q_ commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    // ---- expand ----
    struct { std::uint32_t nwork; std::int32_t has_tclock; std::uint64_t qmask, clockbits, sentinel;
             std::int32_t extrap, pad; }
        ea{Wu, has_tclock_ ? 1 : 0, qbits_, clockbits_, sentinel_, extrap_ ? 1 : 0, 0};
    use(e, "fr_expand");
    bind(e, {&ch.wf, &ch.wr, &fa_, &arena_, &ch.rkey, &ch.rflags, &ch.rg_off, &ch.rg, &ch.rr_off,
             &ch.rr, &locbase_, &inv_off_, &inv_, &actm_, &kshift_, &kmask_, &tdbm_, &bz_, &sk0_,
             &sv0_, &ctr_});
    [e setBuffer:ch.wf.b offset:w0 * 4 atIndex:0];
    [e setBuffer:ch.wr.b offset:w0 * 4 atIndex:1];
    [e setBytes:&ea length:sizeof(ea) atIndex:21];
    [e setBuffer:Lx_.b offset:0 atIndex:22];
    [e setBuffer:Ux_.b offset:0 atIndex:23];
    [e setBuffer:zref_.b offset:0 atIndex:24];
    dispatch_sg(e, W, (nn_ + n_) * 4);
    mark(cb, e, 0);
    // ---- sort by discrete key ----
    Buf* sk = &sk0_; Buf* sv = &sv0_;
    for (int pass = 0; pass < passes_; ++pass) {
      Buf* dk = (sk == &sk0_) ? &sk1_ : &sk0_;
      Buf* dv = (sv == &sv0_) ? &sv1_ : &sv0_;
      struct { std::uint32_t n, ntiles, shift; } ra{Wu, ntiles, std::uint32_t(pass * 8)};
      use(e, "fr_rs_hist");
      bind(e, {sk, &ghist_});
      [e setBytes:&ra length:sizeof(ra) atIndex:2];
      [e dispatchThreadgroups:MTLSizeMake(ntiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      scan(e, ghist_, ghist_, ntiles * 256, ntiles * 256);
      use(e, "fr_rs_scatter");
      bind(e, {sk, sv, dk, dv, &ghist_});
      [e setBytes:&ra length:sizeof(ra) atIndex:5];
      [e dispatchThreadgroups:MTLSizeMake(ntiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      sk = dk; sv = dv;
    }
    sorted_k_ = sk; sorted_v_ = sv;
    mark(cb, e, 1);
    // ---- group into buckets, and find each bucket's stored zones ----
    struct { std::uint32_t W, nbase, ndelta; } ga{Wu, std::uint32_t(nbase_), std::uint32_t(ndelta_)};
    use(e, "fr_heads");
    bind(e, {sk, &head_, &ctr_});
    [e setBytes:&ga length:sizeof(ga) atIndex:3];
    [e setBuffer:keep1_.b offset:0 atIndex:4];
    [e setBuffer:keep_.b offset:0 atIndex:5];
    dispatch_threads(e, W);
    scan(e, head_, hpre_, Wu, Wu + 1);
    use(e, "fr_buckets");
    bind(e, {sk, &head_, &hpre_, &bstart_, &bkey_});
    [e setBytes:&ga length:sizeof(ga) atIndex:5];
    dispatch_threads(e, W);
    use(e, "fr_bucket_setup");
    bind(e, {&bkey_, &bstart_, &hpre_, &ctr_, &base_k_, &delta_k_, &binfo_, &rowcnt_});
    [e setBytes:&ga length:sizeof(ga) atIndex:8];
    dispatch_threads(e, W);
    mark(cb, e, 2);
    // ---- reduce, phase 1: each candidate against its bucket's stored zones ----
    struct { std::uint32_t W; } rd{Wu};
    const int tg_red = (nn_ + 2 * n_) * 4;
    indirect(e, ctr_, 0, 1, sgs_, 0);
    use(e, "fr_reduce_stored");
    bind(e, {&hpre_, &head_, &binfo_, &bkey_, &base_a_, &delta_a_, &arena_, &alive_, sv, &bz_,
             &Lloc_, &Uloc_, &Lq_, &Uq_, &locbase_, &kshift_, &kmask_, &keep1_, &ctr_, &zref_});
    dispatch_sg_indirect(e, 0, tg_red);
    mark(cb, e, 3);
    scan(e, keep1_, k1pre_, Wu, Wu + 1);
    use(e, "fr_survivors");
    bind(e, {&keep1_, &k1pre_, &surv_, &ctr_});
    dispatch_threads(e, W);
    use(e, "fr_keep_init");
    bind(e, {&keep1_, &keep_, &ctr_});
    dispatch_threads(e, W);
    use(e, "fr_rows2");
    bind(e, {&binfo_, &hpre_, &k1pre_, &rowcnt_});
    [e setBytes:&rd length:sizeof(rd) atIndex:4];
    [e setBuffer:ctr_.b offset:0 atIndex:5];
    dispatch_threads(e, W);
    scan(e, rowcnt_, rowoff_, Wu, Wu + 1);
    // ---- reduce, phase 2: survivors all-pairs, and stored zones against survivors ----
    indirect(e, rowoff_, W, 1, sgs_, 1);
    use(e, "fr_reduce_pairs");
    bind(e, {&rowoff_, &hpre_, &binfo_, &bkey_, &base_a_, &delta_a_, &arena_, &alive_, sv, &bz_,
             &Lloc_, &Uloc_, &Lq_, &Uq_, &locbase_, &kshift_, &kmask_, &keep_, &k1pre_, &surv_});
    [e setBytes:&rd length:sizeof(rd) atIndex:20];
    [e setBuffer:ctr_.b offset:0 atIndex:21];
    [e setBuffer:zref_.b offset:0 atIndex:22];
    dispatch_sg_indirect(e, 1, tg_red);
    mark(cb, e, 4);
    // ---- intern the survivors' zones, place the survivors, merge them into the delta ----
    scan(e, keep_, kpre_, Wu, Wu + 1);
    use(e, "fr_zhash");
    bind(e, {&keep_, &ctr_, sv, &bz_, &hsh_});
    [e setBytes:&Wu length:4 atIndex:5];
    dispatch_sg_indirect(e, 0, 16);
    struct { std::uint32_t W, mask; } ia{Wu, std::uint32_t(zcap_ - 1)};
    use(e, "fr_intern");
    bind(e, {&keep_, &ctr_, sv, &bz_, &arena_, &hsh_, &zhash_, &ztab_, &rep_, &zslot_});
    [e setBytes:&ia length:sizeof(ia) atIndex:10];
    dispatch_threads(e, W);
    use(e, "fr_zwin");
    bind(e, {&keep_, &ctr_, &rep_, &win_});
    [e setBytes:&Wu length:4 atIndex:4];
    dispatch_threads(e, W);
    scan(e, win_, wpre_, Wu, Wu + 1);
    struct { std::uint32_t W, zcount; } za{Wu, std::uint32_t(zcount_)};
    indirect(e, ctr_, 0, std::uint32_t(nn_), 256, 2);
    use(e, "fr_zplace");
    bind(e, {&win_, &wpre_, sv, &bz_, &arena_, &hsh_, &zhash_, &ztab_, &zslot_});
    [e setBytes:&za length:sizeof(za) atIndex:9];
    [e dispatchThreadgroupsWithIndirectBuffer:ind_.b indirectBufferOffset:2 * 12
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    struct { std::uint32_t W, acount, fbase, zcount; }
        pa{Wu, std::uint32_t(acount_), std::uint32_t(nfn_), std::uint32_t(zcount_)};
    use(e, "fr_place_new");
    bind(e, {&keep_, &kpre_, sk, &ctr_, &new_k_, &new_a_, &alive_, &fnk_, &fna_});
    [e setBytes:&pa length:sizeof(pa) atIndex:9];
    [e setBuffer:rep_.b offset:0 atIndex:10];
    [e setBuffer:wpre_.b offset:0 atIndex:11];
    [e setBuffer:zref_.b offset:0 atIndex:12];
    dispatch_threads(e, W);
    // delta' = merge(delta, survivors); the survivor count is read on the device (kpre[W]).
    merge(e, delta_k_, delta_a_, ndelta_, new_k_, new_a_, kpre_, W * 4, W, nullptr, nullptr,
          delta2_k_, delta2_a_);
    mark(cb, e, 5);
    [e endEncoding];
    [cb commit];
    return cb;
  }

  // After the batch's command buffer completed: bookkeeping, and true on a target hit.
  bool finish_batch(id<MTLCommandBuffer> cb, std::size_t W) {
    if (!trace_) gpu_ms_[6] += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
    const std::uint32_t* ctr = ctr_.as<std::uint32_t>();
    const std::uint32_t nvalid = ctr[0], hit = ctr[1];
    lvl_valid_ += nvalid;
    if (P_.stats) P_.stats->generated += nvalid;
    if (hit != kNoHit) return true;
    if (ctr[4]) throw Overflow{};
    const std::uint32_t kept = kpre_.as<std::uint32_t>()[W];
    lvl_rows_ += rowoff_.as<std::uint32_t>()[W];
    lvl_buckets_ += hpre_.as<std::uint32_t>()[W];
    lvl_kept_ += kept;
    nlive_ += kept;
    nlive_ -= ctr[3];   // stored zones a survivor strictly covered
    ndisc_ += ctr[2];   // discrete states seen for the first time
    std::swap(delta_k_, delta2_k_); std::swap(delta_a_, delta2_a_);
    ndelta_ += kept;
    acount_ += kept;
    zcount_ += wpre_.as<std::uint32_t>()[W];   // zones not already stored
    nfn_ += kept;
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
    id<MTLCommandBuffer> cb = [q_ commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    alive_flags(e, base_a_, nbase_, fl_a_);
    scan(e, fl_a_, fl_ap_, std::uint32_t(nbase_), std::uint32_t(nbase_ + 1));
    alive_flags(e, delta_a_, ndelta_, fl_b_);
    scan(e, fl_b_, fl_bp_, std::uint32_t(ndelta_), std::uint32_t(ndelta_ + 1));
    merge(e, base_k_, base_a_, nbase_, delta_k_, delta_a_, nbuf_, 0, ndelta_, &fl_ap_, &fl_bp_,
          base2_k_, base2_a_);
    [e endEncoding];
    commit_wait(cb);
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
      id<MTLCommandBuffer> cb = [q_ commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      alive_flags(e, fna_, nfn_, fl_a_);
      scan(e, fl_a_, fl_ap_, std::uint32_t(nfn_), std::uint32_t(nfn_ + 1));
      struct { std::uint32_t n; } ca{std::uint32_t(nfn_)};
      use(e, "fr_compact_frontier");
      bind(e, {&fnk_, &fna_, &fl_a_, &fl_ap_, &fk_, &fa_});
      [e setBytes:&ca length:sizeof(ca) atIndex:6];
      dispatch_threads(e, nfn_);
      [e endEncoding];
      commit_wait(cb);
    }
    nf_ = nfn_ ? fl_ap_.as<std::uint32_t>()[nfn_] : 0;
    // Arena compaction once dead zones outnumber live ones by a wide margin.
    if ((acount_ > (std::size_t(1) << 20) && acount_ > 3 * (nbase_ + ndelta_)) ||
        (stress_ && acount_ > nbase_ + ndelta_))
      gc();
  }

  // Compaction: entries renumbered in base order, and the arena reduced to the zones a live
  // entry names; the zone table is rebuilt over the compacted arena.
  void gc() {
    fold();
    const std::size_t n = nbase_, nz = zcount_;
    reserve(zmark_, (nz + 1) * 4, false); reserve(zpre_, (nz + 1) * 4, false);
    reserve(zref2_, std::max<std::size_t>(n, 1024) * 4, false);
    reserve(alive2_, std::max<std::size_t>(n, 1024) * 4, false);
    reserve(remap_, acount_ * 4, false);
    id<MTLCommandBuffer> cb = [q_ commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    const std::uint32_t nu = std::uint32_t(n), fu = std::uint32_t(nf_);
    fill(e, zmark_, nz, 0);
    use(e, "fr_gc_mark");
    bind(e, {&base_a_, &zref_, &zmark_});
    [e setBytes:&nu length:4 atIndex:3];
    dispatch_threads(e, n);
    scan(e, zmark_, zpre_, std::uint32_t(nz), std::uint32_t(nz + 1));
    [e endEncoding];
    commit_wait(cb);
    const std::size_t nz2 = zpre_.as<std::uint32_t>()[nz];
    reserve(arena2_, std::max<std::size_t>(nz2, 1024) * nn_ * gz_, false);
    reserve(zhash2_, std::max<std::size_t>(nz2, 1024) * 4, false);
    cb = [q_ commandBuffer];
    e = [cb computeCommandEncoder];
    struct { std::uint32_t nzones, nbase; } ga{std::uint32_t(nz), nu};
    use(e, "fr_gc_zcopy");
    bind(e, {&zmark_, &zpre_, &arena_, &arena2_, &zhash_, &zhash2_});
    [e setBytes:&ga length:sizeof(ga) atIndex:6];
    dispatch_threads(e, nz * nn_);
    use(e, "fr_gc_entries");
    bind(e, {&base_a_, &zref_, &zref2_, &zpre_, &remap_, &alive2_});
    [e setBytes:&ga length:sizeof(ga) atIndex:6];
    dispatch_threads(e, n);
    if (nf_) {
      use(e, "fr_remap");
      bind(e, {&fa_, &remap_});
      [e setBytes:&fu length:4 atIndex:2];
      dispatch_threads(e, nf_);
    }
    [e endEncoding];
    commit_wait(cb);
    std::swap(arena_, arena2_); std::swap(zhash_, zhash2_);
    std::swap(zref_, zref2_); std::swap(alive_, alive2_);
    acount_ = n;
    zcount_ = nz2;
    // The old arena is garbage; give its pages back.
    [arena2_.b setPurgeableState:MTLPurgeableStateEmpty];
    arena2_.b = nil;
    rebuild_table();
  }

  // The zone table holds at least twice the zones that can be stored by the end of the next
  // batch, so linear probing stays short.
  void ensure_table(std::size_t need) {
    if (zcap_ >= 2 * need) return;
    while (zcap_ < 2 * need) zcap_ = zcap_ ? zcap_ * 2 : 4096;
    rebuild_table();
  }
  void rebuild_table() {
    reserve(ztab_, zcap_ * 4, false);
    id<MTLCommandBuffer> cb = [q_ commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    fill(e, ztab_, zcap_, 0xffffffffu);
    struct { std::uint32_t W, mask; } ia{std::uint32_t(zcount_), std::uint32_t(zcap_ - 1)};
    use(e, "fr_zrebuild");
    bind(e, {&zhash_, &ztab_});
    [e setBytes:&ia length:sizeof(ia) atIndex:2];
    dispatch_threads(e, zcount_);
    [e endEncoding];
    commit_wait(cb);
  }
  void fill(id<MTLComputeCommandEncoder> e, const Buf& b, std::size_t n, std::uint32_t v) {
    const std::uint32_t a[2] = {std::uint32_t(n), v};
    use(e, "fr_fill_u32");
    bind(e, {&b});
    [e setBytes:a length:sizeof(a) atIndex:1];
    dispatch_threads(e, n);
  }
  // The kernels' zone hash, on the host, for the zone stored at `slot` (fr_entry_hash).
  static std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
  }
  std::uint32_t zone_hash(std::size_t slot) const {
    std::uint32_t h = 0;
    for (int e = 0; e < nn_; ++e) {
      const std::int32_t v = gz_ == 1 ? std::int32_t(arena_.as<std::int8_t>()[slot * nn_ + e])
                           : gz_ == 2 ? std::int32_t(arena_.as<std::int16_t>()[slot * nn_ + e])
                                      : arena_.as<std::int32_t>()[slot * nn_ + e];
      h += mix32(std::uint32_t(e) * 0x9e3779b9u ^ std::uint32_t(v));
    }
    return mix32(h);
  }

  // ---------------------------------------------------------------- encoding helpers
  void use(id<MTLComputeCommandEncoder> e, const char* name) {
    cur_ = pso_.at(name);
    [e setComputePipelineState:cur_];
  }
  void bind(id<MTLComputeCommandEncoder> e, std::initializer_list<const Buf*> bufs) {
    NSUInteger i = 0;
    for (const Buf* b : bufs) [e setBuffer:b->b offset:0 atIndex:i++];
  }
  void dispatch_threads(id<MTLComputeCommandEncoder> e, std::size_t n) {
    if (n == 0) return;
    const NSUInteger tg = std::min<NSUInteger>(256, cur_.maxTotalThreadsPerThreadgroup);
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  }
  // One SIMD-group per item, `tg_per_sg` bytes of threadgroup memory each.
  void dispatch_sg(id<MTLComputeCommandEncoder> e, std::size_t items, int tg_per_sg) {
    if (items == 0) return;
    const NSUInteger len = (NSUInteger(sgs_) * tg_per_sg + 15) & ~NSUInteger(15);
    [e setThreadgroupMemoryLength:len atIndex:0];
    const std::size_t groups = (items + sgs_ - 1) / sgs_;
    [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(sgs_ * 32, 1, 1)];
  }
  // Indirect dispatch: slot `slot` of ind_ := threadgroups covering src[index] * mul items at
  // per_tg items a threadgroup, computed on the device.
  void indirect(id<MTLComputeCommandEncoder> e, const Buf& src, std::size_t index, std::uint32_t mul,
                std::uint32_t per_tg, std::uint32_t slot) {
    struct { std::uint32_t index, mul, per_tg, slot; } ia{std::uint32_t(index), mul, per_tg, slot};
    use(e, "fr_indirect");
    bind(e, {&src, &ind_});
    [e setBytes:&ia length:sizeof(ia) atIndex:2];
    [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  }
  void dispatch_sg_indirect(id<MTLComputeCommandEncoder> e, std::uint32_t slot, int tg_per_sg) {
    const NSUInteger len = (NSUInteger(sgs_) * tg_per_sg + 15) & ~NSUInteger(15);
    [e setThreadgroupMemoryLength:len atIndex:0];
    [e dispatchThreadgroupsWithIndirectBuffer:ind_.b indirectBufferOffset:slot * 12
                        threadsPerThreadgroup:MTLSizeMake(sgs_ * 32, 1, 1)];
  }
  // Exclusive scan of in[0 .. n_in) into out[0 .. n_out); in-place is allowed.
  void scan(id<MTLComputeCommandEncoder> e, const Buf& in, const Buf& out, std::uint32_t n_in,
            std::uint32_t n_out, std::size_t level = 0) {
    if (n_out == 0) return;
    const std::uint32_t nblk = (n_out + kTile - 1) / kTile;
    reserve(scan_sums_[level], std::size_t(nblk) * 4, false);
    struct { std::uint32_t n_in, n_out; } sa{n_in, n_out};
    use(e, "fr_scan_block");
    bind(e, {&in, &out, &scan_sums_[level]});
    [e setBytes:&sa length:sizeof(sa) atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(nblk, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (nblk > 1) {
      scan(e, scan_sums_[level], scan_sums_[level], nblk, nblk, level + 1);
      use(e, "fr_scan_add");
      bind(e, {&out, &scan_sums_[level]});
      [e setBytes:&sa length:sizeof(sa) atIndex:2];
      dispatch_threads(e, n_out);
    }
  }
  void alive_flags(id<MTLComputeCommandEncoder> e, const Buf& idx, std::size_t n, const Buf& flag) {
    if (n == 0) return;
    const std::uint32_t nu = std::uint32_t(n);
    use(e, "fr_alive_flags");
    bind(e, {&idx, &alive_, &flag});
    [e setBytes:&nu length:4 atIndex:3];
    dispatch_threads(e, n);
  }
  // out = merge(a[0..na), b[0..*nb)), with *nb read on the device from `nbp` at `nb_off`; when
  // `apre` / `bpre` are given, only entries whose flag is set survive (their prefix sums).
  void merge(id<MTLComputeCommandEncoder> e, const Buf& ak, const Buf& aa, std::size_t na,
             const Buf& bk, const Buf& ba, const Buf& nbp, std::size_t nb_off, std::size_t nb_cap,
             const Buf* apre, const Buf* bpre, const Buf& ok, const Buf& oa) {
    struct { std::uint32_t na, nb_cap, filter; } ma{std::uint32_t(na), std::uint32_t(nb_cap), apre ? 1u : 0u};
    const Buf& ap = apre ? *apre : nbp;   // unused when not filtering, but must be bound
    const Buf& bp = bpre ? *bpre : nbp;
    if (na) {
      use(e, "fr_merge_a");
      bind(e, {&ak, &aa, &bk});
      [e setBuffer:nbp.b offset:nb_off atIndex:3];
      [e setBuffer:ap.b offset:0 atIndex:4];
      [e setBuffer:bp.b offset:0 atIndex:5];
      [e setBuffer:ok.b offset:0 atIndex:6];
      [e setBuffer:oa.b offset:0 atIndex:7];
      [e setBytes:&ma length:sizeof(ma) atIndex:8];
      dispatch_threads(e, na);
    }
    if (nb_cap) {
      use(e, "fr_merge_b");
      bind(e, {&bk, &ba, &ak});
      [e setBuffer:nbp.b offset:nb_off atIndex:3];
      [e setBuffer:ap.b offset:0 atIndex:4];
      [e setBuffer:bp.b offset:0 atIndex:5];
      [e setBuffer:ok.b offset:0 atIndex:6];
      [e setBuffer:oa.b offset:0 atIndex:7];
      [e setBytes:&ma length:sizeof(ma) atIndex:8];
      dispatch_threads(e, nb_cap);
    }
  }
  // Trace mode only: close the commands so far, run them, and charge their GPU time to `slot`, so
  // the trace can break a command buffer down by phase. Otherwise a no-op.
  void mark(__strong id<MTLCommandBuffer>& cb, __strong id<MTLComputeCommandEncoder>& e, int slot) {
    if (!trace_) return;
    [e endEncoding];
    commit_wait(cb);
    gpu_ms_[slot] += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
    cb = [q_ commandBuffer];
    e = [cb computeCommandEncoder];
  }
  void commit_wait(id<MTLCommandBuffer> cb, bool commit = true) {
    if (commit) [cb commit];
    [cb waitUntilCompleted];
    FR_CHECK(cb.status == MTLCommandBufferStatusCompleted,
             std::string("command buffer failed: ") + (cb.error ? cb.error.localizedDescription.UTF8String : "?"));
  }

  // ---------------------------------------------------------------- buffers
  void reserve(Buf& b, std::size_t bytes, bool keep) {
    bytes = std::max<std::size_t>(bytes, 16);
    if (b.b && b.bytes() >= bytes) return;
    const std::size_t want = std::max(bytes, b.bytes() + b.bytes() / 2);
    id<MTLBuffer> nb = [dev_ newBufferWithLength:want options:MTLResourceStorageModeShared];
    FR_CHECK(nb != nil, "out of GPU memory (" + std::to_string(want >> 20) + " MB buffer)");
    if (keep && b.b) std::memcpy(nb.contents, b.b.contents, b.bytes());
    if (b.b) [b.b setPurgeableState:MTLPurgeableStateEmpty];
    b.b = nb;
  }
  template <class T>
  void put(Buf& b, const std::vector<T>& v) {
    reserve(b, std::max<std::size_t>(1, v.size()) * sizeof(T), false);
    if (!v.empty()) std::memcpy(b.b.contents, v.data(), v.size() * sizeof(T));
  }
  // The initial zone, stored on the host; throws Overflow like the kernels when it does not fit.
  void store_zone(std::size_t slot, const raw_t* z) {
    for (int k = 0; k < nn_; ++k) {
      const bool fits = z[k] >= kInf || (gz_ == 1 ? (z[k] > -127 && z[k] < kSInf)
                                         : gz_ == 2 ? (z[k] > -32767 && z[k] < kNInf) : true);
      if (!fits) throw Overflow{};
    }
    if (gz_ == 1) {
      for (int k = 0; k < nn_; ++k) zput(arena_.as<sraw_t>() + slot * nn_, k, z[k]);
    } else if (gz_ == 2) {
      for (int k = 0; k < nn_; ++k) zput(arena_.as<nraw_t>() + slot * nn_, k, z[k]);
    } else {
      std::memcpy(arena_.as<std::int32_t>() + slot * nn_, z, std::size_t(nn_) * 4);
    }
  }

  const FrontierProblem& P_;
  const Network& net_;
  int n_ = 0, nn_ = 0, np_ = 0, sgs_ = 1, gz_ = 2, passes_ = 1, key_bits_ = 0, nthreads_ = 1;
  bool has_tclock_ = false, lane_cmp_ = true;
  int min_bits_ = 8;
  std::uint64_t clockbits_ = 0, qbits_ = 0, sentinel_ = 0;
  std::size_t wmax_ = 0;
  bool trace_ = false, stress_ = false;
  int level_ = 0;
  std::size_t nf_prev_ = 0, lvl_w_ = 0, lvl_valid_ = 0, lvl_kept_ = 0, lvl_rows_ = 0, lvl_buckets_ = 0;
  double t_enum_ = 0, t_wait_ = 0;
  double gpu_ms_[8] = {};

  id<MTLDevice> dev_ = nil;
  id<MTLCommandQueue> q_ = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pso_;
  id<MTLComputePipelineState> cur_ = nil;

  // model tables
  Buf locbase_, inv_off_, inv_, actm_, Lloc_, Uloc_, Lq_, Uq_, kshift_, kmask_, tdbm_, Lx_, Ux_;
  // passed set and arena
  Buf arena_, arena2_, alive_, alive2_, remap_;
  // zone interning: zref[entry] -> arena slot; the table of arena slots by zone hash
  Buf zref_, zref2_, zhash_, zhash2_, ztab_, zmark_, zpre_, hsh_, rep_, zslot_, win_, wpre_;
  std::size_t zcount_ = 0, zcap_ = 0;
  Buf base_k_, base_a_, base2_k_, base2_a_, delta_k_, delta_a_, delta2_k_, delta2_a_;
  std::size_t acount_ = 0, nbase_ = 0, ndelta_ = 0;
  std::size_t nlive_ = 1, ndisc_ = 1;   // live zones, distinct discrete states (the initial one)
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
  Buf* sorted_k_ = nullptr;
  Buf* sorted_v_ = nullptr;
  Buf bz_, sk0_, sk1_, sv0_, sv1_, ghist_, head_, hpre_, bstart_, bkey_, binfo_, rowcnt_,
      rowoff_, keep_, kpre_, keep1_, k1pre_, surv_, new_k_, new_a_, ctr_, nbuf_;
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
        if (got >= 32) return -1;
        bits = got * 2;
        if (p.stats) *p.stats = before;
        if (std::getenv("HORA_GPU_TRACE"))
          std::fprintf(stderr, "[gpu] a zone exceeds %d-bit entries: restarting with %d-bit\n", got, bits);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "warning: %s; falling back to the CPU engine\n", e.what());
        return -1;
      }
    }
  }
}

}  // namespace detail
}  // namespace hora

#endif  // HORA_HAVE_METAL
