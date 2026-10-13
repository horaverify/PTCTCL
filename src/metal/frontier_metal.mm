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
// Every finite entry of a zone over constants <= M lies in [-2M-1, 2M+1]; 16-bit storage is
// used while that stays far inside int16 (the sentinel is 0x7ffe).
constexpr std::int32_t kNarrowMaxConst = 4000;

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
  explicit FrontierMetal(const FrontierProblem& p) : P_(p), net_(*p.net) {}

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
    narrow_ = maxc < kNarrowMaxConst;
    gz_ = narrow_ ? 2 : 4;

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
      @"GZ" : narrow_ ? @"short" : @"int",
      @"FR_LANE_CMP" : @(lane_cmp_ ? 1 : 0),
    };
    std::string src = std::string(hdbm::metal::gdbm_source()) + metal::frontier_kernels_source();
    id<MTLLibrary> lib = [dev_ newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                            options:o error:&err];
    FR_CHECK(lib != nil, std::string("kernel compile failed: ") +
                             (err ? err.localizedDescription.UTF8String : "?"));
    for (const char* name : {"fr_expand", "fr_scan_block", "fr_scan_add", "fr_rs_hist", "fr_rs_scatter",
                             "fr_heads", "fr_buckets", "fr_bucket_setup", "fr_reduce_stored", "fr_survivors",
                             "fr_keep_init", "fr_rows2", "fr_reduce_pairs", "fr_place_new",
                             "fr_copy_new", "fr_merge_a", "fr_merge_b", "fr_alive_flags",
                             "fr_compact_frontier", "fr_gc_copy", "fr_gc_index", "fr_remap"}) {
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
    if (trace_)
      std::fprintf(stderr, "[gpu] %s | dim %d, %d procs, key %d bits, %s zones, %d SIMD-groups/tg, "
                   "batch <= %zu items, reduce: %s\n", dev_.name.UTF8String, n_, np_, key_bits_,
                   narrow_ ? "16-bit" : "32-bit", sgs_, wmax_,
                   lane_cmp_ ? "a pair per lane" : "a pair per SIMD-group");
    return true;
  }

  // ---------------------------------------------------------------- search
  int search() {
    // Initial state: arena slot 0, the only entry of the base and of the frontier.
    const std::uint64_t k0 = P_.keys.encode(P_.s0.locs, P_.s0.ints);
    reserve(arena_, std::size_t(nn_) * gz_ * 1024, false);
    store_zone(0, P_.s0.zone.data().data());
    reserve(alive_, 1024 * 4, false); alive_.as<std::uint32_t>()[0] = 1;
    reserve(base_k_, 1024 * 8, false); reserve(base_a_, 1024 * 4, false);
    base_k_.as<std::uint64_t>()[0] = k0; base_a_.as<std::uint32_t>()[0] = 0;
    reserve(delta_k_, 1024 * 8, false); reserve(delta_a_, 1024 * 4, false);
    reserve(fk_, 1024 * 8, false); reserve(fa_, 1024 * 4, false);
    fk_.as<std::uint64_t>()[0] = k0; fa_.as<std::uint32_t>()[0] = 0;
    acount_ = 1; nbase_ = 1; ndelta_ = 0; nf_ = 1;
    reserve(ctr_, 64, false);

    while (nf_ > 0) {
      if (P_.cancel && P_.cancel->load(std::memory_order_relaxed)) return 0;
      if (P_.stats) P_.stats->explored += nf_;
      const auto t_level = Clock::now();
      build_recipes();
      t_enum_ = ms_since(t_level);
      t_a_ = t_b_ = t_work_ = 0;
      std::fill(std::begin(gpu_ms_), std::end(gpu_ms_), 0.0);
      lvl_w_ = lvl_valid_ = lvl_kept_ = lvl_rows_ = lvl_buckets_ = 0;
      nfn_ = 0;
      std::size_t f = 0;
      while (f < nf_) {
        std::size_t W = 0, f1 = f;
        while (f1 < nf_) {
          const std::size_t c = slot_recipes(f1);
          if (W > 0 && W + c > wmax_) break;
          W += c; ++f1;
        }
        if (W > 0) {
          const auto t0 = Clock::now();
          build_work(f, f1, W);
          t_work_ += ms_since(t0);
          lvl_w_ += W;
          if (run_batch(W)) return 1;
        }
        f = f1;
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
                     "passed %10zu buckets %8zu rows %10zu | enum %7.2f work %6.2f A %7.2f B %7.2f end %6.2f ms\n",
                     level_, nf_prev_, lvl_w_, lvl_valid_, lvl_kept_, nbase_ + ndelta_, lvl_buckets_,
                     lvl_rows_, t_enum_,
                     t_work_, t_a_, t_b_, ms_since(t_end));
      if (trace_)
        std::fprintf(stderr, "[gpu]            gpu: expand %7.2f sort %7.2f group %7.2f | phase1 %8.2f "
                     "phase2 %8.2f ms\n", gpu_ms_[0], gpu_ms_[1], gpu_ms_[2], gpu_ms_[3], gpu_ms_[4]);
      ++level_;
    }
    return 0;
  }

  // Recipes of every distinct key in the frontier. Equal keys are adjacent (each batch's
  // survivors arrive sorted), so a run of equal keys shares one enumeration.
  void build_recipes() {
    nf_prev_ = nf_;
    const std::uint64_t* fk = fk_.as<std::uint64_t>();
    run_of_.resize(nf_);
    runs_.clear();
    for (std::size_t i = 0; i < nf_; ++i) {
      if (i == 0 || fk[i] != fk[i - 1]) runs_.push_back(i);
      run_of_[i] = std::uint32_t(runs_.size() - 1);
    }
    const long nr = long(runs_.size());
    tables_.resize(nthreads_);
    for (auto& t : tables_) t.clear();
    run_tid_.resize(nr); run_lbeg_.resize(nr); run_cnt_.resize(nr);
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 16) num_threads(nthreads_)
#endif
    for (long r = 0; r < nr; ++r) {
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
    // Concatenate the per-thread tables straight into the shared recipe buffers.
    std::vector<std::size_t> rb(nthreads_ + 1, 0), gb(nthreads_ + 1, 0), xb(nthreads_ + 1, 0);
    for (int t = 0; t < nthreads_; ++t) {
      rb[t + 1] = rb[t] + tables_[t].size();
      gb[t + 1] = gb[t] + tables_[t].g.size() / 3;
      xb[t + 1] = xb[t] + tables_[t].r.size();
    }
    const std::size_t R = rb[nthreads_];
    reserve(rkey_, std::max<std::size_t>(1, R) * 8, false);
    reserve(rflags_, std::max<std::size_t>(1, R) * 4, false);
    reserve(rg_off_, (R + 1) * 4, false);
    reserve(rr_off_, (R + 1) * 4, false);
    reserve(rg_, std::max<std::size_t>(1, gb[nthreads_]) * 12, false);
    reserve(rr_, std::max<std::size_t>(1, xb[nthreads_]) * 4, false);
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel for schedule(static, 1) num_threads(nthreads_)
#endif
    for (int t = 0; t < nthreads_; ++t) {
      const RecipeTable& T = tables_[t];
      if (T.size()) {
        std::memcpy(rkey_.as<std::uint64_t>() + rb[t], T.key.data(), T.size() * 8);
        std::memcpy(rflags_.as<std::uint32_t>() + rb[t], T.flags.data(), T.size() * 4);
      }
      if (!T.g.empty()) std::memcpy(rg_.as<std::int32_t>() + gb[t] * 3, T.g.data(), T.g.size() * 4);
      if (!T.r.empty()) std::memcpy(rr_.as<std::int32_t>() + xb[t], T.r.data(), T.r.size() * 4);
      std::uint32_t* go = rg_off_.as<std::uint32_t>();
      std::uint32_t* ro = rr_off_.as<std::uint32_t>();
      for (std::size_t r = 0; r < T.size(); ++r) {
        go[rb[t] + r] = std::uint32_t(gb[t] + T.g_off[r]);
        ro[rb[t] + r] = std::uint32_t(xb[t] + T.r_off[r]);
      }
    }
    rg_off_.as<std::uint32_t>()[R] = std::uint32_t(gb[nthreads_]);
    rr_off_.as<std::uint32_t>()[R] = std::uint32_t(xb[nthreads_]);
    run_beg_.resize(nr);
    for (long r = 0; r < nr; ++r) run_beg_[r] = std::uint32_t(rb[run_tid_[r]] + run_lbeg_[r]);
  }
  std::size_t slot_recipes(std::size_t f) const { return run_cnt_[run_of_[f]]; }

  // Work items of frontier slots [f0, f1): every slot times every recipe of its key.
  void build_work(std::size_t f0, std::size_t f1, std::size_t W) {
    reserve(wf_, W * 4, false);
    reserve(wr_, W * 4, false);
    std::uint32_t* wf = wf_.as<std::uint32_t>();
    std::uint32_t* wr = wr_.as<std::uint32_t>();
    std::size_t w = 0;
    for (std::size_t f = f0; f < f1; ++f) {
      const std::uint32_t run = run_of_[f], c = run_cnt_[run], b = run_beg_[run];
      for (std::uint32_t t = 0; t < c; ++t) { wf[w] = std::uint32_t(f); wr[w] = b + t; ++w; }
    }
  }

  // One batch: returns true when a successor meets the target.
  bool run_batch(std::size_t W) {
    const std::uint32_t Wu = std::uint32_t(W);
    reserve(bz_, W * nn_ * gz_, false);
    for (Buf* b : {&sk0_, &sk1_}) reserve(*b, W * 8, false);
    for (Buf* b : {&sv0_, &sv1_}) reserve(*b, W * 4, false);
    reserve(head_, W * 4, false); reserve(hpre_, (W + 1) * 4, false);
    reserve(bstart_, W * 4, false); reserve(bkey_, W * 8, false); reserve(binfo_, W * 32, false);
    reserve(rowcnt_, W * 4, false); reserve(rowoff_, (W + 1) * 4, false);
    const std::uint32_t ntiles = std::uint32_t((W + kTile - 1) / kTile);
    reserve(ghist_, std::size_t(ntiles) * 256 * 4, false);
    std::uint32_t* ctr = ctr_.as<std::uint32_t>();
    ctr[0] = 0; ctr[1] = kNoHit; ctr[2] = 0; ctr[3] = 0;

    // ---- A: expand, sort, group ----
    Buf* sk = &sk0_; Buf* sv = &sv0_;
    const auto t_a = Clock::now();
    {
      id<MTLCommandBuffer> cb = [q_ commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      struct { std::uint32_t nwork; std::int32_t has_tclock; std::uint64_t qmask, clockbits, sentinel;
               std::int32_t extrap, pad; }
          ea{Wu, has_tclock_ ? 1 : 0, qbits_, clockbits_, sentinel_, extrap_ ? 1 : 0, 0};
      use(e, "fr_expand");
      bind(e, {&wf_, &wr_, &fa_, &arena_, &rkey_, &rflags_, &rg_off_, &rg_, &rr_off_, &rr_, &locbase_,
               &inv_off_, &inv_, &actm_, &kshift_, &kmask_, &tdbm_, &bz_, &sk0_, &sv0_, &ctr_});
      [e setBytes:&ea length:sizeof(ea) atIndex:21];
      [e setBuffer:Lx_.b offset:0 atIndex:22];
      [e setBuffer:Ux_.b offset:0 atIndex:23];
      dispatch_sg(e, W, (nn_ + n_) * 4);
      mark(cb, e, 0);
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
      mark(cb, e, 1);
      struct { std::uint32_t W, nbase, ndelta; } ga{Wu, std::uint32_t(nbase_), std::uint32_t(ndelta_)};
      use(e, "fr_heads");
      bind(e, {sk, &head_, &ctr_});
      [e setBytes:&ga length:sizeof(ga) atIndex:3];
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
      scan(e, rowcnt_, rowoff_, Wu, Wu + 1);
      mark(cb, e, 2);
      [e endEncoding];
      commit_wait(cb);
    }
    t_a_ += ms_since(t_a);
    const std::uint32_t nvalid = ctr[0], hit = ctr[1];
    lvl_valid_ += nvalid;
    if (P_.stats) P_.stats->generated += nvalid;
    if (hit != kNoHit) return true;
    if (nvalid == 0) return false;
    const std::uint32_t nrows = rowoff_.as<std::uint32_t>()[W];
    lvl_rows_ += nrows;
    lvl_buckets_ += hpre_.as<std::uint32_t>()[W];

    // ---- B: reduce, place, merge into the delta ----
    reserve(keep_, std::size_t(nvalid) * 4, false);
    reserve(keep1_, std::size_t(nvalid) * 4, false);
    reserve(k1pre_, (std::size_t(nvalid) + 1) * 4, false);
    reserve(surv_, std::size_t(nvalid) * 4, false);
    reserve(kpre_, (std::size_t(nvalid) + 1) * 4, false);
    reserve(new_k_, std::size_t(nvalid) * 8, false);
    reserve(new_a_, std::size_t(nvalid) * 4, false);
    reserve(arena_, (acount_ + nvalid) * nn_ * gz_, true);
    reserve(alive_, (acount_ + nvalid) * 4, true);
    reserve(fnk_, (nfn_ + nvalid) * 8, true);
    reserve(fna_, (nfn_ + nvalid) * 4, true);
    reserve(delta2_k_, (ndelta_ + nvalid) * 8, false);
    reserve(delta2_a_, (ndelta_ + nvalid) * 4, false);
    const auto t_b = Clock::now();
    {
      id<MTLCommandBuffer> cb = [q_ commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      struct { std::uint32_t W; } rd{Wu};
      const int tg_red = (nn_ + 2 * n_) * 4;
      // Phase 1: each candidate against its bucket's stored zones.
      use(e, "fr_reduce_stored");
      bind(e, {&hpre_, &head_, &binfo_, &bkey_, &base_a_, &delta_a_, &arena_, &alive_, sv, &bz_,
               &Lloc_, &Uloc_, &Lq_, &Uq_, &locbase_, &kshift_, &kmask_, &keep1_, &ctr_});
      dispatch_sg(e, nvalid, tg_red);
      mark(cb, e, 3);
      scan(e, keep1_, k1pre_, nvalid, nvalid + 1);
      use(e, "fr_survivors");
      bind(e, {&keep1_, &k1pre_, &surv_, &ctr_});
      dispatch_threads(e, nvalid);
      use(e, "fr_keep_init");
      bind(e, {&keep1_, &keep_, &ctr_});
      dispatch_threads(e, nvalid);
      use(e, "fr_rows2");
      bind(e, {&binfo_, &hpre_, &k1pre_, &rowcnt_});
      [e setBytes:&rd length:sizeof(rd) atIndex:4];
      [e setBuffer:ctr_.b offset:0 atIndex:5];
      dispatch_threads(e, W);
      scan(e, rowcnt_, rowoff_, Wu, Wu + 1);
      // Phase 2: survivors all-pairs, and stored zones against survivors. Its row total is only
      // known on the device; phase 1's bound (every stored zone and every candidate) caps it.
      use(e, "fr_reduce_pairs");
      bind(e, {&rowoff_, &hpre_, &binfo_, &bkey_, &base_a_, &delta_a_, &arena_, &alive_, sv, &bz_,
               &Lloc_, &Uloc_, &Lq_, &Uq_, &locbase_, &kshift_, &kmask_, &keep_, &k1pre_, &surv_});
      [e setBytes:&rd length:sizeof(rd) atIndex:20];
      [e setBuffer:ctr_.b offset:0 atIndex:21];
      dispatch_sg(e, nrows, tg_red);
      mark(cb, e, 4);
      scan(e, keep_, kpre_, nvalid, nvalid + 1);
      struct { std::uint32_t W, acount, fbase; } pa{nvalid, std::uint32_t(acount_), std::uint32_t(nfn_)};
      use(e, "fr_place_new");
      bind(e, {&keep_, &kpre_, sk, &ctr_, &new_k_, &new_a_, &alive_, &fnk_, &fna_});
      [e setBytes:&pa length:sizeof(pa) atIndex:9];
      dispatch_threads(e, nvalid);
      use(e, "fr_copy_new");
      bind(e, {&keep_, &kpre_, sv, &ctr_, &bz_, &arena_});
      [e setBytes:&pa length:sizeof(pa) atIndex:6];
      dispatch_threads(e, std::size_t(nvalid) * nn_);
      // delta' = merge(delta, survivors); the survivor count is read on the device (kpre[nvalid]).
      merge(e, delta_k_, delta_a_, ndelta_, new_k_, new_a_, kpre_, std::size_t(nvalid) * 4, nvalid,
            nullptr, nullptr, delta2_k_, delta2_a_);
      [e endEncoding];
      commit_wait(cb);
    }
    const std::uint32_t kept = kpre_.as<std::uint32_t>()[nvalid];
    lvl_kept_ += kept;
    nlive_ += kept;
    nlive_ -= ctr[3];   // stored zones a survivor strictly covered
    ndisc_ += ctr[2];   // discrete states seen for the first time
    std::swap(delta_k_, delta2_k_); std::swap(delta_a_, delta2_a_);
    ndelta_ += kept;
    acount_ += kept;
    nfn_ += kept;
    if (ndelta_ > std::max<std::size_t>(nbase_ / 4, 1u << 16)) fold();
    t_b_ += ms_since(t_b);
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
    if (acount_ > (std::size_t(1) << 20) && acount_ > 3 * (nbase_ + ndelta_)) gc();
  }

  void gc() {
    fold();
    const std::size_t n = nbase_;
    reserve(arena2_, std::max<std::size_t>(n, 1024) * nn_ * gz_, false);
    reserve(alive2_, std::max<std::size_t>(n, 1024) * 4, false);
    reserve(remap_, acount_ * 4, false);
    id<MTLCommandBuffer> cb = [q_ commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    const std::uint32_t nu = std::uint32_t(n), fu = std::uint32_t(nf_);
    use(e, "fr_gc_copy");
    bind(e, {&base_a_, &arena_, &arena2_});
    [e setBytes:&nu length:4 atIndex:3];
    dispatch_threads(e, n * nn_);
    use(e, "fr_gc_index");
    bind(e, {&base_a_, &remap_, &alive2_});
    [e setBytes:&nu length:4 atIndex:3];
    dispatch_threads(e, n);
    if (nf_) {
      use(e, "fr_remap");
      bind(e, {&fa_, &remap_});
      [e setBytes:&fu length:4 atIndex:2];
      dispatch_threads(e, nf_);
    }
    [e endEncoding];
    commit_wait(cb);
    std::swap(arena_, arena2_);
    std::swap(alive_, alive2_);
    acount_ = n;
    // The old arena is garbage; give its pages back.
    [arena2_.b setPurgeableState:MTLPurgeableStateEmpty];
    arena2_.b = nil;
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
  void commit_wait(id<MTLCommandBuffer> cb) {
    [cb commit];
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
  void store_zone(std::size_t slot, const raw_t* z) {
    if (narrow_) {
      std::int16_t* d = arena_.as<std::int16_t>() + slot * nn_;
      for (int k = 0; k < nn_; ++k) zput(reinterpret_cast<nraw_t*>(d), k, z[k]);
    } else {
      std::memcpy(arena_.as<std::int32_t>() + slot * nn_, z, std::size_t(nn_) * 4);
    }
  }

  const FrontierProblem& P_;
  const Network& net_;
  int n_ = 0, nn_ = 0, np_ = 0, sgs_ = 1, gz_ = 2, passes_ = 1, key_bits_ = 0, nthreads_ = 1;
  bool narrow_ = true, has_tclock_ = false, lane_cmp_ = true;
  std::uint64_t clockbits_ = 0, qbits_ = 0, sentinel_ = 0;
  std::size_t wmax_ = 0;
  bool trace_ = false;
  int level_ = 0;
  std::size_t nf_prev_ = 0, lvl_w_ = 0, lvl_valid_ = 0, lvl_kept_ = 0, lvl_rows_ = 0, lvl_buckets_ = 0;
  double t_enum_ = 0, t_work_ = 0, t_a_ = 0, t_b_ = 0;
  double gpu_ms_[8] = {};

  id<MTLDevice> dev_ = nil;
  id<MTLCommandQueue> q_ = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pso_;
  id<MTLComputePipelineState> cur_ = nil;

  // model tables
  Buf locbase_, inv_off_, inv_, actm_, Lloc_, Uloc_, Lq_, Uq_, kshift_, kmask_, tdbm_, Lx_, Ux_;
  // passed set and arena
  Buf arena_, arena2_, alive_, alive2_, remap_;
  Buf base_k_, base_a_, base2_k_, base2_a_, delta_k_, delta_a_, delta2_k_, delta2_a_;
  std::size_t acount_ = 0, nbase_ = 0, ndelta_ = 0;
  std::size_t nlive_ = 1, ndisc_ = 1;   // live zones, distinct discrete states (the initial one)
  bool extrap_ = false;
  // frontier, and the next one as it accumulates
  Buf fk_, fa_, fnk_, fna_;
  std::size_t nf_ = 0, nfn_ = 0;
  // recipes
  Buf rkey_, rflags_, rg_off_, rg_, rr_off_, rr_;
  std::vector<RecipeTable> tables_;
  std::vector<std::size_t> runs_;
  std::vector<std::uint32_t> run_of_, run_beg_, run_cnt_, run_lbeg_;
  std::vector<int> run_tid_;
  // one batch
  Buf wf_, wr_, bz_, sk0_, sk1_, sv0_, sv1_, ghist_, head_, hpre_, bstart_, bkey_, binfo_, rowcnt_,
      rowoff_, keep_, kpre_, keep1_, k1pre_, surv_, new_k_, new_a_, ctr_, nbuf_;
  Buf fl_a_, fl_ap_, fl_b_, fl_bp_;
  std::vector<Buf> scan_sums_;
};

}  // namespace

int frontier_reach_metal(const FrontierProblem& p) {
  @autoreleasepool {
    try {
      FrontierMetal m(p);
      return m.run();
    } catch (const std::exception& e) {
      std::fprintf(stderr, "warning: %s; falling back to the CPU engine\n", e.what());
      return -1;
    }
  }
}

}  // namespace detail
}  // namespace hora

#endif  // HORA_HAVE_METAL
