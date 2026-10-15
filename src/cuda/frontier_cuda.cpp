// frontier_cuda.cpp -- the frontier-batched reachability engine on CUDA.
//
// A port of src/metal/frontier_metal.mm. Each BFS level, the frontier is cut into chunks; for
// each chunk the host enumerates the discrete half of every transition (recipes) while the GPU
// expands the previous chunk, and each batch of work items runs as one stream of kernels:
//
//   expand     one warp per (frontier zone, transition): the successor zone, the target test,
//              a sort key -- the successor's discrete key, or a sentinel when disabled
//   sort       stable radix sort of (key, work item)                         (cub)
//   group      buckets of equal keys, and each bucket's stored zones in the passed set
//   reduce     candidates against stored zones, then all pairs among the survivors
//   union      the CPU engine's convex-union merge
//   intern     the survivors' zones into the arena, each distinct zone stored once
//   place      a passed-set entry per survivor, merged into the delta
//
// Where Metal read the sizes a batch produced through indirect dispatch, the launches here are
// sized by the batch's upper bound and every excess thread exits at once; the batch's counters
// come back to the host in one copy when the stream has drained.
//
// The passed set is a sorted *base* of (key, entry) plus a sorted *delta* receiving each batch's
// survivors, folded into the base once it outgrows a fraction of it. An entry names its zone
// through zref; zones live in a paged arena, are reference-counted, and a zone no live entry names
// is recycled. All device buffers are plain device memory; the host reaches them by explicit copy.
#ifdef HORA_HAVE_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
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

#include "hora/detail/engine_config.hpp"
#include "hora/detail/frontier_gpu.hpp"
#include "hora/detail/semantics.hpp"
#include "hora/zone.hpp"
#include "src/cuda/frontier_cuda_api.hpp"

namespace hora {
namespace detail {
namespace {

namespace cu = hora::cuda;

constexpr int kMaxDim = 64;                   // clocks <= 63: one bit each in a 64-bit mask
constexpr std::uint32_t kNoHit = UINT32_MAX;
constexpr std::size_t kMaxPages = 4096;
constexpr std::size_t kPageBytes = std::size_t(256) << 20;
constexpr std::size_t kChunks = 4;
constexpr std::size_t kMinChunkSlots = 16384;
// Stored zones take the narrowest entry that the model's largest constant M suggests; the expand
// kernel flags any zone that does not fit, and the search restarts one width up.
constexpr std::int32_t kByteMaxConst = 30;
constexpr std::int32_t kShortMaxConst = 4000;
struct Overflow {};
// A deadlock search whose D refinement just grew: the stored covers were made under smaller bounds.
struct Restart {};
// A deadlock seed carried a diagonal atom, which a clock bound cannot express: take the a_M route.
struct Collapse {};

/// State a deadlock search keeps across its restarts: the keys already seeded and what the
/// refinement has accumulated. Bounds only ever grow, so the search restarts finitely often.
struct DlState {
  std::unordered_map<std::uint64_t, char> seeded;   ///< key -> may some valuation be stuck there
  std::vector<std::int32_t> bump;                   ///< per clock; kAluNegInf = no atom
  bool collapsed = false;                           ///< a_M bounds instead of the D refinement
  bool grew = false;                                ///< set within one run
};

struct OutOfMemory : std::runtime_error { using std::runtime_error::runtime_error; };

enum Ctr { kValid = 0, kHit = 1, kNewKeys = 2, kKilled = 3, kWide = 4, kFreed = 5, kBig1 = 6, kBig2 = 7,
           kUnion = 8, kErr = 9, kSumKept = 16, kSumFresh = 17, kSumRows = 18, kSumBuckets = 19, kNCtr = 20 };

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

#define FR_CHECK(c, msg) do { if (!(c)) throw std::runtime_error(std::string("CUDA frontier: ") + (msg)); } while (0)
#define CK(call)                                                                                   \
  do {                                                                                             \
    const cudaError_t e_ = (call);                                                                 \
    if (e_ != cudaSuccess)                                                                         \
      throw std::runtime_error(std::string("CUDA frontier: ") + #call + ": " + cudaGetErrorString(e_)); \
  } while (0)

// A device buffer that grows.
struct Buf {
  void* p = nullptr;
  std::size_t cap = 0;
  template <class T> T* as() const { return static_cast<T*>(p); }
  std::size_t bytes() const { return cap; }
};
// A pinned host buffer, the staging side of an upload.
struct Pin {
  void* p = nullptr;
  std::size_t cap = 0;
  template <class T> T* as() const { return static_cast<T*>(p); }
  void reserve(std::size_t bytes) {
    bytes = std::max<std::size_t>(bytes, 64);
    if (cap >= bytes) return;
    if (p) cudaFreeHost(p);
    const std::size_t want = std::max(bytes, cap + cap / 2);
    p = nullptr;
    if (cudaHostAlloc(&p, want, cudaHostAllocDefault) != cudaSuccess) {
      cudaGetLastError();
      throw OutOfMemory("out of host memory: could not pin a staging buffer");
    }
    cap = want;
  }
  ~Pin() { if (p) cudaFreeHost(p); }
};

class FrontierCuda {
 public:
  FrontierCuda(const FrontierProblem& p, int min_bits, DlState* dl = nullptr)
      : P_(p), net_(*p.net), min_bits_(min_bits), dl_(dl) {}
  ~FrontierCuda() {
    for (Buf* b : all_bufs_()) release(*b);
    for (void* pg : pages_) cudaFree(pg);
    for (auto& s : scan_sums_) release(s);
    for (auto& c : chunks_)
      for (Mirror* m : c.all()) release(m->d);
    if (stream_) cudaStreamDestroy(stream_);
  }
  int bits() const { return gz_ * 8; }

  int run() {
    if (!setup()) return -1;
    return search();
  }

 private:
  struct Mirror { Buf d; Pin h; std::size_t used = 0; };
  struct Chunk {
    Mirror rkey, rflags, rg_off, rg, rr_off, rr, rv_off, rv, rdead, wf, wr;
    std::size_t W = 0;
    std::vector<Mirror*> all() {
      return {&rkey, &rflags, &rg_off, &rg, &rr_off, &rr, &rv_off, &rv, &rdead, &wf, &wr};
    }
  };

  // ---------------------------------------------------------------- set-up
  bool setup() {
    n_ = net_.nclocks + 1;
    nn_ = n_ * n_;
    np_ = net_.nprocs();
    if (n_ > kMaxDim || np_ < 1) return false;
    if (!net_.active.empty() && net_.active_mask.empty()) return false;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { cudaGetLastError(); return false; }
    CK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    dev_name_ = prop.name;
    CK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    hc_pin_.reserve(kNCtr * 4);
    hc_ = hc_pin_.as<std::uint32_t>();
    std::size_t freeb = 0, totalb = 0;
    CK(cudaMemGetInfo(&freeb, &totalb));
    mem_cap_ = static_cast<std::size_t>(double(freeb) * 0.92);
    if (const char* m = std::getenv("HORA_GPU_MEM_GB"))
      mem_cap_ = std::min<std::size_t>(mem_cap_, static_cast<std::size_t>(std::atof(m) * double(1ull << 30)));

    // Largest constant anywhere: decides the stored entry width.
    std::int32_t maxc = 0;
    const bool graw = P_.gsim;
    // The covering's bounds: integer a_LU constants, or the G-simulation's raw bounds.
    const auto& LL = graw ? P_.Lgloc : P_.Lloc;
    const auto& UL = graw ? P_.Ugloc : P_.Uloc;
    const auto& LQ = graw ? P_.Lgq : P_.Lq;
    const auto& UQ = graw ? P_.Ugq : P_.Uq;
    auto see = [&](std::int32_t v) {
      if (v == kAluNegInf || v == kGNone) return;
      maxc = std::max(maxc, std::abs(graw ? raw_bound(v) : v));
    };
    for (const auto& pl : LL) for (const auto& l : pl) for (auto v : l) see(v);
    for (const auto& pl : UL) for (const auto& l : pl) for (auto v : l) see(v);
    for (auto v : LQ) see(v);
    for (auto v : UQ) see(v);
    for (const auto& pr : net_.procs) {
      for (const auto& l : pr.locations) for (const auto& c : l.invariant) see(raw_bound(c.bound));
      for (const auto& e : pr.edges) for (const auto& c : e.clock_guard) see(raw_bound(c.bound));
    }
    for (const auto& c : P_.target->clock_cond) see(raw_bound(c.bound));
    int bits = maxc <= kByteMaxConst ? 8 : maxc < kShortMaxConst ? 16 : 32;
    if (const char* z = std::getenv("HORA_GPU_ZONES")) bits = std::atoi(z);
    bits = std::max(bits, min_bits_);
    gz_ = bits == 8 ? 1 : bits == 16 ? 2 : 4;
    page_shift_ = 10;
    while (page_shift_ < 24 && (std::size_t(2) << page_shift_) * nn_ * gz_ <= kPageBytes) ++page_shift_;
    page_zones_ = std::size_t(1) << page_shift_;

    solo_ = 8;
    union_ = !(std::getenv("HORA_GPU_UNION") && std::string(std::getenv("HORA_GPU_UNION")) == "0");
    if (const char* v = std::getenv("HORA_GPU_SOLO")) solo_ = std::atoi(v);
    antichain_ = std::getenv("HORA_GPU_ANTICHAIN") && std::string(std::getenv("HORA_GPU_ANTICHAIN")) == "1";

    cu::Dims dims{n_, nn_, np_, page_shift_, int(detail::kMergeScan), P_.gsim ? 1 : 0};
    cu::init_kernels(0, 0);
    cu::set_dims(dims);
    sgs_expand_ = cu::max_sgs(std::size_t(nn_ + n_) * 4);
    sgs_red_ = cu::max_sgs(std::size_t(nn_) * 4);
    sgs_union_ = cu::max_sgs(std::size_t(2 * nn_) * 4);
    FR_CHECK(std::size_t(2 * nn_) * 4 <= std::size_t(48) * 1024 * 3, "zone too large for shared memory");

    // Static model tables.
    std::vector<int> locbase(np_ + 1, 0);
    for (int p = 0; p < np_; ++p) locbase[p + 1] = locbase[p] + int(net_.procs[p].locations.size());
    const int nloc = locbase[np_];
    const int none = P_.gsim ? int(kGNone) : int(kAluNegInf);
    std::vector<int> Lloc(std::size_t(nloc) * n_, none), Uloc(std::size_t(nloc) * n_, none);
    for (int p = 0; p < np_; ++p)
      for (int l = 0; l < int(net_.procs[p].locations.size()); ++l) {
        const int gl = locbase[p] + l;
        for (int c = 0; c < n_; ++c) {
          if (!LL.empty()) Lloc[std::size_t(gl) * n_ + c] = LL[p][l][c];
          if (!UL.empty()) Uloc[std::size_t(gl) * n_ + c] = UL[p][l][c];
        }
      }
    std::vector<int> kshift(P_.keys.shift.begin(), P_.keys.shift.end());
    std::vector<std::uint64_t> kmask(P_.keys.mask.begin(), P_.keys.mask.end());
    std::vector<int> Lq(LQ.begin(), LQ.end()), Uq(UQ.begin(), UQ.end());

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
    reserve(ptab_, kMaxPages * 8, false);

    for (int c = 1; c <= net_.nclocks; ++c) {
      clockbits_ |= 1ull << c;
      if (c < int(P_.qmask.size()) && P_.qmask[c]) qbits_ |= 1ull << c;
    }
    key_bits_ = P_.key_bits;
    sentinel_ = 1ull << key_bits_;

    // Batch budget: the per-work-item buffers of one batch take at most a quarter of the memory
    // budget, and at most 4 GB; the passed set and zone store have the rest.
    const std::size_t budget = std::min<std::size_t>(mem_cap_ / 4, std::size_t(4) << 30);
    const std::size_t per_item = std::size_t(nn_) * gz_ + std::size_t(2 * n_) * 4 + 150;
    wmax_ = std::max<std::size_t>(1024, budget / per_item);
    wmax_ = std::min<std::size_t>(wmax_, std::size_t(UINT32_MAX / 2));
#ifdef HORA_HAVE_OPENMP
    // Enumeration threads. On a machine whose every core is busy, one descheduled worker stalls each
    // barrier of the enumeration, so by default leave a few cores free (HORA_GPU_THREADS overrides).
    nthreads_ = std::min(omp_get_max_threads(), 16);
    if (const char* t = std::getenv("HORA_GPU_THREADS")) nthreads_ = std::max(1, std::atoi(t));
#endif
    scan_sums_.resize(8);
    trace_ = std::getenv("HORA_GPU_TRACE") != nullptr;
    stress_ = std::getenv("HORA_GPU_STRESS") != nullptr;
    if (stress_) wmax_ = 256;
    if (trace_)
      std::fprintf(stderr, "[gpu] %s | dim %d, %d procs, key %d bits, %s zones (%zu a page), "
                   "batch <= %zu items, memory budget %.1f GB, %u/%u/%u warps\n",
                   dev_name_.c_str(), n_, np_, key_bits_,
                   gz_ == 1 ? "8-bit" : gz_ == 2 ? "16-bit" : "32-bit", page_zones_, wmax_,
                   double(mem_cap_) / double(1ull << 30), sgs_expand_, sgs_red_, sgs_union_);
    return true;
  }

  // ---------------------------------------------------------------- search
  int search() {
    const std::uint64_t k0 = P_.keys.encode(P_.s0.locs, P_.s0.ints);
    ensure_zones(1);
    const std::uint32_t h0 = store_zone(0, P_.s0.zone.data().data());
    reserve(zhash_, 1024 * 4, false); h2d(zhash_.p, &h0, 4);
    reserve(ztslot_, 1024 * 4, false);
    const std::uint32_t one = 1, zero = 0;
    reserve(zcnt_, 1024 * 4, false); h2d(zcnt_.p, &one, 4);
    reserve(zref_, 1024 * 4, false); h2d(zref_.p, &zero, 4);
    reserve(freel_, 1024 * 4, false); reserve(pend_, 1024 * 4, false);
    zcount_ = 1;
    reserve(ctr_, 128, false);
    CK(cudaMemsetAsync(ctr_.p, 0, 128, stream_));
    ensure_table(1024);
    reserve(alive_, 1024 * 4, false); h2d(alive_.p, &one, 4);
    reserve(base_k_, 1024 * 8, false); reserve(base_a_, 1024 * 4, false);
    h2d(base_k_.p, &k0, 8); h2d(base_a_.p, &zero, 4);
    reserve(delta_k_, 1024 * 8, false); reserve(delta_a_, 1024 * 4, false);
    reserve(fk_, 1024 * 8, false); reserve(fa_, 1024 * 4, false);
    h2d(fk_.p, &k0, 8); h2d(fa_.p, &zero, 4);
    fk_h_.assign(1, k0);
    acount_ = 1; nbase_ = 1; ndelta_ = 0; nf_ = 1;

    while (nf_ > 0) {
      if (P_.cancel && P_.cancel->load(std::memory_order_relaxed)) return 0;
      if (P_.deadlock && deadlock_level()) return 1;
      if (P_.stats) P_.stats->explored += nf_;
      nf_prev_ = nf_;
      t_enum_ = t_wait_ = 0;
      lvl_w_ = lvl_valid_ = lvl_kept_ = lvl_rows_ = lvl_buckets_ = 0;
      nfn_ = 0;
      const std::uint64_t* fk = fk_h_.data();
      runs_.clear();
      for (std::size_t i = 0; i < nf_; ++i)
        if (i == 0 || fk[i] != fk[i - 1]) runs_.push_back(i);
      runs_.push_back(nf_);                         // sentinel: runs_[r + 1] ends run r
      const std::size_t nr = runs_.size() - 1;
      run_tid_.resize(nr); run_lbeg_.resize(nr); run_cnt_.resize(nr); run_beg_.resize(nr);
      const std::size_t chunks = stress_ ? 16 : kChunks;
      const std::size_t chunk_slots =
          std::max<std::size_t>(stress_ ? 1 : kMinChunkSlots, (nf_ + chunks - 1) / chunks);
      std::vector<std::size_t> cuts{0};
      for (std::size_t r = 0; r < nr; ++r)
        if (runs_[r + 1] - runs_[cuts.back()] >= chunk_slots || r + 1 == nr) cuts.push_back(r + 1);
      prepare(cuts[0], cuts[1], chunks_[0]);
      for (std::size_t c = 0; c + 1 < cuts.size(); ++c) {
        Chunk& ch = chunks_[c & 1];
        const bool more = c + 2 < cuts.size();
        for (std::size_t w0 = 0; w0 < ch.W; w0 += wmax_) {
          const std::size_t w1 = std::min(ch.W, w0 + wmax_);
          encode_batch(ch, w0, w1 - w0);
          if (more && w1 == ch.W) prepare(cuts[c + 1], cuts[c + 2], chunks_[(c + 1) & 1]);
          const auto tw = Clock::now();
          wait();
          t_wait_ += ms_since(tw);
          if (finish_batch(w1 - w0)) return 1;
        }
        if (more && ch.W == 0) prepare(cuts[c + 1], cuts[c + 2], chunks_[(c + 1) & 1]);
        if (P_.cancel && P_.cancel->load(std::memory_order_relaxed)) return 0;
      }
      const auto t_end = Clock::now();
      end_level();
      // Extra_LU+ can invent and destroy a deadlock, so a deadlock search never widens.
      if (!extrap_ && !P_.deadlock && ndisc_ && double(nlive_) >= P_.adapt_ratio * double(ndisc_)) {
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

  // ---------------------------------------------------------------- deadlock
  // Test the frontier -- the zones stored at the last level, not yet expanded -- for a deadlocked
  // valuation, and grow the D refinement over its newly seen discrete keys. True on a deadlock.
  // Throws Restart when the refinement grew (every stored cover was made under smaller bounds) and
  // Collapse when it needs a diagonal atom.
  bool deadlock_level() {
    const DeadlockHooks& H = *P_.deadlock;
    const std::uint64_t* fk = fk_h_.data();
    std::vector<std::uint64_t> fresh;
    for (std::size_t i = 0; i < nf_; ++i)
      if ((i == 0 || fk[i] != fk[i - 1]) && !dl_->seeded.count(fk[i])) {
        dl_->seeded.emplace(fk[i], 1);   // placeholder: also dedupes a key that recurs in this level
        fresh.push_back(fk[i]);
      }
    if (!fresh.empty()) {
      std::vector<DeadlockSeed> res(fresh.size());
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 16) num_threads(nthreads_)
#endif
      for (long i = 0; i < long(fresh.size()); ++i) res[i] = H.seed(fresh[i]);
      for (std::size_t i = 0; i < fresh.size(); ++i) {
        dl_->seeded[fresh[i]] = res[i].may_deadlock ? 1 : 0;
        if (dl_->collapsed) continue;               // a_M needs no atoms
        if (res[i].diagonal) throw Collapse{};
        for (const auto& b : res[i].bumps)
          if (b.first > 0 && b.first < int(dl_->bump.size()) && b.second > dl_->bump[b.first]) {
            dl_->bump[b.first] = b.second;
            dl_->grew = true;
          }
      }
    }

    // The frontier zones at keys that can be stuck at all.
    std::vector<std::uint32_t> ent(nf_);
    if (nf_) CK(cudaMemcpy(ent.data(), fa_.p, nf_ * 4, cudaMemcpyDeviceToHost));
    std::vector<std::uint32_t> pick;     // frontier slots to test
    for (std::size_t i = 0; i < nf_; ++i) {
      const auto it = dl_->seeded.find(fk[i]);
      if (it != dl_->seeded.end() && it->second) pick.push_back(std::uint32_t(i));
    }
    const std::size_t zbytes = std::size_t(nn_) * gz_;
    const std::size_t cz = std::max<std::size_t>(1, (std::size_t(128) << 20) / zbytes);
    std::vector<std::uint32_t> sel;
    std::vector<unsigned char> raw;
    for (std::size_t c0 = 0; c0 < pick.size(); c0 += cz) {
      const std::size_t c1 = std::min(pick.size(), c0 + cz), m = c1 - c0;
      sel.resize(m);
      for (std::size_t j = 0; j < m; ++j) sel[j] = ent[pick[c0 + j]];
      reserve(dsel_, m * 4, false);
      reserve(dzone_, m * zbytes, false);
      h2d(dsel_.p, sel.data(), m * 4);
      cu::gather_zones(gz_, reinterpret_cast<cu::Pages>(ptab_.p), zref_.as<std::uint32_t>(), dsel_.as<std::uint32_t>(), std::uint32_t(m), dzone_.p, stream_);
      raw.resize(m * zbytes);
      CK(cudaMemcpyAsync(raw.data(), dzone_.p, m * zbytes, cudaMemcpyDeviceToHost, stream_));
      wait();
      std::atomic<bool> found{false};
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel num_threads(nthreads_)
#endif
      {
        std::vector<std::int32_t> z(nn_);
#ifdef HORA_HAVE_OPENMP
#pragma omp for schedule(dynamic, 8)
#endif
        for (long j = 0; j < long(m); ++j) {
          if (found.load(std::memory_order_relaxed)) continue;
          const unsigned char* src = raw.data() + std::size_t(j) * zbytes;
          for (int k = 0; k < nn_; ++k)
            z[k] = gz_ == 1 ? zget(reinterpret_cast<const sraw_t*>(src), k)
                 : gz_ == 2 ? zget(reinterpret_cast<const nraw_t*>(src), k)
                            : zget(reinterpret_cast<const raw_t*>(src), k);
          if (H.has_deadlock(fk[pick[c0 + j]], z.data())) found.store(true, std::memory_order_relaxed);
        }
      }
      if (found.load()) return true;
    }
    if (dl_->grew) throw Restart{};
    return false;
  }

  // Enumerate the recipes of runs [ra, rb) of the frontier into `ch`, and its work items: every
  // frontier slot of those runs times every recipe of its key. Staged in pinned memory, then
  // uploaded on the stream behind the kernels already queued.
  void prepare(std::size_t ra, std::size_t rb, Chunk& ch) {
    const auto t0 = Clock::now();
    const std::uint64_t* fk = fk_h_.data();
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
    std::vector<std::size_t> rb0(nthreads_ + 1, 0), gb(nthreads_ + 1, 0), xb(nthreads_ + 1, 0),
        vb(nthreads_ + 1, 0);
    for (int t = 0; t < nthreads_; ++t) {
      rb0[t + 1] = rb0[t] + tables_[t].size();
      gb[t + 1] = gb[t] + tables_[t].g.size() / 3;
      xb[t + 1] = xb[t] + tables_[t].r.size();
      vb[t + 1] = vb[t] + tables_[t].v.size() / 3;
    }
    const std::size_t R = rb0[nthreads_];
    const std::size_t nG = std::max<std::size_t>(1, gb[nthreads_]), nX = std::max<std::size_t>(1, xb[nthreads_]),
                      nV = std::max<std::size_t>(1, vb[nthreads_]), nR = std::max<std::size_t>(1, R);
    stage(ch.rkey, nR * 8); stage(ch.rflags, nR * 4); stage(ch.rg_off, (R + 1) * 4);
    stage(ch.rr_off, (R + 1) * 4); stage(ch.rg, nG * 12); stage(ch.rr, nX * 4);
    stage(ch.rv_off, (R + 1) * 4); stage(ch.rv, nV * 12); stage(ch.rdead, nR * 8);
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel for schedule(static, 1) num_threads(nthreads_)
#endif
    for (int t = 0; t < nthreads_; ++t) {
      const RecipeTable& T = tables_[t];
      if (T.size()) {
        std::memcpy(ch.rkey.h.as<std::uint64_t>() + rb0[t], T.key.data(), T.size() * 8);
        std::memcpy(ch.rflags.h.as<std::uint32_t>() + rb0[t], T.flags.data(), T.size() * 4);
        std::memcpy(ch.rdead.h.as<std::uint64_t>() + rb0[t], T.dead.data(), T.size() * 8);
      }
      if (!T.g.empty()) std::memcpy(ch.rg.h.as<std::int32_t>() + gb[t] * 3, T.g.data(), T.g.size() * 4);
      if (!T.r.empty()) std::memcpy(ch.rr.h.as<std::int32_t>() + xb[t], T.r.data(), T.r.size() * 4);
      if (!T.v.empty()) std::memcpy(ch.rv.h.as<std::int32_t>() + vb[t] * 3, T.v.data(), T.v.size() * 4);
      std::uint32_t* go = ch.rg_off.h.as<std::uint32_t>();
      std::uint32_t* ro = ch.rr_off.h.as<std::uint32_t>();
      std::uint32_t* vo = ch.rv_off.h.as<std::uint32_t>();
      for (std::size_t r = 0; r < T.size(); ++r) {
        go[rb0[t] + r] = std::uint32_t(gb[t] + T.g_off[r]);
        ro[rb0[t] + r] = std::uint32_t(xb[t] + T.r_off[r]);
        vo[rb0[t] + r] = std::uint32_t(vb[t] + T.v_off[r]);
      }
    }
    ch.rg_off.h.as<std::uint32_t>()[R] = std::uint32_t(gb[nthreads_]);
    ch.rr_off.h.as<std::uint32_t>()[R] = std::uint32_t(xb[nthreads_]);
    ch.rv_off.h.as<std::uint32_t>()[R] = std::uint32_t(vb[nthreads_]);
    std::vector<std::size_t> woff(rb - ra + 1, 0);
    for (std::size_t r = ra; r < rb; ++r) {
      run_beg_[r] = std::uint32_t(rb0[run_tid_[r]] + run_lbeg_[r]);
      woff[r - ra + 1] = woff[r - ra] + (runs_[r + 1] - runs_[r]) * run_cnt_[r];
    }
    ch.W = woff.back();
    const std::size_t nW = std::max<std::size_t>(1, ch.W);
    stage(ch.wf, nW * 4); stage(ch.wr, nW * 4);
    std::uint32_t* wf = ch.wf.h.as<std::uint32_t>();
    std::uint32_t* wr = ch.wr.h.as<std::uint32_t>();
#ifdef HORA_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(nthreads_)
#endif
    for (long r = long(ra); r < long(rb); ++r) {
      std::size_t w = woff[r - ra];
      for (std::size_t f = runs_[r]; f < runs_[r + 1]; ++f)
        for (std::uint32_t t = 0; t < run_cnt_[r]; ++t) { wf[w] = std::uint32_t(f); wr[w] = run_beg_[r] + t; ++w; }
    }
    for (Mirror* m : ch.all()) upload(*m);
    lvl_w_ += ch.W;
    t_enum_ += ms_since(t0);
  }

  // Enqueue one batch -- work items [w0, w0 + W) of chunk `ch` -- on the stream. Every size that
  // depends on the batch's own results is bounded by W here and read on the device; finish_batch()
  // reads the outcome once the stream has drained.
  void encode_batch(Chunk& ch, std::size_t w0, std::size_t W) {
    const std::uint32_t Wu = std::uint32_t(W);
    reserve(bz_, W * nn_ * gz_, false);
    for (Buf* b : {&sk0_, &sk1_}) reserve(*b, W * 8, false);
    for (Buf* b : {&sv0_, &sv1_}) reserve(*b, W * 4, false);
    reserve(head_, (W + 1) * 4, false); reserve(hpre_, (W + 1) * 4, false);
    reserve(bstart_, W * 4, false); reserve(bkey_, W * 8, false); reserve(binfo_, W * 32, false);
    reserve(rowcnt_, (W + 1) * 4, false); reserve(rowoff_, (W + 1) * 4, false);
    reserve(Lb_, W * n_ * 4, false); reserve(Ub_, W * n_ * 4, false);
    reserve(big1_, W * 4, false); reserve(big2_, (W + nbase_ + ndelta_ + 1) * 4, false); reserve(ulist_, W * 4, false);   // big2_: rows = new survivors plus stored zones
    reserve(keep_, (W + 1) * 4, false); reserve(keep1_, (W + 1) * 4, false);
    reserve(k1pre_, (W + 1) * 4, false); reserve(kpre_, (W + 1) * 4, false);
    reserve(surv_, W * 4, false);
    reserve(new_k_, W * 8, false); reserve(new_a_, W * 4, false);
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
    const std::size_t sort_need = cu::sort_pairs(nullptr, 0, nullptr, nullptr, nullptr, nullptr, Wu,
                                                 key_bits_ + 1, stream_);
    reserve(sort_tmp_, sort_need, false);
    std::uint32_t init[10] = {0, kNoHit, 0, 0, 0, std::uint32_t(freed_mirror_), 0, 0, 0, 0};
    h2d_async(ctr_.p, init, sizeof(init));
    freed_before_ = std::uint32_t(freed_mirror_);

    const int gz = gz_;
    cu::Pages AP = reinterpret_cast<cu::Pages>(ptab_.p);
    // ---- expand ----
    {
      cu::ExpandP e{};
      e.wf = ch.wf.d.as<std::uint32_t>() + w0;
      e.wr = ch.wr.d.as<std::uint32_t>() + w0;
      e.fz = fa_.as<std::uint32_t>();
      e.AP = AP;
      e.rkey = ch.rkey.d.as<std::uint64_t>();
      e.rflags = ch.rflags.d.as<std::uint32_t>();
      e.rg_off = ch.rg_off.d.as<std::uint32_t>();
      e.rg = ch.rg.d.as<int>();
      e.rr_off = ch.rr_off.d.as<std::uint32_t>();
      e.rr = ch.rr.d.as<int>();
      e.rv_off = ch.rv_off.d.as<std::uint32_t>();
      e.rv = ch.rv.d.as<int>();
      e.rdead = ch.rdead.d.as<std::uint64_t>();
      e.tdbm = tdbm_.as<int>();
      e.bz = bz_.p;
      e.skey = sk0_.as<std::uint64_t>();
      e.sval = sv0_.as<std::uint32_t>();
      e.ctr = ctr_.as<std::uint32_t>();
      e.Lx = Lx_.as<int>();
      e.Ux = Ux_.as<int>();
      e.zref = zref_.as<std::uint32_t>();
      e.hshw = hsh_.as<std::uint32_t>();
      e.nwork = Wu;
      e.has_tclock = has_tclock_ ? 1 : 0;
      e.extrap = extrap_ ? 1 : 0;
      e.sentinel = sentinel_;
      cu::expand(gz, e, sgs_expand_, stream_);
    }
    // ---- sort by discrete key ----
    cu::sort_pairs(sort_tmp_.p, sort_tmp_.cap, sk0_.as<std::uint64_t>(), sk1_.as<std::uint64_t>(),
                   sv0_.as<std::uint32_t>(), sv1_.as<std::uint32_t>(), Wu, key_bits_ + 1, stream_);
    Buf* sk = &sk1_;
    Buf* sv = &sv1_;
    // ---- group into buckets, and find each bucket's stored zones ----
    cu::heads(sk->as<std::uint64_t>(), head_.as<std::uint32_t>(), ctr_.as<std::uint32_t>(), Wu,
              keep1_.as<std::uint32_t>(), keep_.as<std::uint32_t>(), stream_);
    scan(head_, hpre_, Wu, Wu + 1);
    cu::buckets(sk->as<std::uint64_t>(), head_.as<std::uint32_t>(), hpre_.as<std::uint32_t>(),
                bstart_.as<std::uint32_t>(), bkey_.as<std::uint64_t>(), Wu, stream_);
    cu::bucket_setup(bkey_.as<std::uint64_t>(), bstart_.as<std::uint32_t>(), hpre_.as<std::uint32_t>(),
                     ctr_.as<std::uint32_t>(), base_k_.as<std::uint64_t>(), delta_k_.as<std::uint64_t>(),
                     binfo_.p, rowcnt_.as<std::uint32_t>(), Wu, std::uint32_t(nbase_), std::uint32_t(ndelta_),
                     stream_);
    cu::bucket_bounds(bkey_.as<std::uint64_t>(), hpre_.as<std::uint32_t>(), Lloc_.as<int>(), Uloc_.as<int>(),
                      Lq_.as<int>(), Uq_.as<int>(), locbase_.as<int>(), kshift_.as<int>(),
                      kmask_.as<std::uint64_t>(), Lb_.as<int>(), Ub_.as<int>(), Wu, stream_);
    // ---- reduce ----
    cu::ReduceP R{};
    R.hpre = hpre_.as<std::uint32_t>();
    R.head = head_.as<std::uint32_t>();
    R.binfo = binfo_.p;
    R.Lb = Lb_.as<int>();
    R.Ub = Ub_.as<int>();
    R.base_a = base_a_.as<std::uint32_t>();
    R.delta_a = delta_a_.as<std::uint32_t>();
    R.AP = AP;
    R.alive = alive_.as<std::uint32_t>();
    R.sval = sv->as<std::uint32_t>();
    R.bz = bz_.p;
    R.keep1 = keep1_.as<std::uint32_t>();
    R.keep = keep_.as<std::uint32_t>();
    R.ctr = ctr_.as<std::uint32_t>();
    R.zref = zref_.as<std::uint32_t>();
    R.big1 = big1_.as<std::uint32_t>();
    R.big2 = big2_.as<std::uint32_t>();
    R.k1pre = k1pre_.as<std::uint32_t>();
    R.surv = surv_.as<std::uint32_t>();
    R.rowoff = rowoff_.as<std::uint32_t>();
    R.zcnt = zcnt_.as<std::uint32_t>();
    R.ztab = ztab_.as<std::uint32_t>();
    R.ztslot = ztslot_.as<std::uint32_t>();
    R.pend = pend_.as<std::uint32_t>();
    R.ulist = ulist_.as<std::uint32_t>();
    R.hshw = hsh_.as<std::uint32_t>();
    R.rowcnt = rowcnt_.as<std::uint32_t>();
    R.chg = nullptr;
    R.W = Wu;
    R.solo = solo_;
    cu::reduce_stored(gz, R, stream_);
    cu::reduce_stored_big(gz, R, sgs_red_, stream_);
    scan(keep1_, k1pre_, Wu, Wu + 1);
    cu::survivors(keep1_.as<std::uint32_t>(), k1pre_.as<std::uint32_t>(), surv_.as<std::uint32_t>(),
                  ctr_.as<std::uint32_t>(), Wu, stream_);
    cu::keep_init(keep1_.as<std::uint32_t>(), keep_.as<std::uint32_t>(), ctr_.as<std::uint32_t>(), Wu, stream_);
    cu::rows2(R, stream_);
    scan(rowcnt_, rowoff_, Wu, Wu + 1);
    cu::reduce_pairs(gz, R, stream_);
    cu::reduce_pairs_big(gz, R, sgs_red_, stream_);
    // ---- convex union, as the CPU engine merges ----
    if (union_) {
      if (antichain_) {
        // Which candidates a union replaced; the pass after it re-establishes the antichain.
        reserve(chg_, (std::size_t(Wu) + 1) * 4, false);
        CK(cudaMemsetAsync(chg_.p, 0, (std::size_t(Wu) + 1) * 4, stream_));
        R.chg = chg_.as<std::uint32_t>();
      }
      cu::merge_union(gz, R, sgs_union_, stream_);
      if (antichain_) cu::antichain(gz, R, stream_);
    }
    // ---- intern the survivors' zones ----
    scan(keep_, kpre_, Wu, Wu + 1);
    {
      cu::InternP ip{};
      ip.keep = keep_.as<std::uint32_t>();
      ip.ctr = ctr_.as<std::uint32_t>();
      ip.sval = sv->as<std::uint32_t>();
      ip.bz = bz_.p;
      ip.AP = AP;
      ip.hsh = hsh_.as<std::uint32_t>();
      ip.zhash = zhash_.as<std::uint32_t>();
      ip.ztab = ztab_.as<std::uint32_t>();
      ip.rep = rep_.as<std::uint32_t>();
      ip.zslot = zslot_.as<std::uint32_t>();
      ip.ctr_err = ctr_.as<std::uint32_t>() + kErr;
      ip.W = Wu;
      ip.mask = std::uint32_t(zcap_ - 1);
      cu::intern(gz, ip, stream_);
    }
    cu::zwin(keep_.as<std::uint32_t>(), ctr_.as<std::uint32_t>(), rep_.as<std::uint32_t>(),
             win_.as<std::uint32_t>(), Wu, stream_);
    scan(win_, wpre_, Wu, Wu + 1);
    cu::winlist(win_.as<std::uint32_t>(), wpre_.as<std::uint32_t>(), wlist_.as<std::uint32_t>(), Wu, stream_);
    {
      cu::ZPlaceP zp{};
      zp.wlist = wlist_.as<std::uint32_t>();
      zp.wpre = wpre_.as<std::uint32_t>();
      zp.sval = sv->as<std::uint32_t>();
      zp.bz = bz_.p;
      zp.AP = AP;
      zp.hsh = hsh_.as<std::uint32_t>();
      zp.zhash = zhash_.as<std::uint32_t>();
      zp.ztab = ztab_.as<std::uint32_t>();
      zp.zslot = zslot_.as<std::uint32_t>();
      zp.freel = freel_.as<std::uint32_t>();
      zp.zid = zid_.as<std::uint32_t>();
      zp.ztslot = ztslot_.as<std::uint32_t>();
      zp.zcnt = zcnt_.as<std::uint32_t>();
      zp.W = Wu;
      zp.zcount = std::uint32_t(zcount_);
      zp.nfree = std::uint32_t(nfree_);
      cu::zplace(gz, zp, stream_);
    }
    // ---- place the survivors, and merge them into the delta ----
    cu::place_new(keep_.as<std::uint32_t>(), kpre_.as<std::uint32_t>(), sk->as<std::uint64_t>(),
                  ctr_.as<std::uint32_t>(), new_k_.as<std::uint64_t>(), new_a_.as<std::uint32_t>(),
                  alive_.as<std::uint32_t>(), fnk_.as<std::uint64_t>(), fna_.as<std::uint32_t>(), Wu,
                  std::uint32_t(acount_), std::uint32_t(nfn_), rep_.as<std::uint32_t>(),
                  zid_.as<std::uint32_t>(), zref_.as<std::uint32_t>(), zcnt_.as<std::uint32_t>(), stream_);
    merge(delta_k_, delta_a_, ndelta_, new_k_, new_a_, kpre_.as<std::uint32_t>() + W, W, nullptr, nullptr,
          delta2_k_, delta2_a_);
    cu::summary(ctr_.as<std::uint32_t>(), kpre_.as<std::uint32_t>(), wpre_.as<std::uint32_t>(),
                rowoff_.as<std::uint32_t>(), hpre_.as<std::uint32_t>(), Wu, stream_);
    CK(cudaMemcpyAsync(hc_, ctr_.p, kNCtr * 4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaGetLastError());
  }

  // After the batch's stream completed: bookkeeping, and true on a target hit.
  bool finish_batch(std::size_t) {
    const std::uint32_t nvalid = hc_[kValid], hit = hc_[kHit];
    lvl_valid_ += nvalid;
    if (P_.stats) P_.stats->generated += nvalid;
    if (hit != kNoHit) return true;
    if (hc_[kWide]) throw Overflow{};
    FR_CHECK(hc_[kErr] == 0, "zone table invariant broken (a full table)");
    const std::uint32_t kept = hc_[kSumKept];
    const std::uint32_t fresh = hc_[kSumFresh];   // zones not already stored
    lvl_rows_ += hc_[kSumRows];
    lvl_buckets_ += hc_[kSumBuckets];
    lvl_kept_ += kept;
    nlive_ += kept;
    nlive_ -= hc_[kKilled];
    ndisc_ += hc_[kNewKeys];
    freed_mirror_ = hc_[kFreed];
    ntomb_ += freed_mirror_ - freed_before_;
    std::swap(delta_k_, delta2_k_); std::swap(delta_a_, delta2_a_);
    ndelta_ += kept;
    acount_ += kept;
    const std::size_t recycled = std::min<std::size_t>(fresh, nfree_);
    nfree_ -= recycled;
    zcount_ += fresh - recycled;
    nfn_ += kept;
    if (ndelta_ > std::max<std::size_t>(nbase_ / 4, 1u << 16) || (stress_ && ndelta_)) fold();
    return false;
  }

  // base := merge(live base, live delta); delta := empty.
  void fold() {
    reserve(fl_a_, (nbase_ + 1) * 4, false); reserve(fl_ap_, (nbase_ + 1) * 4, false);
    reserve(fl_b_, (ndelta_ + 1) * 4, false); reserve(fl_bp_, (ndelta_ + 1) * 4, false);
    reserve(base2_k_, (nbase_ + ndelta_) * 8 + 8, false); reserve(base2_a_, (nbase_ + ndelta_) * 4 + 4, false);
    reserve(nbuf_, 16, false);
    const std::uint32_t nd = std::uint32_t(ndelta_);
    h2d_async(nbuf_.p, &nd, 4);
    alive_flags(base_a_, nbase_, fl_a_);
    scan(fl_a_, fl_ap_, std::uint32_t(nbase_), std::uint32_t(nbase_ + 1));
    alive_flags(delta_a_, ndelta_, fl_b_);
    scan(fl_b_, fl_bp_, std::uint32_t(ndelta_), std::uint32_t(ndelta_ + 1));
    merge(base_k_, base_a_, nbase_, delta_k_, delta_a_, nbuf_.as<std::uint32_t>(), ndelta_,
          fl_ap_.as<std::uint32_t>(), fl_bp_.as<std::uint32_t>(), base2_k_, base2_a_);
    std::uint32_t tot[2] = {0, 0};
    CK(cudaMemcpyAsync(&tot[0], fl_ap_.as<std::uint32_t>() + nbase_, 4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(&tot[1], fl_bp_.as<std::uint32_t>() + ndelta_, 4, cudaMemcpyDeviceToHost, stream_));
    wait();
    const std::size_t n = std::size_t(tot[0]) + tot[1];
    std::swap(base_k_, base2_k_); std::swap(base_a_, base2_a_);
    nbase_ = n;
    ndelta_ = 0;
  }

  void end_level() {
    reserve(fl_a_, (nfn_ + 1) * 4, false); reserve(fl_ap_, (nfn_ + 1) * 4, false);
    reserve(fk_, std::max<std::size_t>(1, nfn_) * 8, false);
    reserve(fa_, std::max<std::size_t>(1, nfn_) * 4, false);
    std::uint32_t nlive_next = 0;
    if (nfn_ > 0) {
      alive_flags(fna_, nfn_, fl_a_);
      scan(fl_a_, fl_ap_, std::uint32_t(nfn_), std::uint32_t(nfn_ + 1));
      cu::compact_frontier(fnk_.as<std::uint64_t>(), fna_.as<std::uint32_t>(), fl_a_.as<std::uint32_t>(),
                           fl_ap_.as<std::uint32_t>(), fk_.as<std::uint64_t>(), fa_.as<std::uint32_t>(),
                           std::uint32_t(nfn_), stream_);
      CK(cudaMemcpyAsync(&nlive_next, fl_ap_.as<std::uint32_t>() + nfn_, 4, cudaMemcpyDeviceToHost, stream_));
      wait();
    }
    nf_ = nlive_next;
    if (nf_ > 0) {
      fk_h_.resize(nf_);
      CK(cudaMemcpy(fk_h_.data(), fk_.p, nf_ * 8, cudaMemcpyDeviceToHost));
    }
    // The zones this level freed become reusable now.
    const std::size_t npend = freed_mirror_;
    if (npend) {
      reserve(freel_, (nfree_ + npend) * 4, true);
      CK(cudaMemcpy(freel_.as<std::uint32_t>() + nfree_, pend_.p, npend * 4, cudaMemcpyDeviceToDevice));
      nfree_ += npend;
      const std::uint32_t z = 0;
      CK(cudaMemcpy(ctr_.as<std::uint32_t>() + kFreed, &z, 4, cudaMemcpyHostToDevice));
      freed_mirror_ = 0;
    }
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
    cu::gc_entries(base_a_.as<std::uint32_t>(), zref_.as<std::uint32_t>(), zref2_.as<std::uint32_t>(),
                   remap_.as<std::uint32_t>(), alive2_.as<std::uint32_t>(), std::uint32_t(n), stream_);
    if (nf_) cu::remap(fa_.as<std::uint32_t>(), remap_.as<std::uint32_t>(), std::uint32_t(nf_), stream_);
    wait();
    std::swap(zref_, zref2_); std::swap(alive_, alive2_);
    acount_ = n;
    for (Buf* b : {&zref2_, &alive2_, &remap_}) release(*b);
  }

  // ---------------------------------------------------------------- the zone store
  void ensure_zones(std::size_t nz) {
    while (pages_.size() * page_zones_ < nz) {
      FR_CHECK(pages_.size() < kMaxPages, "zone arena exceeds its page table");
      const std::size_t bytes = page_zones_ * nn_ * gz_;
      charge_mem(bytes, "a zone page");
      void* pg = nullptr;
      if (cudaMalloc(&pg, bytes) != cudaSuccess) {
        cudaGetLastError();
        throw OutOfMemory("out of GPU memory: the system refused a " + mb(bytes) + " zone page");
      }
      mem_used_ += bytes;
      const std::uint64_t addr = reinterpret_cast<std::uint64_t>(pg);
      CK(cudaMemcpy(ptab_.as<std::uint64_t>() + pages_.size(), &addr, 8, cudaMemcpyHostToDevice));
      pages_.push_back(pg);
    }
  }
  void ensure_table(std::size_t W) {
    const std::size_t live = zcount_ - nfree_;
    if (zcap_ && (live + ntomb_ + W) * 2 <= zcap_ && !(stress_ && ntomb_)) return;
    if (!zcap_) zcap_ = 4096;
    while ((live + W) * 2 > zcap_) zcap_ *= 2;
    rebuild_table();
  }
  void rebuild_table() {
    reserve(ztab_, zcap_ * 4, false);
    cu::fill_u32(ztab_.as<std::uint32_t>(), std::uint32_t(zcap_), 0xffffffffu, stream_);
    const std::uint32_t z = 0;
    CK(cudaMemcpyAsync(ctr_.as<std::uint32_t>() + kErr, &z, 4, cudaMemcpyHostToDevice, stream_));
    cu::zrebuild(zhash_.as<std::uint32_t>(), ztab_.as<std::uint32_t>(), std::uint32_t(zcount_),
                 std::uint32_t(zcap_ - 1), zcnt_.as<std::uint32_t>(), ztslot_.as<std::uint32_t>(),
                 ctr_.as<std::uint32_t>() + kErr, stream_);
    std::uint32_t err = 0;
    CK(cudaMemcpyAsync(&err, ctr_.as<std::uint32_t>() + kErr, 4, cudaMemcpyDeviceToHost, stream_));
    wait();
    FR_CHECK(err == 0, "zone table rebuild found the table full");
    ntomb_ = 0;
  }
  static std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
  }
  // The initial zone, stored on the device; returns its hash (the kernels' fr_entry_hash).
  std::uint32_t store_zone(std::size_t z, const raw_t* d) {
    std::vector<unsigned char> bytes(std::size_t(nn_) * gz_);
    std::uint32_t h = 0;
    for (int k = 0; k < nn_; ++k) {
      const bool fits = d[k] >= kInf || (gz_ == 1 ? (d[k] > -127 && d[k] < kSInf)
                                         : gz_ == 2 ? (d[k] > -32767 && d[k] < kNInf) : true);
      if (!fits) throw Overflow{};
      std::int32_t v;
      if (gz_ == 1) { sraw_t t; zput(&t, 0, d[k]); std::memcpy(&bytes[k], &t, 1); v = t; }
      else if (gz_ == 2) { nraw_t t; zput(&t, 0, d[k]); std::memcpy(&bytes[std::size_t(k) * 2], &t, 2); v = t; }
      else { raw_t t = d[k]; std::memcpy(&bytes[std::size_t(k) * 4], &t, 4); v = t; }
      h += mix32(std::uint32_t(k) * 0x9e3779b9u ^ std::uint32_t(v));
    }
    char* dst = static_cast<char*>(pages_[z >> page_shift_]) + (z & (page_zones_ - 1)) * nn_ * gz_;
    CK(cudaMemcpy(dst, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    return mix32(h);
  }

  // ---------------------------------------------------------------- stream helpers
  void wait() {
    CK(cudaStreamSynchronize(stream_));
    CK(cudaGetLastError());
  }
  void h2d(void* dst, const void* src, std::size_t n) { CK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice)); }
  void h2d_async(void* dst, const void* src, std::size_t n) {
    CK(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream_));
  }
  // Staging for an upload: reserve the pinned side and the device side to `bytes`.
  void stage(Mirror& m, std::size_t bytes) {
    m.h.reserve(bytes);
    reserve(m.d, bytes, false);
    m.used = bytes;
  }
  void upload(Mirror& m) { CK(cudaMemcpyAsync(m.d.p, m.h.p, m.used, cudaMemcpyHostToDevice, stream_)); }

  // Exclusive scan of in[0 .. n_in) into out[0 .. n_out); in-place is allowed.
  void scan(const Buf& in, const Buf& out, std::uint32_t n_in, std::uint32_t n_out, std::size_t level = 0) {
    if (n_out == 0) return;
    FR_CHECK(level < scan_sums_.size(), "scan nests too deep");
    const std::uint32_t tile = 1024;   // four elements for each of 256 threads
    const std::uint32_t nblk = (n_out + tile - 1) / tile;
    reserve(scan_sums_[level], std::size_t(nblk) * 4, false);
    cu::scan_block(in.as<std::uint32_t>(), out.as<std::uint32_t>(), scan_sums_[level].as<std::uint32_t>(), n_in,
                   n_out, tile, nblk, stream_);
    if (nblk > 1) {
      scan(scan_sums_[level], scan_sums_[level], nblk, nblk, level + 1);
      cu::scan_add(out.as<std::uint32_t>(), scan_sums_[level].as<std::uint32_t>(), n_out, tile, stream_);
    }
  }
  void alive_flags(const Buf& idx, std::size_t n, const Buf& flag) {
    cu::alive_flags(idx.as<std::uint32_t>(), alive_.as<std::uint32_t>(), flag.as<std::uint32_t>(),
                    std::uint32_t(n), stream_);
  }
  // out = merge(a[0..na), b[0..*nbp)), with *nbp read on the device; when `apre` / `bpre` are given,
  // only entries whose flag is set survive (their prefix sums).
  void merge(const Buf& ak, const Buf& aa, std::size_t na, const Buf& bk, const Buf& ba, const std::uint32_t* nbp,
             std::size_t nb_cap, const std::uint32_t* apre, const std::uint32_t* bpre, const Buf& ok,
             const Buf& oa) {
    const std::uint32_t filter = apre ? 1u : 0u;
    const std::uint32_t* ap = apre ? apre : nbp;
    const std::uint32_t* bp = bpre ? bpre : nbp;
    cu::merge_a(ak.as<std::uint64_t>(), aa.as<std::uint32_t>(), bk.as<std::uint64_t>(), nbp, ap, bp,
                ok.as<std::uint64_t>(), oa.as<std::uint32_t>(), std::uint32_t(na), std::uint32_t(nb_cap), filter,
                stream_);
    cu::merge_b(bk.as<std::uint64_t>(), ba.as<std::uint32_t>(), ak.as<std::uint64_t>(), nbp, ap, bp,
                ok.as<std::uint64_t>(), oa.as<std::uint32_t>(), std::uint32_t(na), std::uint32_t(nb_cap), filter,
                stream_);
  }

  // ---------------------------------------------------------------- buffers
  // Grow `b` to at least `bytes` (half again its size when the budget allows), keeping its
  // contents when `keep`.
  void reserve(Buf& b, std::size_t bytes, bool keep) {
    bytes = std::max<std::size_t>(bytes, 16);
    if (b.p && b.cap >= bytes) return;
    if (!keep) release(b);
    std::size_t want = std::max(bytes, b.cap + b.cap / 2);
    if (mem_used_ + want > mem_cap_) want = bytes;
    want = (want + 15) & ~std::size_t(15);
    charge_mem(want, "a buffer");
    void* nb = nullptr;
    if (cudaMalloc(&nb, want) != cudaSuccess) {
      cudaGetLastError();
      throw OutOfMemory("out of GPU memory: the system refused a " + mb(want) + " buffer");
    }
    mem_used_ += want;
    if (b.p) CK(cudaMemcpyAsync(nb, b.p, b.cap, cudaMemcpyDeviceToDevice, stream_));
    if (b.p) { CK(cudaStreamSynchronize(stream_)); mem_used_ -= b.cap; cudaFree(b.p); }
    b.p = nb;
    b.cap = want;
  }
  void release(Buf& b) {
    if (!b.p) return;
    mem_used_ -= b.cap;
    cudaFree(b.p);
    b.p = nullptr;
    b.cap = 0;
  }
  void charge_mem(std::size_t bytes, const char* what) const {
    if (mem_used_ + bytes <= mem_cap_) return;
    throw OutOfMemory("out of GPU memory: " + std::string(what) + " of " + mb(bytes) + " would pass the " +
                      mb(mem_cap_) + " budget (" + mb(mem_used_) + " in use; HORA_GPU_MEM_GB sets it)");
  }
  static std::string mb(std::size_t bytes) { return std::to_string((bytes + (1u << 20) - 1) >> 20) + " MB"; }
  template <class T>
  void put(Buf& b, const std::vector<T>& v) {
    reserve(b, std::max<std::size_t>(1, v.size()) * sizeof(T), false);
    if (!v.empty()) CK(cudaMemcpy(b.p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<Buf*> all_bufs_() {
    return {&locbase_, &Lloc_, &Uloc_, &Lq_, &Uq_, &kshift_, &kmask_, &tdbm_, &Lx_, &Ux_, &ptab_, &zhash_,
            &ztslot_, &zcnt_, &ztab_, &freel_, &pend_, &zref_, &zref2_, &alive_, &alive2_, &remap_, &base_k_,
            &base_a_, &base2_k_, &base2_a_, &delta_k_, &delta_a_, &delta2_k_, &delta2_a_, &fk_, &fa_, &fnk_,
            &fna_, &Lb_, &Ub_, &big1_, &big2_, &ulist_, &bz_, &sk0_, &sk1_, &sv0_, &sv1_, &head_, &hpre_,
            &bstart_, &bkey_, &binfo_, &rowcnt_, &rowoff_, &keep_, &kpre_, &keep1_, &k1pre_, &surv_, &new_k_,
            &new_a_, &ctr_, &nbuf_, &hsh_, &rep_, &zslot_, &win_, &wpre_, &zid_, &wlist_, &fl_a_, &fl_ap_,
            &fl_b_, &fl_bp_, &sort_tmp_, &dsel_, &dzone_, &chg_};
  }

  const FrontierProblem& P_;
  const Network& net_;
  int n_ = 0, nn_ = 0, np_ = 0, gz_ = 2, key_bits_ = 0, nthreads_ = 1;
  std::uint32_t sgs_expand_ = 8, sgs_red_ = 8, sgs_union_ = 8;
  std::size_t mem_cap_ = 0, mem_used_ = 0;
  bool has_tclock_ = false;
  int solo_ = 8;
  bool union_ = true;
  bool antichain_ = false;         ///< HORA_GPU_ANTICHAIN=1: re-establish the antichain after merging
  Buf chg_;
  int min_bits_ = 8;
  DlState* dl_ = nullptr;          ///< non-null for a deadlock search
  Buf dsel_, dzone_;               ///< the zones deadlock_level reads back
  std::uint64_t clockbits_ = 0, qbits_ = 0, sentinel_ = 0;
  std::size_t wmax_ = 0;
  bool trace_ = false, stress_ = false;
  int level_ = 0;
  std::size_t nf_prev_ = 0, lvl_w_ = 0, lvl_valid_ = 0, lvl_kept_ = 0, lvl_rows_ = 0, lvl_buckets_ = 0;
  double t_enum_ = 0, t_wait_ = 0;
  std::string dev_name_;
  cudaStream_t stream_ = nullptr;
  std::uint32_t* hc_ = nullptr;    // the batch's counters, as they came back (pinned: an async copy)
  Pin hc_pin_;

  Buf locbase_, Lloc_, Uloc_, Lq_, Uq_, kshift_, kmask_, tdbm_, Lx_, Ux_;
  std::vector<void*> pages_;
  Buf ptab_;
  int page_shift_ = 16;
  std::size_t page_zones_ = 0;
  Buf zhash_, ztslot_, zcnt_, ztab_, freel_, pend_;
  std::size_t zcount_ = 0, zcap_ = 0, nfree_ = 0, ntomb_ = 0;
  std::uint32_t freed_before_ = 0, freed_mirror_ = 0;
  Buf zref_, zref2_, alive_, alive2_, remap_;
  Buf base_k_, base_a_, base2_k_, base2_a_, delta_k_, delta_a_, delta2_k_, delta2_a_;
  std::size_t acount_ = 0, nbase_ = 0, ndelta_ = 0;
  std::size_t nlive_ = 1, ndisc_ = 1;
  bool extrap_ = false;
  Buf fk_, fa_, fnk_, fna_;
  std::vector<std::uint64_t> fk_h_;
  std::size_t nf_ = 0, nfn_ = 0;
  Chunk chunks_[2];
  std::vector<RecipeTable> tables_;
  std::vector<std::size_t> runs_;
  std::vector<std::uint32_t> run_beg_, run_cnt_, run_lbeg_;
  std::vector<int> run_tid_;
  Buf Lb_, Ub_, big1_, big2_, ulist_;
  Buf bz_, sk0_, sk1_, sv0_, sv1_, head_, hpre_, bstart_, bkey_, binfo_, rowcnt_, rowoff_, keep_, kpre_, keep1_,
      k1pre_, surv_, new_k_, new_a_, ctr_, nbuf_;
  Buf hsh_, rep_, zslot_, win_, wpre_, zid_, wlist_;
  Buf fl_a_, fl_ap_, fl_b_, fl_bp_, sort_tmp_;
  std::vector<Buf> scan_sums_;
};

}  // namespace

int frontier_reach_cuda(const FrontierProblem& p) {
  const ReachStats before = p.stats ? *p.stats : ReachStats{};
  int bits = 8;
  DlState dl;
  if (p.deadlock) dl.bump.assign(p.Lq.size(), kAluNegInf);
  FrontierProblem q;
  for (;;) {
    // A deadlock search runs under the bounds its refinement has grown to, or under a_M's.
    q = p;
    if (p.deadlock) {
      auto collapse = [](std::vector<std::int32_t>& lo, std::vector<std::int32_t>& hi) {
        for (std::size_t c = 0; c < lo.size() && c < hi.size(); ++c) lo[c] = hi[c] = std::max(lo[c], hi[c]);
      };
      // The bounds the covering reads: raw under the G-simulation, integer constants under a_LU.
      auto& qLloc = p.gsim ? q.Lgloc : q.Lloc;
      auto& qUloc = p.gsim ? q.Ugloc : q.Uloc;
      auto& qLq = p.gsim ? q.Lgq : q.Lq;
      auto& qUq = p.gsim ? q.Ugq : q.Uq;
      if (dl.collapsed) {
        collapse(qLq, qUq);
        collapse(q.Lx, q.Ux);
        for (std::size_t pr = 0; pr < qLloc.size(); ++pr)
          for (std::size_t l = 0; l < qLloc[pr].size(); ++l) collapse(qLloc[pr][l], qUloc[pr][l]);
      } else {
        for (std::size_t c = 1; c < dl.bump.size() && c < qLq.size(); ++c)
          if (dl.bump[c] != kAluNegInf) {
            // A raised constant is a non-strict raw bound under the G-simulation: the safe side.
            const std::int32_t b = p.gsim ? le(dl.bump[c]) : dl.bump[c];
            qLq[c] = std::max(qLq[c], b);
            qUq[c] = std::max(qUq[c], b);
          }
      }
      dl.grew = false;
    }
    FrontierCuda m(q, bits, p.deadlock ? &dl : nullptr);
    try {
      return m.run();
    } catch (const Overflow&) {
      const int got = m.bits();
      if (got >= 32) throw std::runtime_error("CUDA frontier: a zone exceeds 32-bit entries");
      bits = got * 2;
      if (p.stats) *p.stats = before;
      if (std::getenv("HORA_GPU_TRACE"))
        std::fprintf(stderr, "[gpu] a zone exceeds %d-bit entries: restarting with %d-bit\n", got, bits);
    } catch (const Restart&) {
      if (p.stats) *p.stats = before;
      if (std::getenv("HORA_GPU_TRACE")) std::fprintf(stderr, "[gpu] deadlock refinement grew: restarting\n");
    } catch (const Collapse&) {
      dl.collapsed = true;
      if (p.stats) *p.stats = before;
      if (std::getenv("HORA_GPU_TRACE"))
        std::fprintf(stderr, "[gpu] diagonal deadlock atom: restarting under a_M bounds\n");
    }
  }
}

}  // namespace detail
}  // namespace hora

#endif  // HORA_HAVE_CUDA
