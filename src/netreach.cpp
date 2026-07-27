#include "ptctlmc/netreach.hpp"
#include <cstdio>
#include <chrono>
#ifdef __APPLE__
#include <malloc/malloc.h>
#include <mach/mach.h>
#endif
#ifdef __linux__
#include <malloc.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ptctlmc/fed.hpp"
#include "ptctlmc/mindbm.hpp"   // minimal-graph packed zone storage (UDBM mingraph port)
#include "ptctlmc/net_gpu.hpp"

#ifdef PTCTLMC_HAVE_OPENMP
#include <omp.h>
#endif

namespace ptctlmc {
namespace {

// The discrete part of a symbolic state is copied for every successor and again
// for every survivor entering the next frontier. Backing locs/ints with the same
// per-thread pool as the zone (dbm_pool.hpp) turns those millions of tiny
// vector copies from malloc/free pairs — which serialize the OpenMP workers on
// the global allocator lock — into lock-free free-list pops.
using LocVec = std::vector<int, PoolAllocator<int>>;
using IntVec = std::vector<long, PoolAllocator<long>>;

struct SymState {
  LocVec locs;
  IntVec ints;
  Dbm zone;
};

#ifdef __linux__
static std::size_t linux_rss_bytes() {
  FILE* f = std::fopen("/proc/self/statm", "r");
  if (!f) return 0;
  unsigned long size = 0, resident = 0;
  const int n = std::fscanf(f, "%lu %lu", &size, &resident);
  std::fclose(f);
  if (n < 2) return 0;
  const long ps = sysconf(_SC_PAGESIZE);
  if (ps <= 0) return 0;
  return static_cast<std::size_t>(resident) * static_cast<std::size_t>(ps);
}

static std::size_t linux_heap_bytes() {
#if defined(__GLIBC__)
  return static_cast<std::size_t>(mallinfo2().uordblks);
#else
  return 0;
#endif
}
#endif


// ---- zone interning --------------------------------------------------------
// Discrete-heavy symmetric models (viking) reach millions of states but only a
// handful of DISTINCT zone matrices (~120 for viking15's 2.77M states). Storing
// each distinct DBM once and referencing it by a 4-byte index collapses the
// resident zone set from ~1.5 GB to a few KB (L2-resident), removing the
// memory-bandwidth wall that made large viking trail LTSmin. Interned zones are
// IMMUTABLE: subsumption and try_union only ever READ the stored zone (mutating
// the incoming fresh one), so reads need no lock; only the first sight of a new
// distinct zone takes the mutex. A per-thread byte-keyed cache skips the mutex
// once a thread has seen a zone -- for viking, after the first BFS level every
// intern is a lock-free hit. Low-sharing models (fischer, ~80% distinct) simply
// intern more zones; correctness is unaffected -- two byte-identical canonical
// DBMs are the same zone, and the merge is unchanged.
// Distinct zones are stored as MINIMAL-GRAPH packed records (see mindbm.hpp): only the
// non-redundant constraints, 16-bit bounds when they fit, ~144 B for a 21-clock zone vs 1764 B
// dense. Dedup on the packed bytes is exact (packing is a deterministic function of the
// canonical matrix), so byte-identical records are the same zone.
//
// The interner is engineered so its *bookkeeping* stays a small fraction of the zone bytes --
// the node-based std::unordered_map it replaced cost as much as the zones themselves (and the
// per-thread map caches DOUBLED that). Three structures instead:
//   * a BUMP ARENA of slabs owns the packed bytes contiguously (no per-record vector object or
//     malloc header/rounding -- ~40 B/zone saved),
//   * a FLAT open-addressing table holds one int32 index per slot (0=empty); the KEY BYTES are
//     recovered from the stored record on probe, so the table is ~5.7 B/zone, not ~56,
//   * a small fixed per-thread DIRECT-MAPPED cache (bounded, ~0.5 MB/thread) absorbs repeat
//     interns lock-free -- all a high-sharing model (viking: ~120 distinct zones) needs, and
//     cheap for a low-sharing one (fischer) where no cache would help anyway.
//
// `get(idx)` UNPACKS on read into a per-thread dense scratch Dbm (the algebra needs a dense
// matrix); stored bytes are immutable, so reads are lock-free. Every get() call site consumes
// the zone before this thread's next get(), so one scratch per thread is safe.
struct ZoneIntern {
  // idx -> record pointer, in fixed CHUNKs allocated on demand (adaptive growth). Each slot is
  // published under mtx before `n` is release-stored, so a reader holding idx < n sees it.
  static constexpr int kChunkBits = 16;
  static constexpr int kChunk = 1 << kChunkBits;        // 65536 zones/chunk
  static constexpr int kChunkMask = kChunk - 1;
  static constexpr int kChunks = 1 << 14;               // -> up to ~1.07e9 distinct zones
  static constexpr std::size_t kSlabBytes = 1u << 22;   // 4 MB packed-record slabs
  static constexpr int kCacheBits = 15;                 // 32768 lines/thread (~0.5 MB)
  static constexpr std::uint32_t kCacheMask = (1u << kCacheBits) - 1;

  struct CacheLine { std::uint64_t h; std::int32_t idx; };

  std::deque<std::vector<std::uint8_t>> slabs;          // bump arena; slab data addresses are stable
  std::uint8_t* bumpPtr = nullptr;                      // current slab cursor (mtx-guarded)
  std::size_t bumpLeft = 0;
  std::vector<const std::uint8_t**> chunks;             // chunks[c][.] -> packed-record bytes
  std::vector<std::int32_t> htab;                       // open-addressing: 0=empty, else idx+1
  std::uint64_t hmask;
  std::size_t hcount = 0;
  std::vector<std::vector<CacheLine>> tcache;           // [thread][line] direct-mapped
  // Reclamation: dead records are recycled in place (no rebuild, no zidx remap). A freed
  // record slot returns to a free-list keyed by its exact byte length (packed records take
  // only a few distinct lengths per model, so reuse is near-total); a freed zidx returns to
  // `freeIdx`. intern() reuses both before growing the arena, so slab memory is bounded by the
  // PEAK LIVE working set and recycled internally -- RSS never depends on the OS reclaiming.
  std::unordered_map<std::uint32_t, std::vector<std::uint8_t*>> freeRecs;
  std::vector<int> freeIdx;
  std::vector<std::int32_t> htabScratch;                // reused sweep buffer (avoids per-level realloc)
  std::mutex mtx;
  std::atomic<int> n{0};
  int dim;                                              // clock dim (nclocks+1); all zones share it

  ZoneIntern(int dim2_, int nthreads)
      : chunks(kChunks, nullptr), htab(1u << 16, 0), hmask((1u << 16) - 1),
        tcache(nthreads > 0 ? nthreads : 1) {
    dim = 1; while (dim * dim < dim2_) ++dim;           // dim2_ = dim*dim
    for (auto& tc : tcache) tc.assign(1u << kCacheBits, CacheLine{0, -1});
  }
  ~ZoneIntern() { for (auto* c : chunks) delete[] c; }

  static std::uint64_t hash_bytes(const std::uint8_t* p, std::size_t len) {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < len; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
  }
  const std::uint8_t* record(int idx) const { return chunks[idx >> kChunkBits][idx & kChunkMask]; }

  const Dbm& get(int idx) const {
    const std::uint8_t* rec = record(idx);
    static thread_local Dbm scratch;
    if (scratch.dim() != dim) scratch = Dbm(dim - 1);
    mindbm_unpack(rec, scratch.data().data());
    return scratch;
  }

  // Decode into caller-owned storage instead of the shared per-thread scratch, so a caller
  // can hold several zones live at once. get() overwrites one scratch, which forces any
  // multi-zone fold to re-unpack on every visit; with this the fold unpacks each zone once.
  void get_into(int idx, Dbm& out) const {
    if (out.dim() != dim) out = Dbm(dim - 1);
    mindbm_unpack(record(idx), out.data().data());
  }
  int zdim() const { return dim; }

  // Probe the flat table for the packed bytes [q,q+qlen). Returns idx or -1; on miss, *slot is
  // the empty table position to insert at. Caller holds mtx.
  int find_locked(const std::uint8_t* q, std::size_t qlen, std::uint64_t h, std::size_t* slot) const {
    std::size_t i = h & hmask;
    while (htab[i]) {
      const int idx = htab[i] - 1;
      const std::uint8_t* rec = record(idx);
      if (mindbm_size(rec) == qlen && std::memcmp(rec, q, qlen) == 0) return idx;
      i = (i + 1) & hmask;
    }
    *slot = i;
    return -1;
  }
  void rehash_locked() {
    const std::size_t ncap = (hmask + 1) * 2;
    std::vector<std::int32_t> nt(ncap, 0);
    const std::uint64_t nmask = ncap - 1;
    const int cnt = n.load(std::memory_order_relaxed);
    for (int idx = 0; idx < cnt; ++idx) {
      const std::uint8_t* rec = record(idx);
      std::size_t i = hash_bytes(rec, mindbm_size(rec)) & nmask;
      while (nt[i]) i = (i + 1) & nmask;
      nt[i] = idx + 1;
    }
    htab.swap(nt);
    hmask = nmask;
  }

  // Intern `z` (canonical, non-empty), returning its distinct-zone index. `z` is not modified.
  int intern(const Dbm& z, int tid) {
    static thread_local std::vector<std::uint8_t> qbuf;
    static thread_local std::vector<std::uint32_t> bits;
    static thread_local std::vector<int> first, next;
    if (static_cast<int>(first.size()) != dim) { first.resize(dim); next.resize(dim); }
    bits.resize(mindbm_bits2words(static_cast<std::uint32_t>(dim) * dim));
    qbuf.resize(mindbm_bound(dim));
    const std::size_t qlen =
        mindbm_pack(z.data().data(), dim, qbuf.data(), bits.data(), first.data(), next.data());
    const std::uint64_t h = hash_bytes(qbuf.data(), qlen);

    // Lock-free fast path: per-thread direct-mapped cache. Verify bytes (a 64-bit-hash clash
    // returning the wrong index would be UNSOUND), so a hit is always correct.
    CacheLine& cl = tcache[tid][static_cast<std::uint32_t>(h) & kCacheMask];
    if (cl.idx >= 0 && cl.h == h) {
      const std::uint8_t* rec = record(cl.idx);
      if (mindbm_size(rec) == qlen && std::memcmp(rec, qbuf.data(), qlen) == 0) return cl.idx;
    }

    std::lock_guard<std::mutex> lk(mtx);
    std::size_t slot = 0;
    int idx = find_locked(qbuf.data(), qlen, h, &slot);
    if (idx < 0) {
      // A record slot: reuse a freed one of the exact length, else bump-allocate (qlen is a
      // multiple of 4 -> stays 4-aligned).
      std::uint8_t* rec;
      auto fit = freeRecs.find(static_cast<std::uint32_t>(qlen));
      if (fit != freeRecs.end() && !fit->second.empty()) {
        rec = fit->second.back();
        fit->second.pop_back();
      } else {
        if (bumpLeft < qlen) {
          const std::size_t sz = qlen > kSlabBytes ? qlen : kSlabBytes;
          slabs.emplace_back(sz);
          bumpPtr = slabs.back().data();
          bumpLeft = sz;
        }
        rec = bumpPtr;
        bumpPtr += qlen; bumpLeft -= qlen;
      }
      std::memcpy(rec, qbuf.data(), qlen);
      // A zidx: reuse a freed one, else grow.
      if (!freeIdx.empty()) {
        idx = freeIdx.back();
        freeIdx.pop_back();
      } else {
        idx = n.load(std::memory_order_relaxed);
        const int c = idx >> kChunkBits;
        if (c >= kChunks) throw std::runtime_error("zone intern table overflow (too many distinct zones)");
        if (!chunks[c]) chunks[c] = new const std::uint8_t*[kChunk];
        n.store(idx + 1, std::memory_order_release);
      }
      chunks[idx >> kChunkBits][idx & kChunkMask] = rec;
      htab[slot] = idx + 1;
      ++hcount;
      if (hcount * 10 >= (hmask + 1) * 7) rehash_locked();  // htab-only; record ptrs unaffected
    }
    cl = CacheLine{h, idx};   // fill this thread's cache (single-writer per tid)
    return idx;
  }

  // Reclaim interned records no longer referenced by any live passed-set node. `alive[idx]` is
  // set for every zidx still referenced (a live arena node or the next frontier). Dead records
  // return to the size-classed free-list and their zidx to `freeIdx` (recycled in place -- no
  // rebuild, no remap); the htab is rebuilt from survivors. The per-thread caches are NOT
  // cleared: a hit verifies bytes, so a stale hash->idx entry (idx since recycled to another
  // zone) simply fails the byte-compare and falls through. STOP-THE-WORLD (all workers joined).
  void sweep(const std::vector<std::uint8_t>& alive) {
    htabScratch.assign(htab.size(), 0);      // reused buffer; htab.size() only grows
    std::vector<std::int32_t>& nh = htabScratch;
    const std::uint64_t nmask = htab.size() - 1;
    std::size_t live = 0;
    for (std::size_t s = 0; s < htab.size(); ++s) {
      const std::int32_t v = htab[s];
      if (!v) continue;
      const int idx = v - 1;
      std::uint8_t* rec = const_cast<std::uint8_t*>(record(idx));
      if (idx < static_cast<int>(alive.size()) && alive[idx]) {
        std::size_t i = hash_bytes(rec, mindbm_size(rec)) & nmask;
        while (nh[i]) i = (i + 1) & nmask;
        nh[i] = idx + 1;
        ++live;
      } else {
        freeRecs[static_cast<std::uint32_t>(mindbm_size(rec))].push_back(rec);
        freeIdx.push_back(idx);
      }
    }
    htab.swap(nh);
    hmask = nmask;
    hcount = live;
  }

  // Records currently handed out: live zones PLUS garbage not yet reclaimed. `n` is a
  // high-water index counter that never shrinks (sweep recycles indices into `freeIdx`
  // instead of renumbering), so `n` alone cannot drive the sweep trigger -- it crosses any
  // fixed threshold once and stays there. This is the quantity a growth-based trigger must
  // watch: it rises with every fresh intern and drops back to the live set after a sweep.
  std::size_t allocated() const {
    return static_cast<std::size_t>(n.load(std::memory_order_relaxed)) - freeIdx.size();
  }

  // Byte accounting (PTCTLMC_STATS): where the interner's resident bytes go.
  void account(std::size_t& packed, std::size_t& mapb, std::size_t& cacheb) const {
    for (const auto& s : slabs) packed += s.capacity();
    mapb += htab.capacity() * sizeof(std::int32_t);
    for (const auto& tc : tcache) cacheb += tc.capacity() * sizeof(CacheLine);
  }
};

// Shared dense-zone interner for the GPU driver's passed set. NOT currently wired in -- see the
// note at the passed-set declaration in the GPU driver for why (it is a memory/time trade that
// only pays on low-distinct-zone models and needs per-model routing to be a win).
//
// The GPU merge kernels consume DENSE matrices, so unlike the CPU path (which stores minimal
// constraint systems, see ZoneIntern) this keeps zones dense -- but it stores each distinct zone
// ONCE and hands out an index. The passed set then costs 4 bytes per (discrete state, zone)
// instead of d2 words. On models with few distinct zones the saving is enormous (viking15: 2.77M
// stored zones, 120 distinct); on models where every zone differs it costs one hash per survivor
// and nothing is lost, because the dense copy had to be written anyway.
//
// Thread-safety mirrors the CPU interner: records live in fixed-size chunks that are never moved,
// so an index handed out to a reader stays valid while other threads intern. `n` is published with
// release semantics after its chunk is installed, so any index < n has readable storage.
struct GpuZoneIntern {
  static constexpr int kChunkBits = 13;                 // 8192 zones per chunk
  static constexpr int kChunk = 1 << kChunkBits;
  static constexpr int kChunkMask = kChunk - 1;
  static constexpr int kMaxChunks = 1 << 17;            // 1G zones ceiling
  static constexpr int kStripes = 512;

  explicit GpuZoneIntern(int d2) : d2_(d2), stripes_(kStripes) {}
  ~GpuZoneIntern() { for (auto& c : chunks_) delete c.load(std::memory_order_relaxed); }

  const raw_t* get(int idx) const {
    return chunks_[idx >> kChunkBits].load(std::memory_order_acquire)->data() +
           static_cast<std::size_t>(idx & kChunkMask) * d2_;
  }

  // mode: Dedup interns (one copy per distinct zone); Append stores every zone without hashing,
  // which isolates the cost of the hash+lookup from the cost of the indexed layout itself.
  enum class Mode { Dedup, Append };
  void set_mode(Mode m) { mode_ = m; }

  int intern(const raw_t* z) {
    if (mode_ == Mode::Append) {
      const int idx = alloc();
      std::memcpy(const_cast<raw_t*>(get(idx)), z, static_cast<std::size_t>(d2_) * sizeof(raw_t));
      return idx;
    }
    const std::uint64_t h = hash(reinterpret_cast<const std::uint32_t*>(z));
    Stripe& s = stripes_[h & (kStripes - 1)];
    std::lock_guard<std::mutex> lk(s.m);
    auto& bucket = s.tab[h];
    for (const int cand : bucket)
      if (std::memcmp(get(cand), z, static_cast<std::size_t>(d2_) * sizeof(raw_t)) == 0) return cand;
    const int idx = alloc();
    std::memcpy(const_cast<raw_t*>(get(idx)), z, static_cast<std::size_t>(d2_) * sizeof(raw_t));
    bucket.push_back(idx);
    return idx;
  }

  int count() const { return n_.load(std::memory_order_acquire); }
  std::size_t bytes() const {
    return static_cast<std::size_t>(chunks_.size()) * kChunk * d2_ * sizeof(raw_t);
  }

 private:
  // Hash the WHOLE zone, in four independent lanes.
  //
  // Two failure modes to avoid. Hashing a prefix is catastrophic: canonical DBMs routinely agree
  // on their leading rows, so a prefix key collapses hundreds of thousands of distinct zones into
  // one bucket and the linear scan below goes quadratic (fischer11: 2.6s -> 395s). Hashing all
  // words with textbook FNV is correct but serial -- one multiply per word on the critical path,
  // and this runs for every merged zone on every level. Four lanes give the same coverage with
  // four multiplies in flight, then mix once at the end.
  std::uint64_t hash(const std::uint32_t* z) const {
    std::uint64_t h0 = 1469598103934665603ULL, h1 = h0 ^ 0x9e3779b97f4a7c15ULL,
                  h2 = h0 ^ 0xc2b2ae3d27d4eb4fULL, h3 = h0 ^ 0x165667b19e3779f9ULL;
    int k = 0;
    for (; k + 4 <= d2_; k += 4) {
      h0 = (h0 ^ z[k])     * 1099511628211ULL;
      h1 = (h1 ^ z[k + 1]) * 1099511628211ULL;
      h2 = (h2 ^ z[k + 2]) * 1099511628211ULL;
      h3 = (h3 ^ z[k + 3]) * 1099511628211ULL;
    }
    for (; k < d2_; ++k) h0 = (h0 ^ z[k]) * 1099511628211ULL;
    std::uint64_t h = h0 ^ (h1 * 3) ^ (h2 * 5) ^ (h3 * 7);
    h ^= h >> 29; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 32;   // final avalanche
    return h;
  }
  // LOCK-FREE in the common case. This used to take `growMtx_` on every call, which serialised
  // all 12 threads on one mutex for millions of zones -- 8-12 CPU-seconds of contention per run,
  // and the dominant cost of the passed rebuild. The index comes from an atomic bump; the mutex
  // is taken only when a chunk boundary is crossed (once per kChunk allocations), and then only
  // to install the chunk.
  //
  // `chunks_` is pre-sized and never resized: growing it with push_back would reallocate the
  // pointer array under readers calling get() concurrently, which is a use-after-free (it
  // segfaulted on fischer11). Slots are published with release / read with acquire, so a reader
  // that sees the pointer also sees the storage it points to.
  int alloc() {
    const int idx = n_.fetch_add(1, std::memory_order_relaxed);
    const int c = idx >> kChunkBits;
    if (c >= kMaxChunks) throw std::runtime_error("GPU zone interner overflow");
    if (!chunks_[c].load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lk(growMtx_);
      if (!chunks_[c].load(std::memory_order_relaxed))
        chunks_[c].store(new std::vector<raw_t>(static_cast<std::size_t>(kChunk) * d2_),
                         std::memory_order_release);
    }
    return idx;
  }

  struct Stripe {
    std::unordered_map<std::uint64_t, std::vector<int>> tab;
    std::mutex m;
  };
  int d2_;
  Mode mode_ = Mode::Dedup;
  std::vector<std::atomic<std::vector<raw_t>*>> chunks_{kMaxChunks};   // fixed: never resized
  std::deque<Stripe> stripes_;
  std::mutex growMtx_;
  std::atomic<int> n_{0};
};

// Flat open-addressing passed/waiting table for one shared-passed stripe.
// Replaces std::unordered_map<dk, std::vector<Entry>>, which cost ~2 system
// mallocs per discrete state (a map node + a bucket vector) -- the dominant
// cost on discrete-heavy models (viking, ~1 zone/dk), where LTSmin's flat
// lockless table wins. Here:
//   * `tab` is an open-addressing slot array: dk -> arena index of the antichain
//     HEAD (or -1). It only holds small (dk,int) pairs, so a rehash moves no
//     SymStates.
//   * `arena` is a std::deque of nodes with STABLE addresses (never moved on
//     growth), each holding one zone (Entry) and the `next` index of the next
//     zone for the same dk. Dead (subsumed/merged) nodes are tombstoned.
// The per-bucket merge is byte-for-byte the same aLU + convex-union logic as the
// legacy add_to; only the container changed. The brand-new-dk path (the whole of
// viking) allocates nothing per state and skips state_bounds.
struct NetPassed {
  // A node stores ONLY the zone + BFS level. The discrete part (locs/ints) is
  // NOT kept: it is losslessly packed in the bucket's dk (disc_key/decode_key are
  // exact inverses) and identical for every zone in the bucket, so storing it per
  // node was pure redundancy -- two pooled-vector allocations and ~72 B per
  // stored state. Dropping it shrinks the resident passed set (the working set
  // that thrashes cache at 2.7M+ states, the large-viking bottleneck) and removes
  // the per-state alloc the profile was dominated by. locs/ints are reconstructed
  // from dk only when a survivor re-enters the frontier (once per survivor, in
  // for_level). The merge below never needs them: it touches stored nodes only
  // for their zone, and the aLU bounds come from the INCOMING state's locs.
  struct Node { int zidx; int level; int next; bool dead; };   // zidx: index into the shared ZoneIntern
  struct Slot { std::uint64_t dk; int head; };
  std::deque<Node> arena;
  std::vector<Slot> tab;
  std::uint64_t mask;
  std::size_t used = 0;   // distinct discrete states (occupied slots)
  std::size_t lastCompact = 0;  // arena.size() at last compaction (geometric trigger)
  ZoneIntern* zi = nullptr;   // shared distinct-zone table (set before first add)

  NetPassed() : tab(1024, Slot{0, -1}), mask(1023) {}

  static std::uint64_t mix(std::uint64_t dk) { return dk * 0x9e3779b97f4a7c15ULL; }

  std::size_t slot_of(std::uint64_t dk) const {
    std::size_t i = static_cast<std::size_t>(mix(dk) & mask);
    while (tab[i].head != -1 && tab[i].dk != dk) i = (i + 1) & mask;
    return i;
  }
  void grow() {
    const std::size_t ncap = (mask + 1) * 2;
    std::vector<Slot> nt(ncap, Slot{0, -1});
    const std::uint64_t nmask = ncap - 1;
    for (const auto& s : tab)
      if (s.head != -1) {
        std::size_t i = static_cast<std::size_t>(mix(s.dk) & nmask);
        while (nt[i].head != -1) i = (i + 1) & nmask;
        nt[i] = s;
      }
    tab.swap(nt);
    mask = nmask;
  }

  // Merge zone `s.zone` (stamped `level`) into dk's antichain. Only the zone is
  // stored; the rest of `s` is discarded. Returns the arena index of the newly
  // stored node (a new survivor) or -1 if subsumed and dropped. `bounds(locs, ..)`
  // fills the per-location aLU bounds from the INCOMING state; it is only invoked
  // when the discrete state already exists, so a brand-new dk pays nothing for it.
  template <class BoundsFn>
  int add(SymState&& s, std::uint64_t dk, int level, bool has_diag, BoundsFn&& bounds, int tid) {
    if (used * 10 >= (mask + 1) * 7) grow();
    const std::size_t si = slot_of(dk);
    if (tab[si].head == -1) {                       // brand-new discrete state (fast path)
      const int idx = static_cast<int>(arena.size());
      arena.push_back(Node{zi->intern(s.zone, tid), level, -1, false});
      tab[si] = Slot{dk, idx};
      ++used;
      return idx;
    }
    thread_local std::vector<std::int32_t> Ls, Us;
    if (!has_diag) bounds(s.locs.data(), Ls, Us);
    const int head = tab[si].head;
    // Unpack the live chain ONCE into a reused thread-local array, then fold over the array
    // rather than re-walking the chain. zi->get() decodes a packed min-graph record into a
    // single shared scratch on every call, so the three folds below used to re-run
    // mindbm_unpack for each node they visited -- and (4) restarts its scan after every
    // successful union, re-unpacking the whole chain again each time. Chains also thread
    // through a std::deque via `next`, so walking them three times is three passes of
    // pointer-chasing over cache-cold nodes, and dead nodes are traversed to be skipped.
    // merge_level (the batched path) already folds over a compact `live` array for exactly
    // this reason; this brings the incremental path in line. Same order, same predicates,
    // same results -- only the number of unpacks changes.
    // (1) FAST PATH -- decide the overwhelmingly common "duplicate, drop it" case straight
    // off the PACKED records, with no unpack and no closure.
    //
    // Profiling arbiter17: of 12.3M add() calls, 89.8% are killed
    // immediately by the subsumption test below against a chain whose mean length is 1.00 --
    // and reaching that test cost a mindbm_unpack per probe, i.e. a full O(dim^3) dbm_close
    // to rebuild the redundant entries. That was 36.3s of unpacking to feed 1.6s of actual
    // comparisons: 22x more time decompressing the stored zone than comparing against it.
    //
    // mindbm_includes_dense answers "s.zone subseteq stored" directly from the minimal
    // constraint system in O(|G|) (|G| ~ dim). Firing it early is sound in both modes:
    //   * has_diag: test (2) IS plain inclusion, so this is that test, verbatim.
    //   * aLU: plain inclusion implies aLU subsumption -- a witness needs
    //     Zp[y][x] < Z[y][x], which cannot happen when Z subseteq Z'. So an early hit is a
    //     conclusion the aLU test would also have reached.
    // When it does NOT fire we fall through to the full dense folds below, so nothing is
    // lost -- aLU still prunes everything it pruned before. State counts are unchanged.
    for (int i = head; i != -1; i = arena[i].next) {
      if (arena[i].dead) continue;
      if (mindbm_includes_dense(zi->record(arena[i].zidx), s.zone.data().data(), s.zone.dim()))
        return -1;
    }
    // `chain` holds the unpacked zones by fixed position; `live` indexes the ones still in
    // play. Removals go through `live` with an ORDERED erase (never swap-with-last): the
    // convex union in (4) is order-sensitive, so the survivors must keep their newest-first
    // relative order for the merge to reproduce the chain-walking result exactly. This is
    // the same discipline merge_level uses on its own `live` array.
    thread_local std::vector<Dbm> chain;
    thread_local std::vector<int> cidx;
    thread_local std::vector<std::size_t> live;
    cidx.clear();
    live.clear();
    std::size_t nc = 0;
    for (int i = head; i != -1; i = arena[i].next) {
      if (arena[i].dead) continue;
      if (nc == chain.size()) chain.emplace_back(zi->zdim() - 1);
      zi->get_into(arena[i].zidx, chain[nc]);
      cidx.push_back(i);
      live.push_back(nc);
      ++nc;
    }
    // (2) s subsumed by a stored zone? -> drop
    for (std::size_t k = 0; k < nc; ++k) {
      const Dbm& ez = chain[k];
      if (has_diag ? ez.includes(s.zone) : s.zone.alu_subsumed_by(ez, Ls, Us))
        return -1;
    }
    // (3) drop stored zones now subsumed by s
    for (std::size_t k = 0; k < live.size();) {
      const Dbm& ez = chain[live[k]];
      if (has_diag ? s.zone.includes(ez) : ez.alu_subsumed_by(s.zone, Ls, Us)) {
        arena[cidx[live[k]]].dead = true;
        live.erase(live.begin() + k);
      } else ++k;
    }
    // (4) convex-union merge: absorb stored zones whose union with s is exact.
    // try_union mutates only s.zone (the incoming fresh zone); the interned zone
    // is read-only, so this never disturbs another state's stored zone.
    for (bool merged = true; merged;) {
      merged = false;
      for (std::size_t k = 0; k < live.size(); ++k) {
        if (s.zone.try_union(chain[live[k]])) {
          arena[cidx[live[k]]].dead = true;
          live.erase(live.begin() + k);
          merged = true;
          break;
        }
      }
    }
    // (5) prepend s to the chain
    const int idx = static_cast<int>(arena.size());
    arena.push_back(Node{zi->intern(s.zone, tid), level, tab[si].head, false});
    tab[si].head = idx;
    return idx;
  }

  // Reclaim dead arena tombstones by rebuilding every chain live-only. Subsumption/merge mark
  // nodes dead but never free them, so the arena accumulates the whole history (~two-thirds
  // dead on fischer). This rebuild drops them, shrinking the arena to the live antichain.
  //
  // MUST be called in a between-levels phase with EXACTLY ONE thread per stripe (no other
  // access to this NetPassed) -- it swaps `arena` and rewrites `tab[].head`/`next`. Amortized
  // O(1) per node: the geometric guard only scans after the arena has doubled since the last
  // compaction, and a rebuild resets that baseline. Every occupied slot keeps >=1 live node
  // (both add() and merge_level revive `s`), so a chain never becomes empty and the
  // open-addressing probe chain is preserved.
  void maybe_compact() {
    const std::size_t sz = arena.size();
    if (sz < 8192 || sz < lastCompact * 2) return;
    std::size_t dead = 0;
    for (const auto& n : arena) if (n.dead) ++dead;
    if (dead * 2 < sz) { lastCompact = sz; return; }   // not enough waste yet
    std::deque<Node> na;
    for (auto& s : tab) {
      if (s.head == -1) continue;
      int newhead = -1, tail = -1;
      for (int i = s.head; i != -1; i = arena[i].next) {
        if (arena[i].dead) continue;
        const int ni = static_cast<int>(na.size());
        na.push_back(Node{arena[i].zidx, arena[i].level, -1, false});
        if (newhead == -1) newhead = ni; else na[tail].next = ni;
        tail = ni;
      }
      s.head = newhead;   // invariant: >=1 live per occupied slot, so never -1
    }
    arena.swap(na);
    lastCompact = arena.size();
  }

  // Visit the live zones stamped `level` for discrete state dk. The callback
  // receives the stored zone; the caller reconstructs locs/ints from dk.
  template <class F>
  void for_level(std::uint64_t dk, int level, F&& f) const {
    std::size_t i = static_cast<std::size_t>(mix(dk) & mask);
    while (tab[i].head != -1 && tab[i].dk != dk) i = (i + 1) & mask;
    if (tab[i].head == -1) return;
    for (int n = tab[i].head; n != -1; n = arena[n].next)
      if (!arena[n].dead && arena[n].level == level) f(arena[n].zidx);
  }

  void stats(std::size_t& disc, std::size_t& zones) const {
    disc += used;
    for (const auto& n : arena)
      if (!n.dead) ++zones;
  }
  // Byte accounting (PTCTLMC_STATS): arena nodes (incl. dead tombstones) + slot table.
  void account(std::size_t& arenaNodes, std::size_t& deadNodes, std::size_t& arenaB,
               std::size_t& tabB) const {
    arenaNodes += arena.size();
    for (const auto& n : arena) if (n.dead) ++deadNodes;
    arenaB += arena.size() * sizeof(Node);
    tabB += tab.capacity() * sizeof(Slot);
  }
  // Probe (PTCTLMC_STATS): visit every live zone, to measure distinct-zone sharing.
  template <class F>
  void for_each_live_zone(F&& f) const {
    for (const auto& nd : arena)
      if (!nd.dead) f(zi->get(nd.zidx));
  }
};

bool apply_all(Dbm& z, const std::vector<Constraint>& cs) {
  for (const auto& c : cs)
    if (!z.constrain(c.i, c.j, c.bound)) return false;
  return true;
}

}  // namespace

// ---- general modelling-language guard/update -------------------------------
// Only edges that need more than `var OP const` carry these (see NEdge). The
// common case is a null check, so this costs nothing for Fischer/viking.
//
// These two are deliberately OUTSIDE the anonymous namespace: the GPU engines
// (src/metal, src/cuda) must apply exactly the same semantics as the CPU path,
// and duplicating them there is how the backends silently diverged before --
// the Metal expander dropped `gen_update` entirely and explored roughly half of
// viking's state space. Declared in netreach.hpp.

// Scratch buffer so a guard that calls a function cannot perturb the live state.
bool gen_ok(const Network& net, const NEdge& e, const long* ints, std::size_t n) {
  if (!e.gen_guard) return true;
  thread_local std::vector<long> scratch;
  scratch.assign(ints, ints + n);
  EvalCtx cx;
  cx.cells = scratch.data();
  cx.funcs = &net.funcs;
  const long v = eval(*e.gen_guard, cx);
  return !cx.fault && v != 0;
}

// Runs the general update in place. Returns false if it faulted (bad index,
// division by zero, runaway loop) or if a write left a variable's declared
// range -- the latter would silently collide in the packed discrete key, so it
// is refused rather than mis-encoded.
bool gen_apply(const Network& net, const NEdge& e, long* ints, std::size_t n) {
  if (!e.gen_update) return true;
  thread_local std::vector<long> buf;
  buf.assign(ints, ints + n);
  EvalCtx cx;
  cx.cells = buf.data();
  cx.funcs = &net.funcs;
  cx.locals.assign(static_cast<std::size_t>(e.gen_locals) + 4, 0);
  bool returned = false;
  long ret = 0;
  exec(*e.gen_update, cx, &returned, &ret);
  if (cx.fault) return false;
  for (std::size_t i = 0; i < buf.size(); ++i) {
    if (buf[i] < net.intvars[i].lo || buf[i] > net.intvars[i].hi) return false;
    ints[i] = buf[i];
  }
  return true;
}

namespace {

bool int_ok(const std::vector<IntAtom>& g, const IntVec& ints) {
  for (const auto& a : g) {
    const long x = ints[a.var];
    bool r = false;
    switch (a.op) {
      case IntOp::Eq: r = x == a.val; break;
      case IntOp::Ne: r = x != a.val; break;
      case IntOp::Lt: r = x < a.val; break;
      case IntOp::Le: r = x <= a.val; break;
      case IntOp::Gt: r = x > a.val; break;
      case IntOp::Ge: r = x >= a.val; break;
    }
    if (!r) return false;
  }
  return true;
}

// Fills `inv` (cleared first) rather than returning a fresh vector: this runs
// 1-3x per successor in the hottest loop, so the allocation is pooled away by
// reusing a caller-owned scratch buffer.
void all_invariants(const Network& net, const LocVec& locs, std::vector<Constraint>& inv) {
  inv.clear();
  for (int p = 0; p < net.nprocs(); ++p)
    for (const auto& c : net.procs[p].locations[locs[p]].invariant) inv.push_back(c);
}

void frozen_flags(const Network& net, const LocVec& locs, bool* committed, bool* urgent) {
  *committed = *urgent = false;
  for (int p = 0; p < net.nprocs(); ++p) {
    const Location& l = net.procs[p].locations[locs[p]];
    if (l.committed) *committed = true;
    if (l.urgent) *urgent = true;
  }
}

// Reset clocks no process will read before resetting AND that the query does not
// read (qmask) — collapses zones that differ only in dead clocks, soundly.
void reset_inactive(const Network& net, const LocVec& locs, Dbm& z,
                    const std::vector<char>& qmask) {
  static const bool off = std::getenv("PTCTLMC_NOACTIVE") != nullptr;
  if (off) return;
  if (net.active.empty()) return;
  std::vector<char> a(net.nclocks + 1, 0);
  for (int p = 0; p < net.nprocs(); ++p) {
    const auto& ap = net.active[p][locs[p]];
    for (int c = 1; c <= net.nclocks; ++c)
      if (ap[c]) a[c] = 1;
  }
  for (int c = 1; c <= net.nclocks; ++c)
    if (!a[c] && !qmask[c]) z.reset(c);
}

enum ExtrapMode { ExNone = 0, ExLuPlus = 1, ExNoDiag = 2 };

// Does the network compare two clocks anywhere (a diagonal constraint)? If so, the
// LU/aLU abstraction is unsound and we fall back to exact zones (diagonals kept exact).
bool network_has_diagonal(const Network& net) {
  auto diag = [](const std::vector<Constraint>& cs) {
    for (const auto& c : cs) if (c.i > 0 && c.j > 0) return true;
    return false;
  };
  for (const auto& p : net.procs) {
    for (const auto& l : p.locations) if (diag(l.invariant)) return true;
    for (const auto& e : p.edges) if (diag(e.clock_guard)) return true;
  }
  return false;
}

std::string disc_key(const LocVec& locs, const IntVec& ints) {
  std::string k;
  k.reserve(locs.size() * 4 + ints.size() * 8);
  for (int l : locs) k.append(reinterpret_cast<const char*>(&l), sizeof(int));
  for (long v : ints) { std::int64_t x = v; k.append(reinterpret_cast<const char*>(&x), sizeof(std::int64_t)); }
  return k;
}

// The parent zone is passed SEPARATELY (szone) rather than read from s.zone, so
// callers that already hold the zone elsewhere -- notably the shared-passed path,
// where it lives in the intern table -- can pass it by reference instead of
// copying a 576 B DBM into the frontier state for every expansion (millions of
// pointless copies on large viking). s.zone is ignored; s carries only locs/ints.
void successors_net(const Network& net, const SymState& s, const Dbm& szone,
                    const std::vector<std::int32_t>& L,
                    const std::vector<std::int32_t>& U, const std::vector<char>& qmask, int emode, std::vector<SymState>& out) {
  thread_local std::vector<Constraint> inv;  // reused invariant scratch (see all_invariants)
  bool src_committed, src_urgent;
  frozen_flags(net, s.locs, &src_committed, &src_urgent);
  for (int p = 0; p < net.nprocs(); ++p) {
    if (src_committed && !net.procs[p].locations[s.locs[p]].committed) continue;
    for (int ei : net.procs[p].edges_of(s.locs[p])) {
      const NEdge& e = net.procs[p].edges[ei];
      if (e.sync_chan >= 0) continue;
      if (!int_ok(e.int_guard, s.ints) || !gen_ok(net, e, s.ints.data(), s.ints.size())) continue;
      Dbm z1 = szone;
      if (!apply_all(z1, e.clock_guard)) continue;
      for (int r : e.clock_resets) z1.reset(r);

      SymState ns;
      ns.locs = s.locs;
      ns.locs[p] = e.tgt;
      ns.ints = s.ints;
      for (const auto& a : e.int_updates) ns.ints[a.var] = a.val;
      if (!gen_apply(net, e, ns.ints.data(), ns.ints.size())) continue;

      all_invariants(net, ns.locs, inv);
      if (!apply_all(z1, inv)) continue;
      bool nc, nu;
      frozen_flags(net, ns.locs, &nc, &nu);
      if (!nc && !nu) { z1.delay(); apply_all(z1, inv); }
      reset_inactive(net, ns.locs, z1, qmask);
      if (emode == ExLuPlus) z1.extrapolate_lu_plus(L, U);
      else if (emode == ExNoDiag) z1.extrapolate_lu_plus_nodiag(L, U);
      if (z1.empty()) continue;
      ns.zone = std::move(z1);
      out.push_back(std::move(ns));
    }
  }

  // ---- binary channel synchronisation (a! in proc i  ||  a? in proc j) ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1) continue;  // sender a!
      if (net.chan_broadcast[se.sync_chan]) continue;      // broadcast handled below
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      const bool i_comm = net.procs[i].locations[s.locs[i]].committed;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        for (int ej : net.procs[j].edges_of(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan != se.sync_chan || re.sync_dir != -1) continue;  // receiver a?
          if (src_committed && !(i_comm || net.procs[j].locations[s.locs[j]].committed)) continue;
          if (!int_ok(re.int_guard, s.ints) || !gen_ok(net, re, s.ints.data(), s.ints.size())) continue;
          Dbm z1 = szone;
          if (!apply_all(z1, se.clock_guard)) continue;
          if (!apply_all(z1, re.clock_guard)) continue;
          for (int r : se.clock_resets) z1.reset(r);
          for (int r : re.clock_resets) z1.reset(r);
          SymState ns;
          ns.locs = s.locs;
          ns.locs[i] = se.tgt;
          ns.locs[j] = re.tgt;
          ns.ints = s.ints;
          for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;  // sender then receiver
          for (const auto& a : re.int_updates) ns.ints[a.var] = a.val;
          if (!gen_apply(net, se, ns.ints.data(), ns.ints.size()) || !gen_apply(net, re, ns.ints.data(), ns.ints.size())) continue;
          all_invariants(net, ns.locs, inv);
          if (!apply_all(z1, inv)) continue;
          bool nc, nu;
          frozen_flags(net, ns.locs, &nc, &nu);
          if (!nc && !nu) { z1.delay(); apply_all(z1, inv); }
          reset_inactive(net, ns.locs, z1, qmask);
          if (emode == ExLuPlus) z1.extrapolate_lu_plus(L, U);
      else if (emode == ExNoDiag) z1.extrapolate_lu_plus_nodiag(L, U);
          if (z1.empty()) continue;
          ns.zone = std::move(z1);
          out.push_back(std::move(ns));
        }
      }
    }
  }

  // ---- broadcast synchronisation: a! fires with ALL enabled a? (0 or more) ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1 || !net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      // Each other process contributes at most one enabled a? (nondeterministic choice).
      std::vector<std::pair<int, std::vector<int>>> parts;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        std::vector<int> es;
        for (int ej : net.procs[j].edges_of(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan == se.sync_chan && re.sync_dir == -1 && int_ok(re.int_guard, s.ints) && gen_ok(net, re, s.ints.data(), s.ints.size()))
            es.push_back(ej);
        }
        if (!es.empty()) parts.push_back({j, std::move(es)});
      }
      std::vector<std::size_t> pick(parts.size(), 0);
      for (;;) {
        bool part_committed = net.procs[i].locations[s.locs[i]].committed;
        for (const auto& pr : parts)
          if (net.procs[pr.first].locations[s.locs[pr.first]].committed) part_committed = true;
        if (!src_committed || part_committed) {
          Dbm z1 = szone;
          bool ok = apply_all(z1, se.clock_guard);
          for (std::size_t t = 0; ok && t < parts.size(); ++t)
            ok = apply_all(z1, net.procs[parts[t].first].edges[parts[t].second[pick[t]]].clock_guard);
          if (ok) {
            SymState ns;
            ns.locs = s.locs; ns.ints = s.ints;
            ns.locs[i] = se.tgt;
            for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;
            for (int r : se.clock_resets) z1.reset(r);
            for (std::size_t t = 0; t < parts.size(); ++t) {
              const NEdge& re = net.procs[parts[t].first].edges[parts[t].second[pick[t]]];
              ns.locs[parts[t].first] = re.tgt;
              for (const auto& a : re.int_updates) ns.ints[a.var] = a.val;
              for (int r : re.clock_resets) z1.reset(r);
            }
            all_invariants(net, ns.locs, inv);
            if (apply_all(z1, inv)) {
              bool nc, nu;
              frozen_flags(net, ns.locs, &nc, &nu);
              if (!nc && !nu) { z1.delay(); apply_all(z1, inv); }
              reset_inactive(net, ns.locs, z1, qmask);
              if (emode == ExLuPlus) z1.extrapolate_lu_plus(L, U);
              else if (emode == ExNoDiag) z1.extrapolate_lu_plus_nodiag(L, U);
              if (!z1.empty()) { ns.zone = std::move(z1); out.push_back(std::move(ns)); }
            }
          }
        }
        if (parts.empty()) break;
        std::size_t k = 0;
        for (; k < parts.size(); ++k) { if (++pick[k] < parts[k].second.size()) break; pick[k] = 0; }
        if (k == parts.size()) break;
      }
    }
  }
}

// Slot-reusing successor generation for the shared-passed hot path. Semantically
// identical to successors_net above, but instead of building a fresh SymState
// (two pooled loc/int vectors + a DBM) per successor and push_back-ing it -- ~3
// allocations for every one of tens of millions of successors, all discarded by
// add() which keeps only the interned zone -- it BUILDS EACH SUCCESSOR IN PLACE in
// a reused `out` slot. The caller does not clear `out`; it reads out[0..return).
// Because add() interns the zone by copy (never moves the SymState out), the slots
// survive the caller's loop and are reused on the next frontier state, so after a
// short warm-up the per-successor allocation is ZERO. (The legacy successors_net
// stays for the seq/legacy/single-TA paths, which accumulate rather than reuse and
// are slated for removal.)
std::size_t gen_succ_reuse(const Network& net, const SymState& s, const Dbm& szone,
                           const std::vector<std::int32_t>& L, const std::vector<std::int32_t>& U,
                           const std::vector<char>& qmask, int emode, std::vector<SymState>& out) {
  thread_local std::vector<Constraint> inv;
  std::size_t on = 0;
  auto slot = [&]() -> SymState& { if (on == out.size()) out.emplace_back(); return out[on]; };
  bool src_committed, src_urgent;
  frozen_flags(net, s.locs, &src_committed, &src_urgent);

  // ---- location edges ----
  for (int p = 0; p < net.nprocs(); ++p) {
    if (src_committed && !net.procs[p].locations[s.locs[p]].committed) continue;
    for (int ei : net.procs[p].edges_of(s.locs[p])) {
      const NEdge& e = net.procs[p].edges[ei];
      if (e.sync_chan >= 0) continue;
      if (!int_ok(e.int_guard, s.ints) || !gen_ok(net, e, s.ints.data(), s.ints.size())) continue;
      SymState& ns = slot();
      ns.zone = szone;                                    // into reused capacity
      if (!apply_all(ns.zone, e.clock_guard)) continue;
      for (int r : e.clock_resets) ns.zone.reset(r);
      ns.locs.assign(s.locs.begin(), s.locs.end());
      ns.locs[p] = e.tgt;
      ns.ints.assign(s.ints.begin(), s.ints.end());
      for (const auto& a : e.int_updates) ns.ints[a.var] = a.val;
      if (!gen_apply(net, e, ns.ints.data(), ns.ints.size())) continue;
      all_invariants(net, ns.locs, inv);
      if (!apply_all(ns.zone, inv)) continue;
      bool nc, nu;
      frozen_flags(net, ns.locs, &nc, &nu);
      if (!nc && !nu) { ns.zone.delay(); apply_all(ns.zone, inv); }
      reset_inactive(net, ns.locs, ns.zone, qmask);
      if (emode == ExLuPlus) ns.zone.extrapolate_lu_plus(L, U);
      else if (emode == ExNoDiag) ns.zone.extrapolate_lu_plus_nodiag(L, U);
      if (ns.zone.empty()) continue;
      ++on;
    }
  }

  // ---- binary channel synchronisation ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1) continue;
      if (net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      const bool i_comm = net.procs[i].locations[s.locs[i]].committed;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        for (int ej : net.procs[j].edges_of(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan != se.sync_chan || re.sync_dir != -1) continue;
          if (src_committed && !(i_comm || net.procs[j].locations[s.locs[j]].committed)) continue;
          if (!int_ok(re.int_guard, s.ints) || !gen_ok(net, re, s.ints.data(), s.ints.size())) continue;
          SymState& ns = slot();
          ns.zone = szone;
          if (!apply_all(ns.zone, se.clock_guard)) continue;
          if (!apply_all(ns.zone, re.clock_guard)) continue;
          for (int r : se.clock_resets) ns.zone.reset(r);
          for (int r : re.clock_resets) ns.zone.reset(r);
          ns.locs.assign(s.locs.begin(), s.locs.end());
          ns.locs[i] = se.tgt;
          ns.locs[j] = re.tgt;
          ns.ints.assign(s.ints.begin(), s.ints.end());
          for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;
          for (const auto& a : re.int_updates) ns.ints[a.var] = a.val;
          if (!gen_apply(net, se, ns.ints.data(), ns.ints.size()) || !gen_apply(net, re, ns.ints.data(), ns.ints.size())) continue;
          all_invariants(net, ns.locs, inv);
          if (!apply_all(ns.zone, inv)) continue;
          bool nc, nu;
          frozen_flags(net, ns.locs, &nc, &nu);
          if (!nc && !nu) { ns.zone.delay(); apply_all(ns.zone, inv); }
          reset_inactive(net, ns.locs, ns.zone, qmask);
          if (emode == ExLuPlus) ns.zone.extrapolate_lu_plus(L, U);
          else if (emode == ExNoDiag) ns.zone.extrapolate_lu_plus_nodiag(L, U);
          if (ns.zone.empty()) continue;
          ++on;
        }
      }
    }
  }

  // ---- broadcast synchronisation ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1 || !net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      std::vector<std::pair<int, std::vector<int>>> parts;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        std::vector<int> es;
        for (int ej : net.procs[j].edges_of(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan == se.sync_chan && re.sync_dir == -1 && int_ok(re.int_guard, s.ints) && gen_ok(net, re, s.ints.data(), s.ints.size()))
            es.push_back(ej);
        }
        if (!es.empty()) parts.push_back({j, std::move(es)});
      }
      std::vector<std::size_t> pick(parts.size(), 0);
      for (;;) {
        bool part_committed = net.procs[i].locations[s.locs[i]].committed;
        for (const auto& pr : parts)
          if (net.procs[pr.first].locations[s.locs[pr.first]].committed) part_committed = true;
        if (!src_committed || part_committed) {
          SymState& ns = slot();
          ns.zone = szone;
          bool ok = apply_all(ns.zone, se.clock_guard);
          for (std::size_t t = 0; ok && t < parts.size(); ++t)
            ok = apply_all(ns.zone, net.procs[parts[t].first].edges[parts[t].second[pick[t]]].clock_guard);
          if (ok) {
            ns.locs.assign(s.locs.begin(), s.locs.end());
            ns.ints.assign(s.ints.begin(), s.ints.end());
            ns.locs[i] = se.tgt;
            for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;
            for (int r : se.clock_resets) ns.zone.reset(r);
            for (std::size_t t = 0; t < parts.size(); ++t) {
              const NEdge& re = net.procs[parts[t].first].edges[parts[t].second[pick[t]]];
              ns.locs[parts[t].first] = re.tgt;
              for (const auto& a : re.int_updates) ns.ints[a.var] = a.val;
              for (int r : re.clock_resets) ns.zone.reset(r);
            }
            all_invariants(net, ns.locs, inv);
            if (apply_all(ns.zone, inv)) {
              bool nc, nu;
              frozen_flags(net, ns.locs, &nc, &nu);
              if (!nc && !nu) { ns.zone.delay(); apply_all(ns.zone, inv); }
              reset_inactive(net, ns.locs, ns.zone, qmask);
              if (emode == ExLuPlus) ns.zone.extrapolate_lu_plus(L, U);
              else if (emode == ExNoDiag) ns.zone.extrapolate_lu_plus_nodiag(L, U);
              if (!ns.zone.empty()) ++on;
            }
          }
        }
        if (parts.empty()) break;
        std::size_t k = 0;
        for (; k < parts.size(); ++k) { if (++pick[k] < parts[k].second.size()) break; pick[k] = 0; }
        if (k == parts.size()) break;
      }
    }
  }
  return on;
}

void extend_bounds(std::vector<std::int32_t>& L, std::vector<std::int32_t>& U,
                   const std::vector<Constraint>& cs) {
  for (const auto& c : cs) {
    const int v = raw_bound(c.bound);
    if (c.i > 0 && c.j == 0) { if (v > U[c.i]) U[c.i] = v; }
    else if (c.i == 0 && c.j > 0) { const int lb = -v; if (lb > L[c.j]) L[c.j] = lb; }
    else if (c.i > 0 && c.j > 0) {
      const int a = std::abs(v);
      if (a > U[c.i]) U[c.i] = a;
      if (a > U[c.j]) U[c.j] = a;
      if (a > L[c.i]) L[c.i] = a;
      if (a > L[c.j]) L[c.j] = a;
    }
  }
}

}  // namespace

// GPU merge routing mode (see netreach.hpp). Default naive = always the one-thread-per-bucket GPU
// merge (the original, published behavior); --adaptive turns on the calibrated CPU/GPU routing.
static bool g_adaptive_merge = false;
void set_adaptive_merge(bool on) { g_adaptive_merge = on; }
static bool g_gpu_intern = false;
void set_gpu_intern(bool on) { g_gpu_intern = on; }
bool adaptive_merge() { return g_adaptive_merge; }

// Frontier chunking (see netreach.hpp). Off by default = expand a whole BFS level at once.
static bool g_chunking = false;
static int g_chunk_states = 65536;
void set_chunking(bool on, int states) { g_chunking = on; if (states > 0) g_chunk_states = states; }
bool chunking() { return g_chunking; }

#if defined(PTCTLMC_HAVE_METAL) || defined(PTCTLMC_HAVE_CUDA)
// Host port of the net_merge GPU kernel body (dbm_ops.hpp host functions), for ONE discrete-state
// bucket. The GPU runs one thread per bucket, so a level with few/fat buckets stalls on a handful of
// slow lanes; this lets the flat driver merge such buckets across all CPU cores instead. Existing
// antichain is at work_buf[work_off_b .. +exist_cnt_b) (already gathered); new zones are read in
// place from the expand output `oz` at indices new_idx_b[0..new_cnt_b). Writes the final antichain
// back in place, sets WI (isnew flags: existing=0, appended new=1), returns the surviving count.
static int flat_merge_bucket(raw_t* work_buf, int work_off_b, int exist_cnt_b,
                             const raw_t* oz, const int* new_idx_b, int new_cnt_b,
                             const std::int32_t* L, const std::int32_t* U,
                             int dim, bool has_diag, int* WI) {
  const int d2 = dim * dim;
  raw_t* W = work_buf + static_cast<std::size_t>(work_off_b) * d2;
  raw_t zn[kDbmMaxDim * kDbmMaxDim];
  int cur = exist_cnt_b;                              // WI[0..cur) already 0 (existing zones)
  for (int t = 0; t < new_cnt_b; ++t) {
    const raw_t* src = oz + static_cast<std::size_t>(new_idx_b[t]) * d2;
    for (int k = 0; k < d2; ++k) zn[k] = src[k];
    bool subsumed = false;                            // zn subsumed by a stored zone?
    for (int i = 0; i < cur; ++i) {
      const raw_t* zw = W + static_cast<std::size_t>(i) * d2;
      if (has_diag ? dbm_includes(zw, zn, dim) : dbm_is_alu_le(zn, zw, dim, L, U)) { subsumed = true; break; }
    }
    if (subsumed) continue;
    int w = 0;                                        // drop stored zones subsumed by zn
    for (int i = 0; i < cur; ++i) {
      const raw_t* zw = W + static_cast<std::size_t>(i) * d2;
      const bool killed = has_diag ? dbm_includes(zn, zw, dim) : dbm_is_alu_le(zw, zn, dim, L, U);
      if (!killed) {
        if (w != i) { std::memcpy(W + static_cast<std::size_t>(w)*d2, zw, d2*sizeof(raw_t)); WI[w] = WI[i]; }
        ++w;
      }
    }
    cur = w;
    bool merged = true;                               // convex-union absorb (zn becomes the hull)
    while (merged) {
      merged = false;
      for (int kk = 0; kk < cur; ++kk)
        if (dbm_try_union(zn, W + static_cast<std::size_t>(kk)*d2, dim)) {
          for (int i = kk; i < cur - 1; ++i) {
            std::memcpy(W + static_cast<std::size_t>(i)*d2, W + static_cast<std::size_t>(i+1)*d2, d2*sizeof(raw_t));
            WI[i] = WI[i+1];
          }
          --cur; merged = true; break;
        }
    }
    std::memcpy(W + static_cast<std::size_t>(cur)*d2, zn, d2*sizeof(raw_t)); WI[cur] = 1; ++cur;
  }
  return cur;
}
#endif

bool reachable_net(const Network& net, const NetTarget& target, Backend backend, ReachStats* stats) {
  std::vector<std::int32_t> L = net.lbound, U = net.ubound;
  extend_bounds(L, U, target.clock_cond);
  L[0] = U[0] = 0;  // reference clock: mandatory for the aLU subsumption test
  // With diagonal constraints the aLU abstraction is unsound: fall back to exact zones
  // (diagonals kept exact) + plain inclusion subsumption.
  const bool has_diag = network_has_diagonal(net);
  const int emode = has_diag ? ExNoDiag : ExNone;
  std::vector<char> qmask(net.nclocks + 1, 0);  // clocks the query reads -> keep active
  for (const auto& c : target.clock_cond) { if (c.i > 0) qmask[c.i] = 1; if (c.j > 0) qmask[c.j] = 1; }

  // Query clock constants floor every state's aLU bounds (the property may hold anywhere).
  std::vector<std::int32_t> Lq(net.nclocks + 1, kAluNegInf), Uq(net.nclocks + 1, kAluNegInf);
  for (const auto& c : target.clock_cond) {
    const int v = raw_bound(c.bound);
    if (c.i > 0 && c.j == 0) Uq[c.i] = std::max(Uq[c.i], v);
    else if (c.i == 0 && c.j > 0) Lq[c.j] = std::max(Lq[c.j], -v);
    else if (c.i > 0 && c.j > 0) {
      const int a = std::abs(v);
      Uq[c.i] = std::max(Uq[c.i], a); Uq[c.j] = std::max(Uq[c.j], a);
      Lq[c.i] = std::max(Lq[c.i], a); Lq[c.j] = std::max(Lq[c.j], a);
    }
  }
  // Per-state aLU bounds: max of per-location bounds over processes, floored by the query.
  auto state_bounds = [&](const int* locs, std::vector<std::int32_t>& L2,
                          std::vector<std::int32_t>& U2) {
    if (net.lbound_loc.empty()) { L2 = L; U2 = U; return; }  // fall back to global bounds
    L2.assign(net.nclocks + 1, kAluNegInf);
    U2.assign(net.nclocks + 1, kAluNegInf);
    for (int p = 0; p < net.nprocs(); ++p)
      for (int c = 1; c <= net.nclocks; ++c) {
        L2[c] = std::max(L2[c], net.lbound_loc[p][locs[p]][c]);
        U2[c] = std::max(U2[c], net.ubound_loc[p][locs[p]][c]);
      }
    for (int c = 1; c <= net.nclocks; ++c) { L2[c] = std::max(L2[c], Lq[c]); U2[c] = std::max(U2[c], Uq[c]); }
    L2[0] = U2[0] = 0;
  };

  auto is_target = [&](const SymState& s) {
    for (auto& lr : target.loc_req)
      if (s.locs[lr.first] != lr.second) return false;
    if (!int_ok(target.int_cond, s.ints)) return false;
    Dbm t = s.zone;
    return apply_all(t, target.clock_cond);
  };

  bool parallel = (backend == Backend::Omp);
#ifndef PTCTLMC_HAVE_OPENMP
  parallel = false;
#endif
  int nthreads = 1;
#ifdef PTCTLMC_HAVE_OPENMP
  if (parallel) nthreads = omp_get_max_threads();
#endif

  // GPU network backend (Metal or CUDA): the flat GPU-resident driver below runs the whole
  // level loop on the GPU (expand + per-bucket subsumption/convex-union merge). make_net_gpu_*
  // returns null for out-of-scope networks (diagonal constraints) or no device, so we fall back
  // to the CPU path transparently.
  std::unique_ptr<NetGpuExpander> gpu;
  bool use_gpu = false;
#ifdef PTCTLMC_HAVE_METAL
  if (backend == Backend::Metal) gpu = make_net_gpu_metal(net, qmask);
#endif
#ifdef PTCTLMC_HAVE_CUDA
  if (backend == Backend::Cuda) gpu = make_net_gpu_cuda(net, qmask);
#endif
  use_gpu = (gpu != nullptr);
  // SoA work pool (Metal gpusplit merge): host gather/rebuild transpose to/from the transposed
  // layout, and the CPU-merge fallback (AoS) is disabled since it can't read it.
  const bool soa = use_gpu && gpu->soa_work_pool();
  if (use_gpu && std::getenv("PTCTLMC_STATS"))
    std::fprintf(stderr, "[gpu] network expander ACTIVE\n");

  // Passed list SHARDED by discrete-state hash: a given discrete state always lands
  // in the same shard, so each shard is a self-contained antichain owned by exactly
  // one thread during the merge — the O(bucket^2 * dim^3) convex-merge now runs in
  // parallel with no locks. Per discrete state: antichain + convex-union merge.
  // Discrete-state key packed losslessly into a uint64: a fixed bit-field per process
  // location + per int variable. This replaces the per-state std::string key, whose
  // alloc/format/hash dominated the profile — keys are now compared and hashed as plain
  // integers (what UPPAAL does with its "compressed integer" discrete states).
  const int nfields = net.nprocs() + static_cast<int>(net.intvars.size());
  std::vector<int> dshift(nfields);
  std::vector<std::uint64_t> dmask(nfields);  // per-field value mask (for decoding dk -> locs/ints)
  {
    int off = 0;
    auto bits = [](long n) { int b = 0; while ((1LL << b) < n) ++b; return b; };
    auto set = [&](int f, int w) { dshift[f] = off; dmask[f] = (w >= 64) ? ~0ULL : ((1ULL << w) - 1); off += w; };
    for (int p = 0; p < net.nprocs(); ++p) set(p, bits(static_cast<long>(net.procs[p].locations.size())));
    for (std::size_t v = 0; v < net.intvars.size(); ++v)
      set(net.nprocs() + static_cast<int>(v), bits(net.intvars[v].hi - net.intvars[v].lo + 1));
    // >= 64 (not > 64): keeps every field offset < 64, so a trailing zero-width
    // field (single-value int/location) can never trigger a `value << 64` UB shift.
    if (off >= 64) {
      // Say WHAT overflowed, not just that something did: the widest fields are almost
      // always a few int arrays, and the user can only act on the breakdown.
      std::vector<std::pair<int, std::string>> w;
      for (int p = 0; p < net.nprocs(); ++p)
        w.emplace_back(bits(static_cast<long>(net.procs[p].locations.size())),
                       "proc[" + std::to_string(p) + "] (" +
                       std::to_string(net.procs[p].locations.size()) + " locs)");
      for (std::size_t v = 0; v < net.intvars.size(); ++v)
        w.emplace_back(bits(net.intvars[v].hi - net.intvars[v].lo + 1),
                       "int " + net.intvars[v].name + " [" + std::to_string(net.intvars[v].lo) +
                       "," + std::to_string(net.intvars[v].hi) + "]");
      std::stable_sort(w.begin(), w.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      std::string msg = "reachable_net: discrete state needs " + std::to_string(off) +
                        " bits but the packed key holds 63 (network too wide).\n"
                        "  " + std::to_string(net.nprocs()) + " processes + " +
                        std::to_string(net.intvars.size()) + " int vars. Widest fields:";
      for (std::size_t i = 0; i < w.size() && i < 8; ++i)
        msg += "\n    " + std::to_string(w[i].first) + " bits  " + w[i].second;
      throw std::runtime_error(msg);
    }
  }
  // Decode a packed key back into location + int vectors (inverse of disc_key).
  auto decode_key = [&](std::uint64_t dk, int* locs, long* ints) {
    for (int p = 0; p < net.nprocs(); ++p) locs[p] = static_cast<int>((dk >> dshift[p]) & dmask[p]);
    for (std::size_t v = 0; v < net.intvars.size(); ++v)
      ints[v] = static_cast<long>((dk >> dshift[net.nprocs() + v]) & dmask[net.nprocs() + v]) + net.intvars[v].lo;
  };
  auto disc_key = [&](const LocVec& locs, const IntVec& ints) -> std::uint64_t {
    std::uint64_t k = 0;
    for (int p = 0; p < static_cast<int>(locs.size()); ++p)
      k |= static_cast<std::uint64_t>(static_cast<unsigned>(locs[p])) << dshift[p];
    for (std::size_t v = 0; v < ints.size(); ++v)
      k |= static_cast<std::uint64_t>(ints[v] - net.intvars[v].lo) << dshift[net.nprocs() + v];
    return k;
  };

  // initial symbolic state
  SymState s0;
  s0.locs.resize(net.nprocs());
  for (int p = 0; p < net.nprocs(); ++p) s0.locs[p] = net.procs[p].initial;
  s0.ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0.ints[v] = net.intvars[v].init;
  s0.zone = Dbm(net.nclocks);
  {
    std::vector<Constraint> inv;
    all_invariants(net, s0.locs, inv);
    if (!apply_all(s0.zone, inv)) return false;
    bool c, u;
    frozen_flags(net, s0.locs, &c, &u);
    if (!c && !u) { s0.zone.delay(); apply_all(s0.zone, inv); }
    reset_inactive(net, s0.locs, s0.zone, qmask);
    // no extrapolation: exact zones + aLU subsumption keep the passed list finite & minimal
  }
  if (is_target(s0)) return true;

#if defined(PTCTLMC_HAVE_METAL) || defined(PTCTLMC_HAVE_CUDA)
  // ---- FLAT GPU-resident driver (Metal / CUDA) -------------------------------------------
  // Everything stays as flat zone arrays across levels: no SymState round-trip, no per-level
  // rebuild of a map of Dbm copies (which the profiler showed dominated the metal wall time).
  // GPU does expand + the per-bucket subsumption/convex-union merge; the host only groups by
  // dk (cheap integer hashing) and marshals flat zone spans.
  if (use_gpu) {
    const int np = net.nprocs(), ni = static_cast<int>(net.intvars.size()), dim = net.nclocks + 1, d2 = dim * dim;
    auto int_ok_flat = [&](const std::vector<IntAtom>& g, const int* iv) {
      for (const auto& a : g) {
        const long x = iv[a.var]; bool r;
        switch (a.op) { case IntOp::Eq: r = x==a.val; break; case IntOp::Ne: r = x!=a.val; break;
          case IntOp::Lt: r = x<a.val; break; case IntOp::Le: r = x<=a.val; break;
          case IntOp::Gt: r = x>a.val; break; default: r = x>=a.val; }
        if (!r) return false;
      }
      return true;
    };
    // The passed set holds ZONE INDICES into a shared dense pool rather than a private dense copy
    // per (discrete state, zone). With `--gpu-intern` the pool DEDUPLICATES, which is a large win
    // wherever few distinct zones are reachable: viking15 stores 2.77M zones drawn from just 120
    // distinct ones, i.e. GBs of unified memory representing ~120 KB of zone data. Without the
    // flag the pool simply appends, reproducing the old layout's memory with the same indexed
    // access path, so the flag isolates deduplication from everything else.
    GpuZoneIntern zi(d2);
    zi.set_mode(g_gpu_intern ? GpuZoneIntern::Mode::Dedup : GpuZoneIntern::Mode::Append);
    std::unordered_map<std::uint64_t, std::vector<int>> passed;   // dk -> antichain zone indices
    std::vector<int> fl(s0.locs.begin(), s0.locs.end());
    std::vector<long> fi(s0.ints.begin(), s0.ints.end());
    std::vector<raw_t> fz(s0.zone.data().begin(), s0.zone.data().end());
    passed[disc_key(s0.locs, s0.ints)].assign(1, zi.intern(s0.zone.data().data()));
    // Optional phase profiler (PTCTLMC_MTLPROF): host-side timing of the two synchronous GPU
    // calls (expand/merge, which block until the kernel finishes since outputs are read in place)
    // vs. the host marshaling/rebuild. Zero overhead when the env var is unset.
    const bool mtlprof = std::getenv("PTCTLMC_MTLPROF") != nullptr;
    // Memory accounting (PTCTLMC_MEMPROF): where the resident bytes actually go — the passed list
    // (live zone bytes vs vector CAPACITY, i.e. growth slack), the duplicated frontier copy, and the
    // transient per-level pools the GPU buffers are sized to. Zero cost when unset.
    const bool memprof = std::getenv("PTCTLMC_MEMPROF") != nullptr;
    // CPU-expand / GPU-merge hybrid (PTCTLMC_HYBRID=1): each phase on the hardware the
    // measurements say it wins on. Zero-copy via unified memory.
    const bool hybrid = std::getenv("PTCTLMC_HYBRID") != nullptr;
    int hyb_threads = 1;
    std::size_t hyb_ratio = 8;   // successors staged per frontier state (adapts per level)
    int hyb_prev_nb = 1024;      // previous level's distinct-bucket count (sizes the bucket table)
#ifdef PTCTLMC_HAVE_OPENMP
    hyb_threads = omp_get_max_threads();
#endif
    std::size_t peak_nwork = 0, peak_wtot = 0, peak_front = 0;
    std::size_t peak_rss_ = 0, peak_rss_heap_ = 0, peak_rss_gpu_ = 0;
    double t_expand = 0, t_merge = 0, t_bucket = 0, t_gather = 0, t_rebuild = 0, t_binit = 0, t_bmain = 0, t_bcnt = 0, t_bpfx = 0, t_bscat = 0;
    int lvl_cpu = 0, lvl_gpu = 0;   // merge routing actually taken, per level
    std::size_t tot_work = 0, tot_valid = 0;   // GPU lane utilisation: survivors / work items
    int prof_nb = 0, prof_maxsz = 0; double prof_meansz = 0; long long prof_peak_sum2 = 0;
    double gpu_rate = -1, cpu_rate = -1;   // adaptive merge routing: work-units (sum bucket^2) per ms
    auto _now = [] { return std::chrono::steady_clock::now(); };
    auto _ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    // Rebuild sub-phase attribution (PTCTLMC_RBPROF): copy/transpose vs intern vs frontier write.
    const bool rbprof = std::getenv("PTCTLMC_RBPROF") != nullptr;
    const int RBT = std::max(1, hyb_threads);
    std::vector<double> rb_copy(RBT * 8, 0.0), rb_intern(RBT * 8, 0.0), rb_front(RBT * 8, 0.0);
    auto rbnow = [](bool on) {
      return on ? std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now().time_since_epoch()).count() : 0.0;
    };
    auto ompt = [] {
#ifdef PTCTLMC_HAVE_OPENMP
      return omp_get_thread_num() * 8;   // stride 8 doubles = 64B, one cache line per thread
#else
      return 0;
#endif
    };
    const auto _wall0 = _now();
    const int niw = ni ? ni : 1;   // ints stride (kept >=1 so the vector is never zero-sized)
    while (!fl.empty()) {
      const int total = static_cast<int>(fl.size() / np);
      // --chunking: expand only a bounded PREFIX of the frontier this iteration and carry the tail
      // forward. The frontier is a worklist, so batching it only changes the schedule; the GPU pools
      // are then sized to a chunk instead of the widest level, making peak memory independent of
      // level width. Default (off) expands the whole level, exactly as before.
      const int nfr = g_chunking ? std::min(total, g_chunk_states) : total;
      const bool whole = (nfr == total);
      if (stats) stats->explored += nfr;
      const int* ev = nullptr; const int* ol = nullptr; const int* oi = nullptr; const raw_t* oz = nullptr;
      const auto _te = _now();
      std::size_t nwork;
      if (hybrid) {
        // CPU-EXPAND / GPU-MERGE, ZERO-COPY. Successor generation is branchy,
        // serially-dependent small-matrix work the CPU wins; the per-bucket
        // subsumption/convex-union merge is thousands of independent jobs the GPU wins.
        // Unified memory lets each phase run where it is fastest with NO transfer: the
        // CPU expander writes successors DIRECTLY into the buffers the merge kernel
        // reads. Threads claim slots in blocks (one atomic per BLK successors, not per
        // successor), so the only per-successor cost is the write itself.
        const int NT = hyb_threads;
        constexpr std::size_t BLK = 256;
        const std::size_t cap = std::max<std::size_t>(
            static_cast<std::size_t>(nfr) * hyb_ratio + BLK * NT, 1024);
        int *sv = nullptr, *sl = nullptr, *si2 = nullptr; raw_t* sz = nullptr;
        gpu->stage_expand(cap, np, ni, d2, sv, sl, si2, sz);
        // Zero the flags once instead of tracking each thread's unused block tail: a
        // slot is a successor only if it was written, so unclaimed holes stay invalid.
        std::memset(sv, 0, cap * sizeof(int));
        std::atomic<std::size_t> wnext{0};
        // Overflow spill (only when this level's fan-out beat the estimate).
        std::vector<std::vector<int>>   spl(NT);
        std::vector<std::vector<long>>  spi(NT);
        std::vector<std::vector<raw_t>> spz(NT);
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel num_threads(NT)
#endif
        {
          int tid = 0;
#ifdef PTCTLMC_HAVE_OPENMP
          tid = omp_get_thread_num();
#endif
          std::vector<SymState> buf;
          SymState cur; cur.locs.resize(np); cur.ints.resize(ni);
          Dbm pz(net.nclocks);
          std::size_t mi = 0, me = 0;
          bool spilled = false;
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp for schedule(dynamic, 8) nowait
#endif
          for (long i = 0; i < static_cast<long>(nfr); ++i) {
            for (int p = 0; p < np; ++p) cur.locs[p] = fl[static_cast<std::size_t>(i) * np + p];
            for (int v = 0; v < ni; ++v) cur.ints[v] = fi[static_cast<std::size_t>(i) * niw + v];
            std::copy(fz.begin() + static_cast<std::size_t>(i) * d2,
                      fz.begin() + static_cast<std::size_t>(i + 1) * d2, pz.data().begin());
            const std::size_t n = gen_succ_reuse(net, cur, pz, L, U, qmask, emode, buf);
            for (std::size_t k = 0; k < n; ++k) {
              const SymState& ns = buf[k];
              if (!spilled && mi == me) {                       // claim the next block
                const std::size_t b = wnext.fetch_add(BLK, std::memory_order_relaxed);
                if (b + BLK > cap) spilled = true; else { mi = b; me = b + BLK; }
              }
              if (!spilled) {
                const std::size_t w = mi++;
                sv[w] = 1;
                for (int p = 0; p < np; ++p) sl[w * np + p] = ns.locs[p];
                for (int v = 0; v < ni; ++v) si2[w * ni + v] = static_cast<int>(ns.ints[v]);
                std::memcpy(sz + w * d2, ns.zone.data().data(), d2 * sizeof(raw_t));
              } else {
                spl[tid].insert(spl[tid].end(), ns.locs.begin(), ns.locs.end());
                spi[tid].insert(spi[tid].end(), ns.ints.begin(), ns.ints.end());
                spz[tid].insert(spz[tid].end(), ns.zone.data().begin(), ns.zone.data().end());
              }
            }
          }
        }
        const std::size_t direct = std::min<std::size_t>(wnext.load(), cap);
        std::size_t spill = 0;
        for (int t = 0; t < NT; ++t) spill += spz[t].size() / d2;
        if (spill) {   // estimate was low: keep what is written, append the tail
          gpu->stage_expand(direct + spill, np, ni, d2, sv, sl, si2, sz, /*preserve=*/true);
          std::size_t w = direct;
          for (int t = 0; t < NT; ++t) {
            const std::size_t m = spz[t].size() / d2;
            for (std::size_t k = 0; k < m; ++k, ++w) {
              sv[w] = 1;
              for (int p = 0; p < np; ++p) sl[w * np + p] = spl[t][k * np + p];
              for (int v = 0; v < ni; ++v) si2[w * ni + v] = static_cast<int>(spi[t][k * ni + v]);
              std::memcpy(sz + w * d2, spz[t].data() + k * d2, d2 * sizeof(raw_t));
            }
          }
        }
        nwork = direct + spill;
        // Adapt the estimate to what this level actually produced (x2 headroom), so the
        // spill path stays cold even as the fan-out changes across the search.
        if (nfr) hyb_ratio = std::max<std::size_t>(4, 2 * (nwork / nfr + 1));
        ev = sv; ol = sl; oi = si2; oz = sz;
      } else if (whole) {
        nwork = gpu->expand(nfr, fl, fi, fz, ev, ol, oi, oz);   // no copy in the default path
      } else {
        std::vector<int>   cfl(fl.begin(), fl.begin() + static_cast<std::size_t>(nfr) * np);
        std::vector<long>  cfi(fi.begin(), fi.begin() + static_cast<std::size_t>(nfr) * niw);
        std::vector<raw_t> cfz(fz.begin(), fz.begin() + static_cast<std::size_t>(nfr) * d2);
        nwork = gpu->expand(nfr, cfl, cfi, cfz, ev, ol, oi, oz);
      }
      if (mtlprof) {
        const double _dte = _ms(_te, _now());
        t_expand += _dte; tot_work += nwork;
        // Per-level (items, ms) so the fixed-vs-variable split of a dispatch can be REGRESSED
        // from many points within one model, instead of inferred from a two-point fit across
        // models (which is exactly determined, hence unfalsifiable -- and wrong: it mispredicts
        // arbiter17 by 2.7x because per-item cost scales with the zone size).
        if (std::getenv("PTCTLMC_LVLPROF"))
          std::fprintf(stderr, "[lvl] frontier=%d nwork=%zu expand_ms=%.3f\n", nfr, nwork, _dte);
      }
      if (memprof && nwork > peak_nwork) peak_nwork = nwork;
      if (nwork == 0) {
        if (whole) break;
        // this chunk produced nothing: drop it and continue with the remaining frontier
        std::vector<int>   tl(fl.begin() + static_cast<std::size_t>(nfr) * np,  fl.end());
        std::vector<long>  ti(fi.begin() + static_cast<std::size_t>(nfr) * niw, fi.end());
        std::vector<raw_t> tz(fz.begin() + static_cast<std::size_t>(nfr) * d2,  fz.end());
        fl.swap(tl); fi.swap(ti); fz.swap(tz);
        continue;
      }
      // The GPU now holds its own copy of the frontier (persistent device buffer) and the host
      // copies are dead until the end-of-level swap. Release them NOW: otherwise they sit alongside
      // the next frontier being built during the rebuild, i.e. a full extra copy of the frontier at
      // exactly the moment memory peaks.
      // Only safe when the whole frontier was expanded; when chunking, the tail is still needed.
      if (whole) { std::vector<int>().swap(fl); std::vector<long>().swap(fi); std::vector<raw_t>().swap(fz); }
      // group VALID successors by dk (reading straight from GPU buffers); target check on the fly
      // Bucket the valid successors with a FLAT (CSR) layout: a per-work-item bucket id plus a
      // per-bucket count, then a prefix sum. The obvious `vector<vector<int>>` heap-allocates one
      // small vector PER BUCKET PER LEVEL (millions of alloc/free pairs over a run, and the
      // fragmentation dominated resident memory); this is two flat arrays instead.
      const auto _tb = _now();
      // Bucketing stays SERIAL by measurement, not by omission. A lock-free open-addressing
      // table (one CAS per key, stable counting-sort scatter) was implemented and measured
      // three ways -- oversized table 328ms, nb-sized 467ms, monotonic 336ms -- against this
      // loop's ~200ms. A private-map parallel version (per-thread maps + distinct-key merge +
      // parallel counting-sort scatter) was also measured: 268ms -> 241ms on viking15, i.e. ~1%
      // of wall time, because the loop is bound by streaming the expand output out of unified
      // memory while the GPU is using the same bandwidth -- not by the hashing. Neither variant
      // is worth the complexity here. See docs/GUIDE.md.
      std::unordered_map<std::uint64_t, int> bidx;
      std::vector<std::uint64_t> bdk;
      std::vector<int> bcnt;                              // per-bucket count of new zones
      std::vector<int> w_bucket(nwork, -1);               // work item -> bucket id (-1 = not a successor)
      for (std::size_t w = 0; w < nwork; ++w) {
        if (!ev[w]) continue;
        if (mtlprof) ++tot_valid;
        if (stats) ++stats->generated;
        const int* wl = ol + w * np;
        const int* wi = oi + w * ni;
        bool tgt = true;
        for (const auto& lr : target.loc_req) if (wl[lr.first] != lr.second) { tgt = false; break; }
        if (tgt && !int_ok_flat(target.int_cond, wi)) tgt = false;
        if (tgt) {
          Dbm td(net.nclocks);
          std::copy(oz + w * d2, oz + w * d2 + d2, td.data().begin());
          if (apply_all(td, target.clock_cond)) return true;
        }
        std::uint64_t dk = 0;
        for (int p = 0; p < np; ++p) dk |= static_cast<std::uint64_t>(static_cast<unsigned>(wl[p])) << dshift[p];
        for (int v = 0; v < ni; ++v) dk |= static_cast<std::uint64_t>(wi[v] - net.intvars[v].lo) << dshift[np + v];
        auto it = bidx.find(dk);
        int bi;
        if (it == bidx.end()) { bi = static_cast<int>(bdk.size()); bidx.emplace(dk, bi); bdk.push_back(dk); bcnt.push_back(0); }
        else bi = it->second;
        w_bucket[w] = bi; ++bcnt[bi];
      }
      const int nb = static_cast<int>(bdk.size());
      const auto _tp = _now();
      std::vector<int> work_off(nb), exist_cnt(nb), new_off(nb), new_cnt(nb);
      std::vector<std::int32_t> Lm(static_cast<std::size_t>(nb) * dim), Um(static_cast<std::size_t>(nb) * dim);
      // counts + prefix sums. The NEW zones are never copied: the merge reads them in place from
      // the expand output by work-item index, so we only flatten the index list (integers, cheap).
      std::size_t ntot = 0, wtot = 0;  // total new zones; total work-pool zones (exist+new)
      for (int b = 0; b < nb; ++b) {
        new_cnt[b] = bcnt[b];
        auto pit = passed.find(bdk[b]);
        exist_cnt[b] = (pit == passed.end()) ? 0 : static_cast<int>(pit->second.size());
        new_off[b] = static_cast<int>(ntot); ntot += new_cnt[b];
        work_off[b] = static_cast<int>(wtot); wtot += exist_cnt[b] + new_cnt[b];
      }
      // Scatter the work-item indices into the flat CSR array. Iterating w ascending preserves the
      // per-bucket ordering the merge relies on (the convex union is order-sensitive), so this is
      // bit-for-bit equivalent to the old vector-of-vectors.
      if (mtlprof) t_bpfx += _ms(_tp, _now());
      std::vector<int> new_idx(ntot);
      {
        std::vector<int> cursor(new_off);
        for (std::size_t w = 0; w < nwork; ++w) {
          const int bi = w_bucket[w];
          if (bi >= 0) new_idx[cursor[bi]++] = static_cast<int>(w);
        }
      }
      if (mtlprof) t_bucket += _ms(_tb, _now());
      { std::vector<int>().swap(w_bucket); }   // dead once scattered — release before the merge peak
      if (memprof && wtot > peak_wtot) peak_wtot = wtot;
      const auto _tg = _now();
      raw_t* work_buf = nullptr;
      gpu->stage_merge(wtot, d2, work_buf);
      // PARALLEL: gather only the EXISTING antichains into the work buffer (unified memory);
      // successor zones stay in the expand output and are read in place by the merge kernel.
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 32)
#endif
      for (int b = 0; b < nb; ++b) {
        auto pit = passed.find(bdk[b]);
        if (pit != passed.end()) {
          raw_t* dst = work_buf + static_cast<std::size_t>(work_off[b]) * d2;
          if (soa) {   // transpose AoS antichain -> SoA pool: zone i entry k at dst[k*cap + i]
            const std::size_t cap = static_cast<std::size_t>(exist_cnt[b]) + new_cnt[b];
            for (int i = 0; i < exist_cnt[b]; ++i) {
              const raw_t* src = zi.get(pit->second[i]);
              for (int k = 0; k < d2; ++k) dst[static_cast<std::size_t>(k) * cap + i] = src[k];
            }
          } else {
            for (int i = 0; i < exist_cnt[b]; ++i)
              std::memcpy(dst + static_cast<std::size_t>(i) * d2, zi.get(pit->second[i]),
                          static_cast<std::size_t>(d2) * sizeof(raw_t));
          }
        }
        int dl[64]; long di[32];
        std::vector<std::int32_t> L2, U2;
        decode_key(bdk[b], dl, di);
        state_bounds(dl, L2, U2);
        for (int c = 0; c < dim; ++c) { Lm[static_cast<std::size_t>(b) * dim + c] = L2[c]; Um[static_cast<std::size_t>(b) * dim + c] = U2[c]; }
      }
      if (mtlprof) t_gather += _ms(_tg, _now());
      std::vector<int> out_work_off, out_count;
      const raw_t* out_zones = nullptr;
      const int* out_isnew = nullptr;
      // Merge routing. NAIVE (default): always the GPU one-thread-per-bucket merge (original
      // behavior). ADAPTIVE (--adaptive): CALIBRATE each path's throughput (work ~ sum bucket^2, per
      // ms) on early BFS levels, then route each later level to the predicted-faster of {GPU, 12 CPU
      // cores}. This rescues merge-heavy models (e.g. philosophers, where the GPU merge is 93% of the
      // wall time — one thread per bucket is the wrong shape) with no effect where the GPU already wins.
      long long sum2 = 0, sumsz = 0; int maxsz = 0;
      for (int b = 0; b < nb; ++b) { const long long s = exist_cnt[b] + new_cnt[b]; sum2 += s*s; sumsz += s; if (s > maxsz) maxsz = (int)s; }
      if (mtlprof && sum2 > prof_peak_sum2) { prof_peak_sum2 = sum2; prof_nb = nb; prof_maxsz = maxsz; prof_meansz = nb ? (double)sumsz/nb : 0; }
      bool cpu_merge = false;
      if (g_adaptive_merge && !soa) {   // SoA pool is unreadable by the AoS CPU merge -> GPU only
        if (cpu_rate < 0) cpu_merge = true;         // calibrate the CPU path first (never catastrophic)
        else if (gpu_rate < 0) cpu_merge = false;   // then calibrate the GPU path
        else cpu_merge = (cpu_rate >= gpu_rate);    // route to the higher-throughput path
      }
      std::vector<int> host_isnew_v, host_count;
      const auto _tm = _now();
      if (cpu_merge) {
        host_isnew_v.assign(wtot, 0);          // existing zones start not-new; appended new zones -> 1
        host_count.assign(nb, 0);
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
        for (int b = 0; b < nb; ++b)
          host_count[b] = flat_merge_bucket(work_buf, work_off[b], exist_cnt[b], oz,
                                            new_idx.data() + new_off[b], new_cnt[b],
                                            Lm.data() + static_cast<std::size_t>(b)*dim,
                                            Um.data() + static_cast<std::size_t>(b)*dim,
                                            dim, has_diag, host_isnew_v.data() + work_off[b]);
        out_work_off = work_off; out_count = std::move(host_count);
        out_zones = work_buf; out_isnew = host_isnew_v.data();
      } else {
        gpu->merge(nb, dim, has_diag, work_off, exist_cnt, new_idx, new_off, new_cnt, Lm, Um,
                   out_work_off, out_count, out_zones, out_isnew);
      }
      if (cpu_merge) ++lvl_cpu; else ++lvl_gpu;
      const double _dtm = _ms(_tm, _now());
      if (mtlprof) t_merge += _dtm;
      if (g_adaptive_merge && sum2 > 0 && _dtm > 0.5) {   // update throughput calibration (skip trivial levels)
        const double rate = static_cast<double>(sum2) / _dtm;
        if (cpu_merge) cpu_rate = (cpu_rate < 0) ? rate : 0.5 * cpu_rate + 0.5 * rate;
        else           gpu_rate = (gpu_rate < 0) ? rate : 0.5 * gpu_rate + 0.5 * rate;
      }
      // rebuild passed + assemble the flat next frontier (survivors). Prefix-sum the per-bucket
      // survivor counts, pre-create the passed nodes, then fill everything in parallel.
      const auto _tr = _now();
      // Survivor counts, then offsets. Counting scans every merged zone -- O(total zones), the
      // same order as the rebuild copies below -- so it runs in parallel; only the offset
      // prefix-sum stays serial, and that is O(nb) integer adds over data already in cache.
      // (Fusing the two, as this loop used to, forced the whole O(total zones) scan to be
      // serial purely because `frtot` was carried through it.)
      // Survivor counts and offsets. Kept fused and serial deliberately: splitting the count
      // into an omp parallel-for and leaving only the offset prefix-sum serial was measured on
      // arbiter17/Metal and is not faster (rebuild 563ms vs 545ms) -- buckets average ~10 zones,
      // so the scan is memory-bound and the per-bucket parallel overhead plus false sharing on
      // fr_cnt costs more than the scan itself.
      std::vector<int> fr_off(nb), fr_cnt(nb);
      std::size_t frtot = 0;
      for (int b = 0; b < nb; ++b) {
        int c = 0;
        for (int i = 0; i < out_count[b]; ++i) if (out_isnew[static_cast<std::size_t>(out_work_off[b]) + i]) ++c;
        fr_cnt[b] = c; fr_off[b] = static_cast<int>(frtot); frtot += c;
      }
      if (memprof && frtot > peak_front) peak_front = frtot;
      std::vector<int> nl(frtot * np);
      std::vector<long> nin(frtot * (ni ? ni : 1));
      std::vector<raw_t> nz(frtot * d2);
      // A bucket with NO survivor had all its new zones subsumed — its antichain is unchanged, so
      // `passed[dk]` is already correct. Skip its rebuild copy AND frontier gather entirely; only
      // changed buckets are cleared + refilled (their element ptrs stay stable across later inserts).
      std::vector<std::vector<int>*> pzp(nb, nullptr);
      for (int b = 0; b < nb; ++b) if (fr_cnt[b] > 0) { pzp[b] = &passed[bdk[b]]; }
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 32)
#endif
      for (int b = 0; b < nb; ++b) {
        if (fr_cnt[b] == 0) continue;   // unchanged bucket — nothing to do
        int dl[64]; long di[32];
        decode_key(bdk[b], dl, di);
        auto& pz = *pzp[b];
        pz.resize(static_cast<std::size_t>(out_count[b]));
        const std::size_t poolbase = static_cast<std::size_t>(out_work_off[b]) * d2;
        const std::size_t cap = static_cast<std::size_t>(exist_cnt[b]) + new_cnt[b];
        int fpos = fr_off[b];
        std::vector<raw_t> tmp(d2);        // one merged zone, reused across this bucket
        const int _tid = ompt();
        for (int i = 0; i < out_count[b]; ++i) {
          const std::size_t slot = static_cast<std::size_t>(out_work_off[b]) + i;
          raw_t* pzi = tmp.data();
          const auto _r0 = rbnow(rbprof);
          if (soa)   // transpose SoA pool -> AoS zone: entry k at poolbase + k*cap + i
            for (int k = 0; k < d2; ++k) pzi[k] = out_zones[poolbase + static_cast<std::size_t>(k) * cap + i];
          else
            std::memcpy(pzi, out_zones + slot * d2, static_cast<std::size_t>(d2) * sizeof(raw_t));
          const auto _r1 = rbnow(rbprof);
          pz[i] = zi.intern(pzi);
          const auto _r2 = rbnow(rbprof);
          if (rbprof) { rb_copy[_tid] += _r1 - _r0; rb_intern[_tid] += _r2 - _r1; }
          if (out_isnew[slot]) {
            for (int p = 0; p < np; ++p) nl[static_cast<std::size_t>(fpos) * np + p] = dl[p];
            for (int v = 0; v < ni; ++v) nin[static_cast<std::size_t>(fpos) * ni + v] = di[v];
            std::memcpy(nz.data() + static_cast<std::size_t>(fpos) * d2, pzi, static_cast<std::size_t>(d2) * sizeof(raw_t));
            ++fpos;
            if (rbprof) rb_front[_tid] += rbnow(true) - _r2;
          }
        }
      }
      if (mtlprof) t_rebuild += _ms(_tr, _now());
#ifdef __APPLE__
      if (memprof) {   // sample the TRUE resident peak per level, with its composition
        mach_task_basic_info info{};
        mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &cnt) == KERN_SUCCESS &&
            info.resident_size > peak_rss_) {
          malloc_statistics_t ms{}; malloc_zone_statistics(malloc_default_zone(), &ms);
          peak_rss_ = info.resident_size;
          peak_rss_heap_ = ms.size_in_use;
          peak_rss_gpu_ = gpu->device_bytes();
        }
      }
#endif
      // When chunking, the unexpanded tail of this frontier carries over to the next iteration.
      if (!whole) {
        nl.insert(nl.end(),  fl.begin() + static_cast<std::size_t>(nfr) * np,  fl.end());
        nin.insert(nin.end(), fi.begin() + static_cast<std::size_t>(nfr) * niw, fi.end());
        nz.insert(nz.end(),  fz.begin() + static_cast<std::size_t>(nfr) * d2,  fz.end());
      }
      fl.swap(nl); fi.swap(nin); fz.swap(nz);
    }
    if (memprof) {
      std::size_t live = 0, cap = 0;
      for (const auto& kv : passed) { live += kv.second.size(); cap += kv.second.capacity(); }
      const double GB = 1024.0 * 1024.0 * 1024.0, B = sizeof(raw_t);
      // per-bucket container overhead: unordered_map node (~48 B: hash+next+key+vector object) + malloc header
      const double node_ov = passed.size() * 64.0;
      std::fprintf(stderr,
        "[memprof] buckets=%zu  zones=%zu\n"
        "          passed live   = %6.2f GB   (the actual data)\n"
        "          passed capacity= %6.2f GB   (+%.0f%% vector growth slack)\n"
        "          bucket overhead= %6.2f GB   (~64 B/bucket: map node + malloc header)\n"
        "          frontier copy  = %6.2f GB   (peak %zu zones — DUPLICATES passed)\n"
        "          gpu expand pool= %6.2f GB   (peak %zu work items, persistent)\n"
        "          gpu merge pool = %6.2f GB   (peak %zu work zones, persistent)\n"
        "          GPU buffers ACTUAL = %6.2f GB  (all 7 persistent pools, measured)\n"
        "          ---- accounted = %6.2f GB\n",
        passed.size(), live / (std::size_t)d2,
        live * B / GB, cap * B / GB, live ? 100.0 * (double)(cap - live) / (double)live : 0.0,
        node_ov / GB,
        peak_front * (double)d2 * B / GB, peak_front,
        peak_nwork * (double)d2 * B / GB, peak_nwork,
        peak_wtot * (double)d2 * B / GB, peak_wtot,
        gpu->device_bytes() / GB,
        (cap * B + node_ov + peak_front * (double)d2 * B + gpu->device_bytes()) / GB);
#ifdef __APPLE__
      // Separate "bytes the program is using" from "bytes malloc is holding but not returning to
      // the OS". A large gap here means the cost is allocator RETENTION from per-level churn, not
      // live data — a different fix (pooling) than shrinking any one structure.
      {
        malloc_statistics_t st{};
        malloc_zone_statistics(malloc_default_zone(), &st);
        std::fprintf(stderr,
          "          malloc in-use  = %6.2f GB   (live heap)\n"
          "          malloc reserved= %6.2f GB   (held by allocator, not returned to OS)\n"
          "          >>> PEAK RSS   = %6.2f GB   at peak: heap %.2f GB + gpu %.2f GB = %.2f GB\n",
          st.size_in_use / GB, st.size_allocated / GB,
          peak_rss_ / GB, peak_rss_heap_ / GB, peak_rss_gpu_ / GB,
          (peak_rss_heap_ + peak_rss_gpu_) / GB);
      }
#endif
    }
    if (mtlprof) {
      const double wall = _ms(_wall0, _now());
      std::fprintf(stderr, "[mtlprof] expand=%.0fms merge=%.0fms host=%.0fms wall=%.0fms  "
                   "(merge %.0f%%)  peak-level: nb=%d meansz=%.1f maxsz=%d\n"
                   "[mtlprof]   host split: bucket=%.0fms (passed-lookups %.0fms) gather=%.0fms rebuild=%.0fms other=%.0fms\n"
                   "[mtlprof]   MERGE ROUTING: %d levels CPU, %d levels GPU\n"
                   "[mtlprof]   LANE UTILISATION: %zu work items -> %zu survivors (%.1f%% survive)\n",
                   t_expand, t_merge, wall - t_expand - t_merge, wall,
                   100.0 * t_merge / wall, prof_nb, prof_meansz, prof_maxsz,
                   t_bucket, t_bpfx, t_gather, t_rebuild,
                   wall - t_expand - t_merge - t_bucket - t_gather - t_rebuild, lvl_cpu, lvl_gpu, tot_work, tot_valid,
                   tot_work ? 100.0 * (double)tot_valid / (double)tot_work : 0.0);
      if (rbprof) {
        double c = 0, in = 0, f = 0;
        for (int t = 0; t < RBT; ++t) { c += rb_copy[t*8]; in += rb_intern[t*8]; f += rb_front[t*8]; }
        std::fprintf(stderr, "[rbprof]   rebuild CPU-time split: copy/transpose=%.0fms intern=%.0fms frontier-write=%.0fms\n", c, in, f);
      }
    }
    return false;
  }
#endif

  // ======================================================================
  // SHARED-PASSED (hybrid): one stripe-locked passed/waiting table that any
  // worker may touch, locking only the stripe of the discrete state it is
  // updating -- LTSmin's "shared table + fine-grained locking, no static
  // partition" (Alg. 3/4). The per-bucket update is unchanged: PTCTLMC's exact
  // aLU subsumption + convex-union merge (add_to). Expansion and insertion are
  // FUSED, so there is no serial per-level routing barrier; strict-BFS level
  // synchronisation is kept (it minimises the state count). At 1 thread this is
  // bit-identical to the legacy path (same per-bucket merge, same order).
    int nbbits = 10;                                   // >= 1024 stripes, and >> nthreads
    while ((1 << nbbits) < std::max(1024, nthreads * 64)) ++nbbits;
    const std::size_t NB = static_cast<std::size_t>(1) << nbbits;
    const int shift = 64 - nbbits;
    const int dim2 = (net.nclocks + 1) * (net.nclocks + 1);
    ZoneIntern zi(dim2, nthreads);                     // ONE shared distinct-zone table
    std::vector<NetPassed> smap(NB);                   // flat per-stripe passed tables
    for (auto& mp : smap) mp.zi = &zi;
    std::vector<std::mutex> smtx(NB);                  // one lock per stripe (value-init in place)
    auto stripe_of = [&](std::uint64_t dk) -> std::size_t {
      return static_cast<std::size_t>((dk * 0x9e3779b97f4a7c15ULL) >> shift);
    };

    // The frontier holds only (dk, zone-index) per state -- 12 B, not a full
    // SymState. locs/ints come from dk (decode_key) and the zone from the intern
    // table, reconstructed into a REUSED per-thread scratch at expansion. This
    // removes the ~1.6 GB of interned-zone copies the old make_survivor did (one
    // full DBM copied into every frontier state) and shrinks the resident
    // frontier ~50× so it stays cache-resident during expansion.
    const int np_ = net.nprocs(), ni_ = static_cast<int>(net.intvars.size());
    std::vector<std::pair<std::uint64_t, int>> frontier;
    {
      const std::uint64_t dk = disc_key(s0.locs, s0.ints);
      const int z0 = zi.intern(s0.zone, 0);
      smap[stripe_of(dk)].add(std::move(s0), dk, 0, has_diag, state_bounds, 0);
      frontier.push_back({dk, z0});
    }

    int level = 0;
    std::size_t nexpanded = 0;
    // Peak-RSS composition sampler (PTCTLMC_MEMPROF): sample at each level boundary and
    // remember the composition at the moment RSS was highest.
    const bool memprof = std::getenv("PTCTLMC_MEMPROF") != nullptr;
    std::size_t pk_rss = 0, pk_lvl = 0, pk_front = 0, pk_pack = 0, pk_map = 0, pk_cache = 0,
                pk_arena = 0, pk_tab = 0, pk_heap = 0;
    std::atomic<bool> found{false};
    const bool single = (nthreads == 1);   // seq / omp -p1: skip locks + the touched union
    // Fused batched merge: O(1) critical section + one lock-free settle pass per level.
    // Zone-interner GC trigger: sweep once the allocation level reaches `sweep_at`, then set
    // the next target to double the surviving live set (see the sweep site below).
    constexpr std::size_t kSweepFloor = 50000;   // don't bother GCing a small interner
    std::size_t sweep_at = kSweepFloor;
    while (!frontier.empty()) {
      if (stats) stats->explored += frontier.size();
      if (has_diag && (nexpanded += frontier.size()) > 500000)
        throw std::runtime_error("state cap exceeded — model likely has unbounded diagonal clocks");
      ++level;

      // Expand each frontier state and insert its successors DIRECTLY into the
      // shared table, locking only the touched stripe (expansion and insertion
      // → no serial routing barrier). Inserting during expansion is what keeps the
      // convex-union merge order stable enough to match the legacy state counts.
      std::vector<std::unordered_set<std::uint64_t>> tls_touched(nthreads);
      std::vector<std::size_t> tls_gen(nthreads, 0);
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel num_threads(nthreads)
      {
        const int tid = omp_get_thread_num();
        auto& touched = tls_touched[tid];
        std::vector<SymState> buf;
        SymState cur;                    // reused per thread: reconstructed frontier state
        cur.locs.resize(np_);
        cur.ints.resize(ni_);
#pragma omp for schedule(dynamic, 8)
        for (long i = 0; i < static_cast<long>(frontier.size()); ++i) {
          if (found.load(std::memory_order_relaxed)) continue;
          decode_key(frontier[i].first, cur.locs.data(), cur.ints.data());
          // buf is NOT cleared: gen_succ_reuse rebuilds slots in place and returns
          // the count, so the loc/int/zone storage is reused across frontier states.
          const std::size_t ns = gen_succ_reuse(net, cur, zi.get(frontier[i].second), L, U, qmask, emode, buf);
          for (std::size_t sk = 0; sk < ns; ++sk) {
            SymState& s = buf[sk];
            ++tls_gen[tid];
            if (is_target(s)) { found.store(true, std::memory_order_relaxed); break; }
            const std::uint64_t dk = disc_key(s.locs, s.ints);
            const std::size_t si = stripe_of(dk);
            if (single) {
              if (smap[si].add(std::move(s), dk, level, has_diag, state_bounds, tid) >= 0) touched.insert(dk);
            } else {
              std::lock_guard<std::mutex> lk(smtx[si]);
              if (smap[si].add(std::move(s), dk, level, has_diag, state_bounds, tid) >= 0) touched.insert(dk);
            }
          }
        }
      }
#else
      {
        auto& touched = tls_touched[0];
        std::vector<SymState> buf;
        SymState cur;
        cur.locs.resize(np_);
        cur.ints.resize(ni_);
        for (auto& fr : frontier) {
          decode_key(fr.first, cur.locs.data(), cur.ints.data());
          const std::size_t ns = gen_succ_reuse(net, cur, zi.get(fr.second), L, U, qmask, emode, buf);
          for (std::size_t sk = 0; sk < ns; ++sk) {
            SymState& s = buf[sk];
            ++tls_gen[0];
            if (is_target(s)) return true;
            const std::uint64_t dk = disc_key(s.locs, s.ints);
            if (smap[stripe_of(dk)].add(std::move(s), dk, level, has_diag, state_bounds, 0) >= 0)
              touched.insert(dk);
          }
        }
      }
#endif
      if (stats) for (std::size_t g : tls_gen) stats->generated += g;
      if (found.load()) return true;

      // Collect this level's survivors (entries stamped `level`) into the next
      // frontier. Union the per-thread touched sets so each dk is scanned once;
      // the tables are read-only here (no writers), so for_level is a safe read.
      std::vector<std::uint64_t> tdk;
      if (single) {
        tdk.assign(tls_touched[0].begin(), tls_touched[0].end());
      } else {
        std::unordered_set<std::uint64_t> allt;
        for (auto& t : tls_touched) allt.insert(t.begin(), t.end());
        tdk.assign(allt.begin(), allt.end());
      }
      // Survivors go into the next frontier as (dk, zone-index) pairs -- no
      // SymState build, no zone copy here (that happens once, lazily, at the next
      // expansion into the reused `cur`).
      std::vector<std::vector<std::pair<std::uint64_t, int>>> nf(nthreads);
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(nthreads)
      for (long k = 0; k < static_cast<long>(tdk.size()); ++k) {
        const std::uint64_t dk = tdk[k];
        auto& buf = nf[omp_get_thread_num()];
        smap[stripe_of(dk)].for_level(dk, level, [&](int zidx) { buf.push_back({dk, zidx}); });
      }
#else
      for (const std::uint64_t dk : tdk)
        smap[stripe_of(dk)].for_level(dk, level, [&](int zidx) { nf[0].push_back({dk, zidx}); });
#endif
      std::vector<std::pair<std::uint64_t, int>> next;
      for (auto& v : nf)
        for (auto& p : v) next.push_back(p);
      frontier.swap(next);
      // Reclaim dead arena tombstones: one thread per stripe (exclusive), between levels.
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 8) num_threads(nthreads)
#endif
      for (long s = 0; s < static_cast<long>(smap.size()); ++s) smap[s].maybe_compact();

      // Reclaim interned zones that no live passed-set node references any more (subsumed /
      // merged out). Mark the survivors -- referenced by a live arena node or the next frontier
      // -- then sweep the rest back to the interner's free-list for in-place reuse.
      //
      // TRIGGER: growth-based, not per-level. The old guard was `zi.n > 50000`, but `n` is a
      // high-water counter that never shrinks, so it latched true at the first big level and
      // ran a full mark+sweep on EVERY level thereafter. Both halves are serial and O(all
      // arena nodes + htab slots), so the cost grew with the state space while the expand
      // phase stayed parallel -- on arbiter17 that was 2.8s of a 9.3s run (30%), and it is
      // what capped the 12-thread speedup at ~5x.
      //
      // Sweeping when the allocation level has DOUBLED since the last sweep makes the work
      // amortized O(1) per interned zone (a sweep costing O(A) is paid for by the A/2 fresh
      // interns that triggered it), so a run does O(log) sweeps instead of one per level. The
      // memory it trades away is bounded: garbage never exceeds the live set, i.e. the
      // interner stays within 2x of its pinned-to-live size.
      if (zi.allocated() >= sweep_at) {
        std::vector<std::uint8_t> alive(zi.n.load(), 0);
        for (auto& mp : smap)
          for (const auto& nd : mp.arena)
            if (!nd.dead) alive[nd.zidx] = 1;
        for (const auto& pr : frontier) alive[pr.second] = 1;
        zi.sweep(alive);
        sweep_at = std::max<std::size_t>(kSweepFloor, zi.allocated() * 2);
      }

      if (memprof) {
#ifdef __APPLE__
        mach_task_basic_info info{};
        mach_msg_type_number_t c = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &c) == KERN_SUCCESS &&
            info.resident_size > pk_rss) {
          std::size_t pack = 0, mapb = 0, cacheb = 0, an = 0, dn = 0, ab = 0, tb = 0;
          zi.account(pack, mapb, cacheb);
          for (const auto& mp : smap) mp.account(an, dn, ab, tb);
          pk_rss = info.resident_size; pk_lvl = level;
          pk_front = frontier.capacity() * sizeof(std::pair<std::uint64_t, int>);
          pk_pack = pack; pk_map = mapb; pk_cache = cacheb; pk_arena = ab; pk_tab = tb;
          pk_heap = mstats().bytes_used;   // process-wide live heap (all zones)
        }
#elif defined(__linux__)
        const std::size_t rss = linux_rss_bytes();
        if (rss > pk_rss) {
          std::size_t pack = 0, mapb = 0, cacheb = 0, an = 0, dn = 0, ab = 0, tb = 0;
          zi.account(pack, mapb, cacheb);
          for (const auto& mp : smap) mp.account(an, dn, ab, tb);
          pk_rss = rss; pk_lvl = level;
          pk_front = frontier.capacity() * sizeof(std::pair<std::uint64_t, int>);
          pk_pack = pack; pk_map = mapb; pk_cache = cacheb; pk_arena = ab; pk_tab = tb;
          pk_heap = linux_heap_bytes();
        }
#endif
      }
    }
    if (memprof) {
      auto MB = [](std::size_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
      const std::size_t known = pk_front + pk_pack + pk_map + pk_cache + pk_arena + pk_tab;
      std::fprintf(stderr,
                   "[memprof] peak-RSS=%.0fMB @level %zu | heap-in-use=%.0fMB | packed=%.0f map=%.0f "
                   "cache=%.0f arena=%.0f tab=%.0f frontier=%.0f known=%.0f other=%.0f\n",
                   MB(pk_rss), pk_lvl, MB(pk_heap), MB(pk_pack), MB(pk_map), MB(pk_cache),
                   MB(pk_arena), MB(pk_tab), MB(pk_front), MB(known), MB(pk_rss) - MB(known));
      std::fprintf(stderr, "[memprof] distinct-zones=%d htab-slots=%zu (load=%.2f)\n",
                   zi.n.load(), zi.htab.capacity(),
                   zi.htab.capacity() ? double(zi.n.load()) / zi.htab.capacity() : 0.0);
    }
    if (std::getenv("PTCTLMC_STATS")) {
      std::size_t zones = 0, disc = 0;
      for (const auto& mp : smap) mp.stats(disc, zones);
      std::unordered_set<std::uint64_t> distinct;   // distinct-zone probe (FNV-1a of zone bytes)
      for (const auto& mp : smap)
        mp.for_each_live_zone([&](const Dbm& z) {
          const auto& d = z.data();
          std::uint64_t h = 1469598103934665603ULL;
          for (raw_t r : d) { h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(r)); h *= 1099511628211ULL; }
          distinct.insert(h);
        });
      std::fprintf(stderr, "[stats] discrete-states=%zu stored-zones=%zu distinct-zones=%zu generated=%zu (shared-passed, %zu stripes)\n",
                   disc, zones, distinct.size(), stats ? stats->generated : 0, NB);
      std::size_t packedB = 0, mapB = 0, cacheB = 0;
      zi.account(packedB, mapB, cacheB);
      std::size_t arenaNodes = 0, deadNodes = 0, arenaB = 0, tabB = 0;
      for (const auto& mp : smap) mp.account(arenaNodes, deadNodes, arenaB, tabB);
      auto MB = [](std::size_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
      std::fprintf(stderr,
                   "[bytes] packed-zones=%.0fMB intern-map=%.0fMB tcaches=%.0fMB | arena=%.0fMB "
                   "(%zu nodes, %zu dead=%.0f%%) tab=%.0fMB\n",
                   MB(packedB), MB(mapB), MB(cacheB), MB(arenaB), arenaNodes, deadNodes,
                   arenaNodes ? 100.0 * deadNodes / arenaNodes : 0.0, MB(tabB));
    }
    return false;
}

bool exists_invariant_run(const Network& net, const DiscPred& phi, Backend backend,
                          ReachStats* stats) {
  (void)backend;  // liveness graph search is sequential (cycle detection); omp TODO
  std::vector<std::int32_t> L = net.lbound, U = net.ubound;
  const int emode = network_has_diagonal(net) ? ExNoDiag : ExLuPlus;  // diagonals: keep exact
  const std::vector<char> qmask(net.nclocks + 1, 0);  // liveness predicate has no clocks

  auto sat = [&](const SymState& s) -> bool {
    for (const auto& r : phi.locs) {
      const bool eq = (s.locs[r.proc] == r.loc);
      if (eq != r.equal) return false;
    }
    return int_ok(phi.ints, s.ints);
  };
  auto key = [&](const SymState& s) -> std::string {
    std::string k = disc_key(s.locs, s.ints);
    const auto& d = s.zone.data();
    k.append(reinterpret_cast<const char*>(d.data()), d.size() * sizeof(raw_t));
    return k;
  };

  // initial symbolic state (same construction as reachable_net)
  SymState s0;
  s0.locs.resize(net.nprocs());
  for (int p = 0; p < net.nprocs(); ++p) s0.locs[p] = net.procs[p].initial;
  s0.ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0.ints[v] = net.intvars[v].init;
  s0.zone = Dbm(net.nclocks);
  {
    std::vector<Constraint> inv;
    all_invariants(net, s0.locs, inv);
    if (!apply_all(s0.zone, inv)) return false;
    bool c, u;
    frozen_flags(net, s0.locs, &c, &u);
    if (!c && !u) { s0.zone.delay(); apply_all(s0.zone, inv); }
    reset_inactive(net, s0.locs, s0.zone, qmask);
    if (emode == ExLuPlus) s0.zone.extrapolate_lu_plus(L, U);
    else if (emode == ExNoDiag) s0.zone.extrapolate_lu_plus_nodiag(L, U);
  }
  if (!sat(s0)) return false;  // no run from the initial state can stay in phi

  // Build the phi-restricted extrapolated zone graph (nodes deduped by exact zone).
  std::unordered_map<std::string, int> id;
  std::vector<SymState> node;
  std::vector<std::vector<int>> adj;
  auto intern = [&](SymState&& s) -> int {
    std::string k = key(s);
    auto it = id.find(k);
    if (it != id.end()) return it->second;
    const int n = static_cast<int>(node.size());
    id.emplace(std::move(k), n);
    node.push_back(std::move(s));
    adj.emplace_back();
    return n;
  };
  // A location where every clock can grow without bound (after delay + invariant) lets
  // time diverge => an infinite run that stays put — an E[] witness on its own.
  auto time_diverges = [&](const SymState& s) -> bool {
    if (net.nclocks == 0) return false;
    Dbm z = s.zone;
    z.delay();
    std::vector<Constraint> inv;
    all_invariants(net, s.locs, inv);
    apply_all(z, inv);
    if (z.empty()) return false;
    for (int i = 1; i <= net.nclocks; ++i)
      if (z.get(i, 0) != kInf) return false;
    return true;
  };

  intern(std::move(s0));
  for (int u = 0; u < static_cast<int>(node.size()); ++u) {
    if (stats) ++stats->explored;
    if (time_diverges(node[u])) return true;
    std::vector<SymState> succ;
    successors_net(net, node[u], node[u].zone, L, U, qmask, emode, succ);
    for (auto& ns : succ) {
      if (!sat(ns)) continue;
      const int v = intern(std::move(ns));
      adj[u].push_back(v);
    }
  }

  // Reachable cycle? Iterative three-colour DFS; a back edge (to a GREY node) = cycle.
  enum { White, Grey, Black };
  std::vector<char> color(node.size(), White);
  std::vector<std::pair<int, std::size_t>> stk;
  for (int root = 0; root < static_cast<int>(node.size()); ++root) {
    if (color[root] != White) continue;
    stk.push_back({root, 0});
    color[root] = Grey;
    while (!stk.empty()) {
      const int u = stk.back().first;
      const std::size_t idx = stk.back().second;
      if (idx < adj[u].size()) {
        stk.back().second = idx + 1;
        const int v = adj[u][idx];
        if (color[v] == Grey) return true;
        if (color[v] == White) { color[v] = Grey; stk.push_back({v, 0}); }
      } else {
        color[u] = Black;
        stk.pop_back();
      }
    }
  }
  return false;
}

bool leads_to_net(const Network& net, const DiscPred& p, const DiscPred& q, Backend backend,
                  ReachStats* stats) {
  (void)backend;
  std::vector<std::int32_t> L = net.lbound, U = net.ubound;
  const int emode = network_has_diagonal(net) ? ExNoDiag : ExLuPlus;
  const std::vector<char> qmask(net.nclocks + 1, 0);

  auto holds = [&](const DiscPred& pr, const SymState& s) -> bool {
    for (const auto& r : pr.locs) { const bool eq = (s.locs[r.proc] == r.loc); if (eq != r.equal) return false; }
    return int_ok(pr.ints, s.ints);
  };
  auto key = [&](const SymState& s) -> std::string {
    std::string k = disc_key(s.locs, s.ints);
    const auto& d = s.zone.data();
    k.append(reinterpret_cast<const char*>(d.data()), d.size() * sizeof(raw_t));
    return k;
  };
  auto extrap = [&](Dbm& z) {
    if (emode == ExLuPlus) z.extrapolate_lu_plus(L, U);
    else if (emode == ExNoDiag) z.extrapolate_lu_plus_nodiag(L, U);
  };

  SymState s0;
  s0.locs.resize(net.nprocs());
  for (int pr = 0; pr < net.nprocs(); ++pr) s0.locs[pr] = net.procs[pr].initial;
  s0.ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0.ints[v] = net.intvars[v].init;
  s0.zone = Dbm(net.nclocks);
  {
    std::vector<Constraint> inv;
    all_invariants(net, s0.locs, inv);
    if (!apply_all(s0.zone, inv)) return true;  // unreachable init => vacuously holds
    bool c, u; frozen_flags(net, s0.locs, &c, &u);
    if (!c && !u) { s0.zone.delay(); apply_all(s0.zone, inv); }
    reset_inactive(net, s0.locs, s0.zone, qmask);
    extrap(s0.zone);
  }

  // Build the full reachable graph (exact-hash nodes, extrapolated for finiteness).
  std::unordered_map<std::string, int> id;
  std::vector<SymState> node;
  std::vector<std::vector<int>> adj;
  auto intern = [&](SymState&& s) -> int {
    std::string k = key(s);
    auto it = id.find(k);
    if (it != id.end()) return it->second;
    const int n = static_cast<int>(node.size());
    id.emplace(std::move(k), n); node.push_back(std::move(s)); adj.emplace_back();
    return n;
  };
  auto time_diverges = [&](const SymState& s) -> bool {
    if (net.nclocks == 0) return false;
    Dbm z = s.zone; z.delay();
    std::vector<Constraint> inv; all_invariants(net, s.locs, inv); apply_all(z, inv);
    if (z.empty()) return false;
    for (int i = 1; i <= net.nclocks; ++i) if (z.get(i, 0) != kInf) return false;
    return true;
  };
  intern(std::move(s0));
  for (int u = 0; u < static_cast<int>(node.size()); ++u) {
    if (stats) ++stats->explored;
    std::vector<SymState> succ;
    successors_net(net, node[u], node[u].zone, L, U, qmask, emode, succ);
    for (auto& ns : succ) { const int v = intern(std::move(ns)); adj[u].push_back(v); }
  }
  const int N = static_cast<int>(node.size());
  std::vector<char> pv(N), qv(N), seed(N, 0);
  for (int i = 0; i < N; ++i) { pv[i] = holds(p, node[i]); qv[i] = holds(q, node[i]); }
  // Seeds of an infinite/maximal ¬q run: a ¬q deadlock, or a ¬q state where time diverges.
  for (int i = 0; i < N; ++i)
    if (!qv[i] && (adj[i].empty() || time_diverges(node[i]))) seed[i] = 1;

  // ¬q-cycle nodes are also seeds — iterative Tarjan SCC over the ¬q-subgraph.
  std::vector<int> tidx(N, -1), low(N, 0);
  std::vector<char> onstk(N, 0);
  std::vector<int> tstk;
  int counter = 0;
  struct Frame { int u; std::size_t i; };
  for (int s = 0; s < N; ++s) {
    if (qv[s] || tidx[s] != -1) continue;
    std::vector<Frame> fs;
    fs.push_back({s, 0}); tidx[s] = low[s] = counter++; tstk.push_back(s); onstk[s] = 1;
    while (!fs.empty()) {
      const int u = fs.back().u;  // copy: fs may reallocate on push_back below
      bool descended = false;
      while (fs.back().i < adj[u].size()) {
        const int v = adj[u][fs.back().i++];
        if (qv[v]) continue;
        if (tidx[v] == -1) {
          tidx[v] = low[v] = counter++; tstk.push_back(v); onstk[v] = 1; fs.push_back({v, 0});
          descended = true; break;
        } else if (onstk[v]) { if (tidx[v] < low[u]) low[u] = tidx[v]; }
      }
      if (descended) continue;
      if (low[u] == tidx[u]) {
        std::vector<int> comp; int w;
        do { w = tstk.back(); tstk.pop_back(); onstk[w] = 0; comp.push_back(w); } while (w != u);
        bool cyc = comp.size() > 1;
        if (!cyc) for (int v : adj[u]) if (v == u) cyc = true;  // self-loop
        if (cyc) for (int c : comp) seed[c] = 1;
      }
      fs.pop_back();
      if (!fs.empty() && low[u] < low[fs.back().u]) low[fs.back().u] = low[u];
    }
  }

  // escape = ¬q nodes that can reach a seed via ¬q-edges (reverse reachability).
  std::vector<std::vector<int>> radj(N);
  for (int u = 0; u < N; ++u)
    if (!qv[u]) for (int v : adj[u]) if (!qv[v]) radj[v].push_back(u);
  std::vector<char> escape(N, 0);
  std::vector<int> stk;
  for (int i = 0; i < N; ++i) if (seed[i]) { escape[i] = 1; stk.push_back(i); }
  while (!stk.empty()) {
    const int u = stk.back(); stk.pop_back();
    for (int w : radj[u]) if (!escape[w]) { escape[w] = 1; stk.push_back(w); }
  }
  // p --> q fails iff a reachable p ∧ ¬q state can stay in ¬q forever (escape).
  for (int i = 0; i < N; ++i) if (pv[i] && !qv[i] && escape[i]) return false;
  return true;
}

// Does the (delay-closed, extrapolated) symbolic state `s` contain a concrete valuation
// with no outgoing action transition -- a deadlock?  A whole zone that has no successor is
// the easy case; the hard case is a PARTIAL deadlock, where the zone has successors but a
// sub-region of it can never reach any of them (e.g. an invariant caps time before the only
// guard opens). We compute it exactly: mirror successors_net's three transition kinds, but
// for each transition that CAN fire collect its SOURCE zone within `s.zone` -- the pre-reset
// valuations from which it fires and lands in a legal target (guards ∧ reset^{-1}(target
// invariant)). The union of the time-pasts (down()) of those source zones is the set of
// non-deadlocked valuations (for a frozen source -- committed/urgent -- no time may pass, so
// there is no down()). Anything in `s.zone` left over after subtracting that union is a set
// of deadlock states. Enabledness is invariant under LU-extrapolation, so working on the
// extrapolated zone is sound -- the same assumption the rest of the engine already relies on.
static bool state_has_deadlock(const Network& net, const SymState& s) {
  if (s.zone.empty()) return false;
  bool src_committed, src_urgent;
  frozen_flags(net, s.locs, &src_committed, &src_urgent);
  const bool frozen = src_committed || src_urgent;  // no delay allowed from this state

  // Rebuild the true delay-closure of this state's zone. The stored zone has the
  // active-clock reduction applied, which pins DEAD clocks to 0 with an upper bound and so
  // is no longer delay-closed -- down() could not let time pass those clocks, inventing
  // spurious deadlocks. Re-opening every upper bound (delay) and re-imposing only the real
  // location invariant restores the closure; dead clocks become free, which is sound since
  // no guard or invariant here reads them. Frozen locations do not delay, so leave them be.
  Dbm Z = s.zone;
  if (!frozen) {
    Z.delay();
    std::vector<Constraint> cinv;
    all_invariants(net, s.locs, cinv);
    if (!apply_all(Z, cinv)) return false;  // real invariant unsatisfiable: no concrete state
  }

  std::vector<Constraint> tinv;  // target-invariant scratch
  Federation live;

  // `guarded` = Z ∩ (participant clock guards), pre-reset. Pull it back through the target
  // invariant under the transition's resets and, if a non-empty source region survives, add
  // it to `live`. reset^{-1}(inv): a constraint on a reset clock becomes one on clock 0 (its
  // post-reset value), so it either constrains the surviving source clocks or, if it reduces
  // to `0 <| bound`, gates the whole transition on feasibility of that constant.
  auto add_src = [&](Dbm guarded, const LocVec& tgt_locs,
                     const std::vector<const std::vector<int>*>& reset_lists) {
    if (guarded.empty()) return;
    auto is_reset = [&](int c) {
      for (auto* rl : reset_lists) for (int r : *rl) if (r == c) return true;
      return false;
    };
    all_invariants(net, tgt_locs, tinv);
    for (const auto& c : tinv) {
      const int i = is_reset(c.i) ? 0 : c.i;
      const int j = is_reset(c.j) ? 0 : c.j;
      if (i == j) { if (c.bound < kZeroLe) return; continue; }  // 0 <| bound: constant gate
      if (!guarded.constrain(i, j, c.bound)) return;
    }
    if (!guarded.empty()) live.push_back(std::move(guarded));
  };

  // ---- local (unsynchronised) edges ----
  for (int p = 0; p < net.nprocs(); ++p) {
    if (src_committed && !net.procs[p].locations[s.locs[p]].committed) continue;
    for (int ei : net.procs[p].edges_of(s.locs[p])) {
      const NEdge& e = net.procs[p].edges[ei];
      if (e.sync_chan >= 0) continue;
      if (!int_ok(e.int_guard, s.ints) || !gen_ok(net, e, s.ints.data(), s.ints.size())) continue;
      IntVec nints = s.ints;
      for (const auto& a : e.int_updates) nints[a.var] = a.val;
      if (!gen_apply(net, e, nints.data(), nints.size())) continue;
      Dbm g = Z;
      if (!apply_all(g, e.clock_guard)) continue;
      LocVec tl = s.locs; tl[p] = e.tgt;
      add_src(std::move(g), tl, {&e.clock_resets});
    }
  }

  // ---- binary channel synchronisation (a! || a?) ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1) continue;
      if (net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      const bool i_comm = net.procs[i].locations[s.locs[i]].committed;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        for (int ej : net.procs[j].edges_of(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan != se.sync_chan || re.sync_dir != -1) continue;
          if (src_committed && !(i_comm || net.procs[j].locations[s.locs[j]].committed)) continue;
          if (!int_ok(re.int_guard, s.ints) || !gen_ok(net, re, s.ints.data(), s.ints.size())) continue;
          IntVec nints = s.ints;
          for (const auto& a : se.int_updates) nints[a.var] = a.val;
          for (const auto& a : re.int_updates) nints[a.var] = a.val;
          if (!gen_apply(net, se, nints.data(), nints.size()) ||
              !gen_apply(net, re, nints.data(), nints.size())) continue;
          Dbm g = Z;
          if (!apply_all(g, se.clock_guard) || !apply_all(g, re.clock_guard)) continue;
          LocVec tl = s.locs; tl[i] = se.tgt; tl[j] = re.tgt;
          add_src(std::move(g), tl, {&se.clock_resets, &re.clock_resets});
        }
      }
    }
  }

  // ---- broadcast synchronisation: a! fires with all enabled a? (0 or more) ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1 || !net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      std::vector<std::pair<int, std::vector<int>>> parts;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        std::vector<int> es;
        for (int ej : net.procs[j].edges_of(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan == se.sync_chan && re.sync_dir == -1 &&
              int_ok(re.int_guard, s.ints) && gen_ok(net, re, s.ints.data(), s.ints.size()))
            es.push_back(ej);
        }
        if (!es.empty()) parts.push_back({j, std::move(es)});
      }
      std::vector<std::size_t> pick(parts.size(), 0);
      for (;;) {
        bool part_committed = net.procs[i].locations[s.locs[i]].committed;
        for (const auto& pr : parts)
          if (net.procs[pr.first].locations[s.locs[pr.first]].committed) part_committed = true;
        if (!src_committed || part_committed) {
          Dbm g = Z;
          bool ok = apply_all(g, se.clock_guard);
          std::vector<const std::vector<int>*> rls{&se.clock_resets};
          LocVec tl = s.locs; tl[i] = se.tgt;
          for (std::size_t t = 0; ok && t < parts.size(); ++t) {
            const NEdge& re = net.procs[parts[t].first].edges[parts[t].second[pick[t]]];
            ok = apply_all(g, re.clock_guard);
            rls.push_back(&re.clock_resets);
            tl[parts[t].first] = re.tgt;
          }
          if (ok) add_src(std::move(g), tl, rls);
        }
        if (parts.empty()) break;
        std::size_t k = 0;
        for (; k < parts.size(); ++k) { if (++pick[k] < parts[k].second.size()) break; pick[k] = 0; }
        if (k == parts.size()) break;
      }
    }
  }

  // Non-deadlocked = valuations that can (delay into, unless frozen) some enabled source
  // region. Subtract that from the zone; a non-empty remainder is a set of deadlock states.
  if (!frozen)
    for (auto& W : live) W.down();
  return !fed_subtract(Z, live).empty();
}

bool deadlock_reachable(const Network& net, Backend backend, ReachStats* stats) {
  (void)backend;  // zone-graph search with subsumption is sequential, like leads_to_net
  std::vector<std::int32_t> L = net.lbound, U = net.ubound;
  const int emode = network_has_diagonal(net) ? ExNoDiag : ExLuPlus;  // diagonals: keep exact
  // Active-clock reduction stays ON here (small state space); state_has_deadlock rebuilds the
  // delay-closure it perturbs, so the exploration keeps the same zones reachable_net would.
  const std::vector<char> qmask(net.nclocks + 1, 0);
  auto extrap = [&](Dbm& z) {
    if (emode == ExLuPlus) z.extrapolate_lu_plus(L, U);
    else if (emode == ExNoDiag) z.extrapolate_lu_plus_nodiag(L, U);
  };

  // Initial symbolic state -- same construction as reachable_net / leads_to_net.
  SymState s0;
  s0.locs.resize(net.nprocs());
  for (int p = 0; p < net.nprocs(); ++p) s0.locs[p] = net.procs[p].initial;
  s0.ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0.ints[v] = net.intvars[v].init;
  s0.zone = Dbm(net.nclocks);
  {
    std::vector<Constraint> inv;
    all_invariants(net, s0.locs, inv);
    if (!apply_all(s0.zone, inv)) return false;  // empty init => no reachable state, no deadlock
    bool c, u;
    frozen_flags(net, s0.locs, &c, &u);
    if (!c && !u) { s0.zone.delay(); apply_all(s0.zone, inv); }
    reset_inactive(net, s0.locs, s0.zone, qmask);
    extrap(s0.zone);
  }

  // Passed list keyed by discrete state, storing an antichain of extrapolated zones under
  // inclusion subsumption. Subsumption is SOUND for deadlock because deadlock is a
  // per-valuation property: the deadlock valuations of a subsumed zone lie inside the
  // subsuming zone and are found when that one is checked. (aLU subsumption would NOT be
  // sound here -- it can drop concrete valuations -- so we use exact inclusion.)
  std::unordered_map<std::string, std::vector<Dbm>> passed;
  std::vector<SymState> work;
  auto try_insert = [&](SymState&& s) {
    auto& bucket = passed[disc_key(s.locs, s.ints)];
    for (const auto& z : bucket) if (z.includes(s.zone)) return;  // subsumed => skip
    bucket.erase(std::remove_if(bucket.begin(), bucket.end(),
                 [&](const Dbm& z) { return s.zone.includes(z); }), bucket.end());
    bucket.push_back(s.zone);
    work.push_back(std::move(s));
  };

  try_insert(std::move(s0));
  while (!work.empty()) {
    SymState s = std::move(work.back());
    work.pop_back();
    if (stats) ++stats->explored;
    if (state_has_deadlock(net, s)) return true;
    std::vector<SymState> succ;
    successors_net(net, s, s.zone, L, U, qmask, emode, succ);
    for (auto& ns : succ) { if (stats) ++stats->generated; try_insert(std::move(ns)); }
  }
  return false;
}

}  // namespace ptctlmc
