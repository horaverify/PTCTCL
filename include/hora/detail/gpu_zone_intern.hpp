/// @file gpu_zone_intern.hpp
/// @brief Shared dense-zone interner for the GPU driver a s passed set.
#pragma once

#include "hora/net_gpu.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace hora {
namespace detail {

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

  // NARROW STORAGE. A DBM entry is (value << 1) | weak, and in a zone graph over a model whose
  // largest clock constant is M every finite entry lies in [-2M-1, 2M+1] (an entry is a shortest
  // path, and any path to or from the reference clock is bounded by two direct border edges). So
  // whenever M is small -- every real timed-automata model -- the whole passed set fits in 16-bit
  // entries, halving what is by far the biggest resident structure at scale: fischer16 stores
  // 4.95M zones of 289 entries, which is 5.7 GB dense in 32-bit and 2.9 GB in 16.
  //
  // The pools and kernels stay 32-bit; only this arena narrows, and the gather/rebuild loops that
  // touch it were already a per-entry copy, so widening/narrowing there is free (and they move
  // half the bytes). `fits16` decides once, from the model; a model with huge constants simply
  // keeps 32-bit storage.
  static constexpr std::int32_t kInf16 = 0x7ffe;   // sentinel for kInf in a narrowed entry
  GpuZoneIntern(int d2, bool narrow) : d2_(d2), esz_(narrow ? 2 : 4), stripes_(kStripes) {}
  ~GpuZoneIntern() { for (auto& c : chunks_) delete c.load(std::memory_order_relaxed); }

  // Widen zone `idx` into `out` (d2 entries).
  void load(int idx, raw_t* out) const {
    const std::uint8_t* p = base(idx);
    if (esz_ == 4) { std::memcpy(out, p, static_cast<std::size_t>(d2_) * sizeof(raw_t)); return; }
    const std::int16_t* q = reinterpret_cast<const std::int16_t*>(p);
    for (int k = 0; k < d2_; ++k) out[k] = (q[k] == kInf16) ? kInf : static_cast<raw_t>(q[k]);
  }
  // As load(), but into the NARROW device frontier buffer: when the arena is already narrow
  // this is a plain memcpy, which is the common case.
  void load_gz(int idx, gz_t* out) const {
    const std::uint8_t* p = base(idx);
    if (esz_ == 2) { std::memcpy(out, p, static_cast<std::size_t>(d2_) * sizeof(gz_t)); return; }
    const raw_t* q = reinterpret_cast<const raw_t*>(p);
    for (int k = 0; k < d2_; ++k) out[k] = gz_store(q[k]);
  }

  // load_gz(), strided: entry k goes to out[k*stride] (the SoA merge pool).
  void load_gz_strided(int idx, gz_t* out, std::size_t stride) const {
    const std::uint8_t* p = base(idx);
    if (esz_ == 2) {
      const gz_t* q = reinterpret_cast<const gz_t*>(p);
      for (int k = 0; k < d2_; ++k) out[static_cast<std::size_t>(k) * stride] = q[k];
      return;
    }
    const raw_t* q = reinterpret_cast<const raw_t*>(p);
    for (int k = 0; k < d2_; ++k) out[static_cast<std::size_t>(k) * stride] = gz_store(q[k]);
  }

  // As load(), but strided: entry k goes to out[k*stride] (the SoA merge pool).
  void load_strided(int idx, raw_t* out, std::size_t stride) const {
    const std::uint8_t* p = base(idx);
    if (esz_ == 4) {
      const raw_t* q = reinterpret_cast<const raw_t*>(p);
      for (int k = 0; k < d2_; ++k) out[static_cast<std::size_t>(k) * stride] = q[k];
      return;
    }
    const std::int16_t* q = reinterpret_cast<const std::int16_t*>(p);
    for (int k = 0; k < d2_; ++k)
      out[static_cast<std::size_t>(k) * stride] = (q[k] == kInf16) ? kInf : static_cast<raw_t>(q[k]);
  }

  // mode: Dedup interns (one copy per distinct zone); Append stores every zone without hashing,
  // which isolates the cost of the hash+lookup from the cost of the indexed layout itself.
  enum class Mode { Dedup, Append };
  void set_mode(Mode m) { mode_ = m; }

  int intern(const raw_t* z) {
    if (mode_ == Mode::Append) {
      const int idx = alloc();
      store(idx, z);
      return idx;
    }
    const std::uint64_t h = hash(reinterpret_cast<const std::uint32_t*>(z));
    Stripe& s = stripes_[h & (kStripes - 1)];
    std::lock_guard<std::mutex> lk(s.m);
    auto& bucket = s.tab[h];
    for (const int cand : bucket) if (equals(cand, z)) return cand;
    const int idx = alloc();
    store(idx, z);
    bucket.push_back(idx);
    return idx;
  }

  // Return a slot to the pool. In Append mode the arena would otherwise grow with the CUMULATIVE
  // number of interned zones rather than the LIVE one -- and the rebuild re-interns a bucket's
  // WHOLE antichain every time the bucket is touched, so a bucket touched k times over the search
  // costs k*|antichain| slots. Measured on fischer12: ~2000 chunks x 5.4 MB = 10.9 GB of arena
  // for 2.7M live zones (1.8 GB of actual data). With the free list the arena is bounded by the
  // peak live set. Striped by index so the 12 rebuild threads do not share one lock.
  // Slots are recycled ONE LEVEL LATE. The frontier holds arena indices rather than copies of
  // the zones, so a slot it names must survive until that frontier has been expanded -- and the
  // merges that drop slots run during exactly the level that is consuming it. Releases therefore
  // go to a pending list and only reach the free list at the level boundary, by which point the
  // frontier that could have named them is gone. (Deferring costs a level's worth of slots, not
  // a level's worth of zones: the arena is still bounded by the live set plus one level.)
  void release(int idx) {
    if (mode_ != Mode::Append) return;   // Dedup shares slots between buckets: not ours to free
    Stripe& s = stripes_[static_cast<unsigned>(idx) & (kStripes - 1)];
    std::lock_guard<std::mutex> lk(s.m);
    s.pendingIdx.push_back(idx);
  }
  // Call at a level boundary, after the previous frontier has been fully expanded.
  void flush_released() {
    for (auto& s : stripes_) {
      std::lock_guard<std::mutex> lk(s.m);
      s.freeIdx.insert(s.freeIdx.end(), s.pendingIdx.begin(), s.pendingIdx.end());
      s.pendingIdx.clear();
    }
  }

  int count() const { return n_.load(std::memory_order_acquire); }
  int entry_bytes() const { return esz_; }
  // Allocated chunks, not the (pre-sized, mostly null) chunk-pointer array.
  std::size_t bytes() const {
    const std::size_t nch = (static_cast<std::size_t>(n_.load(std::memory_order_acquire)) + kChunk - 1) / kChunk;
    return nch * kChunk * d2_ * esz_;
  }

 private:
  std::uint8_t* base(int idx) const {
    return chunks_[idx >> kChunkBits].load(std::memory_order_acquire)->data() +
           static_cast<std::size_t>(idx & kChunkMask) * d2_ * esz_;
  }
  void store(int idx, const raw_t* z) {
    std::uint8_t* p = base(idx);
    if (esz_ == 4) { std::memcpy(p, z, static_cast<std::size_t>(d2_) * sizeof(raw_t)); return; }
    std::int16_t* q = reinterpret_cast<std::int16_t*>(p);
    for (int k = 0; k < d2_; ++k) q[k] = (z[k] >= kInf) ? static_cast<std::int16_t>(kInf16)
                                                       : static_cast<std::int16_t>(z[k]);
  }
  bool equals(int idx, const raw_t* z) const {
    const std::uint8_t* p = base(idx);
    if (esz_ == 4) return std::memcmp(p, z, static_cast<std::size_t>(d2_) * sizeof(raw_t)) == 0;
    const std::int16_t* q = reinterpret_cast<const std::int16_t*>(p);
    for (int k = 0; k < d2_; ++k) {
      const raw_t v = (q[k] == kInf16) ? kInf : static_cast<raw_t>(q[k]);
      if (v != z[k]) return false;
    }
    return true;
  }

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
    if (mode_ == Mode::Append) {   // reuse a released slot before growing the arena
      Stripe& s = stripes_[next_free_.fetch_add(1, std::memory_order_relaxed) & (kStripes - 1)];
      std::lock_guard<std::mutex> lk(s.m);
      if (!s.freeIdx.empty()) { const int idx = s.freeIdx.back(); s.freeIdx.pop_back(); return idx; }
    }
    const int idx = n_.fetch_add(1, std::memory_order_relaxed);
    const int c = idx >> kChunkBits;
    if (c >= kMaxChunks) throw std::runtime_error("GPU zone interner overflow");
    if (!chunks_[c].load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lk(growMtx_);
      if (!chunks_[c].load(std::memory_order_relaxed))
        chunks_[c].store(new std::vector<std::uint8_t>(static_cast<std::size_t>(kChunk) * d2_ * esz_),
                         std::memory_order_release);
    }
    return idx;
  }

  struct Stripe {
    std::unordered_map<std::uint64_t, std::vector<int>> tab;
    std::vector<int> freeIdx;      // Append mode: slots ready for reuse
    std::vector<int> pendingIdx;   // ...and slots released during the level now being expanded
    std::mutex m;
  };
  int d2_;
  int esz_;                       // bytes per DBM entry in the arena: 2 (narrowed) or 4
  Mode mode_ = Mode::Dedup;
  std::vector<std::atomic<std::vector<std::uint8_t>*>> chunks_{kMaxChunks};   // fixed: never resized
  std::deque<Stripe> stripes_;
  std::mutex growMtx_;
  std::atomic<int> n_{0};
  std::atomic<unsigned> next_free_{0};   // round-robins release()/alloc() over the stripes
};

}  // namespace detail
}  // namespace hora
