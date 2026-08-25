/// @file zone_intern.hpp
/// @brief Packed min-graph zone interner shared by every stripe.
#pragma once

#include "ptctlmc/dbm.hpp"
#include "ptctlmc/mindbm.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace ptctlmc {
namespace detail {

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
  // SHARDED. This table used to be one arena behind one mutex, and intern() sits on the
  // per-successor hot path, so every inserted state serialized there. The per-thread cache
  // below cannot absorb that: a symbolic run interns each distinct zone exactly ONCE (on
  // arbiter16, 589,824 states -> 589,824 distinct zones and 1 cache hit in 589,825 calls),
  // so ~100% of calls reached the lock. Hold time was tiny (~0.2 CPU-s) but the acquisition
  // RATE was not -- threads waited ~20x longer than the lock was ever held, and the wait grew
  // with thread count (0.5 / 2.5 / 4.5 CPU-s at 4 / 8 / 12), capping speedup at ~5.5x.
  //
  // The table is now split into independent shards selected by the zone's hash, each with its
  // own mutex, arena, and htab -- the same fine-grained-locking shape the passed list already
  // uses for its stripes. A given zone always hashes to the same shard, so distinct-zone
  // identity is unchanged and state counts stay identical.
  static constexpr int kShardBits = 6;
  static constexpr int kShards = 1 << kShardBits;
  static constexpr std::uint64_t kShardMask = kShards - 1;
  // zidx = (local << kShardBits) | shard. INTERLEAVED on purpose: with the shard in the high
  // bits the maximum index would be kShards x the zone count, and the GC's `alive` array is
  // sized by that bound. Interleaved, the bound stays ~= the zone count.
  //
  // Chunk-pointer arrays are preallocated and never resized, so lock-free readers in record()
  // cannot race a reallocation. Both constants are therefore sized so the fixed cost summed
  // over ALL shards stays small (~2 MB), not per table as before.
  //
  // CAPACITY. These used to be 1<<10 / 1<<11 -> 2.1M zones/shard, ~134M total, and that cap was
  // REACHABLE: twolock15 and twolock16 hit it and died. The throw below is raised inside an
  // `omp parallel` region, where an exception may not leave the structured block, so it did not
  // surface as main()'s "error:" line -- every worker called std::terminate and the process took
  // SIGABRT. The benchmark harness saw only a nonzero exit and recorded ERR at a suspiciously
  // repeatable 16.5 GB (twolock15) / 21.0 GB (twolock16), identical at 24 and 32 threads because
  // the trigger is a zone COUNT, not a time. At 16 threads the same runs merely timed out first,
  // which is why the crash looked thread-count-dependent and was not.
  //
  // Sized now so RAM is always the binding constraint and the harness's memory guard reports an
  // honest MEMOUT instead of a crash: 16.8M zones/shard x 64 = ~1.07G zones, which at the ~127 B
  // per zone these models cost would need ~136 GB. The hard ceiling is the ENCODING, not this
  // choice -- zidx is a signed 32-bit int (see enc() and NetPassed::Node::zidx), so
  // kChunkBits + log2(kChunks) + kShardBits must stay < 31. 12 + 12 + 6 = 30 leaves one bit of
  // margin; widening past 2.1G zones would mean a 64-bit zidx in every passed-set node, which
  // costs more resident memory than it buys.
  static constexpr int kChunkBits = 12;
  static constexpr int kChunk = 1 << kChunkBits;        // 4096 zones/chunk (32 KB of pointers)
  static constexpr int kChunkMask = kChunk - 1;
  static constexpr int kChunks = 1 << 12;               // -> 16.8M zones/shard, ~1.07G total
  // Slabs start small and double, so 64 shards do not each commit a big slab up front. The CAP
  // matters as much as the floor: whatever a shard's final slab is, the unused tail of it is
  // wasted, and that waste is paid 64 times. Doubling all the way to 4 MB cost ~40 MB on
  // arbiter17 (+17% RSS); capping at 128 KB bounds total slack to kShards * 128 KB (~8 MB
  // worst case) while still keeping the slab count low for large runs.
  static constexpr std::size_t kSlabMin = 1u << 14;     // 16 KB
  static constexpr std::size_t kSlabMax = 1u << 17;     // 128 KB
  static constexpr int kHtabInitBits = 10;              // 1024 slots/shard, grows by rehash
  static constexpr int kCacheBits = 15;                 // 32768 lines/thread (~0.5 MB)
  static constexpr std::uint32_t kCacheMask = (1u << kCacheBits) - 1;

  struct CacheLine { std::uint64_t h; std::int32_t idx; };

  // One independently-locked partition. Cache-line aligned so neighbouring shards' mutexes and
  // bump cursors do not share a line (false sharing here would undo the point of sharding).
  struct alignas(128) Shard {
    std::deque<std::vector<std::uint8_t>> slabs;        // bump arena; slab data addresses are stable
    std::uint8_t* bumpPtr = nullptr;                    // current slab cursor (mtx-guarded)
    std::size_t bumpLeft = 0;
    std::size_t slabSz = kSlabMin;                      // next slab's size (doubles to kSlabMax)
    std::vector<const std::uint8_t**> chunks;           // chunks[c][.] -> packed-record bytes
    std::vector<std::int32_t> htab;                     // open-addressing: 0=empty, else local+1
    std::uint64_t hmask;
    std::size_t hcount = 0;
    // Reclamation: dead records are recycled in place (no rebuild, no zidx remap). A freed
    // record slot returns to a free-list keyed by its exact byte length (packed records take
    // only a few distinct lengths per model, so reuse is near-total); a freed local index
    // returns to `freeIdx`. intern() reuses both before growing the arena, so slab memory is
    // bounded by the PEAK LIVE working set -- RSS never depends on the OS reclaiming.
    std::unordered_map<std::uint32_t, std::vector<std::uint8_t*>> freeRecs;
    std::vector<int> freeIdx;
    std::vector<std::int32_t> htabScratch;              // reused sweep buffer (avoids per-level realloc)
    std::mutex mtx;
    std::atomic<int> n{0};                              // high-water LOCAL index for this shard
    Shard()
        : chunks(kChunks, nullptr), htab(std::size_t(1) << kHtabInitBits, 0),
          hmask((std::size_t(1) << kHtabInitBits) - 1) {}
    ~Shard() { for (auto* c : chunks) delete[] c; }
  };

  std::vector<Shard> shards;
  std::vector<std::vector<CacheLine>> tcache;           // [thread][line] direct-mapped
  int dim;                                              // clock dim (nclocks+1); all zones share it
  // Bumped by every sweep(). get()'s decode memo is only valid within one epoch: a sweep
  // recycles indices in place, so afterwards the same idx can denote a DIFFERENT zone.
  std::uint64_t epoch = 0;

  ZoneIntern(int dim2_, int nthreads)
      : shards(kShards), tcache(nthreads > 0 ? nthreads : 1) {
    dim = 1; while (dim * dim < dim2_) ++dim;           // dim2_ = dim*dim
    for (auto& tc : tcache) tc.assign(1u << kCacheBits, CacheLine{0, -1});
  }

  static int enc(int local, int shard) { return (local << kShardBits) | shard; }

  static std::uint64_t hash_bytes(const std::uint8_t* p, std::size_t len) {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < len; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
  }
  // Shard selector. MUST NOT reuse the low bits that index a shard's htab (`h & hmask`):
  // every key routed to shard s has `h & kShardMask == s`, so if the slot index also came from
  // the low bits, only hmask/kShards of a shard's slots could ever be a home position -- the
  // other 63/64 are reachable only by linear probing. At the 70% growth trigger that packs
  // ~0.7*cap entries into cap/64 home slots (~45 deep each), and a probe is not cheap here: it
  // chases chunks[][] and byte-compares the record. Measured on arbiter16 that cost 31.2
  // probes per locked lookup versus 1.68 for the unsharded table -- which is what ate the
  // sharding win. Mixing and taking the HIGH bits makes the two indices independent.
  static int shard_of(std::uint64_t h) {
    std::uint64_t x = h;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
    return static_cast<int>(x >> (64 - kShardBits));
  }
  const std::uint8_t* record(int idx) const {
    const Shard& sh = shards[idx & kShardMask];
    const int loc = idx >> kShardBits;
    return sh.chunks[loc >> kChunkBits][loc & kChunkMask];
  }

  /// @brief Decode zone @p idx into a shared per-thread scratch.
  ///
  /// Memoised on the last index decoded by this thread. Callers hit the same index repeatedly --
  /// the expand path alone reads `zi.get(frontier[i].second)` from several call sites per state --
  /// and each miss costs a full mindbm_unpack, which is an O(dim^3) closure. On viking12 the
  /// decode path (this plus mindbm_unpack) is 40% of all instructions, because that family stores
  /// ~1 zone per discrete state and so re-decodes zones it has just touched.
  ///
  /// SOUNDNESS: the memo is keyed on (interner, epoch, index), not index alone. sweep() recycles
  /// a dead zidx in place, so after a sweep the same index can denote a DIFFERENT zone -- exactly
  /// the aliasing that made the per-thread intern cache unsound (see sweep()). Bumping the epoch
  /// there invalidates every memo at once.
  // Lines in the per-thread decode cache. 16 dense zones per thread is ~28 KB at dim 21, so the
  // whole cache stays L1/L2-resident where the packed records it replaces do not.
  //
  // A ONE-entry memo was tried first and recovered only ~4% of the ~40% that decoding costs on
  // viking12: it catches consecutive reads of the SAME index, but within a level the reads are
  // spread over many distinct indices, so almost every one missed.
  static constexpr int kDecodeLinesMax = 64;
  /// @brief Lines in the per-thread decode cache. Power of two.
  ///
  /// Not a free parameter in one direction: 1 line and 16 lines favour different families. 16
  /// wins on viking (viking17 -7.9%, viking20 -6.4%) where a level touches many distinct zones;
  /// 1 was better on single, whose long chains re-read one zone repeatedly and whose working set
  /// then fits without evicting it. 16 is the shipping value.
  static constexpr int decode_lines() { return 16; }

  /// @brief Decode zone @p idx into a per-thread cache and return it.
  ///
  /// Decoding is the price of minimal-graph storage and it is charged per zone TOUCHED, not per
  /// zone stored: reconstructing the redundant entries is an $O(dim^3)$ closure. On viking12 the
  /// decode path was 40% of all instructions, because that family keeps ~1 zone per discrete
  /// state and so re-decodes records it touched moments earlier.
  ///
  /// INDEXING: on the LOCAL-index bits, not the low bits of @p idx. A zidx is
  /// @c (local << kShardBits) | shard, so its low bits are the shard number -- indexing on those
  /// would map every zone of a shard onto the same handful of lines and defeat the cache.
  ///
  /// SOUNDNESS: keyed on (interner, epoch, index). sweep() recycles a dead zidx IN PLACE, so
  /// after a sweep the same index can denote a different zone; keying on the index alone is the
  /// aliasing that made the intern cache unsound (see sweep()). The epoch is bumped there, which
  /// invalidates every line at once.
  const Dbm& get(int idx) const {
    struct DecodeCache {
      std::vector<Dbm> z;
      std::vector<int> tag;
      const ZoneIntern* owner = nullptr;
      std::uint64_t epoch = 0;
      int dim = 0;
    };
    static thread_local DecodeCache dc;
    if (dc.dim != dim) {                       // first use, or a differently-dimensioned run
      dc.z.assign(decode_lines(), Dbm(dim - 1));
      dc.tag.assign(decode_lines(), -1);
      dc.dim = dim;
      dc.owner = nullptr;
    }
    if (dc.owner != this || dc.epoch != epoch) {   // cheap: only tags, no reallocation
      std::fill(dc.tag.begin(), dc.tag.end(), -1);
      dc.owner = this;
      dc.epoch = epoch;
    }
    const int line = (idx >> kShardBits) & (decode_lines() - 1);
    if (dc.tag[line] == idx) return dc.z[line];
    mindbm_unpack(record(idx), dc.z[line].data().data());
    dc.tag[line] = idx;
    return dc.z[line];
  }

  // Decode into caller-owned storage instead of the shared per-thread scratch, so a caller
  // can hold several zones live at once. get() overwrites one scratch, which forces any
  // multi-zone fold to re-unpack on every visit; with this the fold unpacks each zone once.
  void get_into(int idx, Dbm& out) const {
    if (out.dim() != dim) out = Dbm(dim - 1);
    mindbm_unpack(record(idx), out.data().data());
  }
  void get_into_raw(int idx, raw_t* out) const { mindbm_unpack(record(idx), out); }
  int zdim() const { return dim; }

  // Probe one shard's table for the packed bytes [q,q+qlen). Returns the shard-LOCAL index or
  // -1; on miss, *slot is the empty table position to insert at. Caller holds sh.mtx.
  static int find_locked(const Shard& sh, const std::uint8_t* q, std::size_t qlen,
                         std::uint64_t h, std::size_t* slot) {
    std::size_t i = h & sh.hmask;
    while (sh.htab[i]) {
      const int loc = sh.htab[i] - 1;
      const std::uint8_t* rec = sh.chunks[loc >> kChunkBits][loc & kChunkMask];
      if (mindbm_size(rec) == qlen && std::memcmp(rec, q, qlen) == 0) return loc;
      i = (i + 1) & sh.hmask;
    }
    *slot = i;
    return -1;
  }
  // Grow one shard's table. Rebuilt from the LIVE htab entries rather than by walking
  // 0..n-1: after a sweep, indices sitting in `freeIdx` are not live, and their chunk slots
  // still point at records that have been returned to `freeRecs` (and may already have been
  // handed to a different zone). Re-inserting those would put stale entries in the table and
  // let a later probe dedup a zone against another zone's bytes.
  static void rehash_locked(Shard& sh) {
    const std::size_t ncap = (sh.hmask + 1) * 2;
    std::vector<std::int32_t> nt(ncap, 0);
    const std::uint64_t nmask = ncap - 1;
    for (std::size_t s = 0; s < sh.htab.size(); ++s) {
      const std::int32_t v = sh.htab[s];
      if (!v) continue;
      const int loc = v - 1;
      const std::uint8_t* rec = sh.chunks[loc >> kChunkBits][loc & kChunkMask];
      std::size_t i = hash_bytes(rec, mindbm_size(rec)) & nmask;
      while (nt[i]) i = (i + 1) & nmask;
      nt[i] = loc + 1;
    }
    sh.htab.swap(nt);
    sh.hmask = nmask;
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

    // The hash picks the shard, so a given zone always lands in the same one -- dedup stays
    // exact and only this shard's lock is taken.
    const int sid = shard_of(h);
    Shard& sh = shards[sid];
    std::lock_guard<std::mutex> lk(sh.mtx);
    std::size_t slot = 0;
    int loc = find_locked(sh, qbuf.data(), qlen, h, &slot);
    if (loc < 0) {
      // A record slot: reuse a freed one of the exact length, else bump-allocate (qlen is a
      // multiple of 4 -> stays 4-aligned).
      std::uint8_t* rec;
      auto fit = sh.freeRecs.find(static_cast<std::uint32_t>(qlen));
      if (fit != sh.freeRecs.end() && !fit->second.empty()) {
        rec = fit->second.back();
        fit->second.pop_back();
      } else {
        if (sh.bumpLeft < qlen) {
          const std::size_t sz = qlen > sh.slabSz ? qlen : sh.slabSz;
          sh.slabs.emplace_back(sz);
          sh.bumpPtr = sh.slabs.back().data();
          sh.bumpLeft = sz;
          if (sh.slabSz < kSlabMax) sh.slabSz = std::min(kSlabMax, sh.slabSz * 2);
        }
        rec = sh.bumpPtr;
        sh.bumpPtr += qlen; sh.bumpLeft -= qlen;
      }
      std::memcpy(rec, qbuf.data(), qlen);
      // A local index: reuse a freed one, else grow. The record pointer is published BEFORE
      // the release-store of `n`, so any reader holding an index < n sees readable storage.
      if (!sh.freeIdx.empty()) {
        loc = sh.freeIdx.back();
        sh.freeIdx.pop_back();
        sh.chunks[loc >> kChunkBits][loc & kChunkMask] = rec;
      } else {
        loc = sh.n.load(std::memory_order_relaxed);
        const int c = loc >> kChunkBits;
        if (c >= kChunks) throw std::runtime_error("zone intern table overflow (too many distinct zones)");
        if (!sh.chunks[c]) sh.chunks[c] = new const std::uint8_t*[kChunk];
        sh.chunks[c][loc & kChunkMask] = rec;
        sh.n.store(loc + 1, std::memory_order_release);
      }
      sh.htab[slot] = loc + 1;
      ++sh.hcount;
      if (sh.hcount * 10 >= (sh.hmask + 1) * 7) rehash_locked(sh);  // htab-only; records unaffected
    }
    const int idx = enc(loc, sid);
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
    for (int sid = 0; sid < kShards; ++sid) {
      Shard& sh = shards[sid];
      sh.htabScratch.assign(sh.htab.size(), 0);   // reused buffer; htab.size() only grows
      std::vector<std::int32_t>& nh = sh.htabScratch;
      const std::uint64_t nmask = sh.htab.size() - 1;
      std::size_t live = 0;
      for (std::size_t s = 0; s < sh.htab.size(); ++s) {
        const std::int32_t v = sh.htab[s];
        if (!v) continue;
        const int loc = v - 1;
        const int idx = enc(loc, sid);            // `alive` is indexed by the ENCODED zidx
        std::uint8_t* rec = const_cast<std::uint8_t*>(record(idx));
        if (idx < static_cast<int>(alive.size()) && alive[idx]) {
          std::size_t i = hash_bytes(rec, mindbm_size(rec)) & nmask;
          while (nh[i]) i = (i + 1) & nmask;
          nh[i] = loc + 1;
          ++live;
        } else {
          sh.freeRecs[static_cast<std::uint32_t>(mindbm_size(rec))].push_back(rec);
          sh.freeIdx.push_back(loc);
        }
      }
      sh.htab.swap(nh);
      sh.hmask = nmask;
      sh.hcount = live;
    }
    // INVALIDATE THE PER-THREAD CACHES. Mandatory for soundness, not an optimisation.
    //
    // A cache line is (record hash -> index), and intern()'s fast path validates a hit by
    // comparing the RECORD BYTES. Sweeping does not erase a freed record -- it pushes the
    // memory onto freeRecs and the index onto freeIdx -- so a freed entry still compares
    // equal to the zone it used to hold, and the fast path would hand back an index that is
    // on the free list. The caller stores that index in a passed-list node; the next intern()
    // that pops the same index writes a different record into chunks[loc], and the node now
    // denotes a DIFFERENT zone. That is not a lost optimisation, it is a wrong answer: the
    // state keeps its discrete key and acquires someone else's zone, so the search explores
    // states the model cannot reach (found on fischer6+, where it turned a satisfied mutual-
    // exclusion property into a spurious counterexample).
    //
    // Byte equality cannot distinguish the two cases, so the cache has to be dropped. It costs
    // nthreads * 2^kCacheBits stores on a sweep, and sweeps are O(log) in the run.
    for (auto& tc : tcache) std::fill(tc.begin(), tc.end(), CacheLine{0, -1});
    ++epoch;   // invalidates every thread's get() decode memo, same reason as above
  }

  // Records currently handed out: live zones PLUS garbage not yet reclaimed. `n` is a
  // high-water index counter that never shrinks (sweep recycles indices into `freeIdx`
  // instead of renumbering), so `n` alone cannot drive the sweep trigger -- it crosses any
  // fixed threshold once and stays there. This is the quantity a growth-based trigger must
  // watch: it rises with every fresh intern and drops back to the live set after a sweep.
  std::size_t allocated() const {
    std::size_t a = 0;
    for (const auto& sh : shards)
      a += static_cast<std::size_t>(sh.n.load(std::memory_order_relaxed)) - sh.freeIdx.size();
    return a;
  }

  // Exclusive upper bound on any zidx currently handed out -- what the GC's `alive` array must
  // be sized to. Every valid index is (local << kShardBits) | shard with local < shard.n, so
  // the largest possible index is below (max local n) << kShardBits.
  std::size_t index_bound() const {
    int mx = 0;
    for (const auto& sh : shards) mx = std::max(mx, sh.n.load(std::memory_order_relaxed));
    return static_cast<std::size_t>(mx) << kShardBits;
  }


};

}  // namespace detail
}  // namespace ptctlmc
