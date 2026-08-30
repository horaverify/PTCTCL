/// @file net_passed.hpp
/// @brief Flat open-addressing passed/waiting table for one shared-passed stripe.
#pragma once

#include "hora/zone.hpp"
#include "hora/detail/engine_config.hpp"
#include "hora/detail/sym_state.hpp"
#include "hora/detail/zone_intern.hpp"
#include "hdbm/dbm_ops_legacy.hpp"
#include "hdbm/mindbm.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

namespace hora {
namespace detail {

/// @brief Subsumption test used by the passed set: a_LU by default, G-simulation under `--sim=g`.
///
/// @param Z    the candidate zone, as a dense canonical matrix
/// @param Zp   the stored zone to test against
/// @param dim  zone dimension, so each matrix is `dim * dim`
/// @param L    per-clock lower bounds for a_LU
/// @param U    per-clock upper bounds for a_LU
/// @param Lr   per-clock lower bounds carrying strictness, for the G-simulation
/// @param Ur   per-clock upper bounds carrying strictness; empty selects a_LU regardless
/// @return @c true iff @p Z is simulated by @p Zp, so @p Z may be discarded
///
/// @warning Both ::NetPassed::add and ::NetPassed::probe_kill must use this one function. If the
///          two drifted, the lock-free probe could discard a state the authoritative merge would
///          have kept.
/// @note The G-simulation differs from a_LU only in carrying each guard's strictness rather than
///       its constant alone.
inline bool sim_le(const raw_t* Z, const raw_t* Zp, int dim,
                   const std::vector<std::int32_t>& L, const std::vector<std::int32_t>& U,
                   const std::vector<raw_t>& Lr, const std::vector<raw_t>& Ur) {
  if (detail::options().gsim && !Ur.empty()) {
    return dbm_is_g_le(Z, Zp, dim, Lr.data(), Ur.data());
  }
  return dbm_is_alu_le(Z, Zp, dim, L.data(), U.data());
}

/// @brief Flat open-addressing passed/waiting table for one shared-passed stripe.
///
/// Each discrete state owns an antichain of zones. @c tab is an open-addressing slot array
/// mapping a packed discrete key to the arena index of its antichain head, and @c arena is a
/// deque of nodes with stable addresses, each holding one zone and the index of the next zone for
/// the same key. Subsumed or merged nodes are tombstoned rather than freed, and reclaimed in bulk
/// by ::NetPassed::maybe_compact.
///
/// @invariant Every occupied slot has at least one live node, so a chain never becomes empty and
///            the open-addressing probe chain is never broken.
/// @note The slot array holds only small (key, index) pairs, so growing it moves no zones.
struct NetPassed {
  /// @brief One stored zone in a discrete state's antichain.
  ///
  /// @note A node stores only the zone and its BFS level. The discrete part is not kept: it is
  ///       packed losslessly in the chain's key and is identical for every zone in the chain, so
  ///       storing it per node would be pure redundancy. It is reconstructed from the key only
  ///       when a survivor re-enters the frontier. The merge never needs it — it reads stored
  ///       nodes only for their zone, and the bounds come from the *incoming* state.
  struct Node {
    int zidx;   ///< index into the shared ZoneIntern
    int level;  ///< BFS level at which this zone was stored
    int next;   ///< next node of the same chain, or -1
    bool dead;  ///< tombstoned by subsumption or merging
  };

  /// @brief One slot of the open-addressing table: a discrete key and its antichain head.
  struct Slot {
    std::uint64_t dk;       ///< the packed discrete key
    int head;               ///< arena index of the antichain head, or -1 when the slot is free
    std::uint32_t visits;   ///< times add() has walked this chain; drives mirror promotion
  };
  std::deque<Node> arena;
  std::vector<Slot> tab;
  std::uint64_t mask;
  std::size_t used = 0;         ///< distinct discrete states, i.e. occupied slots
  std::size_t lastCompact = 0;  ///< arena.size() at the last compaction
  /// Tombstone count, maintained incrementally by the sites that retire a node. Counting dead
  /// nodes by scanning the arena would be O(arena) and would have to be rate-limited, which is
  /// what keeps compaction from firing when waste actually peaks; maintaining it makes the
  /// trigger O(1) and so evaluable at every level.
  std::size_t deadCount = 0;
  ZoneIntern* zi = nullptr;     ///< shared distinct-zone table; must be set before the first add

  /// @brief Dense, uncompressed copy of one long chain's zones.
  ///
  /// Insertion folds several predicates over a discrete state's whole antichain, and the stored
  /// zones are packed minimal-graph records, so each visit would otherwise cost a decompression —
  /// a full O(dim^3) closure. That is the right trade while chains hold about one zone, as they
  /// do on process networks, but the cost is quadratic in chain length, and models whose entire
  /// passed set is one long chain of mutually incomparable zones do exist.
  ///
  /// A mirrored chain keeps its zones dense and canonical in append order, so the folds read them
  /// in place with no decompression and no copy.
  ///
  /// @invariant The packed records remain the storage of record; this is only a cache, so
  ///            dropping a mirror is always safe.
  /// @invariant `z` and `node` are parallel: slot @c i of `node` describes the zone at offset
  ///            `i * d2` of `z`.
  struct HotChain {
    std::vector<raw_t> z;   ///< dense zones, `zdim^2` entries each, in append order
    std::vector<int> node;  ///< parallel arena indices; -1 once the zone is retired
    std::size_t ndead = 0;  ///< retired slots, triggering compaction past half
  };
  std::unordered_map<std::uint64_t, HotChain> hot;   ///< discrete key -> mirror, for mirrored chains only

  // A chain is mirrored once the decompression it has already cost, visits * length, crosses
  // detail::kHotWork. Promoting on accumulated work rather than on length alone lets short but
  // heavily hammered chains promote too, which a length-only rule can never do, and introduces no
  // model-specific constant: a chain earns its mirror by being expensive, which is exactly the
  // property that makes the mirror pay. See DESIGN.md 8.11.
  //
  // Mirror policy, the merge window and the legacy-engine switch all live in
  // detail/engine_config.hpp, since they are process-wide policy rather than per-table state.

  /// @brief Construct an empty table; the slot array grows on demand.
  NetPassed() : tab(64, Slot{0, -1, 0}), mask(63) {}

  /// @brief Fibonacci-hash a discrete key, to spread keys whose low bits are structured.
  static std::uint64_t mix(std::uint64_t dk) { return dk * 0x9e3779b97f4a7c15ULL; }

  /// @brief Find the slot owning @p dk, or the free slot where it would be inserted.
  /// @param dk the packed discrete key
  /// @return an index into @c tab; the slot is free iff its @c head is -1
  /// @pre The table has a free slot, which the load-factor check in ::add guarantees.
  std::size_t slot_of(std::uint64_t dk) const {
    std::size_t i = static_cast<std::size_t>(mix(dk) & mask);
    while (tab[i].head != -1 && tab[i].dk != dk) i = (i + 1) & mask;
    return i;
  }
  /// @brief Double the slot array and reinsert every occupied slot.
  /// @post Arena nodes are untouched; only slot positions change.
  void grow() {
    const std::size_t ncap = (mask + 1) * 2;
    std::vector<Slot> nt(ncap, Slot{0, -1, 0});
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

  /// @brief Read-only test: would ::add discard this state as already covered?
  ///
  /// This is ::add's fast path with every write removed, so it can run under a *shared* stripe
  /// lock and let the common "already covered" answer be decided concurrently instead of
  /// serialising on the exclusive lock. It answers the large majority of insertions on models
  /// whose symmetry sends many successors to the same discrete state, and hence the same stripe.
  ///
  /// @tparam BoundsFn  callable filling the per-location bounds of the incoming state
  /// @param dk        the packed discrete key
  /// @param s         the candidate state
  /// @param exact_sub compare by plain inclusion instead of the simulation test
  /// @param bounds    fills the bounds from the incoming state's locations
  /// @return @c true iff ::add would certainly return -1 for this state
  ///
  /// @post The table is unmodified. In particular @c visits is not touched, since that field is
  ///       written only under the exclusive lock.
  ///
  /// @note This is a filter, not an oracle: @c false means "fall through to ::add", which redoes
  ///       the test and the full merge. A false negative therefore costs one wasted lock
  ///       acquisition and can never change the result.
  /// @note Sound because a stored zone is only ever retired by one that subsumes it, so a @c true
  ///       here remains true even if the chain changes before ::add runs.
  /// @note For a mirrored chain the probe runs the full simulation test rather than only the
  ///       packed plain-inclusion filter. Those are exactly the zone-heavy chains, where nearly
  ///       every successor is simulated but not plainly included, so the weak filter would pass
  ///       them all through to the exclusive lock and serialise the search.
  template <class BoundsFn>
  bool probe_kill(std::uint64_t dk, const SymState& s, bool exact_sub, BoundsFn&& bounds) const {
    const std::size_t si = slot_of(dk);
    if (tab[si].head == -1 || tab[si].dk != dk) return false;   // absent, or a probe-chain miss
    const raw_t* sz = s.zone.data().data();
    const int dim = s.zone.dim();
    auto hit = hot.find(dk);
    if (hit != hot.end()) {
      const HotChain& hc = hit->second;
      const std::size_t d2 = static_cast<std::size_t>(dim) * dim;
      thread_local std::vector<std::int32_t> Ls, Us;
      thread_local std::vector<raw_t> Lr, Ur;
      if (!exact_sub) bounds(s.locs.data(), Ls, Us, Lr, Ur);
      for (std::size_t i = hc.node.size(); i-- > 0;) {
        if (hc.node[i] < 0) continue;
        const raw_t* ez = hc.z.data() + i * d2;
        if (exact_sub ? dbm_includes(ez, sz, dim)
                      : sim_le(sz, ez, dim, Ls, Us, Lr, Ur))
          return true;
      }
      return false;
    }
    for (int i = tab[si].head; i != -1; i = arena[i].next) {
      if (arena[i].dead) continue;
      if (mindbm_includes_dense(zi->record(arena[i].zidx), sz, dim)) return true;
    }
    return false;
  }

  /// @brief Merge a zone into a discrete state's antichain.
  ///
  /// @tparam BoundsFn  callable filling the per-location bounds of the incoming state
  /// @param s         the incoming state; only its zone is stored, the rest is discarded
  /// @param dk        the packed discrete key
  /// @param level     BFS level to stamp the stored zone with
  /// @param exact_sub compare zones by plain inclusion instead of the simulation test
  /// @param bounds    fills the bounds from the incoming state's locations
  /// @param tid       worker id, for the zone interner's per-thread free list
  /// @return the arena index of the newly stored node, or -1 if the zone was subsumed and dropped
  ///
  /// @pre Called under the stripe's **exclusive** lock.
  /// @post The chain for @p dk is again an antichain: no stored zone subsumes another.
  ///
  /// @warning @p exact_sub is required whenever the property is per-valuation rather than
  ///          per-location, because the simulation may drop concrete valuations. It is also
  ///          forced by diagonal constraints, which make the a_LU family unsound outright.
  /// @note @p bounds is invoked only when the discrete state already exists, so a brand-new key
  ///       pays nothing for it and allocates nothing.
  template <class BoundsFn>
  int add(SymState&& s, std::uint64_t dk, int level, bool exact_sub, BoundsFn&& bounds, int tid) {
    if (used * 10 >= (mask + 1) * 7) grow();
    const std::size_t si = slot_of(dk);
    if (tab[si].head == -1) {                       // brand-new discrete state (fast path)
      const int idx = static_cast<int>(arena.size());
      arena.push_back(Node{zi->intern(s.zone, tid), level, -1, false});
      tab[si] = Slot{dk, idx, 1};
      ++used;
      return idx;
    }
    // This walk is about to cost a decompression of the whole live chain unless it is mirrored,
    // so count it: `visits` is the "how often is this chain hammered" half of the promotion rule.
    // Saturating, so a pathologically hot chain cannot wrap back around to cold. Safe to write
    // here because add() holds the stripe's exclusive lock; probe_kill deliberately does not
    // touch it.
    if (tab[si].visits != UINT32_MAX) ++tab[si].visits;
    thread_local std::vector<std::int32_t> Ls, Us;
    thread_local std::vector<raw_t> Lr, Ur;
    if (!exact_sub) bounds(s.locs.data(), Ls, Us, Lr, Ur);   // only needed by the simulation test
    const int head = tab[si].head;
    // The live chain is decompressed once into a reused thread-local array and the folds below
    // read that array, rather than each fold re-walking the chain and decompressing every node it
    // visits — which matters because fold (4) restarts its scan after each successful union.
    // Chains thread through a deque via `next`, so repeated walks are also repeated
    // pointer-chasing over cache-cold nodes. Same order, same predicates, same results; only the
    // number of decompressions changes.
    //
    // (1) Fast path: decide the common "duplicate, drop it" case straight off the packed records,
    // with no decompression and no closure. mindbm_includes_dense answers "s.zone subseteq
    // stored" directly from the minimal constraint system in O(|G|), where |G| is on the order of
    // dim, against the O(dim^3) closure that decompressing a stored zone costs.
    //
    // Firing it early is sound in both modes:
    //   * exact_sub: test (2) is plain inclusion, so this is that test verbatim.
    //   * simulation: plain inclusion implies simulation — a witness would need
    //     Zp[y][x] < Z[y][x], which cannot hold when Z subseteq Z'. So an early hit is a
    //     conclusion the simulation test would also have reached.
    // When it does not fire, control falls through to the full dense folds, so nothing is lost
    // and the stored-state counts are unchanged.
    for (int i = head; i != -1; i = arena[i].next) {
      if (arena[i].dead) continue;
      if (mindbm_includes_dense(zi->record(arena[i].zidx), s.zone.data().data(), s.zone.dim()))
        return -1;
    }
    // `zp` points at each live zone's dense canonical matrix, newest-first, and `live` indexes
    // those still in play. Removals go through `live` with an ordered erase, never a
    // swap-with-last: the convex union in (4) is order-sensitive, so the survivors must keep
    // their newest-first relative order for the result to match the chain-walking one exactly.
    //
    // The zones come either from this chain's dense mirror, with no decompression at all, or from
    // a reused thread-local scratch filled once per call.
    const int dim = zi->zdim();
    const std::size_t d2 = static_cast<std::size_t>(dim) * dim;
    thread_local std::vector<const raw_t*> zp;
    thread_local std::vector<int> cidx;
    thread_local std::vector<std::size_t> live, hpos;   // hpos: slot in the mirror, or npos
    thread_local std::vector<raw_t> scratch;
    zp.clear(); cidx.clear(); live.clear(); hpos.clear();
    constexpr std::size_t kCold = static_cast<std::size_t>(-1);
    auto hit = hot.find(dk);
    HotChain* hc = (hit == hot.end()) ? nullptr : &hit->second;
    if (hc) {
      for (std::size_t i = hc->node.size(); i-- > 0;) {          // backwards = newest first
        if (hc->node[i] < 0) continue;
        live.push_back(zp.size());
        zp.push_back(hc->z.data() + i * d2);
        cidx.push_back(hc->node[i]);
        hpos.push_back(i);
      }
    } else {
      std::size_t nc = 0;
      for (int i = head; i != -1; i = arena[i].next) if (!arena[i].dead) ++nc;
      scratch.resize(nc * d2);            // sized before any pointer into it is taken
      std::size_t k = 0;
      for (int i = head; i != -1; i = arena[i].next) {
        if (arena[i].dead) continue;
        zi->get_into_raw(arena[i].zidx, scratch.data() + k * d2);
        live.push_back(k);
        zp.push_back(scratch.data() + k * d2);
        cidx.push_back(i);
        hpos.push_back(kCold);
        ++k;
      }
    }
    const raw_t* sz = s.zone.data().data();
    auto retire = [&](std::size_t k) {
      arena[cidx[live[k]]].dead = true; ++deadCount;
      if (hc && hpos[live[k]] != kCold) { hc->node[hpos[live[k]]] = -1; ++hc->ndead; }
    };
    // (2) s subsumed by a stored zone? -> drop
    for (std::size_t k = 0; k < zp.size(); ++k)
      if (exact_sub ? dbm_includes(zp[k], sz, dim)
                    : sim_le(sz, zp[k], dim, Ls, Us, Lr, Ur))
        return -1;
    // (3) drop stored zones now subsumed by s
    for (std::size_t k = 0; k < live.size();) {
      const raw_t* ez = zp[live[k]];
      if (exact_sub ? dbm_includes(sz, ez, dim)
                    : sim_le(ez, sz, dim, Ls, Us, Lr, Ur)) {
        retire(k);
        live.erase(live.begin() + k);
      } else ++k;
    }
    // (4) Convex-union merge: absorb stored zones whose union with s is exactly their union.
    // try_union mutates only s.zone, the incoming fresh zone; the interned zone is read-only, so
    // this can never disturb another state's stored zone.
    //
    // Bounded to the kMergeScan newest live zones. Merging is an optimisation and never a
    // requirement — skipping one only leaves two zones where one would do — so capping the scan
    // is sound, and an unbounded scan is quadratic exactly where chains are long. Zones that
    // merge are nearly always recent neighbours, since a merge happens when a successor extends a
    // zone stored moments earlier, so a small window retains effectively all of the reduction.
    constexpr std::size_t mscan = detail::kMergeScan;
    for (bool merged = true; merged;) {
      merged = false;
      for (std::size_t k = 0; k < std::min(live.size(), mscan); ++k) {
        if (dbm_try_union(s.zone.data().data(), zp[live[k]], dim)) {
          retire(k);
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
    // Keep the mirror in step, and promote a chain that has just earned one. After (4), `s.zone`
    // is canonical and is the zone actually stored, so it is what the mirror records.
    if (hc) {
      hc->z.insert(hc->z.end(), s.zone.data().begin(), s.zone.data().end());
      hc->node.push_back(idx);
      detail::hot_bytes().fetch_add(d2 * sizeof(raw_t), std::memory_order_relaxed);
      if (hc->ndead * 2 > hc->node.size()) compact_hot(*hc, d2);
    } else {
      // A chain earns its mirror once the decompression it has already cost, visits * length,
      // crosses detail::kHotWork, so short but hot chains promote too. `visits` was incremented
      // on entry to this walk. Admission is also capped by a process-wide byte budget.
      const std::size_t L = live.size() + 1;
      const std::size_t cost = L * d2 * sizeof(raw_t);
      const bool want = L >= detail::kHotMinLen &&
                        static_cast<std::size_t>(tab[si].visits) * L >= detail::kHotWork;
      if (want && detail::hot_bytes().load(std::memory_order_relaxed) + cost <= detail::hot_budget()) {
        HotChain& nh = hot[dk];
        nh.z.reserve(L * d2);
        for (std::size_t k = live.size(); k-- > 0;) {            // oldest-first append order
          nh.z.insert(nh.z.end(), zp[live[k]], zp[live[k]] + d2);
          nh.node.push_back(cidx[live[k]]);
        }
        nh.z.insert(nh.z.end(), s.zone.data().begin(), s.zone.data().end());
        nh.node.push_back(idx);
        detail::hot_bytes().fetch_add(cost, std::memory_order_relaxed);
      }
    }
    return idx;
  }

  /// @brief Drop retired slots from a mirror, preserving append order.
  /// @param[in,out] hc  the mirror to compact
  /// @param d2          entries per dense zone, i.e. `zdim * zdim`
  /// @post `hc.ndead` is zero and the surviving zones keep their relative order.
  /// @note Amortised O(1) per retirement: it runs only once half the slots are dead, and a
  ///       rebuild costs O(live).
  static void compact_hot(HotChain& hc, std::size_t d2) {
    std::vector<raw_t> nz;
    std::vector<int> nn;
    nz.reserve(hc.z.size());
    nn.reserve(hc.node.size());
    for (std::size_t i = 0; i < hc.node.size(); ++i) {
      if (hc.node[i] < 0) continue;
      nz.insert(nz.end(), hc.z.begin() + i * d2, hc.z.begin() + (i + 1) * d2);
      nn.push_back(hc.node[i]);
    }
    // Return the reclaimed bytes to the budget. Without this, a long run of promote/retire churn
    // would ratchet detail::hot_bytes() upwards until admission stopped permanently.
    detail::hot_bytes().fetch_sub((hc.z.size() - nz.size()) * sizeof(raw_t), std::memory_order_relaxed);
    hc.z.swap(nz);
    hc.node.swap(nn);
    hc.ndead = 0;
  }

  /// Minimum arena size before compaction is considered.
  ///
  /// @note The floor is deliberately at *stripe* scale, not whole-passed-set scale. The passed
  ///       set is split over at least a thousand stripes, so each holds only a small fraction of
  ///       the total nodes, and a floor chosen for the whole set would never be reached by any
  ///       individual stripe — leaving tombstones to accumulate without bound. The waste ratio
  ///       tested in ::maybe_compact is the real guard; this floor only skips trivially small
  ///       stripes.
  static constexpr std::size_t kCompactFloor = 512;

  /// @brief Reclaim tombstones by rebuilding every chain with its live nodes only.
  ///
  /// Subsumption and merging mark nodes dead but never free them, so the arena otherwise
  /// accumulates the entire history of the search. Insertion walks dead nodes before skipping
  /// them, so tombstones are paid for on every later probe.
  ///
  /// @pre Called between levels with **exactly one thread per stripe** and no other access to
  ///      this table: it swaps the arena and rewrites every slot head and next index.
  /// @post @c deadCount is zero, every mirror's arena indices are renumbered to match, and every
  ///       occupied slot still has at least one live node.
  /// @note Amortised O(1) per tombstone: a rebuild costs O(live) and drops every dead node, so no
  ///       rate-limiting guard is needed and the trigger is a comparison rather than a scan.
  void maybe_compact() {
    const std::size_t sz = arena.size();
    // Compact once a third of the stripe is tombstones.
    if (sz < kCompactFloor || deadCount * 3 < sz) return;
    // Arena indices are renumbered here and the dense mirrors hold them, so the old-to-new map is
    // recorded and the mirrors rewritten below rather than dropped. Dropping them would force the
    // next insertion to decompress the whole chain again, which is the very cost the mirror
    // exists to remove.
    std::vector<int> remap;
    if (!hot.empty()) remap.assign(sz, -1);
    std::deque<Node> na;
    for (auto& s : tab) {
      if (s.head == -1) continue;
      int newhead = -1, tail = -1;
      for (int i = s.head; i != -1; i = arena[i].next) {
        if (arena[i].dead) continue;
        const int ni = static_cast<int>(na.size());
        na.push_back(Node{arena[i].zidx, arena[i].level, -1, false});
        if (!remap.empty()) remap[i] = ni;
        if (newhead == -1) newhead = ni; else na[tail].next = ni;
        tail = ni;
      }
      s.head = newhead;   // never -1: every occupied slot keeps at least one live node
    }
    if (!remap.empty()) {
      const std::size_t d2 = static_cast<std::size_t>(zi->zdim()) * zi->zdim();
      for (auto& hp : hot) {
        // A live mirror slot always names a live arena node (retire() nulls the slot in the
        // same breath as the tombstone), so remap[] never hands back -1 for one.
        for (int& n : hp.second.node) if (n >= 0) n = remap[n];
        compact_hot(hp.second, d2);        // drops the nulled slots and resets ndead
      }
    }
    arena.swap(na);
    lastCompact = arena.size();
    deadCount = 0;                 // every tombstone was dropped by the rebuild above
  }

  /// @brief Visit the live zones of one discrete state that carry a given level stamp.
  ///
  /// @tparam F  callable taking the interned zone index of each matching zone
  /// @param dk     the packed discrete key
  /// @param level  the BFS level to select
  /// @param f      invoked once per matching zone
  ///
  /// @pre No worker is mutating this table; the level-end collection runs after every writer has
  ///      joined.
  /// @note The callback receives only the zone index. The caller reconstructs the locations and
  ///       integer variables from @p dk, which is why nodes need not store them.
  template <class F>
  void for_level(std::uint64_t dk, int level, F&& f) const {
    std::size_t i = static_cast<std::size_t>(mix(dk) & mask);
    while (tab[i].head != -1 && tab[i].dk != dk) i = (i + 1) & mask;
    if (tab[i].head == -1) return;
    for (int n = tab[i].head; n != -1; n = arena[n].next)
      if (!arena[n].dead && arena[n].level == level) f(arena[n].zidx);
  }

  /// @brief Accumulate this stripe's live-zone and distinct-discrete-state counts.
  ///
  /// @param[in,out] live  incremented by the live zones held here
  /// @param[in,out] disc  incremented by the distinct discrete states held here
  ///
  /// @note Both are O(1); neither scans the arena.
  /// @note Their ratio is the signal the adaptive abstraction switch reads at each level: it says
  ///       whether this model's zones are collapsing to one antichain entry per discrete state or
  ///       piling up as mutually incomparable alternatives.
  void counts(std::size_t& live, std::size_t& disc) const {
    live += arena.size() - deadCount;
    disc += used;
  }

};

}  // namespace detail
}  // namespace hora
