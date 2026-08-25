/// @file net_passed.hpp
/// @brief Flat open-addressing passed/waiting table for one shared-passed stripe.
#pragma once

#include "ptctlmc/detail/engine_config.hpp"
#include "ptctlmc/detail/sym_state.hpp"
#include "ptctlmc/detail/zone_intern.hpp"
#include "ptctlmc/dbm_ops_legacy.hpp"
#include "ptctlmc/mindbm.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

namespace ptctlmc {
namespace detail {

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


// One place decides which simulation the passed set subsumes with, so add() and probe_kill()
// cannot drift apart. a_LU is the default; the G-simulation differs only in carrying each
// guard's strictness (see dbm_is_g_le).
inline bool sim_le(const raw_t* Z, const raw_t* Zp, int dim,
                   const std::vector<std::int32_t>& L, const std::vector<std::int32_t>& U,
                   const std::vector<raw_t>& Lr, const std::vector<raw_t>& Ur) {
  if (detail::options().gsim && !Ur.empty()) {
#ifndef NDEBUG
#endif
    static std::atomic<long> nG{0};
    if (std::getenv("PTCTLMC_STATS") && nG.fetch_add(1, std::memory_order_relaxed) == 0)
      std::fprintf(stderr, "[gsim] G-simulation test ACTIVE\n");
    return dbm_is_g_le(Z, Zp, dim, Lr.data(), Ur.data());
  }
  return dbm_is_alu_le(Z, Zp, dim, L.data(), U.data());
}

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
  // `visits`: how many times add() has walked this chain. Drives adaptive mirror promotion
  // (see detail::hot_work()). Free -- it lands in the padding after `head`, so Slot stays 16 bytes.
  // Only ever written under the stripe's EXCLUSIVE lock; the shared-lock kill probe must not
  // touch it.
  struct Slot { std::uint64_t dk; int head; std::uint32_t visits; };
  std::deque<Node> arena;
  std::vector<Slot> tab;
  std::uint64_t mask;
  std::size_t used = 0;   // distinct discrete states (occupied slots)
  std::size_t lastCompact = 0;  // arena.size() at last compaction (geometric trigger)
  // Live tombstone count, maintained by the two sites that retire a node. Counting dead nodes
  // by scanning the arena is O(arena) and had to be rate-limited behind a doubling guard, which
  // meant compaction almost never fired at the moment waste actually peaked. Tracking it
  // incrementally makes the trigger O(1), so it can be evaluated every level.
  std::size_t deadCount = 0;
  std::size_t deadBySubsume = 0, deadByMerge = 0;   // PTCTLMC_STATS attribution
  // [RDAE PROBE, temporary] histogram of (retirement level - insertion level). Bucket 0 = the
  // node died inside its OWN level, i.e. it was never emitted into a frontier and cost nothing.
  // Bucket g>=1 = it was emitted and expanded g levels before the covering zone showed up.
  std::uint64_t gapHist[66] = {0};
  ZoneIntern* zi = nullptr;   // shared distinct-zone table (set before first add)

  // DENSE MIRROR of a long chain.
  //
  // add() folds three predicates over the whole antichain of a discrete state, and the stored
  // zones are packed min-graph records, so each visit costs a mindbm_unpack (a full O(dim^3)
  // closure). That is the right trade while chains are ~1 zone long -- which they are on every
  // process-network benchmark here (phil 1.11 zones/state, viking 1.00) -- but it is QUADRATIC
  // in the chain length, and zone-heavy models do exist: the `single` family is one automaton
  // with one location and N independent clocks, so its ENTIRE passed list is one chain of ~4.6k
  // incomparable zones. That made 4.6k inserts x 4.6k unpacks = 21.6M closures, i.e. 22 of its
  // 22.6 seconds.
  //
  // Past kHotLen live zones a chain therefore also keeps its zones DENSE and canonical, in
  // append order (iterate backwards for the newest-first order the convex-union merge needs).
  // The fold then reads them in place -- no unpack, no copy -- and the mirror is maintained
  // incrementally: push_back on insert, `node[i] = -1` on retire, compacted when half-dead. The
  // packed records stay the storage of record; this is a cache, and it exists only for chains
  // that are long enough to pay for it.
  struct HotChain {
    std::vector<raw_t> z;      // dense zones, zdim^2 each, append order
    std::vector<int> node;     // parallel arena indices; -1 once retired
    std::size_t ndead = 0;
  };
  std::unordered_map<std::uint64_t, HotChain> hot;   // dk -> mirror

  // Live zones a chain must reach before it is mirrored densely. The mirror trades memory (a
  // dense copy of the chain, vs. the packed record the interner shares between identical zones)
  // for dropping an O(chain) unpack from every insert, so the threshold decides which chains are
  // long enough to be worth it. PTCTLMC_HOTLEN overrides it; 0 disables mirroring entirely.
  //
  // A FIXED length threshold is the wrong question, and measurably so. What the mirror saves is
  // an O(L*dim^3) unpack on EVERY VISIT to a chain of L live zones; what it costs is L*dim^2*4
  // bytes, paid once. The payoff is therefore driven by visits*L, not by L alone -- a length-16
  // chain walked 100 times wastes far more than a length-64 chain walked once. Tuned on `single`
  // (chains ~1000 long) the constant landed at 64, which never fires on process networks, where
  // chains are short but hammered: phil13 burned 1.14e9 instructions in mindbm_unpack (11.5% of
  // the run, 6.7x the pre-interning baseline) waiting for a threshold it never reached.
  //
  // So promote on accumulated work instead: visits*L >= detail::hot_work(). This GENERALISES the old
  // rule rather than replacing it -- a length-64 chain still promotes on its first visit
  // (64*1 >= 64), while a length-16 chain now promotes on its 4th and a length-8 on its 8th.
  // Nothing needs to be known up front and no model-specific constant is involved: a chain earns
  // its mirror by being expensive, which is exactly the property that makes the mirror pay.
  // Mirror policy, merge window and the legacy-engine switch all live in
  // detail/engine_config.hpp -- they are process-wide policy, not per-table state.

  NetPassed() : tab(64, Slot{0, -1, 0}), mask(63) {}   // grows on demand; 16k stripes x 64 slots

  static std::uint64_t mix(std::uint64_t dk) { return dk * 0x9e3779b97f4a7c15ULL; }

  std::size_t slot_of(std::uint64_t dk) const {
    std::size_t i = static_cast<std::size_t>(mix(dk) & mask);
    while (tab[i].head != -1 && tab[i].dk != dk) i = (i + 1) & mask;
    return i;
  }
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

  // Merge zone `s.zone` (stamped `level`) into dk's antichain. Only the zone is
  // stored; the rest of `s` is discarded. Returns the arena index of the newly
  // stored node (a new survivor) or -1 if subsumed and dropped. `bounds(locs, ..)`
  // fills the per-location aLU bounds from the INCOMING state; it is only invoked
  // when the discrete state already exists, so a brand-new dk pays nothing for it.
  // READ-ONLY duplicate test: is `z` already covered by a stored zone for `dk`?
  //
  // This is add()'s fast path, verbatim, with nothing that writes. On phil13 it answers 83% of
  // all 23M add() calls, and running it under a SHARED lock lets those answer concurrently
  // instead of serializing on the stripe's mutex. Measured before this: 374% CPU of a possible
  // 800%, with 615k voluntary context switches -- threads were blocking, not computing, because
  // phil's symmetry sends ~14.5 successors to the SAME discrete state (hence the same stripe).
  //
  // A `true` here is only a FILTER: it means "add() would have returned -1", so the caller skips
  // the state. Anything else falls through to the exclusive-locked add(), which redoes this test
  // and the full merge, so a false negative costs one wasted lock and never changes the result.
  // Plain inclusion is sound in both subsumption modes for the same reason add() documents.
  //
  // For a chain with a DENSE MIRROR the probe runs add()'s real test (2) -- aLU subsumption, or
  // plain inclusion in exact mode -- instead of only the packed plain-inclusion filter. Those
  // chains are exactly the zone-heavy ones, where nearly every successor is aLU-subsumed but
  // NOT plainly included, so the weak filter passed them all through to the exclusive lock and
  // the search ran single-file however many cores it had (`single6`: 733% CPU, zero speedup).
  // Under the shared lock the same work runs on every core. Sound for the same reason: a stored
  // zone is only ever retired by one that subsumes it, so a `true` here stays true.
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

  template <class BoundsFn>
  // `exact_sub`: compare zones with plain inclusion instead of the aLU test. Required whenever
  // the property is per-VALUATION rather than per-location (deadlock), because aLU may drop
  // concrete valuations; also forced by diagonal constraints, which make aLU unsound outright.
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
    // This walk is about to cost an unpack of the whole live chain unless it is mirrored, so
    // count it: `visits` is the "how often is this chain hammered" half of the promotion rule in
    // detail::hot_work(). Saturating, so a pathologically hot chain cannot wrap back to cold. Safe to
    // write here -- add() runs under the stripe's exclusive lock (the shared-lock kill probe in
    // maybe_subsumed() deliberately does not touch it).
    if (tab[si].visits != UINT32_MAX) ++tab[si].visits;
    thread_local std::vector<std::int32_t> Ls, Us;
    thread_local std::vector<raw_t> Lr, Ur;
    if (!exact_sub) bounds(s.locs.data(), Ls, Us, Lr, Ur);   // only needed by the simulation test
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
    //   * exact_sub: test (2) IS plain inclusion, so this is that test, verbatim.
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
    // `zp` points at each live zone's DENSE canonical matrix, newest-first; `live` indexes the
    // ones still in play. Removals go through `live` with an ORDERED erase (never
    // swap-with-last): the convex union in (4) is order-sensitive, so the survivors must keep
    // their newest-first relative order for the merge to reproduce the chain-walking result
    // exactly. This is the same discipline merge_level uses on its own `live` array.
    //
    // The zones come either from this chain's dense mirror (no unpack at all) or, for a chain
    // still below kHotLen, from a reused thread-local scratch that is filled once per call.
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
      {   // [RDAE PROBE, temporary]
        const int g = level - arena[cidx[live[k]]].level;
        ++gapHist[g < 0 ? 0 : (g > 64 ? 65 : g)];
      }
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
        ++deadBySubsume;
        retire(k);
        live.erase(live.begin() + k);
      } else ++k;
    }
    // (4) convex-union merge: absorb stored zones whose union with s is exact.
    // try_union mutates only s.zone (the incoming fresh zone); the interned zone
    // is read-only, so this never disturbs another state's stored zone.
    //
    // Bounded to the kMergeScan NEWEST live zones. Merging is an optimization, never a
    // requirement -- skipping one only leaves two zones where one would do -- so a cap is sound,
    // and the whole-chain scan is quadratic exactly where chains are long: on `single6` it ran
    // 4.6k try_unions per insert, which was most of the run. Zones that merge are nearly always
    // recent neighbours (a merge happens when a successor extends a zone stored moments before),
    // so a small window keeps effectively all of the reduction.
    const std::size_t mscan = detail::merge_scan();
    for (bool merged = true; merged;) {
      merged = false;
      for (std::size_t k = 0; k < std::min(live.size(), mscan); ++k) {
        if (detail::engine_legacy() ? dbm_try_union_legacy(s.zone.data().data(), zp[live[k]], dim)
                            : dbm_try_union(s.zone.data().data(), zp[live[k]], dim)) {
          ++deadByMerge;
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
    // Keep the mirror in step; promote a chain that has just crossed kHotLen. `s.zone` is
    // canonical and, after (4), is the zone actually stored, so it is what the mirror holds.
    if (hc) {
      hc->z.insert(hc->z.end(), s.zone.data().begin(), s.zone.data().end());
      hc->node.push_back(idx);
      detail::hot_bytes().fetch_add(d2 * sizeof(raw_t), std::memory_order_relaxed);
      if (hc->ndead * 2 > hc->node.size()) compact_hot(*hc, d2);
    } else {
      // ADAPTIVE PROMOTION (see detail::hot_work()): a chain earns its mirror once the unpacking it has
      // already cost, visits*L, crosses the threshold -- so short-but-hot chains promote too,
      // which a length-only rule can never do. `visits` was incremented on entry to this walk.
      const std::size_t L = live.size() + 1;
      const std::size_t cost = L * d2 * sizeof(raw_t);
      const std::size_t pinned = detail::hot_len_fixed();
      const bool want = (detail::mirror_mode() == detail::MirrorMode::Off) ? false
                      : pinned ? (L >= pinned)                       // ablation: length only
                               : (L >= detail::hot_min_len() &&
                                  static_cast<std::size_t>(tab[si].visits) * L >= detail::hot_work());
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

  // Drop retired slots from a mirror, preserving append order. Amortized O(1) per retirement:
  // it only runs once half the slots are dead, and a rebuild costs O(live).
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
    // Give the reclaimed slots back to the budget, or a long run of promote/retire churn would
    // ratchet detail::hot_bytes() up until admission stops for good.
    detail::hot_bytes().fetch_sub((hc.z.size() - nz.size()) * sizeof(raw_t), std::memory_order_relaxed);
    hc.z.swap(nz);
    hc.node.swap(nn);
    hc.ndead = 0;
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
  // Floor is PER STRIPE. The passed list is split over NB (>=1024) stripes, so a stripe holds
  // roughly total_nodes/NB: on phil13 that is 3.86M/1024 ~= 3.8k, and the old 8192 floor meant
  // compaction never ran on ANY stripe (a stripe would need an 8.4M-node total to reach it).
  // Tombstones then accumulated unbounded -- 54% of the arena on phil13 -- and add()'s chain
  // scan walks dead nodes before skipping them, so 37% of the 120M nodes it visited were
  // garbage. The dead*2 < sz test below is the real guard (compact only past ~50% waste); this
  // floor only exists to skip trivially small stripes, so it belongs at stripe scale.
  static constexpr std::size_t kCompactFloor = 512;
  void maybe_compact() {
    const std::size_t sz = arena.size();
    // Compact once a THIRD of the stripe is tombstones. Rebuilding costs O(live) and drops every
    // dead node, so the work is amortized O(1) per tombstone however often this fires -- no
    // doubling guard needed, and the trigger is now a comparison rather than an arena scan.
    // add() walks dead nodes before skipping them, so waste here is paid on every later probe.
    if (sz < kCompactFloor || deadCount * 3 < sz) return;
    // Arena indices are renumbered here, and the dense mirrors hold them, so record old -> new
    // and rewrite the mirrors below rather than dropping them (dropping would force the next
    // add() to re-unpack the whole chain -- the very cost the mirror exists to remove).
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
      s.head = newhead;   // invariant: >=1 live per occupied slot, so never -1
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

  // Live zones and distinct discrete states, in O(1). The ratio between them is the signal the
  // adaptive abstraction switch reads at each level (see reach_engine): it says whether this
  // model's zones are collapsing into one antichain entry per discrete state or piling up as
  // mutually incomparable alternatives.
  void retire_causes(std::size_t& sub, std::size_t& mrg) const {
    sub += deadBySubsume; mrg += deadByMerge;
  }

  // [RDAE PROBE, temporary]
  void gap_hist(std::uint64_t* out) const { for (int i = 0; i < 66; ++i) out[i] += gapHist[i]; }

  void counts(std::size_t& live, std::size_t& disc) const {
    live += arena.size() - deadCount;
    disc += used;
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

  // Probe (PTCTLMC_FEDPROBE, measurement only): visit each discrete state's live antichain as a
  // vector of zones, so an offline pass can ask how many members a NON-convex (federation) union
  // would let it drop. Never called on any search path.
  template <class F>
  void for_each_antichain(F&& f) const {
    std::vector<Dbm> zs;
    for (const auto& s : tab) {
      if (s.head == -1) continue;
      zs.clear();
      for (int n = s.head; n != -1; n = arena[static_cast<std::size_t>(n)].next)
        if (!arena[static_cast<std::size_t>(n)].dead) zs.push_back(zi->get(arena[static_cast<std::size_t>(n)].zidx));
      if (!zs.empty()) f(zs);
    }
  }
};

}  // namespace detail
}  // namespace ptctlmc
