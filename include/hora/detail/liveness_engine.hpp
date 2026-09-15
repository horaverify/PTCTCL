/// @file hora/detail/liveness_engine.hpp
/// @brief Private data model for the network-liveness engine.
///
/// This header is deliberately private to the liveness implementation.  The public query API
/// remains in @ref hora/netreach.hpp; this file separates the engine's state and contracts
/// from the transition and fixpoint code in @c netliveness.cpp.
///
/// The production lifecycle is intentionally single-construction:
///
/// 1. The forward search discovers discrete keys, exact candidate zones, and raw semantic
///    edges.  At the same time it grows the deadlock-preserving @c D refinement.
/// 2. A newly learned @c D atom is pulled backward through already known reset-labelled key
///    edges.  A provisional cover may be revoked, so its source edge retains a compact replay
///    handle; the exact target is materialised only if the refined D map invalidates its cover.
/// 3. Once both worklists are stable, the active records are compacted into the final quotient
///    by redirecting the raw edges already stored.  No successor relation is generated again.
/// 4. The backward greatest fixpoint runs once on that stable quotient.
///
/// The small fixed-map group near the end of @ref LivenessEngine is retained for internal diagnostics only.
/// No public liveness route uses it.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hora/zone.hpp"
#include "hora/liveness.hpp"
#include "hdbm/zone_store.hpp"
#include "hora/detail/sym_state.hpp"

namespace hora {

/// @brief The symbolic recipe for one network transition.
///
/// @ref parts holds every participating local edge.  @ref abstain holds source-side guard
/// complements for broadcast receivers that did not participate; it is needed to invert that
/// broadcast exactly during the backward predecessor operation.
struct GfpTag {
  std::vector<std::pair<int, int>> parts;
  std::vector<Constraint> abstain;
};

/// @brief Liveness classification of a graph edge, carried in @ref OtfEdge::deferred.
///
/// The exact backward operator may take a predecessor through an edge only when its exact
/// successor is genuinely represented at the target.  A cover accepted by simulation rather than
/// by inclusion does not satisfy that: `X = post_e(Z_src)` can be disjoint from the stored
/// `Z_tgt`, so `Z_src /\ Pre_e(W_tgt)` is empty even when a real run goes through `X`.  Such an
/// edge is a lazy promise, not a transition, and owes a refinement obligation to any witness that
/// needs it.
///
/// These live in the `deferred` field rather than a new one because that field is dead once
/// ::LivenessEngine::online_finalize_graph runs, and every existing `deferred = -1` write already
/// means "no promise outstanding".  Negative values keep that meaning; the non-negative values
/// remain the online construction's own replay marker.
/// The BACKWARD operator and FORWARD cycle detection disagree about what counts as safe, and the
/// difference is the whole reason ::LivenessEngine::forward_buchi needs its own marker.
///
///   * `X subseteq Z_tgt` (inclusion cover) is SAFE BACKWARD -- `Pre` against a superset of the
///     exact successor loses no predecessor -- but it is a SUBSUMPTION EDGE FORWARD: it jumps to
///     a strictly larger zone, and a cycle closed through it need not be realised by any run.
///   * only an edge that CREATED its target, so `Z_tgt` is exactly `post_e(Z_src)`, is an
///     ordinary edge in the sense of Herbreteau et al.'s Theorem 6.
enum : int {
  kEdgeExact = -1,             ///< created its target: `Z_tgt == post_e(Z_src)`
  kEdgeMaterialized = -2,      ///< an unresolved cover the refinement loop has since made exact
  kEdgeSubsumed = -3,          ///< cover by inclusion: safe backward, a subsumption edge forward
};

/// @brief One graph edge: its target record and the transition recipe that produced it.
struct OtfEdge {
  int tgt;
  int tag;
  // Online construction only: a provisional target may be represented by a replay recipe
  // instead of a materialised state.  Non-negative marks that promise outstanding; the negative
  // values are the liveness classification above.  Final graph edges never stay non-negative.
  int deferred = kEdgeExact;
  // Which successor of the source this edge is, in ::tagged_successors order.  The refinement
  // loop replays `(src, ordinal)` to regenerate the exact successor a cover discarded, so it must
  // survive independently of the edge's position in its adjacency vector.
  int ordinal = -1;
};

/// @brief One half-space used by the deadlock-preserving D refinement.
struct AtomC {
  int i;
  int j;
  raw_t b;

  bool operator==(const AtomC& o) const;
};

/// @brief Sentinel for AtomMap::bump: no bound is requested for that clock.
inline constexpr std::int32_t kNoBump = -1;

/// @brief The monotone per-key D refinement used to preserve deadlock under covering.
///
/// Diagonal atoms remain key-local cell obligations in @ref diag.  A non-diagonal atom is
/// represented by raising a clock's LU bounds in @ref bump.  @ref live prevents active-clock
/// reduction from erasing values the refinement still reads.
struct AtomMap {
  std::unordered_map<std::string, std::vector<AtomC>> diag;
  std::vector<std::int32_t> bump;
  std::vector<char> live;
  bool usable = false;

  // Diagnostic counters only; no verdict depends on them.
  std::size_t seed_keys = 0;
  std::size_t seed_atoms = 0;
  std::size_t nondiag_atoms = 0;
  std::size_t closure_atoms = 0;
  std::size_t keys_tagged = 0;
  std::size_t rounds = 0;
  std::size_t bumped = 0;
  std::size_t dropped_vacuous = 0;
  std::size_t dropped_disjoint = 0;
};

/// @brief A transition shape in the discrete-key skeleton used to pull D atoms backward.
struct OnlineKeyEdge {
  int src;
  int tgt;
  int reset_class;
  int next_out;
  int next_in;
};

/// @brief The part of a key's record that only the D refinement ever touches.
///
/// None of it exists until D gains an atom: the key skeleton, the replay index and the cached
/// invariant are all built on first use.  On every benchmark row D stays empty, so this is never
/// allocated, and holding it inline cost every key 48 bytes for nothing -- on the families where
/// each reachable state has a key of its own that was the largest block after the zones.
struct OnlineKeyExtra {
  // Intrusive lists into LivenessEngine::online_key_edges_.  This avoids three heap-allocated vectors per
  // discrete key while preserving exact incoming propagation and source-local deduplication.
  int incoming = -1;
  int outgoing = -1;
  // Packed `(raw-adjacency slot, edge ordinal)` handles for revocable cover edges at this key.
  // The edge itself is the replay recipe; no separate provisional-state object is allocated.
  std::vector<std::uint64_t> deferred;
  std::unique_ptr<Dbm> inv;
  bool inv_ready = false;
};

/// @brief One discrete key of the online quotient.
///
/// The location and integer vectors are NOT here: they live in @ref LivenessEngine's key arenas at
/// a fixed stride, reached through @ref LivenessEngine::key_locs_to. Two heap vectors per key cost
/// more in headers and allocator metadata than the locations themselves, and on the families where
/// every reachable state has its own key that was the largest single block of memory after the
/// zones.
struct OnlineKeyInfo {
  int id = -1;
  // Materialised and active nodes use intrusive lists stored once per node in LivenessEngine.  Most keys
  // own exactly one record, so a heap-allocated vector per key was mostly allocator metadata.
  int node_head = -1;
  std::uint64_t packed_key = 0;
  std::uint32_t node_count = 0;
  int active_head = -1;
  std::uint32_t active_count = 0;
  bool may_deadlock = false;
  std::unique_ptr<std::vector<AtomC>> diag;
  std::unique_ptr<OnlineKeyExtra> extra;
  /// @brief The D-only part, allocated on first use.
  OnlineKeyExtra& ext() {
    if (!extra) extra = std::make_unique<OnlineKeyExtra>();
    return *extra;
  }
  int incoming() const { return extra ? extra->incoming : -1; }
  std::size_t deferred_count() const { return extra ? extra->deferred.size() : 0; }
};

/// @brief One diagonal atom waiting to be pulled through the incoming-key worklist.
struct OnlineAtomWork {
  int key;
  AtomC atom;
};

/// @brief Instrumentation for the lazy liveness refinement loop.
///
/// Counters and timings only.  Nothing here decides a verdict; @ref exhausted qualifies one, in
/// the same sense as hora::GfpStats::capped -- when it is set the loop gave up and the answer
/// came from the exact-route fallback instead.
struct LazyStats {
  bool ran = false;                  ///< the loop was entered at all
  bool exhausted = false;            ///< budget spent; verdict delegated to the exact route
  bool skipped_gfp = false;          ///< verdict reached without any backward pass at all
  std::size_t nodes_before = 0;      ///< graph size when the loop was entered
  std::size_t edges_exact = 0;       ///< edges whose exact successor is represented exactly
  std::size_t edges_covered = 0;     ///< edges accepted by simulation rather than inclusion
  std::size_t covered_in_scc = 0;    ///< of those, edges whose target key can recur
  std::size_t obligations = 0;       ///< unresolved obligations at the first backward failure
  std::size_t materialised = 0;      ///< covers turned into real transitions
  std::size_t new_nodes = 0;         ///< records created by refinement and its descendants
  std::size_t replays = 0;           ///< forward successor recomputations for replay
  std::size_t backward_visits = 0;   ///< backward relaxations across every pass
  std::size_t scc_builds = 0;        ///< key-graph SCC computations
  std::size_t cone_nodes = 0;        ///< nodes re-solved after a refinement wave
  std::size_t cone_of = 0;           ///< ... out of this many in the psi-closure
  std::size_t rounds = 0;            ///< refinement iterations completed
  std::size_t waves = 0;             ///< backward passes that produced an obligation set
  double ms_base_gfp = 0;            ///< first backward pass, before any refinement
  double ms_backward = 0;            ///< every later backward pass
  double ms_forward = 0;             ///< replay plus incremental forward expansion
  double ms_scc = 0;                 ///< key-graph construction and SCC maintenance
};

/// @brief A mutable view of one slot's edges inside ::RawAdjArena.
struct EdgeSpan {
  OtfEdge* b = nullptr;
  OtfEdge* e = nullptr;
  OtfEdge* begin() const { return b; }
  OtfEdge* end() const { return e; }
  std::size_t size() const { return static_cast<std::size_t>(e - b); }
  bool empty() const { return b == e; }
  OtfEdge& operator[](std::size_t i) const { return b[i]; }
};

/// @brief Every expanded record's raw successor edges, one contiguous run per slot, in blocks.
///
/// A record's edges are generated in one batch when it is expanded and never grow afterwards --
/// refinement only rewrites their targets -- so they need no vector of their own.  A
/// `vector<vector<OtfEdge>>` paid a 24-byte header plus an allocator block per expanded record.
/// One growing vector for all of them was the next step, and it still cost up to its own size
/// again: a doubling vector holds half its capacity idle, and each regrowth briefly holds the old
/// and the new buffer together, which is what the peak resident size records -- 67 MB of capacity
/// behind 35 MB of edges on csma13 `A[] not deadlock`.  Fixed blocks have neither cost, and a
/// span stays valid for the life of the arena.
///
/// A slot costs a 4-byte running count, whose differences are the slot lengths and which also
/// numbers the edges (the ordinary-edge bits index by it), plus an 8-byte block reference.  A run
/// that does not fit in what is left of the current block moves, whole, into a fresh one.
class RawAdjArena {
 public:
  RawAdjArena() : off_(1, 0) {}
  /// @brief Number of slots.
  std::size_t size() const { return off_.size() - 1; }
  EdgeSpan operator[](std::size_t s) const {
    const std::size_t len = off_[s + 1] - off_[s];
    if (len == 0) return EdgeSpan{};
    OtfEdge* b = blocks_[static_cast<std::size_t>(ref_[s] >> 32)].get() +
                 static_cast<std::uint32_t>(ref_[s]);
    return EdgeSpan{b, b + len};
  }
  /// @brief Open a new, empty slot after every existing one.
  int open_slot() {
    off_.push_back(off_.back());
    ref_.push_back((static_cast<std::uint64_t>(blocks_.size() ? blocks_.size() - 1 : 0) << 32) | fill_);
    return static_cast<int>(size() - 1);
  }
  /// @brief Append @p e to slot @p s. @return @c false unless @p s is the newest slot.
  bool push(std::size_t s, const OtfEdge& e) {
    if (s + 1 != size() || off_.back() >= UINT32_MAX) return false;
    const std::size_t len = off_.back() - off_[s];
    if (fill_ == cap_) {                       // no room left: a fresh block, and the run moves
      cap_ = std::max<std::size_t>(kBlock, 2 * (len + 1));
      blocks_.emplace_back(new OtfEdge[cap_]);
      OtfEdge* nb = blocks_.back().get();
      if (len) {
        const OtfEdge* old = (*this)[s].begin();
        std::copy(old, old + len, nb);
      }
      ref_[s] = static_cast<std::uint64_t>(blocks_.size() - 1) << 32;
      fill_ = len;
    }
    blocks_.back()[fill_++] = e;
    ++off_.back();
    return true;
  }
  /// @brief Running index of slot @p s's first edge.
  std::size_t first(std::size_t s) const { return off_[s]; }
  std::size_t edge_count() const { return off_.back(); }
  void clear() {
    blocks_.clear();
    ref_.clear();
    off_.assign(1, 0);
    fill_ = cap_ = 0;
  }

 private:
  static constexpr std::size_t kBlock = std::size_t(1) << 16;   ///< edges per block, 1 MiB
  std::vector<std::unique_ptr<OtfEdge[]>> blocks_;
  std::vector<std::uint64_t> ref_;             ///< per slot: block index << 32 | offset in it
  std::vector<std::uint32_t> off_;             ///< running edge count at each slot's start
  std::size_t fill_ = 0;                       ///< edges used in the newest block
  std::size_t cap_ = 0;                        ///< capacity of the newest block
};

/// @brief Private liveness engine instance: one graph construction and one backward solve.
class LivenessEngine {
 public:
  LivenessEngine(const Network& n, const DiscPred& pp, const DiscPred& qq,
      bool force_exact = false, const std::atomic<bool>* cancel = nullptr,
      const Network* dead = nullptr);

  /// @brief Decide discrete `p --> q` with the single online-D construction.
  ///
  /// A provisional D-covered successor is retained only as an edge replay handle.  If its key's
  /// D map changes, that local successor is regenerated and retested; the stable quotient then
  /// feeds one backward greatest fixpoint.
  bool decide_leads_to(GfpStats* gs);

  /// @brief Decide `E[] psi` after the same single online-D forward construction.
  bool decide_exists_invariant_run(GfpStats* gs);

  /// @brief Exact/inclusion route used for diagonal models or an explicit exact request.
  bool run(GfpStats* gs);

  /// @brief Exact/inclusion E[] route used for diagonal models or an explicit exact request.
  bool eg_from_initial(GfpStats* gs);

  /// @brief The closure and solver used by the E[] route after a graph is complete.
  std::vector<int> eg_closure();
  bool eg_probe();
  bool eg_verdict(GfpStats* gs);

  /// @brief Query `!q` without resizing its cache; the forward loop guarantees that precondition.
  bool is_nq_fast(int i);

  void set_eg_early(bool v);
  bool eg_early_hit() const;
  /// @brief Did the lazy refinement loop spend its budget without reaching a verdict?
  ///
  /// When set, ::LivenessEngine::decide_exists_invariant_run returned nothing usable and the
  /// caller owes the query a run on the exact route, which covers by inclusion and is therefore
  /// complete without any refinement at all.
  bool lazy_exhausted() const { return lazy_.exhausted; }
  bool was_cancelled() const;
  bool was_capped() const;
  void set_avoid(const detail::DelayAvoidSet* a, bool forward = true);

  // ---- diagnostic-only fixed-map support ---------------------------------
  // These methods are a reference check for the online D implementation.  No public query calls
  // them, and none is used to retry, replace, or rebuild a production graph.
  void release();
  void build_forward(GfpStats* gs);
  bool finish_verdict(GfpStats* gs);
  /// @brief Is a deadlocked valuation reachable, decided on the D-refined quotient?
  ///
  /// Builds the forward graph only, then reads the deadlock predicate off the stored zones.
  /// The covering used during that construction is the G-simulation refined by the D atoms,
  /// which preserves `Dead` (docs/design.md, "Theorem (completion sound)"), so a node covered
  /// away cannot hide a deadlock its cover lacks; and stored zones are exact under that
  /// covering, so a deadlock read off one is genuinely reachable. The relation is therefore
  /// sound in both directions here, and it is strictly coarser than the `a_M` bisimulation the
  /// plain reachability engine must fall back to.
  /// @param[out] ok  cleared when the construction did not complete, in which case the return
  ///                 value proves nothing and the caller must use another route
  bool deadlock_reachable_refined(GfpStats* gs, bool* ok);
  bool deadlock_refinement_is_empty();
  bool build_atom_map(AtomMap& am);
  void set_atoms(const AtomMap* a);

 private:
  // Query configuration.
  const Network& net;
  const DiscPred& p;
  const DiscPred& q;
  const Network* dead_net;
  const detail::DelayAvoidSet* avoid_ = nullptr;
  bool avoid_forward_ = true;
  bool scc_iter_ = true;
  bool eg_early_ = false;
  /// @brief Auto-selected depth-first `E[] psi` order for a statically single-key model.
  ///
  /// Reconstructed 2026-09-14 after an over-broad `git checkout` discarded it.
  bool online_auto_eg_dfs_ = false;
  bool eg_early_hit_ = false;
  // `p --> q`: a reached `p /\ !q` valuation that can end a maximal run is already a
  // counterexample, so the forward search may stop the moment one is found.
  bool leadsto_early_ = false;
  bool online_eg_prune_ = false;
  /// @brief `p --> q` is being decided by the forward route (src/liveness/forward_buchi.cpp).
  ///
  /// Read by ::LivenessEngine::grow: that route runs BEFORE the seal, where the online path
  /// normally leaves `pc` and `qc` unsized because nothing consults the vertex predicates until
  /// ::LivenessEngine::online_finalize_graph has rebuilt the graph.  The forward route consults
  /// them every round, to find the accepting nodes its closure is seeded from.
  bool fwd_leadsto_ = false;
  /// @brief Stop the forward search at the first record holding a deadlocked valuation.
  bool deadlock_scan_ = false;
  bool deadlock_hit_ = false;   ///< set by the scan; meaningful only when @ref deadlock_scan_
  bool online_deadlock_at(int v, bool no_successor);
  bool online_eg_repair_ = false;
  // The E[] probe budget, amortized over the WHOLE online construction rather than over one
  // drain.  ::LivenessEngine::online_drain_forward_work is entered three times -- the base
  // forward pass, the ::LivenessEngine::online_complete_eg_history repair, and each wake-up of
  // ::LivenessEngine::online_finalize_graph -- and the second and third are entered with the
  // graph already at very nearly its final size.  Held as locals, the budget restarted at zero
  // there and the first expansion bought a probe over the entire graph, which is the sweep the
  // decision then runs again.  Members keep the amortization argument the schedule is built on:
  // total probe work never exceeds total exploration work, counted over the whole run.
  std::size_t online_probed_nodes_ = 0;       ///< summed cost of every probe so far, in records
  std::size_t online_nodes_at_last_probe_ = 0;  ///< graph size when exploration last resumed
  bool force_exact_ = false;
  const std::atomic<bool>* cancel_ = nullptr;
  bool cancelled_ = false;

  // Covering and backward-solver state.
  std::vector<std::int32_t> L, U, aL, aU, bL, bU, M;
  // The per-location a_LU fold, split into the part that varies with the location vector and the
  // part that does not; see ::LivenessEngine::build_local_bound_tables.
  std::vector<std::int32_t> base_l_, base_u_;
  std::vector<std::vector<int>> vary_clocks_;
  int emode = 0;
  std::vector<char> qmask;
  bool use_alu = false;
  static constexpr std::size_t fedcap = 4096;
  bool capped = false;
  std::size_t passes = 0;
  std::size_t fixpoint_iterations = 0;
  std::size_t expand_calls = 0;
  std::size_t pre_calls = 0;
  std::size_t fed_max = 0;
  std::size_t nonempty = 0;
  double fed_avg = 0;
  bool settled_early = false;
  std::size_t outer_expands = 0;
  std::size_t outer_nodes = 0;
  bool fwd_ok_ = false;
  // --- HORA_FC prototype: SCC-local infinite-witness operator ---------------------------
  // The finite half of the verdict (deadlock, divergence) is already complete: `stop` survives
  // the operator and nonemptiness climbs the exact creating-edge spine. The infinite half is
  // where a simulation-only cover can lose a witness, because `pre` runs against a target zone
  // that need not contain the exact successor. Fix: drop `stop` (an infinite witness never uses
  // it) and widen each node's carrier from Z_v to a set that DOES contain every covered
  // successor. Extra_LU+(Z_v) is such a set and is still one DBM, unlike the exact downward
  // closure, which is non-convex. With no stop term only cyclic support survives, so the pass
  // confines itself to the SCCs without needing a separate driver.
  bool fc_mode_ = false;
  // HORA_MERGED: one operator instead of two passes. `stop` stays computed on the EXACT zone
  // (a widened zone fabricates stuck valuations), while the recursive predecessor term is
  // carried by Extra_LU+(Z_v). Sound because every surviving valuation still needs a real
  // `pre` step or a real stop; complete because tight covering puts every discarded exact
  // successor inside the target's carrier. The readout intersects the exact zone: the carrier
  // holds valuations below it that no run reaches.
  bool carrier_mode_ = false;
  // Extra_LU+(Z_v) is read once per probe by both the tight cover test and the carrier
  // operator, so recomputing it there costs a ZoneStore::load plus an O(dim^2) extrapolation
  // on the hottest path in the engine. Cached per node, and -- the case that actually pays --
  // nodes whose zone is ALREADY its own extrapolation are flagged instead of copied, so they
  // keep the baseline's no-copy `zs_.with` comparison.
  enum : char { kCarUnknown = 0, kCarSame = 1, kCarWide = 2 };
  std::vector<char> carrier_state_;
  std::vector<Dbm> carrier_;
  /// @return Extra_LU+(Z_v), or nullptr when it equals Z_v and the stored zone should be used.
  const Dbm* carrier_of(int v);
  void raise_by_psi(std::vector<std::int32_t>& l, std::vector<std::int32_t>& u) const;
  void carrier_invalidate(int v) {
    if (v >= 0 && v < (int)carrier_state_.size()) carrier_state_[v] = kCarUnknown;
  }
  // HORA_TIGHT=1: replace the a_LU covering test by containment in the Extra_LU+ carrier,
  // i.e. cover Z by Z' only when Z is a SUBSET of Extra_LU+(Z'). That makes Lemma A true by
  // construction -- the exact successor a cover discards is always inside the target's carrier --
  // so the convex (federation-free) completeness argument for the infinite case goes through.
  // Strictly finer than a_LU, so it stores more nodes; this flag exists to price that.
  bool tight_ = false;
  bool eg_verdict_fc();

  // Final graph storage. The ZoneStore owns dense zone matrices; node[i].zone is scratch-only once
  // a record is interned. The final graph's discrete half. The exact/inclusion route keeps it in
  // `node`; the online route leaves it in the key arenas and points at them through `node_key_`,
  // because several records share a key and copying the locations into every one of them cost more
  // than the arenas themselves. ::nlocs is the one place that knows which representation is live.
  std::vector<detail::SymState> node;
  std::vector<int> node_key_;
  // Records created so far. In online mode `node` stays empty until the graph is sealed, so it
  // cannot serve as the counter; ::nrec is the one place that knows which is authoritative.
  std::size_t online_nrec_ = 0;
  std::vector<std::vector<OtfEdge>> adj;
  std::vector<GfpTag> tags_;
  std::unordered_map<std::uint64_t, std::vector<int>> tagidx_;
  std::unordered_map<std::uint64_t, std::vector<int>> bucket;
  std::vector<char> expanded;
  std::vector<char> pc, qc;
  bool intern_grew = false;
  ZoneStore zs_;
  int zdim_ = 0;

  // Fixed-map data only.  Online mode instead classifies immutable records directly.
  const AtomMap* amap_ = nullptr;
  std::vector<std::uint64_t> ntag_;
  std::vector<std::uint64_t> ntag_out_;

  // Online D state.  Active exact records and raw edges survive refinement; provisional covered
  // successors use compact edge replay handles until their local D map is stable.
  bool online_d_ = false;
  AtomMap online_am_;
  std::deque<OnlineKeyInfo> online_keys_;
  // Discrete state of every key, flat: `key_locs_` at stride nprocs, `key_ints_` at stride
  // |intvars|. Empty when the network has no integer variables, which is the common case.
  // Locations are stored as narrow as the largest location count allows -- one byte for up to
  // 256 locations per process, which is every model here -- because on families where each
  // reachable state is its own key this arena is one `nprocs`-wide row per state: 84 bytes a key
  // at int width on viking20. Exactly one of the three is in use; ::key_loc_width_ says which.
  std::vector<std::uint8_t> key_loc8_;
  std::vector<std::uint16_t> key_loc16_;
  std::vector<int> key_loc32_;
  int key_loc_width_ = 4;
  std::vector<int> key_loc_scratch_;   // ::online_configure_bounds decodes a key's row here
  std::vector<long> key_ints_;
  int key_loc_stride_ = 0;
  int key_int_stride_ = 0;
  // Hash-only fallback for wide discrete keys.  Collisions are resolved by comparing the
  // immutable location and integer vectors in `online_key_by_id_`.
  std::unordered_map<std::uint64_t, std::vector<int>> online_key_index_;
  // Lossless packed-key fast path.  Models needing 64 or more discrete bits fall back to the
  // exact hash bucket above; no lossy key is ever used.
  bool online_packed_key_usable_ = false;
  std::vector<int> online_disc_shift_;
  std::vector<std::uint64_t> online_disc_mask_;
  std::vector<std::uint64_t> online_packed_key_keys_;
  std::vector<int> online_packed_key_values_;  // -1 marks an empty flat-table slot
  std::size_t online_packed_key_mask_ = 0;
  std::size_t online_packed_key_count_ = 0;
  std::vector<OnlineKeyInfo*> online_key_by_id_;
  std::vector<int> online_key_of_;
  std::vector<int> online_next_in_key_;
  std::vector<OnlineKeyEdge> online_key_edges_;
  bool online_key_graph_enabled_ = false;
  std::vector<std::vector<int>> online_reset_sets_;
  std::vector<int> online_tag_reset_class_;
  std::deque<OnlineAtomWork> online_atom_work_;
  std::vector<int> online_dirty_keys_;
  std::vector<char> online_dirty_queued_;
  bool online_global_dirty_ = false;
  bool online_failed_ = false;
  int online_bound_key_ = -1;
  std::size_t online_bound_epoch_ = 0;
  std::size_t online_bound_cached_epoch_ = std::numeric_limits<std::size_t>::max();
  std::vector<char> online_active_;
  std::vector<int> online_active_next_;
  std::vector<int> online_active_prev_;
  std::vector<int> online_cover_;
  // Only expanded representatives own an outgoing raw-edge vector.  `online_raw_slot_[v]`
  // maps a persistent active record to that compact storage, avoiding an empty vector for every
  // provisionally covered successor.
  RawAdjArena online_raw_adj_;
  /// @brief Has every active record been compacted against every record that became active later?
  ///
  /// True while activation happens only through ::online_intern's fresh-record path, whose cover
  /// scan considers every active candidate.  Cleared for good by anything else that activates a
  /// record (a D recheck, the E[] history repair, the seal's wake-up, a refinement wave) or that
  /// makes that scan skip candidates (refinement's forbidden, unsafe and merged targets), after
  /// which ::online_expand compacts again as it always did.
  bool online_compact_complete_ = true;
  /// @brief ::online_intern's convex-union scratch: per active-list position, the candidate version
  /// that record last failed to merge with.  A member only so the hot path does not allocate.
  std::vector<int> online_union_tried_;
  std::vector<int> online_raw_owner_;
  std::vector<int> online_raw_slot_;
  // ORDINARY-EDGE BITS, one per raw edge, recorded when the edge is made.  Deciding whether an
  // edge is ordinary needs its exact successor zone, which exists only at that moment: afterwards
  // it costs a full ::tagged_successors replay of the source.  A bit is meaningful only while the
  // edge still names the record it was classified against -- every retarget clears it -- and
  // only while no D atom has refined the covering relation (@ref online_edge_ord_valid_).
  // A bit's index is its edge's index in the raw arena.
  std::vector<std::uint64_t> online_edge_ord_;
  std::size_t online_edge_ord_n_ = 0;
  bool online_edge_ord_valid_ = true;
  void online_push_edge_ord(bool ordinary) {
    if ((online_edge_ord_n_ & 63) == 0) online_edge_ord_.push_back(0);
    if (ordinary) online_edge_ord_.back() |= std::uint64_t(1) << (online_edge_ord_n_ & 63);
    ++online_edge_ord_n_;
  }
  bool online_edge_ord_at(int slot, std::size_t pos) const {
    const std::size_t i = online_raw_adj_.first(slot) + pos;
    return i < online_edge_ord_n_ && ((online_edge_ord_[i >> 6] >> (i & 63)) & 1);
  }
  void online_set_edge_ord(int slot, std::size_t pos, bool ordinary) {
    if (slot < 0 || static_cast<std::size_t>(slot) >= online_raw_adj_.size()) return;
    const std::size_t i = online_raw_adj_.first(slot) + pos;
    if (i >= online_edge_ord_n_) return;
    const std::uint64_t bit = std::uint64_t(1) << (i & 63);
    if (ordinary) online_edge_ord_[i >> 6] |= bit;
    else online_edge_ord_[i >> 6] &= ~bit;
  }
  bool online_deferred_index_enabled_ = false;
  /// @brief `A[] not deadlock` keeps no raw edge while D is empty.
  ///
  /// A raw edge is read only by the D refinement -- to index replay recipes, to build the key
  /// skeleton atoms are pulled back through, and to replay a revoked cover -- and a reachability
  /// query needs none of that until D gains its first atom. On most models it never does, and the
  /// edges were the second-largest block of the run. ::online_rebuild_raw_edges recreates them,
  /// once, when the first atom arrives; from then on every expansion stores its edges as usual.
  bool online_lazy_edges_ = false;
  // What ::online_rebuild_raw_edges needs instead of the edges, kept while ::online_lazy_edges_ is
  // on: per expanded record its id and successor count, then per successor its target key and its
  // tag with the revocable-cover bit, as LEB128 varints relative to the previous record and to the
  // source key. A few bytes an edge against the arena's sixteen, and replaying it regenerates no
  // successor -- which, when the first atom arrives late, is most of what the search has done.
  std::vector<std::vector<std::uint8_t>> online_edge_log_;
  std::size_t online_edge_log_bytes_ = 0;
  int online_edge_log_prev_ = 0;
  void online_log_varint(std::uint64_t x) {
    const std::size_t chunk = std::size_t(1) << 20;
    do {
      if (online_edge_log_.empty() || online_edge_log_.back().size() == chunk) {
        online_edge_log_.emplace_back();
        online_edge_log_.back().reserve(chunk);
      }
      online_edge_log_.back().push_back(static_cast<std::uint8_t>((x & 0x7f) | (x > 0x7f ? 0x80 : 0)));
      ++online_edge_log_bytes_;
      x >>= 7;
    } while (x);
  }
  static std::uint64_t zigzag(std::int64_t x) {
    return (static_cast<std::uint64_t>(x) << 1) ^ static_cast<std::uint64_t>(x >> 63);
  }
  static std::int64_t unzigzag(std::uint64_t x) {
    return static_cast<std::int64_t>(x >> 1) ^ -static_cast<std::int64_t>(x & 1);
  }
  void online_rebuild_raw_edges();
  std::deque<int> online_work_;

  // ---- lazy liveness refinement (HORA_LAZY) --------------------------------------------------
  // A record materialised to discharge a liveness obligation must not be collapsed by the very
  // antichain it was created to escape, or the loop makes no progress.  Pinning is checked in
  // ::LivenessEngine::online_compact_key next to its existing root guard, and is strictly
  // stronger than the forbidden-cover relation below: the record can never be covered at all.
  // Set before ::LivenessEngine::setup on the `E[] psi` online route when HORA_LAZY is on.
  // It turns Extra_LU+ extrapolation ON for a route that normally runs without any, because
  // suspending covers is exactly what the refinement loop does and a_LU/D subsumption is
  // otherwise the ONLY thing making the forward search finite.  Extrapolation restores a bound
  // that does not depend on covering: finitely many extrapolated zones exist per discrete key,
  // so the refinement can at worst walk to the fully inclusion-covered Extra_LU+ graph.
  // Measured to cost nothing in state count on top of a_LU/D (arbiter8 1280/1280,
  // fischer8 17215/17215, fischer6 1342/1342, viking10 3892/3892).
  bool lazy_mode_ = false;
  // The root zone BEFORE extrapolation.  Every valuation the fixpoint certifies is joined to the
  // seed by a chain of genuine ::exact_predecessor steps, so an intermediate value need not lie
  // in its own node's stored zone -- the run reaches it.  The one place extrapolation can invent
  // a witness is the READOUT, where `W[0]` may hold valuations outside the true initial zone.
  // Retained so the verdict can be cut back to it; ::LivenessEngine::build_all extrapolates the
  // root in place, so it cannot be recovered afterwards.
  Dbm lazy_root_exact_;
  // Exact zones of the MATERIALISED records, keyed by record id.
  //
  // Those are the only stored zones on this route that over-approximate: ::lazy_materialise
  // extrapolates them so refinement terminates.  The closure readout needs every point it
  // certifies to be genuinely reachable, so it cuts them back here.  Storing this for every node
  // would be prohibitive -- twolock10 q3 is 206k records at 61x61 entries, ~3 GB -- but the
  // materialised ones are a small minority (900 of 4792 on viking10 q3, and none at all on any
  // benchmark row, where the recurrence criterion leaves the obligation set empty).
  std::unordered_map<int, Dbm> lazy_exact_;
  std::vector<char> online_pinned_;
  // `forbidden_cover(src, ordinal, cover_target)`.  Pinning protects the record; this protects
  // the *decision*, so a later D recheck replaying the same recipe cannot re-establish the cover
  // that was refined away.  Keyed by the recipe, since that is what replay reconstructs.
  std::unordered_set<std::uint64_t> online_forbidden_cover_;
  static std::uint64_t forbid_key(int src, int ordinal) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(src)) << 32) |
           static_cast<std::uint32_t>(ordinal);
  }
  // Set while ::LivenessEngine::lazy_materialise interns a replayed successor, naming the cover
  // it must refuse.  -1 outside that window.
  int online_refuse_cover_ = -1;
  /// @brief Records that must not be used as cover targets while an unsafe SCC is re-explored.
  ///
  /// Herbreteau et al., ACM TOCL 2021, section 5: when the subsumptions inside an unsafe SCC are
  /// removed and its nodes re-explored, new subsumption must be restricted to SAFE or FRESH
  /// nodes.  Otherwise "the grey nodes were subsumed by nodes in their inhabiting black SCCs, and
  /// hence the successors of grey nodes will also be subsumed by the corresponding successors in
  /// the black SCCs" -- the exploration is stopped in one step and falls straight back into the
  /// same bad cycle.  Empty whenever no re-exploration is in progress.
  std::vector<char> online_unsafe_target_;
  /// @brief Set once the forward route starts refining: stop merging zones by convex union.
  ///
  /// A merged node is the union of several post images, so an edge into it is an inclusion cover
  /// and cannot carry a witness cycle.  That is harmless while the graph is only being explored --
  /// and merging is what keeps grc6 q2 at 1.4s instead of 38s -- but once refinement is trying to
  /// make cycles close exactly, re-merging the successors of a freshly materialised node mints a
  /// new subsumption edge for every one that was just discharged, and single4 q1 never finishes.
  bool fwd_refining_ = false;
  /// @brief The forward Buchi route is in use, so node identity is a_LU-equivalence.
  static constexpr bool fwd_route_ = true;
  /// @brief Records whose stored zone is a convex UNION of several post images.
  ///
  /// Stopping new merges when refinement begins is not enough: the ones made during the first
  /// exploration remain, and the re-exploration routes straight back into them, so the witness
  /// cycle still cannot close through a node that is not a single post image.  They are therefore
  /// barred as cover targets while refining, exactly like the nodes of an unsafe SCC.
  std::vector<char> online_merged_;
  std::vector<char> online_inq_;

  // General graph mechanics.
  /// @brief How many records exist, whichever construction is running.
  std::size_t nrec() const {
    if (online_d_) return online_nrec_;
    return node_key_.empty() ? node.size() : node_key_.size();
  }
  /// @brief Node @p v's integer variables, `nints_n()` of them.
  const long* nints(int v) const {
    return node_key_.empty() ? node[v].ints.data() : key_ints(node_key_[v]);
  }
  std::size_t nlocs_n(int v) const {
    return node_key_.empty() ? node[v].locs.size() : static_cast<std::size_t>(key_loc_stride_);
  }
  std::size_t nints_n(int v) const {
    return node_key_.empty() ? node[v].ints.size() : static_cast<std::size_t>(key_int_stride_);
  }
  /// @brief Copy node @p v's discrete half into @p s, leaving its zone alone.
  void nstate(int v, detail::SymState& s) const {
    if (node_key_.empty()) s.locs.assign(node[v].locs.begin(), node[v].locs.end());
    else key_locs_to(node_key_[v], s.locs);
    const long* i = nints(v);
    s.ints.assign(i, i + nints_n(v));
  }
  bool cancelled();
  void setup();
  void build_local_bound_tables();
  void local_bounds(const int* locs, std::vector<std::int32_t>& outL,
                    std::vector<std::int32_t>& outU) const;
  bool holds(const DiscPred& pr, const detail::SymState& s) const;
  bool is_p(int i);
  bool p_discrete(const detail::SymState& s) const;
  Federation p_region(int i);
  Federation p_region_of(const detail::SymState& s) const;
  bool leadsto_witness(const detail::SymState& s, bool no_successor,
                       bool key_may_deadlock);
  bool online_leadsto_witness(int v, bool no_successor);
  bool is_nq(int i);
  void grow();
  int intern(detail::SymState&& s);
  int intern_tag(const GfpTag& t);
  void expand(int u);
  bool time_diverges(const detail::SymState& s) const;
  Federation stop_set(int u);
  bool stop_known_empty(int v);
  bool orig_time_open(const detail::SymState& s) const;
  void solve_greatest_fixpoint(const std::vector<int>& S, std::vector<Federation>& W);
  bool certificate_in(const std::vector<int>& S);
  std::vector<int> accepting_closure();
  void fill_stats(GfpStats* gs);
  bool final_verdict_violated();

  // Stored-zone access and covering.
  void zstore(int i, Dbm&& z);
  const Dbm& zread(int i, Dbm& sc) const;
  const detail::SymState& sread(int i, detail::SymState& sc) const;
  bool alu_le_stored(const Dbm& z, int j) const;
  bool alu_le_new(int j, const Dbm& z) const;
  bool stored_includes(int j, const Dbm& z) const;
  bool includes_stored(const Dbm& z, int j) const;
  bool base_cover(const Dbm& a, const Dbm& b) const;
  void split_by_atoms(detail::SymState&& s, std::vector<detail::SymState>& out);
  void classify_atoms(const Dbm& z, const std::vector<AtomC>& as,
                      std::uint64_t* in, std::uint64_t* out) const;
  bool cover_rec(const Dbm& a, std::uint64_t ain, const Dbm& b, std::uint64_t bout,
                 const std::vector<AtomC>& as, int depth);
  void base_forward();
  bool build_all();

  // Online-D discovery, refinement, and compaction.
  void online_reset();
  void online_configure_bounds(int key);
  bool online_covers(const detail::SymState& a, int b, int key);
  /// @brief Are candidate @p a and stored record @p b equivalent under the D-refined relation?
  ///
  /// Mutual covering, both directions in the SAME relation the construction uses.  The forward
  /// route's node identity must match the abstraction its graph was built with: covering applies
  /// the D atoms, so the graph is ZG^{a_D}, and testing one direction with D and the other with
  /// plain a_LU mixes a fine relation with a coarse one -- too many edges come out ordinary and
  /// the cycle search invents witnesses.
  bool online_equiv(const detail::SymState& a, int b, int key);
  bool online_covers(int a, int b, int key);
  bool online_cover_rec(const Dbm& a, const Dbm& b, const std::vector<AtomC>& as, int depth);
  int online_append(detail::SymState&& s, int key);
  int online_intern(detail::SymState&& s, int key, int source = -1, int ordinal = -1,
                    int* deferred = nullptr, bool* ordinary = nullptr);
  /// @brief Is an edge whose exact successor is @p s, landing on stored record @p j, ORDINARY?
  ///
  /// Ordinary in the sense of src/liveness/forward_buchi.cpp: @p j is a single post image (not a
  /// convex-union merge) and the successor is equivalent to it under the covering relation, in
  /// both directions.  @pre @p s is already covered by @p j, which supplies the first direction.
  bool online_ordinary_to(const detail::SymState& s, int j, int key);
  bool online_replay(int source, int ordinal, int key, detail::SymState& out) const;
  bool online_is_nq(int v);
  bool online_eg_terminal_witness(int v, bool no_restricted_successor);
  /// @brief Greatest-fixpoint witness probe over a read-only snapshot of the online graph.
  /// @return @c true only when a real `E[] psi` witness exists; @c false says nothing.
  /// @see LivenessEngine::online_eg_probe for why a miss is not a verdict.
  /// @param work  if non-null, receives the exact-predecessor computations this probe's
  ///              greatest-fixpoint sweep actually spent -- the probe's real cost, which the
  ///              record count it was handed does not predict.  Zero on every early return.
  bool online_eg_probe(std::size_t* work = nullptr);
  /// @brief Does the explored psi-closure already contain a cycle of ORDINARY edges?
  ///
  /// Herbreteau et al.'s Theorem 6 direction that needs no liveness compatibility: a cycle of
  /// ordinary edges is a path of the abstract zone graph, hence an infinite run, and every record
  /// of the closure is joined to the initial state by a psi-path.  So a hit is a real witness on a
  /// partial graph; a miss says nothing.  No zone operation at all -- the classification was paid
  /// for when each edge was made.
  bool online_eg_cycle_probe();
  /// @brief ::eg_closure read straight off the raw adjacency, with no snapshot installed.
  ///
  /// The same node set: a raw target that is not itself an expanded record stands for its expanded
  /// coverer, exactly as ::lazy_snapshot resolves it.  Order is depth-first from the initial record.
  std::vector<int> online_eg_closure_raw();
  int online_coverer(int v, bool require_expanded);
  int online_resolve_cover(int v, bool require_expanded);
  void online_set_cover(int v, int cover);
  void online_activate(int v);
  void online_compact_key(int v);
  void online_recheck_key(int key);
  void online_recheck_dirty();
  void online_enable_deferred_index();
  bool online_add_diag(int key, const AtomC& atom, bool closure);
  bool online_raise_bump(int clock, std::int32_t bound, bool closure);
  void online_seed_key(int key);
  void online_pullback(const OnlineKeyEdge& e, const AtomC& atom);
  void online_drain_atoms();
  int online_register_key(const detail::LocVec& locs, const detail::IntVec& ints,
                          std::uint64_t packed = 0, bool packed_ready = false);
  /// @brief Decode key @p id's `key_loc_stride_` locations into @p out.
  void key_locs_to(int id, int* out) const {
    const std::size_t base = static_cast<std::size_t>(id) * key_loc_stride_;
    switch (key_loc_width_) {
      case 1: for (int p = 0; p < key_loc_stride_; ++p) out[p] = key_loc8_[base + p]; break;
      case 2: for (int p = 0; p < key_loc_stride_; ++p) out[p] = key_loc16_[base + p]; break;
      default: for (int p = 0; p < key_loc_stride_; ++p) out[p] = key_loc32_[base + p]; break;
    }
  }
  void key_locs_to(int id, detail::LocVec& out) const {
    out.resize(static_cast<std::size_t>(key_loc_stride_));
    key_locs_to(id, out.data());
  }
  /// @brief Is key @p id's location vector exactly @p locs?
  bool key_locs_equal(int id, const detail::LocVec& locs) const {
    if (static_cast<int>(locs.size()) != key_loc_stride_) return false;
    const std::size_t base = static_cast<std::size_t>(id) * key_loc_stride_;
    for (int p = 0; p < key_loc_stride_; ++p) {
      const int l = key_loc_width_ == 1 ? key_loc8_[base + p]
                  : key_loc_width_ == 2 ? key_loc16_[base + p] : key_loc32_[base + p];
      if (l != locs[p]) return false;
    }
    return true;
  }
  void key_locs_append(const detail::LocVec& locs) {
    switch (key_loc_width_) {
      case 1: for (int l : locs) key_loc8_.push_back(static_cast<std::uint8_t>(l)); break;
      case 2: for (int l : locs) key_loc16_.push_back(static_cast<std::uint16_t>(l)); break;
      default: key_loc32_.insert(key_loc32_.end(), locs.begin(), locs.end()); break;
    }
  }
  std::size_t key_locs_bytes() const {
    return key_loc8_.capacity() + key_loc16_.capacity() * 2 + key_loc32_.capacity() * sizeof(int);
  }
  /// @brief Integer variables of key @p id, `key_int_stride_` of them (possibly none).
  const long* key_ints(int id) const {
    return key_int_stride_ ? key_ints_.data() + static_cast<std::size_t>(id) * key_int_stride_
                           : nullptr;
  }
  /// @brief Copy key @p id's discrete state into @p s, leaving its zone alone.
  void key_state(int id, detail::SymState& s) const {
    key_locs_to(id, s.locs);
    const long* i = key_ints(id);
    s.ints.assign(i, i + key_int_stride_);
  }
  int online_reset_class(int tag);
  void online_enable_key_graph();
  void online_register_edge(int src, int tgt, int reset_class);
  void online_expand(int u);
  bool online_build_all();

  // ---- lazy liveness refinement -------------------------------------------------------------
  /// @brief One unresolved covered edge, and what it takes to make it exact.
  struct LazyObligation {
    int src;        ///< source record
    int ordinal;    ///< which successor of @ref src, in ::tagged_successors order
    int slot;       ///< index into `online_raw_adj_`
    int pos;        ///< index of the edge within that raw vector
    int cover;      ///< the record the exact successor was covered by
    int key;        ///< discrete key of both
    int scc;        ///< key-SCC the target recurs in; a live one, by construction
    int depth;      ///< BFS distance of @ref src from node 0 within the psi-closure
  };
  /// @brief Decide `E[] psi` with demand-driven refinement of unresolved covers.
  ///
  /// @param[out] sat set only when the loop reaches a verdict of its own
  /// @return @c false when the budget was spent, in which case @p sat is untouched and the
  ///         caller must fall back to the exact route
  bool lazy_refine_eg(bool* sat);
  /// @brief Install a read-only view of the online graph as `adj` / `node_key_`.
  ///
  /// @param[out] lost  optional: edges the projection had to DROP because their target is a
  ///                   query-relevant record with no expanded representative.  Zero means the
  ///                   view is the whole graph; anything else means it is a truncation, which a
  ///                   cycle search must not read a negative verdict out of.
  /// @return @c false when the online state cannot be projected, which is never expected here
  bool lazy_snapshot(std::vector<std::vector<OtfEdge>>* saved_adj, std::vector<int>* saved_key,
                     std::size_t* lost = nullptr);
  void lazy_restore(std::vector<std::vector<OtfEdge>>* saved_adj, std::vector<int>* saved_key);
  /// @brief Key-projected SCC ids, and each component's size.
  ///
  /// Computed once: the key graph is closed under refinement, because a covered successor's
  /// enabled transitions are a subset of its coverer's and the coverer is expanded, so every key
  /// edge a materialised record can produce is already present.
  /// @param nodes  optional membership filter: only these records contribute keys and edges.
  ///               Restricting it to the accepting closure is sound -- a witness run never leaves
  ///               that set -- and much sharper, because the components come out smaller and both
  ///               conjuncts then have fewer keys to bind and fewer transitions to avoid.
  void lazy_build_key_sccs(std::vector<int>* comp, std::vector<char>* comp_live,
                           const std::vector<char>* nodes = nullptr);
  void lazy_build_key_sccs_csr(std::vector<int>* comp, std::vector<char>* comp_live,
                               const std::vector<char>* nodes);
  /// @brief Collect the unresolved covered edges that can still carry a recurrent witness.
  void lazy_collect(const std::vector<int>& S, const std::vector<Federation>& W,
                    const std::vector<int>& key_comp, const std::vector<char>& comp_live,
                    std::vector<LazyObligation>* out);
  /// @brief Replay one obligation's exact successor and splice it into the graph.
  /// @return @c false on replay failure, which abandons the loop rather than the verdict
  bool lazy_materialise(const LazyObligation& ob);
  /// @brief ::LivenessEngine::online_replay with the source's successor batch cached.
  bool online_replay_cached(int source, int ordinal, int key, detail::SymState& out);
  int lazy_replay_src_ = -1;                       ///< source the cached batch belongs to
  std::vector<detail::SymState> lazy_replay_succ_; ///< its successors, in tagged_successors order
  // ---- forward Buchi route (src/liveness/forward_buchi.cpp) ---------------------------------
  /// @brief Decide `E[] psi` or `p --> q` by cycle detection, with no backward pass.
  ///
  /// Herbreteau/Srivathsan/Tran/Walukiewicz, FSTTCS 2016, Theorem 6: the abstract zone graph has
  /// an infinite accepting path iff a *liveness compatible* subsumption graph has one consisting
  /// of ordinary edges -- where liveness compatible (their C5) means no cycle carries both an
  /// accepting node and a subsumption edge.  For `E[] psi` every node of the psi-closure is
  /// accepting, so C5 reduces to "no cover edge lies on a cycle", and the verdict becomes two
  /// Tarjan runs instead of a greatest fixpoint: no `Pre`, no federations, no zone algebra at all.
  /// `p --> q` is the same search with a different seed.  Its counterexample is
  /// `EF(p /\ EG !q)`, so the accepting set is the `!q` closure of the `p /\ !q` nodes rather
  /// than the psi-closure of the initial vertex, and the answer is negated: a witness there
  /// REFUTES the query.  See ::LivenessEngine::fwd_leadsto_ok for when that is legitimate.
  ///
  /// @param leadsto      seed from every accepting `p /\ !q` node instead of the initial vertex
  /// @param[out] witness set only on a verdict: an infinite or maximal accepting run exists
  /// @return @c false when it declines, and the caller must use the backward route
  bool forward_buchi(bool leadsto, bool* witness);
  /// @brief The `!q` closure of the accepting nodes, as the forward route needs it.
  ///
  /// ::LivenessEngine::accepting_closure answers the same question on the SEALED graph.  This one
  /// runs against the snapshot ::LivenessEngine::lazy_snapshot installs, where a record that lost
  /// its raw slot to compaction has no adjacency of its own and must not be seeded: an empty
  /// adjacency reads as a maximal run to ::LivenessEngine::stop_set, and the node would report a
  /// witness it does not have.  Dropping it loses nothing, because its coverer carries the same
  /// discrete key -- hence the same `p` and `q` -- and a larger zone.
  std::vector<int> fwd_accepting_closure();
  /// @brief May `p --> q` be handed to the forward route at all?
  ///
  /// @return @c false when the verdict needs a zone the forward search does not carry
  bool fwd_leadsto_ok() const;
  /// @brief Emit the per-query refinement census under `HORA_REFINE_STATS`.
  void lazy_report(bool verdict);
  /// @brief Nodes whose backward value can have changed since the last fixpoint pass.
  ///
  /// Every materialised record (its value starts at its own zone, not empty) and the source of
  /// every rewritten edge (its predecessor term gained a case).  Seeds the cone below.
  std::vector<int> lazy_changed_;
  /// @brief Restrict @p S to the nodes that can reach a changed one, changed nodes included.
  ///
  /// Sound because the loop only refines when the readout found EVERY `W[v]` empty, so a node
  /// outside the cone already sits at its own fixpoint value: it cannot reach a change, so none
  /// of its successors is in the cone either, and its whole subsystem is untouched.  Dropping it
  /// from `S` is exactly how ::LivenessEngine::solve_greatest_fixpoint is told "this one is
  /// empty" -- `inS[w] = 0` -- so no change to the solver is needed.
  void lazy_cone(const std::vector<int>& S, std::vector<int>* cone) const;
  LazyStats lazy_;
  void online_drain_forward_work();
  void online_complete_eg_history();
  void online_base_forward();
  void online_finalize_graph();
};

}  // namespace hora
