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
/// difference is the whole reason ::LivenessEngine::forward_buchi_eg needs its own marker.
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

/// @brief One discrete key of the online quotient.
///
/// The location and integer vectors are NOT here: they live in @ref LivenessEngine's `key_locs_` /
/// `key_ints_` arenas at a fixed stride, reached through @ref LivenessEngine::key_locs. Two heap vectors per
/// key cost more in headers and allocator metadata than the locations themselves, and on the
/// families where every reachable state has its own key that was the largest single block of
/// memory after the zones.
struct OnlineKeyInfo {
  int id = -1;
  std::uint64_t packed_key = 0;
  // Intrusive lists into LivenessEngine::online_key_edges_.  This avoids three heap-allocated vectors per
  // discrete key while preserving exact incoming propagation and source-local deduplication.
  int incoming = -1;
  int outgoing = -1;
  // Packed `(raw-adjacency slot, edge ordinal)` handles for revocable cover edges at this key.
  // The edge itself is the replay recipe; no separate provisional-state object is allocated.
  std::vector<std::uint64_t> deferred;
  // Materialised and active nodes use intrusive lists stored once per node in LivenessEngine.  Most keys
  // own exactly one record, so a heap-allocated vector per key was mostly allocator metadata.
  int node_head = -1;
  std::size_t node_count = 0;
  int active_head = -1;
  std::size_t active_count = 0;
  std::unique_ptr<std::vector<AtomC>> diag;
  std::unique_ptr<Dbm> inv;
  bool inv_ready = false;
  bool may_deadlock = false;
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
  bool eg_early_hit_ = false;
  // `p --> q`: a reached `p /\ !q` valuation that can end a maximal run is already a
  // counterexample, so the forward search may stop the moment one is found.
  bool leadsto_early_ = false;
  bool online_eg_prune_ = false;
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
  std::vector<int> key_locs_;
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
  std::vector<std::vector<OtfEdge>> online_raw_adj_;
  std::vector<int> online_raw_owner_;
  std::vector<int> online_raw_slot_;
  bool online_deferred_index_enabled_ = false;
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
  /// @brief The exact post images a merged record absorbed, by record id.
  ///
  /// The convex union is set-theoretically exact, so a merged node really does contain each
  /// piece -- merging never invents a run.  What it breaks is the ORDINARY-EDGE TEST, which asks
  /// whether the target's stored zone is a_LU-equivalent to the exact successor: against the
  /// union that fails, the edge is misclassified as a cover, and the cycle search loses it.
  /// Keeping the pieces lets the test ask "equivalent to the node, OR to any piece it absorbed",
  /// which is the same question the unmerged graph would have answered.  Only merged records
  /// appear here, and only on the forward route.
  std::unordered_map<int, std::vector<Dbm>> online_merge_pieces_;
  std::vector<char> online_inq_;

  // General graph mechanics.
  /// @brief How many records exist, whichever construction is running.
  std::size_t nrec() const {
    if (online_d_) return online_nrec_;
    return node_key_.empty() ? node.size() : node_key_.size();
  }
  /// @brief Node @p v's locations, `nlocs_n()` of them.
  const int* nlocs(int v) const {
    return node_key_.empty() ? node[v].locs.data() : key_locs(node_key_[v]);
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
    const int* l = nlocs(v);
    s.locs.assign(l, l + nlocs_n(v));
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
                    int* deferred = nullptr);
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
  /// @brief Locations of key @p id, `key_loc_stride_` of them.
  const int* key_locs(int id) const {
    return key_locs_.data() + static_cast<std::size_t>(id) * key_loc_stride_;
  }
  /// @brief Integer variables of key @p id, `key_int_stride_` of them (possibly none).
  const long* key_ints(int id) const {
    return key_int_stride_ ? key_ints_.data() + static_cast<std::size_t>(id) * key_int_stride_
                           : nullptr;
  }
  /// @brief Copy key @p id's discrete state into @p s, leaving its zone alone.
  void key_state(int id, detail::SymState& s) const {
    const int* l = key_locs(id);
    s.locs.assign(l, l + key_loc_stride_);
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
  /// @return @c false when the online state cannot be projected, which is never expected here
  bool lazy_snapshot(std::vector<std::vector<OtfEdge>>* saved_adj, std::vector<int>* saved_key);
  void lazy_restore(std::vector<std::vector<OtfEdge>>* saved_adj, std::vector<int>* saved_key);
  /// @brief Key-projected SCC ids, and each component's size.
  ///
  /// Computed once: the key graph is closed under refinement, because a covered successor's
  /// enabled transitions are a subset of its coverer's and the coverer is expanded, so every key
  /// edge a materialised record can produce is already present.
  void lazy_build_key_sccs(std::vector<int>* comp, std::vector<char>* comp_live);
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
  // ---- forward-only Buchi prototype (src/liveness/forward_buchi.cpp, HORA_FWD) --------------
  /// @brief Decide `E[] psi` by cycle detection on the forward graph, with no backward pass.
  ///
  /// Herbreteau/Srivathsan/Tran/Walukiewicz, FSTTCS 2016, Theorem 6: the abstract zone graph has
  /// an infinite accepting path iff a *liveness compatible* subsumption graph has one consisting
  /// of ordinary edges -- where liveness compatible (their C5) means no cycle carries both an
  /// accepting node and a subsumption edge.  For `E[] psi` every node of the psi-closure is
  /// accepting, so C5 reduces to "no cover edge lies on a cycle", and the verdict becomes two
  /// Tarjan runs instead of a greatest fixpoint: no `Pre`, no federations, no zone algebra at all.
  /// @param[out] sat set only when the prototype reaches a verdict of its own
  /// @return @c false when it declines, and the caller must use the backward route
  bool forward_buchi_eg(bool* sat);
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
