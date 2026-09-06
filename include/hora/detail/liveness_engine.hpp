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

/// @brief One graph edge: its target record and the transition recipe that produced it.
struct OtfEdge {
  int tgt;
  int tag;
  // Online construction only: a provisional target may be represented by a replay recipe
  // instead of a materialised state.  Final graph edges always carry -1 here.
  int deferred = -1;
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
  void online_drain_forward_work();
  void online_complete_eg_history();
  void online_base_forward();
  void online_finalize_graph();
};

}  // namespace hora
