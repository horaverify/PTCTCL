/// @file netreach.hpp
/// @brief Public entry points of the network engine: reachability, deadlock, liveness and the
///        global engine switches that select how the search is performed.
///
/// Everything declared here operates on a ptctlmc::Network — a parallel composition of timed
/// automata over shared clocks and shared bounded integer variables. A symbolic state is a
/// (location vector, integer-values vector, zone) triple; the passed list is keyed by the
/// discrete part and holds an antichain of zones per key.
///
/// The switches in the first half of this header are process-global and are set once, from the
/// command line, before any query runs. They exist because each one changes *which* symbolic
/// states are stored — never a verdict — so they are opt-in per run rather than always on.
///
/// @see DESIGN.md at the repository root for the correctness arguments behind the abstractions
///      used here and for what was measured and rejected.
#pragma once

#include <atomic>

#include "ptctlmc/network.hpp"
#include "ptctlmc/reach.hpp"

namespace ptctlmc {

/// @brief Select how the GPU backends schedule the passed-list merge.
///
/// In the default *naive* mode the merge always runs as one GPU thread per discrete-state
/// bucket. In *adaptive* mode the flat driver calibrates GPU against multi-core CPU merge
/// throughput on the early BFS levels and routes each later level to the predicted-faster path,
/// which recovers models with few but very wide antichains, where one thread per bucket is the
/// wrong shape.
///
/// @param on @c true to enable adaptive routing, @c false for the naive schedule
/// @note No effect on the CPU-only backends (ptctlmc::Backend::Seq, ptctlmc::Backend::Omp).
/// @note Routing changes only the schedule, never the merge result, so the verdict is unaffected.
void set_adaptive_merge(bool on);

/// @brief Report whether adaptive GPU merge routing is enabled.
/// @return the value last passed to ::set_adaptive_merge (default @c false)
bool adaptive_merge();

/// @brief Deduplicate zones in the GPU passed set.
///
/// The GPU driver stores zones densely because its merge kernels consume dense matrices, which
/// costs one full matrix per (discrete state, zone). Interning collapses repeats to a single
/// stored matrix plus a 4-byte index per survivor, so the per-level rebuild copies indices
/// instead of matrices. This is a large win where few distinct zones are reachable and neutral
/// where nearly every zone is distinct, hence opt-in.
///
/// @param on @c true to intern GPU-side zones
/// @note Trades a hash lookup per zone for memory; it does not change which states are stored.
/// @see ptctlmc::detail::GpuZoneIntern
void set_gpu_intern(bool on);

/// @brief Expand each BFS level in bounded batches instead of in one shot.
///
/// With chunking off, a level is expanded whole, so the unified-memory pools must be sized to
/// the widest level and peak memory scales with level width. With it on, the pools are sized to
/// one batch and peak memory becomes independent of level width, raising the model size that
/// fits in RAM.
///
/// @param on     @c true to process the frontier in batches
/// @param states frontier states per batch; @c 0 sizes each batch from the memory budget set by
///               ::set_chunk_budget
/// @note Sound: the frontier is a worklist, so batching only reorders the work. It can shift
///       order-sensitive convex-union counts, never the verdict.
void set_chunking(bool on, int states = 0);

/// @brief Set the GPU pool budget used to size an automatic chunk.
/// @param bytes GPU pool bytes per batch; @c 0 selects the built-in default
/// @note Only consulted when chunking is on and ::set_chunking was given `states == 0`.
void set_chunk_budget(std::size_t bytes);

/// @brief Report whether frontier chunking is enabled.
/// @return the value last passed to ::set_chunking (default @c false)
bool chunking();

/// @brief Does @p net compare two clocks anywhere, i.e. contain a constraint `x_i - x_j ~ c`?
///
/// @param net the network under analysis
/// @return @c true if any guard or invariant in @p net is diagonal
/// @note The LU and a_LU abstractions are unsound in the presence of diagonal constraints
///       (Bouyer, FMSD 24(3), 2004), so a network for which this returns @c true is explored
///       with diagonals kept exact instead.
/// @note Defined once, in netreach.cpp; callers that need only the predicate declare it here.
bool network_has_diagonal(const Network& net);

/// @brief Explore one representative per orbit of the network's automorphism group.
///
/// @param on @c true to enable process-symmetry reduction
/// @warning Exact for the query, but it changes which states are stored, so it is opted into
///          per run rather than enabled by default.
/// @see symmetry.hpp for how the group is computed.
void set_symmetry(bool on);

/// @brief Report whether process-symmetry reduction is enabled.
/// @return the value last passed to ::set_symmetry (default @c false)
bool symmetry_on();

/// @brief Derive integer ranges from an interval analysis instead of from the declarations.
///
/// The discrete key spends `bits(hi - lo + 1)` per integer cell, taken from the declared range.
/// An unannotated `int` in the modelling language declares `[-32768, 32767]`, so a model with a
/// handful of unannotated array cells can overflow the 63-bit key on width alone, before state
/// count enters into it. With this on the range comes from an over-approximating interval
/// analysis of the values a cell can actually hold.
///
/// @param on @c true to narrow ranges before packing the discrete key
/// @see ptctlmc::narrow_int_ranges
void set_narrow_ranges(bool on);

/// @brief Report whether integer-range narrowing is enabled.
/// @return the value last passed to ::set_narrow_ranges (default @c false)
bool narrow_ranges_on();

/// @brief Canonicalise dead integer cells before computing the discrete key.
///
/// An integer cell that no process can read before overwriting, and that the query does not
/// mention, cannot influence any run from the current state, so two states differing only in it
/// have the same future. Such cells are set to their low value before the key is computed, which
/// merges those states. This is the discrete analogue of the active-clock reduction already
/// applied to zones.
///
/// @param on @c true to canonicalise dead cells
/// @warning Changes which states are stored; off by default.
/// @see ptctlmc::Network::iread for the liveness analysis this consumes.
void set_slice(bool on);

/// @brief Report whether the discrete liveness reduction is enabled.
/// @return the value last passed to ::set_slice (default @c false)
bool slice_on();

/// @brief Subsume with the diagonal-free G-simulation instead of a_LU.
///
/// The G-simulation is the a_LU test carried out on bounds that retain each guard's strictness
/// rather than only its constant. For a diagonal-free constraint map the relation collapses to
/// at most one upper and one lower bound per clock (Gastin–Mukherjee–Srivathsan, CAV 2019), so
/// the single strictness bit per bound is the entire difference from a_LU here.
///
/// @param on @c true to use the G-simulation
/// @warning Changes which states are stored; off by default.
/// @see ptctlmc::Network::lraw_loc, ptctlmc::Network::uraw_loc
void set_gsim(bool on);

/// @brief Store liveness zones as minimal graphs instead of dense matrices.
///
/// Far smaller resident, at one `mindbm_unpack` per read. Exact either way — the codec round
/// trips a canonical DBM bit for bit — so this only trades memory against time. Whether it pays
/// depends on the model's read rate; see ptctlmc::ZoneStore.
void set_packed_zones(bool on);

/// @brief Report whether G-simulation subsumption is enabled.
/// @return the value last passed to ::set_gsim (default @c false)
bool gsim_on();

/// @brief Is minimal-graph zone storage in force? @see set_packed_zones
bool packed_zones_on();

/// @brief Memoise the successor-zone pipeline.
///
/// A successor zone is a function of (source zone, edge, target-location data), and on models
/// with few distinct zones the same O(dim^3) computation recurs enormously often. The cache is
/// adaptive: a sliding window disables it when the hit rate is low, so it costs nothing where it
/// does not pay.
///
/// @param on @c true to enable the cache
/// @note A pure cache: it changes neither the stored states nor the verdict.
void set_memo(bool on);

/// @brief Report whether successor-zone memoisation is enabled.
/// @return the value last passed to ::set_memo (default @c false)
bool memo_on();

// ---- general modelling-language guard/update -------------------------------

/// @brief Evaluate an edge's general-expression guard against an integer valuation.
///
/// The `var OP const` and `var := const` fast paths live in ptctlmc::NEdge::int_guard and
/// ptctlmc::NEdge::int_updates; anything richer — arithmetic, array indexing, function calls —
/// is compiled into ptctlmc::NEdge::gen_guard and ptctlmc::NEdge::gen_update. Every backend must
/// route through this pair so that the CPU and GPU host paths cannot drift apart.
///
/// @param net  the network owning @p e (supplies the function table)
/// @param e    the edge whose guard is to be evaluated
/// @param ints integer valuation, @p n cells, indexed as ptctlmc::Network::intvars
/// @param n    number of cells in @p ints
/// @return @c true if the guard holds, or if @p e carries no general guard
/// @post @p ints is unmodified: a guard may call a function that writes, so it runs on a scratch
///       copy.
/// @note Taken as pointer plus length rather than a vector so that the pool-allocated integer
///       vector of the CPU path and the plain vector of the GPU host loop can both call it.
bool gen_ok(const Network& net, const NEdge& e, const long* ints, std::size_t n);

/// @brief Apply an edge's general-expression update to an integer valuation, in place.
///
/// @param net  the network owning @p e (supplies the function table)
/// @param e    the edge whose update is to be run
/// @param[in,out] ints integer valuation, @p n cells
/// @param n    number of cells in @p ints
/// @return @c true on success, or if @p e carries no general update (in which case @p ints is
///         untouched); @c false if evaluation faulted — bad index, division by zero, runaway
///         loop — or if a write left a variable's declared range
/// @pre ptctlmc::NEdge::int_updates has already been applied to @p ints.
/// @post On a @c false return the transition must be discarded; @p ints may have been partially
///       written and is not a valid successor valuation.
bool gen_apply(const Network& net, const NEdge& e, long* ints, std::size_t n);

/// @brief `E<> target` — is a symbolic state matching @p target reachable in @p net?
///
/// Zone-graph exploration with a passed list keyed by the discrete state (location vector plus
/// integer values), combining inclusion subsumption, an exact-DBM hash and LU extrapolation for
/// termination.
///
/// @param net     the network under analysis
/// @param target  the locations, integer conditions and clock constraints sought
/// @param backend which engine performs the search; all backends yield the same verdict
/// @param[out] stats optional exploration counters; may be null
/// @return @c true if some reachable symbolic state matches @p target
/// @note ptctlmc::Backend::Cuda and ptctlmc::Backend::Metal require a build configured with that
///       backend and a suitable device.
bool reachable_net(const Network& net, const NetTarget& target, Backend backend = Backend::Seq,
                   ReachStats* stats = nullptr);

// ---- Liveness (location/integer predicates) -------------------------------

/// @brief One location atom of a ptctlmc::DiscPred.
///
/// Requires process @ref proc to be in location @ref loc when @ref equal is @c true, and to be
/// anywhere else when it is @c false.
struct LocReq { int proc; int loc; bool equal; };

/// @brief A predicate over the discrete part of a state alone.
///
/// The conjunction of every atom in @ref locs and @ref ints. Because it never mentions a clock,
/// it partitions the state space along discrete-state boundaries, and the engines can test it
/// per node without splitting any zone.
///
/// @invariant Clock constraints are not representable here. The query layer rejects a liveness
///            formula whose operands mention clocks as unsupported rather than approximating it.


namespace detail { struct DelayAvoidSet; }   // include detail/net_semantics.hpp to use one

/// @brief One conjunctive term: locations, integers and a convex clock region.
/// @see DiscPred::forbid
struct DiscTerm {
  std::vector<LocReq> locs;
  std::vector<IntAtom> ints;
  std::vector<Constraint> clocks;
};

struct DiscPred {
  std::vector<LocReq> locs;    ///< location atoms, all of which must hold
  std::vector<IntAtom> ints;   ///< integer atoms, all of which must hold
  /// @brief Clock atoms, all of which must hold; empty for a purely discrete predicate.
  ///
  /// A clock atom does not partition the vertices of the zone graph the way a location or
  /// integer atom does: one stored zone can hold valuations on both sides of it. It is handled
  /// by restricting the zones rather than by filtering the vertices — the conjunction is
  /// adjoined to every location invariant, which is exact because a conjunction of constraints
  /// is convex and delay inside a convex region is `(Z /\ C)^ /\ C`. Intersecting only AFTER
  /// the delay would be unsound: for `Z = {x=0}` and this predicate `x>=5`, it yields
  /// `[5,inf)` where the true answer is empty, since every run leaving `x=0` passes through
  /// valuations violating the predicate.
  ///
  /// @note Because the atoms become invariants, the LU bounds and the active-clock sets are
  ///       recomputed from them rather than adjusted by hand; see ::exists_invariant_run.
  std::vector<Constraint> clocks;
  /// @brief The negation of the predicate, in disjunctive normal form; empty when unused.
  ///
  /// A predicate that is a plain conjunction rides on @ref locs, @ref ints and @ref clocks, which
  /// is the cheap path: the clock part is convex and becomes a location invariant. Everything
  /// else -- a disjunction, a negation, or any mix -- is carried here instead, as `!psi` put into
  /// DNF. Each term is a conjunction, so each is a discrete guard plus a **convex** clock region,
  /// and the run must avoid every term whose guard holds where it is.
  ///
  /// @note This is the general representation and the other three fields are then empty. It is
  ///       what lets the engine take an arbitrary boolean combination without ever representing a
  ///       non-convex set as a predicate: the non-convexity lives in the union of terms, which
  ///       ptctlmc::detail::DelayAvoidSet handles by intersecting per-term results.
  std::vector<DiscTerm> forbid;
  /// @brief Read @ref forbid as the predicate even when it is EMPTY, i.e. as `false`.
  ///
  /// Without this an empty predicate is an empty conjunction, which is `true`. The `E[]` entry
  /// point needs the opposite reading: it passes "where psi's discrete part FAILS", and a psi
  /// with no discrete part fails nowhere. Read as a conjunction that would hold everywhere and
  /// empty `V_psi`, turning every such query into NOT satisfied.
  bool is_dnf = false;

};

/// @brief `E[] phi` — does @p net have an infinite, time-divergent run all of whose states
///        satisfy @p phi?
///
/// Decided on the @p phi-restricted extrapolated zone graph, by finding either a reachable cycle
/// or a reachable location in which time can diverge.
///
/// @param net the network under analysis
/// @param phi the invariant the run must maintain
/// @param backend requested backend
/// @param[out] stats optional exploration counters; may be null
/// @return @c true if such a run exists
/// @note The search is sequential; @p backend is accepted for interface uniformity and ignored.
/// @brief `E[] psi` where `psi` is a boolean over discrete atoms conjoined with a clock
///        conjunction.
/// @param net         the network under analysis
/// @param psi_clocks  `psi`'s clock conjunction; adjoined to every location invariant
/// @param nq          holds exactly where `psi`'s discrete part fails
/// @param[out] stats  optional counters; may be null
/// @return @c true iff the initial state admits a maximal `psi`-execution
/// @note Decided by ptctlmc::exists_invariant_run_otf, i.e. backward. The forward cycle search
///       that used to live here was unsound on a subsumed graph; see KNOWN_LIMITATION.md.
bool exists_invariant_run(const Network& net, const std::vector<Constraint>& psi_clocks,
                          const DiscPred& nq, ReachStats* stats = nullptr,
                          const std::vector<DiscTerm>* forbid = nullptr);

// ---- p --> q  (leads-to) --------------------------------------------------

/// @brief Instrumentation for the backward greatest-fixpoint liveness engine.
///
/// Counters only. Nothing here participates in deciding a query, with the sole exception of
/// @ref capped, which qualifies the verdict.
///
/// @warning If @ref capped is set, the returned verdict is unusable and must not be reported;
///          see ::leads_to_otf.
struct GfpStats {
  std::size_t fwd_nodes = 0;       ///< symbolic states stored by the forward search
  std::size_t disc_keys = 0;       ///< distinct discrete states behind them
  std::size_t fwd_edges = 0;       ///< edges of the forward zone graph
  std::size_t iters = 0;           ///< backward fixpoint rounds, summed over all passes
  std::size_t fed_max = 0;         ///< widest converged federation, in zones
  double      fed_avg = 0;         ///< mean width of the non-empty converged federations
  std::size_t nonempty = 0;        ///< nodes whose converged federation is non-empty
  bool        capped = false;      ///< federation-width cap hit: the verdict is UNUSABLE
  std::size_t expanded_nodes = 0;  ///< successor computations actually performed
  std::size_t passes = 0;          ///< confined backward passes run
  std::size_t pre_calls = 0;       ///< exact predecessor computations
  bool        early = false;       ///< settled before the forward search finished
  bool        inconclusive = false;  ///< reserved; never written by the current engine
  std::size_t oracle_nodes = 0;      ///< reserved; never written by the current engine
  /// @brief States generated and expanded by the outer forward search.
  ///
  /// Counted so as to be comparable like for like with UPPAAL's verifyta, which reports states
  /// *generated* rather than states stored and therefore excludes the initial state.
  std::size_t outer_nodes = 0, outer_expands = 0;
};

/// @brief `p --> q` — does every reachable state satisfying @p p inevitably reach one satisfying
///        @p q on all maximal runs?
///
/// Solved as the negation `E<> (p /\ EG !q)`: the reachable zone graph is built forward while a
/// backward greatest fixpoint runs over it, so a violated query can settle without the whole
/// graph ever being built.
///
/// @param net the network under analysis
/// @param p   the antecedent predicate
/// @param q   the consequent predicate
/// @param[out] gs optional instrumentation; may be null
/// @return @c true if the property holds
/// @post If @p gs is non-null and `gs->capped` is set, the return value is meaningless and the
///       query must be reported as unresolved rather than answered.
/// @see netliveness.cpp for the engine, and DESIGN.md for why a partial backward pass is already
///      a certificate.
/// @brief `E[] psi` from the initial state, on the backward greatest-fixpoint engine.
///
/// @param net the network, already restricted by `psi`'s clock conjunction if it had one
/// @param nq  holds exactly where `psi`'s DISCRETE part fails, so `V_psi` is its complement;
///            an arbitrary boolean travels in ptctlmc::DiscPred::forbid
/// @param[out] gs statistics; may be null
/// @return @c true iff the initial state admits a maximal execution all of whose states satisfy
///         `psi`
///
/// @note Replaces the forward cycle search this header used to expose. Subsumption can redirect
///       a successor to a strictly larger covering zone and close a cycle no concrete run
///       realises, so forward cycle detection on the stored graph is unsound (Laarman 2013,
///       Tripakis 2009) — the same reason `p --> q` is evaluated backward.
/// @param witness_only  believe only a witness. The deadlock-freedom proof and the exact lane
///        exist to turn "no witness under a simulation" into a verdict; a caller that reads a
///        @c false as "not proved here" rather than as `NOT satisfied` needs neither, and paying
///        for them is pure loss (they cost a full reachability over @p unrestricted).
bool exists_invariant_run_otf(const Network& net, const DiscPred& nq, GfpStats* gs = nullptr,
                              const Network* unrestricted = nullptr,
                              const detail::DelayAvoidSet* avoid = nullptr,
                              bool witness_only = false);

/// @brief `p --> q` where either operand constrains clocks or carries boolean structure.
/// @param p   the antecedent as a positive DNF; evaluated at one valuation under `EF`
/// @param nq  `q` itself, as the region the run must avoid wherever its discrete guard holds
bool leads_to_clocks(const Network& net, const DiscPred& p, const DiscPred& nq,
                     GfpStats* gs = nullptr);

bool leads_to_otf(const Network& net, const DiscPred& p, const DiscPred& q,
                  GfpStats* gs = nullptr);

/// @brief `E<> deadlock` — is a deadlock state reachable in @p net?
///
/// A symbolic state is a deadlock when it has no action successor: no edge, including any
/// synchronisation, is enabled anywhere in its delay-closed zone, so the network can only let
/// time pass with nothing left to do. `A[] not deadlock` is the negation of this.
///
/// @param net the network under analysis
/// @param backend requested backend
/// @param[out] stats optional exploration counters; may be null
/// @param cancel optional flag polled during the search; when it becomes @c true the search
///        stops early. May be null.
/// @return @c true if a deadlock state is reachable
/// @warning A cancelled search returns @c false, which is indistinguishable from "no deadlock
///          found". The caller owns @p cancel and must decide whether the result is meaningful.
/// @note Explored on the exact-hash LU-extrapolated zone graph. Extrapolation preserves guard
///       enabledness — guards only compare clocks to constants at most the maximum — so a state
///       deadlocks exactly when its extrapolated image does. The completion that makes this hold
///       for the subsumption test as well is described in DESIGN.md.
bool deadlock_reachable(const Network& net, Backend backend = Backend::Seq,
                        ReachStats* stats = nullptr,
                        const std::atomic<bool>* cancel = nullptr);

}  // namespace ptctlmc
