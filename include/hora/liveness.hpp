/// @file hora/liveness.hpp
/// @brief Liveness over a network: `E[] psi` and `p --> q`.
///
/// Both are decided by one backward greatest fixpoint over a zone graph built once. Zone
/// subsumption is fatal forwards and harmless backwards — a subsumption edge enters the
/// backward operator only as an upper bound — so the fixpoint runs on the fully subsumed
/// graph, at the state count of plain reachability.
///
/// @see docs/specs/ for the correctness arguments, including the early-witness proof.
#pragma once

#include <atomic>
#include <cstddef>
#include <vector>
#include "hora/network.hpp"
#include "hora/predicate.hpp"
#include "hora/reach.hpp"
#include "hora/zone.hpp"
#include "hora/detail/sym_state.hpp"

namespace hora {

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
/// @note Decided by hora::exists_invariant_run_otf, i.e. backward. The forward cycle search
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
///            an arbitrary boolean travels in hora::DiscPred::forbid
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

}  // namespace hora
