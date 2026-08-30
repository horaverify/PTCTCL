/// @file hora/predicate.hpp
/// @brief Predicates over the discrete part of a state, and over a convex clock region.
///
/// The query front end lowers a parsed formula into these; the engines evaluate them. A
/// hora::DiscPred is read as a DISJUNCTION of its terms when `is_dnf` is set, and as a single
/// conjunction otherwise — the two readings are not interchangeable, and which one applies is
/// recorded on the predicate rather than inferred at each use.
#pragma once

#include <vector>
#include "hora/network.hpp"
#include "hora/zone.hpp"

namespace hora {

/// @brief One location atom of a hora::DiscPred.
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
  ///       hora::detail::DelayAvoidSet handles by intersecting per-term results.
  std::vector<DiscTerm> forbid;
  /// @brief Read @ref forbid as the predicate even when it is EMPTY, i.e. as `false`.
  ///
  /// Without this an empty predicate is an empty conjunction, which is `true`. The `E[]` entry
  /// point needs the opposite reading: it passes "where psi's discrete part FAILS", and a psi
  /// with no discrete part fails nowhere. Read as a conjunction that would hold everywhere and
  /// empty `V_psi`, turning every such query into NOT satisfied.
  bool is_dnf = false;

};

}  // namespace hora
