/// @file hora/reachability.hpp
/// @brief Reachability over a network: `E<> phi`, `A[] phi`, and deadlock.
///
/// A symbolic state is a (location vector, integer-values vector, zone) triple; the passed list
/// is keyed by the discrete part and holds an antichain of zones per key. Which abstraction
/// bounds that antichain is chosen in hora/config.hpp.
#pragma once

#include <atomic>
#include <cstddef>
#include "hora/network.hpp"
#include "hora/reach.hpp"

namespace hora {

/// @brief Evaluate an edge's general-expression guard against an integer valuation.
///
/// The `var OP const` and `var := const` fast paths live in hora::NEdge::int_guard and
/// hora::NEdge::int_updates; anything richer — arithmetic, array indexing, function calls —
/// is compiled into hora::NEdge::gen_guard and hora::NEdge::gen_update. Every backend must
/// route through this pair so that the CPU and GPU host paths cannot drift apart.
///
/// @param net  the network owning @p e (supplies the function table)
/// @param e    the edge whose guard is to be evaluated
/// @param ints integer valuation, @p n cells, indexed as hora::Network::intvars
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
/// @pre hora::NEdge::int_updates has already been applied to @p ints.
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
/// @note hora::Backend::Cuda and hora::Backend::Metal require a build configured with that
///       backend and a suitable device.
bool reachable_net(const Network& net, const NetTarget& target, Backend backend = Backend::Seq,
                   ReachStats* stats = nullptr);

// ---- Liveness (location/integer predicates) -------------------------------

}  // namespace hora
