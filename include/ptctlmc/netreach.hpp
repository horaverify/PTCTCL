// netreach.hpp — E<> reachability over a network of timed automata.
#pragma once

#include "ptctlmc/network.hpp"
#include "ptctlmc/reach.hpp"

namespace ptctlmc {

// E<> : is a discrete/clock state matching `target` reachable? Zone-graph
// exploration with a passed list keyed by discrete state (location vector + int
// values), inclusion subsumption + exact-DBM hash + LU extrapolation.
bool reachable_net(const Network& net, const NetTarget& target, Backend backend = Backend::Seq,
                   ReachStats* stats = nullptr);

// ---- Liveness (location/integer predicates) -------------------------------
// A per-discrete-state predicate: a conjunction of location (in)equalities and
// integer conditions. Clock constraints are NOT allowed here (would need zone
// splitting) — the query layer rejects them as unsupported.
struct LocReq { int proc; int loc; bool equal; };  // equal ? loc==loc : loc!=loc
struct DiscPred {
  std::vector<LocReq> locs;
  std::vector<IntAtom> ints;
};

// E[] phi : does there exist an infinite (time-divergent) run all of whose states
// satisfy `phi`? Detected as a reachable cycle — or a location where time can
// diverge — in the phi-restricted extrapolated zone graph.
bool exists_invariant_run(const Network& net, const DiscPred& phi, Backend backend = Backend::Seq,
                          ReachStats* stats = nullptr);

}  // namespace ptctlmc
