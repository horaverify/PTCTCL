// reach.hpp — symbolic zone-graph reachability (the E<> / A[] core).
#pragma once

#include <vector>

#include "hora/automaton.hpp"

namespace hora {

struct ReachStats {
  std::size_t explored = 0;   // symbolic states added to the passed list
  std::size_t generated = 0;  // successor zones generated
  // States actually POPPED AND EXPANDED, as opposed to merely stored.  The two differ by the
  // branching factor -- one expansion interns up to one record per outgoing transition -- and
  // other tools do not all report the same one of them: TChecker's VISITED_STATES is this
  // quantity, while `explored` above is its STORED_STATES.  Reported separately so a comparison
  // can state which it is using instead of mixing them.
  std::size_t expanded = 0;
};

enum class Backend { Seq, Omp, Metal, Cuda };

// E<> : is a state with location `target_loc` (target_loc < 0 => any location) and
// satisfying the extra clock constraints `target` reachable? Zone-graph exploration
// with passed-list inclusion (subsumption) + max-extrapolation for termination.
// `backend` selects seq / OpenMP level-sync frontier — both give identical verdicts.
bool reachable(const TimedAutomaton& ta, int target_loc,
               const std::vector<Constraint>& target = {}, ReachStats* stats = nullptr,
               Backend backend = Backend::Seq);

// Set the OpenMP dynamic-schedule chunk size (default 8).
// Larger values reduce scheduling overhead but coarsen early-exit granularity.
void set_omp_chunk(int chunk);
// Read it back. The network expander (netreach.cpp) needs this too: it used to hard-code
// `schedule(dynamic, 8)`, so --omp-chunk only ever affected the single-TA path in reach.cpp
// and was a silent no-op for every multi-process model.
int omp_chunk();

}  // namespace hora
