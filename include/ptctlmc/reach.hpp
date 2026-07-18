// reach.hpp — symbolic zone-graph reachability (the E<> / A[] core).
#pragma once

#include <vector>

#include "ptctlmc/ta.hpp"

namespace ptctlmc {

struct ReachStats {
  std::size_t explored = 0;   // symbolic states added to the passed list
  std::size_t generated = 0;  // successor zones generated
};

enum class Backend { Seq, Omp, Cuda };

// E<> : is a state with location `target_loc` (target_loc < 0 => any location) and
// satisfying the extra clock constraints `target` reachable? Zone-graph exploration
// with passed-list inclusion (subsumption) + max-extrapolation for termination.
// `backend` selects seq / OpenMP level-sync frontier / CUDA — all give identical
// verdicts (Cuda requires a CUDA-enabled build + GPU).
bool reachable(const TimedAutomaton& ta, int target_loc,
               const std::vector<Constraint>& target = {}, ReachStats* stats = nullptr,
               Backend backend = Backend::Seq);

}  // namespace ptctlmc
