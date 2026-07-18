// cuda_reach.hpp — GPU zone-graph reachability entry point (no CUDA types leak
// here, so CPU translation units may include it; the implementation exists only in
// CUDA-enabled builds, src/cuda/engine_cuda.cu).
#pragma once

#include <vector>

#include "ptctlmc/reach.hpp"
#include "ptctlmc/ta.hpp"

namespace ptctlmc {

// Same semantics as reachable(), but successor generation (the DBM pipeline) runs
// on the GPU. Requires nClocks+1 <= PTCTLMC_CUDA_MAX_DIM; throws otherwise.
bool reachable_cuda(const TimedAutomaton& ta, int target_loc,
                    const std::vector<Constraint>& target = {}, ReachStats* stats = nullptr);

}  // namespace ptctlmc
