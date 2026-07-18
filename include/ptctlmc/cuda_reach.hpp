// cuda_reach.hpp — GPU zone-graph reachability entry point (no CUDA types leak
// here, so CPU translation units may include it; the implementation exists only in
// CUDA-enabled builds, src/cuda/engine_cuda.cu).
#pragma once

#include <cstdint>
#include <vector>

#include "ptctlmc/reach.hpp"
#include "ptctlmc/ta.hpp"

namespace ptctlmc {

// GPU zone-graph reachability over a DIAGONAL-FREE automaton `ta` (the caller applies
// diagonal splitting first). `accept[loc]` marks target locations; `tgt_clocks` are the
// target's clock constraints; `L`/`U` are the precomputed Extra_LU+ bounds. Successor
// generation runs on the GPU; host does the subsumption merge.
bool reachable_cuda(const TimedAutomaton& ta, const std::vector<char>& accept,
                    const std::vector<Constraint>& tgt_clocks,
                    const std::vector<std::int32_t>& L, const std::vector<std::int32_t>& U,
                    ReachStats* stats = nullptr);

}  // namespace ptctlmc
