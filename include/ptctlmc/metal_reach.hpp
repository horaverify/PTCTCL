// metal_reach.hpp — Metal GPU zone-graph reachability entry point (macOS).
//
// The Apple-GPU peer of cuda_reach.hpp: same semantics as reachable(), but the DBM
// successor pipeline runs on the Metal GPU (one thread per (frontier-state, edge)),
// while the host does the serial passed-list subsumption merge. No Metal/Obj-C types
// leak here, so CPU translation units may include it; the implementation exists only
// in Metal-enabled builds (src/metal/engine_metal.mm).
#pragma once

#include <vector>

#include "ptctlmc/reach.hpp"
#include "ptctlmc/ta.hpp"

namespace ptctlmc {

// Same semantics/verdict as reachable(); DBM successor generation runs on the GPU.
// Rejects diagonal constraints (Extra_LU+ is unsound with diagonals on the GPU path).
bool reachable_metal(const TimedAutomaton& ta, int target_loc,
                     const std::vector<Constraint>& target = {}, ReachStats* stats = nullptr);

}  // namespace ptctlmc
