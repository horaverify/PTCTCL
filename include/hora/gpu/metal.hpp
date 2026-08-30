// metal_reach.hpp — Metal GPU zone-graph reachability entry point (macOS).
//
// The Apple-GPU peer of cuda_reach.hpp: same semantics as reachable(), but the DBM
// successor pipeline runs on the Metal GPU (one thread per (frontier-state, edge)),
// while the host does the serial passed-list subsumption merge. No Metal/Obj-C types
// leak here, so CPU translation units may include it; the implementation exists only
// in Metal-enabled builds (src/metal/engine_metal.mm).
#pragma once

#include <cstdint>
#include <vector>

#include "hora/reach.hpp"
#include "hora/automaton.hpp"

namespace hora {

// GPU zone-graph reachability over a diagonal-free automaton `ta` (the caller applies
// diagonal splitting first). `accept[loc]` marks target locations; `tgt_clocks` are the
// target's clock constraints (checked on the zone); `L`/`U` are the precomputed Extra_LU+
// bounds. DBM successor generation runs on the GPU; host does the subsumption merge.
bool reachable_metal(const TimedAutomaton& ta, const std::vector<char>& accept,
                     const std::vector<Constraint>& tgt_clocks,
                     const std::vector<std::int32_t>& L, const std::vector<std::int32_t>& U,
                     ReachStats* stats = nullptr);

}  // namespace hora
