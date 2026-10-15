/// @file sym_state.hpp
/// @brief The symbolic state (discrete part + zone) and its pool-backed vector aliases.
#pragma once


#include <cstdint>
#include <utility>
#include <vector>

#include "hora/zone.hpp"

namespace hora {
namespace detail {

// The discrete part of a symbolic state is copied for every successor and again
// for every survivor entering the next frontier. Backing locs/ints with the same
// per-thread pool as the zone (dbm_pool.hpp) turns those millions of tiny
// vector copies from malloc/free pairs — which serialize the OpenMP workers on
// the global allocator lock — into lock-free free-list pops.
using LocVec = std::vector<int, PoolAllocator<int>>;
using IntVec = std::vector<long, PoolAllocator<long>>;

struct SymState {
  LocVec locs;
  IntVec ints;
  Dbm zone;
};


/// @brief What the deadlock-preserving D refinement asks of one discrete key.
///
/// The universe seed of LivenessEngine::online_seed_key, as a pure function of the key so that an
/// engine other than the online construction -- the frontier-batched GPU search -- can read it.
struct DeadlockSeed {
  bool may_deadlock = false;   ///< some valuation of `Inv(q)` can be stuck here
  bool diagonal = false;       ///< a distinguishing atom is diagonal: not expressible as a bound
  /// Non-diagonal atoms as (clock, constant): the clock's L and U bounds must reach the constant.
  std::vector<std::pair<int, std::int32_t>> bumps;
};

}  // namespace detail
}  // namespace hora
