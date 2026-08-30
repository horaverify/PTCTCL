// ta.hpp — internal timed-automaton representation.
//
// A location has an invariant; an edge has a guard, a set of clocks to reset, and
// an optional synchronisation channel (networks come later). Clock constraints are
// stored DBM-ready as x_i - x_j <| bound so guards/invariants apply directly to a
// zone. Clock index 0 is the always-zero reference clock.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "hora/zone.hpp"
#include "hdbm/constraint.hpp"  // Constraint, c_le/.., negate — now in HDBM

namespace hora {

struct Location {
  std::string name;
  bool urgent = false;
  bool committed = false;
  std::vector<Constraint> invariant;
};

struct Edge {
  int src = 0, tgt = 0;
  std::vector<Constraint> guard;
  std::vector<int> resets;  // clock indices (1..nclocks) reset to 0
  int sync = -1;            // channel id, -1 = none
};

struct TimedAutomaton {
  int nclocks = 0;
  std::vector<std::string> clock_names;  // index 0 = "0" reference
  std::vector<Location> locations;
  std::vector<Edge> edges;
  int initial = 0;

  std::vector<std::int32_t> max_const;   // per-clock (0..nclocks) max constant (Extra_M)
  std::vector<std::int32_t> lbound;      // per-clock max lower-bound constant (Extra_LU L)
  std::vector<std::int32_t> ubound;      // per-clock max upper-bound constant (Extra_LU U)

  void compute_max_constants();
  std::vector<int> edges_from(int loc) const;
};

}  // namespace hora
