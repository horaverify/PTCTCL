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

#include "ptctlmc/dbm.hpp"

namespace ptctlmc {

struct Constraint {
  int i, j;      // clocks (0..nclocks); 0 = reference
  raw_t bound;   // x_i - x_j <| bound
};

// Readable constraint builders.
inline Constraint c_le(int x, std::int32_t v) { return {x, 0, le(v)}; }   // x <= v
inline Constraint c_lt(int x, std::int32_t v) { return {x, 0, lt(v)}; }   // x <  v
inline Constraint c_ge(int x, std::int32_t v) { return {0, x, le(-v)}; }  // x >= v
inline Constraint c_gt(int x, std::int32_t v) { return {0, x, lt(-v)}; }  // x >  v
inline Constraint c_diff_le(int x, int y, std::int32_t v) { return {x, y, le(v)}; }  // x - y <= v

// Negation of  x_i - x_j <| b  is  x_j - x_i <| -b  with flipped strictness.
inline Constraint negate(const Constraint& c) {
  const std::int32_t v = raw_bound(c.bound);
  return {c.j, c.i, raw_is_strict(c.bound) ? le(-v) : lt(-v)};
}

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

}  // namespace ptctlmc
