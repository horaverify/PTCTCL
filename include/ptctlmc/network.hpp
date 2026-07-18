// network.hpp — a network of timed automata (parallel composition) with shared
// clocks and shared bounded integer variables. Discrete state = (location vector,
// integer-values vector); zones are over the shared clocks. This is what real
// UPPAAL benchmarks (Fischer, CSMA/CD, ...) need and where state spaces explode.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ptctlmc/ta.hpp"

namespace ptctlmc {

struct IntVar {
  std::string name;
  long lo = 0, hi = 0, init = 0;
};

enum class IntOp { Eq, Ne, Lt, Le, Gt, Ge };
struct IntAtom { int var; IntOp op; long val; };   // intvar OP const
struct IntAsgn { int var; long val; };             // intvar := const

struct NEdge {
  int src = 0, tgt = 0;
  std::vector<Constraint> clock_guard;
  std::vector<IntAtom> int_guard;
  std::vector<int> clock_resets;
  std::vector<IntAsgn> int_updates;
  int sync_chan = -1;   // channel id (-1 none)
  int sync_dir = 0;     // +1 '!', -1 '?'
};

struct NProc {
  std::string name;
  std::vector<Location> locations;   // clock invariants + urgent/committed
  std::vector<NEdge> edges;
  int initial = 0;
  std::vector<int> edges_from(int loc) const;
};

struct Network {
  int nclocks = 0;
  std::vector<std::string> clock_names;   // index 0 = "0"
  std::vector<IntVar> intvars;
  std::vector<std::string> channels;
  std::vector<NProc> procs;

  std::vector<std::int32_t> lbound, ubound;  // global Extra_LU bounds over clocks
  // Per-LOCATION LU bounds (static guard analysis): lbound_loc[p][l][c] = max lower
  // guard constant on clock c reachable from process p's location l without c being
  // reset first; kAluNegInf ("−∞") if c is never guarded from there. These make the
  // aLU subsumption far coarser away from guards — the key to UPPAAL's zone count.
  std::vector<std::vector<std::vector<std::int32_t>>> lbound_loc, ubound_loc;
  // active[p][l][c] = 1 if clock c may be read by process p from location l before
  // being reset (active-clock reduction: inactive clocks are reset to collapse zones).
  std::vector<std::vector<std::vector<char>>> active;

  void compute_bounds();
  void compute_loc_bounds();
  void compute_active();
  int nprocs() const { return static_cast<int>(procs.size()); }

  // resolve "Proc.Loc" -> (proc index, location index); returns false if unknown.
  bool resolve_location(const std::string& proc, const std::string& loc, int* pi, int* li) const;
  int intvar_index(const std::string& name) const;
};

// E<> target: every listed process must be in the given location, all int atoms
// hold, and the zone satisfies the clock constraints.
struct NetTarget {
  std::vector<std::pair<int, int>> loc_req;  // (proc, location)
  std::vector<IntAtom> int_cond;
  std::vector<Constraint> clock_cond;
};

}  // namespace ptctlmc
