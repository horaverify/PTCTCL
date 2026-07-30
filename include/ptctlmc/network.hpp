// network.hpp — a network of timed automata (parallel composition) with shared
// clocks and shared bounded integer variables. Discrete state = (location vector,
// integer-values vector); zones are over the shared clocks. This is what real
// UPPAAL benchmarks (Fischer, CSMA/CD, ...) need and where state spaces explode.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ptctlmc/expr.hpp"
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

  // General modelling-language guard/update, used only when the edge needs more
  // than `var OP const` / `var := const` -- array elements, arithmetic, or a call
  // (`e == front()`, `enqueue(e)`). Kept SEPARATE from the fast-path vectors
  // above so the existing static analyses (Extra_LU bounds, aLU per-location
  // bounds, active-clock reduction) and the validated Fischer/viking behaviour
  // are untouched: those read int_guard/int_updates, which stay exactly as
  // before for models that do not need this.
  //
  // Semantics: gen_guard is ANDed with int_guard; gen_update runs AFTER
  // int_updates. A faulting guard is false; a faulting update kills the edge.
  ExprP gen_guard;
  StmtP gen_update;
  int gen_locals = 0;   // frame size needed by gen_update
};

struct NProc {
  std::string name;
  std::vector<Location> locations;   // clock invariants + urgent/committed
  std::vector<NEdge> edges;
  int initial = 0;

  // Outgoing edges indexed by source location, built once by build_edge_index().
  // successors_net() is called once per (state, process) and used to call
  // edges_from() there, which allocated a fresh vector and linearly scanned ALL
  // edges every time -- the dominant malloc/scan cost on discrete-heavy models
  // (viking). Reading edges_of(loc) instead is allocation-free.
  std::vector<std::vector<int>> edges_by_loc;
  void build_edge_index();
  const std::vector<int>& edges_of(int loc) const { return edges_by_loc[loc]; }

  std::vector<int> edges_from(int loc) const;  // still used off the hot path
};

struct Network {
  int nclocks = 0;
  std::vector<std::string> clock_names;   // index 0 = "0"
  std::vector<IntVar> intvars;
  std::vector<std::string> channels;
  std::vector<char> chan_broadcast;  // 1 if declared `broadcast chan` (fires with 0+ receivers)
  std::vector<NProc> procs;
  // User functions referenced by NEdge::gen_guard / gen_update. Arrays are
  // flattened into consecutive `intvars` cells, so a function body addresses
  // exactly the same int state the engine already packs.
  std::vector<Func> funcs;

  std::vector<std::int32_t> lbound, ubound;  // global Extra_LU bounds over clocks
  // Per-LOCATION LU bounds (static guard analysis): lbound_loc[p][l][c] = max lower
  // guard constant on clock c reachable from process p's location l without c being
  // reset first; kAluNegInf ("−∞") if c is never guarded from there. These make the
  // aLU subsumption far coarser away from guards — the key to UPPAAL's zone count.
  std::vector<std::vector<std::vector<std::int32_t>>> lbound_loc, ubound_loc;
  // active[p][l][c] = 1 if clock c may be read by process p from location l before
  // being reset (active-clock reduction: inactive clocks are reset to collapse zones).
  std::vector<std::vector<std::vector<char>>> active;
  // Same information as one bit per clock: active_mask[p][l] has bit c set iff active[p][l][c].
  // The active set of a LOCATION VECTOR is then nprocs OR-s instead of an nprocs x nclocks
  // scan, which reset_inactive runs once per generated successor (8.6% of phil10). Empty when
  // nclocks > 63; the caller falls back to the byte form there.
  std::vector<std::vector<std::uint64_t>> active_mask;

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
