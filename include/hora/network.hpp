// network.hpp — a network of timed automata (parallel composition) with shared
// clocks and shared bounded integer variables. Discrete state = (location vector,
// integer-values vector); zones are over the shared clocks. This is what real
// UPPAAL benchmarks (Fischer, CSMA/CD, ...) need and where state spaces explode.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "hora/expression.hpp"
#include "hora/automaton.hpp"

namespace hora {

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
  // (`e == front()`, `enqueue(e)`). Kept separate from the fast-path vectors
  // above so the existing static analyses (Extra_LU bounds, aLU per-location
  // bounds, active-clock reduction) and the validated Fischer/viking behaviour
  // are untouched: those read int_guard/int_updates, which stay exactly as
  // before for models that do not need this.
  //
  // Semantics: gen_guard is ANDed with int_guard; gen_update runs after
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
  // edges_from() there, which allocated a fresh vector and linearly scanned all
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
  // 1 if declared `urgent chan`: time may not pass in any state where a synchronisation on this
  // channel is enabled. Unlike an urgent location, whose urgency is a property of the location
  // vector alone, this is a property of the whole discrete state -- which processes sit where AND
  // the integer valuation, since the participants' integer guards decide enabledness.
  //
  // It is NOT a property of the zone, and that is a language restriction rather than an accident:
  // an edge synchronising on an urgent channel may not carry a clock guard (the same restriction
  // UPPAAL imposes). load_uppaal_network rejects a model that breaks it. Were clock guards
  // allowed, enabledness could switch on partway through a delay and the delay would have to be
  // clipped at that instant, which is a zone subtraction with a non-convex result. See
  // docs/limitations.md.
  std::vector<char> chan_urgent;
  // True iff any entry of `chan_urgent` is set. Every urgency test is gated on this first, so a
  // network without urgent channels runs exactly the code it ran before they existed.
  bool has_urgent_chan = false;
  std::vector<NProc> procs;
  // User functions referenced by NEdge::gen_guard / gen_update. Arrays are
  // flattened into consecutive `intvars` cells, so a function body addresses
  // exactly the same int state the engine already packs.
  std::vector<Func> funcs;

  std::vector<std::int32_t> lbound, ubound;  // global Extra_LU bounds over clocks
  // Per-location LU bounds (static guard analysis): lbound_loc[p][l][c] = max lower
  // guard constant on clock c reachable from process p's location l without c being
  // reset first; kAluNegInf ("−∞") if c is never guarded from there. These make the
  // aLU subsumption far coarser away from guards — the key to UPPAAL's zone count.
  std::vector<std::vector<std::vector<std::int32_t>>> lbound_loc, ubound_loc;
  // The same per-location bounds as raw bounds, i.e. carrying each guard's strictness rather
  // than only its constant. This is the diagonal-free G(q) of the G-simulation: by Cor. 3.6/3.10
  // a diagonal-free constraint map collapses to at most one upper and one lower bound per clock,
  // so G adds exactly one bit per bound over L/U -- which is what dbm_is_g_le consumes.
  // kGNone marks "no constraint on this clock in this direction".
  std::vector<std::vector<std::vector<raw_t>>> lraw_loc, uraw_loc;
  // active[p][l][c] = 1 if clock c may be read by process p from location l before
  // being reset (active-clock reduction: inactive clocks are reset to collapse zones).
  std::vector<std::vector<std::vector<char>>> active;
  // Same information as one bit per clock: active_mask[p][l] has bit c set iff active[p][l][c].
  // The active set of a location vector is then nprocs OR-s instead of an nprocs x nclocks
  // scan, which reset_inactive runs once per generated successor (8.6% of phil10). Empty when
  // nclocks > 63; the caller falls back to the byte form there.
  std::vector<std::vector<std::uint64_t>> active_mask;

  void compute_bounds();
  void compute_loc_bounds();
  void compute_active();
  // iread[p][l][v] = 1 if process p, starting at location l, may read integer cell v before p
  // itself writes it. The discrete analogue of `active` for clocks, and computed the same way:
  // a per-process backwards fixpoint, no product construction.
  //
  // A cell is dead at a location vector L when iread[p][L[p]][v] is 0 for every process p (and
  // the query does not read it). Two states differing only in a dead cell have the same future:
  // any read of v is performed by some process q from L[q], and every path of q from there
  // writes v before reading it, so no read anywhere in any run observes the entry value. The
  // universal quantifier is what makes that argument work -- an existential one would not.
  // Dead cells are therefore canonicalised to `lo` before the discrete key is computed, which
  // merges states that differ only in unobservable data.
  //
  // Conservative where it must be: any cell mentioned by a general expression guard/update
  // (arrays, arithmetic, calls) is treated as read everywhere, since those are not analysed.
  std::vector<std::vector<std::vector<char>>> iread;
  std::vector<std::vector<std::uint64_t>> iread_mask;   // bit form, when intvars.size() <= 63
  void compute_int_live();
  int nprocs() const { return static_cast<int>(procs.size()); }

  // resolve "Proc.Loc" -> (proc index, location index); returns false if unknown.
  bool resolve_location(const std::string& proc, const std::string& loc, int* pi, int* li) const;
  int intvar_index(const std::string& name) const;
};

// E<> target: every listed process must be in the given location, all int atoms
// hold, and the zone satisfies the clock constraints.
// Replace each integer variable's declared range by an over-approximation of the values it can
// actually hold (src/intrange.cpp). The discrete key spends bits(hi-lo+1) per cell, and UPPAAL's
// default range for an undeclared `int` is [-32768,32767], so a model with a few unannotated
// array cells overflows the 63-bit key on width alone rather than on state count. Returns true
// when at least one variable narrowed. over-approximates: no reachable value is ever excluded,
// which is what keeps it sound against gen_apply's range check.
bool narrow_int_ranges(Network& net);

struct NetTarget {
  std::vector<std::pair<int, int>> loc_req;  // (proc, location)
  std::vector<IntAtom> int_cond;
  std::vector<Constraint> clock_cond;
};

/// @brief Does @p net compare two clocks anywhere, i.e. contain a constraint `x_i - x_j ~ c`?
///
/// @param net the network under analysis
/// @return @c true if any guard or invariant in @p net is diagonal
/// @note The LU and a_LU abstractions are unsound in the presence of diagonal constraints
///       (Bouyer, FMSD 24(3), 2004), so a network for which this returns @c true is explored
///       with diagonals kept exact instead.
/// @note Defined once, in netreach.cpp; callers that need only the predicate declare it here.
bool network_has_diagonal(const Network& net);

}  // namespace hora
