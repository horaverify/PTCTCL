/// @file src/liveness/internal.cpp
/// @brief Definitions of the names shared between the liveness translation units.
///
/// @see src/liveness/internal.hpp for what is shared and why.

#include "hora/zone.hpp"
#include "hora/detail/liveness_engine.hpp"
#ifdef HORA_HAVE_OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#include "hdbm/mindbm.hpp"
#include "hora/detail/semantics.hpp"
#if defined(__APPLE__)
#endif
#include "src/liveness/internal.hpp"

namespace hora {

/// @brief Resident set in MiB, for the HORA_RSS phase report; 0 when it is not asked for.
std::size_t rss_mb() {
  static const bool on = std::getenv("HORA_RSS") != nullptr;
  if (!on) return 0;                       // do not pay a syscall for a diagnostic nobody reads
#if defined(__APPLE__)
  ::mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
    return 0;
  return info.resident_size >> 20;
#else
  return 0;
#endif
}

/// @brief Prove that the deadlock refinement is globally empty before exploration.
///
/// With no source of urgency and no invariant anchoring a clock above to clock zero, every
/// valuation can delay forever: lower and difference constraints are preserved by a uniform
/// delay.  Consequently no discrete key has a partial deadlock region, so D is exactly a_LU.
/// This is a proof, not a fallback; it removes all revocable-cover bookkeeping on the common
/// time-open case while retaining the same one forward graph and one backward solve.
bool structurally_empty_d(const Network& net) {
  bool time_open = !net.has_urgent_chan;
  bool one_key = true;
  for (const NProc& proc : net.procs)
    for (const Location& loc : proc.locations) {
      if (loc.urgent || loc.committed) time_open = false;
      for (const Constraint& c : loc.invariant)
        if (c.i > 0 && c.j == 0 && c.bound < kInf) time_open = false;
    }
  for (const NProc& proc : net.procs) one_key = one_key && proc.locations.size() == 1;
  for (const IntVar& var : net.intvars) one_key = one_key && var.lo == var.hi;
  if (time_open) return true;
  if (!one_key) return false;

  // With exactly one possible discrete key, an empty seed cannot acquire an atom by pullback:
  // every transition is a self-loop on that same empty set.  Evaluate that one seed exactly.
  SymState state;
  state.locs.resize(net.nprocs());
  for (int p = 0; p < net.nprocs(); ++p) state.locs[p] = net.procs[p].initial;
  state.ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) state.ints[v] = net.intvars[v].init;
  state.zone = Dbm::universe(net.nclocks);
  Dbm inv = Dbm::universe(net.nclocks);
  bool committed, urgent;
  frozen_flags(net, state.locs, state.ints, &committed, &urgent);
  std::vector<Constraint> constraints;
  all_invariants(net, state.locs, constraints);
  if (!committed && !urgent) inv.delay();
  if (!apply_all(inv, constraints)) return true;

  Federation stuck, escapes;
  if (!state_has_deadlock(net, state, &stuck, &escapes)) return true;
  if (!committed && !urgent) {
    bool open = true;
    for (int c = 1; c <= net.nclocks; ++c)
      if (inv.get(c, 0) != kInf) { open = false; break; }
    if (open) return true;
  }
  for (const Dbm& escape : escapes) {
    Dbm reachable = escape;
    reachable.intersect(inv);
    if (!reachable.empty()) return false;  // partial Dead(key): D really has a seed
  }
  return true;                              // Dead(key) is the whole invariant
}

OnlinePerf online_perf;

}  // namespace hora
