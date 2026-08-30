/// @file src/liveness/exists_invariant_run.cpp
/// @brief `E[] psi`: restrict the network by psi, then hand it to the liveness engine.
///
/// The clock part of psi folds into the location invariants where it is convex, and is carried
/// as a forbidden region where it is not. What remains is a discrete predicate the engine can
/// filter vertices with.

#include "hora/zone.hpp"
#include "hora/config.hpp"
#include "hora/liveness.hpp"
#include "hora/reachability.hpp"
#include "hora/symmetry.hpp"
#include <cstdio>
#include <sys/resource.h>
#include <array>
#include <chrono>
#ifdef __APPLE__
#include <malloc/malloc.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif
#ifdef __linux__
#include <malloc.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <queue>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "hdbm/dbm_ops_legacy.hpp"
#include "hora/detail/engine_config.hpp"
#include "hora/detail/spin_rw.hpp"
#include "hora/detail/sym_state.hpp"
#include "hora/detail/zone_intern.hpp"
#include "hora/detail/gpu_zone_intern.hpp"
#include "hora/detail/passed_set.hpp"
#include "hora/detail/semantics.hpp"
#include "hdbm/mindbm.hpp"   // minimal-graph packed zone storage (UDBM mingraph port)
#include "hora/gpu/network.hpp"
#ifdef HORA_HAVE_OPENMP
#include <omp.h>
#endif
#ifdef __linux__
#if defined(__GLIBC__)
#endif
#endif

namespace hora {

using detail::apply_all;
using detail::SymState;

bool exists_invariant_run(const Network& net_in, const std::vector<Constraint>& psi_clocks,
                          const DiscPred& nq, ReachStats* stats,
                          const std::vector<DiscTerm>* forbid) {
  // The psi-restricted network: psi's clock conjunction adjoined to every location invariant.
  // Convex, so this is exact -- delay inside a convex C is (Z /\ C)^ /\ C -- and it makes the
  // two side conditions automatic rather than hand-maintained: compute_bounds derives the LU
  // bounds from it, so no extrapolation merges zones across the boundary, and compute_active sees
  // the clocks read at every location, so the reduction cannot zero them.
  Network restricted;
  if (!psi_clocks.empty()) {
    restricted = net_in;
    for (auto& proc : restricted.procs)
      for (auto& loc : proc.locations)
        loc.invariant.insert(loc.invariant.end(), psi_clocks.begin(), psi_clocks.end());
    restricted.compute_bounds();
    restricted.compute_loc_bounds();
    restricted.compute_active();
    restricted.compute_int_live();
  }
  const Network& net = psi_clocks.empty() ? net_in : restricted;

  // The non-convex part of psi, if any: terms of `!psi` that constrain clocks. Each is a
  // conjunction, hence a convex region with a discrete guard, and the run must avoid every term
  // whose guard holds. Empty for a psi whose clock part is a plain conjunction -- that one is
  // already in the invariants above and costs nothing here.
  detail::DelayAvoidSet av;
  std::vector<DiscPred> guards;
  if (forbid && !forbid->empty()) {
    guards.reserve(forbid->size());
    for (const auto& t : *forbid) {
      if (t.clocks.empty()) continue;             // purely discrete: handled by the vertex filter
      detail::DelayAvoid one;
      one.forbid = Dbm::universe(net.nclocks);
      if (!apply_all(one.forbid, t.clocks) || one.forbid.empty()) continue;
      one.fup = one.forbid; one.fup.delay();
      guards.push_back(DiscPred{});
      guards.back().locs = t.locs;
      guards.back().ints = t.ints;
      av.parts.push_back(std::move(one));
    }
    for (std::size_t i = 0; i < av.parts.size(); ++i)     // bind after the vector stops growing
      av.parts[i].when = (guards[i].locs.empty() && guards[i].ints.empty()) ? nullptr : &guards[i];
  }

  // The avoided region's constants must reach the LU bounds, and its clocks must stay live.
  //
  // For a convex psi this is automatic: the conjunction became a location invariant above, so
  // compute_bounds derived the bounds from it and compute_active saw the clocks read everywhere. A
  // non-convex psi has no such invariant -- the region is routed around, not conjoined -- so
  // nothing tells the abstraction about it, and a_LU subsumption or Extra_LU+ then merges zones
  // lying on opposite sides of the very boundary the region draws.
  Network bounded;
  const Network* engine_net = &net;
  if (!av.parts.empty()) {
    bounded = net;
    for (const auto& t : *forbid) {
      for (const auto& c : t.clocks) {
        const std::int32_t k = raw_bound(c.bound) < 0 ? -raw_bound(c.bound) : raw_bound(c.bound);
        auto bump = [&](int idx, bool lo, bool hi) {
          if (idx <= 0 || idx >= (int)bounded.lbound.size()) return;
          if (lo) bounded.lbound[idx] = std::max(bounded.lbound[idx], k);
          if (hi) bounded.ubound[idx] = std::max(bounded.ubound[idx], k);
          for (auto& pl : bounded.lbound_loc) for (auto& l : pl)
            if (idx < (int)l.size() && lo) l[idx] = std::max(l[idx], k);
          for (auto& pu : bounded.ubound_loc) for (auto& l : pu)
            if (idx < (int)l.size() && hi) l[idx] = std::max(l[idx], k);
        };
        if (c.i && c.j) { bump(c.i, true, true); bump(c.j, true, true); }
        else if (c.j == 0 && c.i) bump(c.i, false, true);
        else if (c.i == 0 && c.j) bump(c.j, true, false);
        // The region reads these clocks at every location, so the active-clock reduction must
        // not zero them. Marking them active everywhere is the analogue of qmask on this path.
        for (auto& pa : bounded.active) for (auto& l : pa) {
          if (c.i > 0 && c.i < (int)l.size()) l[c.i] = 1;
          if (c.j > 0 && c.j < (int)l.size()) l[c.j] = 1;
        }
      }
    }
    bounded.active_mask.clear();          // rebuilt from `active` below
    for (const auto& pa : bounded.active) {
      bounded.active_mask.emplace_back();
      for (const auto& l : pa) {
        std::uint64_t m = 0;
        if (bounded.nclocks <= 63)
          for (int c = 1; c <= bounded.nclocks; ++c) if (l[c]) m |= std::uint64_t(1) << c;
        bounded.active_mask.back().push_back(m);
      }
    }
    if (bounded.nclocks > 63) bounded.active_mask.clear();
    engine_net = &bounded;
  }

  // ---- witness search in a sub-system, before the full one -----------------
  //
  // `E[] psi` needs one run, and a run in which some processes never take an edge is a perfectly
  // legal run of the whole system: a transition is an option, never an obligation. So a maximal
  // psi-run found with a set of processes' edges suppressed IS a maximal psi-run of the network
  // as given, and `sat` from such a search is a verdict for the network as given. Nothing about
  // the model, the query or the answer changes -- only where the search looks first.
  //
  // It looks there because the witnesses are local and the search is not. TChecker's lasso for
  // A witness is often a handful of steps of one process while every other stays put,
  // sitting in `idle`; the breadth-first search reaches it only after enumerating every
  // interleaving of nine processes down to that depth. Same witness, 2816 states against 3.
  //
  //
  // soundness rests on two things and is checked rather than assumed:
  //
  //   * maximality is judged in the full network. `net_in` goes in as the unrestricted network
  //     exactly as below, so ::LivenessEngine::stop_set takes its restricted branch and asks
  //     hora::detail::state_has_deadlock about `net_in` -- a state that only looks stuck
  //     because its neighbours were suppressed is not read as the end of a maximal run.
  //   * nothing may force a suppressed process to move. Urgent and committed locations do, and
  //     so does an urgent channel, so the lane declines outright when any suppressed process has
  //     such a location or the network uses urgent channels. Time-passage is otherwise identical:
  //     the invariants are untouched, only edges are withheld.
  //
  // Only sat is believed. A sub-system that finds nothing has proved nothing -- the witness may
  // need every process -- so the search falls through to the full one below.
  const std::size_t nprocs = engine_net->procs.size();
  if (!net_in.has_urgent_chan && nprocs > 1) {
    std::vector<char> keep(nprocs, 0);
    auto mark_locs = [&](const std::vector<LocReq>& ls) {
      for (const auto& l : ls)
        if (l.proc >= 0 && (std::size_t)l.proc < nprocs) keep[l.proc] = 1;
    };
    mark_locs(nq.locs);
    for (const auto& t : nq.forbid) mark_locs(t.locs);
    // The clocks psi constrains, and every process that guards or resets one of them: suppressing
    // those would change which valuations psi can be maintained through.
    std::vector<char> psi_clock(engine_net->nclocks + 1, 0);
    auto mark_clocks = [&](const std::vector<Constraint>& cs) {
      for (const auto& c : cs) {
        if (c.i > 0 && c.i <= engine_net->nclocks) psi_clock[c.i] = 1;
        if (c.j > 0 && c.j <= engine_net->nclocks) psi_clock[c.j] = 1;
      }
    };
    mark_clocks(psi_clocks);
    for (const auto& t : nq.forbid) mark_clocks(t.clocks);
    // Read the footprint off `net_in`, NOT off the psi-restricted network: the restriction
    // adjoined psi's clock conjunction to every location invariant, so scanning those would
    // report every process as touching psi's clocks and keep the whole system.
    for (std::size_t p = 0; p < nprocs && p < net_in.procs.size(); ++p) {
      for (const auto& e : net_in.procs[p].edges) {
        for (int rc : e.clock_resets)
          if (rc > 0 && rc <= net_in.nclocks && psi_clock[rc]) keep[p] = 1;
        for (const auto& c : e.clock_guard)
          if ((c.i > 0 && psi_clock[c.i]) || (c.j > 0 && psi_clock[c.j])) keep[p] = 1;
      }
      for (const auto& l : net_in.procs[p].locations)
        for (const auto& c : l.invariant)
          if ((c.i > 0 && psi_clock[c.i]) || (c.j > 0 && psi_clock[c.j])) keep[p] = 1;
    }
    // Synchronisation closure, and it is one partner per channel and direction, not all of them.
    // A kept process's `c!` edge needs somebody offering `c?` to fire; it does not need everybody
    // who could. Keeping every partner is what makes this lane useless on exactly the models it
    // is for -- arbiter's nine clients all share `acq`/`rel`, so "keep all partners" closes over
    // the whole system and freezes nothing. A broadcast send needs no partner at all.
    for (bool grew = true; grew;) {
      grew = false;
      for (std::size_t p = 0; p < nprocs && !grew; ++p) {
        if (!keep[p]) continue;
        for (const auto& e : engine_net->procs[p].edges) {
          if (e.sync_chan < 0 || e.sync_dir == 0) continue;
          if (e.sync_dir > 0 && (std::size_t)e.sync_chan < engine_net->chan_broadcast.size() &&
              engine_net->chan_broadcast[e.sync_chan])
            continue;                                   // broadcast `!` fires with 0 receivers
          bool have = false;
          int candidate = -1;
          for (std::size_t o = 0; o < nprocs && !have; ++o) {
            if (o == p) continue;
            for (const auto& oe : engine_net->procs[o].edges) {
              if (oe.sync_chan != e.sync_chan || oe.sync_dir != -e.sync_dir) continue;
              if (keep[o]) { have = true; break; }
              if (candidate < 0) candidate = (int)o;
            }
          }
          if (!have && candidate >= 0) { keep[candidate] = 1; grew = true; break; }
        }
      }
    }
    std::size_t frozen = 0;
    bool safe = true;
    for (std::size_t p = 0; p < nprocs && safe; ++p) {
      if (keep[p]) continue;
      ++frozen;
      for (const auto& l : engine_net->procs[p].locations)
        if (l.urgent || l.committed) safe = false;      // urgency would force it to move
    }
    if (std::getenv("HORA_EG_SUBDBG")) {
      std::string k;
      for (std::size_t p = 0; p < nprocs; ++p) k += keep[p] ? '+' : '.';
      std::fprintf(stderr, "SUB keep=[%s] frozen=%zu safe=%d\n", k.c_str(), frozen, (int)safe);
    }
    if (safe && frozen > 0) {
      Network sub = *engine_net;
      for (std::size_t p = 0; p < nprocs; ++p)
        if (!keep[p]) {
          sub.procs[p].edges.clear();
          sub.procs[p].build_edge_index();
        }
      // The abstraction data IS recomputed for the sub-system. Carrying the full network's bounds
      // over looks conservative -- they are larger, and a larger bound only makes a_LU finer -- but
      // the active-clock marks come with them, and a clock marked active only because a
      // now-suppressed process reads it keeps a dimension alive that the sub-system never
      // constrains. Recomputing is also sound in its own right -- every guard the smaller bounds
      // must dominate is still present, since only edges were removed.
      sub.compute_bounds();
      sub.compute_loc_bounds();
      sub.compute_active();
      sub.compute_int_live();
      GfpStats gsub;
      const bool sub_sat = exists_invariant_run_otf(sub, nq, &gsub, &net_in,
                                                    av.parts.empty() ? nullptr : &av,
                                                    /*witness_only=*/true);
      if (std::getenv("HORA_EG_SUBDBG"))
        std::fprintf(stderr, "SUB result=%s nodes=%zu\n", sub_sat ? "SAT" : "no witness",
                     (std::size_t)gsub.outer_nodes);
      if (sub_sat) {
        if (stats) stats->explored = gsub.outer_nodes;
        return true;
      }
    }
  }

  GfpStats gs;
  const bool r = exists_invariant_run_otf(*engine_net, nq, &gs, &net_in,
                                          av.parts.empty() ? nullptr : &av);
  if (stats) stats->explored = gs.outer_nodes;
  return r;
}

/// @brief Does the symbolic state @p s contain a valuation with no outgoing action transition?
///
/// A zone with no successor at all is the easy case. The hard case is a *partial* deadlock,
/// where the zone has successors but some sub-region of it can never reach any of them — for
/// instance where an invariant caps time before the only guard opens. This computes the partial
/// case exactly rather than approximating it.
///
/// The three transition kinds of ::successors_net are mirrored. For each transition that can
/// fire, its *source* region within `s.zone` is collected: the pre-reset valuations from which it
/// fires and lands in a legal target, i.e. the guards intersected with the target invariant
/// pulled back through the resets. The union of the time-pasts of those source regions is the set
/// of non-deadlocked valuations, and whatever remains of the zone after subtracting it is
/// deadlocked.
///
/// @param net  the network under analysis
/// @param s    the symbolic state to test
/// @param[out] stuck    if non-null, receives the deadlocked sub-region; cleared when there is none
/// @param[out] escapes  if non-null, receives the escape regions themselves, for callers that
///                      need to know *which* constraints decide stuckness rather than only the
///                      residue
/// @return @c true iff some valuation of @p s is deadlocked
///
/// @pre  For an exact answer, `s.zone` must already satisfy the invariant of `s.locs`. States
///       produced by the forward engine satisfy this, since ::close_succ_zone applies it.
/// @post `*escapes` is inclusion-minimal; a region contained in a sibling is never added.
///
/// @warning At a **frozen** state — committed or urgent — neither the delay nor the invariant is
///          applied, so the zone subtracted from is `s.zone` as given. A caller that passes the
///          bare universe for such a state therefore receives `Dead` computed against the
///          universe rather than against `Inv(q)`, which over-approximates. This is the
///          conservative direction and costs precision only, but such a caller must intersect the
///          result with `Inv(q)` itself. See docs/limitations.md.
/// @note Enabledness is invariant under LU-extrapolation, so working on the extrapolated zone is
///       sound — the same assumption the rest of the engine relies on.
/// @see docs/design.md §2 for the deadlock completion this feeds.

}  // namespace hora
