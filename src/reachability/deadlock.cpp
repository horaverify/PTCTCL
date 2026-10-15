/// @file src/reachability/deadlock.cpp
/// @brief Whether a symbolic state contains a deadlocked valuation, and whether one is reachable.
///
/// A zone with no successor at all is the easy case. The hard one is a PARTIAL deadlock, where
/// the zone has successors but a sub-region of it can never reach any of them; that is computed
/// exactly rather than approximated.

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
#include "hora/detail/passed_set.hpp"
#include "hora/detail/semantics.hpp"
#include "hdbm/mindbm.hpp"   // minimal-graph packed zone storage (UDBM mingraph port)
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
using detail::LocVec;
using detail::IntVec;
using detail::all_invariants;
using detail::frozen_flags;
using detail::int_ok;
using detail::RecvOpt;
using detail::build_broadcast_parts;

bool detail::state_has_deadlock(const Network& net, const SymState& s,
                                Federation* stuck, Federation* escapes) {
  if (s.zone.empty()) return false;
  bool src_committed, src_urgent;
  frozen_flags(net, s.locs, s.ints, &src_committed, &src_urgent);
  const bool frozen = src_committed || src_urgent;  // no delay allowed from this state

  // Rebuild the true delay-closure of this state's zone. The stored zone carries the
  // active-clock reduction, which pins dead clocks to 0 with an upper bound and so is no longer
  // delay-closed; without this step down() could not let time pass those clocks and would invent
  // spurious deadlocks. Re-opening every upper bound and re-imposing the real location invariant
  // restores the closure. Dead clocks become free, which is sound because no guard or invariant
  // reachable from here reads them.
  //
  // The invariant is applied whether or not the location is frozen; only the delay is skipped.
  // A frozen location still constrains its clocks, and omitting the invariant left @p Z as the
  // bare universe, which reports valuations stuck that the invariant forbids from arising at all.
  // That was conservative rather than wrong -- a spurious residue only ever costs precision --
  // but it made the caller decline on keys that cannot deadlock.
  Dbm Z = s.zone;
  {
    if (!frozen) Z.delay();
    std::vector<Constraint> cinv;
    all_invariants(net, s.locs, cinv);
    if (!apply_all(Z, cinv)) return false;  // real invariant unsatisfiable: no concrete state
  }

  std::vector<Constraint> tinv;  // target-invariant scratch
  Federation live;
  // Set when an escape is produced that covers the whole of Z; see ::add_src. The enumeration
  // below then has nothing left to learn and stops.
  //
  // @warning `*escapes` is NOT populated on that path. No caller reads it on a `false` return.
  bool covered = false;

  // `guarded` is Z intersected with the participants' clock guards, before any reset. It is
  // pulled back through the target invariant under the transition's resets, and if a non-empty
  // source region survives it joins `live`. In that pullback a constraint on a reset clock
  // becomes a constraint on clock 0, its post-reset value, so it either constrains the surviving
  // source clocks or, when it reduces to `0 ~ bound`, gates the whole transition on whether that
  // constant is satisfiable.
  auto add_src = [&](Dbm guarded, const LocVec& tgt_locs,
                     const std::vector<const std::vector<int>*>& reset_lists) {
    if (guarded.empty()) return;
    auto is_reset = [&](int c) {
      for (auto* rl : reset_lists) for (int r : *rl) if (r == c) return true;
      return false;
    };
    all_invariants(net, tgt_locs, tinv);
    for (const auto& c : tinv) {
      const int i = is_reset(c.i) ? 0 : c.i;
      const int j = is_reset(c.j) ? 0 : c.j;
      if (i == j) { if (c.bound < kZeroLe) return; continue; }  // 0 <| bound: constant gate
      if (!guarded.constrain(i, j, c.bound)) return;
    }
    // Kept inclusion-minimal rather than simply appended. `live` is consumed only through its
    // union — by the subtraction below, and by `escapes` for the atom construction — so a member
    // contained in a sibling denotes nothing new. It is not free either: the subtraction peels
    // the remainder once per member and the piece count multiplies, so a redundant escape costs
    // an extra peel of an already-large arrangement. Two escapes that agree on their clock guard
    // and differ only in which process moved are the common case.
    //
    // Pruning before down() is sound because W subseteq W' implies down(W) subseteq down(W'), so
    // the union of the time-pasts is unchanged.
    if (guarded.empty()) return;
    // the whole-zone escape, detected where it is produced rather than after the fact. `guarded`
    // starts as Z and is only ever constrained, and dbm_constrain writes only a strictly tighter
    // bound -- which, on a canonical DBM, strictly shrinks the set. So `guarded == Z` entrywise
    // iff nothing tightened, iff this transition is enabled from every valuation of Z. Then
    // down(guarded) contains Z and Dead is empty, whatever the remaining transitions do.
    //
    // The same conclusion is available from the post-loop inclusion test below, but only after
    // the O(nprocs^2) synchronisation enumeration has run and `fed_add` has kept `live`
    // inclusion-minimal at O(|live|^2) DBM comparisons. This reaches it on the first unguarded
    // edge, for one memcmp.
    if (std::memcmp(guarded.data().data(), Z.data().data(),
                    sizeof(raw_t) * Z.data().size()) == 0) { covered = true; return; }
    fed_add(live, guarded);
  };

  // ---- local (unsynchronised) edges ----
  for (int p = 0; p < net.nprocs() && !covered; ++p) {
    if (src_committed && !net.procs[p].locations[s.locs[p]].committed) continue;
    for (int ei : net.procs[p].edges_of(s.locs[p])) {
      const NEdge& e = net.procs[p].edges[ei];
      if (e.sync_chan >= 0) continue;
      if (!int_ok(e.int_guard, s.ints) || !gen_ok(net, e, s.ints.data(), s.ints.size())) continue;
      IntVec nints = s.ints;
      for (const auto& a : e.int_updates) nints[a.var] = a.val;
      if (!gen_apply(net, e, nints.data(), nints.size())) continue;
      Dbm g = Z;
      if (!apply_all(g, e.clock_guard)) continue;
      LocVec tl = s.locs; tl[p] = e.tgt;
      add_src(std::move(g), tl, {&e.clock_resets});
    }
  }

  // ---- binary channel synchronisation (a! || a?) ----
  for (int i = 0; i < net.nprocs() && !covered; ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      if (covered) break;
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1) continue;
      if (net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      const bool i_comm = net.procs[i].locations[s.locs[i]].committed;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        for (int ej : net.procs[j].edges_of(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan != se.sync_chan || re.sync_dir != -1) continue;
          if (src_committed && !(i_comm || net.procs[j].locations[s.locs[j]].committed)) continue;
          if (!int_ok(re.int_guard, s.ints) || !gen_ok(net, re, s.ints.data(), s.ints.size())) continue;
          IntVec nints = s.ints;
          for (const auto& a : se.int_updates) nints[a.var] = a.val;
          for (const auto& a : re.int_updates) nints[a.var] = a.val;
          if (!gen_apply(net, se, nints.data(), nints.size()) ||
              !gen_apply(net, re, nints.data(), nints.size())) continue;
          Dbm g = Z;
          if (!apply_all(g, se.clock_guard) || !apply_all(g, re.clock_guard)) continue;
          LocVec tl = s.locs; tl[i] = se.tgt; tl[j] = re.tgt;
          add_src(std::move(g), tl, {&se.clock_resets, &re.clock_resets});
        }
      }
    }
  }

  // ---- broadcast synchronisation: a! fires with all enabled a? (0 or more) ----
  for (int i = 0; i < net.nprocs() && !covered; ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      if (covered) break;
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1 || !net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      std::vector<std::pair<int, std::vector<RecvOpt>>> parts;   // see build_broadcast_parts
      build_broadcast_parts(net, s, se, i, parts);
      std::vector<std::size_t> pick(parts.size(), 0);
      for (;;) {
        bool part_committed = net.procs[i].locations[s.locs[i]].committed;
        for (std::size_t t = 0; t < parts.size(); ++t)
          if (parts[t].second[pick[t]].edge >= 0 &&
              net.procs[parts[t].first].locations[s.locs[parts[t].first]].committed) part_committed = true;
        if (!src_committed || part_committed) {
          Dbm g = Z;
          bool ok = apply_all(g, se.clock_guard);
          std::vector<const std::vector<int>*> rls{&se.clock_resets};
          LocVec tl = s.locs; tl[i] = se.tgt;
          for (std::size_t t = 0; ok && t < parts.size(); ++t) {
            const RecvOpt& o = parts[t].second[pick[t]];
            if (o.edge < 0) { ok = apply_all(g, o.extra); continue; }   // abstained: nothing moves
            const NEdge& re = net.procs[parts[t].first].edges[o.edge];
            ok = apply_all(g, re.clock_guard);
            rls.push_back(&re.clock_resets);
            tl[parts[t].first] = re.tgt;
          }
          if (ok) add_src(std::move(g), tl, rls);
        }
        if (parts.empty()) break;
        std::size_t k = 0;
        for (; k < parts.size(); ++k) { if (++pick[k] < parts[k].second.size()) break; pick[k] = 0; }
        if (k == parts.size()) break;
      }
    }
  }

  if (covered) {
    if (stuck) stuck->clear();
    if (escapes) escapes->clear();
    return false;
  }
  // Non-deadlocked = valuations that can (delay into, unless frozen) some enabled source
  // region. Subtract that from the zone; a non-empty remainder is a set of deadlock states.
  if (!frozen)
    for (auto& W : live) W.down();
  // The escape regions themselves, for callers needing to know which constraints decide
  // stuckness rather than only the residue. Since Dead = Z \ union(live), membership is decided
  // by the atoms of each escape — one convex zone per outgoing transition — whereas the residue
  // is an arrangement whose piece count, and hence atom count, is far larger for the same set.
  if (escapes) *escapes = live;
  // Short circuit when a single escape already covers Z, which is the common case of a location
  // with one unguarded outgoing edge. The subtraction peels Z once per escape and the piece count
  // multiplies, so the arrangement it builds dominates the sweep; one DBM inclusion test per
  // escape avoids building it at all.
  //
  // This is exact, not a heuristic: Z subseteq W subseteq union(live) means Dead(q) is empty,
  // which is precisely what the subtraction would have reported.
  for (const Dbm& W : live)
    if (W.includes(Z)) {
      if (stuck) stuck->clear();
      return false;
    }
  Federation d = fed_subtract(Z, live);      // exactly the valuations with nothing enabled
  if (stuck) *stuck = d;
  return !d.empty();
}

detail::DeadlockSeed detail::deadlock_seed(const Network& net, const LocVec& locs,
                                            const IntVec& ints) {
  DeadlockSeed out;
  const int nc = net.nclocks;
  SymState u;
  u.locs = locs;
  u.ints = ints;
  bool fc, fu;
  frozen_flags(net, u.locs, u.ints, &fc, &fu);
  std::vector<Constraint> ci;
  all_invariants(net, u.locs, ci);

  u.zone = Dbm::universe(nc);
  Dbm inv = Dbm::universe(nc);
  if (!fc && !fu) inv.delay();
  if (!apply_all(inv, ci)) return out;

  Federation stuck, esc;
  if (!state_has_deadlock(net, u, &stuck, &esc)) return out;
  out.may_deadlock = true;

  Federation kept;
  kept.reserve(esc.size());
  for (const Dbm& w : esc) {
    Dbm t = w;
    t.intersect(inv);
    if (t.empty()) continue;
    kept.push_back(w);
  }
  if (kept.empty()) return out;                 // Dead(q) == Inv(q): no distinguishing atom
  esc.swap(kept);

  const int dim = nc + 1;
  std::vector<std::uint32_t> mbits(mindbm_bits2words(std::uint32_t(dim) * std::uint32_t(dim)));
  std::vector<int> mfirst(dim), mnext(dim);
  // Atoms of one federation: how many diagonal ones, and the non-diagonal ones as (clock, bound).
  auto extract = [&](const Federation& f, std::size_t* ndiag,
                     std::vector<std::pair<int, std::int32_t>>& nond) {
    for (const Dbm& d : f) {
      if (d.empty()) continue;
      mindbm_analyze(d.data().data(), dim, mbits.data(), mfirst.data(), mnext.data());
      for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
          if (i == j) continue;
          const std::uint32_t idx = std::uint32_t(i) * std::uint32_t(dim) + std::uint32_t(j);
          if (!((mbits[idx >> 5] >> (idx & 31)) & 1u)) continue;
          const raw_t r = d.get(i, j);
          if (r >= kInf) continue;
          if (i == 0 && r >= kZeroLe) continue;
          if (inv.get(i, j) <= r) continue;
          if (raw_add(inv.get(j, i), r) < kZeroLe) continue;
          if (i > 0 && j > 0) { ++*ndiag; continue; }
          const int c = i == 0 ? j : i;
          const std::int32_t b = i == 0 ? -raw_bound(r) : raw_bound(r);
          bool seen = false;
          for (auto& x : nond)
            if (x.first == c) { x.second = std::max(x.second, b); seen = true; break; }
          if (!seen) nond.push_back({c, b});
        }
    }
  };
  std::size_t esc_diag = 0, res_diag = 0;
  std::vector<std::pair<int, std::int32_t>> esc_nond, res_nond;
  extract(stuck, &res_diag, res_nond);
  extract(esc, &esc_diag, esc_nond);
  // The escape arrangement and the residue carve out the same boundary, so either set of atoms
  // keeps the covering deadlock-preserving. Take one without a diagonal when there is one.
  if (esc_diag == 0) out.bumps = std::move(esc_nond);
  else if (res_diag == 0) out.bumps = std::move(res_nond);
  else out.diagonal = true;
  return out;
}

}  // namespace hora
