/// @file src/liveness/operators.cpp
/// @brief The symbolic operators: successors, the exact predecessor, federations.
///
/// These are the transition relation in both directions, plus the federation normalisation the
/// backward sweep depends on. None of them touches the engine's state.

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

namespace {

/// @brief Can every atom of the guard @p g hold somewhere in the canonical zone @p z?
///
/// One atom `x_i - x_j ~ b` empties a canonical zone exactly when it closes a negative cycle with
/// the shortest path back, `z(j,i)`, so each atom is decided in O(1). A conjunction can still die
/// from atoms interacting with each other, so this only REJECTS: a guard it passes is applied in
/// full afterwards. What it saves is the zone copy, which every disabled edge used to pay before
/// its first constraint failed.
bool guard_may_hold(const Dbm& z, const std::vector<Constraint>& g) {
  for (const Constraint& c : g)
    if (raw_add(z.get(c.j, c.i), c.bound) < kZeroLe) return false;
  return true;
}

}  // namespace

void tagged_successors(const Network& net, const SymState& s,
                            const std::vector<std::int32_t>& L,
                            const std::vector<std::int32_t>& U,
                            const std::vector<char>& qmask, int emode,
                            std::vector<SymState>& out, std::vector<GfpTag>& tags,
                            const detail::DelayAvoidSet* avoid = nullptr) {
  std::vector<Constraint> inv;
  // A routed delay can split, so one transition may yield several successors. They share the
  // transition, hence the tag, which is duplicated alongside each extra piece.
  std::vector<Dbm> split;
  auto extra = [&]() -> std::vector<Dbm>* { return avoid ? (split.clear(), &split) : nullptr; };
  auto emit_extra = [&](const SymState& ns, const GfpTag& t) {
    for (Dbm& pz : split) {
      SymState c; c.locs = ns.locs; c.ints = ns.ints; c.zone = std::move(pz);
      if (c.zone.empty()) continue;
      out.push_back(std::move(c));
      tags.push_back(t);
    }
    split.clear();
  };
  bool src_committed, src_urgent;
  frozen_flags(net, s.locs, s.ints, &src_committed, &src_urgent);

  // ---- local edges ----
  for (int p = 0; p < net.nprocs(); ++p) {
    if (src_committed && !net.procs[p].locations[s.locs[p]].committed) continue;
    for (int ei : net.procs[p].edges_of(s.locs[p])) {
      const NEdge& e = net.procs[p].edges[ei];
      if (e.sync_chan >= 0) continue;
      if (!int_ok(e.int_guard, s.ints) || !gen_ok(net, e, s.ints.data(), s.ints.size())) continue;
      if (!guard_may_hold(s.zone, e.clock_guard)) continue;
      Dbm z1 = s.zone;
      if (!apply_all(z1, e.clock_guard)) continue;
      for (int r : e.clock_resets) z1.reset(r);
      SymState ns;
      ns.locs = s.locs; ns.locs[p] = e.tgt;
      ns.ints = s.ints;
      for (const auto& a : e.int_updates) ns.ints[a.var] = a.val;
      if (!gen_apply(net, e, ns.ints.data(), ns.ints.size())) continue;
      if (!close_succ_zone(net, ns.locs, ns.ints, z1, inv, qmask, L, U, emode, avoid, extra())) continue;
      GfpTag t; t.parts.push_back({p, ei});
      emit_extra(ns, t);
      if (z1.empty()) continue;
      ns.zone = std::move(z1);
      out.push_back(std::move(ns));
      tags.push_back(std::move(t));
    }
  }

  // ---- binary channel synchronisation ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1) continue;
      if (net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      if (!guard_may_hold(s.zone, se.clock_guard)) continue;   // no partner can rescue the sender
      const bool i_comm = net.procs[i].locations[s.locs[i]].committed;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        for (int ej : net.procs[j].edges_of(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan != se.sync_chan || re.sync_dir != -1) continue;
          if (src_committed && !(i_comm || net.procs[j].locations[s.locs[j]].committed)) continue;
          if (!int_ok(re.int_guard, s.ints) || !gen_ok(net, re, s.ints.data(), s.ints.size())) continue;
          if (!guard_may_hold(s.zone, re.clock_guard)) continue;
          Dbm z1 = s.zone;
          if (!apply_all(z1, se.clock_guard)) continue;
          if (!apply_all(z1, re.clock_guard)) continue;
          for (int r : se.clock_resets) z1.reset(r);
          for (int r : re.clock_resets) z1.reset(r);
          SymState ns;
          ns.locs = s.locs; ns.locs[i] = se.tgt; ns.locs[j] = re.tgt;
          ns.ints = s.ints;
          for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;
          for (const auto& a : re.int_updates) ns.ints[a.var] = a.val;
          if (!gen_apply(net, se, ns.ints.data(), ns.ints.size()) ||
              !gen_apply(net, re, ns.ints.data(), ns.ints.size())) continue;
          if (!close_succ_zone(net, ns.locs, ns.ints, z1, inv, qmask, L, U, emode, avoid, extra())) continue;
          GfpTag t; t.parts.push_back({i, ei}); t.parts.push_back({j, ej});
          emit_extra(ns, t);
          if (z1.empty()) continue;
          ns.zone = std::move(z1);
          out.push_back(std::move(ns));
          tags.push_back(std::move(t));
        }
      }
    }
  }

  // ---- broadcast: a! fires together with every enabled a? (0 or more receivers) ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1 || !net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
      if (!guard_may_hold(s.zone, se.clock_guard)) continue;
      std::vector<std::pair<int, std::vector<RecvOpt>>> parts;   // see build_broadcast_parts
      build_broadcast_parts(net, s, se, i, parts);
      std::vector<std::size_t> pick(parts.size(), 0);
      for (;;) {
        bool part_committed = net.procs[i].locations[s.locs[i]].committed;
        for (std::size_t t2 = 0; t2 < parts.size(); ++t2)
          if (parts[t2].second[pick[t2]].edge >= 0 &&
              net.procs[parts[t2].first].locations[s.locs[parts[t2].first]].committed) part_committed = true;
        if (!src_committed || part_committed) {
          Dbm z1 = s.zone;
          bool ok = apply_all(z1, se.clock_guard);
          for (std::size_t t2 = 0; ok && t2 < parts.size(); ++t2) {
            const RecvOpt& o = parts[t2].second[pick[t2]];
            ok = o.edge >= 0 ? apply_all(z1, net.procs[parts[t2].first].edges[o.edge].clock_guard)
                             : apply_all(z1, o.extra);
          }
          if (ok) {
            SymState ns;
            ns.locs = s.locs; ns.ints = s.ints;
            ns.locs[i] = se.tgt;
            for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;
            for (int r : se.clock_resets) z1.reset(r);
            GfpTag tg; tg.parts.push_back({i, ei});
            for (std::size_t t2 = 0; t2 < parts.size(); ++t2) {
              const RecvOpt& o = parts[t2].second[pick[t2]];
              if (o.edge < 0) {          // abstained: nothing moves, but the guard's complement
                                         // is part of what enables this step; keep it for Pre
                tg.abstain.insert(tg.abstain.end(), o.extra.begin(), o.extra.end());
                continue;
              }
              const int pj = parts[t2].first, ej = o.edge;
              const NEdge& re = net.procs[pj].edges[ej];
              ns.locs[pj] = re.tgt;
              for (const auto& a : re.int_updates) ns.ints[a.var] = a.val;
              for (int r : re.clock_resets) z1.reset(r);
              tg.parts.push_back({pj, ej});
            }
            if (close_succ_zone(net, ns.locs, ns.ints, z1, inv, qmask, L, U, emode, avoid, extra())) {
              emit_extra(ns, tg);
              if (!z1.empty()) {
                ns.zone = std::move(z1);
                out.push_back(std::move(ns));
                tags.push_back(std::move(tg));
              }
            }
          }
        }
        if (parts.empty()) break;
        std::size_t k = 0;
        for (; k < parts.size(); ++k) { if (++pick[k] < parts[k].second.size()) break; pick[k] = 0; }
        if (k == parts.size()) break;
      }
    }
  }
}

/// @brief An empty zone of the caller's dimension.
///
/// @param nclocks number of clocks; the returned DBM has dimension `nclocks + 1`
/// @return a DBM @c z with `z.empty() == true` and `z.dim() == nclocks + 1`
///
/// @warning Do not substitute a default-constructed @c Dbm here. That is a zero-clock
///          placeholder — a 1x1 matrix holding @c kZeroLe — and emptiness is decided by
///          scanning the diagonal for a negative entry, so `Dbm().empty()` is @b false. Such a
///          value would pass the `!pz.empty()` filter in ::LivenessEngine::solve_greatest_fixpoint and enter a federation
///          of `nclocks + 1`-dimensional zones, where the next inclusion test reads
///          `dim * dim` entries out of a 4-byte allocation.
Dbm empty_zone(int nclocks) {
  Dbm z(nclocks);
  z.set(0, 0, lt(0));      // 0 - 0 < 0 is unsatisfiable, so dbm_empty() is true
  return z;
}


/// @brief The exact concrete predecessor of one tagged transition.
///
/// Inverts the forward step described by @p t, mapping a set @p X of target valuations to the
/// set of source valuations from which the step can reach @p X. One code path serves local
/// edges, binary synchronisation and broadcast: every participating edge contributes its guard
/// and its resets.
///
///     forward:  z1 = Z /\ (all guards) ; reset (all resets) ; /\ inv' ;
///               if !frozen' { delay ; /\ inv' }
///     backward: undo the active-clock reduction at the target ; /\ inv' ; down ; /\ inv' ;
///               /\ (x_r = 0 for every reset r) ; free those r ; /\ (all guards) ;
///               down (the delay before the action) ; /\ inv(src)
///
/// @param net   the network under analysis
/// @param t     the tag produced for this transition by ::tagged_successors
/// @param sn    the source node's discrete data
/// @param tn    the target node's discrete data
/// @param rmask `rmask[c] = 1` for every clock reset by @p t; see ::tag_reset_mask
/// @param X     a set of target valuations
/// @return the exact set of source valuations that can reach @p X by this transition, or an
///         empty zone of dimension `net.nclocks + 1` if there are none
///
/// @pre @p t, @p sn and @p tn come from the same call to ::tagged_successors.
/// @post No extrapolation and no active-clock reduction is applied in the forward direction:
///       the result is the exact concrete predecessor. Abstraction enters the fixpoint only
///       through the caller's confinement `W_v subseteq Z_v`.
///
/// @note Neither node's zone is read, only its discrete data. The caller must not materialise
///       one: doing so decompressed two full DBMs out of the ZoneStore per call, for nothing.
/// @note The two clock sets are quantified out with ::dbm_free_many rather than a `free` per
///       clock. Both are exact; the batch form relaxes every flagged clock at once.
Dbm exact_predecessor(const Network& net, const GfpTag& t, const PreNode& sn, const PreNode& tn,
                   const std::vector<Constraint>& invs,
                   const std::vector<unsigned char>& rmask, unsigned char* dead_scratch,
                   const Dbm& X) {
  Dbm Y = X;
  const int dim = net.nclocks + 1;
  const Constraint* tinv = invs.data() + tn.inv_off;
  const Constraint* sinv = invs.data() + sn.inv_off;
  const auto apply = [](Dbm& z, const Constraint* c, std::uint32_t n) {
    for (std::uint32_t k = 0; k < n; ++k)
      if (!z.constrain(c[k].i, c[k].j, c[k].bound)) return false;
    return true;
  };

  // Undo the active-clock reduction at the target. Those clocks appear in no edge's reset set,
  // so without this Pre would silently demand that they also read 0 at the source -- an
  // over-constrained Pre that collapses the fixpoint and misses violations. Freeing them is the
  // exact inverse of the reduction, and it is sound because they are dead: nothing reads them.
  if (tn.dead) {
    std::memset(dead_scratch, 0, static_cast<std::size_t>(dim));
    for (std::uint64_t m = tn.dead; m; m &= m - 1) dead_scratch[__builtin_ctzll(m)] = 1;
    dbm_free_many(Y.data().data(), dim, dead_scratch);
  }

  if (!apply(Y, tinv, tn.inv_len)) return empty_zone(net.nclocks);
  if (!tn.frozen) { Y.down(); if (!apply(Y, tinv, tn.inv_len)) return empty_zone(net.nclocks); }

  // Undo every participant's resets: a reset clock reads 0 after the step, and its value before
  // the step is unconstrained by the step itself.
  for (const auto& pe : t.parts)
    for (int r : net.procs[pe.first].edges[pe.second].clock_resets) {
      if (!Y.constrain(r, 0, le(0))) return empty_zone(net.nclocks);
      if (!Y.constrain(0, r, le(0))) return empty_zone(net.nclocks);
    }
  dbm_free_many(Y.data().data(), dim, rmask.data());

  for (const auto& pe : t.parts)
    if (!apply_all(Y, net.procs[pe.first].edges[pe.second].clock_guard))
      return empty_zone(net.nclocks);
  // The abstaining receivers' guard complements belong here and nowhere else: like a
  // participant's guard they constrain the source valuation, and unlike one they undo nothing,
  // so they apply after the resets have been freed and before the pre-action delay.
  if (!t.abstain.empty() && !apply_all(Y, t.abstain)) return empty_zone(net.nclocks);

  // Delay before the action. A stored zone is delay-closed as a set, which is why the forward
  // operator may write `Z /\ g`; a single valuation may still have to let time pass first.
  if (!sn.frozen) Y.down();
  if (!apply(Y, sinv, sn.inv_len)) return empty_zone(net.nclocks);
  return Y;
}

/// @brief ::exact_predecessor for a run confined to a non-convex predicate.
///
/// @param avoid the forbidden region; must be non-null
/// @param[in,out] out  the pieces of the predecessor are appended
///
/// @par Why this is federation-valued
/// ::exact_predecessor takes two time-predecessors, one at the target and one at the source. With a region
/// to avoid, each of them routes around it and each can therefore split, so the predecessor of a
/// single zone is in general several zones. Collapsing them to one -- by hull, or by keeping the
/// largest -- under-approximates `Pre`, and `Pre` occurs positively in the fixpoint, so that
/// silently loses witnesses: it turned `E[] x <= 2 || x >= 8` on a model whose witness is an
/// infinite zero-time loop from satisfied into NOT satisfied.
///
/// @note Every step other than the two time-predecessors maps a zone to a zone, so the federation
///       is threaded through mechanically and only the two routed steps branch.
void exact_predecessor_avoiding(const Network& net, const GfpTag& t, const SymState& src,
                          const SymState& tgt, const Dbm& X,
                          const detail::DelayAvoidSet& avoid, Federation& out) {
  Federation cur{X};
  auto map_each = [&](auto&& fn) {                     // zone -> bool(zone&), dropping failures
    Federation next;
    for (Dbm& z : cur) { Dbm y = z; if (fn(y) && !y.empty()) next.push_back(std::move(y)); }
    cur.swap(next);
  };
  auto route = [&](const LocVec& locs, const IntVec& ints) {
    if (!avoid.applies(locs, ints)) { map_each([](Dbm& y) { y.down(); return true; }); return; }
    Federation next;
    Federation scratch;
    for (const Dbm& z : cur) {
      Federation acc;
      bool first = true;
      for (const auto& part : avoid.parts) {
        if (!part.applies(locs, ints)) continue;
        detail::pre_avoiding(z, part.forbid, part.fup, scratch);
        if (first) { acc = scratch; first = false; }
        else acc = fed_intersect(acc, scratch);
        if (acc.empty()) break;
      }
      if (first) { Dbm y = z; y.down(); fed_add(next, y); continue; }
      for (Dbm& q : acc) if (!q.empty()) fed_add(next, q);
    }
    cur.swap(next);
  };

  if (!net.active.empty())
    map_each([&](Dbm& Y) {
      for (int c = 1; c <= net.nclocks; ++c) {
        bool live = false;
        for (int pr = 0; pr < net.nprocs() && !live; ++pr)
          if (net.active[pr][tgt.locs[pr]][c]) live = true;
        if (!live) Y.free(c);
      }
      return true;
    });

  std::vector<Constraint> inv;
  all_invariants(net, tgt.locs, inv);
  bool nc, nu; frozen_flags(net, tgt.locs, tgt.ints, &nc, &nu);
  map_each([&](Dbm& Y) { return apply_all(Y, inv); });
  if (!nc && !nu) {
    route(tgt.locs, tgt.ints);
    map_each([&](Dbm& Y) { return apply_all(Y, inv); });
  }
  map_each([&](Dbm& Y) {
    for (const auto& pe : t.parts)
      for (int r : net.procs[pe.first].edges[pe.second].clock_resets) {
        if (!Y.constrain(r, 0, le(0))) return false;
        if (!Y.constrain(0, r, le(0))) return false;
      }
    for (const auto& pe : t.parts)
      for (int r : net.procs[pe.first].edges[pe.second].clock_resets) Y.free(r);
    for (const auto& pe : t.parts)
      if (!apply_all(Y, net.procs[pe.first].edges[pe.second].clock_guard)) return false;
    if (!t.abstain.empty() && !apply_all(Y, t.abstain)) return false;
    return true;
  });
  std::vector<Constraint> sinv;
  all_invariants(net, src.locs, sinv);
  bool sc, su; frozen_flags(net, src.locs, src.ints, &sc, &su);
  if (!sc && !su) route(src.locs, src.ints);
  map_each([&](Dbm& Y) { return apply_all(Y, sinv); });
  for (Dbm& z : cur) if (!z.empty()) out.push_back(std::move(z));
}

// ---- small federation helpers (Federation = std::vector<Dbm>, union semantics) ----

/// @brief Does @p f denote the empty set?
bool fed_is_empty(const Federation& f) {
  for (const auto& z : f) if (!z.empty()) return false;
  return true;
}

/// @brief Is every point of @p a inside the union @p b?
bool fed_contains(const Federation& a, const Federation& b) {
  for (const auto& z : a) {
    if (z.empty()) continue;
    if (!fed_covers(b, z)) return false;
  }
  return true;
}
/// @brief Normalise a federation in place: drop empties and redundant members, merge convex pairs.
///
/// Drops empty members, drops members already covered by one that is kept, replaces a kept
/// member by a newcomer that contains it, and merges any pair whose union is *exactly* convex.
///
/// The containment test runs both ways. `z` inside a kept `o` is dropped; a kept `o` inside `z`
/// is overwritten by `z`, which denotes the same union — `o subseteq z` and `z` was going to be
/// added regardless — and leaves one member where the one-directional test left two.
///
/// @param f            the federation to compact, modified in place
/// @param cap          maximum number of members to keep
/// @param[in,out] capped  set to @c true if @p cap was exceeded; never cleared
///
/// @post The set denoted by @p f is unchanged unless `*capped` was set. Convex-union merging
///       is exact in both directions — @c Dbm::try_union succeeds only when the union really is
///       a single zone — so it adds no point and removes none.
/// @warning The caller must treat `*capped` as sticky and must not read @p f as a converged
///          value once it is set; see ::LivenessEngine::certificate_in.
/// @note Without the convex-union merge the federations grow monotonically with every @c Pre
///       and the width cap becomes the binding constraint on the whole algorithm.
void fed_compact(Federation& f, std::size_t cap, bool* capped) {
  Federation out;
  bool merged = false;
  for (auto& z : f) {
    if (z.empty()) continue;
    bool absorbed = false;
    for (auto& o : out) {
      if (o.includes(z)) { absorbed = true; break; }          // already covered
      if (z.includes(o)) { o = z; absorbed = true; merged = true; break; }  // z subsumes o
      if (o.try_union(z)) { absorbed = true; merged = true; break; }  // exactly convex union
    }
    if (!absorbed) out.push_back(std::move(z));
    if (out.size() > cap) { *capped = true; break; }
  }
  // One settling pass: a merge can make an earlier member redundant against a later one. It is
  // gated on a merge having happened, because without one no member was reshaped, so the second
  // pass would re-test exactly the pairs the first already rejected — and each of those tests is
  // a try_union.
  if (!*capped && merged && out.size() > 1) {
    Federation settled;
    for (auto& z : out) {
      bool absorbed = false;
      for (auto& o : settled) {
        if (o.includes(z)) { absorbed = true; break; }
        if (o.try_union(z)) { absorbed = true; break; }
      }
      if (!absorbed) settled.push_back(std::move(z));
    }
    out.swap(settled);
  }
  f.swap(out);
}



/// @name Confined backward passes
/// @{
/// The backward fixpoint below is always run *confined* to a node set @c S: nodes outside @c S
/// are pinned to the empty federation. The limit of the confined iteration is a post-fixpoint
/// of the operator on the finished graph, and by Knaster–Tarski the post-fixpoints of a
/// greatest fixpoint are exactly its under-approximations (`X ⊆ F(X) ⇒ X ⊆ νF`). A non-empty
/// value at a `p /\ !q` node of @c S is therefore already a sound violated verdict, even though
/// the graph outside @c S has never been looked at.
///
/// @warning A node may enter @c S only if it is expanded. The stopping term ::LivenessEngine::stop_set
///          reads the *absence* of successors (`adj[v].empty()`), and an unexpanded node has an
///          empty adjacency for the wrong reason. Every other term reads only edges that exist,
///          so it survives a partial graph unchanged.
///
/// @note Intermediate iterates of a descending greatest-fixpoint iteration are only
///       *pre*-fixpoints, hence over-approximations, hence conclusive in neither direction.
///       Only the converged limit may be read; see ::LivenessEngine::certificate_in.
///
/// @see @c docs/design.md, "On-the-fly liveness", for the full argument.
/// @}


/// @brief Value equality for ::GfpTag, so the tag pool can deduplicate.
bool tag_equal(const GfpTag& a, const GfpTag& b) {
  if (a.parts.size() != b.parts.size() || a.abstain.size() != b.abstain.size()) return false;
  for (std::size_t i = 0; i < a.parts.size(); ++i)
    if (a.parts[i] != b.parts[i]) return false;
  for (std::size_t i = 0; i < a.abstain.size(); ++i) {
    const Constraint& x = a.abstain[i];
    const Constraint& y = b.abstain[i];
    if (x.i != y.i || x.j != y.j || x.bound != y.bound) return false;
  }
  return true;
}

/// @brief Hash of a ::GfpTag's value, keyed on the same fields ::tag_equal compares.
std::uint64_t tag_hash(const GfpTag& t) {
  std::uint64_t h = 1469598103934665603ULL;
  const auto mix = [&h](std::uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  };
  for (const auto& pr : t.parts) { mix(std::uint64_t(std::uint32_t(pr.first))); mix(std::uint64_t(std::uint32_t(pr.second))); }
  mix(0xabcdULL);
  for (const auto& c : t.abstain) {
    mix(std::uint64_t(std::uint32_t(c.i)));
    mix(std::uint64_t(std::uint32_t(c.j)));
    mix(std::uint64_t(std::uint32_t(c.bound)));
  }
  return h;
}

bool AtomC::operator==(const AtomC& o) const {
  return i == o.i && j == o.j && b == o.b;
}

/// @brief Hash a discrete location/integer key without materialising its byte representation.
/// Defined beside the ordinary passed-set implementation below.
std::uint64_t disc_hash(const LocVec& locs, const IntVec& ints);

/// @brief Avalanche a packed discrete key for the flat open-addressing table.
std::uint64_t mix_packed_key(std::uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

}  // namespace hora
