/// @file netliveness.cpp
/// @brief The liveness engine: `p --> q` (leads-to) by a backward greatest fixpoint.
///
/// This translation unit implements ptctlmc::leads_to_otf, declared in
/// @ref ptctlmc/netreach.hpp and reached from the query router. It is the only
/// leads-to engine in the build.
///
/// `p --> q` asks whether every reachable state satisfying @c p inevitably reaches a state
/// satisfying @c q on all maximal runs. Its negation is `E<> (p /\ EG !q)`, and @c EG is a
/// greatest fixpoint, so the engine builds the reachable zone graph forward and then solves
///
///     W  =  W  /\  ( NoSucc  \/  Pre(W) )
///
/// backward, one @ref ptctlmc::Federation per node. A non-empty value at a node satisfying
/// `p /\ !q` is a counterexample.
///
/// @par Relationship to the forward engine
/// This file shares no mutable state with @ref netreach.cpp and calls into it only through
/// ptctlmc::network_has_diagonal and ptctlmc::deadlock_reachable, both externally visible. The
/// transition semantics the two engines must agree on live in
/// @ref ptctlmc/detail/net_semantics.hpp and are inline there, so each translation unit gets
/// its own inlinable copy.
///
/// @par Design notes
/// The correctness arguments behind the choices made here — why a partial backward pass is
/// already a certificate, why stored zones must be exact, why the deadlock completion works,
/// and what was measured and rejected — are collected in @c DESIGN.md at the repository root.

#include "ptctlmc/netreach.hpp"
#include "ptctlmc/reach.hpp"

#ifdef PTCTLMC_HAVE_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <cstdint>
#include <deque>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ptctlmc/fed.hpp"
#include "ptctlmc/mindbm.hpp"
#include "ptctlmc/zone_store.hpp"
#include "ptctlmc/detail/sym_state.hpp"
#include "ptctlmc/detail/net_semantics.hpp"

namespace ptctlmc {
namespace {
using detail::SymState;
using detail::LocVec;
using detail::IntVec;
// The shared transition semantics (detail/net_semantics.hpp).
using detail::ExNone;
using detail::ExLuPlus;
using detail::ExNoDiag;
using detail::apply_all;
using detail::int_ok;
using detail::all_invariants;
using detail::frozen_flags;
using detail::reset_inactive;
using detail::close_succ_zone;
using detail::disc_key_bytes;
using detail::RecvOpt;
using detail::build_broadcast_parts;
using detail::state_has_deadlock;
}  // namespace

/// @brief The set of edges that jointly produced one transition of the network.
///
/// One entry for a local edge, two for a binary synchronisation, and `1 + n` for a broadcast
/// (the sender plus every receiver that fired). Keeping all three shapes in one list is what
/// lets ::gfp_pre have a single code path.
///
/// @invariant Every `(process, edge)` pair in @ref parts was enabled in the source state, and
///            the transition's effect is the composition of their guards, resets and updates.
struct GfpTag {
  std::vector<std::pair<int,int>> parts;   ///< (process index, edge index) of each participant

  /// @brief Guard complements contributed by broadcast receivers that did *not* fire.
  ///
  /// A receiver that abstains still constrains the transition: it stays put precisely because
  /// its own clock guard fails, so the step is enabled only where the *complement* of that
  /// guard holds. @ref parts cannot express this — an abstaining receiver contributes no edge —
  /// so the constraints ride here. This is the backward-direction counterpart of
  /// ptctlmc::detail::build_broadcast_parts.
  ///
  /// @invariant These are a pure guard on the *source* valuation: no reset, no location change
  ///            and no integer update. ::gfp_pre may therefore apply them at exactly the point
  ///            it applies the participants' guards, and nowhere else.
  /// @note Empty for local edges, for binary synchronisation, and for every broadcast whose
  ///       receivers all carry unguarded `a?` edges.
  std::vector<Constraint> abstain;
};

/// @brief Generate the successors of @p s, each tagged with the edges that produced it.
///
/// Mirrors the forward engine's successor generation for local edges, binary synchronisation
/// and broadcast, and additionally reports which edges produced each successor so that
/// ::gfp_pre can invert the step exactly.
///
/// @param net    the network under analysis
/// @param s      the source symbolic state
/// @param L,U    per-clock lower/upper bounds for extrapolation
/// @param qmask  clocks the query forces to stay live (see ptctlmc::detail::reset_inactive)
/// @param emode  extrapolation mode: ptctlmc::detail::ExNone, ExLuPlus or ExNoDiag
/// @param[out] out   successors, appended
/// @param[out] tags  one tag per appended successor
///
/// @post `out.size()` and `tags.size()` grow by the same amount, and `tags[k]` describes
///       `out[k]`.
/// @post No successor with an empty zone is emitted.
static void gfp_succ_tagged(const Network& net, const SymState& s,
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
      const bool i_comm = net.procs[i].locations[s.locs[i]].committed;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        for (int ej : net.procs[j].edges_of(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan != se.sync_chan || re.sync_dir != -1) continue;
          if (src_committed && !(i_comm || net.procs[j].locations[s.locs[j]].committed)) continue;
          if (!int_ok(re.int_guard, s.ints) || !gen_ok(net, re, s.ints.data(), s.ints.size())) continue;
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

  // ---- broadcast: a! fires together with EVERY enabled a? (0 or more receivers) ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_of(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1 || !net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints) || !gen_ok(net, se, s.ints.data(), s.ints.size())) continue;
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
///          value would pass the `!pz.empty()` filter in ::Otf::gfp_on and enter a federation
///          of `nclocks + 1`-dimensional zones, where the next inclusion test reads
///          `dim * dim` entries out of a 4-byte allocation.
static Dbm gfp_empty(int nclocks) {
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
///               down (the delay BEFORE the action) ; /\ inv(src)
///
/// @param net  the network under analysis
/// @param t    the tag produced for this transition by ::gfp_succ_tagged
/// @param src  the source symbolic state
/// @param tgt  the target symbolic state
/// @param X    a set of target valuations
/// @return the exact set of source valuations that can reach @p X by this transition, or an
///         empty zone of dimension `net.nclocks + 1` if there are none
///
/// @pre @p t, @p src and @p tgt come from the same call to ::gfp_succ_tagged.
/// @post No extrapolation and no active-clock reduction is applied in the forward direction:
///       the result is the exact concrete predecessor. Abstraction enters the fixpoint only
///       through the caller's confinement `W_v subseteq Z_v`.
static Dbm gfp_pre(const Network& net, const GfpTag& t, const SymState& src, const SymState& tgt,
                   const Dbm& X) {
  Dbm Y = X;

  // Undo the active-clock reduction at the target. Those clocks appear in no edge's reset set,
  // so without this Pre would silently demand that they also read 0 at the SOURCE -- an
  // over-constrained Pre that collapses the fixpoint and misses violations. Freeing them is the
  // exact inverse of the reduction, and it is sound because they are dead: nothing reads them.
  if (!net.active.empty()) {
    for (int c = 1; c <= net.nclocks; ++c) {
      bool live = false;
      for (int pr = 0; pr < net.nprocs() && !live; ++pr)
        if (net.active[pr][tgt.locs[pr]][c]) live = true;
      if (!live) Y.free(c);
    }
  }

  std::vector<Constraint> inv;
  all_invariants(net, tgt.locs, inv);
  bool nc, nu; frozen_flags(net, tgt.locs, tgt.ints, &nc, &nu);
  if (!apply_all(Y, inv)) return gfp_empty(net.nclocks);
  if (!nc && !nu) { Y.down(); if (!apply_all(Y, inv)) return gfp_empty(net.nclocks); }

  // Undo every participant's resets: a reset clock reads 0 after the step, and its value before
  // the step is unconstrained by the step itself.
  for (const auto& pe : t.parts)
    for (int r : net.procs[pe.first].edges[pe.second].clock_resets) {
      if (!Y.constrain(r, 0, le(0))) return gfp_empty(net.nclocks);
      if (!Y.constrain(0, r, le(0))) return gfp_empty(net.nclocks);
    }
  for (const auto& pe : t.parts)
    for (int r : net.procs[pe.first].edges[pe.second].clock_resets) Y.free(r);

  for (const auto& pe : t.parts)
    if (!apply_all(Y, net.procs[pe.first].edges[pe.second].clock_guard)) return gfp_empty(net.nclocks);
  // The abstaining receivers' guard complements belong here and nowhere else: like a
  // participant's guard they constrain the source valuation, and unlike one they undo nothing,
  // so they apply after the resets have been freed and before the pre-action delay.
  if (!t.abstain.empty() && !apply_all(Y, t.abstain)) return gfp_empty(net.nclocks);

  // Delay BEFORE the action. A stored zone is delay-closed as a SET, which is why the forward
  // operator may write `Z /\ g`; a single valuation may still have to let time pass first.
  std::vector<Constraint> sinv;
  all_invariants(net, src.locs, sinv);
  bool sc, su; frozen_flags(net, src.locs, src.ints, &sc, &su);
  if (!sc && !su) { Y.down(); }
  if (!apply_all(Y, sinv)) return gfp_empty(net.nclocks);
  return Y;
}

/// @brief ::gfp_pre for a run confined to a NON-CONVEX predicate.
///
/// @param avoid the forbidden region; must be non-null
/// @param[in,out] out  the pieces of the predecessor are appended
///
/// @par Why this is federation-valued
/// ::gfp_pre takes two time-predecessors, one at the target and one at the source. With a region
/// to avoid, each of them routes around it and each can therefore SPLIT, so the predecessor of a
/// single zone is in general several zones. Collapsing them to one -- by hull, or by keeping the
/// largest -- under-approximates `Pre`, and `Pre` occurs positively in the fixpoint, so that
/// silently loses witnesses: it turned `E[] x <= 2 || x >= 8` on a model whose witness is an
/// infinite zero-time loop from satisfied into NOT satisfied.
///
/// @note Every step other than the two time-predecessors maps a zone to a zone, so the federation
///       is threaded through mechanically and only the two routed steps branch.
static void gfp_pre_avoid(const Network& net, const GfpTag& t, const SymState& src,
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
static bool gfp_fed_empty(const Federation& f) {
  for (const auto& z : f) if (!z.empty()) return false;
  return true;
}

/// @brief Is every point of @p a inside the union @p b?
static bool gfp_fed_subset(const Federation& a, const Federation& b) {
  for (const auto& z : a) {
    if (z.empty()) continue;
    if (!fed_subtract(z, b).empty()) return false;
  }
  return true;
}
/// @brief Normalise a federation in place: drop empties and redundant members, merge convex pairs.
///
/// Drops empty members, drops members already covered by one that is kept, and merges any pair
/// whose union is *exactly* convex.
///
/// @param f            the federation to compact, modified in place
/// @param cap          maximum number of members to keep
/// @param[in,out] capped  set to @c true if @p cap was exceeded; never cleared
///
/// @post The set denoted by @p f is unchanged unless `*capped` was set. Convex-union merging
///       is exact in both directions — @c Dbm::try_union succeeds only when the union really is
///       a single zone — so it adds no point and removes none.
/// @warning The caller must treat `*capped` as sticky and must not read @p f as a converged
///          value once it is set; see ::Otf::certificate_in.
/// @note Without the convex-union merge the federations grow monotonically with every @c Pre
///       and the width cap becomes the binding constraint on the whole algorithm.
static void gfp_fed_compact(Federation& f, std::size_t cap, bool* capped) {
  Federation out;
  for (auto& z : f) {
    if (z.empty()) continue;
    bool absorbed = false;
    for (auto& o : out) {
      if (o.includes(z)) { absorbed = true; break; }          // already covered
      if (o.try_union(z)) { absorbed = true; break; }         // exactly convex union: merge
    }
    if (!absorbed) out.push_back(std::move(z));
    if (out.size() > cap) { *capped = true; break; }
  }
  // One settling pass: a merge can make an earlier member redundant against a later one.
  if (!*capped && out.size() > 1) {
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
/// value at a `p /\ !q` node of @c S is therefore already a sound VIOLATED verdict, even though
/// the graph outside @c S has never been looked at.
///
/// @warning A node may enter @c S only if it is EXPANDED. The stopping term ::Otf::stop_set
///          reads the *absence* of successors (`adj[v].empty()`), and an unexpanded node has an
///          empty adjacency for the wrong reason. Every other term reads only edges that exist,
///          so it survives a partial graph unchanged.
///
/// @note Intermediate iterates of a descending greatest-fixpoint iteration are only
///       *pre*-fixpoints, hence over-approximations, hence conclusive in neither direction.
///       Only the converged limit may be read; see ::Otf::certificate_in.
///
/// @see @c DESIGN.md, "On-the-fly liveness", for the full argument.
/// @}


/// @brief One outgoing edge of the liveness graph: a target node and the tag that produced it.
struct OtfEdge {
  int tgt;      ///< index into ::Otf::node
  int tag;      ///< index into ::Otf::tags_, the deduplicated tag pool
};

/// @brief Value equality for ::GfpTag, so the tag pool can deduplicate.
inline bool gfp_tag_eq(const GfpTag& a, const GfpTag& b) {
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

/// @brief Hash of a ::GfpTag's value, keyed on the same fields ::gfp_tag_eq compares.
inline std::uint64_t gfp_tag_hash(const GfpTag& t) {
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

/// @name The deadlock-completion constraint map
/// @{
/// a_LU covering is a *simulation*, and deadlock — "this valuation enables nothing" — is a
/// universally quantified negative property that simulation does not preserve
/// (Behrmann–Bouyer–Larsen–Pelánek, STTT 8(3):204–215, 2006, Lemma 3). The completion repairs
/// that without weakening the covering: conceptually add a sink with an edge guarded by
/// `Dead(q)` at every discrete key `q`, so a maximal run that *ends* in a deadlock becomes a
/// run that *reaches* the sink — an existential property every simulation preserves. The sink
/// is never materialised. Its only effect is on which atoms the covering test must carry,
/// namely the atoms of `Dead(q)`, closed under pre.
///
/// An atom is a half-space `x_i - x_j <| b`, and the two kinds are discharged differently:
///
/// - **Diagonal** (`i > 0` and `j > 0`) — invariant under time elapse, so agreement on it
///   survives delay and it can ride as a *cell tag* on the discrete key (::AtomMap::diag).
///   Zones the atom classifies differently are never allowed to cover one another, and the
///   covering kernel itself is untouched.
/// - **Non-diagonal** (`i == 0` or `j == 0`, i.e. a bound on one clock) — *not* delay
///   invariant, so no cell tag can carry it. The a_LU preorder carries it instead: with
///   `L(x) >= c` and `U(x) >= c` its three clauses collapse to "equal, or both above `c`",
///   which decides every constraint on `x` with constant `<= c` the same way for both
///   valuations. Such an atom is therefore discharged by *raising that one clock's bounds*
///   (::AtomMap::bump) — a_M on that clock alone rather than a_M on the whole model.
///
/// With both in place the relation `ν R ν'  iff  ν <=_LU ν'` (raised bounds) *and* `ν, ν'`
/// agree on the diagonal atoms is a simulation for the completed automaton: `<=_LU` is a
/// simulation for the original automaton, diagonal agreement is preserved by delay and — via
/// the pre-closure in ::Otf::build_atom_map — by every edge, and `Dead(q)` is a boolean
/// combination of `Inv(q)` (which every stored zone satisfies) with the diagonal and
/// non-diagonal atoms, so `ν |= Dead(q)` iff `ν' |= Dead(q)`.
///
/// @see @c DESIGN.md, "The deadlock completion", for why `Dead(q)` is a diagonal set even in a
///      diagonal-free automaton, why no per-clock `(L,U)` scheme can preserve deadlock on its
///      own, and why each of the three seed drops is safe.
/// @}

/// @brief One atom of the constraint map: the half-space `x_i - x_j <| b`.
struct AtomC {
  int i, j;                                  ///< clock indices; 0 is the reference clock
  raw_t b;                                   ///< packed bound, so the atom is `x_i - x_j <| b`
  bool operator==(const AtomC& o) const { return i == o.i && j == o.j && b == o.b; }
};

/// @brief Sentinel for ::AtomMap::bump: clock constants are >= 0, so -1 means "not raised".
static constexpr std::int32_t kNoBump = -1;

/// @brief The refinement of a_LU that makes the covering test deadlock-preserving.
///
/// Produced by ::Otf::build_atom_map from a finished forward graph and consumed by a second
/// ::Otf run through ::Otf::set_atoms.
///
/// @invariant @ref diag holds at most 64 atoms per key, because the per-node classification is
///            a @c uint64_t bitmask; ::Otf::build_atom_map returns @c false rather than exceed
///            it.
/// @invariant Every clock read by an atom of @ref diag, and every clock raised in @ref bump, is
///            marked in @ref live.
struct AtomMap {
  std::unordered_map<std::string, std::vector<AtomC>> diag;  ///< discrete key -> cell-tag atoms
  std::vector<std::int32_t> bump;   ///< clock -> constant L(x),U(x) must reach (::kNoBump = none)
  std::vector<char> live;           ///< clocks the atoms read: reset_inactive must not zero them
  bool usable = false;              ///< false if the construction declined; fall back to the race

  // ---- instrumentation; none of it affects a verdict ----
  std::size_t seed_keys = 0, seed_atoms = 0, nondiag_atoms = 0, closure_atoms = 0;
  std::size_t keys_tagged = 0, rounds = 0, bumped = 0;
  std::size_t dropped_vacuous = 0;    ///< escape regions that miss Inv(q) entirely
  std::size_t dropped_disjoint = 0;   ///< atoms Inv(q) refutes, so their tag bit is constantly 0
};

/// @brief One `p --> q` search: forward zone-graph construction plus a backward greatest fixpoint.
///
/// A single instance performs one search under one covering discipline. ::leads_to_otf may
/// build several: one to derive the constraint map, one to search under it, and — only when the
/// map cannot be represented — one exact-covering lane raced against a deadlock-freedom proof.
class Otf {
 public:
  /// @param n            the network under analysis
  /// @param pp,qq        the leads-to predicates; clock constraints are not permitted in either
  /// @param force_exact  use plain inclusion covering instead of a_LU (sound outright, larger)
  /// @param cancel       polled during the search; when it is set the run abandons itself
  Otf(const Network& n, const DiscPred& pp, const DiscPred& qq,
      bool force_exact = false, const std::atomic<bool>* cancel = nullptr,
      const Network* dead = nullptr)
      : net(n), p(pp), q(qq), force_exact_(force_exact), cancel_(cancel),
        dead_net(dead ? dead : &n) {}

  /// @brief Build the graph and decide the query.
  /// @return @c true iff `p --> q` holds
  /// @warning The result is meaningless if ::was_cancelled() is true afterwards.
  bool run(GfpStats* gs);

  /// @brief Decide `E[] psi` from the initial state, with `psi`'s discrete part as `!q`.
  /// @return @c true iff the initial state admits a maximal `psi`-execution
  /// @warning Meaningless if ::was_capped() is true afterwards; the caller retries exactly.
  bool eg_from_initial(GfpStats* gs);

  /// @brief The `E[] psi` verdict on a graph ::Otf::build_forward has already built.
  /// @note The counterpart of ::Otf::finish_verdict, which does the same for `p --> q`: it pays
  ///       the backward pass that ::Otf::build_forward deliberately deferred.
  bool eg_verdict(GfpStats* gs);

  /// @brief Drop the forward graph and everything indexed by it.
  ///
  /// After this the instance may not be run, finished or queried again; only destruction is
  /// legal. It exists because ::leads_to_otf builds a SECOND engine over the same network once
  /// the constraint map is in hand, and the first one's graph — nodes, zones, adjacency and the
  /// discrete-key buckets — is dead from that point on but would otherwise stay resident for the
  /// whole of the second search, doubling the peak. The map itself owns its atoms, so nothing
  /// released here is reachable from it.
  void release() {
    std::vector<SymState>().swap(node);
    std::vector<std::vector<OtfEdge>>().swap(adj);
    std::unordered_map<std::uint64_t, std::vector<int>>().swap(bucket);
    std::vector<GfpTag>().swap(tags_);
    zs_.reset(zdim_);
    std::unordered_map<std::uint64_t, std::vector<int>>().swap(tagidx_);
    std::vector<char>().swap(expanded);
    std::vector<char>().swap(pc);
    std::vector<char>().swap(qc);
    std::vector<std::uint64_t>().swap(ntag_);
    std::vector<std::uint64_t>().swap(ntag_out_);
  }

  /// @brief Run the forward half only, leaving the backward fixpoint unpaid.
  ///
  /// ::build_atom_map reads only the finished forward *graph* — the reachable discrete keys and
  /// the key-edge relation with their resets — and if the resulting map is non-empty the whole
  /// run is discarded in favour of a fresh search under the refined covering. Splitting the
  /// phases lets the caller decide the map first and pay the fixpoint exactly once: through
  /// ::finish_verdict when the map came out empty, or inside the refined search otherwise.
  ///
  /// @post ::build_atom_map and ::dtheta_empty may be called; ::finish_verdict must be called
  ///       before any verdict is published.
  void build_forward(GfpStats* gs);

  /// @brief Pay the backward fixpoint deferred by ::build_forward and return the verdict.
  /// @pre ::build_forward has run on this instance.
  /// @return @c true iff `p --> q` holds
  bool finish_verdict(GfpStats* gs);

  /// @brief Structural certificate: is the deadlock completion's seed empty at every reachable key?
  ///
  /// @pre The forward graph has been built (::run or ::build_forward).
  /// @return @c true iff plain a_LU covering was already deadlock-sound for this model, so this
  ///         run's verdict may be published as it stands
  bool dtheta_empty();

  /// @brief Did this run abandon its exploration part-way?
  /// @note A cancelled run's verdict is not a verdict. A racing caller must consult this before
  ///       publishing a result.
  bool was_cancelled() const { return cancelled_; }
  /// @brief Did the federation width cap fire? A verdict from a capped run is not a verdict.
  bool was_capped() const { return capped; }
  /// @brief Install the forbidden region of a non-convex `psi`. Must precede ::Otf::run.
  ///
  /// @param a        the region; null clears it
  /// @param forward  whether the FORWARD pipeline is restricted by it too
  ///
  /// `E[] psi` sets @p forward, because there the whole run must stay inside `psi` and a state
  /// outside it is not reachable for the query at all. `p --> q` must NOT: it is violated by
  /// `EF(p /\ EG !q)`, and the path to the `p`-state is ordinary reachability that need not stay
  /// in `!q`. There the region confines only the fixpoint -- the seed, `Pre`, and the stop set.
  void set_avoid(const detail::DelayAvoidSet* a, bool forward = true) {
    avoid_ = a;
    avoid_forward_ = forward;
  }

  /// @brief Derive the deadlock-completion constraint map from this run's finished graph.
  /// @pre The forward graph has been built (::run or ::build_forward).
  /// @param[out] am  the map; `am.usable` distinguishes "built" from "declined"
  /// @return @c false if the construction declined (more atoms than the tag can hold, or the
  ///         pre-closure failed to converge); the caller must then fall back
  bool build_atom_map(AtomMap& am);

  /// @brief Install a constraint map, to be used by the next ::run on this instance.
  /// @pre No search has been started on this instance yet.
  void set_atoms(const AtomMap* a) { amap_ = a; }

 private:
  const Network& net;
  const DiscPred& p;
  const DiscPred& q;
  /// @brief The network deadlock is judged against; see ::Otf::stop_set. Equals @ref net unless
  ///        `E[] psi` supplied the unrestricted network alongside the psi-restricted one.
  const Network* dead_net;
  /// @brief The forbidden region of a non-convex `psi`, or null. Used by the successor pipeline
  ///        through ptctlmc::detail::close_succ_zone and, backward, by ::gfp_pre.
  const detail::DelayAvoidSet* avoid_ = nullptr;
  bool avoid_forward_ = true;   ///< see ::Otf::set_avoid
  /// @brief Iterate the fixpoint component-wise; see ::Otf::gfp_on. Set by the `E[]` path only.
  bool scc_iter_ = false;
  bool force_exact_ = false;                   ///< inclusion covering, no a_LU
  const std::atomic<bool>* cancel_ = nullptr;  ///< set by the winner of the race
  bool cancelled_ = false;

  /// @brief Poll the cancellation flag, latching it.
  /// @post Once this has returned @c true it returns @c true forever, so nothing downstream of
  ///       the first observation can see the flag flip back and resume a half-abandoned search.
  bool cancelled() {
    if (cancelled_) return true;
    if (cancel_ && cancel_->load(std::memory_order_relaxed)) cancelled_ = true;
    return cancelled_;
  }

  std::vector<std::int32_t> L, U, aL, aU, bL, bU;
  int emode = 0;
  std::vector<char> qmask;
  bool use_alu = false;
  static constexpr std::size_t fedcap = 4096;
  bool capped = false;
  std::size_t passes = 0, gfp_iters = 0, expand_calls = 0, pre_calls = 0;
  std::size_t fed_max = 0, nonempty = 0;
  double fed_avg = 0;
  bool settled_early = false;
  /// @brief Outer-loop accounting, kept separate so counts compare like-for-like with UPPAAL.
  ///
  /// UPPAAL's leads-to counters report only the outer reachability up to the first violating
  /// p-state; the nested inevitability check is not counted. This engine interns every state
  /// including the backward pass's, so the two are tallied apart.
  std::size_t outer_expands = 0, outer_nodes = 0;

  std::vector<SymState> node;                  ///< the forward graph's vertices
  std::vector<std::vector<OtfEdge>> adj;       ///< tagged out-edges, indexed like ::node
  // THE TAG POOL. A tag names the edges that fire on a transition, so it is a property of the
  // *transition*, not of the (source, target) pair: the same participant set fires out of every
  // state that offers it, and on these models the graph carries ~8 edges per node against a few
  // hundred distinct tags in total. Stored by value, a GfpTag is two std::vectors -- 48 bytes of
  // headers per edge plus a heap block for `parts`, and `abstain` is empty for every edge that
  // is not a broadcast with guarded receivers. Interned, an edge costs one int.
  std::vector<GfpTag> tags_;                   ///< the pool; ::OtfEdge::tag indexes it
  std::unordered_map<std::uint64_t, std::vector<int>> tagidx_;  ///< hash -> candidate pool slots
  // KEYED BY HASH, NOT BY THE KEY. The discrete state is already stored, once, in
  // node[i].locs/ints; keeping a std::string copy of it as the map key stored it a second time
  // and cost 25 MB of the 178 MB live on arbiter14 -- one heap string plus one map node per
  // distinct discrete state. The map is keyed by a hash of (locs, ints) instead, and every
  // candidate is checked for exact discrete equality against the stored node before any zone
  // comparison, so a collision costs one extra vector compare and never merges two states.
  //
  // @note ::GfpStats::disc_keys therefore counts occupied hash buckets. Collisions make it a
  //       lower bound on the number of distinct discrete states, not an exact count.
  std::unordered_map<std::uint64_t, std::vector<int>> bucket;  ///< disc-state hash -> node indices
  std::vector<char> expanded;                  ///< 1 once ::expand has computed the out-edges
  std::vector<char> pc, qc;                    ///< p/q predicate caches; 2 = not yet evaluated
  bool intern_grew = false;                    ///< last ::intern rewrote a stored zone

  // THE STORED ZONES. ptctlmc::ZoneStore keeps them dense -- the covering predicates read the
  // stored operand at an arbitrary (i, j), which a minimal-graph record cannot answer without
  // re-running its O(dim^3) closure -- while adapting the ENTRY WIDTH to the bounds the model
  // actually produces: one byte, then two, then four, widening in place rather than refusing a
  // model or keeping a per-zone dense fallback. Every model in this suite fits one byte (the
  // widest stored bound measured is |raw| = 9).
  ptctlmc::ZoneStore zs_;
  int zdim_ = 0;                               ///< nclocks + 1

  /// @brief Store node @p i's zone, and drop the dense copy the node was carrying.
  void zstore(int i, Dbm&& z) {
    zs_.store(i, z);
    node[i].zone = Dbm();                      // the store owns it now
  }

  /// @brief Node @p i's zone, widened into @p sc.
  const Dbm& zread(int i, Dbm& sc) const {
    zs_.load(i, sc);
    return sc;
  }

  /// @brief Node @p i as a whole state; @p sc is reused, so its buffers survive between calls.
  const SymState& sread(int i, SymState& sc) const {
    sc.locs.assign(node[i].locs.begin(), node[i].locs.end());
    sc.ints.assign(node[i].ints.begin(), node[i].ints.end());
    zread(i, sc.zone);
    return sc;
  }

  // The covering tests, each instantiated against the stored operand at its ACTUAL entry type.
  // ZoneStore::with does the dispatch; it is the same kernel either way, so a narrow operand is
  // not an approximation of a wide one.
  bool alu_le_stored(const Dbm& z, int j) const {
    return zs_.with(j, [&](const auto* p) {
      return dbm_is_alu_le(z.data().data(), p, zdim_, bL.data(), bU.data());
    });
  }
  bool alu_le_new(int j, const Dbm& z) const {
    return zs_.with(j, [&](const auto* p) {
      return dbm_is_alu_le(p, z.data().data(), zdim_, bL.data(), bU.data());
    });
  }
  bool stored_includes(int j, const Dbm& z) const {
    return zs_.with(j, [&](const auto* p) { return dbm_includes(p, z.data().data(), zdim_); });
  }
  bool includes_stored(const Dbm& z, int j) const {
    return zs_.with(j, [&](const auto* p) { return dbm_includes(z.data().data(), p, zdim_); });
  }

  const AtomMap* amap_ = nullptr;              ///< null = ordinary a_LU, no cells

  /// @brief Per-node "In" mask: bit @c a is set iff the node's zone lies wholly inside atom @c a.
  /// @note Meaningful only when ::amap_ is installed and the node's key carries atoms; 0 otherwise.
  std::vector<std::uint64_t> ntag_;
  /// @brief Per-node "Out" mask: bit @c a is set iff the node's zone is disjoint from atom @c a.
  /// @note Stored zones are not atom-homogeneous, so one bit cannot classify them. Together the
  ///       two masks give the three-valued classification In / Out / Straddle (neither bit set);
  ///       see ::Otf::classify_atoms.
  std::vector<std::uint64_t> ntag_out_;

  void setup();
  bool holds(const DiscPred& pr, const SymState& s) const;
  bool is_p(int i);
  Federation p_region(int i);
  bool is_nq(int i);
  void grow();
  int intern(SymState&& s);
  int intern_tag(const GfpTag& t);
  void expand(int u);
  bool time_diverges(const SymState& s) const;
  Federation stop_set(int u);                  ///< exact NoSucc; requires `expanded[u]`
  bool orig_time_open(const SymState& s) const;  ///< can time run away in the UNRESTRICTED system?

  void gfp_on(const std::vector<int>& S, std::vector<Federation>& W);
  bool certificate_in(const std::vector<int>& S);
  std::vector<int> accepting_closure();

  void split_by_atoms(SymState&& s, std::vector<SymState>& out);
  void classify_atoms(const Dbm& z, const std::vector<AtomC>& as,
                      std::uint64_t* in, std::uint64_t* out) const;

  /// @brief The covering kernel without any atom obligation: a_LU, or plain inclusion.
  bool base_cover(const Dbm& a, const Dbm& b) const {
    return use_alu ? a.alu_subsumed_by(b, bL, bU) : b.includes(a);
  }
  bool cover_rec(const Dbm& a, std::uint64_t ain, const Dbm& b, std::uint64_t bout,
                 const std::vector<AtomC>& as, int depth);
  void base_forward();                         ///< the forward half; ::setup must have run
  void fill_stats(GfpStats* gs);
  bool fwd_ok_ = false;                        ///< ::build_all found a satisfiable initial state
  bool build_all();                            ///< seed the graph with the initial state
  bool final_verdict_violated();               ///< one exact pass over the finished graph
 public:
};

/// @brief Choose the covering discipline and the extrapolation mode for this run.
///
/// @post `use_alu` and `emode` are consistent: the search terminates, and every stored zone is
///       either exact (a_LU covering) or extrapolated (inclusion covering) — never both.
/// @post `qmask` marks every clock the constraint map reads, so ::reset_inactive keeps them.
///
/// @invariant **Stored zones are exact whenever a_LU covering is in force.** Finiteness of the
/// zone graph then comes from a_LU subsumption, not from extrapolation, and the two settings
/// must move together: plain inclusion does not bound the zone graph, so with a_LU off the only
/// thing that terminates the search is Extra_LU+, and with Extra_LU+ on every stored zone
/// over-approximates the reachable set. See DESIGN.md, "Stored zones must be exact", for the
/// counterexample; in short, ::Otf::stop_set asks whether some valuation of the stored zone
/// enables *nothing*, enabledness is monotone rather than invariant under extrapolation, and a
/// spurious stop valuation occurs positively in the fixpoint operator, so it fabricates a
/// counterexample rather than losing one.
///
/// @note Diagonal models cannot use a_LU at all (Bouyer 2004), so they keep ExNoDiag with
///       inclusion covering and extrapolation stays load-bearing there.
void Otf::setup() {
  zdim_ = net.nclocks + 1;
  zs_.reset(zdim_);
  L = net.lbound;
  U = net.ubound;
  use_alu = !force_exact_ && !network_has_diagonal(net);
  emode = network_has_diagonal(net) ? ExNoDiag : (use_alu ? ExNone : ExLuPlus);
  // EXPERIMENT ONLY (PTCTLMC_GFP_EXTRAP=1): keep a_LU covering but extrapolate the stored zones
  // as well, i.e. local Extra_LU+ PLUS subsumption. The invariant above says this is unsound, and
  // says why -- a spurious stop valuation enters the fixpoint positively and fabricates a
  // counterexample. The switch exists to measure the claim, not to be used.
  static const bool force_extrap = [] {
    const char* e = std::getenv("PTCTLMC_GFP_EXTRAP");
    return e && *e == '1';
  }();
  if (force_extrap && use_alu) emode = ExLuPlus;
  // Clocks the query forces to stay live. A leads-to predicate carries no clock constraints, so
  // this starts all-zero; reset_inactive reads it as "additionally keep alive", so all-zero is
  // MAXIMAL reduction, not none. The reduction is safe for a backward analysis because gfp_pre
  // frees every clock dead at the target before undoing the step.
  qmask.assign(net.nclocks + 1, 0);
  aL = net.lbound;
  aU = net.ubound;
  aL[0] = aU[0] = 0;
  // The constraint map has two global effects (see ::AtomMap):
  //
  // (1) A non-diagonal atom is discharged by raising its clock's L and U. Raising either only
  //     ever makes a_LU FINER, so it is sound wherever applied; applying it to the global
  //     arrays here and to the per-location arrays in Otf::intern keeps it consistent along
  //     every path, which is what the simulation argument needs.
  // (2) reset_inactive pins clocks no guard will read again to 0, but the atoms DO read them --
  //     that is the point of Dead(q) -- so every clock the map mentions joins qmask. Without
  //     this the reduction erases exactly the differences the cell tag exists to observe.
  //
  // The pinning in (2) is kept for seed atoms as well as closure atoms even though seed clocks
  // are arguably active at their own key by construction: `active` is a static
  // over-approximation, so the argument is not a proof and the conservative choice stands.
  if (amap_) {
    for (int c = 1; c <= net.nclocks && c < (int)amap_->bump.size(); ++c) {
      if (amap_->bump[c] != kNoBump) {
        aL[c] = std::max(aL[c], amap_->bump[c]);
        aU[c] = std::max(aU[c], amap_->bump[c]);
      }
      if (amap_->live[c]) qmask[c] = 1;
    }
  }
}

bool Otf::holds(const DiscPred& pr, const SymState& s) const {
  // A predicate carrying a DNF denotes the DISJUNCTION of its terms, not a conjunction. Only the
  // E[] entry point builds one, and only ever from discrete literals: it needs `!psi_discrete`,
  // which for an arbitrary boolean `psi` is a disjunction and has no conjunctive form.
  if (pr.is_dnf || !pr.forbid.empty()) {
    for (const auto& t : pr.forbid) {
      // A term constraining a clock forbids only part of the vertex, so it cannot exclude the
      // vertex outright; detail::DelayAvoidSet carries it and cuts the zone instead.
      if (!t.clocks.empty()) continue;
      bool all = true;
      for (const auto& r : t.locs)
        if ((s.locs[r.proc] == r.loc) != r.equal) { all = false; break; }
      if (all && !int_ok(t.ints, s.ints)) all = false;
      if (all) return true;
    }
    return false;
  }
  for (const auto& r : pr.locs) {
    const bool eq = (s.locs[r.proc] == r.loc);
    if (eq != r.equal) return false;
  }
  return int_ok(pr.ints, s.ints);
}

void Otf::grow() {
  const std::size_t n = node.size();
  if (expanded.size() < n) expanded.resize(n, 0);
  if (pc.size() < n) pc.resize(n, 2);
  if (qc.size() < n) qc.resize(n, 2);
  if (adj.size() < n) adj.resize(n);
  if (ntag_.size() < n) ntag_.resize(n, 0);
  if (ntag_out_.size() < n) ntag_out_.resize(n, 0);
}

bool Otf::is_p(int i) {
  grow();
  if (pc[i] != 2) return pc[i] != 0;
  // The OVER-approximation: does some term's DISCRETE guard hold, so the node could contain a
  // p-valuation at all? Seeding the accepting closure wide is harmless, because the verdict is
  // decided by the exact region test in ::Otf::certificate_in.
  //
  // ::Otf::holds is deliberately not reused. Its DNF branch SKIPS a term that constrains a clock,
  // which is right for `q` -- there the clock part is carried by the forbidden region and a term
  // cannot exclude a whole vertex -- and exactly wrong for `p`, where such a term is the reason
  // the node is interesting. Reusing it made `x >= 0 --> P.wait` hold vacuously: no term counted,
  // so no node seeded the closure, so no violation could be found.
  bool ok;
  if (p.is_dnf || !p.forbid.empty()) {
    ok = false;
    for (const auto& t : p.forbid) {
      bool all = true;
      for (const auto& r : t.locs)
        if ((node[i].locs[r.proc] == r.loc) != r.equal) { all = false; break; }
      if (all && !int_ok(t.ints, node[i].ints)) all = false;
      if (all) { ok = true; break; }
    }
  } else {
    ok = holds(p, node[i]);
  }
  pc[i] = ok ? 1 : 0;
  return pc[i] != 0;
}

/// @brief The region of `p` at node @p i, as a federation; empty when `p` cannot hold there.
///
/// `p` in DNF is a union of conjunctive terms, each a discrete guard plus a convex clock region.
/// At a fixed node the discrete guards are decided, so the region is the union of the clock parts
/// of the terms that survive.
///
/// @note Returns the whole zone when `p` constrains no clock, which is the pre-existing case and
///       makes the intersection in ::Otf::certificate_in a no-op there.
Federation Otf::p_region(int i) {
  Federation out;
  SymState cst;
  const SymState& sv = sread(i, cst);
  if (p.forbid.empty()) { out.push_back(sv.zone); return out; }
  for (const auto& t : p.forbid) {
    bool guard = true;
    for (const auto& r : t.locs)
      if ((sv.locs[r.proc] == r.loc) != r.equal) { guard = false; break; }
    if (guard && !int_ok(t.ints, sv.ints)) guard = false;
    if (!guard) continue;
    Dbm z = sv.zone;
    if (!apply_all(z, t.clocks) || z.empty()) continue;
    fed_add(out, z);
  }
  return out;
}

bool Otf::is_nq(int i) {
  grow();
  if (qc[i] == 2) qc[i] = holds(q, node[i]) ? 1 : 0;
  return qc[i] == 0;
}

/// @brief Classify a zone against each atom of its key: inside, disjoint, or straddling.
///
/// Stored zones are kept whole rather than cut along the atoms, so a zone may *straddle* an
/// atom. One bit per atom cannot express that, and reading a straddle as "disjoint from phi" is
/// the one mistake that makes the covering test unsound, so the classification is three-valued
/// and rides in two masks. Both bits come off the canonical DBM in O(1):
///
///     Z.get(i,j) <= b                   =>  Z subseteq phi        (In)
///     raw_add(Z.get(j,i), b) < kZeroLe  =>  Z /\ phi = empty      (Out)
///     otherwise                                                   (Straddle)
///
/// @param z   the zone to classify
/// @param as  the atoms of @p z's discrete key; at most 64 are read
/// @param[out] in   bit @c a set iff @p z lies wholly inside atom @c a
/// @param[out] out  bit @c a set iff @p z is disjoint from atom @c a
///
/// @pre @p z is canonical (closed).
/// @post `(*in & *out) == 0`; a bit set in neither mask means @p z straddles that atom.
void Otf::classify_atoms(const Dbm& z, const std::vector<AtomC>& as,
                         std::uint64_t* in, std::uint64_t* out) const {
  std::uint64_t i_ = 0, o_ = 0;
  const std::size_t n = std::min<std::size_t>(as.size(), 64);
  for (std::size_t a = 0; a < n; ++a) {
    const std::uint64_t bit = std::uint64_t(1) << a;
    if (z.get(as[a].i, as[a].j) <= as[a].b) { i_ |= bit; continue; }
    if (raw_add(z.get(as[a].j, as[a].i), as[a].b) < kZeroLe) o_ |= bit;
  }
  *in = i_;
  *out = o_;
}

/// @brief The covering test refined by the constraint map's atoms.
///
/// `a` is covered by `b` when, for every valuation @c nu in @p a, there is a @c nu' in @p b
/// with `nu <=_LU nu'` **and**, for every atom @c phi of the key, `nu' |= phi => nu |= phi`.
/// That implication direction only: it is what stops a_LU covering from dropping a deadlock. If
/// the covering zone satisfies an escape atom and the covered one does not, the simulator
/// escapes where the simulated valuation is stuck, and the deadlock is lost.
///
/// An atom is discharged for all pairs at once exactly when @p b is Out (no @c nu' satisfies
/// it) or @p a is In (every @c nu does). An atom that is neither is *blocked*, and only blocked
/// atoms are split. The recursion is Gastin–Mukherjee–Srivathsan, CAV 2019, Lemma 17, adapted
/// to the implication obligation rather than to agreement:
///
///     a <= b   <==   (a /\ phi <= b)  and  (a /\ ~phi <= b /\ ~phi)
///
/// `nu |= phi` discharges the phi-obligation for any @c nu' whatever, so the first conjunct may
/// keep the whole of @p b; `nu !|= phi` needs `nu' !|= phi`, so the second must narrow @p b as
/// well.
///
/// @param a,b    the candidate covered and covering zones
/// @param ain    the In-mask of @p a (see ::classify_atoms)
/// @param bout   the Out-mask of @p b
/// @param as     the atoms of the shared discrete key
/// @param depth  recursion depth, for diagnostics only
/// @return @c true iff @p a is covered by @p b under the refined relation
///
/// @pre @p a and @p b have the same discrete key, and @p as are that key's atoms.
/// @post Terminates in at most `2^|blocked|` leaves: each conjunct resolves one blocked atom
///       (In on the left, Out on both sides on the right), so the blocked set strictly shrinks.
/// @note ::base_cover is a *necessary* condition for both conjuncts — both test sub-zones of
///       @p a against sub-zones of @p b — so asking it first is sound and doubles as a
///       prefilter that rejects a hopeless probe before any zone is copied.
/// @note At depth 0 the two masks come from the caller's cache, so an unblocked probe costs one
///       mask operation plus the base test.
bool Otf::cover_rec(const Dbm& a, std::uint64_t ain, const Dbm& b, std::uint64_t bout,
                    const std::vector<AtomC>& as, int depth) {
  const std::size_t n = std::min<std::size_t>(as.size(), 64);
  const std::uint64_t all = n >= 64 ? ~std::uint64_t(0) : ((std::uint64_t(1) << n) - 1);
  const std::uint64_t blocked = all & ~bout & ~ain;
  if (!base_cover(a, b)) return false;   // necessary for both conjuncts; also the reject prefilter
  if (blocked == 0) return true;
  const int k = __builtin_ctzll(blocked);
  const AtomC& at = as[k];
  const Constraint nc = negate(Constraint{at.i, at.j, at.b});
  std::uint64_t xin, xout;
  {
    Dbm ap = a;                                              // a /\ phi  <=  b
    if (ap.constrain(at.i, at.j, at.b)) {
      classify_atoms(ap, as, &xin, &xout);
      if (!cover_rec(ap, xin, b, bout, as, depth + 1)) return false;
    }
  }
  Dbm an = a;                                                // a /\ ~phi  <=  b /\ ~phi
  if (!an.constrain(nc.i, nc.j, nc.bound)) return true;      // a lies wholly inside phi
  Dbm bn = b;
  if (!bn.constrain(nc.i, nc.j, nc.bound)) return false;     // ...but b does not: no witness
  std::uint64_t nin, nout;
  classify_atoms(an, as, &nin, &nout);
  classify_atoms(bn, as, &xin, &xout);
  return cover_rec(an, nin, bn, xout, as, depth + 1);
}

/// @brief Insert @p s into the passed set, returning the node that now represents it.
///
/// Covering is a_LU subsumption when @ref use_alu is set and plain zone inclusion otherwise,
/// refined in both cases by the constraint map's cell tags when one is installed.
///
/// Three outcomes, in this order: an existing node already covers @p s (return it, nothing is
/// stored); @p s covers an existing node, which is then *enlarged in place*; or the exact
/// convex union of @p s with some stored node is itself a zone, in which case that node absorbs
/// @p s. Only if all three fail is a new node appended.
///
/// @param s  the candidate state; consumed
/// @return the index into @ref node representing @p s
///
/// @post @ref intern_grew is @c true iff a stored zone was rewritten (enlarged or merged). The
///       caller must then invalidate that node's adjacency: it was computed from the smaller
///       zone. ::Otf::expand does this.
/// @post Every stored zone remains atom-homogeneous only in the sense ::Otf::cover_rec needs,
///       i.e. not at all: straddling zones are stored whole and resolved at comparison time.
///
/// @note The two rewriting outcomes are what a purely append-only passed set would forgo. An
///       append-only rule makes every stored node immutable from birth, which is what "start
///       backward work only on vertices known not to change" wants, but it pays for that with
///       an antichain of zones the rewriting version collapses. Rewriting is kept, and the
///       invalidation above is what makes it safe.
/// @brief Hash of a discrete state, computed without materialising its byte key.
static std::uint64_t disc_hash(const LocVec& locs, const IntVec& ints) {
  std::uint64_t h = 1469598103934665603ULL;
  const auto mix = [&h](std::uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  };
  for (int l : locs) mix(std::uint64_t(std::uint32_t(l)));
  mix(0x5bf03635ULL);
  for (long v : ints) mix(std::uint64_t(std::int64_t(v)));
  return h;
}

int Otf::intern(SymState&& s) {
  intern_grew = false;
  std::string bkey = disc_key_bytes(s.locs, s.ints);
  // THE CELL TAG. `tag` is the In-mask and `tag_out` the Out-mask of ::classify_atoms; a zone
  // set in neither straddles that atom, and every obligation a straddle leaves undecidable is
  // deferred to Otf::cover_rec.
  //
  // The obligation the tag enforces is the IMPLICATION `Z |= phi => Z' |= phi`, in that
  // direction only, where Z covers Z'. Dead(q) = Inv(q) /\ AND_e !Esc_e with each Esc_e a
  // conjunction of atoms, so a stuck valuation needs, per outgoing edge e, some atom of Esc_e
  // that its simulator also falsifies. Equivalently `tag(Z) & ~tag(Z') == 0`, which is strictly
  // coarser than same-cell equality: equality needlessly refuses the pair (Z disjoint from phi,
  // Z' contained in phi).
  std::uint64_t tag = 0, tag_out = 0;
  bool tagged = false;
  const std::vector<AtomC>* atoms = nullptr;
  if (amap_) {
    auto it = amap_->diag.find(bkey);
    if (it != amap_->diag.end() && !it->second.empty()) {
      const std::vector<AtomC>& as = it->second;
      tagged = true;
      atoms = &as;
      classify_atoms(s.zone, as, &tag, &tag_out);
    }
  }
  auto& b = bucket[disc_hash(s.locs, s.ints)];
  // Exact discrete equality, checked before any zone test: the bucket is a hash bucket, so it
  // may hold nodes of a different discrete state.
  auto same_disc = [&](int j) {
    const SymState& n = node[j];
    if (n.locs.size() != s.locs.size() || n.ints.size() != s.ints.size()) return false;
    for (std::size_t t = 0; t < s.locs.size(); ++t)
      if (n.locs[t] != s.locs[t]) return false;
    for (std::size_t t = 0; t < s.ints.size(); ++t)
      if (n.ints[t] != s.ints[t]) return false;
    return true;
  };
  if (use_alu) {
    bL.assign(net.nclocks + 1, kAluNegInf);
    bU.assign(net.nclocks + 1, kAluNegInf);
    if (net.lbound_loc.empty()) {
      bL = aL;
      bU = aU;
    } else {
      for (int pr = 0; pr < net.nprocs(); ++pr)
        for (int c = 1; c <= net.nclocks; ++c) {
          bL[c] = std::max(bL[c], net.lbound_loc[pr][s.locs[pr]][c]);
          bU[c] = std::max(bU[c], net.ubound_loc[pr][s.locs[pr]][c]);
        }
      bL[0] = bU[0] = 0;
    }
    // The per-location bounds must carry the non-diagonal atoms' constants too, for the same
    // reason the global arrays do in Otf::setup: raising L or U only makes a_LU finer, and the
    // simulation argument needs the raise to hold along every path.
    //
    // Raising one clock to `max(L, U)` is a_M ON THAT CLOCK: the three clauses of the LU
    // preorder — nu'(x) = nu(x), or L(x) < nu'(x) < nu(x), or U(x) < nu(x) < nu'(x) — collapse
    // to "equal, or both above M(x)", which is symmetric, hence a bisimulation, hence
    // deadlock-preserving (Behrmann–Bouyer–Larsen–Pelánek, STTT 8(3), 2006, Lemma 1 against
    // Lemma 3). Raising EVERY clock is the global a_M fallback; see DESIGN.md for why it is not
    // the default.
    if (amap_)
      for (int c = 1; c <= net.nclocks && c < (int)amap_->bump.size(); ++c)
        if (amap_->bump[c] != kNoBump) {
          bL[c] = std::max(bL[c], amap_->bump[c]);
          bU[c] = std::max(bU[c], amap_->bump[c]);
        }
  }
  const int dim2 = s.zone.dim();
  const bool tagcheck = tagged;
  const bool reccheck = tagcheck;
  Dbm zsc;                                                  // widening scratch, reused
  auto covered = [&](int j) {
    if (reccheck) {                                         // straddling operands: recurse
      return cover_rec(s.zone, tag, zread(j, zsc), ntag_out_[j], *atoms, 0);
    }
    if (tagcheck && (ntag_[j] & ~tag) != 0) return false;   // Z |= phi but Z' does not: refuse
    return use_alu ? alu_le_stored(s.zone, j) : stored_includes(j, s.zone);
  };
  // WHAT THE REWRITING PATHS OWE.
  //
  //   In-place enlargement replaces the stored zone by the newcomer, which under a_LU need not
  //   be a superset of it, so everything already routed to j must still be covered. That is
  //   transitivity of the covering relation, and the relation is transitive (<=_LU is a preorder
  //   and `nu' |= phi => nu |= phi` composes), so the ordinary three-valued test in the opposite
  //   direction is exactly and only what is owed.
  //
  //   The convex-union merge owes nothing at all: try_union succeeds only when the union really
  //   is a single zone, so the stored zone afterwards is a set superset of both operands.
  //   Growing the right-hand side of an exists-witness cannot invalidate a witness, and the
  //   newcomer is covered by the result via the identity witness.
  //
  // Both paths must REFRESH the stored masks, since the stored zone changed.
  auto covers = [&](int j) {
    if (reccheck) {
      return cover_rec(zread(j, zsc), ntag_[j], s.zone, tag_out, *atoms, 0);
    }
    if (tagcheck && ntag_[j] != tag) return false;          // rewriting: equality, see above
    return use_alu ? alu_le_new(j, s.zone) : includes_stored(s.zone, j);
  };
  for (int j : b)
    if (same_disc(j) && covered(j)) return j;
  for (std::size_t k = 0; k < b.size(); ++k)
      if (same_disc(b[k]) && covers(b[k])) {
        zstore(b[k], Dbm(s.zone));
        if (reccheck) { ntag_[b[k]] = tag; ntag_out_[b[k]] = tag_out; }
        intern_grew = true;
        return b[k];
      }
  // EXACT CONVEX-UNION MERGE. Dbm::try_union succeeds only when the union of the two zones
  // really is a single zone, so the stored zone afterwards denotes exactly the set the two it
  // replaced denoted: no point added, none removed. That exactness is what makes the merge
  // usable here at all — unlike a convex hull it is not an over-approximation, so it cannot
  // invent a certificate in the under-approximating backward lane.
  //
  // It earns its keep on antichains: grc4 stores ~14.5 zones per discrete key without it,
  // against 1.00–1.44 on every other family, which is the shape of an antichain of zones whose
  // pairwise unions happen to be convex.
  {
    for (int j : b) {
      if (!same_disc(j)) continue;
      Dbm t = zread(j, zsc);              // copy: try_union rewrites its left operand
      if (t.dim() != dim2) continue;
      if (t.try_union(s.zone)) {          // union is exactly convex: absorb, no approximation
        if (reccheck) classify_atoms(t, *atoms, &ntag_[j], &ntag_out_[j]);
        zstore(j, std::move(t));
        intern_grew = true;
        return j;
      }
    }
  }
  const int id = static_cast<int>(node.size());
  Dbm znew = std::move(s.zone);           // taken out first: zstore decides where it lands
  s.zone = Dbm();
  node.push_back(std::move(s));
  if (ntag_.size() <= static_cast<std::size_t>(id)) ntag_.resize(id + 1, 0);
  if (ntag_out_.size() <= static_cast<std::size_t>(id)) ntag_out_.resize(id + 1, 0);
  ntag_[id] = tag;
  ntag_out_[id] = tag_out;
  zstore(id, std::move(znew));
  b.push_back(id);
  ++outer_nodes;
  grow();
  return id;
}

/// @brief Emit the pieces of @p s that the passed set should intern.
///
/// @param s        the generated state; consumed
/// @param[out] out cleared, then filled with the pieces
///
/// @post The pieces partition @p s exactly, so post, pre and ::Otf::stop_set all distribute
///       over them and the greatest fixpoint is unchanged.
///
/// @note The engine splits **at comparison**, not at creation: this emits @p s whole and
///       ::Otf::cover_rec resolves each straddled atom only for the pair of zones actually
///       being compared. Splitting at creation was measured and is worse — 1179 stored nodes on
///       grc4 against 921 — because it cuts zones along atoms that no comparison ever
///       interrogates. See @c DESIGN.md, "Measured and rejected".
/// @note A piece need not be delay-closed, and ::Otf::stop_set stays correct anyway:
///       ptctlmc::detail::state_has_deadlock rebuilds `Up(Z) /\ Inv(q)` before testing, and
///       ::Otf::gfp_on intersects the result back with the node's own zone, so
///       `(Up(Z) /\ Inv) \ live` cut down to `Z` is exactly `Z \ live`.
void Otf::split_by_atoms(SymState&& s, std::vector<SymState>& out) {
  out.clear();
  const std::vector<AtomC>* at = nullptr;
  if (amap_) {
    auto it = amap_->diag.find(disc_key_bytes(s.locs, s.ints));
    if (it != amap_->diag.end() && !it->second.empty()) at = &it->second;
  }
  out.push_back(std::move(s));
  if (!at) return;
  // Split at comparison: the zone is emitted whole and every deferred obligation is discharged
  // in Otf::cover_rec, so both exits from here produce the same single-element `out`.
  //
  // The lookup above therefore has no effect on the result. It costs a disc_key_bytes string and
  // a hash per generated successor whenever a constraint map is installed, on the hottest
  // forward path, and it is the split-at-creation implementation's only surviving trace.
}

/// @brief Generate and intern the successors of node @p u.
///
/// @param u  node index; must be in range
///
/// @post `expanded[u]` is set unless the expansion re-entered @p u through a rewrite, or the
///       search was cancelled before any work was done.
/// @post For every successor @c v, `adj[u]` gains an edge to @c v.
///
/// @warning If ::Otf::intern rewrote a stored zone, that node's adjacency was computed from the
///          smaller zone and is now stale. It is cleared here and `expanded` is reset so the
///          driver re-expands it.
/// @note This is the single choke point through which every forward step passes, so it is where
///       cancellation is polled.
/// @brief Add @p t to the tag pool, reusing an equal tag if one is already there.
///
/// @param t  the tag produced for one transition
/// @return the index into ::Otf::tags_ that ::OtfEdge::tag must carry
///
/// @note Tags are immutable once pooled, and ::gfp_pre only ever reads them, so sharing one
///       between edges is safe. Deduplication is by VALUE, not by identity: two transitions
///       that fire the same participants with the same abstain constraints invert identically.
int Otf::intern_tag(const GfpTag& t) {
  const std::uint64_t h = gfp_tag_hash(t);
  auto& slots = tagidx_[h];
  for (int i : slots)
    if (gfp_tag_eq(tags_[i], t)) return i;
  const int id = static_cast<int>(tags_.size());
  tags_.push_back(t);
  slots.push_back(id);
  return id;
}

void Otf::expand(int u) {
  grow();
  if (cancelled()) return;   // choke point: every forward step goes through here
  if (expanded[u]) return;
  expanded[u] = 1;
  ++expand_calls;
  ++outer_expands;
  std::vector<SymState> succ;
  std::vector<GfpTag> tags;
  std::vector<SymState> pieces;
  SymState usc;
  gfp_succ_tagged(net, sread(u, usc), L, U, qmask, emode, succ, tags,
                  avoid_forward_ ? avoid_ : nullptr);
  for (std::size_t k = 0; k < succ.size(); ++k) {
    pieces.clear();
    if (amap_) {
      split_by_atoms(std::move(succ[k]), pieces);
    } else {
      pieces.push_back(std::move(succ[k]));
    }
    // One tag serves every piece of this successor: the pieces partition the successor zone,
    // and the transition that produced them is the same one.
    const int tid = intern_tag(tags[k]);
    for (std::size_t pi = 0; pi < pieces.size(); ++pi) {
    const int v = intern(std::move(pieces[pi]));
    grow();
    adj[u].push_back(OtfEdge{v, tid});
    // A rewrite invalidates the rewritten node's adjacency; see the @warning above.
    if (intern_grew) {
      if (v != u) {
        expanded[v] = 0;
        adj[v].clear();
      } else {
        expanded[u] = 0;
      }
    }
    }
  }
}

/// @brief Can time grow without bound from @p s, with no transition ever taken?
///
/// @param s  a symbolic state
/// @return @c true iff every clock is unbounded in the delay closure of @p s's zone under the
///         location invariants — i.e. the whole zone admits an infinite time-divergent run.
///
/// @note A committed or urgent location admits no positive delay at all, so it can never
///       diverge. Delaying unconditionally would report divergence there and fabricate a
///       maximal-run witness out of nothing.
bool Otf::time_diverges(const SymState& s) const {
  if (net.nclocks == 0) return false;
  { bool fc, fu; frozen_flags(net, s.locs, s.ints, &fc, &fu); if (fc || fu) return false; }
  std::vector<Constraint> inv;
  all_invariants(net, s.locs, inv);
  // With a forbidden region the delay is routed, and divergence must be witnessed by a piece of
  // the routed delay: a trajectory that would run into the region stops there rather than
  // diverging, and one that already lies past it may still run away.
  if (avoid_ && avoid_->applies(s.locs, s.ints)) {
    Federation parts;
    avoid_->delay(s.zone, s.locs, s.ints, parts);
    for (Dbm& p : parts) {
      if (!apply_all(p, inv) || p.empty()) continue;
      bool unbounded = true;
      for (int i = 1; i <= net.nclocks && unbounded; ++i)
        if (p.get(i, 0) != kInf) unbounded = false;
      if (unbounded) return true;
    }
    return false;
  }
  Dbm z = s.zone;
  z.delay();
  apply_all(z, inv);
  if (z.empty()) return false;
  for (int i = 1; i <= net.nclocks; ++i)
    if (z.get(i, 0) != kInf) return false;
  return true;
}

/// @brief The end-of-run term of the fixpoint operator: the valuations of node @p u from which
///        the run may legitimately stop.
///
/// @param u  node index; @p u must be expanded, or an empty adjacency is read as "stuck" when it
///          only means "not yet explored"
/// @return the federation of valuations of `node[u].zone` that either enable no transition ever
///         or admit a time-divergent run
///
/// @pre Stored zones are exact (::Otf::setup's invariant). A stop valuation occurs *positively*
///      in the fixpoint operator, so a spurious one fabricates a counterexample rather than
///      losing one.
/// @note Two of the three cases are whole-zone and exact as such: an empty adjacency means every
///       valuation is stuck, and ::time_diverges is a property of the delay-closed zone as a
///       whole. Only the third — partial deadlock — needs the per-valuation subtraction, and
///       ptctlmc::detail::state_has_deadlock supplies it, rebuilding the delay closure first so
///       that the active-clock reduction cannot leave the stored zone unclosed.
/// @brief Can time run away for ever here in the UNRESTRICTED system?
///
/// @param s the state; only its location vector and integers are read
/// @return @c true iff the original invariant leaves every clock unbounded after a delay, so a
///         valuation with no enabled transition continues by delaying rather than stopping
///
/// @note Only meaningful when a `psi`-restriction is in force. It is what separates a genuine
///       TIMELOCK -- a state that truly cannot be extended, and therefore ends a maximal run --
///       from a state that merely cannot be extended *inside psi*.
bool Otf::orig_time_open(const SymState& s) const {
  if (net.nclocks == 0) return false;
  bool c, u;
  frozen_flags(*dead_net, s.locs, s.ints, &c, &u);
  if (c || u) return false;                       // frozen: no positive delay at all
  Dbm z = Dbm::universe(net.nclocks);
  z.delay();
  std::vector<Constraint> inv;
  all_invariants(*dead_net, s.locs, inv);
  if (!apply_all(z, inv)) return false;
  for (int i = 1; i <= net.nclocks; ++i)
    if (z.get(i, 0) != kInf) return false;        // some clock is capped: time cannot run away
  return true;
}

Federation Otf::stop_set(int u) {
  Federation f;
  SymState usc;
  const SymState& su = sread(u, usc);
  // A psi-restriction changes what "maximal" means, so the two cases are kept apart rather than
  // merged. Unrestricted -- every `p --> q` run -- keeps the original reading exactly: psi is
  // discrete there, a delay can never leave it, so a node with no discrete successor always ends
  // a maximal run.
  // A psi-restriction may arrive as a restricted network (psi's clock part convex, folded into
  // the invariants) or as a forbidden region (non-convex, routed around). Either one changes what
  // maximality means, so both take the second branch.
  if (dead_net == &net && !avoid_) {
    if (adj[u].empty() || time_diverges(su)) { f.push_back(su.zone); return f; }
    Federation d0;
    if (state_has_deadlock(net, su, &d0)) f.swap(d0);
    return f;
  }
  // Restricted: `adj[u].empty()` no longer implies maximality. The successors may all have been
  // dropped for leaving psi, and a run that can only continue by leaving psi is not maximal, it
  // is simply not a witness. Two things genuinely end a maximal psi-run:
  //
  //   (i)  time diverges INSIDE psi -- judged on the restricted invariant, which is `net`;
  //   (ii) the state is timelocked in the ORIGINAL system -- no transition is ever enabled AND
  //        the original invariant caps the delay, so there is no divergent continuation either.
  //
  // Dropping (ii)'s second half is what made `E[] x <= 3` on af-sat.xml report satisfied: the
  // sink location is deadlocked, but the original bounds no clock there, so its maximal run is
  // the divergent one -- which carries x past 3 and out of psi.
  if (time_diverges(su)) { f.push_back(su.zone); return f; }
  if (orig_time_open(su)) return f;               // a deadlock here continues by delaying
  Federation d;
  // Deadlock is judged in `dead_net`, which is the UNRESTRICTED network when `E[] psi` supplied
  // one. Maximality is a property of the original system: a run that can only continue by
  // LEAVING psi is not maximal, it is simply not a witness. Judging it against the psi-restricted
  // network instead reads "psi stops holding" as "time stops" and reports the resulting timelock
  // as a maximal run -- `E[] P0.idle && x0 <= 40` on fischer4 becomes satisfied, where the true
  // answer is that every continuation past x0 = 40 leaves psi.
  //
  // Divergence is the other way round and stays on `net`: the run must diverge INSIDE psi, so the
  // restricted invariant is exactly the right one to consult.
  if (state_has_deadlock(*dead_net, su, &d)) f.swap(d);
  return f;
}

/// @brief Solve `W = W /\ (NoSucc \/ Pre(W))` by descending iteration confined to @p S.
///
/// @param S      the nodes to iterate over; every other node is pinned to the empty federation
/// @param[out] W indexed by node; sized to `node.size()`, empty outside @p S
///
/// @pre Every element of @p S is an expanded `!q` node (see ::Otf::stop_set).
/// @post On normal return @p W is the exact greatest fixpoint of the operator *restricted to*
///       @p S, hence a post-fixpoint of the operator on the finished graph, hence a sound
///       under-approximation of the true value.
/// @post If @ref capped or @ref cancelled fired, the iteration did **not** converge and @p W is
///       an arbitrary intermediate iterate. It proves nothing in either direction.
///
/// @warning @ref capped is sticky. A capped pass must be reported as UNUSABLE, never laundered
///          into a verdict; see ::Otf::certificate_in.
void Otf::gfp_on(const std::vector<int>& S, std::vector<Federation>& W) {
  ++passes;
  const std::size_t N = node.size();
  std::vector<char> inS(N, 0);
  for (int v : S) inS[v] = 1;
  W.assign(N, Federation{});
  std::vector<Federation> nosucc(N);
  // THE SEED IS NOT MATERIALISED. Every node of S starts at its own zone, and writing all |S|
  // of those copies down before the sweep begins is the peak of the entire run -- one full
  // dense DBM per node, held simultaneously, on top of the copy already in `node`. `seeded[v]`
  // stands in for that copy: it means "W[v] is logically { node[v].zone }". The sweep
  // materialises a seed only for the node it is processing, and replaces it with `W[v] /\ cand`
  // in the same step, so at most one seed is ever live. On a satisfied query that refinement is
  // already draining towards empty, which is why the peak never re-forms.
  //
  // A seed is never empty: ::Otf::intern stores no empty zone, so `seeded[v]` implies non-empty
  // and every emptiness test below can short-circuit on it.
  std::vector<char> seeded(N, 0);
  for (int v : S) {
    seeded[v] = 1;
    nosucc[v] = stop_set(v);
  }
  // When the forward graph was NOT restricted, the stored zones still straddle `psi` and the
  // fixpoint has to do the confining itself: `W[v]` starts at `Z_v /\ psi(v)`, not at `Z_v`. The
  // seed is otherwise left unmaterialised (see `seeded`), so only the nodes that actually need
  // cutting are written down here.
  std::vector<Federation> psi_seed(N);
  const bool confine = avoid_ && !avoid_forward_;
  if (confine)
    for (int v : S) {
      SymState cst;
      const SymState& sv = sread(v, cst);
      if (!avoid_->applies(sv.locs, sv.ints)) continue;
      Federation cut;
      avoid_->restrict(sv.zone, sv.locs, sv.ints, cut);
      psi_seed[v] = std::move(cut);
      seeded[v] = 0;                       // no longer "logically its own zone"
      W[v] = psi_seed[v];
      if (W[v].empty()) { nosucc[v].clear(); continue; }
      // The stop set was computed against the whole zone; cut it to psi as well, or a valuation
      // outside psi could stop the run and count as a witness.
      if (!nosucc[v].empty()) nosucc[v] = fed_intersect(nosucc[v], psi_seed[v]);
    }
  // Widening scratch, reused across the whole sweep so a narrow-stored zone costs a copy of
  // its entries and never an allocation.
  Dbm vsc, jsc;
  SymState vst, jst;
  // ROUND-ROBIN LOCKSTEP SWEEP, deliberately, not for want of a worklist. Replacing it with a
  // chaotic worklist keyed on the reverse edges saves 0.1% of Pre calls on fischer7 and turns grc4
  // `T0.in --> T0.far` from SATISFIED into UNUSABLE: reordering cannot change the limit of a
  // monotone descending iteration, but it does change the intermediate federation SHAPES, and
  // `fedcap` is a resource bound on those. The lockstep sweep acts as a de-facto normaliser.
  // See DESIGN.md, "Measured and rejected", for the numbers -- and KNOWN_LIMITATION.md for why
  // the same sensitivity is what blocks a parallel schedule here.
  // SCC-DECOMPOSED LOCKSTEP. The sweep above is kept verbatim; what changes is the SET it is
  // swept over.
  //
  // A node's `cand` reads only its SUCCESSORS' W, so the descent propagates emptiness one edge
  // per global round and the round count tracks the graph's DIAMETER, not anything about the
  // property. grc5 `E[] x0 <= 15 && !T0.in` needed 47 rounds over 14610 nodes -- 1.59M Pre calls
  // -- where its satisfied sibling needed 6 on the identical graph.
  //
  // Iterating strongly connected components in reverse topological order removes exactly that.
  // When a component is reached its successors outside it are already final, so it converges
  // against fixed outside values and is never revisited; the result is the same greatest fixpoint
  // because the operator is local and Tarjan emits components successors-first.
  //
  // This is NOT the chaotic worklist recorded as measured and rejected. That one reordered the
  // visits and changed the intermediate federation SHAPES, which `fedcap` bounds -- it turned
  // grc4 `T0.in --> T0.far` UNUSABLE. Here the lockstep round-robin survives INSIDE each
  // component, so the de-facto normalisation it provides is kept where the iteration actually
  // happens; only the outer partition into components is new.
  // Enabled for `E[] psi` and not for `p --> q`, which is a measured split rather than a
  // preference. `E[] psi` reads ONE node, so emptiness has to propagate the full diameter back to
  // it and the round count is dominated by that; `p --> q` reads every p-node and settles long
  // before diameter matters. Best of two runs, decomposed / undecomposed:
  //
  //     E[]        grc5 x0<=15 && !T0.in  0.69     fischer7 x0<=40   0.75
  //                fischer6 x0<=40        0.76     grc5 x0<=15       0.98
  //     leads-to   fischer6 P0.req        0.77     csma6 Bus         0.91
  //                arbiter8 C0.cs         1.00     grc4 T0.in        1.02
  //                phil6 P0.Eat           1.06     grc5 T0.in        1.26  <-- the one loss
  //
  // grc5 leads-to is the biggest leads-to case in the suite and it loses a quarter, so the
  // decomposition does not go there. PTCTLMC_GFP_SCC=0 or =1 overrides either way, for measuring.
  const bool scc_on = [&] {
    const char* e = std::getenv("PTCTLMC_GFP_SCC");
    if (e) return *e == '1';
    return scc_iter_;
  }();
  std::vector<std::vector<int>> sccs;
  if (!scc_on) {
    sccs.push_back(S);                     // one component: exactly the undecomposed sweep
  } else {
    std::vector<int> idx(N, -1), low(N, 0), stk;
    std::vector<char> on(N, 0);
    int next = 0;
    // Iterative Tarjan: the graphs here reach 10^5 nodes and the recursive form would not survive
    // the stack.
    std::vector<std::pair<int, std::size_t>> call;
    for (int root : S) {
      if (idx[root] != -1) continue;
      call.push_back({root, 0});
      idx[root] = low[root] = next++;
      stk.push_back(root);
      on[root] = 1;
      while (!call.empty()) {
        auto& [v, ei] = call.back();
        if (ei < adj[v].size()) {
          const int w = adj[v][ei++].tgt;
          if (!inS[w]) continue;
          if (idx[w] == -1) {
            idx[w] = low[w] = next++;
            stk.push_back(w);
            on[w] = 1;
            call.push_back({w, 0});
          } else if (on[w]) {
            low[v] = std::min(low[v], idx[w]);
          }
        } else {
          if (low[v] == idx[v]) {
            std::vector<int> comp;
            for (;;) {
              const int w = stk.back();
              stk.pop_back();
              on[w] = 0;
              comp.push_back(w);
              if (w == v) break;
            }
            // Restore S's own order inside the component, so the lockstep sweep walks it exactly
            // as the undecomposed one did.
            std::sort(comp.begin(), comp.end());
            sccs.push_back(std::move(comp));
          }
          const int child = v;
          call.pop_back();
          if (!call.empty()) low[call.back().first] = std::min(low[call.back().first], low[child]);
        }
      }
    }
  }

  std::size_t it = 0;
  for (const auto& comp : sccs) {
   if (capped || cancelled()) break;
   bool changed = true;
   while (changed && !capped) {
    if (cancelled()) break;
    changed = false;
    ++it;
    for (int v : comp) {
      if (!seeded[v] && gfp_fed_empty(W[v])) continue;
      Federation cand = nosucc[v];
      for (const auto& e : adj[v]) {
        const int j = e.tgt;
        if (!inS[j]) continue;
        if (!seeded[j] && gfp_fed_empty(W[j])) continue;
        // A seeded successor contributes exactly one zone, its own, without being written down.
        const Dbm* one = seeded[j] ? &zread(j, jsc) : nullptr;
        const std::size_t ny = one ? 1u : W[j].size();
        for (std::size_t yi = 0; yi < ny; ++yi) {
          const Dbm& Y = one ? *one : W[j][yi];
          ++pre_calls;
          if (avoid_) {
            gfp_pre_avoid(net, tags_[e.tag], sread(v, vst), sread(j, jst), Y, *avoid_, cand);
          } else {
            Dbm pz = gfp_pre(net, tags_[e.tag], sread(v, vst), sread(j, jst), Y);
            if (!pz.empty()) cand.push_back(std::move(pz));
          }
        }
      }
      // Materialise v's own seed now, for this one node only: the swap below overwrites it with
      // the refinement, so the copy does not outlive the iteration.
      if (seeded[v]) { W[v].clear(); W[v].push_back(zread(v, vsc)); seeded[v] = 0; }
      gfp_fed_compact(cand, fedcap, &capped);
      // `W[v] /\ cand`, computed directly rather than as the double complement
      // `subtract(z, subtract(z, cand))` the textbook formulation uses. Both denote the same
      // set, but the inner subtraction peels z by every constraint of every member of cand, so
      // its output is an ARRANGEMENT whose size is not bounded by |cand| at all, and the outer
      // subtraction then peels z by all of those. The intersection of two zones is a single
      // zone, so the direct form needs exactly |cand| pieces.
      Federation nw;
      for (const auto& z : W[v]) {
        for (const auto& c : cand) {
          Dbm t = z;
          t.intersect(c);
          if (!t.empty()) nw.push_back(std::move(t));
        }
      }
      gfp_fed_compact(nw, fedcap, &capped);
      if (!gfp_fed_subset(W[v], nw)) changed = true;
      W[v].swap(nw);
      if (capped) break;
    }
    if (it > 100000) break;
   }
  }
  // Restore the documented postcondition on every exit path. A converged sweep materialised
  // every seed as it went, so this loop does nothing; an aborted one (capped or cancelled) can
  // still hold seeds, and W must denote the iterate the sweep actually reached rather than
  // reading as empty at those nodes.
  for (int v : S)
    if (seeded[v]) { W[v].clear(); W[v].push_back(zread(v, vsc)); seeded[v] = 0; }
  gfp_iters += it;
}

/// @brief Is there a counterexample certificate within @p S?
///
/// @param S  a set of expanded `!q` nodes
/// @return @c true iff the converged confined fixpoint leaves a non-empty value at some node of
///         @p S satisfying `p`
///
/// @post @ref fed_max, @ref fed_avg and @ref nonempty describe the converged federations.
///
/// @warning A @c false return is meaningful only when @ref capped is clear. A capped pass did
///          not converge and proves nothing in either direction; returning @c false for it
///          would launder "out of federation budget" into a silent SATISFIED. Callers must
///          check @ref capped and report UNUSABLE.
bool Otf::certificate_in(const std::vector<int>& S) {
  if (S.empty()) return false;
  std::vector<Federation> W;
  gfp_on(S, W);
  std::size_t tot = 0, ne = 0;
  for (int v : S) {
    if (gfp_fed_empty(W[v])) continue;
    ++ne;
    tot += W[v].size();
    if (W[v].size() > fed_max) fed_max = W[v].size();
  }
  nonempty = ne;
  fed_avg = ne ? double(tot) / double(ne) : 0.0;
  if (capped) return false;                      // caller checks `capped` before believing this
  for (int v : S) {
    if (!is_p(v) || gfp_fed_empty(W[v])) continue;
    // The exact test: a violation needs a valuation that satisfies `p` AND survives the fixpoint.
    // With a purely discrete `p` the region is the whole zone and this reduces to the old check.
    if (p.forbid.empty()) return true;
    if (!gfp_fed_empty(fed_intersect(W[v], p_region(v)))) return true;
  }
  return false;
}

/// @brief Construct and intern the initial symbolic state.
///
/// @return @c false iff the initial zone is empty under the initial locations' invariants, in
///         which case no node is stored at all and the caller sees an empty graph
///
/// @post On success node 0 exists and is the (only) root of the search.
/// @note The construction order — invariants, then delay unless the initial location is frozen,
///       then invariants again, then the active-clock reduction, then extrapolation — must match
///       ptctlmc::detail::close_succ_zone, or the root would sit in a different abstraction from
///       every state reached from it.
bool Otf::build_all() {
  SymState s0;
  s0.locs.resize(net.nprocs());
  for (int pr = 0; pr < net.nprocs(); ++pr) s0.locs[pr] = net.procs[pr].initial;
  s0.ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0.ints[v] = net.intvars[v].init;
  s0.zone = Dbm(net.nclocks);
  {
    std::vector<Constraint> inv;
    all_invariants(net, s0.locs, inv);
    if (!apply_all(s0.zone, inv)) return false;     // unreachable init: caller sees no nodes
    bool c, u;
    frozen_flags(net, s0.locs, s0.ints, &c, &u);
    // The forbidden region applies to the ROOT too, and it must REPLACE the delay rather than
    // follow it. This construction does not go through close_succ_zone, and a plain delay
    // followed by a restriction is the very error the psi-restriction exists to avoid: the zone
    // travels through the region first and is only cut afterwards, so the root ends up holding
    // valuations no run can reach without leaving psi. Every verdict is read at the initial
    // vertex, so that alone decides queries wrongly.
    if (avoid_ && avoid_forward_ && avoid_->applies(s0.locs, s0.ints)) {
      Federation parts;
      if (!c && !u) avoid_->delay(s0.zone, s0.locs, s0.ints, parts);
      else avoid_->restrict(s0.zone, s0.locs, s0.ints, parts);
      // The root must be a single zone. Merge only where the union is exactly convex; otherwise
      // keep the largest piece, which UNDER-approximates the root and so can lose a witness but
      // never invent one.
      Dbm merged;
      bool have = false;
      for (Dbm& pz : parts) {
        if (pz.empty()) continue;
        if (!apply_all(pz, inv) || pz.empty()) continue;
        if (!have) { merged = pz; have = true; continue; }
        if (!merged.try_union(pz) && pz.includes(merged)) merged = pz;
      }
      if (!have) return false;
      s0.zone = std::move(merged);
    } else if (!c && !u) {
      s0.zone.delay();
      apply_all(s0.zone, inv);
    }
    reset_inactive(net, s0.locs, s0.zone, qmask);
    if (emode == ExLuPlus) s0.zone.extrapolate_lu_plus(L, U);
    else if (emode == ExNoDiag) s0.zone.extrapolate_lu_plus_nodiag(L, U);
  }
  if (amap_) {
    std::vector<SymState> pieces;
    split_by_atoms(std::move(s0), pieces);
    for (auto& pc2 : pieces) intern(std::move(pc2));
  } else {
    intern(std::move(s0));
  }
  return true;
}


/// @brief The `!q` nodes that can influence the verdict.
///
/// @return every expanded node reachable through `!q` edges only from an expanded `p /\ !q`
///         node, the accepting nodes themselves included
///
/// @note Restricting the final pass to this set is EXACT, not an approximation. The operator is
///       defined per node from that node's own successors, so an accepting node's value depends
///       on exactly this closure; and `p --> q` is decided by the accepting nodes alone, so
///       nothing outside it is ever read.
std::vector<int> Otf::accepting_closure() {
  grow();
  const std::size_t N = node.size();
  std::vector<char> in(N, 0);
  std::vector<int> stk, out;
  for (std::size_t v = 0; v < N; ++v)
    if (expanded[v] && is_nq((int)v) && is_p((int)v)) { in[v] = 1; stk.push_back((int)v); }
  while (!stk.empty()) {
    const int v = stk.back();
    stk.pop_back();
    out.push_back(v);
    for (const auto& e : adj[v])
      if (!in[e.tgt] && expanded[e.tgt] && is_nq(e.tgt)) { in[e.tgt] = 1; stk.push_back(e.tgt); }
  }
  return out;
}

/// @brief Decide the query by one exact backward pass over the finished graph.
///
/// @return @c true iff `p --> q` is violated
///
/// @pre The forward construction has terminated, so the accepting closure is complete and the
///      confined operator coincides with the full one at every accepting node.
/// @warning Check @ref capped before believing a @c false return; see ::Otf::certificate_in.
bool Otf::final_verdict_violated() {
  return certificate_in(accepting_closure());
}

/// @brief Decide whether the deadlock completion's atom seed is empty at every reachable key.
///
/// @return @c true iff no reachable discrete key contributes an atom, in which case plain a_LU
///         covering already preserves deadlock and the search that produced this graph was
///         sound as it stands
///
/// @pre The forward graph has been built, so @ref node enumerates every reachable discrete key.
///
/// @note A key contributes nothing in exactly three cases, each checked below: `Dead(q)` is
///       empty (there is no sink edge at all); `Dead(q)` is all of `Inv(q)` (the sink edge is
///       always enabled, and every stored zone satisfies `Inv(q)`); or `Inv(q)` bounds no clock
///       at a non-frozen key (time is open, so ::Otf::stop_set's divergence disjunct already
///       admits the whole zone). See @c DESIGN.md for why each drop is safe.
/// @note The sweep is STRUCTURAL — it asks about `Dead(q)` over all of `Inv(q)`, not over any
///       reachable zone — so it costs one federation computation per discrete key and no
///       search: 256 keys on arbiter6, against the 28591-state reachability traversal the
///       deadlock-freedom lane of ::leads_to_otf's race pays for the same conclusion.
/// @note Enumerating keys off the a_LU graph is sound because a simulation never loses a
///       reachable discrete key: the covering zone does at least what the covered one does.
bool Otf::dtheta_empty() {
  std::unordered_set<std::string> seen;
  for (std::size_t i = 0; i < node.size(); ++i) {
    const std::string k = disc_key_bytes(node[i].locs, node[i].ints);
    if (!seen.insert(k).second) continue;
    SymState u; u.locs = node[i].locs; u.ints = node[i].ints;
    u.zone = Dbm::universe(net.nclocks);
    Dbm inv = Dbm::universe(net.nclocks);
    bool c2, u2; frozen_flags(net, u.locs, u.ints, &c2, &u2);
    std::vector<Constraint> ci; all_invariants(net, u.locs, ci);
    if (!c2 && !u2) inv.delay();
    if (!apply_all(inv, ci)) continue;                        // vacuous key
    Federation stuck, esc;
    if (!state_has_deadlock(net, u, &stuck, &esc)) continue;   // Dead(q) = empty
    bool open = true;
    for (int c = 1; c <= net.nclocks; ++c)
      if (inv.get(c, 0) != kInf) { open = false; break; }
    // The time-open drop is a claim about DIVERGENCE, so it is available only where time can
    // pass. At a committed/urgent key no positive delay is legal, so Inv(q) is the undelayed
    // universe and trivially bounds no clock -- exactly when this test would fire and exactly
    // when it would be wrong. Frozen keys fall through to the Dead=Inv test instead.
    if (open && !c2 && !u2) continue;                          // divergence already covers stop
    // Inv \ Dead = Inv ^ union Esc, so "Dead(q) = Inv(q)" is "nothing in Inv(q) escapes": one
    // DBM intersection per outgoing transition, instead of peeling Inv(q) by the residue
    // ARRANGEMENT. Same predicate; see the twin test in Otf::build_atom_map for the cost.
    bool some_escapes = false;
    for (const Dbm& W : esc) {
      Dbm t = W;
      t.intersect(inv);
      if (!t.empty()) { some_escapes = true; break; }
    }
    if (!some_escapes) continue;                               // Dead(q) = Inv(q): atoms implied
    return false;                                              // PARTIAL: real atoms are needed
  }
  return true;
}


/// @brief Derive the deadlock-completion constraint map from this run's finished graph.
///
/// @param[out] am  the map; @ref AtomMap::usable reports whether it may be used
/// @return @c false iff the construction declined — more than 64 atoms at one key, no clocks, or
///         63 clocks or more — in which case @p am must not be consulted and the caller falls
///         back to ::Otf::dtheta_empty and the race
///
/// @pre The forward graph has been built, so @ref node enumerates every reachable discrete key.
/// @post `am.diag` maps each contributing key to its diagonal atoms, `am.bump[c]` is the
///       constant `L(c)` and `U(c)` must reach for every non-diagonal atom on clock `c`, and
///       `am.live[c]` is set for every clock either mentions.
/// @post `am.diag` is empty whenever every clock was raised all the way to `M`: the collapsed
///       preorder is then already a bisimulation, so cell tags on top of it would only split
///       zones a_M keeps apart anyway, and the construction must never end up *worse* than the
///       a_M fallback it exists to beat.
///
/// @invariant At most 64 atoms per key, because the per-node classification is a @c uint64_t
///            bitmask. An atom past bit 63 would be silently dropped from the covering test
///            while still splitting zones — a lost obligation, i.e. a wrong verdict, not merely
///            a coarseness loss.
///
/// @note **Seeding.** Only a key with a genuinely partial `Dead(q)` contributes; the three
///       degenerate cases ::Otf::dtheta_empty decides carry no information and are skipped here
///       for the same reasons.
/// @note **Where the atoms come from.** They are read off the canonical DBMs of the `Dead(q)`
///       federation rather than derived from the guards. `Dead` is a disjunction of conjunctions
///       of DBM entries, so those entries decide membership exactly and every atom family comes
///       for free — including the lower bound that survives `Down()`, which the DBM re-derives
///       when it re-closes after zeroing row 0 and which a hand-written formula for `Down(K)`
///       gets wrong in half the cases.
/// @note **Pre-closure.** The map must satisfy `[R]^-1(G(q')) ⊆ G(q)` for every edge
///       `(q, g, R, q')`, or two zones sharing a cell at `q` could step into different cells at
///       `q'`. The pullback of `x_i - x_j <| b` through a reset set `R` is the index
///       substitution `i, j -> 0` for reset clocks — the same rule
///       ptctlmc::detail::state_has_deadlock uses for target invariants. A pullback landing on
///       clock 0 is non-diagonal and is discharged by the bound raise instead. Dropping an atom
///       at the target only ever *relaxes* the clause at the source, so the seed drops stay safe
///       under closure. Guard atoms need no seeding: a_LU is already a simulation for the
///       original automaton, and the only edge this construction adds is the sink's.
/// @brief Rebuild a discrete key's invariant zone `Inv(q)` from any node carrying that key.
///
/// Must reproduce, step for step, what ::Otf::build_atom_map's seed loop computes: universe,
/// delay unless the key is frozen, then the invariants. A key whose invariant is unsatisfiable
/// is "vacuous" and the seed loop skips it, which is what a @c false return reports.
static bool key_invariant(const Network& net, const SymState& rep, int nc, Dbm& out) {
  out = Dbm::universe(nc);
  bool fc, fu;
  frozen_flags(net, rep.locs, rep.ints, &fc, &fu);
  std::vector<Constraint> ci;
  all_invariants(net, rep.locs, ci);
  if (!fc && !fu) out.delay();
  return apply_all(out, ci);
}

bool Otf::build_atom_map(AtomMap& am) {
  const int nc = net.nclocks;
  am.diag.clear();
  am.bump.assign(nc + 1, kNoBump);
  am.live.assign(nc + 1, 0);
  am.usable = false;
  if (nc <= 0 || nc >= 63) return false;

  // The hard representation limit of the uint64 cell tag, not a tuning knob. Declining below it
  // declines into the race, which is far slower: grc5 needs 36 seed atoms at one key, so a cap
  // of 32 sent it to the race (20.9s) or to a_M (18.5s) where the construction settles it in
  // 3.4s.
  constexpr std::size_t kMaxAtoms = 64;

  // A REPRESENTATIVE, NOT THE ZONE. This map used to hold `Inv(q)` as a dense Dbm per discrete
  // key, and it was the largest single allocation the engine made -- 130 MB on arbiter14 and
  // 733 MB on arbiter16, against 107 MB and 542 MB for the entire stored graph. It is read in
  // exactly one place, the pre-closure below, and the pre-closure does not run at all unless
  // some key seeded an atom: on every arbiter size the seed is empty, the function returns at
  // "NOTHING TO CLOSE", and every one of those zones was built and thrown away unread.
  //
  // The key's invariant is a function of its location vector, so a node index is a complete
  // stand-in for it at four bytes. ::key_invariant rebuilds the zone from that node, and the
  // pre-closure memoises the handful of keys it actually queries.
  std::unordered_map<std::string, int> invrep;
  std::unordered_map<std::string, Dbm> invmemo;   // rebuilt on demand, only in the pre-closure

  // ---- seed ---------------------------------------------------------------
  for (std::size_t i = 0; i < node.size(); ++i) {
    const std::string k = disc_key_bytes(node[i].locs, node[i].ints);
    if (invrep.count(k)) continue;
    SymState u;
    u.locs = node[i].locs;
    u.ints = node[i].ints;
    u.zone = Dbm::universe(nc);                 // NOT Dbm(nc), which is the ZERO zone
    Dbm inv = Dbm::universe(nc);
    bool fc, fu;
    frozen_flags(net, u.locs, u.ints, &fc, &fu);
    std::vector<Constraint> ci;
    all_invariants(net, u.locs, ci);
    if (!fc && !fu) inv.delay();
    if (!apply_all(inv, ci)) continue;                       // vacuous key
    invrep.emplace(k, static_cast<int>(i));
    Federation stuck, esc;
    if (!state_has_deadlock(net, u, &stuck, &esc)) continue;  // Dead(q) = empty
    bool open = true;
    for (int c = 1; c <= nc; ++c)
      if (inv.get(c, 0) != kInf) { open = false; break; }
    if (open && !fc && !fu) continue;                         // divergence already covers stop
    // TWO TESTS SHARING ONE INTERSECTION.
    //
    // (a) Is Dead(q) = Inv(q)? The identity
    //         Inv \ Dead = Inv \ (Inv \ union Esc) = Inv ^ union Esc
    //     turns that into "does NO valuation of Inv(q) escape", which is |Esc| independent DBM
    //     intersections -- one per outgoing transition. The direct form,
    //     `subtract(inv, stuck).empty()`, instead peels Inv(q) once per piece of the RESIDUE,
    //     and the residue is an arrangement: 472 pieces at grc4's one partial key, 126ms, two
    //     thirds of the whole structural sweep.
    //
    //     The identity needs only `inv subseteq Z`, NOT `inv == Z`, where Z is the zone
    //     state_has_deadlock actually subtracted from. That distinction is load-bearing: at a
    //     FROZEN key state_has_deadlock skips both the delay and the invariant, so its Z is the
    //     bare universe while `inv` here has the invariant applied. Inclusion holds either way
    //     (equal when not frozen; `inv = universe ^ Inv subseteq universe = Z` when frozen), and
    //         x in inv \ (Z \ E)  iff  x in inv and not (x in Z and x notin E)  iff  x in inv ^ E
    //     follows from `x in inv => x in Z` alone.
    //
    // (b) Which escape regions are worth seeding? Dead(q) = Inv(q) \ union Esc, so a transition
    //     whose escape region misses Inv(q) subtracts nothing and its atoms decide nothing --
    //     yet each of its constraints would be seeded, pulled back through the whole key graph,
    //     and split against. Those members are dropped here.
    bool some_escapes = false;
    {
      Federation kept;
      kept.reserve(esc.size());
      for (const Dbm& W : esc) {
        Dbm t = W;
        t.intersect(inv);
        if (t.empty()) { ++am.dropped_vacuous; continue; }
        some_escapes = true;
        kept.push_back(W);
      }
      esc.swap(kept);
    }
    if (!some_escapes) continue;                              // Dead(q) = Inv(q): atoms implied
    ++am.seed_keys;
    // WHICH ZONES TO READ THE ATOMS OFF -- two exact representations of the same set, computed
    // both ways and the cheaper kept.
    //
    // Dead(q) = Inv(q) \ union Esc(q), and every stored zone satisfies Inv(q). Membership in
    // Dead(q) is therefore decided either by the atoms of the RESIDUE's disjoint pieces, or by
    // the atoms of the ESCAPE zones alone -- one convex zone per outgoing transition -- since a
    // valuation is dead iff it escapes by none of them. Neither dominates: the residue's cut
    // constraints are artefacts of the peeling order and blow up on grc4 (3234 occurrences
    // against 25 for the escapes), while the escapes carry diagonals that pull back through
    // resets into non-diagonal atoms and cost csma four clock bounds the residue does not. A
    // non-diagonal atom is weighted four times heavier in the comparison because it is
    // discharged by raising that clock to M GLOBALLY rather than by a local cell.
    //
    // The atoms come from the MINIMAL GRAPH, not the dense matrix: a canonical DBM's redundant
    // entries are implied by shortest paths through the others, so the non-redundant edges alone
    // decide membership. Reading the dense matrix seeds one atom per DERIVED bound and buys
    // nothing.
    const int dim = nc + 1;
    std::vector<std::uint32_t> mbits(mindbm_bits2words(std::uint32_t(dim) * std::uint32_t(dim)));
    std::vector<int> mfirst(dim), mnext(dim);
    auto extract = [&](const Federation& f, std::vector<AtomC>& g,
                       std::vector<std::pair<int, std::int32_t>>& nd) {
      for (const Dbm& d : f) {
        if (d.empty()) continue;
        mindbm_analyze(d.data().data(), dim, mbits.data(), mfirst.data(), mnext.data());
        for (int a = 0; a < dim; ++a)
          for (int b2 = 0; b2 < dim; ++b2) {
            if (a == b2) continue;
            const std::uint32_t idx = std::uint32_t(a) * std::uint32_t(dim) + std::uint32_t(b2);
            if (!((mbits[idx >> 5] >> (idx & 31)) & 1u)) continue;   // redundant: implied
            const raw_t r = d.get(a, b2);
            if (r >= kInf) continue;
            if (a == 0 && r >= kZeroLe) continue;               // x_b2 >= 0: trivial
            // BOTH CONSTANT CASES. Every stored zone at q satisfies Inv(q), so an atom Inv(q)
            // IMPLIES is true on all of them and an atom Inv(q) is DISJOINT from is false on all
            // of them; either way its tag bit never varies and it distinguishes nothing. On a
            // canonical DBM both are one entry comparison.
            if (inv.get(a, b2) <= r) continue;                  // implied by Inv(q): always true
            if (raw_add(inv.get(b2, a), r) < kZeroLe) {         // disjoint from Inv(q): never
              ++am.dropped_disjoint;
              continue;
            }
            if (a > 0 && b2 > 0) {
              const AtomC at{a, b2, r};
              if (std::find(g.begin(), g.end(), at) == g.end()) g.push_back(at);
            } else {
              const int cl = (a == 0) ? b2 : a;
              const std::int32_t cst = (a == 0) ? -raw_bound(r) : raw_bound(r);
              bool seen = false;
              for (auto& e2 : nd)
                if (e2.first == cl) { e2.second = std::max(e2.second, cst); seen = true; break; }
              if (!seen) nd.push_back({cl, cst});
            }
          }
      }
    };
    std::vector<AtomC> gr, ge;
    std::vector<std::pair<int, std::int32_t>> nr, ne2;
    extract(stuck, gr, nr);
    extract(esc, ge, ne2);
    const std::size_t cr = gr.size() + 4 * nr.size();
    const std::size_t ce = ge.size() + 4 * ne2.size();
    const bool take_esc = ce <= cr;
    std::vector<AtomC>& g = take_esc ? ge : gr;
    for (const auto& e2 : (take_esc ? ne2 : nr)) {
      ++am.nondiag_atoms;
      if (e2.second > am.bump[e2.first]) am.bump[e2.first] = e2.second;
    }
    am.seed_atoms += g.size() + (take_esc ? ne2.size() : nr.size());
    if (!g.empty()) {
      if (g.size() > kMaxAtoms) return false;
      am.diag.emplace(k, std::move(g));
    }
  }

  // NOTHING TO CLOSE. The pre-closure propagates atoms BACKWARDS from keys that carry them, so
  // with no seed anywhere there is nothing to move and the key-edge relation it would move them
  // along need not be built at all. That relation costs a string key and a reset mask per graph
  // EDGE -- 9798 nodes' worth on fischer7 -- and it is the only work this function does that
  // Otf::dtheta_empty does not. Returning here is what keeps an empty-seed model a literal
  // no-op rather than a no-op plus a sweep.
  {
    bool any_seed = !am.diag.empty();
    for (int c = 1; c <= nc && !any_seed; ++c)
      if (am.bump[c] != kNoBump) any_seed = true;
    if (!any_seed) {
      am.usable = true;
      return true;
    }
  }

  // ---- the key-edge relation, with the union of each transition's resets --
  struct KE { std::string s, t; std::uint64_t r; };
  std::vector<KE> kedges;
  {
    std::unordered_set<std::string> seenE;
    const std::size_t n = std::min(adj.size(), node.size());
    for (std::size_t u = 0; u < n; ++u) {
      const std::string ks = disc_key_bytes(node[u].locs, node[u].ints);
      for (const auto& e : adj[u]) {
        std::uint64_t rm = 0;
        for (const auto& pr : tags_[e.tag].parts)
          for (int rc : net.procs[pr.first].edges[pr.second].clock_resets)
            if (rc >= 1 && rc < 64) rm |= std::uint64_t(1) << rc;
        const std::string kt = disc_key_bytes(node[e.tgt].locs, node[e.tgt].ints);
        std::string sig = ks;
        sig += '\x01';
        sig += kt;
        sig.append(reinterpret_cast<const char*>(&rm), sizeof(rm));
        if (!seenE.insert(sig).second) continue;
        kedges.push_back(KE{ks, kt, rm});
      }
    }
  }

  // ---- pre-closure fixpoint ----------------------------------------------
  bool changed = true;
  while (changed) {
    changed = false;
    if (++am.rounds > 4096) return false;
    for (const KE& e : kedges) {
      auto it = am.diag.find(e.t);
      if (it == am.diag.end()) continue;
      const std::vector<AtomC> tgt = it->second;   // copy: am.diag may rehash below
      for (const AtomC& a : tgt) {
        const int i2 = (a.i > 0 && ((e.r >> a.i) & 1)) ? 0 : a.i;
        const int j2 = (a.j > 0 && ((e.r >> a.j) & 1)) ? 0 : a.j;
        if (i2 == j2) continue;                    // both reset: a constant, not an atom
        if (i2 > 0 && j2 > 0) {
          const Dbm* ivp = nullptr;
          {
            auto mit = invmemo.find(e.s);
            if (mit == invmemo.end()) {
              auto rit = invrep.find(e.s);
              if (rit != invrep.end()) {
                Dbm rebuilt;
                if (key_invariant(net, node[rit->second], nc, rebuilt))
                  mit = invmemo.emplace(e.s, std::move(rebuilt)).first;
              }
            }
            if (mit != invmemo.end()) ivp = &mit->second;
          }
          if (ivp) {
            const Dbm& iv2 = *ivp;
            if (iv2.get(i2, j2) <= a.b) continue;                             // Inv(q) implies
            // ...and, symmetrically, Inv(q) refutes: the pullback lands outside the source key's
            // invariant, so no zone stored there can ever satisfy it. A closure atom is a
            // constraint from some OTHER key's deadlock, and the further it is pulled back the
            // less it has to do with the invariant it arrives at, so this drop matters more here
            // than at the seed.
            if (raw_add(iv2.get(j2, i2), a.b) < kZeroLe) {
              ++am.dropped_disjoint;
              continue;
            }
          }
          std::vector<AtomC>& g = am.diag[e.s];
          const AtomC at{i2, j2, a.b};
          if (std::find(g.begin(), g.end(), at) == g.end()) {
            if (g.size() >= kMaxAtoms) return false;
            g.push_back(at);
            ++am.closure_atoms;
            changed = true;
          }
        } else {
          const int cl = (i2 == 0) ? j2 : i2;
          const std::int32_t cst = (i2 == 0) ? -raw_bound(a.b) : raw_bound(a.b);
          if (cst > am.bump[cl]) {
            am.bump[cl] = cst;
            ++am.closure_atoms;
            changed = true;
          }
        }
      }
    }
  }

  // IF EVERY CLOCK ENDED UP AT M, THE CELLS ARE REDUNDANT. With L(x) = U(x) = M(x) for every
  // clock the LU preorder collapses to "equal, or both above M", which is symmetric, hence a
  // bisimulation, hence deadlock-preserving on its own (Behrmann-Bouyer-Larsen-Pelanek, STTT
  // 8(3), 2006, Lemma 1). Cell tags on top of that would only split zones a_M already keeps
  // apart, so dropping them is what stops the construction from ever being WORSE than the a_M
  // fallback it exists to beat.
  {
    bool all_at_m = true;
    for (int c = 1; c <= nc && all_at_m; ++c) {
      const std::int32_t mc = std::max(net.lbound[c], net.ubound[c]);
      if (am.bump[c] < mc) all_at_m = false;
    }
    if (all_at_m) am.diag.clear();
  }

  // ---- active-clock reduction must keep every clock the map reads ---------
  for (const auto& kv : am.diag)
    for (const AtomC& a : kv.second) {
      if (a.i > 0) am.live[a.i] = 1;
      if (a.j > 0) am.live[a.j] = 1;
    }
  for (int c = 1; c <= nc; ++c)
    if (am.bump[c] != kNoBump) { am.live[c] = 1; ++am.bumped; }
  am.keys_tagged = am.diag.size();
  am.usable = true;
  // `am.bumped` records how many clocks were raised. A bump is "a_M on one clock", so a map that
  // bumps every clock has already paid the whole a_M price and its cells are pure loss; the
  // all-at-M shortcut above catches the total case, and this counter reports the margin.
  return true;
}

/// @brief Build the whole reachable zone graph.
///
/// @pre ::Otf::setup has run.
/// @post @ref fwd_ok_ is @c false iff the initial state is vacuous; otherwise every reachable
///       node is stored and expanded, subject to cancellation.
///
///       frontier. A frontier that pushes a successor whenever `!expanded[v]` is harmless while
///       ::Otf::intern only appends, since `expanded` then goes 0 -> 1 once per node; it becomes
///       quadratic as soon as intern rewrites, because every rewrite resets `expanded[v]` and
///       re-queues `v` once per in-edge, every level. Deduplicating changes only how often a
///       node is queued, never which nodes are explored, so stored-state counts are unaffected.
void Otf::base_forward() {
      fwd_ok_ = build_all();
      if (!fwd_ok_) return;
      std::deque<int> work;
      std::vector<char> inq;
      auto push = [&](int v) {
        if (inq.size() < node.size()) inq.resize(node.size(), 0);
        if (!inq[v]) { inq[v] = 1; work.push_back(v); }
      };
      push(0);
      while (!work.empty()) {
        const int u = work.front();
        work.pop_front();
        grow();
        if (inq.size() < node.size()) inq.resize(node.size(), 0);
        inq[u] = 0;
        expand(u);                                  // no-op if still expanded
        grow();
        // A successor needs work exactly when it has no adjacency of its own: brand new, or
        // reset because a merge enlarged it.
        for (const auto& e : adj[u])
          if (!expanded[e.tgt]) push(e.tgt);
      }
}

void Otf::build_forward(GfpStats* gs) {
  setup();
  base_forward();
  fill_stats(gs);
}

bool Otf::finish_verdict(GfpStats* gs) {
  const bool violated = fwd_ok_ && final_verdict_violated();
  fill_stats(gs);
  return !violated;
}

bool Otf::eg_from_initial(GfpStats* gs) {
  setup();
  base_forward();
  return eg_verdict(gs);
}

bool Otf::eg_verdict(GfpStats* gs) {
  scc_iter_ = true;             // the verdict is one node's, so diameter is what costs here
  bool res = false;
  if (fwd_ok_ && !node.empty()) {
    grow();
    // The psi-closure REACHED FROM THE INITIAL VERTEX. ::Otf::accepting_closure seeds from every
    // p-node because `p --> q` accepts a witness anywhere; here the run must start at the initial
    // state, so a node reachable only by leaving psi and coming back must not be admitted.
    std::vector<int> S;
    if (expanded[0] && is_nq(0)) {
      std::vector<char> in(node.size(), 0);
      std::vector<int> stk{0};
      in[0] = 1;
      while (!stk.empty()) {
        const int v = stk.back();
        stk.pop_back();
        S.push_back(v);
        for (const auto& e : adj[v])
          if (!in[e.tgt] && expanded[e.tgt] && is_nq(e.tgt)) { in[e.tgt] = 1; stk.push_back(e.tgt); }
      }
    }
    if (!S.empty()) {
      std::vector<Federation> W;
      gfp_on(S, W);
      if (!capped) res = !gfp_fed_empty(W[0]);
    }
  }
  fill_stats(gs);
  return res;
}

bool Otf::run(GfpStats* gs) {
  setup();
  base_forward();
  const bool violated = fwd_ok_ && final_verdict_violated();
  fill_stats(gs);
  return !violated;
}

void Otf::fill_stats(GfpStats* gs) {
  if (gs) {
    gs->fwd_nodes = node.size();
    gs->disc_keys = bucket.size();
    gs->fwd_edges = 0;
    for (const auto& a : adj) gs->fwd_edges += a.size();
    gs->iters = gfp_iters;
    gs->capped = capped;
    gs->fed_max = fed_max;
    gs->fed_avg = fed_avg;
    gs->nonempty = nonempty;
    gs->expanded_nodes = expand_calls;
    gs->passes = passes;
    gs->pre_calls = pre_calls;
    gs->early = settled_early;
    // The reported state count EXCLUDES the initial state, matching verifyta: its `E<> p`
    // figures are this count minus one on every query measured (5 against 6, 6 against 7, 9
    // against 10), which pins the convention as "states generated", not "states stored".
    gs->outer_nodes = outer_nodes;
    gs->outer_expands = outer_expands;
  }
}

/// @brief Decide `p --> q` on @p net.
///
/// @param net     the network under analysis
/// @param p,q     the leads-to antecedent and consequent, as discrete-state predicates
/// @param[out] gs statistics of the run that produced the verdict; may be @c nullptr
/// @return @c true iff `p --> q` **holds**
///
/// @par Why the search is staged
/// a_LU covering is a simulation. It preserves the positive witnesses — an infinite run of
/// discrete steps, and time divergence — unconditionally, because a simulating valuation does at
/// least what the simulated one does. It does *not* preserve the third witness, deadlock, which
/// is a universally quantified negative property. So ::Otf::stop_set's deadlock disjunct may be
/// lost under a_LU covering, and the error is one-sided: stored zones are exact and pre/stop are
/// exact concrete operations, so a witness *found* under a_LU is genuine, and only the *absence*
/// of one is inconclusive.
///
/// The staging follows from that. The forward graph is built once under a_LU; the deadlock
/// completion's constraint map is then derived from it (::Otf::build_atom_map). An empty map is
/// itself the proof that nothing could have been lost, so the deferred backward pass settles the
/// query outright. A non-empty map is installed and the search is redone under the refined
/// covering, which preserves the sink's reachability and hence deadlock.
///
/// @note The forward pass is run *without* its backward half, because a non-empty map makes the
///       refined search replace it wholesale — on these models the backward fixpoint is around
///       80% of a phase's wall clock. ::Otf::finish_verdict pays it on exactly the paths that
///       still need the a_LU run’s own verdict.
///
/// @par The fallback race
/// When the map cannot be represented (more than 64 atoms at one key) the obligation is
/// discharged instead by racing two lanes that decide the *same* proposition: (A) prove the
/// model deadlock-free, so nothing could have been dropped; (B) redo the search with exact
/// inclusion covering, which is sound outright. Neither dominates and which wins is not
/// predictable — fischer7 costs 180420 symbolic states for (A) against 30994 for (B), arbiter6
/// 28591 against 66393 — so both run and the first conclusive answer cancels the loser. The
/// cores buy not speed but the removal of an unguessable choice.
///
/// @warning Lane A is conclusive only when it answers "deadlock-free". "A deadlock is reachable"
///          says nothing about *this* query — the dropped witness may or may not have been one —
///          so that outcome must neither publish a verdict nor cancel lane B.
/// @note Both lanes are cancellable. Lane B polls ::Otf::cancelled from ::Otf::expand, the choke
///       point for every forward step, and at the head of each backward sweep, so the pair
///       really does finish in `min(A, B)` rather than always in B's time.
/// @brief `E[] psi` from the initial state, decided by the backward greatest fixpoint.
///
/// @param net the network, already restricted by `psi`'s clock conjunction if it had one
/// @param nq  a predicate holding exactly where `psi`'s DISCRETE part fails, so `V_psi` is its
///            complement; carry an arbitrary boolean through ptctlmc::DiscPred::forbid
/// @param[out] gs statistics; may be null
/// @return @c true iff the initial state admits a maximal execution all of whose states satisfy
///         `psi`
///
/// @par Why this replaces a forward cycle search
/// The forward construction subsumes, and redirecting a successor to a strictly larger covering
/// zone can close a cycle that no concrete run realises — the unsoundness of forward cycle
/// detection on a subsumed graph (Laarman 2013, Tripakis 2009) that this engine exists to avoid.
/// `E[] psi` is the paper's own normal form for every liveness query, so it needs no machinery
/// that `p --> q` did not already need; it needs a different QUESTION asked of the same fixpoint.
///
/// @par The question
/// ::leads_to_otf asks whether SOME `p`-node has a surviving valuation, because `p --> q` is
/// violated by a witness anywhere. `E[] psi` is anchored at the initial state instead, so the
/// closure is seeded at the initial vertex rather than at every `p`-node, and the verdict reads
/// `W` at that vertex alone.
///
/// @note Testing `W[v0]` for non-emptiness, rather than for membership of the all-zero valuation,
///       is exact here. The initial zone is the delay-closure of the origin under the invariant,
///       so every valuation in it is reachable from the origin by delay; `psi`'s clock part is an
///       invariant of the restricted network and its discrete part cannot change during a delay,
///       so that delay stays inside `psi`.
bool exists_invariant_run_otf(const Network& net, const DiscPred& nq, GfpStats* gs,
                              const Network* unrestricted,
                              const detail::DelayAvoidSet* avoid) {
  static const DiscPred kTrue{};
  auto make = [&](bool exact) {
    auto e = std::make_unique<Otf>(net, kTrue, nq, exact, nullptr, unrestricted);
    e->set_avoid(avoid);
    return e;
  };
  // Diagonal models never used a_LU at all -- Otf::setup forces inclusion covering there, since
  // a_LU is unsound on diagonal constraints (Bouyer 2004) -- so this run is already exact.
  if (network_has_diagonal(net)) return make(false)->eg_from_initial(gs);

  // The SAME staging ::leads_to_otf uses, and for the same reason. Plain a_LU covering is a
  // simulation, so it preserves an infinite run and time divergence but NOT deadlock, which is
  // universally quantified: a witness found under it is genuine, the absence of one is not. The
  // deadlock-completed relation of Definition "deadlock-simulation" repairs exactly that, and it
  // is what the constraint map installs -- so the answer is to derive the map and search under
  // it, not to abandon a_LU for exact covering.
  auto e1 = make(false);
  e1->build_forward(gs);
  AtomMap am;
  if (e1->build_atom_map(am) && am.usable) {
    // EMPTY MAP: the completion adds nothing, so the covering already used preserved deadlock and
    // this run's own verdict is the answer. Its backward pass was deferred, not skipped.
    if (am.diag.empty() && am.bumped == 0) return e1->eg_verdict(gs);
    e1->release();                                  // one graph resident at a time, not two
    auto e2 = make(false);
    e2->set_atoms(&am);
    return e2->eg_from_initial(gs);
  }
  // The map could not be represented (more than 64 atoms at one key). Fall back to exact
  // inclusion covering, which is sound outright. ::leads_to_otf races this against a
  // deadlock-freedom proof; here it is taken directly, which is slower and never wrong.
  return make(true)->eg_from_initial(gs);
}

/// @brief `p --> q` where either operand constrains clocks or carries boolean structure.
///
/// @param net the network under analysis
/// @param p   the antecedent, as a positive DNF; evaluated at one valuation under `EF`
/// @param nq  `!psi` for `psi = !q`, i.e. `q` itself, as the region the run must avoid
/// @param[out] gs statistics; may be null
/// @return @c true iff `p --> q` holds
///
/// @note The forward graph is NOT restricted. `p --> q` is violated by `EF(p /\ EG !q)`, and the
///       path to the `p`-state is ordinary reachability that need not stay in `!q` -- so the
///       region confines the fixpoint only. `E[] psi` is the opposite case and restricts both.
bool leads_to_clocks(const Network& net, const DiscPred& p, const DiscPred& nq, GfpStats* gs) {
  detail::DelayAvoidSet av;
  std::vector<DiscPred> guards;
  guards.reserve(nq.forbid.size());
  for (const auto& t : nq.forbid) {
    if (t.clocks.empty()) continue;                 // purely discrete: the vertex filter has it
    detail::DelayAvoid one;
    one.forbid = Dbm::universe(net.nclocks);
    if (!apply_all(one.forbid, t.clocks) || one.forbid.empty()) continue;
    one.fup = one.forbid; one.fup.delay();
    guards.push_back(DiscPred{});
    guards.back().locs = t.locs;
    guards.back().ints = t.ints;
    av.parts.push_back(std::move(one));
  }
  for (std::size_t i = 0; i < av.parts.size(); ++i)
    av.parts[i].when = (guards[i].locs.empty() && guards[i].ints.empty()) ? nullptr : &guards[i];

  // The region's constants must reach the LU bounds and its clocks must stay live, for the same
  // reason they must in ptctlmc::exists_invariant_run: nothing else tells the abstraction that
  // the boundary matters, and a_LU would merge zones across it.
  Network bounded = net;
  if (!av.parts.empty()) {
    for (const auto& t : nq.forbid)
      for (const auto& c : t.clocks) {
        const std::int32_t k = raw_bound(c.bound) < 0 ? -raw_bound(c.bound) : raw_bound(c.bound);
        auto bump = [&](int idx, bool lo, bool hi) {
          if (idx <= 0 || idx >= (int)bounded.lbound.size()) return;
          if (lo) bounded.lbound[idx] = std::max(bounded.lbound[idx], k);
          if (hi) bounded.ubound[idx] = std::max(bounded.ubound[idx], k);
          for (auto& pl : bounded.lbound_loc) for (auto& l : pl)
            if (lo && idx < (int)l.size()) l[idx] = std::max(l[idx], k);
          for (auto& pu : bounded.ubound_loc) for (auto& l : pu)
            if (hi && idx < (int)l.size()) l[idx] = std::max(l[idx], k);
        };
        if (c.i && c.j) { bump(c.i, true, true); bump(c.j, true, true); }
        else if (c.j == 0 && c.i) bump(c.i, false, true);
        else if (c.i == 0 && c.j) bump(c.j, true, false);
        for (auto& pa : bounded.active) for (auto& l : pa) {
          if (c.i > 0 && c.i < (int)l.size()) l[c.i] = 1;
          if (c.j > 0 && c.j < (int)l.size()) l[c.j] = 1;
        }
      }
    bounded.active_mask.clear();
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
  }

  auto make = [&](bool exact) {
    auto e = std::make_unique<Otf>(bounded, p, nq, exact);
    e->set_avoid(av.parts.empty() ? nullptr : &av, /*forward=*/false);
    return e;
  };
  if (network_has_diagonal(net)) return make(false)->run(gs);
  auto e1 = make(false);
  e1->build_forward(gs);
  AtomMap am;
  if (e1->build_atom_map(am) && am.usable) {
    if (am.diag.empty() && am.bumped == 0) return e1->finish_verdict(gs);
    e1->release();
    auto e2 = make(false);
    e2->set_atoms(&am);
    return e2->run(gs);
  }
  return make(true)->run(gs);
}

bool leads_to_otf(const Network& net, const DiscPred& p, const DiscPred& q, GfpStats* gs) {
  Otf e1(net, p, q);

  // Diagonal models never used a_LU at all -- Otf::setup forces inclusion covering, since a_LU
  // is unsound on diagonal constraints (Bouyer 2004) -- so this run is already exact: there is
  // nothing to discharge and no atom construction to run.
  if (network_has_diagonal(net)) return e1.run(gs);

  // Forward only; the deferred backward pass is paid below by e1.finish_verdict() on exactly the
  // paths that still need this run's own verdict.
  e1.build_forward(gs);

  // THE CONSTRAINT MAP. It subsumes the structural certificate -- an empty map IS
  // Otf::dtheta_empty's condition -- so where the certificate fires this is the same sweep and
  // the same a_LU search node for node, and where the certificate declines this carries the
  // model instead of falling back to the race. Best of 3, `Bus.Active --> Bus.Idle` and
  // `T0.in --> T0.far`:
  //
  //     model    certificate   atom map   a_M       a_LU floor (unsound)
  //     csma6      0.138        0.018     0.142     0.012
  //     csma7      3.985        0.053     9.729     0.033
  //     grc4       0.085        0.042     0.056     0.031
  //     grc5      21.936        3.535    19.475     5.815
  //
  // grc5 is worth a second look: the REFINED covering beats the coarser one it refines (3.5s
  // against the 5.8s floor), because splitting a zone along an atom of Dead(q) also splits the
  // backward federations passing through it, and narrow federations converge faster than the
  // arrangement one wide zone produces. The straddle rate is not the bill either -- grc4 splits
  // 100% of its tagged zones and stores 1.34x the a_LU nodes, and the refined search still costs
  // what it costs. See DESIGN.md for both.
  //
  // The verdict is sound either way round: VIOLATED is genuine outright, because zones are exact
  // and pre/stop are exact concrete operations, and HOLDS is sound because the refined covering
  // preserves the sink's reachability and hence deadlock.
  AtomMap am;
  if (e1.build_atom_map(am) && am.usable) {
    if (am.diag.empty() && am.bumped == 0) {
      // EMPTY MAP: the completion adds nothing, so the a_LU covering already used was
      // deadlock-sound and this run's own verdict is the answer. Its backward fixpoint was
      // deferred, not skipped -- pay it here, exactly once, on the graph already built.
      return e1.finish_verdict(gs);
    }
    // e1 is dead from here: the map is a value that owns its atoms, and every path that still
    // needs e1's own verdict (the empty-map return above, and the fall-through below) is out of
    // reach. Releasing before e2 starts is what keeps the peak at one graph instead of two.
    e1.release();
    Otf e2(net, p, q);
    e2.set_atoms(&am);
    GfpStats gs2;
    const bool holds2 = e2.run(&gs2);
    if (gs) *gs = gs2;
    return holds2;
  }

  // The map could not be represented, so the a_LU run's verdict is owed after all: pay the
  // backward pass build_forward deferred, then fall through to the certificate and the race.
  if (!e1.finish_verdict(gs)) return false;        // genuine witness

  // The structural certificate (::Otf::dtheta_empty): if the completion's seed is empty at every
  // reachable key, the a_LU covering was sound as it stands. One federation computation per
  // discrete key and no search, against a full reachability traversal for the same conclusion.
  if (e1.dtheta_empty()) return true;

  std::atomic<bool> done{false};
  std::atomic<int> verdict{-1};                     // 0 = holds, 1 = violated
  GfpStats gs_b;

  std::thread lane_b([&] {
    Otf e(net, p, q, /*force_exact=*/true, &done);
    const bool holds = e.run(&gs_b);
    if (e.was_cancelled()) return;                  // abandoned: its verdict is not a verdict
    if (!done.exchange(true)) verdict.store(holds ? 0 : 1, std::memory_order_release);
  });

  {
    ReachStats dl;
    const bool may_dl = deadlock_reachable(net, Backend::Omp, &dl, &done);
    // A cancelled reach_engine returns false, indistinguishable from "no deadlock reachable",
    // so this lane may only be trusted if it was not the one cancelled.
    if (!done.load(std::memory_order_acquire) && !may_dl && !done.exchange(true))
      verdict.store(0, std::memory_order_release);  // a_LU found nothing and was sound
  }

  lane_b.join();
  const int v = verdict.load(std::memory_order_acquire);
  if (v == 1 && gs) *gs = gs_b;                     // lane B won, and it disagreed with a_LU
  if (v >= 0) return v == 0;

  // Unreachable as the lanes stand: lane B is cancelled only by lane A concluding
  // "deadlock-free", which publishes verdict 0. Kept so that a future change to either lane
  // cannot silently turn an inconclusive pair into a wrong answer.
  return Otf(net, p, q, /*force_exact=*/true).run(nullptr);
}

}  // namespace ptctlmc
