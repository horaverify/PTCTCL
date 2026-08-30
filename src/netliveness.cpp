/// @file netliveness.cpp
/// @brief Symbolic successors, online D refinement, and the backward liveness solve.
///
/// The private state layout and the production lifecycle are documented in
/// @ref netliveness.hpp.  In brief, `p --> q` discovers the graph and its deadlock-preserving
/// D refinement together, seals the active quotient by redirecting already stored edges, and
/// then solves
///
///     W = W /\ (NoSucc \/ Pre(W))
///
/// once, backward.  There is no preliminary LU graph and no regenerated successor relation.
/// A non-empty value at a `p /\ !q` node is a counterexample.
///
/// Transition semantics are shared with reachability through
/// @ref ptctlmc/detail/net_semantics.hpp so the two engines use the same local, binary, and
/// broadcast rules.

#include "ptctlmc/netliveness.hpp"

#ifdef PTCTLMC_HAVE_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstdio>
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

#include "ptctlmc/mindbm.hpp"
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

// Temporary profiling is intentionally opt-in: it is useful for separating successor
// generation from quotient maintenance without perturbing normal benchmark runs.
struct OnlinePerf {
  bool enabled = std::getenv("PTCTLMC_ONLINE_PROFILE") != nullptr;
  std::uint64_t cover_calls = 0;
  std::uint64_t cover_ns = 0;
  std::uint64_t intern_calls = 0;
  std::uint64_t intern_active_scan = 0;
  std::uint64_t intern_cover_probe_ns = 0;
  std::uint64_t intern_join_ns = 0;
  std::uint64_t intern_append_ns = 0;
  std::uint64_t deferred_created = 0;
  std::uint64_t deferred_replayed = 0;
  std::uint64_t deferred_promoted = 0;
  std::uint64_t coverer_calls = 0;
  std::uint64_t coverer_active_scan = 0;
  std::uint64_t compact_calls = 0;
  std::uint64_t compact_active_scan = 0;
  std::uint64_t recheck_calls = 0;
  std::uint64_t recheck_nodes = 0;
  std::uint64_t successor_ns = 0;
  std::uint64_t register_ns = 0;
  std::uint64_t register_key_ns = 0;
  std::uint64_t register_tag_ns = 0;
  std::uint64_t register_reset_ns = 0;
  std::uint64_t register_edge_ns = 0;
  std::uint64_t intern_ns = 0;
};

OnlinePerf online_perf;

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

class OnlinePerfTimer {
 public:
  explicit OnlinePerfTimer(std::uint64_t* total) : total_(total), start_(std::chrono::steady_clock::now()) {}
  ~OnlinePerfTimer() {
    *total_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - start_).count();
  }

 private:
  std::uint64_t* total_;
  std::chrono::steady_clock::time_point start_;
};
}  // namespace

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

bool AtomC::operator==(const AtomC& o) const {
  return i == o.i && j == o.j && b == o.b;
}

/// @brief Hash a discrete location/integer key without materialising its byte representation.
/// Defined beside the ordinary passed-set implementation below.
static std::uint64_t disc_hash(const LocVec& locs, const IntVec& ints);

/// @brief Avalanche a packed discrete key for the flat open-addressing table.
static std::uint64_t mix_packed_key(std::uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// ---- Otf lifetime, cache, and stored-zone helpers --------------------------

Otf::Otf(const Network& n, const DiscPred& pp, const DiscPred& qq,
         bool force_exact, const std::atomic<bool>* cancel, const Network* dead)
    : net(n), p(pp), q(qq), dead_net(dead ? dead : &n), force_exact_(force_exact),
      cancel_(cancel) {}

bool Otf::cancelled() {
  if (cancelled_) return true;
  if (cancel_ && cancel_->load(std::memory_order_relaxed)) cancelled_ = true;
  return cancelled_;
}

bool Otf::is_nq_fast(int i) {
  if (qc[i] == 2) qc[i] = holds(q, node[i]) ? 1 : 0;
  return qc[i] == 0;
}

void Otf::set_eg_early(bool v) { eg_early_ = v; }

bool Otf::eg_early_hit() const { return eg_early_hit_; }

bool Otf::was_cancelled() const { return cancelled_; }

bool Otf::was_capped() const { return capped; }

void Otf::set_avoid(const detail::DelayAvoidSet* a, bool forward) {
  avoid_ = a;
  avoid_forward_ = forward;
}

void Otf::set_atoms(const AtomMap* a) { amap_ = a; }

void Otf::release() {
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

void Otf::zstore(int i, Dbm&& z) {
  zs_.store(i, z);
  node[i].zone = Dbm();
}

const Dbm& Otf::zread(int i, Dbm& sc) const {
  zs_.load(i, sc);
  return sc;
}

const SymState& Otf::sread(int i, SymState& sc) const {
  sc.locs.assign(node[i].locs.begin(), node[i].locs.end());
  sc.ints.assign(node[i].ints.begin(), node[i].ints.end());
  zread(i, sc.zone);
  return sc;
}

bool Otf::alu_le_stored(const Dbm& z, int j) const {
  return zs_.with(j, [&](const auto* pz) {
    return dbm_is_alu_le(z.data().data(), pz, zdim_, bL.data(), bU.data());
  });
}

bool Otf::alu_le_new(int j, const Dbm& z) const {
  return zs_.with(j, [&](const auto* pz) {
    return dbm_is_alu_le(pz, z.data().data(), zdim_, bL.data(), bU.data());
  });
}

bool Otf::stored_includes(int j, const Dbm& z) const {
  return zs_.with(j, [&](const auto* pz) {
    return dbm_includes(pz, z.data().data(), zdim_);
  });
}

bool Otf::includes_stored(const Dbm& z, int j) const {
  return zs_.with(j, [&](const auto* pz) {
    return dbm_includes(z.data().data(), pz, zdim_);
  });
}

bool Otf::base_cover(const Dbm& a, const Dbm& b) const {
  return use_alu ? a.alu_subsumed_by(b, bL, bU) : b.includes(a);
}

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
  M.resize(net.nclocks + 1);
  for (int c = 0; c <= net.nclocks; ++c) M[c] = std::max(net.lbound[c], net.ubound[c]);
  M[0] = 0;
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
  // Online D deliberately uses the same active-clock reduction as the ordinary construction.
  // This remains safe while its map grows: a seed atom reads a clock in an invariant or an
  // outgoing guard at that key, so compute_active keeps it. Pullback either resets that clock
  // (making its old value irrelevant) or propagates the target's active bit to the source.
  // Thus a D-relevant value cannot be erased before a later atom needs it.
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
  if (online_d_) {
    if (online_active_.size() < n) online_active_.resize(n, 0);
    if (online_active_next_.size() < n) online_active_next_.resize(n, -1);
    if (online_active_prev_.size() < n) online_active_prev_.resize(n, -1);
    if (online_cover_.size() < n) online_cover_.resize(n, -1);
    if (online_raw_slot_.size() < n) online_raw_slot_.resize(n, -1);
    if (online_inq_.size() < n) online_inq_.resize(n, 0);
    if (online_eg_prune_ && qc.size() < n) qc.resize(n, 2);
    return;
  }
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

// ---- Online D: persistent, monotonically refined forward quotient -------------------------

void Otf::online_reset() {
  online_am_ = AtomMap{};
  online_am_.bump.assign(net.nclocks + 1, kNoBump);
  online_am_.live.assign(net.nclocks + 1, 0);
  online_am_.usable = true;
  online_keys_.clear();
  online_key_index_.clear();
  online_packed_key_keys_.clear();
  online_packed_key_values_.clear();
  online_packed_key_mask_ = 0;
  online_packed_key_count_ = 0;
  online_disc_shift_.assign(net.nprocs() + static_cast<int>(net.intvars.size()), 0);
  online_disc_mask_.assign(net.nprocs() + static_cast<int>(net.intvars.size()), 0);
  online_packed_key_usable_ = true;
  int disc_bits = 0;
  const auto field_bits = [](long nvalues) {
    int b = 0;
    std::uint64_t cap = 1;
    while (cap < static_cast<std::uint64_t>(nvalues) && b < 63) { cap <<= 1; ++b; }
    if (cap < static_cast<std::uint64_t>(nvalues)) return 64;
    return b;
  };
  for (int p = 0; p < net.nprocs(); ++p) {
    online_disc_shift_[p] = disc_bits;
    const int width = field_bits(static_cast<long>(net.procs[p].locations.size()));
    online_disc_mask_[p] = width == 0 ? 0 : (width >= 64 ? ~std::uint64_t(0)
                                                        : ((std::uint64_t(1) << width) - 1));
    disc_bits += width;
  }
  for (std::size_t v = 0; v < net.intvars.size(); ++v) {
    const int field = net.nprocs() + static_cast<int>(v);
    online_disc_shift_[field] = disc_bits;
    const int width = field_bits(net.intvars[v].hi - net.intvars[v].lo + 1);
    online_disc_mask_[field] = width == 0 ? 0 : (width >= 64 ? ~std::uint64_t(0)
                                                            : ((std::uint64_t(1) << width) - 1));
    disc_bits += width;
  }
  // Keep every shift strictly below 64.  Wider networks retain the previous exact vector-key
  // lookup; this fast path is optional and never aliases two discrete states.
  if (disc_bits >= 64) {
    online_packed_key_usable_ = false;
    online_disc_shift_.clear();
    online_disc_mask_.clear();
  } else {
    constexpr std::size_t initial_capacity = 1024;
    online_packed_key_keys_.resize(initial_capacity);
    online_packed_key_values_.assign(initial_capacity, -1);
    online_packed_key_mask_ = initial_capacity - 1;
  }
  online_key_by_id_.clear();
  online_local_bounds_narrow_ = true;
  const auto narrow_bound = [&](std::int32_t value) {
    return value == kAluNegInf ||
           (value > std::numeric_limits<std::int16_t>::min() &&
            value <= std::numeric_limits<std::int16_t>::max());
  };
  for (std::int32_t value : net.lbound) online_local_bounds_narrow_ &= narrow_bound(value);
  for (std::int32_t value : net.ubound) online_local_bounds_narrow_ &= narrow_bound(value);
  for (const auto& proc : net.lbound_loc)
    for (const auto& loc : proc)
      for (std::int32_t value : loc) online_local_bounds_narrow_ &= narrow_bound(value);
  for (const auto& proc : net.ubound_loc)
    for (const auto& loc : proc)
      for (std::int32_t value : loc) online_local_bounds_narrow_ &= narrow_bound(value);
  online_local_l16_.clear();
  online_local_u16_.clear();
  online_local_l_.clear();
  online_local_u_.clear();
  online_key_of_.clear();
  online_next_in_key_.clear();
  online_key_edges_.clear();
  online_key_graph_enabled_ = false;
  online_reset_sets_.clear();
  online_tag_reset_class_.clear();
  online_atom_work_.clear();
  online_dirty_keys_.clear();
  online_dirty_queued_.clear();
  online_global_dirty_ = false;
  online_failed_ = false;
  online_bound_key_ = -1;
  online_bound_epoch_ = 0;
  online_bound_cached_epoch_ = std::numeric_limits<std::size_t>::max();
  online_active_.clear();
  online_active_next_.clear();
  online_active_prev_.clear();
  online_cover_.clear();
  online_raw_adj_.clear();
  online_raw_owner_.clear();
  online_raw_slot_.clear();
  online_deferred_index_enabled_ = false;
  online_work_.clear();
  online_inq_.clear();
}

/// @brief Compute the current D-covering bounds at one discrete key.
///
/// `AtomMap::bump` is append-only in online mode.  It is intentionally applied here, at every
/// cover check, rather than once in setup: a later non-diagonal deadlock atom must refine every
/// existing certificate without changing any stored exact zone.
void Otf::online_configure_bounds(int key) {
  if (!use_alu) return;
  if (online_bound_cached_epoch_ == online_bound_epoch_ && online_bound_key_ == key) return;
  online_bound_key_ = key;
  online_bound_cached_epoch_ = online_bound_epoch_;
  const std::size_t dim = static_cast<std::size_t>(net.nclocks + 1);
  const std::size_t off = static_cast<std::size_t>(key) * dim;
  if (online_local_bounds_narrow_) {
    bL.resize(dim);
    bU.resize(dim);
    constexpr std::int16_t narrow_inf = std::numeric_limits<std::int16_t>::min();
    for (std::size_t c = 0; c < dim; ++c) {
      bL[c] = online_local_l16_[off + c] == narrow_inf
                  ? kAluNegInf : online_local_l16_[off + c];
      bU[c] = online_local_u16_[off + c] == narrow_inf
                  ? kAluNegInf : online_local_u16_[off + c];
    }
  } else {
    bL.assign(online_local_l_.begin() + static_cast<std::ptrdiff_t>(off),
              online_local_l_.begin() + static_cast<std::ptrdiff_t>(off + dim));
    bU.assign(online_local_u_.begin() + static_cast<std::ptrdiff_t>(off),
              online_local_u_.begin() + static_cast<std::ptrdiff_t>(off + dim));
  }
  for (int c = 1; c <= net.nclocks && c < (int)online_am_.bump.size(); ++c)
    if (online_am_.bump[c] != kNoBump) {
      bL[c] = std::max(bL[c], online_am_.bump[c]);
      bU[c] = std::max(bU[c], online_am_.bump[c]);
    }
}

/// @brief Does the exact candidate @p a have a current D cover in stored node @p b?
bool Otf::online_covers(const SymState& a, int b, int key) {
  const bool profile = online_perf.enabled;
  if (profile) ++online_perf.cover_calls;
  std::unique_ptr<OnlinePerfTimer> timer;
  if (profile) timer = std::make_unique<OnlinePerfTimer>(&online_perf.cover_ns);
  online_configure_bounds(key);
  Dbm bsc;
  const Dbm& bz = zread(b, bsc);
  const OnlineKeyInfo& ki = *online_key_by_id_[key];
  if (!ki.diag) return base_cover(a.zone, bz);
  const std::vector<AtomC>& diag = *ki.diag;
  if (diag.size() > 64) return online_cover_rec(a.zone, bz, diag, 0);
  std::uint64_t ain, aout, bin, bout;
  classify_atoms(a.zone, diag, &ain, &aout);
  classify_atoms(bz, diag, &bin, &bout);
  return cover_rec(a.zone, ain, bz, bout, diag, 0);
}

/// @brief Does stored node @p a have a current D cover in stored node @p b?
///
/// Both records are already known to share a discrete key.  Keeping this form avoids rebuilding
/// a temporary SymState (and copying its location and integer vectors) during compaction and
/// certificate rechecks.
bool Otf::online_covers(int a, int b, int key) {
  const bool profile = online_perf.enabled;
  if (profile) ++online_perf.cover_calls;
  std::unique_ptr<OnlinePerfTimer> timer;
  if (profile) timer = std::make_unique<OnlinePerfTimer>(&online_perf.cover_ns);
  online_configure_bounds(key);
  Dbm asc;
  const Dbm& az = zread(a, asc);
  Dbm bsc;
  const Dbm& bz = zread(b, bsc);
  const OnlineKeyInfo& ki = *online_key_by_id_[key];
  if (!ki.diag) return base_cover(az, bz);
  const std::vector<AtomC>& diag = *ki.diag;
  if (diag.size() > 64) return online_cover_rec(az, bz, diag, 0);
  std::uint64_t ain, aout, bin, bout;
  classify_atoms(az, diag, &ain, &aout);
  classify_atoms(bz, diag, &bin, &bout);
  return cover_rec(az, ain, bz, bout, diag, 0);
}

/// @brief The D-cell obligation without the ordinary engine's 64-bit classification cache.
///
/// The common path uses ::cover_rec, whose masks make atom comparisons essentially free.  A
/// genuinely online construction cannot turn a newly found atom 65 into a restart, though, so
/// this slow-path evaluates the same branch rule directly from canonical DBM entries.  It is
/// reached only for unusually wide key-local atom sets; all atom positions remain append-only.
bool Otf::online_cover_rec(const Dbm& a, const Dbm& b, const std::vector<AtomC>& as, int depth) {
  if (!base_cover(a, b)) return false;
  for (const AtomC& at : as) {
    const bool ain = a.get(at.i, at.j) <= at.b;  // every valuation of a satisfies phi
    const bool bout = raw_add(b.get(at.j, at.i), at.b) < kZeroLe;  // b is disjoint from phi
    if (ain || bout) continue;

    // Identical to cover_rec's split: on phi, b itself is enough; on !phi, b must be cut too.
    Dbm ap = a;
    if (ap.constrain(at.i, at.j, at.b) && !online_cover_rec(ap, b, as, depth + 1)) return false;
    const Constraint nc = negate(Constraint{at.i, at.j, at.b});
    Dbm an = a;
    if (!an.constrain(nc.i, nc.j, nc.bound)) return true;
    Dbm bn = b;
    if (!bn.constrain(nc.i, nc.j, nc.bound)) return false;
    return online_cover_rec(an, bn, as, depth + 1);
  }
  return true;
}

int Otf::online_append(SymState&& s, int key) {
  const int id = static_cast<int>(node.size());
  Dbm z = std::move(s.zone);
  // During the online phase the immutable discrete half lives once in OnlineKeyInfo.  Parking
  // a provisional record only needs its exact zone and this key id; copying its location and
  // integer vectors into every raw candidate was pure duplication and dominated memory.
  node.emplace_back();
  zstore(id, std::move(z));
  online_key_of_.push_back(key);
  OnlineKeyInfo& ki = *online_key_by_id_[key];
  online_next_in_key_.push_back(ki.node_head);
  ki.node_head = id;
  ++ki.node_count;
  ++outer_nodes;
  grow();
  return id;
}

/// @brief Return a current D coverer for @p v, optionally requiring completed successors.
///
/// A queued active representative is a valid *provisional* coverer: it is guaranteed to be
/// expanded before the forward phase can quiesce, and if a later representative replaces it,
/// that replacement has already been expanded and covers it transitively.  Allowing this avoids
/// activating a whole same-level antichain merely because its first representative has not yet
/// reached the front of the FIFO queue.  Final edge redirection asks for an expanded coverer.
int Otf::online_coverer(int v, bool require_expanded) {
  if (v < 0 || v >= (int)node.size()) return -1;
  if (online_active_[v] && (!require_expanded || expanded[v])) return v;
  const OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[v]];
  if (online_perf.enabled) {
    ++online_perf.coverer_calls;
    online_perf.coverer_active_scan += ki.active_count;
  }
  const int cached = online_cover_[v];
  if (cached >= 0 && cached < (int)node.size() && online_active_[cached] &&
      (!require_expanded || expanded[cached]) && online_covers(v, cached, ki.id))
    return cached;
  for (int j = ki.active_head; j >= 0; j = online_active_next_[j])
    if (j != v && online_active_[j] && (!require_expanded || expanded[j]) &&
        online_covers(v, j, ki.id)) return j;
  return -1;
}

/// @brief Resolve a previously recorded cover without repeating a D comparison.
///
/// The forward worklist rechecks every inactive materialised record after each map refinement.
/// Once quiescent, its cover points to an active representative or through another inactive
/// record to one.  D-covering is transitive, so final edge redirection can chase that chain
/// without repeating a DBM cover test for every raw edge.
int Otf::online_resolve_cover(int v, bool require_expanded) {
  if (v < 0 || v >= (int)node.size()) return -1;
  int cur = v;
  std::size_t hops = 0;
  while (!online_active_[cur]) {
    const int next = online_cover_[cur];
    if (next < 0 || next >= (int)node.size() || next == cur || ++hops > node.size()) return -1;
    cur = next;
  }
  if (require_expanded && !expanded[cur]) return -1;

  // Path compression makes repeated redirects through an inactive record constant-time.
  while (v != cur && !online_active_[v]) {
    const int next = online_cover_[v];
    online_cover_[v] = cur;
    v = next;
  }
  return cur;
}

/// @brief Remember the current D certificate of an inactive materialised record.
void Otf::online_set_cover(int v, int cover) {
  if (v < 0 || v >= (int)node.size()) return;
  online_cover_[v] = cover;
}

void Otf::online_activate(int v) {
  if (v < 0 || v >= (int)node.size()) return;
  if (!online_active_[v]) {
    online_active_[v] = 1;
    online_set_cover(v, -1);
    OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[v]];
    online_active_prev_[v] = -1;
    online_active_next_[v] = ki.active_head;
    if (ki.active_head >= 0) online_active_prev_[ki.active_head] = v;
    ki.active_head = v;
    ++ki.active_count;
  }
  if (!expanded[v] && !online_inq_[v]) {
    online_inq_[v] = 1;
    // Before D changes, preserve ordinary FIFO order: it produces smaller aLU antichains on
    // dense keys.  After the first change, prioritise the first record at a new key so a
    // downstream seed can propagate before sibling zones expand under the coarser map.
    const OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[v]];
    bool key_seen_semantically = false;
    for (int r = ki.node_head; r >= 0; r = online_next_in_key_[r])
      if (r != v && expanded[r]) { key_seen_semantically = true; break; }
    if (key_seen_semantically || !online_deferred_index_enabled_) online_work_.push_back(v);
    else online_work_.push_front(v);
  }
}

/// @brief Replace only the *representative status* of zones a fresh active node covers.
///
/// The old records and their raw outgoing edges stay intact.  A future map refinement may wake
/// one again, but no enlarged zone or stale adjacency ever has to be reconstructed.  The new
/// representative is allowed to be queued rather than expanded: its eventual expansion is the
/// simulation witness for the provisional cover, and a refinement can revoke it before then.
void Otf::online_compact_key(int v) {
  if (!online_active_[v]) return;
  OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[v]];
  if (online_perf.enabled) {
    ++online_perf.compact_calls;
    online_perf.compact_active_scan += ki.active_count;
  }
  for (int j = ki.active_head; j >= 0;) {
    const int next = online_active_next_[j];
    // Keep the distinguished initial record active.  Leads-to may quotient it away safely, but
    // E[] reads vertex 0 directly; retaining this one representative keeps the online graph
    // reusable for that query shape too without a special root redirect.
    // An unexpanded record already has an exact incoming edge, but not yet its own semantic
    // adjacency.  Redirecting that incoming edge to a simulator is sufficient for forward
    // reachability, not for the exact backward predecessor used below.  Let it expand once;
    // after that it may leave the forward antichain while remaining as a historical graph
    // vertex with its exact zone and exact outgoing edges.
    if (j == 0 || j == v || !online_active_[j] ||
        (online_eg_repair_ && !expanded[j])) {
      j = next;
      continue;
    }
    if (online_covers(j, v, ki.id)) {
      online_active_[j] = 0;
      online_set_cover(j, v);
      const int prev = online_active_prev_[j];
      if (prev >= 0) online_active_next_[prev] = next;
      else ki.active_head = next;
      if (next >= 0) online_active_prev_[next] = prev;
      online_active_next_[j] = online_active_prev_[j] = -1;
      --ki.active_count;
      // A stale deque entry is harmless (the consumer checks online_active_), but its bit must
      // not suppress a later reactivation after a D refinement invalidates this cover.
      online_inq_[j] = 0;
      j = next;
      continue;
    }
    j = next;
  }
}

/// @brief Intern an active candidate or defer a revocably covered successor by replay recipe.
int Otf::online_intern(SymState&& s, int key, int source, int ordinal, int* deferred) {
  if (deferred) *deferred = -1;
  // online_build_all registers the root, and online_expand registers its whole successor batch
  // and settles the atom worklist *before* it gets here.  Keeping that boundary batched matters:
  // doing a no-op map drain and a no-op inactive-record scan for every successor was pure hot-path
  // overhead, and it could not reveal a D refinement that the preceding batch phase missed.
  OnlineKeyInfo& ki = *online_key_by_id_[key];
  const bool profile = online_perf.enabled;
  if (profile) {
    ++online_perf.intern_calls;
    online_perf.intern_active_scan += ki.active_count;
  }
  std::chrono::steady_clock::time_point cover_start;
  if (profile) cover_start = std::chrono::steady_clock::now();
  int cover = -1;
  for (int j = ki.active_head; j >= 0; j = online_active_next_[j])
    if (online_active_[j] && online_covers(s, j, key)) { cover = j; break; }
  if (profile)
    online_perf.intern_cover_probe_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - cover_start).count();

  // Unlike an a_LU/D certificate, literal zone inclusion cannot be invalidated by a later atom:
  // every concrete valuation and every successor of `s` is already present in `cover`.  Do not
  // retain a fake "provisional" record in that case.  The append-only provenance is needed only
  // for covers that rely on the relation currently being refined.
  if (cover >= 0 && stored_includes(cover, s.zone)) return cover;

  // Do not turn an ordinary provisional cover into a full stored state.  The expanded source
  // is immutable, and successor generation is deterministic, so (source, ordinal) is an exact
  // recipe for this candidate.  A later D refinement replays only recipes at the affected key;
  // until then the raw edge points at the current representative.  Literal inclusion has already
  // returned above, so every recipe retained here is genuinely revocable.
  if (cover >= 0 && source >= 0 && ordinal >= 0) {
    const int raw_slot = source < static_cast<int>(online_raw_slot_.size())
                             ? online_raw_slot_[source]
                             : -1;
    if (raw_slot < 0 || static_cast<std::uint64_t>(raw_slot) > UINT32_MAX ||
        static_cast<std::uint64_t>(ordinal) > UINT32_MAX) {
      online_failed_ = true;
      return cover;
    }
    const std::uint64_t handle = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(raw_slot)) << 32) |
                                 static_cast<std::uint32_t>(ordinal);
    if (online_deferred_index_enabled_) ki.deferred.push_back(handle);
    if (deferred) *deferred = 1;
    if (profile) ++online_perf.deferred_created;
    return cover;
  }

  // The source-less path is used for the root and for a deferred successor that has just lost
  // its cover.  Exact convex-union canonicalisation may still keep the active antichain small;
  // the union is set-theoretically exact and is materialised as a fresh immutable record.
  std::chrono::steady_clock::time_point join_start;
  if (profile) join_start = std::chrono::steady_clock::now();
  if (cover < 0) {
    Dbm old;
    for (int j = ki.active_head; j >= 0; j = online_active_next_[j]) {
      // Keep vertex 0 as the E[] anchor.  It is harmlessly redundant for leads-to, but
      // absorbing it here would force a special root redirect later.
      if (j == 0 || !online_active_[j]) continue;
      Dbm joined = s.zone;
      if (!joined.try_union(zread(j, old)) || joined == s.zone) continue;
      s.zone = std::move(joined);
      break;
    }
  }
  if (profile)
    online_perf.intern_join_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - join_start).count();

  std::chrono::steady_clock::time_point append_start;
  if (profile) append_start = std::chrono::steady_clock::now();
  const int id = online_append(std::move(s), key);
  if (profile)
    online_perf.intern_append_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - append_start).count();
  if (cover < 0) {
    online_activate(id);               // otherwise the materialised record remains inactive
    // Rebalance before this new representative reaches the work queue.  If it covers a queued
    // peer, the peer need not generate successors before becoming inactive; if a future atom
    // breaks the cover, that materialised record becomes active again.
    online_compact_key(id);
  } else {
    online_set_cover(id, cover);
  }
  return id;
}

/// @brief Regenerate exactly one provisionally covered successor from its immutable source.
bool Otf::online_replay(int source, int ordinal, int key, SymState& out) const {
  if (source < 0 || source >= static_cast<int>(node.size()) || ordinal < 0 || key < 0 ||
      key >= static_cast<int>(online_key_by_id_.size()))
    return false;

  const int source_key = online_key_of_[source];
  if (source_key < 0 || source_key >= static_cast<int>(online_key_by_id_.size())) return false;
  const OnlineKeyInfo& sk = *online_key_by_id_[source_key];
  SymState src;
  src.locs = sk.locs;
  src.ints = sk.ints;
  zread(source, src.zone);

  std::vector<SymState> succ;
  std::vector<GfpTag> stags;
  gfp_succ_tagged(net, src, L, U, qmask, emode, succ, stags,
                  avoid_forward_ ? avoid_ : nullptr);
  if (ordinal >= static_cast<int>(succ.size())) return false;

  const OnlineKeyInfo& tk = *online_key_by_id_[key];
  SymState& candidate = succ[static_cast<std::size_t>(ordinal)];
  if (candidate.locs != tk.locs || candidate.ints != tk.ints) return false;
  out = std::move(candidate);
  return true;
}

/// @brief Test the discrete E[] restriction while online nodes still share key storage.
bool Otf::online_is_nq(int v) {
  if (v < 0 || v >= static_cast<int>(online_key_of_.size())) return false;
  grow();
  if (qc[v] == 2) {
    const OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[v]];
    SymState state;
    state.locs = ki.locs;
    state.ints = ki.ints;
    qc[v] = holds(q, state) ? 1 : 0;
  }
  return qc[v] == 0;
}

/// @brief Certify an E[] witness that ends here, without running a partial backward solve.
///
/// Every materialised online record is an exact reachable zone reached through states that have
/// stayed inside psi.  A time-divergent valuation or a genuine deadlock of the unrestricted
/// network is therefore already a concrete maximal psi-run.  This is stronger than a graph-cycle
/// guess: no subsumption edge participates in the certificate.
bool Otf::online_eg_terminal_witness(int v, bool no_restricted_successor) {
  if (!online_eg_prune_ || v < 0 || v >= static_cast<int>(online_key_of_.size())) return false;
  const OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[v]];
  SymState s;
  s.locs = ki.locs;
  s.ints = ki.ints;
  zread(v, s.zone);
  if (time_diverges(s)) return true;

  // A universe seed that proved Dead(key) empty cannot contain a deadlocked sub-zone here.
  // Conversely, a key with a possible deadlock is tested exactly on this reached zone.  The
  // cheap no-successor case is also tested because psi may have removed every restricted edge.
  if (!ki.may_deadlock && !no_restricted_successor) return false;
  if (dead_net != &net || avoid_) {
    // If the original location lets time run for ever, an action-deadlocked valuation continues
    // by delay outside psi and is not the end of a maximal run.
    if (orig_time_open(s)) return false;
  }
  Federation dead;
  return state_has_deadlock(*dead_net, s, &dead);
}

/// @brief Hash a discrete state without allocating its byte key.
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
    // Lemma 3). Raising every clock yields the global a_M relation; the production online path
    // refines only the clocks required by its discovered D atoms.
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

/// @brief Reference-path hook for atom splitting.
///
/// The production online path never calls this routine: it discharges atom obligations while
/// testing a cover rather than splitting materialised records.  The reference path uses the same
/// comparison-time strategy, so its output is the original zone as one piece.
void Otf::split_by_atoms(SymState&& s, std::vector<SymState>& out) {
  out.clear();
  out.push_back(std::move(s));
}

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

/// @brief Generate and intern every symbolic successor of node @p u.
///
/// A rewritten representative has successors computed for an older, smaller zone.  Its cached
/// adjacency is therefore cleared and the ordinary forward worklist expands it again.  This is
/// the non-online exact/inclusion path only; online records are immutable and use
/// ::Otf::online_expand instead.
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
///       ptctlmc::detail::state_has_deadlock supplies it, recomputing the delay closure first so
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

  // ONE NODE'S REFINEMENT, `W[v] /\ (NoSucc(v) \/ Pre(W))`. Factored out so the lockstep sweep
  // and the worklist below drive exactly the same operator; only the ORDER differs.
  auto relax = [&](int v) -> bool {
    if (!seeded[v] && gfp_fed_empty(W[v])) return false;
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
    const bool shrank = !gfp_fed_subset(W[v], nw);
    W[v].swap(nw);
    return shrank;
  };

  // WORKLIST INSIDE A COMPONENT, for `E[] psi` only. `relax` reads only v's SUCCESSORS, so v can
  // change only after one of them has -- and the lockstep sweep re-reads every node of the
  // component every round regardless. That is free where the decomposition is fine-grained
  // (grc5: 2717 components over |S|=7079, 58916 node visits) and ruinous where the closure is a
  // single component, which is the common shape here:
  //
  //     fischer7  |S|=3774  sccs=1     280 rounds   985707 visits   3.45M Pre   26.3s
  //     twolock7  |S|=4158  sccs=1     126 rounds   449145 visits   1.57M Pre   12.6s
  //     grc5      |S|=7079  sccs=2717 5458 rounds    58916 visits   0.30M Pre    2.5s
  //
  // DESIGN.md records a chaotic worklist as measured and REJECTED, because reordering changes
  // the intermediate federation shapes and `fedcap` bounds those -- it turned grc4
  // `T0.in --> T0.far` UNUSABLE. That measurement was on leads-to, which is why this is confined
  // to @ref scc_iter_, which is now on for `p --> q` as well; see its declaration.
  // The E[] suite runs at a maximum federation width of 186 against a cap of 4096, with zero cap
  // hits across 108 rows, so the bound that killed it there is not close to binding here.
  const bool worklist = scc_on && [] {
    const char* e = std::getenv("PTCTLMC_GFP_WORKLIST");
    return !e || *e != '0';
  }();

  std::vector<std::vector<int>> rev;
  std::vector<char> in_comp, in_q;
  if (worklist) {
    rev.resize(N);
    for (int v : S)
      for (const auto& e : adj[v])
        if (inS[e.tgt]) rev[e.tgt].push_back(v);
    in_comp.assign(N, 0);
    in_q.assign(N, 0);
  }

  std::size_t it = 0;
  for (const auto& comp : sccs) {
   if (capped || cancelled()) break;
   if (!worklist) {
    bool changed = true;
    while (changed && !capped) {
     if (cancelled()) break;
     changed = false;
     ++it;
     for (int v : comp) {
       if (relax(v)) changed = true;
       if (capped) break;
     }
     if (it > 100000) break;
    }
    continue;
   }
   // A predecessor OUTSIDE the component needs no attention: Tarjan emits components
   // successors-first, so such a predecessor lives in a component not yet processed and will
   // read this one's final value when its own turn comes.
   for (int v : comp) in_comp[v] = 1;
   std::deque<int> work(comp.begin(), comp.end());
   for (int v : comp) in_q[v] = 1;
   while (!work.empty() && !capped) {
     if (cancelled()) break;
     const int v = work.front();
     work.pop_front();
     in_q[v] = 0;
     ++it;
     if (it > 100000000) break;
     if (!relax(v)) continue;
     for (int u : rev[v])
       if (in_comp[u] && !in_q[u]) { in_q[u] = 1; work.push_back(u); }
   }
   for (int v : comp) { in_comp[v] = 0; in_q[v] = 0; }
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
  if (online_d_) {
    // Unlike ordinary intern(), this retains the exact root even if a future key-level atom
    // will make its first cover certificate invalid.  The root is registered before it is
    // compared, so a local Dead(root-key) seed participates from the very first D decision.
    const int root_key = online_register_key(s0.locs, s0.ints);
    online_drain_atoms();
    online_recheck_dirty();
    online_intern(std::move(s0), root_key);
  } else if (amap_) {
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

/// @brief Reference-only check for whether a finished graph needs any D atom.
///
/// No public liveness route calls this method.  The production construction seeds the same
/// condition as each key is discovered.  This version remains useful for comparing the online
/// seeding rule against a finished graph.
///
/// A key contributes no atom when `Dead(q)` is empty, covers its whole invariant, or time can
/// diverge from that non-frozen key.  The check is structural: it evaluates `Dead(q)` over the
/// key invariant rather than over individual reached zones.
bool Otf::dtheta_empty() {
  std::unordered_set<std::string> seen;
  for (std::size_t i = 0; i < node.size(); ++i) {
    const std::string k = disc_key_bytes(node[i].locs, node[i].ints);
    if (!seen.insert(k).second) continue;
    SymState u; u.locs = node[i].locs; u.ints = node[i].ints;
    u.zone = Dbm::universe(net.nclocks);
    Dbm inv = Dbm::universe(net.nclocks);
    bool c2, u2; frozen_flags(*dead_net, u.locs, u.ints, &c2, &u2);
    std::vector<Constraint> ci; all_invariants(*dead_net, u.locs, ci);
    if (!c2 && !u2) inv.delay();
    if (!apply_all(inv, ci)) continue;                        // vacuous key
    Federation stuck, esc;
    if (!state_has_deadlock(*dead_net, u, &stuck, &esc)) continue;   // Dead(q) = empty
    bool open = true;
    for (int c = 1; c <= net.nclocks; ++c)
      if (inv.get(c, 0) != kInf) { open = false; break; }
    // The time-open drop is a claim about DIVERGENCE, so it is available only where time can
    // pass. At a committed/urgent key no positive delay is legal, so Inv(q) is the undelayed
    // universe and trivially bounds no clock -- exactly when this test would fire and exactly
    // when it would be wrong. Frozen keys fall through to the Dead=Inv test instead.
    if (open && !c2 && !u2) continue;                          // divergence already covers stop
    // Inv \ Dead = Inv ^ union Esc, so "Dead(q) = Inv(q)" is "nothing in Inv(q) escapes": one
    // DBM intersection per outgoing transition, instead of peeling the invariant by a residue.
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


/// @brief Reference-only fixed-map construction from a finished graph.
///
/// No production route calls this routine.  It is retained to compare online seed/pullback
/// behaviour against the old finished-graph formulation.  It derives diagonal cell atoms and
/// non-diagonal LU-bound raises from partial `Dead(q)` regions, then closes them backwards over
/// reset-labelled edges.
///
/// The representation is intentionally limited to 64 diagonal atoms per key because this
/// diagnostic path uses a 64-bit cached classification.  The online path switches to a dynamic
/// classification path and does not inherit that limit.
///
/// @brief Reconstruct a key invariant from its discrete representative.
///
/// Both the online and reference seed rules use an invariant obtained by delay-closing the
/// universe unless the key is frozen, then applying the location invariants.
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
  // DEADLOCK IS JUDGED IN @ref dead_net, SO THE COMPLETION MUST BE BUILT THERE TOO.
  //
  // For `p --> q` the two are the same automaton and this is a no-op. For `E[] psi` they are not:
  // the search runs on the psi-RESTRICTED network, psi's clock conjunction having been adjoined
  // to every location invariant, while ::Otf::stop_set asks ptctlmc::detail::state_has_deadlock
  // about the UNRESTRICTED one -- maximality is a property of the original system. Deriving
  // Dead(q) from `net` here would complete the covering for the restricted automaton's deadlocks
  // and leave the fixpoint reading the unrestricted automaton's, so the covering could merge two
  // states that differ in exactly the property the completion exists to preserve. That is the
  // grc5 `E[] x0 <= 15 && !T0.in` wrong answer, and it is why the sibling `E[] !T0.in`, which
  // has no clock part and therefore no second network, was always right.

  const int nc = net.nclocks;
  am.diag.clear();
  am.bump.assign(nc + 1, kNoBump);
  am.live.assign(nc + 1, 0);
  am.usable = false;
  if (nc <= 0 || nc >= 63) return false;

  // The reference representation uses a 64-bit atom cache.  It must decline rather than silently
  // omit an atom from the covering relation.
  constexpr std::size_t kMaxAtoms = 64;

  // Store one representative node per key; rebuild its invariant only if backward closure needs
  // it.  The invariant depends on the discrete key, not on the particular reached zone.
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
    frozen_flags(*dead_net, u.locs, u.ints, &fc, &fu);
    std::vector<Constraint> ci;
    all_invariants(*dead_net, u.locs, ci);
    if (!fc && !fu) inv.delay();
    if (!apply_all(inv, ci)) continue;                       // vacuous key
    invrep.emplace(k, static_cast<int>(i));
    Federation stuck, esc;
    if (!state_has_deadlock(*dead_net, u, &stuck, &esc)) continue;  // Dead(q) = empty
    bool open = true;
    for (int c = 1; c <= nc; ++c)
      if (inv.get(c, 0) != kInf) { open = false; break; }
    if (open && !fc && !fu) continue;                         // divergence already covers stop
    // `Dead(q) = Inv(q) \ union Esc(q)`.  First detect the constant case using the equivalent
    // question "does any valuation of Inv(q) escape?".  Then discard escape regions disjoint
    // from the invariant: their atoms cannot distinguish a stored valuation at this key.
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

  // No seed means no backward closure obligation.
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
                if (key_invariant(*dead_net, node[rit->second], nc, rebuilt))
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

  // If every clock reached M, the LU relation itself is symmetric enough to preserve deadlock;
  // keeping cell tags would only add needless splits.
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

// ---- Online D atom worklist ---------------------------------------------------------------

/// @brief Index provisional replay edges only if D actually changes.
///
/// While D is empty no key can be dirty, so per-key vectors containing every revocable edge are
/// dead weight.  The raw adjacency already contains the exact replay marker and target key.  At
/// the first real atom/bound raise, index the explored prefix once; later edges are appended to
/// their key directly.  This keeps the zero-atom case allocation-free without losing a recipe.
void Otf::online_enable_deferred_index() {
  if (online_deferred_index_enabled_) return;
  online_deferred_index_enabled_ = true;
  for (std::size_t raw_slot = 0; raw_slot < online_raw_adj_.size(); ++raw_slot) {
    if (raw_slot > UINT32_MAX) { online_failed_ = true; return; }
    std::vector<OtfEdge>& edges = online_raw_adj_[raw_slot];
    for (std::size_t ordinal = 0; ordinal < edges.size(); ++ordinal) {
      const OtfEdge& edge = edges[ordinal];
      if (edge.deferred < 0) continue;
      if (ordinal > UINT32_MAX || edge.tgt < 0 ||
          edge.tgt >= static_cast<int>(online_key_of_.size())) {
        online_failed_ = true;
        return;
      }
      const int key = online_key_of_[edge.tgt];
      const std::uint64_t handle =
          (static_cast<std::uint64_t>(static_cast<std::uint32_t>(raw_slot)) << 32) |
          static_cast<std::uint32_t>(ordinal);
      online_key_by_id_[key]->deferred.push_back(handle);
    }
  }
}

bool Otf::online_add_diag(int key, const AtomC& atom, bool closure) {
  if (key < 0 || key >= static_cast<int>(online_key_by_id_.size())) {
    online_failed_ = true;
    return false;
  }
  OnlineKeyInfo& ki = *online_key_by_id_[key];
  if (!ki.diag) ki.diag = std::make_unique<std::vector<AtomC>>();
  std::vector<AtomC>& as = *ki.diag;
  if (std::find(as.begin(), as.end(), atom) != as.end()) return false;
  online_enable_deferred_index();
  if (online_failed_) return false;
  if (!online_key_graph_enabled_) online_enable_key_graph();
  if (online_failed_) return false;
  as.push_back(atom);
  online_atom_work_.push_back(OnlineAtomWork{key, atom});
  if (!online_dirty_queued_[key]) {
    online_dirty_queued_[key] = 1;
    online_dirty_keys_.push_back(key);
  }
  if (closure) ++online_am_.closure_atoms;
  return true;
}

bool Otf::online_raise_bump(int clock, std::int32_t bound, bool closure) {
  if (clock <= 0 || clock >= (int)online_am_.bump.size()) return false;
  if (bound <= online_am_.bump[clock]) return false;
  online_enable_deferred_index();
  if (online_failed_) return false;
  online_am_.bump[clock] = bound;
  online_am_.live[clock] = 1;
  ++online_bound_epoch_;
  online_global_dirty_ = true;
  if (closure) ++online_am_.closure_atoms;
  return true;
}

/// @brief Seed the completion map at a newly discovered discrete key.
///
/// This deliberately uses the universe, not the zone that discovered the key: D is a property
/// of the completed automaton at the discrete key, and a later reached zone must not make the
/// local deadlock condition disappear.
void Otf::online_seed_key(int key_id) {
  if (online_failed_) return;
  const int nc = net.nclocks;
  OnlineKeyInfo& ki = *online_key_by_id_.at(key_id);
  SymState u;
  u.locs = ki.locs;
  u.ints = ki.ints;
  bool fc, fu;
  frozen_flags(*dead_net, u.locs, u.ints, &fc, &fu);
  std::vector<Constraint> ci;
  all_invariants(*dead_net, u.locs, ci);

  // In a non-frozen state, time is unbounded whenever the invariant has no absolute upper
  // constraint x <= c.  Difference and lower constraints alone cannot anchor an upper bound to
  // clock 0.  Detect that syntactically before allocating and closing two dense DBMs; this is the
  // overwhelmingly common seed case in scalable networks.
  if (!fc && !fu) {
    bool has_upper = false;
    for (const Constraint& c : ci)
      if (c.i > 0 && c.j == 0 && c.bound < kInf) { has_upper = true; break; }
    if (!has_upper) return;
  }

  u.zone = Dbm::universe(nc);
  Dbm inv = Dbm::universe(nc);
  if (!fc && !fu) inv.delay();
  // Keep this seed computation local.  Most keys never participate in an atom pullback, and a
  // dense invariant DBM per discrete key is far larger than the key skeleton itself.  If a
  // later pullback actually reaches this key, online_pullback reconstructs and caches the same
  // invariant lazily.
  if (!apply_all(inv, ci)) return;

  // A non-frozen time-open key has an infinite delay from every valuation of its invariant.
  // Its deadlock completion is therefore empty, so constructing all enabled/escape federations
  // cannot reveal a seed atom.  This test used to sit after state_has_deadlock(), which made
  // every such key pay the full symbolic subtraction before reaching the same conclusion.
  bool open = true;
  for (int c = 1; c <= nc; ++c)
    if (inv.get(c, 0) != kInf) { open = false; break; }
  if (open && !fc && !fu) return;                 // divergence already covers every stop

  Federation stuck, esc;
  if (!state_has_deadlock(*dead_net, u, &stuck, &esc)) return;
  ki.may_deadlock = true;

  Federation kept;
  kept.reserve(esc.size());
  for (const Dbm& w : esc) {
    Dbm t = w;
    t.intersect(inv);
    if (t.empty()) { ++online_am_.dropped_vacuous; continue; }
    kept.push_back(w);
  }
  if (kept.empty()) return;                       // Dead(q) == Inv(q): no distinguishing atom
  esc.swap(kept);
  ++online_am_.seed_keys;

  const int dim = nc + 1;
  std::vector<std::uint32_t> mbits(mindbm_bits2words(std::uint32_t(dim) * std::uint32_t(dim)));
  std::vector<int> mfirst(dim), mnext(dim);
  auto extract = [&](const Federation& f, std::vector<AtomC>& diag,
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
          if (raw_add(inv.get(j, i), r) < kZeroLe) {
            ++online_am_.dropped_disjoint;
            continue;
          }
          if (i > 0 && j > 0) {
            const AtomC a{i, j, r};
            if (std::find(diag.begin(), diag.end(), a) == diag.end()) diag.push_back(a);
          } else {
            const int c = i == 0 ? j : i;
            const std::int32_t b = i == 0 ? -raw_bound(r) : raw_bound(r);
            bool seen = false;
            for (auto& x : nond)
              if (x.first == c) { x.second = std::max(x.second, b); seen = true; break; }
            if (!seen) nond.push_back({c, b});
          }
        }
    }
  };

  std::vector<AtomC> residue_diag, escape_diag;
  std::vector<std::pair<int, std::int32_t>> residue_nond, escape_nond;
  extract(stuck, residue_diag, residue_nond);
  extract(esc, escape_diag, escape_nond);
  const bool take_escape = escape_diag.size() + 4 * escape_nond.size()
                         <= residue_diag.size() + 4 * residue_nond.size();
  const std::vector<AtomC>& diag = take_escape ? escape_diag : residue_diag;
  const auto& nond = take_escape ? escape_nond : residue_nond;
  online_am_.seed_atoms += diag.size() + nond.size();
  for (const AtomC& a : diag) online_add_diag(ki.id, a, /*closure=*/false);
  for (const auto& x : nond) {
    ++online_am_.nondiag_atoms;
    online_raise_bump(x.first, x.second, /*closure=*/false);
  }
}

/// @brief Pull one target diagonal atom backwards through one reset-labelled key edge.
void Otf::online_pullback(const OnlineKeyEdge& e, const AtomC& atom) {
  const std::vector<int>& resets = online_reset_sets_[e.reset_class];
  const auto reset = [&](int c) {
    return c > 0 && std::binary_search(resets.begin(), resets.end(), c);
  };
  const int i = reset(atom.i) ? 0 : atom.i;
  const int j = reset(atom.j) ? 0 : atom.j;
  if (i == j) return;
  if (i == 0 || j == 0) {
    const int c = i == 0 ? j : i;
    const std::int32_t b = i == 0 ? -raw_bound(atom.b) : raw_bound(atom.b);
    online_raise_bump(c, b, /*closure=*/true);
    return;
  }

  OnlineKeyInfo& si = *online_key_by_id_.at(e.src);
  if (!si.inv_ready) {
    SymState rep;
    rep.locs = si.locs;
    rep.ints = si.ints;
    si.inv_ready = true;
    Dbm inv;
    if (key_invariant(*dead_net, rep, net.nclocks, inv))
      si.inv = std::make_unique<Dbm>(std::move(inv));
  }
  if (si.inv) {
    if (si.inv->get(i, j) <= atom.b) return;
    if (raw_add(si.inv->get(j, i), atom.b) < kZeroLe) {
      ++online_am_.dropped_disjoint;
      return;
    }
  }
  online_add_diag(si.id, AtomC{i, j, atom.b}, /*closure=*/true);
}

void Otf::online_drain_atoms() {
  while (!online_atom_work_.empty() && !online_failed_) {
    OnlineAtomWork w = std::move(online_atom_work_.front());
    online_atom_work_.pop_front();
    if (w.key < 0 || w.key >= static_cast<int>(online_key_by_id_.size())) continue;
    const OnlineKeyInfo& ki = *online_key_by_id_[w.key];
    // A pullback may append an atom to the source of a self-loop.  Copy each intrusive edge
    // record before the call so neither atom-map growth nor a future edge-arena reallocation can
    // invalidate the iteration state.
    for (int ei = ki.incoming; ei >= 0;) {
      const OnlineKeyEdge edge = online_key_edges_[ei];
      ei = edge.next_in;
      online_pullback(edge, w.atom);
    }
  }
}

int Otf::online_register_key(const LocVec& locs, const IntVec& ints,
                             std::uint64_t packed, bool packed_ready) {
  // Most generated successors revisit an existing discrete key.  The packed path is lossless;
  // wider keys use the exact vector bucket below.  D atoms are indexed by this numeric key, so
  // the online path never serialises a discrete state into a temporary string.
  std::uint64_t h = 0;
  std::vector<int>* slots = nullptr;
  if (online_packed_key_usable_) {
    if (!packed_ready) {
      packed = 0;
      for (int p = 0; p < static_cast<int>(locs.size()); ++p)
        packed |= static_cast<std::uint64_t>(static_cast<unsigned>(locs[p]))
                  << online_disc_shift_[p];
      for (std::size_t v = 0; v < ints.size(); ++v)
        packed |= static_cast<std::uint64_t>(ints[v] - net.intvars[v].lo)
                  << online_disc_shift_[net.nprocs() + static_cast<int>(v)];
    }
    std::size_t slot = static_cast<std::size_t>(mix_packed_key(packed)) & online_packed_key_mask_;
    while (online_packed_key_values_[slot] >= 0) {
      if (online_packed_key_keys_[slot] == packed) return online_packed_key_values_[slot];
      slot = (slot + 1) & online_packed_key_mask_;
    }
  } else {
    h = disc_hash(locs, ints);
    slots = &online_key_index_[h];
    for (int id : *slots) {
      const OnlineKeyInfo& old = *online_key_by_id_[id];
      if (old.locs == locs && old.ints == ints) return id;
    }
  }

  online_keys_.emplace_back();
  OnlineKeyInfo& ki = online_keys_.back();
  ki.id = static_cast<int>(online_key_by_id_.size());
  ki.packed_key = packed;
  ki.locs = locs;
  ki.ints = ints;
  online_key_by_id_.push_back(&ki);
  online_dirty_queued_.push_back(0);
  const std::size_t dim = static_cast<std::size_t>(net.nclocks + 1);
  if (online_local_bounds_narrow_) {
    constexpr std::int16_t narrow_inf = std::numeric_limits<std::int16_t>::min();
    const auto pack = [](std::int32_t value) {
      return value == kAluNegInf ? std::numeric_limits<std::int16_t>::min()
                                 : static_cast<std::int16_t>(value);
    };
    const std::size_t off = online_local_l16_.size();
    online_local_l16_.resize(off + dim, narrow_inf);
    online_local_u16_.resize(off + dim, narrow_inf);
    if (net.lbound_loc.empty()) {
      for (std::size_t c = 0; c < dim; ++c) {
        online_local_l16_[off + c] = pack(net.lbound[c]);
        online_local_u16_[off + c] = pack(net.ubound[c]);
      }
    } else {
      for (int pr = 0; pr < net.nprocs(); ++pr)
        for (int c = 1; c <= net.nclocks; ++c) {
          online_local_l16_[off + static_cast<std::size_t>(c)] = std::max(
              online_local_l16_[off + static_cast<std::size_t>(c)],
              pack(net.lbound_loc[pr][locs[pr]][c]));
          online_local_u16_[off + static_cast<std::size_t>(c)] = std::max(
              online_local_u16_[off + static_cast<std::size_t>(c)],
              pack(net.ubound_loc[pr][locs[pr]][c]));
        }
      online_local_l16_[off] = online_local_u16_[off] = 0;
    }
  } else {
    const std::size_t off = online_local_l_.size();
    online_local_l_.resize(off + dim, kAluNegInf);
    online_local_u_.resize(off + dim, kAluNegInf);
    if (net.lbound_loc.empty()) {
      std::copy(net.lbound.begin(), net.lbound.end(), online_local_l_.begin() + off);
      std::copy(net.ubound.begin(), net.ubound.end(), online_local_u_.begin() + off);
    } else {
      for (int pr = 0; pr < net.nprocs(); ++pr)
        for (int c = 1; c <= net.nclocks; ++c) {
          online_local_l_[off + static_cast<std::size_t>(c)] = std::max(
              online_local_l_[off + static_cast<std::size_t>(c)],
              net.lbound_loc[pr][locs[pr]][c]);
          online_local_u_[off + static_cast<std::size_t>(c)] = std::max(
              online_local_u_[off + static_cast<std::size_t>(c)],
              net.ubound_loc[pr][locs[pr]][c]);
        }
      online_local_l_[off] = online_local_u_[off] = 0;
    }
  }
  if (online_packed_key_usable_) {
    if ((online_packed_key_count_ + 1) * 10 >= online_packed_key_values_.size() * 7) {
      std::vector<std::uint64_t> old_keys = std::move(online_packed_key_keys_);
      std::vector<int> old_values = std::move(online_packed_key_values_);
      const std::size_t capacity = old_values.size() * 2;
      online_packed_key_keys_.resize(capacity);
      online_packed_key_values_.assign(capacity, -1);
      online_packed_key_mask_ = capacity - 1;
      for (std::size_t old = 0; old < old_values.size(); ++old) {
        if (old_values[old] < 0) continue;
        std::size_t slot = static_cast<std::size_t>(mix_packed_key(old_keys[old])) &
                           online_packed_key_mask_;
        while (online_packed_key_values_[slot] >= 0)
          slot = (slot + 1) & online_packed_key_mask_;
        online_packed_key_keys_[slot] = old_keys[old];
        online_packed_key_values_[slot] = old_values[old];
      }
    }
    std::size_t slot = static_cast<std::size_t>(mix_packed_key(packed)) & online_packed_key_mask_;
    while (online_packed_key_values_[slot] >= 0)
      slot = (slot + 1) & online_packed_key_mask_;
    online_packed_key_keys_[slot] = packed;
    online_packed_key_values_[slot] = ki.id;
    ++online_packed_key_count_;
  } else {
    slots->push_back(ki.id);
  }
  online_seed_key(ki.id);
  return ki.id;
}

int Otf::online_reset_class(int tag) {
  if (tag < 0 || tag >= static_cast<int>(tags_.size())) {
    online_failed_ = true;
    return -1;
  }
  if (online_tag_reset_class_.size() <= static_cast<std::size_t>(tag))
    online_tag_reset_class_.resize(static_cast<std::size_t>(tag) + 1, -1);
  int& cached = online_tag_reset_class_[tag];
  if (cached >= 0) return cached;

  std::vector<int> resets;
  for (const auto& pe : tags_[tag].parts)
    for (int c : net.procs[pe.first].edges[pe.second].clock_resets)
      if (c >= 1 && c <= net.nclocks) resets.push_back(c);
  std::sort(resets.begin(), resets.end());
  resets.erase(std::unique(resets.begin(), resets.end()), resets.end());

  for (std::size_t i = 0; i < online_reset_sets_.size(); ++i)
    if (online_reset_sets_[i] == resets) return cached = static_cast<int>(i);
  online_reset_sets_.push_back(std::move(resets));
  return cached = static_cast<int>(online_reset_sets_.size() - 1);
}

/// @brief Build the reset-labelled key skeleton only when the first diagonal D atom exists.
///
/// Before that point there is nothing to pull backward.  Raw graph edges already retain their
/// transition tags and target representatives, so the exact key skeleton for the explored
/// prefix can be indexed once, locally, without regenerating a successor or rebuilding a graph.
void Otf::online_enable_key_graph() {
  if (online_key_graph_enabled_) return;
  online_key_graph_enabled_ = true;
  for (std::size_t raw_slot = 0; raw_slot < online_raw_adj_.size(); ++raw_slot) {
    if (raw_slot >= online_raw_owner_.size()) { online_failed_ = true; return; }
    const int owner = online_raw_owner_[raw_slot];
    if (owner < 0 || owner >= static_cast<int>(online_key_of_.size())) {
      online_failed_ = true;
      return;
    }
    const int src = online_key_of_[owner];
    for (const OtfEdge& edge : online_raw_adj_[raw_slot]) {
      if (edge.tgt < 0 || edge.tgt >= static_cast<int>(online_key_of_.size())) {
        online_failed_ = true;
        return;
      }
      online_register_edge(src, online_key_of_[edge.tgt], online_reset_class(edge.tag));
      if (online_failed_) return;
    }
  }
}

void Otf::online_register_edge(int src, int tgt, int reset_class) {
  if (reset_class < 0 || reset_class >= static_cast<int>(online_reset_sets_.size())) {
    online_failed_ = true;
    return;
  }
  OnlineKeyInfo& source = *online_key_by_id_.at(src);
  for (int ei = source.outgoing; ei >= 0; ei = online_key_edges_[ei].next_out) {
    const OnlineKeyEdge& old = online_key_edges_[ei];
    if (old.tgt == tgt && old.reset_class == reset_class) return;
  }
  const int id = static_cast<int>(online_key_edges_.size());
  OnlineKeyInfo& target = *online_key_by_id_.at(tgt);
  online_key_edges_.push_back(OnlineKeyEdge{src, tgt, reset_class, source.outgoing,
                                             target.incoming});
  source.outgoing = id;
  target.incoming = id;

  // An edge discovered after its target's atoms must observe all of them immediately; future
  // atoms arrive through online_atom_work_ and use the same incoming list.
  if (target.diag) {
    const std::vector<AtomC> atoms = *target.diag;
    for (const AtomC& a : atoms) online_pullback(online_key_edges_[id], a);
  }
}

void Otf::online_recheck_key(int key) {
  if (key < 0 || key >= static_cast<int>(online_key_by_id_.size())) return;
  OnlineKeyInfo& ki = *online_key_by_id_[key];
  const std::size_t count = ki.node_count;
  int v = ki.node_head;  // activation itself does not append ahead of this captured list
  if (online_perf.enabled) {
    ++online_perf.recheck_calls;
    online_perf.recheck_nodes += count + ki.deferred.size();
  }
  for (std::size_t n = 0; n < count && v >= 0; ++n) {
    const int current = v;
    v = online_next_in_key_[current];
    if (online_active_[current]) continue;
    const int c = online_cover_[current];
    if (c >= 0 && c < (int)node.size() && online_active_[c]) {
      if (online_covers(current, c, ki.id))
        continue;  // its recorded certificate survived the newly settled D map
    }
    const int cover = online_coverer(current, /*require_expanded=*/false);
    if (cover >= 0) {
      online_set_cover(current, cover);
    } else {
      online_activate(current);
    }
  }

  // Provisional successors that were never active have no node and no stored DBM.  Recreate
  // only those belonging to this D-dirty key, retest the exact candidate, and materialise it
  // only if the refined map leaves it uncovered.  The raw edge remains in the same graph and is
  // simply retargeted; this is local lazy reconstruction, not another reachability traversal.
  const std::size_t deferred_count = ki.deferred.size();
  for (std::size_t n = 0; n < deferred_count; ++n) {
    const std::uint64_t handle = ki.deferred[n];
    const int raw_slot = static_cast<int>(static_cast<std::uint32_t>(handle >> 32));
    const int ordinal = static_cast<int>(static_cast<std::uint32_t>(handle));
    if (raw_slot < 0 || raw_slot >= static_cast<int>(online_raw_adj_.size()) ||
        raw_slot >= static_cast<int>(online_raw_owner_.size()) || ordinal < 0 ||
        ordinal >= static_cast<int>(online_raw_adj_[raw_slot].size())) {
      online_failed_ = true;
      return;
    }
    OtfEdge& edge = online_raw_adj_[raw_slot][ordinal];
    if (edge.deferred < 0) continue;
    const int source = online_raw_owner_[raw_slot];

    SymState candidate;
    if (!online_replay(source, ordinal, ki.id, candidate)) {
      online_failed_ = true;
      return;
    }
    if (online_perf.enabled) ++online_perf.deferred_replayed;

    int cover = -1;
    if (edge.tgt >= 0 && edge.tgt < static_cast<int>(node.size()) && online_active_[edge.tgt] &&
        online_covers(candidate, edge.tgt, ki.id)) {
      cover = edge.tgt;
    } else {
      for (int j = ki.active_head; j >= 0; j = online_active_next_[j])
        if (online_active_[j] && online_covers(candidate, j, ki.id)) {
          cover = j;
          break;
        }
    }

    if (cover >= 0) {
      edge.tgt = cover;
      // These two relations are future-proof.  Once either holds, even another D refinement
      // cannot wake the candidate, so its replay recipe can be discarded permanently.
      const bool permanent = stored_includes(cover, candidate.zone);
      if (permanent) {
        edge.deferred = -1;
      }
      continue;
    }

    int nested_deferred = -1;
    const int v = online_intern(std::move(candidate), ki.id, -1, -1, &nested_deferred);
    if (nested_deferred >= 0) {
      online_failed_ = true;
      return;
    }
    edge.tgt = v;
    edge.deferred = -1;
    if (online_perf.enabled) ++online_perf.deferred_promoted;
  }
}

void Otf::online_recheck_dirty() {
  if (online_global_dirty_) {
    online_global_dirty_ = false;
    online_dirty_keys_.clear();
    std::fill(online_dirty_queued_.begin(), online_dirty_queued_.end(), 0);
    const std::size_t count = online_key_by_id_.size();
    for (std::size_t key = 0; key < count; ++key)
      online_recheck_key(static_cast<int>(key));
    return;
  }
  std::vector<int> keys = std::move(online_dirty_keys_);
  online_dirty_keys_.clear();
  for (int key : keys) {
    if (key >= 0 && key < static_cast<int>(online_dirty_queued_.size()))
      online_dirty_queued_[key] = 0;
    online_recheck_key(key);
  }
}

/// @brief Build the root through the common exact root pipeline, then retain it online.
bool Otf::online_build_all() {
  return build_all();
}

/// @brief Generate one raw state's semantic successors exactly once.
///
/// The two phases are intentional.  First register every target key and, once diagonal D exists,
/// its reset-labelled key edge.  A new atom may then invalidate an earlier certificate.  Only
/// after the atom worklist reaches its current fixed point are the successors covered or made
/// active.
void Otf::online_expand(int u) {
  grow();
  if (cancelled() || online_failed_ || u < 0 || u >= (int)node.size()) return;
  if (!online_active_[u] || expanded[u]) return;

  expanded[u] = 1;
  ++expand_calls;
  ++outer_expands;

  std::vector<SymState> succ;
  std::vector<GfpTag> stags;
  const int src = online_key_of_[u];
  const OnlineKeyInfo& src_info = *online_key_by_id_[src];
  const bool profile = online_perf.enabled;
  SymState usc;
  usc.locs = src_info.locs;
  usc.ints = src_info.ints;
  zread(u, usc.zone);
  std::chrono::steady_clock::time_point successor_start;
  if (profile) successor_start = std::chrono::steady_clock::now();
  gfp_succ_tagged(net, usc, L, U, qmask, emode, succ, stags,
                  avoid_forward_ ? avoid_ : nullptr);
  std::chrono::steady_clock::time_point successor_done;
  if (profile) successor_done = std::chrono::steady_clock::now();

  // A concrete terminal/divergent witness is conclusive immediately.  Unlike the old repeated
  // GFP probe this performs no backward pass on a changing graph; if it misses, construction
  // continues to D-quiescence and the one normal backward pass decides the query.
  if (eg_early_ && online_eg_terminal_witness(u, succ.empty())) {
    eg_early_hit_ = true;
    return;
  }

  std::vector<int> target_keys(succ.size());
  std::vector<int> tag_ids(succ.size());
  const auto packed_successor = [&](std::size_t k) {
    std::uint64_t packed = src_info.packed_key;
    for (const auto& part : stags[k].parts) {
      const int p = part.first;
      const std::uint64_t field_mask = online_disc_mask_[p] << online_disc_shift_[p];
      packed = (packed & ~field_mask) |
               ((static_cast<std::uint64_t>(static_cast<unsigned>(succ[k].locs[p])) &
                 online_disc_mask_[p]) << online_disc_shift_[p]);
    }
    // Generated and ordinary updates may change integer variables independently of the
    // participating process list.  Integer vectors are normally empty; when present, compare
    // them exactly and patch only changed fields.
    for (std::size_t v = 0; v < succ[k].ints.size(); ++v) {
      if (succ[k].ints[v] == src_info.ints[v]) continue;
      const int field = net.nprocs() + static_cast<int>(v);
      const std::uint64_t field_mask = online_disc_mask_[field] << online_disc_shift_[field];
      const std::uint64_t value = static_cast<std::uint64_t>(succ[k].ints[v] - net.intvars[v].lo);
      packed = (packed & ~field_mask) |
               ((value & online_disc_mask_[field]) << online_disc_shift_[field]);
    }
    return packed;
  };
  for (std::size_t k = 0; k < succ.size(); ++k) {
    if (profile) {
      auto t = std::chrono::steady_clock::now();
      const std::uint64_t packed = online_packed_key_usable_ ? packed_successor(k) : 0;
      target_keys[k] = online_register_key(succ[k].locs, succ[k].ints, packed,
                                            online_packed_key_usable_);
      online_perf.register_key_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t).count();
      t = std::chrono::steady_clock::now();
      tag_ids[k] = intern_tag(stags[k]);
      online_perf.register_tag_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t).count();
    } else {
      const std::uint64_t packed = online_packed_key_usable_ ? packed_successor(k) : 0;
      target_keys[k] = online_register_key(succ[k].locs, succ[k].ints, packed,
                                            online_packed_key_usable_);
      tag_ids[k] = intern_tag(stags[k]);
    }
    if (online_failed_) return;
  }
  if (online_key_graph_enabled_) {
    for (std::size_t k = 0; k < succ.size(); ++k) {
      if (profile) {
        auto t = std::chrono::steady_clock::now();
        const int reset_class = online_reset_class(tag_ids[k]);
        online_perf.register_reset_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t).count();
        t = std::chrono::steady_clock::now();
        online_register_edge(src, target_keys[k], reset_class);
        online_perf.register_edge_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t).count();
      } else {
        online_register_edge(src, target_keys[k], online_reset_class(tag_ids[k]));
      }
      if (online_failed_) return;
    }
  }
  std::chrono::steady_clock::time_point registration_done;
  if (profile) registration_done = std::chrono::steady_clock::now();
  online_drain_atoms();
  online_recheck_dirty();
  if (cancelled() || online_failed_) return;

  // Only expanded representatives own an outgoing raw-edge vector.  Keep its slot separate
  // from the persistent record id.  Provisionally covered successors are not materialised, so
  // allocating an empty vector for every possible target would dominate the metadata footprint.
  if (online_raw_slot_[u] < 0) {
    online_raw_slot_[u] = static_cast<int>(online_raw_adj_.size());
    online_raw_adj_.emplace_back();
    online_raw_owner_.push_back(u);
  }
  const int raw_slot = online_raw_slot_[u];
  online_raw_adj_[raw_slot].reserve(online_raw_adj_[raw_slot].size() + succ.size());
  for (std::size_t k = 0; k < succ.size(); ++k) {
    int deferred = -1;
    const int v = online_intern(std::move(succ[k]), target_keys[k], u,
                                static_cast<int>(k), &deferred);
    online_raw_adj_[raw_slot].push_back(OtfEdge{v, tag_ids[k], deferred});
    if (cancelled() || online_failed_) return;
  }
  if (profile) {
    const auto intern_done = std::chrono::steady_clock::now();
    online_perf.successor_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        successor_done - successor_start).count();
    online_perf.register_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        registration_done - successor_done).count();
    online_perf.intern_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        intern_done - registration_done).count();
  }

  // Keep the active quotient lean.  A materialised representative that becomes covered turns
  // inactive; its already generated raw edges remain available if a later atom reactivates it.
  online_compact_key(u);
}

/// @brief Drain the coupled expansion / atom / cover-certificate worklists to quiescence.
void Otf::online_drain_forward_work() {
  for (;;) {
    online_drain_atoms();
    online_recheck_dirty();
    if (cancelled() || online_failed_ || eg_early_hit_ || online_work_.empty()) break;

    const int u = online_work_.front();
    online_work_.pop_front();
    if (u < 0 || u >= (int)node.size()) continue;
    online_inq_[u] = 0;
    if (!online_active_[u] || expanded[u]) continue;
    if (online_eg_prune_ && !online_is_nq(u)) {
      // E[] never returns to psi after leaving it.  Keep the target vertex for incoming edge
      // redirection and D seeding, but its successors cannot influence the initial-state GFP.
      expanded[u] = 1;
      if (online_raw_slot_[u] < 0) {
        online_raw_slot_[u] = static_cast<int>(online_raw_adj_.size());
        online_raw_adj_.emplace_back();
        online_raw_owner_.push_back(u);
      }
      continue;
    }
    online_expand(u);
  }
}

/// @brief Finish exact adjacencies only when the terminal-witness fast path did not decide E[].
///
/// During the first forward phase an unexpanded representative may leave the active antichain;
/// its exact zone is parked, not deleted.  This is what lets a concrete terminal witness be
/// reached with the small D quotient.  If no such witness exists, the backward solver needs the
/// exact incoming image of every materialised record, so those parked records are activated and
/// expanded here before the single GFP starts.  No state is rediscovered from the root and no
/// graph is rebuilt: this drains retained records through the same online-D worklist.
void Otf::online_complete_eg_history() {
  online_eg_repair_ = true;
  const std::size_t parked = node.size();
  std::size_t pending = 0;
  for (std::size_t v = 0; v < parked; ++v)
    if (!expanded[v]) { ++pending; online_activate(static_cast<int>(v)); }
  if (online_perf.enabled)
    std::fprintf(stderr, "ONLINE-EG-REPAIR before=%zu pending=%zu queued=%zu\n",
                 parked, pending, online_work_.size());
  online_drain_forward_work();
  if (online_perf.enabled) {
    std::size_t remaining = 0;
    for (std::size_t v = 0; v < node.size(); ++v) remaining += !expanded[v];
    std::fprintf(stderr, "ONLINE-EG-REPAIR after=%zu remaining=%zu\n",
                 node.size(), remaining);
  }
}

/// @brief Start the persistent D forward construction and settle all online refinement work.
void Otf::online_base_forward() {
  fwd_ok_ = online_build_all();
  if (!fwd_ok_) return;
  online_drain_forward_work();
}

/// @brief Seal the final D quotient by redirecting raw edges through final active coverers.
///
/// No successor is regenerated here.  The defensive wake-up loop only handles an edge whose
/// target lost its last certificate after the main worklist checked it; it activates that retained
/// raw record and lets the ordinary online worklist expand it for the first (and only) time.
void Otf::online_finalize_graph() {
  if (!fwd_ok_ || cancelled() || online_failed_) return;

  for (;;) {
    bool woke = false;
    for (std::size_t u = 0; u < node.size(); ++u) {
      if (!online_active_[u] || !expanded[u]) continue;
      const int raw_slot = online_raw_slot_[u];
      if (raw_slot < 0) { online_failed_ = true; return; }
      for (const OtfEdge& e : online_raw_adj_[raw_slot]) {
        if (online_resolve_cover(e.tgt, /*require_expanded=*/true) >= 0) continue;
        // A live replay recipe must still resolve through the certificate established by the
        // last key recheck.  Activating its old cover would not reconstruct the candidate and
        // would therefore be the wrong defensive recovery.
        if (e.deferred >= 0) { online_failed_ = true; return; }
        // This is normally an inactive materialised record whose last D cover was invalidated.
        // It is activated, never rediscovered from the root.  An active but
        // unexpanded target is also explicitly queued here, guarding against stale queue bits
        // left by representative compaction.
        if (!online_active_[e.tgt] || !expanded[e.tgt]) {
          online_activate(e.tgt);
          woke = true;
        } else {
          online_failed_ = true;
          return;
        }
      }
    }
    if (!woke) break;
    online_drain_forward_work();
    if (cancelled() || online_failed_) return;
  }

  online_am_.keys_tagged = 0;
  for (const OnlineKeyInfo* ki : online_key_by_id_)
    if (ki->diag && !ki->diag->empty()) ++online_am_.keys_tagged;
  online_am_.bumped = 0;
  for (int c = 1; c <= net.nclocks; ++c)
    if (online_am_.bump[c] != kNoBump) ++online_am_.bumped;
  online_am_.usable = !online_failed_;

  // Every materialised record remains a backward vertex, including an old representative that
  // later left the forward antichain.  Its incoming edge was generated for this exact zone; a
  // forward-sound redirection to a simulator is not an exact inverse for gfp_pre.  Compaction
  // therefore removes records only from future cover scans, never from the finished graph.
  // With that invariant the existing numbering, zones, and raw adjacency can all be moved.
  bool already_dense = online_active_.size() == node.size();
  for (std::size_t v = 0; v < node.size() && already_dense; ++v)
    already_dense = expanded[v] && online_raw_slot_[v] >= 0;

  if (already_dense) {
    adj.clear();
    adj.resize(node.size());
    for (std::size_t v = 0; v < node.size(); ++v) {
      const OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[v]];
      node[v].locs = ki.locs;
      node[v].ints = ki.ints;
      std::vector<OtfEdge>& raw = online_raw_adj_[online_raw_slot_[v]];
      for (OtfEdge& edge : raw) {
        // Materialised targets keep their own identity, even after leaving the active forward
        // antichain. A never-materialised provisional successor already names its final active
        // coverer here; dirty-key replay would have promoted it if that certificate had failed.
        if (edge.tgt < 0 || edge.tgt >= static_cast<int>(node.size()) ||
            !expanded[edge.tgt] || online_raw_slot_[edge.tgt] < 0) {
          const int representative = online_resolve_cover(edge.tgt, /*require_expanded=*/true);
          if (representative < 0) { online_failed_ = true; return; }
          edge.tgt = representative;
        }
        edge.deferred = -1;
      }
      adj[v] = std::move(raw);
    }
  } else {
    // Reindex only when refinement really left inactive materialised records.  No successor is
    // recomputed and no D decision is revisited.
    std::vector<int> remap(node.size(), -1);
    std::vector<SymState> compact_nodes;
    compact_nodes.reserve(node.size());
    ptctlmc::ZoneStore compact_zones;
    compact_zones.reset(zdim_);
    for (std::size_t old = 0; old < node.size(); ++old) {
      if (!expanded[old] || online_raw_slot_[old] < 0) continue;
      const int now = static_cast<int>(compact_nodes.size());
      remap[old] = now;
      const OnlineKeyInfo& ki = *online_key_by_id_[online_key_of_[old]];
      SymState s;
      s.locs = ki.locs;
      s.ints = ki.ints;
      Dbm z;
      zread(static_cast<int>(old), z);
      compact_zones.store(now, z);
      compact_nodes.push_back(std::move(s));
    }
    std::vector<std::vector<OtfEdge>> compact_adj(compact_nodes.size());
    for (std::size_t old = 0; old < node.size(); ++old) {
      const int now = remap[old];
      if (now < 0) continue;
      const int raw_slot = online_raw_slot_[old];
      std::vector<OtfEdge>& out = compact_adj[now];
      out.reserve(online_raw_adj_[raw_slot].size());
      for (const OtfEdge& e : online_raw_adj_[raw_slot]) {
        int representative = e.tgt;
        if (representative < 0 || representative >= static_cast<int>(node.size()) ||
            !expanded[representative] || online_raw_slot_[representative] < 0)
          representative = online_resolve_cover(e.tgt, /*require_expanded=*/true);
        if (representative < 0) { online_failed_ = true; return; }
        const int tgt = remap[representative];
        if (tgt < 0) { online_failed_ = true; return; }
        out.push_back(OtfEdge{tgt, e.tag});
      }
    }
    node.swap(compact_nodes);
    adj.swap(compact_adj);
    zs_ = std::move(compact_zones);
  }
  bucket.clear();
  for (std::size_t v = 0; v < node.size(); ++v)
    bucket[disc_hash(node[v].locs, node[v].ints)].push_back(static_cast<int>(v));
  expanded.assign(node.size(), 1);
  pc.assign(node.size(), 2);
  qc.assign(node.size(), 2);
  ntag_.assign(node.size(), 0);
  ntag_out_.assign(node.size(), 0);
  online_active_.clear();
  online_active_next_.clear();
  online_active_prev_.clear();
  online_cover_.clear();
  online_raw_adj_.clear();
  online_raw_owner_.clear();
  online_raw_slot_.clear();
  online_inq_.clear();
  online_work_.clear();
  online_key_by_id_.clear();
  online_local_l16_.clear();
  online_local_u16_.clear();
  online_local_l_.clear();
  online_local_u_.clear();
  online_key_of_.clear();
  online_next_in_key_.clear();
  online_keys_.clear();
  online_key_index_.clear();
  online_packed_key_keys_.clear();
  online_packed_key_values_.clear();
  online_packed_key_mask_ = 0;
  online_packed_key_count_ = 0;
  online_disc_shift_.clear();
  online_disc_mask_.clear();
  online_key_edges_.clear();
  online_reset_sets_.clear();
  online_tag_reset_class_.clear();
  online_atom_work_.clear();
  online_dirty_keys_.clear();
  online_dirty_queued_.clear();
  online_bound_key_ = -1;
  // Subsequent code is the ordinary backward solver over the compact final graph.
  online_d_ = false;
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
      // EARLY WITNESS PROBE. `E[] psi` is satisfied on 220 of the 254 decided rows of the
      // benchmark suite, and on those the whole forward graph is built to certify a witness that
      // a fraction of it already contains -- fischer8 `E[] x0 <= 40 && !P1.cs` needs 292 of its
      // 129468 nodes. ::Otf::eg_probe is sound in the SAT direction on a partial graph, so the
      // search may stop the moment one fires; NOT is never concluded here.
      //
      // Hoisted out of the per-EDGE test below: ::Otf::is_nq opens with a ::Otf::grow it does not
      // need here, the loop running right after one, and at one call per edge that is not free.
      const bool psi_prune = eg_early_;
      //
      // WHEN TO PROBE: amortization, not tuning. A probe that misses is wasted work whose cost
      // grows with the graph, so it has to be paid for. The rule is that exploration since the
      // last probe must have cost at least as much as ALL the probing done so far; only then is
      // another probe affordable. Two things follow with no constant anywhere:
      //
      //   * total probe time can never exceed total exploration time, so a query whose witness
      //     never appears -- every `NOT` row -- pays at most a factor of two, by construction;
      //   * the spacing calibrates itself to the model. Where a probe is cheap against the
      //     exploration it interrupts the probes come thick and the witness is caught early;
      //     where the fixpoint is expensive (grc) they thin out on their own.
      //
      // The alternative, a geometric schedule on the node count, needs a first size and a ratio,
      // and both are fitted: ratio 4 puts consecutive checkpoints at 25% and 100% of the final
      // graph, which is why grc7 -- whose witness is only provable at 57% -- explored all of it.
      const auto clk_now = [] { return std::chrono::steady_clock::now(); };
      static const bool probe_dbg = [] { return std::getenv("PTCTLMC_EG_PROBEDBG") != nullptr; }();
      long long probe_ns = 0;                  // cumulative cost of every probe so far
      auto since_probe = clk_now();            // when exploration resumed after the last one
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
        //
        // AND, for `E[] psi`, only if it still satisfies psi. psi's CLOCK part is folded into the
        // restricted network's invariants, so the forward search never leaves it; its DISCRETE
        // part is not, and was left to ::Otf::is_nq at closure time -- so the search explored the
        // whole reachable graph and then discarded 73% of it on fischer7. A maximal psi-run is in
        // psi at every step, so a run reaching a non-psi node has already left psi and neither
        // that node nor anything beyond it can lie on a witness. Not expanding it therefore
        // removes nothing from ::Otf::eg_closure: a psi-node reachable only through a non-psi
        // node was never admitted to it in the first place. It removes 3x the STORED nodes and
        // almost no time, because those nodes were never in the closure the fixpoint iterates.
        for (const auto& e : adj[u])
          if (!expanded[e.tgt] && !(psi_prune && !is_nq_fast(e.tgt))) push(e.tgt);
        if (psi_prune &&
            std::chrono::duration_cast<std::chrono::nanoseconds>(clk_now() - since_probe).count()
                >= probe_ns) {
          const auto p0 = clk_now();
          const bool hit = eg_probe();
          probe_ns +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(clk_now() - p0).count();
          if (probe_dbg)
            std::fprintf(stderr, "PROBE at_nodes=%zu hit=%d cumulative_probe_ms=%lld\n",
                         node.size(), (int)hit, probe_ns / 1000000);
          if (hit) { eg_early_hit_ = true; return; }
          since_probe = clk_now();
        }
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
  if (eg_early_hit_) { fill_stats(gs); return true; }   // witness proven on the partial graph
  return eg_verdict(gs);
}

std::vector<int> Otf::eg_closure() {
  std::vector<int> S;
  if (node.empty()) return S;
  grow();
  // The psi-closure REACHED FROM THE INITIAL VERTEX. ::Otf::accepting_closure seeds from every
  // p-node because `p --> q` accepts a witness anywhere; here the run must start at the initial
  // state, so a node reachable only by leaving psi and coming back must not be admitted.
  if (!expanded[0] || !is_nq(0)) return S;
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
  return S;
}

bool Otf::eg_probe() {
  if (!fwd_ok_ || node.empty()) return false;
  const bool saved_capped = capped;         // sticky, and a probe must not condemn the real run
  const bool saved_scc = scc_iter_;
  scc_iter_ = true;
  bool sat = false;
  const std::vector<int> S = eg_closure();
  if (!S.empty()) {
    std::vector<Federation> W;
    gfp_on(S, W);
    if (!capped) sat = !gfp_fed_empty(W[0]);
  }
  capped = saved_capped;
  scc_iter_ = saved_scc;
  return sat;
}

bool Otf::eg_verdict(GfpStats* gs) {
  scc_iter_ = true;             // the verdict is one node's, so diameter is what costs here
  bool res = false;
  if (fwd_ok_ && !node.empty()) {
    const std::vector<int> S = eg_closure();
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

/// @brief The one-construction D path used by ordinary non-diagonal liveness queries.
///
/// The map begins empty, but this is not a preliminary LU graph.  Active exact records and raw
/// semantic edges are built once; a provisionally covered successor is represented only by its
/// source-edge replay handle.  Atom pullback and expansion share one worklist, and only the
/// stable active quotient reaches the backward GFP.
bool Otf::run_online_d(GfpStats* gs) {
  // Diagonal constraints already select the exact inclusion lane.  There is no a_LU/D quotient
  // to refine there, and preserving the existing exact route avoids imposing online bookkeeping
  // where it cannot buy anything.
  if (network_has_diagonal(net) || force_exact_) return run(gs);

  // D = a_LU is known before the first successor in a globally time-open network.  Entering the
  // revocable-cover machinery in that case cannot refine a single cover; it only duplicates the
  // ordinary a_LU graph's key, edge, and replay metadata.
  if (structurally_empty_d(*dead_net)) return run(gs);

  online_d_ = true;
  online_reset();
  online_perf = OnlinePerf{};
  fwd_ok_ = false;
  setup();
  const auto forward_start = std::chrono::steady_clock::now();
  online_base_forward();
  const auto forward_done = std::chrono::steady_clock::now();
  std::size_t raw_nodes = node.size(), raw_edges = 0, active_nodes = 0, active_slots = 0;
  std::size_t live_deferred = 0;
  for (const auto& edges : online_raw_adj_)
    live_deferred += std::count_if(edges.begin(), edges.end(),
                                  [](const OtfEdge& e) { return e.deferred >= 0; });
  for (std::size_t v = 0; v < online_active_.size(); ++v) active_nodes += online_active_[v] != 0;
  for (const OnlineKeyInfo& ki : online_keys_) {
    active_slots += ki.active_count;
    for (int v = ki.active_head; v >= 0; v = online_active_next_[v])
      if (online_raw_slot_[v] >= 0) raw_edges += online_raw_adj_[online_raw_slot_[v]].size();
  }
  const std::size_t key_count = online_key_by_id_.size();
  const std::size_t key_edge_count = online_key_edges_.size();
  const std::size_t seed_atoms = online_am_.seed_atoms;
  const std::size_t closure_atoms = online_am_.closure_atoms;
  const std::size_t seed_keys = online_am_.seed_keys;
  const std::size_t diag_keys = std::count_if(
      online_keys_.begin(), online_keys_.end(),
      [](const OnlineKeyInfo& ki) { return ki.diag && !ki.diag->empty(); });
  const std::size_t bumped = std::count_if(online_am_.bump.begin() + 1, online_am_.bump.end(),
                                           [](std::int32_t x) { return x != kNoBump; });
  if (fwd_ok_ && !cancelled() && !online_failed_) online_finalize_graph();
  const auto finalised = std::chrono::steady_clock::now();

  // `online_failed_` is defensive only (the online atom and reset representations are dynamic),
  // but never turn an unfinished quotient into a normal answer if an invariant is broken.
  if (online_failed_ || cancelled()) capped = true;
  const bool violated = fwd_ok_ && !online_failed_ && !cancelled() && final_verdict_violated();
  const auto solved = std::chrono::steady_clock::now();
  if (online_perf.enabled) {
    const auto ms = [](auto d) {
      return std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
    };
    std::fprintf(stderr,
                 "ONLINE-PERF forward_ms=%lld finalize_ms=%lld backward_ms=%lld raw_nodes=%zu "
                 "active_nodes=%zu active_slots=%zu active_edges=%zu keys=%zu key_edges=%zu seed_keys=%zu "
                 "seed_atoms=%zu closure_atoms=%zu diag_keys=%zu bumped=%zu cover_calls=%llu "
                 "cover_ms=%.3f intern=%llu intern_scan=%llu probe_ms=%.3f "
                 "join_ms=%.3f append_ms=%.3f "
                 "deferred=%zu deferred_created=%llu deferred_replayed=%llu deferred_promoted=%llu "
                 "coverer=%llu coverer_scan=%llu "
                 "compact=%llu compact_scan=%llu recheck=%llu recheck_nodes=%llu succ_ms=%.3f "
                 "register_ms=%.3f key_ms=%.3f tag_ms=%.3f reset_ms=%.3f edge_ms=%.3f intern_ms=%.3f\n",
                 static_cast<long long>(ms(forward_done - forward_start)),
                 static_cast<long long>(ms(finalised - forward_done)),
                 static_cast<long long>(ms(solved - finalised)), raw_nodes, active_nodes,
                 active_slots, raw_edges, key_count, key_edge_count, seed_keys, seed_atoms, closure_atoms,
                 diag_keys, bumped,
                 static_cast<unsigned long long>(online_perf.cover_calls),
                 online_perf.cover_ns / 1e6,
                 static_cast<unsigned long long>(online_perf.intern_calls),
                 static_cast<unsigned long long>(online_perf.intern_active_scan),
                 online_perf.intern_cover_probe_ns / 1e6, online_perf.intern_join_ns / 1e6,
                 online_perf.intern_append_ns / 1e6, live_deferred,
                 static_cast<unsigned long long>(online_perf.deferred_created),
                 static_cast<unsigned long long>(online_perf.deferred_replayed),
                 static_cast<unsigned long long>(online_perf.deferred_promoted),
                 static_cast<unsigned long long>(online_perf.coverer_calls),
                 static_cast<unsigned long long>(online_perf.coverer_active_scan),
                 static_cast<unsigned long long>(online_perf.compact_calls),
                 static_cast<unsigned long long>(online_perf.compact_active_scan),
                 static_cast<unsigned long long>(online_perf.recheck_calls),
                 static_cast<unsigned long long>(online_perf.recheck_nodes),
                 online_perf.successor_ns / 1e6, online_perf.register_ns / 1e6,
                 online_perf.register_key_ns / 1e6, online_perf.register_tag_ns / 1e6,
                 online_perf.register_reset_ns / 1e6, online_perf.register_edge_ns / 1e6,
                 online_perf.intern_ns / 1e6);
  }
  fill_stats(gs);
  return !violated;
}

bool Otf::eg_from_initial_online_d(GfpStats* gs) {
  if (network_has_diagonal(net) || force_exact_) return eg_from_initial(gs);

  online_d_ = true;
  online_reset();
  online_perf = OnlinePerf{};
  fwd_ok_ = false;
  // Discrete states outside psi are terminal for this query and need no successor expansion.
  // This is forward pruning only: the sole backward pass still starts after D stabilises.
  online_eg_prune_ = true;
  setup();
  online_base_forward();
  if (online_perf.enabled) {
    std::fprintf(stderr,
                 "ONLINE-EG raw_nodes=%zu keys=%zu seed_keys=%zu seed_atoms=%zu closure_atoms=%zu "
                 "deferred_created=%llu deferred_replayed=%llu deferred_promoted=%llu\n",
                 node.size(), online_key_by_id_.size(), online_am_.seed_keys,
                 online_am_.seed_atoms, online_am_.closure_atoms,
                 static_cast<unsigned long long>(online_perf.deferred_created),
                 static_cast<unsigned long long>(online_perf.deferred_replayed),
                 static_cast<unsigned long long>(online_perf.deferred_promoted));
  }
  if (eg_early_hit_) { fill_stats(gs); return true; }
  online_complete_eg_history();
  if (eg_early_hit_) { fill_stats(gs); return true; }
  if (fwd_ok_ && !cancelled() && !online_failed_) online_finalize_graph();
  if (online_failed_ || cancelled()) { capped = true; fill_stats(gs); return false; }
  return eg_verdict(gs);
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

/// @brief `E[] psi` from the initial state, decided by the backward greatest fixpoint.
///
/// @param net the network, already restricted by `psi`'s clock conjunction if it had one
/// @param nq  a predicate holding exactly where `psi`'s DISCRETE part fails, so `V_psi` is its
///            complement; carry an arbitrary boolean through ptctlmc::DiscPred::forbid
/// @param[out] gs statistics; may be null
/// @return @c true iff the initial state admits a maximal execution all of whose states satisfy
///         `psi`
///
/// The forward construction and D refinement are the same persistent construction used by
/// leads-to.  The difference is only the seed and readout of the final GFP: E[] starts from the
/// initial node and reads that node's value, whereas leads-to seeds every `p /\ !q` node.
///
/// @note @p witness_only remains in the public helper signature for compatibility.  The online
/// construction has one execution mode, so it does not select an alternative graph-building path.
bool exists_invariant_run_otf(const Network& net, const DiscPred& nq, GfpStats* gs,
                              const Network* unrestricted,
                              const detail::DelayAvoidSet* avoid, bool witness_only) {
  static const DiscPred kTrue{};
  auto make = [&](bool exact) {
    auto e = std::make_unique<Otf>(net, kTrue, nq, exact, nullptr, unrestricted);
    e->set_avoid(avoid);
    e->set_eg_early(true);                          // SAT may be concluded on a partial graph
    return e;
  };
  (void)witness_only;
  return make(false)->eg_from_initial_online_d(gs);
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
  // The clock/boolean front-end uses the same persistent quotient.  Its forbidden region still
  // confines only the backward phase (`forward=false` above), while D atoms are discovered from
  // the ordinary reachable key skeleton as the online construction grows.
  return make(false)->run_online_d(gs);
}

/// @brief Decide the discrete leads-to query `p --> q`.
///
/// D starts empty and is refined while reachable keys and raw semantic edges are discovered.
/// A provisionally covered successor is kept only as an edge replay handle.  If a later atom
/// invalidates its certificate, that one successor is regenerated and materialised; the graph is
/// never rebuilt.  Once the coupled worklist stabilises, existing edges are redirected to the
/// active D representatives and one backward GFP decides the verdict.
bool leads_to_otf(const Network& net, const DiscPred& p, const DiscPred& q, GfpStats* gs) {
  Otf e(net, p, q);
  return e.run_online_d(gs);
}

}  // namespace ptctlmc
