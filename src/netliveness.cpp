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

#include <algorithm>
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
                            std::vector<SymState>& out, std::vector<GfpTag>& tags) {
  std::vector<Constraint> inv;
  bool src_committed, src_urgent;
  frozen_flags(net, s.locs, &src_committed, &src_urgent);

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
      if (!close_succ_zone(net, ns.locs, z1, inv, qmask, L, U, emode)) continue;
      if (z1.empty()) continue;
      ns.zone = std::move(z1);
      out.push_back(std::move(ns));
      GfpTag t; t.parts.push_back({p, ei}); tags.push_back(std::move(t));
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
          if (!close_succ_zone(net, ns.locs, z1, inv, qmask, L, U, emode)) continue;
          if (z1.empty()) continue;
          ns.zone = std::move(z1);
          out.push_back(std::move(ns));
          GfpTag t; t.parts.push_back({i, ei}); t.parts.push_back({j, ej}); tags.push_back(std::move(t));
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
            if (close_succ_zone(net, ns.locs, z1, inv, qmask, L, U, emode) && !z1.empty()) {
              ns.zone = std::move(z1);
              out.push_back(std::move(ns));
              tags.push_back(std::move(tg));
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
  bool nc, nu; frozen_flags(net, tgt.locs, &nc, &nu);
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
  bool sc, su; frozen_flags(net, src.locs, &sc, &su);
  if (!sc && !su) { Y.down(); }
  if (!apply_all(Y, sinv)) return gfp_empty(net.nclocks);
  return Y;
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
  GfpTag tag;   ///< the participating edges, for ::gfp_pre
};

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
      bool force_exact = false, const std::atomic<bool>* cancel = nullptr)
      : net(n), p(pp), q(qq), force_exact_(force_exact), cancel_(cancel) {}

  /// @brief Build the graph and decide the query.
  /// @return @c true iff `p --> q` holds
  /// @warning The result is meaningless if ::was_cancelled() is true afterwards.
  bool run(GfpStats* gs);

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
  std::vector<std::vector<int>> pred;          ///< reverse edges, indexed like ::node
  std::unordered_map<std::string, std::vector<int>> bucket;  ///< discrete key -> node indices
  std::vector<char> expanded;                  ///< 1 once ::expand has computed the out-edges
  std::vector<char> pc, qc;                    ///< p/q predicate caches; 2 = not yet evaluated
  bool intern_grew = false;                    ///< last ::intern rewrote a stored zone
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
  bool is_nq(int i);
  void grow();
  int intern(SymState&& s);
  void expand(int u);
  bool time_diverges(const SymState& s) const;
  Federation stop_set(int u);                  ///< exact NoSucc; requires `expanded[u]`

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
  L = net.lbound;
  U = net.ubound;
  use_alu = !force_exact_ && !network_has_diagonal(net);
  emode = network_has_diagonal(net) ? ExNoDiag : (use_alu ? ExNone : ExLuPlus);
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
  if (pred.size() < n) pred.resize(n);
  if (ntag_.size() < n) ntag_.resize(n, 0);
  if (ntag_out_.size() < n) ntag_out_.resize(n, 0);
}

bool Otf::is_p(int i) {
  grow();
  if (pc[i] == 2) pc[i] = holds(p, node[i]) ? 1 : 0;
  return pc[i] != 0;
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
  auto& b = bucket[bkey];
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
  auto covered = [&](int j) {
    if (reccheck) {                                         // straddling operands: recurse
      return cover_rec(s.zone, tag, node[j].zone, ntag_out_[j], *atoms, 0);
    }
    if (tagcheck && (ntag_[j] & ~tag) != 0) return false;   // Z |= phi but Z' does not: refuse
    return use_alu ? s.zone.alu_subsumed_by(node[j].zone, bL, bU) : node[j].zone.includes(s.zone);
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
      return cover_rec(node[j].zone, ntag_[j], s.zone, tag_out, *atoms, 0);
    }
    if (tagcheck && ntag_[j] != tag) return false;          // rewriting: equality, see above
    return use_alu ? node[j].zone.alu_subsumed_by(s.zone, bL, bU) : s.zone.includes(node[j].zone);
  };
  for (int j : b)
    if (covered(j)) return j;
  for (std::size_t k = 0; k < b.size(); ++k)
      if (covers(b[k])) {
        node[b[k]].zone = s.zone;
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
      if (node[j].zone.dim() != dim2) continue;
      Dbm t = node[j].zone;
      if (t.try_union(s.zone)) {          // union is exactly convex: absorb, no approximation
        node[j].zone = std::move(t);
        if (reccheck) classify_atoms(node[j].zone, *atoms, &ntag_[j], &ntag_out_[j]);
        intern_grew = true;
        return j;
      }
    }
  }
  const int id = static_cast<int>(node.size());
  node.push_back(std::move(s));
  if (ntag_.size() <= static_cast<std::size_t>(id)) ntag_.resize(id + 1, 0);
  if (ntag_out_.size() <= static_cast<std::size_t>(id)) ntag_out_.resize(id + 1, 0);
  ntag_[id] = tag;
  ntag_out_[id] = tag_out;
  b.push_back(id);
  ++outer_nodes;
  grow();
  return id;
}

// SPLIT A GENERATED ZONE INTO ATOM-HOMOGENEOUS PIECES.
//
// Splitting is set-theoretic: the pieces partition the zone exactly, so post, pre and stop_set
// all distribute over them and the greatest fixpoint is unchanged. It is also what makes the
// cheap cell tag EXACT -- a straddling zone has no well-defined tag, so the tagged covering
// test would stop being a well-quasi-order and the search could loop.
//
// stop_set stays correct on a piece even though a piece need not be delay-closed:
// state_has_deadlock rebuilds the delay closure Up(Z) /\ Inv(q) before testing, which is the
// set the escape regions must be measured against, and gfp_on intersects the result back with
// node[v].zone. (Up(Z) /\ Inv) \ live, intersected with Z, is exactly Z \ live.
void Otf::split_by_atoms(SymState&& s, std::vector<SymState>& out) {
  out.clear();
  const std::vector<AtomC>* at = nullptr;
  if (amap_) {
    auto it = amap_->diag.find(disc_key_bytes(s.locs, s.ints));
    if (it != amap_->diag.end() && !it->second.empty()) at = &it->second;
  }
  out.push_back(std::move(s));
  if (!at) return;
  // SPLIT AT COMPARISON: emit the zone WHOLE. Every deferred obligation is discharged in
  // Otf::cover_rec.
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
  gfp_succ_tagged(net, node[u], L, U, qmask, emode, succ, tags);
  for (std::size_t k = 0; k < succ.size(); ++k) {
    pieces.clear();
    if (amap_) {
      split_by_atoms(std::move(succ[k]), pieces);
    } else {
      pieces.push_back(std::move(succ[k]));
    }
    for (std::size_t pi = 0; pi < pieces.size(); ++pi) {
    const int v = intern(std::move(pieces[pi]));
    grow();
    adj[u].push_back(OtfEdge{v, tags[k]});
    pred[v].push_back(u);
    // WITNESS AT GENERATION, not at expansion.
    // In-place enlargement invalidates the enlarged node's adjacency (it was computed from
    // the smaller zone).
    if (intern_grew) {
      if (v != u) {
        expanded[v] = 0;
        for (const auto& e : adj[v]) {
          auto& pv = pred[e.tgt];
          pv.erase(std::remove(pv.begin(), pv.end(), v), pv.end());
        }
        adj[v].clear();
      } else {
        expanded[u] = 0;
      }
    }
    }
  }
}

bool Otf::time_diverges(const SymState& s) const {
  if (net.nclocks == 0) return false;
  // FROZEN LOCATIONS DO NOT DELAY. Delaying unconditionally reports divergence at a
  // committed/urgent location, where no positive delay is legal at all -- inventing a
  // maximal-run witness out of nothing.
  { bool fc, fu; frozen_flags(net, s.locs, &fc, &fu); if (fc || fu) return false; }
  Dbm z = s.zone;
  z.delay();
  std::vector<Constraint> inv;
  all_invariants(net, s.locs, inv);
  apply_all(z, inv);
  if (z.empty()) return false;
  for (int i = 1; i <= net.nclocks; ++i)
    if (z.get(i, 0) != kInf) return false;
  return true;
}

// The END-OF-RUN term, whole-zone by necessity, not by laziness -- identical to
// leads_to_gfp's. Stored zones are LU-extrapolated, so they contain points the system
// never reaches; each is LU-SIMULATED BY a reachable one, so "this point has an infinite
// !q run" transfers to the reachable point (sound) while "this point is STUCK" does not
// (unsound). Hence the whole-zone convention, inherited from the shipping engine, together
// with its known incompleteness on partial deadlocks.
//
// The EXACT stop set: the valuations of Z_v from which no transition is ever enabled, or
// from which time diverges.
//
// The whole-zone form admits all of Z_v as soon as ONE valuation is stuck, which
// over-approximates and leaves the "non-empty fixpoint implies a witness" direction
// conditional. The exact set is already computed inside state_has_deadlock -- it subtracts
// the union of the enabled source regions from the delay-closure of Z_v -- and merely
// discarded. The two whole-zone cases stay exact: an empty adjacency means every valuation
// is stuck, and time_diverges is a property of the delay-closed zone as a whole.
//
// Historical note, because the naive version of this WAS measured wrong: a per-valuation
// stuck test on the STORED zone reported a false violation of arbiter4 `C1.cs --> C1.idle`,
// witnessed by x0..x3 > 4 -- points Extra_LU+ invents (U = 4). That failure was traced to the
// active-clock reduction leaving the stored zone not delay-closed, which state_has_deadlock
// now repairs before testing. This uses the repaired routine.
Federation Otf::stop_set(int u) {
  Federation f;
  if (adj[u].empty() || time_diverges(node[u])) { f.push_back(node[u].zone); return f; }
  Federation d;
  if (state_has_deadlock(net, node[u], &d)) f.swap(d);
  return f;
}

// Descending greatest-fixpoint iteration CONFINED TO S. Nodes outside S are pinned to
// empty, which is what makes the limit a post-fixpoint of the full operator rather than
// merely of this one. S must contain only EXPANDED !q nodes.
void Otf::gfp_on(const std::vector<int>& S, std::vector<Federation>& W) {
  ++passes;
  const std::size_t N = node.size();
  std::vector<char> inS(N, 0);
  for (int v : S) inS[v] = 1;
  W.assign(N, Federation{});
  std::vector<Federation> nosucc(N);
  for (int v : S) {
    W[v].push_back(node[v].zone);
    nosucc[v] = stop_set(v);
  }
  // ROUND-ROBIN, and it is not an oversight -- a worklist was tried and REVERTED.
  //
  // The obvious optimisation is a true chaotic worklist: W(v) reads only its successors, so
  // re-queue v exactly when one of them shrinks (`pred` is maintained for it). Measured, that
  // is a NET LOSS twice over:
  //   * it saves nothing where it should. `fischer7 P0.req --> P0.wait` under the hull goes
  //     825891 -> 824775 Pre calls, 0.1%. Every node genuinely needs recomputing, because the
  //     values do propagate: there is no wasted sweep to remove.
  //   * it REGRESSES grc4 `T0.in --> T0.far` from SATISFIED (128 states) to UNUSABLE. Changing
  //     the order does not change the limit of a monotone descending iteration, but it does
  //     change the intermediate FEDERATION SHAPES, and the width cap is a resource bound on
  //     those. Letting some nodes descend far ahead of their neighbours fragments the
  //     federations enough to trip the cap at 5147 Pre calls where the lockstep sweep converges
  //     at 56892.
  // The lockstep sweep is doing real work as a de-facto widening-free normaliser. Do not
  // "optimise" it again without measuring the cap.
  bool changed = true;
  std::size_t it = 0;
  while (changed && !capped) {
    if (cancelled()) return;
    changed = false;
    ++it;
    for (int v : S) {
      if (gfp_fed_empty(W[v])) continue;
      Federation cand = nosucc[v];
      for (const auto& e : adj[v]) {
        const int j = e.tgt;
        if (!inS[j] || gfp_fed_empty(W[j])) continue;
        for (const auto& Y : W[j]) {
          ++pre_calls;
          Dbm pz = gfp_pre(net, e.tag, node[v], node[j], Y);
          if (!pz.empty()) cand.push_back(std::move(pz));
        }
      }
      gfp_fed_compact(cand, fedcap, &capped);
      // z /\ cand, DIRECTLY.
      //
      // The published form is `fed_subtract(z, fed_subtract(z, cand))`, which denotes the same
      // set but computes it as a double complement: the inner subtraction peels z by every
      // constraint of every member of cand, so its output is an ARRANGEMENT of |cand| zones --
      // its size is not bounded by |cand| at all -- and the outer subtraction then peels z by
      // all of those. The intersection of two zones is a single zone (dbm_intersect is
      // elementwise min plus a close), so `z /\ cand` needs exactly |cand| pieces:
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
  gfp_iters += it;
}

// A certificate over S: some p /\ !q node in S keeps a non-empty value once the confined
// iteration has CONVERGED. Intermediate iterates are pre-fixpoints and say nothing, so the
// test may only be applied to the limit -- which is why gfp_on runs to fixpoint before
// this looks at anything.
//
// A CAPPED pass has not converged, so it proves nothing IN EITHER DIRECTION. Returning
// `false` here would launder "I ran out of federation budget" into "no certificate", i.e.
// into a silent SATISFIED -- the worst possible failure mode for a verification tool. The
// `capped` flag is therefore sticky and the caller must report UNUSABLE rather than a
// verdict; see Otf::run and the printout in main_gfp.cpp.
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
  for (int v : S)
    if (is_p(v) && !gfp_fed_empty(W[v])) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Forward construction, shared. Level-synchronous BFS, the same shape as leads_to_gfp's
// phase 1. `trigger` is called after every level and may settle the query.
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
    frozen_flags(net, s0.locs, &c, &u);
    if (!c && !u) {
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


// The !q nodes that can actually influence the verdict: those forward-reachable, THROUGH !q
// EDGES ONLY, from a p /\ !q node. F is defined per node from its own successors, so the
// value at an accepting node depends on exactly this set and on nothing else -- restricting
// to it is EXACT, not an approximation, and it is what keeps the final pass from iterating
// over !q states no accepting state can ever reach. (`p --> q` is decided by the accepting
// nodes alone, so nothing outside the closure is ever read.)
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

// One exact pass over the FINISHED graph: S = the accepting closure, on which F_S = F at
// every accepting node, so the limit is the true greatest fixpoint where it is read. This is
// the published algorithm, and every mode falls back to it when no certificate fired early.
bool Otf::final_verdict_violated() {
  return certificate_in(accepting_closure());
}

// THE DEADLOCK-COMPLETION CERTIFICATE.
//
// Deadlock is a negative property, so a_LU covering can lose a deadlock witness. Rather than
// weaken the covering to a_M or pay a second search (the gate/race), put the deadlock INTO the
// automaton: add a sink with an edge guarded by Dead(q) at every q. The completed automaton is
// deadlock-free by construction, so a maximal run that ENDS in a deadlock becomes a run that
// REACHES the sink -- an existential property, which any simulation preserves. The completion is
// never materialised; its only effect is on which atoms the constraint map has to carry, namely
// the atoms of Dead(q), closed under pre.
//
// Those atoms may be dropped at q when Dead(q) is EMPTY (nothing to preserve), when it is ALL of
// Inv(q) (its atoms are already carried by the invariant), or when Inv(q) bounds no clock (time
// is open, so the divergence disjunct of stop already covers the whole zone). If every reachable
// key falls into one of those three cases the seed is empty everywhere, the constraint map is
// the ordinary one, and PLAIN a_LU COVERING IS ALREADY SOUND -- no gate, no second search, and
// no weakening to a_M.
//
// This sweep decides exactly that. It is STRUCTURAL: it asks about Dead(q) over all of Inv(q),
// not over any reachable zone, so it needs no search -- one federation computation per DISCRETE
// KEY (256 of them on arbiter6, against the 28591-state reachability traversal the gate's
// deadlock-freedom lane used to pay). Enumerating the keys from the a_LU graph is sound because
// a simulation never loses a reachable discrete key: the covering zone does at least what the
// covered one does, so every key the covered zone could reach is still generated.
bool Otf::dtheta_empty() {
  std::unordered_set<std::string> seen;
  for (std::size_t i = 0; i < node.size(); ++i) {
    const std::string k = disc_key_bytes(node[i].locs, node[i].ints);
    if (!seen.insert(k).second) continue;
    SymState u; u.locs = node[i].locs; u.ints = node[i].ints;
    u.zone = Dbm::universe(net.nclocks);
    Dbm inv = Dbm::universe(net.nclocks);
    bool c2, u2; frozen_flags(net, u.locs, &c2, &u2);
    std::vector<Constraint> ci; all_invariants(net, u.locs, ci);
    if (!c2 && !u2) inv.delay();
    if (!apply_all(inv, ci)) continue;                        // vacuous key
    Federation stuck, esc;
    if (!state_has_deadlock(net, u, &stuck, &esc)) continue;   // Dead(q) = empty
    bool open = true;
    for (int c = 1; c <= net.nclocks; ++c)
      if (inv.get(c, 0) != kInf) { open = false; break; }
    // (iii) is a claim about TIME DIVERGENCE, so it is only available where time can pass. At a
    // committed/urgent key no positive delay is legal, Inv(q) is the undelayed universe and so
    // trivially bounds no clock -- which is exactly when this test would fire, and exactly when
    // it is wrong. Frozen keys fall through to (ii)/PARTIAL.
    if (open && !c2 && !u2) continue;                          // divergence already covers stop
    // Inv \ Dead = Inv ^ union Esc, so "Dead(q) = Inv(q)" is "nothing in Inv(q) escapes" --
    // one DBM intersection per outgoing transition instead of peeling Inv(q) by the residue
    // ARRANGEMENT. Same predicate; see the twin test in build_atom_map for why it matters.
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


// BUILD THE DEADLOCK-COMPLETION CONSTRAINT MAP from this run's finished graph.
//
// SEED. Only a key with a genuinely PARTIAL Dead(q) contributes. The three degenerate cases
// dtheta_empty() already decides carry no information and are skipped: Dead(q) empty (there is
// no sink edge), Dead(q) = Inv(q) (the sink edge is always enabled and every stored zone
// satisfies Inv(q)), and Inv(q) time-open at a NON-frozen key (the divergence disjunct of stop
// already admits the whole zone). The frozen guard is not optional -- at a committed/urgent key
// Inv(q) is the undelayed universe and so trivially bounds no clock, which is exactly where
// that reasoning fails; see Change 5b.
//
// The atoms are read straight off the CANONICAL DBMs of the Dead(q) federation rather than
// derived from the guards. Dead = \/_i /\_j c_ij, so the truth values of the c_ij decide
// membership exactly and every atom family comes for free -- including the lower bound that
// survives Down(), which dbm_down re-derives when it re-closes after zeroing row 0, and which a
// hand-derived formula for Down(K) got wrong in half the cases.
//
// Atoms IMPLIED BY Inv(q) are dropped: every stored zone satisfies Inv(q), so such an atom is
// constantly true on everything the search will ever compare and distinguishes nothing. On a
// canonical DBM the test is one entry comparison.
//
// PRE-CLOSURE. The map must satisfy [R]^-1(G(q')) subseteq G(q) for every edge (q,g,R,q'), or
// two zones sharing a cell at q could step into different cells at q'. The pullback of
// x_i - x_j <| b through a reset set R is the index substitution i,j -> 0 for reset clocks --
// the same rule state_has_deadlock uses for target invariants. A pullback that lands on clock 0
// is a NON-diagonal atom and is discharged by the bound raise instead. Dropping an atom at the
// target only ever RELAXES the clause at the source, so the three seed drops above stay safe
// under closure. Guard atoms need no seeding: a_LU is already a simulation for the ORIGINAL
// automaton, and the only edge this construction adds is the sink's.
bool Otf::build_atom_map(AtomMap& am) {
  const int nc = net.nclocks;
  am.diag.clear();
  am.bump.assign(nc + 1, kNoBump);
  am.live.assign(nc + 1, 0);
  am.usable = false;
  if (nc <= 0 || nc >= 63) return false;

  // HARD 64. The per-node cell tag is a uint64 bitmask, so an atom past bit 63 would be
  // silently dropped from the covering test while still splitting zones -- a lost obligation,
  // i.e. a wrong verdict, not merely a coarseness loss. Exceeding the cap returns "unusable"
  // and falls back to the race, which is always correct.
  //
  // THE DEFAULT IS THE HARD LIMIT, not half of it. It used to be 32, which cost nothing while
  // the construction was expensive -- declining early meant declining into the race, and the
  // race was competitive. It is not competitive any more, and 32 was where grc5 fell out:
  // one key there needs 36 seed atoms, so the map was declared unusable and grc5 ran the race
  // (20.9s) or a_M (18.5s) when the construction does it in 3.4s. There is no reason to
  // decline below the point where the representation actually breaks.
  constexpr std::size_t kMaxAtoms = 64;

  std::unordered_map<std::string, Dbm> invof;

  // ---- seed ---------------------------------------------------------------
  for (std::size_t i = 0; i < node.size(); ++i) {
    const std::string k = disc_key_bytes(node[i].locs, node[i].ints);
    if (invof.count(k)) continue;
    SymState u;
    u.locs = node[i].locs;
    u.ints = node[i].ints;
    u.zone = Dbm::universe(nc);                 // NOT Dbm(nc): that is the ZERO zone (Change 5a)
    Dbm inv = Dbm::universe(nc);
    bool fc, fu;
    frozen_flags(net, u.locs, &fc, &fu);
    std::vector<Constraint> ci;
    all_invariants(net, u.locs, ci);
    if (!fc && !fu) inv.delay();
    if (!apply_all(inv, ci)) continue;                       // vacuous key
    invof.emplace(k, inv);
    Federation stuck, esc;
    if (!state_has_deadlock(net, u, &stuck, &esc)) continue;  // Dead(q) = empty
    bool open = true;
    for (int c = 1; c <= nc; ++c)
      if (inv.get(c, 0) != kInf) { open = false; break; }
    if (open && !fc && !fu) continue;                         // divergence already covers stop
    // Dead(q) = Inv(q)? Written as `fed_subtract(inv, stuck).empty()` this peels Inv(q) once
    // per piece of the RESIDUE, and the residue is an arrangement -- 472 pieces at grc4's one
    // partial key, which is 126ms, two thirds of the whole structural sweep and the single
    // largest term in the atom construction's bill. It never needed the residue at all:
    //     Inv \ Dead = Inv \ (Inv \ union Esc) = Inv ^ union Esc,
    // so the test is "does NO valuation of Inv(q) escape", which is |Esc| independent DBM
    // intersections -- one per outgoing transition, five on that key.
    //
    // THE PREMISE IS INCLUSION, NOT EQUALITY, and the difference is real. `stuck` is
    // `Z \ union esc` for the `Z` state_has_deadlock built, and that `Z` is NOT always this
    // `inv`: at a FROZEN key state_has_deadlock skips both the delay and the invariant, so its
    // Z is the bare universe while `inv` here has the invariant applied. What the identity
    // needs is only `inv subseteq Z` --
    //     x in inv \ (Z \ E)  iff  x in inv and not (x in Z and x notin E)  iff  x in inv ^ E,
    // using x in inv => x in Z -- and that holds both ways round: equal when not frozen, and
    // `inv = universe ^ Inv subseteq universe = Z` when frozen. A self-check ran the
    // old federation form alongside and aborts on any disagreement; it is clean across the
    // 48-query differential and the three liveness regressions.
    //
    // ESCAPES THAT MISS Inv(q) ARE NOT ESCAPES. Dead(q) = Inv(q) \ union Esc, so a transition
    // whose escape region does not meet Inv(q) subtracts nothing and its atoms decide nothing
    // -- yet every one of its constraints would be seeded, pulled back through the whole key
    // graph, and split against. Drop those members here, where the intersection is being
    // computed for the Dead=Inv test anyway.
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
    // WHICH ZONES TO READ THE ATOMS OFF -- two exact representations of the same set.
    //
    // Dead(q) = Inv(q) \ union Esc(q), and every stored zone satisfies Inv(q). So membership in
    // Dead(q) is decided EITHER by the atoms of the residue's disjoint pieces (Dead = \/_i /\_j
    // c_ij) OR by the atoms of the ESCAPE zones alone -- one convex zone per outgoing transition
    // -- since nu is dead iff it escapes by none of them. Both are exact; neither dominates.
    // The residue's cut constraints are artefacts of the peeling order and blow up on grc4
    // (3234 occurrences against 25 for the escapes), while the escapes carry diagonals that pull
    // back through resets into non-diagonal atoms and cost csma four clock bounds the residue
    // does not. So compute both and keep the cheaper, counting a non-diagonal atom heavier
    // because it is discharged by raising that clock to M GLOBALLY, not by a local cell.
    //
    // MINIMAL GRAPH, not the dense matrix. A canonical DBM's redundant entries are implied by
    // shortest paths through the others, so the non-redundant edges alone decide membership --
    // mindbm's round trip reproduces the canonical matrix exactly. Reading the dense matrix
    // seeds one atom per DERIVED bound and buys nothing.
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
            // BOTH CONSTANT CASES, not just one. Every stored zone at q satisfies Inv(q), so an
            // atom Inv(q) IMPLIES is true on all of them and an atom Inv(q) is DISJOINT from is
            // false on all of them; either way its tag bit never varies and it distinguishes
            // nothing. Only the first was being dropped. On a canonical DBM both are one entry
            // comparison: implied is inv(a,b2) <= r, disjoint is inv(b2,a) + r < (<=,0).
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
  // with no seed anywhere it has nothing to move and the key-edge relation it would move them
  // along need not be built at all. That relation is a string key and a reset mask per graph
  // EDGE -- on fischer7, 9798 nodes' worth -- and building it is the only work this function
  // does that the certificate it replaced did not. Returning here is what makes the nine
  // empty-seed models a literal no-op rather than a no-op plus a sweep.
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
        for (const auto& pr : e.tag.parts)
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
          auto iv = invof.find(e.s);
          if (iv != invof.end()) {
            if (iv->second.get(i2, j2) <= a.b) continue;                      // Inv(q) implies
            // ...and, symmetrically, Inv(q) refutes: the pullback lands outside the source
            // key's invariant, so no zone stored there can ever satisfy it. This is where the
            // drop earns its keep -- a closure atom is a constraint from SOMEWHERE ELSE's
            // deadlock, and the further it is pulled back the less it has to do with the
            // invariant it arrives at.
            if (raw_add(iv->second.get(j2, i2), a.b) < kZeroLe) {
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
  // clock the LU preorder collapses to "equal, or both above M", which is SYMMETRIC, hence a
  // bisimulation, hence deadlock-preserving on its own (Behrmann-Bouyer-Larsen-Pelanek 2006,
  // Lemma 1). Keeping the cell tags on top of that would only split zones a_M already keeps
  // apart. Dropping them here is what stops the construction from ever being WORSE than the
  // a_M fallback it is meant to beat.
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
  // HOW CLOSE TO a_M DID THE BUMPS LAND? The bump is "a_M on one clock", so a map that bumps
  // every clock to M has already paid the whole a_M price and the cells on top of it are pure
  // loss. The all-at-M shortcut above catches only the total case; this reports the margin, and
  // it is the number that decides whether the construction can pay off on a given model at all.
  return true;
}

// The forward half of Base mode. `setup()` must already have run.
void Otf::base_forward() {
      // Control: builds the whole graph and runs one exact pass, i.e. Algorithm 1 through
      // this code path. Kept so the differential can check that the shared machinery
      // reproduces the published node count.
      fwd_ok_ = build_all();
      if (!fwd_ok_) return;
      // DEDUPLICATED WORKLIST, not a level-synchronous frontier.
      //
      // The level-synchronous form pushes a successor whenever `!seen[v] || !expanded[v]`, with
      // no dedup WITHIN a level. That is harmless while `intern` only ever appends, because
      // `expanded` then goes 0 -> 1 once per node. It is quadratic as soon as `intern` MERGES:
      // every enlargement resets `expanded[v]` to 0, so v is re-queued once per in-edge, every
      // level, and the frontier fills with duplicates of the same handful of nodes. That, and
      // not the abstraction, is most of what made the hull lane slow on `fischer`.
      //
      // An `inq` flag makes each node resident at most once. Nothing about WHAT is explored
      // changes -- the closure is the same, so stored-state counts are unaffected -- only how
      // many times it is queued.
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
    // D6 over-approximates, so a non-empty fixpoint is NOT a witness -- it may rest entirely on
    // hull points the system never reaches. Only the empty direction concludes.
    // Do NOT count the initial state, matching verifyta. Its `E<> p` figures are exactly our
    // count minus one on every query measured (5 against 6, 6 against 7, 9 against 10), which
    // pins the convention: it reports the states GENERATED, not the root it started from.
    gs->outer_nodes = outer_nodes;
    gs->outer_expands = outer_expands;
  }
}

// Entry point. Returns true iff `p --> q` HOLDS.
// DEADLOCK-FREEDOM GATE, AND THE RACE THAT DISCHARGES IT.
//
// SUPERSEDED AS THE DEFAULT, kept as the fallback. What follows is the original argument and
// it is still exactly why the race is correct -- but the race now runs only when the
// deadlock-completion constraint map cannot be represented (more than 64 atoms at one key).
// The default is: build the forward graph, derive the map, and either return phase 1's own
// verdict (empty map -- the certificate) or search once more under the refined covering. The
// order below, "phase 1 to a verdict, then discharge", is therefore no longer what happens on
// the default path: phase 1 stops after its FORWARD half, because the refined search would
// otherwise repeat its backward fixpoint. See Otf::build_forward.
//
// a_LU covering is a SIMULATION. It preserves the positive witnesses -- an infinite run of
// discrete steps, and time-divergence -- unconditionally, because a simulating valuation does
// at least what the simulated one does. It does NOT preserve the third witness, DEADLOCK,
// which is "enables nothing": a universally quantified negative property that simulation says
// nothing about. So `stop`'s deadlock disjunct may be LOST under a_LU covering.
//
// The error is one-sided, and that is the whole lever. Stored zones are exact (see the note in
// Otf::setup), and pre/stop are exact concrete operations, so a witness FOUND under a_LU sits
// at a genuinely reachable valuation and is genuine. Only the ABSENCE of a witness is
// inconclusive. Hence:
//
//   phase 1  run under a_LU. Violated => done, and the gate is never paid at all. This is the
//            common case for counterexample queries, which is exactly where this engine is
//            far ahead of verifyta -- hoisting the gate in front of it would make every such
//            query pay for a full reachability traversal whose answer it never needs.
//   phase 2  no witness. Two ways to discharge the obligation, deciding the SAME proposition:
//              (A) prove the model deadlock-free -- then nothing could have been dropped, so
//                  the a_LU answer was sound after all;
//              (B) redo the search with exact-inclusion covering, which is sound outright.
//
// NEITHER DOMINATES and which one wins is not predictable: guessing (A) is cheap means
// deciding deadlock-freedom, and guessing (B)'s size is the query itself. Measured symbolic
// states in the shipping engine: fischer7 is 180420 for (A) against 30994 for (B); arbiter6 is
// 28591 against 66393. A factor of 5.8 one way, 2.3 the other.
//
// So run both and take the first CONCLUSIVE answer, cancelling the loser. What the cores buy
// here is not speed -- neither lane runs faster than it would alone -- but the removal of an
// undecidable choice. A sequential tool must gamble and can lose by 5.8x.
//
// BOTH LANES ARE CANCELLABLE, and that is the difference from the shipping engine's version at
// src/netreach.cpp:3486. There, `search()` never reads the stop flag, so `lane_b.join()` waits
// for lane B unconditionally and the wall clock is always lane B's -- lane A can only ever add
// work. Here lane B polls through Otf::cancelled() (checked in expand(), the choke point for
// every forward step, and at the head of each backward fixpoint sweep), so the pair really does
// finish in min(A, B).
//
// ASYMMETRY TO PRESERVE: lane A is conclusive only when it answers "deadlock-free". "A deadlock
// is reachable" says nothing about THIS query -- the dropped witness may or may not have been
// one -- so that outcome must not publish a verdict and must not cancel lane B.
bool leads_to_otf(const Network& net, const DiscPred& p, const DiscPred& q, GfpStats* gs) {
  // Phase 1.
  Otf e1(net, p, q);

  // Diagonal models never used a_LU (Otf::setup forces inclusion), so phase 1 is already
  // exact: there is nothing to discharge and no atom construction to run.
  if (network_has_diagonal(net)) return e1.run(gs);

  // THE GENERAL CONSTRUCTION. It subsumes the certificate -- an empty map IS dtheta_empty's
  // condition -- so where the certificate used to fire, this is the same structural sweep and
  // the same a_LU search, node for node. Where the certificate DECLINED it carries the model
  // instead of falling back to the race. Best of 3, `Bus.Active --> Bus.Idle` and
  // `T0.in --> T0.far`:
  //
  //     model    certificate   THIS     a_M       a_LU floor (unsound)
  //     csma6      0.138       0.018    0.142     0.012
  //     csma7      3.985       0.053    9.729     0.033
  //     grc4       0.085       0.042    0.056     0.031
  //     grc5      21.936       3.535   19.475     5.815
  //
  // grc5 is the one to look at twice: the REFINED covering beats the coarser one it refines
  // (3.5s against the 5.8s floor), because splitting a zone along an atom of Dead(q) also
  // splits the backward federations that pass through it, and narrower federations converge
  // faster than the arrangement one wide zone produces.
  //
  // The race below survives only for a map that exceeds the 64-bit tag cap.
  //
  // FORWARD ONLY, for now. The atom map is read off this graph; if it turns out non-empty the
  // refined search replaces this run wholesale and phase 1's backward fixpoint -- 80% of its
  // wall clock -- would have been dead work. It is paid below, via e1.finish_verdict(), on
  // exactly the paths that still need phase 1's own verdict.
  e1.build_forward(gs);

  // Phase 2: THE ATOM MAP (see AtomMap and Otf::build_atom_map). Where the deadlock-completion
  // certificate only DECIDES whether the completion's seed is empty, this builds the seed,
  // closes it under pre, and redoes the search under a_LU refined by the resulting cells.
  //
  // The verdict is sound either way round. VIOLATED is genuine outright -- zones are exact and
  // pre/stop are exact concrete operations -- and HOLDS is sound because the refined covering
  // preserves the sink's reachability, hence deadlock. If the map exceeds its atom cap the
  // whole block falls through to the certificate and the race, unchanged.
  //
  // NO CLOSURE-DOMINANCE PREDICTOR. There used to be one here: when the pre-closure added more
  // atoms than the seed did, it abandoned the construction and ran a_M instead, because on grc4
  // (139 closure atoms against 25 seed, 100% straddle) the atoms lane measured 0.18s against
  // a_M's 0.05s. That reading was wrong about WHERE grc4's time went. Split by phase, the atoms
  // lane was 0.035s of searching behind 0.126s of structural sweep and 0.025s of a phase-1
  // backward fixpoint that the refined search then repeated. Both of those are now gone -- the
  // sweep by the `Inv ^ Esc` rewrite of the Dead=Inv test, the fixpoint by deferring phase 1's
  // backward half -- and the atoms lane wins outright on the very model the predictor was built
  // to hand to a_M. The straddle rate was never the bill: grc4 splits 100% of its tagged zones,
  // stores 1.34x the a_LU nodes for it, and the refined search still costs the same.
  AtomMap am;
  if (e1.build_atom_map(am) && am.usable) {
    if (am.diag.empty() && am.bumped == 0) {
      // EMPTY MAP: this IS the deadlock-completion certificate, so phase 1's a_LU covering was
      // already deadlock-sound and phase 1's own verdict is the answer. Its backward fixpoint
      // was deferred, not skipped -- pay it here, exactly once, on the graph already built.
      return e1.finish_verdict(gs);
    }
    Otf e2(net, p, q);
    e2.set_atoms(&am);
    GfpStats gs2;
    const bool holds2 = e2.run(&gs2);
    if (gs) *gs = gs2;
    return holds2;
  }

  // The construction cannot run (atom cap), so phase 1's verdict is owed after all: pay the
  // backward pass build_forward deferred, then rejoin the certificate/race path.
  if (!e1.finish_verdict(gs)) return false;        // genuine witness

  // THE DEADLOCK-COMPLETION CERTIFICATE (see Otf::dtheta_empty). If the completion's
  // constraint-map seed is empty at every reachable key, the ordinary constraint map already
  // preserves deadlock and phase 1's a_LU covering was sound as it stands. One federation
  // computation per DISCRETE KEY and no search -- against a full reachability traversal for
  // the deadlock-freedom lane.
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
  if (v == 1 && gs) *gs = gs_b;                     // lane B won and it disagreed with phase 1
  if (v >= 0) return v == 0;

  // Unreachable in the current logic: lane B is cancelled only by lane A concluding
  // "deadlock-free", which publishes verdict 0. Kept so that a future change to either lane
  // cannot silently turn an inconclusive pair into a wrong answer.
  return Otf(net, p, q, /*force_exact=*/true).run(nullptr);
}

}  // namespace ptctlmc
