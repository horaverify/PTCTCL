// netliveness.cpp — p --> q (leads-to), the backward greatest-fixpoint liveness engine.
//
// Split out of netreach.cpp, which held both engines and ran to 4000+ lines. This half is a
// separate entry point (leads_to_otf, declared in netreach.hpp) reached only from the query
// router; it shares no mutable state with the forward search and calls into it only through
// network_has_diagonal and deadlock_reachable, both already externally visible. The transition
// semantics the two engines MUST agree on live in detail/net_semantics.hpp and are inline there,
// so this translation unit gets its own inlinable copy of them.

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

// ============================================================================
// PROTOTYPE (candidate C1): backward greatest-fixpoint liveness on the
// subsumption-closed forward graph.
//
// NOT ON ANY PRODUCTION PATH. This file is a COPY of src/netreach.cpp used only
// by the standalone gfp_proto binary; the shipping engine is untouched.
// See docs/superpowers/specs/parallel-search-novelty.md, candidate C1.
//
// Phase 1  forward BFS with inclusion subsumption -> ONE zone per reachable
//          discrete state on these families (the measured floor), with every
//          transition TAGGED by the edge that produced it.
// Phase 2  restrict to !q.
// Phase 3  chaotic backward greatest fixpoint  W <- W  /\  ( NoSucc \/ Pre(W) ),
//          one federation per node.
//
// The key point the candidate rests on: a SUBSUMPTION EDGE (target zone stored
// is strictly bigger than the real successor) is FATAL for forward cycle
// detection -- it closes loops that the concrete system cannot run -- but it is
// HARMLESS here, because the backward operator only ever uses the target's W as
// an upper bound, and over-approximating the target keeps Omega subseteq W,
// which is the direction soundness needs.
// ============================================================================

// A transition is a SET of participating edges: one for a local edge, two for a binary
// synchronisation, and 1 + n for a broadcast (sender plus every receiver that fired). Keeping
// them in one list means gfp_pre has a single code path for all three.
struct GfpTag {
  std::vector<std::pair<int,int>> parts;   // (process, edge index)
  // BROADCAST ABSTENTION, the backward-direction half of build_broadcast_parts.
  //
  // A receiver that does NOT fire still constrains the transition: it stays put precisely
  // because its own clock guard fails, so the step is enabled only where the COMPLEMENT of
  // that guard holds. `parts` cannot say that -- an abstaining receiver contributes no edge --
  // so the constraints ride here. They are a pure guard on the SOURCE: no reset, no location
  // change, no integer update, which is why gfp_pre can apply them at exactly the same point
  // as the participants' guards and nowhere else. Empty for local edges, binary sync, and
  // every broadcast whose receivers all carry unguarded a? edges (csma).
  std::vector<Constraint> abstain;
};

// GfpStats now lives in include/ptctlmc/netreach.hpp, alongside the three leads-to entry
// points it instruments. It used to be declared here AND hand-copied into main_gfp.cpp,
// which was an ODR hazard waiting to fire the moment a field was added on one side only.

// Tagged successor generation. Mirrors successors_net for LOCAL edges, BINARY synchronisation
// and BROADCAST, and additionally reports which edges produced each successor.
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
                                         // is part of what enables this step -- keep it for Pre
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

// Exact concrete predecessor of one tagged transition. One code path for local, binary and
// broadcast: every participating edge contributes its guard and its resets.
//
//   forward:  z1 = Z /\ (all guards) ; reset (all resets) ; /\ inv' ;
//             if !frozen' { delay ; /\ inv' }
//   backward: undo the active-clock reduction at the target ; /\ inv' ; down ; /\ inv' ;
//             /\ (x_r = 0 for every reset r) ; free those r ; /\ (all guards) ;
//             down (the delay BEFORE the action) ; /\ inv(src)
//
// No extrapolation and no reset_inactive appear in the forward direction here: this is the exact
// concrete Pre. The abstraction enters only through the caller's confinement W_v subseteq Z_v.
// An EMPTY zone OF THE RIGHT DIMENSION.
//
// `Dbm()` is documented as "a zero-clock placeholder, so a Dbm can live in a container": it is
// a 1x1 matrix holding kZeroLe, and `dbm_empty` tests the DIAGONAL for a negative entry, so
// `Dbm().empty()` is FALSE. Returning it from gfp_pre to mean "the predecessor is empty"
// therefore hands the caller a non-empty 1x1 DBM, which `if (!pz.empty())` duly pushes into a
// federation of (nclocks+1)-dimensional ones. The next `dbm_includes` inside gfp_fed_compact
// reads dim*dim entries out of a 4-byte allocation.
//
// Caught by AddressSanitizer as a heap-buffer-overflow on grc4
// returned SATISFIED on 4 of 10 identical runs and UNUSABLE on the other 6: the verdict was a
// function of whatever heap garbage followed the placeholder. The bug is in gfp_pre and is
// therefore shared by the published leads_to_gfp, not introduced by the on-the-fly work; the
// hull abstraction merely takes the early-return paths often enough to make it fire.
static Dbm gfp_empty(int nclocks) {
  Dbm z(nclocks);
  z.set(0, 0, lt(0));      // 0 - 0 < 0 is unsatisfiable, so dbm_empty() is true
  return z;
}

static Dbm gfp_pre(const Network& net, const GfpTag& t, const SymState& src, const SymState& tgt,
                   const Dbm& X) {
  Dbm Y = X;

  // Undo the active-clock reduction at the target. Those clocks are in no edge's reset set, so
  // without this Pre silently demands they also be 0 at the SOURCE -- an over-constrained Pre
  // that collapses the fixpoint and MISSES violations (measured on phil4/6, grc4, arbiter4/5).
  // Freeing them is the exact inverse and is sound because they are dead: nothing reads them.
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
  // The abstaining receivers' guard complements, applied here and not elsewhere: like a
  // participant's guard they constrain the source valuation, and unlike one they undo nothing,
  // so they belong after the resets have been freed and before the pre-action delay.
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
static bool gfp_fed_empty(const Federation& f) {
  for (const auto& z : f) if (!z.empty()) return false;
  return true;
}
// Is every point of `a` inside the union `b`?
static bool gfp_fed_subset(const Federation& a, const Federation& b) {
  for (const auto& z : a) {
    if (z.empty()) continue;
    if (!fed_subtract(z, b).empty()) return false;
  }
  return true;
}
// Drop empties, drop members covered by one already kept, and MERGE any pair whose union is
// exactly convex. The convex-union merge is the important part and it is free of charge in
// both directions: dbm_try_union succeeds only when the union really is a single zone, so it
// adds no point and removes none. Without it the federations grow monotonically with every
// Pre and the width cap becomes the binding constraint on the whole algorithm.
//
// (The published version also built a `rest` federation per member and then discarded it
// without reading it -- quadratic work for nothing. Removed.)
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



// ============================================================================
// ON-THE-FLY LIVENESS  (branch `liveness-on-the-fly`).
//
// `leads_to_gfp` above is three strictly sequential phases: build the whole aLU-subsumed
// reachable graph, restrict to !q, then run a chaotic backward greatest fixpoint over it.
// Its cost is a function of the MODEL and never of the QUERY -- a `p --> q` violated three
// transitions from the initial state still pays for every reachable symbolic state.
//
// WHY THIS LOOKS HARD, AND WHY IT IS NOT.
//
// EG is a GREATEST fixpoint, so the textbook iteration starts at TOP and DESCENDS, and an
// intermediate iterate is only a PRE-fixpoint (W >= F(W)). A pre-fixpoint is an
// over-approximation and therefore says nothing: "W(v) is non-empty" cannot settle a
// violation, so there is no early exit and no obvious way to interleave with construction.
// That is the real content of "the values are no longer necessarily descending, and you
// need a substantially different convergence/correctness proof".
//
// But a violation never needed the fixpoint. It needs a CERTIFICATE, and by Knaster-Tarski
// the certificates of a greatest fixpoint are exactly its POST-fixpoints:
//
//        X  subseteq  F(X)        ==>        X  subseteq  nu F.                        (*)
//
// So take ANY set S of nodes explored so far, pin every node outside S to EMPTY, and run
// the ordinary descending iteration confined to S. Its limit nu F_S is the exact greatest
// fixpoint of the restricted operator. Restricting only DELETES disjuncts, so F_S <= F
// pointwise, and monotonicity gives nu F_S subseteq nu F: the limit of a pass over a
// FRAGMENT is a post-fixpoint of the operator on the FINISHED graph. By (*) it is a sound
// under-approximation of the final answer, computed without ever seeing the final graph.
//
// Two consequences, and they are the entire design:
//
//   * nu F_S(v) non-empty at a p /\ !q node v  =>  the query is VIOLATED, with exactly the
//     verdict the finished graph would have produced. Stop, and never build the rest.
//   * S = every expanded !q node of the FINISHED graph gives F_S = F, so the last pass IS
//     the published algorithm. Nothing is given up when the property holds.
//
// Under (*) no value has to descend across passes: each pass is a self-contained
// certificate search, and monotonicity in the graph comes for free. The four directions
// below are implemented against that yardstick and measured against each other.
//
// THE ONE RESTRICTION, and it is the only one. A node may enter S only if it is EXPANDED.
// The stopping term of the backward operator needs the ABSENCE of successors
// (`adj[v].empty()`), and an unexpanded node has an empty adjacency for the wrong reason.
// Everything else survives a partial graph untouched: the Pre disjuncts only ever need
// edges that EXIST, never the absence of one. This is what makes the scheme on-the-fly at
// all -- backward work on a half-built graph is legitimate for every node whose own
// successors have been computed, no matter what the rest of the graph is doing.
// ============================================================================


struct OtfEdge {
  int tgt;
  GfpTag tag;
};

// ---------------------------------------------------------------------------
// THE GENERAL DEADLOCK-COMPLETION CONSTRUCTION.
//
// Otf::dtheta_empty decides only the DEGENERATE case: if Dead(q) is empty, or all of Inv(q),
// or Inv(q) is time-open at a non-frozen key, the deadlock completion's constraint-map seed is
// empty and plain a_LU covering is already deadlock-sound. When some key has a genuinely
// PARTIAL Dead(q) the seed is non-empty and the map has to CARRY the atoms of Dead(q), closed
// under pre. This is that map.
//
// An atom is a half-space x_i - x_j <| b. Two kinds, and they are discharged differently:
//
//   DIAGONAL (i > 0 and j > 0). Invariant under time elapse, so agreement on it survives
//   delay, so it can be carried as a CELL TAG: every stored zone is split until it lies wholly
//   inside or wholly outside the atom, the resulting bit is appended to the discrete key, and
//   zones in different cells are never compared. The covering test itself is untouched.
//
//   NON-DIAGONAL (i == 0 or j == 0), i.e. a bound on one clock. NOT delay invariant: two
//   valuations agreeing on `x <= 5` need not agree on it after delaying, so no cell tag can
//   carry it and splitting would not restore the property. What DOES carry it is the a_LU
//   preorder itself -- with L(x) >= c and U(x) >= c its three clauses collapse to "equal, or
//   both above c", which decides every constraint on x with constant <= c the same way for
//   both valuations. So a non-diagonal atom is discharged by RAISING that one clock's bounds:
//   a_M on that clock alone, rather than a_M on the whole model.
//
// With both in place the relation
//     nu R nu'  iff  nu <=_LU nu' (raised bounds)  and  nu, nu' agree on the diagonal atoms
// is a simulation for the COMPLETED automaton A^\perp, because (a) <=_LU is a simulation for A,
// (b) diagonal agreement is preserved by delay and, via the pre-closure below, by every edge,
// and (c) Dead(q) is a boolean combination of Inv(q) (which every stored zone satisfies), the
// diagonal atoms and the non-diagonal ones -- so nu |= Dead(q) iff nu' |= Dead(q). Deadlock is
// then an EXISTENTIAL reachability of the sink, which R preserves.
struct AtomC {
  int i, j;
  raw_t b;                                   // x_i - x_j <| b
  bool operator==(const AtomC& o) const { return i == o.i && j == o.j && b == o.b; }
};

static constexpr std::int32_t kNoBump = -1;  // clock constants are >= 0, so -1 is "unbumped"

struct AtomMap {
  std::unordered_map<std::string, std::vector<AtomC>> diag;  // discrete key -> cell-tag atoms
  std::vector<std::int32_t> bump;   // clock -> constant L(x),U(x) must reach (kNoBump = none)
  std::vector<char> live;           // clocks the atoms read: reset_inactive must not zero them
  bool usable = false;
  std::size_t seed_keys = 0, seed_atoms = 0, nondiag_atoms = 0, closure_atoms = 0;
  std::size_t keys_tagged = 0, rounds = 0, bumped = 0;
  // Atoms the minimisation removed: `dropped_vacuous` = escape regions that miss Inv(q)
  // entirely; `dropped_disjoint` = atoms Inv(q) refutes, so their tag bit is constantly 0.
  std::size_t dropped_vacuous = 0, dropped_disjoint = 0;
};

class Otf {
 public:
  Otf(const Network& n, const DiscPred& pp, const DiscPred& qq,
      bool force_exact = false, const std::atomic<bool>* cancel = nullptr)
      : net(n), p(pp), q(qq), force_exact_(force_exact), cancel_(cancel) {}

  // Returns true iff `p --> q` HOLDS. Meaningless if was_cancelled().
  bool run(GfpStats* gs);

  // THE FORWARD HALF ALONE, and the backward half separately.
  //
  // The atom construction reads only the finished forward GRAPH -- the reachable discrete keys
  // and the key-edge relation with their resets -- and then, if the map is non-empty, throws
  // phase 1 away and searches again under the refined covering. Phase 1's backward fixpoint is
  // ~80% of its wall clock (grc4: 20ms of 25ms) and none of it survives that. So build the
  // graph, decide the map, and pay the fixpoint exactly once: here if the map came out empty
  // (the certificate case, where phase 1's own verdict IS the answer), in the refined search
  // otherwise. Base mode only -- the candidate-confirm modes settle mid-build, which is
  // precisely the case build_atom_map may not read.
  void build_forward(GfpStats* gs);
  bool finish_verdict(GfpStats* gs);          // true iff `p --> q` HOLDS; needs build_forward

  // Structural certificate: is the deadlock-completion's constraint-map seed empty at every
  // reachable key? If so, plain a_LU covering was already deadlock-sound. Needs a built graph.
  bool dtheta_empty();

  // A cancelled run abandoned its exploration part-way, so its verdict is not a verdict.
  // The racing caller in leads_to_otf must consult this before publishing a result.
  bool was_cancelled() const { return cancelled_; }

  // Derive the deadlock-completion constraint map from THIS run's finished
  // graph (so it needs run() to have happened), and install one on a fresh run.
  bool build_atom_map(AtomMap& am);
  void set_atoms(const AtomMap* a) { amap_ = a; }

 private:
  const Network& net;
  const DiscPred& p;
  const DiscPred& q;
  bool force_exact_ = false;                   // gate lane B: inclusion covering, no aLU
  const std::atomic<bool>* cancel_ = nullptr;  // set by the winner of the race
  bool cancelled_ = false;

  // Latched, so one observation is enough: nothing downstream of the first check may see the
  // flag flip back and resume a half-abandoned search.
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
  // OUTER vs INNER accounting. verifyta's leads-to counters report only the OUTER
  // reachability up to the first violating p-state -- measured: on fischer4/csma4/phil4 the
  // `p --> q` figures are byte-identical to `E<> p`, so the nested inevitability check is not
  // counted at all. Ours counts every interned state including the inner DFS's. To compare
  // like with like, split them.
  std::size_t outer_expands = 0, outer_nodes = 0;

  std::vector<SymState> node;
  std::vector<std::vector<OtfEdge>> adj;
  std::vector<std::vector<int>> pred;          // maintained for D1's closure test
  std::unordered_map<std::string, std::vector<int>> bucket;
  std::vector<char> expanded;
  std::vector<char> pc, qc;                    // p/q caches, 2 = unknown
  bool intern_grew = false;
  const AtomMap* amap_ = nullptr;              // null = ordinary a_LU, no cells
  // Per-node cell tag: bit a is set iff node's zone is contained in atom a of its key. Only
  // meaningful when amap_ is installed and the key carries atoms; 0 otherwise.
  std::vector<std::uint64_t> ntag_;
  // Stored zones are NOT atom-homogeneous, so one "contained" bit no longer classifies them:
  // `ntag_out_` carries the second bit of the three-valued classification
  // (In = ntag_, Out = ntag_out_, Straddle = neither).
  std::vector<std::uint64_t> ntag_out_;

  void setup();
  bool holds(const DiscPred& pr, const SymState& s) const;
  bool is_p(int i);
  bool is_nq(int i);
  void grow();
  int intern(SymState&& s);
  void expand(int u);
  bool time_diverges(const SymState& s) const;
  Federation stop_set(int u);                  // EXACT NoSucc; needs expanded[u]

  void gfp_on(const std::vector<int>& S, std::vector<Federation>& W);
  bool certificate_in(const std::vector<int>& S);
  std::vector<int> accepting_closure();

  void split_by_atoms(SymState&& s, std::vector<SymState>& out);
  // Split-at-comparison support; see the block comment above Otf::cover_rec.
  void classify_atoms(const Dbm& z, const std::vector<AtomC>& as,
                      std::uint64_t* in, std::uint64_t* out) const;
  bool base_cover(const Dbm& a, const Dbm& b) const {
    return use_alu ? a.alu_subsumed_by(b, bL, bU) : b.includes(a);
  }
  bool cover_rec(const Dbm& a, std::uint64_t ain, const Dbm& b, std::uint64_t bout,
                 const std::vector<AtomC>& as, int depth);
  void base_forward();                         // Base mode's forward half; setup() done already
  void fill_stats(GfpStats* gs);
  bool fwd_ok_ = false;                        // build_all() found a satisfiable initial state
  bool build_all();                            // full forward BFS; true if a certificate fired
  bool final_verdict_violated();               // one exact pass over the finished graph
 public:
};

void Otf::setup() {
  L = net.lbound;
  U = net.ubound;
  // FINITENESS COMES FROM aLU SUBSUMPTION ON EXACT ZONES, NOT FROM EXTRAPOLATION.
  //
  // These two settings are ONE decision and must move together. Plain inclusion does not bound
  // the zone graph, so with `use_alu` off the only thing that terminates this search is
  // Extra_LU+; and with Extra_LU+ on, every stored zone is an OVER-APPROXIMATION of the
  // reachable set, which is fatal for the `stop` term specifically.
  //
  // WHY IT IS FATAL. Otf::stop_set computes an exact per-valuation deadlock residue, but it
  // computes it from node[u].zone. Extra_LU+(Z) is contained in a_LU(Z), so every valuation
  // extrapolation ADDS does at most what some real valuation does -- enabledness is MONOTONE,
  // not invariant. Hence "some valuation of the extrapolated zone enables e" transfers back to
  // Z, but "some valuation of the extrapolated zone enables NOTHING" does not. stop_set asks
  // the second question. Concretely (diagonal-free, two clocks, one edge):
  //
  //     location q, invariant x <= 10;  edge q -> q2 guarded y >= 3
  //     => L(x)=0, U(x)=10, L(y)=3, U(y)=0
  //     reach q with y reset 6 after x:  Z = { x - y = 6, 6 <= x <= 10 }
  //
  // Z is deadlock-free: from anywhere, delay until y >= 3, i.e. x >= 9, still under the
  // invariant. Extra_LU+ drops the DIAGONAL x - y <= 6 because 6 > L(x) = 0, leaving
  // { x >= 6, y >= 0, x - y >= 6 }, which contains (x=10, y=0) -- that needs d >= 3 for the
  // guard and d <= 0 for the invariant, so it is permanently stuck. Nothing in Z is.
  // state_has_deadlock's delay-closure repair does NOT save this: the dropped constraint is a
  // diagonal, and in a diagonal-free automaton no invariant ever restores a diagonal.
  //
  // `stop` occurs POSITIVELY in the fixpoint operator, so a spurious stop valuation ENLARGES
  // the greatest fixpoint and fabricates a certificate: `p --> q` reported VIOLATED when it
  // holds. That is the opposite error direction from aLU covering, which can only ever lose a
  // witness, and unlike that one it is not recoverable by any later check.
  //
  // On exact zones the whole picture closes: stored zones are genuinely reachable, so stop_set
  // is exact; cycle and divergence witnesses transfer under simulation unconditionally; and the
  // only residual hole is that aLU COVERING may drop a deadlock witness -- one-sided, and
  // discharged by the deadlock-freedom gate (see leads_to_net, src/netreach.cpp:3459).
  //
  // Diagonal models cannot use aLU at all, so they keep ExNoDiag + inclusion and extrapolation
  // stays load-bearing there; their stop term is still exposed. G-simulation is the fix for
  // that case and would land it in this same shape (exact zones + simulation covering); see
  // docs/superpowers/specs/g-simulation-extraction.md.
  //
  use_alu = !force_exact_ && !network_has_diagonal(net);
  emode = network_has_diagonal(net) ? ExNoDiag : (use_alu ? ExNone : ExLuPlus);
  // qmask is the set of clocks the QUERY forces to stay live; a leads-to predicate has no
  // clock constraints, so it is all-zero and the active-clock reduction runs in FULL.
  // (This does NOT disable the reduction -- reset_inactive treats qmask as "additionally keep
  // alive", so all-zero is maximal reduction. What makes the reduction safe for a BACKWARD
  // analysis is the
  // repair inside gfp_pre: it FREES every clock that is dead at the target before undoing
  // the step. Without that, Pre silently demands the dead clocks be 0 at the SOURCE too --
  // an over-constrained Pre that collapses the fixpoint and MISSES violations.)
  qmask.assign(net.nclocks + 1, 0);
  aL = net.lbound;
  aU = net.ubound;
  aL[0] = aU[0] = 0;
  // THE CONSTRAINT MAP'S TWO GLOBAL EFFECTS (see AtomMap).
  //
  // (1) Non-diagonal atoms are discharged by raising that clock's LU bounds. Raising L or U
  //     only ever makes a_LU FINER, so it is sound wherever it is applied; applying it to the
  //     global arrays (and, in intern, to the per-location ones) keeps it consistent along
  //     every path, which is what the simulation argument needs.
  // (2) reset_inactive pins clocks that no guard will read again to 0. The atoms DO read them
  //     -- that is the whole point of Dead(q) -- so every clock occurring in the map is added
  //     to qmask, which reset_inactive treats as "additionally keep alive". Without this the
  //     reduction erases exactly the differences the cell tag is meant to observe.
  //
  // Whether the pinning in (2) is actually owed for SEED atoms is open: a clock read by Esc(q)
  // is read by an outgoing guard or a target invariant at q, hence active at q by construction,
  // and a diagonal that survives pullback involves only clocks some reachable guard still
  // reads. That argument is NOT a proof -- `active` is a static over-approximation computed
  // elsewhere -- so the pinning stays.
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

// SPLIT AT COMPARISON, the alternative to split_by_atoms' split at
// creation. Stored zones are whole, so they may STRADDLE an atom, and the split -- if it is
// needed at all -- happens inside the covering test.
//
// THE THREE-VALUED CLASSIFICATION. With homogeneous zones one bit per atom suffices; with
// straddling zones it does not, and reading a straddle as "disjoint from phi" is the one
// mistake that makes this unsound. Both bits come off the canonical DBM in O(1):
//
//     Z.get(i,j) <= b                       =>  Z subseteq phi          (In)
//     raw_add(Z.get(j,i), b) < kZeroLe      =>  Z /\ phi = empty        (Out)
//     otherwise                                                          (Straddle)
//
// WHAT THE COVERING TEST OWES. `a covered by b` must witness, for every nu in a, some nu' in b
// with nu <=_LU nu' AND, for every atom phi of the key, `nu' |= phi => nu |= phi`. (That
// direction only, and it is the same obligation the homogeneous tag rule discharges -- see the
// note in Otf::intern. It is NOT the g-simulation relation of the forward diagonal work.)
//
// The base case is the EXISTING test, unchanged: `base_cover` (a_LU, or plain inclusion on
// diagonal models) plus, per atom, the requirement that the obligation hold for ALL pairs at
// once -- which it does exactly when `b` is Out (no nu' satisfies phi) or `a` is In (every nu
// does). An atom that is neither is BLOCKED, and only blocked atoms are split. On homogeneous
// operands "blocked" degenerates to "b In and a Out", i.e. bit-for-bit the implication rule.
//
// THE RECURSION (Gastin-Mukherjee-Srivathsan, CAV 2019, Lemma 17), adapted to the implication
// obligation rather than to agreement:
//
//     a <= b   <==   (a /\ phi <= b)  and  (a /\ ~phi <= b /\ ~phi)
//
// nu |= phi discharges the phi-obligation for any nu' whatever, so the first conjunct may keep
// the whole of b; nu !|= phi needs nu' !|= phi, so the second must narrow b as well. Each
// conjunct resolves phi (In on the left, Out on both sides on the right), so the blocked set
// strictly shrinks and the recursion terminates in at most 2^|blocked| leaves.
//
// The whole-operand base test is also a NECESSARY condition for the conjunction -- both
// conjuncts test sub-zones of a against sub-zones of b -- so it doubles as a prefilter that
// kills a hopeless probe before any zone is copied. That is what keeps the common reject path
// the same price it is in the split-at-creation mode.
// THE THREE ARMS, and why split-at-COMPARISON is the default.
//
// Split-at-creation cuts every zone along every atom at its key the moment it is produced, so
// stored zones are atom-homogeneous and covering is one mask AND. Split-at-comparison stores
// zones whole and splits transiently inside the covering test. The LAZY arm also stored
// whole but, when a covering decision genuinely turns on a stored zone's straddle, splits that
// node PERMANENTLY -- once per zone rather than once per probe, and only on atoms that actually
// arose (Herbreteau-Srivathsan-Walukiewicz's lazy abstraction, refine where the property is at
// stake). Lazy is split-at-comparison plus that rewrite, so it inherits its correctness whole.
//
// MEASURED, all three arms, zero verdict differences across five configurations:
//
//   model  a_LU floor  creation      cmp       lazy
//   grc4          883      1179      921        921
//   grc5        13939     14571    12554      12554
//   csma7        1696      2368     2368       2368
//
// Comparison wins and is never slower. The reason is not that its test is cheaper -- it is that
// the recursion is NEVER ENTERED: 0 of grc4's 216 blocked probes and 0 of grc5's 2660 recurse.
// At the keys that carry atoms, the a_LU antichain is already wide enough that no covering
// succeeds at all, so the atoms never refuse a merge that would otherwise have happened, and
// the eager split's only effect is fragmentation. Two measurements pin that: with the
// convex-union merge off, creation goes 1179 -> 2579 on grc4 while comparison is unchanged at
// 921 -- the merge exists to glue back what the eager split broke -- and comparison without the
// atom-clock pinning reaches 883 on grc4, exactly the a_LU floor.
//
// LAZY WAS A NULL RESULT: 0 splits and 0 activations on every model here, so it was
// bit-identical to comparison. Forced to fire it stored MORE than either, because activating an
// atom re-enables eager splitting for that whole key thereafter. It is gone.
//
// Do not read the forward-reachability result as contradicting this. There the same recursion
// costs 5x, because coverings at atom-carrying keys DO succeed there and it genuinely fires.
// Which arm wins is a property of whether the atoms ever change a covering decision.
//
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

// `ain` is the In-mask of `a` and `bout` the Out-mask of `b`; they are the only two of the four
// masks the blocked set reads. At depth 0 they come from the caller's cache (the newcomer's own
// classification, and ntag_out_[j] for the stored node), so an unblocked probe costs exactly
// one mask op plus the base test -- the same price as the homogeneous tag rule.
bool Otf::cover_rec(const Dbm& a, std::uint64_t ain, const Dbm& b, std::uint64_t bout,
                    const std::vector<AtomC>& as, int depth) {
  const std::size_t n = std::min<std::size_t>(as.size(), 64);
  const std::uint64_t all = n >= 64 ? ~std::uint64_t(0) : ((std::uint64_t(1) << n) - 1);
  const std::uint64_t blocked = all & ~bout & ~ain;
  // Necessary condition for both conjuncts, so it is safe to ask it first and it doubles as
  // the reject prefilter.
  if (!base_cover(a, b)) return false;
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

// aLU subsumption, exactly as leads_to_gfp's phase 1, plus the APPEND-ONLY switch.
//
// APPEND-ONLY (D1, D2). The published intern ENLARGES a stored node in place when a
// newcomer covers it -- `node[j].zone = s.zone` -- and that silently invalidates work:
// adj[j] was computed from the smaller zone, so j has to be re-expanded and any backward
// value already computed for j refers to a zone that no longer exists. Append-only refuses
// the enlargement and stores the newcomer as a separate node. Every stored node is then
// immutable from birth, which is exactly the property "start backward work only on
// vertices known not to change" needs -- and it is why D1 cannot be implemented WITHOUT
// D2's append-only rule: with in-place enlargement, "this node will never change" is not
// establishable before the forward phase terminates. The price is the antichain: the
// bucket keeps zones the published version would have collapsed.
int Otf::intern(SymState&& s) {
  intern_grew = false;
  std::string bkey = disc_key_bytes(s.locs, s.ints);
  // CELL TAG. split_by_atoms has already made every zone that reaches here atom-HOMOGENEOUS,
  // so `Z subseteq phi` is decided by one canonical DBM entry and the tag is well defined.
  //
  // WHICH TAG RULE. The tag is what stops a_LU covering from dropping a deadlock: if the
  // covering zone Z satisfies an escape atom phi and the covered Z' does not, the simulator
  // escapes where the simulated one is stuck, and Dead is lost. Spelt out: nu' in Z simulates
  // nu in Z', and Dead(q) = Inv(q) /\ AND_e !Esc_e with each Esc_e a CONJUNCTION of atoms, so
  // nu |= Dead needs, per e, some atom of Esc_e that nu falsifies to also be falsified by nu'.
  // The obligation is therefore the IMPLICATION `Z |= phi  =>  Z' |= phi`, in that direction
  // only -- equivalently tag(Z) & ~tag(Z') == 0. It is strictly coarser than same-cell
  // equality, which needlessly refuses the (Z disjoint from phi, Z' contained in phi) pair.
  //
  // The two REWRITING paths still need equality, and that is not a detail: in-place
  // enlargement and the convex-union merge both replace a stored zone, and a zone assembled
  // out of two different cells would STRADDLE phi, destroying the homogeneity the tag is read
  // from. Guarded below.
  //
  // SPLIT AT COMPARISON. split_by_atoms emits the zone whole, so `tag` is the In-mask,
  // `tag_out` the Out-mask, and everything a straddle makes undecidable is deferred to
  // Otf::cover_rec.
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
    // a_M MODE: collapse the two bound arrays to M = max(L, U) per clock.
    //
    // With L = U = M the three clauses of the LU preorder -- nu'(x)=nu(x), or
    // L(x) < nu'(x) < nu(x), or U(x) < nu(x) < nu'(x) -- collapse to "equal, or both above
    // M(x)", which is SYMMETRIC. A symmetric simulation is a bisimulation, and deadlock IS
    // preserved by bisimulation. So the same dbm_is_alu_le kernel, fed collapsed bounds,
    // is a deadlock-sound covering test and needs no gate at all.
    //
    // Behrmann-Bouyer-Larsen-Pelanek, STTT 8(3):204-215 (2006): Lemma 1 (=_M is a
    // bisimulation), Lemma 3 (<=_LU is only a simulation), and the unnumbered prose in the
    // "Simulation & Bisimulation" paragraph of section 3 stating that Uppaal's deadlock
    // properties are not preserved by simulation but are by bisimulation. This is also what
    // UPPAAL itself does: it falls back from LU to LOCAL MAX-CONSTANTS for deadlock and
    // liveness queries, not to exact zones.
    // Per-location LU bounds must carry the non-diagonal atoms' constants too; see setup().
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
  // THE REWRITING PATHS UNDER SPLIT-AT-COMPARISON. The equality guard exists only to keep the
  // stored zone HOMOGENEOUS, which is not a property this mode has or wants. What each path
  // actually owes is:
  //
  //   in-place enlargement -- the stored zone is REPLACED by the newcomer, which under a_LU is
  //     not a superset of it, so everything already routed to j must still be covered. That is
  //     transitivity of the covering relation, and the relation IS transitive (<=_LU is a
  //     preorder and `nu'|=phi => nu|=phi` composes), so the ordinary three-valued test in the
  //     opposite direction is exactly and only what is owed.
  //
  //   convex-union merge -- try_union succeeds only when the union is EXACTLY the union, so the
  //     stored zone afterwards is a set SUPERSET of both operands. Growing the right-hand side
  //     of an exists-witness can never invalidate a witness, and the newcomer is covered by the
  //     result via the identity witness. So the merge owes NOTHING here: no tag condition at
  //     all, and it becomes strictly more powerful than in the homogeneous mode, where a
  //     cross-cell union was refused outright.
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
  // EXACT CONVEX-UNION MERGE.
  //
  // The liveness graph inherited "no merge" from leads_to_net, where it is opt-in and was
  // measured a net loss -- but that measurement was made on an EXACT-ZONE graph, not on this
  // aLU-subsumed one, and the reachability engine has the merge ON by default precisely
  // because it collapses antichains. grc4 stores 14.5 zones per discrete key here against
  // 1.00-1.44 on every other family, which is exactly the shape of an antichain full of zones
  // whose union happens to be convex.
  //
  // dbm_try_union succeeds ONLY when the union of the two zones really is a single zone, so
  // the stored zone afterwards denotes exactly the same set of valuations as the two it
  // replaced: no point is added and none is removed. That is what makes it usable in the
  // UNDER-approximating lane as well -- unlike the hull, it is not an over-approximation, so
  // it cannot invent a certificate.
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
