/// @file bisim.cpp
/// @brief Zone-based timed (bi)simulation and weak simulation engine.
///
/// One greatest fixpoint over the disjoint-clock product of the two automata. The relation at a
/// location pair is a federation of zones; the iteration starts from the full relation and
/// repeatedly removes valuations failing delay-stability or action-stability until nothing more
/// is removed. The two automata are related iff the origin survives at the initial location pair.
///
/// The same iteration serves all four modes, parameterised by whether the relation is required to
/// be symmetric — bisimulation against simulation — and by strong against weak, the latter taking
/// τ over a chosen observable alphabet.
///
/// @note Decidability of timed bisimulation, and the region/zone product construction the
///       fixpoint runs over, are due to Čerāns (1992); the zone-based formulation follows
///       Weise and Lenzkes (1997) and Tripakis and Yovine (2001).
/// @note Starting from the full relation and shrinking is what makes this a *greatest* fixpoint:
///       the relation is the largest one that is stable, so an early stop yields a relation that
///       is too large, never too small.
/// @see bisim.hpp for the precise semantics of states, delays and edge enabling.
#include "ptctlmc/bisim.hpp"

#include <cstdint>

#include "ptctlmc/dbm.hpp"
#include "ptctlmc/fed.hpp"

namespace ptctlmc {
namespace {

// ---- product-lifted automaton -------------------------------------------------
/// @brief An edge of one automaton lifted into the product clock space.
///
/// @invariant Guards and resets name *product* clock indices. Automaton A keeps its own indices;
///            automaton B's are shifted by A's clock count, which is what makes the two clock sets
///            disjoint.
struct LEdge {
  int src, dst;                   ///< source and target location, in that automaton's numbering
  std::vector<Constraint> guard;  ///< guard, over product clock indices
  std::vector<int> resets;        ///< reset clocks, as product clock indices
  int act;                        ///< action label
};

// ---- basic zone constructors --------------------------------------------------
/// @brief The zone containing every non-negative valuation of @p nclk clocks.
/// @param nclk number of clocks
/// @return the universe zone, closed
/// @note This is the starting point of the greatest fixpoint, not the zero zone.
Dbm universe(int nclk) {
  Dbm z(nclk);
  const int dim = z.dim();
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j) z.set(i, j, (i == j) ? kZeroLe : kInf);
  for (int i = 1; i < dim; ++i) z.set(0, i, kZeroLe);  // x_i >= 0
  z.close();
  return z;
}

/// @brief Does this zone contain the all-zero valuation?
/// @note The initial states of both automata have every clock at zero, so this is the test that
///       reads the verdict off the converged relation.
bool contains_origin(const Dbm& z) {
  const int dim = z.dim();
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j)
      if (z.get(i, j) < kZeroLe) return false;  // origin (all=0) violates a strict/neg bound
  return true;
}
/// @brief Does any zone of this federation contain the all-zero valuation?
bool contains_origin(const Federation& f) {
  for (const auto& z : f) if (contains_origin(z)) return true;
  return false;
}

// ---- DBM helper ops (blueprint §2) --------------------------------------------
/// @brief Time-past: the valuations from which some delay reaches @p z.
/// @return `{ v | exists d >= 0 with v + d in z }`, computed by relaxing every lower bound
Dbm down(Dbm z) {
  for (int i = 1; i < z.dim(); ++i) z.set(0, i, kZeroLe);
  z.close();
  return z;
}
/// @brief Remove every bound on one clock, leaving it unconstrained.
/// @param z the zone
/// @param x the product clock index to free
/// @note Used to invert a reset: the predecessors of a reset are the valuations agreeing with the
///       target on every other clock and arbitrary on the reset one.
Dbm free_clock(Dbm z, int x) {
  const int dim = z.dim();
  for (int k = 0; k < dim; ++k) if (k != x) z.set(x, k, kInf);
  for (int k = 1; k < dim; ++k) if (k != x) z.set(k, x, kInf);
  z.set(0, x, kZeroLe);
  z.set(x, x, kZeroLe);
  z.close();
  return z;
}
/// @brief Apply max-constant extrapolation, which is what makes the fixpoint terminate.
/// @param[in,out] z the zone
/// @param M per-clock maximum constants over the product
void extrapolate(Dbm& z, const std::vector<std::int32_t>& M) {
  dbm_extrapolate(z.data().data(), z.dim(), M.data());
}
/// @brief Extrapolate every zone of a federation.
Federation extrapolate_fed(Federation f, const std::vector<std::int32_t>& M) {
  Federation out;
  for (auto& z : f) { extrapolate(z, M); if (!z.empty()) out.push_back(std::move(z)); }
  return out;
}

// ---- federation ops -----------------------------------------------------------
/// @brief Federation difference, `F \ G`.
/// @note The result is generally non-convex, which is why the relation is carried as a federation
///       rather than as a single zone.
Federation fed_minus(const Federation& F, const Federation& G) {
  Federation out;
  for (const auto& z : F) { Federation d = fed_subtract(z, G); for (auto& p : d) out.push_back(std::move(p)); }
  return out;
}
// fed_union comes from HDBM's fed.hpp. This file used to carry its own copy ("concat + light
// subsumption reduce"), which became an ambiguous overload once HDBM published the federation
// set algebra. They compute the same POINT SET; HDBM's is the better representation, because its
// fed_add also drops members that the incoming zone subsumes (the local one only skipped the
// incoming zone when an existing member already covered it, so subsumed members accumulated) and
// it screens empty zones explicitly. Nothing here depends on the representation: the fixpoint
// tests below go through shrank()/fed_minus, which compare point sets, so a tighter antichain
// changes only the amount of work.
/// @brief Did the refinement step remove any valuation?
/// @return @c true iff @p newF is strictly smaller than @p oldF
/// @note This is the fixpoint's termination test. The iteration is monotone decreasing, so a step
///       that removes nothing means the relation is stable.
bool shrank(const Federation& oldF, const Federation& newF) {
  return !fed_minus(oldF, newF).empty();
}

// { v | v[resets:=0] ∈ Z }: pin the reset clocks to 0, intersect, then free them again.
// With Z the universe this is the universe, which is why an absent invariant costs nothing.
/// @brief Predecessor of a zone under a set of resets.
/// @param Z       the target zone
/// @param resets  the product clocks the edge resets
/// @return the valuations that land in @p Z after the resets
Dbm reset_pre(Dbm Z, const std::vector<int>& resets) {
  for (int x : resets) { Z.constrain(x, 0, kZeroLe); Z.constrain(0, x, kZeroLe); }
  Z.close();
  if (Z.empty()) return Z;
  for (int x : resets) Z = free_clock(Z, x);
  return Z;
}

// ---- edge predecessor (blueprint §3.1) ----------------------------------------
// { v | v ⊨ guard ∧ v[resets:=0] ∈ G }
/// @brief Predecessor of a federation under one lifted edge.
///
/// Pins the reset clocks to zero, frees them to invert the reset, then conjoins the guard —
/// which is evaluated in the source, before the reset, and so must be applied last.
///
/// @param e the lifted edge
/// @param G the target federation
/// @param M per-clock maximum constants, for the extrapolation
/// @return the source valuations from which @p e fires into @p G
Federation edge_pre(const LEdge& e, const Federation& G, const std::vector<std::int32_t>& M) {
  Federation out;
  for (const Dbm& Zt : G) {
    Dbm Z = Zt;
    for (int x : e.resets) { Z.constrain(x, 0, kZeroLe); Z.constrain(0, x, kZeroLe); }  // pin x=0
    Z.close();
    if (Z.empty()) continue;
    for (int x : e.resets) Z = free_clock(Z, x);  // un-reset (predecessor of the reset)
    bool ok = true;
    for (const auto& c : e.guard) if (!Z.constrain(c.i, c.j, c.bound)) { ok = false; break; }
    if (!ok) continue;
    Z.close();
    if (!Z.empty()) { extrapolate(Z, M); out.push_back(std::move(Z)); }
  }
  return out;
}

// ---- engine -------------------------------------------------------------------
/// @brief The product automaton and the state of the refinement over it.
struct Engine {
  int dim;                      ///< product zone dimension
  int nA, nB;                   ///< clock counts of A and of B
  int laA, laB;                 ///< location counts of A and of B
  std::vector<std::int32_t> M;  ///< per-clock maximum constants over the product
  std::vector<LEdge> ea, eb;    ///< lifted edges; A keeps its indices, B's are shifted by nA
  std::vector<Dbm> invA, invB;  ///< location invariants, lifted into the product space
  std::set<int> obs;            ///< observable alphabet; only consulted when @c weak
  bool weak;                    ///< treat non-observable actions as τ

  /// @brief Index of a location pair in the relation array.
  int key(int lA, int lB) const { return lA * laB + lB; }
  /// @brief Must this action be matched by an equally-labelled action?
  /// @return always @c true in the strong case; in the weak case only for observable actions
  bool observable(int act) const { return !weak || obs.count(act) > 0; }
  const std::vector<Dbm>& inv(int side) const { return side == 0 ? invA : invB; }

  // TauMap over a fixed other-side location, indexed by the responder's own location.
  using Map = std::vector<Federation>;

  // §3.3 backward τ-closure (zero time) over `side` edges; side==0 -> A, 1 -> B.
  void tau0_pre_close(int side, Map& T) const {
    const std::vector<LEdge>& E = side == 0 ? ea : eb;
    const std::vector<Dbm>& IV = inv(side);
    bool grow = true;
    while (grow) {
      grow = false;
      for (const auto& e : E) {
        if (observable(e.act)) continue;  // τ = non-observable
        // The source invariant is re-imposed AFTER extrapolation: extrapolation may only
        // widen, and a τ chain may not pass through a location whose invariant fails here.
        Federation ex = fed_intersect(extrapolate_fed(edge_pre(e, T[e.dst], M), M), IV[e.src]);
        Federation nu = fed_union(T[e.src], ex);
        if (!fed_minus(nu, T[e.src]).empty()) { T[e.src] = std::move(nu); grow = true; }
      }
    }
  }

  // §3.5 responder matchable set for a challenger edge e on side C (0=A,1=B).
  // `R` is the current relation; `Lo` responder's location. Returns valuations where
  // the challenger can fire e and the responder can match to a still-related successor.
  Federation resp_set(const LEdge& e, int C, const std::vector<Federation>& R, int Lo) const {
    const int O = 1 - C;
    const std::vector<LEdge>& OE = O == 0 ? ea : eb;
    const std::vector<Dbm>& OIV = inv(O);
    const int loLocs = O == 0 ? laA : laB;
    // Tgt[lo''] = challenger leg applied to R[(lC', lo'')]
    Map Tgt(loLocs);
    for (int lo = 0; lo < loLocs; ++lo) {
      const int kk = (C == 0) ? key(e.dst, lo) : key(lo, e.dst);
      Tgt[lo] = edge_pre(e, R[kk], M);
    }
    if (observable(e.act)) {
      if (!weak) {
        Federation S;
        for (const auto& f : OE)
          if (f.src == Lo && f.act == e.act)
            S = fed_union(S, fed_intersect(edge_pre(f, Tgt[f.dst], M), OIV[f.src]));
        return S;
      }
      // WEAK observable:  τ0* ; a-edge ; τ0*
      Map T2 = Tgt; tau0_pre_close(O, T2);
      Map A1(loLocs);
      for (const auto& f : OE)
        if (observable(f.act) && f.act == e.act)
          A1[f.src] =
              fed_union(A1[f.src], fed_intersect(edge_pre(f, T2[f.dst], M), OIV[f.src]));
      Map T3 = A1; tau0_pre_close(O, T3);
      return T3[Lo];
    }
    // WEAK, challenger τ: responder answers with τ0* (incl. staying put).
    Map T2 = Tgt; tau0_pre_close(O, T2);
    return T2[Lo];
  }

  // §3.2 delay-stability of one federation. `avail` is the set of product valuations the
  // CHALLENGER's own invariant lets it delay into: InvA[lA] for a simulation, InvA ∪ InvB for
  // a bisimulation (either side may issue the delay challenge). Because R ⊆ InvA ∩ InvB by
  // construction, `avail \ R` already contains both failure modes — "the delay leaves the
  // relation" and "the responder's invariant forbids that much time" — so one subtraction
  // does both. With universal invariants this is exactly the old `universe \ F`.
  Federation delay_stable(const Federation& F, const Federation& avail) const {
    Federation comp = fed_minus(avail, F);
    Federation bad;
    for (const auto& z : comp) bad.push_back(down(z));
    return fed_minus(F, bad);
  }

  // The challenger's true enabled set for edge `e` on side C with the responder at `Lo`:
  // guard ∧ own invariant ∧ responder's invariant ∧ (post-reset ⊨ own target invariant).
  // Overstating this would delete pairs that never received a challenge.
  Dbm enabled_zone(const LEdge& e, int C, int Lo) const {
    Dbm z = constrain_guard(e.guard);
    z.intersect(inv(C)[e.src]);
    z.intersect(inv(1 - C)[Lo]);
    if (z.empty()) return z;
    z.intersect(reset_pre(inv(C)[e.dst], e.resets));
    return z;
  }

  std::vector<Federation> run(bool symmetric) {
    std::vector<Federation> R(static_cast<std::size_t>(laA) * laB);
    for (int lA = 0; lA < laA; ++lA)
      for (int lB = 0; lB < laB; ++lB) {
        Dbm z = invA[lA];
        z.intersect(invB[lB]);
        if (!z.empty()) R[key(lA, lB)] = Federation{z};
      }

    bool changed = true;
    while (changed) {
      changed = false;
      // (1) delay-stability. Pure delay only (down): a τ move is a silent ACTION, matched
      // in action-stability by the other side's own τ — it must NOT unconditionally remove
      // points here (that breaks reflexivity). Holds for strong and weak alike.
      for (int lA = 0; lA < laA; ++lA)
        for (int lB = 0; lB < laB; ++lB) {
          Federation& f = R[key(lA, lB)];
          if (f.empty()) continue;
          Federation avail{invA[lA]};
          if (symmetric) fed_add(avail, invB[lB]);
          Federation nf = delay_stable(f, avail);
          if (shrank(f, nf)) { f = std::move(nf); changed = true; }
        }
      // (2) action-stability — A challenges (responder B)
      for (const auto& e : ea)
        for (int lB = 0; lB < laB; ++lB) {
          const int p = key(e.src, lB);
          if (R[p].empty()) continue;
          Dbm ez = enabled_zone(e, 0, lB);
          if (ez.empty()) continue;
          Federation bad = fed_minus(Federation{ez}, resp_set(e, 0, R, lB));
          Federation nf = fed_minus(R[p], bad);
          if (shrank(R[p], nf)) { R[p] = std::move(nf); changed = true; }
        }
      // B challenges (responder A) — only for bisimulation
      if (symmetric)
        for (const auto& e : eb)
          for (int lA = 0; lA < laA; ++lA) {
            const int p = key(lA, e.src);
            if (R[p].empty()) continue;
            Dbm ez = enabled_zone(e, 1, lA);
            if (ez.empty()) continue;
            Federation bad = fed_minus(Federation{ez}, resp_set(e, 1, R, lA));
            Federation nf = fed_minus(R[p], bad);
            if (shrank(R[p], nf)) { R[p] = std::move(nf); changed = true; }
          }
    }
    return R;
  }

  Dbm constrain_guard(const std::vector<Constraint>& g) const {
    Dbm z = universe(nA + nB);
    for (const auto& c : g) z.constrain(c.i, c.j, c.bound);  // incremental close: stays canonical
    return z;
  }
};

/// @brief Lift one edge into the product clock space.
/// @param e     the edge
/// @param shift added to every clock index; 0 for automaton A, A's clock count for B
/// @note Index 0 is the reference clock and is shared, so it is never shifted.
LEdge lift(const BEdge& e, int shift) {
  LEdge l;
  l.src = e.src; l.dst = e.dst; l.act = e.act;
  for (const auto& c : e.guard)
    l.guard.push_back({c.i == 0 ? 0 : c.i + shift, c.j == 0 ? 0 : c.j + shift, c.bound});
  for (int r : e.resets) l.resets.push_back(r + shift);
  return l;
}

/// @brief Lift every location invariant of one automaton into the product space.
///
/// Each invariant constrains that side's clocks and leaves the other side's unconstrained.
///
/// @param A          the automaton
/// @param shift      clock index shift for this side
/// @param prodclocks total clocks in the product
/// @param[out] lifted if non-null, also receives the shifted constraints themselves
/// @return one lifted invariant zone per location
/// @note An absent or empty invariant yields the universe, so every rule downstream degenerates
///       correctly for an automaton without invariants.
std::vector<Dbm> lift_invariants(const BAut& A, int shift, int prodclocks,
                                 std::vector<std::vector<Constraint>>* lifted) {
  std::vector<Dbm> out;
  out.reserve(A.nlocs);
  for (int l = 0; l < A.nlocs; ++l) {
    Dbm z = universe(prodclocks);
    std::vector<Constraint> cs;
    if (l < static_cast<int>(A.invariants.size()))
      for (const auto& c : A.invariants[l]) {
        const Constraint s{c.i == 0 ? 0 : c.i + shift, c.j == 0 ? 0 : c.j + shift, c.bound};
        cs.push_back(s);
        z.constrain(s.i, s.j, s.bound);
      }
    lifted->push_back(std::move(cs));
    out.push_back(std::move(z));
  }
  return out;
}

/// @brief Build the product engine for two automata.
/// @param A    the first automaton
/// @param B    the second
/// @param kind which relation to decide
/// @return the engine, with edges and invariants lifted and the max constants computed
Engine build(const BAut& A, const BAut& B, BisimKind kind) {
  Engine E;
  E.nA = A.nclocks; E.nB = B.nclocks; E.dim = A.nclocks + B.nclocks + 1;
  E.laA = A.nlocs; E.laB = B.nlocs;
  E.weak = (kind == BisimKind::WeakBisim || kind == BisimKind::WeakSim);
  for (const auto& e : A.edges) E.ea.push_back(lift(e, 0));
  for (const auto& e : B.edges) E.eb.push_back(lift(e, A.nclocks));
  std::vector<std::vector<Constraint>> ia, ib;
  E.invA = lift_invariants(A, 0, A.nclocks + B.nclocks, &ia);
  E.invB = lift_invariants(B, A.nclocks, A.nclocks + B.nclocks, &ib);
  // max constants over BOTH automata (per product clock). Invariant constants count: Extra_M
  // only widens bounds strictly above le(M[i]), so an invariant x<=k survives iff M[x] >= k.
  E.M.assign(E.dim, 0);
  auto bump = [&](const Constraint& c) {
    const int v = std::abs(raw_bound(c.bound));
    if (c.i > 0 && v > E.M[c.i]) E.M[c.i] = v;
    if (c.j > 0 && v > E.M[c.j]) E.M[c.j] = v;
  };
  for (const auto* es : {&E.ea, &E.eb})
    for (const auto& e : *es)
      for (const auto& c : e.guard) bump(c);
  for (const auto* iv : {&ia, &ib})
    for (const auto& cs : *iv)
      for (const auto& c : cs) bump(c);
  return E;
}

}  // namespace

/// @brief Decide whether two timed automata are related.
/// @param A    the first automaton
/// @param B    the second
/// @param kind which relation to decide
/// @param obs  the observable alphabet; consulted only for the weak variants
/// @return @c true iff the initial states are related, i.e. the origin survives at the initial
///         location pair
bool decide_bisim(const BAut& A, const BAut& B, BisimKind kind, const std::set<int>& obs) {
  bool related = false;
  bisim_relation(A, B, kind, obs, &related);
  return related;
}

/// @brief Compute the full converged relation, not only the verdict at the initial pair.
/// @param A    the first automaton
/// @param B    the second
/// @param kind which relation to compute
/// @return one federation per location pair, indexed as ::Engine::key does
/// @note Exposed for the quotient construction and for tests, which need to inspect the relation
///       at pairs other than the initial one.
std::vector<Federation> bisim_relation(const BAut& A, const BAut& B, BisimKind kind,
                                       const std::set<int>& obs, bool* related) {
  Engine E = build(A, B, kind);
  E.obs = obs;
  const bool symmetric = (kind == BisimKind::StrongBisim || kind == BisimKind::WeakBisim);
  std::vector<Federation> R = E.run(symmetric);
  if (related) *related = contains_origin(R[E.key(A.initial, B.initial)]);
  return R;
}

}  // namespace ptctlmc
