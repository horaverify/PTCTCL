// bisim.cpp — zone-based timed (bi)simulation / (weak) simulation engine.
//
// Unified greatest-fixpoint over the disjoint-clock product X = X_A ⊎ X_B. The
// relation R[(lA,lB)] is a federation of zones; we start from the full relation and
// repeatedly remove valuations that fail delay-stability or action-stability, until a
// fixpoint. The initial states are related iff the origin survives in R[(l0A,l0B)].
// Parameterised by symmetric?(bi vs sim) and strong-vs-weak(τ over a chosen alphabet).
// Implements the vetted research blueprint (Čerāns / Weise–Lenzkes / Tripakis–Yovine).
#include "ptctlmc/bisim.hpp"

#include <cstdint>

#include "ptctlmc/dbm.hpp"
#include "ptctlmc/fed.hpp"

namespace ptctlmc {
namespace {

// ---- product-lifted automaton -------------------------------------------------
struct LEdge {
  int src, dst;
  std::vector<Constraint> guard;  // over product clock indices
  std::vector<int> resets;        // product clock indices
  int act;
};

// ---- basic zone constructors --------------------------------------------------
Dbm universe(int nclk) {
  Dbm z(nclk);
  const int dim = z.dim();
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j) z.set(i, j, (i == j) ? kZeroLe : kInf);
  for (int i = 1; i < dim; ++i) z.set(0, i, kZeroLe);  // x_i >= 0
  z.close();
  return z;
}

bool contains_origin(const Dbm& z) {
  const int dim = z.dim();
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j)
      if (z.get(i, j) < kZeroLe) return false;  // origin (all=0) violates a strict/neg bound
  return true;
}
bool contains_origin(const Federation& f) {
  for (const auto& z : f) if (contains_origin(z)) return true;
  return false;
}

// ---- DBM helper ops (blueprint §2) --------------------------------------------
Dbm down(Dbm z) {  // { v | ∃d>=0. v+d ∈ z } : relax every lower bound to x_i >= 0
  for (int i = 1; i < z.dim(); ++i) z.set(0, i, kZeroLe);
  z.close();
  return z;
}
Dbm free_clock(Dbm z, int x) {  // remove all bounds on clock x
  const int dim = z.dim();
  for (int k = 0; k < dim; ++k) if (k != x) z.set(x, k, kInf);
  for (int k = 1; k < dim; ++k) if (k != x) z.set(k, x, kInf);
  z.set(0, x, kZeroLe);
  z.set(x, x, kZeroLe);
  z.close();
  return z;
}
void extrapolate(Dbm& z, const std::vector<std::int32_t>& M) {
  dbm_extrapolate(z.data().data(), z.dim(), M.data());
}
Federation extrapolate_fed(Federation f, const std::vector<std::int32_t>& M) {
  Federation out;
  for (auto& z : f) { extrapolate(z, M); if (!z.empty()) out.push_back(std::move(z)); }
  return out;
}

// ---- federation ops -----------------------------------------------------------
Federation fed_minus(const Federation& F, const Federation& G) {  // F \ G
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
bool shrank(const Federation& oldF, const Federation& newF) {  // did new lose points?
  return !fed_minus(oldF, newF).empty();
}

// ---- edge predecessor (blueprint §3.1) ----------------------------------------
// { v | v ⊨ guard ∧ v[resets:=0] ∈ G }
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
struct Engine {
  int dim;
  int nA, nB;                 // clock counts
  int laA, laB;               // location counts
  std::vector<std::int32_t> M;
  std::vector<LEdge> ea, eb;  // lifted edges (A: clocks unchanged; B: shifted by nA)
  std::set<int> obs;
  bool weak;

  int key(int lA, int lB) const { return lA * laB + lB; }
  bool observable(int act) const { return !weak || obs.count(act) > 0; }

  // TauMap over a fixed other-side location, indexed by the responder's own location.
  using Map = std::vector<Federation>;

  // §3.3 backward τ-closure (zero time) over `side` edges; side==0 -> A, 1 -> B.
  void tau0_pre_close(int side, Map& T) const {
    const std::vector<LEdge>& E = side == 0 ? ea : eb;
    bool grow = true;
    while (grow) {
      grow = false;
      for (const auto& e : E) {
        if (observable(e.act)) continue;  // τ = non-observable
        Federation ex = extrapolate_fed(edge_pre(e, T[e.dst], M), M);
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
          if (f.src == Lo && f.act == e.act) S = fed_union(S, edge_pre(f, Tgt[f.dst], M));
        return S;
      }
      // WEAK observable:  τ0* ; a-edge ; τ0*
      Map T2 = Tgt; tau0_pre_close(O, T2);
      Map A1(loLocs);
      for (const auto& f : OE)
        if (observable(f.act) && f.act == e.act)
          A1[f.src] = fed_union(A1[f.src], edge_pre(f, T2[f.dst], M));
      Map T3 = A1; tau0_pre_close(O, T3);
      return T3[Lo];
    }
    // WEAK, challenger τ: responder answers with τ0* (incl. staying put).
    Map T2 = Tgt; tau0_pre_close(O, T2);
    return T2[Lo];
  }

  // §3.2 strong delay-stability of one federation.
  Federation delay_stable_strong(const Federation& F) const {
    Federation comp = fed_minus(Federation{universe(nA + nB)}, F);  // ¬F
    Federation bad;
    for (const auto& z : comp) bad.push_back(down(z));
    return fed_minus(F, bad);
  }

  bool decide(int l0A, int l0B, bool symmetric) {
    std::vector<Federation> R(static_cast<std::size_t>(laA) * laB);
    const Dbm U = universe(nA + nB);
    for (auto& f : R) f = Federation{U};

    bool changed = true;
    while (changed) {
      changed = false;
      // (1) delay-stability. Pure delay only (down): a τ move is a silent ACTION, matched
      // in action-stability by the other side's own τ — it must NOT unconditionally remove
      // points here (that breaks reflexivity). Holds for strong and weak alike.
      for (auto& f : R) {
        Federation nf = delay_stable_strong(f);
        if (shrank(f, nf)) { f = std::move(nf); changed = true; }
      }
      // (2) action-stability — A challenges (responder B)
      for (const auto& e : ea)
        for (int lB = 0; lB < laB; ++lB) {
          Federation en{constrain_guard(e.guard)};
          Federation bad = fed_minus(en, resp_set(e, 0, R, lB));
          const int p = key(e.src, lB);
          Federation nf = fed_minus(R[p], bad);
          if (shrank(R[p], nf)) { R[p] = std::move(nf); changed = true; }
        }
      // B challenges (responder A) — only for bisimulation
      if (symmetric)
        for (const auto& e : eb)
          for (int lA = 0; lA < laA; ++lA) {
            Federation en{constrain_guard(e.guard)};
            Federation bad = fed_minus(en, resp_set(e, 1, R, lA));
            const int p = key(lA, e.src);
            Federation nf = fed_minus(R[p], bad);
            if (shrank(R[p], nf)) { R[p] = std::move(nf); changed = true; }
          }
    }
    return contains_origin(R[key(l0A, l0B)]);
  }

  Dbm constrain_guard(const std::vector<Constraint>& g) const {
    Dbm z = universe(nA + nB);
    for (const auto& c : g) z.constrain(c.i, c.j, c.bound);
    return z;
  }
};

LEdge lift(const BEdge& e, int shift) {
  LEdge l;
  l.src = e.src; l.dst = e.dst; l.act = e.act;
  for (const auto& c : e.guard)
    l.guard.push_back({c.i == 0 ? 0 : c.i + shift, c.j == 0 ? 0 : c.j + shift, c.bound});
  for (int r : e.resets) l.resets.push_back(r + shift);
  return l;
}

}  // namespace

bool decide_bisim(const BAut& A, const BAut& B, BisimKind kind, const std::set<int>& obs) {
  Engine E;
  E.nA = A.nclocks; E.nB = B.nclocks; E.dim = A.nclocks + B.nclocks + 1;
  E.laA = A.nlocs; E.laB = B.nlocs;
  E.obs = obs;
  E.weak = (kind == BisimKind::WeakBisim || kind == BisimKind::WeakSim);
  const bool symmetric = (kind == BisimKind::StrongBisim || kind == BisimKind::WeakBisim);
  for (const auto& e : A.edges) E.ea.push_back(lift(e, 0));
  for (const auto& e : B.edges) E.eb.push_back(lift(e, A.nclocks));
  // max constants over BOTH automata (per product clock).
  E.M.assign(E.dim, 0);
  auto scan = [&](const std::vector<LEdge>& es) {
    for (const auto& e : es)
      for (const auto& c : e.guard) {
        const int v = std::abs(raw_bound(c.bound));
        if (c.i > 0 && v > E.M[c.i]) E.M[c.i] = v;
        if (c.j > 0 && v > E.M[c.j]) E.M[c.j] = v;
      }
  };
  scan(E.ea); scan(E.eb);
  return E.decide(A.initial, B.initial, symmetric);
}

}  // namespace ptctlmc
