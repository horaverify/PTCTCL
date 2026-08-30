/// @file bisim.hpp
/// @brief Timed (bi)simulation and weak simulation between two timed automata.
///
/// Decides, over the disjoint-clock product of the two automata, whether A and B are timed
/// bisimilar, or whether A is timed-simulated by B, in the strong or the weak sense — weak taking
/// τ over a chosen observable alphabet. The decision procedure is greatest-fixpoint partition
/// refinement over federations of zones.
///
/// @par Semantics
/// A pair (l, v) is a state only when `v ⊨ Inv(l)`. A delay d is available iff `v + d ⊨ Inv(l)`;
/// the invariant is convex, so testing the endpoint covers the whole trajectory. An edge is
/// enabled only when the *post-reset* valuation satisfies the **target** invariant.
///
/// A delay challenge must be matched by an **equal** delay on the other side, in the weak case as
/// well as the strong one. An invariant that stops time on the responder's side therefore kills
/// the pair. Weakness relaxes only the action clause, to `τ* a τ*` with the τ steps taken at zero
/// delay.
///
/// @pre Both automata are diagonal-free. Location invariants are supported.
#pragma once

#include <set>
#include <vector>

#include "hora/zone.hpp"
#include "hdbm/fed.hpp"  // Federation
#include "hora/automaton.hpp"   // Constraint

namespace hora {

// A minimal labelled timed automaton for (bi)simulation. Clocks are 1..nclocks;
// guards are diagonal-free constraints over those clocks (index 0 = reference).
struct BEdge {
  int src = 0, dst = 0;
  std::vector<Constraint> guard;
  std::vector<int> resets;  // clock indices 1..nclocks
  int act = 0;              // action label (integer)
};
struct BAut {
  int nclocks = 1;
  int nlocs = 1;
  int initial = 0;
  std::vector<BEdge> edges;
  // Location invariants. Either empty (no invariant anywhere — the historical behaviour) or
  // of size nlocs; invariants[l] is a conjunction of diagonal-free constraints that every
  // valuation at l must satisfy. A short vector is padded with "no invariant".
  std::vector<std::vector<Constraint>> invariants;
};

enum class BisimKind { StrongBisim, StrongSim, WeakBisim, WeakSim };

// Decide the relation. For *Sim, returns whether A <= B (B simulates A). `obs` lists
// observable action labels; used only for the Weak* kinds (Strong* treats every label
// as observable). Both automata start in `initial` with all clocks zero.
bool decide_bisim(const BAut& A, const BAut& B, BisimKind kind, const std::set<int>& obs = {});

// The relation itself — the partition the refinement already computes, which a quotient needs.
// Indexed [lA * B.nlocs + lB]; a product valuation (v_A, v_B) is in the relation for (lA,lB)
// iff it lies in the returned federation (clocks 1..A.nclocks are A's, the next B.nclocks are
// B's). `*related`, when non-null, receives what decide_bisim would have returned.
std::vector<Federation> bisim_relation(const BAut& A, const BAut& B, BisimKind kind,
                                       const std::set<int>& obs = {}, bool* related = nullptr);

}  // namespace hora
