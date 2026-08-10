// bisim.hpp — timed (bi)simulation / (weak) simulation between two timed automata.
//
// Decides, over the disjoint-clock product X_A ⊎ X_B, whether A and B are timed
// bisimilar / A is timed-simulated by B, in the strong or weak (τ over a chosen
// observable alphabet) sense. Greatest-fixpoint partition refinement on federations
// of zones; see the research blueprint. Diagonal-free; location invariants supported.
//
// Semantics (docs/superpowers/specs/weaksim-quotient.md §A.1): (l,v) is a state only when
// v ⊨ Inv(l); a delay d is available iff v+d ⊨ Inv(l) (convexity of the invariant makes the
// endpoint test cover the trajectory); an edge is enabled only when the post-reset valuation
// satisfies the TARGET invariant. A delay challenge is matched by an EQUAL delay on the other
// side, in the weak case too — so an invariant that stops time on the responder's side kills
// the pair. Weak relaxes only the action clause, to τ* a τ* with the τs at zero delay.
#pragma once

#include <set>
#include <vector>

#include "ptctlmc/fed.hpp"  // Federation
#include "ptctlmc/ta.hpp"   // Constraint

namespace ptctlmc {

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

}  // namespace ptctlmc
