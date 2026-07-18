// bisim.hpp — timed (bi)simulation / (weak) simulation between two timed automata.
//
// Decides, over the disjoint-clock product X_A ⊎ X_B, whether A and B are timed
// bisimilar / A is timed-simulated by B, in the strong or weak (τ over a chosen
// observable alphabet) sense. Greatest-fixpoint partition refinement on federations
// of zones; see the research blueprint. Diagonal-free, no invariants (core).
#pragma once

#include <set>
#include <vector>

#include "ptctlmc/ta.hpp"  // Constraint

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
};

enum class BisimKind { StrongBisim, StrongSim, WeakBisim, WeakSim };

// Decide the relation. For *Sim, returns whether A <= B (B simulates A). `obs` lists
// observable action labels; used only for the Weak* kinds (Strong* treats every label
// as observable). Both automata start in `initial` with all clocks zero.
bool decide_bisim(const BAut& A, const BAut& B, BisimKind kind, const std::set<int>& obs = {});

}  // namespace ptctlmc
