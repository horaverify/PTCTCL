// fed.hpp — federations (finite unions of zones) and the zone set-difference the
// timed-bisimulation refinement needs. A federation is just a std::vector<Dbm>; the
// zones in a subtraction result are pairwise disjoint but that is not required in
// general. These operate over a shared dimension (product clock space X_A ⊎ X_B).
#pragma once

#include <vector>

#include "ptctlmc/dbm.hpp"
#include "ptctlmc/ta.hpp"  // Constraint, negate

namespace ptctlmc {

using Federation = std::vector<Dbm>;

// a \ b as a set of DISJOINT zones. Standard construction: peel a by b's constraints —
// for each constraint c of b, the slice of the remainder that VIOLATES c is outside b,
// and we keep intersecting the remainder with c to move on. Empty slices are dropped.
inline Federation zone_subtract(const Dbm& a, const Dbm& b) {
  Federation out;
  if (a.empty()) return out;
  if (b.empty()) { out.push_back(a); return out; }
  const int dim = a.dim();
  Dbm rem = a;  // part of `a` still inside `b` w.r.t. constraints processed so far
  for (int i = 0; i < dim && !rem.empty(); ++i)
    for (int j = 0; j < dim; ++j) {
      if (i == j) continue;
      const raw_t bij = b.get(i, j);
      if (bij >= kInf) continue;                 // b unconstrained on (i,j): nothing to peel
      if (a.get(i, j) <= bij && rem.get(i, j) <= bij)
        continue;                                // a already at least as tight -> no slice here
      Dbm slice = rem;                           // rem ∩ ¬(x_i - x_j <| bij)
      const Constraint nc = negate(Constraint{i, j, bij});
      if (slice.constrain(nc.i, nc.j, nc.bound)) out.push_back(std::move(slice));
      if (!rem.constrain(i, j, bij)) { rem = Dbm(); return out; }  // remainder emptied
    }
  return out;
}

// a \ (union of fed). Repeatedly subtract each zone of `fed` from every piece.
inline Federation fed_subtract(const Dbm& a, const Federation& fed) {
  Federation cur{a};
  for (const auto& b : fed) {
    Federation next;
    for (const auto& z : cur) {
      Federation d = zone_subtract(z, b);
      for (auto& p : d) next.push_back(std::move(p));
    }
    cur.swap(next);
    if (cur.empty()) break;
  }
  return cur;
}

// Is `a` covered by the union of `fed`?  (a \ fed is empty)
inline bool fed_covers(const Federation& fed, const Dbm& a) { return fed_subtract(a, fed).empty(); }

}  // namespace ptctlmc
