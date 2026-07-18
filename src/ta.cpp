#include "ptctlmc/ta.hpp"

#include <algorithm>
#include <cstdlib>

namespace ptctlmc {

void TimedAutomaton::compute_max_constants() {
  max_const.assign(nclocks + 1, 0);
  lbound.assign(nclocks + 1, 0);
  ubound.assign(nclocks + 1, 0);
  auto scan = [&](const std::vector<Constraint>& cs) {
    for (const auto& c : cs) {
      const int v = raw_bound(c.bound);
      if (c.i > 0 && c.j == 0) {                 // x_i <| v : upper bound on x_i
        if (v > ubound[c.i]) ubound[c.i] = v;
      } else if (c.i == 0 && c.j > 0) {          // -x_j <| v : x_j >| -v : lower bound on x_j
        const int lb = -v;
        if (lb > lbound[c.j]) lbound[c.j] = lb;
      } else if (c.i > 0 && c.j > 0) {           // diagonal x_i - x_j <| v : conservative
        const int a = std::abs(v);
        if (a > ubound[c.i]) ubound[c.i] = a;
        if (a > ubound[c.j]) ubound[c.j] = a;
        if (a > lbound[c.i]) lbound[c.i] = a;
        if (a > lbound[c.j]) lbound[c.j] = a;
      }
    }
  };
  for (const auto& l : locations) scan(l.invariant);
  for (const auto& e : edges) scan(e.guard);
  for (int c = 0; c <= nclocks; ++c) max_const[c] = std::max(lbound[c], ubound[c]);
  max_const[0] = lbound[0] = ubound[0] = 0;
}

std::vector<int> TimedAutomaton::edges_from(int loc) const {
  std::vector<int> out;
  for (int e = 0; e < static_cast<int>(edges.size()); ++e)
    if (edges[e].src == loc) out.push_back(e);
  return out;
}

}  // namespace ptctlmc
