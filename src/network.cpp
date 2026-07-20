#include "ptctlmc/network.hpp"

#include <algorithm>
#include <cstdlib>

namespace ptctlmc {

std::vector<int> NProc::edges_from(int loc) const {
  std::vector<int> out;
  for (int e = 0; e < static_cast<int>(edges.size()); ++e)
    if (edges[e].src == loc) out.push_back(e);
  return out;
}

// One O(E) pass builds the per-location outgoing-edge lists, so the hot path
// (successors_net) never scans or allocates to find a location's edges again.
void NProc::build_edge_index() {
  edges_by_loc.assign(locations.size(), {});
  for (int e = 0; e < static_cast<int>(edges.size()); ++e) {
    const int src = edges[e].src;
    if (src >= 0 && src < static_cast<int>(edges_by_loc.size()))
      edges_by_loc[src].push_back(e);
  }
}

void Network::compute_bounds() {
  lbound.assign(nclocks + 1, 0);
  ubound.assign(nclocks + 1, 0);
  auto scan = [&](const std::vector<Constraint>& cs) {
    for (const auto& c : cs) {
      const int v = raw_bound(c.bound);
      if (c.i > 0 && c.j == 0) { if (v > ubound[c.i]) ubound[c.i] = v; }
      else if (c.i == 0 && c.j > 0) { const int lb = -v; if (lb > lbound[c.j]) lbound[c.j] = lb; }
      else if (c.i > 0 && c.j > 0) {
        const int a = std::abs(v);
        ubound[c.i] = std::max(ubound[c.i], a);
        ubound[c.j] = std::max(ubound[c.j], a);
        lbound[c.i] = std::max(lbound[c.i], a);
        lbound[c.j] = std::max(lbound[c.j], a);
      }
    }
  };
  for (const auto& p : procs) {
    for (const auto& l : p.locations) scan(l.invariant);
    for (const auto& e : p.edges) scan(e.clock_guard);
  }
  lbound[0] = ubound[0] = 0;

  // compute_bounds() is the single network-finalisation point (called once after
  // all templates/edges are loaded), so build the outgoing-edge index here too.
  for (auto& p : procs) p.build_edge_index();
}

void Network::compute_loc_bounds() {
  lbound_loc.assign(nprocs(), {});
  ubound_loc.assign(nprocs(), {});
  auto scan = [&](const std::vector<Constraint>& cs, std::vector<std::int32_t>& Lc,
                  std::vector<std::int32_t>& Uc) {
    for (const auto& c : cs) {
      const int v = raw_bound(c.bound);
      if (c.i > 0 && c.j == 0) Uc[c.i] = std::max(Uc[c.i], v);            // x_i <= v
      else if (c.i == 0 && c.j > 0) Lc[c.j] = std::max(Lc[c.j], -v);      // x_j >= -v
      else if (c.i > 0 && c.j > 0) {
        const int a = std::abs(v);
        Uc[c.i] = std::max(Uc[c.i], a); Uc[c.j] = std::max(Uc[c.j], a);
        Lc[c.i] = std::max(Lc[c.i], a); Lc[c.j] = std::max(Lc[c.j], a);
      }
    }
  };
  for (int p = 0; p < nprocs(); ++p) {
    const NProc& proc = procs[p];
    const int nl = static_cast<int>(proc.locations.size());
    lbound_loc[p].assign(nl, std::vector<std::int32_t>(nclocks + 1, kAluNegInf));
    ubound_loc[p].assign(nl, std::vector<std::int32_t>(nclocks + 1, kAluNegInf));
    bool changed = true;
    while (changed) {
      changed = false;
      for (int l = 0; l < nl; ++l) {
        std::vector<std::int32_t> Lc(nclocks + 1, kAluNegInf), Uc(nclocks + 1, kAluNegInf);
        scan(proc.locations[l].invariant, Lc, Uc);
        for (int ei : proc.edges_from(l)) {
          const NEdge& e = proc.edges[ei];
          scan(e.clock_guard, Lc, Uc);  // guard is evaluated at l (before the reset)
          std::vector<char> rst(nclocks + 1, 0);
          for (int r : e.clock_resets) rst[r] = 1;
          for (int c = 1; c <= nclocks; ++c)
            if (!rst[c]) {  // c survives this edge: inherit the target location's bounds
              Lc[c] = std::max(Lc[c], lbound_loc[p][e.tgt][c]);
              Uc[c] = std::max(Uc[c], ubound_loc[p][e.tgt][c]);
            }
        }
        for (int c = 1; c <= nclocks; ++c) {
          if (Lc[c] > lbound_loc[p][l][c]) { lbound_loc[p][l][c] = Lc[c]; changed = true; }
          if (Uc[c] > ubound_loc[p][l][c]) { ubound_loc[p][l][c] = Uc[c]; changed = true; }
        }
      }
    }
  }
}

void Network::compute_active() {
  active.assign(nprocs(), {});
  auto read = [&](const std::vector<Constraint>& cs, std::vector<char>& s) {
    for (const auto& c : cs) { if (c.i > 0) s[c.i] = 1; if (c.j > 0) s[c.j] = 1; }
  };
  for (int p = 0; p < nprocs(); ++p) {
    const NProc& proc = procs[p];
    const int nl = static_cast<int>(proc.locations.size());
    active[p].assign(nl, std::vector<char>(nclocks + 1, 0));
    bool changed = true;
    while (changed) {
      changed = false;
      for (int l = 0; l < nl; ++l) {
        std::vector<char> A(nclocks + 1, 0);
        read(proc.locations[l].invariant, A);
        for (int ei : proc.edges_from(l)) {
          const NEdge& e = proc.edges[ei];
          read(e.clock_guard, A);
          std::vector<char> rst(nclocks + 1, 0);
          for (int r : e.clock_resets) rst[r] = 1;
          for (int c = 1; c <= nclocks; ++c)
            if (active[p][e.tgt][c] && !rst[c]) A[c] = 1;
        }
        for (int c = 1; c <= nclocks; ++c)
          if (A[c] && !active[p][l][c]) { active[p][l][c] = 1; changed = true; }
      }
    }
  }
}

bool Network::resolve_location(const std::string& proc, const std::string& loc, int* pi,
                               int* li) const {
  for (int p = 0; p < nprocs(); ++p)
    if (proc.empty() || procs[p].name == proc)
      for (int l = 0; l < static_cast<int>(procs[p].locations.size()); ++l)
        if (procs[p].locations[l].name == loc) { *pi = p; *li = l; return true; }
  return false;
}

int Network::intvar_index(const std::string& name) const {
  for (int i = 0; i < static_cast<int>(intvars.size()); ++i)
    if (intvars[i].name == name) return i;
  return -1;
}

}  // namespace ptctlmc
