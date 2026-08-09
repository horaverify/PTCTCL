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
  lraw_loc.assign(nprocs(), {});
  uraw_loc.assign(nprocs(), {});
  // Track the raw bound alongside the integer. Raw ordering already puts le(v) above lt(v), so
  // `max` on raw is exactly the "weakest of the two" rule Cor. 3.6 needs: given both `x < 5` and
  // `x <= 5`, the collapsed constraint is `x <= 5`.
  auto rmax = [](raw_t& slot, raw_t r) { if (r != kGNone && (slot == kGNone || r > slot)) slot = r; };
  auto scan = [&](const std::vector<Constraint>& cs, std::vector<std::int32_t>& Lc,
                  std::vector<std::int32_t>& Uc, std::vector<raw_t>& Lr, std::vector<raw_t>& Ur) {
    for (const auto& c : cs) {
      const int v = raw_bound(c.bound);
      const bool strict = raw_is_strict(c.bound);
      if (c.i > 0 && c.j == 0) {                                          // x_i <| v
        Uc[c.i] = std::max(Uc[c.i], v);
        rmax(Ur[c.i], strict ? lt(v) : le(v));
      } else if (c.i == 0 && c.j > 0) {                                   // x_j >| -v
        Lc[c.j] = std::max(Lc[c.j], -v);
        rmax(Lr[c.j], strict ? lt(-v) : le(-v));
      } else if (c.i > 0 && c.j > 0) {
        // Diagonal: G would keep it as a diagonal constraint. We do not (dbm_is_g_le is the
        // diagonal-free instance), so fall back to the same conservative per-clock widening the
        // integer path uses, with NON-strict bounds -- the safe choice when strictness is lost.
        const int a = std::abs(v);
        Uc[c.i] = std::max(Uc[c.i], a); Uc[c.j] = std::max(Uc[c.j], a);
        Lc[c.i] = std::max(Lc[c.i], a); Lc[c.j] = std::max(Lc[c.j], a);
        rmax(Ur[c.i], le(a)); rmax(Ur[c.j], le(a));
        rmax(Lr[c.i], le(-a)); rmax(Lr[c.j], le(-a));
      }
    }
  };
  for (int p = 0; p < nprocs(); ++p) {
    const NProc& proc = procs[p];
    const int nl = static_cast<int>(proc.locations.size());
    lbound_loc[p].assign(nl, std::vector<std::int32_t>(nclocks + 1, kAluNegInf));
    ubound_loc[p].assign(nl, std::vector<std::int32_t>(nclocks + 1, kAluNegInf));
    lraw_loc[p].assign(nl, std::vector<raw_t>(nclocks + 1, kGNone));
    uraw_loc[p].assign(nl, std::vector<raw_t>(nclocks + 1, kGNone));
    bool changed = true;
    while (changed) {
      changed = false;
      for (int l = 0; l < nl; ++l) {
        std::vector<std::int32_t> Lc(nclocks + 1, kAluNegInf), Uc(nclocks + 1, kAluNegInf);
        std::vector<raw_t> Lr(nclocks + 1, kGNone), Ur(nclocks + 1, kGNone);
        scan(proc.locations[l].invariant, Lc, Uc, Lr, Ur);
        for (int ei : proc.edges_from(l)) {
          const NEdge& e = proc.edges[ei];
          scan(e.clock_guard, Lc, Uc, Lr, Ur);  // guard is evaluated at l (before the reset)
          std::vector<char> rst(nclocks + 1, 0);
          for (int r : e.clock_resets) rst[r] = 1;
          for (int c = 1; c <= nclocks; ++c)
            if (!rst[c]) {  // c survives this edge: inherit the target location's bounds
              Lc[c] = std::max(Lc[c], lbound_loc[p][e.tgt][c]);
              Uc[c] = std::max(Uc[c], ubound_loc[p][e.tgt][c]);
              rmax(Lr[c], lraw_loc[p][e.tgt][c]);
              rmax(Ur[c], uraw_loc[p][e.tgt][c]);
            }
        }
        for (int c = 1; c <= nclocks; ++c) {
          if (Lc[c] > lbound_loc[p][l][c]) { lbound_loc[p][l][c] = Lc[c]; changed = true; }
          if (Uc[c] > ubound_loc[p][l][c]) { ubound_loc[p][l][c] = Uc[c]; changed = true; }
          if (Lr[c] != kGNone && (lraw_loc[p][l][c] == kGNone || Lr[c] > lraw_loc[p][l][c]))
            { lraw_loc[p][l][c] = Lr[c]; changed = true; }
          if (Ur[c] != kGNone && (uraw_loc[p][l][c] == kGNone || Ur[c] > uraw_loc[p][l][c]))
            { uraw_loc[p][l][c] = Ur[c]; changed = true; }
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
  // Bit-pack the fixpoint for the per-successor lookup (see Network::active_mask).
  active_mask.clear();
  if (nclocks <= 63) {
    active_mask.resize(nprocs());
    for (int p = 0; p < nprocs(); ++p) {
      active_mask[p].assign(active[p].size(), 0);
      for (std::size_t l = 0; l < active[p].size(); ++l)
        for (int c = 1; c <= nclocks; ++c)
          if (active[p][l][c]) active_mask[p][l] |= std::uint64_t(1) << c;
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


// Cells mentioned anywhere in a general expression / statement. Those edges are not analysed,
// so every cell they touch is conservatively live everywhere.
static void gen_cells(const ExprP& e, std::vector<char>& s) {
  if (!e) return;
  if (e->op == EOp::Var && e->var >= 0 && e->var < (int)s.size()) s[e->var] = 1;
  if (e->op == EOp::Index)
    for (int c = e->var; c < e->var + e->arr_len && c < (int)s.size(); ++c)
      if (c >= 0) s[c] = 1;
  for (const auto& k : e->kids) gen_cells(k, s);
}
static void gen_cells(const StmtP& t, std::vector<char>& s) {
  if (!t) return;
  if (t->var >= 0) {
    const int n = t->arr_len > 0 ? t->arr_len : 1;
    for (int c = t->var; c < t->var + n && c < (int)s.size(); ++c) s[c] = 1;
  }
  gen_cells(t->idx, s);
  gen_cells(t->rhs, s);
  gen_cells(t->cond, s);
  for (const auto& k : t->kids) gen_cells(k, s);
}

void Network::compute_int_live() {
  const int nv = static_cast<int>(intvars.size());
  iread.assign(nprocs(), {});
  iread_mask.clear();
  if (nv == 0) return;

  // Cells no analysis covers: anything a general guard/update mentions, and anything a user
  // function body touches. Live everywhere, unconditionally.
  std::vector<char> always(nv, 0);
  for (const auto& f : funcs) gen_cells(f.body, always);
  for (const auto& p : procs)
    for (const auto& e : p.edges) { gen_cells(e.gen_guard, always); gen_cells(e.gen_update, always); }

  for (int p = 0; p < nprocs(); ++p) {
    const NProc& proc = procs[p];
    const int nl = static_cast<int>(proc.locations.size());
    iread[p].assign(nl, std::vector<char>(nv, 0));
    bool changed = true;
    while (changed) {
      changed = false;
      for (int l = 0; l < nl; ++l) {
        std::vector<char> A(nv, 0);
        for (int ei : proc.edges_from(l)) {
          const NEdge& e = proc.edges[ei];
          for (const auto& g : e.int_guard)                       // the guard reads it here
            if (g.var >= 0 && g.var < nv) A[g.var] = 1;
          std::vector<char> wrt(nv, 0);
          for (const auto& u : e.int_updates)
            if (u.var >= 0 && u.var < nv) wrt[u.var] = 1;         // `var := const` reads nothing
          for (int v = 0; v < nv; ++v)
            if (iread[p][e.tgt][v] && !wrt[v]) A[v] = 1;          // read later, not written here
        }
        for (int v = 0; v < nv; ++v)
          if (A[v] && !iread[p][l][v]) { iread[p][l][v] = 1; changed = true; }
      }
    }
    for (int l = 0; l < nl; ++l)
      for (int v = 0; v < nv; ++v)
        if (always[v]) iread[p][l][v] = 1;
  }

  if (nv <= 63) {
    iread_mask.assign(nprocs(), {});
    for (int p = 0; p < nprocs(); ++p) {
      iread_mask[p].assign(procs[p].locations.size(), 0);
      for (std::size_t l = 0; l < procs[p].locations.size(); ++l) {
        std::uint64_t m = 0;
        for (int v = 0; v < nv; ++v) if (iread[p][l][v]) m |= std::uint64_t(1) << v;
        iread_mask[p][l] = m;
      }
    }
  }
}

}  // namespace ptctlmc
