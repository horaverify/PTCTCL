#include "ptctlmc/netreach.hpp"
#include <cstdio>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef PTCTLMC_HAVE_OPENMP
#include <omp.h>
#endif

namespace ptctlmc {
namespace {

struct SymState {
  std::vector<int> locs;
  std::vector<long> ints;
  Dbm zone;
};

bool apply_all(Dbm& z, const std::vector<Constraint>& cs) {
  for (const auto& c : cs)
    if (!z.constrain(c.i, c.j, c.bound)) return false;
  return true;
}

bool int_ok(const std::vector<IntAtom>& g, const std::vector<long>& ints) {
  for (const auto& a : g) {
    const long x = ints[a.var];
    bool r = false;
    switch (a.op) {
      case IntOp::Eq: r = x == a.val; break;
      case IntOp::Ne: r = x != a.val; break;
      case IntOp::Lt: r = x < a.val; break;
      case IntOp::Le: r = x <= a.val; break;
      case IntOp::Gt: r = x > a.val; break;
      case IntOp::Ge: r = x >= a.val; break;
    }
    if (!r) return false;
  }
  return true;
}

std::vector<Constraint> all_invariants(const Network& net, const std::vector<int>& locs) {
  std::vector<Constraint> inv;
  for (int p = 0; p < net.nprocs(); ++p)
    for (const auto& c : net.procs[p].locations[locs[p]].invariant) inv.push_back(c);
  return inv;
}

void frozen_flags(const Network& net, const std::vector<int>& locs, bool* committed, bool* urgent) {
  *committed = *urgent = false;
  for (int p = 0; p < net.nprocs(); ++p) {
    const Location& l = net.procs[p].locations[locs[p]];
    if (l.committed) *committed = true;
    if (l.urgent) *urgent = true;
  }
}

// Reset clocks no process will read before resetting AND that the query does not
// read (qmask) — collapses zones that differ only in dead clocks, soundly.
void reset_inactive(const Network& net, const std::vector<int>& locs, Dbm& z,
                    const std::vector<char>& qmask) {
  static const bool off = std::getenv("PTCTLMC_NOACTIVE") != nullptr;
  if (off) return;
  if (net.active.empty()) return;
  std::vector<char> a(net.nclocks + 1, 0);
  for (int p = 0; p < net.nprocs(); ++p) {
    const auto& ap = net.active[p][locs[p]];
    for (int c = 1; c <= net.nclocks; ++c)
      if (ap[c]) a[c] = 1;
  }
  for (int c = 1; c <= net.nclocks; ++c)
    if (!a[c] && !qmask[c]) z.reset(c);
}

enum ExtrapMode { ExNone = 0, ExLuPlus = 1, ExNoDiag = 2 };

// Does the network compare two clocks anywhere (a diagonal constraint)? If so, the
// LU/aLU abstraction is unsound and we fall back to exact zones (diagonals kept exact).
bool network_has_diagonal(const Network& net) {
  auto diag = [](const std::vector<Constraint>& cs) {
    for (const auto& c : cs) if (c.i > 0 && c.j > 0) return true;
    return false;
  };
  for (const auto& p : net.procs) {
    for (const auto& l : p.locations) if (diag(l.invariant)) return true;
    for (const auto& e : p.edges) if (diag(e.clock_guard)) return true;
  }
  return false;
}

std::string disc_key(const std::vector<int>& locs, const std::vector<long>& ints) {
  std::string k;
  k.reserve(locs.size() * 4 + ints.size() * 8);
  for (int l : locs) k.append(reinterpret_cast<const char*>(&l), sizeof(int));
  for (long v : ints) { std::int64_t x = v; k.append(reinterpret_cast<const char*>(&x), sizeof(std::int64_t)); }
  return k;
}

void successors_net(const Network& net, const SymState& s, const std::vector<std::int32_t>& L,
                    const std::vector<std::int32_t>& U, const std::vector<char>& qmask, int emode, std::vector<SymState>& out) {
  bool src_committed, src_urgent;
  frozen_flags(net, s.locs, &src_committed, &src_urgent);
  for (int p = 0; p < net.nprocs(); ++p) {
    if (src_committed && !net.procs[p].locations[s.locs[p]].committed) continue;
    for (int ei : net.procs[p].edges_from(s.locs[p])) {
      const NEdge& e = net.procs[p].edges[ei];
      if (e.sync_chan >= 0) continue;
      if (!int_ok(e.int_guard, s.ints)) continue;
      Dbm z1 = s.zone;
      if (!apply_all(z1, e.clock_guard)) continue;
      for (int r : e.clock_resets) z1.reset(r);

      SymState ns;
      ns.locs = s.locs;
      ns.locs[p] = e.tgt;
      ns.ints = s.ints;
      for (const auto& a : e.int_updates) ns.ints[a.var] = a.val;

      const std::vector<Constraint> inv = all_invariants(net, ns.locs);
      if (!apply_all(z1, inv)) continue;
      bool nc, nu;
      frozen_flags(net, ns.locs, &nc, &nu);
      if (!nc && !nu) { z1.delay(); apply_all(z1, inv); }
      reset_inactive(net, ns.locs, z1, qmask);
      if (emode == ExLuPlus) z1.extrapolate_lu_plus(L, U);
      else if (emode == ExNoDiag) z1.extrapolate_lu_plus_nodiag(L, U);
      if (z1.empty()) continue;
      ns.zone = std::move(z1);
      out.push_back(std::move(ns));
    }
  }

  // ---- binary channel synchronisation (a! in proc i  ||  a? in proc j) ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_from(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1) continue;  // sender a!
      if (net.chan_broadcast[se.sync_chan]) continue;      // broadcast handled below
      if (!int_ok(se.int_guard, s.ints)) continue;
      const bool i_comm = net.procs[i].locations[s.locs[i]].committed;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        for (int ej : net.procs[j].edges_from(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan != se.sync_chan || re.sync_dir != -1) continue;  // receiver a?
          if (src_committed && !(i_comm || net.procs[j].locations[s.locs[j]].committed)) continue;
          if (!int_ok(re.int_guard, s.ints)) continue;
          Dbm z1 = s.zone;
          if (!apply_all(z1, se.clock_guard)) continue;
          if (!apply_all(z1, re.clock_guard)) continue;
          for (int r : se.clock_resets) z1.reset(r);
          for (int r : re.clock_resets) z1.reset(r);
          SymState ns;
          ns.locs = s.locs;
          ns.locs[i] = se.tgt;
          ns.locs[j] = re.tgt;
          ns.ints = s.ints;
          for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;  // sender then receiver
          for (const auto& a : re.int_updates) ns.ints[a.var] = a.val;
          const std::vector<Constraint> inv = all_invariants(net, ns.locs);
          if (!apply_all(z1, inv)) continue;
          bool nc, nu;
          frozen_flags(net, ns.locs, &nc, &nu);
          if (!nc && !nu) { z1.delay(); apply_all(z1, inv); }
          reset_inactive(net, ns.locs, z1, qmask);
          if (emode == ExLuPlus) z1.extrapolate_lu_plus(L, U);
      else if (emode == ExNoDiag) z1.extrapolate_lu_plus_nodiag(L, U);
          if (z1.empty()) continue;
          ns.zone = std::move(z1);
          out.push_back(std::move(ns));
        }
      }
    }
  }

  // ---- broadcast synchronisation: a! fires with ALL enabled a? (0 or more) ----
  for (int i = 0; i < net.nprocs(); ++i) {
    for (int ei : net.procs[i].edges_from(s.locs[i])) {
      const NEdge& se = net.procs[i].edges[ei];
      if (se.sync_chan < 0 || se.sync_dir != 1 || !net.chan_broadcast[se.sync_chan]) continue;
      if (!int_ok(se.int_guard, s.ints)) continue;
      // Each other process contributes at most one enabled a? (nondeterministic choice).
      std::vector<std::pair<int, std::vector<int>>> parts;
      for (int j = 0; j < net.nprocs(); ++j) {
        if (j == i) continue;
        std::vector<int> es;
        for (int ej : net.procs[j].edges_from(s.locs[j])) {
          const NEdge& re = net.procs[j].edges[ej];
          if (re.sync_chan == se.sync_chan && re.sync_dir == -1 && int_ok(re.int_guard, s.ints))
            es.push_back(ej);
        }
        if (!es.empty()) parts.push_back({j, std::move(es)});
      }
      std::vector<std::size_t> pick(parts.size(), 0);
      for (;;) {
        bool part_committed = net.procs[i].locations[s.locs[i]].committed;
        for (const auto& pr : parts)
          if (net.procs[pr.first].locations[s.locs[pr.first]].committed) part_committed = true;
        if (!src_committed || part_committed) {
          Dbm z1 = s.zone;
          bool ok = apply_all(z1, se.clock_guard);
          for (std::size_t t = 0; ok && t < parts.size(); ++t)
            ok = apply_all(z1, net.procs[parts[t].first].edges[parts[t].second[pick[t]]].clock_guard);
          if (ok) {
            SymState ns;
            ns.locs = s.locs; ns.ints = s.ints;
            ns.locs[i] = se.tgt;
            for (const auto& a : se.int_updates) ns.ints[a.var] = a.val;
            for (int r : se.clock_resets) z1.reset(r);
            for (std::size_t t = 0; t < parts.size(); ++t) {
              const NEdge& re = net.procs[parts[t].first].edges[parts[t].second[pick[t]]];
              ns.locs[parts[t].first] = re.tgt;
              for (const auto& a : re.int_updates) ns.ints[a.var] = a.val;
              for (int r : re.clock_resets) z1.reset(r);
            }
            const std::vector<Constraint> inv = all_invariants(net, ns.locs);
            if (apply_all(z1, inv)) {
              bool nc, nu;
              frozen_flags(net, ns.locs, &nc, &nu);
              if (!nc && !nu) { z1.delay(); apply_all(z1, inv); }
              reset_inactive(net, ns.locs, z1, qmask);
              if (emode == ExLuPlus) z1.extrapolate_lu_plus(L, U);
              else if (emode == ExNoDiag) z1.extrapolate_lu_plus_nodiag(L, U);
              if (!z1.empty()) { ns.zone = std::move(z1); out.push_back(std::move(ns)); }
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

void extend_bounds(std::vector<std::int32_t>& L, std::vector<std::int32_t>& U,
                   const std::vector<Constraint>& cs) {
  for (const auto& c : cs) {
    const int v = raw_bound(c.bound);
    if (c.i > 0 && c.j == 0) { if (v > U[c.i]) U[c.i] = v; }
    else if (c.i == 0 && c.j > 0) { const int lb = -v; if (lb > L[c.j]) L[c.j] = lb; }
    else if (c.i > 0 && c.j > 0) {
      const int a = std::abs(v);
      if (a > U[c.i]) U[c.i] = a;
      if (a > U[c.j]) U[c.j] = a;
      if (a > L[c.i]) L[c.i] = a;
      if (a > L[c.j]) L[c.j] = a;
    }
  }
}

}  // namespace

bool reachable_net(const Network& net, const NetTarget& target, Backend backend, ReachStats* stats) {
  std::vector<std::int32_t> L = net.lbound, U = net.ubound;
  extend_bounds(L, U, target.clock_cond);
  L[0] = U[0] = 0;  // reference clock: mandatory for the aLU subsumption test
  // With diagonal constraints the aLU abstraction is unsound: fall back to exact zones
  // (diagonals kept exact) + plain inclusion subsumption.
  const bool has_diag = network_has_diagonal(net);
  const int emode = has_diag ? ExNoDiag : ExNone;
  std::vector<char> qmask(net.nclocks + 1, 0);  // clocks the query reads -> keep active
  for (const auto& c : target.clock_cond) { if (c.i > 0) qmask[c.i] = 1; if (c.j > 0) qmask[c.j] = 1; }

  // Query clock constants floor every state's aLU bounds (the property may hold anywhere).
  std::vector<std::int32_t> Lq(net.nclocks + 1, kAluNegInf), Uq(net.nclocks + 1, kAluNegInf);
  for (const auto& c : target.clock_cond) {
    const int v = raw_bound(c.bound);
    if (c.i > 0 && c.j == 0) Uq[c.i] = std::max(Uq[c.i], v);
    else if (c.i == 0 && c.j > 0) Lq[c.j] = std::max(Lq[c.j], -v);
    else if (c.i > 0 && c.j > 0) {
      const int a = std::abs(v);
      Uq[c.i] = std::max(Uq[c.i], a); Uq[c.j] = std::max(Uq[c.j], a);
      Lq[c.i] = std::max(Lq[c.i], a); Lq[c.j] = std::max(Lq[c.j], a);
    }
  }
  // Per-state aLU bounds: max of per-location bounds over processes, floored by the query.
  auto state_bounds = [&](const std::vector<int>& locs, std::vector<std::int32_t>& L2,
                          std::vector<std::int32_t>& U2) {
    if (net.lbound_loc.empty()) { L2 = L; U2 = U; return; }  // fall back to global bounds
    L2.assign(net.nclocks + 1, kAluNegInf);
    U2.assign(net.nclocks + 1, kAluNegInf);
    for (int p = 0; p < net.nprocs(); ++p)
      for (int c = 1; c <= net.nclocks; ++c) {
        L2[c] = std::max(L2[c], net.lbound_loc[p][locs[p]][c]);
        U2[c] = std::max(U2[c], net.ubound_loc[p][locs[p]][c]);
      }
    for (int c = 1; c <= net.nclocks; ++c) { L2[c] = std::max(L2[c], Lq[c]); U2[c] = std::max(U2[c], Uq[c]); }
    L2[0] = U2[0] = 0;
  };

  auto is_target = [&](const SymState& s) {
    for (auto& lr : target.loc_req)
      if (s.locs[lr.first] != lr.second) return false;
    if (!int_ok(target.int_cond, s.ints)) return false;
    Dbm t = s.zone;
    return apply_all(t, target.clock_cond);
  };

  bool parallel = (backend == Backend::Omp);
#ifndef PTCTLMC_HAVE_OPENMP
  parallel = false;
#endif
  int nthreads = 1;
#ifdef PTCTLMC_HAVE_OPENMP
  if (parallel) nthreads = omp_get_max_threads();
#endif
  const int S = parallel ? nthreads : 1;  // passed-list shards (== merge threads)

  // Passed list SHARDED by discrete-state hash: a given discrete state always lands
  // in the same shard, so each shard is a self-contained antichain owned by exactly
  // one thread during the merge — the O(bucket^2 * dim^3) convex-merge now runs in
  // parallel with no locks. Per discrete state: antichain + convex-union merge.
  struct Entry { SymState s; int level; };
  std::vector<std::unordered_map<std::string, std::vector<Entry>>> shards(S);
  std::hash<std::string> hsh;
  auto shard_of = [&](const std::string& dk) -> int {
    return S == 1 ? 0 : static_cast<int>(hsh(dk) % static_cast<unsigned>(S));
  };
  auto add_to = [&](std::unordered_map<std::string, std::vector<Entry>>& mp, SymState&& s,
                    const std::string& dk, int level, std::unordered_set<std::string>& touched) {
    std::vector<std::int32_t> Ls, Us;
    state_bounds(s.locs, Ls, Us);  // per-location aLU bounds for this discrete state
    auto& b = mp[dk];
    for (const auto& e : b)
      if (has_diag ? e.s.zone.includes(s.zone) : s.zone.alu_subsumed_by(e.s.zone, Ls, Us))
        return;  // s subsumed by a stored zone (aLU, or plain inclusion for diagonal models)
    b.erase(std::remove_if(b.begin(), b.end(),
                           [&](const Entry& e) {
                             return has_diag ? s.zone.includes(e.s.zone)
                                             : e.s.zone.alu_subsumed_by(s.zone, Ls, Us);
                           }),
            b.end());  // drop stored zones now subsumed by s
    // Convex-union merge: absorb stored zones whose union with s is exactly convex.
    for (bool merged = true; merged;) {
      merged = false;
      for (std::size_t k = 0; k < b.size(); ++k)
        if (s.zone.try_union(b[k].s.zone)) { b.erase(b.begin() + k); merged = true; break; }
    }
    b.push_back(Entry{std::move(s), level});
    touched.insert(dk);
  };

  // initial symbolic state
  SymState s0;
  s0.locs.resize(net.nprocs());
  for (int p = 0; p < net.nprocs(); ++p) s0.locs[p] = net.procs[p].initial;
  s0.ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0.ints[v] = net.intvars[v].init;
  s0.zone = Dbm(net.nclocks);
  {
    const std::vector<Constraint> inv = all_invariants(net, s0.locs);
    if (!apply_all(s0.zone, inv)) return false;
    bool c, u;
    frozen_flags(net, s0.locs, &c, &u);
    if (!c && !u) { s0.zone.delay(); apply_all(s0.zone, inv); }
    reset_inactive(net, s0.locs, s0.zone, qmask);
    // no extrapolation: exact zones + aLU subsumption keep the passed list finite & minimal
  }
  if (is_target(s0)) return true;

  std::vector<SymState> frontier;
  frontier.push_back(s0);
  {
    const std::string dk = disc_key(s0.locs, s0.ints);
    std::unordered_set<std::string> t0;
    add_to(shards[shard_of(dk)], std::move(s0), dk, 0, t0);
  }

  int level = 0;
  std::size_t nexpanded = 0;
  while (!frontier.empty()) {
    if (stats) stats->explored += frontier.size();
    if (has_diag && (nexpanded += frontier.size()) > 500000)  // unbounded-diagonal safety cap
      throw std::runtime_error("state cap exceeded — model likely has unbounded diagonal clocks");
    ++level;
    // (1) generate successors of the frontier in parallel
    std::vector<std::vector<SymState>> tls(nthreads);
    if (parallel) {
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel
      {
        auto& buf = tls[omp_get_thread_num()];
#pragma omp for schedule(dynamic, 8)
        for (long i = 0; i < static_cast<long>(frontier.size()); ++i)
          successors_net(net, frontier[i], L, U, qmask, emode, buf);
      }
#endif
    } else {
      for (auto& f : frontier) successors_net(net, f, L, U, qmask, emode, tls[0]);
    }

    // (2) route each successor to its shard (serial, cheap); early-exit on target
    std::vector<std::vector<std::pair<std::string, SymState>>> byshard(S);
    for (auto& buf : tls)
      for (auto& s : buf) {
        if (stats) ++stats->generated;
        if (is_target(s)) return true;
        std::string dk = disc_key(s.locs, s.ints);
        const int sh = shard_of(dk);
        byshard[sh].emplace_back(std::move(dk), std::move(s));
      }

    // (3) merge each shard independently, in parallel; collect this level's survivors
    std::vector<std::vector<SymState>> nf(S);
    if (parallel) {
#ifdef PTCTLMC_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(S)
      for (int t = 0; t < S; ++t) {
        std::unordered_set<std::string> touched;
        for (auto& pr : byshard[t]) add_to(shards[t], std::move(pr.second), pr.first, level, touched);
        for (const auto& dk : touched)
          for (const auto& e : shards[t][dk])
            if (e.level == level) nf[t].push_back(e.s);
      }
#endif
    } else {
      std::unordered_set<std::string> touched;
      for (auto& pr : byshard[0]) add_to(shards[0], std::move(pr.second), pr.first, level, touched);
      for (const auto& dk : touched)
        for (const auto& e : shards[0][dk])
          if (e.level == level) nf[0].push_back(e.s);
    }

    // (4) assemble the next frontier
    std::vector<SymState> next;
    for (auto& v : nf)
      for (auto& s : v) next.push_back(std::move(s));
    frontier.swap(next);
  }
  if (std::getenv("PTCTLMC_STATS")) {
    std::size_t zones = 0, disc = 0;
    for (const auto& mp : shards) {
      disc += mp.size();
      for (const auto& kv : mp) zones += kv.second.size();
    }
    std::fprintf(stderr, "[stats] discrete-states=%zu stored-zones=%zu\n", disc, zones);
  }
  return false;
}

bool exists_invariant_run(const Network& net, const DiscPred& phi, Backend backend,
                          ReachStats* stats) {
  (void)backend;  // liveness graph search is sequential (cycle detection); omp TODO
  std::vector<std::int32_t> L = net.lbound, U = net.ubound;
  const int emode = network_has_diagonal(net) ? ExNoDiag : ExLuPlus;  // diagonals: keep exact
  const std::vector<char> qmask(net.nclocks + 1, 0);  // liveness predicate has no clocks

  auto sat = [&](const SymState& s) -> bool {
    for (const auto& r : phi.locs) {
      const bool eq = (s.locs[r.proc] == r.loc);
      if (eq != r.equal) return false;
    }
    return int_ok(phi.ints, s.ints);
  };
  auto key = [&](const SymState& s) -> std::string {
    std::string k = disc_key(s.locs, s.ints);
    const auto& d = s.zone.data();
    k.append(reinterpret_cast<const char*>(d.data()), d.size() * sizeof(raw_t));
    return k;
  };

  // initial symbolic state (same construction as reachable_net)
  SymState s0;
  s0.locs.resize(net.nprocs());
  for (int p = 0; p < net.nprocs(); ++p) s0.locs[p] = net.procs[p].initial;
  s0.ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0.ints[v] = net.intvars[v].init;
  s0.zone = Dbm(net.nclocks);
  {
    const std::vector<Constraint> inv = all_invariants(net, s0.locs);
    if (!apply_all(s0.zone, inv)) return false;
    bool c, u;
    frozen_flags(net, s0.locs, &c, &u);
    if (!c && !u) { s0.zone.delay(); apply_all(s0.zone, inv); }
    reset_inactive(net, s0.locs, s0.zone, qmask);
    if (emode == ExLuPlus) s0.zone.extrapolate_lu_plus(L, U);
    else if (emode == ExNoDiag) s0.zone.extrapolate_lu_plus_nodiag(L, U);
  }
  if (!sat(s0)) return false;  // no run from the initial state can stay in phi

  // Build the phi-restricted extrapolated zone graph (nodes deduped by exact zone).
  std::unordered_map<std::string, int> id;
  std::vector<SymState> node;
  std::vector<std::vector<int>> adj;
  auto intern = [&](SymState&& s) -> int {
    std::string k = key(s);
    auto it = id.find(k);
    if (it != id.end()) return it->second;
    const int n = static_cast<int>(node.size());
    id.emplace(std::move(k), n);
    node.push_back(std::move(s));
    adj.emplace_back();
    return n;
  };
  // A location where every clock can grow without bound (after delay + invariant) lets
  // time diverge => an infinite run that stays put — an E[] witness on its own.
  auto time_diverges = [&](const SymState& s) -> bool {
    if (net.nclocks == 0) return false;
    Dbm z = s.zone;
    z.delay();
    apply_all(z, all_invariants(net, s.locs));
    if (z.empty()) return false;
    for (int i = 1; i <= net.nclocks; ++i)
      if (z.get(i, 0) != kInf) return false;
    return true;
  };

  intern(std::move(s0));
  for (int u = 0; u < static_cast<int>(node.size()); ++u) {
    if (stats) ++stats->explored;
    if (time_diverges(node[u])) return true;
    std::vector<SymState> succ;
    successors_net(net, node[u], L, U, qmask, emode, succ);
    for (auto& ns : succ) {
      if (!sat(ns)) continue;
      const int v = intern(std::move(ns));
      adj[u].push_back(v);
    }
  }

  // Reachable cycle? Iterative three-colour DFS; a back edge (to a GREY node) = cycle.
  enum { White, Grey, Black };
  std::vector<char> color(node.size(), White);
  std::vector<std::pair<int, std::size_t>> stk;
  for (int root = 0; root < static_cast<int>(node.size()); ++root) {
    if (color[root] != White) continue;
    stk.push_back({root, 0});
    color[root] = Grey;
    while (!stk.empty()) {
      const int u = stk.back().first;
      const std::size_t idx = stk.back().second;
      if (idx < adj[u].size()) {
        stk.back().second = idx + 1;
        const int v = adj[u][idx];
        if (color[v] == Grey) return true;
        if (color[v] == White) { color[v] = Grey; stk.push_back({v, 0}); }
      } else {
        color[u] = Black;
        stk.pop_back();
      }
    }
  }
  return false;
}

bool leads_to_net(const Network& net, const DiscPred& p, const DiscPred& q, Backend backend,
                  ReachStats* stats) {
  (void)backend;
  std::vector<std::int32_t> L = net.lbound, U = net.ubound;
  const int emode = network_has_diagonal(net) ? ExNoDiag : ExLuPlus;
  const std::vector<char> qmask(net.nclocks + 1, 0);

  auto holds = [&](const DiscPred& pr, const SymState& s) -> bool {
    for (const auto& r : pr.locs) { const bool eq = (s.locs[r.proc] == r.loc); if (eq != r.equal) return false; }
    return int_ok(pr.ints, s.ints);
  };
  auto key = [&](const SymState& s) -> std::string {
    std::string k = disc_key(s.locs, s.ints);
    const auto& d = s.zone.data();
    k.append(reinterpret_cast<const char*>(d.data()), d.size() * sizeof(raw_t));
    return k;
  };
  auto extrap = [&](Dbm& z) {
    if (emode == ExLuPlus) z.extrapolate_lu_plus(L, U);
    else if (emode == ExNoDiag) z.extrapolate_lu_plus_nodiag(L, U);
  };

  SymState s0;
  s0.locs.resize(net.nprocs());
  for (int pr = 0; pr < net.nprocs(); ++pr) s0.locs[pr] = net.procs[pr].initial;
  s0.ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0.ints[v] = net.intvars[v].init;
  s0.zone = Dbm(net.nclocks);
  {
    const std::vector<Constraint> inv = all_invariants(net, s0.locs);
    if (!apply_all(s0.zone, inv)) return true;  // unreachable init => vacuously holds
    bool c, u; frozen_flags(net, s0.locs, &c, &u);
    if (!c && !u) { s0.zone.delay(); apply_all(s0.zone, inv); }
    reset_inactive(net, s0.locs, s0.zone, qmask);
    extrap(s0.zone);
  }

  // Build the full reachable graph (exact-hash nodes, extrapolated for finiteness).
  std::unordered_map<std::string, int> id;
  std::vector<SymState> node;
  std::vector<std::vector<int>> adj;
  auto intern = [&](SymState&& s) -> int {
    std::string k = key(s);
    auto it = id.find(k);
    if (it != id.end()) return it->second;
    const int n = static_cast<int>(node.size());
    id.emplace(std::move(k), n); node.push_back(std::move(s)); adj.emplace_back();
    return n;
  };
  auto time_diverges = [&](const SymState& s) -> bool {
    if (net.nclocks == 0) return false;
    Dbm z = s.zone; z.delay(); apply_all(z, all_invariants(net, s.locs));
    if (z.empty()) return false;
    for (int i = 1; i <= net.nclocks; ++i) if (z.get(i, 0) != kInf) return false;
    return true;
  };
  intern(std::move(s0));
  for (int u = 0; u < static_cast<int>(node.size()); ++u) {
    if (stats) ++stats->explored;
    std::vector<SymState> succ;
    successors_net(net, node[u], L, U, qmask, emode, succ);
    for (auto& ns : succ) { const int v = intern(std::move(ns)); adj[u].push_back(v); }
  }
  const int N = static_cast<int>(node.size());
  std::vector<char> pv(N), qv(N), seed(N, 0);
  for (int i = 0; i < N; ++i) { pv[i] = holds(p, node[i]); qv[i] = holds(q, node[i]); }
  // Seeds of an infinite/maximal ¬q run: a ¬q deadlock, or a ¬q state where time diverges.
  for (int i = 0; i < N; ++i)
    if (!qv[i] && (adj[i].empty() || time_diverges(node[i]))) seed[i] = 1;

  // ¬q-cycle nodes are also seeds — iterative Tarjan SCC over the ¬q-subgraph.
  std::vector<int> tidx(N, -1), low(N, 0);
  std::vector<char> onstk(N, 0);
  std::vector<int> tstk;
  int counter = 0;
  struct Frame { int u; std::size_t i; };
  for (int s = 0; s < N; ++s) {
    if (qv[s] || tidx[s] != -1) continue;
    std::vector<Frame> fs;
    fs.push_back({s, 0}); tidx[s] = low[s] = counter++; tstk.push_back(s); onstk[s] = 1;
    while (!fs.empty()) {
      const int u = fs.back().u;  // copy: fs may reallocate on push_back below
      bool descended = false;
      while (fs.back().i < adj[u].size()) {
        const int v = adj[u][fs.back().i++];
        if (qv[v]) continue;
        if (tidx[v] == -1) {
          tidx[v] = low[v] = counter++; tstk.push_back(v); onstk[v] = 1; fs.push_back({v, 0});
          descended = true; break;
        } else if (onstk[v]) { if (tidx[v] < low[u]) low[u] = tidx[v]; }
      }
      if (descended) continue;
      if (low[u] == tidx[u]) {
        std::vector<int> comp; int w;
        do { w = tstk.back(); tstk.pop_back(); onstk[w] = 0; comp.push_back(w); } while (w != u);
        bool cyc = comp.size() > 1;
        if (!cyc) for (int v : adj[u]) if (v == u) cyc = true;  // self-loop
        if (cyc) for (int c : comp) seed[c] = 1;
      }
      fs.pop_back();
      if (!fs.empty() && low[u] < low[fs.back().u]) low[fs.back().u] = low[u];
    }
  }

  // escape = ¬q nodes that can reach a seed via ¬q-edges (reverse reachability).
  std::vector<std::vector<int>> radj(N);
  for (int u = 0; u < N; ++u)
    if (!qv[u]) for (int v : adj[u]) if (!qv[v]) radj[v].push_back(u);
  std::vector<char> escape(N, 0);
  std::vector<int> stk;
  for (int i = 0; i < N; ++i) if (seed[i]) { escape[i] = 1; stk.push_back(i); }
  while (!stk.empty()) {
    const int u = stk.back(); stk.pop_back();
    for (int w : radj[u]) if (!escape[w]) { escape[w] = 1; stk.push_back(w); }
  }
  // p --> q fails iff a reachable p ∧ ¬q state can stay in ¬q forever (escape).
  for (int i = 0; i < N; ++i) if (pv[i] && !qv[i] && escape[i]) return false;
  return true;
}

}  // namespace ptctlmc
