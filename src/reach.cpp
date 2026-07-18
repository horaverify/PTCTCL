#include "ptctlmc/reach.hpp"

#include <cstdlib>
#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include "ptctlmc/cuda_reach.hpp"

#ifdef PTCTLMC_HAVE_OPENMP
#include <omp.h>
#endif

namespace ptctlmc {
namespace {

bool apply(Dbm& z, const std::vector<Constraint>& cs) {
  for (const auto& c : cs)
    if (!z.constrain(c.i, c.j, c.bound)) return false;
  return true;
}

void delay_in(Dbm& z, const Location& loc) {
  if (loc.urgent || loc.committed) return;  // no time may pass
  z.delay();
  apply(z, loc.invariant);
}

// Extend LU bounds with the constants appearing in `cs` (so extrapolation never
// widens past what a guard / invariant / query condition needs — this is what keeps
// Extra_LU sound w.r.t. the property being checked).
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

// Extrapolation policy: LU+ (fast, sound for diagonal-FREE automata) or per-pair
// k-normalisation (sound WITH diagonal constraints). `C` is the per-pair max-constant
// matrix used by the diagonal path.
struct Extrap {
  const std::vector<std::int32_t>*L, *U;
  bool diagonal = false;
  void apply(Dbm& z) const {
    if (diagonal) z.extrapolate_lu_plus_nodiag(*L, *U);  // keep diagonals exact
    else z.extrapolate_lu_plus(*L, *U);
  }
};

// Any difference constraint x_i - x_j with i>0 && j>0 is a "diagonal".
bool has_diagonal(const std::vector<Constraint>& cs) {
  for (const auto& c : cs) if (c.i > 0 && c.j > 0) return true;
  return false;
}

bool initial_zone(const TimedAutomaton& ta, Dbm& z0, const Extrap& ex) {
  z0 = Dbm(ta.nclocks);
  if (!apply(z0, ta.locations[ta.initial].invariant)) return false;
  delay_in(z0, ta.locations[ta.initial]);
  ex.apply(z0);
  return true;
}

void successors(const TimedAutomaton& ta, int loc, const Dbm& z,
                std::vector<std::pair<int, Dbm>>& out, const Extrap& ex) {
  for (int ei : ta.edges_from(loc)) {
    const Edge& e = ta.edges[ei];
    Dbm z1 = z;
    if (!apply(z1, e.guard)) continue;
    for (int r : e.resets) z1.reset(r);
    const Location& tl = ta.locations[e.tgt];
    if (!apply(z1, tl.invariant)) continue;
    delay_in(z1, tl);
    ex.apply(z1);
    if (!z1.empty()) out.emplace_back(e.tgt, std::move(z1));
  }
}

}  // namespace

bool reachable(const TimedAutomaton& ta, int target_loc, const std::vector<Constraint>& target,
               ReachStats* stats, Backend backend) {
  if (backend == Backend::Cuda) {
#ifdef PTCTLMC_HAVE_CUDA
    return reachable_cuda(ta, target_loc, target, stats);
#else
    throw std::runtime_error("CUDA backend not available in this build");
#endif
  }
  bool parallel = (backend == Backend::Omp);

  // Effective LU bounds = model bounds widened with the query's clock constants.
  std::vector<std::int32_t> L = ta.lbound, U = ta.ubound;
  extend_bounds(L, U, target);

  // Choose the extrapolation: fast Extra_LU+ for diagonal-free TA, sound per-pair
  // k-normalisation when any guard/invariant compares two clocks.
  Extrap ex;
  ex.L = &L; ex.U = &U;
  for (const auto& l : ta.locations) if (has_diagonal(l.invariant)) ex.diagonal = true;
  for (const auto& e : ta.edges) if (has_diagonal(e.guard)) ex.diagonal = true;
  if (has_diagonal(target)) ex.diagonal = true;

  const int nloc = static_cast<int>(ta.locations.size());
  // Passed list per location: zone-inclusion (subsumption) for pruning power, plus
  // an exact-DBM hash set as an O(1) fast path for re-visits (avoids the O(N)
  // inclusion scan for exact duplicates, which dominate).
  std::vector<std::vector<Dbm>> pz(nloc);
  std::vector<std::unordered_set<std::string>> pk(nloc);
  auto key = [](const Dbm& z) {
    return std::string(reinterpret_cast<const char*>(z.data().data()),
                       z.data().size() * sizeof(raw_t));
  };
  auto is_target = [&](int loc, const Dbm& z) {
    if (target_loc >= 0 && loc != target_loc) return false;
    Dbm t = z;
    return apply(t, target);
  };
  auto seen = [&](int loc, const Dbm& z) {
    if (pk[loc].count(key(z))) return true;      // exact duplicate (O(1))
    for (const auto& zp : pz[loc])
      if (zp.includes(z)) return true;           // subsumed by a larger zone (O(N))
    return false;
  };
  auto add = [&](int loc, const Dbm& z) { pk[loc].insert(key(z)); pz[loc].push_back(z); };

  // Safety cap on the DIAGONAL path only (diagonal-free terminates via Extra_LU+). An
  // unbounded-clock diagonal model has no finite exact zone graph — give up loudly
  // instead of looping. Diagonal-free models are never capped.
  constexpr std::size_t kStateCap = 40000;
  std::size_t nstored = 0;
  auto cap = [&] {
    if (ex.diagonal && ++nstored > kStateCap)
      throw std::runtime_error("state cap exceeded — model likely has unbounded diagonal clocks");
  };

  Dbm z0(ta.nclocks);
  if (!initial_zone(ta, z0, ex)) return false;
  if (is_target(ta.initial, z0)) return true;
  add(ta.initial, z0);
  if (stats) ++stats->explored;

#ifndef PTCTLMC_HAVE_OPENMP
  parallel = false;
#endif

  if (!parallel) {
    std::deque<std::pair<int, Dbm>> waiting;
    waiting.emplace_back(ta.initial, z0);
    while (!waiting.empty()) {
      auto st = std::move(waiting.front());
      waiting.pop_front();
      std::vector<std::pair<int, Dbm>> succ;
      successors(ta, st.first, st.second, succ, ex);
      for (auto& s : succ) {
        if (stats) ++stats->generated;
        if (is_target(s.first, s.second)) return true;
        if (seen(s.first, s.second)) continue;
        add(s.first, s.second);
        if (stats) ++stats->explored;
        cap();
        waiting.push_back(std::move(s));
      }
    }
    return false;
  }

#ifdef PTCTLMC_HAVE_OPENMP
  const int nthreads = omp_get_max_threads();
  std::vector<std::pair<int, Dbm>> frontier;
  frontier.emplace_back(ta.initial, z0);
  while (!frontier.empty()) {
    std::vector<std::vector<std::pair<int, Dbm>>> tls(nthreads);
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      auto& buf = tls[tid];
#pragma omp for schedule(dynamic, 8)
      for (long i = 0; i < static_cast<long>(frontier.size()); ++i)
        successors(ta, frontier[i].first, frontier[i].second, buf, ex);
    }
    std::vector<std::pair<int, Dbm>> next;
    for (auto& buf : tls)
      for (auto& s : buf) {
        if (stats) ++stats->generated;
        if (is_target(s.first, s.second)) return true;
        if (seen(s.first, s.second)) continue;
        add(s.first, s.second);
        if (stats) ++stats->explored;
        cap();
        next.push_back(std::move(s));
      }
    frontier.swap(next);
  }
  return false;
#endif
}

}  // namespace ptctlmc
