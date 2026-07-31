#include "ptctlmc/reach.hpp"

#include <atomic>
#include <cstdlib>
#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include "ptctlmc/cuda_reach.hpp"
#include "ptctlmc/metal_reach.hpp"

#ifdef PTCTLMC_HAVE_OPENMP
#include <omp.h>
#endif

namespace ptctlmc {

static int g_omp_chunk = 8;  // tunable via set_omp_chunk() / --omp-chunk CLI flag
void set_omp_chunk(int chunk) { if (chunk > 0) g_omp_chunk = chunk; }
int omp_chunk() { return g_omp_chunk; }

namespace {

bool apply_cs(Dbm& z, const std::vector<Constraint>& cs) {
  for (const auto& c : cs)
    if (!z.constrain(c.i, c.j, c.bound)) return false;
  return true;
}

void delay_in(Dbm& z, const Location& loc) {
  if (loc.urgent || loc.committed) return;  // no time may pass
  z.delay();
  apply_cs(z, loc.invariant);
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
  void apply_cs(Dbm& z) const {
    if (diagonal) z.extrapolate_lu_plus_nodiag(*L, *U);  // keep diagonals exact
    else z.extrapolate_lu_plus(*L, *U);
  }
};

// Any difference constraint x_i - x_j with i>0 && j>0 is a "diagonal".
bool has_diagonal(const std::vector<Constraint>& cs) {
  for (const auto& c : cs) if (c.i > 0 && c.j > 0) return true;
  return false;
}

// --- diagonal splitting: rewrite a TA with diagonal guards into an equivalent
// diagonal-FREE one, so Extra_LU+ / aLU is sound and exploration terminates. Each
// distinct diagonal atom d ≡ (x_i - x_j ~ c) becomes a boolean recorded in the
// location: since x_i - x_j is constant except at resets, d's truth only changes when
// x_i or x_j is reset, and the new truth is a NON-diagonal condition on the surviving
// clock. Locations split into (loc, bitvector); reset edges split on those conditions.
struct SplitTA { TimedAutomaton ta; int nb = 1; std::vector<Constraint> atoms; };

int find_atom(const std::vector<Constraint>& D, const Constraint& c) {
  for (int k = 0; k < static_cast<int>(D.size()); ++k)
    if (D[k].i == c.i && D[k].j == c.j && D[k].bound == c.bound) return k;
  return -1;
}
bool diag_true_at_zero(const Constraint& c) {  // x_i - x_j = 0 satisfies (0 ~ value)?
  return raw_is_strict(c.bound) ? (0 < raw_bound(c.bound)) : (0 <= raw_bound(c.bound));
}

SplitTA split_diagonals(const TimedAutomaton& in, const std::vector<Constraint>& query) {
  SplitTA S;
  std::vector<Constraint>& D = S.atoms;
  auto add_atom = [&](const Constraint& c) { if (c.i > 0 && c.j > 0 && find_atom(D, c) < 0) D.push_back(c); };
  for (const auto& e : in.edges) for (const auto& c : e.guard) add_atom(c);
  for (const auto& l : in.locations) for (const auto& c : l.invariant) add_atom(c);
  for (const auto& c : query) add_atom(c);
  const int k = static_cast<int>(D.size());
  if (k == 0) { S.ta = in; S.nb = 1; return S; }
  if (k > 16) throw std::runtime_error("diagonal splitting: too many distinct diagonal atoms");
  const int nb = 1 << k;
  S.nb = nb;
  TimedAutomaton& out = S.ta;
  out.nclocks = in.nclocks; out.clock_names = in.clock_names;
  const int NL = static_cast<int>(in.locations.size());
  auto nloc = [&](int L, int b) { return L * nb + b; };
  auto valid = [&](int L, int b) -> bool {  // diagonal invariant atoms must hold in (L,b)
    for (const auto& c : in.locations[L].invariant)
      if (c.i > 0 && c.j > 0) { const int id = find_atom(D, c); if (id >= 0 && !(b & (1 << id))) return false; }
    return true;
  };
  out.locations.resize(static_cast<std::size_t>(NL) * nb);
  for (int L = 0; L < NL; ++L)
    for (int b = 0; b < nb; ++b) {
      Location loc;
      loc.name = in.locations[L].name; loc.urgent = in.locations[L].urgent; loc.committed = in.locations[L].committed;
      for (const auto& c : in.locations[L].invariant) if (!(c.i > 0 && c.j > 0)) loc.invariant.push_back(c);
      out.locations[nloc(L, b)] = std::move(loc);
    }
  int b0 = 0;
  for (int d = 0; d < k; ++d) if (diag_true_at_zero(D[d])) b0 |= (1 << d);
  out.initial = nloc(in.initial, b0);

  for (const auto& e : in.edges) {
    std::vector<Constraint> gnd;  // non-diagonal guard part
    int req = 0;                  // diagonal atoms the guard requires TRUE in the source
    for (const auto& c : e.guard) {
      if (c.i > 0 && c.j > 0) { const int id = find_atom(D, c); if (id >= 0) req |= (1 << id); }
      else gnd.push_back(c);
    }
    std::vector<char> rst(in.nclocks + 1, 0);
    for (int r : e.resets) rst[r] = 1;
    struct Rule { int type; Constraint cond; };  // 0 unchanged, 1 const-false, 2 const-true, 3 split
    std::vector<Rule> rule(k);
    std::vector<int> splitIdx;
    for (int d = 0; d < k; ++d) {
      const Constraint& c = D[d];
      const bool ri = rst[c.i], rj = rst[c.j];
      if (!ri && !rj) rule[d] = {0, {}};
      else if (ri && rj) rule[d] = {diag_true_at_zero(c) ? 2 : 1, {}};
      else if (ri) { rule[d] = {3, Constraint{0, c.j, c.bound}}; splitIdx.push_back(d); }   // -x_j ~ c
      else { rule[d] = {3, Constraint{c.i, 0, c.bound}}; splitIdx.push_back(d); }            //  x_i ~ c
    }
    const int ns = static_cast<int>(splitIdx.size());
    for (int bsrc = 0; bsrc < nb; ++bsrc) {
      if ((bsrc & req) != req || !valid(e.src, bsrc)) continue;
      for (int combo = 0; combo < (1 << ns); ++combo) {
        int btgt = 0;
        std::vector<Constraint> g = gnd;
        for (int d = 0; d < k; ++d) {
          if (rule[d].type == 0) { if ((bsrc >> d) & 1) btgt |= (1 << d); }
          else if (rule[d].type == 2) btgt |= (1 << d);
        }
        for (int s = 0; s < ns; ++s) {
          const int d = splitIdx[s];
          if ((combo >> s) & 1) { btgt |= (1 << d); g.push_back(rule[d].cond); }
          else g.push_back(negate(rule[d].cond));
        }
        if (!valid(e.tgt, btgt)) continue;
        Edge ne;
        ne.src = nloc(e.src, bsrc); ne.tgt = nloc(e.tgt, btgt);
        ne.guard = std::move(g); ne.resets = e.resets; ne.sync = e.sync;
        out.edges.push_back(std::move(ne));
      }
    }
  }
  out.compute_max_constants();
  return S;
}

bool initial_zone(const TimedAutomaton& ta, Dbm& z0, const Extrap& ex) {
  z0 = Dbm(ta.nclocks);
  if (!apply_cs(z0, ta.locations[ta.initial].invariant)) return false;
  delay_in(z0, ta.locations[ta.initial]);
  ex.apply_cs(z0);
  return true;
}

void successors(const TimedAutomaton& ta, int loc, const Dbm& z,
                std::vector<std::pair<int, Dbm>>& out, const Extrap& ex) {
  for (int ei : ta.edges_from(loc)) {
    const Edge& e = ta.edges[ei];
    Dbm z1 = z;
    if (!apply_cs(z1, e.guard)) continue;
    for (int r : e.resets) z1.reset(r);
    const Location& tl = ta.locations[e.tgt];
    if (!apply_cs(z1, tl.invariant)) continue;
    delay_in(z1, tl);
    ex.apply_cs(z1);
    if (!z1.empty()) out.emplace_back(e.tgt, std::move(z1));
  }
}

}  // namespace

bool reachable(const TimedAutomaton& ta, int target_loc, const std::vector<Constraint>& target,
               ReachStats* stats, Backend backend) {
  bool parallel = (backend == Backend::Omp);

  // Effective LU bounds = model bounds widened with the query's clock constants.
  std::vector<std::int32_t> L = ta.lbound, U = ta.ubound;
  extend_bounds(L, U, target);

  // Diagonal constraints make LU/aLU unsound. Rewrite the automaton diagonal-FREE by
  // splitting locations on the truth of each diagonal atom, then use fast Extra_LU+.
  bool diag = has_diagonal(target);
  for (const auto& l : ta.locations) if (has_diagonal(l.invariant)) diag = true;
  for (const auto& e : ta.edges) if (has_diagonal(e.guard)) diag = true;

  SplitTA split;
  const TimedAutomaton* Wp = &ta;
  int nb = 1;
  if (diag) {
    split = split_diagonals(ta, target);
    Wp = &split.ta;
    nb = split.nb;
    L = split.ta.lbound; U = split.ta.ubound;
    extend_bounds(L, U, target);
  }
  const TimedAutomaton& W = *Wp;
  Extrap ex;
  ex.L = &L; ex.U = &U; ex.diagonal = false;  // W is diagonal-free

  // Translate the target: location (÷ nb), non-diagonal clock atoms (checked on the
  // zone), and diagonal atoms (checked via the split location's boolean bit).
  std::vector<Constraint> tgt_clocks;
  std::vector<int> tgt_diag_bits;
  for (const auto& c : target) {
    const int id = (diag && c.i > 0 && c.j > 0) ? find_atom(split.atoms, c) : -1;
    if (id >= 0) tgt_diag_bits.push_back(id);
    else tgt_clocks.push_back(c);
  }

  const int nloc = static_cast<int>(W.locations.size());

  // GPU backends: run zone exploration over the (now diagonal-free) automaton W. Since
  // the target may be a *set* of split locations (original target_loc × diagonal-bit
  // combos), pass a per-location accept mask; the GPU then checks accept[loc] plus the
  // clock target on the zone — so diagonal models work on the GPU via the same split.
  if (backend == Backend::Cuda || backend == Backend::Metal) {
    std::vector<char> accept(nloc, 0);
    for (int loc = 0; loc < nloc; ++loc) {
      if (target_loc >= 0 && loc / nb != target_loc) continue;
      bool ok = true;
      for (int bit : tgt_diag_bits) if (!((loc % nb) & (1 << bit))) { ok = false; break; }
      accept[loc] = ok ? 1 : 0;
    }
    if (backend == Backend::Cuda) {
#ifdef PTCTLMC_HAVE_CUDA
      return reachable_cuda(W, accept, tgt_clocks, L, U, stats);
#else
      throw std::runtime_error("CUDA backend not available in this build");
#endif
    }
#ifdef PTCTLMC_HAVE_METAL
    return reachable_metal(W, accept, tgt_clocks, L, U, stats);
#else
    throw std::runtime_error("Metal backend not available in this build");
#endif
  }

  std::vector<std::vector<Dbm>> pz(nloc);
  std::vector<std::unordered_set<std::string>> pk(nloc);
  auto key = [](const Dbm& z) {
    return std::string(reinterpret_cast<const char*>(z.data().data()),
                       z.data().size() * sizeof(raw_t));
  };
  auto is_target = [&](int loc, const Dbm& z) {
    if (target_loc >= 0 && loc / nb != target_loc) return false;
    for (int bit : tgt_diag_bits) if (!((loc % nb) & (1 << bit))) return false;
    Dbm t = z;
    return apply_cs(t, tgt_clocks);
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

  Dbm z0(W.nclocks);
  if (!initial_zone(W, z0, ex)) return false;
  if (is_target(W.initial, z0)) return true;
  add(W.initial, z0);
  if (stats) ++stats->explored;

#ifndef PTCTLMC_HAVE_OPENMP
  parallel = false;
#endif

  if (!parallel) {
    std::deque<std::pair<int, Dbm>> waiting;
    waiting.emplace_back(W.initial, z0);
    while (!waiting.empty()) {
      auto st = std::move(waiting.front());
      waiting.pop_front();
      std::vector<std::pair<int, Dbm>> succ;
      successors(W, st.first, st.second, succ, ex);
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
  frontier.emplace_back(W.initial, z0);
  // hit: set by whichever thread first finds the target.
  // With OpenMP 4.0 cancel, other threads stop at their next iteration
  // boundary (cancellation point) — independent of chunk size.
  std::atomic<bool> hit{false};
  while (!frontier.empty()) {
    std::vector<std::vector<std::pair<int, Dbm>>> tls(nthreads);
#pragma omp parallel shared(hit)
    {
      const int tid = omp_get_thread_num();
      auto& buf = tls[tid];
#pragma omp for schedule(dynamic, g_omp_chunk)
      for (long i = 0; i < static_cast<long>(frontier.size()); ++i) {
        // Cancellation point: every thread checks here at each new iteration.
        // If any thread executed `omp cancel for` below, all others stop here
        // immediately — no need to finish the current chunk first.
#pragma omp cancellation point for
        const std::size_t prev = buf.size();
        successors(W, frontier[i].first, frontier[i].second, buf, ex);
        for (std::size_t j = prev; j < buf.size(); ++j) {
          if (is_target(buf[j].first, buf[j].second)) {
            hit.store(true, std::memory_order_relaxed);
            // Signal all threads to stop at their next cancellation point.
#pragma omp cancel for
            break;
          }
        }
      }
    }
    // Early exit: target was found during parallel expansion.
    if (hit.load()) return true;
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
