// env_quotient.cpp — measurement, not a unit test.
//
// Question (docs/superpowers/specs/weaksim-quotient.md, Job B): does the ENVIRONMENT of a
// Fischer-style network collapse under weak (bi)simulation once everything that is not an
// interaction with the property's processes C is internal? If the number of classes stops
// growing with the number of environment processes k, an `Ehat` can be built automatically and
// the environment-abstraction lead is alive.
//
// Build: cmake --build build-mostopt --target ptctlmc_envquot -j8
// Run:   ptctlmc_envquot sizes|quot|cmp ...   (see usage() at the bottom)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "ptctlmc/bisim.hpp"
#include "ptctlmc/fed.hpp"

using namespace ptctlmc;

// ── the model ───────────────────────────────────────────────────────────────────────
//
// gen_fischer.py, per process:  idle --[id==0]--> req(x<=K) --[id:=pid, x:=0]--> wait
//                               wait --[x>K && id==pid]--> cs --[id:=0]--> idle
//                               wait --[x>K && id!=pid]--> idle
//
// `id` is encoded as an explicit component of the location, with the value abstracted to
//   d ∈ {FREE} ∪ {OWN_0..OWN_{k-1}} ∪ {C}          (k+2 values)
// The environment's processes only ever test `id==0` / `id==own pid` and only ever write
// `id := own pid` / `id := 0`, so the identity of the *C*-side writer is irrelevant to them:
// every value a C process can write is "not 0 and not mine". Which E process owns the lock
// must be kept, because only the owner may enter cs. This is the {undefined, mine, someone
// else's} abstraction, refined by the owner's identity.
//
// C touches `id` with exactly five operations, and those are the interface:
//   C_RD0  read id==0        enabled iff d==FREE      (C's idle->req guard)
//   C_WR   write id:=cpid    always enabled           (C's req->wait assignment — unguarded)
//   C_RDEQ read id==cpid     enabled iff d==C         (C's wait->cs guard)
//   C_RDNE read id!=cpid     enabled iff d!=C         (C's wait->idle guard)
//   C_WR0  write id:=0       always enabled           (C's cs->idle assignment — unguarded)
// None of them carries a clock guard, which is exactly right: C's clock lives in C, so the
// environment must tolerate C touching `id` at any moment.

enum : int { IDLE = 0, REQ = 1, WAIT = 2, CS = 3 };
enum : int {
  C_RD0 = 1, C_WR = 2, C_RDEQ = 3, C_RDNE = 4, C_WR0 = 5,      // interface with C
  E_RD0 = 11, E_WR = 12, E_RDEQ = 13, E_RDNE = 14, E_WR0 = 15  // the environment's own
};

// Observable alphabets, from most to least abstract.
//  V1 "C-view"   : only C's own operations. E's accesses to `id` are internal, because C can
//                  see them only by performing one of its own reads later.
//  V2 "writes"   : plus E's writes (all E writes carry ONE label — C cannot tell which E
//                  process wrote). E's reads stay internal: they do not change shared state.
//  V3 "literal"  : every read and write of `id` observable — the literal reading of the brief.
//                  Every edge of the model is then observable, so weak == strong here.
static std::set<int> alphabet(int v) {
  std::set<int> s{C_RD0, C_WR, C_RDEQ, C_RDNE, C_WR0};
  if (v >= 2) { s.insert(E_WR); s.insert(E_WR0); }
  if (v >= 3) { s.insert(E_RD0); s.insert(E_RDEQ); s.insert(E_RDNE); }
  return s;
}

// Labels for the CONTEXT-CONSTRAINED variant (mode `ctx`), where the C side is not an
// unconstrained input environment but genuine Fischer processes whose own edges carry the
// observable labels. This is the difference between "minimise E for EVERY context" (modes
// `quot`/`cmp`, which is what bisimulation-as-a-congruence buys you) and "minimise E for the
// context it actually has".
static int obs_label(int proc, int kind) { return 1000 + proc * 16 + kind; }

static int ipow4(int e) { int r = 1; while (e--) r *= 4; return r; }

struct Env {
  BAut aut;
  int k = 0, K = 2, nd = 0;
  int qpart(int loc, int i) const { return (loc / nd) / ipow4(i) % 4; }
  int dpart(int loc) const { return loc % nd; }
  int mk(int q, int d) const { return q * nd + d; }
  std::string name(int loc) const {
    static const char* qn[4] = {"idle", "req", "wait", "cs"};
    std::string s = "(";
    for (int i = 0; i < k; ++i) { if (i) s += ","; s += qn[qpart(loc, i)]; }
    const int d = dpart(loc);
    s += "|id=";
    s += (d == 0 ? "free" : (d == nd - 1 ? "C" : "own" + std::to_string(d - 1)));
    return s + ")";
  }
};

static Env make_env(int k, int K) {
  Env E; E.k = k; E.K = K; E.nd = k + 2;
  const int nq = ipow4(k);
  BAut& A = E.aut;
  A.nclocks = k;
  A.nlocs = nq * E.nd;
  A.initial = E.mk(0, 0);  // every process idle, id free
  A.invariants.assign(A.nlocs, {});
  for (int loc = 0; loc < A.nlocs; ++loc) {
    const int q = loc / E.nd, d = E.dpart(loc);
    for (int i = 0; i < k; ++i)
      if (E.qpart(loc, i) == REQ) A.invariants[loc].push_back(c_le(i + 1, K));
    const int OWN = 1;  // d == i+1 means process i owns
    for (int i = 0; i < k; ++i) {
      const int qi = E.qpart(loc, i), step = ipow4(i);
      const int clk = i + 1;
      auto setq = [&](int nq2) { return E.mk(q + (nq2 - qi) * step, d); };
      if (qi == IDLE && d == 0)  // idle -> req, guard id==0, x:=0
        A.edges.push_back({loc, setq(REQ), {}, {clk}, E_RD0});
      if (qi == REQ)  // req -> wait, id:=pid, x:=0   (no clock guard; invariant x<=K forces it)
        A.edges.push_back({loc, E.mk(q + (WAIT - qi) * step, i + OWN), {}, {clk}, E_WR});
      if (qi == WAIT && d == i + OWN)  // wait -> cs, x>K && id==pid
        A.edges.push_back({loc, setq(CS), {c_gt(clk, K)}, {}, E_RDEQ});
      if (qi == WAIT && d != i + OWN)  // wait -> idle, x>K && id!=pid
        A.edges.push_back({loc, setq(IDLE), {c_gt(clk, K)}, {}, E_RDNE});
      if (qi == CS)  // cs -> idle, id:=0
        A.edges.push_back({loc, E.mk(q + (IDLE - qi) * step, 0), {}, {}, E_WR0});
    }
    // C's five operations on `id`.
    if (d == 0) A.edges.push_back({loc, loc, {}, {}, C_RD0});
    A.edges.push_back({loc, E.mk(q, E.nd - 1), {}, {}, C_WR});
    if (d == E.nd - 1) A.edges.push_back({loc, loc, {}, {}, C_RDEQ});
    else A.edges.push_back({loc, loc, {}, {}, C_RDNE});
    A.edges.push_back({loc, E.mk(q, 0), {}, {}, C_WR0});
  }
  return E;
}

// ── context-constrained variant ─────────────────────────────────────────────────────
//
// `nobs` leading Fischer processes ARE the C side: their own edges carry the observable
// labels, so the only things C can do to `id` are the things a Fischer process can do, at the
// times a Fischer process can do them. The remaining `nenv` processes are the environment and
// are entirely internal. `id`'s value is the owning process index (0 = free) — the C side's
// guards need to tell "owned by me" from "owned by anyone else", so ownership identity is kept
// for every process.
struct Sys {
  BAut aut;
  int nobs = 0, nenv = 0, nproc = 0, nd = 0, K = 2;
};

// `strict` selects Fischer's real `wait --[x>K]--> cs` guard; `false` weakens it to `x>=K`,
// which is the textbook way to break the protocol and is used only to show that the
// mutual-exclusion validation below can fail.
static Sys make_sys(int nobs, int nenv, int K, bool strict = true) {
  Sys S; S.nobs = nobs; S.nenv = nenv; S.nproc = nobs + nenv; S.nd = S.nproc + 1; S.K = K;
  const int nq = ipow4(S.nproc);
  BAut& A = S.aut;
  A.nclocks = S.nproc;
  A.nlocs = nq * S.nd;
  A.initial = 0;
  A.invariants.assign(A.nlocs, {});
  auto mk = [&](int q, int d) { return q * S.nd + d; };
  for (int loc = 0; loc < A.nlocs; ++loc) {
    const int q = loc / S.nd, d = loc % S.nd;
    for (int i = 0; i < S.nproc; ++i) {
      const int qi = (q / ipow4(i)) % 4, step = ipow4(i), clk = i + 1, own = i + 1;
      if (qi == REQ) A.invariants[loc].push_back(c_le(clk, K));
      auto lab = [&](int kind) { return i < nobs ? obs_label(i, kind) : 10 + kind; };
      auto setq = [&](int nq2) { return mk(q + (nq2 - qi) * step, d); };
      if (qi == IDLE && d == 0) A.edges.push_back({loc, setq(REQ), {}, {clk}, lab(1)});
      if (qi == REQ) A.edges.push_back({loc, mk(q + (WAIT - qi) * step, own), {}, {clk}, lab(2)});
      const Constraint past = strict ? c_gt(clk, K) : c_ge(clk, K);
      if (qi == WAIT && d == own) A.edges.push_back({loc, setq(CS), {past}, {}, lab(3)});
      if (qi == WAIT && d != own) A.edges.push_back({loc, setq(IDLE), {past}, {}, lab(4)});
      if (qi == CS) A.edges.push_back({loc, mk(q + (IDLE - qi) * step, 0), {}, {}, lab(5)});
      // The safety property is a predicate on C's LOCATIONS (`A[] not (P1.cs && P2.cs)`), and
      // a simulation preserves state predicates only when they are observable. Expose each C
      // process's local state as an observable self-loop, i.e. keep the C side fully concrete.
      if (i < nobs) A.edges.push_back({loc, loc, {}, {}, obs_label(i, 8 + qi)});
    }
  }
  return S;
}

static std::set<int> ctx_alphabet(int nobs, int av) {
  std::set<int> s;
  for (int p = 0; p < nobs; ++p)
    for (int kind = 1; kind <= 11; ++kind) s.insert(obs_label(p, kind));
  if (av >= 2) { s.insert(12); s.insert(15); }                       // env writes
  if (av >= 3) { s.insert(11); s.insert(13); s.insert(14); }         // env reads
  return s;
}

// ── forward zone graph (for reachability and for the symbolic state count) ───────────
static std::vector<std::int32_t> maxconst(const BAut& A) {
  std::vector<std::int32_t> M(A.nclocks + 1, 0);
  auto bump = [&](const Constraint& c) {
    const int v = std::abs(raw_bound(c.bound));
    if (c.i > 0 && v > M[c.i]) M[c.i] = v;
    if (c.j > 0 && v > M[c.j]) M[c.j] = v;
  };
  for (const auto& e : A.edges) for (const auto& c : e.guard) bump(c);
  for (const auto& iv : A.invariants) for (const auto& c : iv) bump(c);
  return M;
}

struct ZoneGraph { std::vector<Federation> per_loc; std::size_t nstates = 0, nreach_locs = 0; };

static ZoneGraph explore(const BAut& A) {
  const auto M = maxconst(A);
  std::vector<std::vector<int>> out(A.nlocs);
  for (int i = 0; i < static_cast<int>(A.edges.size()); ++i) out[A.edges[i].src].push_back(i);
  auto apply_inv = [&](Dbm& z, int l) {
    for (const auto& c : A.invariants[l]) if (!z.constrain(c.i, c.j, c.bound)) return false;
    return !z.empty();
  };
  ZoneGraph G; G.per_loc.assign(A.nlocs, {});
  std::vector<std::pair<int, Dbm>> work;
  {
    Dbm z = Dbm::zero(A.nclocks);
    if (apply_inv(z, A.initial)) {
      z.delay();
      if (apply_inv(z, A.initial)) {
        dbm_extrapolate(z.data().data(), z.dim(), M.data());
        G.per_loc[A.initial].push_back(z);
        work.push_back({A.initial, z});
      }
    }
  }
  while (!work.empty()) {
    auto [l, Z] = work.back(); work.pop_back();
    for (int ei : out[l]) {
      const BEdge& e = A.edges[ei];
      Dbm z = Z;
      bool ok = true;
      for (const auto& c : e.guard) if (!z.constrain(c.i, c.j, c.bound)) { ok = false; break; }
      if (!ok || z.empty()) continue;
      for (int r : e.resets) z.reset(r);
      if (!apply_inv(z, e.dst)) continue;
      z.delay();
      if (!apply_inv(z, e.dst)) continue;
      dbm_extrapolate(z.data().data(), z.dim(), M.data());
      if (fed_covers(G.per_loc[e.dst], z)) continue;
      fed_add(G.per_loc[e.dst], z);
      work.push_back({e.dst, z});
    }
  }
  for (const auto& f : G.per_loc) { G.nstates += f.size(); if (!f.empty()) G.nreach_locs++; }
  return G;
}

// ── the class count ─────────────────────────────────────────────────────────────────
//
// The engine returns R[(l,l')] as a federation over the DOUBLED clock set (A's clocks are
// 1..k, B's are k+1..2k). Two locations of the same automaton denote the same behaviour when
// some renaming of the clocks makes every valuation related to its own image, both ways:
//
//   l ~ l'   iff   ∃ σ ∈ S_k .  D_σ ⊆ R[(l,l')]  and  D_{σ⁻¹} ⊆ R[(l',l)],
//   D_σ = { (v,w) : w_{σ(i)} = v_i }  ∩  Inv(l) ∩ Inv(l')
//
// The permutation is what lets "process 1 is waiting" merge with "process 2 is waiting"; the
// engine's relation already contains every matching of the two clock sets, we only have to
// ask for the right one. Classes are the blocks of the equivalence generated by ~.
//
// The intersection with the two invariants is not cosmetic: valuations outside Inv(l) are not
// states of l, so demanding that they be related would make no two locations with different
// invariants ever merge.
static Dbm diag_zone(int k, const std::vector<int>& sigma) {
  Dbm z = Dbm::universe(2 * k);
  for (int i = 0; i < k; ++i) {
    const int a = i + 1, b = k + sigma[i] + 1;
    z.constrain(a, b, kZeroLe);
    z.constrain(b, a, kZeroLe);
  }
  return z;
}

struct UF {
  std::vector<int> p;
  explicit UF(int n) : p(n) { std::iota(p.begin(), p.end(), 0); }
  int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
  void join(int a, int b) { a = find(a); b = find(b); if (a != b) p[a] = b; }
};

int main(int argc, char** argv) {
  auto usage = [] {
    std::puts("usage:\n"
              "  ptctlmc_envquot sizes  <kmax> [K]\n"
              "  ptctlmc_envquot quot   <k> <alphabet 1|2|3> <weak|strong> [K]\n"
              "  ptctlmc_envquot cmp    <k1> <k2> <alphabet 1|2|3> <weak|strong> [K]\n"
              "  ptctlmc_envquot ctx    <nobs> <k1> <k2> <alphabet 1|2|3> <weak|strong> [K]\n");
    return 2;
  };
  if (argc < 3) return usage();
  const std::string mode = argv[1];
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto secs = [](auto a, auto b) {
    return std::chrono::duration<double>(b - a).count();
  };

  if (mode == "sizes") {
    const int kmax = std::atoi(argv[2]);
    const int K = argc > 3 ? std::atoi(argv[3]) : 2;
    std::printf("%3s %8s %8s %10s %12s %8s %8s\n", "k", "locs", "edges", "reach-locs",
                "symb-states", "orbits", "sec");
    for (int k = 1; k <= kmax; ++k) {
      Env E = make_env(k, K);
      auto t0 = now();
      ZoneGraph G = explore(E.aut);
      // Symmetry orbits of the reachable locations under permutation of the k identical
      // environment processes. Exactly one process can own `id`, so a canonical form is
      // (kind of d, the owner's local state, the sorted multiset of the others' local states).
      std::vector<std::vector<int>> keys;
      for (int loc = 0; loc < E.aut.nlocs; ++loc) {
        if (G.per_loc[loc].empty()) continue;
        const int d = E.dpart(loc);
        const int dkind = (d == 0) ? 0 : (d == E.nd - 1 ? 2 : 1);
        const int owner = (dkind == 1) ? d - 1 : -1;
        std::vector<int> key{dkind, owner >= 0 ? E.qpart(loc, owner) : -1};
        std::vector<int> rest;
        for (int i = 0; i < k; ++i) if (i != owner) rest.push_back(E.qpart(loc, i));
        std::sort(rest.begin(), rest.end());
        key.insert(key.end(), rest.begin(), rest.end());
        keys.push_back(std::move(key));
      }
      std::sort(keys.begin(), keys.end());
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      std::printf("%3d %8d %8zu %10zu %12zu %8zu %8.2f\n", k, E.aut.nlocs, E.aut.edges.size(),
                  G.nreach_locs, G.nstates, keys.size(), secs(t0, now()));
      std::fflush(stdout);
    }
    return 0;
  }

  if (mode == "quot") {
    if (argc < 5) return usage();
    const int k = std::atoi(argv[2]);
    const int av = std::atoi(argv[3]);
    const bool weak = std::string(argv[4]) == "weak";
    const int K = argc > 5 ? std::atoi(argv[5]) : 2;
    Env E = make_env(k, K);
    ZoneGraph G = explore(E.aut);
    std::printf("k=%d K=%d alphabet=V%d %s : locs=%d reach-locs=%zu symb-states=%zu\n", k, K, av,
                weak ? "weak" : "strong", E.aut.nlocs, G.nreach_locs, G.nstates);
    std::fflush(stdout);
    auto t0 = now();
    bool self = false;
    std::vector<Federation> R = bisim_relation(
        E.aut, E.aut, weak ? BisimKind::WeakBisim : BisimKind::StrongBisim, alphabet(av), &self);
    const double tfix = secs(t0, now());
    std::printf("  fixpoint: %.2f s   (self-bisimilar sanity: %s)\n", tfix, self ? "yes" : "NO");
    std::fflush(stdout);
    if (!self) { std::puts("  ABORT: the automaton is not bisimilar to itself — engine bug."); return 1; }

    std::vector<int> reach;
    for (int l = 0; l < E.aut.nlocs; ++l) if (!G.per_loc[l].empty()) reach.push_back(l);
    std::vector<int> perm(k);
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<std::pair<std::vector<int>, Dbm>> diags;
    do { diags.push_back({perm, diag_zone(k, perm)}); } while (std::next_permutation(perm.begin(), perm.end()));

    // invariants lifted into the doubled clock space, once for each side.
    auto lift_inv = [&](int l, int shift) {
      Dbm z = Dbm::universe(2 * k);
      for (const auto& c : E.aut.invariants[l])
        z.constrain(c.i == 0 ? 0 : c.i + shift, c.j == 0 ? 0 : c.j + shift, c.bound);
      return z;
    };
    std::vector<Dbm> ivA(E.aut.nlocs), ivB(E.aut.nlocs);
    for (int l = 0; l < E.aut.nlocs; ++l) { ivA[l] = lift_inv(l, 0); ivB[l] = lift_inv(l, k); }

    auto t1 = now();
    UF uf(E.aut.nlocs);
    std::size_t merges = 0;
    for (std::size_t a = 0; a < reach.size(); ++a)
      for (std::size_t b = a + 1; b < reach.size(); ++b) {
        const int l = reach[a], m = reach[b];
        if (uf.find(l) == uf.find(m)) continue;
        for (const auto& [sg, D] : diags) {
          std::vector<int> inv(k);
          for (int i = 0; i < k; ++i) inv[sg[i]] = i;
          Dbm Dlm = D;   // states of l, matched to their image in m
          Dlm.intersect(ivA[l]); Dlm.intersect(ivB[m]);
          Dbm Dml = diag_zone(k, inv);
          Dml.intersect(ivA[m]); Dml.intersect(ivB[l]);
          if (Dlm.empty() || Dml.empty()) continue;
          if (fed_covers(R[static_cast<std::size_t>(l) * E.aut.nlocs + m], Dlm) &&
              fed_covers(R[static_cast<std::size_t>(m) * E.aut.nlocs + l], Dml)) {
            uf.join(l, m); merges++; break;
          }
        }
      }
    std::vector<int> roots;
    for (int l : reach) roots.push_back(uf.find(l));
    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
    std::printf("  CLASSES(reachable locations) = %zu   (from %zu locations, %zu merges, %.2f s)\n",
                roots.size(), reach.size(), merges, secs(t1, now()));
    for (int r : roots) {
      std::size_t n = 0;
      int rep = -1;
      for (int l : reach) if (uf.find(l) == r) { n++; if (rep < 0) rep = l; }
      std::printf("    class %-4d size %-5zu rep %s\n", r, n, E.name(rep).c_str());
    }
    return 0;
  }

  if (mode == "mutex") {  // mutex <nmax> [K] — validate the flattened model against Fischer's
    const int nmax = std::atoi(argv[2]);           // own correctness argument, both ways round.
    const int K = argc > 3 ? std::atoi(argv[3]) : 2;
    std::printf("%3s %6s %8s %10s %12s %10s %10s\n", "N", "locs", "edges", "reach-locs",
                "symb(Extra_M)", "2-in-cs", "2-in-cs?x>=K");
    for (int n = 2; n <= nmax; ++n) {
      int bad[2] = {0, 0};
      std::size_t symb = 0, rl = 0, nl = 0, ne = 0;
      for (int variant = 0; variant < 2; ++variant) {
        Sys S = make_sys(0, n, K, variant == 0);
        ZoneGraph G = explore(S.aut);
        for (int loc = 0; loc < S.aut.nlocs; ++loc) {
          if (G.per_loc[loc].empty()) continue;
          int incs = 0;
          for (int i = 0; i < n; ++i) if ((loc / S.nd / ipow4(i)) % 4 == CS) incs++;
          if (incs >= 2) bad[variant]++;
        }
        if (variant == 0) { symb = G.nstates; rl = G.nreach_locs; nl = S.aut.nlocs; ne = S.aut.edges.size(); }
      }
      std::printf("%3d %6zu %8zu %10zu %12zu %10d %10d\n", n, nl, ne, rl, symb, bad[0], bad[1]);
      std::fflush(stdout);
    }
    return 0;
  }

  if (mode == "ctx") {  // ctx <nobs> <k1> <k2> <alphabet> <weak|strong> [K]
    if (argc < 7) return usage();
    const int nobs = std::atoi(argv[2]), k1 = std::atoi(argv[3]), k2 = std::atoi(argv[4]);
    const int av = std::atoi(argv[5]);
    const bool weak = std::string(argv[6]) == "weak";
    const int K = argc > 7 ? std::atoi(argv[7]) : 2;
    Sys A = make_sys(nobs, k1, K), B = make_sys(nobs, k2, K);
    ZoneGraph GA = explore(A.aut), GB = explore(B.aut);
    std::printf("ctx nobs=%d : C||E_%d (locs=%d, symb=%zu) vs C||E_%d (locs=%d, symb=%zu), V%d %s\n",
                nobs, k1, A.aut.nlocs, GA.nstates, k2, B.aut.nlocs, GB.nstates, av,
                weak ? "weak" : "strong");
    std::fflush(stdout);
    auto t0 = now();
    const bool sim = decide_bisim(A.aut, B.aut, weak ? BisimKind::WeakSim : BisimKind::StrongSim,
                                  ctx_alphabet(nobs, av));
    std::printf("  C||E_%d <= C||E_%d : %s   (%.2f s)\n", k1, k2, sim ? "YES" : "no",
                secs(t0, now()));
    std::fflush(stdout);
    auto t1 = now();
    const bool bi = decide_bisim(A.aut, B.aut, weak ? BisimKind::WeakBisim : BisimKind::StrongBisim,
                                 ctx_alphabet(nobs, av));
    std::printf("  C||E_%d ~= C||E_%d : %s   (%.2f s)\n", k1, k2, bi ? "YES" : "no",
                secs(t1, now()));
    return 0;
  }

  if (mode == "cmp") {
    if (argc < 6) return usage();
    const int k1 = std::atoi(argv[2]), k2 = std::atoi(argv[3]);
    const int av = std::atoi(argv[4]);
    const bool weak = std::string(argv[5]) == "weak";
    const int K = argc > 6 ? std::atoi(argv[6]) : 2;
    Env A = make_env(k1, K), B = make_env(k2, K);
    std::printf("E_%d (locs=%d) vs E_%d (locs=%d), alphabet V%d, %s\n", k1, A.aut.nlocs, k2,
                B.aut.nlocs, av, weak ? "weak" : "strong");
    std::fflush(stdout);
    auto t0 = now();
    const bool sim =
        decide_bisim(A.aut, B.aut, weak ? BisimKind::WeakSim : BisimKind::StrongSim, alphabet(av));
    std::printf("  E_%d <= E_%d : %s   (%.2f s)\n", k1, k2, sim ? "YES" : "no", secs(t0, now()));
    std::fflush(stdout);
    auto t1 = now();
    const bool bi = decide_bisim(A.aut, B.aut, weak ? BisimKind::WeakBisim : BisimKind::StrongBisim,
                                 alphabet(av));
    std::printf("  E_%d ~= E_%d : %s   (%.2f s)\n", k1, k2, bi ? "YES" : "no", secs(t1, now()));
    return 0;
  }
  return usage();
}
