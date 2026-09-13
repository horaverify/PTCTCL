/// @file src/liveness/fixpoint.cpp
/// @brief The backward greatest fixpoint, and the verdict read from it.
///
/// Solves `W = W /\\ (NoSucc \\/ Pre(W))` once, confined to a node set. Nodes the graph alone
/// proves empty never enter the sweep.

#include "hora/zone.hpp"
#include "hora/detail/liveness_engine.hpp"
#ifdef HORA_HAVE_OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#include "hdbm/mindbm.hpp"
#include "hora/detail/semantics.hpp"
#if defined(__APPLE__)
#endif
#include "src/liveness/internal.hpp"

namespace hora {

namespace {
// HORA_OVL: census of accepted-but-inexact covers, split by whether X meets its coverer.
const bool g_ovl = std::getenv("HORA_OVL") != nullptr;
std::size_t g_ovl_disj = 0, g_ovl_part = 0, g_ovl_disj_pieces = 0, g_ovl_part_pieces = 0;
struct OvlReport { ~OvlReport() { if (g_ovl) std::fprintf(stderr,
    "OVERLAP disjoint_covers=%zu (debt %zu pieces) | overlapping_covers=%zu (debt %zu pieces)\n",
    g_ovl_disj, g_ovl_disj_pieces, g_ovl_part, g_ovl_part_pieces); } };
OvlReport g_ovl_report;
}  // namespace

void LivenessEngine::solve_greatest_fixpoint(const std::vector<int>& S, std::vector<Federation>& W) {
  ++passes;
  const std::size_t N = nrec();
  std::vector<char> inS(N, 0);
  for (int v : S) inS[v] = 1;
  W.assign(N, Federation{});
  std::vector<Federation> nosucc(N);
  // the seed IS NOT materialised. Every node of S starts at its own zone, and writing all |S|
  // of those copies down before the sweep begins is the peak of the entire run -- one full
  // dense DBM per node, held simultaneously, on top of the copy already in `node`. `seeded[v]`
  // stands in for that copy: it means "W[v] is logically { node[v].zone }". The sweep
  // materialises a seed only for the node it is processing, and replaces it with `W[v] /\ cand`
  // in the same step, so at most one seed is ever live. On a satisfied query that refinement is
  // already draining towards empty, which is why the peak never re-forms.
  //
  // A seed is never empty: ::LivenessEngine::intern stores no empty zone, so `seeded[v]` implies non-empty
  // and every emptiness test below can short-circuit on it.
  std::vector<char> seeded(N, 0);
  for (int v : S) seeded[v] = 1;
  // Gated on the carrier flags: ::LivenessEngine::carrier_of derives its bounds per node only on
  // that route, and ::LivenessEngine::eg_probe calls this solver with both clear, where the
  // ambient bL/bU it would otherwise read are unconfigured scratch.
  const bool opt_mode = std::getenv("HORA_OPT") != nullptr && (fc_mode_ || carrier_mode_);
  std::vector<char> dbg_hadextra(N, 0);
  // --- SCC identity, for the cycle-only restriction below --------------------------------
  //
  // The operator is `W[v] /\ (NoSucc(v) \/ Pre(W))`, and `NoSucc(v)` is computed exactly on Z_v
  // and enters as a UNION term, so it survives every refinement: a finite (deadlock) or
  // time-divergent witness at a node can never be refined away. Reading the whole closure
  // (::LivenessEngine::eg_verdict, HORA_ANYNODE) rather than node 0 alone, such a witness is
  // therefore never lost, whatever covering discarded -- the prefix to that node need not
  // survive, because every node of ::LivenessEngine::eg_closure is psi-reachable by construction.
  //
  // So the discarded successors are owed to CYCLE witnesses only. A cycle lies inside a single
  // strongly connected component, so every step it takes is an intra-component edge, and an extra
  // on an edge LEAVING a component can never be part of one. Those are dropped here.
  //
  // Sound because dropping seed material only shrinks the gfp; complete because what is dropped
  // provably cannot carry the only witness kind that needs it. Requires the closure readout: with
  // the node-0 readout the prefix matters and this restriction would be unsound.
  std::vector<Federation> extra;      // per node, the discarded successors it is owed
  std::vector<int> sccid(N, -1);
  const bool cycle_only = std::getenv("HORA_CYCLE_EXTRAS") != nullptr;
  if (cycle_only && (fc_mode_ || carrier_mode_)) {
    std::vector<int> idx(N, -1), low(N, 0), stk, path;
    std::vector<char> on(N, 0);
    int next = 0, ncomp = 0;
    for (int r : S) {
      if (idx[r] != -1) continue;
      path.assign(1, r);
      std::vector<std::size_t> it(1, 0);
      idx[r] = low[r] = next++; stk.push_back(r); on[r] = 1;
      while (!path.empty()) {
        const int v = path.back();
        if (it.back() < adj[v].size()) {
          const int w = adj[v][it.back()++].tgt;
          if (!inS[w]) continue;
          if (idx[w] == -1) {
            idx[w] = low[w] = next++; stk.push_back(w); on[w] = 1;
            path.push_back(w); it.push_back(0);
          } else if (on[w]) {
            low[v] = std::min(low[v], idx[w]);
          }
        } else {
          if (low[v] == idx[v]) {
            for (;;) { const int u = stk.back(); stk.pop_back(); on[u] = 0; sccid[u] = ncomp;
                       if (u == v) break; }
            ++ncomp;
          }
          path.pop_back(); it.pop_back();
          if (!path.empty()) low[path.back()] = std::min(low[path.back()], low[v]);
        }
      }
    }
  }
  // --- carrier seeds: the EXACT discarded successors ------------------------------------
  //
  // A covered edge `v --e--> j` points at the stored `Z_j`, which need not contain the exact
  // successor `X = post_e(Z_v)` the cover discarded -- covering is a simulation, not inclusion.
  // Backward `pre` run against `Z_j` alone therefore cannot see `X`, and a real infinite witness
  // through `X` is lost. The carrier is the set that repairs this: it must satisfy
  //
  //     X subseteq C_j   for every accepted cover into j.                            (Lemma A)
  //
  // `Extra_LU+(Z_j)` is *a* convex superset satisfying that, and was what this seeded before.
  // It is far larger than the obligation requires, and the cost is not academic: it widens EVERY
  // node (measured: 1152 of 1152 on arbiter8), and a wide seed drains by peeling slivers rather
  // than collapsing -- 83 refinements per node against the exact seed's 1.0, with the union in
  // `relax` never covering the value so the early exit dies too.
  //
  // The MINIMAL set satisfying Lemma A is the obligation itself:
  //
  //     C_j = Z_j  union  { post_e(Z_v) : v --e--> j accepted, X not subseteq Z_j }.
  //
  // It is a federation, which costs nothing here because `W` is one anyway, and it is exact --
  // every valuation in it is a genuine successor of a stored zone, so no widening garbage enters
  // the fixpoint and the drain keeps the exact seed's shape. Nodes that never received an inexact
  // cover get `C_j = Z_j` and keep the cheap unmaterialised `seeded` representation.
  //
  // The successors are regenerated here rather than recorded during construction because
  // ::LivenessEngine::online_finalize_graph renumbers records, so a table built at cover time
  // would be stale by the time the gfp reads it. Regenerating costs one forward sweep.
  //
  // A successor is attributed to every outgoing edge whose target shares its discrete key. That
  // can add a piece to a node the cover did not actually land in, which is harmless: the piece is
  // still a genuine successor of a stored zone at that key, so it holds only real states, and
  // Lemma A only ever asks for MORE to be present.
  // HORA_MEASURE_CLOSURE: compute the EXACT carrier -- the least fixpoint of
  //     C_v superseteq {Z_v}   and   post_e(C_v) subseteq C_j  for every edge,
  // with federations throughout and no hull anywhere. This is the object the design note calls
  // the exact downward closure. It is pure measurement: it prints its size and then leaves the
  // seed alone, so the verdict below is unaffected. The question it answers is whether the exact
  // object is affordable, or whether a convex approximation is forced.
  if ((fc_mode_ || carrier_mode_) && std::getenv("HORA_MEASURE_CLOSURE")) {
    std::vector<Federation> C(N);
    std::vector<std::uint64_t> kh(N, 0);
    SymState msc, mtc;
    std::vector<SymState> msucc;
    std::vector<GfpTag> mtags;
    Dbm mz;
    for (int v : S) { nstate(v, mtc); kh[v] = disc_hash(mtc.locs, mtc.ints); C[v].push_back(zread(v, mz)); }
    const std::size_t cap = 40 * S.size() + 1000;
    std::size_t total = S.size(), rounds = 0;
    bool changed = true, exploded = false;
    const auto t_start = std::chrono::steady_clock::now();
    while (changed && !exploded && rounds < 200) {
      changed = false; ++rounds;
      for (int v : S) {
        if (adj[v].empty()) continue;
        nstate(v, msc);
        for (std::size_t pi = 0; pi < C[v].size(); ++pi) {
          msc.zone = C[v][pi];
          if (msc.zone.empty()) continue;
          msucc.clear(); mtags.clear();
          tagged_successors(net, msc, L, U, qmask, emode, msucc, mtags,
                            avoid_forward_ ? avoid_ : nullptr);
          for (const SymState& x : msucc) {
            if (x.zone.empty()) continue;
            const std::uint64_t xh = disc_hash(x.locs, x.ints);
            for (const auto& e : adj[v]) {
              const int j = e.tgt;
              if (!inS[j] || kh[j] != xh) continue;
              nstate(j, mtc);
              if (mtc.locs != x.locs || mtc.ints != x.ints) continue;
              if (fed_covers(C[j], x.zone)) continue;      // already inside the closure
              C[j].push_back(x.zone); ++total; changed = true;
              if (total > cap) { exploded = true; break; }
            }
            if (exploded) break;
          }
          if (exploded) break;
        }
        if (exploded) break;
      }
      if (!exploded) { total = 0; for (int v : S) { bool cp = false; fed_compact(C[v], fedcap, &cp); total += C[v].size(); } }
    }
    std::size_t wmax = 0, nonsingleton = 0;
    for (int v : S) { wmax = std::max(wmax, C[v].size()); if (C[v].size() > 1) ++nonsingleton; }
    const double ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t_start).count();
    std::fprintf(stderr, "EXACT-CLOSURE %s rounds=%zu pieces=%zu (nodes=%zu, %.2f/node) "
                 "maxwidth=%zu multi_piece_nodes=%zu ms=%.0f\n",
                 exploded ? "EXPLODED" : "converged", rounds, total, S.size(),
                 (double)total / (double)S.size(), wmax, nonsingleton, ms);
  }
  // OPTIMISTIC OPERATOR (HORA_OPT). The pessimistic operator targets `W[j]` and so loses a
  // witness whose next state left `Z_j`; repairing that is what costs 20x. This targets
  // `Extra_LU+(Z_j)` instead -- the very set tight covering certifies `post_e(Z_v)` lies inside --
  // and gates the edge on `W[j]` merely being NON-EMPTY:
  //
  //     W[v] := W[v] /\ ( NoSucc(v) \/ \/_{e : W[tgt(e)] != {}} Pre_e(Extra_LU+(Z_tgt(e))) ).
  //
  // `V_v = A_v /\ Z_v` is a POST-fixpoint of it: for z in V_v with a run z ->_e z', tight covering
  // puts z' in Extra_LU+(Z_j), and z' has a run so its ⊑_LU-dominator in Z_j has one too (⊑ is a
  // simulation), which keeps W[j] non-empty. So the gfp CONTAINS V, and an empty result is a
  // CONCLUSIVE `NOT` -- the half we currently pay the most for. A non-empty result is only a
  // candidate and decides nothing.
  //
  // It is cheap because the predecessor target does not depend on W: one zone per edge rather
  // than a federation, so the sweep degenerates to propagating emptiness.
  //
  // THE TARGET. It must contain `a_D(W[j])` -- that is what makes V a post-fixpoint -- and it
  // must SHRINK with W[j], or the operator carries no information and stabilises on its seed.
  // A convex superset of the LU-closure can keep neither upper bounds nor diagonals: `v ≼ u`
  // allows `v_x > u_x` whenever `v_x > L_x`, so the closure is unbounded above in every clock
  // with a finite L, and x_i can be raised while x_j is held. What survives is the per-clock
  // LOWER bound, and only for clocks the zone keeps under U_x -- above U_x the closure admits
  // points below u as well, so the bound must go.
  //
  // That is enough to be useful only because ::exact_predecessor conjoins the TARGET INVARIANT
  // before stepping back, which reinstates the upper bounds psi lives in. A real successor
  // satisfies that invariant, so intersecting with it keeps the superset property intact.
  auto lu_super = [&](int j, Dbm& out) {
    const int dim = net.nclocks + 1;
    out = Dbm::universe(net.nclocks);
    Dbm one;
    const Dbm* pieces_one = nullptr;
    if (seeded[j]) pieces_one = &zread(j, one);   // opt mode leaves every carrier kCarSame
    const std::size_t np = pieces_one ? 1u : W[j].size();
    for (int x = 1; x <= net.nclocks; ++x) {
      raw_t lo = -kInf, hi = -kInf;                 // weakest bound over the members
      for (std::size_t k = 0; k < np; ++k) {
        const Dbm& z = pieces_one ? *pieces_one : W[j][k];
        if (z.empty()) continue;
        lo = std::max(lo, z.get(0, x));
        hi = std::max(hi, z.get(x, 0));
      }
      if (lo == -kInf) continue;                    // no live member
      const bool bounded_under_U = hi < kInf && x < (int)U.size() && raw_bound(hi) <= U[x];
      if (bounded_under_U) out.set(0, x, lo);       // keep the lower bound; drop the upper
    }
    (void)dim;
    out.close();
  };
  std::vector<Dbm> optc;
  if (opt_mode) { carrier_state_.assign(N, kCarSame); carrier_.assign(N, Dbm()); }
  if (!opt_mode && (fc_mode_ || carrier_mode_)) {
    carrier_state_.assign(N, kCarSame);        // no Extra_LU+ carrier on this route at all
    carrier_.assign(N, Dbm());
    extra.assign(N, Federation{});
    std::vector<SymState> succ;
    std::vector<GfpTag> stags;
    SymState ssc, tsc;
    Dbm zsc;
    // One hash per node, computed once. Matching a successor to the edge it was covered by is a
    // discrete-key comparison, and doing it by decoding the target costs a ::nstate -- two vector
    // copies -- per (successor, edge) PAIR. That decode was the whole cost of this prologue when
    // it sat in the inner loop. The hash reduces the pair test to an integer compare, and the
    // decode happens only on a hit, which is at most once per successor.
    std::size_t dbg_succ = 0, dbg_keymatch = 0;
    std::vector<std::uint64_t> khash(N, 0);
    for (int v : S) { nstate(v, tsc); khash[v] = disc_hash(tsc.locs, tsc.ints); }
    for (int v : S) {
      if (adj[v].empty()) continue;
      succ.clear();                       // ::tagged_successors APPENDS; a reused buffer must be
      stags.clear();                      // emptied or every node inherits its predecessors' set
      tagged_successors(net, sread(v, ssc), L, U, qmask, emode, succ, stags,
                        avoid_forward_ ? avoid_ : nullptr);
      for (const SymState& x : succ) {
        if (x.zone.empty()) continue;
        ++dbg_succ;
        const std::uint64_t xh = disc_hash(x.locs, x.ints);
        for (const auto& e : adj[v]) {
          const int j = e.tgt;
          if (!inS[j] || khash[j] != xh) continue;
          if (cycle_only && sccid[v] != sccid[j]) continue;   // leaves the component: no cycle
          nstate(j, tsc);                                           // confirm: hashes can collide
          if (tsc.locs != x.locs || tsc.ints != x.ints) continue;   // different discrete key
          ++dbg_keymatch;
          if (stored_includes(j, x.zone)) continue;                 // exact cover: nothing owed
          // Census for the overlap question: does the discarded successor even MEET the zone that
          // covers it? A disjoint cover owes the whole of X; an overlapping one owes only the
          // part sticking out. If the disjoint ones are few but carry most of the debt, refusing
          // exactly those trades few extra nodes for most of the backward cost.
          if (g_ovl) {
            Dbm jz2;
            const Dbm& zj = zread(j, jz2);
            const bool meets = x.zone.intersects(zj);
            std::size_t owed = 0;
            for (const Dbm& d : zone_subtract(x.zone, zj)) if (!d.empty()) ++owed;
            if (meets) { ++g_ovl_part; g_ovl_part_pieces += owed; }
            else       { ++g_ovl_disj; g_ovl_disj_pieces += owed; }
          }
          // Only `X \ Z_j` is owed -- the rest of X is already in the seed. Storing X whole
          // makes the seed a heavily OVERLAPPING cover of the same set, which is the worst
          // input for both halves of the sweep: every overlapping pair survives the
          // disjointness pre-test and produces a product, and ::fed_compact then has to
          // consider gluing each pair back. The difference is disjoint from Z_j by
          // construction, so those pairs disappear.
          Dbm jz;
          for (Dbm& d : zone_subtract(x.zone, zread(j, jz)))
            if (!d.empty()) extra[j].push_back(std::move(d));
        }
      }
    }
    // The obligation is `X subseteq C_v`, so C_v may be ANY superset of the pieces -- and a
    // superset costs nothing in soundness. The witness chain starts at `W[0] /\ Z_0`, which is
    // real, and every step is a genuine transition with the target invariant applied by
    // ::exact_predecessor, so the chain stays real whatever unreachable points C_v also holds;
    // over-approximating C_v can only add candidates the operator then has to justify.
    //
    // So keep C_v CONVEX: one DBM, not a federation. A federation-valued seed propagates -- a
    // multi-piece W[j] makes `cand` multi-piece at every predecessor, `W[v] /\ cand` multiplies
    // piece counts, and ::fed_compact then spends a `dbm_try_union` per pair gluing the shards
    // back together, which measured as the largest single cost in the sweep. The convex hull of
    // the same pieces satisfies Lemma A just as exactly and keeps every node a single zone, so
    // the drain keeps the shape it has with no carrier at all.
    //
    // It is also far tighter than the Extra_LU+ carrier this replaced: a hull of finitely many
    // reachable successors stays bounded, where Extra_LU+ deletes upper bounds outright.
    std::size_t dbg_nodes = 0, dbg_raw = 0;
    const int hdim = net.nclocks + 1;
    static const bool seed_fed = [] {
      const char* e = std::getenv("HORA_SEED"); return !e || *e != 'h';   // federation by default
    }();
    for (int v : S) {
      if (extra[v].empty()) continue;                               // C_v = Z_v; stays seeded
      dbg_hadextra[v] = 1;
      ++dbg_nodes; dbg_raw += extra[v].size();
      if (seed_fed) {                                               // exact, federation-valued
        // The zone itself is NOT pruned by the escape filter below -- NoSucc(v) is judged on it
        // and enters the operator as a union term, so dropping its stuck part would lose stop
        // witnesses the operator cannot put back. Only the extras are prunable, so they stay in
        // `extra[v]` until the filter runs and are adjoined there.
        W[v].clear();
        W[v].push_back(zread(v, zsc));
        seeded[v] = 0;
        continue;
      }
      Dbm h = zread(v, zsc);
      for (const Dbm& x : extra[v]) dbm_hull(h.data().data(), h.data().data(), x.data().data(), hdim);
      carrier_[v] = std::move(h);
      carrier_state_[v] = kCarWide;                                 // seeded[v] stays 1
    }
    if (std::getenv("HORA_GFP_PROFILE"))
      std::fprintf(stderr, "ATTRIB succ=%zu keymatches=%zu  targets_per_succ=%.2f\n",
                   dbg_succ, dbg_keymatch, dbg_succ ? (double)dbg_keymatch / dbg_succ : 0.0),
      std::fprintf(stderr, "HULL-SEED nodes=%zu of %zu  pieces_hulled=%zu avg=%.1f\n",
                   dbg_nodes, S.size(), dbg_raw, dbg_nodes ? (double)dbg_raw / dbg_nodes : 0.0);
  }
  // When the forward graph was NOT restricted, the stored zones still straddle `psi` and the
  // fixpoint has to do the confining itself: `W[v]` starts at `Z_v /\ psi(v)`, not at `Z_v`. The
  // seed is otherwise left unmaterialised (see `seeded`), so only the nodes that actually need
  // cutting are written down here.
  const bool confine = avoid_ && !avoid_forward_;
  std::vector<Federation> psi_seed;
  std::vector<char> psi_cut;                // v's seed was restricted, so its stop set owes the cut
  if (confine) { psi_seed.resize(N); psi_cut.assign(N, 0); }
  if (confine)
    for (int v : S) {
      SymState cst;
      const SymState& sv = sread(v, cst);
      if (!avoid_->applies(sv.locs, sv.ints)) continue;
      Federation cut;
      avoid_->restrict(sv.zone, sv.locs, sv.ints, cut);
      psi_seed[v] = std::move(cut);
      psi_cut[v] = 1;
      seeded[v] = 0;                       // no longer "logically its own zone"
      W[v] = psi_seed[v];
    }
  // The stopping term is computed ON DEMAND, and every caller below is written to ask for it
  // only when its answer can still matter.
  //
  // It occurs POSITIVELY in `W /\ (NoSucc \/ Pre(W))`, so wherever `Pre(W)` already covers
  // `W[v]` the refinement is `W[v]` whatever NoSucc holds there --- the node's stop set is never
  // read. ::LivenessEngine::stop_set is the expensive term of the whole operator (a delay, the
  // invariants, a divergence test and a per-valuation deadlock test per node), and on a graph
  // whose fixpoint drains through its cycles the nodes that ever need it are a small minority.
  // Computing it for every node up front was pricing the operator's rarest case into every one.
  std::vector<char> nosucc_done(N, 0);
  std::size_t nosucc_calls = 0;
  auto nosucc_of = [&](int v) -> const Federation& {
    if (fc_mode_) { nosucc_done[v] = 1; nosucc[v].clear(); return nosucc[v]; }
    if (!nosucc_done[v]) {
      nosucc_done[v] = 1;
      ++nosucc_calls;
      nosucc[v] = stop_set(v);
      // A confined seed cuts the value down to psi, and the stop set is judged against the whole
      // stored zone, so it owes the same cut: a valuation outside psi must not end the run. An
      // empty restriction leaves nothing that could.
      if (confine && psi_cut[v]) {
        if (psi_seed[v].empty()) nosucc[v].clear();
        else if (!nosucc[v].empty()) nosucc[v] = fed_intersect(nosucc[v], psi_seed[v]);
      }
    }
    return nosucc[v];
  };

  // Widening scratch, reused across the whole sweep so a narrow-stored zone costs a copy of
  // its entries and never an allocation.
  Dbm vsc, jsc, fcsc;
  // Z_v, or its Extra_LU+ carrier in fc mode. Both land in the caller's scratch.
  auto zcar = [&](int v, Dbm& sc) -> const Dbm& {
    if (fc_mode_ || carrier_mode_) {
      if (const Dbm* car = carrier_of(v)) return *car;   // cached; no load, no extrapolation
    }
    zread(v, sc);
    return sc;
  };
  SymState vst, jst;
  // round-robin lockstep sweep, deliberately, not for want of a worklist. The lockstep sweep acts
  // as a de-facto normaliser. SCC-decomposed lockstep. The sweep above is kept verbatim; what
  // changes is the set it is swept over.
  //
  // A node's `cand` reads only its successors' W, so the descent propagates emptiness one edge per
  // global round and the round count tracks the graph's diameter, not anything about the property.
  //
  // Iterating strongly connected components in reverse topological order removes exactly that. When
  // a component is reached its successors outside it are already final, so it converges against
  // fixed outside values and is never revisited; the result is the same greatest fixpoint because
  // the operator is local and Tarjan emits components successors-first.
  //
  // Here the lockstep round-robin survives inside each component, so the de-facto normalisation it
  // provides is kept where the iteration actually happens; only the outer partition into components
  // is new. Enabled for `E[] psi` and not for `p --> q`. `E[] psi` reads ONE node, so emptiness has
  // to propagate the full diameter back to it and the round count is dominated by that; `p --> q`
  // reads every p-node and settles long before diameter matters. HORA_GFP_SCC=0 or =1 overrides
  // either way. See docs/design.md section 8 for the measurements behind the split. decomposition
  // does not go there. HORA_GFP_SCC=0 or =1 overrides either way, for measuring.
  const bool scc_on = [&] {
    const char* e = std::getenv("HORA_GFP_SCC");
    if (e) return *e == '1';
    return scc_iter_;
  }();
  // Tarjan runs unconditionally: the emptiness prune below reads its components, and the sweep
  // reads them as an order only when `scc_on`.
  std::vector<std::vector<int>> comps;
  {
    std::vector<int> idx(N, -1), low(N, 0), stk;
    std::vector<char> on(N, 0);
    int next = 0;
    // Iterative Tarjan: the graphs here reach 10^5 nodes and the recursive form would not survive
    // the stack.
    std::vector<std::pair<int, std::size_t>> call;
    for (int root : S) {
      if (idx[root] != -1) continue;
      call.push_back({root, 0});
      idx[root] = low[root] = next++;
      stk.push_back(root);
      on[root] = 1;
      while (!call.empty()) {
        auto& [v, ei] = call.back();
        if (ei < adj[v].size()) {
          const int w = adj[v][ei++].tgt;
          if (!inS[w]) continue;
          if (idx[w] == -1) {
            idx[w] = low[w] = next++;
            stk.push_back(w);
            on[w] = 1;
            call.push_back({w, 0});
          } else if (on[w]) {
            low[v] = std::min(low[v], idx[w]);
          }
        } else {
          if (low[v] == idx[v]) {
            std::vector<int> comp;
            for (;;) {
              const int w = stk.back();
              stk.pop_back();
              on[w] = 0;
              comp.push_back(w);
              if (w == v) break;
            }
            // Restore S's own order inside the component, so the lockstep sweep walks it exactly
            // as the undecomposed one did.
            std::sort(comp.begin(), comp.end());
            comps.push_back(std::move(comp));
          }
          const int child = v;
          call.pop_back();
          if (!call.empty()) low[call.back().first] = std::min(low[call.back().first], low[child]);
        }
      }
    }
  }

  // structural emptiness, decided on the graph alone.
  //
  // At the fixpoint, `W[v] = W[v] /\ (NoSucc(v) \/ \/_e Pre_e(W[j_e]))`. So a non-empty W[v]
  // needs either a non-empty stop set at v, or a successor inside S that is itself non-empty.
  // Chase the second case: it yields an unbounded walk of non-empty nodes inside S, which in a
  // finite graph must revisit a node and therefore meets a cycle. Every node with a non-empty
  // value can thus reach, inside S, a node that is on a cycle or can stop --- and every node
  // that cannot is exactly empty, with no zone computation at all.
  //
  // Nothing here reads a zone. `nosucc` is already built for all of S, and Tarjan emits
  // components successors-first, so one pass in that order settles it: a singleton's successors
  // outside itself live in components already decided.
  std::vector<char> live_node(N, 0);
  std::size_t live_count = 0;
  for (const auto& comp : comps) {
    bool live = comp.size() > 1;                       // a nontrivial component IS a cycle
    for (int v : comp) {
      if (live) break;
      // Successors first: this is a disjunction, and reaching a live node is decided by a
      // pointer chase where the stop set costs a zone computation. Only a node with nothing
      // live to reach has to ask whether the run may legitimately stop in it.
      for (const auto& e : adj[v])
        if (inS[e.tgt] && (e.tgt == v || live_node[e.tgt])) { live = true; break; }
      if (!live && !nosucc_of(v).empty()) live = true;
    }
    if (!live) continue;
    for (int v : comp) { live_node[v] = 1; ++live_count; }
  }
  // Pin the rest to the empty federation, which is what `inS[v] = 0` means to ::relax.
  if (live_count < S.size())
    for (int v : S)
      if (!live_node[v]) {
        inS[v] = 0;
        seeded[v] = 0;
        W[v].clear();
        nosucc[v].clear();
        nosucc_done[v] = 1;                // pinned empty: never worth computing
      }

  // Filter `comps` in place rather than moving the survivors into a second vector: on a graph
  // whose components are mostly singletons the outer vector alone is 24 bytes per component,
  // and holding two of them was a measurable share of the peak.
  if (scc_on) {
    comps.erase(std::remove_if(comps.begin(), comps.end(),
                               [&](const std::vector<int>& c) {
                                 return c.empty() || !live_node[c.front()];
                               }),
                comps.end());
  } else {
    std::vector<int> flat;                 // one component: the undecomposed sweep, minus the
    flat.reserve(live_count);              // nodes just proved empty
    for (int v : S)
      if (live_node[v]) flat.push_back(v);
    comps.clear();
    if (!flat.empty()) comps.push_back(std::move(flat));
  }
  const std::vector<std::vector<int>>& sccs = comps;

  // ::PreNode explains why these are cached rather than recomputed inside `Pre`. Only nodes that
  // are actually the source or target of a live edge get one, so the table is an index plus a
  // deque grown on demand rather than N records: where the prune above left nothing to sweep it
  // costs one `int` per node instead of a whole PreNode, and the deque keeps references stable
  // across growth, which ::relax relies on when it holds two of them at once.
  const int dim = net.nclocks + 1;
  std::vector<std::int32_t> pindex;
  std::deque<PreNode> pnodes;
  // One arena for every node's invariants, with the spans interned. Interning by content collapses
  // that to one copy per distinct list.
  std::vector<Constraint> pinvs;
  std::vector<Constraint> iscratch;
  std::unordered_map<std::uint64_t, std::vector<std::pair<std::uint32_t, std::uint32_t>>> pinv_index;
  std::vector<unsigned char> dead_scratch(dim, 0);
  auto intern_inv = [&](const std::vector<Constraint>& cs) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const Constraint& c : cs) {
      h ^= std::uint64_t(std::uint32_t(c.i)) * 0x9e3779b97f4a7c15ULL; h = (h << 7) | (h >> 57);
      h ^= std::uint64_t(std::uint32_t(c.j)) * 0xc2b2ae3d27d4eb4fULL; h = (h << 11) | (h >> 53);
      h ^= std::uint64_t(std::uint32_t(c.bound)) * 0x165667b19e3779f9ULL;
    }
    auto& slots = pinv_index[h];
    for (const auto& sl : slots) {
      if (sl.second != cs.size()) continue;
      bool same = true;
      for (std::size_t k = 0; k < cs.size() && same; ++k) {
        const Constraint& a = pinvs[sl.first + k];
        same = a.i == cs[k].i && a.j == cs[k].j && a.bound == cs[k].bound;
      }
      if (same) return sl;
    }
    const std::pair<std::uint32_t, std::uint32_t> span{
        static_cast<std::uint32_t>(pinvs.size()), static_cast<std::uint32_t>(cs.size())};
    pinvs.insert(pinvs.end(), cs.begin(), cs.end());
    slots.push_back(span);
    return span;
  };
  auto prep = [&](int v) -> const PreNode& {
    if (pindex.empty()) pindex.assign(N, -1);
    if (pindex[v] >= 0) return pnodes[static_cast<std::size_t>(pindex[v])];
    pindex[v] = static_cast<std::int32_t>(pnodes.size());
    pnodes.emplace_back();
    PreNode& pn = pnodes.back();
    pn.ready = true;
    SymState dsc;
    nstate(v, dsc);
    const LocVec& locs = dsc.locs;
    all_invariants(net, locs, iscratch);
    const auto span = intern_inv(iscratch);
    pn.inv_off = span.first;
    pn.inv_len = span.second;
    bool c2, u2;
    frozen_flags(net, locs, dsc.ints, &c2, &u2);
    pn.frozen = c2 || u2;
    // The mask needs one bit per clock. Wider networks keep no active_mask either (see
    // Network::compute_active), and the reduction is then off, so there is nothing to record.
    if (net.nclocks > 63) return pn;
    if (!net.active_mask.empty()) {                  // one OR per process, see reset_inactive
      std::uint64_t a = 0;
      for (int pr = 0; pr < net.nprocs(); ++pr) a |= net.active_mask[pr][locs[pr]];
      pn.dead = ~a & (((std::uint64_t(1) << net.nclocks) - 1) << 1);
    } else if (!net.active.empty()) {
      for (int c = 1; c <= net.nclocks; ++c) {
        bool live = false;
        for (int pr = 0; pr < net.nprocs() && !live; ++pr)
          if (net.active[pr][locs[pr]][c]) live = true;
        if (!live) pn.dead |= std::uint64_t(1) << c;
      }
    }
    return pn;
  };

  // One reset mask per tag, not per edge: tags are pooled by value, so two edges with the same
  // transition recipe share it. Built lazily for the same reason as ::PreNode.
  std::vector<std::vector<unsigned char>> rmask(tags_.size());
  std::vector<char> rmask_ready(tags_.size(), 0);
  auto tag_resets = [&](int tag) -> const std::vector<unsigned char>& {
    if (!rmask_ready[tag]) {
      rmask_ready[tag] = 1;
      rmask[tag].assign(dim, 0);
      for (const auto& pe : tags_[tag].parts)
        for (int r : net.procs[pe.first].edges[pe.second].clock_resets)
          if (r >= 1 && r <= net.nclocks) rmask[tag][r] = 1;
    }
    return rmask[tag];
  };

  // ESCAPE PRUNE. An extra piece is worth keeping only if it can be a witness, and a valuation
  // has exactly three ways to be one:
  //
  //   * it lies on a CYCLE -- then it takes transitions forever, so it must be able to take one
  //     at all, i.e. it lies in Esc(v), the union of the outgoing transitions' enabling regions;
  //   * it is a STOP witness -- Dead(v), unable to move even in the unrestricted network. The D
  //     refinement exists to preserve deadlock under covering, so its dominator in Z_v is Dead
  //     too and `NoSucc(v)` already holds a representative; the extra adds nothing;
  //   * neither -- it cannot move in the psi-RESTRICTED graph yet is not Dead in the unrestricted
  //     one, so its only continuation leaves psi. That is not a psi-witness at all.
  //
  // Only the first survives, so `extra /\ Esc(v)` loses nothing. And the extras are precisely the
  // a_LU-relaxed valuations -- clocks pushed above L, hence past guards still open in Z_v -- which
  // is exactly the population expected to have stopped escaping.
  //
  // ::exact_predecessor against the UNIVERSE is the enabling region: it applies the target
  // invariant, the delay, the resets and the guards, and yields the source valuations from which
  // the step is possible at all.
  if ((fc_mode_ || carrier_mode_) && !avoid_ && !extra.empty()) {
    std::size_t before = 0, after = 0, nodes = 0;
    const Dbm univ = Dbm::universe(net.nclocks);
    for (int v : S) {
      if (extra[v].empty()) continue;
      ++nodes;
      Federation esc;
      for (const auto& e : adj[v]) {
        if (!inS[e.tgt]) continue;
        Dbm pz = exact_predecessor(net, tags_[e.tag], prep(v), prep(e.tgt), pinvs,
                                   tag_resets(e.tag), dead_scratch.data(), univ);
        if (!pz.empty()) esc.push_back(std::move(pz));
      }
      fed_compact(esc, fedcap, &capped);
      Federation kept;
      for (const Dbm& x : extra[v]) {
        ++before;
        for (const Dbm& c : esc) {
          if (!x.intersects(c)) continue;
          Dbm t = x;
          t.intersect(c);
          if (!t.empty()) kept.push_back(std::move(t));
        }
      }
      fed_compact(kept, fedcap, &capped);
      for (Dbm& k : kept) W[v].push_back(std::move(k));
      fed_compact(W[v], fedcap, &capped);
      // UNION SUBSUMPTION. ::fed_compact drops a member only when a SINGLE other member contains
      // it, or when two are exactly convex together. It never asks whether a member is covered by
      // the UNION of the rest -- and the extras all cluster around Z_v, which is exactly the shape
      // where that holds. Removing such a member leaves the denoted set identical, so this is
      // exact: it costs no completeness, it only shrinks the representation. Piece count is what
      // drives |cand| and the products in the refinement, so it is the representation that costs.
      for (std::size_t k = 0; k < W[v].size(); ) {
        if (W[v].size() <= 1) break;
        Federation rest;
        rest.reserve(W[v].size() - 1);
        for (std::size_t t = 0; t < W[v].size(); ++t)
          if (t != k && !W[v][t].empty()) rest.push_back(W[v][t]);
        if (!rest.empty() && fed_covers(rest, W[v][k])) W[v].erase(W[v].begin() + k);
        else ++k;
      }
      after += W[v].size();
    }
    if (std::getenv("HORA_GFP_PROFILE"))
      std::fprintf(stderr, "ESC+UNION-PRUNE nodes=%zu extras_in=%zu members_kept=%zu (%.1f%% removed)\n",
                   nodes, before, after,
                   before ? 100.0 * (double)(before - after) / (double)before : 0.0);
  }
  // one node'S refinement, `W[v] /\ (NoSucc(v) \/ Pre(W))`. Factored out so the lockstep sweep
  // and the worklist below drive exactly the same operator; only the order differs.
  // `W[v] /\ (NoSucc(v) \/ \/_e Pre_e(W[j]))`, with the union accumulated in `cand`.
  //
  // That union is MONOTONE in the edges already processed, so the moment `cand` covers all of
  // W[v] the result is W[v] whatever the remaining edges contribute -- the intersection cannot
  // shrink below what a subset of the union already contains. The sweep can stop there, skipping
  // both the remaining predecessors and the compaction of `cand` altogether. Near the fixpoint
  // most visits are of exactly that shape, which is where the work was going.
  auto covers_value = [&](const Dbm& z, int v) {
    // A seeded node's value is its own stored zone, which ::includes_stored reads at its own
    // entry type -- widening it into a scratch on every probe is the cost this test exists to
    // avoid paying elsewhere.
    //
    // The question is always `z ⊇ W[v]`, and W[v] of a seeded node is its CARRIER, not its
    // stored zone. Asking it the other way round -- carrier ⊇ z -- is the early exit's
    // negation, and it holds almost always, because `z` is one predecessor piece and the
    // carrier is the widest set the node can hold: `relax` then reports "no change" on its
    // first edge and the seed never drains, so a drained-to-empty gfp reads as a witness.
    if (seeded[v]) {
      if (fc_mode_ || carrier_mode_) {
        // A node whose zone already equals its carrier keeps the baseline's no-copy test.
        if (const Dbm* car = carrier_of(v)) return z.includes(*car);
      }
      return includes_stored(z, v);
    }
    for (const auto& w : W[v])
      if (!w.empty() && !z.includes(w)) return false;
    return true;
  };
  // Scratch for the early exit's per-member cover flags; reused so a visit costs no allocation.
  std::vector<char> covflag;
  std::size_t dbg_nwraw = 0, dbg_nwkept = 0, dbg_refine = 0, dbg_wmax = 0, dbg_candmax = 0, dbg_candsum = 0, dbg_candn = 0;
  std::vector<int> dbg_rounds(N, 0), dbg_deathround(N, 0);
  auto relax = [&](int v) -> bool {
    if (!seeded[v] && fed_is_empty(W[v])) return false;
    Federation cand;                                 // the predecessor term alone, for now
    // `W[v] /\ (NoSucc \/ cand)` is `W[v]` as soon as cand covers every MEMBER of W[v], and a
    // member may be covered by a different member of cand than its neighbour is. Asking instead
    // for one `pz` to cover the whole value -- which is what ::covers_value does -- is the same
    // question only while values are single zones. Once a seed carries the discarded successors
    // the values are federations, that test essentially never fires, and the sweep pays every
    // edge on every visit: measured at 10.4 predecessor calls per visit against the exact
    // operator's 0.76, with each resulting change re-queueing the node's predecessors.
    std::size_t ncov = 0;
    const std::size_t nmem = seeded[v] ? 0 : W[v].size();
    if (nmem) {
      covflag.assign(nmem, 0);
      for (std::size_t k = 0; k < nmem; ++k)
        if (W[v][k].empty()) { covflag[k] = 1; ++ncov; }   // denotes nothing: trivially covered
    }
    for (const auto& e : adj[v]) {
      const int j = e.tgt;
      if (!inS[j]) continue;
      if (!seeded[j] && fed_is_empty(W[j])) continue;
      // A seeded successor contributes exactly one zone, its own, without being written down.
      // Under the optimistic operator the target is the node's Extra_LU+ carrier and does not
      // depend on W at all -- W[j] being non-empty (tested above) is the whole dependence.
      Dbm optsc;
      if (opt_mode) lu_super(j, optsc);
      const Dbm* one = opt_mode ? &optsc : (seeded[j] ? &zcar(j, jsc) : nullptr);
      const std::size_t ny = one ? 1u : W[j].size();
      const PreNode* pv = avoid_ ? nullptr : &prep(v);
      const PreNode* pj = avoid_ ? nullptr : &prep(j);
      for (std::size_t yi = 0; yi < ny; ++yi) {
        const Dbm& Y = one ? *one : W[j][yi];
        ++pre_calls;
        if (avoid_) {
          exact_predecessor_avoiding(net, tags_[e.tag], sread(v, vst), sread(j, jst), Y, *avoid_, cand);
        } else {
          Dbm pz = exact_predecessor(net, tags_[e.tag], *pv, *pj, pinvs, tag_resets(e.tag),
                           dead_scratch.data(), Y);
          if (pz.empty()) continue;
          if (nmem) {
            for (std::size_t k = 0; k < nmem; ++k)
              if (!covflag[k] && pz.includes(W[v][k])) { covflag[k] = 1; ++ncov; }
            if (ncov == nmem) return false;        // every member covered: the value cannot move
          } else if (covers_value(pz, v)) {
            return false;                          // the union already covers W[v]; see above
          }
          cand.push_back(std::move(pz));
        }
      }
    }
    fed_compact(cand, fedcap, &capped);
    dbg_candmax = std::max(dbg_candmax, cand.size()); dbg_candsum += cand.size(); ++dbg_candn;
    // Materialise v's own seed now, for this one node only: the swap below overwrites it with
    // the refinement, so the copy does not outlive the iteration.
    if (seeded[v]) { W[v].clear(); W[v].push_back(zcar(v, vsc)); seeded[v] = 0; }
    // `W[v] /\ (NoSucc \/ Pre)` is `W[v]` as soon as `Pre` alone covers it, so the stopping term
    // is asked for only when the predecessor term has already failed to reproduce the value.
    auto escapes = [&] {
      for (const auto& z : W[v]) {
        if (z.empty()) continue;                     // denotes nothing: cannot witness a shrink
        bool whole = false;
        for (const auto& c : cand)
          if (c.includes(z)) { whole = true; break; }
        if (!whole && !fed_covers(cand, z)) return true;
      }
      return false;
    };
    if (!escapes()) return false;
    if (const Federation& ns = nosucc_of(v); !ns.empty()) {
      for (const auto& c : ns) cand.push_back(c);
      fed_compact(cand, fedcap, &capped);
      if (!escapes()) return false;
    }
    // `W[v] /\ cand`, computed directly rather than as the double complement
    // `subtract(z, subtract(z, cand))` the textbook formulation uses. Both denote the same
    // set, but the inner subtraction peels z by every constraint of every member of cand, so
    // its output is an arrangement whose size is not bounded by |cand| at all, and the outer
    // subtraction then peels z by all of those. The intersection of two zones is a single
    // zone, so the direct form needs exactly |cand| pieces.
    //
    // Reached only when `escapes()` said the value really moves, so none of this is spent on a
    // visit that changes nothing: such a visit leaves the stored representation alone, which
    // denotes the same set -- every test the fixpoint makes of a value (::covers_value,
    // ::fed_is_empty, `escapes` itself) reads the union and not the member list.
    Federation nw;
    for (const auto& z : W[v]) {
      bool whole = false;
      for (const auto& c : cand)
        if (c.includes(z)) { whole = true; break; }
      // `z /\ cand == z`, and every product `z /\ c'` is a subset of it, so the inner loop
      // below could only produce pieces the compaction would absorb again. Emit z and move on.
      if (whole) { nw.push_back(z); continue; }
      for (const auto& c : cand) {
        // Most (z, c) pairs of a federation do not meet at all, and forming the product to find
        // that out costs a full DBM copy plus a close. ::Dbm::intersects decides it in O(dim^2)
        // reads with neither -- and the copy was the single hottest thing in this sweep.
        if (!z.intersects(c)) continue;
        Dbm t = z;
        t.intersect(c);
        if (!t.empty()) nw.push_back(std::move(t));
      }
    }
    dbg_nwraw += nw.size();
    fed_compact(nw, fedcap, &capped);
    dbg_nwkept += nw.size(); ++dbg_refine;
    dbg_wmax = std::max(dbg_wmax, nw.size());
    if (v < (int)dbg_rounds.size()) {
      ++dbg_rounds[v];
      bool allempty = true;
      for (const auto& z : nw) if (!z.empty()) { allempty = false; break; }
      if (allempty && dbg_deathround[v] == 0) dbg_deathround[v] = dbg_rounds[v];
    }
    W[v].swap(nw);
    return true;
  };

  // worklist inside A component, for `E[] psi` only. `relax` reads only v's successors, so v can
  // change only after one of them has -- and the lockstep sweep re-reads every node of the
  // component every round regardless. That is free where the decomposition is fine-grained and
  // ruinous where the closure is a single component, which is the common shape here.
  //
  // That measurement was on leads-to, which is why this is confined to @ref scc_iter_, which is now
  // on for `p --> q` as well; see its declaration. The E[] suite runs at a maximum federation width
  // of 186 against a cap of 4096, with zero cap hits across 108 rows, so the bound that killed it
  // there is not close to binding here.
  const bool worklist = scc_on && [] {
    const char* e = std::getenv("HORA_GFP_WORKLIST");
    return !e || *e != '0';
  }();

  std::vector<std::vector<int>> rev;
  std::vector<char> in_comp, in_q;
  if (worklist) {
    rev.resize(N);
    for (int v : S)
      for (const auto& e : adj[v])
        if (inS[e.tgt]) rev[e.tgt].push_back(v);
    in_comp.assign(N, 0);
    in_q.assign(N, 0);
  }

  std::size_t it = 0;
  for (const auto& comp : sccs) {
   if (capped || cancelled()) break;
   if (!worklist) {
    bool changed = true;
    while (changed && !capped) {
     if (cancelled()) break;
     changed = false;
     ++it;
     for (int v : comp) {
       if (relax(v)) changed = true;
       if (capped) break;
     }
     if (it > 100000) break;
    }
    continue;
   }
   // A predecessor outside the component needs no attention: Tarjan emits components
   // successors-first, so such a predecessor lives in a component not yet processed and will
   // read this one's final value when its own turn comes.
   for (int v : comp) in_comp[v] = 1;
   std::deque<int> work(comp.begin(), comp.end());
   for (int v : comp) in_q[v] = 1;
   while (!work.empty() && !capped) {
     if (cancelled()) break;
     const int v = work.front();
     work.pop_front();
     in_q[v] = 0;
     ++it;
     if (it > 100000000) break;
     if (!relax(v)) continue;
     for (int u : rev[v])
       if (in_comp[u] && !in_q[u]) { in_q[u] = 1; work.push_back(u); }
   }
   for (int v : comp) { in_comp[v] = 0; in_q[v] = 0; }
  }
  // Restore the documented postcondition on every exit path. A converged sweep materialised
  // every seed as it went, so this loop does nothing; an aborted one (capped or cancelled) can
  // still hold seeds, and W must denote the iterate the sweep actually reached rather than
  // reading as empty at those nodes.
  for (int v : S)
    if (seeded[v]) { W[v].clear(); W[v].push_back(zcar(v, vsc)); seeded[v] = 0; }
  fixpoint_iterations += it;
  if (std::getenv("HORA_GFP_PROFILE")) {
    int hist[8] = {0,0,0,0,0,0,0,0};
    std::size_t withextra = 0, never = 0;
    for (int v : S) {
      if (!dbg_hadextra[v]) continue;
      ++withextra;
      const int d = dbg_deathround[v];
      if (d == 0) ++never; else hist[d < 7 ? d : 7]++;
    }
    std::fprintf(stderr, "EXTRA-DEATH nodes_with_extras=%zu died_at_round: 1=%d 2=%d 3=%d 4=%d 5=%d 6=%d >=7=%d never=%zu\n",
                 withextra, hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7], never);
  }
  if (std::getenv("HORA_GFP_PROFILE"))
    std::fprintf(stderr, "WIDTH refines=%zu nw_raw=%zu nw_kept=%zu avg_kept=%.1f maxW=%zu "
                 "cand_avg=%.1f cand_max=%zu\n", dbg_refine, dbg_nwraw, dbg_nwkept,
                 dbg_refine ? (double)dbg_nwkept/dbg_refine : 0.0, dbg_wmax,
                 dbg_candn ? (double)dbg_candsum/dbg_candn : 0.0, dbg_candmax);
  if (std::getenv("HORA_GFP_PROFILE")) {
    std::size_t big = 0, nontrivial = 0;
    for (const auto& c : sccs) { big = std::max(big, c.size()); if (c.size() > 1) ++nontrivial; }
    std::size_t stopn = 0, stopmembers = 0, wmembers = 0;
    for (int v : S) {
      if (nosucc_done[v] && !nosucc[v].empty()) { ++stopn; stopmembers += nosucc[v].size(); }
      wmembers += W[v].size();
    }
    std::fprintf(stderr, "GFP-PROFILE |S|=%zu N=%zu sccs=%zu nontrivial=%zu maxcomp=%zu "
                 "visits=%zu pre=%zu stopcalls=%zu stopnodes=%zu stopmembers=%zu wmembers=%zu "
                 "fedmax=%zu live=%zu\n",
                 S.size(), N, sccs.size(), nontrivial, big, it, pre_calls, nosucc_calls, stopn, stopmembers,
                 wmembers, fed_max, live_count);
  }
}

/// @brief Is there a counterexample certificate within @p S?
///
/// @param S  a set of expanded `!q` nodes
/// @return @c true iff the converged confined fixpoint leaves a non-empty value at some node of
///         @p S satisfying `p`
///
/// @post @ref fed_max, @ref fed_avg and @ref nonempty describe the converged federations.
///
/// @warning A @c false return is meaningful only when @ref capped is clear. A capped pass did
///          not converge and proves nothing in either direction; returning @c false for it
///          would launder "out of federation budget" into a silent satisfied. Callers must
///          check @ref capped and report unusable.
bool LivenessEngine::certificate_in(const std::vector<int>& S) {
  if (S.empty()) return false;
  std::vector<Federation> W;
  solve_greatest_fixpoint(S, W);
  std::size_t tot = 0, ne = 0;
  for (int v : S) {
    if (fed_is_empty(W[v])) continue;
    ++ne;
    tot += W[v].size();
    if (W[v].size() > fed_max) fed_max = W[v].size();
  }
  nonempty = ne;
  fed_avg = ne ? double(tot) / double(ne) : 0.0;
  if (capped) return false;                      // caller checks `capped` before believing this
  for (int v : S) {
    if (!is_p(v) || fed_is_empty(W[v])) continue;
    // The exact test: a violation needs a valuation that satisfies `p` AND survives the fixpoint.
    // With a purely discrete `p` the region is the whole zone and this reduces to the old check.
    if (p.forbid.empty()) return true;
    if (!fed_is_empty(fed_intersect(W[v], p_region(v)))) return true;
  }
  return false;
}

/// @brief Construct and intern the initial symbolic state.
///
/// @return @c false iff the initial zone is empty under the initial locations' invariants, in
///         which case no node is stored at all and the caller sees an empty graph
///
/// @post On success node 0 exists and is the (only) root of the search.
/// @note The construction order — invariants, then delay unless the initial location is frozen,
///       then invariants again, then the active-clock reduction, then extrapolation — must match
///       hora::detail::close_succ_zone, or the root would sit in a different abstraction from
///       every state reached from it.
bool LivenessEngine::build_all() {
  SymState s0;
  s0.locs.resize(net.nprocs());
  for (int pr = 0; pr < net.nprocs(); ++pr) s0.locs[pr] = net.procs[pr].initial;
  s0.ints.resize(net.intvars.size());
  for (std::size_t v = 0; v < net.intvars.size(); ++v) s0.ints[v] = net.intvars[v].init;
  s0.zone = Dbm(net.nclocks);
  {
    std::vector<Constraint> inv;
    all_invariants(net, s0.locs, inv);
    if (!apply_all(s0.zone, inv)) return false;     // unreachable init: caller sees no nodes
    bool c, u;
    frozen_flags(net, s0.locs, s0.ints, &c, &u);
    // The forbidden region applies to the root too, and it must replace the delay rather than
    // follow it. This construction does not go through close_succ_zone, and a plain delay
    // followed by a restriction is the very error the psi-restriction exists to avoid: the zone
    // travels through the region first and is only cut afterwards, so the root ends up holding
    // valuations no run can reach without leaving psi. Every verdict is read at the initial
    // vertex, so that alone decides queries wrongly.
    if (avoid_ && avoid_forward_ && avoid_->applies(s0.locs, s0.ints)) {
      Federation parts;
      if (!c && !u) avoid_->delay(s0.zone, s0.locs, s0.ints, parts);
      else avoid_->restrict(s0.zone, s0.locs, s0.ints, parts);
      // The root must be a single zone. Merge only where the union is exactly convex; otherwise
      // keep the largest piece, which under-approximates the root and so can lose a witness but
      // never invent one.
      Dbm merged;
      bool have = false;
      for (Dbm& pz : parts) {
        if (pz.empty()) continue;
        if (!apply_all(pz, inv) || pz.empty()) continue;
        if (!have) { merged = pz; have = true; continue; }
        if (!merged.try_union(pz) && pz.includes(merged)) merged = pz;
      }
      if (!have) return false;
      s0.zone = std::move(merged);
    } else if (!c && !u) {
      s0.zone.delay();
      apply_all(s0.zone, inv);
    }
    reset_inactive(net, s0.locs, s0.zone, qmask);
    if (lazy_mode_) lazy_root_exact_ = s0.zone;      // before extrapolation; see its declaration
    if (emode == ExLuPlus) s0.zone.extrapolate_lu_plus(L, U);
    else if (emode == ExNoDiag) s0.zone.extrapolate_lu_plus_nodiag(L, U);
  }
  if (online_d_) {
    // Unlike ordinary intern(), this retains the exact root even if a future key-level atom
    // will make its first cover certificate invalid.  The root is registered before it is
    // compared, so a local Dead(root-key) seed participates from the very first D decision.
    const int root_key = online_register_key(s0.locs, s0.ints);
    online_drain_atoms();
    online_recheck_dirty();
    online_intern(std::move(s0), root_key);
  } else if (amap_) {
    std::vector<SymState> pieces;
    split_by_atoms(std::move(s0), pieces);
    for (auto& pc2 : pieces) intern(std::move(pc2));
  } else {
    intern(std::move(s0));
  }
  return true;
}


/// @brief The `!q` nodes that can influence the verdict.
///
/// @return every expanded node reachable through `!q` edges only from an expanded `p /\ !q`
///         node, the accepting nodes themselves included
///
/// @note Restricting the final pass to this set is exact, not an approximation. The operator is
///       defined per node from that node's own successors, so an accepting node's value depends
///       on exactly this closure; and `p --> q` is decided by the accepting nodes alone, so
///       nothing outside it is ever read.
std::vector<int> LivenessEngine::accepting_closure() {
  grow();
  const std::size_t N = nrec();
  std::vector<char> in(N, 0);
  std::vector<int> stk, out;
  for (std::size_t v = 0; v < N; ++v)
    if (expanded[v] && is_nq((int)v) && is_p((int)v)) { in[v] = 1; stk.push_back((int)v); }
  while (!stk.empty()) {
    const int v = stk.back();
    stk.pop_back();
    out.push_back(v);
    for (const auto& e : adj[v])
      if (!in[e.tgt] && expanded[e.tgt] && is_nq(e.tgt)) { in[e.tgt] = 1; stk.push_back(e.tgt); }
  }
  return out;
}

/// @brief Decide the query by one exact backward pass over the finished graph.
///
/// @return @c true iff `p --> q` is violated
///
/// @pre The forward construction has terminated, so the accepting closure is complete and the
///      confined operator coincides with the full one at every accepting node.
/// @warning Check @ref capped before believing a @c false return; see ::LivenessEngine::certificate_in.
bool LivenessEngine::final_verdict_violated() {
  return certificate_in(accepting_closure());
}

/// @brief Reference-only check for whether a finished graph needs any D atom.
///
/// No public liveness route calls this method.  The production construction seeds the same
/// condition as each key is discovered.  This version remains useful for comparing the online
/// seeding rule against a finished graph.
///
/// A key contributes no atom when `Dead(q)` is empty, covers its whole invariant, or time can
/// diverge from that non-frozen key.  The check is structural: it evaluates `Dead(q)` over the
/// key invariant rather than over individual reached zones.

}  // namespace hora
