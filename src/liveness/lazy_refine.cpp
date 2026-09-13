/// @file src/liveness/lazy_refine.cpp
/// @brief Demand-driven repair of the covers the exact backward operator cannot justify.
///
/// The forward search accepts a cover `X <=_D Z_tgt` and discards the exact successor
/// `X = post_e(Z_src)`.  Covering is a simulation, not inclusion, so `X` may be disjoint from
/// `Z_tgt`, and then `Z_src /\ Pre_e(W_tgt)` is empty even though a real run goes through `X`.
/// Simulation preserves the continuation *from* `X`; it does not put the same predecessor
/// valuation into `Z_tgt`.  So an empty greatest fixpoint is not by itself a `NOT`: it is a `NOT`
/// only once every covered edge that could carry a recurrent witness has been accounted for.
///
/// The finite half needs none of this.  `NoSucc(v)` is computed exactly on `Z_v` and enters the
/// operator as a union term, so a deadlock or divergence witness survives every refinement.  Only
/// cycles are owed, which is what confines the obligations below to recurrent components.
///
/// Nothing here widens a zone.  The backward operator keeps exact stored-zone semantics
/// throughout; the repair is to make the graph exact where it matters, not to relax the operator.

#include "hora/config.hpp"
#include "hora/zone.hpp"
#include "hora/detail/liveness_engine.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>
#include "hora/detail/semantics.hpp"
#include "src/liveness/internal.hpp"

namespace hora {

namespace {

/// @brief Wall-clock accumulator in milliseconds, so the phase timings sum to the loop's cost.
struct Phase {
  double* sink;
  std::chrono::steady_clock::time_point t0;
  explicit Phase(double* s) : sink(s), t0(std::chrono::steady_clock::now()) {}
  ~Phase() {
    *sink += std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::steady_clock::now() - t0).count() / 1000.0;
  }
};

/// @brief How many refinements are allowed before the loop gives up, as a multiple of the graph.
///
/// `emode` is ::ExNone on this route, so a_LU/D subsumption is the ONLY reason the forward search
/// terminates -- there is no extrapolation underneath it.  Forbidding covers removes that
/// argument, so an unbounded loop has no finiteness proof at all and the budget is load-bearing
/// rather than a tuning knob.  Exhausting it delegates the verdict to the exact route, which
/// covers by inclusion and is therefore complete on its own.
std::size_t budget_multiple() {
  if (const char* e = std::getenv("HORA_LAZY_BUDGET")) {
    const long v = std::atol(e);
    if (v > 0) return static_cast<std::size_t>(v);
  }
  return 4;
}

}  // namespace

bool LivenessEngine::lazy_snapshot(std::vector<std::vector<OtfEdge>>* saved_adj,
                                   std::vector<int>* saved_key, std::size_t* lost) {
  const std::size_t n = nrec();
  if (lost) *lost = 0;
  if (n == 0 || online_raw_slot_.size() < n || online_key_of_.size() < n) return false;

  std::vector<std::vector<OtfEdge>> snap(n);
  for (std::size_t v = 0; v < n; ++v) {
    if (!expanded[v] || online_raw_slot_[v] < 0) continue;
    const std::vector<OtfEdge>& raw = online_raw_adj_[online_raw_slot_[v]];
    std::vector<OtfEdge>& out = snap[v];
    out.reserve(raw.size());
    for (const OtfEdge& e : raw) {
      int t = e.tgt;
      if (t < 0 || t >= static_cast<int>(n) || !expanded[t] || online_raw_slot_[t] < 0)
        t = online_resolve_cover(e.tgt, /*require_expanded=*/true);
      if (t < 0 || t >= static_cast<int>(n)) {
        // The edge is DROPPED, and the snapshot is then a truncation of the real graph rather
        // than a view of it.  For the backward operator that is conservative -- a missing
        // `Pre_e` term only shrinks `W` -- but a cycle search reads a missing edge as a missing
        // cycle, and a truncated graph can answer "no witness" for a query that has one.  So the
        // losses are counted, and the caller decides what its own conclusion is worth.
        //
        // A target that FAILS the query predicate is not a loss: under
        // ::LivenessEngine::online_eg_prune_ it is deliberately left unexpanded, and no witness
        // can pass through it, so neither route ever wanted the edge.
        if (lost && !(online_eg_prune_ && !online_is_nq(e.tgt))) ++*lost;
        continue;
      }
      // The classification travels with the edge.  ::LivenessEngine::solve_greatest_fixpoint does
      // not read it -- the backward operator is unchanged, and an unresolved cover still
      // contributes `Pre_e(W_tgt)`, which is exactly the term that comes out empty and creates
      // the obligation.  It is read by ::LivenessEngine::lazy_collect afterwards.
      out.push_back(OtfEdge{t, e.tag, e.deferred, e.ordinal});
    }
  }
  saved_adj->swap(adj);
  adj.swap(snap);
  saved_key->swap(node_key_);
  node_key_.assign(online_key_of_.begin(), online_key_of_.begin() + n);
  return true;
}

void LivenessEngine::lazy_restore(std::vector<std::vector<OtfEdge>>* saved_adj,
                                  std::vector<int>* saved_key) {
  node_key_.swap(*saved_key);
  adj.swap(*saved_adj);
  saved_adj->clear();
  saved_key->clear();
}

void LivenessEngine::lazy_build_key_sccs(std::vector<int>* comp, std::vector<char>* comp_live,
                                         const std::vector<char>* nodes) {
  Phase phase(&lazy_.ms_scc);
  ++lazy_.scc_builds;
  const std::size_t nk = online_key_by_id_.size();
  comp->assign(nk, -1);
  comp_live->clear();
  if (nk == 0) return;

  // The key-projected adjacency, deduplicated per source.  Built from the raw edges rather than
  // read out of `online_key_edges_`: that skeleton exists only once D has an atom to pull back
  // (::LivenessEngine::online_enable_key_graph), and D finds no atom at all on the models this
  // loop is for, so it is empty exactly when it would be needed.
  // ONLY THE KEYS A WITNESS MAY LIVE IN.  A maximal `E[] psi` run satisfies psi at every step,
  // and a `p --> q` counterexample is inside `!q` from the accepting node onwards, so a key where
  // the query predicate fails lies on no witness and neither do its edges.  Dropping them splits
  // the key graph into smaller components, and BOTH conjuncts below get easier: fewer keys for
  // one clock to be bounded at, fewer transitions that could reset it.  The reasoning is
  // unchanged -- an infinite witness run's key sequence is an infinite path in this restricted
  // graph, so it still settles inside one of its SCCs.
  //
  // `E[] psi` barely notices: ::LivenessEngine::online_eg_prune_ already leaves a non-psi record
  // unexpanded, so its key contributes no edges either way.  `p --> q` has no such prune -- the
  // path to the `p`-node is ordinary reachability that may leave `!q` -- and without this the
  // whole reachable graph is one component the criterion cannot kill: grc4 `T0.in --> T0.far`
  // reported `key-sccs=1 dead=0`, C5 then condemned every cover on a cycle, and refinement walked
  // toward the containment graph (15 rounds, 37,656 refinements, 1,335 -> 28,405 records) while
  // grc5 did not finish at all.
  // A record contributes at all only if @p nodes admits it.  The caller passes the accepting
  // closure, which on a leads-to query is a tiny slice of the graph -- 512 records of 624,452 on
  // twolock10 `P0.reqB --> P0.waitB` -- while this function otherwise walks every record twice
  // and cost 876 ms of that query's 915 ms route time.
  const auto admits = [&](std::size_t v) {
    return !nodes || (v < nodes->size() && (*nodes)[v]);
  };
  std::vector<char> key_ok(nk, 1);
  {
    detail::SymState ks;
    for (std::size_t k = 0; k < nk; ++k) {
      key_state(static_cast<int>(k), ks);
      key_ok[k] = holds(q, ks) ? 0 : 1;
    }
  }
  // With a filter, a key is in play only if some admitted record carries it.  Without this the
  // Tarjan below still roots at every `!q` key in the whole graph.
  if (nodes) {
    std::vector<char> key_used(nk, 0);
    for (std::size_t v = 0; v < nrec(); ++v)
      if (admits(v) && expanded[v] && online_raw_slot_[v] >= 0) {
        const int kv = online_key_of_[v];
        if (kv >= 0 && kv < static_cast<int>(nk)) key_used[kv] = 1;
      }
    for (std::size_t k = 0; k < nk; ++k)
      if (!key_used[k]) key_ok[k] = 0;
  }

  std::vector<std::vector<int>> kadj(nk);
  std::vector<std::vector<int>> ktag(nk);           // the transition behind each key edge
  {
    std::vector<int> seen(nk, -1);
    for (std::size_t v = 0; v < nrec(); ++v) {
      if (!admits(v) || !expanded[v] || online_raw_slot_[v] < 0) continue;
      const int kv = online_key_of_[v];
      if (kv < 0 || kv >= static_cast<int>(nk) || !key_ok[kv]) continue;
      for (const OtfEdge& e : online_raw_adj_[online_raw_slot_[v]]) {
        const int t = (e.tgt >= 0 && e.tgt < static_cast<int>(nrec()) && admits(e.tgt))
                          ? online_key_of_[e.tgt] : -1;
        if (t < 0 || t >= static_cast<int>(nk) || !key_ok[t] || seen[t] == kv) continue;
        seen[t] = kv;
        kadj[kv].push_back(t);
        ktag[kv].push_back(e.tag);
      }
    }
  }

  // Iterative Tarjan; the key graph reaches 10^5 nodes and the recursive form would not survive
  // the stack, exactly as in ::LivenessEngine::solve_greatest_fixpoint.
  std::vector<int> idx(nk, -1), low(nk, 0), stk;
  std::vector<char> on(nk, 0);
  std::vector<std::pair<int, std::size_t>> call;
  std::vector<int> members;                       // component sizes, 0 when provably acyclic
  int next = 0, ncomp = 0;
  for (std::size_t root = 0; root < nk; ++root) {
    if (idx[root] != -1 || !key_ok[root]) continue;   // no witness passes through it
    call.push_back({static_cast<int>(root), 0});
    idx[root] = low[root] = next++;
    stk.push_back(static_cast<int>(root));
    on[root] = 1;
    while (!call.empty()) {
      auto& [v, ei] = call.back();
      if (ei < kadj[v].size()) {
        const int w = kadj[v][ei++];
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
          int size = 0;
          for (;;) {
            const int w = stk.back();
            stk.pop_back();
            on[w] = 0;
            (*comp)[w] = ncomp;
            ++size;
            if (w == v) break;
          }
          // A singleton with a self-loop recurs just as a larger component does.
          if (size == 1) {
            bool loop = false;
            for (int w : kadj[v]) if (w == v) { loop = true; break; }
            if (!loop) size = 0;                  // acyclic: no infinite run can stay here
          }
          members.push_back(size);
          ++ncomp;
        }
        const int child = v;
        call.pop_back();
        if (!call.empty()) low[call.back().first] = std::min(low[call.back().first], low[child]);
      }
    }
  }

  // ---- which components can host an infinite psi-run ---------------------------------------
  //
  // A component is DEAD -- owes nothing -- when BOTH hold:
  //
  //  (a) some clock is bounded above at every key of the component and is reset by NO transition
  //      inside it.  Along a run confined there that clock never decreases and never passes its
  //      bound, so no TIME-DIVERGENT run can stay in the component.
  //
  //  (b) every cycle of the component must let time pass.  Checked by deleting every edge that
  //      both lower-guards `z >= c`, c > 0, AND resets `z` -- taking such an edge twice forces c
  //      time units in between -- and asking whether what remains is acyclic.  If it is, every
  //      cycle uses at least one deleted edge and therefore burns time.
  //
  // Neither conjunct alone is sound.  Without (b) a ZENO cycle survives: `E[] psi` as implemented
  // here AND by UPPAAL 5.0.0 accepts a run that fires forever with time frozen, so a component
  // where time merely cannot diverge may still carry a witness.  Without (a) a component that
  // resets its bounded clocks runs forever quite legitimately.
  //
  // (b) MUST be per-cycle, not per-component.  An earlier version asked only whether some clock
  // was reset somewhere in the component and lower-guarded somewhere in it; that admits a reset
  // on one cycle and a guard on another, and single4 -- one location, self-loops `xi >= 1, xi = 0`
  // -- was wrongly declared dead, turning a SAT into a NOT against both the baseline and verifyta.
  //
  // The masks are read from EVERY edge, not from the key-graph representative.  `kadj` above is
  // deduplicated by target because Tarjan needs nothing more, but four self-loops that reset four
  // different clocks collapse to one entry there, and reading resets off that entry sees one
  // clock instead of four -- which is the other half of the same single4 bug.
  comp_live->assign(ncomp, 1);
  for (std::size_t k = 0; k < nk; ++k)
    if ((*comp)[k] >= 0 && members[(*comp)[k]] == 0) (*comp_live)[(*comp)[k]] = 0;
  if (net.nclocks > 63 || ncomp == 0) return;   // no room for the mask: keep every component

  const std::uint64_t all_clocks = net.nclocks >= 63
                                       ? ~std::uint64_t(0) << 1
                                       : (((std::uint64_t(1) << net.nclocks) - 1) << 1);
  // Per-tag masks, computed once: tags are pooled by value, so two edges with the same recipe
  // share them.
  std::vector<std::uint64_t> tag_reset(tags_.size(), 0);
  std::vector<char> tag_burns(tags_.size(), 0);
  for (std::size_t t = 0; t < tags_.size(); ++t) {
    std::uint64_t r = 0, lo = 0;
    for (const auto& pe : tags_[t].parts) {
      const auto& ed = net.procs[pe.first].edges[pe.second];
      for (int c : ed.clock_resets)
        if (c >= 1 && c <= net.nclocks) r |= std::uint64_t(1) << c;
      // `z >= c` with c > 0 is stored as `0 - z <= -c`: the i == 0 side, strictly negative bound.
      for (const Constraint& g : ed.clock_guard)
        if (g.i == 0 && g.j > 0 && g.j <= net.nclocks && raw_bound(g.bound) < 0)
          lo |= std::uint64_t(1) << g.j;
    }
    tag_reset[t] = r;
    tag_burns[t] = (r & lo) != 0;
  }

  // (a)'s two masks, and the residual adjacency for (b).
  std::vector<std::uint64_t> bounded(ncomp, all_clocks), reset(ncomp, 0);
  std::vector<Constraint> inv;
  detail::SymState ksc;
  for (std::size_t k = 0; k < nk; ++k) {
    const int c = (*comp)[k];
    if (c < 0 || !key_ok[k]) continue;
    key_state(static_cast<int>(k), ksc);
    all_invariants(net, ksc.locs, inv);
    std::uint64_t here = 0;
    for (const Constraint& ct : inv)          // `x <= c` only: a difference or lower bound
      if (ct.i > 0 && ct.j == 0 && ct.i <= net.nclocks && ct.bound < kInf)  // anchors nothing
        here |= std::uint64_t(1) << ct.i;
    bounded[c] &= here;
  }

  // (b) is checked PER CLOCK, and per cycle.
  //
  // Tripakis' condition: some clock `z` is reset on every cycle of the component and guarded
  // `z >= c` on every cycle.  Then each traversal forces c time units between the reset and the
  // guard -- or, if the guard precedes the reset in cycle order, between two traversals -- so an
  // infinite run through the component diverges in time.
  //
  // "Every cycle contains an edge of kind K" is checked by DELETING the K-edges and asking
  // whether what remains is acyclic.  Two such tests per clock, both O(edges).
  //
  // Asking instead for one edge that resets AND guards the same clock -- the obvious cheap
  // version -- is far too strict: fischer resets `x_i` on one edge and guards it on a later one,
  // so no single edge qualifies, every component stays live, and refinement then blows the graph
  // from 17,215 records to 238,180.
  std::vector<std::uint64_t> tag_lower(tags_.size(), 0);
  for (std::size_t t = 0; t < tags_.size(); ++t) {
    std::uint64_t lo = 0;
    for (const auto& pe : tags_[t].parts)
      for (const Constraint& g : net.procs[pe.first].edges[pe.second].clock_guard)
        if (g.i == 0 && g.j > 0 && g.j <= net.nclocks && raw_bound(g.bound) < 0)
          lo |= std::uint64_t(1) << g.j;
    tag_lower[t] = lo;
  }

  // Intra-component key edges, kept once per (source, target, tag) so the per-clock passes below
  // can filter them without touching the records again.
  struct KEdge { int src, tgt, tag; };
  std::vector<KEdge> kedges;
  {
    std::unordered_set<std::uint64_t> seen_edge;
    for (std::size_t v = 0; v < nrec(); ++v) {
      if (!expanded[v] || online_raw_slot_[v] < 0) continue;
      const int kv = online_key_of_[v];
      if (!admits(v) || kv < 0 || kv >= static_cast<int>(nk) || !key_ok[kv] ||
          (*comp)[kv] < 0) continue;
      for (const OtfEdge& e : online_raw_adj_[online_raw_slot_[v]]) {
        if (e.tgt < 0 || e.tgt >= static_cast<int>(nrec()) || !admits(e.tgt)) continue;
        const int kt = online_key_of_[e.tgt];
        if (kt < 0 || kt >= static_cast<int>(nk) || !key_ok[kt]) continue;
        const int c = (*comp)[kv];
        if ((*comp)[kt] != c) continue;     // leaves the component: on no cycle inside it
        if (e.tag < 0 || e.tag >= static_cast<int>(tags_.size())) continue;
        reset[c] |= tag_reset[e.tag];       // EVERY edge, not the key-graph representative
        const std::uint64_t sig = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(kv)) << 40) ^
                                  (static_cast<std::uint64_t>(static_cast<std::uint32_t>(kt)) << 16) ^
                                  static_cast<std::uint64_t>(static_cast<std::uint32_t>(e.tag));
        if (seen_edge.insert(sig).second) kedges.push_back(KEdge{kv, kt, e.tag});
      }
    }
  }

  // Is the component acyclic once the edges selected by `drop` are removed?
  std::vector<char> cyclic(ncomp, 0);
  std::vector<std::vector<int>> sub(nk);
  std::vector<int> idx2(nk, -1), low2(nk, 0), stk2;
  std::vector<char> on2(nk, 0);
  std::vector<std::pair<int, std::size_t>> call2;
  auto residual_cyclic = [&](auto&& drop) {
    for (auto& a : sub) a.clear();
    for (const KEdge& ke : kedges)
      if (!drop(ke.tag)) sub[ke.src].push_back(ke.tgt);
    std::fill(cyclic.begin(), cyclic.end(), 0);
    std::fill(idx2.begin(), idx2.end(), -1);
    std::fill(on2.begin(), on2.end(), 0);
    stk2.clear();
    int next2 = 0;
    for (std::size_t root = 0; root < nk; ++root) {
      if (idx2[root] != -1 || (*comp)[root] < 0) continue;
      call2.clear();
      call2.push_back({static_cast<int>(root), 0});
      idx2[root] = low2[root] = next2++;
      stk2.push_back(static_cast<int>(root));
      on2[root] = 1;
      while (!call2.empty()) {
        auto& [v, ei] = call2.back();
        if (ei < sub[v].size()) {
          const int w = sub[v][ei++];
          if (idx2[w] == -1) {
            idx2[w] = low2[w] = next2++;
            stk2.push_back(w);
            on2[w] = 1;
            call2.push_back({w, 0});
          } else if (on2[w]) {
            low2[v] = std::min(low2[v], idx2[w]);
          }
        } else {
          if (low2[v] == idx2[v]) {
            int size = 0, any = v;
            for (;;) {
              const int w = stk2.back();
              stk2.pop_back();
              on2[w] = 0;
              any = w;
              ++size;
              if (w == v) break;
            }
            bool loop = false;
            if (size == 1)
              for (int w : sub[v]) if (w == v) { loop = true; break; }
            if (size > 1 || loop) cyclic[(*comp)[any]] = 1;
          }
          const int child = v;
          call2.pop_back();
          if (!call2.empty())
            low2[call2.back().first] = std::min(low2[call2.back().first], low2[child]);
        }
      }
    }
  };

  // (b) EVERY CYCLE MUST BURN TIME, with different cycles allowed to use different clocks.
  //
  // Tripakis' condition asks for ONE clock reset and guarded on every cycle of the component.  A
  // network never has one: fischer's process 3 cycles resetting x3, process 5 resetting x5, so
  // the single-clock test fails on models where every cycle plainly burns time.  Checking it per
  // PROCESS instead is sound but blind to time forced across processes -- one process's invariant
  // against another's guard on a shared clock -- and arbiter is exactly that shape.
  //
  // The exact characterisation: a cycle is NON-burning iff, for every clock z, it avoids the
  // reset edges R_z or avoids the lower-guard edges G_z.  So
  //
  //     every cycle burns  <=>  for every choice f : z -> {R_z, G_z},
  //                             the subgraph  E \ union_z f(z)  is acyclic.
  //
  // Only clocks BOTH reset and guarded inside the component can make a cycle burn, so the choice
  // ranges over those alone.  And removing more edges can only destroy cycles, never create one,
  // so a DFS over the choices prunes the moment the current subgraph is acyclic -- every
  // extension of that branch is acyclic too.  In practice the search collapses immediately.
  std::vector<char> burns(ncomp, 0);
  {
    // Intra-component edges per component, and the clocks that can burn there.
    std::vector<std::vector<std::size_t>> inside(ncomp);
    std::vector<std::uint64_t> comp_reset(ncomp, 0), comp_lower(ncomp, 0);
    for (std::size_t i = 0; i < kedges.size(); ++i) {
      const int c = (*comp)[kedges[i].src];
      if (c < 0 || (*comp)[kedges[i].tgt] != c) continue;
      inside[c].push_back(i);
      comp_reset[c] |= tag_reset[kedges[i].tag];
      comp_lower[c] |= tag_lower[kedges[i].tag];
    }
    std::vector<char> removed(kedges.size(), 0);
    std::vector<int> st(nk, 0);
    std::vector<std::pair<int, std::size_t>> dfs;
    std::vector<std::vector<std::size_t>> g(nk);
    // Does component `c` still contain a cycle once `drop` has removed its edges?
    auto cyclic_after = [&](int c, auto&& drop) {
      for (std::size_t i : inside[c]) { g[kedges[i].src].clear(); g[kedges[i].tgt].clear(); }
      for (std::size_t i : inside[c])
        if (!drop(i)) g[kedges[i].src].push_back(i);
      for (std::size_t i : inside[c]) { st[kedges[i].src] = 0; st[kedges[i].tgt] = 0; }
      for (std::size_t i0 : inside[c]) {
        if (st[kedges[i0].src]) continue;
        dfs.clear();
        dfs.push_back({kedges[i0].src, 0});
        st[kedges[i0].src] = 1;
        while (!dfs.empty()) {
          auto& [v, ei] = dfs.back();
          if (ei < g[v].size()) {
            const int w = kedges[g[v][ei++]].tgt;
            if (st[w] == 1) return true;
            if (st[w] == 0) { st[w] = 1; dfs.push_back({w, 0}); }
          } else { st[v] = 2; dfs.pop_back(); }
        }
      }
      return false;
    };
    // After `drop`, does any edge satisfying `probe` lie on a cycle?  Both ends in one nontrivial
    // SCC of the surviving subgraph is exactly that.
    auto cyclic_through = [&](int c, auto&& drop, auto&& probe) {
      std::vector<std::size_t> keep;
      for (std::size_t i : inside[c]) if (!drop(i)) keep.push_back(i);
      for (std::size_t i : keep) { g[kedges[i].src].clear(); g[kedges[i].tgt].clear(); }
      for (std::size_t i : keep) g[kedges[i].src].push_back(i);
      // Tarjan over the surviving subgraph, recording component ids in `st`.
      std::vector<int> idx(nk, -1), low(nk, 0), stk;
      std::vector<char> on(nk, 0);
      std::vector<int> cid(nk, -1);
      std::vector<char> big;
      std::vector<std::pair<int, std::size_t>> call;
      int next = 0, nc = 0;
      for (std::size_t i : keep) {
        const int r = kedges[i].src;
        if (idx[r] != -1) continue;
        call.push_back({r, 0});
        idx[r] = low[r] = next++; stk.push_back(r); on[r] = 1;
        while (!call.empty()) {
          auto& [v, ei] = call.back();
          if (ei < g[v].size()) {
            const int w = kedges[g[v][ei++]].tgt;
            if (idx[w] == -1) { idx[w] = low[w] = next++; stk.push_back(w); on[w] = 1; call.push_back({w, 0}); }
            else if (on[w]) low[v] = std::min(low[v], idx[w]);
          } else {
            if (low[v] == idx[v]) {
              int sz = 0;
              for (;;) { const int w = stk.back(); stk.pop_back(); on[w] = 0; cid[w] = nc; ++sz; if (w == v) break; }
              bool loop = false;
              if (sz == 1) for (std::size_t j : g[v]) if (kedges[j].tgt == v) { loop = true; break; }
              big.push_back(sz > 1 || loop);
              ++nc;
            }
            const int ch = v;
            call.pop_back();
            if (!call.empty()) low[call.back().first] = std::min(low[call.back().first], low[ch]);
          }
        }
      }
      for (std::size_t i : keep)
        if (probe(i) && cid[kedges[i].src] >= 0 && cid[kedges[i].src] == cid[kedges[i].tgt] &&
            big[cid[kedges[i].src]])
          return true;
      return false;
    };

    for (int c = 0; c < ncomp; ++c) {
      if (members[c] == 0 || inside[c].empty()) continue;
      // Two polynomial tests, together sufficient for "every cycle burns time".
      //
      //   (1) every cycle resets SOMETHING: deleting every reset edge leaves the component
      //       acyclic;
      //   (2) for each clock z, every cycle that resets z also guards it: in the component minus
      //       the z-guard edges, no z-reset edge lies on a cycle.
      //
      // A cycle then resets some z (1) and guards that same z (2), so each traversal forces the
      // guard's constant to elapse between the reset and the guard -- or, if the guard precedes
      // the reset in cycle order, between two traversals.  Either way time diverges.
      //
      // This replaces the exact characterisation, which is inherently exponential: a cycle is
      // non-burning iff for every z it avoids R_z or avoids G_z, so deciding it means recursing
      // `all_burn(C) = all_burn(C \ R_z) and all_burn(C \ G_z)` over all clocks.  With pruning
      // that collapses on most models but not on arbiter12, where it cost 3.59s of a 3.60s query.
      // These two tests are O(clocks * edges) with no search at all, and they are still sharp
      // enough for the shape that matters -- fischer resets x_p and guards it several edges later
      // on the same cycle, which the single-clock Tripakis test misses and (2) catches.
      const std::uint64_t both = comp_reset[c] & comp_lower[c];
      if (!both) continue;                               // nothing here can burn: (b) fails
      // (1)
      if (cyclic_after(c, [&](std::size_t i) { return tag_reset[kedges[i].tag] != 0; })) continue;
      // (2)
      bool ok = true;
      for (int z = 1; z <= net.nclocks && z < 64 && ok; ++z) {
        const std::uint64_t bit = std::uint64_t(1) << z;
        if (!(comp_reset[c] & bit)) continue;            // z is not reset here: nothing to check
        // Remove the z-guard edges, then ask whether any z-reset edge sits on a surviving cycle.
        if (cyclic_through(c, [&](std::size_t i) { return (tag_lower[kedges[i].tag] & bit) != 0; },
                           [&](std::size_t i) { return (tag_reset[kedges[i].tag] & bit) != 0; }))
          ok = false;
      }
      burns[c] = ok ? 1 : 0;
      // HORA_FWD_EXACT decides the components the cheap tests could not kill, exactly.  OFF by
      // default: measured, those components are precisely the ones where the exact search is
      // expensive, so turning it on trades 51 agree / 0 wrong / 4 timeout for more timeouts, not
      // fewer -- it moves cost from refinement into the criterion without removing it.  Kept
      // because it is the exact characterisation and the right starting point if the search can
      // be made cheaper.
      // `--criterion poly` stops here; the other two settle what the cheap tests left live.
      const auto mode = detail::options().criterion;
      if (ok || mode == detail::EngineOptions::Criterion::Poly) continue;

      // Branch only on clocks that ACTUALLY conflict in the graph.
      //
      // A cycle is non-burning exactly when the clocks it resets and the clocks it lower-guards
      // are disjoint: a clock both reset and guarded on the cycle forces the guard's constant to
      // elapse on every traversal.  So, on a set of edges:
      //
      //   * take its SCCs.  For one SCC, union the resets and union the guards over its edges.
      //     DISJOINT means every cycle inside it is non-burning -- its masks are subsets of these
      //     -- so a witness exists and the component is live, with no search at all.
      //   * otherwise some clock z lies in both unions.  Every cycle either avoids R_z or avoids
      //     G_z, so recurse on the two subgraphs and require both to burn.
      //   * acyclic bottoms out as "burns": no cycle left to be non-burning.
      //
      // This is the exact characterisation, but the branching clock comes from a conflict the
      // graph actually contains rather than from enumerating all 2^k assignments -- most of which
      // correspond to no cycle at all.  The cheap tests above are what keep it from being reached:
      // test (1) failing is already a complete proof (a cycle with no reset is non-burning for
      // every clock) and returns without any search.
      std::vector<char> gone(kedges.size(), 0);
      std::function<bool(const std::vector<std::size_t>&)> all_burn =
          [&](const std::vector<std::size_t>& es) -> bool {
        if (es.empty()) return true;
        // SCCs of `es`, by the same iterative Tarjan used elsewhere.
        std::vector<std::vector<std::size_t>> g2(nk);
        for (std::size_t i : es) g2[kedges[i].src].push_back(i);
        std::vector<int> idx(nk, -1), low(nk, 0), stk, cid(nk, -1);
        std::vector<char> on(nk, 0);
        std::vector<std::pair<int, std::size_t>> call;
        int next = 0, nc = 0;
        std::vector<char> big;
        for (std::size_t i : es) {
          const int r = kedges[i].src;
          if (idx[r] != -1) continue;
          call.push_back({r, 0});
          idx[r] = low[r] = next++; stk.push_back(r); on[r] = 1;
          while (!call.empty()) {
            auto& [v, ei] = call.back();
            if (ei < g2[v].size()) {
              const int w = kedges[g2[v][ei++]].tgt;
              if (idx[w] == -1) { idx[w] = low[w] = next++; stk.push_back(w); on[w] = 1; call.push_back({w, 0}); }
              else if (on[w]) low[v] = std::min(low[v], idx[w]);
            } else {
              if (low[v] == idx[v]) {
                int sz = 0;
                for (;;) { const int w = stk.back(); stk.pop_back(); on[w] = 0; cid[w] = nc; ++sz; if (w == v) break; }
                bool loop = false;
                if (sz == 1) for (std::size_t j : g2[v]) if (kedges[j].tgt == v) { loop = true; break; }
                big.push_back(sz > 1 || loop);
                ++nc;
              }
              const int ch = v;
              call.pop_back();
              if (!call.empty()) low[call.back().first] = std::min(low[call.back().first], low[ch]);
            }
          }
        }
        // Each nontrivial SCC decided on its own.
        for (int k2 = 0; k2 < nc; ++k2) {
          if (!big[k2]) continue;
          std::vector<std::size_t> sub;
          std::uint64_t R = 0, G = 0;
          for (std::size_t i : es)
            if (cid[kedges[i].src] == k2 && cid[kedges[i].tgt] == k2) {
              sub.push_back(i);
              R |= tag_reset[kedges[i].tag];
              G |= tag_lower[kedges[i].tag];
            }
          if (sub.empty()) continue;
          const std::uint64_t conflict = R & G;
          if (!conflict) return false;                 // every cycle here is non-burning
          int z = 1;
          while (z < 64 && !(conflict & (std::uint64_t(1) << z))) ++z;
          const std::uint64_t bit = std::uint64_t(1) << z;
          std::vector<std::size_t> noR, noG;
          for (std::size_t i : sub) {
            if (!(tag_reset[kedges[i].tag] & bit)) noR.push_back(i);
            if (!(tag_lower[kedges[i].tag] & bit)) noG.push_back(i);
          }
          if (!all_burn(noR) || !all_burn(noG)) return false;
        }
        return true;
      };
      burns[c] = all_burn(inside[c]) ? 1 : 0;
    }
  }


  for (int c = 0; c < ncomp; ++c) {
    if (!(*comp_live)[c]) continue;                        // already acyclic
    const bool time_pinned = (bounded[c] & ~reset[c]) != 0;   // (a)
    if (time_pinned && burns[c]) (*comp_live)[c] = 0;         // (b)
  }
  if (std::getenv("HORA_LAZY_DBG")) {
    std::size_t dead = 0, fail_a = 0, fail_b = 0, fail_both = 0, biggest = 0;
    for (int c = 0; c < ncomp; ++c) {
      dead += !(*comp_live)[c];
      if (!(*comp_live)[c] || members[c] == 0) continue;
      const bool a_ok = (bounded[c] & ~reset[c]) != 0;
      if (a_ok && !burns[c]) ++fail_b;
      else if (!a_ok && burns[c]) ++fail_a;
      else ++fail_both;
      biggest = std::max(biggest, static_cast<std::size_t>(members[c]));
    }
    std::fprintf(stderr, "LAZY live-why: fail_a=%zu fail_b=%zu fail_both=%zu biggest_live=%zu\n",
                 fail_a, fail_b, fail_both, biggest);
    std::fprintf(stderr, "LAZY key-sccs=%d dead=%zu (acyclic, or time-pinned with every cycle "
                 "burning time)\n", ncomp, dead);
  }
}

void LivenessEngine::lazy_collect(const std::vector<int>& S, const std::vector<Federation>& W,
                                  const std::vector<int>& key_comp,
                                  const std::vector<char>& comp_live,
                                  std::vector<LazyObligation>* out) {
  out->clear();
  // Censused here rather than after the seal: ::LivenessEngine::online_finalize_graph resets
  // every mark, so a count taken from the final `adj` reports every edge as exact.
  const bool census = lazy_.rounds == 0;
  if (census) { lazy_.edges_exact = 0; lazy_.edges_covered = 0; lazy_.covered_in_scc = 0; }
  const std::size_t n = nrec();
  std::vector<char> inS(n, 0);
  for (int v : S) if (v >= 0 && v < static_cast<int>(n)) inS[v] = 1;

  // Distance from the seed inside the psi-closure, for the "nearest to the query seed"
  // tie-break: the obligations that matter first are the ones the seed reaches soonest.
  std::vector<int> depth(n, -1);
  if (!S.empty() && inS[0]) {
    std::deque<int> q{0};
    depth[0] = 0;
    while (!q.empty()) {
      const int v = q.front();
      q.pop_front();
      for (const OtfEdge& e : adj[v])
        if (inS[e.tgt] && depth[e.tgt] < 0) { depth[e.tgt] = depth[v] + 1; q.push_back(e.tgt); }
    }
  }

  for (int v : S) {
    if (v < 0 || v >= static_cast<int>(n) || online_raw_slot_[v] < 0) continue;
    // A source already certified winning cannot be helped by refining its covers, and the
    // verdict does not need it: something at `v` already survives the fixpoint.
    if (!W.empty() && v < static_cast<int>(W.size()) && !fed_is_empty(W[v])) continue;
    const std::vector<OtfEdge>& raw = online_raw_adj_[online_raw_slot_[v]];
    for (std::size_t pos = 0; pos < raw.size(); ++pos) {
      const OtfEdge& e = raw[pos];
      if (census) { if (e.deferred >= 0) ++lazy_.edges_covered; else ++lazy_.edges_exact; }
      if (e.deferred < 0) continue;                       // exact, or already materialised
      if (e.tgt < 0 || e.tgt >= static_cast<int>(n)) continue;
      const int key = online_key_of_[e.tgt];
      if (key < 0 || key >= static_cast<int>(key_comp.size())) continue;
      const int comp = key_comp[key];
      if (comp < 0 || comp >= static_cast<int>(comp_live.size())) continue;
      if (census && comp_live[comp]) ++lazy_.covered_in_scc;
      // Only a component that can host an infinite psi-run owes anything.  The finite half of
      // the verdict is complete on its own and survives every cover.
      if (!comp_live[comp]) continue;
      if (online_forbidden_cover_.count(forbid_key(v, e.ordinal))) continue;
      out->push_back(LazyObligation{v, e.ordinal, online_raw_slot_[v], static_cast<int>(pos),
                                    e.tgt, key, comp, depth[v] < 0 ? INT32_MAX : depth[v]});
    }
  }
}

void LivenessEngine::lazy_cone(const std::vector<int>& S, std::vector<int>* cone) const {
  const std::size_t n = nrec();
  std::vector<char> inS(n, 0);
  for (int v : S) if (v >= 0 && v < static_cast<int>(n)) inS[v] = 1;
  // Reverse adjacency over the psi-closure only; the sweep never leaves it either.
  std::vector<std::vector<int>> rev(n);
  for (int v : S)
    for (const OtfEdge& e : adj[v])
      if (e.tgt >= 0 && e.tgt < static_cast<int>(n) && inS[e.tgt]) rev[e.tgt].push_back(v);

  std::vector<char> seen(n, 0);
  std::vector<int> stk;
  for (int v : lazy_changed_) {
    if (v < 0 || v >= static_cast<int>(n) || !inS[v] || seen[v]) continue;
    seen[v] = 1;
    stk.push_back(v);
  }
  cone->clear();
  while (!stk.empty()) {
    const int v = stk.back();
    stk.pop_back();
    cone->push_back(v);
    for (int u : rev[v])
      if (!seen[u]) { seen[u] = 1; stk.push_back(u); }
  }
  // ::LivenessEngine::solve_greatest_fixpoint walks `S` in order inside each component, and the
  // lockstep sweep it runs there is order-sensitive for how fast it settles, not for what it
  // settles on.  Restore S's order so a cone pass behaves like the full pass it replaces.
  std::sort(cone->begin(), cone->end());
}

bool LivenessEngine::online_replay_cached(int source, int ordinal, int key,
                                          detail::SymState& out) {
  if (source != lazy_replay_src_) {
    lazy_replay_src_ = source;
    lazy_replay_succ_.clear();
    if (source < 0 || source >= static_cast<int>(nrec())) return false;
    const int skey = online_key_of_[source];
    if (skey < 0 || skey >= static_cast<int>(online_key_by_id_.size())) return false;
    detail::SymState src;
    key_state(online_key_by_id_[skey]->id, src);
    zread(source, src.zone);
    std::vector<GfpTag> stags;
    tagged_successors(net, src, L, U, qmask, emode, lazy_replay_succ_, stags,
                      avoid_forward_ ? avoid_ : nullptr);
  }
  if (ordinal < 0 || ordinal >= static_cast<int>(lazy_replay_succ_.size())) return false;
  // Confirm the discrete key, exactly as ::LivenessEngine::online_replay does: a stale ordinal
  // would otherwise materialise a successor at the wrong key.
  if (key < 0 || key >= static_cast<int>(online_key_by_id_.size())) return false;
  const OnlineKeyInfo& tk = *online_key_by_id_[key];
  const detail::SymState& cand = lazy_replay_succ_[static_cast<std::size_t>(ordinal)];
  const int* tl = key_locs(tk.id);
  if (!std::equal(cand.locs.begin(), cand.locs.end(), tl, tl + key_loc_stride_)) return false;
  const long* ti = key_ints(tk.id);
  if (!std::equal(cand.ints.begin(), cand.ints.end(), ti, ti + key_int_stride_)) return false;
  out = cand;
  return true;
}

bool LivenessEngine::lazy_materialise(const LazyObligation& ob) {
  Phase phase(&lazy_.ms_forward);
  detail::SymState x;
  ++lazy_.replays;
  // Per-source replay cache.  ::LivenessEngine::online_replay regenerates the source's WHOLE
  // successor batch and keeps one element; obligations arrive sorted by source, so the rest of
  // each group reuses it.  Valid because a record's zone and discrete state are immutable once
  // appended -- refinement only ever rewrites edge TARGETS -- but it must still be dropped
  // whenever the engine is reset or a new wave starts, since record ids are reused across the
  // sub-system lane and the full run.  Measured: refine is 5.0s of arbiter6 q3's 7.2s.
  if (!online_replay_cached(ob.src, ob.ordinal, ob.key, x)) return false;
  if (x.zone.empty()) return false;

  // EXTRAPOLATE THE MATERIALISED ZONE, and only it.
  //
  // Refinement is the one thing on this route with no finiteness argument: `emode` is ExNone, so
  // a_LU/D subsumption is what makes the ordinary forward search terminate, and suspending covers
  // is exactly what the loop does. Extra_LU+ supplies a bound that does not depend on covering --
  // finitely many extrapolated zones per key -- and the key set is FIXED, because a covered
  // successor's enabled transitions are a subset of its coverer's, so refinement introduces no
  // key the graph does not already have. With the inclusion dedup below, that leaves at most one
  // materialised record per (key, extrapolation class).
  //
  // Only this zone. Extrapolating the whole search perturbs it for nothing: the ordinary nodes
  // already terminate under a_LU, and the global floor measurably moved five early-exit SAT rows
  // (arbiter q5 12 -> 15, csma10 q5 14 -> 16, phil8 q4 13 -> 14) by shifting when the first
  // witness appears. Confined here, every state count is bit-identical to the default route.
  //
  // Free for completeness, which is the part that looks wrong at first glance. The obligation is
  // discharged by `X subseteq Z_j`, so storing a SUPERSET of `X` discharges it just as well; the
  // backward operator then sees every run through `X` and owes nothing. And soundness is
  // unaffected because the verdict is read at node 0 cut to its pre-extrapolation zone, and node
  // 0 is never a materialised record -- every later step of a certified chain is a genuine
  // ::exact_predecessor step, so intermediate nodes may hold points no run reaches.
  Dbm exact = x.zone;                       // kept for the readout; see ::lazy_exact_
  // The forward route needs the materialised record to be EXACTLY the discarded successor: its
  // whole point is to turn a subsumption edge into an ordinary one, and an extrapolated zone is
  // a strictly larger set, so the edge would still land somewhere the successor is not.  The
  // backward route has no such need and takes the extrapolation as its termination bound.
  if (x.zone.empty()) return false;

  // Reuse by INCLUSION, append otherwise.
  //
  // ::LivenessEngine::online_intern must not be used: its scan accepts an a_LU/D cover, and a
  // successor covered by simulation is still a promise -- the edge would be recorded as
  // discharged while remaining exactly the thing this loop exists to remove.
  //
  // But refusing every reuse is the opposite error, and a worse one.  `X subseteq Z_j` is
  // precisely the condition that makes the edge exact: a run through `X` is a run through `Z_j`,
  // so `Pre_e(W_j)` sees it and nothing is owed.  Appending unconditionally instead duplicates
  // the same zone at the same key without bound, and each duplicate is expanded and mints fresh
  // covered edges -- measured on arbiter6 as 256 -> 103,056 records, geometric per wave, with no
  // fixed point at all.  Deduplicating by inclusion is what makes the loop's fixed point the
  // inclusion-covered graph, i.e. bounded by the containment graph.
  const std::size_t before = nrec();
  online_forbidden_cover_.insert(forbid_key(ob.src, ob.ordinal));
  int id = -1;
  {
    // Reuse by INCLUSION for the backward route: `X subseteq Z_j` is all the exact predecessor
    // operator needs, and reusing keeps the graph small.
    //
    // The forward route needs EQUALITY.  Its whole job is to turn a subsumption edge into an
    // ordinary one, and ordinary means `Z_tgt == post_e(Z_src)`; a record that merely contains
    // `X` leaves the edge subsuming, the obligation is marked discharged, and the loop then
    // stalls with unsafe covers it has already forbidden itself from touching.
    // Reuse by INCLUSION for the backward route: `X subseteq Z_j` is all the exact predecessor
    // operator needs.
    //
    // The forward route reuses by a_LU EQUIVALENCE, because that is the node identity of the
    // zone graph it is deciding on -- Figure 1 of Herbreteau et al. adds a fresh node only when
    // no existing one satisfies `a_LU(Z') = a_LU(Z1)`.  Exact equality instead makes every
    // materialised successor a new node, the search leaves ZG^{a_LU} for the exact zone graph,
    // and cycles that ought to close never do.
    const OnlineKeyInfo& ki = *online_key_by_id_[ob.key];
    online_configure_bounds(ob.key);            // the covering test reads the per-key bL/bU
    Dbm jz;
    for (int j = ki.node_head; j >= 0; j = online_next_in_key_[j]) {
      const bool hit = alu_le_stored(x.zone, j) && alu_le_new(j, x.zone);
      if (hit) { id = j; break; }
    }
  }
  if (id < 0) id = online_append(std::move(x), ob.key);
  if (id < 0 || online_failed_) return false;

  grow();
  if (online_pinned_.size() < nrec()) online_pinned_.resize(nrec(), 0);
  online_pinned_[id] = 1;
  // Only when the record IS the extrapolation of this successor.  A reused record already holds
  // a zone that includes `X`, and its own exactness is whatever it already was.
  if (nrec() > before) lazy_exact_.emplace(id, std::move(exact));
  if (!online_active_[id]) online_activate(id);

  // Rewrite the promise into a real transition.  The edge now names a record whose stored zone
  // IS the exact successor, so the backward operator's `Pre_e` against it is exact.
  if (ob.slot >= 0 && ob.slot < static_cast<int>(online_raw_adj_.size()) &&
      ob.pos >= 0 && ob.pos < static_cast<int>(online_raw_adj_[ob.slot].size())) {
    OtfEdge& e = online_raw_adj_[ob.slot][ob.pos];
    e.tgt = id;
    e.deferred = kEdgeMaterialized;
  }

  if (!expanded[id] && !online_inq_[id]) { online_inq_[id] = 1; online_work_.push_back(id); }
  online_drain_forward_work();
  if (online_failed_ || cancelled()) return false;
  // The refined edge's source gained a predecessor case, and every record the drain created
  // starts at its own zone rather than at empty.  Both must be re-solved; see ::lazy_cone.
  lazy_changed_.push_back(ob.src);
  lazy_changed_.push_back(id);
  for (std::size_t v = before; v < nrec(); ++v) lazy_changed_.push_back(static_cast<int>(v));
  lazy_.new_nodes += nrec() - before;
  ++lazy_.materialised;
  return true;
}

bool LivenessEngine::lazy_refine_eg(bool* sat) {
  lazy_.ran = true;
  lazy_.nodes_before = nrec();
  const std::size_t budget = budget_multiple() * (nrec() + 1);

  std::vector<int> key_comp;
  std::vector<char> comp_live;
  lazy_build_key_sccs(&key_comp, &comp_live);

  // NO LIVE COMPONENT: the verdict is `NOT`, and no backward pass is needed to say so.
  //
  // `E[] psi` has exactly two witness shapes. An INFINITE run must eventually stay inside one
  // key-SCC and visit its keys infinitely often -- if every component is dead, no such run
  // exists anywhere reachable. A FINITE maximal run, or one that diverges in place, is a `Stop`
  // witness, and the forward search already tests for it at every psi record:
  // ::LivenessEngine::online_eg_terminal_witness mirrors ::LivenessEngine::stop_set's restricted
  // branch case for case -- `time_diverges`, then `orig_time_open`, then
  // `state_has_deadlock(*dead_net, ...)` -- and halts the construction on the first hit. Covering
  // cannot hide one: D preserves deadlock, and a_LU is a simulation so it preserves divergence,
  // which puts a representative of any discarded witness in the coverer.
  //
  // So: every component dead AND the forward search ran to completion without a terminal hit
  // means neither witness shape exists. The greatest fixpoint would confirm that at length -- it
  // is 96% of the runtime on fischer8 q3 and 89% on twolock10 q3 -- having been told the answer
  // by the criterion in 2.5ms and 34.5ms respectively.
  //
  // Guarded on `eg_early_` because that is what arms the terminal test; without it the `Stop`
  // half was never checked and this shortcut would skip a real witness.
  if (eg_early_ && !eg_early_hit_) {
    bool any_live = false;
    for (char c : comp_live)
      if (c) { any_live = true; break; }
    if (!any_live) {
      lazy_.skipped_gfp = true;
      *sat = false;
      return true;
    }
  }

  std::vector<LazyObligation> obligations;
  std::vector<std::vector<OtfEdge>> saved_adj;
  std::vector<int> saved_key;

  for (;;) {
    if (cancelled() || online_failed_) return false;

    if (!lazy_snapshot(&saved_adj, &saved_key)) return false;
    bool winning = false;
    bool usable = false;
    {
      const bool saved_capped = capped;      // sticky, and a probe must not condemn the real run
      const std::vector<int> S = eg_closure();
      if (!S.empty()) {
        std::vector<Federation> W;
        const std::size_t visits_before = fixpoint_iterations;
        // After a refinement wave only the cone of the change can move; see ::lazy_cone.  The
        // first pass has no previous fixpoint to lean on and must sweep the whole closure.
        std::vector<int> cone;
        const bool first = lazy_.rounds == 0;
        if (!first) {
          Phase phase(&lazy_.ms_scc);
          lazy_cone(S, &cone);
          lazy_.cone_nodes += cone.size();
          lazy_.cone_of += S.size();
        }
        const std::vector<int>& sweep = first ? S : cone;
        {
          Phase phase(first ? &lazy_.ms_base_gfp : &lazy_.ms_backward);
          if (!sweep.empty()) solve_greatest_fixpoint(sweep, W);
          else W.assign(nrec(), Federation{});     // nothing could have moved
        }
        lazy_changed_.clear();
        lazy_.backward_visits += fixpoint_iterations - visits_before;
        if (!capped) {
          usable = true;
          // CLOSURE READOUT, and it is what makes "only cycles are owed" true.
          //
          // `Stop_v` is exact on `Z_v` and enters the operator as a union term, and the D
          // refinement preserves deadlock under covering, so a finite or divergent witness is
          // safe AT ITS OWN NODE. Reading `W[0]` alone does not make it safe, because the chain
          // of `Pre` steps from node 0 down to that node still has to survive -- and a disjoint
          // cover anywhere on that PREFIX breaks it. Under the node-0 readout the prefix is
          // therefore a second, independent source of incompleteness, which no
          // recurrence-based obligation filter can see: the component carrying the witness may
          // be provably dead and still owe the prefix.
          //
          // Every `v` in ::LivenessEngine::eg_closure is joined to the initial state by a
          // psi-path in the stored graph, and on this route every point of a stored zone is
          // genuinely reached by such a path: `emode` is ExNone so no zone is extrapolated,
          // ::LivenessEngine::online_intern merges only by exact `try_union`, and the forward
          // search starts at node 0 and expands psi-nodes only. So a non-empty `W[v]` composes
          // into a real witness -- psi-path 0 ->* v, then the maximal psi-run `W[v]` certifies --
          // without the prefix having to survive the fixpoint at all.
          //
          // The precondition is the missing extrapolation. With the HORA_LAZY_XP floor on, a
          // stored zone holds valuations no run reaches and this argument fails, so that
          // configuration keeps the node-0 readout and keeps the prefix exposure with it.
          if (emode == ExNone) {
            winning = false;
            for (int v : S) {
              const Federation& wv = W[v];
              if (fed_is_empty(wv)) continue;
              // Node 0 must be a real initial valuation, and a materialised record holds the
              // extrapolation of its successor rather than the successor: cut both back.
              const Dbm* cut = nullptr;
              if (v == 0 && lazy_root_exact_.dim() > 0) cut = &lazy_root_exact_;
              if (const auto it = lazy_exact_.find(v); it != lazy_exact_.end()) cut = &it->second;
              if (cut && fed_is_empty(fed_intersect(wv, Federation{*cut}))) continue;
              winning = true;
              break;
            }
          } else {
            winning = lazy_root_exact_.dim() > 0
                          ? !fed_is_empty(fed_intersect(W[0], Federation{lazy_root_exact_}))
                          : !fed_is_empty(W[0]);
          }
          if (!winning) {
            Phase phase(&lazy_.ms_scc);
            lazy_collect(S, W, key_comp, comp_live, &obligations);
          }
        }
      } else {
        usable = true;                       // no psi-closure at all: the seed cannot witness
      }
      capped = saved_capped;
    }
    lazy_restore(&saved_adj, &saved_key);

    if (!usable) return false;               // capped: nothing was decided either way
    if (winning) { *sat = true; return true; }
    if (obligations.empty()) { *sat = false; return true; }
    if (lazy_.rounds == 0) lazy_.obligations = obligations.size();
    if (lazy_.materialised >= budget) { lazy_.exhausted = true; return false; }

    // ONE WAVE PER FIXPOINT, not one refinement per fixpoint.
    //
    // The obligations in hand were all produced by the SAME backward failure, so they are all
    // demand-driven in exactly the sense the design requires -- discharging them is not "refine
    // every cover in an SCC", it is "discharge what this failure asked for". What re-running the
    // fixpoint between two of them buys is the chance to notice a witness one refinement earlier,
    // and it costs a full pass: measured on viking10, 23.5ms per round against a 17.2ms base
    // pass, i.e. the sweep is repeated essentially in full 1,152 times.
    //
    // So the wave is drained first and the fixpoint re-run once at the end of it. The obligation
    // SET is unchanged; only the point at which the verdict is re-tested moves. A witness created
    // mid-wave is still seen, one wave later rather than one refinement later.
    std::sort(obligations.begin(), obligations.end(),
              [](const LazyObligation& a, const LazyObligation& b) {
                if (a.scc != b.scc) return a.scc < b.scc;
                if (a.depth != b.depth) return a.depth < b.depth;
                if (a.src != b.src) return a.src < b.src;
                return a.ordinal < b.ordinal;
              });
    static const bool dbg = std::getenv("HORA_LAZY_DBG") != nullptr;
    if (dbg)
      std::fprintf(stderr, "LAZY wave=%zu nodes=%zu obligations=%zu materialised=%zu\n",
                   lazy_.waves + 1, nrec(), obligations.size(), lazy_.materialised);
    ++lazy_.waves;
    for (const LazyObligation& ob : obligations) {
      if (lazy_.materialised >= budget) { lazy_.exhausted = true; return false; }
      ++lazy_.rounds;
      if (!lazy_materialise(ob)) return false;
      if (eg_early_hit_) { *sat = true; return true; }  // the drain's own probe found a witness
      if (cancelled() || online_failed_) return false;
    }
  }
}

void LivenessEngine::lazy_report(bool verdict) {
  static const bool on = std::getenv("HORA_REFINE_STATS") != nullptr;
  if (!on || !lazy_.ran) return;
  // The counts were taken by ::LivenessEngine::lazy_collect while the marks were still live;
  // recounting here would read the sealed graph, where every one has been reset to exact.
  std::fprintf(stderr,
               "REFINE verdict=%s exhausted=%d skipped_gfp=%d nodes_before=%zu nodes_after=%zu "
               "edges_exact=%zu edges_covered=%zu covered_in_scc=%zu obligations=%zu "
               "materialised=%zu "
               "new_nodes=%zu replays=%zu rounds=%zu waves=%zu backward_visits=%zu scc_builds=%zu "
               "cone=%zu/%zu "
               "ms_base_gfp=%.1f ms_backward=%.1f ms_forward=%.1f ms_scc=%.1f\n",
               verdict ? "SAT" : "NOT", static_cast<int>(lazy_.exhausted),
               static_cast<int>(lazy_.skipped_gfp), lazy_.nodes_before,
               nrec(), lazy_.edges_exact, lazy_.edges_covered, lazy_.covered_in_scc,
               lazy_.obligations, lazy_.materialised, lazy_.new_nodes, lazy_.replays, lazy_.rounds, lazy_.waves,
               lazy_.backward_visits, lazy_.scc_builds, lazy_.cone_nodes, lazy_.cone_of,
               lazy_.ms_base_gfp, lazy_.ms_backward,
               lazy_.ms_forward, lazy_.ms_scc);
}

}  // namespace hora
