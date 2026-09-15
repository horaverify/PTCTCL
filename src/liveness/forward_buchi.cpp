/// @file src/liveness/forward_buchi.cpp
/// @brief Decide `E[] psi` and `p --> q` by cycle detection on the forward graph, with no
///        backward pass at all.  The production route; `-DHORA_LIVENESS_GFP=ON` selects the
///        backward greatest fixpoint instead.
///
/// The backward greatest fixpoint is 89-96% of the runtime on the expensive refuted rows
/// (fischer8 q3, twolock10 q3).  It is there because a cycle in a SUBSUMED zone graph need not
/// correspond to a run -- a cover edge can close a loop no concrete execution realises.  This
/// file excludes the cover edges from the cycle search instead, and repairs the ones that
/// matter.
///
/// The justification is Herbreteau, Srivathsan, Tran and Walukiewicz, "Why Liveness for Timed
/// Automata Is Hard, and What We Can Do About It", FSTTCS 2016.  Their Definition 5 calls a
/// subsumption graph *liveness compatible* when
///
///     C5: no cycle contains both an accepting node and a subsumption edge,
///
/// and Theorem 6 states that the abstract zone graph has an infinite accepting path IFF a
/// liveness compatible subsumption graph has one made of ordinary edges.  Both directions, so the
/// reduction is exact rather than an under-approximation.
///
/// Every node of the accepting set is accepting here -- the run has only to stay in it -- so C5
/// collapses to "no cover edge lies on a cycle", and the whole verdict is two Tarjan runs:
///
///   * a cycle among EXACT edges is an accepting run          -> witness
///   * a cover edge inside a cycle violates C5                -> that cover must be refined
///   * neither                                                -> no infinite witness
///
/// No `Pre`, no federations, no zone algebra whatsoever in the decision.  The two queries differ
/// only in what the accepting set is, and in the sign of the answer:
///
///   * `E[] psi`  accepts the psi-closure of the initial vertex; a witness SATISFIES it.
///   * `p --> q`  accepts the `!q` closure of the `p /\ !q` nodes; a witness REFUTES it.
///
/// @note The premise the cycle argument needs -- a node being one abstracted post image -- is met
/// here.  The classical argument behind Theorem 6 needs every node to be exactly
/// `a(post_t(.))` of a single predecessor: a cycle then gives `a(post^n(Z)) = Z`, and since
/// `a(empty) = empty`, `Z` non-empty forces `post^n(Z)` non-empty, which is what yields the
/// concrete run.  ::LivenessEngine::online_intern can break that premise -- it merges by exact
/// convex union (`Dbm::try_union`), so a stored zone can be a union of several post images rather
/// than one.  A merged node is therefore never the target of an ordinary edge (see `refresh_ord`
/// below), and ::LivenessEngine::fwd_refining_ stops the merging outright once refinement begins.
///
/// @note Node identity throughout is `a_LU`-EQUIVALENCE, not zone equality; see the comment in
/// `refresh_ord` below, which is where getting it wrong costs both termination and the verdict.

#include "hora/zone.hpp"
#include "hora/detail/liveness_engine.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include "hora/detail/semantics.hpp"
#include "src/liveness/internal.hpp"

namespace hora {

namespace {

/// @brief Nontrivial SCCs of a subgraph, by iterative Tarjan.
///
/// @param n      node count
/// @param inS    membership of the node set being searched
/// @param succ   yields the successors of a node, already filtered by the caller
/// @param[out] comp   component id per node, -1 outside @p inS
/// @param[out] cyclic per component, whether it is nontrivial or a self-loop
/// @return number of components
template <class Succ>
int scc_of(std::size_t n, const std::vector<char>& inS, Succ&& succ, std::vector<int>* comp,
           std::vector<char>* cyclic) {
  comp->assign(n, -1);
  cyclic->clear();
  std::vector<int> idx(n, -1), low(n, 0), stk;
  std::vector<char> on(n, 0);
  std::vector<std::pair<int, std::size_t>> call;
  std::vector<int> scratch;
  int next = 0, ncomp = 0;
  for (std::size_t root = 0; root < n; ++root) {
    if (!inS[root] || idx[root] != -1) continue;
    call.push_back({static_cast<int>(root), 0});
    idx[root] = low[root] = next++;
    stk.push_back(static_cast<int>(root));
    on[root] = 1;
    while (!call.empty()) {
      auto& [v, ei] = call.back();
      scratch.clear();
      succ(v, &scratch);
      if (ei < scratch.size()) {
        const int w = scratch[ei++];
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
          bool loop = false;
          if (size == 1) {
            scratch.clear();
            succ(v, &scratch);
            for (int w : scratch)
              if (w == v) { loop = true; break; }
          }
          cyclic->push_back(size > 1 || loop);
          ++ncomp;
        }
        const int child = v;
        call.pop_back();
        if (!call.empty()) low[call.back().first] = std::min(low[call.back().first], low[child]);
      }
    }
  }
  return ncomp;
}

}  // namespace

bool LivenessEngine::forward_buchi(bool leadsto, bool* witness) {
  const auto t0 = std::chrono::steady_clock::now();
  static const bool dbg = std::getenv("HORA_FWD_DBG") != nullptr;
  std::size_t rounds = 0, refined = 0, nodes_before = nrec();
  double ms_ord = 0, ms_scc = 0, ms_refine = 0;

  // ORDINARY-EDGE CACHE.
  //
  // `ord[v]` holds the targets of v's ordinary edges: those whose target's stored zone IS the
  // exact successor, equality and not inclusion.  It is recomputed per node with one
  // ::tagged_successors, and cached, because only two kinds of node can change between rounds --
  // a node refinement created, and the source of an edge refinement rewrote.  Everything else
  // keeps last round's answer, which turns the per-round cost from "one forward sweep" into
  // "one forward sweep over what actually moved".
  std::vector<std::vector<int>> ord;
  std::vector<char> dirty;
  auto refresh_ord = [&](const std::vector<int>& S, const std::vector<char>& inS) {
    const auto t = std::chrono::steady_clock::now();
    ord.resize(nrec());
    dirty.resize(nrec(), 1);
    detail::SymState ssc;
    std::vector<detail::SymState> succ;
    std::vector<GfpTag> stags;
    Dbm tz;
    // NO ROUND-0 SHORTCUT.  Deriving `ord` from the `deferred` marks looks free -- they are
    // assigned when the edge is created -- but `kEdgeExact` means "created its target", i.e.
    // EXACT ZONE EQUALITY, and the ordinary-edge relation is a_LU EQUIVALENCE.  Using the marks
    // reintroduces exactly the wrong node identity that this whole route depends on getting
    // right, on the one round that decides most queries.
    for (int v : S) {
      if (!dirty[v]) continue;
      dirty[v] = 0;
      ord[v].clear();
      if (adj[v].empty()) continue;
      succ.clear();
      stags.clear();
      tagged_successors(net, sread(v, ssc), L, U, qmask, emode, succ, stags,
                        avoid_forward_ ? avoid_ : nullptr);
      for (const OtfEdge& e : adj[v]) {
        if (e.tgt < 0 || e.tgt >= static_cast<int>(inS.size()) || !inS[e.tgt]) continue;
        if (e.ordinal < 0 || e.ordinal >= static_cast<int>(succ.size())) continue;
        if (succ[e.ordinal].zone.empty()) continue;
        // ::LivenessEngine::alu_le_stored reads the per-key `bL`/`bU`, so the target's key has to
        // be installed first.  Without it the test runs against whatever bounds were left over
        // and the search silently produces no verdict at all.
        online_configure_bounds(online_key_of_[e.tgt]);
        // a_LU EQUIVALENCE, not zone equality.  The paper's zone graph (Figure 1) identifies
        // nodes by `a_LU(Z') = a_LU(Z1)`; that is the relation whose finiteness lets a fresh
        // exploration terminate and a real cycle close.  Comparing exact zones puts the search on
        // the exact zone graph instead, where fresh nodes never repeat -- single4 q1 answered NOT
        // for a satisfied query and arbiter4 q5 never finished.
        // Node identity is a_LU EQUIVALENCE -- mutual a_LU covering -- matching Figure 1 of
        // Herbreteau et al., which adds a node only when no existing one satisfies
        // `a_LU(Z') = a_LU(Z1)`.
        //
        // In principle this should be the D-refined relation, since ::LivenessEngine::online_covers
        // applies the D atoms and the graph being built is therefore ZG^{a_D}.  It is not, because
        // `online_covers` tests one direction only and pairing it with the a_LU test for the other
        // mixes a fine relation with a coarse one: too many edges come out ordinary and the cycle
        // search invents witnesses (measured: false SAT on fischer, viking, arbiter, twolock).
        // A symmetric D test would be needed.  The two relations coincide whenever D has no atom,
        // which is every row of this corpus, so nothing is lost here -- but this is the place to
        // fix for a model where D fires.
        detail::SymState xs;
        xs.zone = succ[e.ordinal].zone;
        // A MERGED TARGET IS NEVER ORDINARY, and matching the absorbed PIECES is not a way
        // around that.  It is tempting: ::LivenessEngine::online_intern merges by exact convex
        // union, so `post_e(Z_src)` can equal one piece of `Z_tgt` and the edge looks like the
        // one the unmerged graph would have had.  But the merged node's OUTGOING edges are
        // computed from the whole union, so a cycle through it composes "enter via piece P" with
        // "leave via P u P' u ...", which is not one concrete run -- the premise Theorem 6 needs,
        // a node being ONE abstracted post image, is exactly what merging gave up.
        //
        // Measured on phil4 `P0.Eat --> P0.Dropped`: P2 laps Think -> HoldL -> Eat -> Dropped
        // while P0 sits in Eat under `c0 <= 2`, and each lap must burn a time unit to pass
        // `c2 >= 1`, so at most two fit and the query is SATISFIED.  With the piece match that
        // four-node cycle read as ordinary and the route answered NOT.  grc4 `T0.in --> T0.far`
        // is the same shape.
        //
        // Left as a cover the edge becomes an ordinary C5 obligation, and refinement splits the
        // piece back out into a record of its own -- the honest repair, and a cheap one: phil4
        // settles in 3 refinements and 8 extra records.
        const bool ordinary = online_equiv(xs, e.tgt, online_key_of_[e.tgt]) &&
                              !(e.tgt < static_cast<int>(online_merged_.size()) &&
                                online_merged_[e.tgt]);
        if (ordinary) ord[v].push_back(e.tgt);
      }
    }
    ms_ord += std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - t).count() / 1000.0;
  };

  // Components already shown to carry no cycle and to owe no unsafe cover stay that way: the
  // only thing refinement does is add records and turn subsumption edges into ordinary ones, and
  // a component whose nodes are all settled gains neither.  Skipping them is what keeps the later
  // rounds cheap once the graph is mostly decided.
  std::vector<char> settled;

  for (;; ++rounds) {
    if (cancelled() || online_failed_) return false;
    // A CONCRETE WITNESS, already proven.  ::LivenessEngine::online_drain_forward_work runs the
    // early-witness probes -- an exact backward solve over the partial closure for `E[] psi`, a
    // `p /\ !q` valuation that ends a maximal run for leads-to -- and stops the moment one
    // succeeds; the refinement wave below drains, so the hit can land in the middle of one.  That
    // flag is the verdict and must be read before anything else, because the same stop leaves
    // every record the wave had queued UNEXPANDED, and `lazy_snapshot` then projects a truncated
    // graph in which the cycles this search is looking for have been cut away.  Measured on
    // single5 `E[] x0 <= 6`: the probe hit at 381 records, the drains stopped, and the next two
    // rounds watched the full-graph cycle count fall 78 -> 15 -> 0 and answered NOT for a
    // satisfied query.
    if (eg_early_hit_) { *witness = true; return true; }

    // `E[] psi` WITH EVERY COMPONENT DEAD, DECIDED ON THE RAW GRAPH.
    //
    // That round needs three things -- the psi-closure, the recurrence criterion over it, and the
    // stop scan -- and none of them reads the resolved adjacency the snapshot below builds: the
    // criterion walks the raw edges itself, the closure resolves covers as it goes exactly as
    // ::eg_closure does over the snapshot, and on this construction the stop scan was already paid
    // at expansion (::stop_known_empty).  The snapshot is a full copy of every edge -- as large as
    // the raw adjacency itself, 16 MB of 174 on arbiter14 q2 -- bought only to be thrown away on
    // the round that decides every refuted row.  So the criterion runs first, here, and the
    // snapshot is taken only when some component is live and the cycle search needs it.
    std::vector<int> crit_comp;
    std::vector<char> crit_live;
    bool have_crit = false;
    if (!leadsto) {
      const std::vector<int> S0 = online_eg_closure_raw();
      bool all_known = !S0.empty();
      for (int v : S0)
        if (!stop_known_empty(v)) { all_known = false; break; }
      if (all_known) {
        std::vector<char> in0(nrec(), 0);
        for (int v : S0) in0[v] = 1;
        lazy_build_key_sccs(&crit_comp, &crit_live, &in0);
        have_crit = true;
        bool live = false;
        for (int v : S0) {
          const int k = online_key_of_[v];
          const int c = (k >= 0 && k < static_cast<int>(crit_comp.size())) ? crit_comp[k] : -1;
          if (c < 0 || c >= static_cast<int>(crit_live.size()) || crit_live[c]) { live = true; break; }
        }
        if (!live) {
          if (dbg)
            std::fprintf(stderr, "FWD-DIAG round=%zu nrec=%zu |S|=%zu all-components-dead raw-graph"
                                 " ms=%.1f\n", rounds, nrec(), S0.size(),
                         std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count() / 1000.0);
          *witness = false;
          return true;
        }
      }
    }

    std::vector<std::vector<OtfEdge>> saved_adj;
    std::vector<int> saved_key;
    std::size_t lost_edges = 0;
    if (!lazy_snapshot(&saved_adj, &saved_key, &lost_edges)) return false;

    bool decided = false, verdict = false, truncated = false;
    std::size_t exact_cycles = 0, unsafe = 0, covers = 0, edges = 0, r_block = 0;
    std::vector<LazyObligation> obligations;
    {
      // The accepting set.  For `E[] psi` it is the psi-closure of the initial vertex, and an
      // empty one means node 0 already fails psi.  For `p --> q` it is the `!q` closure of the
      // `p /\ !q` nodes, and an empty one means the query has no antecedent to violate: both
      // read as "no witness", which is SAT for the first query and REFUTED-NOT for the second.
      const std::vector<int> S = leadsto ? fwd_accepting_closure() : eg_closure();
      if (S.empty()) {
        decided = true;
        verdict = false;
      } else {
        const std::size_t n = nrec();
        std::vector<char> inS(n, 0);
        for (int v : S)
          if (v >= 0 && v < static_cast<int>(n)) inS[v] = 1;
        settled.resize(n, 0);

        // THE CRITERION RUNS FIRST, because it can retire the whole round.
        //
        // It is a sufficient condition for "no infinite run can stay in this key component"; see
        // ::LivenessEngine::lazy_build_key_sccs for the two conjuncts.  Its usual job below is to
        // exonerate covers, but when it kills EVERY component the accepting set touches it has
        // said something much stronger: an infinite run's key sequence is an infinite path in a
        // finite key graph, so it must end up inside one key-SCC, and every candidate SCC is
        // dead.  No infinite witness exists anywhere in `S`, and the ordinary-edge sweep and both
        // Tarjan passes below are then searching for a cycle that cannot be there.
        //
        // This is the arbiter ladder.  arbiter12 `E[] (C0.idle || C0.want) && x0 <= 40` reports
        // `key-sccs=2050 dead=2050` and then spent 158 ms re-deriving every node's successors for
        // `ord` and 66 ms on the two searches; arbiter14 spent 982 ms and 518 ms.  Every one of
        // those covers was exonerated anyway -- `dead_tgt=180224` on arbiter12 -- so the round's
        // answer is unchanged and only the work goes away.
        //
        // The one thing this gives up is a cross-check: today an ordinary cycle found in a
        // component the criterion calls dead would short-circuit to a witness before the
        // criterion is ever consulted.  That cannot happen if the criterion is sound -- conjunct
        // (b) makes every cycle burn time and (a) forbids a time-divergent run -- so the two can
        // only ever disagree when the criterion is wrong, and the shortcut makes that a verdict
        // instead of a contradiction.
        std::vector<int> key_comp;
        std::vector<char> comp_live;
        // Already computed above over the same closure when the raw-graph shortcut ran and found a
        // live component; the snapshot's ::eg_closure is the same node set.
        if (have_crit) {
          key_comp.swap(crit_comp);
          comp_live.swap(crit_live);
          have_crit = false;
        } else {
          lazy_build_key_sccs(&key_comp, &comp_live, &inS);
        }
        auto live_at = [&](int node) {
          if (node < 0 || node >= static_cast<int>(nrec())) return true;
          const int k = online_key_of_[node];
          if (k < 0 || k >= static_cast<int>(key_comp.size())) return true;
          const int c = key_comp[k];
          if (c < 0 || c >= static_cast<int>(comp_live.size())) return true;
          return comp_live[c] != 0;
        };
        const double ms_crit = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - t0).count() / 1000.0;
        bool any_live = false;
        for (int v : S)
          if (live_at(v)) { any_live = true; break; }

        // R: THE PART OF THE CLOSURE A WITNESS MAY BE READ AT.
        //
        // `S` is closed under ALL edges, cover edges included, and that is the wrong relation to
        // compose a witness back along.  A cover edge `u -> v` says `post_e(u) subseteq
        // a_LU(Z_v)`: `Z_v` is the LARGER zone, so a run starting in `Z_v` need not start
        // anywhere `u` can reach, and a cycle behind such an edge is not a run from `u` at all.
        // An ORDINARY edge is `a_LU`-EQUIVALENCE, which is symmetric, so a run from `Z_tgt` is
        // matched by a run from `post_e(Z_src)` and the witness does compose backwards.
        //
        // `E[] psi` never needed this.  ::LivenessEngine::online_eg_prune_ expands only
        // psi-nodes, so every node of the psi-closure was CREATED as the exact post image of an
        // expanded psi-node -- an ordinary edge, by construction -- and the creator chain back to
        // node 0 is already all-ordinary.  R is S there and computing it would be dead work.
        //
        // `p --> q` has no such prune: the path to the `p`-node is ordinary reachability that may
        // leave `!q`, so a node of the `!q` closure can be reachable from an accepting node only
        // THROUGH a cover, and R is then strictly smaller than S.
        //
        // No benchmark row is known to distinguish the two -- the phil4 and grc4 false witnesses
        // that first exposed a gap here were the merged-node bug above, not this -- so R is here
        // because it is the relation the composition argument actually licenses, not because a
        // measurement forced it.
        std::vector<char> inR;
        const auto build_R = [&] {                 // needs `ord`, so only after refresh_ord
          if (!leadsto) { inR = inS; return; }
          inR.assign(n, 0);
          std::vector<int> rstk;
          for (int v : S)
            if (is_nq(v) && is_p(v)) { inR[v] = 1; rstk.push_back(v); }
          while (!rstk.empty()) {
            const int v = rstk.back();
            rstk.pop_back();
            for (int w : ord[v])
              if (w >= 0 && w < static_cast<int>(n) && !inR[w]) { inR[w] = 1; rstk.push_back(w); }
          }
        };

        // R-EXTENSION OBLIGATIONS: what makes the R restriction complete and not merely sound.
        //
        // Both readouts -- the ordinary cycle and the `stop_set` scan -- are confined to R,
        // because a cover edge does not compose a witness backwards.  That is sound, but on its
        // own it is not complete: a `stop` valuation, or an ordinary cycle, sitting at a node of
        // `S \ R` is a genuine witness that neither readout would ever look at, and C5 owes
        // nothing for it either -- the cover that keeps it out of R need not lie on any cycle,
        // and an acyclic cover is never refined.  The route would then answer "no witness" for a
        // query that has one.
        //
        // So a cover `u -> w` with `u` in R and `w` in `S \ R` is an obligation in its own
        // right.  Discharging it makes the edge ordinary, which puts the materialised `w` into R;
        // at the fixed point no cover leaves R inside S, hence R = S and the readouts see the
        // whole accepting set.  Termination is the C5 argument unchanged: a discharged cover is
        // forbidden and never recreated, and the graph is finite.
        //
        // `E[] psi` has R = S by construction, so `rlist` is always empty and none of this runs.
        //
        // The usefulness filter is what stops this from refining a whole leads-to closure: a node
        // beyond the cover can only matter if it REACHES a readout, so a cover that blocks
        // nothing is left alone.  Reachability is taken over every edge of `G[S]`, cover edges
        // included, which over-approximates what a run can do and therefore only ever refines
        // more than strictly needed.
        struct RCover { int src, ordinal, tgt; };
        std::vector<RCover> rlist;
        // An edge from R to `S \ R` is a cover by definition: an ordinary one would have put its
        // target in R.  So no `ord` lookup is needed to classify it.
        const auto collect_r_leaving = [&] {
          rlist.clear();
          if (!leadsto) return;                    // R = S
          for (int v : S) {
            if (!inR[v]) continue;
            for (const OtfEdge& e : adj[v])
              if (inS[e.tgt] && !inR[e.tgt]) rlist.push_back(RCover{v, e.ordinal, e.tgt});
          }
        };
        const auto push_obligation = [&](int v, int ordinal, int tgt) {
          if (ordinal < 0 || online_forbidden_cover_.count(forbid_key(v, ordinal))) return;
          const int slot = online_raw_slot_[v];
          int pos = -1;
          if (slot >= 0)
            for (std::size_t r = 0; r < online_raw_adj_[slot].size(); ++r)
              if (online_raw_adj_[slot][r].ordinal == ordinal) { pos = static_cast<int>(r); break; }
          if (pos >= 0)
            obligations.push_back(LazyObligation{v, ordinal, slot, pos, tgt,
                                                 online_key_of_[tgt], 0, 0});
        };
        // @p readout says whether a node could itself carry a witness readout, now or after any
        // amount of refinement.  Seeds the backward sweep over `G[S]`.
        const auto owe_r_extensions = [&](auto&& readout) {
          collect_r_leaving();
          if (rlist.empty()) return;
          std::vector<std::vector<int>> rev(n);
          for (int v : S)
            for (const OtfEdge& e : adj[v])
              if (inS[e.tgt]) rev[e.tgt].push_back(v);
          std::vector<char> useful(n, 0);
          std::vector<int> stk;
          for (int v : S)
            if (readout(v)) { useful[v] = 1; stk.push_back(v); }
          while (!stk.empty()) {
            const int v = stk.back();
            stk.pop_back();
            for (int u : rev[v])
              if (!useful[u]) { useful[u] = 1; stk.push_back(u); }
          }
          for (const RCover& rc : rlist) {
            if (!useful[rc.tgt]) continue;
            ++r_block;
            push_obligation(rc.src, rc.ordinal, rc.tgt);
          }
        };

        if (!any_live) {
          // No infinite witness is possible, so only the MAXIMAL FINITE half is left -- the same
          // `stop_set` pass the `unsafe == 0` branch runs below, and the same reasoning for why a
          // non-empty one at any `v` of R is a witness.
          //
          // `E[] psi` needs nothing computed for this: R is S there.  `p --> q` restricts the
          // scan to R, which is defined through `ord`, so it still pays the sweep -- but not the
          // two searches.
          decided = true;
          verdict = false;
          // The stop scan comes FIRST and R is built only if it finds something.  R is defined
          // through `ord`, and for `p --> q` that means a second full successor sweep -- 624,452
          // records on twolock10 `P0.reqB --> P0.waitB` -- whose only use is to decide whether a
          // witness found here composes back to an accepting node.  On a row with no maximal-run
          // witness at all, which is every satisfied row, that sweep would be paid for nothing.
          // `E[] psi` has R = S and needs no sweep either way.
          std::vector<char> has_stop(n, 0);
          bool stop_outside_R = false;
          for (int v : S) {
            if (!leadsto && stop_known_empty(v)) continue;   // decided at expansion
            if (stop_set(v).empty()) continue;
            has_stop[v] = 1;
            if (!leadsto) { verdict = true; break; }
            if (inR.empty()) { refresh_ord(S, inS); build_R(); }
            if (inR[v]) { verdict = true; break; }
            stop_outside_R = true;      // a real witness the composition cannot reach yet
          }
          if (!verdict && stop_outside_R) {
            // Every component is dead, so `stop` is the only witness shape left, and one of them
            // sits beyond a cover.  Refine what blocks R from reaching it rather than reporting a
            // verdict this readout cannot support.  `has_stop` is complete here: the scan above
            // ran to the end of `S` precisely because it never found one inside R.
            owe_r_extensions([&](int v) { return has_stop[v] != 0; });
            if (r_block != 0) decided = false;
          }
          if (dbg) {
            std::size_t rsz = S.size();
            if (!inR.empty()) { rsz = 0; for (int v : S) if (inR[v]) ++rsz; }
            std::fprintf(stderr, "FWD-DIAG round=%zu nrec=%zu |S|=%zu |R|=%zu all-components-dead"
                                 " stop_witness=%d rblock=%zu lost=%zu setup+crit=%.1fms"
                                 " stopscan=%.1fms\n",
                         rounds, n, S.size(), rsz, (int)verdict, r_block, lost_edges, ms_crit,
                         std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count() / 1000.0 - ms_crit);
          }
        } else {

        refresh_ord(S, inS);
        build_R();

        const auto t_scc = std::chrono::steady_clock::now();
        // (1) A cycle of ORDINARY edges is an accepting run -- every node of the psi-closure is
        //     accepting -- so this is a witness and the search stops here.
        std::vector<int> comp;
        std::vector<char> cyclic;
        auto ord_succ = [&](int v, std::vector<int>* out) {
          for (int w : ord[v]) out->push_back(w);
        };
        scc_of(n, inS, ord_succ, &comp, &cyclic);
        // Only a component R reaches.  R is closed under ordinary edges, so a cyclic component
        // with one member in R lies wholly inside it.
        for (int v : S)
          if (inR[v] && comp[v] >= 0 && cyclic[comp[v]]) { ++exact_cycles; break; }

        if (exact_cycles != 0) {
          decided = true;
          verdict = true;
        } else {
          // (2) C5: a subsumption edge on a cycle makes the graph not liveness compatible.
          //     Pruned three ways, each sound:
          //       * the cover must lie on a cycle of the FULL graph at all;
          //       * its target key's component must be able to host an infinite psi-run, or
          //         there is no witness there to lose whatever the cycle structure says;
          //       * its source must not be settled.
          // The recurrence criterion, ON by default.  C5 alone is sound and complete by
          // Theorem 6, but it is purely structural: with the psi-closure one big component it
          // condemns every cover, and the loop then refines toward the containment graph --
          // arbiter6 q3 and fischer6 q3 do not finish.  The criterion supplies the semantic half:
          // a cover whose target key sits in a component that cannot host an infinite psi-run is
          // safe however many cycles pass through it, because there is no witness there to lose.
          // With it those rows answer in milliseconds.
          //
          // It is a sufficient condition for "dead" and so can only ever leave MORE covers unsafe
          // than necessary, never fewer -- it cannot manufacture a verdict, only save work.
          // `--criterion` selects how the components it leaves live are settled.  It has already
          // run, above the ordinary-edge sweep, because an all-dead census retires the round
          // outright; `live_at` is its readout.
          std::vector<int> fcomp;
          std::vector<char> fcyclic;
          auto all_succ = [&](int v, std::vector<int>* out) {
            for (const OtfEdge& e : adj[v])
              if (inS[e.tgt]) out->push_back(e.tgt);
          };
          scc_of(n, inS, all_succ, &fcomp, &fcyclic);
          // WHY a round reached the answer it did.  The three numbers that separate the possible
          // causes of a negative verdict: `fullcycles` distinguishes "the closure is acyclic, so
          // no infinite run exists at all" from "it has cycles but every cover on one was ruled
          // safe"; `dead_tgt` says how many of those cycle edges the recurrence criterion
          // exonerated; and `lost` says whether the view was the whole graph.
          if (dbg) {
            std::size_t fcyc = 0, dead_tgt = 0, expanded_n = 0;
            for (char c : fcyclic) fcyc += c ? 1 : 0;
            for (std::size_t v = 0; v < n; ++v) if (expanded[v]) ++expanded_n;
            for (int v : S)
              for (const OtfEdge& e : adj[v])
                if (inS[e.tgt] && fcomp[v] >= 0 && fcomp[v] == fcomp[e.tgt] &&
                    fcyclic[fcomp[v]] && !live_at(e.tgt)) ++dead_tgt;
            std::fprintf(stderr,
                         "FWD-DIAG round=%zu nrec=%zu expanded=%zu |S|=%zu fullcycles=%zu "
                         "dead_tgt=%zu lost=%zu\n",
                         rounds, n, expanded_n, S.size(), fcyc, dead_tgt, lost_edges);
          }

          // UNSAFE components, per Definition 3.3: a component is unsafe when it carries a
          // subsumption edge with both ends inside it AND contains an accepting node.
          //
          // "Accepting" is where our criterion enters the paper's framework.  Every node of the
          // closure is nominally accepting, which makes C5 condemn every cover.  But
          // a component the criterion proves cannot host an infinite psi-run has, for Buchi
          // purposes, no accepting node at all -- no run visits it infinitely often -- so C5 has
          // nothing to say about its cycles.  `live_at` is exactly that test.
          std::vector<char> comp_clean(fcyclic.size(), 1);
          for (int v : S) {
            if (settled[v] || !inR[v]) continue;   // a cover R cannot reach owes nothing
            for (const OtfEdge& e : adj[v]) {
              ++edges;
              if (std::find(ord[v].begin(), ord[v].end(), e.tgt) != ord[v].end()) continue;
              ++covers;
              if (!inS[e.tgt]) continue;
              if (fcomp[v] < 0 || fcomp[v] != fcomp[e.tgt] || !fcyclic[fcomp[v]]) continue;
              if (!live_at(e.tgt)) continue;
              ++unsafe;
              comp_clean[fcomp[v]] = 0;
              push_obligation(v, e.ordinal, e.tgt);
            }
          }
          // The other half of the obligation set; see `owe_r_extensions`.  A node can still carry
          // a readout if it sits in a cyclic component whose keys can host an infinite psi-run --
          // refinement may yet turn that component's cycle ordinary -- or if it already ends a
          // maximal run.
          owe_r_extensions([&](int v) {
            if (fcomp[v] >= 0 && fcyclic[fcomp[v]] && live_at(v)) return true;
            return !stop_set(v).empty();
          });
          // Nodes of the UNSAFE components are barred from being cover targets for the
          // re-exploration that follows, per Herbreteau et al. section 5.  Without this the
          // successors of a freshly materialised node are immediately subsumed by nodes of the
          // very component it was extracted from, the exploration stops in one step, and the
          // loop refines forever -- which is what arbiter and single did.
          // SAFE components: those that cannot REACH an unsafe one.  This is the marking from
          // section 5 that the earlier attempt got wrong -- it barred every component that was
          // not locally clean, which on a closure that is one big component barred essentially
          // every cover target and made the re-exploration explode (arbiter4 q3, 12.5s -> 34.4s).
          //
          // Reachability is what the paper actually asks for, and it is much more generous: a
          // component with no unsafe cover of its own is still safe only if nothing downstream of
          // it is unsafe, but equally a whole tail of the graph becomes safe at once.  Safe nodes
          // stay legal cover targets, so the re-exploration keeps almost all of its subsumption.
          //
          // Tarjan emits a component only after everything reachable from it, so successors carry
          // SMALLER ids and one ascending pass propagates the flag.
          // ONE ASCENDING PASS, which is what the paragraph above describes and what the
          // previous code did not do: it iterated RECORDS, whose order says nothing about
          // component id, so a component could be flagged after its predecessors had already been
          // scanned and the flag needed a repeat-until-quiet loop to travel -- O(depth of the
          // condensation) passes over every edge, inside the refinement loop.  Bucketing the
          // cross-component edges by source component once makes the propagation a single sweep
          // in ascending id order, because a successor's id is always smaller.
          std::vector<char> reaches_unsafe(fcyclic.size(), 0);
          for (std::size_t c = 0; c < fcyclic.size(); ++c) reaches_unsafe[c] = !comp_clean[c];
          std::vector<std::vector<int>> csucc(fcyclic.size());
          for (int v : S)
            for (const OtfEdge& e : adj[v]) {
              if (!inS[e.tgt] || fcomp[v] < 0 || fcomp[e.tgt] < 0) continue;
              if (fcomp[v] == fcomp[e.tgt]) continue;
              csucc[fcomp[v]].push_back(fcomp[e.tgt]);
            }
          for (std::size_t c = 0; c < fcyclic.size(); ++c)
            if (!reaches_unsafe[c])
              for (int w : csucc[c])
                if (reaches_unsafe[w]) { reaches_unsafe[c] = 1; break; }
          // A component that reaches nothing unsafe is finished with, permanently.
          for (int v : S)
            if (!settled[v] && fcomp[v] >= 0 && !reaches_unsafe[fcomp[v]]) settled[v] = 1;
          // Cover targets for the re-exploration: SAFE or FRESH only.
          online_unsafe_target_.assign(nrec(), 0);
          for (int v : S)
            if (fcomp[v] >= 0 && reaches_unsafe[fcomp[v]]) online_unsafe_target_[v] = 1;
          if (unsafe == 0 && lost_edges != 0) {
            // THE VIEW IS NOT THE GRAPH.  `lazy_snapshot` could not place some edge whose target
            // is a psi-record with no expanded representative, so "no ordinary cycle and no
            // unsafe cover" was decided on a subgraph and proves nothing: the missing edge may be
            // the one that closes a witness cycle.  Decline and let the exact backward route
            // answer, rather than report a verdict this view cannot support.  The guard above
            // removes the one cause we know of, so this is a safety net and not a routine exit.
            truncated = true;
          } else if (unsafe == 0 && r_block == 0) {
            // BOTH HALVES, and this is the second one.
            //
            // Both queries are witnessed by an infinite run OR by a maximal FINITE one -- a
            // deadlock, or a state time can diverge in.  The cycle search above decides only the
            // infinite half.  Answering "no witness" on "no ordinary cycle and liveness
            // compatible" silently assumes the forward search's early-exit caught every `Stop`
            // witness, and on single4 it does not: q2 and q3 are satisfied, no cycle exists, and
            // the route reported `NOT`.
            //
            // ::LivenessEngine::stop_set is the same term the backward operator uses, computed
            // exactly on the stored zone, and a non-empty one at any `v` of the accepting set IS
            // a witness.  Every such `v` is joined to a seed by a path inside the set -- the
            // initial vertex for `E[] psi`, an accepting `p /\ !q` node for `p --> q` -- so the
            // maximal run that ends at `v` composes with that path into a maximal run from the
            // seed.  One pass over `S`, no fixpoint.
            decided = true;
            verdict = false;
            for (int v : S)
              if (inR[v] && !(!leadsto && stop_known_empty(v)) && !stop_set(v).empty()) {
                verdict = true;
                break;
              }
          }
        }
        ms_scc += std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - t_scc).count() / 1000.0;
        }                                          // end: some component is live
      }
    }
    lazy_restore(&saved_adj, &saved_key);

    const auto report = [&](const char* how) {
      const double ms = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - t0).count() / 1000.0;
      if (dbg)
        std::fprintf(stderr,
                     "FWD nodes=%zu->%zu edges=%zu covers=%zu cycles=%zu unsafe=%zu rblock=%zu "
                     "rounds=%zu refined=%zu %s%s ms=%.2f (ord=%.2f scc=%.2f refine=%.2f)\n",
                     nodes_before, nrec(), edges, covers, exact_cycles, unsafe, r_block, rounds,
                     refined,
                     leadsto ? "leadsto " : "", how, ms, ms_ord, ms_scc, ms_refine);
    };

    if (truncated) { report("undecided=truncated"); return false; }
    if (decided) {
      report(verdict ? "witness=yes" : "witness=no");
      *witness = verdict;
      return true;
    }
    if (obligations.empty()) {                // unsafe covers with nothing left to refine
      report("undecided=stuck");
      return false;
    }

    // Refine the whole wave, then look again.  No fixpoint runs here, so a wave costs a forward
    // replay per obligation plus the incremental re-classification above -- which is why this can
    // afford to refine thousands of covers where the backward loop could not.
    const auto t_ref = std::chrono::steady_clock::now();
    const std::size_t before_nodes = nrec();
    // BATCH BY SOURCE.  ::LivenessEngine::online_replay runs a full ::tagged_successors over the
    // source to recover one successor, and a source typically owes several obligations -- 10,248
    // obligations over ~1,280 sources on arbiter8 q3, so the same expansion was repeated eight
    // times.  Sorting by source lets the replay cache hit for the rest of each group; it is 94%
    // of the loop's runtime, so this is where the time is.
    std::sort(obligations.begin(), obligations.end(),
              [](const LazyObligation& a, const LazyObligation& b) {
                if (a.src != b.src) return a.src < b.src;
                return a.ordinal < b.ordinal;
              });
    settled.assign(settled.size(), 0);        // the graph is about to move
    fwd_refining_ = true;                     // from here on nodes must stay single post images
    online_compact_complete_ = false;         // ::online_intern now skips barred cover targets
    lazy_replay_src_ = -1;                    // drop the replay cache: a new wave, fresh records
    lazy_replay_succ_.clear();
    for (const LazyObligation& ob : obligations) {
      ++refined;
      if (ob.src < static_cast<int>(dirty.size())) dirty[ob.src] = 1;
      if (!lazy_materialise(ob)) { report("undecided=replay-failed"); return false; }
      if (cancelled() || online_failed_) { report("undecided=aborted"); return false; }
      // The replay drains the forward worklist, so the probe can prove the query here; see the
      // top of the loop.  Either early-witness shape counts -- a `p /\ !q` valuation that ends a
      // maximal run, or an `E[] psi` terminal one -- and both are concrete.  Stopping now also
      // stops the wave from queueing records that would never be expanded.
      if (eg_early_hit_) { report("witness=yes-probe"); *witness = true; return true; }
    }
    dirty.resize(nrec(), 1);
    for (std::size_t v = before_nodes; v < nrec(); ++v) dirty[v] = 1;
    ms_refine += std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - t_ref).count() / 1000.0;
    online_unsafe_target_.clear();          // the restriction lasts only for this re-exploration
  }
}

}  // namespace hora
