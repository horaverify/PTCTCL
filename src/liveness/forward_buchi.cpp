/// @file src/liveness/forward_buchi.cpp
/// @brief PROTOTYPE: decide `E[] psi` by cycle detection on the forward graph, with no backward
///        pass at all.  Enabled by HORA_FWD; the production route is unaffected.
///
/// The backward greatest fixpoint is 89-96% of the runtime on the expensive refuted rows
/// (fischer8 q3, twolock10 q3).  It is there because a cycle in a SUBSUMED zone graph need not
/// correspond to a run -- a cover edge can close a loop no concrete execution realises.  This
/// file asks what happens if the cover edges are simply excluded from the cycle search instead.
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
/// For `E[] psi` every node of the psi-closure is accepting -- the run has only to stay in psi --
/// so C5 collapses to "no cover edge lies on a cycle", and the whole verdict is two Tarjan runs:
///
///   * a cycle among EXACT edges is an accepting run          -> SAT
///   * a cover edge inside a cycle violates C5                -> that cover must be refined
///   * neither                                                -> NOT
///
/// No `Pre`, no federations, no zone algebra whatsoever in the decision.
///
/// @note The premise the cycle argument needs -- a node being one abstracted post image -- is met
/// here.  The classical argument behind Theorem 6 needs every node to be exactly
/// `a(post_t(.))` of a single predecessor: a cycle then gives `a(post^n(Z)) = Z`, and since
/// `a(empty) = empty`, `Z` non-empty forces `post^n(Z)` non-empty, which is what yields the
/// concrete run.  ::LivenessEngine::online_intern deliberately breaks that premise -- it merges by
/// exact convex union (`Dbm::try_union`), and the fixed-map ::LivenessEngine::intern also rewrites
/// zones in place -- so a stored zone can be strictly larger than any single post image, and then
/// `post^n(Z_0)` can be EMPTY while every zone on the cycle is non-empty.  That is precisely the
/// unsoundness hora/liveness.hpp records: "`Z1 = post(Z)` gives every valuation of `Z1` a
/// predecessor in `Z`, not every valuation of `Z` a successor in `Z1`."
///
/// So a cycle found here is a *candidate*, and this file exists to measure how often the candidate
/// is right and what the cycle search costs, not to be believed.  Reading HORA_FWD_TRUST=1 makes
/// it answer anyway, for differential runs against the backward route.

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

bool LivenessEngine::forward_buchi_eg(bool* sat) {
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
        bool ordinary = online_equiv(xs, e.tgt, online_key_of_[e.tgt]);
        if (!ordinary) {
          // A merged target is a union of exact post images, so the successor may match one of
          // the PIECES even though it cannot match the union.  That is the same question the
          // unmerged graph would have answered; see ::LivenessEngine::online_merge_pieces_.
          const auto it = online_merge_pieces_.find(e.tgt);
          if (it != online_merge_pieces_.end())
            for (const Dbm& pc : it->second)
              if (pc == succ[e.ordinal].zone) { ordinary = true; break; }
        }
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
    // E[] probe -- an exact backward solve over the partial closure -- and stops the moment it
    // succeeds; the refinement wave below drains, so the hit can land in the middle of one.  That
    // flag is the verdict and must be read before anything else, because the same stop leaves
    // every record the wave had queued UNEXPANDED, and `lazy_snapshot` then projects a truncated
    // graph in which the cycles this search is looking for have been cut away.  Measured on
    // single5 `E[] x0 <= 6`: the probe hit at 381 records, the drains stopped, and the next two
    // rounds watched the full-graph cycle count fall 78 -> 15 -> 0 and answered NOT for a
    // satisfied query.
    if (eg_early_hit_) { *sat = true; return true; }

    std::vector<std::vector<OtfEdge>> saved_adj;
    std::vector<int> saved_key;
    std::size_t lost_edges = 0;
    if (!lazy_snapshot(&saved_adj, &saved_key, &lost_edges)) return false;

    bool decided = false, verdict = false, truncated = false;
    std::size_t exact_cycles = 0, unsafe = 0, covers = 0, edges = 0;
    std::vector<LazyObligation> obligations;
    {
      const std::vector<int> S = eg_closure();
      if (S.empty()) {
        decided = true;                       // node 0 fails psi: no run can stay in it
        verdict = false;
      } else {
        const std::size_t n = nrec();
        std::vector<char> inS(n, 0);
        for (int v : S)
          if (v >= 0 && v < static_cast<int>(n)) inS[v] = 1;
        settled.resize(n, 0);
        refresh_ord(S, inS);

        const auto t_scc = std::chrono::steady_clock::now();
        // (1) A cycle of ORDINARY edges is an accepting run -- every node of the psi-closure is
        //     accepting -- so this is a witness and the search stops here.
        std::vector<int> comp;
        std::vector<char> cyclic;
        auto ord_succ = [&](int v, std::vector<int>* out) {
          for (int w : ord[v]) out->push_back(w);
        };
        scc_of(n, inS, ord_succ, &comp, &cyclic);
        for (std::size_t c = 0; c < cyclic.size(); ++c) exact_cycles += cyclic[c] ? 1 : 0;

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
          // HORA_FWD_NOCRIT turns it off to price it.
          std::vector<int> key_comp;
          std::vector<char> comp_live;
          lazy_build_key_sccs(&key_comp, &comp_live);
          auto live_at = [&](int node) {
            if (node < 0 || node >= static_cast<int>(nrec())) return true;
            const int k = online_key_of_[node];
            if (k < 0 || k >= static_cast<int>(key_comp.size())) return true;
            const int c = key_comp[k];
            if (c < 0 || c >= static_cast<int>(comp_live.size())) return true;
            return comp_live[c] != 0;
          };
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
          // "Accepting" is where our criterion enters the paper's framework.  For `E[] psi` every
          // node of the closure is nominally accepting, which makes C5 condemn every cover.  But
          // a component the criterion proves cannot host an infinite psi-run has, for Buchi
          // purposes, no accepting node at all -- no run visits it infinitely often -- so C5 has
          // nothing to say about its cycles.  `live_at` is exactly that test.
          std::vector<char> comp_clean(fcyclic.size(), 1);
          for (int v : S) {
            if (settled[v]) continue;
            for (const OtfEdge& e : adj[v]) {
              ++edges;
              if (std::find(ord[v].begin(), ord[v].end(), e.tgt) != ord[v].end()) continue;
              ++covers;
              if (!inS[e.tgt]) continue;
              if (fcomp[v] < 0 || fcomp[v] != fcomp[e.tgt] || !fcyclic[fcomp[v]]) continue;
              if (!live_at(e.tgt)) continue;
              ++unsafe;
              comp_clean[fcomp[v]] = 0;
              if (e.ordinal < 0 || online_forbidden_cover_.count(forbid_key(v, e.ordinal)))
                continue;
              const int slot = online_raw_slot_[v];
              int pos = -1;
              if (slot >= 0)
                for (std::size_t r = 0; r < online_raw_adj_[slot].size(); ++r)
                  if (online_raw_adj_[slot][r].ordinal == e.ordinal) {
                    pos = static_cast<int>(r);
                    break;
                  }
              if (pos >= 0)
                obligations.push_back(LazyObligation{v, e.ordinal, slot, pos, e.tgt,
                                                     online_key_of_[e.tgt], 0, 0});
            }
          }
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
          std::vector<char> reaches_unsafe(fcyclic.size(), 0);
          for (std::size_t c = 0; c < fcyclic.size(); ++c) reaches_unsafe[c] = !comp_clean[c];
          for (int v : S)
            for (const OtfEdge& e : adj[v]) {
              if (!inS[e.tgt] || fcomp[v] < 0 || fcomp[e.tgt] < 0) continue;
              if (fcomp[v] == fcomp[e.tgt]) continue;
              if (reaches_unsafe[fcomp[e.tgt]]) reaches_unsafe[fcomp[v]] = 1;
            }
          for (std::size_t pass = 1; pass < fcyclic.size(); ++pass) {  // to a fixpoint
            bool moved = false;
            for (int v : S)
              for (const OtfEdge& e : adj[v]) {
                if (!inS[e.tgt] || fcomp[v] < 0 || fcomp[e.tgt] < 0) continue;
                if (fcomp[v] == fcomp[e.tgt] || !reaches_unsafe[fcomp[e.tgt]]) continue;
                if (!reaches_unsafe[fcomp[v]]) { reaches_unsafe[fcomp[v]] = 1; moved = true; }
              }
            if (!moved) break;
          }
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
          } else if (unsafe == 0) {
            // BOTH HALVES, and this is the second one.
            //
            // `E[] psi` is satisfied by an infinite run OR by a maximal FINITE one -- a deadlock,
            // or a state time can diverge in.  The cycle search above decides only the infinite
            // half.  Answering `NOT` on "no ordinary cycle and liveness compatible" silently
            // assumes the forward search's early-exit caught every `Stop` witness, and on single4
            // it does not: q2 and q3 are satisfied, no cycle exists, and the route reported `NOT`.
            //
            // ::LivenessEngine::stop_set is the same term the backward operator uses, computed
            // exactly on the stored zone, and a non-empty one at any `v` of the psi-closure IS a
            // witness: `v` is joined to the initial state by a psi-path, so the maximal run that
            // ends there composes into a maximal run from the start.  One pass over `S`, no
            // fixpoint.
            decided = true;
            verdict = false;
            for (int v : S)
              if (!stop_set(v).empty()) { verdict = true; break; }
          }
        }
        ms_scc += std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - t_scc).count() / 1000.0;
      }
    }
    lazy_restore(&saved_adj, &saved_key);

    const auto report = [&](const char* how) {
      const double ms = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - t0).count() / 1000.0;
      if (dbg)
        std::fprintf(stderr,
                     "FWD nodes=%zu->%zu edges=%zu covers=%zu cycles=%zu unsafe=%zu rounds=%zu "
                     "refined=%zu %s ms=%.2f (ord=%.2f scc=%.2f refine=%.2f)\n",
                     nodes_before, nrec(), edges, covers, exact_cycles, unsafe, rounds, refined,
                     how, ms, ms_ord, ms_scc, ms_refine);
    };

    if (truncated) { report("undecided=truncated"); return false; }
    if (decided) {
      report(verdict ? "verdict=SAT" : "verdict=NOT");
      *sat = verdict;
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
    lazy_replay_src_ = -1;                    // drop the replay cache: a new wave, fresh records
    lazy_replay_succ_.clear();
    for (const LazyObligation& ob : obligations) {
      ++refined;
      if (ob.src < static_cast<int>(dirty.size())) dirty[ob.src] = 1;
      if (!lazy_materialise(ob)) { report("undecided=replay-failed"); return false; }
      if (cancelled() || online_failed_) { report("undecided=aborted"); return false; }
      // The replay drains the forward worklist, so the probe can prove the query here; see the
      // top of the loop.  Stopping now also stops the wave from queueing records that would
      // never be expanded.
      if (eg_early_hit_) { report("verdict=SAT-probe"); *sat = true; return true; }
    }
    dirty.resize(nrec(), 1);
    for (std::size_t v = before_nodes; v < nrec(); ++v) dirty[v] = 1;
    ms_refine += std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - t_ref).count() / 1000.0;
    online_unsafe_target_.clear();          // the restriction lasts only for this re-exploration
  }
}

}  // namespace hora
