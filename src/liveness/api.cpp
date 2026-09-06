/// @file src/liveness/api.cpp
/// @brief Query drivers and the public liveness entry points.
///
/// Selects a construction for the query at hand and returns a verdict.

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

void LivenessEngine::base_forward() {
      fwd_ok_ = build_all();
      if (!fwd_ok_) return;
      std::deque<int> work;
      std::vector<char> inq;
      auto push = [&](int v) {
        if (inq.size() < nrec()) inq.resize(nrec(), 0);
        if (!inq[v]) { inq[v] = 1; work.push_back(v); }
      };
      push(0);
      // early witness probe. ::LivenessEngine::eg_probe is sound in the sat direction on a partial
      // graph, so the search may stop the moment one fires; NOT is never concluded here.
      //
      // Hoisted out of the per-edge test below: ::LivenessEngine::is_nq opens with a
      // ::LivenessEngine::grow it does not need here, the loop running right after one, and at one
      // call per edge that is not free.
      const bool psi_prune = eg_early_;
      //
      // when TO probe: amortization, not tuning. A probe that misses is wasted work whose cost
      // grows with the graph, so it has to be paid for. The rule is that exploration since the
      // last probe must have cost at least as much as all the probing done so far; only then is
      // another probe affordable. Two things follow with no constant anywhere:
      //
      //   * total probe time can never exceed total exploration time, so a query whose witness
      //     never appears -- every `NOT` row -- pays at most a factor of two, by construction;
      //   * the spacing calibrates itself to the model. Where a probe is cheap against the
      //     exploration it interrupts the probes come thick and the witness is caught early;
      //     where the fixpoint is expensive (grc) they thin out on their own.
      //
      // The alternative, a geometric schedule on the node count, needs a first size and a ratio,
      // and both are fitted: ratio 4 puts consecutive checkpoints at 25% and 100% of the final
      // graph, so a witness only provable late costs the whole exploration.
      const auto clk_now = [] { return std::chrono::steady_clock::now(); };
      static const bool probe_dbg = [] { return std::getenv("HORA_EG_PROBEDBG") != nullptr; }();
      long long probe_ns = 0;                  // cumulative cost of every probe so far
      auto since_probe = clk_now();            // when exploration resumed after the last one
      while (!work.empty()) {
        const int u = work.front();
        work.pop_front();
        grow();
        if (inq.size() < nrec()) inq.resize(nrec(), 0);
        inq[u] = 0;
        expand(u);                                  // no-op if still expanded
        // A witness found inside expand() settles the query, so the search must stop here.
        // ::LivenessEngine::expand sets this for all three early-exit shapes -- a deadlocked
        // valuation under `A[] not deadlock`, a leads-to counterexample, an E[] terminal witness
        // -- and returns, but returning only ends THAT record: without this the loop drained the
        // rest of the worklist and explored the whole graph for an answer already decided.
        // ::LivenessEngine::online_drain_forward_work has always broken on the same flag; the
        // base route did not. Measured on phil10 `A[] not deadlock`: the hit fires at 35668
        // records and the search ran on to 71451.
        if (eg_early_hit_) return;
        grow();
        // A successor needs work exactly when it has no adjacency of its own: brand new, or reset
        // because a merge enlarged it.
        //
        // AND, for `E[] psi`, only if it still satisfies psi. A maximal psi-run is in psi at every
        // step, so a run reaching a non-psi node has already left psi and neither that node nor
        // anything beyond it can lie on a witness. Not expanding it therefore removes nothing from
        // ::LivenessEngine::eg_closure: a psi-node reachable only through a non-psi node was never
        // admitted to it in the first place. It removes 3x the stored nodes and almost no time,
        // because those nodes were never in the closure the fixpoint iterates.
        for (const auto& e : adj[u])
          if (!expanded[e.tgt] && !(psi_prune && !is_nq_fast(e.tgt))) push(e.tgt);
        if (psi_prune &&
            std::chrono::duration_cast<std::chrono::nanoseconds>(clk_now() - since_probe).count()
                >= probe_ns) {
          const auto p0 = clk_now();
          const bool hit = eg_probe();
          probe_ns +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(clk_now() - p0).count();
          if (probe_dbg)
            std::fprintf(stderr, "PROBE at_nodes=%zu hit=%d cumulative_probe_ms=%lld\n",
                         nrec(), (int)hit, probe_ns / 1000000);
          if (hit) { eg_early_hit_ = true; return; }
          since_probe = clk_now();
        }
      }
}

void LivenessEngine::build_forward(GfpStats* gs) {
  setup();
  base_forward();
  fill_stats(gs);
}

bool LivenessEngine::deadlock_reachable_refined(GfpStats* gs, bool* ok) {
  deadlock_scan_ = true;
  setup();
  base_forward();
  fill_stats(gs);
  *ok = fwd_ok_;
  if (!fwd_ok_) return false;
  if (deadlock_hit_) return true;          // the search stopped at the witness
  static const bool dl_dbg = std::getenv("HORA_DL_DBG") != nullptr;
  SymState sc;
  for (std::size_t i = 0; i < nrec(); ++i) {
    if (cancelled()) { *ok = false; return false; }
    if (detail::state_has_deadlock(net, sread(static_cast<int>(i), sc))) {
      if (dl_dbg)
        std::fprintf(stderr, "DL-POSTHOC found at rec=%zu of %zu  expanded=%d active=%d\n",
                     i, nrec(), (int)expanded[i], (int)online_active_[i]);
      return true;
    }
  }
  if (dl_dbg) std::fprintf(stderr, "DL-POSTHOC no record deadlocks (nrec=%zu)\n", nrec());
  return false;
}

bool LivenessEngine::finish_verdict(GfpStats* gs) {
  const bool violated = fwd_ok_ && final_verdict_violated();
  fill_stats(gs);
  return !violated;
}

bool LivenessEngine::eg_from_initial(GfpStats* gs) {
  setup();
  base_forward();
  if (eg_early_hit_) { fill_stats(gs); return true; }   // witness proven on the partial graph
  return eg_verdict(gs);
}

std::vector<int> LivenessEngine::eg_closure() {
  std::vector<int> S;
  if (nrec() == 0) return S;
  grow();
  // The psi-closure reached from the initial vertex. ::LivenessEngine::accepting_closure seeds from every
  // p-node because `p --> q` accepts a witness anywhere; here the run must start at the initial
  // state, so a node reachable only by leaving psi and coming back must not be admitted.
  if (!expanded[0] || !is_nq(0)) return S;
  std::vector<char> in(nrec(), 0);
  std::vector<int> stk{0};
  in[0] = 1;
  while (!stk.empty()) {
    const int v = stk.back();
    stk.pop_back();
    S.push_back(v);
    for (const auto& e : adj[v])
      if (!in[e.tgt] && expanded[e.tgt] && is_nq(e.tgt)) { in[e.tgt] = 1; stk.push_back(e.tgt); }
  }
  return S;
}

bool LivenessEngine::eg_probe() {
  if (!fwd_ok_ || nrec() == 0) return false;
  const bool saved_capped = capped;         // sticky, and a probe must not condemn the real run
  const bool saved_scc = scc_iter_;
  scc_iter_ = true;
  bool sat = false;
  const std::vector<int> S = eg_closure();
  if (!S.empty()) {
    std::vector<Federation> W;
    solve_greatest_fixpoint(S, W);
    if (!capped) sat = !fed_is_empty(W[0]);
  }
  capped = saved_capped;
  scc_iter_ = saved_scc;
  return sat;
}

bool LivenessEngine::eg_verdict(GfpStats* gs) {
  scc_iter_ = true;             // the verdict is one node's, so diameter is what costs here
  bool res = false;
  if (fwd_ok_ && nrec() != 0) {
    const std::vector<int> S = eg_closure();
    if (!S.empty()) {
      std::vector<Federation> W;
      solve_greatest_fixpoint(S, W);
      if (!capped) res = !fed_is_empty(W[0]);
    }
  }
  fill_stats(gs);
  return res;
}

bool LivenessEngine::run(GfpStats* gs) {
  setup();
  base_forward();
  // A concrete `p /\ EG !q` witness found during construction is already the verdict; see
  // ::LivenessEngine::leadsto_witness. `run` also serves `E[] psi`, whose own early hit is handled by its
  // caller, so this only fires when leads-to armed it.
  if (leadsto_early_ && eg_early_hit_) { fill_stats(gs); return false; }
  const bool violated = fwd_ok_ && final_verdict_violated();
  fill_stats(gs);
  return !violated;
}

/// @brief The one-construction D path used by ordinary non-diagonal liveness queries.
///
/// The map begins empty, but this is not a preliminary LU graph.  Active exact records and raw
/// semantic edges are built once; a provisionally covered successor is represented only by its
/// source-edge replay handle.  Atom pullback and expansion share one worklist, and only the
/// stable active quotient reaches the backward gfp.
bool LivenessEngine::decide_leads_to(GfpStats* gs) {
  // A violated `p --> q` needs no finished graph: see ::LivenessEngine::leadsto_witness. Armed before either
  // delegation below, so the exact and structurally-empty-D routes get it too. Disabled with
  // HORA_LEADSTO_EARLY=0, for measuring what it saves.
  leadsto_early_ = [] {
    const char* e = std::getenv("HORA_LEADSTO_EARLY");
    return !e || *e != '0';
  }();

  // Diagonal constraints already select the exact inclusion lane.  There is no a_LU/D quotient
  // to refine there, and preserving the existing exact route avoids imposing online bookkeeping
  // where it cannot buy anything.
  if (network_has_diagonal(net) || force_exact_) return run(gs);

  // D = a_LU is known before the first successor in a globally time-open network.  Entering the
  // revocable-cover machinery in that case cannot refine a single cover; it only duplicates the
  // ordinary a_LU graph's key, edge, and replay metadata.
  if (structurally_empty_d(*dead_net)) return run(gs);

  online_d_ = true;
  online_reset();
  online_perf = OnlinePerf{};
  fwd_ok_ = false;
  setup();
  const auto forward_start = std::chrono::steady_clock::now();
  online_base_forward();
  if (eg_early_hit_) {                     // a concrete `p /\ EG !q` witness was reached
    fill_stats(gs);
    return false;
  }
  const auto forward_done = std::chrono::steady_clock::now();
  const std::size_t rss_fwd = rss_mb();
  std::size_t raw_nodes = nrec(), raw_edges = 0, active_nodes = 0, active_slots = 0;
  std::size_t live_deferred = 0;
  for (const auto& edges : online_raw_adj_)
    live_deferred += std::count_if(edges.begin(), edges.end(),
                                  [](const OtfEdge& e) { return e.deferred >= 0; });
  for (std::size_t v = 0; v < online_active_.size(); ++v) active_nodes += online_active_[v] != 0;
  for (const OnlineKeyInfo& ki : online_keys_) {
    active_slots += ki.active_count;
    for (int v = ki.active_head; v >= 0; v = online_active_next_[v])
      if (online_raw_slot_[v] >= 0) raw_edges += online_raw_adj_[online_raw_slot_[v]].size();
  }
  // What the D machinery is holding, whether or not D ever gained an atom.
  std::size_t dropped_records = 0, dropped_unexpanded = 0, inactive_expanded = 0;
  for (std::size_t v = 0; v < nrec(); ++v) {
    const bool keep = expanded[v] && online_raw_slot_[v] >= 0;
    if (!keep) { ++dropped_records; if (!expanded[v]) ++dropped_unexpanded; }
    else if (!online_active_[v]) ++inactive_expanded;
  }
  const std::size_t key_count = online_key_by_id_.size();
  const std::size_t key_edge_count = online_key_edges_.size();
  const std::size_t seed_atoms = online_am_.seed_atoms;
  const std::size_t closure_atoms = online_am_.closure_atoms;
  const std::size_t seed_keys = online_am_.seed_keys;
  const std::size_t diag_keys = std::count_if(
      online_keys_.begin(), online_keys_.end(),
      [](const OnlineKeyInfo& ki) { return ki.diag && !ki.diag->empty(); });
  const std::size_t bumped = std::count_if(online_am_.bump.begin() + 1, online_am_.bump.end(),
                                           [](std::int32_t x) { return x != kNoBump; });
  if (fwd_ok_ && !cancelled() && !online_failed_) online_finalize_graph();
  const auto finalised = std::chrono::steady_clock::now();
  const std::size_t rss_fin = rss_mb();

  // `online_failed_` is defensive only (the online atom and reset representations are dynamic),
  // but never turn an unfinished quotient into a normal answer if an invariant is broken.
  if (online_failed_ || cancelled()) capped = true;
  const bool violated = fwd_ok_ && !online_failed_ && !cancelled() && final_verdict_violated();
  const auto solved = std::chrono::steady_clock::now();
  if (std::getenv("HORA_RSS"))
    std::fprintf(stderr, "RSS-MB forward=%zu finalize=%zu backward=%zu\n", rss_fwd, rss_fin, rss_mb());
  if (online_perf.enabled) {
    const auto ms = [](auto d) {
      return std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
    };
    std::fprintf(stderr,
                 "ONLINE-PERF dropped=%zu dropped_unexp=%zu inactive_exp=%zu "
                 "forward_ms=%lld finalize_ms=%lld backward_ms=%lld raw_nodes=%zu "
                 "active_nodes=%zu active_slots=%zu active_edges=%zu keys=%zu key_edges=%zu seed_keys=%zu "
                 "seed_atoms=%zu closure_atoms=%zu diag_keys=%zu bumped=%zu cover_calls=%llu "
                 "cover_ms=%.3f intern=%llu intern_scan=%llu probe_ms=%.3f "
                 "join_ms=%.3f append_ms=%.3f "
                 "deferred=%zu deferred_created=%llu deferred_replayed=%llu deferred_promoted=%llu "
                 "coverer=%llu coverer_scan=%llu "
                 "compact=%llu compact_scan=%llu deact=%llu deact_unexp=%llu deact_perm=%llu "
                 "recheck=%llu recheck_nodes=%llu succ_ms=%.3f "
                 "register_ms=%.3f key_ms=%.3f tag_ms=%.3f reset_ms=%.3f edge_ms=%.3f intern_ms=%.3f\n",
                 dropped_records, dropped_unexpanded, inactive_expanded,
                 static_cast<long long>(ms(forward_done - forward_start)),
                 static_cast<long long>(ms(finalised - forward_done)),
                 static_cast<long long>(ms(solved - finalised)), raw_nodes, active_nodes,
                 active_slots, raw_edges, key_count, key_edge_count, seed_keys, seed_atoms, closure_atoms,
                 diag_keys, bumped,
                 static_cast<unsigned long long>(online_perf.cover_calls),
                 online_perf.cover_ns / 1e6,
                 static_cast<unsigned long long>(online_perf.intern_calls),
                 static_cast<unsigned long long>(online_perf.intern_active_scan),
                 online_perf.intern_cover_probe_ns / 1e6, online_perf.intern_join_ns / 1e6,
                 online_perf.intern_append_ns / 1e6, live_deferred,
                 static_cast<unsigned long long>(online_perf.deferred_created),
                 static_cast<unsigned long long>(online_perf.deferred_replayed),
                 static_cast<unsigned long long>(online_perf.deferred_promoted),
                 static_cast<unsigned long long>(online_perf.coverer_calls),
                 static_cast<unsigned long long>(online_perf.coverer_active_scan),
                 static_cast<unsigned long long>(online_perf.compact_calls),
                 static_cast<unsigned long long>(online_perf.compact_active_scan),
                 static_cast<unsigned long long>(online_perf.compact_deact),
                 static_cast<unsigned long long>(online_perf.compact_deact_unexp),
                 static_cast<unsigned long long>(online_perf.compact_deact_permanent),
                 static_cast<unsigned long long>(online_perf.recheck_calls),
                 static_cast<unsigned long long>(online_perf.recheck_nodes),
                 online_perf.successor_ns / 1e6, online_perf.register_ns / 1e6,
                 online_perf.register_key_ns / 1e6, online_perf.register_tag_ns / 1e6,
                 online_perf.register_reset_ns / 1e6, online_perf.register_edge_ns / 1e6,
                 online_perf.intern_ns / 1e6);
  }
  fill_stats(gs);
  return !violated;
}

bool LivenessEngine::decide_exists_invariant_run(GfpStats* gs) {
  if (network_has_diagonal(net) || force_exact_) return eg_from_initial(gs);

  online_d_ = true;
  online_reset();
  online_perf = OnlinePerf{};
  fwd_ok_ = false;
  // Discrete states outside psi are terminal for this query and need no successor expansion.
  // This is forward pruning only: the sole backward pass still starts after D stabilises.
  online_eg_prune_ = true;
  setup();
  online_base_forward();
  if (online_perf.enabled) {
    std::fprintf(stderr,
                 "ONLINE-EG raw_nodes=%zu keys=%zu seed_keys=%zu seed_atoms=%zu closure_atoms=%zu "
                 "deferred_created=%llu deferred_replayed=%llu deferred_promoted=%llu\n",
                 nrec(), online_key_by_id_.size(), online_am_.seed_keys,
                 online_am_.seed_atoms, online_am_.closure_atoms,
                 static_cast<unsigned long long>(online_perf.deferred_created),
                 static_cast<unsigned long long>(online_perf.deferred_replayed),
                 static_cast<unsigned long long>(online_perf.deferred_promoted));
  }
  const std::size_t rss_fwd = rss_mb();
  if (eg_early_hit_) { fill_stats(gs); return true; }
  online_complete_eg_history();
  if (eg_early_hit_) { fill_stats(gs); return true; }
  if (fwd_ok_ && !cancelled() && !online_failed_) online_finalize_graph();
  if (online_failed_ || cancelled()) { capped = true; fill_stats(gs); return false; }
  const std::size_t rss_fin = rss_mb();
  const bool r = eg_verdict(gs);
  if (std::getenv("HORA_RSS"))
    std::fprintf(stderr, "RSS-MB eg_forward=%zu eg_finalize=%zu eg_backward=%zu nodes=%zu\n",
                 rss_fwd, rss_fin, rss_mb(), nrec());
  return r;
}

void LivenessEngine::fill_stats(GfpStats* gs) {
  if (gs) {
    gs->fwd_nodes = nrec();
    gs->disc_keys = bucket.size();
    gs->fwd_edges = 0;
    for (const auto& a : adj) gs->fwd_edges += a.size();
    gs->iters = fixpoint_iterations;
    gs->capped = capped;
    gs->fed_max = fed_max;
    gs->fed_avg = fed_avg;
    gs->nonempty = nonempty;
    gs->expanded_nodes = expand_calls;
    gs->passes = passes;
    gs->pre_calls = pre_calls;
    gs->early = settled_early;
    gs->outer_nodes = outer_nodes;
    gs->outer_expands = outer_expands;
  }
}

/// @brief `E[] psi` from the initial state, decided by the backward greatest fixpoint.
///
/// @param net the network, already restricted by `psi`'s clock conjunction if it had one
/// @param nq  a predicate holding exactly where `psi`'s discrete part fails, so `V_psi` is its
///            complement; carry an arbitrary boolean through hora::DiscPred::forbid
/// @param[out] gs statistics; may be null
/// @return @c true iff the initial state admits a maximal execution all of whose states satisfy
///         `psi`
///
/// The forward construction and D refinement are the same persistent construction used by
/// leads-to.  The difference is only the seed and readout of the final gfp: E[] starts from the
/// initial node and reads that node's value, whereas leads-to seeds every `p /\ !q` node.
///
/// @note @p witness_only remains in the public helper signature for compatibility.  The online
/// construction has one execution mode, so it does not select an alternative graph-building path.
bool exists_invariant_run_otf(const Network& net, const DiscPred& nq, GfpStats* gs,
                              const Network* unrestricted,
                              const detail::DelayAvoidSet* avoid, bool witness_only) {
  static const DiscPred kTrue{};
  auto make = [&](bool exact) {
    auto e = std::make_unique<LivenessEngine>(net, kTrue, nq, exact, nullptr, unrestricted);
    e->set_avoid(avoid);
    e->set_eg_early(true);                          // SAT may be concluded on a partial graph
    return e;
  };
  (void)witness_only;
  return make(false)->decide_exists_invariant_run(gs);
}

/// @brief `p --> q` where either operand constrains clocks or carries boolean structure.
///
/// @param net the network under analysis
/// @param p   the antecedent, as a positive DNF; evaluated at one valuation under `EF`
/// @param nq  `!psi` for `psi = !q`, i.e. `q` itself, as the region the run must avoid
/// @param[out] gs statistics; may be null
/// @return @c true iff `p --> q` holds
///
/// @note The forward graph is NOT restricted. `p --> q` is violated by `EF(p /\ EG !q)`, and the
///       path to the `p`-state is ordinary reachability that need not stay in `!q` -- so the
///       region confines the fixpoint only. `E[] psi` is the opposite case and restricts both.
bool leads_to_clocks(const Network& net, const DiscPred& p, const DiscPred& nq, GfpStats* gs) {
  detail::DelayAvoidSet av;
  std::vector<DiscPred> guards;
  guards.reserve(nq.forbid.size());
  for (const auto& t : nq.forbid) {
    if (t.clocks.empty()) continue;                 // purely discrete: the vertex filter has it
    detail::DelayAvoid one;
    one.forbid = Dbm::universe(net.nclocks);
    if (!apply_all(one.forbid, t.clocks) || one.forbid.empty()) continue;
    one.fup = one.forbid; one.fup.delay();
    guards.push_back(DiscPred{});
    guards.back().locs = t.locs;
    guards.back().ints = t.ints;
    av.parts.push_back(std::move(one));
  }
  for (std::size_t i = 0; i < av.parts.size(); ++i)
    av.parts[i].when = (guards[i].locs.empty() && guards[i].ints.empty()) ? nullptr : &guards[i];

  // The region's constants must reach the LU bounds and its clocks must stay live, for the same
  // reason they must in hora::exists_invariant_run: nothing else tells the abstraction that
  // the boundary matters, and a_LU would merge zones across it.
  Network bounded = net;
  if (!av.parts.empty()) {
    for (const auto& t : nq.forbid)
      for (const auto& c : t.clocks) {
        const std::int32_t k = raw_bound(c.bound) < 0 ? -raw_bound(c.bound) : raw_bound(c.bound);
        auto bump = [&](int idx, bool lo, bool hi) {
          if (idx <= 0 || idx >= (int)bounded.lbound.size()) return;
          if (lo) bounded.lbound[idx] = std::max(bounded.lbound[idx], k);
          if (hi) bounded.ubound[idx] = std::max(bounded.ubound[idx], k);
          for (auto& pl : bounded.lbound_loc) for (auto& l : pl)
            if (lo && idx < (int)l.size()) l[idx] = std::max(l[idx], k);
          for (auto& pu : bounded.ubound_loc) for (auto& l : pu)
            if (hi && idx < (int)l.size()) l[idx] = std::max(l[idx], k);
        };
        if (c.i && c.j) { bump(c.i, true, true); bump(c.j, true, true); }
        else if (c.j == 0 && c.i) bump(c.i, false, true);
        else if (c.i == 0 && c.j) bump(c.j, true, false);
        for (auto& pa : bounded.active) for (auto& l : pa) {
          if (c.i > 0 && c.i < (int)l.size()) l[c.i] = 1;
          if (c.j > 0 && c.j < (int)l.size()) l[c.j] = 1;
        }
      }
    bounded.active_mask.clear();
    for (const auto& pa : bounded.active) {
      bounded.active_mask.emplace_back();
      for (const auto& l : pa) {
        std::uint64_t m = 0;
        if (bounded.nclocks <= 63)
          for (int c = 1; c <= bounded.nclocks; ++c) if (l[c]) m |= std::uint64_t(1) << c;
        bounded.active_mask.back().push_back(m);
      }
    }
    if (bounded.nclocks > 63) bounded.active_mask.clear();
  }

  auto make = [&](bool exact) {
    auto e = std::make_unique<LivenessEngine>(bounded, p, nq, exact);
    e->set_avoid(av.parts.empty() ? nullptr : &av, /*forward=*/false);
    return e;
  };
  // The clock/boolean front-end uses the same persistent quotient.  Its forbidden region still
  // confines only the backward phase (`forward=false` above), while D atoms are discovered from
  // the ordinary reachable key skeleton as the online construction grows.
  return make(false)->decide_leads_to(gs);
}

/// @brief Decide the discrete leads-to query `p --> q`.
///
/// D starts empty and is refined while reachable keys and raw semantic edges are discovered.
/// A provisionally covered successor is kept only as an edge replay handle.  If a later atom
/// invalidates its certificate, that one successor is regenerated and materialised; the graph is
/// never rebuilt.  Once the coupled worklist stabilises, existing edges are redirected to the
/// active D representatives and one backward gfp decides the verdict.
bool leads_to_otf(const Network& net, const DiscPred& p, const DiscPred& q, GfpStats* gs) {
  LivenessEngine e(net, p, q);
  return e.decide_leads_to(gs);
}

}  // namespace hora