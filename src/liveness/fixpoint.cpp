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
  for (int v : S) {
    seeded[v] = 1;
    nosucc[v] = stop_set(v);
  }
  // When the forward graph was NOT restricted, the stored zones still straddle `psi` and the
  // fixpoint has to do the confining itself: `W[v]` starts at `Z_v /\ psi(v)`, not at `Z_v`. The
  // seed is otherwise left unmaterialised (see `seeded`), so only the nodes that actually need
  // cutting are written down here.
  const bool confine = avoid_ && !avoid_forward_;
  std::vector<Federation> psi_seed;
  if (confine) psi_seed.resize(N);          // only the region route ever writes one
  if (confine)
    for (int v : S) {
      SymState cst;
      const SymState& sv = sread(v, cst);
      if (!avoid_->applies(sv.locs, sv.ints)) continue;
      Federation cut;
      avoid_->restrict(sv.zone, sv.locs, sv.ints, cut);
      psi_seed[v] = std::move(cut);
      seeded[v] = 0;                       // no longer "logically its own zone"
      W[v] = psi_seed[v];
      if (W[v].empty()) { nosucc[v].clear(); continue; }
      // The stop set was computed against the whole zone; cut it to psi as well, or a valuation
      // outside psi could stop the run and count as a witness.
      if (!nosucc[v].empty()) nosucc[v] = fed_intersect(nosucc[v], psi_seed[v]);
    }
  // Widening scratch, reused across the whole sweep so a narrow-stored zone costs a copy of
  // its entries and never an allocation.
  Dbm vsc, jsc;
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
      if (!nosucc[v].empty()) { live = true; break; }  // the run may legitimately stop here
      for (const auto& e : adj[v])
        if (inS[e.tgt] && (e.tgt == v || live_node[e.tgt])) { live = true; break; }
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
    if (seeded[v]) return includes_stored(z, v);
    for (const auto& w : W[v])
      if (!w.empty() && !z.includes(w)) return false;
    return true;
  };
  auto relax = [&](int v) -> bool {
    if (!seeded[v] && fed_is_empty(W[v])) return false;
    Federation cand = nosucc[v];
    for (const auto& c : cand)
      if (covers_value(c, v)) return false;
    for (const auto& e : adj[v]) {
      const int j = e.tgt;
      if (!inS[j]) continue;
      if (!seeded[j] && fed_is_empty(W[j])) continue;
      // A seeded successor contributes exactly one zone, its own, without being written down.
      const Dbm* one = seeded[j] ? &zread(j, jsc) : nullptr;
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
          if (covers_value(pz, v)) return false;   // the union already covers W[v]; see above
          cand.push_back(std::move(pz));
        }
      }
    }
    // Materialise v's own seed now, for this one node only: the swap below overwrites it with
    // the refinement, so the copy does not outlive the iteration.
    if (seeded[v]) { W[v].clear(); W[v].push_back(zread(v, vsc)); seeded[v] = 0; }
    fed_compact(cand, fedcap, &capped);
    // `W[v] /\ cand`, computed directly rather than as the double complement
    // `subtract(z, subtract(z, cand))` the textbook formulation uses. Both denote the same
    // set, but the inner subtraction peels z by every constraint of every member of cand, so
    // its output is an arrangement whose size is not bounded by |cand| at all, and the outer
    // subtraction then peels z by all of those. The intersection of two zones is a single
    // zone, so the direct form needs exactly |cand| pieces.
    Federation nw;
    // A converged sweep still visits every node, and near the fixpoint most visits change
    // nothing. Detect that per member: if some `c` in cand already contains z, then
    // `z /\ cand == z`, and every other product `z /\ c'` is a subset of that, so the inner
    // loop below can only produce pieces the compaction would absorb again. Emit z and move on.
    //
    // If that holds for every member, `nw` denotes W[v] member for member, so W[v] is unchanged
    // and there is nothing to propagate -- skip the compaction and, more to the point, skip
    // ::fed_contains, which peels W[v] by every member of nw through fed_subtract.
    bool unchanged = true;
    for (const auto& z : W[v]) {
      bool whole = false;
      for (const auto& c : cand)
        if (c.includes(z)) { whole = true; break; }
      if (whole) { nw.push_back(z); continue; }
      unchanged = false;
      for (const auto& c : cand) {
        Dbm t = z;
        t.intersect(c);
        if (!t.empty()) nw.push_back(std::move(t));
      }
    }
    if (unchanged) return false;
    fed_compact(nw, fedcap, &capped);
    const bool shrank = !fed_contains(W[v], nw);
    W[v].swap(nw);
    return shrank;
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
    if (seeded[v]) { W[v].clear(); W[v].push_back(zread(v, vsc)); seeded[v] = 0; }
  fixpoint_iterations += it;
  if (std::getenv("HORA_GFP_PROFILE")) {
    std::size_t big = 0, nontrivial = 0;
    for (const auto& c : sccs) { big = std::max(big, c.size()); if (c.size() > 1) ++nontrivial; }
    std::size_t stopn = 0, stopmembers = 0, wmembers = 0;
    for (int v : S) {
      if (!nosucc[v].empty()) { ++stopn; stopmembers += nosucc[v].size(); }
      wmembers += W[v].size();
    }
    std::fprintf(stderr, "GFP-PROFILE |S|=%zu N=%zu sccs=%zu nontrivial=%zu maxcomp=%zu "
                 "visits=%zu pre=%zu stopnodes=%zu stopmembers=%zu wmembers=%zu "
                 "fedmax=%zu live=%zu\n",
                 S.size(), N, sccs.size(), nontrivial, big, it, pre_calls, stopn, stopmembers,
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
