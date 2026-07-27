// netreach.hpp — E<> reachability over a network of timed automata.
#pragma once

#include "ptctlmc/network.hpp"
#include "ptctlmc/reach.hpp"

namespace ptctlmc {

// GPU (Metal/CUDA) merge routing. NAIVE (default): the passed-list merge always runs as one GPU
// thread per discrete-state bucket. ADAPTIVE: the flat driver calibrates the GPU vs 12-core-CPU merge
// throughput on early BFS levels and routes each later level to the faster path — rescuing
// merge-heavy models (few, fat antichains) where one thread per bucket is the wrong shape. No effect
// on the CPU-only (seq/omp) backends. Set once from the CLI (`--adaptive`).
void set_adaptive_merge(bool on);
bool adaptive_merge();

// GPU passed-set storage (`--gpu-intern`). The GPU driver stores zones DENSE (its merge kernels
// consume dense matrices), one copy per (discrete state, zone). On models where few distinct zones
// are reachable that is hugely redundant -- viking15 holds 2.77M zones drawn from 120 distinct ones
// -- and deduplicating them halves peak RSS there AND cuts wall-clock ~25%, because the per-level
// rebuild then writes a 4-byte index per survivor instead of copying a full dense zone. It is
// neutral on models whose zones are nearly all distinct (fischer, arbiter), so it is opt-in rather
// than default; see the evaluation in docs/paperopenmp2.
void set_gpu_intern(bool on);

// GPU frontier chunking (`--chunking`). Off by default: a BFS level is expanded in one shot, so the
// unified-memory pools are sized to the WIDEST level and peak memory scales with level width. On, the
// frontier is processed in bounded batches, so the pools are sized to a chunk instead and peak memory
// becomes independent of level width (raising the model size that fits in RAM). Sound: the frontier is
// a worklist, so processing it in batches only changes the schedule — it may shift the order-sensitive
// convex-union counts slightly, never the verdict. `states` = frontier states per chunk.
void set_chunking(bool on, int states = 65536);
bool chunking();

// ---- general modelling-language guard/update -------------------------------
// The `var OP const` / `var := const` fast paths live in NEdge::int_guard and
// NEdge::int_updates; anything richer (arithmetic, arrays, calls) is compiled
// into NEdge::gen_guard / NEdge::gen_update. Both backends MUST route through
// these two so CPU and GPU cannot drift apart.
//
// gen_ok:    evaluates gen_guard against `ints` WITHOUT mutating it (a guard may
//            call a function, so it runs on a scratch copy). True if absent.
// gen_apply: runs gen_update in place, AFTER int_updates have been applied.
//            Returns false if it faulted (bad index, division by zero, runaway
//            loop) or if a write left a variable's declared range -- either way
//            the transition is discarded. True (and a no-op) if absent.
//
// Taken as pointer + length rather than a vector so the pool-allocated IntVec of
// the CPU path and the plain std::vector of the GPU host loop can both call them.
bool gen_ok(const Network& net, const NEdge& e, const long* ints, std::size_t n);
bool gen_apply(const Network& net, const NEdge& e, long* ints, std::size_t n);

// E<> : is a discrete/clock state matching `target` reachable? Zone-graph
// exploration with a passed list keyed by discrete state (location vector + int
// values), inclusion subsumption + exact-DBM hash + LU extrapolation.
bool reachable_net(const Network& net, const NetTarget& target, Backend backend = Backend::Seq,
                   ReachStats* stats = nullptr);

// ---- Liveness (location/integer predicates) -------------------------------
// A per-discrete-state predicate: a conjunction of location (in)equalities and
// integer conditions. Clock constraints are NOT allowed here (would need zone
// splitting) — the query layer rejects them as unsupported.
struct LocReq { int proc; int loc; bool equal; };  // equal ? loc==loc : loc!=loc
struct DiscPred {
  std::vector<LocReq> locs;
  std::vector<IntAtom> ints;
};

// E[] phi : does there exist an infinite (time-divergent) run all of whose states
// satisfy `phi`? Detected as a reachable cycle — or a location where time can
// diverge — in the phi-restricted extrapolated zone graph.
bool exists_invariant_run(const Network& net, const DiscPred& phi, Backend backend = Backend::Seq,
                          ReachStats* stats = nullptr);

// p --> q  (leads-to): does every reachable state satisfying `p` inevitably reach a state
// satisfying `q` on all maximal runs? Location/integer predicates only.
bool leads_to_net(const Network& net, const DiscPred& p, const DiscPred& q,
                  Backend backend = Backend::Seq, ReachStats* stats = nullptr);

// E<> deadlock : is a deadlock state reachable? A symbolic state is a deadlock when it
// has NO action successor -- no edge (including synchronisation) is enabled anywhere in
// its delay-closed zone, so the network can only let time pass with nothing left to do.
// Explored on the exact-hash LU-extrapolated zone graph; extrapolation preserves guard
// enabledness (guards only compare clocks to constants <= the max), so a state deadlocks
// iff its extrapolated image does -- the same `adj.empty()` convention leads_to_net uses.
// `A[] not deadlock` is then !deadlock_reachable(...).
bool deadlock_reachable(const Network& net, Backend backend = Backend::Seq,
                        ReachStats* stats = nullptr);

}  // namespace ptctlmc
