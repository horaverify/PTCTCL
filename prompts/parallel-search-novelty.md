# Prompt: find a genuinely new parallel search algorithm for the zone-graph model checker

Repo: `/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC` (branch `main`).
Deliverable: write incrementally to `docs/superpowers/specs/parallel-search-novelty.md`.
**Do not stop at the first plausible idea. Loop: generate, attack, discard, generate again, until a
candidate survives your own adversarial check with a number attached.**

## The goal, stated exactly

Find an algorithm that **explores or stores fewer symbolic states than UPPAAL** on `A[]` safety
queries, where **parallelism is part of why it works** — not a speedup bolted onto a sequential
method.

This is, at bottom, a **parallel graph-search problem**. Treat it as one. The system happens to be
timed automata, but the structure is: explore a huge implicit graph, avoid revisiting, avoid
storing more than necessary, do it on many cores.

## Encouragement, because the negatives below are long

The engine already **parallelises AND stores fewer states than UPPAAL** on several families
(csma7 1716 vs 2368; phil7 2228 vs 2606; single6 −66%). Thirty years of work says parallelising a
subsumption search makes state counts *grow* — Behrmann et al. measured 45 001 → 110 634 on 17
nodes — and this engine does the opposite. So the space is not closed; the obvious moves are.

## Established facts — contradicting these without evidence wastes the run

- On fischer, arbiter, viking, twolock, csma the engine stores **exactly one zone per reachable
  discrete state** and ties `verifyta`'s count to the digit, on *explored* as well as *stored*, at
  every size. Every antichain has exactly **one** member, so every pairwise zone technique is
  provably a no-op there.
- `verifyta -A` (maximal over-approximation) changes fischer10 by **zero** on both axes.
- Search order does not help: `verifyta -o{0,1,2}` gives identical counts; DFS is 31.6% worse on
  phil. Thread count leaves fischer8 bit-identical from 1 to 12 threads.
- `explored − stored` equals exactly the cross-level retirement count: ~41% of expansions on
  fischer10/twolock10, 28% on phil12, 0% on arbiter/viking/csma. Retirement lag is a **model
  constant** (4 on fischer, 6 on twolock) equal to the shortest key-preserving cycle of one
  template. Retroactive cancellation of those was implemented and cancels **zero**.
- The reachable discrete-state graph of every family is a **single SCC** covering 100% of states.
- Every clock is process-local; zero urgent and zero committed locations anywhere.

## Ruled out — do not propose these, each was measured or proved

symmetry reduction (the user considers it cheating and these benchmarks are trivially symmetric) ·
classical partial-order reduction · zone subsumption variants, federation/union coverage, coarser
abstractions of any kind · G-simulation and other simulation refinements · constraint-map
contraction · IC3/PDR and CEGAR (a sibling tool measured CEGAR a NET LOSS: 251 032 nodes vs 68 624,
because the finer structure was also acting as a canonicaliser — **any proposal that coarsens must
argue why it will not destroy merge opportunities and store MORE**) · sweep-line (proved vacuous:
single SCC) · cross-discrete-state subsumption (collapse factor exactly 1) · racing/portfolio of
two strategies (explicitly excluded by the user) · anything whose payoff is proportional to how
replicated the model is.

## What to actually explore

**1. Learn from OTHER FIELDS with the same structure.** This is the main instruction. Parallel
graph search, and things shaped like it, have thirty years of tricks that verification never
imported. For each, say what the analogue would be here and whether it survives:
 - **Direction-optimizing BFS** (Beamer, Asanović, Patterson, SC 2012): when the frontier is large,
   stop expanding forward and instead ask, for each unvisited vertex, whether any parent is in the
   frontier. It cuts edges traversed by an order of magnitude. What is the bottom-up step for a
   zone graph, and does the engine have (or could it cheaply get) predecessors?
 - Δ-stepping, bucket-based and priority-based parallel search.
 - Frontier compaction, work-stealing, and lock-free visited structures from GPU graph processing.
 - Memoisation and incremental/dynamic algorithms: dynamic connectivity, incremental SCC,
   self-adjusting computation, "reuse the answer you already computed".
 - Parallel SAT: what is actually shared, and why sharing beats dividing.
 - Planning and heuristic search: pattern databases, abstraction heuristics, duplicate detection in
   external-memory search (delayed duplicate detection is *exactly* "we already explored this").
 - Databases: query result caching, common subexpression elimination, join-order reuse.

**2. Reuse of work already done.** The user's phrasing: *reduce the links we already found*. In this
engine the successor zone is a function of `(source zone, edge)`, and on some models very few
DISTINCT zones exist — viking stores 2.77M zones drawn from **120 distinct ones**. So the same zone
transformation is recomputed enormous numbers of times. Is there a formulation where that
redundancy removes *nodes* and not merely work? Can a path be replayed rather than re-explored?

**3. Monotonicity, simulation, homomorphism — but NOT symmetry.** Is there a sound map `h` on the
search graph that preserves transitions and the property, and is not a group action? Look
specifically at:
 - **Well-structured transition systems**: a well-quasi-order on states with monotone transitions
   lets you store only minimal (or maximal) elements and work with downward-closed sets. This is the
   machinery behind the parameterized timed results. Does any wqo apply to these networks, and what
   would the stored set become?
 - Retractions/foldings of the discrete graph that are not permutations.
 - Any homomorphism whose quotient is smaller AND whose soundness does not require replication.

**4. Clever use of TIME itself.** The models are timed and the engine currently treats time only
through zones. Is there structure in the time dimension that a *search* can exploit — states that
differ only by a shift, orderings induced by clock values, bounds that let a whole subtree be
skipped?

**5. Designed-for-parallel from the start.** What would the algorithm be if it were designed for
1000 cores rather than adapted from a sequential one? Where does the level-synchronous barrier
have to go, and what replaces the guarantee it currently buys (it is kept because it minimises the
state count)?

## Method — this matters as much as the ideas

- **Measure, do not argue.** Binary: `./build-mostopt/ptctlmc`, `PTCTLMC_STATS=1` prints
  discrete-states / stored-zones / distinct-zones / generated. Models in `benchmarks/models`,
  queries in `benchmarks/queries`. `verifyta` at
  `/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta -u model.xml q.q`.
  Build: `cmake -S . -B build-mostopt -DCMAKE_BUILD_TYPE=Release && cmake --build build-mostopt -j8`.
- **Attack your own idea before reporting it.** For each candidate ask: is it sound; does it reduce
  STATES or only work; does the reduction survive the fact that antichains are singletons; does it
  need a property these models lack; is it published already (search, and cite authors/venue/year).
- **A measured negative is a real deliverable** — it stops the project spending on that axis. But do
  not stop after negatives: loop and generate more.
- **Never report a timeout or a crash as agreement or as a pass.** Earlier work here logged
  timed-out comparisons as "skipped" and the clean-looking summary concealed a genuine wrong
  verdict. State which runs completed.
- Where sources do not settle something, write `NOT ESTABLISHED IN SOURCES` rather than guessing.
- Do not modify `src/netreach.cpp` or `src/bisim.cpp` — other work is in flight there. Prototype in
  a scratch file or a copy.

## What success looks like

One algorithm, stated precisely enough to implement, with: the soundness argument; which metric it
moves (explored / stored / both); an expected factor derived from these models; the smallest
experiment that would refute it; and an honest statement of where it measures zero.
