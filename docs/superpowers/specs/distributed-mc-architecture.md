# A multi-core architecture for PTCTLMC, derived from distributed model checking

Status: design document. Written incrementally; sections land in order.
Date: 2026-08-10.

**Scope note / exclusions honoured.** LTSmin and opaal are excluded by instruction and are not
cited as design sources anywhere below (the CNDFS *algorithm* is discussed in §2.4 as an
algorithm, not as a toolset to imitate — flagged explicitly at that point). IC3/PDR, CEGAR,
symmetry reduction and classical partial-order reduction are excluded and are not proposed.
Where a distributed technique tempted me toward something already tried and removed in
`doc_1/sections/10-limits.tex`, I say so out loud in §4.7.

---

## 0. Read this first

Two things in this document outrank the architecture, and both came from running experiments
rather than proposing them (§1.5):

1. **A `p --> q` query spends ~97% of its time in the deadlock pre-pass**, not in anything to do
   with cycle detection. Measured: fischer7, 5.79 s of 5.97 s; fischer8 at 8 threads, 54.82 s of
   56.81 s. The pre-pass explores **14×** the states of the same model's reachability because it
   must disable `a_LU`. **The best available speedup on liveness is not a parallel algorithm.**
2. **The repo has no liveness queries.** All 11 benchmark queries are `A[]` safety, so the
   liveness engine rewritten today is covered by no benchmark and no differential oracle — and
   §4.5 identifies a *possible soundness bug* in its use of subsumption that such an oracle would
   catch.

The architecture below is still worth building, and §6 says in what order. But it is ordered by
measured expected value, not by novelty.

### 0.1 The one-paragraph answer

Distributed model checkers spend almost all of their design budget on three problems that do not
exist in shared memory: **ownership** (who stores a state), **routing** (getting a state to its
owner), and **distributed termination detection** (proving globally that no message is in flight).
PTCTLMC already banked the win from dropping the first two. The deeper lesson of the literature
is not about where states live but about **what threads are allowed to tell each other**, and both
strands converge on one rule: *share only facts that can never be retracted, never share
progress*. Renault, Duret-Lutz, Kordon and Poitrenaud state it as "the shared information is
stable: the fact that two states belong to the same SCC, or that a state is DEAD, will never
change"; PTCTLMC has already proved its own instance of it as **coverage monotonicity**. The
architecture is: **one append-only monotone store, two fixpoints over it, no partition, no
routing, no termination-detection protocol** — with coverage monotonicity applied in the direction
it is sound and deliberately **withheld** in the direction it is not.

That withholding is the load-bearing part. Coverage monotonicity licenses *dropping a state*; it
does **not** license *collapsing a node*, and zone subsumption is provably unsound for Büchi
acceptance. `leads_to_net` collapses nodes today under a **timing** guard ("nothing has looked at
it yet") rather than the literature's **structural** criterion (*no cycle containing both an
accepting node and a subsumption edge*). §4.5 replaces the collapse with tagged subsumption edges
and per-SCC local refinement.

**On the brief's direct question — is a DFS-free OWCTY/MAP approach better here? No, and two of
the four reasons I first wrote down were wrong** (§4.6): OWCTY needs no predecessor edges — that
is what "One-Way" means — and `p --> q` negates to a *persistence* property whose automaton is
**weak**, the exact class on which OWCTY is linear and optimal. The surviving objection is that
OWCTY is not on-the-fly, and on-the-fly early termination is the entire source of this engine's
measured 40×–750× node-count advantage over `verifyta` on liveness. Keep Tarjan; make the *seed
loop* parallel with a shared monotone fact store; and note that it is 3% of the runtime.

---

## 1. The engine as it actually is (baseline for everything below)

Read from the source, not from the write-up, except where the write-up states a measurement.

### 1.1 Reachability — `reach_engine`, `src/netreach.cpp:702`

| aspect | current state |
|---|---|
| search order | level-synchronous parallel BFS, explicit barrier per level |
| shared structure | passed set striped `NB >= max(16384, 64*threads)` ways |
| lock | per-stripe custom 8-byte spin-then-yield reader/writer lock (`detail/spin_rw.hpp`) |
| per discrete state | an **antichain** of zones under `a_LU` (or G-) simulation |
| plus | an **exact convex-union merge** (`dbm_try_union`), which neither UPPAAL nor TChecker has |
| insertion | two-phase: shared-lock `probe_kill` (kills ~83% of attempts), then exclusive `add` |
| routing | **none** — expansion and insertion are fused; no per-level routing phase |
| frontier collect | bucketed by the same hash as the stripe index, parallel, no serial union |
| interner | `ZoneIntern`, 64-way sharded (was a global-mutex bottleneck; fixed) |
| early exit | relaxed atomic flag, authoritative read after the barrier |
| reclamation | per-stripe compaction (amortised) + growth-triggered mark-sweep interner GC |

Soundness of the shared-lock probe rests on **Theorem (Coverage monotonicity)**,
`doc_1/sections/06-theory.tex:53`: for the antichain `A` at a discrete key before an `add` and
`A'` after, `Cov(A) ⊆ Cov(A')`. Its proof rests on the **Replacement Lemma** (`:28`): every
execution of `add` that retires a stored zone `Z` also stores a `z*` with `Z ⊑ z*`. Corollary
(`:86`): a "yes" from `probe_kill` cannot go stale.

### 1.2 Liveness — `leads_to_net`, `src/netreach.cpp:2431`

Rewritten today. On-the-fly nested SCC emptiness for `p --> q`. Structure:

- **Outer discovery** over the full reachable graph: level-synchronous, parallel
  (`#pragma omp parallel` at `:2770`), because a `p ∧ ¬q` state may only be reachable *through*
  `q` states, so the `¬q` analysis alone would never see it.
- **Inner `neg_scc`** (`:2653`): iterative Tarjan restricted to the `¬q` subgraph, **sequential**,
  with an *eager escape* optimisation (the DFS stack is by construction a `¬q` path, so the
  moment any node escapes, every node on the stack escapes with it).
- Three witnesses for "stays in `¬q` forever": (1) no successors / time diverges, (2) a `¬q`
  cycle, (3) an edge into a finished component that already escapes.
- Subsumption **is** used on the liveness graph, and this is load-bearing: exact-hash nodes gave
  `arbiter6` 1 343 096 nodes; plain-inclusion antichains gave 66 522 zones over 256 discrete
  keys; `a_LU` collapses that further. `a_LU` is **gated on deadlock-freedom**
  (`:2507-2508`), because witness (1) is a *per-valuation* property that a simulating zone does
  not preserve. Convex-union merge is **opt-in and off by default** here — measured a net loss
  on half the models (`arbiter6` 66 137 → 75 818, `twolock6` 35 351 → 41 943).
- Results: beats `verifyta` everywhere measured (`arbiter6` 256 nodes vs 10 453; `fischer7`
  11 168 vs 238 617).

### 1.3 The three facts that constrain any proposal

1. **The state space is at a hard floor on five of eight families.** Measured
   (`docs/superpowers/specs/2026-08-09-state-reduction-design.md`): stored zones / discrete
   states = 1.000 on fischer, arbiter, viking, twolock, csma. And
   `docs/superpowers/specs/2026-08-10-w3-partial-order-reduction.md` measures **0.0% independent
   process pairs** on exactly those five families. There is nothing redundant left to remove
   there. A proposal that promises fewer states on those models is wrong before it is written.
2. **Scaling is already 1.62–1.75× from 16 to 32 threads.** Adding threads is not a
   contribution. The contribution has to be in *what is shared* or *what work is avoided*.
3. **All 11 benchmark queries are `A[]` safety.** There are no `E<>`-witness and no liveness
   queries in the suite. This is the single most important fact about the evaluation plan (§6),
   because it means the liveness architecture below is currently **unmeasurable on the repo's own
   suite** and the first experimental obligation is to fix that, not to optimise.

### 1.4 The structural asymmetry nobody has exploited yet

`reach_engine` and `leads_to_net` are two different data structures for the same thing.

| | `reach_engine` | `leads_to_net` |
|---|---|---|
| passed set | `NetPassed`, 16k stripes, `SpinRW` | `std::unordered_map<std::string, std::vector<int>>` |
| key | packed 63-bit `dk`, one word | `std::string` from `disc_key` |
| lock | per-stripe reader/writer | **one `std::mutex gmtx`** (`:2758`) |
| zones | interned, min-graph packed, 4-byte refs | full `SymState` per node, dense `Dbm` |
| retirement | subsume + convex-union merge | subsume only, and **only while untouched** |
| edges | not stored (BFS re-derives) | `adj` adjacency list, explicit |

This asymmetry is the largest single piece of avoidable work in the engine and §4 collapses it.

### 1.5 MEASURED DURING THIS TASK — and it reorders every priority below

I ran the experiments this document proposes rather than only proposing them. Machine: this Mac,
**12 cores**, `build-mostopt/ptctlmc --backend omp`, thread count via `OMP_NUM_THREADS`, wall time
via `/usr/bin/time -p`. Liveness query used: `P0.req --> P0.wait` (holds; `req` has invariant
`x0<=2` and its only outgoing edge goes to `wait`, so it is the exhaustive case).

**Table A — the same model, three query kinds, one thread.**

| model | query | states explored | wall (1 thread) |
|---|---|---|---|
| fischer7 | `A[] not (P1.cs && P2.cs)` | 11 951 | **0.07 s** |
| fischer7 | `E<> deadlock` | 167 805 | **5.79 s** |
| fischer7 | `P0.req --> P0.wait` | 11 438 | **5.97 s** |
| arbiter6 | `A[] not (C1.cs && C2.cs)` | 256 | **0.01 s** |
| arbiter6 | `E<> deadlock` | 30 090 | **1.39 s** |

**Table B — thread scaling.**

| model / query | 1 | 2 | 4 | 8 | 12 | speedup |
|---|---|---|---|---|---|---|
| fischer7 `p --> q` | 5.97 | 3.74 | 2.05 | 1.42 | 1.24 | **4.8× @12** |
| fischer8 `p --> q` | 236.28 | 171.39 | 120.40 | 56.81 | (timed out in harness) | **4.2× @8** |
| fischer7 `A[]` | 0.07 | — | 0.03 | 0.02 | 0.02 | (too fast to read) |
| fischer7 `E<> deadlock` | 5.79 | — | — | — | 1.10 | **5.3× @12** |
| fischer8 `E<> deadlock` | — | — | — | 54.82 (1 858 837 states) | — | — |

**Finding 1 — the deadlock pre-pass is ~97% of a `p --> q` query.** `leads_to_net` calls
`deadlock_reachable` before it starts (`src/netreach.cpp:2507`). On fischer7 that pre-pass alone
is **5.79 s of the query's 5.97 s**; at 12 threads, **1.10 s of 1.24 s**. On fischer8 at 8
threads it is **54.82 s of 56.81 s (96.5%)**. The liveness graph the engine was rewritten to
build costs roughly **0.18 s** on fischer7 — about **3%** of the query.

**Finding 2 — the pre-pass is expensive because `force_exact` disables `a_LU`.** fischer7's
reachability explores **11 951** states; the deadlock query explores **167 805** — a **14×**
blow-up on the identical model, purely from replacing `a_LU` subsumption with exact inclusion
(`E<> deadlock` is a per-valuation property, `03-antichain.tex`). On fischer8 it is
**1 858 837** states. The gate that buys `a_LU` for liveness is paid for with a 14× state space.

**Finding 3 — my own §4.3 premise is both confirmed and irrelevant.** Subtracting Table A/B rows,
the liveness graph builder goes from ≈0.18 s (1 thread) to ≈0.14 s (12 threads) on fischer7 —
roughly **1.3×**, which is very poor and is consistent with the global `std::mutex` diagnosis in
§1.4 (two acquisitions per frontier node, one of them copying a whole `SymState`). But it is
**3% of the query**, so fixing it changes nothing until the pre-pass is fixed. The 4.8× headline
scaling of `p --> q` is almost entirely the **reachability engine** scaling on the pre-pass.

**Caveat.** These subtractions are estimates: `E<> deadlock` as a standalone query and
`deadlock_reachable` as a pre-pass are the same call but not the same process, so the ≈0.18 s
liveness-builder figure carries the noise of two runs. It is directionally unambiguous — 97% vs
3% is not a measurement-error question — but the exact split should be produced with an in-process
timer (experiment E2) before it goes in a paper.

**Consequence for the rest of this document.** Every layer below is re-ranked by this. The
interesting parallel-algorithms question (how to parallelise cycle detection) is a question about
3% of the runtime. The interesting *engineering* question — how to decide deadlock-freedom
without an exact-zone traversal — is 97%. I have kept both, in that order of honesty rather than
that order of glamour.

---

## 2. What the distributed literature actually teaches

### 2.0 The one result that dominates the whole design: subsumption is a *safety-only* licence

Before any distributed algorithm, the constraint that shapes everything.

**Established in sources.** Zone inclusion / subsumption is sound for reachability and **unsound
for Büchi acceptance / liveness**. The reason is structural: a subsumption edge redirects the
search from a zone `Z` to a larger stored zone `Z' ⊇ Z`, so the resulting graph
over-approximates the runs of the automaton; an accepting cycle in that graph need not
correspond to any run of the timed automaton. Stated in the survey by Bouyer, Gastin,
Herbreteau, Sankur and Srivathsan, *Zone-based verification of timed automata: extrapolations,
simulations and what next?* (2022, arXiv:2207.07479 / Springer LNCS 13660): for liveness,
keeping only the zones maximal under inclusion "is no longer sound", and a subsumption edge can
create a Büchi-accepting path with no corresponding run.
<https://arxiv.org/pdf/2207.07479>

The line of work that establishes and repairs this:

| ref | what it establishes |
|---|---|
| Tripakis, *Checking timed Büchi automata emptiness on simulation graphs*, ACM TOCL 10(3), 2009 (earlier Verimag TR-2006-1) | the language of a **strongly non-Zeno** TBA is empty iff the exact simulation graph has no accepting cycle; also for the zone-closed simulation graph. Results obtained by *lifting* cycles to the region-closed graph via commutation properties of the approximation operators. <https://dl.acm.org/doi/10.1145/1507244.1507245> <https://www-verimag.imag.fr/TR/TR-2006-1.pdf> |
| Li (Guangyuan Li), *Checking timed Büchi automata emptiness using LU-abstractions*, FORMATS 2009, LNCS 5813 | extends the above to the coarser LU abstractions. <https://link.springer.com/chapter/10.1007/978-3-642-04368-0_18> |
| Herbreteau, Srivathsan, Walukiewicz, *Efficient emptiness check for timed Büchi automata*, CAV 2010, LNCS 6174, 148–161; journal version *Formal Methods in System Design* 40(2), 2012 | an SCC-based emptiness check with **restricted (liveness-compatible) subsumption**, built by an *iterative* construction: start with a subsumption graph, and when the graph exhibits a cycle that the restriction cannot justify, refine by disabling the offending subsumption edges and rebuild. <https://link.springer.com/chapter/10.1007/978-3-642-14295-6_15> <https://link.springer.com/article/10.1007/s10703-011-0133-1> |
| Herbreteau & Srivathsan, *Efficient on-the-fly emptiness check for timed Büchi automata*, ATVA 2010 | on-the-fly variant; uses the expensive non-Zenoness construction only when required. <https://www.labri.fr/perso/herbrete/publications/HS-ATVA2010.pdf> |
| Herbreteau, Srivathsan, Tran, Walukiewicz, *Why liveness for timed automata is hard, and what we can do about it*, FSTTCS 2016, LIPIcs 65, 48:1–48:14, DOI 10.4230/LIPIcs.FSTTCS.2016.48 | **unless P=NP, liveness is strictly harder than reachability**: a family of automata on which the standard reachability algorithm is in P but liveness is NP-hard. |

> The exact wording of the counterexample in the FSTTCS 2016 and FMSD 2012 PDFs could not be
> extracted through the fetch tool (compressed streams). The *statements* above are corroborated
> by the survey and by the CAV 2013 paper below; the verbatim counterexample is
> **NOT ESTABLISHED IN SOURCES** at this level of detail and should be read in the PDFs before
> the theorem in §4.5 is written up.

**This is the pivot of the whole document.** PTCTLMC's `leads_to_net` currently *does* use
subsumption on the liveness graph and *does* get away with it — but for a reason that is
specific to this engine and is written down at `src/netreach.cpp:2477-2505`:

- `DiscPred` **forbids clock constraints**, so `p` and `q` are functions of the discrete key
  alone and never of the zone. Hence if `Z_u ⊆ Z_v` at the same discrete key, every run of `u`
  is a run of `v` over the same sequence of discrete states.
- The engine is not checking general Büchi acceptance. It is checking `p --> q`, which fails iff
  some reachable `p ∧ ¬q` state can stay in `¬q` forever — and "forever" is witnessed by a
  `¬q` **cycle**, by **deadlock**, or by **time divergence**. Cycles and divergence transfer
  upward along a simulation; **deadlock does not**, which is exactly why `a_LU` is gated on
  deadlock-freedom at `:2507`.
- The engine is checking an **over-approximating** direction with an **existential** witness.
  Subsumption over-approximates runs; over-approximating runs can only *create* spurious
  witnesses, never destroy real ones. So the engine's use of subsumption is sound in the
  direction "no witness found ⇒ property holds"... **and this is precisely where the danger
  lies**: a spurious `¬q` cycle created by a subsumption edge yields a spurious *violation*.

**Consequence I want on the record, because §4 depends on it and the current code does not
state it in this form:** the argument at `:2477` gives soundness of *dropping* `u` when
`Z_u ⊆ Z_v` (no witness is lost). It does **not**, by itself, establish that a cycle found in
the graph-with-subsumption-edges is a real cycle. The code sidesteps this by never adding a
subsumption *edge*: it **retires `u` into `v` via union-find `fwd`** and re-resolves every edge
(`resolve` at `:2543`), i.e. it *merges nodes*, which is a quotient, not an edge redirection.
Whether the quotient can create a cycle that no run realises is the obligation that §4.5 has to
discharge, and it is the same obligation Laarman et al. discharge in the paper below. **I have
not verified that the existing code discharges it**; see §6, experiment E0, which is the
cheapest thing in this document and the one I would run first.

### 2.1 The closest published prior art — and why the proposal must differ from it

Laarman, Olesen, Dalsgaard, Larsen, van de Pol, *Multi-core Emptiness Checking of Timed Büchi
Automata Using Inclusion Abstraction*, CAV 2013, LNCS 8044, 968–983, DOI
10.1007/978-3-642-39799-8_69. <https://link.springer.com/chapter/10.1007/978-3-642-39799-8_69>

They: show Büchi emptiness is **not** preserved under inclusion abstraction but that *some other
structural properties are*; on the strength of those, give a variant of **NDFS** that exploits
subsumption, and extend the multi-core **CNDFS** algorithm with subsumption — "the first
parallel LTL model checking algorithm for timed automata". Measured: subsumption halves the
states on an FDDI case study; speedups up to **40 on 48 cores**.

I am citing this as **prior art to differentiate from, not as a design source**, and I flag it
explicitly because it is implemented in the excluded toolset. Two things follow, and they are
the reason the architecture in §4 is shaped the way it is:

1. **The published parallel-timed-liveness point in the design space is DFS-based.** It is
   NDFS/CNDFS + subsumption. So "parallel NDFS with subsumption" is taken, is excluded, and is
   not available as a contribution.
2. **The DFS-free point is, as far as these searches go, open for timed automata.** I searched
   for OWCTY / MAP / parallel-SCC applied to timed zone graphs and found none. A parallel
   SCC-based emptiness check with a *shared* union-find exists for untimed generalized Büchi
   automata — Renault, Duret-Lutz, Kordon, Poitrenaud, *Variations on parallel explicit
   emptiness checks for generalized Büchi automata*, STTT 18(6), 2016 — and multi-core on-the-fly
   SCC decomposition exists (Bloemen, Laarman, van de Pol, PPoPP 2016), but **the combination
   "DFS-free parallel liveness + zone subsumption + antichain" is NOT ESTABLISHED IN SOURCES as
   existing**. That is the gap §4 aims at.
   <https://link.springer.com/article/10.1007/s10009-016-0422-5>
   <https://dl.acm.org/doi/10.1145/2851141.2851161>

### 2.1a The exact soundness criterion, and it is checkable

From the CAV 2013 line, the criterion that makes a subsumption graph usable for liveness:

> A subsumption graph is **liveness compatible** iff it contains **no cycle that has both an
> accepting node and a subsumption edge.**

and the concrete NDFS rule they derive from it: prune the outer (blue) search when it reaches a
state subsumed by a state on which the inner (red) search has been called. Both their algorithm
and Herbreteau–Srivathsan–Walukiewicz's iterative construction compute liveness-compatible
subsumption graphs; they differ in *how* they get there (a stack-discipline restriction vs an
iterative refinement that disables offending edges and rebuilds).
<https://link.springer.com/chapter/10.1007/978-3-642-39799-8_69>

**This criterion is the single most useful sentence in this entire document for PTCTLMC**,
because `leads_to_net` currently satisfies *no stated criterion at all*. Its restriction is
`untouched = !expanded[j] && tidx[j] == -1` (`src/netreach.cpp:2570`) — a **timing** condition
("nothing has looked at it yet"), not a **structural** one. See §4.5 and experiment E0.

### 2.2 The Renault et al. parallel SCC emptiness check — the shape to copy

Renault, Duret-Lutz, Kordon, Poitrenaud, *Parallel Explicit Model Checking for Generalized Büchi
Automata*, TACAS 2015, LNCS 9035; journal version *Variations on parallel explicit emptiness
checks for generalized Büchi automata*, STTT 19(6), 2017, 653–673.
<https://www.lrde.epita.fr/dload/papers/renault.15.tacas.pdf>
<https://link.springer.com/article/10.1007/s10009-016-0422-5>

Read directly from the paper (pp. 2, 4, 5, 6):

- **What is shared.** A lock-free **union-find over `Q ∪ {Dead}`**, whose classes are labelled
  with the acceptance marks occurring in the SCC. Methods `make_set`, `contains`,
  `unite(s1, s2, acc)`, `same_set`. `unite` returns the acceptance-mark set of the resulting
  class, and returns `∅` if the class contains `Dead`. **An accepting cycle is reported as soon
  as `unite` returns `F`** (the full acceptance set). Implemented thread-safe with
  compare-and-swap over linked lists plus a hash table, following Anderson & Woll.
- **What is NOT shared.** Everything about a thread's *progress*: the `dfs` stack, the `live`
  stack, the `livenum` map, the `pstack`. Each thread runs its **own complete DFS** over the
  whole automaton, in a **randomised successor order** (swarming).
- **Why no synchronisation points and no repair procedures are needed** — the paper's own
  sentence, and this is the load-bearing one:

  > "threads only share structural information about the automaton of the form *'states x and y
  > are in the same SCC'* or *'state x cannot be part of a counterexample'*. Since threads do not
  > share any information about the progress of their search, we can actually mix threads with
  > different strategies in the same emptiness check."

  and

  > "Note that the shared information is stable: the fact that two states belong to the same SCC,
  > or that a state is DEAD, will never change over the execution of the algorithm. All threads
  > may therefore reuse this information freely to accelerate their exploration, and to find
  > accepting cycles collaboratively."

- **Contrast with the NDFS/CNDFS family**, stated by the same paper: parallel NDFS shares
  progress, and therefore needs *synchronisation points* (a state handled by several threads has
  its status updated only after all threads have finished) or *a posteriori repair procedures*.
  "So far, attempts to design scalable parallel DFS-based emptiness check that does not require
  such mechanisms have failed."
- Two strategies on the same canvas — **Tarjan** (publishes a `unite` on every lowlink update:
  number of `unite`s = transitions in each SCC, so it shares *more*) and **Dijkstra**
  (`unite` only when a potential root is invalidated: exactly `n-1` unites for an SCC of `n`
  states, so it shares *less* but works less). They can be **mixed in one run** precisely
  because no progress is shared.
- Reported: "the new emptiness checks have better speedup than existing algorithms in half of
  our experiments." Exact core counts and factors: **NOT ESTABLISHED IN SOURCES** at the level of
  detail fetched here (the TACAS PDF's evaluation section was not read).

**Why this matters more than OWCTY for PTCTLMC:** the shared-information invariant here —
*stable, monotone facts, never progress* — is **the same invariant PTCTLMC already proved for
its passed set**. Coverage monotonicity (`Cov(A) ⊆ Cov(A')`) says exactly "the fact that `z` is
covered is stable". Renault et al. say "the fact that `x` is DEAD is stable". These are two
instances of one discipline, and §4 makes that discipline the architecture.

### 2.3 ZEUS — distributed backward zone reachability, and the end of that road

Braberman, Olivero, Schapachnik. Main citations:

- *Zeus: A Distributed Timed Model-Checker Based on Kronos*, **PDMC 2002** (workshop affiliated
  with CONCUR 2002, Brno), ENTCS 68(4):503–522.
  <https://doi.org/10.1016/S1571-0661(05)80389-5>
- *Issues in distributed timed model checking — Building Zeus*, **STTT 7(1):4–18, 2005**.
  <https://doi.org/10.1007/s10009-004-0143-z>
- *On-the-fly Workload Prediction and Redistribution in the Distributed Timed Model Checker Zeus*,
  **PDMC 2004**, ENTCS 128(3):3–18. <https://doi.org/10.1016/j.entcs.2004.10.015>
- *Dealing with practical limitations of distributed timed model checking for timed automata*,
  **FMSD 29(2):197–214, 2006** — the journal culmination.
  <https://doi.org/10.1007/s10703-006-0012-3>
- Schapachnik, PhD thesis, Univ. de Buenos Aires, 2007, 146 pp, in English.
  <https://bibliotecadigital.exactas.uba.ar/download/tesis/tesis_n4125_Schapachnik.pdf>

**What it computes.** A distributed **backward** (predecessor) zone fixpoint over an *explicit,
pre-composed* global location graph, on Kronos DBM libraries: initialise `R_s = I(s)` at target
locations, iterate `PredE = ⋃ pred_e(ΔR_t, ψ(e), α(e))`, `PredT = pred_τ(PredE)`,
`ΔR_s = PredT − R_s`, until nothing changes; answer = is an initial state in `R`.

**Why backward — and it is not the reason one would guess.** From the FMSD 2006 abstract:

> "Although forward is better suited for on-the-fly construction of the model, the one known as
> backwards provides the basis for the verification of arbitrary formulae of the TCTL logic, and
> more importantly, for controller synthesis."

So: **inheritance from Kronos (a backward TCTL checker) plus full TCTL and controller synthesis
— not partitioning, not target-drivenness, not any passed-set argument.** The authors say
partitioning is *harder* under backward, and the 2007 thesis goes on to build a distributed
*forward* Zeus as well. This matters for PTCTLMC because PTCTLMC *is* a PTCTL checker; the one
argument that would justify backward here is the TCTL-completeness one, and it is orthogonal to
parallelism.

**Partitioning — explicitly NOT hash.** ENTCS 2002 §3: "While most work in the area uses an
on-the-fly construction of the control graph and a hashing function for its nodes… **we introduce
an a priori control graph partitioning which in the first version is based on the tool METIS**."
The unit is a *global composed location*; all DBMs of a location live on one node. Strategies:
METIS min-cut; target-location-weighted min-cut; oracle-weighted (DBM ops measured in a prior
sequential run); random/round-robin; and the winning **observer-induced** heuristic (spread the
heavy observer locations). Dynamic load balancing was added in PDMC 2004 via **ParMETIS**
repartitioning driven by an online workload-prediction memo table (80–90% hit rate), later
restricted to migrating only the heaviest locations because naive migration was catastrophic
(Conveyor4AB: 1627 s → **3859 s** with naive migration).

**Termination detection.** No named classical algorithm for the backward engine — a **centralised
coordinator**: "either a capsule finds an initial location, or at the end of an iteration all of
them have reported no changes (ΔR = ∅)". Correctness via **Cousot's asynchronous iterations**
(1978) over the lattice of regions ordered by inclusion, with `pred_τ`/`pred_e` monotone. The
*forward* Zeus in the thesis uses classical pairwise message counting.

**Measured — and the negative results are the valuable part.**
- ENTCS 2002, TGC: **superlinear** speedups on reachable instances (S₂ = 7.64, S₆ = 10.78) and
  **catastrophic slowdowns** on unreachable ones (S₂ = **0.17**, 889 s → 5155 s). Random
  distribution "performed extremely bad".
- Root cause: **zone-set fragmentation.** Regions (unions of DBMs) have no canonical form, so
  changing evaluation order changes the *syntactic* zone count; region differentiation is O(n·m)
  in the zone counts. RCS5 unreachable: ≈**697 M** operations sequential vs ≈**1.3 B** on two
  processors asynchronously.
- Consequence: they abandoned asynchrony for a **synchronous** engine that "mimics the operations
  that Kronos would perform on a monoprocessor" — STTT 2005 abstract calls this outcome
  "surprising … although being intuitively less attractive".
- The final ceiling is a Pareto workload: MinePump has **5 heavy locations out of 4452** carrying
  92% of the work; Conveyor6A **2 of 1344** carrying 88%; RS-C 4 of 25 975 carrying 67%. And the
  heavy set *changes every iteration*. Their `%waste` (idle time) metric reaches **80–85%**.
- **Their verdict, verbatim** (FMSD 2006 §6): "some locations monopolize the total workload,
  meaning that **close-to-linear speedups are not feasible without splitting locations among
  processors**, which in turn seems impractical for the backwards calculus in a distributed
  environment (i.e., no shared-memory). **This point signal a quasi-end of the road for the
  distribution of the backwards algorithm.**"

**Why backward did not become standard** — four reasons, from primary sources:
1. **Discrete/data variables are not invertible.** Bouyer, *An introduction to timed automata*
   (MOVEP'14): "**However the backward computation is not appropriate to manipulate other
   variables (think for instance of assignment `i := j.k + l`)**". The FORMATS 2022 survey says
   the same in print. Farkas, Tóth, Hajdu, Vörös, *Backward Reachability Analysis for Timed
   Automata with Data Variables*, AVoCS 2018 / ECEASST 76 (2019) is the paper about this obstacle:
   "**calculating predecessor states for systems with data variables is computationally expensive,
   prohibiting the successful application of this approach so far**"; and "**Kronos does not
   support data variables.**" <https://hajduakos.github.io/publications/avocs2018.pdf>
2. **No on-the-fly composition and no early termination** — backward needs the whole global
   control graph materialised.
3. **Not an abstraction problem.** Contrary to the natural guess, backward *terminates natively*
   and needs no extrapolation at all (the region abstraction is a bisimulation and every backward
   set is a finite union of regions). The `LU`/simulation programme is a *forward-only* research
   line with ~25 years of compounding investment; that asymmetry is part of the answer but is not
   a correctness objection.
4. **Diagonal constraints: backward is BETTER.** Farkas et al.: "for … timed automata with
   diagonal constraints **backward exploration is the only zone-based method that does not
   introduce an additional exponential factor**", citing Bouyer, *Untameable Timed Automata!*
   (STACS 2003) and Bouyer–Laroussinie–Reynier (FORMATS 2005).

**NOT ESTABLISHED IN SOURCES:** any published claim that urgent channels or committed locations
specifically obstruct backward analysis.

**Transfer to shared memory:** the one thing that transfers is the diagnosis, not the design.
Zeus's ceiling was "a few locations monopolise the work and we cannot split a location across
nodes **because there is no shared memory**". PTCTLMC *has* shared memory, and its analogous
adversarial case is named in `10-limits.tex`: the `single` family, "a model whose reachable
states all share one discrete state has one stripe, hence one lock and one antichain, and does
not parallelise". Zeus's verdict is, almost word for word, PTCTLMC's stated limit — and Zeus
says the fix requires shared memory, which PTCTLMC has and does not yet use for this. That is a
real, specific opportunity (splitting *within* an antichain), and `10-limits.tex` already notes
it "would need a different concurrency argument than Thm. coverage, because the merge in step (4)
is global to the chain". I am **not** proposing it here — it is a separate design problem — but
the Zeus literature is the strongest argument that it is the right next adversarial case.

### 2.4 Distributed UPPAAL (Behrmann, Hune, Vaandrager) — the search-order result

**Behrmann, Hune, Vaandrager, *Distributing Timed Model Checking — How the Search Order Matters*,
CAV 2000, LNCS 1855, 216–231.** <https://doi.org/10.1007/10722167_19>

MPI, distributed memory, following Stern & Dill. Passed and waiting lists partitioned by a
distribution function `h`. Their finding, verbatim:

> "A key problem we had to face in our work is that **the number of symbolic states that has to
> be explored depends on the order in which the state exploration proceeds. In particular, the
> number of states tends to grow if state space exploration is parallelized.**"

and the mechanism, which is *exactly* PTCTLMC's antichain:

> "consider two states (l, D) and (l, D′) with same location vector l but different time zones
> satisfying D ⊂ D′. **If (l, D) arrives first and is explored before arrival of (l, D′), then
> (l, D′) will also be explored** … **However, if (l, D′) arrives first … then (l, D) will not be
> explored** because it is covered by a state in the passed list."

Measured: DACAPO 45 001 states on 1 node → **110 634 on 17 nodes**. Their fix is a **waiting list
ordered by depth** (an approximation of BFS), which held DACAPO to 45k → 52k and in some cases
*reduced* counts below the sequential baseline. Distributed DFS and reverse-priority both made it
much worse and were dropped.

**A constraint that is pure distributed-memory and disappears here:** "to make the inclusion
checks of the time zones in the waiting and the passed lists possible, **states with the same
location vector must be mapped to the same node**. The hash value of a symbolic state is
therefore only based on the location vector." They confirm the consequence experimentally:
"**None of the D1 experiments** [hashing the complete state including the zone] **terminated
within a reasonable time frame**". PTCTLMC hashes on the packed discrete key for the same reason,
but for it this is a *cache-locality* choice, not a correctness constraint — any thread can read
any antichain.

**Follow-up: Behrmann, *A Performance Study of Distributed Timed Automata Reachability Analysis*,
PDMC 2002, ENTCS 68(4):486–502; and *Distributed reachability analysis in timed automata*,
STTT 7(1):19–30, 2005** (single author). <https://doi.org/10.1007/s10009-003-0111-z>
Abstract: "**The load balancing problems are caused by the symbolic nature of the representation
of the states of a timed automaton**", fixed by a proportional controller plus locality and
compression, reaching "**speedups between 50% and 90% of linear**" on 14 nodes.

The transferable mechanism, and it is a good one:

> "the waiting list does not have O(1) insertion time… If the load on node A becomes slightly
> higher compared to node B, more time is spent inserting states… the **exploration rate of A
> drops** … **Thus a slight difference in the load of A and B causes the difference to increase,
> resulting in an unstable system where the load of one or more nodes quickly drops to zero.**"

This is a *positive-feedback* load instability driven by a non-O(1) insertion, not a static
partitioning problem. PTCTLMC's analogue is precise and worth watching: an antichain whose chain
length grows makes `add` slower at exactly the stripe that is hottest. The convex-union merge
holding chains at ≈1 zone per discrete key is what prevents it — which is a second, independent
argument for `09-novelty.tex` N2's claim that merging is a *scalability* mechanism.

**One critique worth knowing, from Schapachnik's 2007 thesis Ch. 6**, aimed at distributed
UPPAAL: redirected states are *not* checked against the Visited set, so
"**is their speedup a result of distributed processing, or … the fact that their proportional
redirector is having the effect of avoiding inclusion checks that, statistically were going to
fail anyway?**" He backs it with measured success rates of inclusion checks: RCS5 **0.07%**,
MinePump **0.01%**, Conv6ABC 0.48% — but FDDI4 **46.75%**. His "interesting sequel" is: what
happens if a *sequential* checker randomly skips inclusion checks? **For PTCTLMC this question is
already answered and answered the other way:** `10-limits.tex` measures that removing the shared
coverage probe (per-thread local antichains) is 1.29–1.92× *worse*, because the probe answers 83%
of successors. PTCTLMC's probe succeeds an order of magnitude more often than Zeus's or
UPPAAL's, so the trade Schapachnik proposes is a clear loss here. Worth stating in the paper —
it closes an obvious reviewer suggestion with a number.

### 2.5 Distributed Murphi — the original partition-by-hash, and what it actually measured

Stern & Dill, *Parallelizing the Murφ Verifier*, **CAV 1997, LNCS 1254, 256–267**; journal version
**FMSD 18(2):117–129, 2001**. <https://doi.org/10.1007/3-540-63166-6_26>
<https://doi.org/10.1023/A:1008771324652>
(Stern's dissertation is **TU München, Dr.-Ing., 1997**, not Stanford — the work was done at
Stanford in Dill's group with Dill as third examiner.)

- **Partition.** The state table is partitioned across N nodes; each node has a local FIFO queue.
  `Send(s)` canonicalises `s` (symmetry reduction) then computes the owner as `h(s_c)` and sends
  one **active message** `Receive(s_c)`. **Universal hashing** (Carter & Wegman 1979). No explicit
  balancer at all: they call it **"randomized load balancing"**, and quantify it —
  `Pr(|Y/Ȳ − 1| > r) < 2φ(z)/z` with `z = r·√(n/(N−1))`; for n = 10⁸, N = 32, P(relative error
  > 0.5%) < 2.73×10⁻¹⁹.
- **Termination detection — not Dijkstra–Scholten, not Safra.** A master-coordinated
  **send-freezing message count**: the master, after being idle past a threshold (~0.1 s),
  broadcasts `ReportCounters()`; each node sets `StopSend := true` (blocking further sends) and
  replies `Sent − Received + |Q|`; terminate iff the sum is 0. The correctness argument is
  inherited from an unpublished Berkeley distributed-task-queue proof (C.-P. Wen), not from the
  classical literature.
- **The (n−1)/n fact, verified verbatim.** Their runtime model is
  `t_N = 4·o·m·(N−1)/N² + t₁/N`, derived by assuming "on average, a fraction **(N−1)/N** of
  [messages] go to nodes other than the sender". So essentially *every transition sends a
  message*.
- **What the measurements actually say about communication.** Using a LogGP-parameterised NOW:
  **insensitive to latency L and to gap g** (degrading either by +100 µs was negligible),
  **insensitive to bandwidth** (per-node requirement < 1 MB/s), and **sensitive only to
  per-message overhead o**. Speedups: **44.2× (SCI) and 53.7× (DASH) on 63 SP2 nodes**;
  26.6–29.4× on 32 NOW nodes. Message aggregation at 10 states/message cut messages ~7× and
  halved runtime under high overhead.

**The partition-by-hash "conclusion" is workload-dependent, and this is the part usually
misremembered.** Lerda & Sisto, *Distributed-Memory Model Checking with SPIN*, SPIN 1999,
LNCS 1680, 22–39, measured hash partitioning **2.6–11× slower** than a structural
single-component partition on Bakery/Leader/Philo, despite hash giving much better balance
(partition rate 0.987 vs 0.775 on Bakery, but 883 297 messages vs 38 874, and 113 s vs 10 s).
Their single-component partition has average cross-transition fraction `d·k/P < k/P` against
hash's `(N−1)/N`. Their own explanation: in Murφ successor generation is expensive so balance
dominates; in SPIN it is cheap so cross-edges dominate.
<https://doi.org/10.1007/3-540-48234-2_3>

**Where PTCTLMC sits on that axis, and it is decisive.** Timed successor generation is
*expensive* — delay, constrain, reset, canonicalise, extrapolate, per outgoing edge, all
O(dim³) DBM work — so PTCTLMC is firmly in the Murφ regime where balance beats locality. But the
whole axis is a distributed-memory axis: with a shared heap there are **no** cross-partition
messages to trade against balance, so PTCTLMC gets Murφ's balance *and* SPIN's locality by
choosing a shared table. `04-parallel.tex` already made this choice; the literature above is the
justification it does not currently cite.

**PReach** (Bingham et al., Intel/UBC), *Industrial Strength Distributed Explicit State Model
Checking*, PDMC-HiBi 2010, 28–36 — the modern Stern–Dill descendant. Its termination detection is
Stern–Dill's. Three additions, and one correction to the common recollection: its load balancing
is **not work stealing and not random redistribution** — it is *push-based*, piggybacking queue
length on outgoing state messages and pushing a chunk to any peer whose queue is `LBFactor = 5`
times smaller. **Crediting** *is* real credit-based flow control (C credits per peer, restored by
acknowledgement), and it was decisive: LDASH on 8 workers finished in ~600 s with crediting and
timed out at 21 600 s without. Batching (1000 states or 1 s) gave 10–20×. "Network bandwidth was
never a performance bottleneck" — the same conclusion Stern & Dill reached 13 years earlier.
<https://www.cs.ubc.ca/~depaulfm/Papers/pdmc10-camera-ready.pdf>

**Transfer to shared memory: almost nothing, and that is the point.** Hash partitioning,
crediting, batching, push-balancing and message-count termination are all answers to *messages*.
Under a shared heap every one of them evaporates. What survives is (i) the empirical claim that
excess parallelism per BFS level is ample (Stern & Dill measured 20k–80k states per level), which
directly supports keeping level-synchronous BFS, and (ii) the Lerda–Sisto axis, which tells you
*why* a shared table is the right call for an engine with expensive successor generation.

### 2.6 The DFS-free line (OWCTY, MAP) and parallel SCC

The organising fact, from the survey chapter — Barnat, Bloemen, Duret-Lutz, Laarman, Petrucci,
van de Pol, Renault, *Parallel Model Checking Algorithms for Linear-Time Temporal Logic*, Ch. 12
in *Handbook of Parallel Constraint Reasoning*, Springer 2018, 457–507
(<https://ris.utwente.nl/ws/files/50528000/handbookLTL.pdf>):

- The taxonomy splits **multi-core/shared-memory, DFS-based** from **distributed/message-passing,
  BFS-based**, and the organising argument is complexity-theoretic: **lexicographic DFS is
  P-complete (Reif)**, hence "inherently sequential"; **parallel BFS and BFS-based SCC
  decomposition are in NC**.
- They enumerate **three positions**, and this is the useful framing for §4.6:
  1. **ignore the theory and use randomised parallel DFS anyway** (justified because processors ≪
     graph size);
  2. **take it seriously and use BFS-based fixpoints** (OWCTY, MAP, OWCTY+MAP) with worse-than-
     linear worst case but good scalability;
  3. **circumvent it via *free* (non-lexicographic) DFS**, which is in NC for planar graphs and
     RNC in general.
- Their summary table classifies algorithms by complexity, scalability, **optimality**,
  **on-the-fly capability**, and TGBA support, and the headline trade-off is: *DFS algorithms are
  runtime-optimal but not necessarily scalable; BFS algorithms sacrifice optimality for
  scalability, except for weak automata where OWCTY is also optimal.*
- For the distributed setting they name the mechanism: static **hash-based partitioning** with an
  **owner/partition function**, and note the fundamental problem that traversing an edge usually
  requires communicating with the machine holding the target, "leading to a dramatic communication
  overhead".

**This is the sentence that decides §4.6 for PTCTLMC: OWCTY trades optimality for scalability,
and PTCTLMC's measured liveness advantage is entirely an optimality (on-the-fly early
termination) advantage.** Trading it away to gain scalability on a phase that is measured below
at ~3% of the query would be a bad trade by two independent arguments.

#### 2.6.1 OWCTY — and a correction to a claim I made before reading it properly

Origin (symbolic): Fisler, Fraer, Kamhi, Vardi, Yang, *Is There a Best Symbolic Cycle-Detection
Algorithm?*, TACAS 2001, LNCS 2031, 420–434. Explicit/distributed: **Černá & Pelánek,
*Distributed Explicit Fair Cycle Detection (Set Based Approach)*, SPIN 2003, LNCS 2648, 49–73**
(tech report FIMU-RS-2002-09, <https://www.fi.muni.cz/reports/files/2002/FIMU-RS-2002-09.pdf>).

The fixpoint, verbatim from the report:

```
Reachability(S) = μZ.(S ∪ image(Z))
Elimination(S)  = νZ.(S ∩ image(Z))

proc OWCTY(G, F, init):
  S := Reachability(init)
  repeat until S unchanged:
      S := Reachability(S ∩ F)      // strip leading SCCs with no accepting state
      S := Elimination(S)           // strip leading trivial SCCs
  return S ≠ ∅
```

`Elimination(S)` is "successive removal of states that do not have predecessors in S", i.e. a
topological peel.

> **Correction to my §4.6, reason 2.** I wrote that OWCTY's elimination step is backward and that
> PTCTLMC would need to store predecessor edges or compute zone pre-images. **That is wrong, and
> it is exactly what the name means.** From the handbook chapter: "the algorithm maintains an
> integer value associated with every vertex to keep the number of not-yet-removed direct
> predecessors, the so called **indegree**. The unique feature of the Owcty algorithm is that the
> indegrees are updated **without the need to enumerate the predecessors of a state**. In fact,
> the algorithm only performs **forward** traversal procedures… **This is exactly what the
> One-Way in the name of the algorithm stands for.**" So OWCTY works on an implicitly generated,
> `next-state`-only state space. The Forward–Backward family (FB, OBF) *does* need reverse edges;
> OWCTY does not. §4.6 is corrected below.

**Complexity.** `O(h·(n+m))` explicit, where **h is the height of the SCC quotient graph** —
because each external iteration removes all initial unfair components. Handbook Table 12.1 gives
the coarser `O(V·(V+E))` for general Büchi, **`O(V+E)` and optimal for *weak* Büchi**.

**Why OWCTY and not Emerson–Lei**, in the authors' own words, and this is a genuinely
transferable design principle: "The need of synchronisation after each procedure is the reason
why we have chosen the One Way Catch Them Young algorithm as a base… **The number of external
iterations determines the number of necessary synchronisations.**"

**Partition / termination.** Static hash partitioning by an `Owner` function; a designated
**Manager** performs a two-phase quiescence check (all workers idle **and** no pending messages)
followed by a global barrier, once per phase.

**Measured (Černá & Pelánek §6), and this is the number that matters here:** with a fair cycle
present, sequential NestedDFS is *dramatically* faster because it is on-the-fly — Ring2:
**NDFS 12.82 s vs DETECT-CYCLE 305.51 s**. Counterexamples, though, are far shorter under BFS
(Ring2: NDFS 2754+363 states vs 52+14).

#### 2.6.2 MAP

**Brim, Černá, Moravec, Šimša, *Accepting Predecessors Are Better than Back Edges in Distributed
LTL Model-Checking*, FMCAD 2004, LNCS 3312, 352–366.**
<https://paradise.fi.muni.cz/pub/pdf/fmcad04.pdf>

`map(v)` = the maximal (in a fixed vertex order) accepting predecessor of `v`.
**Lemma 1:** `map(v) = v` ⇒ an accepting cycle exists. **Corollary 1:** `map(u) ≠ map(v)` ⇒ `u`
and `v` are not on a common cycle — which partitions the graph into independently explorable
subgraphs. The converse of Lemma 1 fails, so a **deleting transformation** removes accepting
states that are provably not on any cycle and the process iterates (Theorem 2).

Propagation "resembles the relaxation procedure as used in the single source shortest path
problem… **relaxations can be performed in an arbitrary order in a BFS manner, hence in
parallel**." **Complexity `O(a²·m)`** (a = accepting vertices), reported elsewhere as
`O(V²·(V+E))`.

**Ordering is the catch**, and the negative result is sharp: "for every graph an optimal ordering
exists, however, **to find it is as difficult as to define a DFS postorder**, which is hard to
parallelize."

**MAP vs OWCTY, stated by the handbook chapter:** "Algorithm Map works on the fly… **This is not
the case with algorithm Owcty**, as to properly compute the indegrees, the whole graph has to be
traversed. On the other hand, the time complexity of Owcty is quadratic, while Map is cubic." So
**MAP for bug-hunting, OWCTY for full verification**, and DiVinE ships the combination
`OTF_OWCTY`. Independent confirmation from the other camp (Laarman & van de Pol, PDMC 2011):
"For systems with bugs… ENDFS and LNDFS outperform OWCTY-MAP… On examples without bugs,
OWCTY-MAP beats both."

#### 2.6.3 Why NDFS does not parallelise — and the caveat nobody quotes

Reif, *Depth-first search is inherently sequential*, **IPL 20:229–234, 1985**. The handbook's
statement of what he actually proved: "**Reif actually showed that *lexicographic* DFS is
P-complete** … given a graph and **a fixed ordering of transitions from each state**, there is
probably no parallel algorithm to even check whether node x will be visited before node y in the
DFS post-order, observing the fixed transition ordering."

**The caveat, from Bloemen's thesis, and it matters for §4.6:** "none of the algorithms mentioned
… require a fixed order on outgoing edges of the states … showing that the **oft-cited
theoretical result from Reif does not apply directly**. In fact, it might even be the case that
the more general problem of non-lexicographic DFS is in NC." *Free* DFS is in NC for planar
graphs (He & Yesha 1988) and in RNC in general (Aggarwal, Anderson, Kao 1990).

**So "Tarjan is inherently sequential" is, strictly, not what the theorem says.** What is
sequential is *lexicographic* DFS. Every practical parallel scheme in this literature — CNDFS,
Renault et al., UFSCC — exploits precisely that gap by randomising the successor order per worker
(swarming) and sharing only order-independent facts.

#### 2.6.4 Multi-core SCC: UFSCC, and where the shared structures actually break

Bloemen, Laarman, van de Pol, *Multi-Core On-The-Fly SCC Decomposition*, **PPoPP 2016**, 8:1–8:12.
<https://dl.acm.org/doi/10.1145/2851141.2851161> (full algorithm + proofs also in Bloemen's
Twente PhD thesis, <https://research.utwente.nl/files/122499728/thesis.pdf>). Applied to LTL in
Bloemen & van de Pol, *Multi-core SCC-Based LTL Model Checking*, **HVC 2016**, LNCS 10028, 18–33.

Framing: "The main advantages of Tarjan's SCC algorithm are its linear time complexity and
ability to return SCCs **on-the-fly**… **Until now, most parallel SCC algorithms sacrifice both:
they run in quadratic worst-case time and/or require the full graph in advance.**"

Mechanism: an **iterable concurrent union-find** — parent pointer + a **P-bit worker set** + a
`next` pointer forming a cyclic list + status flags. A worker that finds its own bit already set
on a partial SCC's root has found a cycle through another worker's work. `Unite` ORs the worker
sets onto the new root, and **the new root is always the higher array index**, without which
`Find` can loop forever. Done states are lazily removed from the cyclic list. Worst-case work
`O(P·(n+m))`, runtime `O(n+m)`, memory `O(P·n)`.

Measured: 64 cores, 62 BEEM graphs — **4.1× to 35.5× over sequential Tarjan, geometric mean
14.84×**; **6.42× geometric mean over Renault et al.**; **10–30× faster than Tarjan for graphs
containing a large SCC**, where prior work showed no speedup at all. As an LTL emptiness check
(HVC 2016), acceptance marks live in the union-find root and are updated atomically inside the
brief `Unite` lock; for state-based Büchi "an accepting cycle is detected as soon as an accepting
state is united with another state".

**The limitation that is directly relevant to PTCTLMC:** both UFSCC (contention on the cyclic
list of Busy states) and Renault et al. (contention on `unite`) **degrade specifically on graphs
with few large SCCs** — Renault et al. "even performs worse than Tarjan's sequential algorithm"
there. PTCTLMC's adversarial family is `single`: one discrete state, thousands of zones. If the
liveness graph of such a model is one giant SCC, this is the same wall, reached from the same
direction.

#### 2.6.5 Termination detection: what survives the port to shared memory

Classics: **Dijkstra & Scholten**, *Termination detection for diffusing computations*, IPL
11(1):1–4, 1980 (spanning tree of activation edges + per-node deficit). **Safra's algorithm**,
Dijkstra EWD998, 1987 (token ring, white/black colouring, per-node send−receive counter).
**Mattern**, *Algorithms for distributed termination detection*, Distributed Computing 2:161–175,
1987 (four-counter, time algorithm, vector counters) and *Global quiescence detection based on
credit distribution and recovery*, IPL 30(4):195–200, 1989.
<https://vs.inf.ethz.ch/publ/papers/mattern-dc-1987.pdf>
<https://vs.inf.ethz.ch/publ/papers/Credit_TermDet.pdf>

Every one of these exists to construct, by circulating information, a consistent cut proving that
the deficit `D(t) = Sent(t) − Received(t)` is zero and all processes are passive. **Under
cache-coherent shared memory that quantity is a single atomic integer.** Tokens, colours, waves
and credit shares are elaborate substitutes for a shared variable PTCTLMC already has.

What genuinely survives, and §4.7 uses all four:
1. **Count in-flight work, not queue emptiness.** Incrementing before hand-off and decrementing
   after full consumption is the residual of `D(t)` and it is not optional — "all queues empty"
   races against a worker between pop and push.
2. **Memory ordering.** The counter must be release/acquire-ordered with the work it guards.
   Message passing gave this for free; shared memory does not.
3. **Contention on the counter is the new cost**, and the right structure is the shared-memory
   analogue of Mattern's **vector counters**: per-worker counters summed hierarchically only when
   a worker suspects quiescence.
4. **Phase-structured vs dynamic-pool.** A barrier is sufficient for a phase-structured algorithm
   (OWCTY) and its cost is the *number of phases* — which is precisely why Černá & Pelánek chose
   OWCTY. A dynamic-pool algorithm (CNDFS, UFSCC) needs **no termination-detection algorithm at
   all beyond thread join**, and the interesting problem moves to work distribution and to
   contention on the shared structure. PTCTLMC's outer search is phase-structured and its inner
   search is a dynamic pool, so it needs the barrier for the first and nothing for the second.

---

## 3. What stops applying because this is shared memory

A distributed model checker's architecture is mostly an answer to constraints PTCTLMC does not
have. Naming them is not pedantry: each one that disappears frees a design decision, and two of
them free decisions this engine is still paying for.

| distributed-memory assumption | holds here? | what its removal buys |
|---|---|---|
| **No shared heap.** A state exists on exactly one node; to look at it you send a message. | **No.** | Ownership becomes optional. Already exploited: the striped passed set has no static partition (`04-parallel.tex`). |
| **Routing.** A successor generated on node *i* but owned by node *j* must be serialised, sent, and re-inserted. Under uniform hashing this is (n−1)/n of *all* transitions. | **No.** | Expansion and insertion are fused; no per-level routing phase. Already exploited. |
| **Termination detection is a distributed problem** requiring Dijkstra–Scholten / Safra / credit schemes, because "no work anywhere and no message in flight" is not locally observable. | **No.** | A level barrier plus an atomic in-flight counter decides it exactly. Already exploited. **But see below — this is where the unexploited value is.** |
| **Load balancing must move state.** Rebalancing means migrating states and rewriting owner tables. | **No.** | `schedule(dynamic, 8)` over a frontier. Already exploited. |
| **Memory is per-node; a state costs the same everywhere.** | **No.** | One interned zone record is visible to all threads at one address. Already exploited (`ZoneIntern`, 64-way sharded, 4-byte refs). |
| **Communication is expensive relative to computation**, so algorithms are designed to minimise *messages*, even at the cost of extra *states*. | **No** — and inverted. | A shared-memory engine should be designed to minimise *cache-line contention and redundant work*, which sometimes means sharing **more**, not less. |
| **A global barrier costs a message round**, so distributed algorithms avoid barriers and prefer asynchronous fixpoints. | **No.** | **A barrier is nearly free and it is a *grace period*.** This is the assumption whose removal PTCTLMC has *not* cashed in, and §4.2 cashes it. |

### 3.1 The two unexploited consequences

**(A) A level barrier is a quiescent point, therefore a reclamation epoch is free.**
Distributed BFS avoids barriers. PTCTLMC keeps them anyway, for a *state-count* reason
(`04-parallel.tex`: BFS order minimises stored states because a zone reached by a shorter path
is more general). That decision has an unclaimed side effect: at the barrier **no worker holds
any reference into the passed set**. Epoch-based reclamation, hazard pointers, RCU — all the
machinery that makes lock-free structures hard — is *already paid for*. The engine currently
uses that only for parallel compaction; §4.2 uses it to remove the read lock entirely.

**(B) "Share only stable facts" is cheaper in shared memory than anywhere else.**
In a cluster, publishing a fact costs a broadcast, so distributed algorithms publish sparingly
and mostly recompute. In shared memory, publishing a fact costs one atomic store to a cache line
that other threads will read anyway. Renault et al.'s design — threads share *facts* (same-SCC,
DEAD) and never *progress* — is only affordable because the union-find is a shared heap object.
PTCTLMC's coverage-monotonicity theorem is the same kind of licence, and the engine currently
uses it in exactly one place (`probe_kill`). §4 generalises it to a discipline.

---

## 4. The architecture: one monotone publication store, two fixpoints, no partition

**Slogan: share facts, never progress. Publish monotonically, never retract. Delete only at a
barrier.**

Everything below is one design. It is stated as five layers because each layer has its own
soundness obligation, but they are not independent options — layers 2–5 are only affordable
because of layer 1, and layer 5 is only *sound* because of layer 4.

### 4.1 The principle, and why it is the right one for this engine

Three shared structures in this engine want to be the same structure, and all three obey the same
law:

| structure | the fact it publishes | monotone? | current owner |
|---|---|---|---|
| passed set antichain | "zone `z` at `dk` is covered" | **yes** — Thm. coverage monotonicity, `06-theory.tex:53` | `reach_engine` |
| retirement forwarding `fwd` | "node `u` was absorbed into `v`" | **yes** — union-find, merge-only | `leads_to_net:2543` |
| liveness verdict cache `escapes` / `comp_of`+`esc_comp` | "node `u` escapes" / "`u` is dead" | **yes** — once decided, never revised | `leads_to_net:2532` |

A fact that is never retracted can be read **without excluding the writer** and **without
validating the answer afterwards**. That is the entire content of `cor:probe`, and it is
independently the entire content of Renault et al.'s "the shared information is stable". The
architecture is: *make every shared structure in the engine a monotone fact store, and make the
representation append-only so that the second obligation (`06-theory.tex:100`, "the theorem is
about the semantics of the answer being stable; the lock is about the representation being
intact") is discharged too.*

The distributed literature cannot state this principle in this form, because in a cluster
publishing a fact is a message and reading one is a round trip; there, the winning move is to
share *as little as possible* and recompute. Here the winning move is the opposite. That
inversion is the whole reason this is a shared-memory architecture and not a ported one.

### 4.2 Layer 1 — the append-only antichain: remove the read lock

**Change.** A stripe's chain for a discrete key becomes a singly-linked list of **immutable**
arena records, published by a single release-store of the head. Nothing already linked is ever
rewritten during a level.

```
struct Node {                      // 16 B, unchanged size
  int  zidx;                       // interned zone, immutable after publication
  int  level;                      // immutable
  int  next;                       // immutable after publication  <-- the change
  std::atomic<int> fwd;            // -1 = live; else the absorber's id. ONE-WAY.
};
```

- **insert**: fill the record, then `head.store(n, release)` (or CAS the head). A reader that
  acquire-loads `head` sees a fully initialised record.
- **retire**: `fwd.store(absorber, release)` — a single word, one way, never reverted. Replaces
  the current in-place `dead` flag plus chain surgery.
- **read path (`probe_kill`)**: acquire-load head, walk `next`, skip records whose `fwd != -1`,
  test inclusion on the packed min-graph. **No lock at all.**
- **write path (`add`)**: still exclusive per stripe, because the antichain update is a
  read-modify-write over the chain. The lock is now a plain spinlock, not a reader/writer lock,
  and it carries **17% of its former traffic**.
- **compaction / interner GC**: moved to the level barrier, where they already run
  (`04-parallel.tex`), and there they need no synchronisation at all.

**Soundness, stated as two obligations, both discharged.**

1. *Semantic:* a "yes" from the probe cannot go stale — Theorem (coverage monotonicity) and its
   corollary, unchanged. Note the theorem's proof currently invokes "stripe updates at a given
   `dk` are serialised by that stripe's exclusive lock", which **still holds** — writers are
   still exclusive. Only readers are released.
2. *Representational:* a reader never observes a torn structure, because **no field a reader
   reads is ever written twice**. `zidx`, `level`, `next` are written once before publication;
   `fwd` transitions `-1 → id` exactly once. A reader that misses a concurrent retirement simply
   reads a live zone that is about to be absorbed — and by the Replacement Lemma the absorber
   contains it, so the answer it computes is still a correct "covered" or a harmless "not
   covered". A reader that misses a concurrent insertion returns "not covered" and pays one
   wasted exclusive acquisition, which is the already-accepted false-no case.

**Why the level barrier is what makes this affordable, and why a cluster could not do it.** The
one thing an append-only structure cannot do is reclaim. Lock-free reclamation normally needs
hazard pointers or epochs. Here the epoch is handed to us: at the barrier every worker is
provably between levels and holds no reference into the arena, so compaction and GC can move,
renumber and free with no readers present and no protocol. **A distributed engine avoids
barriers because a barrier costs a message round; PTCTLMC keeps barriers for a state-count
reason and therefore gets a free grace period.** This is the single clearest case in the design
of a distributed-memory constraint inverting into a shared-memory asset.

**Provenance, honestly.** Lock-free append-only lists with a monotone tombstone word are
textbook (Harris-style linked lists, RCU, epoch-based reclamation). Nothing there is new. What
is new is the *pairing*: this project already isolated the two obligations and proved the first
one; the claim here is that the second is discharged by append-only, and that level-synchronous
BFS — kept for an unrelated reason — pays for the reclamation. Expected effect: removes lock
traffic on ~83% of insertion attempts; on `phil13` the write-lock convoy was measured at 378 of
759 CPU-seconds under `std::shared_mutex` and ~15% of thread time as spin under `SpinRW`, so
this is bounded above by ~15% and is **not** the main prize. The main prize is that layer 1 is
what makes layers 3–5 possible at all.

### 4.3 Layer 2 — one store for both engines

Today `leads_to_net` re-implements the passed set with `std::unordered_map<std::string,
std::vector<int>>` under **one `std::mutex`** (`:2758`), keeping a full `SymState` (dense `Dbm`,
`locs`, `ints`) per node. `reach_engine` has 16 384 stripes, `SpinRW`, a packed 63-bit key,
interned min-graph zones and 4-byte references. This is the largest avoidable cost in the engine
and it is pure duplication.

The reason the two cannot be merged *today* is a genuine conflict, and naming it is the design
step:

> Reachability needs to **delete and renumber** (tombstone + compaction + interner slot
> recycling). Liveness needs **stable node identity**, because `adj` stores plain indices and
> Tarjan carries per-node live state.

Layer 1 dissolves it. Under append-only + monotone `fwd`:

- identity is stable *by construction* — a record is never moved during a level and never
  logically deleted, only forwarded;
- `resolve()` (path-compressing union-find over `fwd`) is exactly what `leads_to_net` already
  does at `:2543`, promoted from a private vector to the shared store;
- compaction may still renumber **at the barrier**, and for liveness queries it either runs in
  forwarding-preserving mode (leave a 16-byte stub) or is disabled; it is a per-query-class knob,
  not a fork of the data structure.

**What this buys, concretely and independently of any parallelism:** liveness stops paying a
`std::string` key per state (`disc_key` returns `std::string`; `reach_engine` uses a packed
`std::uint64_t`), stops storing `locs`/`ints` per node (the packed key is a lossless inverse —
`net_passed.hpp:56-65` documents this exact saving for reachability), stops holding dense DBMs
(interned min-graph records instead), and stops serialising every graph mutation on one mutex.
None of that is "adding threads"; it is removing a global lock and an O(bytes) key from the hot
path of a search that is already parallel.

### 4.4 Layer 3 — the deadlock gate (MEASURED at ~97% of a `p --> q` query)

Read `src/netreach.cpp:2506-2508`:

```
ReachStats dl_stats;
const bool may_deadlock = deadlock_reachable(net, backend, &dl_stats);
const bool use_alu = !may_deadlock && !network_has_diagonal(net);
```

`p --> q` therefore costs **one complete parallel reachability traversal of the state space**
(`deadlock_reachable` → `reach_engine` with `force_exact`) *before the liveness search starts*,
and then **a second complete traversal** to build the liveness graph — under a global mutex,
with a different passed set, with no reuse of anything the first pass computed. The comment says
"The check costs one reachability run and buys back two orders of magnitude", which is a correct
trade against the alternative, but it is not the only option.

**Change.** One traversal. The outer level-synchronous BFS builds the shared store and, per
node, records the two bits the liveness layer needs, both of which are computable from data the
successor generation already has in hand:

- `has_deadlock(u)` — `state_has_deadlock` (`:2855`) is already an exact per-state computation
  over the same three transition kinds `successors_net` enumerates. Computing it during
  expansion reuses the guard/reset/invariant work instead of redoing the whole search.
- `time_diverges(u)` — currently recomputed (`:2602`) with a fresh `delay()` + invariant apply,
  called from three places, including inside Tarjan's inner loop.

**§1.5 measured what this costs: ~97% of the query on fischer7 and fischer8, and the pre-pass
explores 14× the states of the same model's reachability because `force_exact` disables `a_LU`
(11 951 → 167 805 on fischer7; 1 858 837 on fischer8).**

#### The obvious fix is UNSOUND, and saying so is the point of this subsection

My first instinct — and I wrote it into an earlier draft of this document — was: *start with
`a_LU` on, compute `state_has_deadlock` per node during the search, and if any node is
deadlocked, publish the bit and restart in exact mode.* One-way, monotone, at a level boundary,
exactly the shape of `thm:adapt`.

**It does not work, and the reason is the same simulation asymmetry as §4.5.** The `a_LU` search
stores a subset of the reachable symbolic states; a state `z` dropped because `z ⊑_aLU Z` never
has its successors generated. `a_LU` is a simulation, so every state reachable from `z` is
simulated by *some* state reachable from `Z` — but **simulation does not preserve deadlock**
(that is the whole reason the gate exists). So a deadlocked state reachable only through a
dropped state is never generated and never tested, the abort never fires, and we wrongly conclude
deadlock-freedom. Testing the dropped states themselves as they are discarded is not enough
either — the miss can be arbitrarily deep. Any scheme of the form "run coarse, detect, abort" is
unsound here, and it is unsound for a reason the engine's own code already knows.

#### What is actually worth trying, in order

1. **A static sufficient condition for deadlock-freedom.** The gate does not need the exact
   answer; it needs a sound `false`. A syntactic, whole-model check that implies no reachable
   valuation can deadlock — e.g. every location either has no invariant, or has an outgoing edge
   whose guard is implied by the invariant's own upper bound and whose target invariant is
   satisfiable after the reset — would discharge the gate in O(model), not O(state space). Every
   model this check accepts saves the entire 97%. Every model it rejects falls back to today's
   behaviour, so it is a pure win with no soundness risk. **This is the single highest-value item
   in this document and it is not a parallel algorithm.** Whether the benchmark families satisfy
   such a condition is an open question I did not have time to settle; fischer, arbiter, viking,
   twolock and csma are all measured deadlock-*free* by the engine, so the condition has
   something to find.
2. **Lazy gating.** Run the `a_LU` liveness search *first*. If it finds no witness, the verdict
   depends on `a_LU` being sound, so the gate must be paid. If it finds a witness, check which of
   the three witness kinds it is: **a `¬q` cycle and time divergence both transfer along a
   simulation** (the code says so at `:2493-2495`), so a witness of kind 1-divergence or kind 2
   is valid **without the gate**. Only a deadlock witness needs it. This makes violating queries
   free of the pre-pass entirely, and leaves the holds case paying. Given that the current
   measured advantage over `verifyta` is on violating queries, this is a real win on exactly the
   cases the project already advertises.
3. **Fuse, don't eliminate.** When the gate must be paid, pay it once: build the exact graph and
   derive the `a_LU`-subsumed liveness graph as a quotient of it, rather than traversing twice.
   Measured ceiling on this alone: ~3% (fischer7, 5.97 → ≈5.79 s). Worth doing as cleanup, not
   worth doing first.
4. **Only then**, per node, reuse `state_has_deadlock` and `time_diverges` results computed during
   expansion rather than recomputing them (`time_diverges` is called from three sites, one of them
   inside Tarjan's inner loop, and each call does a fresh `delay()` plus invariant application).

**This layer is engineering, not a contribution — and it is worth an order of magnitude more than
everything else in this document.** Those two facts are not in tension; §5.3 says so plainly.

### 4.5 Layer 4 — the subsumption firewall: where coverage monotonicity does *not* apply

This is the layer the brief specifically asks about, and the answer is uncomfortable.

**Coverage monotonicity licenses dropping a state. It does not license collapsing a node.**

`cor:probe` says: if `z` is covered at time `t` it is covered at every `t' ≥ t`, so discarding
`z` is safe. For *reachability* that is the whole story, because reachability asks only "is the
covered region large enough to contain a target".

For *liveness* it is not, and the reason is structural rather than temporal. `leads_to_net`
retires `u` into `v` when `Z_u ⊑ Z_v` and rewrites every edge pointing at `u` to point at `v`
(`fwd[j] = n` at `:2589`, `resolve` at `:2543`). Runs from `u` are a subset of runs from `v`, so
**no witness is lost** — that is what the comment at `:2477-2482` proves, and it is correct. But
the quotient **adds in-edges to `v`**: a predecessor `w` whose real successor is `Z_u` now
appears to step into `Z_v ⊇ Z_u`. A cycle `v →* w → v` in the quotient uses that edge, and the
segment `v →* w` was computed from `Z_v`, not from `Z_u`. Such a cycle need not be realised by
any run. **A spurious `¬q` cycle yields a spurious escape, hence a spurious "property fails".**

So the direction of soundness is:

| claim | status |
|---|---|
| the engine never misses a violation (no witness lost) | **sound**, proved at `:2477`, and this is the direction coverage monotonicity licenses |
| the engine never reports a violation that does not exist | **NOT ESTABLISHED** — neither by the code comments nor by any theorem in `doc_1`, and the literature says the general property is *false* |

The current guard, `untouched = !expanded[j] && tidx[j] == -1` (`:2570`), is a **timing**
condition — "nothing has looked at this node yet". The literature's criterion is a **structural**
one:

> a subsumption graph is liveness compatible iff it has **no cycle containing both an accepting
> node and a subsumption edge** (Laarman, Olesen, Dalsgaard, Larsen, van de Pol, CAV 2013).

`untouched` does not imply that criterion: a node can be untouched when it is retired and later
find itself on a cycle, because the predecessor edge that closes the cycle may be created many
levels afterwards. **I could not construct a concrete counterexample within this task and I am
not asserting the code is wrong** — see experiment E0, which is designed to settle it in an hour.

**The firewall (the design change).** Stop collapsing. Make subsumption an *edge*, and tag it.

```
retire(u, v):   fwd[u] = v            // keep u as a node
                sub_edge[u] = true    // u has exactly one out-edge, u --SUB--> v
                                      // u is never expanded
```

Then the SCC layer sees subsumption explicitly and can enforce the criterion directly:

- **Cheap path (the common case).** A closed component containing **no** `SUB` edge is
  liveness-compatible by construction, and the existing escape logic decides it unchanged. On
  the five families where the reachability antichain is a singleton per discrete key
  (`fischer, arbiter, viking, twolock, csma` — measured ratio 1.000), retirements are rare and
  most components will be `SUB`-free.
- **Candidate path.** A component containing a `SUB` edge *and* an accepting node is a
  **candidate**, not a witness. Refine it *locally*: disable the offending `SUB` edges — i.e.
  actually expand the subsumed node `u` with its own zone `Z_u` — and re-run the escape decision
  on that component only. Repeat; each refinement strictly removes a `SUB` edge, so it
  terminates.

That local refinement is Herbreteau–Srivathsan–Walukiewicz's iterative liveness-compatible
construction, **scoped to a single SCC instead of the whole graph**, and made possible by having
the `SUB` edges tagged rather than quotiented away. That scoping is the piece I believe is new;
see §5 for the provenance split, and note that HSW's version is sequential and global, and
Laarman et al.'s version is a DFS-stack discipline inside NDFS, which is not available here
because §4.5's SCC layer is not an NDFS.

**Where coverage monotonicity still applies inside liveness, and it does:** dropping a state
from the *frontier* because its zone is covered is still sound for the *discovery* fixpoint (it
cannot make a reachable discrete state unreachable). The firewall is only about the *cycle*
fixpoint. Keeping those two uses of the same structure distinct is the point of the layering.

### 4.6 Layer 5 — the liveness fixpoint: not OWCTY, and here is why

The brief asks directly whether a DFS-free OWCTY/MAP-style approach beats Tarjan here. **My answer
is: not worth it — but for one reason, not four, and two of the reasons I first wrote down turned
out to be wrong.** Recording that is more useful than a clean-looking argument.

**Two arguments I retract.**

- ✗ *"OWCTY's elimination step is backward and PTCTLMC has no predecessor edges."* **Wrong, and it
  is what the name means.** OWCTY maintains indegrees "**without the need to enumerate the
  predecessors of a state** … the algorithm only performs forward traversal procedures … **This is
  exactly what the One-Way in the name of the algorithm stands for**" (§2.6.1). FB and OBF need
  reverse edges; OWCTY does not. It runs on a `next-state`-only interface, which is exactly what
  PTCTLMC has.
- ✗ *"OWCTY is only linear on weak automata and `p --> q` is probably not weak."* **The opposite
  is true.** `p --> q` fails iff `F(p ∧ G¬q)` holds — a **persistence** property, and by the
  handbook's Property 12 "one can build a **weak** automaton equivalent to any LTL persistence".
  The negated-property automaton for a response query is weak (inside the `¬q` trap component,
  every cycle is accepting), so **OWCTY is `O(V+E)` and time-optimal for exactly this query
  class** (handbook Table 12.1: Owcty on weak Büchi — `O(V+E)`, Optimal: Yes). The complexity
  objection to OWCTY does not apply to `p --> q`.

**The one argument that survives, and it is decisive.**

OWCTY is **not on-the-fly** (handbook Table 12.1: On-the-fly — *No*; MAP: *Partially*). Černá &
Pelánek measured the cost themselves: with a fair cycle present, sequential NDFS beat their
distributed set-based algorithm **12.82 s vs 305.51 s** on Ring2, purely because NDFS stops at the
first witness. `leads_to_net`'s entire advertised advantage is that property — `arbiter6` 256
nodes vs `verifyta` 10 453, `fischer7` 11 168 vs 238 617, and the code's own note at `:2661` that
eager escape took `fischer7` from **748k nodes to 1**. Adopting OWCTY would trade a 40×–750×
node-count advantage on violating queries for a constant factor on a phase that §1.5 measures at
**3% of the query**.

**And the premise of the question needs correcting too.** "Tarjan is inherently sequential" is not
what Reif proved. He proved *lexicographic* DFS is P-complete — **with the transition order fixed
in advance** (§2.6.3). Bloemen states the consequence bluntly: the result "does not apply
directly" to these algorithms, and free (non-lexicographic) DFS may well be in NC. Every practical
parallel scheme in this literature exploits that gap by randomising successor order per worker and
sharing only order-independent facts. DFS is not the obstacle; **sharing DFS progress** is.

**If the workload ever changes, the answer changes.** If a future benchmark suite were dominated
by *violating* liveness queries, the thing to add is **MAP** (partially on-the-fly, designed for
bug-hunting, `O(a²·m)`), or DiVinE's `OTF_OWCTY` combination — not OWCTY alone. That is the
literature's own conclusion (MAP for bugs, OWCTY for full verification), and it is confirmed from
the other camp by Laarman & van de Pol, PDMC 2011.

**The proposal, therefore:**

```
shared:   UF over  Nodes ∪ {Dead, Escape}          // lock-free, merge-only
          publish:  unite(x, y, mark)              // "x and y are in the same SCC"
                    unite(x, Dead)                 // "x cannot escape"     (monotone)
                    unite(x, Escape)               // "x escapes"           (monotone)
          query:    same_set(x, Dead) / same_set(x, Escape) / same_set(x, y)

per thread (never shared):  dfs stack, live stack, livenum map, pstack
```

- The **outer discovery stays a level-synchronous parallel BFS** over the shared store. It is not
  replaced by swarmed DFS, because BFS order is what minimises stored zones
  (`04-parallel.tex`) and that is this project's measured priority.
- The **inner escape analysis becomes parallel over the level's accepting seeds.** The loop at
  `:2828` (`for (int u : accepting) { ... neg_scc(u) ... }`) is today sequential and is the only
  sequential phase left in the liveness path. Each seed gets a worker running the existing
  iterative Tarjan with its **own** stacks, consulting and publishing to the shared UF.
- **Soundness of the sharing**, stated per fact:
  - `same_set(x, y)` — "x and y are in one SCC" is stable because mutual reachability is
    monotone in a growing graph and is never invalidated by later discovery.
  - `same_set(x, Dead)` — "x cannot escape" is decided only when x's component is closed *and*
    every component reachable from it is closed; reverse-topological order gives that in one pass,
    which is exactly the argument already written at `:2619-2623`.
  - `same_set(x, Escape)` — monotone by inspection: `escapes[u]` is only ever set, never cleared.
  - Therefore no synchronisation point and no repair procedure is needed, for the same reason
    Renault et al. need none. This is the same monotone-fact licence as coverage monotonicity, and
    that identification is the conceptual core of the whole architecture.
- **Eager escape survives** and gets stronger: today `stack_has_p()` scans one thread's stack; with
  a shared `Escape` class, one thread's discovery immediately prunes every other thread's search.
- **Swarming (randomised successor order per thread) is optional and default-off.** It is the
  standard diversity mechanism, but this project already measured that intra-level expansion
  ordering does not pay (`10-limits.tex`: ordering by generality gave `phil12` 0.95× and *more*
  states). I would ship it behind a flag and treat it as unproven here.

### 4.7 Termination detection and load balancing — the short section, deliberately

**Termination detection.** Distributed model checkers spend real design effort here
(Dijkstra–Scholten diffusing computations, Safra's token, Mattern's credit schemes) because "no
node has work and no message is in flight" is not locally observable. **None of that is needed.**

- *Outer BFS:* termination is `frontier.empty()` observed after a barrier. Already exact.
- *Inner SCC layer:* one `std::atomic<int> active` incremented when a worker takes a seed and
  decremented when it finishes, plus the seed queue's emptiness. Quiescence is
  `active == 0 && queue.empty()` read once at the barrier. This is Dijkstra–Scholten degenerated
  to a single counter — the diffusing-computation *invariant* is the same; the *protocol* (parent
  acknowledgements, tokens, colours) exists only to make the counter observable without shared
  memory, and evaporates.
- The one real hazard is the classic one: a worker that has decremented but whose *published
  facts* are not yet visible. A release-store on publish and an acquire-load at the barrier fix
  it; the barrier is already a full fence. Message passing gave this ordering for free via the
  network layer; shared memory does not, and this is the one obligation the port *adds* rather
  than removes.
- **What must not be naive: a single global counter.** It serialises exactly where you least want
  it. The right structure is the shared-memory analogue of **Mattern's vector counters** (1987):
  per-worker counters, summed by a hierarchical reduction only when a worker suspects quiescence.
  That is the one piece of the Dijkstra–Scholten / Safra / Mattern corpus that transfers directly
  rather than being subsumed by the shared heap (§2.6.5).
- **And note what Černá & Pelánek's design principle implies for us.** They chose OWCTY over
  Emerson–Lei because "**the number of external iterations determines the number of necessary
  synchronisations**". PTCTLMC's outer loop has one barrier per BFS *level*, which is many more
  synchronisations than OWCTY's `h`. That is a real cost, and the project has accepted it
  deliberately for a state-count reason (`04-parallel.tex`). It is worth stating in the paper as a
  conscious trade against the distributed literature's default, rather than leaving a reviewer to
  raise it.

**Load balancing.** Also short, and also because of shared memory. The outer level uses
`schedule(dynamic, 8)` and the hash gives the balance. The inner layer's units of work are
*seeds*, and seed cost is wildly non-uniform (one seed may settle in one node, another may walk a
million). So: a shared seed deque with dynamic hand-out, plus — and this is where the shared
`Dead`/`Escape` classes earn their keep a second time — a worker that starts on a seed already
resolved by another worker's published facts drops it in O(1). No work stealing between *stacks*
is needed or wanted, because splitting a DFS stack is exactly the "share progress" move the
design refuses.

### 4.8 What I was tempted by, and why I am not proposing it

Named explicitly, per the brief.

1. **Per-thread local liveness graphs with a merge at the barrier.** Tempting because the global
   `gmtx` in `leads_to_net` is so obviously bad. **Rejected — this is already tried and removed**
   (`10-limits.tex`: per-thread local antichains with a drain at the barrier were consistently
   slower, up to 1.92× on `fischer10`, because a private table cannot see the shared coverage
   probe that answers 83% of successors; on `phil12` the drain spent 19.4 s of thread time against
   a 3.8 s wall). The measured conclusion there — *a local-buffer design is only viable with a
   shared coverage probe* — applies verbatim, and layer 1 goes the other way: make the shared
   probe cheaper, not private.
2. **Partition-and-route the passed set (distributed Murphi's design).** Rejected; the project
   already chose a shared table over a static partition and `04-parallel.tex` gives the reason
   (clustered models starve workers, and every level needs a routing phase). Restating it would be
   a regression.
3. **Ordering the seed queue by anything clever.** Rejected on the project's own negative result:
   expansion ordering by generality did not pay (`10-limits.tex`), and a worklist order cannot
   affect a fixpoint.
4. **Backward (pre-image) reachability à la ZEUS.** Rejected — but on the sources' reasons, not
   mine (§2.3): discrete/data assignments are not invertible ("**the backward computation is not
   appropriate to manipulate other variables**", Bouyer MOVEP'14; "**Kronos does not support data
   variables**", Farkas et al. AVoCS 2018), and backward cannot do on-the-fly composition or early
   termination. PTCTLMC's networks have integer variables and its whole `a_LU`/extrapolation/
   interner stack is forward-only. Note what is *not* a reason: backward terminates natively and
   needs no abstraction at all, and on **diagonal constraints backward is strictly better** —
   which matters, because diagonal constraints are the one case where `10-limits.tex` records that
   this engine's algorithmic advantage disappears. A backward path for diagonal models is a
   defensible future direction; it is not a parallelism proposal and I am not making it here.
5. **A second, DFS-free liveness engine alongside the Tarjan one.** Rejected on the project's own
   stated principle (`04-parallel.tex`: "there is one parallel implementation to get right rather
   than one per query form"). Layer 5 keeps one engine.
6. **Convex-union merge on the liveness graph.** Rejected: already measured a net loss
   (`:2571-2580`, `arbiter6` 66 137 → 75 818, `twolock6` 35 351 → 41 943) and it would additionally
   create *new* zones, which under §4.5 would multiply `SUB` edges. Keep it opt-in and off.

---

## 5. Provenance: new, assembled, or plumbing

Stated in the style of `09-novelty.tex`, because that is the standard this project already holds
itself to, and because an inflated claim here would be caught immediately.

### 5.1 Assembled from published parts, and credited

| part | source |
|---|---|
| shared monotone fact store as the *only* inter-thread channel; "share facts, never progress"; lock-free union-find carrying SCC membership + a `Dead` class; per-thread private DFS stacks; strategy mixing | Renault, Duret-Lutz, Kordon, Poitrenaud, TACAS 2015 / STTT 19(6) 2017 |
| liveness-compatible subsumption graph = *no cycle with an accepting node and a subsumption edge* | Laarman, Olesen, Dalsgaard, Larsen, van de Pol, CAV 2013 |
| iterative refinement that disables offending subsumption edges and re-derives | Herbreteau, Srivathsan, Walukiewicz, CAV 2010 / FMSD 40(2) 2012 |
| subsumption is sound for reachability and unsound for Büchi acceptance; the survey statement | Bouyer, Gastin, Herbreteau, Sankur, Srivathsan, arXiv:2207.07479 |
| liveness is strictly harder than reachability unless P=NP | Herbreteau, Srivathsan, Tran, Walukiewicz, FSTTCS 2016 |
| Büchi emptiness on simulation graphs; strongly non-Zeno | Tripakis, ACM TOCL 10(3), 2009; Li, FORMATS 2009 |
| lock-free append-only list, one-way tombstone word, epoch-based reclamation | textbook (Harris; Michael; RCU). No novelty claimed. |
| concurrent union-find via CAS | Anderson & Woll, as used by Renault et al. |
| level-synchronous parallel BFS; shared table over static partition; the antichain; the convex-union merge; coverage monotonicity | **this project's existing work**, `doc_1/sections/03,04,06,09` |

### 5.2 New here, as far as these searches establish

Three claims, in descending order of confidence.

**A. Level-synchronous BFS is a reclamation epoch, therefore the passed-set read path needs no
lock.** The project already isolated the two obligations — semantic stability (proved) and
representational integrity (currently discharged by a shared lock, `06-theory.tex:100`). The
claim is that the *second* is discharged by making the antichain append-only with a one-way
`fwd` word, and that the barrier PTCTLMC keeps for a **state-count** reason is exactly the grace
period an append-only structure needs. The parts are textbook; the observation that a
subsumption-based timed passed set can be made read-lock-free this way, and that the enabling
barrier is already paid for by an unrelated decision, is not something I found stated anywhere.
Confidence: high that it is correct; medium that it is unpublished.

**B. Unifying "coverage monotonicity" and "stable SCC facts" as one discipline, and running both
fixpoints against one monotone store.** Renault et al.'s invariant and this project's theorem are
the same shape, and neither literature notices the other: the parallel-SCC line does not use
subsumption, and the timed-subsumption line (CAV 2013) went the NDFS route, which *must* share
progress and therefore *must* have synchronisation points or repair procedures. Combining them —
a subsumption antichain and an SCC fact store as two monotone publications into one shared
structure — is the architectural claim. Confidence: high that the combination is new. **Its
practical value is now in doubt**: §1.5 measures the phase it optimises at 3% of a `p --> q`
query, and E4 may show the seed loop has nothing to parallelise. It becomes valuable only after
§4.4 removes the pre-pass, at which point the liveness builder's measured ≈1.3× scaling becomes
the whole story. Publishing this claim before re-running E1 would be premature.

**C. SCC-local subsumption refinement.** Tagging retirements as `SUB` edges rather than
quotienting them away, letting the SCC layer test the CAV 2013 criterion *per component*, and
refining only inside a candidate component (rather than HSW's global rebuild or CAV 2013's
NDFS-stack discipline, which is unavailable outside an NDFS). Confidence: medium. It is a
scoping of a known technique; the honest framing is "the technique is HSW's, the scoping and the
edge-tagging are new, and the reason the scoping is affordable is that the components are small
because the antichain is a singleton per discrete key on five of eight families".

### 5.3 Not new, not claimed, and worth more than everything above it

Getting rid of the deadlock pre-pass (§4.4) is **engineering**, carries no theorem, and is
**measured at ~97% of a `p --> q` query** (§1.5). Those facts are not in tension; the write-up
should state all three. `09-novelty.tex` already holds this project to exactly this standard
("Engineering: real, necessary, not a theoretical claim … they are nonetheless where most of the
measured speedup lives"), and this is another instance of it.

### 5.4 Claims retracted during the writing of this document

Listed because a design document that hides its own corrections is less useful than one that
shows them.

- *"OWCTY's elimination is backward, so PTCTLMC would need predecessor edges or zone pre-images."*
  **Retracted** — OWCTY maintains indegrees by forward traversal only; that is what "One-Way"
  means (§2.6.1).
- *"OWCTY is only linear on weak automata, and `p --> q` probably is not weak."* **Retracted** —
  `¬(p --> q)` is a persistence property, so its automaton *is* weak and OWCTY is linear and
  optimal on it (§4.6).
- *"Start with `a_LU`, detect a deadlock during the search, and abort to exact mode."*
  **Retracted as unsound** — `a_LU` is a simulation and simulation does not preserve deadlock, so
  a deadlocked state reachable only through a dropped state is never generated and the abort never
  fires (§4.4).
- *"The global mutex in `leads_to_net` is the liveness bottleneck."* **Half-retracted** — it is
  the bottleneck of the liveness *builder* (≈1.3× on 12 threads), but that phase is 3% of the
  query, so the claim was true and the priority was wrong (§1.5, E1).

---

## 6. The smallest experiments that would refute this

Ordered by cost. **E1 and E2 have already been run — see §1.5.** E2 is confirmed at ~97%; E1 is
answered in a way that partly refutes my own layer-2/3 premise. E0 is now the most important
unrun experiment, because it targets a possible soundness bug.

> **Status of the refutation test.** The single smallest experiment that could have refuted this
> architecture was E1 ("does liveness scale today?"). It ran. The answer is: `p --> q` scales
> **4.8× on 12 cores**, which looks healthy, but ~97% of that time is the *reachability* engine
> running the deadlock pre-pass, and the liveness graph builder itself scales ≈**1.3×**. So the
> premise of §4.3 (global mutex is the bottleneck of the liveness builder) is **confirmed**, and
> the premise that fixing it matters is **refuted at current scale**. §4.4 is now the headline and
> §4.2/§4.3/§4.6 are contingent on §4.4 landing first.

### E0 — is the existing liveness subsumption creating spurious witnesses? (≈1 hour)

Two variants, run both.

*E0a, instrumentation only (~20 lines).* When `neg_scc` returns true, walk the certifying
structure and record whether any edge on it was `resolve()`d — i.e. whether the witness used a
subsumption redirection. If, across every liveness query anyone can construct, **no** reported
witness ever uses one, then §4.5's hazard is real in theory but unexercised in practice, and the
firewall is cheap insurance rather than a bug fix.

*E0b, differential (~1 hour).* Add an env flag that forces `use_alu = false` **and** disables the
inclusion-based retirement at `:2586` entirely (pure exact-hash nodes — the pre-rewrite
behaviour, which the comment at `:2472` says was `arbiter6` = 1 343 096 nodes, so keep the models
tiny). Compare **verdicts only**, not times.
**Refutes §4.5 if:** verdicts always agree. **Confirms a bug if:** subsumption-on ever reports
`fails` where subsumption-off reports `holds`.

*Blocking prerequisite for both, and it is a finding in its own right:* **the repo has no
liveness queries.** All 11 benchmark queries are `A[]` safety. The liveness path that was
rewritten today is therefore covered by no benchmark and no differential oracle. Writing
`p --> q` queries for fischer/arbiter/viking/twolock/csma and adding them to
`benchmarks/verify_parity.py` is a precondition for *any* claim about liveness, including the
existing one that it "beats verifyta everywhere measured".

### E1 — does liveness scale today? — **RUN. Answer: yes on paper, no where it matters.**

Result in §1.5, Table B. fischer7 `p --> q`: 5.97 s → 1.24 s on 12 threads (**4.8×**). fischer8:
236.28 → 56.81 on 8 threads (**4.2×**). Both figures are dominated by the deadlock pre-pass, which
runs on the *reachability* engine and scales 5.3×. The residual liveness-graph phase goes
0.18 s → 0.14 s, i.e. ≈**1.3×**.

**What this settles.** §4.3's diagnosis of `leads_to_net` (one `std::mutex`, `std::string` keys,
`SymState` copies under the lock) is correct — 1.3× is what that looks like. §4.3's *priority* is
wrong: it is 3% of the query. Do §4.4 first.

**What it does not settle.** 12 cores is not 32. The reachability engine's published 16→32 figure
is 1.62–1.75×; nothing here measures liveness past 12. If the pre-pass is removed by §4.4, the
liveness builder becomes 100% of the query and its 1.3× becomes the headline — at which point
§4.2/§4.3 move from "3%" to "everything". **Re-run E1 immediately after §4.4 lands.**

### E2 — how much of a `p --> q` query is the deadlock pre-pass? — **RUN. Answer: ~97%.**

fischer7: pre-pass 5.79 s of 5.97 s at 1 thread; 1.10 s of 1.24 s at 12 threads.
fischer8 at 8 threads: 54.82 s of 56.81 s (**96.5%**), and the pre-pass explores **1 858 837**
states against the model's 11 951-state (fischer7) / much smaller reachability count, because
`force_exact` disables `a_LU`.

**Remaining work on E2:** replace the two-process subtraction with an in-process timer around
`deadlock_reachable` at `:2507` before any of this is written up, and extend the table to
arbiter/viking/twolock/csma. The 97/3 split is not a measurement-error question, but the exact
figures are.

### E3 — is the shared read lock worth removing? (existing profiler)

`phprof` already instruments per-phase time. Add a counter for time inside
`lock_shared`/`unlock_shared` on the probe path.
**Refutes §4.2's own justification if:** under ~5% of expand time. (Note: layer 1 would still be
needed as the *enabler* for layer 2's stable identity, but it would have to be justified on that
ground alone, not on lock cost.) Prior evidence suggests the honest expectation is small: spin
was measured at ≈15% of thread time, and that figure covers writers too.

### E4 — is there anything to parallelise in the seed loop? (one counter) — **kills layer 5 if it fails**

Count `accepting.size()` per level, and count how many seeds actually invoke `neg_scc` rather
than hitting the `comp_of[u] >= 0` memo (`:2830`).
**Refutes §4.6/§4.5's parallel-seed design if:** the count is essentially always 0 or 1 — which is
plausible, because the code's own comment at `:2746-2748` says the inner search "either settles
the verdict within a handful of nodes or is never reached at all because the property holds".
If so, layer 5 collapses to nothing and the architecture reduces to layers 1–4, i.e. to
"one shared store, one traversal" — which is still worth doing, but the SCC-parallelism claim
must be dropped.

### E5 — does the firewall cost anything? (after E0)

With `SUB` edges tagged and kept as real nodes (§4.5), measure node count and wall time against
today's collapsing behaviour.
**Refutes §4.5's affordability if:** node counts blow up materially — the retired nodes are no
longer collapsed, so on models where retirement is frequent the graph grows. Expected small on
the five families measured at 1.000 zones per discrete key; **expected bad on `single`**, which is
the adversarial family and should be in the table.

### 6.1 Order of work, revised by the measurements

1. **Write liveness queries and put them in the parity harness.** All 11 benchmark queries are
   `A[]` safety. The liveness path rewritten today is covered by **no benchmark and no
   differential oracle**, so the claim that it "beats verifyta everywhere measured" currently
   rests on ad-hoc runs. This blocks everything and is a finding in its own right.
2. **E0a** (instrument: does any reported witness use a `resolve()`d subsumption edge?). Possible
   soundness bug; ~20 lines; do it before anything else touches the liveness path.
3. **§4.4 item 1** — hunt for a static sufficient condition for deadlock-freedom. **This is worth
   ~97% of a `p --> q` query** and it is a model-analysis problem, not a parallelism problem.
   If no such condition exists, do **§4.4 item 2** (lazy gating), which at minimum makes violating
   queries free of the pre-pass.
4. **Re-run E1.** With the pre-pass gone, the liveness builder is the whole query and its 1.3×
   becomes the number that matters.
5. **E4** (count accepting seeds per level). Cheap, and it can delete layer 5 outright.
6. Only then implement, in order: layer 2 (§4.3, unify onto the striped store), layer 1 (§4.2,
   append-only read-lock-free), layer 4 (§4.5, the subsumption firewall), layer 5 (§4.6, parallel
   seeds over a shared fact store).

Note the deliberate inversion, now backed by numbers rather than by intuition: the **least novel**
item (a static deadlock-freedom check) is first because it is measured at ~97% of the runtime, and
the **most novel** layer (parallel SCC over a shared monotone fact store) is last because it
targets a phase measured at 3% and E4 may show there is nothing there at all. A design document
that ordered these by novelty rather than by expected value would be selling something.

