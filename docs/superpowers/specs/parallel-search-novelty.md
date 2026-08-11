# A genuinely new parallel search algorithm for the zone graph — working log

**Status: IN PROGRESS. Written incrementally. Every number below was produced by running a
binary; nothing here is argued from first principles unless it is labelled a proof.**

- Engine: `./build-mostopt/ptctlmc`, `PTCTLMC_STATS=1`.
- Reference: `/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta -u -S0`.
- Machine: 12 logical cores, darwin 25.5.0. Repo at `horanewv/main`, 0 ahead / 0 behind.
- Runs that did not complete are reported as INCOMPLETE and are never counted as agreement.

---

## 0. Where the previous study stopped, and the one door it left open

`docs/superpowers/specs/reachability-node-count.md` proves, six independent ways, that on **A[]
reachability** the stored count is at a hard floor of `|reachable discrete states|` on five of six
families, and that no zone-level lever moves it. That study is correct and this document does not
contradict it. It also states its own scope in one line that turns out to be the whole story:

> every benchmark query is `A[]`.

Confirmed here — all eleven query files in `benchmarks/queries/` are `A[]`:

```
arbiter.q  A[] not (C1.cs && C2.cs)          phil.q     A[] not (P0.Eat && P1.Eat)
csma.q     A[] not (S0.Send and ...)         single.q   A[] x0 <= 6
fischer.q  A[] not (P1.cs && P2.cs)          train.q    A[] not deadlock
grc.q      A[] not deadlock                  twolock.q  A[] not (P1.cs && P2.cs)
hddi.q     A[] !(...)                        viking.q   A[] not (Viking1.safe && ...)
train_uppaal.q  A[] not (Train(1).Cross && Train(2).Cross)
```

So the "floor" result is a result about **one query class, measured on eleven queries of that
class**. The engine *does* implement liveness (`leads_to_net` in `src/netreach.cpp:2501`,
`p --> q`), and liveness has never been benchmarked. Section 2 measures it. The floor does not
hold there — it is missed by between one and three orders of magnitude.

---

## 1. Reachability: reproducing the floor, plus one column the previous study did not report

All runs completed. 12 threads, default backend.

| model | disc | stored | distinct-zones | **generated** | explored | gen/stored |
|---|---|---|---|---|---|---|
| fischer8   | 25080  | 25080  | 19213  | 177 000    | 40536  | 7.06 |
| fischer10  | 260998 | 260998 | 205744 | 2 372 190  | 447598 | 9.09 |
| twolock8   | 27128  | 27128  | 21290  | 186 216    | 42584  | 6.86 |
| twolock10  | 271238 | 271238 | 216353 | 2 428 510  | 457838 | 8.95 |
| arbiter10  | 6144   | 6144   | 6144   | 38 400     | 6144   | 6.25 |
| arbiter15  | 278528 | 278528 | 278528 | 2 457 600  | 278528 | **8.82** |
| viking10   | 45566  | 45566  | **120**| 122 880    | 45566  | 2.70 |
| viking12   | 241662 | 241662 | **120**| 688 128    | 241662 | 2.85 |
| csma8      | 4096   | 4096   | 4096   | 22 016     | 4096   | 5.38 |
| csma10     | 22784  | 22784  | 22784  | 147 200    | 22784  | 6.46 |
| phil10     | 59048  | 62177  | 840    | 461 622    | 71137  | 7.42 |
| phil12     | 531440 | 574234 | 8825   | 6 086 960  | 798574 | 10.60 |

`stored` and `explored` reproduce the previous study to the digit. The new column is
**`generated`** — the number of successor zones actually materialised (guard intersect, reset,
delay, invariant intersect, extrapolate, canonicalise) and then, in the overwhelming majority of
cases, thrown away because the target already stores a zone that covers them.

**The previous study's phrase "there is nothing at all to win there" applies to `stored` and
`explored`. It is false for `generated`:** arbiter15, one of the families declared to be at the
absolute floor on both reported axes, materialises **2 457 600 zones to store 278 528** — 88.7 %
of all DBM work is provably redundant. viking12 materialises 688 128 zones drawn from **120
distinct values**.

This is the surface the brief's phrase *"reduce the links we already found"* points at, and it is
untouched. Whether it can be converted into a *state* reduction rather than a *work* reduction is
open and is attacked in §4.

---

## 2. Liveness: the floor does not hold, and here the engine LOSES to UPPAAL badly

`leads_to_net` builds the reachable graph with **exact zone equality** as node identity by default
(`src/netreach.cpp:2686`: `lt_nosub = getenv("PTCTLMC_LTSUB") == nullptr`). Subsumption is
implemented behind that flag with the Herbreteau–Srivathsan–Walukiewicz / Laarman restriction
(a subsumption edge may not witness a cycle), and the source comment states plainly why it is not
the default:

> the restriction is sound but not provably complete — an absorbed newcomer is never expanded,
> so a witness that only exists below it is not looked for.

### 2a. A liveness query that is SATISFIED forces exhaustive search; the violated ones do not

Violated liveness queries stop almost immediately and measure nothing:

| model | query | verdict | explored |
|---|---|---|---|
| fischer4 | `P1.req --> P1.cs`   | NOT satisfied | 8 |
| fischer4 | `P1.cs  --> P1.idle` | NOT satisfied | 48 |
| csma4    | `S0.Send --> S0.Wait`| NOT satisfied | 8 |
| phil4    | `P0.Eat --> P0.Think`| NOT satisfied | 16 |
| arbiter4 | `C1.want --> C1.cs`  | NOT satisfied | 2 |

`arbiter`'s `C1.cs --> C1.idle` is **satisfied** and is therefore the measurement instrument used
below. All runs in the next table completed.

### 2b. First measurement, and why it had to be redone

The naive reading of the CLI's `explored` number is **wrong for liveness** and is recorded here so
the mistake is not repeated. `leads_to_net` does not run one search. It runs `search(use_alu=true)`
and then, if that finds no witness, **races** two lanes that decide the same proposition — lane A
proves the model deadlock-free (which retroactively validates the aLU run), lane B redoes the
search with exact inclusion (`src/netreach.cpp:3230-3260`). Both lanes call `search` with the *same*
`ReachStats*`, so the CLI figure is a sum over whichever lanes ran, and it varies with which lane
wins. The first numbers taken here (arbiter4 8192, arbiter5 136204) are such sums.

Two consequences:

- All liveness measurements below use `PTCTLMC_NORACE=1 PTCTLMC_LTDEBUG=1` and read the
  per-search line `[lt] FINAL nodes=… discrete-keys=… zones/dk=…`, which is unambiguous.
- **The racing portfolio the brief rules out is already in the engine, on the liveness path**, and
  it is load-bearing (the source records fischer7 180420 for lane A vs 30994 for lane B, and
  arbiter6 28591 vs 66393 — neither lane dominates). Any new liveness algorithm must be compared
  against the *lane* it replaces, not against the raced total.

### 2c. The measured gap, cleanly

Query `C1.cs --> C1.idle` (satisfied, therefore exhaustive). All runs completed.

| model | query | reach floor | liveness graph nodes | discrete keys | **zones per discrete state** | UPPAAL stored | UPPAAL / floor | PTC / UPPAAL |
|---|---|---|---|---|---|---|---|---|
| arbiter4 | `C1.cs --> C1.idle`        | 48  | 4 096  | 48  | **85.3**  | 297   | 6.2×  | 13.8× worse |
| arbiter5 | `C1.cs --> C1.idle`        | 112 | 68 102 | 112 | **608.1** | 1 671 | 14.9× | 40.8× worse |
| arbiter6 | `C1.cs --> C1.idle`        | 256 | INCOMPLETE (killed at ~15 min) | — | — | — | — | — |
| csma4    | `Bus.Collided --> Bus.Idle`| 104 | 3 341  | 104 | **32.1**  | 401   | 3.9×  | 8.3× worse |
| csma5    | `Bus.Collided --> Bus.Idle`| 272 | 60 271 | 272 | **221.6** | 2 491 | 9.2×  | 24.2× worse |

Two independent families, same shape. The reachability engine stores **1.00** zones per discrete
state on both and ties UPPAAL to the digit; liveness stores 32–608 of them. The ratio grows fast
with instance size — arbiter 85.3 → 608.1 and csma 32.1 → 221.6, roughly 7× per added process.
UPPAAL's own liveness grows too, but far more slowly (arbiter 6.2× → 14.9×, csma 3.9× → 9.2×).
arbiter6 liveness did not finish in 15 minutes and was killed; its reachability is 256 states and
finishes instantly. That run is reported as INCOMPLETE and is not counted as anything.

Violated liveness queries are the opposite story and must not be conflated with the above: on
`viking4`, `phil4`, `phil6`, `fischer4` the engine returns NOT satisfied after a handful of nodes,
matching verifyta's verdict every time. The engine is already excellent there. **Every number in
this document is about SATISFIED liveness queries, which is where the exhaustive cost lives.**

Three readings, all consequential:

1. **The engine's liveness stores 13.8× more per discrete state than UPPAAL on arbiter4 and 40.8×
   more on arbiter5**, and the gap is widening. This is the largest measured deficit in the project
   and the exact mirror image of the reachability result.
2. **UPPAAL itself is 6.2×–14.9× above the reachability floor and also widening.** The headroom is
   therefore not "catch up with UPPAAL" — it is an order of magnitude that *neither* tool takes.
3. The cause is structural, not an implementation defect: node identity for liveness must be exact
   zone equality, because inclusion subsumption creates spurious cycles (F1/F2). The `LTSUB`
   restriction is sound but explicitly **not complete** (`src/netreach.cpp:2680`), so it cannot be
   the answer.

**Target, stated numerically: a sound and complete liveness procedure that stores O(1) zones per
reachable discrete state would beat UPPAAL by 6.2× on arbiter4 and 14.9× on arbiter5 on `stored`,
with the factor growing with instance size — while the reachability engine that produces that
graph is already at the proved floor.**

### 2c. Why this is the right target, stated against the brief's own constraints

The brief asks for an algorithm that explores or stores fewer symbolic states than UPPAAL on
`A[]` **as well as** liveness. On `A[]` the engine already ties or beats UPPAAL and is provably at
the floor, so "fewer than UPPAAL" there is capped at the csma/phil/single margins that already
exist. On liveness the engine is *behind* UPPAAL by 11–34×, UPPAAL is behind the reachability
floor by 15–36×, and **both ratios grow with size**. Every ruled-out technique in the brief
(symmetry, POR, subsumption variants, coarser abstraction, sweep-line, CEGAR, racing) was ruled
out by measurement *on the `A[]` queries*. None of those measurements constrains liveness.

**Target, stated numerically: a sound and complete liveness procedure whose explored count is
within a small constant of the reachability floor would beat UPPAAL by 14.8× on arbiter4 and
35.7× on arbiter5, with the factor increasing.**

---

## 3. Structural facts established so far (inputs to candidate generation)

- **F1.** Node identity for liveness must currently be exact zone equality, because inclusion
  subsumption creates *spurious cycles*: the diagnosed grc4 bug (`docs/.../liveness-subsumption-bug.md`)
  is a back-edge `504 → 94` where the real successor (`x0 >= 4`) is a strict subset of the stored
  node (`x0 >= 0`), so `intern()` returns the stored id and the loop closes on itself. The concrete
  system can do three laps, not infinitely many.
- **F2.** The bound that dies is the one extrapolation legitimately deletes. `Extra_LU+` drops
  `x0 <= 15` because `15 > L[x0] = 4`; what actually bounds the loop is the *growing lower bound*
  on `x0` (0, 5, 10, 15), and subsumption erases exactly that. So the liveness-critical
  information is a **monotone counter hidden in a lower bound**, not in the zone's shape.
- **F3.** The reachable discrete graph of every family is a single SCC covering 100 % of states
  (previous study). Any liveness filter that looks only at discrete cycles therefore fires never.
  This kills the "abstract-graph-as-cycle-filter" family before it is proposed — same reason
  sweep-line was vacuous.
- **F4.** Every clock is process-local; zero urgent and zero committed locations anywhere.
- **F5.** `generated / stored` is 2.7×–10.6× on every family including the three at the reachability
  floor (§1).
- **F6.** Distinct extrapolated zones are extremely few on some families: viking 120 over 241 662
  stored; phil4 **7** over 80; phil10 840 over 62 177.

---

## 4. Candidate log

Each candidate is stated, then attacked. A candidate is only promoted when it survives the attack
*with a number attached*. Candidates are numbered in the order they were generated, including the
ones that died.

### C1 — Backward greatest-fixpoint liveness on the subsumption-closed forward graph

**Status: PROPOSED, under adversarial attack. The key soundness step is written out in full
because it is the entire idea; if it is wrong the candidate is dead.**

#### The observation the candidate is built on

The prior work treats "subsumption destroys the information liveness needs" as the obstacle. Read
the grc4 diagnosis again (F1/F2) and it says something more specific:

- extrapolation is **not** the culprit. `Extra_LU+` drops the *upper* bound `x0 <= 15`, but the
  location invariant `x0 <= 15` is re-applied at every node, so the deadline is never actually
  lost.
- the culprit is **subsumption**, which merges the family `x0 >= 0`, `x0 >= 5`, `x0 >= 10`,
  `x0 >= 15` into its maximum `x0 >= 0`. What is destroyed is a *monotone progress measure*: how
  much of the deadline has already been consumed.

Now note what a **backward greatest fixpoint** computes. `nu W. Z ∩ Pre(W)` is, by definition, the
set of valuations from which arbitrarily many further `¬q` steps are possible. On the grc4 cycle it
converges to `x0 ∈ [0, 15]` minus the valuations too far along to complete another 5-unit lap, and
then to `∅`. **The greatest fixpoint recovers exactly the progress information that forward
subsumption erased** — and it does so while storing one object per discrete state, not 85–608.

That is the candidate in one sentence: *do not try to fix subsumption for liveness; keep the
subsumed graph at the floor, and recover the lost progress measure with a backward gfp over it.*

#### Mechanism (precise)

1. **Forward phase.** The existing parallel BFS with full aLU subsumption. Produces graph `G` with
   one maximal extrapolated zone `Z(v)` per reachable discrete state `v`. `|V| = |Disc|`.
   This is the proved floor and already ties UPPAAL.
2. **Restrict.** `Z_0(v) := Z(v)` if `v ⊨ ¬q`, else `∅`. (Legal because `DiscPred` forbids clock
   constraints — `p` and `q` are functions of the discrete key alone; the engine already relies on
   this for its LTSUB soundness argument.)
3. **Backward phase.** Chaotic parallel iteration to the greatest fixpoint

   ```
   W(v)  <-  W(v)  ∩  ⋃ { Pre_e(W(v'))  :  e = v -> v' in G }
   ```

   with `Pre_e(X) = ( ( X ∧ inv(v') ) free(resets(e)) ∧ guard(e) ∧ inv(v) ).down() ∧ inv(v)`.
   All primitives exist in HDBM: `Dbm::down()`, `Dbm::free(k)`, `Dbm::intersect`,
   `Federation` + `fed_subtract` for the union and the emptiness test.
   One `Federation` is stored per node. Iterate until no `W(v)` changes.
4. **Verdict.** The property is violated iff some reachable `v ⊨ p ∧ ¬q` has `W_∞(v) ≠ ∅`, or a
   deadlock / timelock witness fires (the existing local tests, which are already sound because
   they are properties of a single stored node).

#### Why parallelism is constitutive, not bolted on

Step 3 is a **monotone chaotic iteration**: the fixpoint is independent of the order in which
nodes are updated, so workers may update any node at any time, read stale neighbours, and still
converge to the same limit. There is no barrier, no stack, no SCC decomposition and **no DFS at
all**. Every existing liveness algorithm for this setting — NDFS, CNDFS, Tarjan, UFSCC, OWCTY,
MAP — is built on DFS or on repeated reachability, and DFS order is exactly what makes parallel
liveness hard. This candidate deletes the DFS rather than parallelising it.

#### Termination

`W` is decreasing, every iterate is a sub-federation of the extrapolated `Z(v)`, and extrapolated
zones have bounded integer constants, so the lattice of sub-federations with those bounds is
finite. Hence the iteration terminates. (Note this is *not* the usual objection to backward TCTL
fixpoints, which is that they explore unreachable regions — here `W ⊆ Z` confines the whole
computation to the forward-reachable, already-abstracted set.)

#### Soundness — the step that decides the candidate

`W_∞ ⊇ Ω`, where `Ω(v)` is the true set of valuations at `v` admitting an infinite `¬q` run: `Ω` is
a post-fixpoint of the same operator and `Ω ⊆ R ⊆ Z`, so `Ω ⊆ gfp`. Therefore
**`W_∞ = ∅` proves the property holds.** That direction is unconditional.

The other direction is the risk. `W_∞(v) ≠ ∅` might be witnessed by a valuation in `Z(v) \ R(v)`,
i.e. one that extrapolation added and that is not actually reachable. The intended argument is:

> `Extra_LU+`/aLU is contained in the downward closure of a **time-abstract simulation** `≼_LU`
> (Behrmann, Bouyer, Larsen, Pelánek, *Lower and upper bounds in zone-based abstractions of timed
> automata*, TACAS 2004 / STTT 2006). So every `ν ∈ Z(v)` is simulated by some *reachable*
> `ν' ∈ R(v)`. A simulation matches every move, and `p`/`q` depend only on the discrete state, so
> if `ν` has an infinite `¬q` run then `ν'` has one too — and `ν'` is reachable. The witness is
> therefore genuine.

**KNOWN HOLE, stated rather than hidden:** a *time-abstract* simulation matches moves but not
durations, so it need not preserve **time-divergence**. A divergent run from `ν` may be matched by
a Zeno run from `ν'`. This is precisely the published reason liveness is hard for timed automata,
and it is the one thing that must be settled before this candidate is worth implementing. The
standard remedy is an added divergence clock / strongly-non-Zeno construction; whether that is
needed here, and what it costs in states, is `NOT ESTABLISHED IN SOURCES` at the time of writing
and is being checked.

#### Expected factor, derived from the measured tables

Storage becomes one federation per reachable discrete state, i.e. the reachability floor, against
the measured liveness cost:

| model | now (zones/disc) | C1 (zones/disc) | UPPAAL stored | C1 vs UPPAAL |
|---|---|---|---|---|
| arbiter4 | 85.3  | 1 + federation width | 297   | **6.2×** fewer |
| arbiter5 | 608.1 | 1 + federation width | 1 671 | **14.9×** fewer |
| csma4    | 32.1  | 1 + federation width | 401   | **3.9×** fewer |

The caveat is honest and material: a *federation* per node is not a *zone* per node. If the gfp
federations stay narrow (1–3 zones) the factors above hold; if they blow up the candidate degrades
to the status quo. Federation width is therefore the number to measure first.

#### Smallest experiment that would refute it

Prototype step 3 only, offline, on the already-built arbiter4 graph, and compare the verdict on the
188-query differential set the project already uses. Two refutations:
- **any verdict disagreement with the exact engine** ⇒ the simulation argument or the divergence
  handling is wrong;
- **mean federation width > ~8 on arbiter5** ⇒ the storage win evaporates and the candidate is
  pointless even if sound.

#### Where it measures zero

- On **violated** queries. The existing engine already finds counterexamples within a handful of
  nodes (measured: phil6 9 nodes, fischer7 6 nodes) because witnesses sit near the initial state.
  C1 must run the whole forward phase first, so it is strictly worse there. C1 only pays on
  *satisfied* liveness queries — which is where all the time and all the states currently go.
- On **reachability**. C1 says nothing about `A[]` and cannot; that floor stands.

---

### C1 — BUILT AND MEASURED

A working prototype exists. It is **not** on any production path: `src/netreach.cpp` and
`src/bisim.cpp` are untouched, as required. The prototype is a *copy* of the engine plus one new
routine, built as a separate binary outside the repo tree:

```
<scratch>/proto/src/netreach_gfp.cpp   copy of src/netreach.cpp + leads_to_gfp()
<scratch>/proto/main_gfp.cpp           differential harness: runs BOTH engines, compares verdicts
<scratch>/proto/CMakeLists.txt         links every core source EXCEPT netreach.cpp
```

`gfp_proto <model.xml> <Proc.Loc> <Proc.Loc>` runs `p --> q` through the shipping
`leads_to_net` *and* through `leads_to_gfp`, and prints both verdicts. **Verdict disagreement is
the refutation this experiment was built to look for.**

#### The headline measurement

Satisfied query `C1.cs --> C1.idle`, all runs completed:

| | arbiter4 | arbiter5 |
|---|---|---|
| PTCTLMC today (exact-zone liveness graph) | 4 096 | 68 102 |
| UPPAAL `States stored` | 297 | 1 671 |
| UPPAAL `States explored` | 710 | 3 998 |
| **C1 prototype, stored** | **48** | **112** |
| reachability floor (`\|reachable discrete states\|`) | 48 | 112 |

**C1 lands exactly on the reachability floor — 1.00 zones per discrete state — and returns the
same verdict.** Against UPPAAL's `stored` that is **6.2× fewer on arbiter4 and 14.9× fewer on
arbiter5**, with the factor growing with instance size; against the current engine it is
**85× and 608×**.

Storage at the fixpoint, `PTCTLMC_GFPALU=1`:

| model | query | stored | disc keys | zones/disc | gfp federation width avg / max |
|---|---|---|---|---|---|
| arbiter4 | `C1.cs --> C1.idle`     | 48    | 48    | **1.00** | 1.00 / 1 |
| arbiter5 | `C1.cs --> C1.idle`     | 112   | 112   | **1.00** | 1.00 / 1 |
| fischer4 | `P1.req --> P1.wait`    | 222   | 220   | 1.01 | 1.00 / 1 |
| fischer5 | `P1.req --> P1.wait`    | 762   | 727   | 1.05 | 1.00 / 1 |
| fischer6 | `P1.req --> P1.wait`    | 2 694 | 2 378 | 1.13 | 1.00 / 1 |
| twolock4 | `P1.reqA --> P1.waitA`  | 286   | 284   | 1.01 | 1.00 / 1 |
| phil6    | `P0.Eat --> P0.Dropped` | 813   | 728   | 1.12 | 1.00 / 1 |

The one number the spec said had to be measured first — **federation width — came out at 1.00
(max 1)**. The gfp does not blow up into wide unions on these models, so the storage win is real
rather than traded away.

#### What the differential testing found — three genuine defects, all in the prototype

This is the part worth keeping, because the failures were informative rather than incidental.

1. **Missing "let time pass forever" witness.** First sweep: 6 agree / 78 disagree, *all* in the
   safe direction (prototype said SATISFIED where the reference said violated). Cause: a maximal
   `¬q` run need take **no discrete transition at all** — it can sit in a `¬q` location and let
   time diverge. The fixpoint had no case for it. Adding the reference's own `time_diverges` test
   as an end condition took the sweep to 94 agree / 2 disagree.
2. **Missing pre-action delay in `Pre`.** A stored zone is delay-closed *as a set*, so the forward
   operator may write `Z ∧ g`; but an individual valuation may have to let time pass before the
   guard opens. Omitting that `down()` shrank the enabling region.
3. **THE IMPORTANT ONE — a per-valuation stuck test is unsound under extrapolation.** On arbiter4
   the prototype reported a violation of `C1.cs --> C1.idle` witnessed by the zone
   `x0, x1, x2, x3 > 4`. Both verifyta and the reference engine say the property holds. Those
   valuations are ones `Extra_LU+` *invents* (U = 4) and the system never reaches.

Defect 3 is not a coding slip; it is the candidate's stated hole, located precisely:

> `Extra_LU+`/aLU adds valuations that are **LU-simulated by** a reachable one, so an added point
> can do **at most** what the reachable point can. Therefore
> *"this point has an infinite `¬q` run"* transfers to the reachable point — **sound**;
> *"this point is stuck"* does **not** transfer — **unsound**.

**So the simulation argument in the soundness sketch above is correct for the cycle witness and
wrong for the deadlock witness.** The fix is to evaluate the stopping conditions whole-zone, which
is exactly the convention the reference engine already uses (`adj(v).empty() || time_diverges(v)`),
and which inherits its documented incompleteness on *partial* deadlocks
(`src/netreach.cpp:2680`) — an incompleteness that is not introduced here and is identical on both
sides of the comparison. With that change the sweep reached **130 agree / 4 disagree**; the 4 are
phil `P0.Eat --> {P0.Think, P0.HoldL}`, all in the safe direction, and are still open.

#### Why aLU subsumption is legitimate here — and it is what buys the floor

Plain zone inclusion keeps a much larger antichain than the reachability engine does: 11.33
zones/disc on arbiter4 and 45.29 on arbiter5, which beats the current engine 7.5×/13.4× but is
still 1.8×/3.0× **worse** than UPPAAL. Switching the forward phase to the aLU simulation is what
lands on 1.00. It is sound for this algorithm for a reason that is specific to the cycle witness:
when a newcomer `S` is dropped because a stored `W` aLU-simulates it, every behaviour of `S` is a
behaviour of `W`, and `W` is itself a genuinely reachable symbolic state. So the fixpoint over `W`
finds an infinite `¬q` run iff one exists from `S` or from `W` — no witness is lost and none is
invented. This is the same asymmetry as defect 3, used in the direction where it holds.

#### Status and what remains

- **Established by measurement:** the algorithm reaches the reachability floor on a satisfied
  liveness query, with federation width 1, and agrees with the reference on 130 of 134 queries.
- **Open:** the 4 phil disagreements; whether the whole-zone stopping conditions can be made
  complete without leaving the floor; the time-divergence/Zeno question for the cycle witness,
  which the arbiter/fischer/twolock/phil evidence does not yet settle; a parallel implementation
  of the chaotic iteration (the prototype's fixpoint loop is sequential — the parallelism claim is
  so far an argument about the algorithm's *shape*, not a measurement).
- **Not yet checked:** prior art. Backward greatest-fixpoint TCTL is classical (Henzinger,
  Nicollin, Sifakis, Yovine 1994; Kronos). The specific combination — gfp confined to the
  aLU-subsumed forward-reachable graph, with the observation that subsumption edges are harmless
  backward though fatal forward — is what would have to be new. `NOT ESTABLISHED IN SOURCES` yet.

---

### C2 — Asynchronous staleness deferral for `A[]` reachability

The reachability side. The prior study proved `stored` is at a hard floor, but left `explored`
57–72 % above its own floor on fischer/twolock, showed a uniform delay is a provable no-op, and
named **one undecided question as deciding the whole idea**:

> is the retiring zone *causally independent* of the retired one? If the bigger zone at `q` arrives
> via a path that runs *through* `q`'s own earlier expansion, then parking `q` parks its own
> retirer … and the policy degenerates to a timeout with pure loss.

#### That question is now answered — measured, not argued

`gfp_proto <model> --retire-causality` runs a level-synchronous BFS with aLU subsumption in which
every node records its BFS parent and birth level. On every retirement it walks the parent chain of
the state that produced the enlarging successor and asks whether the retiree lies on it.

| model | stored | expansions | cross-level retirements (the tax) | **INDEPENDENT** | dependent | gap histogram |
|---|---|---|---|---|---|---|
| fischer4 | 222   | 270    | 48    | **48 (100 %)**   | 0 | 0:12  4:48 |
| fischer5 | 762   | 1 012  | 250   | **250 (100 %)**  | 0 | 0:110 4:250 |
| fischer6 | 2 694 | 3 774  | 1 080 | **1080 (100 %)** | 0 | 0:630 4:1080 |
| fischer7 | 9 798 | 14 012 | 4 214 | **4214 (100 %)** | 0 | 0:2912 4:4214 |
| twolock5 | 922   | 1 172  | 250   | **250 (100 %)**  | 0 | 0:110 6:250 |
| twolock6 | 3 078 | 4 158  | 1 080 | **1080 (100 %)** | 0 | 0:630 6:1080 |
| twolock7 | 10 694| 14 908 | 4 214 | **4214 (100 %)** | 0 | 0:2912 6:4214 |
| phil6/8/10 | — | — | **0** | — | — | 0 only |
| arbiter5 | 112 | 112 | **0** | — | — | (no retirements at all) |

**Answer: on fischer and twolock the retirer is independent of the retiree in 100 % of cases, at
every size measured.** The gap histogram independently reproduces the prior study's signature — a
single constant gap, **4 on fischer and 6 on twolock**, with no tail, and zero retirements on
arbiter — which is good evidence the probe is measuring the same phenomenon.

Two limits, stated rather than buried:

- The probe's BFS is faithful but **not bit-identical to the engine**: it has no convex-union merge,
  so its absolute counts differ (its phil10 gives 94 776 / 94 776, which is exactly *UPPAAL's*
  phil10 figure, while the engine's merge reaches 62 177). On phil the engine and the probe
  therefore disagree about *where* retirements land: the probe sees **zero** cross-level
  retirements on phil6/8/10 while the prior study measured a decaying cascade on phil12. The
  fischer/twolock result is unaffected — there the engine, UPPAAL and the probe all agree.
- The BFS tree records **one** parent per node, so "independent" means *the shortest discovery path
  to the retirer does not pass through the retiree*. That is existence of an independent path, which
  is exactly what the deferral policy needs; it is not a statement about every path.

#### The negative that changes the design: there is no cheap oracle

The prior study framed the fix as needing "an oracle that names, per state, the ones about to be
superseded". Measured on fischer5, retirement rate per `(process, location)`:

```
0.217  48/221   Pi.idle       0.394  76/193   Pi.req       0.380 126/332  Pi.wait
```

uniform across all five symmetric processes, and `Pi.cs` never retires. **No location-vector
feature predicts retirement** (32.8 % of nodes are retired; no feature separates them). A
zone-level predictor would cost what it saves. So the oracle formulation is dead.

#### The reformulation — observe instead of predict, which is why parallelism is constitutive

The oracle is not needed. Replace *prediction* with *observation*:

> Park a popped state `q` instead of expanding it. Let other workers run other branches. If `q`'s
> zone grows while parked, discard the parked expansion — the tax is avoided outright. Expand `q`
> only once it has stayed quiescent while the frontier advanced `g` levels elsewhere, and
> unconditionally when the frontier would otherwise be empty (which keeps it terminating).

Under a **level-synchronous** schedule this is provably worthless: §4c of the prior study shows a
uniform delay delays the retirer strictly more than the retiree (`g·(W+1) < W` is false for
`g ≥ 1`). Under an **asynchronous, barrier-free** frontier it is not, and the reason is exactly the
measurement above: the retirer's discovery path does not run through the parked state, so parking
`q` does not stall its own retirer. **The 100 %-independence result is the licence for the
asynchrony, and the asynchrony is the whole mechanism** — this is the one place in the project
where extra cores buy a state-count reduction that no sequential schedule can have.

- **Metric moved:** `explored` only. `stored` is at the proved floor and C2 does not touch it.
- **Expected factor:** up to the full cross-level retirement count — 41.7 % of expansions on
  fischer10 and 40.8 % on twolock10 (prior study's figures), 28.6 % on fischer6 by this probe.
- **Where it measures zero:** arbiter, viking, csma (no retirements at all), and phil on this
  probe's BFS. It is a two-family result and must be reported as one.
- **Smallest refuting experiment:** implement parking on the existing frontier with a staleness
  counter and measure `explored` on fischer6/7. If it does not fall by ≈30 % the mechanism is wrong.
  A second refutation: if wall-clock rises more than the expansion count falls, the parked-state
  bookkeeping costs more than the tax.

---

---

## 4bis. Prior art — most of the first framing was killed

An adversarial prior-art search was run against C1's own claims. It killed the majority of them.
Recorded here because the surviving claim is narrower and far more defensible.

**Not novel, must be cited rather than claimed:**

| claim | prior art |
|---|---|
| `EG` / fair-`EG` as a greatest fixpoint; the fair-cycle hull | Emerson & Lei, LICS 1986; Somenzi/Ravi/Bloem FMCAD 2002 prove EL hull-optimal |
| `p --> q` violated iff `EF(p ∧ EG¬q)` | elementary CTL duality; it is how UPPAAL defines the operator |
| **restricting CTL fixpoints (incl. gfp) to the forward-reachable set** | the **default** in VIS (`model_check -d 1`), a flag in NuSMV (`-f`), and Moon/Jang/Hachtel/Somenzi/Yuan/Pixley, ICCAD 1998 |
| **chaotic/asynchronous monotone iteration as the route to parallel MC, vs. DFS** | Cousot 1977 (asynchronous iterative methods); **Cohen, Namjoshi, Sa'ar, Zuck, Kisyova, HVC 2010 §3.1** says this explicitly, citing Cousot; the DFS-is-sequential half is Reif 1985 and the framing of Barnat et al., *Handbook of Parallel Constraint Reasoning* 2018 |
| "reachability first, then a shrinking fixpoint over R, in parallel" | that is **OWCTY** (Fisler/Fraer/Kamhi/Vardi/Yang TACAS 2001), Alg. 11 of Barnat et al. 2018 |

So §"why parallelism is constitutive" in C1 must be written as *applying* Cousot/Cohen et al., not
as making that argument. That has been corrected in the paper.

**What survives, and it is specific to timed automata:**

> The reason liveness is hard for timed automata is that the subsumption which makes reachability
> efficient is unsound for Büchi emptiness — it closes cycles the concrete system cannot run
> (Tripakis; Herbreteau et al. FSTTCS 2016; Laarman et al. CAV 2013). Every published response
> either *weakens* subsumption (sound, not complete — what this engine implements) or *abandons* it
> (the 32–608× blow-up measured in §2c).
>
> **That obstacle is an artefact of searching FORWARD.** Backward, the redirected edge is harmless,
> because the target's value enters only as an upper bound and over-approximating an upper bound
> preserves `Ω ⊆ W`. Hence liveness can be decided on the **fully aLU-subsumed** zone graph — at the
> reachability floor, 1.00 zones per discrete state.

The untimed symbolic literature cannot have addressed this: BDD reachability has no notion of a
subsumption edge, so the question does not arise there. The timed literature has the notion but
approaches liveness through DFS-shaped emptiness checking, where the answer really is negative.

The search found **no work combining** (a) zone-based timed liveness, (b) a chaotically iterated
fixpoint, and (c) restriction to the already-subsumed forward-reachable zone graph.
That combination plus the asymmetry theorem is the contribution. Two of the workflow's independent
generator agents converged on the same algorithm from different lenses, which is corroboration that
it is the natural move once the asymmetry is seen — not that it is published.

---

## 4ter. C1 final validation

Differential suite: 164 leads-to queries over fischer4/5/6, twolock4/5, arbiter4/5, viking4,
phil4/6, grc4 — every ordered pair of locations of one representative process — run through both
the shipping engine and the prototype.

```
AGREE=164   DISAGREE=0   CAPPED=0   ERROR/TIMEOUT=0
```

(That is the FINAL sweep, after broadcast support and the larger federation cap. The earlier
136/0 sweep excluded csma and flagged four grc4 runs unusable; both exclusions are now lifted.)

Not counted, and reported rather than hidden:

- **4 `grc4` runs hit the prototype's federation-width cap (256)** and aborted as UNUSABLE rather
  than returning a verdict. grc4 is also the model with the highest zones/disc (18.32). This is the
  clearest place the algorithm is not yet known to hold up.
- **6 `csma` runs refused**: the prototype does not model broadcast channels and exits with a
  diagnostic instead of exploring a truncated graph. **csma liveness is therefore untested**, though
  it is one of the two families that motivate the work.
- **arbiter6 did not terminate** under the *reference* engine in 15 min and was killed. INCOMPLETE.

Reaching 136/0 required one further fix beyond those already logged: **the active-clock reduction
must be inverted inside `Pre`.** `reset_inactive` pins clocks no guard will read again to 0; they
are in no edge's reset set, so a naive `Pre` never frees them and silently demands they be 0 at the
source too. That over-constrains `Pre`, collapses the fixpoint, and **loses real counterexamples** —
measured on phil4/6, grc4 and arbiter4/5, all confirmed against `verifyta`. Freeing them at the top
of `Pre` is the exact inverse and is sound because those clocks are dead.

This also **corrects an earlier claim in this document**: the arbiter false violation from the
per-valuation stuck test was caused by the active-clock reduction breaking delay-closure (the
engine's own `state_has_deadlock` documents that hazard), **not** by LU extrapolation inventing
unreachable points. The abstraction-asymmetry theorem still holds as an argument, but there is no
measured instance of it, and the paper now says so.

---

---

## 4quater. C2 is DEAD — measured, not argued

The "observe, don't predict" reformulation was implemented and it saves **exactly zero**.

`gfp_proto <model> --defer D` runs a **barrier-free global FIFO** (an asynchronous frontier
serialised into one interleaving). Queue entries are `(node, version-at-push, park-count)`; a pop
whose version is stale is dropped (that is the intended saving), a pop with `park-count < D` is
re-queued with the count incremented (that is the parking), otherwise the node is expanded.

| model | level-sync | D=0 | D=1 | D=2 | D=4 | D=8 | D=16 |
|---|---|---|---|---|---|---|---|
| fischer5 | 1 012 | 990 | 990 | 990 | 990 | 990 | 990 |
| fischer6 | 3 774 | 3 514 | 3 514 | 3 514 | 3 514 | 3 514 | 3 514 |
| twolock5 | 1 172 | 1 150 | 1 150 | 1 150 | 1 150 | 1 150 | 1 150 |
| twolock6 | 4 158 | 3 898 | 3 898 | 3 898 | 3 898 | 3 898 | 3 898 |

The columns are not close, they are **identical**. Parking raised re-queue operations from 0 to
56 224 on fischer6 and changed the expansion count by nothing.

**Why, and the theorem is more general than the prior study stated:** a *uniform* parking policy is
a no-op under **any** schedule, not merely a level-synchronous one. Parking every popped node by the
same budget applies the same shift to the retiree and to its retirer, so their relative order is
unchanged. Removing the barrier does not help — **the barrier was never what made `g(W+1) < W` true;
uniformity was.** My reformulation silently reintroduced uniformity: "wait a fixed amount for every
node" is a uniform delay by another name.

So the `explored` axis on `A[]` is now closed on both routes:

- **uniform deferral** — no-op, proved and now measured at six budgets on four models;
- **selective deferral** — needs a per-state oracle, and no location-vector predictor exists.

The causal-independence result (100%) stands and is worth keeping, because it removes the objection
that was thought to be fatal — but removing it turns out not to be sufficient. **There is one
algorithm in this project, not two.**

Incidental, not claimed: the barrier-free FIFO expands 2.2–6.9% fewer states than level-synchronous
BFS (1012→990, 3774→3514) while also storing fewer. The two searches compute different antichains,
so it is not a like-for-like comparison.

---

## 4quinquies. Three open items from iteration 2, now closed

**csma is no longer untested.** Broadcast channels are implemented in the prototype (the tag is now
a *list* of participating edges, which unifies local / binary-sync / broadcast in one `Pre`):

| model | prototype stored | zones/disc | engine (exact) | UPPAAL stored | vs UPPAAL |
|---|---|---|---|---|---|
| csma4 | **128** | 1.23 | 3 341 | 401 | **3.1× fewer** |
| csma5 | **352** | 1.29 | 60 271 | 2 491 | **7.1× fewer** |

**grc4's federation cap was the limit, not the algorithm.** Raising the cap from 256 to 4096 makes
grc4 converge at federation width **1.00** and agree. The four previously-UNUSABLE runs now return
verdicts.

**The Zeno question is answered — and the answer is not what the paper assumed.** A purpose-built
discriminator `zeno.xml`:

```
Start --(x:=0)--> Loop     Loop invariant x<=1
Loop  --[x<1]--> Loop      (self-loop, NO reset -> can only fire infinitely often in bounded time)
Loop  --[x>=1]--> Done
```

No deadlock (at x=1 the exit is enabled) and time cannot diverge in `Loop`. So **every
time-divergent maximal run reaches `Done`**, and only Zeno runs avoid it. A divergence-aware
semantics must answer *satisfied*; a Zeno-permissive one *violated*.

```
verifyta -u -S0        : NOT satisfied
reference leads_to_net : NOT satisfied
gfp prototype          : NOT satisfied      AGREE=YES
```

**All three are Zeno-permissive.** So the divergence gap is *not* a defect the algorithm introduces
relative to its oracle — it faithfully reproduces the semantics UPPAAL and the engine implement. It
is a limit of all three, and a divergence-aware semantics would need the standard fresh-clock
construction added to each. The paper now says this instead of listing divergence as the most likely
way the algorithm is wrong.

---

## 4sexies. ZEUS — prior art the literature search missed, surfaced by our own memory

`MEMORY.md` carried a warning the web search did not reproduce:

> ZEUS (Braberman, Olivero, Schapachnik) is a **parallel backward TA zone engine**, so "nobody has
> parallelised backward TA search" is false.

Checked, and it is correct. ZEUS is a **distributed** timed model checker built on Kronos doing
**backwards** computation over timed automata (PDMC @ CONCUR 2002; FMSD 2006; plus companion work on
workload prediction). It partitions by assigning each automaton location to one processor.

So neither *backward* nor *parallel* is novel here, and the paper now says so explicitly. What
survives is narrower: Kronos-lineage backward analysis computes over the state space directly and
has **no forward phase with an LU/aLU simulation abstraction** — that machinery postdates it and
belongs to the forward, UPPAAL-lineage engines. The claim is about the *interaction* between that
abstraction and the direction of the fixpoint, which cannot arise in a tool that has no such
abstraction.

**NOT ESTABLISHED IN SOURCES:** whether ZEUS or its successors combine a forward pass and, crucially,
whether any confine the backward fixpoint to a *subsumption-closed* forward-reachable graph. The
primary texts returned HTTP 403 / paywall. Secondary summaries mention combining forward and backward
analysis but none states the confining set is subsumption-closed. **This is the single most important
open question about the novelty claim and must be settled from primary sources before publication.**

The same memory also warned that the "63× liveness gap" framing is a strawman because it compares
against an internal unsound configuration. That objection does **not** apply to the headline here:
the comparison is against **UPPAAL's own `States stored`** on the same liveness query — 48 vs 297,
112 vs 1671, 128 vs 401, 352 vs 2491 — which is like-for-like liveness against liveness.

---

## 4septies. Scope: the same fixpoint decides `E[]` and `A<>`, not just leads-to

The algorithm is not leads-to-specific. It computes a greatest fixpoint, so it covers the
**greatest-fixpoint half** of the logic:

| query | fixpoint | status |
|---|---|---|
| `E<> φ` / `A[] φ` | least (plain reachability) | already at the floor — the gfp adds nothing |
| `E[] φ` | `νW. φ ∧ (stop ∨ Pre W)` | **decided, 51/51 agreement** |
| `A<> φ` | `¬ E[] ¬φ` | immediate from the above |
| `p --> q` | `EF(p ∧ EG ¬q)` | **decided, 164/164 agreement** |

**The trap is the verdict rule, not the fixpoint.** The first `E[]` attempt scored 20/34 and every
failure was an UNSOUND "satisfied". Reason: `p --> q` asks whether *some reachable* `p ∧ ¬q` state
has a non-empty gfp — that is `EF(p ∧ EG¬q)`. `E[] φ` asks whether **the initial state** does — that
is `EG φ` at `s0`. Reusing the first rule for the second computes `EF EG φ`, strictly weaker, and it
said "satisfied" for every predicate not holding initially (contradicted by verifyta on fischer4
`E[] P1.wait`, arbiter4 `E[] C1.want`, phil4 `E[] P0.HoldL`, viking4 `E[] Viking1.safe`).
Restricting the check to the root node gives 51/51.

---

## 4octies. The `generated` surface: one fix works, the other is dead

`generated/stored` is 2.85–10.60 on **every** family, including the three at the absolute floor on
`stored` and `explored`. Each generated zone costs a full Post (guard, resets, invariant, delay,
invariant, extrapolate, O(n³) closure) and is then usually thrown away. `gfp_proto --genprofile`
measures the two reuse factors that decide the two candidate fixes:

- **R_A** — backward-image filter (the direction-optimizing/Beamer analogue the brief pushed
  hardest): cache `Pre_e(W)` per *(target node, edge label)*, test the source zone against it in
  O(n²) instead of computing Post in O(n³). Pays only if the same (target, label) is hit by many
  different sources.
- **R_B** — zone-alphabet memoisation: cache `Post_e` per *(source zone VALUE, edge label)*. Pays
  only if the same (zone value, label) recurs.

| model | discarded | distinct source zone values | R_A | R_B |
|---|---|---|---|---|
| arbiter10 | 84.0 % | 6 144 / 6 144 stored | **1.00** | **1.00** |
| csma8     | 79.8 % | 5 888 / 5 888 | **1.07** | **1.00** |
| fischer7  | 58.9 % | 6 266 / 7 737 | 1.54 | 1.13 |
| twolock7  | 60.7 % | 7 106 / 8 633 | 1.45 | 1.12 |
| viking10  | 62.9 % | **65** / 45 566 | 1.00 | **293.3** |
| phil8     | 77.7 % | **193** / 8 486 | 1.11 | **27.8** |

**R_A ≈ 1 everywhere ⇒ the backward-image filter is DEAD.** Each (target, edge-label) pair is hit
essentially once, so computing `Pre` once per pair costs exactly what computing `Post` costs. This
is a measured negative on the brief's own headline suggestion (direction-optimizing BFS): the
bottom-up step has no reuse to exploit in a zone graph, because the vertex fan-in is spread across
distinct edge labels rather than concentrated on one.

**R_B is 27.8×–293× on viking and phil, and 1.00–1.13 elsewhere.** The families split cleanly:

- **zone-degenerate** (viking 65 distinct zone values over 45 566 states; phil 193 over 8 486):
  memoising the transition `(zone-id, edge) → zone-id` replaces almost every Post with a hash
  lookup. viking10 generates 122 880 zones from **419** distinct (value, edge) pairs.
- **zone-rich** (arbiter, csma, fischer, twolock): every stored state has its own zone value, so
  there is nothing to memoise, and R_A rules out the other cache too.

So 59–84 % of all Post work is provably redundant everywhere, but **only the zone-degenerate half
is reachable by caching**. On the zone-rich half the only remaining lever is making Post itself
cheaper (incremental O(k·n²) closure instead of a full O(n³) re-canonicalisation) — engineering,
not algorithm.

Note the engine already exploits the zone alphabet for *storage* (`--gpu-intern`, "large win on
models with few distinct zones, e.g. viking; neutral elsewhere"). What is not exploited is the
*transition*: the interner dedups the zones it stores but still recomputes each one.

---

## 4nonies. Transition memoisation: BUILT and MEASURED (time and memory)

Implemented in the prototype (`--memo 0|1`). The forward search is split into a **discrete half**
(enumerate transitions: integer guards, sync shape, target locations) and a **zone half** (the DBM
pipeline). The memo elides the zone half.

**Key design — this is where it lives or dies.** The key must capture everything `Post` reads and
nothing more, and must be O(1):

`(source zone id, transition CONTENT signature, target location SIGNATURE)`

- Keying on the target's **discrete id** instead of its signature gives **0 hits on every family** —
  measured. `Post` reads the target locations only through its combined invariant, its
  urgent/committed flags, and its active-clock mask; many location vectors share one such
  signature, and that sharing is the whole effect.
- Keying the transition by **(proc, edge index)** instead of by guard/reset **content** loses ~2×,
  because symmetric processes' edges are interchangeable: viking10 gives 293× by identity and
  **561×** by content.

### Result — `stored` is bit-identical to the baseline on every family (pure memoisation)

| model | hit rate | Posts base → memo | reduction | wall base → memo | speed | peak RSS base → memo |
|---|---|---|---|---|---|---|
| viking12 | 100.0 % | 688 128 → **222**    | **3100×** | 751 → 650 ms | 1.16× | 246 → 278 MB (+13 %) |
| viking10 |  99.8 % | 122 880 → **219**    | **561×**  | 140 → 96 ms  | 1.46× | — |
| phil12   |  99.0 % | 8 596 512 → **83 017** | **103×** | 19 628 → 10 144 ms | **1.93×** | 1311 → 1454 MB (+11 %) |
| phil10   |  98.4 % | 603 880 → 9 761      | 61.9×     | 902 → 532 ms | 1.70× | — |
| arbiter10 |  0.0 % | 38 400 → 38 400      | 1.00×     | 43 → 60 ms   | **0.73× (slower)** | 7.4 → 17.0 MB (**+130 %**) |
| fischer7 |   3.9 % | 42 609 → 40 944      | 1.04×     | 16 → 31 ms   | **0.53× (slower)** | 6.8 → 18.8 MB (**+176 %**) |

### Reading it honestly

1. **A 103× cut in Post gives only 1.93× wall-clock.** Amdahl: `Post` is roughly half of phil12's
   runtime. The rest is transition enumeration, aLU subsumption tests and passed-set hashing. So
   the ceiling for this technique alone is ~2×, not 100×, and further gains have to come from the
   *other* half.
2. **Where reuse is absent the memo is a straight loss** — 1.4–1.9× slower and 2.3–2.8× more
   memory, for nothing. It must be adaptive: sample the hit rate over the first few thousand
   expansions and drop the tables if it is below ~20 %.
3. **Memory is +11–13 % where it pays**, and the dominant term is not the memo (3.8 MB on phil12)
   but the per-location-vector signature cache (**66.9 MB**). That is a prototype artefact: the
   engine already computes `disc_key` per state for the passed set, so the signature id belongs
   *in the bucket*, which makes that term free. Zone interning (18 MB) is also already paid by the
   existing `--gpu-intern`.

So the realistic engine-side figure is ~2× wall-clock on the zone-degenerate families at
near-zero extra memory, and nothing (turned off) elsewhere.

---

## 5. Final position

| | C1 (liveness) | C2 (reachability `explored`) |
|---|---|---|
| status | **ALIVE, implemented, validated** | **DEAD, measured** |
| metric | `stored` and `explored` | `explored` only |
| vs UPPAAL | **6.2× / 14.9× fewer stored** (arbiter4/5); 3.1× / 7.1× (csma4/5); factor grows with size | n/a — `A[]` already ties |
| vs current engine | **85× / 608×** (arbiter), 26× / 171× (csma) | — |
| parallelism | chaotic iteration, order-independent, no DFS anywhere | irrelevant: uniform deferral is a no-op under *any* schedule |
| implemented? | yes | yes — and it saves exactly 0 at every budget |
| where it is zero | violated queries; all of `A[]` | everywhere |
| open | federation width unbounded in theory; partial-deadlock flag over-approximates; parallel speedup unmeasured | only **selective** deferral remains, and no cheap predictor is known |

**One algorithm, not two.** The reachability side produced a measured negative that closes the
`explored` axis for uniform policies, plus one result worth keeping (100% causal independence,
which removes the objection previously thought fatal — just not sufficiently).

