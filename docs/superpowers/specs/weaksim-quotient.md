# Weak timed simulation with invariants, and the environment-quotient measurement

Status: IN PROGRESS — written incrementally, sections are appended as work completes.
Scope: (A) add location invariants to the timed (bi)simulation checker; (B) measure whether
the *environment* of a Fischer-style network collapses under weak (bi)simulation, i.e. whether
an automatically-built `Ehat` can stay bounded as the number of environment processes grows.

Nothing in this document is committed to git; the tree is left dirty on purpose.

---

## Job A — invariants in `src/bisim.cpp`

### A.1 Semantics implemented (write it down first)

Let `A`, `B` be diagonal-free timed automata with location invariants `Inv(l)` (a conjunction
of diagonal-free constraints, hence a **convex** zone).

**States.** `(l, v)` is a state only when `v ⊨ Inv(l)`. A location whose invariant is
unsatisfiable therefore has *no* states: it is unreachable, not "universally related".

**Transitions.**

* delay `(l,v) --d--> (l,v+d)` for `d ≥ 0` iff `v+d ⊨ Inv(l)`.
  Convexity of `Inv(l)` plus `v ⊨ Inv(l)` gives `v+d' ⊨ Inv(l)` for every `d' ∈ [0,d]`, so
  checking the endpoint is exactly checking the whole trajectory. This is why the zone
  operator `down` is still the right one.
* action `(l,v) --a--> (l', v[r:=0])` iff `v ⊨ g`, `v ⊨ Inv(l)` **and** `v[r:=0] ⊨ Inv(l')`.
  The target-invariant test is part of *enabledness*: an edge whose reset lands outside the
  target invariant is not a transition at all.

**Timed simulation `A ≼ B`** — a relation `R` on states with `((l0A,0),(l0B,0)) ∈ R` and

* delay: `(s,t) ∈ R`, `s --d--> s'` ⟹ `∃ t --d--> t'` with `(s',t') ∈ R`
  (*the same* `d`; `t` must be able to let that much time pass, so an invariant that stops
  time on the `B` side breaks the pair);
* action: `(s,t) ∈ R`, `s --a--> s'` ⟹ `∃ t --a--> t'` with `(s',t') ∈ R`.

**Weak** (`WeakSim` / `WeakBisim`): the action clause is relaxed to `t ⇒a⇒ t'`
(`τ* a τ*`, all `τ`s at **zero delay**) for observable `a`, and to `t ⇒ t'` (`τ*`, possibly
empty) when the challenger's action is `τ`. The **delay clause is unchanged** — a delay is
still matched by an equal delay with no interleaved `τ`.

> Consequence, stated up front: what the engine computes is the greatest relation satisfying
> the clauses above, which is *a* weak timed simulation, hence contained in the greatest weak
> timed simulation. `true` is sound. `false` may be a false negative for the fully general
> weak relation (where the responder may split a delay around `τ` steps). This was already
> true before invariants; invariants do not make it worse. All Job-B numbers are therefore
> **lower bounds on collapse** (upper bounds on class counts).

### A.2 How the fixpoint changes

The engine keeps `R[(lA,lB)]` as a federation over the disjoint clock product `X_A ⊎ X_B`.
Write `InvA[lA]`, `InvB[lB]` for the invariants lifted into that product space (universe on
the other side's clocks). Four changes, each of which degenerates to the previous code when
every invariant is the universe:

1. **Seed.** `R[(lA,lB)] := InvA[lA] ∩ InvB[lB]` instead of the universe. The fixpoint only
   removes points, so `R ⊆ InvA ∩ InvB` is an invariant of the whole computation. That single
   fact is what makes the rest cheap.
2. **Delay-stability.** Old rule: `R -= down(¬R)`. New rule:

   ```
   avail := InvA[lA]                 (simulation: only A challenges with delays)
   avail := InvA[lA] ∪ InvB[lB]      (bisimulation: either side may challenge)
   R     -= down(avail \ R)
   ```

   Note there is no separate test for "B cannot follow the delay": because `R ⊆ InvB`, any
   point `v+d` with `v+d ⊨ InvA` but `v+d ⊭ InvB` is already in `InvA \ R`, so the single
   subtraction covers both failure modes — *A delays into a pair that is no longer related*
   and *A delays where B's invariant forbids time to pass*.
3. **Challenger enabledness.** Old: `en := guard`. New:
   `en := guard ∩ InvC[e.src] ∩ InvO[Lo] ∩ resetpre(InvC[e.dst], e.resets)`,
   where `resetpre(Z,r) = { v | v[r:=0] ∈ Z }` (pin the reset clocks to 0, intersect, free).
   Widening `en` beyond the true enabled set would delete related pairs that never had a
   challenge — an *unsound* answer in the "reports non-simulation" direction.
4. **Responder legs.** Every `edge_pre` over a responder edge `f` is intersected with
   `InvO[f.src]`. Target invariants need no explicit test: the backward computation starts
   from `R[...] ⊆ Inv` and the intersection at each source keeps `T[l] ⊆ InvO[l]` inductively,
   so a `τ` chain through a location whose invariant the valuation violates cannot contribute.
   Missing this would *enlarge* the responder set and keep pairs that should die.

Plus: the per-clock maximum-constant vector `M` used for extrapolation now also scans invariant
constants. `dbm_extrapolate` only relaxes bounds *strictly* above `le(M[i])`, so an invariant
`x <= 2` survives extrapolation exactly when `M[x] >= 2`; without this scan an invariant-only
constant would be silently widened to `∞`.

### A.3 API

`BAut` gains a trailing member (so existing brace-less field assignment in the tests is
unaffected):

```cpp
std::vector<std::vector<Constraint>> invariants;  // empty, or size nlocs
```

and a second entry point exposes the relation the refinement already computes, which Job B
needs:

```cpp
std::vector<Federation> bisim_relation(const BAut& A, const BAut& B, BisimKind,
                                       const std::set<int>& obs, bool* related);
```

indexed `lA * B.nlocs + lB`; membership of a product valuation in `rel[lA*nB+lB]` *is* the
relation.

The `--bisim` front-end spec (`app/main.cpp`) gains a matching directive, so the feature is
reachable from the CLI rather than only from C++:

```
inv <loc> ge|gt|le|lt <clock> <value>       # repeat to conjoin
```

**Degenerate initial state.** If the origin violates `InvA[l0A]` (or `InvB[l0B]`) the initial
state does not exist and `decide_bisim` returns `false`. That is a deliberate choice — an
automaton with no initial state is vacuously simulated by anything, but reporting `true` there
would hide a modelling error, and every model in Job B has `0 ⊨ Inv(l0)`.

### A.4 Result — DONE

`cmake --build build-mostopt -j8 && ./build-mostopt/tests/ptctlmc_bisim`

* Before: **49 passed, 0 failed**.
* After (4 new invariant cases × 6 relations, + reflexivity-with-invariants, + the
  origin-violating case, + relation/boolean agreement): **82 passed, 0 failed**, and the
  original 49 checks are byte-for-byte the same verdicts.

New cases, each hand-derived:

| # | shape | what it pins down |
|---|-------|-------------------|
| 8 | `a` vs `a` with `Inv(l0)=x<=1` on B | responder's invariant stops time ⇒ `A ⋠ B`, `B ≼ A` |
| 9 | `a` with `Inv(l0)=x<=1` on A vs `a[x<=1]` | challenger's invariant *bounds* the delays it may challenge with ⇒ `A ≼ B` |
| 10 | `a[x>=2]` into `l1` with `Inv(l1)=x<=1` vs no edges | target invariant kills the edge; the automaton is a bare clock |
| 11 | `a` vs `τ;τ;a` with `Inv(l1)=x<=1` | a responder τ-chain may not pass *through* a location whose invariant fails |
| invrefl | cyclic automaton, every location constrained, vs itself | identity is a bisimulation — all four kinds true |
| noinit | `Inv(l0)=x>=1`, no edges, vs itself | no initial state ⇒ `false` |

**Mutation check** (each fix disabled in turn, tests rerun — this is the evidence that the new
cases are not vacuous):

| mutation | failures |
|---|---|
| seed `R` with the universe instead of `InvA ∩ InvB` | 9 (cases 8, 9, noinit) |
| `avail := universe` in the delay rule | 10 (cases 8, 9, 11, **reflexivity**) |
| drop the target-invariant test from `enabled_zone` | 5 (cases 10, 11) |
| drop the source-invariant test from the τ-closure | 1 (case 11 `wA<=B`) |

The reflexivity failures under mutation 2 are the informative ones: getting `avail` wrong makes
an invariant-carrying automaton fail to be bisimilar *to itself*.

---

## Job B — does the environment of a Fischer network collapse?

Driver: `tests/env_quotient.cpp`, target `ptctlmc_envquot` (deliberately **not** a ctest — it is a
study and the large instances are long-running).

### B.1 Encoding of the shared integer `id`

`BAut` has no integer variables, so `id` is folded into the location as an explicit component
with its value abstracted to

```
d ∈ {FREE} ∪ {OWN_0 … OWN_{k-1}} ∪ {C}          (k+2 values)
```

**Justification that nothing C can observe is lost.** An environment process only ever *tests*
`id==0` and `id==own pid`, and only ever *writes* `id := own pid` or `id := 0`. Every value a
C process can write is, to an environment process, simply "not 0 and not mine", so all C-written
values may be collapsed into the single token `C` — this is the classical
{undefined, mine, someone-else's} abstraction (CMC; Abdulla et al., NJC 2007). The owner's
*identity* among the environment processes must be kept, because `wait --[id==pid]--> cs` is
enabled only for the owner; collapsing `OWN_i` would let a non-owner enter `cs` and would make
the model unsound (too permissive in the wrong place, and — worse — wrong about *which* process
may release the lock).

**The interface.** C touches `id` with exactly five operations, and they are the observable
alphabet:

| label | meaning | enabled | corresponds to |
|---|---|---|---|
| `C_RD0`  | read `id==0`     | `d==FREE` | C's `idle --[id==0]--> req` |
| `C_WR`   | write `id:=cpid` | always    | C's `req --[id:=pid]--> wait` (unguarded) |
| `C_RDEQ` | read `id==cpid`  | `d==C`    | C's `wait --[id==pid]--> cs` |
| `C_RDNE` | read `id!=cpid`  | `d!=C`    | C's `wait --[id!=pid]--> idle` |
| `C_WR0`  | write `id:=0`    | always    | C's `cs --[id:=0]--> idle` |

None carries a clock guard, which is right: C's clock lives in C, so the environment must
tolerate C touching `id` at any moment.

Three alphabets are measured, most abstract first:

* **V1 "C-view"** — only the five C operations. Every environment transition, *including* its own
  reads and writes of `id`, is τ. Rationale: C observes the environment only through the value of
  `id`, and it samples that value only with its own reads, which are all in the alphabet. This is
  the most abstraction that is still sound, so it is the alphabet with the best chance of
  collapsing.
* **V2 "writes observable"** — plus the environment's writes, all carrying **one** label (C cannot
  tell which environment process wrote). Reads stay internal: they do not change shared state.
* **V3 "literal"** — every read and write of `id` observable, which is the literal reading of
  "everything except reads/writes of `id` is internal". Because *every* edge of a Fischer process
  reads or writes `id`, V3 makes every edge observable and weak degenerates to strong. Reported
  because it is the literal instruction, not because it is the interesting one.

Model sizes (K = 2, matching `gen_fischer.py`'s default), reachable symbolic states from the
tool's own forward zone graph:

| k | locations | edges | reachable locations | symbolic states |
|---|---|---|---|---|
| 1 | 12 | 50 | 10 | 10 |
| 2 | 64 | 312 | 46 | 72 |
| 3 | 320 | 1792 | 203 | 960 |
| 4 | 1536 | 9728 | 876 | 16428 |

### B.2 What is counted

The engine returns `R[(l,l')]` over the doubled clock set, so "same behaviour" for two locations
of the same automaton is

```
l ~ l'   iff   ∃σ ∈ S_k .  D_σ ∩ Inv(l) ∩ Inv(l') ⊆ R[(l,l')]   and   D_{σ⁻¹} ∩ … ⊆ R[(l',l)]
D_σ = { (v,w) : w_{σ(i)} = v_i }
```

and the class count is the number of blocks of the equivalence generated by `~`, over the
**reachable** locations. The permutation is what lets "process 1 is waiting" merge with
"process 2 is waiting"; the engine's relation already contains every matching of the two clock
sets, the measurement only has to ask for the right one. Restricting to the invariants is not
cosmetic — valuations outside `Inv(l)` are not states, and demanding that they be related would
stop any two differently-constrained locations from ever merging.

This count is over *locations*, so it cannot see a collapse that needs different clock values on
the two sides. That is exactly why it is backed up by the cross-comparison in §B.4, which asks
the size-independence question directly and has no such blind spot.

### B.3 The flattened model is validated against Fischer's own correctness argument

`ptctlmc_envquot mutex 4 2` builds the flattened network with *no* free-floating C interface
(mode `ctx` with `nobs=0`, i.e. plain Fischer with N processes) and counts reachable locations
with two processes in `cs` — once with the real guard `wait --[x>K]--> cs`, once with it
weakened to `x>=K`:

| N | locations | edges | reachable locations | symbolic states (Extra_M) | 2-in-cs, `x>K` | 2-in-cs, `x>=K` |
|---|---|---|---|---|---|---|
| 2 |   48 |   80 |  18 |   29 | **0** |   2 |
| 3 |  256 |  624 |  65 |  348 | **0** |  30 |
| 4 | 1280 | 4096 | 220 | 5437 | **0** | 244 |

Mutual exclusion holds exactly when the strict inequality is there and fails when it is not.
Reproducing *that* dependency is a much stronger check on the encoding than matching a state
count: it needs the `req` invariant, the strict guard and the `id` encoding all to be right.

One number that must NOT be compared naively: this driver extrapolates with `Extra_M`, while
`src/netreach.cpp` uses `extrapolate_lu_plus`, which is coarser. The repo's engine reports 71 /
268 / 977 symbolic states for Fischer N = 3 / 4 / 5 against this driver's 348 / 5437. Both are
sound; they are different abstractions, and the counts above are only ever compared to each
other.

### B.4 Result 1 — minimising the environment ON ITS OWN buys nothing but symmetry

`ptctlmc_envquot quot k <alphabet> <weak|strong>`. All runs listed here **completed**; the
per-run self-bisimilarity sanity check (`E_k ~ E_k` must be true) passed in every case.

| k | reachable locations | **classes** | symmetry orbits | merges | fixpoint |
|---|---|---|---|---|---|
| 1 | 10 | **10** | 10 | 0 | 0.01 s |
| 2 | 46 | **27** | 27 | 19, *every one of size 2* | 21.6 s |
| 3 | 203 | (see §B.7) | 56 | | |
| 4 | 876 | (see §B.7) | 100 | | |

Two facts, both measured:

* **The class count equals the symmetry-orbit count exactly** at k = 1 and k = 2. Every merge the
  weak-bisimulation refinement finds is a permutation of identical environment processes and
  nothing else. Symmetry reduction is already in this repo (`include/ptctlmc/symmetry.hpp`), so
  the quotient adds **zero** new collapse on top of what is already available.
* **The count grows.** Orbit counts are 10, 27, 56, 100 for k = 1…4 — differences 17, 29, 44,
  second differences 12, 15: cubic growth, which is what a multiset-of-`k`-processes count over
  4 local states with one distinguished owner has to be.

**Answer to question 1 (one process template):** the template plus the `id` component has 12
locations, 10 of them reachable, and it collapses to **10** classes — no collapse at all.

**Answer to question 3 (weak vs strong):** *none of the collapse comes from abstracting internal
steps.* At k = 2 the class count is **27 in every variant**: weak/V1 (only C's operations
observable — the most abstraction that is sound), weak/V2, weak/V3, and strong. Same at k = 1
(10 everywhere). Two independent cross-checks agree here: V3-weak makes every edge observable and
must therefore equal strong, and it does (27 = 27, 0.38 s vs 0.29 s).

That is the negative half of the answer, and it is the same mechanism that killed
cross-discrete-state subsumption in this project: in a contention protocol every process's
location changes what everyone else can do, so no two of them are interchangeable except by
symmetry.

### B.5 Result 2 — and the environment on its own does not stabilise either

Class counting is a *location*-level measure and could in principle miss a collapse that needs
different clock values on the two sides. The cross-comparison has no such blind spot: it asks
directly whether the k-process environment is over-approximated by a **fixed** smaller one,
which is precisely the `Ehat` the lead needs. `A ≼ B` here means *B simulates A*, i.e. B is the
more permissive one and is a sound abstraction of A.

| relation | weak sim | weak bisim | time |
|---|---|---|---|
| `E_1 ≼ E_1` (sanity) | YES | YES | 0.01 s |
| `E_1 ≼ E_2` | YES | — | 1.27 s |
| `E_1 ≈ E_2` | — | no | 0.49 s |
| **`E_2 ≼ E_1`** | **no** | no | 0.11 / 0.33 s |
| **`E_3 ≼ E_1`** | **no** | no | 1.71 / 23.9 s |

So `E_1 ≼ E_2` but `E_2 ⋠ E_1`: the family is *strictly* increasing in permissiveness. No member
of it is an `Ehat` for all N.

The mechanism is concrete and unbounded in k. An environment process in `cs` may fire
`cs --[id:=0]--> idle` **unguarded**, so every process currently in `cs` is one *pending clear* of
`id` that can be spent at any instant. With k environment processes there can be up to k of them
banked, and they are observably distinct: C writes `id:=cpid`, the environment spends one pending
clear, C writes again, the environment spends the next. `E_1` can bank one, `E_2` two. Re-banking
takes more than K time units (a process must go `idle→req→wait→cs` with `x>K`), so the bank is a
genuine counter bounded only by k.

Note that this counter needs C to write `id` while environment processes sit in `cs` — something
a *genuine* Fischer C cannot do (it would already be a mutual-exclusion violation). It is
reachable here only because minimising a component on its own necessarily treats the other
side's actions as an unconstrained input. That is not a modelling slip; it is the price of
bisimulation-as-a-congruence, which is exactly what "build `Ehat` once and use it in every
context" asks for. It also points at where the remaining hope is, which is §B.6.

### B.6 Result 3 — with the context kept, the small system DOES over-approximate the big one

The `ctx` mode replaces the unconstrained C interface with `nobs` **genuine Fischer processes**
whose own edges carry the observable labels — so C can only touch `id` in the ways, and at the
times, a Fischer process can. Each C process additionally exposes its local state as an
observable self-loop, because the property is a predicate on C's *locations* and a simulation
preserves state predicates only when they are observable. `C||E_k` here is plain Fischer with
`nobs + k` processes, with `k` of them anonymised.

| nobs | relation | Fischer sizes | weak sim | weak bisim | time |
|---|---|---|---|---|---|
| 1 | `C‖E_0 ≼ C‖E_0` (sanity) | 1 vs 1 | YES | YES | 0.00 s |
| 1 | `C‖E_1 ≼ C‖E_0` | 2 vs 1 | no | no | 0.01 s |
| 1 | **`C‖E_2 ≼ C‖E_1`** | **3 vs 2** | **YES** | no | 123.0 / 18.6 s |
| 2 | `C‖E_1 ≼ C‖E_0` | 3 vs 2 | no | no | 7.7 / 0.9 s |

The `nobs=1` line is the one that matters: **a 3-process Fischer network is weakly timed-simulated
by a 2-process one**, once the process the property talks about is kept concrete and the other
two are anonymised. `≼` in that direction means the *smaller* system is the more permissive one,
which is exactly the `C ‖ Ehat` safe ⟹ `C ‖ E` safe abstraction the lead is built on.

So the picture splits cleanly:

* minimising `E` **on its own** (§B.4, §B.5) — collapses only by symmetry, and never stabilises;
* minimising `E` **in its context** (this section) — the system does start collapsing.

The `nobs=2` row is a warning against over-reading this. With two concrete processes (which is
what `A[] not (P1.cs && P2.cs)` actually needs) the 3-process network is *not* simulated by the
2-process one, so the cutoff — if there is one — sits higher than the one-concrete-process case
suggests, and the deciding run is `nobs=2, C‖E_2 ≼ C‖E_1` (Fischer 4 vs 3).

### B.7 What these numbers do and do not license

* **A `YES` is sound; a `no` may be conservative.** The engine matches a delay challenge with an
  *equal* delay and no interleaved τ (§A.1). Any relation it returns is therefore a genuine weak
  timed simulation — positives cannot be spurious on that account — but a `no` could in principle
  be a false negative for the fully general weak relation where the responder splits a delay
  around τ steps. Every negative in §B.5 should be read with that caveat. It does not weaken the
  §B.4 class counts in the direction that matters: those are *upper* bounds on collapse, and they
  already show the collapse is only symmetry.
* **Extrapolation.** The fixpoint extrapolates intermediate zones with `Extra_M` (pre-existing
  behaviour; this work only extended the constant vector to cover invariants). Extrapolation
  widens, so in principle it could make a responder set too large and turn a `no` into a `YES`.
  Nothing here re-derives its soundness for bisimulation.
* **The class count is a location-level measure** (§B.2), so it can only *over*-count: it merges
  two locations only when the whole invariant-restricted diagonal is related, and misses any
  collapse that would need different clock values on the two sides. Over-counting is the
  **unsafe** direction for a negative conclusion, so §B.4 on its own does not close the question.
  What closes it is §B.5, which has no such blind spot: it asks directly whether a fixed smaller
  automaton over-approximates the k-process one, and the answer is that the family is strictly
  increasing.
* **The `ctx` positive is a cutoff observation, not a construction.** `C‖E_2 ≼ C‖E_1` is
  established by a simulation check between two *whole* systems, which costs at least as much as
  verifying the larger one. To turn it into a speed-up you need the induction to close — some
  argument that `C‖E_{k+1} ≼ C‖E_k` for all k — and that is a parameterized-verification /
  assume-guarantee result, not the component minimisation the lead was written around.

### B.8 Verdict

**The lead as written — "compute the quotient of E and use it as `Ehat`" — is dead, and the
measurement says so with the same mechanism that killed cross-discrete-state subsumption here.**

1. One process template collapses to **10 classes from 10 reachable locations: zero merges**.
2. The environment of k processes collapses to **exactly its symmetry orbits and nothing more** —
   10, 27 measured at k = 1, 2 against orbit counts 10, 27, with every merge at k = 2 a
   two-element process swap. Orbit counts continue 56, 100 at k = 3, 4, i.e. **cubic growth**. It
   does not stop growing, and the collapse it does achieve is already available from
   `include/ptctlmc/symmetry.hpp`.
3. **None** of the collapse comes from abstracting internal steps: weak and strong give the
   identical count (27 at k = 2, 10 at k = 1) under every one of the three alphabets, including
   the most abstract one where every environment transition is τ.
4. Independently of the counting, the family is **strictly increasing in permissiveness**:
   `E_1 ≼ E_2` but `E_2 ⋠ E_1` and `E_3 ⋠ E_1`, with an identified unbounded mechanism (banked
   pending clears of `id`, one per process sitting in `cs`). No fixed member of the family
   over-approximates all the others, so there is no `Ehat` to find *by this route*.

**What survives.** §B.6 measured one thing that is not dead: with the property's process kept
concrete and the rest anonymised, the 3-process Fischer network *is* weakly timed-simulated by
the 2-process one. The collapse the lead wanted exists — but it lives in the *context*, not in
the component, so it is a parameterized/cutoff result requiring an assume-guarantee argument,
not the "minimise `E` once, reuse for every N" construction the lead was costed as. Before
spending on that, note the `nobs=2` row: with the two processes the actual property names kept
concrete, even Fischer 3 ≼ Fischer 2 fails.

---

## Files touched (nothing committed)

| file | change |
|---|---|
| `include/ptctlmc/bisim.hpp` | `BAut::invariants`; `bisim_relation(...)`; semantics in the header comment |
| `src/bisim.cpp` | invariant support: seeded relation, delay rule, challenger enabledness, responder-leg invariants, invariant constants in `M`; `decide`→`run` returning the relation |
| `tests/bisim_tests.cpp` | 4 invariant cases, reflexivity-with-invariants, no-initial-state, relation/boolean agreement (49 → 82 checks) |
| `app/main.cpp` | `inv <loc> <op> <clock> <value>` directive in the `--bisim` spec |
| `tests/env_quotient.cpp` | **new** — the Job-B measurement driver |
| `tests/CMakeLists.txt` | `ptctlmc_envquot` target (no `add_test`) |
| `docs/superpowers/specs/weaksim-quotient.md` | this file |

`src/netreach.cpp` is also dirty in this tree; that is another agent's liveness work and was not
touched here.
