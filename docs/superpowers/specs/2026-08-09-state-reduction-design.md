# State reduction beyond the zone floor — W0, W2, W1

Branch: `mostoptimized`. Oracle: UPPAAL 5.0.0 `verifyta`
(`/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta`).

## 1. The thesis

Two independent facts agree, and together they set the whole direction.

**Measured, on this engine.** Running `PTCTLMC_STATS=1 ptctlmc --backend omp` across the
benchmark families gives `stored-zones / discrete-states`:

| model | discrete states | stored zones | ratio |
|---|---|---|---|
| fischer10 | 260 998 | 260 998 | 1.000 |
| fischer11 | 837 949 | 837 949 | 1.000 |
| arbiter14 | 131 072 | 131 072 | 1.000 |
| arbiter16 | 589 824 | 589 824 | 1.000 |
| viking12 | 241 662 | 241 662 | 1.000 |
| twolock10 | 271 238 | 271 238 | 1.000 |
| csma10 | 22 784 | 22 784 | 1.000 |
| phil11 | 177 146 | 189 559 | 1.070 |
| phil12 | 531 440 | 576 434 | 1.085 |
| single6 | 1 | 2 535 | 2535 |

On seven of the ten instances above — five of the eight families — the antichain is a
**singleton at every discrete state**. The convex-union merge has already extracted every
zone-level win available. Those five families include every one for which
`doc_1/sections/07-uppaal.tex` records a state count *identical to `verifyta`*: fischer12,
viking17, twolock13.

**Stated, by the authors of the abstractions we use.** Bouyer–Gastin–Herbreteau–Sankur–
Srivathsan, *Zone-based verification of timed automata: extrapolations, simulations and what
next?* (arXiv 2207.07479), §8:

> "by applying extrapolation/simulation the number of zones per control state is usually
> small. The major bottleneck is the large number of control states q appearing in the
> forward computation."

**Consequence.** `|reachable dk|` is a hard floor for the current search, and we sit on it.
No zone-level technique — union/federation coverage, coarser extrapolation, stronger merges —
can remove a single state on those models; a coarser abstraction can only make *more*
discrete states reachable, never fewer. Beating `verifyta` on state count therefore requires
attacking the *discrete* dimension (W2, and later W3) or making the *simulation* strictly
coarser than `a_LU` (W1).

One positioning fact worth keeping in front of us: `verifyta` is **extrapolation**-based
(survey §7.1); this engine is **simulation**-based. We are already one generation ahead on
that axis and W1 goes two.

## 2. Scope and order

**W0 → W2 → W1.** W3 (partial-order reduction) is explicitly out of scope for this document
and gets its own after W1 lands.

Each workstream lands behind its own flag, default **off**, so the baseline stays byte-comparable
throughout. That is what makes the final evaluation table publishable rather than anecdotal.

---

## 3. W0 — the differential oracle gate

Nothing else may land before this. W2 and W1 both change which states are stored, and both
change verdicts when wrong; a wall-clock benchmark cannot detect that.

### What exists

`benchmarks/bench_run.py` already drives the oracle: it runs `verifyta -u <xml> <q>`, parses
`States stored : N` / `States explored : N`, and computes a `parity` column. That is a
*performance* harness with a correctness column bolted on. W0 inverts the emphasis.

### What we build

`benchmarks/verify_parity.py` — correctness first, state counts second, timing not at all.

For every (model, query) pair in `benchmarks/models` × `benchmarks/queries`:

1. Run `verifyta -u` → verdict, states stored, states explored.
2. Run the **baseline** binary (built from `main`) at `--backend omp -p1` with `PTCTLMC_STATS=1`
   → verdict, `discrete-states`, `stored-zones`, `distinct-zones`, `generated`, `explored`.
3. Run the **branch** binary with the workstream flag on, same settings.

Then:

- **Verdict parity is a hard gate.** Any three-way disagreement is a FAIL, printed with the
  model, query and all three verdicts. Non-zero exit. This is the loop's stop condition.
- **State deltas are the result.** Emit CSV plus a markdown table: branch vs baseline and
  branch vs `verifyta`, as absolute counts and percentages, for discrete states, stored zones
  and explored states.

### Determinism

All PTCTLMC runs at **one thread**. `thm:determinism` says the verdict is deterministic but
the stored count is not, because the convex-union merge is order-sensitive; at `-p1` the
counts are reproducible and therefore comparable across commits. Multi-thread runs belong in
the performance harness, not here.

### Invariant builds

A subset of models additionally runs under a `PTCTLMC_DEBUG_KNOBS` build with
`PTCTLMC_CHECKINV=1` and `PTCTLMC_CHECKMERGE=1`. These check the specific lemmas of
`sections/06-theory.tex` rather than the end-to-end verdict, which is the only independent
evidence we have — `verifyta` agreement establishes agreement, not correctness.

### Exit criteria

- Runs the full model × query matrix, one command, non-zero exit on any verdict mismatch.
- Reproduces the state counts already recorded in `07-uppaal.tex` for phil12, fischer12,
  viking17, twolock13, single6 — if it cannot, the harness is wrong, not the table.
- Baseline snapshot committed as `benchmarks/baseline-mostoptimized.csv`.

---

## 4. W2 — discrete liveness reduction

### The idea

`dk` packs the location vector **and every integer variable** ([netreach.cpp:826]). Nothing
asks whether a variable's value can still be observed. Two states that differ only in a
variable no process will ever read again are semantically identical and are today stored as
two distinct states.

This is precisely the *clock* active-variable reduction the engine already performs
(`Network::compute_active` + `reset_inactive`), applied to the discrete dimension. Clocks get
it; integers do not.

### The analysis

Mirror `compute_active` exactly. Per process `p`, per location `l`, per variable `v`, compute
by backwards fixpoint:

```
ireads[p][l][v] = true  iff  process p, starting at l, may READ v
                             before p itself writes v
```

Seed: `v` occurs in the `int_guard` of an edge out of `l`, or in the right-hand side of an
`int_updates` entry on such an edge. Propagate: `ireads[p][l][v] |= ireads[p][tgt][v]` for
every edge `l → tgt` that does **not** write `v`.

`v` is **dead at location vector L** iff `ireads[p][L[p]][v]` is false for every process `p`,
and `v` is not read by the query.

### Soundness

The claim is that if `v` is dead at `L`, then two configurations differing only in `v`'s
value have identical futures with respect to the query, so they may be identified.

Consider any run from `L`. Any read of `v` in that run is performed by some process `q` from
its current location. Because `ireads[q][L[q]][v]` is false, every path of `q` from `L[q]`
writes `v` before reading it. Hence that read is preceded, *on `q`'s own path*, by a write of
`v`. Interleaved writes by other processes only change which write the read observes; in no
case does the read observe the entry value. Therefore no read in any run from `L` depends on
`v`'s value at `L`. ∎

The argument is per-process and therefore compositional — no product construction — which is
what makes it cheap. Note it is an *exact* liveness statement for reads, not an
over-approximation: the `∀p` quantifier is what makes it sound, and weakening it to `∃p`
would not be.

### Two components

**(a) Static elimination.** A variable read by no guard, no invariant, no update RHS and not
by the query is dead *everywhere*. Drop it from the `dk` bit-packing entirely. This narrows
the key, which also relieves the 63-bit width limit that currently rejects wide networks
([netreach.cpp:838]).

**(c) Range narrowing.** Found by the first W0 gate run, and it turns W2 from a possible
zero into a capability gain. `train0`/`train1` **crash today**:

```
error: reachable_net: discrete state needs 91 bits but the packed key holds 63
  4 processes + 5 int vars. Widest fields:
    16 bits  int Gate.list[0] [-32768,32767]   (x4)
    16 bits  int Gate.len
```

The model declares `int list[N+1]` and `int len` with no explicit range, so the parser assigns
the full 16-bit default and five variables consume 80 of the 63 available bits. The values
those variables actually take are train ids in `0..N-1` and a queue length in `0..N` — three
bits each. `verifyta` verifies `train0` in 55 stored states; we do not verify it at all.

So: an interval fixpoint over the initial values, the constant assignments, and the update
expressions, joined per variable to an over-approximation of its reachable range, and pack
`dk` from the *inferred* range rather than the declared one. Over-approximation is what keeps
it sound — no reachable value may be excluded — and where inference cannot bound a variable it
falls back to the declared range, i.e. to today's behaviour.

Note this is independent of the liveness analysis and pays even when every variable is live:
`(a)` removes variables, `(c)` narrows the ones that remain, and either can shrink `dk` alone.

**(b) Dynamic canonicalisation.** For a variable dead only at some location vectors: before
computing `disc_key`, overwrite `ints[v] := intvars[v].lo` wherever `v` is dead at the
successor's location vector. Identical in shape to `reset_inactive` overwriting dead clocks,
and placed at the same point in the successor path.

The bitmask trick applies verbatim: precompute `iread_mask[p][l]` as one word per location
when `|intvars| ≤ 63`, so the per-successor cost is `nprocs` OR-s and a popcount loop rather
than an `nprocs × nvars` scan.

### Conservative fallbacks

- Edges carrying `gen_guard` / `gen_update` (general expressions, arrays, function calls) are
  not analysed. Every variable reachable by any general expression is marked live everywhere.
  Same bail-out discipline as `symmetry_group`.
- Every variable named in the query — `NetTarget::int_cond` or `DiscPred::ints` — is live
  everywhere. This is the `qmask` analogue and it is not optional.

### Flag and ablation

`--slice` enables it; default off. `PTCTLMC_NOSLICE` disables it inside a `--slice` build for
A/B measurement, under `PTCTLMC_DEBUG_KNOBS` so the release build pays nothing.

### Honest expectation

On models where every variable is genuinely live everywhere — Fischer's `id` is read by a
guard out of `idle`, so it is live at every location — this yields **exactly zero**. The win
is on models carrying counters, mode flags, or ids consumed at one point and then dead. W0
measures it; we do not guess, and we report a zero as a zero.

---

## 5. W1 — contracting constraint maps (a new algorithm, not a port)

We are not porting TChecker's G-simulation. We import the *abstraction* — which is published,
proven, and better than what we run today — and put a **new refinement algorithm** on top of it
that only makes sense because our search is parallel. The G-simulation is the substrate; the
contribution is the direction we move in and why we are allowed to.

### 5.1 The observation

Refining a constraint map has a direction, and the two directions are not equally compatible
with a parallel antichain.

Write `G*` for the least-fixpoint constraint map (the standard one) and `⪯_G` for the induced
simulation. The relevant monotonicity — which the literature extraction must confirm before any
code is written — is that **a larger constraint set gives a finer simulation**: more constraints
to preserve means fewer zones subsumed. `G*` is therefore the *coarsest sound* choice, and every
`G ⊇ G*` is still sound but prunes less.

That gives two possible refinement strategies:

| direction | constraint set | simulation | covered region | parallel probe |
|---|---|---|---|---|
| **CAV'13 lazy abstraction** | starts empty, **grows** | gets **finer** | **shrinks** | **unsound** |
| **this proposal** | starts over-approximate, **shrinks** toward `G*` | gets **coarser** | **grows** | **sound** |

`thm:coverage` says the covered region at each discrete state only ever grows, and
`cor:probe` turns that into the licence for the shared-lock `probe_kill` — a kill decided at
time `t` is still valid at any later `t'`. Lazy abstraction moves the wrong way: it makes the
abstraction finer over time, so a state that was covered can stop being covered, and every
concurrent read decided against the old, coarser abstraction becomes retroactively wrong. That
is why CAV'13's algorithm cannot simply be dropped into this engine.

Contraction moves the right way. Shrinking `G` makes the simulation coarser, so the covered
region only grows, so `thm:coverage` survives untouched and the probe stays sound with no new
synchronisation.

### 5.2 The algorithm

1. **Start from a cheap sound over-approximation.** For a network, take the union of the
   per-process constraint maps `⋃_p G_p(L[p])`. Computable statically, per process, with no
   product construction — the same compositional trick the engine already uses for LU bounds
   ([netreach.cpp:768]). Sound because it is a superset of `G*`; weak because it is too big.
2. **Contract in the background.** Spare cores — the ones a sequential tool does not have —
   run a refiner that removes constraints from `G(L)` at individual location vectors as it
   establishes they cannot matter there, moving each entry down toward `G*(L)`.
3. **Explorers read whatever strength is currently published.** No barrier, no re-checking of
   stored zones: a zone stored under a larger `G` stays valid and simply becomes *more*
   effective at subsuming newcomers once `G` shrinks. (This is item (ii) the extraction must
   confirm.)

The result is an abstraction that gets **coarser as the search runs**, funded entirely by
parallelism, with the soundness argument carried by a theorem the codebase already has.

### 5.3 Why this is a contribution and not an optimisation

- It is the **dual** of the published lazy-abstraction direction, and the choice of direction is
  forced by a property of *parallel* search. A sequential tool has no reason to prefer it —
  which is presumably why it has not been done.
- It converts the compositional over-approximation from a *compromise* (we cannot afford the
  product fixpoint) into a *starting point* (we begin sound and cheap, then improve), which
  removes the main objection to compositional constraint maps for networks.
- It composes with, rather than competes against, the exact convex-union merge: merging reduces
  how many zones a chain holds, contraction increases how many newcomers each one kills.

### 5.4 What must be established before writing code

The extraction is scoped to answer exactly these, and W1 does not start until it has:

- **(A)** Is every `G ⊇ G*` sound, and strictly finer? (If the direction is the other way round,
  the whole scheme inverts and must be rethought.)
- **(B)** Does contraction preserve soundness at every intermediate point, and do already-stored
  zones stay valid across a shrink?
- **(C)** Can `G` be shrunk at one location vector independently, or does the fixpoint couple
  them so a local shrink forces global recomputation? This decides whether the refiner runs
  per-stripe or needs a barrier — i.e. whether the parallelism claim survives contact.

If (A) or (B) fails, W1 falls back to **static compositional `G` only** — still strictly better
than `a_LU`, still worth having, with no contraction and no research claim.

### 5.5 Settled points

**The result being imported.** The `G`-simulation (Gastin–Mukherjee–Srivathsan) is, restricted
to diagonal-free automata, **strictly coarser than the LU-simulation, with a simulation test
that remains quadratic in the number of clocks** (survey p.12). Strictly coarser means
strictly more zones pruned, on every model, with no benchmark tuning. It additionally stays
sound in the presence of diagonal constraints, where this engine currently abandons `a_LU`
entirely and falls back to plain inclusion.

**The mechanism.** A constraint *map* `G(q)` — a set of constraints per state, computed as a
least fixpoint propagating guards backwards through resets — replaces the two integers per
clock that `lbound_loc` / `ubound_loc` hold today.

**Why networks force the issue.** `G(q)` is defined per state of a single automaton. For a
network, `q` is the location vector, so the fixpoint is over the product and costs what the
search costs. This engine already sidesteps the analogous problem for LU by storing
per-process, per-location bounds and taking a max over processes ([netreach.cpp:768]). The
compositional union of per-process constraint maps is the same move for constraint *sets*, and
§5.2 step 1 makes it the starting point of the contraction rather than a permanent compromise.

**Staging.** Static compositional `G` lands first and is measured on its own — it is a complete,
useful result even if contraction never happens, and it is the baseline contraction has to beat.
Contraction lands second, gated on (A)/(B)/(C).

**Touchpoints.** `dbm_is_alu_le` → new predicate; `state_bounds` → constraint-map lookup;
`NetPassed::add` steps (2) and (3); `NetPassed::probe_kill`. Behind `--sim=alu|g`, default
`alu`.

---

## 5b. MEASURED RESULTS (2026-08-10)

All numbers at one thread, `verifyta` 5.0.0 as oracle, 33 (model, query) pairs at N <= 7.

### W2(c) `--narrow` — behaviour-neutral, and it unblocks a family

Stored-zone counts **identical to baseline on all 33 pairs**. That is the invariant that had to
hold: narrowing changes how `dk` is packed, never which states are reachable, so any movement
would have meant the interval analysis was unsound. Nothing moved.

`train0`/`train1` go from "rejected, 91 bits needed" to running. The analysis infers
`list[0..3] ∈ [0,2]`, which is exactly the model's own `typedef int[0,N-1] id_t` with `N = 3`.

### W2(a)/(b) `--slice` — measures exactly ZERO

Every integer variable is live everywhere in every benchmark family, so no state is ever
canonicalised. This is the outcome §4 predicted in advance and it is recorded, not buried. The
analysis is correct and costs nothing when it fires on nothing; it is retained because it is a
precondition for any model that does carry dead discrete state.

### W1 `--sim=g` — large where it matters, invisible everywhere else

With the adaptive widening switch disabled (`PTCTLMC_ADAPT=0`) so the simulation is isolated:

| model | a_LU | G | verifyta |
|---|---|---|---|
| single4 | 22 | 24 | 34 |
| single5 | 203 | **130** | 181 |
| single6 | 1975 | **375** | 1116 |
| phil, fischer, csma, twolock, viking, arbiter | — | unchanged to the digit | — |

`single` was the **only** family where this engine stored more than `verifyta` (1959 vs 1116,
+75.5%). At 375 it stores 66% *fewer*. Everything else is untouched, which is expected: those
families sit at one zone per discrete state, and no simulation however coarse can go below that.

Two honest qualifications. `single4` gets slightly *worse* (22 -> 24) — a coarser subsumption
changes which zones are retired and which convex unions become available, so counts can move
either way at small N. And the size of the win is surprising given [THESIS] p. 36 calls the
diagonal-free improvement "not drastic"; `single` is N independent clocks under one location
with strict guards, which is precisely the shape where the strictness bit compounds.

### What this says about the programme

The floor argument of §1 held up exactly. W1 and W2 both left the five at-the-floor families
untouched, because nothing operating on zones or on `dk` width can remove a state that is one
zone at one reachable discrete state. What they did instead is (a) close the one family where we
*lost* to `verifyta`, and (b) make a family verifiable that could not run at all. Neither is the
headline "beat UPPAAL everywhere" result; both are real.

## 6. Risks

- **W2 yields zero on the current suite.** Plausible. It is still correct, still narrows `dk`,
  and still lifts the 63-bit ceiling. Reported honestly either way.
- **Compositional `G` turns out unsound.** Then W1 reduces to per-location `G` computed lazily,
  which collides with `thm:coverage`, and W1 becomes research rather than implementation. The
  extraction is scoped to answer exactly this before any code is written.
- **`verifyta` disagrees on a model we currently pass.** Treated as our bug until proven
  otherwise; the oracle is the reference tool.
- **State-count comparisons are meaningless if taken at >1 thread.** Enforced by W0 running at
  `-p1`.

## 7. Out of scope

W3 (partial-order reduction), TChecker as a second oracle, GPU backends, and liveness queries
(`E[] φ`, `p --> q`) — the latter have no subsumption at all today and are a separate,
larger opportunity noted in `sections/10-limits.tex`.
