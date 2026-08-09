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

## 5. W1 — G-simulation (design decision only)

Detailed algorithm specification is deferred to a companion document, pending literature
extraction now in progress. What is already settled, and why:

**The result being imported.** The `G`-simulation (Gastin–Mukherjee–Srivathsan) is, restricted
to diagonal-free automata, **strictly coarser than the LU-simulation, with a simulation test
that remains quadratic in the number of clocks** (survey p.12). Strictly coarser means
strictly more zones pruned, on every model, with no benchmark tuning. It additionally stays
sound in the presence of diagonal constraints, where this engine currently abandons `a_LU`
entirely and falls back to plain inclusion.

**The mechanism.** A constraint *map* `G(q)` — a set of constraints per state, computed as a
least fixpoint propagating guards backwards through resets — replaces the two integers per
clock that `lbound_loc` / `ubound_loc` hold today.

**The open design question, and where our contribution sits.** `G(q)` is defined per state of
a single automaton. For a network, `q` is the location vector, so the fixpoint is over the
product and costs what the search costs. This engine already sidesteps the analogous problem
for LU by storing per-process, per-location bounds and taking a max over processes
([netreach.cpp:768]). Whether the same compositional move is sound for constraint *sets* is
the question the literature extraction must answer.

**Decision: static-compositional first, lazy second.** Lazy refinement (CAV'13) makes the
abstraction *finer* over time, so the covered region **shrinks**, which invalidates
`thm:coverage` and makes the shared-lock `probe_kill` unsound — the one theorem the parallel
design rests on. Static-compositional `G` does not move at all during the search and preserves
the theorem untouched. So: land static-compositional, measure it, and treat reconciling lazy
refinement with a concurrently-read antichain as the research contribution rather than a task.

**Touchpoints.** `dbm_is_alu_le` → new predicate; `state_bounds` → constraint-map lookup;
`NetPassed::add` steps (2) and (3); `NetPassed::probe_kill`. Behind `--sim=alu|g`, default
`alu`.

---

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
