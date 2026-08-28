# Known limitations

What this engine does *not* do, and why. Each entry gives the mechanism, not just the symptom, so
that a surprising result can be matched against the cause.

Ordered by how easily the limitation goes unnoticed. None of them is silent: each is
reported, refused or visible in the output.

---

## 1. An urgent-channel synchronisation may not be clock-guarded

`urgent chan` is implemented. What is **rejected**, with a diagnostic naming the process, edge and
channel, is a clock guard on an edge that synchronises over one:

```
error: clock guard on an edge synchronising over the urgent channel 'go'
       (process Q, C -> D). Urgent-channel synchronisation may not be clock-guarded
```

**Mechanism.** An urgent channel stops time exactly while a synchronisation on it is enabled, so
enabledness is consulted at every state, by both engines, on the hot successor path. The
implementation rests on it being decidable from the **discrete** state alone — locations plus the
integer valuation — which is what lets `urgent_sync_enabled` fold into `frozen_flags` beside the
urgent-*location* test and cost a single predictable branch on models that declare no urgent
channel. Integer guards on the participants are fully supported, since they are part of the
discrete state.

A **clock** guard would break that. Enabledness would become a property of the zone, and worse, one
that can switch on partway through a delay: correct handling would have to clip the delay at the
instant the guard opens. Two things make that expensive rather than merely fiddly. The clip is a
zone subtraction whose result is in general **non-convex**, so a single transition would yield
several successor zones. And the exact operator is not the obvious one — the set reachable by delay
from `Z` while avoiding an urgent region `U` is *not* `Z↑ \ strictfuture(U ∩ Z↑)`. That formula
under-approximates whenever `Z` is thick in the time direction and `U` is a slab strictly inside it:
for `Z = [0,5]`, `U = {x = 2}` it yields `[0,5]`, while the true answer is `[0,∞)`, because a
valuation starting at `x = 3` never meets `U` at all and may delay forever. Under-approximating the
reachable set is the **unsafe** direction — it turns `A[]` violations invisible — so no
approximation is applied and the model is refused instead.

**Why this costs nothing in practice.** UPPAAL imposes exactly this restriction on its own
modelling language, so no model UPPAAL accepts is refused here.

**Workaround.** Move the clock guard onto a separate, non-urgent edge, or push the timing condition
into a location invariant.

**Related routing.** A model declaring an urgent channel is always verified by the network engine,
even when it is diagonal and one process wide. The single-automaton engine in `src/reach.cpp` drops
synchronisation labels entirely and would let time pass where the model forbids it, so the
diagonal preference in `app/main.cpp` yields to urgency (see §2 for what that costs).

**Test models.** `tests/models/urgent-chan/` — `urgent-basic.xml`, `urgent-broadcast.xml`,
`urgent-intguard.xml`, `urgent-live.xml`, plus `urgent-clip.xml`, which is the rejected case.

---

## 2. Diagonal networks fall back to exact covering

A network containing any constraint comparing two clocks (`x - y ~ c`) is explored with a
strictly weaker covering relation than a diagonal-free one.

**Mechanism.** `network_has_diagonal` (`src/netreach.cpp`) scans every invariant and edge guard for
a constraint with both indices non-zero. When it fires, the engine sets `exact_sub` and switches the
extrapolation to the diagonal-preserving `ExNoDiag`. It must: the LU family of abstractions is
unsound in the presence of diagonal constraints (Bouyer, *Forward analysis of updatable timed
automata*, FMSD 24(3), 2004), so `a_LU` subsumption cannot be used at all. Covering degrades from
simulation-based subsumption to plain zone inclusion, which prunes far less.

**Consequence.** Time and memory only — every verdict remains correct. The stored-state count can be
much larger than for a comparable diagonal-free model, and `reach_engine` additionally imposes a
500 000-state cap on diagonal networks, throwing rather than running unbounded when a model has
diagonal clocks that never stabilise.

**Status.** A G-simulation route that covers the diagonal case exists on branch `gsim-diagonal`. It
is not merged: as recorded there it costs about **5x** in time, which is a worse trade than the
weaker covering it replaces for every model measured. Not re-measured for this document.

---

## 3. `p --> q` predicates may constrain only the discrete state

`E[] phi` and `A<> phi` accept **arbitrary boolean** predicates -- `&&`, `||`, `!`, parentheses,
any nesting, over locations, integers and clocks mixed. `p --> q` still accepts neither clocks nor
boolean structure. `E<>` and `A[]` accept clock constraints normally.

**Mechanism.** A plain conjunction is convex and is adjoined to every location invariant, which is
exact because delay inside a convex region is `(Z /\ C)^ /\ C`. Anything else is handled by
putting the NEGATION of the predicate into DNF: each term is a conjunction, hence a discrete guard
plus a convex clock region, and `detail::DelayAvoidSet` routes each delay around every term whose
guard holds. The non-convexity lives in the union of terms and is absorbed by intersecting the
per-term results, so no non-convex set is ever represented as a predicate.

**What is missing.** `leads_to_otf` reads only `DiscPred::locs` and `ints`, so `p --> q` refuses
both clocks and boolean structure. The argument for it is written up
(`UppalOMP/sec/subsec/clock_predicates.tex`: `p` is a seed intersection, `q` goes through the same
restriction as `E[]`) but the wiring is not done.

**See also §7** -- `E[]` accepting clock constraints made a pre-existing soundness weakness in
the `E[]` search reachable, and that is currently the more serious problem of the two.

**Mechanism.** `DiscPred` (`include/ptctlmc/netreach.hpp`) carries `clocks`, a conjunction the run
maintains, and `avoid`, a conjunction the run must never enter; at most one is ever populated.
`build_pred` (`src/uppaal.cpp`) routes a clock atom into `clocks`, and `negate_single` routes an
all-clock predicate into `avoid` un-negated. A predicate mixing kinds under a negation matches
neither and is refused there. `leads_to_otf` reads only `locs` and `ints`, which is why
`p --> q` still refuses both.

**Consequence.** None to soundness: the query is **refused, not approximated**. The checker prints

```
-- Formula is UNSUPPORTED (T0.x > 5)
```

and moves on. Like every other entry here, it is visible in the output rather than silent.

**Workaround.** Add an observer automaton that turns the timing condition into a location, then
write the liveness property over that location.

---

## 4. `state_has_deadlock` over-reports at frozen keys

**Mechanism.** At a state whose location vector is committed or urgent, `state_has_deadlock`
(`src/netreach.cpp`) skips both the delay and the application of the combined invariant — correctly,
since no time may pass there. The zone it subtracts the escape regions from is therefore whatever
the caller supplied. The liveness engine's atom-map construction calls it with a *bare universe*
zone for a key rather than with `Inv(q)`, so at a frozen key it computes

```
Dead = universe \ union(Esc)     instead of     Dead = Inv(q) \ union(Esc)
```

Since `Inv(q) subseteq universe`, the result is an over-approximation of the true deadlock set.

**Consequence.** Conservative — it costs precision, never soundness. The over-reported region lies
entirely outside `Inv(q)`, i.e. outside the states that exist at all, and the caller in
`src/netliveness.cpp` intersects with `inv` before using it. That intersection is exactly why the
identity `Inv \ Dead = Inv ^ union Esc` needs only `inv subseteq Z` and not `inv == Z`.

`deadlock_reachable` is unaffected: the zones it passes come from the forward engine, where
`close_succ_zone` has already applied the invariant.

**Contract for new callers.** A caller passing a zone not already restricted to `Inv(q)` must
intersect the returned `stuck` federation with `Inv(q)` itself. This is stated on the function.

---

## 5. The atom construction disengages past 64 atoms at one key

**Mechanism.** The liveness engine tags each node with an `In`/`Out` bitmask classifying it against
the escape atoms of its discrete key (`classify_atoms`, `AtomMap::diag` in `src/netliveness.cpp`).
The tag is a `uint64_t`, so at most 64 atoms per key can be represented. `Otf::build_atom_map`
checks against `kMaxAtoms = 64` and returns `false` rather than truncate — truncating would drop
atoms and silently coarsen the classification.

**Consequence.** Correctness is preserved: the construction declines as a whole and the query falls
back to the race lane, which reaches the same verdict without the atom-based covering. The fallback
is substantially slower, so a model that trips the cap is slow rather than wrong.

**Detection.** A liveness query that is unexpectedly slow on a model with many outgoing transitions
at a single discrete state is the signature. The cap is a fixed 64; it is not configurable.


---

## 6. The liveness engine is sequential, and the obvious ways to parallelise it do not pay

**Mechanism.** `src/netliveness.cpp` contains no `#pragma omp` at all, against twelve in
`src/netreach.cpp`. `--backend omp` and `OMP_NUM_THREADS` change nothing for `p --> q`: wall time
and peak RSS on `arbiter14` are identical at 1, 4, 16 and 32 threads. The reachability engine is
parallel; the liveness engine that replaced `leads_to_net` is not.

**Where the time actually goes.** Measured per phase, and it is not the same phase twice:

| phase | `arbiter14` | `phil9` |
| --- | --- | --- |
| forward search (`base_forward`) | 2.18 s (58%) | 0.28 s (11%) |
| constraint map (`build_atom_map`) | 1.33 s (36%) | 0.11 s (4%) |
| backward fixpoint (`gfp_on`) | 0.10 s (3%) | 2.16 s (84%) |

There is no single hot phase to parallelise. A model is dominated either by the forward half or by
the backward one, and the two want opposite treatments.

**Why the outer-search route does not pay.** Parallelising successor generation — the shape
`mostoptimized` used, generation is pure so it runs unlocked while interning stays serial — is
capped by Amdahl at **1.33x** even on the most forward-heavy family, because generation is only 25%
of `arbiter14`. Measured, it was worse than that: 1.19x wall at 16 threads for **2.6x** the CPU, and
*slower* than the sequential driver at one thread. The cause is the allocator, not the barrier.
Zones come from `ThreadDbmPool`, which is `thread_local`; workers allocate them and the serial
absorb frees them on the main thread, so the worker pools never recycle and every generated zone
becomes a real cross-thread `malloc`/`free`. Producer-consumer allocation defeats a per-thread pool.

**Why the backward route does not pay either.** `gfp_on`'s round-robin sweep parallelises cleanly if
it is made Jacobi — read the previous round's `W`, write a second buffer — and it is *sound*: the
operator is monotone and the iteration descends from the same start, so both schedules converge to
the same greatest fixpoint. Measured, at 16 threads: `phil9` **6.9x** faster (4.86 s -> 0.71 s),
`fischer7` 2.5x, `csma9` 2.3x, `twolock8` unchanged — and `arbiter14` **4.6x slower**, because
Gauss-Seidel propagates along a chain within one round and Jacobi advances one edge per round. A
fixpoint that converged in ~2 rounds and cost 3% of the run became the whole run. The families that
win are the ones where the fixpoint was already expensive; the families that lose are the ones where
it was free. A schedule that switches between the two on a measured trigger is the obvious next
step, and is not implemented.

**Consequence.** The `t16`/`t24`/`t32` columns of a liveness sweep measure the same sequential engine
three times. `benchmarks/scripts/run_live_pipeline.sh` therefore runs ptctlmc liveness at one thread
count only; running three would triple a multi-day sweep for no information.

---

## 7. `E[]` on a clock-bounded predicate disagrees with UPPAAL, in both directions

**This is the currently open soundness issue.** Discovered by `benchmarks/eg_queries`, which is
committed as NOT VALIDATED for that reason.

| model | query | ptctlmc | UPPAAL |
|---|---|---|---|
| fischer4 | `E[] (P0.idle \|\| P0.wait) && x0 <= 40` | satisfied | **NOT** |
| arbiter4 | `E[] (C0.idle \|\| C0.want) && x0 <= 40` | satisfied | **NOT** |
| twolock4 | `E[] (P0.idle \|\| P0.waitA) && x0 <= 30` | satisfied | **NOT** |
| grc4 | `E[] x0 <= 15 && !T0.in` | **NOT** | satisfied |
| grc4 | `E[] (T0.far \|\| T0.near) && x0 <= 15` | **NOT** | satisfied |

**Mechanism, for the first three.** `exists_invariant_run` decides `E[]` by forward cycle
detection on an **extrapolated** zone graph. `UppalOMP/sec/subsec/liveness.tex` states, citing
Laarman 2013 and Tripakis 2009, that redirecting a successor to a strictly larger covering zone
can introduce cycles corresponding to no concrete execution — which is precisely why the leads-to
engine evaluates backward over the same graph instead. Extra_LU+ frees the bounds on the clocks
the query does not mention, so a cycle can close on an abstracted zone that no run realises.
Spurious cycles add witnesses, which is the direction these three run.

**The grc4 pair is not explained by that** and is a separate, undiagnosed defect: a spurious cycle
can only turn NOT into satisfied, never the reverse, so those two are a *missing* witness.

**Not introduced by the clock-predicate work; made reachable by it.** Every clock-free `E[]` query
terminates at the initial state, because with no clock constrained `time_diverges` is itself a
witness — `E[] !(P1.cs && P2.cs)` on fischer5 explores 1 symbolic state. Only a clock bound forbids
that trivial witness and drives the search deep enough to close a cycle. The `d4fb785` binary
agrees with UPPAAL on every clock-free `E[]` query tried and cannot run these at all.

**The predicate machinery is exonerated.** The DNF and the LU arrays were dumped for the fischer4
query and are correct: two terms, one carrying the two negated locations and one the negated clock
atom, with `L(x0)` correctly floored to 40.

**Consequence.** `E[]` and `A<>` verdicts on predicates that constrain a clock are not
trustworthy. `E<>`, `A[]` and `p --> q` are unaffected — they do not go through this search.

**Fix direction.** The same one the leads-to engine already took: decide `E[]` by a backward
greatest fixpoint over the stored graph rather than by forward cycle detection, reusing
`netliveness.cpp` rather than `exists_invariant_run`. `E[] psi` is the paper's own normal form, so
the engine is already the right shape for it.
