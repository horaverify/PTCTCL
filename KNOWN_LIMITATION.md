# Known limitations

What this engine does *not* do, and why. Each entry gives the mechanism, not just the symptom, so
that a surprising result can be matched against the cause.

Ordered by how easily the limitation goes unnoticed. The first one is silent.

---

## 1. `urgent chan` is accepted and silently downgraded to an ordinary channel

**This is the one that fails silently.** A model declaring `urgent chan` is verified against
different semantics than it declares, and nothing in the output says so.

**Mechanism.** The declaration parser strips the keyword purely to let the rest of the declaration
parse (`src/uppaal.cpp`, the `urgent`/`broadcast`/`chan` prefix peel). The resulting flag is used
only to compute the remaining substring and is then discarded: there is no urgent-channel field
anywhere in `Network`, so nothing downstream could consult it even in principle.

**Consequence.** An urgent channel must force a synchronisation to happen without delay whenever it
is enabled. Downgraded to an ordinary channel, that obligation disappears: the engine explores runs
in which time passes while the urgent synchronisation is enabled. Those runs do not exist in the
declared model.

The error is **not** one-sided. The extra runs can reach states the real model cannot, so an
`A[]` safety property can be reported violated when it holds; and the missing urgency can leave a
run available that the real model would have forced to progress, so a liveness property can be
reported satisfied when it fails.

**Detection.** Grep the model for `urgent chan` before trusting a verdict. There is no runtime
warning.

**Workaround.** Encode urgency explicitly — give each sender a committed location, or add a clock
`u` with invariant `u <= 0` reset on entry to the sending location.

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

## 3. Liveness predicates may constrain only the discrete state

`E[] phi`, `A<> phi` and `p --> q` accept predicates over locations and integer variables only.
`E<>` and `A[]` accept clock constraints normally.

**Mechanism.** These queries are carried by `DiscPred` (`include/ptctlmc/netreach.hpp`), which holds
a `LocReq` list and an `IntAtom` list and has no field for clock constraints. `build_pred`
(`src/uppaal.cpp`) parses each conjunct and returns failure the moment `parse_net_atom` yields a
clock constraint. Correspondingly `exists_invariant_run` builds an all-zero `qmask`, since no
liveness predicate can name a clock that would need protecting from the active-clock reduction.

**Consequence.** None to soundness: the query is **refused, not approximated**. The checker prints

```
-- Formula is UNSUPPORTED (T0.x > 5)
```

and moves on. This limitation is visible in the output, which is what separates it from item 1.

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

## 6. The `omp` backend's state count is not reproducible

Verdicts are deterministic. **Stored-state counts are not.**

**Mechanism.** Workers insert into the shared striped passed set as they expand, and the
convex-union merge in `NetPassed::add` is order-sensitive: whether two zones merge into one depends
on which arrives first. Thread interleaving therefore changes how many zones survive, and the
per-level merge window (`kMergeScan`) sees a different newest-first ordering from run to run. The
adaptive mirror-promotion rule is likewise driven by per-chain visit counts, which are
schedule-dependent.

**Consequence.** The verdict is unaffected — merging is an optimisation, and skipping one only
leaves two zones where one would do. Only the count moves. Measured on `fischer6 A[] not deadlock`,
six consecutive `--backend omp` runs:

| backend | stored states |
|---|---|
| `omp`, 6 runs | 12 005 / 12 011 / 12 058 / 12 081 / 12 082 / 12 089 |
| `seq`, any run | **11 889** |

A spread of 84 states, and consistently above the sequential count.

**Rule.** Any state count quoted in a paper, a benchmark table or a regression baseline must come
from `--backend seq`, which is deterministic. Use `omp` for timing, `seq` for counts.
