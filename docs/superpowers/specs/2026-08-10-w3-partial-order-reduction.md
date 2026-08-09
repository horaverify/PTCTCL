# W3 — partial-order reduction

Branch: `mostoptimized`. Follows the W0/W2/W1 document; the floor argument there is the premise.

## 1. Available independence, measured first

POR removes interleavings of *independent* actions. Before building ample sets, the same
discipline that settled W1 contraction: measure how much independence the models actually have.

Two processes are statically **dependent** if they share a synchronisation channel, or one writes
a variable the other reads or writes. (Read/read is independent. Clocks are already known to be
process-local in every model, so they contribute no dependence.)

| model | processes | independent pairs | dependent pairs | % independent |
|---|---|---|---|---|
| **phil6** | 12 | 48 | 18 | **72.7%** |
| grc4 | 6 | 4 | 11 | 26.7% |
| fischer6 | 6 | 0 | 15 | **0.0%** |
| csma6 | 7 | 0 | 21 | **0.0%** |
| arbiter6 | 7 | 0 | 21 | **0.0%** |
| viking6 | 7 | 0 | 21 | **0.0%** |
| twolock6 | 6 | 0 | 15 | **0.0%** |
| single6 | 1 | — | — | n/a |

The zeros are not artefacts, they are the protocols:

* **fischer, twolock** — every edge reads or writes the shared `id` / `a` / `b`. This is a
  shared-variable mutual-exclusion protocol; that is what it means for it to work.
* **arbiter** — every client synchronises with the one `Arb` process on `acq`/`rel`, so any two
  clients' actions compete for the same channel and firing one disables the other.
* **viking** — all vikings share the `take`/`release` torch channels and the side variable `L`.
* **csma** — `cd` is a broadcast channel, so it involves every process at once.

## 2. What this means for the programme

Cross-referencing with the W0/W2/W1 document: the five families at **0% independence** are
*exactly* the five sitting on the reachable-discrete-state floor and tying `verifyta` to the
digit. So POR cannot move them either.

Stated plainly, and this is the programme's real conclusion:

> For these process-replicated benchmarks with a shared resource, the reachable discrete state
> space **is** the answer, and both tools already compute it exactly. No abstraction, no
> subsumption and no partial-order method can reduce it, because there is nothing redundant left
> in it.

The wins available were, and turned out to be, exactly three:

1. **zones pile up at one discrete state** → `single`, addressed by W1's G-simulation
   (single6 −66% vs `verifyta`; single7 completes where `verifyta` times out);
2. **discrete data is wide or dead** → `train`, addressed by W2's range narrowing
   (91 bits → runs at all);
3. **genuine concurrency** → `phil` at 72.7%, and `grc` at 26.7%. This is W3's target, and it is
   the only one left.

## 3. Scope

W3 targets `phil`-shaped networks: pairwise channel synchronisation, no shared variables. That is
one benchmark family here plus part of a second. The technique is general; the *suite* is not a
good showcase for it, and adding a POR-friendly family (a pipeline, a producer/consumer farm)
would be needed to evaluate it honestly rather than to flatter it.

## 3b. The blocker: independence does NOT give a diamond here

Independence is necessary for POR, not sufficient. In global-time semantics two independent
actions commute only if the *zone* they produce is the same in both orders, and it generally is
not — the survey states the obstacle directly (arXiv 2207.07479 §8.2):

> "suppose that `x` is reset at `a` and `y` is reset at `b`, then `ab` and `ba` lead to different
> zones keeping track of the different order of the resets. There is an implicit synchronization
> due to the global nature of time."

**This applies to `phil`, the one family with independence.** Its philosopher template:

```
invariant  c0 <= 2      guard  c0 >= 1      assignment  c0 = 0      sync  take0!/take1!/put0!/put1!
no urgent locations, no committed locations
```

Every philosopher resets its own clock, has an invariant bounding delay, and nothing is urgent —
so time passes between two independent philosophers' actions and the order of their resets is
recorded in the zone as the sign of `c_i − c_j`. Firing `P0` then `P3` and firing `P3` then `P0`
reach the same discrete state with **different, incomparable zones**. Exploring one order loses
valuations the other reaches.

**Measured confirmation.** If independent actions commuted at the zone level, two interleavings
reaching one discrete state would produce *identical* zones and be caught as exact duplicates.
Instead they arrive distinct and have to be reconciled:

| model | discrete states | stored zones | subsumed by newer | absorbed into convex union |
|---|---|---|---|---|
| phil6 | 728 | 732 | 13 | 49 |
| phil8 | 6 560 | 6 797 | 378 | 754 |
| phil10 | 59 048 | 62 030 | 13 679 | **11 242** |

On `phil10`, 11 242 zones arrived at a discrete state already holding a *different* zone whose
union with the newcomer happened to be convex, and 13 679 more were strictly subsumed. Those
24 921 events are the interleaving divergence, counted. The diamond does not hold.

**The three sound escapes, and why two are closed here.**

1. **Zero-time independence** — if the independent actions are urgent or committed, no time passes
   between them and the resets coincide. `phil` has **no** urgent and **no** committed locations,
   and neither does any other model in the suite. Closed.
2. **Reset-free, invariant-free independence** — actions that reset no clocks record no order, so
   they commute. `phil`'s edges reset (`c0 = 0`) and its locations carry invariants (`c0 <= 2`),
   which also makes the permitted delay differ between the two orders. Closed.
3. **POR modulo abstraction** (Hansen et al., CAV'14) — do not require the two zones to be equal,
   only equivalent under the simulation already in use. This one is open, and it is the only
   route left.

**Consequence: implementing classical ample-set POR here would be unsound**, not merely
unprofitable. It would silently drop reachable valuations on the only family it applies to. It is
therefore not implemented.

## 3c. What the engine already recovers

Worth recording, because it explains why `phil` is not exploding despite the divergence: the
**exact convex-union merge is already reclaiming most of the interleaving redundancy**, at the
zone level rather than the interleaving level. That is what holds `phil` at 1.05 zones per
discrete state instead of letting order-divergent zones accumulate. POR would attack the same
redundancy earlier — by not generating the interleavings — but the engine is not paying full
price for them today.

## 3d. The only remaining route, stated precisely

**POR modulo simulation on a mutable subsumption antichain.** Do not require `Z_ab = Z_ba`;
require `Z_ab ⊑ Z_ba` under the engine's own simulation (`a_LU` or `G`). The measurements above
say this is often *nearly* true — 11 242 of the divergent pairs on `phil10` were close enough to
have a convex union. The research content is:

* deciding the equivalence **without computing both orders**, or the reduction pays nothing;
* the interaction with the antichain: if `v` is expanded with a reduced successor set and is
  later killed because `Z_u ⊑ Z_v`, the behaviours `v` deferred are lost — unaddressed anywhere;
* the ignoring proviso under level-synchronous parallel BFS, which has neither a DFS stack nor a
  sequential order.

This is a research project, not an implementation task, and it is the specific open challenge the
survey ends on.

## 4. Design constraints already established

* **No local-time semantics.** The survey (arXiv 2207.07479 §8.2) records that there is no finite
  abstraction for local-time semantics compatible with POR in general. Going that route would
  also replace the entire zone layer, interner, simulation test and convex merge. Ruled out.
* **Global-time semantics with a syntactic independence relation** is the route the survey points
  at ("discovering independent actions in the standard global-time semantics, either statically
  or dynamically"). Section 1's analysis is exactly that relation.
* **The two hard parts remain open and are the contribution**, not the ample sets themselves:
  1. **POR × subsumption.** If `v` is expanded with a reduced successor set and `u` is later
     killed because `Z_u ⊑ Z_v`, the behaviours `v` deferred are lost. Ample sets on a mutable
     subsumption antichain are unaddressed in the literature.
  2. **The ignoring proviso under level-synchronous parallel BFS.** The standard provisos assume
     a DFS stack. This engine has neither a stack nor a sequential order.
