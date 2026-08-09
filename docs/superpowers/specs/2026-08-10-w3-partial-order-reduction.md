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
