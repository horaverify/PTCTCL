# The early `p --> q` witness: statement, proof, side conditions

Written to be lifted into a paper. Everything here is about the *forward* phase deciding a
violation before the graph is finished; nothing about it is specific to HORA's D refinement,
and it does not depend on the covering relation at all.

Implementation: `Otf::leadsto_witness` in `src/netliveness.cpp`.

## 1. Setting

Let `N` be a network of timed automata and `S` its semantics: states are pairs `(l, nu)` of a
discrete state and a clock valuation, with delay transitions `(l, nu) -> (l, nu + d)` and action
transitions. A run is *maximal* when it cannot be extended — either it is infinite, or it ends
in a state from which neither a delay nor an action leads anywhere new.

UPPAAL's leads-to is

    p --> q   ==   A[] (p imply A<> q)

so it is **violated** exactly when

    exists a reachable state s |= p  with a maximal run from s along which q never holds,

i.e. `EF(p /\ EG !q)`.

The symbolic engine decides `EG !q` as a greatest fixpoint over the zone graph. For a vertex `v`
with stored zone `Z_v` and outgoing tagged edges `e : v -> j`,

    W[v]  =  Z_v  /\  ( NoSucc(v)  \/  \/_e  Pre_e(W[j]) )                              (1)

where `Pre_e` is the exact concrete predecessor of the transition that `e` records, and

    NoSucc(v)  =  { nu in Z_v : the maximal run from (l_v, nu) ENDS at (l_v, nu) }       (2)

is the *stopping term*: the valuations of `v` from which the run may legitimately stop, either
because no transition is ever enabled from them (a deadlock, possibly only on a sub-region of the
zone) or because time can diverge there. `W` is the greatest fixpoint of (1) restricted to the
vertex set `S` of interest, and the query is violated iff `W[v] /\ p != empty` at some `v |= p`.

## 2. The proposition

> **Proposition.** Let `v` be a vertex whose stored zone `Z_v` is an exactly reachable zone, whose
> discrete state satisfies `!q`, and let
>
>     nu  in  NoSucc(v)  /\  p(v)
>
> for some valuation `nu`. Then `p --> q` is violated, and `nu` witnesses it.

*Proof.* Define the candidate assignment

    X[v] = Z_v /\ NoSucc(v),        X[u] = empty  for every other vertex u.

`X` is a **post-fixpoint** of the operator (1): the only non-trivial obligation is at `v`, where

    X[v] = Z_v /\ NoSucc(v)  subseteq  Z_v /\ ( NoSucc(v) \/ ... )

holds because `X[v] subseteq NoSucc(v)` by construction; the disjunction can only make the
right-hand side larger, and at every other vertex `X[u] = empty` is contained in anything.

By Knaster–Tarski the greatest fixpoint of a monotone operator is the union of its
post-fixpoints, so `X subseteq W`, hence

    Z_v /\ NoSucc(v)  subseteq  W[v].

If `nu` also satisfies `p`, then `W[v] /\ p != empty`, which is exactly the violation condition.
Concretely the counterexample is the run that reaches `nu` and then stops (or lets time diverge)
there: it satisfies `p` at `nu`, it is maximal by the definition (2) of `NoSucc`, and it never
satisfies `q` because it never leaves the discrete state of `v`, which satisfies `!q`. ∎

**What the proposition is *not*.** It is not a new fixpoint theory and not a property of the
covering relation. It says only that one *disjunct* of the operator can be read at a single
vertex, before the graph is complete, because that disjunct alone already certifies a
post-fixpoint. It is the same observation that makes on-the-fly liveness work — an intermediate
iterate of a descending greatest-fixpoint computation is a *pre*-fixpoint and therefore useless
in both directions, but any post-fixpoint, however small, is a certificate.

**Practical consequence.** `NoSucc(v)` reads no successor, no cycle and no other vertex. It is a
per-state, zone-local predicate. So a *single expanded state* can decide a violated `p --> q`,
whereas a construct-then-solve architecture must finish the forward graph before its backward
pass can conclude anything.

## 3. Side conditions, and why each is load-bearing

The proposition assumes `Z_v` is an **exactly reachable** zone, and that `NoSucc(v)` and the
`!q` test are evaluated correctly. Three things can break that, and each is guarded.

### 3.1 Extrapolation (guarded by the `a_LU` lane)

`NoSucc` occurs *positively* in (1). An over-approximated `Z_v` can therefore contribute a
*spurious* stop valuation, which does not lose a counterexample — it fabricates one. The engine's
covering discipline already ties the two together: with `a_LU` covering in force, stored zones
carry no extrapolation and are exact; where extrapolation is load-bearing (the diagonal route,
which cannot use `a_LU` at all — Bouyer 2004) stored zones over-approximate. The witness is
therefore confined to the `a_LU` lane.

This is not an extra assumption invented for the early exit: the backward pass evaluates the same
`NoSucc` on the same stored zones, so the condition it needs is the one the engine already
maintains.

### 3.2 The active-clock reduction (handled inside `NoSucc`)

Stored zones are not literally sets of reachable valuations: the active-clock reduction pins to 0
every clock that no process will read again before resetting it. Two facts make the witness
survive this.

1. Every guard and invariant reachable from `v` mentions only *active* clocks — that is the
   definition of the reduction — so whether a valuation is stuck, and whether time can diverge,
   depends only on the active components. For any `nu` in the stored zone there is a genuinely
   reachable `nu'` agreeing with it on those, and `nu'` is the actual witness.
2. The reduction destroys delay-closure (it pins a dead clock with an *upper* bound). The
   deadlock test rebuilds the closure before subtracting — freeing the dead clocks and
   re-imposing the real invariant — precisely so that `down()` cannot invent a stop valuation.

Again the backward pass depends on exactly the same repair.

### 3.3 A non-convex `q` (guarded by refusing the region route)

When `q` constrains clocks, it is not decided by the discrete test. The engine carries the clock
part as a forbidden region (`DelayAvoidSet`) that the backward pass intersects into the seed, and
the discrete predicate evaluator *deliberately ignores* terms carrying a clock — correct for the
backward pass, wrong for a witness. A stop valuation could then satisfy `q` through the part the
discrete test skipped, and the "witness" would certify nothing.

Deciding it here would mean intersecting `NoSucc(v)` with the `!q` region as well as with `p`.
Until that is done, the witness is off whenever a forbidden region is in force, and those queries
take the ordinary construct-then-solve path. (A clock-carrying `p` is fine and *is* handled: `p`'s
region is intersected in exactly, so an over-approximating discrete `p`-test cannot leak.)

## 4. What it does not cover

The witness is a **stop** witness. When the counterexample is an infinite run through a *cycle* —
no single state certifies an infinite run — it correctly does not fire, and the backward fixpoint
decides the query as before. In the benchmark set this is `phil P0.Eat --> P0.Think`,
`arbiter C0.cs --> C0.want` and `csma Bus.Transmitting --> Bus.Collided`; their state counts are
unchanged, which is the regression evidence that the witness is not firing where it must not.

Extending it to cycle witnesses means finding a cycle in the `!q` subgraph on the fly whose
composed `Pre` has a non-empty fixpoint — a strictly harder problem, and open here.

## 5. Evidence

**Differential, three engines.** Every query was run on (a) the pre-change binary, (b) the new
binary, and (c) the new binary with `HORA_LEADSTO_EARLY=0`. Comparison (b) against (c) isolates
the witness from every other change on the branch.

| suite | queries | disagreements |
|---|---|---|
| every ordered pair of locations within a template, plus cross-template pairs, on arbiter4, fischer4, csma4, twolock4 | 180 | 0 |
| the same on grc4 and phil4 (invariants, urgency, committed locations) | 29 | 0 |
| clock atoms on either side of `-->`, on fischer4, arbiter4, grc4, csma4 | 112 | 0 |
| **total** | **321** | **0** |

The witness fired on 132 of the 321, so the agreement is not vacuous.

**Full suite.** 408 cases (`live`, `eg`, `violated`, `reach`, `deadlock` × sizes 4–9), 0 verdict
mismatches.

**Effect, where it fires** (states explored, baseline -> new):

| query | size | states | speedup |
|---|---|---|---|
| `fischer P0.wait --> P0.cs` | 7 | 11951 -> 9 | ×54 |
| `twolock P0.waitB --> P0.reqB` | 7 | 12847 -> 121 | ×47.5 |
| `fischer P0.cs --> P0.req` | 7 | 11951 -> 37 | ×18 |
| `grc T0.near --> T0.far` | 5 | 13980 -> 3519 | ×71.6 |
| `grc T0.near --> T0.far` | 6 | baseline exceeds 45 s | 2.37 s |
| `arbiter C0.want --> C0.cs` | 9 | 2816 -> 2 | — |

## 6. A note on the degenerate rows

`grc T0.far --> T0.in` and `csma Bus.Idle --> Bus.Collided` name the *initial* location, which in
both models carries no invariant, so "stay there for ever" is already a maximal `!q` run and the
first expanded state decides the query. These look like weak benchmarks and are not: the baseline
still explores the entire reachable graph (13980 states on grc5, and does not finish grc6 or grc7
inside 60 s), because a build-then-solve architecture has no place to notice. They are the
sharpest illustration of the architectural point, and the recorded `verifyta` and `tchecker`
results contain no violated leads-to row at all, so the comparison there is open.
