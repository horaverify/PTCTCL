# `E[]` benchmark queries

Non-trivial `E[]` queries for comparing HORA against UPPAAL (and, with a generator, TChecker).

    python3 benchmarks/scripts/bench_eg.py --sizes 4,5,6 --repeat 3

Reports per (model, query): both verdicts, HORA's stored-state count, and best-of-N wall time
for each tool. A verdict mismatch is flagged loudly; `unsup` is a refusal, not a wrong answer.

## Why every query carries a clock bound

An `E[]` query with no clock constraint is not a benchmark. Time divergence is itself a witness,
so with every clock unbounded the search succeeds at the initial state and stops:
`E[] !(P1.cs && P2.cs)` on fischer5 explores **1** symbolic state. A clock bound forbids that
trivial witness and forces the whole psi-restricted graph to be built.

The bound also makes these harder than the safety queries on the same models, because the query
constant defeats the extrapolation that keeps the safety search small.

Four shapes per family, chosen so the predicate machinery is exercised rather than just the
search: a bare clock bound; a clock bound with a negated location; a location **disjunction** with
a clock bound; and a **negated conjunction** with a clock bound.

## The non-convex row

Each family carries one query putting a clock under a **disjunction** —
`E[] x0 <= 40 && (x1 <= 30 || x1 >= 35)`. The run must stay inside a non-convex region, so it
exercises the routed delay (`delay_avoiding` forward, `pre_avoiding` backward) rather than the
invariant path every other row takes. It is also the most expensive row in each family, since the
forbidden region splits zones: fischer4 goes from 248 stored states to 334.

## Status

Validated: 28 rows, zero verdict mismatches against UPPAAL 5.0.0. Across arbiter/csma/fischer/grc/phil/twolock these queries agree with UPPAAL 5.0.0 on
every row. That was not true when the suite was written: it immediately exposed five wrong
verdicts, which is what the suite is for. The cause was that `E[]` was decided by forward cycle
detection on a subsumed graph — unsound, per Laarman 2013 / Tripakis 2009, and the very reason
`p --> q` is evaluated backward. `E[]` is now decided backward too, on the same greatest fixpoint.

The state counts moved by orders of magnitude in the process, because the backward engine uses
a_LU subsumption with **per-location** bounds where the forward one could only extrapolate under a
single global pair:

| query | forward (old) | backward (now) |
|---|---|---|
| fischer4 `E[] x0 <= 40` | 9188 | **248** |
| grc4 `E[] x0 <= 15` | 1221269 | **1031** |

## TChecker

Not runnable as-is: TChecker's liveness tool is label-based Büchi
(`tck-liveness -a couvscc -l LABEL`), not TCTL. `E[] psi` needs the psi-restricted model built up
front — psi's clock part as an invariant on every location, psi's discrete part as a location
filter — then a cycle over fully-labelled states. That is the same construction the engine
performs internally, so a generator alongside `benchmarks/scripts/gen_tck_live.py` would do it.
TChecker is not installed here and **no TChecker arm has been run**.
