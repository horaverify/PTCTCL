# `E[]` benchmark queries

Non-trivial `E[]` queries, for comparing PTCTLMC against UPPAAL and TChecker.

## Why they need a clock bound

An `E[]` query with no clock constraint is not a benchmark. `time_diverges` fires at the initial
state — every clock can grow without bound, so time divergence is itself a witness — and the
search stops at **1 symbolic state**. `E[] !(P1.cs && P2.cs)` on fischer5 explores 1 state;
`E[] x0 <= 100` explores 157563. The clock bound is what forbids the trivial witness and forces
the search to build the whole psi-restricted zone graph.

That also makes these queries genuinely harder than the safety ones: `A[] not deadlock` on
fischer5 explores 2068 states against 157563 here, because the query constant defeats the
extrapolation that keeps the safety search small.

Each file holds five queries per family, mixing the shapes: a bare clock bound, a clock bound
with a negated location, a location disjunction, a clock disjunction, and a negated conjunction.

## STATUS: NOT VALIDATED — these queries found a disagreement with UPPAAL

Five of the queries below disagree with UPPAAL 5.0.0, **in both directions**:

| model | query | ptctlmc | UPPAAL |
|---|---|---|---|
| fischer4 | `E[] (P0.idle \|\| P0.wait) && x0 <= 40` | satisfied | **NOT** |
| arbiter4 | `E[] (C0.idle \|\| C0.want) && x0 <= 40` | satisfied | **NOT** |
| twolock4 | `E[] (P0.idle \|\| P0.waitA) && x0 <= 30` | satisfied | **NOT** |
| grc4 | `E[] x0 <= 15 && !T0.in` | **NOT** | satisfied |
| grc4 | `E[] (T0.far \|\| T0.near) && x0 <= 15` | **NOT** | satisfied |

The predicate machinery is not the cause — dumping the internal state for the first query shows
the DNF is exactly right (two terms: one with the two negated locations, one with the negated
clock atom) and the LU flooring is correct (`L(x0) = 40`).

**The first three look like the unsoundness the paper already documents.** `exists_invariant_run`
decides `E[]` by *forward cycle detection on an extrapolated zone graph*, and
`UppalOMP/sec/subsec/liveness.tex` states plainly, citing Laarman 2013 and Tripakis 2009, that
"redirecting an exact successor to a strictly larger covering zone may introduce cycles in `G`
that do not correspond to concrete executions, yielding spurious liveness witnesses" — which is
exactly why the leads-to engine evaluates backward instead. Extrapolation frees the bounds on the
clocks the query does *not* mention, so a cycle can close on an abstracted zone that no concrete
run realises. Spurious cycles add witnesses, which matches the direction of these three.

**The grc4 pair is the opposite direction and is NOT explained by that** — a spurious cycle can
only turn NOT into satisfied. A missing witness is a separate defect and is not yet diagnosed.

This is a **pre-existing** weakness that this work made reachable rather than introduced: before
`E[]` accepted clock constraints there was no query that explored far enough to close a spurious
cycle, because every clock-free `E[]` query terminates at the root on time divergence. The
`d4fb785` binary agrees with UPPAAL on every clock-free query tried, and cannot run these at all.

Do not use these numbers as a benchmark until both directions are resolved.

## TChecker

Not runnable as-is. TChecker's liveness tool is label-based Buchi
(`tck-liveness -a couvscc -l LABEL`), not TCTL, so `E[] psi` needs the psi-restricted model
generated up front — psi's clock part as an invariant on every location, psi's discrete part as a
location filter — and then a cycle over fully-labelled states. That is the same construction
`exists_invariant_run` performs internally, so a generator alongside
`benchmarks/scripts/gen_tck_live.py` would do it. TChecker is not installed here and no TChecker
arm has been run.
