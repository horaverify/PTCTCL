# Differential against UPPAAL 5.0.0

`verifyta` from `/Applications/UPPAAL-5.0.0.app`, rev. 714BA9DB36F49691, June 2023.
Each row is one (model, query) pair run through both tools; verdicts normalised to
SAT / NOT / UNSUP / ERROR and compared.

| round | pairs | SAT | NOT | refused | disagreements |
|---|---|---|---|---|---|
| 1 — `round1-verdicts.txt` | 77 | 57 | 17 | 3 ERROR | **0** |
| 2 — `round2-clock-liveness.txt` | 82 | 53 | 28 | 1 UNSUP | **0** real |
| 3 — `round3-nonconvex-af.txt` | 58 | 29 | 29 | 0 | **0** |

**217 pairs, zero real disagreements.**

Round 1 covers the urgent-channel models, the clock-liveness models, the liveness
regressions, and `A[] not deadlock` + each family's reach and liveness query suites over
arbiter/csma/fischer/grc/phil/twolock/viking/single at sizes 4-6.

Round 2 stresses the new feature — clock atoms in `E[]` / `A<>` — on real benchmark models
rather than the toy: both bound directions, strict and non-strict, satisfiable and not, plus
mixed discrete+clock predicates.

## The disagreement round 1 originally found, and how it was closed

    A<> x >= 5 && x <= 10      hora: UNSUPPORTED      UPPAAL: NOT satisfied

`A<> phi = !E[] !phi`, and the complement of a two-atom conjunction is a disjunction, which the
delay-inside-a-convex-region argument does not cover — so HORA refused what UPPAAL decides.

Closed by `detail::delay_avoiding`: the region to AVOID is `phi` itself, which is convex even
though `!phi` is not, so the delay is routed around it in closed form rather than the predicate
being maintained. Round 3 is the regression suite for that operator — the same forms over one and
two clocks on the toy models and on six real benchmark models, deliberately balanced at 29
satisfied against 29 not, so a trivially-refusing implementation could not pass it.

`bruteforce-delay-avoiding.py` is the independent check of the operator itself: 3049 random
(Z, forbid) pairs over two clocks on a 14x14 grid, compared against explicit trajectory
enumeration, no mismatch.

## Still refused

    A<> P.T && x >= 0          hora: UNSUPPORTED      UPPAAL: satisfied

A negation spanning a discrete atom and a clock atom is a disjunction across two different kinds
of state, which neither mechanism represents. See KNOWN_LIMITATION.md 3.

## Not a disagreement

Round 2's flagged row is a typo in the harness: `E[] P0.Eat && x0 >= 0` names clock `x0`, which
phil4 does not have (its clocks are `c0..c3`). Both tools refuse it — HORA as UNSUPPORTED,
UPPAAL as an error. Re-run with `c0`, both say NOT satisfied.

## Agreement worth naming

`urgent-clip.xml` is rejected by both, and for the same reason. UPPAAL:
`EXCEPTION: Clock guards are not allowed on urgent edges.` The restriction in
KNOWN_LIMITATION.md 1 is UPPAAL's own.
