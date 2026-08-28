# Differential against UPPAAL 5.0.0

`verifyta` from `/Applications/UPPAAL-5.0.0.app`, rev. 714BA9DB36F49691, June 2023.
Each row is one (model, query) pair run through both tools; verdicts normalised to
SAT / NOT / UNSUP / ERROR and compared.

| round | pairs | SAT | NOT | refused | disagreements |
|---|---|---|---|---|---|
| 1 — `round1-verdicts.txt` | 77 | 56 | 17 | 3 ERROR + 1 UNSUP | **1** (see below) |
| 2 — `round2-clock-liveness.txt` | 82 | 53 | 28 | 1 UNSUP | **0** real |

Round 1 covers the urgent-channel models, the clock-liveness models, the liveness
regressions, and `A[] not deadlock` + each family's reach and liveness query suites over
arbiter/csma/fischer/grc/phil/twolock/viking/single at sizes 4-6.

Round 2 stresses the new feature — clock atoms in `E[]` / `A<>` — on real benchmark models
rather than the toy: both bound directions, strict and non-strict, satisfiable and not, plus
mixed discrete+clock predicates.

## The one genuine disagreement

    A<> x >= 5 && x <= 10      ptctlmc: UNSUPPORTED      UPPAAL: NOT satisfied

This is the documented convexity restriction, not a wrong verdict. `A<> phi = !E[] !phi`, and
the complement of a two-atom conjunction is a disjunction, which the delay-inside-a-convex-region
argument does not cover. PTCTLMC refuses; UPPAAL decides it. **A capability gap in UPPAAL's
favour.**

## Not a disagreement

Round 2's flagged row is a typo in the harness: `E[] P0.Eat && x0 >= 0` names clock `x0`, which
phil4 does not have (its clocks are `c0..c3`). Both tools refuse it — PTCTLMC as UNSUPPORTED,
UPPAAL as an error. Re-run with `c0`, both say NOT satisfied.

## Agreement worth naming

`urgent-clip.xml` is rejected by both, and for the same reason. UPPAAL:
`EXCEPTION: Clock guards are not allowed on urgent edges.` The restriction in
KNOWN_LIMITATION.md 1 is UPPAAL's own.
