# PTCTLMC — Parallel Timed-CTL Model Checker (GPU + OpenMP)

PTCTLMC checks **UPPAAL-style timed-CTL queries** (`E<>`, `A[]`) over **timed
automata**, parallelizing the zone-graph exploration with **OpenMP** and **CUDA**.
It's the timed sibling of [PCTLMC](../PCTLMC) (the CTL engine): same philosophy —
own a GPU-portable core, validate against the reference tool (`verifyta`), ship
three interchangeable backends.

> Status: single-automaton clock fragment, validated against UPPAAL's `verifyta`.
> See [`ROADMAP.md`](ROADMAP.md). Networks/liveness are the next milestone.

## What it does

Given a UPPAAL `.xml` model and a `.q` query file, PTCTLMC reports, for each query,
`Formula is satisfied.` / `Formula is NOT satisfied.` — agreeing with `verifyta`.

```
.xml + .q ─▶ parser ─▶ TA IR ─▶ zone-graph exploration ─▶ verdict
             (own       clocks,   (DBM zones: delay/reset/     E<> / A[]
              minimal    locations, guard/invariant/extrapolate,
              XML+expr)  edges      passed-list subsumption)
                                    backend: seq | omp | cuda
```

## The core: DBMs and zones

A **zone** (set of clock valuations) is a **Difference Bound Matrix** — a dense
`(n+1)×(n+1)` integer matrix of bounds on `x_i − x_j`. All operations (canonicalize
via Floyd–Warshall, `delay`/up, `reset`, conjoin guards/invariants, emptiness,
inclusion/subsumption, LU/max extrapolation) live in
[`dbm_ops.hpp`](include/ptctlmc/dbm_ops.hpp) as **`__host__ __device__` functions**
— so the CPU unit tests validate the *exact* code the GPU runs.

Reachability is a zone-graph search with a **passed list using inclusion
(subsumption)** and **extrapolation** for termination — the standard UPPAAL
approach.

## Backends

| backend | what's parallel |
|---|---|
| `seq` | reference (sequential passed/waiting) |
| `omp` | **shared‑passed hybrid**: one stripe‑locked passed table (no static partition, no serial barrier) with fused expand+insert, zone interning, and successor storage reuse — LTSmin's parallel shape with PTCTLMC's exact convex‑union merge (`--legacy-passed` for the old static‑shard design) |
| `metal` / `cuda` | one GPU thread per (frontier-state, edge): full DBM pipeline on device; host does the subsumption merge. Auto-enabled when the toolkit/framework is present |

All give identical verdicts (`seq`==`omp` is unit-tested; GPU mirrors them and
shares the validated DBM ops).

**Head‑to‑head vs the 2012 opaal+LTSmin multi‑core tool** (rebuilt from source; see
`LTStest/`): the `omp` hybrid **wins every viking size** — LTSmin's own benchmark —
and **crushes Fischer** (~1000× at N=6), because PTCTLMC's exact‑zone convex‑union
merge keeps the symbolic/discrete ratio ≈1 where LTSmin's extrapolation explodes it
to ~50.

## Build & run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/ptctlmc --backend omp model.xml query.q
# GPU build (H100):
cmake -S . -B build-cuda -DCMAKE_CUDA_ARCHITECTURES=90 && cmake --build build-cuda -j
./build/ptctlmc --backend cuda model.xml query.q
```

## Correctness

Validated by **differential testing against `verifyta`** (UPPAAL 5.0.0):
`tests/validate_ta.py model.xml query.q` compares verdicts per query. The curated
suite in `tests/models/` (lamp, periodic-deadline, watchdog, two-clock-race,
urgent/committed, stopwatch-cycle — 36 queries) all match `verifyta`; run
`tests/run_all_ta.sh`.

## Supported fragment (v1)

Single `<template>`, clock variables, location invariants + urgent/committed,
edge guards + clock resets, conjunctive clock constraints incl. diagonals
(`x-y<=c`), queries `E<>`/`A[]` over a location and/or clock bounds.
**Deferred:** networks + channel synchronisation, integer variables, liveness
(`A<>`, `E[]`, `-->`), trace generation. See `ROADMAP.md`.

## License

MIT for our code. Uses UPPAAL's `verifyta` only as an external test oracle; the
shipped engine contains no UDBM (GPL) or UTAP (LGPL) code — the DBM is our own.
