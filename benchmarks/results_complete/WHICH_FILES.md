# Which results file to take from where

Everything a table or plot needs, and where the number actually comes from. Read this before
mixing any two folders — several of them are **not** comparable, and the reasons are recorded
below rather than left to be rediscovered.

Generated 2026-07-31. Sweep driver: `benchmarks/run_complete.sh`, stop with `./stop_complete.sh`.

---

## The short version

| you want | take it from |
|---|---|
| PTCTLMC headline numbers (all families) | `results_complete/adaptive/ptctlmc_t<T>_c<C>_adaptive_2h.csv` |
| PTCTLMC legacy-engine column (`single`, `csma` only) | `results_complete/legacy/ptctlmc_t<T>_c<C>_legacy_2h.csv` |
| UPPAAL `verifyta` baseline | `results_old/verifyta_only.csv` (600 s) + `results_old/timeout2h/verifyta_2h.csv` (2 h) |
| queries used | `benchmarks/queries/<family>.q` |

`<T>` in {16, 24, 32}, `<C>` in {8, 4}.

---

## `results_complete/adaptive/` — the headline result

Current engine: current convex union + **adaptive** dense-mirror promotion.
All six families: `phil`, `single`, `twolock`, `fischer`, `viking`, `csma`.

Budget: 2 h, 100 GB, 2 reps, strike-limit 1 (one timeout in a family skips every larger size).
Pinned: each job runs on exactly as many cores as it has threads (`taskset -c 0-<T-1>`).

## `results_complete/legacy/` — the reference-engine column

Same binary, `PTCTLMC_ENGINE=legacy`: the **pre-`10583cf` convex union**
(`dbm_try_union_legacy`) with that era's fixed `hotlen=64` mirror. This reproduces the
2026-07-29 evaluation build to within 2%:

| phil13, t32 c4 | |
|---|---|
| reference build (`build-spin/ptctlmc`, 2026-07-28) | 12.31 s |
| `PTCTLMC_ENGINE=legacy` | 12.27 s |

Only **`single` and `csma`** are run here, by request — the two families whose reference
numbers are intact and worth a like-for-like legacy column.

**Both engines do convex merging.** The only difference is which `try_union` runs plus the
mirror policy. `10583cf` added a per-call `mindbm_analyze` that prunes redundant facets: a win
at large `dim`, a loss at `phil`'s. The dense mirror, the shared kill probe and the `SpinRW`
stripe locks are common to both engines — each was suspected of the difference and each was
cleared.

### Caveat on the legacy column for `single`

`legacy` is **faster than the 2026-07-28 reference on `single`** (single6: 3.39 s vs 7.34 s),
not equal to it. That is deliberate: legacy reverts only the convex union, and keeps the
single-TA routing fix from `4d16d13`, which is a genuine search improvement (single6 explores
5775 states instead of 16830). Reverting that too would discard a correct 2× for no reason. If
the paper's legacy column must also predate the routing fix, that is a *different* build and
is not what this folder contains.

---

## `results_old/` — the previous sweep. Read the warnings.

| file | what it is |
|---|---|
| `verifyta_only.csv` | UPPAAL, 1 thread, **600 s** budget. viking20/phil15+/single7+/twolock14+ are TIMEOUT rows here. |
| `timeout2h/verifyta_2h.csv` | UPPAAL re-run of exactly those timeouts at a **2 h** budget. **This is where viking20 = 1253.98 s comes from**, i.e. the 13.8× headline. |
| `ptctlmc_t<T>_c<C>_ms32_hl64_600s_v2.csv` | previous PTCTLMC sweep, 600 s budget, unpinned |
| `timeout2h/ptctlmc_t<T>_c<C>_2h.csv` | its 2 h re-run of the timeouts only (incomplete: t16 c2/c4/c8 only) |

**verifyta rows are still valid.** It is an unchanged external binary and reproduces its
recorded times to within 1–2% (phil12: 20.02 s today vs 19.86/19.66 s recorded), which is why
this sweep does not re-run it.

**Do not mix `results_old` PTCTLMC rows with `results_complete` rows in one table.** Different
budget (600 s vs 2 h), different pinning (unpinned vs pinned), different engine. Use
`results_complete/legacy/` when you want a legacy-engine number.

### `results_old/ptctlmc_t32_c8_*` is the least trustworthy file in the tree

`run_chunk_sweep.sh` only ever produced chunk 2, 4 and 16. The `c8` CSVs came from an earlier,
separate run whose binary is not identified — and `results_old/old/` contains a file literally
named `ptctlmc_t24_c8.MIXEDBINARY.csv`. Prefer `c4` or `c16` from that sweep when you need a
historical PTCTLMC number.

---

## Queries — `benchmarks/queries/`

One directory for all platforms (the old `queries_linux` / `queries_macos` split had drifted
apart and is now `queries_linux_old/`). Every query is verified **satisfied** on a small size,
which is what makes state counts meaningful: a falsified property makes both tools stop at the
first counterexample, so the count stops measuring the state space.

| family | query | verifyta explored (small size) |
|---|---|---|
| fischer | `A[] not (P1.cs && P2.cs)` | fischer8: 40536 |
| phil | `A[] not (P0.Eat && P1.Eat)` | phil8: 8486 |
| twolock | `A[] not (P1.cs && P2.cs)` | twolock8: 42584 |
| arbiter | `A[] not (C1.cs && C2.cs)` | arbiter8: 1280 |
| csma | `A[] not (S0.Send and S1.Send and Bus.Transmitting)` | csma8: 5888 |
| single | `A[] x0 <= 6` | single5: 570 |
| viking | `A[] not (Viking1.safe && Viking1.unsafe)` | viking10: 45566 |

**The old csma query was broken.** `queries_linux_old/csma.q` was
`A[] not (S1.Send && S2.Send)`, which is **violated after 3 explored states** — so any csma
number taken with it measured nothing. `prepare_models.py` would also have rewritten it, and
its Linux column named `S1.cs`, a location csma does not have. Both are fixed; `prepare_models.py`
now writes `queries/` and reproduces these files byte-identically.

---

## Reproducing a row

```bash
cd benchmarks
# headline
OMP_NUM_THREADS=32 ../build-omp/ptctlmc --backend omp --omp-chunk 4 \
    models/csma14.xml queries/csma.q
# legacy engine
PTCTLMC_ENGINE=legacy OMP_NUM_THREADS=32 ../build-omp/ptctlmc --backend omp --omp-chunk 4 \
    models/csma14.xml queries/csma.q
```

Engine/knob environment variables, all read once per process:

| variable | meaning |
|---|---|
| `PTCTLMC_ENGINE` | `adaptive` (default) or `legacy` |
| `PTCTLMC_MIRROR` | `adaptive` (default) / `fixed` / `off` — `off` is an **ablation**, not a historical mode |
| `PTCTLMC_HOTLEN` | pins length-only mirror promotion (pre-adaptive rule) |
| `PTCTLMC_HOTWORK`, `PTCTLMC_HOTMINLEN` | adaptive promotion gates (defaults 16, 8) |
| `PTCTLMC_HOTBUDGET` | mirror memory cap in MB (default: ¼ of RAM, clamped to [1, 32] GB) |
| `PTCTLMC_MERGESCAN` | convex-merge scan window (default 32; 0 = unbounded) |
| `PTCTLMC_BIN` | benchmark harness only: measure a different binary |

---

## Provenance of the reference build

The 2026-07-29 results were produced by `PTCTCL@4d16d13` + `HDBM@ad14ac4` + an uncommitted
`try_union` incremental-close fix (now `HDBM stash@{0}`, and preserved as
`HDBM/include/ptctlmc/dbm_ops_legacy.hpp`). Confirmed from the reflogs: PTCTCL sat at `60851d4`
from 07-27 12:52 until `4d16d13` was committed 07-30 11:07, and HDBM sat at `ad14ac4` from the
07-27 clone until 07-31. The compiled binary is preserved at `build-spin/ptctlmc`.
`PTCTCLworkspace/legacy/` holds a worktree of that exact combination.
