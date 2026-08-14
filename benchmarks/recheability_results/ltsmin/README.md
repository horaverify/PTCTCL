# `clean_results/ltsmin/` — the opaal+LTSmin baseline

opaal + LTSmin (`opaal2lts-mc`), the FORMATS 2012 "Multi-Core Reachability for Timed
Automata" tool, run as a parallel baseline alongside PTCTLMC and UPPAAL `verifyta`.

Driver: `benchmarks/run_ltsmin_complete.sh`. Started 2026-08-10.
Stop it with `touch clean_results/ltsmin/STOP` (finishes the running job, then exits).

| file | what it is |
|---|---|
| `ltsmin_t16_2h.csv`, `ltsmin_t24_2h.csv`, `ltsmin_t32_2h.csv` | one CSV per thread count |
| `bench_t<T>_2h.log` | per-job harness log (one line per run, live) |
| `driver.log` | the sweep driver's own log: which job started when |

There is no chunk dimension. `--omp-chunk` is a PTCTLMC knob and means nothing to LTSmin,
so where the PTCTLMC sweep has `t<T>_c<C>` this has `t<T>` only. The `mergescan` / `hotlen`
columns are `-` for the same reason.

## Methodology — identical to the PTCTLMC sweep

Same families, same sizes, same budget, so the columns line up row for row with
`results_old/results_complete/adaptive/ptctlmc_t<T>_c<C>_adaptive_2h.csv`:

- **2 h timeout, 100 GB RSS guard** per (model, threads, rep); over either is recorded
  `TIMEOUT` / `MEMOUT`, not a crash
- **2 reps**, **strike-limit 1** (one timeout in a family skips every larger size)
- **pinned**: `taskset -c 0-<T-1>`, exactly as many cores as threads
- **threads** 16, 24, 32
- **cheap sizes first, expensive tails last**, so a complete cheap table exists early
- families `phil`, `single`, `twolock`, `viking`, `fischer`, `csma` — the five of
  `clean_results/verifyta.csv` plus `csma` (which has PINS modules and a PTCTLMC column,
  but no verifyta baseline row)

Resumable: the harness never repeats a `(model, tool, threads, rep)` that already has an
`OK`/`TIMEOUT`/`MEMOUT` row in the output CSV, so the driver can be restarted freely.

## `--state=table -s 25` — the opaal authors' own setting

Taken from their code rather than chosen here:

- `LTStest/src/opaal-ltsmin-succgen/bin/opaal_ltsmin:45-48` — `-s/--size` default **25**,
  `--state` default **table**
- `LTStest/authentic/opaal-ltsmin-succgen-r188/added-files-verbatim/bin/benchmark.sh:19` —
  their benchmark script: `opaal_ltsmin --state=table -s 25 --threads=$t -o $o -u$u $m`

LTSmin's *own* defaults are `-s 24` and `--state=tree`; earlier viking runs in this repo
used `tree -s 26`. Three measurements from 2026-08-10 that justify not sizing it up:

- **Overflow is loud, never silent.** phil12 at `-s 10`:
  `Error: hash table full! Change -s/--ratio.` and exit 255 → the harness records `ERR`.
  A store that is too small can therefore never masquerade as a short state count.
- **Sizing up costs real memory.** viking15 @32t: `tree -s 26` → 10.3 GB RSS,
  `tree -s 30` → 21.5 GB. The extra 2^30x8 B of table really is paged in, and against the
  100 GB guard that buys nothing but earlier MEMOUTs.
- **Memory, not table capacity, is the binding limit.** LTSmin spends ~2.4 KB/state here
  (viking15: 4.26 M states → 9.8 GB), so the 100 GB guard bites near ~40 M states, above
  what 2^25 = 33.5 M holds but reached first on the models that matter. viking20 (verifyta
  counts 143.7 M states) will MEMOUT at any table size.

Search order is `--strategy=sbfs`, and the modules are the prebuilt `benchmarks/so_linux/*.so`.

## Reading the numbers — two warnings

**State counts cannot compare tools.** `states_verifyta == 2*states_PTCTLMC - 2` exactly at
every size, and LTSmin counts differently again (phil4: 82 for both; viking6: 1854-1961 vs
verifyta's 1374; single4: 131-133 vs 105). These are counting conventions, not abstraction
advantages. **Compare wall-clock and scaling only.**

**LTSmin state counts and times are nondeterministic** under concurrency, in *both*
directions — subsumption races cause either re-exploration or luckier subsumption. Measured
on viking15 @32t: 4.25 M–8.3 M states and 7–19 s across probes. Sound either way, but it
means a single run means nothing: 2 reps are mandatory and the spread must be reported.

## Confound to disclose

These `.so` modules link a **thread-safety fix to UDBM** (`ENABLE_DBM_NEW=1` plus
`base::ConcurrentItemAllocator` on the two global free-list allocators, linking
`-ltbbmalloc`). Stock UDBM's process-global unsynchronised allocators segfault above one
thread, so *no* multi-core number is obtainable without it. `--threads=1` is bit-identical
to stock. The authors' own patched DBM (`UPPAAL-dbm-2.0.8-opaal3`) is lost with its host, so
byte-faithful reproduction is impossible — see the `ltsmin-opaal-baseline` note.
