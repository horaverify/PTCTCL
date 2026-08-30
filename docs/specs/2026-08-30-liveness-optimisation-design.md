# Liveness engine: time and memory, without changing a single verdict

Date: 2026-08-30. Baseline: `24cc44b` ("Final algo").

Scope: the liveness engine only — `p --> q` (`leads_to_otf`, `leads_to_clocks`) and `E[] psi`
(`exists_invariant_run_otf`). The reachability *algorithm* is not touched. Three shared HDBM
primitives and one shared `netreach.cpp` function are, so reachability inherits the speedups and
joins the verification gate; see [Blast radius](#blast-radius).

Every change here is **exact**: same verdicts, and — except where explicitly noted for the two
that change search order — the same stored-state counts. Nothing is tuned to a benchmark, and no
new magic constant is introduced.

## 1. What the profile says

`HORA_ONLINE_PROFILE` splits forward construction from the backward greatest fixpoint. The
split is model-shaped, not uniform, and no single change can therefore serve every family:

| family | forward | backward | why |
|---|---|---|---|
| arbiter, fischer, twolock, csma | 85–98 % | ~0 | the `¬q`-closure is **acyclic with zero stopping nodes** — 0 `Pre` calls |
| phil | 24 % | 76 % | 96 nontrivial SCCs, largest 5054 |
| grc | 11 % | 89 % | one SCC of 84 474 (grc6); ~27 relaxes per node |
| single | 100 % | 0 | one discrete key, a 9164-zone antichain |

Two measurements reframed the problem:

- **`stopnodes = 0` on every benchmark measured.** The stopping term of the fixpoint operator is
  empty everywhere, so `W = W ∧ Pre(W)` and a non-empty value requires an infinite path in `S`.
- **grc's cost is antichain width, not state count.** grc6 has **920 discrete keys** and
  **139 476 active zones** — 152 zones per key, and 357 active records scanned per intern
  (`intern_scan = 580 751 535` over `intern = 1 626 047`).

## 2. Findings, and the fix for each

All landed. Numbers are `Release` against `Release`, both engines re-run on the same machine
(the recorded reference results predate this hardware and are not comparable).

### F1 — `dbm_free` runs a Floyd–Warshall it does not need *(HDBM)*

`Dbm::free(k)` is called only from `gfp_pre` / `gfp_pre_avoid`, once per clock dead at the target
plus once per reset clock — 8–12 times per `Pre` call. Each call ends in `dbm_close`, `O(dim³)`.

Freeing only *relaxes* constraints, so no entry can tighten; and the projection's tight bounds on
the surviving clocks are already the closed `d[i][j]`, because `d[i][j]` was the minimum over all
paths *including* those through `x_k`, and every such path is a valid consequence of the zone that
survives projection. **The proof is already written in the same header**, for `dbm_free_many`,
which the reach engine uses; the single-clock version never received it.

`dbm_down` has the same shape: it relaxes row 0 to `kZeroLe` and closes. One pass suffices —
`d[0][j] := min(kZeroLe, min_{i≥1} d[i][j])` — because only row 0 relaxes, and no row `i ≥ 1` can
tighten through it (`d[i][0] + d[0][j]_new ≥ d[i][0] + d[0][j]_old ≥ d[i][j]`, and for the
minimising `k*`, `d[i][k*] ≤ d[i][0] + d[0][k*]_old ≤ d[i][0]`).

> Backward pass: phil9 3.3×, phil10 3.7×, phil11 3.8×.

### F2 — `dbm_try_union` analyses eagerly on pairs that fail immediately *(HDBM)*

`facets_ok` counts strict facets, runs `mindbm_analyze` (`O(dim³)`) when there are at least eight,
then tests one crossing region per surviving facet. Most pairs escape on the first or second
region, so the analysis is paid to prune facets that are never reached.

Test regions as they are found and start the analysis only after `mindbm_analyze_min_facets()`
region tests have passed, then use it to skip the remainder. This reuses the existing threshold —
no new constant — and removes the separate facet-count pass. Verified result-identical: `explored`
is the same at `HORA_ANALYZEMIN` 0, 8 and 10⁵ on grc5 (12 184) and single7 (9164). On this
suite the analysis then never pays at all; the lazy scheme keeps the path for workloads where
pairs mostly succeed.

> single7 19.0 s → 7.3 s. try_union + `mindbm_analyze` was 54 % of grc5's backward and 85 % of
> single7's whole run.

### F3 — `gfp_pre` recomputes per-key data every call and loads two zones it never reads

Per call it runs `all_invariants` twice (an `O(nprocs)` scan into a fresh heap vector each),
`frozen_flags` twice, an `O(nclocks × nprocs)` triple-indirect dead-clock scan, and
`sread(v)` / `sread(j)`, which decompress two full DBMs out of the `ZoneStore` — `gfp_pre` reads
only `locs` and `ints`.

Cache `{invariant constraints, frozen flag, dead-clock byte mask}` per node, lazily, in `gfp_on`;
pass it to `gfp_pre`; drop `sread` from the non-avoid path; free the dead set with a single
`dbm_free_many`, and precompute each tag's reset mask beside `tags_`.

> Backward pass: phil 19–21 %, grc5 9 %.

### F4 — `state_has_deadlock` builds the whole escape federation before asking whether one escape already covers `Z`

The "a single escape covers `Z`" short-circuit exists, but it runs after the `O(nprocs²)`
synchronisation enumeration and after `fed_add` has maintained `live` inclusion-minimal at
`O(|live|²)` DBM inclusion tests.

An escape that never tightened `Z` is detectable the moment it is produced: `guarded ⊆ Z` always,
and a canonical DBM is unique for a non-empty zone, so `guarded == Z` entrywise iff no constraint
was strictly tighter. `down(guarded) ⊇ guarded = Z`, so `Dead` is empty and the enumeration can
stop. One `memcmp` per transition, and it aborts the rest.

`*escapes` is then not populated. No caller reads it on a `false` return (`online_seed_key`,
`build_atom_map` and `dtheta_empty` all `return`/`continue`; `stop_set` and `deadlock_reachable`
never ask for it), and the contract says so.

> Forward pass: 8–10 %.

### F5 — `relax` builds every product even when one is already the answer

`nw = { z ∧ c : z ∈ W[v], c ∈ cand }`. If some `c ⊇ z` then `z ∧ ⋃cand = z` and every other
product `z ∧ c'` is a subset of it, so the inner loop is redundant — emit `z` and move on. If that
holds for every `z`, then `nw` denotes `W[v]` member for member: return "unchanged" without
`gfp_fed_compact` and without the `fed_subtract`-based `gfp_fed_subset`.

> Backward pass: 9–10 %.

### F6 — `gfp_fed_compact` tests inclusion one way and always settles twice

Add the reverse test (`z ⊇ o` → replace `o`), and run the settling pass only when the first pass
actually merged something: without a merge nothing was reshaped, so the second pass can only
repeat the first.

> Backward pass: grc5 13 %, phil11 9 %.

### F7 — `online_intern`'s convex-union join stops at the first success

Each successful `try_union` enlarges the candidate, which can make a previously non-convex pair
convex. Iterating to a fixpoint is still exact (`try_union` succeeds only when the union really is
one zone) and **reduces exploration**.

> grc5 13 980 → 12 184 states (−13 %), grc4 923 → 788 (−15 %), phil11 254 958 → 247 943, phil10
> 75 120 → 73 455. arbiter / fischer / twolock / csma / single unchanged.

The join loop is `O(width²)` per intern and grc6's antichains are 152 wide, so this was the one
change with a cost risk. It does not regress there: grc6's E[] row drops 75 212 → 68 026 states,
and no family's time regresses.

### F8 — atom-free cover probes decompress the stored zone for nothing

`Otf::online_covers` calls `zread(b, bsc)` unconditionally, then — when the key carries no
diagonal atom — hands the widened copy to `base_cover`. `alu_le_stored` / `stored_includes` read
the stored zone at its own entry type through `ZoneStore::with`, with no widening at all. 93 % of
grc6's keys carry no atom (63 of 920), and the probe runs 654 400 329 times.

> ~4 % on grc5 / phil10 / fischer9 / arbiter14; expected to matter more where antichains are wide.

## 3. Not yet built — the three fronts chosen

### Landing 4 — memory

Peak is ~1.04 KB per stored state on arbiter15, of which the zone is 289 B. The other 750 B:

- `OnlineKeyInfo` holds `locs` and `ints` as two heap vectors per key — ~260 B once headers and
  allocator overhead are counted. Flatten into `key_locs_` / `key_ints_` arenas with fixed stride;
  make `deferred` a lazily allocated pointer; narrow `node_count` / `active_count` to `int`.
- A 68 B/key LU slice caches a fold that should not be expensive in the first place. For each
  process `p`, precompute `Vary_p = { c : lbound_loc[p][·][c] or ubound_loc[p][·][c] varies with
  the location }` and the constant floor contributed by the rest. `online_configure_bounds` then
  folds `Σ_p |Vary_p|` entries instead of `nprocs × nclocks` — 17 instead of 289 on arbiter — so
  the per-key slice can be deleted and recomputed on demand. This also cuts `register_key`, which
  is 40 % of arbiter's forward phase.
- `online_raw_adj_` and the final `adj` are both `vector<vector<OtfEdge>>`: 24 B of header plus
  ~16 B of allocator per node. Both are append-only per node, so a chunked arena with
  `(offset, count)` fits exactly. The final `OtfEdge` no longer needs `deferred`: 12 B → 8 B.
- Finalize builds `compact_nodes` / `compact_zones` / `compact_adj` while the originals are still
  alive — an 86 MB spike on fischer10. Compaction only ever drops, so `remap[old] ≤ old` and the
  whole thing can be done in place.
- `gfp_on` allocates six `N`-sized vectors (`W`, `nosucc`, `psi_seed`, `inS`, `seeded`, `rev`)
  where only `|S|` entries are ever used — `|S|` is 30 % of `N` on fischer9. Index by position in
  `S`; allocate `psi_seed` only when `confine`; store `rev` as CSR.

Estimate: 25–35 % of peak RSS.

### Landing 5a — prove the fixpoint empty structurally

`W[v] ≠ ∅` requires an infinite path inside `S` or a stopping node. So, using the Tarjan run that
`gfp_on` already performs:

```
Live = backward-cone_S( {v : v is in a nontrivial SCC, or has a self-loop}
                      ∪ {v : nosucc[v] ≠ ∅} )
```

Pin every `v ∈ S \ Live` to `∅`, and skip the sweep entirely when `Live` is empty. Exact, and
essentially free. On arbiter, fischer, twolock and csma the closure has zero nontrivial SCCs and
zero stopping nodes, so this decides the query with no zone work and no `W` allocation at all.

`nosucc` is already computed for all of `S` before the sweep, so the cone costs one `O(|S| + E)`
pass. Whether `nosucc` can additionally be *deferred* — evaluated in reverse-topological order and
skipped for nodes that already have a `Live` successor — is a measurement, not a commitment.

### Landing 5b — early violation witness for `p --> q`

`E[] psi` has `eg_early_`: a reached node that is time-divergent or genuinely deadlocked is a
concrete maximal-run witness, with no subsumption edge in the certificate. `p --> q` never uses
it, though a reached node with `is_p ∧ is_nq ∧ stop_set ≠ ∅` is exactly a counterexample —
`EF(p ∧ EG¬q)` witnessed by a run that stops right there.

Enabling it lets the forward search stop long before the graph is complete on violated queries.
When `p` carries clock atoms the witness must be intersected with `p_region(v)`.

This deliberately changes `explored` on violated queries — that is the point — so those rows of
the differential compare verdicts only.

## 4. Blast radius

| primitive | call sites | reachability affected |
|---|---|---|
| `dbm_free` | all in `netliveness.cpp` (`gfp_pre`, `gfp_pre_avoid`) | no |
| `dbm_down` | liveness ×5, **plus `netreach.cpp` `state_has_deadlock`** | yes — every `A[] not deadlock` and `deadlock_reachable` |
| `dbm_try_union` | liveness ×4, **plus `netreach.cpp:895`**, the reach passed-list merge | yes — the reach hot loop |
| F4 early-out | `netreach.cpp` `state_has_deadlock` | yes |

The CUDA and Metal kernels use `dbm_reset` and `dbm_delay`, not `dbm_free` or `dbm_down`;
`dbm_try_union` is already host-only. Measured on reachability, baseline against prototype:

```
A[] not (P1.cs && P2.cs) etc.   identical verdicts and state counts, times within noise
A[] not deadlock   fischer7  1.99s -> 1.36s  1.46x   81720 states, unchanged
                   csma7     3.73s -> 2.79s  1.34x   68839 states, unchanged
                   twolock7  2.35s -> 1.60s  1.47x   96736 states, unchanged
```

## 5. Landings

Each is independently buildable and independently gated by §6.

1. **HDBM primitives** — F1, F2, plus property tests. Separate commit in the HDBM repository.
2. **Backward operator** — F3, F5, F6 in `netliveness.cpp`.
3. **Forward construction** — F4 (`netreach.cpp`), F7 and F8 (`netliveness.cpp`).
4. **Memory** — the whole of Landing 4 above, one bullet at a time, differential after each.
5. **Structural proof and early exit** — 5a then 5b.

Landing 4 touches the hottest data structures and goes last for that reason.

## 6. Verification

- `ctest` (version, selftest, unit, bisim) after every landing.
- **Property tests for Landing 1**, in `HDBM/test/` and mirrored into `tests/dbm_tests.cpp` so
  they run under the normal `ctest` (HDBM's own test target builds only when HDBM is the
  top-level project). Over randomly generated closed zones: the new `dbm_free` and `dbm_down`
  agree entrywise with the old close-based versions, and `dbm_try_union` accepts exactly the same
  pairs as the always-analyse and never-analyse variants. These are the properties the arguments
  above claim, so they are what the tests should assert.
- **`benchmarks/scripts/live_diff.py`**, committed: baseline binary against new, over
  `live_queries` × sizes 4–10, `eg_queries` × sizes 4–7, `reach_queries`, and `A[] not deadlock`
  across the families. Records verdict, `explored`, wall time and peak RSS per case; aborts on the
  first verdict disagreement. Alternates run order per case so cache effects do not favour one
  build. This is the gate.
- Peak RSS per case from `/usr/bin/time -l`; phase-level RSS from the `HORA_RSS` diagnostic.
- `tests/models/liveness-regressions` and `tests/run_all_ta.sh`. `verifyta` is not installed on
  this machine, so the differential oracle is the baseline binary; the `verifyta` agreement
  already recorded for the baseline transfers by verdict equality.

## 7. Diagnostics kept

Two env-gated diagnostics, in the style of the existing `ONLINE-PERF`, because they are what made
this analysis possible:

- `HORA_GFP_PROFILE` — `|S|`, `N`, SCC count and largest component, relax visits, `Pre` calls,
  stopping nodes, federation width.
- `HORA_RSS` — resident set at end of forward, end of finalize, end of backward.

## 8. Explicitly out of scope

**The grc backward wall.** `grc6 T0.in --> T0.far` takes 833 s with the Landing 1–3 prototype:
88.8 s forward, **744 s backward**, `|S| = 122 072` with a single SCC of 84 474, 3.36 M relax
visits and 78.6 M `Pre` calls. The relax count per node (~27) tracks the model's clock constants,
so the levers are per-relax cost — which F3, F5 and F6 attack — and iteration order. Two
principled candidates were identified and are **not** being attempted here:

- re-running Tarjan on the surviving subgraph as nodes drain to `∅`, so a large component peels
  into smaller ones (the SCC-refinement shape of symbolic fair-cycle algorithms, driven here by
  zone emptiness);
- Bourdoncle's weak topological order inside a component in place of the flat FIFO.

`DESIGN.md` records that reordering changes intermediate federation shapes and that `fedcap`
bounds those — a chaotic worklist previously turned grc4 `T0.in --> T0.far` from SATISFIED into
UNUSABLE. Either candidate must be measured against that failure mode before it is believed.

A second recorded observation for whoever picks this up: grc's forward phase is **`O(nodes ×
antichain width)`** in cover probes, and the width is 152 on grc6. A cheap exact reject prefilter
ahead of the `a_LU`/D probe is the obvious lever there, and F8 is only the first, smallest piece
of it.

## 9. Results

All verdicts agree with the baseline. `explored` differs only where the iterated convex-union
join legitimately merges more (F7), or where the early witness settles a violation sooner.

### Whole suite, one machine, both engines re-run

408 cases — `live`, `eg`, `violated`, `reach`, `deadlock` × sizes 4–9 — **0 verdict mismatches**.

| suite | cases | time | peak RSS |
|---|---|---|---|
| live | 44 | 2.30× | 1.10× |
| eg | 210 | 2.59× | 1.00× |
| violated | 66 | **21.8×** | **2.39×** |
| reach | 44 | 1.12× | 1.00× |
| deadlock | 44 | 1.18× | 1.00× |

Reachability state counts are **identical on every row**. Its speedup comes entirely from the
shared HDBM primitives, and its memory is unchanged, as intended — no reachability algorithm was
touched.

### Per model, leads-to

```
             time                       peak RSS
arbiter15    2.45s -> 2.03s   1.21x     336MB -> 299MB   1.12x
fischer10    2.92s -> 1.93s   1.51x     528MB -> 371MB   1.42x
twolock10    2.34s -> 1.84s   1.27x     554MB -> 401MB   1.38x
phil10       2.59s -> 0.75s   3.45x      98MB ->  88MB   1.11x
phil11      18.57s -> 4.09s   4.54x                      252MB
grc5         4.70s -> 2.30s   2.04x      22MB ->  19MB   1.16x
csma10       0.19s -> 0.17s   1.12x      44MB ->  37MB   1.19x
single7     18.20s -> 7.11s   2.56x
```

### Violated leads-to

The suite for this is new (`benchmarks/live_queries_violated`); every pre-existing leads-to row,
ours and every recorded `verifyta`/`tchecker` row alike, is satisfied.

```
fischer7  P0.wait --> P0.cs     11951 ->    9 states   x54
twolock7  P0.waitB --> P0.reqB  12847 ->  121          x47.5
fischer7  P0.cs --> P0.req      11951 ->   37          x18
grc5      T0.near --> T0.far    13980 -> 3519          x71.6
grc6      T0.near --> T0.far    baseline >45s -> 2.37s
grc6/7    T0.far --> T0.in      baseline >45s -> ~6ms
```

Peak RSS on those rows falls by up to 31× (`fischer9` 221.6 MB → 7.1 MB), because the search
stops instead of building the graph. Cycle-witness rows are unchanged in both, which is the
evidence that the witness does not fire where it must not.

### Large models, and the reference comparison

Wall time is not comparable across machines, so the recorded `verifyta`/`tchecker` results are
not a timing baseline. **Peak RSS is**, which makes the memory column the one that can be read
against them directly. Both HORA builds were re-run here; the reference columns are the
recorded ones.

```
model        base GB   new GB      x  |  tchecker  vs tck  |  time base -> new
arbiter15      0.326    0.252   1.29x |     0.273   1.08x  |   2.47s -> 1.98s  1.25x
arbiter16      0.743    0.550   1.35x |     0.563   1.02x  |   5.88s -> 4.77s  1.23x
arbiter17      1.572    1.170   1.34x |     1.182   1.01x  |  13.95s -> 11.34s 1.23x
arbiter18      3.212    2.552   1.26x |     2.500   0.98x  |  32.69s -> 26.50s 1.23x
fischer10      0.529    0.280   1.89x |     1.089   3.89x  |   2.88s -> 1.87s  1.54x
fischer11      1.694    0.920   1.84x |     4.986   5.42x  |  11.25s -> 7.44s  1.51x
twolock10      0.541    0.273   1.98x |     0.914   3.35x  |   2.37s -> 1.76s  1.35x
twolock11      1.704    0.929   1.83x |     4.153   4.47x  |   9.31s -> 7.00s  1.33x
csma12         0.221    0.139   1.59x |     0.136   0.98x  |   1.34s -> 1.10s  1.22x
phil11         0.338    0.188   1.80x |     5.566  29.61x  |  17.73s -> 3.97s  4.47x
```

Verdicts agree on every row. We are 3.4–29× under `tchecker` where the quotient wins on state
count, and at **parity** — arbiter18 and csma12 marginally worse — where it does not. Before this
work arbiter18 stood at 0.78× of `tchecker`.

### Per-state footprint, and a statistic that does not mean what it looks like

Normalising total memory by states is tempting and, across tools, wrong. `bench.py` scrapes
`verifyta`'s count from **`States explored`** — a cumulative visit counter — while ours is
expanded nodes. Dividing resident memory by the first gives bytes per *visit*, not bytes per
stored state, and for a tool that re-explores those differ by orders of magnitude.

The arithmetic makes it concrete. On phil11 (22 processes, 11 clocks) the floor for **one**
stored state is about 58 bytes: ~6 for a bit-packed discrete state plus ~53 for a minimal-graph
zone at dimension 12. `verifyta`'s recorded 0.061 GB over 2 120 725 counted states is **30.9
bytes**, below that floor — so the counted number is not the resident set, and no conclusion
about footprint can be drawn from it.

What the recorded numbers *do* show is the algorithm. UPPAAL decides `p --> q` as
`A[](p imply A<> q)` by running a nested `A<>` search from each reachable `p`-state: the passed
list only ever holds one of those searches, so memory stays flat while the explored counter
accumulates across all of them. The throughput gives it away —

```
verifyta   arbiter7   183524 explored   127.0s  ->    1445 states/s
           csma7      324899 explored    97.4s  ->    3334 states/s
           twolock7   282072 explored    27.5s  ->   10251 states/s
           phil11    2120725 explored     9.7s  ->  217957 states/s
```

— 1445 states/s being nowhere near UPPAAL's reachability throughput.

Head to head on every model where both have a recorded row:

| | verifyta | ours | |
|---|---|---|---|
| arbiter7 | 127.0 s, 0.035 GB | 0.004 s, 0.007 GB | 5× less memory |
| csma7 | 97.4 s, 0.049 GB | 0.01 s, 0.008 GB | 6× less |
| fischer7 | 111.6 s, 0.065 GB | 0.03 s, 0.014 GB | 4.6× less |
| twolock7 | 27.5 s, 0.043 GB | 0.03 s, 0.014 GB | 3× less |
| phil11 | 9.7 s, 0.061 GB | 4.15 s, 0.189 GB | 3× more |

So we are under `verifyta` on memory on four of five, and far faster on all five. phil11 is where
its side of the trade pays, not the general case.

The real distinction is architectural and should be stated as such rather than as a compression
gap: **we materialise an edge-labelled graph and solve one backward fixpoint on it; UPPAAL keeps
no graph and re-explores.** Memory-heavy and fast against memory-light and slow.

### Where our own footprint goes, and what compressing the zone buys

Measured, our ~827 bytes a stored state on phil11:

| | B/state |
|---|---|
| ZoneStore, all records (dense `dim²`, retired ones included) | 204 |
| key arenas + `OnlineKeyInfo` | 137 |
| adjacency edges | 67 |
| adjacency vector-of-vectors headers | 40 |
| per-node side arrays | 38 |
| backward `W` federations | ~156 |

`--zones packed` stores each zone as the minimal graph `mindbm` packs instead of the dense
matrix, and unpacks it on every read. Verdicts and state counts are identical — the codec round
trips a canonical DBM bit for bit — so it trades memory against time and nothing else.

```
model        states | dense GB  packed GB  saved | dense s  packed s   cost
arbiter15    278528 |    0.251     0.196  1.28x |    2.20      8.47  3.85x
arbiter16    589824 |    0.551     0.428  1.29x |    5.34     21.70  4.06x
arbiter17   1245184 |    1.165     0.884  1.32x |   12.71     53.75  4.23x
fischer10    447598 |    0.279     0.244  1.14x |    1.99      5.67  2.85x
fischer11   1464971 |    0.921     0.758  1.22x |    7.75     22.70  2.93x
twolock10    457838 |    0.274     0.251  1.09x |    1.86      5.45  2.93x
twolock11   1487499 |    0.916     0.743  1.23x |    7.43     21.84  2.94x
csma12       120832 |    0.138     0.117  1.18x |    1.14      2.98  2.61x
phil10        73455 |    0.060     0.056  1.07x |    0.76      1.56  2.05x
phil11       247943 |    0.191     0.165  1.16x |    4.15      9.16  2.21x
grc5          12184 |    0.015     0.015  1.00x |    2.40      4.10  1.71x
```

1.0–1.3× of memory for 1.7–4.2× of time, so dense stays the default and the choice is the
user's. The codec delivers exactly what it promises — on arbiter17 the zone goes from 289 bytes
to about 53, a 5.5× compression — but the zone is only a quarter of what a stored state costs, so
the total barely moves. `grc5` is the clearest case for leaving it off: small zones already, and
it scans 357 active records per intern, so it pays the unpack with nothing to show.

## 10. What was measured and rejected


- **A whole-zone stop-set shortcut.** `NoSucc(v) = Z_v` puts `v` at its fixpoint immediately.
  Implemented, verified against 102 queries, measured — and it never fires: the stopping term is
  **empty at every node of every model in the suite**, leads-to and E[] alike. Removed rather than
  carried. The corollary is worth recording: on this suite the backward fixpoint is purely
  `W = W /\ Pre(W)`, and its memory is the transient `W` federations of the descent, not `nosucc`.
- **The rest of the D reserve.** The online search materialises far more records than it keeps
  (614212 against 447598 on fischer10) and every excess record is UNEXPANDED when dropped; they
  are held only because a later D atom could revoke the cover that retired them, and on arbiter,
  fischer, twolock and phil D never gains an atom at all. Two ways to reclaim them were sized:
  *batch pre-reduction* is worth nothing (`deact_sib = 0` — not one retirement is by a same-batch
  sibling), and *freeing the retired zone slot* needs a slot free list, which breaks the direct
  node-to-slot map the in-place finalize compaction relies on, plus an 8-byte replay recipe per
  record wherever the cover is a_LU-only rather than literal inclusion (100% of retirements on
  phil and csma, 90% on grc, but 0% on fischer and twolock). Netted against the ~121 bytes a slot
  it recovers: 1.2–4.3% of peak per family, for an indirection on a path called 654M times.
  Declined. What sizing it *did* turn up was two straight copies of data that already existed —
  the placeholder `node` during construction and the per-record copy of each key — which are the
  two changes that took peak memory from 1.1× to 1.4–2.0×.
- **`comps.shrink_to_fit()` and a flatter `PreNode`** as memory fixes for the E[] regression: both
  made it *worse*. Bisecting across the branch and instrumenting the E[] phases found the actual
  cause (the invariant arena at ~120 B/node); interning the spans fixed it.
