# Fully-GPU network backend (Metal + CUDA) — design

**Date:** 2026-07-18
**Status:** approved (brainstorming) → pending implementation plan

## 1. Goal

Make the GPU backends (`--backend metal`, `--backend cuda`) support **networks of timed
automata** (e.g. Fischer) with **full feature parity to the OpenMP path** and **full GPU
parallelism**. Today `--backend metal`/`cuda` on a network silently runs **fully serial**:
`reachable_net` (src/netreach.cpp:320) sets `parallel = (backend == Backend::Omp)`, so any GPU
backend falls through to the sequential path. The existing GPU engines
(`engine_metal.mm`, `engine_cuda.cu`) only handle a **single** timed automaton
(`reachable`/`reachable_metal`): one GPU thread per (frontier-state, edge) for successor zones,
with a **host-side serial** subsumption merge.

Networks add everything that engine can't do: a multi-process discrete state (`locs[]`),
integer variables with guards/updates, binary + broadcast channel synchronisation, committed/
urgent locations, and the sharded aLU-subsumption + convex-union passed-list merge.

**Primary payoff:** the user builds CUDA on an H100. A fully-parallel GPU network backend lets
that hardware accelerate networks, which no CPU can match at scale.

## 2. Non-goals

- Beating the 12-core OpenMP backend *on this Mac*. This workload is branchy/divergent
  (edge enumeration, sync pairing, dynamic antichains) — the GPU's weak spot. Metal-vs-omp on
  the Mac may be a wash or a loss; that is acceptable and will be reported honestly. The H100
  is the target.
- Diagonal-constraint networks beyond what the CPU path already supports (the exact-zone
  `ExNoDiag` fallback is kept identical; no new diagonal handling).
- Changing the CPU (`seq`/`omp`) semantics. The CPU path stays the correctness oracle.

## 3. Approach: fully-GPU, staged (chosen over hybrid / big-bang)

Successor generation **and** the subsumption merge run on the GPU. Delivered in feature stages,
each validated to match `omp` **exactly** (state counts + verdict) before the next begins.

## 4. Architecture — host orchestrates, GPU does the work

Keep the level-synchronised BFS loop on the host (mirrors the single-TA engine). Each per-level
phase is a GPU kernel; the host reads back only three scalars per level: `target_found`,
`next_frontier_size`, `done`. No per-state data crosses the bus mid-run (unified memory on Mac;
device-resident on H100).

Per level, four kernels:

1. **Expand** — successor generation. Work item = one GPU thread per (frontier-state,
   candidate-transition). Transition count per state is data-dependent, so use the standard
   GPU expand pattern: a **count** pass (thread per frontier state counts enabled transitions),
   a **prefix-sum** over counts, then a **generate** pass (thread per (state, transition-slot)).
   Each generate thread runs the full successor pipeline: committed filter → int_guard →
   `z1 = s.zone` → clock_guard ∩ → resets → build `ns.locs/ns.ints` → invariants ∩ →
   delay (unless committed/urgent) → invariants ∩ → reset_inactive → extrapolate → emptiness →
   compute packed `dk`. Emits `{dk, locs, ints, zone, valid}`.
2. **Group** — assign each valid successor to its discrete-state bucket via a GPU
   open-addressing hash table `dk → bucket` (find-or-insert with atomics). Atomic-append the
   successor into that bucket's "new arrivals" list. No GPU sort required.
3. **Merge** — **one GPU thread per active `dk`-bucket**. Each thread runs the *identical
   sequential* logic the CPU `add_to` does, but only for its bucket: aLU-subsumption (drop new
   zones subsumed by a stored zone; drop stored zones subsumed by a new one) then convex-union
   (`try_union`). Survivors are appended to the passed list; those new this level are flagged.
4. **Compact** — prefix-sum + gather the flagged survivors into the next frontier array.

**Why the merge parallelises well:** the passed list is an antichain *per discrete state*, and
distinct `dk`s are fully independent. OpenMP parallelises across `S = nthreads` hash-shards; the
GPU parallelises across *every distinct discrete state touched this level* — typically thousands
of independent buckets. Within a bucket the merge is sequential but small. It reuses the exact
DBM ops, so the GPU runs validated logic.

## 5. Data structures

**Network encoding (built once, host → GPU buffers):**
- Per-process edge tables in CSR form: `edges_from` offsets per (process, location) → edge
  records. Each edge: `src_loc, tgt_loc`, offset/len into `clock_guard`, `clock_resets`,
  `int_guard` (IntAtom), `int_updates` (IntAsgn), and (Stage 2) `sync_chan, sync_dir`.
- Per-location invariants (constraint array + offsets), `committed`/`urgent` flags.
- Active-clock table for `reset_inactive` (optional; can be deferred, must stay sound).
- `dshift[]` bit-packing table + int-var `lo[]`; global `L`/`U` and per-location `lbound_loc`/
  `ubound_loc` for per-state aLU bounds (`state_bounds`).
- Broadcast flags per channel (Stage 2).

**Frontier state:** parallel arrays `front_locs[nprocs·N]`, `front_ints[nintvars·N]`,
`front_zone[dim²·N]`; `dk` recomputed on-GPU from locs/ints.

**Passed list:** open-addressing hash table (keys = `dk` uint64, values = bucket metadata:
offset into a global zone pool + count + capacity). Zone pool is append-only via atomic bump.
Bucket growth handled by per-bucket capacity + overflow chunks (design detail for the plan).

**Limits:** `kMaxDim = 24` (≤23 clocks; matches CUDA). Throw a clear error above it, as today.

## 6. Staging & validation

Each stage must reproduce `seq`/`omp` **state counts and verdict exactly** on its models before
the next stage starts (per superpowers:verification-before-completion — evidence before claims).

- **Stage 1 — shared-variable networks (Fischer-class):** location edges + int guards/updates,
  **no channels**. Internal de-risking ladder:
  - **1a:** GPU **Expand** + *host* subsumption merge. Validates successor generation against
    the CPU oracle without yet tackling GPU merge.
  - **1b:** move Group+Merge+Compact to the GPU. Validates the full-GPU pipeline.
  - **Target models:** Fischer N6–N12 (counts 3458 / … / 447598 / 1464971 / 4758404); the
    2-int collision model `net2int` (9 states); any shared-var test models.
- **Stage 2 — channels:** binary sync (sender/receiver pairing: Expand gains a work category
  for `(state, sender-edge, receiver-edge)` pairs on a matching channel), then broadcast (sender
  fires with all enabled receivers — a bounded participant product). Validate on `bcast`,
  `handshake`, and other channel models.

**Benchmarks:** report Metal vs omp vs verifyta honestly across N; add new network benchmarks
(see §8) so parity is tested on more than Fischer.

## 7. CUDA mirror

Every kernel is authored in MSL and mirrored to CUDA in lockstep, keeping
`reachable_net_metal` / `reachable_net_cuda` signatures identical (as the single-TA engines
already are). New header `include/ptctlmc/net_gpu.hpp` declares both entry points behind a
CPU-safe interface (no GPU types leak), following `cuda_reach.hpp`. `reachable_net` dispatches to
them when `backend ∈ {Metal, Cuda}` and the network is within limits, else falls back to the
existing seq/omp path (so unsupported features degrade gracefully, never wrongly).

## 8. New benchmarks (requested alongside this work)

Add three network generators/models beyond Fischer to exercise parity and scaling:
1. **CSMA/CD** (N stations) — shared bus, collisions; classic timed-automata benchmark, shared
   variables + a broadcast-ish contention pattern.
2. **Train–Gate** (N trains) — binary channel sync (approach/leave/go/stop) + a controller queue;
   exercises Stage 2 binary channels.
3. **Lynch–Shavit / token-ring mutex** (N nodes) — shared-variable mutex variant distinct from
   Fischer; different discrete-state shape and int-var usage for the packed key.

Each with a generator (`benchmarks/generators/`), a mutual-exclusion or reachability query, and
verifyta cross-check.

## 9. Risks & open questions

- **GPU merge correctness** is the highest-risk piece (dynamic hash table + atomics + per-bucket
  sequential merge). Mitigated by the 1a→1b ladder (host-merge oracle first) and exact-count
  validation.
- **Convex-union order-sensitivity:** a bucket's zones may merge in a different order than seq,
  which *could* shift the exact stored-zone count. But seq and omp already match exactly under
  different orders, implying `try_union` effectively doesn't fire on Fischer — so exact matches
  are expected. Verdicts are sound regardless of order.
- **Passed-list memory:** Fischer N12 ≈ 4.7M zones × dim²(169)×4B ≈ 3.2 GB. Fine on H100 (80 GB);
  tight on Mac unified memory for large N — cap/report, don't crash.
- **Broadcast sync** (Stage 2) has a variable-arity participant product; needs a bound to avoid
  combinatorial blow-up in a single kernel thread.
- **Metal may lose to omp on the Mac** — expected; report honestly.

## 10. Success criteria

- `--backend metal` and (build-permitting) `--backend cuda` run networks on the GPU, fully
  parallel, with **no serial fallback** for supported feature sets.
- Identical state counts + verdicts to `seq`/`omp` on Fischer N6–12, the collision model, the
  channel test models, and the three new benchmarks.
- Unit tests + all existing models still green; CPU paths unchanged.
- Honest Metal-vs-omp-vs-verifyta benchmark table; CUDA kept in sync for the H100.
