# PTCTLMC — Parallel Timed-CTL Model Checker (UPPAAL-style), roadmap & design

**Goal:** a standalone C++/CUDA tool that checks **TCTL / UPPAAL-style queries** over
**timed automata**, parallelized with **OpenMP and CUDA** — doing for timed-automata
*zone-graph* verification what PCTLMC (sibling folder) does for CTL: keep the hot
inner loop data-parallel so it maps onto a GPU.

## Guiding decisions (mirroring PCTLMC, adapt after research)

1. **Oracle = `verifyta`** (UPPAAL, installed locally at
   `~/Documents/TimedAutomata/uppaal/uppaal-5.0.0-linux64/bin/verifyta`). Used for
   differential testing, not at runtime.
2. **Own the core for GPU-portability.** UPPAAL's **UDBM** (DBM library) and **UTAP**
   (parser) are open source and can be *studied / reused if license-compatible*, but
   a DBM is a small, well-understood dense matrix — we implement our own
   allocation-free DBM (the analogue of PCTLMC's bitset/CSR) so it runs on host *and*
   device. Parsing (UPPAAL XML) is heavier; we may use UTAP or a compact own parser
   for a textual TA subset. **Research phase decides.**
3. **Dual backend.** One zone-graph exploration + TCTL algorithm, behind a primitive
   interface, implemented for `seq` / `omp` / `cuda` — same pattern as PCTLMC.

## Core objects
- **DBM** (Difference Bound Matrix): (n+1)×(n+1) bounds over clocks; operations:
  canonicalize (Floyd–Warshall close), `up`/delay, `reset(x:=0)`, conjoin guard/
  invariant, emptiness, inclusion, LU/k-extrapolation (for termination).
- **Zone graph**: symbolic states = (location vector, DBM); successors via
  delay + edge (guard ∩, reset, target invariant ∩) + extrapolation.
- **Query subset** (what verifyta checks): `E<> φ` (reachability), `A[] φ` (safety =
  ¬E<>¬φ), `A<> φ` / `E[] φ` (liveness), `φ --> ψ` (leads-to).

## Goal

**Outperform UPPAAL** on timed model checking by exploiting GPU parallelism +
state-space reduction — not merely match it. Phase 1 builds a *correct, validated*
baseline (matches `verifyta`); Phase 2 makes it *faster*.

## Phase 1 — correct baseline (DONE, validated vs verifyta)
- **M0 — Research** ✅ UDBM (GPL→own DBM), UTAP (LGPL), verifyta oracle, zone algorithm.
- **M1 — DBM library** ✅ dense DBM, all ops as shared `__host__ __device__` code
  (`dbm_ops.hpp`), unit-tested (canonical form, inclusion, LU/max extrapolation).
- **M2 — TA model + parser** ✅ own minimal UPPAAL XML + expression/query parser →
  TA IR (clocks, locations+invariants, urgent/committed, edges+guards+resets).
- **M3 — Zone graph (seq)** ✅ passed/waiting with inclusion subsumption + max
  extrapolation; `E<>`/`A[]` reachability.
- **M4 — Differential vs verifyta** ✅ 6 curated models / 36 queries all match
  (incl. diagonals, urgent/committed). Harness `tests/validate_ta.py`.
- **M5 — OpenMP exploration** ✅ level-sync frontier BFS, parallel DBM pipeline +
  serial subsumption merge; `seq==omp` unit-tested.
- **M6 — CUDA backend** ✅ per-(frontier,edge) DBM pipeline on device, compiled for
  sm_90/H100 (host-orchestrated level-sync; not yet fully on-device).

## Phase 2 — optimizations + networks (DONE / measured)
- **P1 — Networks of TA + integer variables** ✅ parallel composition (location
  vectors + int values), committed-location priority. Fischer(N) validated vs
  verifyta. (Channels: still TODO.)
- **P3 — Active-clock reduction** ✅ per-location backward fixpoint; reset dead
  clocks → Fischer(5) state count 153,816 → 12,599.
- **P4 — LU extrapolation** ✅ (query-aware bounds keep it sound w.r.t. the property).
- **Passed list** ✅ per-discrete-state bucket + zone-inclusion subsumption + exact-
  DBM hash fast path.
- **Incremental close** ✅ O(dim²) constrain (was O(dim³) Floyd–Warshall each time).
- **2000-test differential** ✅ seq / omp / verifyta agree on 2000 random models
  (`tests/difftest_ta.py`), 0 disagreements. CUDA-ready.

## Benchmark results (honest)
- **Single-automaton, clock-heavy** (`gen_single.py`, n=6, 9033 zones): PTCTLMC
  **seq 1.6s / omp 1.3s vs verifyta 4.9s — we win ~3×.**
- **Fischer(N)** (networks): verifyta wins. State-count gap grows (N=2: 1.2×, N=4:
  4.6×, N=6: 44× — UPPAAL 3458 states vs our 152k). UPPAAL's zone abstraction on its
  flagship benchmark is tighter than ours; matching it is a research problem.

## Still open (to close the Fischer gap / finish)
- **P2 — Fully on-device on-the-fly** exploration (frontier + passed-with-inclusion
  on GPU). Current CUDA is host-orchestrated level-sync.
- **Tighter abstraction** to match UPPAAL on Fischer: federation subsumption
  (remove dominated passed states), better search order, minimal-DBM, symmetry.
- **P5 — Bisimulation quotienting** (state-space reduction).
- Channels (binary/broadcast) for CSMA/CD, train-gate; liveness `A<>`/`E[]`/`-->`.
