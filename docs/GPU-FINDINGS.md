# GPU (Metal) zone exploration: why it loses to the CPU

> **Scope.** Every result here is on a *unified-memory Apple GPU* with a 32 KB threadgroup
> budget. §2.3 — the dominant cause — is substantially a consequence of that budget, and a
> datacenter GPU relaxes it (H100: 228 KB shared memory per SM, ~3.3 TB/s dedicated
> bandwidth, ~8x the lanes). Because the workload is latency-bound with millions of
> INDEPENDENT zones, absolute throughput should scale with GPU size even at unchanged
> utilisation. Claim "GPU zone exploration loses on Apple Silicon, with this mechanism" —
> **not** "GPUs cannot do this". See §6.

Status: **negative result, structural.** Ten implementation attempts, all measured,
including three distinct DBM representations.
The fast path in PTCTLMC has no GPU dependency. This document records why, so nobody
re-runs these experiments — in particular §2.2, which refutes the most attractive idea
(wavefront / stream compaction) on measurement rather than on intuition.

Hardware used for every number below: **Apple M4 Pro** — 8 performance + 4 efficiency CPU
cores, 16-core GPU (~2000 shader lanes), unified memory.
Models: `viking15` (11 clocks → 12×12 DBM, 144 entries ≈ 576 B) and `fischer10`.

---

## 1. The headline measurements

| experiment | GPU | CPU | note |
|---|---|---|---|
| Successor expansion, identical work | 1400 ms | **491 ms** | CPU 2.85× faster |
| Per-bucket merge, zones written by GPU | 258 ms | — | GPU's best case |
| Per-bucket merge, zones written by CPU | 571 ms | **100 ms** | coherency penalty, see §2.6 |
| `--adaptive` free per-level choice | 1 level | **122 levels** | GPU chosen only for the calibration probe |

`--adaptive` calibrates both paths on early BFS levels and then routes each later level to
the measured-faster one. Given a completely free choice it picks the CPU on
**122 of 124 levels** (fischer10: 23 CPU / 1 GPU; viking15: 99 CPU / 1 GPU). The single GPU
level is the calibration probe itself.

Consequence: the configuration that wins Fischer (`--backend metal --adaptive` with
`PTCTLMC_HYBRID=1`, 0.42 s vs OpenMP's 2.09 s) does **100 % of its work on the CPU**. Its win
is algorithmic — a bucketed, batched per-discrete-state merge over compact arrays with no
zone interning — not GPU acceleration. Do not present it as a GPU result.

---

## 2. Why — seven causes that multiply

Naively the GPU has ~170× more lanes than the CPU has cores. Each of the following removes a
chunk of that advantage; together they remove all of it.

### 2.1 Each lane is individually slow
A shader lane runs ~1.4 GHz, in-order, no branch predictor, tiny cache. An M4 P-core runs
~4.5 GHz, out-of-order, with megabytes of cache. On **branchy scalar** code a lane is roughly
10–20× slower per unit of work. So ~2000 lanes are worth maybe 100–200 CPU-core-equivalents —
*if* fully utilised. The rest of this section is why they are not.

### 2.2 Divergence — HYPOTHESIS TESTED AND REFUTED
The natural theory is that lanes run in lockstep groups of 32, most edges fail at the first
guard, and one surviving lane drags 31 idle ones through the pipeline. **Measured, this is
false.** Lane survival (fraction of work items that yield a valid successor):

| model | work items | survivors | survive |
|---|---|---|---|
| viking15 | 8,355,840 | 8,355,840 | **100.0 %** |
| fischer10 | 3,317,950 | 2,372,190 | 71.5 % |

On viking there is *no* early-exit divergence at all — every lane does full work. Note also
that the **integer guard is already applied host-side** when the work list is built (see
`work_ints`, "post-update"), so the cheap non-DBM filter is already in place.

Consequence: **stream compaction / wavefront scheduling would recover ~0 % on viking and at
most ~28 % on fischer.** Do not implement it. This was the most attractive-sounding idea and
it is dead on measurement.

### 2.3 The working-set dilemma  ← THE actual mechanism
Because §2.5 leaves almost no parallelism *inside* one zone, every thread must own a whole
zone: 576 B at dim 12, up to 2.3 KB at MAXDIM 24. That creates a dilemma with no good side:

- **Keep it private** (`int z[MAXDIM*MAXDIM]`, the shipped kernel): far too large for the
  ~128–256 registers a thread gets, so it **spills to thread-local device memory** and all
  ~2600 DBM ops per successor run at device-memory latency.
- **Keep it on-chip** (threadgroup memory, attempts 8–9): 32 threads × 576 B = 18 KB of the
  32 KB threadgroup budget, so only ~1 threadgroup stays resident per core — about 32 threads
  per core. Occupancy collapses and there is nothing left to hide latency with.

Measured both ways on viking15 expand: private **1392 ms**, threadgroup 1599 ms, threadgroup
with bank-conflict padding 1575 ms. fischer10: private **187 ms**, threadgroup 278 ms, padded
297 ms. The private/spilling version wins because high thread count buys more latency hiding
than on-chip residency buys in access speed.

This is the crux: the zone is simultaneously **too big to keep on-chip at useful occupancy**
and **too big to keep in registers**. It is a direct consequence of §2.5 and is not a tuning
failure.

### 2.4 Arithmetic intensity is low — but this is NOT what is binding today
Each successor reads ~576 B and writes ~576 B for ~2600 integer ops: ≈2.3 ops/byte, against a
roofline ridge point of ≈10 ops/byte. So the workload is memory-bound *in character*, and on
unified memory the GPU shares bandwidth with the CPU, so there is no bandwidth advantage.

**However, we are nowhere near that bound.** Measured on viking15 (8,355,840 successors):

| | rate | % GPU compute peak (~2.8 Top/s) | % bandwidth (~273 GB/s) |
|---|---|---|---|
| GPU expand, 1400 ms | 5.97 M succ/s | **~0.5 %** | ~2.5 % |
| CPU expand, 491 ms | 17.0 M succ/s | ~20 % of CPU peak | ~7 % |

The bandwidth ceiling for this kernel is ≈237 M succ/s — about **40× above** what the GPU
achieves. The GPU is therefore limited by **utilisation** (§2.2, §2.3), not by any physical
limit. An earlier revision of this document claimed arithmetic intensity capped the upside;
that was wrong and is corrected here. The CPU runs at ~20 % of its peak while the GPU runs at
~0.5 % of its own — the loss is in the kernel, not in the hardware.

### 2.5 The core operation is serially dependent
`close()` is Floyd–Warshall: `for k / for i / for j`, where each `k` depends on the previous.
At dim 12 that is 12 strictly sequential rounds of only 144 cell updates. There is almost no
parallelism *inside* one zone, which forces parallelising *across* zones — which is precisely
what causes 2.2 and 2.3.

### 2.6 Unified-memory coherency (measured)
When the CPU writes zones and the GPU reads them, the GPU must snoop them out of CPU caches.
Identical merge work: **258 ms when GPU-written → 571 ms when CPU-written**. This is what
kills the CPU-expand/GPU-merge hybrid: the moment the CPU produces the zones, the GPU is the
wrong consumer.

### 2.7 Cooperation cannot pay for itself  ← why no representation change rescues this
One zone per *thread* keeps **32 independent zones in flight per SIMD group**. Every
cooperative scheme — attempt 5 (SIMD group shares one zone in threadgroup memory) and
attempt 10 (one zone distributed across the group's registers, lane *i* owns row *i*) —
processes **1 zone per SIMD group**. Cooperation must therefore make a single zone more than
**32× faster** merely to break even.

The parallelism available inside one zone is only `dim`-wide: at dim 12 that is 12 of 32 lanes
busy and 20 idle, an immediate 2.7× loss, and the ceiling is 12× before counting
`simd_shuffle` traffic and Floyd–Warshall's serial `k` loop (§2.5). **12× best case against a
32× deficit.** Both attempts lost, by different mechanisms, exactly as this predicts.

This is the deepest reason a GPU-friendly *representation* does not exist here: the workload's
parallelism is across zones, and it is already fully exploited by the simplest possible
mapping. The only lever that keeps one-zone-per-thread while shrinking the working set is a
narrower/sparser encoding (e.g. int16 bounds, ~2× — bounded, and we are ~40× short).

**One-line summary:** a GPU is a huge crew of slow workers who must all perform the identical
step in unison; this workload is millions of tiny independent jobs that each branch
differently and finish at different times. Close to a worst-case fit.

---

## 3. Everything tried, and what it measured

| # | attempt | outcome |
|---|---|---|
| 1 | Input/frontier zone interning | no gain; regressed fischer 0.58 → 1.62 s |
| 2 | GPU-side output interning (lock-free MSL hash table) | **incorrect results** — MSL atomics are relaxed-only, no release/acquire pairing available for the publish |
| 3 | SoA coalesced writes | correct, no gain (1509 vs 1450 ms) |
| 4 | MAXDIM 24 → 20 (shrink per-thread footprint) | no gain |
| 5 | SIMD-group cooperative kernel (threadgroup-shared DBM) | correct but **slower** (1837 vs 1401 ms) — the per-thread version already hides latency via millions of independent threads |
| 6 | CPU expand + GPU merge, zero-copy via unified memory | merge 258 → 571 ms (§2.6); beat pure Metal everywhere but lost to CPU merge |
| 7 | `--adaptive` per-level routing | declines the GPU on 122/124 levels |
| 8 | Zone in **threadgroup memory**, sized by real dim not MAXDIM | correct, **slower**: viking15 expand 1392 → 1599 ms, fischer10 187 → 278 ms (occupancy collapse, §2.3) |
| 9 | ...plus odd-stride **bank-conflict padding** (`d2+1`) | still slower: 1575 ms / 297 ms — conflicts were not the binding cost |
| 10 | **Row-distributed zone**: one SIMD group per zone, lane *i* owns row *i*, `simd_shuffle` for d[k][*] | correct, **worst yet**: viking15 1425 → 2536 ms, fischer10 183 → 351 ms. Cuts per-lane footprint 24× (dim vs dim²) and still loses — see §2.7 |

Attempt 5 is worth noting: the "obvious" fix for 2.3 (cooperate on one zone per SIMD group
instead of one zone per thread) made things *worse*, because it trades away the only thing
that was working — abundant independent threads.

---

## 4. What has NOT been tried

**Wavefront / stream compaction and edge-binning are ruled OUT** by the §2.2 measurement, not
merely untried: with 100 % lane survival on viking there is no divergence to recover. This was
the most promising-sounding idea; it is refuted.

What genuinely remains untried:

1. **Full GPU residency.** Keep the passed table, frontier and zones on the GPU across all
   levels so nothing round-trips, removing §2.6 and per-level dispatch overhead. Requires
   solving attempt 2 (a correct lock-free table) with `atomic_thread_fence` /
   `mem_flags::mem_device` for publish ordering, since acquire/release atomics are unavailable
   in MSL. This addresses transfer and dispatch cost — but **not** §2.3, which is the dominant
   term, so it cannot by itself make the GPU win.
2. **A fundamentally smaller zone representation** (e.g. minimal constraint graphs as UPPAAL
   uses, or narrower bound types) to shrink the per-thread working set enough to escape the
   §2.3 dilemma. This is a change to the core data structure, not to the kernel, and it would
   change CPU code too.

**Honest expected payoff: low.** The GPU does run at ~0.5 % of compute peak (§2.4), so the
*headroom* is real — but we now know what consumes it, and it is §2.3, which is a consequence
of the algorithm's shape rather than of the kernel's coding. Both horns of that dilemma have
been measured. Idea 1 does not touch it; idea 2 does, but by changing the core zone
representation for the whole tool.

The remaining honest statement is: **a GPU cannot be made to win this workload by kernel
engineering OR by changing the zone representation** (§2.7). Ten attempts, three of which were designed
specifically against the leading hypothesis of the day, and the leading hypothesis (divergence)
turned out to be measurably absent.

---

## 5. Conclusion

DBM-based zone exploration is a poor SIMT fit for reasons that are structural rather than
incidental. Reporting this as a measured negative result — with the mechanism identified — is
more useful than another speedup table, particularly when benchmarking against a multi-core
CPU tool. The practical consequence for PTCTLMC is that the fast path is CPU-only, and the
Metal backend should be presented as *attempted and slower*, with the mechanism above.

---

## 6. Would a bigger GPU change the answer?

Partly yes, and the paper should say so rather than overclaim.

**What improves with a larger GPU**
- §2.3 (the dominant cause). The on-chip route died because 32 threads x 576 B = 18 KB of a
  32 KB threadgroup budget left ~1 resident threadgroup per core. An H100's 228 KB per SM
  holds ~390 zones on-chip — the failure mode largely disappears.
- §2.4 headroom. We run at ~0.5 % of peak; the workload is latency-bound with abundant
  independent work, so more SMs means more threads in flight and more latency hiding.
  Scaling crudely by lanes and bandwidth suggests ~50–70 M successors/s versus our CPU's
  17 M — a plausible 3–4x win.

**What does NOT improve**
- §2.5, the serial Floyd–Warshall `k` loop: unchanged, it is algorithmic.
- §2.7, the cooperation arithmetic: NVIDIA warps are also 32 wide, so one-zone-per-thread
  still dominates any cooperative mapping. (Harmless — that is the mapping to use anyway.)
- §2.6 gets *worse* on a discrete GPU: PCIe transfers instead of cache snooping, unless the
  entire search is made GPU-resident (which needs a correct lock-free table, attempt 2).

**Fair-comparison caveats.** An H100 must be compared against a similarly-priced CPU (a
128-core server part, not 12 consumer cores), and the GPU-resident design is a prerequisite,
not an optimisation — otherwise transfers eat the gain.
