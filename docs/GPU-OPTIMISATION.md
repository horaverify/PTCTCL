# Making the GPU path fast: what worked, and why

This is the working record of an optimisation pass over the Metal backend and the flat
GPU driver. It supersedes the pessimistic conclusion of [`GPU-FINDINGS.md`](GPU-FINDINGS.md)
for the *network* engine — that document's measurements were of the single-automaton
engine and of an earlier driver, and its central claim ("a GPU cannot be made to win this
workload by kernel engineering") turned out to be answerable, though not in the way its
own list of untried ideas predicted.

Hardware for every number: **Apple M4 Pro**, 8P+4E cores, 16-core GPU, 24 GB unified
memory. `verifyta` is UPPAAL 5.0.0, single-threaded. `omp` is the 12-thread CPU backend.

## The headline

Wall-clock seconds, min of 3, and peak RSS in GB. `metal-before` is the starting point of
this pass; every other column is the current tree.

| model | `verifyta` | `omp` | `metal` before | `metal` | `omp --sym` | `metal --sym` |
|---|---:|---:|---:|---:|---:|---:|
| fischer12 | 46.76 | 20.23 | 8.80 | **6.53** / 4.6 | 0.70 | **0.33** / 0.36 |
| twolock12 | 50.62 | 20.03 | 8.53 | **6.67** / 4.9 | 0.68 | **0.33** / 0.36 |
| arbiter17 | 12.13 | 6.65 | 3.85 | **2.40** / 3.5 | 0.12 | 0.16 / 0.15 |
| csma14 | 9.30 | 3.81 | 2.03 | **1.30** / 1.8 | 0.11 | 0.14 / 0.08 |
| phil12 | 10.34 | 1.98 | 2.63 | **1.46** / 1.6 | 0.22 | **0.20** / 0.15 |

Without symmetry, `metal` now beats *both* `verifyta` and the 12-core `omp` backend on
every model, and peak RSS on fischer12 went 11.1 GB → 4.6 GB. With symmetry the standard
sizes become trivial (13 k–170 k states), so GPU dispatch overhead shows on the two
smallest; at sizes where there is work to do the GPU wins there too:

| model | `omp --sym` | `metal --sym` |
|---|---:|---:|
| arbiter24 | 18.97 | **9.38** |
| csma20 | 8.46 | **4.39** |
| fischer16 | 79.03 | **28.32** |
| phil14 | 3.91 | **3.05** |
| phil15 | **27.65** | 57.54 |

`verifyta` does not finish fischer16 inside 400 s. **phil15 is the one place the GPU
still loses** — see "What is still open".

Two independent things are happening. The `metal` column is engineering: the same state
space explored 1.5–1.7× faster and in a third of the memory. The symmetry columns are an
*algorithmic* reduction — explore one representative per automorphism orbit — which now
composes with the GPU instead of disabling it.

---

## Part 1 — the algebra

### 1.1 The convex-union test was O(dim⁵)

The exact convex-union merge (`dbm_try_union`) is what collapses a discrete state's zone
antichain to a single zone and is the reason PTCTLMC's stored counts match UPPAAL's. It was
also expensive enough that the GPU merge kernel shipped with it **off** — which cost node
count: csma14 stored 984 043 zones without it against 618 496 with.

With `H` the convex hull, `a ∪ b = H` iff `H \ a ⊆ b` and `H \ b ⊆ a`. Writing `z` as the
intersection of its constraints, `H \ z = ⋃_(i,j) (H ∧ ¬z_ij)`, so it suffices to check each
*crossing region* against the other zone. The old implementation formed a region for every one
of the O(dim²) entries and re-canonicalised each with a full O(dim³) Floyd–Warshall.

Two observations collapse that:

1. **Only the minimal graph's constraints need a region.** The minimal graph defines the same
   zone, so `¬z` is exactly the union over *its* edges and the redundant entries contribute
   nothing. That is O(dim) facets instead of O(dim²). `mindbm_analyze` already computed it.
2. **A crossing region is the hull with one edge tightened, and the hull is already canonical.**
   For canonical `a`, `b`: `h_ij = max(a_ij, b_ij) ≤ max(a_ik+a_kj, b_ik+b_kj) ≤ h_ik + h_kj`.
   So re-canonicalising a region is `dbm_constrain`'s O(dim²) incremental close, and the final
   close of the hull is not needed at all.

Together: **O(dim³) where it was O(dim⁵)** — at dim 13, ~2.2·10³ bound operations per pair
instead of ~3.7·10⁵.

Verified against the previous implementation on 400 000 random zone pairs: identical verdicts
*and* identical hulls. 185 611 of the convex verdicts were additionally checked pointwise on a
half-integer lattice — the direction that would be unsound if the rewrite were wrong.

Consequence: convex-union merging is now **on by default** on the GPU. csma14's stored count
drops to exactly the OpenMP backend's, and fischer12's peak RSS by 24%, at no time cost.

### 1.2 The reset-inactive scan

The active-clock reduction resets any clock no reachable guard will read. The kernel decided
that per clock by scanning every process's `[location][clock]` entry: `nprocs × nclocks` device
loads per successor — 289 on arbiter17, on par with the entire rest of the zone pipeline. One
`uint32` bitmask per location, OR'd over the processes, replaces it with `nprocs` loads.

### 1.3 `dbm_constrain`'s no-op case

Most guards and invariants are already implied by the incoming zone. That case returned
`!dbm_empty(...)`, an O(dim) diagonal scan, for a matrix it had not touched. Every caller in
the successor pipeline holds a non-empty zone (the frontier zone is non-empty, and every step
that could empty it returns false and aborts), so it can return `true`.

---

## Part 2 — the host, which was the real bottleneck

The profiler (`PTCTLMC_MTLPROF=1`, plus a new `[engprof]` line that splits `expand()` into
host enumeration / staging / kernel) said the GPU backend was spending **44–60% of its wall
clock on the host**, much of it single-threaded. A backend that runs at 69% CPU on a
12-core machine is not GPU-bound.

### 2.1 Recipe enumeration was serial

`expand()` enumerated successor "recipes" (participant lists) in one serial loop that
`push_back`ed into six host vectors and uploaded them into six freshly-allocated `MTLBuffer`s
per level. 286 ms of a 1756 ms fischer11 run; 719 ms of arbiter17's 3537 ms.

It is now a **counting pass and a writing pass over one shared enumeration body**, both
OpenMP-parallel over frontier states. The prefix sum between them gives each state a disjoint,
index-ordered slot range, so the recipe order is bit-identical to the serial version — which
matters, because the convex-union merge is order-sensitive. The writing pass targets unified
memory directly: no host vectors, no upload, no per-level allocation.

> 286→70 ms, 719→235 ms, 386→97 ms.

The double enumeration costs ~1.6× the scalar work and spreads it over 12 cores. Sharing one
body between the two passes is not a tidiness point: if they disagreed about how many recipes
a state has, the second pass would write outside its slot range.

### 2.2 The bucket index allocated per key

Grouping successors by discrete state used a `std::unordered_map` rebuilt every level — one
malloc per distinct key, 227 k of them per level on fischer11, and a pointer chase per probe.
An open-addressing table with a generation stamp reuses one allocation for the whole run and
clears in O(1).

### 2.3 The discrete key is now packed on the GPU

The bucketing pass is the level's serial section. It read back each successor's `nprocs`
locations plus `nintvars` integers — 48 bytes on fischer12 — purely to recompute an 8-byte
key. The expand kernel now packs the key itself from the same `dshift`/`intlo` layout the
driver uses, so the host reads 8 bytes per successor and **the per-successor location array
is not written by the kernel at all**: 41.6 M × 13 words of stores saved over a fischer12 run.

The target test comes off the key too: the required locations are a fixed bit pattern in it,
so the common non-target case is two integer operations rather than a walk over the locations.

### 2.4 Zone identity is carried through the merge

The passed-set rebuild re-interned a bucket's *whole* antichain whenever any one zone in it
changed — mean 5.3 zones on fischer12, of which usually one is new. The merge's per-slot flag
became a **tag**: the driver seeds each existing zone's slot with its passed-set index, the
merge carries it through its compaction (it already moved the flag), and writes `-1` for the
zones it creates. A kept zone then needs no copy out of the work pool and no re-intern.

---

## Part 3 — memory

The GPU path peaked at **13 GB on fischer12 where the OpenMP backend used 0.86**. Four
separate causes, none of which was the zone data.

### 3.1 Metal's dropped pool generations (≈5 GB)

Releasing a grown-past buffer is not enough — the driver reclaims its pages lazily. Measured:
our live pools summed to 6.35 GB while Metal had **11.26 GB mapped**. `setPurgeableState:Empty`
on the outgoing buffer declares its contents dead so the pages come back immediately. (Safe
here: `grow()` does not preserve contents; `grow_keep()` copies first.)

### 3.2 The passed arena grew with cumulative interns, not live zones (≈10 GB)

`GpuZoneIntern`'s append mode allocated a fresh slot per `intern()` call, and §2.4's rebuild
called it for every zone of every touched bucket. Measured ~2000 chunks × 5.4 MB = 10.9 GB of
arena holding 2.7 M live zones. Slots now return to a striped free list.

Finding this took a detour: the 5408 KB block size is exactly `8192 × dim² × 4` bytes, which
also happens to be the size of a plausible driver-side per-thread private-memory arena, and it
scaled with `dim²` in the right way. It was the interner's chunk size.

### 3.3 A BFS level is now batched against a byte budget

The expand and merge pools are sized by the **widest** BFS level (`nwork × dim²` words, and
`nwork` is ~12× the frontier). `--chunking` existed to bound them, but it appended the
un-expanded tail of a level to that level's survivors and swapped — which mixes level *k* and
level *k+1* into one frontier. Strict BFS is lost and fischer12's explored count went 4.76 M →
12.0 M, so the flag was never usable.

The fix is a scheduling one. The level's remaining states and the next level are separate
lists, and the next frontier is read at **level end** off a per-zone "added during this level"
flag packed into the spare bit of the stored zone index. A zone that a later batch absorbs into
a convex union therefore never reaches the frontier at all. Batching is now *exact*: identical
explored counts at every batch size, verified down to 64-state batches.

Batch size comes from a byte budget (a sixth of RAM, capped at 4 GB) and the measured fan-out,
so a level that already fits is still expanded whole and small models pay nothing.

### 3.4 Per-batch scratch is hoisted

With batching a level is many batches, and the per-batch vectors (some hundreds of MB) were
allocated and freed each time. Reused buffers keep their capacity.

---

## Part 4 — symmetry reduction, and why it belongs in the kernel

Many of these networks are N copies of one template: permuting the copies maps the network onto
itself, so the reachable space is covered by exploring **one representative per orbit**. That is
a reduction no amount of parallelism can give you, and it multiplies with parallelism rather
than competing with it.

`--symmetry` already existed but **forced the CPU path** (`!use_sym && backend == Metal`), so
the two largest wins in the tool could not be had at once. Two changes fixed that.

### 4.1 The canonicaliser is now a kernel

Canonicalisation is |G| integer comparisons over data the thread already holds in registers —
exactly what a GPU lane is for. The kernel reproduces the host `canonicalize` exactly: the
lexicographically smallest (locations, integer values), ties broken by the smallest zone matrix.

The zone is never materialised twice. A tie is resolved by comparing the two candidate matrices
**through their inverse clock permutations** (a permuted entry `(a,b)` is `z[inv[a]][inv[b]]`),
and only the winner is written — permuted directly into the output buffer. So canonicalisation
costs this kernel *no extra private memory*, which matters because the private zone array is
what it is bound on.

It makes two passes over the group rather than one pass plus a tie list. A fixed-size tie list
would cap the group the kernel can accept, and truncating it would be **unsound**: two states of
one orbit could then choose different representatives and both be explored.

### 4.2 The symmetry search only ever seeded from process 0

`symmetry_group()` built its generators from images of process 0, so it found nothing whenever
process 0 is the one asymmetric component — exactly the shape of **arbiter** (an `Arb` plus N
interchangeable clients) and **CSMA/CD** (a `Bus` plus N stations). Two of the five benchmark
families silently got the trivial group. Seeding from each process in turn costs np² propagations
over an np-process network and finds them.

Any closed subgroup of the automorphism group gives a sound reduction, so a partial generating
set is safe; the closure is still verified and still complete.

### 4.3 What it buys

| model | explored, plain | explored, `--symmetry` | ratio |
|---|---:|---:|---:|
| fischer12 | 4 758 404 | 169 618 | 28× |
| twolock12 | 4 807 556 | 171 762 | 28× |
| arbiter17 | 1 245 184 | 19 056 | 65× |
| csma14 | 618 496 | 13 238 | 47× |
| phil12 | 908 838 | 70 805 | 13× |

At fischer16, `metal --symmetry` finishes in 32 s where the 12-core CPU takes 79 s and
`verifyta` does not finish at all.

**Symmetry stays opt-in.** It deliberately explores a different state space, so the
integer-for-integer parity against `verifyta` that the evaluation rests on is only meaningful
with it off. Report both.

---

## Correctness

Everything above is checked three ways:

- `ctest` (unit + selftest + bisim) after every change.
- **80/80** curated model × backend combinations agree with `verifyta` — `seq`, `omp`,
  `metal --adaptive`, `metal --adaptive --chunking 64`, and all four again with `--symmetry`.
- The `dbm_try_union` rewrite additionally has a randomised differential test against the
  previous implementation (400 k zone pairs, verdicts *and* hulls) plus a pointwise lattice
  check of the convex verdicts.

Explored/stored counts are reported at every step and were required to be unchanged by every
change that was not deliberately a node-count reduction.

---

## Part 5 — two bugs the measurements exposed

Neither is a tuning question; both made the GPU look worse than it is.

**The CPU fallback ran single-threaded.** `parallel` was `backend == Backend::Omp`, decided
before the GPU factory ran, so a GPU backend that fell back to the CPU path — out-of-scope
model, no device — used one core. arbiter24 took **93 s** that way against `omp`'s 20 s. It
reads as "the GPU lost"; it was "the fallback used one core". With the decision moved after
the factory (and the next item fixed) arbiter24 is **9.4 s**.

**The GPU refused any network with more than 23 clocks.** That bound dates from when every
private array in the shader was sized to the 24-clock worst case. Since the shader is now
compiled per network with `-DPT_DIM`, it is only an eligibility test — and it was silently
dropping arbiter24 and up onto the (then single-threaded) CPU path.

---

## What is still open

- **Philosophers at N ≥ 15: one bucket is the whole critical path.** At phil15 a level of
  24 910 discrete states contains one holding an antichain of **5361 zones**. The merge is
  parallel *across* buckets, so that bucket's O(k²) subsumption runs on a single thread while
  eleven cores idle: 175 s of CPU time inside a 55 s wall, 3.2× of a possible 12. The merge is
  92% of the run.

  Two things were tried and are recorded as negative results. Routing it to the GPU's
  SIMD-group-per-bucket merge (`--naive`) is *worse* — it did not finish in 9 minutes. And a
  CPU merge that parallelises *within* a big bucket regressed badly at every granularity tried:
  a parallel region per new zone spent 595 s of **system** time inside a 273 s wall on team
  setup and barriers, and hoisting to one team per bucket (`omp single` for the serial
  compaction and convex-union steps, `omp for` for the two scans) did not converge either. The
  scans are only O(dim²) per element, so they are too fine-grained for a barrier per new zone.

  The shape that should work is a *persistent* team with its own lightweight synchronisation,
  or restructuring the merge so a whole batch of new zones is tested against the antichain at
  once — one barrier per batch instead of per zone. Not attempted.

- **16-bit bounds.** Every zone is `dim²` int32. At fischer16 the passed set alone is
  4.95 M zones × 289 entries × 4 B = 5.7 GB, against the CPU backend's ~0.63 GB — the CPU
  stores *minimal graphs* (`mindbm`), the GPU needs dense matrices for its kernels. 16-bit
  packed bounds halve all of it and double effective bandwidth on a latency-bound kernel.
  Gate on a static bound so a model whose constants could overflow keeps int32.
- **The merge work pool could be read by index.** The gather (761 ms on fischer12) copies each
  bucket's existing antichain out of the interner into the merge pool. If the arena were itself
  a Metal buffer, the merge could read it by index and the gather would disappear.
- **CUDA parity.** The CUDA engine still lacks device-side key packing and the canonicaliser;
  both are short ports of the Metal kernel's tail.
