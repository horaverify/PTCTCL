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
this pass; the `metal` column is the current tree.

| model | `verifyta` | `omp` | `metal` before | `metal` now |
|---|---:|---:|---:|---:|
| fischer12 | 43.02 / 0.55 | 19.08 / 0.85 | 8.80 / 11.1 | **6.40** / **1.88** |
| twolock12 | 47.64 / 0.55 | 19.00 / 0.88 | 8.53 / 10.6 | **6.48** / **1.92** |
| arbiter17 | 11.81 / 0.21 | 6.37 / 0.34 | 3.85 / 7.4 | **2.06** / **1.34** |
| csma14 | 8.86 / 0.14 | 3.61 / 0.22 | 2.03 / 4.0 | **1.04** / **0.88** |
| phil12 | 9.94 / 0.18 | 1.87 / 0.13 | 2.63 / 2.8 | **1.25** / **0.65** |

`metal` beats *both* `verifyta` and the 12-core `omp` backend on every model, exploring
**exactly the same state space** — integer-for-integer against `verifyta` on fischer,
twolock and arbiter, and fewer on csma and philosophers only because the exact
convex-union merge is more aggressive than UPPAAL's representation.

Nothing here changes what is explored. A comparison that reduced the state space (process
symmetry, say) would not be like-for-like against `verifyta` and is deliberately not part
of this: see "Not used: symmetry reduction".

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

**The CPU backends get it too**, since `dbm_try_union` is shared. They always *had* the exact
convex union — it is why PTCTLMC explores fewer states than UPPAAL on csma and philosophers —
but not an affordable one. A/B of `omp` against the previous O(dim⁵) implementation, identical
state counts:

| model | O(dim⁵) | O(dim³) | |
|---|---:|---:|---|
| phil13 | 13.42 | **9.88** | −26% |
| phil12 | 2.41 | **1.87** | −22% |
| csma14 | 3.73 | **3.61** | −3% |
| fischer12 | 19.76 | **19.08** | −3% |

The gap tracks how often zones actually union: constantly on philosophers, almost never on
Fischer.

### 1.1a Do we explore fewer states than UPPAAL?

On two of the five families, and it is this merge that does it. `explored`, symmetry off, so
like-for-like:

| model | PTCTLMC | `verifyta` | |
|---|---:|---:|---|
| fischer12 | 4 758 404 | 4 758 404 | identical |
| twolock12 | 4 807 556 | 4 807 556 | identical |
| arbiter17 | 1 245 184 | 1 245 184 | identical |
| csma14 | **618 496** | 991 232 | −38% |
| phil12 | **908 838** | 1 140 361 | −20% |

Sound: the union replaces two zones by a single zone only when that zone *is* their union, so no
valuation is added. On Fischer, two-lock and arbiter the reachable zones simply never union, and
the counts match integer for integer.

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

### 3.5 The frontier was storing the passed set's zones a second time

The frontier held a dense zone per state for the whole level — but every one of those zones had
just been interned into the passed arena. It holds the arena **index** now (4 B per state), and a
batch's zones are widened straight into the GPU's frontier buffer. 0.24 GB on fischer12, 0.81 GB
on fischer16, and it removes a pass: the level-end readout used to widen every zone into a host
array that `expand()` then copied into the device buffer.

Slots are consequently recycled **one level late**: a frontier entry now names a slot rather than
owning a copy, and the merges that drop slots run during exactly the level consuming that
frontier, so `release()` defers to a pending list flushed at the level boundary.

### 3.6 Narrow zone storage — arena, frontier, expand output

A DBM entry is `(value << 1) | weak`, and in a zone graph over a model whose largest clock
constant is *M* every finite entry lies in `[-2M-1, 2M+1]` — an entry is a shortest path, and any
path to or from the reference clock is bounded by two direct border edges. For any real
timed-automata model that fits in **16 bits**.

The passed arena, the frontier buffer, the expand output pool **and the merge work pool** all
narrowed. It costs nothing:
every one of them is only ever read or written a whole matrix at a time through a per-entry loop
into a private 32-bit working matrix, so the conversion rides along with a copy that already
existed.

The merge work pool needed one more step, because the CPU merge indexes it directly rather than
copying out of it. `dbm_includes`, `dbm_is_alu_le`, `dbm_try_union`, `dbm_hull`, `dbm_axis_gap`
and `mindbm_analyze` are now templated on the entry type of their **read-only operands**, reading
through `zget()`. The algebra itself is untouched — every working matrix, every mutated operand
and every intermediate is still `raw_t` — and a `raw_t` operand deduces to the identity, so not
one call site changed. It is the same shape the MSL kernels already used for their
address-space-generic ops.

The randomised differential test covers the narrow instantiation: 200 k zone pairs with the
second operand stored 16-bit, checking `dbm_try_union`'s verdict *and* hull, and `dbm_includes`
and `dbm_is_alu_le` in both directions, against the 32-bit call.

Narrowing is also *faster*, because per-successor store traffic halves: fischer12 6.93 → 6.40 s
and 2.32 → 1.88 GB.

### 3.7 The batch budget is the knob

The expand and merge pools scale linearly with the per-batch byte budget, which makes it the
primary memory/time control. `--mem <GB>` sets it; the default is a sixteenth of RAM capped at
1 GB. Measured on fischer12 (peak RSS / wall):

| `--mem` | 2 GB | 1 GB | 0.5 GB | 0.25 GB |
|---|---:|---:|---:|---:|
| peak RSS | 3.14 | 2.34 | ~2.1 | ~2.0 |
| wall | 6.90 | 7.17 | 8.34 | 9.60 |

1 GB is ~1% of time for 26% of memory. Below that the extra batches start to cost, and RSS hits a
floor set by the passed arena rather than the pools.

---

## Part 4 — not used: symmetry reduction

Many of these networks are N copies of one template, so the reachable space is covered by
exploring one representative per automorphism orbit. `--symmetry` implements that and it is
enormous — 28x fewer states on fischer12, 65x on arbiter17 — but it explores a *different*
state space from `verifyta`, so any speedup measured with it on is not a like-for-like
comparison. It stays **off**, it is **not** part of any number in this document, and the
GPU path does not implement it (`--symmetry` still routes to the CPU, as before).

Recorded only so the option is not rediscovered as new: the reduction is sound, and a
device-side canonicaliser is cheap (|G| integer comparisons over data the thread already
holds; ties resolved through inverse clock permutations so the zone need never be
materialised twice). If the evaluation ever wants it, it has to be enabled for *both* sides
of the comparison, which means giving UPPAAL a scalar-set model.

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

- **Memory: what is left, and what each would cost.** fischer12's 2.14 GB is now roughly
  0.90 GB passed arena + 0.68 GB GPU pools + 0.16 GB bucket overhead + slack, against the
  OpenMP backend's 0.88 GB total. Three things could still move it:
    - *Minimal-graph storage for the arena* (what the CPU backend does): 0.90 → ~0.2 GB. But
      unpacking is an O(dim³) close, and the gather plus the frontier materialisation do ~10 M
      loads on a fischer12 run — several seconds on a 6.4 s run. **Measured as a bad trade at
      these dimensions, and it gets worse as dim grows.**
    - ~~*Narrowing the merge work pool*~~ — **done**, see §3.6.
    - *Zone dedup* (`--gpu-intern`) helps only philosophers (arena 0.20 → 0.01 GB, RSS 0.93 →
      0.75) and *hurts* fischer/arbiter/csma, where nearly every zone is distinct and dedup mode
      also disables slot recycling. It stays a flag.
- **The merge work pool could be read by index.** The gather (761 ms on fischer12) copies each
  bucket's existing antichain out of the interner into the merge pool. If the arena were itself
  a Metal buffer, the merge could read it by index and the gather would disappear.
- **CUDA parity.** The CUDA engine still lacks device-side key packing; it is a short port
  of the Metal kernel's tail.
