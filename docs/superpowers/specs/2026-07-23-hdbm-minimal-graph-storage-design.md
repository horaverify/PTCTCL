# HDBM Minimal-Graph Zone Storage — Design

**Date:** 2026-07-23
**Status:** approved (brainstorming), pending spec review
**Goal:** cut resident memory of stored clock zones ~10–12× by porting UPPAAL/UDBM's
minimal-graph zone compression into HDBM, and route the passed-sets of all three
backends (host / Metal / CUDA) through the compressed form.

## Problem

PTCTLMC interns every *distinct* reachable zone as a **dense** difference-bound matrix:
`ZoneIntern::store` is a `std::deque<Dbm>`, and each `Dbm` owns a `dim×dim` array of
`int32` raw bounds (`netreach.cpp:119`, `dbm.hpp`). For arbiter20 (20 clocks → dim 21)
that is `21×21×4 = 1764` bytes per stored zone; at ~11.5M distinct zones the passed set
alone is ~20 GB. UPPAAL solves the identical problem for ~1.5 GB because UDBM stores each
zone as a **minimal graph** (`mingraph.c`): only the non-redundant constraints, with a
bit-mask/couple index and 16-bit bounds when they fit (`mingraph_coding.h`), ≈ 144 bytes
for the same zone.

The operations are not the problem — a dense matrix is required to run Floyd–Warshall
(`dbm_close`), `constrain`, `delay`, subsumption, etc., and UDBM keeps a dense `raw_t*` for
exactly that. The waste is purely in the **stored / transferred** bytes. Therefore the fix
is a *storage codec*, not a new working representation: operate dense, store minimal.

## Non-goals

- **Operating directly on the packed form.** It yields *zero* extra memory saving (the
  stored bytes are identical whether or not a read unpacks first) at large correctness
  risk. Every consumer unpacks to a dense scratch DBM, runs the existing op unchanged,
  and — for producers — repacks. This mirrors UDBM exactly.
- Changing the zone *algebra* in `dbm_ops.hpp`. Those functions stay byte-for-byte the
  operating representation.

## Architecture

### Component: `mindbm` (new, in HDBM)

New `include/ptctlmc/mindbm.hpp` + `src/mindbm.cpp`. Every function is annotated
`PTCTLMC_HD` (the `__host__ __device__` macro already defined in `dbm_ops.hpp`) so the
CUDA `.cu` and Metal `.mm`/`.metal` paths compile the *same* codec that the CPU unit tests
validate. Public surface:

| function | contract |
|---|---|
| `int mindbm_analyze(const raw_t* d, int dim, uint32_t* bitMatrix)` | Larsen–Larsson–Pettersson–Yi reduction. `d` must be closed & non-empty. Sets one bit per surviving constraint in `bitMatrix` (`bits2intsize(dim*dim)` words), returns the count. Line-for-line port of `mingraph_analyzeForMinDBM`. |
| `size_t mindbm_bound(int dim)` | worst-case packed byte size for a `dim` zone; lets callers size a scratch buffer / device slab without packing first. |
| `size_t mindbm_pack(const raw_t* d, int dim, void* out)` | writes the packed record to `out`, returns bytes written. Chooses format dynamically (below). `d` closed & non-empty. |
| `int mindbm_unpack(const void* in, raw_t* d)` | rebuilds the **closed dense** DBM into `d` (fills redundant edges via `dbm_close`), returns dim. Inverse of `pack` up to canonicalisation. |
| `size_t mindbm_size(const void* in)` | byte length of a packed record, for walking a flat arena. |
| `uint32_t mindbm_hash(const void* in)` | hash over the packed bytes, for the intern dedup map. |

### Packed record format

Identical in spirit to `mingraph_coding.h`, adapted to HDBM's raw encoding
(`raw = (value<<1) | weak`, `weak=1` ⇒ `≤`, `kInf = 1<<30`). Header is `uint32_t[1]`
(plus an optional second word for large constraint counts):

```
word[0]:
  0x0000ffff  dim
  0x00010000  bounds encoded on 16 bits (else 32)
  0x00020000  index = (i,j) couples (else bit-matrix)
  0x00040000  minimal reduction used (else dense copy without diagonal)
  0x00180000  couple index width: 0→4b, 1→8b, 2→16b per index
  0x00200000  "many constraints": count is in word[1] (else in the high bits)
  0xffc00000  constraint count when not "many" (max 0x3ff)
```

Payload = the surviving bounds (int16 when *all* fit `dbm_INF16 = SHRT_MAX>>1`, else
int32) followed by their indices (bit-matrix words, or packed 4/8/16-bit couples).
`pack` computes the byte cost of each valid `(16?, couples-vs-matrix, index-width)`
combination and emits the smallest — UDBM's dynamic choice. Infinity maps to a 16-bit
sentinel (`dbm_LS_INF16`) exactly as `mingraph_raw16to32` / `mingraph_raw32to16` do.

### Stage 1 — host `ZoneIntern` stores packed bytes  *(≈90% of the RSS win)*

`netreach.cpp`:
- Replace `std::deque<Dbm> store` with a **flat byte arena** of packed records
  (chunked `std::deque<std::vector<std::byte>>` or a bump arena with stable chunk
  addresses — keeps the "stable address per interned zone" invariant the readers rely on).
- `intern(const Dbm& z, tid)`: `mindbm_analyze`→`mindbm_pack` into a thread-local scratch,
  dedup on the packed bytes (`ZoneKey` becomes {ptr,len}; hash via `mindbm_hash`), append
  on miss, return the index.
- Read consumers — `alu_subsumed_by`, `try_union`, `includes` (`netreach.cpp:205–231`) —
  `mindbm_unpack` the stored record into a **per-thread dense scratch `Dbm`**, then call the
  existing op unchanged. The stored record stays immutable, so the lock-free reader
  invariant holds.

### Stage 2 — Metal passed-set packed

`src/metal/…` and `include/ptctlmc/metal_reach.hpp`: the device zone buffer holds packed
records plus a parallel **offset buffer** (prefix sum of record sizes, since records are
variable length). Successor kernel: unpack neighbour from the buffer into threadgroup/
thread dense scratch → op → `mindbm_analyze`+`mindbm_pack` → write. The codec compiles into
the Metal translation unit via the shared header. Frontier construction becomes two-pass
(compute sizes → prefix-sum → write) to place variable-length records.

### Stage 3 — CUDA passed-set packed

Same shape as Stage 2 for `src/cuda/…` / `cuda_reach.hpp` / `net_gpu*`: packed device slab +
offset array, unpack-op-repack in the kernel, two-pass frontier write. Ordered last because
it is the largest kernel restructure.

## Data flow (per successor, all backends)

```
stored packed record ──mindbm_unpack──► dense scratch DBM ──ops(close/constrain/…)──►
    dense result ──mindbm_analyze+mindbm_pack──► new packed record ──► passed set
```

## Error handling & edge cases

- `dim ≤ 2`: UDBM special-cases these (nothing to reduce); port the same shortcuts.
- Bounds that overflow int16: format selection falls back to 32-bit automatically; never a
  correctness issue, only size.
- Empty zone: never stored (callers already drop empties before intern); `pack` asserts
  non-empty & closed, matching UDBM preconditions.
- Arena stable addresses: interned records must never move (readers hold pointers across
  threads). Chunked arena with `delete[]`-on-destroy chunks, like the existing `ZoneIntern`.

## Testing / validation

1. **Round-trip (property):** for random closed zones and for the *actual* reachability
   zones of the benchmark models, `mindbm_unpack(mindbm_pack(d)) == dbm_close(d)` exactly.
2. **Differential (integration):** every benchmark model reports the **identical reachable
   state count and verdict** with the codec ON vs OFF — a build flag keeps the dense path
   for A/B comparison during bring-up.
3. **Size check:** packed bytes/zone compared against UDBM `dbm_writeToMinDBM` on identical
   zones, to confirm we reach the same footprint (target ≈ 10–12× vs dense).
4. **Per-stage:** Stage 1 validated on CPU builds before Metal/CUDA; each device stage
   re-runs the differential suite on that backend.

## Rollout / risk

- HDBM lives as an **in-tree checkout** (`PTCTLMC/HDBM/`, already the CMake-preferred path)
  during development so the local codec is used immediately; pushed to
  `github.com/horaverify/HDBM` when Stage 1 is green.
- A compile-time `PTCTLMC_ZONE_PACKED` flag (default off until validated) lets each stage
  land behind a switch and be A/B'd against the dense path.
- Biggest risk is the device two-pass frontier restructure (Stages 2–3); Stage 1 delivers
  the dominant CPU win independently, so value ships even if device stages slip.
