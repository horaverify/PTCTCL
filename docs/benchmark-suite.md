# PTCTLMC Benchmark Suite & Evaluation Plan

A TACAS/FORMATS-grade benchmark suite for **PTCTLMC**, a from-scratch parallel
model checker for **networks of timed automata** (the UPPAAL model class), with
`seq` / `omp` / `metal` backends validated differentially against UPPAAL
`verifyta` 5.0. This document is a *plan* — a survey of the standard TA
benchmarks, a characteristics matrix, per-benchmark generator specifications, and
an evaluation protocol. No code is changed by adopting it.

> **Design north star.** The suite must (a) span the model *characteristics* that
> matter for a DBM/zone checker and a parallel one, and (b) be structured to
> demonstrate *where a parallel checker beats single-threaded UPPAAL*. The single
> most important fairness control is that **PTCTLMC and `verifyta` explore the same
> number of symbolic states** — only then is a wall-clock speedup meaningful.

---

## 0. What PTCTLMC actually supports (authoritative scope)

Everything below is read off the source, not the README prose. A benchmark is
useful **only if both `verifyta` and PTCTLMC accept it and agree on the verdict**,
so every proposed model is cross-checked against these limits (§5, §6).

### 0.1 Model/parser scope (`src/uppaal.cpp`, `src/netreach.cpp`)

| Feature | Supported? | Exact rule (from the parser) |
|---|---|---|
| XML dialect | ✅ | UPPAAL flat-1.6 `<nta>`; **one `<template>` = one process instance**, in document order. The `<system>` line is *ignored* — no `Proc(i)` instantiation, no parameters. Generators must emit flattened templates. |
| Clocks | ✅ | `clock a, b;` in global or per-template `<declaration>`. |
| Bounded ints | ✅ | `int[lo,hi] x, y=init;`. Bare `int x;` defaults to `[-32768,32767]`. |
| Binary channels | ✅ | `chan a, b;` — engine pairs exactly one `a!` with one `a?` (enumerates if several receivers are enabled). |
| Broadcast channels | ✅ | `broadcast chan c;` — one `c!` fires with **0 or more** enabled `c?`. |
| Committed / urgent | ✅ | `<committed/>` / `<urgent/>` child of a `<location>`. |
| Location invariants | ✅ clock-only | Invariant labels are parsed as clock constraints; **integer atoms in an invariant are silently discarded**. |
| Clock guards | ✅ | Conjunctions (`&&`, `and`, `,`) of `clock OP const`. |
| Diagonal guards | ⚠️ CPU-only | `x - y OP const` is parsed, but see §0.3 — it disables the GPU path and LU widening. |
| Integer guards | ✅ | `intvar OP const`, `OP ∈ {<=,>=,==,!=,<,>}`. |
| Clock resets | ✅ **to 0 only** | `x = 0` / `x := 0`. **Any non-zero clock assignment is silently dropped.** |
| Integer updates | ✅ **constant only** | `x := const`. **`x = x + 1` / increments / expressions are NOT parsed** (rhs must be a literal). |
| Guards comparing two variables | ❌ | No `i == j`, no `x == i` (clock vs int), no `x >= y + 2`. One side must be a literal constant. |
| Channel arrays, `select`, functions, arrays, `scalarset` | ❌ | Not tokenised at all. |
| Queries | `E<>`, `A[]`, `E[]`, `A<>`, `p --> q` | `deadlock` keyword is *unsupported*. |

**The two limits that drive model design** are: *clock resets only to 0*, and
*integer updates only to a constant*. Any protocol whose semantics need a counter
(`n++`) must encode the count in **location structure** instead (see GRC, §5.6) or
in a **process-id-valued shared variable** written with constants (see Fischer,
§5.1). Both patterns are demonstrated by the existing generators and are the key
to keeping "counter" protocols in scope.

### 0.2 Which queries run in parallel (the speedup story)

| Query | Engine entry point | Backends | Notes |
|---|---|---|---|
| `E<> φ` | `reachable_net` | **seq / omp / metal** | Level-synchronised frontier BFS; parallel successor pipeline + sharded subsumption merge; GPU-resident driver on `metal`. **This is the fast path.** |
| `A[] φ` (safety) | `reachable_net` via `¬E<>¬φ` | **seq / omp / metal** | Safety reduces to reachability of a "bad" state → same parallel path. Explores the *whole* reachable set (no early exit) → the best, most reproducible speedup showcase. |
| `E[] φ`, `A<> φ`, `p --> q` | `exists_invariant_run` / `leads_to_net` | **seq only** | Liveness = cycle/SCC search over the whole extrapolated graph; not parallelised. Use for **coverage/correctness**, *not* for the speedup numbers. |

Headline speedup tables must therefore be built from **`A[]`-safety** and
**`E<>`** queries. Liveness queries are reported for correctness coverage only.

### 0.3 Diagonal constraints and the GPU fallback

`network_has_diagonal(net)` scans every guard/invariant for a constraint with two
positive clock indices (`x - y OP c`). If **any** exists:

- `make_net_gpu_metal` returns `null` → the run **transparently falls back to the
  CPU** (`omp`/`seq`), even with `--backend metal`.
- Extrapolation is switched off (exact zones, `ExNoDiag`) and subsumption reverts
  to plain zone inclusion → the passed list is larger and there is a **500 000
  symbolic-state safety cap** (throws beyond it).

Consequence: diagonal models are **CPU-only and can blow up**. Keep them small,
and use them deliberately to exercise the diagonal axis (§5.9), never as GPU
speedup benchmarks.

### 0.4 Discrete-state packing cap (network width)

`reachable_net` packs the discrete state into a **single `uint64`**: `bits(#locs_p)`
per process + `bits(hi−lo+1)` per int var. The sum must be **< 64 bits** or the
run throws `"network too wide"`. This caps the scaling parameter of every network
benchmark; each spec below states its cap. (Example: Fischer needs `2N` location
bits + `bits(N+1)` id bits, so `N ≲ 28`.)

### 0.5 CLI and statistics (what the harness reads)

```
ptctlmc [--backend seq|omp|metal] model.xml query.q
```

Per query it prints `-- Formula is satisfied./NOT satisfied. [explored N symbolic
states]` (this `N` = `ReachStats.explored` = symbolic states added to the passed
list). Setting `PTCTLMC_STATS=1` additionally prints `discrete-states=… stored-
zones=…` to stderr, and `[gpu] network expander ACTIVE` when the Metal path is
really running (crucial for detecting silent CPU fallback). `PTCTLMC_NOACTIVE=1`
disables active-clock reduction (useful to explain state-count differences).

---

## 1. Survey of the standard TA benchmarks

The canonical scalable timed-automata benchmarks recur across the tool papers of
UPPAAL, TChecker, Theta, and the abstraction literature. The most useful curated
sources:

- **Fontana & Cleaveland, "Timed Automata Benchmark Description," arXiv:2005.13151
  (2020).** Documents CSMA/CD, Fischer, GRC (railroad crossing), Leader election,
  plus FDDI and PATHOS "often used to benchmark timed automata model checker
  speed," each with automata figures, parameters, and safety/liveness formulas.
  <https://arxiv.org/abs/2005.13151>
- **Herbreteau, Srivathsan, Walukiewicz, "Better abstractions for timed automata,"
  LICS 2012 (arXiv:1110.3705; journal version *Information & Computation*).**
  Introduces the **aLU abstraction** that PTCTLMC implements, and evaluates on
  **Fischer, CSMA/CD, FDDI, and Lynch** — "for CSMA, FDDI, Fischer and Lynch, the
  three algorithms generated and expanded the same number of nodes."
  <https://arxiv.org/abs/1110.3705>
- **TChecker** (open-source TA checker, LaBRI). Ships the reference model set:
  `fischer`, `csma-cd`, `fddi`, `critical-region`, `lynch`(-Shavit), each `N`-scalable
  (e.g. `fischer7.xml`, `csma-cd9.xml`, `fddi10.xml`, `lynch10.xml`,
  `critical-region5.xml`). <https://github.com/fredher/tchecker> ·
  <https://www.labri.fr/perso/herbrete/tchecker/index.html>
- **XtaBenchmarkSuite** (Farkas & Bergmann, "Towards Reliable Benchmarks of Timed
  Automata," 2018). A UPPAAL-XTA collection whose six *scalable* models are exactly
  **Fischer [AL92], CSMA [Y97], FDDI/Token-Ring [Jain94], Critical, Lynch, and
  Train-gate [AHV93]**, plus ~20 fixed models (Bang-Olufsen, circuits, schedulers).
  <https://github.com/farkasrebus/XtaBenchmarkSuite> ·
  <https://theta.mit.bme.hu/publications/minisy2018fr.pdf>
- **NUS PAT "Benchmarks of Timed Automata."** Tabulates Fischer, Lynch-Shavit,
  CSMA/CD, Critical Region, and Token-Ring/FDDI with scaling by process count and
  by the magnitude of the timing constants.
  <https://www.comp.nus.edu.sg/~pat/bddlib/timedexp.html>
- **Herbreteau et al., "Fast detection of cycles in timed automata,"
  arXiv:1410.4509.** Uses the same families for *liveness* benchmarking.
  <https://arxiv.org/abs/1410.4509>

### 1.1 The canonical families, summarised

| Family | Models | Scaling param | Typical query | Origin |
|---|---|---|---|---|
| **Fischer** | timed mutual exclusion via a shared id-lock | `N` processes | `A[]` mutex safety; `-->` liveness | Abadi–Lamport; ubiquitous |
| **Lynch-Shavit** | mutex combining Fischer + Lamport, two shared vars | `N` processes | `A[]` mutex safety | Lynch–Shavit; TChecker `lynch` |
| **CSMA/CD** | `N` stations on one bus, collision detect + backoff | `N` stations | `E<>` collision; `A[]` no-undetected-overlap | Tripakis–Yovine (KRONOS) |
| **GRC / Train-gate** | `N` trains, one gate + controller | `N` trains | `A[]` "train in ⇒ gate down" | Alur–Henzinger–Vardi; UPPAAL demo |
| **FDDI / Token ring** | `N` stations pass a token round a ring | `N` stations | `E<>`/`A[]` token-rotation timing | Jain'94; KRONOS |
| **Dining Philosophers** | `N` phils + `N` forks, timed think/eat | `N` philosophers | `E<>` a phil eats; `A[]` no shared-fork clash | classic concurrency |
| **Critical Region** | `N` readers/writers count into a region | `N` processes | `A[]` region safety | PAT/TChecker `critical-region` |
| **Leader election** | `N` processes elect a leader | `N` processes | `A[]` parent-consistency; `A<>` a leader elected | Fontana–Cleaveland; Milner-style |
| **Milner scheduler** | `N` cyclic cells pass an activation token | `N` cells | `A[]` timing; `-->` each cell runs | Milner CCS, timed |

### 1.2 Why the classic UPPAAL train-gate demo is *not* usable as-is

The shipped UPPAAL `demo/train-gate.xml` declares `chan appr[N], stop[N],
leave[N]; urgent chan go[N]; id_t list[N+1]; int[0,N] len;` and a `Gate`
template with `enqueue()/dequeue()/front()/tail()` functions and `select e :
id_t` on transitions (verified against the model source). Channel arrays,
integer arrays, user functions, `select`, and `len++` are **all outside
PTCTLMC's parser**. The suite therefore uses the **GRC formulation** (§5.6),
where the train queue length is encoded in the *controller's location structure*
(2N+2 locations) rather than an incremented integer — a faithful, in-scope
stand-in for the same protocol. This is the standard textbook GRC (Fontana &
Cleaveland §5; Alur–Henzinger–Vardi).

---

## 2. Characteristics matrix

The axes that actually move the needle for a DBM/zone checker **and** for a
*parallel* one:

- **C1 — Clocks / DBM dimension.** A zone is a dense `(K+1)²` DBM; every op is
  `O(K²)`–`O(K³)`. "Few clocks" (1/process) vs "many clocks" (2/process, FDDI) is
  the difference between cheap and expensive per-zone work.
- **C2 — Product size / state count.** The scaling parameter `N`; drives the number
  of symbolic states (the parallel work). Target ≈ `10⁴`–`10⁷`.
- **C3 — Synchronisation style.** shared-var / binary-channel / broadcast-channel /
  none — exercises different successor-generation code paths.
- **C4 — Integer-variable pressure.** 0 / 1 / 2 / many ints → widens the packed
  discrete key and the successor branching.
- **C5 — Diagonal constraints.** none (GPU-eligible, LU widening) vs present
  (CPU-only, exact zones, blow-up risk).
- **C6 — State-space shape.** **wide BFS** (large frontier per level → lots of
  independent work → *parallel-friendly*) vs **deep/narrow** (small frontier,
  long chains → *parallel-hostile*).
- **C7 — Property type.** `A[]` safety (explores everything, deterministic work,
  best speedup) vs `E<>` reachability (may early-exit, order-sensitive) vs
  liveness (sequential only).

| # | Benchmark | C1 clocks | C2 state count | C3 sync | C4 ints | C5 diag | C6 shape | C7 property | GPU? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Fischer** | 1/proc | high (exp in N) | shared var | 1 | no | **wide** | `A[]`, `-->` | ✅ |
| 2 | **Two-lock / Lynch-Shavit** | 1/proc | high | shared var ×2 | 2 | no | wide | `A[]` | ✅ |
| 3 | **Arbiter** | 1/proc | medium (serialised) | **binary chan** | 0 | no | medium | `A[]` | ✅ |
| 4 | **CSMA/CD** | 1/proc + bus | high | **broadcast** | 0 | no | wide | `E<>`, `A[]` | ✅ |
| 5 | **Dining Philosophers** | 1/proc | **very high** | binary chan ×2N | 0 | no | **widest** | `E<>`, `A[]` | ✅ |
| 6 | **GRC / Train-gate** | N+2 | medium–high | binary chan | 0 | no | medium (heterogeneous) | `E<>`, `A[]` | ✅ |
| 7 | **FDDI token ring** | **2/proc (2N+1)** | low (polynomial) | binary chan | 0 | no | **deep/narrow** | `E<>`, `A[]` | ✅ |
| 8 | **Single (many-clock)** | **n in one TA** | high (zone blow-up) | none | 0 | no | wide-zone | `E<>`, `A[]` | ✅¹ |
| 9 | **Diagonal ring** | 1/proc | small (capped) | none/shared | 0 | **yes** | narrow | `E<>`, `A[]` | ❌ (CPU) |
| 10 | *Leader election* (opt.) | 1/proc | high | binary chan ×N² | 0 | no | wide | `E<>`, `A<>` | ✅ |
| 11 | *Milner scheduler* (opt.) | 1/proc | medium | binary chan | 0 | no | ring/medium | `A[]`, `-->` | ✅ |

¹ Single-automaton model; runs through the `reachable` (single-TA) path.

**Coverage argument.** Every axis is spanned and, critically, *contrasted*:
few-clocks (1–4) vs many-clocks (7, 8); all three sync styles (shared 1/2, binary
3/5/6/7, broadcast 4, none 8); int-pressure 0→2 (3/4 → 1 → 2); diagonal on (9) vs
off (all others); and — the axis that makes or breaks a *parallel* story — wide
BFS (1, 2, 4, 5) vs deep/narrow (7). Philosophers (5) is the deliberate best-case
for the parallel backends; FDDI (7) is the deliberate worst-case (narrow frontier,
so it honestly bounds where parallelism stops helping and clock-dimension cost
takes over). Safety-vs-reachability (C7) is covered by giving most benchmarks both
an `A[]` (explore-all) and an `E<>` (early-exit) query so the two behaviours can be
compared on the *same* model.

---

## 3. State-count target and scaling ranges

The scaling ranges below are chosen to sweep ≈`10⁴`→`10⁷` *explored symbolic
states* (post-reduction — PTCTLMC applies active-clock reduction + aLU + convex-
union merge, so counts are far below the naive product; e.g. Fischer(5) is
~12.6k, not ~154k). FDDI is the intentional exception: its reachable zone graph is
*polynomial* in `N`, so we scale it to grow the **DBM dimension** (clocks), not the
state count.

| Benchmark | Param | Range (start → ~10⁷) | Cap reason |
|---|---|---|---|
| Fischer | `N` procs | 5 → 13 | width cap `N≲28`; 10⁷ well before that |
| Two-lock | `N` procs | 4 → 10 | 6 locs + 2 ints → `N≲16` |
| Arbiter | `N` clients | 6 → 16 | serialised → needs more procs for volume |
| CSMA/CD | `N` stations | 4 → 12 | width `N≲30` |
| Philosophers | `N` phils | 6 → 16 | width `3N<64` → `N≲21` |
| GRC | `N` trains | 4 → 14 | controller `2N+2` locs; width fine to `N≈20` |
| FDDI | `N` stations | 5 → 25 | scales *clocks* (2N+1); state count stays low |
| Single | `n` clocks, `D` | (n=4,D=6) → (n=9,D=20) | zone blow-up combinatorial |
| Diagonal ring | `N` stages | 3 → 8 | CPU exact zones + 500k cap |

---

## 4. Reused vs new generators

- **Reuse as-is:** `gen_fischer.py`, `gen_twolock.py`, `gen_arbiter.py`,
  `gen_csma.py`, `gen_single.py` (all already emit `verifyta`-valid, PTCTLMC-
  parseable flat-1.6 XML; verified against the parser).
- **New generators to implement:** `gen_phil.py`, `gen_grc.py`, `gen_fddi.py`,
  `gen_diag.py`, and optionally `gen_leader.py`, `gen_milner.py`.

Every generator takes the scaling parameter as `argv[1]` and prints XML to stdout,
matching the existing convention. Each spec below is detailed enough to implement
without further research.

---

## 5. Per-benchmark specifications

Notation: locations are `Name`; clock `x_i` is process `i`'s clock; `k`, `D`, `σ`,
`λ`, … are integer constants. All guards/invariants shown are already within the
parser grammar of §0.1. XML must escape `<`→`&lt;`, `>`→`&gt;`, `&`→`&amp;`.

### 5.1 Fischer (existing `gen_fischer.py`) — shared-variable mutex, wide BFS

- **Stresses:** C3 shared int, C6 wide symmetric BFS, C7 both. The flagship
  parallel benchmark and the field's default.
- **Declaration:** `clock x0..x{N-1}; int[0,N] id;`
- **Per process `Pi` (locations):** `idle`, `req` (inv `x_i<=k`), `wait`, `cs`.
  Edges: `idle→req [id==0] {x_i:=0}`; `req→wait {id:=i+1, x_i:=0}`;
  `wait→cs [x_i>k && id==i+1]`; `wait→idle [x_i>k && id!=i+1]`; `cs→idle {id:=0}`.
- **Query (safety, headline):** `A[] not (P0.cs and P1.cs)` → *satisfied*, explores
  all. **Query (reachability):** `E<> P0.cs` → *satisfied*, early-exits.
  **Query (liveness, coverage):** `P0.wait --> P0.cs`.
- **Scope:** fully in scope, diagonal-free, **GPU-eligible**. Default `k=2`.

### 5.2 Two-lock / Lynch-Shavit family (existing `gen_twolock.py`) — 2-int pressure

- **Stresses:** C4 (two shared ints — exercises the multi-field packed key that a
  single-int Fischer never hits), C6 wide.
- **Declaration:** `clock x0..; int[0,N] a; int[0,N] b;` Each process runs Fischer
  on `a` then, holding `a`, Fischer on `b`, before `cs`; releases both with
  `{a:=0, b:=0}`. Six locations `idle,reqA,waitA,reqB,waitB,cs` (invariants
  `x_i<=k` on `reqA,reqB`).
- **Query:** `A[] not (P0.cs and P1.cs)` (mutex holds by the nested-lock argument).
- **Scope:** in scope, diagonal-free, **GPU-eligible**. Represents the literature's
  **Lynch-Shavit** slot (two-variable timed mutex).

### 5.3 Arbiter (existing `gen_arbiter.py`) — binary-channel central server

- **Stresses:** C3 binary handshake, C4=0, and a *serialised* shape (only one client
  in `cs`) → a useful contrast to the wide mutexes.
- **Declaration:** `clock x0..; chan acq, rel;` `Arb`: `Free --acq?--> Busy
  --rel?--> Free`. `Ci`: `idle --[x_i>=P]--> want --acq!,{x_i:=0}--> cs (inv
  x_i<=D) --rel!,{x_i:=0}--> idle`.
- **Query:** `A[] not (C0.cs and C1.cs)` (mutex by construction — the arbiter
  serialises). Also `E<> C0.cs`.
- **Scope:** in scope, diagonal-free, **GPU-eligible**.

### 5.4 CSMA/CD (existing `gen_csma.py`) — broadcast bus

- **Stresses:** C3 **broadcast** (one `cd!` → all senders back off), C1 `N+1`
  clocks (per-station `y_i` + bus `c`), C6 wide.
- **Declaration:** `clock y0.., c; broadcast chan cd; chan begin, end;` `Bus`:
  `Idle`, `Active` (inv `c<=σ`), `Transmitting`, `Collided`. Stations `Si`:
  `Wait`, `Send` (inv `y_i<=λ`), `Retry`. (Full transition set already in the
  generator; `σ`=propagation, `λ`=frame, `λ>σ`.)
- **Query (reachability):** `E<> Bus.Collided` → *satisfied*.
  **Query (safety):** `A[] not (S0.Send and S1.Send and Bus.Transmitting)`
  (no undetected overlap) → *satisfied*, explores all.
- **Scope:** in scope, diagonal-free, **GPU-eligible**.

### 5.5 Dining Philosophers (**new** `gen_phil.py`) — the parallel showcase

- **Stresses:** C3 binary channels at scale (`2N`), C6 the **widest** frontier
  (many philosophers act independently each level), C4=0, diagonal-free →
  **GPU-eligible**. Chosen as the best-case demonstration that `omp`/`metal`
  dominate single-threaded `verifyta`.
- **Declaration:** `clock c0..c{N-1}; chan take0..take{N-1}, put0..put{N-1};`
  (2N channels; philosophers have clocks, forks do not.)
- **Fork `Fi` (2 locations):** `Free`, `Used`.
  `Free --take{i}?--> Used`; `Used --put{i}?--> Free`.
- **Philosopher `Pi` (4 locations):** `Think`, `HoldL`, `Eat` (inv `c_i<=EAT`),
  `Dropped`. Philosopher `i` uses fork `i` as *left* and fork `(i+1)%N` as *right*:
  1. `Think --take{i}!, {c_i:=0}--> HoldL` (grab left fork)
  2. `HoldL --take{(i+1)%N}!, {c_i:=0}--> Eat` (grab right fork; start eat timer)
  3. `Eat --[c_i>=1] put{i}!--> Dropped` (release left)
  4. `Dropped --put{(i+1)%N}!--> Think` (release right)
  Channel `take{i}` has **two possible senders** (phil `i` as left, phil
  `(i-1+N)%N` as right) and one receiver (fork `i`) — the engine enumerates the
  pairing correctly (binary-channel handling in `successors_net`).
- **Query (reachability):** `E<> P0.Eat` → *satisfied*, early-exit.
  **Query (safety, headline, explore-all):** `A[] not (P0.Eat and P1.Eat)` →
  *satisfied* (adjacent philosophers 0 and 1 share fork 1, so they can never eat
  together; confirming it forces exploration of the **entire** reachable set).
- **Width:** `3N < 64` → `N ≲ 21`. Recommended `N=6..16` for `10⁴`–`10⁷`.
- **Note:** the classic symmetric protocol has a reachable deadlock (everyone holds
  their left fork). That does **not** affect these queries (we do not use the
  unsupported `deadlock` keyword). If a deadlock-free variant is wanted, make the
  highest-indexed philosopher grab right-then-left (asymmetric) — a one-line
  generator branch — but the symmetric version is the standard and is fine here.

### 5.6 GRC / Train-gate (**new** `gen_grc.py`) — heterogeneous, counter-in-locations

- **Stresses:** C3 binary channels, **heterogeneous** components (trains + gate +
  controller), and the *encode-the-counter-in-locations* technique that keeps a
  queue-based protocol in scope without integer increments. Canonical demo name
  reviewers expect; diagonal-free → **GPU-eligible**.
- **Constants (paper defaults, chosen so safety holds):** `CD=1` (controller
  delay), `GLT=2` (gate lower time), `GRT=2` (raise time), `TP=4` (train pending),
  `TDU=15` (in-region upper), `TDL=1` (lower). Note `TP ≥ CD+GLT` guarantees the
  gate is down before a train reaches `in`.
- **Declaration:** `clock x0..x{N-1}, y, z; chan approach, exit, lower, raise;`
- **Train `Ti` (3 locations):** `far`, `near` (inv `x_i<=TP`), `in` (inv
  `x_i<=TDU`).
  `far --approach!, {x_i:=0}--> near`;
  `near --[x_i>=TP], {x_i:=0}--> in`;
  `in --[x_i>=TDL] exit!--> far`.
- **Gate `G` (4 locations):** `up`, `low` (inv `y<=GLT`), `down`, `rai` (inv
  `y<=GRT`).
  `up --lower?, {y:=0}--> low`; `low --[y>=GLT]--> down`;
  `down --raise?, {y:=0}--> rai`; `rai --[y>=GRT]--> up`.
- **Controller `Ctl` (2N+2 locations):** `up`, `low_1..low_N`, `dn_1..dn_N`, `rai`.
  Clock `z`; invariant `z<=CD` on each `low_k` and on `rai`. Transitions (the
  location index *is* the count of approaching-or-in trains):
  - `up --approach?, {z:=0}--> low_1`
  - `low_k --approach?--> low_{k+1}` for `k<N`
  - `low_k --lower!--> dn_k` (issue lower to gate)
  - `low_1 --exit?, {z:=0}--> up` (the only pending train left before lowering)
  - `dn_k --approach?--> dn_{k+1}` for `k<N`
  - `dn_k --exit?--> dn_{k-1}` for `k>1`
  - `dn_1 --exit?, {z:=0}--> rai` (last train leaving)
  - `rai --raise!--> up`
- **Query (safety, headline):** `A[] not (T0.in and G.up)` → *satisfied* (a train
  is never inside the region while the gate is fully up; explores all).
  **Query (reachability sanity):** `E<> T0.in` → *satisfied*.
- **Width:** trains `2N` bits + gate 2 + controller `bits(2N+2)` — comfortably
  `<64` up to `N≈20`. Recommended `N=4..14`.

### 5.7 FDDI token ring (**new** `gen_fddi.py`) — high DBM dimension, deep/narrow

- **Stresses:** C1 **two clocks per station (2N+1 total)** → large DBMs; C6
  **deep/narrow** frontier (one token → mostly one station active) → the honest
  *worst case* for parallelism, where per-zone `O(K²)` cost dominates. Diagonal-
  free variant → GPU-eligible, but expect little omp/metal speedup (small
  frontier); its value is the **clock-dimension axis** and the `verifyta`
  comparison on wide DBMs.
- **Constants:** `SA` = synchronous transmission budget (e.g. `20`), `TRTT` =
  target token-rotation bound for the property (e.g. `50`). Diagonal-free.
- **Declaration:** `clock x0..x{N-1}, w0..w{N-1}; chan tt0..tt{N-1};`
  (`x_i` = transmission timer, `w_i` = idle-since-release timer; `tt_i` = "token
  arrives at station i".)
- **Station `Si` (2 locations):** `idle`, `active` (inv `x_i<=SA`).
  `idle --tt{i}?, {x_i:=0}--> active` (token arrives, begin transmitting);
  `active --[x_i>=SA] tt{(i+1)%N}!, {w_i:=0}--> idle` (done, pass token, start idle
  timer).
- **Initial state:** station 0 starts in `active` (holds the token), all others
  `idle`. (Implement by emitting `<init ref="s0_active"/>` for `S0` and
  `<init ref="s{i}_idle"/>` otherwise.)
- **Query (reachability, tests the 2nd clock):** `E<> (S0.idle and w0>=TRTT)` →
  token rotation exceeded the target (verdict depends on constants; pick `TRTT`
  so both tools agree). **Query (safety):** `A[] not (S0.idle and w0>=TRTT)`.
- **Width:** `N` location bits (`< 64` → `N≲60`), but **clocks 2N** make DBMs the
  bottleneck. Recommended `N=5..25`.
- **Scope caveat:** if a *faithful* FDDI variant with an explicit rotation-timer
  comparison forces a diagonal (`x_i - w_j`), it drops to CPU/exact zones per
  §0.3 — keep the spec above (constant comparisons only) to remain GPU-eligible.

### 5.8 Single many-clock automaton (existing `gen_single.py`) — zone blow-up, PTCTLMC's win

- **Stresses:** C1 `n` clocks in **one** template → combinatorial zone explosion;
  diagonal-free; runs through the single-TA `reachable` path. This is the model on
  which PTCTLMC already **beats `verifyta` ~3×** (ROADMAP: n=6,D=6 → 9033 zones,
  seq 1.6s / omp 1.3s vs verifyta 4.9s). Keep it as the "we win on clock-heavy
  zone search" datapoint.
- **Model:** one location `A` (inv `⋀_i x_i<=D`), a self-loop per clock
  `A --[x_i>=1], {x_i:=0}--> A`.
- **Query:** `E<> (x0>=D and x1>=D)` (a deep corner of the zone graph) and/or an
  `A[]` bound. Provide a `.q` alongside.
- **Scope:** in scope, diagonal-free. Recommended `(n,D)`: `(4,6)`→`(9,20)`.

### 5.9 Diagonal ring (**new** `gen_diag.py`) — the diagonal axis + CPU-fallback validation

- **Stresses:** C5 **diagonal constraints** — the only benchmark that exercises the
  `network_has_diagonal` path: GPU expander returns null (CPU-only), exact zones,
  plain-inclusion subsumption, 500k cap. Dual purpose: (a) *validate* diagonal
  correctness against `verifyta`, and (b) *measure* the cost of losing LU widening.
- **Declaration:** `clock x0..x{N-1};` (ring of `N` stages, pred of `i` is
  `(i-1+N)%N`).
- **Stage `Pi` (2 locations):** `A` (inv `x_i<=M`, e.g. `M=10`), `B`.
  `A --[x_i - x{pred} <= D and x_i>=1], {x_i:=0}--> B` (the diagonal — process `i`'s
  clock may not drift more than `D` ahead of its predecessor's);
  `B --[x_i>=1], {x_i:=0}--> A`.
  Use a small `D` (e.g. `2`) so the reachable set stays under the cap.
- **Query:** `E<> P0.B` (reachability) and a diagonal safety
  `A[] not (x0 - x1 > BIG)`.
- **Scope:** **CPU-only** (diagonal). `--backend metal` silently falls back — verify
  via `PTCTLMC_STATS` that `[gpu] … ACTIVE` does **not** appear. Keep `N=3..8`.

### 5.10 Optional extensions

- **Leader election (`gen_leader.py`).** `N` single-automaton processes, parent
  stored in location (`0` = no parent, `j` = parent is `j`), `O(N²)` binary
  `req_ij` channels, per-process clock `x_i` (inv `x_i<=CPD`). Adds asymmetric-
  resolution + many-channel coverage and a natural liveness query. Safety
  (explore-all): `A[] not (P0.par1 and P1.par0)` (no mutual-parent 2-cycle) →
  *satisfied*. Liveness: `A<> P0.leader` (sequential). Width caps `N≲14` (`N+1`
  locations/process). Follows Fontana–Cleaveland §6.
- **Milner cyclic scheduler (`gen_milner.py`).** `N` cyclic cells in a ring passing
  an activation token via binary channels; 1 clock/cell; a **ring/medium** shape
  distinct from the wide mutexes. Good for a `-->` liveness coverage datapoint and
  a moderate-frontier parallel measurement.

---

## 6. Cross-check: does each model pass *both* tools?

Before any timing, run the differential harness (`tests/validate_ta.py`, already
present) on every `(model, query)` at small `N` — a benchmark is admitted only if
`verifyta` accepts it **and** the two tools return the identical verdict.

| Benchmark | PTCTLMC parser | `verifyta` accepts | Diagonal → GPU? | Headline query | Verdict (both agree) |
|---|---|---|---|---|---|
| Fischer | ✅ | ✅ | no → GPU | `A[]` mutex | satisfied |
| Two-lock | ✅ | ✅ | no → GPU | `A[]` mutex | satisfied |
| Arbiter | ✅ | ✅ | no → GPU | `A[]` mutex | satisfied |
| CSMA/CD | ✅ | ✅ | no → GPU | `E<>` collision / `A[]` | sat / sat |
| Philosophers | ✅ | ✅ | no → GPU | `A[]` no-clash | satisfied |
| GRC | ✅ | ✅ | no → GPU | `A[]` train-in⇒gate-down | satisfied |
| FDDI | ✅ | ✅ | no → GPU | `E<>`/`A[]` rotation | (tune constants) |
| Single | ✅ | ✅ | no → GPU¹ | `E<>` deep zone | (design → sat) |
| Diagonal ring | ✅ | ✅ | **yes → CPU** | `E<>`/`A[]` | (design; both agree) |
| Leader (opt.) | ✅ (`N≲14`) | ✅ | no → GPU | `A[]` consistency | satisfied |
| Milner (opt.) | ✅ | ✅ | no → GPU | `A[]`/`-->` | (design) |

¹ Single-template model → single-TA path; GPU eligibility is via `reach.cpp`, not
the network expander.

**Gotchas already accounted for in the specs above:** clock resets are only ever to
`0`; integers are only ever assigned constants (Fischer's `id:=i+1`, never a
counter); no guard compares two variables; invariants are clock-only; every
`A[]` query is written in the negatable `A[] not(loc ∧ loc / clock / int)` form the
engine supports (never `A[]` over a *positive* location predicate, which
`check_net_query` reports unsupported); no query uses `deadlock`.

---

## 7. Evaluation protocol (TACAS-grade)

### 7.1 Configurations under test

- `verifyta` (UPPAAL 5.0.0) — the reference, **single-threaded by default** (state
  this in every table caption).
- PTCTLMC `--backend seq` — sequential baseline (parallel-efficiency denominator).
- PTCTLMC `--backend omp` — 12 threads (`OMP_NUM_THREADS=12`). *Per the perf notes,
  omp is the fast network backend and already beats `verifyta` ~2.6–3.5× on
  Fischer.*
- PTCTLMC `--backend metal` — GPU; **only for diagonal-free benchmarks**, and only
  where `[gpu] network expander ACTIVE` is confirmed (else it is silently the CPU
  path and must be reported as such, not as a GPU number).

### 7.2 Timing methodology

- **Warm-up + min-of-k.** For each `(config, benchmark, N, query)`: 1 discarded
  warm-up run, then `k=5` timed runs; report the **minimum** wall-clock (min is the
  least-noisy estimator of the machine's true capability) and also the median for
  transparency. Never report a single run.
- **Timeout** 300 s (600 s for the largest `N`); a timeout is recorded as `>T`,
  never as a number, and truncates that benchmark's scaling curve.
- **What is timed.** Prefer *verification-only* time. PTCTLMC prints its result
  immediately; wrap the whole process and, for the largest `N`, note that parse
  time is negligible relative to search. For `verifyta`, use its reported
  verification/CPU time when available, else subtract a measured empty-model
  startup baseline so UPPAAL's fixed JVM-free-but-nonzero startup is not charged as
  search time. Document exactly which number is used.
- **Environment.** Single quiet machine; fixed CPU governor / disable turbo if
  possible (or report that variance is why min-of-k is used); record CPU model +
  core count and the Apple-silicon GPU for `metal`; Release build, same compiler
  flags for all backends.

### 7.3 What to tabulate (state-count parity is mandatory)

For each `(benchmark, N, query)` record:

1. **explored symbolic states — PTCTLMC vs `verifyta`.** PTCTLMC's `[explored N
   symbolic states]` (and `PTCTLMC_STATS` `discrete-states`/`stored-zones`);
   `verifyta`'s explored/stored count via its statistics flag (`-u`; confirm the
   exact flag with `verifyta --help` for 5.0). **These must match** (see §7.5). Put
   them side by side in the table — the reader must be able to see parity at a
   glance. A row where they diverge is either excluded from the speedup discussion
   or accompanied by an explanation (search order, federation subsumption).
2. **wall-clock** for `seq`, `omp`, `metal`, `verifyta` (min and median).
3. **speedups:** `verifyta / omp`, `verifyta / metal` (headline), `seq / omp`
   (parallel efficiency, ÷12 gives per-core scaling), `omp / metal` (does the GPU
   pay off).
4. **peak RSS** per config; mark OOM.
5. **verdict** (must be identical across all four — pre-checked by the differential
   harness; a timing run with a mismatched verdict is void).

### 7.4 How to present

- **Scaling plot — states vs N** (log-y): PTCTLMC and `verifyta` curves *overlaid*.
  If they coincide, that single picture is the fairness proof; publish it first.
- **Scaling plot — wall-clock vs N** (log-y): one line per config, per benchmark.
- **Speedup-vs-`verifyta` vs N**: `omp` and `metal` on one axis; the crossover `N`
  (where the parallel backends overtake single-threaded UPPAAL) is the headline.
- **Speedup-vs-`seq` vs N** (parallel efficiency): shows scaling independent of the
  UPPAAL comparison, and honestly reveals the narrow-frontier ceiling on FDDI.
- **Grouped narrative:** lead with the wide-BFS winners (Philosophers, Fischer,
  CSMA), then the honest deep/narrow case (FDDI) where parallelism saturates, then
  the single-TA win (Single) and the CPU-only diagonal case (Diagonal ring).

### 7.5 Fairness / threats-to-validity checklist

- [ ] **Same reachable state space.** Explored/stored counts match `verifyta`
      (±small, explained). This is the #1 control; a speedup over a tool doing
      *more* work is meaningless. PTCTLMC's per-location aLU bounds + active-clock
      reduction + convex-union merge are what make the counts match UPPAAL — verify
      per benchmark, not just Fischer.
- [ ] **Same query & semantics.** Identical verdict across `seq/omp/metal/verifyta`
      (differential harness) before any timing.
- [ ] **Same abstraction settings.** `verifyta` run with **default** options
      (exact zone search) — **no** over-approximation (`-A` convex-hull) or
      under-approximation, which would change the state space. PTCTLMC uses exact
      zones + *sound* aLU. State both.
- [ ] **Single-threaded reference stated.** Every table/plot caption says
      `verifyta` is single-threaded. Report **both** raw `omp` (throughput story)
      and `omp ÷ 12` normalised (a conservative per-core comparison) so a reviewer
      cannot claim the win is "just more cores."
- [ ] **No asymmetric reductions.** Neither tool uses symmetry reduction (PTCTLMC
      has none; the flattened generators use no `scalarset`, so UPPAAL gets none
      either). Fair by construction — keep it that way.
- [ ] **Search order for `E<>`.** `E<>`-true queries early-exit and are order-
      sensitive; match `verifyta`'s default BFS to PTCTLMC's level-sync BFS, and
      base **headline** speedups on **`A[]`-safety** (explore-all, order-
      independent). Report `E<>` separately and note the caveat.
- [ ] **GPU actually ran.** For every `metal` number, confirm
      `[gpu] network expander ACTIVE`; a diagonal model (or a too-wide one) silently
      falls back to CPU and must not be reported as a GPU result.
- [ ] **Min-of-k + warm-up** applied uniformly; report min *and* median; timeouts as
      `>T`.
- [ ] **Same hardware, same build.** One machine, Release, documented CPU/GPU;
      `OMP_NUM_THREADS` pinned and reported.
- [ ] **Parse/startup accounted for** consistently and disclosed (UPPAAL's fixed
      startup vs PTCTLMC's; subtract or amortise at large `N`).
- [ ] **Determinism.** `seq == omp == metal` verdicts are unit-tested; re-assert on
      each benchmark.

---

## 8. Recommended first implementation steps

For the engineer, in priority order:

1. **`bench_run.py` — the evaluation harness.** Without it there are no defensible
   numbers. It should: sweep `N` for a benchmark; run `seq/omp/metal/verifyta` with
   warm-up + min-of-k + timeout; parse *both* tools' explored-state counts and
   **assert they match** (fail loudly on divergence); confirm `metal` really used
   the GPU (`PTCTLMC_STATS` → `ACTIVE`); and emit a tidy CSV (`benchmark,N,query,
   config,states_ptctlmc,states_verifyta,t_min,t_median,rss,verdict`) that feeds
   the plots of §7.4. This is the backbone of the whole evaluation.
2. **`gen_phil.py` — Dining Philosophers.** The strongest *new* datapoint: diagonal-
   free (GPU-eligible), the widest BFS in the suite, channel-heavy, and with an
   `A[]`-safety explore-all query it is the cleanest demonstration that `omp`/`metal`
   beat single-threaded `verifyta`. Spec in §5.5.
3. **`gen_grc.py` — GRC / Train-gate.** Adds the canonical demo name that reviewers
   expect, plus heterogeneous-component and binary-channel coverage, using the
   counter-in-locations technique that keeps it in scope (§5.6). It also documents,
   in code, *how* PTCTLMC handles a "queue" protocol without integer increments —
   a reusable pattern.

After those three, add `gen_fddi.py` (clock-dimension axis / honest parallel
worst case) and `gen_diag.py` (diagonal axis + CPU-fallback validation); the two
optional generators (`gen_leader.py`, `gen_milner.py`) round out liveness and ring-
shape coverage.

---

## Sources

- Fontana & Cleaveland, *Timed Automata Benchmark Description*, arXiv:2005.13151 —
  <https://arxiv.org/abs/2005.13151>
- Herbreteau, Srivathsan, Walukiewicz, *Better abstractions for timed automata*,
  LICS 2012 / arXiv:1110.3705 — <https://arxiv.org/abs/1110.3705>
- Herbreteau et al., *Fast detection of cycles in timed automata*, arXiv:1410.4509 —
  <https://arxiv.org/abs/1410.4509>
- TChecker — <https://github.com/fredher/tchecker> ,
  <https://www.labri.fr/perso/herbrete/tchecker/index.html>
- Farkas & Bergmann, *Towards Reliable Benchmarks of Timed Automata* (2018) &
  XtaBenchmarkSuite — <https://theta.mit.bme.hu/publications/minisy2018fr.pdf> ,
  <https://github.com/farkasrebus/XtaBenchmarkSuite>
- NUS PAT, *Benchmarks of Timed Automata* —
  <https://www.comp.nus.edu.sg/~pat/bddlib/timedexp.html>
- UPPAAL — <https://uppaal.org/> (train-gate demo model, `demo/train-gate.xml`,
  uses channel/int arrays + functions + `select` → out of PTCTLMC scope; motivates
  the GRC formulation).
