# Can parallelism reduce the symbolic-state count on A[] reachability?

**Verdict: NO on the stored count — the floor is hard, and this document proves it six
independent ways. The one axis that is *not* at its floor is the `explored` count, where a
measured 36–38 % of expansions are provably wasted; the single strongest idea attacks that,
and its central obstacle is also proved here.**

- **Binary:** `./build-mostopt/ptctlmc` (HDBM on `main`), `PTCTLMC_STATS=1`.
- **Reference:** `/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta -u`.
- **Machine:** 12 logical cores (8 perf), darwin 25.5.0.
- **New measurement-only probes added for this study** (both env-gated, never on a search path):
  `PTCTLMC_FEDPROBE` (+`PTCTLMC_FEDPROBECAP`) in `src/netreach.cpp` and
  `NetPassed::for_each_antichain` in `include/ptctlmc/detail/net_passed.hpp`; and
  `PTCTLMC_GAPHIST`, which finally prints the already-collected `gapHist` array.
  `./build-mostopt/ptctlmc --selftest` → OK after both.

---

## 0. There are two counts, and only one of them was ever at the floor

`PTCTLMC_STATS=1` prints `stored-zones` (live passed-set size at the fixpoint); the CLI prints
`explored N symbolic states` (sum of frontier sizes = number of **expansions**). verifyta prints
both as `States stored` / `States explored`. The project's "we tie verifyta" fact is about
`stored`. The gap between the two is where every remaining opportunity lives.

| model | PTC disc | PTC stored | PTC explored | verifyta stored | verifyta explored | stored/disc | explored/stored |
|---|---|---|---|---|---|---|---|
| fischer8  | 25080 | 25080 | 40536 | 25080 | 40536 | 1.000 | 1.616 |
| fischer10 | 260998 | 260998 | 447598 | 260998 | 447598 | 1.000 | 1.715 |
| twolock8  | 27128 | 27128 | 42584 | 27128 | 42584 | 1.000 | 1.570 |
| twolock10 | 271238 | 271238 | 457838 | 271238 | 457838 | 1.000 | 1.688 |
| arbiter10 | 6144 | 6144 | 6144 | 6144 | 6144 | 1.000 | **1.000** |
| arbiter15 | 278528 | 278528 | 278528 | 278528 | 278528 | 1.000 | **1.000** |
| viking10  | 45566 | 45566 | 45566 | 45566 | 45566 | 1.000 | **1.000** |
| viking12  | 241662 | 241662 | 241662 | 241662 | 241662 | 1.000 | **1.000** |
| csma8     | 4096 | 4096 | 4096 | 5888 | 5888 | 1.000 | **1.000** |
| csma10    | 22784 | 22784 | 22784 | 34304 | 34304 | 1.000 | **1.000** |
| phil10    | 59048 | 62030 | 71476 | 94776 | 94776 | 1.051 | 1.152 |
| phil12    | 531440 | 576027 | 799551 | 1140361 | 1140361 | 1.084 | 1.388 |

(PTC columns are the default 12-thread run except where noted. phil is the one family whose
counts move with thread count at all: phil12 is 576027/799551 at 12 threads and 577183/801749 at
1 thread, a 0.20 % spread — see §1a. Every other row is bit-identical at every thread count.)

Three facts fall out immediately:

1. **The tie with verifyta is exact on *both* axes**, not just on `stored`. fischer10 explores
   447598 states in both tools; twolock10 explores 457838 in both. Two independently written
   engines agreeing to the digit on the *wasted* work as well as the stored work is strong
   evidence that both numbers are canonical properties of BFS-with-subsumption, not artefacts.
2. **Three of the six families are at the absolute floor on both axes** (arbiter, viking, csma:
   `explored == stored == |reachable discrete states|`). There is *nothing at all* to win there,
   by any technique, parallel or not.
3. The addressable surface is exactly: `stored` above the discrete floor on **phil only**
   (5.1–8.4 %), and `explored` above `stored` on **fischer, twolock, phil only**
   (61.6 %, 57.0 %, 15.2 % at the small sizes; 71.5 %, 68.8 %, 38.8 % at the larger ones).

---

## 1. Direction (a) — ORDER: rejected, four measurements

### 1a. The count is bit-identical at every thread count on the tying families

```
fischer8 t=1  stored=25080 explored=40536      phil10 t=1  stored=62030 explored=71476
fischer8 t=2  stored=25080 explored=40536      phil10 t=2  stored=62065 explored=71059
fischer8 t=4  stored=25080 explored=40536      phil10 t=4  stored=62121 explored=71839
fischer8 t=8  stored=25080 explored=40536      phil10 t=8  stored=61982 explored=71908
fischer8 t=12 stored=25080 explored=40536      phil10 t=12 stored=62089 explored=71894
```

fischer8 does not move by one state across a 12× change in thread count. phil10 — the only
family whose antichains are not singletons, hence the only one where the order-sensitive convex
merge can bite — spreads 62121 − 61982 = **139 states, 0.22 %**.

In fairness, this is the one place in the whole study where parallelism does win on count, and it
should be stated rather than buried: phil12 stores **576027 at 12 threads against 577183 at 1
thread**, so the parallel run is genuinely 1156 states (0.20 %) smaller, and phil10's minimum
across the sweep also falls at 8 threads rather than 1. The effect is real, reproducible in sign,
and far too small to build on.

A "run *k* different orders in parallel and keep the smallest passed set" scheme therefore has a
hard ceiling of **~0.2 % on one family and exactly 0 % on the other five**. It also costs *k*×
the memory, which is the actual binding constraint on these runs. Rejected.

### 1b. `stored` is order-invariant wherever it is at the floor; where it is not, BFS is already best

`verifyta -o{0,1,2}`:

| model | order 0 (BFS) | order 1 (DFS) | order 2 (random DFS) |
|---|---|---|---|
| fischer8 stored / explored | 25080 / **40536** | 25080 / 75184 | 25080 / 67281 |
| viking10 stored / explored | 45566 / 45566 | 45566 / 45566 | 45566 / 45566 |
| csma8 stored / explored | 5888 / 5888 | 5888 / 5888 | 5888 / 5888 |
| phil10 stored / explored | **94776 / 94776** | 124746 / 202573 | 125081 / 173473 |

On fischer/viking/csma `stored` does not move at all — independent confirmation, from a
different tool, that the discrete state space is the floor rather than an artefact of one search
strategy. On phil, where there *is* slack, DFS is **31.6 % worse on stored and 2.1× worse on
explored**. In every case BFS is the best order available, and BFS is what the engine runs.

This agrees with `doc_1/sections/10-limits.tex` ("expansion ordering by generality", phil12
841637 vs 803845) and with Behrmann et al. CAV 2000 (Dacapo 45001 → 110634 when the depth
ordering is relaxed). Every measurement of order ever taken in or around this project points the
same way: **BFS depth order is a local optimum that any parallel relaxation moves away from.**

---

## 2. Direction (b) — SPARE CORES BUYING PRECISION: rejected, with the headroom measured first

### 2a. On five of six families every antichain contains exactly ONE zone

`PTCTLMC_FEDPROBE=1` walks the final passed set and, for every discrete state's antichain,
greedily drops any member covered by the **union** of the survivors (`fed_covers`, exact — so a
member may be covered jointly by several others without being contained in any single one, which
is precisely what the pairwise aLU/convex-merge test cannot see). This is the maximum a
non-convex/federation representation could ever remove.

```
fischer8   antichains-with>=2=0  maxk=1  zones=25080  after-federation-reduce=25080  removable=0 (0.00%)
twolock8   antichains-with>=2=0  maxk=1  zones=27128  after-federation-reduce=27128  removable=0 (0.00%)
csma8      antichains-with>=2=0  maxk=1  zones=4096   after-federation-reduce=4096   removable=0 (0.00%)
viking10   antichains-with>=2=0  maxk=1  zones=45566  after-federation-reduce=45566  removable=0 (0.00%)
arbiter10  antichains-with>=2=0  maxk=1  zones=6144   after-federation-reduce=6144   removable=0 (0.00%)
phil10     antichains-with>=2=867  maxk=37   zones=62030  -> 60966  removable=1064  (1.72%)
phil12     antichains-with>=2=6071 maxk=298  zones=577183 -> 553279 removable=23904 (4.14%)
```

**Answer to the question the brief asked to measure first — "how many stored zones would a
non-convex union merge actually remove?": zero on five of six families, 1.7 % on phil10, 4.1 %
on phil12.** The histogram says why: 58181 of phil10's 59048 antichains have exactly one member,
and every one of fischer/twolock/csma/viking/arbiter's antichains does. A merge operator, however
strong, needs at least two zones in a bucket to do anything.

The same argument kills every other member of the "buy precision with idle cores" family at
once. A stronger simulation (G-sim), an exact-inclusion test where aLU is too coarse, a deeper
subsumption probe — all of them are *tests between two zones in the same bucket*. With one zone
per bucket there is no pair, so they are provably no-ops on those five families. This is not a
prediction; it is what `maxk=1` means.

Even where it is not a no-op, it does not reach the floor: phil12 sits 45743 zones (8.6 %) above
its discrete floor of 531440; a perfect federation merge removes 23904 of them and leaves 21839
(4.1 %) — because the remaining pairs have genuinely non-convex, genuinely non-covered unions.
The cost would be replacing an O(n²) aLU test with `fed_covers` on antichains of up to **298**
members. That is a large exponential bill for ≤4 %, on one family.

### 2b. A bigger convex-merge window saturates 25 states from where it already is

`PTCTLMC_MERGESCAN` sweep on phil10 (t=1), i.e. exactly "let the spare core scan further":

| MERGESCAN | 8 | 32 (default) | 128 | 1024 | 10⁶ (unbounded) |
|---|---|---|---|---|---|
| stored | 62264 | 62030 | 62005 | 62005 | 62005 |

The window saturates at 128 and an **unbounded** merge scan buys **25 zones, 0.04 %**. The
default of 32 is already within a rounding error of infinite compute.

### 2c. Coarser abstraction: also zero, and this one is the interesting negative

LU widening (`PTCTLMC_EXTRAP=1`) leaves `stored`, `generated` and `explored` byte-identical on
fischer8, twolock8, phil10, csma8 and viking10; only `distinct-zones` falls (twolock8 21308 →
20751, viking10 120 → 111), which is a memory-sharing effect, not a count.

The decisive version of this test is verifyta's own **convex-hull over-approximation** (`-A`),
which is the most aggressive zone abstraction either tool has — it keeps exactly one hull per
discrete state:

| model | exact stored / explored | `-A` stored / explored |
|---|---|---|
| fischer8  | 25080 / 40536 | **25080 / 40536** |
| fischer10 | 260998 / 447598 | **260998 / 447598** |
| twolock10 | 271238 / 457838 | **271238 / 457838** |
| arbiter15 | 278528 / 278528 | 278528 / 278528 |
| viking12  | 241662 / 241662 | 241662 / 241662 |
| csma10    | 34304 / 34304 | 22784 / 22784 |
| phil12    | 1140361 / 1140361 | 531440 / 531440 |

Two conclusions, both important.

* **On fischer and twolock, throwing away *all* zone precision changes neither count — not even
  the 71 % re-exploration overhead.** So on those families the entire state count, including the
  waste, is a property of the *discrete* reachability structure. No zone-level technique, no
  amount of precision bought with idle cores, and no over-approximation however coarse can move
  it by one state. That is as tight a proof of a hard floor as this benchmark set can give.
* On csma and phil the over-approximation *does* help verifyta — and it lands exactly on
  `|reachable discrete states|` (csma10 22784, phil12 531440), which is where PTCTLMC's **exact**
  engine already is on csma (22784) and 8.4 % above on phil. In other words the convex-union
  merge already buys, soundly, what UPPAAL can only reach by giving up exactness; and the
  remaining phil gap is precisely the price of exactness, not a missing optimisation.

### 2d. The general theorem behind all of 2c

Let `R` be the exact reachable set and `A ⊇ R` any sound over-approximation (over-approximation
is what A[] needs: a proof of safety on `A` proves safety on `R`). Suppose `A` abstracts only the
*timing* component, i.e. it keeps discrete states distinct and widens zones. Then
`disc(A) ⊇ disc(R)` and the abstract search stores at least `|disc(A)| ≥ |disc(R)|` symbolic
states. When the exact engine already stores exactly `|disc(R)|` — the measured situation on five
of six families — **no such abstraction can store fewer, and a coarser one can only store more.**

That covers, without further experiments, every zone-precision knob: coarser extrapolation,
convex hull, interpolation-based widening, and the whole "run a coarse copy on spare cores and
take whichever finishes with a proof" portfolio. The `-A` rows in §2c are this theorem being
confirmed empirically by the reference tool's own strongest zone abstraction.

The theorem deliberately does **not** cover abstractions that merge *discrete* states, which can
in principle store fewer objects than `|disc(R)|`. That is the only remaining escape from the
floor, and it is exactly the ground already ruled out by prior work in this project: symmetry
reduction (UPPAAL has it via scalar sets), partial-order reduction (unsound here — 24921 measured
divergence events on phil10), and predicate abstraction / CEGAR (a sibling tool measured a net
loss, 251032 nodes against 68624, because the finer structure was also acting as a canonicaliser).
The CEGAR result was not bad luck: coarsening the discrete state also destroys the canonical form
that lets one zone stand for one discrete state, and this study shows that canonical form is
precisely what puts the engine on the floor in the first place.

---

## 3. Direction (c) — SPECULATION: rejected on definition

Speculative expansion with retraction *adds* expansions: work that is thrown away was still
work performed, and `explored` counts expansions performed. The only version that reduces a
count is speculation that *avoids* an expansion (deferral), which is section 4. Speculation that
injects an optimistically-large zone at a state is an over-approximation and is answered by 2c/2d
— and answered specifically by `-A` on fischer, which injects the largest legal zone at every
discrete state and still explores 447598.

---

## 4. The one axis that is NOT at its floor — and exactly what stands in the way

### 4a. `explored − stored` is, to the state, the number of cross-level retirements

`PTCTLMC_GAPHIST=1` prints the histogram of (retirement level − insertion level):

```
fischer8   retirements=32705  same-level(g=0)=17249  cross-level=15456   raw: 0:17249  4:15456
fischer10  retirements=459280 same-level(g=0)=272680 cross-level=186600  raw: 0:272680 4:186600
twolock8   retirements=32669  same-level(g=0)=17213  cross-level=15456   raw: 0:17213  6:15456
twolock10  retirements=450181 same-level(g=0)=263581 cross-level=186600  raw: 0:263581 6:186600
phil10     retirements=24921  same-level(g=0)=15475  cross-level=9446    raw: 0:15475  4:9091 8:344 12:11
phil12     retirements=471858 same-level(g=0)=247292 cross-level=224566
           raw: 0:247292 4:211632 8:11578 12:1047 16:241 20:63 24:5
arbiter10  retirements=0
viking10   retirements=0
csma10     retirements=6853   same-level(g=0)=6853   cross-level=0
```

Check against the table in §0 (all t=1, so `stored`/`explored` are the t=1 values):

| model | explored − stored | cross-level retirements |
|---|---|---|
| fischer8 | 40536 − 25080 = 15456 | 15456 |
| fischer10 | 447598 − 260998 = 186600 | 186600 |
| twolock8 | 42584 − 27128 = 15456 | 15456 |
| twolock10 | 457838 − 271238 = 186600 | 186600 |
| phil10 | 71476 − 62030 = 9446 | 9446 |
| phil12 | 801749 − 577183 = 224566 | 224566 |

Exact, on every model. **Every wasted expansion is a zone that was stored, expanded, and then
superseded by a more general zone for the same discrete state that arrived some levels later.**
Nothing else contributes. And the three zero-tax families are zero for the reason the probe
shows: arbiter and viking never retire a single zone (the first zone stored at a discrete state
is already final), and csma's 6853 retirements all happen *inside their own level*, before the
node ever reaches the frontier, so they cost no expansion.

### 4b. The revisit gap is a model constant, not a distribution

This is the surprise. The gap is not spread out: it is a **single value on fischer and twolock —
exactly 4 on fischer at 8 *and* at 10 processes, exactly 6 on twolock at 8 *and* at 10** — and on
phil a geometrically decaying arithmetic progression in multiples of 4 (phil12:
4:211632, 8:11578, 12:1047, 16:241, 20:63, 24:5), i.e. one zone growing repeatedly on successive
trips round the ring. The gap is a structural constant of the model (the length of the discrete
cycle that grows the zone) and does not change with instance size.

Read naively, the histogram is an invitation: a policy that holds a newly stored zone for
W = 4 levels before expanding it would catch **100 %** of fischer's tax and 96.2 % of phil10's,
removing 38 % / 36 % / 13 % of all expansions on fischer10 / twolock10 / phil12.

### 4c. …and a *uniform* hold is provably a no-op. This is the crux.

Let a uniform policy insert a depth-`d` state into the passed set at logical time `T(d)` and
expand it at `T(d) + W`. Its successors are inserted at `T(d) + W + 1`, so `T(d) = d·(W+1)`.
The retiring zone sits at depth `d + g`, hence is inserted at `T(d) + g·(W+1)`. It catches the
retiree only if

  `g·(W + 1) < W`,

which is false for every `g ≥ 1` and `W ≥ 0`. **A uniform delay delays the retirer strictly more
than it delays the retiree.** It costs a (W+1)× longer critical path and saves exactly zero
expansions. The same inequality is why relaxing depth order (Behrmann's 45001 → 110634, and the
DFS columns in §1b) makes things worse rather than better: under any schedule that discovers
states in non-decreasing depth order, the retirer is *structurally* later than the retiree, so
the retiree's expansion is unavoidable unless it is *selectively* skipped.

So the tax is not reachable by any global scheduling parameter. It is reachable only by an
**oracle** that names, per state, the ones that are about to be superseded.

---

## 5. Verdict

**No.** Parallelism cannot reduce the stored symbolic-state count on these A[] queries. The
floor is `|reachable discrete states|`, the engine is on it for five of six families, and the
floor is hard against every lever available:

| lever | measured effect on `stored` |
|---|---|
| more threads (1 → 12) | 0 states on fischer; ±0.22 % on phil |
| different search order (BFS/DFS/random) | 0 on fischer, viking, csma; **+31.6 %** (worse) on phil |
| unbounded convex-merge window | −25 states (−0.04 %) on phil10 |
| perfect non-convex (federation) merge | 0 on five families; −1.7 % phil10, −4.1 % phil12 |
| stronger simulation / exact inclusion | provably 0 on five families (antichains are singletons, `maxk=1`) |
| LU widening | 0 everywhere |
| maximal zone over-approximation (`verifyta -A`) | 0 on fischer/twolock/arbiter/viking |
| any sound abstraction | provably ≥ 0 (§2d) |

The `explored` count is a different story and is genuinely 57–72 % above its own floor on
fischer/twolock and 15–39 % on phil — but §4c shows the obstacle is not compute, and parallelism
is a headwind there rather than a tailwind.

### The single strongest idea, stated with its ceiling and its blocker

**Selective deferral driven by the model's revisit-gap constant, on a barrier-free frontier.**

The mechanism, and why parallelism is the lever rather than a bystander: the gap is a *model
constant* (fischer 4, twolock 6, phil 4/8/12), it is learnable from the first few hundred
retirements at negligible cost, and it identifies the tax exactly — 100 % of fischer's and
twolock's, 96 % of phil10's. A worker that pops a state whose discrete key the predictor flags as "will be
superseded in `g` levels" does not expand it; it parks it. Section 4c proves this cannot work
under a level-synchronous schedule, where parking a state also parks its own retirer. It can
only work on an **asynchronous, barrier-free frontier where other workers run other branches
ahead by `g` levels while the parked state waits** — which is exactly a capability a
single-threaded tool does not have, and the only place in this entire study where extra cores
buy something no sequential tool could buy. Ceiling, measured: 186600 of 447598 expansions on
fischer10 (41.7 %), 186600 of 457838 on twolock10 (40.8 %), 224566 of 801749 on phil12 (28.0 %),
and **exactly zero** on arbiter, viking and csma.

**Do not start it before settling the one question that decides it, which this study did not
settle:** is the retiring zone *causally independent* of the retired one? If the bigger zone at
`q` arrives via a path that runs *through* `q`'s own earlier expansion, then parking `q` parks
its own retirer, the predictor fires on a state that will never be superseded while parked, and
the policy degenerates to a timeout with pure loss. The evidence is mixed and worrying: fischer and
twolock show exactly one growth per state (a single gap value, 4 and 6, with no tail at all),
which is consistent with independence; but phil12's decaying cascade at 4, 8, 12, 16, 20, 24 is
exactly the signature of a state growing *itself* one ring-trip at a time, and phil is the family
with the largest absolute prize. Deciding it needs one instrumented run recording, for each
cross-level retirement, whether the retirer is a descendant of the retiree — a predecessor-tracking
probe, perhaps a day of work, and far cheaper than building the asynchronous engine first. Run it
on fischer10 and phil12 and the answer is likely to differ between them, which would make the idea
worth ~40 % of expansions on fischer/twolock and nothing on phil.

If that probe says "descendant", the honest conclusion is that **both** counts are hard floors on
this benchmark set and the project should stop spending on the state-count axis altogether and
compete on wall-clock and memory, where it already wins.

### What this says about the paper

The strongest defensible claims are the ones now measured rather than asserted:

* the tie with verifyta holds on `explored` as well as `stored`, to the digit, at every size
  tested — a much stronger differential-testing statement than the current one;
* `stored` is order-invariant on the tying families, verified from *both* tools;
* the exact convex-union merge reaches, soundly, the count UPPAAL can only reach by switching on
  its convex-hull over-approximation (csma10: 22784 exact vs verifyta 34304 exact / 22784 `-A`);
  on phil12 the exact engine is 8.4 % above a floor that verifyta needs `-A` to touch at all;
* `explored − stored` is exactly the cross-level retirement count, and the revisit gap is a model
  constant — a clean, quotable characterisation of where zone-based BFS wastes work, which as far
  as this study can tell is not stated anywhere in the literature.
