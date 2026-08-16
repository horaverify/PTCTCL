# On-the-fly liveness: five directions, implemented and measured

Branch `liveness-on-the-fly`. All code in `nove_mc/prototype/netreach_gfp.cpp` (class `Otf`,
entry point `leads_to_otf`), selected by `PTCTLMC_GFPMODE`. The shipping engine is untouched;
so is the published `leads_to_gfp`, which is still reachable as `PTCTLMC_GFPMODE=gfp` and is
the baseline every number below is quoted against.

Harness: `run_otf.py` (differential + state counts), `summarise_otf.py` (aggregation),
`queries.txt` (48 queries, small models), `scale_queries.txt` (32 queries, growing models).

---

## 0. The result in three lines

* **Making the backward gfp on-the-fly needs no new convergence theory.** A greatest fixpoint
  does not have to be reached to prove a violation; by Knaster–Tarski a *post-fixpoint* is
  already a certificate, and confining the ordinary descending iteration to any subset of
  already-expanded nodes produces one. Details in §1.
* **It buys 145× fewer symbolic states on violated queries (geometric mean, 23 queries on the
  larger models; best case 4022×) and costs nothing on satisfied ones** (0.96–1.01×).
* **The best direction is not one of the three proposed.** D1/D2 (append-only, pipelined)
  *lose* 11–16 % of states on average and up to 2.5× on `grc4`; D4 changes the state count by
  exactly zero; D3 works; **D5 — forward candidate generation with exact backward confirmation
  — dominates D3** and additionally answers queries the published algorithm cannot answer at
  all.

---

## 1. Why a greatest fixpoint can go on-the-fly after all

`EG` is a greatest fixpoint, so the iteration starts at ⊤ and descends, and an intermediate
iterate is a **pre**-fixpoint `W ⊇ F(W)`. A pre-fixpoint is an over-approximation, so
"`W(v)` is non-empty" settles nothing, there is no early exit, and the values stop descending
the moment the graph underneath them changes. That is the real content of *"you need a
substantially different convergence/correctness proof"*.

The way out is that a violation never needed the fixpoint. It needs a certificate, and the
certificates of a greatest fixpoint are exactly its **post**-fixpoints:

>   `X ⊆ F(X)` implies `X ⊆ νF`.    (Knaster–Tarski)   (*)

Take any set `S` of nodes explored so far, pin every node outside `S` to ∅, and run the
ordinary descending iteration confined to `S`. Its limit `νF_S` is the exact greatest fixpoint
of the restricted operator. Restricting only *deletes disjuncts*, so `F_S ≤ F` pointwise, and
monotonicity gives

>   `νF_S ⊆ νF`,

i.e. **the limit of a pass over a fragment is a post-fixpoint of the operator on the finished
graph**. By (*) it is a sound under-approximation of the final answer, computed without ever
seeing the final graph. Hence:

* `νF_S(v) ≠ ∅` at a `p ∧ ¬q` node ⟹ **VIOLATED**, with exactly the verdict the finished graph
  would have produced. Stop; never build the rest.
* `S` = every expanded `¬q` node of the finished graph gives `F_S = F`, so the last pass *is*
  the published algorithm. Nothing is given up when the property holds.

No value ever has to descend across passes: each pass is a self-contained certificate search,
and monotonicity in the graph is free. This is what D3 and D5 rest on.

**The one restriction, and it is the only one.** A node may enter `S` only if it is
**expanded**. The stopping term needs the *absence* of successors (`adj[v].empty()`), and an
unexpanded node has an empty adjacency for the wrong reason. Everything else survives a partial
graph untouched — the `Pre` disjuncts only ever need edges that *exist*, never the absence of
one. That asymmetry is what makes backward work on a half-built graph legitimate at all.

**Caveat kept honest.** A pass must run to **convergence** before its values mean anything.
When the federation-width cap fires the pass has not converged and is neither an over- nor an
under-approximation. The first version of this code returned `false` there, laundering "out of
budget" into a silent SATISFIED; it now reports `UNUSABLE` and the harness scores it as the
absence of a verdict. See §4.

---

## 2. What each direction turned out to be

| | direction | what it actually is here | state count |
|---|---|---|---|
| **D4** | overlap the *preprocessing* | `is_p`/`is_nq`/stop-term computed during expansion instead of in a post-pass | **exactly 1.00×**, by construction — it does not change the search |
| **D2** | append-only hybrid | refuse the in-place enlargement in `intern`; store the newcomer beside the old node so every stored node is immutable from birth | **0.84–0.89×** — a real loss |
| **D1** | pipelined conservative | D2 **plus** running the pass only over *frozen* nodes (expanded, and unable to reach an unexpanded node — a reverse sweep from the unexpanded set) | **0.85–0.89×** |
| **D3** | fully incremental | a confined pass over the accepting closure after every BFS level | **11.3× / 109.7×** |
| **D5** | candidate-confirm | forward DFS on the fully subsumed graph generates candidates (a `¬q` cycle or a stop node); each is confirmed by an exact confined backward pass over the DFS stack | **12.6× / 145.0×** |

Two findings worth stating plainly because they were not obvious in advance:

**D1 requires D2, and that is the whole difficulty of D1.** With the published in-place
enlargement (`node[j].zone = s.zone`), "this vertex will never change" is not establishable
before the forward phase terminates — an enlargement silently invalidates the adjacency the
backward operator just read. Making it establishable means refusing enlargement, and refusing
enlargement is exactly what costs the state count. The user's own suspicion — *"'known not to
change' may be hard to establish before forward termination"* — is confirmed, and the
mechanism is now named: it is the antichain-collapsing enlargement, not the graph shape.

**D4 is a wall-clock idea, not a state-count idea, and it is worth ~0 here** — the
preprocessing it overlaps is three predicate evaluations per node against a successor
computation that costs a delay, a constrain, a reset set, a canonicalisation and an
extrapolation.

---

## 3. Numbers

Baseline is `PTCTLMC_GFPMODE=gfp`, the published three-phase algorithm, `PTCTLMC_GFPALU=1`.
Metric is **symbolic states stored** at the moment the verdict is produced. Geometric mean of
the per-query ratio; `best`/`worst` are the extremes of that ratio.

### 3.1 `queries.txt` — 48 queries, arbiter4/5, fischer4/5, csma4/5, phil4, twolock4, viking4, grc4

```
=== VIOLATED (39 queries) ===          === SATISFIED (9 queries) ===
mode   total    geo-mean  best  worst  mode   total    geo-mean  best  worst
gfp    17676      1.00x                gfp     3415      1.00x
base   17676      1.00x   1.0x  1.0x   base    3415      1.00x   1.0x  1.0x
d1     31156      0.85x   1.3x  0.4x   d1      6345      0.89x   1.0x  0.4x
d2     31222      0.84x   1.0x  0.4x   d2      6345      0.89x   1.0x  0.4x
d3      3475     11.32x 469.0x  1.0x   d3      3415      1.00x   1.0x  1.0x
d4     17676      1.00x   1.0x  1.0x   d4      3415      1.00x   1.0x  1.0x
d5      2926     12.64x 469.0x  1.2x   d5      3128      1.01x   1.1x  1.0x
```

### 3.2 `scale_queries.txt` — 32 queries, fischer6/7/8, arbiter6/8/10, csma6/8, twolock5/6, phil6/8, viking6

```
=== VIOLATED (23 queries) ===          === SATISFIED (6 queries) ===
mode   total    geo-mean   best  worst mode   total    geo-mean  best  worst
gfp   137806      1.00x                gfp    21100      1.00x
d3       937    109.70x 4022.4x  7.5x  d3     21100      1.00x   1.0x  1.0x
d5       517    145.04x 4022.4x  9.1x  d5     23145      0.96x   1.0x  0.8x
```

**The saving grows with the model**, which is the property that matters: 12.6× on the small set,
145× on the larger one. Representative single queries:

| query | gfp | D3 | D5 |
|---|---:|---:|---:|
| `fischer8 P0.idle --> P0.cs` | 36202 | 9 | **9** |
| `fischer8 P0.req --> P0.cs` | 36202 | 165 | **46** |
| `arbiter10 C0.idle --> C0.cs` | 6144 | 11 | **11** |
| `phil8 P0.Think --> P0.Eat` | 8486 | 9 | **9** |
| `grc4 G.up --> G.low` | 2345 | 5 | **5** |
| `csma8 S0.Send --> S0.Wait` | *(no verdict)* | 277 | **82** |

**D5 beats D3 where the witness is deep**, because D3's trigger is "after every BFS level" while
D5's is "a candidate actually appeared": `fischer8 P0.req --> P0.cs` 165 → 46,
`csma8 S0.Send --> S0.Wait` 277 → 82, `fischer5 P0.req --> P0.cs` 56 → 22.

### 3.3 Satisfied queries: no gain, and none was available

D3 is exactly 1.00×; D5 is 0.96–1.01×. This is not a shortfall of the implementation, it is the
shape of the problem: `p --> q` holds only if **every reachable** `p ∧ ¬q` state fails to have
an infinite `¬q` run, so the reachable set has to be enumerated to know the `p` states are all
accounted for. The forward graph is already at 1.00–1.29 zones per reachable discrete state on
six of seven families, so there is nothing left to take at the zone level; a win here has to
skip *discrete* states. That is the open front, and it is what
`liveness-onthefly-candidate.md` attacks.

(The `0.8x`/`1.1x` outliers are search-order effects on the aLU antichain, not lost states:
`grc4 T0.in --> T0.far` stores 2058 zones under D5's DFS order against 2345 under BFS, over the
**same 128 discrete keys**. `grc4` is the one family where subsumption does not collapse to
~1 zone per discrete state — 18.3 there — so its stored count is order-sensitive.)

---

## 4. The result that was not being looked for: robustness

On `csma8` the published algorithm blows the federation-width cap and **has no verdict at all**
on 3 of 3 queries. D3 and D5 settle two of the three exactly, because the certificate lives on a
fragment where the fixpoint is trivial while the full-graph fixpoint explodes:

| `csma8` query | published gfp | D5 | shipping `leads_to_net` |
|---|---|---|---|
| `S0.Wait --> S0.Send` | UNUSABLE (cap, 53 s) | NOT satisfied, **1 state expanded** | NOT satisfied, 1 explored |
| `S0.Send --> S0.Wait` | UNUSABLE (cap, 17 s) | NOT satisfied, **12 states expanded** | NOT satisfied, 12 explored |
| `Bus.Active --> Bus.Idle` | UNUSABLE (cap, 23 s) | UNUSABLE (cap, 6 s) | *(satisfied on csma4/5/6)* |

Both D5 verdicts are confirmed against the shipping engine, and **D5 expands exactly as many
states as the shipping forward engine does** (1 and 12) while remaining the *exact backward*
algorithm — i.e. it reaches the shipping engine's on-the-fly behaviour without inheriting its
need to refuse subsumption edges. Raising the cap does not rescue the published algorithm:
`Bus.Active --> Bus.Idle` with `PTCTLMC_GFPFEDCAP=2000000` does not converge in 10 minutes.

**Federation explosion is the backward operator's real failure mode**, and it is orthogonal to
everything else on this branch. It is the one place where the forward engine is strictly more
robust.

---

## 4b. Against the shipping forward engine

The published gfp is the right baseline for *this branch*, but the tool the user actually has is
`src/netreach.cpp`'s `leads_to_net` — a lazy forward Tarjan with eager escape, which is already
on-the-fly. The question that decides whether D5 is worth shipping is whether it dominates
**both**. Like-for-like metric: D5's `expanded` (successor computations performed) against
`leads_to_net`'s `stats.explored`. `run_otf.py --ref`, 48 queries.

**Verdict agreement: 47/48; the 48th is a timeout of the shipping engine, not a disagreement.**

**Violated queries (39).** D5 never expands more, and expands up to 5.5× fewer:

```
shipping-explored / D5-expanded :  geo-mean 1.18x   min 1.00x   max 5.52x
```

31 of 39 are dead-on 1.00× — both engines stop at the same state, which is what should happen
when two correct on-the-fly searches find the same shallow witness. The gaps are where the
witness needs a cycle: `grc4 T0.near --> T0.in` 9505 → **1722**, `fischer5 P0.cs --> P0.idle`
116 → **56**, `twolock4 P0.waitB --> P0.cs` 125 → **64**. That is the subsumption asymmetry
paying off: the shipping engine has to refuse subsumption edges when arguing that a run goes on
forever (`exact_edge`), and walks further because of it; D5 follows every edge and lets the
backward pass arbitrate.

**Satisfied queries (9).** This is where the backward gfp was already known to win, and D5
inherits it intact:

| query | D5 expanded | shipping explored |
|---|---:|---:|
| `arbiter5 C1.cs --> C1.idle` | **112** | 136204 |
| `arbiter4 C1.cs --> C1.idle` | **48** | 8192 |
| `fischer4 P0.req --> P0.wait` | **270** | 1468 |
| `grc4 T0.in --> T0.far` | **2681** (3.85 s) | *timeout at 240 s* |
| `csma4 Bus.Active --> Bus.Idle` | **128** | 3341 |
| `csma5 Bus.Active --> Bus.Idle` | **352** | 60271 |

Caveat: `leads_to_net` settles a satisfied query by *racing* an aLU search against an
exact-inclusion search and a deadlock gate; `stats.explored` sums the two searches and omits
the gate, so it is indicative rather than a clean count. The direction is not in doubt, though
— the shipping engine keeps node identity at **exact zone equality** by default, which on these
queries is 32 zones per discrete state on `csma4`, 222 on `csma5` and 5514 on `grc4`, against
the gfp's 1.23/1.35/18.32.

> **Retracted.** An earlier revision of this section reported `csma4`/`csma5` as a *loss*
> for D5 (128 vs 42, 352 vs 62) and recommended keeping D5 merely as one lane of a race. Those
> two shipping-engine numbers were measured against a **truncated** search — see §4c. With the
> truncation repaired they are 3341 and 60271, i.e. D5 wins by 26× and 171×, and no satisfied
> query in the set is a loss. The lane recommendation stands on the *general* grounds that
> neither strategy is predictable, not on the csma evidence, which was an artefact.

---

## 4c. A truncation bug in the shipping engine, found by this comparison

`leads_to_net`'s outer BFS skips a frontier node that is already expanded:

```cpp
{ std::lock_guard<std::mutex> lk(gmtx); cur = node[u]; if (expanded[u]) continue; }
```

`continue` skips the node's **successor publication**, not just its expansion — and the inner
`neg_scc` Tarjan calls `expand()` on whole regions before the outer BFS gets to them. Those
regions' successors therefore never enter the outer frontier and the reachable set is
truncated. Measured against `benchmarks/parity.csv`, whose discrete counts are verified against
UPPAAL:

| model | reachable discrete states | `leads_to_net` reaches | repaired |
|---|---:|---:|---:|
| `csma4` | 104 | **21** | 104 |
| `fischer4` | 220 | **217** | 220 |
| `arbiter4` | 48 | 48 | 48 |

On `csma4 Bus.Active --> Bus.Idle` the engine returns SATISFIED having examined **21 of 104**
reachable discrete states. This is **soundness-relevant, not merely a performance artefact**: a
`p ∧ ¬q` state may be reachable *only through `q` states*, which `neg_scc` never enters, so the
outer BFS is the only thing that can find it. No wrong verdict is exhibited on the current
benchmark set — every query still agrees — but the mechanism can miss a violation.

This is the *same bug class* as the one D5 hit on day one of this branch (§5.1), which is how
it was noticed: D5's outer search was truncating for exactly the same reason, and checking
whether the shipping engine did the same turned out to be the most valuable thing in this
comparison.

The repair is in the prototype copy only (`PTCTLMC_LTNOFIXBFS` restores the old behaviour for
A/B); **`src/netreach.cpp` is still unpatched** and should be fixed separately, with its own
differential run, because it is the shipping path.

---

## 5. Differential validation

* `queries.txt`, 7 modes × 48 queries: **0 disagreements**, 0 capped.
* `scale_queries.txt`, 3 modes × 32 queries: **0 disagreements** among converged runs; 3 capped
  for `gfp`, 1 for D3/D5 (all `csma8`).
* Spot-checked against the shipping `leads_to_net` on both `csma8` certificates.
* `PTCTLMC_GFPMODE=base` reproduces the published node count on **every** query in both sets
  (17676 / 3415 / 137806 / 21100 totals identical), which is what licenses reading the D1–D5
  columns as differences of algorithm rather than of reimplementation.

Two bugs were found and fixed by this exercise, both in code written today:

1. **D5's outer BFS skipped nodes the inner DFS had already expanded**, and skipping them also
   skipped *publishing their successors*, silently truncating the reachable set. It produced
   plausible-looking SATISFIED verdicts on fewer states (`csma5 Bus.Active --> Bus.Idle` at 41
   states instead of 352). A `p ∧ ¬q` state may only be reachable *through* `q` states, which
   the inner DFS never enters, so the outer search must cover everything.
2. **A capped pass was being read as "no certificate"**, i.e. as SATISFIED. Now `UNUSABLE`.

One pre-existing documentation error was found: `leads_to_gfp` claims the active-clock
reduction is disabled and that `qmask` is all-ones. `qmask` is all-**zeros**, and
`reset_inactive` reads it as *additionally keep alive*, so the reduction runs in full. What
makes it safe backward is the repair inside `gfp_pre`, which frees every clock dead at the
target. The `.tex` matches the code; only the comment is wrong. Corrected in the new code and
noted here rather than edited into the published function.

---

## 6. Two changes that are orthogonal wins

**Accepting-closure restriction.** `F` is defined per node from its own successors, so the value
at an accepting node depends only on the `¬q`-forward-closure of the accepting set. Restricting
the pass to that closure is **exact, not an approximation**, and it is a pure time win:
`grc4 T0.in --> T0.far` 5.87 s → 2.03 s, `grc4 G.up --> G.low` 0.13 s → 0.02 s. It applies to
the published algorithm unchanged.

**Refutation budget (D5).** A confirmation costs one confined pass over the DFS stack. On every
violated query in the set the certificate is confirmed on the **first** candidate (`passes=1`,
1–3 `Pre` calls — including `grc4 T0.near --> T0.in`, where the DFS first walks 1722 states).
Only queries that *hold* refute repeatedly. Spending a fixed budget (default 8) and then giving
up on early exit cannot change a verdict — it only defers the answer to the exact final pass —
and it cut `grc4 T0.in --> T0.far` from 9.31 s / 22 passes to 3.43 s / 9 passes.

---

## 7. Recommendation

Ship **D5** (`PTCTLMC_GFPMODE=d5`) as the liveness path, with the accepting-closure restriction
and the refutation budget on by default. It is the published algorithm plus an early exit that
is free when the property holds, and it is the only variant that also rescues capped queries.
Keep it as a **lane** in the race `leads_to_net` already runs rather than as a replacement —
not because any satisfied query in the set favours the forward engine (none does, once §4c's
truncation is repaired), but because the forward engine settles 31 of 39 violated queries at
exactly the same state and is far cheaper per state, while D5 is the only thing that finishes
`grc4 T0.in --> T0.far` at all. Which lane wins is not predictable, which is the same argument
the existing aLU-vs-exact race already rests on.

**Fix §4c in `src/netreach.cpp` first.** It is a soundness-relevant truncation on the shipping
path and it is independent of everything else here.

Do **not** pursue D1, D2 or D4 further: D1/D2 pay a measured 11–16 % average and up to 2.5 %
worst-case state penalty for a pipelining benefit that D3/D5 obtain without it, and D4 is
provably a no-op on the metric.

The remaining open front is satisfied queries, where the reachable state space is a genuine
floor for `p --> q` and only a *discrete*-state reduction can move it. See
`liveness-onthefly-candidate.md`, and note its own conclusion that query-stabiliser symmetry
reduction is a comparable or larger lever and is already half-implemented in
`src/symmetry.cpp`.

---

## Reproducing

```bash
cd nove_mc/prototype
mkdir -p src && for f in ../../src/*.cpp; do b=$(basename $f); [ "$b" = netreach.cpp ] && continue; ln -sf ../../../src/$b src/$b; done
ln -sf ../netreach_gfp.cpp src/netreach_gfp.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
python3 run_otf.py --modes gfp,base,d1,d2,d3,d4,d5 --timeout 200
python3 run_otf.py --modes gfp,d3,d5 --queries scale_queries.txt --timeout 200 --out otf_scale.json
python3 summarise_otf.py otf_results.json
```

| env var | effect |
|---|---|
| `PTCTLMC_GFPMODE` | `gfp` (published) / `base` / `d1`…`d5` |
| `PTCTLMC_GFPALU` | aLU subsumption in the forward phase — required to be at the floor |
| `PTCTLMC_GFPFEDCAP` | federation-width cap, default 4096; a run that hits it prints `UNUSABLE` |
| `PTCTLMC_OTFREFUTE` | D5 refutation budget, default 8; `0` disables early confirmation entirely |


---

## 8. Why does verifyta explore fewer states on a violated query?

It is a fair objection: on a violated query both tools stop at the first witness, so neither
should be able to "finish early". Measured on `grc4 T0.near --> T0.in`, the 860-vs-1768 gap
decomposes into two independent factors, and neither is an algorithmic advantage in the
liveness procedure.

**1. Search-order luck, worth about 1.4x.** verifyta's own numbers swing wildly with `-o`:

| verifyta search order | states stored | states explored |
|---|---:|---:|
| `-o 0` breadth-first (**default**) | **860** | **463** |
| `-o 1` depth-first | 1433 | 6559 |
| `-o 2` | 1433 | 3068 |

Its default order happens to walk into the witness early on this query. Our D5 stores 1768 and
expands 1535 — within 1.23x of verifyta's *own* worst stored figure, and **3–4x better than
verifyta on states explored** under its non-default orders. The same experiment on our side
(`PTCTLMC_OTFORDER=dfs`) moves us 1768 → 1466 stored / 1535 → 1956 expanded. Both tools are
order-sensitive on this query; verifyta's default is the lucky one.

**2. Zone representation, worth about 1.45x, and query-independent.** For the *whole* grc4
state space verifyta stores 1619 (`A[] true`) where our liveness graph stores 2345. That factor
is present before any query is asked, and it is the same 14.5-zones-per-discrete-key phenomenon
of §10.6. It is the only part of the gap that is really ours to fix.

`1.45 x 1.42 ~ 2.06`, which is the observed `1768 / 860`.

So the honest reading is: there is no early-stopping advantage to explain. There is a modest
representation deficit on one model, multiplied by a lucky default search order on one query.
Every other violated query in the set is already at or below verifyta (32 of 39 win or tie;
355 against 418 in total excluding this one).
