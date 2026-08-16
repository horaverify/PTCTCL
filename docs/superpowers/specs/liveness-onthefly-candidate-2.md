# LADDER: one merged zone graph, two lanes, and a federation algebra that does not explode

Branch `liveness-on-the-fly`. Target: `p --> q` (violated iff `EF(p ∧ EG ¬q)`) and `E[] φ`, with
`p`, `q` **discrete** predicates only. Baseline: `nove_mc/prototype/netreach_gfp.cpp` —
`leads_to_gfp` (published three-phase), class `Otf` modes D5/D6/D7, and `src/netreach.cpp`'s
`leads_to_net`.

This document attacks **both** open problems of `liveness-onthefly-results.md` §4 and the
`liveness-onthefly-candidate.md` follow-up brief:

* **(A)** an abstraction strictly between aLU and the convex hull;
* **(B)** the federation explosion of the exact backward operator.

Everything numbered below was **measured** in a scratch build of this prototype
(`PTCTLMC_X*` env switches, §7.1) unless marked `[argued]` or `[predicted]`.

---

## 0. Verdict first

Three findings, in decreasing order of how much they matter.

1. **Problem (B) is not a theory problem. It is a two-line representation bug.** The backward
   operator computes `z ∩ cand` as `fed_subtract(z, fed_subtract(z, cand))`. That is the right
   *set* and a catastrophic *representation*: double subtraction builds the cell arrangement of
   the complement, whose width is exponential in `|cand|`, where `fed_intersect(W, cand)` gives
   the same set in `≤ |W|·|cand|` convex pieces. Independently, `gfp_fed_compact` is a strictly
   weaker compaction than the library's own `fed_reduce` — it never tries the exact convex union
   and never tests joint coverage (and builds a `rest` federation it then discards).
   **Measured, `csma8 Bus.Active --> Bus.Idle`** — the query neither the published algorithm nor
   D5 can answer, and which does not converge at `FEDCAP=2e6` in ten minutes:

   | config | verdict | max federation width | time |
   |---|---|---:|---:|
   | D5 as shipped | UNUSABLE (cap) | 4097 | 6.00 s |
   | D5 + `fed_reduce` | **SATISFIED** | 16 | 0.57 s |
   | D5 + `fed_intersect` | **SATISFIED** | 4 | 0.21 s |
   | D5 + both | **SATISFIED** | **1** | **0.20 s** |

   Either half alone rescues it. On `grc4 T0.in --> T0.far` the same two changes take D5 from
   3.61 s / 227042 `gfp_pre` calls / width 596 to **0.07 s / 17940 / width 5** (48×), and the
   published `leads_to_gfp` from 5.77 s. **The exact backward operator converges here after all.**

2. **Problem (A)'s premise — "D6 is slow because the hull is too coarse for fischer" — is
   false.** D6 answers `fischer7 P0.req --> P0.wait` correctly; it was slow for two mechanical
   reasons, neither of which is the hull. With the algebra fixed it goes 35.5 s → 17.6 s; with
   the *forward closure* changed from a level-synchronous BFS to a deduplicated worklist (the
   hull's in-place enlargement resets `expanded`, and the BFS then re-publishes whole successor
   lists every level, accumulating a quadratic frontier — 99 % of the 17.6 s was in that loop, by
   `sample`) it goes to **0.20 s**. That is **178× off the number in the brief**, and it is
   4.0× *faster* than D5 while storing 1.50× fewer states.

3. **The abstraction that (A) asks for is not "hull with a width bound".** It is a **merge
   ladder** with three rungs of strictly increasing loss — aLU subsumption, then *exact* convex
   union (`dbm_try_union`, which adds no point and is therefore sound in **both** directions),
   then a bounded lossy hull that marks the node `INFLATED` — carried on **one** node set that
   serves **both** lanes, with the two lanes distinguished only by *which disjuncts of the
   backward operator they are allowed to use*. Measured over **80 queries** (`queries.txt` 48 +
   `scale_queries.txt` 32): at width ≥ 2, **0 disagreements and 0 abstentions**; at width 1 (=
   pure hull) one abstention. Against D7 the saving is that D7 pays for **two** graphs (D5's plus
   D6's) and LADDER pays for one:

   | satisfied query | published gfp | D5 | D6 | D7 total = D5+D6 | **LADDER w=2** |
   |---|---:|---:|---:|---:|---:|
   | `grc4 T0.in --> T0.far` | 2345 | 2058 | 128 | 2186 | **227** |
   | `fischer7 P0.req --> P0.wait` | 9798 | 11598 | 7737 | 19335 | **9236** |
   | `csma8 Bus.Active --> Bus.Idle` | *(no verdict)* | 5888 | 4096 | 9984 | **4989** |
   | `csma6 Bus.Active --> Bus.Idle` | 928 | 928 | — | — | **778** |

**What is not claimed.** LADDER does not remove *discrete* states, so it cannot beat D6/D7 on the
families where aLU is already at 1.00 zones per discrete key (arbiter, phil, viking, twolock —
measured identical node counts, and a 19 % *time* loss against the published algorithm on
`arbiter10`). The state-level open front of `liveness-onthefly-results.md` §3.3 stays open; what
this document closes is the *robustness* and *time* front, and it does so with a measured 289×
on the one query the shipping engine cannot finish at all.

---

## 1. The algorithm

### 1.1 Primitives reused unchanged

| primitive | file | role here |
|---|---|---|
| `gfp_succ_tagged` | `netreach_gfp.cpp:3539` | successors + the participating-edge tag |
| `gfp_pre` | `netreach_gfp.cpp:3718` | exact concrete `Pre` of one tagged transition |
| `state_has_deadlock` | `netreach_gfp.cpp:3323` | exact partial-deadlock test (rebuilds delay closure) |
| `Otf::time_diverges` | `netreach_gfp.cpp:5346` | "can stay put forever" |
| `Dbm::extrapolate_lu_plus` / `_nodiag` | `HDBM/dbm.hpp` | `Extra_{LU}^+`, A1/A2 |
| `dbm_is_alu_le` (`Dbm::alu_subsumed_by`) | `HDBM/dbm_ops.hpp:366` | rung 1 |
| `dbm_try_union` (`Dbm::try_union`) | `HDBM/dbm_ops.hpp:585` | rung 2 — merges **only when the union is convex** |
| `dbm_hull` | `HDBM/dbm_ops.hpp:523` | rung 3 |
| `fed_subtract`, `fed_intersect`, `fed_reduce` | `HDBM/fed.hpp:58,116,173` | federation algebra |

Two are used **instead of** what the current code uses, and that is change (C3):

* `fed_intersect(W, cand)` replaces `fed_subtract(z, fed_subtract(z, cand))`;
* `fed_reduce(f)` replaces `gfp_fed_compact(f, cap, &capped)`.

### 1.2 Notation

For a node `v` of the stored graph let `Z_v` be its zone, `disc(v)` its discrete key
(`locs`+`ints`), `adj(v)` its tagged out-edges. `Reach(v)` is the set of clock valuations `ν`
with `(disc(v), ν)` reachable, and

```
Ω(v) = { ν ∈ Reach(v) : some maximal run from (disc(v),ν) stays in ¬q forever }
Ω̂(v) = { ν ∈ Z_v      : some maximal run from (disc(v),ν) stays in ¬q forever }
```

`p --> q` is **violated** iff `Ω(v) ≠ ∅` for some `v ⊨ p ∧ ¬q`. `E[] φ` is the same fixpoint with
`p = φ`, `q = ¬φ`, read at the initial node only (`PTCTLMC_GFPINITONLY`).

The backward operator, per node, over a vector `W` of federations:

```
stopset(v)  =  Z_v          if  adj(v)=∅  ∨  time_diverges(Z_v)  ∨  state_has_deadlock(v)
               ∅            otherwise                                        (whole-zone, §5.3)

F(W)_v      =  ∅                                                   if v ⊨ q
               Z_v ∩ ( stopset(v) ∪ ⋃_{e∈adj(v)} Pre_e(W_{tgt(e)}) )  if v ⊨ ¬q
```

`F_S` is `F` with `W_j` pinned to `∅` for every `j ∉ S`; `F_S ≤ F` pointwise, so `νF_S ⊆ νF`.

### 1.3 (C1) The merge ladder — `intern`

Merging happens **only inside one discrete-key bucket**. That is not a convenience: the
active-clock set, the invariants and the enabled edge set are functions of the discrete state
alone, so every zone in a bucket has the same live-clock mask and the same `reset_inactive`
behaviour. Cross-key merging would break `gfp_pre` (§5.1); this never does.

```
intern(s):                                        # s is a fresh successor SymState
  b ← bucket[disc_key(s)]
  # rung 1 — aLU subsumption. Exact for BOTH verdicts (04-soundness, Cor. alu).
  for j in b: if s.zone ⊑_aLU Z_j:  return j
  for j in b: if Z_j ⊑_aLU s.zone:  Z_j ← s.zone ; grew ← 1 ; return j
  # rung 2 — EXACT convex union. Adds no point, so still exact for BOTH verdicts.
  for j in b:
      h ← Z_j ; if dbm_try_union(h, s.zone):  Z_j ← h ; grew ← 1 ; return j
  # width bound
  if |b| < k[disc_key(s)]:  b ← b ∪ {new node with zone s.zone} ; return it
  # rung 3 — LOSSY hull into the partner that relaxes the fewest DBM entries.
  j* ← argmin_{j∈b} |{ (a,c) : dbm_hull(Z_j, s.zone)[a,c] ≠ Z_j[a,c] }|
  Z_{j*} ← dbm_hull(Z_{j*}, s.zone) ; grew ← 1 ; INFLATED[j*] ← 1 ; return j*
```

`k[·]` is a **per-key** width bound, default `k₀ = 2`, raised only by refinement (§1.7).
`INFLATED` is sticky and **never propagates**: §2.4 proves that inflation of a node only
disqualifies two *local* uses of it, not everything downstream. That is the whole reason the
scheme is affordable.

`grew` triggers the existing invalidation in `Otf::expand` (`netreach_gfp.cpp:5316`):
`expanded[j] ← 0`, `adj[j]` cleared, `pred` fixed up.

### 1.4 (C5) The forward closure is a worklist, not a level BFS

```
build():
  wl ← [root] ; inq[root] ← 1
  while wl ≠ ∅:
     u ← pop(wl) ; inq[u] ← 0
     expand(u)                                   # no-op if already expanded
     if ¬expanded[u] ∧ ¬inq[u]:  push(u)         # u invalidated itself (self-loop hull growth)
     for e in adj(u):
        if ¬expanded[e.tgt] ∧ ¬inq[e.tgt]:  inq[e.tgt] ← 1 ; push(e.tgt)
```

This is ordinary chaotic ascending-chain iteration. It is **required** as soon as `intern`
merges: with a merging `intern`, `expanded` is reset from inside `expand`, and the shipped
level-synchronous loop re-publishes a node's whole successor list on every subsequent level,
accumulating duplicates in `next`. Measured on `fischer7 P0.req --> P0.wait` with D6: **17.60 s →
0.20 s**, with `sample` attributing 99 % of the 17.6 s to that loop and to `Otf::grow`.

### 1.5 (C2) One graph, two lanes

Both lanes run the *same* operator on the *same* node set. They differ in exactly two places.

```
                      over-lane  (proves HOLDS)         under-lane (proves VIOLATED)
S                     expanded ∧ ¬q ∧ accepting-closure  expanded ∧ ¬q  (inflated ALLOWED)
stopset(v) used?      yes, for every v                   only if ¬INFLATED[v]
verdict read at       every accepting v; "all ∅" ⇒ HOLDS  accepting v with ¬INFLATED[v]; "≠∅" ⇒ VIOLATED
compaction            fed_reduce, then κ_k (hull-widen)   fed_reduce, then ν_k (drop members)
```

`κ_k` and `ν_k` are the width bounds of (C4):

```
κ_k(X):   fed_reduce(X)                     # exact: inclusion-minimal + convex unions
          while |X| > k:  replace the pair (a,b) minimising |{entries dbm_hull(a,b) relaxes}|
                          by dbm_hull(a,b)                       # ENLARGES  → over-lane only
ν_k(X):   fed_reduce(X)
          while |X| > k:  drop the member with the fewest tight facets   # SHRINKS → under-lane only
```

Both are inert on this benchmark set once (C3) is in — measured maximum width after `fed_reduce`
is **1–26** across all 80 queries — so they are a *guarantee*, not a working part. Their point is
that neither lane can ever report `UNUSABLE`.

### 1.6 The driver

```
LADDER(net, p, q):
  build()                                        # §1.4, merge ladder §1.3

  # ---- under-lane, on the fly: D5's candidate generator, restricted -------------
  # identical to Otf::run_dfs_candidates (netreach_gfp.cpp:5589) except that
  #   * a "kind 1" (stop) candidate is only raised at a node with ¬INFLATED,
  #   * confirm() builds S from the DFS stack and confirms with gfp_under (below).
  # Interleaved with build(); a confirmed candidate returns VIOLATED immediately.

  # ---- final passes, if no candidate was confirmed ------------------------------
  S ← accepting_closure()                        # netreach_gfp.cpp:5716, exact restriction
  W ← gfp_over(S)
  if ∀ v ∈ S with v ⊨ p:  W_v = ∅ :  return HOLDS
  V ← gfp_under(S)
  if ∃ v ∈ S with v ⊨ p ∧ ¬INFLATED[v] ∧ V_v ≠ ∅ :  return VIOLATED
  return REFINE                                  # §1.7
```

with

```
gfp_over(S):                                     gfp_under(S):
  W_v ← Z_v for v ∈ S, ∅ otherwise                 V_v ← Z_v for v ∈ S, ∅ otherwise
  repeat until no v changes:                       repeat until no v changes:
    for v ∈ S:                                       for v ∈ S:
      cand ← stopset(v)                                cand ← INFLATED[v] ? ∅ : stopset(v)
      for e ∈ adj(v): for Y ∈ W_{tgt(e)}:              for e ∈ adj(v): for Y ∈ V_{tgt(e)}:
          cand ← cand ∪ { gfp_pre(e,v,tgt,Y) }             cand ← cand ∪ { gfp_pre(e,v,tgt,Y) }
      cand ← fed_reduce(cand)                          cand ← fed_reduce(cand)
      new  ← κ_k( fed_intersect(W_v, cand) )           new  ← ν_k( fed_intersect(V_v, cand) )
      changed ← ¬fed_included_in(W_v, new)             changed ← ¬fed_included_in(V_v, new)
      W_v ← new                                        V_v ← new
```

The two bodies are one function with two flags. `fed_intersect` and `fed_reduce` are (C3).

### 1.7 (C6) The refinement criterion, precisely

`REFINE` fires exactly when the over-lane residue is non-empty at some accepting node **and** the
under-lane found nothing at any exact accepting node. Define

```
R      = { v : W_v ≠ ∅ }                                   # over-lane residue
Blame  = { v ∈ R : INFLATED[v] ∧ v is ¬q-reachable from some accepting r with V_r = ∅ }
```

`Blame` is computed by one forward sweep over `¬q` edges from the accepting nodes with an empty
under-value, intersected with `R ∩ INFLATED`. **Refinement step:** for every
`κ ∈ { disc(v) : v ∈ Blame }` set `k[κ] ← 2·k[κ]`, discard the graph, and rebuild.

* **Progress.** Every rebuild strictly increases `k[κ]` for at least one key that carried a lossy
  hull, so after at most `⌈log₂ A_κ⌉` refinements of key `κ` (with `A_κ` its aLU antichain width)
  rung 3 never fires there again and every node of `κ` is exact. When no key is ever hulled the
  graph *is* D5's aLU graph and LADDER is exactly D5 + D6 on the same nodes: complete relative to
  the published algorithm.
* **Cost.** Node counts across the refinement sequence are non-decreasing and bounded by D5's;
  since `k` doubles, the total work over all rounds is at most `2×` the last round's, i.e. at
  most `2×` D5. `[argued]`
* **Measured frequency.** `REFINE` fires **once** in 80 queries, and only at `k₀ = 1`
  (`grc4 T0.near --> T0.in`). At `k₀ = 2` it never fires. So the loop is a completeness backstop,
  not a working part.

---

## 2. Soundness and completeness

At the rigour of `nove_mc/src/04-soundness.tex`, whose numbering is reused where a result is
imported unchanged.

### 2.1 Assumptions, all of them

* **(A1) Extensivity.** `Extra_LU⁺` is extensive: `Z ⊆ Extra(Z)`. Used only where an
  over-approximation is safe.
* **(A2) Simulation.** `Extra_LU⁺` is an `≼_LU`-simulation abstraction: every `ν ∈ Extra(Z)` is
  `≼_LU`-simulated **by** some reachable `ν' ∈ Z`. Used in the direction of
  Theorem `asym`(i) only.
* **(A3) Delay closure.** A stored zone is delay-closed *as a set* after `gfp_succ_tagged`
  (delay is applied after the action unless the target is committed/urgent). `gfp_pre` therefore
  re-applies `down()` at the source — the "delay BEFORE the action" (§5.2).
* **(A4) Active-clock reduction.** `reset_inactive` runs in full (`qmask` all-zero; discrete
  predicates constrain no clock). Its inverse lives in `gfp_pre`, which **frees** every clock
  dead at the *target* before undoing the step (§5.1). Merging never crosses discrete keys, so
  every zone in a bucket carries the same reduction.
* **(A5) Partial deadlock.** `stopset` is a whole-zone term: if `state_has_deadlock(v)` holds for
  *any* valuation of `Z_v`, the *whole* `Z_v` enters the stop set. This is the shipping engine's
  and `leads_to_gfp`'s convention and it is inherited verbatim (§5.3).
* **(A6) Zeno.** The *cycle* witness is not divergence-checked, in this algorithm, in
  `leads_to_gfp`, in `leads_to_net` and in `verifyta` alike (`04-soundness` §Zeno, `zeno.xml`).
  Unchanged here. `[open]`
* **(A7) Discrete predicates.** `p`, `q` are functions of `locs`/`ints` only
  (`04-soundness` Lemma `disc-pred`). Everything below fails if `q` constrains clocks.

### 2.2 The four new lemmas

**Lemma L1 (merging changes zones, never transitions).**
*For every node `v` and every `e ∈ adj(v)`, `e` is a real transition of the network out of
`disc(v)`, and `gfp_pre(e, v, tgt(e), X)` is exactly `{ν : ν --e--> ν' for some ν' ∈ X}`.*

*Proof.* `gfp_succ_tagged` emits an edge tagged with the participating `(process, edge)` pairs;
guards, resets, updates and invariants are functions of `disc(v)` alone, so the tag identifies a
transition of the network independently of which zone sits at `v`. `gfp_pre` inverts exactly that
transition and applies no abstraction (no `Extra`, no `reset_inactive` forward). A merged zone
changes only *which* edges are emitted (`post_e` is applied to a larger set, so an edge may be
emitted that no part alone would emit) and never *what* an emitted edge means. ∎

**Lemma L2 (rungs 1 and 2 preserve genuineness).**
*Define `EXACT(v)`: every `ν ∈ Z_v` is `≼_LU`-simulated by some `ν' ∈ Reach(v)`. Rung 1 and rung 2
preserve `EXACT`; rung 3 does not, and sets `INFLATED`.*

*Proof.* Rung 1 is `04-soundness` Cor. `alu` verbatim: a dropped newcomer is simulated by the
stored zone, and an enlargement replaces the stored zone by a newcomer that is itself a reachable
symbolic state. Rung 2 sets `Z_j ← Z_j ∪ Z_new` **exactly** — `dbm_try_union` merges only when
the union is already a zone, so no point is added; every point of the result lies in `Z_j` or in
`Z_new`, each of which satisfies `EXACT` by induction. Rung 3 takes `dbm_hull`, which is the
smallest zone containing the union and in general strictly larger, so the property is dropped and
recorded. ∎

**Lemma L3 (`stopset` is monotone increasing in the zone).**
*If `Z ⊆ Z'` at the same discrete state then `time_diverges(Z) ⇒ time_diverges(Z')` and
`state_has_deadlock(Z) ⇒ state_has_deadlock(Z')`, while `adj = ∅` is monotone decreasing.*

*Proof.* `time_diverges` delays, re-applies invariants, and requires every `Z(i,0)` to be `kInf`;
delay and invariant application are monotone and the entry-wise upper bound of a superset is
weaker, so the test can only turn from false to true. `state_has_deadlock` returns
`¬(Z \ live)=∅` where `live` depends only on the discrete state; a larger `Z` can only enlarge
the residue. `adj(v)` is `post`-generated, and `post` is monotone, so a larger zone has at least
as many edges. ∎

**Lemma L4 (free narrowing).**
*Let `ν_v` be any maps with `ν_v(X) ⊆ X`. The chaotic iteration*
`V_v ← ν_v( V_v ∩ cand_v(V) )` *started at `V_v = Z_v` (`v ∈ S`), `∅` otherwise, terminates (§3)
at a vector `V` with `V ⊆ F_S(V)`.*

*Proof.* The loop exits only after a sweep in which no `v` changed, i.e. for every `v ∈ S`,
`V_v ⊆ ν_v(V_v ∩ cand_v(V))`. Since `ν_v` shrinks, `ν_v(V_v ∩ cand_v(V)) ⊆ V_v ∩ cand_v(V)`, so
`V_v = ν_v(V_v ∩ cand_v(V))` and in particular `V_v ⊆ cand_v(V)`. `V_v ⊆ Z_v` holds throughout
(initial, and every update intersects with `V_v`). Hence
`V_v ⊆ Z_v ∩ cand_v(V) = F_S(V)_v`. ∎

`L4` is the whole content of (C4) in the under direction: **any** shrinking — dropping federation
members, keeping one convex zone, keeping only the piece that contains a chosen anchor — is free,
and it needs **no separate verification pass**, because the post-fixpoint inequality is exactly
the local condition the update already enforces.

The dual, used by the over-lane, is the ordinary invariant argument and is weaker in form because
`κ_k` is not monotone as a function:

**Lemma L4′ (free widening under an invariant).**
*Let `κ_v` be any maps with `κ_v(X) ⊇ X`. The iteration `W_v ← W_v ∩ κ_v(cand_v(W))` started at
`W_v = Z_v` maintains `Ω̂_v ⊆ W_v` at every step, and `W` is non-increasing.*

*Proof.* Non-increasing is immediate from the outer `W_v ∩ ·`. For the invariant: `Ω̂` is a
post-fixpoint of `F` (`04-soundness` Thm `sound-holds`, whose proof needs only that a maximal run
either continues by a real transition — L1 — or ends, in which case `stopset` fires by A5). If
`Ω̂ ⊆ W` then by monotonicity of `F`, `Ω̂ ⊆ F(Ω̂) ⊆ F(W)`, so
`Ω̂_v ⊆ W_v ∩ F(W)_v = W_v ∩ Z_v ∩ cand_v(W) = W_v ∩ cand_v(W) ⊆ W_v ∩ κ_v(cand_v(W))`. ∎

Note the consequence: the over-lane may be read **at any moment**, converged or not. `W_v = ∅` at
every accepting `v` proves `HOLDS` immediately.

### 2.3 The over-lane concludes HOLDS

**Theorem 1.** *If `gfp_over(S)` reaches `W_v = ∅` for every `v ∈ S` with `v ⊨ p`, and `S` is the
accepting closure of the **completed** forward graph, then `p --> q` holds.*

*Proof.* By L4′, `Ω̂_v ⊆ W_v = ∅` for every accepting `v`. `Ω(v) ⊆ Ω̂(v)` because
`Reach(v) ⊆ Z_v` up to `≼_LU` (A1/A2 + rung-1/2 by L2; rung 3 only enlarges `Z_v`, which
strengthens the containment). By `04-soundness` Lemma `disc-pred` a matched run visits the same
discrete states, so `Ω(v) = ∅` for every accepting `v`. Hence no reachable `p ∧ ¬q` state has an
infinite `¬q` run. ∎

The over-lane therefore tolerates **every** rung of the ladder, including rung 3, and this is why
LADDER's *satisfied*-query state count is D6's, not D5's.

**Completeness of the over-lane.** It is complete relative to `Ω̂` on the merged graph: it
concludes `HOLDS` iff `νF(v) = ∅` at every accepting `v`. It abstains exactly when the merged
zones carry a spurious infinite `¬q` run. Measured: **0 abstentions in 80 queries at `k₀ = 2`,
1 at `k₀ = 1`** — and the one abstention was on a query that is genuinely VIOLATED, where the
over-lane is *supposed* to abstain and the under-lane is supposed to speak.

### 2.4 The under-lane concludes VIOLATED — and why inflation does not propagate

**Theorem 2.** *Let `V` be the stabilised vector of `gfp_under(S)`. If `V_r ≠ ∅` for some
`r ∈ S` with `r ⊨ p ∧ ¬q` and `¬INFLATED[r]`, then `p --> q` is violated.*

*Proof.* By L4, `V ⊆ F'_S(V)` where `F'` is `F` with the `stopset` disjunct deleted at every
inflated node. `F' ≤ F`, so `V ⊆ F(V)`, and unwinding `V_r ⊆ stopset(r) ∪ ⋃_e Pre_e(V_{tgt(e)})`
co-inductively gives, for each `ν ∈ V_r`, a **maximal run** in the concrete semantics:

* if `ν ∈ Pre_e(V_j)`, then by **L1** `e` is a real transition and there is a real `ν'` with
  `ν --e--> ν'` and `ν' ∈ V_j`; recurse at `j`;
* if `ν ∈ stopset(v)` at some `v` on the way, then `¬INFLATED[v]` (the disjunct was deleted
  otherwise), so by **L2** `Z_v` satisfies `EXACT` and the whole-zone stop test is being applied
  to a zone of genuinely-simulated valuations — precisely the situation `04-soundness`
  Cor. `whole` licenses. Modulo (A5) the run ends there.

Every node visited on this run stays in `¬q` because `S ⊆ ¬q` and `F` pins `q` nodes to `∅`.
So `ν` has a maximal `¬q` run.

Finally, `¬INFLATED[r]` and L2 give `EXACT(r)`: `ν` is `≼_LU`-simulated by a **reachable**
`ν' ∈ Reach(r)`, and `04-soundness` Thm `asym`(i) transfers the infinite `¬q` run to `ν'`. Since
`r ⊨ p ∧ ¬q` and `ν'` is reachable, `Ω(r) ≠ ∅`. ∎

**The load-bearing corollary.** The proof consumes inflation at exactly two points: (i) the
`stopset` disjunct, and (ii) the *root*. Nowhere does it need `Z_j` to be exact for an
intermediate `j`, because the witnessing run is *generated forward from `ν`* by real transitions
— the intermediate valuations are reachable **by construction**, whatever their node's zone
contains. Hence:

> **Inflation is a local disqualifier, not a contaminant.** It does not need to be forward-closed
> over the graph.

This is the difference between a usable design and an unusable one. Forward-closing inflation on
`grc4` at `k=1` would mark essentially all 128 nodes; the local rule marks 85 and disqualifies
only their `stopset` disjunct and their use as a root.

**Where L3 bites.** L3 is why the `stopset` disjunct *must* be deleted at inflated nodes:
`time_diverges` and `state_has_deadlock` are monotone **increasing** in the zone, so a hull can
manufacture a divergence or a deadlock that no merged part has. This trap is new to the ladder —
neither D5 nor D6 exhibits it (D5 has no hulls; D6 is over-approximating everywhere and monotone
increase is the *safe* direction for it) — and it is the single most likely place to get this
implementation wrong.

**Completeness of the under-lane.** Relative to the published algorithm it is complete on any
graph where no rung-3 hull fired, since then `F' = F`, `S` is the accepting closure and the
result is `leads_to_gfp`'s own fixpoint. On a graph with inflated nodes it may abstain, and
`REFINE` (§1.7) removes the inflation that is to blame. Measured abstention: 1 of 80 at `k₀ = 1`,
0 of 80 at `k₀ = 2`.

### 2.5 The pair is sound because the lanes approximate in opposite directions

This is D7's argument, unchanged: Theorem 1 concludes only `HOLDS`, Theorem 2 concludes only
`VIOLATED`, and the two are mutually exclusive on the same query, so whichever speaks is right.
What is new is that both lanes now read the **same** node set, so a disagreement would be a bug
rather than an inconsistency between two searches.

### 2.6 `E[] φ`

`E[] φ ≡ ¬ leads_to(φ, ¬φ)` evaluated at the initial node (`main_gfp.cpp --eg`,
`PTCTLMC_GFPINITONLY`). Both theorems apply verbatim with `S` restricted to `φ`-nodes and the
root fixed to node 0. Node 0 must be non-inflated for the under-lane; it is, because the root
zone is interned before any successor arrives and rung 3 can only fire on a bucket that is
already at width `k`. `[argued]`

---

## 3. Termination

**Forward.** The zone domain reachable by `post` + `Extra_LU⁺` at a fixed discrete key is finite
(A1/A2; the standard argument). Each bucket is bounded by `k[·]` under rungs 1–3, and rung 3 only
fires when the bound is reached, so `|node| ≤ Σ_κ min(k[κ], A_κ) ≤ Σ_κ A_κ = |D5's graph|`.
A node's zone changes only by growing (rung 1 enlargement, rung 2 union, rung 3 hull); zone
growth at a fixed key is a strictly ascending chain in a finite lattice, so `grew` fires finitely
often, so the worklist empties.

**Backward.** Each `V_v` (resp. `W_v`) takes values in the set of federations over
`{DBMs ⊆ Z_v}`, which is finite because a canonical non-empty DBM inside `Z_v` has entry `(i,j)`
in `[-Z_v(j,i), Z_v(i,j)]` and `Z_v`'s finite entries are bounded by the model constants after
`Extra_LU⁺`. The convergence test accepts only sweeps with **no loss**, and every update either
loses something (progress in a finite well-founded order, once `κ_k`/`ν_k` are applied
deterministically) or leaves `V` fixed. With `k = 1` the per-node domain is the finite set of
sub-DBMs of `Z_v` and the number of sweeps is bounded by `|S| · O(dim² · M)`, `M` the largest
model constant — a **bound the current unbounded-federation operator does not have**, which is
why it needs `if (it > 100000) break;` (`netreach_gfp.cpp:5376`).

**Refinement.** §1.7: `k[κ]` doubles and is capped at `A_κ`; at that point rung 3 is dead at `κ`.
Finitely many keys ⇒ finitely many refinements.

**Not proved:** anything about Zeno runs (A6).

---

## 4. Why it beats D5 / D6 / D7 — states first, then time

### 4.1 States

The honest metric for a *race* is total symbolic states stored across both lanes, because
`leads_to_race` (`netreach_gfp.cpp:5830`) runs D6 to completion in a thread and D5 to completion
in the main thread, always. `gs->fwd_nodes` reports only the winner.

**Satisfied queries** (the front where `liveness-onthefly-results.md` §3.3 says only a discrete
reduction can help — LADDER does not claim one; it claims the *union* of D5's and D6's floors on
one graph):

| query | gfp | D5 | D6 | D7 total | **LADDER k₀=2** | vs D7 |
|---|---:|---:|---:|---:|---:|---:|
| `grc4 T0.in --> T0.far` | 2345 | 2058 | 128 | 2186 | **227** | 9.6× |
| `fischer7 P0.req --> P0.wait` | 9798 | 11598 | 7737 | 19335 | **9236** | 2.1× |
| `fischer6 P0.req --> P0.wait` | 2694 | 2939 | — | — | **2601** | — |
| `csma8 Bus.Active --> Bus.Idle` | *(cap)* | 5888 | 4096 | 9984 | **4989** | 2.0× |
| `csma6 Bus.Active --> Bus.Idle` | 928 | 928 | — | — | **778** | — |
| `csma5 Bus.Active --> Bus.Idle` | 352 | 352 | — | — | **302** | — |
| `arbiter10 C1.cs --> C1.idle` | 6144 | 6144 | — | — | **6144** | 1.0× |
| `arbiter5 C1.cs --> C1.idle` | 112 | 112 | — | — | **112** | 1.0× |

At `k₀ = 1` (pure hull rung) `grc4` is **128** — the discrete floor, 16.1× below D5 — at the cost
of the one abstention. The dial between 128 and 2058 is `k₀`, measured: `k=1 → 128`,
`k=2 → 227`, `k=4 → 371`, `k=∞ → 2126`.

The decomposition of that dial is itself a result: **rung 2 alone (aLU + exact convex union, no
hull, no one-sidedness at all) is worth 1.13–1.18×** — `fischer7` 11598 → 10246, `csma8`
5888 → 4989, `grc4` 2058 → 2126 (a wash). So the *free* rung buys little and the *lossy* rung
buys everything on `grc`. That is the precise answer to the brief's question "is there an
abstraction strictly between aLU and the convex hull": **yes, and its useful part is the lossy
rung with a width bound of 2, not the exact rung.**

**Violated queries.** The under-lane is driven by D5's DFS candidate generator over the same
graph, so the state counts are D5's up to the difference the merged edges make: a hull closes
*more* cycles, so candidates appear no later than in D5 and the confirmation pass is the same
size. `[predicted]` — the prototype in §7.1 does **not** implement the interleaving and pays the
full graph on violated queries; this is the single largest unverified claim of this document, and
§7.2 states the number that would refute it.

### 4.2 Time (all measured, same machine, `PTCTLMC_GFPALU=1`)

| satisfied query | gfp | D5 as shipped | D5 + (C3) | **LADDER k₀=2** | speedup vs best prior |
|---|---:|---:|---:|---:|---:|
| `grc4 T0.in --> T0.far` | 5.771 s | 3.614 s | 0.074 s | **0.020 s** | **181×** (`leads_to_net`: timeout at 240 s) |
| `csma8 Bus.Active --> Bus.Idle` | 23.2 s *(cap)* | 6.10 s *(cap)* | 0.212 s | **0.181 s** | **no verdict → verdict** |
| `csma6 Bus.Active --> Bus.Idle` | 0.399 s | 0.389 s | 0.028 s | **0.026 s** | 15× |
| `fischer7 P0.req --> P0.wait` | 0.219 s | 0.539 s | 0.530 s | **0.136 s** | 4.0× (D6 as shipped: 35.5 s → 261×) |
| `fischer6 P0.req --> P0.wait` | 0.055 s | 0.047 s | 0.044 s | **0.030 s** | 1.6× |
| `arbiter10 C1.cs --> C1.idle` | **1.319 s** | 4.651 s | 4.465 s | 1.570 s | **0.84× — a loss** |
| `arbiter5 C1.cs --> C1.idle` | 0.017 s | 0.016 s | 0.016 s | 0.015 s | 1.1× |
| `csma4 Bus.Active --> Bus.Idle` | 0.017 s | 0.017 s | 0.016 s | 0.016 s | 1.1× |

Attribution of the (C3) speedup, `csma8 Bus.Active --> Bus.Idle`, max federation width:
`4097 (cap) → 16` with `fed_reduce` alone, `→ 4` with `fed_intersect` alone, `→ 1` with both.
On `grc4 T0.in --> T0.far`: `596 → 49 → 26 → 5`, `gfp_pre` calls `227042 → 25764 → 41009 → 17940`.

### 4.3 The worst case, named

1. **Families already at 1.00 zones per discrete key.** `arbiter`, `phil`, `viking`, `twolock`:
   measured node counts identical to D5's at every `k`. The ladder then costs pure overhead — one
   `dbm_try_union` per bucket member per `intern`. On `arbiter10` LADDER is **19 % slower than
   the published algorithm** (1.570 s vs 1.319 s). This is the honest worst case and it is real.
2. **Merging can generate more downstream states than it saves.** This project already measured
   it, on the shipping engine's `PTCTLMC_LTMERGE` path (`netreach_gfp.cpp:2757`): `fischer6`
   55623 → 48336 (win) but `arbiter6` 66137 → **75818** and `twolock6` 35351 → **41943** (loss),
   because a merged zone's successors differ from either part's. LADDER applies rung 2 *after*
   aLU has already collapsed the bucket, and measured no such regression on this set — but the
   mechanism is live and `arbiter`/`twolock` are precisely the families where it bit before.
   §7.2 makes this a refutation criterion.
3. **A query that needs `REFINE`.** Two graph builds instead of one. Measured once in 80.
4. **A model whose backward federations are genuinely non-convex and wide.** `κ_k`/`ν_k` then
   trade the verdict for termination: the over-lane loses precision, the under-lane loses
   witnesses. Nothing on this benchmark set exercises them (max width 26 post-`fed_reduce`), so
   their cost is **unmeasured**.
5. **Zeno.** Unchanged and still open (A6).

---

## 5. Traps that apply

Each of these is a bug that has already been made once in this codebase.

### 5.1 `reset_inactive` breaks the backward `Pre` unless dead clocks are freed

`gfp_pre` (`netreach_gfp.cpp:3718`) must free every clock dead **at the target** before undoing
the step; otherwise `Pre` demands they be 0 at the *source* too, over-constrains, collapses the
fixpoint and **misses violations** (measured on `phil4/6`, `grc4`, `arbiter4/5`). The comment on
`leads_to_gfp` claiming the reduction is disabled and `qmask` is all-ones is wrong in both halves
— `qmask` is all-**zeros** and `reset_inactive` reads it as *additionally keep alive*, so the
reduction runs in full (`liveness-onthefly-results.md` §5).

**What the ladder adds:** merging is confined to one discrete key, so every zone in a bucket has
the same live-clock mask. Merging across keys would produce a zone on which `reset_inactive` has
been applied inconsistently and `gfp_pre`'s repair would be wrong for part of it. Do not do it.

### 5.2 The delay is BEFORE the action

`gfp_pre` applies `Y.down()` at the source after the guards. A stored zone is delay-closed as a
set, which is what licenses the forward `Z ∩ g`, but a single valuation may have to let time pass
first. Removing that `down()` silently under-approximates `Pre`.

### 5.3 The stop test is whole-zone by necessity, and a hull makes it worse

`04-soundness` Thm `asym`(ii): `≼_LU` transfers *continuation*, not *stuckness*. A per-valuation
stuck test on an extrapolated zone invents deadlocks — the `arbiter4 C1.cs --> C1.idle` false
violation witnessed by `x₀..x₃ > 4`. Hence the whole-zone convention, and hence (A5): the
whole-zone convention is itself an **over**-approximation of the end-of-run set (if *some*
valuation of `Z_v` is stuck, *all* of `Z_v` is credited with being able to stop). LADDER inherits
that from `leads_to_gfp` and `leads_to_net` verbatim and does not make it worse; the honest
reading is that both tools decide `p --> q` *modulo the partial-deadlock convention*.

**What the ladder adds:** Lemma L3. On an inflated node the stop test is applied to a zone
strictly larger than anything reachable, and `time_diverges`/`state_has_deadlock` are monotone
*increasing*, so the term can fire on pure hull points. Deleting the disjunct at inflated nodes in
the under-lane is not an optimisation; it is the soundness condition of Theorem 2. Conversely
`adj(v) = ∅` is monotone *decreasing*, so it can fail on a hull where it should hold — which is
why the over-lane must keep `state_has_deadlock` enabled (`PTCTLMC_GFPNOPD` must **not** be set)
exactly as D6's comment already states.

### 5.4 Subsumption closes cycles the concrete system cannot run

`liveness-subsumption-bug.md`: on `grc4`, node 504's true successor `x₀ ≥ 4` is absorbed into node
94's `x₀ ≥ 0` and the loop closes with the growth of `x₀` erased. The forward search is therefore
only a **candidate generator**; the backward pass arbitrates. A hull absorbs strictly more, so the
candidate rate rises and D5's refutation budget (`PTCTLMC_OTFREFUTE`, default 8) matters more, not
less. Do not attempt to fix this by marking subsumption edges: `liveness-subsumption-bug.md` §4
shows the marking has to be enforced at four separate places in a Tarjan and is still only
sufficient, never necessary.

### 5.5 A capped fixpoint is neither an over- nor an under-approximation

Returning `false` from a capped pass launders "out of budget" into a silent SATISFIED. The current
code reports `UNUSABLE` (`Otf::certificate_in`). LADDER removes the failure mode rather than
reporting it — `κ_k`/`ν_k` bound the width by construction — but the same discipline applies to
the **under-lane's empty result**: `V = ∅` everywhere is *not* `HOLDS`, it is *no certificate*.
Only Theorem 1 may conclude `HOLDS`.

### 5.6 In-place enlargement invalidates adjacency

`node[j].zone ← s.zone` (rung 1) and both merge rungs make `adj[j]` stale. `Otf::expand` already
resets `expanded[j]` and clears `adj[j]`; the trap is the **driver**, not the intern: a
level-synchronous BFS re-publishes the reset node's successors on every later level and the
frontier grows quadratically (§1.4, measured 88×). Use the worklist.

### 5.7 `fed_subtract` is not an intersection primitive

`fed_subtract(z, fed_subtract(z, cand))` denotes `z ∩ cand` but constructs the arrangement of
`cand`'s complement, splitting each piece by up to `2·dim²` constraints per subtrahend at each of
two levels. `fed_intersect({z}, cand)` denotes the same set in `≤ |cand|` convex pieces. Related:
`gfp_fed_compact` (`netreach_gfp.cpp:3778`) builds a federation `rest` and never reads it — the
joint-coverage test it was evidently meant to perform is missing, and so is the convex-union
merge that `fed_reduce` already implements two headers away.

---

## 6. Prior art

An adversarial search was run. **Nothing in §1 is claimed as new theory.** The composition may be
new; each ingredient is not.

**Greatest fixpoints for liveness; reachable-set restriction.** `EG` as a greatest fixpoint and
fair-`EG` as a nested fixpoint is Emerson & Lei, *Efficient model checking in fragments of the
propositional mu-calculus* (LICS 1986); Somenzi, Ravi & Bloem (FMCAD 2002) show EL is
complexity-optimal among GSH-derivable algorithms. Restricting CTL fixpoints to the
forward-reachable set is default in VIS and a flag in NuSMV. OWCTY (Fisler, Fraer, Kamhi, Vardi,
Yang, TACAS 2001) is reachability-then-shrinking-fixpoint in parallel. Chaotic iteration is Cousot
(1977). All already in `07-related.tex`.

**Early exit from a descending greatest-fixpoint iteration.** Farn Wang, *Under-approximation of
the Greatest Fixpoint(s) in Real-Time System Verification* (arXiv `cs/0501059`, `cs/0501060`), and
the EDGF technique in his TCTL-inevitability work (CIAA 2003, arXiv `cs/0304003`), test emptiness
of the intersection with the target at every iteration to stop early. **This is prior art for the
branch's D3/D5 early exit and for §1.6's `gfp_under`, and `07-related.tex` still does not cite
it.** It should.

**Narrowing, and the fact that any post-fixpoint is a certificate.** Cousot & Cousot (POPL 1977;
and the *Abstract Interpretation: Past, Present and Future*, CSL-LICS 2014 retrospective) define
widening/narrowing and explicitly treat "order duality for approximations from above and below,
ascending and descending iterations". Lemma L4 is a specialisation of the standard fact that a
descending iteration's limit is a post-fixpoint, applied to an *arbitrarily* narrowed operator;
Amato, Scozzari et al., *Efficiently intertwining widening and narrowing* (arXiv 1503.00883) work
in the same space. **Claim L4 as an application, not as a theorem.** What is genuinely load-bearing
and not obviously written down anywhere is the *corollary* that no verification pass is needed
because the post-fixpoint condition is exactly the local update condition — that is a one-line
observation and should be presented as such.

**Bounded-width disjunctive domains.** Bagnara, Hill & Zaffanella, *Widening operators for powerset
domains* (VMCAI 2004; STTT 2006) define widenings on the finite powerset construction, parametric
in a "connector" that decides which pair to merge. `κ_k` is exactly one of their instantiations
with `dbm_hull` as the connector and "fewest relaxed entries" as the heuristic; Sankaranarayanan
et al., *Static Analysis in Disjunctive Numerical Domains* (SAS 2006) is the same idea.
**`κ_k` is prior art in full.**

**Convex merging of zones.** Merging two zones at the same location when their union is already
convex, so that no state is added, is standard: André, Fribourg, Soulat, *Merge and conquer: state
merging in parametric timed automata* (ATVA 2013) and André & Marinho, *Efficient convex zone
merging in parametric timed automata* (FORMATS 2022). UPPAAL's convex-hull over-approximation
(`-A`) is Balarin (1996) / Wong-Toi and is the classical lossy version. **Rungs 2 and 3 are both
prior art**; what is not standard is running them *inside one graph* with a per-node marker.

**Three-valued / may-must abstraction.** The scheme "one abstract model, over-approximating for
one verdict and under-approximating for the other, refine when neither speaks" is exactly
may/must (modal) abstraction: Larsen & Thomsen modal transition systems (1988); Bruns & Godefroid
three-valued model checking (CAV 1999); Godefroid, Huth & Jagadeesan (CONCUR 2001); Godefroid &
Jagadeesan, *On the expressiveness of 3-valued models* (VMCAI 2003); Shoham & Grumberg,
*3-valued abstraction: more precision at less cost* (LICS 2006), which also introduces
**per-state precision markers**. LADDER's `INFLATED` flag is a must/may distinction under another
name — with the specialisation, which I have not found stated, that in a **backward** liveness
fixpoint on a *zone* graph the "must" requirement collapses to two local uses (the stop disjunct
and the root) because `Pre` under a real transition tag is exact regardless of the target zone
(§2.4). **State it as a specialisation of may/must abstraction, not as a new framework.**

**Abstraction refinement for timed automata.** Herbreteau, Srivathsan, Walukiewicz and co-authors,
*Abstraction refinement algorithms for timed automata* (CAV 2019, arXiv 1905.07365) refine the set
of clock constraints admissible in a zone from a spurious counterexample; Dierks, Kupferschmid &
Larsen (FORMATS 2007) do localisation reduction. §1.7's refinement is coarser than either (it
doubles a per-key width bound rather than recovering a constraint), and those papers would
**subsume it** if implemented — as `liveness-onthefly-candidate.md` §6 already concedes for QCUT's
`Inv` layer.

**Timed liveness and subsumption.** Tripakis, *Checking timed Büchi automata emptiness on
simulation graphs* (TOCL 2009); Laarman, Olesen, Dalsgaard, Larsen & van de Pol, *Multi-core
emptiness checking of timed Büchi automata using inclusion abstraction* (CAV 2013), which proves
Büchi emptiness is **not** preserved under subsumption and builds an ndfs that exploits the
structural properties that are; Herbreteau, Srivathsan, Tran & Walukiewicz, *Why liveness for
timed automata is hard, and what we can do about it* (FSTTCS 2016 / TOCL 21(3), 2020), which
separates liveness from reachability complexity-theoretically and builds *liveness-compatible
subsumption graphs* by an iterative construction. Akshay, Gastin, Govind & Srivathsan (CONCUR
2024) give a further new zone-based Büchi non-emptiness algorithm. **The branch's premise — that
the backward direction makes subsumption harmless — is the counterpoint to this whole line, and
the honest framing is that LADDER buys robustness on top of that premise, not a new answer to it.**

**Certified/certifying liveness.** Wimmer et al., *Certifying emptiness of timed Büchi automata*
(FORMATS 2020) produce and machine-check certificates for TBA emptiness. That is the `HOLDS`
direction as a checkable object, i.e. Theorem 1's `W` under another name. Cite it; do not claim
the certificate idea.

**Backward timed analysis / distributed.** Henzinger, Nicollin, Sifakis & Yovine (1994) and the
Kronos lineage; ZEUS (Braberman, Olivero, Schapachnik, PDMC 2002 / FMSD 2006) for the distributed
backward form. Already recorded in `parallel-search-novelty.md` §4sexies.

### Verdict

The one sentence this document is entitled to:

> In a **backward** greatest-fixpoint liveness check over zones, a may/must (over/under) pair can
> share a *single* merged zone graph, because the "must" obligation of the under-approximating
> lane reduces to two local conditions — the node carrying the end-of-run disjunct, and the node
> the verdict is read at — while every `Pre` edge stays exact no matter how coarsely its target's
> zone was merged.

Everything else — the fixpoint, the narrowing, the powerset width bound, the convex merge, the
three-valued framing, the refinement loop — is standard, and the largest practical result in this
document (§0.1, the 289× and the `csma8` verdict) is **a bug fix, not an algorithm**. Lead with
the measurement.

---

## 7. Implementation sketch, and the experiment that would refute it

### 7.1 What to change, in order of value per line

All in `nove_mc/prototype/netreach_gfp.cpp` unless stated. A scratch build with these behind
`PTCTLMC_X*` env switches produced every measured number above; the switches exist only so the
A/B is possible and should not survive into the shipping path.

1. **(C3), ~6 lines, ships on its own.** In `Otf::gfp_on` (`:5376`) and in `leads_to_gfp`'s
   phase 3 (`:3794`ff), replace
   `fed_subtract(z, fed_subtract(z, cand))` by `fed_intersect(W_v, cand)` and
   `gfp_fed_compact(f, cap, &capped)` by `fed_reduce(f)`. Keep the cap as a tripwire; it should
   never fire again. **This alone converts `csma8 Bus.Active --> Bus.Idle` from UNUSABLE to a
   0.21 s verdict and `grc4 T0.in --> T0.far` from 3.6 s to 0.07 s, with zero changes to the
   state count and zero to the theory.** Ship it first, on its own differential run.
2. **(C5), ~12 lines.** Replace the level-synchronous frontier in `Otf::run` (Base/D6/D8 arm)
   and in `run_bfs_incremental` with the deduplicated worklist of §1.4. Required by any merging
   `intern`; worth 88× on `fischer7` D6.
3. **(C1), ~35 lines.** The ladder in `Otf::intern` (`:5218`), plus a `std::vector<char>
   inflated` member resized in `Otf::grow`, plus `PTCTLMC_OTFWIDTH` (already present for D8).
   Note the ladder is **not** D8 as currently written: D8 uses `Dbm::includes` for the covered
   test, which is strictly weaker than aLU and measured **worse than D5** on states
   (`fischer7` 27968 vs 11598). Rung 1 must be `alu_subsumed_by`.
4. **(C2), ~25 lines.** Two flags on `gfp_on`: `use_stop_at_inflated` and `widen` vs `narrow`.
   `accepting_closure` (`:5716`) is unchanged for the over-lane; the under-lane's `S` is the same
   set, and the *root* filter (`is_p(v) ∧ ¬inflated[v]`) goes in `certificate_in`.
5. **(C4), ~20 lines.** `κ_k`/`ν_k` after `fed_reduce`. Default `k = 64`: high enough never to
   fire on this benchmark set, low enough to be a hard bound.
6. **(C6), ~30 lines.** The `Blame` sweep and the rebuild loop, with a hard cap of 4 refinements.
7. **Driver.** `run_dfs_candidates` (`:5589`) already is the under-lane's on-the-fly driver; the
   only edits are the `¬inflated` guard on kind-1 candidates and calling `gfp_under` from
   `confirm`. Then `leads_to_race` (`:5830`) becomes unnecessary: LADDER is a single-threaded
   algorithm that does what D7 does with two threads and two graphs.

Do **not** touch `src/netreach.cpp` in the same change. It still carries the §4c truncation bug of
`liveness-onthefly-results.md`, which is soundness-relevant and independent.

### 7.2 The falsifiable experiment

`benchmarks/models/{arbiter,fischer,csma,phil,twolock,viking,grc}*.xml`, harness
`nove_mc/prototype/run_otf.py` with `queries.txt` (48) and `scale_queries.txt` (32), plus
`--ref` against `leads_to_net`. Report symbolic states stored at the moment of the verdict,
`gfp_pre` calls, maximum federation width, wall time, and verdict.

**Predictions. Any one of these failing refutes the corresponding claim.**

| # | claim | prediction | refuted if |
|---|---|---|---|
| P1 | (C3) is a pure win | verdicts on all 80 queries identical to the current `d5`/`gfp` verdicts; `csma8 Bus.Active --> Bus.Idle` SATISFIED in < 1 s; `grc4 T0.in --> T0.far` < 0.2 s; max width ≤ 32 everywhere | **any** verdict changes, or any query still caps at `FEDCAP=4096` |
| P2 | (C5) is required and sufficient for the hull's forward cost | `fischer7 P0.req --> P0.wait` under the hull lane < 0.5 s (from 35.5 s) | > 2 s |
| P3 | the ladder at `k₀=2` is two-sided complete on this set | **0 disagreements and 0 `REFINE` events** over all 80 queries | any disagreement (**fatal**), or > 4 `REFINE` events |
| P4 | states, satisfied queries | `grc4 T0.in --> T0.far` ≤ 250 (D5: 2058); `fischer7 P0.req --> P0.wait` ≤ 9500 (D5: 11598, D7 total: 19335); `csma8 Bus.Active --> Bus.Idle` ≤ 5100 (D7 total: 9984) | any of the three exceeds D5's own count |
| P5 | **states, violated queries — the largest untested claim** | with the D5 driver wired in, the geometric mean of `LADDER-expanded / D5-expanded` over the 39 violated queries of `queries.txt` is ≤ 1.05, and no query exceeds 2× D5 | geo-mean > 1.2, or any query > 5× D5 — which would mean the merged graph destroys D5's early exit and the two lanes must stay separate after all |
| P6 | merging does not blow up downstream, contra `PTCTLMC_LTMERGE` | `arbiter{6,8,10}`, `twolock{5,6}`, `phil{6,8}`, `viking6` node counts **within 1.02×** of D5's at `k₀=2` | any of them exceeds 1.10× D5 — the ATVA-2013 merging regression is live and rung 2 must be gated per family |
| P7 | the width bounds are insurance, not a working part | `κ_k`/`ν_k` fire on **0** of 80 queries at `k = 64` | they fire, in which case §4.3(4)'s unmeasured precision cost becomes load-bearing and must be measured before shipping |
| P8 | against the shipping engine | LADDER settles `grc4 T0.in --> T0.far` (which `leads_to_net` cannot finish in 240 s) and matches its verdict on the other 79 | any disagreement (**fatal**) |

**The cheapest refutation to run first is P1**: it is six lines, needs no new mode, and if the 80
verdicts move, everything above is void.

**A well-argued negative result is available here too.** If P5 fails — if the merged graph costs
the under-lane its early exit — then the correct conclusion is that D7's *two* graphs are not
waste but specialisation, LADDER should be cut back to (C3)+(C5) as pure bug fixes applied to D5
and D6 separately, and the D7 race kept. That outcome would still leave the two largest measured
numbers in this document intact, because both belong to (C3).

---

## 8. What this document does not claim

* No discrete-state reduction. `p --> q` that **holds** still requires the reachable discrete set
  to be enumerated (`liveness-onthefly-results.md` §3.3), and LADDER enumerates it. The open front
  named there is still open.
* No new answer to Herbreteau–Srivathsan–Walukiewicz. Subsumption remains unsound for forward
  cycle detection; this branch's premise (harmless backward) is unchanged and unextended.
* No Zeno-awareness. Same semantics as `verifyta` and `leads_to_net`, same gap.
* No claim that `INFLATED` is a new abstraction framework — it is may/must under another name
  (§6), with one specialisation that makes it cheap on a backward zone fixpoint.
* No production code. Every number above comes from a scratch copy of the prototype outside the
  repository; the repository's `netreach_gfp.cpp` was not modified.
