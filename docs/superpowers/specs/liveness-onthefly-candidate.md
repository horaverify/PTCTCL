# QCUT: deciding `p --> q` on the ¬q-cone of an invariant-certified entry set

Branch `liveness-on-the-fly`. Candidate successor to D1–D5, aimed at the one case where all
five are stuck: **SATISFIED** queries, where every existing variant pays the whole reachable
zone graph.

Status of every claim below is marked: **[proved]**, **[measured]**, **[predicted]** (a number
this document commits to in advance, which the experiment of §7 can refute), **[open]**.

---

## 0. Verdict first

**The idea.** For a SATISFIED verdict the backward greatest fixpoint never needs the reachable
graph. It needs two things: (i) an over-approximation of the set of `p ∧ ¬q` states, and (ii) the
`¬q`-forward-closure of that set. The `q`-region exists in D1–D5 for one reason only — to
*discover* the reachable `p ∧ ¬q` states — and that job can be done by a **product-free
inductive discrete invariant** that costs *zero symbolic states* to compute. Delete the
`q`-region and you delete every reachable discrete state outside the `¬q`-cone.

**What is skipped, exactly.** Every reachable discrete state that neither satisfies `p ∧ ¬q` nor
is reachable from a `p ∧ ¬q` state along a path that stays in `¬q`. On `arbiter5
C1.cs --> C1.idle` that is 96 of the 112 reachable discrete states. On `phil4
P0.Eat --> P0.Dropped` it is 71 of 80.

**Why skipping is sound.** Theorem `thm:sound-holds` of `04-soundness.tex` is *unconditional* in
one direction: `Ω ⊆ W^∞`, so **W empty ⟹ holds**, and that direction survives *any*
over-approximation of the graph — bigger zones, extra edges, extra nodes, a coarser confining
set. QCUT exploits exactly this: every approximation it makes can only turn a HOLDS into an
ABSTAIN, never a VIOLATED into a HOLDS. It therefore answers **HOLDS or ABSTAIN**, never
VIOLATED, and defers to D5 when it abstains.

**Measured/predicted saving** (§4): 1.4×–9× fewer discrete states on 7 of the 9 satisfied queries
in `nove_mc/prototype/queries.txt` plus 3 more from the 164-query set; **a measured LOSS on
`csma4`/`csma5`**, where the invariant generator is defeated by broadcast.

**Prior-art verdict** (§6): every ingredient is published — Emerson–Lei, reachable-set
restriction (VIS/NuSMV), cone-of-influence / live-component analysis for timed automata,
compositional invariant generation (D-Finder, Petri-net place invariants and traps), invisible
invariants for `p --> q` bounded response, backward TCTL (Kronos/ZEUS), Farn Wang's
under-approximated greatest fixpoints. **The combination — replacing the forward reachability
pass that confines a zone-based backward gfp with a product-free inductive discrete invariant,
and restricting the fixpoint carrier to the ¬q-cone of the p-cylinder — is not something the
search found published. The novelty claim is weak and should be stated as such.** The honest
headline is: *symmetry reduction under the query stabiliser is a comparable or larger lever on
half these queries, it is textbook, and it is already half-implemented in `src/symmetry.cpp`.*
QCUT composes with it multiplicatively (§4.4).

---

## 1. The algorithm

### 1.1 Primitives reused unchanged

| primitive | file | role in QCUT |
|---|---|---|
| `gfp_succ_tagged(net,s,L,U,qmask,emode,out,tags)` | `netreach_gfp.cpp:3512` | the **only** successor generator; returns the *full* successor list plus per-transition edge tags |
| `gfp_pre(net,tag,src,tgt,X)` | `:3671` | exact concrete `Pre`, incl. the active-clock inversion and the pre-action `down()` |
| `Dbm`, `Federation`, `fed_subtract`, `gfp_fed_*` | `:3717`–`:3744` | fixpoint values |
| `dbm_is_alu_le` / `Dbm::alu_subsumed_by` | via `Otf::intern` `:5159` | forward subsumption inside the cone |
| `state_has_deadlock(net,s)` | `:3297` | partial-deadlock flag in `stop` |
| `time_diverges` | `Otf::time_diverges` `:5229` | divergence flag in `stop` |
| `extrapolate_lu_plus` / `_nodiag` | `Dbm` | `Extra`, chosen by `network_has_diagonal` |
| `all_invariants`, `frozen_flags`, `reset_inactive`, `apply_all` | `:161`–`:212` | seed construction |
| `symmetry_group` / `canonicalize` | `src/symmetry.cpp`, `netreach.cpp:769` | optional orbit layer (§4.4) |

Nothing in this list is modified. QCUT is a different *driver* over the same operators.

### 1.2 Notation

Write `Disc` for discrete states `v = (locs, ints)`, `I_v` for the conjunction of location
invariants at `v` (`all_invariants`), `frozen(v)` for committed-or-urgent (`frozen_flags`),
`ρ_v` for the active-clock reduction at `v` (`reset_inactive`), and `Extra` for the extrapolation
selected by `emode`. `Z↑` is `Dbm::delay()`, `Z↓` is `Dbm::down()`.

The **forward operator** as implemented (one branch of `gfp_succ_tagged`, edge set `e`):

```
Post_e(v, Z) = Extra( ρ_{v'} ( D_{v'}( reset_e( Z ∧ g_e ) ∧ I_{v'} ) ) )
D_{v'}(Y)   = if frozen(v') then Y else (Y↑ ∧ I_{v'})
```

The **seed zone** at a discrete state `v` — the top element of that operator's image at `v`:

```
seed(v) = Extra( ρ_v ( D_v( ⊤ ∧ I_v ) ) )                                        (Def. SEED)
```

where `⊤` is the universal DBM — **`Dbm::universe(nclocks)`**, `HDBM/include/ptctlmc/dbm.hpp:47`.
Note that `Dbm(nclocks)` is documented as *"the zone in which every clock is 0"*, i.e. the origin,
**not** `⊤` — see §5, trap T6.

**Lemma 1 (seed is the top).** For every `v` and every `Z ⊆ ⊤`, and every edge set `e` into `v`,
`Post_e(u, Z) ⊆ seed(v)`. **[proved]** — `∧ g`, `reset`, `∧ I`, `↑`, `ρ`, `Extra` are each
monotone w.r.t. `⊆`, and `Post_e(u,Z)` is `Extra(ρ_v(D_v(W ∧ I_v)))` for some `W ⊆ ⊤`.
Monotonicity of `Extra_LU+` is the standard property of LU-extrapolation; `Extra_LU+_nodiag`
likewise. ∎

Lemma 1 is why seeding with `seed(v)` costs exactly **one zone per seeded discrete state**: any
zone the cone search later derives at `v` is subsumed by the seed already stored there, so
`intern` returns the seed and the bucket never grows.

### 1.3 The discrete invariant layer `Inv`

`Inv ⊆ Disc` must satisfy **`Reach_disc ⊆ Inv`** and nothing else. Two product-free generators,
neither of which stores a single symbolic state:

**(L) Linear place invariants.** Places = all `(process, location)` pairs. Build the incidence
matrix over the transition set
`{local edges} ∪ {binary sync pairs} ∪ {broadcast sender edges alone} ∪ {broadcast receiver
edges alone}`, and take the rational nullspace `{y : Δ·y = 0 for every row Δ}`. Requiring each
broadcast *receiver* edge to be `y`-neutral on its own makes every `k`-receiver instance neutral,
so the computed basis is a **subset** of the true P-invariants — sound, possibly weak. Then
`Inv_L = { v : y·χ(v) = y·χ(v₀) for every basis vector y }`, `χ` the location-indicator vector.
Cost: Gaussian elimination on `|places| = Σ|L_i|` columns, i.e. 47 columns for `arbiter15`.

**(T) Thread-modular process × integer products.** For each process `j`, the least set
`TM_j ⊆ L_j × IntVec` containing `(init_j, iv₀)` and closed under (a) `j`'s own edges whose
integer guard holds, optionally composed with one other process's assignment (the sync partner),
and (b) *any other* process's edge whose integer guard holds, with `j` stuttering. Then
`Inv_T = { v : (locs[j], ints) ∈ TM_j for every j }`. Cost `Σ_j |L_j| · |Dom(ints)|`.

`Inv = Inv_L ∩ Inv_T`. **[proved]** each conjunct is inductive by construction, hence
`Reach_disc ⊆ Inv`. **[measured]** on this suite `Inv` is *exact on the entry cylinder*
(`spurious_entry = 0`) for `arbiter4/5`, `fischer4/5`, `twolock4`, `phil4`, `grc4`, and
**fails completely on `csma4/5`** (§4.3).

An optional third generator **(C)** — the symmetry/counter quotient — repairs `csma`; it is
classical (§6) and is a plug-in, not a contribution.

`Inv` is evaluated per state as a handful of integer dot-products plus `nprocs` table lookups.

### 1.4 Seed enumeration

```
SEED = { v ∈ Disc : p(v) ∧ ¬q(v) ∧ Inv(v) }
```

enumerated by backtracking over processes and integer cells with bounds propagation on the
linear conjuncts of `Inv_L` and membership propagation on `Inv_T`. On `arbiter` the constraint
`#Arb.Busy − Σ_j #C_j.cs = 0` together with `C1 = cs` immediately forces `Arb = Busy` and
`C_j ≠ cs` for `j ≠ 1`, so the enumeration is dead-end-free and output-linear. **[measured]**
`|SEED| = 16` on `arbiter5`, which is exactly the reachable count.

A **budget** `B` guards the enumeration: if `|SEED| > B` (default `B = 4·|SEED_reach_estimate|`,
or simply a user cap), QCUT abstains before doing any zone work.

### 1.5 The driver

```
qcut(net, p, q) -> HOLDS | ABSTAIN
  # ---- phase 0: certify (no symbolic states) --------------------------------
  Inv  <- place_invariants(net) ∩ thread_modular(net)
  SEED <- enumerate { v : p(v) ∧ ¬q(v) ∧ Inv(v) }        # budget-guarded
  if |SEED| > B: return ABSTAIN
  if SEED = ∅:   return HOLDS                            # no p∧¬q state exists at all

  # ---- phase 1: the ¬q-cone --------------------------------------------------
  node <- []; adj <- []; full_succ_empty <- []; bucket <- {}
  for v in SEED: intern( SymState(v, seed(v)) )          # Lemma 1: one zone each
  frontier <- all node ids
  while frontier nonempty:
     for u in frontier, not yet expanded:
        succ, tags <- gfp_succ_tagged(net, node[u], L, U, qmask, emode)
        full_succ_empty[u] <- (succ = ∅)                 # TRAP T1: the FULL list
        for k in succ:
           w <- discrete part of succ[k]
           if q(w)        : continue                     # the q-cut
           if not Inv(w)  : continue                     # the Inv-cut  (Lemma 3)
           j <- intern(succ[k])                          # aLU, as Otf::intern
           adj[u].push({j, tags[k]})
           if j unexpanded: frontier' .push(j)

  # ---- phase 2: the confined greatest fixpoint -------------------------------
  W <- gfp_on( all nodes )                               # Otf::gfp_on, unchanged
       #   W[v] <- W[v] ∧ ( stop(v) ∨ ⋃_e Pre_e(W[v']) ),  W[v] init = node[v].zone
       #   stop(v) = full_succ_empty[v] ∨ time_diverges(v) ∨ state_has_deadlock(v)

  # ---- phase 3: verdict ------------------------------------------------------
  if fed_empty(W[v]) for every node v with p(v): return HOLDS
  return ABSTAIN
```

**Sequencing with the existing engine.** QCUT never returns VIOLATED, so it must not replace
D5. The recommended router:

```
1. D5 with a node budget β (β = 256 covers 38 of the 39 violated queries measured; grc4
   T0.near-->T0.in needs 1903 and is the exception).   -> VIOLATED, done.
2. QCUT.                                               -> HOLDS, done.
3. D5 to completion.                                   -> either verdict.
```

Steps 1 and 2 are independent and may be raced, as the shipping engine already races lanes.

---

## 2. Soundness and completeness

Rigour and notation follow `nove_mc/src/04-soundness.tex`; the statements there are used, not
re-proved.

### 2.1 Assumptions, all of them

* **(A0) Discrete predicates.** `p`, `q` are functions of `(locs, ints)` only. Used in
  Lemma `lem:disc-pred` and, here, in Lemma 3 and Theorem 2.
* **(A1) Extensivity.** `Extra` is extensive: `Z ⊆ Extra(Z)`. Also **monotone**: `Z ⊆ Z' ⟹
  Extra(Z) ⊆ Extra(Z')`. (Monotonicity is *additionally* needed here, for Lemma 1; `04-soundness`
  needs only extensivity.)
* **(A2) Simulation abstraction.** every `ν ∈ Extra(Z)` is `≼LU`-simulated by some `ν' ∈ Z`.
  Used only for the aLU subsumption in `intern` (Cor. `cor:alu`), exactly as before.
* **(A3) Active-clock reduction is a bisimulation.** `ρ_v` zeroes clocks that no reachable
  guard or invariant reads from `v`; two valuations differing only in `v`-dead clocks have the
  same future. We therefore work throughout in the **ρ-reduced** transition system, in which
  `Reach(v) ⊆ seed(v)` holds literally, and `gfp_pre` inverts `ρ` at the target
  (`netreach_gfp.cpp:3679`). *This assumption is inherited, not introduced.*
* **(A4) Delay-closure.** stored zones are delay-closed as sets in non-frozen locations; a single
  valuation may still need to delay before a guard opens, which is why `gfp_pre` ends with the
  outer `down()`.
* **(A5) Partial deadlock.** `stop` uses `state_has_deadlock` as a **per-node flag**, not a
  per-valuation set (Cor. `cor:whole`). Inherited.
* **(A6) Zeno.** the cycle witness is Zeno-permissive, identically to `verifyta -u -S0`,
  `leads_to_net` and `leads_to_gfp` (`04-soundness` §Zeno). Inherited; QCUT changes nothing here.
* **(A7) `Reach_disc ⊆ Inv`.** Proved per generator in §1.3.

### 2.2 The three new lemmas

**Lemma 2 (seeds cover).** For every reachable `(v, ν)` with `v ⊨ p ∧ ¬q`, the node QCUT
stores for `v` has `ν` in its zone. **[proved]**
*Proof.* `v ∈ Reach_disc ⊆ Inv` by (A7), and `p(v) ∧ ¬q(v)`, so `v ∈ SEED` and a node with zone
`seed(v)` is interned. `ν ∈ Reach(v)`, and `Reach(v) ⊆ seed(v)`: the forward closure that
produced `(v,ν)` is a composition of `Post_e`, and each `Post_e(·, Z) ⊆ seed(v)` by Lemma 1.
Two seeds never merge in `intern` (different `disc_key` ⟹ different bucket), so the node is
`(v, seed(v))` itself. ∎

**Lemma 3 (the Inv-cut removes no relevant transition).** Let `(v,ν)` be reachable with
`v ⊨ p ∧ ¬q`, and let `(v,ν) = (v₀,ν₀) →_{e₁} (v₁,ν₁) →_{e₂} …` be any maximal run that stays in
`¬q`. Then every `v_i ∈ Inv`, so no step of the run is cut by the `Inv`-filter; and no step is
cut by the `q`-filter. **[proved]**
*Proof.* Each `(v_i, ν_i)` is reachable (it is reached from the reachable `(v,ν)`), so
`v_i ∈ Reach_disc ⊆ Inv` by (A7). `v_i ⊨ ¬q` by hypothesis, and `q` is a function of `v_i` alone
by (A0). ∎

Lemma 3 is the load-bearing one, and it is worth stating what it does **not** say. The `Inv`-cut
*does* remove transitions from *spurious* nodes — nodes in the cone that are not reachable, or
reachable valuations that leave `Inv`… which cannot happen, precisely by (A7). So the cut is
free of cost on genuine runs and strictly precision-*improving* on spurious ones. That is the
unusual part: the same filter shrinks the graph and sharpens the fixpoint.

**Lemma 4 (`stop` is over-approximated).** With `full_succ_empty[v]` computed from the **full**
`gfp_succ_tagged` output, `stop_QCUT(v) ⊇ { ν ∈ seed(v) : the maximal run from (v,ν) ends at v }`.
**[proved]** — identical to the argument in `04-soundness` Cor. `cor:whole`; the point is only
that the `q`-cut and `Inv`-cut must not be allowed to *manufacture* an empty adjacency. See trap
T1. ∎

### 2.3 Soundness

Let `Ĝ` be the graph QCUT builds, `F̂` its Phase-2 operator, `Ŵ^∞` the fixpoint.

**Theorem 1 (emptiness proves the property).** If `Ŵ^∞(n) = ∅` for every node `n` with
`p(n)`, then `p --> q` **holds**. **[proved]**

*Proof.* Suppose not: some reachable `(v,ν)` with `v ⊨ p ∧ ¬q` has a maximal `¬q` run `π`.
Define, for each node `n = (u, Ẑ_u)` of `Ĝ`,
`Ω̂(n) = { μ ∈ Ẑ_u : some maximal ¬q run from (u,μ) exists }`.
`Ω̂` is a post-fixpoint of `F̂`: for `μ ∈ Ω̂(n)`, the witnessing run either takes a step `e` to
`(u', μ')`; by Lemma 3 that step survives both cuts, so `gfp_succ_tagged` produced it and
`intern` returned some node `n'` whose zone (aLU-)covers `μ'` — hence `μ ∈ Pre_e(Ω̂(n'))`, using
Cor. `cor:alu` to transfer the run across the subsumption in the aLU case and plain containment
otherwise. Or the run does not continue, and then `μ ∈ stop_QCUT(n)` by Lemma 4. And
`Ω̂(n) ⊆ Ẑ_u` by construction. So `Ω̂ ⊆ F̂(Ω̂)`, and Knaster–Tarski gives `Ω̂ ⊆ Ŵ^∞`.
By Lemma 2 the node `n_v = (v, seed(v))` exists with `ν ∈ seed(v)`, and `ν ∈ Ω̂(n_v)` by the
choice of `π`. So `Ŵ^∞(n_v) ≠ ∅` and `p(n_v)` — contradiction. ∎

Theorem 1 uses **only** the direction of `04-soundness` Thm `thm:sound-holds` that is
unconditional. It does not use (A2) except through `cor:alu`, and it does not care that
`seed(v) ⊋ Reach(v)`: over-approximating the ambient zone only makes `Ŵ^∞` larger, which is the
safe direction for concluding "holds". Subsumption edges are harmless for the same reason they
are harmless in Algorithm 1.

**Corollary (monotone failure).** Every approximation QCUT makes — a loose `Inv`, the maximal
seed zone, aLU subsumption, the per-node `stop` flag — enlarges `Ŵ^∞`. Enlarging `Ŵ^∞` can turn
a HOLDS into an ABSTAIN and can never turn a VIOLATED into a HOLDS. **[proved]**

### 2.4 Completeness, honestly

QCUT is **not** a decision procedure on its own, and deliberately does not try to be.

**Theorem 2 (relative completeness).** If QCUT returns ABSTAIN, the router falls through to D5,
which is verdict-identical to `leads_to_gfp` (measured 47/47 with 0 disagreements, and
`leads_to_gfp` is 164/164 against `leads_to_net`/`verifyta`). Hence the *router* is complete
exactly to the degree D5 is. **[proved, given the existing measurements]**

Why QCUT cannot answer VIOLATED, stated precisely. `Ŵ^∞(n) ≠ ∅` fails to imply a violation for
three independent reasons, and each is real:
1. **`n` may be unreachable.** `Inv ⊋ Reach_disc` in general.
2. **The witness valuation may be unreachable at a reachable `n`.** `seed(v) ⊋ Reach(v)` by
   construction.
3. **`stop` is a per-node flag** (A5). At a seed the flag is evaluated on `seed(v)`, which
   contains more valuations than `Reach(v)`, so `state_has_deadlock` can fire where the
   reachable zone has no deadlock. This is a *strictly larger* imprecision than Algorithm 1's,
   because Algorithm 1 evaluates the same flag on the tighter forward zone.

A tempting repair — "ask the engine `E<> (v ∧ Ŵ_v)`; SAT ⟹ VIOLATED" — is **unsound** because of
(3): a reachable valuation in `Ŵ_v` may owe its membership to a spurious deadlock flag. It is
sound in the other direction (all such queries UNSAT ⟹ HOLDS) but that costs a full reachability
sweep, which is exactly what QCUT exists to avoid. Recorded as a rejected variant, §8.5.

---

## 3. Termination

**Proposition.** Every phase terminates. **[proved]**

* **Phase 0.** `Inv_L` is Gaussian elimination on a fixed matrix. `Inv_T` is a least fixpoint over
  the finite set `L_j × Dom(ints)`, monotone, hence terminating. Seed enumeration is bounded by
  the finite cylinder and by `B`.
* **Phase 1.** `intern` stores a node only when the newcomer is not covered by any stored zone at
  the same `disc_key`. Zones are `Extra`-images, so every finite constant is bounded by the
  network's `L,U` constants; there are finitely many such DBMs over finitely many clocks, and
  finitely many `disc_key`s. So the node set is finite and the frontier loop terminates. Note the
  in-place enlargement path of `Otf::intern` (`node[j].zone = s.zone`) can re-open an expanded
  node; the standard argument applies — enlargement is strictly increasing in a finite lattice, so
  it happens finitely often. Under `seed`-seeding it happens *never* at a seeded key (Lemma 1).
* **Phase 2.** `Otf::gfp_on` unchanged: `W[v]` is only ever replaced by `W[v] ∩ C`, so the vector
  is pointwise decreasing, every `W[v]` stays inside a fixed `Extra`-image zone, and the
  sub-federations of such a zone form a finite lattice (Prop. `prop:term` of
  `03-liveness-algorithm.tex`). The `fedcap`/`capped` guard is retained; a capped pass returns
  ABSTAIN, never HOLDS (`certificate_in` already refuses to conclude from a capped pass — here the
  refusal must be inverted: capped ⟹ ABSTAIN, see trap T5).

---

## 4. Why it explores fewer states — and where it does not

### 4.1 The accounting

D1–D5 all store `Reach` = every reachable symbolic state, measured at 1.00–1.29 zones per
reachable discrete state on six of seven families (18.32 on `grc4`). So the metric reduces to
**discrete states visited**.

QCUT visits
```
CONE(p,q) = the ¬q-forward-closure, inside Inv, of { v : p(v) ∧ ¬q(v) ∧ Inv(v) }
```
and, when `Inv` is exact on the entry cylinder, `CONE(p,q) = ` the `¬q`-closure of
`Reach ∩ p ∧ ¬q`, which is a **subset of `Reach`** because `Reach` is successor-closed. So with
an exact-on-the-cylinder `Inv` the saving is *never negative in discrete states*, and is
`|Reach| − |CONE|` states. **[proved]**

Zone count: seeds cost exactly 1 zone each (Lemma 1). Cone-only discrete states can carry more
than one zone; that is the residual risk and it is what the experiment measures.

### 4.2 The numbers

Measured on an untimed abstraction of the models. Method, so it is reproducible: parse the
UPPAAL XML; keep locations, integer guards (`var OP const`), integer assignments, channels and
the broadcast flag; **drop every clock guard, invariant, reset, and the committed/urgent flags**
— each omission only *adds* behaviour, so the resulting discrete reachable set `R`
over-approximates `Reach_disc`. Then BFS from the initial state with local / binary-sync /
broadcast successor rules, and compute `|R|`, the entry set `{s ∈ R : p(s) ∧ ¬q(s)}`, and its
`¬q`-forward-closure. The abstraction is **validated**: its `|R|` reproduces the published
`|Disc|` exactly on `arbiter4` (48), `arbiter5` (112), `csma4` (104), `csma5` (272), `phil4` (80).
On `fischer`, `twolock` and `grc` it over-approximates (clock guards do constrain the discrete
reachable set there), so those rows are stated as **ratios**, not counts. **[measured]**

| model / query | stored, D1–D5 | `|R|` untimed | `|CONE|` untimed | cone fraction | QCUT predicted stored |
|---|---:|---:|---:|---:|---:|
| `arbiter4 C0.cs --> C0.idle` | 48 | 48 | 8 | 16.7 % | **≈ 8** |
| `arbiter4 C1.cs --> C1.idle` | 48 | 48 | 8 | 16.7 % | **≈ 8** |
| `arbiter4 Arb.Busy --> Arb.Free` | 48 | 48 | 32 | 66.7 % | **≈ 32** |
| `arbiter5 C1.cs --> C1.idle` | 112 | 112 | 16 | 14.3 % | **≈ 16** |
| `arbiter5 Arb.Busy --> Arb.Free` | 112 | 112 | 80 | 71.4 % | **≈ 80** |
| `phil4 P0.Eat --> P0.Dropped` | 82 | 80 | 9 | 11.3 % | **≈ 9** |
| `fischer4 P0.req --> P0.wait` | 222 | 752 | 160 | 21.3 % | **≈ 47** |
| `fischer5 P1.req --> P1.wait` | 762 | 3 552 | 768 | 21.6 % | **≈ 165** |
| `twolock4 P1.reqA --> P1.waitA` | 286 | 9 168 | 1 152 | 12.6 % | **≈ 36** |
| `grc4 T0.in --> T0.far` | 2 345 | 324 | 108 | 33.3 % | **≈ 780** |
| `csma4 Bus.Active --> Bus.Idle` | 128 | 104 | 88 (227 with the real `Inv`) | 84.6 % | **LOSS, ≈ 280** |
| `csma5 Bus.Active --> Bus.Idle` | 352 | 272 | 240 (697 with the real `Inv`) | 88.2 % | **LOSS, ≈ 900** |

The "predicted stored" column is the falsifiable commitment of §7: cone fraction × current
stored, rounded.

Asymptotics on `arbiter_n` **[proved]**: `|Disc| = 2^n + n·2^{n-1}`, `|CONE(C1.cs, C1.idle)| =
2^{n-1}`, so the ratio is `n + 2`. `arbiter10` → 12×, `arbiter15` → 17×.

### 4.3 The worst case, measured not imagined

**`csma` is a real, measured loss.** `Bus.Active --> Bus.Idle` needs the invariant
"`Bus.Active ⟹ exactly one sender in Send`". That invariant *is* inductive, but it is **not** a
place invariant: the broadcast `cd!` moves a *variable* number of tokens `Send → Retry`, and no
linear form is preserved across a variable-arity transition. Generator (L) therefore returns
nothing useful, (T) is vacuous (`csma` has no integer variables), `SEED` degenerates to the whole
`p ∧ ¬q` cylinder (81 states on `csma4` against 32 reachable), and the cone blows past `|Reach|`:
**227 > 104** on `csma4`, **697 > 272** on `csma5`. **[measured]**

Three further worst cases, all structural:

* **`q` nearly always false.** Then `¬q` is nearly everything, the cone is nearly `Reach`, and
  QCUT is the current algorithm plus the invariant overhead. `arbiter Arb.Busy --> Arb.Free`
  already shows this shape at 71 %.
* **A large `p`-cylinder with few reachable members and a weak `Inv`.** Then `|SEED| ≫
  |Reach ∩ p ∧ ¬q|` and QCUT loses before it starts. The budget `B` is the only defence.
* **VIOLATED queries.** QCUT cannot answer them; run unconditionally it is pure overhead. This is
  why the router puts a budgeted D5 first.

There is also a **precision** worst case distinct from the state-count one: the seed zone `⊤ ∧ I_v`
discards every correlation between clocks that reachability would have established. On `fischer`
the satisfied proof depends on such a correlation (`x0 ≤ 2` at `P0.req` bounds how much time can
pass, hence how many `x_j > 2` steps the other processes can take). The seed keeps `x0 ≤ 2`, so
the argument survives **[argued, not measured]** — but if the same widening were applied at
*every* cone node rather than only at seeds, it would not: each re-widening would restore `x_j`
to `⊤` and manufacture an infinite `¬q` cycle. **Do not widen inside the cone.** (On `arbiter` it
happens to survive full widening; on `fischer` it provably does not.)

### 4.4 The uncomfortable comparison: symmetry

The stabiliser of the query, applied to the *same* untimed abstraction. Method: processes are
interchangeable when their location-name list, initial location and multiset of edge signatures
`(src, tgt, channel, direction, int-guard, int-assignment)` coincide; the **stabiliser** drops
from each such class every process named in `p` or `q`; a state is canonicalised by sorting the
locations within each remaining class. On `arbiter5 C1.cs --> C1.idle` this reproduces the
hand-count 23 exactly, which validates it. **[measured]**

| query | `|R|` | `|R/G|` | `|CONE|` | `|CONE/G|` |
|---|---:|---:|---:|---:|
| `arbiter4 C1.cs --> C1.idle` | 48 | 18 | 8 | **4** |
| `arbiter5 C1.cs --> C1.idle` | 112 | 23 | 16 | **5** |
| `arbiter5 Arb.Busy --> Arb.Free` | 112 | 11 | 80 | **5** |
| `csma5 Bus.Active --> Bus.Idle` | 272 | 20 | 240 | **14** |
| `grc4 T0.in --> T0.far` | 324 | 120 | 108 | **40** |
| `fischer4 P0.req --> P0.wait` | 752 | 752 | 160 | **160** |

Read it honestly:

* Symmetry alone beats the cone alone on `Arb.Busy --> Arb.Free` (11 vs 80) and on `csma5`
  (20 vs 240) — the two queries where the cone is weakest.
* The cone alone beats symmetry alone on `C1.cs --> C1.idle` (8 vs 18) and on `fischer`, where the
  `id` constants break the plain process symmetry entirely (a scalarset permutation would be
  needed).
* **They compose multiplicatively.** 112 → 5 on `arbiter5 C1.cs --> C1.idle` is 22×.

Symmetry reduction is textbook (§6) and `src/symmetry.cpp` already computes the automorphism
group and `netreach.cpp:769` already canonicalises. Two things are *not* done and would be needed:
(i) restricting the group to the **stabiliser** of the processes named in `p` and `q` — the query
here is asymmetric, so the standard "symmetric formula" theorems do not apply directly; (ii)
making `gfp_pre` aware that a quotient edge is a real edge **composed with a permutation**, so the
target zone must be un-permuted before `Pre` — this is precisely the class of bug commit
`41eaaf4` ("quotient edges") already had to fix once.

A useful observation that does survive as (mildly) new: the invariant layer is
**query-independent**, so the symmetry quotient may be used to compute `Inv` *even when the query
is asymmetric*. You get the reduction where it is free and pay full price only in the cone.

---

## 5. Traps

The code comments record bugs that cost hours. Which apply, and one new one.

**T1 — `adj[v].empty()` must read the FULL successor list. NEW, and the most dangerous.**
`leads_to_net` already records this (`liveness-subsumption-bug.md` §5: "`adj[u].empty()` still
reads the FULL adjacency — a state whose only successors were absorbed is *not* deadlocked").
QCUT cuts *two* kinds of successor (`q` and `¬Inv`), so a node whose only successors are `q`-states
would report `adj = ∅`, `stop` would fire, `W` would be the whole zone, and every seed upstream of
it would report a spurious witness — turning a HOLDS into an ABSTAIN on essentially every query.
`full_succ_empty[u]` is computed before either filter, in the same pass, at zero extra cost.

**T2 — the active-clock reduction must be inverted, not disabled.** `gfp_pre:3679` frees every
clock inactive at the *target*. Without it `Pre` silently demands those clocks be 0 at the source,
`W` collapses to ∅, and **violations are missed** — measured on `phil4/6`, `grc4`, `arbiter4/5`
(`03-liveness-algorithm.tex` §activeclock). QCUT keeps `gfp_pre` byte-identical. Note the
*direction* of that failure: it collapses `W`, i.e. it makes HOLDS *easier*. Under QCUT, whose only
verdict is HOLDS, a broken `Pre` is therefore **silently unsound**, not merely incomplete. This is
a genuine regression in blast radius and the ablation `PTCTLMC_GFPNOPD`-style switch must be kept
and exercised.

**T3 — the delay before the action.** The outer `Y.down()` at `gfp_pre:3712`. A stored zone is
delay-closed *as a set*, so the forward operator may write `Z ∧ g`; an individual valuation may
have to let time pass before the guard opens. Unchanged.

**T4 — whole-zone vs per-valuation stuck test.** `stop` must stay whole-zone/per-node (Cor.
`cor:whole`). QCUT makes this *worse*, not better: the flag is evaluated on `seed(v)`, the largest
zone at `v`. Expect more spurious `stop` firings and therefore more ABSTAINs than Algorithm 1
would have. Measured impact is exactly what §7's experiment reports.

**T5 — `capped` inverts.** `Otf::certificate_in` returns `false` when the federation cap is hit,
because "an unusable pass proves nothing" and its output is a *violation* certificate. QCUT's
output is a *holds* certificate, so a capped pass must return **ABSTAIN**, not HOLDS. Getting this
backwards yields a silent unsound "satisfied" on any query that hits the cap — `grc4` did hit a
256-cap historically.

**T6 — `Dbm(nclocks)` is not `⊤`.** `HDBM/include/ptctlmc/dbm.hpp:37` documents the `int`
constructor as *"the zone in which every clock is 0"*, and `build_all` relies on that (it builds
`Dbm(net.nclocks)` and then *delays* it). `seed(v)` needs the universal DBM: use
`Dbm::universe(nclocks)` (`dbm.hpp:47`). Writing `Dbm(n)` in `seed` would silently seed every
entry with `x = 0`, an **under**-approximation — the one direction that breaks Theorem 1, and one
that would produce a confident, silent, wrong "satisfied".

**T7 — subsumption closing unrunnable cycles.** Does **not** apply. That is the whole point of the
branch: the redirected edge enters the backward operator only as an upper bound. QCUT keeps aLU
subsumption in the cone for exactly this reason (`04-soundness` Cor. `cor:alu`).

**T8 — a stale comment in the existing code.** `leads_to_gfp:3750` and `Otf::setup:5104` both say
"ACTIVE-CLOCK REDUCTION IS DISABLED HERE (qmask all-ones keeps every clock live)" and then write
`qmask(net.nclocks + 1, 0)` — **all zeros**. `reset_inactive` treats `qmask[c] = 1` as *protect*,
so all-zeros means the reduction is fully **enabled**. The `.tex` (§`sec:liveness:activeclock`,
which describes the reduction as applied forward and inverted in `Pre`) matches the code; the
comment does not. Anyone building `seed(v)` from that comment will build the wrong zone. Worth
fixing independently of this proposal.

---

## 6. Prior art

An adversarial search was run against this design. Every ingredient is published.

**Greatest fixpoints for liveness, and reachable-set restriction.** `EG` as a gfp and fair-`EG` as
a nested fixpoint is Emerson & Lei (LICS 1986); Somenzi, Ravi & Bloem (FMCAD 2002) prove EL
complexity-optimal among GSH-derivable algorithms. Restricting CTL fixpoints to the
forward-reachable set is default behaviour in VIS and a flag in NuSMV, and is the subject of Moon,
Jang, Hachtel, Somenzi, Yuan & Pixley (ICCAD 1998). Reachability-then-shrinking-fixpoint in
parallel is OWCTY (Fisler, Fraer, Kamhi, Vardi, Yang, TACAS 2001). Chaotic iteration is Cousot
(1977). **All already recorded in `07-related.tex`; nothing here is claimed.**

**Backward timed analysis.** The backward TCTL algorithm over zones is Henzinger, Nicollin,
Sifakis & Yovine (*Symbolic model checking for real-time systems*, 1994) and the Kronos lineage;
its distributed form is ZEUS (Braberman, Olivero, Schapachnik, PDMC/CONCUR 2002; FMSD 2006).
Backward TCTL computes `EG` over the whole state space *including unreachable regions*, which is
the classical reason it lost to forward search. QCUT is a *forward* closure inside `¬q` seeded
from a `p`-cylinder — the carrier is different — but the family resemblance is close and must be
stated.

**Under-approximating a greatest fixpoint.** Farn Wang, *Under-approximation of the Greatest
Fixpoint(s) in Real-Time System Verification* (arXiv cs/0501059, cs/0501060), and the EDGF
("Early Decision on the Greatest Fixpoint") technique in his TCTL-inevitability work
(*TCTL Inevitability Analysis of Dense-Time Systems*, CIAA 2003 / arXiv cs/0304003), test
emptiness of the intersection with the target at every iteration so as to stop early. **This is
prior art for the branch's own D3/D5 early-exit, not only for QCUT, and `07-related.tex` does not
currently cite it. It should.**

**Timed liveness and subsumption.** Tripakis, *Checking timed Büchi automata emptiness on
simulation graphs* (TOCL 2009); Herbreteau, Srivathsan, Tran & Walukiewicz, *Why liveness for
timed automata is hard, and what we can do about it* (FSTTCS 2016 / TOCL 2020); Laarman, Olesen,
Dalsgaard, Larsen & van de Pol, *Multi-core emptiness checking of timed Büchi automata using
inclusion abstraction* (CAV 2013). The shared premise — subsumption is unsound for Büchi
emptiness — is the one the branch's backward direction sidesteps. Unchanged by this proposal.

**Cone of influence / live components / slicing for timed automata.** Live-component analysis is
explicitly a cone-of-influence abstraction for timed models (Salah/…, *Speeding Up Model Checking
of Timed-Models by Combining Scenario Specialization and Live Component Analysis*, FORMATS 2009);
*Improving the Verification of Timed Systems Using Influence Information* (TACAS 2002); `ObsSlice`
is an observer-based slicer for timed automata. **The idea "restrict the search using the
property's cone" is squarely prior art.** What those tools slice is the *model*; what QCUT
restricts is the *carrier of one fixpoint*, which is a different object but not a different idea.

**Localization reduction / CEGAR for timed automata.** Abstract away clocks and discrete variables
and reintroduce them to kill spurious counterexamples — *Abstraction Refinement Algorithms for
Timed Automata* (arXiv 1905.07365); Dierks, Kupferschmid & Larsen (FORMATS 2007). This is the
natural competitor to QCUT's `Inv` layer and would likely subsume it if implemented.

**Compositional invariant generation, which is where `Inv` comes from.** D-Finder (Bensalem,
Bozga, Nguyen, Sifakis, CAV 2009 and *Compositional Verification for Component-Based Systems*,
ATVA 2008) computes component invariants plus **interaction invariants that "correspond to traps
of the Petri net and are computed symbolically as solutions of a set of boolean equations"**, and
proves deadlock-freedom *without* state-space exploration, falling back to exploration only for
the few surviving configurations. Petri-net place invariants by linear algebra are folklore
(Murata 1989); Sankaranarayanan, Sipma & Manna, *Petri Net Analysis Using Invariant Generation*
(VMCAI 2004). **Generator (L) is a P-invariant computation and generator (T) is a thread-modular
component invariant. Nothing about the invariant layer is new; the only design decision is using
it as a `Reach` *replacement* rather than as a deadlock check.**

**Proving liveness by invariants rather than exploration.** Fang, Piterman, Pnueli & Zuck,
*Liveness by Invisible Invariants* (FORTE 2006) and *Liveness with Invisible Ranking*, which
target exactly **bounded response — "once a `p`-state is reached, it takes a bounded number of
rounds to reach a `q`-state"**, i.e. `p --> q`, and discharge it by generated invariants and
ranking rather than by state exploration. Untimed and parameterized, and the invariants are
*candidates* validated by a small-model theorem rather than inductive by construction, but the
strategic move is the same one.

**Symmetry.** Emerson & Sistla (1993), Ip & Dill (1996), and for this exact setting Hendriks,
Behrmann, Larsen, Niebert & Vaandrager, *Adding Symmetry Reduction to Uppaal* (FORMATS 2003) with
scalarsets. §4.4's numbers are a re-measurement of a known technique.

**Liveness-to-safety and k-liveness.** Biere, Artho & Schuppan (FMICS 2002), Schuppan & Biere for
infinite state; Claessen & Sörensson (FMCAD 2012); Cimatti, Griggio, Mover & Tonetta apply
k-liveness to hybrid systems (CAV 2014). Quantitatively these go the **wrong way** here: the
Biere–Artho–Schuppan encoding doubles the state vector (a saved copy plus a `looped` bit), which
in a zone setting means doubling the clocks — every stored DBM grows from `(n+1)²` to `(2n+1)²`
and the discrete state space squares. k-liveness needs a counter and repeated safety checks over
the *whole* reachable space. Both are strictly worse than `|Reach|` on these families and are
rejected on that ground alone. **[argued]**

### Verdict

The one sentence this document is entitled to:

> For a zone-based `p --> q` check whose backward greatest fixpoint is confined to a forward
> graph, the confining set can be a product-free inductive **discrete invariant** rather than the
> forward-reachable set, and the fixpoint's carrier can be restricted to the `¬q`-closure of the
> `p`-cylinder — which removes reachable discrete states from the search rather than merely
> zones from a discrete state.

The search found no paper stating that. It also found that each half of it is standard, and that
the closest untimed relative (*Liveness by Invisible Invariants*) targets literally the same
`p --> q` shape by literally the same strategic move. **Claim it as an engineering composition,
not as a theorem, and lead with the measurement rather than the novelty.**

---

## 7. Implementation sketch, and the experiment that would refute it

### 7.1 What to add

New file `nove_mc/prototype/qcut.cpp` (or an appended section of `netreach_gfp.cpp`, matching
how `Otf` was added), plus `OtfMode::D6`:

1. `struct DiscInv` — `std::vector<std::vector<Fraction/long>> pinv_basis`, `std::vector<long>
   pinv_const`, `std::vector<std::unordered_set<uint64_t>> tm[j]`. One method
   `bool holds(const LocVec&, const IntVec&) const`. ~150 lines.
   * `place_invariants(net)` — build the incidence matrix over `Σ|L_i|` places from
     `net.procs[i].edges` (local / binary via `sync_chan,sync_dir` / broadcast via
     `net.chan_broadcast`), rational Gaussian elimination, return the nullspace basis. ~80 lines.
   * `thread_modular(net, j)` — BFS over `(loc_j, IntVec)` using `int_ok` and the existing
     `IntAsgn` application. Skip when `net.intvars.empty()`. ~50 lines.
   * Fall back to `holds() == true` (no pruning) for any construct not analysed
     (`gen_guard`/`gen_update`), exactly as `compute_int_live` does.
2. `seed(v)` — 12 lines, reusing `Dbm::universe`, `all_invariants`, `frozen_flags`,
   `reset_inactive` and `extrapolate_lu_plus`.
3. `enumerate_seeds(net, p, q, Inv, B)` — backtracking over processes with bounds propagation on
   the linear conjuncts. ~120 lines.
4. `class QCut` — a near-copy of `Otf` with three deltas: seeded frontier instead of `build_all`;
   the two filters in `expand`; `full_succ_empty[]` recorded before the filters; `stop_term`
   reading it; verdict inverted (`HOLDS` iff all `W` empty at `p`-nodes; `capped ⟹ ABSTAIN`).
   ~200 lines.
5. `run_otf.py` gains mode `d6`, and reports `ABSTAIN` as a distinct outcome (not a verdict).

Nothing in `src/` changes. `gfp_pre`, `gfp_succ_tagged`, `gfp_on`, `state_has_deadlock`,
`fed_subtract` are used verbatim.

### 7.2 The falsifiable experiment

Run all 48 queries of `nove_mc/prototype/queries.txt` under `d3`, `d5`, `d6`, plus the 12 rows of
§4.2, with `PTCTLMC_GFPALU=1`. Record verdict, `stored`, and for `d6` also `|SEED|`, `|CONE|`
discrete keys, zones/disc, and ABSTAIN reason.

**Any one of these refutes the proposal:**

* **R1 (soundness).** `d6` returns HOLDS on any query where `leads_to_gfp`, `leads_to_net` or
  `verifyta -u -S0` says NOT satisfied. One instance kills it outright.
* **R2 (precision).** `d6` ABSTAINs on more than 2 of the 9 satisfied queries in
  `queries.txt`. The whole bet is that the maximal seed zone `⊤ ∧ I_v` is precise enough; if
  `state_has_deadlock` or `time_diverges` fires spuriously at seeds, it is not, and the design is
  dead as stated. (`csma4`/`csma5` are *predicted* to abstain or lose and are excluded from the
  count of 2.)
* **R3 (state count).** `d6`'s `stored` is ≥ `d3`/`d5`'s on any of `arbiter4 C0.cs-->C0.idle`,
  `arbiter4 C1.cs-->C1.idle`, `arbiter5 C1.cs-->C1.idle`, `phil4 P0.Eat-->P0.Dropped`,
  `fischer4 P0.req-->P0.wait`, `twolock4 P1.reqA-->P1.waitA`, `grc4 T0.in-->T0.far`.
* **R4 (the numbers).** `d6`'s `stored` exceeds the §4.2 prediction by more than 2× on any of
  those seven. The predictions are: 8, 8, 16, 9, 47, 36, 780.
* **R5 (zones per discrete state).** `d6` stores more than 2.0 zones per cone discrete state on
  any family other than `grc4`. Lemma 1 guarantees 1.00 at seeds; if the cone-only states inflate,
  the discrete saving is eaten by zones.
* **R6 (the invariant is free).** Phase 0 wall-time exceeds 5 % of `d5`'s total on any model up to
  `arbiter15` / `fischer10`. It should be microseconds.

**Scaling check, which no existing lane can do.** `arbiter6…arbiter12` `C1.cs --> C1.idle`:
`leads_to_net` cannot finish `arbiter6` in 15 minutes and `leads_to_gfp` pays `2^n + n·2^{n-1}`.
QCUT predicts `2^{n-1}` — 32, 64, …, 2048 — i.e. a factor `n+2`. A measured departure from
`2^{n-1}` at any `n` refutes the asymptotic claim independently of the small instances.

**Ablations that must be run.** (a) `Inv` off — predicts the measured `cone_free` blow-up (54, 162,
227, 1080) and therefore a loss on every model, confirming that the invariant layer, not the cone,
is doing the work on `arbiter`; (b) seed zone `= ⊤` at *every* node, not only seeds — predicts
HOLDS still on `arbiter` and ABSTAIN on `fischer` (§4.3); (c) `full_succ_empty` computed *after*
the filters instead of before — predicts ABSTAIN nearly everywhere (trap T1), and is the cheapest
way to confirm the trap is real rather than theoretical.

---

## 8. Rejected alternatives, with reasons

The brief named six angles. Each was examined; five are rejected here, with the reason and, where
it exists, the measurement.

**8.1 Merging discrete states (the crux question: an abstraction coarser than aLU).**
The general licence is real and worth stating, because it is the correct generalisation of
`04-soundness`: *any* map `h` on symbolic states with (i) zone containment, (ii) every concrete
transition covered by an abstract edge with an over-approximating `Pre`, (iii)
`stop_abs ⊇ ⋃ stop_conc`, (iv) `h` refines `p` and `q`, preserves `Ω ⊆ W` and hence the HOLDS
verdict. aLU is the special case where `h` is the identity on the discrete component; coarsening
the *discrete* component is the only way below 1 zone per discrete state. **[proved]**
It nonetheless fails on this suite, for a specific and measurable reason: condition (iii). Every
cheap discrete merge — chaotic-environment process projection, Cartesian (independent-attribute)
abstraction, pairwise products — loses the mutual-exclusion invariant, and the abstract state
`(Arb.Free ∧ C1.cs)` it admits is a **partial deadlock** (`x1 = 4`, `rel!` unavailable), so `stop`
fires and the fixpoint is non-empty everywhere. The measured proxy: the un-pruned `p ∧ ¬q`
cylinder *is* the Cartesian over-approximation of the entry set, and it gives 162 (vs 112) on
`arbiter5` and 227 (vs 104) on `csma4`. **[measured]** QCUT therefore **deletes** discrete states
rather than merging them; deleting is the only cheap discrete coarsening that survives
condition (iii).

**8.2 Per-process backward fixpoint over `network.hpp:105`-style location analyses.**
Useless as a *pruner* on these families: every process can reach every one of its locations in its
own graph, so the "can still reach `p`" filter is vacuous. It is however exactly the shape of
generator (T), which *is* used — over `(location, integer vector)` rather than location alone,
which is where the fischer/twolock invariants come from. **[measured]** — (T) gives
`spurious_entry = 0` on `fischer4/5`, `twolock4`.

**8.3 Liveness-to-safety and k-liveness.** Rejected quantitatively in §6: the BAS encoding
doubles the clock vector (DBM area ×4, discrete space squared); k-liveness needs repeated safety
checks over the whole reachable space. Both exceed `|Reach|`, which is the number to beat.

**8.4 Lazy `a≼LU` / adaptive simulation.** Orthogonal and worth doing *separately*: it attacks
zones per discrete state, which is already 1.00–1.29 everywhere except `grc4`. `grc4`'s 18.32 is
the one place it would pay, and it is an 18× lever on the largest satisfied query in the suite.
Recommended as an independent work item, not as an answer to the satisfied-query problem.

**8.5 Backward reachability confirmation of an abstract witness.** "`Ŵ_v ≠ ∅` ⟹ ask
`E<> (v ∧ Ŵ_v)`; SAT ⟹ VIOLATED" would make QCUT complete on its own. **Unsound**, by (3) of
§2.4: a reachable valuation may be in `Ŵ_v` only because `state_has_deadlock` fired on the
oversized seed zone. The UNSAT direction is sound but costs a full reachability sweep, defeating
the purpose. Rejected.

**8.6 Bisimulation quotient before the analysis.** `src/bisim.cpp` computes a zone-based
(bi)simulation over the *product* of two automata — a `|X_A| × |X_B|` federation fixpoint. Using
it to quotient the model before liveness costs more than the liveness analysis it would speed up.
Symmetry (§4.4) is the cheap special case and is the one to use. **[argued]**

---

## 9. What this document does not claim

* It does not claim a decision procedure. QCUT answers HOLDS or ABSTAIN.
* It does not claim any improvement on VIOLATED queries. D3/D5 own those and are unchanged.
* It does not claim novelty for the invariant layer, for the cone idea, for symmetry, or for
  restricting a fixpoint to a subset. §6 attributes all four.
* It does not claim the `arbiter`-style saving generalises. It is measured to **fail on `csma`**,
  in this suite, for a reason (broadcast defeats place invariants) that is structural and will
  recur.
* The 12 predictions of §4.2 are derived from an untimed abstraction that is exact on
  `arbiter`/`csma`/`phil` and over-approximate on `fischer`/`twolock`/`grc`. The `fischer`,
  `twolock` and `grc` rows are ratio-extrapolations and should be read as the weakest evidence
  here.
