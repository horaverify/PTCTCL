# G-simulation for timed automata — implementable extraction

Literature extraction for implementation. Every formula below is traced to a source; anything
I derived myself is explicitly marked **[DERIVED]**. Where the literature is silent I write
**NOT ESTABLISHED IN SOURCES**.

## Sources used (all fetched and read in full)

| Tag | Document | How cited below |
|---|---|---|
| **[SURVEY]** | Bouyer, Gastin, Herbreteau, Sankur, Srivathsan, *Zone-based verification of timed automata: extrapolations, simulations and what next?* arXiv:2207.07479 (FORMATS'22 invited). | page numbers of the arXiv PDF (27 pp.) |
| **[CAV19]** | Gastin, Mukherjee, Srivathsan, *Fast algorithms for handling diagonal constraints in timed automata*, arXiv:1904.08590 (CAV'19). | definition/lemma numbers |
| **[CONCUR18]** | Gastin, Mukherjee, Srivathsan, *Reachability in timed automata with diagonal constraints*, arXiv:1806.11007 (CONCUR'18). | — |
| **[THESIS]** | S. Mukherjee, *Reachability in Timed Automata with Diagonal Constraints and Updates*, PhD thesis, CMI, 2022. `https://mukherjee-sayan.github.io/files/sayan-thesis.pdf` | chapter/section + numbered result |
| **[FSTTCS20]** | Gastin, Mukherjee, Srivathsan, *Reachability for updatable timed automata made faster and more effective*, arXiv:2009.13260. | — |

[SURVEY] reference `[51]`/`[52]` = [CAV19]; `[79]` = [THESIS]; `[50]` = [CONCUR18]; `[53]` = [FSTTCS20].
[THESIS] is by far the most detailed source and is the one to implement from: [SURVEY]
Definition 2 = [THESIS] Definition 3.24, [SURVEY] Definition 3 = [THESIS] Definitions 3.1/3.15,
[SURVEY] Theorem 3 = [THESIS] Theorems 3.23 + 4.1-series + 4.26.

---

## TL;DR — headline answers

**§4 (networks).** Nobody computes `G` over the product. Upstream TChecker does not implement
the `G`-simulation at all (only `a_LU`/`a_M`). The only implementation — Mukherjee's unmerged
fork `mukherjee-sayan/tchecker`, `-a gsim` — computes the map **per process**, indexed by a
single process's location id, by a worklist over that process's own edges, and assembles the
value for a location vector by **plain union** of the per-process sets, recomputed on every
subsumption check. Upstream does the same shape for LU bounds with **pointwise max**. Neither
builds the product; neither does it lazily during search. The papers never mention networks.

**§5 (is the compositional union sound?).** **Yes for diagonal-free networks, unconditionally —
even with clocks shared and reset across processes** (Prop. 5.1, proved from the definition of
`pre`). With diagonal constraints the plain union is **unsound in general** (Prop. 5.2, explicit
counterexample); it is sound under clock ownership (Prop. 5.1b) or, unconditionally, after a
cheap `pre`-saturation of each per-process map (Prop. 5.3, ≤3× and only on diagonals). The
existing fork does the plain union with no cross-process handling — an unaddressed gap for the
very models `gsim` exists for.

**§A (monotonicity).** The premise "a larger `G` is finer but still sound" is **half right**.
*Precision*: yes, monotone, `G ⊇ H ⟹ ⊑_G ⊆ ⊑_H` ([THESIS] Prop. 3.2) — larger prunes less.
*Soundness*: **NO — refuted by counterexample.** Supersets of `G*` are not sound in general.
The correct invariant is not "≥ `G*`" but "**is an A-map**": guard-containing and closed under
`pre` ([THESIS] Def. 3.22 + Thm. 3.23). `G*` is the ⊆-least A-map, hence the coarsest sound
choice; every *A-map* above it is sound but weaker. A-maps are closed under ∪ and ∩.

**§B (contraction).** (i) Shrinking is sound at every intermediate point **iff every intermediate
map is still an A-map**; "stay above `G*`" is necessary but insufficient. A removal is legal iff
the constraint is not a guard atom and not a `pre`-image of a *current* successor constraint —
and removals **never create obligations elsewhere**, only enable more removals at predecessors.
(ii) Stored zones need **no re-checking**: nothing stored depends on `G` (zones are stored
exactly — the fork indeed runs `NO_EXTRAPOLATION`), past prunings stay valid, and stored zones
become strictly *stronger* subsumers. Coverage grows on every shrink. Mixed generations across
threads are safe (composition of simulations is a simulation; stale = larger = conservative).
(iii) **No published algorithm contracts the constraint map.** All prior work (CAV'13 lazy
abstraction, CAV'19 abstraction refinement, interpolant-based lazy checking) only **grows**
bounds. **NOT ESTABLISHED IN SOURCES → treat as novel.** One real caveat: descending iteration
converges to the greatest fixpoint *below* the start, not to `G*`; reset-free cycles make
constraints self-justifying, so use DRed/support-counting, not naive re-derivability.

**§C (local shrink).** **Local. No global barrier.** The A-map clauses constrain a state's set
only in terms of its own *outgoing* transitions and its *immediate successors*' sets, and
shrinking a successor only relaxes predecessors' clauses. Legality is a one-step lookahead;
stale (larger) successor reads are conservative; per-stripe background contraction is safe as
long as `G` is published shrink-only (CAS or per-state lock — a resurrected constraint would
break a predecessor's invariant).

---

## 0. Notation and conventions (fix these before reading anything else)

**Atomic constraints.** `φ ::= x ◁ c | c ◁ x | x − y ◁ c | c ◁ x − y` with `◁ ∈ {<, ≤}`,
`c ∈ ℤ` ([THESIS] Def. 2.9). `x ◁ c` and `c ◁ x` are *non-diagonal*; the others *diagonal*.
The zero clock `x₀` (always 0) turns `x ◁ c` into `x − x₀ ◁ c`.

**Bound pairs.** A bound is a pair `(◁, z)`, `◁ ∈ {<,≤}`, `z ∈ ℤ`.
([THESIS] p. 21, verbatim)

```
(≤,z₁) + (≤,z₂) = (≤, z₁+z₂)
(<,z₁) + (<,z₂) = (<,z₁) + (≤,z₂) = (≤,z₁) + (<,z₂) = (<, z₁+z₂)
(◁₁,z₁) < (◁₂,z₂)  iff  z₁ < z₂,  or  z₁ = z₂ and ◁₁ = < and ◁₂ = ≤
```

**DBM indexing — THE ONE THING TO GET RIGHT.** [THESIS] uses a *transposed* index relative to
the usual DBM convention. Verbatim, [THESIS] p. 21:

> "Given a zone `Z`, `Z_xy = (◁_xy , z_xy)` represents the constraint `y − x ◁_xy z_xy` present
> in the zone `Z` … Note that, `Z_xy` denotes the atomic constraint involving the difference
> `y − x` and **not** `x − y`. This convention is chosen to align with a graph representation."

So with the standard DBM `D[i][j] ≜ bound on (x_i − x_j)` — which is what
`HDBM/include/ptctlmc/dbm_ops.hpp` uses (`Z[i*dim+j]`) — the translation is

```
Z_xy  =  D[y][x]          (Gastin index = transpose of DBM index)
Z_x0  =  D[0][x]          ( "−x ◁ z"      : lower bound on x )
Z_0y  =  D[y][0]          ( "y ◁ z"       : upper bound on y )
```

Every DBM formula in §2 below is given in **both** conventions. Do not mix them.

**Canonicity.** All zones are assumed non-empty and canonical (`Z_xy ≤ Z_xw + Z_wy`,
[THESIS] Def. 2.10). Tightness ([THESIS] Prop. 2.11) is used throughout the proofs.

---

## 1. THE CONSTRAINT MAP `G(q)`

### 1.1 `pre(φ, R)` — exact definition, all cases

[THESIS] Definition 3.19 (verbatim): *"Given an atomic constraint `φ` and a set of clocks `R`,
the pre of `φ` under the reset of clocks in `R`, is the constraint `[R]⁻¹(φ)`, where
`[R]⁻¹(φ) = φ[0/x, ∀x ∈ R]`."* — i.e. **syntactic substitution of 0 for every reset clock**,
then simplification. [SURVEY] writes this as `pre(φ, Y)`; [CAV19] Definition 7 writes it as
`wp(⊑_φ, R)`. The three agree. Case table (merging [SURVEY] p. 13, [CAV19] Def. 7,
[THESIS] Def. 3.19):

| `φ` | condition | `pre(φ, R)` |
|---|---|---|
| `x ◁ c` (upper, non-diag) | `x ∉ R` | `{x ◁ c}` |
| `x ◁ c` | `x ∈ R` | `∅` (becomes `0 ◁ c`, a constant) |
| `c ◁ x` (lower, non-diag) | `x ∉ R` | `{c ◁ x}` |
| `c ◁ x` | `x ∈ R` | `∅` (becomes `c ◁ 0`, a constant) |
| `x − y ◁ c` | `x ∉ R, y ∉ R` | `{x − y ◁ c}` |
| `x − y ◁ c` | `x ∉ R, y ∈ R` | `{x ◁ c}` if `c ≥ 0`, else `∅` |
| `x − y ◁ c` | `x ∈ R, y ∉ R` | `{−c ◁ y}` if `−c ≥ 0`, else `∅` |
| `x − y ◁ c` | `x ∈ R, y ∈ R` | `∅` |

and `pre(G, R) = ⋃_{φ∈G} pre(φ, R)` ([CAV19] Def. 7, last line).

Reading notes that matter for the implementation:

* The `c ≥ 0` / `−c ≥ 0` side conditions are [CAV19] Definition 7. They are exactly the case
  where the substituted constraint is *vacuous over ℝ≥0* (always true, or never true). Both
  vacuous cases are correctly dropped: a constraint that no valuation ever satisfies, and a
  constraint every valuation satisfies, impose nothing on `⊑`. [SURVEY] p. 13 and [THESIS]
  Def. 3.19 fold these into "`∅`" / "`⊤`,`⊥`". **Dropping them is sound**, see the `x ∈ R`
  branches of the proof of [THESIS] Lemma 3.20 (`[R]v(x) = [R]v'(x) = 0` makes the requirement
  hold unconditionally).
* `{−c ◁ y}` is the lower-bound form: `−y ◁ c ⟺ −c ◁ y`. [SURVEY] writes `{−y ⋈ c}`,
  [CAV19] writes `{−c ◁ y}`. Same thing. Note the constant can be *negative* here, which is
  why §2 keeps `d ∈ ℤ` rather than `d ∈ ℕ`.
* `pre` never invents a new constant ([THESIS] p. 42, [CAV19] p. 8) — this is the termination
  argument, §1.4.

**Updates (UTA).** [SURVEY] p. 14: replace `(q,g,R,q′)` with `(q,g,up,q′)` and `pre(φ,R)` with
`pre(φ,up)` where, writing `up_x = c` (resp. `y+d`) when `up` maps `x` to `x:=c` (resp.
`x:=y+d`), *"the constraint `pre(φ, up)` is obtained by replacing occurrence of `x` in `φ` with
`up_x`"*. Example from [SURVEY]: `pre(x − y ≤ 5, x := z − 2) = {z − y ≤ 7}`. Under updates the
least fixpoint **may be infinite** ([SURVEY] p. 14, [CAV19] §7 example: self-loop
`(q, x ◁ c, x := x−1, q)` forces `x ◁ c+1, x ◁ c+2, …`). [CAV19] Definition 24 additionally
requires `wp(⊑_{x≥0 | x∈X}, up) ⊆ G(q)` so that `up(v′)` stays a valid (non-negative)
valuation. **If PTCTLMC has no updates, ignore this paragraph entirely.**

### 1.2 The fixpoint

[SURVEY] **Definition 2 (Constraint map)**, p. 13, verbatim:

> "The constraint map `G` of `A` associates a set `G(q)` of constraints to every state `q ∈ Q`.
> It is obtained as the **least fixpoint** of the following system of equations:
> ```
> G(q) = ⋃_{(q,g,Y,q′)∈T} ( {atomic constraints of g} ∪ ⋃_{φ∈G(q′)} pre(φ, Y) )
> ```"

Identical to [THESIS] Definition 3.24 ("state based guards") and [CAV19] Definition 8.

### 1.3 What actually has to hold — the A-map (this is the key to §5/§A/§B)

The least fixpoint is *not* what the soundness theorem needs. [THESIS] Definition 3.22
(verbatim):

> "An **A-map** is a tuple `G := (G(q))_{q∈Q}`, where each `G(q)` is a set of atomic
> constraints associated with the state `q`, satisfying the following two conditions: for every
> state `q` of `A` and for every transition `t = (q, g, R, q′)` of `A`,
> – `G(q)` contains the atomic constraints present in `g`,
> – for every `φ ∈ G(q′)`, `G(q)` contains the atomic constraint `[R]⁻¹(φ)`."

[THESIS] **Theorem 3.23**: *"Given a timed automaton `A`, possibly containing diagonal
constraints, the relation `⊑_G` … is a simulation relation, **if** the tuple `(G(q))_{q∈Q}` is
an A-map."* Same statement in [CAV19] Lemma 10, whose proof uses only `g ∈ G(q)` and
`wp(⊑_{G(q₁)}, R) ⊆ G(q)`.

**Consequence (used everywhere below):** the A-maps are exactly the *post-fixpoints* of the
equation system of §1.2, i.e. every `G` with `G(q) ⊇ F(G)(q)`. The least fixpoint `G*` is the
**smallest** A-map (Knaster–Tarski: `lfp ⊆` every post-fixpoint), hence — by [THESIS]
Proposition 3.2 — the **coarsest** simulation in this family. Soundness is a *closure*
property, not a size property. See §A.

### 1.4 Pseudocode (worklist, backwards over transitions)

Direct implementation of [THESIS] Algorithm 2 (p. 42) turned into a worklist. `pred(q)` is the
set of transitions *into* `q`.

```
compute_G(A = (Q, X, q0, T)):
  for q in Q:  G[q] := ∅
  W := ∅                                     # worklist of states whose G changed
  for t = (q, g, R, q') in T:                # seed: guards (and invariants, see note)
      new := atoms_of(g)  \ G[q]
      if new ≠ ∅: G[q] |= new;  W.push(q)
  while W ≠ ∅:
      q' := W.pop()
      for t = (q, g, R, q') in pred(q'):      # backward step
          new := { pre(φ, R) : φ ∈ G[q'] } \ G[q]      # pre returns ∅ or a singleton
          if new ≠ ∅: G[q] |= new;  W.push(q)
  return G
```

Two implementation notes that the papers do not spell out:

* **[DERIVED] Invariants.** [THESIS]/[CAV19] model timed automata *without* location
  invariants; the simulation conditions are stated for guards only. If your model has
  invariants `I(q)` (UPPAAL/TChecker do), the simulation must also preserve them
  (`v ⊨ I(q) ⇒ v′ ⊨ I(q)`, and after the transition `[R]v ⊨ I(q′) ⇒ [R]v′ ⊨ I(q′)`). The
  standard encoding — fold `I(q) ∧ I(q′)[R]` into the guard `g` of each transition and add
  `atoms_of(I(q))` to `G(q)` for every `q` — restores exactly [THESIS] Lemma 3.18. Not stated in
  any of the papers, but **confirmed by the reference implementation**: `compute_amap` seeds
  `G[loc]` from `system.invariant(loc->id())` before seeding `G[src]` from guards
  (`mukherjee-sayan/tchecker`, `src/clockbounds/solver.cc:1697-1711`, see §4.3).
* Only one **upper** and one **lower** bound per clock ever needs to be *kept* among the
  non-diagonal constraints (see §3.2, [THESIS] Cor. 3.6 and 3.10) — dedupe aggressively.

### 1.5 Termination and complexity

**Termination.** [THESIS] Theorem 3.26 + p. 42: *"none of the steps in Algorithm 2 generates a
constraint with a new constant"*; [CAV19] p. 8–9: *"All constraints present in the set
`wp(⊑_{G(q₁)}, R)` contain constants which are already present in `⊑_{G(q₁)}` … Since no new
constants are generated in this process, the fixed point computation terminates."*
Formally: let `C` be the finite set of constants occurring in guards of `A` together with their
negations. Every constraint ever produced lies in the finite universe
`Φ_C = { x ◁ c, c ◁ x, x − y ◁ c : x,y ∈ X, c ∈ C, ◁ ∈ {<,≤} }`,
`|Φ_C| = O(|X|² · |C|)`. Each `G(q)` only grows, inside `Φ_C`; hence ≤ `|Q|·|Φ_C|` increments.

**Complexity.** With `k = |X|`, `m = |T|`, `n = |Q|`, `Γ = max_q |G(q)| ≤ |Φ_C| = O(k²|C|)`:
each pop of a state `q'` costs `Σ_{t ∈ pred(q')} O(|G(q')|)`, and a state is re-pushed at most
`Γ` times, giving **`O(m · Γ²)` = `O(m · k⁴|C|²)`** worst case, and in practice
`O(m·Γ)` with a "changed-set" worklist (propagate only the newly added constraints, not all of
`G(q')`). Memory `O(n·Γ)`.
For **diagonal-free** automata `pre` never creates a diagonal, so `Γ = O(k)` and the whole
analysis is `O(m·k)` — i.e. exactly the cost of the classical local-LU static guard analysis.

---

## 2. THE SIMULATION TEST

### 2.1 `⊑_G` on configurations, and its lifting to zones

[THESIS] **Definition 3.1** (= [SURVEY] Definition 3, = [CAV19] Definition 5), verbatim:

> "Given two valuations `v, v′` and a (finite or infinite) set of atomic constraints `G`,
> `v ⊑_G v′` if `∀φ ∈ G` and `∀δ ∈ ℝ≥0`:  if `v + δ ⊨ φ` then `v′ + δ ⊨ φ`."

Lifted, verbatim:

* [THESIS] **Definition 3.15** (configurations): `(q,v) ⊑_G (q′,v′)` iff `q = q′` and `v ⊑_{G(q)} v′`.
* [THESIS] **Definition 3.4** (zones): `Z ⊑_G Z′` iff `∀v ∈ Z ∃v′ ∈ Z′ . v ⊑_G v′`.
* [THESIS] **Definition 3.16** (nodes): `(q,Z) ⊑_G (q,Z′)` iff `Z ⊑_{G(q)} Z′`.
  Trivially true when `Z = ∅`; false when `Z ≠ ∅ = Z′`.

Two structural facts used constantly:

* [THESIS] **Proposition 3.2** (antitone in `G`): `G ⊇ G′` and `v ⊑_G v′` ⟹ `v ⊑_{G′} v′`.
  **Bigger constraint set ⇒ finer (smaller) relation.** Lifts to zones directly.
* [THESIS] **Proposition 3.3** (per-valuation decomposition): `v ⊑_G v′` iff `v ⊑_{φ} v′` for
  every `φ ∈ G`. **This does NOT lift to zones** — that is exactly why the zone test needs
  pairs of constraints (Theorem 4.8 below).

**Single-constraint characterisations** (these are the "`a_LU`-style" atoms):

* [THESIS] **Theorem 3.5**: `v ⊑_{x ◁ c} v′` iff `v ⊭ x ◁ c` **or** `v′(x) ≤ v(x)`.
* [THESIS] **Theorem 3.9**: `v ⊑_{d ◁ y} v′` iff `v′ ⊨ d ◁ y` **or** `v(y) ≤ v′(y)`.
* [CAV19] **Proposition 11**: `v ⊑_G v′` iff `v ⊑_{G⁻} v′` and for every *diagonal* `φ ∈ G`,
  `v ⊨ φ ⇒ v′ ⊨ φ` (the `∀δ` is irrelevant for diagonals: `v(x)−v(y)` is time-invariant).
  `G⁻` = non-diagonal part of `G`.

### 2.2 The decomposition theorem (why pairs suffice)

[THESIS] **Theorem 4.8**, verbatim: *"Let `Z, Z′` be two non-empty canonical zones and
`G_nd = G_u ∪ G_ℓ` be a finite set of non-diagonal constraints with `G_u` containing only upper
bound constraints and `G_ℓ` containing only lower bound constraints. Then, `Z ⊑_{G_nd} Z′` iff
the following two conditions hold: 1. `Z ⊑_{φ} Z′` for every `φ ∈ G_nd`, [and] 2.
`Z ⊑_{φ_u, φ_ℓ} Z′` for every `φ_u ∈ G_u` and `φ_ℓ ∈ G_ℓ`."*

(The "or" printed in the PDF between 1. and 2. is a typo — the proof, the `(⇒)` direction and
Algorithm 5 all treat it as a conjunction. Verified against Algorithm 5, [THESIS] p. 80.)

Proof idea: the distance graph `G↑v` of `↑v = {v′ | v ⊑_{G_nd} v′}` only has edges incident to
`0`, so any simple cycle in `min(G↑v, Z′)` uses at most two of them; canonicity of `Z′`
collapses the rest to a single `Z′` edge. Hence a negative cycle involves at most one upper and
one lower constraint.

### 2.3 The exact per-constraint / per-pair test on DBM entries

Fix non-empty canonical `Z, Z′` and let `G_nd = G_u ⊎ G_ℓ` (upper `x ◁₁ c`, lower `d ◁₂ y`).
[THESIS] **Theorem 4.26**, verbatim, in the [THESIS] index convention (`Z_xy` bounds `y − x`):

> `Z ⋢_{G_nd} Z′` iff at least one of the following three conditions hold:
> 1. `(◁′_{x0}, z′_{x0}) < (◁_{x0}, z_{x0})`  and  `(◁_{x0}, z_{x0}) + (◁₁, c) ≥ (≤,0)`
>    for some `x ◁₁ c ∈ G_u`,
> 2. `(◁′_{0y}, z′_{0y}) < (◁_{0y}, z_{0y})`  and  `(◁′_{0y}, z′_{0y}) + (◁₂, −d) < (≤,0)`
>    for some `d ◁₂ y ∈ G_ℓ`,
> 3. `(◁′_{xy}, z′_{xy}) < (◁_{xy}, z_{xy})`  and  `(◁_{x0}, z_{x0}) + (◁₁, c) ≥ (≤,0)`  and
>    `(◁′_{xy}, z′_{xy}) + (◁₂, −d) < (◁_{x0}, z_{x0})`
>    for some `x ◁₁ c ∈ G_u` and `d ◁₂ y ∈ G_ℓ`.

**The same thing in standard DBM indexing** (`D[i][j]` = bound on `x_i − x_j`, as in
`HDBM/dbm_ops.hpp`; `Z_uv = D[v][u]`), which is what you implement:

```
Z ⋢_G Z'   iff   ∃ (x ◁₁ c) ∈ G_u :        D[0][x] + (◁₁,c) ≥ (≤,0)        # (U-sat)
                                     and   D'[0][x] < D[0][x]              # (U-cut)
           or    ∃ (d ◁₂ y) ∈ G_ℓ :        D'[y][0] + (◁₂,−d) < (≤,0)      # (L-unsat')
                                     and   D'[y][0] < D[y][0]              # (L-cut)
           or    ∃ (x ◁₁ c) ∈ G_u, (d ◁₂ y) ∈ G_ℓ, x ≠ y :
                                           D[0][x] + (◁₁,c) ≥ (≤,0)        # (U-sat)
                                     and   D'[y][x] < D[y][x]              # (diag-cut)
                                     and   D'[y][x] + (◁₂,−d) < D[0][x]    # (pair)
```

Meaning of each conjunct (from the lemmas that produce them — this is what makes it
implementable and debuggable):

| conjunct | source | meaning |
|---|---|---|
| `D[0][x] + (◁₁,c) ≥ (≤,0)` | [THESIS] Lemma 4.10 | `Z ∩ ⟦x ◁₁ c⟧ ≠ ∅` — some `v ∈ Z` still satisfies the upper guard |
| `D'[0][x] < D[0][x]` | [THESIS] Lemma 4.11 | `∃v∈Z ∀v′∈Z′ : v(x) < v′(x)` — `Z` reaches lower `x` than `Z′` can |
| `D'[y][0] + (◁₂,−d) < (≤,0)` | [THESIS] Lemma 4.14 | `Z′ ∩ ⟦d ◁₂ y⟧ = ∅` — no `v′ ∈ Z′` satisfies the lower guard |
| `D'[y][0] < D[y][0]` | [THESIS] Lemma 4.15 | `∃v∈Z ∀v′∈Z′ : v′(y) < v(y)` — `Z` reaches higher `y` than `Z′` |
| `D'[y][x] < D[y][x]` | [THESIS] Lemma 4.24 | `Z` reaches a larger `y − x` than `Z′` allows |
| `D'[y][x] + (◁₂,−d) < D[0][x]` | [THESIS] Lemma 4.23 | the two-edge negative cycle through the `Z′` diagonal edge |

**Algorithm** ([THESIS] Algorithm 5, p. 80, transcribed to standard DBM indexing):

```
bool Z ⊑_{G_nd} Z' :                          # pre: Z, Z' non-empty and canonical
    for (x ◁₁ c) in G_u:
        if  D[0][x] + (◁₁,c) >= (<=,0)  and  D'[0][x] < D[0][x]:      return false
    for (d ◁₂ y) in G_ℓ:
        if  D'[y][0] + (◁₂,-d) < (<=,0) and  D'[y][0] < D[y][0]:      return false
    for (x ◁₁ c) in G_u, (d ◁₂ y) in G_ℓ with x ≠ y:
        if  D[0][x] + (◁₁,c) >= (<=,0)
        and D'[y][x] + (◁₂,-d) < D[0][x]
        and D'[y][x] < D[y][x]:                                       return false
    return true
```

**Correctness statement.** [THESIS] Theorem 4.26 + Algorithm 5: for non-empty canonical
`Z, Z′` and a finite **diagonal-free** `G_nd`, the procedure returns `true` **iff**
`∀v ∈ Z ∃v′ ∈ Z′ . ∀φ ∈ G_nd ∀δ ≥ 0 . (v+δ ⊨ φ ⇒ v′+δ ⊨ φ)`. It is exact (not an
over- or under-approximation) — necessary *and* sufficient.

**Complexity.** [THESIS] p. 80: by Corollaries 3.6/3.10 `G_nd` can be reduced to at most one
upper and one lower bound per clock, so `|G_u|, |G_ℓ| ≤ |X|` and the algorithm is
**`O(|X|²)`**, matching the `⪯_LU` test of Herbreteau–Srivathsan–Walukiewicz. *"if `G_nd`
contains much lesser number of constraints than `2|X|`, then the time taken … can also
potentially be much lesser"* — with only lower bounds it is linear. This is the
"quadratic in the number of clocks" claim of [SURVEY] p. 14.

Preconditions to enforce in code: `Z ≠ ∅`, `Z′ ≠ ∅`, both canonical (run `dbm_close` first);
handle `Z = ∅ ⇒ true` and `Z ≠ ∅ = Z′ ⇒ false` before entering ([THESIS] p. 32 remark,
[CAV19] Algorithm 1 lines 2–5).

### 2.4 With diagonals in `G` (for completeness)

[CAV19] **Lemma 17**: for a diagonal `φ = x − y ◁ c ∈ G` and `G′ = G \ {φ}`,
`Z ⊑_G Z′` iff `Z ∩ φ ⊑_{G′} Z′ ∩ φ` **and** `Z ∩ ¬φ ⊑_{G′} Z′`. Recursing gives [CAV19]
Algorithms 1/2 with `O(2^d · |X|²)` for `d` diagonals in `G` ([CAV19] Theorem 18). The test is
**NP-complete** ([SURVEY] p. 14, [THESIS] §4.4, [CAV19] Theorem 22). If your `G(q)` is
diagonal-free (guaranteed when the automaton is diagonal-free, since `pre` of a non-diagonal is
non-diagonal), you never enter this branch.

---

## 3. RELATION TO `a_LU` — and an important negative result

### 3.1 The claim, and the exact reason it holds

[THESIS] **Definition 3.12**: from a diagonal-free `G_nd`,
`L_{G_nd}(y) = max{d | d ◁ y ∈ G_nd}`, `U_{G_nd}(x) = max{c | x ◁ c ∈ G_nd}` (`max ∅ = −∞`).

[THESIS] **Theorem 3.13**: `v ⪯_{LU(G_nd)} v′` **implies** `v ⊑_{G_nd} v′`.
[THESIS] **Corollary 3.14**: same for zones. So `⪯_LU ⊆ ⊑_G`: **`⊑_G` is coarser**.

[THESIS] **Example 3.2.1** (verbatim data): `G_nd = {x < 1}`, `v(x) = 1`, `v′(x) = 1.1`. Then
`L(x) = −∞`, `U(x) = 1`; `v′(x) > v(x)` but `v(x) = U(x)` so `v ⋠_{LU} v′`; yet `v ⊭ x < 1`, so
by Theorem 3.5 `v ⊑_G v′`. [THESIS] p. 36: *"This improvement is due to the fact that the
`L_{G_nd}`, `U_{G_nd}` bounds fail to distinguish between the inequalities `<` and `≤`."*

**A concrete zone-level example, verified by hand against both tests [DERIVED]:**
`G(q) = {x < 1}` (i.e. the only guard out of `q` is `x < 1`), `Z = {x = 1}`, `Z′ = {x = 2}`.
DBMs: `D[0][x] = (≤,−1), D[x][0] = (≤,1)`; `D′[0][x] = (≤,−2), D′[x][0] = (≤,2)`.
* `a_LU` (`U(x)=1, L(x)=−∞`, code path `dbm_is_alu_le`): `D[0][x] = (≤,−1) ≥ (≤,−U(x)) = (≤,−1)` ✓;
  take `y = 0` (`L(0)=0`): `D′[0][x] = (≤,−2) < (≤,−1) = D[0][x]` ✓ and
  `(≤,−2) + (<,0) = (<,−2) < (≤,−1)` ✓ → **returns false, `Z` is NOT subsumed.**
* `G`-test, condition (1): `D′[0][x] < D[0][x]` ✓ but
  `D[0][x] + (<,1) = (≤,−1)+(<,1) = (<,0) ≥ (≤,0)` ✗ (since `(<,0) < (≤,0)`) → fails.
  No lower bounds ⇒ conditions (2),(3) vacuous → **`Z ⊑_G Z′` holds, `Z` is pruned.**

So `G` subsumes a zone `a_LU` does not, on a two-line automaton. Note the mechanism: `Z` sits
*exactly on* the boundary `x = 1` of a strict guard `x < 1`, so `Z` can never satisfy that guard
again; `a_LU`, which rounds `x < 1` to `U(x) = 1`, cannot see this.

### 3.2 The negative result you need to know before building anything

**On diagonal-free `G`, a set of constraints is provably NO stronger than one `(◁,c)` pair per
clock per direction.** [THESIS] **Corollary 3.6**: for `G_nd = {x ◁₁ c₁, …, x ◁_k c_k}` all
upper bounds on the *same* clock, `v ⊑_{G_nd} v′` iff `v ⊑_{x ◁ c} v′` where
`(◁,c) = max{(◁_i,c_i)}`. **Corollary 3.10**: dually for lower bounds. Combined with
Proposition 3.3 (per-valuation decomposition) this collapses any diagonal-free `G_nd` to
**at most one upper and one lower bound pair per clock**, *exactly*, with no loss — and
[THESIS] itself relies on this (p. 67: *"due to Corollary 3.6 and Corollary 3.10, it will be
assumed that `G_nd` contains at most one lower and one upper bound constraint per clock"*).

Therefore, for **diagonal-free automata**:

```
⊑_G   ≡   "a_LU with strictness bits":  per state q, per clock x,
          keep (◁ᵁ_q(x), U_q(x)) and (◁ᴸ_q(y), L_q(y)) instead of the integers U_q(x), L_q(y).
```

and the backward propagation of §1.4, restricted to non-diagonals, **is** the classical
per-location LU static guard analysis (Behrmann et al., "static guard analysis"; [SURVEY] §4.2
"state-dependent constants", p. 10 — the *"limited propagation idea"* the survey says will be
"presented in the context of `G`-simulation in Section 5"): `pre(x ◁ c, R) = ∅` when `x ∈ R` is
literally the rule *"if `x` is compared to constant `c` and then reset before reaching state
`q`, then constant `c` is not taken into account"* ([SURVEY] p. 10).

**So: what does `G` buy over per-location `L/U`?**
1. **Strictness** (`<` vs `≤`). Real but small; one epsilon of coarseness per bound. This is the
   *only* gain in the diagonal-free case — [THESIS] p. 36 says so, and adds *"this improvement
   of `⊑_{G_nd}` over `⊑_LU` is not drastic and therefore it may not show large improvements in
   practice."* Quote that sentence to anyone who expects a big win here.
2. **Diagonals.** This is the actual payload. When `G(q)` contains diagonal constraints there is
   *no* per-clock bound representation at all: `a_LU` is outright unsound on automata with
   diagonal guards, and the previous state of the art had to *eliminate* diagonals by an
   exponential automaton/zone blow-up ([SURVEY] p. 12: *"Previous approaches for dealing with
   diagonals had to eliminate the diagonal constraints explicitly and therefore resulted in a
   blow-up in space. Here the difficulty gets absorbed into the time to do the simulation
   test."*). A *set of constraints* beats *two integers per clock* precisely and only when the
   set is not clock-decomposable.
3. **Updates.** `⊑_G` extends to updatable TA; `⪯_LU` does not ([SURVEY] p. 14).
4. **Fewer constraints ⇒ faster test.** `G(q)` is often much smaller than `2|X|`; the test loops
   over `G_u × G_ℓ`, not over all clock pairs ([THESIS] p. 80). For a network where most clocks
   are irrelevant at a given location vector this is a real constant-factor win over an
   `O(|X|²)` `a_LU` sweep.

**Implementation consequence.** `dbm_is_alu_le` in
`HDBM/include/ptctlmc/dbm_ops.hpp:365` is *already* Theorem 4.26 in disguise. Line-by-line:
`U[x] == −∞ → continue` ≙ no `φ_u` for `x`; `Z[0][x] < (≤,−U[x]) → continue` ≙ negation of
(U-sat) with `(◁₁,c) = (≤,U(x))`; `Z'[y][x] < Z[y][x]` ≙ (diag-cut) (and, at `y=0`, (U-cut));
`Z'[y][x] + (<,−L[y]) < Z[0][x]` ≙ (pair) with `(◁₂,d) = (<,L(y))` (and, at `x=0`, (L-unsat′)).
**The `G`-test is the same loop with `(◁,c)` pairs drawn from `G_u × G_ℓ` instead of integers
drawn from `U × L`.** Migrating costs ~20 lines, not a rewrite.

*(Detail: [THESIS] Algorithm 5 does not exclude `x = y` in the third loop. With canonical DBMs
`D[x][x] = D′[x][x] = (≤,0)`, so `D′[y][x] < D[y][x]` is false when `x = y` and the case can
never fire. Including or excluding `x = y` is equivalent; the underlying proof
([THESIS] Thm. 4.8, cycles of Fig. 4.7) requires `x ≠ y` anyway.)*

---

## 4. NETWORKS — what is actually done

**Answer: nobody computes `G` over the product. Both the LU machinery in upstream TChecker and
the only existing `G`-simulation implementation compute the map PER PROCESS, indexed by a single
process's location id, and combine the per-process entries at the location vector on the fly —
`max` for LU bounds, plain UNION for `G`. The papers do not discuss networks at all.**

I verified every code claim below by reading the cloned sources directly.

### 4.1 The papers: silent

[SURVEY], [CAV19], [CONCUR18], [THESIS] and [FSTTCS20] all define `A` as a *single* timed
automaton `(Q, X, q₀, T, F)` and `G(q)` per state `q ∈ Q`. Networks/products/synchronisation
appear in [SURVEY] only in §7 (partial-order methods, local-time semantics — a different topic)
and in the tool description on p. 20 (*"TChecker models consist in networks of timed automata
that communicate through handshaking"*), never in connection with `G`. **The question of how to
obtain `G` for a location vector without building the product is NOT ESTABLISHED IN SOURCES.**

### 4.2 Upstream TChecker: no `G` at all; LU is per-process, combined by `max`

Upstream `ticktac-project/tchecker` (HEAD `d711ace`, 2025-12-09) **does not implement the
`G`-simulation**: no constraint map, no `wp`, no occurrence of "Gastin" in the tree. Its
simulations are `a_LU` / `a_M` (`tchecker::dbm::is_alu_le`, `src/dbm/dbm.cc:807`) plus a
sync-`a_LU` variant for local-time semantics. `tchecker::ta::has_diagonal_constraint` exists
(`src/ta/static_analysis.cc:37`) but is **never called**; the bound solver is explicitly named
`df_solver_t`, "*Clock bounds solver for diagonal free timed automata*".

The per-vloc combination, `src/clockbounds/clockbounds.cc:219-231` (verified verbatim):

```cpp
void local_lu_map_t::bounds(tchecker::vloc_t const & vloc, map_t & L, map_t & U) const {
  tchecker::clockbounds::clear(L);  tchecker::clockbounds::clear(U);
  for (tchecker::clock_id_t id : vloc) {        // id = a location of ONE process
    assert(id < _loc_nb);
    tchecker::clockbounds::update(L, *_L[id]);  // update() is a POINTWISE MAX
    tchecker::clockbounds::update(U, *_U[id]);
  }
}
```

`_L`/`_U` are indexed by *system location id* and sized `system.locations_count()` — the **sum**
over processes, not the product. The result is memoised per reachable vloc
(`cache_local_lu_map_t`, `include/tchecker/clockbounds/cache.hh:33`) and consumed by the
subsumption predicate in `src/tck-reach/zg-aLU-covreach.cc:47-57`.

**Crucially, upstream is aware of the shared-clock hazard and patches it explicitly** — in
`df_solver_t::add_assignment`, `src/clockbounds/solver.cc:165-177` (verified verbatim):

```cpp
  // We need to propagate shared variables only when x != y
  if (x == y) return;
  // Propagation across processes: L_{m,x} - L_{l1,y} <= c / U_{m,x} - U_{l1,y} <= c
  // for every location m in another process
  for (tchecker::loc_id_t m = 0; m < _loc_number; ++m)
    if (_loc_pid[m] != _loc_pid[l1]) { updateL(index(m,x), index(l1,y), LE, c);
                                       updateU(index(m,x), index(l1,y), LE, c); }
```

This is the LU analogue of exactly the obligation my §5.2/§5.3 identifies: a clock written in
one process but constrained in another must have its bound propagated across processes. Upstream
does it eagerly, statically, over all pairs of locations in different processes.

### 4.3 The one `G`-simulation implementation: `mukherjee-sayan/tchecker` fork

Branch `master`, HEAD `07a66d8` (2023-03-07), commits `5ad977b` "add gsim implementation" and
`c15d0bb` "BUG FIX: amap computation"; based on a ~2021 TChecker snapshot, **never merged
upstream**. README cites [CAV19] and [FSTTCS20]; adds the algorithm option `-a gsim`.

* Data: `namespace tchecker::amap`, `class a_map_t`, with `std::vector<...> _G` (diagonals) and
  `_Gdf` (non-diagonals), both **indexed by system location id** — i.e. per process location.
* **Combination at a location vector = plain union**, `src/clockbounds/clockbounds.cc:683-697`
  (verified verbatim):

```cpp
void a_map_t::bounds(vloc_t const & vloc, std::vector<...diagonal...> & G,
                                          std::vector<...simple...> & Gdf) const {
  G.clear(); Gdf.clear();
  for (tchecker::clock_id_t id : vloc) {
    for (const auto & diag    : _G[id])   G.push_back(diag);
    for (const auto & nondiag : _Gdf[id]) Gdf.push_back(nondiag);
  }
}
```

  No dedup across processes, no `max` reduction, no re-saturation. Not memoised either — it is
  rebuilt on **every** subsumption check (`src/tck-reach/zg-gsim.cc:62-67`).
* **Fixpoint scope = per-process edges**, `src/clockbounds/solver.cc:1629-1711` (verified):
  `compute_amap` seeds `G[loc]` from each location's **invariant** and `G[src]` from each edge's
  guard, then `find_fixpoint` loops `for (edge : system.edges())` — the individual process
  automata's edges — propagating every constraint of `G[edge->tgt()]` back to `G[edge->src()]`
  via `propagate_constraint` (which is `wp`/`pre`). `system.edges()` enumerates per-process
  edges; **the product is never built**, and there is no synchronisation awareness: `wp` is taken
  w.r.t. one edge's guard/statement, never a synchronised vedge's conjoined guard and
  simultaneous updates.
* **No cross-process propagation for shared clocks.** The analogue of upstream's
  `add_assignment` loop over "*every location m in another process*" has **no counterpart** in
  `namespace amap`. So the fork does for `G` exactly the plain union that upstream deliberately
  does **not** do for LU.
* Non-termination under updates is detected by a `find_cutoff_bound` heuristic
  (`solver.cc:1599-1627`) which throws
  `"this algorithm cannot check reachability in this input automaton: non-terminating fixpoint
  computation"`.
* Test coverage: `test/unit-tests/test-amap.hh` (879 lines) exercises only expression-level `wp`
  / `compute_pre`; **no multi-process system is tested**. The benchmarks shipped in the same
  commit (`examples/thesis/fischer/*`, `job-shop/*`) *are* networks, so the union is used on
  networks in practice — but its soundness for networks is not argued anywhere in the fork's
  source, comments, README or tests.
* The test itself: `src/dbm/dbm.cc:795` `is_g_le_nd` (the 3-condition non-diagonal test of §2.3),
  `is_g_le_star` (recursive diagonal split, [CAV19] Alg. 2), `is_g_le`. The search runs with
  `ELAPSED_SEMANTICS, NO_EXTRAPOLATION` — `G`-simulation **fully replaces** extrapolation,
  zones are stored exactly (`src/tck-reach/zg-gsim.cc:150-171`). This directly corroborates §B.2
  point 1.

### 4.4 Verdict for item 4

| | upstream TChecker | `mukherjee-sayan` fork |
|---|---|---|
| implements `⊑_G`? | **no** — `a_LU`/`a_M` only | **yes** (`-a gsim`) |
| per-state info indexed by | one process's location id | one process's location id |
| combination at a vloc | **pointwise max** of `L`,`U` | **plain union** of constraint lists |
| product built for the analysis? | **no** | **no** |
| lazy/during search? | no — static, memoised per vloc | no — static, recomputed per compare |
| cross-process fix for shared clocks | **yes**, explicit | **none found** |

So: **compositionally, per process, statically, combined at the location vector.** Not over the
product, not lazily. Which makes §5 the load-bearing question, and makes the missing
cross-process propagation in the fork a real (unaddressed) soundness question — see §5.2.

---

## 5. SOUNDNESS OF A COMPOSITIONAL OVER-APPROXIMATION

**Short answer: the union over processes is sound *if and only if* it is still closed under
`pre` for the *product's* resets. Under "no shared clocks" it is, verbatim. With shared clocks
or cross-process diagonals it is NOT, and I give a counterexample. A cheap unconditional fix
(`pre`-saturating each per-process map, ≤3× blow-up) makes it sound with no hypotheses.**

Everything in this section is **[DERIVED]** from [THESIS] Definition 3.22 + Theorem 3.23. The
sources do not discuss networks (see §4).

### 5.0 Setup

Processes `P_1 … P_n`, `P_p` with locations `Loc_p`, local transitions
`(ℓ_p, g_p, R_p, ℓ′_p)`. The product/network `N` has states `q̄ = (q_1,…,q_n)` and, for each
synchronisation over a participating set `S ⊆ {1..n}` with local transitions
`(q_p, g_p, R_p, q′_p)` for `p ∈ S`, a transition

```
(q̄, g, R, q̄′)   with   g = ⋀_{p∈S} g_p,   R = ⋃_{p∈S} R_p,
                        q̄′_p = q′_p (p ∈ S),   q̄′_p = q̄_p (p ∉ S).
```

`G_p` = an A-map of `P_p` *viewed as a standalone TA* (computed by §1.4 on `P_p`'s own
transitions, guards and resets — no product anywhere). Define

```
G_∪(q̄)  =  ⋃_p G_p(q̄_p).
```

### 5.1 The two obligations, checked one by one

`G_∪` is sound iff it is an A-map of `N` ([THESIS] Def. 3.22), i.e. for every product transition
`(q̄, g, R, q̄′)`:

**(i) `atoms_of(g) ⊆ G_∪(q̄)`. — ALWAYS HOLDS.**
`atoms_of(g) = ⋃_{p∈S} atoms_of(g_p)` and `atoms_of(g_p) ⊆ G_p(q̄_p)` because `G_p` is an A-map
of `P_p` and `(q̄_p, g_p, R_p, q′_p)` is a transition of `P_p`. ∎

**(ii) `pre(G_∪(q̄′), R) ⊆ G_∪(q̄)`. — CONDITIONAL.**
Take `φ ∈ G_p(q̄′_p)`. What `G_p` guarantees is `pre(φ, R_p) ∈ G_p(q̄_p)` (or, for `p ∉ S`,
`φ ∈ G_p(q̄_p)` since `q̄′_p = q̄_p`). What the product *demands* is `pre(φ, R) ∈ G_∪(q̄)` with
`R = ⋃_{p∈S} R_p ⊇ R_p`. **These coincide exactly when `R` and `R_p` reset the same clocks among
those occurring in `φ`.**

> **Proposition 5.1 [DERIVED] — the diagonal-free case needs NO hypothesis.** If every
> constraint in every `G_p` is **non-diagonal** — which is automatic when the network is
> diagonal-free, since `pre` of a non-diagonal is a non-diagonal or nothing (§1.1) — then `G_∪`
> is an A-map of `N`, **even with clocks shared, read and reset across processes**. Hence
> `⊑_{G_∪}` is a simulation and the forward analysis is sound, complete and terminating.
>
> *Proof.* Obligation (i) is §5.1(i). For (ii), let `φ ∈ G_p(q̄′_p)` be non-diagonal on clock `x`.
> *If `x ∈ R`*: `pre(φ, R) = ∅` — nothing to prove. *If `x ∉ R`*: then `pre(φ,R) = φ`, and also
> `x ∉ R_p ⊆ R`, so `pre(φ, R_p) = φ`. If `p ∈ S`, `G_p` being an A-map of `P_p` gives
> `φ = pre(φ,R_p) ∈ G_p(q̄_p)`; if `p ∉ S`, `q̄′_p = q̄_p` gives `φ ∈ G_p(q̄_p)` directly. Either
> way `φ ∈ G_∪(q̄)`. ∎
>
> The reason this works: for a non-diagonal, `pre` can only **delete**, and a *larger* reset set
> deletes at least as much. The product's reset set is larger than any process's, so it never
> demands a constraint the process did not already own.

> **Proposition 5.1b [DERIVED] — with diagonals, ownership suffices.** If (H1) the clocks
> partition, `X = ⊎_p X_p`; (H2) every guard of `P_p` mentions only clocks of `X_p`; (H3)
> `R_p ⊆ X_p` — then `G_∪` is an A-map of `N`.
>
> *Proof.* (H2)+(H3) and the fact that `pre` never introduces a clock (it only substitutes `0`)
> give: every `φ ∈ G_p(ℓ)` mentions only clocks of `X_p`. Hence `R ∩ clocks(φ) = R_p ∩
> clocks(φ)` and `pre(φ, R) = pre(φ, R_p) ∈ G_p(q̄_p) ⊆ G_∪(q̄)`. For `p ∉ S`, `R ∩ X_p = ∅`
> so `pre(φ,R) = φ ∈ G_p(q̄_p)`. Obligation (i) is §5.1(i). ∎

> **Proposition 5.2 [DERIVED] — without the hypotheses `G_∪` is UNSOUND.** Counterexample:
> `P_1` owns `x`, `P_2` owns `y`. `P_1` has at `ℓ_1` a self-loop guarded `x − y ≤ 5`
> (a cross-process diagonal, so (H2) fails). `P_2` has at `ℓ_2` an unguarded transition
> `ℓ_2 → ℓ′_2` resetting `y`, taken alone (`S = {2}`). Then
> `G_1(ℓ_1) = {x − y ≤ 5}` (its `pre` under `R_1 = ∅` is itself), `G_2 = ∅` everywhere, and
> `G_∪((ℓ_1,ℓ_2)) = G_∪((ℓ_1,ℓ′_2)) = {x − y ≤ 5}`.
> Obligation (ii) demands `pre(x−y ≤ 5, {y}) = x ≤ 5 ∈ G_∪((ℓ_1,ℓ_2))` — it is not.
> And the simulation genuinely breaks: take `v = (x{=}5, y{=}5)`, `v′ = (x{=}6, y{=}6)`.
> `v ⊑_{x−y≤5} v′` (both have `x−y = 0`, invariant under delay). After the transition,
> `[{y}]v` has `x−y = 5 ⊨ φ` while `[{y}]v′` has `x−y = 6 ⊭ φ`. So condition 2b of a simulation
> ([THESIS] p. 38) fails. Note the missing constraint `x ≤ 5` is exactly what would have
> separated `v` from `v′` (`v(x)=5 ⊨`, `v′(x)=6 ⊭`). ∎

> **Proposition 5.3 [DERIVED] — unconditional fix by `pre`-saturation.** Define
> ```
> G_p^sat(ℓ) = { pre(φ, R) : φ ∈ G_p(ℓ), R ⊆ X }        # = the reset-collapse closure
> G_⊎(q̄)    = ⋃_p G_p^sat(q̄_p)
> ```
> Then `G_⊎` is an A-map of `N` **with no hypotheses on clock ownership**.
>
> The closure is tiny: `pre` of a non-diagonal is itself or nothing; `pre` of `x − y ◁ c` is one
> of `{x−y◁c}`, `{x◁c}` (if `c ≥ 0`), `{−c◁y}` (if `−c ≥ 0`), `∅`. So
> `|G_p^sat(ℓ)| ≤ 3·|G_p(ℓ)|`, and `G_p^sat = G_p` whenever `G_p(ℓ)` is diagonal-free.
> **For a diagonal-free network, saturation is a no-op and the plain union is already sound.**
>
> *Proof.* `pre` is substitution of `0`, so substitutions compose:
> `pre(pre(ψ, R′), R) = pre(ψ, R ∪ R′)` (vacuous results are dropped on both sides, which is
> sound by the `x∈R` branches of [THESIS] Lemma 3.20). Let `φ ∈ G_⊎(q̄′)`, say
> `φ = pre(ψ, R′)` with `ψ ∈ G_p(q̄′_p)`, and let `(q̄,g,R,q̄′)` be a product transition.
> *Case `p ∈ S`*: `χ := pre(ψ, R_p) ∈ G_p(q̄_p)` since `G_p` is an A-map. As `R_p ⊆ R ⊆ R∪R′`,
> `pre(φ, R) = pre(ψ, R∪R′) = pre(χ, R∪R′) ∈ G_p^sat(q̄_p) ⊆ G_⊎(q̄)`.
> *Case `p ∉ S`*: `q̄′_p = q̄_p`, so `ψ ∈ G_p(q̄_p)` and `pre(φ,R) = pre(ψ, R∪R′) ∈ G_p^sat(q̄_p)`.
> Obligation (i) holds because `G_p ⊆ G_p^sat` (take `R = ∅`). ∎

### 5.2 Answer to the question as posed

* **Is the compositional union sound?**
  * **Diagonal-free network → YES, unconditionally, plain union.** (Prop. 5.1.) Shared clocks,
    cross-process resets, arbitrary synchronisation: all fine. This is the case PTCTLMC is in if
    it has no diagonal guards, and it is the answer to the question as posed.
  * **With diagonals → yes for the plain union if no clock is reset by a process other than the
    one whose constraints mention it** (Prop. 5.1b); **no in general** (Prop. 5.2);
    **yes unconditionally for the `pre`-saturated union** (Prop. 5.3, ≤3× blow-up on diagonals
    only).
  * **This is a real gap in the only existing implementation.** The `mukherjee-sayan/tchecker`
    fork does exactly the plain union (§4.3, `a_map_t::bounds`) with **no** cross-process
    propagation, while upstream TChecker's LU solver *does* propagate across processes for
    shared clocks (§4.2, `df_solver_t::add_assignment`). For diagonal-free models the fork is
    covered by Prop. 5.1 — but `gsim` exists precisely for models *with* diagonals, where
    Prop. 5.2 applies. I could not find this discussed anywhere in the fork or the papers:
    **NOT ESTABLISHED IN SOURCES.** If you implement `G` for a network with diagonal guards,
    apply Prop. 5.3.
* **"An over-approximation of `G` makes the simulation FINER but should remain sound — is that
  direction right?"** The *precision* half is right and is a theorem ([THESIS] Prop. 3.2). The
  *soundness* half is **WRONG as stated** — see §A. Over-approximating is safe only when the
  over-approximation is still `pre`-closed. The compositional union happens to be `pre`-closed
  (under 5.1/5.3), which is why the shortcut works; it is not a consequence of it being larger.
* **Cost.** `Σ_p |Loc_p|` fixpoint states instead of `Π_p |Loc_p|`, computed once, statically,
  before the search. `G_∪(q̄)` never needs to be materialised: keep the `n` per-process sets and
  iterate the test's `G_u × G_ℓ` loops over the concatenation. Also **cache the reduction**:
  by [THESIS] Cor. 3.6/3.10 the non-diagonal part collapses to one `(◁,c)` upper and one lower
  bound per clock, so `G_∪(q̄)`'s non-diagonal part is computed by a per-clock `max` over
  processes — the *identical* operation your `bounds(s.locs.data(), Ls, Us)` already performs
  for `a_LU`, just with strictness bits attached.
* **Invariants / committed locations:** fold per-process invariants into per-process guards
  before running §1.4 on each process (see §1.4 note). Compositionality is unaffected.

---

## A. MONOTONICITY OF SOUNDNESS IN `G`

> **Verdict: precision is monotone; soundness is NOT. `G*` is the coarsest sound map, but it is
> false that every superset of `G*` is sound. The correct invariant is "is an A-map"
> (= `pre`-closed and guard-containing), which is a *closure* property, not a size property.**

### A.1 Precision (the easy half) — CONFIRMED

[THESIS] **Proposition 3.2**, verbatim: *"Let `G` be a set of atomic constraints and `v, v′` be
two valuations. Then, `v ⊑_G v′` implies `v ⊑_{G′} v′`, for every `G′` satisfying `G ⊇ G′`."*
Lifting to zones is immediate from Definition 3.4. So

```
G ⊇ H  (pointwise)   ⟹   ⊑_G ⊆ ⊑_H .
```

**Larger constraint set ⇒ finer relation ⇒ strictly less pruning ⇒ smaller covered region.**
Strictness is genuine: [THESIS] Example 3.2.1 and §3.1 above exhibit `(v,v′)` related by the
smaller set and not by the larger one. Unambiguous, cited, no caveats.

### A.2 Soundness (the half that matters) — REFUTED as stated

**Claim under test:** "`G ⊇ G*` pointwise ⟹ `⊑_G` is still a simulation."
**This is false.** [DERIVED] counterexample:

```
A:  q0 --( g = true, R = ∅ )--> q1      (no other transitions)
G*(q0) = G*(q1) = ∅.
Take  G(q0) = ∅,  G(q1) = {x ≤ 5}.      Pointwise G ⊇ G*.  ✓
```

`⊑_{G(q0)} = ⊑_∅` relates *every* pair, so `(q0, v) ⊑_G (q0, v′)` for `v(x)=0, v′(x)=6`.
The transition is enabled from both, and afterwards we need `(q1,v) ⊑_G (q1,v′)`, i.e.
`v ⊑_{x≤5} v′` — but `v ⊨ x ≤ 5` and `v′ ⊭ x ≤ 5`. Condition 2b of a simulation ([THESIS]
p. 38 / [SURVEY] Definition 1(2)) fails. `⊑_G` is **not** a simulation, and forward analysis
using it can prune a node whose successors are not covered → **missed reachable states**
(incomplete search, i.e. a false "unreachable"/"safe" answer).

The failure mechanism is exactly the one to watch for: **enlarging `G` at a state creates new
proof obligations at its predecessors.** Size alone carries no information.

### A.3 The correct characterisation

```
⊑_G is a simulation                  ⟸    G is an A-map ([THESIS] Def. 3.22, Thm. 3.23)
G is an A-map                        ⟺    G is a post-fixpoint of the §1.2 operator F
                                          (i.e. G(q) ⊇ F(G)(q) for all q)
G* = lfp F                           =    the ⊆-least A-map (Knaster–Tarski)
                                     =    the ⊇-coarsest simulation of this family (by A.1)
```

**[DERIVED] Lattice structure of the sound family.** `F` is monotone and, because `pre` acts
constraint-by-constraint, `F(G ∪ H) = F(G) ∪ F(H)` and `F(G ∩ H) ⊆ F(G) ∩ F(H)`. Hence the
A-maps of a fixed automaton are **closed under both pointwise union and pointwise
intersection**, forming a complete lattice with bottom `G*` and top "everything in `Φ_C`".
Practical readings:

* Any two sound maps can be combined by `∩` to get a sound, *coarser* map. (Useful: run two
  independent contraction heuristics and intersect.)
* `G*` is the unique best choice; every other sound map is strictly worse at pruning.
* You may **not** freely add constraints. You may add a constraint `φ` to `G(q)` only if you
  also add `pre(φ, R)` to every predecessor `p` of `q` along `(p,g,R,q)`, transitively — i.e.
  adding requires a *backward closure sweep*.

**[DERIVED] A slightly larger sound family (non-diagonals only).** By [THESIS] Cor. 3.6/3.10,
`⊑_{x ◁′ c′} ⊆ ⊑_{x ◁ c}` whenever `(◁′,c′) ≥ (◁,c)`, and dually for lower bounds. So in
Definition 3.22 the requirement "`G(q)` *contains* `φ`" can be weakened to "`G(q)` contains a
**non-diagonal** constraint on the same clock and side that **dominates** `φ`". This is why the
classical local-`L/U` analysis stores one integer per clock per side and is still sound.
**This relaxation is invalid for diagonals**: `⊑_{x−y ≤ 5} ⊄ ⊑_{x−y ≤ 0}` (take `v: x−y = 0`,
`v′: x−y = 3`). Diagonal constraints must be present *literally*.

---

## B. CONTRACTION — starting above `G*` and shrinking during the search

### B.1 (i) Does shrinking preserve soundness at every intermediate point?

**Refuted as stated; confirmed with the corrected invariant.**

* "Never go below `G*`" is **necessary but not sufficient**: §A.2's counterexample map is
  strictly above `G*` and unsound.
* The correct invariant, which must hold at *every* instant, is:

  > **INV: the current family `G` is an A-map — for every state `q` and every outgoing
  > transition `(q,g,R,q′)`: `atoms_of(g) ⊆ G(q)` and `pre(G(q′), R) ⊆ G(q)`.**

  (`INV ⇒ G ⊇ G*` automatically, so the "never below `G*`" side condition is subsumed.)

* **[DERIVED] Legality of a single removal.** Removing `φ` from `G(q)` preserves INV iff

  ```
  ∀ outgoing (q,g,R,q′):   φ ∉ atoms_of(g)   and   φ ∉ pre(G(q′), R)
  ```

  — using the *current* `G(q′)` of the immediate successors, and (for non-diagonal `φ`) modulo
  the domination relaxation of §A.3.

* **[DERIVED] Removals never create obligations elsewhere.** Every INV clause constrains `G(q)`
  *from below* in terms of `G(successor)`. Shrinking `G(q′)` only *weakens* the clause at every
  predecessor of `q′`; it can never break it. So a removal can only violate the removing state's
  **own** clauses — which is what the legality test above checks. Contraction is therefore
  *stable*: legal removals compose in any order and any interleaving, and the family stays an
  A-map throughout. Removing at `q′` only *enables* further removals at predecessors — the
  enabling wave travels **backwards** along transitions.

* **[DERIVED] The direction is exactly right for your engine.** By A.1, each removal makes `⊑`
  strictly coarser, so `↓_{q,G} Z` grows for every stored `Z`, so
  `Cov(q) = ⋃_{stored Z} ↓_{q,G} Z` **grows**. Coverage monotonicity is preserved by *both*
  engine events: inserting a zone, and shrinking `G`. This is the dual of CAV'13 lazy
  abstraction (Herbreteau–Srivathsan–Walukiewicz, arXiv:1301.3127), which *"dynamically updates
  LU-bounds during exploration … refined only when they enable a transition that is impossible
  in the unabstracted system"* — bounds only **grow** there, so the covered region **shrinks**,
  which is exactly what a concurrent read probe cannot tolerate. Your reading of the two
  directions is correct.

* **[DERIVED] Caveat that will bite: descending iteration does not reach `G*`.** Contracting is
  a descending Kleene iteration from a post-fixpoint; it converges to the **greatest fixpoint
  below the starting map**, not to `lfp F`. Concrete stall: a self-loop `(q, true, ∅, q)` with
  `{x ≤ 5} ∈ G(q)` — `F(G)(q) ⊇ pre({x≤5}, ∅) = {x≤5}`, so `{x≤5}` justifies itself forever,
  although `G*(q) = ∅`. Any reset-free cycle (extremely common in networks: a process idling
  while others move) makes constraints self-supporting. **Remedy:** this is precisely
  *incremental least-fixpoint maintenance under deletion* — the standard solutions are DRed
  (delete-and-rederive: over-delete the transitive consequences, then re-derive what still has a
  guard-grounded support) or support/derivation-depth counting (store with each `φ ∈ G(q)` the
  minimal number of backward steps to a guard atom, and drop `φ` when no finite-depth support
  remains). Do not use naive "is it still derivable from my successors" — that is the
  self-justifying test that stalls. This is standard Datalog-maintenance machinery, not
  timed-automata-specific.

### B.2 (ii) Do already-stored zones stay valid when `G` shrinks?

**Yes — no re-checking, and stored zones become strictly more powerful subsumers.**
Three separate statements, all **[DERIVED]** but each a one-line consequence of A.1:

1. **Nothing stored depends on `G`.** This is *the* structural advantage of the simulation
   approach over the extrapolation approach: zones are stored **exactly**, and `G` appears only
   as a parameter of the test `Z ⊑_{G(q)} Z′` ([SURVEY] §5, p. 12: *"The simulation approach, in
   principle, eliminates this problem"* — the problem being that *"extrapolations need to store
   abstracted zones explicitly"*). Therefore a change to `G` cannot invalidate any stored
   object. **Hard requirement: do not write extrapolated zones into the passed list if you
   intend to contract `G`.** (If PTCTLMC ever calls `dbm_extrapolate_lu_plus` on a zone before
   storing it, that zone *does* depend on the bounds and the argument above no longer applies to
   it. Subsumption via `dbm_is_alu_le` on unmodified zones is fine.)
2. **Past pruning decisions stay valid.** A node discarded earlier because `Z ⊑_{G_old} Z″` was
   discarded under a relation that was a genuine simulation at the time (INV held). Since
   `⊑_{G_old} ⊆ ⊑_{G_new}`, the same pair is still related after the shrink. Nothing to redo.
3. **Stored zones get stronger, not weaker.** After the shrink, a stored `Z″` subsumes a
   superset of what it subsumed before. The *only* consequence is that the passed set may stop
   being a strict antichain (some stored `Z₁` may now be subsumed by stored `Z₂`). That is a
   memory/efficiency issue, not soundness; clean it lazily or never.

**[DERIVED] Mixed-generation concurrency is safe — you do not need a consistent snapshot.**
Two facts:

* *Composition.* If `⊑₁` and `⊑₂` are simulations then so is `⊑₁ ∘ ⊑₂` (standard: replay the
  delay/transition witnesses through the middle configuration; both conditions of [SURVEY]
  Definition 1 go through). So a covering chain `Z ⊑_{G_gen3} Z′ ⊑_{G_gen7} Z″` built by
  different threads at different generations is still a valid covering. Antichain "kill"
  operations across generations are therefore safe.
* *Stale reads are conservative.* A thread that reads an **older, larger** `G(q)` runs a
  **finer** test (A.1) — it may fail to subsume something that is in fact subsumed. That costs
  work, never correctness. So the probe side needs no synchronisation beyond tear-free
  publication of the constraint-set pointer.
* *Requirement.* `G(q)` must be **monotonically non-increasing** over the run. A lost update
  that resurrects a removed constraint would re-introduce an obligation at predecessors that
  have already shrunk, breaking INV. Publish with CAS/`release` and reject any store that is not
  a subset of the current value (or take a per-state lock — contention is negligible, this runs
  once per location vector per contraction round).

### B.3 (iii) Has anyone published the contracting direction?

**NOT ESTABLISHED IN SOURCES — I found no algorithm that starts from an over-approximate
constraint map / bound map and contracts it during the search.** What exists is uniformly the
*refining* (growing) direction:

* Herbreteau, Srivathsan, Walukiewicz, *Lazy abstractions for timed automata*, CAV'13,
  arXiv:1301.3127 — "dynamically updates LU-bounds during exploration"; bounds start minimal and
  are **increased** on demand when an abstraction admits an infeasible transition, then
  propagated backwards. Abstraction gets **finer** over time.
* Roussanaly, Sankur, Markey, *Abstraction refinement algorithms for timed automata*, CAV'19,
  arXiv:1905.07365 — CEGAR over zone/predicate abstractions; refinement only.
* Tóth, Majzik, *Lazy reachability checking for timed automata using interpolants*, FORMATS'17 —
  interpolant-driven refinement; refinement only.
* [CAV19] §3 / [THESIS] Ch. 3 compute `G` **statically, once, before the search** (`lfp` by
  [THESIS] Algorithm 2). Neither they, [SURVEY], [CONCUR18] nor [FSTTCS20] discusses recomputing,
  weakening or specialising `G` during exploration.

Searched: the five primary sources cover-to-cover, plus web searches for lazy/dynamic/adaptive
LU bounds, coarsening/weakening of clock bounds, and antichain+simulation for TA. If a
contracting scheme exists it is not in the mainline zone-based-TA literature. **Treat it as
novel** — and note the two things that make it *specifically* attractive here and are, as far as
I can tell, unremarked anywhere: (a) contraction is the unique direction compatible with a
concurrent monotone-coverage probe, and (b) for networks the natural *starting* over-approximation
(§5) is free, whereas the exact `G*` over the product is not.

---

## C. LOCAL SHRINK — can one location vector be contracted independently?

**Yes. No global barrier, no global recomputation. One-step successor lookahead, and stale
reads are conservative.** [DERIVED], from the shape of [THESIS] Definition 3.22.

**Why it is local.** The A-map conditions are indexed by *transitions* and each one constrains
the **source** state's set in terms of the **target** state's set. So:

```
the clauses that mention G(q̄) as the constrained side   =   { clauses of q̄'s OUTGOING transitions }
the clauses that mention G(q̄) as the constraining side  =   { clauses of q̄'s INCOMING transitions },
                                                             and there G(q̄) appears only on the
                                                             right of a ⊆, so shrinking it RELAXES them.
```

Hence a removal at `q̄` can violate only `q̄`'s own clauses. **The legality test is:**

```
can_remove(q̄, φ):
    for each outgoing product transition (q̄, g, R, q̄'):     # requires q̄ FULLY EXPANDED
        if φ ∈ atoms_of(g):                 return false
        if φ ∈ pre(G[q̄'], R):               return false     # reads successors' CURRENT sets
    return true
```

Consequences for the engine:

* **Per-stripe background thread: yes.** Contraction of the location vectors owned by a stripe
  needs read access to the constraint sets of their successors (possibly in other stripes) and
  write access only to its own. No barrier, no global fixpoint pass.
* **Stale successor reads are safe.** In a shrink-only regime `G_actual(q̄′) ⊆ G_stale(q̄′)`, so
  `pre(G_actual, R) ⊆ pre(G_stale, R)`; if `φ ∉ pre(G_stale(q̄′), R)` then a fortiori
  `φ ∉ pre(G_actual(q̄′), R)`. A stale read can only make a removal look **illegal** when it is
  actually legal — never the reverse. So the contraction thread may read successors without
  locks.
* **Precondition: `q̄` must be fully expanded.** All outgoing product transitions of `q̄` must be
  known before any removal at `q̄` (an unenumerated transition could be the one requiring `φ`).
  Natural hook: run contraction on a location vector when it is popped from waiting *after* its
  successors have been generated. Until then it keeps its compositional value `G_∪(q̄)`, which is
  a valid A-map value by §5 — so INV holds from time zero with no initialisation pass.
* **What contraction is actually computing.** `G*_N(q̄)` over the *product* differs from `G_∪(q̄)`
  for two reasons: (a) local transitions of a process that cannot fire at `q̄` (no synchronisation
  partner) contribute their guard atoms to `G_p` but not to `G*_N(q̄)`; (b) constraints propagated
  in `G_p` from `P_p`-locations that are unreachable in combination with the other processes'
  locations. Contraction driven by the search removes exactly these, because the search sees the
  *real* outgoing transitions of `q̄` and the *real* successors. Framed differently: **contraction
  = lazily computing the product least fixpoint from above, seeded for free by the compositional
  over-approximation.** How much this actually gains on real models is
  **NOT ESTABLISHED IN SOURCES** — no measurements exist for `G` on networks.
* **Only reachable vectors matter.** INV only has to hold on the part of the product reachable
  from the initial node (the covering argument for completeness never leaves it), so contraction
  never needs to touch unexplored vectors.
* **Ordering heuristic [DERIVED].** Since the enabling wave travels backwards, a contraction pass
  in reverse-BFS / reverse-topological order of the explored graph removes the most per sweep;
  on SCCs, pair it with the DRed/support-counting remedy of §B.1 or the pass will stall at the
  greatest fixpoint below `G_∪`.
* **Memory cost [DERIVED].** Before contraction, `G(q̄)` need not exist as an object: it is the
  union (or per-clock `max`) of `n` per-process sets, computable in `O(n)` from the vloc — this
  is what both TChecker paths do (§4.2, §4.3). Contraction requires *materialising* a per-vloc
  set for every contracted location vector, i.e. `O(|reachable vlocs| · |G|)` extra memory.
  Mitigations: contract only vlocs that are subsumption hot-spots (large antichain buckets);
  store the contracted value as a **delta** (bitmask of removed indices) against the
  compositional baseline — for a diagonal-free model this is one bit per per-clock bound, and
  the "shrink-only" discipline of §B.2 becomes a monotone bitmask-OR, which is trivially
  lock-free and word-tearing safe.

---

## D. Minimal implementation checklist

1. Compute `G_p` per process by §1.4 (guards **and invariants** into the seed). Diagonal-free
   model ⇒ `G_p` is diagonal-free ⇒ collapse to one `(◁,c)` upper and one lower bound per clock
   ([THESIS] Cor. 3.6/3.10).
2. `G(q̄)` = per-clock `max` of the `(◁,c)` pairs over processes — *identical* to your existing
   `bounds(s.locs.data(), Ls, Us)`, plus a strictness bit per bound. Sound by Prop. 5.1.
   If the model has diagonal guards, use Prop. 5.3's saturated union instead.
3. Replace `dbm_is_alu_le` with the §2.3 loop: same three conditions, `(◁₁,c)`/`(◁₂,d)` pairs
   instead of `U[x]`/`L[y]` integers, iterate over `G_u × G_ℓ` rather than all clock pairs.
   `O(|X|²)` worst case, less when `|G| < 2|X|`.
4. Do **not** extrapolate zones you store (§B.2 point 1).
5. If you add contraction: maintain INV (§B.1), enforce shrink-only publication (§B.2), gate
   removals on full expansion of the vloc (§C), and use support counting rather than
   re-derivability to avoid the reset-free-cycle stall (§B.1).
