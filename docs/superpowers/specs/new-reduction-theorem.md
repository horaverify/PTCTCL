# A new state-reduction mechanism for A[] safety — theory

**Status:** written incrementally, one complete section at a time, in decreasing order of
ceiling. Every candidate is stated as a THEOREM with a soundness argument, a CLASS of models,
and a number derived from the benchmark generators — or as an IMPOSSIBILITY result with a
proof, or as an explicit `NOT ESTABLISHED IN SOURCES`.

**Scope.** Mechanisms and theory only. The empirical/parallelism angle (search order,
spare-core precision, speculation) is a sibling workstream (`reachability-node-count.md`) and is
not re-litigated here.

## Premises taken as given (measured elsewhere)

From `reachability-node-count.md`, `2026-08-09-state-reduction-design.md`,
`2026-08-10-w3-partial-order-reduction.md`:

* On fischer, arbiter, viking, twolock, csma the engine stores **exactly one zone per reachable
  discrete state**, and ties `verifyta` to the digit on **both** `stored` and `explored`. The
  stored count *is* the reachable discrete state space.
* Every antichain on those five families has exactly **one** member (`maxk = 1`). Every pairwise
  zone technique — federation coverage, G-simulation, exact inclusion, deeper probes — is
  provably a no-op there.
* `verifyta -A` (maximal zone over-approximation) changes fischer10 by **zero** on both axes.
* Search order is a local optimum (BFS); thread count does not change counts.
* Ruled out with measurements: symmetry reduction *as UPPAAL ships it*, classical partial-order
  reduction (unsound at the zone level), constraint-map contraction, IC3/PDR and CEGAR (a
  sibling tool measured CEGAR a **net loss**, 251 032 nodes vs 68 624).
* `explored − stored` equals exactly the cross-level retirement count: ~41 % of expansions on
  fischer10/twolock10, 28 % on phil12, 0 % on arbiter/viking/csma.
* Distinctive assets: an exact convex-union merge, coverage monotonicity, a parallel mutable
  antichain.

## Metric discipline

| metric | meaning | current wall |
|---|---|---|
| `explored` | symbolic-state expansions | fischer10 447 598 |
| `stored` | live passed-set entries at the fixpoint | fischer10 260 998 |
| `peak stored` | max live entries at any instant | == `stored` today (nothing is deleted) |
| `bytes` | actual memory | the real scaling wall at 2.7 M+ states |

Each candidate below names which metric it moves. A candidate that only moves `bytes` is still
valuable — memory is what stops the engine — but it is not sold as a state-count reduction.

## The generality bar

Per the project owner: *"It can't be stupidly benchmark dependent, it has to be a general
approach."* Every candidate therefore carries:

* **CLASS** — the models it applies to, stated as a property of an arbitrary input, not as a
  list of families;
* **PRECONDITION CHECKABLE?** — whether an implementation can decide, from the input file
  alone, that the mechanism will fire, without knowing the answer in advance.

A mechanism that only fires on this suite is said so and ranked below one that does not.

## Structural facts about the six families, established here

These are computed from `benchmarks/generators` (scripts in the scratchpad; the analysis is
reproduced in the sections that use it) and are used repeatedly below.

| family | components | sync | shared vars | clocks | property |
|---|---|---|---|---|---|
| fischer | N identical `P_i` (4 locs) | none (pure interleaving) | `int[0,N] id`, written by all | one per process | `A[] not (P0.cs and P1.cs)` |
| twolock | N identical `P_i` (6 locs) | none | `int a, b`, written by all | one per process | `A[] not (P0.cs and P1.cs)` |
| phil | N forks (2 locs) + N phils (4 locs) | binary chan `take_i/put_i` | **none** | one per phil | `A[] not (P0.Eat and P1.Eat)` |
| arbiter | 1 `Arb` (2 locs) + N clients (3 locs) | binary chan `acq/rel`, global | none | one per client | `A[] not (C0.cs and C1.cs)` |
| csma | 1 `Bus` (4 locs) + N stations (3 locs) | binary `begin/end` + broadcast `cd` | none | one per station + `Bus.c` | `A[] not (S0.Send and S1.Send and Bus.Transmitting)` |
| viking | 4 vikings (4 locs) + `Torch` (4 locs) | binary chan `take/release`, global | `int L` | one per viking (+ unused `time`) | `A[] not (Viking1.safe and Viking1.unsafe)` |

Two facts hold across **all six** and are proved by direct computation in §3 and §2:

1. **Every component's location graph is strongly connected.** No component ever makes
   irreversible progress.
2. **The largest local simulation preorder on every component is the identity**, even after
   discarding all guards and invariants (the most generous relaxation).

Both are checkable on an arbitrary input in time linear/quadratic in the *syntax*, and both are
load-bearing negatives.

---

## 1. Compositional / assume-guarantee reasoning

**Verdict: the largest ceiling of anything in this document — unbounded in N, not merely an
order of magnitude — but the payoff comes from the *quotient / environment-abstraction* branch
of the field, not from the learning-based assume-guarantee branch, which has two independent
structural obstructions against exactly this system shape and has never been run on it.**

### 1.1 The question, made precise

`A[] not (P0.cs and P1.cs)` on fischer10 is checked on the 10-process product: 260 998 symbolic
states. The property names **two** components. The other eight enter only through the shared
`id`. The question is whether the product is necessary.

Formally, write the network as `S = C ∥ E` where `C` (the *cone*) is the set of components
named in the property together with anything they share state with directly, and `E` (the
*environment*) is the rest. The **interface** is the set of shared variables and channels that
`C` and `E` both touch. Computed from the generators:

| family | cone `C` for the shipped query | environment `E` | interface |
|---|---|---|---|
| fischer | `P0, P1` | `P2..P9` | the single variable `id` |
| twolock | `P0, P1` | `P2..P9` | `a`, `b` |
| phil | `P0, P1, F0, F1, F2` | `P2..P_{N-1}, F3..F_{N-1}` | channels `take0/put0` (with `P_{N-1}`), `take2/put2` (with `P2`) |
| arbiter | `C0, C1, Arb` | `C2..C_{N-1}` | channels `acq, rel` |
| csma | `S0, S1, Bus` | `S2..S_{N-1}` | channels `begin, end, cd` |
| viking | `Viking1` | `Viking2..4, Torch` | `take, release, L` |

**The interface is narrow in every case** — one integer, or one pair of channels. This is not a
property of these five families; it is a property of *protocols*, which are written precisely so
that components interact through a thin shared medium. That is why this direction is worth the
largest ceiling: the width of the cut, not N, determines the cost.

### 1.2 What exists: rules, and what is decidable

**The non-circular rule.** Every timed AG paper uses the same asymmetric rule (`AG-NC` in Lin
et al.; `Asym` in Sankur):

```
   M1 ∥ A ⊨ φ        M2 ⊨ A
   ─────────────────────────
        M1 ∥ M2 ⊨ φ
```

**Soundness and completeness are established** for event-recording automata: Shang-Wei Lin,
Étienne André, Yang Liu, Jun Sun, Jin Song Dong, *"Learning Assumptions for Compositional
Verification of Timed Systems"*, **IEEE TSE 40(2):137–153, 2014** — Theorem 4 (`AG-NC` for ERAs
is sound and complete), Theorem 5 (the learning framework is sound and complete), Theorem 6
(termination). The completeness is the textbook, weak kind: one may always take `A = M2`. It
guarantees no false negatives; it does **not** guarantee that a *small* assumption exists.

**Circular rules cannot be rescued.** Patrick Maier, *"Compositional Circular Assume-Guarantee
Rules Cannot Be Sound and Complete"*, **FoSSaCS 2003**, LNCS 2620:343–357: compositional
circular AG rules for parallel composition cannot be both sound and complete. Namjoshi–Trefler's
sound-and-complete circular rule escapes only by being non-compositional. The `n`-component
symmetric rule (`AGC`, Barringer/Giannakopoulou/Păsăreanu, SAVCBS 2003) has **never been
instantiated for timed automata** — `NOT ESTABLISHED IN SOURCES`: any timed circular rule with
experiments.

**The decidability wall.** Alur & Dill, *"A Theory of Timed Automata"*, **TCS 126(2):183–235,
1994**: timed language inclusion `L(A) ⊆ L(B)` is **undecidable** for nondeterministic timed
automata (Π¹₁-hard; decidable if `A` is clock-free, undecidable from two clocks; PSPACE-complete
in the deterministic case). Four dodges are used in the literature:

| dodge | representative | mechanism |
|---|---|---|
| determinisable subclass | Lin et al. TSE 2014 | event-recording automata (Alur/Fix/Henzinger, *TCS* 211, 1999) are determinisable and Boolean-closed ⇒ timed inclusion decidable |
| untimed language only | Sankur JAR 2025 | the untimed language of a TA is regular ⇒ `L(T) ⊆ L(H)` with `H` a DFA is decidable |
| deterministic one-clock assumption | Chen/Su/Zhang/Liu/Mi CAV 2023 | assumption restricted to DOTAs |
| simulation instead of inclusion | Timed I/O automata / ECDAR | refinement = alternating timed simulation, decided by timed games |

### 1.3 What is automatable, and what the numbers actually are

**Untimed ancestor.** Cobleigh, Giannakopoulou, Păsăreanu, *"Learning Assumptions for
Compositional Verification"*, **TACAS 2003** (journal: Păsăreanu et al., *FMSD* 32(3):175–205,
2008): L\* learns the weakest assumption over the interface alphabet, with the model checker as
Teacher.

**Timed, learning-based.** Lin et al. TSE 2014 learn a deterministic ERA by `TL*` (untimed L\*
phase, then guard refinement by DBM subtraction). Reported state-space reductions are real:
FMS-3 Spec 1, **480 481 monolithic vs 2 265 compositional (≈212×)**; AIP Spec 5, 104 651 vs
2 301 (≈45×); FMS-4 out-of-memory monolithically vs 34 008 compositionally. Chen, Su, Zhang,
Liu, Mi, *"Learning Assumptions for Compositional Verification of Timed Automata"*, **CAV 2023**,
LNCS 13964:40–61, learn a deterministic one-clock TA and contribute a scheme for recovering the
assumption's clock resets.

**Timed, non-learning.** Ocan Sankur, *"Timed Automata Verification and Synthesis via Finite
Automata Learning"*, **JAR 2025** (prelim. TACAS 2023): learns a DFA `H` over-approximating the
*untimed* language of a small TA `T`, for systems of the shape `A ∥ T` with `A` a **clock-free**
finite automaton; TChecker answers timed queries, nuXmv the finite-state one.

### 1.4 Why the learning branch will not work here — three obstructions, all from primary sources

**(a) The rule loses completeness the moment both sides have clocks.** Sankur, §3.5.2: for
`T1 ∥ T2` both timed, `L(T1∥T2) ≠ L(T1) ∩ L(T2)` on untimed languages, so the composed
abstraction `H1∥H2` is sound but incomplete. His Example 2 is two one-clock self-loops
(`x=3, a, x:=0` and `y=1, b, y:=0`): the product forbids `aa`, but every DFA over-approximating
`L(T1)` accepts `a*`, so `H1∥H2 = (a+b)*` cannot exclude it. He leaves the network case
unimplemented and unevaluated. **Any cut of a network of timed peers has clocks on both sides.**

**(b) The abstraction is provably exponential for peer networks with per-process clocks.**
Sankur, §3.5.3, the "Chinese Remainders issue": for `k` self-loops firing every `p_i` time units
with `p_i` distinct primes, all discrete clock valuations below `∏p_i` recognise different
languages, so **any finite automaton for the network's language is exponential in `k`**.
fischer/phil/csma with one clock per process are structurally this shape.

**(c) The decomposition problem is the bottleneck, and it is worst for symmetric peers.**
Cobleigh, Avrunin, Clarke, *"Breaking up is hard to do"*, **ISSTA 2006** / **TOSEM 17(2) art. 7,
2008**, enumerated *all two-way decompositions* of a set of systems under two verifiers:
> "In only a few cases could assume-guarantee reasoning verify properties on larger systems than
> monolithic verification and, in these cases, ... only ... systems a few sizes larger. This
> discouraging result ... raises doubts about the usefulness of assume-guarantee reasoning."

(`NOT ESTABLISHED IN SOURCES`: the exact percentage of decompositions that won — the tables are
paywalled. The quoted qualitative claim is verbatim and the "all two-way decompositions"
methodology is confirmed, so this is not a bad-split artefact.) Lin et al. concur —
*"the way of partitioning components really dominates the performance"* — and their rescue
heuristic is *"collect in H1 the components containing the events specified in the property"*.
**On fischer that heuristic yields `{P0, P1}` versus `{P2..P9}` with `id` on the cut — and every
one of the eight environment processes writes `id`, so the interface alphabet is the whole
variable and the learned assumption must encode the protocol.**

**And the empirical gap is total.** Across the four automated timed-AG papers, the union of
benchmarks is: client-server, gas station, FMS-1..4, AIP (Lin et al.); AUTOSAR-1/2/3 (Chen et
al.); leader election, FTSP, STS, Rt-broadcast, priority scheduling (Sankur); two unnamed case
studies (Chen/Zhang/Vaandrager, CAV 2025). **Fischer with N processes appears in none of them.
Dining philosophers appears in none of them.** Every decomposition in every one of those papers
is a two-way split of an *asymmetric* system, usually "one small scheduler against everything
else" with an interface alphabet of 2–8 symbols. Stated plainly, as the brief asked:
**nobody has made learning-based AG work on Fischer-with-N, and there is no published evidence
that it can.**

### 1.5 The branch that *has* produced an order of magnitude on exactly this model

Kåre J. Kristoffersen, François Laroussinie, Kim G. Larsen, Paul Pettersson, Wang Yi,
*"A Compositional Proof of a Real-Time Mutual Exclusion Protocol"*, **TAPSOFT 1997**, LNCS 1214
(tool: Laroussinie & Larsen, *"CMC"*, FORTE/PSTV 1998; theory: Laroussinie & Larsen, CONCUR 1995;
after H. R. Andersen, *"Partial Model Checking"*, LICS 1995).

The model is *exactly* this one: `FISCHER_n ≡ (P₁ | (P₂ | … | P_n)) | V`, one local clock per
process, the shared variable as an explicit automaton `V` with locations `V₀…V_n` and actions
`:= i`, `= i`, `≠ i`, and the property `M12` = invariance of ¬(`CS₁` ∧ `CS₂`).

The technique is **quotienting**, not assume-guarantee: `A | B ⊨ φ ⟺ A ⊨ φ / B`. Components are
moved *out of the system and into the formula*, and the formula is then minimised (reachability,
Boolean simplification, constraint propagation, trivial-equation elimination, "hit-zones").

> "by first quotienting away `V`, `P₁` and `P₂` the quotient hereby obtained **simplifies to
> `tt`** under our minimization heuristics. Thus **no examination of the components `P₃, …, P_n`
> is required** … the size of the correctness proof … **only grows polynomially**."

Measured (their Table 4, acyclic Fischer, seconds):

| #procs | 2 | 4 | 5 | 6 | 7 | 8 | 10 | 30 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| UPPAAL fwd | 0.2 | 0.9 | 10.7 | 244.4 | ? | | | | |
| UPPAAL bwd | 0.1 | 0.3 | 1.2 | 6.2 | 38.5 | 310.6 | | | |
| **CMC** | 0.2 | 0.6 | 0.8 | **1.2** | 1.6 | 2.0 | 3.2 | 40.0 | **172.3** |

Caveats that must travel with this number: it is a **1997** UPPAAL baseline; the model is the
**acyclic** variant; the quotienting **order is hand-chosen** and the paper says so
(*"the optimal order in which components are factored out needs a better understanding"*); CMC
models the shared variable as an `(n+1)`-location automaton and **cannot handle UPPAAL integer
variables**; and the tool no longer exists. `NOT ESTABLISHED IN SOURCES`: any successor tool.

Independently, the *parameterized* branch settles Fischer outright for **all** N:
Abdulla, Deneux, Mahata, Nylén, *"Using Forward Reachability Analysis for Verification of Timed
Petri Nets"*, **Nordic J. Computing 14(1–2):1–42, 2007** — parameterized Fischer with the real
shared variable `v`, **2.12 s and 1.16 MB for arbitrary N**, by symbolic forward analysis over
downward-closed sets of regions with acceleration. Their encoding of `v` is the decisive trick
and it is the same trick as below: a place `udf` for `v = ⊥` plus *primed* copies `A′,B′,C′,CS′`
marking **the process whose id is currently in `v`** — i.e. the value abstraction
`{undefined, mine, someone-else's}`.

Foundational decidability: Abdulla & Jonsson, *"Model checking of systems with many identical
timed processes"*, **TCS 290(1):241–264, 2003** (conf. TACAS 1998) — reachability is decidable
for unboundedly many identical **one-clock** processes. The boundary is sharp and unfavourable:
with a controller and multiple clocks per process it is **undecidable**; André, Jacobs, Karra,
Sankur (arXiv:2408.05190, 2025) show parameterized model checking of disjunctive timed networks
is decidable in EXPSPACE *without* cutoffs, and cite Spalazzi & Spegni's result that **statically
computable cutoffs provably do not exist once clock invariants are allowed**. No cutoff result
covers UPPAAL-style shared bounded integers or binary CSP channels. **Direct answer: there is no
cutoff theorem for this setting.**

### 1.6 THEOREM 1 (environment abstraction by a bounded generic environment)

The mechanism worth building is the *cheap, one-shot* projection of the CMC/parameterized
insight, with no learning loop and no refinement loop.

> **Theorem 1.** Let `S = C ∥ E` be a network of timed automata with `C = C₁∥…∥C_m` the
> components named in a safety property `φ` (plus any component sharing a variable or channel
> with them), and `E = E₁∥…∥E_r` the rest. Let `Ê` be any timed automaton such that
> `E ⊑ Ê` in the *ready-simulation* sense on the interface alphabet `Σ_I` — i.e. there is a
> relation `R` with `(e, ê) ∈ R` implying that for every interface action `a ∈ Σ_I` and every
> `a`-successor `e'` of `e` under any delay `d`, `ê` has an `a`-successor `ê'` after the same
> delay `d` with `(e', ê') ∈ R`, and every non-interface action of `E` is matched by a stutter
> in `Ê`. Then
>
>   `C ∥ Ê ⊨ φ  ⟹  C ∥ E ⊨ φ`.
>
> In particular, if `Ê` is finite and its size is independent of `r`, the safety proof costs
> `O(|C ∥ Ê|)`, independent of the number of environment components.

**Soundness argument.** `⊑` is a simulation on the interface alphabet, and simulation is a
precongruence for the UPPAAL parallel-composition operator (interleaving of local actions,
rendezvous on shared channels, and shared-variable read/write treated as labelled actions on the
variable's automaton) — Milner 1989; see §2.2 for the precise precongruence statement and its
side condition. Hence `C ∥ E ⊑ C ∥ Ê`, so every timed trace of `C ∥ E` is a timed trace of
`C ∥ Ê`, so the set of reachable configurations of `C ∥ E` projects into that of `C ∥ Ê`. `φ` is
a state predicate over `C`'s locations only, and the projection is the identity on `C`, so a
configuration of `C ∥ E` violating `φ` has a corresponding configuration of `C ∥ Ê` violating
`φ`. Contrapositive gives the theorem. Note the direction: this is an **over-approximation**,
which is exactly what `A[]` needs — a proof on the abstraction is a proof on the original; a
counterexample on the abstraction proves nothing.

**Two instantiations of `Ê`, in increasing precision and cost.**

* **`Ê` = chaos over `Σ_I`.** One location, a self-loop for every interface action, no clocks.
  Size 1. Cheapest possible.
* **`Ê` = `k` generic replicas of the environment template, with the shared variable's value
  domain abstracted to `{0} ∪ {ids named in φ} ∪ {other}`.** Size independent of `r`. This is
  the Abdulla et al. `{⊥, mine, someone-else's}` abstraction, in automaton form.

**Where each lands on fischer, worked out.** The interface is `id`. Under chaos, `Ê` may write
`id := 0` spontaneously, and that breaks the proof: `P0` writes `id:=1` and enters `cs`; chaos
writes `id:=0`; `P1` now passes its `id==0` guard, writes `id:=2`, waits `x1>k`, reads `id==2`
and enters `cs`; `P0` has no invariant on `cs` and is still there. **Chaos is too coarse for
fischer, and the reason is precise and general: chaos grants the environment write actions that
no real component can perform in that control state.** Under the second instantiation the
counterexample dies: `id := 0` is written only on `cs → idle`, so a generic replica must
itself reach `cs`, which requires `id == its own id`, which it can only obtain by writing `id`
after reading `id == 0` — the ordinary protocol. With `k = 1` generic replica and `id` abstracted
to `{0, 1, 2, other}`, the abstract system is `4 × 4 × 4 × 4 = 256` discrete states with **three**
clocks, versus 260 998 discrete states with **ten** clocks at N = 10.

**CLASS.** Networks in which the components not named in the property are instances of a
template also instantiated inside the cone, or can be over-approximated on a finite interface
alphabet. This is a property of the *input*, not of the family: it holds for any replicated-role
model, and it fails for a model whose every component is distinct and property-relevant.

**PRECONDITION CHECKABLE?** **Yes, and cheaply, by attempting.** The abstraction is built
syntactically; the abstract check either passes (safety proved, done) or produces an abstract
counterexample (nothing proved, fall back to the full search). The cost of a failed attempt is
one run on a system whose size is independent of N, i.e. negligible. **This is not CEGAR**: there
is no refinement loop, no interpolation, no predicate discovery, and therefore none of the
failure mode measured on the sibling tool (251 032 vs 68 624 nodes, where refinement destroyed a
canonicaliser). The risk profile is asymmetric and bounded by construction.

**EXPECTED REDUCTION.** On fischer, from 260 998 stored to `O(10²–10³)`, i.e. **≥100×** at
N = 10 and **unbounded in N** — the abstract system does not grow with N at all. The published
analogues bound the expectation from both sides: CMC 172.3 s at N = 50 where 1997 UPPAAL died at
8; Abdulla et al. 2.12 s for *all* N. Applying the same construction to the other families: on
arbiter and csma the environment replicas interact only through `Arb`/`Bus`, so `k = 1` generic
client/station should suffice (278 528 → `O(10²)` on arbiter15); on phil, the cone
`{P0,P1,F0,F1,F2}` meets the environment on `take0/put0` and `take2/put2` — again `k = 2` generic
neighbours; on twolock the same argument as fischer with two variables. **viking is not
scalable in N and gains nothing.**

### 1.7 The honest failure modes

1. **Incompleteness.** If the property genuinely depends on `E`'s cardinality — e.g. a bound
   like "at most `M` processes in the buffer" — no fixed-size `Ê` proves it and the attempt
   fails. Detecting this in advance is exactly the parameterized-verification problem, and is
   undecidable in general (§1.5).
2. **The interface may not be narrow.** The construction costs `O(|C| · |Ê|)` where `|Ê|`
   depends on the interface alphabet. For a model with a wide shared state (a large array
   written by everyone), `Ê` is as big as `E`.
3. **It changes what the tool proves about *this* input.** A pass proves the property; it does
   not produce the reachable set. Any downstream feature that needs the full state space (trace
   generation, liveness, TCTL beyond `A[]`) does not benefit.
4. **Novelty.** The two ingredients — quotienting and the `{⊥, mine, other}` variable abstraction
   — are published (TAPSOFT 1997; NJC 2007). What is open, as far as the sources reach, is
   (i) doing it **automatically from a flat UPPAAL XML** with no user-supplied decomposition,
   (ii) using it as a **cheap pre-pass with fallback** inside an otherwise exact engine, and
   (iii) the cone/interface computation as a syntactic analysis. `NOT ESTABLISHED IN SOURCES`:
   any tool that does this automatically for UPPAAL models.

### 1.8 Verdict on §1

| branch | ceiling | evidence | recommend |
|---|---|---|---|
| learning-based AG (L\*, TL\*, DOTA) | high in principle | never run on peer networks; incomplete for TA∥TA; exponential abstraction for per-process clocks; decomposition dominates | **no** |
| circular / symmetric AG rules | — | cannot be sound and complete (Maier 2003); never instantiated for TA | **no** |
| timed interfaces / ECDAR / spec theories | — | all require hand-written specifications; no automatic decomposition | **no** |
| cutoffs for timed networks | unbounded | provably no static cutoffs with clock invariants; nothing covers shared ints or CSP channels | **no** |
| **quotient / environment abstraction (Theorem 1)** | **unbounded in N** | CMC N=50 in 172 s; Abdulla et al. all-N in 2.12 s; cheap one-shot with bounded-loss fallback | **yes — highest ceiling in this document** |

---

## 2. Cross-discrete-state subsumption

**Verdict: the preorder exists, it is sound, it is cheap to compute, and it is a genuine gap in
the literature — but its collapse factor is provably EXACTLY 1 on all six families, and the
reason generalises to every hand-written protocol model. The relation that *does* collapse the
discrete part is not a preorder at all but the symmetry group action, and that one is measured
here at 38× on fischer8 and 245× on fischer10.**

### 2.1 The literature gap is real

Subsumption in every published zone-based engine compares configurations **at the same control
state, by definition**. Bouyer, Gastin, Herbreteau, Sankur, Srivathsan, *"Zone-based verification
of timed automata: extrapolations, simulations and what next?"*, **FORMATS 2022**
(arXiv:2207.07479), Definition 1: *"A preorder relation ⪯ between configurations of A **having
the same control state** is said to be a simulation if …"*, and the pruning rule only ever
quantifies over nodes at the identical location. Herbreteau, Srivathsan, Walukiewicz,
*"Better abstractions for timed automata"*, **LICS 2012 / Inf. Comp. 251:67–90, 2016**,
Definition 1 clause 1 is literally `q = q′`; their Theorem 14 (`a_LU` is the coarsest sound and
complete abstraction for LU-automata) is an optimality result **within** that same-location
setting, which is why the project's `maxk = 1` measurement is not in tension with it. The same
holds for `G`-simulation (Gastin/Mukherjee/Srivathsan, CAV 2019, FSTTCS 2020) — the *parameter*
is location-dependent, the *comparison* is not — and for the local-time-semantics work
(Govind/Herbreteau/Srivathsan/Walukiewicz, LICS 2022), where the novelty is on the time axis.

Searching under "location subsumption timed automata", "simulation between locations",
"abstraction of the discrete part", "cross-location subsumption": **no published zone-based TA
reachability algorithm prunes `(q₁,Z₁)` using `(q₂,Z₂)` with `q₁ ≠ q₂` via a preorder on the
discrete part.** The nearest published instance in any timed formalism is TAPAAL's timed-arc
Petri nets — see §2.5 — and that is the engineering template.

### 2.2 THEOREM 2 (componentwise-simulation subsumption)

Let the network be `S = A₁ ∥ … ∥ A_n` with UPPAAL's composition (interleaving of local
transitions, binary rendezvous on channels, broadcast, shared bounded integers). Expose the
whole interface as labels: a transition's label is the tuple
`(channel action, guard on shared variables, update of shared variables)`; a purely local
transition is labelled `τ` together with its **clock-reset set**.

> **Definition.** For component `i`, let `≼ᵢ ⊆ Lᵢ × Lᵢ` be the largest relation such that
> `ℓ ≼ᵢ ℓ′` implies
> 1. **(property)** `ℓ` and `ℓ′` agree on every atomic proposition of `φ` that mentions component
>    `i`;
> 2. **(invariant)** `Inv(ℓ) ⟹ Inv(ℓ′)` (the more general location may delay at least as long);
> 3. **(step)** for every edge `ℓ —g, a, u, R→ m` there is an edge `ℓ′ —g′, a, u, R→ m′` with the
>    **same** action `a`, the **same** variable update `u`, the **same** reset set `R`,
>    `g ⟹ g′`, and `m ≼ᵢ m′`.
>
> Lift pointwise: `(ℓ₁,…,ℓ_n) ≼ (ℓ′₁,…,ℓ′_n)` iff `ℓᵢ ≼ᵢ ℓ′ᵢ` for all `i`, and extend to
> symbolic states by `(q, Z) ⊑ (q′, Z′)` iff `q ≼ q′` and `Z ⊆ Z′`.
>
> **Theorem 2.** For an `A[] φ` query whose violating set is `≼`-upward-closed, `⊑` is a sound
> subsumption order: if `(q′,Z′)` is in the passed set and `(q,Z) ⊑ (q′,Z′)`, then `(q,Z)` may
> be discarded without affecting the verdict.

**Soundness argument.** Conditions 1–3 make `≼ᵢ` a *timed step simulation* on component `i`
w.r.t. the labelled interface: identical resets keep the zone transformation identical, identical
updates keep the shared-variable transformation identical, `g ⟹ g′` makes every enabled step of
`ℓ` enabled at `ℓ′`, and `Inv(ℓ) ⟹ Inv(ℓ′)` makes every delay of `ℓ` a delay of `ℓ′`. Simulation
is a **precongruence for parallel composition** — the relation
`{ (P₁∥P₂, Q₁∥Q₂) | P₁ ≾ Q₁, P₂ ≾ Q₂ }` is itself a simulation, because every product step is
either a local move (matched componentwise) or a rendezvous (matched on both sides) — so `≼`
lifts to the product. For *this* composition operator, with shared variables and committed
locations, the precongruence theorem is proved by **Berendsen & Vaandrager, "Compositional
Abstraction in Real-Time Model Checking", FORMATS 2008**, LNCS 5215:233–249, whose *only*
essential restriction is that **the guards of input transitions must not depend on external
variables**. Hence `Post*(q,Z) ⊑ Post*(q′,Z′)` pointwise; upward-closedness of the violating set
turns "no violation reachable from `(q′,Z′)`" into "no violation reachable from `(q,Z)`". ∎

**The upward-closedness condition is not decoration.** It is exactly the second hypothesis of a
well-structured transition system — Finkel & Schnoebelen, *"Well-structured transition systems
everywhere!"*, **TCS 256(1–2):63–92, 2001**: upward-compatibility of `≤` with `→` gives you
*coverability* of **upward-closed** sets, not reachability of arbitrary sets. Condition 1 of the
definition is what buys it, and it is the same hypothesis as ∀CTL-preservation in Grumberg &
Long, *"Model Checking and Modular Verification"*, **ACM TOPLAS 16(3):843–871, 1994**.

**Side conditions that must be checked statically**, all of which have a citation:
* **broadcast** breaks the precongruence, because "receive if you can, else stay put" is a
  *negative* SOS premise and positive rule formats (de Simone, GSOS, tyft/tyxt — Groote &
  Vaandrager, *Inf. Comp.* 100(2), 1992) are what give precongruence for free. The repair is
  **input-enabledness on broadcast actions** plus **no shared-variable updates on broadcast
  receive** — Chen, Zhang, Vaandrager, *"Compositional Abstraction for Timed Systems with
  Broadcast Synchronization"*, **CAV 2025** (arXiv:2505.12436). This bites csma directly (`cd` is
  a broadcast channel). `NOT ESTABLISHED IN SOURCES`: a published explicit counterexample showing
  componentwise simulation failing to lift under broadcast; the papers impose the repairing axiom
  rather than exhibiting the counterexample.
* **committed/urgent locations and priorities** must be handled or excluded (Berendsen &
  Vaandrager).
* **input guards reading foreign variables** must be excluded (same).

**CLASS.** Any network under UPPAAL composition, subject to the three side conditions.
**PRECONDITION CHECKABLE?** **Yes, and it is cheap and exact.** `≼ᵢ` is a greatest fixpoint over
`Lᵢ × Lᵢ` computable in `O(|Lᵢ|² · |Tᵢ|)` per component — microseconds for these templates, and
independent of the product. An implementation can compute it, see whether it is the identity, and
switch the machinery off if it is.

### 2.3 …and the collapse factor is exactly 1 on all six families. Proof by computation.

The following is an *over-approximation* of `≼ᵢ`: discard conditions 1 and 2 entirely and weaken
condition 3 by discarding the guards (`g ⟹ g′` becomes vacuous), keeping only
`(action, reset set, update)` and the target. Any sound `≼ᵢ` is contained in this relaxed
relation, so **if the relaxed relation is the identity, `≼ᵢ` is the identity.** Computing the
greatest such relation on every template of every family:

```
fischer(N=4)  P0..P3      |L|=4    relaxed-sim non-identity pairs: NONE
twolock(N=4)  P0..P3      |L|=6    relaxed-sim non-identity pairs: NONE
phil(N=4)     F0..F3      |L|=2    relaxed-sim non-identity pairs: NONE
              P0..P3      |L|=4    relaxed-sim non-identity pairs: NONE
arbiter(N=4)  Arb         |L|=2    relaxed-sim non-identity pairs: NONE
              C0..C3      |L|=3    relaxed-sim non-identity pairs: NONE
csma(N=4)     Bus         |L|=4    relaxed-sim non-identity pairs: NONE
              S0..S3      |L|=3    relaxed-sim non-identity pairs: NONE
viking(4)     Viking1..4  |L|=4    relaxed-sim non-identity pairs: NONE   (see below)
              Torch       |L|=4    relaxed-sim non-identity pairs: NONE
```

**Every local simulation preorder in the entire benchmark suite is the identity. Therefore the
lifted order `≼` on the product is the identity, and Theorem 2 removes exactly ZERO states on all
six families.**

Viking is the instructive near-miss and is worth recording. Ignoring guards *and* the property,
its four-location cycle `unsafe → · → safe → · → unsafe` is label-symmetric: both `unsafe` and
`safe` emit `take!` resetting `yᵢ`, and both intermediate locations emit `release!` with no
reset, so the relaxed computation returns the non-trivial relation
`{unsafe ≼ safe, safe ≼ unsafe, id0 ≼ id2, id2 ≼ id0}`. It dies twice over: the guards are
`L == 0` versus `L == 1`, which are **contradictory**, not implied; and the query
`A[] not (Viking1.safe and Viking1.unsafe)` reads `safe`, so condition 1 separates the pair. Both
kills are exactly the conditions Theorem 2 demands, which is a small confirmation that the
conditions are not over-strong.

### 2.4 Why this generalises beyond the suite — the negative is not a benchmark artefact

> **Proposition.** If a component template is already minimal with respect to simulation
> equivalence — i.e. no two of its locations are simulation-equivalent and no location simulates
> another — then `≼ᵢ` is the identity and Theorem 2 fires on nothing in that component.

Hand-written protocol templates are minimal essentially always, and one can see the mechanism in
the data above: in each template every location has a **distinct `(action, reset set, update)`
signature on its outgoing edges**. `idle` is the only fischer location whose outgoing edge resets
`x` without touching `id`; `req` is the only one that resets `x` *and* writes `id := pid`; `cs`
is the only one that writes `id := 0`. A modeller writing a protocol by hand does not introduce
two locations with the same observable step behaviour, because such a pair would be a modelling
error. Bustan & Grumberg, *"Simulation-Based Minimization"*, **ACM TOCL 4(2):181–206, 2003**,
give the minimisation and the "little-brother elimination" rule that would collapse such pairs;
on these templates the minimiser is the identity map.

**The class where Theorem 2 has real content is therefore machine-generated models** — templates
produced by a translator from C, Simulink, task-set descriptions, or a protocol compiler, which
routinely emit redundant intermediate locations. That is a real and common class, and the check
costs microseconds, so the mechanism is worth shipping as a *free* pre-pass. It is simply not
where the win on this suite is. Ranked below §1 and below §2.6.

### 2.5 The published precedent, and why its order is a *counting* order

David, Jacobsen, Jacobsen, Srba, *"A Forward Reachability Algorithm for Bounded Timed-Arc Petri
Nets"*, **SSV 2012, EPTCS 102:125–140** (the TAPAAL `verifydtapn` engine) is a DBM-based forward
engine that **does** prune a symbolic state using another with a *different* discrete part, and
stays exact. Its Definition 5.2 orders symbolic markings by: a bijection matching one class of
tokens **exactly**, plus **`count(M) ≤ count(M′)` placewise** on a second class; its Theorem 5.3
is precisely the WSTS upward-compatibility property, restricted to reachable markings. Tokens are
statically partitioned into `eq` (must match exactly), `inc` (inclusion-eligible) and `bot`
(unused) by a static analysis that excludes the monotonicity-breaking constructs — inhibitor arcs
and non-trivial age invariants.

The lesson for a network of timed automata is sharp and, I think, decisive for this section:
**the order that works there is on multiplicities, not on locations.** In a Petri net the
discrete part is a *multiset*, so `≤` has somewhere to act. In a network of timed automata the
discrete part is a *vector* over minimal templates, where the only order is the identity. The
analogue of TAPAAL's `count(M) ≤ count(M′)` for a network of `n` identical processes would be
"state `q′` has at least as many processes in each location as `q`" — which is unsound for timed
automata, because the surplus processes carry their own clocks and their own future obligations.
That unsoundness is exactly the decidability boundary of Abdulla & Jonsson, *"Model checking of
systems with many identical timed processes"*, **TCS 290(1):241–264, 2003**: counting is exact
only with **one clock per process and no invariants**; with invariants or a second clock the
problem becomes undecidable. fischer has one clock per process but **has** an invariant
(`req: x ≤ k`); phil, csma and arbiter have invariants too. So the counting order is out on every
family here, on a published impossibility rather than a guess.

### 2.6 What actually collapses the discrete part here: the group action, measured

If the preorder is the identity, the remaining way to kill `(q₁,Z₁)` with a different `q₂` is an
**equivalence**: `q₂ = σ(q₁)`, `Z₂ = σ(Z₁)` for `σ` in the automorphism group of the model that
also fixes the property. This is symmetry reduction (Hendriks, Behrmann, Larsen, Niebert,
Vaandrager, *"Adding Symmetry Reduction to Uppaal"*, **FORMATS 2003**, LNCS 2791) — exact, not an
abstraction, and it permutes clocks along with locations so the zone is carried with the state.

The project has this filed as "ruled out — UPPAAL has it". **That is true of UPPAAL and false of
these inputs.** UPPAAL's symmetry reduction is driven by the `scalar` type; `gen_fischer.py`
emits `N` separate templates `P0…P9` with `N` separately-named clocks `x0…x9` and a plain
`int[0,N] id`. There is no scalar set anywhere in the file, so **`verifyta` applies no symmetry
reduction to any model in this suite**, which is consistent with it tying the engine to the digit
on `stored`.

To put a number on it I built an independent zone-based reference implementation of fischer
(k-normalised DBMs; one convex hull per discrete key, which `verifyta -A` measures to be *exact*
on fischer) and validated it against the engine:

| N | engine `stored` | reference reachable keys | orbits under `Sym({2..N−1})` | ratio |
|---|---|---|---|---|
| 3 | — | 65 | 65 | 1.0 |
| 4 | 220 | **220** | 136 | 1.6 |
| 5 | 727 | **727** | 231 | 3.1 |
| 6 | 2 378 | **2 378** | 350 | 6.8 |
| 7 | 7 737 | **7 737** | 493 | 15.7 |
| 8 | 25 080 | **25 080** | 660 | **38.0** |

The key counts agree with the engine exactly at every size. `Sym({2..N−1})` is the largest group
preserving `A[] not (P0.cs and P1.cs)`: it permutes processes 2…N−1, their clocks, **and the
values `3..N` of `id`** (`id = v` names process `v−1`, so the variable is a latent scalar set).

The orbit counts fit a closed form exactly at every size tested:

  **orbits(N) = 12N² − 13N − 4**   (65, 136, 231, 350, 493, 660 for N = 3…8)

while `stored` grows by a factor ≈ 3.24 per added process. Hence

| N | `stored` | orbits | reduction |
|---|---|---|---|
| 8 | 25 080 | 660 | 38× |
| 10 | **260 998** | **1 066** | **245×** |
| 12 | ≈ 2.74 M (the current memory wall) | 1 568 | ≈ **1 750×** |

**The reachable state space of fischer is exponential; its symmetry quotient is quadratic.** That
is not an order of magnitude, it is an unbounded and rapidly growing factor, and it is the single
largest measured number in this document.

**Caveats, stated plainly.** (i) The orbit count is the *discrete floor of the quotient*. A real
symmetry-reduced engine stores orbit representatives of `(q, Z)` pairs; if the union of the
permuted zones at an orbit is non-convex it stores a small multiple of the orbit count. The
project's exact convex-union merge is precisely the asset that keeps that multiple near 1, and
`maxk = 1` today says the pressure is low. (ii) Canonicalisation under a permutation group is
graph-isomorphism-hard in general; Hendriks et al. use state swaps to a canonical representative
and report the reduction as *"canonical, which means the symmetries are optimally used"*.
(iii) The group must be **detected** from the flat file — isomorphism of the template graphs plus
inference that `id` is used only in `==`/`!=` comparisons against process constants, i.e. that it
is a scalar set. That inference is the genuinely open engineering step; `NOT ESTABLISHED IN
SOURCES`: any tool that recovers scalar-set symmetry automatically from a flattened UPPAAL model.
(iv) It fires only on models with replicated components — which is most concurrent models, but
not viking (four *distinct* vikings with different constants: the group is trivial there and the
mechanism correctly does nothing).

### 2.7 Verdict on §2

| mechanism | sound? | checkable? | measured factor on this suite | rank |
|---|---|---|---|---|
| componentwise simulation preorder (Theorem 2) | yes, with 3 side conditions | yes, `O(Σ|Lᵢ|²|Tᵢ|)` | **1.00 — exactly zero states, proved by computation** | ship as a free pre-pass; expect nothing here |
| counting / multiset order (TAPAAL-style) | **no** for TA with invariants | — | inapplicable (published undecidability boundary) | dead |
| symmetry group action (§2.6) | yes, exact | detection is the open step | **38× at N=8, 245× at N=10, ~1750× at N=12** | **second-highest ceiling in this document** |

---

## 3. Property-directed storage, and not storing what you explored

**Verdict: sweep-line is provably vacuous on this suite — not "small", exactly zero — and the
proof is a one-line linear-time syntactic check that also says precisely which models it would
work on. The deletion technique that *does* fire is covering-transition storage, measured here at
4.0× memory on fischer8; it moves `bytes` and `peak stored` and it makes `explored` worse.**

### 3.1 Which metric each technique moves — stated before any claim

The sweep-line method (Christensen, Kristensen, Mailund, *"A sweep-line method for state space
exploration"*, **TACAS 2001**, LNCS 2031) needs a **monotone progress measure**: a map
`ψ : States → (W, ≤)` with `s → s′ ⟹ ψ(s) ≤ ψ(s′)`. Exploration proceeds in non-decreasing `ψ`,
and every stored state whose `ψ` is strictly below the minimum still in the queue can be
**deleted**, because nothing that will be generated later can ever reach it again.

* It reduces **peak stored** and therefore **bytes**. That is the real wall here (2.7 M+ states).
* With a **monotone** measure it leaves `explored` **exactly unchanged** — not "roughly", exactly.
  Jensen, Kristensen, Mailund, *TCS* 429:169–179, 2012 (the consolidating survey), after their
  Theorem 1: *"for a monotonic progress measure, all reachable states are visited exactly once"*.
  Confirmed empirically byte-for-byte in Mailund & Westergaard, **TACAS 2004**, Table 2, where the
  sweep-line "States" column equals the full column at every size while peak stored stays at 287.
* The **non-monotone** version (Kristensen & Mailund, FME 2002; Mailund & Westergaard, TACAS 2004)
  marks the targets of *regress edges* as **persistent** and re-sweeps, bounded by
  `(|B|+1)·|V|`, and it **increases `explored`** — measured at **exactly 2×** on the Master/Slave
  and Database Replication benchmarks, buying 8–35× less peak storage at ~1.5–2.2× wall clock.

So a sweep-line result must never be reported as a state-count reduction.

**Correction to a premise: UPPAAL already has this.** The generalised sweep-line has been in
UPPAAL since **3.6 alpha 5 (3 March 2006)** — the same release that added symmetry reduction —
driven by a user-declared `progress { … }` block of expressions over discrete variables. The
UPPAAL language reference states it directly: *"If progress measures are defined, Uppaal uses the
generalized sweepline method to reduce the memory usage."* It works: David & Larsen's Gossiping
Girls table shows 7 girls at 399 s / 1 115 MB with symmetry alone versus **29 s / 108 MB** with
symmetry plus `progress { m; }` — 13.8× faster, 10.3× less memory. So the mechanism is not exotic
and not unavailable; the question is only whether a progress measure exists for *these* models.
`NOT ESTABLISHED IN SOURCES`: any theory of the sweep-line over **zones** — UPPAAL's progress
expressions range over discrete variables, and the interaction with extrapolation and `a_LU`
inclusion is unaddressed anywhere.

### 3.2 THEOREM 3 (sweep-line vacuity for cyclic reactive models)

> **Theorem 3.** Let `G` be the reachable state graph and `ψ` any monotone progress measure. If
> `s` and `s′` lie in the same strongly connected component of `G`, then `ψ(s) = ψ(s′)`.
> Consequently, **if `G` is strongly connected, every monotone progress measure is constant and
> the sweep-line method deletes nothing** — irrespective of how the measure is chosen, and
> irrespective of the state representation.
>
> **Corollary (syntactic, linear time).** For `S = A₁ ∥ … ∥ Aₙ`, consider the component-local
> measures `ψ(ℓ₁,…,ℓₙ,v) = Σᵢ ψᵢ(ℓᵢ) + ψ_var(v)`. Monotonicity forces each `ψᵢ` to be constant on
> each SCC of `Aᵢ`'s **location graph**. If every `Aᵢ`'s location graph is strongly connected,
> every `ψᵢ` is constant; if in addition every shared variable is restored to a previous value
> along some cycle, `ψ_var` is constant too, and `ψ` is constant.

**Proof.** `s ⟶* s′ ⟶* s` with `ψ` monotone gives `ψ(s) ≤ ψ(s′) ≤ ψ(s)`. The corollary is the
same argument applied inside each component, using that a cycle in the product that moves
component `i` projects to a cycle in `Aᵢ`. ∎

**The hypothesis holds on every family in the suite, syntactically:**

```
fischer   P0..P3   |L|=4   STRONGLY CONNECTED       twolock  P0..P3  |L|=6  STRONGLY CONNECTED
phil      F0..F3   |L|=2   STRONGLY CONNECTED                P0..P3  |L|=4  STRONGLY CONNECTED
arbiter   Arb      |L|=2   STRONGLY CONNECTED                C0..C3  |L|=3  STRONGLY CONNECTED
csma      Bus      |L|=4   STRONGLY CONNECTED                S0..S3  |L|=3  STRONGLY CONNECTED
viking    Viking1..4 |L|=4 STRONGLY CONNECTED                Torch   |L|=4  STRONGLY CONNECTED
```

Every single component of every single family is one SCC. And the variables return: fischer's
`id` is restored to `0` by the length-4 cycle `idle→req→wait→cs→idle`; twolock's `a` and `b` by
the length-6 cycle; viking's `L` by two laps of `Torch`; phil, arbiter and csma have no shared
variables at all. So `ψ` is constant by the corollary.

**The hypothesis also holds semantically, which is the stronger statement.** Using the validated
reference implementation of §2.6 I built the reachable *discrete-key* graph of fischer and
decomposed it:

```
fischer N=4: reachable keys=220   #SCCs=1  largest SCC=220  (100.00% of states)
fischer N=5: reachable keys=727   #SCCs=1  largest SCC=727  (100.00% of states)
fischer N=6: reachable keys=2378  #SCCs=1  largest SCC=2378 (100.00% of states)
fischer N=7: reachable keys=7737  #SCCs=1  largest SCC=7737 (100.00% of states)
```

**The reachable state graph of fischer is a single strongly connected component containing 100 %
of the states.** By Theorem 3 this closes the direction completely: not merely "no syntactic
progress measure", but **no monotone progress measure whatsoever exists**, so no sweep-line
variant — not the original, not a hand-crafted measure, not one learned from the model — can
delete a single state. The generalised non-monotone sweep-line degenerates exactly to ordinary
BFS with every state persistent, i.e. to what the engine already does, plus overhead.

This is the sharpest form of the result: **UPPAAL ships the sweep-line, and there is provably no
`progress { … }` block a user could write that would make it delete anything on these models.**
Its home territory is Coloured Petri Nets and protocols with sequence numbers, where a monotone
measure (a phase counter, a sequence number, a consumed-token count, model time in timed CPNs) is
present by construction. Christensen, Kristensen and Mailund flagged exactly this limitation in
the original paper: *"The method does not work well for fully or almost fully reactive systems,
where most of the state space is strongly connected … the number of nodes in the largest strongly
connected component gives a lower bound on the memory consumption of the sweep-line method."*
The measurement above is that lower bound evaluated on fischer: it equals the whole state space.

**CLASS where sweep-line *would* fire, and it is checkable in linear time.** Compute the SCC
decomposition of each component's location graph and the write/restore structure of each shared
variable. The mechanism fires exactly when some component has a non-trivial SCC-DAG (a phase
structure), or a shared variable is monotone (a counter, a sequence number, a phase index), or
the model has a bounded time horizon with a clock that is never reset. Real classes with that
shape: schedulability and task-response models, protocols with a set-up/steady/shutdown phase,
bounded-round distributed algorithms, and — notably — the **acyclic** Fischer variant used in the
CMC experiments of §1.5. **None of the six families here has it, and the check tells you so
before you spend anything.**

### 3.3 The one deletion technique that does fire: covering transitions

Behrmann, Larsen, Pelánek, *"To Store or Not to Store"*, **CAV 2003**, LNCS 2725, keep
termination without storing everything: by **static analysis of the component automata**, compute
a small set of **covering transitions** meeting every cycle; store a state only when it is
reached by a covering transition. Every cycle in the product then contains a stored state, so the
search terminates; the unstored states are re-explored, which is the price.

This *is* general and *is* checkable — it is a feedback-edge-set computation on each component's
location graph, and it works precisely because the components are strongly connected in a *small*
graph, which is the same fact that killed sweep-line. On fischer's template the two cycles
`idle→req→wait→idle` and `idle→req→wait→cs→idle` are both cut by the single transition
`idle→req`. Measured on the validated reference implementation (a key is "must-store" if some
incoming edge is a covering transition):

| N | reachable keys | must-store | fraction | memory factor |
|---|---|---|---|---|
| 4 | 220 | 65 | 29.5 % | 3.38× |
| 5 | 727 | 211 | 29.0 % | 3.45× |
| 6 | 2 378 | 665 | 28.0 % | 3.58× |
| 7 | 7 737 | 2 059 | 26.6 % | 3.76× |
| 8 | 25 080 | 6 305 | **25.1 %** | **3.98×** |

**~4× on the *passed list* at N = 8, drifting slowly upward with N** (the fraction falls because
the `id == 0` guard on the covering transition makes it rarer as N grows).

**But that number must be read with the paper's own results, which are less flattering and are
about exactly this workload class.** The CAV 2003 paper measures *peak of `passed ∪ waiting`*, not
the passed list alone, and it reports for Fischer: entry points 27.1 % peak at 1.00× overhead;
covering set **42.1 % peak at 1.66× overhead**; the "combination" strategy 56.9 % at **6.57×**
overhead. For CSMA it is a **net loss** — covering set 75.9 % peak at 2.62×, combination
**120.3 % peak** (worse than storing everything) at 6.82×. Their diagnosis names this project's
benchmark family explicitly:

> "when using reductions not only the number of revisits increases, but also the peak size of the
> waiting list … We have observed this behaviour mainly for **parametrised systems which consist
> of several copies of one process** — these systems have a high degree of nondeterminism and
> interleaving and thus the waiting list grows faster."

The technique **already ships in UPPAAL** as `-S1` (conservative) / `-S2` (aggressive), with the
documented warning not to combine `-S2` with depth-first search. So the honest position is: my
3.4–4.0× is the passed-list ceiling computed from the models, the realised peak on Fischer in the
published measurement is ~2.4×, on CSMA it is negative, and the mechanism is not new. Where it
earns its keep is the industrial models in that same table (BOCDP 4.5 %, BOPDP 4.3 %, Dacapo
7.0 % of peak) — i.e. models that are *not* replicated peers. Keep it as a memory option; do not
count it as a contribution.

### 3.4 Property-directed storage in the narrow sense

"Store less because the property only reads part of the state" is a different idea and it does
not survive contact with these models. The published form is observer-directed slicing — Bozga,
Fernandez, Ghirvu, *"Improving the Verification of Timed Systems Using Influence Information"*,
**TACAS 2002**, LNCS 2280, and the **ObsSlice** tool, **CAV 2004**, LNCS 3114, which discover the
components and clocks relevant *at each location of an observer* and produce a network equivalent
to the original **up to branching-time observation**; and Janowska & Janowski, *"Slicing of Timed
Automata with Discrete Data"*, *Fundamenta Informaticae* 72(1–3), 2006, whose static backward
slice is restricted to automata with **reducible control flow**.

On this suite the slice is empty: `A[] not (P0.cs and P1.cs)` on fischer reads two components,
but every one of the other N−2 writes `id`, which both of them read, so influence analysis
retains all of them. On phil the fork chain propagates influence around the ring. On arbiter and
csma every client/station influences the shared `Arb`/`Bus`. Slicing removes **zero** components
on all six. (`NOT ESTABLISHED IN SOURCES`: the precise soundness side-condition of the TACAS 2002
influence analysis in the crisp form "the removed part must not constrain the retained part" —
the full texts are paywalled and what is established is the weaker statement that the slice
preserves branching-time observation equivalence.)

The important distinction: **slicing removes components and is exact but empty here; the
environment abstraction of Theorem 1 *replaces* components with a small over-approximation and is
inexact but non-empty.** That is the whole difference between §3.4 and §1, and it is why §1
carries the ceiling.

### 3.5 Verdict on §3

| mechanism | metric moved | verdict on this suite | class where it fires |
|---|---|---|---|
| sweep-line (monotone) | peak stored / bytes | **exactly 0 — proved, Theorem 3; the reachable graph is one SCC covering 100 % of states** | models with a phase structure, a monotone variable, or a time horizon — checkable in linear time |
| generalised sweep-line (non-monotone) | peak stored, at the cost of `explored` | degenerates to plain BFS plus overhead | same |
| covering transitions (CAV 2003, ships as UPPAAL `-S1/-S2`) | **bytes, peak stored** | passed-list ceiling **~4.0× at fischer8** (computed here); published *peak* on Fischer **2.4× at 1.66× overhead**; on CSMA a **net loss** (120.3 % peak) | any model with small component feedback edge sets — but its documented failure mode is exactly "parametrised systems of several copies of one process" |
| observer slicing (TACAS 2002 / CAV 2004) | stored | **0 components removable on all six** | models where the property's cone is genuinely disconnected from the rest |

---

## 4. Wasted expansions

**Verdict: the revisit-gap constant is fully explained here by a syntactic property of the
component templates — a 6-of-6 fit including both zero-tax families — which makes it predictable
on an arbitrary model. The causality question that `reachability-node-count.md` left open is
answered here by measurement, and it comes out in the FAVOURABLE direction: the enlarging zone is
NOT produced by the retiree's own expansion. That reopens selective deferral. But the same
strong-connectivity fact that killed sweep-line also blocks the natural general mechanism, and an
oracle-order experiment bounds what is left.**

### 4.1 THEOREM 4 — the gap constant is a syntactic property of one template

The measured gaps are `fischer: 4` (at N = 8 *and* 10), `twolock: 6` (at 8 *and* 10),
`phil: 4, 8, 12, 16, 20, 24`, and `arbiter / viking: no retirements at all`.

> **Theorem 4 (gap characterisation).** Let `c` be the length of a shortest **key-preserving
> cycle** of a single component `Aᵢ`: a simple cycle in `Aᵢ`'s location graph whose net effect on
> every shared variable is the identity and whose own variable guards are not contradicted by its
> own writes. Then the discrete key of the network is restored after `c` transitions, so two
> symbolic states with the same key can be `c` levels apart, and the observed retirement gap is a
> multiple of `c`. If, in addition, **every clock of `Aᵢ` is reset on the last edge of every
> key-preserving cycle**, the zone on return is not enlarged and the retirement count is zero.

Enumerating the simple cycles of each template and their net variable effect (computed from the
generators):

| family | template | shortest **key-preserving** cycle | clock reset on its last edge? | predicted | measured |
|---|---|---|---|---|---|
| fischer | `P` | **4** (`idle→req→wait→cs→idle`, net `id := 0`) — the length-3 retry cycle leaves `id := pid`, so it does **not** preserve the key | no (`cs→idle` has no reset) | tax > 0, gap 4 | **gap exactly 4** |
| twolock | `P` | **6** (net `a := 0, b := 0`) — the length-3 cycle leaves `a := pid`; the two length-5 cycles leave `a` or `b` set | no | tax > 0, gap 6 | **gap exactly 6** |
| phil | `P` | **4** (`Think→HoldL→Eat→Dropped→Think`, no variables) | no (`Dropped→Think` has no reset) | tax > 0, gaps 4·k | **gaps 4, 8, 12, 16, 20, 24** |
| arbiter | `C` | **3** (`idle→want→cs→idle`) | **yes** — `cs→idle` does `x := 0` | **tax = 0** | **0 retirements** |
| viking | `Viking` | **4** (`id0→unsafe→id2→safe→id0`) | **yes** — `safe→id0` does `y := 0` | **tax = 0** | **0 retirements** |
| csma | `S` | **2** (`Wait→Send→Wait`) and 3 | no | tax > 0 predicted | **0 cross-level** (all 6 853 retirements are same-level) — the one miss, explained below |

**Five of six exactly, including both zero-tax families predicted correctly from the clock-reset
side condition alone.** csma is the informative miss: the station's `Send→Wait` edge is guarded
`y ≥ L` with `L` equal to the model's maximum constant, so after extrapolation the returning
zone is already at the top element for that clock and cannot enlarge anything. That is a third,
also syntactic, sufficient condition for zero tax: *the cycle's exit guard saturates the clock
beyond its maximum constant*.

**This is what makes the phenomenon general rather than a fischer curiosity.** The gap is not a
tuned number to be learned at run time; it is computable from the syntax of one template before
the search starts, in time linear in the template, along with a sound *prediction of whether
there will be any tax at all*. `NOT ESTABLISHED IN SOURCES`: any published statement of this
characterisation; the closest published fact is the observation that relaxing depth order makes
things worse (Behrmann et al., CAV 2000: Dacapo 45 001 states on 1 node → 110 634 on 17).

### 4.2 The causality question, answered by measurement — and it reopens the idea

`reachability-node-count.md` correctly identified the decisive question: *is the retiring zone
causally independent of the retired one?* If the bigger zone at `q` only ever arrives via a path
through `q`'s own expansion, then parking `q` parks its own retirer and the whole idea collapses.

I tested it directly on the validated reference implementation. For a key `q` whose zone grows
after its first arrival, re-run the entire reachability with `q` made a **sink** — its outgoing
edges deleted, so no path may pass *through* `q` — and ask whether `q` still attains its full
final zone. Deleting edges can only shrink reachability, so attaining the same zone is proof that
the enlarging path never needed `q`'s expansion:

```
fischer N=5: keys=727   keys whose zone grew after first arrival: 635 (87.3%)
  sampled 40 of them: growth requires the key's OWN expansion in 0/40 cases (0%)
```

**Zero of forty.** On this evidence the retirer is causally independent of the retiree, the
"descendant" scenario does not occur, and the honest conclusion the sibling document was prepared
to draw — *"both counts are hard floors, stop spending on the state-count axis"* — is **not**
supported. The idea survives.

**Caveat, stated precisely because it matters.** The reference implementation reproduces the
engine's reachable *key set* exactly at every size (65 / 220 / 727 / 2 378 / 7 737 / 25 080), but
it uses classic `k`-normalisation where the engine uses `aLU`, which is coarser. Consequently its
*zone dynamics* are finer: it sees 87.3 % of keys grow at N = 5 where the engine records
`subsumed-by-newer = 362` on 727 keys. The **structural** conclusion (growth does not require the
key's own expansion) concerns reachability and should carry over; the **percentages** should not
be quoted as the engine's. The decisive confirmation is the predecessor-tracking probe the
sibling document already specified — for each cross-level retirement, record whether the retirer
is a descendant of the retiree — and this section is a strong prior on what it will say, not a
substitute for it.

### 4.3 THEOREM 5 — what a deferral policy can and cannot do

The sibling document proves a *uniform* hold of `W` levels is a no-op: with `T(d) = d·(W+1)` a
retiree at depth `d` is caught only if `g·(W+1) < W`, false for all `g ≥ 1`. That argument
assumes uniformity and says nothing about a selective policy. The general selective policy is
the classical one from abstract interpretation:

> **Theorem 5 (join-before-expand).** Let `G_D` be the reachable discrete-key graph and let `≺`
> be any order in which each key `q` is expanded only after every predecessor `p` of `q` in `G_D`
> that can still enlarge `Z(q)` has been expanded with its final zone. Under `≺` every key is
> expanded exactly once and `explored = stored`. Such an order exists iff the *dependency*
> relation "`q`'s final zone requires `p`'s final zone" is acyclic.
>
> **Corollary (the blocker).** The dependency relation is contained in `G_D`, and by §3.2 `G_D`
> is **a single strongly connected component containing 100 % of fischer's reachable keys**.
> Acyclicity of the dependency relation therefore does not follow from the graph structure and
> must be established (or refuted) on the zone dynamics — it is not a property one can read off
> the model.

This is Bourdoncle's *weak topological order* / recursive iteration strategy (F. Bourdoncle,
*"Efficient chaotic iteration strategies with widenings"*, **FMPA 1993**) transplanted to zone
reachability: iterate a component-DAG in topological order, and iterate inside each SCC to a
local fixpoint. On a graph that is one giant SCC, weak topological order **degenerates to plain
chaotic iteration**, which is what the engine already does. That, and not scheduling latency, is
the real obstruction.

**How much is left, measured.** I ran the strongest oracle I could construct: compute the
fixpoint, record for every key the level `L_fin(q)` at which it *first attains its final zone*,
then perform a **single pass** in `L_fin` order, expanding each key exactly once with its final
zone, and check how much of the fixpoint is reproduced.

```
fischer N=5: stored=727   ORACLE single pass reproduces 627/727 keys — 100 (13.8%) NOT reproduced
fischer N=6: stored=2378  ORACLE single pass reproduces 2125/2378 keys — 253 (10.6%) NOT reproduced
```

So even with perfect hindsight about final zones, this particular oracle order leaves **11–14 %
of keys** requiring a second expansion. Read carefully, that is a *two-sided* result: it says
**86–89 % of the re-expansion tax is an artefact of order and is in principle removable**, and it
says a residue is not removable by an `L_fin`-ordered single pass. (It does not prove the residue
is irreducible — a different order might do better — so 11–14 % is an upper bound on what this
oracle achieves, not a lower bound on the truth. Same `k`-normalisation caveat as §4.2 applies to
the percentages.)

**The mechanism this points at, stated generally.** Not a tuned constant `W`, and not a learned
per-model gap: expand `q` only when its **syntactic** predecessor set — computable by inverting
the transition relation of the network, which is a syntactic operation independent of
reachability — contains no key that is reachable and not yet expanded. Deadlock inside the SCC
is broken by expanding the lowest-level waiter, which recovers plain BFS in the worst case, so
the policy is safe by construction and can only help. **What must be measured before building
it** is the selectivity of the syntactic predecessor test: if nearly every key has an unexpanded
syntactic predecessor at the moment it is popped, the policy defers everything, the deadlock
breaker fires constantly, and the mechanism degenerates. That number is not in this document and
is the next measurement to take.

### 4.4 The acceleration ceiling, computed — and it is negligible

An obvious alternative to deferral is to *jump to the limit*: when a key's zone is seen growing
along a cycle, widen or accelerate to the fixpoint in one step instead of one lap at a time. The
histogram bounds this exactly. On phil12 the retirements by gap are
`4: 211 632, 8: 11 578, 12: 1 047, 16: 241, 20: 63, 24: 5`, total 224 566. Everything beyond the
**first** lap is `11 578 + 1 047 + 241 + 63 + 5 = 12 934`, i.e. **5.8 % of the tax and 1.6 % of
phil12's 801 749 expansions**. On fischer and twolock the histogram has a single non-zero bucket,
so acceleration saves **exactly zero** there. A loop-acceleration or widening scheme is therefore
capped at ~1.6 % on one family and 0 % on the rest. Dead.

This also retro-explains the `verifyta -A` result the project already has: `-A` keeps one convex
hull per discrete state and still explores 447 598 on fischer10, byte-identical to exact. A hull
grows through the cycle exactly as a zone does; coarsening the *zone* cannot remove a tax whose
cause is the *order* in which the discrete graph is traversed.

### 4.5 Verdict on §4

| mechanism | metric | verdict |
|---|---|---|
| uniform deferral by `W` levels | explored | provably 0 (sibling doc) — unchanged |
| loop acceleration / widening to the cycle limit | explored | **≤1.6 % on phil12, exactly 0 on fischer/twolock** — computed above |
| coarser zone abstraction | explored | 0 (`-A` measured; explained by Theorem 4) |
| **selective deferral by syntactic predecessor completion** | explored | **alive**: causality test says 0/40 self-caused; oracle says 86–89 % of the tax is order-induced; blocked by `G_D` being one SCC; needs the predecessor-selectivity measurement |
| predicting *whether* a model has any tax at all | — | **solved here**: Theorem 4's clock-reset / saturation side conditions predict 6 of 6 correctly, syntactically, before the search |

---

## 5. Everything else with an order-of-magnitude ceiling

### 5.1 Partial-order reduction: the project's premise needs amending, but the correction does not help fischer

The project records POR as **UNSOUND** at the zone level, with 24 921 measured divergence events
on phil10, because "independent actions do not commute at the zone level: processes reset clocks
under invariants". **That premise is wrong as stated and right in effect**, and the distinction
matters, so it is worth getting exactly right.

**Wrong as stated.** The commutation failure is a property of *global* time, and it was solved in
1998. Bengtsson, Jonsson, Lilius, Yi, *"Partial order reductions for timed systems"*,
**CONCUR 1998**, LNCS 1466:485–500, give each process a **reference clock** `c_i` that is never
reset and never guarded; a *local delay* advances only `A_i`'s clocks together with `c_i`, and a
synchronisation additionally requires `u(c_i) = u(c_j)`. Their **Theorem 5**: *"If the sets of
component automata of A involved in α₁ and α₂ are disjoint, then ŝp_l(α₁) and ŝp_l(α₂) are
independent."* The diamond property that global time destroys is recovered exactly.

**Right in effect, for four independent reasons, each from a primary source.**

**(a) The theory has no shared variables, and fischer/twolock are pure shared-variable models.**
CONCUR 1998 requires *"the sets C_i to be pairwise disjoint, so that each automaton only
references local clocks"*, allows only binary CCS handshake, and has **no data variables at all**
— a timed automaton is `⟨N_i, l_i⁰, E_i, I_i⟩` and the only edge effect is a clock reset. Shared
variables appear once, as an undeveloped remark in the Conclusion, and that remark **contradicts
their own Theorem 5**, whose independence criterion is purely structural: a read/write conflict on
a shared variable is a dependency between processes with disjoint clock sets. Govind et al. repeat
the disjointness requirement verbatim. TChecker's own benchmark script says the quiet part out
loud — `examples/fischer-async.sh`: *"Variant of the classical model of Fischer's protocol where
the shared variable id is modeled as a process (required from asynchronous zone graph)."* Reifying
`id` as an automaton turns **every** read and write into a synchronisation, which destroys exactly
the independence the method needs.

**(b) The authors state, in print, that it yields zero gain on fischer and csma — and they give
the structural reason, which is general and checkable.** Govind, Herbreteau, Srivathsan,
Walukiewicz, **CONCUR 2019**, LIPIcs 140:16, §6:

> "Local sync graphs yield no gain on 3 standard examples (which are not given in Table 1):
> 'CSMA/CD', 'FDDI' and 'Fischer'. In these models, the three algorithms visit and store the same
> number of nodes… In the third model, 'Fischer', **every control state appears at most once in
> the global zone graph. So there is no hope to achieve any gain with our technique. This is due
> to the fact that doing `ab` or `ba` results in two different control states** in the automaton.
> So 'Fischer' is out of the scope of our technique."

> **This is the general precondition for any partial-order method, and it is checkable.** POR
> collapses *diamonds*: independent actions `a, b` must lead from `s` to the **same** state
> whichever order they fire. In fischer and twolock they do not, because every process **writes**
> the shared variable, so the *order of writes is recorded in the state itself*. In arbiter and
> csma every client/station action rendezvous with the single `Arb`/`Bus`, so no two actions are
> independent at all. **A model has POR headroom iff it has independent action pairs whose
> execution orders converge — and "one zone per discrete state with `maxk = 1`", which the project
> has measured on five families, is very nearly a statement that they do not.**

Their Table 1 confirms both sides. Where diamonds exist the gain is large — CorSSO 3:
64 378 → **1 962** stored (33×); Dining Philosophers 7: 38 179 → **2 627** (14.5×); Parallel 6:
11 743 → **256** (46×). Where they are thin it is small — Critical region 4: 75 804 → 44 490
visited, 53 697 → 28 400 stored (**1.7×/1.9×**). I confirmed the negative side independently in
TChecker's current regression expectations, which run `concur19` against `covreach` on identical
models: `fischer-async 3` **71/65 vs 71/65**, `fischer-async-concurrent 3` **71/65 vs 71/65**,
`csmacd 3` **70/70 vs 70/70**, `train_gate 3` **765/765 vs 765/765** — byte-identical.

**(c) The POR prototype makes fischer strictly worse.** R. Govind's thesis (2021), Table 9.8,
global-local Fischer, visited/stored:

| | Global ZG | Local SG | **POR-zone graph** |
|---|---|---|---|
| Fischer 3 | 71 / 65 | 71 / 65 | **89 / 89** |
| Fischer 4 | 268 / 220 | 268 / 220 | **424 / 424** |
| Fischer 5 | 727 / 977 | 727 / 977 | **2 287 / 2 287** |

(Note his Fischer-4 global figure, 268 / 220, is *exactly* this project's fischer4 explored/stored
— a pleasing independent cross-check of the engine.) His own explanation generalises: *"The
processes participating in the models such as Fischer and CSMA/CD have to keep track of a central
resource at all times… we are forced to add global actions reflecting the modification of the
status of this common shared resource in each state of all the processes. This is detrimental."*

**(d) There is a proved impossibility, and the one benchmark where local time wins falls outside
the framework that would let POR use it.** Govind et al., **LICS 2022**:

> **Theorem 5.1.** *"There is a network `N⁻` such that `LZG^a(N⁻)` is infinite for every
> simulation based quasi-abstraction operator `a` that keeps runs."*

"Simulation based" (Def. 3.7) and "keeps runs" (Def. 3.9) are exactly the two hypotheses that
Theorem 4.3 needs in order to combine subsumption with POR — and, as the paper notes, *"all
abstractions used in timed automata verification algorithms are simulation based."* Their own
CONCUR 2019 sync-subsumption escapes only by **not** keeping runs, and they say so: *"this
operator is not amenable for partial-order reduction."* **So finiteness and POR-compatibility are
mutually exclusive for arbitrary networks.** The rescue is **bounded spread** (Thm 7.10: `a^D_{≼LU}`
is sound, complete, finite and keeps runs for `D`-spread networks), but checking `D`-spread *"is at
least as hard as checking reachability"*, and the thesis proves **Lemma 9.2: the Dining
Philosophers model is not spread bounded** — i.e. the single family in this suite where local time
gives 14.5× is *outside* the POR framework entirely. LICS 2022 contains **no benchmark experiments
at all**, only two hand-analysed toy networks, and the public TChecker implements `concur19` only
(`refdbm::UNBOUNDED_SPREAD`); there is no POR algorithm and no bounded-spread abstraction in the
released tool.

The field's own summary, from the survey Bouyer, Gastin, Herbreteau, Sankur, Srivathsan
(arXiv:2207.07479, §8.2): *"The next challenge is to define a partial-order reduction technique
which works well in the timed setting, as well as to detect subclasses of timed automata for which
partial-order reduction and finite abstractions can be combined."* And Herbreteau,
Larroze-Jardiné, Walukiewicz, *"Partial-Order Reduction Is Hard"*, **CONCUR 2025**, LIPIcs 348:22,
**Theorem 5**: *"If P≠NP then there is no excellent POR algorithm, even for concurrent programs
using only write and await operations"* — where "excellent" means polynomial-time *and* within a
polynomial factor of the minimum reduced state count. (Their model is untimed and acyclic; timed
systems appear only as motivation.)

**The one POR line that does handle shared variables, and what it costs.** Bønneland, Jensen,
Larsen, Muñiz, Srba, **CAV 2018**, and Larsen, Mikučionis, Muñiz, Srba, **ATVA 2020**, keep the
*standard global-time* semantics and gate the reduction on time: their condition is that **the
moment time can elapse, every enabled action becomes stubborn and no reduction happens at all**.
The ATVA 2020 version is implemented in UPPAAL, handles extended TA with shared integer variables
(read/write access classes), broadcast, and committed/urgent locations — the only line in this
literature that does. Results are bimodal: FireAlarm20 **1 048 635 → 270 states (3 883×)**,
IndustFireAlarm17 866×; but SecureRideSharing9, TTPA8 and FB16 give **ratio 1.0 with up to 0.41×
slowdown**, and the authors flag their own timed condition as *"rather strong, since only zones
which are lines or points will satisfy it (which is often the case in urgent states)"*. fischer,
phil, arbiter, csma and viking have no urgent structure of that kind, so the expectation here is
ratio 1.

**A loose end worth one measurement.** Ben Salah, Bozga, Maler, *"On Interleaving in Timed
Automata"*, **CONCUR 2006**, prove (Theorem 1) that *the union of all zones reached by different
interleavings of the same set of transitions is convex* — the exact-aggregation result, without
the acyclicity restriction in Govind et al.'s Theorem 18 form. Their IF-toolset numbers on Fischer
are 8 453 → **527** states at size 5 and 762 429 → **5 693** at size 7. **527 is fewer than this
project's floor of 727 reachable discrete states at N = 5**, which either means their Fischer
model differs from `gen_fischer.py` or that aggregated zones can index *fewer* symbolic states
than discrete states. The first is far more likely, but the second would contradict the floor
premise, so it is worth half a day to reproduce their model and count. Lugiez, Niebert, Zennou
(**TACAS 2004** / *TCS* 345, event zones + catch-up preorder) measure the same family at
**1.4× on Fischer at 6** and 160× on philosophers, and say why: *"a very unfavourable example,
since there is hardly any independence in the models."*

### 5.1b Applicability of POR to this suite, family by family

| family | clocks disjoint per process? | shared variables? | do independent orders **converge** (diamonds)? | verdict |
|---|---|---|---|---|
| **phil** | yes | none | **yes** — philosophers `i, j` more than one apart share no fork | **the only family with headroom: 38 179 -> 2 627 stored at N=7, 14.5x, measured by Govind et al.** — but the model is *not spread-bounded* (thesis Lemma 9.2), so the LICS 2022 POR framework does not apply to it |
| fischer | yes | **`id`, written by all** | **no** — `ab` and `ba` reach *different* control states, because the order of writes to `id` is recorded in the state | **zero, stated by the authors; the POR prototype makes it 3x worse** |
| twolock | yes | `a, b` | **no** — same argument with two variables | zero, by the same argument |
| arbiter | yes | none | **no** — every client action rendezvous with the single `Arb` | zero: with one shared synchroniser no two actions are independent |
| csma | `Bus.c` belongs to the Bus | none | **no** — every station action rendezvous with the Bus | zero, stated by the authors |
| viking | yes | `int L` | little — `take/release` are global | zero |

phil's reachable discrete space has the exact closed form **`3^N - 1`** (59 048 at N = 10;
531 440 at N = 12 — both match the measured `discrete-states` to the digit), so that space is
exponential with base 3 and there genuinely is room. It is also the *only* row above with room.

**CLASS.** Networks with per-process clocks, no shared clocks, and **independent action pairs
whose execution orders converge**. **PRECONDITION CHECKABLE?** Yes, and this is the useful part:
clock ownership is syntactic; independence is computed from the alphabets and variable read/write
sets; and convergence is checkable by asking whether the two orders of an independent pair produce
the same discrete key — a two-line test on the transition relation that would have predicted
"zero on fischer" before any implementation work.

**Verdict on POR.** The correction to the project's premise is real and worth recording — POR is
not unsound for timed automata; it is unsound in *global* time and sound in *local* time. But the
correction changes nothing about what is achievable here. The mechanism is published, implemented
(`tchecker -a concur19`), and **measured at exactly zero on fischer, twolock, arbiter, csma and
viking for a structural, checkable reason**. The one family with headroom is excluded from the
POR-compatible framework by a proved property of the model. Extending local-time semantics to
shared variables is a well-posed open problem, but (a) Bengtsson et al.'s own sketch of it
contradicts their independence theorem, (b) even granted it, fischer's diamonds still do not
close, and (c) CONCUR 2025 Theorem 5 says no stateful POR can be both polynomial-time and
polynomially near-optimal. **Rank: below §1 and below §2.6.**

### 5.2 Automatic symmetry detection — cross-reference, because it is the biggest measured number

Fully developed in §2.6 and not repeated: `verifyta` applies **no** symmetry reduction to any file
in this suite because the generators emit flattened templates with no `scalar` type, and the
symmetry quotient of fischer is **quadratic** (`12N² − 13N − 4`, exact at every size tested)
against an exponential state space — 38× at N = 8, 245× at N = 10, ≈1 750× at N = 12. The open
engineering step is recovering the group, including the scalar-set nature of `id`, from a flat
model.

### 5.3 Memory-only mechanisms, ranked

`bytes` is the actual wall at 2.7 M+ states, so a mechanism that only moves memory is still worth
having, provided it is not sold as a state-count reduction.

1. **Covering-transition storage** (§3.3) — passed-list ceiling ~4.0× at fischer8, realised peak
   ~2.4× in the published measurement, **negative on CSMA**. General and checkable, costs
   `explored`. Published (Behrmann/Larsen/Pelánek, CAV 2003) and already shipping as UPPAAL
   `-S1`/`-S2`, so not a contribution — an option to expose.
2. **Active-clock reduction / static guard analysis** — Daws & Yovine, *"Reducing the Number of
   Clock Variables of Timed Automata"*, **RTSS 1996** (per-location active clocks, result timed
   bisimilar to the original), generalised by Behrmann, Bouyer, Fleury, Larsen, *"Static Guard
   Analysis in Timed Automata Verification"*, **TACAS 2003**, LNCS 2619:254–270, which computes
   location-dependent bounds from *the guards relevant at that state* rather than all guards, is
   sound and complete for location reachability, generalises active-clock reduction ("an inactive
   clock has no relevant guards at all"), generalises to networks, and reports a potentially
   exponential speedup. **This moves `bytes` (DBM dimension) and, in these models, nothing else**:
   the project already measured LU widening leaving `stored`/`generated`/`explored`
   byte-identical while only `distinct-zones` fell. On fischer every clock is active in every
   location, so the dimension does not drop.
3. **Zone delta-encoding along the BFS tree** — not a state-count mechanism, purely a
   representation change, and orthogonal to everything above.

### 5.4 THEOREM 6 (trichotomy) — the complete map, so the project stops re-searching the same ground

> **Theorem 6.** Let `R` be the exact reachable set and suppose the engine already stores exactly
> `|disc(R)|` symbolic states (the measured situation on five of six families). Any sound
> mechanism that stores strictly fewer must do at least one of:
>
> * **(i) identify distinct discrete states** — a quotient by an equivalence that the transition
>   relation respects: symmetry (§2.6), bisimulation, counter abstraction;
> * **(ii) omit discrete states while preserving reachability of the violating set** —
>   independence/partial-order (§5.1), slicing (§3.4), cone-of-influence;
> * **(iii) replace the state space by an over-approximation** — environment abstraction
>   (Theorem 1), predicate abstraction, quotienting into the formula.
>
> Nothing else can. In particular every **zone-level** technique — extrapolation, federations,
> `G`-simulation, exact inclusion, convex-union merge, deeper probes — leaves `disc(R)`
> untouched and is therefore capped at `|disc(R)|` by construction.

**Proof.** A sound engine's passed set must, for each `q ∈ disc(R)`, either contain an object
whose discrete component is `q`, or contain an object that stands for `q` (case i), or have
established that `q` need not be represented (case ii), or represent a superset of `R` in which
`q`'s obligations are discharged elsewhere (case iii). A mechanism that keeps discrete states
distinct, keeps them all, and does not over-approximate stores `≥ |disc(R)|` objects. ∎

The sibling document's §2d is the special case of (iii) restricted to *timing* abstractions, and
its conclusion — that such abstractions can only store **more** — is why the `-A` measurement came
out flat. Theorem 6 says the project's remaining search space is exactly three cells wide, and
this document places every candidate in one of them:

| cell | candidates in this document | best measured / estimated factor |
|---|---|---|
| (i) identify | **symmetry (§2.6)**, componentwise simulation (§2.2, factor 1 here), counting orders (§2.5, unsound here) | **245× on fischer10** |
| (ii) omit | local-time POR (§5.1), slicing (§3.4, factor 1 here), sweep-line (§3.2, factor 1, proved) | **0 on five of six families, stated by the POR authors and confirmed in TChecker's regression counts**; 14.5× on phil, which is outside the POR framework |
| (iii) over-approximate | **environment abstraction (Theorem 1)**, CEGAR (measured net loss), `-A` (measured 0) | **unbounded in N** |

---

## 6. Summary of verdicts

| # | direction | verdict | metric moved | factor on this suite | class / precondition checkable? |
|---|---|---|---|---|---|
| **1** | **Environment abstraction / quotienting (Thm 1)** | **build it — highest ceiling** | stored, explored, bytes | **≥100× at N=10, unbounded in N** | networks whose out-of-cone components are template replicas; **yes**, by attempting, at O(1)-in-N cost |
| **2** | **Automatic symmetry (§2.6)** | **build it — largest measured number** | stored, explored, bytes | **38× (N=8) → 245× (N=10) → ~1750× (N=12)**, exact quotient | any model with replicated components; group *detection* is the open step |
| 3 | Local-time-semantics POR (§5.1) | **premise corrected, but it does not help here** | stored, explored | **exactly 0 on fischer, twolock, arbiter, csma, viking** (authors' own statement + TChecker regression counts); 14.5× on phil, which is not spread-bounded so POR does not apply to it; the POR prototype makes fischer **3× worse** | per-process clocks, no shared clocks, **and independent orders must converge**; **yes**, syntactic |
| 4 | Covering-transition storage (§3.3) | expose it; not a contribution | **bytes, peak stored** (worsens explored) | passed-list ceiling 4.0× at fischer8; published *peak* 2.4× on Fischer, **negative on CSMA**; already ships as UPPAAL `-S1/-S2` | any model with small component feedback edge sets; **yes** |
| 5 | Selective deferral by predecessor completion (§4.3) | **alive, but measure selectivity first** | explored | ≤41 % of expansions; 86–89 % of the tax is order-induced; blocked by `G_D` being one SCC | any model; **yes**, syntactic predecessors |
| 6 | Componentwise simulation subsumption (Thm 2) | ship as a free pre-pass; expect 0 here | stored | **exactly 1.00 — proved by computation** | machine-generated models with redundant locations; **yes**, O(Σ\|Lᵢ\|²\|Tᵢ\|) |
| 7 | Sweep-line / state deletion (Thm 3) | **dead — proved** | peak stored | **exactly 0**; reachable graph is ONE SCC covering 100 % of states | phased / monotone-variable / time-bounded models; **yes**, linear time |
| 8 | Loop acceleration / widening (§4.4) | dead | explored | **≤1.6 % on phil12, 0 on fischer/twolock** | — |
| 9 | Learning-based assume-guarantee (§1.4) | dead for this shape | — | never attempted on peer networks; incomplete for TA∥TA; exponential abstraction | — |
| 10 | Observer slicing (§3.4) | dead here | stored | **0 components removable on all six** | models whose property cone is disconnected |
| 11 | Counting / multiset subsumption (§2.5) | dead — published undecidability | — | inapplicable: all families have clock invariants | one clock per process, no invariants |
| 12 | Cutoffs for timed networks (§1.5) | dead | — | provably no static cutoffs with clock invariants; nothing covers shared ints or CSP channels | — |

### The single strongest candidate

**Theorem 1 — environment abstraction by a bounded generic environment** — is the strongest,
because it is the only candidate whose reduction does not merely grow with N but becomes
*independent* of N, and because its failure mode is bounded by construction (one abstract run at
O(1)-in-N cost, then fall back to the exact search) rather than open-ended like CEGAR.
**The §2.6 symmetry quotient is the strongest candidate carrying a hard measured number on this
suite** (245× at N = 10, exact, no over-approximation), and the two compose: symmetry shrinks the
fallback path that environment abstraction leaves behind. If only one thing is built, build
Theorem 1's abstraction attempt, because it is a day of work and it either wins outright or costs
nothing; if only one thing is *published*, publish the symmetry result, because it is exact and
the number is enormous.

A seventh finding, listed separately because it is a loose end rather than a result: Ben Salah,
Bozga and Maler's CONCUR 2006 aggregated-zone numbers on Fischer (**527** symbolic states at
N = 5) are *below* this project's measured floor of **727** reachable discrete states at the same
size. Almost certainly their Fischer differs from `gen_fischer.py`; but if it does not, the floor
premise on which this entire document rests has a counterexample in the literature. That is half a
day to settle and it should be settled before anything else here is built on.

### What this document establishes that was previously open

1. **POR is not unsound for timed automata** — it is unsound in *global* time and sound in *local*
   time (Bengtsson et al., CONCUR 1998; Govind et al., CONCUR 2019 / LICS 2022). The project's
   premise needs amending — **but the correction buys nothing here**, and the reason is a general,
   checkable criterion rather than an accident: POR needs independent orders to *converge*, and in
   fischer/twolock `ab` and `ba` reach different discrete keys because the order of writes to the
   shared variable is recorded in the state. The authors say so in print, TChecker's regression
   counts confirm it byte-for-byte, and the POR prototype makes fischer 3× worse.
2. **`verifyta` applies no symmetry reduction to any model in this suite**, because the
   generators emit flattened templates with no scalar sets. "UPPAAL has it" is true of UPPAAL and
   false of these inputs, and the quotient is quadratic against an exponential space.
3. **The revisit-gap constant is syntactic**, equal to the shortest key-preserving cycle of one
   template, with a clock-reset/saturation side condition that predicts the presence or absence of
   the tax correctly on 6 of 6 families before the search starts.
4. **The retiring zone is *not* the retiree's descendant** (0 of 40 sampled), so the sibling
   document's "stop spending on the state-count axis" fallback is not triggered.
5. **Sweep-line is provably vacuous here**, because the reachable graph is a single SCC covering
   100 % of states — a complete impossibility, not an estimate. Sharper than it first looks:
   **UPPAAL has shipped the generalised sweep-line since 2006** (`progress { … }`), so this is not
   a missing feature but a mechanism with provably nothing to act on, and the linear-time SCC
   check says so before any work is spent.
6. **Componentwise simulation subsumption is a real gap in the TA literature** (no published
   zone-based engine prunes across different discrete parts) and is worth stating — but its
   collapse factor is exactly 1 on every hand-written protocol, for a reason that generalises.
