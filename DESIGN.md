# PTCTLMC design notes

This document records the correctness arguments and the measurements behind the design of
PTCTLMC's liveness engine (`src/netliveness.cpp`) and of the parts of the forward engine
(`src/netreach.cpp`) it depends on. It is organised by topic. Each section states a claim, the
argument that discharges it, and — where a design choice was decided by measurement rather than
by proof — the numbers that decided it.

Notation is plain ASCII. `a_LU` is the LU simulation abstraction; `a_M` is the abstraction
obtained by setting `L = U = M` per clock; `Extra_LU+` is the LU+ extrapolation operator;
`down(X)` (equivalently `X^down`) is the time-predecessor of `X`; `Z |= phi` means the zone `Z`
lies wholly inside the half-space `phi`.

---

## 1. What the engine decides, and the shape of the computation

`p --> q` (leads-to) asks whether every reachable state satisfying `p` inevitably reaches a state
satisfying `q` on all maximal runs. Its negation is `E<> (p /\ EG !q)`. `EG` is a greatest
fixpoint, so the engine builds the reachable zone graph forward and solves

    W  =  W  /\  ( NoSucc  \/  Pre(W) )

backward, one federation of DBMs per node. A non-empty component at a node satisfying `p /\ !q`
is a counterexample. `EG phi` is the same computation with `psi = phi` inspected at the initial
vertex; `AF phi` follows by the duality `AF phi = !EG !phi`.

The backward direction is not a stylistic choice. Subsumption may redirect an edge to a zone that
strictly covers its exact successor, so every edge of the stored graph satisfies only
`post_e(Z_v) subseteq Z_v'`. Forward cycle detection over such a graph closes loops through
subsumption edges whose target is not the real successor, and is unsound (Tripakis 2009; Laarman
et al., CAV 2013). The backward operator never constructs a witness: it reads a target's value
only as an upper bound, which is the direction soundness needs, so a subsumption edge is fatal
forward and harmless backward.

Three witness classes support a maximal `psi`-execution:

| Witness class | Form | Preserved by a simulation? |
|---|---|---|
| infinitely many discrete steps (a cycle) | "can keep stepping" | yes |
| time diverges at `v` | "can delay forever" | yes |
| **deadlock at `v`** | **"enables nothing"** | **no** |

The first two are existential and transfer under any simulation without side conditions. The
third is a universally quantified *negative* property. Simulation gives `nu` enables `e` implies
`nu'` enables `e`, and nothing in the other direction. The whole of Section 2 exists to repair
exactly that third case, and it is a local repair — one witness of three, not the foundation.

**Lemma (positive witnesses transfer).** Let `nu <=_LU nu'`. If `(q, nu)` admits a maximal
`psi`-execution that is either infinite in discrete steps or time-divergent, then `(q, nu')`
admits one with the same sequence of discrete states. *Proof.* `<=_LU` is a simulation closed
under delay and reset, so every step of `nu` is matched by `nu'` at the same discrete state, and
`psi` reads only the discrete state. Divergence is the ability to delay without bound, matched
for the same reason.

**Lemma (deadlock does not transfer).** There exist `nu <=_LU nu'` with `(q, nu)` deadlocked and
`(q, nu')` not. *Proof.* One clock `x`, `L(x) = 1`, `U(x) = 3`, a single edge guarded `x <= 3`.
Take `nu(x) = 5`, `nu'(x) = 2`. Then `L(x) < nu'(x) < nu(x)`, so `nu <=_LU nu'`, while `nu` can
never satisfy the guard and `nu'` satisfies it immediately.

The `<=_LU` preorder is, for every clock `x`:

    nu'(x) = nu(x)   or   L(x) < nu'(x) < nu(x)   or   U(x) < nu(x) < nu'(x)

and the abstraction is the downward closure `a_LU(Z) = { nu | exists nu' in Z. nu <=_LU nu' }`.
Every valuation the abstraction adds therefore does *at most* what some real valuation does.

---

## 2. The deadlock completion

### 2.1 The construction

Deadlock is unpreservable by simulation because it is a negative property. The completion stops
treating it as one. For a discrete state `q` define

    Esc(q)  = { ( g_e /\ Inv_q /\ [R_e]^-1 Inv_{tgt(q,e)} )^down  |  e in out(q) }
    Dead(q) = Inv_q \ union Esc(q)

`Esc(q)` collects, per outgoing edge, the valuations from which that edge can still be taken after
some delay; `Dead(q)` is what remains, i.e. exactly the valuations from which no transition is
ever enabled. `Dead(q)` is in general not convex and is represented as a finite union of zones.

**Definition (deadlock completion).** `A^perp` is `A` extended with a fresh location `bot` with
`Inv_bot = true` and an unguarded self-loop, together with an edge `(q, C, {}, bot)` for every `q`
and every zone `C` in the representation of `Dead(q)`.

**Lemma (completion).** `A^perp` is deadlock-free, and the maximal `psi`-executions of `A` that
end in a deadlock are exactly the maximal `psi`-executions of `A^perp` that reach `bot`.
*Proof.* At `(q, nu)` with `nu |= Inv_q` exactly one of `nu in union Esc(q)` and `nu in Dead(q)`
holds, so in the first case an original edge is enabled after some delay and in the second the
`bot`-edge is enabled immediately; no deadlock survives. `Dead(q)` is closed under delay within
`Inv_q`, since each member of `Esc(q)` is a time-predecessor and therefore downward closed, so an
execution that delays *into* a deadlock is matched by the `bot`-edge at the instant it becomes
stuck. Conversely the `bot`-edge is enabled only on genuinely stuck valuations. Finally
`bot !|= psi` and time diverges at `bot`.

The completion removes the third witness case entirely: over `A^perp` a liveness witness is either
an infinite run or a run reaching `bot`, and both are existential, hence preserved by any
simulation. Liveness-with-deadlock over `A` becomes liveness-with-only-cycle-and-divergence
witnesses over `A^perp`.

**The completion is never materialised.** Its edges contribute no symbolic states: `post` of a
`bot`-edge is contained in the source zone, and the `bot` self-loop reproduces exactly the term
`Z_v /\ Dead(q(v))` that the stopping term already contributes to the fixpoint operator. Its only
effect is on which constraints the covering test must respect. Writing `G(q)` for the constraint
map of `A` in the sense of Gastin–Mukherjee–Srivathsan (CAV 2019) — the guards and invariants at
`q`, closed under `pre` — the completion adds at each `q` the atoms

    Datoms(q) = { phi tight in Dead(q) } \ { phi | phi /\ Inv_q = empty }

again closed under `pre`. Write `<=_G` for the resulting simulation.

**Theorem (completion sound).** `<=_G` is a simulation of `A` that preserves invariants,
divergence and `Dead`: if `nu <=_G nu'` at `q` then `nu in Dead(q)` implies `nu' in Dead(q)`.
Consequently, if `Z <=_G Z'` at `q` then covering `Z` by `Z'` loses no witness of the fixpoint
operator. *Proof.* That `<=_G` is a simulation is Gastin–Mukherjee–Srivathsan applied to
`A^perp`, whose transitions include those of `A`. Invariants are carried by `G`. For deadlock,
`nu in Dead(q)` means `nu` satisfies some member `C` of the representation of `Dead(q)`, whose
tight atoms all lie in `Datoms(q)`; each is preserved at zero delay, hence `nu' |= C` and
`nu' in Dead(q)`. The statement about executions follows by coinduction, using the positive-witness
lemma for the two positive cases and the clause just proved for the third.

### 2.2 `Dead(q)` is a DIAGONAL set even in a diagonal-free automaton

This is the fact that decides the cost of the construction, and it is not apparent from the
definition.

Each member of `Esc(q)` is a **time-predecessor**, and for a conjunction of non-diagonal atoms

    down( x <= u  AND  y >= l )  =  { x - y <= u - l }

Delaying a region that has an upper bound on one clock and a lower bound on another leaves a
constraint on their *difference*: to still be able to reach `y >= l` one must not already have
spent more than `u - l` of `x`'s budget. So the escape sets — and hence their complement
`Dead(q)` — constrain clock differences, and negating an escape region and taking its
time-predecessor manufactures a diagonal constraint out of two non-diagonal ones.

This is not an artefact of the DBM encoding. "The window closes before the guard opens" is
intrinsically a statement about how far apart two clocks are.

### 2.3 Why no per-clock `(L, U)` scheme can preserve deadlock on its own

Immediately: an `(L, U)` pair is a per-clock object. It can only say things of the form "clock `x`
above/below constant `c`". `Dead(q)` is a boolean combination of constraints on clock
*differences*, and no family of per-clock bounds can separate two valuations that agree
clock-by-clock on every such bound while differing on `x - y`. The deadlock-not-preserved lemma of
Section 1 exhibits this concretely with one clock and one guard.

This is also why the standard fallback (Behrmann–Bouyer–Larsen–Pelanek 2006) must go all the way
to a bisimulation rather than to some intermediate per-clock bound: within the LU family the only
deadlock-preserving point is the degenerate one where the preorder becomes symmetric. There is no
useful intermediate.

Anything that beats `a_M` on a deadlock-sensitive query must therefore key on structure other than
LU bounds. Herbreteau–Srivathsan–Walukiewicz (Inf. Comput. 2016, arXiv:1110.3705, Thm 14) prove
`a_{<=LU}` is the *coarsest* abstraction sound and complete for reachability among automata with
the same LU bounds, so no gain is available from re-tuning the bounds. The completion keys on the
constraint map instead, which is exactly a structure the LU bounds cannot express.

### 2.4 The three drop conditions

`Datoms(q)` is often empty, and emptiness can be recognised **statically**. The atoms may be
dropped at `q` whenever

1. **`Dead(q) = empty`** — there is no `bot`-edge at `q` at all, so there is nothing to preserve.
2. **`Dead(q) = Inv(q)`** — the `bot`-edge guard is then vacuously true on everything the search
   will ever store, since every stored zone at `q` satisfies `Inv(q)`. Reaching `bot` from `q` is
   unconditional and no atom is required.
3. **`Inv(q)` bounds no clock, and `q` is not frozen** — time diverges at `q`, so the divergence
   disjunct of the stopping term already admits the whole zone and the deadlock disjunct
   contributes nothing new.

Condition (iii) **must** exclude frozen (committed or urgent) locations. At a committed or urgent
location no positive delay is legal at all, while `Inv(q)` there is the *undelayed* universe and
therefore bounds no clock vacuously. Un-gated, (iii) fires exactly where its reasoning fails. The
regression `cex3.xml` (`P.U --> P.Live`, correct answer NOT satisfied) is the witness; frozen keys
fall through to (ii) or to PARTIAL.

Conditions (i) and (ii) rest on the premise that **every stored zone at `q` lies inside `Inv(q)`**.
That premise holds because the active-clock analysis seeds the active set out of the invariants, so
the reduction can never zero a clock an invariant reads.

Two further points about the test:

* **It is structural, not a search.** `Dead(q)` is computed over all of `Inv(q)` rather than over
  any reachable zone. That is the conservative direction, and it costs one federation computation
  per discrete key and no exploration — 256 keys on arbiter6, against the 28 591-state
  reachability traversal a deadlock-freedom proof would cost on the same model.
* **Enumerating keys from the covered graph is complete.** A simulation never loses a reachable
  discrete key: the covering zone does at least what the covered one does, so every key the covered
  zone could reach is still generated.

`Inv(q)` and the universe must be built from the *universe* DBM, not from the zero DBM. A DBM
constructed from a clock count alone is the zone where every clock is `0`; delaying it writes only
column 0, so every diagonal entry stays at `(<=, 0)` and the "universe" is really the
one-dimensional ray `x1 = ... = xn >= 0`. `Dead(q)` computed on a measure-zero set makes (i) and
(ii) meaningless and the certificate fires on false evidence. The regression `cex1.xml`
(`P.Q --> P.Live`, correct answer NOT satisfied) is the witness.

If every reachable key satisfies one of (i)–(iii) then `Datoms` is identically empty, `<=_G`
coincides with ordinary LU subsumption, and **plain `a_LU` covering is already sound for liveness
with deadlock** — no gate, no second search, no weakening to `a_M`.

### 2.5 How the two kinds of atom are discharged

An atom is a half-space `x_i - x_j <| b`. Two kinds, discharged differently:

**DIAGONAL (`i > 0` and `j > 0`).** Invariant under time elapse — delaying adds the same amount to
both clocks, so their difference is unchanged — hence agreement on it survives delay, hence it can
be carried as a **cell tag** attached to the discrete key: zones classified differently by the atom
are never allowed to cover one another. The covering kernel itself is untouched.

**NON-DIAGONAL (`i == 0` or `j == 0`), i.e. a bound on one clock.** *Not* delay invariant: two
valuations agreeing on `x <= 5` need not agree on it after delaying, so no cell tag can carry it.
What does carry it is the `a_LU` preorder itself. With `L(x) >= c` and `U(x) >= c` its three
clauses collapse to "equal, or both above `c`", which decides every constraint on `x` with constant
`<= c` the same way for both valuations. A non-diagonal atom is therefore discharged by **raising
that one clock's bounds**: `a_M` on that clock alone, rather than `a_M` on the whole model.

With both in place the relation

    nu R nu'   iff   nu <=_LU nu' (under the raised bounds)
                     and nu, nu' agree on the diagonal atoms

is a simulation for the completed automaton, because (a) `<=_LU` is a simulation for the original
automaton, (b) diagonal agreement is preserved by delay and, via the pre-closure, by every edge,
and (c) `Dead(q)` is a boolean combination of `Inv(q)` (which every stored zone satisfies), the
diagonal atoms and the non-diagonal ones — so `nu |= Dead(q)` iff `nu' |= Dead(q)`. Deadlock is
then existential reachability of the sink, which `R` preserves.

Raising `L` or `U` only ever makes `a_LU` **finer**, so it is sound wherever it is applied. It is
applied both to the global bound arrays and, at insertion time, to the per-location ones, which is
what keeps it consistent along every path — the consistency the simulation argument needs.

The active-clock reduction pins clocks that no guard will read again to `0`. The atoms *do* read
them — that is the whole point of `Dead(q)` — so every clock occurring in the map is added to the
query's live-clock mask. Without this the reduction erases exactly the differences the cell tag
exists to observe. (Whether the pinning is strictly owed for *seed* atoms is open: a clock read by
`Esc(q)` is read by an outgoing guard or a target invariant at `q`, hence active at `q` by
construction, and a diagonal that survives pullback involves only clocks some reachable guard still
reads. That is an argument, not a proof — the active set is a static over-approximation — so the
pinning stays.)

### 2.6 Pre-closure

The map must satisfy `[R]^-1 (G(q')) subseteq G(q)` for every edge `(q, g, R, q')`, or two zones
sharing a cell at `q` could step into different cells at `q'`. The pullback of `x_i - x_j <| b`
through a reset set `R` is the index substitution `i, j -> 0` for reset clocks — the same rule the
deadlock test uses for target invariants. Special cases:

* both clocks reset: the atom becomes a constant, not an atom, and is discarded;
* one clock reset: the atom **decays** into an ordinary per-clock bound and is discharged by the
  bound raise instead of by a cell;
* neither reset: the atom survives verbatim.

Dropping an atom at the target only ever **relaxes** the clause at the source, so the three seed
drops stay safe under closure.

Guard atoms need no seeding: `a_LU` is already a simulation for the *original* automaton, and the
only edge this construction adds is the sink's.

Two minimisations apply at closure, and the second is where the drop earns its keep:

* an atom **implied** by `Inv(q)` is constantly true on every zone stored there;
* an atom **refuted** by `Inv(q)` (disjoint from it) is constantly false on every zone stored
  there.

Either way its tag bit never varies and it distinguishes nothing. On a canonical DBM both are one
entry comparison: implied is `inv(a,b) <= r`; disjoint is `inv(b,a) + r < (<=, 0)`. A closure atom
is a constraint from *somewhere else's* deadlock, and the further it is pulled back the less it has
to do with the invariant it arrives at, so the refutation drop is the productive one.

### 2.7 Which zones the atoms are read off

`Dead(q) = Inv(q) \ union Esc(q)`, and every stored zone satisfies `Inv(q)`. So membership in
`Dead(q)` is decided **either** by the atoms of the residue's disjoint pieces
(`Dead = OR_i AND_j c_ij`) **or** by the atoms of the escape zones alone — one convex zone per
outgoing transition — since a valuation is dead iff it escapes by none of them. Both are exact and
neither dominates:

* the residue's cut constraints are artefacts of the peeling order and blow up: **3234** atom
  occurrences on grc4 against **25** for the escapes;
* the escapes carry diagonals that pull back through resets into non-diagonal atoms, costing csma
  four clock bounds the residue does not.

Both are therefore computed and the cheaper kept, counting a non-diagonal atom **four times
heavier** because it is discharged by raising that clock to `M` *globally* rather than by a local
cell.

The atoms are read straight off the **canonical** DBMs rather than derived from the guards. Since
`Dead = OR_i AND_j c_ij`, the truth values of the `c_ij` decide membership exactly and every atom
family comes for free — including the lower bound that survives `down()`, which the DBM `down`
operation re-derives when it re-closes after zeroing row 0, and which a hand-derived formula for
`down(K)` gets wrong in half the cases.

They are read off the **minimal graph**, not the dense matrix. A canonical DBM's redundant entries
are implied by shortest paths through the others, so the non-redundant edges alone decide
membership, and the minimal-graph round trip reproduces the canonical matrix exactly. Reading the
dense matrix seeds one atom per *derived* bound and buys nothing.

**Escapes that miss `Inv(q)` are not escapes.** Since `Dead(q) = Inv(q) \ union Esc(q)`, a
transition whose escape region does not meet `Inv(q)` subtracts nothing and its atoms decide
nothing — yet every one of its constraints would otherwise be seeded, pulled back through the whole
key graph, and split against. Such members are dropped where the intersection is being computed for
the `Dead = Inv` test anyway.

### 2.8 `Inv \ Dead = Inv /\ union Esc`

Testing "is `Dead(q)` all of `Inv(q)`?" by subtracting the residue peels `Inv(q)` once per piece of
the residue, and the residue is an *arrangement*: **472 pieces** at grc4's single partial key,
which is **126 ms** — two thirds of the whole structural sweep, and the single largest term in the
atom construction's bill. The identity

    Inv \ Dead  =  Inv \ (Inv \ union Esc)  =  Inv /\ union Esc

gives the same predicate as "does *no* valuation of `Inv(q)` escape", which is `|Esc|` independent
DBM intersections — one per outgoing transition, five on that key. Measured on grc4: **126 ms ->
0.9 ms**.

The premise is **inclusion, not equality**, and the difference is real. The residue `stuck` is
`Z \ union Esc` for the `Z` the deadlock test built, and that `Z` is not always this `inv`: at a
frozen key the deadlock test skips both the delay and the invariant, so its `Z` is the bare
universe while `inv` has the invariant applied. What the identity needs is only `inv subseteq Z`:

    x in inv \ (Z \ E)  iff  x in inv and not (x in Z and x notin E)  iff  x in inv /\ E

using `x in inv => x in Z` — and that holds both ways round: equal when not frozen, and
`inv = universe /\ Inv subseteq universe = Z` when frozen.

### 2.9 The atom cap, and the all-at-`M` shortcut

The per-node cell tag is a `uint64` bitmask, so an atom past bit 63 would be silently dropped from
the covering test while still splitting zones — a **lost obligation**, i.e. a wrong verdict, not
merely a coarseness loss. Exceeding the cap returns "unusable" and falls back to the gate and race
(Section 7), which is always correct.

The cap sits at the **hard limit of the representation, 64**, not at half of it. A cap of 32 costs
nothing while the construction is expensive — declining early means declining into the race, and
the race is competitive — but it is where grc5 falls out: one key there needs **36** seed atoms, so
the map would be declared unusable and grc5 would run the race (**20.9 s**) or `a_M` (**18.5 s**)
where the construction does it in **3.4 s**. There is no reason to decline below the point where
the representation actually breaks. The construction also declines outright for `nclocks >= 63`
and for `nclocks <= 0`, and if the pre-closure fails to converge within 4096 rounds.

**If every clock ends up raised to `M`, the cells are redundant.** With `L(x) = U(x) = M(x)` for
every clock the LU preorder collapses to "equal, or both above `M`", which is symmetric, hence a
bisimulation, hence deadlock-preserving on its own. Keeping the cell tags on top of that would only
split zones `a_M` already keeps apart. Dropping them in that case is what stops the construction
from ever being *worse* than the `a_M` fallback it is meant to beat.

---

## 3. Why `L = U = M` is a bisimulation, and why deadlock queries need one

Set `L = U = M` with `M(x) = max(L(x), U(x))` per clock. The three clauses of the LU preorder —

    nu'(x) = nu(x),   or   L(x) < nu'(x) < nu(x),   or   U(x) < nu(x) < nu'(x)

— collapse to "equal, or both above `M(x)`". That relation is **symmetric**. A symmetric simulation
is a bisimulation, and deadlock *is* preserved by bisimulation. So the same `dbm_is_alu_le` kernel,
fed collapsed bounds, **is** the `a_M` test and **is** deadlock-sound; it needs no gate at all.

**Citation.** Behrmann, Bouyer, Larsen, Pelanek, *Lower and upper bounds in zone-based abstractions
of timed automata*, STTT 8(3):204–215, 2006: **Lemma 1** (`=_M` is a bisimulation), **Lemma 3**
(`<=_LU` is only a simulation), and the unnumbered prose in the "Simulation & Bisimulation"
paragraph of Section 3 stating that UPPAAL's deadlock properties are not preserved by simulation
but are by bisimulation.

**UPPAAL itself does exactly this.** Its extrapolation is Lower/Upper for reachability but *Local*
(per-location max constants) "whenever the query contains either the deadlock keyword or involves a
liveness property". It falls back from LU to local max constants — not to exact zones. Any
implementation that falls back to exact zones for deadlock is strictly stronger than UPPAAL and
pays for it.

`E<> deadlock` and `A[] not deadlock` therefore run on `a_M`, obtained by **collapsing the
network's bound arrays** rather than by threading a flag through the search. That is deliberate:
every downstream consumer of `L`/`U` becomes `M`-flavoured at once and consistently. It matters
beyond the covering test — the adaptive path can turn extrapolation back on mid-search, and
`Extra_LU+` with `L = U = M` is `Extra_M+`, which, unlike `Extra_LU+`, can neither invent nor
destroy a deadlock. Threading a flag would have left that path unsound. The deadlock predicate
itself still closes over the original network: it reads guards, invariants and edges, never bounds.

Diagonal networks and G-simulation covering keep exact behaviour instead, because diagonals make
every LU-family abstraction unsound and the G bound arrays carry raw encoded bounds with their own
sentinel, so collapsing them correctly is a separate exercise.

### Measured: `A[] not deadlock`, exact zones against `a_M`

Exact inclusion costs `E<> deadlock` **167 805** states on fischer7 against **11 951** for plain
reachability on the same model. Against `a_M`:

| model | exact zones | `a_M` | states | time |
|---|---|---|---|---|
| fischer6 | sat / 21 702 / 0.07 s | sat / 12 036 / 0.05 s | 1.8x | 1.6x |
| fischer7 | sat / 175 006 / 0.99 s | sat / 83 344 / 0.44 s | 2.1x | 2.2x |
| fischer8 | sat / 1 830 337 / 42.35 s | sat / 777 649 / 14.32 s | 2.4x | **3.0x** |
| arbiter6 | sat / 26 648 / 0.97 s | sat / 7 188 / 0.10 s | 3.7x | **9.4x** |
| arbiter7 | **timeout > 120 s** | sat / 80 350 / **8.00 s** | — | — |
| csma6 | sat / 25 171 / 0.19 s | sat / 6 263 / 0.04 s | 4.0x | 4.6x |
| csma7 | **timeout > 120 s** | sat / 68 839 / **1.81 s** | — | — |
| phil4 | NOT sat / 31 | NOT sat / 31 | 1.0x | 1.0x |
| grc4 | NOT sat / 618 | NOT sat / 613 | 1.0x | 1.0x |

15 of 15 models agree on verdict, **including the two where a deadlock is genuinely reachable**
(phil4, grc4). That is the completeness check that matters, since the risk of a coarser abstraction
is losing a deadlock, not inventing one. arbiter7 and csma7 move from timeout to solved — a query
class the engine could not previously answer at all.

---

## 4. Stored zones must be exact

**Invariant: whenever `a_LU` covering is in force, stored zones are exact.** Finiteness of the zone
graph then comes from `a_LU` subsumption, not from extrapolation.

The covering discipline and the extrapolation mode are **one decision** and cannot move
independently:

* Plain inclusion alone does not terminate. It does not bound the zone graph; with `a_LU` off, the
  extrapolation is the only thing that makes the search finite.
* `a_LU` alone does not repair anything if the stored zones remain extrapolated, because it is the
  zone *content*, not the covering, that breaks the stopping term.

### Why extrapolated zones break the stopping term

The stopping term computes an exact **per-valuation** deadlock residue, and it computes it from the
stored zone. `Extra_LU+(Z)` is contained in `a_LU(Z)`, so every valuation extrapolation adds does
*at most* what some real valuation does. Enabledness is therefore **monotone**, not invariant:

* "some valuation of the extrapolated zone enables `e`" implies "some valuation of `Z` does" —
  holds;
* "some valuation of the extrapolated zone enables **nothing**" implies the same for `Z` —
  **fails**.

The stopping term asks the second question. And `stop` occurs **positively** in the fixpoint
operator, so a spurious stop valuation **enlarges** the greatest fixpoint and **fabricates** a
certificate: `p --> q` reported VIOLATED when it holds. This is the opposite error direction from
`a_LU` covering, which can only ever lose a witness, and unlike that one it is not recoverable by
any later check.

### The `Extra_LU+` counterexample

Diagonal-free, two clocks, one edge — so the ordinary LU+ extrapolation genuinely applies.

    location q, invariant  x <= 10
    edge     q -> q2, guard  y >= 3
    =>  L(x) = 0, U(x) = 10, L(y) = 3, U(y) = 0

Reach `q` with `y` reset 6 time units after `x`. The stored, delay-closed, invariant-respecting
zone is

    Z = { x - y = 6,  6 <= x <= 10 }        (equivalently 0 <= y <= 4)

`Z` is deadlock-free: from any valuation, delay until `y >= 3`, i.e. `x >= 9`, which the invariant
`x <= 10` still permits.

Now apply `Extra_LU+`. The diagonal `x - y <= 6` is dropped because `6 > L(x) = 0`; `x <= 10` and
`y <= 4` go for the same reason. What survives is

    Extra_LU+(Z) = { x >= 6,  y >= 0,  x - y >= 6 }

The deadlock test rebuilds the delay closure and re-applies the real invariant, giving
`{ 6 <= x <= 10, 0 <= y <= x - 6 }`. That contains `(x = 10, y = 0)`, which needs `d >= 3` to
satisfy the guard and `d <= 0` to satisfy the invariant: **permanently stuck**. No valuation of `Z`
is.

The invariant re-application does not repair this and cannot: the constraint that was dropped is a
**diagonal**, and in a diagonal-free automaton no invariant ever restores a diagonal. The
correlation `x - y = 6` is exactly what guaranteed `y` reached 3 before `x` reached 10.

Consistency check against the theory: `(10, 0) <=_LU (6, 0)`, and `(6, 0)` in `Z` delays to
`(9, 3)` and fires. The simulator does more, as it must.

### A related counterexample, on the stored-zone side

A per-valuation stuck test on an unrepaired stored zone reports a false violation of arbiter4
`C1.cs --> C1.idle`, witnessed by `x0..x3 > 4` — points `Extra_LU+` invents at `U = 4` and the
system never reaches. The proximate cause there is that the **active-clock reduction leaves the
stored zone not delay-closed**: it pins dead clocks to `0` with an upper bound, so `down()` cannot
let time pass them and the test invents deadlocks. The deadlock test therefore rebuilds the true
delay closure — re-open every upper bound, re-impose only the real location invariant — before
testing. Dead clocks become free, which is sound because no guard or invariant there reads them.
With the repaired routine, arbiter4 `C1.cs --> C1.idle` stays SATISFIED.

Both counterexamples are why the exactness constraint looks arbitrary at the point in the code
where the covering discipline and the extrapolation mode are chosen together, and why it is not.

### What exactness costs: nothing measurable

Full differential over 48 leads-to queries across arbiter4/5, fischer4/5, csma4/5, phil4, twolock4,
viking4, grc4:

| | result |
|---|---|
| queries | 48 |
| identical verdict **and** stored-node count | **48** |
| verdict differences | 0 |
| node-count differences | 0 |

`a_LU` subsumption already covers everything the extrapolation would have widened, so extrapolating
on top of it only costs time. The forward engine records the same observation for reachability
(zones per discrete state: phil 1.11, viking 1.00; extrapolation there changes no state count and
costs 20–35%).

### What `a_LU` itself is worth

`a_LU` is doing all the work, and it is worth keeping. The third column is `a_LU` disabled:

| query | `a_LU` + exact zones | `a_LU` + `Extra_LU+` | inclusion + `Extra_LU+` |
|---|---|---|---|
| grc4 `T0.in --> T0.far` | SAT / 883 | SAT / 883 | SAT / 1183 |
| fischer5 `P0.cs --> P0.idle` | NOT / 762 | NOT / 762 | NOT / 2770 |
| arbiter5 `C1.cs --> C1.idle` | SAT / 112 | SAT / 112 | SAT / 2726 |
| csma5 `Bus.Active --> Bus.Idle` | SAT / 272 | SAT / 272 | SAT / 3336 |

Without simulation-based covering the liveness graph is enormously larger than the reachability
passed set for the same model: on arbiter6, **256 discrete keys carrying 66 522 zones — 260 zones
per discrete state** — where the reachability engine stores exactly one. Before subsumption the
graph reaches **1 343 096** nodes where `verifyta` stores **10 453**. `a_LU` is what collapses that.

### The one exception

Diagonal models cannot use `a_LU` at all (Bouyer 2004). They keep diagonal-preserving extrapolation
with plain inclusion covering, and extrapolation stays load-bearing there.

---

## 5. The covering-test contract

### 5.1 The reference clock must be `(<=, 0)`

Both bound arrays fed to a simulation kernel must carry `(<=, 0)` at index 0, not the "no bound"
sentinel.

For the `a_LU` kernel this is documented as mandatory and the code sets `L[0] = U[0] = 0`. For the
G-simulation kernel the same requirement holds and the failure mode is concrete: both of its loops
start at index 0 and `continue` on the sentinel. A sentinel at the reference clock therefore
**deletes two witness families** — the `(U-cut)` family at `y = 0` and the `(L-unsat')` family at
`x = 0`.

Removing witnesses from an existentially quantified test can only turn its answer from FALSE to
TRUE. A spurious TRUE means the passed set **prunes a zone that is not actually simulated**, i.e.
drops reachable behaviour. Measured against point-wise ground truth: **~19% unsound TRUEs** with the
sentinel, versus **zero** for `(<=, 0)`.

Minimal case: `Z = {0 <= x <= 10}`, `Zp = {5 <= x <= 10}`, `G = {x <= 100, x >= 100}`.

The query's own clock constants must also floor the bounds exactly as they do for `L`/`U`, or a
state satisfying the property could be abstracted away. The flooring is non-strict — the safe
direction.

### 5.2 The classification is three-valued

Stored zones are kept whole rather than cut along the atoms, so a zone may **straddle** an atom.
One bit per atom cannot express that, and reading a straddle as "disjoint from `phi`" is the one
mistake that makes the covering test unsound. The classification is therefore three-valued and
rides in two masks, both read off the canonical DBM in O(1):

    Z.get(i, j) <= b                    =>  Z subseteq phi        (In)
    raw_add(Z.get(j, i), b) < kZeroLe   =>  Z /\ phi = empty      (Out)
    otherwise                                                     (Straddle)

The two masks are disjoint by construction; a bit set in neither means the zone straddles that
atom.

### 5.3 The obligation, and the recursion

`a` is covered by `b` when, for every valuation `nu` in `a`, there is a `nu'` in `b` with
`nu <=_LU nu'` **and**, for every atom `phi` of the key, `nu' |= phi => nu |= phi`. **That
implication direction only.** It is what stops `a_LU` covering from dropping a deadlock: if the
covering zone satisfies an escape atom and the covered one does not, the simulator escapes where the
simulated valuation is stuck, and the deadlock is lost.

Spelt out: `nu'` in `Z` simulates `nu` in `Z'`, and `Dead(q) = Inv(q) /\ AND_e !Esc_e` with each
`Esc_e` a **conjunction** of atoms, so `nu |= Dead` needs, per outgoing edge `e`, some atom of
`Esc_e` that `nu` falsifies to also be falsified by `nu'`. Equivalently `tag(Z) & ~tag(Z') == 0`.
This is strictly **coarser** than same-cell equality, which needlessly refuses the pair
(`Z` disjoint from `phi`, `Z'` contained in `phi`).

An atom is discharged for all pairs at once exactly when `b` is Out (no `nu'` satisfies it) or `a`
is In (every `nu` does). An atom that is neither is **blocked**, and only blocked atoms are split.
The recursion is Gastin–Mukherjee–Srivathsan, CAV 2019, **Lemma 17**, adapted to the implication
obligation rather than to agreement:

    a <= b   <==   (a /\ phi <= b)   and   (a /\ ~phi <= b /\ ~phi)

`nu |= phi` discharges the `phi`-obligation for any `nu'` whatever, so the first conjunct may keep
the whole of `b`; `nu !|= phi` needs `nu' !|= phi`, so the second must narrow `b` as well. Each
conjunct resolves one blocked atom (In on the left, Out on both sides on the right), so the blocked
set strictly shrinks and the recursion terminates in at most `2^|blocked|` leaves. That `2^d` split
is the source of the NP-completeness of the general test.

The un-refined base test is a **necessary** condition for both conjuncts — both test sub-zones of
`a` against sub-zones of `b` — so asking it first is sound and doubles as a prefilter that rejects a
hopeless probe before any zone is copied. At depth 0 both masks come from the caller's cache, so an
unblocked probe costs one mask operation plus the base test.

### 5.4 The two rewriting paths

Insertion has two outcomes that **replace** a stored zone, and each owes something different:

* **In-place enlargement.** The stored zone is replaced by the newcomer, which under `a_LU` is not
  a superset of it, so everything already routed to that node must still be covered. That is
  transitivity of the covering relation, and the relation *is* transitive (`<=_LU` is a preorder
  and `nu' |= phi => nu |= phi` composes), so the ordinary three-valued test in the opposite
  direction is exactly and only what is owed.
* **Convex-union merge.** The union operation succeeds only when the union is *exactly* the union,
  so the stored zone afterwards is a set **superset** of both operands. Growing the right-hand side
  of an exists-witness can never invalidate a witness, and the newcomer is covered by the result via
  the identity witness. The merge therefore owes **nothing**: no tag condition at all.

Both paths must refresh the stored masks, since the stored zone changed.

---

## 6. Confined backward passes: why a partial graph is enough

The straightforward algorithm is three strictly sequential phases: build the whole `a_LU`-subsumed
reachable graph, restrict to `!q`, then run a backward greatest fixpoint over it. Its cost is a
function of the **model** and never of the **query** — a `p --> q` violated three transitions from
the initial state still pays for every reachable symbolic state.

`EG` is a **greatest** fixpoint, so the textbook iteration starts at TOP and **descends**, and an
intermediate iterate is only a **pre-fixpoint** (`W >= F(W)`). A pre-fixpoint is an
over-approximation and therefore says nothing: "`W(v)` is non-empty" cannot settle a violation, so
there is no early exit from the iteration itself.

A violation, however, never needed the fixpoint. It needs a **certificate**, and by
**Knaster–Tarski** the certificates of a greatest fixpoint are exactly its **post-fixpoints**:

    X  subseteq  F(X)        ==>        X  subseteq  nu F.                        (*)

So take any set `S` of nodes explored so far, pin every node outside `S` to EMPTY, and run the
ordinary descending iteration confined to `S`. Its limit `nu F_S` is the exact greatest fixpoint of
the restricted operator. Restricting only **deletes** disjuncts, so `F_S <= F` pointwise, and
monotonicity gives `nu F_S subseteq nu F`: the limit of a pass over a **fragment** is a
post-fixpoint of the operator on the **finished** graph. By (*) it is a sound under-approximation of
the final answer, computed without ever seeing the final graph.

**Theorem (partial certificate).** For every explored set `S_k subseteq V_psi`,
`nu F_{S_k} subseteq nu F`. *Proof.* For every vector `X`, `F_{S_k}(X)` is obtained from `F(X)` by
omitting predecessor terms whose target lies outside `S_k`, so `F_{S_k}(X) subseteq F(X)`. In
particular `nu F_{S_k} = F_{S_k}(nu F_{S_k}) subseteq F(nu F_{S_k})`, so `nu F_{S_k}` is a
post-fixpoint of `F`. Since `nu F` is the greatest post-fixpoint of `F`,
`nu F_{S_k} subseteq nu F`.

Two consequences:

* `nu F_S(v)` non-empty at a `p /\ !q` node `v` implies the query is **VIOLATED**, with exactly the
  verdict the finished graph would have produced.
* `S =` every expanded `!q` node of the **finished** graph gives `F_S = F`, so the last pass is the
  ordinary algorithm. Nothing is given up when the property holds.

**Completeness** is restored by making the eliminations local to a round rather than carried over:
if no earlier pass finds a witness, forward exploration continues until the frontier is exhausted,
`S_k = V_psi`, `F_{S_k} = F`, and the final pass computes `nu F`.

### The one restriction

**A node may enter `S` only if it is EXPANDED.** The stopping term of the backward operator needs
the **absence** of successors, and an unexpanded node has an empty adjacency for the wrong reason.
Everything else survives a partial graph untouched: the `Pre` disjuncts only ever need edges that
*exist*, never the absence of one. Backward work on a half-built graph is therefore legitimate for
every node whose own successors have been computed, whatever the rest of the graph is doing.

### Node immutability, and what in-place enlargement costs

An **append-only** insertion rule — never rewrite a stored zone; store a covering newcomer as a
separate node — makes every stored node immutable from birth. That is precisely the property "start
backward work only on vertices known not to change" wants, and it is why an append-only rule and a
"begin backward work early" rule are coupled: with in-place enlargement, "this node will never
change" is not establishable before the forward phase terminates. Append-only pays for that
property with the antichain, keeping zones a rewriting insertion would have collapsed.

The engine keeps rewriting (both enlargement and the convex-union merge) and pays the coupling
explicitly instead: **whenever a stored zone is rewritten, that node's adjacency is invalidated** —
it was computed from the smaller zone — the node is marked unexpanded, its out-edges are dropped and
the corresponding reverse edges removed, and it is re-queued. The invalidation happens at
**generation**, not at expansion, so no backward value ever refers to a zone that no longer exists.

### The forward worklist must deduplicate

A level-synchronous frontier that pushes a successor whenever it is unseen or unexpanded is harmless
while insertion only ever appends — `expanded` then goes 0 -> 1 once per node. It becomes
**quadratic** as soon as insertion rewrites: every enlargement resets `expanded` to 0, so the node
is re-queued once per in-edge, every level, and the frontier fills with duplicates of the same
handful of nodes. A residency flag makes each node resident at most once. Nothing about *what* is
explored changes — the closure is the same, so stored-state counts are unaffected — only how many
times it is queued.

### Restricting the final pass to the accepting closure is exact

The `!q` nodes that can influence the verdict are those forward-reachable, **through `!q` edges
only**, from a `p /\ !q` node. `F` is defined per node from its own successors, so the value at an
accepting node depends on exactly this set and on nothing else. Restricting to it is **exact**, not
an approximation, and it is what keeps the final pass from iterating over `!q` states no accepting
state can ever reach.

### Parallel evaluation

For fixed `S`, every component starts from `Z_v` and only decreases, so stale reads can only observe
larger values, delaying reductions without compromising the greatest fixpoint.

**Theorem (chaotic evaluation).** For fixed `S`, every fair asynchronous evaluation of `F_S`
terminates at `nu F_S`, provided reads observe complete previously published values. *Proof.*
Throughout, every published component contains its value in `nu F_S`. This holds initially and is
preserved by every update because `pre_e` is monotone and stale reads can only return larger values.
Since all components only decrease over a finite symbolic domain, the computation stabilises; since
every vertex in `S` is updated repeatedly, the result is a fixpoint of `F_S` containing `nu F_S`,
hence equal to it by greatestness.

### The exact predecessor

`Pre` is exact and applies no abstraction in the forward direction:

    forward:  z1 = Z /\ (all guards) ; reset (all resets) ; /\ inv' ;
              if !frozen' { delay ; /\ inv' }
    backward: undo the active-clock reduction at the target ; /\ inv' ; down ; /\ inv' ;
              /\ (x_r = 0 for every reset r) ; free those r ; /\ (all guards) ;
              down (the delay BEFORE the action) ; /\ inv(src)

Abstraction enters the fixpoint only through the caller's confinement `W_v subseteq Z_v`.

Two subtleties:

* **The active-clock reduction must be undone at the target.** Those clocks appear in no edge's
  reset set, so without the repair `Pre` silently demands that they also read `0` at the **source**
  — an over-constrained `Pre` that collapses the fixpoint and **misses violations**. Freeing them is
  the exact inverse of the reduction, and it is sound because they are dead: nothing reads them.
  This repair is what makes the active-clock reduction safe for a *backward* analysis, and it is why
  the query's live-clock mask can be all-zero (maximal reduction) for a leads-to predicate, which
  carries no clock constraints.
* **The delay before the action is a separate `down`.** A stored zone is delay-closed as a *set*,
  which is why the forward operator may write `Z /\ g`; a single valuation may still have to let
  time pass first.

An empty predecessor must be returned as an empty DBM **of the caller's dimension**. A
default-constructed DBM is a zero-clock placeholder — a 1x1 matrix holding `(<=, 0)` — and emptiness
is decided by scanning the diagonal for a negative entry, so such a value is *not* empty. It would
pass the non-empty filter, enter a federation of `nclocks + 1`-dimensional zones, and the next
inclusion test would read `dim * dim` entries out of a 4-byte allocation.

---

## 7. The fallback lane: the deadlock-freedom gate and the race

When the constraint map cannot be represented — more than 64 atoms at one key — the engine falls
back to a construction that is correct without the map. It is retained for that case and for
diagonal networks.

### The error is one-sided

Under `a_LU` covering on **exact** stored zones, every reported witness sits at a genuinely
reachable stored node whose own zone was tested, and `Pre` and the divergence test are exact
concrete operations. So:

> `a_LU` can only ever **lose** a witness, never invent one.

A witness found under `a_LU` is therefore genuine and settles the query immediately. Only the
*absence* of a witness is inconclusive, and only because a deadlock witness may have been covered
away.

### Consequence: pay for soundness only when the answer depends on it

    run the search under a_LU
    if a witness is found   -> VIOLATED. Sound outright; the gate is never paid.
    if no witness is found  -> the answer rests on a possibly-dropped deadlock, discharged
                               one of two ways:
      (A) prove the model deadlock-free      -> the a_LU answer was sound after all
      (B) redo the search with exact inclusion -> unconditionally sound

**Asymmetry that must be preserved:** lane A is conclusive only when it answers *deadlock-free*. "A
deadlock is reachable" says nothing about this query — the dropped witness may or may not have been
one — so that outcome publishes no verdict and does not cancel lane B.

### Neither lane dominates, and which wins is not predictable

Deciding which is cheaper is as hard as the query. Measured symbolic states:

| model | (A) deadlock-freedom gate | (B) exact re-run |
|---|---|---|
| fischer7 | 180 420 | 30 994 |
| arbiter6 | 28 591 | 66 393 |

A factor of **5.8** one way, **2.3** the other.

So both are run concurrently and the first conclusive answer is taken, cancelling the loser. This is
sound because both branches decide the same proposition. What the extra cores buy here is not speed
— neither lane runs faster than it would alone — but the **removal of an undecidable choice**. A
sequential tool must gamble on (A) or (B) and can lose by 5.8x.

Wall clock, satisfied queries (violated ones never reach the gate at all):

| query | A only | B only | race | winner |
|---|---|---|---|---|
| fischer5 `P0.req --> P0.wait` | 0.82 s | **0.02 s** | 0.02 s | B, by **41x** |
| fischer6 `P0.req --> P0.wait` | 0.35 s | **0.22 s** | 0.26 s | B |
| fischer7 `P0.req --> P0.wait` | **5.21 s** | 7.97 s | 5.55 s | A |
| arbiter5 `C1.cs --> C1.idle` | **0.03 s** | 0.04 s | 0.04 s | A |
| arbiter6 `C1.cs --> C1.idle` | **1.24 s** | 13.78 s | 1.28 s | A, by **11x** |
| csma5 `Bus.Active --> Bus.Idle` | 0.06 s | **0.05 s** | 0.06 s | B |
| csma6 `Bus.Active --> Bus.Idle` | **0.67 s** | 1.36 s | 0.73 s | A |

All three strategies return the same verdict on every query. Read the first three rows together:
**fischer5 favours B by 41x and fischer7 favours A — the same model family, the winner flips as it
scales.** That rules out "classify the model once and pick". The race pays `min(A, B)` plus 3–18%
overhead and is never exposed to either.

### Both lanes must be cancellable

A race in which only one lane observes the stop flag is not a race: the pair then always costs the
non-observing lane's wall clock, and the observing lane can only ever add work. Here the exact-lane
search polls the cancellation flag at the forward-expansion choke point every step passes through,
and at the head of each backward fixpoint sweep. The flag is **latched**, so nothing downstream of
the first observation can see it flip back and resume a half-abandoned search. The pair really does
finish in `min(A, B)`.

A cancelled reachability search returns "not found", which is indistinguishable from "no deadlock
reachable", so lane A's result may only be trusted when lane A was not the lane cancelled. A
cancelled run's verdict is not a verdict, and the racing caller consults that before publishing.

### Ordering: phase 1 first

On **violated** queries — precisely where this engine is orders of magnitude ahead of `verifyta` —
the gate is never paid at all. Hoisting it in front would make every counterexample query pay for a
full reachability traversal whose answer it never needs: on fischer7, **169 252 states and 3.86 s**
for the gate, against **6 nodes** for the liveness query itself.

Violated queries are unchanged to the millisecond with the gate on and off. Satisfied queries on the
small differential set pay 0.01–0.06 s absolute.

### Where the completion supersedes it

The completion's constraint map **subsumes** the structural certificate — an empty map is exactly
the certificate's condition — so where the certificate fires, the map is the same structural sweep
and the same `a_LU` search, node for node. Where the certificate declines, the map carries the model
instead of falling back to the race. Best of 3, `Bus.Active --> Bus.Idle` and `T0.in --> T0.far`:

| model | certificate + race | constraint map | `a_M` | `a_LU` floor (unsound here) |
|---|---|---|---|---|
| csma6 | 0.138 | **0.018** | 0.142 | 0.012 |
| csma7 | 3.985 | **0.053** | 9.729 | 0.033 |
| grc4 | 0.085 | **0.042** | 0.056 | 0.031 |
| grc5 | 21.936 | **3.535** | 19.475 | 5.815 |

**grc5 is the row to read twice: the refined covering beats the coarser abstraction it refines**
(3.5 s against the 5.8 s floor), because splitting a zone along an atom of `Dead(q)` also splits the
backward federations that pass through it, and narrower federations converge faster than the
arrangement one wide zone produces. The split is paid for twice.

Since the map is derived from the finished forward *graph* and, when non-empty, replaces the run
wholesale, **phase 1 stops after its forward half**. Its backward fixpoint is ~80% of its wall clock
(grc4: 20 ms of 25 ms) and none of it would survive. It is paid exactly once: on the empty-map path,
where phase 1's own verdict is the answer, or inside the refined search otherwise.

### Where the seed is empty

| model | keys | `Dead = empty` | time-open | **partial** | certificate |
|---|---|---|---|---|---|
| fischer4/5/6 | 220 / 727 / 2378 | all | 0 | **0** | fires |
| arbiter4/5/6 | 48 / 112 / 256 | all | 0 | **0** | fires |
| twolock4, viking4 | 284 / 206 | all | 0 | **0** | fires |
| phil4 | 80 | 79 | 1 | **0** | fires |
| csma4 | 104 | 100 | 0 | **4** | declines |
| csma5 | 272 | 267 | 0 | **5** | declines |
| grc4 | 128 | 127 | 0 | **1** | declines |

The seed is empty at every reachable key on **9 of 12** networks, and LU subsumption is used
unchanged there, at no cost. The remaining three — two csma instances and grc — carry a genuinely
partial `Dead(q)` at a handful of keys.

Certificate against the two alternatives, where it fires:

| query | certificate | race | `a_M` |
|---|---|---|---|
| fischer6 | **0.03 s / 2 694** | 0.20 s / 2 694 | 0.09 s / 12 401 |
| fischer7 | **0.10 s / 9 798** | 2.42 s / 9 798 | 2.05 s / 87 585 |
| arbiter5 | **0.00 s / 112** | 0.02 s / 112 | 0.01 s / 1 154 |
| arbiter6 | **0.00 s / 256** | 0.15 s / 256 | 0.64 s / 10 118 |
| csma6 | 0.16 s / 688 | 0.16 s / 688 | 0.13 s / 6 529 |
| grc4 | 0.20 s / 883 | **0.08 s / 883** | 0.05 s / 1 167 |

Where it fires: 22x faster than the race and 19x faster than `a_M` on fischer7, storing **8.9x fewer
states** than `a_M` there and **39.5x fewer** on arbiter6.

### A no-op must cost nothing

The pre-closure propagates atoms **backwards** from keys that carry them, so with no seed anywhere
it has nothing to move and the key-edge relation it would move them along need not be built at all.
That relation is a string key and a reset mask per graph **edge** — on fischer7, 9798 nodes' worth —
and building it is the only work the map construction does that the structural certificate did not.
Returning early on an empty seed is what makes the nine empty-seed models a literal no-op rather
than a no-op plus a sweep.

---

## 8. Measured and rejected

Recorded with numbers so that none of these is re-attempted on intuition.

### 8.1 Split-at-creation vs split-at-comparison vs lazy

Three ways to honour a straddling zone. **Split-at-creation** cuts every zone along every atom at
its key the moment it is produced, so stored zones are atom-homogeneous and covering is one mask
AND. **Split-at-comparison** stores zones whole and splits transiently inside the covering test.
**Lazy** stores whole but, when a covering decision genuinely turns on a stored zone's straddle,
splits that node permanently — once per zone rather than once per probe, and only on atoms that
actually arose (Herbreteau–Srivathsan–Walukiewicz's lazy abstraction). Lazy is split-at-comparison
plus that rewrite, so it inherits its correctness whole.

Stored states, zero verdict differences across five configurations:

| model | `a_LU` floor | creation | **comparison** | lazy |
|---|---|---|---|---|
| grc4 | 883 | 1179 | **921** | 921 |
| grc5 | 13 939 | 14 571 | **12 554** | 12 554 |
| csma7 | 1696 | 2368 | 2368 | 2368 |

**Comparison wins and is never slower.** The reason is not that its test is cheaper — it is that
**the recursion is never entered**: 0 of grc4's 216 blocked probes and 0 of grc5's 2660 recurse. At
the keys that carry atoms the `a_LU` antichain is already wide enough that no covering succeeds at
all, so the atoms never refuse a merge that would otherwise have happened, and an eager split only
fragments the passed set.

Two measurements pin that reading: with the convex-union merge off, creation goes **1179 -> 2579**
on grc4 while comparison is unchanged at **921** — the merge exists to glue back what the eager
split broke — and comparison without the atom-clock pinning reaches **883** on grc4, exactly the
`a_LU` floor.

**The lazy arm is a null result: it fires 0 times.** 0 splits and 0 activations on every model, so
it is bit-identical to comparison. Forced to fire, it stores *more* than either, because activating
an atom re-enables eager splitting for that whole key thereafter.

This does **not** contradict the forward-reachability result (Section 8.7), where the same recursion
costs 5x because coverings at atom-carrying keys *do* succeed there and it genuinely fires. Which
arm wins is a property of whether the atoms ever change a covering decision.

### 8.2 Atom minimisation

Atom minimisation — the obvious lever — **drops zero atoms on every model**. The atoms that survive
seeding nearly all separate zones, so there is nothing for a minimisation pass to remove. (This is
distinct from the *drop conditions*, which do remove atoms: those are the "implied by `Inv(q)`",
"disjoint from `Inv(q)`" and "escape misses `Inv(q)`" tests, and they are cheap entry comparisons,
not a minimisation pass.)

### 8.3 The closure-dominance predictor

A predictor that abandons the construction and runs `a_M` whenever the pre-closure adds more atoms
than the seed did. On grc4 (139 closure atoms against 25 seed, 100% straddle) the map lane measured
0.18 s against `a_M`'s 0.05 s, which appeared to justify it.

**That reading was wrong about where grc4's time went.** Split by phase, the map lane was 0.035 s of
searching behind 0.126 s of structural sweep and 0.025 s of a phase-1 backward fixpoint that the
refined search then repeated. Both of those are eliminated — the sweep by the `Inv /\ Esc` rewrite
of the `Dead = Inv` test (Section 2.8), the fixpoint by deferring phase 1's backward half — and the
map lane wins outright on the very model the predictor was built to hand to `a_M`. The predictor
**picks wrong** and is not used.

**The straddle rate was never the bill:** grc4 splits 100% of its tagged zones and the refined
search still costs the same.

### 8.4 `a_M` as a global fallback

`a_M` is sound for every liveness query and needs no construction at all, which makes it the obvious
default. It is not the default because the constraint map beats it on every instance measured,
including the 100%-straddle one:

| model | constraint map | `a_M` |
|---|---|---|
| csma6 | **0.017 s** | 0.130 s |
| csma7 | **0.050 s** | **9.26 s** |
| grc5 | **3.38 s** | 18.3 s |
| grc4 | **0.041 s** | 0.056 s |

csma7 at **0.050 s against 9.26 s** is the decisive figure: a 185x margin on the largest csma
instance. `a_M` remains available as the correctness floor and is what UPPAAL resorts to for these
queries.

Two costs are bought with the map: grc4's *violated* queries lose ~8 ms, because the map lane does
not take phase 1's early-VIOLATED exit; and stored-state counts change, which is a visible product
difference.

### 8.5 The chaotic worklist in place of the round-robin lockstep sweep

The obvious optimisation of the backward iteration is a true chaotic worklist: `W(v)` reads only its
successors, so re-queue `v` exactly when one of them shrinks (the reverse-edge relation is
maintained for it). Measured, it is a **net loss twice over**:

* **It saves nothing where it should.** fischer7 `P0.req --> P0.wait` goes **825 891 -> 824 775**
  `Pre` calls: **0.1%**. Every node genuinely needs recomputing, because the values do propagate;
  there is no wasted sweep to remove.
* **It regresses grc4 `T0.in --> T0.far` from SATISFIED (128 states) to UNUSABLE.** Changing the
  order does not change the limit of a monotone descending iteration, but it does change the
  intermediate **federation shapes**, and the width cap is a resource bound on those. Letting some
  nodes descend far ahead of their neighbours fragments the federations enough to trip the cap at
  **5147** `Pre` calls where the lockstep sweep converges at **56 892**.

The lockstep sweep is doing real work as a de-facto widening-free normaliser. It must not be
re-optimised without measuring the cap.

### 8.6 The double-complement form of the confinement step

The published form of `Z /\ cand` is `fed_subtract(z, fed_subtract(z, cand))`, which denotes the same
set but computes it as a **double complement**: the inner subtraction peels `z` by every constraint
of every member of `cand`, so its output is an **arrangement** of `|cand|` zones — its size is not
bounded by `|cand|` at all — and the outer subtraction then peels `z` by all of those.

The intersection of two zones is a single zone (elementwise min plus a close), so `z /\ cand` needs
exactly `|cand|` pieces. The direct form is used.

The same over-eager use of subtraction appears in the `Dead = Inv` test and is replaced by the
`Inv /\ union Esc` identity (Section 2.8, **126 ms -> 0.9 ms** on grc4), and in the whole-zone
short-circuit of the deadlock test (Section 10).

### 8.7 The completion applied forward, to reachability with diagonal constraints

A network with diagonal guards falls outside every LU-family abstraction, and the completion applies
there unchanged — it is stated over constraint maps, so `Datoms` is added to whichever map the
covering test uses, with the G-simulation of Gastin–Mukherjee–Srivathsan as the natural base. The
diagonal part of the map is seeded from diagonal guards and invariants and closed under `pre`, with
the case analysis closure requires: an atom survives a reset verbatim only when neither of its
clocks is reset; reset one and it **decays** into an ordinary per-clock bound; reset both and it
becomes a constant test.

It behaves as the theory says: **correct on every instance**, and collapsing the antichain by a
factor that grows with the model — **1.04x, 1.15x, 1.43x, 1.90x** fewer stored states at
`N = 4, 5, 6, 7`, still growing.

It is nevertheless **slower — a factor of five at seven processes** — and the cause is the `2^d`
split of the covering test, not the surrounding machinery. Profiling puts the time in the Lemma 17
recursion itself, on a model where zones genuinely straddle the atoms.

**The contrast identifies when the construction is worth its cost.** Backward, splitting a zone
along an atom also narrows the federations that later pass back through it, so the split earns its
cost **twice**; forward, reachability gets no such second benefit and it earns it once. Only the
doubled version clears the bar. The question is therefore not eager against lazy, nor backward
against forward, but **whether the atoms ever change a covering decision**. Where they do not, they
are carried for soundness against a case that does not arise, and the right implementation is
whichever pays least for carrying them. Where they do, the split is real work and is charged
accordingly.

### 8.8 A whole-zone stopping term

Admitting all of `Z_v` as soon as **one** valuation is stuck over-approximates the stopping term.
Since `stop` occurs positively in the fixpoint operator, over-approximating it enlarges the greatest
fixpoint and can produce spurious witnesses — the same error direction as extrapolated zones
(Section 4), and it leaves the "non-empty fixpoint implies a witness" direction conditional.

The exact per-valuation residue is already computed inside the deadlock test — it subtracts the
union of the enabled source regions from the delay closure of `Z_v` — and is returned rather than
discarded. Measured over the 48-query set: **0 disagreements with verifyta 5.0.0, 0 verdict changes
and 0 state-count changes** against the whole-zone convention. The over-approximation was doing no
work on this benchmark set.

The whole-zone form is also **incomplete** on partial deadlocks — the grc4 `T0.near --> T0.in`
family, where the counterexample is a partial deadlock that an "adjacency is empty" test cannot see.

Two cases stay whole-zone and stay **exact**: an empty adjacency means every valuation is stuck, and
time divergence is a property of the delay-closed zone as a whole (if some clock is bounded after
delay and the invariant, no valuation can diverge; if none is, all can).

### 8.9 Per-thread local antichains in the forward passed set

Each thread inserts into its own table during a level, and a bucket owner drains them into the
shared one at the boundary, so the expand phase takes no lock at all. Correct, and consistently
**slower** at 8 threads: phil11 **1.48x**, viking12 **1.29x**, twolock10 **1.90x**, fischer10
**1.92x**. Removing the lock saves the ~15% of busy thread-time that was spin, but a private
antichain cannot see the read-only kill probe that answers 83% of successors, so the drain pays more
back than it saves (phil12: 3.2M gathered pairs, 1.5M of them exact duplicates, 19.4 s of drain
thread-time against a 3.8 s wall). Not to be retried without a design that keeps a shared kill probe.

### 8.10 Parallel bucketing of the GPU expand output

Measured three ways against the serial loop's ~200 ms: a lock-free open-addressing table with one
CAS per key and a stable counting-sort scatter — oversized table **328 ms**, `nb`-sized **467 ms**,
monotonic **336 ms**; and a private-map parallel version — **268 ms -> 241 ms** on viking15, ~1% of
wall time. The loop is bound by streaming the expand output out of unified memory while the GPU uses
the same bandwidth, not by the hashing. The serial loop stays, by measurement.

### 8.11 A fixed length threshold for the dense antichain mirror

What a dense mirror saves is an `O(L * dim^3)` unpack on **every visit** to a chain of `L` live
zones; what it costs is `L * dim^2 * 4` bytes, paid once. The payoff is driven by `visits * L`, not
by `L` alone. Tuned on a zone-heavy family (chains ~1000 long) a length-only constant lands at 64,
which never fires on process networks, where chains are short but hammered: phil13 burns **1.14e9**
instructions unpacking (11.5% of the run, 6.7x the pre-interning baseline) waiting for a threshold
it never reaches. Promotion on accumulated work, `visits * L >= threshold`, **generalises** the
length rule rather than replacing it: a length-64 chain still promotes on its first visit, a
length-16 chain on its 4th, a length-8 on its 8th.

### 8.12 A forward SCC search over the subsumed graph

The alternative to a backward fixpoint is a forward Tarjan SCC search with exact-edge
restrictions. It is a different algorithm with a different soundness argument, and it is **known
unsound under subsumption**: forward cycle detection closes loops a subsumption edge invented,
which the backward fixpoint cannot do because it only ever reads a target's value as an upper
bound. Its cost is also a function of the model rather than of the query.

Measured over 46 leads-to queries across 7 model families, symbolic states explored:

| strategy | states |
|---|---|
| on-the-fly backward fixpoint | **16 489** |
| whole-graph backward fixpoint (three sequential phases) | 99 391 |
| forward SCC search | 436 131 |

Zero disagreements among the arms on every query that all three answered; the forward SCC search
timed out at 120 s on grc4 `T0.in --> T0.far`, which the on-the-fly engine answers in 0.04 s.

A whole-graph backward fixpoint is not an independent check of the on-the-fly one: they share the
transition tagging, the exact predecessor and the federation helpers — the hard parts — and differ
only in scheduling, computing the same fixpoint over the whole graph instead of on the fly. A bug
in the shared machinery shows up identically in both, so the comparison cannot see it. What checks
the verdicts instead is Section 8.15.

### 8.13 Hit rate alone as the memo criterion

A transition memo tuned on hit rate alone accepts phil12, which sustains a **78%** hit rate and is
still **33% slower**, because its table holds 117 629 entries per thread: each entry is consulted
only ~4.6 times, so lookup and memory pressure exceed the work saved. viking15 consults each of its
933 entries ~855 times. **Reuse density** (lookups per entry) separates these cleanly where hit rate
does not, and it is dimensionless — nothing is tied to model size.

### 8.14 Most-general-first expansion order

A stored zone that a later, more general zone subsumes was expanded for nothing — on phil12,
469 805 of 1 047 142 arena nodes (**45%**) end up retired. Expanding the most general zones of a
level first should therefore let them kill their narrower siblings on insertion rather than a level
later. The generality proxy is free, since a minimal constraint system stores a more general zone in
a *shorter* packed record, and reordering a worklist cannot affect the fixpoint, so the change is
sound.

It does not pay. At 8 threads, most-general-first against insertion order:

| model | speedup | stored states |
|---|---|---|
| phil12 | 0.95x | 841 637 vs 803 845 (**worse**) |
| fischer11 | 0.98x | — |
| twolock11 | 1.00x | — |
| viking15 | 0.98x | — |

Retirement is driven by zones that arrive *levels* apart, which no intra-level ordering can reach.

### 8.15 What checks the verdicts

Since no independent second implementation of the liveness fixpoint would be independent (Section
8.12), the verdicts are checked by three things that do not share code with it:

* **`tests/models/liveness-regressions/`** — models whose correct answers are known independently
  of this engine, because each was found *by* a wrong verdict. `cex1.xml` (`P.Q --> P.Live`, NOT
  satisfied) pins the universe-versus-zero-zone distinction in the seed sweep; `cex2.xml`
  (`P.U --> P.Live`, SATISFIED) pins the frozen-blind divergence test; `cex3.xml`
  (`P.U --> P.Live`, NOT satisfied) pins the frozen gate on drop condition (iii);
  `bcast-receiver.xml` pins broadcast abstention (`E<> P.T && Q.A` is SATISFIED with the receiver's
  `cd?` guarded `y >= 5`, and is a false negative without abstention).
* **An `E[]` / `A<>` differential against `verifyta`.**
* **A recorded verdict and state-count table over the full leads-to query set**, diffed on every
  change to the engine.

Standing evidence: 0 disagreements with `verifyta` 5.0.0 over the 48-query set; the 48-query
differential unchanged in verdict and stored-node count across the abstraction changes; `A[] not
deadlock` verdicts unchanged across 7 models; unit suite 105 checks, 0 failures.

---

## 9. Atoms, straddle, and why the straddle rate does not predict cost

Call a zone `Z` **straddling** for an atom `phi` when both `Z /\ phi` and `Z /\ !phi` are non-empty.
A non-straddling zone lies wholly on one side, the atom cannot distinguish it from any other zone on
that side, and the covering test reduces to the ordinary LU instance. A straddling zone must be cut,
and with `d` atoms the split is `2^d` in the worst case — the source of the NP-completeness of the
general test.

**The covering test's cost is governed not by the number of atoms but by how many stored zones
straddle one.** The two regimes are sharply separated in practice, and along a line that is *not*
the diagonal-free/diagonal one:

* Where the atoms sit at the discrete state whose `Dead(q)` is partial, the reachable zones there
  already respect the deadlock boundary — being cut by it is what made the state partially
  deadlocked — and **no zone straddles**.
* Where atoms have instead been propagated by `pre` to states with no deadlock of their own, nothing
  aligns with them and **every zone is cut**.

Both regimes occur among diagonal-free networks. A generic estimate over random zones puts the
straddle rate at **54%**; grc4, the worst case, straddles **100%** of its tagged zones.

**The straddling regime is not the expensive one, and this is against expectation.** On the worst
case for straddling — every stored zone cut, at every tagged state — the refined search is no more
expensive than the unrefined one it replaces, and cheaper than the `a_M` fallback:

* grc4 straddles every zone and still wins: **0.041 s** against `a_M`'s **0.056 s**.
* grc4 stores **921** nodes against the `a_LU` floor of **883** — a 1.04x antichain cost under
  split-at-comparison (1.34x under split-at-creation) — and the refined search still costs the same.
* grc5, the largest instance, runs in **3.38 s** where plain LU subsumption, which is unsound there,
  takes **5.82 s**: the refined covering is cheaper than the **coarser abstraction it refines**.

The `2^d` bound is real but is a worst case over atoms that genuinely separate zones; the atoms that
survive the drop conditions nearly all do separate, so the split does work rather than waste it. In
the backward setting it can pay for itself twice: cutting a zone along an atom of `Dead(q)` also
narrows the federations that later pass back through it.

**The practical consequence is that atom count and straddle rate are the wrong things to tune
against.** What dominated measurement until the phases were separated was neither: it was computing
`Inv(q) \ Dead(q)` by subtracting the residue — an arrangement of zones — where the identity
`Inv(q) \ Dead(q) = Inv(q) /\ union Esc(q)` gives it in one intersection per outgoing transition
(Section 2.8).

---

## 10. Supporting invariants

### 10.1 Frozen (committed and urgent) locations do not delay

At a committed or urgent location no positive delay is legal. Three places must consult the frozen
flags, and each has a distinct failure mode if it does not:

* **The divergence test.** Delaying unconditionally reports divergence at a frozen location,
  inventing a maximal-run witness out of nothing. Regression: `cex2.xml` (`P.U --> P.Live`, correct
  answer SATISFIED).
* **Drop condition (iii) of the seed** (Section 2.4). Regression: `cex3.xml`.
* **The deadlock test** and the exact predecessor, which omit the inner time-predecessor when the
  target is frozen and the outer one when the source is frozen.

The divergence and drop-condition cases are **coupled**: repairing the divergence test alone makes
`cex3` wrong, and gating (iii) alone leaves `cex2` wrong. They must hold together.

### 10.2 Broadcast receivers may abstain

A broadcast `a!` fires whether or not any receiver is enabled, and each `a?` takes part exactly when
its **own** guard holds. Collecting receivers by their **int** guard alone and then conjoining their
**clock** guards forces every int-enabled receiver to participate: the run where a receiver's clock
guard fails and it stays put is never generated. Reachable states are lost — an `E<>` false negative
and, worse, an `A[]` false **positive**. Inside the deadlock test the same defect over-approximates
the escape set and can therefore **invent** deadlocks.

Each receiver process gets an explicit **abstain** option guarded by the complement of the
disjunction of its edges' clock guards. Each clock guard is a conjunction, so its complement is a
disjunction of negated atoms, and the complement of the whole disjunction is the cross product of
those — a DNF whose disjuncts are each convex, hence each a separate option. Sizes are the product
of the guard lengths, i.e. tiny in practice.

**The collapse case is what makes it safe:** when any receiver edge has an empty clock guard the
complement of `true` is false, no abstain option exists, and the option list is byte-identical to a
version without abstention. csma is exactly that case — its five `cd?` edges carry no guard — so
`A[] not deadlock` verdicts and state counts are unchanged across 11 models (csma4/5/6 at
202 / 1271 / 6259 on the deterministic sequential backend).

Backward, an abstaining receiver contributes no edge, so it cannot be named in a transition tag at
all. Its guard complement rides in a separate field of the tag. It is a **pure guard on the source
valuation** — no reset, no location change, no integer update — so the predecessor applies it at
exactly the point it applies the participants' guards: after the resets have been freed, before the
pre-action delay, and nowhere else.

One copy of this logic drifting from another is what produced the defect, so the transition
semantics both engines are built out of live in a single header, `inline`, with each translation
unit getting its own inlinable copy.

### 10.3 The convex-union merge is exact, not a hull

The union operation succeeds **only** when the union of the two zones really is a single zone, so
the stored zone afterwards denotes exactly the same set of valuations as the two it replaced: no
point is added and none is removed. That is what makes it usable in the **under-approximating** lane
as well — unlike a convex hull, it is not an over-approximation, so it cannot invent a certificate.

It is load-bearing rather than incidental. Without it the federations of the backward pass grow
monotonically with every `Pre` and the width cap becomes the binding constraint on the whole
algorithm. On the forward side it collapses antichains: grc4 stores **14.5 zones per discrete key**
in the liveness graph against **1.00–1.44** on every other family — the shape of an antichain full
of zones whose union happens to be convex.

Federation compaction drops empty members, drops members already covered by one that is kept, and
merges any pair whose union is exactly convex. One settling pass follows, because a merge can make
an earlier member redundant against a later one.

The forward passed set bounds its merge scan to the newest `kMergeScan = 32` live zones. Merging is
an optimisation, never a requirement — skipping one only leaves two zones where one would do — so a
cap is sound, and the whole-chain scan is quadratic exactly where chains are long. Zones that merge
are nearly always recent neighbours, so a small window keeps effectively all of the reduction.
Removals from the live list use an **ordered** erase, never swap-with-last: the convex union is
order-sensitive and the survivors must keep their newest-first relative order.

### 10.4 The federation width cap and UNUSABLE

The backward federations are capped at `fedcap = 4096` members, and the iteration is additionally
bounded at 100 000 rounds.

**A capped pass has not converged, so it proves nothing in either direction.** Returning "no
certificate" from a capped pass would launder "ran out of federation budget" into a silent
SATISFIED — the worst possible failure mode for a verification tool. The cap flag is therefore
**sticky**, is never cleared, and the caller must report UNUSABLE rather than a verdict. A federation
whose cap flag is set must not be read as a converged value.

Certificates are only meaningful at the **limit**: intermediate iterates are pre-fixpoints and say
nothing, which is why the iteration runs to fixpoint before anything is inspected.

### 10.5 The active-clock reduction

Clocks that no process will read before resetting, and that the query does not read, are pinned to
`0`, which collapses zones differing only in dead clocks. Two consequences elsewhere:

* it leaves the stored zone **not delay-closed**, which the deadlock test repairs by rebuilding the
  delay closure and re-imposing the real invariant before testing (Section 4);
* it must be **undone at the target** by the exact predecessor, or `Pre` demands dead clocks read
  `0` at the source too (Section 6).

Every clock the constraint map reads is added to the query mask so the reduction cannot erase it
(Section 2.5). The discrete analogue — canonicalising integer cells no process can read before
overwriting and the query does not mention — is applied at the same point in the successor path.

### 10.6 The deadlock test: exactness and its short circuits

The test mirrors all three transition kinds and, for each transition that *can* fire, collects its
**source** zone within `Z` — the pre-reset valuations from which it fires and lands in a legal
target (`guards /\ reset^-1(target invariant)`). Inverting a reset in a target invariant is the index
substitution `c -> 0` for reset clocks; a constraint that reduces to `0 <| b` is a constant gate on
the whole transition. The union of the time-pasts of those source zones is the set of non-deadlocked
valuations (no time-past for a frozen source). Anything left over is a set of deadlock states.

Two exact short circuits, both replacing a subtraction whose output is an arrangement:

* **Inclusion-minimal escapes, not append.** The escape set is consumed only through its **union**,
  so a member contained in a sibling denotes nothing new — but it is not free: the subtraction peels
  the remainder once per member and the piece count multiplies, so a redundant escape costs a whole
  extra peel of an already-exploded arrangement. Two escapes agreeing on their clock guard and
  differing only in which process moved is the common case (grc4's trains). Pruning **before** the
  time-past is safe: `W subseteq W'` implies `down(W) subseteq down(W')`, so the union of the
  time-pasts is unchanged.
* **One-escape short circuit.** The subtraction is built even when a **single** escape already
  covers `Z`, which is the overwhelmingly common case (a location with one unguarded outgoing edge).
  One DBM inclusion test per escape decides that outright. Exact, not a heuristic:
  `Z subseteq W subseteq union(live)` means `Dead(q)` is empty, which is precisely what the
  subtraction would have reported.

### 10.7 State-count accounting

`verifyta`'s leads-to counters report only the outer reachability up to the first violating
`p`-state; the nested inevitability check is not counted. This engine interns every state including
the backward pass's, so the two are tallied apart and reported separately.

`verifyta` also does **not** count the initial state. Its `E<> p` figures are exactly this engine's
count minus one on every query measured (5 against 6, 6 against 7, 9 against 10), which pins the
convention: it reports the states *generated*, not the root it started from.

### 10.8 Adaptive extrapolation on the forward side

`a_LU` and `Extra_LU+` are **not** interchangeable: `a_LU` only ever **prunes** — it drops a zone
when a stored one simulates it — whereas `Extra_LU+` **widens**, mapping many exact zones onto one
representative. On a model whose reachable zones are mutually `a_LU`-incomparable, pruning does
nothing and widening collapses them.

Which of the two wins is a property of the model, and the engine already measures the distinguishing
signal. Zones per discrete state near 1 means `a_LU` is collapsing them and widening would only cost
time (phil 1.11, viking 1.00: extrapolation there changes no state count and costs 20–35%). A ratio
that climbs means the antichains are filling with mutually incomparable zones, which `a_LU` cannot
prune and widening can (one location, thousands of zones: widening takes it from 3873 stored to
3676, which is `verifyta`'s 3673). The search therefore starts exact and switches on once the ratio
crosses a threshold of 4.0.

Switching mid-run is sound in both directions of the mixed passed list it leaves behind: a stored
**exact** zone that fails to subsume a new widened one only costs an extra stored state, and a
stored **widened** zone that subsumes a new exact one genuinely contains it. Finiteness is
guaranteed from the switch onwards. This is also why the deadlock path collapses the bound arrays
rather than threading a flag (Section 3): the adaptive switch would otherwise turn on an
extrapolation that is unsound for deadlock.

---

## 11. Complexity

Let `|G| = |V| + |E|` and let `H` bound the number of strict decreases of a fixpoint component.

**Theorem.** The on-the-fly algorithm terminates in `O(H |G|)` time. For a fixed clock set and
maximum constant, it runs in `O(|G|)` time. *Proof.* Whenever `W_v` decreases, only predecessors of
`v` need reconsidering. Since each component decreases at most `H` times, every edge is processed at
most `H` times, so the backward computation costs `O(H |E|)`; together with the `O(|G|)` forward
exploration the algorithm runs in `O(H |G|)`. Each `W_v` is a union of regions contained in `Z_v`;
for a fixed clock set and maximum constant the number of regions, and hence `H`, are constant.
Finiteness of the region domain also guarantees termination.

---

## 12. Novelty and citations

No deadlock-preserving covering preorder for timed automata coarser than `M`-extrapolation appears
to be published.

* **Behrmann, Bouyer, Larsen, Pelanek**, *Lower and upper bounds in zone-based abstractions of timed
  automata*, STTT 8(3):204–215, 2006. Lemma 1 (`=_M` is a bisimulation), Lemma 3 (`<=_LU` is only a
  simulation), and the "Simulation & Bisimulation" prose of Section 3 on deadlock. This is the
  source of the `a_M` fallback that the completion beats, and of the reason UPPAAL abandons the LU
  abstraction for deadlock and liveness queries.
* **Bouyer**, 2004. Diagonal constraints make LU-family abstractions unsound; diagonal networks keep
  exact covering.
* **Herbreteau, Srivathsan, Walukiewicz**, Inf. Comput. 2016 (arXiv:1110.3705), **Thm 14**:
  `a_{<=LU}` is the *coarsest* abstraction sound and complete for reachability among automata with
  the same LU bounds. This is why any deadlock-preserving gain must key on structure other than LU
  bounds, as the completion does. The same line contributes the lazy-abstraction idea evaluated in
  Section 8.1.
* **Gastin, Mukherjee, Srivathsan**, CAV 2019, **Lemma 17**: the covering test for constraint-map
  simulations, and the `2^d` split it implies. Adapted here to an implication obligation rather than
  to agreement. The G-simulation line never mentions deadlock; since `<=_G` is likewise only a
  simulation, it inherits the deadlock obstruction verbatim and is repaired by the same completion,
  so the diagonal-free and diagonal cases are unified rather than separate.
* **Mukherjee**, thesis, 2022. Cor. 3.6/3.10 collapse a diagonal-free constraint map to one upper
  and one lower bound per clock; Cor. 3.6's "weakest wins" rule is the compositional union used when
  combining per-location bounds over processes.
* **Tripakis**, 2009, and **Laarman et al.**, CAV 2013. Forward cycle detection over a subsumed zone
  graph is unsound; the backward fixpoint is what removes that failure mode.

The completion is stated over **constraint maps**, so `Datoms` is added to whichever map the covering
test uses. For diagonal-free networks the base map is the LU instance; for diagonal networks the
G-simulation is the natural base, and its maps are produced by the same backward fixpoint that yields
the LU bounds.

---

## 13. Scope and known limitations

* **Diagonal networks** fall outside every LU-family abstraction and are handled with exact covering
  throughout. This is an implementation restriction and not a limitation of the completion: the
  construction applies unchanged over the G-simulation's constraint map, and was measured to be
  correct at every size there while costing a factor of five at seven processes (Section 8.7).
* **The atom cap.** A constraint map needing more than 64 atoms at one key, or a network with 63 or
  more clocks, falls back to the gate and race (Section 7), which remains correct.
* **Frozen keys and the structural sweep.** The deadlock test skips the invariant at frozen keys, so
  it over-reports on the structural sweep. This is neutralised by the seed drop conditions, which do
  not fire on the over-reported keys.
* **Leads-to predicates carry no clock constraints.** Clock constraints in `p` or `q` would require
  zone splitting and are rejected at the query layer.
* **Federation width.** A run that exhausts the federation budget reports UNUSABLE, never a verdict
  (Section 10.4).
