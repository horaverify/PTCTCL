# The composition theorem, and why nobody computes the environment automatically

A study note for lead 1. Papers in `PAPERS.md` §1.

---

## 1. The rule

Split the system into the part the property talks about and the rest:

```
S  =  C ∥ E
```

`C` holds every component named in `φ`, plus anything sharing state with them. `E` is everything
else. Now replace `E` by some `Ê`:

> **Theorem.** If `E ⊑ Ê` (a simulation on the shared interface) then
> `C ∥ Ê ⊨ φ  ⟹  C ∥ E ⊨ φ`.

Two ingredients, and only the second is deep.

**(a) Simulation transfers safety.** If `Ê` can do everything `E` can, every run of `C ∥ E` has a
matching run of `C ∥ Ê`. A safety property is a statement that no run reaches a bad state, so if
the bigger behaviour set has no bad run, neither does the smaller one.

**(b) Simulation survives composition.** We need `E ⊑ Ê` to imply `C ∥ E ⊑ C ∥ Ê`. A relation with
that property is a **precongruence**, and it is not automatic — it has to be proved for the
specific composition operator. Berendsen & Vaandrager (FORMATS 2008) prove it for UPPAAL's
operator, *including shared variables and committed locations*, which is the version we need. The
one essential restriction they impose is that guards on input transitions must not read external
variables.

**The direction is one-way.** `Ê` may invent behaviour, so `C ∥ Ê` can be unsafe while `C ∥ E` is
safe. A "safe" answer is trustworthy; an "unsafe" answer means you learned nothing and must fall
back to the real system. That asymmetry is what makes the technique cheap to *try* — and is the
whole argument for racing it against the exact search (lead 8).

## 2. What `Ê` looks like when it works

For Fischer with the query `A[] not (P1.cs ∧ P2.cs)`:

- `C` = `P1`, `P2`, and the shared variable `id`.
- `E` = the other `N−2` processes.

Those processes touch `C` **only** through `id`. And `P1` cannot tell *which* of them holds it —
only whether `id` is `0`, is its own, or is somebody else's. So `Ê` is one generic process that
claims `id` for "someone else" and releases it, under the same timing constraints: about
**3 locations and 1 clock**, versus `N−2` processes and `N−2` clocks.

Measured against our engine: **256 discrete states with 3 clocks, versus 260 998 with 10** — and
the abstract system does not grow with `N` at all.

Abdulla et al. (NJC 2007) use exactly this value abstraction — a place for "undefined" plus primed
copies marking *the process whose id is currently held* — and settle Fischer for **all** `N` in
2.12 s.

## 3. Why it is built by hand — the part worth understanding

The obvious thought is: `E` is a timed automaton, weak timed simulation/bisimulation is
decidable, so **compute the quotient**. Minimise `E` and use that as `Ê`. Nobody does this. Four
reasons, and the second is the real one.

### 3.1 Quotienting `E` needs `E`

The quotient of the environment is taken over the environment's state space, and `E` is `N−2`
processes composed — still exponential. You have to minimise *per component, before composing*
(compositional minimisation), which is a real technique but brings its own problem: a component
minimised **in isolation** keeps all the behaviour its context would never allow, so the isolated
quotient is often no smaller. The standard remedy is to minimise *modulo an interface* that
constrains the context (Graf–Steffen; Krimm–Mounier) — and now you need the interface, which is
the thing you were trying to construct.

### 3.2 A quotient preserves too much — this is the crux

**Quotienting merges states that are already equivalent. The win here comes from merging states
that are NOT equivalent.**

The 3-location `Ê` above is *not* bisimilar to `E`, and it is not simulation-equivalent either.
Eight processes with eight independent clocks genuinely have more behaviour than one generic
process: they can hold `id` in orders and at times the single process cannot reproduce. `Ê`
simulates `E` — it is *coarser* — and that coarsening throws away information the property cannot
observe.

No quotient construction can produce it, because a quotient is by definition behaviour-preserving.
What produces it is **abstraction**: deciding what the property cannot see and collapsing along
that. That decision requires knowing what `φ` can distinguish, which is exactly the insight a human
supplies, and exactly why CEGAR exists and is hard.

So: minimisation gives you the *smallest automaton with the same behaviour*. Environment
abstraction needs the *smallest automaton with at least the behaviour*. The second is unboundedly
smaller and is not a quotient of anything.

### 3.3 Simulation quotients are worse-behaved than bisimulation ones

Bisimulation has a canonical minimal quotient. Simulation preorders do not, in the same clean way,
and for timed automata deciding simulation is already costly (federation-based partition
refinement). So even the weaker prize costs real time.

### 3.4 The undecidability wall shapes everything

Timed **language inclusion** is undecidable for nondeterministic timed automata (Alur & Dill, TCS
1994). So "is `Ê` permissive enough" cannot be asked as a language question at all — everyone uses
**simulation** instead, which is decidable and stronger than inclusion. That is why the whole
field is phrased in simulations and refinements rather than in languages, and it is why ECDAR and
the Timed I/O Automata line define refinement as alternating timed simulation.

## 4. What the field did instead: learn the assumption

Since `Ê` cannot be computed by minimisation, the modern branch **learns** it. `L*` learns the
weakest assumption over the interface alphabet with the model checker as Teacher (Cobleigh,
Giannakopoulou, Păsăreanu, TACAS 2003); the timed version learns a deterministic event-recording
automaton (Lin et al., IEEE TSE 2014), later a deterministic one-clock TA (Chen et al., CAV 2023),
or a DFA over the untimed language (Sankur, JAR 2025).

Learning produces a *lossy* automaton, which is the right shape. Reported reductions are real:
FMS-3 480 481 → 2 265 states (≈212×).

The catch is what they must restrict to stay decidable — event-recording automata, or one clock,
or clock-free — and whether our models fall inside any of those classes is the open question for
lead 1.

## 5. What this means for the experiment now running

The `WeakSim` quotient experiment tests §3.1/§3.2, and it is worth doing, but its expected outcome
should be calibrated:

- If the quotient **collapses well**, compositional minimisation gives a cheap partial `Ê` and the
  lead is easier than feared.
- If it **does not collapse**, that does **not** kill lead 1. It confirms §3.2 — that the useful
  abstraction is lossy and not a quotient — and redirects the effort to the learning branch or to a
  hand-designed generic environment with a machine-checked simulation proof.

The second outcome is the more likely one, and it is still informative: it tells us the missing
piece is *constructing* a lossy `Ê`, while the *checking* of `E ⊑ Ê` is already solved by the
simulation checker in this repo.
