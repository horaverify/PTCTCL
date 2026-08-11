# How to build `Ê` automatically

Follow-on to `composition-theory.md`. That note explains the theorem and why `Ê` cannot be obtained
by *minimising* the environment — minimisation preserves behaviour, and the whole win comes from
losing behaviour the property cannot see. This note is about the remaining question:

> **`Ê` has to be invented. Can a machine invent it?**

Nothing here is implemented. It is a menu of routes, ranked by how well they fit *these* models,
with the soundness obligation each one carries.

**One fact that makes all of this cheaper than it sounds:** the *checking* half is already solved.
Deciding `E ⊑ Ê` is a timed simulation check, and this repo has one (`include/ptctlmc/bisim.hpp`,
`BisimKind::WeakSim`, greatest-fixpoint partition refinement over federations). It needs invariant
support added. So every route below only has to *propose* an `Ê`; a wrong proposal is caught
mechanically, not by a human proof.

---

## Route A — generic replica plus value abstraction (best fit here)

**The construction.** If the environment is `k` copies of one template `T` sharing a variable `v`:

1. Take **one** copy of `T`.
2. Abstract `v`'s domain to what `C` can actually distinguish. Collect the constants `C` compares
   `v` against — for Fischer, `P1` and `P2` test `v == 0` and `v == own id` — and quotient the
   domain into `{0, mine, someone-else's}`.
3. Let the single copy write "someone-else's" freely, subject to `T`'s own timing constraints.

That is the 3-location, 1-clock `Ê`. Abdulla et al. (NJC 2007) use exactly this value abstraction
and settle Fischer for all `N`.

**Why it is automatable.** Every step is syntactic: detect that the environment is `k` instances of
one template (the generators emit exactly that, and it is checkable on an arbitrary input by
comparing template structure); read the comparison constants out of `C`'s guards; emit the
abstracted copy. No insight about the *property* is needed beyond which constants `C` mentions.

**Soundness obligation.** `E ⊑ Ê` must hold. Two ways: prove it once, generically, for
"`k` copies ⊑ one permissive copy" (a small paper-sized lemma); or just run the simulation checker
on each instance and let it fail loudly. The second is available today.

**Why it is not symmetry reduction.** Symmetry explores one representative per orbit and is exact.
This replaces `k` components by one *more permissive* component and is an over-approximation, so
the answer is one-way. Different mechanism, different guarantee — and it is not what UPPAAL's
`scalar` types do.

**Risk.** The single copy may be *too* permissive: it can hold `v` at times the real environment
never could, producing a spurious counterexample. Then the abstract check says "unsafe", we learn
nothing, and we fall back. That is Route B.

---

## Route B — refine by adding replicas, not by refining predicates

**The loop.** Try `Ê` = 1 generic replica. If the abstract system is safe, done — answer holds for
the real system. If it reports a violation, that violation may be spurious, so try `Ê` = 2 generic
replicas, then 3, and so on, up to `k = N`, at which point the abstraction *is* the real system and
the answer is exact.

**Why this shape is attractive.** It is abstraction refinement without the trap this project has
already measured. The TCTLSAT CEGAR experience was that coarsening cost *more* nodes, because the
finer structure was also acting as a canonicaliser — here nothing about the zones is coarsened, only
the number of components is reduced, so merge behaviour is untouched.

Each level is an over-approximation of the real system, so **every level is sound**, and the loop
terminates by construction at `k = N`. Small `k` is cheap, so a wrong guess costs little.

**Open question.** Does `k` stay small in practice? For Fischer the folklore answer is 2 — mutual
exclusion between two processes needs one other claimant to be interesting. If `k` is 2 or 3
regardless of `N`, this is the whole lead. If `k` grows with `N`, it degenerates to the full system
and buys nothing. **This is the cheapest thing to measure and should be measured first.**

---

## Route C — learn the assumption (what the field actually does)

`L*` learns the weakest assumption over the interface alphabet with the model checker as Teacher
(Cobleigh, Giannakopoulou, Păsăreanu, TACAS 2003). The timed descendants learn a deterministic
event-recording automaton (Lin et al., IEEE TSE 2014), a deterministic one-clock TA (Chen et al.,
CAV 2023), or a DFA over the *untimed* language (Sankur, JAR 2025).

**Numbers are real:** FMS-3 480 481 → 2 265 states (≈212×); AIP 104 651 → 2 301 (≈45×).

**The obstruction, and it is specific.** Each of these restricts to a class where timed inclusion
is decidable — event-recording, or one clock, or clock-free — because for general nondeterministic
timed automata inclusion is undecidable (Alur & Dill, TCS 1994). Our models have shared bounded
integers and CSP-style binary/broadcast channels, and whether they fall inside any of those classes
is **unresolved**. That is the question to answer before spending on this route.

---

## Route D — interface-constrained minimisation

Minimise `E` *modulo an interface* that restricts what the context can do (Graf–Steffen;
Krimm–Mounier), rather than in isolation, where a component keeps behaviour its context would never
allow.

**The catch is circular:** you need the interface to do the minimisation, and the interface is
close to the thing you were constructing. It can be broken by starting with a trivially permissive
interface ("`C` may do anything") and tightening, but each tightening is itself an abstraction
choice. Worth knowing about; not the first thing to try.

---

## Recommended order

1. **Measure Route B's `k`.** Build `Ê` for `k = 1, 2, 3` by hand for Fischer — a morning's work,
   since the templates are generated by a script we control — and check at what `k` the abstract
   system becomes safe. If `k` is small and independent of `N`, everything else follows.
2. **Automate Route A** for replicated networks, with the simulation checker as the safety net.
3. **Settle whether our models fit any decidable class** before touching Route C.
4. Route D only if 1–3 stall.

## What would kill the whole idea

If the abstract system reports a violation at every `k < N` — i.e. the property genuinely depends on
all `N` processes — then no environment abstraction helps and the lead is closed. For a mutual
exclusion property that would be surprising, but it is exactly what step 1 measures, and it is a
one-morning experiment rather than an argument.
