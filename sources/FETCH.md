# Reading list — environment abstraction / compositional verification for timed automata

Backing material for **lead 1** in `../leads/main.tex` and section 1 of
`../docs/superpowers/specs/new-reduction-theorem.md`.

Four PDFs are here. The rest are behind Springer LNCS / IEEE / Elsevier paywalls, so only the
identifiers are recorded — fetch them through an institutional login.

## In this folder

| file | paper |
|---|---|
| `alur-dill-1994-theory-of-TA.pdf` | Alur & Dill, *A Theory of Timed Automata*, TCS 126(2):183–235, 1994. The wall: timed language inclusion is undecidable for nondeterministic TA. Every automation attempt below is a way around this one result. |
| `sankur-2025-TA-verification-via-FA-learning.pdf` | Sankur, *Timed Automata Verification and Synthesis via Finite Automata Learning*, JAR 2025 (prelim. TACAS 2023). Dodges undecidability by learning a DFA over the *untimed* language. |
| `bouyer-2022-zone-based-verification-survey.pdf` | Bouyer, Gastin, Herbreteau, Sankur, Srivathsan, *Zone-based verification of timed automata: extrapolations, simulations and what next?*, arXiv:2207.07479. Not about compositionality, but section 8 is the source of the framing this whole programme rests on: the bottleneck is the number of control states. |
| `herbreteau-2013-lazy-abstractions.pdf` | Herbreteau, Srivathsan, Walukiewicz, *Lazy Abstractions for Timed Automata*, CAV 2013. The refinement direction that is incompatible with a concurrently-read antichain — useful as the contrast to the contraction idea. |

## Not open access — read the first one first

**START HERE.** Kristoffersen, Laroussinie, Larsen, Pettersson, Yi, *A Compositional Proof of a
Real-Time Mutual Exclusion Protocol*, **TAPSOFT 1997**, LNCS 1214. Fischer at N=50 in 172 s. It is
our model, it is the result lead 1 would reproduce, and it says in its own words what stopped the
line: the decomposition order was hand-chosen.

Theory underneath it:
- Andersen, *Partial Model Checking*, **LICS 1995**. `A ∥ B ⊨ φ ⟺ A ⊨ φ / B`.
- Laroussinie & Larsen, **CONCUR 1995**; tool paper *CMC*, FORTE/PSTV 1998.

Soundness foundation — this is what makes lead 1 a theorem:
- Berendsen & Vaandrager, *Compositional Abstraction in Real-Time Model Checking*, **FORMATS 2008**,
  LNCS 5215, pp. 233–249. Simulation is a **precongruence** for UPPAAL composition, including
  shared variables and committed locations. <https://doi.org/10.1007/978-3-540-85778-5_17>

Automation attempts (where the field is now):
- Lin, André, Liu, Sun, Dong, *Learning Assumptions for Compositional Verification of Timed
  Systems*, **IEEE TSE 40(2):137–153, 2014**. FMS-3 480 481 → 2 265 states (212×); AIP 104 651 →
  2 301 (45×). <https://doi.org/10.1109/TSE.2013.57>
- Chen, Su, Zhang, Liu, Mi, *Learning Assumptions for Compositional Verification of Timed
  Automata*, **CAV 2023**, LNCS 13964:40–61. Deterministic one-clock assumptions.
- Cobleigh, Giannakopoulou, Păsăreanu, *Learning Assumptions for Compositional Verification*,
  **TACAS 2003**; journal version Păsăreanu et al., *FMSD* 32(3):175–205, 2008. The untimed
  ancestor, L\* with the model checker as Teacher.

Limits:
- Maier, *Compositional Circular Assume-Guarantee Rules Cannot Be Sound and Complete*,
  **FoSSaCS 2003**, LNCS 2620:343–357.

Parameterized branch — the sharpest counterpoint, settles Fischer for *all* N in 2.12 s:
- Abdulla, Deneux, Mahata, Nylén, *Using Forward Reachability Analysis for Verification of Timed
  Petri Nets*, **Nordic J. Computing 14(1–2):1–42, 2007**.
- Abdulla & Jonsson, *Model checking of systems with many identical timed processes*,
  **TCS 290(1):241–264, 2003** (conf. TACAS 1998). Decidable for unboundedly many identical
  ONE-CLOCK processes; undecidable with a controller and several clocks each.
