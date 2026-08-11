# Papers cited across this work

Checklist of everything referenced in `../leads/`, `../paper/section/related.tex` and
`../docs/superpowers/specs/`. **have** = PDF is in this folder.

---

## 1. Environment abstraction / compositional verification — lead 1

| have | paper |
|---|---|
| | **Kristoffersen, Laroussinie, Larsen, Pettersson, Yi.** *A Compositional Proof of a Real-Time Mutual Exclusion Protocol.* TAPSOFT 1997, LNCS 1214. — **read this first**: Fischer at N=50 in 172 s; it is our exact model, and it names its own blocker (hand-chosen decomposition order). |
| | **Andersen.** *Partial Model Checking.* LICS 1995. — the origin: `A ∥ B ⊨ φ ⟺ A ⊨ φ / B`. |
| | **Laroussinie, Larsen.** CONCUR 1995 — timed version. Tool paper: *CMC*, FORTE/PSTV 1998. |
| | **Berendsen, Vaandrager.** *Compositional Abstraction in Real-Time Model Checking.* FORMATS 2008, LNCS 5215:233–249. doi:10.1007/978-3-540-85778-5_17 — simulation is a **precongruence** for UPPAAL composition (shared variables, committed locations). This is what makes lead 1 a theorem. |
| | **Lin, André, Liu, Sun, Dong.** *Learning Assumptions for Compositional Verification of Timed Systems.* IEEE TSE 40(2):137–153, 2014. doi:10.1109/TSE.2013.57 — the serious automated attempt; FMS-3 480 481 → 2 265 states (212×). |
| | **Chen, Su, Zhang, Liu, Mi.** *Learning Assumptions for Compositional Verification of Timed Automata.* CAV 2023, LNCS 13964:40–61. |
| | **Cobleigh, Giannakopoulou, Păsăreanu.** *Learning Assumptions for Compositional Verification.* TACAS 2003; journal: Păsăreanu et al., FMSD 32(3):175–205, 2008. — untimed ancestor, L\*. |
| | **Maier.** *Compositional Circular Assume-Guarantee Rules Cannot Be Sound and Complete.* FoSSaCS 2003, LNCS 2620:343–357. |
| | **Abdulla, Deneux, Mahata, Nylén.** *Using Forward Reachability Analysis for Verification of Timed Petri Nets.* Nordic J. Computing 14(1–2):1–42, 2007. — parameterized Fischer, **all N**, 2.12 s. |
| | **Abdulla, Jonsson.** *Model checking of systems with many identical timed processes.* TCS 290(1):241–264, 2003 (conf. TACAS 1998). — decidable for many identical **one-clock** processes; undecidable with a controller and several clocks each. |

## 2. Zone abstractions and simulations

| have | paper |
|---|---|
| ✅ | **Bouyer, Gastin, Herbreteau, Sankur, Srivathsan.** *Zone-based verification of timed automata: extrapolations, simulations and what next?* arXiv:2207.07479, 2022. — §8 is the source of this programme's framing. |
| ✅ | **Herbreteau, Srivathsan, Walukiewicz.** *Lazy Abstractions for Timed Automata.* CAV 2013. arXiv:1301.3127. |
| | **Herbreteau, Srivathsan, Walukiewicz.** *Better abstractions for timed automata.* LICS 2012. arXiv:1110.3705 — the `a_LU` simulation the engine uses. |
| | **Gastin, Mukherjee, Srivathsan.** *Fast algorithms for handling diagonal constraints in timed automata.* CAV 2019. arXiv:1904.08590 — the G-simulation. |
| | **Gastin, Mukherjee, Srivathsan.** *Reachability in timed automata with diagonal constraints.* CONCUR 2018. arXiv:1806.11007. |
| | **Behrmann, Bouyer, Larsen, Pelánek.** *Lower and upper bounds in zone-based abstractions of timed automata.* (LU bounds.) |
| | **Daws, Tripakis.** Extrapolation / model checking of real-time reachability, 1998. |
| | **Larsen, Larsson, Pettersson, Yi.** *Efficient verification of real-time systems: compact data structure and state-space reduction.* RTSS 1997 — minimal constraint systems. |

## 3. Liveness, Büchi emptiness, and subsumption

| have | paper |
|---|---|
| | **Tripakis.** *Checking timed Büchi automata emptiness on simulation graphs.* ACM TOCL 10(3), 2009. — subsumption is a **safety-only** licence. |
| | **Li (Guangyuan).** *Checking timed Büchi automata emptiness using LU-abstractions.* FORMATS 2009, LNCS 5813. |
| | **Herbreteau, Srivathsan, Walukiewicz.** *Efficient emptiness check for timed Büchi automata.* CAV 2010; journal FMSD 40(2):122–146, 2012. — **liveness-compatible subsumption**: no cycle carrying both an accepting node and a subsumption edge. Directly relevant to the current bug. |
| | **Herbreteau, Srivathsan.** *Efficient on-the-fly emptiness check for timed Büchi automata.* ATVA 2010. |
| | **Herbreteau, Srivathsan, Tran, Walukiewicz.** *Why liveness for timed automata is hard, and what we can do about it.* FSTTCS 2016, LIPIcs 65:48:1–48:14. — liveness is strictly harder than reachability unless P=NP. |
| | **Laarman, Olesen, Dalsgaard, Larsen, van de Pol.** *Multi-core Emptiness Checking of Timed Büchi Automata Using Inclusion Abstraction.* CAV 2013, LNCS 8044:968–983. — the multi-core version. |

## 4. Parallel and distributed model checking

| have | paper |
|---|---|
| | **Behrmann, Hune, Vaandrager.** *Distributing Timed Model Checking — How the Search Order Matters.* CAV 2000, LNCS 1855:216–231. doi:10.1007/10722167_19 — parallelising a subsumption search makes state counts **grow**: Dacapo 45 001 → 110 634 on 17 nodes. |
| | **Braberman, Olivero, Schapachnik.** *Issues in distributed timed model checking — building Zeus.* STTT 7(1):4–18, 2005; and *Dealing with practical limitations…*, FMSD 29(2):197–214, 2006. |
| | **Stern, Dill.** Parallelizing the Murφ verifier, 1997 — the original partition-by-hash. |
| | **Dalsgaard, Laarman, Larsen, Olesen, van de Pol.** *Multi-Core Reachability for Timed Automata.* FORMATS 2012. |
| | **Laarman, van de Pol, Weber.** *Boosting Multi-Core Reachability Performance with Shared Hash Tables.* FMCAD 2010. |
| | **Barnat, Brim, Ročkai.** *DiVinE 2.0.* — OWCTY and MAP: DFS-free parallel LTL. |
| | **Blelloch, Fineman, Gibbons, Shun.** *Internally deterministic parallel algorithms can be fast.* PPoPP 2012. |

## 5. Partial-order reduction and interleaving

| have | paper |
|---|---|
| | **Bengtsson, Jonsson, Lilius, Yi.** *Partial order reductions for timed systems.* CONCUR 1998 — local-time semantics. |
| | **Govind, Herbreteau, Srivathsan, Walukiewicz.** *Abstractions for the local-time semantics of timed automata: a foundation for partial-order methods.* LICS 2022. hal-03644039 |
| | **Govind, Herbreteau, Srivathsan, Walukiewicz.** *Revisiting Local Time Semantics for Networks of Timed Automata.* CONCUR 2019, LIPIcs 140. |
| | **Hansen, Lin, Liu, Nguyen, Sun.** *Diamonds are a girl's best friend: partial order reduction for timed automata with abstractions.* CAV 2014. |
| | **Ben Salah, Bozga, Maler.** *On Interleaving in Timed Automata.* CONCUR 2006. — union of zones from different interleavings of the same transition set is convex; the general form of this engine's convex-union merge. |
| | **Lugiez, Niebert, Zennou.** *A partial order semantics approach to the clock explosion problem of timed automata.* TACAS 2004; TCS 345 — event zones. |

## 6. Other techniques referenced

| have | paper |
|---|---|
| ✅ | **Alur, Dill.** *A Theory of Timed Automata.* TCS 126(2):183–235, 1994. — timed language inclusion undecidable. |
| ✅ | **Sankur.** *Timed Automata Verification and Synthesis via Finite Automata Learning.* JAR 2025 (prelim. TACAS 2023). arXiv:2211.13462 |
| | **De Wulf, Doyen, Henzinger, Raskin.** *Antichains: a new algorithm for checking universality of finite automata.* CAV 2006. |
| | **Christensen, Kristensen, Mailund.** *A sweep-line method for state space exploration.* TACAS 2001. — proved vacuous here (single SCC). |
| | **Hendriks, Behrmann, Larsen, Niebert, Vaandrager.** *Adding symmetry reduction to UPPAAL.* FORMATS 2003 — the `scalar` type. |
| | **RTWBS** (this group). *Asymmetric Timed Refinement for Correct Post-Deployment Updates in Real-Time Systems.* — see `../RTWBS.pdf`. |
