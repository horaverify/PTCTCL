# Liveness soundness regressions

Three models that each caught a real wrong verdict. All are tiny and diagonal-free.

The prototype binary is gone: the backward-gfp engine it was prototyping is now the
shipping engine, so these run against `ptctlmc` itself. Put the query in a file and:

    echo '<p> --> <q>' > q.q && ./build/ptctlmc --backend seq <model>.xml q.q

There is one liveness engine (`leads_to_otf`) and no environment switches: the `gfp` and
`net` arms of `PTCTLMC_LIVENESS`, and the OTF ablation switches below, were removed on the
`shipping` branch after a full 46-query differential found zero disagreements.

| model | query | correct answer | what it catches |
|---|---|---|---|
| `cex1.xml` | `P.Q --> P.Live` | **NOT satisfied** | The deadlock-completion certificate computed `Dead(q)` on `Dbm(nclocks)` — the **zero** zone, i.e. the ray `x1=...=xn>=0` — instead of `Dbm::universe`. Conditions (i)/(ii) were then meaningless and the certificate fired on false evidence, reporting the property satisfied. |
| `cex2.xml` | `P.U --> P.Live` | **SATISFIED** | `time_diverges` delayed the zone without consulting `frozen_flags`, so it reported divergence at an urgent location where no positive delay is legal — inventing a maximal-run witness. This one was **also wrong in the shipping engine** (`src/netreach.cpp:2650`, `:2977`), not just the prototype. |
| `cex3.xml` | `P.U --> P.Live` | **NOT satisfied** | Certificate drop-condition (iii) ("`Inv(q)` bounds no clock, so time diverges, so `stop` already covers the zone") fired at a *frozen* key, where `Inv(q)` is the undelayed universe and so trivially bounds no clock — exactly where the reasoning does not hold. |

`cex2` and `cex3` are **coupled**: fixing the frozen-blind `time_diverges` alone makes `cex3`
wrong, and gating (iii) on non-frozen alone leaves `cex2` wrong. Land them together.

These used to be checked under four configurations (atoms lane, certificate-then-race,
gate+race, a_M) selected by `PTCTLMC_OTFNOATOMS` / `OTFNOCERT` / `OTFAM` / `OTFNOGATE`. Those
switches are gone. The shipping path is the one the table above is checked against: the atoms
lane whenever the deadlock-completion seed is non-empty, the certificate's early return when it
is empty, and the gate + race only when the atom map exceeds its 64-atom cap.

## `bcast-receiver.xml` — broadcast receivers with clock guards (FIXED)

    E<> P.T && Q.A     correct answer: SATISFIED     was: NOT satisfied     now: satisfied
    E<> P.T && Q.B     correct answer: SATISFIED     was: satisfied         now: satisfied

A broadcast `a!` fires whether or not any receiver is enabled; each `a?` participates exactly
when its own guard holds. `successors_net` builds its receiver list from processes with an
**int**-enabled `a?` edge only (`src/netreach.cpp:339-348`), then `if (!es.empty())` **forces**
that process to participate and conjoins its **clock** guard
(`src/netreach.cpp:357-358`). The branch where the clock guard fails and the receiver stays put
is never generated.

Here `Q`'s `cd?` is guarded `y>=5`. At `y=0` the sender should fire alone, leaving `(P.T, Q.A)` —
a reachable state the engine never generates. Missing reachable states means `E<>` false
negatives and, worse, `A[]` false **positives**.

Not exercised by any benchmark: csma is the only broadcast model and its five `cd?` edges carry
no guard at all, so `apply_all` of an empty guard is a no-op. Every measurement in
`UppalOMP/LIVENESS_FIX_CHANGELOG.md` is therefore unaffected.

**Fixed** by `build_broadcast_parts` in `src/netreach.cpp`, used at all **five** broadcast loops.
Three are in the reachability prefix (plain successors, the slot-reusing hot path, and the one
inside `state_has_deadlock`); the other two arrived with the liveness engine and had the same
defect — `gfp_succ_tagged`, which is what `Otf::expand` calls and therefore the whole shipping
liveness path, and `memo_enumerate`. Backward, an abstaining receiver cannot be named by
`GfpTag::parts` at all, so `GfpTag::abstain` carries its guard complement for `gfp_pre`. Each
receiver process now gets an explicit ABSTAIN option guarded by the complement of the disjunction
of its edges' clock guards — a DNF whose disjuncts are each convex, hence each a separate option.

The collapse case is what makes this safe to land: if any receiver edge has an **empty** clock
guard, the complement of `true` is false, no abstain option is generated, and the option list is
exactly what the old code produced. csma is that case, so it is unaffected — verified, `A[] not
deadlock` on csma4/5/6 gives 202 / 1271 / 6259 states on the `seq` backend before and after.
(The `omp` backend wobbles by a few states run to run — 6269 / 6267 / 6264 on csma6 — which is
pre-existing subsumption-order nondeterminism, not this change.)
