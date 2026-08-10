# Liveness subsumption: the wrong verdict on `grc4  T0.in --> T0.far`

Status: ROOT CAUSE FOUND AND DEMONSTRATED, FIXED, MEASURED.
A *second, independent and pre-existing* wrong verdict was found on the way; it is
diagnosed here and deliberately NOT fixed (see section 7).

Everything below was produced by running the binaries, not by argument. Every run that
did not finish is reported as UNKNOWN and is never counted as agreement.

---

## 1. Reproduction (all completed; no timeouts)

```
echo 'T0.in --> T0.far' > /tmp/q.q
PTCTLMC_LTSUB=1 ./build-mostopt/ptctlmc --backend omp benchmarks/models/grc4.xml /tmp/q.q
   -> NOT satisfied,      205 states,   0.01 s          <-- WRONG
./build-mostopt/ptctlmc  --backend omp benchmarks/models/grc4.xml /tmp/q.q
   -> satisfied,      1190376 states, 486    s
./build-base/ptctlmc     --backend omp benchmarks/models/grc4.xml /tmp/q.q
   -> satisfied,       705806 states,   2.4  s
verifyta -u                                             -> satisfied
```

`--backend seq` reproduces identically (211 states), so it is not a data race.

## 2. Instrumentation added (debug only, all env-gated, all off by default)

In `src/netreach.cpp`, inside `leads_to_net`:

| env var | what it does |
|---|---|
| `PTCTLMC_LTTRACE` | at every site that can return "violated", print WHICH of the three witnesses fired, the DFS stack, and for each node its location vector, its full DBM, `|adj|`, `time_diverges`, `is_p`, `is_q`, and every out-edge with raw id / resolved id / absorbed flag |
| `PTCTLMC_LTWATCH="l0,l1,..."` | dump every expanded node whose location-NAME vector matches, with its zone and its successor list |
| `PTCTLMC_LTABSDBG` | count newcomers that were absorbed even though they are deadlocked (a witness thrown away); costs one successor computation per absorption |
| `PTCTLMC_LTNOSUBEDGE` | ablation: restore the unrestricted (wrong) behaviour |

## 3. What actually fires — the offending edge

```
[lttrace] WITNESS in neg_scc(94): kind2 CYCLE: edge back to a node on the DFS stack
[lttrace]   DFS stack (root..top): 94(P) 252(P) 504(P)
   94 = T0.in T1.far  T2.far T3.far G.down Ctl.dn_1
  252 = T0.in T1.near T2.far T3.far G.down Ctl.dn_2
  504 = T0.in T1.in   T2.far T3.far G.down Ctl.dn_2
[lttrace]   back-edge 504 -> raw 94 resolved 94 sub=1
```

Three facts from that dump settle the diagnosis:

1. **It is the CYCLE witness.** Not deadlock, not divergence. The brief's hypothesis that
   witnesses (1) and (2) might be unsound under subsumption is refuted for this query —
   they are local to a stored node, and every stored node is a genuinely reachable
   symbolic state.
2. **`raw == resolved` on every edge of the cycle.** No `fwd[]` / `resolve()` redirection
   is involved. That is exactly why disabling RETIREMENT changed nothing.
3. The cycle is a real *discrete* cycle: T1 approaches, enters and leaves the crossing
   while T0 sits in `in`. One lap costs at least `x1>=4` then `x1>=1`, i.e. **5 time
   units**.

### Why it is spurious — the zones

Global `Extra_LU` bounds for grc4 (printed by the same trace):

```
x0: L=4 U=15   x1: L=4 U=15   x2: L=4 U=15   x3: L=4 U=15   y: L=2 U=2   z: L=0 U=1
```

Node 94's stored zone:

```
x1 = x2 = x3 = y = z = 0 ,   x0 >= 0 ,   x0 UNBOUNDED ABOVE
```

`x0 <= 15` — the invariant of `T0.in`, the only thing that bounds how long T0 can stay in
the crossing — is gone, because `Extra_LU+` relaxes any entry whose bound exceeds `L[i]`
and `15 > L[x0] = 4`. That is ordinary, sound-for-reachability extrapolation; UPPAAL does
the same and `build-base` stores exactly the same zone.

What survives extrapolation is the **lower** bound on `x0`. The true `leave` successor of
node 504 is

```
x1 = x2 = x3 = y = z = 0 ,   x0 >= 4
```

a **strict subset** of node 94's zone. Under exact zone equality this becomes a new node,
and the next lap gives `x0 >= 5`, then `>= 10`, then `>= 15`, at which point `Extra_LU+`
tightens the entry to `x0 > 15`, the invariant `x0 <= 15` empties the zone, and the loop
provably cannot be repeated. That is why `build-base` says SAT.

Under subsumption `covered_by(94)` succeeds, `intern()` returns 94, the edge is recorded
as `504 -> 94`, and the loop closes on itself with the growth of `x0` erased.

**This is a subsumption edge in the classical sense** — the newcomer was absorbed into a
strictly larger stored zone — even though `resolve()` is the identity on it, because
absorption of a NEWCOMER happens by *returning an existing id from `intern`*, never by
writing `fwd[]`.

## 4. Why the earlier repair attempt did not work

The brief reports that "mark subsumption edges and forbid a cycle that uses one" (the
Herbreteau–Srivathsan–Walukiewicz / Laarman criterion) did not help. The trace shows why
that is consistent with the diagnosis, and it is a placement problem, not a theory
problem: the witness fires in the **eager back-edge test**, near the top of `neg_scc`,

```cpp
if (tidx[v] != -1 && onstk[v] && stack_has_p()) return true;
```

which runs long before any component is ever popped. A nested Tarjan applied only to the
component-pop criterion never gets a chance to veto it. The marking has to be enforced at
*every* place an edge is used to conclude "a run can go on forever":

* the eager back-edge test,
* the self-loop test at component pop,
* the "component has more than one member" test (i.e. the Tarjan DFS itself must not
  traverse such edges),
* the kind-3 propagation into an already-escaping finished component.

## 5. The fix

`src/netreach.cpp`, `leads_to_net`:

* `intern()` sets `intern_sub` when it returns an existing id whose stored zone is not
  byte-identical to the state handed in.
* Adjacency entries carry that flag in bit 30 of the node id (`kSubBit`), so the marker
  costs **no extra memory** — a parallel `vector<vector<char>>` would have cost more than
  the analysis on a 1.2M-node liveness graph.
* `exact_edge(u,k)` is true iff the edge is not a subsumption edge **and** `resolve()` is
  the identity on it. The second half catches the other way to become inexact: a node that
  was later RETIRED into a bigger newcomer, where `resolve()` does redirect.
* The Tarjan pass and all four escape criteria consult `exact_edge`.
* `adj[u].empty()` (the deadlock witness) still reads the FULL adjacency — a state whose
  only successors were absorbed is *not* deadlocked — and the outer reachability BFS still
  follows every edge, so no reachable state goes unvisited.

Soundness of the restriction is exact, not heuristic: every surviving edge `u -> v` has
`v` equal to the extrapolated successor zone computed from `u`, so the restricted graph is
a subgraph of the edge relation the exact-equality engine builds. A cycle found in it is a
cycle `build-base` would also find, and the deadlock/divergence witnesses are local to a
stored node. **The restricted search can therefore only report witnesses the exact engine
would also report.**

It is *not* proven complete: an absorbed newcomer is never expanded, so a witness that
exists only below it is not looked for. That is why the DEFAULT is left at exact zone
equality. `PTCTLMC_LTNOSUBEDGE=1` restores the old behaviour for A/B runs.

`PTCTLMC_LTSUB` is unchanged as the opt-in for subsumption; with no `PTCTLMC_LTSUB`,
`covered_by` only succeeds on byte-equal zones and `covers_j` is hard-false, so
`intern_sub` is never set and `fwd[]` is never written — `exact_edge` is identically true
and the default code path is bit-for-bit the one that was there before.

## 6. Measurements

### 6.1 The reported query

```
PTCTLMC_LTSUB=1 ./build-mostopt/ptctlmc --backend omp grc4.xml 'T0.in --> T0.far'
   before:  NOT satisfied      205 states   0.01 s     WRONG
   after:       satisfied    11660 states   0.1  s     agrees with verifyta and build-base
```

### 6.2 Differential sweep vs verifyta — 188 leads-to queries

Two generated query sets (all ordered location pairs of the first template, plus
cross-template pairs, plus a hand-picked set aimed at the resource/bus/torch components),
one query per process invocation, 600–900 s budget each. Both verdict directions occur
(SAT and UNSAT) in every model.

Set A — 134 queries over grc4, fischer4, fischer5, fischer6, twolock5, arbiter5, phil6,
csma6, viking6. Set B — 54 queries over grc4, fischer6, twolock5, arbiter5, phil6, csma6,
viking6.

| | queries | disagreements with verifyta |
|---|---|---|
| `PTCTLMC_LTSUB=1` **with the fix** | 188 | 4 (all grc4, all also wrong in exact mode — see §7) |
| `PTCTLMC_LTSUB=1 PTCTLMC_LTNOSUBEDGE=1` (old) | 188 | 8 |

The eight the old code got wrong: grc4 `T0.in --> T0.far`, `T0.near --> T0.far`,
`T0.far --> T1.far`, `T0.near --> T1.far`, `T1.in --> T1.far`, phil6 `P0.Eat -->
P0.Dropped`, `P0.Eat --> F0.Free`, and grc4 `T0.near --> T0.in`. The fix repairs seven of
them; the eighth (`T0.near --> T0.in`) is the separate bug of §7 and is wrong in every
mode. **So the bug was never grc4-specific — phil6 was affected too.**

### 6.3 The decisive equivalence: subsuming vs exact, all 14 grc4 queries

`build-base` (the pre-today, exact-hash, no-subsumption binary) against
`PTCTLMC_LTSUB=1` on the fixed binary, all 14 completed, none timed out:

```
                            build-base   LTSUB+fix   LTSUB old   verifyta
T0.far  --> T0.near           UNSAT        UNSAT       UNSAT       UNSAT
T0.far  --> T0.in             UNSAT        UNSAT       UNSAT       UNSAT
T0.near --> T0.far            SAT          SAT         UNSAT       UNSAT  (*)
T0.near --> T0.in             SAT          SAT         SAT         UNSAT  (*)
T0.in   --> T0.far            SAT          SAT         UNSAT       SAT
T0.in   --> T0.near           UNSAT        UNSAT       UNSAT       UNSAT
T0.far  --> T1.far            SAT          SAT         UNSAT       UNSAT  (*)
T0.far  --> T1.near           UNSAT        UNSAT       UNSAT       UNSAT
T0.near --> T1.far            SAT          SAT         UNSAT       UNSAT  (*)
T0.near --> T1.near           UNSAT        UNSAT       UNSAT       UNSAT
T0.far  --> Ctl.up            UNSAT        UNSAT       UNSAT       UNSAT
T0.far  --> Ctl.low_1         UNSAT        UNSAT       UNSAT       UNSAT
T0.near --> Ctl.up            UNSAT        UNSAT       UNSAT       UNSAT
T0.near --> Ctl.low_1         UNSAT        UNSAT       UNSAT       UNSAT
```

**LTSUB with the fix is verdict-identical to the exact-equality engine on 14/14.** The
four rows marked (*) are the separate bug of §7: the exact engine gets them wrong too, and
so does the pre-today binary, so they are not a regression of anything done here.

### 6.4 Reachability untouched

```
PTCTLMC_STATS=1 ./build-mostopt/ptctlmc --backend omp fischer10.xml fischer.q
   discrete-states=260998 stored-zones=260998        (required: 260998 / 260998)
PTCTLMC_STATS=1 ./build-mostopt/ptctlmc --backend omp arbiter14.xml arbiter.q
   discrete-states=131072 stored-zones=131072        (required: 131072 / 131072)
```

Standard `A[]` benchmark queries, verifyta vs default vs LTSUB, all 9 models: all agree.
`tests/validate_liveness.sh` passes in both modes (9/9). `ptctlmc_unit`: 105 checks, 0
failures.

---

## 7. A SECOND, independent wrong verdict — found, diagnosed, NOT fixed

The four grc4 queries marked (*) above are wrong in **every** configuration, including
`build-base`. They are not caused by subsumption.

`leads_to_net`'s first witness is `adj[u].empty()`, i.e. *the whole zone has no
successor*. UPPAAL's counterexample for `T0.near --> T0.in` ends in

```
( T0.near T1.near T2.near T3.near G.rai Ctl.low_4 )
      x_i <= 1 ,  y <= 2 ,  z <= 1 ,  z <= y
```

`Ctl.low_4` must fire `lower!` before `z` reaches 1, but `lower?` is only offered by
`G.up`, and `G` is in `rai` and can only leave at `y >= 2`, which `z <= 1` and `y <= z + 2`
make unreachable in time. That is a **time-lock**, and it traps only *part* of the zone:
the valuation `y = 2, z = 1` is in the zone and *does* enable `G.rai -> G.up`. The
`PTCTLMC_LTWATCH` dump confirms this exactly:

```
[ltwatch] node=27045 p=1 q=0 |succ|=1  zone: ... y-0<=2 z-0<=1 z-y<=0 ...
[ltwatch]     succ -> near near near near up low_4
```

One successor, so `adj[u].empty()` is false and the witness is invisible. The whole-zone
test cannot see a **partial** deadlock. `PTCTLMC_LTABSDBG` reports **0** absorbed
deadlocked newcomers on this query, so subsumption is not implicated at all.

The engine already computes partial deadlocks exactly — `state_has_deadlock()` in the same
file, used by `deadlock_reachable` — so the repair is to use that as the kind-1 witness
instead of `adj[u].empty()`. It is a separate change with its own regression surface
(it would flip verdicts on a large number of queries and interacts with the extrapolated
zone representation), so it is recorded here rather than bundled into a subsumption fix.

Incidentally this is what made the old, unrestricted subsumption look *right* on three of
those queries: it invented a cycle witness that happened to agree with a true violation it
could not otherwise see. Two bugs cancelling.

## 8. Files changed

* `src/netreach.cpp` — `leads_to_net` only. The subsumption-edge marker, `exact_edge` and
  its use at the four escape criteria; the debug instrumentation listed in §2; the comment
  block in `intern()` rewritten to record the real mechanism.

Nothing else is touched. Reachability, the GPU paths and `reach_engine` are untouched.
