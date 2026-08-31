# Violated `p --> q` queries

Every row of `../live_queries` is SATISFIED, and so is every recorded `verifyta` and `tchecker`
row in `../live_results` — the whole reference comparison to date is on satisfied queries. That
leaves the violated half of the property unmeasured for *every* tool, and unexercised in our own
engine for the one path only a violation reaches: `Otf::leadsto_witness`, which stops the forward
search the moment a reached `p /\ !q` valuation can end a maximal run.

## Why one state can decide it

`p --> q` fails exactly when some reachable state satisfies `p /\ EG !q`. The greatest fixpoint
that decides `EG !q` is

    W = W /\ (NoSucc \/ Pre(W))

and `NoSucc(v)` — the valuations of `v` from which the run may legitimately stop, by deadlock or
by time divergence — is **per state and zone-local**. It reads no successor, no cycle and no
graph. So a single expanded state that satisfies `p`, fails `q`, and has a non-empty `NoSucc` is
already a complete counterexample: the maximal run stops (or runs on) right there and never
reaches `q`.

The baseline engine could not use that, for an architectural reason rather than a mathematical
one: it builds one forward graph, then runs one backward fixpoint, and reads the verdict only
after that fixpoint converges. Nothing is concludable mid-construction. What makes reading it
early sound is that the certificates of a greatest fixpoint are its *post*-fixpoints
(Knaster–Tarski), so a witness found on a graph fragment is already a witness on the finished
one — the same argument the on-the-fly work rests on.

## The three shapes, kept deliberately

- **Stop witness, found after real exploration.** The state count collapses because the search
  ends at the witness. `fischer P0.cs --> P0.req` (11951 -> 37 at size 7),
  `fischer P0.wait --> P0.cs` (11951 -> 9, x54), `twolock P0.waitB --> P0.reqB` (12847 -> 121),
  `grc T0.near --> T0.far` (923 -> 505 at size 4, 13980 -> 3519 at size 5).

- **Cycle witness.** The counterexample is an infinite run through a cycle, which no single state
  certifies, so the early exit correctly does not fire and the backward fixpoint decides it. The
  state count is unchanged, which is the point of having these rows: they are the regression
  guard that the early exit is not firing where it must not.
  `arbiter C0.cs --> C0.want`, `csma Bus.Transmitting --> Bus.Collided`,
  `phil P0.Eat --> P0.Think`.

- **Violated at the initial state.** `grc T0.far --> T0.in` and `csma Bus.Idle --> Bus.Collided`
  name the *initial* location, which carries no invariant, so "stay there for ever" is already a
  maximal `!q` run and the very first expanded state decides the query. These look degenerate and
  are not: the baseline explores the whole reachable graph anyway (13980 states on grc5, and it
  does not finish grc6 or grc7 inside 60 s), because building-then-solving cannot notice. They are
  the sharpest case for the reference comparison, not the weakest.

## Not yet measured

Whether `verifyta` and `tchecker` shortcut any of these is **unknown and unmeasured here**;
neither binary is installed on this machine, and the recorded results carry only satisfied rows.
Running `../scripts/live_diff.py` gives our own old-vs-new numbers; the reference side needs the
existing `../scripts/run_live_pipeline.sh` on a machine that has the tools.
