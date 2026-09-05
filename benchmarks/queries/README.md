# Benchmark queries

These files ARE the benchmark definition. They are hand-curated per family, not generated:
`prepare_models.py` builds models and LTSmin `.so` modules only, and does not write anything
here. Edit a query by editing the file.

Every one is chosen so that the tools have to do real work. A query that is falsified, or
satisfied at the initial state, measures the search order rather than the state space -- see the
rationale below, which records what was tried and rejected, with measured counts.

Directories: `reach/` safety, `live/` leads-to, `eg/` the `E[]` set (four shapes per family, see
`eg/README`), `deadlock/` `A[] not deadlock`, `live_violated/` deliberately falsified variants.

## Why each family has the query it has

Each family gets TWO queries, written to reach_queries/ and live_queries/.

A liveness query has to clear a HIGHER bar than "is it true". Three shapes look fine and
are worthless, and each was measured on size 4 before being rejected:

  falsified leads-to    stops at the first counterexample -- `P0.req --> P0.cs` on fischer
                        decides in 6 states against a 268-state space. Measures search
                        order, not the state space. Same trap as a falsified safety query.
  E[] p                 satisfied in ONE state: the witness is the initial state and the
                        greatest fixpoint stops there. `E[] not P0.cs` = 1 state.
  A<> p                 falsified in one state wherever the system may idle.

What works is a leads-to that HOLDS, which forces the backward fixpoint over the whole
reachable set. Those cost MORE than the family's safety query, which is the point --
measured explored-state counts at size 4, liveness vs reach:
  phil 260 vs 80    fischer 1468 vs 268   twolock 2014 vs 332
  arbiter 8192 vs 48   csma 6682 vs 104   single 14074 vs 61

The shape that holds is "a location whose invariant forces progress reaches its unique
successor". Where no such location exists the family gets NO liveness query rather than a
vacuous one -- see viking below.
    FAMILIES = {
    "fischer": ((4, 20), "A[] not (P1.cs && P2.cs)",
    # req has an invariant and one way out; cs does NOT (a process may sit in the
    # critical section), so `P0.cs --> P0.idle` is falsified and unusable.
    "P0.req --> P0.wait"),
    "phil":    ((4, 20), "A[] not (P0.Eat && P1.Eat)",
    # Eat -> Dropped is forced; Dropped -> Think is not (16-state refutation).
    "P0.Eat --> P0.Dropped"),
    "twolock": ((4, 20), "A[] not (P1.cs && P2.cs)",
    # The second lock costs more than the first: reqB 2014 states vs reqA 1448.
    "P0.reqB --> P0.waitB"),
    "arbiter": ((4, 25), "A[] not (C1.cs && C2.cs)",
    # `C0.want --> C0.cs` is FALSE -- the arbiter grants no fairness, so a client
    # can be passed over forever (refuted in 1 state). Releasing is forced.
    "C0.cs --> C0.idle"),
    # No mutual exclusion to state on csma: the interesting invariant is that the bus is never
    # transmitting while two senders are both in Send. Reproduces the documented figures --
    # csma14 = 618496 states for HORA against 991232 for verifyta.
    "csma":    ((4, 22), "A[] not (S0.Send and S1.Send and Bus.Transmitting)",
    # The BUS is the component with forced progress, not a station: `S0.Send -->
    # S0.Wait` is refuted in 8 states (a collision sends it to Retry instead).
    "Bus.Transmitting --> Bus.Idle"),
    # `A[] not deadlock` is FALSIFIED on grc -- measured NOT satisfied in 883 states on grc4 --
    # so the run stops at the first counterexample and the count measures the path to it, not the
    # state space. `A[] true` has no bad state to stop at and is the honest full-exploration probe.
    # The liveness query holds (grc4 788 states, grc5 12184, both tools agreeing SAT), so it
    # exhausts the reachable set the way the other families' leads-to queries do.
    "grc":     ((4, 20), "A[] true", "T0.in --> T0.far"),
    # singleN is ONE process with ONE location and N independent clocks (self-loop per clock).
    # It has no critical section, so the mutex query copied from the fischer/arbiter families
    # referenced a location that does not exist: verifyta rejected it with "has no member named
    # cs" and HORA reported UNSUPPORTED, so every single* row was an error for both tools.
    # The clocks are global, so constrain one directly. x0<=6 is the location invariant, so the
    # property HOLDS and the whole state space is explored (x0<=2 is refuted by both tools, so
    # it is not vacuous).
    # single is ONE location with N independent clocks, so there is no "next location" to lead
    # to; the reflexive leads-to on the only location is the honest form. It is trivially TRUE
    # as a property, and that is fine here -- the point is the cost, and deciding it still runs
    # the backward fixpoint over the entire reachable zone graph (14074 states against 61 for
    # the safety query). Clock-predicate antecedents were tried first and do NOT work: see the
    # note at the bottom of this file.
    "single":  ((4, 20), "A[] x0 <= 6", "P.A --> P.A"),
    # NO liveness query for viking. Every named location (Viking*.safe/.unsafe, Torch.free/
    # .one/.two) is a place the system may sit in forever -- none carries a progress-forcing
    # invariant -- so every leads-to over them is falsified, and the crossing locations that DO
    # force progress are anonymous in the source model and cannot be named in a query. Six
    # candidates were measured, all refuted in 0-21 states. A vacuous query here would measure
    # nothing, so the family simply has no liveness column.
    # The "no liveness query" note above is out of date: every candidate it rejected was a
    # location-to-location leads-to, and what works here puts a CLOCK in the response. It holds and
    # scales -- viking4 206 states, viking15 2768894, viking17 13631486, viking20 MEMOUT -- so it
    # measures the state space. Note gen_tck_live.py cannot encode it (that generator handles only
    # `Process.location --> Process.location`), which is why viking has no tchecker liveness column.
    "viking":  ((4, 26), "A[] not (Viking1.safe && Viking1.unsafe)", "Viking1.unsafe --> y1 >= 5"),
    "train":   ((0, 1),  "A[] not deadlock", None),
    # The old reach query compared a PROCESS to LOCATION names (`ST1==station_y_sync`). ST1 is a
    # process and station_y_sync one of its locations, so verifyta rejected the whole query
    # ("Unknown identifier: station_y_sync") and every hddi row was an ERR for all three tools.
    # Spelled correctly as location predicates it still asks for a disjunction under `A[] !(...)`,
    # which hora's safety path cannot lower, so hddi uses `A[] true` exactly as grc does: no bad
    # state to stop at, hence the full state space. The leads-to holds on every size.
    "hddi":    ((0, 20), "A[] true", "ST1.station_z_sync --> ST1.station_z_idle"),
    }

## Families with no query of a given kind

* `train` has no liveness query: no property both holds and forces exploration.
* `grc` and `hddi` have no `E[]` query file; they are not part of that suite.
* `viking` has no tchecker liveness model -- `gen_tck_live.py` encodes only
  `Process.location --> Process.location`, and viking's response is a clock (`--> y1 >= 5`).
* tchecker refuses exactly one `E[]` query per family (8 in total): a clock constraint under
  `||` or `!` is not expressible as a TChecker invariant. Recorded in
  `models/tck_live_e/manifest.json` under `refused`.
