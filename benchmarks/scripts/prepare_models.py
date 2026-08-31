#!/usr/bin/env python3
"""Materialise the benchmark set: models, queries, and (optionally) LTSmin .so modules.

The generators emit XML on stdout; this script drives them over a size range and
writes everything into a layout the harness understands:

    benchmarks/models/<family><N>.xml     HORA / verifyta input
    benchmarks/queries/<family>.q         one safety query per family (platform-independent)
    benchmarks/so_<plat>/<family><N>.so   opaal+LTSmin PINS module (with --so)

Building .so modules needs opaal, which is Python 2 (see gen_so_linux.sh/gen_so_macos.sh);
without it the three-way comparison degrades to verifyta vs HORA, which the harness
handles by skipping the LTSmin rows rather than failing.

    python3 prepare_models.py                 # models + queries for every family (auto-detect platform)
    python3 prepare_models.py --so            # also build LTSmin modules
    python3 prepare_models.py --linux         # force the Linux .so toolchain (queries are shared)
    python3 prepare_models.py --only phil,fischer --range 4 10
"""
import argparse, os, subprocess, sys
from tqdm import tqdm
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(SCRIPTS)  # benchmarks/
GEN = os.path.join(HERE, "generators")

# family -> (default N range, query). ONE query per family, not a (macOS, Linux) pair: the query
# text is model-level CTL with nothing platform-specific in it, and keeping two copies is what let
# them drift. The Linux column had rotted badly by 2026-07-31 -- `csma` named a location `S1.cs`
# that the family does not have (its locations are Active/Collided/Idle/Retry/Send/Transmitting/
# Wait), and `phil` named F0/F1 where the generator emits P0/P1.
#
# EVERY query below is verified SATISFIED on a small size of its family, which matters more than
# it sounds: a falsified property makes both tools stop at the first counterexample, so the state
# count stops measuring the state space and starts measuring search order. The old csma query
# `A[] not (S1.Send && S2.Send)` was exactly that -- violated after 3 explored states.
# Verified with verifyta -u -s (explored counts): fischer8 40536, phil8 8486, twolock8 42584,
# arbiter8 1280, csma8 5888, single5 570, viking10 45566.
# --------------------------------------------------------------------------- liveness
# Each family gets TWO queries, written to reach_queries/ and live_queries/.
#
# A liveness query has to clear a HIGHER bar than "is it true". Three shapes look fine and
# are worthless, and each was measured on size 4 before being rejected:
#
#   falsified leads-to    stops at the first counterexample -- `P0.req --> P0.cs` on fischer
#                         decides in 6 states against a 268-state space. Measures search
#                         order, not the state space. Same trap as a falsified safety query.
#   E[] p                 satisfied in ONE state: the witness is the initial state and the
#                         greatest fixpoint stops there. `E[] not P0.cs` = 1 state.
#   A<> p                 falsified in one state wherever the system may idle.
#
# What works is a leads-to that HOLDS, which forces the backward fixpoint over the whole
# reachable set. Those cost MORE than the family's safety query, which is the point --
# measured explored-state counts at size 4, liveness vs reach:
#   phil 260 vs 80    fischer 1468 vs 268   twolock 2014 vs 332
#   arbiter 8192 vs 48   csma 6682 vs 104   single 14074 vs 61
#
# The shape that holds is "a location whose invariant forces progress reaches its unique
# successor". Where no such location exists the family gets NO liveness query rather than a
# vacuous one -- see viking below.
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
    "grc":     ((4, 20), "A[] not deadlock", None),
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
    "viking":  ((4, 26), "A[] not (Viking1.safe && Viking1.unsafe)", None),
    "train":   ((0, 1),  "A[] not deadlock", None),
    "hddi":    ((0, 20), "A[] !((ST1==station_y_sync || ST1==station_y_async || ST1==station_z_async || ST1==station_z_sync) && (ST2==station_y_sync || ST2==station_y_async || ST2==station_z_async || ST2==station_z_sync))", None),
}

# KNOWN LIMITATION (measured 2026-08-11, worth a look before relying on liveness numbers):
# a leads-to whose ANTECEDENT is a clock constraint decides false without exploring anything.
# On single4, whose only location carries the invariant x0<=6:
#     x0 >= 6 --> x0 <= 0     NOT satisfied, 0 states     (should hold: at x0==6 the invariant
#                                                          blocks time, so the reset must fire)
#     x0 >= 6 --> x0 >= 6     NOT satisfied, 0 states     (reflexive -- cannot be false)
# The reflexive case is the tell: `P.A --> P.A` on the same model is satisfied over 14074
# states. Location-predicate antecedents are fine; clock-predicate ones look broken.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated family names")
    ap.add_argument("--range", nargs=2, type=int, default=None, metavar=("LO", "HI"))
    ap.add_argument("--so", action="store_true", help="also build LTSmin .so modules")
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--linux", action="store_true", help="generate Linux queries/modules")
    ap.add_argument("--macos", action="store_true", help="generate macOS queries/modules")
    ap.add_argument("--queries-only", action="store_true",
                    help="write reach_queries/ and live_queries/ and nothing else. Use this "
                         "while a sweep is running: regenerating a model file the harness is "
                         "about to read would swap the input mid-measurement.")
    args = ap.parse_args()

    # Auto-detect platform if not specified
    if not args.linux and not args.macos:
        args.linux = sys.platform != "darwin"
        args.macos = sys.platform == "darwin"

    want = [f.strip() for f in args.only.split(",") if f.strip()] or sorted(FAMILIES)
    mdir = os.path.join(args.outdir, "models", "xml"); os.makedirs(mdir, exist_ok=True)
    # Two query sets, one dir each, and ONE dir per platform (see FAMILIES): the query text is
    # model-level CTL with nothing platform-specific in it, and keeping per-platform copies is
    # exactly what let them drift. --linux/--macos still select the .so toolchain below, which
    # genuinely is platform-specific.
    rqdir = os.path.join(args.outdir, "queries", "reach"); os.makedirs(rqdir, exist_ok=True)
    lqdir = os.path.join(args.outdir, "queries", "live");  os.makedirs(lqdir, exist_ok=True)
    sodir = os.path.join(args.outdir, "models", "so_linux" if args.linux else "so_macos"); os.makedirs(sodir, exist_ok=True)
    nolive = []

    made, failed = 0, []
    for fam in tqdm(want):
        if fam not in FAMILIES:
            print("unknown family %r (have: %s)" % (fam, ", ".join(sorted(FAMILIES))), file=sys.stderr)
            continue
        (lo, hi), reach_q, live_q = FAMILIES[fam]
        if args.range:
            lo, hi = args.range
        open(os.path.join(rqdir, fam + ".q"), "w").write(reach_q + "\n")
        if live_q:
            open(os.path.join(lqdir, fam + ".q"), "w").write(live_q + "\n")
        else:
            # No file at all, rather than an empty or vacuous one: the harness skips a family
            # whose query is missing, which is the correct outcome for a family with no valid
            # liveness property.
            nolive.append(fam)
        if args.queries_only:
            continue
        gen = os.path.join(GEN, "gen_%s.py" % fam)
        if not os.path.exists(gen):
            failed.append("%s: no generator" % fam); continue
        for n in tqdm(range(lo, hi + 1), desc=fam, leave=False):
            xml = os.path.join(mdir, "%s%d.xml" % (fam, n))
            try:
                r = subprocess.run([sys.executable, gen, str(n)], capture_output=True, check=True, text=True)
                open(xml, "w").write(r.stdout)
            except subprocess.CalledProcessError as e:
                if os.path.exists(xml):
                    os.remove(xml)
                failed.append("%s%d: %s" % (fam, n, e.stderr[:70])); continue
            made += 1
            if args.so:
                so = os.path.join(sodir, "%s%d.so" % (fam, n))
                if args.linux:
                    r = subprocess.run([os.path.join(SCRIPTS, "generators", "gen_so_linux.sh"), xml, so],
                                       capture_output=True)
                else:
                    r = subprocess.run([os.path.join(HERE, "generators", "gen_so_macos.sh"), xml, so],
                                       capture_output=True)
                if r.returncode:
                    if os.path.exists(so):
                        os.remove(so)
                    failed.append("%s%d.so: %s" % (fam, n, r.stderr.decode().strip()[:70]))
    print("queries -> %s  and  %s" % (rqdir, lqdir))
    if nolive:
        print("no liveness query (no property that both holds and forces exploration): %s"
              % ", ".join(sorted(nolive)))
    if not args.queries_only:
        print("generated %d models into %s" % (made, mdir))
    if failed:
        print("\n%d issue(s):" % len(failed))
        for f in failed[:15]:
            print("  " + f)


if __name__ == "__main__":
    main()
