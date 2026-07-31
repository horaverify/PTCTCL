#!/usr/bin/env python3
"""Materialise the benchmark set: models, queries, and (optionally) LTSmin .so modules.

The generators emit XML on stdout; this script drives them over a size range and
writes everything into a layout the harness understands:

    benchmarks/models/<family><N>.xml     PTCTLMC / verifyta input
    benchmarks/queries/<family>.q         one safety query per family (platform-independent)
    benchmarks/so_<plat>/<family><N>.so   opaal+LTSmin PINS module (with --so)

Building .so modules needs opaal, which is Python 2 (see gen_so_linux.sh/gen_so_macos.sh);
without it the three-way comparison degrades to verifyta vs PTCTLMC, which the harness
handles by skipping the LTSmin rows rather than failing.

    python3 prepare_models.py                 # models + queries for every family (auto-detect platform)
    python3 prepare_models.py --so            # also build LTSmin modules
    python3 prepare_models.py --linux         # force the Linux .so toolchain (queries are shared)
    python3 prepare_models.py --only phil,fischer --range 4 10
"""
import argparse, os, subprocess, sys
from tqdm import tqdm
HERE = os.path.dirname(os.path.abspath(__file__))
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
FAMILIES = {
    "fischer": ((4, 20), "A[] not (P1.cs && P2.cs)"),
    "phil":    ((4, 20), "A[] not (P0.Eat && P1.Eat)"),
    "twolock": ((4, 20), "A[] not (P1.cs && P2.cs)"),
    "arbiter": ((4, 25), "A[] not (C1.cs && C2.cs)"),
    # No mutual exclusion to state on csma: the interesting invariant is that the bus is never
    # transmitting while two senders are both in Send. Reproduces the documented figures --
    # csma14 = 618496 states for PTCTLMC against 991232 for verifyta.
    "csma":    ((4, 22), "A[] not (S0.Send and S1.Send and Bus.Transmitting)"),
    "grc":     ((4, 20), "A[] not deadlock"),
    # singleN is ONE process with ONE location and N independent clocks (self-loop per clock).
    # It has no critical section, so the mutex query copied from the fischer/arbiter families
    # referenced a location that does not exist: verifyta rejected it with "has no member named
    # cs" and PTCTLMC reported UNSUPPORTED, so every single* row was an error for both tools.
    # The clocks are global, so constrain one directly. x0<=6 is the location invariant, so the
    # property HOLDS and the whole state space is explored (x0<=2 is refuted by both tools, so
    # it is not vacuous).
    "single":  ((4, 20), "A[] x0 <= 6"),
    "viking":  ((4, 26), "A[] not (Viking1.safe && Viking1.unsafe)"),
    "train":   ((0, 1),  "A[] not deadlock"),
    "hddi":    ((0, 20), "A[] !((ST1==station_y_sync || ST1==station_y_async || ST1==station_z_async || ST1==station_z_sync) && (ST2==station_y_sync || ST2==station_y_async || ST2==station_z_async || ST2==station_z_sync))"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated family names")
    ap.add_argument("--range", nargs=2, type=int, default=None, metavar=("LO", "HI"))
    ap.add_argument("--so", action="store_true", help="also build LTSmin .so modules")
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--linux", action="store_true", help="generate Linux queries/modules")
    ap.add_argument("--macos", action="store_true", help="generate macOS queries/modules")
    args = ap.parse_args()

    # Auto-detect platform if not specified
    if not args.linux and not args.macos:
        args.linux = sys.platform != "darwin"
        args.macos = sys.platform == "darwin"

    want = [f.strip() for f in args.only.split(",") if f.strip()] or sorted(FAMILIES)
    mdir = os.path.join(args.outdir, "models"); os.makedirs(mdir, exist_ok=True)
    # One queries/ dir for every platform (see FAMILIES). --linux/--macos still select the .so
    # toolchain below, which genuinely is platform-specific.
    qdir = os.path.join(args.outdir, "queries"); os.makedirs(qdir, exist_ok=True)
    sodir = os.path.join(args.outdir, "so_linux" if args.linux else "so_macos"); os.makedirs(sodir, exist_ok=True)

    made, failed = 0, []
    for fam in tqdm(want):
        if fam not in FAMILIES:
            print("unknown family %r (have: %s)" % (fam, ", ".join(sorted(FAMILIES))), file=sys.stderr)
            continue
        (lo, hi), query = FAMILIES[fam]
        if args.range:
            lo, hi = args.range
        open(os.path.join(qdir, fam + ".q"), "w").write(query + "\n")
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
                    r = subprocess.run([os.path.join(HERE, "gen_so_linux.sh"), xml, so],
                                       capture_output=True)
                else:
                    r = subprocess.run([os.path.join(HERE, "gen_so_macos.sh"), xml, so],
                                       capture_output=True)
                if r.returncode:
                    if os.path.exists(so):
                        os.remove(so)
                    failed.append("%s%d.so: %s" % (fam, n, r.stderr.decode().strip()[:70]))
    print("generated %d models into %s" % (made, mdir))
    if failed:
        print("\n%d issue(s):" % len(failed))
        for f in failed[:15]:
            print("  " + f)


if __name__ == "__main__":
    main()
