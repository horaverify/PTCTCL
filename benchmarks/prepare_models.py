#!/usr/bin/env python3
"""Materialise the benchmark set: models, queries, and (optionally) LTSmin .so modules.

The generators emit XML on stdout; this script drives them over a size range and
writes everything into a layout the harness understands:

    benchmarks/models/<family><N>.xml     PTCTLMC / verifyta input
    benchmarks/queries_<plat>/<family>.q  one safety query per family (platform-specific)
    benchmarks/so_<plat>/<family><N>.so   opaal+LTSmin PINS module (with --so)

Building .so modules needs opaal, which is Python 2 (see gen_so_linux.sh/gen_so_macos.sh);
without it the three-way comparison degrades to verifyta vs PTCTLMC, which the harness
handles by skipping the LTSmin rows rather than failing.

    python3 prepare_models.py                 # models + queries for every family (auto-detect platform)
    python3 prepare_models.py --so            # also build LTSmin modules
    python3 prepare_models.py --linux         # force Linux queries/modules
    python3 prepare_models.py --only phil,fischer --range 4 10
"""
import argparse, os, subprocess, sys
from tqdm import tqdm
HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "generators")

# family -> (default N range, (macOS query, Linux query)). Queries explore the whole space rather
# than terminating early, so timings measure exploration and not luck.
FAMILIES = {
    "fischer": ((4, 20), ("A[] not (P1.cs && P2.cs)", "A[] not (P1.cs && P2.cs)")),
    "phil":    ((4, 20), ("A[] not (P0.Eat && P1.Eat)", "A[] not (F0.Eat && F1.Eat)")),  # Linux uses F0-F3
    "twolock": ((4, 20), ("A[] not (P1.cs && P2.cs)", "A[] not (P1.cs && P2.cs)")),  # Linux uses P0-P3
    "arbiter": ((4, 25), ("A[] not (P1.cs && P2.cs)", "A[] not (C1.cs && C2.cs)")),  # Linux uses C0-C3
    "csma":    ((4, 22), ("A[] not (P1.cs && P2.cs)", "A[] not (S1.cs && S2.cs)")),  # Linux uses S0-S3
    "grc":     ((4, 20), ("A[] not deadlock", "A[] not deadlock")),  # OK for both
    # singleN is ONE process with ONE location and N independent clocks (self-loop per clock).
    # It has no critical section, so the mutex query copied from the fischer/arbiter families
    # referenced a location that does not exist: verifyta rejected it with "has no member named
    # cs" and PTCTLMC reported UNSUPPORTED, so every single* row was an error for both tools.
    # The clocks are global, so constrain one directly. x0<=6 is the location invariant, so the
    # property HOLDS and the whole state space is explored (x0<=2 is refuted by both tools, so
    # it is not vacuous).
    "single":  ((4, 20), ("A[] x0 <= 6", "A[] x0 <= 6")),
    "viking":  ((4, 26), ("A[] not (Viking1.safe && Viking1.unsafe)", "A[] not (Viking1.safe && Viking1.unsafe)")),
    "train":   ((0,1), ("A[] not deadlock", "A[] not deadlock")),
    "hddi":    ((0, 20), ("A[] !((ST1==station_y_sync || ST1==station_y_async || ST1==station_z_async || ST1==station_z_sync) && (ST2==station_y_sync || ST2==station_y_async || ST2==station_z_async || ST2==station_z_sync))", "A[] !((ST1==station_y_sync || ST1==station_y_async || ST1==station_z_async || ST1==station_z_sync) && (ST2==station_y_sync || ST2==station_y_async || ST2==station_z_async || ST2==station_z_sync))")),
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
    qdir = os.path.join(args.outdir, "queries_linux" if args.linux else "queries_macos"); os.makedirs(qdir, exist_ok=True)
    sodir = os.path.join(args.outdir, "so_linux" if args.linux else "so_macos"); os.makedirs(sodir, exist_ok=True)

    made, failed = 0, []
    for fam in tqdm(want):
        if fam not in FAMILIES:
            print("unknown family %r (have: %s)" % (fam, ", ".join(sorted(FAMILIES))), file=sys.stderr)
            continue
        (lo, hi), queries = FAMILIES[fam]
        if args.range:
            lo, hi = args.range
        query = queries[1] if args.linux else queries[0]
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
