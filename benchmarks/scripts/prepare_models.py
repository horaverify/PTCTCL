#!/usr/bin/env python3
"""Materialise the benchmark set: models and (optionally) LTSmin .so modules.

Queries are NOT written here -- they are tracked under benchmarks/queries/; see its README.

The generators emit XML on stdout; this script drives them over a size range and
writes everything into a layout the harness understands:

    benchmarks/models/<family><N>.xml     HORA / verifyta input
    benchmarks/so_<plat>/<family><N>.so   opaal+LTSmin PINS module (with --so)

Building .so modules needs opaal, which is Python 2 (see gen_so_linux.sh/gen_so_macos.sh);
without it the three-way comparison degrades to verifyta vs HORA, which the harness
handles by skipping the LTSmin rows rather than failing.

    python3 prepare_models.py                 # models for every family (auto-detect platform)
    python3 prepare_models.py --so            # also build LTSmin modules
    python3 prepare_models.py --linux         # force the Linux .so toolchain
    python3 prepare_models.py --only phil,fischer --range 4 10
"""
import argparse, os, subprocess, sys
from tqdm import tqdm
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(SCRIPTS)  # benchmarks/
# generators/ sits next to this file, under scripts/ -- NOT under benchmarks/. This pointed at
# benchmarks/generators, which does not exist, so every family failed with "no generator". It went
# unnoticed because the script was invoked as --queries-only, which returns before reaching it.
GEN = os.path.join(SCRIPTS, "generators")


# --------------------------------------------------------------------------- size ladders
# Family -> the (lo, hi) size range its generator can produce.
#
# The QUERIES are not here. They live in benchmarks/queries/<kind>/<family>.q, are tracked, and
# are hand-curated -- see benchmarks/queries/README.md for why each family has the query it has,
# including the shapes that were measured and rejected. They used to be written from a table in
# this file, which drifted: by the time it was removed five entries disagreed with what had been
# measured for weeks, and re-running this script would have silently reverted grc and hddi to
# queries that are falsified or that verifyta rejects outright.
FAMILIES = {
    "fischer": (4, 20),
    "phil":    (4, 20),
    "twolock": (4, 20),
    "arbiter": (4, 25),
    "csma":    (4, 22),
    "grc":     (4, 20),
    "single":  (4, 20),
    "viking":  (4, 26),
    "train":   (0, 1),
    "hddi":    (0, 20),
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
    ap.add_argument("--linux", action="store_true", help="use the Linux .so toolchain")
    ap.add_argument("--macos", action="store_true", help="use the macOS .so toolchain")
    args = ap.parse_args()

    # Auto-detect platform if not specified
    if not args.linux and not args.macos:
        args.linux = sys.platform != "darwin"
        args.macos = sys.platform == "darwin"

    want = [f.strip() for f in args.only.split(",") if f.strip()] or sorted(FAMILIES)
    mdir = os.path.join(args.outdir, "models", "xml"); os.makedirs(mdir, exist_ok=True)
    sodir = os.path.join(args.outdir, "models", "so_linux" if args.linux else "so_macos"); os.makedirs(sodir, exist_ok=True)

    made, failed = 0, []
    for fam in tqdm(want):
        if fam not in FAMILIES:
            print("unknown family %r (have: %s)" % (fam, ", ".join(sorted(FAMILIES))), file=sys.stderr)
            continue
        lo, hi = FAMILIES[fam]
        if args.range:
            lo, hi = args.range
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
    print("generated %d models into %s" % (made, mdir))
    if failed:
        print("\n%d issue(s):" % len(failed))
        for f in failed[:15]:
            print("  " + f)


if __name__ == "__main__":
    main()
