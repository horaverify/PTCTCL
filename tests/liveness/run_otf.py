#!/usr/bin/env python3
"""Verdict + state-count harness for `p --> q` on the shipping binary.

Runs every query in queries.txt and records verdict, symbolic states explored, and
wall clock, so a change to the liveness engine can be diffed against a recorded run.

    ./run_otf.py                          # every query
    ./run_otf.py --filter csma            # one family

There is ONE liveness engine: `leads_to_otf`, the on-the-fly backward greatest
fixpoint. It used to share the binary with two other implementations selected by
HORA_LIVENESS -- `gfp` (the whole-graph three-phase form) and `net` (the original
forward SCC search) -- which this harness ran side by side looking for a disagreement.
It never found one: the last full differential, 46 queries over 7 model families,
agreed on every query that all three answered, and `net` timed out on the one
(grc4 T0.in-->T0.far) it could not. Both were deleted with that result recorded.

What still checks the verdicts, now that the arms are gone:
  * tests/models/liveness-regressions/ -- four models whose correct answers are known
    independently of this engine (each was found by a wrong verdict).
  * tests/validate_liveness.sh -- E[]/A<> differential against verifyta.
  * this file, as a state-count regression against a saved run.
"""
import argparse, json, os, re, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MODELS = os.path.join(REPO, "benchmarks", "models")

# "-- Formula is satisfied.  [explored 128 symbolic states]" / "is NOT satisfied."
# "NOT satisfied" is two words: a `(\S+)` here silently matched only the satisfied lines,
# and the state count went missing on exactly the queries it mattered for.
VERDICT = re.compile(r"Formula is (satisfied|NOT satisfied)")
EXPLORED = re.compile(r"\[explored (\d+) symbolic states\]")
UNSUPPORTED = re.compile(r"Formula is UNSUPPORTED")


def find_binary(explicit):
    if explicit:
        return explicit
    cands = [os.path.join(REPO, d, "hora")
             for d in ("build", "build-freshmain", "build-fix", "build-base")]
    for c in cands:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    sys.exit("no hora binary found; pass --bin (looked in: %s)" % ", ".join(cands))


def load_queries(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 3:
                out.append(tuple(parts))
    return out


def run(binary, model, p, q, timeout, backend):
    env = dict(os.environ)
    xml = os.path.join(MODELS, model + ".xml")
    # The query goes in a file: `hora <model> <queries>` is the only shape the CLI takes.
    with tempfile.NamedTemporaryFile("w", suffix=".q", delete=False) as qf:
        qf.write("%s --> %s\n" % (p, q))
        qpath = qf.name
    t0 = time.time()
    try:
        r = subprocess.run([binary, "--backend", backend, xml, qpath],
                           env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"verdict": "TIMEOUT", "explored": None, "secs": timeout, "rc": None}
    finally:
        os.unlink(qpath)
    dt = time.time() - t0
    o = r.stdout
    if UNSUPPORTED.search(o):
        return {"verdict": "UNSUPPORTED", "explored": None, "secs": dt, "rc": r.returncode}
    v = VERDICT.search(o)
    e = EXPLORED.search(o)
    return {
        "verdict": ("SATISFIED" if v.group(1) == "satisfied" else "NOT") if v else "ERR",
        "explored": int(e.group(1)) if e else None,
        "secs": dt,
        "rc": r.returncode,
        "stderr": r.stderr[-400:] if r.returncode != 0 else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=os.path.join(HERE, "queries.txt"))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--backend", default="seq", help="seq is reproducible; omp wobbles by a "
                                                     "few states run to run")
    ap.add_argument("--bin", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "otf_results.json"))
    ap.add_argument("--filter", default=None, help="substring match on the model name")
    a = ap.parse_args()

    binary = find_binary(a.bin)
    qs = load_queries(a.queries)
    if a.filter:
        qs = [x for x in qs if a.filter in x[0]]
    print("binary:", binary, "| backend:", a.backend, "|", len(qs), "queries\n")

    results = {}
    for (model, p, q) in qs:
        key = "%s %s-->%s" % (model, p, q)
        r = run(binary, model, p, q, a.timeout, a.backend)
        results[key] = r
        n = r["explored"]
        print("%-44s %s/%s/%.2fs" % (key, r["verdict"][:3],
                                     n if n is not None else "-", r["secs"]), flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)

    total = sum(r["explored"] for r in results.values() if r["explored"])
    print("\nstates explored: %d" % total)
    rc = 0

    print("\nwrote", a.out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
