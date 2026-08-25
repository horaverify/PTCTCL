#!/usr/bin/env python3
"""Differential + state-count harness for `p --> q` on the SHIPPING binary.

Runs every query in queries.txt under every requested liveness arm of `ptctlmc` and
records verdict, symbolic states explored, and wall clock. Arms disagreeing with the
baseline are the refutation this harness exists to find.

    ./run_otf.py                          # otf only, the shipping default
    ./run_otf.py --arms otf,gfp,net       # the full differential
    ./run_otf.py --arms otf,net --filter csma

The arms are PTCTLMC_LIVENESS values; see include/ptctlmc/netreach.hpp.

  otf   on-the-fly backward greatest fixpoint          (leads_to_otf, the default)
  gfp   the whole-graph three-phase form               (leads_to_gfp)
  net   the original forward SCC search                (leads_to_net, the oracle)

This used to live in nove_mc/prototype/ and drive a separate `gfp_proto` binary linked
against a private copy of netreach.cpp. There is no second copy any more: the engine it
was prototyping IS the shipping engine, so the harness drives `ptctlmc` itself. Anything
this harness can no longer distinguish is something the two builds no longer differ on.
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
    cands = [os.path.join(REPO, d, "ptctlmc")
             for d in ("build", "build-freshmain", "build-fix", "build-base")]
    for c in cands:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    sys.exit("no ptctlmc binary found; pass --bin (looked in: %s)" % ", ".join(cands))


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


def run(binary, model, p, q, arm, timeout, backend):
    env = dict(os.environ)
    env["PTCTLMC_LIVENESS"] = arm
    xml = os.path.join(MODELS, model + ".xml")
    # The query goes in a file: `ptctlmc <model> <queries>` is the only shape the CLI takes.
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
    ap.add_argument("--arms", default="otf", help="comma-separated PTCTLMC_LIVENESS values; "
                                                  "the FIRST is the differential baseline")
    ap.add_argument("--queries", default=os.path.join(HERE, "queries.txt"))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--backend", default="seq", help="seq is reproducible; omp wobbles by a "
                                                     "few states run to run")
    ap.add_argument("--bin", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "otf_results.json"))
    ap.add_argument("--filter", default=None, help="substring match on the model name")
    a = ap.parse_args()

    binary = find_binary(a.bin)
    arms = a.arms.split(",")
    qs = load_queries(a.queries)
    if a.filter:
        qs = [x for x in qs if a.filter in x[0]]
    print("binary:", binary, "| backend:", a.backend, "| arms:", ",".join(arms),
          "|", len(qs), "queries\n")

    results = {}
    for (model, p, q) in qs:
        key = "%s %s-->%s" % (model, p, q)
        results[key] = {}
        line = ["%-44s" % key]
        for arm in arms:
            r = run(binary, model, p, q, arm, a.timeout, a.backend)
            results[key][arm] = r
            n = r["explored"]
            line.append("%s:%s/%s/%.2fs" % (arm, r["verdict"][:3],
                                            n if n is not None else "-", r["secs"]))
        print(" ".join(line), flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)

    rc = 0
    if len(arms) > 1:
        base = arms[0]
        bad = []
        for k, v in results.items():
            b = v[base]["verdict"]
            for arm in arms[1:]:
                x = v[arm]["verdict"]
                # TIMEOUT/UNSUPPORTED/ERR are the ABSENCE of a verdict, not a verdict.
                # Comparing against one would manufacture a fake disagreement and, worse,
                # hide the case where one arm answers a query the other cannot.
                if b in ("SATISFIED", "NOT") and x in ("SATISFIED", "NOT") and b != x:
                    bad.append((k, arm, b, x))
        print("\n=== DISAGREEMENTS vs %s (both sides answered) ===" % base)
        if not bad:
            print("none")
        for k, arm, b, x in bad:
            print("  %s  %s=%s  %s=%s" % (k, base, b, arm, x))
            rc = 1

        print("\n=== states explored, by arm ===")
        tot = {arm: 0 for arm in arms}
        for k, v in results.items():
            for arm in arms:
                if v[arm]["explored"]:
                    tot[arm] += v[arm]["explored"]
        for arm in arms:
            print("  %-4s %d" % (arm, tot[arm]))

    print("\nwrote", a.out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
