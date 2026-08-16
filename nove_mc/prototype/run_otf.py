#!/usr/bin/env python3
"""Differential + state-count harness for the on-the-fly liveness experiment.

Runs every query in queries.txt under every requested mode of gfp_proto and records
  verdict, symbolic states stored, states expanded, wall clock.
Modes disagreeing with the baseline are the refutation this harness exists to find.
"""
import argparse, os, re, subprocess, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BIN = os.path.join(HERE, "build", "gfp_proto")
MODELS = os.path.join(REPO, "benchmarks", "models")

VERDICT = re.compile(r"prototype leads_to_gfp\s*:\s*(UNUSABLE|SATISFIED|NOT)")
STORAGE = re.compile(r"nodes=(\d+) disc-keys=(\d+) zones/dk=([\d.]+) edges=(\d+)")
FIX = re.compile(r"rounds=(\d+) nonempty-nodes=(\d+) fed-width avg=([\d.]+) max=(\d+) capped=(\S+)")
EXPANDED = re.compile(r"expanded=(\d+) passes=(\d+) pre-calls=(\d+) early=(\S+)")
# "NOT satisfied" is two words, so `(\S+)` silently matched only the SATISFIED lines and the
# shipping engine's state count went missing on exactly the queries it mattered for.
REFV = re.compile(r"reference leads_to_net\s*:\s*(SATISFIED|NOT satisfied)\s*\(explored=(\d+)\)")


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


def run(model, p, q, mode, timeout, with_ref):
    env = dict(os.environ)
    env["PTCTLMC_GFPALU"] = "1"
    env["PTCTLMC_GFPMODE"] = mode
    if not with_ref:
        env["PROTO_NOREF"] = "1"
    xml = os.path.join(MODELS, model + ".xml")
    t0 = time.time()
    try:
        r = subprocess.run([BIN, xml, p, q], env=env, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"verdict": "TIMEOUT", "nodes": None, "expanded": None, "secs": timeout}
    dt = time.time() - t0
    o = r.stdout
    v = VERDICT.search(o)
    s = STORAGE.search(o)
    e = EXPANDED.search(o)
    fx = FIX.search(o)
    ref = REFV.search(o)
    return {
        "verdict": v.group(1) if v else "ERR",
        "ref": ref.group(1) if ref else None,
        "ref_explored": int(ref.group(2)) if ref else None,
        "nodes": int(s.group(1)) if s else None,
        "disc": int(s.group(2)) if s else None,
        "edges": int(s.group(4)) if s else None,
        "expanded": int(e.group(1)) if e else None,
        "rounds": int(fx.group(1)) if fx else None,
        "capped": fx.group(5) if fx else None,
        "secs": dt,
        "stderr": r.stderr[-400:] if r.returncode not in (0, 1) else "",
        "rc": r.returncode,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="base")
    ap.add_argument("--queries", default=os.path.join(HERE, "queries.txt"))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--ref", action="store_true", help="also run the shipping engine (slow)")
    ap.add_argument("--out", default=os.path.join(HERE, "otf_results.json"))
    ap.add_argument("--filter", default=None)
    a = ap.parse_args()

    modes = a.modes.split(",")
    qs = load_queries(a.queries)
    if a.filter:
        qs = [x for x in qs if a.filter in x[0]]
    results = {}
    for (model, p, q) in qs:
        key = f"{model} {p}-->{q}"
        results[key] = {}
        line = [f"{key:44s}"]
        for m in modes:
            r = run(model, p, q, m, a.timeout, a.ref)
            results[key][m] = r
            n = r["nodes"]
            line.append(f"{m}:{r['verdict'][:3]}/{n if n is not None else '-'}"
                        f"/{r['secs']:.2f}s")
            if a.ref and r.get("ref_explored") is not None:
                line.append(f"| shipping-engine:{r['ref'][:3]}/{r['ref_explored']}")
        print(" ".join(line), flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)

    # Differential report. UNUSABLE is the absence of a verdict, not a verdict: a capped
    # fixpoint has not converged, so it is neither an over- nor an under-approximation.
    # Comparing against it would manufacture fake disagreements and, worse, hide the case
    # where one algorithm answers a query the other cannot.
    if len(modes) > 1:
        base = modes[0]
        bad, rescued, lost = [], [], []
        for k, v in results.items():
            b = v[base]["verdict"]
            for m in modes[1:]:
                x = v[m]["verdict"]
                if b == "UNUSABLE" and x in ("SATISFIED", "NOT"):
                    rescued.append((k, m, x))
                elif x == "UNUSABLE" and b in ("SATISFIED", "NOT"):
                    lost.append((k, m, b))
                elif b != x and "UNUSABLE" not in (b, x):
                    bad.append((k, m, b, x))
        print("\n=== DISAGREEMENTS vs", base, "(real, both sides converged) ===")
        print("none" if not bad else "")
        for k, m, b, x in bad:
            print(f"  {k}  {base}={b}  {m}={x}")
        if rescued:
            print(f"\n=== ANSWERED where {base} was UNUSABLE (capped) ===")
            for k, m, x in rescued:
                print(f"  {k}  {m}={x}")
        if lost:
            print(f"\n=== UNUSABLE where {base} converged ===")
            for k, m, b in lost:
                print(f"  {k}  {base}={b}  {m}=UNUSABLE")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
