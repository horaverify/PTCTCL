#!/usr/bin/env python3
"""Run the same leads-to query set through UPPAAL's verifyta and record its own figures.

`States stored` is the like-for-like number against our stored symbolic states; `States
explored` is the like-for-like number against `expanded`. Defaults are left alone on purpose
-- the point of the comparison is what a user actually gets, not what a tuned invocation gets.
"""
import argparse, json, os, re, subprocess, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MODELS = os.path.join(REPO, "benchmarks", "models")
VERIFYTA = "/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta"

SAT = re.compile(r"Formula is (NOT )?satisfied")
STORED = re.compile(r"States stored\s*:\s*(\d+)")
EXPLORED = re.compile(r"States explored\s*:\s*(\d+)")


def load_queries(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and len(line.split()) == 3:
                out.append(tuple(line.split()))
    return out


def run(model, p, q, timeout, verifyta):
    with tempfile.NamedTemporaryFile("w", suffix=".q", delete=False) as fh:
        fh.write(f"{p} --> {q}\n")
        qf = fh.name
    try:
        t0 = time.time()
        try:
            r = subprocess.run([verifyta, "-u", "-s", os.path.join(MODELS, model + ".xml"), qf],
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"verdict": "TIMEOUT", "stored": None, "explored": None, "secs": timeout}
        dt = time.time() - t0
        o = r.stdout + r.stderr
        m = SAT.search(o)
        st, ex = STORED.search(o), EXPLORED.search(o)
        return {
            "verdict": ("NOT" if m.group(1) else "SATISFIED") if m else "ERR",
            "stored": int(st.group(1)) if st else None,
            "explored": int(ex.group(1)) if ex else None,
            "secs": dt,
            "raw": "" if m else o[-300:],
        }
    finally:
        os.unlink(qf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=os.path.join(HERE, "queries.txt"))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default=os.path.join(HERE, "uppaal.json"))
    ap.add_argument("--verifyta", default=VERIFYTA)
    a = ap.parse_args()

    res = {}
    for (model, p, q) in load_queries(a.queries):
        key = f"{model} {p}-->{q}"
        r = run(model, p, q, a.timeout, a.verifyta)
        res[key] = r
        print(f"{key:44s} {r['verdict'][:4]:<5} stored={r['stored']} explored={r['explored']} "
              f"{r['secs']:.2f}s", flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
