#!/usr/bin/env python3
"""Benchmark E[] queries: PTCTLMC against UPPAAL, across model sizes.

    python3 benchmarks/scripts/bench_eg.py [--sizes 4,5,6] [--repeat 3] [--timeout 120]

Reports, per (model, query): verdict from each tool, PTCTLMC's stored-state count, and best-of-N
wall time for each. A verdict mismatch is flagged; a PTCTLMC `UNSUPPORTED` is reported as `unsup`
and is a refusal, not a wrong answer.
"""
import argparse, pathlib, re, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
PTCTLMC = ROOT / "build-freshmain" / "ptctlmc"
VERIFYTA = pathlib.Path(
    "/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta")
FAMILIES = ["fischer", "arbiter", "csma", "phil", "grc", "twolock"]


def verdict(out):
    if out is None:
        return "timeout"
    if "-- Formula is NOT satisfied" in out:
        return "NOT"
    if "-- Formula is satisfied" in out:
        return "SAT"
    if "UNSUPPORTED" in out:
        return "unsup"
    return "err"


def run(cmd, timeout):
    t0 = time.time()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return None, timeout
    return out, time.time() - t0


def best(cmd, timeout, repeat):
    out, lo = run(cmd, timeout)
    for _ in range(repeat - 1):
        o2, t2 = run(cmd, timeout)
        lo = min(lo, t2)
        out = out if out is not None else o2
    return out, lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="4,5,6")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--no-uppaal", action="store_true")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    if not PTCTLMC.exists():
        sys.exit(f"missing {PTCTLMC}; build it first")
    use_u = VERIFYTA.exists() and not args.no_uppaal
    qf = pathlib.Path("/tmp/_bench_eg.q")

    print(f"{'model':<10} {'ptctlmc':>8} {'uppaal':>8} {'states':>10} "
          f"{'t_ptctlmc':>10} {'t_uppaal':>10}  query", flush=True)
    mismatch = 0
    for fam in FAMILIES:
        qs = [q for q in (BENCH / "eg_queries" / f"{fam}.q").read_text().split("\n") if q.strip()]
        for n in sizes:
            model = BENCH / "models" / f"{fam}{n}.xml"
            if not model.exists():
                continue
            for q in qs:
                qf.write_text(q + "\n")
                o, ta = best([str(PTCTLMC), "--backend", "seq", str(model), str(qf)],
                             args.timeout, args.repeat)
                va = verdict(o)
                m = re.search(r"explored (\d+)", o or "")
                states = m.group(1) if m else "-"
                vb, tb = "-", 0.0
                if use_u:
                    o2, tb = best([str(VERIFYTA), "-q", str(model), str(qf)],
                                  args.timeout, args.repeat)
                    vb = verdict(o2)
                flag = ""
                if use_u and va not in ("unsup", "timeout") and vb not in ("timeout",) and va != vb:
                    flag = "   <-- MISMATCH"
                    mismatch += 1
                # flushed per row: a long run should stream, not appear all at once at the end
                print(f"{fam+str(n):<10} {va:>8} {vb:>8} {states:>10} "
                      f"{ta:>10.3f} {tb:>10.3f}  {q}{flag}", flush=True)
    print(f"\nverdict mismatches: {mismatch}")


if __name__ == "__main__":
    main()
