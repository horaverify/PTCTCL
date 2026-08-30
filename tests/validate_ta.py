#!/usr/bin/env python3
"""Differential tester for HORA vs verifyta (UPPAAL) on timed automata.

Runs both tools on a model.xml + query.q and compares per-query verdicts (in file
order). Unsupported HORA queries are reported as skips, not mismatches.

Usage:
  validate_ta.py model.xml query.q --hora ./build/hora --verifyta <path>
"""
import argparse, re, subprocess, sys

SAT = re.compile(r"Formula is (satisfied|NOT satisfied)", re.IGNORECASE)


def run(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    return p.stdout + p.stderr, None


def verdicts(out):
    res = []
    for m in SAT.finditer(out):
        res.append(m.group(1).lower().startswith("satisf") and "sat" or "unsat")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("query")
    ap.add_argument("--hora", required=True)
    ap.add_argument("--verifyta", required=True)
    args = ap.parse_args()

    pout, perr = run([args.hora, args.model, args.query])
    vout, verr = run([args.verifyta, "-q", args.model, args.query])
    if perr or verr:
        print(f"tool error: hora={perr} verifyta={verr}")
        sys.exit(2)

    pv = verdicts(pout)
    vv = verdicts(vout)
    # HORA also emits UNSUPPORTED lines; align by walking its raw lines.
    p_lines = [l for l in pout.splitlines() if "Formula is" in l]
    n = max(len(p_lines), len(vv))
    ok = bad = skip = 0
    print(f"=== {args.model} / {args.query} ===")
    vi = 0
    for i, pl in enumerate(p_lines):
        v = vv[vi] if vi < len(vv) else "?"
        if "UNSUPPORTED" in pl:
            print(f"  [{i}] SKIP (hora unsupported)   verifyta={v}")
            skip += 1
            vi += 1
            continue
        p = "sat" if "is satisfied" in pl.lower() else "unsat"
        if p == v:
            print(f"  [{i}] OK   {p}")
            ok += 1
        else:
            print(f"  [{i}] DIFF hora={p} verifyta={v}")
            bad += 1
        vi += 1
    print(f"result: ok={ok} diff={bad} skip={skip}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
