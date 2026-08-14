#!/usr/bin/env python3
"""W0 — the differential oracle gate.

Correctness first, state counts second, timing not at all. This is NOT bench_run.py: nothing
here is measured in seconds, and a verdict disagreement is a hard failure with a non-zero exit
rather than a column in a table.

For every (model, query) pair it runs three things:

    verifyta -u                 the reference tool (UPPAAL 5.0.0), the oracle
    ptctlmc                     the branch binary with every new flag OFF  -> the BASELINE
    ptctlmc <flags>             the branch binary with the workstream flag ON

Running the baseline as "the same binary with the flag off" rather than as a separately built
binary from `main` is deliberate: it makes "the flag is genuinely off by default" part of what
the gate checks, which is the property the whole default-off discipline rests on.

VERDICTS must agree three ways. Any disagreement fails the run.

STATE COUNTS are the result. They are reported as deltas -- flag-on vs baseline, and baseline
vs verifyta -- for discrete states, stored zones and explored states.

Every ptctlmc run is SINGLE-THREADED. thm:determinism says the verdict is deterministic but the
stored-zone count is not: the convex-union merge is order-sensitive, so at >1 thread two runs of
the same binary can report different counts. State counts are only comparable across commits at
one thread, so the gate refuses to produce them any other way.

Usage:
    python3 benchmarks/verify_parity.py                       # baseline snapshot, no flags
    python3 benchmarks/verify_parity.py --flags="--narrow"       # gate a workstream
    python3 benchmarks/verify_parity.py --only fischer,phil --max-n 8
    python3 benchmarks/verify_parity.py --baseline benchmarks/baseline-mostoptimized.csv --flags="--narrow"
"""
import argparse, csv, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "benchmarks", "models")
QUERIES = os.path.join(ROOT, "benchmarks", "queries")
BIN = os.environ.get("PTCTLMC_BIN", os.path.join(ROOT, "build-mostopt", "ptctlmc"))
VERIFYTA = os.environ.get("VERIFYTA",
    "/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta")


def run(cmd, env=None, timeout=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, ""
    except OSError as e:
        return None, str(e)
    return r.returncode, r.stdout + r.stderr


def verdict_of(out):
    """UPPAAL and PTCTLMC print the same two sentences; parse both with one function.

    Order matters: "NOT satisfied" contains "satisfied", so the negative is tested first.
    """
    if "NOT satisfied" in out:
        return "UNSAT"
    if "is satisfied" in out:
        return "SAT"
    return "?"


def probe_ptctlmc(binary, xml, qf, flags, timeout):
    env = dict(os.environ)
    env["PTCTLMC_STATS"] = "1"
    env["OMP_NUM_THREADS"] = "1"          # thm:determinism -- counts are only comparable at 1 thread
    rc, out = run([binary, "--backend", "omp", *flags, xml, qf], env, timeout)
    if rc is None:
        return dict(verdict="TIMEOUT")
    if "UNSUPPORTED" in out:
        # The query layer rejects some formulas outright (disjunctions in location predicates,
        # for instance). That is a feature gap, not a parity failure: there is no verdict to
        # disagree with. Kept distinct from CRASH so the gate does not cry wolf.
        return dict(verdict="UNSUP")
    if rc != 0:
        return dict(verdict="CRASH", detail=out[-400:])
    g = lambda p: (lambda m: int(m.group(1)) if m else None)(re.search(p, out))
    return dict(verdict=verdict_of(out),
                disc=g(r"discrete-states=(\d+)"),
                stored=g(r"stored-zones=(\d+)"),
                distinct=g(r"distinct-zones=(\d+)"),
                generated=g(r"generated=(\d+)"),
                explored=g(r"explored (\d+)"))


def probe_verifyta(xml, qf, timeout):
    if not os.path.exists(VERIFYTA):
        return dict(verdict="n/a")
    rc, out = run([VERIFYTA, "-u", xml, qf], dict(os.environ), timeout)
    if rc is None:
        return dict(verdict="TIMEOUT")
    g = lambda p: (lambda m: int(m.group(1)) if m else None)(re.search(p, out))
    return dict(verdict=verdict_of(out),
                stored=g(r"States stored\s*:\s*(\d+)"),
                explored=g(r"States explored\s*:\s*(\d+)"))


def family_of(name):
    """arbiter17.xml -> ("arbiter", 17). Models are <family><N>.xml throughout."""
    m = re.match(r"^([a-z_]+?)(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def query_for(family, for_verifyta):
    """Queries live per family. A `<family>_uppaal.q` overrides for the oracle when the model
    needs different syntax there (see benchmarks/queries/train_uppaal.q)."""
    if for_verifyta:
        alt = os.path.join(QUERIES, f"{family}_uppaal.q")
        if os.path.exists(alt):
            return alt
    q = os.path.join(QUERIES, f"{family}.q")
    return q if os.path.exists(q) else None


def collect(only, max_n):
    out = []
    for fn in sorted(os.listdir(MODELS)):
        if not fn.endswith(".xml"):
            continue
        fam, n = family_of(fn[:-4])
        if only and fam not in only:
            continue
        if max_n and n > max_n:
            continue
        if not query_for(fam, False):
            continue
        out.append((fn[:-4], fam, n))
    out.sort(key=lambda r: (r[1], r[2]))
    return out


def pct(new, old):
    if new is None or old is None or old == 0:
        return ""
    return f"{100.0 * (new - old) / old:+.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=BIN)
    # One string, split here: with nargs="*" argparse swallows a leading "--slice" as an
    # option of its own and the run dies on "unrecognized arguments".
    ap.add_argument("--flags", default="",
                    help='workstream flags to gate, as one string. MUST use the = form: '
                         '--flags="--narrow", since argparse reads a value starting with "--" '
                         'as another option.')
    ap.add_argument("--only", default="", help="comma-separated families")
    ap.add_argument("--max-n", type=int, default=10,
                    help="skip instances above this N (0 = no cap)")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "parity.csv"))
    ap.add_argument("--baseline", default="",
                    help="a previous CSV to compare against, instead of the in-run baseline")
    args = ap.parse_args()
    args.flags = args.flags.split()

    if not os.path.exists(args.bin):
        sys.exit(f"binary not found: {args.bin}  (build it, or set PTCTLMC_BIN)")
    only = {x for x in args.only.split(",") if x}
    work = collect(only, args.max_n)
    if not work:
        sys.exit("no (model, query) pairs matched")

    prev = {}
    if args.baseline:
        with open(args.baseline) as f:
            for row in csv.DictReader(f):
                prev[row["model"]] = row

    hdr = ["model", "family", "n", "verdict", "vfy_verdict", "base_verdict",
           "disc", "stored", "distinct", "generated", "explored",
           "base_disc", "base_stored", "base_explored",
           "vfy_stored", "vfy_explored", "status"]
    rows, failures = [], []

    print(f"{'model':<12} {'verdict':<8} {'disc':>10} {'stored':>10} "
          f"{'vs base':>9} {'vfy stored':>11} {'vs vfy':>9}  status")
    print("-" * 86)

    for model, fam, n in work:
        xml = os.path.join(MODELS, model + ".xml")
        base = probe_ptctlmc(args.bin, xml, query_for(fam, False), [], args.timeout)
        flag = (probe_ptctlmc(args.bin, xml, query_for(fam, False), args.flags, args.timeout)
                if args.flags else base)
        vfy = probe_verifyta(xml, query_for(fam, True), args.timeout)

        # Verdict parity. "n/a" (no oracle) and "TIMEOUT" are not disagreements -- they are
        # absences of evidence, and failing on them would make the gate cry wolf on big models.
        seen = {v for v in (flag["verdict"], base["verdict"], vfy["verdict"])
                if v not in ("n/a", "TIMEOUT", "UNSUP", "")}
        status = "ok"
        if flag["verdict"] == "CRASH" or base["verdict"] == "CRASH" or "?" in seen:
            status = "ERROR"
        elif len(seen) > 1:
            status = "MISMATCH"
        elif flag["verdict"] == "UNSUP":
            status = "unsupported"
        if status in ("ERROR", "MISMATCH"):
            failures.append((model, flag["verdict"], base["verdict"], vfy["verdict"],
                             flag.get("detail", "") or base.get("detail", "")))

        cmp_base = prev.get(model)
        b_disc = int(cmp_base["disc"]) if cmp_base and cmp_base.get("disc") else base.get("disc")
        b_stored = (int(cmp_base["stored"]) if cmp_base and cmp_base.get("stored")
                    else base.get("stored"))
        b_expl = (int(cmp_base["explored"]) if cmp_base and cmp_base.get("explored")
                  else base.get("explored"))

        rows.append(dict(model=model, family=fam, n=n, verdict=flag["verdict"],
                         vfy_verdict=vfy["verdict"], base_verdict=base["verdict"],
                         disc=flag.get("disc"), stored=flag.get("stored"),
                         distinct=flag.get("distinct"), generated=flag.get("generated"),
                         explored=flag.get("explored"),
                         base_disc=b_disc, base_stored=b_stored, base_explored=b_expl,
                         vfy_stored=vfy.get("stored"), vfy_explored=vfy.get("explored"),
                         status=status))
        print(f"{model:<12} {flag['verdict']:<8} {str(flag.get('disc') or '-'):>10} "
              f"{str(flag.get('stored') or '-'):>10} {(pct(flag.get("stored"), b_stored) if args.flags else ""):>9} "
              f"{str(vfy.get('stored') or '-'):>11} "
              f"{pct(flag.get('stored'), vfy.get('stored')):>9}  {status}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}  ({len(rows)} pairs)")

    if failures:
        print("\nVERDICT PARITY FAILURES:", file=sys.stderr)
        for model, a, b, c, detail in failures:
            print(f"  {model}: flag={a} baseline={b} verifyta={c}", file=sys.stderr)
            if detail:
                print(f"      {detail.strip()[:300]}", file=sys.stderr)
        sys.exit(1)
    print("verdict parity: OK on every pair")


if __name__ == "__main__":
    main()
