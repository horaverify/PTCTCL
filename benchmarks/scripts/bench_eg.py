#!/usr/bin/env python3
"""bench_eg.py -- the `E[]` sweep: PTCTLMC vs UPPAAL vs TChecker.

    python3 benchmarks/scripts/bench_eg.py [--sizes 4-12] [--tools ptctlmc,verifyta,tchecker]
        [--repeat 2] [--timeout 600] [--outdir ../results_liveness_e] [--cores 16-31] [--detach]

Reports, per (model, query, tool): the VERDICT, the tool's own state count, wall
time and peak RSS. A verdict disagreement between two tools that both answered is
flagged -- that is what the suite is for, and it has caught wrong answers before.

MEASUREMENT is bench.py's `measure()`, imported rather than re-implemented: own
process group, /usr/bin/time, group-wide RSS guard, group kill on timeout.

RESUME: a row is (model, qidx, tool, threads, rep); rows that finished
definitively (OK/TIMEOUT/MEMOUT) are seeded from the CSV and never re-run, so an
interrupted sweep continues where it stopped. Sizes are swept outermost, so a
complete cheap table exists for every family before the expensive tails start,
and --strike-limit stops growing a (family, query, tool) once it has timed out.

THE STATE COUNTS ARE NOT COMPARABLE ACROSS TOOLS, the times are: PTCTLMC computes
the whole greatest fixpoint, TChecker's couvscc stops at the first cycle it finds
(fischer4 q0: 248 vs 5). Both decide the same question; only one of them has to
finish exploring to do it.

TCHECKER runs the psi-restricted models from gen_tck_eg.py (`tck_live_e/`), which
is generated on demand. It cannot encode the non-convex clock rows at all -- those
are reported `n/a`, with the reason in tck_live_e/manifest.json.
"""
import argparse, csv, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench import measure, PLAT                                     # noqa: E402

BENCH = os.path.dirname(HERE)
PROJ = os.path.dirname(BENCH)
MODELS = os.path.join(BENCH, "models")
QDIR = os.path.join(BENCH, "eg_queries")
TCK_EG = os.path.join(BENCH, "tck_live_e")
LABEL = "TC_EG"

FAMILIES = ["fischer", "arbiter", "csma", "phil", "grc", "twolock"]
CSV_COLS = ["family", "model", "size", "qidx", "tool", "threads", "rep", "verdict",
            "states", "seconds", "rss_gb", "status", "query"]

DEF_PTCTLMC = os.environ.get("PTCTLMC_BIN") or os.path.join(
    PROJ, "build_shipping" if PLAT == "linux" else "build-mac", "ptctlmc")
DEF_VERIFYTA = os.environ.get("VERIFYTA_BIN") or (
    "/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta"
    if PLAT == "macos" else "verifyta")
DEF_TCK = os.environ.get("TCHECKER_LIVENESS_BIN") or os.path.join(
    os.path.dirname(PROJ), "tchecker", "install", "bin", "tck-liveness")


def verdict_of(tool, out):
    """SAT | NOT | unsup | -- what the tool decided, independent of how it exited."""
    if tool == "tchecker":
        if "CYCLE true" in out:
            return "SAT"
        if "CYCLE false" in out:
            return "NOT"
        return "-"
    if "UNSUPPORTED" in out:
        return "unsup"
    if "-- Formula is NOT satisfied" in out or "Formula is NOT satisfied" in out:
        return "NOT"
    if "-- Formula is satisfied" in out or "Formula is satisfied" in out:
        return "SAT"
    return "-"


STATES_RE = {"ptctlmc": r"explored ([0-9]+) symbolic",
             "verifyta": r"States explored\s*:\s*([0-9]+)",
             "tchecker": r"VISITED_STATES\s+([0-9]+)"}


def parse_sizes(spec):
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            lo, _, hi = tok.partition("-")
            out += list(range(int(lo), int(hi) + 1))
        elif tok:
            out.append(int(tok))
    return sorted(set(out))


def queries_of(fam):
    path = os.path.join(QDIR, "%s.q" % fam)
    if not os.path.exists(path):
        return []
    return [q.strip() for q in open(path).read().splitlines() if q.strip()]


def tck_model(fam, n, idx):
    return os.path.join(TCK_EG, "%s%d__q%d.tck" % (fam, n, idx))


def argv_for(a, tool, xml, qfile, fam, n, idx, th):
    if tool == "ptctlmc":
        if th > 1:
            return [a.ptctlmc_bin, "--backend", "omp", xml, qfile]
        return [a.ptctlmc_bin, "--backend", "seq", xml, qfile]
    if tool == "verifyta":
        # NO `-s`. `--silence-progress` zeroes UPPAAL's state counter on LIVENESS
        # queries -- `E[]` reports 0 for every row, and `p --> q` under-reports
        # (arbiter4 `C0.cs --> C0.idle`: 506 with it, 710 without). Reachability is
        # unaffected (48 either way), which is why it went unnoticed.
        return [a.verifyta_bin, "-u", xml, qfile]
    return [a.tchecker_bin, "-a", "couvscc", "-l", LABEL, tck_model(fam, n, idx)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="4-12")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--tools", default="ptctlmc,verifyta,tchecker")
    ap.add_argument("--repeat", type=int, default=2, help="reps per row (best-of is read off the CSV)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--memgb", type=float, default=40.0)
    ap.add_argument("--strike-limit", type=int, default=1,
                    help="skip larger sizes of a (family, query, tool) after this many "
                         "consecutive timeouts (0 disables)")
    ap.add_argument("--threads", nargs="+", type=int, default=[1],
                    help="ptctlmc thread counts; 1 means --backend seq")
    ap.add_argument("--outdir", default=os.path.join(BENCH, "results_liveness_e"))
    ap.add_argument("--cores", default=None,
                    help="taskset core list for every measured run (e.g. 16-31), so a "
                         "sweep can share the machine with another one")
    ap.add_argument("--regen", type=int, default=1, help="regenerate tck_live_e first")
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--ptctlmc-bin", default=DEF_PTCTLMC)
    ap.add_argument("--verifyta-bin", default=DEF_VERIFYTA)
    ap.add_argument("--tchecker-bin", default=DEF_TCK)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    if a.detach:
        log_path = os.path.join(a.outdir, "driver.log")
        argv = [sys.executable, os.path.abspath(__file__)] + \
               [x for x in sys.argv[1:] if x != "--detach"]
        with open(log_path, "a") as lg:
            p = subprocess.Popen(["setsid", "nohup"] + argv, stdout=lg,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        print("detached pid %d -> %s" % (p.pid, log_path))
        return 0

    tools = [t.strip() for t in a.tools.split(",") if t.strip()]
    fams = [f.strip() for f in a.families.split(",") if f.strip()]
    sizes = parse_sizes(a.sizes)
    stop_file = os.path.join(a.outdir, "STOP")

    def log(msg):
        print("[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg), flush=True)

    with open(os.path.join(a.outdir, "bench.pid"), "w") as fh:
        fh.write("%d\n" % os.getpid())
    with open(os.path.join(a.outdir, "bench.pgid"), "w") as fh:
        fh.write("%d\n" % os.getpgid(0))

    if a.regen and "tchecker" in tools:
        log("regenerating the psi-restricted TChecker models")
        subprocess.run([sys.executable, os.path.join(HERE, "gen_tck_eg.py")], check=True)
    refused = {}
    man = os.path.join(TCK_EG, "manifest.json")
    if os.path.exists(man):
        refused = json.load(open(man)).get("refused", {})

    log("bench_eg  tools=%s sizes=%s families=%s reps=%d timeout=%ds memgb=%g cores=%s"
        % (",".join(tools), a.sizes, ",".join(fams), a.repeat, a.timeout, a.memgb, a.cores))
    for repo, path in (("PTCTCL", PROJ), ("HDBM", os.path.join(PROJ, "HDBM"))):
        try:
            h = subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
            log("%s @ %s" % (repo, h))
        except Exception:
            pass

    out_csv = os.path.join(a.outdir, "eg_results.csv")
    done, timeouts = {}, {}
    if os.path.exists(out_csv):
        for r in csv.DictReader(open(out_csv)):
            if r.get("status") in ("OK", "TIMEOUT", "MEMOUT"):
                done[(r["model"], r["qidx"], r["tool"], r["threads"], r["rep"])] = r
            if r.get("status") in ("TIMEOUT", "MEMOUT"):
                timeouts.setdefault((r["family"], r["qidx"], r["tool"], r["threads"]),
                                    set()).add(int(r["size"]))
    log("seeded %d completed runs from %s" % (len(done), out_csv))

    new = not os.path.exists(out_csv)
    fh = open(out_csv, "a")
    w = csv.writer(fh)
    if new:
        w.writerow(CSV_COLS)
        fh.flush()

    qdir = os.path.join(a.outdir, "q")
    os.makedirs(qdir, exist_ok=True)
    pre = ["taskset", "-c", a.cores] if a.cores else []
    verdicts = {}

    def struck(fam, idx, tool, th, n):
        if a.strike_limit <= 0:
            return False
        prev = sorted(x for x in timeouts.get((fam, str(idx), tool, str(th)), ()) if x < n)
        streak, m = 0, n - 1
        for x in reversed(prev):
            if x != m:
                break
            streak, m = streak + 1, m - 1
        return streak >= a.strike_limit

    print("%-11s %-4s %-9s %-4s %-6s %10s %10s %8s  %s"
          % ("model", "q", "tool", "th", "verdict", "states", "secs", "status", "query"),
          flush=True)
    for n in sizes:
        for fam in fams:
            xml = os.path.join(MODELS, "%s%d.xml" % (fam, n))
            if not os.path.exists(xml):
                continue
            for idx, q in enumerate(queries_of(fam)):
                if os.path.exists(stop_file):
                    log("STOP present -- exiting."); fh.close(); return 0
                qfile = os.path.join(qdir, "%s%d_q%d.q" % (fam, n, idx))
                with open(qfile, "w") as qh:
                    qh.write(q + "\n")
                model = "%s%d" % (fam, n)
                for tool in tools:
                    if tool == "tchecker" and not os.path.exists(tck_model(fam, n, idx)):
                        why = refused.get("%s q%d" % (fam, idx), {}).get(
                            "reason", "no psi-restricted model")
                        print("  %-11s q%-3d %-9s %-4s %-6s %10s %10s %8s  %s"
                              % (model, idx, tool, "-", "n/a", "-", "-", "SKIP", why),
                              flush=True)
                        continue
                    ths = a.threads if tool == "ptctlmc" else [1]
                    for th in ths:
                        if struck(fam, idx, tool, th, n):
                            print("  %-11s q%-3d %-9s %-4d %-6s %10s %10s %8s  %s"
                                  % (model, idx, tool, th, "-", "-", "-", "SKIP",
                                     "timed out at a smaller size"), flush=True)
                            continue
                        timed_out = False
                        for rep in range(1, a.repeat + 1):
                            key = (model, str(idx), tool, str(th), str(rep))
                            if key in done:
                                r = done[key]
                                verdicts.setdefault((model, idx), {})[tool] = r["verdict"]
                                timed_out = timed_out or r["status"] in ("TIMEOUT", "MEMOUT")
                                continue
                            if timed_out:
                                continue
                            env = {"OMP_NUM_THREADS": str(th)} if tool == "ptctlmc" else {}
                            cmd = pre + argv_for(a, tool, xml, qfile, fam, n, idx, th)
                            secs, rss, st, out = measure(cmd, env, a.timeout, a.memgb)
                            v = verdict_of(tool, out) if st == "OK" else (
                                "timeout" if st == "TIMEOUT" else
                                "memout" if st == "MEMOUT" else verdict_of(tool, out))
                            g = re.search(STATES_RE[tool], out)
                            states = g.group(1) if g else "-"
                            if st == "OK" and v == "-":
                                st = "ERR"          # exited 0 having decided nothing
                            w.writerow([fam, model, n, idx, tool, th, rep, v, states,
                                        "%.3f" % secs, "%.3f" % rss, st, q])
                            fh.flush()
                            print("  %-11s q%-3d %-9s %-4d %-6s %10s %10.2f %8s  %s"
                                  % (model, idx, tool, th, v, states, secs, st, q),
                                  flush=True)
                            if st == "OK":
                                verdicts.setdefault((model, idx), {})[tool] = v
                            timed_out = timed_out or st in ("TIMEOUT", "MEMOUT")
                            if timed_out:
                                timeouts.setdefault((fam, str(idx), tool, str(th)),
                                                    set()).add(n)
                seen = {t: v for t, v in verdicts.get((model, idx), {}).items()
                        if v in ("SAT", "NOT")}
                if len(set(seen.values())) > 1:
                    print("  *** MISMATCH %s q%d %s   %s"
                          % (model, idx, seen, q), flush=True)
    fh.close()
    log("ALL DONE -- %s" % out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
