#!/usr/bin/env python3
"""Evaluation harness: UPPAAL verifyta vs opaal+LTSmin vs PTCTLMC (OpenMP).

Per run: 30 min wall timeout, 15 GB RSS guard (whole process group). A run exceeding
either is recorded as TIMEOUT / MEMOUT rather than a number. Results stream to
raw.csv after every run so progress is watchable live.

Already-completed (model, tool, threads, rep) pairs with status OK are SKIPPED, and are
seeded from any earlier CSV given with --seed, so nothing is measured twice.
"""
import argparse, csv, os, re, signal, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "LTStest"))   # opaal+LTSmin tree
VIK  = os.path.join(ROOT, "src/opaal-ltsmin-succgen/tests/vikingtests")
MOD  = os.path.join(ROOT, "models")
SO   = os.path.join(ROOT, "so-macos")
OPAAL   = os.path.join(ROOT, "deps-macos/bin/opaal2lts-mc")
PTCTLMC = os.path.abspath(os.path.join(HERE, "..", "build-mac", "ptctlmc"))
VERIFYTA = "/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta"
TABLE_BITS = 26

QDIR = os.path.join(HERE, "queries"); os.makedirs(QDIR, exist_ok=True)
def q(name, text):
    p = os.path.join(QDIR, name)
    open(p, "w").write(text + "\n")
    return p
VIK_Q = q("viking.q", "A[] not (Viking1.safe && Viking1.unsafe)")
FIS_Q = q("fischer.q", "A[] not (P1.cs && P2.cs)")
TG_Q  = q("train.q",   "A[] not (Train1.Cross && Train2.Cross)")
TG_QU = q("train_uppaal.q", "A[] not (Train(1).Cross && Train(2).Cross)")


def tbb_proxy():
    try:
        p = subprocess.check_output(["brew", "--prefix", "tbb"], text=True).strip() + "/lib/libtbbmalloc_proxy.dylib"
        return p if os.path.exists(p) else None
    except Exception:
        return None


def run(cmd, env_extra, timeout, memgb):
    """(secs, peak_rss_gb, status, stdout). status: OK | TIMEOUT | MEMOUT | ERR."""
    env = dict(os.environ); env.update(env_extra or {})
    t0 = time.perf_counter()
    try:
        p = subprocess.Popen(["/usr/bin/time", "-l"] + cmd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    except Exception as e:
        return 0.0, 0.0, "ERR", str(e)
    i, peak = 0, 0.0
    while p.poll() is None:
        if time.perf_counter() - t0 > timeout:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL); p.wait()
            return time.perf_counter() - t0, peak, "TIMEOUT", ""
        if i % 5 == 0:
            try:
                rss = subprocess.check_output(["ps", "-o", "rss=", "-g", str(os.getpgid(p.pid))], text=True)
                tot = sum(int(x) for x in rss.split()) / 1048576.0
                peak = max(peak, tot)
                if tot > memgb:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL); p.wait()
                    return time.perf_counter() - t0, tot, "MEMOUT", ""
            except Exception:
                pass
        i += 1; time.sleep(0.05)
    dt = time.perf_counter() - t0
    out = p.stdout.read().decode("utf-8", "replace")
    m = re.search(r"([0-9]+)\s+maximum resident set size", out)
    return dt, (int(m.group(1)) / (1024 ** 3)) if m else peak, "OK", out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viking", nargs="*", type=int, default=[10, 12, 15, 17])
    ap.add_argument("--fischer", nargs="*", type=int, default=[6, 7, 8, 9, 10, 11])
    ap.add_argument("--train", action="store_true", default=True)
    ap.add_argument("--threads", nargs="+", type=int, default=[4, 8, 12])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800)   # 30 minutes
    ap.add_argument("--memgb", type=float, default=15.0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "ltsmin_raw.csv"))
    ap.add_argument("--seed", nargs="*", default=[])
    ap.add_argument("--families", default="",
                    help="fam:N,N,N;fam:N,N  e.g. phil:6,8,10;arbiter:8,12,16")
    args = ap.parse_args()
    fams = {}
    for part in filter(None, args.families.split(";")):
        fam, _, ns = part.partition(":")
        fams[fam.strip()] = [int(x) for x in ns.split(",") if x.strip()]
    args.families = fams

    done = {}
    for src in list(args.seed) + [args.out]:
        if not os.path.exists(src): continue
        with open(src) as fh:
            for r in csv.DictReader(fh):
                if r.get("status") == "OK":
                    done[(r["model"], r["tool"], r["threads"], r["rep"])] = r
    print("seeded %d completed runs" % len(done), flush=True)

    new = not os.path.exists(args.out)
    fh = open(args.out, "a")
    w = csv.writer(fh)
    if new:
        w.writerow(["family", "model", "tool", "threads", "rep", "states", "seconds", "rss_gb", "status"]); fh.flush()
    for r in done.values():   # carry seeded rows into this file so it is self-contained
        if new:
            w.writerow([r["family"], r["model"], r["tool"], r["threads"], r["rep"],
                        r["states"], r["seconds"], r["rss_gb"], r["status"]])
    fh.flush()

    proxy = tbb_proxy()
    lts_env = {"DYLD_INSERT_LIBRARIES": proxy} if proxy else {}

    def emit(fam, model, tool, th, rep, states, secs, rss, status):
        w.writerow([fam, model, tool, th, rep, states, "%.3f" % secs, "%.3f" % rss, status]); fh.flush()
        print("  %-11s %-13s t=%-3s r=%s %10s %9.2fs %6.2fGB  %s"
              % (model, tool, th, rep, states, secs, rss, status), flush=True)

    def todo(model, tool, th, rep):
        return (model, tool, str(th), str(rep)) not in done

    def bench(fam, model, xml, so, pq, vq):
        for rep in range(1, args.reps + 1):
            if todo(model, "verifyta", 1, rep):
                s, m, st, o = run([VERIFYTA, "-u", "-s", xml, vq], {}, args.timeout, args.memgb)
                g = re.search(r"States explored\s*:\s*([0-9]+)", o)
                emit(fam, model, "verifyta", 1, rep, g.group(1) if g else "-", s, m, st)
        for th in args.threads:
            for rep in range(1, args.reps + 1):
                if os.path.exists(so) and todo(model, "ltsmin", th, rep):
                    s, m, st, o = run([OPAAL, "--threads=%d" % th, "--strategy=sbfs", "-s", str(TABLE_BITS), so],
                                      lts_env, args.timeout, args.memgb)
                    g = re.search(r"Explored ([0-9]+) states", o)
                    emit(fam, model, "ltsmin", th, rep, g.group(1) if g else "-", s, m, st)
            for rep in range(1, args.reps + 1):
                if todo(model, "ptctlmc-omp", th, rep):
                    s, m, st, o = run([PTCTLMC, "--backend", "omp", xml, pq],
                                      {"OMP_NUM_THREADS": str(th)}, args.timeout, args.memgb)
                    g = re.search(r"explored ([0-9]+) symbolic", o)
                    emit(fam, model, "ptctlmc-omp", th, rep, g.group(1) if g else "-", s, m, st)

    # generated families: benchmarks/models/<fam><N>.xml with queries/<fam>.q.
    # An LTSmin module is used when one exists; otherwise the harness records
    # verifyta and PTCTLMC only, rather than skipping the model.
    import glob, re as _re
    for fam, ns in sorted(args.families.items()):
        for n in ns:
            xml = os.path.join(HERE, "models", "%s%d.xml" % (fam, n))
            if not os.path.exists(xml):
                continue
            q = os.path.join(QDIR, "%s.q" % fam)
            so = os.path.join(HERE, "so", "%s%d.so" % (fam, n))
            if not os.path.exists(so):
                so = os.path.join(SO, "%s%d.so" % (fam, n))
            bench(fam, "%s%d" % (fam, n), xml, so, q, q)

    for n in args.viking:
        bench("viking", "viking%d" % n, os.path.join(VIK, "viking%d.xml" % n),
              os.path.join(SO, "viking%d.so" % n), VIK_Q, VIK_Q)
    for n in args.fischer:
        bench("fischer", "fischer%d" % n, os.path.join(MOD, "fischer%d.xml" % n),
              os.path.join(SO, "fischer%d.so" % n), FIS_Q, FIS_Q)
    if args.train:
        bench("train", "train-gate", os.path.join(MOD, "train-gate.xml"),
              os.path.join(SO, "train-gate.so"), TG_Q, TG_QU)
    print("done -> %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
