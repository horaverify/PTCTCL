#!/usr/bin/env python3
"""Evaluation harness: UPPAAL verifyta vs opaal+LTSmin vs PTCTLMC (OpenMP).

Per run: 30 min wall timeout, 15 GB RSS guard (whole process group). A run exceeding
either is recorded as TIMEOUT / MEMOUT rather than a number. Results stream to
raw.csv after every run so progress is watchable live.

Already-completed (model, tool, threads, rep) pairs with status OK are SKIPPED, and are
seeded from any earlier CSV given with --seed, so nothing is measured twice.
"""
import argparse, csv, os, re, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "LTStest"))   # opaal+LTSmin tree
PLAT = "macos" if sys.platform == "darwin" else "linux"
VIK  = os.path.join(ROOT, "src/opaal-ltsmin-succgen/tests/vikingtests")
MOD  = os.path.join(ROOT, "models")
SO   = os.path.join(HERE, f"so_{PLAT}")
OPAAL = os.path.join(ROOT, "deps/bin/opaal2lts-mc")
PTCTLMC = os.path.abspath(os.path.join(HERE, "..", f"build-{'mac' if PLAT == 'macos' else 'omp'}", "ptctlmc"))
# PTCTLMC_BIN overrides which binary is measured. Needed to sweep an OLD build against the
# current one: the pre-2026-07-30 engine cannot be recovered from the current source by any
# flag (the dense mirror arrived with the packed zone store, so turning the mirror off leaves
# the store paying an unpack per chain walk and is worse than either), but the binary that
# produced results_old still exists in build-spin/ and reproduces those numbers exactly.
PTCTLMC = os.environ.get("PTCTLMC_BIN") or PTCTLMC
if PLAT == "macos":
    VERIFYTA = "/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta"
else:
    # On Linux, find verifyta in PATH
    import shutil
    VERIFYTA = shutil.which("verifyta") or "verifyta"  # fallback to command name for PATH lookup
TABLE_BITS = 26

# One queries/ dir for every platform: the query text is model-level CTL and has nothing
# platform-specific in it, and the old per-platform copies had drifted apart. Every query here
# is checked to be SATISFIED on a small size of its family, which matters more than it sounds --
# a falsified property makes both tools stop at the first counterexample, so the state counts
# stop being a measure of the state space. queries_linux_old/csma.q was exactly that case:
# `A[] not (S1.Send && S2.Send)` is violated after 3 explored states.
QDIR = os.path.join(HERE, "queries"); os.makedirs(QDIR, exist_ok=True)

def model_file(*cands):
    """First candidate that exists, else the first (so the caller reports a clear SKIP).

    The opaal+LTSmin tree (ROOT/models, ROOT/.../vikingtests) and benchmarks/models hold
    overlapping but DIFFERENT size ranges -- e.g. fischer goes to 11 in the LTSmin tree but to 20
    in benchmarks/models. Prefer the LTSmin tree so a model paired with an .so module keeps using
    that exact file, and fall back to the local generated one for the sizes it lacks.
    """
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]

def q(name, text=None):
    p = os.path.join(QDIR, name)
    if text:
        open(p, "w").write(text + "\n")
    return p
VIK_Q = q("viking.q")
FIS_Q = q("fischer.q")
TG_Q  = q("train.q")
TG_QU = q("train_uppaal.q")


def tbb_proxy():
    try:
        p = subprocess.check_output(["brew", "--prefix", "tbb"], text=True).strip() + "/lib/libtbbmalloc_proxy.dylib"
        return p if os.path.exists(p) else None
    except Exception:
        return None


def _pgrp_rss_gb(pgid):
    """Sum RSS (GB) over every live process in process group `pgid`.

    The workload runs as a child of the /usr/bin/time wrapper, so watching only the
    wrapper's own pid (as this used to) reports ~0 and never trips the memory guard.
    Scanning the whole group captures the wrapper, the tool, and its worker threads.
    """
    try:
        out = subprocess.check_output(["ps", "-e", "-o", "rss=,pgid="], text=True)
    except Exception:
        return 0.0
    tot_kb = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1] == str(pgid):
            tot_kb += int(parts[0])
    return tot_kb / 1048576.0


def _kill_group(p, pgid):
    """SIGKILL the whole process group, then reap the wrapper.

    Plain p.kill() only kills /usr/bin/time; the tool it forked (opaal / verifyta /
    ptctlmc, each with 4-6 worker threads) survives as an orphan and keeps burning CPU.
    Enough orphans accumulate that later runs measure a saturated machine -- the
    'benchmarks running in parallel' symptom. Killing the group takes the tool with it.
    """
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        p.wait()
    except Exception:
        pass


def run(cmd, env_extra, timeout, memgb):
    """(secs, peak_rss_gb, status, stdout). status: OK | TIMEOUT | MEMOUT | ERR."""
    env = dict(os.environ); env.update(env_extra or {})
    t0 = time.perf_counter()
    try:
        time_cmd = ["/usr/bin/time", "-v"] if PLAT == "linux" else ["/usr/bin/time", "-l"]
        # start_new_session=True puts the wrapper (and everything it forks) into its own
        # process group so we can kill the whole tree on timeout/memout -- no orphans.
        p = subprocess.Popen(time_cmd + cmd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, start_new_session=True)
    except Exception as e:
        return 0.0, 0.0, "ERR", str(e)
    try:
        pgid = os.getpgid(p.pid)
    except ProcessLookupError:
        pgid = p.pid
    i, peak = 0, 0.0
    while p.poll() is None:
        if time.perf_counter() - t0 > timeout:
            _kill_group(p, pgid)
            return time.perf_counter() - t0, peak, "TIMEOUT", ""
        if i % 5 == 0:
            tot = _pgrp_rss_gb(pgid)
            peak = max(peak, tot)
            if tot > memgb:
                _kill_group(p, pgid)
                return time.perf_counter() - t0, tot, "MEMOUT", ""
        i += 1; time.sleep(0.05)
    dt = time.perf_counter() - t0
    out = p.stdout.read().decode("utf-8", "replace")
    # Extract memory: macOS "-l" or Linux "-v" format
    m = re.search(r"([0-9]+)\s+maximum resident set size", out) or \
        re.search(r"Maximum resident set size \(kbytes\):\s+([0-9]+)", out)
    rss_kb = int(m.group(1)) if m else 0
    rss_gb = (rss_kb / (1024 ** 2)) if PLAT == "linux" else (rss_kb / (1024 ** 3))
    # A nonzero exit is a genuine failure (verifyta type-error=1, ptctlmc unsupported=2);
    # 'satisfied' and 'not satisfied' both exit 0. Record it as ERR instead of a silent
    # OK with no state count, which is what masked the broken queries.
    status = "OK" if p.returncode == 0 else "ERR"
    return dt, max(peak, rss_gb), status, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viking", nargs="*", type=int, default=[10, 12, 15, 17])
    ap.add_argument("--hddi", nargs="*", type=int, default=[])
    ap.add_argument("--fischer", nargs="*", type=int, default=[6, 7, 8, 9, 10, 11])
    ap.add_argument("--train", action="store_true", default=True)
    # --train defaults on and is a store_true, so there was no way to turn it off from the CLI.
    ap.add_argument("--no-train", action="store_false", dest="train",
                    help="skip the train-gate model")
    ap.add_argument("--threads", nargs="+", type=int, default=[4, 8, 12])
    ap.add_argument("--only-ptctl-omp", action="store_true",
                    help="run only PTCTLMC OpenMP backend (skip verifyta and ltsmin)")
    ap.add_argument("--only-verifyta", action="store_true",
                    help="run only UPPAAL verifyta (skip ltsmin and PTCTLMC)")
    ap.add_argument("--omp-chunk", type=int, default=None,
                    help="pass --omp-chunk N to PTCTLMC when using backend omp")
    # Passed-list tuning knobs, read by NetPassed in src/netreach.cpp. Both are recorded in the
    # CSV and are part of the identity of a run, so several settings can sweep into ONE file
    # without the resume logic mistaking one config's row for another's.
    ap.add_argument("--mergescan", type=int, default=None,
                    help="PTCTLMC_MERGESCAN: newest live zones a new zone tries to convex-merge "
                         "with (0 = unbounded whole-chain scan). Binary default 32.")
    ap.add_argument("--hotlen", type=int, default=None,
                    help="PTCTLMC_HOTLEN: live zones a chain must reach before it gets a dense "
                         "mirror (0 = never mirror). Binary default 64.")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800)   # 30 minutes
    ap.add_argument("--memgb", type=float, default=15.0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "full.csv"))
    ap.add_argument("--seed", nargs="*", default=[])
    ap.add_argument("--families", default="",
                    help="fam:N,N,N;fam:N,N  e.g. phil:6,8,10;arbiter:8,12,16")
    ap.add_argument("--strike-limit", type=int, default=3,
                    help="within a family, once a tool has TIMED OUT this many consecutive "
                         "sizes it is skipped for all larger sizes (0 disables). Persists "
                         "across restarts via the recorded results.")
    args = ap.parse_args()
    fams = {}
    for part in filter(None, args.families.split(";")):
        fam, _, ns = part.partition(":")
        fams[fam.strip()] = [int(x) for x in ns.split(",") if x.strip()]
    args.families = fams

    # The knob setting is part of a run's IDENTITY, not just a note about it: a sweep appends
    # every setting to the same CSV, and the resume logic must not read "phil12 at mergescan=8"
    # as already covered by "phil12 at mergescan=32". Only PTCTLMC reads the knobs, so rows for
    # verifyta/ltsmin carry "-", as do rows written before these columns existed (hence the
    # .get(..., "-") when seeding) -- an unset knob means "whatever the binary defaults to".
    MS = "-" if args.mergescan is None else str(args.mergescan)
    HL = "-" if args.hotlen is None else str(args.hotlen)
    def knobs_of(tool):
        return (MS, HL) if tool == "ptctlmc-omp" else ("-", "-")
    def knob_env():
        e = {}
        if args.mergescan is not None: e["PTCTLMC_MERGESCAN"] = str(args.mergescan)
        if args.hotlen    is not None: e["PTCTLMC_HOTLEN"]    = str(args.hotlen)
        return e

    def size_of(model):
        mm = re.search(r"(\d+)$", model)
        return int(mm.group(1)) if mm else 0

    # Per-(family, tool) timeout streaks, so a tool that has clearly blown past the cap for
    # the last few sizes of a family is not run on the (even bigger) remaining ones. A size
    # counts as a timeout for a tool iff it has recorded runs and NONE of them finished OK.
    # Seeded from the CSV so the rule survives restarts: three earlier timeouts on disk skip
    # the next size without re-running anything.
    # Keyed by (family, tool, mergescan, hotlen): a streak must not cross knob settings, or one
    # config's timeout would silently skip that size for every other config in the same sweep.
    fam_tool_outcomes = {}   # (family, tool, ms, hl) -> {size N: timed_out bool}
    def record_outcome(fam, tool, N, timed_out, knobs=None):
        d = fam_tool_outcomes.setdefault((fam, tool) + (knobs or knobs_of(tool)), {})
        if d.get(N) is False:            # a success at this size stays a success
            return
        d[N] = timed_out
    def should_skip(fam, tool, N):
        if args.strike_limit <= 0:
            return False
        d = fam_tool_outcomes.get((fam, tool) + knobs_of(tool), {})
        streak = 0
        for n in sorted((x for x in d if x < N), reverse=True):   # walk DOWN from just below N
            if d[n]:
                streak += 1
                if streak >= args.strike_limit:
                    return True
            else:
                break                     # a success breaks the consecutive run
        return False

    # A run that DEFINITIVELY completed -- OK or a real TIMEOUT/MEMOUT -- is never repeated.
    # (ERR, e.g. an unsupported query, is left to re-run; it is cheap and may be transient.)
    done = {}
    all_rows = []
    for src in list(args.seed) + [args.out]:
        if not os.path.exists(src): continue
        with open(src) as fh:
            for r in csv.DictReader(fh):
                all_rows.append(r)
                if r.get("status") in ("OK", "TIMEOUT", "MEMOUT"):
                    done[(r["model"], r["tool"], r["threads"], r["rep"],
                          r.get("mergescan") or "-", r.get("hotlen") or "-")] = r
    # Seed the streaks: group history by (family, tool, knobs, model) and fold into outcomes.
    hist = {}
    for r in all_rows:
        hist.setdefault((r["family"], r["tool"], r.get("mergescan") or "-",
                         r.get("hotlen") or "-", r["model"]), []).append(r.get("status"))
    for (fam, tool, ms, hl, model), sts in hist.items():
        if any(s == "OK" for s in sts):
            record_outcome(fam, tool, size_of(model), False, (ms, hl))
        elif any(s in ("TIMEOUT", "MEMOUT") for s in sts):
            record_outcome(fam, tool, size_of(model), True, (ms, hl))
    print("seeded %d completed runs" % len(done), flush=True)

    new = not os.path.exists(args.out)
    fh = open(args.out, "a")
    w = csv.writer(fh)
    if new:
        w.writerow(["family", "model", "tool", "threads", "rep", "states", "seconds", "rss_gb",
                    "status", "mergescan", "hotlen"]); fh.flush()
    for r in done.values():   # carry seeded rows into this file so it is self-contained
        if new:
            w.writerow([r["family"], r["model"], r["tool"], r["threads"], r["rep"],
                        r["states"], r["seconds"], r["rss_gb"], r["status"],
                        r.get("mergescan") or "-", r.get("hotlen") or "-"])
    fh.flush()

    proxy = tbb_proxy()
    lts_env = {"DYLD_INSERT_LIBRARIES": proxy} if proxy else {}

    def emit(fam, model, tool, th, rep, states, secs, rss, status):
        ms, hl = knobs_of(tool)
        w.writerow([fam, model, tool, th, rep, states, "%.3f" % secs, "%.3f" % rss, status,
                    ms, hl]); fh.flush()
        print("  %-11s %-13s t=%-3s r=%s %10s %9.2fs %6.2fGB  %s  ms=%s hl=%s"
              % (model, tool, th, rep, states, secs, rss, status, ms, hl), flush=True)

    def bench(fam, model, xml, so, pq, vq):
        N = size_of(model)
        # A MISSING model must not be measured. The dedicated --viking/--hddi/--fischer loops read
        # from the opaal+LTSmin tree, which does not carry every size the generated families do
        # (fischer stops at 11 there), and they never checked existence -- so each missing size was
        # handed to the tool as a bogus path and recorded as a genuine ERR row indistinguishable
        # from a real tool failure. fischer12-20 were "failing" for exactly this reason while
        # verifying fine by hand.
        if not os.path.exists(xml):
            print("  %-11s SKIP  (no model file: %s)" % (model, xml), flush=True)
            return

        # Run `tool` (unless the family already struck it out at smaller sizes), then fold
        # its outcome into the streak. `runs` returns the list of per-run statuses.
        def run_tool(tool, runs):
            if should_skip(fam, tool, N):
                print("  %-11s %-13s   SKIP  (%d+ consecutive TIMEOUTs earlier in '%s')"
                      % (model, tool, args.strike_limit, fam), flush=True)
                return
            sts = runs()
            if sts:  # a size with no OK run at all is a timeout for this tool
                record_outcome(fam, tool, N, all(s in ("TIMEOUT", "MEMOUT") for s in sts))

        # Once a tool TIMES OUT on this model, every remaining run of it on the SAME model is
        # skipped: further reps of a config are deterministic, and — since we try the most
        # threads first (fastest for these parallel tools) — a timeout there implies the
        # slower thread counts time out too. This stops burning a fresh 600s per rep/config on
        # a model already known to be over budget (the arbiter10-rep2 case).
        def verifyta_runs():
            sts, timed_out = [], False
            for rep in range(1, args.reps + 1):
                key = (model, "verifyta", "1", str(rep)) + knobs_of("verifyta")
                if key in done:
                    st = done[key]["status"]; sts.append(st)
                    timed_out = timed_out or st in ("TIMEOUT", "MEMOUT"); continue
                if timed_out:
                    print("  %-11s %-13s t=1   r=%s   SKIP  (already TIMED OUT on %s)"
                          % (model, "verifyta", rep, model), flush=True)
                    sts.append("TIMEOUT"); continue
                s, m, st, o = run([VERIFYTA, "-u", "-s", xml, vq], {}, args.timeout, args.memgb)
                g = re.search(r"States explored\s*:\s*([0-9]+)", o)
                emit(fam, model, "verifyta", 1, rep, g.group(1) if g else "-", s, m, st)
                sts.append(st)
                timed_out = timed_out or st in ("TIMEOUT", "MEMOUT")
            return sts

        def threaded_runs(tool, cmd_of, env_of, regex):
            sts, timed_out = [], False
            for th in sorted(args.threads, reverse=True):   # most threads (fastest) first
                for rep in range(1, args.reps + 1):
                    key = (model, tool, str(th), str(rep)) + knobs_of(tool)
                    if key in done:
                        st = done[key]["status"]; sts.append(st)
                        timed_out = timed_out or st in ("TIMEOUT", "MEMOUT"); continue
                    if timed_out:
                        print("  %-11s %-13s t=%-3s r=%s   SKIP  (already TIMED OUT on %s)"
                              % (model, tool, th, rep, model), flush=True)
                        sts.append("TIMEOUT"); continue
                    s, m, st, o = run(cmd_of(th), env_of(th), args.timeout, args.memgb)
                    g = re.search(regex, o)
                    emit(fam, model, tool, th, rep, g.group(1) if g else "-", s, m, st)
                    sts.append(st)
                    timed_out = timed_out or st in ("TIMEOUT", "MEMOUT")
            return sts

        if not args.only_ptctl_omp:
            run_tool("verifyta", verifyta_runs)
        if (not args.only_ptctl_omp) and (not args.only_verifyta) and os.path.exists(so):
            run_tool("ltsmin", lambda: threaded_runs(
                "ltsmin",
                lambda th: [OPAAL, "--threads=%d" % th, "--strategy=sbfs", "-s", str(TABLE_BITS), so],
                lambda th: lts_env,
                r"Explored ([0-9]+) states"))
        if args.only_verifyta:
            return
        run_tool("ptctlmc-omp", lambda: threaded_runs(
            "ptctlmc-omp",
            lambda th: ([PTCTLMC, "--backend", "omp"] +
                        (["--omp-chunk", str(args.omp_chunk)] if args.omp_chunk is not None else []) +
                        [xml, pq]),
            lambda th: {"OMP_NUM_THREADS": str(th), **knob_env()},
            r"explored ([0-9]+) symbolic"))

    # generated families: benchmarks/models/<fam><N>.xml with queries/<fam>.q.
    # An LTSmin module is used when one exists; otherwise the harness records
    # verifyta and PTCTLMC only, rather than skipping the model.
    for fam, ns in sorted(args.families.items()):
        for n in ns:
            xml = os.path.join(HERE, "models", "%s%d.xml" % (fam, n))
            if not os.path.exists(xml):
                continue
            q = os.path.join(QDIR, "%s.q" % fam)
            so = os.path.join(SO, "%s%d.so" % (fam, n))
            bench(fam, "%s%d" % (fam, n), xml, so, q, q)

    for n in args.viking:
        bench("viking", "viking%d" % n,
              model_file(os.path.join(VIK, "viking%d.xml" % n),
                         os.path.join(HERE, "models", "viking%d.xml" % n)),
              os.path.join(SO, "viking%d.so" % n), VIK_Q, VIK_Q)
    for n in args.hddi:
        bench("hddi", "hddi%d" % n,
              model_file(os.path.join(HERE, "models", "hddi%d.xml" % n),
                         os.path.join(MOD, "hddi%d.xml" % n)),
              os.path.join(SO, "hddi%d.so" % n), os.path.join(QDIR, "hddi.q"), os.path.join(QDIR, "hddi.q"))
    for n in args.fischer:
        bench("fischer", "fischer%d" % n,
              model_file(os.path.join(MOD, "fischer%d.xml" % n),
                         os.path.join(HERE, "models", "fischer%d.xml" % n)),
              os.path.join(SO, "fischer%d.so" % n), FIS_Q, FIS_Q)
    if args.train:
        bench("train", "train-gate", os.path.join(MOD, "train-gate.xml"),
              os.path.join(SO, "train-gate.so"), TG_Q, TG_QU)
    print("done -> %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
