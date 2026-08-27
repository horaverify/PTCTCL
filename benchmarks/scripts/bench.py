#!/usr/bin/env python3
"""bench.py — the single benchmark entry point.

Replaces the eleven run_*/stop_*/bench_* scripts that used to live in benchmarks/.
Everything they did is a flag here:

    run     measure tools over families x sizes x thread-counts, resumably
    stop    stop a detached sweep (soft or hard)
    status  progress of a sweep

MEASUREMENT (unchanged from the harness this supersedes, because those details were
paid for in wrong numbers):
  * every run goes under /usr/bin/time in its OWN process group, and a timeout or a
    memory overrun kills the GROUP -- killing only the wrapper leaves the tool and its
    worker threads burning CPU, and enough orphans make later runs measure a saturated
    machine;
  * the RSS guard sums the whole group, because the workload is a child of the wrapper
    and watching the wrapper alone reports ~0 and never trips;
  * a nonzero exit is ERR, never a silent OK with no state count -- that is what once
    masked a query that was falsified (and so measured nothing but the path to the first
    counterexample);
  * a model file that does not exist is SKIP, never a run -- a bogus path handed to a
    tool records as ERR and is indistinguishable from a real failure.

IDENTITY AND RESUME: a row is identified by
    (model, tool, threads, rep, chunk, sim, engine, mergescan, hotlen)
so several configurations can stream into one CSV and a restart re-measures none of
them. Anything that changes what is being measured belongs in that key; if you add a
knob, add a column. Rows that predate a column read as "-" (whatever the binary
defaults to), so older CSVs still seed.

ORDERING: work is emitted size-major (all families at size 4, then 5, ...), so a
complete cheap table exists for every family early and the expensive tails run last.
Combined with --strike-limit this replaces the hand-maintained cheap/slow size lists
the old drivers carried.

Examples
--------
    # the standard sweep: ptctlmc, G-simulation, every family, 3 thread counts, detached
    ./bench.py run --tool ptctlmc --sim g --threads 16 24 32 --families all \
        --timeout 7200 --memgb 100 --outdir ../recheability_results/ptctlmc_gsim --detach

    # one family, quick
    ./bench.py run --tool ptctlmc --families phil:4-13 --threads 16 --timeout 100

    # the other tools
    ./bench.py run --tool ltsmin   --threads 24 --ltsmin-store table --ltsmin-size 25
    ./bench.py run --tool verifyta --families all --timeout 7200

    ./bench.py status --outdir ../recheability_results/ptctlmc_gsim
    ./bench.py stop   --outdir ../recheability_results/ptctlmc_gsim [--soft]
"""
import argparse, csv, json, os, re, shutil, signal, subprocess, sys, time

# ---------------------------------------------------------------------------- paths
HERE  = os.path.dirname(os.path.abspath(__file__))          # benchmarks/scripts
BENCH = os.path.dirname(HERE)                               # benchmarks
PROJ  = os.path.dirname(BENCH)                              # PTCTCL
LTS   = os.path.abspath(os.path.join(PROJ, "..", "LTStest"))  # opaal+LTSmin tree
PLAT  = "macos" if sys.platform == "darwin" else "linux"

MODELS = os.path.join(BENCH, "models")
QDIR   = os.path.join(BENCH, "queries")
SO     = os.path.join(BENCH, "so_%s" % PLAT)
VIKDIR = os.path.join(LTS, "src/opaal-ltsmin-succgen/tests/vikingtests")
LTSMOD = os.path.join(LTS, "models")
OPAAL  = os.path.join(LTS, "deps/bin/opaal2lts-mc")

DEFAULT_PTCTLMC = os.path.join(PROJ, "build-%s" % ("mac" if PLAT == "macos" else "omp"), "ptctlmc")
PTCTLMC = os.environ.get("PTCTLMC_BIN") or DEFAULT_PTCTLMC
VERIFYTA = ("/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta"
            if PLAT == "macos" else (shutil.which("verifyta") or "verifyta"))
TCHECKER = os.environ.get("TCHECKER_BIN") or os.path.join(
    os.path.dirname(LTS), "tchecker", "install", "bin", "tck-reach")

CSV_COLS = ["family", "model", "tool", "threads", "rep", "states", "seconds", "rss_gb",
            "status", "mergescan", "hotlen", "chunk", "sim", "engine", "queries"]
# The first eleven columns and their order are load-bearing: make_clean_results.py and the
# plot_*.py readers index them by name out of files written before chunk/sim/engine existed.
# New columns go on the END.

# family -> default size ladder. Sizes that do not exist as a model file are skipped, so a
# generous ladder costs nothing but lets --strike-limit find each family's wall by itself.
FAMILIES = {
    "phil":    list(range(4, 18)),
    "single":  list(range(4, 10)),
    "twolock": list(range(4, 18)),
    "fischer": list(range(4, 18)),
    "csma":    list(range(4, 19)) + [20, 22],
    "arbiter": list(range(4, 21)),
    "grc":     list(range(4, 21)),
    "viking":  [4, 6, 10, 12, 15, 17, 20],
    "hddi":    [2, 10, 15, 20],
    "train":   [0, 1],                    # the train-gate pair, generated like the rest
}
# Families the UPPAAL comparison covered.
UPPAAL_FAMILIES = ["phil", "single", "twolock", "fischer", "viking"]

# Families that do NOT yield a valid measurement, and why. Measured, not assumed -- each was
# run before being listed here. "all" excludes these; name one explicitly (or pass
# --include-unmeasurable) to run it anyway.
#
# A falsified property is not a slow benchmark, it is a different benchmark: both tools stop
# at the first counterexample, so the wall time measures the path to that state and not the
# state space. That is why grc is out rather than merely noted.
UNMEASURABLE = {
    "ptctlmc-omp": {
        "hddi":  "query UNSUPPORTED by ptctlmc (location-equality disjunctions); exit 1",
        "train": "ptctlmc exits 1 on train0/train1 (int-array model features)",
        "grc":   "'A[] not deadlock' is NOT satisfied -- run stops at the first "
                 "counterexample, so the state count is not a state-space measure",
    },
    "verifyta": {},
    "ltsmin": {},
    "tchecker": {
        "hddi":   "reach query is a disjunction-of-equalities over integer variables, not a "
                  "location conjunction -- no mechanical utot label injection for it",
        "train":  "utot conversion fails on train0/train1 (int-array model features)",
        "grc":    "'A[] not deadlock' -- deadlock is not a tck-reach label query",
        "single": "reach query is a clock-threshold invariant (x0<=6), not a location "
                  "conjunction -- no mechanical utot label injection for it",
    },
}


def model_path(fam, n):
    """Model file for (family, size), preferring the LTSmin tree where one exists.

    The opaal+LTSmin tree and benchmarks/models hold overlapping but DIFFERENT size ranges
    (fischer stops at 11 there, goes to 20 here). Prefer the LTSmin tree so a model paired
    with a .so PINS module keeps using that exact file, and fall back to the generated one
    for the sizes it lacks.
    """
    if fam == "viking":
        cands = [os.path.join(VIKDIR, "viking%d.xml" % n), os.path.join(MODELS, "viking%d.xml" % n)]
    elif fam in ("fischer", "hddi"):
        cands = [os.path.join(LTSMOD, "%s%d.xml" % (fam, n)), os.path.join(MODELS, "%s%d.xml" % (fam, n))]
    else:
        cands = [os.path.join(MODELS, "%s%d.xml" % (fam, n))]
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


def model_name(fam, n):
    return "%s%d" % (fam, n)


def query_dir(kind):
    """reach_queries/ or live_queries/, falling back to the legacy flat queries/ dir.

    The fallback exists because sweeps launched before the split still read queries/, and a
    running sweep must keep resolving its queries; drop it once nothing in flight needs it.
    """
    d = os.path.join(BENCH, "%s_queries" % ("reach" if kind == "reach" else "live"))
    if os.path.isdir(d):
        return d
    if kind == "reach" and os.path.isdir(QDIR):
        return QDIR
    return d


def query_path(fam, tool, kind="reach"):
    """Query for a family. train has a separate UPPAAL phrasing; everything else shares one.

    Returns a path that may not exist: a family with no query of this kind (e.g. viking has
    no valid liveness property) is SKIPPED by the caller rather than run against nothing.
    """
    d = query_dir(kind)
    if fam == "train" and tool == "verifyta" and kind == "reach":
        return os.path.join(d, "train_uppaal.q")
    return os.path.join(d, "%s.q" % fam)


def so_path(fam, n):
    return os.path.join(SO, "%s.so" % model_name(fam, n))


# ------------------------------------------------------------------------ measuring
def _pgrp_rss_gb(pgid):
    """Sum RSS (GB) over every live process in process group `pgid`."""
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
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        p.wait()
    except Exception:
        pass


def measure(cmd, env_extra, timeout, memgb):
    """(secs, peak_rss_gb, status, output). status: OK | TIMEOUT | MEMOUT | ERR."""
    env = dict(os.environ)
    env.update(env_extra or {})
    t0 = time.perf_counter()
    time_cmd = ["/usr/bin/time", "-v"] if PLAT == "linux" else ["/usr/bin/time", "-l"]
    try:
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
        i += 1
        time.sleep(0.05)
    dt = time.perf_counter() - t0
    out = p.stdout.read().decode("utf-8", "replace")
    m = (re.search(r"([0-9]+)\s+maximum resident set size", out) or
         re.search(r"Maximum resident set size \(kbytes\):\s+([0-9]+)", out))
    rss_kb = int(m.group(1)) if m else 0
    rss_gb = (rss_kb / (1024 ** 2)) if PLAT == "linux" else (rss_kb / (1024 ** 3))
    status = "OK" if p.returncode == 0 else "ERR"
    return dt, max(peak, rss_gb), status, out


# ---------------------------------------------------------------------------- tools
class Tool:
    """A measurable tool: how to invoke it, what environment it needs, how to read its
    state count, and whether thread count means anything to it."""

    def __init__(self, name, threaded, states_re):
        self.name, self.threaded, self.states_re = name, threaded, states_re

    def argv(self, a, xml, q, th, so):
        raise NotImplementedError

    def env(self, a, th):
        return {}

    def available(self, xml, so):
        return True


class PtctlmcTool(Tool):
    def __init__(self):
        super().__init__("ptctlmc-omp", True, r"explored ([0-9]+) symbolic")

    def argv(self, a, xml, q, th, so):
        cmd = [a.ptctlmc_bin, "--backend", "omp"]
        if a.chunk is not None:
            cmd += ["--omp-chunk", str(a.chunk)]
        if a.sim != "alu":
            cmd += ["--sim", a.sim]
        cmd += a.ptctlmc_extra
        return cmd + [xml, q]

    def env(self, a, th):
        e = {"OMP_NUM_THREADS": str(th)}
        if a.engine != "adaptive":
            e["PTCTLMC_ENGINE"] = a.engine
        if a.mergescan is not None:
            e["PTCTLMC_MERGESCAN"] = str(a.mergescan)
        if a.hotlen is not None:
            e["PTCTLMC_HOTLEN"] = str(a.hotlen)
        return e


class VerifytaTool(Tool):
    def __init__(self):
        super().__init__("verifyta", False, r"States explored\s*:\s*([0-9]+)")

    def argv(self, a, xml, q, th, so):
        return [a.verifyta_bin, "-u", "-s", xml, q]


class LtsminTool(Tool):
    def __init__(self):
        super().__init__("ltsmin", True, r"Explored ([0-9]+) states")

    def argv(self, a, xml, q, th, so):
        return [a.opaal_bin, "--threads=%d" % th, "--strategy=sbfs",
                "--state=%s" % a.ltsmin_store, "-s", str(a.ltsmin_size), so]

    def available(self, xml, so):
        return os.path.exists(so)   # needs a compiled PINS module, not the xml


def _tchecker_labels():
    """family -> comma-separated tchecker labels, generated by injecting labels into
    benchmarks/tck/*.tck for the bad-state conjunction each family's reach query checks
    (see benchmarks/tck/labels.json). Families not listed have no labeled .tck models."""
    path = os.path.join(BENCH, "tck", "labels.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


TCHECKER_LABELS = _tchecker_labels()


class TcheckerTool(Tool):
    """tchecker's tck-reach. Consumes .tck, not UPPAAL .xml, so it is only available for
    models that have been converted into benchmarks/tck/ AND whose family has labels
    injected for the bad state its reach query checks (see TCHECKER_LABELS)."""

    def __init__(self):
        super().__init__("tchecker", False, r"VISITED_STATES\s+([0-9]+)")

    def _family(self, xml):
        m = re.match(r"^([a-zA-Z_]+)", os.path.basename(xml))
        return m.group(1) if m else ""

    def argv(self, a, xml, q, th, so):
        tck = os.path.join(BENCH, "tck", os.path.basename(xml).replace(".xml", ".tck"))
        labels = TCHECKER_LABELS[self._family(xml)]
        return [a.tchecker_bin, "-a", "reach", "-l", labels, tck]

    def available(self, xml, so):
        if self._family(xml) not in TCHECKER_LABELS:
            return False
        tck = os.path.join(BENCH, "tck", os.path.basename(xml).replace(".xml", ".tck"))
        return os.path.exists(tck)


TOOLS = {"ptctlmc": PtctlmcTool(), "verifyta": VerifytaTool(),
         "ltsmin": LtsminTool(), "tchecker": TcheckerTool()}


# ------------------------------------------------------------------------- selection
def parse_families(spec, tool=None, include_unmeasurable=False, log=print):
    """'all' | 'uppaal' | 'phil:4-13;csma:4,6,8' -> {fam: [sizes]}.

    'all' means every family that yields a valid measurement for `tool`; the excluded ones
    are reported with their reason rather than dropped silently.
    """
    bad = {} if include_unmeasurable else UNMEASURABLE.get(tool, {})
    if not spec or spec == "all":
        for f, why in sorted(bad.items()):
            if f in FAMILIES:
                log("  EXCLUDED family %-8s %s" % (f, why))
        return {f: list(ns) for f, ns in FAMILIES.items() if f not in bad}
    if spec == "uppaal":
        return {f: list(FAMILIES[f]) for f in UPPAAL_FAMILIES}
    out = {}
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        fam, _, ns = part.partition(":")
        fam = fam.strip()
        if fam not in FAMILIES:
            sys.exit("unknown family %r (known: %s)" % (fam, ", ".join(sorted(FAMILIES))))
        if not ns.strip():
            out[fam] = list(FAMILIES[fam])
            continue
        sizes = []
        for tok in ns.split(","):
            tok = tok.strip()
            if "-" in tok:
                lo, _, hi = tok.partition("-")
                sizes += list(range(int(lo), int(hi) + 1))
            elif tok:
                sizes.append(int(tok))
        out[fam] = sizes
    return out


def work_items(fams, order="family"):
    """The run order. Always strictly sequential -- one model at a time, one process at a
    time; `order` only decides which model comes next.

    family (default): one family completed end to end, ascending in size, before the next
        family starts. This is the order the earlier drivers used and the one the CSV is
        read in.
    size: all families at size 4, then all at size 5, ... Gets a cheap row for every family
        early, at the cost of a log that interleaves families.
    """
    if order == "size":
        items = sorted(((n, fam) for fam, ns in fams.items() for n in ns))
        return [(fam, n) for n, fam in items]
    return [(fam, n) for fam in sorted(fams) for n in sorted(fams[fam])]


# ------------------------------------------------------------------------------ run
def cmd_run(a):
    tool = TOOLS[a.tool]
    os.makedirs(a.outdir, exist_ok=True)

    if a.detach:
        return detach(a)

    fams = parse_families(a.families, tool.name, a.include_unmeasurable)

    threads = a.threads if tool.threaded else [1]
    tag = a.tag or a.tool
    stop_file = os.path.join(a.outdir, "STOP")
    pidfile = os.path.join(a.outdir, "bench.pid")
    with open(pidfile, "w") as fh:
        fh.write("%d\n" % os.getpid())
    with open(os.path.join(a.outdir, "bench.pgid"), "w") as fh:
        fh.write("%d\n" % os.getpgid(0))

    def log(msg):
        print("[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg), flush=True)

    log("bench.py run  tool=%s sim=%s engine=%s chunk=%s" % (a.tool, a.sim, a.engine, a.chunk))
    log("threads=%s reps=%d timeout=%ds memgb=%g strike=%d pin=%d"
        % (threads, a.reps, a.timeout, a.memgb, a.strike_limit, a.pin))
    log("families: %s" % "; ".join("%s:%s" % (f, ",".join(map(str, n))) for f, n in sorted(fams.items())))
    for repo, path in (("PTCTCL", PROJ), ("HDBM", os.path.join(PROJ, "HDBM"))):
        try:
            h = subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
            log("%s @ %s" % (repo, h))
        except Exception:
            pass

    knobs = (("-" if a.mergescan is None else str(a.mergescan)),
             ("-" if a.hotlen is None else str(a.hotlen)),
             ("-" if a.chunk is None else str(a.chunk)),
             a.sim, a.engine, a.queries)

    for th in threads:
        if os.path.exists(stop_file):
            log("STOP present -- exiting."); break
        out_csv = os.path.join(a.outdir, "%s_t%d%s.csv"
                               % (tag, th, "" if a.chunk is None else "_c%d" % a.chunk))
        run_one_config(a, tool, fams, th, out_csv, knobs, log, stop_file)
        log("=== THREADS=%d COMPLETE -> %s" % (th, out_csv))

    try:
        os.remove(pidfile)
    except OSError:
        pass
    log("ALL DONE -- %s" % a.outdir)


def run_one_config(a, tool, fams, th, out_csv, knobs, log, stop_file):
    """One (tool, threads, knobs) pass over every family, resumable and strike-limited."""
    ms, hl, ch, sim, engine, qk = knobs

    # Seed from this file plus any --seed files: a run that DEFINITIVELY finished (OK, or a
    # real TIMEOUT/MEMOUT) is never repeated. ERR is left to re-run -- it is cheap and may
    # be transient.
    done, all_rows = {}, []
    for src in list(a.seed) + [out_csv]:
        if not os.path.exists(src):
            continue
        with open(src) as fh:
            for r in csv.DictReader(fh):
                all_rows.append(r)
                if r.get("status") in ("OK", "TIMEOUT", "MEMOUT"):
                    done[(r["model"], r["tool"], r["threads"], r["rep"],
                          r.get("mergescan") or "-", r.get("hotlen") or "-",
                          r.get("chunk") or "-", r.get("sim") or "-",
                          r.get("engine") or "-", r.get("queries") or "reach")] = r

    # Per-(family, tool, knobs) timeout streaks, seeded from disk so the rule survives a
    # restart. A size counts as timed out iff it has runs and NONE finished OK.
    outcomes = {}

    def record(fam, N, timed_out):
        d = outcomes.setdefault(fam, {})
        if d.get(N) is False:
            return
        d[N] = timed_out

    hist = {}
    for r in all_rows:
        if r["tool"] != tool.name:
            continue
        if (r.get("sim") or "-") != sim or (r.get("engine") or "-") != engine:
            continue
        if (r.get("chunk") or "-") != ch or (r.get("queries") or "reach") != qk:
            continue
        hist.setdefault((r["family"], r["model"]), []).append(r.get("status"))
    for (fam, model), sts in hist.items():
        n = int(re.search(r"(\d+)$", model).group(1)) if re.search(r"(\d+)$", model) else 0
        record(fam, n, not any(s == "OK" for s in sts))

    def struck(fam, N):
        if a.strike_limit <= 0:
            return False
        d, streak = outcomes.get(fam, {}), 0
        for n in sorted((x for x in d if x < N), reverse=True):
            if d[n]:
                streak += 1
                if streak >= a.strike_limit:
                    return True
            else:
                break
        return False

    new = not os.path.exists(out_csv)
    fh = open(out_csv, "a")
    w = csv.writer(fh)
    if new:
        w.writerow(CSV_COLS)
        for r in done.values():     # carry seeded rows in so the file is self-contained
            w.writerow([r.get(c, "-") or "-" for c in CSV_COLS])
    fh.flush()
    print("seeded %d completed runs" % len(done), flush=True)

    for fam, n in work_items(fams, a.order):
        if os.path.exists(stop_file):
            log("STOP present -- exiting."); break
        model, xml = model_name(fam, n), model_path(fam, n)
        if not os.path.exists(xml):
            continue                                     # size not generated: silent skip
        so, q = so_path(fam, n), query_path(fam, tool.name, a.queries)
        if not os.path.exists(q):
            print("  %-11s %-13s   SKIP  (no %s query for '%s')"
                  % (model, tool.name, a.queries, fam), flush=True)
            continue
        if not tool.available(xml, so):
            print("  %-11s %-13s   SKIP  (input missing for this tool)" % (model, tool.name),
                  flush=True)
            continue
        if struck(fam, n):
            print("  %-11s %-13s   SKIP  (%d+ consecutive TIMEOUTs earlier in '%s')"
                  % (model, tool.name, a.strike_limit, fam), flush=True)
            continue

        sts, timed_out = [], False
        for rep in range(1, a.reps + 1):
            key = (model, tool.name, str(th), str(rep), ms, hl, ch, sim, engine, qk)
            if key in done:
                sts.append(done[key]["status"])
                timed_out = timed_out or done[key]["status"] in ("TIMEOUT", "MEMOUT")
                continue
            # Once a config times out on a model, its remaining reps are skipped: further
            # reps are deterministic in cost and would burn the whole budget again.
            if timed_out:
                print("  %-11s %-13s t=%-3s r=%s   SKIP  (already TIMED OUT on %s)"
                      % (model, tool.name, th, rep, model), flush=True)
                sts.append("TIMEOUT")
                continue
            # Pin each run to exactly as many cores as it has threads (16 -> 0-15), which is
            # the methodology the recorded baselines were measured under; anything else makes
            # a t16 number incomparable to the t16 column already on disk.
            pre = ["taskset", "-c", "0-%d" % (th - 1)] if (a.pin and tool.threaded) else []
            secs, rss, st, out = measure(pre + tool.argv(a, xml, q, th, so), tool.env(a, th),
                                         a.timeout, a.memgb)
            g = re.search(tool.states_re, out)
            states = next((x for x in (g.groups() if g else ()) if x), "-") if g else "-"
            w.writerow([fam, model, tool.name, th, rep, states, "%.3f" % secs, "%.3f" % rss,
                        st, ms, hl, ch, sim, engine, qk])
            fh.flush()
            print("  %-11s %-13s t=%-3s r=%s %10s %9.2fs %6.2fGB  %s"
                  % (model, tool.name, th, rep, states, secs, rss, st), flush=True)
            sts.append(st)
            timed_out = timed_out or st in ("TIMEOUT", "MEMOUT")
        if sts:
            record(fam, n, all(s in ("TIMEOUT", "MEMOUT") for s in sts))
    fh.close()


def detach(a):
    """Re-exec this same command under setsid+nohup, pinned, with output to a log."""
    os.makedirs(a.outdir, exist_ok=True)
    log_path = os.path.join(a.outdir, "driver.log")
    argv = [sys.executable, os.path.abspath(__file__)] + \
           [x for x in sys.argv[1:] if x != "--detach"]
    # The driver itself is NOT pinned: each measured run pins itself to exactly as many cores
    # as it has threads (see run_one_config), so pinning the driver too would confine a
    # 32-thread run to whatever the driver inherited.
    with open(log_path, "a") as lg:
        p = subprocess.Popen(["setsid", "nohup"] + argv, stdout=lg, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    print("detached pid %d -> %s" % (p.pid, log_path))
    print("stop with: %s stop --outdir %s" % (os.path.abspath(__file__), a.outdir))
    return 0


# ----------------------------------------------------------------------- stop/status
def cmd_stop(a):
    stop_file = os.path.join(a.outdir, "STOP")
    open(stop_file, "w").close()
    print("wrote %s -- the driver exits at the next model boundary." % stop_file)
    if a.soft:
        return 0
    pgid_f = os.path.join(a.outdir, "bench.pgid")
    if os.path.exists(pgid_f):
        pgid = int(open(pgid_f).read().strip())
        try:
            os.killpg(pgid, signal.SIGTERM)
            print("SIGTERM -> process group %d" % pgid)
        except (ProcessLookupError, PermissionError) as e:
            print("process group %d not killed: %s" % (pgid, e))
    for exe in ("ptctlmc", "opaal2lts-mc", "verifyta", "tck-reach"):
        subprocess.run(["pkill", "-9", "-x", exe], stderr=subprocess.DEVNULL)
    print("killed leftover tool processes.")
    return 0


def cmd_status(a):
    pidfile = os.path.join(a.outdir, "bench.pid")
    running = False
    if os.path.exists(pidfile):
        pid = int(open(pidfile).read().strip())
        running = os.path.exists("/proc/%d" % pid)
        print("driver pid %d: %s" % (pid, "RUNNING" if running else "not running"))
    else:
        print("no bench.pid in %s" % a.outdir)
    for f in sorted(os.listdir(a.outdir)):
        if not f.endswith(".csv"):
            continue
        rows = list(csv.DictReader(open(os.path.join(a.outdir, f))))
        st = {}
        for r in rows:
            st[r["status"]] = st.get(r["status"], 0) + 1
        last = rows[-1]["model"] if rows else "-"
        print("  %-34s %4d rows  %-30s last=%s"
              % (f, len(rows), " ".join("%s=%d" % kv for kv in sorted(st.items())), last))
    return 0


# ------------------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="measure a tool over families x sizes x threads")
    r.add_argument("--tool", default="ptctlmc", choices=sorted(TOOLS))
    r.add_argument("--families", default="all",
                   help="'all' | 'uppaal' | 'phil:4-13;csma:4,6,8' (default all)")
    r.add_argument("--include-unmeasurable", action="store_true",
                   help="do not drop families known to produce invalid measurements")
    r.add_argument("--queries", default="reach", choices=["reach", "live"],
                   help="which query set to measure: reach_queries/ (default) or "
                        "live_queries/. Recorded in the CSV, so both can stream into one dir.")
    r.add_argument("--order", default="family", choices=["family", "size"],
                   help="family: finish one family before starting the next (default). "
                        "size: sweep all families at each size. Both are strictly "
                        "sequential -- one model, one process at a time.")
    r.add_argument("--threads", nargs="+", type=int, default=[16, 24, 32])
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--timeout", type=int, default=7200, help="per run, seconds (default 2h)")
    r.add_argument("--memgb", type=float, default=100.0, help="RSS cap per run (default 100)")
    r.add_argument("--strike-limit", type=int, default=1,
                   help="skip larger sizes after this many consecutive timeouts (0 disables)")
    r.add_argument("--outdir", default=os.path.join(BENCH, "recheability_results", "run"))
    r.add_argument("--tag", default=None, help="CSV basename prefix (default: the tool name)")
    r.add_argument("--seed", nargs="*", default=[], help="extra CSVs to seed completed runs from")
    r.add_argument("--detach", action="store_true", help="run under setsid+nohup and return")
    r.add_argument("--pin", type=int, default=1, help="taskset-pin to as many cores as threads")
    # ptctlmc knobs
    r.add_argument("--sim", default="alu", choices=["alu", "g"],
                   help="zone subsumption: aLU (default) or the G-simulation")
    r.add_argument("--engine", default="adaptive", choices=["adaptive", "legacy"])
    r.add_argument("--chunk", type=int, default=8, help="--omp-chunk (default 8, the binary default)")
    r.add_argument("--mergescan", type=int, default=None, help="PTCTLMC_MERGESCAN")
    r.add_argument("--hotlen", type=int, default=None, help="PTCTLMC_HOTLEN")
    r.add_argument("--ptctlmc-bin", default=PTCTLMC)
    r.add_argument("--ptctlmc-extra", nargs="*", default=[], help="extra ptctlmc flags")
    # other tools
    r.add_argument("--verifyta-bin", default=VERIFYTA)
    r.add_argument("--opaal-bin", default=OPAAL)
    r.add_argument("--tchecker-bin", default=TCHECKER)
    r.add_argument("--ltsmin-size", type=int, default=26, help="LTSmin -s (log2 store size)")
    r.add_argument("--ltsmin-store", default="tree", choices=["tree", "table", "cleary-tree"])
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("stop", help="stop a detached sweep")
    s.add_argument("--outdir", required=True)
    s.add_argument("--soft", action="store_true",
                   help="only write STOP; let the current model finish")
    s.set_defaults(fn=cmd_stop)

    t = sub.add_parser("status", help="progress of a sweep")
    t.add_argument("--outdir", required=True)
    t.set_defaults(fn=cmd_status)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
