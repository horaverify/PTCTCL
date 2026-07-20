#!/usr/bin/env python3
"""PTCTLMC master evaluation harness — timing + memory + parity vs UPPAAL, all backends.

Configs measured per (model, N):
    seq        1 CPU thread (deterministic parity oracle)
    omp        12-thread OpenMP
    metal      Apple GPU, NAIVE merge (one thread per bucket)
    metal-ad   Apple GPU, ADAPTIVE merge (--adaptive: calibrated CPU/GPU merge routing)
    verifyta   single-threaded UPPAAL (reference)

For each (model, N) it:
  1. generates the model;
  2. runs every config once under /usr/bin/time -l — capturing verdict, symbolic-state counts,
     peak RSS, and a single-run wall in one shot;
  3. checks VERDICT agreement across all configs and state-count PARITY (seq stored/explored ==
     verifyta, integer-for-integer — the fairness control; parallel backends vary within the
     order-sensitive convex-union merge so only their verdict is required);
  4. times each config min-of-k (k shrinks as a single run gets slower — deterministic baselines
     need no repeats), and writes a CSV row immediately (survives a long run).

Stress mode (default here): climb N per model by +1 until metal-adaptive's single run exceeds a
wall cap, peak RSS exceeds a memory cap (24 GB machine), or a backend fails.

Usage: python3 benchmarks/bench_run.py [--only fischer,phil] [--grid] [--out results.csv]
"""
import argparse, csv, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "benchmarks", "generators")
BIN = os.environ.get("PTCTLMC_BIN", os.path.join(ROOT, "build-omp", "ptctlmc"))
VERIFYTA = os.environ.get("VERIFYTA",
    "/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta")

# name -> (generator, grid Ns, start N for stress, query). Queries force FULL exploration.
# Grids are RAM-capped for a 24 GB machine: the ceiling per model is set so metal-ADAPTIVE peak RSS
# stays <~9 GB (the adaptive CPU-merge path allocates host buffers on top of the GPU buffers, so it is
# the memory-hungriest config — e.g. fischer N12 metal-ad = 12.9 GB vs naive 8.1 GB).
BENCH = {
    "fischer": ("gen_fischer.py", [8, 9, 10, 11, 12],       8,  "A[] not (P0.cs and P1.cs)"),
    "twolock": ("gen_twolock.py", [7, 8, 9, 10, 11],        4,  "A[] not (P0.cs and P1.cs)"),
    "arbiter": ("gen_arbiter.py", [11, 12, 13, 14, 15, 16], 5,  "A[] not (C0.cs and C1.cs)"),
    "csma":    ("gen_csma.py",    [9, 10, 11, 12, 13, 14],  3,  "A[] not (S0.Send and S1.Send and Bus.Transmitting)"),
    "phil":    ("gen_phil.py",    [7, 8, 9, 10, 11, 12],    6,  "A[] not (P0.Eat and P1.Eat)"),
}
MAX_N = 40
WALL_CAP = 35.0      # stop climbing once metal-adaptive's single run exceeds this (s)
MEM_CAP_GB = 13.0    # backstop: stop a model once any config's peak RSS exceeds this (24 GB machine)

OMP12 = dict(os.environ); OMP12["OMP_NUM_THREADS"] = "12"
# label -> (argv-builder(xml, qf) -> list, env). metal/omp use all 12 cores (metal marshals on CPU).
def _p(*extra):
    return lambda xml, qf: [BIN, *extra, xml, qf]
CONFIGS = [
    ("seq",      _p("--backend", "seq"),                dict(os.environ)),
    ("omp",      _p("--backend", "omp"),                OMP12),
    ("metal",    _p("--backend", "metal"),             OMP12),   # naive one-thread-per-bucket merge
    ("metal-ad", _p("--backend", "metal", "--adaptive"), OMP12), # calibrated CPU/GPU merge routing
    # --chunking sweep: expand the frontier in bounded batches so the GPU pools are sized to a chunk
    # rather than the widest BFS level. Sizes span 64x to trace the memory/time trade-off; a size
    # larger than a model's peak frontier simply never triggers (identical to metal-ad).
    ("mad-c4k",   _p("--backend", "metal", "--adaptive", "--chunking", "4096"),   OMP12),
    ("mad-c16k",  _p("--backend", "metal", "--adaptive", "--chunking", "16384"),  OMP12),
    ("mad-c64k",  _p("--backend", "metal", "--adaptive", "--chunking", "65536"),  OMP12),
    ("mad-c256k", _p("--backend", "metal", "--adaptive", "--chunking", "262144"), OMP12),
]


def run(cmd, env=None, timeout=None, rss=False):
    """Run a command; return (rc, combined_output, elapsed_s, rss_gb|None). rss via /usr/bin/time -l."""
    full = (["/usr/bin/time", "-l"] + cmd) if rss else cmd
    t0 = time.perf_counter()
    try:
        r = subprocess.run(full, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "", timeout, None
    el = time.perf_counter() - t0
    out = r.stdout + r.stderr
    g = None
    if rss:
        m = re.search(r"(\d+)\s+maximum resident set size", out)
        if m: g = round(int(m.group(1)) / (1024 ** 3), 2)
    return r.returncode, out, el, g


def gen_model(gen, N, path):
    with open(path, "w") as f:
        subprocess.run([sys.executable, os.path.join(GEN, gen), str(N)], stdout=f, check=True)


def parse_ptctlmc(out):
    verdict = "SAT" if re.search(r"is satisfied", out) else ("UNSAT" if "NOT satisfied" in out else "?")
    me = re.search(r"explored (\d+)", out); ms = re.search(r"stored-zones=(\d+)", out)
    return verdict, (int(me.group(1)) if me else None), (int(ms.group(1)) if ms else None), ("network expander ACTIVE" in out)


def probe_ptctlmc(argv, env, timeout):
    e = dict(env); e["PTCTLMC_STATS"] = "1"
    rc, out, el, rss = run(argv, e, timeout, rss=True)
    if rc is None: return ("TIMEOUT", None, None, False, None, timeout, False)
    if rc != 0:    return ("CRASH", None, None, False, None, el, False)
    v, ex, st, gpu = parse_ptctlmc(out)
    return (v, ex, st, gpu, rss, el, True)


def probe_verifyta(xml, qf, timeout):
    if not os.path.exists(VERIFYTA): return ("n/a", None, None, None, None, False)
    rc, out, el, rss = run([VERIFYTA, "-u", xml, qf], dict(os.environ), timeout, rss=True)
    if rc is None: return ("TIMEOUT", None, None, None, timeout, False)
    v = "SAT" if "is satisfied" in out else ("UNSAT" if "NOT satisfied" in out else "?")
    ms = re.search(r"States stored\s*:\s*(\d+)", out); me = re.search(r"States explored\s*:\s*(\d+)", out)
    return (v, (int(me.group(1)) if me else None), (int(ms.group(1)) if ms else None), rss, el, True)


def time_min(cmd, env, single, timeout):
    """min-of-k wall; k shrinks as the single-run wall grows (deterministic baselines need no repeats)."""
    k = 5 if single < 2 else 3 if single < 15 else 1
    ts = []
    run(cmd, env, timeout)  # warm-up
    for _ in range(k):
        rc, _, el, _ = run(cmd, env, timeout)
        if rc is None or rc != 0: return None
        ts.append(el)
    return round(min(ts), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--climb", action="store_true", help="adaptive N-climb to the ceiling (default: fixed grid)")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results.csv"))
    args = ap.parse_args()
    if not os.path.exists(BIN): sys.exit(f"binary not found: {BIN}")

    names = [x for x in args.only.split(",") if x] or list(BENCH)
    labels = [c[0] for c in CONFIGS]
    hdr = (["bench", "N", "verdict", "parity", "stored", "explored", "vfy_stored", "vfy_explored"]
           + [f"t_{l}" for l in labels] + ["t_verifyta"]
           + [f"rss_{l}" for l in labels] + ["rss_verifyta"]
           + ["vfy/omp", "vfy/metal", "vfy/metal-ad", "metal/metal-ad", "omp/metal-ad", "gpu", "status"])
    # RESUME: if the output CSV already has rows, skip those (bench, N) and append the rest.
    done = set()
    if os.path.exists(args.out):
        try:
            with open(args.out) as f:
                for r in csv.DictReader(f):
                    if r.get("t_seq") not in (None, ""): done.add((r["bench"], int(r["N"])))
        except Exception:
            done = set()
    mode = "a" if done else "w"
    fp = open(args.out, mode, newline=""); w = csv.DictWriter(fp, fieldnames=hdr)
    if mode == "w": w.writeheader()
    fp.flush()
    if done: print(f"# RESUME: skipping {len(done)} completed rows already in {args.out}", flush=True)
    print("# " + " ".join(hdr[:2]) + "  parity  stored | " + " ".join(f"t_{l}" for l in labels)
          + " t_vfy | metal/ad omp/ad  status", flush=True)

    for name in names:
        gen, grid, startN, q = BENCH[name]
        Ns = range(startN, MAX_N + 1) if args.climb else grid
        for N in Ns:
            if (name, N) in done:
                continue   # already measured (resume)
            xml = os.path.join("/tmp", f"bench_{name}_{N}.xml"); qf = os.path.join("/tmp", f"bench_{name}.q")
            gen_model(gen, N, xml); open(qf, "w").write(q + "\n")
            # probe each config once (verdict, counts, rss, single-run wall)
            pr = {}
            ok = True
            for label, mk, env in CONFIGS:
                pr[label] = probe_ptctlmc(mk(xml, qf), env, args.timeout)
                if not pr[label][6]: ok = False
            if not ok:
                print(f"{name} {N}: backend failed ({ {l: pr[l][0] for l in labels} }); stop", flush=True); break
            vv, vex, vst, vrss, vel, vok = probe_verifyta(xml, qf, args.timeout)
            sv, sex, sst = pr["seq"][0], pr["seq"][1], pr["seq"][2]
            verds = {pr[l][0] for l in labels} | ({vv} if vok and vv not in ("?",) else set())
            verdict_ok = len(verds) == 1
            parity = ("EXACT" if (vst is not None and sst == vst and sex == vex)
                      else ("stored=" if vst is not None and sst == vst else (f"s{sst}v{vst}" if vst else "n/a")))
            peak_rss = max([pr[l][4] for l in labels if pr[l][4] is not None] + ([vrss] if vrss else []) + [0])
            gpu_active = pr["metal"][3]
            # timing (min-of-k, k by single-run wall)
            t = {}
            for label, mk, env in CONFIGS:
                t[label] = time_min(mk(xml, qf), env, pr[label][5], args.timeout)
            tv = time_min([VERIFYTA, xml, qf], dict(os.environ), vel, args.timeout) if vok else None
            f = lambda a, b: round(a / b, 2) if (a and b) else None
            row = dict(bench=name, N=N, verdict=sv, parity=parity, stored=sst, explored=sex,
                       vfy_stored=vst, vfy_explored=vex, t_verifyta=tv, rss_verifyta=vrss,
                       gpu=("ACTIVE" if gpu_active else "cpu-fallback"),
                       status=("OK" if verdict_ok else "VERDICT-MISMATCH"))
            for l in labels: row[f"t_{l}"] = t[l]; row[f"rss_{l}"] = pr[l][4]
            row["vfy/omp"] = f(tv, t["omp"]); row["vfy/metal"] = f(tv, t["metal"])
            row["vfy/metal-ad"] = f(tv, t["metal-ad"]); row["metal/metal-ad"] = f(t["metal"], t["metal-ad"])
            row["omp/metal-ad"] = f(t["omp"], t["metal-ad"])
            w.writerow(row); fp.flush()
            print(f"{name:<8}{N:>3} {parity:>7} {str(sst):>9} | "
                  + " ".join(f"{str(t[l]):>7}" for l in labels) + f" {str(tv):>6} | "
                  + f"{str(row['metal/metal-ad']):>7} {str(row['omp/metal-ad']):>6}  {row['status']}"
                  + ("" if gpu_active else " CPU-FALLBACK"), flush=True)
            # ceiling: the RAM backstop is ALWAYS active (protects the 24 GB machine); the wall-clock
            # cap only applies when climbing N adaptively.
            if peak_rss > MEM_CAP_GB:
                print(f"    ^ peak RSS {peak_rss}GB > {MEM_CAP_GB}GB — stopping {name} (RAM safety)", flush=True); break
            if args.climb and t["metal-ad"] and t["metal-ad"] > WALL_CAP:
                print(f"    ^ metal-ad {t['metal-ad']}s > {WALL_CAP}s — ceiling for {name}", flush=True); break
    fp.close()
    print(f"\ndone -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
