#!/usr/bin/env python3
"""PTCTLMC peak-memory (RSS) sweep — the memory-footprint companion to bench_run.py.

For every (model, N) that appears in benchmarks/results_stress.csv this measures the
PEAK resident-set size (RSS, in GB, via /usr/bin/time -l) of ALL FOUR backends:

  seq       one CPU core
  omp       12-thread OpenMP  (OMP_NUM_THREADS=12)
  metal     Apple GPU + 12 CPU cores
  verifyta  single-threaded UPPAAL 5.0 (the reference tool)

RSS is deterministic enough that a single run per config suffices (no min-of-k). A
config whose single run exceeds the timeout (default 240 s) is recorded as blank
(OOM/timeout) — that is expected data at the top of the scaling curve on the 24 GB
machine, and the guard keeps us out of heavy swap.

The models/queries are exactly bench_run.py's BENCH dict (imported, not copied), so
this profiles the identical state spaces as the timing run. For the omp and metal
backends the peak RSS already lives in results_stress.csv (columns rss_omp_gb /
rss_metal_gb); we reuse those to save time and only re-measure when a cell is blank.
seq and verifyta are NOT in that CSV, so they are always measured here.

  python3 benchmarks/mem_profile.py                 -> benchmarks/mem_profile.csv
  python3 benchmarks/mem_profile.py --timeout 240

Output columns: bench,N,stored,rss_seq_gb,rss_omp_gb,rss_metal_gb,rss_verifyta_gb
Rows are flushed immediately so a long run survives interruption.
"""
import argparse, csv, os, sys

# Reuse bench_run's definitions verbatim (importing does not run its main()).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_run import BENCH, BIN, VERIFYTA, gen_model, peak_rss_gb  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRESS_CSV = os.path.join(ROOT, "benchmarks", "results_stress.csv")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def configs_from_stress():
    """Ordered [(bench, N, stored, rss_omp_gb, rss_metal_gb)] from the timing run."""
    out = []
    with open(STRESS_CSV) as fp:
        for r in csv.DictReader(fp):
            out.append((r["bench"], int(r["N"]), r.get("stored", ""),
                        fnum(r.get("rss_omp_gb")), fnum(r.get("rss_metal_gb"))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=240, help="per-run timeout (s); over -> blank")
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "mem_profile.csv"))
    args = ap.parse_args()
    if not os.path.exists(BIN):
        sys.exit(f"binary not found: {BIN}  (build first, or set PTCTLMC_BIN)")

    e_seq = dict(os.environ)
    e_omp = dict(os.environ); e_omp["OMP_NUM_THREADS"] = "12"

    hdr = ["bench", "N", "stored", "rss_seq_gb", "rss_omp_gb", "rss_metal_gb", "rss_verifyta_gb"]
    fp = open(args.out, "w", newline="")
    w = csv.DictWriter(fp, fieldnames=hdr); w.writeheader(); fp.flush()
    print(f"# writing {args.out}  (timeout={args.timeout}s; seq/verifyta measured, omp/metal reused when present)")
    print(f"{'bench':<9}{'N':>3} {'stored':>10} {'seq':>7} {'omp':>7} {'metal':>7} {'vfy':>7}  (GB)", flush=True)

    for name, N, stored, rss_omp, rss_metal in configs_from_stress():
        gen, _, _, q = BENCH[name]
        xml = os.path.join("/tmp", f"mem_{name}_{N}.xml")
        qf = os.path.join("/tmp", f"mem_{name}.q")
        gen_model(gen, N, xml)
        open(qf, "w").write(q + "\n")

        # seq / verifyta: always measured here (not in results_stress.csv).
        rss_seq = peak_rss_gb([BIN, "--backend", "seq", xml, qf], e_seq, args.timeout)
        rss_vfy = (peak_rss_gb([VERIFYTA, "-u", xml, qf], e_seq, args.timeout)
                   if os.path.exists(VERIFYTA) else None)
        # omp / metal: reuse the timing run's peak RSS; measure only if that cell was blank.
        if rss_omp is None:
            rss_omp = peak_rss_gb([BIN, "--backend", "omp", xml, qf], e_omp, args.timeout)
        if rss_metal is None:
            rss_metal = peak_rss_gb([BIN, "--backend", "metal", xml, qf], e_seq, args.timeout)

        row = dict(bench=name, N=N, stored=stored, rss_seq_gb=rss_seq, rss_omp_gb=rss_omp,
                   rss_metal_gb=rss_metal, rss_verifyta_gb=rss_vfy)
        w.writerow(row); fp.flush()
        print(f"{name:<9}{N:>3} {str(stored):>10} {str(rss_seq):>7} {str(rss_omp):>7} "
              f"{str(rss_metal):>7} {str(rss_vfy):>7}", flush=True)

    fp.close()
    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()
