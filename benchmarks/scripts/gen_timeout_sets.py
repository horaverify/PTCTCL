#!/usr/bin/env python3
"""Derive the 2-hour re-run work lists from the 600 s sweep results.

Reads every results/hora_t<T>_c<C>_v2.csv plus results/verifyta_only.csv and, per
configuration, emits the models that DID NOT PRODUCE A NUMBER at the 600 s budget:

  * TIMEOUT / MEMOUT   -- ran out of wall clock or RAM
  * ERR                -- crashed (twolock15/16 at 24 and 32 threads)
  * requested-but-absent -- never run at all because --strike-limit 1 skipped every larger
    size of a family after the first timeout. These have no row, and leaving them out
    would silently keep the hole that the longer budget is meant to fill.

A model counts as failed only if NONE of its rows is OK, so a config that timed out on
rep 1 but finished rep 2 is not re-run.

Output: results/timeout2h/worklist.tsv, one line per job, read by run_timeout2h.sh:

    tool<TAB>threads<TAB>chunk<TAB>families<TAB>viking<TAB>fischer

`families` is the --families string, `viking`/`fischer` are space-separated size lists
for the harness's dedicated flags ('-' when empty). Regenerating is safe and idempotent.
"""
import csv, glob, os, re, sys

from tool_names import canonical  # pre-rename CSVs say 'ptctlmc'

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmarks/
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "timeout2h")

# What each sweep run was ASKED to measure -- needed to spot the sizes that were skipped
# (no row at all) rather than measured. Taken from run_chunk_sweep.sh (the _v2 files) and
# from results/old/bench_t*_c8_v2.log (the c8 files, which also covered fischer 15,16).
REQ_V2 = {"phil": list(range(4, 18)), "single": list(range(4, 9)),
          "twolock": list(range(4, 17)), "viking": [4, 6, 10, 12, 15, 17, 20],
          "fischer": list(range(4, 15))}
REQ_C8 = dict(REQ_V2, fischer=list(range(4, 17)))
# verifyta_only.csv covered the same grid as the _v2 sweep.
REQ_VERIFYTA = REQ_V2

# Families driven by --families vs. by their own dedicated flag in bench_ltsmin.py.
DEDICATED = ("viking", "fischer")


def size(model):
    m = re.search(r"(\d+)$", model)
    return int(m.group(1)) if m else 0


def failed_models(csv_path, requested):
    """Models with no OK row: measured-and-failed, plus requested-but-never-run."""
    seen = {}
    for r in csv.DictReader(open(csv_path)):
        seen.setdefault(r["model"], set()).add(r["status"])
    out = {}
    for fam, sizes in requested.items():
        bad = []
        for n in sizes:
            m = "%s%d" % (fam, n)
            if m not in seen:
                bad.append((n, "SKIPPED"))
            elif "OK" not in seen[m]:
                bad.append((n, "/".join(sorted(seen[m]))))
        if bad:
            out[fam] = bad
    return out


def as_job(tool, threads, chunk, bad):
    fams = ";".join("%s:%s" % (f, ",".join(str(n) for n, _ in sorted(bad[f])))
                    for f in sorted(bad) if f not in DEDICATED)
    vik = " ".join(str(n) for n, _ in sorted(bad.get("viking", [])))
    fis = " ".join(str(n) for n, _ in sorted(bad.get("fischer", [])))
    return "\t".join([tool, str(threads), str(chunk), fams or "-", vik or "-", fis or "-"])


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs, report = [], []

    vpath = os.path.join(RES, "verifyta_only.csv")
    bad = failed_models(vpath, REQ_VERIFYTA)
    jobs.append(as_job("verifyta", 1, 0, bad))
    report.append(("verifyta_only.csv", bad))

    def tc(path):   # sort by (threads, chunk) NUMERICALLY -- a plain filename sort puts c16 first
        # Not anchored on `_v2.csv$`: the 600 s sweep files also carry the settings they were
        # produced with (hora_t16_c8_ms32_hl64_600s_v2.csv), now that mergescan/hotlen are
        # sweepable and a bare name no longer says which values a file used.
        m = re.search(r"hora_t(\d+)_c(\d+)", os.path.basename(path))
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    for path in sorted(glob.glob(os.path.join(RES, "hora_t*_c*_v2.csv")), key=tc):
        # Not anchored on `_v2.csv$`: the 600 s sweep files also carry the settings they were
        # produced with (hora_t16_c8_ms32_hl64_600s_v2.csv), now that mergescan/hotlen are
        # sweepable and a bare name no longer says which values a file used.
        m = re.search(r"hora_t(\d+)_c(\d+)", os.path.basename(path))
        if not m:
            continue
        T, C = int(m.group(1)), int(m.group(2))
        bad = failed_models(path, REQ_C8 if C == 8 else REQ_V2)
        jobs.append(as_job("hora-omp", T, C, bad))
        report.append((os.path.basename(path), bad))

    with open(os.path.join(OUT, "worklist.tsv"), "w") as fh:
        fh.write("# tool\tthreads\tchunk\tfamilies\tviking\tfischer\n")
        for j in jobs:
            fh.write(j + "\n")

    with open(os.path.join(OUT, "worklist.report.txt"), "w") as fh:
        for name, bad in report:
            fh.write("%s\n" % name)
            for fam in sorted(bad):
                for n, why in sorted(bad[fam]):
                    fh.write("    %-10s %s\n" % ("%s%d" % (fam, n), why))
            fh.write("\n")

    sys.stdout.write(open(os.path.join(OUT, "worklist.report.txt")).read())
    print("-> %s" % os.path.join(OUT, "worklist.tsv"))


if __name__ == "__main__":
    main()
