#!/usr/bin/env python3
"""Differential gate for engine changes: an old binary against a new one, same machine.

Runs both builds over the query suites and records, per case, the verdict, the reported
symbolic-state count, wall time and peak RSS.  A verdict disagreement is a hard failure and
aborts the sweep, because a timing comparison between builds that answer differently is
meaningless.

State counts are compared and reported but are NOT a failure: a change may legitimately merge
more zones (a tighter antichain) or settle earlier on a violated query.  Both show up as a
`states` delta in the output, so a change that was not supposed to move them is visible.

Run order alternates per case so neither build systematically enjoys a warm page cache.

    live_diff.py OLD_BIN NEW_BIN OUT.csv [--suites live,eg,reach,deadlock] [--sizes 4-10]

Peak RSS comes from `/usr/bin/time -l` on macOS and `/usr/bin/time -v` on Linux; where neither
is available the column is left empty rather than guessed.
"""

import argparse
import csv
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "benchmarks" / "models" / "xml"
FAMILIES = ("arbiter", "csma", "fischer", "grc", "phil", "twolock", "single", "viking")

# (suite name, query directory, or a literal query)
SUITES = {
    "live": ("queries/live", None),
    # Violated leads-to. The `live` suite is satisfied on every row, so without this the early
    # witness in Otf::leadsto_witness -- which only ever fires on a violation -- is untested.
    "violated": ("queries/live_violated", None),
    "eg": ("queries/eg", None),
    "reach": ("queries/reach", None),
    "deadlock": (None, "A[] not deadlock"),
}


def queries_for(suite, family):
    """Yield the queries this suite poses to this family."""
    directory, literal = SUITES[suite]
    if literal is not None:
        return [literal]
    path = ROOT / "benchmarks" / directory / f"{family}.q"
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def jobs(suites, sizes):
    for suite in suites:
        for family in FAMILIES:
            for index, query in enumerate(queries_for(suite, family), start=1):
                for size in sizes:
                    model = MODELS / f"{family}{size}.xml"
                    if model.is_file():
                        yield f"{suite}:{family}{size}:q{index}", model, query


def peak_rss_command():
    """A `/usr/bin/time` invocation that reports peak RSS, or None."""
    if not Path("/usr/bin/time").exists():
        return None
    return ["/usr/bin/time", "-l"] if platform.system() == "Darwin" else ["/usr/bin/time", "-v"]


def parse_peak_rss(text):
    """Peak RSS in bytes from a `/usr/bin/time` report, or None."""
    match = re.search(r"([0-9]+)\s+maximum resident set size", text)
    if match:
        return int(match.group(1))                      # macOS: bytes
    match = re.search(r"Maximum resident set size \(kbytes\): ([0-9]+)", text)
    if match:
        return int(match.group(1)) * 1024               # GNU: kilobytes
    return None


def measure(binary, model, query, timeout, timer):
    """Return (seconds, peak_rss_bytes|None, states|None, verdict)."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".q", delete=False)
    handle.write(query + "\n")
    handle.close()
    try:
        command = (timer or []) + [str(binary), "--backend", "omp", str(model), handle.name]
        environment = dict(os.environ, OMP_NUM_THREADS="1")
        started = time.perf_counter()
        # Own process group, killed as a group on timeout. subprocess's own timeout kills only
        # the direct child, which is /usr/bin/time -- the model checker under it survives and
        # spins forever, and enough orphans make every later measurement on the machine wrong.
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, env=environment, start_new_session=True)
        try:
            output = child.communicate(timeout=timeout)[0]
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            child.communicate()
            return None, None, None, "TIMEOUT"
        elapsed = time.perf_counter() - started
        states = re.search(r"explored ([0-9]+) symbolic", output)
        verdict = re.search(r"Formula is (NOT )?satisfied", output)
        if verdict is None:
            return elapsed, None, None, "ERROR: " + " ".join(output.split())[:160]
        return (elapsed, parse_peak_rss(output),
                states.group(1) if states else None,
                "NOT" if verdict.group(1) else "SAT")
    finally:
        os.unlink(handle.name)


COLUMNS = ["case", "query",
           "old_s", "old_rss_mb", "old_states", "old_verdict",
           "new_s", "new_rss_mb", "new_states", "new_verdict",
           "speedup_x", "rss_ratio_x", "states_delta", "agree"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("old_bin")
    parser.add_argument("new_bin")
    parser.add_argument("out_csv")
    parser.add_argument("--suites", default="live,eg")
    parser.add_argument("--sizes", default="4-9")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    lo, _, hi = args.sizes.partition("-")
    sizes = range(int(lo), int(hi or lo) + 1)
    suites = [s for s in args.suites.split(",") if s]
    for suite in suites:
        if suite not in SUITES:
            parser.error(f"unknown suite {suite!r}; choose from {', '.join(SUITES)}")

    timer = peak_rss_command()
    if timer is None:
        print("warning: no /usr/bin/time — RSS columns will be empty", file=sys.stderr)

    binaries = {"old": args.old_bin, "new": args.new_bin}
    work = list(jobs(suites, sizes))
    rows, mismatches = [], []

    for number, (name, model, query) in enumerate(work, start=1):
        row = dict.fromkeys(COLUMNS, "")
        row["case"], row["query"] = name, query
        for label in (["old", "new"] if number % 2 else ["new", "old"]):
            seconds, rss, states, verdict = measure(binaries[label], model, query,
                                                    args.timeout, timer)
            row[f"{label}_s"] = f"{seconds:.4f}" if seconds is not None else ""
            row[f"{label}_rss_mb"] = f"{rss / 1048576:.1f}" if rss else ""
            row[f"{label}_states"] = states or ""
            row[f"{label}_verdict"] = verdict

        # Only two real verdicts can disagree. A timeout or an error on one side is reported and
        # kept, but it is not evidence that the builds answer differently -- and a change whose
        # whole point is to decide a case the baseline could not would otherwise abort the sweep.
        real = {"SAT", "NOT"}
        if row["old_verdict"] in real and row["new_verdict"] in real:
            row["agree"] = "OK" if row["old_verdict"] == row["new_verdict"] else "MISMATCH"
        else:
            row["agree"] = "INCOMPARABLE"
        if row["old_s"] and row["new_s"]:
            row["speedup_x"] = f"{float(row['old_s']) / float(row['new_s']):.3f}"
        if row["old_rss_mb"] and row["new_rss_mb"]:
            row["rss_ratio_x"] = f"{float(row['old_rss_mb']) / float(row['new_rss_mb']):.3f}"
        if row["old_states"] and row["new_states"]:
            row["states_delta"] = int(row["new_states"]) - int(row["old_states"])

        rows.append(row)
        print(f"{number:4d}/{len(work)} {name:<24} "
              f"old {row['old_verdict']:<4} {row['old_s'] or '-':>9}s  "
              f"new {row['new_verdict']:<4} {row['new_s'] or '-':>9}s  "
              f"x{row['speedup_x'] or '-':<7} d_states={row['states_delta']!s:<8} {row['agree']}",
              flush=True)

        with open(args.out_csv, "w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        if row["agree"] == "MISMATCH":
            mismatches.append(row)
            print(f"\nVERDICT MISMATCH on {name}: "
                  f"old={row['old_verdict']} new={row['new_verdict']}\n  query: {query}",
                  file=sys.stderr)
            return 1

    comparable = [r for r in rows if r["old_s"] and r["new_s"] and r["agree"] == "OK"]
    old_total = sum(float(r["old_s"]) for r in comparable)
    new_total = sum(float(r["new_s"]) for r in comparable)
    moved = [r for r in rows if r["states_delta"] not in ("", 0)]
    timeouts = [r for r in rows if "TIMEOUT" in (r["old_verdict"], r["new_verdict"])]
    decided = [r for r in timeouts if r["new_verdict"] in ("SAT", "NOT")]
    errors = [r for r in rows if r["old_verdict"].startswith("ERROR")
              or r["new_verdict"].startswith("ERROR")]

    print(f"\n{len(rows)} cases, {len(mismatches)} verdict mismatches, "
          f"{len(timeouts)} with a timeout, {len(errors)} errors")
    print(f"total (comparable rows only): old {old_total:.2f}s  new {new_total:.2f}s  "
          f"{old_total / new_total:.3f}x" if new_total else "")
    if decided:
        print(f"{len(decided)} case(s) the baseline could not decide within the timeout:")
        for row in decided:
            print(f"  {row['case']:<24} new={row['new_verdict']} in {row['new_s']}s")
    if moved:
        print(f"{len(moved)} cases changed their state count:")
        for row in moved[:20]:
            print(f"  {row['case']:<24} {row['old_states']:>10} -> {row['new_states']:<10} "
                  f"({row['states_delta']:+})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
