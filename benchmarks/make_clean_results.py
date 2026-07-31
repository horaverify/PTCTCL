#!/usr/bin/env python3
"""Merge the 600 s benchmark results with the 2 h re-runs into benchmarks/clean_results/.

One output csv per benchmark configuration, same schema as the input:

    results/verifyta_only.csv        + results/timeout2h/verifyta_2h.csv
        -> clean_results/verifyta.csv
    results/ptctlmc_t32_c16_v2.csv   + results/timeout2h/ptctlmc_t32_c16_2h.csv
        -> clean_results/ptctlmc_t32_c16.csv

Substitution rule, applied per (family, model) cell:

  * the cell did not finish at 600 s (TIMEOUT / MEMOUT / ERR) and the 2 h run has
    rows for it  ->  every 600 s row of that cell is replaced by the 2 h rows,
    whatever they say. A 2 h TIMEOUT is kept, exactly as asked.
  * the cell did not finish and there is no 2 h row yet (the campaign is still
    running)      ->  the 600 s row is kept untouched.
  * the 2 h run has a cell that never appears in the 600 s file (e.g. phil17,
    which was only ever attempted with the 2 h budget) -> appended at the end.
  * OK cells are always kept from the 600 s run.

Every substitution is logged to clean_results/CLEANING_REPORT.txt, and the script
finally prints the verifyta / ptctlmc speedups: best first, then the averages.

Averages are geometric means -- the arithmetic mean of ratios is not a meaningful
summary, since a 10x win and a 0.1x loss must cancel, not average to 5.05x. The
single family is left out by default (--exclude); its numbers are still printed
separately so nothing disappears silently.

Usage:
    python3 make_clean_results.py                 # results/ -> clean_results/
    python3 make_clean_results.py --agg min       # use the fastest rep instead of the mean
    python3 make_clean_results.py --top 20        # longer speedup table
    python3 make_clean_results.py --exclude ''    # average over every family
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Statuses that mean "this cell produced a real, usable measurement".
OK_STATUS = {"OK"}
# Anything else (TIMEOUT, MEMOUT, ERR, ...) is a failed cell and is a candidate
# for replacement by its 2 h counterpart.


# --------------------------------------------------------------------------- io


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a benchmark csv, tolerating CRLF line endings and stray whitespace."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return [], []
        fields = [f.strip() for f in reader.fieldnames]
        rows = []
        for raw in reader:
            row = {}
            for key, value in raw.items():
                if key is None:
                    continue
                row[key.strip()] = (value or "").strip()
            if any(row.values()):
                rows.append(row)
    return fields, rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})


def cell_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("family", ""), row.get("model", ""))


def group_cells(rows: list[dict[str, str]]) -> "OrderedDict[tuple[str, str], list[dict[str, str]]]":
    """Group rows by (family, model), keeping first-appearance order."""
    groups: OrderedDict[tuple[str, str], list[dict[str, str]]] = OrderedDict()
    for row in rows:
        groups.setdefault(cell_key(row), []).append(row)
    return groups


# ------------------------------------------------------------------- discovery


def config_key(stem: str) -> str:
    """results file stem -> canonical config name (also the output file name)."""
    return re.sub(r"_(v2|only)$", "", stem)


def discover(results_dir: Path, twoh_dir: Path) -> list[tuple[str, Path, Path | None]]:
    """(config, 600s csv, 2h csv or None) for every top-level csv in results/."""
    pairs = []
    for base in sorted(results_dir.glob("*.csv")):
        cfg = config_key(base.stem)
        twoh = twoh_dir / f"{cfg}_2h.csv"
        pairs.append((cfg, base, twoh if twoh.is_file() else None))
    return pairs


# ------------------------------------------------------------------- cleaning


def clean_one(
    cfg: str, base_path: Path, twoh_path: Path | None
) -> tuple[list[str], list[dict[str, str]], list[str]]:
    """Merge one configuration. Returns (fields, rows, report lines)."""
    fields, base_rows = read_csv(base_path)
    base_groups = group_cells(base_rows)

    twoh_groups: "OrderedDict[tuple[str, str], list[dict[str, str]]]" = OrderedDict()
    if twoh_path is not None:
        twoh_fields, twoh_rows = read_csv(twoh_path)
        if twoh_fields and twoh_fields != fields:
            # Not fatal: we project onto the 600 s schema when writing.
            missing = [f for f in fields if f not in twoh_fields]
            if missing:
                print(
                    f"  warning: {twoh_path.name} is missing columns {missing}",
                    file=sys.stderr,
                )
        twoh_groups = group_cells(twoh_rows)

    report: list[str] = []
    out_rows: list[dict[str, str]] = []
    used_twoh: set[tuple[str, str]] = set()

    for key, rows in base_groups.items():
        failed = [r for r in rows if r.get("status") not in OK_STATUS]
        if not failed:
            out_rows.extend(rows)
            continue

        replacement = twoh_groups.get(key)
        old = "/".join(sorted({r.get("status", "?") for r in failed}))
        if replacement is None:
            report.append(
                f"  kept   {key[1]:<12} {old:<8} 600s  (no 2h result yet)"
            )
            out_rows.extend(rows)
            continue

        new = "/".join(sorted({r.get("status", "?") for r in replacement}))
        secs = ", ".join(r.get("seconds", "?") for r in replacement)
        report.append(
            f"  2h     {key[1]:<12} {old:<8} -> {new:<8} ({len(replacement)} row(s), {secs} s)"
        )
        out_rows.extend(replacement)
        used_twoh.add(key)

    # 2 h cells that never appear in the 600 s file at all.
    for key, rows in twoh_groups.items():
        if key in base_groups or key in used_twoh:
            continue
        status = "/".join(sorted({r.get("status", "?") for r in rows}))
        report.append(f"  added  {key[1]:<12} {'-':<8} -> {status:<8} (2h only)")
        out_rows.extend(rows)

    return fields, out_rows, report


# -------------------------------------------------------------------- speedups


def aggregate_seconds(rows: list[dict[str, str]], agg: str) -> float | None:
    """Wall-clock over the successful reps of one cell, or None if it never finished."""
    values = []
    for row in rows:
        if row.get("status") not in OK_STATUS:
            continue
        try:
            values.append(float(row["seconds"]))
        except (KeyError, ValueError):
            continue
    if not values:
        return None
    return min(values) if agg == "min" else statistics.mean(values)


def budget_seconds(rows: list[dict[str, str]]) -> float | None:
    """For a failed cell: the largest budget it was given (600 s or 7200 s)."""
    values = []
    for row in rows:
        try:
            values.append(float(row["seconds"]))
        except (KeyError, ValueError):
            continue
    return max(values) if values else None


def geomean(values: list[float]) -> float:
    """Geometric mean -- the right average for ratios: a 10x win and a 0.1x loss cancel."""
    return math.exp(statistics.fmean(math.log(v) for v in values))


def speedup_report(
    cleaned: dict[str, list[dict[str, str]]],
    agg: str,
    top: int,
    exclude: set[str],
    min_seconds: float,
) -> None:
    verifyta = cleaned.get("verifyta")
    if verifyta is None:
        print("\nno verifyta results found -- skipping the speedup report")
        return

    ref = {k: v for k, v in group_cells(verifyta).items()}
    ref_time = {k: aggregate_seconds(v, agg) for k, v in ref.items()}
    ref_budget = {k: budget_seconds(v) for k, v in ref.items()}

    exact: list[tuple[float, str, tuple[str, str], float, float]] = []
    lower: list[tuple[float, str, tuple[str, str], float, float]] = []

    for cfg, rows in sorted(cleaned.items()):
        if cfg == "verifyta":
            continue
        for key, cell in group_cells(rows).items():
            ours = aggregate_seconds(cell, agg)
            if ours is None or ours <= 0 or key not in ref:
                continue
            theirs = ref_time[key]
            if theirs is not None:
                exact.append((theirs / ours, cfg, key, theirs, ours))
            else:
                bound = ref_budget[key]
                if bound:
                    lower.append((bound / ours, cfg, key, bound, ours))

    label = "fastest rep" if agg == "min" else "mean over reps"
    print(f"\n{'=' * 78}\nverifyta / ptctlmc-omp speedup  ({label}, cleaned data)\n{'=' * 78}")

    if not exact:
        print("no (model, config) pair where both tools finished")
    else:
        exact.sort(reverse=True)
        best, cfg, key, theirs, ours = exact[0]
        print(
            f"\nHIGHEST SPEEDUP: {best:.2f}x  on {key[1]} with {cfg}"
            f"  (verifyta {theirs:.3f} s vs ptctlmc {ours:.3f} s)"
        )

        print(f"\ntop {min(top, len(exact))} of {len(exact)} pairs where both finished:")
        print(f"  {'speedup':>9}  {'model':<12} {'config':<16} {'verifyta':>10} {'ptctlmc':>10}")
        for sp, cfg, key, theirs, ours in exact[:top]:
            print(f"  {sp:>8.2f}x  {key[1]:<12} {cfg:<16} {theirs:>10.3f} {ours:>10.3f}")

        # Best config per model, so the table above cannot be dominated by one family.
        # min_seconds drops models both tools finish at their startup floor (~0.06 s),
        # where the ratio measures process spawn cost, not model checking.
        kept = [e for e in exact if e[2][0] not in exclude and e[3] >= min_seconds]
        per_model: "OrderedDict[str, tuple[float, str, float, float]]" = OrderedDict()
        for sp, cfg, key, theirs, ours in kept:
            if key[1] not in per_model:
                per_model[key[1]] = (sp, cfg, theirs, ours)
        faster = [(sp, m, cfg) for m, (sp, cfg, _, _) in per_model.items() if sp > 1.0]
        slower = [(sp, m, cfg) for m, (sp, cfg, _, _) in per_model.items() if sp <= 1.0]
        print(
            f"\nptctlmc faster on {len(faster)}/{len(per_model)} models "
            f"(best config each); verifyta faster on {len(slower)}"
        )
        if slower:
            worst = min(slower)
            print(f"  worst model for ptctlmc: {worst[1]} at {worst[0]:.2f}x ({worst[2]})")

        # ---- averages -------------------------------------------------------
        excl = ", ".join(sorted(exclude)) if exclude else "nothing"
        floor = f", verifyta >= {min_seconds:g} s" if min_seconds > 0 else ""
        print(f"\n{'-' * 78}\naverage speedup  (excluding: {excl}{floor})\n{'-' * 78}")

        if not kept:
            print("nothing left to average")
        else:
            best_each = [sp for sp, _, _, _ in per_model.values()]
            print(
                f"\nAVERAGE SPEEDUP: {geomean(best_each):.2f}x  "
                f"(geometric mean over {len(best_each)} models, best config each)"
                f"\n                 {statistics.fmean(best_each):.2f}x arithmetic, "
                f"{statistics.median(best_each):.2f}x median"
            )

            by_cfg: "OrderedDict[str, list[float]]" = OrderedDict()
            for sp, cfg, _, _, _ in kept:
                by_cfg.setdefault(cfg, []).append(sp)
            print("\nper configuration:")
            print(
                f"  {'config':<18} {'n':>3} {'geomean':>9} {'median':>8} "
                f"{'min':>8} {'max':>8}"
            )
            for cfg, sps in sorted(by_cfg.items(), key=lambda kv: -geomean(kv[1])):
                print(
                    f"  {cfg:<18} {len(sps):>3} {geomean(sps):>8.2f}x "
                    f"{statistics.median(sps):>7.2f}x {min(sps):>7.2f}x {max(sps):>7.2f}x"
                )

            # Configs solve different model sets, so also average over the models
            # every config managed -- that is the only apples-to-apples comparison.
            sets = [ {k[1] for _, c, k, _, _ in kept if c == cfg} for cfg in by_cfg ]
            common = set.intersection(*sets) if sets else set()
            if common and any(len(s) != len(common) for s in sets):
                print(f"\n  same, restricted to the {len(common)} models solved by every config:")
                for cfg in sorted(
                    by_cfg,
                    key=lambda c: -geomean([sp for sp, x, k, _, _ in kept if x == c and k[1] in common]),
                ):
                    sps = [sp for sp, x, k, _, _ in kept if x == cfg and k[1] in common]
                    print(f"  {cfg:<18} {len(sps):>3} {geomean(sps):>8.2f}x")

        dropped = [e for e in exact if e[2][0] in exclude]
        if dropped:
            drop_best: "OrderedDict[str, float]" = OrderedDict()
            for sp, _, key, _, _ in dropped:
                drop_best.setdefault(key[1], sp)
            print(
                f"\nexcluded from the averages: {len(dropped)} pair(s) over "
                f"{len(drop_best)} model(s) -- "
                + ", ".join(f"{m} {sp:.2f}x" for m, sp in drop_best.items())
            )
        tiny = {e[2][1] for e in exact if e[2][0] not in exclude and e[3] < min_seconds}
        if tiny:
            print(
                f"also below the {min_seconds:g} s floor, not averaged: "
                f"{len(tiny)} model(s) -- {', '.join(sorted(tiny))}"
            )

    if lower:
        lower.sort(reverse=True)
        print(
            "\nlower bounds -- verifyta never finished these, so the real speedup is larger:"
        )
        print(f"  {'speedup':>9}  {'model':<12} {'config':<16} {'budget':>10} {'ptctlmc':>10}")
        for sp, cfg, key, bound, ours in lower[:top]:
            print(f"  >={sp:>7.2f}x  {key[1]:<12} {cfg:<16} {bound:>10.3f} {ours:>10.3f}")


# ------------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "results", help="600 s results dir")
    ap.add_argument("--timeout2h", type=Path, default=None, help="2 h results dir (default: <results>/timeout2h)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: <results>/../clean_results)")
    ap.add_argument("--agg", choices=("mean", "min"), default="mean", help="how to combine reps in the speedup report")
    ap.add_argument("--top", type=int, default=15, help="rows in the speedup table")
    ap.add_argument(
        "--exclude",
        default="single",
        help="comma-separated families left out of the averages (default: single; '' to keep all)",
    )
    ap.add_argument(
        "--min-seconds",
        type=float,
        default=0.0,
        help="leave models verifyta solves faster than this out of the averages "
        "(default 0; try 1.0 to drop the startup-noise models)",
    )
    args = ap.parse_args()
    exclude = {f.strip() for f in args.exclude.split(",") if f.strip()}

    results_dir: Path = args.results
    twoh_dir: Path = args.timeout2h or (results_dir / "timeout2h")
    out_dir: Path = args.out or (results_dir.parent / "clean_results")

    if not results_dir.is_dir():
        print(f"error: no such directory: {results_dir}", file=sys.stderr)
        return 1
    if not twoh_dir.is_dir():
        print(f"warning: no 2h directory at {twoh_dir} -- copying results as-is", file=sys.stderr)

    pairs = discover(results_dir, twoh_dir)
    if not pairs:
        print(f"error: no csv files in {results_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "cleaning report -- 600 s results with the 2 h re-runs substituted in",
        f"  600 s results : {results_dir}",
        f"  2 h re-runs   : {twoh_dir}",
        f"  output        : {out_dir}",
        "",
    ]
    cleaned: dict[str, list[dict[str, str]]] = {}

    for cfg, base_path, twoh_path in pairs:
        fields, rows, notes = clean_one(cfg, base_path, twoh_path)
        if not fields:
            print(f"skipping empty file {base_path.name}", file=sys.stderr)
            continue
        out_path = out_dir / f"{cfg}.csv"
        write_csv(out_path, fields, rows)
        cleaned[cfg] = rows

        src = twoh_path.name if twoh_path else "none"
        subs = sum(1 for n in notes if n.lstrip().startswith(("2h", "added")))
        pending = sum(1 for n in notes if n.lstrip().startswith("kept"))
        print(
            f"{out_path.relative_to(out_dir.parent)}: {len(rows)} rows, "
            f"{subs} cell(s) from 2h, {pending} unresolved  [2h source: {src}]"
        )
        report_lines.append(f"{cfg}.csv   <- {base_path.name} + {src}")
        report_lines.extend(notes if notes else ["  (nothing to substitute)"])
        report_lines.append("")

    (out_dir / "CLEANING_REPORT.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {len(cleaned)} file(s) + CLEANING_REPORT.txt to {out_dir}")

    speedup_report(cleaned, args.agg, args.top, exclude, args.min_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
