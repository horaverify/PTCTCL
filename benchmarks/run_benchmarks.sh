#!/bin/bash
# Single entry point for the evaluation.
#
#   ./run_benchmarks.sh              # full three-way sweep, resumes from existing results
#   ./run_benchmarks.sh --quick      # viking+fischer only, 1 rep, 5 min cap (smoke test)
#   ./run_benchmarks.sh --prepare    # (re)generate models and LTSmin modules first
#
# Results stream to results/ltsmin_raw.csv after every run and the harness SKIPS any
# (model, tool, threads, rep) already recorded OK, so it is safe to interrupt and
# re-run. Figures and tables are regenerated from that CSV by plot_ltsmin.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p results

ARGS=(--threads 4 8 12 --reps 2 --timeout 1800 --memgb 15)
for a in "$@"; do
  case "$a" in
    --quick)   ARGS=(--threads 4 8 --reps 1 --timeout 300 --memgb 15) ;;
    --prepare) python3 prepare_models.py --so ;;
    *)         ARGS+=("$a") ;;
  esac
done

echo "==> benchmarking (results/ltsmin_raw.csv, resumable)"
python3 bench_ltsmin.py "${ARGS[@]}" --seed results/ltsmin_raw.csv 2>&1 | tee -a results/bench.log

echo "==> regenerating figures and tables"
python3 plot_ltsmin.py
echo "==> done. Rebuild the paper with:  (cd ../docs/paper-openmp && make)"
