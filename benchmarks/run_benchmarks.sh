#!/bin/bash
# Single entry point for the evaluation.
#
#   ./run_benchmarks.sh              # full three-way sweep over every family
#   ./run_benchmarks.sh --quick      # 1 rep, 60 s cap, small sizes (smoke test)
#   ./run_benchmarks.sh --prepare    # (re)generate models and LTSmin modules first
#
# Every run is capped at TIMEOUT seconds and MEMGB gigabytes; a run exceeding either is
# recorded as TIMEOUT/MEMOUT and never as a number. Results stream to results/full.csv
# after each run, and any (model, tool, threads, rep) already recorded OK is SKIPPED --
# so this is safe to interrupt and re-run, and it resumes where it stopped.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p results

OUT="results/full.csv"
TIMEOUT=600 # seconds, which is 10 minutes
MEMGB=100
THREADS=(4 8 12)
REPS=2

VIKING=( 6 12 15 17)
FISCHER=( 6 9 10 11)
# generated families: <name>:<N,N,N>. Each needs models/<name><N>.xml and queries/<name>.q.
FAMILIES="phil:6,8,10;twolock:7,9,11;arbiter:8,12,16;csma:8,11,14;grc:5,7,9"

for a in "$@"; do
  case "$a" in
    --quick)
      TIMEOUT=6; REPS=1; THREADS=(2 4 6)
      VIKING=(4 6 10); FISCHER=(4 5 6)
      FAMILIES="phil:6;twolock:7;arbiter:8;csma:8;grc:5"
      ;;
    --prepare)
      echo "==> generating models and LTSmin modules"
      python3 prepare_models.py --so
      ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

echo "==> three-way sweep: verifyta vs opaal+LTSmin vs PTCTLMC"
echo "    cap ${TIMEOUT}s / ${MEMGB}GB, threads ${THREADS[*]}, ${REPS} rep(s)"
echo "    -> $OUT (resumable; already-OK runs are skipped)"
python3 bench_ltsmin.py \
  --viking  "${VIKING[@]}" \
  --fischer "${FISCHER[@]}" \
  --families "$FAMILIES" \
  --threads "${THREADS[@]}" --reps "$REPS" \
  --timeout "$TIMEOUT" --memgb "$MEMGB" \
  --seed "$OUT" --out "$OUT" 2>&1 | tee -a results/bench.log

echo "==> regenerating figures and tables"
python3 plot_ltsmin.py
