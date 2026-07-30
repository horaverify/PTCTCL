#!/usr/bin/env bash
# Sequential threads x omp-chunk sweep for ptctlmc-omp.
#
#   threads 16, 24, 32  x  --omp-chunk 2, 4, 16     -> 9 runs, one after the other
#
# Each run gets its OWN csv (results/ptctlmc_t<T>_c<C>_v2.csv) and is started with no
# --seed, so nothing is carried over or skipped from another run. Runs never overlap:
# the loop is strictly sequential, and between runs any stray ptctlmc is killed and the
# machine is left to settle until the load average drops back to idle. Families and
# sizes are exactly those measured in results/verifyta_only.csv.
#
# Launch detached (survives closing the ssh session):
#   setsid nohup ./run_chunk_sweep.sh > results/chunk_sweep.launcher.log 2>&1 < /dev/null &
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
mkdir -p results

THREADS=(16 24 32)
CHUNKS=(2 4 16)
REPS=2
TIMEOUT=600      # seconds per run, as in the c8_v2 runs
MEMGB=100
STRIKE=1         # 1 timeout in a family skips all larger sizes of it, as in the c8_v2 runs

# Exactly the families/sizes present in results/verifyta_only.csv.
FAMILIES="phil:4,5,6,7,8,9,10,11,12,13,14,15,16,17;single:4,5,6,7,8;twolock:4,5,6,7,8,9,10,11,12,13,14,15,16"
VIKING=(4 6 10 12 15 17 20)
FISCHER=(4 5 6 7 8 9 10 11 12 13 14)

# Pre-flight: never append to (or re-seed from) an existing csv -- that would silently
# skip runs and mix two measurements in one file.
for T in "${THREADS[@]}"; do
  for C in "${CHUNKS[@]}"; do
    if [ -e "results/ptctlmc_t${T}_c${C}_v2.csv" ]; then
      echo "ABORT: results/ptctlmc_t${T}_c${C}_v2.csv already exists; move it away first." >&2
      exit 1
    fi
  done
done

echo "$$" > results/chunk_sweep.pid

# Kill leftovers from the previous run and wait for the box to go quiet again, so a
# finishing run can never bleed into the timings of the next one.
settle() {
  pkill -9 -x ptctlmc 2>/dev/null
  sleep 20
  for _ in $(seq 1 60); do          # up to 10 min
    load=$(awk '{printf "%d", $1 * 10}' /proc/loadavg)
    [ "$load" -lt 20 ] && break     # 1-min load average below 2.0
    sleep 10
  done
}

echo "launcher pid $$ started $(date -Is) on $(hostname), $(nproc) cores"
for T in "${THREADS[@]}"; do
  for C in "${CHUNKS[@]}"; do
    OUT="results/ptctlmc_t${T}_c${C}_v2.csv"
    LOG="results/bench_t${T}_c${C}_v2.log"
    settle
    echo "=== $(date -Is)  START threads=$T chunk=$C -> $OUT"
    python3 bench_ltsmin.py \
      --only-ptctl-omp --no-train \
      --families "$FAMILIES" \
      --viking "${VIKING[@]}" \
      --fischer "${FISCHER[@]}" \
      --threads "$T" --omp-chunk "$C" --reps "$REPS" \
      --timeout "$TIMEOUT" --memgb "$MEMGB" --strike-limit "$STRIKE" \
      --out "$OUT" > "$LOG" 2>&1
    rc=$?
    echo "=== $(date -Is)  DONE  threads=$T chunk=$C rc=$rc rows=$(( $(wc -l < "$OUT" 2>/dev/null || echo 1) - 1 ))"
  done
done
settle
rm -f results/chunk_sweep.pid
echo "ALL DONE $(date -Is)"
