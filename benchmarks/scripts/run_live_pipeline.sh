#!/usr/bin/env bash
set -euo pipefail

# Liveness sweep: every tool, every family with a live query, strictly sequential:
#   ptctlmc (t16 -> t24 -> t32) -> ltsmin (t16 -> t24 -> t32) -> verifyta -> tchecker
# Each bench.py invocation is itself sequential across threads and families (see bench.py).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Liveness results sit BESIDE the reachability results, not inside them: they are a separate
# measurement (different query set, different tools answering), not a subset of the reach sweep.
LIVE_DIR="$BENCH_DIR/live_results"

export TMPDIR="$BENCH_DIR/.tmp"
mkdir -p "$TMPDIR"

PTCTLMC_LIVE_DIR="$LIVE_DIR/ptctlmc"
LTSMIN_LIVE_DIR="$LIVE_DIR/ltsmin"
UPPAAL_LIVE_DIR="$LIVE_DIR/uppaal"
TCHECKER_LIVE_DIR="$LIVE_DIR/tchecker"

mkdir -p "$PTCTLMC_LIVE_DIR" "$LTSMIN_LIVE_DIR" "$UPPAAL_LIVE_DIR" "$TCHECKER_LIVE_DIR"

PTCTLMC_BIN="${PTCTLMC_BIN:-$BENCH_DIR/../build_shipping/ptctlmc}"

# The tchecker liveness models carry a leads-to observer generated from live_queries/, so they
# go stale the moment a query changes. Regenerate before measuring rather than trusting the
# checked-in copies -- tck-liveness would happily answer the old property.
echo "[tchecker] regenerating the leads-to observer models"
python3 "$SCRIPT_DIR/gen_tck_live.py"

echo "=== Liveness pipeline started: $(date -Iseconds) ==="

# ONE THREAD COUNT, not three. The liveness engine has no OpenMP in it at all (see
# KNOWN_LIMITATION.md #6): wall time and peak RSS are identical at 1, 4, 16 and 32 threads, so a
# t16/t24/t32 sweep measures the same sequential run three times and triples a multi-day job for
# no information. t16 is kept rather than t1 because it is the column live_results/ptctlmc_old
# holds for the previous engine, so the two stay directly comparable.
echo "[ptctlmc] live, threads 1 -- the liveness engine is sequential, all families"
PTCTLMC_BIN="$PTCTLMC_BIN" python3 "$SCRIPT_DIR/bench.py" run \
  --tool ptctlmc \
  --queries live \
  --families all \
  --threads 1 \
  --outdir "$PTCTLMC_LIVE_DIR" \
  --tag ptctlmc_live \
  --ptctlmc-bin "$PTCTLMC_BIN"

echo "[ltsmin] live, threads 16 -> 24 -> 32 (sequential), all families"
python3 "$SCRIPT_DIR/bench.py" run \
  --tool ltsmin \
  --queries live \
  --families all \
  --threads 16 24 32 \
  --outdir "$LTSMIN_LIVE_DIR" \
  --tag ltsmin_live

echo "[verifyta] live, all families"
python3 "$SCRIPT_DIR/bench.py" run \
  --tool verifyta \
  --queries live \
  --families all \
  --outdir "$UPPAAL_LIVE_DIR" \
  --tag verifyta_live

echo "[tchecker] live, all families"
python3 "$SCRIPT_DIR/bench.py" run \
  --tool tchecker \
  --queries live \
  --families all \
  --outdir "$TCHECKER_LIVE_DIR" \
  --tag tchecker_live

echo "=== Liveness pipeline finished: $(date -Iseconds) ==="
