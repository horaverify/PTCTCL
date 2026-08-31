#!/usr/bin/env bash
set -euo pipefail

# Liveness sweep, tchecker THEN verifyta, strictly sequential.
#
# hora already has its liveness table (live_results/hora) and ltsmin is excluded from
# liveness for every family by UNMEASURABLE_LIVE (its cndfs decides Buchi emptiness with no
# time-divergence test, so it answers a different question), which is why those two are not
# re-run here. Both remaining tools resume from their existing CSV: rows already measured are
# seeded and skipped, so an interrupted sweep continues where it stopped.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LIVE_DIR="$BENCH_DIR/results/live"

export TMPDIR="$BENCH_DIR/.tmp"
mkdir -p "$TMPDIR"

UPPAAL_LIVE_DIR="$LIVE_DIR/uppaal"
TCHECKER_LIVE_DIR="$LIVE_DIR/tchecker"
mkdir -p "$UPPAAL_LIVE_DIR" "$TCHECKER_LIVE_DIR"

# The tchecker liveness models carry a leads-to observer generated from live_queries/, so they
# go stale the moment a query changes. Regenerate before measuring rather than trusting the
# checked-in copies -- tck-liveness would happily answer the old property.
echo "[tchecker] regenerating the leads-to observer models"
python3 "$SCRIPT_DIR/generators/gen_tck_live.py"

echo "=== Liveness pipeline (tchecker -> verifyta) started: $(date -Iseconds) ==="

echo "[tchecker] live, all families"
python3 "$SCRIPT_DIR/bench.py" run \
  --tool tchecker \
  --queries live \
  --families all \
  --outdir "$TCHECKER_LIVE_DIR" \
  --tag tchecker_live

echo "[verifyta] live, all families"
python3 "$SCRIPT_DIR/bench.py" run \
  --tool verifyta \
  --queries live \
  --families all \
  --outdir "$UPPAAL_LIVE_DIR" \
  --tag verifyta_live

echo "=== Liveness pipeline finished: $(date -Iseconds) ==="
