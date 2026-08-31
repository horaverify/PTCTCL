#!/usr/bin/env bash
set -euo pipefail

# E[] sweep, tchecker THEN verifyta, strictly sequential -- both tools are single-threaded, so
# running them concurrently would only make each other's timings noisy.
#
# ptctlmc's E[] arm is measured separately (results_eg/ptctlmc). Both tools here resume from
# their existing CSV: a row that already finished is seeded and skipped.
#
# The 6 non-convex-clock queries (the last one of each family) are REFUSED by gen_tck_eg.py and
# land as SKIP with the reason from tck_live_e/manifest.json. That is a refusal, not a gap: a
# clock disjunction is not expressible as a TChecker invariant. verifyta still measures them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EG_DIR="$BENCH_DIR/results/eg"

export TMPDIR="$BENCH_DIR/.tmp"
mkdir -p "$TMPDIR" "$EG_DIR/uppaal" "$EG_DIR/tchecker"

# The psi-restricted models encode eg_queries/ directly, so they go stale the moment a query
# changes; regenerate rather than trust the checked-in copies.
echo "[tchecker] regenerating the psi-restricted E[] models"
python3 "$SCRIPT_DIR/generators/gen_tck_eg.py"

echo "=== E[] pipeline (tchecker -> verifyta) started: $(date -Iseconds) ==="

echo "[tchecker] eg, all families"
python3 "$SCRIPT_DIR/bench.py" run \
  --tool tchecker --queries eg --families all \
  --outdir "$EG_DIR/tchecker" --tag tchecker_eg

echo "[verifyta] eg, all families"
python3 "$SCRIPT_DIR/bench.py" run \
  --tool verifyta --queries eg --families all \
  --outdir "$EG_DIR/uppaal" --tag verifyta_eg

echo "=== E[] pipeline finished: $(date -Iseconds) ==="
