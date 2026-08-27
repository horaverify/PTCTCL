#!/usr/bin/env bash
set -euo pipefail

# Master sequential pipeline:
#   1) clean up any stray tool processes from prior runs
#   2) reach continuation: tchecker (all families) -> ltsmin(16,24,32) (missing families)
#   3) liveness sweep: ptctlmc(16,24,32) -> ltsmin(16,24,32) -> verifyta -> tchecker, all families

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for exe in ptctlmc opaal2lts-mc verifyta tck-reach; do
  pkill -9 -x "$exe" 2>/dev/null || true
done

echo "=== Full pipeline started: $(date -Iseconds) ==="
"$SCRIPT_DIR/run_continue_pipeline.sh"
"$SCRIPT_DIR/run_live_pipeline.sh"
echo "=== Full pipeline finished: $(date -Iseconds) ==="
