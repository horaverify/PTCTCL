#!/usr/bin/env bash
# Only the families that actually have a gap. `--families all` is WRONG here: it fills every
# pre-existing hole in the table too (phil16 rep2 has been missing since HEAD and costs ~1h51m),
# which is not what this run is for.
#   hora reach:     hddi        -- newly measurable, needs t=24 and t=32 (t=16 already done)
#   verifyta reach: hddi        -- was ERR under the old malformed query
#   tchecker reach: hddi + grc  -- both were excluded, both now run under `A[] true`
#
# Phase 1 is hora alone (t=32 uses every core). Phase 2 is the three single-core jobs together.
set -u
BENCH=/home/luca/Documents/ParallelCTLModelChecker/PTCTCL/benchmarks
HORA=/home/luca/Documents/ParallelCTLModelChecker/PTCTCL/build_shipping/hora
BP="python3 $BENCH/scripts/bench.py"
log() { echo "[$(date -Is)] $*"; }

log "=== phase 1/2: hora reach hddi, t=16,24,32 -- ALONE ==="
$BP run --tool hora --queries reach --families hddi \
   --threads 16 24 32 --reps 2 --timeout 7200 --memgb 100 --sim g \
   --hora-bin "$HORA" --outdir "$BENCH/results/reach/hora" --tag hora
log "=== phase 1 done ==="

log "=== phase 2/2: verifyta(c30) + tchecker(c31) + hora E[](c24) in parallel ==="
$BP run --tool verifyta --queries reach --families hddi \
   --threads 1 --reps 2 --timeout 7200 --memgb 100 --sim alu --cores 30-30 \
   --outdir "$BENCH/results/reach/uppaal" --tag verifyta &
P1=$!
$BP run --tool tchecker --queries reach --families 'hddi;grc' \
   --threads 1 --reps 2 --timeout 7200 --memgb 100 --sim alu --cores 31-31 \
   --outdir "$BENCH/results/reach/tchecker" --tag tchecker &
P2=$!
$BP run --tool hora --queries eg --families all \
   --threads 1 --reps 2 --timeout 7200 --memgb 100 --sim alu --cores 24-24 \
   --hora-bin "$HORA" --outdir "$BENCH/results/eg/hora" --tag hora_eg &
P3=$!
wait $P1; log "verifyta reach done (rc=$?)"
wait $P2; log "tchecker reach done (rc=$?)"
wait $P3; log "hora E[] done (rc=$?)"
log "=== ALL DONE ==="
