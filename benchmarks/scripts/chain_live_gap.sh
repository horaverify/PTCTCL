#!/usr/bin/env bash
# Waits for the three sweeps in flight, then fills the LIVE gap: grc and hddi, three tools, one
# core each.
#
#   grc  was excluded from the live sweep by a STALE reason -- the message named `A[] not
#        deadlock`, which is the old REACH query; the live query is `T0.in --> T0.far` and is
#        SATISFIED (grc4 788 records, grc5 12184), so it exhausts the state space rather than
#        stopping at a counterexample.
#   hddi had no live query at all. `ST1.station_z_sync --> ST1.station_z_idle` is satisfied in
#        both hora and verifyta, and its tck_live models are generated now.
set -u
BENCH=/home/luca/Documents/ParallelCTLModelChecker/PTCTCL/benchmarks
HORA=/home/luca/Documents/ParallelCTLModelChecker/PTCTCL/build_shipping/hora
BP="python3 $BENCH/scripts/bench.py"
log() { echo "[$(date -Is)] $*"; }

log "waiting for the in-flight sweeps to finish..."
while pgrep -f "bench.py run --tool hora --queries eg" >/dev/null \
   || pgrep -f "bench.py run --tool verifyta --queries reach" >/dev/null \
   || pgrep -f "bench.py run --tool verifyta --queries deadlock" >/dev/null; do
  sleep 60
done
log "=== all clear; launching live grc+hddi on three cores ==="

$BP run --tool hora --queries live --families 'grc;hddi' \
   --threads 1 --reps 2 --timeout 7200 --memgb 100 --sim alu --cores 25-25 \
   --hora-bin "$HORA" --outdir "$BENCH/results/live/hora" --tag hora_live &
P1=$!
$BP run --tool verifyta --queries live --families 'grc;hddi' \
   --threads 1 --reps 2 --timeout 7200 --memgb 100 --sim alu --cores 26-26 \
   --outdir "$BENCH/results/live/uppaal" --tag verifyta_live &
P2=$!
$BP run --tool tchecker --queries live --families 'grc;hddi' \
   --threads 1 --reps 2 --timeout 7200 --memgb 100 --sim alu --cores 27-27 \
   --outdir "$BENCH/results/live/tchecker" --tag tchecker_live &
P3=$!
wait $P1; log "hora live done (rc=$?)"
wait $P2; log "verifyta live done (rc=$?)"
wait $P3; log "tchecker live done (rc=$?)"
log "=== ALL DONE ==="
