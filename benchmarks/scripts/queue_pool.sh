#!/usr/bin/env bash
# A JOB POOL, not a barrier.
#
# Keeps at most MAX bench.py sweeps in flight and starts the next queued one as soon as ANY of
# them exits. The barrier version this replaces waited for every in-flight sweep to drain, which
# idles cores for hours behind whichever job is slowest -- verifyta deadlock walls at 7200 s per
# family, so the live gap would have sat untouched all night behind it.
#
# Each queued job is pinned to its own core, so the order they start in never matters, and every
# one is resumable: a job that was already partly run is seeded from its CSV and fills only its
# gaps.
set -u
MAX=3
BENCH=/home/luca/Documents/ParallelCTLModelChecker/PTCTCL/benchmarks
HORA=/home/luca/Documents/ParallelCTLModelChecker/PTCTCL/build_shipping/hora
log() { echo "[$(date -Is)] $*"; }

# Count sweeps, not this script: `bench.py run` appears only in the python command lines.
inflight() { pgrep -f "bench\.py run" 2>/dev/null | wc -l; }

# core : tool : queries : families : outdir : tag : extra
JOBS=(
  "25:hora:live:grc;hddi:$BENCH/results/live/hora:hora_live:--hora-bin $HORA"
  "26:verifyta:live:grc;hddi:$BENCH/results/live/uppaal:verifyta_live:"
  "27:tchecker:live:grc;hddi:$BENCH/results/live/tchecker:tchecker_live:"
)

pids=()
for job in "${JOBS[@]}"; do
  IFS=':' read -r core tool queries fams outdir tag extra <<< "$job"
  while [ "$(inflight)" -ge "$MAX" ]; do sleep 30; done
  log "slot free ($(inflight)/$MAX) -- starting $tool $queries [$fams] on core $core"
  rm -f "$outdir/STOP"
  # shellcheck disable=SC2086
  python3 "$BENCH/scripts/bench.py" run --tool "$tool" --queries "$queries" --families "$fams" \
     --threads 1 --reps 2 --timeout 7200 --memgb 100 --sim alu --cores "$core-$core" \
     $extra --outdir "$outdir" --tag "$tag" > "$outdir/gap.log" 2>&1 &
  pids+=($!)
  sleep 15          # let the new sweep register before the next inflight check
done

log "all queued jobs started; waiting"
for p in "${pids[@]}"; do
  wait "$p"
  log "sweep $p finished (rc=$?)"
done
log "=== ALL DONE ==="
