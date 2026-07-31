#!/usr/bin/env bash
# Re-run, with a 2-HOUR wall budget, ONLY the benchmarks that failed to produce a number
# at the 600 s budget -- per tool and, for ptctlmc, per (threads, chunk) configuration.
#
# The work list is not hardcoded here: gen_timeout_sets.py derives it from the sweep CSVs
# into results/timeout2h/worklist.tsv, so what runs is exactly what failed. Regenerate it
# (./gen_timeout_sets.py) if the source CSVs change.
#
# TWO ISOLATED PHASES, never overlapping:
#   phase 1  verifyta, 1 thread                    -> results/timeout2h/verifyta_2h.csv
#   phase 2  ptctlmc-omp, one run per (T,C)        -> results/timeout2h/ptctlmc_t<T>_c<C>_2h.csv
#
# Phase 2 is GATED: it starts only once results/timeout2h/ptctl.enabled exists. Until then
# the driver sits in a poll loop after phase 1 (this is the "append" -- ptctlmc waits for
# UPPAAL to finish AND for the gate). Nothing about phase 2 can perturb phase 1's timings.
#
# Launch detached (survives closing the terminal / ssh session):
#   cd .../PTCTCL/benchmarks
#   setsid nohup ./run_timeout2h.sh > results/timeout2h/driver.log 2>&1 < /dev/null &
#
# Stop everything: ./stop_timeout2h.sh   (see results/timeout2h/HOW_TO_STOP.md)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

OUTDIR="results/timeout2h"
mkdir -p "$OUTDIR"

TIMEOUT=${TIMEOUT:-7200}       # 2 hours per (model, tool, threads, rep)
# 100 GB per run, same as the 600 s sweep (which never came close -- peak was 21 GB). With a 2 h
# budget the big twolock/phil sizes can run far past that, and since the zone-interner cap was
# raised out of the way, memory is now what stops them: a run over this is killed and recorded
# as MEMOUT rather than crashing. The box has 125 GB. Override with:  MEMGB=50 ./run_timeout2h.sh
MEMGB=${MEMGB:-100}
REPS=${REPS:-2}
# Within a family, one timeout at size N skips every larger size. The work lists are ordered
# ascending and these families grow ~5-8x per size, so a size larger than one that blew a
# 2 h budget cannot finish -- this only avoids burning 2 h per hopeless size. Set to 0 to
# force every listed size to be attempted.
STRIKE=${STRIKE:-1}

WORKLIST="$OUTDIR/worklist.tsv"
GATE="$OUTDIR/ptctl.enabled"     # phase 2 runs only once this exists
STOP="$OUTDIR/STOP"              # emergency brake: checked before every run
PIDFILE="$OUTDIR/timeout2h.pid"
GATE_WAIT_MAX=${GATE_WAIT_MAX:-172800}   # 48 h; give up waiting for the gate after this

log() { echo "[$(date -Is)] $*"; }

stop_requested() {
  if [ -e "$STOP" ]; then
    log "STOP file present ($STOP) -- exiting without starting anything further."
    return 0
  fi
  return 1
}

# Kill leftovers and wait for the box to go quiet, so a finishing run can never bleed into
# the timings of the next one. Same policy as run_chunk_sweep.sh.
settle() {
  pkill -9 -x ptctlmc 2>/dev/null
  pkill -9 -x verifyta 2>/dev/null
  sleep 20
  for _ in $(seq 1 60); do          # up to 10 min
    load=$(awk '{printf "%d", $1 * 10}' /proc/loadavg)
    [ "$load" -lt 20 ] && break     # 1-min load average below 2.0
    sleep 10
  done
}

if [ ! -s "$WORKLIST" ]; then
  echo "ABORT: $WORKLIST missing or empty. Run ./gen_timeout_sets.py first." >&2
  exit 1
fi
if [ -e "$STOP" ]; then
  echo "ABORT: $STOP exists. Remove it before launching." >&2
  exit 1
fi

# Record BOTH the pid and the process-group id: stop_timeout2h.sh kills the whole group, which
# is what actually takes the python harness and the tool it forked down with it.
echo "$$" > "$PIDFILE"
ps -o pgid= -p $$ | tr -d ' ' > "$OUTDIR/timeout2h.pgid"

log "driver pid $$ (pgid $(cat "$OUTDIR/timeout2h.pgid")) on $(hostname), $(nproc) cores"
log "timeout=${TIMEOUT}s reps=$REPS memgb=$MEMGB strike=$STRIKE"

# Families dropped from EVERY job, space-separated. Set to `single` on 2026-07-30 by request.
# The worklist is DERIVED (gen_timeout_sets.py) and is deliberately left intact -- filtering here
# instead means the exclusion is one variable, is printed in the driver log, and is undone without
# regenerating anything:  DROPFAM= ./run_timeout2h.sh   restores the full list.
#
# What it costs, from the one config that completed at 2 h (t16 c2): single was 3.1 h of that
# job's ~10 h. single7 does produce a real number the 600 s budget could not (114028 states in
# ~32 min), so that row is the genuine loss; single8 timed out at the full 2 h for a blank row,
# and strike-limit 1 means it is the only size that follows single7. Across the 12 remaining
# configs this saves roughly 1.5 days.
DROPFAM=${DROPFAM-single}

# Strip DROPFAM entries from a `fam:N,N;fam:N,N` spec, preserving order. Emits "-" if nothing is
# left, which run_job already understands as "pass no --families at all".
drop_families() {
  local spec=$1 keep="" part name d skip
  [ -z "${DROPFAM:-}" ] && { printf '%s' "$spec"; return; }
  [ "$spec" = "-" ] && { printf '%s' "$spec"; return; }
  local IFS=';'
  for part in $spec; do
    [ -z "$part" ] && continue
    name=${part%%:*}
    skip=0
    for d in $DROPFAM; do [ "$name" = "$d" ] && skip=1; done
    [ "$skip" = 1 ] && continue
    keep="${keep:+$keep;}$part"
  done
  printf '%s' "${keep:--}"
}

# One harness invocation. Reruns are resumable: bench_ltsmin.py never repeats a (model, tool,
# threads, rep) that already has an OK/TIMEOUT/MEMOUT row in the output csv, so relaunching
# after a stop picks up where it left off instead of starting over.
run_job() {
  local tool=$1 threads=$2 chunk=$3 families=$4 viking=$5 fischer=$6 out=$7 lg=$8
  local orig=$families
  families=$(drop_families "$families")
  [ "$families" != "$orig" ] && log "  DROPFAM='$DROPFAM': families '$orig' -> '$families'"
  # Nothing left to measure for this config -- skip rather than invoke the harness with no models.
  if [ "$families" = "-" ] && [ "$viking" = "-" ] && [ "$fischer" = "-" ]; then
    log "SKIP  $tool threads=$threads chunk=$chunk -- no models left after DROPFAM"
    return 0
  fi
  local -a cmd=(python3 bench_ltsmin.py --no-train
                --threads "$threads" --reps "$REPS"
                --timeout "$TIMEOUT" --memgb "$MEMGB" --strike-limit "$STRIKE"
                --out "$out")
  [ "$families" != "-" ] && cmd+=(--families "$families")
  [ "$viking"   != "-" ] && cmd+=(--viking $viking)
  [ "$fischer"  != "-" ] && cmd+=(--fischer $fischer)
  if [ "$tool" = "verifyta" ]; then
    cmd+=(--only-verifyta)
  else
    cmd+=(--only-ptctl-omp --omp-chunk "$chunk")
  fi
  log "START $tool threads=$threads chunk=$chunk -> $out"
  log "  ${cmd[*]}"
  # < /dev/null matters: the phase loops read the worklist on stdin, and without this the
  # harness (and the tool it forks) inherits that fd and could swallow the remaining jobs.
  "${cmd[@]}" < /dev/null >> "$lg" 2>&1
  local rc=$?
  log "DONE  $tool threads=$threads chunk=$chunk rc=$rc rows=$(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 ))"
}

# ---------------------------------------------------------------- phase 1: verifyta, 1 thread
log "=== PHASE 1: verifyta (1 thread)"
while IFS=$'\t' read -r tool threads chunk families viking fischer; do
  case "$tool" in \#*|"") continue;; esac
  [ "$tool" = "verifyta" ] || continue
  stop_requested && exit 3
  settle
  run_job "$tool" "$threads" "$chunk" "$families" "$viking" "$fischer" \
          "$OUTDIR/verifyta_2h.csv" "$OUTDIR/verifyta_2h.log"
done < "$WORKLIST"
log "=== PHASE 1 COMPLETE"

# ------------------------------------------------------------------------- gate before phase 2
if [ ! -e "$GATE" ]; then
  log "waiting for the phase-2 gate ($GATE) -- 'touch' it to let ptctlmc start"
  waited=0
  while [ ! -e "$GATE" ]; do
    stop_requested && exit 3
    sleep 60
    waited=$((waited + 60))
    [ $((waited % 1800)) -eq 0 ] && log "  still waiting for gate (${waited}s)"
    if [ "$waited" -ge "$GATE_WAIT_MAX" ]; then
      log "gate never appeared after ${GATE_WAIT_MAX}s -- exiting. Phase 1 results are complete."
      rm -f "$PIDFILE"; exit 4
    fi
  done
fi
log "gate present -- starting phase 2"

# ------------------------------------------------- phase 2: ptctlmc-omp, one run per (T, C)
log "=== PHASE 2: ptctlmc-omp"
while IFS=$'\t' read -r tool threads chunk families viking fischer; do
  case "$tool" in \#*|"") continue;; esac
  [ "$tool" = "ptctlmc-omp" ] || continue
  stop_requested && exit 3
  settle
  run_job "$tool" "$threads" "$chunk" "$families" "$viking" "$fischer" \
          "$OUTDIR/ptctlmc_t${threads}_c${chunk}_2h.csv" \
          "$OUTDIR/bench_t${threads}_c${chunk}_2h.log"
done < "$WORKLIST"
log "=== PHASE 2 COMPLETE"

settle
rm -f "$PIDFILE"
log "ALL DONE"
