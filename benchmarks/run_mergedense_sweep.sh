#!/usr/bin/env bash
# MERGESCAN x HOTLEN sweep for ptctlmc-omp at chunk 8, on 32 / 16 / 8 isolated cores.
#
# Two one-dimensional sweeps through the current default (mergescan=32, hotlen=64), which is
# measured once and shared by both:
#
#   MERGESCAN  {4, 8, 16, 32, 64, 0=unbounded}   at hotlen=64
#     dbm_try_union is up to O(dim^5) per call (O(dim^2) facets x an O(dim^3) close), and step (4)
#     of NetPassed::add restarts the scan after every successful merge, so the window has a steep
#     known cost. The question is whether it buys anything: if merges really are recent
#     neighbours, 8 should capture nearly all of them and would be both a better default and a
#     stronger claim. This knob CHANGES WHAT IS STORED, so the `states` column is the point, not
#     an aside -- it is a precision/time tradeoff.
#
#   HOTLEN     {0=off, 16, 32, 64, 128}          at mergescan=32
#     At the default 64 a chain pays up to 64 packed-record decompressions per insertion before
#     the dense mirror engages -- exactly the cost the mirror exists to delete -- so 64 looks
#     conservative. What lowering it costs is memory: a dense mirror is dim^2 x 4 B per zone
#     against a packed record of roughly a third that, so `rss_gb` is the column to read against
#     `seconds`. 0 is the ABLATION that justifies the mechanism existing at all.
#
# 10 distinct configs x 3 thread counts = 30 harness invocations, all appending to ONE csv;
# --mergescan/--hotlen are recorded per row and are part of a run's identity, so nothing is
# measured twice and nothing is mistaken for another config's row.
#
# CORE ISOLATION: each invocation is pinned with taskset to exactly as many cores as it has
# threads (32 -> 0-31, 16 -> 0-15, 8 -> 0-7). Runs are strictly sequential, so a pinned run has
# its cores to itself. Nothing else is bound, matching the earlier sweeps this compares against.
#
# Launch detached (survives closing the terminal / ssh session):
#   cd .../PTCTCL/benchmarks
#   setsid nohup ./run_mergedense_sweep.sh > results/mergedense/driver.log 2>&1 < /dev/null &
#
# Stop everything: ./stop_mergedense_sweep.sh   (--soft / --status)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

OUTDIR="results/mergedense"
mkdir -p "$OUTDIR"

OUT="$OUTDIR/ptctlmc_thread_c8_merge_desne_2h.csv"
CHUNK=${CHUNK:-8}
TIMEOUT=${TIMEOUT:-7200}        # 2 h per (model, threads, rep, config)
MEMGB=${MEMGB:-100}             # box has 125 GB; a run over this is recorded MEMOUT, not crashed
REPS=${REPS:-2}
STRIKE=${STRIKE:-1}             # per (family, tool, mergescan, hotlen) -- streaks never cross configs

# Sizes are capped at the largest that COMPLETED in the 600 s reference sweep (t16 c8):
# phil15 / twolock15 / fischer15 / single7 all blew that budget, and a TIMEOUT row carries no
# state count, so at 2 h x 30 invocations they would cost ~180 h to learn nothing comparable.
#
# `single` is EXCLUDED by request. Measured before launch (2026-07-30), that exclusion is what
# decides what this sweep can find, so it is recorded here rather than discovered in the table:
# on fischer12 / twolock12 / viking17 the two knobs move NOTHING -- state counts identical to the
# digit and wall time inside 1.5% across ms in {4,32,0} and hl in {0,64}. That matches the comment
# on merge_scan() in netreach.cpp: chains are ~1 zone long on every process-network model, so the
# scan window has nothing to scan and the mirror never reaches its threshold. phil does vary, but
# it varies just as much at a FIXED config (71421/71665/71583 on three phil10 runs at 32:64), so
# that is its known parallel-merge nondeterminism, not the knob.
# single6, by contrast, responds hard: ms 32->3676 states/1.41s, 8->6064/4.50s, 0->4202/1.98s;
# hl 64->1.41s vs 0->9.30s. So the informative rows all live in the family that is excluded, and
# what the four families below establish is the (real, but negative) claim that the default is
# safe on process networks. Adding single back costs nothing extra: the harness never repeats a
# completed (model, tool, threads, rep, ms, hl), so re-running with single appended to FAMILIES
# fills in only the new rows.
# ONLY single, by decision after the pre-flight above: it is the one family where either knob
# binds at all, so it is the only one whose rows can show the precision/time tradeoff this sweep
# exists to measure. phil / twolock / fischer / viking are deliberately NOT run -- they were
# measured to return byte-identical state counts and sub-1.5% time across the whole config grid,
# so 20 of the 30 job-hours would have gone into reproducing one number ten times.
# Sizes 4-8 with STRIKE=1: single7 blew the 600 s budget at 16 threads but the routing fix has
# since cut single6 from 20278 zones/8.46 s to 3676/1.41 s, so 7 is plausible inside 2 h and 8 is
# there to be skipped automatically if 7 is not.
FAMILIES="single:4,5,6,7,8"
VIKING=""
FISCHER=""

# ms:hl, most informative first, so a sweep stopped early still answers the main questions:
# the default baseline, then the proposed better default, then the two extremes (unbounded scan,
# mirror off), then the rest of each axis.
CONFIGS=(32:64 8:64 0:64 32:0 4:64 16:64 64:64 32:16 32:32 32:128)

# 16 first: that is the thread count the current claim is stated at and the one with a 600 s
# reference sweep to compare against, so the complete story lands first if this is interrupted.
THREADLIST=${THREADLIST:-"16 32 8"}

STOP="$OUTDIR/STOP"
PIDFILE="$OUTDIR/mergedense.pid"

log() { echo "[$(date -Is)] $*"; }

stop_requested() {
  if [ -e "$STOP" ]; then
    log "STOP file present ($STOP) -- exiting without starting anything further."
    return 0
  fi
  return 1
}

# Kill leftovers and wait for the box to go quiet, so a finishing run can never bleed into the
# timings of the next one. Same policy as run_timeout2h.sh / run_chunk_sweep.sh.
settle() {
  pkill -9 -x ptctlmc 2>/dev/null
  sleep 20
  for _ in $(seq 1 60); do          # up to 10 min
    load=$(awk '{printf "%d", $1 * 10}' /proc/loadavg)
    [ "$load" -lt 20 ] && break     # 1-min load average below 2.0
    sleep 10
  done
}

if [ -e "$STOP" ]; then
  echo "ABORT: $STOP exists. Remove it before launching." >&2
  exit 1
fi
if ! command -v taskset >/dev/null 2>&1; then
  echo "ABORT: taskset not found -- core isolation is part of this experiment." >&2
  exit 1
fi

echo "$$" > "$PIDFILE"
ps -o pgid= -p $$ | tr -d ' ' > "$OUTDIR/mergedense.pgid"

log "driver pid $$ (pgid $(cat "$OUTDIR/mergedense.pgid")) on $(hostname), $(nproc) cores"
log "chunk=$CHUNK timeout=${TIMEOUT}s reps=$REPS memgb=$MEMGB strike=$STRIKE"
log "threads: $THREADLIST"
log "configs (mergescan:hotlen): ${CONFIGS[*]}"
log "out: $OUT"

njobs=0
for th in $THREADLIST; do
  if [ "$th" -gt "$(nproc)" ]; then
    log "SKIP threads=$th -- only $(nproc) cores available"
    continue
  fi
  cpus="0-$((th - 1))"
  for cfg in "${CONFIGS[@]}"; do
    ms=${cfg%%:*}; hl=${cfg##*:}
    stop_requested && { rm -f "$PIDFILE"; exit 3; }
    settle
    lg="$OUTDIR/bench_t${th}_c${CHUNK}_ms${ms}_hl${hl}.log"
    njobs=$((njobs + 1))
    log "START job $njobs/30  threads=$th cpus=$cpus mergescan=$ms hotlen=$hl -> $lg"
    # < /dev/null so the harness (and the tool it forks) cannot swallow this loop's stdin.
    taskset -c "$cpus" python3 bench_ltsmin.py --no-train \
        --threads "$th" --reps "$REPS" \
        --timeout "$TIMEOUT" --memgb "$MEMGB" --strike-limit "$STRIKE" \
        --only-ptctl-omp --omp-chunk "$CHUNK" \
        --mergescan "$ms" --hotlen "$hl" \
        --families "$FAMILIES" --viking $VIKING --fischer $FISCHER \
        --out "$OUT" < /dev/null >> "$lg" 2>&1
    rc=$?
    log "DONE  job $njobs  threads=$th ms=$ms hl=$hl rc=$rc rows=$(( $(wc -l < "$OUT" 2>/dev/null || echo 1) - 1 ))"
  done
done

settle
rm -f "$PIDFILE"
log "ALL DONE -- $OUT"
