#!/usr/bin/env bash
# The complete sweep: PTCTLMC-omp, 2-hour / 100 GB budget, into results_complete/.
# NO verifyta in this sweep -- ptctlmc only. verifyta numbers come from results_old (it is an
# unchanged external binary and reproduces its recorded times to within 1-2%).
#
# TWO ENGINES, one binary, selected by PTCTLMC_ENGINE:
#
#   adaptive  (default) current convex union + adaptive dense-mirror promotion.
#             ALL families -- this is the headline result.
#
#   legacy    the 2026-07-29 evaluation engine: the pre-`10583cf` convex union
#             (dbm_try_union_legacy) with that era's fixed hl=64 mirror. Reproduces the
#             reference build to within 2% (phil13 12.27 vs 12.31 s, fischer12 14.80 vs 14.81,
#             viking17 8.50 vs 8.31).
#             ONLY single and csma -- by request, those are the two families whose reference
#             numbers are intact and worth a like-for-like legacy column.
#
# Both engines do convex merging. The ONLY difference is which try_union runs (plus the mirror
# policy). The dense mirror, the shared kill probe and the SpinRW stripe locks are common to
# both; each was suspected of the phil difference and each was cleared. The whole of it was
# `10583cf` adding a per-call mindbm_analyze, which prunes redundant facets -- a win at large
# dim, a loss at phil's.
#
# ORDER: 16 threads first, then 24, then 32 -- so the low-thread runs finish early and the
# machine is only fully committed at the end.
#
# CORE PINNING: each job is pinned to exactly as many cores as it has threads (16 -> 0-15,
# 24 -> 0-23, 32 -> 0-31). Unlike the earlier sweeps this is ON, because development work
# (compiles) runs concurrently on the remaining cores while 16- and 24-thread jobs are
# measured. Set PIN=0 to reproduce the older unpinned methodology.
#
# Within each (engine, threads, chunk) the CHEAP models run first and the ones known to need
# the full budget run LAST, so a complete cheap table exists early and the expensive tail never
# blocks it. The boundary is measured, not guessed (600 s reference sweep + the 2 h re-run):
#   cheap   twolock13 91 s, fischer13 93 s, phil14 118 s, viking20 159 s, twolock14 362 s,
#           fischer14 372 s -- all finished inside 600 s
#   slow    phil15 1085-1580 s OK, phil16 TIMEOUT@7200; single7 1622-1947 s OK, single8 TIMEOUT;
#           twolock15 ~1375 s, twolock16 ~5900 s; fischer15 ~1360 s, fischer16 ~5970 s
#
# Launch detached:
#   cd .../PTCTCL/benchmarks
#   setsid nohup ./run_complete.sh > results_complete/driver.log 2>&1 < /dev/null &
# Stop: ./stop_complete.sh   (--soft / --status)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

OUTROOT=${OUTROOT:-results_complete}
mkdir -p "$OUTROOT"

TIMEOUT=${TIMEOUT:-7200}        # 2 h per (model, tool, threads, rep) -- fair to both engines
MEMGB=${MEMGB:-100}             # box has 125 GB; over this is recorded MEMOUT, not crashed
REPS=${REPS:-2}
STRIKE=${STRIKE:-1}             # one timeout in a family skips every larger size of it
PIN=${PIN:-1}

THREADLIST=${THREADLIST:-"16 24 32"}
CHUNKLIST=${CHUNKLIST:-"8 4"}

# Families/sizes of the reference sweeps, split at the measured cheap/slow boundary.
ADA_FAST="phil:4,5,6,7,8,9,10,11,12,13,14;single:4,5,6;twolock:4,5,6,7,8,9,10,11,12,13,14"
ADA_VIK_FAST="4 6 10 12 15 17 20"
ADA_FIS_FAST="4 5 6 7 8 9 10 11 12 13 14"
ADA_CSMA_FAST="csma:4,5,6,7,8,9,10,11,12,13,14,15,16,17,18"
ADA_SLOW="phil:15,16,17;single:7,8;twolock:15,16;csma:20,22"
ADA_FIS_SLOW="15 16"

# legacy: single + csma only.
LEG_FAST="single:4,5,6;csma:4,5,6,7,8,9,10,11,12,13,14,15,16,17,18"
LEG_SLOW="single:7,8;csma:20,22"

STOP="$OUTROOT/STOP"
PIDFILE="$OUTROOT/complete.pid"

log() { echo "[$(date -Is)] $*"; }
stop_requested() { [ -e "$STOP" ] && { log "STOP present -- exiting."; return 0; }; return 1; }

# Kill leftovers and wait for the box to go quiet, so a finishing run never bleeds into the next
# one's timings. NOTE: this waits on the WHOLE machine's load, so a concurrent compile delays the
# next job rather than corrupting it.
settle() {
  pkill -9 -x ptctlmc 2>/dev/null
  sleep 20
  for _ in $(seq 1 60); do
    load=$(awk '{printf "%d", $1 * 10}' /proc/loadavg)
    [ "$load" -lt 40 ] && break
    sleep 10
  done
}

[ -e "$STOP" ] && { echo "ABORT: $STOP exists." >&2; exit 1; }
echo "$$" > "$PIDFILE"
ps -o pgid= -p $$ | tr -d ' ' > "$OUTROOT/complete.pgid"

log "driver pid $$ (pgid $(cat "$OUTROOT/complete.pgid")) on $(hostname), $(nproc) cores"
log "timeout=${TIMEOUT}s reps=$REPS memgb=$MEMGB strike=$STRIKE pin=$PIN"
log "threads: $THREADLIST   chunks: $CHUNKLIST"
log "PTCTCL $(git -C .. rev-parse --short HEAD)  HDBM $(git -C ../HDBM rev-parse --short HEAD)"

# One harness invocation. Resumable: bench_ltsmin.py never repeats a
# (model, tool, threads, rep, mergescan, hotlen) that already has a row in the output csv.
run_job() {
  local engine=$1 th=$2 ch=$3 fams=$4 vik=$5 fis=$6 out=$7 lg=$8
  local -a pre=()
  [ "$PIN" = "1" ] && pre=(taskset -c "0-$((th - 1))")
  local -a cmd=(python3 bench_ltsmin.py --no-train
                --only-ptctl-omp --omp-chunk "$ch"
                --threads "$th" --reps "$REPS"
                --timeout "$TIMEOUT" --memgb "$MEMGB" --strike-limit "$STRIKE"
                --families "$fams" --out "$out")
  cmd+=(--viking $vik)
  cmd+=(--fischer $fis)
  log "  PTCTLMC_ENGINE=$engine ${pre[*]} ${cmd[*]}"
  PTCTLMC_ENGINE="$engine" "${pre[@]}" "${cmd[@]}" < /dev/null >> "$lg" 2>&1
  log "DONE  engine=$engine t=$th c=$ch rc=$? rows=$(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 ))"
}

for th in $THREADLIST; do
  for ch in $CHUNKLIST; do
    AO="$OUTROOT/adaptive/ptctlmc_t${th}_c${ch}_adaptive_2h.csv"
    AL="$OUTROOT/adaptive/bench_t${th}_c${ch}_adaptive_2h.log"
    LO="$OUTROOT/legacy/ptctlmc_t${th}_c${ch}_legacy_2h.csv"
    LL="$OUTROOT/legacy/bench_t${th}_c${ch}_legacy_2h.log"
    mkdir -p "$OUTROOT/adaptive" "$OUTROOT/legacy"

    # --- adaptive, all families: cheap -> csma -> slow
    stop_requested && { rm -f "$PIDFILE"; exit 3; }; settle
    log "START adaptive/cheap  t=$th c=$ch"
    run_job adaptive "$th" "$ch" "$ADA_FAST" "$ADA_VIK_FAST" "$ADA_FIS_FAST" "$AO" "$AL"

    stop_requested && { rm -f "$PIDFILE"; exit 3; }; settle
    log "START adaptive/csma   t=$th c=$ch"
    run_job adaptive "$th" "$ch" "$ADA_CSMA_FAST" "" "" "$AO" "$AL"

    # --- legacy, single + csma only (cheap first)
    stop_requested && { rm -f "$PIDFILE"; exit 3; }; settle
    log "START legacy/cheap    t=$th c=$ch"
    run_job legacy "$th" "$ch" "$LEG_FAST" "" "" "$LO" "$LL"

    # --- the expensive tails, last
    stop_requested && { rm -f "$PIDFILE"; exit 3; }; settle
    log "START adaptive/slow   t=$th c=$ch"
    run_job adaptive "$th" "$ch" "$ADA_SLOW" "" "$ADA_FIS_SLOW" "$AO" "$AL"

    stop_requested && { rm -f "$PIDFILE"; exit 3; }; settle
    log "START legacy/slow     t=$th c=$ch"
    run_job legacy "$th" "$ch" "$LEG_SLOW" "" "" "$LO" "$LL"
  done
  log "=== THREADS=$th COMPLETE (both engines, both chunks)"
done

settle
rm -f "$PIDFILE"
log "ALL DONE -- $OUTROOT"
