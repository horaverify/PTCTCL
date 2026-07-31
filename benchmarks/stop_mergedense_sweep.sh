#!/usr/bin/env bash
# EMERGENCY STOP for the MERGESCAN x HOTLEN sweep (run_mergedense_sweep.sh).
#
#   ./stop_mergedense_sweep.sh          stop everything now (driver + harness + running tool)
#   ./stop_mergedense_sweep.sh --soft   let the run in progress finish, then stop before the next
#   ./stop_mergedense_sweep.sh --status show what is running, without touching anything
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
OUTDIR="results/mergedense"
MODE=${1:-hard}

case "$MODE" in
  --status)
    echo "driver pid file : $(cat "$OUTDIR/mergedense.pid" 2>/dev/null || echo '(none -- not running)')"
    echo "process group   : $(cat "$OUTDIR/mergedense.pgid" 2>/dev/null || echo '(none)')"
    echo "STOP flag       : $([ -e "$OUTDIR/STOP" ] && echo present || echo absent)"
    echo "--- live processes:"
    pgrep -a -f "run_mergedense_sweep.sh|bench_ltsmin.py|ptctlmc" | grep -v stop_mergedense || echo "(none)"
    echo "--- rows so far, per config:"
    tr -d '\r' < "$OUTDIR/ptctlmc_thread_c8_merge_desne_2h.csv" 2>/dev/null \
      | awk -F, 'NR>1{printf "  t=%-3s ms=%-3s hl=%-4s %s\n", $4, $10, $11, $9}' \
      | sort | uniq -c | sort -k2 || echo "  (no csv yet)"
    echo "--- last driver log lines:"
    tail -5 "$OUTDIR/driver.log" 2>/dev/null || echo "(no driver.log)"
    exit 0
    ;;
  --soft)
    touch "$OUTDIR/STOP"
    echo "STOP flag set at $OUTDIR/STOP."
    echo "The driver exits before it starts the next config; the one in progress finishes."
    echo "To abort that one too, run: $0"
    exit 0
    ;;
esac

# Hard stop. The flag goes down FIRST so the driver cannot start a new run in the window between
# us killing the current tool and killing the driver itself.
touch "$OUTDIR/STOP"

PGID=$(cat "$OUTDIR/mergedense.pgid" 2>/dev/null)
if [ -n "${PGID:-}" ] && kill -0 -- "-$PGID" 2>/dev/null; then
  echo "killing process group $PGID (driver + harness + tool)"
  kill -TERM -- "-$PGID" 2>/dev/null
  sleep 3
  kill -KILL -- "-$PGID" 2>/dev/null
fi

# The benchmarked tool is started in its OWN session by the harness (start_new_session=True), so
# it is NOT in the driver's process group and survives the kill above. Take it out by name.
pkill -9 -f "run_mergedense_sweep.sh" 2>/dev/null
pkill -9 -f "bench_ltsmin.py"         2>/dev/null
pkill -9 -x ptctlmc                   2>/dev/null

rm -f "$OUTDIR/mergedense.pid"
sleep 1
echo "--- still alive (should be empty):"
pgrep -a -f "run_mergedense_sweep.sh|bench_ltsmin.py|ptctlmc" | grep -v stop_mergedense || echo "(none)"
echo
echo "Stopped. Rows already in $OUTDIR/*.csv are intact and complete."
echo "To resume: rm $OUTDIR/STOP  and relaunch run_mergedense_sweep.sh -- finished (model,"
echo "threads, rep, mergescan, hotlen) rows are not re-measured, so it picks up where it left off."
