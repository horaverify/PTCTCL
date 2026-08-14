#!/usr/bin/env bash
# EMERGENCY STOP for the 2-hour re-run (run_timeout2h.sh).
#
#   ./stop_timeout2h.sh          stop everything now (driver + harness + running tool)
#   ./stop_timeout2h.sh --soft   let the run in progress finish, then stop before the next one
#   ./stop_timeout2h.sh --status show what is running, without touching anything
#
# See results/timeout2h/HOW_TO_STOP.md.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
OUTDIR="results/timeout2h"
MODE=${1:-hard}

case "$MODE" in
  --status)
    echo "driver pid file : $(cat "$OUTDIR/timeout2h.pid" 2>/dev/null || echo '(none -- not running)')"
    echo "process group   : $(cat "$OUTDIR/timeout2h.pgid" 2>/dev/null || echo '(none)')"
    echo "STOP flag       : $([ -e "$OUTDIR/STOP" ] && echo present || echo absent)"
    echo "phase-2 gate    : $([ -e "$OUTDIR/ptctl.enabled" ] && echo present || echo absent)"
    echo "--- live processes:"
    pgrep -a -f "run_timeout2h.sh|bench_ltsmin.py|ptctlmc|verifyta" | grep -v stop_timeout2h || echo "(none)"
    echo "--- last driver log lines:"
    tail -5 "$OUTDIR/driver.log" 2>/dev/null || echo "(no driver.log)"
    exit 0
    ;;
  --soft)
    touch "$OUTDIR/STOP"
    echo "STOP flag set at $OUTDIR/STOP."
    echo "The driver exits before it starts the next benchmark; the one in progress finishes"
    echo "(up to 2 h). To abort that one too, run: $0"
    exit 0
    ;;
esac

# Hard stop. The flag goes down FIRST so the driver cannot start a new run in the window
# between us killing the current tool and killing the driver itself.
touch "$OUTDIR/STOP"

PGID=$(cat "$OUTDIR/timeout2h.pgid" 2>/dev/null)
if [ -n "${PGID:-}" ] && kill -0 -- "-$PGID" 2>/dev/null; then
  echo "killing process group $PGID (driver + harness + tool)"
  kill -TERM -- "-$PGID" 2>/dev/null
  sleep 3
  kill -KILL -- "-$PGID" 2>/dev/null
fi

# Belt and braces: the benchmarked tools are started in their OWN session by the harness
# (start_new_session=True), so they are NOT in the driver's process group and survive the
# kill above. Take them out by name.
pkill -9 -f "run_timeout2h.sh"  2>/dev/null
pkill -9 -f "bench_ltsmin.py"   2>/dev/null
pkill -9 -x ptctlmc             2>/dev/null
pkill -9 -x verifyta            2>/dev/null
pkill -9 -f "verifyta -u -s"    2>/dev/null

rm -f "$OUTDIR/timeout2h.pid"
sleep 1
echo "--- still alive (should be empty):"
pgrep -a -f "run_timeout2h.sh|bench_ltsmin.py|ptctlmc|verifyta" | grep -v stop_timeout2h || echo "(none)"
echo
echo "Stopped. Results already written to $OUTDIR/*.csv are intact and complete rows."
echo "To resume later: rm $OUTDIR/STOP  and relaunch run_timeout2h.sh -- finished rows are"
echo "not re-measured, so it picks up where it left off."
