#!/usr/bin/env bash
# EMERGENCY STOP for the complete sweep (run_complete.sh).
#
#   ./stop_complete.sh          stop everything now (driver + harness + running tool)
#   ./stop_complete.sh --soft   let the run in progress finish, then stop before the next one
#   ./stop_complete.sh --status show what is running, without touching anything
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
OUTROOT=${OUTROOT:-results_complete}
MODE=${1:-hard}

case "$MODE" in
  --status)
    echo "driver pid file : $(cat "$OUTROOT/complete.pid" 2>/dev/null || echo '(none -- not running)')"
    echo "process group   : $(cat "$OUTROOT/complete.pgid" 2>/dev/null || echo '(none)')"
    echo "STOP flag       : $([ -e "$OUTROOT/STOP" ] && echo present || echo absent)"
    echo "--- live processes:"
    pgrep -a -f "run_complete.sh|bench_ltsmin.py|ptctlmc" | grep -v stop_complete || echo "(none)"
    echo "--- rows so far:"
    for f in "$OUTROOT"/adaptive/*.csv "$OUTROOT"/legacy/*.csv; do
      [ -e "$f" ] || continue
      printf "  %-58s %s\n" "$(basename "$f")" "$(( $(wc -l < "$f") - 1 ))"
    done
    echo "--- last driver log lines:"
    tail -6 "$OUTROOT/driver.log" 2>/dev/null || echo "(no driver.log)"
    exit 0
    ;;
  --soft)
    touch "$OUTROOT/STOP"
    echo "STOP flag set at $OUTROOT/STOP."
    echo "The driver exits before it starts the next benchmark; the one in progress finishes"
    echo "(up to 2 h). To abort that one too, run: $0"
    exit 0
    ;;
esac

# Hard stop. The flag goes down FIRST so the driver cannot start a new run in the window
# between us killing the current tool and killing the driver itself.
touch "$OUTROOT/STOP"

PGID=$(cat "$OUTROOT/complete.pgid" 2>/dev/null)
if [ -n "${PGID:-}" ] && kill -0 -- "-$PGID" 2>/dev/null; then
  echo "killing process group $PGID (driver + harness + tool)"
  kill -TERM -- "-$PGID" 2>/dev/null
  sleep 3
  kill -KILL -- "-$PGID" 2>/dev/null
fi

# Belt and braces: the benchmarked tool is started in its OWN session by the harness
# (start_new_session=True), so it is NOT in the driver's process group and survives the
# kill above. Take it out by name.
pkill -9 -f "run_complete.sh" 2>/dev/null
pkill -9 -f "bench_ltsmin.py" 2>/dev/null
pkill -9 -x ptctlmc           2>/dev/null

rm -f "$OUTROOT/complete.pid"
sleep 2
echo "--- still alive (should be empty):"
pgrep -a -f "run_complete.sh|bench_ltsmin.py|ptctlmc" | grep -v stop_complete || echo "(none)"
echo
echo "Stopped. Rows already written to $OUTROOT are intact and complete."
echo "To resume: rm $OUTROOT/STOP  and relaunch run_complete.sh -- finished rows are not"
echo "re-measured, so it picks up where it left off."
