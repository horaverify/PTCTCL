#!/usr/bin/env bash
# Validate every curated timed model in tests/models/ against verifyta, for each
# CPU backend. Robust for nohup background runs.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PT="${PT:-$ROOT/build/hora}"
VT="${VT:-/home/luca/Documents/TimedAutomata/uppaal/uppaal-5.0.0-linux64/bin/verifyta}"
MODELS="$ROOT/tests/models"

echo "=== HORA timed-automata validation vs verifyta ($(date)) ==="
fail=0
for be in seq omp; do
  echo "--- backend: $be ---"
  for x in "$MODELS"/*.xml; do
    [ -e "$x" ] || continue
    q="${x%.xml}.q"; [ -e "$q" ] || continue
    name="$(basename "$x" .xml)"
    "$PT" --backend "$be" "$x" "$q" >/tmp/pt.$$ 2>&1
    "$VT" -q "$x" "$q" >/tmp/vt.$$ 2>&1
    ps=$(grep -c 'is satisfied' /tmp/pt.$$); pn=$(grep -c 'NOT satisfied' /tmp/pt.$$)
    vs=$(grep -c 'is satisfied' /tmp/vt.$$); vn=$(grep -c 'NOT satisfied' /tmp/vt.$$)
    if [ "$ps" = "$vs" ] && [ "$pn" = "$vn" ]; then
      echo "  $name: MATCH (sat=$ps unsat=$pn)"
    else
      echo "  $name: DIFF  hora(sat=$ps,unsat=$pn) verifyta(sat=$vs,unsat=$vn)"
      fail=$((fail+1))
    fi
  done
done
rm -f /tmp/pt.$$ /tmp/vt.$$
echo "======================================================"
[ "$fail" = 0 ] && echo "ALL MODELS MATCH verifyta" || echo "$fail mismatch(es)"
exit "$fail"
