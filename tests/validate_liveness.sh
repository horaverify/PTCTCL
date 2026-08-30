#!/usr/bin/env bash
# Differential liveness check: E[]/A<> location-predicate queries vs verifyta.
set -u
VERIFYTA="${VERIFYTA:-/home/luca/Documents/TimedAutomata/uppaal/uppaal-5.0.0-linux64/bin/verifyta}"
BIN="${BIN:-./build/hora}"
DIR="$(dirname "$0")/models"
fail=0
for m in live cyc; do
  paste -d'|' \
    <("$VERIFYTA" -q "$DIR/$m.xml" "$DIR/$m.q" 2>/dev/null | grep -oE 'is (NOT )?satisfied') \
    <("$BIN" "$DIR/$m.xml" "$DIR/$m.q" 2>/dev/null | grep -oE 'is (NOT )?satisfied') \
  | while IFS='|' read -r a b; do
      [ "$a" = "$b" ] && echo "  ok:  $a" || { echo "  MISMATCH: verifyta=$a hora=$b"; fail=1; }
    done
done
[ $fail -eq 0 ] && echo "LIVENESS MATCHES verifyta" || echo "LIVENESS MISMATCH"
