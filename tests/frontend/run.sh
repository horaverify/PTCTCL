#!/bin/sh
# Declared constants must reach edge guards and location invariants.
#
# `parse_side2` resolves a bound against the clock map, then the integer-variable map, then
# `std::stol`. A declared constant is in neither map and is not a numeral, so the conversion threw
# and a `catch (...)` left the bound at 0 -- silently turning `c <= SA` into `c <= 0` and replacing
# the model with a degenerate one. It cost every hddi verdict: four of five `E[] psi` queries
# disagreed with verifyta 5.0.0, and no diagnostic was ever printed. The scope now reaches the
# guard parser, and an unresolvable bound raises ModelError rather than defaulting.
#
# usage: run.sh <hora-binary> <source-dir>
set -eu
HORA=$1
HERE=$2/tests/frontend
TMP=${TMPDIR:-/tmp}/hora_frontend_$$
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT
fails=0

verdict() {
  printf '%s\n' "$2" > "$TMP/q.q"
  out=$("$HORA" "$1" "$TMP/q.q" 2>&1 || true)
  case "$out" in
    *'is NOT satisfied'*) printf 'NOT' ;;
    *'is satisfied'*)     printf 'SAT' ;;
    *)                    printf 'NORESULT' ;;
  esac
}

# `const int SA = 20;` bounds the invariant, and `const int HALF = SA / 2;` -- a constant
# expression over another constant -- guards the edge. With either read as 0 the clock is pinned
# at 0 and all four expectations invert.
M=$HERE/const_bounds.xml
for case in "E<>_T.c_>=_15 SAT" "E<>_T.c_>=_25 NOT" "E<>_T.B_&&_T.c_<_10 NOT" "E<>_T.B_&&_T.c_>=_10 SAT"; do
  set -- $case
  q=$(printf '%s' "$1" | tr '_' ' ')
  got=$(verdict "$M" "$q")
  [ "$got" = "$2" ] || { printf 'FAIL %s: got %s, expected %s\n' "$q" "$got" "$2"; fails=$((fails + 1)); }
done

if [ "$fails" -eq 0 ]; then
  echo "frontend tests: all passed"
else
  echo "frontend tests: $fails FAILED"
  exit 1
fi
