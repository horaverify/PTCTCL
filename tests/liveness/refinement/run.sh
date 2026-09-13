#!/bin/sh
# Regression test for the forward `E[] psi` route (src/liveness/forward_buchi.cpp).
#
# THE RECURRENCE CRITERION. A key-SCC owes refinement obligations only when it can host an
# infinite psi-run. It is dead when some clock bounded at every one of its keys is reset by none
# of its transitions, AND every cycle must let time pass. The second conjunct is not optional:
# `E[] psi` as implemented here AND by UPPAAL 5.0.0 accepts a ZENO witness, so a component where
# time merely cannot diverge may still carry an infinite run. The three `cycle_*` models are the
# minimal discriminator and their expected verdicts come from verifyta 5.0.0, not from this
# engine.
#
# THE CRITERION MODES. `--criterion` selects how components the polynomial tests leave live are
# settled. All three are meant to agree on every verdict; only their cost differs.
#
# usage: run.sh <hora-binary> <source-dir>
set -eu
HORA=$1
SRC=$2
HERE=$SRC/tests/liveness/refinement
MODELS=$SRC/benchmarks/models/xml
QUERIES=$SRC/benchmarks/queries/eg
TMP=${TMPDIR:-/tmp}/hora_refine_$$
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT
fails=0

note() { printf '%s\n' "$*"; }
fail() { printf 'FAIL %s\n' "$*"; fails=$((fails + 1)); }

verdict() {
  out=$("$@" 2>/dev/null || true)
  case "$out" in
    *'is NOT satisfied'*) printf 'NOT' ;;
    *'is satisfied'*)     printf 'SAT' ;;
    *)                    printf 'NORESULT' ;;
  esac
}

# ---- 1. the criterion discriminates, and agrees with verifyta ------------------------------
# cycle_zeno         no guards, so the cycle fires with time frozen. `x` is never reset, so time
#                    cannot diverge -- yet the Zeno cycle IS a witness, and the component must
#                    stay LIVE or a real witness loses its obligation.
# cycle_timed        the cycle must burn >= 1 time unit and `x` is still never reset, so no
#                    infinite run can stay there. DEAD.
# cycle_timed_reset  the cycle burns time and resets `x`: a genuine time-divergent witness. LIVE.
for case in "cycle_zeno SAT 0" "cycle_timed NOT 1" "cycle_timed_reset SAT 0"; do
  set -- $case
  model=$1; want_verdict=$2; want_dead=$3
  for crit in poly guided exact; do
    got=$(verdict "$HORA" --criterion "$crit" "$HERE/$model.xml" "$HERE/probe.q")
    [ "$got" = "$want_verdict" ] ||
      fail "$model --criterion $crit: verdict $got, expected $want_verdict (verifyta 5.0.0)"
  done
  # The criterion is only consulted when no ordinary cycle was found: a witness short-circuits
  # the search, so the two satisfied models print no component census at all.  Check it where it
  # actually runs.
  if [ "$want_verdict" = "NOT" ]; then
    dead=$(HORA_LAZY_DBG=1 "$HORA" "$HERE/$model.xml" "$HERE/probe.q" 2>&1 |
           sed -n 's/.*key-sccs=[0-9]* dead=\([0-9]*\).*/\1/p' | tail -1)
    [ "${dead:-missing}" = "$want_dead" ] ||
      fail "$model: dead components ${dead:-missing}, expected $want_dead"
  fi
done

# ---- 2. every criterion mode reaches the same verdict ---------------------------------------
# Both the satisfied rows, decided by the early witness or by an ordinary cycle, and the refuted
# ones, decided by the criterion with no refinement at all.
for row in "arbiter4 3 NOT" "arbiter4 2 SAT" "arbiter4 5 SAT" "fischer4 3 NOT" "fischer4 1 SAT" \
           "viking4 3 NOT" "viking4 5 SAT" "csma4 5 SAT" "phil4 2 SAT" "single4 1 SAT" \
           "single4 5 SAT" "twolock4 3 NOT" "grc4 1 SAT"; do
  set -- $row
  model=$1; line=$2; want=$3
  fam=$(printf '%s' "$model" | sed 's/[0-9]*$//')
  sed -n "${line}p" "$QUERIES/$fam.q" > "$TMP/q.q"
  for crit in poly guided exact; do
    got=$(verdict "$HORA" --criterion "$crit" "$MODELS/$model.xml" "$TMP/q.q")
    [ "$got" = "$want" ] || fail "$model q$line --criterion $crit: got $got, expected $want"
  done
done

if [ "$fails" -eq 0 ]; then
  note "forward liveness tests: all passed"
else
  note "forward liveness tests: $fails FAILED"
  exit 1
fi
