#!/bin/sh
# Regression test for the lazy liveness refinement path (HORA_LAZY).
#
# Two things need pinning down, and neither is covered by the benchmark suite.
#
# THE RECURRENCE CRITERION. A key-SCC owes refinement obligations only when it can host an
# infinite psi-run. It is declared dead when (a) some clock bounded at every key of the component
# is reset by none of its transitions, AND (b) some clock is reset inside it and guarded
# `z >= c`, c > 0, inside it. (b) is not optional: `E[] psi` as implemented here AND by UPPAAL
# 5.0.0 accepts a ZENO witness, so a component where time merely cannot diverge may still carry
# an infinite run. The three `cycle_*` models are the minimal discriminator, and their expected
# verdicts were taken from verifyta 5.0.0 rather than from this engine.
#
# THE REFINEMENT PATH ITSELF. The criterion is effective enough that no row of the `E[]` benchmark
# suite produces a single obligation -- replay, pinning, forbidden covers, inclusion dedup and the
# wave loop would otherwise never execute under test. HORA_LAZY_NOFILTER forces every cyclic
# component live, which puts the whole path back under test against a model whose verdict is
# known independently.
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

# Runs $HORA and echoes "SAT" or "NOT"; anything else is a hard failure of the run itself.
verdict() {
  out=$("$@" 2>/dev/null || true)
  case "$out" in
    *'is NOT satisfied'*) printf 'NOT' ;;
    *'is satisfied'*)     printf 'SAT' ;;
    *)                    printf 'NORESULT' ;;
  esac
}

# ---- 1. the recurrence criterion discriminates ---------------------------------------------
# cycle_zeno         no guards at all, so the cycle can fire with time frozen. `x` is never
#                    reset, so time cannot diverge -- yet the Zeno cycle IS a witness. The
#                    component must stay LIVE, or a real witness loses its obligation.
# cycle_timed        the cycle must burn >= 1 time unit and `x` is still never reset, so the
#                    component can host no infinite run at all. DEAD.
# cycle_timed_reset  the cycle burns time and resets `x`, so it is a genuine time-divergent
#                    witness. LIVE.
for case in "cycle_zeno SAT 0" "cycle_timed NOT 1" "cycle_timed_reset SAT 0"; do
  set -- $case
  model=$1; want_verdict=$2; want_dead=$3
  got=$(verdict "$HORA" "$HERE/$model.xml" "$HERE/probe.q")
  [ "$got" = "$want_verdict" ] || fail "$model: verdict $got, expected $want_verdict (verifyta 5.0.0)"
  got_lazy=$(HORA_LAZY=1 verdict "$HORA" "$HERE/$model.xml" "$HERE/probe.q")
  [ "$got_lazy" = "$want_verdict" ] || fail "$model: HORA_LAZY verdict $got_lazy, expected $want_verdict"
  dead=$(HORA_LAZY=1 HORA_LAZY_DBG=1 "$HORA" "$HERE/$model.xml" "$HERE/probe.q" 2>&1 |
         sed -n 's/.*key-sccs=[0-9]* dead=\([0-9]*\).*/\1/p' | tail -1)
  [ "${dead:-missing}" = "$want_dead" ] ||
    fail "$model: dead components ${dead:-missing}, expected $want_dead"
done

# ---- 2. the refinement path runs, converges, and is right ----------------------------------
# With the criterion suppressed, viking4 q3 offers a small genuine workload: covered edges whose
# exact successors are disjoint from their coverers, inside a component now treated as live.
sed -n '3p' "$QUERIES/viking.q" > "$TMP/v3.q"
base=$(verdict "$HORA" "$MODELS/viking4.xml" "$TMP/v3.q")
[ "$base" = "NOT" ] || fail "viking4 q3: baseline verdict $base, expected NOT"

stats=$(HORA_LAZY=1 HORA_LAZY_NOFILTER=1 HORA_REFINE_STATS=1 \
        "$HORA" "$MODELS/viking4.xml" "$TMP/v3.q" 2>&1 || true)
case "$stats" in
  *'is NOT satisfied'*) : ;;
  *) fail "viking4 q3: forced-refinement verdict differs from baseline NOT" ;;
esac
field() { printf '%s\n' "$stats" | sed -n "s/.*$1=\([0-9]*\).*/\1/p" | tail -1; }
[ "$(field exhausted)" = "0" ] ||
  fail "viking4 q3: refinement exhausted its budget and fell back; it must converge"
[ "$(field materialised)" -gt 0 ] 2>/dev/null ||
  fail "viking4 q3: no cover was materialised, so the refinement path never ran"
[ "$(field obligations)" -gt 0 ] 2>/dev/null ||
  fail "viking4 q3: no obligation was raised, so the test exercises nothing"
note "refinement path: obligations=$(field obligations) materialised=$(field materialised) \
new_nodes=$(field new_nodes) waves=$(field waves)"

# What this test does NOT claim. Forced refinement converges cheaply on viking, and does NOT on
# arbiter or fischer: with the criterion suppressed, arbiter6 q3 walks to roughly 26,700 records
# against a 256-record a_LU graph, flattening onto the containment graph (24,334) rather than
# growing without bound -- inclusion dedup in ::LivenessEngine::lazy_materialise is what supplies
# that ceiling. Proving `NOT` there by refinement genuinely requires making the whole recurrent
# region exact, so the containment graph is the honest floor for that route and no amount of
# tuning beats it. That is the argument FOR the criterion, not a defect of the loop, and it is
# why HORA_LAZY_NOFILTER is a stress switch rather than a supported mode. Do not add arbiter or
# fischer rows below expecting them to converge.


# ---- 3. HORA_LAZY never changes a verdict --------------------------------------------------
# Both the satisfied rows (decided by the early witness, before the loop is reached) and the
# refuted ones (decided by the criterion, with no refinement at all).
for row in "arbiter4 3 NOT" "arbiter4 2 SAT" "fischer4 3 NOT" "fischer4 1 SAT" \
           "viking4 3 NOT" "viking4 5 SAT" "csma4 5 SAT" "phil4 2 SAT"; do
  set -- $row
  model=$1; line=$2; want=$3
  fam=$(printf '%s' "$model" | sed 's/[0-9]*$//')
  sed -n "${line}p" "$QUERIES/$fam.q" > "$TMP/q.q"
  b=$(verdict "$HORA" "$MODELS/$model.xml" "$TMP/q.q")
  l=$(HORA_LAZY=1 verdict "$HORA" "$MODELS/$model.xml" "$TMP/q.q")
  [ "$b" = "$want" ] || fail "$model q$line: baseline $b, expected $want"
  [ "$l" = "$b" ] || fail "$model q$line: HORA_LAZY changed the verdict, $b -> $l"
done

if [ "$fails" -eq 0 ]; then
  note "refinement tests: all passed"
else
  note "refinement tests: $fails FAILED"
  exit 1
fi
