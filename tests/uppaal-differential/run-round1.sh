#!/bin/zsh
# Differential: ptctlmc vs UPPAAL 5.0.0 verifyta, one query per line.
cd /Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC
V=/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta
S=/private/tmp/claude-501/-Users-brdlcu-Documents-Github-PTCTLMC/0206a6ad-f8e7-42c3-b6c1-73a249aed5e9/scratchpad
OUT=$S/uppaal-diff.txt; : > $OUT
TMP=$S/one.q

norm() {  # stdin -> SAT | NOT | UNSUP | ERROR
  local t="$(cat)"
  if   echo "$t" | grep -q -- "-- Formula is satisfied"      ; then echo SAT
  elif echo "$t" | grep -q -- "-- Formula is NOT satisfied"  ; then echo NOT
  elif echo "$t" | grep -qi "UNSUPPORTED"                    ; then echo UNSUP
  elif echo "$t" | grep -qi "EXCEPTION\|error"               ; then echo ERROR
  else echo OTHER; fi
}

run_pair() {  # $1=model absolute  $2=query text
  local m="$1" q="$2"
  print -r -- "$q" > $TMP
  local a b
  a=$(./build-freshmain/ptctlmc --backend seq "$m" "$TMP" 2>&1 | norm)
  b=$("$V" -q "$m" "$TMP" 2>&1 | norm)
  local mark="ok  "
  [[ "$a" != "$b" ]] && mark="DIFF"
  printf "%s | %-6s %-6s | %-28s | %s\n" "$mark" "$a" "$b" "$(basename $m)" "$q" >> $OUT
}

# 1. urgent-chan models
for pair in "urgent-basic.xml:E<> P.Bad" "urgent-broadcast.xml:E<> P.Bad" \
            "urgent-intguard.xml:E<> P.Bad" "urgent-intguard.xml:E<> P.Bad && k==1" \
            "urgent-live.xml:P.A --> P.B" "urgent-clip.xml:E<> P.Bad" \
            "urgent-clip.xml:E<> P.Ok" "urgent-clip.xml:E<> P.Late"; do
  run_pair "$PWD/tests/models/urgent-chan/${pair%%:*}" "${pair#*:}"
done
# 2. clock-liveness models (the new E[]/A<> feature)
for q in "E[] x <= 5" "E[] x >= 5" "E[] x >= 0" "A<> x >= 5" "E[] x >= 5 && x <= 10" \
         "A<> x >= 5 && x <= 10" "E[] P.T && x <= 5" "A<> x >= 0" "E[] x < 5" "A<> x > 5"; do
  run_pair "$PWD/tests/models/clock-liveness/eg-clock.xml" "$q"
done
# 3. liveness regressions
run_pair "$PWD/tests/models/liveness-regressions/cex1.xml" "P.Q --> P.Live"
run_pair "$PWD/tests/models/liveness-regressions/cex2.xml" "P.U --> P.Live"
run_pair "$PWD/tests/models/liveness-regressions/cex3.xml" "P.U --> P.Live"
run_pair "$PWD/tests/models/liveness-regressions/bcast-receiver.xml" "E<> P.T && Q.A"
run_pair "$PWD/tests/models/liveness-regressions/bcast-receiver.xml" "E<> P.T && Q.B"
# 4. benchmarks: A[] not deadlock + each family's reach and live query suites
for m in arbiter4 arbiter5 arbiter6 csma4 csma5 csma6 fischer4 fischer5 fischer6 \
         grc4 grc5 phil4 phil5 phil6 twolock4 twolock5 viking4 single4 single5; do
  [[ -f benchmarks/models/$m.xml ]] || continue
  run_pair "$PWD/benchmarks/models/$m.xml" "A[] not deadlock"
  fam=$(echo $m | sed 's/[0-9]*$//')
  for d in reach_queries live_queries; do
    [[ -f benchmarks/$d/$fam.q ]] || continue
    while IFS= read -r line; do
      [[ -z "${line// }" ]] && continue
      run_pair "$PWD/benchmarks/models/$m.xml" "$line"
    done < benchmarks/$d/$fam.q
  done
done
echo "DONE $(grep -c . $OUT) rows, $(grep -c '^DIFF' $OUT) differences" >> $OUT
