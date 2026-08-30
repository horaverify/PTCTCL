#!/bin/zsh
# Round 2: the NEW feature (clock atoms in E[]/A<>) on REAL benchmark models.
cd /Users/brdlcu/Documents/Github/HORA/HORA
V=/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta
S=/private/tmp/claude-501/-Users-brdlcu-Documents-Github-HORA/0206a6ad-f8e7-42c3-b6c1-73a249aed5e9/scratchpad
OUT=$S/uppaal-diff2.txt; : > $OUT
TMP=$S/two.q
norm() { local t="$(cat)"
  if   echo "$t" | grep -q -- "-- Formula is satisfied"     ; then echo SAT
  elif echo "$t" | grep -q -- "-- Formula is NOT satisfied" ; then echo NOT
  elif echo "$t" | grep -qi "UNSUPPORTED"                   ; then echo UNSUP
  elif echo "$t" | grep -qi "EXCEPTION\|error"              ; then echo ERROR
  else echo OTHER; fi; }
run_pair() { local m="$1" q="$2"; print -r -- "$q" > $TMP
  local a=$(./build-freshmain/hora --backend seq "$m" "$TMP" 2>&1 | norm)
  local b=$("$V" -q "$m" "$TMP" 2>&1 | norm)
  local mark="ok  "; [[ "$a" != "$b" ]] && mark="DIFF"
  printf "%s | %-6s %-6s | %-14s | %s\n" "$mark" "$a" "$b" "$(basename $m)" "$q" >> $OUT; }

# clock names per family, taken from the models themselves
for m in fischer4 fischer5; do for c in x0 x1; do for q in \
  "E[] $c <= 2" "E[] $c >= 2" "E[] $c < 1" "E[] $c >= 0" "E[] $c <= 100" \
  "A<> $c >= 2" "A<> $c <= 2" "A<> $c > 0" "A<> $c >= 0"; do
  run_pair "$PWD/benchmarks/models/$m.xml" "$q"; done; done; done
for m in phil4 grc4 csma4 arbiter4 single4 twolock4 viking4; do
  cl=$(grep -oE "clock [a-zA-Z0-9_, ]+;" benchmarks/models/$m.xml | head -1 | sed 's/clock //;s/;//' | cut -d, -f1 | tr -d ' ')
  [[ -z "$cl" ]] && continue
  for q in "E[] $cl <= 3" "E[] $cl >= 3" "E[] $cl >= 0" "A<> $cl >= 3" "A<> $cl >= 0" "A<> $cl <= 3"; do
    run_pair "$PWD/benchmarks/models/$m.xml" "$q"; done; done
# mixed discrete+clock
run_pair "$PWD/benchmarks/models/fischer4.xml" "E[] P0.wait && x0 <= 2"
run_pair "$PWD/benchmarks/models/fischer4.xml" "E[] P0.req && x0 <= 2"
run_pair "$PWD/benchmarks/models/fischer4.xml" "E[] not P0.cs && x0 >= 0"
run_pair "$PWD/benchmarks/models/phil4.xml"    "E[] P0.Eat && x0 >= 0"
echo "DONE $(grep -c . $OUT) rows, $(grep -c '^DIFF' $OUT) differences" >> $OUT
