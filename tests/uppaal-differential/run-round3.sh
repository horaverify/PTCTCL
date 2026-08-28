#!/bin/zsh
# Round 3: the non-convex A<> forms that delay_avoiding unlocked, satisfied AND not.
cd /Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC
V=/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta
S=/private/tmp/claude-501/-Users-brdlcu-Documents-Github-PTCTLMC/0206a6ad-f8e7-42c3-b6c1-73a249aed5e9/scratchpad
OUT=$S/uppaal-diff3.txt; : > $OUT; TMP=$S/three.q
norm() { local t="$(cat)"
  if   echo "$t" | grep -q -- "-- Formula is satisfied"     ; then echo SAT
  elif echo "$t" | grep -q -- "-- Formula is NOT satisfied" ; then echo NOT
  elif echo "$t" | grep -qi "UNSUPPORTED"                   ; then echo UNSUP
  elif echo "$t" | grep -qi "EXCEPTION\|error"              ; then echo ERROR
  else echo OTHER; fi; }
run_pair() { local m="$1" q="$2"; print -r -- "$q" > $TMP
  local a=$(./build-freshmain/ptctlmc --backend seq "$m" "$TMP" 2>&1 | norm)
  local b=$("$V" -q "$m" "$TMP" 2>&1 | norm)
  local mark="ok  "; [[ "$a" != "$b" ]] && mark="DIFF"
  printf "%s | %-6s %-6s | %-14s | %s\n" "$a==$b" "$a" "$b" "$(basename $m)" "$q" >> /dev/null
  printf "%s | %-6s %-6s | %-14s | %s\n" "$mark" "$a" "$b" "$(basename $m)" "$q" >> $OUT; }

for m in eg-clock af-sat; do
 for q in "A<> x >= 5 && x <= 10" "A<> x >= 5 && x <= 6" "A<> x > 5 && x < 10" \
          "A<> x >= 0 && x <= 0" "A<> x >= 1 && x <= 2" "A<> x >= 100 && x <= 101" \
          "A<> x >= 2 && x <= 3 && x <= 9" "A<> x > 0 && x <= 1000" \
          "E[] x >= 4 && x <= 8" "E[] x <= 3" "E[] x >= 0 && x <= 100" \
          "A<> x >= 3" "A<> x <= 3" "E[] x >= 1"; do
   run_pair "$PWD/tests/models/clock-liveness/$m.xml" "$q"; done; done
# same forms on real benchmark models, two clocks each
for m in fischer4 fischer5 phil4 grc4 csma4 arbiter4; do
  cl=$(grep -oE "clock [a-zA-Z0-9_, ]+;" benchmarks/models/$m.xml | head -1 | sed 's/clock //;s/;//' | cut -d, -f1 | tr -d ' ')
  for q in "A<> $cl >= 1 && $cl <= 3" "A<> $cl >= 0 && $cl <= 100" "A<> $cl > 2 && $cl < 5" \
           "A<> $cl >= 5 && $cl <= 5" "E[] $cl >= 1 && $cl <= 4"; do
    run_pair "$PWD/benchmarks/models/$m.xml" "$q"; done; done
echo "DONE $(grep -c . $OUT) rows, $(grep -c '^DIFF' $OUT) differences" >> $OUT
