#!/bin/zsh
cd /Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC
V=/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta
S=/private/tmp/claude-501/-Users-brdlcu-Documents-Github-PTCTLMC/0206a6ad-f8e7-42c3-b6c1-73a249aed5e9/scratchpad
OUT=$S/uppaal-diff4.txt; : > $OUT; TMP=$S/four.q
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
  printf "%s | %-6s %-6s | %-12s | %s\n" "$mark" "$a" "$b" "$(basename $m .xml)" "$q" >> $OUT; }

for m in eg-clock af-sat; do
 M="$PWD/tests/models/clock-liveness/$m.xml"
 for q in \
   "E[] x < 5 || x > 10" "A<> x < 5 || x > 10" \
   "E[] (x >= 1 && x <= 2) || (x >= 5 && x <= 6)" \
   "A<> (x >= 1 && x <= 2) || (x >= 5 && x <= 6)" \
   "E[] (x < 2) || (x > 3 && x < 7)" \
   "A<> (x < 2) || (x > 3 && x < 7)" \
   "E[] !(x >= 5)" "A<> !(x >= 5)" "E[] !(x >= 5 && x <= 10)" "A<> !(x >= 5 && x <= 10)" \
   "E[] P.T || (x >= 1 && x <= 3)" "A<> P.T || (x >= 1 && x <= 3)" \
   "E[] !P.T && x <= 4" "A<> !P.T && x <= 4" \
   "E[] (P.S && x <= 2) || (P.T && x >= 1)" \
   "A<> (P.S && x <= 2) || (P.T && x >= 1)" \
   "E[] x == 3" "A<> x == 3" "E[] !(x == 3)" "A<> !(x == 3)" \
   "E[] x <= 2 && (x >= 0 || x >= 1)" \
   "A<> (x >= 2 || x <= 1) && x <= 100" ; do
  run_pair "$M" "$q"; done; done
for m in fischer4 phil4 grc4; do
  M="$PWD/benchmarks/models/$m.xml"
  cl=$(grep -oE "clock [a-zA-Z0-9_, ]+;" benchmarks/models/$m.xml | head -1 | sed 's/clock //;s/;//' | cut -d, -f1 | tr -d ' ')
  for q in "E[] $cl <= 2 || $cl >= 8" "A<> $cl <= 2 || $cl >= 8" \
           "E[] ($cl >= 1 && $cl <= 2) || ($cl >= 5 && $cl <= 6)" \
           "A<> ($cl >= 1 && $cl <= 2) || ($cl >= 5 && $cl <= 6)" \
           "E[] !($cl >= 3)" "A<> !($cl >= 3 && $cl <= 4)"; do
    run_pair "$M" "$q"; done; done
echo "DONE $(grep -c . $OUT) rows, $(grep -c '^DIFF' $OUT) differences" >> $OUT
