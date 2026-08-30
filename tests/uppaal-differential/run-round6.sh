#!/bin/zsh
cd /Users/brdlcu/Documents/Github/HORA/HORA
V=/Applications/UPPAAL-5.0.0.app/Contents/Resources/uppaal/bin/verifyta
S=/private/tmp/claude-501/-Users-brdlcu-Documents-Github-HORA/0206a6ad-f8e7-42c3-b6c1-73a249aed5e9/scratchpad
OUT=$S/uppaal-diff6.txt; : > $OUT; TMP=$S/six.q
norm() { local t="$(cat)"
  if   echo "$t" | grep -q -- "-- Formula is satisfied"     ; then echo SAT
  elif echo "$t" | grep -q -- "-- Formula is NOT satisfied" ; then echo NOT
  elif echo "$t" | grep -qi "UNSUPPORTED"                   ; then echo UNSUP
  else echo ERR; fi; }
run_pair() { local m="$1" q="$2"; print -r -- "$q" > $TMP
  local a=$(./build-freshmain/hora --backend seq "$m" "$TMP" 2>&1 | norm)
  local b=$("$V" -q "$m" "$TMP" 2>&1 | norm)
  local mk="ok  "; [[ "$a" != "$b" ]] && mk="DIFF"
  printf "%s | %-6s %-6s | %-12s | %s\n" "$mk" "$a" "$b" "$(basename $m .xml)" "$q" >> $OUT; }

for spec in "fischer4:P0.req:P0.wait:x0" "fischer5:P0.req:P0.wait:x0" "arbiter4:C0.cs:C0.idle:x0" \
            "csma4:Bus.Transmitting:Bus.Idle:y0" "phil4:P0.Eat:P0.Dropped:c0" \
            "grc4:T0.in:T0.far:x0" "twolock4:P0.reqB:P0.waitB:x0"; do
  m=${spec%%:*}; r=${spec#*:}; pl=${r%%:*}; r2=${r#*:}; ql=${r2%%:*}; cl=${r2##*:}
  M="$PWD/benchmarks/models/$m.xml"
  for q in "$pl --> $ql" "$pl --> $cl >= 0" "$pl --> $cl >= 2" "$pl --> $ql && $cl >= 0" \
           "$pl && $cl >= 0 --> $ql" "$pl || $ql --> $ql" "$pl --> $ql || $cl >= 100" \
           "$pl --> !($cl >= 2 && $cl <= 3)" "$cl >= 0 --> $ql"; do
    run_pair "$M" "$q"; done; done
for m in af-sat eg-clock zeno; do
  M="$PWD/tests/models/clock-liveness/$m.xml"
  for q in "P.S --> P.T" "P.S --> x >= 3" "P.S --> x >= 100" "x >= 0 --> P.T" \
           "P.S --> x <= 2 || x >= 8" "P.S || P.T --> P.T" "P.S --> !(x >= 1 && x <= 2)"; do
    run_pair "$M" "$q"; done; done
echo "DONE $(grep -c . $OUT) rows, $(grep -c '^DIFF' $OUT) differences" >> $OUT
