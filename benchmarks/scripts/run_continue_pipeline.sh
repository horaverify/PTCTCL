#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$BENCH_DIR/results/reach"

# Never trust an inherited TMPDIR: under nohup it can point at a since-removed
# session tmp dir, and mktemp failing there kills the whole script under set -e.
export TMPDIR="$BENCH_DIR/.tmp"
mkdir -p "$TMPDIR"

HORA_DIR="$RESULTS_DIR/hora_gsim"
UPPAAL_DIR="$RESULTS_DIR/uppaal"
LTSMIN_DIR="$RESULTS_DIR/ltsmin"
TCHECKER_DIR="$RESULTS_DIR/tchecker"

VERIFYTA_BASE="$UPPAAL_DIR/verifyta_continue"
LTS_BASE="$LTSMIN_DIR/lts_thread_continue"
VERIFYTA_CONTINUE_CSV="$UPPAAL_DIR/verifyta_continue.csv"
LTS_CONTINUE_CSV="$LTSMIN_DIR/lts_thread_continue.csv"

mkdir -p "$UPPAAL_DIR" "$LTSMIN_DIR" "$TCHECKER_DIR"

families_in() {
  # Print distinct families in one or more CSVs (ignoring repeated headers).
  awk -F, 'NR==1 {next} $1 != "" && $1 != "family" {print $1}' "$@" | sort -u
}

missing_family_spec() {
  # Convert set difference into bench.py family spec: fam1;fam2;...
  # args: <all_set_file> <done_set_file>
  comm -23 "$1" "$2" | paste -sd ';' -
}

run_verifyta_continue() {
  local all_file done_file missing
  all_file="$(mktemp)"
  done_file="$(mktemp)"
  trap 'rm -f "$all_file" "$done_file"' RETURN

  families_in "$HORA_DIR"/*.csv > "$all_file"
  if [[ -f "$UPPAAL_DIR/verifyta.csv" ]]; then
    families_in "$UPPAAL_DIR/verifyta.csv" > "$done_file"
  else
    : > "$done_file"
  fi

  missing="$(missing_family_spec "$all_file" "$done_file")"
  if [[ -z "$missing" ]]; then
    echo "[verifyta] no missing families detected."
    return 0
  fi

  echo "[verifyta] missing families: $missing"
  python3 "$SCRIPT_DIR/bench.py" run \
    --tool verifyta \
    --families "$missing" \
    --outdir "$UPPAAL_DIR" \
    --tag "$(basename "$VERIFYTA_BASE")" \
    --seed "$UPPAAL_DIR/verifyta.csv"

  # bench.py writes per-thread names; verifyta uses thread 1.
  if [[ -f "${VERIFYTA_BASE}_t1_c8.csv" ]]; then
    cp -f "${VERIFYTA_BASE}_t1_c8.csv" "$VERIFYTA_CONTINUE_CSV"
  elif [[ -f "${VERIFYTA_BASE}_t1.csv" ]]; then
    cp -f "${VERIFYTA_BASE}_t1.csv" "$VERIFYTA_CONTINUE_CSV"
  fi
}

run_ltsmin_continue() {
  local all_file done_file missing
  all_file="$(mktemp)"
  done_file="$(mktemp)"
  trap 'rm -f "$all_file" "$done_file"' RETURN

  families_in "$HORA_DIR"/*.csv > "$all_file"
  if ls "$LTSMIN_DIR"/*.csv >/dev/null 2>&1; then
    families_in "$LTSMIN_DIR"/*.csv > "$done_file"
  else
    : > "$done_file"
  fi

  missing="$(missing_family_spec "$all_file" "$done_file")"
  if [[ -z "$missing" ]]; then
    echo "[ltsmin] no missing families detected."
    return 0
  fi

  echo "[ltsmin] missing families: $missing"
  echo "[ltsmin] running threads 16 -> 24 -> 32 (sequential)"

  python3 "$SCRIPT_DIR/bench.py" run \
    --tool ltsmin \
    --threads 16 \
    --families "$missing" \
    --outdir "$LTSMIN_DIR" \
    --tag "$(basename "$LTS_BASE")" \
    --seed "$LTSMIN_DIR/ltsmin_t16_2h.csv" "$LTSMIN_DIR/ltsmin_t24_2h.csv"

  python3 "$SCRIPT_DIR/bench.py" run \
    --tool ltsmin \
    --threads 24 \
    --families "$missing" \
    --outdir "$LTSMIN_DIR" \
    --tag "$(basename "$LTS_BASE")" \
    --seed "$LTSMIN_DIR/ltsmin_t16_2h.csv" "$LTSMIN_DIR/ltsmin_t24_2h.csv"

  python3 "$SCRIPT_DIR/bench.py" run \
    --tool ltsmin \
    --threads 32 \
    --families "$missing" \
    --outdir "$LTSMIN_DIR" \
    --tag "$(basename "$LTS_BASE")" \
    --seed "$LTSMIN_DIR/ltsmin_t16_2h.csv" "$LTSMIN_DIR/ltsmin_t24_2h.csv"

  # Merge thread CSVs into the requested single continuation CSV.
  : > "$LTS_CONTINUE_CSV"
  first=1
  for f in "${LTS_BASE}_t16_c8.csv" "${LTS_BASE}_t24_c8.csv" "${LTS_BASE}_t32_c8.csv" \
           "${LTS_BASE}_t16.csv" "${LTS_BASE}_t24.csv" "${LTS_BASE}_t32.csv"; do
    [[ -f "$f" ]] || continue
    if [[ $first -eq 1 ]]; then
      cat "$f" > "$LTS_CONTINUE_CSV"
      first=0
    else
      tail -n +2 "$f" >> "$LTS_CONTINUE_CSV"
    fi
  done
}

run_tchecker_all() {
  echo "[tchecker] running all families"
  python3 "$SCRIPT_DIR/bench.py" run \
    --tool tchecker \
    --families all \
    --outdir "$TCHECKER_DIR" \
    --tag tchecker
}

echo "=== Continuation pipeline started: $(date -Iseconds) ==="
run_tchecker_all
run_ltsmin_continue
echo "=== Continuation pipeline finished: $(date -Iseconds) ==="
echo "=== Continuation pipeline finished: $(date -Iseconds) ==="
