#!/bin/bash
# XML -> opaal successor generator (.cpp) -> LTSmin PINS module (.so), macOS/arm64.
#
#   ./gen_so_macos.sh <model.xml> <out.so>
#
# This is the macOS counterpart of LTStest/build_so.sh, whose generation half is
# Linux-only (it sources a hardcoded conda path and an x86_64-linux UDBM prefix).
# The compile half is delegated to LTStest/build_so_macos.sh, which already handles
# the Darwin specifics (.so suffix is mandatory; -dynamiclib hard-errors on
# unresolved symbols).
#
# PREREQUISITE: opaal is Python 2 code. Set OPAAL_PY to a Python 2.7 interpreter
# that can import pyuppaal, e.g.
#   OPAAL_PY=$(conda run -n opaal27 which python) ./gen_so_macos.sh m.xml m.so
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LTS="$(cd "$HERE/../../LTStest" && pwd)"
XML="$1"; OUT="$2"; CPP="${OUT%.so}.cpp"
OPAAL_PY="${OPAAL_PY:-/opt/anaconda3/envs/opaal27/bin/python}"

if ! "$OPAAL_PY" -c "" >/dev/null 2>&1; then
  echo "gen_so_macos.sh: no Python 2 interpreter ('$OPAAL_PY')." >&2
  echo "  opaal's successor generator is Python 2 only. Set OPAAL_PY, e.g.:" >&2
  echo "    OPAAL_PY=\$(conda run -n opaal27 which python) $0 $XML $OUT" >&2
  exit 2
fi

export PYTHONPATH="$LTS/src/opaal-ltsmin-succgen:$LTS/src/pyuppaal:$LTS/stubs"
"$OPAAL_PY" -c "
import sys, pyuppaal
from opaal.model_parsers.pyuppaal.gensuccgen import GenerateSuccessorGenerator
nta = pyuppaal.NTA.from_xml(open('$XML'))
s = GenerateSuccessorGenerator(nta)
out = open('$CPP', 'w'); s.generateCFile(out); out.close()
sys.stderr.write('usesDBM=%s\n' % s.usesDBM)
"
"$LTS/build_so_macos.sh" "$CPP" "$OUT"
echo "built $OUT"
