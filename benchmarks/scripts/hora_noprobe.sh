#!/bin/sh
# hora with the E[] early-exit probe disabled. Same binary, same flags; only the amortized
# witness probe in LivenessEngine::online_drain_forward_work is off, so E[] is decided purely by
# exhausting the psi-restricted graph. Exists so bench.py can measure the probe-off side through
# --hora-bin; a direct run just sets HORA_EG_ONLINE_PROBE=0.
exec env HORA_EG_ONLINE_PROBE=0 /home/luca/Documents/ParallelCTLModelChecker/PTCTCL/build_shipping/hora "$@"
