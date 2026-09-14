#!/bin/sh
# hora with the EXACT recurrence criterion AND depth-first expansion for E[]. hora_exact.sh plus
# the exploration order of hora_dfs.sh; see both for why these go through --hora-bin.
#
# Exists for `single`. Its only location caps every clock and its guards are re-enabled by resets,
# so no record is terminal and the witness is always a CYCLE, which only the partial-graph probe
# can stop the construction for. Under BFS the probe's work-charged budget spaces the probes past
# the end of the graph, so every cell explores the whole aLU graph first: single6 q4 17238
# records / 17 s under BFS against 165 records / 0.01 s under DFS, same verdict.
exec /home/luca/Documents/ParallelCTLModelChecker/PTCTCL/build_shipping/hora --criterion exact --exploration dfs "$@"
