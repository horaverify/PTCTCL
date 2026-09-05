#!/bin/sh
# hora with depth-first expansion for E[]. Same binary, same flags; only the order the online
# construction pops its worklist changes, which is what lets a CYCLIC witness close after one
# path's worth of records instead of the whole breadth. Exists so bench.py can measure the DFS
# side through --hora-bin; a direct run just passes --exploration dfs.
exec /home/luca/Documents/ParallelCTLModelChecker/PTCTCL/build_shipping/hora --exploration dfs "$@"
