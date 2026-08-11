# Prototype for the backward-gfp liveness algorithm

This is the artifact behind `nove_mc/main.pdf` and
`docs/superpowers/specs/parallel-search-novelty.md`.

**The shipping engine is untouched.** `netreach_gfp.cpp` is a *copy* of `src/netreach.cpp` with
functions appended; nothing in `src/` was modified. Verify with:

```bash
diff <(head -3440 ../../src/netreach.cpp) <(head -3440 netreach_gfp.cpp)
```

which shows only an added `#include <array>`.

## Build

It compiles every core source **except** `src/netreach.cpp`, which this copy replaces (same external
symbols, so nothing else changes).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
```

`CMakeLists.txt` hard-codes `REPO=/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC`; edit if moved.

## What was added

| symbol | purpose |
|---|---|
| `leads_to_gfp` | Algorithm 1: forward aLU-subsumed reachability, then a backward greatest fixpoint over it, one federation per node |
| `gfp_succ_tagged` | successor generation that also reports the *list* of participating edges (local / binary sync / broadcast) |
| `gfp_pre` | exact concrete predecessor of one tagged transition — one code path for all three transition kinds |
| `retire_causality` | probe answering "is the retiring zone causally independent of the retired one?" |
| `defer_probe` | Algorithm 2 (staleness deferral) on a barrier-free global FIFO — the measured negative |

## Run

```bash
# differential: runs BOTH the shipping leads_to_net and the prototype, compares verdicts
PTCTLMC_GFPALU=1 ./build/gfp_proto ../../benchmarks/models/arbiter5.xml C1.cs C1.idle

# the two probes
./build/gfp_proto ../../benchmarks/models/fischer6.xml --retire-causality x
./build/gfp_proto ../../benchmarks/models/fischer6.xml --defer 4

# the Zeno discriminator (all three tools answer NOT satisfied -> all are Zeno-permissive)
./build/gfp_proto zeno.xml Z.Start Z.Done

# E[] phi -- the SAME greatest fixpoint, verdict taken at the initial state only
PTCTLMC_GFPALU=1 ./build/gfp_proto ../../benchmarks/models/fischer4.xml --eg P1.idle
```

Exit code is 0 when the two engines agree, 1 when they disagree, 3 on a refusal.

| env var | effect |
|---|---|
| `PTCTLMC_GFPALU=1` | use aLU simulation in phase 1 (this is what reaches the reachability floor; plain inclusion gives ~11–45 zones/disc instead of 1.0) |
| `PTCTLMC_GFPFEDCAP=N` | federation-width cap, default 4096. A run that hits it reports `capped=YES(UNUSABLE)` and its verdict must be discarded |
| `PTCTLMC_GFPDBG=1` | dump the violating node, its stop-term status and its out-edges |
| `PTCTLMC_GFPNOEND=1` | ablation: drop the stopping term |
| `PTCTLMC_GFPNOPD=1` | ablation: drop the partial-deadlock term |
| `PTCTLMC_GFPINITONLY=1` | take the verdict at the INITIAL state only (this is what makes `E[]` correct; without it the same fixpoint answers `EF EG phi`, which is strictly weaker) |

## Result

- **`p --> q`**: 164 differential queries over fischer4/5/6, twolock4/5, arbiter4/5, viking4,
  phil4/6, csma4/5, grc4 --- **164 agree, 0 disagree, 0 capped, 0 errors.**
- **`E[] phi`**: 51 queries over the same families against `exists_invariant_run` ---
  **51 agree, 0 disagree.**
