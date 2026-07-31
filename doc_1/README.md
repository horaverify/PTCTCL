# doc_1 — the OpenMP engine, explained

This folder explains **how PTCTLMC's OpenMP parallelisation actually works**,
what in it is genuinely new, and which of the new parts carry a theorem.
It is not a paper — it is the document a paper gets written *from*.

```sh
make          # -> main.pdf
make watch    # rebuild on save
make clean
```

## Layout

One file per section, one file per figure.

```
main.tex               document skeleton; \input order = reading order
preamble.tex           packages, monochrome TikZ styles, theorem envs, macros
Makefile

sections/
  00-how-to-read.tex   the two audiences; code-to-mechanism trace table
  01-problem.tex       what the tool answers; why this is NOT embarrassingly parallel
  02-background.tex    zones, DBMs, the two finiteness mechanisms; cited facts
  03-antichain.tex     the 5-step insertion procedure that everything must preserve
  04-parallel.tex      THE ARCHITECTURE: levels, stripes, two-phase lock, collect, GC
  05-storage.tex       minimal graphs, interning, why memory is a parallelism story
  06-theory.tex        THE PROOFS
  07-uppaal.tex        positioning vs UPPAAL, LTSmin/opaal, TChecker
  08-evaluation.tex    measured numbers, and what they do and do not show
  09-novelty.tex       blunt new / ported / engineering split + abstract draft
  10-limits.tex        boundaries, negative results, threats to validity
  11-refs.tex          bibliography

figs/
  fig-layers.tex       the three storage layers
  fig-level.tex        anatomy of one BFS level (phases, barrier, tail idle)
  fig-stripes.tex      the striped passed set
  fig-protocol.tex     the two-phase shared/exclusive lock protocol
  fig-add.tex          the five steps of add()
  fig-hull.tex         geometry of the convex-union test
  fig-mingraph.tex     dense DBM vs minimal graph, and the O(|G|) test
  fig-intern.tex       the 64-shard zone interner
  fig-coverage.tex     coverage monotonicity
  fig-scaling.tex      measured 16/24/32-thread scaling
```

**Figures are strictly monochrome** — greys and hatch patterns only, nothing
distinguishable by hue. If you add a figure, do not use `\addplot+` in pgfplots:
the `+` appends the default *coloured* cycle list.

## The short version

The passed set of a competitive timed model checker is not a visited set — it
is an **antichain of zones under subsumption**, and it is **mutable**: inserting
a zone deletes the stored zones it covers and absorbs the ones whose union with
it is convex. That is normally fatal to concurrent reads.

The central result (`06-theory.tex`, Theorem "Coverage monotonicity" + its
corollary) is that this particular update rule only ever retires a zone *into
something that contains it*, so the covered region is monotonically
non-decreasing. Therefore a "this successor is already covered" answer computed
under a **shared** lock cannot go stale, and the 83 % of insertions that end
that way never serialise.

Everything else follows from making the remaining exclusive section small: the
exact convex-union merge holds antichains at ≈1 zone per discrete state, and the
minimal-graph inclusion test decides the common case without decompressing.

## Where each claim comes from

| Claim | Source |
|---|---|
| algorithm & data structures | `src/netreach.cpp` (`reach_engine`, `NetPassed`, `ZoneIntern`, `SpinRW`) |
| zone algebra, aLU, convex union | `HDBM/include/ptctlmc/dbm_ops.hpp` |
| minimal-graph codec + inclusion | `HDBM/include/ptctlmc/mindbm.hpp` |
| timings & state counts | `benchmarks/clean_results/*.csv` |

## Reading order if you are short on time

1. `09-novelty.tex` — what is being claimed.
2. `06-theory.tex` §"The concurrency argument" — why it is allowed.
3. `04-parallel.tex` §"The two-phase lock protocol" — how it is done.
4. `10-limits.tex` — what a reviewer will attack.
