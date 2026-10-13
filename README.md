# HORA

A model checker for **timed CTL** over **networks of timed automata**, in the UPPAAL style.
HORA answers `E<>`, `A[]`, `E[]` and `p --> q` over a `.xml` model and a `.q` query file, and
reports the same verdicts as UPPAAL's `verifyta`.

```
.xml + .q ─▶ parser ─▶ network IR ─▶ zone-graph search ─▶ verdict
             (own       locations,   (DBM zones: delay, reset, guard,
              XML +      clocks,      invariant, extrapolation, a_LU
              expression int vars,    subsumption; backward greatest
              front end) channels     fixpoint for liveness)
```

Zones are Difference Bound Matrices, provided by [HDBM](HDBM/) — a standalone header-only
library in its own repository, with no dependency on this tool.

## Building

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/hora model.xml query.q
```

The Metal back end is detected automatically and is optional; without it, the
OpenMP and sequential back ends are always available.

```sh
./build/hora --backend omp   model.xml query.q   # parallel CPU (default)
./build/hora --backend metal model.xml query.q   # GPU (E<> and A[] queries; others run on the CPU)
./build/hora --backend seq   model.xml query.q   # sequential reference
./build/hora --zones packed  model.xml query.q   # smaller zone storage, slower probes
./build/hora                                     # full option list
```

## What it decides

| query | meaning |
|---|---|
| `E<> φ` | some reachable state satisfies φ |
| `A[] φ` | every reachable state satisfies φ, including `A[] not deadlock` |
| `E[] φ` | some maximal run satisfies φ throughout |
| `p --> q` | every `p`-state is eventually followed by `q` on every maximal run |

`φ` may combine location predicates, integer-variable constraints and clock constraints, with
conjunction, disjunction and negation. Networks may synchronise over binary and broadcast
channels, and may use urgent and committed locations and urgent channels.

## How liveness works

`p --> q` fails exactly when some reachable state satisfies `p ∧ EG¬q`, and `EG` is a greatest
fixpoint. HORA builds the zone graph **once**, then solves

```
W = W ∧ (NoSucc ∨ Pre(W))
```

backwards over it. The essential observation is that **zone subsumption is fatal forwards and
harmless backwards**: a subsumption edge redirects to a zone larger than the true successor,
which forwards manufactures cycles the system cannot run, but backwards only ever enters as an
upper bound — the direction soundness needs. Liveness therefore runs on the fully subsumed
graph, at the same state count as plain reachability.

Two refinements matter:

- **The `D` refinement.** Plain `a_LU` subsumption does not preserve deadlock, and the stopping
  term `NoSucc` reads exactly that. HORA discovers a deadlock-preserving refinement of `a_LU`
  *while* it explores, so there is one construction rather than a preliminary graph and a rebuild.
- **The early witness.** A reached `p ∧ ¬q` valuation that can end a maximal run is already a
  counterexample, so a violated query needs no finished graph. See
  [the soundness argument](docs/specs/2026-08-30-early-leadsto-witness-soundness.md).

## Correctness

Verdicts are checked differentially against `verifyta` (UPPAAL 5.0.0) where it is available, and
against the previous build otherwise:

```sh
ctest --test-dir build                                  # unit, zone-algebra property, bisimulation
tests/validate_ta.py model.xml query.q                  # one model against verifyta
tests/run_all_ta.sh                                     # the curated suite
benchmarks/scripts/live_diff.py OLD NEW out.csv \
    --suites live,eg,violated,reach,deadlock            # old vs new, aborts on disagreement
```

The zone algebra carries property tests: the primitives that claim to need no canonicalisation
are checked entrywise against the closing versions over randomly generated zones, and the exact
convex-union merge is checked against an independent set-difference oracle.

## Layout

```
include/hora/          public headers
  zone.hpp               HDBM's zone algebra, imported into `hora`
  automaton.hpp          a single timed automaton
  network.hpp            a network of them
  expression.hpp         the model's expression language
  predicate.hpp          lowered query predicates
  model_io.hpp           reading a model file
  query.hpp              a parsed query, and answering one
  reachability.hpp       E<> / A[] / deadlock
  liveness.hpp           E[] / p --> q
  config.hpp             every process-global engine switch
  bisimulation.hpp       timed (bi)simulation between two automata
  symmetry.hpp           process-symmetry reduction
  detail/                private to the engine

src/
  model/                 the model in memory: automaton, network, expressions, integer ranges
  frontend/              reading a model file and a query file
  query/                 lowering a query and selecting an engine
  reachability/          the forward zone-graph search, and deadlock
  liveness/              the online quotient, the D refinement, the backward fixpoint
  analysis/              bisimulation, symmetry
  metal/                 the GPU engine: frontier-batched reachability on Metal

tests/                 unit, property and differential tests, curated models
benchmarks/            models, query suites, measurement drivers, recorded results
docs/                  design.md, limitations.md, specs/
HDBM/                  the zone library (separate repository)
```

## License

MIT for HORA's own code. `verifyta` is used only as an external test oracle; the shipped engine
contains no UDBM (GPL) or UTAP (LGPL) code — the zone algebra is HDBM's own.
