# TChecker Liveness Observers

Generate observer-instrumented TChecker models from the converted models and
the `live_queries` leads-to properties:

```bash
python3 benchmarks/scripts/gen_tck_live.py
```

The generator supports local properties of the form `Process.p --> Process.q`.
It duplicates the named process into `clear` and `pending` observer states. The
pending variants carry the `TC_LIVE_PENDING` label.

Check a generated model with:

```bash
taskset -c 30 tchecker/install/bin/tck-liveness \
  -a couvscc -l TC_LIVE_PENDING benchmarks/tck_live/phil4.tck
```

`CYCLE true` is a counterexample to the leads-to property; `CYCLE false` means
no accepting counterexample cycle was found.