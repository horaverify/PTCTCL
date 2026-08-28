# `urgent chan` — discriminating models

An urgent channel forbids the passage of time whenever a **synchronisation on it is enabled**.
(Contrast an urgent *location*, which forbids delay based on the location vector alone.) Before
the fix, `urgent chan` was parsed and silently downgraded to an ordinary channel, so every verdict
below was computed against semantics the model did not declare.

Run each with `--backend seq`.

| model | query | correct | before | after |
|---|---|---|---|---|
| `urgent-basic.xml` | `E<> P.Bad` | **NOT satisfied** | satisfied ✗ | NOT satisfied ✓ |
| `urgent-broadcast.xml` | `E<> P.Bad` | **NOT satisfied** | satisfied ✗ | NOT satisfied ✓ |
| `urgent-intguard.xml` | `E<> P.Bad` | **satisfied** | satisfied | satisfied ✓ |
| `urgent-intguard.xml` | `E<> P.Bad && k==1` | **NOT satisfied** | satisfied ✗ | NOT satisfied ✓ |
| `urgent-live.xml` | `P.A --> P.B` | **satisfied** | NOT satisfied ✗ | satisfied ✓ |
| `urgent-clip.xml` | any | **rejected** | satisfied ✗ | rejected ✓ |

## What each one isolates

**`urgent-basic.xml`** — the plain case. `go` is urgent and both `go!` and `go?` are unguarded, so
the synchronisation is enabled in the initial state and *no* delay is legal there. `P.Bad` needs
`x>=5` from `P.A`, so it is unreachable. Downgraded, time passes at `(A,C)` and `Bad` is reached.

**`urgent-broadcast.xml`** — `urgent broadcast chan`, one process, **no receiver at all**. A
broadcast send fires with zero receivers, so the sender being enabled is itself enough to stop
time. This one also covers the engine *routing*: a one-process model reaches the network engine
only because `app/main.cpp` sends urgent-channel models there unconditionally.

**`urgent-intguard.xml`** — urgency conditioned on the **integer** valuation, which is legal and is
why `frozen_flags` had to start taking the integers as well as the locations. `Q`'s `go?` is
guarded `k==1`, and `R` reaches `R1` either setting `k=1` or leaving it at `0`, under an invariant
`x<=2` that caps the delay before it. The location vector `(A,C,R1)` is therefore urgent for
`k==1` and not urgent for `k==0`: `E<> P.Bad` stays satisfied via the `k==0` branch while
`E<> P.Bad && k==1` must become unreachable. A location-only urgency test cannot separate these.

**`urgent-live.xml`** — the **liveness** engine (`leads_to_otf` / `gfp_succ_tagged` / `gfp_pre`),
not the reachability prefix. Under the downgrade, "sit in `P.A` and let time diverge" is a legal
maximal run that never reaches `P.B`, so `P.A --> P.B` fails. Urgency removes that run: time
cannot pass at `(A,C)`, and the only continuation is the synchronisation itself.

**`urgent-clip.xml`** — the **rejected** case, kept as a diagnostic test. `Q`'s `go?` carries a
clock guard `x>=3`, which would make enabledness a property of the zone and require clipping the
delay at the instant it switches on. That is refused with an error rather than approximated; see
KNOWN_LIMITATION.md §1 for why no sound approximation is applied.

`BASELINE-d4fb785.txt` is the full pre-change regression baseline.
