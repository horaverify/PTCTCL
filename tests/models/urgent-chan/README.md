# `urgent chan` discriminating models

Expected verdicts under **correct** urgent-channel semantics (an urgent channel forbids the
passage of time whenever a synchronisation on it is enabled).

| model | query | correct | before the fix |
|---|---|---|---|
| `urgent-basic.xml` | `E<> P.Bad` | **NOT satisfied** | satisfied (wrong) |
| `urgent-clip.xml`  | `E<> P.Bad` | **NOT satisfied** | satisfied (wrong) |
| `urgent-clip.xml`  | `E<> P.Ok`  | **satisfied**     | satisfied |
| `urgent-clip.xml`  | `E<> P.Late`| **NOT satisfied** | satisfied (wrong) |

`urgent-basic.xml` — `go` is urgent and both `go!`/`go?` are unguarded, so the sync is enabled
in the initial state and **no** delay is legal there. `P.Bad` needs `x>=5` from `P.A`, so it is
unreachable. Downgraded to an ordinary channel, time passes at `(A,C)` and `Bad` is reached.

`urgent-clip.xml` — `Q`'s `go?` is guarded `x>=3`, so the urgent sync is *not* enabled until
`x=3`. Delay from the initial state is therefore legal on `x<3` and stops **at** `x=3`: the
instant enabledness switches on is reachable, anything strictly later is not. That pins the clip
boundary from both sides — `P.Ok` (`x>=3`) must stay reachable while `P.Late` (`x>3`) must not.

`BASELINE-d4fb785.txt` is the full pre-change regression baseline.
