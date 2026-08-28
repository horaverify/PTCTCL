# Clock constraints in `E[]` / `A<>` predicates

`eg-clock.xml`: `P: S --a!--> T`, `T` self-loops; `Q: A --a?--> A`. Clock `x` is never reset and
no location carries an invariant, so `x` grows without bound from `0`.

| query | verdict | why |
|---|---|---|
| `E[] x <= 5` | satisfied | sync to `T` at `x=0`, then loop on `T` forever without delaying |
| `E[] x >= 5` | **NOT satisfied** | **the trap** — see below |
| `E[] x >= 0` | satisfied | vacuous |
| `A<> x >= 5` | NOT satisfied | `= !E[] x<5`, and the instant loop on `T` stays at `x=0` |
| `E[] x >= 5 && x <= 10` | NOT satisfied | convex conjunction, accepted; still fails at `x=0` |
| `A<> x >= 5 && x <= 10` | **UNSUPPORTED** | the negation is non-convex; refused, not approximated |
| `E[] P.T && x <= 5` | NOT satisfied | `P.T` is false in the initial state |

## The trap: `E[] x >= 5`

This is the query that decides whether the restriction is applied in the right place. The
predicate is adjoined to every location invariant, so it is imposed **before** the delay as well
as after; the initial zone `{x=0}` is emptied and the search explores **0** states.

Restricting only *after* the delay — the obvious implementation, since `close_succ_zone` has
already delayed by the time it returns — computes `{x=0}^ ∩ (x>=5) = [5,∞)` and reports a
witness sitting at `x=7`. There is no such run: every execution leaves `x=0` through valuations
violating `x>=5`, so no `x>=5`-execution exists at all.

The intersection is exact where it is because a conjunction of clock constraints is convex, and
the set reachable by delay while *staying* in a convex `C` is `(Z ∩ C)^ ∩ C` — every
intermediate point of the segment is a convex combination of its two endpoints, both in `C`.
Convexity is also why `A<>`/`-->` admit at most one clock atom in the negated position: the
complement of two or more is a disjunction, and a delay may leave it and re-enter.

See `UppalOMP/sec/subsec/clock_predicates.tex` for the statement and proof.
