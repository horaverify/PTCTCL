# The `!=` soundness bug in opaal's successor generator

**Affects:** `opaal` (the LTSmin front-end), not PTCTLMC.
**Symptom:** silently dropped clock constraints, hence an over-approximated state space, on any
model whose guard mixes a clock constraint with an integer `!=` — in our benchmark set that is
**Fischer** and **two-lock**.
**File:** `src/opaal-ltsmin-succgen/opaal/model_parsers/pyuppaal/util.py`,
function `expression_to_fedupdate_c`.

This is a correctness fix to the *baseline*, made so the comparison in the paper is fair. It
does not change PTCTLMC.

---

## What the function does

`expression_to_fedupdate_c` translates a UPPAAL guard into the C that applies it to a DBM
federation. A guard is a syntax tree, and only its **clock** conjuncts belong in the
federation update; **integer** conjuncts are discrete guards enforced elsewhere, by
`guard_code` / `expression_to_c`. The function signals "this subtree is not a clock
constraint, skip it" by raising `NotClockExpressionException`.

## The bug

Two separate mistakes combined.

**1. `NotEqual` was not handled at all.** `a != b` is never a convex constraint, so it cannot
be a single DBM entry, and the translator had no case for it.

**2. `NotClockExpressionException` was caught at the top of the guard, not per conjunct.**
The `And` case translated both children in one expression, so a single exception aborted the
whole guard and the top-level handler returned `""`.

Together these are unsound. Take a guard from Fischer:

```
x0 > 2 && id != 1
```
- `x0 > 2` is a clock constraint and **must** be applied to the federation.
- `id != 1` is an integer guard, correctly enforced separately.

The old code translated the `And`, hit `id != 1`, raised `NotClockExpressionException`, and let
it propagate past the whole guard. The handler returned the empty string — so **`x0 > 2` was
silently dropped as well**. The successor was computed from an unconstrained zone.

Dropping a clock guard *weakens* the constraint, so the generated state space is an
over-approximation: no reachable state is lost (the safety verdict stays sound), but extra
unreachable states are explored. The failure is silent — no warning, no error, just a larger
state count and a slower run.

## The fix

Two changes, both in `expression_to_fedupdate_c`.

**Catch per conjunct.** Each child of an `And` is translated independently, so a
non-clock conjunct contributes nothing while its clock siblings still emit their constraints:

```python
if node.type == 'And':
    result = ""
    for ch in node.children:
        try:
            result += expression_to_fedupdate_c(ch, succgen, sname)
        except NotClockExpressionException:
            pass          # integer conjunct: handled by guard_code, not here
    return result
```

**Handle `NotEqual` explicitly, and distinguish the two cases.** On integers it is a discrete
guard the federation should ignore; on a *clock* it is genuinely non-convex (`x < c ∨ x > c`)
and cannot be represented by one DBM at all, so it must be a hard error rather than a silent
approximation:

```python
elif node.type == 'NotEqual':
    def _is_clock(n):
        return bool(succgen) and n.type == 'Identifier' and n.leaf in succgen.clock_indexes
    if _is_clock(node.children[0]) or _is_clock(node.children[1]):
        raise IllegalExpressionException('clock != is non-convex and unsupported: ' + node.type)
    raise NotClockExpressionException
```

The asymmetry is the point. `IllegalExpressionException` refuses to generate code for something
the representation cannot express; `NotClockExpressionException` says "not my job, someone else
enforces this". Conflating them is what produced the bug.

## Why it matters for the evaluation

Fischer and two-lock both use `id != 0`-style guards alongside clock constraints, so both were
affected. Without the fix, opaal+LTSmin explores an over-approximated state space on exactly
the family where we report our largest advantage, which would make the comparison meaningless:
part of its slowdown would be work that a correct translation never has to do.

With the fix, LTSmin's remaining state-count inflation is attributable to its *design* — it
stores one state per (configuration, zone) pair with no convex-union merging — rather than to
a translation defect. That is the difference the paper claims, so it must not be contaminated
by a fixable bug in the baseline.

## Reproducing

```bash
cd LTStest
./build_so_macos.sh <generated.cpp> <out.so>     # or benchmarks/gen_so_macos.sh from XML
```
Regenerate any `.so` built before this fix: the bug is in code generation, so a stale module
still contains the dropped constraint. Compare explored-state counts before and after — the
fixed module reports fewer states on Fischer and two-lock, and unchanged counts elsewhere.
