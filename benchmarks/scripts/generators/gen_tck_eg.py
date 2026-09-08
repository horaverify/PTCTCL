#!/usr/bin/env python3
"""Generate TChecker models for UPPAAL `E[] psi` properties.

TChecker has no TCTL: its liveness tool decides label-based Buchi emptiness
(`tck-liveness -a couvscc -l LABEL`). `E[] psi` is answered by handing it the
psi-RESTRICTED model, in which every state satisfies psi by construction, and
labelling every state accepting -- then "some cycle exists" is exactly "some
infinite psi-run exists".

The restriction is built entirely out of the *invariant* of one process (the
first one declared), because a location invariant is the only attribute the
semantics evaluates on the state AFTER a transition -- and, for a synchronised
transition, after every participating edge's statement has run. Guarding the
edges instead would be wrong there: each edge's `provided` sees the PRE-state,
so a two-process sync could step into a psi-violating state unnoticed.

  * psi's clock part goes in verbatim: `invariant:... && x0 <= 40`.
  * psi's discrete part is a boolean formula over `Process.location` atoms,
    which no TChecker expression can name. So each process mentioned by psi
    gets an int variable `__loc_P` mirroring its location (assigned on every
    one of its edges), and the formula is rendered over those variables.
    Disjunction and negation are not in the expression grammar (expressions are
    conjunctions, doc/file-format.md), so they are encoded through the one
    branching construct that is: `(if expr then int_term else int_term)`.

The int variables are a function of the locations, so they add no reachable
states.

WHAT THIS ANSWERS, EXACTLY: "is there a reachable psi-state on a psi-cycle".
That is `E[] psi` for a deadlock-free model under a psi that bounds a clock,
which is every query in eg_queries/ -- see the two gaps below.

  * A maximal run may also END, in a genuine deadlock of the ORIGINAL model
    (UPPAAL counts that run as a witness; a cycle search does not see it). It
    cannot matter where the model is deadlock-free -- benchmarks/reach_queries
    measures `A[] not deadlock` satisfied for arbiter/csma/fischer/phil/twolock
    -- but `grc` HAS deadlocks, so a `grc` row where TChecker says CYCLE false
    and UPPAAL says satisfied is this gap, not a bug.
  * A maximal run may also diverge in time without a cycle (delay forever).
    Every query here upper-bounds a clock, which forbids that.

A query whose CLOCK part is non-convex (`x1 <= 30 || x1 >= 35`) is REFUSED, not
approximated: a TChecker invariant is a conjunction, and the encodings that
route around that (a region observer, or per-process location copies) can only
switch region on a step that all processes take at once, so with disjoint
regions they can never switch at all. Those rows are reported unencodable and
left to HORA and UPPAAL.
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_tck_live import parse_attributes, format_attributes   # noqa: E402

BENCH = HERE.parent.parent
SOURCE_DIR = BENCH / "models" / "tck"
QUERY_DIR = BENCH / "queries" / "eg"
DEFAULT_OUTPUT = BENCH / "models" / "tck_live_e"
LABEL = "TC_EG"
LOC_VAR = "__loc_%s"

LOCATION_RE = re.compile(r"^location:([^:]+):([^\{]+)(\{.*\})$")
EDGE_RE = re.compile(r"^edge:([^:]+):([^:]+):([^:]+):([^\{]+)(\{.*\})$")
CLOCK_RE = re.compile(r"^clock:(\d+):(\w+)$")
DOTTED_RE = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")


def flatten_clocks(text, clocks):
    """`ST1.z` -> `ST1_z` where that is a clock of the model.

    UPPAAL names a clock LOCAL to a template `P.c`; utot flattens it to `P_c`, and the clock
    part of psi is copied into the invariant VERBATIM, so it has to be spelled the way the
    .tck model spells it. Families whose clocks are global (`x0`, `y1`, `c0`) never hit this;
    hddi's `ST1.z` does. A dotted name that is not a clock is a `Process.location` atom and is
    left exactly as it was.
    """
    return DOTTED_RE.sub(
        lambda m: "%s_%s" % (m.group(1), m.group(2))
        if "%s_%s" % (m.group(1), m.group(2)) in clocks else m.group(0), text)


# --------------------------------------------------------------------- query AST
class Atom:
    """A location atom `P.l`, or a clock atom kept as its original text."""

    def __init__(self, kind, text, proc=None, loc=None):
        self.kind, self.text, self.proc, self.loc = kind, text, proc, loc

    def clocks(self):
        return self.kind == "clock"


class Not:
    def __init__(self, f):
        self.f = f

    def clocks(self):
        return self.f.clocks()


class BinOp:
    def __init__(self, op, l, r):
        self.op, self.l, self.r = op, l, r

    def clocks(self):
        return self.l.clocks() or self.r.clocks()


TOKEN_RE = re.compile(r"\s*(&&|\|\||<=|>=|==|!=|[<>!()]|[A-Za-z_][A-Za-z0-9_.\[\]]*|\d+|-)")


def tokenize(text):
    out, i = [], 0
    while i < len(text):
        m = TOKEN_RE.match(text, i)
        if not m:
            if text[i].isspace():
                i += 1
                continue
            raise ValueError("cannot tokenize %r at %d" % (text, i))
        out.append(m.group(1))
        i = m.end()
    return out


class Parser:
    """`||` over `&&` over `!` over atoms; an atom is a clock comparison, a
    `Process.location`, or a parenthesised formula."""

    def __init__(self, tokens, clocks):
        self.t, self.i, self.clocks = tokens, 0, clocks

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self, tok=None):
        got = self.peek()
        if tok is not None and got != tok:
            raise ValueError("expected %r, found %r" % (tok, got))
        self.i += 1
        return got

    def parse(self):
        f = self.disjunction()
        if self.peek() is not None:
            raise ValueError("trailing tokens from %r" % self.peek())
        return f

    def disjunction(self):
        f = self.conjunction()
        while self.peek() == "||":
            self.take()
            f = BinOp("||", f, self.conjunction())
        return f

    def conjunction(self):
        f = self.unary()
        while self.peek() == "&&":
            self.take()
            f = BinOp("&&", f, self.unary())
        return f

    def unary(self):
        if self.peek() == "!":
            self.take()
            return Not(self.unary())
        if self.peek() == "(":
            self.take()
            f = self.disjunction()
            self.take(")")
            return f
        return self.atom()

    def atom(self):
        start = self.i
        term = self.take()
        if term is None:
            raise ValueError("unexpected end of query")
        if self.peek() in ("<=", ">=", "==", "!=", "<", ">", "-"):
            while self.peek() not in (None, "&&", "||", ")"):
                self.take()
            text = flatten_clocks(" ".join(self.t[start:self.i]), self.clocks)
            if not any(c in self.clocks for c in re.findall(r"[A-Za-z_]\w*", text)):
                raise ValueError("comparison over no clock is not supported: %r" % text)
            return Atom("clock", text)
        if "." not in term:
            raise ValueError("not a Process.location atom: %r" % term)
        proc, _, loc = term.partition(".")
        return Atom("loc", term, proc, loc)


# ------------------------------------------------------------------ tck rendering
def _term(f, ids):
    """An int_term that is 1 where `f` holds and 0 where it does not."""
    if isinstance(f, Atom):
        return "(if %s == %d then 1 else 0)" % (LOC_VAR % f.proc, ids[f.proc][f.loc])
    if isinstance(f, Not):
        return "(if %s == 0 then 1 else 0)" % _term(f.f, ids)
    if f.op == "&&":
        return "(if %s == 1 && %s == 1 then 1 else 0)" % (_term(f.l, ids), _term(f.r, ids))
    return "(if %s == 1 then 1 else %s)" % (_term(f.l, ids), _term(f.r, ids))


def render(f, ids):
    """`f` as a TChecker expression. Conjunctions of literals come out readable;
    anything else goes through the int_term encoding."""
    if isinstance(f, Atom):
        return "%s == %d" % (LOC_VAR % f.proc, ids[f.proc][f.loc])
    if isinstance(f, Not) and isinstance(f.f, Atom):
        return "%s != %d" % (LOC_VAR % f.f.proc, ids[f.f.proc][f.f.loc])
    if isinstance(f, BinOp) and f.op == "&&":
        return "%s && %s" % (render(f.l, ids), render(f.r, ids))
    return "%s == 1" % _term(f, ids)


def procs_of(f, out):
    if isinstance(f, Atom):
        if f.kind == "loc":
            out.add(f.proc)
    elif isinstance(f, Not):
        procs_of(f.f, out)
    else:
        procs_of(f.l, out)
        procs_of(f.r, out)
    return out


def split(f):
    """psi -> (clock conjuncts, discrete conjuncts). Raises if a conjunct mixes
    the two, or if a clock constraint sits under `||` / `!` (non-convex)."""
    conjuncts, stack = [], [f]
    while stack:
        g = stack.pop()
        if isinstance(g, BinOp) and g.op == "&&":
            stack += [g.r, g.l]
        else:
            conjuncts.append(g)
    clock, disc = [], []
    for g in conjuncts:
        if isinstance(g, Atom) and g.kind == "clock":
            clock.append(g.text)
        elif g.clocks():
            raise ValueError("clock constraint under '||' or '!' is not a TChecker "
                             "invariant (expressions are conjunctions)")
        else:
            disc.append(g)
    return clock, disc


# ---------------------------------------------------------------------- the model
def scan(source):
    """(process order, {process: {location: id}}, {process: initial location},
    clock names, last int/clock declaration line index)."""
    order, locs, init, clocks, decl = [], {}, {}, set(), 0
    for i, line in enumerate(source.splitlines()):
        if line.startswith("process:"):
            p = line.split(":", 1)[1].strip()
            order.append(p)
            locs.setdefault(p, {})
            continue
        m = CLOCK_RE.match(line)
        if m:
            clocks.add(m.group(2))
            decl = i
            continue
        if line.startswith("int:"):
            decl = i
            continue
        m = LOCATION_RE.match(line)
        if m:
            p, name = m.group(1), m.group(2)
            locs.setdefault(p, {})[name] = len(locs.setdefault(p, {}))
            if any(k == "initial" for k, _ in parse_attributes(m.group(3))):
                init[p] = name
    return order, locs, init, clocks, decl


def instrument(source, clock_conj, disc_conj, locs, init, tracked, host):
    """The psi-restricted model: `__loc_P` for each tracked process, the psi
    invariant and the accepting label on every location of `host`."""
    ids = {p: locs[p] for p in locs}
    extra = [" && ".join(clock_conj)] if clock_conj else []
    extra += [render(g, ids) for g in disc_conj]
    psi = " && ".join(extra)

    out, lines = [], source.splitlines()
    _, decl = None, scan(source)[4]
    for i, line in enumerate(lines):
        out.append(line)
        if i == decl:
            for p in sorted(tracked):
                out.append("int:1:0:%d:%d:%s"
                           % (max(len(locs[p]) - 1, 1), locs[p][init[p]], LOC_VAR % p))
            continue
        m = LOCATION_RE.match(line)
        if m and m.group(1) == host:
            attrs = parse_attributes(m.group(3))
            done_inv = False
            for j, (k, v) in enumerate(attrs):
                if k == "invariant":
                    attrs[j] = (k, "%s && %s" % (v.strip(), psi) if psi else v)
                    done_inv = True
                elif k == "labels":
                    attrs[j] = (k, "%s,%s" % (v.strip(), LABEL) if v.strip() else LABEL)
            if not done_inv and psi:
                attrs.append(("invariant", psi))
            if not any(k == "labels" for k, _ in attrs):
                attrs.append(("labels", LABEL))
            out[-1] = "location:%s:%s%s" % (m.group(1), m.group(2), format_attributes(attrs))
            continue
        m = EDGE_RE.match(line)
        if m and m.group(1) in tracked:
            p, tgt, attrs = m.group(1), m.group(3), parse_attributes(m.group(5))
            assign = "%s=%d" % (LOC_VAR % p, ids[p][tgt])
            for j, (k, v) in enumerate(attrs):
                if k == "do":
                    attrs[j] = (k, "%s;%s" % (v.strip(), assign))
                    break
            else:
                attrs.append(("do", assign))
            out[-1] = "edge:%s:%s:%s:%s%s" % (m.group(1), m.group(2), m.group(3),
                                              m.group(4), format_attributes(attrs))
    return "\n".join(out) + "\n"


def family_of(stem):
    m = re.match(r"([A-Za-z_]+)(\d+)$", stem)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=SOURCE_DIR)
    ap.add_argument("--queries", type=Path, default=QUERY_DIR)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    a = ap.parse_args()

    a.outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"label": LABEL, "models": {}, "refused": {}}
    written = 0
    for model in sorted(a.source.glob("*.tck"), key=lambda p: family_of(p.stem)[::-1]):
        fam, size = family_of(model.stem)
        qfile = a.queries / ("%s.q" % fam) if fam else None
        if qfile is None or not qfile.exists():
            continue
        source = model.read_text()
        order, locs, init, clocks, _ = scan(source)
        queries = [q.strip() for q in qfile.read_text().splitlines() if q.strip()]
        for idx, q in enumerate(queries):
            key = "%s__q%d" % (model.stem, idx)
            body = q[3:].strip() if q.startswith("E[]") else None
            try:
                if body is None:
                    raise ValueError("not an E[] query: %r" % q)
                f = Parser(tokenize(body), clocks).parse()
                clock_conj, disc_conj = split(f)
                tracked = procs_of(f, set())
                for p in sorted(tracked):
                    if p not in locs:
                        raise ValueError("no process %r in the model" % p)
                for g in disc_conj:
                    for atom in re.findall(r"([A-Za-z_]\w*)\.([A-Za-z_]\w*)", str(q)):
                        if "%s_%s" % atom in clocks:
                            continue          # a local clock `P.c`, not a location atom
                        if atom[0] in locs and atom[1] not in locs[atom[0]]:
                            raise ValueError("no location %s.%s in the model" % atom)
            except ValueError as e:
                manifest["refused"].setdefault("%s q%d" % (fam, idx),
                                               {"query": q, "reason": str(e)})
                continue
            text = instrument(source, clock_conj, disc_conj, locs, init, tracked, order[0])
            (a.outdir / (key + ".tck")).write_text(text)
            manifest["models"][key] = {
                "family": fam, "size": size, "query_index": idx, "query": q,
                "accepting_label": LABEL,
                "command": "tck-liveness -a couvscc -l %s %s" % (LABEL, key + ".tck"),
            }
            written += 1
    (a.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote %d psi-restricted models to %s" % (written, a.outdir))
    for k, v in sorted(manifest["refused"].items()):
        print("  REFUSED %-12s %-46s %s" % (k, v["query"], v["reason"]))


if __name__ == "__main__":
    main()
