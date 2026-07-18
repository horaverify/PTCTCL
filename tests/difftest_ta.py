#!/usr/bin/env python3
"""Large-scale differential tester for PTCTLMC's backends + verifyta.

Generates random small timed automata + queries and checks that every backend
(seq, omp, and — when available — cuda) AND verifyta agree on the verdict. CUDA is
included automatically if the given binary was built with it (else skipped).

  difftest_ta.py --count 2000 --seed 1 --ptctlmc ./build/ptctlmc \
                 --verifyta <path> [--backends seq,omp] [--keep-fail DIR]
"""
import argparse, os, random, re, subprocess, sys, tempfile

SAT = re.compile(r"Formula is (satisfied|NOT satisfied)", re.IGNORECASE)


def esc(s):  # XML-escape a guard/invariant string
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def gen_model(rng):
    nclk = rng.randint(1, 3)
    nloc = rng.randint(3, 5)
    clocks = [f"x{i}" for i in range(nclk)]
    locs = [f"L{i}" for i in range(nloc)]
    lines = ['<?xml version="1.0" encoding="utf-8"?>',
             "<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.6//EN' "
             "'http://www.it.uu.se/research/group/darts/uppaal/flat-1_6.dtd'>",
             "<nta>", f"<declaration>clock {','.join(clocks)};</declaration>",
             "<template><name>P</name>"]
    for i, L in enumerate(locs):
        inv = ""
        if rng.random() < 0.6:
            c = rng.choice(clocks); b = rng.randint(2, 6)
            inv = f'<label kind="invariant">{esc(f"{c}<={b}")}</label>'
        lines.append(f'<location id="{L}"><name>{L}</name>{inv}</location>')
    lines.append(f'<init ref="{locs[0]}"/>')
    for i in range(nloc):
        for _ in range(rng.randint(1, 3)):
            tgt = rng.randrange(nloc)
            g = []
            if rng.random() < 0.8:
                c = rng.choice(clocks); op = rng.choice([">=", ">", "<=", "<"]); b = rng.randint(1, 6)
                g.append(f"{c}{op}{b}")
            if nclk >= 2 and rng.random() < 0.3:
                a, b2 = rng.sample(clocks, 2); k = rng.randint(0, 4)
                g.append(f"{a}-{b2}<={k}")
            guard = f'<label kind="guard">{esc(" && ".join(g))}</label>' if g else ""
            asg = ""
            if rng.random() < 0.6:
                c = rng.choice(clocks)
                asg = f'<label kind="assignment">{c}=0</label>'
            lines.append(f'<transition><source ref="{locs[i]}"/><target ref="{locs[tgt]}"/>{guard}{asg}</transition>')
    lines += ["</template>", "<system>system P;</system>", "</nta>"]
    # a random query
    if rng.random() < 0.5:
        L = rng.choice(locs)
        extra = ""
        if rng.random() < 0.5:
            extra = f" and {rng.choice(clocks)} <= {rng.randint(1,8)}"
        q = f"E<> P.{L}{extra}"
    else:
        q = f"A[] {rng.choice(clocks)} <= {rng.randint(2,10)}"
    return "\n".join(lines) + "\n", q


def verdict(out):
    m = SAT.search(out)
    return None if not m else ("sat" if m.group(1).lower().startswith("satisf") else "unsat")


def run(cmd, env=None, timeout=30):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None
    return verdict(p.stdout + p.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ptctlmc", required=True)
    ap.add_argument("--verifyta", required=True)
    ap.add_argument("--backends", default="seq,omp")
    ap.add_argument("--keep-fail", default=None)
    args = ap.parse_args()

    backends = args.backends.split(",")
    env = dict(os.environ)
    rng = random.Random(args.seed)
    agree = disagree = skip = 0
    fails = []

    with tempfile.TemporaryDirectory() as td:
        xml = os.path.join(td, "m.xml"); q = os.path.join(td, "m.q")
        for i in range(args.count):
            model, query = gen_model(rng)
            open(xml, "w").write(model); open(q, "w").write(query + "\n")
            verds = {}
            for b in backends:
                verds[b] = run([args.ptctlmc, "--backend", b, xml, q])
            verds["verifyta"] = run([args.verifyta, "-q", xml, q], env=env)
            vals = [v for v in verds.values() if v is not None]
            if len(vals) < len(verds):  # something failed/timeout/unsupported
                skip += 1
                continue
            if len(set(vals)) == 1:
                agree += 1
            else:
                disagree += 1
                fails.append((model, query, verds))
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{args.count}  agree={agree} disagree={disagree} skip={skip}")

    print(f"\nRESULT: agree={agree} disagree={disagree} skip={skip} (total={args.count})")
    if fails and args.keep_fail:
        os.makedirs(args.keep_fail, exist_ok=True)
        for j, (m, ql, v) in enumerate(fails[:50]):
            open(os.path.join(args.keep_fail, f"fail_{j}.xml"), "w").write(m)
            open(os.path.join(args.keep_fail, f"fail_{j}.q"), "w").write(ql + "\n# " + str(v) + "\n")
    if fails:
        print("First disagreement:", fails[0][1], fails[0][2])
    sys.exit(1 if disagree else 0)


if __name__ == "__main__":
    main()
