"""Brute-force check of  Pre^avoid(X) = (X n ~(F^))v  u  ((X n (F^ \ F))v n (F^ \ F))
for 2 clocks on an integer grid. Delay adds d to BOTH clocks; v denotes time-predecessor."""
import random
N = 14
PTS = [(x, y) for x in range(N + 1) for y in range(N + 1)]

def sat(p, cons):
    x, y = p
    for var, op, v in cons:
        t = x if var == 'x' else y if var == 'y' else x - y
        if op == '<=' and not t <= v: return False
        if op == '>=' and not t >= v: return False
    return True

def region(cons): return {p for p in PTS if sat(p, cons)}
def up(S):
    out = set()
    for (x, y) in S:
        d = 0
        while x + d <= N and y + d <= N:
            out.add((x + d, y + d)); d += 1
    return out
def down(S):
    out = set()
    for (x, y) in S:
        d = 0
        while x - d >= 0 and y - d >= 0:
            out.add((x - d, y - d)); d += 1
    return out

def truth(X, F):
    """nu such that some delay from nu lands in X without ever entering F."""
    out = set()
    for (x, y) in PTS:
        if (x, y) in F: continue
        d = 0
        while x + d <= N and y + d <= N:
            if (x + d, y + d) in F: break          # crossing F is forbidden
            if (x + d, y + d) in X: out.add((x, y)); break
            d += 1
    return out

def formula(X, F):
    Fup = up(F)
    before = set(PTS) - Fup                        # ~(F^): strictly before F
    after = Fup - F                                # F^ \ F: strictly after F
    return down(X & before) | (down(X & after) & after)

random.seed(11)
bad = tested = 0
for _ in range(4000):
    def rnd(k):
        return [(random.choice(['x','y','x-y']), random.choice(['<=','>=']),
                 random.randint(-4, N)) for _ in range(k)]
    X = region(rnd(random.randint(1, 3)))
    F = region(rnd(random.randint(1, 3)))          # convex: a conjunction
    if not X: continue
    tested += 1
    t, f = truth(X, F), formula(X, F)
    if t != f:
        bad += 1
        if bad <= 3:
            print("MISMATCH  only-truth:", sorted(t - f)[:6], " only-formula:", sorted(f - t)[:6])
print(f"tested {tested} random (X,F) pairs on a {N}x{N} grid: {bad} mismatches")
