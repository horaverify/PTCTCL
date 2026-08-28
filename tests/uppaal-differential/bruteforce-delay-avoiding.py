"""Brute-force check of  allowed(Z) = (Z^ n ~(C^)) u ((Z n (C^ \ C))^)
for 2 clocks on an integer grid. Delay adds d to BOTH clocks."""
import itertools, random
N = 14                      # grid 0..N
PTS = [(x,y) for x in range(N+1) for y in range(N+1)]

def sat(p, cons):           # cons: list of ('x'|'y'|'x-y', op, val)
    x,y = p
    for var,op,v in cons:
        t = x if var=='x' else y if var=='y' else x-y
        if op=='<=' and not t<=v: return False
        if op=='>=' and not t>=v: return False
    return True

def region(cons): return {p for p in PTS if sat(p,cons)}

def up(S):                  # time successor within grid
    out=set()
    for (x,y) in S:
        d=0
        while x+d<=N and y+d<=N:
            out.add((x+d,y+d)); d+=1
    return out

def truth(Z, C):            # exact: delay from Z\C avoiding C at every intermediate step
    out=set()
    for (x,y) in Z:
        if (x,y) in C: continue
        d=0
        while x+d<=N and y+d<=N:
            if (x+d,y+d) in C: break        # blocked: cannot pass through C
            out.add((x+d,y+d)); d+=1
    return out

def formula(Z, C):
    Cup, Zup = up(C), up(Z)
    term1 = Zup - Cup                       # Z^ n ~(C^)
    term2 = up(Z & (Cup - C))               # already strictly past C
    return term1 | term2

random.seed(7)
bad=0; tested=0
for trial in range(4000):
    def rnd_cons(k):
        cs=[]
        for _ in range(k):
            var=random.choice(['x','y','x-y']); op=random.choice(['<=','>='])
            v=random.randint(-4,N)
            cs.append((var,op,v))
        return cs
    Z = region(rnd_cons(random.randint(1,3)))
    C = region(rnd_cons(random.randint(1,3)))   # convex: conjunction of constraints
    if not Z: continue
    tested+=1
    t, f = truth(Z,C), formula(Z,C)
    if t!=f:
        bad+=1
        if bad<=3:
            print("MISMATCH"); print(" only in truth:", sorted(t-f)[:8]); print(" only in formula:", sorted(f-t)[:8])
print(f"tested {tested} random (Z,C) pairs on a {N}x{N} grid: {bad} mismatches")
