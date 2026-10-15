#!/usr/bin/env python3
"""cuda_bench.py -- the reach suite on the CUDA backend, next to the CPU backends on the same host.

For every (family, size) of the reach suite, run each requested tool once per repetition and
record wall time, explored states and the verdict in one CSV, rows keyed like bench.py's:

  cuda_bench.py --hora build/hora --tools cuda,omp16,omp1 --out benchmarks/results/cuda/reach.csv \
                --families arbiter,csma,fischer,grc,phil,single,twolock,viking,hddi --timeout 600

Tools: cuda (--backend cuda), omp<N> (--backend omp, N threads), seq (--backend seq).
A timeout is recorded as TIMEOUT with the timeout as its time; a verdict that disagrees with the
first tool's verdict for the same model is recorded as MISMATCH. Resumable: finished rows are kept.
"""
import argparse, csv, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VERD = re.compile(r'-- Formula is (satisfied|NOT satisfied|UNSUPPORTED)'
                  r'[^\[\n]*(?:\[explored (\d+) symbolic states(?:, generated (\d+))?)?')
FIELDS = ['family', 'model', 'tool', 'rep', 'states', 'generated', 'seconds', 'status', 'verdict', 'gpu_mb', 'query']


def size_of(m):
    return int(re.search(r'(\d+)$', m).group(1))


def run(hora, tool, model, query, timeout, queryset):
    env = dict(os.environ)
    args = [hora, '--stats', model, query]
    if tool == 'cuda':
        args[1:1] = ['--backend', 'cuda']
        env['OMP_NUM_THREADS'] = env.get('HORA_CUDA_OMP', '16')
    elif tool == 'seq':
        args[1:1] = ['--backend', 'seq']
        env['OMP_NUM_THREADS'] = '1'
    elif tool.startswith('omp'):
        args[1:1] = ['--backend', 'omp']
        env['OMP_NUM_THREADS'] = tool[3:] or '16'
    t = time.time()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    peak = 0                       # peak GPU memory of this process, MiB, polled from nvidia-smi
    while True:
        try:
            out, err = proc.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            if time.time() - t > timeout:
                proc.kill(); proc.communicate()
                return dict(seconds=timeout, status='TIMEOUT', states='-', generated='-', verdict='-',
                            gpu_mb=peak or '-')
            q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            for ln in q.stdout.splitlines():
                f = [x.strip() for x in ln.split(',')]
                if len(f) == 2 and f[0] == str(proc.pid) and f[1].isdigit():
                    peak = max(peak, int(f[1]))
    dt = time.time() - t
    p = subprocess.CompletedProcess(args, proc.returncode, out, err)
    gm = peak or '-'
    ms = list(VERD.finditer(p.stdout))
    if p.returncode not in (0, 2) or not ms or 'falling back' in p.stderr:
        return dict(seconds=dt, status='ERR', states='-', generated='-', verdict='-', gpu_mb=gm,
                    err=(p.stderr.strip()[-200:]))
    m = ms[0]
    return dict(seconds=dt, status='OK', states=m.group(2) or '-', generated=m.group(3) or '-', gpu_mb=gm,
                verdict={'satisfied': 'SAT', 'NOT satisfied': 'UNSAT'}.get(m.group(1), m.group(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hora', required=True)
    ap.add_argument('--tools', default='cuda,omp16')
    ap.add_argument('--families', default='arbiter,csma,fischer,grc,phil,single,twolock,viking,hddi')
    ap.add_argument('--queries', default='reach', help='directory under benchmarks/queries')
    ap.add_argument('--timeout', type=float, default=600)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--min-size', type=int, default=4)
    ap.add_argument('--max-size', type=int, default=99)
    ap.add_argument('--out', required=True)
    ap.add_argument('--strike', type=int, default=1,
                    help='skip a tool\'s larger sizes of a family after this many timeouts')
    a = ap.parse_args()
    tools = a.tools.split(',')
    done = {}
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done[(r['model'], r['tool'], r['rep'])] = r
    new = not os.path.exists(a.out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out = open(a.out, 'a', newline='')
    w = csv.DictWriter(out, FIELDS)
    if new:
        w.writeheader()
    xml = os.path.join(ROOT, 'benchmarks/models/xml')
    models = []
    for f in a.families.split(','):
        for m in os.listdir(xml):
            mm = re.fullmatch(rf'({f})(\d+)\.xml', m)
            if mm and a.min_size <= int(mm.group(2)) <= a.max_size:
                models.append((f, m[:-4], int(mm.group(2))))
    models.sort(key=lambda x: (x[2], x[0]))     # size-major, as bench.py orders its sweeps
    strikes = {}
    verdicts = {}
    for fam, model, size in models:
        q = os.path.join(ROOT, 'benchmarks/queries', a.queries, f'{fam}.q')
        if not os.path.exists(q):
            continue
        qtext = open(q).read().strip().splitlines()[0]
        for tool in tools:
            if strikes.get((fam, tool), 0) >= a.strike:
                continue
            for rep in range(1, a.reps + 1):
                key = (model, tool, str(rep))
                if key in done:
                    r = done[key]
                else:
                    r = run(a.hora, tool, os.path.join(xml, model + '.xml'), q, a.timeout, a.queries)
                    row = dict(family=fam, model=model, tool=tool, rep=rep, query=qtext,
                               **{k: r[k] for k in ('states', 'generated', 'seconds', 'status', 'verdict', 'gpu_mb')})
                    if r['status'] == 'OK':
                        v = verdicts.setdefault(model, r['verdict'])
                        if v != r['verdict']:
                            row['status'] = 'MISMATCH'
                    row['seconds'] = f"{float(r['seconds']):.3f}"
                    w.writerow(row)
                    out.flush()
                    print(f"{model:12s} {tool:7s} rep{rep} {row['status']:8s} {row['seconds']:>9s}s "
                          f"states {row['states']} {row['verdict']} gpu {row['gpu_mb']}MiB {r.get('err', '')}", flush=True)
                    r = row
                if r['status'] == 'OK':
                    verdicts.setdefault(model, r['verdict'])
                if r['status'] == 'TIMEOUT':
                    strikes[(fam, tool)] = strikes.get((fam, tool), 0) + 1
                    break
                if r['status'] == 'ERR':
                    break


if __name__ == '__main__':
    main()
