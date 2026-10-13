#!/usr/bin/env python3
"""The Metal backend against the CPU backend, verdict by verdict, with the CPU side cached.

The CPU reference (verdicts, explored/generated counts, wall time) is stored per (model, query)
in a JSON file and reused on later runs, so iterating on the GPU engine only re-runs the GPU.
A CPU run that timed out is cached as a timeout too, and is retried only with a longer timeout.

  gpu_difftest.py --hora build/hora --cases cases.txt [--timeout 120]
                  [--cache benchmarks/results/gpu/cpu_reference.json] [--refresh-cpu]
  gpu_difftest.py --hora build/hora --families fischer,csma --sizes 4-8

cases.txt holds one "<model.xml> <query.q>" per line. Exit status 1 on any verdict disagreement,
crash or CPU fallback of the Metal run.
"""
import argparse, json, os, re, subprocess, sys, time

VERD = re.compile(r'-- Formula is (satisfied|NOT satisfied|UNSUPPORTED)'
                  r'[^\[\n]*(?:\[explored (\d+) symbolic states(?:, generated (\d+))?)?')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(hora, backend, model, query, timeout, env=None):
    t = time.time()
    try:
        p = subprocess.run([hora, '--backend', backend, '--stats', model, query],
                           capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {'timeout': timeout}
    res = {'time': time.time() - t,
           'verdicts': [m.group(1) for m in VERD.finditer(p.stdout)],
           'explored': [m.group(2) for m in VERD.finditer(p.stdout)],
           'generated': [m.group(3) for m in VERD.finditer(p.stdout)]}
    if p.returncode not in (0, 2):
        res['error'] = f'rc={p.returncode} {p.stderr.strip()[:300]}'
    if 'falling back' in p.stderr:
        res['error'] = 'FALLBACK ' + p.stderr.strip()[:300]
    return res


def cases_from(args):
    if args.cases:
        for line in open(args.cases):
            line = line.split('#')[0].strip()
            if line:
                yield tuple(line.split()[:2])
        return
    lo, hi = (int(x) for x in args.sizes.split('-'))
    for fam in args.families.split(','):
        for n in range(lo, hi + 1):
            m = os.path.join(ROOT, 'benchmarks/models/xml', f'{fam}{n}.xml')
            if os.path.exists(m):
                yield m, os.path.join(ROOT, 'benchmarks/queries/reach', f'{fam}.q')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hora', required=True)
    ap.add_argument('--cases')
    ap.add_argument('--families', default='arbiter,csma,fischer,grc,phil,single,twolock')
    ap.add_argument('--sizes', default='4-8')
    ap.add_argument('--timeout', type=float, default=120)
    ap.add_argument('--cache', default=os.path.join(ROOT, 'benchmarks/results/gpu/cpu_reference.json'))
    ap.add_argument('--refresh-cpu', action='store_true', help='re-run the CPU side even if cached')
    args = ap.parse_args()

    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    bad = 0
    for model, query in cases_from(args):
        key = f'{os.path.relpath(model, ROOT)}|{os.path.relpath(query, ROOT)}'
        cpu = cache.get(key)
        stale = cpu is None or args.refresh_cpu or ('timeout' in cpu and cpu['timeout'] < args.timeout)
        if stale:
            cpu = run(args.hora, 'omp', model, query, args.timeout)
            cache[key] = cpu
            os.makedirs(os.path.dirname(args.cache), exist_ok=True)
            json.dump(cache, open(args.cache, 'w'), indent=1, sort_keys=True)
        gpu = run(args.hora, 'metal', model, query, args.timeout)
        name = os.path.basename(model)
        ct = 'TO' if 'timeout' in cpu else f"{cpu['time']:7.2f}s"
        gt = 'TO' if 'timeout' in gpu else f"{gpu['time']:7.2f}s"
        if 'timeout' in cpu or 'timeout' in gpu:
            agree = 'timeout' in gpu or 'timeout' in cpu
            print(f'{name:16s} ----  cpu {ct:>8s}  metal {gt:>8s}  {gpu.get("error", "")}', flush=True)
            if 'error' in gpu:
                bad += 1
            continue
        agree = cpu['verdicts'] == gpu['verdicts'] and 'error' not in gpu
        bad += not agree
        sp = cpu['time'] / max(gpu['time'], 1e-3)
        states = ' '.join(f'{a}/{b}' for a, b in zip(cpu['explored'], gpu['explored']))
        print(f'{name:16s} {"OK  " if agree else "BAD "} {" ".join(v[:3] for v in cpu["verdicts"]):8s} '
              f'cpu {ct}  metal {gt}  x{sp:5.2f}  states cpu/metal {states}  {gpu.get("error", "")}',
              flush=True)
    print('disagreements/errors:', bad)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
