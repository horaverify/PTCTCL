#!/usr/bin/env python3
"""cuda_speedup.py -- how many times faster the CUDA backend is than hora-omp and UPPAAL, as a table.

Reads a cuda_bench.py CSV (cuda and omp<N> rows from the same host) and the recorded verifyta
CSV for the same suite, and prints one markdown row per model:

  cuda_speedup.py benchmarks/results/cuda/h100/reach_h100.csv --suite reach [--omp omp16] [--out x.md]

Times are the median over the OK repetitions. "x-fach" is other_time / cuda_time, so > 1 means the
GPU is faster. A timeout on the other side gives a lower bound (">="). The verifyta numbers were
recorded on another host, so the UPPAAL column compares across machines.
"""
import argparse, csv, os, re, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPPAAL = {'reach': 'benchmarks/results/reach/uppaal/verifyta_t1.csv',
          'deadlock': 'benchmarks/results/deadlock/uppaal/verifyta_deadlock_t1.csv'}


def collect(rows, tool_of):
    """(model, tool) -> dict(t=median seconds or None, timeout=seconds or None, rows=[...])"""
    acc = {}
    for r in rows:
        k = (r['model'], tool_of(r))
        acc.setdefault(k, []).append(r)
    out = {}
    for k, rs in acc.items():
        ok = [float(r['seconds']) for r in rs if r['status'] == 'OK']
        to = [float(r['seconds']) for r in rs if r['status'] == 'TIMEOUT']
        out[k] = dict(t=statistics.median(ok) if ok else None, timeout=max(to) if to else None,
                      status=rs[-1]['status'], rows=rs)
    return out


def ratio(other, cuda):
    if cuda['t'] is None:
        return '-'
    if other is None:
        return '-'
    if other['t'] is not None:
        return f"{other['t'] / cuda['t']:.2f}x"
    if other['timeout'] is not None:
        return f">={other['timeout'] / cuda['t']:.1f}x"
    return '-'


def fmt_t(e):
    if e is None:
        return '-'
    if e['t'] is not None:
        return f"{e['t']:.3f}"
    if e['timeout'] is not None:
        return f"TIMEOUT({e['timeout']:.0f})"
    return e['status']


def maxcol(e, col):
    if e is None:
        return '-'
    v = [float(r[col]) for r in e['rows'] if r.get(col, '-') not in ('-', '', None)]
    return f"{max(v):.0f}" if v else '-'


def medcol(e, col):
    v = [float(r[col]) for r in e['rows'] if r['status'] == 'OK' and r.get(col, '-') not in ('-', '', None)]
    return f"{statistics.median(v):.1f}" if v else '-'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--suite', required=True, choices=sorted(UPPAAL))
    ap.add_argument('--omp', default='omp16')
    ap.add_argument('--out')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    hw = sorted({r.get('label', '-') for r in rows})
    me = collect(rows, lambda r: r['tool'])
    up = collect(list(csv.DictReader(open(os.path.join(ROOT, UPPAAL[a.suite])))), lambda r: 'verifyta')
    models = []
    for r in rows:
        if r['tool'] == 'cuda' and r['model'] not in models:
            models.append(r['model'])
    size = lambda m: int(re.search(r'(\d+)$', m).group(1))
    models.sort(key=lambda m: (size(m), m))

    L = [f"### {a.suite} -- CUDA ({', '.join(hw)}) vs hora-{a.omp} vs UPPAAL (verifyta 5.0.0, recorded host)", '',
         f"| Modell | Zustände | CUDA [s] | {a.omp} [s] | CUDA vs {a.omp} | UPPAAL [s] | CUDA vs UPPAAL "
         f"| GPU-Mem nvidia-smi [MiB] | GPU-Alloc hora [MB] | Energie [J] | Ø Leistung [W] | Peak [W] |",
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    sp_o, sp_u = [], []
    for m in models:
        c = me.get((m, 'cuda'))
        o = me.get((m, a.omp))
        u = up.get((m, 'verifyta'))
        st = next((r['states'] for r in c['rows'] if r['status'] == 'OK'), '-')
        ro, ru = ratio(o, c), ratio(u, c)
        if c['t'] and o and o['t']:
            sp_o.append(o['t'] / c['t'])
        if c['t'] and u and u['t']:
            sp_u.append(u['t'] / c['t'])
        L.append(f"| {m} | {st} | {fmt_t(c)} | {fmt_t(o)} | {ro} | {fmt_t(u)} | {ru} "
                 f"| {maxcol(c, 'gpu_mb')} | {maxcol(c, 'gpu_alloc_mb')} | {medcol(c, 'gpu_joules')} "
                 f"| {medcol(c, 'gpu_avg_w')} | {maxcol(c, 'gpu_peak_w')} |")
    gm = lambda v: statistics.geometric_mean(v) if v else float('nan')
    L += ['', f"Geometrisches Mittel: CUDA vs {a.omp} {gm(sp_o):.2f}x ({len(sp_o)} Modelle), "
              f"CUDA vs UPPAAL {gm(sp_u):.2f}x ({len(sp_u)} Modelle)",
          f"Peak GPU-Speicher: {max([float(maxcol(me[(m, 'cuda')], 'gpu_mb')) for m in models if maxcol(me[(m, 'cuda')], 'gpu_mb') != '-'] or [0]):.0f} MiB (nvidia-smi)"]
    text = '\n'.join(L) + '\n'
    print(text)
    if a.out:
        open(a.out, 'w').write(text)


if __name__ == '__main__':
    main()
