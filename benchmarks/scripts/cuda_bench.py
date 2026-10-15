#!/usr/bin/env python3
"""cuda_bench.py -- the reach suite on the CUDA backend, next to the CPU backends on the same host.

For every (family, size) of the reach suite, run each requested tool once per repetition and
record wall time, explored states and the verdict in one CSV, rows keyed like bench.py's:

  cuda_bench.py --hora build/hora --tools cuda,omp16,omp1 --out benchmarks/results/cuda/reach.csv \
                --families arbiter,csma,fischer,grc,phil,single,twolock,viking,hddi --timeout 600

Tools: cuda (--backend cuda), omp<N> (--backend omp, N threads), seq (--backend seq).
gpu_mb is the process's peak device memory as nvidia-smi reports it (CUDA context included);
gpu_alloc_mb is the engine's own allocator peak (HORA_GPU_MEM).
gpu_joules/gpu_avg_w/gpu_peak_w are whole-board energy (NVML counter, idle draw included), mean and peak
power over the run. A timeout is recorded as TIMEOUT with the timeout as its time; a verdict that disagrees with the
first tool's verdict for the same model is recorded as MISMATCH. Resumable: finished rows are kept.
"""
import argparse, csv, ctypes, os, re, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VERD = re.compile(r'-- Formula is (satisfied|NOT satisfied|UNSUPPORTED)'
                  r'[^\[\n]*(?:\[explored (\d+) symbolic states(?:, generated (\d+))?)?')
FIELDS = ['family', 'model', 'tool', 'rep', 'states', 'generated', 'seconds', 'status', 'verdict', 'gpu_mb',
          'gpu_alloc_mb', 'gpu_joules', 'gpu_avg_w', 'gpu_peak_w', 'label', 'query']
GMEM = re.compile(r'\[gpu\] memory \([^)]*\): peak ([0-9.]+) MB')


class Power:
    """Whole-board GPU energy via NVML: the driver's own energy counter (mJ since driver load) read
    before and after the run, plus a 10 ms sampler of the instantaneous draw for average and peak."""

    def __init__(self):
        self.ok = False
        try:
            self.lib = ctypes.CDLL('libnvidia-ml.so.1')
            self.lib.nvmlInit_v2()
            self.h = ctypes.c_void_p()
            self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.h))
            self.ok = self.energy_mj() is not None
        except Exception:
            self.ok = False

    def energy_mj(self):
        v = ctypes.c_ulonglong()
        if self.lib.nvmlDeviceGetTotalEnergyConsumption(self.h, ctypes.byref(v)) != 0:
            return None
        return v.value

    def watts(self):
        v = ctypes.c_uint()
        if self.lib.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(v)) != 0:
            return None
        return v.value / 1000.0

    def start(self):
        self.samples, self.stop = [], False
        self.e0, self.t0 = (self.energy_mj() if self.ok else None), time.time()
        if self.ok:
            def loop():
                while not self.stop:
                    w = self.watts()
                    if w is not None:
                        self.samples.append(w)
                    time.sleep(0.01)
            self.th = threading.Thread(target=loop, daemon=True)
            self.th.start()

    def finish(self):
        """-> dict(gpu_joules, gpu_avg_w, gpu_peak_w); '-' when NVML has no energy counter."""
        if not self.ok:
            return dict(gpu_joules='-', gpu_avg_w='-', gpu_peak_w='-')
        self.stop = True
        self.th.join()
        e1, dt = self.energy_mj(), time.time() - self.t0
        j = (e1 - self.e0) / 1000.0 if e1 is not None and self.e0 is not None else None
        pk = max(self.samples) if self.samples else None
        return dict(gpu_joules=f'{j:.1f}' if j is not None else '-',
                    gpu_avg_w=f'{j / dt:.1f}' if j is not None and dt > 0 else '-',
                    gpu_peak_w=f'{pk:.0f}' if pk is not None else '-')


POWER = Power()


def size_of(m):
    return int(re.search(r'(\d+)$', m).group(1))


def run(hora, tool, model, query, timeout, queryset):
    env = dict(os.environ)
    args = [hora, '--stats', model, query]
    if tool == 'cuda':
        args[1:1] = ['--backend', 'cuda']
        env['OMP_NUM_THREADS'] = env.get('HORA_CUDA_OMP', '16')
        env['HORA_GPU_MEM'] = '1'        # the engine's own allocator peak, on stderr
    elif tool == 'seq':
        args[1:1] = ['--backend', 'seq']
        env['OMP_NUM_THREADS'] = '1'
    elif tool.startswith('omp'):
        args[1:1] = ['--backend', 'omp']
        env['OMP_NUM_THREADS'] = tool[3:] or '16'
    t = time.time()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    POWER.start()
    peak = 0                       # peak GPU memory of this process, MiB, polled from nvidia-smi
    while True:
        try:
            out, err = proc.communicate(timeout=0.05)
            break
        except subprocess.TimeoutExpired:
            if time.time() - t > timeout:
                proc.kill(); proc.communicate()
                return dict(seconds=timeout, status='TIMEOUT', states='-', generated='-', verdict='-',
                            gpu_mb=peak or '-', gpu_alloc_mb='-', **POWER.finish())
            q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            mine, total = 0, 0
            for ln in q.stdout.splitlines():
                f = [x.strip() for x in ln.split(',')]
                if len(f) == 2 and f[1].isdigit():
                    total += int(f[1])
                    if f[0] == str(proc.pid):
                        mine = int(f[1])
            # In a container nvidia-smi reports host PIDs, which never match ours; the benchmark
            # runs one process at a time, so every compute app on the device is then this one.
            peak = max(peak, mine or total)
    dt = time.time() - t
    pw = POWER.finish()
    p = subprocess.CompletedProcess(args, proc.returncode, out, err)
    gm = peak or '-'
    am = [float(x) for x in GMEM.findall(p.stderr)]
    ga = f'{max(am):.1f}' if am else '-'
    ms = list(VERD.finditer(p.stdout))
    if p.returncode not in (0, 2) or not ms or 'falling back' in p.stderr:
        return dict(seconds=dt, status='ERR', states='-', generated='-', verdict='-', gpu_mb=gm, gpu_alloc_mb=ga,
                    **pw, err=(p.stderr.strip()[-200:]))
    m = ms[0]
    return dict(seconds=dt, status='OK', states=m.group(2) or '-', generated=m.group(3) or '-', gpu_mb=gm, gpu_alloc_mb=ga, **pw,
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
    ap.add_argument('--label', default='-', help='hardware tag stored in every row, e.g. H100')
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
                    row = dict(family=fam, model=model, tool=tool, rep=rep, query=qtext, label=a.label,
                               **{k: r[k] for k in ('states', 'generated', 'seconds', 'status', 'verdict', 'gpu_mb',
                                                    'gpu_alloc_mb', 'gpu_joules', 'gpu_avg_w', 'gpu_peak_w')})
                    if r['status'] == 'OK':
                        v = verdicts.setdefault(model, r['verdict'])
                        if v != r['verdict']:
                            row['status'] = 'MISMATCH'
                    row['seconds'] = f"{float(r['seconds']):.3f}"
                    w.writerow(row)
                    out.flush()
                    print(f"{model:12s} {tool:7s} rep{rep} {row['status']:8s} {row['seconds']:>9s}s "
                          f"states {row['states']} {row['verdict']} gpu {row['gpu_mb']}MiB alloc {row['gpu_alloc_mb']}MB {row['gpu_joules']}J avg {row['gpu_avg_w']}W peak {row['gpu_peak_w']}W {r.get('err', '')}", flush=True)
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
