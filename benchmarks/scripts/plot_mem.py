#!/usr/bin/env python3
"""Render peak-RSS (memory footprint) figures from mem_profile.py's CSV.

  python3 benchmarks/plot_mem.py [mem_profile.csv]  ->  benchmarks/plots/memory_*.png

Produces, per model with >= 2 measured points:
  - memory_<model>.png : peak RSS (GB) vs N, one line per backend, with a horizontal
                         dashed line at 24 GB (the M4 Pro unified-memory ceiling).
and one combined panel figure:
  - memory_all.png     : the same, one subplot per model.
Blank cells (OOM/timeout) are simply skipped for that curve.
"""
import csv, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "benchmarks", "mem_profile.csv")
OUT = os.path.join(ROOT, "benchmarks", "plots")
os.makedirs(OUT, exist_ok=True)

MACHINE_GB = 24.0   # Apple M4 Pro unified memory

# (csv column, colour, marker, legend label) — matches plot_results.py's palette.
BACKENDS = [("rss_seq_gb", "#888888", "o", "seq"),
            ("rss_omp_gb", "#1f77b4", "s", "omp"),
            ("rss_metal_gb", "#d62728", "^", "metal"),
            ("rss_verifyta_gb", "#2ca02c", "D", "verifyta")]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


rows = defaultdict(list)
with open(CSV) as fp:
    for r in csv.DictReader(fp):
        rows[r["bench"]].append(r)


def draw(ax, rs, title):
    rs = sorted(rs, key=lambda r: int(r["N"]))
    plotted = False
    for key, col, mk, lab in BACKENDS:
        xs = [int(r["N"]) for r in rs if fnum(r[key]) is not None]
        ys = [fnum(r[key]) for r in rs if fnum(r[key]) is not None]
        if len(xs) >= 2:
            ax.plot(xs, ys, marker=mk, color=col, label=lab)
            plotted = True
    ax.axhline(MACHINE_GB, color="k", ls="--", lw=1.0)
    ax.text(0.02, MACHINE_GB, " 24 GB machine limit", color="k", fontsize=7,
            va="bottom", ha="left", transform=ax.get_yaxis_transform())
    ax.set_title(title)
    ax.set_xlabel("N (model size)")
    ax.set_ylabel("peak RSS (GB)")
    ax.grid(True, alpha=0.3)
    return plotted


# ---- per-model figures --------------------------------------------------------
made = []
for bench, rs in rows.items():
    fig, ax = plt.subplots(figsize=(6, 4))
    if not draw(ax, rs, f"{bench}: peak memory footprint"):
        plt.close(fig); continue
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"memory_{bench}.png"), dpi=130)
    plt.close(fig)
    made.append(bench)

# ---- combined panel figure ----------------------------------------------------
models = sorted(rows)
n = len(models)
cols = 3
rws = (n + cols - 1) // cols
fig, axes = plt.subplots(rws, cols, figsize=(4.6 * cols, 3.4 * rws), squeeze=False)
for idx, bench in enumerate(models):
    ax = axes[idx // cols][idx % cols]
    if draw(ax, rows[bench], bench):
        ax.legend(fontsize=7)
for j in range(n, rws * cols):
    axes[j // cols][j % cols].axis("off")
fig.suptitle("Peak RSS vs model size — metal double-buffers zones and hits the 24 GB ceiling first")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(OUT, "memory_all.png"), dpi=130)
plt.close(fig)

print(f"wrote {OUT}/memory_all.png and per-model: {', '.join('memory_'+b for b in made)}")
