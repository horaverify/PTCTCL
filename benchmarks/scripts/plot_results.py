#!/usr/bin/env python3
"""Render the evaluation figures from the master bench_run.py CSV (5 configs + memory).

  python3 benchmarks/plot_results.py [results_full.csv]  ->  benchmarks/plots/*.png

Figures:
  scaling_<model>.png     wall-clock vs N (log y): seq/omp/metal/metal-ad/verifyta
  speedup_vs_verifyta.png per-model panels: (verifyta / X) for omp, metal, metal-ad
  adaptive_speedup.png    per-model: metal / metal-ad  (how much --adaptive buys)
  memory_<model>.png      peak RSS (GB) vs N per config, with the 24 GB machine limit
  memory_all.png          combined RSS panels
Rows with missing values (ceiling/timeout) are skipped for that curve.
"""
import csv, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # PTCTCL/
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "benchmarks", "results_full.csv")
OUT = os.path.join(ROOT, "benchmarks", "plots"); os.makedirs(OUT, exist_ok=True)

def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None

rows = defaultdict(list)
with open(CSV) as fp:
    for r in csv.DictReader(fp): rows[r["bench"]].append(r)
for b in rows: rows[b].sort(key=lambda r: int(r["N"]))
models = sorted(rows)

# config -> (csv time column, csv rss column, colour, marker, label)
TIME = [("t_seq", "#999999", "o", "seq"), ("t_omp", "#1f77b4", "s", "omp"),
        ("t_metal", "#d62728", "^", "metal (naive)"), ("t_metal-ad", "#e6a000", "D", "metal (adaptive)"),
        ("t_verifyta", "#2ca02c", "v", "verifyta")]
RSS = [("rss_seq", "#999999", "seq"), ("rss_omp", "#1f77b4", "omp"),
       ("rss_metal", "#d62728", "metal (naive)"), ("rss_metal-ad", "#e6a000", "metal (adaptive)"),
       ("rss_verifyta", "#2ca02c", "verifyta")]

def panels(n):
    cols = 3; rws = (n + cols - 1) // cols
    fig, ax = plt.subplots(rws, cols, figsize=(4.3 * cols, 3.2 * rws), squeeze=False)
    return fig, ax, cols, rws

# ---- per-model wall-clock scaling ----
for b in models:
    fig, a = plt.subplots(figsize=(6, 4))
    for col, c, mk, lab in TIME:
        xs = [int(r["N"]) for r in rows[b] if fnum(r[col]) is not None]
        ys = [fnum(r[col]) for r in rows[b] if fnum(r[col]) is not None]
        if len(xs) >= 2: a.plot(xs, ys, marker=mk, color=c, label=lab, ms=5)
    a.set_yscale("log"); a.set_xlabel("N"); a.set_ylabel("wall-clock (s, min-of-k, log)")
    a.set_title(f"{b}: scaling"); a.grid(True, which="both", alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"scaling_{b}.png"), dpi=130); plt.close(fig)

# ---- speedup over verifyta (omp, metal, metal-ad) ----
fig, ax, cols, rws = panels(len(models))
for i, b in enumerate(models):
    a = ax[i // cols][i % cols]
    for col, c, lab in [("vfy/omp", "#1f77b4", "verifyta/omp"), ("vfy/metal", "#d62728", "verifyta/metal"),
                        ("vfy/metal-ad", "#e6a000", "verifyta/metal-ad")]:
        xs = [int(r["N"]) for r in rows[b] if fnum(r[col]) is not None]
        ys = [fnum(r[col]) for r in rows[b] if fnum(r[col]) is not None]
        if xs: a.plot(xs, ys, marker="o", color=c, label=lab, ms=4)
    a.axhline(1.0, color="k", ls="--", lw=0.8); a.set_title(b); a.set_xlabel("N")
    a.set_ylabel("speedup vs verifyta"); a.grid(True, alpha=0.3); a.legend(fontsize=7)
for j in range(len(models), rws * cols): ax[j // cols][j % cols].axis("off")
fig.suptitle("Speedup over single-threaded UPPAAL verifyta (>1 = faster; dashed = break-even)")
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(OUT, "speedup_vs_verifyta.png"), dpi=130); plt.close(fig)

# ---- adaptive win: metal (naive) / metal-adaptive ----
fig, a = plt.subplots(figsize=(7, 4.2))
for b in models:
    xs = [int(r["N"]) for r in rows[b] if fnum(r["metal/metal-ad"]) is not None]
    ys = [fnum(r["metal/metal-ad"]) for r in rows[b] if fnum(r["metal/metal-ad"]) is not None]
    if xs: a.plot(xs, ys, marker="o", label=b, ms=5)
a.axhline(1.0, color="k", ls="--", lw=0.8)
a.set_xlabel("N"); a.set_ylabel("naive metal / adaptive metal  (×)")
a.set_title("What --adaptive buys (>1 = adaptive faster)")
a.grid(True, alpha=0.3); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "adaptive_speedup.png"), dpi=130); plt.close(fig)

# ---- per-model memory ----
for b in models:
    fig, a = plt.subplots(figsize=(6, 4))
    for col, c, lab in RSS:
        xs = [int(r["N"]) for r in rows[b] if fnum(r[col]) is not None]
        ys = [fnum(r[col]) for r in rows[b] if fnum(r[col]) is not None]
        if len(xs) >= 2: a.plot(xs, ys, marker="o", color=c, label=lab, ms=4)
    a.axhline(24, color="k", ls=":", lw=1.0, label="24 GB machine")
    a.set_xlabel("N"); a.set_ylabel("peak RSS (GB)"); a.set_title(f"{b}: memory")
    a.grid(True, alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"memory_{b}.png"), dpi=130); plt.close(fig)

# ---- combined memory ----
fig, ax, cols, rws = panels(len(models))
for i, b in enumerate(models):
    a = ax[i // cols][i % cols]
    for col, c, lab in RSS:
        xs = [int(r["N"]) for r in rows[b] if fnum(r[col]) is not None]
        ys = [fnum(r[col]) for r in rows[b] if fnum(r[col]) is not None]
        if xs: a.plot(xs, ys, marker="o", color=c, label=lab, ms=3)
    a.set_title(b); a.set_xlabel("N"); a.set_ylabel("peak RSS (GB)"); a.grid(True, alpha=0.3); a.legend(fontsize=6)
for j in range(len(models), rws * cols): ax[j // cols][j % cols].axis("off")
fig.suptitle("Peak resident memory (GB) — metal double-buffers zones into GPU-facing buffers")
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(OUT, "memory_all.png"), dpi=130); plt.close(fig)

# ---- chunking trade-off: time and peak RSS vs chunk size, at each model's largest N ----
CH = [("t_metal-ad", "rss_metal-ad", "off"), ("t_mad-c256k", "rss_mad-c256k", "256k"),
      ("t_mad-c64k", "rss_mad-c64k", "64k"), ("t_mad-c16k", "rss_mad-c16k", "16k"),
      ("t_mad-c4k", "rss_mad-c4k", "4k")]
if all(c[0] in rows[models[0]][0] for c in CH):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    xs = list(range(len(CH)))
    for b in models:
        last = rows[b][-1]
        t = [fnum(last[c[0]]) for c in CH]
        r = [fnum(last[c[1]]) for c in CH]
        lab = f"{b} N={last['N']}"
        if any(v is not None for v in t):
            a1.plot(xs, [v if v else float('nan') for v in t], marker="o", label=lab, ms=5)
        if any(v is not None for v in r):
            a2.plot(xs, [v if v else float('nan') for v in r], marker="o", label=lab, ms=5)
    for a, yl, ti in ((a1, "wall-clock (s, log)", "Time cost of chunking"),
                      (a2, "peak RSS (GB)", "Memory effect of chunking")):
        a.set_xticks(xs); a.set_xticklabels([c[2] for c in CH])
        a.set_xlabel("chunk size (frontier states per batch)"); a.set_ylabel(yl)
        a.set_title(ti); a.grid(True, alpha=0.3); a.legend(fontsize=7)
    a1.set_yscale("log")
    fig.suptitle("--chunking bounds the GPU pools but does not reduce peak RSS — and costs time")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "chunking_tradeoff.png"), dpi=130); plt.close(fig)

print(f"wrote figures to {OUT}/  (scaling/memory per model + speedup_vs_verifyta + adaptive_speedup"
      f" + memory_all + chunking_tradeoff)")
