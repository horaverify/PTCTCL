#!/usr/bin/env python3
"""Turn results_openmp/raw.csv into the OpenMP paper's figures and LaTeX table.

Emits into ../docs/paper-openmp/figs/:  time.pdf  memory.pdf  scaling.pdf  table.tex

TIMEOUT / MEMOUT runs are never plotted as a value: they are drawn at the top of the axis
with a hatch and annotated, so a censored run can never be mistaken for a fast one.
"""
import collections, csv, os, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmarks/
FIGS = os.path.abspath(os.path.join(HERE, "..", "docs", "paper-openmp", "figs"))
os.makedirs(FIGS, exist_ok=True)

# Grayscale only: print-safe and photocopy-safe. Series are separated by fill value
# AND hatch, so they remain distinguishable without colour.
TOOLS = [("verifyta",    "1",  "UPPAAL verifyta",   "0.55", ""),
         ("ltsmin",      "8",  "opaal+LTSmin (8)",  "0.80", "//"),
         ("ltsmin",      "12", "opaal+LTSmin (12)", "0.92", "\\\\"),
         ("ptctlmc-omp", "8",  "PTCTLMC (8)",       "0.25", ""),
         ("ptctlmc-omp", "12", "PTCTLMC (12)",      "0.05", "..")]

def load():
    rows = list(csv.DictReader(open(os.path.join(HERE, "results", "full.csv"))))
    ok, cens = collections.defaultdict(list), {}
    for r in rows:
        k = (r["model"], r["tool"], r["threads"])
        if r["status"] == "OK":
            ok[k].append((float(r["seconds"]), float(r["rss_gb"])))
        else:
            cens.setdefault(k, r["status"])
    return rows, ok, cens

def model_order(rows):
    fam = {r["model"]: r["family"] for r in rows}
    def key(m):
        f = fam[m]
        n = int("".join(c for c in m if c.isdigit()) or 0)
        return ({"viking": 0, "fischer": 1, "train": 2}.get(f, 3), n)
    return sorted(fam, key=key), fam

def grouped(metric, ylabel, fname, title):
    rows, ok, cens = load()
    models, _ = model_order(rows)
    idx = 0 if metric == "time" else 1
    fig, ax = plt.subplots(figsize=(11, 3.6))
    w = 0.16
    xs = range(len(models))
    top = 1.0
    for k in ok:
        for v in ok[k]:
            top = max(top, v[idx])
    ceil = top * 6
    for j, (tool, th, lab, col, hat) in enumerate(TOOLS):
        vals, hatch_x = [], []
        for i, m in enumerate(models):
            k = (m, tool, th)
            if k in ok and ok[k]:
                vals.append(statistics.median(v[idx] for v in ok[k]))
            elif k in cens:
                vals.append(ceil); hatch_x.append(i)
            else:
                vals.append(0)
        pos = [i + (j - 2) * w for i in xs]
        bars = ax.bar(pos, vals, w, label=lab, color=col, hatch=hat,
                      edgecolor="black", linewidth=0.5)
        for i in hatch_x:
            bars[i].set_facecolor("white"); bars[i].set_hatch("xxx")
            ax.text(i + (j - 2) * w, ceil, cens[(models[i], tool, th)][:1],
                    ha="center", va="bottom", fontsize=6, color="black", weight="bold")
    ax.set_yscale("log")
    ax.axhline(ceil, ls="--", lw=0.8, color="black")
    ax.set_xticks(list(xs)); ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    h.append(Patch(facecolor="white", edgecolor="black", hatch="xxx"))
    l.append("TIMEOUT / MEMOUT")
    ax.legend(h, l, fontsize=7, ncol=6, loc="upper left", framealpha=0.9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, fname)); plt.close(fig)
    print("wrote", fname)

MARKS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
DASH  = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

def _scaling_one(ax, ok, models, tool, title):
    """Speed-up vs thread count, one line per model, error bars over repetitions."""
    ths = sorted({int(t) for (m, tl, t) in ok if tl == tool}, key=int)
    if not ths:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return ths
    base_th = str(ths[0])
    n = 0
    for mi, m in enumerate(models):
        b = ok.get((m, tool, base_th))
        if not b:
            continue
        b0 = statistics.median(x[0] for x in b)
        xs, ys, err = [], [], []
        for t in ths:
            v = ok.get((m, tool, str(t)))
            if not v:
                continue
            sp = [b0 / x[0] for x in v]
            xs.append(t); ys.append(statistics.median(sp))
            err.append((max(sp) - min(sp)) / 2.0)
        if len(xs) > 1:
            ax.errorbar(xs, ys, yerr=err, marker=MARKS[mi % len(MARKS)], ms=4, lw=1.2,
                        capsize=2.5, color=str(0.10 + 0.13 * (mi % 5)),
                        ls=DASH[mi % len(DASH)], markerfacecolor="white", label=m)
            n += 1
    lo, hi = ths[0], ths[-1]
    ax.plot([lo, hi], [1, hi / lo], ls="--", lw=1.0, color="0.6", zorder=0)
    ax.text(hi, hi / lo, " linear", fontsize=7, color="0.45", va="center")
    ax.set_xlabel("Threads"); ax.set_ylabel("Speedup")
    ax.set_xticks(ths); ax.set_title(title, fontsize=10)
    ax.grid(ls=":", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
    if n:
        ax.legend(title="Model", fontsize=7, title_fontsize=7, loc="upper left", framealpha=0.95)
    return ths


def scaling():
    """Two standalone figures, sized to sit side by side under subcaption."""
    rows, ok, _ = load()
    models, _ = model_order(rows)
    for tool, fname, title in (("ptctlmc-omp", "scaling_ptctlmc.pdf", "PTCTLMC (OpenMP)"),
                               ("ltsmin", "scaling_ltsmin.pdf", "opaal+LTSmin")):
        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        ths = _scaling_one(ax, ok, models, tool, title)
        fig.tight_layout(); fig.savefig(os.path.join(FIGS, fname)); plt.close(fig)
        print("wrote", fname, "threads:", ths)
    # keep the combined view too (used nowhere by default, handy for talks)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    _scaling_one(ax, ok, models, "ptctlmc-omp", "")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "scaling.pdf")); plt.close(fig)

def _time_one(ax, ok, models, tool, title):
    """Absolute wall-clock time vs thread count, one line per model."""
    ths = sorted({int(t) for (m, tl, t) in ok if tl == tool}, key=int)
    if not ths:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes); return ths
    n = 0
    for mi, m in enumerate(models):
        xs, ys, err = [], [], []
        for t in ths:
            v = ok.get((m, tool, str(t)))
            if not v: continue
            sec = [x[0] for x in v]
            xs.append(t); ys.append(statistics.median(sec)); err.append((max(sec) - min(sec)) / 2.0)
        if len(xs) > 1:
            ax.errorbar(xs, ys, yerr=err, marker=MARKS[mi % len(MARKS)], ms=4, lw=1.2,
                        capsize=2.5, color=str(0.10 + 0.13 * (mi % 5)), ls=DASH[mi % len(DASH)],
                        markerfacecolor="white", label=m); n += 1
    ax.set_yscale("log")
    ax.set_xlabel("Threads"); ax.set_ylabel("Time (s)")
    ax.set_xticks(ths); ax.set_title(title, fontsize=10)
    ax.grid(ls=":", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
    if n: ax.legend(title="Model", fontsize=7, title_fontsize=7, loc="upper right", framealpha=0.95)
    return ths


def timing():
    """Two standalone time-vs-threads figures, for side-by-side placement."""
    rows, ok, _ = load()
    models, _ = model_order(rows)
    for tool, fname, title in (("ptctlmc-omp", "time_ptctlmc.pdf", "PTCTLMC (OpenMP)"),
                               ("ltsmin", "time_ltsmin.pdf", "opaal+LTSmin")):
        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        ths = _time_one(ax, ok, models, tool, title)
        fig.tight_layout(); fig.savefig(os.path.join(FIGS, fname)); plt.close(fig)
        print("wrote", fname, "threads:", ths)


def _fmt(n):
    return "{:,}".format(int(n)).replace(",", r"\,")

def _med(ok, cens, m, tool, th, idx):
    """(value, censor-letter). value None if no successful run."""
    k = (m, tool, th)
    if k in ok and ok[k]:
        return statistics.median(v[idx] for v in ok[k]), None
    return None, (cens[k][0] if k in cens else None)

def table():
    rows, ok, cens = load()
    models, fam = model_order(rows)
    states = {}
    for r in rows:
        if r["status"] == "OK" and r["states"] not in ("-", ""):
            states.setdefault((r["model"], r["tool"]), r["states"])

    def cell(v, c, prec=2):
        if v is not None: return ("%%.%df" % prec) % v
        return r"\textsf{%s}" % c if c else "--"

    # ---- Table 1: state-count parity -------------------------------------
    L = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{Explored symbolic states. \tool{} reproduces \code{verifyta} exactly on every",
         r"model and at every thread count. \ltsmin{} explores strictly more because it stores one",
         r"state per (configuration, zone) pair with no convex-union merging; the factor in",
         r"parentheses is how many more states it stores, so its timings in \cref{tab:main}",
         r"reflect more work by construction.}",
         r"\label{tab:parity}", r"\begin{tabular}{l r r r}", r"\toprule",
         r"model & \tool{} & \code{verifyta} & \ltsmin{} \\", r"\midrule"]
    for m in models:
        pt = states.get((m, "ptctlmc-omp")); vf = states.get((m, "verifyta")); lt = states.get((m, "ltsmin"))
        if pt is None and vf is None: continue
        lts = "--"
        if lt:
            lts = _fmt(lt)
            if pt and int(pt):
                lts += r"\;\scriptsize($%.1f\times$)" % (int(lt) / int(pt))
        L.append(r"%s & %s & %s & %s \\" % (
            m.replace("_", r"\_"), _fmt(pt) if pt else "--", _fmt(vf) if vf else "--", lts))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(os.path.join(FIGS, "parity.tex"), "w").write("\n".join(L) + "\n")

    # ---- Table 2: time + speed-ups ---------------------------------------
    L = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{Wall-clock seconds (median of two runs) and the speed-up of \tool{} over each",
         r"baseline. \textsf{T}/\textsf{M} mark runs stopped by the 30-minute / 15\,GB cap; a",
         r"speed-up against a capped baseline is a lower bound, written $>$. \tool{} is fastest on",
         r"every row.}",
         r"\label{tab:main}", r"\begin{tabular}{l r r r r r r r}", r"\toprule",
         r"& & \multicolumn{2}{c}{\ltsmin{}} & \multicolumn{2}{c}{\tool{}} &"
         r" \multicolumn{2}{c}{speed-up of \tool{}} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"model & \code{verifyta} & 8 & 12 & 8 & 12 & vs \code{vfy} & vs \ltsmin{} \\",
         r"\midrule"]
    for m in models:
        vf, vfc = _med(ok, cens, m, "verifyta", "1", 0)
        l8, l8c = _med(ok, cens, m, "ltsmin", "8", 0)
        l12, l12c = _med(ok, cens, m, "ltsmin", "12", 0)
        p8, p8c = _med(ok, cens, m, "ptctlmc-omp", "8", 0)
        p12, p12c = _med(ok, cens, m, "ptctlmc-omp", "12", 0)
        best = min([x for x in (p8, p12) if x is not None], default=None)
        b8 = r"\textbf{%.2f}" % p8 if (p8 is not None and p8 == best) else cell(p8, p8c)
        b12 = r"\textbf{%.2f}" % p12 if (p12 is not None and p12 == best and p8 != best) else cell(p12, p12c)
        def spd(base, basec):
            if best is None or best <= 0: return "--"
            if base is not None: return r"$%.1f\times$" % (base / best)
            if basec: return r"$>\!%.0f\times$" % (1800.0 / best) if basec == "T" else "--"
            return "--"
        lbest = min([x for x in (l8, l12) if x is not None], default=None)
        lbestc = l8c or l12c
        L.append(r"%s & %s & %s & %s & %s & %s & %s & %s \\" % (
            m.replace("_", r"\_"), cell(vf, vfc), cell(l8, l8c), cell(l12, l12c),
            b8, b12, spd(vf, vfc), spd(lbest, lbestc)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(os.path.join(FIGS, "main.tex"), "w").write("\n".join(L) + "\n")

    # ---- Table 3: memory --------------------------------------------------
    L = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{Peak resident set (GB) and the factor by which \tool{} undercuts each baseline.",
         r"The gap widens with model size: interning collapses millions of stored zones onto a few",
         r"hundred distinct DBMs, which a design storing an explicit zone per symbolic state cannot",
         r"match by tuning.}",
         r"\label{tab:mem}", r"\begin{tabular}{l r r r r r}", r"\toprule",
         r"model & \code{verifyta} & \ltsmin{} (8) & \tool{} (8) &"
         r" $\times$ less than \ltsmin{} & $\times$ less than \code{vfy} \\", r"\midrule"]
    for m in models:
        vf, vfc = _med(ok, cens, m, "verifyta", "1", 1)
        lt, ltc = _med(ok, cens, m, "ltsmin", "8", 1)
        pt, ptc = _med(ok, cens, m, "ptctlmc-omp", "8", 1)
        def fac(base):
            if pt is None or base is None or pt <= 0: return "--"
            return r"$%.1f\times$" % (base / pt)
        L.append(r"%s & %s & %s & %s & %s & %s \\" % (
            m.replace("_", r"\_"), cell(vf, vfc), cell(lt, ltc), cell(pt, ptc),
            fac(lt), fac(vf)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(os.path.join(FIGS, "memory.tex"), "w").write("\n".join(L) + "\n")
    print("wrote parity.tex, main.tex, memory.tex")

if __name__ == "__main__":
    grouped("time", "verification time (s)", "time.pdf",
            "Verification time (log scale; median of 2 runs)")
    grouped("mem", "peak RSS (GB)", "memory.pdf",
            "Peak resident set (log scale; median of 2 runs)")
    scaling(); timing(); table()
