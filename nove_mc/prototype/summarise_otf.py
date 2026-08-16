#!/usr/bin/env python3
"""Aggregate otf_results.json / otf_scale.json into the table the write-up quotes.

Splits by VERDICT, because that is the whole point: an on-the-fly scheme can only save
states on violated queries, and a summary that pools the two classes hides both the win
and the absence of a loss.
"""
import json, sys, statistics

def load(path):
    with open(path) as f:
        return json.load(f)

def geo(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return float("nan")
    return statistics.geometric_mean(xs)

def main(paths, baseline="gfp"):
    rows = {}
    for p in paths:
        rows.update(load(p))
    modes = []
    for v in rows.values():
        for m in v:
            if m not in modes:
                modes.append(m)

    for cls, want in (("VIOLATED", "NOT"), ("SATISFIED", "SATISFIED")):
        sel = {k: v for k, v in rows.items()
               if v.get(baseline, {}).get("verdict") == want}
        if not sel:
            continue
        print(f"\n=== {cls}  ({len(sel)} queries) ===")
        print(f"{'mode':<6} {'total states':>13} {'geo-mean vs '+baseline:>22} {'best':>8} {'worst':>8}")
        base_tot = sum(v[baseline]["nodes"] or 0 for v in sel.values())
        for m in modes:
            tot, ratios = 0, []
            ok = True
            for v in sel.values():
                a, b = v[baseline].get("nodes"), v.get(m, {}).get("nodes")
                if b is None:
                    ok = False
                    continue
                tot += b
                if a:
                    ratios.append(a / b)
            if not ok and tot == 0:
                continue
            print(f"{m:<6} {tot:>13} {geo(ratios):>21.2f}x {max(ratios):>7.1f}x {min(ratios):>7.1f}x")
        print(f"  ({baseline} total = {base_tot})")

    # capped accounting
    print("\n=== federation cap (verdict UNUSABLE) ===")
    for m in modes:
        bad = [k for k, v in rows.items() if v.get(m, {}).get("verdict") == "UNUSABLE"]
        print(f"  {m:<6} {len(bad):>3} / {len(rows)}   {', '.join(bad[:4])}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["otf_results.json"])
