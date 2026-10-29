"""Summarise results/eval/<condition>/episodes.jsonl into results/eval_table.md with Wilson CIs.

    python fr3duo_pipeline/eval/summarize.py results/eval [--compare B_native C_control]

Also reports, when both conditions are present, the paired outcome table on the shared seeds and an
exact McNemar p-value (every condition runs the identical seed list, so pairing is legitimate).
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant counts b (A ok, B fail) and c (A fail, B ok)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--compare", nargs=2, default=None, metavar=("COND_X", "COND_Y"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    conds = {}
    for d in sorted(a.root.iterdir()):
        f = d / "episodes.jsonl"
        if f.is_file():
            recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            conds[d.name] = {r["seed"]: r for r in recs}
    lines = ["| condition | n | success | rate | Wilson 95 % | n_action_steps | mean infer ms |", "|---|---|---|---|---|---|---|"]
    for name, recs in conds.items():
        n = len(recs)
        k = sum(r["success"] for r in recs.values())
        lo, hi = wilson(k, n)
        nas = sorted({r.get("n_action_steps") for r in recs.values()})
        ms = [r["infer_ms_mean"] for r in recs.values() if r.get("infer_ms_mean")]
        lines.append(f"| {name} | {n} | {k} | {100 * k / n:.0f} % | {100 * lo:.0f}–{100 * hi:.0f} % | {nas} | {np.mean(ms):.0f} |" if ms
                     else f"| {name} | {n} | {k} | {100 * k / n:.0f} % | {100 * lo:.0f}–{100 * hi:.0f} % | {nas} | – |")
    if a.compare and all(c in conds for c in a.compare):
        x, y = a.compare
        seeds = sorted(set(conds[x]) & set(conds[y]))
        b = sum(conds[x][s]["success"] and not conds[y][s]["success"] for s in seeds)
        c = sum((not conds[x][s]["success"]) and conds[y][s]["success"] for s in seeds)
        both = sum(conds[x][s]["success"] and conds[y][s]["success"] for s in seeds)
        neither = len(seeds) - b - c - both
        lines += ["", f"Paired on {len(seeds)} shared seeds: {x} vs {y}: both ok {both}, only {x} {b}, only {y} {c}, neither {neither}; "
                  f"exact McNemar p = {mcnemar_exact(b, c):.4f}"]
    text = "\n".join(lines)
    out = a.out or a.root.parent / "eval_table.md"
    out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
