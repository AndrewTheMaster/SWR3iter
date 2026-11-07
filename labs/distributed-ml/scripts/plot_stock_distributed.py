#!/usr/bin/env python3
"""Plot stock distributed training: fit time and holdout Logloss vs workers.

Input: results/stage2_stock/records.json (run summarize_stock_distributed.py first
if you need the TSV). Output: figures/milestones/stock_distributed_workers.png.
Dashed lines are the fastest single-host run per dataset (no CTR).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results/stage2_stock/records.json"
OUT = ROOT / "figures/milestones/stock_distributed_workers.png"
DATASETS = [("amazon_onehot", "Amazon one-hot"), ("kdd_num", "KDD numeric"), ("criteo_num", "Criteo numeric")]


def main() -> None:
    records = json.loads(RECORDS.read_text(encoding="utf-8"))
    fig, (ax_fit, ax_ll) = plt.subplots(1, 2, figsize=(10, 4))
    for key, label in DATASETS:
        dist = sorted(
            (r for r in records if r["name"].startswith(key + "_dist") and r.get("exit_code") == 0),
            key=lambda r: r["workers"],
        )
        singles = [r for r in records if r["name"].startswith(key + "_single") and r.get("exit_code") == 0]
        xs = [r["workers"] for r in dist]
        line, = ax_fit.plot(xs, [r["fit_wall_s"] for r in dist], marker="o", label=label)
        ax_ll.plot(xs, [r["quality"]["Logloss"] for r in dist], marker="o", color=line.get_color(), label=label)
        if singles:
            best = min(r["fit_wall_s"] for r in singles)
            ax_fit.axhline(best, color=line.get_color(), linestyle="--", linewidth=1)

    ax_fit.set_xlabel("workers (3 threads each)")
    ax_fit.set_ylabel("fit time, s")
    ax_fit.set_title("Stock distributed fit (dashed: best single-host)")
    ax_fit.set_xticks([1, 2, 4])
    ax_fit.legend()
    ax_ll.set_xlabel("workers (3 threads each)")
    ax_ll.set_ylabel("holdout Logloss")
    ax_ll.set_title("Quality stays flat with workers")
    ax_ll.set_xticks([1, 2, 4])
    ax_ll.legend()
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"figure written to {OUT}")


if __name__ == "__main__":
    main()
