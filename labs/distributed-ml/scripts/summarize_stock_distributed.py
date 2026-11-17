#!/usr/bin/env python3
"""Summarize the stock CatBoost distributed runs (run-worker protocol).

Reads results/stage2_stock/records.json and writes summary.tsv next to it.
Each row is one run; `fit_vs_best_single` divides its fit time by the fastest
single-host run on the same dataset (same model family, no CTR).
Failed runs (exit code != 0) are kept in the table with empty metrics.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results/stage2_stock"
COLUMNS = [
    "name", "dataset", "mode", "workers", "threads_per_worker", "total_threads",
    "iterations", "train_rows", "fit_s", "startup_s", "total_s",
    "logloss", "auc", "exit_code", "fit_vs_best_single",
]


def dataset_of(name: str) -> str:
    return name.split("_")[0]


def main() -> None:
    records = json.loads((RUN_DIR / "records.json").read_text(encoding="utf-8"))

    best_single: dict[str, float] = {}
    for r in records:
        if r["mode"] == "single" and r.get("exit_code", 0) == 0:
            ds = dataset_of(r["name"])
            fit = r["fit_wall_s"]
            best_single[ds] = min(best_single.get(ds, fit), fit)

    rows = []
    for r in records:
        ds = dataset_of(r["name"])
        workers = r.get("workers") or 0
        per_worker = r.get("worker_threads") or r.get("threads")
        total_threads = (workers * per_worker) if r["mode"] == "dist" else r.get("threads")
        quality = r.get("quality") or {}
        ok = r.get("exit_code", 0) == 0
        fit = r.get("fit_wall_s")
        ratio = None
        if ok and fit and ds in best_single:
            ratio = fit / best_single[ds]
        rows.append({
            "name": r["name"],
            "dataset": ds,
            "mode": r["mode"],
            "workers": workers,
            "threads_per_worker": per_worker,
            "total_threads": total_threads,
            "iterations": r.get("iterations"),
            "train_rows": r.get("train_rows"),
            "fit_s": round(fit, 3) if ok and fit else "",
            "startup_s": round(r["startup_s"], 3) if r.get("startup_s") is not None else "",
            "total_s": round(r["total_wall_s"], 3) if ok and r.get("total_wall_s") else "",
            "logloss": round(quality["Logloss"], 6) if quality.get("Logloss") is not None else "",
            "auc": round(quality["AUC"], 6) if quality.get("AUC") is not None else "",
            "exit_code": r.get("exit_code"),
            "fit_vs_best_single": round(ratio, 2) if ratio else "",
        })

    out = RUN_DIR / "summary.tsv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
