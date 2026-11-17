#!/usr/bin/env python3
"""Render consultation figures from the frozen quantile summary JSON."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/stage3/distributed_ctr_quantiles_20261107"
FIG = ROOT / "figures/milestones"
SUMMARY = json.loads((RUN / "summary.json").read_text())
CELLS = SUMMARY["cells"]
LABELS = {
    "amazon_ctr_1w": "Amazon CTR · 1 worker",
    "amazon_ctr_2w": "Amazon CTR · 2 workers",
    "amazon_ctr_4w": "Amazon CTR · 4 workers",
    "criteo_ctr_2w": "Criteo CTR · 2 workers",
    "amazon_noctr_2w": "Amazon без CTR · 2 workers",
}
COLORS = {
    "amazon_ctr_1w": "#1b6f50",
    "amazon_ctr_2w": "#269d75",
    "amazon_ctr_4w": "#72b89b",
    "criteo_ctr_2w": "#c66a23",
    "amazon_noctr_2w": "#747b80",
}


def main():
    if len(CELLS) != 10 or {(c["model"], c["batch"]) for c in CELLS} != {
        (model, batch) for model in LABELS for batch in (128, 4096)
    }:
        raise ValueError("summary does not contain the complete frozen matrix")
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), sharey=True)
    order = list(LABELS)
    for ax, batch in zip(axes, (128, 4096)):
        by_model = {c["model"]: c for c in CELLS if c["batch"] == batch}
        for y, model in enumerate(order):
            effect = by_model[model]["quantiles"]["p95"]
            x = effect["paired_median_change_pct"]
            lo, hi = effect["paired_ci95_pct"]
            ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt="o", capsize=4,
                        color=COLORS[model], markersize=7, linewidth=2)
            ax.annotate(f"{x:+.1f}%", (x, y), xytext=(0, -15),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.axvline(0, color="#333333", lw=1, ls="--")
        ax.set_title(f"Батч {batch} строк")
        ax.set_xlabel("Изменение p95, % · меньше = быстрее")
        ax.set_yticks(range(len(order)), [LABELS[x] for x in order])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.17)
        ax.set_xlim(-37, 10)
    fig.suptitle("Эффект CTR-оптимизации на p95 задержки", fontsize=15)
    fig.text(.5, .02, "Медиана 9 парных сессий; отрезки — 95% bootstrap CI. Готовый Pool, 2 потока.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(FIG / "ctr_p95_distributed_models.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), sharey=True)
    qnames = ("p50", "p90", "p95", "p99")
    xs = np.arange(len(qnames))
    for ax, batch in zip(axes, (128, 4096)):
        by_model = {c["model"]: c for c in CELLS if c["batch"] == batch}
        for model in order:
            effects = [by_model[model]["quantiles"][q] for q in qnames]
            ys = [e["paired_median_change_pct"] for e in effects]
            ax.plot(xs, ys, marker="o", lw=1.8, color=COLORS[model],
                    label=LABELS[model])
            ax.scatter([2], [ys[2]], color=COLORS[model], s=90, zorder=4)
        ax.axhline(0, color="#333333", lw=1, ls="--")
        ax.set_xticks(xs, qnames)
        ax.set_title(f"Батч {batch} строк")
        ax.set_xlabel("Квантиль задержки")
        ax.grid(axis="y", alpha=.17)
    axes[0].set_ylabel("Медиана парного изменения, %")
    fig.suptitle("Эффект оптимизации по квантилям задержки", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(.5, -.01))
    fig.text(.5, .095, "p99 — исследовательская метрика: около 10 верхних вызовов в каждой сессии.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .15, 1, .94))
    fig.savefig(FIG / "ctr_quantile_effects_distributed_models.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    training = [json.loads((ROOT / f"results/stage2/amazon_ctr_spark_mapped_20261107/{w}w/result.json").read_text())
                for w in (1, 2, 4)]
    if [x["executor_count_observed"] for x in training] != [1, 2, 4]:
        raise ValueError("worker counts differ from milestone matrix")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(["1", "2", "4"], [x["fit_seconds"] for x in training],
                  color=[COLORS[f"amazon_ctr_{w}w"] for w in (1, 2, 4)])
    ax.bar_label(bars, labels=[f"{x['fit_seconds']:.1f} с" for x in training], padding=4)
    ax.set_ylim(0, 27)
    ax.set_xlabel("Число worker-процессов")
    ax.set_ylabel("Время fit, с")
    ax.set_title("Распределённое обучение Amazon с CTR")
    ax.text(.5, -.2, "Один запуск на конфигурацию, 24 577 строк, 20 деревьев",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555555")
    fig.tight_layout()
    fig.savefig(FIG / "amazon_ctr_training_workers.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    print("figures written to", FIG)


if __name__ == "__main__":
    main()
