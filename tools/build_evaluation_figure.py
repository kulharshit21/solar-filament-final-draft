"""Generate report figures from the exact verified 0.40 submission metadata."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = Path(r"D:\SOLAR GPT\Submission_040_Recreated_20261111\submission.csv")
OUTPUT = ROOT / "reports" / "figures" / "submission_distribution.png"


def main() -> None:
    counts: Counter[str] = Counter()
    with SUBMISSION.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts[row["filament_id"].rsplit("_", 1)[0]] += 1

    values = sorted(counts.values())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    axes[0].hist(values, bins=range(1, max(values) + 2), color="#2A6F97", edgecolor="white")
    axes[0].set_xlabel("Predicted instances per image")
    axes[0].set_ylabel("Number of test images")
    axes[0].set_title("Instance-count distribution")
    axes[0].set_xticks(range(1, max(values) + 1, 2))

    labels = ["PQ", "SQ", "RQ"]
    scores = [0.4622, 0.6857, 0.6741]
    bars = axes[1].bar(labels, scores, color=["#D1495B", "#2A6F97", "#4C956C"])
    axes[1].set_ylim(0, 0.8)
    axes[1].set_ylabel("Out-of-fold score")
    axes[1].set_title("Calibrated OOF metrics")
    for bar, score in zip(bars, scores):
        axes[1].text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.4f}", ha="center", fontsize=9)

    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
