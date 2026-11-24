#!/usr/bin/env python3
"""Create the Stage A summary figure from the completed n=200 results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

labels = ["Pretrain\n100k", "Pretrain\n1,100k", "Mid 4\n10k", "Think SFT\n1k", "Main"]
x = np.arange(len(labels))

accuracy = np.array([0.655, 0.685, 0.745, 0.705, 0.755])
cal_auc = np.array([0.740, 0.854, 0.845, 0.723, 0.550])
verbal_auc = np.array([np.nan, np.nan, np.nan, 0.759, 0.669])

reason_labels = ["Think SFT 1k", "Main"]
abstention = np.array([0.335, 0.255])
lr_verbal = np.array([14.06, 19.61])
lr_cal = np.array([2.54, 1.59])

plt.rcParams.update(
    {
        "font.family": "Georgia",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)

# A — accuracy across training stages
ax = axes[0, 0]
ax.plot(x, accuracy, color="black", marker="o", linewidth=1.6, markersize=5)
for xi, yi in zip(x, accuracy):
    ax.text(xi, yi + 0.009, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)
ax.axvline(2.5, color="0.75", linewidth=0.8, linestyle="--")
ax.text(1.0, 0.625, "Base / mid", ha="center", color="0.35", fontsize=8)
ax.text(3.5, 0.625, "Reasoning", ha="center", color="0.35", fontsize=8)
ax.set_title("A   Answer accuracy")
ax.set_ylabel("Phase 1 accuracy")
ax.set_xticks(x, labels)
ax.set_ylim(0.61, 0.78)
ax.grid(axis="y", color="0.9", linewidth=0.7)

# B — correctness discrimination
ax = axes[0, 1]
ax.plot(x, cal_auc, color="black", marker="o", linewidth=1.6, markersize=5, label="Calibrated confidence")
ax.plot(x, verbal_auc, color="0.45", marker="s", linewidth=1.6, markersize=5, linestyle="--", label="Verbal confidence")
for xi, yi in zip(x, cal_auc):
    ax.text(xi, yi + 0.012, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)
for xi, yi in zip(x, verbal_auc):
    if np.isfinite(yi):
        ax.text(xi, yi - 0.018, f"{yi:.3f}", ha="center", va="top", fontsize=8, color="0.35")
ax.axhline(0.5, color="0.65", linewidth=0.9, linestyle=":")
ax.axvline(2.5, color="0.75", linewidth=0.8, linestyle="--")
ax.set_title("B   Correctness discrimination")
ax.set_ylabel("AUROC")
ax.set_xticks(x, labels)
ax.set_ylim(0.48, 0.90)
ax.legend(frameon=False, loc="lower left", fontsize=8)
ax.grid(axis="y", color="0.9", linewidth=0.7)

# C — abstention in reasoning checkpoints
ax = axes[1, 0]
bars = ax.bar(reason_labels, abstention * 100, width=0.55, color=["0.25", "0.65"], edgecolor="black", linewidth=0.8)
for bar, value in zip(bars, abstention):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{value:.1%}", ha="center", fontsize=9)
ax.set_title("C   Thought rejection")
ax.set_ylabel("Phase 2 abstention (%)")
ax.set_ylim(0, 42)
ax.grid(axis="y", color="0.9", linewidth=0.7)

# D — incremental prediction of abstention
ax = axes[1, 1]
width = 0.34
xr = np.arange(len(reason_labels))
bars1 = ax.bar(xr - width / 2, lr_verbal, width, color="0.2", edgecolor="black", linewidth=0.7, label="Verbal beyond calibrated")
bars2 = ax.bar(xr + width / 2, lr_cal, width, color="white", edgecolor="black", linewidth=0.9, hatch="///", label="Calibrated beyond verbal")
ax.axhline(3.84, color="0.45", linewidth=1.0, linestyle="--", label="p = .05 threshold")
for bars in (bars1, bars2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.55, f"{bar.get_height():.2f}", ha="center", fontsize=8)
ax.set_title("D   Incremental prediction of abstention")
ax.set_ylabel("Likelihood-ratio statistic")
ax.set_xticks(xr, reason_labels)
ax.set_ylim(0, 23)
ax.legend(frameon=False, fontsize=7.6, loc="upper left")
ax.grid(axis="y", color="0.9", linewidth=0.7)

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("K2-Horizon Stage A (n = 200 per checkpoint)", fontsize=13, fontweight="bold")
fig.text(
    0.5,
    -0.02,
    "All checkpoints use the same 32B architecture. Base checkpoints have no verbal-confidence or Phase 2 readout. "
    "Dashed vertical lines mark the transition to reasoning checkpoints.",
    ha="center",
    fontsize=8,
    color="0.35",
)

png = OUT / "stage_a_summary.png"
pdf = OUT / "stage_a_summary.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
print(png)
print(pdf)
