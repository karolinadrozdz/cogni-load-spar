"""Paper-style thought-rejection figure for completed K2-Horizon checkpoints."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Completed reasoning checkpoints, in training order. SFT values are the n=1,000
# Stage-B runs; Main is the completed n=200 Stage-A run.
labels = ["SFT1 1k\n(n=1,000)", "SFT1 3k\n(n=1,000)", "Main\n(n=200)"]
after_wrong = np.array([0.528, 0.625, 0.4285714286])
after_right = np.array([0.285, 0.290, 0.1986754967])
overall = np.array([0.358, 0.385, 0.255])
selectivity = after_wrong - after_right
x = np.arange(len(labels))

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    }
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.35), gridspec_kw={"width_ratios": [1.15, 1]})
orange = "#e66b2d"
blue = "#3d7edb"

# Left: conditional and overall abstention.
ax1.fill_between(x, after_right, after_wrong, color="0.91", zorder=0)
ax1.plot(x, after_wrong, color=orange, marker="o", markersize=7, linewidth=2.4, label="after a wrong answer")
ax1.plot(x, after_right, color=blue, marker="s", markersize=7, linewidth=2.4, label="after a right answer")
ax1.plot(x, overall, color="0.25", linestyle=":", linewidth=2.0, label="overall")
ax1.set_title("Thought rejection across completed checkpoints")
ax1.set_ylabel(r"$P(\mathrm{abstain})$")
ax1.set_xticks(x, labels)
ax1.set_xlim(-0.16, 2.16)
ax1.set_ylim(0, 0.72)
ax1.set_yticks(np.arange(0, 0.71, 0.1))
ax1.grid(axis="y", color="0.88", linewidth=0.7)
ax1.legend(frameon=False, loc="upper right", fontsize=8.6, handlelength=2.5)
for xi, yw, yr in zip(x, after_wrong, after_right):
    ax1.text(xi, yw + 0.022, f"{yw:.3f}", color=orange, ha="center", fontsize=8.3)
    ax1.text(xi, yr - 0.042, f"{yr:.3f}", color=blue, ha="center", fontsize=8.3)

# Right: error selectivity gap.
ax2.plot(x, selectivity, color="black", marker="o", markersize=7, linewidth=2.4)
ax2.fill_between(x, 0, selectivity, color="0.94", zorder=0)
ax2.axhline(0, color="0.55", linewidth=0.8)
ax2.set_title("Error selectivity peaks at SFT1 3k")
ax2.set_ylabel(r"$\Delta$ selectivity")
ax2.set_xticks(x, labels)
ax2.set_xlim(-0.16, 2.16)
ax2.set_ylim(0, 0.40)
ax2.set_yticks(np.arange(0, 0.41, 0.05))
ax2.grid(axis="y", color="0.88", linewidth=0.7)
ax2.text(
    0.5,
    0.372,
    r"$\Delta=P(\mathrm{abstain}\mid\mathrm{wrong})-P(\mathrm{abstain}\mid\mathrm{right})$",
    ha="center",
    va="center",
    fontsize=8.3,
    color="0.25",
)
for xi, value in zip(x, selectivity):
    ax2.text(xi, value + 0.014, f"{value:.3f}", ha="center", fontsize=8.5)

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(width=0.8)

fig.suptitle("K2-Horizon-32B Thought Rejection Across Training", fontsize=14, fontweight="bold", y=1.01)
fig.text(
    0.5,
    -0.035,
    "PopQA. Shading marks the gap between abstention after wrong and right Phase 1 answers; sample sizes are shown on the x-axis.",
    ha="center",
    fontsize=8.3,
    color="0.35",
)
fig.tight_layout(w_pad=3.0)

png = OUT / "thought_rejection_completed_checkpoints.png"
pdf = OUT / "thought_rejection_completed_checkpoints.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
print(png)
print(pdf)
