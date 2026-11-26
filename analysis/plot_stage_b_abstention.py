"""Plot Stage B abstention and selectivity from the archived analysis files."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
CHECKPOINTS = [
    ("think-sft_1_1000", "SFT 1\n1k"),
    ("think-sft_1_3000", "SFT 1\n3k"),
    ("think-sft_1_6000", "SFT 1\n6k"),
    ("think-sft_1_11000", "SFT 1\n11k"),
    ("think-sft_2_1500", "SFT 2\n1.5k"),
    ("think-sft_2_2500", "SFT 2\n2.5k"),
]


def main():
    metrics = np.full((len(CHECKPOINTS), 3), np.nan)
    for i, (tag, _) in enumerate(CHECKPOINTS):
        path = RUNS / f"k2h32b-popqa-{tag}" / "analysis.json"
        if path.is_file():
            result = json.loads(path.read_text())["P2_abstention"]
            metrics[i] = (
                result["abstention_given_p1_incorrect"],
                result["abstention_given_p1_correct"],
                result["abstentions"] / result["n"],
            )

    wrong, right, overall = metrics.T
    gap = wrong - right
    x = np.arange(len(CHECKPOINTS))
    labels = [label for _, label in CHECKPOINTS]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Georgia", "DejaVu Serif"],
        "font.size": 11,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, (left, right_ax) = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.subplots_adjust(left=.065, right=.98, bottom=.25, top=.78, wspace=.26)
    fig.text(.065, .94, "K2-Horizon-32B: Abstention across checkpoints", fontsize=21, weight="bold")
    fig.text(.065, .888, "PopQA · Stage B · 1,000 questions per run", fontsize=12, color="#4a4a4a")

    left.fill_between(x, right, wrong, color="#ededed")
    for values, color, marker, style, label in [
        (wrong, "#c55426", "o", "-", "After an incorrect answer"),
        (right, "#2862a5", "s", "-", "After a correct answer"),
        (overall, "#4f4f4f", "D", ":", "Overall"),
    ]:
        left.plot(x, values, style, color=color, marker=marker, lw=2.2, ms=5, label=label)
    left.set_title("A   Phase 2 abstention rates", loc="left", pad=13)
    left.set_ylabel("Abstention rate")
    left.set_ylim(0, .79)
    left.legend(frameon=False, loc="upper center", ncols=2, fontsize=9, columnspacing=1.05)

    right_ax.plot(x, gap, "o-", color="#181818", lw=2.2, ms=5)
    for xi, value in zip(x, gap):
        if np.isfinite(value):
            right_ax.annotate(f"{value * 100:.1f}", (xi, value), xytext=(0, 8),
                              textcoords="offset points", ha="center", fontsize=9)
    right_ax.set_title("B   Abstention selectivity", loc="left", pad=13)
    right_ax.set_ylabel("Δ (percentage points)")
    right_ax.set_ylim(0, .40)

    for ax in (left, right_ax):
        ax.set_xticks(x, labels, fontsize=8.7)
        ax.set_xlim(-.35, len(x) - .65)
        ax.axvline(3.5, color="#b6b6b6", linewidth=.9, linestyle="--")
        ax.grid(axis="y", color="#e0e0e0", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=8)
    left.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    right_ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol=""))

    fig.text(.065, .095, "Δ = abstention after an incorrect Phase 1 answer minus abstention after a correct one. "
             "Rates use items with valid verbal confidence.", fontsize=9.5, color="#444444")
    fig.text(.065, .055, "All training checkpoints use T = 0.6. Main uses the SFT 2 step 2.5k weights, "
             "so its reruns are excluded from the training line.", fontsize=9.5, color="#444444")
    output = ROOT / "figures" / "k2h_stage_b_abstention_latest"
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=240, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
