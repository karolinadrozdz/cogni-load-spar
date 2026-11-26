"""Checkpoint line figure linking abstention selectivity to confidence readouts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
FIGURES = ROOT / "figures"
CHECKPOINTS = [
    ("think-sft_1_1000", "1k"),
    ("think-sft_1_3000", "3k"),
    ("think-sft_1_6000", "6k"),
    ("think-sft_1_11000", "11k"),
    ("think-sft_2_1500", "1.5k"),
    ("think-sft_2_2500", "2.5k"),
]


def main():
    records = []
    for tag, label in CHECKPOINTS:
        path = RUNS / f"k2h32b-popqa-{tag}" / "analysis.json"
        data = json.loads(path.read_text())
        abstain = data["P2_abstention"]
        auroc = data["P1_auroc_correct"]
        records.append({
            "label": label,
            "wrong": abstain["abstention_given_p1_incorrect"],
            "right": abstain["abstention_given_p1_correct"],
            "overall": abstain["abstentions"] / abstain["n"],
            "verbal_auc": auroc["verbal"],
            "cal_auc": auroc["cal"],
            "n": abstain["n"],
        })

    x = np.arange(len(records), dtype=float)
    labels = [f"SFT 1\n{r['label']}" if i < 4 else f"SFT 2\n{r['label']}"
              for i, r in enumerate(records)]
    wrong = np.array([r["wrong"] for r in records])
    right = np.array([r["right"] for r in records])
    overall = np.array([r["overall"] for r in records])
    gap = wrong - right
    verbal_auc = np.array([r["verbal_auc"] for r in records])
    cal_auc = np.array([r["cal_auc"] for r in records])

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Georgia", "DejaVu Serif"],
        "font.size": 10.5,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5))
    fig.subplots_adjust(left=.055, right=.985, top=.77, bottom=.25, wspace=.27)
    fig.text(.055, .94, "K2-Horizon-32B: Confidence and selective abstention across training",
             fontsize=19, weight="bold")
    fig.text(.055, .885, "PopQA  |  native thinking  |  1,000 questions per checkpoint  |  temperature 0.6",
             fontsize=11, color="#444444")
    orange, blue, gray = "#bf531f", "#2b64a6", "#555555"

    ax = axes[0]
    ax.fill_between(x, right, wrong, color="#eeeeee", zorder=0)
    ax.plot(x, wrong, "o-", color=orange, lw=2.2, ms=6, label="After wrong answer")
    ax.plot(x, right, "s-", color=blue, lw=2.2, ms=5.5, label="After right answer")
    ax.plot(x, overall, "D:", color=gray, lw=1.7, ms=4.5, label="Overall")
    ax.set_title("A  Phase 2 abstention", loc="left", pad=12)
    ax.set_ylabel("Abstention rate")
    ax.set_ylim(0, .72)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax = axes[1]
    ax.plot(x, gap, "o-", color="#161616", lw=2.3, ms=6)
    ax.axhline(0, color="#777777", lw=.8)
    for xi, yi in zip(x, gap):
        ax.annotate(f"{yi * 100:.1f}", (xi, yi), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_title("B  Error-selective abstention", loc="left", pad=12)
    ax.set_ylabel("Δ (percentage points)")
    ax.set_ylim(0, .40)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol=""))

    ax = axes[2]
    ax.plot(x, verbal_auc, "o-", color=orange, lw=2.2, ms=6,
            label="Verbal confidence")
    ax.plot(x, cal_auc, "s-", color=blue, lw=2.2, ms=5.5,
            label="Calibrated confidence")
    ax.set_title("C  Correctness discrimination", loc="left", pad=12)
    ax.set_ylabel("Phase 1 AUROC")
    ax.set_ylim(.48, .85)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlim(-.25, len(x) - .75)
        ax.axvline(3.5, color="#aaa", lw=.9, linestyle="--")
        ax.grid(axis="y", color="#e3e3e3", lw=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=8)

    fig.text(.055, .075,
             "Δ = P(abstain | wrong Phase 1 answer) − P(abstain | right Phase 1 answer). "
             "Conditional rates use items with valid verbal confidence.",
             fontsize=9, color="#444444")
    fig.text(.055, .037,
             "SFT 2 step 0.5k is pending. Main uses the same weights as SFT 2 step 2.5k and is omitted from the training line.",
             fontsize=9, color="#444444")

    FIGURES.mkdir(exist_ok=True)
    stem = FIGURES / "k2h_stage_b_latest_checkpoint_lines"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    plt.close(fig)
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
