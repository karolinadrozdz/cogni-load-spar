"""Paper-style overview of every completed K2-Horizon training checkpoint."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "figures"

BASE = [
    (f"base-pretrain_{step}", f"{step // 1000}k")
    for step in range(100000, 1100001, 100000)
] + [
    ("mid-mid_1_55000", "M1\n55k"),
    ("mid-mid_2_25000", "M2\n25k"),
    ("mid-mid_3_5500", "M3\n5.5k"),
    ("mid-mid_4_3000", "M4\n3k"),
    ("mid-mid_4_6000", "M4\n6k"),
    ("mid-mid_4_10000", "M4\n10k"),
    ("bridge-sft_1_1000", "SFT1\n1k"),
    ("bridge-sft_1_11000", "SFT1\n11k"),
    ("bridge-main", "SFT2\nfinal"),
]
THINK = [
    ("think-sft_1_1000", "SFT1\n1k"),
    ("think-sft_1_3000", "SFT1\n3k"),
    ("think-sft_1_6000", "SFT1\n6k"),
    ("think-sft_1_11000", "SFT1\n11k"),
    ("think-sft_2_500", "SFT2\n0.5k"),
    ("think-sft_2_1500", "SFT2\n1.5k"),
    ("think-sft_2_2500", "SFT2\n2.5k"),
]


def completed(entries):
    found = []
    for tag, label in entries:
        path = RUNS / f"k2h32b-popqa-{tag}" / "analysis.json"
        if path.exists():
            found.append((tag, label, json.loads(path.read_text())))
    return found


def main():
    base, think = completed(BASE), completed(THINK)
    x_base = np.arange(len(base))
    x_think = np.arange(len(think))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Georgia", "DejaVu Serif"],
        "font.size": 10.2,
        "pdf.fonttype": 42,
        "axes.linewidth": .8,
    })
    fig = plt.figure(figsize=(17.2, 9.5))
    grid = fig.add_gridspec(2, 2, left=.055, right=.975, top=.82, bottom=.13,
                           hspace=.68, wspace=.24, height_ratios=[1, 1])
    ax_base = fig.add_subplot(grid[0, :])
    ax_auc = fig.add_subplot(grid[1, 0])
    ax_abstain = fig.add_subplot(grid[1, 1])
    orange, blue, dark = "#c45523", "#2d67a7", "#272727"

    fig.text(.055, .945, "K2-Horizon-32B across completed training checkpoints",
             fontsize=20, weight="bold")
    fig.text(.055, .895,
             f"PopQA  |  {len(base)} base/bridge and {len(think)} native-thinking checkpoints  |  "
             "1,000 Phase 1 questions per checkpoint",
             fontsize=11, color="#444444")

    accuracy = np.array([d["phase1"]["accuracy"] for _, _, d in base])
    cal_auc = np.array([d["P1_auroc_correct"]["cal"] for _, _, d in base])
    ax_base.plot(x_base, accuracy, "o-", color=orange, ms=4.6, lw=1.9,
                 label="Phase 1 accuracy")
    ax_base.plot(x_base, cal_auc, "s-", color=blue, ms=4.2, lw=1.9,
                 label="Calibrated-confidence AUROC")
    ax_base.set_title("A  Base and bridge checkpoints: answer quality", loc="left", pad=12)
    ax_base.set_ylabel("Accuracy / AUROC")
    ax_base.set_ylim(.55, .91)
    ax_base.set_xticks(x_base, [label for _, label, _ in base], rotation=45,
                       ha="right", fontsize=8.2)
    ax_base.legend(frameon=False, ncol=2, loc="upper left", fontsize=9)
    # Stage boundaries are derived from the completed checkpoint labels.
    groups = [tag.split("-")[0] for tag, _, _ in base]
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax_base.axvline(i - .5, color="#aaa", linestyle="--", lw=.8)

    native_accuracy = np.array([d["phase1"]["accuracy"] for _, _, d in think])
    native_cal = np.array([d["P1_auroc_correct"]["cal"] for _, _, d in think])
    native_verbal = np.array([d["P1_auroc_correct"]["verbal"] for _, _, d in think])
    ax_auc.plot(x_think, native_accuracy, "D:", color=dark, ms=4.8, lw=1.7,
                label="Phase 1 accuracy")
    ax_auc.plot(x_think, native_verbal, "o-", color=orange, ms=5.5, lw=2,
                label="Verbal-confidence AUROC")
    ax_auc.plot(x_think, native_cal, "s-", color=blue, ms=5, lw=2,
                label="Calibrated-confidence AUROC")
    ax_auc.set_title("B  Native thinking: correctness discrimination", loc="left", pad=12)
    ax_auc.set_ylabel("Accuracy / AUROC")
    ax_auc.set_ylim(.49, .82)
    ax_auc.legend(frameon=False, loc="lower left", fontsize=8.3)

    wrong = np.array([d["P2_abstention"]["abstention_given_p1_incorrect"]
                      for _, _, d in think])
    right = np.array([d["P2_abstention"]["abstention_given_p1_correct"]
                      for _, _, d in think])
    delta = wrong - right
    ax_abstain.fill_between(x_think, right, wrong, color="#eeeeee", zorder=0)
    ax_abstain.plot(x_think, wrong, "o-", color=orange, ms=5.5, lw=2,
                    label="After wrong answer")
    ax_abstain.plot(x_think, right, "s-", color=blue, ms=5, lw=2,
                    label="After right answer")
    ax_abstain.plot(x_think, delta, "D:", color=dark, ms=4.8, lw=1.7,
                    label="Selectivity Δ")
    ax_abstain.set_title("C  Native thinking: selective abstention", loc="left", pad=12)
    ax_abstain.set_ylabel("Abstention rate / Δ")
    ax_abstain.set_ylim(0, .7)
    ax_abstain.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax_abstain.legend(frameon=False, fontsize=8.3, loc="upper right")

    native_labels = [label for _, label, _ in think]
    for ax in (ax_auc, ax_abstain):
        ax.set_xticks(x_think, native_labels, fontsize=8.5)
        ax.axvline(3.5, color="#aaa", linestyle="--", lw=.8)

    for ax in (ax_base, ax_auc, ax_abstain):
        ax.set_xlim(-.35, len(ax.get_xticks()) - .65)
        ax.grid(axis="y", color="#e2e2e2", lw=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=7)

    fig.text(.055, .06,
             "Only completed checkpoints are plotted; unmeasured checkpoints have no reserved x-axis positions. "
             "Δ = P(abstain | wrong) − P(abstain | right).",
             fontsize=9, color="#444444")
    fig.text(.055, .033,
             "The 32B model has pretraining, midtraining and SFT stages, with no RL stage. "
             "Main uses the SFT2 2.5k weights; its T=0.6 and T=1.0 reruns are excluded from the training trajectory.",
             fontsize=9, color="#444444")

    OUT.mkdir(exist_ok=True)
    stem = OUT / "k2h_overall_completed_stages"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=230, facecolor="white")
    plt.close(fig)
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
