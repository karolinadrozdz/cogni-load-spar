"""Spec item 5: checkpoint-probe trajectories by trace label (Phase 2, sets A+B)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "results/80b-thinking")
BINS = [0, .25, .5, .75, 1.0001]


def per_trace(g):
    g = g.sort_values("token_pos")
    fa, final = g.forced_answer.values, g.final_answer.iloc[0]
    off = np.where(fa != final)[0]
    commit = 0.0 if len(off) == 0 else g.frac.values[min(off[-1] + 1, len(g) - 1)]
    mid = g[g.kind.isin(["fixed", "marker"])]
    # longest stretch (in trace fraction) the forced answer sat on one non-final option
    runs, start = [], 0
    for i in range(1, len(g) + 1):
        if i == len(g) or fa[i] != fa[start]:
            if fa[start] != final:
                runs.append(g.frac.values[min(i, len(g) - 1)] - g.frac.values[start])
            start = i
    return pd.Series({
        "start_forced_5": int(fa[0] == 5), "start_prob_5": g.prob_5.iloc[0],
        "commit_frac": commit, "n_changes": int((fa[1:] != fa[:-1]).sum()),
        "mean_prob_forced": mid.prob_forced.mean(), "min_prob_forced": mid.prob_forced.min(),
        "longest_nonfinal_stretch": max(runs, default=0.0),
        **{f"prob_final_q{i}": g[(g.frac >= BINS[i]) & (g.frac < BINS[i + 1])].prob_final.mean() for i in range(4)},
    })


def main():
    ck = pd.read_csv(RUN / "checkpoints.csv").query("phase == 2")
    lab = pd.read_csv(RUN / "trace_labels.csv", index_col=0).query("set in ['A', 'B']")
    traj = ck.groupby("question_index").apply(per_trace)
    df = lab.join(traj, on="question_index", how="inner")
    df["abstained"] = (df.set == "A").astype(int)

    cols = ["start_forced_5", "start_prob_5", "commit_frac", "n_changes", "mean_prob_forced",
            "min_prob_forced", "longest_nonfinal_stretch"] + [f"prob_final_q{i}" for i in range(4)]
    out = {"n": int(len(df)), "by_set_kind": {}}
    for (s, k), g in df.groupby(["set", "kind"]):
        out["by_set_kind"][f"{s}_{k}"] = {"n": int(len(g)), **{c: round(float(g[c].median() if c in
            ("commit_frac", "start_prob_5", "longest_nonfinal_stretch") else g[c].mean()), 3) for c in cols}}
    # label-probe conflicts: JUDGE abstentions that sat on one listed option for >= half the trace
    # (looks like a held candidate) and RESOLVE abstentions forced to 5 throughout with no change
    a = df[df.set == "A"]
    j_conf = a[(a.kind == "JUDGE") & (a.longest_nonfinal_stretch >= .5)]
    r_conf = a[(a.kind == "RESOLVE") & (a.n_changes == 0)]
    out["conflicts"] = {
        "JUDGE_abstention_held_option_>=half_trace": {"n": int(len(j_conf)), "of": int((a.kind == "JUDGE").sum()),
                                                       "ids": j_conf.index.tolist()},
        "RESOLVE_abstention_forced_5_throughout": {"n": int(len(r_conf)), "of": int((a.kind == "RESOLVE").sum()),
                                                   "ids": r_conf.index.tolist()},
    }
    df.to_csv(RUN / "trace_labels_checkpoints.csv")
    (RUN / "trace_labels_crosscheck.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
