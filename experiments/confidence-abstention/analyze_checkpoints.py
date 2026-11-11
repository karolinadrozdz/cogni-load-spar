"""Analysis of in-trace confidence checkpoints written by `abstention_modal.py::checkpoints`.

    python analyze_checkpoints.py results/<tag>   # prints a summary, writes <tag>/checkpoints_analysis.json

Each checkpoint is the answer the model gives if the think block is force-closed after the first
`frac` of its trace. `forced_answer` / `prob_forced` are that answer and its probability (softmax over
the valid digits). Only complete (non-truncated) traces are probed. AUROCs are raw (feature higher ->
outcome 1), so a value below 0.5 means the feature runs the other way. Needs pandas, numpy, scikit-learn.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

d = Path(sys.argv[1])
df = pd.read_csv(d / "checkpoints.csv").sort_values(["phase", "question_index", "checkpoint_idx"])
df["matches_final"] = df.matches_final.astype(int)
TRACE = ["phase", "question_index"]
OUTCOME = {1: "is_correct", 2: "is_nota"}
report = {"run": d.name, "n_traces": {int(p): int(g.question_index.nunique()) for p, g in df.groupby("phase")},
          "n_checkpoints": {k: int(c) for k, c in df.kind.value_counts().items()}}


def quart(x):
    x = pd.Series(x).dropna()
    if x.empty:
        return {"n": 0}
    q = x.quantile([0.25, 0.5, 0.75]).round(3)
    return {"n": len(x), "q25": q[0.25], "median": q[0.5], "q75": q[0.75]}


def auroc(y, x):
    y, x = np.asarray(y), np.asarray(x, dtype=float)
    return round(roc_auc_score(y, x), 3) if len(set(y)) == 2 else None  # both classes needed


def commit_frac(g):
    """Earliest frac from which forced_answer equals final_answer at every later checkpoint (NaN if never)."""
    m = g.matches_final.to_numpy()
    if not m[-1]:
        return np.nan
    i = len(m) - 1
    while i > 0 and m[i - 1]:
        i -= 1
    return g.frac.to_numpy()[i]


def per_phase(fn):
    return {"all": fn(df), **{f"phase{p}": fn(g) for p, g in df.groupby("phase")}}


# per-trace table
tr = df.groupby(TRACE).apply(lambda g: pd.Series(dict(
    commit=commit_frac(g), min_prob=g.prob_forced.min(), mean_prob=g.prob_forced.mean(),
    prob_at_start=g.prob_forced.iloc[0], n_answer_changes=int((g.forced_answer != g.forced_answer.shift()).iloc[1:].sum()),
    prob_final_end=g.prob_final.iloc[-1], is_correct=g.is_correct.iloc[0], is_nota=g.is_nota.iloc[0])),
    include_groups=False).reset_index()

# sanity: forced answer with the whole trace should reproduce the run's answer
end = df[df.kind == "end"]
report["sanity"] = {f"phase{p}": {"end_matches_final": round(g.matches_final.mean(), 3), "n": len(g),
                                  "flag": "LOW (< 0.95): forced close is not reproducing the run's answers"
                                  if g.matches_final.mean() < 0.95 else "ok"} for p, g in end.groupby("phase")}

# commitment
report["commitment"] = {"note": "frac from which forced_answer == final_answer at every later checkpoint"}
for p, g in tr.groupby("phase"):
    by = OUTCOME[p]
    report["commitment"][f"phase{p}"] = {"all": quart(g.commit), "never_committed": int(g.commit.isna().sum()),
                                         **{f"{by}={int(k)}": quart(h.commit) for k, h in g.groupby(by)}}

# does confidence differ just before a backtrack marker vs at an arbitrary (fixed) point?
df["prev_prob"] = df.groupby(TRACE).prob_forced.shift()
df["change"] = df.prob_forced - df.prev_prob
BINS = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"]
df["bin"] = pd.cut(df.frac, [0, 0.25, 0.5, 0.75, 1.0], labels=BINS, include_lowest=True)


def backtrack(x):
    mk, fx = x[x.kind == "marker"], x[x.kind == "fixed"]
    r = {"mean_prob_marker": round(mk.prob_forced.mean(), 3), "mean_prob_fixed": round(fx.prob_forced.mean(), 3),
         "n_marker": len(mk), "n_fixed": len(fx),
         "mean_change_to_marker": round(mk.change.mean(), 3), "mean_change_to_fixed": round(fx.change.mean(), 3),
         "by_frac": {}}
    for b in BINS:
        m, f = mk[mk.bin == b], fx[fx.bin == b]
        r["by_frac"][b] = {"marker": round(m.prob_forced.mean(), 3), "n_marker": len(m),
                           "fixed": round(f.prob_forced.mean(), 3), "n_fixed": len(f)}
    return r


report["before_backtrack"] = per_phase(backtrack)

# does the shape of the trajectory predict the outcome, beyond the post-trace answer probability?
traj = {}
for p, g in tr.groupby("phase"):
    g = g.assign(frac_committed=g.commit.fillna(1.0))
    y = g[OUTCOME[p]]
    feats = ["min_prob", "mean_prob", "prob_at_start", "n_answer_changes", "frac_committed"]
    traj[f"phase{p}"] = {"outcome": OUTCOME[p], "n": len(g), "rate": round(y.mean(), 3),
                         "auroc": {f: auroc(y, g[f]) for f in feats},
                         "baseline_post_trace_prob_final": auroc(y, g.prob_final_end)}
report["trajectory_predicts_outcome"] = traj

# phase 2: when, along the trace, does the model first lean towards abstaining?
p2 = df[(df.phase == 2) & (df.is_nota == 1)]
onset = p2[p2.forced_answer == 5].groupby("question_index").frac.min()
start = p2[p2.kind == "start"].set_index("question_index").forced_answer
report["abstention_onset"] = {"n_abstained_traces": int(p2.question_index.nunique()),
                              "never_forced_5": int(p2.question_index.nunique() - len(onset)),
                              "onset_frac": quart(onset),
                              "frac_with_answer_1_to_4_at_start": round(float((start != 5).mean()), 3) if len(start) else None}


def sustained_onset(g):
    """frac of the first checkpoint with forced 5 followed only by 5s (NaN if the trace does not end on 5)."""
    five = (g.forced_answer == 5).to_numpy()
    if not five[-1]:
        return np.nan
    i = len(five) - 1
    while i > 0 and five[i - 1]:
        i -= 1
    return g.frac.to_numpy()[i]


# phase 2: is a forced 5 at token 0 specific to traces that end up abstaining?
all2 = df[df.phase == 2]
s2 = all2[all2.kind == "start"].set_index("question_index")
s2 = s2.assign(five=(s2.forced_answer == 5).astype(int))
p1_correct = pd.read_csv(d / "phases.csv").query("phase == 1").set_index("question_index").is_correct
s2["p1_correct"] = s2.index.map(p1_correct)
ab, an = s2[s2.is_nota == 1], s2[s2.is_nota == 0]
rate = lambda x: {"p": round(float(x.five.mean()), 3), "n": len(x)} if len(x) else {"n": 0}
report["start_forced_5"] = {
    "given_abstained": rate(ab), "given_answered": rate(an),
    "auroc_prob_5_at_start_abstain": auroc(s2.is_nota, s2.prob_5),
    "given_abstained_p1_wrong": rate(ab[ab.p1_correct == 0]), "given_abstained_p1_right": rate(ab[ab.p1_correct == 1]),
    "onset_sustained_abstained": quart(all2[all2.is_nota == 1].groupby("question_index").apply(sustained_onset,
                                                                                               include_groups=False)),
    "commitment_answered": quart(tr[(tr.phase == 2) & (tr.is_nota == 0)].commit)}

(d / "checkpoints_analysis.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str))
