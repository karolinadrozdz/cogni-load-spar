"""Merge judge passes + tiebreaks for a set-B-only answered sample (trace_judging_answered/), reweighted to the answered
Phase 2 population, into trace_labels.csv and trace_labels_summary.json in that directory.

    python trace_labels/summarize_answered.py results/<tag>
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

import summarize

RUN = Path(sys.argv[1])
summarize.J = J = RUN / "trace_judging_answered"


def main():
    df, a, b = summarize.merge_labels(pd.read_csv(J / "sample.csv").set_index("id"))
    p = pd.read_csv(RUN / "phases.csv")
    p2 = p[(p.phase == 2) & (p.truncated == 0)]
    answered = p2[p2.is_nota == 0]
    stratum_n = answered.is_correct.value_counts()
    df["w"] = df.is_correct.map(stratum_n / df.is_correct.value_counts())
    df.drop(columns="w").to_csv(J / "trace_labels.csv")
    has = df.candidate.str.strip() != ""

    def wshare(m):
        return round(float(df.w[m].sum() / df.w.sum()), 3)

    p1 = p[p.phase == 1].set_index("question_index").is_correct
    abst = p2.assign(p1=p1.loc[p2.question_index].values).groupby("p1").is_nota.mean()
    out = {
        "cohen_kappa_kind": round(cohen_kappa_score(a.kind, b.kind), 3),
        "raw_agreement_kind": round((a.kind == b.kind).mean(), 3),
        "n": int(len(df)), "n_disagreed": int(df.disagreed.sum()),
        "sample_right_wrong": [int((df.is_correct == 1).sum()), int((df.is_correct == 0).sum())],
        "answered_population_right_wrong": [int(stratum_n.get(1, 0)), int(stratum_n.get(0, 0))],
        "kind_share_weighted": {k: wshare(df.kind == k) for k in summarize.KINDS},
        "own_candidate_share_weighted": wshare(has),
        "candidate_equals_final_answer_share_weighted": wshare(has & (df.candidate_in_options == df.final_answer)),
        "candidate_correct_weighted": wshare(has & (df.candidate_correct == 1)),
        "phase2_accuracy_answered": round(answered.is_correct.mean(), 3), "n_answered": int(len(answered)),
        "abstention_phase1_right_wrong": [round(abst.get(1), 3), round(abst.get(0), 3)],
        "kind_counts_by_correct": pd.crosstab(df.is_correct, df.kind).to_dict(orient="index"),
    }
    (J / "trace_labels_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
