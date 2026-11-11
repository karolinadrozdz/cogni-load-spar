"""Merge two judge passes + tiebreaks into trace_labels.csv and compute trace_labels_summary.json."""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import cohen_kappa_score

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "results/80b-thinking")
J = RUN / "trace_judging"
KINDS = ["RESOLVE", "JUDGE", "MIXED", "OTHER"]
FIELDS = ["kind", "candidate", "candidate_in_options", "decision_reason", "rationale"]
UNCERTAINTY = re.compile(r"\b(not sure|unsure|don'?t know|do not know|can'?t recall|cannot recall|"
                         r"don'?t remember|can'?t remember|not certain|i think|maybe|i'?m not confident)\b", re.I)


def load(pattern):
    rows = [json.loads(l) for f in sorted(J.glob(pattern)) for l in open(f) if l.strip()]
    return pd.DataFrame(rows).drop_duplicates("id", keep="last").set_index("id")


def traces():
    p = pd.read_csv(RUN / "phases.csv")
    v = pd.read_csv(RUN / "verbal.csv")
    p1 = p[p.phase == 1].set_index("question_index")
    p2 = p[p.phase == 2].set_index("question_index")
    return p1, p2, v.set_index("question_index")


def merge_labels(s):
    a, b = load("labels_a_*.jsonl"), load("labels_b_*.jsonl")
    assert set(a.index) == set(b.index) == set(s.index), (len(a), len(b), len(s))
    a, b = a.loc[s.index], b.loc[s.index]
    disagreed = a.kind != b.kind
    final = a[FIELDS].copy()
    if disagreed.any():
        t = load("labels_tiebreak_*.jsonl")
        missing = set(disagreed[disagreed].index) - set(t.index)
        assert not missing, f"{len(missing)} disagreements lack a tiebreak"
        final.loc[disagreed] = t.loc[disagreed[disagreed].index, FIELDS].values
    df = s.join(final)
    df["kind_a"], df["kind_b"], df["disagreed"] = a.kind, b.kind, disagreed.astype(int)
    df["candidate"] = df.candidate.fillna("").astype(str)
    df["candidate_in_options"] = df.candidate_in_options.astype(int)
    has = df.candidate.str.strip() != ""
    df["candidate_correct"] = (df.candidate_in_options == df.gt_position).astype(int).where(has)
    return df, a, b


def merge():
    df, a, b = merge_labels(pd.read_csv(J / "sample.csv").set_index("id"))
    p1, p2, v = traces()
    text = {}
    for i, r in df.iterrows():
        text[i] = (p2 if r.set in "AB" else v).loc[r.question_index, "reasoning"]
    df["uncertainty_phrases"] = [len(UNCERTAINTY.findall(text[i])) for i in df.index]
    df["phase1_correct"] = p1.loc[df.question_index, "is_correct"].values
    return df, a, b, text


def rate(num, den):
    return {"rate": round(num / den, 3) if den else None, "num": round(float(num), 1), "den": round(float(den), 1)}


def summary(df, a, b):
    out = {}
    out["agreement"] = {
        "cohen_kappa_kind": round(cohen_kappa_score(a.kind, b.kind), 3),
        "raw_agreement_kind": round((a.kind == b.kind).mean(), 3),
        "confusion_a_rows_b_cols": pd.crosstab(a.kind, b.kind).to_dict(orient="index"),
        "raw_agreement_decision_reason": round((a.decision_reason == b.decision_reason).mean(), 3),
        "n_disagreed": int(df.disagreed.sum()),
        "by_set": {k: {"n": int(len(g)), "cohen_kappa_kind": round(cohen_kappa_score(a.kind[g.index], b.kind[g.index]), 3),
                       "raw_agreement_kind": round((a.kind[g.index] == b.kind[g.index]).mean(), 3)}
                   for k, g in df.groupby("set")},
    }
    out["kind_share_by_set"] = {
        s: {"n": int(len(g)), **{k: round((g.kind == k).mean(), 3) for k in KINDS}}
        for s, g in df.groupby("set")}
    out["kind_counts_by_set"] = pd.crosstab(df.set, df.kind).to_dict(orient="index")
    out["judge_plus_mixed_share_of_abstentions"] = round(df[df.set == "A"].kind.isin(["JUDGE", "MIXED"]).mean(), 3)

    # A = all abstentions; B = 100/100 sample of answered rows, reweighted to the answered population
    p = pd.read_csv(RUN / "phases.csv")
    answered = p[(p.phase == 2) & (p.truncated == 0) & (p.is_nota == 0)]
    stratum_n = answered.is_correct.value_counts()
    ab = df[df.set.isin(["A", "B"])].copy()
    ab["w"] = 1.0
    for c in (0, 1):
        m = (ab.set == "B") & (ab.is_correct == c)
        ab.loc[m, "w"] = stratum_n[c] / m.sum()
    det = {}
    for k in KINDS:
        g = ab[ab.kind == k]
        abst, wrong = g[g.set == "A"], g[g.phase1_correct == 0]
        det[k] = {"P(phase1_wrong | abstained, kind)": rate((abst.phase1_correct == 0).sum(), len(abst)),
                  "n_abstained": int(len(abst))}
        if (df.set == "B").any():  # set-A-only runs have no answered sample to reweight
            det[k] |= {"P(abstained | phase1_wrong, kind) [B reweighted]": rate(wrong[wrong.set == "A"].w.sum(), wrong.w.sum()),
                       "n_sampled_answered": int((g.set == "B").sum())}
    out["error_detection_AB"] = det
    a = df[df.set == "A"]
    tab = [[((a.kind == k) & (a.phase1_correct == 0)).sum(), ((a.kind == k) & (a.phase1_correct == 1)).sum()]
           for k in ("JUDGE", "RESOLVE")]
    out["fisher_phase1_wrong_given_abstained_JUDGE_vs_RESOLVE"] = {
        "table_[wrong,right]": [[int(x) for x in t] for t in tab], "p": round(fisher_exact(tab)[1], 3)}
    out["phase1_wrong_base_rate"] = round(1 - p[p.phase == 1].is_correct.mean(), 3)

    def anatomy(g, correct_col):
        r = g[g.kind == "RESOLVE"]
        has = r.candidate.str.strip() != ""
        return {
            "n_resolve": int(len(r)),
            "candidate_correct_listed": int((has & (r.candidate_correct == 1)).sum()),
            "candidate_correct_listed_first_answer_correct": int((has & (r.candidate_correct == 1) & (r[correct_col] == 1)).sum()),
            "candidate_correct_listed_first_answer_wrong": int((has & (r.candidate_correct == 1) & (r[correct_col] == 0)).sum()),
            "knowledge_error (candidate incorrect)": int((has & (r.candidate_correct == 0)).sum()),
            "knowledge_error_overriding_correct_first": int((has & (r.candidate_correct == 0) & (r[correct_col] == 1)).sum()),
            "candidate_incorrect_listed": int((has & (r.candidate_correct == 0) & (r.candidate_in_options > 0)).sum()),
            "candidate_not_listed": int((has & (r.candidate_in_options == 0)).sum()),
            "question_wrong": int((r.decision_reason == "question_wrong").sum()),
            "no_candidate": int((~has).sum()),
            "crosstab_candidate_correct_x_in_options": pd.crosstab(
                r.candidate_correct.fillna(-1).astype(int), r.candidate_in_options).to_dict(orient="index"),
        }
    out["resolve_anatomy"] = {"A_abstentions (first answer = Phase 1)": anatomy(df[df.set == "A"], "phase1_correct")}
    if (df.set == "C").any():
        out["resolve_anatomy"]["C_no_chance (first answer = rated answer)"] = anatomy(df[df.set == "C"], "is_correct")
    out["decision_reason_by_set"] = pd.crosstab(df.set, df.decision_reason).to_dict(orient="index")
    out["instruction_license_share_of_abstentions"] = round((df[df.set == "A"].decision_reason == "instruction_license").mean(), 3)
    out["uncertainty_phrases_mean_by_kind"] = df.groupby("kind").uncertainty_phrases.mean().round(2).to_dict()
    out["checkpoint_crosscheck"] = ("trace_labels_crosscheck.json (trace_labels/crosscheck.py)"
                                    if (RUN / "checkpoints.csv").exists() else "skipped: checkpoints.csv not present")
    return out


def main():
    df, a, b, text = merge()
    cols = ["set", "question_index", "phase", "is_correct", "is_nota", "confidence_class", "phase1_correct",
            "gt_position", "final_answer", "previous_answer", "kind", "candidate", "candidate_in_options",
            "candidate_correct", "decision_reason", "uncertainty_phrases", "rationale", "disagreed", "kind_a", "kind_b"]
    df.reindex(columns=cols).to_csv(RUN / "trace_labels.csv")  # build_batches samples lack confidence_class, previous_answer
    s = summary(df, a, b)
    (RUN / "trace_labels_summary.json").write_text(json.dumps(s, indent=2, default=str))
    # 40-row human audit: 10 per kind present, topped up from disagreed rows
    audit = pd.concat([g.sample(min(10, len(g)), random_state=0) for _, g in df.groupby("kind")])
    rest = df[(df.disagreed == 1) & ~df.index.isin(audit.index)]
    audit = pd.concat([audit, rest.sample(min(40 - len(audit), len(rest)), random_state=0)])
    audit = audit.assign(trace=[text[i] for i in audit.index], human_kind="", human_agrees="")
    audit[["set", "question", "kind", "candidate", "decision_reason", "rationale", "human_kind", "human_agrees", "trace"]] \
        .to_csv(J / "human_audit.csv")
    print(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main()
