"""Headline measures per checkpoint for run directories given in training-stage order, with bootstrap CIs
per checkpoint and paired bootstrap CIs for the difference between adjacent checkpoints.

    python analyze_sweep.py results/olmo32b-s1-step1000 results/olmo32b-s1-step2000 ... --out results/sweep_olmo32b

Writes <out>.csv (one row per run: each measure and its 95% CI) and <out>_diffs.csv (one row per adjacent pair:
later minus earlier, paired 95% CI). Paired = the same Phase 1 items are resampled in both runs, so item
difficulty cancels. tau stays at each run's Phase 0 fit. A measure is NaN where the run lacks it (no Phase 2 or
verbal pass in base mode, no trace when not thinking). The gate_A/B/C columns say whether each tier's measures
mean anything for that checkpoint (None = tier not measured).

The measures depend on the task the runs used (params.json "task"; absent = confidence_abstention): each entry of
SWEEP_TASKS gives items(run_dir) -> one row per item, indexed by item id, with a `question` column (checked to match
across runs); measures(items) -> dict of floats; gates(measures, ci_lo) -> dict of gate_* flags; show = columns printed.
The bootstrap, paired differences and same-items check are shared. Adding a task: TASKS.md.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from analyze import CLOZE, chosen_conf, cloze_choice, ece, fit_tau


def auroc(y, s):
    """Rank-based AUROC (ties averaged); NaN when only one class is present."""
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def items(d):
    """One row per Phase 1 question with every per-item quantity the measures need."""
    df = pd.read_csv(d / "phases.csv")
    p0, p1 = df[df.phase == 0], df[df.phase == 1].copy()
    tau = fit_tau(p0, p0.is_correct.to_numpy())
    t = p1[["question_index", "question", "is_correct", "truncated", "reasoning_tokens"]].copy()
    t["cal"] = chosen_conf(p1, tau)
    t["digit_mass"] = p1.digit_mass if "digit_mass" in p1 else np.nan
    t["consistency"] = p1.consistency if "consistency" in p1 else np.nan
    if CLOZE[0] in p1:
        i0, y0 = cloze_choice(p0)
        i1, y1 = cloze_choice(p1)
        t["cloze_correct"], t["cloze_cal"] = y1, chosen_conf(p1, fit_tau(p0, y0, cols=CLOZE, chosen=i0), CLOZE, i1)
    else:
        t["cloze_correct"] = t["cloze_cal"] = np.nan
    t["verbal"] = t["verbal_valid"] = np.nan
    if (d / "verbal.csv").exists():
        v = pd.read_csv(d / "verbal.csv").set_index("question_index")
        t["verbal"] = t.question_index.map(v.confidence_midpoint)
        t["verbal_valid"] = t.question_index.map(v.confidence_valid)
    p2 = df[df.phase == 2].set_index("question_index")
    t["nota"] = t.question_index.map(p2.is_nota) if len(p2) else np.nan
    t["p2_truncated"] = t.question_index.map(p2.truncated) if len(p2) else np.nan
    return t.set_index("question_index")


def short_answer_items(d):
    a = pd.read_csv(d / "answers.csv")
    t = a[["question_index", "question", "is_correct", "truncated", "reasoning_tokens"]].copy()
    t["short"] = (a.answer_words <= 5).astype(int)
    return t.set_index("question_index")


def short_answer_measures(t):
    y = t.is_correct.to_numpy()
    return dict(accuracy=y.mean(), short_format=t.short.mean(),
                auroc_neg_trace_len=auroc(y, -t.reasoning_tokens.to_numpy()) if t.reasoning_tokens.max() > 0 else np.nan,
                truncated=t.truncated.mean())


def short_answer_gates(m, lo):
    """Accuracy only measures knowledge where the model answers in the short format."""
    return dict(gate_format=bool(m["short_format"] >= 0.9 and m["truncated"] < 0.05))


def mean_or_nan(x):
    return float(np.mean(x)) if len(x) else np.nan


def measures(t):
    y = t.is_correct.to_numpy()
    wrong, right = t.nota[y == 0], t.nota[y == 1]
    v = t[t.verbal_valid == 1]
    cz = t.cloze_correct.notna().to_numpy()
    return dict(
        accuracy=y.mean(),
        auroc_cal=auroc(y, t.cal.to_numpy()),
        ece_cal=ece(t.cal.to_numpy(), y),
        digit_mass=t.digit_mass.mean(),
        cloze_accuracy=t.cloze_correct.mean(),
        cloze_auroc_cal=auroc(t.cloze_correct.to_numpy()[cz], t.cloze_cal.to_numpy()[cz]) if cz.any() else np.nan,
        cloze_ece_cal=ece(t.cloze_cal.to_numpy()[cz], t.cloze_correct.to_numpy()[cz]) if cz.any() else np.nan,
        verbal_valid=t.verbal_valid.mean(),
        auroc_verbal=auroc(v.is_correct.to_numpy(), v.verbal.to_numpy()) if len(v) else np.nan,
        abstention=t.nota.mean(),
        abstention_gap=mean_or_nan(wrong.dropna()) - mean_or_nan(right.dropna()),  # abstain|wrong - abstain|right
        auroc_neg_trace_len=auroc(y, -t.reasoning_tokens.to_numpy()) if t.reasoning_tokens.max() > 0 else np.nan,
        auroc_consistency=auroc(y, t.consistency.to_numpy()) if t.consistency.notna().all() else np.nan,
        truncated=np.nanmean(np.r_[t.truncated, t.p2_truncated]),
    )


def bootstrap(stat, n, reps, rng):
    """95% percentile CI of stat(index array) over item resamples."""
    draws = np.array([list(stat(rng.integers(0, n, n)).values()) for _ in range(reps)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns: measures the run does not have
        return np.nanquantile(draws, 0.025, axis=0), np.nanquantile(draws, 0.975, axis=0)


def gates(m, lo):
    """Tier A: answers in MC format and above chance. B: valid verbal and an abstention rate the paper would keep.
    C: few truncated traces (the probe's end-checkpoint sanity is checked separately)."""
    return dict(gate_A=bool(m["digit_mass"] >= 0.95 and lo["accuracy"] > 0.25) if not np.isnan(m["digit_mass"]) else None,
                gate_B=bool(m["verbal_valid"] >= 0.9 and 0.02 <= m["abstention"] <= 0.95)
                if not np.isnan(m["abstention"]) else None,
                gate_C=bool(m["truncated"] < 0.05) if not np.isnan(m["auroc_neg_trace_len"]) else None)


SWEEP_TASKS = {
    "confidence_abstention": dict(items=items, measures=measures, gates=gates,
                                  show=["accuracy", "auroc_cal", "cloze_accuracy", "cloze_auroc_cal", "auroc_verbal",
                                        "abstention_gap", "auroc_neg_trace_len"]),
    "short_answer": dict(items=short_answer_items, measures=short_answer_measures, gates=short_answer_gates,
                         show=["accuracy", "short_format", "auroc_neg_trace_len"]),
}


def run_params(d):
    return json.loads((d / "params.json").read_text()) if (d / "params.json").exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path, help="run directories, ordered by training stage")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    names = {run_params(d).get("task", "confidence_abstention") for d in a.runs}
    assert len(names) == 1, f"runs are from different tasks: {names}"
    task = SWEEP_TASKS[names.pop()]
    measures, gates = task["measures"], task["gates"]
    tables = [task["items"](d) for d in a.runs]

    rows = []
    for d, t in zip(a.runs, tables):
        m = measures(t)
        lo, hi = bootstrap(lambda i: measures(t.iloc[i]), len(t), a.reps, rng)
        lo, hi = dict(zip(m, lo)), dict(zip(m, hi))
        params = run_params(d)
        row = dict(run=d.name, model=params.get("model"), revision=params.get("revision"),
                   thinking=params.get("thinking"), n_items=len(t), **gates(m, lo))
        for k in m:
            row.update({k: m[k], f"{k}_lo": lo[k], f"{k}_hi": hi[k]})
        rows.append(row)
    per_run = pd.DataFrame(rows)

    diffs = []
    for (da, ta), (db, tb) in zip(zip(a.runs, tables), zip(a.runs[1:], tables[1:])):
        common = ta.index.intersection(tb.index)
        ta, tb = ta.loc[common], tb.loc[common]
        assert (ta.question == tb.question).all(), f"{da.name} and {db.name} are not on the same items"
        delta = lambda i: {k: vb - va for (k, va), vb in zip(measures(ta.iloc[i]).items(), measures(tb.iloc[i]).values())}
        point = delta(np.arange(len(common)))
        lo, hi = bootstrap(delta, len(common), a.reps, rng)
        row = dict(earlier=da.name, later=db.name, n_items=len(common))
        for k, l, h in zip(point, lo, hi):
            row.update({k: point[k], f"{k}_lo": l, f"{k}_hi": h})
        diffs.append(row)
    per_pair = pd.DataFrame(diffs)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(f"{a.out}.csv", index=False)
    per_pair.to_csv(f"{a.out}_diffs.csv", index=False)
    show = task["show"]
    gate_cols = [k for k in per_run if k.startswith("gate_")]
    fmt = lambda r, k: "" if pd.isna(r[k]) else f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"
    gate_title = "gates " + "/".join(k[len("gate_"):] for k in gate_cols)
    print(pd.DataFrame([{"run": r["run"], **{k: fmt(r, k) for k in show},
                         gate_title: "/".join(str(r[k]) for k in gate_cols)} for _, r in per_run.iterrows()])
          .to_string(index=False))
    if len(per_pair):
        print("\nadjacent differences (later - earlier), paired 95% CI:")
        print(pd.DataFrame([{"pair": f"{r['earlier']} -> {r['later']}", **{k: fmt(r, k) for k in show}}
                            for _, r in per_pair.iterrows()]).to_string(index=False))
    print(f"\nwrote {a.out}.csv and {a.out}_diffs.csv")


if __name__ == "__main__":
    main()
