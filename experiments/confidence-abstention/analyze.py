"""Early-signal analysis of one run directory written by abstention_modal.py.

    python analyze.py results/<tag>            # prints a summary, writes <tag>/analysis.json

The paper's two headline tests, on its own measures:
  P1  AUROC(verbal -> correct) < AUROC(calibrated -> correct)          [Phase 1]
  P2  verbal confidence predicts Phase-2 abstention beyond calibrated   [nested logistic, LR test]
Calibrated confidence = softmax(choice logprobs / tau)[chosen option], tau fit on Phase 0 by
minimising ECE (the paper's temperature scaling). Needs pandas, numpy, scipy, scikit-learn, statsmodels.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

d = Path(sys.argv[1])
df = pd.read_csv(d / "phases.csv")
LP = [f"logprob_choice_{k}" for k in range(1, 5)]
rng = np.random.default_rng(0)
report = {"run": d.name, "n": {int(p): int((df.phase == p).sum()) for p in df.phase.unique()}}


def chosen_conf(rows, tau):
    z = rows[LP].to_numpy() / tau
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    idx = rows.final_answer.astype(int).clip(1, 4).to_numpy() - 1
    return p[np.arange(len(rows)), idx]


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    return sum(m.mean() * abs(conf[m].mean() - correct[m].mean())
               for lo, hi in zip(edges[:-1], edges[1:]) if (m := (conf > lo) & (conf <= hi)).any())


def auc_ci(y, a, b, reps=1000):
    """Bootstrap 95% CI of AUROC(a) - AUROC(b)."""
    gaps = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            gaps.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return np.quantile(gaps, [0.025, 0.975]).round(3).tolist()


# descriptives per phase
for p, g in df.groupby("phase"):
    report[f"phase{p}"] = dict(accuracy=round(g.is_correct.mean(), 3), abstention=round(g.is_nota.mean(), 3),
                               truncated=int(g.truncated.sum()), parse=g.parse_method.value_counts().to_dict(),
                               mean_reasoning_tokens=round(g.reasoning_tokens.mean(), 1))

# temperature scaling on phase 0
p0 = df[df.phase == 0]
taus = np.logspace(-1, 1.5, 300)
tau = float(min(taus, key=lambda t: ece(chosen_conf(p0, t), p0.is_correct.to_numpy())))
report["tau"] = round(tau, 2)
report["phase0_ece"] = {"raw": round(ece(chosen_conf(p0, 1), p0.is_correct.to_numpy()), 3),
                        "calibrated": round(ece(chosen_conf(p0, tau), p0.is_correct.to_numpy()), 3)}

# P1: correctness discrimination on phase 1
p1 = df[df.phase == 1].copy()
p1["cal"] = chosen_conf(p1, tau)
p1["raw"] = chosen_conf(p1, 1.0)
verbal = pd.read_csv(d / "verbal.csv")[["question_index", "confidence_midpoint", "confidence_valid"]]
p1 = p1.merge(verbal, on="question_index").rename(columns={"confidence_midpoint": "verbal"})
v = p1[p1.confidence_valid == 1]
y = v.is_correct.to_numpy()
report["verbal_valid"] = f"{len(v)}/{len(p1)}"
report["verbal_class_distribution"] = {round(k, 2): int(c) for k, c in v.verbal.value_counts().sort_index().items()}
p1r = {k: round(roc_auc_score(y, v[k]), 3) for k in ["cal", "raw", "verbal"]}
p1r["cal_minus_verbal_ci95"] = auc_ci(y, v.cal.to_numpy(), v.verbal.to_numpy())
p1r["frac_cal_above_0.99"] = round((v.cal > 0.99).mean(), 3)
if v.reasoning_tokens.max() > 0:  # thinking run: the trace's own correctness signal
    p1r["neg_reasoning_tokens"] = round(roc_auc_score(y, -v.reasoning_tokens), 3)
    p1r["neg_wait_count"] = round(roc_auc_score(y, -v.wait_count), 3)
report["P1_auroc_correct"] = p1r

# P2: verbal beyond calibrated for phase-2 abstention
p2 = df[df.phase == 2][["question_index", "is_nota", "is_correct"]].merge(
    v[["question_index", "cal", "verbal", "is_correct"]].rename(columns={"is_correct": "p1_correct"}),
    on="question_index")
r2 = dict(n=len(p2), abstentions=int(p2.is_nota.sum()),
          abstention_given_p1_correct=round(p2[p2.p1_correct == 1].is_nota.mean(), 3),
          abstention_given_p1_incorrect=round(p2[p2.p1_correct == 0].is_nota.mean(), 3))
if min(p2.is_nota.sum(), (1 - p2.is_nota).sum()) >= 10:
    X = (p2[["cal", "verbal"]] - p2[["cal", "verbal"]].mean()) / p2[["cal", "verbal"]].std(ddof=0)
    fit = lambda cols: sm.Logit(p2.is_nota, sm.add_constant(X[cols])).fit(disp=0, maxiter=200)
    m_cal, m_verb, m_both = fit(["cal"]), fit(["verbal"]), fit(["cal", "verbal"])
    for name, full, reduced in [("verbal_beyond_cal", m_both, m_cal), ("cal_beyond_verbal", m_both, m_verb)]:
        lr = 2 * (full.llf - reduced.llf)
        r2[name] = dict(lr=round(lr, 2), p=float(f"{chi2.sf(lr, 1):.2g}"))
    r2["coef_per_sd"] = {k: round(float(m_both.params[k]), 3) for k in ["cal", "verbal"]}
    r2["pseudo_r2_both"] = round(float(m_both.prsquared), 3)
else:
    r2["not_estimable"] = "fewer than 10 abstentions or 10 answers; nested logit not run"
report["P2_abstention"] = r2

(d / "analysis.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str))
