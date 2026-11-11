"""Trace length against correctness, abstention and item difficulty on existing runs, plus quick verbal checks.

    python analyze_length.py      # reads results/<tag>/, writes results/sweep/length_*.csv, prints summaries

Rows kept everywhere: phase 1 or 2, not truncated, not parsed by the log-prob fallback. Differences of medians between
groups within one run (wrong - right, abstained - answered) are CI'd by resampling the run's rows (the groups are
different items). OLS CIs are cluster-robust by item. diff_loo = item's mean is_correct over the other runs where it
survives the filters (excludes the row's own outcome). PopQA also has log_s_pop = log(1 + subject Wikipedia popularity)
and prop (relation type) from the question file, difficulty measures independent of the model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import rankdata, spearmanr

from analyze_sweep import auroc, bootstrap

R = Path(__file__).parent / "results"
OUT = R / "sweep"
REPS = 1000
THINK_POPQA = ["olmo32b-popqa-think-" + s for s in [
    "sft-1e-4-step2000", "sft-1e-4-step6000", "sft-main", "dpo", "rl-step0050", "rl-step0250", "rl-step0500",
    "rl-main", "rl31-step1100", "rl31-step1500", "rl31-step1900", "rl31-main"]]
QWEN = ["8b-on", "30b-thinking-greedy", "80b-thinking", "80b-thinking-p2neutral", "80b-thinking-p2skip"]
QWEN_SIMPLEQA_PROXY = ["8b-on", "30b-thinking-greedy", "80b-thinking"]
THINK_SIMPLEQA = ["olmo32b-think-sft-main", "olmo32b-think-dpo", "olmo32b-think-rl-main", "olmo32b-think-rl31-main"]
RUN_SETS = {"think_popqa": THINK_POPQA, "qwen_simpleqa": QWEN, "think_simpleqa": THINK_SIMPLEQA}
POPQA_ITEMS = Path(__file__).parent / "data" / "popqa_phase1_questions.csv"
rng = np.random.default_rng(0)


def raw(tag):
    df = pd.read_csv(R / tag / "phases.csv")
    df = df[df.phase.isin([1, 2])].copy()
    df["truncated"] = df.truncated.astype(bool)
    df["keep"] = ~df.truncated & df.parse_method.ne("logprob")
    return df


def load(tag):
    return raw(tag).query("keep").copy()


def med(x):
    return float(np.median(x)) if len(x) else np.nan


def with_ci(stat, n):
    point = stat(np.arange(n))
    lo, hi = bootstrap(stat, n, REPS, rng)
    return {**point, **{f"{k}_lo": v for k, v in zip(point, lo)}, **{f"{k}_hi": v for k, v in zip(point, hi)}}


def split_stats(g, phase):
    """Length and wait by outcome (1 = wrong in phase 1, abstained in phase 2): medians, means, difference, AUROC."""
    y_all = (1 - g.is_correct if phase == 1 else g.is_nota).to_numpy()
    cols = {"len": g.reasoning_tokens.to_numpy(float), "wait": g.wait_count.to_numpy(float)}
    nota, correct = g.is_nota.to_numpy(), g.is_correct.to_numpy()

    def stat(i):
        y, r = y_all[i], {}
        for name, x in cols.items():
            x = x[i]
            r.update({f"{name}_med_1": med(x[y == 1]), f"{name}_med_0": med(x[y == 0]),
                      f"{name}_mean_1": x[y == 1].mean() if (y == 1).any() else np.nan,
                      f"{name}_mean_0": x[y == 0].mean() if (y == 0).any() else np.nan,
                      f"{name}_auroc": auroc(y, x)})
            r[f"{name}_med_diff"] = r[f"{name}_med_1"] - r[f"{name}_med_0"]
        answered = nota[i] == 0
        r["len_auroc_wrong_answered"] = auroc(1 - correct[i][answered], cols["len"][i][answered]) if phase == 2 else np.nan
        return r

    return with_ci(stat, len(g))


def length_by_checkpoint():
    rows = []
    for run_set, tags in RUN_SETS.items():
        for tag in tags:
            df = raw(tag)
            for phase, g in df.groupby("phase"):
                kept = g[g.keep]
                rows.append(dict(run_set=run_set, run=tag, phase=phase, outcome="wrong" if phase == 1 else "abstain",
                                 n=len(kept), n_1=int((1 - kept.is_correct if phase == 1 else kept.is_nota).sum()),
                                 dropped_truncated=int(g.truncated.sum()),
                                 dropped_logprob=int((~g.truncated & ~g.keep).sum()), **split_stats(kept, phase)))
    return pd.DataFrame(rows)


def popqa_items():
    """log_s_pop and prop per Phase 1 question_index (the row position in the question file)."""
    q = pd.read_csv(POPQA_ITEMS)
    return pd.DataFrame({"log_s_pop": np.log(q.s_pop + 1), "prop": q.prop})


def long_table(tags, items=None):
    """Phase 1 rows, one per (item, run), with diff_loo and, when given, the external item measures."""
    L = pd.concat([load(t).query("phase == 1").assign(ckpt=k) for k, t in enumerate(tags)])
    L = L[L.reasoning_tokens > 0].copy()
    L["log_len"] = np.log(L.reasoning_tokens)
    if "consistency" not in L:
        L["consistency"] = np.nan
    g = L.groupby("question_index").is_correct
    s, n = g.transform("sum"), g.transform("count")
    L["diff_loo"] = ((s - L.is_correct) / (n - 1)).where(n > 1)
    L = L[["question_index", "ckpt", "log_len", "reasoning_tokens", "is_correct", "consistency", "diff_loo"]]
    return (L if items is None else L.join(items, on="question_index")).reset_index(drop=True)


def ols(formula, L):
    L = L.dropna(subset=[c for c in ["diff_loo", "log_s_pop"] if c in formula])
    return smf.ols(formula, L).fit(cov_type="cluster", cov_kwds={"groups": L.question_index})


def term(fit, name, label):
    ci = fit.conf_int().loc[name]
    return dict(model=label, term=name, coef=fit.params[name], lo=ci[0], hi=ci[1], n=int(fit.nobs))


def slope(fit, expr, label):
    """Linear combination of coefficients (e.g. the correct-branch slope) with its CI."""
    t = fit.t_test(expr)
    lo, hi = np.asarray(t.conf_int()).ravel()
    return dict(model=label, term=expr, coef=float(np.asarray(t.effect).ravel()[0]), lo=lo, hi=hi, n=int(fit.nobs))


def partial_spearman(x, y, Z):
    """Pearson on ranks after residualising both rank vectors on the columns of Z (OLS with intercept)."""
    Z = np.column_stack([np.ones(len(x)), Z])
    res = lambda r: r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]
    return float(np.corrcoef(res(rankdata(x)), res(rankdata(y)))[0, 1])


def per_checkpoint_partials(L, tags, run_set):
    rows = []
    for k, g in L.groupby("ckpt"):
        gl = g.dropna(subset=["diff_loo"])
        cons = g.dropna(subset=["consistency"])
        rows.append(dict(run_set=run_set, run=tags[k], n=len(g),
                         spearman_len_correct=spearmanr(g.reasoning_tokens, g.is_correct)[0],
                         partial_s_pop=partial_spearman(g.reasoning_tokens, g.is_correct, np.column_stack(
                             [rankdata(g.log_s_pop), pd.get_dummies(g.prop, drop_first=True, dtype=float)]))
                         if "log_s_pop" in g else np.nan,
                         partial_loo=partial_spearman(gl.reasoning_tokens, gl.is_correct, rankdata(gl.diff_loo)),
                         spearman_consistency_diff_loo=spearmanr(cons.consistency, cons.diff_loo, nan_policy="omit")[0]
                         if len(cons) else np.nan))
    return rows


def item_vs_state():
    """Item difficulty vs correctness as predictors of length: OLS coefficients and per-checkpoint partials for the
    PopQA Think line and the Qwen SimpleQA runs; difficulty x correctness interaction and difficulty bins (PopQA)."""
    models, partials, bins = [], [], []
    for run_set, tags in [("think_popqa", THINK_POPQA), ("qwen_simpleqa", QWEN_SIMPLEQA_PROXY)]:
        popqa = run_set == "think_popqa"
        L = long_table(tags, popqa_items() if popqa else None)
        partials += per_checkpoint_partials(L, tags, run_set)
        if popqa:
            models.append(term(ols("log_len ~ C(ckpt) + log_s_pop + C(prop)", L), "log_s_pop", f"{run_set}: between s_pop"))
            joint = ols("log_len ~ C(ckpt) + log_s_pop + C(prop) + is_correct", L)
            models += [term(joint, t, f"{run_set}: joint ckpt FE + s_pop + prop") for t in ("log_s_pop", "is_correct")]
        models.append(term(ols("log_len ~ C(ckpt) + diff_loo", L), "diff_loo", f"{run_set}: between"))
        joint = ols("log_len ~ C(ckpt) + diff_loo + is_correct", L)
        models.append(term(joint, "is_correct", f"{run_set}: joint ckpt FE + diff_loo"))
        if not popqa:
            continue
        within = smf.ols("log_len ~ C(question_index) + C(ckpt) + is_correct", L).fit(
            cov_type="cluster", cov_kwds={"groups": L.question_index})
        models.append(term(within, "is_correct", f"{run_set}: within (item FE + ckpt FE)"))
        r2 = lambda f: smf.ols(f, L).fit().rsquared
        both = r2("log_len ~ C(question_index) + C(ckpt)")
        models += [dict(model=f"{run_set}: R2", term=t, coef=v, n=len(L)) for t, v in [
            ("item FE", r2("log_len ~ C(question_index)")), ("ckpt FE", r2("log_len ~ C(ckpt)")),
            ("item + ckpt FE", both), ("is_correct over item + ckpt FE", within.rsquared - both)]]
        L["log_len_adj"] = L.log_len - L.groupby("ckpt").log_len.transform("mean") + L.log_len.mean()
        for d in ["log_s_pop", "diff_loo"]:
            fit = ols(f"log_len ~ C(ckpt) + {d} * is_correct", L)
            models += [term(fit, f"{d}:is_correct", f"{run_set}: folded-X"),
                       slope(fit, f"{d} = 0", f"{run_set}: folded-X wrong-branch slope"),
                       slope(fit, f"{d} + {d}:is_correct = 0", f"{run_set}: folded-X correct-branch slope")]
            bins += difficulty_bins(L.dropna(subset=[d]), d)
    return pd.DataFrame(models), pd.DataFrame(partials), pd.DataFrame(bins)


def quintile(L, d):
    """Item difficulty quintile on d; bin 1 = easiest."""
    item = L.groupby("question_index")[d].mean()
    return pd.Series(pd.qcut(rankdata(-item.to_numpy(), method="ordinal"), 5, labels=range(1, 6)), index=item.index)


def difficulty_bins(L, d):
    """Checkpoint-adjusted mean log length per (difficulty quintile, correctness); bin 1 = easiest."""
    L = L.assign(bin=L.question_index.map(quintile(L, d)).astype(int))
    rows = []
    for (b, c), g in L.groupby(["bin", "is_correct"]):
        x = g.log_len_adj.to_numpy()
        rows.append(dict(proxy=d, bin=b, is_correct=c, n=len(g), diff_mean=g[d].mean(),
                         **with_ci(lambda i: {"mean_log_len": x[i].mean()}, len(x))))
    return rows


def sensitivity():
    """Accuracy, abstention and abstain|wrong - abstain|right on all rows vs on the filtered rows; `moves` = the
    filtered value lies outside the all-rows CI."""
    rows = []
    for run_set in ["think_popqa", "think_simpleqa"]:
        for tag in RUN_SETS[run_set]:
            df = raw(tag)
            p1, p2 = (df[df.phase == p].set_index("question_index") for p in (1, 2))
            t = pd.DataFrame({"correct": p1.is_correct, "keep1": p1.keep, "nota": p2.is_nota, "keep2": p2.keep})
            c, k1, a, k2 = (t[col].to_numpy() for col in ["correct", "keep1", "nota", "keep2"])

            def stat(i, filt):
                m1 = k1[i] if filt else np.ones(len(i), bool)
                m2 = k2[i] if filt else np.ones(len(i), bool)
                both, cc, aa = m1 & m2, c[i], a[i]
                return dict(accuracy=cc[m1].mean(), abstention=aa[m2].mean(),
                            delta=aa[both & (cc == 0)].mean() - aa[both & (cc == 1)].mean())

            full = with_ci(lambda i: stat(i, False), len(t))
            kept = stat(np.arange(len(t)), True)
            row = dict(run_set=run_set, run=tag)
            for p, g in df.groupby("phase"):
                row[f"p{p}_truncated"] = int(g.truncated.sum())
                row[f"p{p}_logprob"] = int(g.parse_method.eq("logprob").sum())
            for k in kept:
                row.update({f"{k}_all": full[k], f"{k}_all_lo": full[f"{k}_lo"], f"{k}_all_hi": full[f"{k}_hi"],
                            f"{k}_kept": kept[k],
                            f"{k}_moves": not full[f"{k}_lo"] <= kept[k] <= full[f"{k}_hi"]})
            rows.append(row)
    return pd.DataFrame(rows)


CHECKPOINT_RUNS = ["olmo32b-popqa-think-sft-main", "olmo32b-popqa-think-dpo", "olmo32b-popqa-think-rl31-main"]


def flip_check():
    """Phase 1 traces by difficulty: P(forced answer correct at some checkpoint with frac <= 0.5) for wrong traces,
    P(forced answer wrong at such a checkpoint) for correct traces."""
    s_pop = popqa_items().log_s_pop
    cuts = {"s_pop_tercile": pd.qcut(s_pop.rank(method="first"), 3, labels=["hard", "mid", "easy"]),
            "diff_loo_bins": quintile(long_table(THINK_POPQA).dropna(subset=["diff_loo"]), "diff_loo").astype(int)
            .map(lambda b: "1-3" if b <= 3 else "4-5")}
    rows = []
    for tag in CHECKPOINT_RUNS:
        c = pd.read_csv(R / tag / "checkpoints.csv").query("phase == 1 and frac <= 0.5")
        hit = (c.forced_answer == c.gt_position).groupby(c.question_index)
        p1 = load(tag).query("phase == 1").set_index("question_index")
        p1 = p1[p1.index.isin(c.question_index)]
        flip = np.where(p1.is_correct == 1, ~hit.all().reindex(p1.index), hit.any().reindex(p1.index))
        t = pd.DataFrame({"is_correct": p1.is_correct, "len": p1.reasoning_tokens, "flip": flip.astype(float)},
                         index=p1.index)
        for cut, level in cuts.items():
            for (correct, lev), g in t.groupby([t.is_correct, t.index.map(level)]):
                x = g.flip.to_numpy()
                rows.append(dict(run=tag, outcome="correct" if correct else "wrong", cut=cut, level=lev, n=len(g),
                                 median_len=g.len.median(), **with_ci(lambda i: {"p_early_flip": x[i].mean()}, len(x))))
    return pd.DataFrame(rows)


def verbal_coupling():
    """Spearman of verbal-pass trace length with Phase 2 trace length, and with verbal confidence."""
    rows = []
    for run_set, tags in RUN_SETS.items():
        for tag in tags:
            v = pd.read_csv(R / tag / "verbal.csv")
            v = v[~v.truncated.astype(bool)].set_index("question_index")
            p2 = load(tag).query("phase == 2").set_index("question_index").reasoning_tokens
            m = pd.DataFrame({"verbal_len": v.reasoning_tokens, "p2_len": p2, "conf": v.confidence_midpoint,
                              "valid": v.confidence_valid}).dropna(subset=["verbal_len"])
            lp, cv = m.dropna(subset=["p2_len"]), m[m.valid == 1]
            rows.append(dict(run_set=run_set, run=tag, n_len=len(lp), spearman_verbal_len_p2_len=spearmanr(lp.verbal_len, lp.p2_len)[0],
                             n_conf=len(cv), spearman_verbal_len_confidence=spearmanr(cv.verbal_len, cv.conf)[0]))
    return pd.DataFrame(rows)


ROUTE_RUNS = ["olmo32b-popqa-think-sft-main", "olmo32b-popqa-think-rl31-main", "80b-thinking"]


def routes():
    """Phase 2 abstentions by judge route (trace_labels/summarize.py, crosscheck.py outputs). For abstentions commit_frac
    is the sustained onset (first forced 5 followed only by 5s)."""
    rows = []
    for tag in ROUTE_RUNS:
        t = pd.read_csv(R / tag / "trace_labels.csv", index_col=0).query("set == 'A'")
        c = pd.read_csv(R / tag / "trace_labels_checkpoints.csv", index_col=0)
        t = t.join(c[["start_forced_5", "commit_frac"]])
        for kind, g in t.groupby("kind"):
            has = g.candidate_correct.notna()
            rows.append(dict(run=tag, kind=kind, n=len(g), share=len(g) / len(t), n_p1_wrong=int((g.phase1_correct == 0).sum()),
                             p_p1_wrong=(g.phase1_correct == 0).mean(), n_candidate=int(has.sum()),
                             n_candidate_correct=int(g.candidate_correct[has].sum()),
                             p_start_forced_5=g.start_forced_5.mean(), median_sustained_onset=g.commit_frac.median()))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
    a1 = length_by_checkpoint()
    a1.to_csv(OUT / "length_by_checkpoint.csv", index=False)
    print(a1[["run", "phase", "n", "n_1", "dropped_truncated", "dropped_logprob", "len_med_1", "len_med_0", "len_med_diff",
              "len_med_diff_lo", "len_med_diff_hi", "len_auroc", "len_auroc_wrong_answered"]].to_string(index=False))
    models, partials, bins = item_vs_state()
    models.to_csv(OUT / "length_models.csv", index=False)
    partials.to_csv(OUT / "length_item_vs_state.csv", index=False)
    bins.to_csv(OUT / "length_folded_x.csv", index=False)
    print(models.to_string(index=False), partials.to_string(index=False), bins.to_string(index=False), sep="\n\n")
    a5 = sensitivity()
    a5.to_csv(OUT / "length_sensitivity.csv", index=False)
    print(a5.to_string(index=False))
    a6 = flip_check()
    a6.to_csv(OUT / "length_flip.csv", index=False)
    print(a6.to_string(index=False))
    b3 = verbal_coupling()
    b3.to_csv(OUT / "length_verbal.csv", index=False)
    print(b3.to_string(index=False))
    b3 = routes()
    b3.to_csv(OUT / "length_routes.csv", index=False)
    print(b3.to_string(index=False))


if __name__ == "__main__":
    main()
