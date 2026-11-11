"""Outcomes of one steer-<model>-<tag> run: the no-trace gate and the reasoning-mode conditions.

    python analyze_steering.py results/steer-<model>-<tag> [--harness results/<unsteered harness tag>] [--routes <csv>]

Swing = P(abstain | alpha=-2) - P(abstain | alpha=+2), paired over items, bootstrap 95% CI (positive: steering
towards low confidence raises abstention). Slopes: GLM of abstain (logit) or log(1 + reasoning_tokens) on alpha,
SE clustered by item. P1 right/wrong = the item's unsteered no-trace Phase 1 answer (pre-trace runs: the harness
run's Phase 1 answer, and a vector-validity table from vectors.pt). In reasoning mode alpha=0 is
one unsteered run shared by every condition at that temperature. By-route swings need --routes (question_index, route in
resolve/judge/mixed: judge labels of the alpha=0 traces from trace_labels/) and are omitted without it. Gate: at the layer with the largest real swing, abstention monotone
in -alpha, swing >= 20 pp, random and shuffled swings < 5 pp. Writes STEERING-RESULTS.md (numbers only) in the run dir.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ALPHAS = [-2, -1, 0, 1, 2]
REPS = 1000


def swing(g, col="is_nota"):
    w = g[g.alpha.isin([-2, 2])].pivot_table(index="question_index", columns="alpha", values=col)
    if not {-2, 2} <= set(w.columns):
        return np.nan, (np.nan, np.nan)
    d = (w[-2] - w[2]).dropna().to_numpy()
    boots = d[np.random.default_rng(0).integers(0, len(d), (REPS, len(d)))].mean(1)
    return d.mean(), tuple(np.quantile(boots, [0.025, 0.975]))


def slope(g, y, family):
    if g[y].nunique() < 2 or g.alpha.nunique() < 2:  # no variation: coefficient undefined
        return np.nan, np.nan
    r = sm.GLM(g[y].astype(float), sm.add_constant(g.alpha.astype(float)), family=family).fit(
        cov_type="cluster", cov_kwds={"groups": g.question_index.to_numpy()})
    return r.params["alpha"], r.bse["alpha"]


def pct(x):
    return "" if pd.isna(x) else f"{100 * x:.1f}"


def fmt_swing(s):
    return f"{pct(s[0])} [{pct(s[1][0])}, {pct(s[1][1])}]"


def fmt_slope(s):
    return "" if pd.isna(s[0]) else f"{s[0]:.3f} ({s[1]:.3f})"


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(x) for x in r) + " |" for r in rows] + [""]


def rates(g, col="is_nota"):
    by = g.groupby("alpha")[col].mean()
    return [pct(by.get(a)) for a in ALPHAS]


def by_subset(g, keep):
    return swing(g[g.question_index.isin(keep)])


def gate_report(run, right, wrong, base_p1):
    df = load(run, "gate")
    p2, p1 = df[df.phase == 2], df[df.phase == 1]
    out = [f"## No-trace gate: Phase 2, {p2.question_index.nunique()} items", ""]
    rows, swings, monotone = [], {}, {}
    for (vec, L), g in p2.groupby(["vector", "layer"]):
        s = swing(g)
        by = g.groupby("alpha").is_nota.mean()
        if vec == "real":
            swings[L] = s[0]
            monotone[L] = bool(np.all(np.diff(by.reindex(ALPHAS).to_numpy()) <= 0))
        mass = g[g.alpha.isin([-2, 2])].groupby("alpha").digit_mass.mean()
        rows.append([vec, L, *rates(g), fmt_swing(s), fmt_slope(slope(g, "is_nota", sm.families.Binomial())),
                     pct(by_subset(g, right)[0]), pct(by_subset(g, wrong)[0]),
                     " / ".join(f"{mass.get(a, np.nan):.3f}" for a in (-2, 2))])
    out += table(["vector", "layer"] + [f"abstain % a={a:+d}" for a in ALPHAS] +
                 ["swing pp [95% CI]", "logit slope (SE)", "swing P1 right", "swing P1 wrong", "digit mass a=-2 / +2"],
                 rows)

    best = max(swings, key=swings.get)
    ctrl = {vec: swing(p2[(p2.vector == vec) & (p2.layer == best)])[0] for vec in ("random", "shuffled")}
    checks = dict(monotone=monotone[best], swing_ge_20pp=swings[best] >= 0.20,
                  controls_lt_5pp=all(abs(c) < 0.05 for c in ctrl.values()))
    verdict = "PASS" if all(checks.values()) else "FAIL"
    line = (f"L* = {best}: swing {pct(swings[best])} pp, random {pct(ctrl['random'])} pp, "
            f"shuffled {pct(ctrl['shuffled'])} pp; " + ", ".join(f"{k} {v}" for k, v in checks.items())
            + f" -> gate {verdict}")
    print(line)
    out += [line, ""]

    out += ["### Phase 1 accuracy under steering (same items)", ""]
    out += table(["layer", "unsteered %", "a=-2 %", "a=+2 %"],
                 [[L, pct(base_p1), *[pct(g[g.alpha == a].is_correct.mean()) for a in (-2, 2)]]
                  for L, g in p1.groupby("layer")])
    return out


def reason_report(run, right, wrong, harness, routes_csv=None):
    df = load(run, "reason")
    route0 = pd.read_csv(routes_csv).set_index("question_index").route.str.lower() if routes_csv else None
    out = []
    for T, dT in df.groupby("temperature"):
        base = dT[dT.alpha == 0]
        out += [f"## Reasoning mode, T={T:g}, {dT.question_index.nunique()} items", ""]
        if len(base):
            by_p1 = [pct(base[base.question_index.isin(s)].is_nota.mean()) for s in (right, wrong)]
            routes = "" if route0 is None else ", routes " + ", ".join(
                f"{k} {v}" for k, v in route0[route0.index.isin(base.question_index)].value_counts().items())
            out += [f"Unsteered (alpha=0): abstention {pct(base.is_nota.mean())} %, P1 right {by_p1[0]} %, "
                    f"P1 wrong {by_p1[1]} %, median reasoning tokens {base.reasoning_tokens.median():.0f}, "
                    f"mean wait {base.wait_count.mean():.2f}, truncated {int(base.truncated.sum())}{routes}", ""]
            if harness is not None:
                h = harness[harness.question_index.isin(base.question_index)]
                out += [f"Harness run on the same items: abstention {pct(h.is_nota.mean())} %, median reasoning tokens "
                        f"{h.reasoning_tokens.median():.0f}, truncated {int(h.truncated.sum())}", ""]
        if route0 is not None:
            out += [f"By-route swings: routes from {Path(routes_csv).name}.", ""]
        rows = []
        for (pos, L, vec), g in dT[dT.alpha != 0].groupby(["position_mode", "layer", "vector"]):
            g = pd.concat([base, g])
            med = g.groupby("alpha").reasoning_tokens.median()
            g = g.assign(log_tokens=np.log1p(g.reasoning_tokens))
            by_route = [] if route0 is None else [f"{pct(by_subset(g, route0.index[route0 == r])[0])} (n={(route0 == r).sum()})"
                                                  for r in ("resolve", "judge")]
            rows.append([pos, L, vec, *rates(g), fmt_swing(swing(g)), fmt_slope(slope(g, "is_nota", sm.families.Binomial())),
                         pct(by_subset(g, right)[0]), pct(by_subset(g, wrong)[0]), *by_route,
                         " / ".join(f"{med.get(a, np.nan):.0f}" for a in (-2, 0, 2)),
                         fmt_slope(slope(g, "log_tokens", sm.families.Gaussian())),
                         " / ".join(f"{g[g.alpha == a].wait_count.mean():.2f}" for a in (-2, 2)),
                         int(g[g.alpha != 0].truncated.sum())])
        if "probe_layer" in dT and dT.probe_layer.notna().any():
            out += probe_table(dT, base)
        out += table(["position", "layer", "vector"] + [f"abstain % a={a:+d}" for a in ALPHAS] +
                     ["swing pp [95% CI]", "logit slope (SE)", "swing P1 right", "swing P1 wrong"] +
                     ["swing resolve", "swing judge"] * (route0 is not None) +
                     ["median tokens a=-2 / 0 / +2", "log-tokens slope (SE)", "mean wait a=-2 / +2",
                      "truncated (steered)"], rows)
    return out


def probe_table(dT, base):
    """Manipulation check: probe score (residual at L' = L + 8 on its own unit vector) at (a) the last prompt token and
    (b) the position producing the first answer token; mean at each alpha, OLS slope on alpha (item-clustered SE),
    monotone non-decreasing in alpha. alpha = 0 rows carry every layer's score; each condition reads its L'."""
    rows = []
    for (pos, L, vec), g in dT[dT.alpha != 0].groupby(["position_mode", "layer", "vector"]):
        Lp = int(g.probe_layer.iloc[0])
        if Lp <= L:
            rows.append([pos, L, vec, Lp] + [f"uninformative (L' = {Lp} <= L)"] + [""] * 6)
            continue
        b0 = base.assign(**{c: base.get(f"{c}_L{Lp}", np.nan) for c in ("probe_prompt_end", "probe_answer_pos")})
        g = pd.concat([b0, g])
        cells = []
        for col in ("probe_prompt_end", "probe_answer_pos"):
            by = g.groupby("alpha")[col].mean().dropna()
            cells += [" / ".join("" if pd.isna(by.get(a)) else f"{by.get(a):.2f}" for a in ALPHAS),
                      fmt_slope(slope(g.dropna(subset=[col]), col, sm.families.Gaussian())),
                      "" if len(by) < 2 else bool(np.all(np.diff(by.to_numpy()) >= 0))]
        rows.append([pos, L, vec, Lp, *cells, int(g[g.alpha != 0].probe_answer_pos.notna().sum())])
    out = ["Manipulation check: probe score at layer L' = L + 8, means at alpha = " + " / ".join(f"{a:+d}" for a in ALPHAS), ""]
    return out + table(["position", "layer", "vector", "probe layer", "(a) prompt end", "(a) slope (SE)",
                        "(a) monotone", "(b) answer position", "(b) slope (SE)", "(b) monotone",
                        "(b) rows (steered)"], rows)


def load(run, stage):
    """Condition files, not <stage>_phases.csv: runs in parallel containers each concatenate only what they see."""
    return pd.concat([pd.read_csv(f) for f in sorted((run / stage).glob("*.csv"))])


def validity_report(run):
    import torch
    v = torch.load(run / "vectors.pt")
    lo, hi = v["auroc_ci"]
    out = [f"## Pre-trace vector validity, L* = {v['L_star']} (5-fold CV within construction), {v['n_group']} per group",
           "", f"Eval AUROC high vs low at L*: {v['validity'][v['L_star']]['auroc']:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] -> "
           f"{'VALID' if v['valid'] else 'NOT VALID'} (lower bound > 0.55)", ""]
    out += table(["layer", "CV AUROC (construction)", "eval AUROC high vs low", "eval AUROC high vs all mixed",
                  "Spearman with log(1 + s_pop)", "abstained - answered (SD)"],
                 [[L, *(f"{x[k]:.3f}" for k in ("cv_auroc", "auroc", "auroc_all_mixed", "spearman_pop", "abstain_d"))]
                  for L, x in v["validity"].items()])
    return out


def main(run, harness=None, routes=None):
    run = Path(run)
    evals = json.loads((run / "eval_items.json").read_text())["question_index"]
    pretrace = (run / "pretrace_items.csv").exists()
    if pretrace:  # Phase 1 correctness of the harness run the labels came from
        p1 = pd.read_csv(run / "pretrace_items.csv")
    else:
        p1 = pd.read_csv(run / "notrace_phases.csv")
        p1 = p1[p1.phase == 1]
    p1 = p1[p1.question_index.isin(evals)]
    right, wrong = p1.question_index[p1.is_correct == 1], p1.question_index[p1.is_correct == 0]
    out = [f"# Steering results: {run.name}", ""]
    if pretrace:
        out += validity_report(run)
    if (run / "gate").exists():
        out += gate_report(run, right, wrong, p1.is_correct.mean())
    if (run / "reason").exists():
        h = None
        if harness:
            h = pd.read_csv(Path(harness) / "phases.csv")
            h = h[h.phase == 2]
        out += reason_report(run, right, wrong, h, routes)
    (run / "STEERING-RESULTS.md").write_text("\n".join(out))
    print(f"-> {run / 'STEERING-RESULTS.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--harness", default=None)
    ap.add_argument("--routes", default=None, help="csv: question_index, route (resolve/judge/mixed)")
    a = ap.parse_args()
    main(a.run, a.harness, a.routes)
