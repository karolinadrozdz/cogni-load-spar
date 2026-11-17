"""Cross-run CFC analysis (tasks/cfc/PLAN.md, "Hypotheses" and "Metrics"), local, on fetched run directories.

    python tasks/cfc/analyze_cfc.py results/cfc-instruct-sft results/cfc-instruct-dpo results/cfc-instruct-rl \\
        results/cfc-think-sft-main results/cfc-think-dpo results/cfc-think-rl31-main [--n-boot 10000] [--json out.json]

Each run dir holds cfc.csv and params.json; params' ref_tag says which checkpoint judged (olmo32b-popqa-<key>). With no
run dirs given: results/cfc-<key> for the six checkpoints. Tests that need a missing run are skipped.

Bootstrap (all tests): one set of item resamples shared by every run (paired across checkpoints); a pair weighs the
product of its two items' resample counts, so the scored set is recomputed in each resample.
- H2 (primary 1): Instruct SFT -> DPO change in stated P accuracy on cue-matched pairs; one-sided p (increase).
- H2b: DPO -> RL in each line, TOST with margin 0.03 (90% interval inside +-0.03 = equivalent).
- H4 (primary 2): stated self-advantage minus internal self-advantage, pooled (mean) over the three stages; one-sided
  p (> 0). Per stage, self-advantage = 1/2 {[S_M - X_N] + [S_N - X_M]} on one pair set: both models answered both
  items and every judge's internal choice is defined; each half on the pairs its target scores (exactly one right).
- Holm over the two primary p-values. Everything else is descriptive.
- Surface-cue regression (P, scored; random pairs, and matched pairs for their residual popularity difference):
  logistic, no intercept (the design is antisymmetric in the two items), outcome "item i is the correct one" = "the
  stated choice is correct" when the choice is i; predictors (each divided by its SD; constant ones dropped): stated
  preference log-odds for i, delta log s_pop, delta relation base rate, delta question length (characters). The base
  rate is the target's accuracy on the item's relation leaving out BOTH items of the pair (leave-one-out alone would
  leak the outcome on same-relation pairs); on same-relation pairs the delta is therefore 0.
Every result is also given without pairs where a shown answer (P: the target's answer) is the type-mismatched
distractor (choice_4 of the item file; H4: either model's answer).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tasks.cfc import (CHECKPOINTS, PARTNER, REF_PREFIX, STAGES, boot_counts, condition_metrics, tag_key,  # noqa: E402
                       wmean_boot)

MARGIN = 0.03
N_ITEMS = 1000


def load(dirs):
    runs, shas = {}, set()
    for d in dirs:
        d = Path(d)
        if not (d / "cfc.csv").exists():
            print(f"missing {d / 'cfc.csv'}: skipped")
            continue
        assert (d / "params.json").exists(), f"{d} has no params.json (run incomplete?)"
        p = json.loads((d / "params.json").read_text())
        assert p.get("n_pairs") == 0, f"{d.name} ran n_pairs={p.get('n_pairs')}: pilots are not part of the analysis"
        assert set("PSX") <= set(p.get("conditions", "")), f"{d.name} ran conditions {p.get('conditions')}, not PSX"
        assert "score_error" not in p, f"{d.name} has a score_error: re-score from raw_generations.csv first"
        key = tag_key(p["ref_tag"])
        assert key not in runs, f"two runs judged by {key}"
        assert not p.get("cross_ref_tag") or tag_key(p["cross_ref_tag"]) == PARTNER[key]
        shas.add(p.get("pairs_sha256"))
        runs[key] = dict(dir=d, params=p, t=pd.read_csv(d / "cfc.csv"))
    assert len(shas) <= 1, f"runs used different pair sets: {shas}"
    return runs


def scored(t, cond, kind="all", drop=False, col="target_distractor"):
    s = t[(t.condition == cond) & (t.scored == 1)]
    if kind != "all":
        s = s[s.kind == kind]
    return s[s[col] == 0] if drop else s


def wb(C, s, y):
    return wmean_boot(C, s.i.to_numpy(), s.j.to_numpy(), np.asarray(y, float))


def p_le0(b):
    """One-sided bootstrap p for 'statistic > 0'."""
    b = b[np.isfinite(b)]
    return (1 + (b <= 0).sum()) / (len(b) + 1) if len(b) else np.nan


def ci(b, lo=2.5, hi=97.5):
    b = b[np.isfinite(b)]
    return tuple(np.percentile(b, [lo, hi])) if len(b) else (np.nan, np.nan)


def acc(t, cond, kind, C, drop, col="chosen_correct"):
    s = scored(t, cond, kind, drop)
    return s[col].mean(), wb(C, s, s[col]), len(s)


def stage_diff(runs, a, b, C, drop, cond="P", kind="matched"):
    """Later (b) minus earlier (a) stated accuracy, same item resamples."""
    pa, ba, na = acc(runs[a]["t"], cond, kind, C, drop)
    pb, bb, nb = acc(runs[b]["t"], cond, kind, C, drop)
    return pb - pa, bb - ba, na, nb


def h4_stage(runs, m, n, C, drop):
    """Stated and internal self-advantage for one stage (m = Instruct, n = Think), and their difference, with
    bootstrap distributions."""
    rows = {}
    for key, cond in ((m, "S"), (m, "X"), (n, "S"), (n, "X")):
        t = runs[key]["t"]
        rows[(key, cond)] = t[t.condition == cond].set_index("pair_id")
    common = sorted(set.intersection(*(set(r.index) for r in rows.values())))
    ok = pd.Series(True, index=common)
    for r in rows.values():
        r = r.loc[common]
        ok &= r.int_choice_i.notna()
        ok &= r[["self_first_i", "self_first_j", "partner_first_i", "partner_first_j"]].notna().all(axis=1)
        if drop:
            ok &= r.any_distractor == 0
    base = ok[ok].index
    S_M, X_M, S_N, X_N = (rows[k].loc[base] for k in ((m, "S"), (m, "X"), (n, "S"), (n, "X")))
    assert (S_M.scored == X_N.scored).all() and (S_N.scored == X_M.scored).all(), "targets disagree on correctness"
    h1, h2 = S_M.scored == 1, S_N.scored == 1   # target M / target N
    out = dict(n_base=len(base), n_target_m=int(h1.sum()), n_target_n=int(h2.sum()),
               n_answers_differ=int((S_M.sx_identical == False).sum()))  # noqa: E712
    parts = {}
    for what, col in (("stated", "chosen_correct"), ("internal", "int_correct")):
        d1 = (S_M[col] - X_N[col])[h1]
        d2 = (S_N[col] - X_M[col])[h2]
        point = 0.5 * (d1.mean() + d2.mean())
        boot = 0.5 * (wb(C, S_M[h1], d1) + wb(C, S_N[h2], d2))
        parts[what] = (point, boot)
        out[f"sa_{what}"] = point
        out[f"sa_{what}_acc_terms"] = dict(MM=S_M[col][h1].mean(), NM=X_N[col][h1].mean(), NN=S_N[col][h2].mean(),
                                          MN=X_M[col][h2].mean())
    diff = parts["stated"][0] - parts["internal"][0]
    bdiff = parts["stated"][1] - parts["internal"][1]
    out.update(diff=diff, diff_lo=ci(bdiff)[0], diff_hi=ci(bdiff)[1])
    return out, bdiff


def holm(ps):
    """Holm-adjusted p-values (same order as ps)."""
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[k]))
        adj[k] = run
    return adj


def logit_fit(X, y, iters=100):
    """Logistic regression by IRLS (no intercept unless X has one): coefficients, SEs, two-sided Wald p."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        H = X.T @ (X * (mu * (1 - mu))[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    p = [math.erfc(abs(b / s) / math.sqrt(2)) for b, s in zip(beta, se)]
    return beta, se, np.array(p)


def pair_base_rates(per_item, s):
    """Each item's relation base rate (target accuracy) leaving out BOTH items of its pair."""
    tot, cnt = per_item.groupby("prop").c.sum(), per_item.groupby("prop").c.count()
    all_sum, all_n = per_item.c.sum(), len(per_item)
    c = dict(zip(per_item.x, per_item.c))
    out = []
    for x, y, px, py in ((s.i, s.j, s.prop_i, s.prop_j), (s.j, s.i, s.prop_j, s.prop_i)):
        same = (px == py).to_numpy()
        cx, cy = x.map(c).to_numpy(float), y.map(c).to_numpy(float)
        num = px.map(tot).to_numpy(float) - cx - np.where(same, cy, 0)
        den = px.map(cnt).to_numpy(float) - 1 - same
        fallback = (all_sum - cx - cy) / (all_n - 2)  # a relation with no other item
        out.append(np.where(den > 0, num / np.where(den > 0, den, 1), fallback))
    return out


def cue_regression(t, drop, kind="random"):
    """P, scored pairs of one kind: correct_i ~ stated log-odds_i + delta log s_pop + delta base rate + delta length."""
    p = t[t.condition == "P"]
    per_item = pd.concat([p[["i", "correct_i", "prop_i"]].set_axis(["x", "c", "prop"], axis=1),
                          p[["j", "correct_j", "prop_j"]].set_axis(["x", "c", "prop"], axis=1)]).drop_duplicates("x")
    s = scored(t, "P", kind, drop)
    base_i, base_j = pair_base_rates(per_item, s)
    pref = s.pref_i.clip(1e-6, 1 - 1e-6)
    X = pd.DataFrame(dict(stated_logodds=np.log(pref / (1 - pref)),
                          d_log_spop=np.log(s.s_pop_i) - np.log(s.s_pop_j),
                          d_base_rate=base_i - base_j,
                          d_qlen=s.qlen_i - s.qlen_j))
    sd = X.std(ddof=0)
    X = X.loc[:, sd > 1e-12] / sd[sd > 1e-12]  # matched pairs: d_base_rate is 0 by construction
    beta, se, pv = logit_fit(X.to_numpy(float), s.correct_i.to_numpy(float))
    return pd.DataFrame(dict(coef=beta, se=se, p=pv), index=X.columns), len(s)


def fmt(x, lo=None, hi=None, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    s = f"{x:.{d}f}"
    return s + (f" [{lo:.{d}f}, {hi:.{d}f}]" if lo is not None and np.isfinite(lo) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    runs = load(a.runs or [ROOT / "results" / f"cfc-{k}" for k in CHECKPOINTS])
    if not runs:
        raise SystemExit("no runs")
    C = boot_counts(N_ITEMS, a.n_boot, a.seed)
    report = dict(n_boot=a.n_boot, runs={k: str(r["dir"]) for k, r in runs.items()})
    order = [k for k in CHECKPOINTS if k in runs]

    for drop in (False, True):
        tag = "no_distractor" if drop else "all"
        res = report.setdefault(tag, {})
        print(f"\n================ {'WITHOUT distractor-shown pairs' if drop else 'ALL pairs'} "
              f"(item bootstrap, {a.n_boot} resamples) ================")
        # per run, descriptive
        table = []
        for k in order:
            t = runs[k]["t"]
            if drop:
                t = t[t.target_distractor == 0]
            row = dict(run=k)
            for cond in ("P", "S", "X"):
                if not (t.condition == cond).any():
                    continue
                for kind in (("matched", "random", "all") if cond == "P" else ("all",)):
                    pt, b, nn = acc(t, cond, kind, C, False)
                    row[f"{cond}_{kind}"] = fmt(pt, *ci(b))
                    row[f"n_{cond}_{kind}"] = nn
                pi, bi, _ = acc(t, cond, "all", C, False, col="int_correct")
                row[f"{cond}_internal"] = fmt(pi, *ci(bi))
                m = condition_metrics(t[t.condition == cond])
                res.setdefault("per_run", {}).setdefault(k, {})[cond] = {
                    kk: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for kk, v in m.items()}
                if cond == "P":
                    row.update(P_agree=fmt(m["agreement"]), P_agree_cue_conflict=fmt(m["agreement_cue_conflict"]),
                               P_follows_cue=fmt(m["follows_cue"]), P_efficiency=fmt(m["readout_efficiency"]),
                               P_pos_bias=fmt(m["position_bias"]), P_ab_mass=fmt(m["ab_mass"]), P_ties=fmt(m["ties"]))
            table.append(row)
        df = pd.DataFrame(table).set_index("run")
        with pd.option_context("display.width", 250, "display.max_columns", 50):
            print("\nStated (chosen_correct) and internal accuracy, 95% CI; P diagnostics:")
            print(df[[c for c in df.columns if not c.startswith("n_") and not c.startswith("P_") or c in
                      ("P_matched", "P_random", "P_all", "P_internal")]].to_string())
            print(df[[c for c in df.columns if c.startswith("P_") and c not in
                      ("P_matched", "P_random", "P_all", "P_internal")] + [c for c in df.columns if c.startswith("n_")]]
                  .to_string())

        # H2 and H2b
        ps = {}
        if {"instruct-sft", "instruct-dpo"} <= set(runs):
            d, b, na, nb = stage_diff(runs, "instruct-sft", "instruct-dpo", C, drop)
            ps["H2"] = p_le0(b)
            res["H2"] = dict(diff=d, lo=ci(b)[0], hi=ci(b)[1], p_one_sided=ps["H2"], n_sft=na, n_dpo=nb)
            print(f"\nH2  Instruct SFT->DPO, stated P accuracy on matched pairs: {fmt(d, *ci(b))}, one-sided p "
                  f"{ps['H2']:.4f} (n scored {na} / {nb})")
        else:
            print("\nH2  needs instruct-sft and instruct-dpo: skipped")
        for line, (dpo, rl) in (("Instruct", ("instruct-dpo", "instruct-rl")), ("Think", ("think-dpo", "think-rl31-main"))):
            if {dpo, rl} <= set(runs):
                d, b, _, _ = stage_diff(runs, dpo, rl, C, drop)
                lo90, hi90 = ci(b, 5, 95)
                bf = b[np.isfinite(b)]
                p_tost = max((bf <= -MARGIN).mean(), (bf >= MARGIN).mean())
                res[f"H2b_{line}"] = dict(diff=d, lo90=lo90, hi90=hi90, p_tost=p_tost,
                                          equivalent=bool(-MARGIN < lo90 and hi90 < MARGIN))
                print(f"H2b {line} DPO->RL: {d:+.3f}, 90% CI [{lo90:+.3f}, {hi90:+.3f}], TOST p {p_tost:.4f}, "
                      f"{'equivalent' if res[f'H2b_{line}']['equivalent'] else 'NOT shown equivalent'} "
                      f"(margin {MARGIN})")
        print("Stage differences (stated P accuracy), descriptive:")
        for line, keys in (("Instruct", ("instruct-sft", "instruct-dpo", "instruct-rl")),
                           ("Think", ("think-sft-main", "think-dpo", "think-rl31-main"))):
            for x, y in zip(keys, keys[1:]):
                if {x, y} <= set(runs):
                    cells = []
                    for kind in ("matched", "random", "all"):
                        d, b, _, _ = stage_diff(runs, x, y, C, drop, kind=kind)
                        cells.append(f"{kind} {fmt(d, *ci(b))}")
                        res.setdefault("stage_diffs", {})[f"{x}->{y}:{kind}"] = dict(diff=d, lo=ci(b)[0], hi=ci(b)[1])
                    print(f"  {x} -> {y}: " + "; ".join(cells))

        # H4
        stage_boots, stage_rows = [], []
        for stage, m, n in STAGES:
            if {m, n} <= set(runs) and all((runs[k]["t"].condition == "X").any() for k in (m, n)):
                o, bd = h4_stage(runs, m, n, C, drop)
                stage_boots.append(bd)
                stage_rows.append(dict(stage=stage, **{k: v for k, v in o.items() if not k.endswith("_terms")}))
                res.setdefault("H4_stages", {})[stage] = o
        if stage_rows:
            print("\nH4 per stage (descriptive): self-advantage = 1/2{[S_M - X_N] + [S_N - X_M]}")
            print(pd.DataFrame(stage_rows).round(4).to_string(index=False))
        if len(stage_boots) == 3:
            pooled = float(np.mean([r["diff"] for r in stage_rows]))
            bp_ = np.mean(stage_boots, axis=0)
            ps["H4"] = p_le0(bp_)
            res["H4"] = dict(diff=pooled, lo=ci(bp_)[0], hi=ci(bp_)[1], p_one_sided=ps["H4"])
            print(f"H4  pooled stated - internal self-advantage: {fmt(pooled, *ci(bp_))}, one-sided p {ps['H4']:.4f}")
        else:
            print("H4  needs all six runs with the X condition: skipped")
        if len(ps) == 2:
            adj = holm(np.array([ps["H2"], ps["H4"]]))
            res["holm"] = dict(H2=adj[0], H4=adj[1])
            print(f"Holm-adjusted: H2 p {adj[0]:.4f}, H4 p {adj[1]:.4f}")

        # surface-cue regression
        for kind in ("random", "matched"):
            print(f"\nSurface-cue regression (P, {kind} pairs, scored; standardized, no intercept):")
            for k in order:
                t = runs[k]["t"]
                if not (t.condition == "P").any():
                    continue
                try:
                    coefs, nn = cue_regression(t, drop, kind)
                    res.setdefault(f"regression_{kind}", {})[k] = dict(
                        n=nn, **{c: dict(coef=r.coef, se=r.se, p=r.p) for c, r in coefs.iterrows()})
                    print(f"  {k:16} n={nn:4d}  " + "  ".join(f"{c} {r.coef:+.3f} (se {r.se:.3f}, p {r.p:.3g})"
                                                              for c, r in coefs.iterrows()))
                except Exception as e:  # e.g. too few pairs
                    print(f"  {k:16} regression failed: {e!r}")
    if a.json:
        a.json.write_text(json.dumps(report, indent=2, default=lambda x: float(x) if np.isscalar(x) else str(x)))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
