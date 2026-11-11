"""Confidence direction per layer (every 4th, covering the gate sweep) from a no-trace run, as in the paper's released
notebook (Kumaran et al. 2026, Cell B calculate_steering_vectors), its controls, and the eval set.

    python build_vector.py results/steer-<model>-<tag>     # reads notrace_phases.csv, construction_phases.csv, hiddens.pt

The notrace stage of steering_modal.py runs this at its end; run it locally only to rebuild or inspect.
Construction set: no-trace Phase 2 prompts on the Phase 0 questions. margin = max(P1..P4) - P5, P = softmax over the
five digit log-probs / tau, tau fitted on Phase 0. Among correct trials, walk the 75 highest-margin (descending) and the
75 lowest (ascending), keeping a trial only while fewer than int(25 * 0.28) = 7 of its group chose the same option,
until 25 each. v_L = mean(high) - mean(low), stored as a unit vector with scale_L = 0.03 * mean ||h_L|| over all
construction trials. Controls: `random` (one unit Gaussian, seed 0) and `shuffled` (two pseudo-groups of 12-13 real high
plus 13-12 real low items, seed 0, with its component along v_L removed), same scale_L. Eval set: 150 correct + 150 incorrect Phase 1 items by the unsteered
no-trace answer, seed 0, stored in shuffled order so a prefix is mixed. Writes vectors.pt and eval_items.json.

Pre-trace runs (pretrace_items.csv present; the pretrace stage writes it with eval_items.json and pretrace_hiddens.pt):
build_pretrace. Labels from the harness run's Phase 1 k-sample consistency (sample 1 is the Phase 1 answer):
high = k/k, low = 1..k-1 and Phase 1 correct, 70 each (fewer, >= 50, if the cap cannot fill 70) drawn at random
(seed 0) from the non-eval items with a cap of int(0.28 n) per gt_position; same vector, controls and scale recipe;
adds L* (5-fold CV AUROC on the construction items), held-out validity per layer and the bootstrap CI to vectors.pt.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyze import LP, fit_tau


def select(cands, n_group, cap, by="final_answer"):
    """Walk candidates in order, keeping a trial while its option (`by`) has fewer than `cap` in the group."""
    counts, picked = {k: 0 for k in range(1, 5)}, []
    for q, c in zip(cands.question_index, cands[by].astype(int)):
        if counts[c] < cap and len(picked) < n_group:
            picked.append(int(q))
            counts[c] += 1
    return picked, counts


def balanced_split(high, low, seed=0):
    """Two pseudo-groups, each about half real high and half real low (12 + 13 and 13 + 12 for 25), so their
    difference carries no real signal."""
    rng = np.random.default_rng(seed)
    hp, lp, k = rng.permutation(high), rng.permutation(low), len(high) // 2
    return np.concatenate([hp[:k], lp[k:]]).tolist(), np.concatenate([hp[k:], lp[:k]]).tolist()


def directions(h, layers, pos, high, low, scale_rows, norm_frac=0.03):
    """real = unit mean(high) - mean(low) per layer; controls `random` and `shuffled` (orthogonal to real);
    scale_L = norm_frac * mean ||h_L|| over scale_rows. h: (layers, items, hidden), pos: question_index -> item."""
    rows = lambda qs: [pos[q] for q in qs]
    diff = lambda hi, lo: {L: h[k, rows(hi)].mean(0) - h[k, rows(lo)].mean(0) for k, L in enumerate(layers)}
    unit = lambda d: {L: v / v.norm() for L, v in d.items()}
    pseudo = balanced_split(high, low)
    noise = torch.randn(h.shape[-1], generator=torch.Generator().manual_seed(0))
    raw = diff(high, low)
    real = unit(raw)
    # orthogonalised so the control shares no component with the real direction
    shuffled = unit({L: d - (d @ real[L]) * real[L] for L, d in diff(*pseudo).items()})
    return dict(layers=layers, real=real, shuffled=shuffled, random={L: noise / noise.norm() for L in layers},
                scale={L: norm_frac * float(h[k, scale_rows].norm(dim=-1).mean()) for k, L in enumerate(layers)},
                high=high, low=low, pseudo=pseudo), raw


def print_vectors(vectors, raw):
    print("layer  |v_raw|   scale  cos(real, shuffled)")
    for L in vectors["layers"]:
        print(f"{L:5d} {raw[L].norm():8.2f} {vectors['scale'][L]:7.2f}  "
              f"{torch.cosine_similarity(vectors['real'][L], vectors['shuffled'][L], 0):.2f}")


def write_evals(run, p1, n_eval_cell):
    """eval_items.json: n_eval_cell correct + n_eval_cell incorrect Phase 1 items, seed 0, shuffled order."""
    rng = np.random.default_rng(0)
    cells = {k: p1[p1.is_correct == k].question_index.to_numpy() for k in (1, 0)}
    picked = {k: rng.choice(v, min(n_eval_cell, len(v)), replace=False) for k, v in cells.items()}
    evals = [int(q) for q in rng.permutation(np.concatenate([picked[1], picked[0]]))]
    (run / "eval_items.json").write_text(json.dumps(dict(
        question_index=evals, n_correct=len(picked[1]), n_incorrect=len(picked[0])), indent=1))
    print(f"eval items: {len(picked[1])} correct + {len(picked[0])} incorrect"
          + ("" if len(picked[0]) == n_eval_cell else f" (only {len(cells[0])} incorrect available)"))
    return evals


def pretrace_items(harness, p1_questions):
    """One row per Phase 1 item of the harness run: k_correct = how many of its k_answers equal gt_position
    (label high = all, mixed = some, none = 0), Phase 1 correctness, Phase 2 is_nota, s_pop."""
    df = pd.read_csv(Path(harness) / "phases.csv")
    p1 = df[df.phase == 1].set_index("question_index")
    assert (p1.question == p1_questions.question.reindex(p1.index)).all(), "harness Phase 1 questions differ from data"
    k = [str(a).split("|") for a in p1.k_answers]
    items = pd.DataFrame(dict(question_index=p1.index, question=p1.question, gt_position=p1.gt_position,
                              is_correct=p1.is_correct, n_samples=[len(a) for a in k],
                              k_correct=[sum(x == str(gt) for x in a) for a, gt in zip(k, p1.gt_position)],
                              s_pop=p1_questions.s_pop.reindex(p1.index).to_numpy(),
                              is_nota_p2=df[df.phase == 2].set_index("question_index").is_nota.reindex(p1.index)))
    items["label"] = np.where(items.k_correct == items.n_samples, "high",
                              np.where(items.k_correct > 0, "mixed", "none"))
    return items.reset_index(drop=True)


def groups_under_cap(con, n_group, min_group, max_choice):
    """Largest n <= n_group (and >= min_group) at which random draws (seed 0) fill both groups under the cap of
    int(max_choice * n) per gt_position. high = k/k consistent; low = mixed and Phase 1 correct."""
    pools = dict(high=con[con.label == "high"], low=con[(con.label == "mixed") & (con.is_correct == 1)])
    for n in range(n_group, min_group - 1, -1):
        rng, cap, out = np.random.default_rng(0), int(n * max_choice), {}
        for name, pool in pools.items():
            out[name] = select(pool.iloc[rng.permutation(len(pool))], n, cap, by="gt_position")
        if all(len(p) == n for p, _ in out.values()):
            return n, cap, out
    raise AssertionError(f"cannot fill {min_group} per group under the cap: pools "
                         f"{ {k: len(v) for k, v in pools.items()} }")


def is_low(df):
    return (df.label == "mixed") & (df.is_correct == 1)


def cv_auroc(h, pos, con, layers, folds=5):
    """Mean held-out AUROC per layer, folds over construction items (seed 0): direction = mean(high) - mean(low)
    over the training folds' high and low items, scored on the held-out fold's high and low items."""
    from sklearn.metrics import roc_auc_score
    fold = np.random.default_rng(0).permutation(len(con)) % folds
    lab = con[(con.label == "high") | is_low(con)]
    lab_fold = fold[np.flatnonzero((con.label == "high") | is_low(con))]
    y = (lab.label == "high").to_numpy()
    rows = np.array([pos[q] for q in lab.question_index])
    out = {L: [] for L in layers}
    for f in range(folds):
        tr, te = lab_fold != f, lab_fold == f
        for k, L in enumerate(layers):
            v = h[k, rows[tr & y]].mean(0) - h[k, rows[tr & ~y]].mean(0)
            out[L].append(roc_auc_score(y[te], (h[k, rows[te]] @ v).numpy()))
    return {L: float(np.mean(a)) for L, a in out.items()}


def bootstrap_auroc(y, score, reps=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    idx = np.random.default_rng(seed).integers(0, len(y), (reps, len(y)))
    boots = [roc_auc_score(y[i], score[i]) for i in idx if 0 < y[i].sum() < len(i)]
    return tuple(float(x) for x in np.quantile(boots, [0.025, 0.975]))


def build_pretrace(run, n_group=70, min_group=50, max_choice=0.28, norm_frac=0.03, valid_lower=0.55):
    """Pre-trace vector from the construction items (Phase 1 items not in eval_items.json): high (k/k consistent)
    minus low (mixed and Phase 1 correct), random draws (seed 0) capped per gt_position. L* = the sweep layer with the
    best 5-fold CV AUROC within the construction set. Validity on the eval items, per layer: AUROC of the projection
    for high vs low (primary) and high vs all mixed, Spearman with log(1 + s_pop), abstained - answered projection in
    SD units; bootstrap 95% CI of the primary AUROC at L*, valid iff its lower bound > valid_lower."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    run = Path(run)
    items = pd.read_csv(run / "pretrace_items.csv")
    evals = json.loads((run / "eval_items.json").read_text())["question_index"]
    con = items[~items.question_index.isin(evals)]
    n, cap, groups = groups_under_cap(con, n_group, min_group, max_choice)
    (high, high_counts), (low, low_counts) = groups["high"], groups["low"]

    hid = torch.load(run / "pretrace_hiddens.pt")
    pos = {q: j for j, q in enumerate(hid["question_index"])}
    h = hid["h"].float()
    vectors, raw = directions(h, hid["layers"], pos, high, low, [pos[q] for q in con.question_index], norm_frac)
    cv = cv_auroc(h, pos, con, hid["layers"])
    L_star = max(hid["sweep"], key=cv.get)

    ev = items.set_index("question_index").loc[evals]
    rows = [pos[q] for q in evals]
    primary = ((ev.label == "high") | is_low(ev)).to_numpy()
    mixed = ev.label.isin(["high", "mixed"]).to_numpy()
    nota = ev.is_nota_p2.to_numpy() == 1
    y = (ev.label == "high").to_numpy()
    validity, proj = {}, {}
    for k, L in enumerate(hid["layers"]):
        p = proj[L] = (h[k, rows] @ vectors["real"][L]).numpy()
        validity[L] = dict(cv_auroc=cv[L], auroc=float(roc_auc_score(y[primary], p[primary])),
                           auroc_all_mixed=float(roc_auc_score(y[mixed], p[mixed])),
                           spearman_pop=float(spearmanr(p, np.log1p(ev.s_pop)).statistic),
                           abstain_d=float((p[nota].mean() - p[~nota].mean()) / p.std()))
    ci = bootstrap_auroc(y[primary], proj[L_star][primary])
    vectors.update(sweep=hid["sweep"], n_group=n, high_counts=high_counts, low_counts=low_counts, validity=validity,
                   L_star=L_star, auroc_ci=ci, valid=ci[0] > valid_lower)
    torch.save(vectors, run / "vectors.pt")

    print(f"construction {len(con)} items: high {(con.label == 'high').sum()}, mixed {(con.label == 'mixed').sum()} "
          f"(Phase 1 correct {is_low(con).sum()}), none {(con.label == 'none').sum()}")
    print(f"n per group = {n} (cap {cap} per gt_position): high {high_counts}, low {low_counts}")
    print_vectors(vectors, raw)
    print(f"validity on {len(evals)} eval items ({primary.sum()} high/low, {mixed.sum()} high/mixed, "
          f"{int(nota.sum())} abstained):")
    print("layer  CV AUROC  AUROC high/low  AUROC high/mixed  Spearman log(1+s_pop)  abstained-answered (SD)")
    for L in hid["layers"]:
        v = validity[L]
        print(f"{L:5d} {v['cv_auroc']:9.3f} {v['auroc']:14.3f} {v['auroc_all_mixed']:16.3f} {v['spearman_pop']:20.3f} "
              f"{v['abstain_d']:22.3f}" + ("   <- L*" if L == L_star else ""))
    print(f"L* = {L_star} (best 5-fold CV AUROC among sweep layers {hid['sweep']}); eval AUROC high/low at L* "
          f"{validity[L_star]['auroc']:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}] -> "
          f"{'VALID' if vectors['valid'] else 'NOT VALID'} (lower bound > {valid_lower})")


def build(run, n_group=25, n_eval_cell=150, multiplier=3, max_choice=0.28, norm_frac=0.03):
    run = Path(run)
    df = pd.read_csv(run / "notrace_phases.csv")
    p0, p1 = df[df.phase == 0], df[df.phase == 1].reset_index(drop=True)
    tau = fit_tau(p0, p0.is_correct.to_numpy())

    con = pd.read_csv(run / "construction_phases.csv")
    z = con[LP + ["logprob_nota"]].to_numpy() / tau
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    con["p_best"], con["p5"] = p[:, :4].max(1), p[:, 4]
    con["margin"] = con.p_best - con.p5

    ranked = con[con.is_correct == 1].sort_values("margin", ascending=False)
    cap = int(n_group * max_choice)
    high, high_counts = select(ranked.head(n_group * multiplier), n_group, cap)
    low, low_counts = select(ranked.tail(n_group * multiplier).iloc[::-1], n_group, cap)
    assert not set(high) & set(low), f"high and low groups overlap: only {len(ranked)} correct construction trials"

    hid = torch.load(run / "hiddens.pt")
    pos = {q: j for j, q in enumerate(hid["question_index"])}
    h = hid["h"].float()  # (layers, items, hidden)
    g = lambda qs, col: float(con.set_index("question_index").loc[qs, col].mean())
    vectors, raw = directions(h, hid["layers"], pos, high, low, list(range(h.shape[1])))
    vectors.update(sweep=hid["sweep"], tau=tau, high_counts=high_counts, low_counts=low_counts,
                   p_best_high=g(high, "p_best"), p5_high=g(high, "p5"), p_best_low=g(low, "p_best"),
                   p5_low=g(low, "p5"))
    torch.save(vectors, run / "vectors.pt")

    print(f"tau {tau:.2f}; construction accuracy {con.is_correct.mean():.3f}, abstention {con.is_nota.mean():.3f}, "
          f"{len(ranked)} correct; P1 accuracy {p1.is_correct.mean():.3f}")
    print(f"selected {len(high)} high / {len(low)} low (cap {cap} per option); chosen options high {high_counts}, "
          f"low {low_counts}")
    print(f"calibrated P(best) / P(5): high {vectors['p_best_high']:.3f} / {vectors['p5_high']:.3f}, "
          f"low {vectors['p_best_low']:.3f} / {vectors['p5_low']:.3f}")
    print_vectors(vectors, raw)
    write_evals(run, p1, n_eval_cell)


if __name__ == "__main__":
    (build_pretrace if (Path(sys.argv[1]) / "pretrace_items.csv").exists() else build)(sys.argv[1])
