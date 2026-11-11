"""Local CPU checks of the steering pipeline on Qwen/Qwen3-0.6B (a few PopQA items, tiny budgets).

    python test_steering_local.py            # hook checks + Olmo-3 </think> detection
    python test_steering_local.py e2e        # also notrace -> build_vector -> gate -> reason -> analyze_steering,
                                             # and pretrace (fake harness run) -> build_pretrace -> reason -> analyze
"""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

import steering_modal as sm
from abstention_modal import answer_prompt, split_think

MODEL = "Qwen/Qwen3-0.6B"


def test_olmo_think_detection():
    """The tracker flips on exactly the token that completes "</think>", where split_think puts k."""
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-32B-Think")
    prompt = tok("Question: 2+1?\n<think>").input_ids
    gen = tok("So the answer is three, done.</think>\n\n3", add_special_tokens=False).input_ids
    print("Olmo-3 pieces:", [tok.decode([t]) for t in gen])
    tracker = sm.ThinkTracker(tok, len(prompt), 1, [tok.eos_token_id])
    flipped = None
    for m in range(1, len(gen) + 1):
        tracker(torch.tensor([prompt + gen[:m]]), None)
        if tracker.closed[0] and flipped is None:
            flipped = m
    k = split_think(gen, tok, True, "stop")[0]
    print(f"tracker closed after token {flipped}; split_think k = {k}")
    assert flipped == k


def test_olmo3_arch():
    """Same checks on a tiny random-weight Olmo3ForCausalLM (mixed sliding/full attention) with the real tokenizer."""
    from transformers import AutoConfig, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-32B-Think", padding_side="left")
    cfg = AutoConfig.from_pretrained("allenai/Olmo-3-32B-Think")
    cfg.update(dict(num_hidden_layers=8, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=16, layer_types=cfg.layer_types[:8], sliding_window=8))
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    texts = phase2_texts(tok, 3, "native")
    v = torch.randn(64)
    base = sm.generate(model, tok, texts, 24, batch=3)
    for mode in ("start", "think", "post"):
        zero = sm.generate(model, tok, texts, 24, batch=3, steer=(4, v, 0, mode))
        assert [g["ids"] for g in zero] == [g["ids"] for g in base], mode
        pushed = sm.generate(model, tok, texts, 24, batch=3, steer=(4, v, 50, mode))
        print(f"Olmo3 {mode}: alpha 0 identical; steered {[g['steered'] for g in zero]} of "
              f"{[len(g['ids']) for g in zero]}; alpha 50 changed {sum(p['ids'] != b['ids'] for p, b in zip(pushed, base))}/3")


def phase2_texts(tok, n, thinking="on"):
    data = sm.load_items("popqa")
    return [sm.render(tok, answer_prompt(2, *sm.item(data, 1, i)[:2]), thinking, False) for i in range(n)]


def test_alpha0_identity(model, tok):
    # two prompts start a short trace so they close </think> and hit EOS within the budget; one truncates
    texts = phase2_texts(tok, 3)
    texts = [texts[0], texts[1] + "<think>\nI recall this one; it is option 2.\n", texts[2] + "<think>\nShort answer: I know this, it is option"]
    v = torch.randn(model.config.hidden_size, generator=torch.Generator().manual_seed(1))
    base = sm.generate(model, tok, texts, 80, batch=3)
    for mode in ("start", "think", "post"):
        zero = sm.generate(model, tok, texts, 80, batch=3, steer=(14, v, 0, mode))
        assert [g["ids"] for g in zero] == [g["ids"] for g in base], mode
        pushed = sm.generate(model, tok, texts, 80, batch=3, steer=(14, v, 8, mode))
        changed = sum(g["ids"] != b["ids"] for g, b in zip(pushed, base))
        for g in zero:
            closed = "</think>" in tok.decode(g["ids"], skip_special_tokens=True)
            c = split_think(g["ids"], tok, True, g["finish_reason"])[0] if closed else None
            print(f"  {mode:5s}: steered {g['steered']:3d} of {len(g['ids'])} generated "
                  f"(</think> completed at token {c}, {g['finish_reason']})")
        print(f"{mode}: alpha 0 identical to no hook; alpha 8 changed {changed}/3 generations")


def test_selection():
    """Notebook selection at full size on synthetic trials skewed to option 1, and the balanced shuffle."""
    import numpy as np
    from build_vector import balanced_split, select
    rng = np.random.default_rng(1)
    trials = pd.DataFrame(dict(question_index=range(400), final_answer=rng.choice([1, 2, 3, 4], 400, p=[.6, .2, .1, .1]),
                               margin=rng.random(400))).sort_values("margin", ascending=False)
    high, high_counts = select(trials.head(75), 25, int(25 * 0.28))
    low, low_counts = select(trials.tail(75).iloc[::-1], 25, int(25 * 0.28))
    print("chosen options high", high_counts, "low", low_counts)
    assert len(high) == len(low) == 25 and max(list(high_counts.values()) + list(low_counts.values())) == 7
    # walked in margin order: a candidate is skipped only if its option was already full
    top = trials.head(75)
    skipped = top[~top.question_index.isin(high) & (top.margin > top.margin[top.question_index.isin(high)].min())]
    assert all(high_counts[c] == 7 for c in skipped.final_answer)
    real_high = [len(set(g) & set(high)) for g in balanced_split(high, low)]
    print("real high items per shuffled pseudo-group:", real_high)
    assert sorted(real_high) == [12, 13] and all(len(g) == 25 for g in balanced_split(high, low))


def test_e2e(model, tok):
    import analyze_steering
    run = Path(tempfile.mkdtemp()) / "steer-Qwen3-0.6B-local"
    run.mkdir(parents=True)
    # Qwen3-0.6B abstains on ~90% of Phase 2 prompts, so ~20 of 300 construction trials are correct: groups of 4
    cfg = dict(model=MODEL, thinking="on", questions="popqa", n=300, batch=16)
    sm.run_notrace(model, tok, run, cfg, sweep=[10, 14], n_group=4, n_eval_cell=6)
    vecs = torch.load(run / "vectors.pt")
    assert max(list(vecs["high_counts"].values()) + list(vecs["low_counts"].values())) <= 1
    n, k = len(vecs["high"]), len(vecs["high"]) // 2
    assert sorted(len(set(g) & set(vecs["high"])) for g in vecs["pseudo"]) == [k, n - k]
    sm.run_gate(model, tok, run, cfg)
    gate = pd.read_csv(run / "gate_phases.csv")
    p5 = gate[(gate.phase == 2) & (gate.vector == "real")].pivot_table(index="question_index", columns=["layer", "alpha"],
                                                                        values="confidence_5")
    print("mean |P(5 | a=+2) - P(5 | a=-2)| per layer:", {L: round((p5[L, 2] - p5[L, -2]).abs().mean(), 4) for L in (10, 14)})
    assert all((p5[L, 2] != p5[L, -2]).any() for L in (10, 14))
    for layer, vector, alphas, positions, T in ((14, "real", [-2, 0, 2], ["think", "start", "post"], 0.0),
                                                (6, "random", [-2, 2], ["think"], 0.0),
                                                (14, "real", [0, 2], ["think"], 0.6)):
        sm.run_reason(model, tok, run, dict(cfg, layer=layer, vector=vector, alphas=alphas, positions=positions,
                                            temperature=T, seed=42, n=6, budget=96, batch=6))
    analyze_steering.main(run)
    print("outputs:", sorted(p.name for p in run.iterdir()))
    print((run / "STEERING-RESULTS.md").read_text())


def fake_harness(n, path):
    """A harness phases.csv on the first n Phase 1 items: synthetic k_answers (5 samples; ~40% all at gt_position,
    ~40% mixed, ~20% none), Phase 1 correctness = the first sample, random Phase 2 abstention."""
    import numpy as np
    data, rng, rows = sm.load_items("popqa"), np.random.default_rng(3), []
    for i in range(n):
        q, _, gt, _ = sm.item(data, 1, i)
        wrong = [k for k in range(1, 5) if k != gt]
        n_right = rng.choice([5, 1, 2, 3, 4, 0], p=[.4, .1, .1, .1, .1, .2])
        ks = rng.permutation([gt] * n_right + list(rng.choice(wrong, 5 - n_right)))
        rows.append(dict(phase=1, question_index=i, question=q, gt_position=gt, is_correct=int(ks[0] == gt),
                         is_nota=0, k_answers="|".join(map(str, ks))))
        rows.append(dict(phase=2, question_index=i, question=q, gt_position=gt, is_correct=0,
                         is_nota=int(rng.random() < 0.4), k_answers=None))
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path / "phases.csv", index=False)


def test_pretrace_e2e(model, tok):
    """pretrace -> build_pretrace (selection, CV L*, bootstrap CI) -> reason at L* with the probe -> analyze_steering,
    on a fake harness run; then the rebuild from pretrace_items.csv + pretrace_hiddens.pt alone."""
    import analyze_steering
    import build_vector
    root = Path(tempfile.mkdtemp())
    fake_harness(120, root / "harness")
    run = root / "steer-Qwen3-0.6B-pretrace"
    run.mkdir()
    cfg = dict(model=MODEL, thinking="on", questions="popqa", batch=16, harness=str(root / "harness"))
    sm.run_pretrace(model, tok, run, cfg, sweep=[10, 14], n_eval_cell=6, n_group=40, min_group=6)
    vecs = torch.load(run / "vectors.pt")
    items = pd.read_csv(run / "pretrace_items.csv").set_index("question_index")
    evals = json.loads((run / "eval_items.json").read_text())["question_index"]
    n = vecs["n_group"]
    con = items[~items.index.isin(evals)]
    low_pool = con[(con.label == "mixed") & (con.is_correct == 1)]
    print(f"n per group {n}; low pool {len(low_pool)} {low_pool.gt_position.value_counts().to_dict()}")
    assert 6 <= n < 40 and len(vecs["high"]) == len(vecs["low"]) == n and not (set(vecs["high"]) | set(vecs["low"])) & set(evals)
    assert (items.label[vecs["high"]] == "high").all()
    assert ((items.label[vecs["low"]] == "mixed") & (items.is_correct[vecs["low"]] == 1)).all()
    cap = int(n * 0.28)
    assert max(items.gt_position[vecs["high"]].value_counts().max(), items.gt_position[vecs["low"]].value_counts().max()) <= cap
    try:  # n is the largest that fills both groups
        build_vector.groups_under_cap(con.reset_index(), n + 1, n + 1, 0.28)
        raise RuntimeError(f"n = {n + 1} also fills")
    except AssertionError:
        pass
    val = vecs["validity"]
    assert vecs["L_star"] == max(vecs["sweep"], key=lambda L: val[L]["cv_auroc"])
    lo, hi = vecs["auroc_ci"]
    assert lo <= val[vecs["L_star"]]["auroc"] <= hi and vecs["valid"] == (lo > 0.55)
    before = {k: vecs[k] for k in ("high", "low", "L_star", "auroc_ci")}
    (run / "vectors.pt").unlink()
    build_vector.build_pretrace(run, n_group=40, min_group=6)  # what `python build_vector.py <run>` does
    assert {k: torch.load(run / "vectors.pt")[k] for k in before} == before
    Ls, last = vecs["L_star"], max(vecs["layers"])
    reason = dict(cfg, vector="real", temperature=0.0, seed=42, n=6, budget=96, batch=6)
    sm.run_reason(model, tok, run, dict(reason, layer=Ls, alphas=[-2, 0, 2], positions=["start", "think", "post"]))
    sm.run_reason(model, tok, run, dict(reason, layer=Ls - 8, alphas=[2], positions=["think"]))
    sm.run_reason(model, tok, run, dict(reason, layer=last, alphas=[2], positions=["think"]))  # cap: L' = L
    r = pd.read_csv(run / "reason_phases.csv")
    st, base = r[r.alpha != 0], r[r.alpha == 0]
    assert (st.probe_layer == (st.layer + 8).clip(upper=last)).all()
    assert st[st.layer < last].probe_prompt_end.notna().all() and st[st.layer == last].probe_prompt_end.isna().all()
    assert all(base[f"probe_prompt_end_L{k}"].notna().all() for k in vecs["layers"])
    st = st[st.layer == Ls]
    a = st.groupby(["position_mode", "alpha"]).probe_prompt_end.mean()
    a0 = base[f"probe_prompt_end_L{Ls + 8}"].mean()
    print("probe (a) mean by mode, alpha:", a.round(3).to_dict(), "alpha 0:", round(a0, 3))
    assert abs(a["post", -2] - a0) < 1e-4 and abs(a["post", 2] - a0) < 1e-4  # POST leaves the prefill unsteered
    analyze_steering.main(run)
    print((run / "STEERING-RESULTS.md").read_text())


def test_probe_recompute(model, tok):
    """Manipulation check (b): the steered recompute reproduces generation's first-answer-token distribution, and the
    probe moves with alpha downstream of the steered layer."""
    texts = phase2_texts(tok, 3)
    texts = [texts[0], texts[1] + "<think>\nI recall this one; it is option 2.\n", texts[2] + "<think>\nShort answer: I know this, it is option"]
    v = torch.randn(model.config.hidden_size, generator=torch.Generator().manual_seed(1))
    u = torch.randn(model.config.hidden_size, generator=torch.Generator().manual_seed(2))
    u /= u.norm()
    steered_rows = 0
    for mode in ("start", "think", "post"):
        scores = {}
        for alpha in (0, 2):
            gens = sm.generate(model, tok, texts, 80, batch=3, steer=(10, v, alpha, mode), capture=(14,))
            probes = sm.probe_answer_pos(model, tok, texts, gens, (10, v, alpha, mode), {14: u})
            assert alpha or sum(p is not None for p in probes) == 2, "the two short traces must reach an answer"
            steered_rows += alpha and sum(p is not None for p in probes)
            sm.check_faithful(probes, greedy=True, strict=True)
            scores[alpha] = [p and round(p["score"][14], 4) for p in probes]
        print(f"{mode}: probe (b) at alpha 0 / 2: {scores[0]} / {scores[2]}")
    assert steered_rows >= 2, "too few steered rows reached an answer to check the steered recompute"


def test_answer_prefix():
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-32B-Think")
    p = answer_prompt(2, "Q?", ["a", "b", "c", "d"])
    assert sm.render(tok, p, "native", True, "Answer: ") == sm.render(tok, p, "native", True) + "Answer: "
    assert sm.render(tok, p, "native", True).endswith("<think>\n</think>\n\n")
    assert sm.render(tok, p, "native", False).rstrip().endswith("<think>")


if __name__ == "__main__":
    test_selection()
    test_answer_prefix()
    test_olmo_think_detection()
    test_olmo3_arch()
    model, tok = sm.load_model(MODEL, "cpu")
    test_alpha0_identity(model, tok)
    if "e2e" in sys.argv:
        test_e2e(model, tok)
        test_probe_recompute(model, tok)
        test_pretrace_e2e(model, tok)
