"""Predict before answering (Barkan, Black & Sourbut 2025, arXiv 2512.24661, Exp 1), on the PopQA Phase 1 items.

The model sees only the question (no options) and states how likely it is to answer it correctly. The outcome is
its own Phase 1 correctness, taken from an existing confidence_abstention run of the same checkpoint
(--task-opts ref_tag=<tag>), so this prospective signal sits on the same items as the Kumaran measures.

think=1 (default): the checkpoint's own --thinking mode with --budget, as in the confidence_abstention runs, so the
curves sit on one plot with them. The prediction is made before the options are shown; trace_has_gold flags traces
that name the answer.
think=0: the think block is closed before the reply (Barkan's reasoning budget 0); a secondary, no-trace arm.

    modal run --detach abstention_modal.py::main --task predict_before --model allenai/Olmo-3-32B-Think-DPO \
        --thinking native --budget 8192 --max-model-len 12288 --questions popqa --gpu H200 \
        --task-opts ref_tag=olmo32b-popqa-think-dpo --tag pb-olmo32b-think-dpo

Both post-training lines, each in the mode and on the reference run of sweep_batches.tsv (ref_tag = that run's tag):
    Think (--thinking native --budget 8192 --max-model-len 12288):
        Olmo-3-32B-Think-SFT@1e-4-step2000  ref olmo32b-popqa-think-sft-1e-4-step2000
        Olmo-3-32B-Think-SFT                ref olmo32b-popqa-think-sft-main
        Olmo-3-32B-Think-DPO                ref olmo32b-popqa-think-dpo
        Olmo-3-32B-Think (RL step 750)      ref olmo32b-popqa-think-rl-main
        Olmo-3.1-32B-Think (RL step 2300)   ref olmo32b-popqa-think-rl31-main
    Instruct (--thinking off):
        Olmo-3.1-32B-Instruct-SFT --dtype bfloat16  ref olmo32b-popqa-instruct-sft
        Olmo-3.1-32B-Instruct-DPO --dtype bfloat16  ref olmo32b-popqa-instruct-dpo
        Olmo-3.1-32B-Instruct                       ref olmo32b-popqa-instruct-rl
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from abstention_modal import QUESTION_FILES, default_tag, parse_task_opts

PROMPT = ("You will be shown a trivia question. Please do not answer it. Instead, predict whether you would answer it "
          "correctly if you were later asked it as a multiple choice question with 4 options, exactly one of them "
          "correct.\n\nQUESTION: {q}\n\n#######\n\nGiven a single attempt, how likely is it that you would answer "
          "this question correctly? Please estimate the likelihood as a percentage between 0% and 100%. "
          "Please respond *only* with a numerical percentage.")


def parse_pct(text):
    """First integer 0-100 in the reply (Barkan: first integer on the first line); NaN if none."""
    for m in re.finditer(r"\d+(?:\.\d+)?", text):
        v = float(m.group())
        if 0 <= v <= 100:
            return v
    return np.nan


def lp_pct(top):
    """Expected percentage over the numeric tokens 0-100 among the top-20 at the first content token, and their
    total probability mass (low mass: the reply did not start with a number)."""
    vals, ps = [], []
    for t, lp in top.items():
        s = t.strip().rstrip("%")
        if s.isdigit() and 0 <= int(s) <= 100:
            vals.append(int(s))
            ps.append(np.exp(lp))
    if not ps:
        return np.nan, 0.0
    ps = np.array(ps)
    return float(np.dot(vals, ps) / ps.sum()), float(ps.sum())


def close_think_prefix(rendered):
    """Prefill that closes the think block: the native template may already open <think> in the generation prompt."""
    return "\n</think>\n\n" if rendered.rstrip().endswith("<think>") else "<think>\n\n</think>\n\n"


class PredictBefore:
    name = "predict_before"
    options = {"ref_tag": "", "think": 1}

    def config(self, f):
        opts = parse_task_opts(f["task_opts"], self.options)
        assert f["thinking"] != "base", "predict_before needs a chat checkpoint (instructions, no exemplars)"
        assert opts["ref_tag"], "--task-opts ref_tag=<confidence_abstention run of this checkpoint> is required"
        assert f["questions"] in QUESTION_FILES, f"--questions must be one of {sorted(QUESTION_FILES)}"
        return dict(task=self.name, model=f["model"], thinking=f["thinking"], n=f["n"], budget=f["budget"],
                    max_model_len=f["max_model_len"], temperature=f["temperature"], seed=f["seed"], tp=f["tp"],
                    revision=f["revision"] or None, dtype=f["dtype"], questions=f["questions"], **opts,
                    tag=f["tag"] or default_tag(f, "_" + self.name))

    def run(self, ctx):
        cfg = ctx.cfg
        df = pd.read_csv("/data/" + QUESTION_FILES[cfg["questions"]].format(1)).head(cfg["n"])  # Phase 1 items
        ref = pd.read_csv(Path("/results") / cfg["ref_tag"] / "phases.csv")
        p1 = ref[ref.phase == 1].set_index("question_index")
        p2 = ref[ref.phase == 2].set_index("question_index")
        assert (p1.question.reindex(range(len(df))).to_numpy() == df.question.to_numpy()).all(), \
            "ref_tag run does not have the same Phase 1 items"

        prompts = [PROMPT.format(q=q) for q in df.question]
        prefix = close_think_prefix(ctx.render(prompts[0])) if ctx.thinking_active and not cfg["think"] else ""
        max_tokens = cfg["budget"] if ctx.thinking_active and cfg["think"] else 16
        gens = [g[0] for g in ctx.generate(prompts, max_tokens, prefix=prefix)]

        rows = []
        for i, (q, gold, p, g) in enumerate(zip(df.question, df.choice_1, prompts, gens)):
            lp, mass = lp_pct(g["top"])
            rows.append(dict(question_index=i, question=q, prompt=p, generated_text=g["content"],
                             pred_pct=parse_pct(g["content"]), lp_pct=lp, num_mass=mass,
                             p1_correct=int(p1.is_correct[i]), p1_truncated=int(p1.truncated[i]),
                             p2_abstain=int(p2.is_nota[i]) if i in p2.index else np.nan,
                             trace_has_gold=int(str(gold).lower() in g["reasoning"].lower()),
                             thinking=cfg["thinking"], reasoning=g["reasoning"],
                             reasoning_tokens=g["reasoning_tokens"], truncated=g["truncated"],
                             generated_tokens=g["generated_tokens"], finish_reason=g["finish_reason"]))
        return {"predictions.csv": rows}, dict(prefix=prefix, rendered_example=ctx.render(prompts[0], prefix))

    def analyze(self, cfg, out):
        t = items(out)
        m = measures(t, rank_auroc)
        return dict(run=out.name, n=len(t), **{k: round(float(v), 3) for k, v in m.items()})


TASK = PredictBefore()


def rank_auroc(y, s):
    """Rank AUROC (ties averaged) over non-NaN scores; same as analyze_sweep.auroc, without its import."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    keep = ~np.isnan(s)
    y, s = y[keep], s[keep]
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (pd.Series(s).rank().to_numpy()[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---- sweep hooks (analyze_sweep.py) ----------------------------------------------------------------

def items(d):
    a = pd.read_csv(Path(d) / "predictions.csv")
    cols = ["question_index", "question", "pred_pct", "lp_pct", "num_mass", "p1_correct", "p2_abstain",
            "truncated", "trace_has_gold"]
    return a[cols].set_index("question_index")


def measures(t, auroc=None):
    if auroc is None:
        from analyze_sweep import auroc  # local only: the run container does not need analyze_sweep's deps

    def au(y, s):  # AUROC over the items with a score
        y, s = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
        k = ~np.isnan(y) & ~np.isnan(s)
        return auroc(y[k].astype(int), s[k]) if k.sum() > 1 else np.nan

    y = t.p1_correct.to_numpy()
    pred = t.pred_pct.to_numpy(dtype=float)
    keep = ~np.isnan(pred)
    return dict(
        accuracy=y.mean(),
        parse_rate=keep.mean(),
        mean_pred=pred[keep].mean() / 100 if keep.any() else np.nan,
        overconfidence=(pred[keep].mean() / 100 - y[keep].mean()) if keep.any() else np.nan,
        pred_spread=pred[keep].std() / 100 if keep.any() else np.nan,
        auroc_pred=au(y, pred),                        # prospective prediction vs own Phase 1 correctness
        auroc_lp=au(y, t.lp_pct),                      # same, from the number-token distribution
        auroc_pred_abstain=au(t.p2_abstain, -pred),    # low prediction vs abstaining in Phase 2
        mean_num_mass=t.num_mass.mean(),
        trace_has_gold=t.trace_has_gold.mean(),
        truncated=t.truncated.mean(),
    )


def gates(m, lo):
    """The prediction is only readable where the model replies with a number, and spreads its predictions."""
    return dict(gate_format=bool(m["parse_rate"] >= 0.9 and m["truncated"] < 0.05),
                gate_spread=bool(m["pred_spread"] >= 0.05))


show = ["accuracy", "mean_pred", "overconfidence", "auroc_pred", "auroc_lp", "auroc_pred_abstain", "parse_rate"]
