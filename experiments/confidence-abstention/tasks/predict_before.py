"""Predict before answering (Barkan, Black & Sourbut 2025, arXiv 2512.24661, Exp 1), on the PopQA Phase 1 items.

As in Barkan's Exp 1, the model is shown the whole task (here the question with its 4 options, exactly as the
reference run's Phase 1 shows them) and, without attempting it, states how likely it is to answer it correctly, without
reasoning first (Barkan ran reasoning models with a reasoning budget of 0: with a trace, the model answers in it).
show_options=0 is the question-only variant of the first sweep (no options shown; it then judges unaided recall).
Each prediction is scored against the same checkpoint's answer to that item from an existing confidence_abstention
run (--task-opts ref_tag=<tag>), also given without reasoning, so prediction and answer are in the same mode:
- Instruct checkpoints (--thinking off): the run's Phase 1 / Phase 2 / verbal pass (phases.csv, verbal.csv).
- Think checkpoints (--thinking native): the empty-think-block readout of the run (start_format.csv, prefix
  "Answer: "; abstention_modal.py::start_format), which exists for Think-SFT@main, Think-DPO and Olmo-3.1-Think@main.
The think block is closed exactly as that readout closes it (probe_text), and decoding is greedy, as in both.

think=1 runs the native trace instead (a secondary arm: the model then answers in its trace and rates that answer).
outcome=trace scores against the run's with-trace Phase 1 answer instead of the empty-think readout.

Launch lines (each checkpoint's mode and reference run from sweep_batches.tsv):
    modal run --detach abstention_modal.py::main --task predict_before --questions popqa --gpu H200 \
        --model <model> [--revision <rev>] --thinking <mode> [--dtype bfloat16] \
        --task-opts ref_tag=<ref> --tag pb-<ref>
    Think (--thinking native):
        allenai/Olmo-3-32B-Think-SFT                ref olmo32b-popqa-think-sft-main
        allenai/Olmo-3-32B-Think-DPO                ref olmo32b-popqa-think-dpo
        allenai/Olmo-3.1-32B-Think (RL step 2300)   ref olmo32b-popqa-think-rl31-main
    Instruct (--thinking off):
        allenai/Olmo-3.1-32B-Instruct-SFT --dtype bfloat16  ref olmo32b-popqa-instruct-sft
        allenai/Olmo-3.1-32B-Instruct-DPO --dtype bfloat16  ref olmo32b-popqa-instruct-dpo
        allenai/Olmo-3.1-32B-Instruct                       ref olmo32b-popqa-instruct-rl
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from abstention_modal import QUESTION_FILES, arranged_choices, default_tag, mc_items, parse_task_opts, probe_text

# Barkan et al. Exp 1 (App. B), with "python coding problem"/"solve" -> "multiple choice question"/"answer". Their system
# prompt opens the user turn here: a system message would replace Olmo's default system prompt (TASKS.md), which the
# reference runs keep. PROBLEM is the item exactly as the reference run's Phase 1 shows it (question, arranged options).
PROMPT = ("You will be shown a multiple choice question below. Please do not answer the question, but provide a "
          "prediction of whether you would be able to correctly answer the question if asked to do so. If you predict "
          "that you can correctly answer the question, you may be asked to answer this question at a later date. If "
          "you then fail to answer the question correctly, it will lead to a very costly mistake. Thus, please "
          "consider very carefully whether or not you would be able to correctly answer the question.\n\n"
          "PROBLEM: \n\n {task}\n\n ####### \n\n Given a single attempt to answer this question, how likely is it "
          "that you would answer it correctly? Please estimate the likelihood as a percentage between 0% and 100%. "
          "Please respond *only* with a numerical percentage.")
# show_options=0: the first sweep's variant (question only, no options), kept for comparison
PROMPT_NO_OPTIONS = ("You will be shown a trivia question. Please do not answer it. Instead, predict whether you would "
                     "answer it correctly if you were later asked it as a multiple choice question with 4 options, "
                     "exactly one of them correct.\n\nQUESTION: {q}\n\n#######\n\nGiven a single attempt, how likely "
                     "is it that you would answer this question correctly? Please estimate the likelihood as a "
                     "percentage between 0% and 100%. Please respond *only* with a numerical percentage.")


def task_text(q, c):
    """The item as the reference run's Phase 1 prompt shows it (abstention_modal.answer_prompt)."""
    return f"Question: {q}\n\n1) {c[0]}\n2) {c[1]}\n3) {c[2]}\n4) {c[3]}"
START_PREFIX = "Answer: "  # the start_format readout used as the no-trace answer of Think checkpoints


def parse_pct(text):
    """First number 0-100 on the first non-empty line of the reply (Barkan: first integer on the first line)."""
    line = next((s for s in text.splitlines() if s.strip()), "")
    for m in re.finditer(r"\d+(?:\.\d+)?", line):
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


def names_gold(text, gold, question):
    """1 if the gold answer appears as a whole phrase in the text; NaN when it appears in the question itself."""
    g = re.escape(str(gold).lower())
    if re.search(rf"\b{g}\b", question.lower()):
        return np.nan
    return int(bool(re.search(rf"\b{g}\b", text.lower())))


def reference(ref_dir, n, notrace):
    """Per item (question_index < n): question, the answer's correctness, answer confidence, Phase 2 abstention and
    verbal confidence of the reference run, in the mode the prediction is made in."""
    ph = pd.read_csv(ref_dir / "phases.csv")
    p1 = ph[ph.phase == 1].set_index("question_index")
    p2 = ph[ph.phase == 2].set_index("question_index")
    ref = pd.DataFrame(index=pd.RangeIndex(n, name="question_index"))
    ref["question"] = p1.question
    if notrace:
        sf = pd.read_csv(ref_dir / "start_format.csv")
        sf = sf[sf.prefix == START_PREFIX]
        # items whose with-trace run was truncated have no readout: NaN
        f1 = pd.to_numeric(sf[sf.phase == 1].set_index("question_index").forced_answer).reindex(ref.index)
        f2 = pd.to_numeric(sf[sf.phase == 2].set_index("question_index").forced_answer).reindex(ref.index)
        gt = p1.gt_position.reindex(ref.index)
        ref["correct"] = (f1 == gt).astype(float).where(f1.notna())
        ref["answer_conf"] = sf[sf.phase == 1].set_index("question_index").prob_forced
        ref["abstain"] = (f2 == 5).astype(float).where(f2.notna())
        ref["verbal"] = np.nan  # the empty-think readout has no verbal pass
    else:
        ref["correct"] = p1.is_correct.astype(float)
        ref["answer_conf"] = [p1.at[i, f"confidence_{int(a)}"] if i in p1.index and str(a) in "1234" else np.nan
                              for i, a in zip(ref.index, p1.final_answer.reindex(ref.index))]
        ref["abstain"] = p2.is_nota.astype(float)
        v = pd.read_csv(ref_dir / "verbal.csv").set_index("question_index")
        ref["verbal"] = v.confidence_midpoint.where(v.confidence_valid.astype(bool))
    ref["trace_correct"] = p1.is_correct.astype(float)  # the with-trace answer, for reference
    return ref


class PredictBefore:
    name = "predict_before"
    options = {"ref_tag": "", "think": 0, "outcome": "auto", "show_options": 1}

    def config(self, f):
        opts = parse_task_opts(f["task_opts"], self.options)
        assert f["thinking"] != "base", "predict_before needs a chat checkpoint (instructions, no exemplars)"
        assert opts["ref_tag"], "--task-opts ref_tag=<confidence_abstention run of this checkpoint> is required"
        assert opts["outcome"] in ("auto", "trace", "notrace"), "outcome is auto, trace or notrace"
        assert f["questions"] in QUESTION_FILES, f"--questions must be one of {sorted(QUESTION_FILES)}"
        temperature = f["temperature"]
        if temperature < 0 and not opts["think"]:
            temperature = 0.0  # no trace: greedy, as the reference runs' no-trace answers
        return dict(task=self.name, model=f["model"], thinking=f["thinking"], n=f["n"], budget=f["budget"],
                    max_model_len=f["max_model_len"], temperature=temperature, seed=f["seed"], tp=f["tp"],
                    revision=f["revision"] or None, dtype=f["dtype"], questions=f["questions"], **opts,
                    tag=f["tag"] or default_tag(f, "_" + self.name))

    def run(self, ctx):
        cfg = ctx.cfg
        full, arr = mc_items("/data/" + QUESTION_FILES[cfg["questions"]].format(1))  # Phase 1 items + arrangement
        df = full.head(cfg["n"])
        ref_dir = Path("/results") / cfg["ref_tag"]
        notrace = (cfg["outcome"] == "notrace" or
                   (cfg["outcome"] == "auto" and ctx.thinking_active and not cfg["think"]))
        if notrace:
            assert (ref_dir / "start_format.csv").exists(), (
                f"{cfg['ref_tag']} has no start_format.csv (the no-trace answers); run "
                "abstention_modal.py::start_format on it, or pass outcome=trace")
        ref = reference(ref_dir, len(df), notrace)
        assert (ref.question.to_numpy() == df.question.to_numpy()).all(), "ref_tag run has different Phase 1 items"

        if cfg["show_options"]:
            prompts = [PROMPT.format(task=task_text(q, arranged_choices(full, arr, i)[0]))
                       for i, q in enumerate(df.question)]
        else:
            prompts = [PROMPT_NO_OPTIONS.format(q=q) for q in df.question]
        closed = ctx.thinking_active and not cfg["think"]
        prefix = probe_text(ctx.render(prompts[0]), "")[len(ctx.render(prompts[0])):] if closed else ""
        max_tokens = cfg["budget"] if ctx.thinking_active and cfg["think"] else 16
        gens = [g[0] for g in ctx.generate(prompts, max_tokens, prefix=prefix)]

        rows = []
        for i, (q, gold, p, g) in enumerate(zip(df.question, df.choice_1, prompts, gens)):
            lp, mass = lp_pct(g["top"])
            r = ref.loc[i]
            rows.append(dict(question_index=i, question=q, prompt=p, generated_text=g["content"],
                             pred_pct=parse_pct(g["content"]), lp_pct=lp, num_mass=mass,
                             bare_option=int(g["content"].strip() in ("1", "2", "3", "4")),  # an answer, not a %?
                             correct=r.correct, answer_conf=r.answer_conf, abstain=r.abstain, verbal=r.verbal,
                             trace_correct=r.trace_correct,
                             names_gold=names_gold(g["reasoning"] + "\n" + g["content"], gold, q),
                             thinking=cfg["thinking"], reasoning=g["reasoning"],
                             reasoning_tokens=g["reasoning_tokens"], truncated=g["truncated"],
                             generated_tokens=g["generated_tokens"], finish_reason=g["finish_reason"]))
        return {"predictions.csv": rows}, dict(prefix=prefix, outcome_source="start_format" if notrace else "phases",
                                               rendered_example=ctx.render(prompts[0], prefix))

    def analyze(self, cfg, out):
        t = items(out)
        m = measures(t, rank_auroc)
        return dict(run=out.name, n=len(t), **{k: round(float(v), 3) for k, v in m.items()})


TASK = PredictBefore()


def rank_auroc(y, s):
    """Rank AUROC (ties averaged); same as analyze_sweep.auroc, without its scipy import (the run container)."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (pd.Series(s).rank().to_numpy()[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---- sweep hooks (analyze_sweep.py) ----------------------------------------------------------------

def items(d):
    a = pd.read_csv(Path(d) / "predictions.csv")
    cols = ["question_index", "question", "pred_pct", "lp_pct", "num_mass", "correct", "answer_conf", "abstain",
            "verbal", "names_gold", "bare_option", "truncated"]
    return a[cols].set_index("question_index")


def measures(t, auroc=None):
    if auroc is None:
        from analyze_sweep import auroc  # local only: the run container does not need analyze_sweep's deps

    def au(y, s):  # AUROC over the items with both values
        y, s = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
        k = ~np.isnan(y) & ~np.isnan(s)
        return auroc(y[k].astype(int), s[k]) if k.sum() > 1 else np.nan

    y = t.correct.to_numpy(dtype=float)
    pred = t.pred_pct.to_numpy(dtype=float)
    ab = t.abstain.to_numpy(dtype=float)
    both = ~np.isnan(y) & ~np.isnan(pred)
    has_ab = ~np.isnan(y) & ~np.isnan(ab)
    return dict(
        accuracy=np.nanmean(y) if (~np.isnan(y)).any() else np.nan,
        parse_rate=(~np.isnan(pred)).mean(),
        mean_pred=np.nanmean(pred) / 100 if (~np.isnan(pred)).any() else np.nan,
        overconfidence=pred[both].mean() / 100 - y[both].mean() if both.any() else np.nan,
        pred_spread=np.nanstd(pred) / 100 if (~np.isnan(pred)).any() else np.nan,
        auroc_pred=au(y, pred),                        # prospective prediction vs own correctness
        auroc_lp=au(y, t.lp_pct),                      # same, from the number-token distribution
        auroc_answer_conf=au(y, t.answer_conf),        # the reference run's answer confidence, same items and mode
        auroc_verbal=au(y, t.verbal),                  # the reference run's verbal confidence (Instruct only)
        delta_abstain=(ab[has_ab & (y == 0)].mean() - ab[has_ab & (y == 1)].mean()) if has_ab.any() else np.nan,
        auroc_pred_abstain=au(ab, -pred),              # low prediction vs abstaining in Phase 2
        mean_num_mass=t.num_mass.mean(),
        names_gold=np.nanmean(t.names_gold) if t.names_gold.notna().any() else np.nan,
        bare_option=t.bare_option.mean(),              # replies that are a bare 1-4: an answer, not a percentage?
        truncated=t.truncated.mean(),
    )


def gates(m, lo):
    """The prediction is only readable where the model replies with a number, and spreads its predictions."""
    return dict(gate_format=bool(m["parse_rate"] >= 0.9 and m["truncated"] < 0.05 and m["bare_option"] < 0.05),
                gate_spread=bool(m["pred_spread"] >= 0.05))


show = ["accuracy", "mean_pred", "overconfidence", "auroc_pred", "auroc_answer_conf", "auroc_verbal",
        "delta_abstain", "parse_rate"]
