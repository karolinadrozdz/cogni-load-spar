"""Template task: free-form short answers scored by exact match against aliases (TASKS.md walks through it)."""

import re

import numpy as np
import pandas as pd

from abstention_modal import default_tag, parse_task_opts

SHORT_ANSWER_FILE = "short_answer_questions.csv"


def short_answer_prompt(q, base):
    if base:  # completion format, preceded by the exemplars
        return f"Question: {q}\nAnswer:"
    return ("Answer the question with just the answer: a name, number or short phrase, no explanation.\n\n"
            f"Question: {q}\nAnswer:")


def normalize_answer(s):
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


class ShortAnswer:
    name = "short_answer"
    options = {"shots": 4}  # exemplars in --thinking base mode; set with --task-opts shots=2

    def config(self, f):
        opts = parse_task_opts(f["task_opts"], self.options)
        return dict(task=self.name, model=f["model"], thinking=f["thinking"], n=f["n"], budget=f["budget"],
                    max_model_len=f["max_model_len"], temperature=f["temperature"], seed=f["seed"], tp=f["tp"],
                    revision=f["revision"] or None, dtype=f["dtype"], **opts,
                    tag=f["tag"] or default_tag(f, "_" + self.name))

    def run(self, ctx):
        cfg = ctx.cfg
        df = pd.read_csv("/data/" + SHORT_ANSWER_FILE)  # file order is the item order: same items at every checkpoint
        shots = df[df.split == "shot"].head(cfg["shots"])
        test = df[df.split == "test"].head(cfg["n"]).reset_index(drop=True)
        prefix = "".join(short_answer_prompt(q, True) + f" {a.split('|')[0]}\n\n"
                         for q, a in zip(shots.question, shots.answer)) if ctx.base else ""
        prompts = [short_answer_prompt(q, ctx.base) for q in test.question]
        gens = [g[0] for g in ctx.generate([prefix + p for p in prompts], ctx.max_tokens(16),
                                           **({"stop": ["\n"]} if ctx.base else {}))]
        rows = []
        for i, (r, p, g) in enumerate(zip(test.itertuples(), prompts, gens)):
            gold = {normalize_answer(a) for a in r.answer.split("|")}
            rows.append(dict(question_index=i, question=r.question, answer=r.answer, prompt=p,
                             generated_text=g["content"], is_correct=int(normalize_answer(g["content"]) in gold),
                             answer_words=len(g["content"].split()), thinking=cfg["thinking"],
                             reasoning=g["reasoning"], reasoning_tokens=g["reasoning_tokens"],
                             truncated=g["truncated"], generated_tokens=g["generated_tokens"],
                             finish_reason=g["finish_reason"]))
        return {"answers.csv": rows}, dict(budget=ctx.max_tokens(16), rendered_example=ctx.render(prefix + prompts[0]))

    def analyze(self, cfg, out):
        a = pd.read_csv(out / "answers.csv")
        return dict(run=out.name, n=len(a), accuracy=round(a.is_correct.mean(), 3),
                    short_format=round((a.answer_words <= 5).mean(), 3), truncated=int(a.truncated.sum()))


TASK = ShortAnswer()


# ---- sweep hooks (analyze_sweep.py) ----------------------------------------------------------------

def items(d):
    a = pd.read_csv(d / "answers.csv")
    t = a[["question_index", "question", "is_correct", "truncated", "reasoning_tokens"]].copy()
    t["short"] = (a.answer_words <= 5).astype(int)
    return t.set_index("question_index")


def measures(t):
    from analyze_sweep import auroc  # local only: the run and grading containers do not need analyze_sweep's deps
    y = t.is_correct.to_numpy()
    return dict(accuracy=y.mean(), short_format=t.short.mean(),
                auroc_neg_trace_len=auroc(y, -t.reasoning_tokens.to_numpy()) if t.reasoning_tokens.max() > 0 else np.nan,
                truncated=t.truncated.mean())


def gates(m, lo):
    """Accuracy only measures knowledge where the model answers in the short format."""
    return dict(gate_format=bool(m["short_format"] >= 0.9 and m["truncated"] < 0.05))


show = ["accuracy", "short_format", "auroc_neg_trace_len"]
