"""retro_cue: does the J-space content follow a change of cue?

Ask for tracked category A, give the model's own one-token answer back as
history, then ask for B. Read at both answer positions (`pos1`, `pos2`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cogniload import experiment, prompts
from cogniload.experiment import VIEWS, presence, rate

# The first answer, as an earlier assistant turn.
HISTORY = "Answer: {a1}"
TEMPLATES = {"stream": prompts.KEEP_TRACK, "question": prompts.RECENT,
             "prefill": prompts.PREFILL, "history": HISTORY}


def cues(stream) -> tuple[str, str]:
    """A is the queried category; B is the next tracked category after A, wrapping."""
    a = stream.queried_category
    i = stream.tracked.index(a)
    return a, stream.tracked[(i + 1) % len(stream.tracked)]


def build_second_cue(tokenizer: Any, words: list[str], tracked: tuple[str, ...], a: str, b: str,
                     a1: str, *, enable_thinking: bool) -> str:
    """[user: stream + Q(A), assistant: "Answer: {a1}", user: Q(B)], prefilled."""
    first = prompts.stream_turn(words, tracked, prompts.RECENT.format(category=a))
    return prompts.render_chat(tokenizer, [
        {"role": "user", "content": first},
        {"role": "assistant", "content": HISTORY.format(a1=a1)},
        {"role": "user", "content": prompts.RECENT.format(category=b)},
    ], enable_thinking=enable_thinking, prefill=prompts.PREFILL)


def run_stream(ctx, stream) -> tuple[list[dict], dict]:
    a, b = cues(stream)
    tok, thinking = ctx.model.tokenizer, ctx.spec["enable_thinking"]
    meta = experiment.words_and_meta(stream, ctx.pool, ctx.config["stream"]["n_absent_words"])

    text1 = prompts.build_recent(tok, stream.words, stream.tracked, a, enable_thinking=thinking)
    pos1 = ctx.model.encode(text1).shape[-1] - 1
    rows1, logits1 = experiment.read_both(ctx, text1, meta, {pos1: "pos1"})
    # The first answer is the readout pass's own argmax; no separate generation.
    a1 = tok.decode([int(logits1[0].argmax())]).strip()
    q1_a = experiment.score_q1(tok, logits1[0], stream.targets[a])

    text2 = build_second_cue(tok, stream.words, stream.tracked, a, b, a1, enable_thinking=thinking)
    pos2 = ctx.model.encode(text2).shape[-1] - 1
    rows2, logits2 = experiment.read_both(ctx, text2, meta, {pos2: "pos2"})
    q1_b = experiment.score_q1(tok, logits2[0], stream.targets[b])

    rows = ([{**r, "position": 1, "cat_a": a, "cat_b": b} for r in rows1]
            + [{**r, "position": 2, "cat_a": a, "cat_b": b} for r in rows2])
    summary = {
        "stream_id": stream.stream_id, "c_t": stream.c_t, "cat_a": a, "cat_b": b,
        "target_a": stream.targets[a], "target_b": stream.targets[b],
        "a1": a1, "a1_correct": a1.lower() == stream.targets[a].lower(),
        "pos1": pos1, "pos2": pos2,
        **{f"q1a_{k}": v for k, v in q1_a.items()},
        **{f"q1b_{k}": v for k, v in q1_b.items()},
    }
    return rows, summary


def _show(c_t: int, s: pd.DataFrame) -> None:
    print(f"  C_t={c_t}: a1 correct {s['a1_correct'].mean():.0%}, "
          f"Q(B) top-1 {s['q1b_correct'].mean():.0%}")


def run(spec: dict, config: dict, results_dir: Path, *, limit: int | None = None,
        force: bool = False) -> None:
    experiment.run("retro_cue", config["retro_cue"], run_stream, TEMPLATES, spec, config,
                   results_dir, limit=limit, force=force, show=_show)


def table(df: pd.DataFrame, summary: pd.DataFrame, k: int, *, rank_col: str = "rank_wordlike",
          lens: str = "jlens") -> pd.DataFrame:
    """Per (position, C_t): cue A's and cue B's targets, labels and replaced words, floors."""
    p = presence(df, k, rank_col=rank_col)
    p = p[p["lens"] == lens].merge(summary[["stream_id", "c_t", "cat_a", "cat_b"]],
                                   on=["stream_id", "c_t"])
    out = []
    for (at, c_t), g in p.groupby(["readout_at", "c_t"]):
        is_a, is_b = g.category == g.cat_a, g.category == g.cat_b
        s = summary[summary.c_t == c_t]
        out.append({
            "position": int(at[-1]), "c_t": c_t, "n": g["stream_id"].nunique(),
            "target_A": rate(g[(g.role == "target") & is_a]),
            "target_B": rate(g[(g.role == "target") & is_b]),
            "label_A": rate(g[(g.role == "label_tracked") & is_a]),
            "label_B": rate(g[(g.role == "label_tracked") & is_b]),
            "replaced_A": rate(g[(g.role == "replaced") & is_a]),
            "replaced_B": rate(g[(g.role == "replaced") & is_b]),
            "target_other": rate(g[(g.role == "target") & ~is_a & ~is_b]),
            "untracked": rate(g[g.role == "untracked"]),
            "floor_word": rate(g[g.role == "absent"]),
            "floor_label": rate(g[g.role == "label_absent"]),
            "a1_correct": s["a1_correct"].mean(),
            "q1b_top1": s["q1b_correct"].mean(),
        })
    return pd.DataFrame(out).sort_values(["position", "c_t"])


def report(spec: dict, config: dict, results_dir: Path) -> None:
    k, fmt = config["readout"]["primary_k"], dict(index=False, float_format="%.3f")
    for label, df, summary in experiment.runs("retro_cue", spec["alias"], results_dir):
        for rank_col, lens in VIEWS:
            print(f"\nretro_cue {label} [{rank_col}, {lens}, in-band, k={k}]")
            print(table(df, summary, k, rank_col=rank_col, lens=lens).to_string(**fmt))
