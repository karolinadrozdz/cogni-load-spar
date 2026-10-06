"""derived_state: running counts the model must carry vs the same counts on the page.

Same streams, truncated at a seeded point. Arm `derived` shows the plain
stream; arm `copyable` writes each tracked word's running count after it. Read
at every in-stream comma and at the answer, for the digits 0-9 and the number
words zero-nine; a value is present if its digit or its word is.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from cogniload import experiment, prompts, readout
from cogniload.experiment import VIEWS

ARMS = ("derived", "copyable")
# KEEP_TRACK with a running-count instruction; ends in ". " so the question follows.
COUNT_STREAM = (
    "Track these categories: {tracked}. Keep a running count of how many words "
    "from each tracked category have appeared. Here is a sequence of words: {stream}. "
)
COUNT_QUESTION = "How many {category} words have appeared so far? Answer with a single digit."
# Copyable arm, tracked words only: the contrast is tracked state on the page or not.
ANNOTATION = "{word} ({category} {n})"
# Trailing space: " 3" is two tokens on Qwen, so this makes the bare digit the next token.
COUNT_PREFILL = "Answer: "

DIGITS = [str(v) for v in range(10)]
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
VALUE = {**{d: v for v, d in enumerate(DIGITS)}, **{w: v for v, w in enumerate(NUMBER_WORDS)}}


def label(items, tracked, categories, arm: str) -> tuple[list[str], list[dict]]:
    """Stream pieces for `arm`, and every category's count after each item."""
    counts = dict.fromkeys(categories, 0)
    pieces, states = [], []
    for it in items:
        counts[it.category] += 1
        states.append(dict(counts))
        pieces.append(ANNOTATION.format(word=it.word, category=it.category, n=counts[it.category])
                      if arm == "copyable" and it.category in tracked else it.word)
    return pieces, states


def build_count(tokenizer: Any, pieces: list[str], tracked: tuple[str, ...], category: str, *,
                enable_thinking: bool) -> str:
    user = (COUNT_STREAM.format(tracked=", ".join(tracked), stream=", ".join(pieces))
            + COUNT_QUESTION.format(category=category))
    return prompts.render_chat(tokenizer, user, enable_thinking=enable_thinking,
                               prefill=COUNT_PREFILL)


def run_stream(ctx, stream, arm: str) -> tuple[list[dict], dict]:
    tok = ctx.model.tokenizer
    # The seed string is fixed: changing it moves every truncation point.
    p = random.Random(f"exp3-{stream.stream_id}-{stream.c_t}").randint(
        *ctx.config["derived_state"]["trunc_range"])
    categories = sorted({i.category for i in stream.items})
    pieces, states = label(stream.items[:p], stream.tracked, categories, arm)
    queried = stream.queried_category
    text = build_count(tok, pieces, stream.tracked, queried,
                       enable_thinking=ctx.spec["enable_thinking"])
    commas = experiment.delimiters(tok, text, pieces)[:-1]
    last = ctx.model.encode(text).shape[-1] - 1

    ids = ([readout.token_id(tok, d, prefix="") for d in DIGITS]
           + [readout.token_id(tok, w) for w in NUMBER_WORDS])
    meta = {w: {"stream_id": stream.stream_id, "c_t": stream.c_t, "arm": arm}
            for w in DIGITS + NUMBER_WORDS}
    positions = {**dict.fromkeys(commas, "in_stream"), last: "answer"}
    rows, logits = experiment.read_both(ctx, text, meta, positions, ids)

    gen = [tok.decode([i]) for i in experiment.new_tokens(ctx.model, text, 2)]
    logits, expected = logits[-1], states[-1][queried]
    eid = ids[expected]
    # The answer position carries the state after item p.
    pos_states = ([{"token_pos": t, "readout_at": "in_stream", "item": i + 1, "counts": states[i]}
                   for i, t in enumerate(commas)]
                  + [{"token_pos": last, "readout_at": "answer", "item": p, "counts": states[-1]}])
    summary = {
        "stream_id": stream.stream_id, "c_t": stream.c_t, "arm": arm, "p": p,
        "tracked": json.dumps(list(stream.tracked)), "queried_category": queried,
        "expected": expected, "answer_pos": last,
        "gen_tok1": gen[0] if gen else None, "gen_tok2": gen[1] if len(gen) > 1 else None,
        "argmax": tok.decode([int(logits.argmax())]),
        "expected_rank": int(readout.rank_of(logits[None], torch.tensor([eid]))[0, 0]),
        "correct": int(logits.argmax()) == eid,
        # Rank among the digits 0-3, the only counts that occur.
        "rank_0_3": int((logits[ids[:4]] > logits[eid]).sum()) + 1,
        "states": json.dumps(pos_states), "prompt": text,
    }
    return rows, summary


def _show(c_t: int, s: pd.DataFrame) -> None:
    first = s.iloc[0]
    print(f"\n  --- C_t={c_t} first stream prompt ---\n{first['prompt']}\n  ---")
    print(f"  generated: {first['gen_tok1']!r} {first['gen_tok2']!r}; "
          f"expected {first['expected']}")
    print(f"  C_t={c_t}: top-1 {s['correct'].mean():.0%}, "
          f"top-1 among 0-3 {(s['rank_0_3'] == 1).mean():.0%}")


def run(spec: dict, config: dict, results_dir: Path, *, arm: str, limit: int | None = None,
        force: bool = False) -> None:
    templates = {"stream": COUNT_STREAM, "question": COUNT_QUESTION, "prefill": COUNT_PREFILL,
                 "annotation": ANNOTATION if arm == "copyable" else None}
    experiment.run(f"derived_state_{arm}", config["derived_state"],
                   lambda ctx, stream: run_stream(ctx, stream, arm), templates, spec, config,
                   results_dir, limit=limit, force=force, show=_show)


# --- analysis -----------------------------------------------------------------


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def table(df: pd.DataFrame, summary: pd.DataFrame, k: int, *, rank_col: str = "rank_wordlike",
          lens: str = "jlens") -> pd.DataFrame:
    """Per C_t: count groups at the answer (`ans_*`) and pooled over in-stream commas (`in_*`).

    Groups are scored per category. An untracked count is scored only where no
    tracked category has the same value; a stale value (c - 1) is skipped when
    it equals a current tracked count, and the skips are reported.
    """
    rows = df[df["in_band"] & (df["lens"] == lens)]
    bm = rows.groupby(["stream_id", "c_t", "token_pos", "word"])[rank_col].min().reset_index()
    bm["value"] = bm["word"].map(VALUE)
    present = (bm.groupby(["stream_id", "c_t", "token_pos", "value"])[rank_col].min()
               <= k).to_dict()

    out = []
    for c_t, s in summary.groupby("c_t"):
        acc = {key: [] for key in ("ans_queried", "ans_tracked", "ans_untracked", "ans_stale",
                                   "ans_floor", "in_tracked", "in_untracked", "in_stale",
                                   "in_floor", "ans_floor4", "in_floor4",
                                   "in_tracked_just", "in_tracked_other")}
        collisions = n_pos = n_pos_in = skipped = 0
        for _, r in s.iterrows():
            tracked = json.loads(r["tracked"])
            prev = None
            for st in json.loads(r["states"]):
                def hit(v):
                    return present[(r["stream_id"], c_t, st["token_pos"], v)]
                counts = st["counts"]
                cur = {c: counts[c] for c in tracked}
                tvals = set(cur.values())
                pre = "ans" if st["readout_at"] == "answer" else "in"
                if pre == "ans":
                    acc["ans_queried"].append(hit(cur[r["queried_category"]]))
                acc[f"{pre}_tracked"] += [hit(cur[c]) for c in tracked
                                          if pre == "in" or c != r["queried_category"]]
                acc[f"{pre}_untracked"] += [hit(n) for c, n in counts.items()
                                            if c not in tracked and n not in tvals]
                for c in tracked:
                    if cur[c] >= 1:
                        if cur[c] - 1 in tvals:
                            skipped += 1
                        else:
                            acc[f"{pre}_stale"].append(hit(cur[c] - 1))
                acc[f"{pre}_floor"] += [hit(v) for v in range(5, 10)]
                # 4 never occurs as a count but is in range: a matched floor.
                acc[f"{pre}_floor4"].append(hit(4))
                if pre == "in":
                    # The category of the word ending at this comma: the one whose count rose.
                    base = prev or dict.fromkeys(counts, 0)
                    just = next(c for c in counts if counts[c] != base[c])
                    acc["in_tracked_just"] += [hit(cur[c]) for c in tracked if c == just]
                    acc["in_tracked_other"] += [hit(cur[c]) for c in tracked if c != just]
                    prev = counts
                    n_pos_in += 1
                collisions += len(tvals) < len(tracked)
                n_pos += 1
        out.append({
            "c_t": c_t, "n": len(s), "top1": s["correct"].mean(),
            "median_rank": s["expected_rank"].median(), "top1_0_3": (s["rank_0_3"] == 1).mean(),
            **{key: _mean(v) for key, v in acc.items()},
            "collision_rate": collisions / n_pos if n_pos else float("nan"),
            "stale_skipped": skipped, "n_pos_in": n_pos_in,
        })
    return pd.DataFrame(out)


def report(spec: dict, config: dict, results_dir: Path) -> None:
    k, fmt = config["readout"]["primary_k"], dict(index=False, float_format="%.3f")
    for arm in ARMS:
        for label, df, summary in experiment.runs(f"derived_state_{arm}", spec["alias"], results_dir):
            for rank_col, lens in VIEWS:
                print(f"\nderived_state {arm} {label} [{rank_col}, {lens}, in-band, k={k}]")
                print(table(df, summary, k, rank_col=rank_col, lens=lens).to_string(**fmt))
            print(f"\nderived_state {arm} first generated token: "
                  f"{summary['gen_tok1'].value_counts().head(5).to_dict()}")
