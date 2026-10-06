"""forced_demand: every tracked word is needed for the next token.

The question asks for all tracked words in alphabetical order, so to emit the
first one the model must compare them all. Read at the `Answer:` prefill.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cogniload import experiment, prompts
from cogniload.experiment import VIEWS, presence, rate

# Before the readout position, so the model knows every target is needed.
SORTED_LIST = ("List the most recent word for each tracked category, in alphabetical "
               "order. Answer with the words only, separated by commas.")
TEMPLATES = {"stream": prompts.KEEP_TRACK, "question": SORTED_LIST, "prefill": prompts.PREFILL}


def build_sorted_list(tokenizer: Any, words: list[str], tracked: tuple[str, ...], *,
                      enable_thinking: bool) -> str:
    return prompts.render_chat(tokenizer, prompts.stream_turn(words, tracked, SORTED_LIST),
                               enable_thinking=enable_thinking, prefill=prompts.PREFILL)


def run_stream(ctx, stream) -> tuple[list[dict], dict]:
    tok = ctx.model.tokenizer
    text = build_sorted_list(tok, stream.words, stream.tracked,
                             enable_thinking=ctx.spec["enable_thinking"])
    meta = experiment.words_and_meta(stream, ctx.pool, ctx.config["stream"]["n_absent_words"])
    last = ctx.model.encode(text).shape[-1] - 1
    rows, logits = experiment.read_both(ctx, text, meta, {last: "answer"})

    # Case-insensitive: country names are capitalised, and "France" < "apple".
    order = sorted(stream.targets.values(), key=str.lower)
    for row in rows:
        row["emit_order"] = order.index(row["word"]) + 1 if row["role"] == "target" else None
    q1 = experiment.score_q1(tok, logits[0], order[0])
    raw = experiment.free_text(ctx.model, text,
                               max_new_tokens=ctx.config["forced_demand"]["max_new_tokens"])
    emitted = [w.strip().lower() for w in raw.strip().rstrip(".").split(",")]
    summary = {
        "stream_id": stream.stream_id, "c_t": stream.c_t,
        "queried_category": stream.queried_category,
        "sorted_targets": ", ".join(order), "answer_pos": last,
        "list_raw": raw, "list_correct": emitted == [w.lower() for w in order],
        **{f"q1_{k}": v for k, v in q1.items()},
    }
    return rows, summary


def _show(c_t: int, s: pd.DataFrame) -> None:
    print(f"  C_t={c_t}: first word top-1 {s['q1_correct'].mean():.0%}, "
          f"list correct {s['list_correct'].mean():.0%}; "
          f"e.g. {s['list_raw'].iloc[0]!r} vs {s['sorted_targets'].iloc[0]!r}")


def run(spec: dict, config: dict, results_dir: Path, *, limit: int | None = None,
        force: bool = False) -> None:
    experiment.run("forced_demand", config["forced_demand"], run_stream, TEMPLATES, spec,
                   config, results_dir, limit=limit, force=force, show=_show)


# --- analysis -----------------------------------------------------------------


def table(df: pd.DataFrame, summary: pd.DataFrame, k: int, *, rank_col: str = "rank_wordlike",
          lens: str = "jlens") -> pd.DataFrame:
    """Per C_t: targets present per stream, presence by emission order (t1 = alphabetically
    first), the other groups and floors, and behaviour."""
    p = presence(df, k, rank_col=rank_col)
    p = p[p["lens"] == lens]
    emit = df[["stream_id", "c_t", "word", "emit_order"]].drop_duplicates()
    p = p.merge(emit, on=["stream_id", "c_t", "word"], how="left")
    out = []
    for c_t, g in p.groupby("c_t"):
        t = g[g.role == "target"]
        per_stream = t.groupby("stream_id")["present"].sum()
        s = summary[summary.c_t == c_t]
        row = {"c_t": c_t, "n": len(per_stream),
               "targets_mean": per_stream.mean(), "targets_sem": per_stream.sem()}
        row |= {f"t{i}": rate(t[t.emit_order == i]) for i in range(1, 7)}
        row |= {
            "replaced": rate(g[g.role == "replaced"]),
            "label_tracked": rate(g[g.role == "label_tracked"]),
            "label_untracked": rate(g[g.role == "label_untracked"]),
            "untracked": rate(g[g.role == "untracked"]),
            "floor_word": rate(g[g.role == "absent"]),
            "floor_label": rate(g[g.role == "label_absent"]),
            "q1_top1": s["q1_correct"].mean(),
            "q1_median_rank": s.loc[s.q1_expected_rank > 0, "q1_expected_rank"].median(),
            "list_correct": s["list_correct"].mean(),
        }
        out.append(row)
    return pd.DataFrame(out)


def single_cue_table(df: pd.DataFrame, k: int, *, rank_col: str = "rank_wordlike",
                     lens: str = "jlens") -> pd.DataFrame:
    """single_cue at its answer position: the comparison for the same streams."""
    p = presence(df, k, rank_col=rank_col)
    p = p[(p["lens"] == lens) & (p["readout_at"] == "answer")]
    return pd.DataFrame([{
        "c_t": c_t, "n": g["stream_id"].nunique(),
        "target_queried": rate(g[(g.role == "target") & g.is_queried]),
        "target_non_queried": rate(g[(g.role == "target") & ~g.is_queried]),
        "replaced_queried": rate(g[(g.role == "replaced") & g.is_queried]),
        "label_queried": rate(g[(g.role == "label_tracked") & g.is_queried]),
        "label_tracked": rate(g[(g.role == "label_tracked") & ~g.is_queried]),
        "label_untracked": rate(g[g.role == "label_untracked"]),
        "floor_word": rate(g[g.role == "absent"]),
        "floor_label": rate(g[g.role == "label_absent"]),
    } for c_t, g in p.groupby("c_t")])


def report(spec: dict, config: dict, results_dir: Path) -> None:
    k, fmt = config["readout"]["primary_k"], dict(index=False, float_format="%.3f")
    for label, df, summary in experiment.runs("forced_demand", spec["alias"], results_dir):
        for rank_col, lens in VIEWS:
            print(f"\nforced_demand {label} [{rank_col}, {lens}, in-band, k={k}]")
            print(table(df, summary, k, rank_col=rank_col, lens=lens).to_string(**fmt))
        try:
            single = experiment.load("single_cue", spec["alias"], results_dir)
            single = single.merge(df[["stream_id", "c_t"]].drop_duplicates(), on=["stream_id", "c_t"])
            for rank_col, lens in VIEWS:
                print(f"\nsingle_cue answer position, same streams [{rank_col}, {lens}]")
                print(single_cue_table(single, k, rank_col=rank_col, lens=lens).to_string(**fmt))
        except FileNotFoundError:
            print("\n(no single_cue shards for comparison)")
        print("\nforced_demand example outputs:")
        for _, r in summary.head(3).iterrows():
            print(f"  c_t={r.c_t} {r.list_raw!r} vs {r.sorted_targets!r}")
