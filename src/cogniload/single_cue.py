"""single_cue: the keep-track task, with one tracked category queried.

Read at three positions per stream:

- `in_stream`: the comma after the second-to-last word. Every target has been
  seen, none is privileged, and the query is not yet known, so capacity and
  selectivity are measured here.
- `stream_end`: the final `.`, where the model continues the instruction
  rather than the list. A contrast.
- `answer`: the `Answer:` prefill. The queried target is about to be emitted.

A non-queried target is task state the model is not about to say, so it is the
item whose presence separates "holds task state" from "holds the next output".
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from cogniload import experiment, prompts
from cogniload.experiment import presence

# Self-report of every tracked word, generated from a fresh render so no
# category is already answered in context.
LIST_ALL = ("List the most recent word for each tracked category, in order. "
            "Give one line per category as 'category: word', with no explanation.")
TEMPLATES = {"stream": prompts.KEEP_TRACK, "question": prompts.RECENT,
             "prefill": prompts.PREFILL, "list_all": LIST_ALL}


def build_list_all(tokenizer: Any, words: list[str], tracked: tuple[str, ...], *,
                   enable_thinking: bool) -> str:
    return prompts.render_chat(tokenizer, prompts.stream_turn(words, tracked, LIST_ALL),
                               enable_thinking=enable_thinking)


def run_stream(ctx, stream) -> tuple[list[dict], dict]:
    tok, queried = ctx.model.tokenizer, stream.queried_category
    thinking = ctx.spec["enable_thinking"]
    text = prompts.build_recent(tok, stream.words, stream.tracked, queried,
                                enable_thinking=thinking)
    *_, in_stream, stream_end = experiment.delimiters(tok, text, stream.words)
    answer = ctx.model.encode(text).shape[-1] - 1
    meta = experiment.words_and_meta(stream, ctx.pool, ctx.config["stream"]["n_absent_words"])
    rows, logits = experiment.read_both(
        ctx, text, meta, {in_stream: "in_stream", stream_end: "stream_end", answer: "answer"})
    # Scored from the readout pass's own logits, so answer and ranks agree.
    q1 = experiment.score_q1(tok, logits[-1], stream.targets[queried])

    q2_text = experiment.free_text(
        ctx.model, build_list_all(tok, stream.words, stream.tracked, enable_thinking=thinking),
        max_new_tokens=ctx.config["single_cue"]["list_max_new_tokens"])
    q2 = score_q2(q2_text, stream.tracked, stream.targets, stream.words)
    q2_by_category = {r["category"]: r for r in q2}
    for row in rows:
        q2_row = q2_by_category.get(row["category"]) if row["role"] == "target" else None
        row["q2_parsed"] = q2_row["parsed"] if q2_row else None
        row["q2_correct"] = q2_row["correct"] if q2_row else None

    summary = {
        "stream_id": stream.stream_id, "c_t": stream.c_t, "queried_category": queried,
        "in_stream_pos": in_stream, "stream_end_pos": stream_end, "answer_pos": answer,
        "q2_parse_rate": sum(r["parsed"] for r in q2) / len(q2),
        "q2_accuracy": sum(r["correct"] for r in q2) / len(q2),
        "q2_raw": q2_text,
        **{f"q1_{k}": v for k, v in q1.items()},
    }
    return rows, summary


def _show(c_t: int, s: pd.DataFrame) -> None:
    ranks = s["q1_expected_rank"]
    print(f"  C_t={c_t}: Q1 top-1 {s['q1_correct'].mean():.0%} / top-5 {(ranks <= 5).mean():.0%}, "
          f"Q2 {s['q2_accuracy'].mean():.0%} (parse {s['q2_parse_rate'].mean():.0%})")


def run(spec: dict, config: dict, results_dir: Path, *, limit: int | None = None,
        force: bool = False) -> None:
    experiment.run("single_cue", config["single_cue"], run_stream, TEMPLATES, spec, config,
                   results_dir, limit=limit, force=force, show=_show)


# --- list-all self-report -------------------------------------------------------


def parse_q2(text: str, tracked: Sequence[str], vocabulary: Sequence[str]) -> dict[str, str]:
    """One word per tracked category from the free-text list, if found.

    Keyed on category names ("animal: cat"); failing that, stream words in
    order are aligned to `tracked`. A missing category is a parse failure.
    """
    lowered = text.casefold()
    found = {}
    for category in tracked:
        match = re.search(re.escape(category.casefold()) + r"\s*[:\-–]\s*([A-Za-z]+)", lowered)
        if match:
            found[category] = match.group(1)
    if found:
        return found
    vocab = {w.casefold(): w for w in vocabulary}
    return dict(zip(tracked, [vocab[w] for w in re.findall(r"[A-Za-z]+", lowered) if w in vocab]))


def score_q2(text: str, tracked: Sequence[str], targets: dict[str, str],
             vocabulary: Sequence[str]) -> list[dict]:
    """One row per tracked category. `parsed` keeps "unreadable" from scoring as "wrong"."""
    found = parse_q2(text, tracked, vocabulary)
    return [{"category": c, "parsed": c in found, "parsed_word": found.get(c),
             "target_word": targets[c],
             "correct": c in found and experiment.matches(found[c], targets[c])}
            for c in tracked]


# --- analysis -----------------------------------------------------------------


def q1_diagnostic(summary: pd.DataFrame, n_examples: int = 6) -> None:
    """What the model emits at the answer position, to tell a wrong word from a format problem."""
    print(f"\nQ1 DIAGNOSTIC — accuracy {summary['q1_correct'].mean():.0%} "
          f"over {len(summary)} streams")
    print("  most common emitted token at the answer position:")
    for token, n in summary["q1_answer"].value_counts().head(10).items():
        print(f"    {token!r:<24} {n:>4}  ({n / len(summary):.0%})")
    # Stored shards can hold -1 for an expected word that is not one token.
    ranked = summary[summary["q1_expected_rank"] > 0]["q1_expected_rank"]
    if len(ranked):
        print(f"  expected word's rank in the model distribution: "
              f"median {ranked.median():.0f}, "
              f"top-5 {(ranked <= 5).mean():.0%}, top-1 {(ranked == 1).mean():.0%}")
    print("  per C_t: " + ", ".join(
        f"C_t={c}: {g['q1_correct'].mean():.0%}" for c, g in summary.groupby("c_t")))
    print(f"\n  first {n_examples} streams (expected -> answer | top5):")
    for _, r in summary.head(n_examples).iterrows():
        print(f"    {r['q1_expected']!r:<12} -> {r['q1_answer']!r:<14} | {r['q1_top5']}")
    print(f"\n  Q2 accuracy {summary['q2_accuracy'].mean():.0%}, "
          f"parse rate {summary['q2_parse_rate'].mean():.0%}")
    print("  example Q2 output:")
    for raw in summary["q2_raw"].head(2):
        print(f"    {raw!r}")


def floor_check(df: pd.DataFrame, k: int) -> None:
    """Is the queried target present at the answer position at all?

    Both rank columns, because which carries the signal is model-dependent: on
    Qwen3.6-27B `<|im_end|>` and underscore runs flood the full-vocab top-25.
    Below 50% on both, every null downstream is uninterpretable.
    """
    print(f"\nFLOOR CHECK (k={k})")
    best = 0.0
    for col in ("rank", "rank_wordlike"):
        p = presence(df, k, rank_col=col)
        p = p[p["lens"] == "jlens"]
        q = p[(p.readout_at == "answer") & p.is_queried & (p.role == "target")]
        a = p[p.role == "absent"]
        best = max(best, q["present"].mean())
        print(f"  [{col:<14}] queried target at answer: {q['present'].mean():.0%}"
              f"   absent-word floor: {a['present'].mean():.0%}")
    if best < 0.5:
        print("  *** BELOW 50% ON BOTH COLUMNS — diagnose before trusting anything below ***")


def by_layer(df: pd.DataFrame, k: int, *, rank_col: str = "rank") -> pd.DataFrame:
    """Present rate per layer: shows whether targets peak outside the band."""
    rows = df[df["lens"] == "jlens"].copy()
    rows["group"] = "other"
    rows.loc[(rows["role"] == "target") & ~rows["is_queried"], "group"] = "target_non_queried"
    rows.loc[(rows["role"] == "target") & rows["is_queried"], "group"] = "target_queried"
    rows.loc[rows["role"] == "absent", "group"] = "absent"
    rows = rows[rows["group"] != "other"]
    rows["present"] = rows[rank_col] <= k
    return (
        rows.groupby(["readout_at", "layer", "in_band", "group"])["present"]
        .mean().reset_index()
        .pivot_table(index=["readout_at", "layer", "in_band"], columns="group", values="present")
        .reset_index()
    )


def capacity(df: pd.DataFrame, k: int, *, rank_col: str = "rank") -> pd.DataFrame:
    """Non-queried targets present per stream, vs C_t.

    A curve that rises then flattens is a ceiling; one flat at ~1 is the
    deflationary result, so the rise is the part that matters.
    """
    p = presence(df, k, rank_col=rank_col)
    targets = p[(p["role"] == "target") & ~p["is_queried"]]
    per_stream = (targets.groupby(["lens", "readout_at", "c_t", "stream_id"])["present"]
                  .sum().reset_index(name="n_present"))
    return (per_stream.groupby(["lens", "readout_at", "c_t"])["n_present"]
            .agg(["mean", "sem", "count"]).reset_index())


def selectivity(df: pd.DataFrame, k: int, tail_guard: int, *,
                rank_col: str = "rank") -> pd.DataFrame:
    """Non-queried targets vs the untracked final occurrence, on common recency support.

    Recency <= `tail_guard` is dropped: no target can sit there, so role and
    recency are confounded.
    """
    all_items = presence(df, k, rank_col=rank_col)
    p = all_items[all_items["recency"].notna() & (all_items["recency"] > tail_guard)]
    target, untracked, replaced = p["role"] == "target", p["role"] == "untracked", p["role"] == "replaced"
    groups = {
        "target_non_queried": p[target & ~p["is_queried"]],
        "untracked_final": p[untracked & p["is_category_final"]],
        "untracked_superseded": p[untracked & ~p["is_category_final"]],
        # An earlier word of the queried category competes as an answer; one of
        # another tracked category does not.
        "replaced_queried": p[replaced & p["is_queried"]],
        "replaced_other": p[replaced & ~p["is_queried"]],
        "absent": all_items[all_items["role"] == "absent"],
    }
    both = pd.concat([g.assign(group=name) for name, g in groups.items()], ignore_index=True)
    return (both.groupby(["lens", "readout_at", "c_t", "group"])["present"]
            .agg(["mean", "sem", "count"]).reset_index())


def headline(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Present rate per group, pooled over C_t, J-lens, both rank columns. Read against `absent`."""
    out = []
    for rank_col in ("rank", "rank_wordlike"):
        p = presence(df, k, rank_col=rank_col)
        p = p[p["lens"] == "jlens"]
        groups = {
            "target_queried": p[(p.role == "target") & p.is_queried],
            "target_non_queried": p[(p.role == "target") & ~p.is_queried],
            "replaced_queried": p[(p.role == "replaced") & p.is_queried],
            "untracked": p[p.role == "untracked"],
            "absent": p[p.role == "absent"],
            "label_queried": p[(p.role == "label_tracked") & p.is_queried],
            "label_tracked": p[(p.role == "label_tracked") & ~p.is_queried],
            "label_untracked": p[p.role == "label_untracked"],
            "label_absent": p[p.role == "label_absent"],
        }
        for at in ("in_stream", "stream_end", "answer"):
            for name, g in groups.items():
                g = g[g.readout_at == at]
                if len(g):
                    out.append({"rank_col": rank_col, "readout_at": at, "group": name,
                                "present": g["present"].mean(), "n": len(g)})
    wide = pd.DataFrame(out).pivot_table(
        index=["readout_at", "group"], columns="rank_col", values=["present", "n"]).reset_index()
    order = ["in_stream", "stream_end", "answer"]
    wide["_o"] = wide["readout_at"].map(order.index)
    return wide.sort_values(["_o", "group"]).drop(columns=["_o"], level=0)


def report(spec: dict, config: dict, results_dir: Path) -> None:
    k, tail_guard = config["readout"]["primary_k"], config["stream"]["tail_guard"]
    for label, df, summary in experiment.runs("single_cue", spec["alias"], results_dir):
        print(f"\n{'=' * 70}\n{label}: {len(df):,} rows\n{'=' * 70}")
        q1_diagnostic(summary)
        floor_check(df, k)

        print(f"\nHEADLINE — present rate, pooled over C_t (k={k}, J-lens)")
        print("  compare every row against `absent`, the chance floor")
        print(headline(df, k).to_string(index=False, float_format="%.3f"))
        print(f"\nCAPACITY — non-queried targets present (k={k})")
        print(capacity(df, k).to_string(index=False))
        print(f"\nSELECTIVITY / EVICTION — present rate by group (k={k})")
        print(selectivity(df, k, tail_guard).to_string(index=False))
        print(f"\nPER-LAYER — present rate by layer (k={k}, J-lens)")
        print(by_layer(df, k).to_string(index=False, float_format="%.2f"))

        print("\nk-SWEEP — non-queried targets at in_stream")
        sweep = []
        for kk in config["readout"]["k_values"]:
            c = capacity(df, kk)
            sweep.append(c[(c["lens"] == "jlens") & (c["readout_at"] == "in_stream")].assign(k=kk))
        print(pd.concat(sweep).pivot_table(index="k", columns="c_t", values="mean").to_string())
