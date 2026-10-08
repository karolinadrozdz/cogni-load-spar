"""confidence: does the J-space carry uncertainty when the model does not know?

"What is the capital of {entity}?" for real countries, invented countries and
real regions from famous to obscure, prefilled so the next token is the answer.
Read at the `Answer:` prefill in two ways: a fixed list of uncertainty words,
words for something not existing and neutral control words; and the top
word-like tokens at each layer, with nothing chosen in advance.

The comparison that matters is among items the model answers with a name: the
ones it gets right against the ones it gets wrong. The output looks the same,
so a difference in the J-space is not output staging.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch

from cogniload import experiment, prompts, readout, registry
from cogniload.experiment import VIEWS

ROOT = registry.REGISTRY.parents[1]

# "Answer in one word": without it Qwen answers in markdown (argmax ` **`).
QUESTION = "What is the capital of {entity}? Answer in one word."
TEMPLATES = {"question": QUESTION, "prefill": prompts.PREFILL}

#: Words read at the answer position, fixed before any run. `control` is the
#: floor: common words with nothing to do with knowing.
GROUPS: dict[str, list[str]] = {
    "uncertain": ["unknown", "unsure", "uncertain", "guess", "maybe", "perhaps",
                  "unclear", "doubt"],
    "nonexistent": ["fictional", "fake", "imaginary", "nonexistent"],
    "control": ["window", "yellow", "simple", "river", "table", "garden", "silver",
                "music", "bridge", "letter", "winter", "market"],
}
#: First words of an answer that declines to name a capital.
ABSTAIN = {"unknown", "none", "unsure", "uncertain", "sorry", "i", "no", "not", "nothing",
           "there", "it", "this", "fictional", "fictitious", "nonexistent", "invalid",
           "undefined"}


def load_items(path: Path) -> list[dict]:
    """The items of one stimuli file; `links` is only set where the file has it."""
    with Path(path).open(encoding="utf-8", newline="") as f:
        return [{**r, "id": int(r["id"]), "links": int(r["links"]) if r.get("links") else None}
                for r in csv.DictReader(f)]


def words(tokenizer: Any) -> dict[str, dict]:
    """Every token form to read, mapped to its group and concept.

    A concept is read in lower case and capitalised, whichever are one token,
    and is present if either form is.
    """
    out = {}
    for group, concepts in GROUPS.items():
        for concept in concepts:
            forms = [f for f in (concept, concept.capitalize())
                     if len(tokenizer.encode(f" {f}", add_special_tokens=False)) == 1]
            if not forms:
                raise ValueError(f"{concept!r} has no single-token form")
            out |= {f: {"group": group, "concept": concept} for f in forms}
    return out


def build(tokenizer: Any, entity: str, *, enable_thinking: bool) -> str:
    return prompts.render_chat(tokenizer, QUESTION.format(entity=entity),
                               enable_thinking=enable_thinking, prefill=prompts.PREFILL)


def output_type(text: str, answer: str, entity: str = "") -> str:
    """`correct`, `abstain`, `echo` (the entity's own name given back) or `guess`
    (another name: a committed answer that is not the capital).

    Judged on the first word. An apostrophe stays inside it, so "N'Djamena" is a
    name, while "I'm" and "There's" abstain on the part before the apostrophe.
    """
    words = [w.strip("'") for w in re.findall(r"[A-Za-z']+", text)]
    words = [w for w in words if w]
    if not words or re.match(r"\s*n/?a\b", text, re.IGNORECASE):
        return "abstain"
    first = words[0].casefold()
    if answer and first == answer.casefold():
        return "correct"
    if first in ABSTAIN or first.split("'")[0] in ABSTAIN:
        return "abstain"
    # A word of the entity's name, whole or cut short ("Jorvan" for Jorvania).
    name = re.findall(r"[a-z']+", entity.split(",")[0].casefold())
    return "echo" if any(w.startswith(first) for w in name) else "guess"


def run_item(ctx: SimpleNamespace, item: dict,
             word_meta: dict[str, dict]) -> tuple[list[dict], list[dict], dict]:
    """One item: fixed-list rows, top-token rows, and the summary row."""
    tok, answer = ctx.model.tokenizer, item["answer"]
    text = build(tok, item["entity"], enable_thinking=ctx.spec["enable_thinking"])
    last = ctx.model.encode(text).shape[-1] - 1
    meta = dict(word_meta)
    if answer:
        meta[answer] = {"group": "answer", "concept": answer}
    meta = {w: {**m, "id": item["id"]} for w, m in meta.items()}
    rows, logits = experiment.read_both(ctx, text, meta, {last: "answer"})
    top_rows = [{**r, "id": item["id"], "lens": name}
                for name, jacobian in (("jlens", True), ("logit", False))
                for r in readout.top_tokens(ctx.model, ctx.lens, text, layers=ctx.layers,
                                            position=last, k=ctx.config["confidence"]["top_k"],
                                            use_jacobian=jacobian)]

    logits = logits[0]
    top = logits.topk(5)
    decoded = [tok.decode([i]) for i in top.indices.tolist()]
    generated = experiment.free_text(ctx.model, text,
                                     max_new_tokens=ctx.config["confidence"]["max_new_tokens"])
    # Each group's best rank in the model's own next-token distribution: a word
    # that is a runner-up for the output is not held apart from it.
    ids = torch.tensor([readout.token_id(tok, w) for w in word_meta])
    out_rank = pd.Series(readout.rank_of(logits[None], ids)[0].tolist()).groupby(
        [m["group"] for m in word_meta.values()]).min()
    summary = {
        **{f"output_rank_{g}": int(r) for g, r in out_rank.items()},
        **item, "answer_pos": last, "top1": decoded[0],
        # How sure the model is of its first token.
        "top1_prob": float(logits.float().softmax(-1).max()),
        "top5": " | ".join(f"{t}:{v:.2f}" for t, v in zip(decoded, top.values.tolist())),
        # Rank of the capital in the model's own next-token distribution.
        "expected_rank": int(readout.rank_of(
            logits[None], torch.tensor([readout.token_id(tok, answer)]))[0, 0]) if answer else None,
        "generated": generated, "output_type": output_type(generated, answer, item["entity"]),
    }
    return rows, top_rows, summary


def _paths(alias: str, results_dir: Path, name: str, limit: bool) -> tuple[Path, Path, Path]:
    """Fixed-list rows, top-token rows, and the per-item summary, for stimuli set `name`."""
    stem, suffix = Path(results_dir) / alias / f"confidence_{name}", "_limit" if limit else ""
    return (stem.with_name(f"{stem.name}{suffix}.parquet"),
            stem.with_name(f"{stem.name}_top{suffix}.parquet"),
            stem.with_name(f"{stem.name}_summary{suffix}.parquet"))


def run(spec: dict, config: dict, results_dir: Path, *, stimuli_set: str,
        limit: int | None = None, force: bool = False) -> None:
    """Run one stimuli set, or its first `limit` items per condition; write rows,
    top tokens, summary and manifest."""
    sets = config["confidence"]["sets"]
    if stimuli_set not in sets:
        raise KeyError(f"unknown stimuli set {stimuli_set!r}; config has: {', '.join(sets)}")
    shard, top_path, summary_path = _paths(spec["alias"], results_dir, stimuli_set, bool(limit))
    if shard.exists() and not force:
        print(f"{shard.name} exists, skipping (use --force to redo)")
        return
    layers = experiment.recorded_layers(spec, config)
    items = load_items(ROOT / sets[stimuli_set])
    if limit:
        items = [i for c in dict.fromkeys(x["condition"] for x in items)
                 for i in [x for x in items if x["condition"] == c][:limit]]
    model, lens = registry.load(spec)
    ctx = SimpleNamespace(model=model, lens=lens, spec=spec, config=config, layers=layers,
                          band=spec["band"])
    word_meta = words(model.tokenizer)
    print(f"confidence {stimuli_set}: {len(items)} items, layers {layers[0]}-{layers[-1]}, "
          f"band {spec['band']}")

    rows, top_rows, summaries = [], [], []
    for item in items:
        item_rows, item_top, summary = run_item(ctx, item, word_meta)
        rows += item_rows
        top_rows += item_top
        summaries.append(summary)
        print(f"  {item['condition']:<10} {item['entity']:<13} -> {summary['generated']!r}")

    shard.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(shard, index=False)
    pd.DataFrame(top_rows).to_parquet(top_path, index=False)
    pd.DataFrame(summaries).to_parquet(summary_path, index=False)
    manifest = shard.with_name(shard.stem + "_manifest.json")
    manifest.write_text(json.dumps(
        {"spec": spec, "stimuli": sets[stimuli_set], "limit": limit, "templates": TEMPLATES,
         "groups": GROUPS, "words": word_meta, "config": config}, indent=2))


# --- analysis -----------------------------------------------------------------


def table(df: pd.DataFrame, summary: pd.DataFrame, k: int, *, rank_col: str = "rank_wordlike",
          lens: str = "jlens", layers: tuple[int, int] | None = None) -> pd.DataFrame:
    """Per condition and output type, the mean share of each group's concepts present.

    A concept is present if any of its forms has a minimum rank <= k over the
    band, or over the half-open layer range `layers` when that is given.
    `uncertain_any` is the share of items with at least one uncertainty concept
    present. Read every column against `control`. `out_*` is the share of items
    where a word of that group is in the model's own top-k next tokens: where
    it matches the lens columns, the lens is showing runners-up for the output.
    """
    window = df["in_band"] if layers is None else df["layer"].between(layers[0], layers[1] - 1)
    rows = df[window & (df["lens"] == lens)]
    present = (rows.groupby(["id", "group", "concept"])[rank_col].min() <= k).rename("present")
    per_item = present.groupby(["id", "group"]).mean().unstack("group")
    per_item["uncertain_any"] = present.xs("uncertain", level="group").groupby("id").any()
    summary = summary.set_index("id")
    for group in GROUPS:
        per_item[f"out_{group}"] = summary[f"output_rank_{group}"] <= k
    merged = summary[["condition", "output_type"]].join(per_item)
    out = merged.groupby(["condition", "output_type"]).mean(numeric_only=True)
    out.insert(0, "n", merged.groupby(["condition", "output_type"]).size())
    return out.reset_index()


def by_layer(df: pd.DataFrame, summary: pd.DataFrame, k: int, *, rank_col: str = "rank_wordlike",
             lens: str = "jlens") -> pd.DataFrame:
    """Per layer: the share of each group's concepts present, by condition."""
    rows = df[df["lens"] == lens].merge(summary[["id", "condition"]], on="id")
    present = rows.groupby(["layer", "condition", "id", "group", "concept"])[rank_col].min() <= k
    return present.groupby(["layer", "condition", "group"]).mean().unstack(["condition", "group"])


def enriched(top: pd.DataFrame, ids_a: pd.Series, ids_b: pd.Series, *, layers: tuple[int, int],
             lens: str = "jlens", n: int = 15) -> pd.DataFrame:
    """Tokens in the top-k of more items of set A than of set B, over `layers`.

    Exploratory: it shows what the lens holds without a word list. A token
    counts once per item, lower-cased and stripped.
    """
    rows = top[(top["lens"] == lens) & top["layer"].between(layers[0], layers[1] - 1)]
    rows = rows.assign(token=rows["token"].str.strip().str.lower())
    seen = rows.drop_duplicates(["id", "token"])

    def share(ids: pd.Series) -> pd.Series:
        return seen[seen["id"].isin(ids)].groupby("token").size() / max(len(ids), 1)

    out = pd.concat({"a": share(ids_a), "b": share(ids_b)}, axis=1).fillna(0.0)
    out["diff"] = out["a"] - out["b"]
    return out.sort_values("diff", ascending=False).head(n).rename_axis("token").reset_index()


def report(spec: dict, config: dict, results_dir: Path) -> None:
    k, fmt = config["readout"]["primary_k"], dict(index=False, float_format="%.3f")
    lo, hi = config["confidence"]["early_fraction"]
    early = (int(lo * spec["n_layers"]), int(hi * spec["n_layers"]))
    windows = (("in-band", None), (f"early layers {early[0]}-{early[1] - 1}", early))
    runs = [(name, limit, f"{name} {kind}") for name in config["confidence"]["sets"]
            for limit, kind in ((True, "SMOKE (--limit)"), (False, "FULL"))]
    for stimuli_set, limit, label in runs:
        shard, top_path, summary_path = _paths(spec["alias"], results_dir, stimuli_set, limit)
        if not shard.exists():
            continue
        df, summary = pd.read_parquet(shard), pd.read_parquet(summary_path)
        # Typed again from the stored text, so a change to `output_type` reaches old runs.
        summary["output_type"] = [output_type(g, a, e) for g, a, e in
                                  zip(summary["generated"], summary["answer"], summary["entity"])]
        for name, layers in windows:
            for rank_col, lens in VIEWS:
                print(f"\nconfidence {label} [{rank_col}, {lens}, {name}, k={k}]")
                print(table(df, summary, k, rank_col=rank_col, lens=lens,
                            layers=layers).to_string(**fmt))
        for lens in ("jlens", "logit"):
            print(f"\nconfidence {label} per layer [rank_wordlike, {lens}, k={k}]")
            print(by_layer(df, summary, k, lens=lens).to_string(float_format="%.3f"))
        print(f"\nconfidence {label} example outputs:")
        for _, r in summary.groupby(["condition", "output_type"]).head(4).iterrows():
            print(f"  {r.condition:<10} {r.entity:<13} -> {r.generated!r} ({r.output_type})")
        if not top_path.exists():  # a run from before top tokens were stored
            continue
        top, band = pd.read_parquet(top_path), tuple(spec["band"])
        ids = {key: g["id"] for key, g in summary.groupby(["condition", "output_type"])}
        none = pd.Series(dtype=int)
        contrasts = (
            ("region: wrong name (a) vs correct (b)",
             ids.get(("region", "guess"), none), ids.get(("region", "correct"), none)),
            ("fictitious (a) vs real (b)",
             summary[summary.condition == "fictitious"]["id"],
             summary[summary.condition == "real"]["id"]),
        )
        for title, a, b in contrasts:
            if not len(a) or not len(b):
                continue
            for name, layers in (("in-band", band), windows[1]):
                print(f"\nconfidence {label} top tokens, {title} "
                      f"[n={len(a)} vs {len(b)}, jlens, {name}]")
                print(enriched(top, a, b, layers=layers).to_string(**fmt))
