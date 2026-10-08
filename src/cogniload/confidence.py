"""confidence: does the J-space carry uncertainty when the model does not know?

"What is the capital of {entity}?" for real and invented countries, prefilled
so the next token is the answer. Read at the `Answer:` prefill for uncertainty
words, words for something not existing, and neutral control words.

The cell that matters is an invented country the model answers with a name
anyway: no uncertainty word is about to be output there, so one present in the
J-space is not output staging.
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
ABSTAIN = {"unknown", "none", "n", "na", "unsure", "uncertain", "sorry", "i", "no", "not",
           "nothing", "there", "it", "this", "fictional", "fictitious", "nonexistent",
           "invalid", "undefined"}


def load_items(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return [{**r, "id": int(r["id"])} for r in csv.DictReader(f)]


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


def output_type(text: str, answer: str) -> str:
    """`correct`, `abstain`, or `guess` (a committed answer that is not the capital)."""
    first = re.findall(r"[A-Za-z]+", text)
    if answer and first and experiment.matches(first[0], answer):
        return "correct"
    return "abstain" if not first or first[0].casefold() in ABSTAIN else "guess"


def run_item(ctx: SimpleNamespace, item: dict, word_meta: dict[str, dict]) -> tuple[list[dict], dict]:
    tok, answer = ctx.model.tokenizer, item["answer"]
    text = build(tok, item["entity"], enable_thinking=ctx.spec["enable_thinking"])
    last = ctx.model.encode(text).shape[-1] - 1
    meta = dict(word_meta)
    if answer:
        meta[answer] = {"group": "answer", "concept": answer}
    meta = {w: {**m, "id": item["id"]} for w, m in meta.items()}
    rows, logits = experiment.read_both(ctx, text, meta, {last: "answer"})

    logits = logits[0]
    top = logits.topk(5)
    decoded = [tok.decode([i]) for i in top.indices.tolist()]
    generated = experiment.free_text(ctx.model, text,
                                     max_new_tokens=ctx.config["confidence"]["max_new_tokens"])
    summary = {
        **item, "answer_pos": last, "top1": decoded[0],
        "top5": " | ".join(f"{t}:{v:.2f}" for t, v in zip(decoded, top.values.tolist())),
        # Rank of the capital in the model's own next-token distribution.
        "expected_rank": int(readout.rank_of(
            logits[None], torch.tensor([readout.token_id(tok, answer)]))[0, 0]) if answer else None,
        "generated": generated, "output_type": output_type(generated, answer),
    }
    return rows, summary


def _paths(alias: str, results_dir: Path, limit: bool) -> tuple[Path, Path]:
    out, suffix = Path(results_dir) / alias, "_limit" if limit else ""
    return out / f"confidence{suffix}.parquet", out / f"confidence_summary{suffix}.parquet"


def run(spec: dict, config: dict, results_dir: Path, *, limit: int | None = None,
        force: bool = False) -> None:
    """Run every item, or the first `limit` per condition; write rows, summary and manifest."""
    shard, summary_path = _paths(spec["alias"], results_dir, bool(limit))
    if shard.exists() and not force:
        print(f"{shard.name} exists, skipping (use --force to redo)")
        return
    layers = experiment.recorded_layers(spec, config)
    items = load_items(ROOT / config["confidence"]["stimuli"])
    if limit:
        items = [i for c in ("real", "fictitious")
                 for i in [x for x in items if x["condition"] == c][:limit]]
    model, lens = registry.load(spec)
    ctx = SimpleNamespace(model=model, lens=lens, spec=spec, config=config, layers=layers,
                          band=spec["band"])
    word_meta = words(model.tokenizer)
    print(f"confidence: {len(items)} items, layers {layers[0]}-{layers[-1]}, band {spec['band']}")

    rows, summaries = [], []
    for item in items:
        item_rows, summary = run_item(ctx, item, word_meta)
        rows += item_rows
        summaries.append(summary)
        print(f"  {item['condition']:<10} {item['entity']:<13} -> {summary['generated']!r}")

    shard.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(shard, index=False)
    pd.DataFrame(summaries).to_parquet(summary_path, index=False)
    (shard.parent / "confidence_manifest.json").write_text(json.dumps(
        {"spec": spec, "templates": TEMPLATES, "groups": GROUPS, "words": word_meta,
         "config": config}, indent=2))


# --- analysis -----------------------------------------------------------------


def table(df: pd.DataFrame, summary: pd.DataFrame, k: int, *, rank_col: str = "rank_wordlike",
          lens: str = "jlens") -> pd.DataFrame:
    """Per condition and output type: the share of each group's concepts present.

    A concept is present if any of its forms has band-min rank <= k.
    `uncertain_any` is the share of items with at least one uncertainty concept
    present. Read every column against `control`.
    """
    rows = df[df["in_band"] & (df["lens"] == lens)]
    present = (rows.groupby(["id", "group", "concept"])[rank_col].min() <= k).rename("present")
    per_item = present.groupby(["id", "group"]).mean().unstack("group")
    per_item["uncertain_any"] = present.xs("uncertain", level="group").groupby("id").any()
    merged = summary.set_index("id")[["condition", "output_type"]].join(per_item)
    out = merged.groupby(["condition", "output_type"]).mean(numeric_only=True)
    out.insert(0, "n", merged.groupby(["condition", "output_type"]).size())
    return out.reset_index()


def report(spec: dict, config: dict, results_dir: Path) -> None:
    k, fmt = config["readout"]["primary_k"], dict(index=False, float_format="%.3f")
    for limit, label in ((True, "SMOKE (--limit)"), (False, "FULL")):
        shard, summary_path = _paths(spec["alias"], results_dir, limit)
        if not shard.exists():
            continue
        df, summary = pd.read_parquet(shard), pd.read_parquet(summary_path)
        for rank_col, lens in VIEWS:
            print(f"\nconfidence {label} [{rank_col}, {lens}, in-band, k={k}]")
            print(table(df, summary, k, rank_col=rank_col, lens=lens).to_string(**fmt))
        print(f"\nconfidence {label} example outputs:")
        for _, r in summary.groupby("condition").head(5).iterrows():
            print(f"  {r.condition:<10} {r.entity:<13} -> {r.generated!r} ({r.output_type})")
