"""Blocks every experiment shares: layers, streams, the words read, the two-lens
readout, scoring, the per-C_t shard loop, and loading results back.

An experiment module supplies `run_stream(ctx, stream) -> (rows, summary)` and
its own tables; everything else is here.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch

from cogniload import exemplars, readout, registry, stimuli

#: (rank column, lens) views each experiment's table is printed for.
VIEWS = [("rank_wordlike", "jlens"), ("rank", "jlens"), ("rank_wordlike", "logit")]


def layer_range(n_layers: int, fraction: tuple[float, float]) -> list[int]:
    """Layers in `[lo * n, hi * n)` that the lens fits (it has no final-layer matrix)."""
    lo, hi = fraction
    return [l for l in range(int(lo * n_layers), int(hi * n_layers)) if l < n_layers - 1]


def recorded_layers(spec: dict, config: dict) -> list[int]:
    """The layers each readout records. The band must sit inside them, or band-min
    would silently cover only part of it."""
    layers = layer_range(spec["n_layers"], config["readout"]["record_layer_fraction"])
    if spec["band"] is None:
        raise FileNotFoundError(f"no band for alias {spec['alias']!r}; run find_band first")
    if not set(range(*spec["band"])) <= set(layers):
        raise ValueError(f"band {spec['band']} is not inside the recorded layers "
                         f"{layers[0]}-{layers[-1]}; widen readout.record_layer_fraction")
    return layers


def streams(pool: dict, config: dict, c_t: int, n_streams: int) -> list[stimuli.Stream]:
    """Seeded per C_t, so `(stream_id, c_t)` names the same stream in every experiment."""
    s = config["stream"]
    return stimuli.generate(pool, c_t=c_t, n_streams=n_streams, seed=config["seed"] + 1000 * c_t,
                            updates_per_tracked=s["updates_per_tracked"], tail_guard=s["tail_guard"])


def words_and_meta(stream: stimuli.Stream, pool: dict, n_absent: int) -> dict[str, dict]:
    """Every word to read, mapped to its role row.

    Stream words; `n_absent` same-category words not in the stream, which give
    "present" a chance floor; and the category labels, tracked or not, with
    labels that never occur as their floor.
    """
    used = set(stream.words)
    candidates = sorted(w for words in pool.values() for w in words if w not in used)
    absent = random.Random(stream.stream_id).sample(candidates, min(n_absent, len(candidates)))
    blank = {"stream_id": stream.stream_id, "c_t": stream.c_t,
             "queried_category": stream.queried_category, "stream_pos": None,
             "recency": None, "category": None, "role": "absent",
             "is_category_final": False, "is_queried": False}
    meta = {r["word"]: r for r in stream.to_rows()}
    meta |= {w: {**blank, "word": w} for w in absent}
    for category, forms in exemplars.CATEGORY_LABELS.items():
        role = "label_tracked" if category in stream.tracked else "label_untracked"
        meta |= {w: {**blank, "word": w, "category": category, "role": role,
                     "is_queried": category == stream.queried_category} for w in forms}
    meta |= {w: {**blank, "word": w, "role": "label_absent"} for w in exemplars.ABSENT_LABELS}
    return meta


def read_both(ctx: SimpleNamespace, text: str, meta: dict[str, dict],
              positions: dict[int, str]) -> tuple[list[dict], torch.Tensor]:
    """J-lens and logit-lens readout of every word in `meta` at `positions` (token -> name).

    Returns the rows, each carrying its word's metadata, and the model's own
    logits at `positions` from the J-lens pass.
    """
    words = list(meta)
    ids = [readout.token_id(ctx.model.tokenizer, w) for w in words]
    rows, model_logits = [], None
    for lens_name, jacobian in (("jlens", True), ("logit", False)):
        result, logits = readout.read(ctx.model, ctx.lens, text, words, ids, layers=ctx.layers,
                                      positions=list(positions), use_jacobian=jacobian)
        model_logits = logits if model_logits is None else model_logits
        rows += [{**r, **meta[r["word"]], "lens": lens_name, "readout_at": positions[r["token_pos"]],
                  "in_band": ctx.band[0] <= r["layer"] < ctx.band[1]} for r in result]
    return rows, model_logits


def delimiters(tokenizer: Any, text: str, pieces: list[str]) -> list[int]:
    """Token index of the `,` or `.` after each stream piece, located by character offset.

    Reads go at the delimiter, never the word: a word is trivially top-ranked
    at its own position.
    """
    lead = "Here is a sequence of words: "
    char = text.index(lead + ", ".join(pieces)) + len(lead)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    out = []
    for piece in pieces:
        char += len(piece)
        index = next(i for i, (a, b) in enumerate(enc["offset_mapping"]) if a <= char < b)
        token = tokenizer.decode([enc["input_ids"][index]])
        assert "," in token or "." in token, f"expected a delimiter after {piece!r}, got {token!r}"
        out.append(index)
        char += 2
    return out


def matches(a: str, b: str) -> bool:
    """A decoded token has a leading space and may differ in case; neither is an error."""
    return a.strip().casefold() == b.strip().casefold()


def score_q1(tokenizer: Any, logits: torch.Tensor, expected: str) -> dict:
    """Score the next token against `expected`.

    `expected_rank` survives a formatting token winning the argmax, where
    `correct` does not; `top5` tells a near miss from a total miss.
    """
    top = logits.topk(5)
    decoded = [tokenizer.decode([i]) for i in top.indices.tolist()]
    target = torch.tensor([readout.token_id(tokenizer, expected)])
    return {"expected": expected, "answer": decoded[0], "correct": matches(decoded[0], expected),
            "expected_rank": int(readout.rank_of(logits[None], target)[0, 0]),
            "top5": " | ".join(f"{t}:{v:.2f}" for t, v in zip(decoded, top.values.tolist()))}


@torch.no_grad()
def new_tokens(model: Any, prompt: str, max_new_tokens: int) -> list[int]:
    """Greedy continuation ids, through the wrapped HF model (`model.forward` has no lm_head)."""
    ids = model.encode(prompt)
    out = model._hf_model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                   max_new_tokens=max_new_tokens,
                                   pad_token_id=model.tokenizer.eos_token_id)
    return out[0, ids.shape[-1]:].tolist()


def free_text(model: Any, prompt: str, *, max_new_tokens: int) -> str:
    return model.tokenizer.decode(new_tokens(model, prompt, max_new_tokens),
                                  skip_special_tokens=True)


def run(prefix: str, settings: dict, run_stream: Callable, templates: dict, spec: dict,
        config: dict, results_dir: Path, *, limit: int | None = None, force: bool = False,
        show: Callable | None = None) -> None:
    """Run `run_stream` on each stream at each C_t, writing one shard per C_t and a manifest.

    Writes `<prefix>_ct{c}[_limit].parquet` (readout rows) and
    `<prefix>_summary_ct{c}[_limit].parquet` (one row per stream). An existing
    shard is skipped unless `force`, so a crashed sweep resumes.
    """
    layers = recorded_layers(spec, config)
    out = Path(results_dir) / spec["alias"]
    out.mkdir(parents=True, exist_ok=True)
    model, lens = registry.load(spec)
    pool = exemplars.build(model.tokenizer, config["stream"]["n_exemplars_per_category"])
    ctx = SimpleNamespace(model=model, lens=lens, spec=spec, config=config, pool=pool,
                          layers=layers, band=spec["band"])
    n_streams = limit or settings["n_streams"]
    suffix = "_limit" if limit else ""
    print(f"{prefix}: layers {layers[0]}-{layers[-1]}, band {spec['band']}; "
          f"{n_streams} streams per C_t")

    for c_t in settings["c_ts"]:
        shard = out / f"{prefix}_ct{c_t}{suffix}.parquet"
        if shard.exists() and not force:
            print(f"  C_t={c_t}: {shard.name} exists, skipping")
            continue
        rows, summaries = [], []
        for stream in streams(pool, config, c_t, n_streams):
            stream_rows, summary = run_stream(ctx, stream)
            rows += stream_rows
            summaries.append(summary)
        pd.DataFrame(rows).to_parquet(shard, index=False)
        summary = pd.DataFrame(summaries)
        summary.to_parquet(out / f"{prefix}_summary_ct{c_t}{suffix}.parquet", index=False)
        if show:
            show(c_t, summary)

    (out / f"{prefix}_manifest.json").write_text(json.dumps(
        {"spec": spec, "templates": templates, "exemplars": pool, "config": config}, indent=2))


def load(prefix: str, alias: str, results_dir: Path, *, limit: bool = False,
         kind: str = "") -> pd.DataFrame:
    """One experiment's shards concatenated; `kind="_summary"` for the per-stream rows."""
    out = Path(results_dir) / alias
    pattern = f"{prefix}{kind}_ct*{'_limit' if limit else ''}.parquet"
    paths = [p for p in sorted(out.glob(pattern)) if limit or not p.name.endswith("_limit.parquet")]
    if not paths:
        raise FileNotFoundError(f"no {pattern} in {out}")
    return pd.concat(map(pd.read_parquet, paths), ignore_index=True)


def runs(prefix: str, alias: str, results_dir: Path):
    """Yield `(label, rows, summary)` for the `--limit` run and the full run, where present."""
    for limit, label in ((True, "SMOKE (--limit)"), (False, "FULL")):
        try:
            yield (label, load(prefix, alias, results_dir, limit=limit),
                   load(prefix, alias, results_dir, limit=limit, kind="_summary"))
        except FileNotFoundError:
            continue


ITEM_KEYS = ["lens", "readout_at", "stream_id", "c_t", "word", "role", "category", "recency",
             "is_queried", "is_category_final"]


def presence(df: pd.DataFrame, k: int, *, band_only: bool = True,
             rank_col: str = "rank") -> pd.DataFrame:
    """One row per (lens, position, item): band-min rank, and whether it is <= k."""
    rows = df[df["in_band"]] if band_only else df
    grouped = rows.groupby(ITEM_KEYS, dropna=False)[rank_col].min().reset_index()
    grouped["present"] = grouped[rank_col] <= k
    return grouped


def rate(g: pd.DataFrame) -> float:
    return g["present"].mean() if len(g) else float("nan")
