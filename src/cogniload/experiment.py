"""Blocks every experiment shares: layers, the two-lens readout and generation.

Task-independent on purpose: stimuli, the run loop and the tables belong to
each experiment's own module.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from cogniload import readout

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


def read_both(ctx: SimpleNamespace, text: str, meta: dict[str, dict], positions: dict[int, str],
              token_ids: list[int] | None = None) -> tuple[list[dict], torch.Tensor]:
    """J-lens and logit-lens readout of every word in `meta` at `positions` (token -> name).

    Returns the rows, each carrying its word's metadata, and the model's own
    logits at `positions` from the J-lens pass.
    """
    words = list(meta)
    ids = token_ids or [readout.token_id(ctx.model.tokenizer, w) for w in words]
    rows, model_logits = [], None
    for lens_name, jacobian in (("jlens", True), ("logit", False)):
        result, logits = readout.read(ctx.model, ctx.lens, text, words, ids, layers=ctx.layers,
                                      positions=list(positions), use_jacobian=jacobian)
        model_logits = logits if model_logits is None else model_logits
        rows += [{**r, **meta[r["word"]], "lens": lens_name, "readout_at": positions[r["token_pos"]],
                  "in_band": ctx.band[0] <= r["layer"] < ctx.band[1]} for r in result]
    return rows, model_logits


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
