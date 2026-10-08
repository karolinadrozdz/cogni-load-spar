"""J-space readout: the 1-indexed rank of each word's token under the lens.

`rank` is over the full vocabulary. `rank_wordlike` is over word-like tokens
only, which is what Neuronpedia shows; on Qwen punctuation dominates the raw
top-K, so a word can be clearly present and still miss `rank <= 1`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def token_id(tokenizer: Any, word: str, prefix: str = " ") -> int:
    """Id of `word` as it appears after ", ". Qwen digits have no spaced form; use prefix=""."""
    ids = tokenizer.encode(prefix + word, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{prefix + word!r} is {len(ids)} tokens, not 1")
    return ids[0]


def rank_of(logits: torch.Tensor, target_ids: torch.Tensor, *,
            among: torch.Tensor | None = None) -> torch.Tensor:
    """Rank of each target at each position, `[n_positions, n_targets]`.

    Counts strictly greater logits, so ties do not inflate the rank. `among`
    masks the competitors. Looped over targets to hold memory at one
    `[n_positions, vocab]` comparison.
    """
    chosen = logits[:, target_ids]
    out = torch.empty_like(chosen, dtype=torch.long)
    for t in range(target_ids.numel()):
        beats = logits > chosen[:, t : t + 1]
        out[:, t] = (beats & among if among is not None else beats).sum(dim=-1)
    return out + 1


def read(model: Any, lens: Any, prompt: str, words: Sequence[str], token_ids: Sequence[int],
         *, layers: Sequence[int], positions: Sequence[int], use_jacobian: bool = True,
         ) -> tuple[list[dict], torch.Tensor]:
    """One forward pass: a row per (word, layer, position), and the model's own logits there.

    `use_jacobian=False` is the logit-lens control: the same readout with `J_l`
    replaced by the identity.
    """
    # Imported here so the module loads, and tests run, without jlens.
    from jlens.vis import _meaningful_token_mask

    lens_logits, model_logits, input_ids = lens.apply(
        model, prompt, layers=list(layers), positions=list(positions), use_jacobian=use_jacobian)
    # Absolute indices, so `token_pos` means the same thing across calls.
    n = input_ids.shape[-1]
    positions = [p if p >= 0 else n + p for p in positions]
    first = lens_logits[min(lens_logits)]
    targets = torch.as_tensor(list(token_ids), dtype=torch.long, device=first.device)
    wordlike = _meaningful_token_mask(model.tokenizer, first.shape[-1], first.device)
    if not wordlike[targets].all():
        raise ValueError("some target tokens are not word-like; rank_wordlike would be meaningless")

    rows = []
    for layer in sorted(lens_logits):
        logits = lens_logits[layer]
        rank, rank_w = rank_of(logits, targets), rank_of(logits, targets, among=wordlike)
        rows += [{"word": w, "layer": layer, "token_pos": p,
                  "rank": int(rank[pi, i]), "rank_wordlike": int(rank_w[pi, i])}
                 for i, w in enumerate(words) for pi, p in enumerate(positions)]
    return rows, model_logits.cpu()


def _lens_at(model: Any, lens: Any, prompt: str, layers: Sequence[int], position: int,
             use_jacobian: bool) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Lens logits per layer and the model's own logits at one position, and the word-like
    mask `rank_wordlike` counts among."""
    from jlens.vis import _meaningful_token_mask

    lens_logits, model_logits, _ = lens.apply(
        model, prompt, layers=list(layers), positions=[position], use_jacobian=use_jacobian)
    first = lens_logits[min(lens_logits)]
    wordlike = _meaningful_token_mask(model.tokenizer, first.shape[-1], first.device)
    return {l: x[0] for l, x in lens_logits.items()}, model_logits[0], wordlike


def top_tokens(model: Any, lens: Any, prompt: str, *, layers: Sequence[int], position: int,
               n: int, use_jacobian: bool = True) -> tuple[dict[int, list[str]], torch.Tensor]:
    """The top `n` word-like tokens, decoded, at `position` for each layer, and the model's
    own logits there."""
    lens_logits, model_logits, wordlike = _lens_at(model, lens, prompt, layers, position,
                                                   use_jacobian)
    top = {layer: logits.masked_fill(~wordlike, float("-inf")).topk(n).indices.tolist()
           for layer, logits in sorted(lens_logits.items())}
    return ({layer: [model.tokenizer.decode([i]) for i in ids] for layer, ids in top.items()},
            model_logits)


def band_top_tokens(model: Any, lens: Any, prompt: str, *, layers: Sequence[int],
                    position: int, n: int, use_jacobian: bool = True) -> list[tuple[str, int]]:
    """The `n` tokens with the best word-like rank at `position` over `layers` (band-min),
    decoded, with that rank."""
    lens_logits, _, wordlike = _lens_at(model, lens, prompt, layers, position, use_jacobian)
    best = None
    for logits in lens_logits.values():
        rank = logits.masked_fill(~wordlike, float("-inf")).argsort(descending=True).argsort() + 1
        best = rank if best is None else torch.minimum(best, rank)
    ids = best.topk(n, largest=False).indices.tolist()
    return [(model.tokenizer.decode([i]), int(best[i])) for i in ids]
