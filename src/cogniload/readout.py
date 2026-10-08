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
    """Id of `word` as it follows a space. A form with no leading space: prefix=""."""
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


def top_tokens(model: Any, lens: Any, prompt: str, *, layers: Sequence[int], position: int,
               k: int, use_jacobian: bool = True) -> list[dict]:
    """The `k` highest-ranked word-like tokens at one position, per layer.

    The open-vocabulary view: what the lens shows, with no word list chosen in
    advance.
    """
    from jlens.vis import _meaningful_token_mask

    lens_logits, _, _ = lens.apply(model, prompt, layers=list(layers), positions=[position],
                                   use_jacobian=use_jacobian)
    rows = []
    for layer in sorted(lens_logits):
        logits = lens_logits[layer][0]
        wordlike = _meaningful_token_mask(model.tokenizer, logits.shape[-1], logits.device)
        top = logits.masked_fill(~wordlike, float("-inf")).topk(k).indices.tolist()
        rows += [{"layer": layer, "rank": r + 1, "token": model.tokenizer.decode([i])}
                 for r, i in enumerate(top)]
    return rows
