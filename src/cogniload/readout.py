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


JLENS_MAX_SEQ_LEN = 512


def check_positions(model: Any, prompt: str, positions: Sequence[int],
                    max_seq_len: int | None) -> None:
    """Raise if jlens would cut the prompt, or a position is past the prompt it reads."""
    length = {"max_length": max_seq_len} if max_seq_len else {}
    n = model.encode(prompt, **length).shape[-1]
    # A prompt cut at the limit would make "the last token" silently mean the 512th.
    if n >= (max_seq_len or JLENS_MAX_SEQ_LEN):
        full = len(model.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if full > n:
            raise ValueError(f"prompt is longer than the {n}-token lens limit; pass max_seq_len")
    # An index past the sequence is a device-side assert on GPU.
    outside = sorted(p for p in positions if not -n <= p < n)
    if outside:
        raise ValueError(f"positions {outside} are outside the {n}-token encoded prompt "
                         f"(jlens truncates at max_seq_len, 512 by default)")


def read(model: Any, lens: Any, prompt: str, words: Sequence[str], token_ids: Sequence[int],
         *, layers: Sequence[int], positions: Sequence[int], use_jacobian: bool = True,
         max_seq_len: int | None = None) -> tuple[list[dict], torch.Tensor]:
    """One forward pass: a row per (word, layer, position), and the model's own logits there.

    `use_jacobian=False` is the logit-lens control: the same readout with `J_l`
    replaced by the identity. `max_seq_len` overrides jlens's truncation of the
    prompt (512 tokens by default).
    """
    # Imported here so the module loads, and tests run, without jlens.
    from jlens.vis import _meaningful_token_mask

    check_positions(model, prompt, positions, max_seq_len)
    length = {"max_seq_len": max_seq_len} if max_seq_len else {}
    lens_logits, model_logits, input_ids = lens.apply(
        model, prompt, layers=list(layers), positions=list(positions), use_jacobian=use_jacobian,
        **length)
    # Absolute indices, so `token_pos` means the same thing across calls.
    n = input_ids.shape[-1]
    positions = [p if p >= 0 else n + p for p in positions]
    vocab = lens_logits[min(lens_logits)].shape[-1]
    if max(token_ids) >= vocab:
        raise ValueError(f"token ids {sorted(i for i in token_ids if i >= vocab)} are "
                         f"outside the {vocab}-row unembedding")
    # lens.apply returns CPU logits; ranking on the GPU is the same integer count, much faster.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = torch.as_tensor(list(token_ids), dtype=torch.long, device=device)
    wordlike = _meaningful_token_mask(model.tokenizer, vocab, device)
    if not wordlike[targets].all():
        raise ValueError("some target tokens are not word-like; rank_wordlike would be meaningless")

    rows = []
    for layer in sorted(lens_logits):
        logits = lens_logits[layer].to(device)
        rank = rank_of(logits, targets).cpu()
        rank_w = rank_of(logits, targets, among=wordlike).cpu()
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
    decoded = {}
    for layer, logits in sorted(lens_logits.items()):
        ids = logits.masked_fill(~wordlike, float("-inf")).topk(n).indices.tolist()
        decoded[layer] = [model.tokenizer.decode([i]) for i in ids]
    return decoded, model_logits


def band_top_tokens(model: Any, lens: Any, prompt: str, *, layers: Sequence[int],
                    position: int, n: int, use_jacobian: bool = True) -> list[tuple[str, int]]:
    """The `n` tokens with the best word-like rank at `position` over `layers` (band-min),
    decoded, with that rank."""
    lens_logits, _, wordlike = _lens_at(model, lens, prompt, layers, position, use_jacobian)
    best = None
    for logits in lens_logits.values():
        order = logits.masked_fill(~wordlike, float("-inf")).argsort(descending=True)
        # Inverting the sort order gives each token's 0-indexed rank.
        rank = order.argsort() + 1
        best = rank if best is None else torch.minimum(best, rank)
    ids = best.topk(n, largest=False).indices.tolist()
    return [(model.tokenizer.decode([i]), int(best[i])) for i in ids]
