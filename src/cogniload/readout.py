"""J-space readout: rank of a token under the lens, per layer and position.

Presence is `band-min rank <= k`. Ranks are 1-indexed, so `rank == 1` is top-1.

Two rank columns per item: `rank` over the full vocabulary (primary), and
`rank_wordlike` over word-like tokens only. On Qwen punctuation dominates the
raw top-K, so a concept can be clearly present and still miss `rank <= 1` on
full vocab; `rank_wordlike` is also the view Neuronpedia shows (DECISIONS.md
D12).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class Readout:
    """Per-(item, layer, position) lens measurements for one prompt."""

    token_ids: list[int]
    layers: list[int]
    #: Absolute token indices, never relative: this is part of the row key, so
    #: a mix of -1 and absolute indices would corrupt it.
    positions: list[int]
    #: [n_items, n_layers, n_positions], 1-indexed, full vocabulary.
    rank: torch.Tensor
    #: Same shape, ranked among word-like tokens only.
    rank_wordlike: torch.Tensor
    #: Same shape. Raw lens logit, before ranking or any nonlinearity.
    coefficient: torch.Tensor
    #: The model's own next-token logits at `positions`: [n_positions, vocab].
    #: Scoring Q1 from these keeps the answer consistent with the ranks, since
    #: both come from one forward pass.
    model_logits: torch.Tensor

    def band_min_rank(self, *, wordlike: bool = False) -> torch.Tensor:
        """Best (lowest) rank over layers: `[n_items, n_positions]`."""
        source = self.rank_wordlike if wordlike else self.rank
        return source.min(dim=1).values

    def present_at(self, k: int, *, wordlike: bool = False) -> torch.Tensor:
        """`band-min rank <= k`: `[n_items, n_positions]` bool."""
        return self.band_min_rank(wordlike=wordlike) <= k

    def to_rows(self, words: Sequence[str] | None = None, **extra: Any) -> list[dict]:
        """One row per (item, layer, position).

        `words` labels each item, parallel to `token_ids`, and is the join key
        to stream metadata — `**extra` is constant per Readout so it cannot
        carry one.
        """
        labels = list(words) if words is not None else [None] * len(self.token_ids)
        if len(labels) != len(self.token_ids):
            raise ValueError(
                f"{len(labels)} words for {len(self.token_ids)} token_ids"
            )
        return [
            {
                "token_id": token_id,
                "word": labels[i],
                "layer": layer,
                "token_pos": position,
                "rank": int(self.rank[i, li, pi]),
                "rank_wordlike": int(self.rank_wordlike[i, li, pi]),
                "coefficient": float(self.coefficient[i, li, pi]),
                **extra,
            }
            for i, token_id in enumerate(self.token_ids)
            for li, layer in enumerate(self.layers)
            for pi, position in enumerate(self.positions)
        ]


def read(
    model: Any,
    lens: Any,
    prompt: str,
    token_ids: Sequence[int],
    *,
    layers: Sequence[int],
    positions: Sequence[int] | None = None,
    use_jacobian: bool = True,
) -> Readout:
    """Rank `token_ids` under the lens at `layers` and `positions`.

    One forward pass. `positions=None` reads every position;
    `use_jacobian=False` gives the logit-lens control, the same readout with
    `J_l` replaced by the identity.
    """
    # Local import so this module stays importable, and testable, without
    # jlens and its GPU dependency stack.
    from jlens.vis import _meaningful_token_mask

    lens_logits, model_logits, input_ids = lens.apply(
        model, prompt, layers=list(layers), positions=positions,
        use_jacobian=use_jacobian,
    )
    layer_list = sorted(lens_logits)
    first = lens_logits[layer_list[0]]

    # lens.apply accepts negative indices; record absolute ones so the position
    # column means one thing across calls.
    seq_len = input_ids.shape[-1]
    pos_list = (
        [p if p >= 0 else seq_len + p for p in positions]
        if positions is not None
        else list(range(first.shape[0]))
    )

    targets = torch.as_tensor(list(token_ids), dtype=torch.long, device=first.device)
    wordlike = _meaningful_token_mask(model.tokenizer, first.shape[-1], first.device)
    if not wordlike[targets].all():
        raise ValueError(
            "some target tokens are not word-like under the library's mask; "
            "their rank_wordlike would be meaningless"
        )

    ranks, ranks_wordlike, coefficients = [], [], []
    for layer in layer_list:
        logits = lens_logits[layer]                      # [n_positions, vocab]
        # Per-layer logits need not share a device under device_map="auto".
        # `.to()` is free when they already do.
        tgt, mask = targets.to(logits.device), wordlike.to(logits.device)
        coefficients.append(logits[:, tgt].T.float())
        ranks.append(_rank_of(logits, tgt).T)
        ranks_wordlike.append(_rank_of(logits, tgt, among=mask).T)

    # Each entry is [n_items, n_positions]; layers become the middle axis.
    return Readout(
        token_ids=list(token_ids),
        layers=layer_list,
        positions=pos_list,
        rank=torch.stack(ranks, dim=1).cpu(),
        rank_wordlike=torch.stack(ranks_wordlike, dim=1).cpu(),
        coefficient=torch.stack(coefficients, dim=1).cpu(),
        model_logits=model_logits.cpu(),
    )


def _rank_of(
    logits: torch.Tensor, target_ids: torch.Tensor, *, among: torch.Tensor | None = None
) -> torch.Tensor:
    """1-indexed rank of each target at each position: `[n_positions, n_targets]`.

    Counts strictly-greater logits, so ties do not inflate the rank. `among`
    restricts the competitors to a vocabulary mask, giving the word-like rank
    without copying `logits`. Looped over targets to hold peak memory at one
    `[n_positions, vocab]` comparison.
    """
    chosen = logits[:, target_ids]                       # [n_positions, n_targets]
    out = torch.empty_like(chosen, dtype=torch.long)
    for t in range(target_ids.numel()):
        beats = logits > chosen[:, t : t + 1]
        out[:, t] = (beats & among if among is not None else beats).sum(dim=-1)
    return out + 1


def single_token_id(tokenizer: Any, word: str) -> int:
    """Token id for a stream word, in the space-prefixed form it appears in."""
    ids = tokenizer.encode(f" {word}", add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{word!r} is {len(ids)} tokens, not 1")
    return ids[0]


def resolve_token_id(tokenizer: Any, item: str) -> tuple[int, str]:
    """Token id for `item` in whichever form is a single token.

    Returns `(id, form)`. Stream words occur after ", " and so have a
    space-prefixed token; Qwen has no space-prefixed form for digits, which are
    always standalone. Tries the space-prefixed form first so the common case
    matches `single_token_id`, and reports the form rather than silently
    reading a token that does not occur in the text.
    """
    for form, text in (("space", f" {item}"), ("bare", item)):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], form
    raise ValueError(f"{item!r} is not a single token in either form")
