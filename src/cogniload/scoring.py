"""Scoring answers, and choosing the workspace band from R1 readouts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import groupby
from statistics import median
from typing import Any, Sequence

import torch


# --- Q1 scoring ---------------------------------------------------------------


@dataclass(frozen=True)
class Q1Score:
    expected: str
    answer: str
    correct: bool
    #: Rank of the expected word in the model's next-token distribution,
    #: 1-indexed. `correct` is only rank 1, which a formatting quirk — a
    #: leading space or newline winning the argmax — can break while the model
    #: has the answer perfectly well. This degrades gracefully where the binary
    #: does not. -1 if the word is not a single token.
    expected_rank: int
    #: Decoded top-5 with logits. A near miss and a total miss both show up as
    #: `correct=False`, so keep enough to tell them apart without a rerun.
    top5: list[tuple[str, float]]

    def to_row(self) -> dict:
        return {
            "expected": self.expected,
            "answer": self.answer,
            "correct": self.correct,
            "expected_rank": self.expected_rank,
            "top5": " | ".join(f"{t}:{v:.2f}" for t, v in self.top5),
        }


def score_q1(tokenizer: Any, logits: torch.Tensor, expected: str) -> Q1Score:
    """Score the predicted token against the target word."""
    top = logits.topk(5)
    decoded = [tokenizer.decode([i]) for i in top.indices.tolist()]
    try:
        expected_id = tokenizer.encode(f" {expected}", add_special_tokens=False)
        rank = (
            int((logits > logits[expected_id[0]]).sum()) + 1
            if len(expected_id) == 1 else -1
        )
    except Exception:  # pragma: no cover - tokenizer stand-ins in tests
        rank = -1
    return Q1Score(
        expected=expected,
        answer=decoded[0],
        correct=_matches(decoded[0], expected),
        expected_rank=rank,
        top5=list(zip(decoded, top.values.tolist())),
    )


# --- R1 band selection --------------------------------------------------------


def choose_band(
    median_rank_by_layer: dict[int, float],
    *,
    n_layers: int,
    candidate_fraction: tuple[float, float],
    k_band: int,
    min_width: int,
) -> tuple[int, int]:
    """Longest contiguous run of legible layers inside the candidate range.

    Legible means median rank over R1 items `<= k_band`. Returns a half-open
    `[lo, hi)`. Uses only R1 readouts, by design: fitting the band to what
    makes R2 pass would make that gate circular (DECISIONS.md D11).
    """
    lo_frac, hi_frac = candidate_fraction
    lo, hi = int(lo_frac * n_layers), int(hi_frac * n_layers)
    candidates = [l for l in sorted(median_rank_by_layer) if lo <= l < hi]

    # `groupby` below runs over list order, so gapped input would yield a band
    # covering layers R1 never measured — silently, since every layer in
    # 0..n-2 is a valid lens input. Refuse rather than interpolate; measuring
    # every layer costs no extra forward passes anyway.
    gaps = [b - a for a, b in zip(candidates, candidates[1:]) if b - a != 1]
    if gaps:
        raise ValueError(
            f"R1 measured a strided/gapped layer set in {lo}-{hi - 1} "
            f"(gaps of {sorted(set(gaps))}); choose_band would return a band "
            f"covering unmeasured layers. Measure every layer in the candidate "
            f"range — lens.apply returns them all from one forward pass."
        )

    def is_legible(layer: int) -> bool:
        return median_rank_by_layer[layer] <= k_band

    runs = [list(g) for ok, g in groupby(candidates, key=is_legible) if ok]
    runs = [r for r in runs if len(r) >= min_width]
    if not runs:
        raise RuntimeError(
            f"no run of >= {min_width} layers with median rank <= {k_band} in "
            f"layers {lo}-{hi - 1}. The lens is not legible on this model under "
            f"the committed criterion; loosen k_band deliberately and record it, "
            f"or stop (spec Phase 0)."
        )
    best = max(runs, key=len)
    return best[0], best[-1] + 1


def median_rank_by_layer(ranks: Sequence[dict[int, int]]) -> dict[int, float]:
    """Median over R1 items of each layer's rank. Input: one dict per item."""
    return {
        layer: median(r[layer] for r in ranks) for layer in sorted(ranks[0])
    }


# --- Q2 parsing ---------------------------------------------------------------


def _matches(a: str, b: str) -> bool:
    """A decoded token carries a leading space and may differ in case from the
    stream word; neither is an error."""
    return a.strip().casefold() == b.strip().casefold()


def parse_q2(text: str, tracked: Sequence[str], vocabulary: Sequence[str]) -> dict[str, str]:
    """Pull one word per tracked category out of Q2's free text.

    Anchored on known vocabulary rather than on layout, since the model may
    answer "animal: cat, fruit: pear", a bulleted form of the same, or a bare
    "cat, pear". First pass keys off category names; the fallback takes words
    belonging to this stream and aligns them positionally to `tracked`.

    Returns only what was found — a missing category is a parse failure, not a
    wrong answer.
    """
    found: dict[str, str] = {}
    lowered = text.casefold()

    for category in tracked:
        pattern = re.escape(category.casefold()) + r"\s*[:\-–]\s*([A-Za-z]+)"
        match = re.search(pattern, lowered)
        if match:
            found[category] = match.group(1)
    if found:
        return found

    vocab = {w.casefold(): w for w in vocabulary}
    ordered = [vocab[w] for w in re.findall(r"[A-Za-z]+", lowered) if w in vocab]
    return dict(zip(tracked, ordered))


def score_q2(
    text: str,
    tracked: Sequence[str],
    targets: dict[str, str],
    vocabulary: Sequence[str],
) -> list[dict]:
    """One row per tracked category.

    `parsed` separates "the model got it wrong" from "we could not read the
    answer"; without it the second silently scores as the first.
    """
    found = parse_q2(text, tracked, vocabulary)
    return [
        {
            "category": category,
            "parsed": category in found,
            "parsed_word": found.get(category),
            "target_word": targets[category],
            "correct": category in found and _matches(found[category], targets[category]),
        }
        for category in tracked
    ]
