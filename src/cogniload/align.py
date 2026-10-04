"""Locating a readout token inside a rendered prompt.

Always the delimiter after a word, never the word itself: a word is trivially
top-ranked at its own position, and the paper's capacity protocol reads at the
comma for the same reason (DECISIONS.md D15).

The delimiter must also be *inside* the list. At the final `.` the stream is
followed by instruction text, so the model is disposed to continue the
instruction rather than the list, and no stream word appears at any layer.
"""

from __future__ import annotations

from typing import Any


def delimiter_after(tokenizer: Any, text: str, word: str) -> int:
    """Absolute token index of the delimiter (`,` or `.`) following `word`.

    From character offsets, then asserted: a token-count offset would be wrong,
    since the chat template and special tokens shift it. `word` must occur
    exactly once, which holds for stream words.
    """
    hits = [i for i in range(len(text)) if text.startswith(word, i)
            and text[i + len(word):i + len(word) + 1] in (",", ".")]
    if len(hits) != 1:
        raise ValueError(
            f"{word!r} is followed by a delimiter {len(hits)} times in the "
            f"prompt; cannot locate the position unambiguously"
        )
    delimiter_char = hits[0] + len(word)

    encoding = tokenizer(text, return_offsets_mapping=True, truncation=True,
                         max_length=512, add_special_tokens=False)
    for index, (start, end) in enumerate(encoding["offset_mapping"]):
        if start <= delimiter_char < end:
            token = tokenizer.decode([encoding["input_ids"][index]])
            if "," not in token and "." not in token:
                raise AssertionError(
                    f"expected the delimiter after {word!r}, got {token!r} "
                    f"at token {index}"
                )
            return index
    raise AssertionError(
        f"character {delimiter_char} is outside the tokenised prompt — the "
        f"stream may have been truncated at max_length"
    )
