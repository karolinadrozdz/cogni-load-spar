"""Prompt text shared across experiments, and the one way to render a chat prompt.

Templates are plain strings; each experiment writes the ones it used into its
manifest verbatim, so a result always records the text that produced it.
"""

from __future__ import annotations

from typing import Any

# What the Qwen3 chat template appends for each thinking state.
_THINK_OFF = "<think>\n\n</think>\n\n"
_THINK_ON = "<think>\n"

KEEP_TRACK = (
    "Track these categories: {tracked}. "
    "Here is a sequence of words: {stream}. "
    "Remember the most recent word from each tracked category."
)
# "Answer in one word": without it Qwen answers in markdown (argmax ` **`),
# which measures formatting rather than memory.
RECENT = "What was the most recent {category}? Answer in one word."
PREFILL = "Answer:"


def render_chat(tokenizer: Any, messages: list[dict] | str, *, enable_thinking: bool,
                prefill: str = "") -> str:
    """Render a chat prompt (a bare string is one user turn) and assert its thinking state.

    `prefill` goes after the think block, so the readout position is the last
    token of the returned string.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    _assert_thinking(text, enable_thinking)
    return text + prefill


def _assert_thinking(text: str, enable_thinking: bool) -> None:
    # Thinking left on would make the model write its state out, which is a
    # different condition, and nothing else would fail.
    expected, other = (_THINK_ON, _THINK_OFF) if enable_thinking else (_THINK_OFF, _THINK_ON)
    if text.endswith(expected):
        return
    hint = ("template ignored enable_thinking=False and left reasoning ON"
            if text.endswith(other) and not enable_thinking
            else "template does not match the expected Qwen3 thinking contract")
    raise AssertionError(f"{hint}; generation prompt ended with {text[-40:]!r}")


def stream_turn(words: list[str], tracked: tuple[str, ...], question: str) -> str:
    """The keep-track instruction and stream, then `question`, as one user turn."""
    return KEEP_TRACK.format(tracked=", ".join(tracked), stream=", ".join(words)) + " " + question


def build_recent(tokenizer: Any, words: list[str], tracked: tuple[str, ...], category: str,
                 *, enable_thinking: bool) -> str:
    """Stream + "most recent {category}?" in one user turn, prefilled."""
    return render_chat(tokenizer, stream_turn(words, tracked, RECENT.format(category=category)),
                       enable_thinking=enable_thinking, prefill=PREFILL)
