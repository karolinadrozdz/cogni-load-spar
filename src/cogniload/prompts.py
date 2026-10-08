"""The one way to render a chat prompt, and the shared answer prefill.

Each experiment keeps its own templates as plain strings and writes the ones
it used into its manifest verbatim, so a result always records its text.
"""

from __future__ import annotations

from typing import Any

# What the Qwen3 chat template appends for each thinking state.
_THINK_OFF = "<think>\n\n</think>\n\n"
_THINK_ON = "<think>\n"

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
