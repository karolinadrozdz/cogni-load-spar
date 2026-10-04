"""Frozen, content-addressed prompts.

Each prompt carries a sha over its text, and the run manifest records the sha
of the whole set, so editing a template invalidates cached results rather than
contaminating them. Change a prompt by bumping its `version`.

Build prompts only through `build_q1` / `build_q2` / `build_r2`: what context
each query sees is part of the measurement (DECISIONS.md D9).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

# What the Qwen3 chat template appends for each thinking state.
_THINK_OFF = "<think>\n\n</think>\n\n"
_THINK_ON = "<think>\n"


@dataclass(frozen=True)
class Prompt:
    id: str
    version: int
    template: str

    def render(self, **kwargs: Any) -> str:
        return self.template.format(**kwargs)

    @property
    def sha(self) -> str:
        payload = f"{self.id}|v{self.version}|{self.template}"
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


# --- Phase 1: keep-track task -------------------------------------------------

KEEP_TRACK = Prompt(
    id="keep_track.stream",
    version=1,
    template=(
        "Track these categories: {tracked}. "
        "Here is a sequence of words: {stream}. "
        "Remember the most recent word from each tracked category."
    ),
)

# One tracked category. Prefilled, so the next token is the answer.
# "Answer in one word" is the paper's own R2 phrasing. Without it Qwen answers
# in markdown -- the argmax at the prefill is ` **`, with the correct word at
# rank 2 -- which measures formatting rather than memory.
Q1_RECENT = Prompt(
    id="keep_track.q1_recent",
    version=2,
    template="What was the most recent {category}? Answer in one word.",
)
Q1_PREFILL = "Answer:"

# All tracked categories, parsed per category for the self-report measure.
Q2_LIST_ALL = Prompt(
    id="keep_track.q2_list_all",
    version=2,
    template=(
        "List the most recent word for each tracked category, in order. "
        "Give one line per category as 'category: word', with no explanation."
    ),
)

# --- Phase 0 R2: report gate ---------------------------------------------------

R2_THINK_OF = Prompt(
    id="phase0.r2_think_of",
    version=1,
    template="Think of a {category}. Answer in one word.",
)
R2_PREFILL = Q1_PREFILL

#: Every prompt that can reach a model; `prompt_set_sha` covers all of them.
ALL: dict[str, Prompt] = {
    p.id: p
    for p in (KEEP_TRACK, Q1_RECENT, Q2_LIST_ALL, R2_THINK_OF)
}


def prompt_set_sha() -> str:
    joined = "|".join(f"{pid}:{ALL[pid].sha}" for pid in sorted(ALL))
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


# --- Rendering ----------------------------------------------------------------


def render_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
    prefill: str = "",
) -> str:
    """Render a chat prompt and assert the thinking state it was asked for.

    `prefill` is appended to the generation prompt, after the think block, so
    the readout position is the final token of the returned string. This is the
    only prefill convention used here; passing a prefill as a trailing
    assistant message renders a different string (DECISIONS.md D13).
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    _assert_thinking(text, enable_thinking)
    return text + prefill


def _assert_thinking(text: str, enable_thinking: bool) -> None:
    expected = _THINK_OFF if not enable_thinking else _THINK_ON
    if text.endswith(expected):
        return
    # Fail loudly rather than infer a state we cannot confirm: with thinking
    # left on, the internal condition externalizes its state and silently
    # becomes the externalized one.
    other = _THINK_ON if not enable_thinking else _THINK_OFF
    hint = (
        "template ignored enable_thinking=False and left reasoning ON — the "
        "internal condition would collapse into the externalized one"
        if text.endswith(other) and not enable_thinking
        else "template does not match the expected Qwen3 thinking contract"
    )
    raise AssertionError(f"{hint}; generation prompt ended with {text[-40:]!r}")


# --- Turn structure -----------------------------------------------------------


def _stream_turn(stream_words: list[str], tracked: tuple[str, ...], question: str) -> str:
    return (
        KEEP_TRACK.render(tracked=", ".join(tracked), stream=", ".join(stream_words))
        + " "
        + question
    )


def build_q1(
    tokenizer: Any,
    stream_words: list[str],
    tracked: tuple[str, ...],
    category: str,
    *,
    enable_thinking: bool,
) -> str:
    """Stream + "most recent {category}?" in one user turn, prefilled."""
    user = _stream_turn(stream_words, tracked, Q1_RECENT.render(category=category))
    return render_chat(
        tokenizer,
        [{"role": "user", "content": user}],
        enable_thinking=enable_thinking,
        prefill=Q1_PREFILL,
    )


def build_q2(
    tokenizer: Any,
    stream_words: list[str],
    tracked: tuple[str, ...],
    *,
    enable_thinking: bool,
) -> str:
    """Stream + "list all" in one user turn.

    A fresh render, not a continuation of Q1: following Q1's answer would leave
    one tracked category already resolved in context, so its self-report would
    not be comparable to the others.
    """
    user = _stream_turn(stream_words, tracked, Q2_LIST_ALL.render())
    return render_chat(
        tokenizer, [{"role": "user", "content": user}], enable_thinking=enable_thinking
    )


def build_r2(tokenizer: Any, category: str, *, enable_thinking: bool) -> str:
    """Phase 0 R2: "Think of a {category}", prefilled so the next token is the
    reported word. Same prefill and thinking regime as Q1, so the gate measures
    the lens under the regime Phase 1 uses."""
    return render_chat(
        tokenizer,
        [{"role": "user", "content": R2_THINK_OF.render(category=category)}],
        enable_thinking=enable_thinking,
        prefill=R2_PREFILL,
    )
