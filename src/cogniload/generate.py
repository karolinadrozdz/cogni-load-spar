"""Generation. Only Q2 needs it.

Q1 and R2 read the model's own logits out of the readout pass instead, so their
answers are consistent with the measured ranks by construction.
"""

from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def free_text(model: Any, prompt: str, *, max_new_tokens: int = 64) -> str:
    """Greedy continuation of `prompt`, with the prompt stripped off.

    Ids come from `model.encode` so generation sees the same sequence the
    readout pass did. Goes through the wrapped HF model because
    `model.forward` is the bare decoder, with no lm_head.
    """
    try:
        hf = model._hf_model
    except AttributeError as exc:  # pragma: no cover - guards an upstream change
        raise AttributeError(
            "could not reach the wrapped HF model (model._hf_model); jlens "
            "internals may have changed"
        ) from exc

    input_ids = model.encode(prompt)
    output = hf.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=model.tokenizer.eos_token_id,
    )
    return model.tokenizer.decode(
        output[0, input_ids.shape[-1]:], skip_special_tokens=True
    )
