"""The J-space swap intervention.

Exchanges two concepts' coordinates in the J-space, leaving everything
orthogonal to both untouched (DECISIONS.md D10):

    v_t^(l) = row t of W_U @ J_l            # lens vector for token t at layer l
    V       = [v_s, v_t]                    # [d_model, 2]
    c       = V^+ h                         # coordinates, pseudoinverse
    h'      = h + V (sigma(c) - c)          # sigma swaps the two entries

Applied at every band layer and every prompt position, then generate.

`jlens` is read-only, so the write hook lives here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch


@dataclass(frozen=True)
class SwapPlane:
    """The plane spanned by two concepts' lens vectors, at one layer.

    Carries `V` with its pseudoinverse so the hook does not recompute the same
    decomposition on every forward pass.
    """

    V: torch.Tensor       # [d_model, 2]
    V_pinv: torch.Tensor  # [2, d_model]

    @classmethod
    def build(cls, model: Any, lens: Any, token_ids: Sequence[int], layer: int) -> SwapPlane:
        V = lens_vectors(model, lens, token_ids, layer)
        return cls(V, torch.linalg.pinv(V))

    def apply(self, h: torch.Tensor) -> torch.Tensor:
        """`h' = h + V(sigma(c) - c)` with `c = V^+ h`.

        `h` is `[..., d_model]`. Exchanges the two concepts' magnitudes; the
        component of `h` orthogonal to both columns of `V` is unchanged.
        """
        V, V_pinv = self.V.to(h.device), self.V_pinv.to(h.device)
        c = V_pinv @ h.float().unsqueeze(-1)      # [..., 2, 1]
        swapped = c.flip(-2)                      # sigma: exchange the two
        return (h.float() + (V @ (swapped - c)).squeeze(-1)).to(h.dtype)


def lens_vectors(model: Any, lens: Any, token_ids: Sequence[int], layer: int) -> torch.Tensor:
    """Rows of `W_U @ J_l` for `token_ids`, stacked as `[d_model, n]`.

    Only the requested rows are formed: the full product is `[vocab, d_model]`,
    about 5 GB in fp32 at Qwen3.6-27B's vocabulary.
    """
    W_U = _unembed_weight(model)                                   # [vocab, d_model]
    J = lens.jacobians[layer].to(W_U.device)                       # [d_model, d_model]
    rows = W_U[torch.as_tensor(token_ids, device=W_U.device)]      # [n, d_model]
    return (rows.float() @ J.float()).T


@contextmanager
def swapped(
    model: Any,
    lens: Any,
    source_id: int,
    target_id: int,
    layers: Sequence[int],
    *,
    positions: slice = slice(None),
) -> Iterator[None]:
    """Clamp the source/target swap at `layers` for the duration of the block.

    `positions` defaults to every position, which is what R2's protocol wants.
    """
    planes = {
        layer: SwapPlane.build(model, lens, [source_id, target_id], layer)
        for layer in layers
    }
    handles = [
        model.layers[layer].register_forward_hook(_make_hook(plane, positions))
        for layer, plane in planes.items()
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _make_hook(plane: SwapPlane, positions: slice):
    def hook(_module, _inputs, output):
        # Some HF blocks return (hidden, present_kv, ...); patch element 0.
        is_tuple = isinstance(output, tuple)
        h = (output[0] if is_tuple else output).clone()
        h[:, positions] = plane.apply(h[:, positions])
        return (h, *output[1:]) if is_tuple else h

    return hook


def _unembed_weight(model: Any) -> torch.Tensor:
    """`W_U` as `[vocab, d_model]`.

    Uses the head `jlens` resolved through its own `Layout`, so the lens vector
    is defined against the same unembedding the lens was fitted against.
    `model.unembed()` folds in the final norm and is not usable here.
    """
    try:
        weight = model._lm_head.weight
    except AttributeError as exc:  # pragma: no cover - guards an upstream change
        raise AttributeError(
            "could not reach jlens's resolved unembedding (model._lm_head); "
            "jlens internals may have changed"
        ) from exc
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D unembedding, got {tuple(weight.shape)}")
    return weight
