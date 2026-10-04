"""Band selection, Q1 scoring, and the swap algebra.

The swap tests use tiny hand-built tensors rather than a model, so the formula
`h' = h + V(sigma(c) - c)` is checked for the properties it is supposed to have
(exchange the two magnitudes, leave the orthogonal complement alone) without a
GPU. See DECISIONS.md D10.
"""

import pytest
import torch

from cogniload import scoring, swap


# --- band selection (DECISIONS.md D11) ----------------------------------------

ARGS = dict(n_layers=32, candidate_fraction=(0.25, 0.85), k_band=25, min_width=3)


def test_picks_the_longest_legible_run():
    # legible at 8-10 (3 long) and 14-18 (5 long); 5 wins
    ranks = {l: 1000 for l in range(32)}
    for l in [8, 9, 10, 14, 15, 16, 17, 18]:
        ranks[l] = 5
    assert scoring.choose_band(ranks, **ARGS) == (14, 19)


def test_respects_the_candidate_range():
    """A long legible run outside candidate_fraction must not be chosen."""
    ranks = {l: 1000 for l in range(32)}
    for l in range(0, 7):          # below 0.25 * 32 = 8
        ranks[l] = 1
    for l in [10, 11, 12]:
        ranks[l] = 1
    assert scoring.choose_band(ranks, **ARGS) == (10, 13)


def test_runs_shorter_than_min_width_are_rejected():
    ranks = {l: 1000 for l in range(32)}
    ranks[12] = ranks[13] = 1      # only 2 layers
    with pytest.raises(RuntimeError, match="no run of >= 3"):
        scoring.choose_band(ranks, **ARGS)


def test_illegible_lens_raises_rather_than_loosening():
    with pytest.raises(RuntimeError, match="not legible"):
        scoring.choose_band({l: 10_000 for l in range(32)}, **ARGS)


def test_k_band_is_a_threshold_not_a_ranking():
    ranks = {l: 1000 for l in range(32)}
    for l in [12, 13, 14]:
        ranks[l] = 25              # exactly at k_band, inclusive
    assert scoring.choose_band(ranks, **ARGS) == (12, 15)
    ranks[13] = 26
    with pytest.raises(RuntimeError):
        scoring.choose_band(ranks, **ARGS)


def test_median_rank_by_layer():
    out = scoring.median_rank_by_layer([{0: 1, 1: 9}, {0: 5, 1: 3}, {0: 3, 1: 6}])
    assert out == {0: 3.0, 1: 6.0}


# --- Q1 scoring ---------------------------------------------------------------


class _Tok:
    """Vocab of 5 tokens, space-prefixed as stream words appear."""

    _V = [" van", " Van", " bus", ",", " the"]


    def decode(self, ids):
        return "".join(self._V[i] for i in ids)


@pytest.mark.parametrize("expected,correct", [("van", True), ("bus", False)])
def test_scores_on_stripped_casefolded_comparison(expected, correct):
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    s = scoring.score_q1(_Tok(), logits, expected)
    assert s.correct is correct
    assert s.answer == " van"


def test_case_and_leading_space_are_not_errors():
    logits = torch.tensor([4.0, 5.0, 3.0, 2.0, 1.0])  # " Van" wins
    assert scoring.score_q1(_Tok(), logits, "van").correct


def test_top5_makes_a_miss_diagnosable():
    """A near-miss and a total miss must be distinguishable after the fact."""
    logits = torch.tensor([4.0, 1.0, 1.0, 5.0, 1.0])  # "," beats " van"
    s = scoring.score_q1(_Tok(), logits, "van")
    assert not s.correct
    assert [t for t, _ in s.top5][:2] == [",", " van"]
    assert "van" in s.to_row()["top5"]


# --- swap algebra (DECISIONS.md D10) ------------------------------------------


def _plane(V: torch.Tensor) -> swap.SwapPlane:
    return swap.SwapPlane(V, torch.linalg.pinv(V))


def test_swap_exchanges_the_two_coordinates():
    V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])   # e0, e1
    h = torch.tensor([3.0, 7.0, 0.0])
    out = _plane(V).apply(h)
    assert torch.allclose(out, torch.tensor([7.0, 3.0, 0.0]), atol=1e-5)


def test_swap_leaves_the_orthogonal_complement_untouched():
    V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    h = torch.tensor([3.0, 7.0, 42.0])                        # 42 is orthogonal
    assert _plane(V).apply(h)[2] == pytest.approx(42.0)


def test_swap_is_an_involution_on_the_spanned_plane():
    V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    h = torch.tensor([3.0, 7.0, 42.0])
    assert torch.allclose(_plane(V).apply(_plane(V).apply(h)), h, atol=1e-5)


def test_swap_is_a_noop_when_the_two_coordinates_are_equal():
    V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    h = torch.tensor([5.0, 5.0, 1.0])
    assert torch.allclose(_plane(V).apply(h), h, atol=1e-5)


def test_swap_handles_non_orthogonal_lens_vectors():
    """Real lens vectors are not orthogonal; the pseudoinverse is why."""
    V = torch.tensor([[1.0, 0.5], [0.0, 1.0], [0.0, 0.0]])
    h = V @ torch.tensor([2.0, 9.0])
    c = torch.linalg.pinv(V) @ _plane(V).apply(h)
    assert torch.allclose(c, torch.tensor([9.0, 2.0]), atol=1e-4)


def test_swap_broadcasts_over_batch_and_sequence():
    V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    h = torch.zeros(2, 4, 3)
    h[..., 0], h[..., 1], h[..., 2] = 3.0, 7.0, 42.0
    out = _plane(V).apply(h)
    assert out.shape == h.shape
    assert out[..., 0].eq(7.0).all() and out[..., 1].eq(3.0).all()
    assert out[..., 2].eq(42.0).all()


def test_strided_r1_sweep_is_refused_not_interpolated():
    """A gapped layer set would yield a band spanning layers R1 never measured,
    and silently: lens.source_layers is the full 0..n-2 range, so the invented
    layers are still valid lens inputs. R2 and all of Phase 1 would then run on
    unmeasured layers."""
    strided = {l: (5 if l in (10, 12, 14, 16) else 1000) for l in range(0, 32, 2)}
    with pytest.raises(ValueError, match="strided/gapped"):
        scoring.choose_band(strided, **ARGS)


def test_contiguous_sweep_still_works():
    ranks = {l: (5 if l in (10, 11, 12, 13) else 1000) for l in range(32)}
    assert scoring.choose_band(ranks, **ARGS) == (10, 14)


class _RankTok(_Tok):
    """Adds encode() so score_q1 can locate the expected word's token id."""

    def encode(self, text, add_special_tokens=False):
        try:
            return [self._V.index(text)]
        except ValueError:
            return [0, 1]  # multi-token


def test_expected_rank_survives_a_formatting_quirk():
    """A leading space or newline winning the argmax must not look the same as
    the model having no idea: `correct` goes false, `expected_rank` stays 2."""
    logits = torch.tensor([4.0, 1.0, 1.0, 5.0, 1.0])  # "," beats " van"
    s = scoring.score_q1(_RankTok(), logits, "van")
    assert not s.correct
    assert s.expected_rank == 2


def test_expected_rank_is_one_when_correct():
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    s = scoring.score_q1(_RankTok(), logits, "van")
    assert s.correct and s.expected_rank == 1


def test_expected_rank_is_minus_one_for_a_multi_token_word():
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    assert scoring.score_q1(_RankTok(), logits, "nowhere").expected_rank == -1


def test_resolve_token_id_falls_back_to_the_bare_form():
    """Qwen has no space-prefixed token for a digit, so a probe over digits
    must read the bare form -- and say which form it used."""
    from cogniload import readout

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return {" cat": [11], "4": [22]}.get(text, [0, 1])

    assert readout.resolve_token_id(Tok(), "cat") == (11, "space")
    assert readout.resolve_token_id(Tok(), "4") == (22, "bare")
    with pytest.raises(ValueError, match="not a single token"):
        readout.resolve_token_id(Tok(), "nope")
