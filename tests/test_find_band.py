"""Band selection: a committed numeric criterion that refuses rather than loosens."""

import pytest

from cogniload import experiment
from cogniload.find_band import choose_band

ARGS = dict(k_band=25, min_width=3)
LAYERS = experiment.layer_range(32, (0.25, 0.85))


def _medians(legible, rank=5):
    return {l: rank if l in legible else 1000 for l in LAYERS}


def test_layer_range_covers_the_fraction_and_skips_the_final_layer():
    assert LAYERS == list(range(8, 27))
    assert experiment.layer_range(32, (0.5, 1.0))[-1] == 30


def test_picks_the_longest_legible_run():
    assert choose_band(_medians([8, 9, 10, 14, 15, 16, 17, 18]), **ARGS) == (14, 19)


def test_runs_shorter_than_min_width_are_rejected():
    with pytest.raises(RuntimeError, match="no run of >= 3"):
        choose_band(_medians([12, 13]), **ARGS)


def test_illegible_lens_raises_rather_than_loosening():
    with pytest.raises(RuntimeError, match="not legible"):
        choose_band(_medians([]), **ARGS)


def test_k_band_is_inclusive():
    assert choose_band(_medians([12, 13, 14], rank=25), **ARGS) == (12, 15)
    with pytest.raises(RuntimeError):
        choose_band(_medians([12, 13, 14], rank=26), **ARGS)


def test_gapped_layer_set_is_refused_not_interpolated():
    """A gapped set would give a band over layers that were never measured."""
    strided = {l: 5 for l in range(10, 20, 2)}
    with pytest.raises(AssertionError, match="gapped"):
        choose_band(strided, **ARGS)
