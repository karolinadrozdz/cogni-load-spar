"""derived_state: prompt, copyable annotation and counts, and run_stream + table on a fake model."""

import pandas as pd
import pytest
from conftest import CONFIG, POOL

from cogniload import derived_state, experiment
from cogniload.stimuli import Item

TRACKED = ("animal", "fruit")


@pytest.mark.parametrize("arm", derived_state.ARMS)
def test_prompt_ends_in_thinking_off_plus_prefill(tok, arm):
    items = [Item(1, "cat", "animal", "target"), Item(2, "red", "color", "untracked")]
    pieces, _ = derived_state.label(items, TRACKED, ["animal", "color", "fruit"], arm)
    out = derived_state.build_count(tok, pieces, TRACKED, "animal", enable_thinking=False)
    assert out.endswith("<think>\n\n</think>\n\nAnswer: ")
    assert "How many animal words have appeared so far?" in out


def test_copyable_annotation_and_counts():
    items = [Item(1, "cat", "animal", "replaced"), Item(2, "red", "color", "untracked"),
             Item(3, "dog", "animal", "target"), Item(4, "hand", "body part", "untracked"),
             Item(5, "pear", "fruit", "target"), Item(6, "blue", "color", "untracked")]
    cats = ["animal", "body part", "color", "fruit"]
    pieces, states = derived_state.label(items, TRACKED, cats, "copyable")
    assert pieces == ["cat (animal 1)", "red", "dog (animal 2)", "hand", "pear (fruit 1)", "blue"]
    assert derived_state.label(items, TRACKED, cats, "derived")[0] == [i.word for i in items]
    assert states[2] == {"animal": 2, "body part": 0, "color": 1, "fruit": 0}
    assert states[-1] == {"animal": 2, "body part": 1, "color": 2, "fruit": 1}


@pytest.mark.parametrize("arm", derived_state.ARMS)
def test_runner_and_table_on_a_fake_model(ctx, arm):
    rows, summ = [], []
    for c_t in (2, 4):
        for stream in experiment.streams(POOL, CONFIG, c_t, 2):
            r, s = derived_state.run_stream(ctx, stream, arm)
            rows += r
            summ.append(s)
            assert len({x["token_pos"] for x in r}) == s["p"]  # p - 1 commas + the answer
    t = derived_state.table(pd.DataFrame(rows), pd.DataFrame(summ), CONFIG["readout"]["primary_k"])
    assert list(t.c_t) == [2, 4] and (t.n == 2).all()
    for col in ("ans_queried", "in_floor", "ans_floor4", "in_tracked_just", "in_tracked_other"):
        assert t[col].notna().all()
    assert t["n_pos_in"].sum() == sum(x["p"] - 1 for x in summ)
