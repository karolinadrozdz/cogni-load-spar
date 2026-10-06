"""forced_demand and retro_cue: prompt builders, and run_stream + table on a fake model."""

import pandas as pd
from conftest import CONFIG, POOL

from cogniload import experiment, forced_demand, retro_cue

K = CONFIG["readout"]["primary_k"]
WORDS, TRACKED = ["cat", "pear", "red", "dog", "plum"], ("animal", "fruit")


def test_sorted_list_ends_in_thinking_off_and_prefill(tok):
    out = forced_demand.build_sorted_list(tok, WORDS, TRACKED, enable_thinking=False)
    assert out.endswith("<think>\n\n</think>\n\nAnswer:")
    assert "in alphabetical order" in out


def test_second_cue_has_the_first_answer_as_history(tok):
    out = retro_cue.build_second_cue(tok, WORDS, TRACKED, "animal", "fruit", "dog",
                                     enable_thinking=False)
    assert out.endswith("<think>\n\n</think>\n\nAnswer:")
    assert out.count("<|im_start|>user") == 2
    assert "<|im_start|>assistant\nAnswer: dog<|im_end|>" in out


def test_runners_and_tables_on_a_fake_model(ctx):
    rows_a, summ_a, rows_b, summ_b = [], [], [], []
    for c_t in (2, 4):
        for stream in experiment.streams(POOL, CONFIG, c_t, 2):
            r, s = forced_demand.run_stream(ctx, stream)
            rows_a += r
            summ_a.append(s)
            r, s = retro_cue.run_stream(ctx, stream)
            rows_b += r
            summ_b.append(s)
    t = forced_demand.table(pd.DataFrame(rows_a), pd.DataFrame(summ_a), K)
    assert list(t.c_t) == [2, 4] and t["t3"].isna().iloc[0] and not t["t3"].isna().iloc[1]
    t = retro_cue.table(pd.DataFrame(rows_b), pd.DataFrame(summ_b), K)
    assert len(t) == 4 and set(t.position) == {1, 2}
