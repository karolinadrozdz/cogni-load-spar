"""state_tracking: item validity, the recency guard, prompts, and run_item + tables on a fake model."""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import CONFIG

from cogniload import cli, experiment, registry, state_tracking as st

K = CONFIG["readout"]["primary_k"]
N = CONFIG["state_tracking"]["n_streams"]


def test_every_item_is_valid_and_each_cell_is_half_hay_last(tok):
    names, objects = st.pools(tok)
    cells = [(k, h) for k in CONFIG["state_tracking"]["ks"] for h in st.hays(k)]
    assert len(cells) == 14
    for k, h in cells:
        items = st.make_items(k, h, N, names, objects)
        for it in items:
            st.validate(it)
            assert len(it["floor"]) >= 3
        assert sum(it["last_is_needle"] for it in items) == N // 2
        assert st.make_items(k, h, 5, names, objects) == items[:5]


@pytest.mark.parametrize("arm", st.ARMS)
def test_prompt_ends_in_thinking_off_plus_prefill(tok, arm):
    item = st.make_items(2, 1, 1, *st.pools(tok))[0]
    text, ends = st.build(tok, item, arm, "Answer:", enable_thinking=False)
    assert text.endswith("<think>\n\n</think>\n\nAnswer:")
    assert all(text[c] == "." for c in ends) and len(ends) == 3
    assert (" (" in text) == (arm == "copyable")


def test_run_item_and_tables_on_a_fake_model(ctx):
    names, objects = st.pools(ctx.model.tokenizer)
    rows, summ = [], []
    for arm in st.ARMS:
        for k, h in ((1, 1), (3, 6), (4, 1)):
            for item in st.make_items(k, h, 2, names, objects):
                r, s = st.run_item(ctx, item, arm)
                rows += r
                summ.append(s)
    rows, summ = pd.DataFrame(rows), pd.DataFrame(summ)
    # Force some correct items so T3 has data.
    rows.loc[rows.item_id == 0, "correct"] = True
    assert len(st.t1(summ)) == 6
    ans = rows[rows.readout_at == "answer"]
    p = experiment.presence(ans, K, rank_col="rank_wordlike", keys=st.KEYS)
    t2 = st.t2(p, "jlens")
    assert len(t2) == 6 and t2["chain_4"].notna().sum() == 2
    assert len(st.t3(ans, K, "rank_wordlike", "jlens")) == 2
    assert len(st.t4(p, "jlens")) == 3
    control = st.copyable_beside_derived(summ, p)
    assert len(control) == 3
    assert {"top1_derived", "top1_copyable", "copyable_target"} <= set(control)
    groups = st.groups(p)
    assert set(groups.correct) == {True, False} and {"emitted_ll", "chain_4_ll"} <= set(groups)
    best = st.band_best(p, "rank_wordlike", K)
    assert len(best) == 6 and {"median_best_jlens", "any_present_logit"} <= set(best)


def test_hay_split_counts_hays_around_the_last_needle():
    kinds = [False, True, False, False, True, False]
    updates = [{"is_needle": n} for n in kinds]
    assert st.hay_split(updates) == (3, 1)


def test_recency_regression_recovers_the_sign_of_tail_hays():
    rng = np.random.default_rng(0)
    s = pd.DataFrame({"k": rng.integers(1, 6, 2000), "pre_hays": rng.integers(0, 6, 2000),
                      "tail_hays": rng.integers(0, 4, 2000)})
    s["correct"] = rng.random(2000) < 1 / (1 + np.exp(-(1.0 - 0.8 * s.tail_hays)))
    table = st.recency_regression(s).set_index("term")
    assert table.loc["tail_hays", "coef"] < -0.5 and table.loc["tail_hays", "lr_p"] < 1e-6
    assert abs(table.loc["k", "coef"]) < 0.2 and table.loc["k", "lr_p"] > 1e-3


def test_top_tokens_prints_and_writes_no_shards(ctx, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(registry, "load", lambda spec: (ctx.model, ctx.lens))
    monkeypatch.setattr(experiment, "recorded_layers", lambda spec, config: list(range(30, 60)))
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "band.json").write_text('{"band": [38, 54]}')
    cli.main(["state_tracking", "--arm", "derived", "--model", "dev", "--top-tokens", "2",
              "--config", str(registry.REGISTRY.parent / "experiments.yaml"),
              "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert out.count("model top 5") == 2 and "jlens L37" in out and "logit L52" in out
    assert out.count("last-update period, band-min top 25:") == 4
    assert "logit last-update period, top 25 per category over 2 items" in out
    assert not list(tmp_path.rglob("*.parquet"))


def test_last_update_table_keeps_the_just_written_object_apart(ctx):
    names, objects = st.pools(ctx.model.tokenizer)
    rows, summ = [], []
    for item in st.make_items(3, 3, 4, names, objects):
        r, s = st.run_item(ctx, item, "derived")
        rows += r
        summ.append(s)
    rows, summ = pd.DataFrame(rows), pd.DataFrame(summ)
    words = st.last_update_words(summ)
    for it in summ.itertuples():
        new = json.loads(it.updates)[-1]["new"]
        word = words[(words.item_id == it.item_id) & (words.word == new)]
        assert word.column.tolist() == ["just_new"]
    last = rows[rows.in_band & (rows["update"] == 5)].assign(rank_wordlike=1)
    p = experiment.presence(last, K, rank_col="rank_wordlike", keys=st.KEYS)
    table = st.last_update(p, summ)
    # When the last update is a needle, the PoI's current object is the one just written.
    needle = table[table.last_is_needle]
    assert needle.poi_current.isna().all() and (needle.just_new == 1).all()
    assert (table[~table.last_is_needle].poi_current == 1).all() and table.n.sum() == 4


def test_token_category():
    instruction = st.instruction_words("Answer:")
    assert {"holds", "the", "person", "swaps", "what", "answer", "word"} <= instruction
    sort = lambda w: st.token_category(w, ["key", "lamp"], ["Ann"], instruction)
    assert [sort(w) for w in (" key", " Ann", " holding", " Answer", ".", " ", "<|im_end|>",
                              " banana")] == ["list_objects", "names", "instruction", "instruction",
                                              "formatting", "formatting", "formatting", "other"]
