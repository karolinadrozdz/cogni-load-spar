"""hay_factorial and hay_type: item validity, tail_hays strata, shared items across hay arms,
prompts, and that state_tracking's items and prompts are byte-identical to a snapshot."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
from conftest import CONFIG

from cogniload import hay_load as hl, state_tracking as st

SNAPSHOT = Path(__file__).parent / "fixtures" / "state_tracking_snapshot.json"


def snapshot(tok) -> dict:
    """A few state_tracking items and their prompts in both arms."""
    names, objects = st.pools(tok)
    out = {}
    for k, h in ((1, 1), (2, 1), (3, 6), (5, 10)):
        for item in st.make_items(k, h, 3, names, objects):
            key = f"k{k}_h{h}_{item['item_id']}"
            out[key] = {"item": item, **{arm: st.build(tok, item, arm, "Answer:",
                                                       enable_thinking=False)
                                         for arm in st.ARMS}}
    return json.loads(json.dumps(out))


def test_state_tracking_items_and_prompts_match_the_snapshot(tok):
    assert snapshot(tok) == json.loads(SNAPSHOT.read_text())


FAC, TYPE = CONFIG["hay_factorial"], CONFIG["hay_type"]


def type_items(tok, arm, n=TYPE["n_streams"]):
    names, objects = st.pools(tok)
    items = hl.make_items("hay_type", TYPE["k"], TYPE["h"], [TYPE["tail_hays"]], n, names, objects)
    return [{**it, "hay": hl.HAYS[arm]} for it in items]


def test_factorial_items_validate_and_tail_strata_are_equal(tok):
    names, objects = st.pools(tok)
    hs = hl.hs_for_pool(FAC, len(objects))
    n = FAC["n_streams"]
    for k in FAC["ks"]:
        for h in hs:
            tails = hl.tail_strata(h, FAC["tail_hays"])
            items = hl.make_items("hay_factorial", k, h, tails, n, names, objects)
            for it in items:
                st.validate(it)
                assert len(it["floor"]) >= hl.N_FLOOR
                tail = len(it["updates"]) - 1 - max(u["step"] for u in it["updates"] if u["is_needle"])
                assert it["tail_hays"] == tail and it["pre_hays"] == h - tail
            counts = Counter(it["tail_hays"] for it in items)
            assert set(counts) == set(tails) and len(set(counts.values())) == 1
            assert tails == ([0] if h == 0 else [0, 2] if h == 2 else [0, 2, 4])
            assert hl.make_items("hay_factorial", k, h, tails, 5, names, objects) == items[:5]
    # Seeded apart from state_tracking's cells.
    assert hl.make_items("hay_factorial", 3, 4, [0], 1, names, objects)[0]["updates"] != \
        st.make_items(3, 4, 1, names, objects)[0]["updates"]


def test_pool_cap_is_a_run_time_rule():
    assert hl.hs_for_pool(FAC, 23) == FAC["hs"]
    assert hl.hs_for_pool(FAC, 22) == [0, 2, 4, 8, 10]


def test_hay_type_arms_share_items_and_differ_only_in_hay_sentences(tok):
    arms = {arm: type_items(tok, arm) for arm in hl.HAYS}
    strip = lambda items: [{k: v for k, v in it.items() if k != "hay"} for it in items]
    assert strip(arms["same"]) == strip(arms["named"]) == strip(arms["reworded"])
    for i, item in enumerate(arms["same"]):
        assert item["tail_hays"] == TYPE["tail_hays"]
        built = {arm: st.build(tok, arms[arm][i], arm, "Answer:", enable_thinking=False)
                 for arm in hl.HAYS}
        # "same" is exactly state_tracking's derived prompt.
        assert built["same"] == st.build(tok, strip([item])[0], "derived", "Answer:",
                                         enable_thinking=False)
        sentences = {arm: text.split("<|im_start|>user\n")[1].split(" What is")[0].split(". ")
                     for arm, (text, _) in built.items()}
        hay = [False] * st.N_PEOPLE + [not u["is_needle"] for u in item["updates"]]
        for arm in ("named", "reworded"):
            assert len(sentences[arm]) == len(hay)
            for is_hay, a, b in zip(hay, sentences["same"], sentences[arm]):
                assert (a != b) == is_hay


@pytest.mark.parametrize("arm", list(hl.HAYS))
def test_hay_type_prompt_ends_in_thinking_off_plus_prefill(tok, arm):
    text, ends = st.build(tok, type_items(tok, arm, 1)[0], arm, "Answer:", enable_thinking=False)
    assert text.endswith("<think>\n\n</think>\n\nAnswer:")
    assert all(text[c] == "." for c in ends) and len(ends) == TYPE["k"] + TYPE["h"]
    assert st.period_tokens(tok, text, ends)


@pytest.mark.parametrize("h", [0, 12])
def test_factorial_prompt_ends_in_thinking_off_plus_prefill(tok, h):
    item = hl.make_items("hay_factorial", 3, h, [0], 1, *st.pools(tok))[0]
    text, ends = st.build(tok, item, "derived", "Answer:", enable_thinking=False)
    assert text.endswith("<think>\n\n</think>\n\nAnswer:")
    assert all(text[c] == "." for c in ends) and len(ends) == 3 + h


def test_run_item_records_tail_and_pre_hays(ctx):
    for arm in hl.HAYS:
        rows, summary = hl.run_item(ctx, type_items(ctx.model.tokenizer, arm, 1)[0], arm)
        assert summary["arm"] == arm and summary["tail_hays"] == 2 and summary["pre_hays"] == 4
        assert {r["readout_at"] for r in rows} == {"update", "answer"}


def test_reports_read_summaries_and_include_the_reference_cell(tmp_path, capsys):
    out = tmp_path / "dev"
    out.mkdir()
    errors = ["target", "poi_stale_1", "other_current"]
    for k, h in ((1, 0), (3, 0), (3, 4)):
        tails = hl.tail_strata(h, [0, 2, 4])
        pd.DataFrame([{"arm": "derived", "k": k, "h": h, "tail_hays": tails[i % len(tails)],
                       "pre_hays": h - tails[i % len(tails)], "correct": i % 2 == 0,
                       "error": errors[i % 3]} for i in range(12)]).to_parquet(
            out / f"hay_factorial_summary_k{k}_h{h}.parquet")
    for arm in hl.HAYS:
        pd.DataFrame([{"arm": arm, "k": 3, "h": 6, "correct": i % 2 == 0,
                       "error": errors[i % 3]} for i in range(6)]).to_parquet(
            out / f"hay_type_{arm}_summary_k3_h6.parquet")
    hl.report_hay_factorial({"alias": "dev"}, CONFIG, tmp_path)
    hl.report_hay_type({"alias": "dev"}, CONFIG, tmp_path)
    text = capsys.readouterr().out
    assert "tail_hays" in text and "lr_p" in text
    assert "reference: hay_factorial k=3 h=0" in text and "reworded" in text
