"""self_check: does the model know whether its state_tracking answer is right?

state_tracking's derived items; an answer as the assistant turn; then "Is your answer
correct? Answer yes or no." with thinking off and `Answer:` prefilled. One forward pass
reads the model's yes-vs-no logits and both lenses at the last token. Arms: `no_reasoning`
shows the thinking-off run's answer, `answer_only` the thinking-on run's answer without its
trace, `with_trace` the same answer with the trace kept in the assistant turn.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cogniload import experiment, prompts, readout, state_tracking as st
from cogniload import state_tracking_thinking as stt
from cogniload.experiment import VIEWS, presence
from cogniload.state_tracking import FMT

PREFIX = "self_check"
ARMS = ("no_reasoning", "answer_only", "with_trace")
QUESTION = "Is your answer correct? Answer yes or no."
HISTORY = "Answer: {word}"
# The template drops reasoning from turns before the last user message, so the trace is
# inserted by hand, in the form the template gives a kept trace.
TRACE = "<think>\n{reasoning}\n</think>\n\n"
ASSISTANT = "<|im_start|>assistant\n"
# The spaced forms follow `Answer:`; a bare form would follow it with no space.
FORMS = {"yes": [" yes", " Yes", "yes"], "no": [" no", " No", "no"]}
OFF, ON = "state_tracking_derived", stt.PREFIX
ITEM = ["k", "h", "item_id"]
# The judge-pass fields a `no_reasoning` item copies from `answer_only` when the prompts match.
JUDGED = ["n_tokens", "logit_diff", "p_yes", "judge_argmax", "judge_top5", "form_logits"]
JSPACE_KEYS = ["lens", "arm", "k", "h", "item_id", "base", "role", "is_shown", "correct"]
OTHER_IN_PROMPT = ["chain", "other_current", "other_stale"]


def shown_word(emitted: str, objects: list[str], dropped: list[str]) -> tuple[str, bool]:
    """The answer shown to the judge, and whether it is an object: the one object in
    `objects` that the emitted token is (` Bell` -> `bell`) or starts (` V` -> `vase`), else
    the stripped token. As in `correct_prefix`, only an object in `dropped` (its capital is not
    one token) can be started, so ` K` is `kite` even with `key` in the prompt."""
    token = emitted.strip().casefold()
    starts = [o for o in objects
              if token and (o == token or (o in dropped and o.startswith(token)))]
    return (starts[0], True) if len(starts) == 1 else (emitted.strip(), False)


def answer_forms(tokenizer) -> tuple[dict[str, list[str]], list[str]]:
    """The yes and no forms that are one token with an id of their own, and the rest."""
    kept, ids = {}, set()
    for base, forms in FORMS.items():
        for form in forms:
            enc = tokenizer.encode(form, add_special_tokens=False)
            if len(enc) == 1 and enc[0] not in ids:
                kept.setdefault(base, []).append(form)
                ids.add(enc[0])
    assert set(kept) == set(FORMS), f"no single-token form for {set(FORMS) - set(kept)}"
    return kept, [f for forms in FORMS.values() for f in forms if f not in sum(kept.values(), [])]


def user_turn(tokenizer, item: dict, off_prompt: str) -> str:
    """state_tracking's derived user message, checked against the thinking-off run's prompt."""
    prompt, _ = st.build(tokenizer, item, "derived", "", enable_thinking=False)
    user = prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    assert prompts.render_chat(tokenizer, user, enable_thinking=False) == prompt
    assert off_prompt.startswith(prompt), f"item {item['item_id']} is not the thinking-off prompt"
    return user


def build(tokenizer, user: str, word: str, reasoning: str | None = None) -> str:
    """[user: the item, assistant: `Answer: {word}`, user: QUESTION], prefilled, thinking
    off; with `reasoning`, the trace goes into the assistant turn."""
    text = prompts.render_chat(tokenizer, [
        {"role": "user", "content": user},
        {"role": "assistant", "content": HISTORY.format(word=word)},
        {"role": "user", "content": QUESTION},
    ], enable_thinking=False, prefill=prompts.PREFILL)
    if reasoning is None:
        return text
    assert text.count(ASSISTANT) == 2
    at = text.index(ASSISTANT) + len(ASSISTANT)
    # Stripped, as the template trims a kept trace.
    text = text[:at] + TRACE.format(reasoning=reasoning.strip()) + text[at:]
    assert text.endswith(prompts._THINK_OFF + prompts.PREFILL)
    return text


def words_and_meta(item: dict, caps: dict[str, str], forms: dict[str, list[str]],
                   word: str) -> dict[str, dict]:
    """state_tracking_thinking's words (both case forms) and the yes and no forms, each
    form labelled by its token text; `is_shown` marks the shown answer's rows."""
    meta = stt.words_and_meta(item, caps)
    floor = meta[item["floor"][0]]
    meta |= {f: {**floor, "role": base, "form": f, "base": base}
             for base, fs in forms.items() for f in fs}
    return {w: {**m, "is_shown": m["base"] == word} for w, m in meta.items()}


def judge(tokenizer, logits, forms: dict[str, list[str]]) -> dict:
    """Yes vs no from the model's own logits at the judge position."""
    ids = {base: [readout.token_id(tokenizer, f, prefix="") for f in fs]
           for base, fs in forms.items()}
    diff = float(logits[ids["yes"]].logsumexp(0) - logits[ids["no"]].logsumexp(0))
    q = experiment.score_q1(tokenizer, logits, "yes")
    return {"logit_diff": diff, "p_yes": 1 / (1 + math.exp(-diff)),
            "judge_argmax": q["answer"], "judge_top5": q["top5"],
            "form_logits": json.dumps({f: float(logits[i]) for base, fs in forms.items()
                                       for f, i in zip(fs, ids[base])})}


def run_item(ctx, arm: str, item: dict, off: pd.Series, on: pd.Series,
             copy: tuple | None) -> tuple[list[dict], dict]:
    """The judge-position rows and the item's summary row; a `no_reasoning` item whose
    prompt is `answer_only`'s takes `copy` (that arm's summary and rows) instead of a pass."""
    tok = ctx.model.tokenizer
    user = user_turn(tok, item, off["prompt"])
    objects = [o for chain in item["chains"].values() for o in chain]
    off_word, on_word = (shown_word(s["answer"], objects, ctx.caps_dropped)[0]
                         for s in (off, on))
    source = off if arm == "no_reasoning" else on
    word, is_object = shown_word(source["answer"], objects, ctx.caps_dropped)
    text = build(tok, user, word, on["reasoning"] if arm == "with_trace" else None)
    target = item["chains"][item["poi"]][-1]
    correct = word == target
    copied = copy is not None and copy[0].prompt == text
    if copied:
        judged = {c: getattr(copy[0], c) for c in JUDGED}
        rows = copy[1].assign(arm=arm).to_dict("records")
    else:
        meta = words_and_meta(item, ctx.caps, ctx.forms, word)
        ids = [readout.token_id(tok, w, prefix="" if m["role"] in FORMS else " ")
               for w, m in meta.items()]
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        rows, logits = experiment.read_both(ctx, text, meta, {n - 1: "judge"}, token_ids=ids,
                                            max_seq_len=n)
        for r in rows:
            r |= {"arm": arm, "correct": correct}
        judged = {"n_tokens": n, **judge(tok, logits[-1], ctx.forms)}
    summary = {
        "arm": arm, **{c: item[c] for c in ("k", "h", "item_id", "last_is_needle", "poi")},
        "names": json.dumps(item["names"]), "chains": json.dumps(item["chains"]),
        "floor": json.dumps(item["floor"]), "target": target, "emitted": source["answer"],
        "word": word, "answer_is_object": is_object, "correct": correct,
        "source_correct": bool(source["correct"]), "closed": bool(on["closed"]),
        # Same word both ways means byte-identical no_reasoning and answer_only prompts.
        "shared_with_answer_only": off_word == on_word, "copied_from_answer_only": copied,
        "prompt": text, **judged,
    }
    return rows, summary


def source_summary(prefix: str, alias: str, results_dir: Path) -> tuple[pd.DataFrame, str]:
    """A source run's summaries indexed by item: the full run if there is one, else the
    `--limit` run (its items are the full run's first)."""
    for limit in (False, True):
        try:
            s = experiment.load(prefix, alias, results_dir, limit=limit, kind="_summary", cell="k")
        except FileNotFoundError:
            continue
        return s.set_index(ITEM), "limit" if limit else "full"
    raise FileNotFoundError(f"no {prefix} summaries for {alias}; run that stage first")


def source_row(s: pd.DataFrame, item: dict, prefix: str) -> pd.Series:
    """The source run's summary row for `item`, which must have the same chains."""
    key = tuple(item[c] for c in ITEM)
    if key not in s.index:
        raise KeyError(f"{prefix} has no item (k, h, item_id) {key}; narrow --cell or --limit")
    row = s.loc[key]
    assert json.loads(row["chains"]) == item["chains"], f"{prefix} item {key} has other chains"
    return row


def answer_only_copies(out: Path, tag: str, suffix: str) -> dict[int, tuple]:
    """item_id -> (`answer_only` summary row, its rows) for one cell, if that shard exists."""
    shard = out / f"{PREFIX}_answer_only_{tag}{suffix}.parquet"
    summary = out / f"{PREFIX}_answer_only_summary_{tag}{suffix}.parquet"
    if not (shard.exists() and summary.exists()):
        return {}
    rows = pd.read_parquet(shard)
    return {s.item_id: (s, rows[rows.item_id == s.item_id])
            for s in pd.read_parquet(summary).itertuples()}


def _show(key: tuple[int, int], s: pd.DataFrame) -> None:
    if key == (1, 1):
        print(f"\n  --- k=1 h=1 first item prompt ---\n{s.iloc[0]['prompt']}\n  ---")
    by = s.groupby("correct").logit_diff.mean().round(2).to_dict()
    print(f"  k={key[0]} h={key[1]}: shown correct {s.correct.mean():.0%}, mean logit_diff by "
          f"correct {by}, judged yes {(s.p_yes > .5).mean():.0%}, copied "
          f"{int(s.copied_from_answer_only.sum())}; argmax "
          f"{s.judge_argmax.value_counts().head(3).to_dict()}")


def run(spec: dict, config: dict, results_dir: Path, *, arm: str, limit: int | None = None,
        force: bool = False, cell: tuple[int, int] | None = None) -> None:
    """`cell` = (k, h) runs that cell only, under its usual shard name."""
    settings = config["state_tracking"]
    grid = st.grid(settings["ks"])
    if cell is not None and cell not in grid:
        raise ValueError(f"cell {cell} is not in the grid {grid}")
    (off, off_run), (on, on_run) = (source_summary(p, spec["alias"], results_dir)
                                    for p in (OFF, ON))
    out, suffix = Path(results_dir) / spec["alias"], "_limit" if limit else ""
    templates = {**st.base_templates(settings["prefill"]), "self_check_question": QUESTION,
                 "history": HISTORY, "trace": TRACE, "judge_prefill": prompts.PREFILL,
                 "enable_thinking": False, "sources": {OFF: off_run, ON: on_run}}

    def cells(ctx, n):
        tok = ctx.model.tokenizer
        names, objects = st.pools(tok)
        ctx.caps, ctx.caps_dropped = stt.capitals(tok, objects)
        ctx.forms, forms_dropped = answer_forms(tok)
        templates.update(names=names, objects=objects, capitals=ctx.caps,
                         capitals_dropped=ctx.caps_dropped, answer_forms=ctx.forms,
                         answer_forms_dropped=forms_dropped)
        for k, h in grid:
            if cell not in (None, (k, h)):
                continue
            tag = st.cell_tag(k, h)
            copies = answer_only_copies(out, tag, suffix) if arm == "no_reasoning" else {}
            yield (k, h), tag, [(it, source_row(off, it, OFF), source_row(on, it, ON),
                                 copies.get(it["item_id"]))
                                for it in st.make_items(k, h, n, names, objects)]

    experiment.run(f"{PREFIX}_{arm}", settings, lambda ctx, x: run_item(ctx, arm, *x),
                   templates, spec, config, results_dir, limit=limit, force=force, show=_show,
                   cells=cells)


# --- analysis -----------------------------------------------------------------


def auroc(score, positive) -> float:
    """P(a positive outscores a negative), ties counted half: the rank-sum statistic with
    tied scores given their average rank."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if not n_pos or not n_neg:
        return float("nan")
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def behaviour(s: pd.DataFrame) -> pd.DataFrame:
    """Per arm x correct of the shown answer: logit_diff, the share judged yes, the argmax."""
    g = s.groupby(["arm", "correct"])
    out = g.agg(n=("logit_diff", "size"), mean_diff=("logit_diff", "mean"),
                median_diff=("logit_diff", "median"),
                judged_yes=("p_yes", lambda p: (p > .5).mean()))
    out["argmax"] = g["judge_argmax"].agg(lambda a: str(a.value_counts().head(5).to_dict()))
    return out.reset_index()


def discrimination(s: pd.DataFrame) -> pd.DataFrame:
    """AUROC of logit_diff for correct vs wrong shown answers, overall, per k and per h."""
    groups = [("all", "", s), *(("k", k, g) for k, g in s.groupby("k")),
              *(("h", h, g) for h, g in s.groupby("h"))]
    return pd.DataFrame([{"by": by, "value": v, "n": len(g), "n_correct": int(g.correct.sum()),
                          "auroc": auroc(g.logit_diff, g.correct)} for by, v, g in groups])


def calibration(s: pd.DataFrame) -> pd.DataFrame:
    """p_yes in five equal bins against the shown answer's accuracy."""
    bins = pd.cut(s.p_yes, np.linspace(0, 1, 6), include_lowest=True).rename("p_yes_bin")
    return s.groupby(bins, observed=False).agg(
        n=("correct", "size"), mean_p_yes=("p_yes", "mean"),
        accuracy=("correct", "mean")).reset_index().astype({"p_yes_bin": str})


def pair(s: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Arm `a` (`_a` columns) beside arm `b` (`_b`) on the same items."""
    return s[s.arm == a].merge(s[s.arm == b], on=ITEM, suffixes=("_a", "_b"),
                               validate="one_to_one")


def paired_stats(items: str, j: pd.DataFrame) -> dict:
    """logit_diff of b minus a: mean, median, the pairs where b is higher or lower, and the
    two-sided sign-test p on those (ties dropped)."""
    d = j.logit_diff_b - j.logit_diff_a
    higher, lower = int((d > 0).sum()), int((d < 0).sum())
    return {"items": items, "n": len(d), "mean_diff": d.mean(), "median_diff": d.median(),
            "b_higher": higher, "b_lower": lower,
            "share_b_higher": higher / len(d) if len(d) else float("nan"),
            "sign_p": stt.mcnemar(higher, lower)}


def reasoning_vs_none(s: pd.DataFrame) -> pd.DataFrame:
    """b = answer_only vs a = no_reasoning, on the items whose thinking-off answer is wrong;
    where the thinking-on answer is right, b_higher means the right answer is judged higher."""
    j = pair(s, "no_reasoning", "answer_only")
    wrong = j[~j.correct_a]
    return pd.DataFrame([paired_stats("off wrong", wrong),
                         paired_stats("off wrong, on right", wrong[wrong.correct_b]),
                         paired_stats("off wrong, on wrong", wrong[~wrong.correct_b])])


def trace_vs_none(s: pd.DataFrame) -> pd.DataFrame:
    """b = with_trace vs a = answer_only (the same answer), on every item, by closed and by
    correct."""
    j = pair(s, "answer_only", "with_trace")
    return pd.DataFrame([paired_stats("all", j),
                         *(paired_stats(f"closed={c}", g) for c, g in j.groupby("closed_a")),
                         *(paired_stats(f"correct={c}", g) for c, g in j.groupby("correct_a"))])


def jspace(p: pd.DataFrame) -> pd.DataFrame:
    """Presence at the judge position per arm x correct, J-lens with the logit lens (`_ll`)
    beside: the yes and no forms, the shown answer, the target when it is not shown, the other
    in-prompt objects, the floor."""
    groups = {"yes": p.role == "yes", "no": p.role == "no", "shown": p.is_shown,
              "target_not_shown": (p.role == "target") & ~p.is_shown,
              "other_in_prompt": p.role.isin(OTHER_IN_PROMPT) & ~p.is_shown,
              "floor": p.role == "floor"}
    rates = pd.DataFrame({c: p[m].groupby(["arm", "correct", "lens"])["present"].mean()
                          for c, m in groups.items()}).unstack("lens")
    n = p[(p.lens == "jlens") & (p.role == "yes")].groupby(["arm", "correct"]).size()
    out = stt.jlens_beside_logit(rates).join(n.rename("n"))
    return out[["n", *[x for c in groups for x in (c, f"{c}_ll")]]].reset_index()


def load_runs(alias: str, results_dir: Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Per run label, every arm's band rows and summaries."""
    parts = {}
    for arm in ARMS:
        for label, df, s in experiment.runs(f"{PREFIX}_{arm}", alias, results_dir, cell="k"):
            parts.setdefault(label, []).append((df[df.in_band], s))
    return {label: (pd.concat([d for d, _ in x], ignore_index=True),
                    pd.concat([s for _, s in x], ignore_index=True))
            for label, x in parts.items()}


def report(spec: dict, config: dict, results_dir: Path) -> None:
    k, rank_col = config["readout"]["primary_k"], VIEWS[0][0]
    p_fmt = {**FMT, "formatters": {"sign_p": "{:.2g}".format}}
    for label, (rows, s) in load_runs(spec["alias"], results_dir).items():
        head, arms = f"{PREFIX} {label}", set(s.arm)
        print(f"\n{head}: judge behaviour per arm x correct of the shown answer "
              f"(logit_diff = logsumexp yes forms - logsumexp no forms)")
        print(behaviour(s).to_string(**FMT))
        print(f"  shown answer correct differs from the source run's strict correct on "
              f"{int((s.correct != s.source_correct).sum())} of {len(s)} rows; shown answer "
              f"not an object on {int((~s.answer_is_object).sum())}")
        if "no_reasoning" in arms:
            nr = s[s.arm == "no_reasoning"]
            print(f"\n{head} no_reasoning discrimination: AUROC of logit_diff, correct vs wrong")
            print(discrimination(nr).to_string(**FMT))
            print(f"\n{head} no_reasoning calibration: p_yes bins against accuracy")
            print(calibration(nr).to_string(**FMT))
        if {"no_reasoning", "answer_only"} <= arms:
            print(f"\n{head} paired, b = answer_only vs a = no_reasoning, items whose "
                  f"thinking-off answer is wrong")
            print(reasoning_vs_none(s).to_string(**p_fmt))
        if {"answer_only", "with_trace"} <= arms:
            print(f"\n{head} paired, b = with_trace vs a = answer_only (same answer), all items")
            print(trace_vs_none(s).to_string(**p_fmt))
        p = presence(rows, k, rank_col=rank_col, keys=JSPACE_KEYS)
        print(f"\n{head} judge-position contents per arm x correct, J-lens with logit lens "
              f"(_ll) [{rank_col}, in-band, k={k}, forms pooled on base]")
        print(jspace(p).to_string(**FMT))
