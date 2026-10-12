"""state_tracking_thinking: state_tracking's derived items with reasoning on.

The model reasons (sampled, up to a token budget), then `</think>\\n\\nAnswer:`
is appended and one forward pass reads the full text. Read at the answer (all
recorded layers), and in the band only at the last token of each reasoning
sentence (`sentence`) and at the token before each object word the reasoning
writes (`prewrite`). Every object is read in both case forms (` bag`, ` Bag`).
"""

from __future__ import annotations

import json
import math
import re
import resource
import time
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from cogniload import experiment, readout, state_tracking as st
from cogniload.experiment import VIEWS, presence
from cogniload.state_tracking import FMT

PREFIX = "state_tracking_thinking"
END_THINK = "</think>"
# state_tracking's keys without `update`, with `base` for `word` so the two case forms pool.
ANSWER_KEYS = ["lens", "readout_at", "arm", "k", "h", "item_id", "base", "role", "chain_j",
               "hay_matched", "correct", "emitted", "closed"]
PREWRITE_KEYS = ["lens", "k", "h", "item_id", "token_pos", "mention", "writing", "control",
                 "quoted", "base", "role"]
SENTENCE_KEYS = ["lens", "k", "h", "item_id", "token_pos", "sentence", "n_sentences", "base",
                 "role"]
COLUMNS = ["lens", "readout_at", "in_band", "k", "h", "item_id", "token_pos", "word", "base",
           "role", "chain_j", "hay_matched", "correct", "emitted", "arm", "rank_wordlike",
           "mention", "writing", "control", "quoted", "sentence", "n_sentences", "closed"]
# A mention with one of these just before it copies an update sentence rather than a derived state.
QUOTES = ("holding the", "swaps it for the", "holds the")
QUOTE_WINDOW = 40
WRITING = {"target": "poi_final", "chain": "poi_chain", "other_current": "other",
           "other_stale": "other", "floor": "floor"}
# Floor objects the trace writes as ordinary words ("the chain", "stick to", "key step").
PROSE = ("chain", "key", "stick")
# A holder statement: `Name [now|still] holds/has`, `Name is [now|still] holding`, `Name:`,
# `Name ->`, `Name →` or `Name =`, then [the] object, any part in ** markdown.
CLAIM = (r"(?<![A-Za-z])\**({names})\**(?:\s+(?:(?:now|still)\s+)?(?:holds|has)\s+"
         r"|\s+is\s+(?:(?:now|still)\s+)?holding\s+|\s*(?::|->|→|=)\**\s*)"
         r"\**(?:the\s+)?\**({objects})\**(?![A-Za-z])")
CLAIMS = ["true-final", "true-ever", "never"]


def capitals(tokenizer: Any, objects: list[str]) -> tuple[dict[str, str], list[str]]:
    """Object -> capitalised form, for the forms that are one token after a space; and the
    objects whose capitalised form is not."""
    caps = {o: o.capitalize() for o in objects if st.is_one_token(tokenizer, o.capitalize())}
    return caps, [o for o in objects if o not in caps]


def words_and_meta(item: dict, caps: dict[str, str]) -> dict[str, dict]:
    """state_tracking's words, plus each object's capitalised form with the same role row.
    `base` is the lowercase word, so presence can pool the two forms."""
    meta = {w: {**m, "form": "lower", "base": w}
            for w, m in st.words_and_meta(item, "derived").items()}
    objects = [w for w, m in meta.items() if not m["role"].startswith("name")]
    return meta | {caps[o]: {**meta[o], "form": "cap"} for o in objects if o in caps}


@torch.no_grad()
def think(model: Any, prompt: str, settings: dict, seed: int
          ) -> tuple[str, bool, list[int], list[int]]:
    """Sampled reasoning up to `max_thinking_tokens`, stopped at `</think>`: the decoded text
    without `</think>`, whether the model closed it, the prompt ids and the sampled ids."""
    tok = model.tokenizer
    end = readout.token_id(tok, END_THINK, prefix="")
    ids = model.encode(prompt)
    torch.manual_seed(seed)
    out = model._hf_model.generate(
        ids, attention_mask=torch.ones_like(ids), do_sample=True,
        temperature=settings["temperature"], top_p=settings["top_p"], top_k=settings["top_k"],
        max_new_tokens=settings["max_thinking_tokens"], eos_token_id=end,
        pad_token_id=tok.eos_token_id)
    new = out[0, ids.shape[-1]:].tolist()
    closed = end in new
    new = new[:new.index(end)] if closed else new
    return tok.decode(new), closed, ids[0].tolist(), new


def full_text(prompt: str, reasoning: str, closed: bool, prefill: str) -> str:
    """The prompt, the reasoning, the close and the answer prefill: the text the lens reads."""
    return prompt + reasoning + ("" if closed else "\n") + END_THINK + "\n\n" + prefill


def sentence_ends(enc: dict, text: str, span: tuple[int, int]) -> list[int]:
    """Token index of the last token of each reasoning sentence. A token ends a sentence
    if it holds a newline, or a `.?!` followed by whitespace or the end of
    the reasoning; of consecutive such tokens only the first counts."""
    lo, hi = span
    ends, prev = [], False
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if not lo <= a < hi:
            prev = False
            continue
        end = "\n" in text[a:b] or any(
            text[c] in ".?!" and (c + 1 >= hi or text[c + 1].isspace()) for c in range(a, b))
        if end and not prev:
            ends.append(i)
        prev = end
    return ends


def even(xs: list, cap: int) -> list[tuple[int, Any]]:
    """(original index, x) for at most `cap` evenly spaced elements, the first kept, so early
    reasoning is not dropped from long traces only."""
    step = max(1, math.ceil(len(xs) / cap))
    return list(enumerate(xs))[::step][:cap]


def controls(writes: list[tuple]) -> list[str | None]:
    """For each mention, the most recent different object written before it, over every
    mention, so sampling the mentions does not change it."""
    out, before = [], []
    for _, o, *_ in writes:
        out.append(next((x for x in reversed(before) if x != o), None))
        before.append(o)
    return out


def prewrite(enc: dict, text: str, span: tuple[int, int], objects: list[str]
             ) -> list[tuple[int, str, int, int, bool]]:
    """(token before, object, occurrence index, char offset in `text`, quoted) for each object
    word in the reasoning, in either case form, located by character offset."""
    lo, hi = span
    forms = {f: o for o in objects for f in (o, o.capitalize())}
    pattern = re.compile(rf"(?<![A-Za-z])({'|'.join(map(re.escape, forms))})(?![A-Za-z])")
    out, seen = [], {}
    for m in pattern.finditer(text, lo, hi):
        index = experiment.token_at(enc["offset_mapping"], m.start())
        o = forms[m.group()]
        quoted = any(q in text[max(lo, m.start() - QUOTE_WINDOW):m.start()] for q in QUOTES)
        out.append((index - 1, o, seen.get(o, 0), m.start(), quoted))
        seen[o] = seen.get(o, 0) + 1
    return out


def seed_of(item: dict) -> int:
    """The item's sampling seed, fixed by its cell and id."""
    return zlib.crc32(f"{PREFIX}-{item['k']}-{item['h']}-{item['item_id']}".encode())


def reasoning_positions(sentences: list[tuple[int, int]], n_sentences: int, writes: list[tuple],
                        kept: list[tuple[int, tuple]]) -> dict[int, list[tuple[str, dict]]]:
    """Token -> each (readout_at, extra row columns) to read there: the kept sentence ends and
    the kept pre-write positions."""
    at = {}
    for s, t in sentences:
        at.setdefault(t, []).append(("sentence", {"sentence": s, "n_sentences": n_sentences}))
    control = controls(writes)
    for m, (t, o, occurrence, char, quoted) in kept:
        at.setdefault(t, []).append(("prewrite", {"mention": m, "writing": o,
                                                  "occurrence": occurrence, "char": char,
                                                  "quoted": quoted, "control": control[m]}))
    return at


def read_reasoning(ctx, text: str, meta: dict[str, dict], at: dict[int, list[tuple[str, dict]]],
                   n_tokens: int) -> list[dict]:
    """Band-layer rows at the positions in `at`, one copy per readout there."""
    # Band layers only, to hold storage near state_tracking's.
    band = SimpleNamespace(**{**vars(ctx), "layers": [l for l in ctx.layers
                                                      if ctx.band[0] <= l < ctx.band[1]]})
    read, _ = experiment.read_both(band, text, meta, dict.fromkeys(at, "reasoning"),
                                   max_seq_len=n_tokens)
    return [{**r, "readout_at": name, **extra}
            for r in read for name, extra in at[r["token_pos"]]]


def run_item(ctx, item: dict, caps: dict[str, str]) -> tuple[list[dict], dict]:
    """Reason, then read at the answer and the reasoning positions: the rows and the item's
    summary row."""
    tok, settings = ctx.model.tokenizer, ctx.config[PREFIX]
    prompt, _ = st.build(tok, item, "derived", "", enable_thinking=True)
    seed = seed_of(item)
    t0 = time.perf_counter()
    reasoning, closed, prompt_ids, new_ids = think(ctx.model, prompt, settings, seed)
    t_generate = time.perf_counter() - t0
    text = full_text(prompt, reasoning, closed, ctx.config["state_tracking"]["prefill"])
    span = (len(prompt), len(prompt) + len(reasoning))
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    # Positions come from `enc`; the lens pass must read the same ids, untruncated (Qwen adds no
    # BOS, so they are equal, not offset).
    n = len(enc["input_ids"])
    assert ctx.model.encode(text, max_length=n)[0].tolist() == enc["input_ids"], \
        "model.encode and the offset tokenisation disagree"
    meta = words_and_meta(item, caps)
    objects = [w for w, m in meta.items()
               if m["form"] == "lower" and not m["role"].startswith("name")]
    ends = sentence_ends(enc, text, span)
    sentences, n_sentences = even(ends, settings["max_sentences"]), len(ends)
    writes = prewrite(enc, text, span, objects)
    kept = even(writes, settings["max_prewrites"])

    answer = n - 1
    t0 = time.perf_counter()
    rows, logits = experiment.read_both(ctx, text, meta, {answer: "answer"}, max_seq_len=n)
    t_read_answer, t_read_reasoning = time.perf_counter() - t0, 0.0
    at = reasoning_positions(sentences, n_sentences, writes, kept)
    if at:
        t0 = time.perf_counter()
        rows += read_reasoning(ctx, text, meta, at, n)
        t_read_reasoning = time.perf_counter() - t0

    logits, target = logits[-1], item["chains"][item["poi"]][-1]
    q1 = experiment.score_q1(tok, logits, target)
    for r in rows:
        r |= {"correct": q1["correct"], "emitted": experiment.matches(r["word"], q1["answer"]),
              "closed": closed}
    summary = {
        "k": item["k"], "h": item["h"], "item_id": item["item_id"],
        "last_is_needle": item["last_is_needle"], "poi": item["poi"],
        "names": json.dumps(item["names"]), "chains": json.dumps(item["chains"]),
        "updates": json.dumps(item["updates"]), "floor": json.dumps(item["floor"]),
        "seed": seed, "prompt": prompt, "reasoning": reasoning,
        "n_reasoning_tokens": len(new_ids), "closed": closed,
        # Positions come from re-encoding the decoded trace; this says if that is the sample.
        "retok_match": tok.encode(prompt + reasoning, add_special_tokens=False)
        == prompt_ids + new_ids,
        "n_tokens": n, "answer_pos": answer, "sentence_pos": json.dumps([t for _, t in sentences]),
        "n_sentences": n_sentences, "prewrite": json.dumps(writes),
        "prewrite_kept": json.dumps([m for m, _ in kept]),
        "t_generate": t_generate, "t_read_answer": t_read_answer,
        "t_read_reasoning": t_read_reasoning,
        **q1, "error": st.error_class(item, q1["answer"]),
        "restricted_rank": st.restricted_rank(tok, item, logits),
    }
    return rows, summary


def _show(key: tuple[int, int], s: pd.DataFrame) -> None:
    if key == (1, 1):
        first = s.iloc[0]
        print(f"\n  --- k=1 h=1 first item ---\n{first['prompt']}{first['reasoning']}\n  ---")
    print(f"  k={key[0]} h={key[1]} first item: {s.iloc[0]['n_tokens']} tokens, max position "
          f"{s.iloc[0]['answer_pos']}; median s/item: generate {s['t_generate'].median():.0f}, "
          f"read answer {s['t_read_answer'].median():.0f}, read reasoning "
          f"{s['t_read_reasoning'].median():.0f}")
    # ru_maxrss is KiB on Linux, where Modal runs.
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"  k={key[0]} h={key[1]}: top-1 {s['correct'].mean():.0%}, closed "
          f"{s['closed'].mean():.0%}, median reasoning tokens "
          f"{s['n_reasoning_tokens'].median():.0f}; "
          f"argmax {s['answer'].value_counts().head(5).to_dict()}; "
          f"peak RSS {peak_rss:.1f} GiB")


def run(spec: dict, config: dict, results_dir: Path, *, limit: int | None = None,
        force: bool = False, cell: tuple[int, int] | None = None) -> None:
    """`cell` = (k, h) runs that cell only, under its usual shard name."""
    settings = config["state_tracking"]
    templates = {**st.base_templates(settings["prefill"]), "end_think": END_THINK,
                 "registry_enable_thinking": spec["enable_thinking"], "enable_thinking": True,
                 "sentence_sampling": "even", "prewrite_sampling": "even",
                 "quotes": list(QUOTES), "quote_window": QUOTE_WINDOW}
    grid = st.grid(settings["ks"])
    if cell is not None and cell not in grid:
        raise ValueError(f"cell {cell} is not in the grid {grid}")

    def cells(ctx, n):
        names, objects = st.pools(ctx.model.tokenizer)
        caps, dropped = capitals(ctx.model.tokenizer, objects)
        templates.update(names=names, objects=objects, capitals=caps, capitals_dropped=dropped)
        return [((k, h), st.cell_tag(k, h),
                 [(item, caps) for item in st.make_items(k, h, n, names, objects)])
                for k, h in grid if cell in (None, (k, h))]

    # Thinking on overrides the registry's per-alias `enable_thinking: false`.
    experiment.run(PREFIX, settings, lambda ctx, x: run_item(ctx, *x), templates,
                   {**spec, "enable_thinking": True}, config, results_dir, limit=limit,
                   force=force, show=_show, cells=cells)


# --- analysis -----------------------------------------------------------------


def jlens_beside_logit(df: pd.DataFrame) -> pd.DataFrame:
    """`df` with its (column, lens) columns flattened: J-lens as is, the logit lens as `_ll`."""
    df.columns = [c if lens == "jlens" else f"{c}_ll" for c, lens in df.columns]
    return df


def behaviour_table(s: pd.DataFrame, off: pd.DataFrame | None) -> pd.DataFrame:
    """Behaviour per k x h x closed, beside the thinking-off derived run on the same items.
    A forced close answers mid-reasoning, so it is kept apart."""
    keys = ["k", "h", "closed"]
    out = s.groupby(keys).agg(
        n=("correct", "size"), top1=("correct", "mean"),
        med_tokens=("n_reasoning_tokens", "median"), med_rrank=("restricted_rank", "median"),
        retok_match=("retok_match", "mean"))
    if off is not None:
        same = s[[*keys, "item_id"]].merge(off[["k", "h", "item_id", "correct"]])
        out = out.join(same.groupby(keys)["correct"].mean().rename("top1_off"))
    return out.join(st.error_shares(s, keys)).reset_index()


def prewrite_table(p: pd.DataFrame) -> pd.DataFrame:
    """Presence of the object about to be written, by quoted and by what it is (`all` pools
    them), beside a recency-matched control, the most recent different object written before
    it in the reasoning, and the floor objects at the same positions."""
    pos = ["lens", "k", "h", "item_id", "token_pos"]
    w = p[p.base == p.writing]
    c = p[[*pos, "base", "present"]].rename(columns={"base": "control",
                                                     "present": "control_present"})
    w = w.merge(c, on=[*pos, "control"], how="left")
    w["control_present"] = w.control_present.astype(float)
    w["writing_is"] = w.role.map(WRITING)
    w = pd.concat([w, w.assign(writing_is="all")], ignore_index=True)
    floor = p[(p.role == "floor") & (p.base != p.writing)].merge(
        w[[*pos, "writing_is"]], on=pos)
    by = ["quoted", "writing_is", "lens"]
    out = w.groupby(by).agg(
        n=("present", "size"), present=("present", "mean"),
        n_control=("control_present", "count"), control=("control_present", "mean")
    ).join(floor.groupby(by)["present"].mean().rename("floor")).unstack("lens")
    out = jlens_beside_logit(out)
    return out[["n", "present", "control", "floor", "n_control", "present_ll", "control_ll",
                "floor_ll"]].reset_index()


def sentence_table(p: pd.DataFrame) -> pd.DataFrame:
    """At sentence ends, by reasoning progress quintile: the PoI's final object, and any
    in-prompt object."""
    p = p.assign(quintile=(5 * p.sentence // p.n_sentences).astype(int))
    pos = ["lens", "quintile", "k", "h", "item_id", "token_pos"]
    target = p[p.role == "target"].groupby(["quintile", "lens"])["present"].mean()
    any_ = p[p.role.isin(st.IN_PROMPT)].groupby(pos)["present"].any().groupby(
        ["quintile", "lens"]).mean()
    n = p[(p.role == "target") & (p.lens == "jlens")].groupby("quintile").size()
    out = jlens_beside_logit(pd.DataFrame({"target": target, "any_in_prompt": any_})
                             .unstack("lens"))
    return out.join(n.rename("n"))[["n", "target", "target_ll", "any_in_prompt",
                                    "any_in_prompt_ll"]].reset_index()


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p on `b` and `c` discordant pairs."""
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def correct_prefix(s: pd.DataFrame, dropped: list[str]) -> pd.Series:
    """`correct`, or the argmax starts a target whose capitalised form is not one token, so
    `Vase` is emitted as ` V`."""
    start = s.answer.str.strip().str.casefold()
    starts_target = [t in dropped and x != "" and t.startswith(x)
                     for x, t in zip(start, s.expected)]
    return s.correct | pd.Series(starts_target, index=s.index)


def paired_counts(g: pd.DataFrame) -> dict:
    """Top-1 on and off, the four agreement counts, and the McNemar p, over `g`'s pairs."""
    on, off = g.correct_prefix_on, g.correct_prefix_off
    on_only, off_only = int((on & ~off).sum()), int((~on & off).sum())
    return {"n": len(g), "top1_on": on.mean(), "top1_off": off.mean(),
            "strict_on": g.correct_on.mean(), "strict_off": g.correct_off.mean(),
            "on_only": on_only, "off_only": off_only, "both": int((on & off).sum()),
            "neither": int((~on & ~off).sum()), "p": mcnemar(on_only, off_only)}


def paired(s: pd.DataFrame, off: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thinking on vs off on the same items, scored by `correct_prefix` with strict `correct`
    beside: per cell, and pooled over all items and over the items whose last update is a
    hay."""
    cols = ["k", "h", "item_id", "correct", "correct_prefix"]
    j = s[[*cols, "last_is_needle"]].merge(off[cols], on=["k", "h", "item_id"],
                                           suffixes=("_on", "_off"), validate="one_to_one")
    cells = pd.DataFrame([{"k": k, "h": h, **paired_counts(g)}
                          for (k, h), g in j.groupby(["k", "h"])])
    pooled = pd.DataFrame([{"items": "all", **paired_counts(j)},
                           {"items": "hay_last", **paired_counts(j[~j.last_is_needle])}])
    return cells, pooled


def lengths(s: pd.DataFrame) -> pd.DataFrame:
    """Per k x h: median and IQR of reasoning tokens, sentences and object mentions."""
    s = s.assign(mentions=s.prewrite.map(lambda w: len(json.loads(w))))
    out = {}
    for c, short in (("n_reasoning_tokens", "tokens"), ("n_sentences", "sentences"),
                     ("mentions", "mentions")):
        g = s.groupby(["k", "h"])[c]
        out[f"med_{short}"], out[f"iqr_{short}"] = g.median(), g.quantile(.75) - g.quantile(.25)
    return pd.DataFrame(out).reset_index()


def length_regression(s: pd.DataFrame) -> pd.DataFrame:
    """OLS n_reasoning_tokens ~ k + pre_hays + tail_hays: coefficients, SE and t."""
    terms, s = ["k", "pre_hays", "tail_hays"], st.with_hay_split(s)
    X = np.column_stack([np.ones(len(s)), s[terms].to_numpy(float)])
    y = s.n_reasoning_tokens.to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    se = np.sqrt(np.diag(resid @ resid / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    return pd.DataFrame({"term": ["intercept", *terms], "coef": beta, "se": se, "t": beta / se})


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman's rank correlation of `x` and `y`."""
    return float(np.corrcoef(x.rank(), y.rank())[0, 1])


def floor_writes(p: pd.DataFrame, by: str = "writing") -> pd.DataFrame:
    """Unquoted writes of a floor object, per `by` (`writing`, or `prose`: one of PROSE), most
    written first: n and presence, J-lens and logit lens (`_ll`)."""
    w = p[(p.role == "floor") & (p.base == p.writing) & ~p.quoted]
    out = w.assign(prose=w.writing.isin(PROSE)).groupby([by, "lens"])["present"].agg(
        ["size", "mean"]).unstack("lens")
    return pd.DataFrame({"n": out[("size", "jlens")], "present": out[("mean", "jlens")],
                         "present_ll": out[("mean", "logit")]}).sort_values(
        "n", ascending=False).reset_index()


def alternation(words: list[str]) -> str:
    """A regex alternation of `words`, longest first so a word never matches inside another."""
    return "|".join(map(re.escape, sorted(words, key=len, reverse=True)))


def state_claims(reasoning: str, names: list[str], objects: list[str]
                 ) -> list[tuple[int, str, str]]:
    """(char offset of the object word, name, object) for each explicit holder statement in
    the trace, matched in either case; name and object come back in their item form."""
    pattern = re.compile(CLAIM.format(names=alternation(names), objects=alternation(objects)),
                         re.IGNORECASE)
    name_of, object_of = {n.casefold(): n for n in names}, {o.casefold(): o for o in objects}
    return [(m.start(2), name_of[m.group(1).casefold()], object_of[m.group(2).casefold()])
            for m in pattern.finditer(reasoning)]


def claim_class(o: str, chain: list[str]) -> str:
    """`true-final` if `o` ends the name's chain, `true-ever` if it is in it, else `never`."""
    if o == chain[-1]:
        return "true-final"
    return "true-ever" if o in chain else "never"


def claim_table(s: pd.DataFrame) -> pd.DataFrame:
    """One row per state claim, with its `claim_class`."""
    out = []
    for it in s.itertuples():
        chains = json.loads(it.chains)
        objects = [o for c in chains.values() for o in c] + json.loads(it.floor)
        for offset, name, o in state_claims(it.reasoning, list(chains), objects):
            out.append({"k": it.k, "h": it.h, "item_id": it.item_id, "correct": it.correct,
                        "offset": offset, "name": name, "object": o,
                        "claim": claim_class(o, chains[name]), "final": chains[name][-1]})
    return pd.DataFrame(out, columns=["k", "h", "item_id", "correct", "offset", "name",
                                      "object", "claim", "final"])


def claim_summary(s: pd.DataFrame, c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Claims per class per k, with items, items with a claim and items with a `never` one;
    and the share of items with a `never` claim, by correct."""
    keys = ["k", "h", "item_id"]
    items = s.set_index(keys).index
    never = c[c.claim == "never"].set_index(keys).index
    s = s.assign(has_claim=items.isin(c.set_index(keys).index), has_never=items.isin(never))
    by_k = pd.crosstab(c.k, c.claim).reindex(columns=CLAIMS, fill_value=0).join(
        s.groupby("k").agg(items=("item_id", "size"), with_claim=("has_claim", "sum"),
                           with_never=("has_never", "sum")))
    by_correct = s.groupby("correct").agg(n=("item_id", "size"), share_never=("has_never", "mean"))
    return by_k.reset_index(), by_correct.reset_index()


def claim_presence(p: pd.DataFrame, s: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """At the pre-write position of a `never` or `true-ever` claim's object: presence of the
    written object and of the name's final object, J-lens and logit lens (`_ll`)."""
    mention = {}
    for it in s.itertuples():
        for m, w in enumerate(json.loads(it.prewrite)):
            # Claim offsets are in the reasoning, prewrite offsets in the full text.
            mention[(it.k, it.h, it.item_id, w[3] - len(it.prompt))] = m
    c = c[c.claim.isin(["never", "true-ever"])]
    c = c.assign(mention=[mention.get(x) for x in zip(c.k, c.h, c.item_id, c.offset)]).dropna(
        subset="mention").astype({"mention": float})
    x = c.merge(p[["lens", "k", "h", "item_id", "mention", "base", "present"]],
                on=["k", "h", "item_id", "mention"])
    written = x[x.base == x.object].groupby(["claim", "lens"])["present"].mean()
    final = x[x.base == x.final].groupby(["claim", "lens"])["present"].mean()
    rates = jlens_beside_logit(pd.DataFrame({"written": written, "final": final})
                               .unstack("lens"))
    n = x[(x.lens == "jlens") & (x.base == x.object)].groupby("claim").size().rename("n")
    return rates.join(n)[["n", "written", "final", "written_ll", "final_ll"]].reset_index()


def presence_tables(alias: str, results_dir: Path, s: pd.DataFrame, limit: bool, k: int,
                    rank_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Presence at answer, prewrite and sentence positions, reduced one shard at a time."""
    parts = {"answer": [], "prewrite": [], "sentence": []}
    keys = {"answer": ANSWER_KEYS, "prewrite": PREWRITE_KEYS, "sentence": SENTENCE_KEYS}
    suffix = "_limit" if limit else ""
    for cell_k, cell_h in s[["k", "h"]].drop_duplicates().itertuples(index=False):
        path = Path(results_dir) / alias / f"{PREFIX}_{st.cell_tag(cell_k, cell_h)}{suffix}.parquet"
        # A shard can lack the prewrite or sentence columns if no item had such positions.
        names = set(pq.read_schema(path).names)
        df = pd.read_parquet(path, columns=[c for c in COLUMNS if c in names])
        for at, g in df.groupby("readout_at"):
            parts[at].append(presence(g, k, rank_col=rank_col, keys=keys[at]))
    return tuple(pd.concat(parts[at], ignore_index=True) if parts[at] else None
                 for at in ("answer", "prewrite", "sentence"))


def thinking_off_summary(alias: str, results_dir: Path) -> pd.DataFrame | None:
    """state_tracking's derived summaries, the full run if there is one, else the `--limit`
    run, else None."""
    for limit in (False, True):
        try:
            return experiment.load("state_tracking_derived", alias, results_dir, limit=limit,
                                   kind="_summary", cell="k")
        except FileNotFoundError:
            continue
    return None


def print_prewrite(head: str, writes: pd.DataFrame, tag: str) -> None:
    """The pre-write table, with a loud warning where the object about to be written is
    hardly above its control."""
    print(f"\n{head} pre-write: the object about to be written, by quoted "
          f"(copied from an update sentence) and what it is, against the most recent "
          f"other object written and the floor {tag}")
    table = prewrite_table(writes)
    print(table.to_string(**FMT))
    for row in table[table.writing_is == "all"].itertuples():
        if row.present - row.control < st.NEAR_FLOOR:
            print(f"\n  !!! POSITIVE CONTROL FAILS (quoted={row.quoted}): the object "
                  f"about to be written is present {row.present:.3f}, the control "
                  f"{row.control:.3f}; the readout cannot see the next object here !!!")


def report(spec: dict, config: dict, results_dir: Path) -> None:
    k = config["readout"]["primary_k"]
    rank_col, alias = VIEWS[0][0], spec["alias"]
    tag = f"[{rank_col}, in-band, k={k}, either case form]"
    off = thinking_off_summary(alias, results_dir)
    for label, limit, s in experiment.summaries(PREFIX, alias, results_dir, cell="k"):
        head = f"{PREFIX} {label}"
        print(f"\n{head}: T1 behaviour per k x h, top1_off = thinking-off derived on "
              f"the same items (error columns: share of items)")
        print(behaviour_table(s, off).to_string(**FMT))
        print(f"  argmax: {s['answer'].value_counts().head(8).to_dict()}")
        answer, writes, sentences = presence_tables(alias, results_dir, s, limit, k, rank_col)
        print(f"\n{head} answer contents by closed x k x correct, J-lens with logit "
              f"lens (_ll) {tag}; presence pools both case forms, so not comparable to "
              f"state_tracking's lowercase-only figures")
        # contents_by_correct splits by `arm`; here that slot carries `closed`.
        print(st.contents_by_correct(answer.assign(arm=answer.closed))
              .rename(columns={"arm": "closed"}).to_string(**FMT))
        if writes is not None:
            print_prewrite(head, writes, tag)
        if sentences is not None:
            print(f"\n{head} sentence ends by reasoning progress quintile {tag}")
            print(sentence_table(sentences).to_string(**FMT))
        manifest = Path(results_dir) / alias / f"{PREFIX}_manifest.json"
        dropped = json.loads(manifest.read_text())["templates"]["capitals_dropped"]
        print_mined_tables(s, off, writes, dropped, head, tag)


def print_paired(head: str, s: pd.DataFrame, off: pd.DataFrame, dropped: list[str]
                 ) -> pd.DataFrame:
    """Thinking on vs off per cell and pooled; returns `off` with its `correct_prefix`."""
    p_fmt = {**FMT, "formatters": {"p": "{:.2g}".format}}
    # Same tokenizer, so the same capitals are dropped.
    off = off.assign(correct_prefix=correct_prefix(off, dropped))
    print(f"  thinking off: changes {int((off.correct_prefix & ~off.correct).sum())} of "
          f"{len(off)} items")
    cells, pooled = paired(s, off)
    print(f"\n{head} paired thinking on vs off per item, exact McNemar p on the discordant "
          f"pairs")
    print(cells.to_string(**p_fmt))
    print(f"\n{head} paired, pooled over all items and over items whose last update is a hay")
    print(pooled.to_string(**p_fmt))
    return off


def print_lengths(head: str, s: pd.DataFrame, off: pd.DataFrame | None) -> None:
    """Reasoning length per cell, its regression on the item design, and its rank correlation
    with thinking-off top-1."""
    print(f"\n{head} reasoning length per k x h: median and IQR of tokens, sentences and object "
          f"mentions")
    print(lengths(s).to_string(**FMT))
    print(f"\n{head} OLS n_reasoning_tokens ~ k + pre_hays + tail_hays, all traces (n={len(s)}) "
          f"and closed ones (n={int(s.closed.sum())})")
    print(pd.concat([length_regression(t).assign(traces=name)
                     for name, t in (("all", s), ("closed", s[s.closed]))])
          [["traces", "term", "coef", "se", "t"]].to_string(**FMT))
    if off is not None:
        cells = s.groupby(["k", "h"]).n_reasoning_tokens.median().to_frame().join(
            off.groupby(["k", "h"])[["correct_prefix", "correct"]].mean())
        print(f"  Spearman, per-cell median tokens vs thinking-off top-1 over {len(cells)} cells: "
              f"{spearman(cells.n_reasoning_tokens, cells.correct_prefix):.3f} (correct_prefix), "
              f"{spearman(cells.n_reasoning_tokens, cells.correct):.3f} (strict)")


def print_prewrite_by_k(head: str, writes: pd.DataFrame, tag: str) -> None:
    """The unquoted pre-write table per k, and the floor-object writes."""
    print(f"\n{head} pre-write by k, unquoted mentions only {tag}")
    by_k = pd.concat([prewrite_table(writes[writes.k == k]).assign(k=k)
                      for k in sorted(writes.k.unique())])
    print(by_k[~by_k.quoted][["k", "writing_is", "n", "present", "control", "floor",
                              "present_ll", "control_ll", "floor_ll"]].to_string(**FMT))
    print(f"\n{head} unquoted floor-object writes, top 10 words, and {'/'.join(PROSE)} "
          f"(prose) vs the other floor words {tag}")
    print(floor_writes(writes).head(10).to_string(**FMT))
    print(floor_writes(writes, "prose").to_string(**FMT))


def print_incorrect(head: str, s: pd.DataFrame, off: pd.DataFrame | None) -> None:
    """Every incorrect item with thinking on beside its thinking-off error, and the error
    classes."""
    wrong = s.loc[~s.correct, ["k", "h", "item_id", "closed", "answer", "correct",
                               "correct_prefix", "error"]]
    if off is not None:
        wrong = wrong.merge(off[["k", "h", "item_id", "error"]], on=["k", "h", "item_id"],
                            how="left", suffixes=("", "_off"))
    print(f"\n{head} every incorrect item with thinking on, and its thinking-off error")
    print(wrong.to_string(**FMT))
    under_prefix = s.error.where(~s.correct_prefix, "target")
    print(f"  error classes, thinking on: {s.error.value_counts().to_dict()}; under "
          f"correct_prefix: {under_prefix.value_counts().to_dict()}")


def print_claims(head: str, s: pd.DataFrame, writes: pd.DataFrame | None, tag: str) -> None:
    """The trace's state claims per class, and the readout at the claimed objects."""
    c = claim_table(s)
    by_k, by_correct = claim_summary(s, c)
    print(f"\n{head} trace state claims (rule-based, unvalidated) per class per k, and items "
          f"with a claim and with a `never` claim")
    print(by_k.to_string(**FMT))
    print("  share of items with a `never` claim, by correct")
    print(by_correct.to_string(**FMT))
    if writes is not None and len(c):
        print(f"\n{head} EXPLORATORY: at the pre-write position of a `never` or `true-ever` "
              f"claim's object, presence of the written object vs the name's final object {tag}")
        print(claim_presence(writes, s, c).to_string(**FMT))


def print_mined_tables(s: pd.DataFrame, off: pd.DataFrame | None, writes: pd.DataFrame | None,
                       dropped: list[str], head: str, tag: str) -> None:
    """The tables mined from the stored run: on vs off, length, pre-write by k, errors and
    the trace's state claims. `dropped`: objects whose capitalised form is not one token."""
    s = s.assign(correct_prefix=correct_prefix(s, dropped))
    print(f"\n{head} correct_prefix also counts an argmax that starts a target whose "
          f"capitalised form is not one token, capitals_dropped {dropped}: changes "
          f"{int((s.correct_prefix & ~s.correct).sum())} of {len(s)} items")
    if off is not None:
        off = print_paired(head, s, off, dropped)
    print_lengths(head, s, off)
    if writes is not None:
        print_prewrite_by_k(head, writes, tag)
    print_incorrect(head, s, off)
    print_claims(head, s, writes, tag)
