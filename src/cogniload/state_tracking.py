"""state_tracking: who holds what after a chain of conditional updates.

Three people each hold an object. Each update names its person by the object
they hold now, so the final state needs every intermediate, in order. Needles
change the person asked about (the PoI), hays change someone else. Arm
`derived` shows the updates only; arm `copyable` writes the full state after
each one. Items are identical across arms. Read at the answer and at the
period ending each update.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from math import factorial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cogniload import experiment, prompts, readout, registry
from cogniload.experiment import VIEWS, presence, rate

ARMS = ("derived", "copyable")
N_PEOPLE = 3
NAMES = ["Ann", "Ben", "Tom", "Sam", "Kim", "Max", "Dan", "Joe", "Amy", "Eve"]
OBJECTS = ["key", "lamp", "cup", "pen", "map", "coin", "bell", "ring", "box", "bag", "fan", "jar",
           "rope", "drum", "comb", "nail", "clip", "book", "hat", "sock", "fork", "bowl", "vase",
           "flag", "kite", "rock", "shell", "stick", "brush", "chain"]
INITIAL = "{name} holds the {obj}."
UPDATE = "The person holding the {cond} swaps it for the {new}{state}."
# Copyable arm only: the state after the update, names in initial-state order.
STATE = " ({pairs})"
PAIR = "{name}: {obj}"
# "Answer in one word": without it Qwen answers in markdown (argmax ` **`).
QUESTION = "What is {poi} holding? Answer in one word."
IN_PROMPT = ["target", "chain", "other_current", "other_stale"]
# The top-token dump: one cell, every third layer across the band, top tokens per lens.
TOP_TOKEN_CELL = (3, 3)
TOP_TOKEN_LAYERS = range(37, 54, 3)
TOP_TOKENS_LENS, TOP_TOKENS_MODEL = 10, 5
# At the period ending the last update: band-min over the band layers, sorted into these.
TOP_TOKENS_BAND = 25
CATEGORIES = ["list_objects", "names", "instruction", "formatting", "other"]
LAST_UPDATE_COLUMNS = ["just_new", "poi_current", "other_current", "stale", "name_poi",
                       "name_other", "floor"]
# Copyable target presence must clear the floor by this much, or the readout is blind here.
NEAR_FLOOR = 0.1
KEYS = ["lens", "readout_at", "update", "arm", "k", "h", "item_id", "word", "role", "chain_j",
        "hay_matched", "correct", "emitted"]


def pools(tokenizer: Any) -> tuple[list[str], list[str]]:
    """Names and objects that are one token after a space. No object inside another, so
    matching a decoded token to an object is unambiguous."""
    def one_token(w):
        try:
            readout.token_id(tokenizer, w)
        except ValueError:
            return False
        return True
    names = [w for w in NAMES if one_token(w)]
    objects = [w for w in OBJECTS if one_token(w)]
    objects = [o for o in objects if not any(o != p and o in p for p in objects)]
    # 3 + k + h <= 18 objects per item, and at least 3 left for the floor.
    assert len(names) >= 6 and len(objects) >= 21, (names, objects)
    return names, objects


def hays(k: int) -> list[int]:
    return sorted({1, k, 2 * k})


def make_items(k: int, h: int, n: int, names: list[str], objects: list[str]) -> list[dict]:
    """`n` items for cell (k, h). Seeded by the cell alone, so both arms see the same items
    and `--limit n` takes the first n."""
    rng = random.Random(f"state_tracking-{k}-{h}")
    items = []
    for i in range(n):
        people = rng.sample(names, N_PEOPLE)
        objs = rng.sample(objects, N_PEOPLE + k + h)
        poi = rng.choice(people)
        # Alternate items end in a hay, so "the last new object" is not a shortcut.
        last_is_needle = i % 2 == 1
        free = k + h - 1
        needles = set(rng.sample(range(free), k - last_is_needle)) | ({free} if last_is_needle else set())
        state = dict(zip(people, objs))
        chains = {p: [o] for p, o in state.items()}
        updates = []
        for step, new in enumerate(objs[N_PEOPLE:]):
            who = poi if step in needles else rng.choice([p for p in people if p != poi])
            updates.append({"step": step, "is_needle": who == poi, "who": who,
                            "cond": state[who], "new": new})
            state[who] = new
            chains[who].append(new)
            updates[-1]["state"] = dict(state)
        item = {"k": k, "h": h, "item_id": i, "names": people, "poi": poi, "chains": chains,
                "updates": updates, "last_is_needle": last_is_needle,
                "floor": [o for o in objects if o not in objs]}
        validate(item)
        items.append(item)
    return items


def validate(item: dict) -> None:
    objs = [c[0] for c in item["chains"].values()] + [u["new"] for u in item["updates"]]
    assert len(set(objs)) == len(objs), "object reused"
    assert len(item["chains"][item["poi"]]) == item["k"] + 1, "PoI chain is not length k"
    prev = {p: c[0] for p, c in item["chains"].items()}
    for u in item["updates"]:
        assert len(set(u["state"].values())) == N_PEOPLE, "two people share an object"
        changed = [p for p in prev if prev[p] != u["state"][p]]
        assert changed == [u["who"]] and u["is_needle"] == (u["who"] == item["poi"])
        prev = u["state"]
    assert sum(not u["is_needle"] for u in item["updates"]) == item["h"]
    assert item["last_is_needle"] == item["updates"][-1]["is_needle"]


def build(tokenizer: Any, item: dict, arm: str, prefill: str, *,
          enable_thinking: bool) -> tuple[str, list[int]]:
    """The prompt, and the character offset of the period ending each update. Hays use
    `item["hay"]` if the item carries one (hay_type, through `run_item`), else `UPDATE`."""
    first = [INITIAL.format(name=p, obj=item["chains"][p][0]) for p in item["names"]]
    hay = item.get("hay", UPDATE)
    updates = [(UPDATE if u["is_needle"] else hay).format(
        cond=u["cond"], new=u["new"], who=u["who"], state=STATE.format(pairs=", ".join(
        PAIR.format(name=p, obj=o) for p, o in u["state"].items())) if arm == "copyable" else "")
        for u in item["updates"]]
    user = " ".join(first + updates + [QUESTION.format(poi=item["poi"])])
    text = prompts.render_chat(tokenizer, user, enable_thinking=enable_thinking, prefill=prefill)
    char, ends = text.index(user) + len(" ".join(first)), []
    for s in updates:
        char += 1 + len(s)
        ends.append(char - 1)
    return text, ends


def period_tokens(tokenizer: Any, text: str, chars: list[int]) -> list[int]:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    out = []
    for c in chars:
        i = next(i for i, (a, b) in enumerate(enc["offset_mapping"]) if a <= c < b)
        token = tokenizer.decode([enc["input_ids"][i]])
        assert "." in token, f"expected a period at char {c}, got {token!r}"
        out.append(i)
    return out


def words_and_meta(item: dict, arm: str) -> dict[str, dict]:
    """Every item object, the floor objects and the three names, mapped to their role row."""
    base = {"arm": arm, "k": item["k"], "h": item["h"], "item_id": item["item_id"],
            "last_is_needle": item["last_is_needle"], "person": None, "chain_j": None,
            "intro_step": None, "is_hay_new": False, "hay_matched": False}
    updates, poi = item["updates"], item["poi"]
    intro = {u["new"]: u for u in updates}
    # The hay new-object introduced closest before each needle: a recency-matched control.
    matched = set()
    for t, u in enumerate(updates):
        before = [v["new"] for v in updates[:t] if not v["is_needle"]]
        if u["is_needle"] and before:
            matched.add(before[-1])
    meta = {}
    for p, chain in item["chains"].items():
        for i, o in enumerate(chain):
            j = len(chain) - 1 - i
            role = ("target" if j == 0 else "chain") if p == poi else (
                "other_current" if j == 0 else "other_stale")
            u = intro.get(o)
            meta[o] = {**base, "role": role, "person": p, "chain_j": j if p == poi else None,
                       "intro_step": u["step"] if u else None,
                       "is_hay_new": bool(u) and not u["is_needle"], "hay_matched": o in matched}
    meta |= {o: {**base, "role": "floor"} for o in item["floor"]}
    meta |= {p: {**base, "role": "name_poi" if p == poi else "name_other", "person": p}
             for p in item["names"]}
    return meta


def error_class(item: dict, answer: str) -> str:
    for p, chain in item["chains"].items():
        for i, o in enumerate(chain):
            if experiment.matches(answer, o):
                j = len(chain) - 1 - i
                if p == item["poi"]:
                    return "target" if j == 0 else f"poi_stale_{j}"
                return "other_current" if j == 0 else "other_stale"
    return "out_of_prompt"


def run_item(ctx, item: dict, arm: str) -> tuple[list[dict], dict]:
    tok = ctx.model.tokenizer
    text, ends = build(tok, item, arm, ctx.config["state_tracking"]["prefill"],
                       enable_thinking=ctx.spec["enable_thinking"])
    periods = period_tokens(tok, text, ends)
    answer = ctx.model.encode(text).shape[-1] - 1
    rows, logits = experiment.read_both(ctx, text, words_and_meta(item, arm),
                                        {**dict.fromkeys(periods, "update"), answer: "answer"})
    # Behaviour is the readout pass's own logits at the answer, so no generation.
    logits, target = logits[-1], item["chains"][item["poi"]][-1]
    q1 = experiment.score_q1(tok, logits, target)
    ids = [readout.token_id(tok, o) for c in item["chains"].values() for o in c]
    update_of = {t: u for u, t in enumerate(periods)}
    for r in rows:
        r |= {"update": update_of.get(r["token_pos"]), "correct": q1["correct"],
              "emitted": experiment.matches(r["word"], q1["answer"])}
    summary = {
        "k": item["k"], "h": item["h"], "arm": arm, "item_id": item["item_id"],
        "last_is_needle": item["last_is_needle"], "poi": item["poi"],
        "names": json.dumps(item["names"]), "chains": json.dumps(item["chains"]),
        "updates": json.dumps(item["updates"]), "floor": json.dumps(item["floor"]),
        "answer_pos": answer, "period_pos": json.dumps(periods), "prompt": text,
        **q1, "error": error_class(item, q1["answer"]),
        # Rank among the item's in-prompt objects only.
        "restricted_rank": int((logits[ids] > logits[readout.token_id(tok, target)]).sum()) + 1,
    }
    return rows, summary


def _show(key: tuple[int, int], s: pd.DataFrame) -> None:
    if key == (2, 1):
        print(f"\n  --- k=2 h=1 first item prompt ---\n{s.iloc[0]['prompt']}\n  ---")
    print(f"  k={key[0]} h={key[1]}: top-1 {s['correct'].mean():.0%}, restricted top-1 "
          f"{(s['restricted_rank'] == 1).mean():.0%}; argmax {s['answer'].value_counts().head(5).to_dict()}")


def instruction_words(prefill: str) -> set[str]:
    """Casefolded words of the prompt templates, placeholders removed."""
    text = re.sub(r"\{\w+\}", " ", " ".join([INITIAL, UPDATE, STATE, PAIR, QUESTION, prefill]))
    return {w.casefold() for w in re.findall(r"[A-Za-z]+", text)}


def token_category(token: str, objects: list[str], names: list[str], instruction: set[str]) -> str:
    w = token.strip().casefold()
    if w in {o.casefold() for o in objects}:
        return "list_objects"
    if w in {n.casefold() for n in names}:
        return "names"
    if w in instruction:
        return "instruction"
    # Punctuation, whitespace and special tokens such as <|im_end|>.
    if not any(c.isalnum() for c in w) or re.fullmatch(r"<.*>", w):
        return "formatting"
    return "other"


def dump_top_tokens(spec: dict, config: dict, arm: str, n: int) -> None:
    """Print the top tokens at the answer, and band-min top tokens by category at the period
    ending the last update, for the first `n` items of one cell; no shards."""
    recorded = experiment.recorded_layers(spec, config)
    layers = [l for l in TOP_TOKEN_LAYERS if l in recorded]
    band = [l for l in recorded if spec["band"][0] <= l < spec["band"][1]]
    model, lens = registry.load(spec)
    tok, prefill = model.tokenizer, config["state_tracking"]["prefill"]
    names, objects = pools(tok)
    instruction, counts = instruction_words(prefill), {}
    for item in make_items(*TOP_TOKEN_CELL, n, names, objects):
        text, ends = build(tok, item, arm, prefill, enable_thinking=spec["enable_thinking"])
        answer, last = model.encode(text).shape[-1] - 1, period_tokens(tok, text, ends)[-1]
        in_item = {o for c in item["chains"].values() for o in c}
        print(f"\n{arm} k={item['k']} h={item['h']} item {item['item_id']}: PoI {item['poi']}, "
              f"chain {item['chains'][item['poi']]}")
        for lens_name, jacobian in (("jlens", True), ("logit", False)):
            tops, own = readout.top_tokens(model, lens, text, layers=layers, position=answer,
                                           n=TOP_TOKENS_LENS, use_jacobian=jacobian)
            for layer, words in tops.items():
                print(f"  {lens_name} L{layer}: {words}")
        print(f"  model top {TOP_TOKENS_MODEL}: "
              f"{[tok.decode([i]) for i in own.topk(TOP_TOKENS_MODEL).indices.tolist()]}")
        for lens_name, jacobian in (("jlens", True), ("logit", False)):
            by = {c: [] for c in CATEGORIES}
            for word, rank in readout.band_top_tokens(model, lens, text, layers=band,
                                                      position=last, n=TOP_TOKENS_BAND,
                                                      use_jacobian=jacobian):
                category = token_category(word, objects, names, instruction)
                star = "*" if category == "list_objects" and word.strip() in in_item else ""
                by[category].append(f"{word!r}{star}({rank})")
            counts.setdefault(lens_name, Counter()).update({c: len(v) for c, v in by.items()})
            print(f"  {lens_name} last-update period, band-min top {TOP_TOKENS_BAND}: "
                  + " | ".join(f"{c}: {' '.join(v)}" for c, v in by.items() if v))
    for lens_name, c in counts.items():
        print(f"\n{arm} {lens_name} last-update period, top {TOP_TOKENS_BAND} per category over "
              f"{n} items: {dict((k, c[k]) for k in CATEGORIES)}")


def run(spec: dict, config: dict, results_dir: Path, *, arm: str, limit: int | None = None,
        force: bool = False, top_tokens: int | None = None) -> None:
    if top_tokens:
        return dump_top_tokens(spec, config, arm, top_tokens)
    settings = config["state_tracking"]
    templates = {"initial": INITIAL, "update": UPDATE, "question": QUESTION,
                 "prefill": settings["prefill"],
                 "state": STATE if arm == "copyable" else None, "pair": PAIR}

    def cells(ctx, n):
        names, objects = pools(ctx.model.tokenizer)
        # The manifest is written after the loop, so the filtered pools land in it.
        templates.update(names=names, objects=objects)
        return [((k, h), f"k{k}_h{h}", make_items(k, h, n, names, objects))
                for k in settings["ks"] for h in hays(k)]

    experiment.run(f"state_tracking_{arm}", settings, lambda ctx, item: run_item(ctx, item, arm),
                   templates, spec, config, results_dir, limit=limit, force=force, show=_show,
                   cells=cells)


# --- analysis -----------------------------------------------------------------


def t1(s: pd.DataFrame) -> pd.DataFrame:
    """Load: behaviour per arm x k x h, split by whether the last update is a needle."""
    idx = [s.arm, s.k, s.h]
    out = s.groupby(["arm", "k", "h"]).agg(
        n=("correct", "size"), top1=("correct", "mean"), med_rank=("expected_rank", "median"),
        med_rrank=("restricted_rank", "median"))
    split = s.pivot_table(index=["arm", "k", "h"], columns="last_is_needle", values="correct",
                          aggfunc="mean").rename(columns={False: "top1_hay_last",
                                                          True: "top1_needle_last"})
    errors = pd.crosstab(idx, s.error, normalize="index").drop(columns="target", errors="ignore")
    return out.join(split).join(errors).reset_index()


def t2(p: pd.DataFrame, lens: str, by: tuple[str, ...] = ("arm", "k")) -> pd.DataFrame:
    """Answer-position contents per `by` (default arm x k, pooled over h); `ll_*` are the
    logit lens."""
    main, ll = p[p.lens == lens], p[p.lens == "logit"]
    logit = dict(list(ll.groupby(list(by))))
    out = []
    for key, g in main.groupby(list(by)):
        l, k = logit.get(key, ll[:0]), g.k.iloc[0]
        out.append({**dict(zip(by, key)), "n": (g.role == "target").sum(),
                    "target": rate(g[g.role == "target"]),
                    **{f"chain_{j}": rate(g[g.chain_j == j]) for j in range(1, k + 1)},
                    **{r: rate(g[g.role == r]) for r in ("other_current", "other_stale")},
                    "hay_matched": rate(g[g.hay_matched]), "floor": rate(g[g.role == "floor"]),
                    "name_poi": rate(g[g.role == "name_poi"]),
                    "name_other": rate(g[g.role == "name_other"]), "emitted": rate(g[g.emitted]),
                    "ll_target": rate(l[l.role == "target"]), "ll_chain_1": rate(l[l.chain_j == 1])})
    df = pd.DataFrame(out)
    chains = sorted((c for c in df if c.startswith("chain_")), key=lambda c: int(c[6:]))
    return df[[*by, "n", "target", *chains,
               *[c for c in df if c not in chains and c not in (*by, "n", "target")]]]


def t3(rows: pd.DataFrame, k_: int, rank_col: str, lens: str) -> pd.DataFrame:
    """Layer order of the chain, derived arm, k in {3, 4}, correct items. A chain object's
    layer is its first band layer with rank <= k_, else the layer of its minimum rank."""
    g = rows[rows.in_band & (rows.lens == lens) & (rows.arm == "derived") & rows.k.isin([3, 4])
             & rows.correct & rows.role.isin(["target", "chain"])]
    out = []
    for k, x in g.groupby("k"):
        keys = ["h", "item_id", "chain_j"]
        legible = x[x[rank_col] <= k_].groupby(keys)["layer"].min()
        best = x.loc[x.groupby(keys)[rank_col].idxmin()].set_index(keys)["layer"]
        # Columns from the initial object (j = k) to the target (j = 0).
        L = legible.reindex(best.index).fillna(best).unstack("chain_j").sort_index(
            axis=1, ascending=False)
        steps = L.diff(axis=1).iloc[:, 1:]
        out.append({"k": k, "n": len(L), **{f"layer_j{int(j)}": L[j].median() for j in L},
                    "weak_monotone": (steps >= 0).all(axis=1).mean(),
                    "strict": (steps > 0).all(axis=1).mean(), "baseline": 1 / factorial(k)})
    return pd.DataFrame(out)


def t4(p: pd.DataFrame, lens: str) -> pd.DataFrame:
    """Derived arm, per k: wrong items beside correct ones. `fog` = in-prompt objects present."""
    g = p[(p.lens == lens) & (p.arm == "derived")]
    out = []
    for k, x in g.groupby("k"):
        row = {"k": k}
        for name, ok in (("wrong", False), ("right", True)):
            y = x[x.correct == ok]
            row |= {f"n_{name}": (y.role == "target").sum(),
                    f"target_{name}": rate(y[y.role == "target"]),
                    f"emitted_{name}": rate(y[y.emitted]),
                    f"fog_{name}": y[y.role.isin(IN_PROMPT)].groupby(["h", "item_id"])["present"]
                    .sum().mean()}
        out.append(row)
    return pd.DataFrame(out)


def hay_split(updates: list[dict]) -> tuple[int, int]:
    """Hays before and after the PoI's last needle."""
    last = max(t for t, u in enumerate(updates) if u["is_needle"])
    hay = [not u["is_needle"] for u in updates]
    return sum(hay[:last]), sum(hay[last + 1:])


def logistic(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Newton fit of a logistic regression: coefficients, standard errors, log-likelihood."""
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ beta))
        info = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(info, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-X @ beta)), 1e-12, 1 - 1e-12)
    loglik = float(y @ np.log(p) + (1 - y) @ np.log(1 - p))
    return beta, np.sqrt(np.diag(np.linalg.inv(info))), loglik


def recency_regression(s: pd.DataFrame) -> pd.DataFrame:
    """correct ~ k + pre_hays + tail_hays: Wald z, and the likelihood-ratio p (chi-square,
    one degree of freedom) for dropping each term."""
    terms = ["k", "pre_hays", "tail_hays"]
    X = np.column_stack([np.ones(len(s)), s[terms].to_numpy(float)])
    y = s.correct.to_numpy(float)
    beta, se, loglik = logistic(X, y)
    drop = [2 * (loglik - logistic(np.delete(X, i, axis=1), y)[2]) for i in range(len(beta))]
    return pd.DataFrame({"term": ["intercept", *terms], "coef": beta, "se": se, "z": beta / se,
                         "lr_p": [math.erfc(math.sqrt(max(d, 0) / 2)) for d in drop]})


def with_hay_split(s: pd.DataFrame) -> pd.DataFrame:
    split = s.updates.map(lambda u: hay_split(json.loads(u)))
    return s.assign(pre_hays=split.str[0], tail_hays=split.str[1])


def copyable_beside_derived(s: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Top-1 per cell in both arms, and the copyable arm's target and floor presence at the
    answer: the readout's positive control."""
    top1 = s.pivot_table(index=["k", "h"], columns="arm", values="correct", aggfunc="mean")
    copyable = p[p.arm == "copyable"].groupby(["k", "h"])
    return top1.add_prefix("top1_").join(pd.DataFrame({
        "copyable_target": copyable.apply(lambda g: rate(g[g.role == "target"])),
        "copyable_floor": copyable.apply(lambda g: rate(g[g.role == "floor"]))})).reset_index()


def groups(p: pd.DataFrame) -> pd.DataFrame:
    """Answer contents per arm x k x correct, J-lens with the logit lens (`_ll`) beside.
    `emitted` is NaN where the emitted token is not a read word."""
    by = ["arm", "k", "correct"]
    j, ll = (t2(p, lens, tuple(by)) for lens in ("jlens", "logit"))
    cols = ["target", *[c for c in j if c.startswith("chain_")], "other_current", "other_stale",
            "name_poi", "name_other", "emitted", "floor"]
    both = j.merge(ll, on=[*by, "n"], suffixes=("", "_ll"))
    return both[[*by, "n", *[c for x in cols for c in (x, f"{x}_ll")]]]


def band_best(p: pd.DataFrame, rank_col: str, k_: int) -> pd.DataFrame:
    """Per arm x k and lens: the median over items of the best band rank among every word read
    (objects, floor, names), and the share of items where any of them is <= k_."""
    best = p.groupby(["lens", "arm", "k", "h", "item_id"])[rank_col].min()
    out = best.groupby(["arm", "k", "lens"]).agg(
        median_best="median", any_present=lambda r: (r <= k_).mean()).unstack("lens")
    out.columns = [f"{c}_{lens}" for c, lens in out.columns]
    return out.reset_index()


def last_update_words(s: pd.DataFrame) -> pd.DataFrame:
    """Each read word's column at the period ending the item's last update. The last new
    object is `just_new` only; current and stale come from the state after that update."""
    out = []
    for it in s.itertuples():
        last = json.loads(it.updates)[-1]
        column = {o: "floor" for o in json.loads(it.floor)}
        column |= {p: "name_poi" if p == it.poi else "name_other" for p in json.loads(it.names)}
        for p, chain in json.loads(it.chains).items():
            column |= dict.fromkeys(chain, "stale")
            column[last["state"][p]] = "poi_current" if p == it.poi else "other_current"
        column[last["new"]] = "just_new"
        out += [{"arm": it.arm, "k": it.k, "h": it.h, "item_id": it.item_id,
                 "last_is_needle": it.last_is_needle, "word": w, "column": c}
                for w, c in column.items()]
    return pd.DataFrame(out)


def last_update(p: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Period ending the last update, per arm x last_is_needle x correct: J-lens with the
    logit lens (`_ll`) beside."""
    keys = ["arm", "last_is_needle", "correct"]
    m = p.merge(last_update_words(s), on=["arm", "k", "h", "item_id", "word"])
    rates = m.groupby([*keys, "column", "lens"])["present"].mean().unstack(["column", "lens"])
    items = m[m.lens == "jlens"].drop_duplicates([*keys, "k", "h", "item_id"])
    out = items.groupby(keys).size().rename("n").to_frame()
    for c in LAST_UPDATE_COLUMNS:
        out[c], out[f"{c}_ll"] = rates.get((c, "jlens")), rates.get((c, "logit"))
    return out.reset_index()


def recency_and_errors(s: pd.DataFrame, fmt: dict) -> None:
    """Recency split, the pre/tail-hay regression and error classes, derived arm."""
    d = with_hay_split(s[s.arm == "derived"])
    print("\nstate_tracking A1 recency split per cell (derived)")
    print(t1(d)[["k", "h", "n", "top1", "top1_needle_last", "top1_hay_last"]].to_string(**fmt))
    print(f"\nstate_tracking A1 logistic regression correct ~ k + pre_hays + tail_hays "
          f"(derived, n={len(d)})")
    print(recency_regression(d).to_string(**fmt, formatters={"lr_p": "{:.2g}".format}))
    print("\nstate_tracking A1 top-1 (n) by tail_hays (columns) at each k (derived)")
    print(d.groupby(["k", "tail_hays"])["correct"].agg(lambda c: f"{c.mean():.2f} ({len(c)})")
          .unstack("tail_hays", fill_value="").to_string())
    for by in ("k", "h"):
        print(f"\nstate_tracking A2 error classes per {by} (derived; share of items)")
        errors = pd.crosstab(d[by], d.error, normalize="index")
        print(errors.join(d.groupby(by).size().rename("n")).reset_index().to_string(**fmt))


def report(spec: dict, config: dict, results_dir: Path) -> None:
    k, fmt = config["readout"]["primary_k"], dict(index=False, float_format="%.3f")
    data = {}
    for arm in ARMS:
        for label, df, s in experiment.runs(f"state_tracking_{arm}", spec["alias"], results_dir,
                                            cell="k"):
            # The tables read the band only: the answer, and the period ending the last update.
            keep = df.in_band & ((df.readout_at == "answer") | (df["update"] == df.k + df.h - 1))
            data.setdefault(label, []).append((df[keep], s))
    for label, parts in data.items():
        both = pd.concat([d for d, _ in parts], ignore_index=True)
        rows, last = both[both.readout_at == "answer"], both[both.readout_at == "update"]
        s = pd.concat([x for _, x in parts], ignore_index=True)
        print(f"\nstate_tracking {label}: T1 load (error columns: share of items)")
        print(t1(s).to_string(**fmt))
        for arm, g in s.groupby("arm"):
            print(f"  {arm} argmax: {g['answer'].value_counts().head(8).to_dict()}")
        if set(ARMS) <= set(s.arm):
            rank_col, lens = VIEWS[0]
            p = presence(rows[rows.lens == lens], k, rank_col=rank_col, keys=KEYS)
            control = copyable_beside_derived(s, p)
            print(f"\nstate_tracking {label} B top-1 by arm; copyable target presence at the "
                  f"answer [{rank_col}, {lens}, in-band, k={k}]")
            print(control.to_string(**fmt))
            target, floor = rate(p[(p.arm == "copyable") & (p.role == "target")]), rate(
                p[(p.arm == "copyable") & (p.role == "floor")])
            if target - floor < NEAR_FLOOR:
                print(f"\n  !!! POSITIVE CONTROL FAILS: copyable target presence {target:.3f} is "
                      f"near the floor {floor:.3f}; the readout cannot see objects here !!!")
        else:
            print(f"\nstate_tracking {label} B skipped: needs both arms, have {sorted(set(s.arm))}")
        for rank_col, lens in VIEWS:
            tag = f"[{rank_col}, {lens}, in-band, k={k}]"
            p = presence(rows, k, rank_col=rank_col, keys=KEYS)
            print(f"\nstate_tracking {label} T2 answer contents {tag}")
            print(t2(p, lens).to_string(**fmt))
            if (s.arm == "derived").any():
                print(f"\nstate_tracking {label} T3 chain layer order {tag}")
                print(t3(rows, k, rank_col, lens).to_string(**fmt))
                print(f"\nstate_tracking {label} T4 wrong vs right {tag}")
                print(t4(p, lens).to_string(**fmt))
        if label == "FULL":
            rank_col, tag = VIEWS[0][0], f"[{VIEWS[0][0]}, in-band, k={k}]"
            p = presence(rows, k, rank_col=rank_col, keys=KEYS)
            print(f"\nstate_tracking {label} groups by arm x k x correct, J-lens with logit "
                  f"lens (_ll) {tag}")
            print(groups(p).to_string(**fmt))
            print(f"\nstate_tracking {label} best band rank among all read words, per item "
                  f"{tag}")
            print(band_best(p, rank_col, k).to_string(**fmt))
            print(f"\nstate_tracking {label} period ending the last update, by arm x "
                  f"last_is_needle x correct, J-lens with logit lens (_ll) {tag}")
            print(last_update(presence(last, k, rank_col=rank_col, keys=KEYS), s)
                  .to_string(**fmt))
        # The regression needs the full grid.
        if label == "FULL" and (s.arm == "derived").any():
            recency_and_errors(s, fmt)
