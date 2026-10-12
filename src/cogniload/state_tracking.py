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
from cogniload.experiment import LENSES, VIEWS, presence, rate

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
FMT = dict(index=False, float_format="%.3f")


def is_one_token(tokenizer: Any, word: str) -> bool:
    """Whether `word` after a space is a single token."""
    try:
        readout.token_id(tokenizer, word)
    except ValueError:
        return False
    return True


def pools(tokenizer: Any) -> tuple[list[str], list[str]]:
    """Names and objects that are one token after a space. No object inside another, so
    matching a decoded token to an object is unambiguous."""
    names = [w for w in NAMES if is_one_token(tokenizer, w)]
    objects = [w for w in OBJECTS if is_one_token(tokenizer, w)]
    objects = [o for o in objects if not any(o != p and o in p for p in objects)]
    # 3 + k + h <= 18 objects per item, and at least 3 left for the floor.
    assert len(names) >= 6 and len(objects) >= 21, (names, objects)
    return names, objects


def hays(k: int) -> list[int]:
    """The hay counts crossed with `k`."""
    return sorted({1, k, 2 * k})


def grid(ks: list[int]) -> list[tuple[int, int]]:
    """Every (k, h) cell."""
    return [(k, h) for k in ks for h in hays(k)]


def cell_tag(k: int, h: int) -> str:
    """The shard name of cell (k, h)."""
    return f"k{k}_h{h}"


def base_templates(prefill: str) -> dict:
    """The prompt templates every state_tracking variant records in its manifest."""
    return {"initial": INITIAL, "update": UPDATE, "question": QUESTION, "prefill": prefill}


def cast(rng: random.Random, names: list[str], objects: list[str], n_updates: int
         ) -> tuple[list[str], list[str], str]:
    """The item's people, its objects (initial ones first, then one per update) and its PoI."""
    people = rng.sample(names, N_PEOPLE)
    objs = rng.sample(objects, N_PEOPLE + n_updates)
    poi = rng.choice(people)
    return people, objs, poi


def play_item(rng: random.Random, k: int, h: int, item_id: int, people: list[str],
              objs: list[str], poi: str, needles: set[int], objects: list[str]) -> dict:
    """The validated item whose update at each step in `needles` changes the PoI, and whose
    other updates change someone else, drawn from `rng`."""
    state = dict(zip(people, objs))
    chains = {p: [o] for p, o in state.items()}
    updates = []
    for step, new in enumerate(objs[N_PEOPLE:]):
        who = poi if step in needles else rng.choice([p for p in people if p != poi])
        update = {"step": step, "is_needle": who == poi, "who": who, "cond": state[who],
                  "new": new}
        state[who] = new
        chains[who].append(new)
        update["state"] = dict(state)
        updates.append(update)
    item = {"k": k, "h": h, "item_id": item_id, "names": people, "poi": poi, "chains": chains,
            "updates": updates, "last_is_needle": updates[-1]["is_needle"],
            "floor": [o for o in objects if o not in objs]}
    validate(item)
    return item


def make_items(k: int, h: int, n: int, names: list[str], objects: list[str]) -> list[dict]:
    """`n` items for cell (k, h). Seeded by the cell alone, so both arms see the same items
    and `--limit n` takes the first n."""
    rng = random.Random(f"state_tracking-{k}-{h}")
    items = []
    for i in range(n):
        people, objs, poi = cast(rng, names, objects, k + h)
        # Alternate items end in a hay, so "the last new object" is not a shortcut.
        last_step = k + h - 1
        if i % 2 == 1:
            needles = set(rng.sample(range(last_step), k - 1)) | {last_step}
        else:
            needles = set(rng.sample(range(last_step), k))
        items.append(play_item(rng, k, h, i, people, objs, poi, needles, objects))
    return items


def validate(item: dict) -> None:
    """Assert that objects are unique, the PoI's chain has k updates, each update changes
    exactly its person, and there are h hays."""
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


def update_sentence(update: dict, template: str, arm: str) -> str:
    """One update in `template`, with the state after it appended in the copyable arm."""
    state = ""
    if arm == "copyable":
        pairs = ", ".join(PAIR.format(name=p, obj=o) for p, o in update["state"].items())
        state = STATE.format(pairs=pairs)
    return template.format(cond=update["cond"], new=update["new"], who=update["who"],
                           state=state)


def build(tokenizer: Any, item: dict, arm: str, prefill: str, *,
          enable_thinking: bool) -> tuple[str, list[int]]:
    """The prompt, and the character offset of the period ending each update. Hays use
    `item["hay"]` if the item carries one (hay_type, through `run_item`), else `UPDATE`."""
    first = " ".join(INITIAL.format(name=p, obj=item["chains"][p][0]) for p in item["names"])
    hay = item.get("hay", UPDATE)
    updates = [update_sentence(u, UPDATE if u["is_needle"] else hay, arm)
               for u in item["updates"]]
    user = " ".join([first, *updates, QUESTION.format(poi=item["poi"])])
    text = prompts.render_chat(tokenizer, user, enable_thinking=enable_thinking, prefill=prefill)
    char, ends = text.index(user) + len(first), []
    for sentence in updates:
        char += 1 + len(sentence)
        ends.append(char - 1)
    return text, ends


def period_tokens(tokenizer: Any, text: str, chars: list[int]) -> list[int]:
    """Token index of the period at each character offset in `chars`."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    out = []
    for c in chars:
        i = experiment.token_at(enc["offset_mapping"], c)
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
            if p == poi:
                role = "target" if j == 0 else "chain"
            else:
                role = "other_current" if j == 0 else "other_stale"
            u = intro.get(o)
            meta[o] = {**base, "role": role, "person": p, "chain_j": j if p == poi else None,
                       "intro_step": u["step"] if u else None,
                       "is_hay_new": bool(u) and not u["is_needle"], "hay_matched": o in matched}
    meta |= {o: {**base, "role": "floor"} for o in item["floor"]}
    meta |= {p: {**base, "role": "name_poi" if p == poi else "name_other", "person": p}
             for p in item["names"]}
    return meta


def error_class(item: dict, answer: str) -> str:
    """Which in-prompt object the answer is, by holder and how stale, or `out_of_prompt`."""
    for p, chain in item["chains"].items():
        for i, o in enumerate(chain):
            if experiment.matches(answer, o):
                j = len(chain) - 1 - i
                if p == item["poi"]:
                    return "target" if j == 0 else f"poi_stale_{j}"
                return "other_current" if j == 0 else "other_stale"
    return "out_of_prompt"


def restricted_rank(tokenizer: Any, item: dict, logits) -> int:
    """The target's rank among the item's in-prompt objects only."""
    ids = [readout.token_id(tokenizer, o) for c in item["chains"].values() for o in c]
    target = readout.token_id(tokenizer, item["chains"][item["poi"]][-1])
    return int((logits[ids] > logits[target]).sum()) + 1


def run_item(ctx, item: dict, arm: str) -> tuple[list[dict], dict]:
    """Readout rows at the answer and each update's period, and the item's summary row."""
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
        "restricted_rank": restricted_rank(tok, item, logits),
    }
    return rows, summary


def _show(key: tuple[int, int], s: pd.DataFrame) -> None:
    if key == (2, 1):
        print(f"\n  --- k=2 h=1 first item prompt ---\n{s.iloc[0]['prompt']}\n  ---")
    argmax = s["answer"].value_counts().head(5).to_dict()
    print(f"  k={key[0]} h={key[1]}: top-1 {s['correct'].mean():.0%}, restricted top-1 "
          f"{(s['restricted_rank'] == 1).mean():.0%}; argmax {argmax}")


def instruction_words(prefill: str) -> set[str]:
    """Casefolded words of the prompt templates, placeholders removed."""
    text = re.sub(r"\{\w+\}", " ", " ".join([INITIAL, UPDATE, STATE, PAIR, QUESTION, prefill]))
    return {w.casefold() for w in re.findall(r"[A-Za-z]+", text)}


def token_category(token: str, objects: list[str], names: list[str], instruction: set[str]) -> str:
    """Which of CATEGORIES a decoded token falls in."""
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


def print_answer_top_tokens(model: Any, lens: Any, text: str, layers: list[int],
                            answer: int) -> None:
    """Print each lens's top tokens per layer at the answer, then the model's own."""
    for lens_name, jacobian in LENSES:
        tops, own = readout.top_tokens(model, lens, text, layers=layers, position=answer,
                                       n=TOP_TOKENS_LENS, use_jacobian=jacobian)
        for layer, words in tops.items():
            print(f"  {lens_name} L{layer}: {words}")
    top_ids = own.topk(TOP_TOKENS_MODEL).indices.tolist()
    print(f"  model top {TOP_TOKENS_MODEL}: {[model.tokenizer.decode([i]) for i in top_ids]}")


def top_tokens_by_category(model: Any, lens: Any, text: str, band: list[int], position: int,
                           use_jacobian: bool, categorise, in_item: set[str]
                           ) -> dict[str, list[str]]:
    """The band-min top tokens at `position`, grouped by category; an object of this item
    is starred."""
    by = {c: [] for c in CATEGORIES}
    for word, rank in readout.band_top_tokens(model, lens, text, layers=band, position=position,
                                              n=TOP_TOKENS_BAND, use_jacobian=use_jacobian):
        category = categorise(word)
        star = "*" if category == "list_objects" and word.strip() in in_item else ""
        by[category].append(f"{word!r}{star}({rank})")
    return by


def dump_top_tokens(spec: dict, config: dict, arm: str, n: int) -> None:
    """Print the top tokens at the answer, and band-min top tokens by category at the period
    ending the last update, for the first `n` items of one cell; no shards."""
    recorded = experiment.recorded_layers(spec, config)
    layers = [l for l in TOP_TOKEN_LAYERS if l in recorded]
    band = [l for l in recorded if spec["band"][0] <= l < spec["band"][1]]
    model, lens = registry.load(spec)
    tok, prefill = model.tokenizer, config["state_tracking"]["prefill"]
    names, objects = pools(tok)
    instruction = instruction_words(prefill)
    categorise = lambda word: token_category(word, objects, names, instruction)
    counts = {}
    for item in make_items(*TOP_TOKEN_CELL, n, names, objects):
        text, ends = build(tok, item, arm, prefill, enable_thinking=spec["enable_thinking"])
        answer = model.encode(text).shape[-1] - 1
        last = period_tokens(tok, text, ends)[-1]
        in_item = {o for c in item["chains"].values() for o in c}
        print(f"\n{arm} k={item['k']} h={item['h']} item {item['item_id']}: PoI {item['poi']}, "
              f"chain {item['chains'][item['poi']]}")
        print_answer_top_tokens(model, lens, text, layers, answer)
        for lens_name, jacobian in LENSES:
            by = top_tokens_by_category(model, lens, text, band, last, jacobian, categorise,
                                        in_item)
            counts.setdefault(lens_name, Counter()).update({c: len(v) for c, v in by.items()})
            print(f"  {lens_name} last-update period, band-min top {TOP_TOKENS_BAND}: "
                  + " | ".join(f"{c}: {' '.join(v)}" for c, v in by.items() if v))
    for lens_name, c in counts.items():
        per_category = {category: c[category] for category in CATEGORIES}
        print(f"\n{arm} {lens_name} last-update period, top {TOP_TOKENS_BAND} per category over "
              f"{n} items: {per_category}")


def run(spec: dict, config: dict, results_dir: Path, *, arm: str, limit: int | None = None,
        force: bool = False, top_tokens: int | None = None) -> None:
    if top_tokens:
        return dump_top_tokens(spec, config, arm, top_tokens)
    settings = config["state_tracking"]
    templates = {**base_templates(settings["prefill"]),
                 "state": STATE if arm == "copyable" else None, "pair": PAIR}

    def cells(ctx, n):
        names, objects = pools(ctx.model.tokenizer)
        # The manifest is written after the loop, so the filtered pools land in it.
        templates.update(names=names, objects=objects)
        return [((k, h), cell_tag(k, h), make_items(k, h, n, names, objects))
                for k, h in grid(settings["ks"])]

    experiment.run(f"state_tracking_{arm}", settings, lambda ctx, item: run_item(ctx, item, arm),
                   templates, spec, config, results_dir, limit=limit, force=force, show=_show,
                   cells=cells)


# --- analysis -----------------------------------------------------------------


def error_shares(s: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """The share of items in each error class other than `target`, per `keys`."""
    errors = pd.crosstab([s[c] for c in keys], s.error, normalize="index")
    return errors.drop(columns="target", errors="ignore")


def load_table(s: pd.DataFrame) -> pd.DataFrame:
    """Behaviour per arm x k x h, split by whether the last update is a needle."""
    keys = ["arm", "k", "h"]
    out = s.groupby(keys).agg(
        n=("correct", "size"), top1=("correct", "mean"), med_rank=("expected_rank", "median"),
        med_rrank=("restricted_rank", "median"))
    split = s.pivot_table(index=keys, columns="last_is_needle", values="correct",
                          aggfunc="mean").rename(columns={False: "top1_hay_last",
                                                          True: "top1_needle_last"})
    return out.join(split).join(error_shares(s, keys)).reset_index()


def contents_row(g: pd.DataFrame, logit: pd.DataFrame) -> dict:
    """Presence of each role in one group, J-lens `g` with the logit lens's `ll_` columns."""
    k = g.k.iloc[0]
    return {"n": (g.role == "target").sum(), "target": rate(g[g.role == "target"]),
            **{f"chain_{j}": rate(g[g.chain_j == j]) for j in range(1, k + 1)},
            **{r: rate(g[g.role == r]) for r in ("other_current", "other_stale")},
            "hay_matched": rate(g[g.hay_matched]), "floor": rate(g[g.role == "floor"]),
            "name_poi": rate(g[g.role == "name_poi"]),
            "name_other": rate(g[g.role == "name_other"]), "emitted": rate(g[g.emitted]),
            "ll_target": rate(logit[logit.role == "target"]),
            "ll_chain_1": rate(logit[logit.chain_j == 1])}


def answer_contents(p: pd.DataFrame, lens: str, by: tuple[str, ...] = ("arm", "k")
                    ) -> pd.DataFrame:
    """Answer-position contents per `by` (default arm x k, pooled over h); `ll_*` are the
    logit lens."""
    main, ll = p[p.lens == lens], p[p.lens == "logit"]
    logit = dict(list(ll.groupby(list(by))))
    df = pd.DataFrame([{**dict(zip(by, key)), **contents_row(g, logit.get(key, ll[:0]))}
                       for key, g in main.groupby(list(by))])
    chains = sorted((c for c in df if c.startswith("chain_")), key=lambda c: int(c[6:]))
    rest = [c for c in df if c not in chains and c not in (*by, "n", "target")]
    return df[[*by, "n", "target", *chains, *rest]]


def chain_layer_order(rows: pd.DataFrame, present_k: int, rank_col: str,
                      lens: str) -> pd.DataFrame:
    """Layer order of the chain, derived arm, k in {3, 4}, correct items. A chain object's
    layer is its first band layer with rank <= present_k, else the layer of its minimum rank."""
    g = rows[rows.in_band & (rows.lens == lens) & (rows.arm == "derived") & rows.k.isin([3, 4])
             & rows.correct & rows.role.isin(["target", "chain"])]
    out = []
    for k, x in g.groupby("k"):
        keys = ["h", "item_id", "chain_j"]
        legible = x[x[rank_col] <= present_k].groupby(keys)["layer"].min()
        best = x.loc[x.groupby(keys)[rank_col].idxmin()].set_index(keys)["layer"]
        # Columns from the initial object (j = k) to the target (j = 0).
        layers = legible.reindex(best.index).fillna(best).unstack("chain_j").sort_index(
            axis=1, ascending=False)
        steps = layers.diff(axis=1).iloc[:, 1:]
        out.append({"k": k, "n": len(layers),
                    **{f"layer_j{int(j)}": layers[j].median() for j in layers},
                    "weak_monotone": (steps >= 0).all(axis=1).mean(),
                    "strict": (steps > 0).all(axis=1).mean(), "baseline": 1 / factorial(k)})
    return pd.DataFrame(out)


def wrong_vs_right(p: pd.DataFrame, lens: str) -> pd.DataFrame:
    """Derived arm, per k: wrong items beside correct ones. `fog` = in-prompt objects present."""
    g = p[(p.lens == lens) & (p.arm == "derived")]
    out = []
    for k, x in g.groupby("k"):
        row = {"k": k}
        for name, ok in (("wrong", False), ("right", True)):
            y = x[x.correct == ok]
            in_prompt = y[y.role.isin(IN_PROMPT)].groupby(["h", "item_id"])["present"]
            row |= {f"n_{name}": (y.role == "target").sum(),
                    f"target_{name}": rate(y[y.role == "target"]),
                    f"emitted_{name}": rate(y[y.emitted]),
                    f"fog_{name}": in_prompt.sum().mean()}
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
    """`s` with `pre_hays` and `tail_hays` columns from its stored updates."""
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


def contents_by_correct(p: pd.DataFrame) -> pd.DataFrame:
    """Answer contents per arm x k x correct, J-lens with the logit lens (`_ll`) beside.
    `emitted` is NaN where the emitted token is not a read word."""
    by = ["arm", "k", "correct"]
    j, ll = (answer_contents(p, lens, tuple(by)) for lens in ("jlens", "logit"))
    cols = ["target", *[c for c in j if c.startswith("chain_")], "other_current", "other_stale",
            "name_poi", "name_other", "emitted", "floor"]
    both = j.merge(ll, on=[*by, "n"], suffixes=("", "_ll"))
    return both[[*by, "n", *[c for x in cols for c in (x, f"{x}_ll")]]]


def band_best(p: pd.DataFrame, rank_col: str, present_k: int) -> pd.DataFrame:
    """Per arm x k and lens: the median over items of the best band rank among every word read
    (objects, floor, names), and the share of items where any of them is <= present_k."""
    best = p.groupby(["lens", "arm", "k", "h", "item_id"])[rank_col].min()
    out = best.groupby(["arm", "k", "lens"]).agg(
        median_best="median", any_present=lambda r: (r <= present_k).mean()).unstack("lens")
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


def print_recency_and_errors(s: pd.DataFrame) -> None:
    """Recency split, the pre/tail-hay regression and error classes, derived arm."""
    d = with_hay_split(s[s.arm == "derived"])
    print("\nstate_tracking recency split per cell (derived)")
    print(load_table(d)[["k", "h", "n", "top1", "top1_needle_last", "top1_hay_last"]]
          .to_string(**FMT))
    print(f"\nstate_tracking logistic regression correct ~ k + pre_hays + tail_hays "
          f"(derived, n={len(d)})")
    print(recency_regression(d).to_string(**FMT, formatters={"lr_p": "{:.2g}".format}))
    print("\nstate_tracking top-1 (n) by tail_hays (columns) at each k (derived)")
    top1_and_n = lambda c: f"{c.mean():.2f} ({len(c)})"
    print(d.groupby(["k", "tail_hays"])["correct"].agg(top1_and_n)
          .unstack("tail_hays", fill_value="").to_string())
    for by in ("k", "h"):
        print(f"\nstate_tracking error classes per {by} (derived; share of items)")
        errors = pd.crosstab(d[by], d.error, normalize="index")
        print(errors.join(d.groupby(by).size().rename("n")).reset_index().to_string(**FMT))


def load_runs(alias: str, results_dir: Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Per run label, both arms' band rows at the answer and at the period ending the last
    update, and their summaries."""
    parts = {}
    for arm in ARMS:
        for label, df, s in experiment.runs(f"state_tracking_{arm}", alias, results_dir,
                                            cell="k"):
            keep = df.in_band & ((df.readout_at == "answer") | (df["update"] == df.k + df.h - 1))
            parts.setdefault(label, []).append((df[keep], s))
    return {label: (pd.concat([d for d, _ in x], ignore_index=True),
                    pd.concat([s for _, s in x], ignore_index=True))
            for label, x in parts.items()}


def print_positive_control(label: str, rows: pd.DataFrame, s: pd.DataFrame, k: int) -> None:
    """Top-1 by arm beside the copyable target's presence at the answer, with a loud warning
    if that presence is near the floor."""
    rank_col, lens = VIEWS[0]
    p = presence(rows[rows.lens == lens], k, rank_col=rank_col, keys=KEYS)
    print(f"\nstate_tracking {label} positive control: top-1 by arm; copyable target presence "
          f"at the answer [{rank_col}, {lens}, in-band, k={k}]")
    print(copyable_beside_derived(s, p).to_string(**FMT))
    copyable = p[p.arm == "copyable"]
    target = rate(copyable[copyable.role == "target"])
    floor = rate(copyable[copyable.role == "floor"])
    if target - floor < NEAR_FLOOR:
        print(f"\n  !!! POSITIVE CONTROL FAILS: copyable target presence {target:.3f} is "
              f"near the floor {floor:.3f}; the readout cannot see objects here !!!")


def print_view_tables(label: str, rows: pd.DataFrame, s: pd.DataFrame, k: int,
                      by_rank: dict[str, pd.DataFrame]) -> None:
    """Answer contents, chain layer order and wrong vs right, for each of VIEWS."""
    for rank_col, lens in VIEWS:
        tag = f"[{rank_col}, {lens}, in-band, k={k}]"
        p = by_rank[rank_col]
        print(f"\nstate_tracking {label} T2 answer contents {tag}")
        print(answer_contents(p, lens).to_string(**FMT))
        if (s.arm == "derived").any():
            print(f"\nstate_tracking {label} T3 chain layer order {tag}")
            print(chain_layer_order(rows, k, rank_col, lens).to_string(**FMT))
            print(f"\nstate_tracking {label} T4 wrong vs right {tag}")
            print(wrong_vs_right(p, lens).to_string(**FMT))


def print_full_tables(label: str, last: pd.DataFrame, s: pd.DataFrame, k: int,
                      p: pd.DataFrame) -> None:
    """The full run's extra tables: contents by correct, best band rank, and the period
    ending the last update."""
    rank_col = VIEWS[0][0]
    tag = f"[{rank_col}, in-band, k={k}]"
    print(f"\nstate_tracking {label} groups by arm x k x correct, J-lens with logit "
          f"lens (_ll) {tag}")
    print(contents_by_correct(p).to_string(**FMT))
    print(f"\nstate_tracking {label} best band rank among all read words, per item {tag}")
    print(band_best(p, rank_col, k).to_string(**FMT))
    print(f"\nstate_tracking {label} period ending the last update, by arm x "
          f"last_is_needle x correct, J-lens with logit lens (_ll) {tag}")
    print(last_update(presence(last, k, rank_col=rank_col, keys=KEYS), s).to_string(**FMT))


def report(spec: dict, config: dict, results_dir: Path) -> None:
    k = config["readout"]["primary_k"]
    for label, (both, s) in load_runs(spec["alias"], results_dir).items():
        rows, last = both[both.readout_at == "answer"], both[both.readout_at == "update"]
        print(f"\nstate_tracking {label}: T1 load (error columns: share of items)")
        print(load_table(s).to_string(**FMT))
        for arm, g in s.groupby("arm"):
            print(f"  {arm} argmax: {g['answer'].value_counts().head(8).to_dict()}")
        if set(ARMS) <= set(s.arm):
            print_positive_control(label, rows, s, k)
        else:
            print(f"\nstate_tracking {label} positive control skipped: needs both arms, have "
                  f"{sorted(set(s.arm))}")
        by_rank = {rank_col: presence(rows, k, rank_col=rank_col, keys=KEYS)
                   for rank_col in {rank_col for rank_col, _ in VIEWS}}
        print_view_tables(label, rows, s, k, by_rank)
        if label == "FULL":
            print_full_tables(label, last, s, k, by_rank[VIEWS[0][0]])
            # The regression needs the full grid.
            if (s.arm == "derived").any():
                print_recency_and_errors(s)
