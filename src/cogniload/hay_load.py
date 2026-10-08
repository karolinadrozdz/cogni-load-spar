"""hay_factorial and hay_type: state_tracking's derived items with the hays varied.

`tail_hays` is the number of hays after the PoI's last needle, `pre_hays` the
rest. hay_factorial crosses `k` with `h` and stratifies `tail_hays` within each
cell, so `k`, `pre_hays` and `tail_hays` are independent by design. hay_type
fixes `k`, `h` and `tail_hays` and changes only the hay sentence: the same
items in every arm. Prompts, readout and scoring are state_tracking's
`run_item`, so the rows and summaries have its columns.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from cogniload import experiment, state_tracking as st

# Needles always use st.UPDATE; `build` uses these for hays only.
HAYS = {"same": st.UPDATE,
        "named": "{who} swaps the {cond} for the {new}.",
        "reworded": "Whoever has the {cond} trades it for the {new}."}
N_FLOOR = 3


def tail_strata(h: int, tails: list[int]) -> list[int]:
    return [t for t in tails if t <= h]


def hs_for_pool(settings: dict, n_objects: int) -> list[int]:
    """The h grid, with every h capped at `h_cap` if the largest cell would leave
    fewer than N_FLOOR pool objects for the floor."""
    hs = settings["hs"]
    if st.N_PEOPLE + max(settings["ks"]) + max(hs) + N_FLOOR > n_objects:
        hs = sorted({min(h, settings["h_cap"]) for h in hs})
    return hs


def make_items(prefix: str, k: int, h: int, tails: list[int], n: int, names: list[str],
               objects: list[str]) -> list[dict]:
    """`n` items for cell (k, h); item i has `tails[i % len(tails)]` hays after the PoI's
    last needle, so a full cell splits equally. Seeded by `prefix` and the cell, not the arm."""
    rng = random.Random(f"{prefix}-{k}-{h}")
    items = []
    for i in range(n):
        tail = tails[i % len(tails)]
        people = rng.sample(names, st.N_PEOPLE)
        objs = rng.sample(objects, st.N_PEOPLE + k + h)
        poi = rng.choice(people)
        # The last needle goes just before the tail; the other k - 1 anywhere before it.
        last = k + h - tail - 1
        needles = set(rng.sample(range(last), k - 1)) | {last}
        state = dict(zip(people, objs))
        chains = {p: [o] for p, o in state.items()}
        updates = []
        for step, new in enumerate(objs[st.N_PEOPLE:]):
            who = poi if step in needles else rng.choice([p for p in people if p != poi])
            updates.append({"step": step, "is_needle": who == poi, "who": who,
                            "cond": state[who], "new": new})
            state[who] = new
            chains[who].append(new)
            updates[-1]["state"] = dict(state)
        item = {"k": k, "h": h, "item_id": i, "names": people, "poi": poi, "chains": chains,
                "updates": updates, "last_is_needle": tail == 0,
                "floor": [o for o in objects if o not in objs],
                "pre_hays": h - tail, "tail_hays": tail}
        st.validate(item)
        items.append(item)
    return items


def run_item(ctx, item: dict, arm: str) -> tuple[list[dict], dict]:
    rows, summary = st.run_item(ctx, item, arm)
    return rows, summary | {"pre_hays": item["pre_hays"], "tail_hays": item["tail_hays"]}


def _show(prompt_at: tuple[int, int] | None):
    def show(key: tuple[int, int], s: pd.DataFrame) -> None:
        if prompt_at in (None, key):
            print(f"\n  --- k={key[0]} h={key[1]} first item prompt ---\n{s.iloc[0]['prompt']}\n  ---")
        by_tail = s.groupby("tail_hays")["correct"].mean().round(2).to_dict()
        print(f"  k={key[0]} h={key[1]}: top-1 {s['correct'].mean():.0%}, by tail_hays {by_tail}; "
              f"argmax {s['answer'].value_counts().head(5).to_dict()}")
    return show


def run_hay_factorial(spec: dict, config: dict, results_dir: Path, *, limit: int | None = None,
                      force: bool = False) -> None:
    settings = config["hay_factorial"]
    templates = {"initial": st.INITIAL, "update": st.UPDATE, "question": st.QUESTION,
                 "prefill": config["state_tracking"]["prefill"]}

    def cells(ctx, n):
        names, objects = st.pools(ctx.model.tokenizer)
        hs = hs_for_pool(settings, len(objects))
        # The manifest is written after the loop, so the pools and the grid land in it.
        templates.update(names=names, objects=objects, hs=hs, h_capped=hs != settings["hs"])
        if hs != settings["hs"]:
            print(f"  pool of {len(objects)} objects: h capped at {settings['h_cap']}")
        return [((k, h), f"k{k}_h{h}", make_items("hay_factorial", k, h,
                                                  tail_strata(h, settings["tail_hays"]), n,
                                                  names, objects))
                for k in settings["ks"] for h in hs]

    experiment.run("hay_factorial", settings, lambda ctx, item: run_item(ctx, item, "derived"),
                   templates, spec, config, results_dir, limit=limit, force=force,
                   show=_show((3, 4)), cells=cells)


def run_hay_type(spec: dict, config: dict, results_dir: Path, *, arm: str,
                 limit: int | None = None, force: bool = False) -> None:
    settings = config["hay_type"]
    k, h = settings["k"], settings["h"]
    templates = {"initial": st.INITIAL, "update": st.UPDATE, "hay": HAYS[arm],
                 "question": st.QUESTION, "prefill": config["state_tracking"]["prefill"]}

    def cells(ctx, n):
        names, objects = st.pools(ctx.model.tokenizer)
        templates.update(names=names, objects=objects)
        items = make_items("hay_type", k, h, [settings["tail_hays"]], n, names, objects)
        return [((k, h), f"k{k}_h{h}", [{**it, "hay": HAYS[arm]} for it in items])]

    experiment.run(f"hay_type_{arm}", settings, lambda ctx, item: run_item(ctx, item, arm),
                   templates, spec, config, results_dir, limit=limit, force=force,
                   show=_show(None), cells=cells)


# --- analysis -----------------------------------------------------------------

FMT = dict(index=False, float_format="%.3f")


def summaries(prefix: str, alias: str, results_dir: Path):
    """Yield `(label, summary)` for the `--limit` run and the full run, where present."""
    for limit, label in ((True, "SMOKE (--limit)"), (False, "FULL")):
        try:
            yield label, experiment.load(prefix, alias, results_dir, limit=limit,
                                         kind="_summary", cell="k")
        except FileNotFoundError:
            continue


def by_cell(s: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Top-1 and the error-class shares per `keys`."""
    out = s.groupby(keys).agg(n=("correct", "size"), top1=("correct", "mean"))
    errors = pd.crosstab([s[c] for c in keys], s.error, normalize="index")
    return out.join(errors.drop(columns="target", errors="ignore")).reset_index()



def report_hay_factorial(spec: dict, config: dict, results_dir: Path) -> None:
    for label, s in summaries("hay_factorial", spec["alias"], results_dir):
        print(f"\nhay_factorial {label}: T1 per k x h x tail_hays (error columns: share of items)")
        print(by_cell(s, ["k", "h", "tail_hays"]).to_string(**FMT))
        print(f"\nhay_factorial {label}: logistic regression correct ~ k + pre_hays + tail_hays "
              f"(n={len(s)})")
        print(st.recency_regression(s).to_string(**FMT, formatters={"lr_p": "{:.2g}".format}))


def report_hay_type(spec: dict, config: dict, results_dir: Path) -> None:
    settings, alias = config["hay_type"], spec["alias"]
    reference = dict(summaries("hay_factorial", alias, results_dir)).get("FULL")
    reference = [] if reference is None else [
        reference[(reference.k == settings["k"]) & (reference.h == 0)].assign(
            arm=f"reference: hay_factorial k={settings['k']} h=0")]
    runs = {}
    for arm in HAYS:
        for label, s in summaries(f"hay_type_{arm}", alias, results_dir):
            runs.setdefault(label, []).append(s)
    for label, parts in runs.items():
        print(f"\nhay_type {label}: k={settings['k']} h={settings['h']} "
              f"tail_hays={settings['tail_hays']} per arm (error columns: share of items)")
        print(by_cell(pd.concat(parts + reference, ignore_index=True), ["arm"]).to_string(**FMT))
