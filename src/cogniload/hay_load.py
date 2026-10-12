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
from cogniload.state_tracking import FMT

# Needles always use st.UPDATE; `build` uses these for hays only.
HAYS = {"same": st.UPDATE,
        "named": "{who} swaps the {cond} for the {new}.",
        "reworded": "Whoever has the {cond} trades it for the {new}."}
N_FLOOR = 3


def tail_strata(h: int, tails: list[int]) -> list[int]:
    """The `tail_hays` values a cell with `h` hays can hold."""
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
        people, objs, poi = st.cast(rng, names, objects, k + h)
        # The last needle goes just before the tail; the other k - 1 anywhere before it.
        last = k + h - tail - 1
        needles = set(rng.sample(range(last), k - 1)) | {last}
        item = st.play_item(rng, k, h, i, people, objs, poi, needles, objects)
        items.append(item | {"pre_hays": h - tail, "tail_hays": tail})
    return items


def run_item(ctx, item: dict, arm: str) -> tuple[list[dict], dict]:
    """state_tracking's `run_item`, with the item's hay split added to the summary."""
    rows, summary = st.run_item(ctx, item, arm)
    return rows, summary | {"pre_hays": item["pre_hays"], "tail_hays": item["tail_hays"]}


def _show(prompt_at: tuple[int, int] | None):
    """A per-cell progress printer that also shows the first prompt of cell `prompt_at`
    (every cell if None)."""
    def show(key: tuple[int, int], s: pd.DataFrame) -> None:
        if prompt_at in (None, key):
            print(f"\n  --- k={key[0]} h={key[1]} first item prompt ---\n"
                  f"{s.iloc[0]['prompt']}\n  ---")
        by_tail = s.groupby("tail_hays")["correct"].mean().round(2).to_dict()
        print(f"  k={key[0]} h={key[1]}: top-1 {s['correct'].mean():.0%}, by tail_hays {by_tail}; "
              f"argmax {s['answer'].value_counts().head(5).to_dict()}")
    return show


def run_hay_factorial(spec: dict, config: dict, results_dir: Path, *, limit: int | None = None,
                      force: bool = False) -> None:
    settings = config["hay_factorial"]
    templates = st.base_templates(config["state_tracking"]["prefill"])

    def cells(ctx, n):
        names, objects = st.pools(ctx.model.tokenizer)
        hs = hs_for_pool(settings, len(objects))
        # The manifest is written after the loop, so the pools and the grid land in it.
        templates.update(names=names, objects=objects, hs=hs, h_capped=hs != settings["hs"])
        if hs != settings["hs"]:
            print(f"  pool of {len(objects)} objects: h capped at {settings['h_cap']}")
        return [((k, h), st.cell_tag(k, h),
                 make_items("hay_factorial", k, h, tail_strata(h, settings["tail_hays"]), n,
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
        return [((k, h), st.cell_tag(k, h), [{**it, "hay": HAYS[arm]} for it in items])]

    experiment.run(f"hay_type_{arm}", settings, lambda ctx, item: run_item(ctx, item, arm),
                   templates, spec, config, results_dir, limit=limit, force=force,
                   show=_show(None), cells=cells)


# --- analysis -----------------------------------------------------------------


def by_cell(s: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Top-1 and the error-class shares per `keys`."""
    out = s.groupby(keys).agg(n=("correct", "size"), top1=("correct", "mean"))
    return out.join(st.error_shares(s, keys)).reset_index()


def report_hay_factorial(spec: dict, config: dict, results_dir: Path) -> None:
    for label, _, s in experiment.summaries("hay_factorial", spec["alias"], results_dir,
                                            cell="k"):
        print(f"\nhay_factorial {label}: T1 per k x h x tail_hays (error columns: share of items)")
        print(by_cell(s, ["k", "h", "tail_hays"]).to_string(**FMT))
        print(f"\nhay_factorial {label}: logistic regression correct ~ k + pre_hays + tail_hays "
              f"(n={len(s)})")
        print(st.recency_regression(s).to_string(**FMT, formatters={"lr_p": "{:.2g}".format}))


def reference_cell(alias: str, results_dir: Path, k: int) -> list[pd.DataFrame]:
    """hay_factorial's full-run cell (k, h=0) labelled as an arm, or nothing without a full run."""
    for label, _, s in experiment.summaries("hay_factorial", alias, results_dir, cell="k"):
        if label == "FULL":
            return [s[(s.k == k) & (s.h == 0)].assign(arm=f"reference: hay_factorial k={k} h=0")]
    return []


def report_hay_type(spec: dict, config: dict, results_dir: Path) -> None:
    settings, alias = config["hay_type"], spec["alias"]
    reference = reference_cell(alias, results_dir, settings["k"])
    runs = {}
    for arm in HAYS:
        for label, _, s in experiment.summaries(f"hay_type_{arm}", alias, results_dir,
                                                cell="k"):
            runs.setdefault(label, []).append(s)
    for label, parts in runs.items():
        print(f"\nhay_type {label}: k={settings['k']} h={settings['h']} "
              f"tail_hays={settings['tail_hays']} per arm (error columns: share of items)")
        print(by_cell(pd.concat(parts + reference, ignore_index=True), ["arm"]).to_string(**FMT))
