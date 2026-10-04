"""The three numbers, and the floor check that says whether to believe them.

Everything keys off one contrast: a **non-queried** target is task state the
model is not about to say, so it is the only item whose presence distinguishes
"the J-space holds task state" from "the J-space holds what is about to be
output". The queried target is both at once and is reported separately.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PRIMARY_K = 25


def _shards(kind: str, spec_alias: str, results_dir: Path, limit: bool) -> list[Path]:
    out = Path(results_dir) / spec_alias
    pattern = f"{kind}_ct*_limit.parquet" if limit else f"{kind}_ct*.parquet"
    found = [p for p in sorted(out.glob(pattern))
             if limit or not p.name.endswith("_limit.parquet")]
    if not found:
        raise FileNotFoundError(f"no {kind} shards in {out}")
    return found


def load(spec_alias: str, results_dir: Path, *, limit: bool = False) -> pd.DataFrame:
    return pd.concat(
        (pd.read_parquet(p) for p in _shards("readout", spec_alias, results_dir, limit)),
        ignore_index=True,
    )


def load_summary(spec_alias: str, results_dir: Path, *, limit: bool = False) -> pd.DataFrame:
    return pd.concat(
        (pd.read_parquet(p) for p in _shards("summary", spec_alias, results_dir, limit)),
        ignore_index=True,
    )


def q1_diagnostic(summary: pd.DataFrame, n_examples: int = 6) -> None:
    """What is the model actually emitting at the answer position?

    Q1 accuracy near zero is below chance — guessing among a category's three
    words would score ~33%. That means the emitted token is not a stream word
    at all, and the distinction between "wrong word", "punctuation" and
    "refusal" is what says whether the prompt, the position, or the task is
    at fault. `top5` is logged per stream precisely so this needs no re-run.
    """
    print(f"\nQ1 DIAGNOSTIC — accuracy {summary['q1_correct'].mean():.0%} "
          f"over {len(summary)} streams")
    counts = summary["q1_answer"].value_counts().head(10)
    print("  most common emitted token at the answer position:")
    for token, n in counts.items():
        print(f"    {token!r:<24} {n:>4}  ({n / len(summary):.0%})")
    if "q1_expected_rank" in summary:
        ranked = summary[summary["q1_expected_rank"] > 0]["q1_expected_rank"]
        if len(ranked):
            print(f"  expected word's rank in the model distribution: "
                  f"median {ranked.median():.0f}, "
                  f"top-5 {(ranked <= 5).mean():.0%}, top-1 {(ranked == 1).mean():.0%}")
    print(f"  per C_t: " + ", ".join(
        f"C_t={c}: {g['q1_correct'].mean():.0%}"
        for c, g in summary.groupby("c_t")))
    print(f"\n  first {n_examples} streams (expected -> answer | top5):")
    for _, r in summary.head(n_examples).iterrows():
        print(f"    {r['q1_expected']!r:<12} -> {r['q1_answer']!r:<14} | {r['q1_top5']}")
    print(f"\n  Q2 accuracy {summary['q2_accuracy'].mean():.0%}, "
          f"parse rate {summary['q2_parse_rate'].mean():.0%}")
    print("  example Q2 output:")
    for raw in summary["q2_raw"].head(2):
        print(f"    {raw!r}")


ITEM_KEYS = ["lens", "readout_at", "stream_id", "c_t", "word", "role",
             "category", "recency", "is_queried", "is_category_final",
             "q2_correct", "q2_parsed"]


def presence(df: pd.DataFrame, k: int = PRIMARY_K, *, band_only: bool = True,
             rank_col: str = "rank") -> pd.DataFrame:
    """Collapse layers to one row per (lens, position, item): present or not.

    `band_only` restricts the min to in-band layers, which is the headline
    definition. `rank_col` selects full-vocabulary `rank` or `rank_wordlike`
    — the latter is the view Neuronpedia displays, and on Qwen the raw top-K
    is dominated by punctuation and fragments, so the two can differ a lot.
    """
    rows = df[df["in_band"]] if band_only else df
    grouped = rows.groupby(ITEM_KEYS, dropna=False)[rank_col].min().reset_index()
    grouped["present"] = grouped[rank_col] <= k
    return grouped


def by_layer(df: pd.DataFrame, k: int = PRIMARY_K, *, rank_col: str = "rank") -> pd.DataFrame:
    """Present-rate per layer, for the groups that matter.

    R1 picks the band on where single-concept readouts are legible, which on
    Qwen3.5-4B is the deepest layers available. If non-queried targets peak
    somewhere earlier, the band is in the wrong place and this is what shows it.
    """
    rows = df[df["lens"] == "jlens"].copy()
    rows["group"] = "other"
    rows.loc[(rows["role"] == "target") & ~rows["is_queried"], "group"] = "target_non_queried"
    rows.loc[(rows["role"] == "target") & rows["is_queried"], "group"] = "target_queried"
    rows.loc[rows["role"] == "absent", "group"] = "absent"
    rows = rows[rows["group"] != "other"]
    rows["present"] = rows[rank_col] <= k
    return (
        rows.groupby(["readout_at", "layer", "in_band", "group"])["present"]
        .mean().reset_index()
        .pivot_table(index=["readout_at", "layer", "in_band"],
                     columns="group", values="present")
        .reset_index()
    )


def floor_check(df: pd.DataFrame, k: int = PRIMARY_K) -> pd.DataFrame:
    """Is the readout working in the task at all?

    Reports both rank columns, because which one carries the signal is
    model-dependent: on Qwen3.5-4B they agree to within 2 points, while on
    Qwen3.6-27B `<|im_end|>` and underscore runs flood the full-vocab top-25
    and the queried target reads 0.29 on `rank` against 0.74 on
    `rank_wordlike`. Gating on one column alone produced a false alarm.

    Phase 0 R2 does not cover this: it tests single-concept prompts, a far
    easier regime than a 24-word stream. If the queried target is not present
    at the answer position on *either* column, every null downstream is
    uninterpretable and the run should stop here.
    """
    print(f"\nFLOOR CHECK (k={k})")
    best = 0.0
    for col in ("rank", "rank_wordlike"):
        p = presence(df, k, rank_col=col)
        p = p[p["lens"] == "jlens"]
        q = p[(p.readout_at == "answer") & p.is_queried & (p.role == "target")]
        a = p[p.role == "absent"]
        best = max(best, q["present"].mean())
        print(f"  [{col:<14}] queried target at answer: {q['present'].mean():.0%}"
              f"   absent-word floor: {a['present'].mean():.0%}")
    if best < 0.5:
        print("  *** BELOW 50% ON BOTH COLUMNS — diagnose before trusting "
              "anything below ***")
    return _floor_table(df, k)


def _floor_table(df: pd.DataFrame, k: int, *, rank_col: str = "rank") -> pd.DataFrame:
    """Per-(position, role) present rates, for the record."""
    p = presence(df, k, rank_col=rank_col)
    p = p[p["lens"] == "jlens"]
    return pd.DataFrame([
        {"readout_at": at, "role": role, "n": len(g),
         "present_rate": g["present"].mean()}
        for (at, role), g in p.groupby(["readout_at", "role"])
    ])


def capacity(df: pd.DataFrame, k: int = PRIMARY_K, *, rank_col: str = "rank") -> pd.DataFrame:
    """Non-queried targets present vs C_t, per lens and position.

    A curve that *rises then flattens* is a capacity ceiling. A curve flat at
    ~1 target is the deflationary result — the J-space holding only the item
    about to be emitted — and it would satisfy a naive "plateau" criterion, so
    the rise is the part that matters (DECISIONS.md D1).
    """
    p = presence(df, k, rank_col=rank_col)
    targets = p[(p["role"] == "target") & ~p["is_queried"]]
    per_stream = (
        targets.groupby(["lens", "readout_at", "c_t", "stream_id"])["present"]
        .sum().reset_index(name="n_present")
    )
    return (
        per_stream.groupby(["lens", "readout_at", "c_t"])["n_present"]
        .agg(["mean", "sem", "count"]).reset_index()
    )


def selectivity(df: pd.DataFrame, k: int = PRIMARY_K, tail_guard: int = 3,
                *, rank_col: str = "rank") -> pd.DataFrame:
    """Non-queried targets vs the matched control, on common recency support.

    The control is the **untracked final** occurrence: a target is its
    category's last occurrence by definition, so an arbitrary untracked item
    differs in being superseded as well as in not being tracked. Recency 1-3 is
    dropped because `tail_guard` means no target can occupy it, so `role` and
    `recency` are structurally confounded there.
    """
    p = presence(df, k, rank_col=rank_col)
    p = p[p["recency"].notna() & (p["recency"] > tail_guard)]
    target = p[(p["role"] == "target") & ~p["is_queried"]].assign(group="target_non_queried")
    control = p[(p["role"] == "untracked") & p["is_category_final"]].assign(
        group="untracked_final")
    other = p[(p["role"] == "untracked") & ~p["is_category_final"]].assign(
        group="untracked_superseded")
    # Split: an earlier word of the *queried* category is a competing answer
    # candidate, which is a different thing from a superseded item of some
    # other tracked category.
    replaced = p[(p["role"] == "replaced") & p["is_queried"]].assign(
        group="replaced_queried")
    replaced_other = p[(p["role"] == "replaced") & ~p["is_queried"]].assign(
        group="replaced_other")
    floor = presence(df, k, rank_col=rank_col)
    floor = floor[floor["role"] == "absent"].assign(group="absent")

    both = pd.concat([target, control, other, replaced, replaced_other, floor],
                     ignore_index=True)
    return (
        both.groupby(["lens", "readout_at", "c_t", "group"])["present"]
        .agg(["mean", "sem", "count"]).reset_index()
    )


def eviction(df: pd.DataFrame, k: int = PRIMARY_K, *, rank_col: str = "rank") -> pd.DataFrame:
    """Is a superseded item still there when its replacement is?

    The single-position form of the eviction question. The spec's trajectory
    version is confounded: every word spikes at its own position, so a "step at
    replacement" arrives exactly where the replacement's own spike does. Here
    the comparison is between items at one position, which has no such artifact.
    `untracked_superseded` is the baseline — went stale without ever being
    task-relevant.
    """
    s = selectivity(df, k, rank_col=rank_col)
    return s[s["group"].isin(["target_non_queried", "replaced_queried",
                              "replaced_other", "untracked_superseded", "absent"])]


#: Groups too small to mean anything get printed as a count, not a rate.
MIN_N = 10


def headline(df: pd.DataFrame, k: int = PRIMARY_K) -> pd.DataFrame:
    """The contrast the experiment exists to measure, pooled over C_t.

    Both rank columns side by side: `rank` is full vocabulary (the paper's
    protocol), `rank_wordlike` masks to word-like tokens (what Neuronpedia
    displays). On Qwen the raw top-K is mostly punctuation, so a concept can be
    plainly present in the masked view and invisible in the unmasked one.

    Read every row against `absent`, which is the chance floor.
    """
    out = []
    for rank_col in ("rank", "rank_wordlike"):
        p = presence(df, k, rank_col=rank_col)
        p = p[p["lens"] == "jlens"]
        groups = {
            # item identity
            "target_queried": p[(p.role == "target") & p.is_queried],
            "target_non_queried": p[(p.role == "target") & ~p.is_queried],
            "replaced_queried": p[(p.role == "replaced") & p.is_queried],
            "untracked": p[p.role == "untracked"],
            "absent": p[p.role == "absent"],
            # category structure: same contrast one level up
            "label_queried": p[(p.role == "label_tracked") & p.is_queried],
            "label_tracked": p[(p.role == "label_tracked") & ~p.is_queried],
            "label_untracked": p[p.role == "label_untracked"],
            "label_absent": p[p.role == "label_absent"],
        }
        for at in ("in_stream", "stream_end", "answer"):
            for name, g in groups.items():
                g = g[g.readout_at == at]
                if not len(g):
                    continue
                out.append({"rank_col": rank_col, "readout_at": at, "group": name,
                            "present": g["present"].mean(), "n": len(g)})
    wide = pd.DataFrame(out).pivot_table(
        index=["readout_at", "group"], columns="rank_col",
        values=["present", "n"]).reset_index()
    order = ["in_stream", "stream_end", "answer"]
    wide["_o"] = wide["readout_at"].map(order.index)
    return wide.sort_values(["_o", "group"]).drop(columns=["_o"], level=0)


def run(spec, config: dict, results_dir: Path) -> None:
    ks = config["readout"]["k_values"]
    for limit in (True, False):
        try:
            df = load(spec.alias, results_dir, limit=limit)
        except FileNotFoundError:
            continue
        label = "FLOOR RUN (--limit)" if limit else "FULL RUN"
        print(f"\n{'=' * 70}\n{label}: {len(df):,} rows\n{'=' * 70}")
        try:
            q1_diagnostic(load_summary(spec.alias, results_dir, limit=limit))
        except FileNotFoundError:
            pass
        floor_check(df)

        print(f"\nHEADLINE — present rate, pooled over C_t (k={PRIMARY_K}, J-lens)")
        print("  compare every row against `absent`, the chance floor")
        print(headline(df).to_string(index=False, float_format="%.3f"))

        print(f"\nCAPACITY — non-queried targets present (k={PRIMARY_K})")
        print(capacity(df).to_string(index=False))

        print(f"\nSELECTIVITY / EVICTION — present rate by group (k={PRIMARY_K})")
        print(selectivity(df).to_string(index=False))

        print(f"\nPER-LAYER — present rate by layer (k={PRIMARY_K}, J-lens)")
        print(by_layer(df).to_string(index=False, float_format="%.2f"))

        print(f"\nk-SWEEP — non-queried targets at in_stream")
        sweep = []
        for k in ks:
            c = capacity(df, k)
            c = c[(c["lens"] == "jlens") & (c["readout_at"] == "in_stream")]
            sweep.append(c.assign(k=k))
        print(pd.concat(sweep).pivot_table(
            index="k", columns="c_t", values="mean").to_string())
