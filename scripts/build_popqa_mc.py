"""Build data/popqa_phase{0,1}_questions.csv: PopQA as 4-way MCQ in the SimpleQA files' schema.

    python build_popqa_mc.py            # downloads akariasai/PopQA test.tsv (pinned revision) from the HF Hub

1,000 + 1,000 disjoint questions, 100 per subject-popularity (s_pop) decile in each file, rows interleaved by
decile so that any head(n) with n a multiple of 10 is stratified. choice_1 is the ground truth (Wikidata object
label), choice_2/3 are objects of the same relation for other subjects (plausible), choice_4 is an object of a
relation of another type (absurd), as in the SimpleQA set. Extra columns: s_pop, prop.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

SEED = 1042
PER_DECILE = 100  # per file
POPQA_REVISION = "098765c79ea10a2cb19c828324e33281b8336ec0"
# absurd foils come from a relation of another type
RELATION_TYPE = {"author": "person", "composer": "person", "director": "person", "father": "person",
                 "mother": "person", "producer": "person", "screenwriter": "person",
                 "capital": "place", "capital of": "place", "country": "place", "place of birth": "place",
                 "color": "category", "genre": "category", "occupation": "category", "religion": "category",
                 "sport": "category"}
# strings pandas.read_csv would turn into NaN
NA_STRINGS = {"", "na", "n/a", "nan", "null", "none", "#n/a", "-nan", "1.#ind", "1.#qnan", "<na>"}


def load():
    path = hf_hub_download("akariasai/PopQA", "test.tsv", repo_type="dataset", revision=POPQA_REVISION)
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    # multi-object questions appear once per object: keep one row, but every object's aliases count as correct
    answers = df.groupby("question").possible_answers.apply(
        lambda s: {a.strip().lower() for x in s for a in json.loads(x)})
    df = df.drop_duplicates("question").set_index("question", drop=False)
    df["answers"] = answers
    return df


def foils(row, df, by_prop, rng):
    """Two same-relation objects of other subjects (frequency-weighted, so common answers are common foils)
    and one object of another relation type; None if no valid set is found."""
    taken = set(row.answers) | {row.obj.lower(), row.subj.lower()}
    plausible = []
    for obj in rng.sample(by_prop[row.prop], min(len(by_prop[row.prop]), 200)):
        if obj.lower() not in taken and obj.lower() not in NA_STRINGS:
            plausible.append(obj)
            taken.add(obj.lower())
        if len(plausible) == 2:
            break
    other = df[df.prop.map(RELATION_TYPE) != RELATION_TYPE[row.prop]].obj
    absurd = next((o for o in rng.sample(list(other), 50) if o.lower() not in taken | NA_STRINGS), None)
    return None if len(plausible) < 2 or absurd is None else plausible + [absurd]


def main():
    rng = random.Random(SEED)
    df = load()
    df = df[~df.obj.str.lower().isin(NA_STRINGS)]
    by_prop = {p: list(g.obj) for p, g in df.groupby("prop")}
    df["decile"] = pd.qcut(df.s_pop.rank(method="first"), 10, labels=False)
    rows = {0: [], 1: []}
    for dec, g in df.groupby("decile"):
        picked = []
        for q in g.sample(frac=1, random_state=SEED + int(dec)).index:
            f = foils(df.loc[q], df, by_prop, rng)
            if f:
                r = df.loc[q]
                picked.append(dict(question=q, choice_1=r.obj, choice_2=f[0], choice_3=f[1], choice_4=f[2],
                                   s_pop=int(r.s_pop), prop=r.prop, decile=int(dec)))
            if len(picked) == 2 * PER_DECILE:
                break
        assert len(picked) == 2 * PER_DECILE, f"decile {dec}: only {len(picked)} items with valid foils"
        rows[0] += [dict(r, rank=i) for i, r in enumerate(picked[:PER_DECILE])]
        rows[1] += [dict(r, rank=i) for i, r in enumerate(picked[PER_DECILE:])]
    assert not {r["question"] for r in rows[0]} & {r["question"] for r in rows[1]}, "phase 0/1 sets must be disjoint"
    out = Path(__file__).parent / "data"
    for p in (0, 1):
        d = pd.DataFrame(rows[p]).sort_values(["rank", "decile"]).drop(columns=["rank", "decile"])
        d.to_csv(out / f"popqa_phase{p}_questions.csv", index=False)
        back = pd.read_csv(out / f"popqa_phase{p}_questions.csv")
        assert back[[f"choice_{k}" for k in range(1, 5)]].notna().all().all(), "an option would read back as NaN"
        assert (back[[f"choice_{k}" for k in range(1, 5)]].apply(lambda r: r.str.lower().nunique(), axis=1) == 4).all()
        print(f"phase {p}: {len(d)} questions, s_pop median {np.median(d.s_pop):.0f}, "
              f"relations {d.prop.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
