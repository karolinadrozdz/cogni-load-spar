"""Capital-city stimuli: 50 real countries and 50 invented ones.

`stimuli/capitals.csv` with one row per item: `id`, `condition`
(`real` | `fictitious`), `entity`, `answer` (empty for a fictitious entity,
which has no capital to know).
"""

from __future__ import annotations

import csv
from pathlib import Path

OUT = Path(__file__).with_name("capitals.csv")

#: Every capital is one token after a space on the Qwen tokenizer, so the
#: answer is the next token after the prefill. Left out on purpose: countries
#: with a contested or multi-word capital, and those named like their capital.
REAL: list[tuple[str, str]] = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Germany", "Berlin"), ("Italy", "Rome"),
    ("Spain", "Madrid"), ("Russia", "Moscow"), ("China", "Beijing"), ("Egypt", "Cairo"),
    ("Greece", "Athens"), ("Austria", "Vienna"), ("Ireland", "Dublin"),
    ("Portugal", "Lisbon"), ("Norway", "Oslo"), ("Sweden", "Stockholm"),
    ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Poland", "Warsaw"),
    ("Hungary", "Budapest"), ("Belgium", "Brussels"), ("Turkey", "Ankara"),
    ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Syria", "Damascus"), ("Lebanon", "Beirut"),
    ("Kenya", "Nairobi"), ("Tunisia", "Tunis"), ("Senegal", "Dakar"), ("Peru", "Lima"),
    ("Chile", "Santiago"), ("Venezuela", "Caracas"), ("Cuba", "Havana"),
    ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Pakistan", "Islamabad"),
    ("Thailand", "Bangkok"), ("Indonesia", "Jakarta"), ("Philippines", "Manila"),
    ("Afghanistan", "Kabul"), ("Bulgaria", "Sofia"), ("Jamaica", "Kingston"),
    ("Britain", "London"), ("Netherlands", "Amsterdam"), ("Switzerland", "Bern"),
    ("Czechia", "Prague"), ("Scotland", "Edinburgh"), ("Wales", "Cardiff"),
    ("Guyana", "Georgetown"), ("New Zealand", "Wellington"),
    ("Saudi Arabia", "Riyadh"), ("South Korea", "Seoul"),
]

#: Invented, with the endings real country names have. No well-known fictional
#: country (Wakanda, Genovia): a model has an answer for those.
FICTITIOUS: list[str] = [
    "Vermelia", "Dravonia", "Koresta", "Zantoria", "Belmora", "Marvenia", "Quelonia",
    "Sorvania", "Halvaria", "Brenovia", "Nuvaria", "Pelandia", "Rastonia", "Ulmeria",
    "Jorvania", "Kelvaria", "Lothenia", "Fendoria", "Garvenia", "Torvalia", "Zembria",
    "Arvenia", "Dunmora", "Elvaria", "Friselia", "Ostrenia", "Calvoria", "Hestovia",
    "Ilmaria", "Javoria", "Krelovia", "Norvania", "Pravonia", "Rundavia", "Selmora",
    "Tarvonia", "Valdoria", "Wendaria", "Xantria", "Yelmora", "Zorvania", "Altovia",
    "Bremland", "Kovistan", "Tarsland", "Zurakstan", "Drenmark", "Velmark", "Hylstan",
    "Bakristan",
]


def items() -> list[dict]:
    rows = ([{"condition": "real", "entity": e, "answer": a} for e, a in REAL]
            + [{"condition": "fictitious", "entity": e, "answer": ""} for e in FICTITIOUS])
    entities = [r["entity"] for r in rows]
    assert len(REAL) == len(FICTITIOUS) == 50, (len(REAL), len(FICTITIOUS))
    assert len(set(entities)) == len(entities), "an entity is listed twice"
    assert len({a for _, a in REAL}) == len(REAL), "two countries share a capital"
    return [{"id": i, **r} for i, r in enumerate(rows)]


def main() -> None:
    rows = items()
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} items to {OUT}")


if __name__ == "__main__":
    main()
