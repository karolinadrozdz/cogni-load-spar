"""Region-capital stimuli: real first-level regions, from famous to obscure.

`stimuli/regions.csv` with one row per item: `id`, `condition` (`region`),
`entity` ("Morelos, Mexico"), `answer` (the capital) and `links`, the number of
Wikipedia language editions with an article on the region, as a measure of how
well known it is.

Built from `regions_wikidata.csv`, a snapshot of the query below, which is
downloaded if the file is absent. Nothing here is written from memory.
"""

from __future__ import annotations

import csv
import urllib.parse
import urllib.request
from pathlib import Path

import yaml
from tokenizers import Tokenizer

from capitals import REAL

HERE = Path(__file__).parent
RAW, OUT = HERE / "regions_wikidata.csv", HERE / "regions.csv"
FIRST_ID = 100  # capitals.csv holds 0-99

#: Every region a sovereign state lists as a subdivision, with its capital.
#: `national` marks a capital that is also a state's capital.
QUERY = """
SELECT ?region ?regionLabel ?capitalLabel ?countryLabel ?links ?national WHERE {
  ?country wdt:P31 wd:Q3624078 ; wdt:P150 ?region .
  ?region wdt:P36 ?capital ; wikibase:sitelinks ?links .
  FILTER NOT EXISTS { ?region wdt:P576 ?end }
  BIND(EXISTS { ?state wdt:P31 wd:Q3624078 ; wdt:P36 ?capital } AS ?national)
  ?region rdfs:label ?regionLabel . FILTER(LANG(?regionLabel) = "en")
  ?capital rdfs:label ?capitalLabel . FILTER(LANG(?capitalLabel) = "en")
  ?country rdfs:label ?countryLabel . FILTER(LANG(?countryLabel) = "en")
}
"""
MIN_LINKS = 20  # below this the entries are mostly duplicates and stubs


def fetch() -> None:
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": QUERY})
    request = urllib.request.Request(url, headers={
        "Accept": "text/csv", "User-Agent": "cogni-load-spar/0.1 (research stimuli)"})
    with urllib.request.urlopen(request, timeout=180) as response:
        RAW.write_bytes(response.read())


def _plain(text: str) -> bool:
    return text.isascii() and text.replace(" ", "").replace("-", "").isalpha()


def items() -> list[dict]:
    """One item per capital: the best-known region it belongs to."""
    spec = yaml.safe_load((HERE.parent / "configs" / "models.yaml").read_text())["aliases"]["dev"]
    tokenizer = Tokenizer.from_pretrained(spec["hf_id"], revision=spec["hf_revision"])
    used = {name for pair in REAL for name in pair}

    with RAW.open(encoding="utf-8", newline="") as f:
        raw = list(csv.DictReader(f))
    capitals: dict[str, set[str]] = {}
    for r in raw:
        capitals.setdefault(r["region"], set()).add(r["capitalLabel"])

    best: dict[str, dict] = {}
    for r in raw:
        region, capital, country = r["regionLabel"], r["capitalLabel"], r["countryLabel"]
        keep = (
            len(capitals[r["region"]]) == 1 and r["national"] == "false"
            and int(r["links"]) >= MIN_LINKS
            and _plain(region) and _plain(country) and _plain(capital) and " " not in capital
            # The capital must not be readable off the question.
            and capital.lower() not in region.lower() and region.lower() not in capital.lower()
            and "City" not in region and not {region, capital} & used
            # One token after a space, so the answer is the next token after the prefill.
            and len(tokenizer.encode(f" {capital}", add_special_tokens=False).ids) == 1
        )
        if keep and int(r["links"]) > int(best.get(capital, {"links": 0})["links"]):
            best[capital] = r
    rows = sorted(best.values(), key=lambda r: (-int(r["links"]), r["regionLabel"]))
    return [{"id": FIRST_ID + i, "condition": "region",
             "entity": f"{r['regionLabel']}, {r['countryLabel']}",
             "answer": r["capitalLabel"], "links": int(r["links"])} for i, r in enumerate(rows)]


def main() -> None:
    if not RAW.exists():
        fetch()
    rows = items()
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} items to {OUT}")


if __name__ == "__main__":
    main()
