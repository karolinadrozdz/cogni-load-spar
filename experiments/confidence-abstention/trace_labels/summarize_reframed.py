"""Merge judge passes + tiebreaks for the reframed Phase-2 abstentions into trace_labels_reframed.csv and its summary."""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

import summarize

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/80b-thinking-reframed-judging")
summarize.J = OUT / "trace_judging"
N_ITEMS = 1000  # per run; counts below are per 1,000 items


def main():
    df, a, b = summarize.merge_labels(pd.read_csv(summarize.J / "sample.csv").set_index("id"))
    df.to_csv(OUT / "trace_labels_reframed.csv")
    out = {
        "cohen_kappa_kind": round(cohen_kappa_score(a.kind, b.kind), 3),
        "raw_agreement_kind": round((a.kind == b.kind).mean(), 3),
        "n": int(len(df)), "n_disagreed": int(df.disagreed.sum()),
        "abstentions_per_1000_items": {s: g.kind.value_counts().reindex(summarize.KINDS, fill_value=0).to_dict()
                                       for s, g in df.groupby("set")},
        "P(phase1_wrong | abstained, kind)": {s: (1 - g.groupby("kind").phase1_correct.mean()).round(3).to_dict()
                                              for s, g in df.groupby("set")},
    }
    assert all(len(g) <= N_ITEMS for _, g in df.groupby("set"))
    (OUT / "trace_labels_reframed_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
