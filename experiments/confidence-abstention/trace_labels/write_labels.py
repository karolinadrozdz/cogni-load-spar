"""Write the judge workflow's result (trace_labels/judge_workflow.js) into the label files summarize.py reads.

    python trace_labels/write_labels.py <workflow output .json> <batch dir>
"""
import json
import sys
from pathlib import Path

import pandas as pd


def main(result, out):
    res = json.loads(Path(result).read_text())
    res = res.get("result", res)  # the Workflow tool's output file wraps the script's return value
    a = {l["id"]: l for x in res for l in x["a"]}
    b = {l["id"]: l for x in res for l in x["bb"]}
    tie = {l["id"]: l for x in res for l in x["tie"]}
    ids = sorted(pd.read_csv(Path(out) / "sample.csv").id)
    assert set(a) == set(b) == set(ids), f"labels for {len(a)} / {len(b)} of {len(ids)} ids"
    disagreed = [i for i in ids if a[i]["kind"] != b[i]["kind"]]
    assert set(disagreed) <= set(tie), f"{len(set(disagreed) - set(tie))} disagreements lack a tiebreak"
    for name, labels, keep in (("labels_a", a, ids), ("labels_b", b, ids), ("labels_tiebreak", tie, disagreed)):
        (Path(out) / f"{name}_000.jsonl").write_text("".join(json.dumps(labels[i]) + "\n" for i in keep))
    print(f"{len(ids)} traces, {len(disagreed)} tiebreaks -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
