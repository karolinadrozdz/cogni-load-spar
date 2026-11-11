"""Write judge batches of Phase 2 abstentions (non-truncated) for one run, small batches so each judge sees few traces.
--answered C,W instead draws C correct and W wrong answered traces (set B; fewer if the run has fewer).

    python trace_labels/build_batches.py results/<tag> <out_dir> [--ids ids.txt] [--answered 100,100] [--batch-size 10]
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ids", type=Path, help="question_index per line; default all abstentions")
    ap.add_argument("--answered", help="C,W: correct and wrong answered traces instead of abstentions")
    ap.add_argument("--batch-size", type=int, default=10)
    a = ap.parse_args()
    p = pd.read_csv(a.run / "phases.csv")
    p1 = p[p.phase == 1].set_index("question_index")
    p2 = p[(p.phase == 2) & (p.truncated == 0) & (p.is_nota == int(not a.answered))]
    if a.answered:
        n = dict(zip((1, 0), map(int, a.answered.split(","))))
        p2 = pd.concat([g.sample(min(n[k], len(g)), random_state=0) for k, g in p2.groupby("is_correct")])
    s = "B" if a.answered else "A"
    if a.ids:
        p2 = p2[p2.question_index.isin([int(l) for l in a.ids.read_text().split()])]
    rows = [dict(id=f"{s}{int(r.question_index)}", set=s, question_index=int(r.question_index), phase=2,
                 is_correct=int(r.is_correct), is_nota=int(r.is_nota), gt_position=int(r.gt_position),
                 final_answer=int(r.final_answer), phase1_correct=int(p1.loc[r.question_index, "is_correct"]),
                 question=r.question, options={k: r[f"choice_{k}_arranged"] for k in (1, 2, 3, 4)},
                 reasoning=r.reasoning) for _, r in p2.iterrows()]
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop(columns=["reasoning", "options"]).to_csv(a.out / "sample.csv", index=False)
    order = pd.Series(range(len(rows))).sample(frac=1, random_state=0).tolist()  # batch position carries no information
    nb = -(-len(rows) // a.batch_size)
    for b in range(nb):
        with open(a.out / f"batch_{b:03d}.jsonl", "w") as f:
            for i in order[b * a.batch_size:(b + 1) * a.batch_size]:
                f.write(json.dumps({k: rows[i][k] for k in ("id", "phase", "question", "options", "final_answer",
                                                            "reasoning")}) + "\n")
    print(f"{len(rows)} traces ({pd.Series([r['is_correct'] for r in rows]).value_counts().to_dict()}) -> {nb} batches in {a.out}")


if __name__ == "__main__":
    main()
