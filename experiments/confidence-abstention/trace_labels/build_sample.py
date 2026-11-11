"""Draw sets A-D from the 80b-thinking run and write judge batches (spec: TRACE-CLASSIFICATION-SPEC.md)."""
import json
import sys
from pathlib import Path

import pandas as pd

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "results/80b-thinking")
N_BATCHES = 12


def stratified(df, n, seed=0):
    return pd.concat([g.sample(n // 2, random_state=seed) for _, g in df.groupby("is_correct")])


def main():
    p = pd.read_csv(RUN / "phases.csv")
    v = pd.read_csv(RUN / "verbal.csv")
    p, v = p[p.truncated == 0], v[v.truncated == 0]
    p1 = pd.read_csv(RUN / "phases.csv").query("phase == 1").set_index("question_index")
    p2 = p[p.phase == 2]

    sets = {
        "A": p2[p2.is_nota == 1],
        "B": stratified(p2[p2.is_nota == 0], 200),
        "C": v[v.confidence_class == "No chance"],
        "D": stratified(v[v.confidence_class.isin(["Highly likely", "Almost certain"])], 100),
    }
    rows = []
    for s, df in sets.items():
        for _, r in df.iterrows():
            q = p1.loc[r.question_index]
            rows.append(dict(
                id=f"{s}{r.question_index}", set=s, question_index=int(r.question_index),
                phase=2 if s in "AB" else "verbal", is_correct=int(r.is_correct),
                is_nota=int(r.is_nota) if s in "AB" else None,
                confidence_class=r.confidence_class if s in "CD" else None,
                question=r.question, options={k: q[f"choice_{k}_arranged"] for k in (1, 2, 3, 4)},
                gt_position=int(r.gt_position),
                previous_answer=int(q.final_answer) if s in "CD" else None,
                final_answer=r.final_answer if s in "AB" else None,
                reasoning=r.reasoning,
            ))
    rows.sort(key=lambda r: r["id"])
    out = RUN / "trace_judging"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).drop(columns=["reasoning", "options"]).to_csv(out / "sample.csv", index=False)
    # interleave so every batch mixes sets and the judge can't infer set from position
    shuffled = pd.Series(range(len(rows))).sample(frac=1, random_state=0).tolist()
    for b in range(N_BATCHES):
        with open(out / f"batch_{b:02d}.jsonl", "w") as f:
            for i in shuffled[b::N_BATCHES]:
                r = rows[i]
                f.write(json.dumps({k: r[k] for k in ("id", "phase", "question", "options",
                                                      "previous_answer", "final_answer", "reasoning")}) + "\n")
    print({s: len(df) for s, df in sets.items()}, "total", len(rows))


if __name__ == "__main__":
    main()
