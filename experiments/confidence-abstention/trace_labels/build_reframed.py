"""Judge batches of 5 for the abstentions of the reframed Phase-2 runs (S = skip, N = neutral)."""
import json
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
BATCH_SIZE = 5
RUNS = {"S": "80b-thinking-p2skip", "N": "80b-thinking-p2neutral"}


def main():
    rows = []
    for s, tag in RUNS.items():
        p = pd.read_csv(RESULTS / tag / "phases.csv")
        p1 = p[p.phase == 1].set_index("question_index")
        a = p[(p.phase == 2) & (p.truncated == 0) & (p.is_nota == 1)]
        for _, r in a.iterrows():
            rows.append(dict(id=f"{s}{r.question_index}", set=s, question_index=int(r.question_index), phase=2,
                             is_correct=int(r.is_correct), is_nota=1, phase1_correct=int(p1.loc[r.question_index, "is_correct"]),
                             gt_position=int(r.gt_position), final_answer=5, question=r.question,
                             phase2_instruction=r.prompt.split("Remember,")[0].replace("You will be given a 4-way multiple choice question. ", "").strip(),
                             options={k: r[f"choice_{k}_arranged"] for k in (1, 2, 3, 4)}, reasoning=r.reasoning))
    out = RESULTS / "80b-thinking-reframed-judging" / "trace_judging"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop(columns=["reasoning", "options"]).to_csv(out / "sample.csv", index=False)
    order = pd.Series(range(len(rows))).sample(frac=1, random_state=0).tolist()
    for b in range(-(-len(rows) // BATCH_SIZE)):
        with open(out / f"batch_{b:03d}.jsonl", "w") as f:
            for i in order[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]:
                r = rows[i]
                f.write(json.dumps({k: r[k] for k in ("id", "phase", "phase2_instruction", "question", "options",
                                                      "final_answer", "reasoning")}) + "\n")
    print(pd.DataFrame(rows).set.value_counts().to_dict(), "total", len(rows))


if __name__ == "__main__":
    main()
