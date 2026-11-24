"""Re-parse the post-think answer text of finished confidence_abstention runs with the current `parse_answer`.

    python reparse.py            # every results/<tag>/phases.csv; rewrites in place, reruns analyze.py where changed

Only rows the log-prob fallback parsed can change (asserted). Truncated rows have no post-think text and are left
as they are. Runs where at least one row changed get "reparsed": true in params.json.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abstention_modal import parse_answer

R = Path(__file__).parent / "results"


def valid_digits(phase):
    return [1, 2, 3, 4] + ([5] if phase in (2, 4) else [])


def reparse_run(d):
    df = pd.read_csv(d / "phases.csv", dtype={"generated_text": str}, keep_default_na=False, na_values=[""],
                     float_precision="round_trip")
    text = df.generated_text.fillna("").str.replace(r"^(\d+)\.0$", r"\1", regex=True)  # all-numeric runs were saved as floats
    old_answer, old_method = df.final_answer.astype(int).astype(str), df.parse_method.copy()
    new = []
    for i, r in df.iterrows():
        if r.truncated:
            new.append((old_answer[i], old_method[i]))
            continue
        valid = valid_digits(r.phase)
        lps = {k: r[f"logprob_choice_{k}"] for k in range(1, 5)} | ({5: r.logprob_nota} if 5 in valid else {})
        new.append(parse_answer(text[i], lps, valid))
    answer = pd.Series([a for a, _ in new], index=df.index)
    method = pd.Series([m for _, m in new], index=df.index)
    kept = old_method.ne("logprob")
    assert (answer[kept] == old_answer[kept]).all() and (method[kept] == old_method[kept]).all(), d.name
    still = old_method.eq("logprob") & method.eq("logprob")
    assert (answer[still] == old_answer[still]).all(), d.name
    moved = old_method.eq("logprob") & method.eq("prose")
    changed = answer != old_answer
    if not moved.any():
        return 0, 0
    df["final_answer"] = answer.astype(int)
    df["is_correct"] = (df.final_answer == df.gt_position).astype(int)
    df["is_nota"] = (df.final_answer == 5).astype(int)
    df["parse_method"] = method
    lp = df[[f"logprob_choice_{k}" for k in range(1, 5)] + ["logprob_nota"] * ("logprob_nota" in df)].to_numpy()
    df["logprob_chosen"] = lp[np.arange(len(df)), df.final_answer - 1]
    if "k_answers" in df:  # the first sample is the row's own answer; the others were not stored as text
        rows = df.k_answers.notna() & changed
        ks = df.loc[rows, "k_answers"].astype(str).str.split("|")
        ks = [[a] + k[1:] for a, k in zip(answer[rows], ks)]
        df.loc[rows, "k_answers"] = ["|".join(k) for k in ks]
        df.loc[rows, "consistency"] = [max(map(k.count, k)) / len(k) for k in ks]
    df.to_csv(d / "phases.csv", index=False)
    params = json.loads((d / "params.json").read_text())
    (d / "params.json").write_text(json.dumps({**params, "reparsed": True}, indent=2))
    subprocess.run([sys.executable, str(Path(__file__).parent / "analyze.py"), str(d)], check=True,
                   stdout=subprocess.DEVNULL)
    return int(moved.sum()), int(changed.sum())


def main():
    for d in sorted(R.iterdir()):
        if not (d / "phases.csv").exists() or not (d / "params.json").exists():
            continue
        params = json.loads((d / "params.json").read_text())
        if params.get("thinking") == "base" or params.get("task", "confidence_abstention") != "confidence_abstention":
            continue
        moved, changed = reparse_run(d)
        if moved:
            print(f"{d.name}: logprob -> prose {moved}, final_answer changed {changed}")


if __name__ == "__main__":
    main()
