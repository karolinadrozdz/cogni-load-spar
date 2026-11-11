"""Worked example: re-grade a short_answer run's answers.csv against the aliases. `is_correct` is the task's own
exact match (it should reproduce the run's column exactly); `alias_in_answer` also accepts an answer that contains
an alias as a whole-word span ("Paris, France" for "Paris")."""

import modal

from graders import grade_run, grader_function

image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("pandas", "numpy")
         .add_local_python_source("tasks", "abstention_modal"))
options = {}


def grade(out, params, opts):
    import pandas as pd

    from tasks.short_answer import normalize_answer
    assert params.get("task") == "short_answer", "exact_match grades a short_answer run's answers.csv"
    a = pd.read_csv(out / "answers.csv", keep_default_na=False)  # an empty answer stays "", not NaN
    rows = []
    for r in a.itertuples():
        gold = {normalize_answer(x) for x in r.answer.split("|")}
        said = normalize_answer(r.generated_text)
        rows.append(dict(question_index=r.question_index, question=r.question, answer=r.answer,
                         generated_text=r.generated_text, is_correct=int(said in gold),
                         alias_in_answer=int(any(f" {g} " in f" {said} " for g in gold if g)),
                         is_correct_run=int(r.is_correct)))
    g = pd.DataFrame(rows)
    return rows, dict(accuracy=round(g.is_correct.mean(), 4), alias_in_answer=round(g.alias_in_answer.mean(), 4),
                      agreement_with_run=round((g.is_correct == g.is_correct_run).mean(), 4))


@grader_function(image)
def grade_exact_match(tag: str, opts: dict) -> str:
    return grade_run("exact_match", tag, grade, opts)
