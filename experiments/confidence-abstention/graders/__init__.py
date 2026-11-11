"""Post-hoc graders: re-score a finished run's outputs on CPU, each in its own image (TASKS.md: "Grading").

A grader module defines `image`, `options` (defaults for --opts), grade(out, params, opts) -> (rows, extra params),
and one Modal function, grade_<name>(tag, opts), declared with grader_function(image) and returning grade_run(...).
Keep the grader's heavy imports inside grade(): grading_modal.py imports every grader module to register its function.
"""

import importlib
import os
from pathlib import Path

import modal

RESULTS_VOLUME = os.environ.get("ABSTENTION_RESULTS_VOLUME", "abstention-results")

app = modal.App("confidence-grading")
results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

MODULES = {
    "exact_match": "graders.exact_match",
}


def module(name):
    return importlib.import_module(MODULES[name])


def grader_function(image, timeout=60 * 60, **kw):
    return app.function(image=image, volumes={"/results": results}, timeout=timeout, **kw)


def grade_run(name, tag, grade, opts):
    """grade() on results/<tag>/; writes graded_<name>.csv, then grade_params_<name>.json (last: marks it complete)."""
    import json

    import pandas as pd
    out = Path("/results") / tag
    params = json.loads((out / "params.json").read_text())
    rows, extra = grade(out, params, opts)
    pd.DataFrame(rows).to_csv(out / f"graded_{name}.csv", index=False)
    (out / f"grade_params_{name}.json").write_text(json.dumps(
        dict(tag=tag, grader=name, task=params.get("task", "confidence_abstention"), opts=opts, n=len(rows), **extra),
        indent=2, default=str))
    results.commit()
    return f"{tag}: {name} graded {len(rows)} rows -> {RESULTS_VOLUME}/{tag}/graded_{name}.csv"
