"""Post-hoc grading of a finished run, on CPU: reads results/<tag>/ in the volume, writes <tag>/graded_<grader>.csv and
<tag>/grade_params_<grader>.json (last, so its presence means the grading is complete).

    modal run --detach grading_modal.py::grade --tag sa-olmo32b-think-rl-main --grader exact_match
    modal run --detach grading_modal.py::grade --tag <tag> --grader <name> --opts "k=v,..."

Graders are files in graders/, registered in graders.MODULES, each with its own image (never the frozen vLLM one).
Writing a grader: TASKS.md, "Grading".
"""

from abstention_modal import parse_task_opts
from graders import MODULES, app, module

GRADERS = {name: module(name) for name in MODULES}  # importing a grader registers its Modal function on app


@app.local_entrypoint()
def grade(tag: str, grader: str = "exact_match", opts: str = ""):
    if grader not in GRADERS:
        raise SystemExit(f"--grader must be one of {sorted(GRADERS)}")
    g = GRADERS[grader]
    call = getattr(g, f"grade_{grader}").spawn(tag, parse_task_opts(opts, g.options))  # spawn, don't wait (see main)
    print(f"spawned grade {grader} {tag}: {call.object_id}")
