"""Task registry: --task <name> -> the module in this package that defines it (TASKS.md: "Adding a task").

A task module defines TASK (the task object) and the sweep hooks analyze_sweep.py reads: items, measures, gates, show.
Keep imports needed only at run time (vLLM, heavy scorers) inside functions: main() imports the module locally.
"""

import importlib

MODULES = {
    "short_answer": "tasks.short_answer",
}


def module(name):
    return importlib.import_module(MODULES[name])
