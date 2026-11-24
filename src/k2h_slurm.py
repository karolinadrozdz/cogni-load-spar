"""Run the existing harness on Slurm without importing or connecting to Modal.

Only Modal provisioning statements and entrypoint decorators are omitted by the
loader. Experiment functions, configuration, prompts and sampling are unchanged.
"""
import argparse
import ast
import os
from pathlib import Path
from types import SimpleNamespace


def load_harness():
    source = Path(__file__).with_name("abstention_modal.py")
    tree = ast.parse(source.read_text(), filename=str(source))
    kept = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            break  # run() precedes the Modal local entrypoint; probes follow it.
        if isinstance(node, ast.Import) and any(a.name == "modal" for a in node.names):
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in {"image", "app", "cache", "results", "hf_secret"}
            for t in node.targets
        ):
            continue
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
        kept.append(node)
    tree.body = kept
    ns = {"__file__": str(source), "cache": SimpleNamespace(commit=lambda: None),
          "results": SimpleNamespace(commit=lambda: None)}
    exec(compile(tree, str(source), "exec"), ns)
    return ns


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--revision", default="main")
    p.add_argument("--thinking", choices=["native", "base"], default="native")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--budget", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=20480)
    p.add_argument("--temperature", type=float, default=-1.0)
    p.add_argument("--k-samples", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    assert a.tag.startswith("k2h32b-popqa-")
    root = Path(__file__).resolve().parent
    os.environ.setdefault("ABSTENTION_DATA_DIR", str(root / "data"))
    os.environ.setdefault("ABSTENTION_ANALYZE", str(root / "analyze.py"))
    os.environ.setdefault("ABSTENTION_RESULTS_DIR", str(root / "results"))
    h = load_harness()
    flags = dict(vars(a), model="IFM/K2-Horizon-32B", phases="0,1,2", seed=42,
                 phase2_prompt="paper", dtype="auto", questions="popqa", base_phase2_verbal=False)
    print(h["run"](h["TASKS"]["confidence_abstention"].config(flags)), flush=True)
