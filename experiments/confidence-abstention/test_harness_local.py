"""Local CPU checks of the harness (no vLLM, no GPU, no Modal calls).

    python test_harness_local.py             # equivalence: confidence_abstention and short_answer rows vs REF
    python test_harness_local.py render      # Context.render vs rendered_example in every results/*/params.json

equivalence runs ConfidenceAbstention.run and ShortAnswer.run of this tree and of abstention_modal.py at the git
revision REF (before the MC helpers and tasks/ moved) on a fake vLLM engine with deterministic outputs, in every
--thinking mode, and asserts byte-identical CSVs, params and analysis. render rebuilds each run's example prompt
with the MC helpers and renders it with the run's tokenizer (fetched from the HF Hub).
"""

import hashlib
import importlib.util
import io
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
from transformers import AutoTokenizer

HERE = Path(__file__).parent
REF = "185649c"
TOKENIZER = "Qwen/Qwen3-0.6B"

# Context.generate imports SamplingParams from vllm
sys.modules.setdefault("vllm", ModuleType("vllm")).SamplingParams = lambda **kw: SimpleNamespace(**kw)
read_csv = pd.read_csv
pd.read_csv = lambda path, *a, **k: read_csv(str(path).replace("/data/", f"{HERE}/data/", 1), *a, **k)

import abstention_modal as am  # noqa: E402
from tasks import short_answer  # noqa: E402

CONTENTS = ["3", " 2", "The answer is 4.", "**1**", "I'm not sure.", "5", "option (2)", "Paris", "Likely",
            "**Confidence**: Highly likely", "the Eiffel Tower"]


def load_ref():
    src = subprocess.run(["git", "show", f"{REF}:experiments/confidence-abstention/abstention_modal.py"],
                         cwd=HERE, check=True, capture_output=True, text=True).stdout
    path = Path(tempfile.mkdtemp()) / "abstention_modal_ref.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("abstention_modal_ref", path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    return ref


class FakeLLM:
    """Deterministic stand-in for vllm.LLM: the output of a prompt depends only on its text, sample and seed."""

    def __init__(self, tok, thinking):
        self.tok, self.thinking = tok, thinking

    def generate(self, prompts, sp):
        if isinstance(prompts[0], dict):  # continuation_logprobs
            return [SimpleNamespace(prompt_logprobs=[None] + [{t: SimpleNamespace(logprob=-self.rng(p, j).random() * 5)}
                                                             for j, t in enumerate(p["prompt_token_ids"][1:], 1)])
                    for p in prompts]
        return [SimpleNamespace(outputs=[self.sample(p, s, sp) for s in range(getattr(sp, "n", 1))]) for p in prompts]

    def rng(self, *key):
        return random.Random(hashlib.sha256(repr(key).encode()).hexdigest())

    def sample(self, prompt, s, sp):
        r = self.rng(prompt, s, getattr(sp, "seed", None))
        content, finish = r.choice(CONTENTS), "stop"
        text = content
        if self.thinking in ("native", "on"):
            trace = r.choice(["Let me think. Wait, the options.", "Hmm, actually no, option 3 fits.", ""])
            if r.random() < 0.15:
                text, finish = trace + " still thinking", "length"
            else:
                text = trace + "</think>\n\n" + content
        ids = self.tok(text, add_special_tokens=False).input_ids
        digits = ["1", "2", "3", "4", "5", " 3", "1)"]
        logprobs = []
        for j, t in enumerate(ids):
            top = {t: SimpleNamespace(logprob=-r.random(), decoded_token=self.tok.decode([t]))}
            for d in r.sample(digits, 4):
                did = self.tok(d, add_special_tokens=False).input_ids[0]
                top.setdefault(did, SimpleNamespace(logprob=-r.random() * 8, decoded_token=self.tok.decode([did])))
            logprobs.append(top)
        return SimpleNamespace(token_ids=ids, finish_reason=finish, logprobs=logprobs)


def run_task(mod, task, cfg, tok):
    mod.results = SimpleNamespace(commit=lambda: None)  # Context.save commits the Modal volume
    out = Path(tempfile.mkdtemp()) / cfg["tag"]  # same name in both runs: analysis.json records it
    out.mkdir()
    ctx = mod.Context(dict(cfg), FakeLLM(tok, cfg["thinking"]), tok, out)
    files, params = task.run(ctx)
    csvs = {name: pd.DataFrame(rows).to_csv(index=False) for name, rows in files.items()}
    for name, rows in files.items():
        pd.DataFrame(rows).to_csv(out / name, index=False)
    report = task.analyze(cfg, out) if cfg.get("task") == "short_answer" else None
    return csvs, params, report


def test_equivalence():
    ref, tok = load_ref(), AutoTokenizer.from_pretrained(TOKENIZER)
    common = dict(n=20, budget=64, max_model_len=4096, temperature=0.6, seed=42, tp=1, revision=None, dtype="auto",
                  tag="test", model=TOKENIZER)
    for thinking in ("base", "off", "on", "native"):
        cfg = dict(common, thinking=thinking, phases=[0, 1] if thinking == "base" else [0, 1, 2, 4],
                   thresholds=list(range(0, 101, 10)), phase2_prompt="paper", questions="popqa",
                   k_samples=3 if thinking == "on" else 1, verbal=thinking != "base")
        old = run_task(ref, ref.ConfidenceAbstention(), cfg, tok)
        new = run_task(am, am.ConfidenceAbstention(), cfg, tok)
        assert old == new, f"confidence_abstention --thinking {thinking} differs"
        sizes = {k: len(read_csv(io.StringIO(v))) for k, v in new[0].items()}
        print(f"confidence_abstention {thinking}: identical ({sizes})")
        cfg = dict(common, task="short_answer", thinking=thinking, shots=4)
        old = run_task(ref, ref.ShortAnswer(), cfg, tok)
        new = run_task(am, short_answer.TASK, cfg, tok)
        assert old == new, f"short_answer --thinking {thinking} differs"
        print(f"short_answer {thinking}: identical ({new[2]})")


def test_prefix_and_messages():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ctx = am.Context(dict(thinking="native", temperature=0.6, seed=0), None, tok, None)
    p = am.answer_prompt(1, "Q?", ["a", "b", "c", "d"])
    assert ctx.render(p, "") == ctx.render(p) == ctx.render([{"role": "user", "content": p}])
    assert ctx.render(p, "<think>\n\n</think>\n\n") == ctx.render(p) + "<think>\n\n</think>\n\n"
    assert "Be brief." in ctx.render([{"role": "system", "content": "Be brief."}, {"role": "user", "content": p}])
    ids = tok("3 because it fits", add_special_tokens=False).input_ids
    out = SimpleNamespace(token_ids=ids, finish_reason="length",
                          logprobs=[{t: SimpleNamespace(logprob=-0.1, decoded_token=tok.decode([t]))} for t in ids])
    assert ctx.split(out)["truncated"] == 1  # an open think block that hit the budget
    s = ctx.split(out, closed=True)
    assert (s["content"], s["reasoning"], s["truncated"], s["top"]) == ("3 because it fits", "", 0, {"3": -0.1})
    print("prefix / messages / closed split: ok")


def test_render():
    """Every run's rendered_example, rebuilt from the data files and rendered by this tree's Context."""
    checked, toks = {}, {}
    for f in sorted(HERE.glob("results/*/params.json")):
        p = json.loads(f.read_text())
        if "rendered_example" not in p:
            continue
        if p.get("task", "confidence_abstention") != "confidence_abstention":
            raise SystemExit(f"{f}: add a rebuild of the {p['task']} example prompt")
        data, arr = {}, {}
        for k in (0, 1):
            data[k], arr[k] = am.mc_items("/data/" + am.QUESTION_FILES[p.get("questions", "simpleqa")].format(k))
        shots = ""
        if p["thinking"] == "base":
            for i in am.base_exemplars(arr[0]):
                c, gt, _ = am.arranged_choices(data[0], arr[0], i)
                shots += am.mc_shot(data[0].question[i], c, gt)
        prompt = shots + am.answer_prompt(1, data[1].question[0], am.arranged_choices(data[1], arr[1], 0)[0])
        key = (p["model"], p.get("revision"), p["thinking"])
        if p["thinking"] != "base" and key[:2] not in toks:
            toks[key[:2]] = AutoTokenizer.from_pretrained(p["model"], revision=p.get("revision") or None)
        ctx = am.Context(dict(thinking=p["thinking"], temperature=0, seed=0), None, toks.get(key[:2]), None)
        assert ctx.render(prompt) == p["rendered_example"], f"{f.parent.name}: render differs"
        checked.setdefault(key, []).append(f.parent.name)
    for key, runs in checked.items():
        print(f"{key}: {len(runs)} runs identical")
    print(f"{sum(map(len, checked.values()))} runs, {len(checked)} (model, revision, thinking) combinations")


if __name__ == "__main__":
    if sys.argv[1:] == ["render"]:
        test_render()
    else:
        test_prefix_and_messages()
        test_equivalence()
