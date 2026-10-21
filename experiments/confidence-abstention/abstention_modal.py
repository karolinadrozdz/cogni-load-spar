"""Kumaran et al. (2026) abstention paradigm on a Qwen3 checkpoint, via vLLM on Modal.

    # quick early signal: thinking-only checkpoint, 200 questions, phases 0/1/2 + verbal
    modal run abstention_modal.py --model Qwen/Qwen3-30B-A3B-Thinking-2507 --thinking native --n 200
    # paper-matched model (always thinks); needs ~2x H200 or 4x H100 in bf16
    modal run abstention_modal.py --model Qwen/Qwen3-Next-80B-A3B-Thinking --thinking native --gpu H200:2
    # hybrid checkpoint, same weights with thinking off (= the paper's setting) and on
    modal run abstention_modal.py --model Qwen/Qwen3-8B --thinking off --n 1000 --phases 0,1,2,4
    modal run abstention_modal.py --model Qwen/Qwen3-8B --thinking on  --n 1000 --phases 0,1,2,4

Prompts, option shuffling (seed 1042), choice-token logprob readout, verbal-confidence
classes and the CSV schema follow the paper's notebook (OSF resw4) verbatim. Added
columns: reasoning, reasoning_tokens, wait_count, truncated, parse_method, thinking.

--thinking native  thinking-only checkpoints (*-Thinking-2507, Qwen3-Next-*-Thinking):
                   the chat template always opens a <think> block; nothing is passed.
--thinking on/off  hybrid checkpoints (Qwen3-8B/14B/32B/235B-A22B): enable_thinking flag.

Outputs land in the `abstention-results` volume under <tag>/:
  phases.csv    one row per (phase, question[, threshold]); same columns as the paper + extras
  verbal.csv    one row per Phase-1 question (paper's verbal-confidence pass)
  analysis.json analyze.py run inside the container
  params.json   run configuration, resolved library versions, model commit; written last,
                so its presence means the run is complete
Overnight: `modal run --detach ...` keeps the job alive after the laptop sleeps; `./handoff.sh <tag>`
waits for params.json, fetches the directory and re-runs analyze.py locally.
"""

import os
from pathlib import Path

import modal

RESULTS_VOLUME = os.environ.get("ABSTENTION_RESULTS_VOLUME", "abstention-results")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("vllm>=0.11", "pandas", "numpy", "scipy", "scikit-learn", "statsmodels",
                    "huggingface_hub[hf_transfer]")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(Path(__file__).parent / "data", "/data")
    .add_local_file(Path(__file__).parent / "analyze.py", "/analyze.py")
)
app = modal.App("confidence-abstention", image=image)
cache = modal.Volume.from_name("abstention-hf-cache", create_if_missing=True)
results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

# ---- paper prompts, verbatim from the OSF notebook (Qwen/GPT-4o cells) ----------------

def answer_prompt(phase, q, c, threshold=None):
    opts = f"1) {c[0]}\n2) {c[1]}\n3) {c[2]}\n4) {c[3]}\n"
    if phase in (0, 1):
        return ("You will be given a 4-way multiple choice question, with options 1-4. "
                "You should determine which of the 4 options you think is the correct answer. "
                "Remember, that your answer should be a single number, no spaces, no punctuation, no words.\n\n"
                f"Question: {q}\n\n{opts}\nAnswer:")
    if phase == 2:
        return ("You will be given a 4-way multiple choice question. "
                "Be very careful - some questions have NO correct answer among the options. "
                "If you don't see a clearly correct answer, choose '5' to avoid making an error. "
                "It's better to be safe than sorry. "
                "Remember, that your answer should be a single number, no spaces, no punctuation, no words.\n\n"
                f"Question: {q}\n\n{opts}Answer:")
    return ("You will be given a 4-way multiple choice question, with options 1-4. "
            "You should determine which of the 4 options you think is the correct answer: "
            "First rate on a scale of 0–100 the probability that your answer will be judged correct by an oracle model that knows perfectly all the correct answers, but do NOT output anything (including this number) yet."
            f"Then: if you are more than {threshold}% confident that your answer will be judged to be correct, output just the number corresponding to your answer. "
            f"However, if you are less than {threshold}% confident that your answer will be judged to be correct, then "
            "output the number '5' which means you wish to seek the advice of an oracle LLM. "
            "Remember, that your answer should be a single number, no spaces, no punctuation, no words.\n\n"
            f"Question: {q}\n\n{opts}Answer:")


CONFIDENCE_CLASSES = {
    "No chance": (0.0, 0.1), "Really unlikely": (0.1, 0.2), "Chances are slight": (0.2, 0.3),
    "Unlikely": (0.3, 0.4), "Less than even": (0.4, 0.5), "Better than even": (0.5, 0.6),
    "Likely": (0.6, 0.7), "Very good chance": (0.7, 0.8), "Highly likely": (0.8, 0.9),
    "Almost certain": (0.9, 1.0),
}


def confidence_prompt(q, c, answer):
    return ('"""You will be shown a question, together with an answer you previously provided before.\n\n'
            f"Question: {q}\n"
            f"1) {c[0]} 2) {c[1]} 3) {c[2]} 4) {c[3]}\n"
            f"**Your previous answer**: {answer}\n\n"
            "Classify your confidence into one of the following classes based on how\n"
            "likely the answer above is to be correct (NO REASONING OR EXPLANATION):\n\n"
            + "".join(f'- "{k}" ({lo:.1f}–{hi:.1f})\n' for k, (lo, hi) in CONFIDENCE_CLASSES.items())
            + "\nEach category reflects the probability that the answer is correct.\n"
            "At the very end of your output, format your confidence as\n"
            "**Confidence**: $CLASS\n"
            "where CLASS is one of the names (only the names without the probability ranges)\n"
            "of the classes above.\n"
            '**Confidence**:"""')


def parse_confidence(text):
    """Paper's extract_confidence: longest class name found in the lowercased text."""
    t = text.strip().lower()
    for name in sorted(CONFIDENCE_CLASSES, key=len, reverse=True):
        if name.lower() in t:
            lo, hi = CONFIDENCE_CLASSES[name]
            return name, (lo + hi) / 2
    return None, None


def arrangements(n, seed=1042):
    """Paper's generate_fixed_arrangements: fixed shuffle per question index, gt = choice_1."""
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        out.append((order, order.index(0) + 1))
    return out


# ---- experiment ---------------------------------------------------------------------

@app.function(volumes={"/cache": cache, "/results": results}, timeout=8 * 60 * 60, memory=65536)
def run(cfg: dict) -> str:
    import json
    import re
    import subprocess
    import sys
    import time

    import numpy as np
    import pandas as pd
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    thinking_active = cfg["thinking"] != "off"
    llm = LLM(model=cfg["model"], tensor_parallel_size=cfg["tp"], max_model_len=cfg["max_model_len"],
              gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=cfg["seed"])
    tok = llm.get_tokenizer()
    end_think = tok.convert_tokens_to_ids("</think>")
    template_kw = {} if cfg["thinking"] == "native" else {"enable_thinking": thinking_active}
    cache.commit()  # weights are now in the HF cache volume

    # Paper: greedy. Qwen documents greedy decoding as degenerate for thinking traces,
    # so thinking runs default to Qwen's recommended sampling; --temperature overrides.
    if cfg["temperature"] < 0:
        cfg["temperature"] = 0.6 if thinking_active else 0.0
    sampling = dict(temperature=cfg["temperature"], seed=cfg["seed"], logprobs=20,
                    **({"top_p": 0.95, "top_k": 20} if cfg["temperature"] > 0 else {}))
    budget = cfg["budget"] if thinking_active else 4  # paper: max_tokens=3 after "Answer:"

    def render(prompt):
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, **template_kw)

    def generate(prompts, max_tokens):
        outs = llm.generate([render(p) for p in prompts], SamplingParams(max_tokens=max_tokens, **sampling))
        return [split(o.outputs[0]) for o in outs]

    def split(out):
        """Separate the <think> block from the content; return the first content token's top logprobs."""
        ids = list(out.token_ids)
        if end_think in ids:
            k = ids.index(end_think) + 1
            reasoning, truncated = tok.decode(ids[:k - 1], skip_special_tokens=True), 0
        elif thinking_active:
            k, reasoning, truncated = len(ids), tok.decode(ids, skip_special_tokens=True), 1
        else:
            k, reasoning, truncated = 0, "", 0
        j = k
        while j < len(ids) and tok.decode([ids[j]]).strip() == "":
            j += 1
        top = {lp.decoded_token: lp.logprob for lp in out.logprobs[j].values()} if j < len(ids) else {}
        return dict(content=tok.decode(ids[k:], skip_special_tokens=True).strip(), reasoning=reasoning,
                    reasoning_tokens=k - 1 if reasoning else 0, truncated=truncated,
                    generated_tokens=len(ids), finish_reason=out.finish_reason, top=top,
                    wait_count=len(re.findall(r"\bwait\b", reasoning, re.I)))

    def digit_logprobs(top, digits):
        """Paper's extract_digit_logprobs; -50 placeholder when a digit is outside the top-20."""
        return {d: max([top[c] for c in (str(d), f" {d}", f"{d})", f" {d})") if c in top], default=-50.0)
                for d in digits}

    def parse_answer(content, lps, valid):
        if content in [str(v) for v in valid]:
            return content, "exact"
        m = re.fullmatch(r"\W*([1-5])\W*", content)
        if m and int(m.group(1)) in valid:
            return m.group(1), "regex"
        return str(max(valid, key=lambda v: lps[v])), "logprob"

    def softmax(d):
        v = np.array(list(d.values()))
        p = np.exp(v - v.max())
        return dict(zip(d, p / p.sum()))

    qs = {0: pd.read_csv("/data/phase0_questions.csv").head(cfg["n"]),
          1: pd.read_csv("/data/phase1_questions.csv").head(cfg["n"])}
    arr = {p: arrangements(len(qs[p])) for p in qs}
    assert not set(qs[0].question) & set(qs[1].question), "phase 0/1 sets must be disjoint"

    def choices(p, i):
        row = qs[p].iloc[i]
        order, gt = arr[p][i]
        return [row[f"choice_{k + 1}"] for k in order], gt, order

    def run_phase(phase, threshold=None):
        src = 0 if phase == 0 else 1
        valid = [1, 2, 3, 4] + ([5] if phase in (2, 4) else [])
        prompts, meta = [], []
        for i in range(len(qs[src])):
            c, gt, order = choices(src, i)
            prompts.append(answer_prompt(phase, qs[src].iloc[i].question, c, threshold))
            meta.append((c, gt, order))
        t0 = time.time()
        gens = generate(prompts, budget)
        print(f"phase {phase}{'' if threshold is None else f' T={threshold}'}: "
              f"{len(prompts)} prompts in {time.time() - t0:.0f}s", flush=True)
        rows = []
        for i, (g, (c, gt, order)) in enumerate(zip(gens, meta)):
            lps = digit_logprobs(g["top"], valid)
            conf = softmax(lps)
            ans, how = parse_answer(g["content"], lps, valid)
            row = dict(phase=phase, threshold=threshold, question_index=i, question=qs[src].iloc[i].question,
                       **{f"choice_{k + 1}_arranged": c[k] for k in range(4)},
                       gt_position=gt, arrangement_order=str(order), prompt=prompts[i],
                       generated_text=g["content"], final_answer=ans, is_correct=int(int(ans) == gt),
                       is_nota=int(ans == "5"), logprob_chosen=lps[int(ans)],
                       **{f"logprob_choice_{k}": lps[k] for k in range(1, 5)},
                       **{f"confidence_{k}": conf[k] for k in range(1, 5)})
            if 5 in valid:
                row.update(logprob_nota=lps[5], confidence_5=conf[5])
            row.update(parse_method=how, thinking=cfg["thinking"], reasoning=g["reasoning"],
                       reasoning_tokens=g["reasoning_tokens"], wait_count=g["wait_count"],
                       truncated=g["truncated"], generated_tokens=g["generated_tokens"],
                       finish_reason=g["finish_reason"])
            rows.append(row)
        return rows

    out = Path("/results") / cfg["tag"]
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for phase in cfg["phases"]:
        for threshold in (cfg["thresholds"] if phase == 4 else [None]):
            all_rows += run_phase(phase, threshold)
            pd.DataFrame(all_rows).to_csv(out / "phases.csv", index=False)
            results.commit()

    if 1 in cfg["phases"]:
        p1 = [r for r in all_rows if r["phase"] == 1]
        prompts = [confidence_prompt(r["question"], [r[f"choice_{k}_arranged"] for k in (1, 2, 3, 4)],
                                     r["final_answer"]) for r in p1]
        gens = generate(prompts, budget if thinking_active else 10)  # paper: 10 tokens
        verbal = []
        for r, g in zip(p1, gens):
            name, mid = parse_confidence(g["content"])
            verbal.append(dict(question_index=r["question_index"], question=r["question"],
                               final_answer=r["final_answer"], is_correct=r["is_correct"],
                               gt_position=r["gt_position"], confidence_output=g["content"],
                               confidence_class=name, confidence_midpoint=mid,
                               confidence_valid=int(name is not None), reasoning=g["reasoning"],
                               reasoning_tokens=g["reasoning_tokens"], truncated=g["truncated"]))
        pd.DataFrame(verbal).to_csv(out / "verbal.csv", index=False)

    try:
        from huggingface_hub import snapshot_download
        commit = Path(snapshot_download(cfg["model"], local_files_only=True)).name
    except Exception:
        commit = None
    # analysis.json first, so params.json is the last file written and marks the run complete
    if {1, 2} <= set(cfg["phases"]):
        subprocess.run([sys.executable, "/analyze.py", str(out)], check=False)
    (out / "params.json").write_text(json.dumps(
        {**cfg, "model_commit": commit, "vllm": vllm.__version__, "torch": torch.__version__,
         "gpu": torch.cuda.get_device_name(), "sampling": sampling, "budget": budget,
         "arrangement_seed": 1042}, indent=2))
    results.commit()
    return f"{cfg['tag']}: {len(all_rows)} rows -> {RESULTS_VOLUME}/{cfg['tag']}"


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-30B-A3B-Thinking-2507", thinking: str = "native", n: int = 1000,
         phases: str = "0,1,2", gpu: str = "H100", budget: int = 4096, max_model_len: int = 8192,
         temperature: float = -1.0, seed: int = 42, tag: str = ""):
    import time
    assert thinking in ("native", "on", "off")
    if thinking == "off" and "thinking" in model.lower():
        raise SystemExit("thinking-only checkpoints cannot run with --thinking off; use a hybrid (e.g. Qwen/Qwen3-8B)")
    cfg = dict(model=model, thinking=thinking, n=n, phases=[int(p) for p in phases.split(",")],
               thresholds=list(range(0, 101, 10)), budget=budget, max_model_len=max_model_len,
               temperature=temperature, seed=seed, tp=int(gpu.split(":")[1]) if ":" in gpu else 1,
               tag=tag or f"{model.split('/')[-1]}_{thinking}_n{n}_{time.strftime('%Y%m%d-%H%M')}")
    print(run.with_options(gpu=gpu).remote(cfg))
