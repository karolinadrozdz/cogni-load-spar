"""Kumaran et al. (2026) abstention paradigm on a Qwen3 or Olmo 3 checkpoint, via vLLM on Modal.

    # quick early signal: thinking-only checkpoint, 200 questions, phases 0/1/2 + verbal
    modal run --detach abstention_modal.py::main --model Qwen/Qwen3-30B-A3B-Thinking-2507 --thinking native --n 200
    # paper-matched model (always thinks); needs ~2x H200 or 4x H100 in bf16
    modal run --detach abstention_modal.py::main --model Qwen/Qwen3-Next-80B-A3B-Thinking --thinking native --gpu H200:2
    # hybrid checkpoint, same weights with thinking off (= the paper's setting) and on
    modal run --detach abstention_modal.py::main --model Qwen/Qwen3-8B --thinking off --n 1000 --phases 0,1,2,4
    modal run --detach abstention_modal.py::main --model Qwen/Qwen3-8B --thinking on  --n 1000 --phases 0,1,2,4

Prompts, option shuffling (seed 1042), choice-token logprob readout, verbal-confidence
classes and the CSV schema follow the paper's notebook (OSF resw4) verbatim. Added
columns: reasoning, reasoning_tokens, wait_count, truncated, parse_method, thinking, digit_mass
(probability on the valid digits at the readout token); base mode adds cloze_sum_k / cloze_mean_k.

--thinking native  thinking-only checkpoints (*-Thinking-2507, Qwen3-Next-*-Thinking):
                   the chat template always opens a <think> block; nothing is passed.
--thinking on/off  hybrid checkpoints (Qwen3-8B/14B/32B/235B-A22B): enable_thinking flag.
                   Olmo Instruct checkpoints run with off (their template ignores the flag).
--thinking base    no chat template (base / pretraining checkpoints): plain completion of the Phase 0/1
                   prompt after four Phase 0 exemplars, digit-logprob readout plus a cloze readout
                   (log-likelihood of each option text). Phases 0/1 only unless --base-phase2-verbal.
--revision         HF branch, e.g. stage1-step656000 or step_0500 (default: main).
--questions        simpleqa (data/phase{0,1}_questions.csv) or popqa (data/popqa_phase{0,1}_questions.csv).
--k-samples N      N Phase 1 answers per question (the first is the usual one); adds k_answers, consistency.
--dtype            vLLM dtype (default auto); bfloat16 for checkpoints stored in float32 (Olmo-3.1-32B-Instruct-SFT/-DPO).
--task             confidence_abstention (default; everything above) or another task in TASKS, e.g. short_answer;
                   --task-opts "k=v,..." sets that task's options. Writing a new task: TASKS.md.

Outputs (confidence_abstention; other tasks write their own CSVs) land in the `abstention-results` volume under <tag>/:
  phases.csv    one row per (phase, question[, threshold]); same columns as the paper + extras
  verbal.csv    one row per Phase-1 question (paper's verbal-confidence pass)
  analysis.json analyze.py run inside the container
  params.json   run configuration, resolved library versions, model commit; written last,
                so its presence means the run is complete
In-trace confidence checkpoints on a finished thinking run (reads <tag>/phases.csv, writes
checkpoints.csv + checkpoints_params.json; see README):
    modal run --detach abstention_modal.py::checkpoints --tag <tag> --gpu H200:2
Entrypoints main, checkpoints, start_format and prefetch: Modal needs the explicit ::<name> form.
Overnight: `modal run --detach abstention_modal.py::main ...` keeps the job alive after the laptop sleeps; `./handoff.sh <tag>`
waits for params.json, fetches the directory and re-runs analyze.py locally.
"""

import math
import os
import re
import time
from bisect import bisect_left, bisect_right
from pathlib import Path

import modal

RESULTS_VOLUME = os.environ.get("ABSTENTION_RESULTS_VOLUME", "abstention-results")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    # do not edit: any change rebuilds the cached image (vllm 0.32.0), and a rebuild pulls a nvidia-cutlass-dsl that fails to load
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
# HF_TOKEN for authenticated (faster, unthrottled) Hub downloads: `modal secret create huggingface HF_TOKEN=...`
hf_secret = modal.Secret.from_name("huggingface")

# ---- paper prompts, verbatim from the OSF notebook (Qwen/GPT-4o cells) ----------------

# Phase 2 reframings (not in the paper): `skip` removes the none-of-these route, `neutral` removes the caution pressure
PHASE2_INSTRUCTIONS = {
    "paper": ("Be very careful - some questions have NO correct answer among the options. "
              "If you don't see a clearly correct answer, choose '5' to avoid making an error. "
              "It's better to be safe than sorry. "),
    "skip": ("Exactly one of the options is correct. A wrong answer counts against you; skipping does not. "
             "If you are not sure of the answer, choose '5' to skip the question. "),
    "neutral": ("Some questions may have no correct answer among the options. "
                "If none of the options is correct, choose '5'. "),
}


def answer_prompt(phase, q, c, threshold=None, phase2_prompt="paper"):
    opts = f"1) {c[0]}\n2) {c[1]}\n3) {c[2]}\n4) {c[3]}\n"
    if phase in (0, 1):
        return ("You will be given a 4-way multiple choice question, with options 1-4. "
                "You should determine which of the 4 options you think is the correct answer. "
                "Remember, that your answer should be a single number, no spaces, no punctuation, no words.\n\n"
                f"Question: {q}\n\n{opts}\nAnswer:")
    if phase == 2:
        return ("You will be given a 4-way multiple choice question. "
                + PHASE2_INSTRUCTIONS[phase2_prompt] +
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


def digit_logprobs(top, digits):
    """Paper's extract_digit_logprobs; -50 placeholder when a digit is outside the top-20."""
    return {d: max([top[c] for c in (str(d), f" {d}", f"{d})", f" {d})") if c in top], default=-50.0)
            for d in digits}


# answers stated in prose after the trace, in priority order; "option N" / "(N)" come last because the prose also
# enumerates the options ("Option 4 (Tom Waits) is a musician ...")
PROSE_ANSWER_RES = [
    re.compile(r"(?:^|\n)\W*([1-5])\W*\Z"),  # explanation, then the digit alone on the last line
    re.compile(r"(?:answer|option|choice|response)\**\s*(?:is|should be)[\s:*'\"]*(?:option\s*)?\(?([1-5])\b"
               r"|answer\**\s*:[\s*'\"]*(?:option\s*)?\(?([1-5])\b|\*\*([1-5])\*\*", re.I),
    re.compile(r"(?:\bI\s+(?:will\s+)?|\bis\s+to\s+)(?:choose|select)\s*(?:option\s*)?['\"]?([1-5])\b", re.I),
    re.compile(r"option\s*\(?([1-5])\b|\(([1-5])\)", re.I),
]


def parse_answer(content, lps, valid):
    if content in [str(v) for v in valid]:
        return content, "exact"
    m = re.fullmatch(r"\W*([1-5])\W*", content)
    if m and int(m.group(1)) in valid:
        return m.group(1), "regex"
    for pattern in PROSE_ANSWER_RES:
        prose = [d for m in pattern.finditer(content) for d in m.groups() if d and int(d) in valid]
        if prose:
            return prose[-1], "prose"
    return str(max(valid, key=lambda v: lps[v])), "logprob"


def softmax(d):
    import numpy as np
    v = np.array(list(d.values()))
    p = np.exp(v - v.max())
    return dict(zip(d, p / p.sum()))


QUESTION_FILES = {"simpleqa": "phase{}_questions.csv", "popqa": "popqa_phase{}_questions.csv"}


def split_think(ids, tok, thinking_active, finish_reason):
    """Split generated ids at the first "</think>" of the decoded text. Matches the string, not a token id,
    because Olmo's </think> is three ordinary tokens that merge with their neighbours (".</", ">\\n\\n").
    Returns k (first token after </think>), reasoning, content, reasoning_tokens, truncated."""
    decode = lambda m: tok.decode(ids[:m], skip_special_tokens=True)
    text = decode(len(ids))
    at = text.find("</think>")
    if at >= 0:
        upto = range(len(ids) + 1)
        k = bisect_left(upto, at + len("</think>"), key=lambda m: len(decode(m)))
        before = bisect_right(upto, at, key=lambda m: len(decode(m))) - 1  # tokens wholly before </think>
        return k, text[:at], text[at + len("</think>"):].strip(), before, 0
    if thinking_active and finish_reason == "length":
        return len(ids), text, "", len(ids) - 1 if text else 0, 1
    return 0, "", text.strip(), 0, 0  # thinking off, or a non-thinking checkpoint run with --thinking native


# ---- --thinking base: plain completion with exemplars, for checkpoints without a chat template --------

SHOT_POSITIONS = (3, 1, 4, 2)  # gt positions of the exemplars in prompt order: each position once


def base_exemplars(arr0):
    """Phase 0 indices of the exemplars: the last item whose arranged answer is at each of SHOT_POSITIONS.
    None answers 5, so no abstention rate is taught."""
    last = {gt: i for i, (_, gt) in enumerate(arr0)}
    return [last[g] for g in SHOT_POSITIONS]


def mc_shot(q, c, gt):
    return answer_prompt(1, q, c) + f" {gt}\n\n"


def cloze_context(q):
    return f"Question: {q}\nAnswer:"


def cloze_shot(q, answer):
    return cloze_context(q) + f" {answer}\n\n"


# ---- in-trace checkpoints (pure helpers; used by `probe`) -----------------------------------

MARKER_RE = (r"\b(wait|actually|hmm|hold on|let me (?:re-?check|re-?consider|re-?think|double-?check|think again)"
             r"|on second thought|no,)(?!\w)")


def thin(xs, k):
    """At most k items of xs, evenly spaced over it (keeps the first and last when k >= 2)."""
    if len(xs) <= k:
        return list(xs)
    if k <= 1:
        return [xs[len(xs) // 2]] if k == 1 else []
    return [xs[round(i * (len(xs) - 1) / (k - 1))] for i in range(k)]


def select_checkpoints(reasoning, tok, every, max_checkpoints, regex=MARKER_RE):
    """Token positions (probe = first `token_pos` trace tokens) of start / fixed / marker / end checkpoints.
    Half the slots are reserved for fixed intervals and the markers are thinned evenly along the trace,
    so clustered markers cannot take every slot."""
    enc = tok(reasoning, add_special_tokens=False, return_offsets_mapping=True)
    T, ends = len(enc.input_ids), [e for _, e in enc.offset_mapping]
    marks = {}  # position of the token containing the marker's first character -> marker text
    for m in re.finditer(regex, reasoning, re.I):
        pos = bisect_right(ends, m.start())
        if 0 < pos < T:
            marks.setdefault(pos, m.group(1).lower())
    room = max(max_checkpoints - 2, 0)
    fixed = thin([p for p in range(every, T, every) if p not in marks], room // 2)
    keep = thin(sorted(marks), room - len(fixed))
    cps = ([dict(token_pos=0, kind="start", marker=None), dict(token_pos=T, kind="end", marker=None)]
           + [dict(token_pos=p, kind="marker", marker=marks[p]) for p in keep]
           + [dict(token_pos=p, kind="fixed", marker=None) for p in fixed])
    return sorted(cps, key=lambda c: c["token_pos"])


def probe_text(rendered_prompt, prefix_text):
    """Prompt + trace prefix + forced close of the think block (adds <think> if the template did not)."""
    head = rendered_prompt if rendered_prompt.rstrip().endswith("<think>") else rendered_prompt + "<think>\n"
    text = head + prefix_text.rstrip() + "\n</think>\n\n"
    assert text.count("</think>") == 1, "probe text must contain exactly one </think>"
    return text


# ---- runner: what every task gets (see TASKS.md) ---------------------------------------------------

class Context:
    """The loaded checkpoint plus the task-independent helpers: chat-template rendering per --thinking mode,
    generation with the think block split off, prompt-logprob scoring, and saving CSVs to <tag>/."""

    def __init__(self, cfg, llm, tok, out):
        self.cfg, self.llm, self.tok, self.out = cfg, llm, tok, out
        self.base = cfg["thinking"] == "base"
        self.thinking_active = cfg["thinking"] not in ("off", "base")
        self.template_kw = {} if cfg["thinking"] == "native" else {"enable_thinking": self.thinking_active}
        self.sampling = dict(temperature=cfg["temperature"], seed=cfg["seed"], logprobs=20,
                             **({"top_p": 0.95, "top_k": 20} if cfg["temperature"] > 0 else {}))

    def max_tokens(self, answer_tokens):
        """--budget when the model thinks, else the task's own answer length."""
        return self.cfg["budget"] if self.thinking_active else answer_tokens

    def render(self, prompt):
        if self.base:
            return prompt
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True, **self.template_kw)

    def generate(self, prompts, max_tokens, **overrides):
        """All samples per prompt (one unless overrides set n), each a dict from split()."""
        from vllm import SamplingParams
        outs = self.llm.generate([self.render(p) for p in prompts],
                                 SamplingParams(max_tokens=max_tokens, **{**self.sampling, **overrides}))
        return [[self.split(c) for c in o.outputs] for o in outs]

    def split(self, out):
        """Separate the <think> block from the content; return the first content token's top logprobs."""
        tok, ids = self.tok, list(out.token_ids)
        k, reasoning, content, reasoning_tokens, truncated = split_think(ids, tok, self.thinking_active,
                                                                         out.finish_reason)
        j = k
        while j < len(ids) and tok.decode([ids[j]]).strip() == "":
            j += 1
        top = {lp.decoded_token: lp.logprob for lp in out.logprobs[j].values()} if j < len(ids) else {}
        return dict(content=content, reasoning=reasoning, reasoning_tokens=reasoning_tokens, truncated=truncated,
                    generated_tokens=len(ids), finish_reason=out.finish_reason, top=top,
                    wait_count=len(re.findall(r"\bwait\b", reasoning, re.I)))

    def continuation_logprobs(self, pairs):
        """Per-token log-probabilities of each continuation given its context (prompt logprobs, nothing sampled).
        Raw text, no chat template: the context is tokenized as a prompt (BOS if the tokenizer adds one)."""
        from vllm import SamplingParams
        reqs, spans = [], []
        for context, continuation in pairs:
            ctx = self.tok(context).input_ids
            cont = self.tok(continuation, add_special_tokens=False).input_ids
            reqs.append({"prompt_token_ids": ctx + cont})
            spans.append((len(ctx), cont))
        outs = self.llm.generate(reqs, SamplingParams(max_tokens=1, prompt_logprobs=1))
        return [[o.prompt_logprobs[s + t][tid].logprob for t, tid in enumerate(cont)]
                for o, (s, cont) in zip(outs, spans)]

    def save(self, name, rows):
        """Write <tag>/<name> now (partial results survive a crash later in the run)."""
        import pandas as pd
        pd.DataFrame(rows).to_csv(self.out / name, index=False)
        results.commit()


def default_tag(f, suffix=""):
    return (f"{f['model'].split('/')[-1]}{'_' + f['revision'] if f['revision'] else ''}_{f['thinking']}"
            f"{suffix}_n{f['n']}_{time.strftime('%Y%m%d-%H%M')}")


# ---- tasks -----------------------------------------------------------------------------------------
# A task lives in this file because `modal run abstention_modal.py::main` ships only this file to the container
# (plus the image's /data and /analyze.py). Interface and a worked example: TASKS.md.

class ConfidenceAbstention:
    """Kumaran et al. Phases 0/1/2/4 + verbal confidence (the module docstring has the flags)."""
    name = "confidence_abstention"
    options = {}  # uses the dedicated main() flags (--phases, --questions, --phase2-prompt, ...)

    def config(self, f):
        assert f["phase2_prompt"] in PHASE2_INSTRUCTIONS
        assert f["questions"] in QUESTION_FILES
        greedy = f["temperature"] == 0 or (f["temperature"] < 0 and f["thinking"] in ("off", "base"))
        if f["k_samples"] > 1 and greedy:
            raise SystemExit("--k-samples > 1 needs sampling: set --temperature > 0 for a greedy mode")
        phase_list = [int(p) for p in f["phases"].split(",")]
        if f["thinking"] == "base" and not f["base_phase2_verbal"]:  # base checkpoints cannot follow the instruction
            phase_list = [p for p in phase_list if p in (0, 1)]
        # key order fixes the params.json layout
        return dict(model=f["model"], thinking=f["thinking"], n=f["n"], phases=phase_list,
                    thresholds=list(range(0, 101, 10)), budget=f["budget"], max_model_len=f["max_model_len"],
                    temperature=f["temperature"], seed=f["seed"], phase2_prompt=f["phase2_prompt"], tp=f["tp"],
                    revision=f["revision"] or None, dtype=f["dtype"], questions=f["questions"],
                    k_samples=f["k_samples"], verbal=f["thinking"] != "base" or f["base_phase2_verbal"],
                    tag=f["tag"] or default_tag(f, "" if f["questions"] == "simpleqa" else "_" + f["questions"]))

    def run(self, ctx):
        import pandas as pd

        cfg, base, thinking_active = ctx.cfg, ctx.base, ctx.thinking_active
        budget = ctx.max_tokens(4)  # paper: max_tokens=3 after "Answer:"

        data = {p: pd.read_csv("/data/" + QUESTION_FILES[cfg["questions"]].format(p)) for p in (0, 1)}
        qs = {p: data[p].head(cfg["n"]) for p in data}
        arr = {p: arrangements(len(data[p])) for p in data}
        assert not set(data[0].question) & set(data[1].question), "phase 0/1 sets must be disjoint"

        def choices(p, i):
            row = data[p].iloc[i]
            order, gt = arr[p][i]
            return [row[f"choice_{k + 1}"] for k in order], gt, order

        exemplars, mc_shots, cloze_shots = [], "", ""
        if base:
            exemplars = base_exemplars(arr[0])
            for i in exemplars:
                c, gt, _ = choices(0, i)
                mc_shots += mc_shot(data[0].question[i], c, gt)
                cloze_shots += cloze_shot(data[0].question[i], data[0].choice_1[i])

        def cloze(items):
            """Sum and mean log-likelihood of each option text after the cloze prompt."""
            lls = ctx.continuation_logprobs([(cloze_shots + cloze_context(q), f" {option}") for q, c in items for option in c])
            return [(sum(ll), sum(ll) / len(ll)) for ll in lls]

        def run_phase(phase, threshold=None):
            src = 0 if phase == 0 else 1
            valid = [1, 2, 3, 4] + ([5] if phase in (2, 4) else [])
            prompts, meta = [], []
            for i in range(len(qs[src])):
                if src == 0 and i in exemplars:  # an exemplar would be scored with its own answer in context
                    continue
                c, gt, order = choices(src, i)
                prompts.append(answer_prompt(phase, qs[src].iloc[i].question, c, threshold, cfg["phase2_prompt"]))
                meta.append((i, c, gt, order))
            texts = [mc_shots + p for p in prompts]
            stop = {"stop": ["\n"]} if base else {}  # base models run on past the answer digit
            t0 = time.time()
            gens = [g[0] for g in ctx.generate(texts, budget, **stop)]
            extra = (ctx.generate(texts, budget, n=cfg["k_samples"] - 1, seed=cfg["seed"] + 1, **stop)
                     if phase == 1 and cfg["k_samples"] > 1 else None)
            scores = cloze([(qs[src].iloc[i].question, c) for i, c, _, _ in meta]) if base and phase in (0, 1) else None
            print(f"phase {phase}{'' if threshold is None else f' T={threshold}'}: "
                  f"{len(prompts)} prompts in {time.time() - t0:.0f}s", flush=True)
            rows = []
            for n_row, (g, (i, c, gt, order)) in enumerate(zip(gens, meta)):
                lps = digit_logprobs(g["top"], valid)
                conf = softmax(lps)
                ans, how = parse_answer(g["content"], lps, valid)
                row = dict(phase=phase, threshold=threshold, question_index=i, question=qs[src].iloc[i].question,
                           **{f"choice_{k + 1}_arranged": c[k] for k in range(4)},
                           gt_position=gt, arrangement_order=str(order), prompt=prompts[n_row],
                           generated_text=g["content"], final_answer=ans, is_correct=int(int(ans) == gt),
                           is_nota=int(ans == "5"), logprob_chosen=lps[int(ans)],
                           **{f"logprob_choice_{k}": lps[k] for k in range(1, 5)},
                           **{f"confidence_{k}": conf[k] for k in range(1, 5)})
                if 5 in valid:
                    row.update(logprob_nota=lps[5], confidence_5=conf[5])
                row.update(parse_method=how, thinking=cfg["thinking"], reasoning=g["reasoning"],
                           reasoning_tokens=g["reasoning_tokens"], wait_count=g["wait_count"],
                           truncated=g["truncated"], generated_tokens=g["generated_tokens"],
                           finish_reason=g["finish_reason"], digit_mass=sum(math.exp(lps[v]) for v in valid))
                if extra is not None:
                    answers = [ans] + [parse_answer(e["content"], digit_logprobs(e["top"], valid), valid)[0]
                                       for e in extra[n_row]]
                    row.update(k_answers="|".join(answers), consistency=max(map(answers.count, answers)) / len(answers))
                if scores is not None:
                    for k in range(1, 5):
                        row[f"cloze_sum_{k}"], row[f"cloze_mean_{k}"] = scores[4 * n_row + k - 1]
                rows.append(row)
            return rows

        files, all_rows = {}, []
        for phase in cfg["phases"]:
            for threshold in (cfg["thresholds"] if phase == 4 else [None]):
                all_rows += run_phase(phase, threshold)
                ctx.save("phases.csv", all_rows)
                files["phases.csv"] = all_rows

        if 1 in cfg["phases"] and cfg["verbal"]:
            p1 = [r for r in all_rows if r["phase"] == 1]
            prompts = [confidence_prompt(r["question"], [r[f"choice_{k}_arranged"] for k in (1, 2, 3, 4)],
                                         r["final_answer"]) for r in p1]
            gens = [g[0] for g in ctx.generate(prompts, budget if thinking_active else 10)]  # paper: 10 tokens
            verbal = []
            for r, g in zip(p1, gens):
                name, mid = parse_confidence(g["content"])
                verbal.append(dict(question_index=r["question_index"], question=r["question"],
                                   final_answer=r["final_answer"], is_correct=r["is_correct"],
                                   gt_position=r["gt_position"], confidence_output=g["content"],
                                   confidence_class=name, confidence_midpoint=mid,
                                   confidence_valid=int(name is not None), reasoning=g["reasoning"],
                                   reasoning_tokens=g["reasoning_tokens"], truncated=g["truncated"]))
            files["verbal.csv"] = verbal

        rendered_example = ctx.render(mc_shots + answer_prompt(1, qs[1].question[0], choices(1, 0)[0]))
        return files, dict(budget=budget, arrangement_seed=1042, base_exemplars=exemplars,
                           rendered_example=rendered_example)

    def analyze(self, cfg, out):
        """analyze.py writes analysis.json itself; it needs the Phase 0 calibration set and Phase 1."""
        import subprocess
        import sys
        if {0, 1} <= set(cfg["phases"]):
            subprocess.run([sys.executable, "/analyze.py", str(out)], check=False)


SHORT_ANSWER_FILE = "short_answer_questions.csv"


def short_answer_prompt(q, base):
    if base:  # completion format, preceded by the exemplars
        return f"Question: {q}\nAnswer:"
    return ("Answer the question with just the answer: a name, number or short phrase, no explanation.\n\n"
            f"Question: {q}\nAnswer:")


def normalize_answer(s):
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


class ShortAnswer:
    """Template task: free-form short answers scored by exact match against aliases (TASKS.md walks through it)."""
    name = "short_answer"
    options = {"shots": 4}  # exemplars in --thinking base mode; set with --task-opts shots=2

    def config(self, f):
        opts = parse_task_opts(f["task_opts"], self.options)
        return dict(task=self.name, model=f["model"], thinking=f["thinking"], n=f["n"], budget=f["budget"],
                    max_model_len=f["max_model_len"], temperature=f["temperature"], seed=f["seed"], tp=f["tp"],
                    revision=f["revision"] or None, dtype=f["dtype"], **opts,
                    tag=f["tag"] or default_tag(f, "_" + self.name))

    def run(self, ctx):
        import pandas as pd
        cfg = ctx.cfg
        df = pd.read_csv("/data/" + SHORT_ANSWER_FILE)  # file order is the item order: same items at every checkpoint
        shots = df[df.split == "shot"].head(cfg["shots"])
        test = df[df.split == "test"].head(cfg["n"]).reset_index(drop=True)
        prefix = "".join(short_answer_prompt(q, True) + f" {a.split('|')[0]}\n\n"
                         for q, a in zip(shots.question, shots.answer)) if ctx.base else ""
        prompts = [short_answer_prompt(q, ctx.base) for q in test.question]
        gens = [g[0] for g in ctx.generate([prefix + p for p in prompts], ctx.max_tokens(16),
                                           **({"stop": ["\n"]} if ctx.base else {}))]
        rows = []
        for i, (r, p, g) in enumerate(zip(test.itertuples(), prompts, gens)):
            gold = {normalize_answer(a) for a in r.answer.split("|")}
            rows.append(dict(question_index=i, question=r.question, answer=r.answer, prompt=p,
                             generated_text=g["content"], is_correct=int(normalize_answer(g["content"]) in gold),
                             answer_words=len(g["content"].split()), thinking=cfg["thinking"],
                             reasoning=g["reasoning"], reasoning_tokens=g["reasoning_tokens"],
                             truncated=g["truncated"], generated_tokens=g["generated_tokens"],
                             finish_reason=g["finish_reason"]))
        return {"answers.csv": rows}, dict(budget=ctx.max_tokens(16), rendered_example=ctx.render(prefix + prompts[0]))

    def analyze(self, cfg, out):
        import pandas as pd
        a = pd.read_csv(out / "answers.csv")
        return dict(run=out.name, n=len(a), accuracy=round(a.is_correct.mean(), 3),
                    short_format=round((a.answer_words <= 5).mean(), 3), truncated=int(a.truncated.sum()))


TASKS = {t.name: t for t in (ConfidenceAbstention(), ShortAnswer())}


def parse_task_opts(s, defaults):
    """--task-opts "shots=2,foo=x" -> the task's options, each cast to its default's type (int, float or str)."""
    opts = dict(defaults)
    for kv in filter(None, (x.strip() for x in s.split(","))):
        k, v = kv.split("=", 1)
        if k not in defaults:
            raise SystemExit(f"unknown task option {k!r}; this task takes {sorted(defaults)}")
        opts[k] = type(defaults[k])(v)
    return opts


# ---- experiment ---------------------------------------------------------------------

# retries: containers intermittently read a corrupt site-packages file at startup ("bad marshal data");
# a fresh container succeeds
@app.function(volumes={"/cache": cache, "/results": results}, secrets=[hf_secret], timeout=8 * 60 * 60,
              memory=65536, retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run(cfg: dict) -> str:
    import json

    import pandas as pd
    import torch
    import vllm
    from vllm import LLM

    # confidence_abstention configs carry no "task" key
    task = TASKS[cfg.get("task", "confidence_abstention")]
    llm = LLM(model=cfg["model"], revision=cfg["revision"], tokenizer_revision=cfg["revision"], dtype=cfg["dtype"],
              tensor_parallel_size=cfg["tp"], max_model_len=cfg["max_model_len"],
              gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=cfg["seed"])
    cache.commit()  # weights are now in the HF cache volume

    # Paper: greedy. Qwen documents greedy decoding as degenerate for thinking traces,
    # so thinking runs default to Qwen's recommended sampling; --temperature overrides.
    if cfg["temperature"] < 0:
        cfg["temperature"] = 0.6 if cfg["thinking"] not in ("off", "base") else 0.0
    out = Path("/results") / cfg["tag"]
    out.mkdir(parents=True, exist_ok=True)
    ctx = Context(cfg, llm, llm.get_tokenizer(), out)

    files, task_params = task.run(ctx)
    for name, rows in files.items():
        pd.DataFrame(rows).to_csv(out / name, index=False)

    try:
        from huggingface_hub import snapshot_download
        commit = Path(snapshot_download(cfg["model"], revision=cfg["revision"], local_files_only=True)).name
    except Exception:
        commit = None
    # analysis.json first, so params.json is the last file written and marks the run complete
    report = task.analyze(cfg, out)
    if report is not None:
        (out / "analysis.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "params.json").write_text(json.dumps(
        {**cfg, "model_commit": commit, "vllm": vllm.__version__, "torch": torch.__version__,
         "gpu": torch.cuda.get_device_name(), "sampling": ctx.sampling, **task_params}, indent=2))
    results.commit()
    counts = ", ".join(f"{name} {len(rows)} rows" for name, rows in files.items())
    return f"{cfg['tag']}: {counts} -> {RESULTS_VOLUME}/{cfg['tag']}"


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-30B-A3B-Thinking-2507", thinking: str = "native", n: int = 1000,
         phases: str = "0,1,2", gpu: str = "H100", budget: int = 4096, max_model_len: int = 8192,
         temperature: float = -1.0, seed: int = 42, tag: str = "", phase2_prompt: str = "paper",
         revision: str = "", questions: str = "simpleqa", k_samples: int = 1, base_phase2_verbal: bool = False,
         dtype: str = "auto", task: str = "confidence_abstention", task_opts: str = ""):
    assert thinking in ("native", "on", "off", "base")
    if task not in TASKS:
        raise SystemExit(f"--task must be one of {sorted(TASKS)}")
    if thinking == "off" and "think" in model.lower():
        raise SystemExit("thinking-only checkpoints cannot run with --thinking off; use a hybrid (e.g. Qwen/Qwen3-8B)")
    if task_opts and not TASKS[task].options:
        raise SystemExit(f"--task {task} takes no --task-opts")
    flags = dict(locals(), tp=int(gpu.split(":")[1]) if ":" in gpu else 1)
    cfg = TASKS[task].config(flags)
    # spawn, don't wait: a client blocked on .remote() cancels the run if it crashes (e.g. a Wi-Fi/DNS blip),
    # even under --detach. A detached app keeps its last-triggered function alive if the client dies.
    call = run.with_options(gpu=gpu).spawn(cfg)
    print(f"spawned {cfg['tag']}: {call.object_id}; progress: modal app logs <app id above>")


@app.function(volumes={"/cache": cache, "/results": results}, secrets=[hf_secret], timeout=8 * 60 * 60,
              memory=65536, retries=modal.Retries(max_retries=2, initial_delay=10.0))
def probe(cfg: dict) -> str:
    """For each finished trace in <tag>/phases.csv: force-close the think block at several points
    along the trace and read the answer-digit distribution ("if you had to answer now")."""
    import json

    import pandas as pd
    import vllm
    from vllm import LLM, SamplingParams

    out = Path("/results") / cfg["tag"]
    assert (out / "phases.csv").exists(), "the probe reads <tag>/phases.csv of a digit-answer task (TASKS.md)"
    params = json.loads((out / "params.json").read_text())
    model, thinking = cfg["model"] or params["model"], params["thinking"]
    revision = cfg["revision"] or params.get("revision")
    assert thinking not in ("off", "base"), f"no reasoning traces to probe in a --thinking {thinking} run"
    template_kw = {} if thinking == "native" else {"enable_thinking": True}
    df = pd.read_csv(out / "phases.csv")

    llm = LLM(model=model, revision=revision, tokenizer_revision=revision, tensor_parallel_size=cfg["tp"],
              gpu_memory_utilization=0.92, enable_prefix_caching=True,
              max_model_len=max(params["max_model_len"], params["budget"] + 1024), seed=params["seed"])
    tok = llm.get_tokenizer()
    cache.commit()

    rows = []
    for phase in cfg["phases"]:
        valid = [1, 2, 3, 4] + ([5] if phase == 2 else [])
        g = df[(df.phase == phase) & (df.truncated == 0) & df.reasoning.fillna("").str.strip().ne("")]
        if cfg["n"] > 0:
            g = g[g.question_index < cfg["n"]]
        texts, meta = [], []
        for r in g.itertuples():
            rendered = tok.apply_chat_template([{"role": "user", "content": r.prompt}], tokenize=False,
                                               add_generation_prompt=True, **template_kw)
            ids = tok(r.reasoning, add_special_tokens=False).input_ids
            for j, c in enumerate(select_checkpoints(r.reasoning, tok, cfg["every"], cfg["max_checkpoints"])):
                texts.append(probe_text(rendered, tok.decode(ids[:c["token_pos"]], skip_special_tokens=True)))
                meta.append((r, j, c, len(ids)))
        outs = llm.generate(texts, SamplingParams(max_tokens=1, temperature=0, logprobs=20))
        for o, (r, j, c, T) in zip(outs, meta):
            gen = o.outputs[0]
            lps = digit_logprobs({lp.decoded_token: lp.logprob for lp in gen.logprobs[0].values()}, valid)
            conf = softmax(lps)
            sampled = tok.decode(gen.token_ids[:1]).strip()
            forced = sampled if sampled in [str(v) for v in valid] else str(max(valid, key=lambda v: lps[v]))
            rows.append(dict(phase=phase, question_index=r.question_index, checkpoint_idx=j,
                             token_pos=c["token_pos"], trace_tokens=T, frac=c["token_pos"] / T, kind=c["kind"],
                             marker=c["marker"], forced_answer=forced, prob_forced=conf[int(forced)],
                             **{f"prob_{k}": conf.get(k, float("nan")) for k in range(1, 6)},
                             prob_final=conf[int(r.final_answer)], matches_final=int(forced == str(r.final_answer)),
                             final_answer=r.final_answer, is_correct=r.is_correct, is_nota=r.is_nota,
                             gt_position=r.gt_position))
        print(f"phase {phase}: {len(g)} traces, {len(texts)} probes", flush=True)
        pd.DataFrame(rows).to_csv(out / "checkpoints.csv", index=False)
        results.commit()

    kinds = pd.DataFrame(rows).kind.value_counts().to_dict()
    (out / "checkpoints_params.json").write_text(json.dumps(
        dict(tag=cfg["tag"], model=model, revision=revision, thinking=thinking, phases=cfg["phases"], every=cfg["every"],
             max_checkpoints=cfg["max_checkpoints"], n=cfg["n"], regex=MARKER_RE, checkpoints_by_kind=kinds,
             vllm=vllm.__version__), indent=2))
    results.commit()
    return f"{cfg['tag']}: {len(rows)} checkpoints -> {RESULTS_VOLUME}/{cfg['tag']}/checkpoints.csv"


@app.local_entrypoint()
def checkpoints(tag: str, gpu: str = "H100", phases: str = "1,2", every: int = 256, max_checkpoints: int = 20,
                n: int = 0, model: str = "", revision: str = ""):
    cfg = dict(tag=tag, model=model, revision=revision, phases=[int(p) for p in phases.split(",")], every=every,
               max_checkpoints=max_checkpoints, n=n, tp=int(gpu.split(":")[1]) if ":" in gpu else 1)
    call = probe.with_options(gpu=gpu).spawn(cfg)  # spawn, don't wait (see main)
    print(f"spawned probe {cfg['tag']}: {call.object_id}")


def start_texts(df, phase, tok, template_kw, prefixes):
    """The probe's start-checkpoint text for every trace it probed, plus each prefix appended after the closed block."""
    g = df[(df.phase == phase) & (df.truncated == 0) & df.reasoning.fillna("").str.strip().ne("")]
    texts, meta = [], []
    for r in g.itertuples():
        rendered = tok.apply_chat_template([{"role": "user", "content": r.prompt}], tokenize=False,
                                           add_generation_prompt=True, **template_kw)
        for prefix in prefixes:
            texts.append(probe_text(rendered, "") + prefix)
            meta.append((r, prefix))
    return texts, meta


def summarize_start_format(sf):
    """Per phase x prefix: digit mass, digit-first share, commonest first tokens, agreement with the probe."""
    out = []
    for (phase, prefix), s in sf.groupby(["phase", "prefix"], sort=False):
        top5 = s.top_token.value_counts(normalize=True).head(5).round(3)
        agree = s.agrees_probe.dropna()
        out.append(dict(phase=int(phase), prefix=prefix, n=len(s), mean_digit_mass=round(s.digit_mass.mean(), 3),
                        median_digit_mass=round(s.digit_mass.median(), 3),
                        share_digit_first=round(s.top_is_digit.mean(), 3),
                        share_forced_5=round((s.forced_answer == "5").mean(), 3),
                        share_forced_5_digit_first=round((s[s.top_is_digit == 1].forced_answer == "5").mean(), 3),
                        top5_first_tokens={repr(k): v for k, v in top5.items()},
                        agreement_with_probe=round(agree.mean(), 4) if len(agree) else None, n_compared=len(agree)))
    return out


@app.function(volumes={"/cache": cache, "/results": results}, secrets=[hf_secret], timeout=2 * 60 * 60,
              memory=65536, retries=modal.Retries(max_retries=2, initial_delay=10.0))
def start_probe(cfg: dict) -> str:
    """Format check of the probe's start checkpoint (empty closed think block): how much first-token mass is on
    digits, and whether answer prefixes restore a digit-first answer. Same model load and readout as `probe`."""
    import json

    import pandas as pd
    import vllm
    from vllm import LLM, SamplingParams

    out = Path("/results") / cfg["tag"]
    params = json.loads((out / "params.json").read_text())
    cp_params = json.loads((out / "checkpoints_params.json").read_text())
    model, thinking = cp_params["model"], params["thinking"]
    revision = cp_params.get("revision", params.get("revision"))
    assert cp_params["n"] == 0, "start_texts takes every probed trace; this probe ran with --n"
    template_kw = {} if thinking == "native" else {"enable_thinking": True}
    df = pd.read_csv(out / "phases.csv")
    probe_start = pd.read_csv(out / "checkpoints.csv").query("kind == 'start'")
    probe_start = dict(zip(zip(probe_start.phase, probe_start.question_index), probe_start.forced_answer.astype(str)))

    llm = LLM(model=model, revision=revision, tokenizer_revision=revision, tensor_parallel_size=cfg["tp"],
              gpu_memory_utilization=0.92, enable_prefix_caching=True,
              max_model_len=max(params["max_model_len"], params["budget"] + 1024), seed=params["seed"])
    tok = llm.get_tokenizer()
    cache.commit()

    prefixes = [""] + cfg["prefixes"]
    rows = []
    for phase in cfg["phases"]:
        valid = [1, 2, 3, 4] + ([5] if phase == 2 else [])
        texts, meta = start_texts(df, phase, tok, template_kw, prefixes)
        outs = llm.generate(texts, SamplingParams(max_tokens=1, temperature=0, logprobs=20))
        for o, (r, prefix) in zip(outs, meta):
            gen = o.outputs[0]
            top = {lp.decoded_token: lp.logprob for lp in gen.logprobs[0].values()}
            lps = digit_logprobs(top, valid)
            conf = softmax(lps)
            sampled = tok.decode(gen.token_ids[:1]).strip()
            forced = sampled if sampled in [str(v) for v in valid] else str(max(valid, key=lambda v: lps[v]))
            ref = probe_start.get((phase, r.question_index)) if prefix == "" else None
            rows.append(dict(phase=phase, question_index=r.question_index, prefix=prefix,
                             top_token=tok.decode(gen.token_ids[:1]), top_logprob=max(top.values()),
                             top_is_digit=int(sampled in [str(v) for v in valid]),
                             digit_mass=sum(math.exp(lps[v]) for v in valid), forced_answer=forced,
                             prob_forced=conf[int(forced)], **{f"prob_{k}": conf.get(k, float("nan")) for k in range(1, 6)},
                             probe_forced_answer=ref, agrees_probe=None if ref is None else int(forced == ref),
                             final_answer=r.final_answer, is_correct=r.is_correct, is_nota=r.is_nota))
        print(f"phase {phase}: {len(texts) // len(prefixes)} traces x {len(prefixes)} prefixes", flush=True)

    sf = pd.DataFrame(rows)
    sf.to_csv(out / "start_format.csv", index=False)
    summary = summarize_start_format(sf)
    (out / "start_format.json").write_text(json.dumps(
        dict(tag=cfg["tag"], model=model, revision=revision, prefixes=prefixes, n_probe_start=len(probe_start),
             summary=summary, vllm=vllm.__version__), indent=2))
    results.commit()
    for s in summary:
        print(json.dumps(s), flush=True)
    # a low agreement means this readout is not the probe's; no raise, which would retry the whole GPU run
    worst = min((s["agreement_with_probe"] for s in summary if s["agreement_with_probe"] is not None), default=None)
    flag = "" if worst is not None and worst >= 0.98 else " WARNING: start readout does not reproduce checkpoints.csv"
    return f"{cfg['tag']}: {len(rows)} rows -> {RESULTS_VOLUME}/{cfg['tag']}/start_format.csv; agreement {worst}{flag}"


@app.local_entrypoint()
def start_format(tag: str, gpu: str = "H200", phases: str = "1,2", prefixes: str = "Answer: |The answer is "):
    """modal run --detach abstention_modal.py::start_format --tag <tag> --gpu H200"""
    cfg = dict(tag=tag, phases=[int(p) for p in phases.split(",")], prefixes=[p for p in prefixes.split("|") if p],
               tp=int(gpu.split(":")[1]) if ":" in gpu else 1)
    call = start_probe.with_options(gpu=gpu).spawn(cfg)  # spawn, don't wait (see main)
    print(f"spawned start_format {cfg['tag']}: {call.object_id}")


# Weights are 60-130 GB per checkpoint; fetching them inside a GPU run bills the GPU for the whole download.
# Prefetch on CPU containers into the same HF cache the GPU runs read.
@app.function(volumes={"/cache": cache}, secrets=[hf_secret], timeout=3 * 60 * 60, cpu=8, memory=16384)
def fetch(model: str, revision: str) -> str:
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    from huggingface_hub import snapshot_download
    path = snapshot_download(model, revision=revision or None,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "tokenizer*"])
    cache.commit()
    return f"{model}@{revision or 'main'} -> {path}"


@app.function(timeout=4 * 60 * 60)
def fetch_all(pairs: list) -> list:
    return list(fetch.starmap(pairs))


@app.local_entrypoint()
def prefetch(specs: str):
    """Download checkpoints into the HF-cache volume without a GPU, in parallel.

        modal run --detach abstention_modal.py::prefetch --specs "allenai/Olmo-3-1125-32B@stage1-step19000,allenai/Olmo-3-32B-Think"
    With --detach the downloads finish even if this terminal drops (it streams logs until then). Re-running is safe
    (finished files are skipped, partial ones resumed).
    """
    pairs = [tuple(s.strip().split("@", 1)) if "@" in s else (s.strip(), "") for s in specs.split(",") if s.strip()]
    # one spawned function fans out server-side: a detached app only keeps the *last* triggered function alive
    # if this client dies, so spawning each download from here would orphan all but one
    call = fetch_all.spawn(pairs)
    print(f"spawned {len(pairs)} downloads: {call.object_id}")
