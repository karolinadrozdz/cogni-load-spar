"""Confidence steering (Kumaran et al. 2026 recipe) with and without a reasoning trace, HF transformers on Modal.

Stages, each its own `modal run`; outputs land in the abstention-results volume under steer-<model>-<tag>/:
    # no-trace Phase 0 + 1, Phase 2 on the Phase 0 questions with last-prompt-token residuals, then vectors.pt +
    # eval_items.json (build_vector.py)
    modal run --detach steering_modal.py::notrace --model Qwen/Qwen3-8B --thinking on --tag pilot
    # no-trace Phase 2 on the eval items: sweep layers x alpha, controls, Phase 1 capability check
    modal run --detach steering_modal.py::gate --model Qwen/Qwen3-8B --thinking on --tag pilot
    # or, for a thinking checkpoint: pre-trace residuals (last token of the reasoning-mode Phase 2 prompt) on the
    # harness run's Phase 1 items, labelled by its k-sample consistency, then vectors.pt + eval_items.json
    modal run --detach steering_modal.py::pretrace --model allenai/Olmo-3.1-32B-Think --thinking native \
        --harness-tag <harness tag>
    # reasoning mode at one layer: alphas x positions (start / think / post)
    modal run --detach steering_modal.py::reason --model Qwen/Qwen3-8B --thinking on --tag pilot --layer 17 --n 50

--thinking is the checkpoint's reasoning switch, as in abstention_modal.py: native (Olmo-3 Think), on (hybrid Qwen3),
off (Instruct; no-trace stages only). No-trace mode: native -> the think block force-closed empty, exactly the in-trace
probe's start checkpoint, then --answer-prefix (default none); on/off -> the template with enable_thinking=False.

Layout of steer-<model>-<tag>/:
  notrace_phases.csv, construction_phases.csv, hiddens.pt, vectors.pt, eval_items.json   (notrace)
  pretrace_items.csv, pretrace_hiddens.pt, vectors.pt, eval_items.json                   (pretrace)
  gate/<condition>.csv, gate_phases.csv                                                   (gate)
  reason/<condition>.csv, reason_phases.csv                                               (reason)
Rows follow the phases.csv schema plus alpha, layer, position_mode, vector. Finished condition files are skipped,
so re-running a stage resumes it. Analysis: python analyze_steering.py results/steer-<model>-<tag>.
"""

import functools
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import modal

from abstention_modal import (QUESTION_FILES, RESULTS_VOLUME, Context, answer_prompt, arrangements, cache,
                              digit_logprobs, hf_secret, parse_answer, probe_text, results, softmax, split_think)

HERE = Path(__file__).parent
DATA = Path("/data") if Path("/data").exists() else HERE / "data"
GATE_ALPHAS = (-2, -1, 0, 1, 2)

# own image: the abstention image ends in add_local_* layers, so nothing can be pip-installed on top of it;
# transformers pinned to the version the hooks were tested against locally
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "transformers==4.57.6", "accelerate", "pandas", "numpy", "scipy",
                    "scikit-learn", "statsmodels", "huggingface_hub[hf_transfer]")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(HERE / "data", "/data")
    .add_local_python_source("abstention_modal", "analyze", "build_vector")
)
app = modal.App("confidence-steering", image=image)


# ---- model, hooks, generation ---------------------------------------------------------------

def load_model(name, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa",
                                                 device_map=device).eval()
    return model, tok


def decoder_layers(model):
    """Residual after block L = output of model.model.layers[L]; vectors are read and steered there."""
    return model.model.layers


def residual(out):
    return out[0] if isinstance(out, tuple) else out


def sweep_layers(n_layers):
    """Every 4th layer from 25% to 75% depth."""
    return list(range(n_layers // 4, 3 * n_layers // 4 + 1, 4))


class ThinkTracker:
    """StoppingCriteria that never stops: after each appended token, marks rows that just completed "</think>"
    (decoded text, since Olmo's </think> is ordinary BPE that merges with its neighbours) or emitted EOS."""

    def __init__(self, tok, prompt_len, n_rows, eos_ids):
        import torch
        self.tok, self.prompt_len, self.eos = tok, prompt_len, set(eos_ids)
        self.closed = torch.zeros(n_rows, dtype=torch.bool)
        self.finished = torch.zeros(n_rows, dtype=torch.bool)

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        start = max(self.prompt_len, input_ids.shape[1] - 16)  # "</think>" spans at most 8 tokens
        for i, last in enumerate(input_ids[:, -1].tolist()):
            if self.finished[i]:
                continue
            if not self.closed[i] and "</think>" in self.tok.decode(input_ids[i, start:], skip_special_tokens=True):
                self.closed[i] = True
            if last in self.eos:
                self.finished[i] = True
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


class Steer:
    """Forward hook on one decoder layer: h <- h + alpha * v at the positions `mode` selects (v = scale_L * v_hat).
    start: last prompt token; think: that plus every decode step until the row has emitted "</think>";
    post: decode steps after "</think>". Prefill is the first call (left padding: last prompt token at -1)."""

    def __init__(self, model, layer, v, alpha, mode, tracker):
        import torch
        self.v = v.float().to(model.device)
        self.alpha, self.mode, self.tracker = alpha, mode, tracker
        self.steered = torch.zeros(len(tracker.closed), dtype=torch.long)
        self.calls = 0
        self.handle = decoder_layers(model)[layer].register_forward_hook(self.hook)

    def rows(self, prefill):
        import torch
        t = self.tracker
        if prefill:
            return torch.full_like(t.closed, self.mode in ("start", "think"))
        if self.mode == "start":
            return torch.zeros_like(t.closed)
        return ~t.finished & (~t.closed if self.mode == "think" else t.closed)

    def hook(self, module, args, out):
        h = residual(out)
        prefill = not self.calls
        assert prefill or h.shape[1] == 1, "decode step saw more than one position: KV cache is off"
        self.calls += 1
        rows = self.rows(prefill)
        self.steered += rows
        last = h[:, -1].float()
        delta = self.alpha * self.v * rows.to(h.device)[:, None]
        h = h.clone()
        h[:, -1] = (last + delta).to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


class TopLogprobs:
    """LogitsProcessor that records the top-20 raw log-probs per step (vLLM's logprobs=20) for the digit readout.
    Custom processors run before temperature/top-p, so these are unwarped."""

    def __init__(self):
        self.vals, self.ids = [], []

    def __call__(self, input_ids, scores):
        v, i = scores.float().log_softmax(-1).topk(20)
        self.vals.append(v)
        self.ids.append(i)
        return scores


class RowLogprobs:
    """out.logprobs for Context.split: position -> {token id: Logprob-like}, decoded on demand."""

    def __init__(self, tok, vals, ids):
        self.tok, self.vals, self.ids = tok, vals, ids

    def __getitem__(self, j):
        return {t: SimpleNamespace(decoded_token=self.tok.decode([t]), logprob=v)
                for v, t in zip(self.vals[j].tolist(), self.ids[j].tolist())}


def steered_mask(mode, n_prompt, ids, tok, finish_reason):
    """Which positions of prompt + ids the Steer hook shifted: the last prompt token (start, think), then decode
    inputs ids[t] (the forward pass that produces ids[t + 1]) before the row completed "</think>" (think) or from
    the token completing it on (post)."""
    closed = "</think>" in tok.decode(ids, skip_special_tokens=True)
    c = split_think(ids, tok, True, finish_reason)[0] if closed else len(ids)  # tokens through </think>
    m = [False] * (n_prompt + len(ids))
    m[n_prompt - 1] = mode in ("start", "think")
    for t in {"start": (), "think": range(c - 1), "post": range(c - 1, len(ids) - 1)}[mode]:
        m[n_prompt + t] = True
    return m


def expected_steered(mode, ids, tok, finish_reason):
    """Steered forward passes for one row implied by its final ids (one forward pass per generated token)."""
    return sum(steered_mask(mode, 1, ids, tok, finish_reason))


def answer_index(ids, tok, finish_reason):
    """Index in ids of the first content token (Context.split's j); None if truncated or no content."""
    k, _, _, _, truncated = split_think(ids, tok, True, finish_reason)
    while k < len(ids) and tok.decode([ids[k]]).strip() == "":
        k += 1
    return None if truncated or k >= len(ids) else k


def probe_answer_pos(model, tok, texts, gens, steer, us):
    """Manipulation check (b): one forward pass over prompt + ids[:j] (the last position produces the first content
    token ids[j]), steered at exactly the positions generation steered; the residual at each probe layer projected on
    us[layer]. Per row a dict (score per layer, top-1 here, the generation's recorded top-1 and its log-prob), or None
    if truncated."""
    import torch
    L, v, alpha, mode = steer
    layers, out = decoder_layers(model), []
    for text, g in zip(texts, gens):
        j = answer_index(g["ids"], tok, g["finish_reason"])
        if j is None:
            out.append(None)
            continue
        prompt = tok(text).input_ids
        mask = torch.tensor(steered_mask(mode, len(prompt), g["ids"], tok, g["finish_reason"])[:len(prompt) + j])
        seen = {}

        def add(m, a, o):
            h = residual(o)
            h = (h.float() + alpha * v.float().to(h.device) * mask.to(h.device)[None, :, None]).to(h.dtype)
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

        # capture before steer, as in generate
        hooks = [layers[k].register_forward_hook(lambda m, a, o, k=k: seen.update({k: residual(o)[0, -1].float()}))
                 for k in us] + [layers[L].register_forward_hook(add)]
        try:
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([prompt + g["ids"][:j]], device=model.device), use_cache=False,
                               logits_to_keep=1).logits[0, -1].float()
        finally:
            for hk in hooks:
                hk.remove()
        lps = logits.log_softmax(-1)
        ref = int(g["tids"][j, 0])
        out.append(dict(score={k: float(seen[k].cpu() @ u) for k, u in us.items()}, top1=int(lps.argmax()), token=g["ids"][j],
                        ref_top1=ref, lp_diff=abs(float(lps[ref]) - float(g["vals"][j, 0]))))
    return out


def check_faithful(probes, greedy, strict):
    """The recompute reproduces generation: same top-1 as recorded (and as the token chosen, under greedy) and the
    same log-prob. strict (fp32): every row, 1e-3; bf16: >= 98% of rows, median difference < 0.05."""
    import numpy as np
    done = [p for p in probes if p]
    if not done:
        return
    match = np.mean([p["top1"] == p["ref_top1"] and (not greedy or p["top1"] == p["token"]) for p in done])
    diff = [p["lp_diff"] for p in done]
    print(f"  probe recompute on {len(done)} rows: top-1 agreement {match:.3f}, |log-prob diff| median "
          f"{np.median(diff):.2e}, max {max(diff):.2e}", flush=True)
    ok = (match == 1 and max(diff) < 1e-3) if strict else (match >= 0.98 and np.median(diff) < 0.05)
    assert ok, "probe recompute does not reproduce generation"


def generate(model, tok, texts, max_new_tokens, batch=32, steer=None, temperature=0.0, seed=42, capture=()):
    """Batched HF generate (prompts sorted by length, left padding). steer = (layer, v, alpha, mode) or None.
    Returns per prompt: ids (generated, through EOS), finish_reason, vals/ids of the top-20 log-probs per step,
    and the last-prompt-token residual at each `capture` layer. Asserts the steered-position count per row."""
    import torch
    from transformers import LogitsProcessorList, StoppingCriteriaList

    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    # Olmo-3's generation config stops only on <|endoftext|>; the chat turn ends with <|im_end|>.
    # use_cache=True below: Olmo-3 configs ship use_cache=false, which would re-run the whole sequence every step
    eos += [t for t in [tok.get_vocab().get("<|im_end|>")] if t is not None and t not in eos]
    enc_all = [tok(t).input_ids for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(enc_all[i]))
    sampling = (dict(do_sample=True, temperature=temperature, top_p=0.95, top_k=20) if temperature > 0
                else dict(do_sample=False, temperature=None, top_p=None, top_k=None))
    captured = {L: [] for L in capture}
    hooks = [decoder_layers(model)[L].register_forward_hook(
        lambda m, a, o, L=L: captured[L].append(residual(o)[:, -1].float().cpu()) if residual(o).shape[1] > 1 else None)
        for L in capture]
    outs = [None] * len(texts)
    try:
        for b0 in range(0, len(order), batch):
            idx = order[b0:b0 + batch]
            enc = tok.pad({"input_ids": [enc_all[i] for i in idx]}, return_tensors="pt").to(model.device)
            P = enc.input_ids.shape[1]
            tracker = ThinkTracker(tok, P, len(idx), eos)
            top = TopLogprobs()
            st = Steer(model, steer[0], steer[1], steer[2], steer[3], tracker) if steer else None
            torch.manual_seed(seed + b0)
            t0 = time.time()
            try:
                with torch.no_grad():
                    seq = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                                         eos_token_id=eos, use_cache=True, logits_processor=LogitsProcessorList([top]),
                                         stopping_criteria=StoppingCriteriaList([tracker]), **sampling)
            finally:
                if st:
                    st.handle.remove()
            vals, tids = torch.stack(top.vals, 1).cpu(), torch.stack(top.ids, 1).cpu()
            n_new = 0
            for r, i in enumerate(idx):
                gen = seq[r, P:].tolist()
                end = next((k + 1 for k, t in enumerate(gen) if t in eos), None)
                ids, finish = (gen[:end], "stop") if end else (gen, "length")
                n_new += len(ids)
                if st:
                    want = expected_steered(steer[3], ids, tok, finish)
                    assert st.steered[r] == want, f"row {i}: steered {int(st.steered[r])} positions, expected {want}"
                outs[i] = dict(ids=ids, finish_reason=finish, vals=vals[r, :len(ids)], tids=tids[r, :len(ids)],
                               steered=int(st.steered[r]) if st else 0,
                               hidden={L: captured[L][-1][r] for L in capture})
            dt = time.time() - t0
            print(f"  batch {b0 // batch + 1}/{math.ceil(len(order) / batch)}: {len(idx)} x {seq.shape[1] - P} steps, "
                  f"{n_new} tokens in {dt:.0f}s ({n_new / dt:.0f} tok/s)", flush=True)
    finally:
        for h in hooks:
            h.remove()
    return outs


# ---- items, prompts, rows -------------------------------------------------------------------

def load_items(questions):
    import pandas as pd
    return {p: pd.read_csv(DATA / QUESTION_FILES[questions].format(p)) for p in (0, 1)}


@functools.cache
def arrangement(n):
    return arrangements(n)


def item(data, src, i):
    row = data[src].iloc[i]
    order, gt = arrangement(len(data[src]))[i]
    return row.question, [row[f"choice_{k + 1}"] for k in order], gt, order


def render(tok, prompt, thinking, notrace, answer_prefix=""):
    ctx = Context(dict(thinking="off" if notrace and thinking != "native" else thinking, temperature=0, seed=0),
                  None, tok, None)
    text = ctx.render(prompt)
    # the native template always opens a think block: close it empty, as the in-trace probe's start checkpoint
    return probe_text(text, "") + answer_prefix if notrace and thinking == "native" else text


def phase_row(phase, i, q, c, gt, order, prompt, content, lps, g, thinking, cond):
    """One phases.csv row (abstention_modal.py's schema) plus the steering condition."""
    valid = [1, 2, 3, 4] + ([5] if phase == 2 else [])
    conf = softmax(lps)
    ans, how = parse_answer(content, lps, valid)
    row = dict(phase=phase, threshold=None, question_index=i, question=q,
               **{f"choice_{k + 1}_arranged": c[k] for k in range(4)},
               gt_position=gt, arrangement_order=str(order), prompt=prompt, generated_text=content,
               final_answer=ans, is_correct=int(int(ans) == gt), is_nota=int(ans == "5"), logprob_chosen=lps[int(ans)],
               **{f"logprob_choice_{k}": lps[k] for k in range(1, 5)},
               **{f"confidence_{k}": conf[k] for k in range(1, 5)})
    if phase == 2:
        row.update(logprob_nota=lps[5], confidence_5=conf[5])
    row.update(parse_method=how, thinking=thinking, reasoning=g["reasoning"], reasoning_tokens=g["reasoning_tokens"],
               wait_count=g["wait_count"], truncated=g["truncated"], generated_tokens=g["generated_tokens"],
               finish_reason=g["finish_reason"], digit_mass=sum(math.exp(lps[v]) for v in valid), **cond)
    return row


def notrace_rows(model, tok, data, cfg, phase, indices, cond, steer=None, capture=(), src=None):
    """Single forward pass per item; digits read from the last prompt position's logits.
    src: question file (default: Phase 0 questions for phase 0, Phase 1 questions otherwise)."""
    src = (0 if phase == 0 else 1) if src is None else src
    valid = [1, 2, 3, 4] + ([5] if phase == 2 else [])
    metas = [(i, *item(data, src, i)) for i in indices]
    prompts = [answer_prompt(phase, q, c, None, "paper") for _, q, c, _, _ in metas]
    texts = [render(tok, p, cfg["thinking"], True, cfg.get("answer_prefix", "")) for p in prompts]
    gens = generate(model, tok, texts, 1, cfg["batch"],
                    steer=steer, capture=capture)
    rows = []
    for (i, q, c, gt, order), p, g in zip(metas, prompts, gens):
        top = RowLogprobs(tok, g["vals"], g["tids"])[0]
        lps = digit_logprobs({lp.decoded_token: lp.logprob for lp in top.values()}, valid)
        content = tok.decode(g["tids"][0, :1].tolist()).strip()
        g0 = dict(reasoning="", reasoning_tokens=0, wait_count=0, truncated=0, generated_tokens=1,
                  finish_reason="length")
        rows.append(phase_row(phase, i, q, c, gt, order, p, content, lps, g0, "notrace", cond))
    return rows, gens


def save_csv(path, rows):
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    if Path("/results").exists():
        results.commit()


def concat(run, stage):
    import pandas as pd
    files = sorted((run / stage).glob("*.csv"))
    pd.concat([pd.read_csv(f) for f in files]).to_csv(run / f"{stage}_phases.csv", index=False)
    print(f"{stage}: {len(files)} conditions -> {run / f'{stage}_phases.csv'}")


def unsteered():
    return dict(alpha=0, layer=-1, position_mode="none", vector="none")


# ---- stages (plain functions; the Modal functions and test_steering_local.py call these) -----

def run_notrace(model, tok, run, cfg, sweep=None, **build_kw):
    import torch
    from build_vector import build
    data = load_items(cfg["questions"])
    n_layers = len(decoder_layers(model))
    sweep = sweep or sweep_layers(n_layers)
    # vectors also every 4th layer outside the sweep, so reasoning runs at L* +- 8 have one
    layers = sorted(set(range(sweep[0] % 4, n_layers, 4)) | set(sweep))
    rows0, _ = notrace_rows(model, tok, data, cfg, 0, range(min(cfg["n"], len(data[0]))), unsteered())
    rows1, _ = notrace_rows(model, tok, data, cfg, 1, range(min(cfg["n"], len(data[1]))), unsteered())
    # construction set: Phase 2 prompts on the Phase 0 questions, residuals at the position steering acts on
    rowsc, gens = notrace_rows(model, tok, data, cfg, 2, range(min(cfg["n"], len(data[0]))), unsteered(),
                               capture=layers, src=0)
    save_csv(run / "notrace_phases.csv", rows0 + rows1)
    save_csv(run / "construction_phases.csv", rowsc)
    torch.save(dict(layers=layers, sweep=sweep, question_index=[r["question_index"] for r in rowsc],
                    h=torch.stack([torch.stack([g["hidden"][L] for g in gens]) for L in layers]).half()),
               run / "hiddens.pt")
    for p, rows in ((0, rows0), (1, rows1), ("2 (construction)", rowsc)):
        print(f"phase {p}: accuracy {sum(r['is_correct'] for r in rows) / len(rows):.3f}, "
              f"abstention {sum(r['is_nota'] for r in rows) / len(rows):.3f}, "
              f"digit mass {sum(r['digit_mass'] for r in rows) / len(rows):.3f}")
    build(run, **build_kw)


def run_pretrace(model, tok, run, cfg, sweep=None, n_eval_cell=150, **build_kw):
    """Residuals at the last token of the reasoning-mode Phase 2 prompt (just before deliberation), one
    forward pass per harness Phase 1 item; labels from the harness run's k-sample consistency."""
    import numpy as np
    import torch
    from build_vector import build_pretrace, pretrace_items, write_evals
    data = load_items(cfg["questions"])
    n_layers = len(decoder_layers(model))
    sweep = sweep or sweep_layers(n_layers)
    layers = sorted(set(range(sweep[0] % 4, n_layers, 4)) | set(sweep))
    items = pretrace_items(cfg["harness"], data[1])
    evals = write_evals(run, items, n_eval_cell)
    items["split"] = np.where(items.question_index.isin(evals), "eval", "construction")
    save_csv(run / "pretrace_items.csv", items)
    ctx = Context(dict(thinking=cfg["thinking"], temperature=0, seed=0), None, tok, None)
    metas = [item(data, 1, i) for i in items.question_index]
    assert [gt for _, _, gt, _ in metas] == items.gt_position.tolist(), "harness arrangement differs"
    texts = [ctx.render(answer_prompt(2, q, c, None, "paper")) for q, c, _, _ in metas]
    assert cfg["thinking"] != "native" or texts[0].rstrip().endswith("<think>"), texts[0][-80:]
    gens = generate(model, tok, texts, 1, cfg["batch"], capture=layers)
    torch.save(dict(layers=layers, sweep=sweep, question_index=items.question_index.tolist(),
                    h=torch.stack([torch.stack([g["hidden"][L] for g in gens]) for L in layers]).half()),
               run / "pretrace_hiddens.pt")
    build_pretrace(run, **build_kw)


def load_vectors(run):
    import torch
    return torch.load(run / "vectors.pt"), json.loads((run / "eval_items.json").read_text())["question_index"]


def run_gate(model, tok, run, cfg):
    vecs, evals = load_vectors(run)
    data = load_items(cfg["questions"])
    conds = ([(2, "real", L, a) for L in vecs["sweep"] for a in GATE_ALPHAS]
             + [(2, vec, L, a) for vec in ("random", "shuffled") for L in vecs["sweep"] for a in (-2, 2)]
             + [(1, "real", L, a) for L in vecs["sweep"] for a in (-2, 2)])
    for phase, vec, L, a in conds:
        path = run / "gate" / f"p{phase}_{vec}_L{L}_a{a:+d}.csv"
        if path.exists():
            continue
        print(f"gate phase {phase} {vec} L{L} alpha {a:+d}", flush=True)
        rows, _ = notrace_rows(model, tok, data, cfg, phase, evals,
                               dict(alpha=a, layer=L, position_mode="start", vector=vec),
                               steer=(L, vecs["scale"][L] * vecs[vec][L], a, "start"))
        save_csv(path, rows)
    concat(run, "gate")


def run_reason(model, tok, run, cfg):
    import torch
    vecs, evals = load_vectors(run)
    data = load_items(cfg["questions"])
    L, vec, T = cfg["layer"], cfg["vector"], cfg["temperature"]
    # manipulation check, pre-trace runs: residual at L + 8 (capped at the last vector layer) on that layer's own unit
    # vector; uninformative (NaN) if the cap leaves it at or before L. The shared unsteered file records every layer
    probe = "L_star" in vecs
    probe_L = min(L + 8, max(vecs["layers"])) if probe else None
    v = vecs["scale"][L] * vecs[vec][L]
    ctx = Context(dict(thinking=cfg["thinking"], temperature=T, seed=cfg["seed"]), None, tok, None)
    metas = [(i, *item(data, 1, i)) for i in evals[:cfg["n"]]]
    prompts = [answer_prompt(2, q, c, None, "paper") for _, q, c, _, _ in metas]
    texts = [ctx.render(p) for p in prompts]
    for pos in cfg["positions"]:
        for a in cfg["alphas"]:
            # alpha 0 is unsteered whatever the position, layer or vector: one shared file per temperature
            path = run / "reason" / (f"unsteered_T{T:g}.csv" if a == 0 else f"{pos}_{vec}_L{L}_a{a:+d}_T{T:g}.csv")
            if path.exists():
                continue
            mode = "think" if a == 0 else pos
            print(f"reason {mode} {vec} L{L} alpha {a:+d} T={T:g}: {len(texts)} prompts", flush=True)
            ks = [] if not probe else vecs["layers"] if a == 0 else [probe_L] if probe_L > L else []
            us = {k: vecs["real"][k].float() for k in ks}
            gens = generate(model, tok, texts, cfg["budget"], cfg["batch"], steer=(L, v, a, mode),
                            temperature=T, seed=cfg["seed"], capture=ks)
            probes = [None] * len(gens)
            if ks:
                probes = probe_answer_pos(model, tok, texts, gens, (L, v, a, mode), us)
                check_faithful(probes, T == 0, model.dtype == torch.float32)
            rows = []
            for (i, q, c, gt, order), p, g, pr in zip(metas, prompts, gens, probes):
                s = ctx.split(SimpleNamespace(token_ids=g["ids"], finish_reason=g["finish_reason"],
                                              logprobs=RowLogprobs(tok, g["vals"], g["tids"])))
                lps = digit_logprobs(s["top"], [1, 2, 3, 4, 5])
                end = {k: float(g["hidden"][k] @ u) for k, u in us.items()}
                ans = pr["score"] if pr else {}
                cols = ({f"probe_{w}_L{k}": d.get(k) for k in ks for w, d in (("prompt_end", end), ("answer_pos", ans))}
                        if a == 0 else dict(probe_layer=probe_L, probe_prompt_end=end.get(probe_L),
                                            probe_answer_pos=ans.get(probe_L)))
                rows.append(phase_row(2, i, q, c, gt, order, p, s["content"], lps, s, cfg["thinking"],
                                      dict(alpha=a, layer=L, position_mode=mode, vector=vec, temperature=T,
                                           steered_positions=g["steered"], **cols,
                                           probe_top1_match=pr and int(pr["top1"] == pr["ref_top1"]),
                                           probe_lp_diff=pr and pr["lp_diff"])))
            save_csv(path, rows)
    concat(run, "reason")


# ---- Modal ---------------------------------------------------------------------------------

def run_dir(cfg):
    return Path("/results") / f"steer-{cfg['model'].split('/')[-1]}-{cfg['tag']}"


@app.function(volumes={"/cache": cache, "/results": results}, secrets=[hf_secret], timeout=8 * 60 * 60,
              memory=65536, retries=modal.Retries(max_retries=1, initial_delay=10.0))
def stage(cfg: dict) -> str:
    import torch
    import transformers
    model, tok = load_model(cfg["model"])
    cache.commit()
    run = run_dir(cfg)
    run.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    {"notrace": run_notrace, "gate": run_gate, "reason": run_reason, "pretrace": run_pretrace}[cfg["stage"]](
        model, tok, run, cfg)
    name = cfg["stage"] if cfg["stage"] != "reason" else f"reason_L{cfg['layer']}_{cfg['vector']}_T{cfg['temperature']:g}"
    (run / f"{name}_params.json").write_text(json.dumps(
        {**cfg, "minutes": round((time.time() - t0) / 60, 1), "torch": torch.__version__,
         "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name()}, indent=2))
    results.commit()
    return f"{cfg['stage']} done -> {RESULTS_VOLUME}/{run.name}"


def spawn(gpu, **cfg):
    # spawn, don't wait: a client blocked on .remote() cancels the run if it crashes (see abstention_modal.main)
    call = stage.with_options(gpu=gpu).spawn(cfg)
    print(f"spawned {cfg['stage']} steer-{cfg['model'].split('/')[-1]}-{cfg['tag']}: {call.object_id}")


@app.local_entrypoint()
def notrace(model: str, thinking: str, tag: str = "main", n: int = 1000, gpu: str = "H200", batch: int = 32,
            questions: str = "popqa", answer_prefix: str = ""):
    assert not answer_prefix or thinking == "native", "--answer-prefix follows the force-closed think block"
    spawn(gpu, stage="notrace", model=model, thinking=thinking, tag=tag, n=n, batch=batch, questions=questions,
          answer_prefix=answer_prefix)


@app.local_entrypoint()
def gate(model: str, thinking: str, tag: str = "main", gpu: str = "H200", batch: int = 32, questions: str = "popqa",
         answer_prefix: str = ""):
    assert not answer_prefix or thinking == "native", "--answer-prefix follows the force-closed think block"
    spawn(gpu, stage="gate", model=model, thinking=thinking, tag=tag, batch=batch, questions=questions,
          answer_prefix=answer_prefix)


@app.local_entrypoint()
def pretrace(model: str, thinking: str, harness_tag: str, tag: str = "main", gpu: str = "H200", batch: int = 32,
             questions: str = "popqa"):
    assert thinking in ("native", "on"), "the pre-trace position needs a thinking checkpoint"
    spawn(gpu, stage="pretrace", model=model, thinking=thinking, tag=tag, harness_tag=harness_tag,
          harness=f"/results/{harness_tag}", batch=batch, questions=questions)


@app.local_entrypoint()
def reason(model: str, thinking: str, layer: int, tag: str = "main", alphas: str = "-2,-1,0,1,2",
           positions: str = "think,start,post", vector: str = "real", temperature: float = 0.0, seed: int = 42,
           n: int = 300, budget: int = 4096, gpu: str = "H200", batch: int = 32, questions: str = "popqa"):
    assert thinking in ("native", "on"), "reasoning mode needs a thinking checkpoint"
    spawn(gpu, stage="reason", model=model, thinking=thinking, tag=tag, layer=layer,
          alphas=[int(a) for a in alphas.split(",")], positions=positions.split(","), vector=vector,
          temperature=temperature, seed=seed, n=n, budget=budget, batch=batch, questions=questions)
