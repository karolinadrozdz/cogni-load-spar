"""Phase 1: the keep-track run.

Three readout positions per stream (DECISIONS.md D15):

- **in_stream** — the comma after the second-to-last word. Inside the list, so
  the model is still disposed to continue it; every target has been seen; no
  target is privileged and the query is not yet known. This is where capacity
  and selectivity are measured.
- **stream_end** — the final `.`, just outside the list. Kept as a contrast:
  the model is disposed to continue the *instruction* here, not the list.
- **answer** — the `Answer:` prefill token. A retrieval measure: the queried
  target is about to be emitted.

Three forward passes per stream: J-lens readout, logit-lens readout, Q2
generation. Q1 needs none of its own — `lens.apply` returns the model's logits
at the same positions.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from cogniload import align, bands, exemplars, generate, prompts, readout, registry, scoring, stimuli

#: Words from the same categories that are NOT in this stream. Without them
#: "present" has no floor: band-min rank over several layers is a generous
#: criterion and base rates vary a lot across the pool ("red" and "car" are not
#: comparable to "awl" and "vise").
N_ABSENT = 8


def _absent_words(pool: dict[str, list[str]], stream: stimuli.Stream, n: int) -> list[str]:
    used = set(stream.words)
    candidates = sorted(w for words in pool.values() for w in words if w not in used)
    return random.Random(stream.stream_id).sample(candidates, min(n, len(candidates)))


def run_stream(
    model, lens, spec: registry.ModelSpec, stream: stimuli.Stream,
    pool: dict[str, list[str]], layers: list[int], band: tuple[int, int],
    max_new_tokens: int,
) -> tuple[list[dict], dict]:
    """Measure one stream. Returns (readout rows, summary row).

    `layers` is the recorded range; `band` marks which of them are in-band, so
    the analysis can take band-min over the band while still seeing the rest.
    """
    queried = stream.queried_category
    text = prompts.build_q1(
        model.tokenizer, stream.words, stream.tracked, queried,
        enable_thinking=spec.enable_thinking,
    )
    # in_stream: the comma after the second-to-last word, so the readout sits
    # inside the list where the model is still disposed to continue it. Every
    # target has been seen by then -- tail_guard puts only distractors last.
    # stream_end: the final `.`, outside the list, kept as the contrast.
    named = {
        "in_stream": align.delimiter_after(model.tokenizer, text, stream.words[-2]),
        "stream_end": align.delimiter_after(model.tokenizer, text, stream.words[-1]),
        "answer": model.encode(text).shape[-1] - 1,
    }
    positions = list(named.values())
    at_position = {v: k for k, v in named.items()}

    absent = _absent_words(pool, stream, N_ABSENT)
    labels = [(w, c) for c, forms in exemplars.CATEGORY_LABELS.items() for w in forms]
    words = (stream.words + absent + [w for w, _ in labels]
             + exemplars.ABSENT_LABELS)
    token_ids = [readout.single_token_id(model.tokenizer, w) for w in words]

    jlens = readout.read(model, lens, text, token_ids, layers=layers, positions=positions)
    logit = readout.read(model, lens, text, token_ids, layers=layers,
                         positions=positions, use_jacobian=False)

    # Q1 from the same pass as the readout, so answer and ranks agree.
    answer_row = positions.index(named["answer"])
    q1 = scoring.score_q1(model.tokenizer, jlens.model_logits[answer_row],
                          stream.targets[queried])

    q2_text = generate.free_text(
        model,
        prompts.build_q2(model.tokenizer, stream.words, stream.tracked,
                         enable_thinking=spec.enable_thinking),
        max_new_tokens=max_new_tokens,
    )
    q2 = scoring.score_q2(q2_text, stream.tracked, stream.targets, stream.words)

    meta = {r["word"]: r for r in stream.to_rows()}
    blank = {"stream_id": stream.stream_id, "c_t": stream.c_t,
             "queried_category": queried, "stream_pos": None, "recency": None,
             "category": None, "role": "absent", "replaced_at": None,
             "is_category_final": False, "is_queried": False}
    # Labels carry the same contrast the exemplars do, one level up: the
    # category is named in the instruction or it is not.
    for word, category in labels:
        meta[word] = {**blank, "word": word, "category": category,
                      "role": "label_tracked" if category in stream.tracked
                      else "label_untracked",
                      "is_queried": category == queried}
    for word in exemplars.ABSENT_LABELS:
        meta[word] = {**blank, "word": word, "role": "label_absent"}
    absent_meta = blank
    q2_by_category = {r["category"]: r for r in q2}

    rows = []
    for lens_name, result in (("jlens", jlens), ("logit", logit)):
        for row in result.to_rows(words=words, lens=lens_name):
            info = meta.get(row["word"], {**absent_meta, "word": row["word"]})
            row.update(info)
            row["readout_at"] = at_position[row["token_pos"]]
            row["in_band"] = band[0] <= row["layer"] < band[1]
            q2_row = q2_by_category.get(info["category"]) if info["role"] == "target" else None
            row["q2_parsed"] = q2_row["parsed"] if q2_row else None
            row["q2_correct"] = q2_row["correct"] if q2_row else None
            rows.append(row)

    summary = {
        "stream_id": stream.stream_id, "c_t": stream.c_t, "seed": stream.seed,
        "queried_category": queried, **{f"{k}_pos": v for k, v in named.items()},
        "n_prompt_tokens": named["answer"] + 1,
        "q2_parse_rate": sum(r["parsed"] for r in q2) / len(q2),
        "q2_accuracy": sum(r["correct"] for r in q2) / len(q2),
        "q2_raw": q2_text,
        "absent_words": " ".join(absent),
        **{f"q1_{k}": v for k, v in q1.to_row().items()},
    }
    return rows, summary


def run(spec: registry.ModelSpec, config: dict, results_dir: Path, *,
        limit: int | None = None, force: bool = False) -> None:
    band = bands.require(spec.alias, results_dir)
    spec = registry.resolve(spec.alias, results_dir=results_dir)
    settings = config["phase1"]
    out = results_dir / spec.alias
    out.mkdir(parents=True, exist_ok=True)

    lo_frac, hi_frac = config["readout"]["record_layer_fraction"]
    layers = [
        l for l in range(int(lo_frac * spec.n_layers), int(hi_frac * spec.n_layers))
        if l < spec.n_lens_matrices
    ]
    if not set(spec.band_layers()) <= set(layers):
        raise ValueError(
            f"band {band} is not inside the recorded range "
            f"{layers[0]}-{layers[-1]}; widen readout.record_layer_fraction"
        )

    model, lens = registry.load(spec)
    pool = exemplars.build(model.tokenizer, spec.alias,
                           n_per_category=settings["stream"]["n_exemplars_per_category"])

    n_streams = limit or settings["n_streams"]
    print(f"recording layers {layers[0]}-{layers[-1]}, band {band}; "
          f"{n_streams} streams per C_t")

    for c_t in settings["c_ts"]:
        shard = out / f"readout_ct{c_t}{'_limit' if limit else ''}.parquet"
        if shard.exists() and not force:
            print(f"  C_t={c_t}: {shard.name} exists, skipping")
            continue

        rows, summaries = [], []
        streams = stimuli.generate(
            pool, c_t=c_t, n_streams=n_streams, seed=config["seed"] + 1000 * c_t,
            updates_per_tracked=settings["stream"]["updates_per_tracked"],
            tail_guard=settings["stream"]["tail_guard"],
        )
        for stream in streams:
            stream_rows, summary = run_stream(
                model, lens, spec, stream, pool, layers, band,
                config["generation"]["q2_max_new_tokens"],
            )
            rows.extend(stream_rows)
            summaries.append(summary)

        pd.DataFrame(rows).to_parquet(shard, index=False)
        pd.DataFrame(summaries).to_parquet(
            out / shard.name.replace("readout", "summary"), index=False
        )
        def mean(key):
            return sum(s[key] for s in summaries) / len(summaries)

        ranks = [s["q1_expected_rank"] for s in summaries if s["q1_expected_rank"] > 0]
        top5 = sum(r <= 5 for r in ranks) / len(ranks) if ranks else float("nan")
        print(f"  C_t={c_t}: {len(rows)} rows, Q1 top-1 {mean('q1_correct'):.0%} / "
              f"top-5 {top5:.0%}, Q2 {mean('q2_accuracy'):.0%} "
              f"(parse {mean('q2_parse_rate'):.0%})")

    _manifest(spec, config, out)


def _manifest(spec: registry.ModelSpec, config: dict, out: Path) -> None:
    import json
    (out / "manifest.json").write_text(json.dumps({
        **spec.provenance(),
        "prompt_set_sha": prompts.prompt_set_sha(),
        "exemplar_pools_sha": exemplars.pools_sha(),
        "config": config,
    }, indent=2))
