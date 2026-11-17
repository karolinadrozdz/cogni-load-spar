# CFC launch commands (not executed yet)

Spec: `tasks/cfc/PLAN.md`. Run from `experiments/confidence-abstention/` with the project's Modal profile active
(`modal profile activate spar-f26`). Launch about 5 at a time: there is a GPU limit, and HF throttles parallel
downloads (TASKS.md).

Each run needs its reference run (`ref_tag`, same checkpoint) and the same-stage sibling's (`cross_ref_tag`) in the
`abstention-results` volume; the Think references need their `start_format.csv`. The partner map is fixed
(Instruct SFT ↔ Think SFT main, Instruct DPO ↔ Think DPO, Instruct RL ↔ Think RL 3.1 main); a wrong pairing, a model
that is not the reference's, or (Instruct) a dtype other than the reference's fails at launch or before generation.

## 0. Before launching

```
python tasks/cfc/build_pairs.py --check          # pairs_data.py is a fresh build of the pair set
python tasks/cfc/check_local.py --ref <copies of the 6 reference dirs> --tpl <template cache> --out <scratch>
modal run abstention_modal.py::check_tasks       # CPU, seconds: tasks.cfc imports in a run container
```
Pair set hash (pinned with `pairs_sha` below): `248b80df06b4fc8317b043636454753a028e9e805ae7c0a3e45a9ced5e9e8046`.

## 1. Pilots (200 pairs each, all three conditions)

```
modal run --detach abstention_modal.py::main --task cfc --questions popqa --gpu H200 \
    --model allenai/Olmo-3.1-32B-Instruct-DPO --thinking off --dtype bfloat16 \
    --task-opts "ref_tag=olmo32b-popqa-instruct-dpo,cross_ref_tag=olmo32b-popqa-think-dpo,n_pairs=200,pairs_sha=248b80df06b4" \
    --tag cfc-pilot-instruct-dpo
modal run --detach abstention_modal.py::main --task cfc --questions popqa --gpu H200 \
    --model allenai/Olmo-3.1-32B-Instruct-SFT --thinking off --dtype bfloat16 \
    --task-opts "ref_tag=olmo32b-popqa-instruct-sft,cross_ref_tag=olmo32b-popqa-think-sft-main,n_pairs=200,pairs_sha=248b80df06b4" \
    --tag cfc-pilot-instruct-sft
modal run --detach abstention_modal.py::main --task cfc --questions popqa --gpu H200 \
    --model allenai/Olmo-3-32B-Think-DPO --thinking native \
    --task-opts "ref_tag=olmo32b-popqa-think-dpo,cross_ref_tag=olmo32b-popqa-instruct-dpo,n_pairs=200,pairs_sha=248b80df06b4" \
    --tag cfc-pilot-think-dpo
```
Check in each `analysis.json` (fetch with `./handoff.sh <tag>` or `modal volume get abstention-results <tag> results/`):
- `P_ab_mass`, `S_ab_mass`, `X_ab_mass` ≥ 0.9 and `*_share_first_letter` near 1 (A/B in the top-20 as the first token);
  `*_share_first_question` / `*_share_first_digit` near 0;
- `*_position_bias` (mean P(first label)) not extreme, `*_ties` small;
- `params.json`: `rendered_P` / `rendered_X` (Think: ends `<think>\n</think>\n\nAnswer:`), `label_tokens` (the letter
  after the prefix should be the single token `" A"` / `"A"`), `skipped_pairs`, no `score_error`.
Go/no-go (PLAN.md, fixed before the pilots): **GO** if A/B mass ≥ 0.9 and position bias within 0.2–0.8. If Instruct
SFT fails the format, report it as a format failure and run the rest without redesign. Stated accuracy at chance in a
pilot is a result, not a reason to stop.

## 2. Main runs (all 5829 pairs, P + S + X)

```
modal run --detach abstention_modal.py::main --task cfc --questions popqa --gpu H200 \
    --model allenai/Olmo-3.1-32B-Instruct-SFT --thinking off --dtype bfloat16 \
    --task-opts "ref_tag=olmo32b-popqa-instruct-sft,cross_ref_tag=olmo32b-popqa-think-sft-main,pairs_sha=248b80df06b4" \
    --tag cfc-instruct-sft
modal run --detach abstention_modal.py::main --task cfc --questions popqa --gpu H200 \
    --model allenai/Olmo-3.1-32B-Instruct-DPO --thinking off --dtype bfloat16 \
    --task-opts "ref_tag=olmo32b-popqa-instruct-dpo,cross_ref_tag=olmo32b-popqa-think-dpo,pairs_sha=248b80df06b4" \
    --tag cfc-instruct-dpo
modal run --detach abstention_modal.py::main --task cfc --questions popqa --gpu H200 \
    --model allenai/Olmo-3.1-32B-Instruct --thinking off \
    --task-opts "ref_tag=olmo32b-popqa-instruct-rl,cross_ref_tag=olmo32b-popqa-think-rl31-main,pairs_sha=248b80df06b4" \
    --tag cfc-instruct-rl
modal run --detach abstention_modal.py::main --task cfc --questions popqa --gpu H200 \
    --model allenai/Olmo-3-32B-Think-SFT --thinking native \
    --task-opts "ref_tag=olmo32b-popqa-think-sft-main,cross_ref_tag=olmo32b-popqa-instruct-sft,pairs_sha=248b80df06b4" \
    --tag cfc-think-sft-main
modal run --detach abstention_modal.py::main --task cfc --questions popqa --gpu H200 \
    --model allenai/Olmo-3-32B-Think-DPO --thinking native \
    --task-opts "ref_tag=olmo32b-popqa-think-dpo,cross_ref_tag=olmo32b-popqa-instruct-dpo,pairs_sha=248b80df06b4" \
    --tag cfc-think-dpo
modal run --detach abstention_modal.py::main --task cfc --questions popqa --gpu H200 \
    --model allenai/Olmo-3.1-32B-Think --thinking native \
    --task-opts "ref_tag=olmo32b-popqa-think-rl31-main,cross_ref_tag=olmo32b-popqa-instruct-rl,pairs_sha=248b80df06b4" \
    --tag cfc-think-rl31-main
```
Defaults that matter: `--n 1000` (required), `--max-model-len 8192` (prompts ≈ 170–350 tokens), greedy (the readout
is one token; `--temperature` must stay ≤ 0). `--budget` is unused.

## 3. Fetch and analyze

```
for t in cfc-instruct-sft cfc-instruct-dpo cfc-instruct-rl cfc-think-sft-main cfc-think-dpo cfc-think-rl31-main; do
    ./handoff.sh $t; done
python tasks/cfc/analyze_cfc.py results/cfc-instruct-sft results/cfc-instruct-dpo results/cfc-instruct-rl \
    results/cfc-think-sft-main results/cfc-think-dpo results/cfc-think-rl31-main --n-boot 10000 --json results/cfc_analysis.json
```
(`python tasks/cfc/analyze_cfc.py` with no arguments reads the same six `results/cfc-<key>` directories.)

## Cost

About $0.8–1.5 per run on one H200 (≈ $4.54/h): model loading dominates (≈ 7 min ≈ $0.55); the readout is one
token over ≈ 26k unique prompts of ≈ 300 tokens (≈ 34–35k before S/X deduplication; pilots ≈ 1k), a few minutes of
prefill. Pilots ≈ $2–3, main ≈ $5–9, ≈ $8–12 in all.
