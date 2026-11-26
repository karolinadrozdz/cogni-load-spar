# K2-Horizon confidence → abstention experiments

This repository is the reproducible snapshot of our current K2-Horizon-32B experiments on PopQA. It contains the code that generated the runs, the fixed question sets, Slurm launchers, analysis scripts, figures, and all completed aggregate results available on **2026-11-25**.

## Research question

We track confidence and abstention across K2-Horizon training checkpoints. The central comparison is:

1. how well calibrated/token confidence and verbal confidence distinguish correct from incorrect answers; and
2. how well each signal predicts whether the model later abstains.

The checkpoint sweep asks when selective abstention and thought rejection emerge during pretraining, bridge training, SFT, and native-thinking training.

## Snapshot contents

| Result group | Completed runs | Purpose |
|---|---:|---|
| `results/pilot/` | 1 | n=20 end-to-end gate |
| `results/stage-a/` | 5 | n=200 checkpoint probes |
| `results/runs/` | 26 | n=1000 Stage B runs: 18 base/bridge and 8 native-thinking |

Stage B currently contains 26 of the 29 planned checkpoints. Only completed runs are included; empty or interrupted output directories are excluded. See [results/SUMMARY.md](../../New%20project/k2h-confidence-abstention-release/results/SUMMARY.md) for the compact tables and [results/manifest.csv](../../New%20project/k2h-confidence-abstention-release/results/manifest.csv) for the machine-readable inventory.

The availability of item-level files from the older pilot and Stage A snapshot is recorded in [results/ARTIFACT_STATUS.md](../../New%20project/k2h-confidence-abstention-release/results/ARTIFACT_STATUS.md).

The consolidated tables cover all 32 completed runs. A small number of older Stage A item-level artifacts were iCloud-only at packaging time; the exact artifact inventory is recorded in [results/ARTIFACT_STATUS.md](../../New%20project/k2h-confidence-abstention-release/results/ARTIFACT_STATUS.md).

## Repository layout

```text
src/          experiment runner and TAU wrapper
slurm/        cluster setup, pilot, Stage A, Stage B, and recovery launchers
analysis/     statistical analysis and plotting code
config/       checkpoint revisions and sweep definitions
data/         fixed PopQA multiple-choice Phase 0/1 inputs
results/      raw CSV outputs, parameters, analyses, and result manifest
figures/      generated Stage A and thought-rejection figures
docs/         concise reports and experiment notice
environment/  exact package lock and source-repository commit
```

## Reproduction

The original runs used `IFM/K2-Horizon-32B`, Python 3.12, vLLM 0.32.0, PyTorch 2.13.0, deterministic decoding (`temperature=0`, seed 42), and the frozen PopQA inputs in `data/`.

On the TAU cluster, edit only the cluster-specific `ROOT`, partition, and account directives in the relevant launcher, then run:

```bash
sbatch slurm/k2h_setup.sbatch
sbatch slurm/k2h_stage_a_n200_a6000.sbatch
sbatch --array=0-14 slurm/k2h_stage_b_base_a6000.sbatch
sbatch --array=0-13 slurm/k2h_stage_b_think_h200.sbatch
```

The recovery runner is `src/k2h_slurm_recovered.py`; it executes the experiment logic from `src/abstention_modal.py` without changing the prompts, questions, or sampling settings. Result directories contain:

- `params.json`: full run configuration and runtime metadata;
- `phases.csv`: item-level answers, traces, probabilities, and abstention decisions;
- `verbal.csv`: item-level verbal-confidence outputs when applicable;
- `analysis.json`: aggregate metrics.

Regenerate the repository-wide tables with:

```bash
python3 scripts/build_manifest.py
```

## Data and provenance

PopQA preprocessing provenance is documented in [data/README-osf.md](../../New%20project/k2h-confidence-abstention-release/data/README-osf.md). The exact upstream source commit is recorded in `environment/source-repo-commit.txt`. Model weights and Hugging Face caches are intentionally not included.

See [docs/NOTICE.md](../../New%20project/k2h-confidence-abstention-release/docs/NOTICE.md) for the relationship to the reference implementation and prompt provenance.
