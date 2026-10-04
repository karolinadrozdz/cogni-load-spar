"""Modal entrypoint.

    modal run modal_app.py --stage phase0
    modal run modal_app.py --stage phase1 --limit 20
    modal run modal_app.py --stage analyze

Two volumes persist across runs: model weights and lenses in an HF cache (so
the 54 GB prod download happens once), and all results. Everything else is
rebuilt from the image.

GPU follows the registry alias — `dev` (Qwen3.5-4B, ~8 GB) does not need an
H100. Override with --gpu.
"""

import modal

REPO = "https://github.com/anthropics/jacobian-lens"
JLENS_SHA = "581d398613e5602a5af361e1c34d3a92ea82ba8e"  # pinned, as in pyproject

image = (
    modal.Image.debian_slim(python_version="3.11")
    # build-essential, not just git: transformers routes Qwen3.5's rotary
    # embedding through a Triton kernel, and Triton JIT-compiles its CUDA
    # driver shim on first use. debian_slim has gcc but no binutils, so that
    # compile fails with "collect2: cannot find 'ld'".
    .apt_install("git", "build-essential")
    .pip_install(
        f"jlens @ git+{REPO}@{JLENS_SHA}",
        "transformers>=5.5",
        "torch>=2.5",
        "accelerate",          # device_map="auto"
        "pyyaml>=6",
        "pandas>=2.2",
        "pyarrow>=17",
        "numpy>=1.26",
        "scipy>=1.14",
    )
    .env({
        "HF_HOME": "/cache/hf",
        # Persist Triton's compiled kernels, so only the first cold start pays.
        "TRITON_CACHE_DIR": "/cache/triton",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir("src", "/repo/src")
    .add_local_dir("configs", "/repo/configs")
)

app = modal.App("cogni-load-spar", image=image)
cache = modal.Volume.from_name("cogniload-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("cogniload-results", create_if_missing=True)

#: Qwen weights and the Neuronpedia lenses are public, so no HF token is
#: needed. Add `secrets=[modal.Secret.from_name("huggingface-token")]` to the
#: function below if a future alias points at a gated model.
GPU_FOR_ALIAS = {"dev": "L4", "prod": "H100"}


@app.function(
    volumes={"/cache": cache, "/results": results},
    timeout=4 * 60 * 60,
)
def run_stage(stage: str, model: str, limit: int | None, force: bool) -> str:
    """Run one stage and commit the volumes.

    Declared with no GPU; `main` attaches one for the stages that need it, so a
    GPU is only ever added, never ambiguously removed.
    """
    import os
    import subprocess
    import sys

    argv = [
        sys.executable, "-m", "cogniload.cli", stage,
        "--config", "/repo/configs/exp1.yaml",
        "--out-dir", "/results",
        "--model", model,
    ]
    if limit:
        argv += ["--limit", str(limit)]
    if force:
        argv += ["--force"]

    # Overlay on os.environ, never replace it: a bare env= drops PATH, and
    # Triton's JIT then fails to find `ld` when it compiles its CUDA shim.
    env = {**os.environ, "PYTHONPATH": "/repo/src"}
    # Inherit stdout/stderr rather than capturing, so Modal streams progress
    # live instead of going silent for the length of a 150-stream sweep.
    completed = subprocess.run(argv, cwd="/repo", env=env)
    # Commit before raising, so a crash mid-sweep keeps the shards already written.
    results.commit()
    cache.commit()
    if completed.returncode:
        raise RuntimeError(f"{stage} failed with code {completed.returncode}")
    return f"{stage} ok"


@app.local_entrypoint()
def main(
    stage: str = "phase0",
    model: str = "dev",
    limit: int = 0,
    force: bool = False,
    gpu: str = "",
):
    """Dispatch one stage. Run from the repo root — the image picks up `src/`
    and `configs/` relative to the working directory."""
    if stage in ("analyze",):
        print("analyze reads parquet only; no GPU")
        fn = run_stage
    else:
        chosen = gpu or GPU_FOR_ALIAS.get(model, "H100")
        print(f"{stage} on {chosen} (alias {model})")
        fn = run_stage.with_options(gpu=chosen)
    print(fn.remote(stage, model, limit or None, force))
