"""Modal entrypoint: one `cogniload.cli` stage on a GPU (no GPU for `report`).

    modal run modal_app.py --stage find_band --model dev
    modal run modal_app.py --stage confidence --stimuli-set regions --model dev --limit 5
    modal run modal_app.py --stage report --experiment <experiment> --model prod

Run from the repo root: the image picks up `src/` and `configs/` from there.
Two volumes persist: an HF cache (the 54 GB prod download happens once) and
the results.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    # build-essential, not just git: Triton JIT-compiles a CUDA shim for
    # Qwen3.5's rotary embedding, and debian_slim has no `ld`.
    .apt_install("git", "build-essential")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["gpu"])
    .env({
        "HF_HOME": "/cache/hf",
        # Persist Triton's compiled kernels, so only the first cold start pays.
        "TRITON_CACHE_DIR": "/cache/triton",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir("src", "/repo/src")
    .add_local_dir("configs", "/repo/configs")
    .add_local_dir("stimuli", "/repo/stimuli")
)

app = modal.App("cogni-load-spar", image=image)
cache = modal.Volume.from_name("cogniload-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("cogniload-results", create_if_missing=True)

# The models are public, so no HF token. A gated model would need
# `secrets=[modal.Secret.from_name("huggingface-token")]` below.
GPU_FOR_ALIAS = {"dev": "L4", "prod": "H100"}


@app.function(volumes={"/cache": cache, "/results": results}, timeout=4 * 60 * 60)
def run_stage(args: list[str]) -> str:
    """Run the CLI with `args`, then commit the volumes."""
    import os
    import subprocess
    import sys

    argv = [sys.executable, "-m", "cogniload.cli", *args,
            "--config", "/repo/configs/experiments.yaml", "--out-dir", "/results"]
    # Overlay os.environ: a bare env= drops PATH, and Triton then cannot find `ld`.
    # Output is inherited, not captured, so Modal streams progress live.
    completed = subprocess.run(argv, cwd="/repo", env={**os.environ, "PYTHONPATH": "/repo/src"})
    # Commit before raising, so a crash mid-sweep keeps the shards already written.
    results.commit()
    cache.commit()
    if completed.returncode:
        raise RuntimeError(f"{args} failed with code {completed.returncode}")
    return f"{args} ok"


@app.local_entrypoint()
def main(stage: str = "find_band", experiment: str = "", stimuli_set: str = "",
         model: str = "dev",
         limit: int = 0, force: bool = False, gpu: str = ""):
    args = [stage, *([experiment] if experiment else []), "--model", model,
            *(["--set", stimuli_set] if stimuli_set else []),
            *(["--limit", str(limit)] if limit else []),
            *(["--force"] if force else [])]
    # GPU follows the alias; `report` reads parquet only.
    fn = run_stage if stage == "report" else run_stage.with_options(
        gpu=gpu or GPU_FOR_ALIAS.get(model, "H100"))
    print(fn.remote(args))
