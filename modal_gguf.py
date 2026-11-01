"""gguf_compiler.py on Modal: llama.cpp built with CUDA (the commit is recorded in /llama.cpp/COMMIT and in each
model's manifest), transformers 5.16 for the measurements, the opgraph directory beside it.

    modal run --detach modal_gguf.py --steps "prepare,measure" --models unsloth/Llama-3.2-3B,microsoft/Phi-4-mini-instruct
    modal run --detach modal_gguf.py --steps evaluate --models unsloth/Llama-3.2-3B --files stock_Q4_K_M.gguf,plan_x.gguf
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
HF = "/root/.cache/huggingface"
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "cmake", "build-essential", "libcurl4-openssl-dev")
    .run_commands("git clone https://github.com/ggml-org/llama.cpp /llama.cpp && cd /llama.cpp && git rev-parse HEAD > COMMIT",
                  "cd /llama.cpp && cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES='80;89;90' -DLLAMA_CURL=OFF"
                  " -DLLAMA_BUILD_TESTS=OFF -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined"  # no driver at build time (llama.cpp's Dockerfile)
                  " && cmake --build build --config Release -j 16 --target llama-quantize llama-perplexity llama-imatrix")
    .pip_install("torch==2.14.0", "transformers==5.16.1", "datasets==4.6.1", "safetensors==0.8.0", "numpy", "sentencepiece",
                 "protobuf", "mistral-common", "compressed-tensors")
    .run_commands("pip install /llama.cpp/gguf-py")
    .env({"OPGRAPH_DEVICE": "cuda", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT, "/root/opgraph", ignore=["cache/**", "site/**", "**/__pycache__/**", ".git/**", "results/batch/**"])
)
hf = modal.Volume.from_name("opgraph-hf")
app = modal.App("opgraph-gguf", image=image)


@app.function(gpu="H100", cpu=8.0, memory=131072, timeout=8 * 3600, volumes={HF: hf}, ephemeral_disk=524288)
def run(args: list[str]) -> dict:
    import subprocess
    import sys
    script = {"public": "public_eval.py", "script": args[1] if len(args) > 1 else ""}.get(args[0], "gguf_compiler.py")
    rest = args[2:] if args[0] == "script" else args[1:] if args[0] == "public" else args
    p = subprocess.run([sys.executable, script, *rest], cwd="/root/opgraph", capture_output=True, text=True)
    log = p.stdout + p.stderr
    name = "_".join(a.replace("/", "-") for a in args[:3])
    keep = Path(HF) / "opgraph_results" / f"gguf_{name}.log"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_text(log)
    hf.commit()
    return {"args": args, "returncode": p.returncode, "log": log[-5000:]}


@app.function(gpu="B200", cpu=8.0, memory=262144, timeout=10 * 3600, volumes={HF: hf}, ephemeral_disk=524288)
def run_big(args: list[str]) -> dict:
    return run.local(args)


@app.local_entrypoint()
def main(steps: str, models: str, files: str = "", plans: str = ""):
    import json
    out = ROOT / "results" / "gguf"
    out.mkdir(parents=True, exist_ok=True)

    def chain(model):  # the steps for one model, in order, each on its own container
        tag = model.split("/")[-1]
        for step in steps.split(","):
            a = [step, "--model", model]
            if step == "measure":
                a += ["--out", f"/root/.cache/huggingface/opgraph_results/gguf_sens_{tag}.json"]
            if step == "build":
                a += ["--sens", f"/root/.cache/huggingface/opgraph_results/gguf_sens_{tag}.json"]
                for pl in plans.split(","):
                    yield a + ["--plan", f"results/compiler/plans/{pl}.json"]
                continue
            if step == "evaluate":
                a += ["--files", files, "--out", f"/root/.cache/huggingface/opgraph_results/gguf_eval_{tag}.json"]
            yield a
    import itertools
    queues = [list(chain(m)) for m in models.split(",")]
    for wave in itertools.zip_longest(*queues):  # models in parallel, steps in sequence
        calls = [run.spawn(a) for a in wave if a]
        for c in calls:
            r = c.get()
            print(f"[{'done' if r['returncode'] == 0 else 'FAILED'} {' '.join(r['args'][:3])}]\n{r['log'][-1500:]}", flush=True)


@app.local_entrypoint()
def public(model: str, checkpoints: str, dtype: str = "float32", big: bool = False, prepare: bool = True, tasks: str = "", out: str = ""):
    """public_eval.py for one model (after gguf_compiler.py prepare, which makes our llama.cpp quants and the bf16 GGUF):
    modal run --detach modal_gguf.py::public --model google/gemma-4-E4B-it --checkpoints gguf:google/gemma-4-E4B-it-qat-q4_0-gguf/*.gguf"""
    fn = run_big if big else run
    tag = model.split("/")[-1]
    if prepare:
        r = fn.remote(["prepare", "--model", model, "--skip-pure"])
        print(f"[{'done' if r['returncode'] == 0 else 'FAILED'} prepare {model}]\n{r['log'][-2500:]}", flush=True)
    r = fn.remote(["public", "--model", model, "--dtype", dtype, "--checkpoints", checkpoints, *(["--tasks", tasks] if tasks else []),
                   "--out", f"/root/.cache/huggingface/opgraph_results/{out or 'public_' + tag}.json"])
    print(f"[{'done' if r['returncode'] == 0 else 'FAILED'} public {model}]\n{r['log'][-4000:]}", flush=True)


@app.local_entrypoint()
def inspect(files: str):
    """modal run modal_gguf.py::inspect --files 'models--google--gemma-4-E2B-it-qat-q4_0-gguf/snapshots/*/*.gguf'"""
    r = run.remote(["inspect", "--model", "-", "--files", files])
    print(r["log"][-6000:])


@app.local_entrypoint()
def script(path: str, args: str = ""):
    """modal run modal_gguf.py::script --path a_script.py"""
    r = run.remote(["script", path, *args.split()])
    print(r["log"][-8000:])
