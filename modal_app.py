"""Run run_batch.py's jobs on Modal, one GPU per job, all at once; results come back to results/batch/.

    modal run modal_app.py::setup                     # once: SmolLM2-360M, Qwen3-1.7B-Base and WikiText-2 into the opgraph-hf volume
    modal run --detach modal_fetch.py --text          # once, for the text mix
    modal run --detach modal_fetch.py --models google/gemma-4-E2B-it
    modal run --detach modal_app.py --jobs ccSensSmoke
    modal run --detach modal_app.py --jobs ccMG4E2Noise,ccMG4E2Spot,ccMG4E2EmbKv

SmolLM2-360M jobs run on an L40S, larger ones on an H100, H200 or B200, all with 8 CPU cores, since the LDLQ sweeps run on the
CPU. The project directory is copied into each container. The Hugging Face cache lives on a volume mounted where the code
expects it (~/.cache/huggingface). Each job's JSON and log are also kept on the volume in opgraph_results/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
REMOTE = "/root/opgraph"
HF = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "transformers==5.16.1", "datasets==4.6.1", "safetensors==0.8.0", "numpy", "pandas", "pyarrow", "huggingface_hub")
    .env({"OPGRAPH_DEVICE": "cuda", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
          "OPGRAPH_THREADS": "8", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
    .add_local_dir(ROOT, REMOTE, ignore=["site/**", "site2/**", "**/__pycache__/**", ".git/**", "cache/*.pt"])
)
hf = modal.Volume.from_name("opgraph-hf", create_if_missing=True)
app = modal.App("opgraph", image=image)


@app.function(volumes={HF: hf}, timeout=3600, cpu=4.0, memory=16384)
def setup():
    """The models every job needs at import, and WikiText-2, into the volume (the containers have internet here)."""
    import os
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    for repo in ("HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen3-1.7B-Base"):
        print("model", repo, snapshot_download(repo), flush=True)
    load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    hf.commit()
    print("volume ready", flush=True)


def _run(name: str) -> dict:
    """Run one job in /root/opgraph and return its result file."""
    import os
    import subprocess
    import sys
    sys.path.insert(0, REMOTE)
    os.chdir(REMOTE)
    import run_batch
    spec = run_batch.JOBS[name]
    env = dict(os.environ, OPGRAPH_ARCH=spec["arch"], PYTHONUNBUFFERED="1")
    t0 = time.time()
    p = subprocess.run([sys.executable, "run_batch.py", "--job", name], env=env, capture_output=True, text=True)
    out = Path(REMOTE) / "results" / "batch" / f"{name}.json"
    # Also keep the result and the whole log on the volume: a detached run whose local client went away still leaves them there
    keep = Path(HF) / "opgraph_results"
    keep.mkdir(parents=True, exist_ok=True)
    (keep / f"{name}.log").write_text(p.stdout + p.stderr)
    if out.exists():
        (keep / f"{name}.json").write_text(out.read_text())
    hf.commit()
    return {"job": name, "returncode": p.returncode, "seconds": round(time.time() - t0),
            "result": json.loads(out.read_text()) if out.exists() else None,
            "log": (p.stdout + p.stderr)[-6000:]}


@app.function(gpu="L40S", cpu=8.0, memory=32768, timeout=3 * 3600, volumes={HF: hf})
def run_small(name: str) -> dict:
    return _run(name)


@app.function(gpu="H100", cpu=8.0, memory=65536, timeout=6 * 3600, volumes={HF: hf})
def run_large(name: str) -> dict:
    return _run(name)


@app.function(gpu="H200", cpu=8.0, memory=131072, timeout=8 * 3600, volumes={HF: hf})
def run_xlarge(name: str) -> dict:
    return _run(name)


@app.function(gpu="B200", cpu=8.0, memory=131072, timeout=23 * 3600, volumes={HF: hf})
def run_b200(name: str) -> dict:
    return _run(name)


@app.function(volumes={HF: hf}, timeout=3600, cpu=4.0, memory=16384)
def download_model(repo: str):
    """One more model into the volume: config, tokenizer and safetensors weights only (no duplicate .bin copies)."""
    import os
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import list_repo_files, snapshot_download
    files = list_repo_files(repo)
    weights = ["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["*.bin"]
    path = snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.model"] + weights)
    hf.commit()
    print("model", repo, path, sorted(os.listdir(path)), flush=True)


@app.local_entrypoint()
def main(jobs: str, force: bool = False):
    import sys
    sys.path.insert(0, str(ROOT))
    import run_batch
    names = [j for j in jobs.split(",") if j]
    if not names:
        raise SystemExit("no jobs: pass --jobs a,b")
    unknown = [n for n in names if n not in run_batch.JOBS]
    if unknown:
        raise SystemExit(f"unknown jobs: {unknown}")
    out = ROOT / "results" / "batch"
    out.mkdir(parents=True, exist_ok=True)
    if not force:  # a job that already has a result is not paid for twice (--force reruns it)
        done = [n for n in names if (out / f"{n}.json").exists()]
        if done:
            print(f"skipping {len(done)} jobs with results: {', '.join(done)}", flush=True)
        names = [n for n in names if n not in done]
    if not names:
        return
    print(f"launching {len(names)} jobs: {', '.join(names)}", flush=True)
    runner = {"small": run_small, "large": run_large, "xlarge": run_xlarge, "b200": run_b200}
    calls = {n: runner[run_batch.JOBS[n]["gpu"]].spawn(n) for n in names}
    for n, call in calls.items():
        r = call.get()
        if r["result"]:
            (out / f"{n}.json").write_text(json.dumps(r["result"], indent=1))
            print(f"[done {r['seconds']}s] {n}", flush=True)
        else:
            print(f"[FAILED {n}, exit {r['returncode']}] {r['log'][-1500:]}", flush=True)
