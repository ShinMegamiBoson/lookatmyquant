"""Rebuild the packed file (opgraph-packed-1, read by runtime_mlx/fast_q35.py) of a build that was saved only as dense
weights, on a Modal CPU container. Nothing on the volume is changed or removed: the packed file goes to a new folder.

    modal run runtime_mlx/modal_pack.py::inspect                 # what the saved codes and the dense file hold
    modal run runtime_mlx/modal_pack.py::one --name <matrix>     # one matrix: time and exactness
    modal run --detach runtime_mlx/modal_pack.py::pack           # the whole file, then reference logits

    modal volume get opgraph-hf lm2/packed <local folder>
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/root/pack"
HF = "/root/.cache/huggingface"
CODES, DENSE = f"{HF}/lm2/build/codes.pt", f"{HF}/lm2/tuned/dense.pt"
MODEL = f"{HF}/models/Qwen3.5-4B"
OUT = f"{HF}/lm2/packed"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "transformers==5.16.1", "datasets==4.6.1", "safetensors==0.8.0", "numpy")
    .env({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "opgraph", f"{REMOTE}/opgraph", ignore=["**/__pycache__/**"])
    .add_local_file(ROOT / "compiler_artifact.py", f"{REMOTE}/compiler_artifact.py")
    .add_local_file(ROOT / "runtime_mlx" / "pack_dense.py", f"{REMOTE}/pack_dense.py")
)
hf = modal.Volume.from_name("opgraph-hf")
app = modal.App("lamq-pack", image=image)


def _enter():
    import sys
    sys.path.insert(0, REMOTE)


@app.function(volumes={HF: hf}, timeout=1200, cpu=4.0, memory=24576)
def inspect():
    _enter()
    import pack_dense
    pack_dense.inspect(CODES, DENSE, MODEL)


@app.function(volumes={HF: hf}, timeout=1800, cpu=8.0, memory=49152)
def one(name: str = ""):
    _enter()
    import pack_dense
    pack_dense.one(CODES, DENSE, name)


@app.function(volumes={HF: hf}, timeout=3 * 3600, cpu=32.0, memory=131072)
def pack(workers: int = 28, reference: bool = True):
    _enter()
    import pack_dense
    pack_dense.pack(CODES, DENSE, MODEL, OUT, workers=workers, reference=reference)
    hf.commit()


@app.function(volumes={HF: hf}, timeout=1200, cpu=8.0, memory=32768)
def compact():
    _enter()
    import pack_dense
    pack_dense.compact_reference(OUT)
    hf.commit()


@app.function(volumes={HF: hf}, timeout=600, cpu=2.0, memory=4096)
def piece(offset: int, nbytes: int):
    """A byte range of the packed weights as its own small file (to repair a download without fetching it all again)."""
    with open(f"{OUT}/weights.bin", "rb") as f:
        f.seek(offset)
        raw = f.read(nbytes)
    with open(f"{OUT}/piece_{offset}_{nbytes}.bin", "wb") as f:
        f.write(raw)
    hf.commit()
