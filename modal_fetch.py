"""Downloads for the compression compiler's model coverage, into the opgraph-hf volume:
the test models (Hugging Face snapshots, safetensors only) and a text mix for token frequencies, calibration and
evaluation beyond English: FineWeb-Edu (English), FineWeb-2 in ten languages, Python (codeparrot-clean) and chat
(UltraChat). Each text source is kept as documents in compiler_text/<name>.json; the first 70% calibrate and rank
tokens, the last 30% evaluate.

    modal run --detach modal_fetch.py --models allenai/OLMoE-1B-7B-0924,unsloth/Llama-3.2-3B
    modal run --detach modal_fetch.py --text
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HF = "/root/.cache/huggingface"
image = modal.Image.debian_slim(python_version="3.12").pip_install("datasets==4.6.1", "huggingface_hub")
hf = modal.Volume.from_name("opgraph-hf")
app = modal.App("opgraph-fetch", image=image)

LANGS = ["cmn_Hani", "jpn_Jpan", "kor_Hang", "rus_Cyrl", "spa_Latn", "fra_Latn", "deu_Latn", "por_Latn", "arb_Arab", "hin_Deva"]
SOURCES = {"en": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 6e6),
           **{l: ("HuggingFaceFW/fineweb-2", l, "train", "text", 3e6) for l in LANGS},
           "code": ("codeparrot/codeparrot-clean", None, "train", "content", 6e6),
           "chat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft", "messages", 6e6)}


@app.function(volumes={HF: hf}, timeout=3 * 3600, cpu=8.0, memory=32768)
def fetch_model(repo: str) -> str:
    """repo, or repo::pattern|pattern (only those files, e.g. one GGUF of a repo with many)."""
    from huggingface_hub import snapshot_download
    repo, _, pats = repo.partition("::")
    allow = pats.split("|") + ["*.json", "tokenizer*"] if pats else ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*"]
    path = snapshot_download(repo, allow_patterns=allow, ignore_patterns=["original/*", "metal/*", "*.pth"])
    hf.commit()
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return f"{repo}: {path} ({size / 1e9:.1f} GB)"


@app.function(volumes={HF: hf}, timeout=2 * 3600, cpu=4.0, memory=16384)
def fetch_text() -> str:
    from datasets import load_dataset
    out = Path(HF) / "compiler_text"
    out.mkdir(exist_ok=True)
    lines = []
    for name, (repo, cfg, split, field, chars) in SOURCES.items():
        ds = load_dataset(repo, cfg, split=split, streaming=True)
        docs, n = [], 0
        for row in ds:
            v = row[field]
            if field == "messages":  # a conversation as plain turns
                v = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in v)
            if not v or len(v) < 200:
                continue
            docs.append(v[:20000])
            n += len(docs[-1])
            if n >= chars:
                break
        (out / f"{name}.json").write_text(json.dumps({"repo": repo, "config": cfg, "split": split, "docs": docs}, ensure_ascii=False))
        lines.append(f"{name}: {len(docs)} documents, {n / 1e6:.1f}M characters")
        print(lines[-1], flush=True)
    hf.commit()
    return "\n".join(lines)


@app.local_entrypoint()
def main(models: str = "", text: bool = False):
    calls = [fetch_model.spawn(r) for r in models.split(",") if r]
    t = fetch_text.spawn() if text else None
    for c in calls:
        print(c.get(), flush=True)
    if t is not None:
        print(t.get(), flush=True)
