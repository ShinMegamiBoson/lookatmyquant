"""WikiText-2 token windows for calibration (train split) and evaluation (test split), read from the local HF cache."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch

CACHE = Path(__file__).resolve().parent.parent / "cache"


def token_ids(split: str) -> torch.Tensor:
    from .model import DATA_PREFIX
    path = CACHE / f"{DATA_PREFIX}wikitext2-{split}.pt"
    if path.exists():
        return torch.load(path)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from .model import snapshot_dir

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"])
    ids = AutoTokenizer.from_pretrained(snapshot_dir())(text, return_tensors="pt").input_ids[0]
    CACHE.mkdir(exist_ok=True)
    torch.save(ids, path)
    return ids


def calibration(n: int = 64, seqlen: int = 512, seed: int = 0) -> torch.Tensor:
    """Random windows from the train split, as GPTQ, AWQ and SliceGPT sample calibration data."""
    ids = token_ids("train")
    starts = torch.randint(0, len(ids) - seqlen - 1, (n,), generator=torch.Generator().manual_seed(seed))
    return torch.stack([ids[s : s + seqlen] for s in starts])


def evaluation(n: int = 64, seqlen: int = 512) -> torch.Tensor:
    """The first n non-overlapping windows of the test split."""
    return token_ids("test")[: n * seqlen].view(n, seqlen)


def gsm8k_ids(split: str) -> torch.Tensor:
    """GSM8K (grade-school maths word problems with worked answers) as one token stream: a second calibration domain."""
    from .model import DATA_PREFIX
    path = CACHE / f"{DATA_PREFIX}gsm8k-{split}.pt"
    if path.exists():
        return torch.load(path)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from .model import snapshot_dir

    rows = load_dataset("openai/gsm8k", "main", split=split)
    text = "\n\n".join(f"Question: {r['question']}\nAnswer: {r['answer']}" for r in rows)
    ids = AutoTokenizer.from_pretrained(snapshot_dir())(text, return_tensors="pt").input_ids[0]
    CACHE.mkdir(exist_ok=True)
    torch.save(ids, path)
    return ids


def gsm8k_calibration(n: int = 64, seqlen: int = 512, seed: int = 0) -> torch.Tensor:
    ids = gsm8k_ids("train")
    starts = torch.randint(0, len(ids) - seqlen - 1, (n,), generator=torch.Generator().manual_seed(seed))
    return torch.stack([ids[s : s + seqlen] for s in starts])


def gsm8k_evaluation(n: int = 64, seqlen: int = 512) -> torch.Tensor:
    return gsm8k_ids("test")[: n * seqlen].view(n, seqlen)
