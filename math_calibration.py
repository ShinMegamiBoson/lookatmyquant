"""The calibration set of the LookAtMyQuant 2 recipe: the original model's own greedy solutions to MATH training
problems, thinking off, up to --max-new tokens each, cut into windows of --len tokens for quantize.py --math-cache.

    python math_calibration.py --out cache/math_long.pt     (run_batch.py: lm2Calibration)

--math-sources is a Hugging Face cache that holds EleutherAI/hendrycks_math.
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_eval import DEV


def log(msg):
    print(msg, flush=True)


def ask(tok, q):
    """A question as the original was asked it for its worked solutions: step by step, thinking off."""
    return tok.apply_chat_template([{"role": "user", "content": f"{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def math_questions(root):
    """MATH training questions in a fixed shuffled order, before rendering."""
    import random
    import pandas as pd
    files = sorted(glob.glob(f"{root}/datasets--EleutherAI--hendrycks_math/snapshots/*/*/train-*.parquet"))
    assert files, f"no MATH training files under {root}"
    qs = [q for f in files for q in pd.read_parquet(f)["problem"]]
    random.Random(0).shuffle(qs)
    return qs


def math_prompts(tok, root):
    return [ask(tok, q) for q in math_questions(root)]


def solved(model, tok, prompts, T, batch=64, max_new=384, note="math text"):
    """Windows of T tokens from a model's own greedy solutions to the prompts, each with its prompt (tok pads on the left)."""
    t0 = time.time()
    prompts = sorted(prompts, key=len)
    stream = []
    with torch.no_grad():
        for b in range(0, len(prompts), batch):
            enc = tok(prompts[b : b + batch], return_tensors="pt", padding=True).to(DEV)
            ids = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, use_cache=True).cpu()
            for row, mask in zip(ids, enc.attention_mask.cpu()):
                keep = row[int((mask == 0).sum()) :]
                ends = (keep[mask.sum() :] == tok.eos_token_id).nonzero()
                stream.append(keep[: int(mask.sum()) + int(ends[0]) + 1] if len(ends) else keep)
            log(f"  {note}: {min(b + batch, len(prompts))}/{len(prompts)} solved ({time.time() - t0:.0f}s)")
    ids = torch.cat(stream)
    n = len(ids) // T * T
    return ids[:n].view(-1, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/.cache/huggingface/models/Qwen3.5-4B")
    ap.add_argument("--math-sources", default="/root/.cache/huggingface/retrain/calib_sources")
    ap.add_argument("--problems", type=int, default=384)
    ap.add_argument("--skip", type=int, default=4096, help="problems at the front of the shuffled order, left for other uses")
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float32).to(DEV).eval()
    prompts = math_prompts(tok, args.math_sources)[args.skip : args.skip + args.problems]
    w = solved(model, tok, prompts, args.len, batch=args.batch, max_new=args.max_new, note="long solutions")
    w = w[torch.randperm(len(w), generator=torch.Generator().manual_seed(0))]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(w, args.out)
    print(f"=> calibration set done: {len(w)} windows of {args.len} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
