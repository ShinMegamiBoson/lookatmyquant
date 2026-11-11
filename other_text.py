"""The fine-tuning's text that is not math, written by the original model itself: prompts from training splits, each
answered once with thinking on or off at random, cut into windows of --len tokens per kind.

    science    ARC-Challenge train and SciQ train, as reasoned multiple choice
    code       MBPP train, validation and prompt splits
    instruct   Tulu-3's instruction-following prompts with verifiable constraints
    tools      Hermes function-calling conversations, tools given in the system prompt
    math       MATH train, and gsm8k: GSM8K train. More worked solutions for the pool, beside checked_solutions.py's
               (our run's GSM8K solutions were written earlier, with thinking always off)

    python other_text.py --out cache/other_text.pt --report results/other_text.json     (run_batch.py: lm2Text)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_eval import DEV

FILES = {
    "math": [("EleutherAI/hendrycks_math", f"{s}/train-00000-of-00001.parquet") for s in
             ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus")],
    "gsm8k": [("openai/gsm8k", "main/train-00000-of-00001.parquet")],
    "science": [("allenai/ai2_arc", "ARC-Challenge/train-00000-of-00001.parquet"), ("allenai/sciq", "data/train-00000-of-00001.parquet")],
    "code": [("google-research-datasets/mbpp", f"full/{s}-00000-of-00001.parquet") for s in ("train", "validation", "prompt")],
    "instruct": [("allenai/tulu-3-sft-personas-instruction-following", "data/train-00000-of-00001.parquet")],
    "tools": [("NousResearch/hermes-function-calling-v1", f) for f in ("func-calling-singleturn.json", "func-calling.json")],
}


def log(msg):
    print(msg, flush=True)


def fetch(repo, name, root):
    import huggingface_hub.constants as hc
    from huggingface_hub import hf_hub_download
    hc.HF_HUB_OFFLINE = False  # the image is offline for the cache; these few files come from the Hub once
    return hf_hub_download(repo, name, repo_type="dataset", cache_dir=str(root))


def chats(skill, paths, rng, n):
    """[(messages, note)] for one skill: the user's turn, and for tool use the system prompt that lists the tools."""
    import pandas as pd
    out = []
    if skill in ("math", "gsm8k"):
        for p in paths:
            out += [[{"role": "user", "content": f"{r.problem if skill == 'math' else r.question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."}] for r in pd.read_parquet(p).itertuples()]
    elif skill == "science":
        for p in paths:
            d = pd.read_parquet(p)
            for r in d.itertuples():
                if "choices" in d.columns:
                    opts = list(zip(r.choices["label"], r.choices["text"]))
                else:
                    texts = [r.correct_answer, r.distractor1, r.distractor2, r.distractor3]
                    rng.shuffle(texts)
                    opts = list(zip("ABCD", texts))
                q = r.question + "\n" + "\n".join(f"{a}. {t}" for a, t in opts)
                out.append([{"role": "user", "content": f"{q}\n\nThink it through, then give the letter of the answer."}])
    elif skill == "code":
        for p in paths:
            for r in pd.read_parquet(p).itertuples():
                out.append([{"role": "user", "content": f"{r.text}\nYour code should pass this test:\n{r.test_list[0]}"}])
    elif skill == "instruct":
        for r in pd.read_parquet(paths[0]).itertuples():
            out.append([{"role": "user", "content": r.prompt}])
    elif skill == "tools":
        for p in paths:
            for row in json.loads(Path(p).read_text()):
                conv = row.get("conversations") or []
                sys_ = next((c["value"] for c in conv if c["from"] == "system"), None)
                user = next((c["value"] for c in conv if c["from"] == "human"), None)
                if sys_ and user:
                    out.append([{"role": "system", "content": sys_}, {"role": "user", "content": user}])
    rng.shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/.cache/huggingface/models/Qwen3.5-4B")
    ap.add_argument("--per-skill", type=int, default=256)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--skills", default=",".join(FILES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = random.Random(0)
    root = Path(args.out).parent / "calib_sources"
    tok = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16).to(DEV).eval()
    windows, report = {}, {"skills": {}}
    for skill in args.skills.split(","):
        paths = [fetch(r, f, root) for r, f in FILES[skill]]
        convs = chats(skill, paths, rng, args.per_skill)
        prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=rng.random() < 0.5) for c in convs]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        stream, finished, lengths = [], 0, []
        for b in range(0, len(order), args.batch):
            idx = order[b : b + args.batch]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=2048).to(DEV)
            with torch.no_grad():
                ids = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id).cpu()
            for row, mask in zip(ids, enc.attention_mask.cpu()):
                keep = row[int((mask == 0).sum()) :]
                n = int(mask.sum())
                ends = (keep[n:] == tok.eos_token_id).nonzero()
                full = keep[: n + int(ends[0]) + 1] if len(ends) else keep
                finished += bool(len(ends))
                lengths.append(len(full) - n)
                stream.append(full)
            log(f"  {skill}: {min(b + args.batch, len(order))}/{len(order)} ({time.time() - t0:.0f}s)")
        ids = torch.cat(stream)
        windows[skill] = ids[: len(ids) // args.len * args.len].view(-1, args.len)
        report["skills"][skill] = {"prompts": len(convs), "finished": finished, "mean answer tokens": sum(lengths) / max(1, len(lengths)), "windows": len(windows[skill]),
                                   "sources": [f"{r}/{f}" for r, f in FILES[skill]], "example": tok.decode(stream[0][-200:])}
        log(f"{skill}: {json.dumps({k: v for k, v in report['skills'][skill].items() if k != 'example'})}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(windows, args.out)
    report["seconds"] = round(time.time() - t0)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    log(f"=> other text done ({report['seconds']}s): " + json.dumps({k: len(v) for k, v in windows.items()}))


if __name__ == "__main__":
    main()
