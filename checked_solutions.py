"""The math text the LookAtMyQuant 2 fine-tuning distils on: the original model's worked solutions, checked.

The original answers MATH training problems and math_bench problems from a seed the benchmark does not use (every question
reserved for evaluation is excluded by its hash), thinking off, with the benchmark's instruction. Each stated calculation
the checker finds wrong (calc_check.py) is replaced by the exact result and the original continues from there. Only answers
that finish within --max-new and whose final answer is right are kept. Windows of --len tokens each start at a question and
hold whole question-and-answer pairs; a pair that does not fit fills the rest and starts the next window again.

    python checked_solutions.py --out cache/math_checked.pt     (run_batch.py: lm2Solutions)
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import hashlib

import math_bench as B
from calc_check import first_wrong
from math_eval import DEV


def question_hash(question):
    return hashlib.sha256(' '.join(question.split()).encode()).hexdigest()


def log(msg):
    print(msg, flush=True)


def boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, d = i + 7, 1
    while j < len(s) and d:
        d += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i + 7 : j - 1]


def norm(x):
    x = (x or "").replace("dfrac", "frac").replace("tfrac", "frac")
    return re.sub(r"\\!|\\left|\\right|\s|\$|\\text\{([^}]*)\}|\\,|\.$|^\{|\}$", lambda m: m.group(1) or "", x).replace(",", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/.cache/huggingface/models/Qwen3.5-4B")
    ap.add_argument("--math-sources", default="/root/.cache/huggingface/retrain/calib_sources")
    ap.add_argument("--math-problems", type=int, default=384)
    ap.add_argument("--skip", type=int, default=4096, help="as math_calibration.py: MATH problems at the front of the shuffled order are left for other uses")
    ap.add_argument("--bench-seed", type=int, default=424242)
    ap.add_argument("--bench-per-cell", type=int, default=11)
    ap.add_argument("--reserved", default="results/protocols/reserved_questions.json")
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # problems: (kind, question, how to check a final answer)
    reserved = set()
    for f in args.reserved.split(","):
        d = json.loads(Path(f).read_text())
        reserved |= set(d.get("question_hashes", [])) | {question_hash(x["question"]) for x in d.get("problems", [])}
    files = sorted(glob.glob(f"{args.math_sources}/datasets--EleutherAI--hendrycks_math/snapshots/*/*/train-*.parquet"))
    rows = [r for f in files for r in pd.read_parquet(f)[["problem", "solution"]].itertuples(index=False)]
    random.Random(0).shuffle(rows)  # the order math_calibration.py uses
    math = [("math", r.problem, norm(boxed(r.solution))) for r in rows[args.skip : args.skip + args.math_problems]]
    bench = [("bench", p["question"], p) for p in B.make_problems(args.bench_seed, args.bench_per_cell) if question_hash(p["question"]) not in reserved]
    log(f"{len(math)} MATH problems, {len(bench)} math_bench problems (seed {args.bench_seed}; {len(reserved)} reserved questions excluded)")
    items = [x for pair in zip(math, bench) for x in pair] + math[len(bench):] + bench[len(math):]  # alternating
    ask = lambda q: tok.apply_chat_template([{"role": "user", "content": f"{q}\n\n{B.INSTRUCTION}"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    n_tok = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    eos = {t for t in (tok.eos_token_id, tok.pad_token_id) if t is not None}

    def write(jobs):
        """{i: (text, finished)} continuing each (i, prefix, budget) greedily."""
        out = {}
        jobs = sorted(jobs, key=lambda j: n_tok(ask(items[j[0]][1]) + j[1]))
        for b in range(0, len(jobs), args.batch):
            part = jobs[b : b + args.batch]
            enc = tok([ask(items[i][1]) + pre for i, pre, _ in part], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            with torch.no_grad():
                ids = model.generate(**enc, max_new_tokens=max(n for _, _, n in part), do_sample=False, temperature=None, top_p=None, top_k=None,
                                     pad_token_id=tok.pad_token_id, use_cache=True)
            for row, (i, pre, n) in zip(ids[:, enc.input_ids.shape[1] :].tolist(), part):
                row = row[:n]
                k = next((j for j, t in enumerate(row) if t in eos), None)
                out[i] = (pre + tok.decode(row[: len(row) if k is None else k], skip_special_tokens=False), k is not None)
            log(f"  {min(b + len(part), len(jobs))}/{len(jobs)} written ({time.time() - t0:.0f}s)")
        return out

    state = {i: {"text": t, "finished": f, "corrections": 0} for i, (t, f) in write([(i, "", args.max_new) for i in range(len(items))]).items()}
    first_right = {i: s["finished"] and first_wrong(s["text"]) is None for i, s in state.items()}
    for rnd in range(1, args.rounds + 1):
        jobs = []
        for i, s in state.items():
            w = first_wrong(s["text"])
            if w is None:
                continue
            pre = s["text"][: w[0]] + w[1]
            left = args.max_new - n_tok(pre)
            if left > 0:
                s["corrections"] += 1
                jobs.append((i, pre, left))
        if not jobs:
            break
        for i, (t, f) in write(jobs).items():
            state[i] |= {"text": t, "finished": f}
        log(f"round {rnd}: {len(jobs)} answers corrected and continued ({time.time() - t0:.0f}s)")

    def right(i):
        kind, q, gold = items[i]
        s = state[i]
        if not s["finished"] or first_wrong(s["text"]) is not None:
            return False
        return B.score(gold, s["text"])[0] if kind == "bench" else (norm(boxed(s["text"])) == gold and gold != "")

    keep = [i for i in range(len(items)) if right(i)]
    seqs = [tok(ask(items[i][1]) + state[i]["text"] + "<|im_end|>\n", add_special_tokens=False).input_ids for i in keep]
    windows, cur, j = [], [], 0
    while j < len(seqs):
        s = seqs[j]
        if len(cur) + len(s) <= args.len:
            cur += s
            j += 1
            continue
        if cur:  # the next pair fills the rest, cut at the window's end, and starts the next window whole
            windows.append((cur + s)[: args.len])
            cur = []
        else:  # a pair longer than a window: its first window only
            windows.append(s[: args.len])
            j += 1
    W = torch.tensor(windows[: len(windows)], dtype=torch.long)
    W = W[torch.randperm(len(W), generator=torch.Generator().manual_seed(0))]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(W, args.out)
    kinds = lambda ids: {k: sum(items[i][0] == k for i in ids) for k in ("math", "bench")}
    res = {"problems": kinds(range(len(items))), "kept": kinds(keep), "right without corrections": kinds([i for i in keep if first_right[i]]),
           "corrections": sum(s["corrections"] for s in state.values()), "ran out": sum(not s["finished"] for s in state.values()),
           "windows": len(W), "tokens": W.numel(), "mean pair tokens": sum(map(len, seqs)) / max(1, len(seqs)), "seconds": round(time.time() - t0),
           "examples": {items[i][0]: state[i]["text"][:1500] for i in keep[:4]}}
    log("summary: " + json.dumps({k: v for k, v in res.items() if k != "examples"}))
    Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
