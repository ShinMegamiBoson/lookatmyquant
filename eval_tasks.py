"""Downstream benchmarks for a transformers causal LM, from the datasets cached on this machine, in the conventions of
EleutherAI's lm-evaluation-harness:

    arc_c, arc_e, obqa, hellaswag   multiple choice, zero-shot, each choice scored by its log-likelihood; accuracy
                                    normalized by the choice's length in characters (acc_norm)
    csqa                            the same, unnormalized accuracy
    mmlu                            zero-shot, the question with lettered choices, scored by the logits of " A".." D"
    gsm8k                           5-shot chain of thought (the first 5 training problems), greedy, the final number

Every task returns its accuracy and each question's predicted choice, so two models can be compared question by
question (how often a quantized model changes the original's answer is far more sensitive than accuracy).

    res = run_tasks(model, tok, ["mmlu", "arc_c"], device="mps")
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F


BENCH = Path(__file__).parent / "data" / "bench"  # the same splits as JSON, for offline machines (Modal containers)


def _load(*args, split):
    try:
        from datasets import load_dataset
        return load_dataset(*args, split=split)
    except Exception:  # offline without the dataset cached: the exported copy (same rows, same order)
        return json.loads((BENCH / ("__".join([a.replace("/", "_") for a in args] + [split]) + ".json")).read_text())


def _hs_clean(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def mc_items(task, limit=None):
    """[(context, [continuations], gold)] for a multiple-choice task."""
    items = []
    if task in ("arc_c", "arc_e"):
        for d in _load("allenai/ai2_arc", "ARC-Challenge" if task == "arc_c" else "ARC-Easy", split="test"):
            labels = d["choices"]["label"]
            if d["answerKey"] not in labels:
                continue
            items.append((f"Question: {d['question']}\nAnswer:", [" " + t for t in d["choices"]["text"]], labels.index(d["answerKey"])))
    elif task == "obqa":
        for d in _load("allenai/openbookqa", "main", split="test"):
            labels = d["choices"]["label"]
            items.append((d["question_stem"], [" " + t for t in d["choices"]["text"]], labels.index(d["answerKey"].strip())))
    elif task == "csqa":
        for d in _load("tau/commonsense_qa", split="validation"):
            labels = d["choices"]["label"]
            items.append((f"Question: {d['question']}\nAnswer:", [" " + t for t in d["choices"]["text"]], labels.index(d["answerKey"])))
    elif task == "hellaswag":
        for d in _load("Rowan/hellaswag", split="validation"):
            ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
            items.append((_hs_clean(d["activity_label"] + ": " + ctx), [" " + _hs_clean(e) for e in d["endings"]], int(d["label"])))
    else:
        raise ValueError(task)
    return _sample(items, limit)


def _sample(rows, limit):
    """A fixed random subset in the original order (MMLU's test set is sorted by subject, so not the first n)."""
    if not limit or limit >= len(rows):
        return rows
    keep = sorted(random.Random(0).sample(range(len(rows)), limit))
    return [rows[i] for i in keep]


@torch.no_grad()
def score_mc(model, tok, items, device, norm, bs=32):
    """Each choice's summed log-probability given the context; the prediction is the best (per character with norm)."""
    seqs, owner = [], []
    for q, (ctx, conts, _) in enumerate(items):
        c_ids = BOS + tok(ctx, add_special_tokens=False).input_ids
        for j, cont in enumerate(conts):
            k_ids = tok(cont, add_special_tokens=False).input_ids
            seqs.append((c_ids + k_ids, len(c_ids), len(cont)))
            owner.append((q, j))
    scores = [[None] * len(it[1]) for it in items]
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i][0]))  # similar lengths together: less padding
    for b in range(0, len(order), bs):
        grp = [seqs[i] for i in order[b : b + bs]]
        T = max(len(s[0]) for s in grp)
        x = torch.full((len(grp), T), tok.pad_token_id or 0, dtype=torch.long)
        for i, (ids, _, _) in enumerate(grp):
            x[i, : len(ids)] = torch.tensor(ids)
        lp = F.log_softmax(model(x.to(device), use_cache=False).logits.float(), -1).cpu()
        for i, (ids, n_ctx, n_chr) in enumerate(grp):
            tgt = torch.tensor(ids[n_ctx:])
            ll = lp[i, n_ctx - 1 : len(ids) - 1].gather(-1, tgt[:, None]).sum().item()
            q, j = owner[order[b + i]]
            scores[q][j] = ll / n_chr if norm else ll
        del lp
    preds = [max(range(len(s)), key=lambda j: s[j]) for s in scores]
    return {"acc": sum(p == it[2] for p, it in zip(preds, items)) / len(items), "n": len(items), "preds": preds}


@torch.no_grad()
def score_mmlu(model, tok, device, limit=None, bs=16):
    rows = _load("cais/mmlu", "all", split="test")
    rows = _sample(list(rows), limit)
    letters = [tok(" " + L, add_special_tokens=False).input_ids[-1] for L in "ABCD"]
    prompts = []
    for d in rows:
        subj = d["subject"].replace("_", " ")
        body = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", d["choices"]))
        prompts.append(f"The following are multiple choice questions (with answers) about {subj}.\n\n{d['question']}\n{body}\nAnswer:")
    ids = [BOS + tok(p, add_special_tokens=False).input_ids for p in prompts]
    preds = [None] * len(rows)
    order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
    for b in range(0, len(order), bs):
        grp = order[b : b + bs]
        T = max(len(ids[i]) for i in grp)
        x = torch.full((len(grp), T), tok.pad_token_id or 0, dtype=torch.long)
        for r, i in enumerate(grp):
            x[r, : len(ids[i])] = torch.tensor(ids[i])
        lg = model(x.to(device), use_cache=False).logits
        for r, i in enumerate(grp):
            preds[i] = int(lg[r, len(ids[i]) - 1, letters].float().argmax())
    return {"acc": sum(p == d["answer"] for p, d in zip(preds, rows)) / len(rows), "n": len(rows), "preds": preds}


def _gsm_number(text):
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    nums = re.findall(r"-?[\d,]*\.?\d+", text) if not m else [m.group(1)]
    return nums[-1].replace(",", "").rstrip(".") if nums else None


@torch.no_grad()
def score_gsm8k(model, tok, device, limit=250, shots=5, bs=16, max_new=256):
    train, test = _load("openai/gsm8k", "main", split="train"), _load("openai/gsm8k", "main", split="test")
    head = "".join(f"Question: {d['question']}\nAnswer: {d['answer']}\n\n" for d in list(train)[:shots])
    rows = list(test)[:limit]
    side, tok.padding_side = tok.padding_side, "left"
    preds, outs = [], []
    for b in range(0, len(rows), bs):
        grp = rows[b : b + bs]
        bos = tok.bos_token if BOS else ""
        enc = tok([bos + head + f"Question: {d['question']}\nAnswer:" for d in grp], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for d, seq in zip(grp, g[:, enc.input_ids.shape[1] :]):
            text = tok.decode(seq, skip_special_tokens=True).split("\n\nQuestion:")[0].split("\n\n\n")[0]
            outs.append(text)
            preds.append(_gsm_number(text))
    tok.padding_side = side
    gold = [_gsm_number(d["answer"]) for d in rows]
    return {"acc": sum(p is not None and p == g for p, g in zip(preds, gold)) / len(rows), "n": len(rows), "preds": preds,
            "samples": outs[:3]}


NORM = {"arc_c": True, "arc_e": True, "obqa": True, "hellaswag": True, "csqa": False}
BOS = []  # [bos id] while run_tasks scores a model that needs a beginning-of-sequence token (Gemma: near chance without it)


def run_tasks(model, tok, tasks, device="mps", limits=None, log=print, bs=None, bos=False):
    """bs: sequences per forward pass (default 32 for multiple choice, 16 for MMLU and GSM8K); smaller saves memory.
    bos: start every prompt with the beginning-of-sequence token (models whose tokenizer does not add it but that need it)."""
    limits = limits or {}
    BOS[:] = [tok.bos_token_id] if bos and tok.bos_token_id is not None else []
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    out = {}
    for t in tasks:
        t0 = time.time()
        if t == "mmlu":
            r = score_mmlu(model, tok, device, limits.get(t), bs=bs or 16)
        elif t == "gsm8k":
            r = score_gsm8k(model, tok, device, limits.get(t, 250), bs=bs or 16)
        else:
            r = score_mc(model, tok, mc_items(t, limits.get(t)), device, NORM[t], bs=bs or 32)
        r["seconds"] = round(time.time() - t0)
        out[t] = r
        log(f"    {t}: {r['acc']:.4f} (n {r['n']}, {r['seconds']}s)")
    return out
