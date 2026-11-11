"""The exact-answer math benchmark (math_bench.py) on the Qwen3.5 family, greedy, and each model's KL from the original on
16 windows of WikiText-2. --models takes a comma-separated list of:

    orig              the original at --model-path
    dense:<file>      a build's weights, as quantize.py --save and finetune.py --save write them
    bonsai:<repo>     an MLX pack of Prism ML's: 2-bit codes with a scale and a bias per group, stored in a Hadamard-rotated
                      basis. The weights a runtime applies are W R, with R = H·diag(signs) on blocks of the input
    hf:<model>        another model with the original's vocabulary, as it is
    affineN:<model>   that model with every linear layer and embedding rounded as MLX's N-bit files hold it

    python math_eval.py --models orig,dense:build/dense.pt --no-thinking --out results/math.json     (run_batch.py: lm2Math)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import compiler_arch as A
import math_bench as B
from quantize import token_ids

DEV = "cuda" if torch.cuda.is_available() else "mps"


def log(msg):
    print(msg, flush=True)


def sylvester(n):
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def tail(name):
    if "layers." in name:
        return name[name.index("layers.") :]
    return ".".join(name.split(".")[-2:])


@torch.no_grad()
def load_bonsai(model, repo):
    """Writes the pack's weights into the model; returns what happened, tensor family by tensor family."""
    from safetensors import safe_open
    root = next((Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots").iterdir())  # only its weights and configs were fetched
    config = json.loads((root / "config.json").read_text())
    rot = json.loads((root / "hadamard.json").read_text())
    q, block = config["quantization"], rot["prism.hadamard.block_size"]
    bits, g = q["bits"], q["group_size"]
    H = sylvester(block).to(DEV)
    params = {tail(n): p for n, p in model.named_parameters()}
    plain_norms = {tail(n + ".weight") for n, m in model.named_modules() if type(m).__name__.endswith("RMSNorm")}  # Hugging Face's multiplies by 1 + weight, MLX's by weight
    report = {"coded": {}, "copied": {}, "unmapped pack tensors": [], "reshaped": [], "code histogram": [0] * (1 << bits), "bytes of text tensors": 0}
    shifts = torch.arange(0, 32, bits, device=DEV, dtype=torch.int64)
    used = set()
    with safe_open(str(root / "model.safetensors"), "pt") as f:
        keys = [k for k in f.keys() if not any(t in k for t in ("vision", "visual", "mtp"))]
        have = set(keys)
        for k in keys:
            report["bytes of text tensors"] += int(torch.tensor(f.get_slice(k).get_shape()).prod()) * {"U32": 4, "F16": 2, "F32": 4, "BF16": 2}.get(f.get_slice(k).get_dtype(), 4) if not k.endswith(".signs") else 0
            if not k.endswith(".weight") and k.rsplit(".", 1)[1] in ("scales", "biases", "signs"):
                continue
            base = k[: -len(".weight")] if k.endswith(".weight") else k
            name = tail(k)
            if name not in params:
                report["unmapped pack tensors"].append(k)
                continue
            P = params[name]
            used.add(name)
            family = name.split(".")[-2] if name.endswith(".weight") else name.split(".")[-1]
            if base + ".scales" in have:
                words = f.get_tensor(k).to(DEV).long() & 0xFFFFFFFF
                m, n = P.shape
                W = torch.empty(m, n, dtype=torch.float32, device=DEV)
                sc, bi, s = f.get_tensor(base + ".scales").float().to(DEV), f.get_tensor(base + ".biases").float().to(DEV), f.get_tensor(base + ".signs").float().to(DEV)
                for a in range(0, m, 32768):  # one chunk of rows at a time: the table has 248,320
                    c = ((words[a : a + 32768, :, None] >> shifts) & ((1 << bits) - 1)).flatten(1)[:, :n]
                    h = torch.bincount(c.reshape(-1), minlength=1 << bits).tolist()
                    report["code histogram"] = [u + v for u, v in zip(report["code histogram"], h)]
                    x = c.float() * sc[a : a + 32768].repeat_interleave(g, 1)[:, :n] + bi[a : a + 32768].repeat_interleave(g, 1)[:, :n]
                    x = torch.cat([x[:, b : b + block] @ H for b in range(0, n, block)], 1) * s[None, :]
                    W[a : a + 32768] = x
                cos = F.cosine_similarity(W.reshape(1, -1), P.data.float().reshape(1, -1).to(DEV)).item()
                report["coded"].setdefault(family, []).append(round(cos, 4))
                P.data.copy_(W.to(P.dtype))
                del words, W
            else:
                x = f.get_tensor(k).float().to(DEV)
                if name in plain_norms:
                    x = x - 1.0
                if x.shape != P.shape:
                    if x.numel() != P.numel():
                        report["unmapped pack tensors"].append(k)
                        continue
                    x = x.permute(0, 2, 1) if x.dim() == 3 else x.reshape(P.shape)
                    report["reshaped"].append(k)
                cos = F.cosine_similarity(x.reshape(1, -1), P.data.float().reshape(1, -1).to(DEV)).item()
                report["copied"].setdefault(family, []).append(round(cos, 4))
                P.data.copy_(x.to(P.dtype))
    report["model parameters the pack did not set"] = sorted(n for n in params if n not in used)[:20]
    report["coded"] = {k: {"tensors": len(v), "lowest": min(v), "median": sorted(v)[len(v) // 2]} for k, v in report["coded"].items()}
    report["copied"] = {k: {"tensors": len(v), "lowest": min(v), "median": sorted(v)[len(v) // 2]} for k, v in report["copied"].items()}
    report["reshaped"] = len(report["reshaped"])
    return report


@torch.no_grad()
def load_dense(model, path, keep_rows=None):
    d = torch.load(path, mmap=True)
    params = dict(model.named_parameters())
    emb = model.get_input_embeddings().weight
    kept = None if keep_rows is None else emb.data[keep_rows].clone()
    for n, W in d["params"].items():
        if n not in params and n.endswith(".bias"):  # a bias a correction added to a bias-free linear (tradeoff_modal.py biascorr)
            mod = model.get_submodule(n[: -len(".bias")])
            mod.bias = torch.nn.Parameter(W.to(mod.weight.dtype).to(DEV), requires_grad=False)
            continue
        params[n].data.copy_(W.to(params[n].dtype).to(DEV))
    if keep_rows is not None:
        emb.data[keep_rows] = kept
    return {"tensors": len(d["params"]), "GB": d["text weights"]["bits"] / 8e9}


@torch.no_grad()
def load_other(what, tok, bits=0, group=64):
    """Another model with the original's tokenizer, in its place: a Hub name in the cache, or a path. With `bits`, every
    linear layer's and embedding's weight is rounded as MLX's N-bit files hold it (mlx.core.quantize): in groups of 64
    along a row, each value to one of 2^bits levels between its group's lowest and highest, with a scale and a bias for
    each group (counted at 16 bits each). Returns the model and what it holds; the bytes count every parameter not
    rounded at 16 bits, as the original's file stores it."""
    import glob
    import os
    from huggingface_hub.constants import HF_HUB_CACHE
    # (not snapshot_download: the volume holds a model's weights and tokenizer only, which it calls an incomplete snapshot)
    path = what if os.path.isdir(what) else sorted(glob.glob(os.path.join(HF_HUB_CACHE, "models--" + what.replace("/", "--"), "snapshots", "*")))[-1]
    # the vocabulary must be the original's. The chat template is the original's too: Qwen3.5-2B's and -0.8B's differ from
    # the 4B's only in whether thinking is on when nobody says, and solve() always says
    assert AutoTokenizer.from_pretrained(path).get_vocab() == tok.get_vocab(), f"{what} does not share the original's vocabulary"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(DEV).eval()
    seen, coded, size = set(), 0, 0
    for m in model.modules() if bits else ():
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)) and id(m.weight) not in seen and m.weight.shape[1] % group == 0:
            seen.add(id(m.weight))
            W = m.weight.data
            for a in range(0, len(W), 32768):  # a chunk of rows at a time: the table has 248,320
                g = W[a : a + 32768].reshape(-1, W.shape[1] // group, group)
                lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
                scale = ((hi - lo) / ((1 << bits) - 1)).clamp_min(1e-12)
                W[a : a + 32768] = (((g - lo) / scale).round().clamp(0, (1 << bits) - 1) * scale + lo).reshape(-1, W.shape[1])
            coded, size = coded + W.numel(), size + W.numel() * bits // 8 + W.numel() // group * 4
    rest = sum(p.numel() for p in model.parameters() if id(p) not in seen)
    return model, {"model": what, "parameters": coded + rest, "rounded parameters": coded, "bits": bits, "bytes of text tensors": size + rest * 2}


@torch.no_grad()
def restore_layers(model, ref, layers):
    """Every parameter of these decoder layers set to the original's (a diagnostic: the build grows); returns what they replaced, to put back."""
    import compiler_arch as A
    path, _ = A.decoder_layers(model)
    rp, kept = dict(ref.named_parameters()), {}
    for n, p in model.named_parameters():
        if any(n.startswith(f"{path}.{l}.") for l in layers):
            kept[n] = p.data.clone()
            p.data.copy_(rp[n].data)
    return kept


@torch.no_grad()
def wikitext_kl(model, ref_lp, windows):
    lp = torch.cat([F.log_softmax(model(windows[i : i + 4].to(DEV), use_cache=False).logits.float(), -1).cpu() for i in range(0, len(windows), 4)])
    return float((ref_lp.exp() * (ref_lp - lp)).sum(-1).mean())


@torch.no_grad()
def solve(model, tok, problems, batch, max_new, thinking=True, keep_text=False, sampling=None):
    """Greedy answers, longest prompts together so that padding is small, and the same batches for every model."""
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{p['question']}\n\n{B.INSTRUCTION}"}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking) for p in problems]
    order = sorted(range(len(problems)), key=lambda i: (len(prompts[i]), problems[i]["id"]))
    rows, t0 = {}, time.time()
    for b in range(0, len(order), batch):
        idx = order[b : b + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(DEV)
        how = dict(do_sample=True, **sampling) if sampling else dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        ids = model.generate(**enc, max_new_tokens=max_new, **how, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, use_cache=True)
        new = ids[:, enc.input_ids.shape[1] :]
        for j, (row, i) in enumerate(zip(new, idx)):
            eos = [t for t in (tok.eos_token_id, tok.pad_token_id) if t is not None]
            n = next((j for j, t in enumerate(row.tolist()) if t in eos), len(row))
            text = tok.decode(row[:n], skip_special_tokens=False)
            ok, got = B.score(problems[i], text) if n < max_new else (False, None)  # a run cut off by the limit has not answered, whatever it wrote while thinking
            rows[problems[i]["id"]] = {"correct": ok, "answer": got, "tokens": n, "finished": n < max_new, "end": text[-200:]} | ({"text": text} if keep_text else {})
        log(f"  {min(b + batch, len(order))}/{len(order)} ({time.time() - t0:.0f}s): {sum(r['correct'] for r in rows.values())} right so far, {sum(not r['finished'] for r in rows.values())} hit the token limit")
    return rows


def summary(problems, rows):
    by = lambda key: {k: round(sum(rows[p["id"]]["correct"] for p in problems if key(p) == k) / sum(1 for p in problems if key(p) == k), 3) for k in sorted({key(p) for p in problems})}
    return {"accuracy": round(sum(r["correct"] for r in rows.values()) / len(rows), 4), "right": sum(r["correct"] for r in rows.values()), "of": len(rows),
            "by difficulty": by(lambda p: p["difficulty"]), "by family": by(lambda p: p["family"]), "no answer": sum(r["answer"] is None for r in rows.values()),
            "hit the token limit": sum(not r["finished"] for r in rows.values()), "mean tokens": round(sum(r["tokens"] for r in rows.values()) / len(rows))}


def paired(a, b):
    """Problems only A got, only B got, and McNemar's exact two-sided p."""
    x = sum(a[k]["correct"] and not b[k]["correct"] for k in a)
    y = sum(b[k]["correct"] and not a[k]["correct"] for k in a)
    n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(x, y) + 1)) / 2**n)
    return {"only first": x, "only second": y, "p": round(p, 5), "same answer": round(sum(a[k]["answer"] == b[k]["answer"] for k in a) / len(a), 3)}


def apply_spec(model, spec, tok):
    """What a --models entry names, given the original loaded: orig | bonsai:<repo> | dense:<file> | dense+special:<file> |
    hf:<model> (another model of the family as it is) | affineN:<model> (at N bits, load_other). Returns the model and its
    load report (None for orig)."""
    kind, _, what = spec.partition(":")
    load = None
    if kind == "bonsai":
        load = load_bonsai(model, what)
        log(json.dumps(load))
    elif kind == "dense":
        load = load_dense(model, what)
    elif kind == "dense+special":  # with the tokenizer's added tokens' table rows as they were
        load = load_dense(model, what, torch.tensor(sorted(tok.added_tokens_decoder), device=DEV))
    elif kind == "hf" or kind.startswith("affine"):  # hf:Qwen/Qwen3.5-2B, affine8:Qwen/Qwen3.5-0.8B
        model, load = load_other(what, tok, int(kind[6:] or 0))
        for p in model.parameters():
            p.requires_grad_(False)
        log(json.dumps(load))
    return model, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/.cache/huggingface/models/Qwen3.5-4B")
    ap.add_argument("--models", default="orig,bonsai:prism-ml/Ternary-Bonsai-2-4B-mlx-2bit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-cell", type=int, default=15)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-thinking", action="store_true", help="the chat template's empty thinking block: the model reasons in its answer")
    ap.add_argument("--ids-file", default="", help="a JSON list of problem ids: only these")
    ap.add_argument("--problems-file", default="", help="a JSON file whose \"problems\" are math_bench problems to use instead of generating them (e.g. a fresh frozen set)")
    ap.add_argument("--keep-text", action="store_true", help="each answer's whole text, not only its end")
    ap.add_argument("--temperature", type=float, default=0.0, help="0: greedy; else sampled with --top-p and --top-k (Qwen's non-thinking advice: 0.7, 0.8, 20)")
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--restore-layers", default="", help="first-last: these decoder layers given the original's weights after loading (a diagnostic: the build grows)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    problems = json.loads(Path(args.problems_file).read_text())["problems"] if args.problems_file else B.make_problems(args.seed, args.per_cell)
    assert all(B.check(p) for p in problems)
    if args.ids_file:
        keep = set(json.loads(Path(args.ids_file).read_text()))
        problems = [p for p in problems if p["id"] in keep]
    ids = token_ids(tok, "validation")
    wins = ids[: len(ids) // 512 * 512].reshape(-1, 512)
    wins = wins[torch.randperm(len(wins), generator=torch.Generator().manual_seed(0))[:16]]
    res = {"device": DEV, "problems": len(problems), "max new tokens": args.max_new, "greedy": args.temperature == 0, "thinking": not args.no_thinking, "models": {}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ref_lp = None
    all_rows = {}
    for spec in args.models.split(","):
        kind, _, what = spec.partition(":")
        model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float32).to(DEV).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if ref_lp is None:
            with torch.no_grad():
                ref_lp = torch.cat([F.log_softmax(model(wins[i : i + 4].to(DEV), use_cache=False).logits.float(), -1).cpu() for i in range(0, len(wins), 4)])
        rec = {}
        model, load = apply_spec(model, spec, tok)
        if load is not None:
            rec["load"] = load
        if args.restore_layers:
            lo, hi = (int(x) for x in args.restore_layers.split("-"))
            orig_ = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float32).eval()
            rec["restored parameters"] = sum(v.numel() for v in restore_layers(model, orig_, range(lo, hi + 1)).values())
            log(f"layers {lo}-{hi} restored to the original: {rec['restored parameters'] / 1e6:.0f}M parameters")
            del orig_
        rec["wikitext KL"] = wikitext_kl(model, ref_lp, wins)
        log(f"{spec}: WikiText KL {rec['wikitext KL']:.4f} ({time.time() - t0:.0f}s)")
        torch.manual_seed(args.sample_seed)
        sampling = dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k) if args.temperature > 0 else None
        rows = solve(model, tok, problems, args.batch, args.max_new, not args.no_thinking, args.keep_text, sampling)
        rec["summary"] = summary(problems, rows)
        rec["rows"] = rows
        all_rows[spec] = rows
        log(f"{spec}: {json.dumps(rec['summary'])}")
        names = list(all_rows)
        if len(names) > 1:
            rec["against the first model"] = paired(all_rows[names[0]], rows)
            log(f"  against {names[0]}: {json.dumps(rec['against the first model'])}")
        res["models"][spec] = rec
        res["seconds"] = round(time.time() - t0)
        out.write_text(json.dumps(res, indent=1))
        del model
        torch.cuda.empty_cache() if DEV == "cuda" else None
    log(f"=> mathBench done ({res['seconds']}s)")


if __name__ == "__main__":
    main()
