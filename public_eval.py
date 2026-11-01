"""Public quantizations of a model, scored exactly as the compiler's plans are: each checkpoint dequantized into the Hugging Face model, then perplexity and KL to the original on
the first 64 WikiText-2 test windows of 512 tokens and on the text mix's held-out windows (8 per source, seed 2), the
windows quantize.py evaluates. Size = the bytes of the checkpoint's text-model tensors (vision and audio towers
left out), as the plans count every text weight.

    gguf:<repo>/<pattern>   a GGUF file (Google's QAT q4_0, unsloth, bartowski); tensors mapped to the model by content,
                            against our own exact bf16 conversion of it (llama.cpp's convert_hf_to_gguf.py)
    gguf-local:<file>       a GGUF made by gguf_compiler.py prepare (llama.cpp's own recipes with our importance matrix)
    mlx:<repo>              an MLX checkpoint (affine, packed in 32-bit words, a scale and bias per group)
    ct:<repo>               compressed-tensors (Google's QAT w4a16), decompressed by transformers on load
Runs in modal_gguf.py's image.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

HF = Path("/root/.cache/huggingface")
TOWERS = ("vision", "audio", "visual", "mm_", "multi_modal", "embed_vision", "embed_audio", "mmproj")


def log(msg):
    print(msg, flush=True)


def snapshot(repo):
    return Path(glob.glob(str(HF / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / "*/"))[0])


def text_params(model):
    """{name: parameter} of the text model (and its output head), towers left out."""
    return {n: p for n, p in model.named_parameters() if not any(t in n for t in TOWERS)}


# ----------------------------------------------------------------------------- GGUF


def gguf_map(model, base_gguf, cache):
    """{gguf tensor name: (parameter name, transform, scale)} by content: each 2-D (or larger) tensor of our bf16 GGUF
    matched to the text parameter holding the same values, as stored, transposed, or (Llama) un-permuted, possibly times
    a constant (a converter that folds a scale in). Fingerprints (the first 256 values of each layout) find candidates; a
    full comparison confirms. Unmatched tensors on either side are listed in the cache file."""
    if cache.exists():
        return json.loads(cache.read_text())["map"]
    from gguf import GGUFReader, dequantize
    import gguf_compiler as G
    params = {n: p for n, p in text_params(model).items() if p.dim() >= 2}
    cfg = getattr(model.config, "text_config", None) or model.config
    heads = sorted({h for h in (getattr(cfg, "num_attention_heads", None), getattr(cfg, "num_key_value_heads", None)) if isinstance(h, int)})
    by_numel = {}
    for n, p in params.items():
        by_numel.setdefault(p.numel(), []).append(n)

    def layouts(A_, shape):
        yield "id", A_.reshape(shape)
        if A_.dim() == 2 and A_.T.shape == shape:
            yield "T", A_.T
        if A_.dim() == 2 and A_.shape == shape:
            for h in heads:
                if A_.shape[0] % (2 * h) == 0:
                    yield f"unpermute:{h}", G.unpermute(A_, h)
    out, used, unmatched = {}, set(), []
    r = GGUFReader(str(base_gguf))
    for t in r.tensors:
        if len(t.shape) < 2:
            continue
        a = torch.from_numpy(dequantize(t.data, t.tensor_type).astype(np.float32))
        A_ = a.reshape(int(t.shape[1]), int(t.shape[0]))
        found = None
        for n in by_numel.get(A_.numel(), []):
            if n in used:
                continue
            P = params[n].detach().float().cpu()
            for tr, X in layouts(A_, P.shape):
                x, y = X.reshape(-1)[:256], P.reshape(-1)[:256]
                c = float((x * y).sum() / (x * x).sum().clamp(min=1e-30))
                if not torch.allclose(c * x, y, rtol=1e-2, atol=1e-6):
                    continue
                c = float((X * P).sum() / (X * X).sum())
                if float((c * X - P).abs().max()) <= 1e-2 * float(P.abs().max()):
                    found = [n, tr, c]
                    break
            if found:
                break
        if found:
            out[t.name] = found
            used.add(found[0])
        else:
            unmatched.append(t.name)
    missing = sorted(n for n in params if n not in used)
    cache.write_text(json.dumps({"map": out, "unmatched gguf tensors": unmatched, "parameters without a gguf tensor": missing}, indent=1))
    log(f"GGUF map: {len(out)} matched, {len(unmatched)} GGUF tensors unmatched {unmatched[:5]}, {len(missing)} parameters unmapped {missing[:5]}")
    return out


def load_gguf(model, path, mapping):
    """Dequantize every mapped tensor of a GGUF into `model`; returns the bytes of the file's tensors (text only)."""
    from gguf import GGUFReader, dequantize
    import gguf_compiler as G
    params = dict(model.named_parameters())
    r = GGUFReader(str(path))
    nbytes, hit = 0, 0
    for t in r.tensors:
        nbytes += int(t.n_bytes)
        if t.name not in mapping:
            continue
        n, tr, c = mapping[t.name]
        a = torch.from_numpy(dequantize(t.data, t.tensor_type).astype(np.float32))
        P = params[n]
        X = a.reshape(int(t.shape[1]), int(t.shape[0])) if len(t.shape) >= 2 else a
        X = X.T if tr == "T" else G.unpermute(X, int(tr.split(":")[1])) if tr.startswith("unpermute") else X
        P.data.copy_((c * X).reshape(P.shape).to(P.dtype))
        hit += 1
    return nbytes, hit


# ----------------------------------------------------------------------------- MLX


def unpack(words, bits, n):
    """MLX's packing: each row a little-endian stream of `bits`-bit values in 32-bit words; the first n values."""
    w = words.astype(np.uint64)
    w = np.concatenate([w, np.zeros((w.shape[0], 1), dtype=np.uint64)], axis=1)
    j = np.arange(n, dtype=np.uint64) * np.uint64(bits)
    i0, off = (j // 32).astype(np.int64), (j % 32)
    v = (w[:, i0] >> off) | (w[:, i0 + 1] << (np.uint64(32) - off))
    return (v & np.uint64((1 << bits) - 1)).astype(np.float32)


def load_mlx(model, repo):
    from safetensors.torch import load_file
    root = snapshot(repo)
    cfg = json.loads((root / "config.json").read_text())
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    tensors, stored = {}, {}
    for f in sorted(root.glob("*.safetensors")):  # numpy has no bfloat16: through torch, then float32 or uint32 numpy
        for k, x in load_file(str(f)).items():
            stored[k] = x.numel() * x.element_size()  # the bytes as stored
            tensors[k] = x.float().numpy() if x.is_floating_point() else x.numpy()
    params = text_params(model)
    tail = lambda n: n.split("layers.", 1)[1] if "layers." in n else n.rsplit(".", 2)[-2] + "." + n.rsplit(".", 1)[-1]
    by_tail = {}
    for n in params:
        by_tail.setdefault(tail(n), []).append(n)
    nbytes, hit = 0, 0
    for k, v in tensors.items():
        if any(t in k for t in TOWERS):
            continue
        nbytes += stored[k]
        if not k.endswith(".weight"):
            continue
        base = k[: -len(".weight")]
        cands = [n for n in by_tail.get(tail(k), []) if params[n].dim() >= 2]
        if not cands:
            continue
        P = params[cands[0]]
        if base + ".scales" in tensors:
            spec = q.get(base) if isinstance(q.get(base), dict) else q
            bits, g = int(spec.get("bits", 4)), int(spec.get("group_size", 64))
            sc, bi = tensors[base + ".scales"].astype(np.float32), tensors[base + ".biases"].astype(np.float32)
            n_in = sc.shape[-1] * g
            Wq = unpack(v.reshape(v.shape[0], -1), bits, n_in) * np.repeat(sc, g, axis=-1) + np.repeat(bi, g, axis=-1)
        else:
            Wq = v.astype(np.float32)
        if Wq.size != P.numel():
            continue
        P.data.copy_(torch.from_numpy(np.ascontiguousarray(Wq)).reshape(P.shape).to(P.dtype))
        hit += 1
    return nbytes, hit


# ----------------------------------------------------------------------------- compressed-tensors


def load_ct(model, repo, dtype):
    from transformers import AutoModelForCausalLM
    from transformers.utils.quantization_config import CompressedTensorsConfig
    from safetensors import safe_open
    root = snapshot(repo)
    d = AutoModelForCausalLM.from_pretrained(str(root), dtype=dtype, quantization_config=CompressedTensorsConfig(run_compressed=False))
    src = dict(d.named_parameters())
    hit = 0
    for n, P in text_params(model).items():
        if n in src and src[n].shape == P.shape:
            P.data.copy_(src[n].data.to(P.dtype).to(P.device))
            hit += 1
    del d
    nbytes = 0
    for f in root.glob("*.safetensors"):
        with safe_open(str(f), "pt") as fh:
            for k in fh.keys():
                if not any(t in k for t in TOWERS):
                    sl = fh.get_slice(k)
                    nbytes += int(np.prod(sl.get_shape())) * {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I8": 1, "U8": 1, "I64": 8}.get(sl.get_dtype(), 2)
    return nbytes, hit


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--checkpoints", required=True, help="comma list: gguf:<repo>/<glob>, gguf-local:<file>, mlx:<repo>, ct:<repo>, dense:<file>")
    ap.add_argument("--tasks", default="", help="benchmarks for every checkpoint (eval_tasks.py), as the plans run them")
    ap.add_argument("--task-limits", default="mmlu=1000,hellaswag=1000,gsm8k=250")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import compiler_arch as A
    import gguf_compiler as G
    import quantize as qz
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev, dtype = "cuda", getattr(torch, args.dtype)
    src = G.model_dir(args.model)
    tok = AutoTokenizer.from_pretrained(src)
    ref = AutoModelForCausalLM.from_pretrained(src, dtype=dtype).to(dev).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    tag = G.tag_of(args.model, ref.config)
    test = A.with_bos(qz.token_ids(tok, "test", tag)[: 64 * 512].view(64, 512), tok, ref)
    mix = {s: A.with_bos(t, tok, ref) for s, t in A.TextMix(tok, tag).windows("eval", 8, 512, seed=2).items()}
    w = G.work(args.model)
    out = {"model": args.model, "dtype": args.dtype, "windows": {"wikitext": [64, 512], "mix": [8, 512]}, "checkpoints": {}}
    n_text = sum(p.numel() for p in text_params(ref).values())
    out["original"] = {"bytes": 2 * n_text, "weights": n_text, "wikitext": qz.evaluate(ref, None, test, 4),
                       "mix": {s: qz.evaluate(ref, None, t, 4) for s, t in mix.items()}}
    tasks = [t for t in args.tasks.split(",") if t]
    limits = {k: int(n) for k, n in (x.split("=") for x in args.task_limits.split(",") if x)}

    def bench(m):
        if not tasks:
            return {}
        from eval_tasks import run_tasks
        r = run_tasks(m, tok, tasks, dev, limits, log, bos=A.needs_bos(m))
        return {"tasks": {t: {k: v for k, v in x.items() if k != "preds"} for t, x in r.items()}}
    out["original"] |= bench(ref)
    log(f"original: {n_text / 1e9:.3f}B text weights, WikiText ppl {out['original']['wikitext']['ppl']:.4f}, tasks {out['original'].get('tasks')}")
    model = copy.deepcopy(ref)
    mapping = None
    for spec in args.checkpoints.split(","):
        kind, _, what = spec.partition(":")
        t0 = time.time()
        for n, p in model.named_parameters():  # back to the original before each checkpoint
            p.data.copy_(dict(ref.named_parameters())[n].data)
        if kind in ("gguf", "gguf-local"):
            if mapping is None:
                if not (w / "bf16.gguf").exists():
                    G.sh(["python", G.LLAMA / "convert_hf_to_gguf.py", src, "--outtype", "bf16", "--outfile", w / "bf16.gguf"])
                mapping = gguf_map(ref, w / "bf16.gguf", w / "gguf_map.json")
                out["gguf map"] = json.loads((w / "gguf_map.json").read_text()) | {"map": len(mapping)}
            if kind == "gguf":  # <owner>/<repo>/<file pattern>
                parts = what.split("/")
                path = sorted(f for f in snapshot("/".join(parts[:2])).glob("/".join(parts[2:])) if "mmproj" not in f.name)[0]  # not the vision projector
            else:
                path = w / what
            nbytes, hit = load_gguf(model, path, mapping)
        elif kind == "mlx":
            nbytes, hit = load_mlx(model, what)
        elif kind == "ct":
            nbytes, hit = load_ct(model, what, dtype)
        elif kind == "dense":  # a compiler plan's quantized weights (quantize.py --save), at the bits the plan counted
            d = torch.load(what, mmap=True)
            params = dict(model.named_parameters())
            for n, W in d["params"].items():
                params[n].data.copy_(W.to(params[n].dtype))
            nbytes, hit = d["text weights"]["bits"] / 8, len(d["params"])
        else:
            raise SystemExit(f"unknown checkpoint kind {kind}")
        r = {"kind": kind, "source": what, "bytes": nbytes, "GB": nbytes / 1e9, "bits per text weight": 8 * nbytes / n_text, "tensors loaded": hit,
             "wikitext": qz.evaluate(model, ref, test, 4), "mix": {s: qz.evaluate(model, ref, t, 4) for s, t in mix.items()}}
        r["mix kl"] = sum(v["kl"] for v in r["mix"].values()) / len(r["mix"])
        r |= bench(model)
        out["checkpoints"][spec] = r
        log(f"{spec}: {nbytes / 1e9:.3f} GB ({r['bits per text weight']:.3f} bits/weight, {hit} tensors), WikiText ppl {r['wikitext']['ppl']:.4f} "
            f"KL {r['wikitext']['kl']:.4f}, mix KL {r['mix kl']:.4f} ({time.time() - t0:.0f}s)")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
