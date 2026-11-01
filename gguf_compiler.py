"""The compression compiler for llama.cpp: the menu is llama.cpp's own tensor types,
measured as llama.cpp makes them, and the plan is a llama-quantize command, so the result runs in llama.cpp, Ollama and
LM Studio. Runs inside modal_gguf.py's image (llama.cpp built with CUDA, at the commit in /llama.cpp/COMMIT).

    prepare   HF → GGUF (bf16) with llama.cpp's convert_hf_to_gguf.py; an importance matrix (llama-imatrix) on the text mix's
              calibration split; every menu type everywhere (--pure, with the importance matrix; embedding and output head
              forced to the type too); llama.cpp's stock mixes for comparison; evaluation text (WikiText-2 test, the mix's
              held-out split)
    measure   for each type, every tensor of the GGUF dequantized (transformers' GGUF loader, which undoes llama.cpp's q/k
              permutation) and swapped alone into the Hugging Face model: the KL to the original on held-out windows, the
              same as compiler.py measure; bits = the tensor's bytes in the GGUF. Output: a measurement compiler.py
              allocate reads (items named as compiler.py's, options = llama.cpp types)
    build     a plan → llama-quantize with a --tensor-type override per tensor (and the embedding and output types)
    evaluate  llama-perplexity: perplexity and KL to the bf16 GGUF's logits, on WikiText-2 test and on the mix

    python gguf_compiler.py prepare --model unsloth/Llama-3.2-3B
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HF = Path("/root/.cache/huggingface")
LLAMA = Path("/llama.cpp")
BIN = LLAMA / "build" / "bin"
MENU = ["Q2_K", "IQ3_XXS", "Q3_K", "IQ4_XS", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "BF16"]
STOCK = ["Q2_K", "Q3_K_M", "IQ4_XS", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]  # llama.cpp's mixes, the baselines (IQ4_XS: the type's own mix)


def log(msg):
    print(msg, flush=True)


def sh(cmd, **kw):
    cmd = [str(c) for c in cmd]
    log("$ " + " ".join(cmd)[:600])
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode:
        log(p.stdout[-3000:] + p.stderr[-3000:])
        raise SystemExit(f"failed: {cmd[0]}")
    log(f"  ({time.time() - t0:.0f}s)")
    return p.stdout + p.stderr


def model_dir(repo):
    return glob.glob(str(HF / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / "*/"))[0]


def work(repo):
    w = HF / "gguf" / repo.split("/")[-1]
    w.mkdir(parents=True, exist_ok=True)
    return w


def tag_of(repo, cfg):
    return "qwen35" if "qwen3_5" in cfg.model_type else repo.split("/")[-1].lower()


def prepare(args):
    from transformers import AutoConfig, AutoTokenizer
    import compiler_arch as A
    w = work(args.model)
    src = model_dir(args.model)
    base = w / "bf16.gguf"
    if not base.exists():
        sh(["python", LLAMA / "convert_hf_to_gguf.py", src, "--outtype", "bf16", "--outfile", base])
    tok = AutoTokenizer.from_pretrained(src)
    mix = A.TextMix(tok, tag_of(args.model, AutoConfig.from_pretrained(src)))
    calib = w / "calib.txt"  # the text mix's calibration split, as text: every source equally (64 windows of 512 tokens)
    calib.write_text("\n\n".join(tok.decode(x) for x in mix.calibration(64, 512)))
    imat = w / "imatrix.gguf"
    if not imat.exists():
        sh([BIN / "llama-imatrix", "-m", base, "-f", calib, "-o", imat, "-c", "512", "--chunks", "64", "-ngl", "999"])
    for t in ([] if args.skip_pure else MENU):
        f = w / f"pure_{t}.gguf"
        if not f.exists():
            try:  # a type llama.cpp refuses for this model (IQ3_XXS needs importance data for every tensor; Gemma 4's
                # KV-shared layers keep key/value tensors no token reaches) is left off the menu
                sh([BIN / "llama-quantize", "--imatrix", imat, "--pure", "--token-embedding-type", t, "--output-tensor-type", t, base, f, t])
            except SystemExit:
                log(f"  {t}: refused by llama-quantize for this model; not on the menu")
                if f.exists():
                    f.rename(f.with_suffix(".gguf.broken"))  # set aside, not deleted
    for t in STOCK:
        f = w / f"stock_{t}.gguf"
        if not f.exists():
            sh([BIN / "llama-quantize", "--imatrix", imat, base, f, t])
        (w / f"stock_{t}.json").write_text(json.dumps(gguf_tensors(f)))  # each tensor's bytes, for the budgets
    from datasets import load_dataset
    (w / "wikitext_test.txt").write_text("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]))
    ev = mix.windows("eval", 8, 512, seed=2)  # the harness's held-out mix windows, sources interleaved so any prefix covers all
    (w / "mix_eval.txt").write_text("\n\n".join(tok.decode(ev[s][k]) for k in range(8) for s in mix.sources))
    (w / "manifest.json").write_text(json.dumps({"model": args.model, "commit": (LLAMA / "COMMIT").read_text().strip(),
                                                  "menu": MENU, "stock": STOCK}, indent=1))
    log(f"prepared {w}")


def gguf_tensors(path):
    """{gguf tensor name: (bytes, shape)} from a GGUF file."""
    from gguf import GGUFReader
    r = GGUFReader(str(path))
    return {t.name: (int(t.n_bytes), tuple(int(x) for x in t.shape)) for t in r.tensors}


def unpermute(W, n):
    """Undo convert_hf_to_gguf.py's rotary permutation of a Llama q or k matrix (n = its heads)."""
    rows, cols = W.shape
    return W.reshape(n, rows // n // 2, 2, cols).swapaxes(1, 2).reshape(rows, cols)


def dequantized(path, names, cfg):
    """{gguf tensor name: float32 (out, in) tensor} for `names`, in Hugging Face's layout."""
    import numpy as np
    from gguf import GGUFReader, dequantize
    r = GGUFReader(str(path))
    arch = str(bytes(r.fields["general.architecture"].parts[-1]), "utf-8")
    out = {}
    for t in r.tensors:
        if t.name not in names:
            continue
        a = dequantize(t.data, t.tensor_type).astype(np.float32)
        W = torch.from_numpy(a.reshape(int(t.shape[1]), int(t.shape[0])) if len(t.shape) == 2 else a)
        if arch == "llama" and t.name.endswith("attn_q.weight"):
            W = unpermute(W, cfg.num_attention_heads)
        if arch == "llama" and t.name.endswith("attn_k.weight"):
            W = unpermute(W, cfg.num_key_value_heads)
        out[t.name] = W
    return out


def measure(args):
    """Every llama.cpp type, tensor by tensor, swapped alone into the Hugging Face model: KL to the original."""
    import compiler_arch as A
    from transformers import AutoModelForCausalLM, AutoTokenizer
    w = work(args.model)
    src = model_dir(args.model)
    tok = AutoTokenizer.from_pretrained(src)
    dev = "cuda"
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mix = A.TextMix(tok, tag_of(args.model, model.config))
    per = max(1, -(-args.windows // len(mix.sources)))
    wd = mix.windows("calib", per, 512, seed=1)  # compiler.py measure's held-out windows
    held = A.with_bos(torch.cat([wd[s] for s in mix.sources])[: args.windows], tok, model)
    bs = 4
    with torch.no_grad():
        ref_lp = [F.log_softmax(model(held[b : b + bs].to(dev), use_cache=False).logits.float(), -1) for b in range(0, len(held), bs)]

    @torch.no_grad()
    def kl():
        tot = n = 0.0
        for b, lr in zip(range(0, len(held), bs), ref_lp):
            lp = F.log_softmax(model(held[b : b + bs].to(dev), use_cache=False).logits.float(), -1)
            tot += (lr.exp() * (lr - lp)).sum().item()
            n += lp.shape[0] * lp.shape[1]
        return tot / n
    floor = kl()
    import public_eval as P
    lpath, layers = A.decoder_layers(model)
    emb, head, tied = A.head_and_embedding(model)
    ple = A.per_layer_embedding(model)
    params = dict(model.named_parameters())
    if not (w / "bf16.gguf").exists():
        sh(["python", LLAMA / "convert_hf_to_gguf.py", src, "--outtype", "bf16", "--outfile", w / "bf16.gguf"])
    mapping = P.gguf_map(model, w / "bf16.gguf", w / "gguf_map.json")  # gguf tensor → (parameter, layout, scale), by content

    def item_of(pname):
        """The compiler's item for a parameter: a matrix of decoder layer i, the embedding, head or per-layer table, or
        another matrix of the text model (Gemma 4's per-layer projection)."""
        P_ = params[pname]
        if P_ is emb.weight:
            return "embed.0", "embed", "embed", None
        if head is not None and not tied and P_ is head.weight:
            return "head.0", "head", "head", None
        if ple is not None and P_ is ple.weight:
            return "ple.0", "ple", "ple", None
        pre = lpath + "."
        if pname.startswith(pre):
            i, rel = pname[len(pre):].split(".", 1)
            return f"{i}.{rel[: -len('.weight')]}", "linear", rel[: -len(".weight")], int(i)
        return f"other.{pname[: -len('.weight')]}", "linear", "other", None
    out = {"model": args.model, "method": "gguf", "data": "mix", "held-out windows": [len(held), 512], "kl floor (nothing quantized)": floor,
           "tied": tied, "embed buckets": [emb.weight.shape[0]], "embed order": "frequency in the text mix", "device": dev,
           "manifest": json.loads((w / "manifest.json").read_text()), "records": []}
    recs = out["records"]
    from gguf import GGUFReader, dequantize

    def readable(path):
        try:
            GGUFReader(str(path))
            return True
        except Exception:
            path.rename(path.with_suffix(".gguf.broken"))  # a half-written file from a refused type: set aside
            log(f"  {path.name}: unreadable, set aside")
            return False
    for t in [t for t in MENU if (w / f"pure_{t}.gguf").exists() and readable(w / f"pure_{t}.gguf")]:
        t0 = time.time()
        path = w / f"pure_{t}.gguf"
        n_hit = 0
        for gt in GGUFReader(str(path)).tensors:
            if gt.name not in mapping:
                continue
            pname, tr, c = mapping[gt.name]
            P_ = params[pname]
            X = torch.from_numpy(dequantize(gt.data, gt.tensor_type).astype(np.float32))
            X = X.reshape(int(gt.shape[1]), int(gt.shape[0])) if len(gt.shape) >= 2 else X
            X = X.T if tr == "T" else unpermute(X, int(tr.split(":")[1])) if tr.startswith("unpermute") else X
            orig = P_.data.clone()
            P_.data.copy_((c * X).reshape(P_.shape).to(P_.dtype))
            d = kl() - floor
            P_.data.copy_(orig)
            item, kind, role, layer = item_of(pname)
            rec = {"item": item, "kind": kind, "role": role, "option": t, "bits": 8.0 * int(gt.n_bytes), "weights": P_.numel(), "dkl": d,
                   "gguf tensor": gt.name, "stored type": gt.tensor_type.name, "shape": list(P_.shape)}
            if layer is not None:
                rec["layer"] = layer
            if kind != "linear":
                rec |= {"bucket": 0, "rows": P_.shape[0]}
            recs.append(rec)
            n_hit += 1
        log(f"{t}: {n_hit} tensors in {time.time() - t0:.0f}s")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    log(f"done: {len(recs)} records")


def gguf_suffix(hf_suffix):
    """The tail of llama.cpp's tensor name for a Hugging Face parameter (gguf-py's TensorNameMap, without the arch)."""
    m = re.match(r"layers\.(\d+)\.(.+)\.weight$", hf_suffix)
    if not m:
        return {"embed_tokens.weight": "token_embd.weight", "lm_head.weight": "output.weight"}[hf_suffix]
    i, name = m.groups()
    tail = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
            "self_attn.qkv_proj": "attn_qkv", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
            "mlp.gate_up_proj": "ffn_up"}.get(name, name)
    return f"blk.{i}.{tail}.weight"


def build(args):
    """A plan (compiler.py allocate over a gguf measurement, --sens) → llama-quantize with a --tensor-type override for
    every tensor the plan places (the measurement says which GGUF tensor each item is)."""
    w = work(args.model)
    plan = json.loads(Path(args.plan).read_text())
    sens = json.loads(Path(args.sens).read_text())
    gname = {r["item"]: r["gguf tensor"] for r in sens["records"]}
    choice = dict(plan["linears"])
    for kind in ("embed", "head", "ple"):
        if plan.get(kind):
            choice[f"{kind}.0"] = plan[kind]["programs"][0]
    over, want = [], {}
    for item, t in sorted(choice.items()):
        if item in gname:
            want[gname[item]] = t.upper()
            over += ["--tensor-type", f"^{re.escape(gname[item])}$={t.lower()}"]
    out = w / f"plan_{Path(args.plan).stem}.gguf"
    sh([BIN / "llama-quantize", "--imatrix", w / "imatrix.gguf", *over, w / "bf16.gguf", out, args.default])
    from gguf import GGUFReader  # the check: every tensor got the plan's type (llama.cpp falls back where a type cannot apply)
    got = {t.name: t.tensor_type.name for t in GGUFReader(str(out)).tensors}
    off = {n: (want[n], got.get(n)) for n in want if got.get(n) != want[n]}
    log(f"built {out} ({out.stat().st_size / 1e9:.3f} GB): {len(want) - len(off)} of {len(want)} tensors as planned; "
        f"different: {list(off.items())[:8]}")
    return out


def evaluate(args):
    """llama-perplexity on WikiText-2 test and the mix: perplexity, and KL to the bf16 GGUF's logits."""
    w = work(args.model)
    res = {}
    tmp = Path("/tmp/kld")  # the base logits are large (tokens × vocabulary × 2 bytes): local disk, 64 chunks of 512 per text
    tmp.mkdir(exist_ok=True)
    for text in ("wikitext_test", "mix_eval"):
        base_kld = tmp / f"base_{text}.kld"
        common = ["-c", "512", "--chunks", "64", "-ngl", "999"]
        if not base_kld.exists():
            sh([BIN / "llama-perplexity", "-m", w / "bf16.gguf", "-f", w / f"{text}.txt", *common, "--kl-divergence-base", base_kld])
        for f in args.files.split(","):
            g = w / f
            o = sh([BIN / "llama-perplexity", "-m", g, "-f", w / f"{text}.txt", *common, "--kl-divergence", "--kl-divergence-base", base_kld])
            ppl = re.search(r"Mean PPL\(Q\)\s*:\s*([\d.]+)", o) or re.search(r"Final estimate: PPL = ([\d.]+)", o)
            kld = re.search(r"Mean\s+KLD:\s*([\d.]+)", o)
            res.setdefault(f, {"bytes": g.stat().st_size})[text] = {
                "ppl": float(ppl.group(1)) if ppl else None, "kl": float(kld.group(1)) if kld else None}
            log(f"{f} {text}: ppl {res[f][text]['ppl']}, KL {res[f][text]['kl']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


def inspect(args):
    """A GGUF's architecture, tensor count, bytes, and its first tensors' names, types and shapes."""
    from gguf import GGUFReader
    for f in args.files.split(","):
        path = Path(f if f.startswith("/") else sorted(glob.glob(str(HF / "hub" / f)))[0])
        r = GGUFReader(str(path))
        arch = str(bytes(r.fields["general.architecture"].parts[-1]), "utf-8")
        log(f"{path.name}: {arch}, {len(r.tensors)} tensors, {sum(int(t.n_bytes) for t in r.tensors) / 1e9:.3f} GB of tensors, file {path.stat().st_size / 1e9:.3f} GB")
        for t in r.tensors[:12] + [t for t in r.tensors if "per_layer" in t.name or "token_embd" in t.name][:6]:
            log(f"   {t.name} {t.tensor_type.name} {tuple(int(x) for x in t.shape)} {int(t.n_bytes)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "measure", "build", "evaluate", "inspect"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        s.add_argument("--dtype", default="float32")
        s.add_argument("--windows", type=int, default=16)
        s.add_argument("--plan", default="")
        s.add_argument("--sens", default="", help="build: the gguf measurement the plan was allocated from")
        s.add_argument("--default", default="Q4_K", help="build: the type of tensors the plan does not name")
        s.add_argument("--files", default="", help="evaluate: GGUF files in the model's work directory")
        s.add_argument("--out", default="")
        s.add_argument("--skip-pure", action="store_true", help="prepare: only the stock mixes (no type-everywhere files)")
    args = ap.parse_args()
    {"prepare": prepare, "measure": measure, "build": build, "evaluate": evaluate, "inspect": inspect}[args.cmd](args)


if __name__ == "__main__":
    main()
