"""The compression compiler, first version (results/COMPILER.md): a plan gives every stored piece of a model a program,
and the compiler finds the plan that loses least under a hardware target.

    measure    every item alone at every option: quantize it, leave the rest of the model exact, and measure the KL
               divergence to the original model on held-out training windows (never the test split). Items:
                   linear  one matrix ("12.mlp.down_proj"), options tr2 tr3 tr4 lm4 lm5 lm6 lm8 fp16
                   embed   a bucket of the tied embedding / output rows, ranked by token frequency in the training
                           split (quantize.embedding_buckets), options tr1 tr2 tr3 tr4 lm5 lm6 lm8 fp16
                   kv      one layer's cached keys or values (quantize.KVQuant), 2 3 4 5 6 8 or 16 bits
               With --options tr3 a matrix is measured at one width only; allocate predicts its other options from the
               matrices of the same role measured at every width (KL ÷ KL at tr3 is nearly constant per code: 10–90% of
               SmolLM2's 224 matrices within ±10%), and its bits from theirs (same role, same shape).
               --trellis-fit names the value table the tr options use (opgraph.codes.trellis_values: lloyd or ema); the
               plan made from the measurements names it too, and run builds with that table
    allocate   a target prices every option from three numbers — stored bits, cached bits per token of context, and
               decode operations — and sets a budget; dynamic programming over the items (a multiple-choice knapsack)
               finds the options with the least summed KL within it. Summing assumes the items' damage adds up; the
               run says how far it does.
    uniform    a plan with one program everywhere (the baselines), and shuffle, a plan's programs permuted among the
               matrices of the same role (q, k, …) across layers: the same bits, without the per-matrix information
    run        quantize.py --program plan --plan <file> quantizes sequentially to the plan, with cross-term
               targets so later matrices absorb earlier error, and evaluates on the test split

    python compiler.py measure --model HuggingFaceTB/SmolLM2-360M --part linear --layers 0-3 --out results/batch/ccSensL0.json
    python compiler.py allocate --sens 'results/batch/ccSens*.json' --target disk --items linear --bits 3.044 --out results/compiler/plans/x.json
    python compiler.py allocate --sens 'results/batch/ccSens*.json' --target laptop --items linear,embed,kv --match linear=tr3,embed=fp16,kv=16 --out ...

Targets (bits per decoded token; a target's price is stored/batch + decode work·ops/batch + context·cache):
    disk     stored bits: download and storage
    laptop   bits read per decoded token, one sequence at 2,048 tokens (decoding there is bandwidth-bound)
    server   bits read per decoded token, 32 sequences at 16,384 tokens that share every weight read
    npu      laptop, plus decode work priced at 2 bits per operation
The decode operations per weight (OPS) are assumed from instruction counts — a multiply-add, plus bit extraction and a
table read for a code — not measured with kernels.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LINEAR_OPTIONS = ["tr2", "tr3", "tr4", "lm4", "lm5", "lm6", "lm8", "fp16"]
EMBED_OPTIONS = ["tr1", "tr2", "tr3", "tr4", "lm5", "lm6", "lm8", "fp16"]
KV_OPTIONS = [2, 3, 4, 5, 6, 8, 16]
OPS = {"fp": 1.0, "lm": 2.0, "tr": 3.0, "rtn": 2.0, "gptq": 2.0, "gptaq": 2.0, "higgs": 3.0}  # assumed, per weight per token
TARGETS = {
    "disk": {"batch": 1, "ctx": 0, "ops": 0.0},
    "laptop": {"batch": 1, "ctx": 2048, "ops": 0.0},
    "server": {"batch": 32, "ctx": 16384, "ops": 0.0},
    "npu": {"batch": 1, "ctx": 2048, "ops": 2.0},
}


def log(msg):
    print(msg, flush=True)


def span(s, n):
    """'a-b' (inclusive), a comma list of numbers and ranges, or '' (all n), as a sorted list."""
    if not s:
        return list(range(n))
    out = set()
    for part in s.split(","):
        a, b = part.split("-") if "-" in part else (part, part)
        out |= set(range(int(a), min(int(b), n - 1) + 1))
    return sorted(out)


def model_dir(repo, path=""):
    return path or glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*/"))[0]


def menu(s, default):
    """The options to measure: a comma list (fp16 always added) or the default."""
    if not s:
        return default
    opts = s.split(",")
    return opts + ([] if "fp16" in opts else ["fp16"])


# ----------------------------------------------------------------------------- measure


def rate_bits(option, m, n, T, row_rotation=True, block=None):
    """Stored bits of an m × n matrix in `option` (incoherence + row scale + code), exactly as the codes count them: a
    16-bit scale per row, the randomized Hadamards' signs (rows only with a row rotation), b bits per weight, and for the
    trellis the starting state of each run of T outputs (12 bits instead of k)."""
    import quantize as qz
    fam, b, _ = qz.parse(option)
    if fam == "fp":
        return 16.0 * m * n
    if block:  # quantized in slices of `block` columns, each with its own row scales and rotation
        return (n // block) * rate_bits(option, m, block, T, row_rotation)
    bits = b * m * n + 16 * m + n + (m if row_rotation else 0)
    return bits + ((12 - b) * m * n / T if fam == "tr" else 0)


@torch.no_grad()
def ldlq_noise(W, H, d, seed, gen, row_rotation=True):
    """W plus an error shaped as LDLQ's: in the incoherent basis (the same randomized Hadamards the codes use), rounding
    noise η of variance d × (row RMS)² carried through H = A Aᵀ (A upper triangular, A = (I + U) D^½), E = η D^½ A⁻¹, so
    that tr(E H Eᵀ) = Σ D_jj η_j², as LDLQ's error is (QuIP, Theorem 1); then back to the original basis."""
    from opgraph.codes import rht
    m, n = W.shape
    Ri = rht(n, seed + 1)
    Ro = rht(m, seed) if row_rotation else None
    Wr = (Ro @ W @ Ri.T) if row_rotation else W @ Ri.T
    Hr = Ri @ H @ Ri.T
    Hr = Hr + 0.01 * torch.diag(Hr).mean() * torch.eye(n, dtype=Hr.dtype)  # GPTQ's damping
    Lc = torch.linalg.cholesky(Hr.flip(0, 1))  # flip(H) = L Lᵀ, so H = A Aᵀ with A = flip(L), upper triangular
    Au = Lc.flip(0, 1)
    S = Wr.pow(2).mean(1, keepdim=True).sqrt()
    eta = torch.randn(m, n, generator=gen, dtype=torch.float64) * S * d ** 0.5
    E = torch.linalg.solve_triangular(Au, eta * torch.diag(Au)[None, :], upper=True, left=False)
    E = (Ro.T @ E @ Ri) if row_rotation else E @ Ri
    return W + E


NOISE_D = 0.015  # the noise level measured (about a 3-bit code's normalized error); KL is linear in it


def measure(args):
    """Every item alone at every option (or, with --method noise, at LDLQ-shaped noise of one level, no quantization):
    the KL to the original model on held-out calibration-split windows, never the evaluation split."""
    import quantize as qz
    import compiler_arch as A
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t_all = time.time()
    path = model_dir(args.model, args.model_path)
    tok = AutoTokenizer.from_pretrained(path)
    dtype = getattr(torch, args.dtype)
    qcfg = {}
    if json.loads((Path(path) / "config.json").read_text()).get("quantization_config", {}).get("quant_method") == "mxfp4":
        from transformers import Mxfp4Config
        qcfg = {"quantization_config": Mxfp4Config(dequantize=True)}
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **qcfg).to(qz.DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = A.text_config(model.config)
    tag = "qwen35" if "qwen3_5" in model.config.model_type else args.model.split("/")[-1].lower()  # quantize.py's token caches
    T = args.calib_len
    mix = A.TextMix(tok, tag) if args.data == "mix" else None
    if mix:
        calib = mix.calibration(args.calib, T)
        per = max(1, -(-args.windows // len(mix.sources)))
        w = mix.windows("calib", per, T, seed=1)
        held = torch.cat([w[s] for s in mix.sources])[: args.windows]
    else:
        train = qz.token_ids(tok, "train", tag)
        starts = torch.randint(0, len(train) - T - 1, (args.calib,), generator=torch.Generator().manual_seed(0))
        calib = torch.stack([train[s : s + T] for s in starts])
        starts = torch.randint(0, len(train) - T - 1, (args.windows,), generator=torch.Generator().manual_seed(1))
        held = torch.stack([train[s : s + T] for s in starts])
    calib, held = A.with_bos(calib, tok, model), A.with_bos(held, tok, model)
    chunks = [calib[b : b + args.chunk].to(qz.DEV) for b in range(0, len(calib), args.chunk)]
    bs = args.kl_batch
    with torch.no_grad():
        ref_lp = [F.log_softmax(model(held[b : b + bs].to(qz.DEV), use_cache=False).logits.float(), -1) for b in range(0, len(held), bs)]

    @torch.no_grad()
    def kl():
        tot = n = 0.0
        for b, lr in zip(range(0, len(held), bs), ref_lp):
            lp = F.log_softmax(model(held[b : b + bs].to(qz.DEV), use_cache=False).logits.float(), -1)
            tot += (lr.exp() * (lr - lp)).sum().item()
            n += lp.shape[0] * lp.shape[1]
        return tot / n

    t0 = time.time()
    floor = kl()
    _, layers = A.decoder_layers(model)
    stages = A.discover(model, chunks[0][:1, :64])
    method = args.method
    qz.C.TRELLIS_FIT = args.trellis_fit or qz.C.TRELLIS_FIT
    out = {"model": args.model, "parts": args.part, "method": method, "data": args.data, "held-out windows": [len(held), T],
           "calibration": [len(calib), T], "kl floor (nothing quantized)": floor, "seconds per kl": round(time.time() - t0, 3),
           "device": qz.DEV, "trellis fit": qz.C.TRELLIS_FIT, "stages (layer 0)": A.stage_list_text(stages[0]), "records": []}
    if method == "noise":
        out["noise d"] = NOISE_D
    recs = out["records"]
    log(f"model {args.model}: KL floor {floor:.3g}, {out['seconds per kl']}s per KL; stages {out['stages (layer 0)']}")

    def save():
        out["seconds"] = round(time.time() - t_all)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))

    T_tr = 256 if cfg.hidden_size % 256 == 0 else 320
    codes = {}

    def trellis_for(rows, opts):
        Tm = T_tr if rows % T_tr == 0 else next(t for t in (320, 256, 128, 64) if rows % t == 0)
        return Tm, {k: codes.setdefault((k, Tm), qz.C.TrellisCode(12, k, T=Tm)) for k in sorted({qz.parse(o)[1] for o in opts if o.startswith("tr")})}
    parts = args.part.split(",")
    gen = torch.Generator().manual_seed(0)
    lay = span(args.layers, len(layers))

    def try_options(rec, opts, set_w, restore, quantize, noise, m, n, Tm, row_rotation=True, block=None):
        """Measure one item at each option, or once at the noise level (--method noise); fp16 is free."""
        recs.append(rec | {"option": "fp16", "bits": 16.0 * m * n, "dkl": 0.0})
        for o in ([x for x in opts if x != "fp16"] if method == "quant" else ["noise"]):
            t0 = time.time()
            set_w(quantize(o) if method == "quant" else noise())
            d = kl() - floor
            restore()
            recs.append(rec | {"option": o, "bits": rate_bits(o, m, n, Tm, row_rotation, block) if method == "quant" else 0.0,
                               "dkl": d, "seconds": round(time.time() - t0, 2)})

    lin_opts = menu(args.options, LINEAR_OPTIONS)
    if "linear" in parts or "experts" in parts:
        def capture(group):
            """Input statistics (and routed tokens) for the layers in `group` only, from one calibration pass that stops
            after the group's last layer: a big model's statistics for every layer at once do not fit in memory."""
            H, hooks, R = {}, [], {}
            for i in group:
                L = layers[i]
                for st in stages[i]:
                    if st["kind"] == "linear" and st["names"] and "linear" in parts:
                        def f(m, a, key=(i, st["names"][0])):
                            x = a[0].reshape(-1, a[0].shape[-1]).float()
                            s, c = H.get(key, (0, 0))
                            H[key] = (s + x.T @ x, c + len(x))
                        hooks.append(L.get_submodule(st["names"][0]).register_forward_pre_hook(f))
                    if st["kind"] == "experts" and "experts" in parts:
                        def g(m, a, k, key=(i, st["name"])):
                            args_ = list(a) + [k.get(nm) for nm in ("top_k_index", "router_indices") if nm in k]
                            R.setdefault(key, []).append((args_[0].reshape(-1, args_[0].shape[-1]).float().cpu(),
                                                          args_[1].reshape(-1, args_[1].shape[-1]).cpu()))
                        hooks.append(L.get_submodule(st["name"]).register_forward_pre_hook(g, with_kwargs=True))
            last = max(group)
            if last + 1 < len(layers):
                def stop(m, a, k):
                    raise qz.Stop()
                hooks.append(layers[last + 1].register_forward_pre_hook(stop, with_kwargs=True))
            with torch.no_grad():
                for x in chunks:
                    try:
                        model(x, use_cache=False)
                    except qz.Stop:
                        pass
            for h_ in hooks:
                h_.remove()
            return {k: (s / c).cpu().double() for k, (s, c) in H.items()}, R
        groups = [lay[k : k + 4] for k in range(0, len(lay), 4)]
        H, R, done_group = {}, {}, None
        for i in lay:
            t_l = time.time()
            g_ = next(gr for gr in groups if i in gr)
            if g_ is not done_group:  # the next group's statistics, the previous group's freed
                H, R = capture(g_)
                done_group = g_
            L = layers[i]
            for st in stages[i]:
                if st["kind"] == "linear" and "linear" in parts:
                    Hs = H.get((i, st["names"][0])) if st["names"] else None
                    for j, name in enumerate(st["names"]):
                        mod = L.get_submodule(name)
                        W = mod.weight.data.cpu().double()
                        orig = mod.weight.data.clone()
                        m, n = W.shape
                        Tm, tr = trellis_for(m, lin_opts)
                        seed = 1000 * i + 100 * A.block_seed(name) + j
                        rec = {"item": f"{i}.{name}", "kind": "linear", "layer": i, "role": name, "weights": W.numel(), "shape": [m, n], "T": Tm}

                        def set_w(Wq, mod=mod):
                            mod.weight.data = Wq.to(mod.weight.dtype).to(qz.DEV)
                        try_options(rec, lin_opts, set_w, lambda mod=mod, orig=orig: setattr(mod.weight, "data", orig),
                                    lambda o, W=W, seed=seed, tr=tr: qz.word(o, W, Hs, seed, tr).matrix(dtype=torch.float64),
                                    lambda W=W, seed=seed: ldlq_noise(W, Hs, NOISE_D, seed, gen), m, n, Tm)
                    if st["names"]:
                        log(f"  layer {i} {'+'.join(st['names'])}: " + ", ".join(f"{r['option']} {r['dkl']:.2e}" for r in recs[-len(lin_opts):] if r["item"] == f"{i}.{st['names'][0]}"))
                if st["kind"] == "experts" and "experts" in parts:
                    mod = L.get_submodule(st["name"])
                    X = torch.cat([x for x, _ in R[(i, st["name"])]])
                    K = torch.cat([k for _, k in R[(i, st["name"])]])
                    E = mod.gate_up_proj.shape[0]
                    count = torch.bincount(K.reshape(-1), minlength=E)[:E]
                    order = torch.sort(-count, stable=True).indices.tolist()
                    buckets = expert_buckets(E)
                    lo = 0
                    for k_b, size in enumerate(buckets):
                        ids = order[lo : lo + size]
                        lo += size
                        for part in ("gate_up", "down"):
                            mats, Hs_ = [], []
                            for e in ids:
                                gu, dn, t = qz.expert_matrices(mod, e, X.shape[-1])
                                tok_e = (K == e).any(-1).nonzero().squeeze(-1)
                                x = X[tok_e].to(qz.DEV)
                                inp = x if part == "gate_up" else qz.expert_hidden(mod, e, x.to(mod.gate_up_proj.dtype), t).float()
                                Wm = (gu if part == "gate_up" else dn).detach().cpu().double()
                                if len(inp) >= 8:
                                    He = (inp.T @ inp / len(inp)).cpu().double()
                                    He += 0.01 * torch.diag(He).mean() * torch.eye(len(He), dtype=He.dtype)
                                else:
                                    He = torch.eye(Wm.shape[1], dtype=torch.float64)
                                mats.append((e, Wm, t))
                                Hs_.append(He)
                            dst = mod.gate_up_proj if part == "gate_up" else mod.down_proj
                            orig = dst.data.clone()
                            m, n = mats[0][1].shape
                            Tm, tr = trellis_for(m, lin_opts)

                            def set_w(Ws, dst=dst, mats=mats):
                                for (e, _, t), Wq in zip(mats, Ws):
                                    dst.data[e] = (Wq.T if t else Wq).to(dst.dtype).to(dst.device)

                            def quantize(o, mats=mats, Hs_=Hs_, tr=tr, part=part):
                                return [qz.word(o, Wm, He, 1000 * i + 500 + 2 * e + (part == "down"), tr).matrix(dtype=torch.float64)
                                        for (e, Wm, _), He in zip(mats, Hs_)]

                            def noise(mats=mats, Hs_=Hs_, part=part):
                                return [ldlq_noise(Wm, He, NOISE_D, 1000 * i + 500 + 2 * e + (part == "down"), gen) for (e, Wm, _), He in zip(mats, Hs_)]
                            rec = {"item": f"{i}.{st['name']}.b{k_b}.{part}", "kind": "experts", "layer": i, "role": f"experts.{part}",
                                   "module": st["name"], "bucket": k_b, "part": part, "experts": ids, "tokens": int(count[ids].sum()),
                                   "weights": m * n * len(ids), "shape": [m, n], "T": Tm}
                            n_before = len(recs)
                            try_options(rec, lin_opts, set_w, lambda dst=dst, orig=orig: setattr(dst, "data", orig), quantize, noise, m, n, Tm)
                            for r in recs[n_before:]:
                                r["bits"] *= len(ids)
                        log(f"  layer {i} experts bucket {k_b} ({len(ids)} experts, {int(count[ids].sum())} routed tokens): "
                            + ", ".join(f"{r['part']} {r['option']} {r['dkl']:.2e}" for r in recs[-2 * len(lin_opts):] if r.get("bucket") == k_b and r["layer"] == i)[:400])
            log(f"layer {i}: {time.time() - t_l:.0f}s")
            save()
    if "embed" in parts:
        emb, head, tied = A.head_and_embedding(model)
        V = emb.weight.shape[0]
        sizes = qz.embedding_buckets(V)
        freq = mix.frequency(V) if mix else torch.bincount(qz.token_ids(tok, "train", tag).reshape(-1), minlength=V)[:V].double()
        order = torch.sort(-freq, stable=True).indices
        H_out = qz.head_inputs(model, chunks)
        out["embed buckets"] = sizes
        out["embed order"] = "frequency in the text mix" if mix else "frequency in the WikiText-2 training split"
        out["tied"] = tied
        emb_opts = menu(args.embed_options, EMBED_OPTIONS)
        tables = [("embed", emb.weight, qz.embedding_environment(H_out, "both" if tied else "lookup"), 7000)]
        if not tied:
            tables.append(("head", head.weight, qz.embedding_environment(H_out, "head"), 8000))
        ple = A.per_layer_embedding(model)
        if ple is not None:  # Gemma 4's per-layer input table: looked up per token, so plain squared error
            tables.append(("ple", ple.weight, None, 9000))
        ple_block = getattr(cfg, "hidden_size_per_layer_input", None)
        for kind, Wt, Hh, seed0 in tables:
            lo = 0
            for k, size in enumerate(sizes):
                rows = order[lo : lo + size]
                lo += size
                if k not in span(args.embed_buckets, len(sizes)):
                    continue
                ridx = rows.to(Wt.device)
                W = Wt.data[ridx].cpu().double()
                orig = Wt.data[ridx].clone()
                mass = float(freq[rows].sum() / freq.sum())
                blk = ple_block if kind == "ple" else None
                rec = {"item": f"{kind}.{k}", "kind": kind, "bucket": k, "rows": size, "frequency mass": mass, "weights": W.numel(),
                       "shape": list(W.shape), "T": qz.EMBED_T, "block": blk}

                def set_w(Wq, Wt=Wt, ridx=ridx):
                    Wt.data[ridx] = Wq.to(Wt.dtype).to(Wt.device)
                try_options(rec, emb_opts, set_w, lambda Wt=Wt, ridx=ridx, orig=orig: Wt.data.__setitem__(ridx, orig),
                            lambda o, W=W, Hh=Hh, k=k, seed0=seed0, kind=kind: qz.embed_word(W, Hh, o, seed0 + k, ple_block if kind == "ple" else None).matrix(dtype=torch.float64),
                            lambda W=W, Hh=Hh, k=k, seed0=seed0: ldlq_noise(W, Hh if Hh is not None else torch.eye(W.shape[1], dtype=torch.float64), NOISE_D, seed0 + k, gen, row_rotation=False),
                            W.shape[0], W.shape[1], qz.EMBED_T, row_rotation=False, block=blk)
                log(f"  {kind} bucket {k} ({size} rows, {mass:.1%} of tokens): " + ", ".join(f"{r['option']} {r['dkl']:.2e}" for r in recs[-len(emb_opts):] if r["item"] == f"{kind}.{k}"))
                save()
    if "kv" in parts:
        kvq = A.KVCache(model)
        for l in [l for l in span(args.kv_layers, len(layers)) if l in kvq.layers]:
            for kind in ("k", "v"):
                for b in KV_OPTIONS:
                    rec = {"item": f"kv.{l}.{kind}", "kind": "kv", "layer": l, "option": b, "kv bits per token": kvq.bits_per_token(b, l)}
                    if b >= 16:
                        recs.append(rec | {"dkl": 0.0})
                        continue
                    kvq.widths = {(l, kind): b}
                    recs.append(rec | {"dkl": kl() - floor})
                kvq.widths = {}
                log(f"  cache layer {l} {kind}: " + ", ".join(f"{r['option']} {r['dkl']:.2e}" for r in recs[-len(KV_OPTIONS):]))
        kvq.remove()
        save()
    save()
    log(f"done: {len(recs)} records, {out['seconds']}s")


def expert_buckets(E):
    """Experts per layer ranked by routed tokens, in buckets of 1/8, 1/8, 1/4 and 1/2 (fewer when there are few experts)."""
    if E < 8:
        return [E]
    sizes = [E // 8, E // 8, E // 4]
    return sizes + [E - sum(sizes)]


# ----------------------------------------------------------------------------- allocate


def load_sens(pattern):
    """Every record in the files matching `pattern`, grouped by item, and the files' metadata (the embedding buckets, and the
    trellis table the tr options were measured with: Lloyd steps in files from before there was a choice)."""
    items, meta = {}, {}
    for f in sorted(glob.glob(pattern)):
        d = json.loads(Path(f).read_text())
        meta.setdefault("model", d.get("model"))
        fit = d.get("trellis fit", "lloyd")
        assert meta.setdefault("trellis fit", fit) == fit, f"{f} was measured with the {fit} trellis table, earlier files with {meta['trellis fit']}"
        for k in ("embed buckets", "embed order", "tied"):
            if k in d:
                meta[k] = d[k]
        for r in d["records"]:
            items.setdefault(r["item"], {})[str(r["option"])] = r
    return items, meta


DEFAULT_RATIOS = {"tr2": 4.04, "tr4": 0.263, "lm4": 0.48, "lm5": 0.125, "lm6": 0.032, "lm8": 0.002}  # KL ÷ KL at tr3, SmolLM2 and Qwen3.5 medians


def fill_predictions(items):
    """Options an item was not measured at, predicted. Its anchor is tr3 if measured, else the noise level (--method noise).
    KL = KL at the anchor × the median, over items of the same role measured at both (pooled over all roles of the kind
    when its own role has none), of KL(option) ÷ KL(anchor); for a tr3 anchor with no reference at all, SmolLM2's and
    Qwen3.5's medians. Bits come from the shape (rate_bits), exactly."""
    import statistics
    group = lambda r: r.get("role", r["kind"])
    measured = lambda opts, o: o in opts and not opts[o].get("predicted")
    refs = {}
    for opts in items.values():
        r = next(iter(opts.values()))
        if r["kind"] != "kv":
            refs.setdefault(group(r), []).append(opts)
            refs.setdefault("kind:" + r["kind"], []).append(opts)
    wanted = {r["kind"]: set() for opts in items.values() for r in opts.values()}
    for opts in items.values():
        for o, r in opts.items():
            if o not in ("noise",) and not r.get("predicted"):
                wanted[r["kind"]].add(o)
    n = 0
    for opts in items.values():
        r = next(iter(opts.values()))
        if r["kind"] == "kv":
            continue
        anchor = "tr3" if measured(opts, "tr3") else "noise" if measured(opts, "noise") else None
        if anchor is None:
            continue
        for o in sorted(wanted[r["kind"]] - set(opts)):
            ratio = None
            for g in (group(r), "kind:" + r["kind"]):
                ref = [f for f in refs.get(g, []) if measured(f, o) and measured(f, anchor) and f[anchor]["dkl"] > 0]
                if ref:
                    ratio = statistics.median(f[o]["dkl"] / f[anchor]["dkl"] for f in ref)
                    break
            if ratio is None and anchor == "tr3":
                ratio = DEFAULT_RATIOS.get(o)
            if ratio is None:
                continue
            m_, n_ = r["shape"]
            bits = rate_bits(o, m_, n_, r.get("T", 256), r["kind"] not in ("embed", "head", "ple"), r.get("block")) * (len(r["experts"]) if r["kind"] == "experts" else 1)
            opts[o] = {k: v for k, v in opts[anchor].items() if k not in ("rel err", "seconds")} | {"option": o, "dkl": opts[anchor]["dkl"] * ratio,
                                                                                                  "bits": bits, "predicted": True}
            n += 1
        opts.pop("noise", None)
    return n


def price(r, t):
    """One option's cost under target t, in bits per decoded token (stored bits for disk)."""
    if r["kind"] == "kv":
        return t["ctx"] * r["kv bits per token"]
    import quantize as qz
    fam = qz.parse(r["option"])[0] if (r["option"] == "fp16" or qz.PROGRAM.fullmatch(str(r["option"]))) else "other"
    return (r["bits"] + t["ops"] * OPS.get(fam, 2.0) * r["weights"]) / t["batch"]


def knapsack(groups, budget, cells=100_000):
    """groups: per item, a list of (cost, loss). The choice of one option per item with the least summed loss whose summed
    cost is at most `budget`: dynamic programming over the cost, in `cells` steps of what the cheapest plan leaves room
    for (costs rounded up, so the plan never exceeds the budget)."""
    lo = [min(c for c, _ in g) for g in groups]
    room = budget - sum(lo)
    if room < 0:
        raise SystemExit(f"the budget is {-room:.4g} below the cheapest plan")
    unit = room / cells if room > 0 else 1.0
    steps = [[int(math.ceil((c - m) / unit - 1e-9)) for c, _ in g] for g, m in zip(groups, lo)]
    dp = np.zeros(cells + 1)
    pick = np.zeros((len(groups), cells + 1), dtype=np.int8)
    for i, g in enumerate(groups):
        new = np.full(cells + 1, np.inf)
        arg = np.zeros(cells + 1, dtype=np.int8)
        for o, ((_, v), w) in enumerate(zip(g, steps[i])):
            if w > cells:
                continue
            cand = np.full(cells + 1, np.inf)
            cand[w:] = dp[: cells + 1 - w] + v
            better = cand < new
            new[better], arg[better] = cand[better], o
        dp, pick[i] = new, arg
    c, choice = cells, []
    for i in range(len(groups) - 1, -1, -1):
        o = int(pick[i, c])
        choice.append(o)
        c -= steps[i][o]
    return choice[::-1]


def program_of(r, cross=True):
    o = str(r["option"])
    if r["kind"] == "kv":
        return int(o)
    import quantize as qz  # only our own codes take the cross-term target (llama.cpp types are named as llama.cpp does)
    ours = qz.PROGRAM.fullmatch(o) is not None
    return o + "x" if r["kind"] in ("linear", "experts") and cross and ours else o


def summarize(chosen, items):
    """Bits per weight by role and by layer, the embedding buckets' programs, the cache widths."""
    by = {}
    for name, o in chosen.items():
        r = items[name][o]
        if r["kind"] in ("linear", "experts"):
            layer = r.get("layer", int(name.split(".")[0]) if name.split(".")[0].isdigit() else None)  # gguf records name it only
            keys = (f"role {r['role']}",) + ((f"layer {layer:02d}",) if layer is not None else ()) + ("all linears" if r["kind"] == "linear" else "all experts",)
            for key in keys + ((f"experts bucket {r['bucket']}",) if r["kind"] == "experts" else ()):
                b, n = by.get(key, (0.0, 0))
                by[key] = (b + r["bits"], n + r["weights"])
    out = {k: round(b / n, 3) for k, (b, n) in sorted(by.items())}
    for kind in ("embed", "head", "ple"):
        emb = sorted((r.get("bucket", 0), o) for name, o in chosen.items() if (r := items[name][o])["kind"] == kind)
        if emb:
            e = [items[f"{kind}.{k}"][o] for k, o in emb]
            out[f"{kind} programs by bucket (most frequent first)"] = [o for _, o in emb]
            out[f"{kind} bits per weight"] = round(sum(r["bits"] for r in e) / sum(r["weights"] for r in e), 3)
    kv = {name: o for name, o in chosen.items() if items[name][o]["kind"] == "kv"}
    if kv:
        out["cache widths (keys)"] = [int(kv[k]) for k in sorted(kv, key=lambda s: int(s.split(".")[1])) if k.endswith(".k")]
        out["cache widths (values)"] = [int(kv[k]) for k in sorted(kv, key=lambda s: int(s.split(".")[1])) if k.endswith(".v")]
    return out


def write_plan(chosen, items, meta, t_name, t, budget, out_path, extra=None):
    plan = {"model": meta.get("model"), "target": t_name, "target spec": t, "budget": budget,
            "cost": sum(price(items[n][o], t) for n, o in chosen.items()),
            "predicted kl": sum(max(items[n][o]["dkl"], 0.0) for n, o in chosen.items()), "trellis fit": meta["trellis fit"],
            "linears": {n: program_of(items[n][o]) for n, o in sorted(chosen.items()) if items[n][o]["kind"] == "linear"}}
    for kind in ("embed", "head", "ple"):  # the tied table (or the untied input embedding), an untied head, a per-layer table
        emb = sorted((items[n][o]["bucket"], o) for n, o in chosen.items() if items[n][o]["kind"] == kind)
        if emb:
            assert [k for k, _ in emb] == list(range(len(meta["embed buckets"]))), f"every {kind} bucket needs a measurement"
            plan[kind] = {"sizes": meta["embed buckets"], "programs": [o for _, o in emb],
                          "order": meta.get("embed order", "frequency in the WikiText-2 training split")}
    for n, o in sorted(chosen.items()):  # experts: per module, the usage buckets' experts and their gate_up and down programs
        r = items[n][o]
        if r["kind"] == "experts":
            e = plan.setdefault("experts", {}).setdefault(f"{r['layer']}.{r['module']}", {"buckets": [], "gate_up": [], "down": []})
            while len(e["buckets"]) <= r["bucket"]:
                e["buckets"].append(None)
                e["gate_up"].append(None)
                e["down"].append(None)
            e["buckets"][r["bucket"]] = r["experts"]
            e[r["part"]][r["bucket"]] = program_of(r)
    kv = {}
    for n, o in chosen.items():
        if items[n][o]["kind"] == "kv":
            _, l, kind = n.split(".")
            kv.setdefault(l, {})[kind] = int(o)
    if kv:
        plan["kv"] = kv
    plan["summary"] = summarize(chosen, items)
    plan |= extra or {}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(plan, indent=1))
    return plan


def allocate(args):
    items, meta = load_sens(args.sens)
    n_pred = fill_predictions(items)
    if n_pred:
        log(f"{n_pred} options predicted from the items measured at every width")
    kinds = args.items.split(",")
    t = TARGETS[args.target] | json.loads(args.target_override or "{}")
    names = sorted(n for n, opts in items.items() if next(iter(opts.values()))["kind"] in kinds)
    if args.budget:  # an absolute budget, in the target's units (bits for disk): a stock GGUF's size, say
        budget, ref_choice, ref_kl = args.budget, None, None
    elif args.match:  # the budget: what a reference plan costs under this target
        ref = dict(kv.split("=") for kv in args.match.split(","))
        ref_choice = {n: ref[items[n][next(iter(items[n]))]["kind"]] for n in names}
        budget = sum(price(items[n][o], t) for n, o in ref_choice.items()) * args.scale
        ref_kl = sum(max(items[n][o]["dkl"], 0.0) for n, o in ref_choice.items())
    else:  # bits per weight over the weight items
        budget = args.bits * sum(next(iter(items[n].values()))["weights"] for n in names if next(iter(items[n].values()))["kind"] != "kv")
        ref_choice, ref_kl = None, None
    opts = {n: [o for o in items[n] if args.options is None or o in args.options.split(",") or o == "fp16" or o == "16"] for n in names}
    groups = [[(price(items[n][o], t), max(items[n][o]["dkl"], 0.0)) for o in opts[n]] for n in names]
    pick = knapsack(groups, budget)
    chosen = {n: opts[n][k] for n, k in zip(names, pick)}
    extra = {"reference": {"choice": args.match, "predicted kl": ref_kl}} if ref_choice else {}
    plan = write_plan(chosen, items, meta, args.target, t, budget, args.out, extra)
    log(f"{args.out}: cost {plan['cost']:.4g} of budget {budget:.4g}, predicted KL {plan['predicted kl']:.4f}"
        + (f" (reference {ref_kl:.4f})" if ref_kl is not None else ""))
    for k, v in plan["summary"].items():
        if not k.startswith("layer "):
            log(f"  {k}: {v}")


# ----------------------------------------------------------------------------- baselines


def uniform(args):
    """One program for every matrix (and, optionally, every embedding bucket and cache width): from a measurement
    (--sens: its items, so any architecture, experts included) or from a Llama-type or Qwen3.5 config."""
    import quantize as qz
    if args.sens:
        items, meta = load_sens(args.sens)
        plan = {"model": meta.get("model"), "target": "uniform", "trellis fit": meta["trellis fit"], "linears": {}}
        for name, opts in sorted(items.items()):
            r = next(iter(opts.values()))
            if r["kind"] == "linear":
                plan["linears"][name] = args.linear
            elif r["kind"] == "experts":
                e = plan.setdefault("experts", {}).setdefault(f"{r['layer']}.{r['module']}", {"buckets": [], "gate_up": [], "down": []})
                while len(e["buckets"]) <= r["bucket"]:
                    e["buckets"].append(None)
                    e["gate_up"].append(args.linear)
                    e["down"].append(args.linear)
                e["buckets"][r["bucket"]] = r["experts"]
            elif r["kind"] in ("embed", "head", "ple") and args.embed:
                plan.setdefault(r["kind"], {"sizes": meta["embed buckets"], "programs": [args.embed] * len(meta["embed buckets"]),
                                            "order": meta.get("embed order", "")})
            elif r["kind"] == "kv" and args.kv:
                plan.setdefault("kv", {}).setdefault(str(r["layer"]), {})[name.split(".")[2]] = args.kv
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(plan, indent=1))
        log(f"{args.out}: {args.linear} everywhere ({len(plan['linears'])} matrices, {len(plan.get('experts', {}))} experts modules)"
            + (f", embedding {args.embed}" if args.embed else "") + (f", cache {args.kv}" if args.kv else ""))
        return
    cfg = json.loads((Path(model_dir(args.model)) / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    types = cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"]  # Qwen3.5: three DeltaNet layers to one attention
    mix = ["linear_attn" if t == "linear_attention" else "self_attn" for t in types]
    linears = {f"{i}.{blk}.{n}": args.linear for i in range(cfg["num_hidden_layers"]) for blk in (mix[i], "mlp")
               for stage in qz.STAGES[blk] for n in stage}
    plan = {"model": args.model, "target": "uniform", "trellis fit": qz.C.TRELLIS_FIT, "linears": linears}
    if args.embed:
        sizes = qz.embedding_buckets(cfg["vocab_size"])
        plan["embed"] = {"sizes": sizes, "programs": [args.embed] * len(sizes), "order": "frequency in the WikiText-2 training split"}
    if args.kv:
        plan["kv"] = {str(l): {"k": args.kv, "v": args.kv} for l in range(cfg["num_hidden_layers"]) if mix[l] == "self_attn"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(plan, indent=1))
    log(f"{args.out}: {args.linear} everywhere" + (f", embedding {args.embed}" if args.embed else "") + (f", cache {args.kv}" if args.kv else ""))


def shuffle(args):
    """The plan's matrix programs permuted among matrices of the same role across layers (same shapes, so the same bits)."""
    plan = json.loads(Path(args.plan).read_text())
    rng = random.Random(args.seed)
    roles = {}
    for name in plan["linears"]:
        roles.setdefault(name.split(".", 1)[1], []).append(name)
    new = {}
    for role, names in roles.items():
        progs = [plan["linears"][n] for n in names]
        rng.shuffle(progs)
        new |= dict(zip(names, progs))
    plan["linears"] = dict(sorted(new.items()))
    plan["target"] = f"{plan.get('target')} shuffled within roles (seed {args.seed})"
    plan.pop("predicted kl", None)
    Path(args.out).write_text(json.dumps(plan, indent=1))
    log(f"{args.out}: {args.plan} shuffled within roles")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    m.add_argument("--model-path", default="", help="a local model directory (default: the Hugging Face cache snapshot)")
    m.add_argument("--part", default="linear", help="linear, experts, embed (and head, if untied), kv (comma-separated)")
    m.add_argument("--options", default="", help="matrix options to measure (default all; fp16 always)")
    m.add_argument("--embed-options", default="", help="embedding options to measure (default all; fp16 always)")
    m.add_argument("--kl-batch", type=int, default=8, help="held-out windows per forward in the KL")
    m.add_argument("--method", choices=["quant", "noise"], default="quant", help="quantize each item, or inject LDLQ-shaped noise once")
    m.add_argument("--data", choices=["wikitext", "mix"], default="wikitext", help="calibration, held-out windows and token frequencies")
    m.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    m.add_argument("--layers", default="", help="linear layers: a-b or a comma list (default all)")
    m.add_argument("--embed-buckets", default="", help="embedding buckets, a-b (default all)")
    m.add_argument("--kv-layers", default="", help="cache layers, a-b (default all)")
    m.add_argument("--calib", type=int, default=64)
    m.add_argument("--calib-len", type=int, default=512)
    m.add_argument("--chunk", type=int, default=8)
    m.add_argument("--windows", type=int, default=16, help="held-out training windows for the KL")
    m.add_argument("--trellis-fit", choices=["lloyd", "ema"], default="", help="how the tr options' value table is fitted (default: codes.TRELLIS_FIT)")
    m.add_argument("--out", required=True)
    a = sub.add_parser("allocate")
    a.add_argument("--sens", required=True, help="a glob of measure outputs")
    a.add_argument("--target", choices=sorted(TARGETS), default="disk")
    a.add_argument("--target-override", default="", help='JSON merged into the target, e.g. {"ctx": 8192}')
    a.add_argument("--items", default="linear", help="kinds the plan sets: linear, experts, embed, head, kv")
    a.add_argument("--bits", type=float, default=3.0, help="budget in bits per weight over the weight items (disk)")
    a.add_argument("--match", default="", help="budget = this plan's cost under the target: linear=tr3,embed=fp16,kv=16")
    a.add_argument("--scale", type=float, default=1.0, help="times the --match budget")
    a.add_argument("--budget", type=float, default=0.0, help="an absolute budget in the target's units (overrides --bits and --match)")
    a.add_argument("--options", default=None, help="restrict the menu (fp16 and 16 always allowed)")
    a.add_argument("--out", required=True)
    u = sub.add_parser("uniform")
    u.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    u.add_argument("--sens", default="", help="a glob of measure outputs: the plan's items from them")
    u.add_argument("--linear", required=True)
    u.add_argument("--embed", default="")
    u.add_argument("--kv", type=int, default=0)
    u.add_argument("--out", required=True)
    s = sub.add_parser("shuffle")
    s.add_argument("--plan", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    {"measure": measure, "allocate": allocate, "uniform": uniform, "shuffle": shuffle}[args.cmd](args)


if __name__ == "__main__":
    main()
