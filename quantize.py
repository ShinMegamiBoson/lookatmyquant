"""Quantize a model with one program or a compiler plan, then score it against the original.

opgraph's own forward pass covers plain transformers only. This runs any supported model through transformers and captures
each matrix's inputs with hooks, layer by layer and sublayer by sublayer (the "true sequential" order). It then calls
opgraph's per-matrix codes:

    rtn4     round to nearest, 4-bit, groups of 64
    gptq4    GPTQ 4-bit, groups of 64
    gptaq4   GPTQ 4-bit, groups of 64, cross-term (GPTAQ-style) target
    lm4      incoherence + Lloyd–Max 4-bit + LDLQ                      (SmolLM2: 15.98 at 4.02 bits)
    lm4x     the same with the cross-term target                       (SmolLM2: 15.94)
    tr3      incoherence + trellis 3-bit + LDLQ                        (SmolLM2: 16.90 at 3.04 bits)
    tr3x     the same with the cross-term target                       (SmolLM2: 16.89)
    tr2x     incoherence + trellis 2-bit + LDLQ, cross-term target     (SmolLM2: 23.89 at 2.05 bits)
    higgs4   HIGGS (Malinovskii et al. 2024, transformers' HiggsConfig), data-free: Hadamard groups of 1024 inputs, each
             scaled to unit RMS, pairs of weights rounded to transformers' 256-point Gaussian grid (p = 2); higgs3, higgs2
             use its 64- and 16-point grids. Where the inputs are not a multiple of 1024 the group is the largest power of
             two that divides them (2560: 512; 11008: 256) rather than padding. Dense reconstruction, 16-bit scale per group.

--ft-epochs E adds per-layer fine-tuning in the manner of QTIP's (and QuIP#'s): after each stage of a layer is quantized,
Adam trains the layer's remaining continuous parameters (the matrices not yet quantized, the norms, and a scale per row
and per column of every quantized matrix, QTIP's SU and SV) to reproduce the reference layer's output, and keeps the
best epoch on the last --ft-valid calibration windows. The input is the quantized stream (--ft-input quant: the layer
also learns to undo upstream error) or the reference stream (ref: QTIP's choice). The row and column scales are counted
at 16 bits.

Other models (--model NousResearch/Llama-2-7b-hf): plain transformers decoders run the same way (self_attn and mlp
stages; no rotation). --ctx sets the evaluation windows (QuIP#/QTIP report Llama-2 at 4096 over the whole test split:
--ctx 4096,2048 --eval 0), --calib-len the calibration windows, and --calib-source fineweb draws calibration from
FineWeb-Edu (the opgraph_corpus shard on the volume, decoded and re-tokenized) instead of WikiText-2's training split,
so the calibration is out of domain, as in the published methods (RedPajama, C4).

--rotation quarot | pinned first rewrites the residual stream exactly: every
norm's gain (1 + weight in Qwen3.5) folded into the matrices that read it (the final one into the output head, which
unties it from the embedding), the embedding and every reader rotated by Q, every writer (o_proj, the DeltaNet's
out_proj, down_proj) by Qᵀ, and randomized block Hadamards at runtime on the writers' inputs (per head: 256 for
attention, 128 for the DeltaNet; 512 for down_proj). Q is a randomized Hadamard (QuaRot-style) or a random rotation
that leaves the 4 massive-activation channels alone (pinned). The quantization reference is the rotated full-precision
model. The trellis runs over 256
outputs instead of 320 (Qwen3.5's output sizes are multiples of 256). Kept at 16 bits and not counted: the tied
embedding / output matrix, norms, the DeltaNet's short convolutions and decay and step parameters, the vision tower and
the MTP head. Kept at 16 bits and counted: the DeltaNet's two 32-row gate projections (in_proj_a, in_proj_b).

--program plan --plan FILE runs a plan from compiler.py, the compression compiler: a program for every matrix (any of the
above, or fp16 = kept), optionally the embedding rows in buckets by token frequency (each bucket its own program; a
randomized Hadamard over the hidden dimension only, so a lookup still decodes one row) and a width for every layer's
cached keys and values (simulated: forward hooks round each token's key and value vectors per head after a randomized
Hadamard, the first position kept exact). --gauges vo,ud searches exact rewrites before each block is quantized: the
V–O channel scale through grouped-query heads (vo) and the up–down channel scale of the MLP (ud), by the exponent of
the input magnitudes in --gauge-alphas, keeping the one with the least layer-output error on the calibration windows;
--seeds n tries n randomized Hadamard seeds per matrix (also exact) and keeps the least weighted error.

Any decoder (compiler_arch.py): the stages of every layer are discovered from the model (linears that read the same
tensor share input statistics, in execution order), so fused projections, other naming and hybrid layers need no table;
linears with fewer than 64 inputs or outputs stay at 16 bits and are counted; MoE experts (stacked 3-D weights) are
quantized expert by expert on the tokens routed to each, gate_up then down, with programs per usage bucket from the
plan. --data mix calibrates on the text mix (English, ten languages, code, chat) and adds a per-source evaluation;
--dtype bfloat16 holds both models in bf16 (large models; codes are still fitted in float64).

WikiText-2: calibration on 64 random training windows of 512 tokens, perplexity and KL to the original model on the first
64 test windows of 512, as in opgraph/data.py.

    HF_HUB_OFFLINE=1 python quantize.py --program lm4x --out results/qwen35/lm4x.json
"""

from __future__ import annotations

import argparse
import copy
import re
import glob
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from opgraph import codes as C
from opgraph import methods as ops
from opgraph import nested as N
from opgraph.elements import pinned_orthogonal
from opgraph.verbs import SET, block_hadamard
import compiler_arch as A

STAGES = {"linear_attn": [["in_proj_qkv", "in_proj_z"], ["out_proj"]],
          "self_attn": [["q_proj", "k_proj", "v_proj"], ["o_proj"]],
          "mlp": [["gate_proj", "up_proj"], ["down_proj"]]}
KEEP = ("in_proj_a", "in_proj_b")  # the DeltaNet's 32-row gate projections: kept at 16 bits, counted
LAM = 0.1  # the cross term's ridge penalty, a fraction of the mean input energy (as run_modern.py)
# a program is a family and a bit width: rtn<b>, gptq<b>, gptaq<b> (cross-term), lm<b>, lm<b>x (cross-term), tr<b>, tr<b>x;
# fp16 keeps the matrix as stored
PROGRAM = re.compile(r"orig|(rtn|gptq|gptaq|lm|tr|higgs)(\d)(x?)")
DATA_FREE = ("rtn", "higgs", "fp")  # programs that need no input statistics
GAUGE = {"self_attn": "vo", "mlp": "ud"}  # the exact rewrite searched before each block (--gauges)
EMBED_T = 256  # embedding rows per trellis run (every bucket is a multiple of 256 rows)
DEV = os.environ.get("OPGRAPH_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps")  # cpu for local tests (MPS has no float64)
CACHE = Path(__file__).parent / "cache"
BLOCK_SEED = {"linear_attn": 1, "self_attn": 2, "mlp": 3}  # fixed per-matrix seeds for the Hadamard signs


class Stop(Exception):
    pass


class RowColScale(torch.nn.Module):
    """W ↦ diag(sv) W diag(su) around a frozen quantized matrix: the fine-tuned row and column scales."""

    def __init__(self, m, n):
        super().__init__()
        self.su = torch.nn.Parameter(torch.ones(n, device=DEV))
        self.sv = torch.nn.Parameter(torch.ones(m, device=DEV))

    def forward(self, W):
        return self.sv[:, None] * W * self.su[None, :]


def finetune(L, ak, xs, ys, scaled, args):
    """Per-layer fine-tuning (QTIP's recipe): Adam on the layer's continuous parameters, mean squared error to the
    reference layer's outputs, one batch per calibration chunk, the best of --ft-epochs on the held-out chunks kept.
    Returns (initial, best) held-out loss."""
    from torch.nn.utils import parametrize as P
    n_valid = args.ft_valid // args.chunk
    tr, va = list(range(len(xs) - n_valid)), list(range(len(xs) - n_valid, len(xs)))
    scales = [q for m in scaled for q in m.parametrizations.weight[0].parameters()]
    ids = {id(q) for q in scales}
    frozen = {id(m.parametrizations.weight.original) for m in scaled}
    others = [q for q in L.parameters() if id(q) not in ids and id(q) not in frozen]
    for q in scales + others:
        q.requires_grad_(True)
    opt = torch.optim.Adam([{"params": others, "lr": args.ft_lr}, {"params": scales, "lr": args.ft_lr_scale}])

    def held_out():
        with torch.no_grad():
            return sum(F.mse_loss(run_layer(L, xs[c], ak[c]), ys[c]).item() for c in va) / len(va)

    first = best = held_out()
    state = {k: v.detach().clone() for k, v in L.state_dict().items()}
    g = torch.Generator().manual_seed(0)
    for _ in range(args.ft_epochs):
        for c in [tr[k] for k in torch.randperm(len(tr), generator=g).tolist()]:
            loss = F.mse_loss(run_layer(L, xs[c], ak[c]), ys[c])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = held_out()
        if v < best:
            best, state = v, {k: v_.detach().clone() for k, v_ in L.state_dict().items()}
    L.load_state_dict(state)
    for q in L.parameters():
        q.requires_grad_(False)
    return first, best


def log(msg):
    print(msg, flush=True)


def token_ids(tok, split, tag="qwen35"):
    path = CACHE / f"{tag}-wikitext2-{split}.pt"
    if path.exists():
        return torch.load(path)
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    CACHE.mkdir(exist_ok=True)
    torch.save(ids, path)
    return ids


def fineweb_ids(tok, windows=4096):
    """FineWeb-Edu as one token stream in tok's vocabulary: the first 4096 windows of the corpus shard on the volume
    (Qwen3-tokenized, 256 tokens each, documents ended by EOS; about a million tokens), decoded document by document and
    re-tokenized, joined as WikiText is."""
    from transformers import AutoTokenizer
    hf = Path(os.path.expanduser("~/.cache/huggingface"))
    flat = torch.load(hf / "opgraph_corpus" / "fineweb_edu_s0.pt", mmap=True)[:windows].reshape(-1).tolist()
    qtok = AutoTokenizer.from_pretrained(sorted(glob.glob(str(hf / "hub/models--Qwen--Qwen3-0.6B-Base/snapshots/*/")))[-1])
    docs, cur = [], []
    for t in flat:
        if t == qtok.eos_token_id:
            docs.append(cur)
            cur = []
        else:
            cur.append(t)
    text = "\n\n".join(qtok.decode(d) for d in docs + [cur] if d)
    return tok(text, return_tensors="pt").input_ids[0]


def parse(program):
    """(family, bits, cross-term target) of a program name."""
    if program == "fp16":
        return "fp", 16, False
    m = PROGRAM.fullmatch(program)
    if not m or (m.group(1) in ("gptaq",) and m.group(3)) or (m.group(1) in ("rtn", "gptq", "higgs") and m.group(3)):
        raise ValueError(f"unknown program {program}")
    return m.group(1), int(m.group(2)), m.group(1) == "gptaq" or m.group(3) == "x"


class Dense:
    """A quantized matrix kept as its dense reconstruction, with what its format stores (the Net interface word() uses)."""

    def __init__(self, W, bits):
        self.W, self.bits = W, bits

    def matrix(self, dtype=torch.float64):
        return self.W.to(dtype)

    def rate_bits(self):
        return self.bits


@torch.no_grad()
def higgs_word(W, b, p=2):
    """HIGGS as transformers.integrations.higgs.quantize_with_higgs, dequantized: per group of g inputs, the Hadamard
    transform of the weights over their norm (unit RMS), each pair rounded to the nearest grid point, then back."""
    from transformers.integrations.higgs import get_higgs_grid
    rows, n = W.shape
    g = next((g for g in (1024, 512, 256) if n % g == 0), None)
    if g is None:  # no power of two of at least 256 divides n (SmolLM2's 960): the largest divisor with a Hadamard
        g = next(g for g in range(min(n, 1024), 1, -1) if n % g == 0 and _has_hadamard(g))
    grid = get_higgs_grid(p, 2 ** (p * b)).to(DEV, torch.float32)
    H = C.hadamard(g).to(DEV, torch.float32)  # orthogonal; Sylvester's order for a power of two, as fast_hadamard_transform
    Wg = W.to(DEV, torch.float32).reshape(rows, n // g, g)
    s = torch.linalg.norm(Wg, dim=-1, keepdim=True).clamp_min(1e-12) / math.sqrt(g)  # each group's RMS
    T = (Wg @ H) / s
    out = torch.empty_like(T)
    for r in range(0, rows, 256):
        v = T[r : r + 256].reshape(-1, p)
        idx = torch.argmax(2 * v @ grid.T - (grid * grid).sum(-1), dim=-1)
        out[r : r + 256] = grid[idx].reshape(out[r : r + 256].shape)
    What = (out * s) @ H.T
    return Dense(What.reshape(rows, n).cpu().double(), b * W.numel() + 16 * rows * (n // g))


def _has_hadamard(n):
    try:
        C.hadamard(n)
        return True
    except ValueError:
        return False


def word(program, W, H, seed, trellis, in_seed=None, out_rot=True):
    """One matrix's quantized program (an opgraph Net) for its target W and input statistics H."""
    fam, b, _ = parse(program)
    if fam == "fp":  # the checkpoint's bfloat16 values, exactly
        return Dense(W.to(torch.bfloat16).double(), 16 * W.numel())
    if fam == "rtn":
        return ops.rtn(W, b, 64)
    if fam == "higgs":
        return higgs_word(W, b)
    if fam in ("gptq", "gptaq"):
        return ops.gptq_lazy(W, H, b, 64)
    if fam == "lm":
        return N.nested_words([W], H, [seed], b, [b], 0.4, [1.0], N.lloyd_max(b), in_seed=in_seed, out_rot=out_rot)[b][0]
    return C.ldlq_words([W], H, trellis[b], [seed], in_seed=in_seed, out_rot=out_rot)[0]


def embedding_buckets(V, first=256):
    """Sizes of the embedding buckets, rows ranked by frequency: 256, 256, 512, 1024, … doubling, the last bucket the
    rest (49152 rows: nine buckets, the last two 16384). A vocabulary that is not a multiple of 256 leaves the last bucket
    short; embed_word pads it for the trellis."""
    sizes, s = [first], first
    while sum(sizes) + s < V:
        sizes.append(s)
        s *= 2
    return sizes + [V - sum(sizes)]


def frequency_order(ids, V):
    """Token ids by count in `ids`, most frequent first; ties by id (BPE's merge order, roughly its own frequency)."""
    counts = torch.bincount(ids.reshape(-1), minlength=V)[:V]
    return torch.sort(-counts, stable=True).indices


@torch.no_grad()
def head_inputs(model, chunks):
    """E[h hᵀ] of the output head's input (the final normed hidden state) on the calibration chunks."""
    acc = []

    def f(m, a):
        x = a[0].reshape(-1, a[0].shape[-1]).float()
        acc.append((x.T @ x, len(x)))
        raise Stop()
    hk = model.get_output_embeddings().register_forward_pre_hook(f)
    for x in chunks:
        try:
            model(x, use_cache=False)
        except Stop:
            pass
    hk.remove()
    return (sum(a for a, _ in acc) / sum(n for _, n in acc)).cpu().double()


def embedding_environment(H_out, reader="both"):
    """The tied table serves two readers: the output head (error weighted by its inputs' second moment) and the lookup
    (plain squared error); both at equal weight, in units of the head's mean input energy. An untied input embedding has
    only the lookup (the identity), an untied head only the head (with 1% of the identity as damping)."""
    I = torch.eye(len(H_out), dtype=H_out.dtype)
    if reader == "lookup":
        return I
    Hn = H_out / torch.diag(H_out).mean()
    return Hn + (I if reader == "both" else 0.01 * I)


def embed_word(W, H, program, seed, block=None, rec=None):
    """One bucket of embedding rows: a randomized Hadamard over the hidden dimension only (each row stays its own, so a
    lookup decodes one row), a 16-bit RMS scale per row, then the program's code under LDLQ with H: tr<b> a trellis
    over runs of 256 rows, lm<b> Lloyd–Max levels."""
    fam, b, _ = parse(program)
    if fam == "fp":
        if rec is not None:
            rec.append({"dense": W.to(torch.bfloat16)})
        return Dense(W.to(torch.bfloat16).double(), 16 * W.numel())
    if block:  # a table read in slices (Gemma 4's per-layer inputs: 256 per layer): each slice its own rotation and code
        parts = [embed_word(W[:, j : j + block], H, program, seed + 101 * (j // block), rec=rec) for j in range(0, W.shape[1], block)]
        return Dense(torch.cat([q.W for q in parts], 1), sum(q.bits for q in parts))
    lookup = H is None or bool(torch.equal(H, torch.eye(len(H), dtype=H.dtype)))  # a lookup table: plain rounding, no sweep
    R = C.rht(W.shape[1], seed)
    code = trellis_code(b, EMBED_T) if fam == "tr" else C.ScalarCode(SET(N.lloyd_max(b).tolist()))
    rows = W.shape[0]
    pad = (-rows) % EMBED_T if fam == "tr" and TRELLIS_ROWS is None else 0  # the trellis over rows needs whole runs of them: zero rows, dropped after
    Wp = torch.cat([W.double(), torch.zeros(pad, W.shape[1], dtype=torch.float64)]) if pad else W.double()
    Wr = (Wp.float().to(DEV) @ R.float().T.to(DEV)).double().cpu()  # the rotation on the GPU (a wide table is a large product)
    net = C.ldlq_words([Wr], None if lookup else R @ H.double() @ R.T, code, [seed], incoherent=False)[0]
    Q = net.matrix(dtype=torch.float64).reshape(Wp.shape)[:rows]  # the trellis splits rows into runs
    if rec is not None and fam in ("lm", "tr"):  # what the runtime stores: codes, row scales and the rotation's seed
        import lamq_export
        codes = lamq_export.codes_of(net.cores["W"].data.double(), N.lloyd_max(b).double()) if fam == "lm" else lamq_export.trellis_symbols(net.cores["W"], b)
        rec.append({"codes": codes, "scales": net.cores["W.S"].data.double(), "seed": seed, "width": W.shape[1], "bits": b, "values": None if fam == "lm" else code.values})
    return Dense((Q.float().to(DEV) @ R.float().to(DEV)).double().cpu(), net.rate_bits() * rows / (rows + pad))


@torch.no_grad()
def quantize_embedding(E, spec, H, order, seed0=7000, block=None, name=None):
    """One table (the tied embedding / output, or either alone), bucket by bucket as the plan says; (bits, weights)."""
    bits, lo = 0.0, 0
    for k, (size, p) in enumerate(zip(spec["sizes"], spec["programs"])):
        rows = order[lo : lo + size].to(E.device)
        rec = [] if EXPORT is not None else None
        net = embed_word(E.data[rows].cpu().double(), H, p, seed0 + k, block, rec=rec)
        E.data[rows] = net.matrix(dtype=torch.float64).to(E.dtype).to(E.device)
        if EXPORT is not None:
            EXPORT.table_bucket(name, k, rows, p, rec, tuple(E.shape), block)
        bits += net.rate_bits()
        lo += size
    return bits, E.numel()


EXPORT = None  # a lamq_export.Exporter when --export is given
TRELLIS_ROWS = None  # --trellis-rows: (width, look) for trellis paths along each row, which the Apple Silicon runtime reads, instead of along each column


def trellis_code(k, T):
    return C.TrellisCode(12, k, T=T) if TRELLIS_ROWS is None else C.RowTrellisCode(12, k, T=256, width=TRELLIS_ROWS[0], look=TRELLIS_ROWS[1])


class KVQuant:
    """A quantized key/value cache, simulated: forward hooks on every attention layer's k_proj and v_proj (the keys before
    RoPE, which a cache can store and rotate on read) round each token's vector per key/value head — a randomized
    Hadamard over the head dimension, a 16-bit RMS scale, the Lloyd–Max levels of a unit Gaussian — and leave the first
    position (the attention sink) exact. `widths` maps (layer, "k" | "v") to bits; absent or 16 = exact."""

    def __init__(self, model, widths=None, seed=0):
        cfg = model.config
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.widths, self.R, self.levels, self.hooks = dict(widths or {}), {}, {}, []
        self.layers = [l for l, L in enumerate(model.model.layers) if hasattr(L, "self_attn")]
        for l in self.layers:
            for kind in ("k", "v"):
                self.R[l, kind] = C.rht(self.hd, seed + 2 * l + (kind == "v")).float().to(DEV)
                mod = getattr(model.model.layers[l].self_attn, f"{kind}_proj")
                self.hooks.append(mod.register_forward_hook(lambda m, a, out, key=(l, kind): self._round(key, out)))

    def _round(self, key, out):
        b = self.widths.get(key)
        if not b or b >= 16:
            return None
        if b not in self.levels:
            lv = N.lloyd_max(b).float().to(DEV)
            self.levels[b] = (lv, (lv[1:] + lv[:-1]) / 2)
        lv, mid = self.levels[b]
        R = self.R[key]
        x = out.float().unflatten(-1, (-1, self.hd)) @ R.T
        s = x.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-12)
        q = ((lv[torch.bucketize(x / s, mid)] * s) @ R).flatten(-2).to(out.dtype)
        q[:, 0] = out[:, 0]
        return q

    def bits_per_token(self, b):
        """Cache bits per token for one layer's keys (or values) at width b: codes and a 16-bit scale per head."""
        return self.n_kv * (self.hd * 16 if b >= 16 else self.hd * b + 16)

    def bits_per_value(self):
        keys = [(l, kind) for l in self.layers for kind in ("k", "v")]
        return sum(self.bits_per_token(self.widths.get(k) or 16) for k in keys) / (len(keys) * self.n_kv * self.hd)

    def remove(self):
        for h_ in self.hooks:
            h_.remove()


def expert_matrices(mod, e, hidden):
    """Expert e's gate_up and down as (out, in) matrices, and whether the module stores them (in, out), as gpt-oss does:
    gate_up's last dimension is then not the hidden size."""
    gu, dn = mod.gate_up_proj[e], mod.down_proj[e]
    t = mod.gate_up_proj.shape[-1] != hidden
    return (gu.T, dn.T, True) if t else (gu, dn, False)


def expert_hidden(mod, e, x, transposed):
    """The input to expert e's down projection for its tokens x: gate_up in its stored orientation, its bias if any, then
    the module's own gate (transformers 5 experts all define _apply_gate; gpt-oss's is a clamped GLU on interleaved
    halves)."""
    W = mod.gate_up_proj[e]
    g = x @ W if transposed else torch.nn.functional.linear(x, W)
    b = getattr(mod, "gate_up_proj_bias", None)
    return mod._apply_gate(g if b is None else g + b[e])


def routed(mod, L, h, ak):
    """(hidden states, top-k expert indices) that the experts module of layer L receives for input h."""
    grab = {}

    def f(m, a, k):
        args = list(a) + [k.get(n) for n in ("top_k_index", "router_indices", "top_k_weights", "routing_weights") if n in k]
        grab["x"], grab["idx"] = args[0], args[1]
        raise Stop()
    hk = mod.register_forward_pre_hook(f, with_kwargs=True)
    try:
        run_layer(L, h, ak)
    except Stop:
        pass
    finally:
        hk.remove()
    return grab["x"].reshape(-1, grab["x"].shape[-1]), grab["idx"].reshape(-1, grab["idx"].shape[-1])


@torch.no_grad()
def gauge_activations(L, blk, hq, ak):
    """Mean |input| per channel of o_proj (vo) or down_proj (ud) on the stream into this layer."""
    mod = L.self_attn.o_proj if blk == "self_attn" else L.mlp.down_proj
    acc = []

    def f(m, a):
        x = a[0].reshape(-1, a[0].shape[-1]).float()
        acc.append((x.abs().sum(0), len(x)))
        raise Stop()
    hk = mod.register_forward_pre_hook(f)
    for h, a_ in zip(hq, ak):
        try:
            run_layer(L, h, a_)
        except Stop:
            pass
    hk.remove()
    return sum(a for a, _ in acc) / sum(n for _, n in acc)


@torch.no_grad()
def apply_gauge(L, blk, alpha, act, cfg):
    """Exact rewrites of the full-precision layer, before its block is quantized. vo: v's output channel r scaled by d_r
    and the matching input channel of o in every query head that shares key/value head r by 1/d_r. ud: up's row j by
    d_j and down's column j by 1/d_j (the MLP is silu(gate)·up, linear in up). d = (mean |input| of o or down)^(−α),
    at geometric mean 1; α = 0 is the identity."""
    if not alpha:
        return
    if blk == "self_attn":
        at = L.self_attn
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        g = cfg.num_attention_heads // cfg.num_key_value_heads
        assert at.v_proj.bias is None and at.o_proj.in_features == cfg.num_attention_heads * hd
        r_of_c = torch.tensor([((c // hd) // g) * hd + c % hd for c in range(at.o_proj.in_features)], device=act.device)
        act_r = torch.zeros(at.v_proj.out_features, device=act.device).index_add_(0, r_of_c, act) / g
        d = act_r.clamp(min=1e-8) ** -alpha
        d = d / d.log().mean().exp()
        at.v_proj.weight.data *= d[:, None].to(at.v_proj.weight.dtype)
        at.o_proj.weight.data /= d[r_of_c][None, :].to(at.o_proj.weight.dtype)
    else:
        mlp = L.mlp
        assert mlp.up_proj.bias is None
        d = act.clamp(min=1e-8) ** -alpha
        d = d / d.log().mean().exp()
        mlp.up_proj.weight.data *= d[:, None].to(mlp.up_proj.weight.dtype)
        mlp.down_proj.weight.data /= d[None, :].to(mlp.down_proj.weight.dtype)


@torch.no_grad()
def layer_error(L, hq, ys, ak):
    """The layer's output error against the reference layer's (on the reference stream), relative, without the first
    position (whose massive activations would otherwise decide it)."""
    num = den = 0.0
    for h, y, a_ in zip(hq, ys, ak):
        o = run_layer(L, h, a_)
        num += (o[:, 1:] - y[:, 1:]).float().square().sum().item()
        den += y[:, 1:].float().square().sum().item()
    return num / den


@torch.no_grad()
def massive_channels(model, x, k=4, first_layer=4):
    """The residual channels where the first token's activation is largest relative to the other tokens, averaged over
    layers from `first_layer` on (as elements.massive_channels)."""
    outs = []
    hooks = [L.register_forward_hook(lambda m, a, o, i=i: outs.append((i, o[0] if isinstance(o, tuple) else o)))
             for i, L in enumerate(model.model.layers)]
    model(x, use_cache=False)
    for h_ in hooks:
        h_.remove()
    score = sum(h[:, 0].abs().mean(0) / (h[:, 1:].abs().mean((0, 1)) + 1e-6) for i, h in outs if i + 1 >= first_layer)
    return sorted(torch.topk(score, k).indices.tolist())


@torch.no_grad()
def rotate_model(model, Q):
    """Exact: fold the gains, rotate the residual stream by Q, rotate the writers' inputs at runtime."""
    Q = Q.double()
    m, dev = model.model, model.model.embed_tokens.weight.device
    put = lambda mod, W: setattr(mod.weight, "data", W.float().to(dev))
    E = m.embed_tokens.weight.data.cpu().double()
    head = (E * (1 + m.norm.weight.data.cpu().double())[None, :]) @ Q
    model.lm_head.weight = torch.nn.Parameter(head.float().to(dev), requires_grad=False)  # untied
    model.config.tie_word_embeddings = False
    put(m.embed_tokens, E @ Q)
    m.norm.weight.data.zero_()
    for i, L in enumerate(m.layers):
        g1 = 1 + L.input_layernorm.weight.data.cpu().double()
        g2 = 1 + L.post_attention_layernorm.weight.data.cpu().double()
        L.input_layernorm.weight.data.zero_()
        L.post_attention_layernorm.weight.data.zero_()
        lin = hasattr(L, "linear_attn")
        mix = L.linear_attn if lin else L.self_attn
        readers = [getattr(mix, n) for n in (("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b") if lin else ("q_proj", "k_proj", "v_proj"))]
        for mod, g in [(r, g1) for r in readers] + [(L.mlp.gate_proj, g2), (L.mlp.up_proj, g2)]:
            assert mod.bias is None
            put(mod, (mod.weight.data.cpu().double() * g[None, :]) @ Q)
        writer = mix.out_proj if lin else mix.o_proj
        for mod, R in ((writer, block_hadamard(writer.in_features, 128 if lin else 256, seed=1000 + i)),
                       (L.mlp.down_proj, block_hadamard(L.mlp.down_proj.in_features, 512, seed=i))):
            assert mod.bias is None
            put(mod, Q.T @ mod.weight.data.cpu().double() @ R.double())
            Rd = R.float().to(dev)
            mod.register_forward_pre_hook(lambda mm, a, Rd=Rd: (a[0] @ Rd.to(a[0].dtype),) + a[1:])  # x ↦ xR at runtime


def run_layer(L, h, ak):
    a, k = ak
    out = L(h, *a[1:], **k) if a else L(**{**k, "hidden_states": h})
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def evaluate(model, ref, ids, bs):
    nll = kl = 0.0
    n_tok = n_pos = 0
    for b in range(0, len(ids), bs):
        x = ids[b : b + bs].to(DEV)
        lp = F.log_softmax(model(x, use_cache=False).logits.float(), -1)
        nll += -lp[:, :-1].gather(-1, x[:, 1:, None]).sum().item()
        n_tok += x[:, 1:].numel()
        if ref is not None and ref is not model:
            lr = F.log_softmax(ref(x, use_cache=False).logits.float(), -1)
            kl += (lr.exp() * (lr - lp)).sum().item()
            n_pos += x.numel()
        del lp
    out = {"ppl": math.exp(nll / n_tok)}
    if n_pos:
        out["kl"] = kl / n_pos
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", required=True, help="orig, or a family and bit width: rtn4, gptaq5, lm6x, tr3x, ...")
    ap.add_argument("--model-path", default="", help="a local model directory (default: the Hugging Face cache snapshot)")
    ap.add_argument("--size", default="4B")
    ap.add_argument("--model", default="", help="another model's Hugging Face id, from the cache (NousResearch/Llama-2-7b-hf)")
    ap.add_argument("--ctx", default="512", help="evaluation window lengths, the first one reported as ppl and KL (4096,2048)")
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--calib-source", choices=["wikitext", "fineweb"], default="wikitext")
    ap.add_argument("--rotation", choices=["none", "quarot", "pinned"], default="none")
    ap.add_argument("--check-only", action="store_true", help="rotate, report the rotated model's perplexity, stop")
    ap.add_argument("--tasks", default="", help="benchmarks after the perplexity (eval_tasks.py): mmlu,arc_c,arc_e,hellaswag,obqa,csqa,gsm8k")
    ap.add_argument("--task-limit", type=int, default=0, help="questions per task (quick tests); 0 = the defaults")
    ap.add_argument("--task-limits", default="", help="questions per task, task=n comma list (overrides the defaults and --task-limit)")
    ap.add_argument("--hellaswag-n", type=int, default=2000)
    ap.add_argument("--task-bs", type=int, default=0, help="sequences per forward pass in the benchmarks (0: their defaults)")
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--eval", type=int, default=64, help="test windows per length; 0 = the whole test split")
    ap.add_argument("--chunk", type=int, default=8, help="sequences per calibration forward")
    ap.add_argument("--ft-epochs", type=int, default=0, help="per-layer fine-tuning epochs after each stage (0: none)")
    ap.add_argument("--ft-lr", type=float, default=3e-6, help="Adam step for the unquantized matrices and norms (QTIP's)")
    ap.add_argument("--ft-lr-scale", type=float, default=3e-4, help="Adam step for the row and column scales (start at 1)")
    ap.add_argument("--ft-valid", type=int, default=8, help="held-out calibration windows (a multiple of --chunk)")
    ap.add_argument("--ft-input", choices=["quant", "ref"], default="quant")
    ap.add_argument("--layers", type=int, default=0, help="quantize only the first n layers (profiling); 0 = all")
    ap.add_argument("--dense", default="", help="with --program dense: a file of quantized matrices made elsewhere")
    ap.add_argument("--plan", default="", help="with --program plan: a plan from compiler.py")
    ap.add_argument("--gauges", default="", help="exact rewrites searched before each block: vo, ud, or vo,ud")
    ap.add_argument("--gauge-alphas", default="0,0.25,0.5", help="the exponents tried (0 = no rewrite)")
    ap.add_argument("--seeds", type=int, default=1, help="randomized Hadamard seeds tried per matrix (tr, lm)")
    ap.add_argument("--data", choices=["wikitext", "mix"], default="wikitext", help="calibration text (and token frequencies)")
    ap.add_argument("--eval-mix", type=int, default=0, help="held-out windows per text-mix source to evaluate (0: 8 with --data mix)")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--trellis-rows", default="", help="width,look: run trellis paths along each row, settling width columns at a time and looking look columns ahead")
    ap.add_argument("--rot", choices=["both", "in-shared"], default="both",
                    help="Lloyd–Max and trellis words: rotate both sides of each matrix, or only the input, with one rotation shared by a stage's matrices")
    ap.add_argument("--export", default="", help="write the plan in the Apple Silicon runtime's format")
    ap.add_argument("--save", default="", help="keep the quantized text weights (bf16) here, for later evaluation (public_eval.py dense:)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    global EXPORT, TRELLIS_ROWS
    TRELLIS_ROWS = tuple(int(x) for x in args.trellis_rows.split(",")) if args.trellis_rows else None
    if args.export:
        import lamq_export
        assert args.seeds <= 1, "--export records each matrix's seed; --seeds picks among several"
        EXPORT = lamq_export.Exporter()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.program not in ("orig", "dense", "plan"):
        parse(args.program)
    repo = args.model or f"Qwen/Qwen3.5-{args.size}"
    path = args.model_path or glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*/"))[0]
    tok = AutoTokenizer.from_pretrained(path)
    dtype = getattr(torch, args.dtype)
    qcfg = {}
    if json.loads((Path(path) / "config.json").read_text()).get("quantization_config", {}).get("quant_method") == "mxfp4":
        from transformers import Mxfp4Config  # gpt-oss ships MXFP4 experts: the reference is their dequantized values
        qcfg = {"quantization_config": Mxfp4Config(dequantize=True)}
    ref = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **qcfg).to(DEV).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    qwen35 = "qwen3_5" in ref.config.model_type
    name = repo.split("/")[-1]
    tag = "qwen35" if qwen35 else name.lower()  # the token caches: one per tokenizer
    assert qwen35 or args.rotation == "none", "the rotation is written for Qwen3.5's norms"
    ids = token_ids(tok, "test", tag)
    tests = {}
    for c in map(int, args.ctx.split(",")):
        n = args.eval or len(ids) // c
        tests[c] = A.with_bos(ids[: n * c].view(n, c), tok, ref)
    test = next(iter(tests.values()))
    res = {"program": args.program, "model": name, "args": vars(args)}

    n_mix = args.eval_mix or (8 if args.data == "mix" else 0)
    mix = A.TextMix(tok, tag) if (args.data == "mix" or n_mix) else None
    mix_eval = {s_: A.with_bos(w_, tok, ref) for s_, w_ in mix.windows("eval", n_mix, 512, seed=2).items()} if n_mix else {}

    def evaluate_all(m, r):
        """Perplexity (and KL to r) at each window length; the first is the headline. With the text mix, also per source
        (and the mean KL over sources, every source weighted equally)."""
        out = {}
        for c, t in tests.items():
            out[c] = evaluate(m, r, t, max(1, 2048 // c))
            log(f"    ctx {c} ({len(t)} windows): ppl {out[c]['ppl']:.4f}" + (f", KL {out[c]['kl']:.4f}" if "kl" in out[c] else ""))
        extra = {}
        if mix_eval:
            extra["mix"] = {src: evaluate(m, r, t, 4) for src, t in mix_eval.items()}
            if r is not None:
                extra["mix kl"] = sum(v["kl"] for v in extra["mix"].values()) / len(extra["mix"])
            log("    mix: " + ", ".join(f"{k} {v['ppl']:.2f}" + (f"/{v['kl']:.3f}" if "kl" in v else "") for k, v in extra["mix"].items()))
        return {**out[next(iter(tests))], "by ctx": out, **extra}
    t_all = time.time()
    tasks = [t for t in args.tasks.split(",") if t]
    limits = {t: args.task_limit for t in tasks} if args.task_limit else {"hellaswag": args.hellaswag_n, "mmlu": 3000, "gsm8k": 250}
    if args.task_limits:
        limits |= {k: int(v) for k, v in (x.split("=") for x in args.task_limits.split(","))}

    def benchmarks(m):
        if tasks:
            from eval_tasks import run_tasks
            res["tasks"] = run_tasks(m, tok, tasks, DEV, limits, log, bs=args.task_bs or None, bos=A.needs_bos(m))

    if args.program == "orig":
        res |= evaluate_all(ref, None)
        log(f"  original: ppl {res['ppl']:.4f}")
        benchmarks(ref)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
        return
    if args.program == "dense":  # another method's matrices (their effective dense weights), evaluated as ours are
        d = torch.load(args.dense)
        model = copy.deepcopy(ref)
        for key, W in d["linears"].items():
            i, sub = key.split(".", 1)
            A.decoder_layers(model)[1][int(i)].get_submodule(sub).weight.data = W.to(dtype).to(DEV)
        res |= {"method": d["method"], "bits per weight (quantized linears)": d["bits"] / d["weights"],
                "quantized weights": d["weights"], "their forward ppl": d.get("their forward ppl")}
        res |= evaluate_all(model, ref)
        log(f"  {d['method']}: ppl {res['ppl']:.3f}, KL {res['kl']:.4f}, {res['bits per weight (quantized linears)']:.3f} bits/weight")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
        return

    T = args.calib_len
    if args.data == "mix":
        calib = mix.calibration(args.calib, T)
    else:
        train = token_ids(tok, "train", tag) if args.calib_source == "wikitext" else fineweb_ids(tok)
        starts = torch.randint(0, len(train) - T - 1, (args.calib,), generator=torch.Generator().manual_seed(0))
        calib = torch.stack([train[s : s + T] for s in starts])
    calib = A.with_bos(calib, tok, ref)
    chunks = [calib[b : b + args.chunk].to(DEV) for b in range(0, len(calib), args.chunk)]
    model = copy.deepcopy(ref)
    MODULE_NAME = {id(m_): n for n, m_ in model.named_modules()}  # for --export
    if args.rotation != "none":
        from opgraph.codes import rht
        hidden = ref.config.hidden_size
        if args.rotation == "pinned":
            pins = massive_channels(ref, chunks[0])
            res["pinned channels"] = pins
            Q = pinned_orthogonal(hidden, pins, seed=0)
        else:
            Q = rht(hidden, 0)
        for mm in (ref, model):
            rotate_model(mm, Q)
        res["rotated model ppl (exactness check)"] = evaluate(ref, None, test, 4)["ppl"]
        log(f"  rotation {args.rotation}: {res.get('pinned channels', '')} rotated model ppl {res['rotated model ppl (exactness check)']:.4f}")
        if args.check_only:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(res, indent=1))
            return
    tc = A.text_config(ref.config)
    T_tr = 256 if tc.hidden_size % 256 == 0 else 320  # outputs per trellis run: Qwen3.5 and Llama-2 256, SmolLM2 320
    plan = json.loads(Path(args.plan).read_text()) if args.program == "plan" else None
    _, layers = A.decoder_layers(model)
    _, rlayers = A.decoder_layers(ref)
    n_layers = args.layers or len(layers)
    stages_of = A.discover(ref, chunks[0][:1, :64])
    res["stages"] = A.stage_list_text(stages_of[0]) + (f" | last layer: {A.stage_list_text(stages_of[-1])}" if len(stages_of) > 1 else "")
    log(f"  stages: {res['stages']}")

    def prog(i, name):
        return plan["linears"][f"{i}.{name}"] if plan else args.program
    progs = {prog(i, n) for i in range(n_layers) for st in stages_of[i] if st["kind"] == "linear" for n in st["names"]}
    if plan:
        progs |= {q for e in (plan.get("experts") or {}).values() for q in e["gate_up"] + e["down"]}
    elif any(st["kind"] == "experts" for i in range(n_layers) for st in stages_of[i]):
        progs.add(args.program)
    codes = {}

    def trellis_for(rows):
        """The trellis codes for a matrix with `rows` outputs: runs of T_tr, or the largest of 320, 256, 128, 64 that tiles them."""
        T = T_tr if rows % T_tr == 0 else next(t for t in (320, 256, 128, 64) if rows % t == 0)
        return {k: codes.setdefault((k, T), trellis_code(k, T)) for k in sorted({parse(q)[1] for q in progs if parse(q)[0] == "tr"})}
    cross = any(parse(q)[2] for q in progs)
    gauges = [g for g in args.gauges.split(",") if g]
    alphas = [float(x) for x in args.gauge_alphas.split(",")]
    assert not (gauges and args.ft_epochs), "the gauge search and per-layer fine-tuning are not combined"
    keep_ref = cross or args.ft_epochs > 0 or bool(gauges)  # the reference stream: cross term, fine-tuning targets, gauge errors

    bits_e = n_e = 0
    emb, head, tied = A.head_and_embedding(model)
    if plan and (plan.get("embed") or plan.get("head") or plan.get("ple")):  # first, so every layer is quantized against the stream it will see
        t0 = time.time()
        V = emb.weight.shape[0]
        spec = plan.get("embed") or plan.get("head") or plan.get("ple")
        freq = (mix or A.TextMix(tok, tag)).frequency(V) if "mix" in spec.get("order", "") else torch.bincount(token_ids(tok, "train", tag).reshape(-1), minlength=V)[:V].double()
        order = torch.sort(-freq, stable=True).indices
        H_out = head_inputs(ref, chunks)
        if plan.get("embed"):
            pname = {id(p_): n for n, p_ in model.named_parameters()}
            b_, n_ = quantize_embedding(emb.weight, plan["embed"], embedding_environment(H_out, "both" if tied else "lookup"), order, name=pname[id(emb.weight)])
            bits_e, n_e = bits_e + b_, n_e + n_
        if plan.get("head") and not tied:
            b_, n_ = quantize_embedding(head.weight, plan["head"], embedding_environment(H_out, "head"), order, seed0=8000,
                                        name={id(p_): n for n, p_ in model.named_parameters()}[id(head.weight)])
            bits_e, n_e = bits_e + b_, n_e + n_
        if plan.get("ple"):  # Gemma 4's per-layer input table: lookup only
            ple = A.per_layer_embedding(model)
            b_, n_ = quantize_embedding(ple.weight, plan["ple"], None, order, seed0=9000, block=A.text_config(ref.config).hidden_size_per_layer_input,
                                        name={id(p_): n for n, p_ in model.named_parameters()}[id(ple.weight)])
            bits_e, n_e = bits_e + b_, n_e + n_
        log(f"  embedding{'' if tied else ' and head'}{' and per-layer table' if plan.get('ple') else ''}: {bits_e / n_e:.3f} bits/weight ({time.time() - t0:.0f}s)")

    # each layer's call arguments for one calibration chunk (masks, rotary tables; the same for every chunk), and every
    # chunk's input to layer 0
    def layer_args(mm):
        """Every layer's call arguments for every calibration chunk, without the hidden state (fed by the harness): masks
        and rotary tables, and token-dependent inputs (Gemma 4's per-layer inputs, and its shared key/value dictionary,
        one per chunk, which a source layer writes and later layers read)."""
        out = {}
        lay = A.decoder_layers(mm)[1]

        def grab(a, k, i):
            out.setdefault(i, []).append(((None,) + tuple(a[1:]) if a else (), {n: v for n, v in k.items() if n != "hidden_states"}))
        hooks = [L.register_forward_pre_hook(lambda m, a, k, i=i: grab(a, k, i), with_kwargs=True) for i, L in enumerate(lay)]
        with torch.no_grad():
            for x in chunks:
                mm(x, use_cache=False)
        for h_ in hooks:
            h_.remove()
        return out
    kwr = layer_args(ref)
    kwq = layer_args(model)  # its own: per-layer inputs through its quantized tables, and its own shared-KV dictionaries

    def first_input(m, a, k):
        raise Stop(a[0] if a else k["hidden_states"])

    def layer0_inputs(mm):
        out = []
        hk = A.decoder_layers(mm)[1][0].register_forward_pre_hook(first_input, with_kwargs=True)
        for x in chunks:
            try:
                with torch.no_grad():
                    mm(x, use_cache=False)
            except Stop as s:
                out.append(s.args[0].clone())
        hk.remove()
        return out
    hq = layer0_inputs(model if bits_e else ref)  # through the quantized embedding when there is one
    hr = (layer0_inputs(ref) if bits_e else [h.clone() for h in hq]) if keep_ref else None

    def best_word(p, target, H, seed, in_seed=None, out_rot=True):
        """The program's word; with --seeds, the randomized Hadamard seed (an exact choice) with the least H-weighted error."""
        tr = trellis_for(target.shape[0])
        if args.seeds <= 1 or H is None or parse(p)[0] not in ("tr", "lm"):
            return word(p, target, H, seed, tr, in_seed, out_rot)
        best = None
        for t in range(args.seeds):
            net = word(p, target, H, seed + 7919 * t, tr)
            E = net.matrix(dtype=torch.float64) - target
            err = float((E @ H * E).sum())
            if best is None or err < best[0]:
                best = (err, net)
        return best[1]

    def cross_target(H, D):
        lam = LAM * torch.diag(H).mean() * torch.eye(len(H), dtype=H.dtype)
        return torch.eye(len(H), dtype=H.dtype) + torch.linalg.solve(H + lam, D.T).T

    def quantize_stage(i, L, Lr, stage, scaled):
        """One stage's matrices against the input statistics of the stream they will see; returns (bits, weights)."""
        names = stage["names"]
        mods = [L.get_submodule(n) for n in names]
        ps = [prog(i, n) for n in names]
        x_stage = any(parse(p)[2] for p in ps)
        H = M = None
        if any(parse(p)[0] not in DATA_FREE for p in ps):  # the stage's input statistics, from the stream it will see (and the reference)
            n_in = mods[0].in_features
            H = torch.zeros(n_in, n_in, device=DEV)
            D = torch.zeros(n_in, n_in, device=DEV) if x_stage else None
            n_tok = 0
            for c in range(len(chunks)):
                grab = {}

                def hook(name):
                    def f(m, a):
                        grab[name] = a[0].reshape(-1, a[0].shape[-1]).float()
                        raise Stop()
                    return f
                hh = mods[0].register_forward_pre_hook(hook("q"))
                try:
                    with torch.no_grad():
                        run_layer(L, hq[c], kwq[i][c])
                except Stop:
                    pass
                hh.remove()
                if x_stage:
                    hh = Lr.get_submodule(names[0]).register_forward_pre_hook(hook("r"))
                    try:
                        with torch.no_grad():
                            run_layer(Lr, hr[c], kwr[i][c])
                    except Stop:
                        pass
                    hh.remove()
                    D += grab["r"].T @ grab["q"]
                H += grab["q"].T @ grab["q"]
                n_tok += len(grab["q"])
            H = (H / n_tok).cpu().double()
            if x_stage:
                M = cross_target(H, (D / n_tok).cpu().double() - H)  # (X − X̃)ᵀ X̃: zero when nothing upstream changed
        bits_ = n_ = 0.0
        for j, (name, mod, p) in enumerate(zip(names, mods, ps)):
            W = mod.weight.data.cpu().double()
            target = W @ M if parse(p)[2] else W  # W* = W (I + (X − X̃)ᵀ X̃ (X̃ᵀ X̃ + λ)⁻¹)
            seed = 1000 * i + 100 * A.block_seed(name) + j
            in_seed = 1000 * i + 100 * A.block_seed(names[0]) + 1 if args.rot == "in-shared" else None  # one input rotation per stage
            net = best_word(p, target, H, seed, in_seed, args.rot != "in-shared")
            mod.weight.data = net.matrix(dtype=torch.float64).to(mod.weight.dtype).to(DEV)
            if EXPORT is not None:
                EXPORT.matrix(MODULE_NAME[id(mod)] + ".weight", *parse(p)[:2], net, seed, in_seed)
            bits_ += net.rate_bits()
            n_ += W.numel()
            if args.ft_epochs:
                from torch.nn.utils import parametrize
                parametrize.register_parametrization(mod, "weight", RowColScale(*mod.weight.shape))
                mod.parametrizations.weight.original.requires_grad_(False)
                scaled.append(mod)
                bits_ += 16 * sum(mod.weight.shape)
        for name in stage.get("keep", []):  # small linears: kept at 16 bits, counted
            bits_ += 16 * L.get_submodule(name).weight.numel()
            n_ += L.get_submodule(name).weight.numel()
        return bits_, n_

    def expert_programs(i, name, E):
        """Each expert's (gate_up, down) programs: the plan's usage buckets, or the uniform program for all."""
        if not plan:
            return {e: (args.program, args.program) for e in range(E)}
        spec = plan["experts"][f"{i}.{name}"]
        return {e: (spec["gate_up"][k], spec["down"][k]) for k, b in enumerate(spec["buckets"]) for e in b}

    @torch.no_grad()
    def quantize_experts(i, L, Lr, stage):
        """Every expert of an experts module, on the tokens routed to it (in the quantized stream): gate_up against those
        tokens (and the reference stream at the same positions, for the cross term), then down against the expert's own
        quantized intermediate. Experts no calibration token reaches keep the layer's mean input statistics."""
        mod, rmod = L.get_submodule(stage["name"]), Lr.get_submodule(stage["name"])
        E = mod.gate_up_proj.shape[0]
        xs, idx, xr = [], [], []
        for c in range(len(chunks)):
            x, k = routed(mod, L, hq[c], kwq[i][c])
            xs.append(x.float())
            idx.append(k)
            if keep_ref:
                xr.append(routed(rmod, Lr, hr[c], kwr[i][c])[0].float())
        X, K = torch.cat(xs), torch.cat(idx)
        Xr = torch.cat(xr) if keep_ref else None
        H_all = (X.T @ X / len(X)).cpu().double()
        ps = expert_programs(i, stage["name"], E)
        bits_ = n_ = 0.0
        for e in range(E):
            tok_e = (K == e).any(-1).nonzero().squeeze(-1)
            x = X[tok_e]
            gu, dn, t = expert_matrices(mod, e, X.shape[-1])
            for part, W_, inp, ref_inp in (("gate_up", gu, x, None if Xr is None else Xr[tok_e]), ("down", dn, None, None)):
                p = ps[e][0 if part == "gate_up" else 1]
                if part == "down":  # the expert's intermediate through its quantized gate_up (and the reference's)
                    inp = expert_hidden(mod, e, x.to(mod.gate_up_proj.dtype), t).float() if len(x) else x.new_zeros(0, W_.shape[1])
                    ref_inp = expert_hidden(rmod, e, Xr[tok_e].to(rmod.gate_up_proj.dtype), t).float() if Xr is not None and len(x) else None
                W = W_.detach().cpu().double()
                if len(inp) >= 8:
                    H = (inp.T @ inp / len(inp)).cpu().double()
                    H += 0.01 * torch.diag(H).mean() * torch.eye(len(H), dtype=H.dtype)
                else:
                    H = H_all if part == "gate_up" else torch.eye(W.shape[1], dtype=torch.float64)
                target = W
                if parse(p)[2] and ref_inp is not None and len(inp) >= 8:
                    D = (ref_inp.T @ inp / len(inp)).cpu().double() - (inp.T @ inp / len(inp)).cpu().double()
                    target = W @ cross_target(H, D)
                net = word(p, target, H if parse(p)[0] not in DATA_FREE else None, 1000 * i + 500 + 2 * e + (part == "down"),
                           trellis_for(W.shape[0]))
                Q = net.matrix(dtype=torch.float64)
                dst = mod.gate_up_proj if part == "gate_up" else mod.down_proj
                dst.data[e] = (Q.T if t else Q).to(dst.dtype).to(dst.device)
                bits_ += net.rate_bits()
                n_ += W.numel()
        return bits_, n_

    bits = n_w = 0.0
    timing, gauge_log = [], []
    for i in range(n_layers):
        t0 = time.time()
        L, Lr = layers[i], rlayers[i]
        blocks = {}
        for st in stages_of[i]:
            blocks.setdefault((st.get("names") or st.get("keep") or [st.get("name")])[0].split(".")[0], []).append(st)
        scaled, ft_log = [], []
        if args.ft_epochs or gauges:
            with torch.no_grad():
                ys = [run_layer(Lr, h, kwr[i][c]) for c, h in enumerate(hr)]
        for blk, sts in blocks.items():
            if GAUGE.get(blk) not in gauges or any(st["kind"] == "experts" for st in sts):
                for stage in sts:
                    b_, n_ = quantize_experts(i, L, Lr, stage) if stage["kind"] == "experts" else quantize_stage(i, L, Lr, stage, scaled)
                    bits += b_
                    n_w += n_
                    if args.ft_epochs:
                        ft_log.append(finetune(L, kwq[i] if args.ft_input == "quant" else kwr[i], hq if args.ft_input == "quant" else hr, ys, scaled, args))
                continue
            # the gauge search: each exponent applied to the full-precision block, the block quantized, the layer's
            # output error measured; the best kept
            snap = {k: v.clone() for k, v in L.state_dict().items()}
            act = gauge_activations(L, blk, hq, kwq[i])
            with torch.no_grad():
                y0 = run_layer(L, hq[0], kwq[i][0])
            tried = []
            for a in alphas:
                L.load_state_dict(snap)
                apply_gauge(L, blk, a, act, tc)
                with torch.no_grad():
                    exact = float((run_layer(L, hq[0], kwq[i][0]) - y0).abs().max() / y0.abs().max())
                b_ = n_ = 0.0
                for stage in sts:
                    sb, sn = quantize_stage(i, L, Lr, stage, scaled)
                    b_ += sb
                    n_ += sn
                tried.append((layer_error(L, hq, ys, kwq[i]), a, exact, {k: v.clone() for k, v in L.state_dict().items()}, b_, n_))
            err, a, _, state, b_, n_ = min(tried, key=lambda t: t[0])
            L.load_state_dict(state)
            bits += b_
            n_w += n_
            gauge_log.append({"layer": i, "block": blk, "alpha": a, "error by alpha": {str(t[1]): t[0] for t in tried},
                              "exactness by alpha (max relative output change before quantizing)": {str(t[1]): t[2] for t in tried}})
            del tried, snap
        if args.ft_epochs:  # bake the scales into the matrices
            from torch.nn.utils import parametrize
            for mod in scaled:
                parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)
                mod.weight.requires_grad_(False)
            res.setdefault("fine-tuning held-out loss (first, best) per stage", []).append([[round(a, 6), round(b, 6)] for a, b in ft_log])
        with torch.no_grad():  # the next layer's inputs, through the quantized layer (and the reference)
            hq = [run_layer(L, h, kwq[i][c]) for c, h in enumerate(hq)]
            if keep_ref:
                hr = [run_layer(Lr, h, kwr[i][c]) for c, h in enumerate(hr)]
        timing.append(round(time.time() - t0, 1))
        log(f"  layer {i}: {timing[-1]}s, running {bits / n_w:.3f} bits/weight"
            + (" | fine-tuning held-out loss " + ", ".join(f"{a:.3g}→{b:.3g}" for a, b in ft_log) if args.ft_epochs else "")
            + "".join(f" | {g['block']} α {g['alpha']}" for g in gauge_log if g["layer"] == i))
    res |= {"bits per weight (quantized linears)": bits / n_w, "quantized weights": n_w, "seconds per layer": timing,
            "quantize seconds": round(time.time() - t_all)}
    if bits_e:
        res |= {"embedding bits per weight": bits_e / n_e, "embedding weights": n_e,
                "bits per weight (linears and embedding)": (bits + bits_e) / (n_w + n_e)}
    lpath = A.decoder_layers(model)[0].rsplit(".", 1)[0]  # the text model: the module holding the decoder layers
    n_text = sum(p_.numel() for p_ in model.get_submodule(lpath).parameters()) + (0 if tied else head.weight.numel())
    b_text = bits + bits_e + 16.0 * (n_text - n_w - n_e)
    res["text weights"] = {"weights": n_text, "bits": b_text, "GB": b_text / 8e9, "bits per weight": b_text / n_text}
    log(f"  text weights: {n_text / 1e9:.3f}B weights, {b_text / 8e9:.3f} GB ({b_text / n_text:.3f} bits per weight)")
    if gauges:
        res["gauge search"] = gauge_log
    if plan:
        res["plan"] = {k: plan.get(k) for k in ("target", "target spec", "budget", "cost", "predicted kl", "summary")}
        if plan.get("kv"):
            kvq = A.KVCache(model, {(int(l), kind): b for l, kb in plan["kv"].items() for kind, b in kb.items()})
            res["kv bits per value"] = kvq.bits_per_value()
            log(f"  cache: {res['kv bits per value']:.3f} bits per key or value")
    if args.export:
        out = EXPORT.write(args.export, model, path)
        from safetensors.torch import save_file
        with torch.no_grad():  # reference outputs of the quantized model on one window, to check a runtime against
            ids = chunks[0][:1, :128]
            save_file({"ids": ids.cpu().to(torch.int32).contiguous(), "logits": model(ids).logits.float().cpu().to(torch.float16).contiguous()}, str(out / "reference.safetensors"))
        log(f"  exported {len(EXPORT.meta['matrices'])} matrices and {len(EXPORT.meta['tables'])} tables to {out}")
    if args.save:  # every text parameter the program changed, as bf16, with the sizes counted above
        rp = dict(ref.named_parameters())
        changed = {n: p_.detach().to(torch.bfloat16).cpu() for n, p_ in model.named_parameters() if n in rp and not torch.equal(p_, rp[n])}
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"params": changed, "text weights": res["text weights"], "program": args.program, "plan": args.plan}, args.save)
        log(f"  saved {len(changed)} quantized tensors to {args.save}")
    t0 = time.time()
    res |= evaluate_all(model, ref)
    res["eval seconds"] = round(time.time() - t0)
    if "mix kl" in res:
        log(f"  text mix: mean KL {res['mix kl']:.4f} over {len(res['mix'])} sources")
    log(f"  {args.program}: ppl {res['ppl']:.3f}, KL {res.get('kl', float('nan')):.4f}, {res['bits per weight (quantized linears)']:.3f} bits/weight"
        + (f", embedding {res['embedding bits per weight']:.3f}" if bits_e else "") + (f", cache {res['kv bits per value']:.3f}" if "kv bits per value" in res else ""))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    del ref
    (torch.cuda if DEV == "cuda" else torch.mps).empty_cache()
    benchmarks(model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
