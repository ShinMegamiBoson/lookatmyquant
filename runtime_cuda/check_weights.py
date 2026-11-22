"""Decodes matrices from a packed Inkling-2-Mini file and compares them with the original checkpoint's weights.

A right decoder gives the quantization error (a few percent to ~20% relative Frobenius error at 2-4 bits).
A wrong bit order, hash, rotation or row order gives ~100% or more. The `control` column decodes the same matrix
with one convention deliberately broken (input rotation signs dropped), and must come out near 141% (sqrt 2).

    python -m runtime_cuda.check_weights models/Inkling-2-Mini-LookAtMyQuant-8.9GB models/Inkling-2-Mini \
        --layers 0 7 20 23 --experts 0 1 77 127
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .inkling.formats import FMT_PARAMS, ID_TO_FMT, NVFP4_FID, decode_lm, decode_nvfp4, decode_trellis, effective_weight

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
TEACHER_LIN = {
    "self_attn.q_proj.linear": "attn.wq_du.weight",
    "self_attn.k_proj.linear": "attn.wk_dv.weight",
    "self_attn.v_proj.linear": "attn.wv_dv.weight",
    "self_attn.o_proj.linear": "attn.wo_ud.weight",
    "self_attn.o_proj.gate_proj": "attn.gate_attn.weight",
    "self_attn.relative_position_bias.linear": "attn.wr_du.weight",
}


def t(a: np.ndarray, dev) -> torch.Tensor:
    if a.dtype == np.uint32:
        a = a.view(np.int32)
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def teacher_experts(f, layer: int, side: str, experts: list[int], dev) -> torch.Tensor:
    """HF-layout dense experts `[len(experts), rows, cols]`, the same math as transformers' Inkling2LoopDequantizeExperts."""
    p = f"model.llm.layers.{layer}.mlp.experts.{'w13' if side == 'gu' else 'w2'}_weight"
    keys = set(f.keys())
    w = f.get_slice(p)
    idx = torch.tensor(experts)
    weight = torch.stack([w[e] for e in experts]).to(dev)
    scale = torch.stack([f.get_slice(p + "_scale")[e] for e in experts]).to(dev).float()
    if p + "_amax" in keys:
        def per_expert(key):  # stored per expert, or as one scalar for all
            v = f.get_tensor(key).float().reshape(-1)
            return (v[idx] if v.numel() > 1 else v.expand(len(experts))).to(dev)

        amax = per_expert(p + "_amax")
        den = per_expert(p + "_outer_scale_denominator")
        lut = torch.tensor(E2M1, device=dev)
        byte = torch.arange(256, device=dev)
        b2v = torch.stack((lut[byte & 0xF], lut[byte >> 4]), dim=-1)
        E, rows = weight.shape[:2]
        vals = b2v[weight.to(torch.uint8).long()].view(*scale.shape, 16) * scale[..., None]
        vals = vals * (amax / den).reshape(E, *([1] * (vals.dim() - 1)))
        out = vals.reshape(E, rows, -1)
    else:
        E, rows = weight.shape[:2]
        vals = weight.float().view(*scale.shape, 32) * torch.exp2(scale - 127)[..., None]
        out = vals.reshape(E, rows, -1)
    if side == "gu":  # transformers' Interleave(dim=1): stored [g0, u0, g1, u1, ...] -> [g; u]
        sh = out.shape
        out = out.reshape(sh[0], sh[1] // 2, 2, sh[2]).transpose(1, 2).reshape(sh)
    return out


def packed_linear_Q(meta, arr, luts, dev):
    s = meta["safe"]
    m, n, fmt = meta["m"], meta["n"], meta["fmt"]
    codes, scales = t(arr[f"lin_{s}_codes"], dev), t(arr[f"lin_{s}_scales"], dev)
    if fmt.startswith("lm"):
        return decode_lm(codes, scales, t(arr[f"lin_{s}_levels"], dev), m, n, meta["k"])
    return decode_trellis(codes.reshape(m, -1), scales, t(luts[fmt], dev), n, fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packed")
    ap.add_argument("teacher")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 7, 20, 23])
    ap.add_argument("--experts", type=int, nargs="+", default=[0, 1, 77, 127])
    ap.add_argument("--rows", type=int, default=4096, help="embedding/head rows to sample")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    P, Tdir = Path(args.packed), Path(args.teacher)
    luts = dict(np.load(P / "luts.npz"))
    tfiles = sorted(Tdir.glob("model-*.safetensors"))
    assert len(tfiles) == 1, tfiles
    f = safe_open(str(tfiles[0]), "pt")
    print(f"{'matrix':54s} {'fmt':>5s} {'shape':>12s} {'rel_err':>8s} {'control':>8s}")

    def report(name, fmt, W, Wt, W_ctrl):
        print(f"{name:54s} {fmt:>5s} {str(tuple(W.shape)):>12s} {rel(W, Wt):8.4f} {rel(W_ctrl, Wt):8.4f}", flush=True)

    for L in args.layers:
        meta = json.loads((P / f"layer_{L:02d}_meta.json").read_text())
        arr = np.load(P / f"layer_{L:02d}_packed.npz")
        for name, lm in meta["linears"].items():
            s = lm["safe"]
            Q = packed_linear_Q(lm, arr, luts, dev)
            si = t(arr[f"lin_{s}_signs_in"].astype(np.float32), dev)
            so = t(arr[f"lin_{s}_signs_out"].astype(np.float32), dev) if lm.get("out_rot") else None
            W = effective_weight(Q, si, so)
            W_ctrl = effective_weight(Q, torch.ones_like(si), so)
            Wt = f.get_tensor(f"model.llm.layers.{L}.{TEACHER_LIN[name]}").to(dev).float()
            report(f"L{L} {name}", lm["fmt"], W, Wt, W_ctrl)
        for side, n in (("gu", 2304), ("dn", 1152)):
            emap = arr[f"{side}_map"]
            si = t(arr[f"{side}_signs_in"].astype(np.float32), dev)
            Wt_all = teacher_experts(f, L, side, args.experts, dev)
            for j, e in enumerate(args.experts):
                fid, loc = int(emap[e, 0]), int(emap[e, 1])
                if fid == NVFP4_FID:  # no rotation: the decoded weight is the effective one
                    W = decode_nvfp4(*(t(arr[f"{side}_nvfp4_{p}"][loc], dev) for p in ("codes", "scales", "outer")))
                    report(f"L{L} expert {e} {side}", "nvfp4", W, Wt_all[j], torch.zeros_like(W))
                    continue
                fmt = ID_TO_FMT[fid]
                codes = t(arr[f"{side}_{fmt}_codes"][loc], dev)
                scales = t(arr[f"{side}_{fmt}_scales"][loc], dev)
                Q = decode_trellis(codes, scales, t(luts[fmt], dev), n, fmt)
                W = effective_weight(Q, si)
                report(f"L{L} expert {e} {side}", fmt, W, Wt_all[j], effective_weight(Q, torch.ones_like(si)))

    # Embedding and LM head: packed rows are in bucket order; row p holds token order[p].
    for kind, tkey in (("embed", "model.llm.embed.weight"), ("head", "model.llm.unembed.weight")):
        meta = json.loads((P / f"{kind}_meta.json").read_text())
        npz = np.load(P / f"{kind}_packed.npz")
        order = npz["order"]
        teach = f.get_slice(tkey)
        lo = 0
        for bm in meta:
            kb, rows, fmt = bm["kb"], bm["rows"], bm["opt"]
            take = min(rows, args.rows)
            sel = np.linspace(0, rows - 1, take).astype(np.int64)
            scales = t(npz[f"b{kb}_scales"], dev)
            if fmt.startswith("lm"):
                k = bm["k"]
                bpr = (2304 * k) // 8
                codes = npz[f"b{kb}_codes"].reshape(rows, bpr)[sel]
                Q = decode_lm(t(codes, dev), scales[sel], t(npz[f"b{kb}_levels"], dev), take, 2304, k)
            else:
                codes = npz[f"b{kb}_codes"].reshape(rows, -1)[sel]
                Q = decode_trellis(t(codes, dev), scales[sel], t(luts[fmt], dev), 2304, fmt)
            si = t(npz[f"b{kb}_signs_in"].astype(np.float32), dev)
            W = effective_weight(Q, si)
            tok = order[lo + sel]
            Wt = torch.stack([teach[int(i) : int(i) + 1][0] for i in tok]).to(dev).float()
            report(f"{kind} bucket {kb} rows {lo}..{lo + rows}", fmt, W, Wt, effective_weight(Q, torch.ones_like(si)))
            lo += rows


if __name__ == "__main__":
    main()
