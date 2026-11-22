"""Attention for Inkling-2-Mini: GQA with a relative-position bias and one sink logit per head and query.

`decode_attention` is a Triton flash-decoding kernel for one query token that reads the current position from device
memory, so a CUDA graph can replay it at any sequence length and it only touches the keys it needs. `prefill_attention`
handles any number of query tokens with PyTorch's SDPA: the sink is an extra key with a zero value whose bias is the
sink logit.

Semantics (transformers `eager_attention_forward` + MLX `_gqa_sdpa_sinks`):
    score[h, k] = q[h] . K[k] / sqrt(d) + (rel[h, dist] if 0 <= dist < rel_extent else 0),   dist = pos_q - pos_k
    allowed: dist >= 0, and dist < window on sliding layers
    out[h] = sum_k softmax([scores, sink[h]])[k] * V[k]
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _decode_attn_split_kernel(
    Q, KV, REL, SINK, POS, PM, PL, PACC, PROJ,
    stride_kv_s, stride_kv_h, stride_kv_t,
    scale,
    N_REP: tl.constexpr, R_PAD: tl.constexpr, D: tl.constexpr, REL_EXT: tl.constexpr,
    WINDOW: tl.constexpr, BLOCK_K: tl.constexpr, NSPLIT: tl.constexpr, D_REL: tl.constexpr, INLINE_REL: tl.constexpr,
):
    """One (kv head, split) program: online softmax over its share of the keys. Split 0 also carries the sink."""
    kvh = tl.program_id(0)
    sp = tl.program_id(1)
    scale = scale.to(tl.float32)  # torch.compile passes Python floats as float64
    pos = tl.load(POS).to(tl.int32)
    r = tl.arange(0, R_PAD)
    r_ok = r < N_REP
    heads = kvh * N_REP + r
    d = tl.arange(0, D)
    q = tl.load(Q + heads[:, None] * D + d[None, :], mask=r_ok[:, None], other=0.0)  # [R, D]
    lo = 0
    if WINDOW > 0:
        lo = tl.maximum(pos - (WINDOW - 1), 0)
    n_blocks = tl.cdiv(pos + 1 - lo, BLOCK_K)
    per = tl.cdiv(n_blocks, NSPLIT)
    b0 = sp * per
    b1 = tl.minimum(b0 + per, n_blocks)
    if sp == 0:
        m_i = tl.load(SINK + heads, mask=r_ok, other=0.0)  # the sink: one logit with weight 1 and a zero value
        l_i = tl.full((R_PAD,), 1.0, tl.float32)
    else:
        m_i = tl.full((R_PAD,), -1.0e30, tl.float32)
        l_i = tl.zeros((R_PAD,), tl.float32)
    acc = tl.zeros((R_PAD, D), tl.float32)
    if INLINE_REL:  # REL holds the [H, D_REL] relative-position states; the bias is rel . proj[:, dist]
        kr = tl.arange(0, D_REL)
        relq = tl.load(REL + heads[:, None] * D_REL + kr[None, :], mask=r_ok[:, None], other=0.0)  # [R, D_REL]
    for blk in range(b0, b1):
        kk = lo + blk * BLOCK_K + tl.arange(0, BLOCK_K)
        dist = pos - kk
        ok = dist >= 0
        base = KV + kvh * stride_kv_h + kk[:, None] * stride_kv_t + d[None, :]
        k = tl.load(base, mask=ok[:, None], other=0.0).to(tl.float32)  # [BK, D]
        v = tl.load(base + stride_kv_s, mask=ok[:, None], other=0.0).to(tl.float32)
        s = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale  # [R, BK]
        in_rel = ok & (dist < REL_EXT)
        if INLINE_REL:
            pj = tl.load(PROJ + kr[:, None] * REL_EXT + tl.where(in_rel, dist, 0)[None, :], mask=in_rel[None, :], other=0.0)  # [D_REL, BK]
            b = tl.where(in_rel[None, :], tl.sum(relq[:, :, None] * pj[None, :, :], axis=1), 0.0)
        else:
            b = tl.load(REL + heads[:, None] * REL_EXT + tl.where(in_rel, dist, 0)[None, :], mask=r_ok[:, None] & in_rel[None, :], other=0.0)
        s = tl.where(ok[None, :], s + b, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
        m_i = m_new
    o = (kvh * NSPLIT + sp) * R_PAD + r
    tl.store(PM + o, m_i)
    tl.store(PL + o, l_i)
    tl.store(PACC + o[:, None] * D + d[None, :], acc)


@triton.jit
def _decode_attn_merge_kernel(PM, PL, PACC, OUT, N_REP: tl.constexpr, R_PAD: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr):
    kvh = tl.program_id(0)
    r = tl.program_id(1)
    sp = tl.arange(0, NSPLIT)
    o = (kvh * NSPLIT + sp) * R_PAD + r
    m = tl.load(PM + o)
    l = tl.load(PL + o)
    mx = tl.max(m, axis=0)
    w = tl.exp(m - mx)
    d = tl.arange(0, D)
    acc = tl.sum(tl.load(PACC + o[:, None] * D + d[None, :]) * w[:, None], axis=0)
    if r < N_REP:
        tl.store(OUT + (kvh * N_REP + r) * D + d, acc / tl.sum(l * w, axis=0))


def decode_attention(q, kv, rel, sink, pos, window: int, rel_extent: int, block_k: int = 32, nsplit: int = 16, proj=None):
    """q `[H, D]` f32, kv `[2, H_kv, cap, D]` (f16/f32) holding keys up to `pos`, rel `[H, rel_extent]` f32 (or, with
    `proj` `[d_rel, rel_extent]`, the `[H, d_rel]` states whose bias the kernel computes), sink `[H]` f32, pos int32
    `[1]` on device. Returns `[H, D]` f32. Keys are split across `nsplit` programs a head (flash-decoding), merged."""
    H, D = q.shape
    H_kv = kv.shape[1]
    n_rep = H // H_kv
    R = triton.next_power_of_2(n_rep)
    dev = q.device
    pm = torch.empty(H_kv * nsplit * R, device=dev)
    pl = torch.empty_like(pm)
    pacc = torch.empty(H_kv * nsplit * R, D, device=dev)
    out = torch.empty(H, D, device=dev)
    with torch.cuda.device(dev):
        _decode_attn_split_kernel[(H_kv, nsplit)](
            q.float().contiguous(), kv, rel.float().contiguous(), sink.float().contiguous(), pos, pm, pl, pacc,
            proj if proj is not None else rel, kv.stride(0), kv.stride(1), kv.stride(2), 1.0 / math.sqrt(D),
            N_REP=n_rep, R_PAD=R, D=D, REL_EXT=rel_extent, WINDOW=window, BLOCK_K=block_k, NSPLIT=nsplit,
            D_REL=rel.shape[-1] if proj is not None else 16, INLINE_REL=proj is not None,
        )
        _decode_attn_merge_kernel[(H_kv, R)](pm, pl, pacc, out, N_REP=n_rep, R_PAD=R, D=D, NSPLIT=nsplit)
    return out


def prefill_attention(q, K, V, rel, sink, q_pos, k_pos, window: int, rel_extent: int):
    """q `[H, T, D]`, K/V `[H_kv, L, D]`, rel `[T, H, rel_extent]`, sink `[T, H]`, positions `[T]` and `[L]`.
    Returns `[T, H, D]` f32."""
    H, T, D = q.shape
    H_kv, L, _ = K.shape
    n_rep = H // H_kv
    dist = q_pos[:, None] - k_pos[None, :]  # [T, L]
    ok = dist >= 0
    if window > 0:
        ok = ok & (dist < window)
    in_rel = ok & (dist < rel_extent)
    idx = dist.clamp(0, rel_extent - 1)
    bias = torch.gather(rel.permute(1, 0, 2), 2, idx[None].expand(H, T, L))  # [H, T, L]
    bias = torch.where(in_rel[None], bias, 0.0)
    bias = torch.where(ok[None], bias, float("-inf"))
    bias = torch.cat([bias, sink.T[:, :, None]], dim=-1)  # sink: one extra key, zero value
    Kp = torch.cat([K.float(), K.new_zeros(H_kv, 1, D, dtype=torch.float32)], dim=1)
    Vp = torch.cat([V.float(), V.new_zeros(H_kv, 1, D, dtype=torch.float32)], dim=1)
    Kp = Kp.repeat_interleave(n_rep, 0)
    Vp = Vp.repeat_interleave(n_rep, 0)
    out = F.scaled_dot_product_attention(q.float()[None], Kp[None], Vp[None], attn_mask=bias[None])[0]  # [H, T, D]
    return out.transpose(0, 1)
