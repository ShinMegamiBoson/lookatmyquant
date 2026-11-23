"""Attention for Inkling-2-Mini: GQA with a relative-position bias and one sink logit per head and query.

Both `decode_attention` (split-K flash decoding, CUDA-graph-capturable) and `prefill_attention` /
`prefill_attention_kv` (tiled O(1)-memory banded FlashAttention-2 with causal/window block skipping)
support:
  - FP16 linear or sliding-window ring buffers (`ring=True`, `pos % kv_cap`)
  - Block-32 `q8_0` (int8 + fp16 scales) and `q4_0` (packed uint8 nibbles + fp16 scales) global caches
  - Tiered VRAM + pinned host-RAM global caches (`kv_host`, `vram_cells`)

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

from .conv import KV_MODE_IDS


@triton.jit
def _load_kv_tile(
    KV, SC, KV_HOST, SC_HOST,
    kvh, kk, ok, is_v,
    stride_kv_s, stride_kv_h, stride_kv_t,
    stride_sc_s, stride_sc_h, stride_sc_t,
    stride_hkv_s, stride_hkv_h, stride_hkv_t,
    stride_hsc_s, stride_hsc_h, stride_hsc_t,
    KV_CAP, VRAM_CELLS,
    D: tl.constexpr, BK: tl.constexpr,
    RING: tl.constexpr, KV_MODE: tl.constexpr, HAS_HOST: tl.constexpr,
):
    """Load and dequantize one `[BK, D]` tile of K (`is_v=0`) or V (`is_v=1`) into float32."""
    slot_kk = (kk % KV_CAP) if RING else kk
    if HAS_HOST:
        in_vram = ok & (slot_kk < VRAM_CELLS)
        in_host = ok & (slot_kk >= VRAM_CELLS)
        vk = tl.where(in_vram, slot_kk, 0).to(tl.int64)
        hk = tl.where(in_host, slot_kk - VRAM_CELLS, 0).to(tl.int64)
    else:
        in_vram = ok
        vk = tl.where(ok, slot_kk, 0).to(tl.int64)
        in_host = ok & False
        hk = vk

    n_groups: tl.constexpr = D // 32
    if KV_MODE == 0:  # FP16 / FP32
        d = tl.arange(0, D)
        base_v = KV + is_v * stride_kv_s + kvh * stride_kv_h + vk[:, None] * stride_kv_t + d[None, :]
        val = tl.load(base_v, mask=in_vram[:, None], other=0.0).to(tl.float32)
        if HAS_HOST:
            base_h = KV_HOST + is_v * stride_hkv_s + kvh * stride_hkv_h + hk[:, None] * stride_hkv_t + d[None, :]
            hval = tl.load(base_h, mask=in_host[:, None], other=0.0).to(tl.float32)
            val = tl.where(in_host[:, None], hval, val)
        return val
    elif KV_MODE == 1:  # q8_0: int8 [BK, D] + fp16 scales [BK, D//32]
        d = tl.arange(0, D)
        g = tl.arange(0, n_groups)
        base_v = KV + is_v * stride_kv_s + kvh * stride_kv_h + vk[:, None] * stride_kv_t + d[None, :]
        sc_v = SC + is_v * stride_sc_s + kvh * stride_sc_h + vk[:, None] * stride_sc_t + g[None, :]
        q = tl.load(base_v, mask=in_vram[:, None], other=0).to(tl.float32)
        sc = tl.load(sc_v, mask=in_vram[:, None], other=0.0).to(tl.float32)
        if HAS_HOST:
            base_h = KV_HOST + is_v * stride_hkv_s + kvh * stride_hkv_h + hk[:, None] * stride_hkv_t + d[None, :]
            sc_h = SC_HOST + is_v * stride_hsc_s + kvh * stride_hsc_h + hk[:, None] * stride_hsc_t + g[None, :]
            qh = tl.load(base_h, mask=in_host[:, None], other=0).to(tl.float32)
            sch = tl.load(sc_h, mask=in_host[:, None], other=0.0).to(tl.float32)
            q = tl.where(in_host[:, None], qh, q)
            sc = tl.where(in_host[:, None], sch, sc)
        return tl.reshape(tl.reshape(q, (BK, n_groups, 32)) * sc[:, :, None], (BK, D))
    else:  # KV_MODE == 2: q4_0 packed uint8 [BK, D//2] + fp16 scales [BK, D//32]
        dh = tl.arange(0, D // 2)
        g = tl.arange(0, n_groups)
        base_v = KV + is_v * stride_kv_s + kvh * stride_kv_h + vk[:, None] * stride_kv_t + dh[None, :]
        sc_v = SC + is_v * stride_sc_s + kvh * stride_sc_h + vk[:, None] * stride_sc_t + g[None, :]
        packed = tl.load(base_v, mask=in_vram[:, None], other=0x88)
        sc = tl.load(sc_v, mask=in_vram[:, None], other=0.0).to(tl.float32)
        if HAS_HOST:
            base_h = KV_HOST + is_v * stride_hkv_s + kvh * stride_hkv_h + hk[:, None] * stride_hkv_t + dh[None, :]
            sc_h = SC_HOST + is_v * stride_hsc_s + kvh * stride_hsc_h + hk[:, None] * stride_hsc_t + g[None, :]
            ph = tl.load(base_h, mask=in_host[:, None], other=0x88)
            sch = tl.load(sc_h, mask=in_host[:, None], other=0.0).to(tl.float32)
            packed = tl.where(in_host[:, None], ph, packed)
            sc = tl.where(in_host[:, None], sch, sc)
        lo = (packed & 0x0F).to(tl.float32) - 8.0
        hi = (packed >> 4).to(tl.float32) - 8.0
        unpacked = tl.reshape(tl.join(lo, hi), (BK, n_groups, 32))
        return tl.reshape(unpacked * sc[:, :, None], (BK, D))


@triton.jit
def _decode_attn_split_kernel(
    Q, KV, SC, KV_HOST, SC_HOST, REL, SINK, POS, PM, PL, PACC, PROJ,
    stride_kv_s, stride_kv_h, stride_kv_t,
    stride_sc_s, stride_sc_h, stride_sc_t,
    stride_hkv_s, stride_hkv_h, stride_hkv_t,
    stride_hsc_s, stride_hsc_h, stride_hsc_t,
    KV_CAP, VRAM_CELLS,
    scale,
    N_REP: tl.constexpr, R_PAD: tl.constexpr, D: tl.constexpr, REL_EXT: tl.constexpr,
    WINDOW: tl.constexpr, BLOCK_K: tl.constexpr, NSPLIT: tl.constexpr, D_REL: tl.constexpr,
    INLINE_REL: tl.constexpr, RING: tl.constexpr, KV_MODE: tl.constexpr, HAS_HOST: tl.constexpr,
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
        k = _load_kv_tile(
            KV, SC, KV_HOST, SC_HOST, kvh, kk, ok, 0,
            stride_kv_s, stride_kv_h, stride_kv_t,
            stride_sc_s, stride_sc_h, stride_sc_t,
            stride_hkv_s, stride_hkv_h, stride_hkv_t,
            stride_hsc_s, stride_hsc_h, stride_hsc_t,
            KV_CAP, VRAM_CELLS,
            D=D, BK=BLOCK_K, RING=RING, KV_MODE=KV_MODE, HAS_HOST=HAS_HOST,
        )
        v = _load_kv_tile(
            KV, SC, KV_HOST, SC_HOST, kvh, kk, ok, 1,
            stride_kv_s, stride_kv_h, stride_kv_t,
            stride_sc_s, stride_sc_h, stride_sc_t,
            stride_hkv_s, stride_hkv_h, stride_hkv_t,
            stride_hsc_s, stride_hsc_h, stride_hsc_t,
            KV_CAP, VRAM_CELLS,
            D=D, BK=BLOCK_K, RING=RING, KV_MODE=KV_MODE, HAS_HOST=HAS_HOST,
        )
        s = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale  # [R, BK]
        in_rel = ok & (dist < REL_EXT)
        if INLINE_REL:
            pj = tl.load(PROJ + kr[:, None] * REL_EXT + tl.where(in_rel, dist, 0)[None, :], mask=in_rel[None, :], other=0.0)
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


def decode_attention(
    q, kv, rel, sink, pos, window: int, rel_extent: int,
    block_k: int = 32, nsplit: int = 16, proj=None,
    *,
    ring: bool = False,
    kv_mode: str = "fp16",
    scales=None,
    kv_host=None,
    scales_host=None,
    vram_cells: int | None = None,
):
    """q `[H, D]` f32, kv `[2, H_kv, cap, D_stored]` holding keys up to `pos`, rel `[H, rel_extent]` f32 (or, with
    `proj` `[d_rel, rel_extent]`, the `[H, d_rel]` states whose bias the kernel computes), sink `[H]` f32, pos int32
    `[1]` on device. Returns `[H, D]` f32. Keys are split across `nsplit` programs a head (flash-decoding), merged."""
    H, D = q.shape
    H_kv = kv.shape[1]
    n_rep = H // H_kv
    R = triton.next_power_of_2(n_rep)
    dev = q.device
    mode_id = KV_MODE_IDS[kv_mode]
    sc = scales if scales is not None else kv
    has_host = kv_host is not None and vram_cells is not None
    hkv = kv_host if has_host else kv
    hsc = scales_host if (has_host and scales_host is not None) else sc
    kv_cap = int(kv.shape[2])
    v_cells = int(vram_cells) if has_host else kv_cap
    pm = torch.empty(H_kv * nsplit * R, device=dev)
    pl = torch.empty_like(pm)
    pacc = torch.empty(H_kv * nsplit * R, D, device=dev)
    out = torch.empty(H, D, device=dev)
    with torch.cuda.device(dev):
        _decode_attn_split_kernel[(H_kv, nsplit)](
            q.float().contiguous(), kv, sc, hkv, hsc,
            rel.float().contiguous(), sink.float().contiguous(), pos, pm, pl, pacc,
            proj if proj is not None else rel,
            kv.stride(0), kv.stride(1), kv.stride(2),
            sc.stride(0), sc.stride(1), sc.stride(2),
            hkv.stride(0), hkv.stride(1), hkv.stride(2),
            hsc.stride(0), hsc.stride(1), hsc.stride(2),
            kv_cap, v_cells,
            1.0 / math.sqrt(D),
            N_REP=n_rep, R_PAD=R, D=D, REL_EXT=rel_extent, WINDOW=window, BLOCK_K=block_k, NSPLIT=nsplit,
            D_REL=rel.shape[-1] if proj is not None else 16, INLINE_REL=proj is not None,
            RING=ring, KV_MODE=mode_id, HAS_HOST=has_host,
        )
        _decode_attn_merge_kernel[(H_kv, R)](pm, pl, pacc, out, N_REP=n_rep, R_PAD=R, D=D, NSPLIT=nsplit)
    return out


@triton.jit
def _prefill_flash_attn_kernel(
    Q, KV, SC, KV_HOST, SC_HOST, REL, SINK, Q_POS, OUT,
    stride_q_t, stride_q_h,
    stride_kv_s, stride_kv_h, stride_kv_t,
    stride_sc_s, stride_sc_h, stride_sc_t,
    stride_hkv_s, stride_hkv_h, stride_hkv_t,
    stride_hsc_s, stride_hsc_h, stride_hsc_t,
    stride_rel_t, stride_rel_h,
    stride_sink_t, stride_sink_h,
    stride_o_t, stride_o_h,
    T, L, KV_CAP, VRAM_CELLS, scale,
    N_REP: tl.constexpr, D: tl.constexpr, REL_EXT: tl.constexpr, WINDOW: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
    RING: tl.constexpr, KV_MODE: tl.constexpr, HAS_HOST: tl.constexpr,
):
    """Tiled O(1)-memory banded FlashAttention-2 with causal/window block skipping, rel-pos bias, and sinks."""
    h = tl.program_id(0)
    qb = tl.program_id(1)
    kvh = h // N_REP
    scale = scale.to(tl.float32)

    q_offs = qb * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_ok = q_offs < T
    d = tl.arange(0, D)

    q = tl.load(Q + q_offs[:, None] * stride_q_t + h * stride_q_h + d[None, :], mask=q_ok[:, None], other=0.0).to(tl.float32)
    q_pos = tl.load(Q_POS + q_offs, mask=q_ok, other=0).to(tl.int32)
    m_i = tl.load(SINK + q_offs * stride_sink_t + h * stride_sink_h, mask=q_ok, other=0.0).to(tl.float32)
    l_i = tl.where(q_ok, 1.0, 0.0)
    acc = tl.zeros((BLOCK_Q, D), tl.float32)

    # Contiguous query positions for this tile -> exact non-empty key span [k_lo, k_hi)
    q_pos_first = tl.load(Q_POS + qb * BLOCK_Q).to(tl.int32)
    last_idx = tl.minimum((qb + 1) * BLOCK_Q - 1, T - 1)
    q_pos_last = tl.load(Q_POS + last_idx).to(tl.int32)
    k_lo = tl.maximum(q_pos_first - (WINDOW - 1), 0) if WINDOW > 0 else 0
    k_hi = tl.minimum(L, q_pos_last + 1)

    for k0 in range(k_lo, k_hi, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        k_valid = kk < k_hi
        k = _load_kv_tile(
            KV, SC, KV_HOST, SC_HOST, kvh, kk, k_valid, 0,
            stride_kv_s, stride_kv_h, stride_kv_t,
            stride_sc_s, stride_sc_h, stride_sc_t,
            stride_hkv_s, stride_hkv_h, stride_hkv_t,
            stride_hsc_s, stride_hsc_h, stride_hsc_t,
            KV_CAP, VRAM_CELLS,
            D=D, BK=BLOCK_K, RING=RING, KV_MODE=KV_MODE, HAS_HOST=HAS_HOST,
        )
        v = _load_kv_tile(
            KV, SC, KV_HOST, SC_HOST, kvh, kk, k_valid, 1,
            stride_kv_s, stride_kv_h, stride_kv_t,
            stride_sc_s, stride_sc_h, stride_sc_t,
            stride_hkv_s, stride_hkv_h, stride_hkv_t,
            stride_hsc_s, stride_hsc_h, stride_hsc_t,
            KV_CAP, VRAM_CELLS,
            D=D, BK=BLOCK_K, RING=RING, KV_MODE=KV_MODE, HAS_HOST=HAS_HOST,
        )
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale  # [BQ, BK]
        dist = q_pos[:, None] - kk[None, :]
        ok = q_ok[:, None] & k_valid[None, :] & (dist >= 0)
        if WINDOW > 0:
            ok = ok & (dist < WINDOW)
        if q_pos_first - (k0 + BLOCK_K - 1) < REL_EXT:
            in_rel = ok & (dist < REL_EXT)
            safe_d = tl.where(in_rel, dist, 0)
            b = tl.load(
                REL + q_offs[:, None] * stride_rel_t + h * stride_rel_h + safe_d,
                mask=in_rel,
                other=0.0,
            ).to(tl.float32)
            s = s + b
        s = tl.where(ok, s, float("-inf"))
        m_tile = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_tile)
        alpha = tl.exp(m_i - m_new)
        p = tl.where(ok, tl.exp(s - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new

    out = acc / tl.where(q_ok, l_i, 1.0)[:, None]
    tl.store(OUT + q_offs[:, None] * stride_o_t + h * stride_o_h + d[None, :], out, mask=q_ok[:, None])


def prefill_attention_kv(
    q, kv, rel, sink, q_pos, L: int, window: int, rel_extent: int,
    *,
    ring: bool = False,
    kv_mode: str = "fp16",
    scales=None,
    kv_host=None,
    scales_host=None,
    vram_cells: int | None = None,
    block_q: int = 32,
    block_k: int = 32,
):
    """Tiled O(1)-memory banded FlashAttention-2 reading directly from a KV cache slot.

    q: `[T, H, D]` f32, kv: `[2, H_kv, cap, D_stored]`, rel: `[T, H, rel_extent]` f32,
    sink: `[T, H]` f32, q_pos: `[T]` int32/int64 (monotone increasing). Returns `[T, H, D]` f32.
    """
    T, H, D = q.shape
    H_kv = kv.shape[1]
    n_rep = H // H_kv
    mode_id = KV_MODE_IDS[kv_mode]
    sc = scales if scales is not None else kv
    has_host = kv_host is not None and vram_cells is not None
    hkv = kv_host if has_host else kv
    hsc = scales_host if (has_host and scales_host is not None) else sc
    kv_cap = int(kv.shape[2])
    v_cells = int(vram_cells) if has_host else kv_cap
    qf = q.float().contiguous()
    relf = rel.float().contiguous()
    sinkf = sink.float().contiguous()
    qpos_i32 = q_pos.to(torch.int32).contiguous()
    out = torch.empty(T, H, D, dtype=torch.float32, device=q.device)
    with torch.cuda.device(q.device):
        grid = (H, triton.cdiv(T, block_q))
        _prefill_flash_attn_kernel[grid](
            qf, kv, sc, hkv, hsc, relf, sinkf, qpos_i32, out,
            qf.stride(0), qf.stride(1),
            kv.stride(0), kv.stride(1), kv.stride(2),
            sc.stride(0), sc.stride(1), sc.stride(2),
            hkv.stride(0), hkv.stride(1), hkv.stride(2),
            hsc.stride(0), hsc.stride(1), hsc.stride(2),
            relf.stride(0), relf.stride(1),
            sinkf.stride(0), sinkf.stride(1),
            out.stride(0), out.stride(1),
            T, int(L), kv_cap, v_cells, 1.0 / math.sqrt(D),
            N_REP=n_rep, D=D, REL_EXT=rel_extent, WINDOW=window,
            BLOCK_Q=block_q, BLOCK_K=block_k,
            RING=ring, KV_MODE=mode_id, HAS_HOST=has_host,
            num_warps=4, num_stages=1,
        )
    return out


def prefill_attention(q, K, V, rel, sink, q_pos, k_pos, window: int, rel_extent: int):
    """q `[H, T, D]`, K/V `[H_kv, L, D]`, rel `[T, H, rel_extent]`, sink `[T, H]`, positions `[T]` and `[L]`.
    Returns `[T, H, D]` f32. Uses O(1)-memory tiled FlashAttention-2 when `k_pos` is `0..L-1` on CUDA."""
    H, T, D = q.shape
    H_kv, L, _ = K.shape
    if q.is_cuda and D in (64, 128) and L > 0:
        # Fast tiled path whenever k_pos is standard contiguous 0..L-1 and q_pos is monotone
        if T <= 1 or bool((q_pos[1:] >= q_pos[:-1]).all()):
            kv = torch.stack([K.contiguous(), V.contiguous()], dim=0)
            return prefill_attention_kv(
                q.transpose(0, 1), kv, rel, sink, q_pos, L, window, rel_extent,
            )
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
