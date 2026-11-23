"""Causal 4-tap convolutions with their 3-token state updated in place, one Triton launch each.

Static (`ShortConvolution`): out_t = (sum_lag ctx[t + lag] * w[lag] + x_t) / sqrt(5), ctx = [prev3, x] - so `w[3]`
weights the current token. Dynamic (`DynamicShortConvolution`, full-attention K/V): taps are `static[h, d, k] +
dyn[t, h, k]` with tap 0 on the current token, and the output is divided by sqrt(9). Each program owns BLOCK_C
channels and walks the tokens in order, keeping the last three inputs in registers.

`kv_conv_write_` fuses causal conv + K L2-clip / V RMSNorm + cache write, supporting:
  - FP16 linear or ring-buffered sliding-window caches (`ring=True`, `pos % kv_cap`)
  - Block-32 `q8_0` (int8 + fp16 scales) and `q4_0` (packed uint8 nibbles + fp16 scales) global caches
  - Tiered VRAM + pinned host-RAM global caches (`kv_host`, `vram_cells`)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _conv4_kernel(X, PREV, W, OUT, T, C, DYN, HD, DYNAMIC: tl.constexpr, BLOCK_C: tl.constexpr):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    ok = c < C
    p0 = tl.load(PREV + c, mask=ok, other=0.0)  # oldest
    p1 = tl.load(PREV + C + c, mask=ok, other=0.0)
    p2 = tl.load(PREV + 2 * C + c, mask=ok, other=0.0)  # newest
    if DYNAMIC:  # W is static [C, 4] (row c = channel, col k = tap), DYN is [T, heads, 4]
        s0 = tl.load(W + c * 4 + 0, mask=ok, other=0.0)
        s1 = tl.load(W + c * 4 + 1, mask=ok, other=0.0)
        s2 = tl.load(W + c * 4 + 2, mask=ok, other=0.0)
        s3 = tl.load(W + c * 4 + 3, mask=ok, other=0.0)
        head = c // HD
    else:  # W is [4, C]
        w0 = tl.load(W + c, mask=ok, other=0.0)
        w1 = tl.load(W + C + c, mask=ok, other=0.0)
        w2 = tl.load(W + 2 * C + c, mask=ok, other=0.0)
        w3 = tl.load(W + 3 * C + c, mask=ok, other=0.0)
    for t in range(T):
        x = tl.load(X + t * C + c, mask=ok, other=0.0).to(tl.float32)
        if DYNAMIC:
            nh = C // HD
            d = DYN + (t * nh + head) * 4
            y = (x * (s0 + tl.load(d + 0, mask=ok, other=0.0)) + p2 * (s1 + tl.load(d + 1, mask=ok, other=0.0))
                 + p1 * (s2 + tl.load(d + 2, mask=ok, other=0.0)) + p0 * (s3 + tl.load(d + 3, mask=ok, other=0.0)) + x) * (1.0 / 3.0)
        else:
            y = (p0 * w0 + p1 * w1 + p2 * w2 + x * w3 + x) * 0.4472135954999579  # 1 / sqrt(5)
        tl.store(OUT + t * C + c, y, mask=ok)
        p0 = p1
        p1 = p2
        p2 = x
    tl.store(PREV + c, p0, mask=ok)
    tl.store(PREV + C + c, p1, mask=ok)
    tl.store(PREV + 2 * C + c, p2, mask=ok)


def conv4_(prev: torch.Tensor, x: torch.Tensor, w_kc: torch.Tensor, BLOCK_C: int = 256) -> torch.Tensor:
    """Static conv; `prev` `[3, C]` float32 is updated in place. x `[T, C]`, w_kc `[4, C]`."""
    T, C = x.shape
    out = torch.empty(T, C, dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        _conv4_kernel[(triton.cdiv(C, BLOCK_C),)](x.contiguous(), prev, w_kc, out, T, C, x, 1, DYNAMIC=False, BLOCK_C=BLOCK_C)
    return out


def dyn_conv4_(prev, x, dyn_in, st_hdk, proj_w, BLOCK_C: int = 256) -> torch.Tensor:
    """Dynamic conv; `prev` updated in place. st_hdk `[heads, head_dim, 4]`, proj_w `[heads * 4, H]`."""
    T, C = x.shape
    nh, hd = st_hdk.shape[0], st_hdk.shape[1]
    dyn = (dyn_in.float() @ proj_w.T).contiguous()  # [T, heads * 4]
    out = torch.empty(T, C, dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        _conv4_kernel[(triton.cdiv(C, BLOCK_C),)](x.contiguous(), prev, st_hdk.reshape(C, 4), out, T, C, dyn, hd,
                                                  DYNAMIC=True, BLOCK_C=BLOCK_C)
    return out


@triton.jit
def _kv_conv_write_kernel(
    KR, VR, PK, PV, WK, WV, DK, DV,
    KV, SC, KV_HOST, SC_HOST, POS,
    T, C, thr, eps,
    stride_kv_s, stride_kv_h, stride_kv_t,
    stride_sc_s, stride_sc_h, stride_sc_t,
    stride_hkv_s, stride_hkv_h, stride_hkv_t,
    stride_hsc_s, stride_hsc_h, stride_hsc_t,
    KV_CAP, VRAM_CELLS,
    DYNAMIC: tl.constexpr, HD: tl.constexpr,
    RING: tl.constexpr, KV_MODE: tl.constexpr, HAS_HOST: tl.constexpr,
):
    """Per (head, K-or-V) program: conv each token, L2-clip (K) or RMSNorm (V), quantize if needed, write to cache."""
    head = tl.program_id(0)
    is_v = tl.program_id(1)
    thr = thr.to(tl.float32)
    eps = eps.to(tl.float32)
    d = tl.arange(0, HD)
    c = head * HD + d
    X = VR if is_v == 1 else KR
    PREV = PV if is_v == 1 else PK
    W = WV if is_v == 1 else WK
    D = DV if is_v == 1 else DK
    p0 = tl.load(PREV + c)
    p1 = tl.load(PREV + C + c)
    p2 = tl.load(PREV + 2 * C + c)
    if DYNAMIC:  # W: static [C, 4]; D: per-token [T, heads, 4]
        s0 = tl.load(W + c * 4 + 0)
        s1 = tl.load(W + c * 4 + 1)
        s2 = tl.load(W + c * 4 + 2)
        s3 = tl.load(W + c * 4 + 3)
    else:  # W: [4, C]
        w0 = tl.load(W + c)
        w1 = tl.load(W + C + c)
        w2 = tl.load(W + 2 * C + c)
        w3 = tl.load(W + 3 * C + c)
    nh = C // HD
    n_groups: tl.constexpr = HD // 32
    g_idx = tl.arange(0, n_groups)
    d_half = tl.arange(0, HD // 2)
    for t in range(T):
        x = tl.load(X + t * C + c).to(tl.float32)
        if DYNAMIC:
            dd = D + (t * nh + head) * 4
            y = (x * (s0 + tl.load(dd)) + p2 * (s1 + tl.load(dd + 1)) + p1 * (s2 + tl.load(dd + 2)) + p0 * (s3 + tl.load(dd + 3)) + x) * (1.0 / 3.0)
        else:
            y = (p0 * w0 + p1 * w1 + p2 * w2 + x * w3 + x) * 0.4472135954999579
        if is_v == 1:
            y = y * tl.rsqrt(tl.sum(y * y, axis=0) / HD + eps)
        else:
            y = y * tl.minimum(thr / (tl.sqrt(tl.sum(y * y, axis=0)) + 1e-6), 1.0)
        raw_pos = tl.load(POS + t).to(tl.int64)
        slot_pos = (raw_pos % KV_CAP) if RING else raw_pos
        in_host = HAS_HOST and (slot_pos >= VRAM_CELLS)
        if KV_MODE == 0:  # FP16
            yh = y.to(tl.float16)
            if in_host:
                hp = slot_pos - VRAM_CELLS
                tl.store(KV_HOST + is_v * stride_hkv_s + head * stride_hkv_h + hp * stride_hkv_t + d, yh)
            else:
                tl.store(KV + is_v * stride_kv_s + head * stride_kv_h + slot_pos * stride_kv_t + d, yh)
        elif KV_MODE == 1:  # q8_0: block-32 symmetric int8 + fp16 scales
            y_blk = tl.reshape(y, (n_groups, 32))
            amax = tl.max(tl.abs(y_blk), axis=1)
            sc_h = tl.maximum(amax * (1.0 / 127.0), 1e-6).to(tl.float16)
            inv_sc = 1.0 / sc_h.to(tl.float32)
            q = tl.extra.cuda.libdevice.rint(y_blk * inv_sc[:, None])
            q_i8 = tl.clamp(q, -127.0, 127.0).to(tl.int8)
            q_flat = tl.reshape(q_i8, (HD,))
            if in_host:
                hp = slot_pos - VRAM_CELLS
                tl.store(KV_HOST + is_v * stride_hkv_s + head * stride_hkv_h + hp * stride_hkv_t + d, q_flat)
                tl.store(SC_HOST + is_v * stride_hsc_s + head * stride_hsc_h + hp * stride_hsc_t + g_idx, sc_h)
            else:
                tl.store(KV + is_v * stride_kv_s + head * stride_kv_h + slot_pos * stride_kv_t + d, q_flat)
                tl.store(SC + is_v * stride_sc_s + head * stride_sc_h + slot_pos * stride_sc_t + g_idx, sc_h)
        else:  # KV_MODE == 2: q4_0 block-32 symmetric nibbles [-7..7] + 8 packed 2 per uint8 + fp16 scales
            y_blk = tl.reshape(y, (n_groups, 32))
            amax = tl.max(tl.abs(y_blk), axis=1)
            sc_h = tl.maximum(amax * (1.0 / 7.0), 1e-6).to(tl.float16)
            inv_sc = 1.0 / sc_h.to(tl.float32)
            q = tl.extra.cuda.libdevice.rint(y_blk * inv_sc[:, None])
            q_u8 = (tl.clamp(q, -7.0, 7.0) + 8.0).to(tl.uint8)
            q_lo, q_hi = tl.split(tl.reshape(q_u8, (HD // 2, 2)))
            packed = q_lo | (q_hi << 4)
            if in_host:
                hp = slot_pos - VRAM_CELLS
                tl.store(KV_HOST + is_v * stride_hkv_s + head * stride_hkv_h + hp * stride_hkv_t + d_half, packed)
                tl.store(SC_HOST + is_v * stride_hsc_s + head * stride_hsc_h + hp * stride_hsc_t + g_idx, sc_h)
            else:
                tl.store(KV + is_v * stride_kv_s + head * stride_kv_h + slot_pos * stride_kv_t + d_half, packed)
                tl.store(SC + is_v * stride_sc_s + head * stride_sc_h + slot_pos * stride_sc_t + g_idx, sc_h)
        p0 = p1
        p1 = p2
        p2 = x
    tl.store(PREV + c, p0)
    tl.store(PREV + C + c, p1)
    tl.store(PREV + 2 * C + c, p2)


KV_MODE_IDS = {"fp16": 0, "q8_0": 1, "q4_0": 2}


def kv_conv_write_(
    k_raw, v_raw, prev_k, prev_v, kv, pos, thr: float, eps: float, n_kv: int,
    w_k=None, w_v=None, dyn=None,
    *,
    ring: bool = False,
    kv_mode: str = "fp16",
    scales=None,
    kv_host=None,
    scales_host=None,
    vram_cells: int | None = None,
):
    """Conv + clip/norm + cache write for K and V.

    Supports FP16, `q8_0`, and `q4_0` caches, sliding-window ring buffers (`ring=True`),
    and tiered VRAM + pinned host-RAM storage (`kv_host` + `vram_cells`).
    """
    T, C = k_raw.shape
    HD = C // n_kv
    mode_id = KV_MODE_IDS[kv_mode]
    if dyn is not None:
        k_in, v_in, (st_k, pr_k), (st_v, pr_v) = dyn
        dk = (k_in.float() @ pr_k.T).contiguous()
        dv = (v_in.float() @ pr_v.T).contiguous()
        wk, wv = st_k.reshape(C, 4), st_v.reshape(C, 4)
    else:
        dk = dv = k_raw
        wk, wv = w_k, w_v
    sc = scales if scales is not None else kv
    has_host = kv_host is not None and vram_cells is not None
    hkv = kv_host if has_host else kv
    hsc = scales_host if (has_host and scales_host is not None) else sc
    kv_cap = int(kv.shape[2])
    v_cells = int(vram_cells) if has_host else kv_cap
    with torch.cuda.device(k_raw.device):
        _kv_conv_write_kernel[(n_kv, 2)](
            k_raw.contiguous(), v_raw.contiguous(), prev_k, prev_v, wk, wv, dk, dv,
            kv, sc, hkv, hsc, pos,
            T, C, thr, eps,
            kv.stride(0), kv.stride(1), kv.stride(2),
            sc.stride(0), sc.stride(1), sc.stride(2),
            hkv.stride(0), hkv.stride(1), hkv.stride(2),
            hsc.stride(0), hsc.stride(1), hsc.stride(2),
            kv_cap, v_cells,
            DYNAMIC=dyn is not None, HD=HD,
            RING=ring, KV_MODE=mode_id, HAS_HOST=has_host,
        )
