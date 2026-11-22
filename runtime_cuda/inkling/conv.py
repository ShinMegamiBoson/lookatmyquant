"""Causal 4-tap convolutions with their 3-token state updated in place, one Triton launch each.

Static (`ShortConvolution`): out_t = (sum_lag ctx[t + lag] * w[lag] + x_t) / sqrt(5), ctx = [prev3, x] - so `w[3]`
weights the current token. Dynamic (`DynamicShortConvolution`, full-attention K/V): taps are `static[h, d, k] +
dyn[t, h, k]` with tap 0 on the current token, and the output is divided by sqrt(9). Each program owns BLOCK_C
channels and walks the tokens in order, keeping the last three inputs in registers.
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
    KR, VR, PK, PV, WK, WV, DK, DV, KV, POS,
    T, C, thr, eps, stride_kv_s, stride_kv_h, stride_kv_t,
    DYNAMIC: tl.constexpr, HD: tl.constexpr,
):
    """runtime_mlx's kv_sconv4_norm_fused: per (head, K-or-V) program, conv each token, then L2-clip (K) or
    RMSNorm (V) over the head, and write float16 into the cache slot at the token's position."""
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
        pos = tl.load(POS + t).to(tl.int64)
        tl.store(KV + is_v * stride_kv_s + head * stride_kv_h + pos * stride_kv_t + d, y.to(tl.float16))
        p0 = p1
        p1 = p2
        p2 = x
    tl.store(PREV + c, p0)
    tl.store(PREV + C + c, p1)
    tl.store(PREV + 2 * C + c, p2)


def kv_conv_write_(k_raw, v_raw, prev_k, prev_v, kv, pos, thr: float, eps: float, n_kv: int,
                   w_k=None, w_v=None, dyn=None):
    """Conv + clip/norm + cache write for K and V. Static: `w_k`, `w_v` `[4, C]`. Dynamic: `dyn` =
    (k_in, v_in, (st_k [h, d, 4], proj_k), (st_v, proj_v))."""
    T, C = k_raw.shape
    HD = C // n_kv
    if dyn is not None:
        k_in, v_in, (st_k, pr_k), (st_v, pr_v) = dyn
        dk = (k_in.float() @ pr_k.T).contiguous()
        dv = (v_in.float() @ pr_v.T).contiguous()
        wk, wv = st_k.reshape(C, 4), st_v.reshape(C, 4)
    else:
        dk = dv = k_raw
        wk, wv = w_k, w_v
    with torch.cuda.device(k_raw.device):
        _kv_conv_write_kernel[(n_kv, 2)](
            k_raw.contiguous(), v_raw.contiguous(), prev_k, prev_v, wk, wv, dk, dv, kv, pos,
            T, C, thr, eps, kv.stride(0), kv.stride(1), kv.stride(2), DYNAMIC=dyn is not None, HD=HD,
        )
