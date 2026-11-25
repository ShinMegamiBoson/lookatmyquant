"""Causal 4-tap convolutions with their 3-token state updated in place, one Triton launch each.

Static (`ShortConvolution`): out_t = (sum_lag ctx[t + lag] * w[lag] + x_t) / sqrt(5), ctx = [prev3, x] - so `w[3]`
weights the current token. Dynamic (`DynamicShortConvolution`, full-attention K/V): taps are `static[h, d, k] +
dyn[t, h, k]` with tap 0 on the current token, and the output is divided by sqrt(9). Each program owns BLOCK_C
channels and walks the tokens in order, keeping the last three inputs in registers.

`kv_conv_write_` fuses causal conv + K L2-clip / V RMSNorm + cache write, supporting:
  - FP16 linear or ring-buffered sliding-window caches (`ring=True`, `pos % kv_cap`)
  - Block-32 `q8_0` (int8 + fp16 scales) and `q4_0` (packed uint8 nibbles + fp16 scales) global caches
  - Hadamard Lloyd-Max (`lm2`, `lm3`, `lm4`, and asymmetric `lm4_3`, `lm4_2`, `lm3_2`) separate K/V caches
  - Tiered VRAM + pinned host-RAM global caches (`kv_host` / `k_host`+`v_host`, `vram_cells`)
"""
from __future__ import annotations

import numpy as np
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


KV_MODE_IDS = {"fp16": 0, "q8_0": 1, "q4_0": 2, "lm4": 3, "lm3": 4, "lm2": 5}

# Exact unit-Gaussian Lloyd-Max levels and midpoints
LM2_LEVELS = (-1.5104176, -0.4527800, 0.4527800, 1.5104176)
LM2_MIDS = (-0.9815988, 0.0, 0.9815988)

LM3_LEVELS = (-2.1519457, -1.3439093, -0.7560053, -0.2450942, 0.2450942, 0.7560053, 1.3439093, 2.1519457)
LM3_MIDS = (-1.7479275, -1.0499573, -0.5005497, 0.0, 0.5005497, 1.0499573, 1.7479275)

LM4_LEVELS = (
    -2.7325896, -2.0690172, -1.6180464, -1.2562312, -0.9423405, -0.6567591, -0.3880483, -0.1283950,
    0.1283950, 0.3880483, 0.6567591, 0.9423405, 1.2562312, 1.6180464, 2.0690172, 2.7325896,
)
LM4_MIDS = (
    -2.4008034, -1.8435318, -1.4371388, -1.0992858, -0.7995498, -0.5224037, -0.2582217, 0.0,
    0.2582217, 0.5224037, 0.7995498, 1.0992858, 1.4371388, 1.8435318, 2.4008034,
)

LM_LEVELS = {2: LM2_LEVELS, 3: LM3_LEVELS, 4: LM4_LEVELS}
LM_MIDS = {2: LM2_MIDS, 3: LM3_MIDS, 4: LM4_MIDS}

_RHT_SIGN_CACHE: dict[tuple[int, str, int], torch.Tensor] = {}
_LM_LUT_CACHE: dict[tuple[int, str], torch.Tensor] = {}
_LM_MID_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def lm_bytes_per_head(head_dim: int, bits: int) -> int:
    """Packed byte width per KV head for Hadamard Lloyd-Max at 2, 3, or 4 bits."""
    if head_dim < 8 or (head_dim & (head_dim - 1)) != 0:
        raise ValueError(f"Hadamard Lloyd-Max requires power-of-2 head_dim >= 8, got {head_dim}")
    if bits == 4:
        return head_dim // 2
    if bits == 3:
        return 3 * (head_dim // 8)
    if bits == 2:
        return head_dim // 4
    raise ValueError(f"Unsupported LM KV bitwidth: {bits} (expected 2, 3, or 4)")


def parse_kv_mode(mode: str) -> tuple[str, int, int]:
    """Parse a global KV mode string into `(family, k_bits, v_bits)`."""
    m = mode.strip().lower()
    if m == "fp16":
        return ("fp16", 16, 16)
    if m in ("q8_0", "q8"):
        return ("q8_0", 8, 8)
    if m in ("q4_0", "q4"):
        return ("q4_0", 4, 4)
    if m in ("lm2", "lm3", "lm4"):
        b = int(m[2:])
        return ("lm", b, b)
    if m.startswith("lm") and "_" in m:
        parts = m[2:].split("_", 1)
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            kb, vb = int(parts[0]), int(parts[1])
            if kb in (2, 3, 4) and vb in (2, 3, 4):
                return ("lm", kb, vb)
    raise ValueError(
        f"Unsupported KV mode {mode!r}; expected fp16, q8_0, q4_0, lm4, lm4_3, lm4_2, lm3, lm3_2, or lm2"
    )


def default_rht_signs(head_dim: int, device: torch.device | str, seed: int = 20261125) -> torch.Tensor:
    """Deterministic `±1` RHT sign vector `float32[head_dim]` cached on `device`."""
    dev = torch.device(device)
    key = (int(head_dim), str(dev), int(seed))
    t = _RHT_SIGN_CACHE.get(key)
    if t is None:
        rng = np.random.default_rng(int(seed) + int(head_dim))
        arr = (rng.integers(0, 2, size=int(head_dim)) * 2 - 1).astype(np.float32)
        t = torch.from_numpy(arr).to(dev)
        _RHT_SIGN_CACHE[key] = t
    return t


def lm_levels_tensor(bits: int, device: torch.device | str) -> torch.Tensor:
    """Device `float32[2**bits]` Lloyd-Max centroid lookup table cached on `device`."""
    if bits not in LM_LEVELS:
        raise ValueError(f"Unsupported LM bits {bits}")
    dev = torch.device(device)
    key = (int(bits), str(dev))
    t = _LM_LUT_CACHE.get(key)
    if t is None:
        t = torch.tensor(LM_LEVELS[bits], dtype=torch.float32, device=dev)
        _LM_LUT_CACHE[key] = t
    return t


def lm_mids_tensor(bits: int, device: torch.device | str) -> torch.Tensor:
    """Device `float32[2**bits - 1]` Lloyd-Max midpoint thresholds cached on `device`."""
    if bits not in LM_MIDS:
        raise ValueError(f"Unsupported LM bits {bits}")
    dev = torch.device(device)
    key = (int(bits), str(dev))
    t = _LM_MID_CACHE.get(key)
    if t is None:
        t = torch.tensor(LM_MIDS[bits], dtype=torch.float32, device=dev)
        _LM_MID_CACHE[key] = t
    return t


@triton.jit
def _fwht(y, HD: tl.constexpr):
    """Normalized 1D Pease Fast Walsh-Hadamard Transform (`H_HD @ y / sqrt(HD)`)."""
    y = y.to(tl.float32)
    STEPS: tl.constexpr = (
        1 if HD == 2 else (
        2 if HD == 4 else (
        3 if HD == 8 else (
        4 if HD == 16 else (
        5 if HD == 32 else (
        6 if HD == 64 else (
        7 if HD == 128 else 8))))))
    )
    for _ in tl.static_range(STEPS):
        y2 = tl.reshape(y, (HD // 2, 2))
        a, b = tl.split(y2)
        y = tl.reshape(tl.trans(tl.join(a + b, a - b)), (HD,))
    return y * tl.rsqrt(HD * 1.0)


@triton.jit
def _fwht_2d(y, M: tl.constexpr, HD: tl.constexpr):
    """Row-wise normalized Pease FWHT on `[M, HD]` in registers."""
    if M == 1:
        return tl.reshape(_fwht(tl.reshape(y, (HD,)), HD), (1, HD))
    y = y.to(tl.float32)
    STEPS: tl.constexpr = (
        1 if HD == 2 else (
        2 if HD == 4 else (
        3 if HD == 8 else (
        4 if HD == 16 else (
        5 if HD == 32 else (
        6 if HD == 64 else (
        7 if HD == 128 else 8))))))
    )
    for _ in tl.static_range(STEPS):
        y3 = tl.reshape(y, (M, HD // 2, 2))
        a, b = tl.split(y3)
        y = tl.reshape(tl.permute(tl.join(a + b, a - b), (0, 2, 1)), (M, HD))
    return y * tl.rsqrt(HD * 1.0)


@triton.jit
def _rht_vec_kernel(X, S, Y, stride_x, stride_y, HD: tl.constexpr, OUT: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, HD)
    x = tl.load(X + r * stride_x + d).to(tl.float32)
    s = tl.load(S + d).to(tl.float32)
    if OUT:
        y = _fwht(x, HD) * s
    else:
        y = _fwht(x * s, HD)
    tl.store(Y + r * stride_y + d, y)


def _rht_apply(x: torch.Tensor, signs: torch.Tensor | None, inv: bool) -> torch.Tensor:
    shape = x.shape
    hd = int(shape[-1])
    if hd < 2 or (hd & (hd - 1)) != 0 or hd > 256:
        raise ValueError(f"Pease RHT supports power-of-2 head_dim in [2, 256], got {hd}")
    s = default_rht_signs(hd, x.device) if signs is None else signs.float().contiguous()
    if s.shape != (hd,):
        raise ValueError(f"Expected rht_signs shape ({hd},), got {tuple(s.shape)}")
    x2 = x.reshape(-1, hd).float().contiguous()
    out = torch.empty_like(x2)
    with torch.cuda.device(x.device):
        _rht_vec_kernel[(x2.shape[0],)](x2, s, out, x2.stride(0), out.stride(0), HD=hd, OUT=inv)
    return out.reshape(shape)


def rht_fwd(x: torch.Tensor, signs: torch.Tensor | None = None) -> torch.Tensor:
    """Forward deterministic RHT: `_fwht(x * signs)` in float32."""
    return _rht_apply(x, signs, inv=False)


def rht_inv(y: torch.Tensor, signs: torch.Tensor | None = None) -> torch.Tensor:
    """Inverse deterministic RHT: `_fwht(y) * signs` in float32 (exact involution)."""
    return _rht_apply(y, signs, inv=True)


@triton.jit
def _quantize_lm(y_rot, HD: tl.constexpr, BITS: tl.constexpr):
    """Scale by head RMS (FP16) and quantize normalized rotated vector to exact Lloyd-Max indices."""
    rms = tl.sqrt(tl.sum(y_rot * y_rot, axis=0) / HD)
    sc_h = tl.maximum(rms, 1e-6).to(tl.float16)
    inv_sc = 1.0 / sc_h.to(tl.float32)
    x = y_rot * inv_sc
    if BITS == 2:
        hi = x >= tl.full((), 0.0, tl.float32)
        thr = tl.where(hi, tl.full((), 0.9815988, tl.float32), tl.full((), -0.9815988, tl.float32))
        q_idx = tl.where(hi, tl.where(x >= thr, 3, 2), tl.where(x >= thr, 1, 0))
    elif BITS == 3:
        c0 = x >= tl.full((), 0.0, tl.float32)
        c1 = x >= tl.where(c0, tl.full((), 1.0499573, tl.float32), tl.full((), -1.0499573, tl.float32))
        t_pos = tl.where(c1, tl.full((), 1.7479275, tl.float32), tl.full((), 0.5005497, tl.float32))
        t_neg = tl.where(c1, tl.full((), -0.5005497, tl.float32), tl.full((), -1.7479275, tl.float32))
        c2 = x >= tl.where(c0, t_pos, t_neg)
        q_idx = (c0.to(tl.int32) << 2) | (c1.to(tl.int32) << 1) | c2.to(tl.int32)
    else:  # BITS == 4
        b3 = x >= tl.full((), 0.0, tl.float32)
        b2 = x >= tl.where(b3, tl.full((), 1.0992858, tl.float32), tl.full((), -1.0992858, tl.float32))
        b1_pos = tl.where(b2, tl.full((), 1.8435318, tl.float32), tl.full((), 0.5224037, tl.float32))
        b1_neg = tl.where(b2, tl.full((), -0.5224037, tl.float32), tl.full((), -1.8435318, tl.float32))
        b1 = x >= tl.where(b3, b1_pos, b1_neg)
        t0_pos = tl.where(
            b2,
            tl.where(b1, tl.full((), 2.4008034, tl.float32), tl.full((), 1.4371388, tl.float32)),
            tl.where(b1, tl.full((), 0.7995498, tl.float32), tl.full((), 0.2582217, tl.float32)),
        )
        t0_neg = tl.where(
            b2,
            tl.where(b1, tl.full((), -0.2582217, tl.float32), tl.full((), -0.7995498, tl.float32)),
            tl.where(b1, tl.full((), -1.4371388, tl.float32), tl.full((), -2.4008034, tl.float32)),
        )
        b0 = x >= tl.where(b3, t0_pos, t0_neg)
        q_idx = (b3.to(tl.int32) << 3) | (b2.to(tl.int32) << 2) | (b1.to(tl.int32) << 1) | b0.to(tl.int32)
    return q_idx, sc_h


@triton.jit
def _store_lm_vec(
    BUF, SC, BUF_HOST, SC_HOST,
    head, slot_pos, in_host, q_idx, sc_h,
    stride_b_h, stride_b_t, stride_s_h, stride_s_t,
    stride_hb_h, stride_hb_t, stride_hs_h, stride_hs_t,
    VRAM_CELLS,
    HD: tl.constexpr, BITS: tl.constexpr, HAS_HOST: tl.constexpr,
):
    """Bit-pack `q_idx` (`BITS` in 2/3/4) and store codes + FP16 RMS scale to VRAM or pinned host UVA."""
    if HAS_HOST and in_host:
        hp = slot_pos - VRAM_CELLS
        b_base = BUF_HOST + head * stride_hb_h + hp * stride_hb_t
        s_ptr = SC_HOST + head * stride_hs_h + hp * stride_hs_t
    else:
        b_base = BUF + head * stride_b_h + slot_pos * stride_b_t
        s_ptr = SC + head * stride_s_h + slot_pos * stride_s_t

    tl.store(s_ptr, sc_h)
    if BITS == 4:
        q_lo, q_hi = tl.split(tl.reshape(q_idx.to(tl.uint8), (HD // 2, 2)))
        packed = (q_lo | (q_hi << 4)).to(tl.uint8)
        tl.store(b_base + tl.arange(0, HD // 2), packed)
    elif BITS == 2:
        q_e, q_o = tl.split(tl.reshape(q_idx.to(tl.uint8), (HD // 2, 2)))
        pairs = (q_e | (q_o << 2)).to(tl.uint8)
        p_lo, p_hi = tl.split(tl.reshape(pairs, (HD // 4, 2)))
        packed = (p_lo | (p_hi << 4)).to(tl.uint8)
        tl.store(b_base + tl.arange(0, HD // 4), packed)
    else:  # BITS == 3: 3 bit-planes of HD // 8 bytes each
        n_oct: tl.constexpr = HD // 8
        d_oct = tl.arange(0, n_oct)
        q_g = tl.reshape(q_idx, (n_oct, 8))
        bw = (1 << tl.arange(0, 8))[None, :]
        p0 = tl.sum((q_g & 1) * bw, axis=1).to(tl.uint8)
        p1 = tl.sum(((q_g >> 1) & 1) * bw, axis=1).to(tl.uint8)
        p2 = tl.sum(((q_g >> 2) & 1) * bw, axis=1).to(tl.uint8)
        tl.store(b_base + d_oct, p0)
        tl.store(b_base + n_oct + d_oct, p1)
        tl.store(b_base + 2 * n_oct + d_oct, p2)


@triton.jit
def _kv_conv_write_lm_kernel(
    KR, VR, PK, PV, WK, WV, DK, DV, RHT_S,
    K_BUF, K_SC, K_HOST, K_SC_HOST,
    V_BUF, V_SC, V_HOST, V_SC_HOST,
    POS, T, C, thr, eps,
    stride_kb_h, stride_kb_t, stride_ks_h, stride_ks_t,
    stride_hkb_h, stride_hkb_t, stride_hks_h, stride_hks_t,
    stride_vb_h, stride_vb_t, stride_vs_h, stride_vs_t,
    stride_hvb_h, stride_hvb_t, stride_hvs_h, stride_hvs_t,
    KV_CAP, VRAM_CELLS,
    DYNAMIC: tl.constexpr, HD: tl.constexpr,
    K_BITS: tl.constexpr, V_BITS: tl.constexpr,
    RING: tl.constexpr, HAS_HOST: tl.constexpr,
):
    """Per (head, K-or-V) program: conv + clip/norm + RHT + Lloyd-Max quantize + separate K/V store."""
    head = tl.program_id(0)
    is_v = tl.program_id(1)
    thr = thr.to(tl.float32)
    eps = eps.to(tl.float32)
    d = tl.arange(0, HD)
    c = head * HD + d
    rht_s = tl.load(RHT_S + d).to(tl.float32)
    X = VR if is_v == 1 else KR
    PREV = PV if is_v == 1 else PK
    W = WV if is_v == 1 else WK
    D = DV if is_v == 1 else DK
    p0 = tl.load(PREV + c)
    p1 = tl.load(PREV + C + c)
    p2 = tl.load(PREV + 2 * C + c)
    if DYNAMIC:
        s0 = tl.load(W + c * 4 + 0)
        s1 = tl.load(W + c * 4 + 1)
        s2 = tl.load(W + c * 4 + 2)
        s3 = tl.load(W + c * 4 + 3)
    else:
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
        y_rot = _fwht(y * rht_s, HD)
        raw_pos = tl.load(POS + t).to(tl.int64)
        slot_pos = (raw_pos % KV_CAP) if RING else raw_pos
        in_host = HAS_HOST and (slot_pos >= VRAM_CELLS)
        if is_v == 1:
            q_idx, sc_h = _quantize_lm(y_rot, HD=HD, BITS=V_BITS)
            _store_lm_vec(
                V_BUF, V_SC, V_HOST, V_SC_HOST,
                head, slot_pos, in_host, q_idx, sc_h,
                stride_vb_h, stride_vb_t, stride_vs_h, stride_vs_t,
                stride_hvb_h, stride_hvb_t, stride_hvs_h, stride_hvs_t,
                VRAM_CELLS, HD=HD, BITS=V_BITS, HAS_HOST=HAS_HOST,
            )
        else:
            q_idx, sc_h = _quantize_lm(y_rot, HD=HD, BITS=K_BITS)
            _store_lm_vec(
                K_BUF, K_SC, K_HOST, K_SC_HOST,
                head, slot_pos, in_host, q_idx, sc_h,
                stride_kb_h, stride_kb_t, stride_ks_h, stride_ks_t,
                stride_hkb_h, stride_hkb_t, stride_hks_h, stride_hks_t,
                VRAM_CELLS, HD=HD, BITS=K_BITS, HAS_HOST=HAS_HOST,
            )
        p0 = p1
        p1 = p2
        p2 = x
    tl.store(PREV + c, p0)
    tl.store(PREV + C + c, p1)
    tl.store(PREV + 2 * C + c, p2)


def kv_conv_write_lm_(
    k_raw, v_raw, prev_k, prev_v,
    k_buf, k_sc, v_buf, v_sc,
    pos, thr: float, eps: float, n_kv: int,
    w_k=None, w_v=None, dyn=None,
    *,
    k_bits: int = 4,
    v_bits: int = 4,
    rht_signs=None,
    ring: bool = False,
    k_host=None,
    k_sc_host=None,
    v_host=None,
    v_sc_host=None,
    vram_cells: int | None = None,
):
    """Conv + clip/norm + RHT + Hadamard Lloyd-Max quantize + separate K/V cache write.

    Supports asymmetric `k_bits`, `v_bits` in `(2, 3, 4)` and tiered VRAM + pinned host-RAM UVA.
    """
    T, C = k_raw.shape
    HD = C // n_kv
    if k_bits not in (2, 3, 4) or v_bits not in (2, 3, 4):
        raise ValueError(f"k_bits and v_bits must be in (2, 3, 4), got ({k_bits}, {v_bits})")
    if dyn is not None:
        k_in, v_in, (st_k, pr_k), (st_v, pr_v) = dyn
        dk = (k_in.float() @ pr_k.T).contiguous()
        dv = (v_in.float() @ pr_v.T).contiguous()
        wk, wv = st_k.reshape(C, 4), st_v.reshape(C, 4)
    else:
        dk = dv = k_raw
        wk, wv = w_k, w_v
    signs = default_rht_signs(HD, k_raw.device) if rht_signs is None else rht_signs.float().contiguous()
    has_any_host = any(x is not None for x in (k_host, k_sc_host, v_host, v_sc_host))
    has_host = has_any_host and vram_cells is not None
    if has_any_host and not has_host:
        raise ValueError("vram_cells is required when host KV buffers are provided")
    if has_host and any(x is None for x in (k_host, k_sc_host, v_host, v_sc_host)):
        raise ValueError("k_host, k_sc_host, v_host, and v_sc_host must all be provided together")
    hk_buf = k_host if has_host else k_buf
    hk_sc = k_sc_host if has_host else k_sc
    hv_buf = v_host if has_host else v_buf
    hv_sc = v_sc_host if has_host else v_sc
    kv_cap = int(vram_cells + k_host.shape[1]) if has_host else int(k_buf.shape[1])
    v_cells = int(vram_cells) if has_host else kv_cap
    with torch.cuda.device(k_raw.device):
        _kv_conv_write_lm_kernel[(n_kv, 2)](
            k_raw.contiguous(), v_raw.contiguous(), prev_k, prev_v, wk, wv, dk, dv, signs,
            k_buf, k_sc, hk_buf, hk_sc,
            v_buf, v_sc, hv_buf, hv_sc,
            pos, T, C, thr, eps,
            k_buf.stride(0), k_buf.stride(1), k_sc.stride(0), k_sc.stride(1),
            hk_buf.stride(0), hk_buf.stride(1), hk_sc.stride(0), hk_sc.stride(1),
            v_buf.stride(0), v_buf.stride(1), v_sc.stride(0), v_sc.stride(1),
            hv_buf.stride(0), hv_buf.stride(1), hv_sc.stride(0), hv_sc.stride(1),
            kv_cap, v_cells,
            DYNAMIC=dyn is not None, HD=HD,
            K_BITS=k_bits, V_BITS=v_bits,
            RING=ring, HAS_HOST=has_host,
        )


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
