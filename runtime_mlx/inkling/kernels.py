"""Half-warp Apple Silicon Metal kernels + pure-MLX RHT for packed Inkling-2-Mini.

Supports ALL formats in our 9.5 GB Inkling-2-Mini build at full M5 Max memory bandwidth:
  - Row-wise 4 KB SRAM Hash Trellis: `ht2` (K=512 g=4), `ht3` (K=1024 g=2)
  - Row-wise 8 KB SRAM Scalar Trellis: `rtr1..rtr4` (K=4096 g=1)
  - Packed Scalar Lloyd-Max (256B-1KB SRAM LUT): `lm6` (6-bit packed), `lm8` (8-bit)
  - Pure-MLX Sylvester/Paley Randomized Hadamard Transform (zero PyTorch dependency)
"""
from __future__ import annotations

import math
import os
import re
from functools import lru_cache

import mlx.core as mx
import numpy as np

HASH_MULT_U32 = 0x9E3779B1
HASH_MULT_HEX = "0x9e3779b1u"


# ----------------------------------------------------------------------------- Pure-MLX Hadamard / RHT

def _is_prime(q: int) -> bool:
    return q > 1 and all(q % d for d in range(2, math.isqrt(q) + 1))


def _sylvester_np(a: int) -> np.ndarray:
    H = np.ones((1, 1), dtype=np.float64)
    while H.shape[0] < (1 << a):
        H = np.block([[H, H], [H, -H]])
    return H


def _paley_np(q: int) -> np.ndarray:
    squares = {(x * x) % q for x in range(1, q)}
    chi = np.array([0.0] + [1.0 if a in squares else -1.0 for a in range(1, q)], dtype=np.float64)
    idx = (np.arange(q)[None, :] - np.arange(q)[:, None]) % q
    Q = chi[idx]
    S = np.zeros((q + 1, q + 1), dtype=np.float64)
    S[0, 1:] = 1.0
    S[1:, 0] = -1.0
    S[1:, 1:] = Q
    return S + np.eye(q + 1, dtype=np.float64)


@lru_cache(maxsize=32)
def hadamard_factors_mx(n: int) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Cached MLX float32 factors (A, A_T, B, B_T, 1/sqrt(n)) matching opgraph.codes.hadamard_factors(n)."""
    for a in range(int(math.log2(n)), -1, -1):
        if n % (1 << a):
            continue
        p = n // (1 << a)
        if p in (1, 2) or (_is_prime(p - 1) and (p - 1) % 4 == 3):
            if p == 1 and a > 1:
                A = _sylvester_np(a // 2)
                B = _sylvester_np(a - a // 2)
            else:
                A = _sylvester_np(a)
                B = (
                    np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
                    if p == 2
                    else (np.ones((1, 1), dtype=np.float64) if p == 1 else _paley_np(p - 1))
                )
            Am = mx.array(A.astype(np.float32))
            At = mx.array(A.T.astype(np.float32))
            Bm = mx.array(B.astype(np.float32))
            Bt = mx.array(B.T.astype(np.float32))
            inv_s = mx.array(1.0 / math.sqrt(n), dtype=mx.float32)
            return Am, At, Bm, Bt, inv_s
    raise ValueError(f"No Hadamard construction for n={n}")


@lru_cache(maxsize=256)
def _meta_u32(*vals: int) -> mx.array:
    return mx.array(list(vals), dtype=mx.uint32)


@lru_cache(maxsize=1)
def _hw_sg() -> int:
    """Simdgroups per threadgroup for the half-warp kernels.

    M1-M3 GPUs cap these pipelines below 512 threads (M1 Pro reports 384), so they get 8 simdgroups.
    """
    env = os.environ.get("LAMQ_HW_SG")
    if env:
        return int(env)
    m = re.match(r"applegpu_g(\d+)", str(mx.device_info().get("architecture", "")).lower())
    return 16 if m and int(m.group(1)) >= 16 else 8


@mx.compile
def _rht_in_compiled(x: mx.array, signs_f32: mx.array, A: mx.array, Bt: mx.array, inv_s: mx.array) -> mx.array:
    xf = x.astype(mx.float32) * signs_f32
    y = mx.matmul(mx.matmul(A, xf.reshape(-1, A.shape[0], Bt.shape[0])), Bt)
    return (y * inv_s).reshape(x.shape)


@mx.compile
def _rht_out_compiled(y: mx.array, signs_out_f32: mx.array, At: mx.array, B: mx.array, inv_s: mx.array) -> mx.array:
    yf = y.astype(mx.float32).reshape(-1, At.shape[0], B.shape[0])
    z = mx.matmul(mx.matmul(At, yf), B).reshape(y.shape)
    return z * (signs_out_f32 * inv_s)


def rht_apply_in(x: mx.array, signs_f32: mx.array) -> mx.array:
    """Compute `x @ R_in.T` where `R_in = H * signs[None, :] / sqrt(n)`. Matches `pack_dense.back`."""
    n = x.shape[-1]
    A, _, _, Bt, inv_s = hadamard_factors_mx(n)
    return _rht_in_compiled(x, signs_f32, A, Bt, inv_s)


def rht_apply_out(y: mx.array, signs_out_f32: mx.array) -> mx.array:
    """Compute `y @ R_out` (for out_rot=True linears: W = R_out.T @ Q @ R_in => x @ W.T = ((x @ R_in.T) @ Q.T) @ R_out)."""
    m = y.shape[-1]
    _, At, B, _, inv_s = hadamard_factors_mx(m)
    return _rht_out_compiled(y, signs_out_f32, At, B, inv_s)


# ----------------------------------------------------------------------------- Half-Warp Row-Trellis Metal Kernel

@lru_cache(maxsize=64)
def make_halfwarp_trellis_kernel(
    k: int,
    n: int,
    g: int,
    lut_entries: int,
    hash_mult: int,
    sg: int = 16,
    rps: int = 4,
):
    """Unified 16-lane-per-row Metal kernel for `ht1..ht3` (4 KB SRAM) and `rtr1..rtr4` (8 KB SRAM).

    Supports both:
      - Fused top-K MoE experts (`n_slots = 1..8`, `expert_ids[slot]` selects expert slab)
      - Dense matrix matvec / batched prefill (`n_slots = T`, `expert_ids = zeros(T)`, `shared_x = 0`)
    """
    ch = 16 if (n % 256 == 0 and 16 % g == 0) else 8
    assert n % (16 * ch) == 0, f"n={n} not divisible by 16*ch={16*ch}"
    assert ch % g == 0, f"ch={ch} not divisible by g={g}"
    iters = n // (16 * ch)
    wpr = (n * k) // 32
    mod32 = (ch * k) % 32
    need_off = mod32 != 0
    span = (ch - g) * k + 12 + (16 if mod32 == 16 else (31 if need_off else 0))
    nw = (span + 31) // 32
    slots = 1 if g == 1 else (2 if g == 2 else 4)
    vec = "half" if slots == 1 else f"half{slots}"
    log2_K = int(round(math.log2(lut_entries)))
    shift = 32 - log2_K
    use_hash = hash_mult != 0

    header = f"#define RPS {rps}u\n#define SG {sg}u\n"
    L = []
    a = L.append
    a("const uint tg = threadgroup_position_in_grid.x;")
    a("const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;")
    a("const uint hw = lane >> 4, hl = lane & 15u;")
    a("const uint tid = simd * 32u + lane;")
    a(f"const uint iters = {iters}u;")
    a(f"const uint wpr = {wpr}u;")
    a("const uint m_rows = meta[0];")
    a("const uint n_slots = meta[1];")
    a("const uint x_div = meta[2];")
    a("const uint ntg = meta[3];")
    a("bool any_valid = false;")
    a("for (uint s = 0; s < n_slots; s++) {")
    a("    if (expert_ids[s] != 0xFFFFFFFFu) { any_valid = true; break; }")
    a("}")
    a("if (!any_valid) {")
    a("    const ulong total = (ulong)n_slots * m_rows;")
    a(f"    const ulong stride = (ulong)ntg * {sg * 32}u;")
    a(f"    for (ulong i = (ulong)tg * {sg * 32}u + tid; i < total; i += stride) y[i] = 0.0f;")
    a("    return;")
    a("}")
    a(f"threadgroup {vec} tg_lut[{lut_entries}];")
    if slots == 1:
        a(f"for (uint idx = tid; idx < {lut_entries}u; idx += {sg * 32}u) tg_lut[idx] = lut[idx];")
    elif slots == 2:
        a(f"for (uint idx = tid; idx < {lut_entries}u; idx += {sg * 32}u) tg_lut[idx] = half2(lut[2u * idx], lut[2u * idx + 1u]);")
    else:
        a(f"for (uint idx = tid; idx < {lut_entries}u; idx += {sg * 32}u) tg_lut[idx] = half4(lut[4u * idx], lut[4u * idx + 1u], lut[4u * idx + 2u], lut[4u * idx + 3u]);")
    a("threadgroup_barrier(mem_flags::mem_threadgroup);")

    # Round m_rows up to a multiple of RPS so each half-warp handles one slot cleanly!
    a("const uint blocks_per_slot = (m_rows + RPS - 1u) / RPS;")
    a("const uint total_blocks = n_slots * blocks_per_slot;")
    a("const uint blocks_per_tg = SG * 2u;")
    a("const ulong expert_code_stride = (ulong)m_rows * wpr;")
    a("for (uint rb = tg; rb * blocks_per_tg < total_blocks; rb += ntg) {")
    a("    const uint gb = rb * blocks_per_tg + simd * 2u + hw;")
    a("    if (gb >= total_blocks) continue;")
    a("    const uint slot = gb / blocks_per_slot;")
    a("    const uint r0 = (gb - slot * blocks_per_slot) * RPS;")
    a("    const uint eid = expert_ids[slot];")
    a("    if (eid == 0xFFFFFFFFu) {")
    a("        if (hl == 0u) {")
    a("            #pragma unroll")
    a("            for (uint r = 0; r < RPS; r++) {")
    a("                if (r0 + r < m_rows) y[(ulong)slot * m_rows + r0 + r] = 0.0f;")
    a("            }")
    a("        }")
    a("        continue;")
    a("    }")
    a("    const uint x_row = (x_div == 1u) ? slot : (slot / x_div);")
    a(f"    const device float* xs = x + (ulong)x_row * {n}u;")
    a("    const device uint* ecodes = codes + (ulong)eid * expert_code_stride;")
    a("    const device half* escales = scales + (ulong)eid * m_rows;")
    a("    float acc[RPS];")
    a("    for (uint r = 0; r < RPS; r++) acc[r] = 0.0f;")
    a("    for (uint it = 0; it < iters; it++) {")
    a("        const uint c = hl + 16u * it;")
    a(f"        float xr[{ch}];")
    for i in range(ch):
        a(f"        xr[{i}] = xs[c * {ch}u + {i}u];")
    a(f"        const uint B = ({n}u - {ch}u * (c + 1u)) * {k}u, w0 = B >> 5;")
    if need_off:
        a("        const uint o = B & 31u;")
    a("        #pragma unroll")
    a("        for (uint r = 0; r < RPS; r++) {")
    a("            const uint row = min(r0 + r, m_rows - 1u);")
    a("            const device uint* p = ecodes + (ulong)row * wpr + w0;")
    a(f"            uint w[{nw + 1}];")
    for i in range(nw):
        a(f"            w[{i}] = (w0 + {i}u < wpr) ? p[{i}] : 0u;")
    a(f"            w[{nw}] = 0u;")
    const_off = not need_off
    if mod32 == 16:
        a("            if (o != 0u) {")
        for i in range(nw):
            a(f"                w[{i}] = (w[{i}] >> 16u) | (w[{i + 1}] << 16u);")
        a("            }")
        const_off = True
    a("            float a_val = 0.0f;")
    for i in range(0, ch, g):
        last = i + g - 1
        bit = (ch - 1 - last) * k
        if not const_off:
            a(f"            {{ const uint bit = o + {bit}u, wi = bit >> 5, off = bit & 31u;")
            a(f"              const uint s = ((w[wi] >> off) | (off == 0u ? 0u : (w[wi + 1] << (32u - off)))) & 4095u;")
        else:
            wi, off = bit >> 5, bit & 31
            if off + 12 <= 32:
                a(f"            {{ const uint s = extract_bits(w[{wi}], {off}u, 12u);")
            else:
                lo = 32 - off
                a(f"            {{ const uint s = insert_bits(extract_bits(w[{wi}], {off}u, {lo}u), w[{wi + 1}], {lo}u, {12 - lo}u);")
        if use_hash:
            a(f"              const uint idx = (s * {HASH_MULT_HEX}) >> {shift}u;")
        else:
            a("              const uint idx = s;")
        if slots == 1:
            a(f"              a_val += (float)tg_lut[idx] * xr[{i}]; }}")
        else:
            a(f"              const {vec} v = tg_lut[idx];")
            terms = " + ".join(f"(float)v[{t}] * xr[{i + t}]" for t in range(g))
            a(f"              a_val += {terms}; }}")
    a("            acc[r] += a_val;")
    a("        }")
    a("    }")
    a("    #pragma unroll")
    a("    for (uint r = 0; r < RPS; r++) {")
    a("        float s = acc[r];")
    a("        s += simd_shuffle_xor(s, 8u);")
    a("        s += simd_shuffle_xor(s, 4u);")
    a("        s += simd_shuffle_xor(s, 2u);")
    a("        s += simd_shuffle_xor(s, 1u);")
    a("        if (hl == 0u && r0 + r < m_rows) y[(ulong)slot * m_rows + r0 + r] = s * (float)escales[r0 + r];")
    a("    }")
    a("}")
    return mx.fast.metal_kernel(
        name=f"hw_tr_k{k}_n{n}_g{g}_E{lut_entries}_h{int(use_hash)}",
        input_names=["x", "codes", "scales", "lut", "expert_ids", "meta"],
        output_names=["y"],
        header=header,
        source="\n".join(L),
    )


def trellis_matvec(
    x_rot: mx.array,
    codes: mx.array,
    scales_f16: mx.array,
    lut_f16: mx.array,
    expert_ids: mx.array,
    m: int,
    n: int,
    k: int,
    g: int,
    lut_entries: int,
    hash_mult: int,
    shared_x: bool = False,
    tg_count: int = 80,
    x_div: int | None = None,
) -> mx.array:
    """Execute half-warp trellis matvec; returns float32[n_slots, m]."""
    sg = _hw_sg()
    n_slots = int(expert_ids.shape[0])
    x_div_val = int(x_div) if x_div is not None else (n_slots if shared_x else 1)
    blocks_per_slot = (m + 3) // 4
    total_blocks = n_slots * blocks_per_slot
    ntg = max(1, min(tg_count, (total_blocks + 2 * sg - 1) // (2 * sg)))
    kern = make_halfwarp_trellis_kernel(k, n, g, lut_entries, hash_mult, sg=sg)
    meta = _meta_u32(m, n_slots, x_div_val, ntg)
    out = kern(
        inputs=[x_rot.reshape(-1), codes.reshape(-1), scales_f16.reshape(-1), lut_f16.reshape(-1), expert_ids, meta],
        grid=(ntg * sg * 32, 1, 1),
        threadgroup=(sg * 32, 1, 1),
        output_shapes=[(n_slots * m,)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(n_slots, m)


@lru_cache(maxsize=128)
def make_halfwarp_trellis_dense_bt_kernel(
    k: int,
    n: int,
    g: int,
    lut_entries: int,
    hash_mult: int,
    t_blk: int,
    sg: int = 16,
):
    """Multi-token batched dense half-warp trellis kernel: reads each packed row ONCE for all `t_blk` tokens!

    Eliminates 8x redundant weight DRAM traffic and 8x redundant bit-unpacking/LUT lookups on
    dense attention linears & `lm_head` during DSpark block-8 verify and prefill.
    """
    ch = 16 if (n % 256 == 0 and 16 % g == 0) else 8
    assert n % (16 * ch) == 0 and ch % g == 0
    iters = n // (16 * ch)
    wpr = (n * k) // 32
    mod32 = (ch * k) % 32
    need_off = mod32 != 0
    span = (ch - g) * k + 12 + (16 if mod32 == 16 else (31 if need_off else 0))
    nw = (span + 31) // 32
    slots = 1 if g == 1 else (2 if g == 2 else 4)
    vec = "half" if slots == 1 else f"half{slots}"
    log2_K = int(round(math.log2(lut_entries)))
    shift = 32 - log2_K
    use_hash = hash_mult != 0

    header = f"#define SG {sg}u\n#define TBLK {t_blk}u\n"
    L = []
    a = L.append
    a("const uint tg = threadgroup_position_in_grid.x;")
    a("const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;")
    a("const uint hw = lane >> 4, hl = lane & 15u;")
    a("const uint tid = simd * 32u + lane;")
    a(f"const uint iters = {iters}u;")
    a(f"const uint wpr = {wpr}u;")
    a("const uint m_rows = meta[0];")
    a("const uint ntg = meta[1];")
    a(f"threadgroup {vec} tg_lut[{lut_entries}];")
    if slots == 1:
        a(f"for (uint idx = tid; idx < {lut_entries}u; idx += {sg * 32}u) tg_lut[idx] = lut[idx];")
    elif slots == 2:
        a(f"for (uint idx = tid; idx < {lut_entries}u; idx += {sg * 32}u) tg_lut[idx] = half2(lut[2u * idx], lut[2u * idx + 1u]);")
    else:
        a(f"for (uint idx = tid; idx < {lut_entries}u; idx += {sg * 32}u) tg_lut[idx] = half4(lut[4u * idx], lut[4u * idx + 1u], lut[4u * idx + 2u], lut[4u * idx + 3u]);")
    a("threadgroup_barrier(mem_flags::mem_threadgroup);")

    # 1 row per half-warp -> 32 rows per threadgroup, zero register spill across TBLK tokens!
    a("const uint rows_per_tg = SG * 2u;")
    a("for (uint rb = tg; rb * rows_per_tg < m_rows; rb += ntg) {")
    a("    const uint row = rb * rows_per_tg + simd * 2u + hw;")
    a("    if (row >= m_rows) continue;")
    a("    float acc[TBLK];")
    a("    #pragma unroll")
    a("    for (uint t = 0; t < TBLK; t++) acc[t] = 0.0f;")
    a("    const device uint* row_codes = codes + (ulong)row * wpr;")
    a("    for (uint it = 0; it < iters; it++) {")
    a("        const uint c = hl + 16u * it;")
    a(f"        const uint col0 = c * {ch}u;")
    a(f"        const uint B = ({n}u - {ch}u * (c + 1u)) * {k}u, w0 = B >> 5;")
    if need_off:
        a("        const uint o = B & 31u;")
    a("        const device uint* p = row_codes + w0;")
    a(f"        uint w[{nw + 1}];")
    for i in range(nw):
        a(f"        w[{i}] = (w0 + {i}u < wpr) ? p[{i}] : 0u;")
    a(f"        w[{nw}] = 0u;")
    const_off = not need_off
    if mod32 == 16:
        a("        if (o != 0u) {")
        for i in range(nw):
            a(f"            w[{i}] = (w[{i}] >> 16u) | (w[{i + 1}] << 16u);")
        a("        }")
        const_off = True
    for i in range(0, ch, g):
        last = i + g - 1
        bit = (ch - 1 - last) * k
        if not const_off:
            a(f"        {{ const uint bit = o + {bit}u, wi = bit >> 5, off = bit & 31u;")
            a(f"          const uint s = ((w[wi] >> off) | (off == 0u ? 0u : (w[wi + 1] << (32u - off)))) & 4095u;")
        else:
            wi, off = bit >> 5, bit & 31
            if off + 12 <= 32:
                a(f"        {{ const uint s = extract_bits(w[{wi}], {off}u, 12u);")
            else:
                lo = 32 - off
                a(f"        {{ const uint s = insert_bits(extract_bits(w[{wi}], {off}u, {lo}u), w[{wi + 1}], {lo}u, {12 - lo}u);")
        if use_hash:
            a(f"          const uint idx = (s * {HASH_MULT_HEX}) >> {shift}u;")
        else:
            a("          const uint idx = s;")
        if slots == 1:
            a("          const float v0 = (float)tg_lut[idx];")
            a("          #pragma unroll")
            a(f"          for (uint t = 0; t < TBLK; t++) acc[t] += v0 * x[(ulong)t * {n}u + col0 + {i}u]; }}")
        else:
            a(f"          const {vec} v = tg_lut[idx];")
            for u in range(g):
                a(f"          const float vf{u} = (float)v[{u}];")
            a("          #pragma unroll")
            terms = " + ".join(f"vf{u} * x[(ulong)t * {n}u + col0 + {i + u}u]" for u in range(g))
            a(f"          for (uint t = 0; t < TBLK; t++) acc[t] += {terms}; }}")
    a("    }")
    a("    const float sc = (float)scales[row];")
    a("    #pragma unroll")
    a("    for (uint t = 0; t < TBLK; t++) {")
    a("        float s = acc[t];")
    a("        s += simd_shuffle_xor(s, 8u);")
    a("        s += simd_shuffle_xor(s, 4u);")
    a("        s += simd_shuffle_xor(s, 2u);")
    a("        s += simd_shuffle_xor(s, 1u);")
    a("        if (hl == 0u) y[(ulong)t * m_rows + row] = s * sc;")
    a("    }")
    a("}")
    return mx.fast.metal_kernel(
        name=f"hw_tr_bt{t_blk}_k{k}_n{n}_g{g}_E{lut_entries}_h{int(use_hash)}",
        input_names=["x", "codes", "scales", "lut", "meta"],
        output_names=["y"],
        header=header,
        source="\n".join(L),
    )


def trellis_dense_matmul(
    x_rot: mx.array,
    codes: mx.array,
    scales_f16: mx.array,
    lut_f16: mx.array,
    m: int,
    n: int,
    k: int,
    g: int,
    lut_entries: int,
    hash_mult: int,
    tg_count: int = 80,
) -> mx.array:
    """Dense matrix multiply x_rot[T, n] @ W.T -> [T, m] reading packed weights 1x per 8-token block."""
    sg = _hw_sg()
    x2 = x_rot.reshape(-1, n)
    T = int(x2.shape[0])
    if T == 1:
        return trellis_matvec(
            x2, codes, scales_f16, lut_f16, _meta_u32(0),
            m, n, k, g, lut_entries, hash_mult, x_div=1, tg_count=tg_count,
        )
    parts = []
    c_flat = codes.reshape(-1)
    s_flat = scales_f16.reshape(-1)
    l_flat = lut_f16.reshape(-1)
    ntg = max(1, min(tg_count, (m + 2 * sg - 1) // (2 * sg)))
    meta = _meta_u32(m, ntg)
    for t0 in range(0, T, 8):
        tb = min(8, T - t0)
        if tb == 1:
            parts.append(trellis_matvec(
                x2[t0 : t0 + 1], codes, scales_f16, lut_f16, _meta_u32(0),
                m, n, k, g, lut_entries, hash_mult, x_div=1, tg_count=tg_count,
            ))
        else:
            kern = make_halfwarp_trellis_dense_bt_kernel(k, n, g, lut_entries, hash_mult, tb, sg=sg)
            out = kern(
                inputs=[x2[t0 : t0 + tb].reshape(-1), c_flat, s_flat, l_flat, meta],
                grid=(ntg * sg * 32, 1, 1),
                threadgroup=(sg * 32, 1, 1),
                output_shapes=[(tb * m,)],
                output_dtypes=[mx.float32],
            )[0]
            parts.append(out.reshape(tb, m))
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)


@lru_cache(maxsize=64)
def make_halfwarp_trellis_moe_grouped_kernel(
    k: int,
    n: int,
    g: int,
    lut_entries: int,
    hash_mult: int,
    sg: int = 16,
):
    """Unique-expert grouped multi-token MoE kernel for T=2..8 (zero CPU sync, 2.3x fewer DRAM weight reads!).

    Compacts `expert_ids[0..n_slots-1]` (`n_slots <= 64`, `n_experts <= 128`) in threadgroup SRAM so each
    active expert's packed codes are read & decoded ONCE and multiplied against all `1..8` tokens hitting it.
    """
    ch = 16 if (n % 256 == 0 and 16 % g == 0) else 8
    assert n % (16 * ch) == 0 and ch % g == 0
    iters = n // (16 * ch)
    wpr = (n * k) // 32
    mod32 = (ch * k) % 32
    need_off = mod32 != 0
    span = (ch - g) * k + 12 + (16 if mod32 == 16 else (31 if need_off else 0))
    nw = (span + 31) // 32
    slots = 1 if g == 1 else (2 if g == 2 else 4)
    vec = "half" if slots == 1 else f"half{slots}"
    log2_K = int(round(math.log2(lut_entries)))
    shift = 32 - log2_K
    use_hash = hash_mult != 0

    header = f"#define SG {sg}u\n"
    L = []
    a = L.append
    a("const uint tg = threadgroup_position_in_grid.x;")
    a("const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;")
    a("const uint hw = lane >> 4, hl = lane & 15u;")
    a("const uint tid = simd * 32u + lane;")
    a(f"const uint iters = {iters}u;")
    a(f"const uint wpr = {wpr}u;")
    a("const uint m_rows = meta[0];")
    a("const uint n_slots = meta[1];")
    a("const uint x_div = meta[2];")
    a("const uint ntg = meta[3];")
    a("const uint n_experts = meta[4];")

    # Compact active unique experts into <=4-token groups in threadgroup SRAM (100% deterministic across all threadgroups!)
    a("threadgroup uint tg_eids[64];")
    a("threadgroup uint tg_e_cnt[128];")
    a("threadgroup uint tg_e_p0[128];")
    a("threadgroup uint tg_e_p1[128];")
    a("threadgroup uint tg_u_eid[128];")
    a("threadgroup uint tg_u_slots[128];")  # packs 4 uint8 slot indices in 1 uint32!
    a("threadgroup uint tg_u_cnt[128];")
    a("threadgroup uint tg_n_active;")
    a("if (tid < n_slots) tg_eids[tid] = expert_ids[tid];")
    a("threadgroup_barrier(mem_flags::mem_threadgroup);")

    # Zero outputs for inactive (0xFFFFFFFFu) slots in coalesced stride
    a("for (uint s = 0; s < n_slots; s++) {")
    a("    if (tg_eids[s] == 0xFFFFFFFFu) {")
    a(f"        for (uint r = tg * {sg * 32}u + tid; r < m_rows; r += ntg * {sg * 32}u) y[(ulong)s * m_rows + r] = 0.0f;")
    a("    }")
    a("}")

    # Step 1 (parallel across tid < n_experts): collect up to 8 matching slots per expert
    a("if (tid < n_experts) {")
    a("    uint p0 = 0u, p1 = 0u, cnt = 0u;")
    a("    for (uint s = 0; s < n_slots; s++) {")
    a("        if (tg_eids[s] == tid && cnt < 8u) {")
    a("            if (cnt < 4u) p0 |= (s << (cnt * 8u));")
    a("            else p1 |= (s << ((cnt - 4u) * 8u));")
    a("            cnt++;")
    a("        }")
    a("    }")
    a("    tg_e_cnt[tid] = cnt; tg_e_p0[tid] = p0; tg_e_p1[tid] = p1;")
    a("}")
    a("threadgroup_barrier(mem_flags::mem_threadgroup);")
    # Step 2 (deterministic single-thread compaction in ascending expert order so EVERY threadgroup agrees!)
    a("if (tid == 0u) {")
    a("    uint n_act = 0u;")
    a("    for (uint e = 0; e < n_experts; e++) {")
    a("        uint c = tg_e_cnt[e];")
    a("        if (c == 0u) continue;")
    a("        if (c <= 4u) {")
    a("            tg_u_eid[n_act] = e; tg_u_slots[n_act] = tg_e_p0[e]; tg_u_cnt[n_act] = c; n_act++;")
    a("        } else {")
    a("            tg_u_eid[n_act] = e; tg_u_slots[n_act] = tg_e_p0[e]; tg_u_cnt[n_act] = 4u; n_act++;")
    a("            tg_u_eid[n_act] = e; tg_u_slots[n_act] = tg_e_p1[e]; tg_u_cnt[n_act] = c - 4u; n_act++;")
    a("        }")
    a("    }")
    a("    tg_n_active = n_act;")
    a("}")
    a("threadgroup_barrier(mem_flags::mem_threadgroup);")
    a("const uint n_active = tg_n_active;")
    a("if (n_active == 0u) return;")

    a(f"threadgroup {vec} tg_lut[{lut_entries}];")
    if slots == 1:
        a(f"for (uint idx = tid; idx < {lut_entries}u; idx += {sg * 32}u) tg_lut[idx] = lut[idx];")
    elif slots == 2:
        a(f"for (uint idx = tid; idx < {lut_entries}u; idx += {sg * 32}u) tg_lut[idx] = half2(lut[2u * idx], lut[2u * idx + 1u]);")
    else:
        a(f"for (uint idx = tid; idx < {lut_entries}u; idx += {sg * 32}u) tg_lut[idx] = half4(lut[4u * idx], lut[4u * idx + 1u], lut[4u * idx + 2u], lut[4u * idx + 3u]);")
    a("threadgroup_barrier(mem_flags::mem_threadgroup);")

    a("const uint total_rows = n_active * m_rows;")
    a("const uint rows_per_tg = SG * 2u;")
    a("const ulong expert_code_stride = (ulong)m_rows * wpr;")
    a("for (uint rb = tg; rb * rows_per_tg < total_rows; rb += ntg) {")
    a("    const uint g_row = rb * rows_per_tg + simd * 2u + hw;")
    a("    if (g_row >= total_rows) continue;")
    a("    const uint u = g_row / m_rows;")
    a("    const uint row = g_row - u * m_rows;")
    a("    const uint eid = tg_u_eid[u];")
    a("    const uint cnt = tg_u_cnt[u];")
    a("    const uint packed = tg_u_slots[u];")
    a("    const device uint* row_codes = codes + (ulong)eid * expert_code_stride + (ulong)row * wpr;")
    a("    const float sc = (float)scales[(ulong)eid * m_rows + row];")

    def emit_body(c_val: int):
        for b in range(c_val):
            a(f"        const uint s{b} = (packed >> {b * 8}u) & 0xFFu;")
            a(f"        const device float* x{b} = x + (ulong)((x_div == 1u) ? s{b} : (s{b} / x_div)) * {n}u;")
            a(f"        float a{b} = 0.0f;")
        a("        for (uint it = 0; it < iters; it++) {")
        a("            const uint c = hl + 16u * it;")
        a(f"            const uint col0 = c * {ch}u;")
        a(f"            const uint B = ({n}u - {ch}u * (c + 1u)) * {k}u, w0 = B >> 5;")
        if need_off:
            a("            const uint o = B & 31u;")
        a("            const device uint* p = row_codes + w0;")
        a(f"            uint w[{nw + 1}];")
        for i in range(nw):
            a(f"            w[{i}] = (w0 + {i}u < wpr) ? p[{i}] : 0u;")
        a(f"            w[{nw}] = 0u;")
        const_off = not need_off
        if mod32 == 16:
            a("            if (o != 0u) {")
            for i in range(nw):
                a(f"                w[{i}] = (w[{i}] >> 16u) | (w[{i + 1}] << 16u);")
            a("            }")
            const_off = True
        for i in range(0, ch, g):
            last = i + g - 1
            bit = (ch - 1 - last) * k
            if not const_off:
                a(f"            {{ const uint bit = o + {bit}u, wi = bit >> 5, off = bit & 31u;")
                a(f"              const uint s = ((w[wi] >> off) | (off == 0u ? 0u : (w[wi + 1] << (32u - off)))) & 4095u;")
            else:
                wi, off = bit >> 5, bit & 31
                if off + 12 <= 32:
                    a(f"            {{ const uint s = extract_bits(w[{wi}], {off}u, 12u);")
                else:
                    lo = 32 - off
                    a(f"            {{ const uint s = insert_bits(extract_bits(w[{wi}], {off}u, {lo}u), w[{wi + 1}], {lo}u, {12 - lo}u);")
            if use_hash:
                a(f"              const uint idx = (s * {HASH_MULT_HEX}) >> {shift}u;")
            else:
                a("              const uint idx = s;")
            if slots == 1:
                a("              const float v0 = (float)tg_lut[idx];")
                for b in range(c_val):
                    a(f"              a{b} += v0 * x{b}[col0 + {i}u];")
                a("            }")
            else:
                a(f"              const {vec} v = tg_lut[idx];")
                for u in range(g):
                    a(f"              const float vf{u} = (float)v[{u}];")
                for b in range(c_val):
                    tb = " + ".join(f"vf{u} * x{b}[col0 + {i + u}u]" for u in range(g))
                    a(f"              a{b} += {tb};")
                a("            }")
        a("        }")
        for b in range(c_val):
            a(f"        a{b} += simd_shuffle_xor(a{b}, 8u); a{b} += simd_shuffle_xor(a{b}, 4u); a{b} += simd_shuffle_xor(a{b}, 2u); a{b} += simd_shuffle_xor(a{b}, 1u);")
            a(f"        if (hl == 0u) y[(ulong)s{b} * m_rows + row] = a{b} * sc;")

    a("    if (cnt == 1u) {")
    emit_body(1)
    a("    } else if (cnt == 2u) {")
    emit_body(2)
    a("    } else if (cnt == 3u) {")
    emit_body(3)
    a("    } else {")
    emit_body(4)
    a("    }")
    a("}")
    return mx.fast.metal_kernel(
        name=f"hw_moe_grp_k{k}_n{n}_g{g}_E{lut_entries}_h{int(use_hash)}",
        input_names=["x", "codes", "scales", "lut", "expert_ids", "meta"],
        output_names=["y"],
        header=header,
        source="\n".join(L),
    )


def trellis_moe_grouped(
    x_rot: mx.array,
    codes: mx.array,
    scales_f16: mx.array,
    lut_f16: mx.array,
    expert_ids: mx.array,
    m: int,
    n: int,
    k: int,
    g: int,
    lut_entries: int,
    hash_mult: int,
    x_div: int = 1,
    tg_count: int = 80,
) -> mx.array:
    """Unique-expert grouped multi-token MoE matmul for `n_slots <= 64`, `n_experts <= 128`."""
    sg = _hw_sg()
    n_slots = int(expert_ids.shape[0])
    n_experts = int(codes.shape[0])
    if n_slots > 64 or n_experts > 128:
        return trellis_matvec(
            x_rot, codes, scales_f16, lut_f16, expert_ids,
            m, n, k, g, lut_entries, hash_mult, x_div=x_div, tg_count=tg_count,
        )
    kern = make_halfwarp_trellis_moe_grouped_kernel(k, n, g, lut_entries, hash_mult, sg=sg)
    meta = _meta_u32(m, n_slots, x_div, tg_count, n_experts)
    out = kern(
        inputs=[x_rot.reshape(-1), codes.reshape(-1), scales_f16.reshape(-1), lut_f16.reshape(-1), expert_ids, meta],
        grid=(tg_count * sg * 32, 1, 1),
        threadgroup=(sg * 32, 1, 1),
        output_shapes=[(n_slots * m,)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(n_slots, m)


# ----------------------------------------------------------------------------- Half-Warp Lloyd-Max (lm6 / lm8) Metal Kernel

@lru_cache(maxsize=32)
def make_halfwarp_lm_kernel(k: int, n: int, sg: int = 16, rps: int = 4):
    """Half-warp Metal kernel for bit-packed scalar Lloyd-Max (`lm6` or `lm8`) with 256B-1KB SRAM LUT."""
    assert k in (6, 8), f"Only lm6/lm8 supported, got k={k}"
    assert n % 64 == 0, f"n={n} must be divisible by 64"
    # Each of 16 lanes processes ch=8 (or 16) columns per iter; 4 cols = 3 bytes (lm6) or 4 bytes (lm8)
    ch = 16 if n % 256 == 0 else 8
    iters = n // (16 * ch)
    bpr = (n * k) // 8  # bytes per row (divisible by 4 because ch*k/8 is integer and n%64==0)
    n_levels = 1 << k

    header = f"#define RPS {rps}u\n#define SG {sg}u\n"
    L = []
    a = L.append
    a("const uint tg = threadgroup_position_in_grid.x;")
    a("const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;")
    a("const uint hw = lane >> 4, hl = lane & 15u;")
    a("const uint tid = simd * 32u + lane;")
    a(f"const uint iters = {iters}u;")
    a(f"const uint bpr = {bpr}u;")
    a("const uint m_rows = meta[0];")
    a("const uint n_slots = meta[1];")
    a("const uint ntg = meta[2];")
    a(f"threadgroup float tg_lut[{n_levels}];")
    a(f"for (uint idx = tid; idx < {n_levels}u; idx += {sg * 32}u) tg_lut[idx] = levels[idx];")
    a("threadgroup_barrier(mem_flags::mem_threadgroup);")

    a("const uint blocks_per_slot = (m_rows + RPS - 1u) / RPS;")
    a("const uint total_blocks = n_slots * blocks_per_slot;")
    a("const uint blocks_per_tg = SG * 2u;")
    a("for (uint rb = tg; rb * blocks_per_tg < total_blocks; rb += ntg) {")
    a("    const uint gb = rb * blocks_per_tg + simd * 2u + hw;")
    a("    if (gb >= total_blocks) continue;")
    a("    const uint slot = gb / blocks_per_slot;")
    a("    const uint r0 = (gb - slot * blocks_per_slot) * RPS;")
    a(f"    const device float* xs = x + (ulong)slot * {n}u;")
    a("    float acc[RPS];")
    a("    for (uint r = 0; r < RPS; r++) acc[r] = 0.0f;")
    a("    for (uint it = 0; it < iters; it++) {")
    a("        const uint c = hl + 16u * it;")
    a(f"        float xr[{ch}];")
    for i in range(ch):
        a(f"        xr[{i}] = xs[c * {ch}u + {i}u];")
    a(f"        const uint byte_off = (c * {ch}u * {k}u) >> 3;")
    a("        #pragma unroll")
    a("        for (uint r = 0; r < RPS; r++) {")
    a("            const uint row = min(r0 + r, m_rows - 1u);")
    a("            const device uchar* p = codes + (ulong)row * bpr + byte_off;")
    a("            float a_val = 0.0f;")
    if k == 8:
        for i in range(0, ch, 4):
            a(f"            a_val += tg_lut[p[{i}]] * xr[{i}] + tg_lut[p[{i+1}]] * xr[{i+1}] + tg_lut[p[{i+2}]] * xr[{i+2}] + tg_lut[p[{i+3}]] * xr[{i+3}];")
    else:  # k == 6: 4 indices packed in 3 bytes
        for g4 in range(0, ch, 4):
            bo = (g4 // 4) * 3
            a(f"            {{ const uint v = (uint)p[{bo}] | ((uint)p[{bo+1}] << 8) | ((uint)p[{bo+2}] << 16);")
            a(f"              a_val += tg_lut[v & 63u] * xr[{g4}] + tg_lut[(v >> 6) & 63u] * xr[{g4+1}] + tg_lut[(v >> 12) & 63u] * xr[{g4+2}] + tg_lut[(v >> 18) & 63u] * xr[{g4+3}]; }}")
    a("            acc[r] += a_val;")
    a("        }")
    a("    }")
    a("    #pragma unroll")
    a("    for (uint r = 0; r < RPS; r++) {")
    a("        float s = acc[r];")
    a("        s += simd_shuffle_xor(s, 8u);")
    a("        s += simd_shuffle_xor(s, 4u);")
    a("        s += simd_shuffle_xor(s, 2u);")
    a("        s += simd_shuffle_xor(s, 1u);")
    a("        if (hl == 0u && r0 + r < m_rows) y[(ulong)slot * m_rows + r0 + r] = s * (float)scales[r0 + r];")
    a("    }")
    a("}")
    return mx.fast.metal_kernel(
        name=f"hw_lm_k{k}_n{n}",
        input_names=["x", "codes", "scales", "levels", "meta"],
        output_names=["y"],
        header=header,
        source="\n".join(L),
    )


def lm_matvec(
    x_rot: mx.array,
    codes_u8: mx.array,
    scales_f16: mx.array,
    levels_f32: mx.array,
    m: int,
    n: int,
    k: int,
    tg_count: int = 80,
) -> mx.array:
    """Execute half-warp packed lm6/lm8 matvec on x_rot[T, n]; returns float32[T, m]."""
    sg = _hw_sg()
    x2 = x_rot.reshape(-1, n)
    n_slots = int(x2.shape[0])
    blocks_per_slot = (m + 3) // 4
    total_blocks = n_slots * blocks_per_slot
    ntg = max(1, min(tg_count, (total_blocks + 2 * sg - 1) // (2 * sg)))
    kern = make_halfwarp_lm_kernel(k, n, sg=sg)
    meta = _meta_u32(m, n_slots, ntg)
    out = kern(
        inputs=[x2.reshape(-1), codes_u8.reshape(-1), scales_f16.reshape(-1), levels_f32.reshape(-1), meta],
        grid=(ntg * sg * 32, 1, 1),
        threadgroup=(sg * 32, 1, 1),
        output_shapes=[(n_slots * m,)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(n_slots, m)


@lru_cache(maxsize=64)
def make_halfwarp_lm_dense_bt_kernel(k: int, n: int, t_blk: int, sg: int = 16):
    """Multi-token batched half-warp lm6/lm8 kernel: reads each packed row ONCE across `t_blk` tokens!"""
    assert k in (6, 8) and n % 64 == 0
    ch = 16 if n % 256 == 0 else 8
    iters = n // (16 * ch)
    bpr = (n * k) // 8
    n_levels = 1 << k

    header = f"#define SG {sg}u\n#define TBLK {t_blk}u\n"
    L = []
    a = L.append
    a("const uint tg = threadgroup_position_in_grid.x;")
    a("const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;")
    a("const uint hw = lane >> 4, hl = lane & 15u;")
    a("const uint tid = simd * 32u + lane;")
    a(f"const uint iters = {iters}u;")
    a(f"const uint bpr = {bpr}u;")
    a("const uint m_rows = meta[0];")
    a("const uint ntg = meta[1];")
    a(f"threadgroup float tg_lut[{n_levels}];")
    a(f"for (uint idx = tid; idx < {n_levels}u; idx += {sg * 32}u) tg_lut[idx] = levels[idx];")
    a("threadgroup_barrier(mem_flags::mem_threadgroup);")

    a("const uint rows_per_tg = SG * 2u;")
    a("for (uint rb = tg; rb * rows_per_tg < m_rows; rb += ntg) {")
    a("    const uint row = rb * rows_per_tg + simd * 2u + hw;")
    a("    if (row >= m_rows) continue;")
    a("    float acc[TBLK];")
    a("    #pragma unroll")
    a("    for (uint t = 0; t < TBLK; t++) acc[t] = 0.0f;")
    a("    const device uchar* row_codes = codes + (ulong)row * bpr;")
    a("    for (uint it = 0; it < iters; it++) {")
    a("        const uint c = hl + 16u * it;")
    a(f"        const uint col0 = c * {ch}u;")
    a(f"        const device uchar* p = row_codes + ((c * {ch}u * {k}u) >> 3);")
    if k == 8:
        for i in range(0, ch, 4):
            a(f"        {{ const float w0 = tg_lut[p[{i}]], w1 = tg_lut[p[{i+1}]], w2 = tg_lut[p[{i+2}]], w3 = tg_lut[p[{i+3}]];")
            a("          #pragma unroll")
            a(f"          for (uint t = 0; t < TBLK; t++) {{ const device float* xt = x + (ulong)t * {n}u + col0 + {i}u; acc[t] += w0 * xt[0] + w1 * xt[1] + w2 * xt[2] + w3 * xt[3]; }} }}")
    else:
        for g4 in range(0, ch, 4):
            bo = (g4 // 4) * 3
            a(f"        {{ const uint v = (uint)p[{bo}] | ((uint)p[{bo+1}] << 8) | ((uint)p[{bo+2}] << 16);")
            a("          const float w0 = tg_lut[v & 63u], w1 = tg_lut[(v >> 6) & 63u], w2 = tg_lut[(v >> 12) & 63u], w3 = tg_lut[(v >> 18) & 63u];")
            a("          #pragma unroll")
            a(f"          for (uint t = 0; t < TBLK; t++) {{ const device float* xt = x + (ulong)t * {n}u + col0 + {g4}u; acc[t] += w0 * xt[0] + w1 * xt[1] + w2 * xt[2] + w3 * xt[3]; }} }}")
    a("    }")
    a("    const float sc = (float)scales[row];")
    a("    #pragma unroll")
    a("    for (uint t = 0; t < TBLK; t++) {")
    a("        float s = acc[t];")
    a("        s += simd_shuffle_xor(s, 8u);")
    a("        s += simd_shuffle_xor(s, 4u);")
    a("        s += simd_shuffle_xor(s, 2u);")
    a("        s += simd_shuffle_xor(s, 1u);")
    a("        if (hl == 0u) y[(ulong)t * m_rows + row] = s * sc;")
    a("    }")
    a("}")
    return mx.fast.metal_kernel(
        name=f"hw_lm_bt{t_blk}_k{k}_n{n}",
        input_names=["x", "codes", "scales", "levels", "meta"],
        output_names=["y"],
        header=header,
        source="\n".join(L),
    )


def lm_dense_matmul(
    x_rot: mx.array,
    codes_u8: mx.array,
    scales_f16: mx.array,
    levels_f32: mx.array,
    m: int,
    n: int,
    k: int,
    tg_count: int = 80,
) -> mx.array:
    """Dense lm6/lm8 matmul reading packed weights 1x per 8-token block."""
    sg = _hw_sg()
    x2 = x_rot.reshape(-1, n)
    T = int(x2.shape[0])
    if T == 1:
        return lm_matvec(x2, codes_u8, scales_f16, levels_f32, m, n, k, tg_count=tg_count)
    parts = []
    c_flat = codes_u8.reshape(-1)
    s_flat = scales_f16.reshape(-1)
    l_flat = levels_f32.reshape(-1)
    ntg = max(1, min(tg_count, (m + 2 * sg - 1) // (2 * sg)))
    meta = _meta_u32(m, ntg)
    for t0 in range(0, T, 8):
        tb = min(8, T - t0)
        if tb == 1:
            parts.append(lm_matvec(x2[t0 : t0 + 1], codes_u8, scales_f16, levels_f32, m, n, k, tg_count=tg_count))
        else:
            kern = make_halfwarp_lm_dense_bt_kernel(k, n, tb, sg=sg)
            out = kern(
                inputs=[x2[t0 : t0 + tb].reshape(-1), c_flat, s_flat, l_flat, meta],
                grid=(ntg * sg * 32, 1, 1),
                threadgroup=(sg * 32, 1, 1),
                output_shapes=[(tb * m,)],
                output_dtypes=[mx.float32],
            )[0]
            parts.append(out.reshape(tb, m))
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)


# ----------------------------------------------------------------------------- Half-Warp Native NVFP4 MoE Metal Kernel

NVFP4_E2M1_F32 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


@lru_cache(maxsize=1)
def nvfp4_luts_mx() -> tuple[mx.array, mx.array]:
    """Cached `(e2m1_lut[16], e4m3_lut[256])` float32 lookup tables on MLX."""
    e4m3 = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if (b & 0x80) else 1.0
        exp = (b >> 3) & 0xF
        mant = b & 0x7
        if exp == 15 and mant == 7:
            e4m3[b] = 0.0
        elif exp == 0:
            e4m3[b] = sign * (mant / 8.0) * (2.0 ** -6)
        else:
            e4m3[b] = sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
    return mx.array(NVFP4_E2M1_F32), mx.array(e4m3)


@lru_cache(maxsize=16)
def make_halfwarp_nvfp4_kernel(n: int, sg: int = 16, rps: int = 4):
    """Half-warp MoE Metal kernel for exact native NVFP4 (`4.50 bpp`: 4-bit E2M1 + FP8 E4M3 group-16 + FP32 outer)."""
    assert n % 128 == 0, f"NVFP4 kernel expects n divisible by 128, got n={n}"
    iters = n // 128
    wpr = n // 8   # uint32 words per row (8 nibbles per uint32)
    spr = n // 16  # uint8 E4M3 block scales per row

    header = f"#define RPS {rps}u\n#define SG {sg}u\n"
    L = []
    a = L.append
    a("const uint tg = threadgroup_position_in_grid.x;")
    a("const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;")
    a("const uint hw = lane >> 4, hl = lane & 15u;")
    a("const uint tid = simd * 32u + lane;")
    a("const uint m_rows = meta[0];")
    a("const uint n_slots = meta[1];")
    a("const uint x_div = meta[2];")
    a("const uint ntg = meta[3];")
    a("if (n_slots == 0u) return;")
    a("threadgroup float tg_e2m1[16];")
    a("threadgroup float tg_e4m3[256];")
    a("if (tid < 16u) tg_e2m1[tid] = e2m1_lut[tid];")
    a(f"for (uint idx = tid; idx < 256u; idx += {sg * 32}u) tg_e4m3[idx] = e4m3_lut[idx];")
    a("threadgroup_barrier(mem_flags::mem_threadgroup);")

    a("const uint blocks_per_slot = (m_rows + RPS - 1u) / RPS;")
    a("const uint total_blocks = n_slots * blocks_per_slot;")
    a("const uint blocks_per_tg = SG * 2u;")
    a(f"const ulong expert_code_stride = (ulong)m_rows * {wpr}u;")
    a(f"const ulong expert_scale_stride = (ulong)m_rows * {spr}u;")
    a("for (uint rb = tg; rb * blocks_per_tg < total_blocks; rb += ntg) {")
    a("    const uint gb = rb * blocks_per_tg + simd * 2u + hw;")
    a("    if (gb >= total_blocks) continue;")
    a("    const uint slot = gb / blocks_per_slot;")
    a("    const uint r0 = (gb - slot * blocks_per_slot) * RPS;")
    a("    const uint eid = expert_ids[slot];")
    a("    if (eid == 0xFFFFFFFFu) {")
    a("        if (hl == 0u) {")
    a("            #pragma unroll")
    a("            for (uint r = 0; r < RPS; r++) {")
    a("                if (r0 + r < m_rows) y[(ulong)slot * m_rows + r0 + r] = 0.0f;")
    a("            }")
    a("        }")
    a("        continue;")
    a("    }")
    a("    const uint x_row = (x_div == 1u) ? slot : (slot / x_div);")
    a(f"    const device float* xs = x + (ulong)x_row * {n}u;")
    a("    const device uint* ecodes = codes + (ulong)eid * expert_code_stride;")
    a("    const device uchar* escales = scales + (ulong)eid * expert_scale_stride;")
    a("    const float eouter = outer[eid];")
    a("    float acc[RPS];")
    a("    for (uint r = 0; r < RPS; r++) acc[r] = 0.0f;")
    a(f"    for (uint it = 0; it < {iters}u; it++) {{")
    a("        const uint c = hl + 16u * it;")
    a("        const uint sc_idx = c >> 1;")
    a("        float xr[8];")
    for j in range(8):
        a(f"        xr[{j}] = xs[c * 8u + {j}u];")
    a("        #pragma unroll")
    a("        for (uint r = 0; r < RPS; r++) {")
    a("            const uint row = min(r0 + r, m_rows - 1u);")
    a(f"            const uint pk = ecodes[(ulong)row * {wpr}u + c];")
    a(f"            const float bsc = tg_e4m3[escales[(ulong)row * {spr}u + sc_idx]];")
    terms = " + ".join(f"tg_e2m1[(pk >> {4 * j}u) & 15u] * xr[{j}]" for j in range(8))
    a(f"            acc[r] += ({terms}) * bsc;")
    a("        }")
    a("    }")
    a("    #pragma unroll")
    a("    for (uint r = 0; r < RPS; r++) {")
    a("        float s = acc[r];")
    a("        s += simd_shuffle_xor(s, 8u);")
    a("        s += simd_shuffle_xor(s, 4u);")
    a("        s += simd_shuffle_xor(s, 2u);")
    a("        s += simd_shuffle_xor(s, 1u);")
    a("        if (hl == 0u && r0 + r < m_rows) y[(ulong)slot * m_rows + r0 + r] = s * eouter;")
    a("    }")
    a("}")
    return mx.fast.metal_kernel(
        name=f"hw_nvfp4_n{n}",
        input_names=["x", "codes", "scales", "outer", "e2m1_lut", "e4m3_lut", "expert_ids", "meta"],
        output_names=["y"],
        header=header,
        source="\n".join(L),
    )


def nvfp4_matvec(
    x: mx.array,
    codes_u32: mx.array,
    scales_u8: mx.array,
    outer_f32: mx.array,
    expert_ids: mx.array,
    m: int,
    n: int,
    tg_count: int = 80,
    x_div: int = 1,
    e2m1_lut: mx.array | None = None,
    e4m3_lut: mx.array | None = None,
) -> mx.array:
    """Execute half-warp native NVFP4 MoE matvec on unrotated `x`; returns float32[n_slots, m]."""
    sg = _hw_sg()
    n_slots = int(expert_ids.shape[0])
    blocks_per_slot = (m + 3) // 4
    total_blocks = n_slots * blocks_per_slot
    ntg = max(1, min(tg_count, (total_blocks + 2 * sg - 1) // (2 * sg)))
    if e2m1_lut is None or e4m3_lut is None:
        e2m1_d, e4m3_d = nvfp4_luts_mx()
        e2m1_lut = e2m1_d if e2m1_lut is None else e2m1_lut
        e4m3_lut = e4m3_d if e4m3_lut is None else e4m3_lut
    kern = make_halfwarp_nvfp4_kernel(n, sg=sg)
    meta = _meta_u32(m, n_slots, int(x_div), ntg)
    out = kern(
        inputs=[
            x.reshape(-1), codes_u32.reshape(-1), scales_u8.reshape(-1), outer_f32.reshape(-1),
            e2m1_lut, e4m3_lut, expert_ids, meta,
        ],
        grid=(ntg * sg * 32, 1, 1),
        threadgroup=(sg * 32, 1, 1),
        output_shapes=[(n_slots * m,)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(n_slots, m)


def _selftest():
    """Verify bit-exact parity (< 1e-4 vs float64 numpy reference) for ALL kernels on M5 Max."""
    rng = np.random.default_rng(7)
    print(f"Testing MLX half-warp kernels on {mx.device_info().get('architecture')}...", flush=True)
    # 1. RHT orthogonality & exactness
    for n in (2304, 1152, 3072, 1024):
        signs = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        x = rng.standard_normal((3, n)).astype(np.float32)
        y = rht_apply_in(mx.array(x), mx.array(signs))
        x_back = rht_apply_out(y, mx.array(signs))
        mx.eval(x_back)
        err = float(np.linalg.norm(np.array(x_back) - x) / np.linalg.norm(x))
        # Apple M5 Neural Accelerator (MLX 0.32.2) runs float32 GEMM at ~10-bit TF32 mantissa (~7.7e-4/matmul)
        assert err < 3e-3, f"RHT invertibility fail n={n}: {err}"
        print(f"  PASS RHT invertibility n={n}: rel_err={err:.2e}", flush=True)

    # 2. All 4 trellis formats (ht2, ht3, rtr1, rtr4) with non-multiple-of-4 m_rows too!
    for name, k, m, n, g, K, hmult in [
        ("ht2", 2, 2304, 2304, 4, 512, HASH_MULT_U32),
        ("ht3", 3, 2304, 1152, 2, 1024, HASH_MULT_U32),
        ("rtr1", 1, 250, 2304, 1, 4096, 0),
        ("rtr4", 4, 2304, 2304, 1, 4096, 0),
    ]:
        E = 4
        wpr = (n * k) // 32
        codes = rng.integers(0, 2**32, size=(E, m, wpr), dtype=np.uint32)
        scales = rng.uniform(0.01, 0.05, size=(E, m)).astype(np.float16)
        lut = rng.standard_normal((K, g) if g > 1 else (K,)).astype(np.float16)
        eids = np.array([0, 2, 3], dtype=np.uint32)
        x = rng.standard_normal((3, n)).astype(np.float32)
        y_gpu = trellis_matvec(
            mx.array(x), mx.array(codes), mx.array(scales), mx.array(lut), mx.array(eids),
            m, n, k, g, K, hmult, shared_x=False,
        )
        mx.eval(y_gpu)
        # Reference
        shift = np.uint32(32 - int(round(math.log2(K))))
        y_ref = np.zeros((3, m), dtype=np.float64)
        for slot, eid in enumerate(eids):
            acc = np.zeros(m, dtype=np.float64)
            c_e = codes[eid]
            for j in range(0, n, g):
                last = j + g - 1
                bp = (n - 1 - last) * k
                wi, off = bp // 32, bp % 32
                w0 = c_e[:, wi].astype(np.uint64)
                w1 = c_e[:, wi + 1].astype(np.uint64) if wi + 1 < wpr else np.zeros(m, dtype=np.uint64)
                s = ((w0 >> off) | (w1 << (32 - off) if off > 0 else np.uint64(0))) & np.uint64(4095)
                idx = ((s.astype(np.uint32) * np.uint32(hmult)) >> shift).astype(np.int64) if hmult else s.astype(np.int64)
                if g == 1:
                    acc += lut[idx].astype(np.float64) * float(x[slot, j])
                else:
                    acc += lut[idx].astype(np.float64) @ x[slot, j : j + g].astype(np.float64)
            y_ref[slot] = acc * scales[eid].astype(np.float64)
        err = float(np.linalg.norm(np.array(y_gpu) - y_ref) / np.linalg.norm(y_ref))
        assert err < 1e-4, f"Trellis parity fail {name}: {err}"
        print(f"  PASS trellis {name:4s} ({m}x{n}, n_slots=3): rel_err={err:.2e}", flush=True)

    # 3. Packed lm6 & lm8
    for k, m, n in [(6, 250, 2304), (8, 3072, 2304), (6, 2304, 3072)]:
        levels = np.sort(rng.standard_normal(1 << k)).astype(np.float32)
        idx = rng.integers(0, 1 << k, size=(m, n), dtype=np.uint8)
        if k == 8:
            packed = idx.reshape(-1)
        else:
            a = idx.reshape(-1, 4).astype(np.uint32)
            v = a[:, 0] | (a[:, 1] << 6) | (a[:, 2] << 12) | (a[:, 3] << 18)
            packed = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=-1).astype(np.uint8).reshape(-1)
        scales = rng.uniform(0.01, 0.05, size=(m,)).astype(np.float16)
        x = rng.standard_normal((2, n)).astype(np.float32)
        y_gpu = lm_matvec(mx.array(x), mx.array(packed), mx.array(scales), mx.array(levels), m, n, k)
        mx.eval(y_gpu)
        W_ref = levels[idx].astype(np.float64) * scales.astype(np.float64)[:, None]
        y_ref = x.astype(np.float64) @ W_ref.T
        err = float(np.linalg.norm(np.array(y_gpu) - y_ref) / np.linalg.norm(y_ref))
        assert err < 1e-4, f"lm{k} parity fail: {err}"
        print(f"  PASS lm{k} ({m}x{n}, T=2): rel_err={err:.2e}", flush=True)


if __name__ == "__main__":
    _selftest()
