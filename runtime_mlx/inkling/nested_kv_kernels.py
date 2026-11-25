"""Nested 4->3->2 bit-plane KV Metal kernels for PagedInklingPool on Apple Silicon MLX.

Implements:
  - Deterministic bitmask RHT-128 involution (`rht128_vec` / `irht128_vec` over `h128_vec`)
    and `nested_rht128_rows_f32` for prefill query/output transforms
  - `kv_h128_nested4_quantize`: RHT-128 + per-head FP16 RMS scale + 4-bit nested compander
    (`alpha = 0.4`) packed via portable `simd_sum(((bit) & 1u) << lane)` into 3 independent
    plane buffers (`base_u32` for p0/p1 [2.125 b/v], `p2_u32` [+1.0 b/v], `p3_u32` [+1.0 b/v])
  - `kv_h128_nested_dequantize`: exact 4/3/2-bit dequantization for validation
  - `gqa_paged_nested_sdpa_fast`: paged Split-K decode/verify (`B >= 1, K >= 1`) and
    tiled `simdgroup_half8x8` FlashAttention-2 prefill (`B == 1, K >= 16`) reading per-block
    `p2`/`p3` pointers once per 64-token block and touching only resident planes
  - `gqa_paged_nested_coarse_score`: lightweight 2-bit-K-only block scorer (`base_pool` +
    `scale_pool` + `rel_st`/`rel_proj_t`) for cold-block top-k host promotion
"""
from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

from .batch_kernels import PAGED_BLOCK_SIZE, _choose_paged_splits
from .fast_kernels import _H128_HEADER, _f32_arr, _u32

# Nested Compander alpha=0.4 constants (exact unit-Gaussian bounds and centroids)
NESTED4_BOUNDS: tuple[float, ...] = (
    -2.4256576, -1.8188621, -1.4027019, -1.0664619,
    -0.7728234, -0.5038131, -0.2487300, 0.0,
    0.2487300, 0.5038131, 0.7728234, 1.0664619,
    1.4027019, 1.8188621, 2.4256576,
)
NESTED4_LUT: tuple[float, ...] = (
    -2.7551373, -2.0596104, -1.5878378, -1.2230271,
    -0.9130617, -0.6344800, -0.3742360, -0.1237252,
    0.1237252, 0.3742360, 0.6344800, 0.9130617,
    1.2230271, 1.5878378, 2.0596104, 2.7551373,
)
NESTED3_LUT: tuple[float, ...] = (
    -2.2137920, -1.3771117, -0.7647075, -0.2466245,
    0.2466245, 0.7647075, 1.3771117, 2.2137920,
)
NESTED2_LUT: tuple[float, ...] = (
    -1.5786192, -0.4848241, 0.4848241, 1.5786192,
)


def _compute_det_rht128_signs() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Deterministic 128-element +-1 sign vector packed as 4 uint32 lane bitmasks."""
    signs = np.empty((128,), dtype=np.float32)
    masks = [0, 0, 0, 0]
    for i in range(128):
        h = (i ^ 0xA5A5A5A5) & 0xFFFFFFFF
        h = ((h ^ (h >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
        h = ((h ^ (h >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
        neg = (h >> 31) & 1
        signs[i] = -1.0 if neg else 1.0
        if neg:
            masks[i & 3] |= 1 << (i >> 2)
    return signs, (masks[0], masks[1], masks[2], masks[3])


RHT128_SIGNS_NP, RHT128_MASKS = _compute_det_rht128_signs()
RHT128_SIGNS_MX = mx.array(RHT128_SIGNS_NP)

_RHT128_HEADER = (
    _H128_HEADER
    + f"""
#define NESTED_RHT128_M0 0x{RHT128_MASKS[0]:08x}u
#define NESTED_RHT128_M1 0x{RHT128_MASKS[1]:08x}u
#define NESTED_RHT128_M2 0x{RHT128_MASKS[2]:08x}u
#define NESTED_RHT128_M3 0x{RHT128_MASKS[3]:08x}u

inline float4 nested_rht128_signs(uint lane) {{
    return float4(
        ((NESTED_RHT128_M0 >> lane) & 1u) ? -1.0f : 1.0f,
        ((NESTED_RHT128_M1 >> lane) & 1u) ? -1.0f : 1.0f,
        ((NESTED_RHT128_M2 >> lane) & 1u) ? -1.0f : 1.0f,
        ((NESTED_RHT128_M3 >> lane) & 1u) ? -1.0f : 1.0f
    );
}}

inline float4 nested_rht128_vec(float4 v, uint lane) {{
    const float4 s = nested_rht128_signs(lane);
    return h128_vec(v * s, lane) * s;
}}

inline float4 nested_irht128_vec(float4 v, uint lane) {{
    return nested_rht128_vec(v, lane);
}}
"""
)


def _fmt_f_arr(vals: tuple[float, ...]) -> str:
    return ", ".join(f"{v:.8f}f" for v in vals)


_NESTED_CONSTS_HEADER = f"""
constant float NESTED4_BOUNDS[15] = {{{_fmt_f_arr(NESTED4_BOUNDS)}}};
constant float NESTED4_LUT[16]    = {{{_fmt_f_arr(NESTED4_LUT)}}};
constant float NESTED3_LUT[8]     = {{{_fmt_f_arr(NESTED3_LUT)}}};
constant float NESTED2_LUT[4]     = {{{_fmt_f_arr(NESTED2_LUT)}}};

inline uint nested4_quant(float x) {{
    uint q = (x >= NESTED4_BOUNDS[7]) ? 8u : 0u;
    if (x >= NESTED4_BOUNDS[q + 3u]) q |= 4u;
    if (x >= NESTED4_BOUNDS[q + 1u]) q |= 2u;
    if (x >= NESTED4_BOUNDS[q])      q |= 1u;
    return q;
}}

inline float4 decode_nested4(uint4 b0, uint4 b1, uint4 b2, uint4 b3, uint lane, float sc) {{
    const uint4 idx = (((b0 >> lane) & 1u) << 3) | (((b1 >> lane) & 1u) << 2) | (((b2 >> lane) & 1u) << 1) | ((b3 >> lane) & 1u);
    return float4(NESTED4_LUT[idx.x], NESTED4_LUT[idx.y], NESTED4_LUT[idx.z], NESTED4_LUT[idx.w]) * sc;
}}

inline float4 decode_nested3(uint4 b0, uint4 b1, uint4 b2, uint lane, float sc) {{
    const uint4 idx = (((b0 >> lane) & 1u) << 2) | (((b1 >> lane) & 1u) << 1) | ((b2 >> lane) & 1u);
    return float4(NESTED3_LUT[idx.x], NESTED3_LUT[idx.y], NESTED3_LUT[idx.z], NESTED3_LUT[idx.w]) * sc;
}}

inline float4 decode_nested2(uint4 b0, uint4 b1, uint lane, float sc) {{
    const uint4 idx = (((b0 >> lane) & 1u) << 1) | ((b1 >> lane) & 1u);
    return float4(NESTED2_LUT[idx.x], NESTED2_LUT[idx.y], NESTED2_LUT[idx.z], NESTED2_LUT[idx.w]) * sc;
}}
"""


# ----------------------------------------------------------------------------- Row RHT-128 & Quantize/Dequantize


@lru_cache(maxsize=1)
def _make_nested_rht128_rows_f32_kernel():
    src = """
    const uint row = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const float4 v = *((const device float4*)(x + (ulong)row * 128u) + lane);
    *((device float4*)(out + (ulong)row * 128u) + lane) = nested_rht128_vec(v, lane);
    """
    return mx.fast.metal_kernel(
        name="ink_nested_rht128_rows_f32",
        input_names=["x"],
        output_names=["out"],
        header=_RHT128_HEADER,
        source=src,
    )


def nested_rht128_rows_f32(x: mx.array) -> mx.array:
    """Apply deterministic involution RHT-128 along the last dim (`128`) in float32."""
    xc = mx.contiguous(x.astype(mx.float32))
    assert xc.shape[-1] == 128
    n_rows = xc.size // 128
    if n_rows == 0:
        return xc
    return _make_nested_rht128_rows_f32_kernel()(
        inputs=[xc.reshape(-1)],
        grid=(32, n_rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[xc.shape],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=1)
def _make_kv_h128_nested4_quantize_kernel():
    src = """
    const uint row = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const float4 v = float4(*((const device half4*)(x + (ulong)row * 128u) + lane));
    const float4 hv = nested_rht128_vec(v, lane);

    const float ss = simd_sum(dot(hv, hv));
    const half sc_h = half(max(metal::precise::sqrt(ss * (1.0f / 128.0f)), 1e-6f));
    const float inv_sc = 1.0f / float(sc_h);

    const uint q0 = nested4_quant(hv.x * inv_sc);
    const uint q1 = nested4_quant(hv.y * inv_sc);
    const uint q2 = nested4_quant(hv.z * inv_sc);
    const uint q3 = nested4_quant(hv.w * inv_sc);

    const uint p0_w0 = simd_sum(((q0 >> 3) & 1u) << lane);
    const uint p0_w1 = simd_sum(((q1 >> 3) & 1u) << lane);
    const uint p0_w2 = simd_sum(((q2 >> 3) & 1u) << lane);
    const uint p0_w3 = simd_sum(((q3 >> 3) & 1u) << lane);

    const uint p1_w0 = simd_sum(((q0 >> 2) & 1u) << lane);
    const uint p1_w1 = simd_sum(((q1 >> 2) & 1u) << lane);
    const uint p1_w2 = simd_sum(((q2 >> 2) & 1u) << lane);
    const uint p1_w3 = simd_sum(((q3 >> 2) & 1u) << lane);

    const uint p2_w0 = simd_sum(((q0 >> 1) & 1u) << lane);
    const uint p2_w1 = simd_sum(((q1 >> 1) & 1u) << lane);
    const uint p2_w2 = simd_sum(((q2 >> 1) & 1u) << lane);
    const uint p2_w3 = simd_sum(((q3 >> 1) & 1u) << lane);

    const uint p3_w0 = simd_sum((q0 & 1u) << lane);
    const uint p3_w1 = simd_sum((q1 & 1u) << lane);
    const uint p3_w2 = simd_sum((q2 & 1u) << lane);
    const uint p3_w3 = simd_sum((q3 & 1u) << lane);

    if (lane == 0u) {
        sc_out[row] = sc_h;
        device uint4* b4 = (device uint4*)(base_out + (ulong)row * 8u);
        b4[0] = uint4(p0_w0, p0_w1, p0_w2, p0_w3);
        b4[1] = uint4(p1_w0, p1_w1, p1_w2, p1_w3);
        *((device uint4*)(p2_out + (ulong)row * 4u)) = uint4(p2_w0, p2_w1, p2_w2, p2_w3);
        *((device uint4*)(p3_out + (ulong)row * 4u)) = uint4(p3_w0, p3_w1, p3_w2, p3_w3);
    }
    """
    return mx.fast.metal_kernel(
        name="ink_kv_h128_nested4_quantize",
        input_names=["x"],
        output_names=["base_out", "sc_out", "p2_out", "p3_out"],
        header=_RHT128_HEADER + _NESTED_CONSTS_HEADER,
        source=src,
    )


def kv_h128_nested4_quantize(kv_f16: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Quantize FP16 KV `[..., 128]` into `(base_u32, scales_f16, p2_u32, p3_u32)`.

    For `kv_f16` of shape `[T, 2, 8, 128]`:
      - `base_u32`: `[T, 2, 8, 2, 4]` `uint32` (p0, p1 -> 2.125 b/v with FP16 scale)
      - `scales_f16`: `[T, 2, 8]` `float16`
      - `p2_u32`: `[T, 2, 8, 4]` `uint32` (+1.0 b/v)
      - `p3_u32`: `[T, 2, 8, 4]` `uint32` (+1.0 b/v)
    """
    xc = mx.contiguous(kv_f16.astype(mx.float16))
    assert xc.shape[-1] == 128
    prefix = xc.shape[:-1]
    n_rows = xc.size // 128
    if n_rows == 0:
        return (
            mx.zeros((*prefix, 2, 4), dtype=mx.uint32),
            mx.zeros(prefix, dtype=mx.float16),
            mx.zeros((*prefix, 4), dtype=mx.uint32),
            mx.zeros((*prefix, 4), dtype=mx.uint32),
        )
    return _make_kv_h128_nested4_quantize_kernel()(
        inputs=[xc.reshape(-1)],
        grid=(32, n_rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*prefix, 2, 4), prefix, (*prefix, 4), (*prefix, 4)],
        output_dtypes=[mx.uint32, mx.float16, mx.uint32, mx.uint32],
    )


@lru_cache(maxsize=3)
def _make_kv_h128_nested_dequantize_kernel(bits: int):
    in_names = ["base_in", "sc_in"]
    if bits >= 3:
        in_names.append("p2_in")
    if bits == 4:
        in_names.append("p3_in")
    if bits == 4:
        dec = (
            "const uint4 kp2 = *((const device uint4*)(p2_in + (ulong)row * 4u));\n"
            "    const uint4 kp3 = *((const device uint4*)(p3_in + (ulong)row * 4u));\n"
            "    const float4 hv = decode_nested4(kb0, kb1, kp2, kp3, lane, sc);"
        )
    elif bits == 3:
        dec = (
            "const uint4 kp2 = *((const device uint4*)(p2_in + (ulong)row * 4u));\n"
            "    const float4 hv = decode_nested3(kb0, kb1, kp2, lane, sc);"
        )
    else:
        dec = "const float4 hv = decode_nested2(kb0, kb1, lane, sc);"
    src = f"""
    const uint row = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint4 kb0 = *((const device uint4*)(base_in + (ulong)row * 8u));
    const uint4 kb1 = *((const device uint4*)(base_in + (ulong)row * 8u + 4u));
    const float sc = float(sc_in[row]);
    {dec}
    *((device float4*)(out + (ulong)row * 128u) + lane) = nested_irht128_vec(hv, lane);
    """
    return mx.fast.metal_kernel(
        name=f"ink_kv_h128_nested{bits}_dequantize",
        input_names=in_names,
        output_names=["out"],
        header=_RHT128_HEADER + _NESTED_CONSTS_HEADER,
        source=src,
    )


def kv_h128_nested_dequantize(
    base_u32: mx.array,
    scales_f16: mx.array,
    p2_u32: mx.array | None = None,
    p3_u32: mx.array | None = None,
    *,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Dequantize nested 4/3/2-bit bit-planes back into `[..., 128]` vectors."""
    if p2_u32 is None and p3_u32 is not None:
        raise ValueError("Cannot decode p3_u32 without p2_u32")
    assert base_u32.shape[-2:] == (2, 4)
    prefix = base_u32.shape[:-2]
    assert scales_f16.shape == prefix
    bits = 4 if (p2_u32 is not None and p3_u32 is not None) else (3 if p2_u32 is not None else 2)
    n_rows = scales_f16.size
    if n_rows == 0:
        return mx.zeros((*prefix, 128), dtype=dtype)
    ins = [mx.contiguous(base_u32).reshape(-1), mx.contiguous(scales_f16.astype(mx.float16)).reshape(-1)]
    if bits >= 3:
        assert p2_u32 is not None and p2_u32.shape == (*prefix, 4)
        p2_flat = mx.contiguous(p2_u32).reshape(-1)
        ins.append(mx.concatenate([p2_flat, _u32(0, 0, 0, 0)]) if p2_flat.size < 8 else p2_flat)
    if bits == 4:
        assert p3_u32 is not None and p3_u32.shape == (*prefix, 4)
        p3_flat = mx.contiguous(p3_u32).reshape(-1)
        ins.append(mx.concatenate([p3_flat, _u32(0, 0, 0, 0)]) if p3_flat.size < 8 else p3_flat)
    out = _make_kv_h128_nested_dequantize_kernel(bits)(
        inputs=ins,
        grid=(32, n_rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*prefix, 128)],
        output_dtypes=[mx.float32],
    )[0]
    return out.astype(dtype) if dtype != mx.float32 else out


# ----------------------------------------------------------------------------- Nested Paged SDPA (Split-K + FA2)


def _nested_load_tile_kv_snip(bits: int, vec_type: str = "float4") -> str:
    """Load K and V into `tg_k` / `tg_v` (`[16 * 32]`) for `bits` in (4, 3, 2) using `g_blk`, `p2_blk`, `p3_blk`."""
    conv = "half4" if vec_type == "half4" else "float4"
    lines = [
        "const uint tok_in_blk = j_cur & 63u;",
        "const ulong g_pt = (ulong)g_blk * 64u + tok_in_blk;",
        "const uint4 kb0 = *((const device uint4*)(base_pool + g_pt * 128u + (ulong)kvh * 8u));",
        "const uint4 kb1 = *((const device uint4*)(base_pool + g_pt * 128u + (ulong)kvh * 8u + 4u));",
        "const float ksc = float(scale_pool[g_pt * 16u + (ulong)kvh]);",
        "const uint4 vb0 = *((const device uint4*)(base_pool + g_pt * 128u + (ulong)(8u + kvh) * 8u));",
        "const uint4 vb1 = *((const device uint4*)(base_pool + g_pt * 128u + (ulong)(8u + kvh) * 8u + 4u));",
        "const float vsc = float(scale_pool[g_pt * 16u + (ulong)(8u + kvh)]);",
    ]
    if bits >= 3:
        lines.extend([
            "const ulong p2_pt = (ulong)p2_blk * 64u + tok_in_blk;",
            "const uint4 kp2 = *((const device uint4*)(p2_pool + p2_pt * 64u + (ulong)kvh * 4u));",
            "const uint4 vp2 = *((const device uint4*)(p2_pool + p2_pt * 64u + (ulong)(8u + kvh) * 4u));",
        ])
    if bits == 4:
        lines.extend([
            "const ulong p3_pt = (ulong)p3_blk * 64u + tok_in_blk;",
            "const uint4 kp3 = *((const device uint4*)(p3_pool + p3_pt * 64u + (ulong)kvh * 4u));",
            "const uint4 vp3 = *((const device uint4*)(p3_pool + p3_pt * 64u + (ulong)(8u + kvh) * 4u));",
            f"tg_k[r * 32u + lane] = {conv}(decode_nested4(kb0, kb1, kp2, kp3, lane, ksc));",
            f"tg_v[r * 32u + lane] = {conv}(decode_nested4(vb0, vb1, vp2, vp3, lane, vsc));",
        ])
    elif bits == 3:
        lines.extend([
            f"tg_k[r * 32u + lane] = {conv}(decode_nested3(kb0, kb1, kp2, lane, ksc));",
            f"tg_v[r * 32u + lane] = {conv}(decode_nested3(vb0, vb1, vp2, lane, vsc));",
        ])
    else:
        lines.extend([
            f"tg_k[r * 32u + lane] = {conv}(decode_nested2(kb0, kb1, lane, ksc));",
            f"tg_v[r * 32u + lane] = {conv}(decode_nested2(vb0, vb1, lane, vsc));",
        ])
    return " ".join(lines)


@lru_cache(maxsize=2)
def _make_gqa_nested_splitk_reduce_kernel():
    """Pass-2 Split-K merge + inline sink logit + `irht128_vec` inverse for nested paged SDPA."""
    header = "#define LOG2E 1.4426950408889634f\n" + _RHT128_HEADER
    src = """
    const uint ht = threadgroup_position_in_grid.y;
    const uint T = meta[0];
    const uint n_splits = meta[1];
    const uint t = ht / 24u;
    const uint h = ht - t * 24u;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = simd * 32u + lane;

    const device float2* qin2 = (const device float2*)(q_in + (ulong)t * 2304u);
    const device half2* sw2 = (const device half2*)(sink_w + (ulong)h * 2304u);
    float sk_part = 0.0f;
    #pragma unroll
    for (uint i = 0u; i < 9u; i++) {
        const uint idx = tid + i * 128u;
        sk_part += dot(qin2[idx], float2(sw2[idx]));
    }
    sk_part = simd_sum(sk_part) * LOG2E;

    float m_s = -1e30f;
    float l_s = 0.0f;
    float4 a_s = float4(0.0f);
    for (uint s = simd; s < n_splits; s += 4u) {
        const ulong idx_h = ((ulong)s * T + t) * 24u + h;
        const float ms = part_m[idx_h];
        const float ls = part_l[idx_h];
        if (ls > 0.0f) {
            const float4 as = *((const device float4*)(part_o + idx_h * 128u) + lane);
            const float m_new = max(m_s, ms);
            const float al = (l_s > 0.0f) ? exp2(m_s - m_new) : 0.0f;
            const float be = exp2(ms - m_new);
            l_s = l_s * al + ls * be;
            a_s = a_s * al + as * be;
            m_s = m_new;
        }
    }

    threadgroup float tg_m[4], tg_l[4], tg_sk[4];
    threadgroup float4 tg_a[4][32];
    if (lane == 0u) { tg_m[simd] = m_s; tg_l[simd] = l_s; tg_sk[simd] = sk_part; }
    tg_a[simd][lane] = a_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd == 0u) {
        const float sk = tg_sk[0] + tg_sk[1] + tg_sk[2] + tg_sk[3];
        float m_all = sk;
        for (uint s = 0u; s < 4u; s++) m_all = max(m_all, tg_m[s]);
        float l_all = exp2(sk - m_all);
        float4 a_all = float4(0.0f);
        for (uint s = 0u; s < 4u; s++) {
            if (tg_l[s] > 0.0f) {
                const float sc = exp2(tg_m[s] - m_all);
                l_all += tg_l[s] * sc;
                a_all += tg_a[s][lane] * sc;
            }
        }
        const float4 a_out = a_all * (1.0f / l_all);
        *((device float4*)(out + ((ulong)t * 24u + h) * 128u) + lane) = nested_irht128_vec(a_out, lane);
    }
    """
    return mx.fast.metal_kernel(
        name="ink_gqa_nested_splitk_reduce",
        input_names=["part_m", "part_l", "part_o", "q_in", "sink_w", "meta"],
        output_names=["out"],
        header=header,
        source=src,
    )


@lru_cache(maxsize=12)
def _make_gqa_paged_nested_sdpa_kernel(split_out: bool = True, kt_cap: int = 1, use_mask: bool = False):
    """Paged nested 4/3/2-bit Split-K GQA+RelPos+Sinks SDPA (`kt_cap` in {1, 4})."""
    assert kt_cap in (1, 4)
    in_names = ["q", "base_pool", "scale_pool", "p2_pool", "p3_pool", "rel_st", "rel_proj_t"]
    if not split_out:
        in_names.extend(["q_in", "sink_w"])
    in_names.extend(["block_table", "block_p2", "block_p3"])
    if use_mask:
        in_names.append("active_block_mask")
    in_names.extend(["seq_lens", "q_pos0", "kv_starts", "meta"])
    out_names = ["part_m", "part_l", "part_o"] if split_out else ["out"]
    header = "#define LOG2E 1.4426950408889634f\n" + _RHT128_HEADER + _NESTED_CONSTS_HEADER
    mask_check = "if (active_block_mask[bt_off + blk_idx] == 0u) continue;" if use_mask else ""

    def _tile_loop(bits: int, score_body: str) -> str:
        load_snip = _nested_load_tile_kv_snip(bits, vec_type="float4")
        return f"""
                for (uint tile0 = blk_lo; tile0 < blk_hi; tile0 += 16u) {{
                    const uint r_hi = min(16u, blk_hi - tile0);
                    #pragma unroll
                    for (uint p = 0u; p < 4u; p++) {{
                        const uint r = simd + p * 4u;
                        if (r < r_hi) {{
                            const uint j_cur = tile0 + r;
                            {load_snip}
                        }}
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    {score_body}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }}"""

    if kt_cap == 1:
        score_kt1 = """
                    for (uint r = simd; r < r_hi; r += 4u) {
                        const uint j = tile0 + r;
                        const uint dist = q_pos - (kv_start + j);
                        const float4 kv = tg_k[r * 32u + lane];
                        const float4 vv = tg_v[r * 32u + lane];
                        half4 rp0 = half4(0.0h), rp1 = half4(0.0h), rp2 = half4(0.0h), rp3 = half4(0.0h);
                        const bool has_rel = dist < 1024u;
                        if (has_rel) {
                            const device half4* rp = (const device half4*)(rel_proj_t + (ulong)dist * 16u);
                            rp0 = rp[0]; rp1 = rp[1]; rp2 = rp[2]; rp3 = rp[3];
                        }
                        #pragma unroll
                        for (uint g = 0u; g < 3u; g++) {
                            float score = simd_sum(dot(qv[g], kv));
                            if (has_rel) {
                                score += dot(rs0[g], float4(rp0)) + dot(rs1[g], float4(rp1)) + dot(rs2[g], float4(rp2)) + dot(rs3[g], float4(rp3));
                            }
                            const float m_new = max(m_loc[g], score);
                            const float alpha = exp2(m_loc[g] - m_new);
                            const float p_w = exp2(score - m_new);
                            l_loc[g] = l_loc[g] * alpha + p_w;
                            acc[g] = acc[g] * alpha + p_w * vv;
                            m_loc[g] = m_new;
                        }
                    }"""
        block_loop = f"""
        if (sbeg < send) {{
            const uint blk_beg = sbeg >> 6;
            const uint blk_end = (send - 1u) >> 6;
            for (uint blk_idx = blk_beg; blk_idx <= blk_end; blk_idx++) {{
                {mask_check}
                const uint g_blk = block_table[bt_off + blk_idx];
                const int p2_blk = block_p2[g_blk];
                const int p3_blk = block_p3[g_blk];
                const uint blk_lo = max(sbeg, blk_idx << 6);
                const uint blk_hi = min(send, (blk_idx + 1u) << 6);
                if (p3_blk >= 0) {{
                    {_tile_loop(4, score_kt1)}
                }} else if (p2_blk >= 0) {{
                    {_tile_loop(3, score_kt1)}
                }} else {{
                    {_tile_loop(2, score_kt1)}
                }}
            }}
        }}"""
        if split_out:
            epilogue = """
        if (simd == 0u) {
            for (uint g = 0u; g < 3u; g++) {
                float m_all = -1e30f;
                for (uint s = 0u; s < 4u; s++) m_all = max(m_all, tg_m[s][g]);
                float l_all = 0.0f;
                float4 a_all = float4(0.0f);
                if (m_all > -1e20f) {
                    for (uint s = 0u; s < 4u; s++) {
                        if (tg_l[s][g] > 0.0f) {
                            const float sc = exp2(tg_m[s][g] - m_all);
                            l_all += tg_l[s][g] * sc;
                            a_all += tg_a[s][g][lane] * sc;
                        }
                    }
                }
                const uint h = kvh * 3u + g;
                const ulong out_idx = ((ulong)split_id * T + t_global) * 24u + h;
                if (lane == 0u) {
                    part_m[out_idx] = m_all;
                    part_l[out_idx] = l_all;
                }
                *((device float4*)(part_o + out_idx * 128u) + lane) = a_all;
            }
        }"""
        else:
            epilogue = """
        float sk_arr[3];
        const device float2* qin2 = (const device float2*)(q_in + (ulong)t_global * 2304u);
        for (uint g = 0u; g < 3u; g++) {
            const uint h = kvh * 3u + g;
            const device half2* sw2 = (const device half2*)(sink_w + (ulong)h * 2304u);
            float skp = 0.0f;
            #pragma unroll
            for (uint i = 0u; i < 9u; i++) {
                const uint idx = tid + i * 128u;
                skp += dot(qin2[idx], float2(sw2[idx]));
            }
            sk_arr[g] = simd_sum(skp) * LOG2E;
        }
        threadgroup float tg_sk[4][3];
        if (lane == 0u) {
            for (uint g = 0u; g < 3u; g++) tg_sk[simd][g] = sk_arr[g];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {
            for (uint g = 0u; g < 3u; g++) {
                const uint h = kvh * 3u + g;
                const float sk = tg_sk[0][g] + tg_sk[1][g] + tg_sk[2][g] + tg_sk[3][g];
                float m_all = sk;
                for (uint s = 0u; s < 4u; s++) m_all = max(m_all, tg_m[s][g]);
                float l_all = exp2(sk - m_all);
                float4 a_all = float4(0.0f);
                for (uint s = 0u; s < 4u; s++) {
                    if (tg_l[s][g] > 0.0f) {
                        const float sc = exp2(tg_m[s][g] - m_all);
                        l_all += tg_l[s][g] * sc;
                        a_all += tg_a[s][g][lane] * sc;
                    }
                }
                const float4 a_out = a_all * (1.0f / l_all);
                *((device float4*)(out + ((ulong)t_global * 24u + h) * 128u) + lane) = nested_irht128_vec(a_out, lane);
            }
        }"""
        src = f"""
    const uint T = meta[0], K = meta[1], max_blocks = meta[2], is_sliding = meta[3], n_splits = meta[4], j_min_ovr = meta[5];
    const uint gid = threadgroup_position_in_grid.y;
    const uint kvh_sp = gid / T;
    const uint t_global = gid - kvh_sp * T;
    const uint kvh = kvh_sp / n_splits;
    const uint split_id = kvh_sp - kvh * n_splits;

    const uint b = t_global / K;
    const uint lt = t_global - b * K;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = simd * 32u + lane;

    const uint kv_len = seq_lens[b];
    const uint kv_start = kv_starts[b];
    const uint q_pos = q_pos0[b] + lt;
    const ulong bt_off = (ulong)b * max_blocks;

    float4 qv[3], rs0[3], rs1[3], rs2[3], rs3[3];
    for (uint g = 0u; g < 3u; g++) {{
        const uint h = kvh * 3u + g;
        qv[g] = nested_rht128_vec(*((const device float4*)(q + ((ulong)t_global * 24u + h) * 128u) + lane) * (LOG2E / 11.313708498984761f), lane);
        const device float4* rsp = (const device float4*)(rel_st + ((ulong)t_global * 24u + h) * 16u);
        rs0[g] = rsp[0] * LOG2E; rs1[g] = rsp[1] * LOG2E;
        rs2[g] = rsp[2] * LOG2E; rs3[g] = rsp[3] * LOG2E;
    }}

    const int min_pos = is_sliding ? ((int)q_pos - 1023) : 0;
    uint j_lo = (min_pos > (int)kv_start) ? (uint)(min_pos - (int)kv_start) : 0u;
    if (j_min_ovr > j_lo) j_lo = j_min_ovr;
    const uint j_hi = min(kv_len, (q_pos >= kv_start) ? (q_pos - kv_start + 1u) : 0u);
    const uint span = (j_hi > j_lo) ? (j_hi - j_lo) : 0u;
    const uint chunk = (span + n_splits - 1u) / n_splits;
    const uint sbeg = min(j_hi, j_lo + split_id * chunk);
    const uint send = min(j_hi, sbeg + chunk);

    float m_loc[3] = {{-1e30f, -1e30f, -1e30f}};
    float l_loc[3] = {{0.0f, 0.0f, 0.0f}};
    float4 acc[3] = {{float4(0.0f), float4(0.0f), float4(0.0f)}};

    threadgroup float4 tg_k[16u * 32u];
    threadgroup float4 tg_v[16u * 32u];
    {block_loop}

    threadgroup float tg_m[4][3], tg_l[4][3];
    threadgroup float4 tg_a[4][3][32];
    if (lane == 0u) {{
        for (uint g = 0u; g < 3u; g++) {{ tg_m[simd][g] = m_loc[g]; tg_l[simd][g] = l_loc[g]; }}
    }}
    for (uint g = 0u; g < 3u; g++) tg_a[simd][g][lane] = acc[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {epilogue}
    """
    else:
        score_kt4 = """
                    for (uint r = simd; r < r_hi; r += 4u) {
                        const uint j = tile0 + r;
                        const float4 kv = tg_k[r * 32u + lane];
                        const float4 vv = tg_v[r * 32u + lane];
                        for (uint u = 0u; u < n_tok; u++) {
                            if (j < j_lo_t[u] || j >= j_hi_t[u]) continue;
                            const uint dist = q_pos_t[u] - (kv_start + j);
                            half4 rp0 = half4(0.0h), rp1 = half4(0.0h), rp2 = half4(0.0h), rp3 = half4(0.0h);
                            const bool has_rel = dist < 1024u;
                            if (has_rel) {
                                const device half4* rp = (const device half4*)(rel_proj_t + (ulong)dist * 16u);
                                rp0 = rp[0]; rp1 = rp[1]; rp2 = rp[2]; rp3 = rp[3];
                            }
                            #pragma unroll
                            for (uint g = 0u; g < 3u; g++) {
                                float score = simd_sum(dot(qv[u][g], kv));
                                if (has_rel) {
                                    score += dot(rs0[u][g], float4(rp0)) + dot(rs1[u][g], float4(rp1)) + dot(rs2[u][g], float4(rp2)) + dot(rs3[u][g], float4(rp3));
                                }
                                const float m_new = max(m_loc[u][g], score);
                                const float alpha = exp2(m_loc[u][g] - m_new);
                                const float p_w = exp2(score - m_new);
                                l_loc[u][g] = l_loc[u][g] * alpha + p_w;
                                acc[u][g] = acc[u][g] * alpha + p_w * vv;
                                m_loc[u][g] = m_new;
                            }
                        }
                    }"""
        block_loop = f"""
        if (sbeg < send) {{
            const uint blk_beg = sbeg >> 6;
            const uint blk_end = (send - 1u) >> 6;
            for (uint blk_idx = blk_beg; blk_idx <= blk_end; blk_idx++) {{
                {mask_check}
                const uint g_blk = block_table[bt_off + blk_idx];
                const int p2_blk = block_p2[g_blk];
                const int p3_blk = block_p3[g_blk];
                const uint blk_lo = max(sbeg, blk_idx << 6);
                const uint blk_hi = min(send, (blk_idx + 1u) << 6);
                if (p3_blk >= 0) {{
                    {_tile_loop(4, score_kt4)}
                }} else if (p2_blk >= 0) {{
                    {_tile_loop(3, score_kt4)}
                }} else {{
                    {_tile_loop(2, score_kt4)}
                }}
            }}
        }}"""
        src = f"""
    const uint T = meta[0], K = meta[1], max_blocks = meta[2], is_sliding = meta[3], n_splits = meta[4], j_min_ovr = meta[5], n_qt = meta[6];
    const uint gid = threadgroup_position_in_grid.y;
    const uint sp_kvh = 8u * n_splits;
    const uint b_qt = gid / sp_kvh;
    const uint rem = gid - b_qt * sp_kvh;
    const uint kvh = rem / n_splits;
    const uint split_id = rem - kvh * n_splits;
    const uint b = b_qt / n_qt;
    const uint qt = b_qt - b * n_qt;
    const uint lt0 = qt * 4u;
    const uint n_tok = min(4u, K - lt0);
    const uint t0_global = b * K + lt0;

    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint kv_len = seq_lens[b];
    const uint kv_start = kv_starts[b];
    const uint qp_base = q_pos0[b];
    const ulong bt_off = (ulong)b * max_blocks;

    float4 qv[4][3], rs0[4][3], rs1[4][3], rs2[4][3], rs3[4][3];
    uint j_lo_t[4], j_hi_t[4], q_pos_t[4];
    uint j_lo_all = kv_len, j_hi_all = 0u;
    for (uint u = 0u; u < n_tok; u++) {{
        const uint tg = t0_global + u;
        q_pos_t[u] = qp_base + lt0 + u;
        const int mp = is_sliding ? ((int)q_pos_t[u] - 1023) : 0;
        uint jl = (mp > (int)kv_start) ? (uint)(mp - (int)kv_start) : 0u;
        if (j_min_ovr > jl) jl = j_min_ovr;
        const uint jh = min(kv_len, (q_pos_t[u] >= kv_start) ? (q_pos_t[u] - kv_start + 1u) : 0u);
        j_lo_t[u] = jl;
        j_hi_t[u] = jh;
        j_lo_all = min(j_lo_all, jl);
        j_hi_all = max(j_hi_all, jh);
        for (uint g = 0u; g < 3u; g++) {{
            const uint h = kvh * 3u + g;
            qv[u][g] = nested_rht128_vec(*((const device float4*)(q + ((ulong)tg * 24u + h) * 128u) + lane) * (LOG2E / 11.313708498984761f), lane);
            const device float4* rsp = (const device float4*)(rel_st + ((ulong)tg * 24u + h) * 16u);
            rs0[u][g] = rsp[0] * LOG2E; rs1[u][g] = rsp[1] * LOG2E;
            rs2[u][g] = rsp[2] * LOG2E; rs3[u][g] = rsp[3] * LOG2E;
        }}
    }}
    const uint span = (j_hi_all > j_lo_all) ? (j_hi_all - j_lo_all) : 0u;
    const uint chunk = (span + n_splits - 1u) / n_splits;
    const uint sbeg = min(j_hi_all, j_lo_all + split_id * chunk);
    const uint send = min(j_hi_all, sbeg + chunk);

    float m_loc[4][3], l_loc[4][3];
    float4 acc[4][3];
    for (uint u = 0u; u < 4u; u++) {{
        for (uint g = 0u; g < 3u; g++) {{
            m_loc[u][g] = -1e30f; l_loc[u][g] = 0.0f; acc[u][g] = float4(0.0f);
        }}
    }}

    threadgroup float4 tg_k[16u * 32u];
    threadgroup float4 tg_v[16u * 32u];
    {block_loop}

    threadgroup float tg_m[4][3], tg_l[4][3];
    threadgroup float4 tg_a[4][3][32];
    for (uint u = 0u; u < n_tok; u++) {{
        if (lane == 0u) {{
            for (uint g = 0u; g < 3u; g++) {{ tg_m[simd][g] = m_loc[u][g]; tg_l[simd][g] = l_loc[u][g]; }}
        }}
        for (uint g = 0u; g < 3u; g++) tg_a[simd][g][lane] = acc[u][g];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {{
            const uint tg = t0_global + u;
            for (uint g = 0u; g < 3u; g++) {{
                float m_all = -1e30f;
                for (uint s = 0u; s < 4u; s++) m_all = max(m_all, tg_m[s][g]);
                float l_all = 0.0f;
                float4 a_all = float4(0.0f);
                if (m_all > -1e20f) {{
                    for (uint s = 0u; s < 4u; s++) {{
                        if (tg_l[s][g] > 0.0f) {{
                            const float sc = exp2(tg_m[s][g] - m_all);
                            l_all += tg_l[s][g] * sc;
                            a_all += tg_a[s][g][lane] * sc;
                        }}
                    }}
                }}
                const uint h = kvh * 3u + g;
                const ulong out_idx = ((ulong)split_id * T + tg) * 24u + h;
                if (lane == 0u) {{
                    part_m[out_idx] = m_all;
                    part_l[out_idx] = l_all;
                }}
                *((device float4*)(part_o + out_idx * 128u) + lane) = a_all;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    """

    return mx.fast.metal_kernel(
        name=f"ink_gqa_paged_nested_sdpa_sp{int(split_out)}_kt{kt_cap}_m{int(use_mask)}",
        input_names=in_names,
        output_names=out_names,
        header=header,
        source=src,
    )


def _fa2_load_kv_block_snip(bits: int, is_k: bool) -> str:
    side_base = "(ulong)kvh * 8u" if is_k else "(ulong)(8u + kvh) * 8u"
    side_sc = "(ulong)kvh" if is_k else "(ulong)(8u + kvh)"
    side_p = "(ulong)kvh * 4u" if is_k else "(ulong)(8u + kvh) * 4u"
    lines = [
        "const ulong g_pt = (ulong)g_blk * 64u + kt_l;",
        f"const uint4 b0 = *((const device uint4*)(base_pool + g_pt * 128u + {side_base}));",
        f"const uint4 b1 = *((const device uint4*)(base_pool + g_pt * 128u + {side_base} + 4u));",
        f"const float sc = float(scale_pool[g_pt * 16u + {side_sc}]);",
    ]
    if bits == 4:
        lines.extend([
            "const ulong p2_pt = (ulong)p2_blk * 64u + kt_l;",
            "const ulong p3_pt = (ulong)p3_blk * 64u + kt_l;",
            f"const uint4 p2w = *((const device uint4*)(p2_pool + p2_pt * 64u + {side_p}));",
            f"const uint4 p3w = *((const device uint4*)(p3_pool + p3_pt * 64u + {side_p}));",
            "hv = half4(decode_nested4(b0, b1, p2w, p3w, d4, sc));",
        ])
    elif bits == 3:
        lines.extend([
            "const ulong p2_pt = (ulong)p2_blk * 64u + kt_l;",
            f"const uint4 p2w = *((const device uint4*)(p2_pool + p2_pt * 64u + {side_p}));",
            "hv = half4(decode_nested3(b0, b1, p2w, d4, sc));",
        ])
    else:
        lines.append("hv = half4(decode_nested2(b0, b1, d4, sc));")
    inner = " ".join(lines)
    return f"""
        #pragma unroll
        for (uint i = 0u; i < 16u; i++) {{
            const uint idx = tid + i * 128u;
            const uint kt_l = idx >> 5, d4 = idx & 31u;
            const uint kj = k0 + kt_l;
            half4 hv = half4(0.0h);
            if (kj >= k_lo && kj < k_hi) {{
                {inner}
            }}
            *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = hv;
        }}"""


@lru_cache(maxsize=2)
def _make_gqa_paged_nested_flash_sdpa_kernel(use_mask: bool = False):
    """Tiled `simdgroup_half8x8` FlashAttention-2 (`BQ=32, BK=64`) over nested 4/3/2-bit paged KV (`B == 1`)."""
    header = (
        """
    #include <metal_simdgroup_matrix>
    #define ELEMS_H2 reinterpret_cast<thread half2&>
    #define ELEMS_F2 reinterpret_cast<thread float2&>
    """
        + _NESTED_CONSTS_HEADER
    )
    in_names = [
        "q", "base_pool", "scale_pool", "p2_pool", "p3_pool",
        "rel_st", "rel_proj_t", "sinks", "block_table", "block_p2", "block_p3",
    ]
    if use_mask:
        in_names.append("active_block_mask")
    in_names.extend(["seq_lens", "q_pos0", "kv_starts", "meta"])
    mask_check = "if (active_block_mask[blk_idx] == 0u) continue;" if use_mask else ""

    src = f"""
    const uint bh = threadgroup_position_in_grid.y;
    const uint T = meta[0];
    const uint is_sliding = meta[1];
    const uint n_qblocks = meta[2];
    const uint h = bh / n_qblocks;
    const uint qb = bh - h * n_qblocks;
    const uint q0 = qb * 32u;
    const uint kvh = h / 3u;

    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = simd * 32u + lane;

    const uint kv_len = seq_lens[0];
    const uint kv_start = kv_starts[0];
    const uint qp0 = q_pos0[0];

    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane / 2u) % 4u);
    const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
    const uint q_t = q0 + simd * 8u + fm;
    const uint q_pos = qp0 + q_t;

    threadgroup half tg_kv[64 * 136];
    threadgroup half tg_rp[96 * 16];

    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float Q_SCALE = LOG2E / 11.313708498984761f;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {{
        const uint idx = tid + i * 128u;
        const uint qt_l = idx >> 5, d4 = idx & 31u;
        half4 qh = half4(0.0h);
        if (q0 + qt_l < T) {{
            const float4 qf = *(const device float4*)(q + ((ulong)(q0 + qt_l) * 24u + h) * 128u + d4 * 4u);
            qh = half4(qf * Q_SCALE);
        }}
        *(threadgroup half4*)(tg_kv + qt_l * 136u + d4 * 4u) = qh;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 Qm[16];
    #pragma unroll
    for (uint d8 = 0; d8 < 16u; d8++) {{
        simdgroup_load(Qm[d8], tg_kv + (simd * 8u) * 136u + d8 * 8u, 136u, ulong2(0, 0), false);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 rs0 = float4(0.0f), rs1 = float4(0.0f), rs2 = float4(0.0f), rs3 = float4(0.0f);
    float m_row = -1e20f;
    float l_row = 0.0f;
    if (q_t < T) {{
        const device float4* rsp = (const device float4*)(rel_st + ((ulong)q_t * 24u + h) * 16u);
        rs0 = rsp[0] * LOG2E; rs1 = rsp[1] * LOG2E; rs2 = rsp[2] * LOG2E; rs3 = rsp[3] * LOG2E;
        const float sk = sinks[(ulong)q_t * 24u + h] * LOG2E;
        m_row = sk;
        l_row = 1.0f;
    }}

    simdgroup_float8x8 O[16];
    #pragma unroll
    for (uint d8 = 0; d8 < 16u; d8++) O[d8] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint q_pos_min = qp0 + q0;
    const uint q_pos_max = qp0 + min(q0 + 31u, T - 1u);
    const int min_pos = is_sliding ? ((int)q_pos_min - 1023) : 0;
    const uint k_lo = (min_pos > (int)kv_start) ? (uint)(min_pos - (int)kv_start) : 0u;
    const uint k_hi = min(kv_len, (q_pos_max >= kv_start) ? (q_pos_max - kv_start + 1u) : 0u);
    const uint k0_start = k_lo & ~63u;

    for (uint k0 = k0_start; k0 < k_hi; k0 += 64u) {{
        const uint blk_idx = k0 >> 6;
        {mask_check}
        const uint g_blk = block_table[blk_idx];
        const int p2_blk = block_p2[g_blk];
        const int p3_blk = block_p3[g_blk];

        if (p3_blk >= 0) {{
            {_fa2_load_kv_block_snip(4, True)}
        }} else if (p2_blk >= 0) {{
            {_fa2_load_kv_block_snip(3, True)}
        }} else {{
            {_fa2_load_kv_block_snip(2, True)}
        }}

        const int d_min = (int)q_pos_min - (int)(kv_start + k0 + 63u);
        const bool need_rel = (d_min < 1024) && ((d_min + 95) >= 0);
        if (need_rel) {{
            #pragma unroll
            for (uint i = 0; i < 3u; i++) {{
                const uint idx = tid + i * 128u;
                const uint row = idx >> 2, c4 = idx & 3u;
                const int d = d_min + (int)row;
                *(threadgroup half4*)(tg_rp + row * 16u + c4 * 4u) = (d >= 0 && d < 1024) ? *(const device half4*)(rel_proj_t + (ulong)d * 16u + c4 * 4u) : half4(0.0h);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 S[8];
        #pragma unroll
        for (uint kb = 0; kb < 8u; kb++) S[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        #pragma unroll
        for (uint d8 = 0; d8 < 16u; d8++) {{
            #pragma unroll
            for (uint kb = 0; kb < 8u; kb++) {{
                simdgroup_half8x8 Km;
                simdgroup_load(Km, tg_kv + (kb * 8u) * 136u + d8 * 8u, 136u, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[kb], Qm[d8], Km, S[kb]);
            }}
        }}

        const bool full_valid = (q0 + 32u <= T) && (k0 >= k_lo) && (k0 + 64u <= k_hi) && (d_min >= 0) && (!is_sliding || ((int)q_pos_max - (int)(kv_start + k0) < 1024));
        const bool all_rel = full_valid && ((int)q_pos_max - (int)(kv_start + k0) < 1024);

        float m_tile = -1e30f;
        float2 sv[8];
        if (all_rel) {{
            #pragma unroll
            for (uint kb = 0; kb < 8u; kb++) {{
                float2 s2 = ELEMS_F2(S[kb].thread_elements());
                const uint da = (uint)((int)q_pos - (int)(kv_start + k0 + kb * 8u + fn) - d_min);
                const threadgroup half4* rpa = (const threadgroup half4*)(tg_rp + da * 16u);
                const threadgroup half4* rpb = (const threadgroup half4*)(tg_rp + (da - 1u) * 16u);
                s2.x += dot(rs0, float4(rpa[0])) + dot(rs1, float4(rpa[1])) + dot(rs2, float4(rpa[2])) + dot(rs3, float4(rpa[3]));
                s2.y += dot(rs0, float4(rpb[0])) + dot(rs1, float4(rpb[1])) + dot(rs2, float4(rpb[2])) + dot(rs3, float4(rpb[3]));
                sv[kb] = s2;
                m_tile = max(m_tile, max(s2.x, s2.y));
            }}
        }} else {{
            #pragma unroll
            for (uint kb = 0; kb < 8u; kb++) {{
                float2 s2 = ELEMS_F2(S[kb].thread_elements());
                const uint k_a = k0 + kb * 8u + fn, k_b = k_a + 1u;
                const int da = (int)q_pos - (int)(kv_start + k_a), db = da - 1;
                if ((q_t < T) && (k_a >= k_lo) && (k_a < k_hi) && (da >= 0) && (!is_sliding || da < 1024)) {{
                    if (need_rel && da < 1024) {{
                        const threadgroup half4* rp = (const threadgroup half4*)(tg_rp + (uint)(da - d_min) * 16u);
                        s2.x += dot(rs0, float4(rp[0])) + dot(rs1, float4(rp[1])) + dot(rs2, float4(rp[2])) + dot(rs3, float4(rp[3]));
                    }}
                }} else s2.x = -1e30f;
                if ((q_t < T) && (k_b >= k_lo) && (k_b < k_hi) && (db >= 0) && (!is_sliding || db < 1024)) {{
                    if (need_rel && db < 1024) {{
                        const threadgroup half4* rp = (const threadgroup half4*)(tg_rp + (uint)(db - d_min) * 16u);
                        s2.y += dot(rs0, float4(rp[0])) + dot(rs1, float4(rp[1])) + dot(rs2, float4(rp[2])) + dot(rs3, float4(rp[3]));
                    }}
                }} else s2.y = -1e30f;
                sv[kb] = s2;
                m_tile = max(m_tile, max(s2.x, s2.y));
            }}
        }}
        m_tile = max(m_tile, simd_shuffle_xor(m_tile, 1u));
        m_tile = max(m_tile, simd_shuffle_xor(m_tile, 8u));

        const float m_new = max(m_row, m_tile);
        const float alpha = exp2(m_row - m_new);
        l_row *= alpha;
        #pragma unroll
        for (uint d8 = 0; d8 < 16u; d8++) ELEMS_F2(O[d8].thread_elements()) *= alpha;

        float l_tile = 0.0f;
        simdgroup_half8x8 Pm[8];
        #pragma unroll
        for (uint kb = 0; kb < 8u; kb++) {{
            const float px = (sv[kb].x > -1e20f) ? exp2(sv[kb].x - m_new) : 0.0f;
            const float py = (sv[kb].y > -1e20f) ? exp2(sv[kb].y - m_new) : 0.0f;
            l_tile += px + py;
            ELEMS_H2(Pm[kb].thread_elements()) = half2((half)px, (half)py);
        }}
        l_tile += simd_shuffle_xor(l_tile, 1u);
        l_tile += simd_shuffle_xor(l_tile, 8u);
        l_row += l_tile;
        m_row = m_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (p3_blk >= 0) {{
            {_fa2_load_kv_block_snip(4, False)}
        }} else if (p2_blk >= 0) {{
            {_fa2_load_kv_block_snip(3, False)}
        }} else {{
            {_fa2_load_kv_block_snip(2, False)}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kb = 0; kb < 8u; kb++) {{
            #pragma unroll
            for (uint d8 = 0; d8 < 16u; d8++) {{
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, tg_kv + (kb * 8u) * 136u + d8 * 8u, 136u, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(O[d8], Pm[kb], Vm, O[d8]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (q_t < T) {{
        const float inv_l = 1.0f / l_row;
        device float* op = out + ((ulong)q_t * 24u + h) * 128u;
        #pragma unroll
        for (uint d8 = 0; d8 < 16u; d8++) {{
            const float2 ov = ELEMS_F2(O[d8].thread_elements()) * inv_l;
            op[d8 * 8u + fn] = ov.x;
            op[d8 * 8u + fn + 1u] = ov.y;
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_gqa_paged_nested_flash_sdpa_m{int(use_mask)}",
        input_names=in_names,
        output_names=["out"],
        header=header,
        source=src,
    )


@lru_cache(maxsize=1)
def _dummy_u32() -> mx.array:
    out = mx.zeros((4096,), dtype=mx.uint32)
    mx.eval(out)
    return out


def gqa_paged_nested_sdpa_fast(
    q: mx.array,
    base_pool: mx.array,
    scale_pool: mx.array,
    p2_pool: mx.array,
    p3_pool: mx.array,
    rel_st: mx.array,
    rel_proj_t_f16: mx.array,
    q_in: mx.array,
    sink_w_f16: mx.array,
    block_table: mx.array,
    block_p2: mx.array,
    block_p3: mx.array,
    seq_lens: mx.array,
    q_pos0: mx.array,
    kv_starts: mx.array,
    B: int,
    K: int,
    max_span: int,
    is_sliding: bool = False,
    *,
    active_block_mask: mx.array | None = None,
) -> mx.array:
    """Paged batched GQA+RelPos+Sinks SDPA over nested 4->3->2 bit-plane KV blocks."""
    T = B * K
    max_blocks = int(block_table.shape[1])
    p2_buf = p2_pool if p2_pool.size > 0 else _dummy_u32()
    p3_buf = p3_pool if p3_pool.size > 0 else _dummy_u32()
    use_mask = active_block_mask is not None

    if B == 1 and K >= 16:
        sinks = (q_in.reshape(T, 2304).astype(mx.float16) @ sink_w_f16.reshape(24, 2304).T).astype(mx.float32)
        nqb = (T + 31) // 32
        q_rot = nested_rht128_rows_f32(q)
        ins = [
            q_rot.reshape(-1), base_pool, scale_pool, p2_buf, p3_buf,
            rel_st.reshape(-1), rel_proj_t_f16.reshape(-1), sinks.reshape(-1),
            block_table.reshape(-1), block_p2.reshape(-1), block_p3.reshape(-1),
        ]
        if use_mask:
            ins.append(active_block_mask.astype(mx.uint32).reshape(-1))
        ins.extend([
            seq_lens.reshape(-1), q_pos0.reshape(-1), kv_starts.reshape(-1),
            _u32(T, int(is_sliding), nqb),
        ])
        out = _make_gqa_paged_nested_flash_sdpa_kernel(use_mask=use_mask)(
            inputs=ins,
            grid=(128, 24 * nqb, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(T, 3072)],
            output_dtypes=[mx.float32],
        )[0]
        return nested_rht128_rows_f32(out.reshape(T, 24, 128)).reshape(T, 3072)

    kt_cap = 1 if K == 1 else 4
    n_qt = (K + kt_cap - 1) // kt_cap
    eff_span = min(max_span, 1024 + K) if is_sliding else max_span
    n_splits = _choose_paged_splits(B, eff_span, n_qt)
    split_out = (n_splits > 1) or (kt_cap > 1)

    ins = [q.reshape(-1), base_pool, scale_pool, p2_buf, p3_buf, rel_st.reshape(-1), rel_proj_t_f16.reshape(-1)]
    if not split_out:
        ins.extend([q_in.reshape(-1), sink_w_f16.reshape(-1)])
    ins.extend([block_table.reshape(-1), block_p2.reshape(-1), block_p3.reshape(-1)])
    if use_mask:
        ins.append(active_block_mask.astype(mx.uint32).reshape(-1))
    meta = _u32(T, K, max_blocks, int(is_sliding), n_splits, 0) if kt_cap == 1 else _u32(T, K, max_blocks, int(is_sliding), n_splits, 0, n_qt)
    ins.extend([seq_lens.reshape(-1), q_pos0.reshape(-1), kv_starts.reshape(-1), meta])

    kern = _make_gqa_paged_nested_sdpa_kernel(split_out=split_out, kt_cap=kt_cap, use_mask=use_mask)
    if not split_out:
        return kern(
            inputs=ins,
            grid=(128, B * n_qt * 8 * n_splits, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(T, 3072)],
            output_dtypes=[mx.float32],
        )[0]
    pm, pl, po = kern(
        inputs=ins,
        grid=(128, B * n_qt * 8 * n_splits, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n_splits, T, 24), (n_splits, T, 24), (n_splits, T, 24, 128)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return _make_gqa_nested_splitk_reduce_kernel()(
        inputs=[pm.reshape(-1), pl.reshape(-1), po.reshape(-1), q_in.reshape(-1), sink_w_f16.reshape(-1), _u32(T, n_splits)],
        grid=(128, 24 * T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(T, 3072)],
        output_dtypes=[mx.float32],
    )[0]


# ----------------------------------------------------------------------------- Coarse 2-bit K Block Scorer


@lru_cache(maxsize=2)
def _make_gqa_paged_nested_coarse_score_kernel(reduce_q: bool = False):
    """Lightweight 2-bit-K-only block max logit scorer (`base_pool` + `scale_pool` + `rel_st`/`rel_proj_t`)."""
    header = _RHT128_HEADER + _NESTED_CONSTS_HEADER
    if not reduce_q:
        src = """
    // Grid: (128, B * n_qt * 8 * max_blocks, 1), output: float32[B, K, 8, max_blocks]
    const uint K = meta[0], max_blocks = meta[1], n_qt = meta[2];
    const uint gid = threadgroup_position_in_grid.y;
    const uint qt_kvh_blk = 8u * max_blocks;
    const uint b_qt = gid / qt_kvh_blk;
    const uint rem = gid - b_qt * qt_kvh_blk;
    const uint kvh = rem / max_blocks;
    const uint blk_idx = rem - kvh * max_blocks;
    const uint b = b_qt / n_qt;
    const uint qt = b_qt - b * n_qt;
    const uint lt0 = qt * 4u;
    const uint n_tok = min(4u, K - lt0);
    const uint t0_global = b * K + lt0;

    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint kv_len = seq_lens[b];
    const uint kv_start = kv_starts[b];
    const uint qp_base = q_pos0[b];

    uint j_hi_t[4], q_pos_t[4];
    uint j_hi_all = 0u;
    for (uint u = 0u; u < n_tok; u++) {
        q_pos_t[u] = qp_base + lt0 + u;
        const uint jh = min(kv_len, (q_pos_t[u] >= kv_start) ? (q_pos_t[u] - kv_start + 1u) : 0u);
        j_hi_t[u] = jh;
        j_hi_all = max(j_hi_all, jh);
    }

    const uint blk_k0 = blk_idx * 64u;
    const uint blk_k1 = min(j_hi_all, blk_k0 + 64u);
    if (blk_k0 >= blk_k1) {
        if (simd == 0u && lane < n_tok) {
            block_max[(((ulong)b * K + lt0 + lane) * 8u + kvh) * max_blocks + blk_idx] = -1e30f;
        }
        return;
    }

    float4 qv[4][3], rs0[4][3], rs1[4][3], rs2[4][3], rs3[4][3];
    for (uint u = 0u; u < n_tok; u++) {
        const uint tg = t0_global + u;
        for (uint g = 0u; g < 3u; g++) {
            const uint h = kvh * 3u + g;
            qv[u][g] = nested_rht128_vec(*((const device float4*)(q + ((ulong)tg * 24u + h) * 128u) + lane) * (1.0f / 11.313708498984761f), lane);
            const device float4* rsp = (const device float4*)(rel_st + ((ulong)tg * 24u + h) * 16u);
            rs0[u][g] = rsp[0]; rs1[u][g] = rsp[1]; rs2[u][g] = rsp[2]; rs3[u][g] = rsp[3];
        }
    }

    const uint g_blk = block_table[(ulong)b * max_blocks + blk_idx];
    float m_loc[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    for (uint j = blk_k0 + simd; j < blk_k1; j += 4u) {
        const ulong g_pt = (ulong)g_blk * 64u + (j & 63u);
        const uint4 kb0 = *((const device uint4*)(base_pool + g_pt * 128u + (ulong)kvh * 8u));
        const uint4 kb1 = *((const device uint4*)(base_pool + g_pt * 128u + (ulong)kvh * 8u + 4u));
        const float ksc = float(scale_pool[g_pt * 16u + (ulong)kvh]);
        const float4 kv = decode_nested2(kb0, kb1, lane, ksc);
        for (uint u = 0u; u < n_tok; u++) {
            if (j >= j_hi_t[u]) continue;
            const uint dist = q_pos_t[u] - (kv_start + j);
            half4 rp0 = half4(0.0h), rp1 = half4(0.0h), rp2 = half4(0.0h), rp3 = half4(0.0h);
            const bool has_rel = dist < 1024u;
            if (has_rel) {
                const device half4* rp = (const device half4*)(rel_proj_t + (ulong)dist * 16u);
                rp0 = rp[0]; rp1 = rp[1]; rp2 = rp[2]; rp3 = rp[3];
            }
            #pragma unroll
            for (uint g = 0u; g < 3u; g++) {
                float score = simd_sum(dot(qv[u][g], kv));
                if (has_rel) {
                    score += dot(rs0[u][g], float4(rp0)) + dot(rs1[u][g], float4(rp1)) + dot(rs2[u][g], float4(rp2)) + dot(rs3[u][g], float4(rp3));
                }
                m_loc[u] = max(m_loc[u], score);
            }
        }
    }

    threadgroup float tg_m[4][4];
    if (lane == 0u) {
        for (uint u = 0u; u < n_tok; u++) tg_m[simd][u] = m_loc[u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u && lane < n_tok) {
        const float best = max(max(tg_m[0][lane], tg_m[1][lane]), max(tg_m[2][lane], tg_m[3][lane]));
        block_max[(((ulong)b * K + lt0 + lane) * 8u + kvh) * max_blocks + blk_idx] = best;
    }
    """
    else:
        src = """
    // Grid: (128, B * 8 * max_blocks, 1), output: float32[B, 8, max_blocks] (max across all K query tokens & 3 GQA heads)
    const uint K = meta[0], max_blocks = meta[1];
    const uint gid = threadgroup_position_in_grid.y;
    const uint kvh_blk = 8u * max_blocks;
    const uint b = gid / kvh_blk;
    const uint rem = gid - b * kvh_blk;
    const uint kvh = rem / max_blocks;
    const uint blk_idx = rem - kvh * max_blocks;

    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint kv_len = seq_lens[b];
    const uint kv_start = kv_starts[b];
    const uint qp_base = q_pos0[b];

    const uint q_max_pos = qp_base + K - 1u;
    const uint j_hi_all = min(kv_len, (q_max_pos >= kv_start) ? (q_max_pos - kv_start + 1u) : 0u);
    const uint blk_k0 = blk_idx * 64u;
    const uint blk_k1 = min(j_hi_all, blk_k0 + 64u);
    if (blk_k0 >= blk_k1) {
        if (simd == 0u && lane == 0u) {
            block_max[((ulong)b * 8u + kvh) * max_blocks + blk_idx] = -1e30f;
        }
        return;
    }

    const uint g_blk = block_table[(ulong)b * max_blocks + blk_idx];
    threadgroup float4 tg_k[16u * 32u];
    float m_best = -1e30f;

    for (uint tile0 = blk_k0; tile0 < blk_k1; tile0 += 16u) {
        const uint r_hi = min(16u, blk_k1 - tile0);
        #pragma unroll
        for (uint p = 0u; p < 4u; p++) {
            const uint r = simd + p * 4u;
            if (r < r_hi) {
                const ulong g_pt = (ulong)g_blk * 64u + ((tile0 + r) & 63u);
                const uint4 kb0 = *((const device uint4*)(base_pool + g_pt * 128u + (ulong)kvh * 8u));
                const uint4 kb1 = *((const device uint4*)(base_pool + g_pt * 128u + (ulong)kvh * 8u + 4u));
                const float ksc = float(scale_pool[g_pt * 16u + (ulong)kvh]);
                tg_k[r * 32u + lane] = decode_nested2(kb0, kb1, lane, ksc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint lt = 0u; lt < K; lt++) {
            const uint q_pos = qp_base + lt;
            const uint j_hi_t = min(kv_len, (q_pos >= kv_start) ? (q_pos - kv_start + 1u) : 0u);
            if (tile0 >= j_hi_t) continue;
            const uint r_lim = min(r_hi, j_hi_t - tile0);
            const uint tg = b * K + lt;
            float4 qv0 = nested_rht128_vec(*((const device float4*)(q + ((ulong)tg * 24u + (kvh * 3u + 0u)) * 128u) + lane) * (1.0f / 11.313708498984761f), lane);
            float4 qv1 = nested_rht128_vec(*((const device float4*)(q + ((ulong)tg * 24u + (kvh * 3u + 1u)) * 128u) + lane) * (1.0f / 11.313708498984761f), lane);
            float4 qv2 = nested_rht128_vec(*((const device float4*)(q + ((ulong)tg * 24u + (kvh * 3u + 2u)) * 128u) + lane) * (1.0f / 11.313708498984761f), lane);
            const device float4* rsp0 = (const device float4*)(rel_st + ((ulong)tg * 24u + (kvh * 3u + 0u)) * 16u);
            const device float4* rsp1 = (const device float4*)(rel_st + ((ulong)tg * 24u + (kvh * 3u + 1u)) * 16u);
            const device float4* rsp2 = (const device float4*)(rel_st + ((ulong)tg * 24u + (kvh * 3u + 2u)) * 16u);

            for (uint r = simd; r < r_lim; r += 4u) {
                const uint j = tile0 + r;
                const uint dist = q_pos - (kv_start + j);
                const float4 kv = tg_k[r * 32u + lane];
                float s0 = simd_sum(dot(qv0, kv));
                float s1 = simd_sum(dot(qv1, kv));
                float s2 = simd_sum(dot(qv2, kv));
                if (dist < 1024u) {
                    const device half4* rp = (const device half4*)(rel_proj_t + (ulong)dist * 16u);
                    const float4 rp0 = float4(rp[0]), rp1 = float4(rp[1]), rp2 = float4(rp[2]), rp3 = float4(rp[3]);
                    s0 += dot(rsp0[0], rp0) + dot(rsp0[1], rp1) + dot(rsp0[2], rp2) + dot(rsp0[3], rp3);
                    s1 += dot(rsp1[0], rp0) + dot(rsp1[1], rp1) + dot(rsp1[2], rp2) + dot(rsp1[3], rp3);
                    s2 += dot(rsp2[0], rp0) + dot(rsp2[1], rp1) + dot(rsp2[2], rp2) + dot(rsp2[3], rp3);
                }
                m_best = max(m_best, max(s0, max(s1, s2)));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float tg_m[4];
    if (lane == 0u) tg_m[simd] = m_best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u && lane == 0u) {
        block_max[((ulong)b * 8u + kvh) * max_blocks + blk_idx] = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
    }
    """
    return mx.fast.metal_kernel(
        name=f"ink_gqa_paged_nested_coarse_score_r{int(reduce_q)}",
        input_names=["q", "base_pool", "scale_pool", "rel_st", "rel_proj_t", "block_table", "seq_lens", "q_pos0", "kv_starts", "meta"],
        output_names=["block_max"],
        header=header,
        source=src,
    )


def gqa_paged_nested_coarse_score(
    q: mx.array,
    base_pool: mx.array,
    scale_pool: mx.array,
    rel_st: mx.array,
    rel_proj_t_f16: mx.array,
    block_table: mx.array,
    seq_lens: mx.array,
    q_pos0: mx.array,
    kv_starts: mx.array,
    B: int,
    K: int,
    *,
    reduce_q: bool = False,
) -> mx.array:
    """Score paged blocks using ONLY 2-bit K planes (`p0, p1` + `scale_pool`) + RelPos bias.

    Returns:
      - `reduce_q=False`: `float32[B, K, 8, max_blocks]` max logit per query token and KV head
      - `reduce_q=True`:  `float32[B, 8, max_blocks]` max logit across all `K` query tokens and GQA heads
    """
    max_blocks = int(block_table.shape[1])
    if max_blocks == 0 or B == 0 or K == 0:
        return mx.zeros((B, 8, max_blocks) if reduce_q else (B, K, 8, max_blocks), dtype=mx.float32)

    if reduce_q and K <= 4:
        per_q = gqa_paged_nested_coarse_score(
            q, base_pool, scale_pool, rel_st, rel_proj_t_f16,
            block_table, seq_lens, q_pos0, kv_starts, B, K, reduce_q=False,
        )
        return mx.max(per_q, axis=1)

    if reduce_q:
        return _make_gqa_paged_nested_coarse_score_kernel(reduce_q=True)(
            inputs=[
                q.reshape(-1), base_pool, scale_pool, rel_st.reshape(-1), rel_proj_t_f16.reshape(-1),
                block_table.reshape(-1), seq_lens.reshape(-1), q_pos0.reshape(-1), kv_starts.reshape(-1),
                _u32(K, max_blocks),
            ],
            grid=(128, B * 8 * max_blocks, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(B, 8, max_blocks)],
            output_dtypes=[mx.float32],
        )[0]

    n_qt = (K + 3) // 4
    return _make_gqa_paged_nested_coarse_score_kernel(reduce_q=False)(
        inputs=[
            q.reshape(-1), base_pool, scale_pool, rel_st.reshape(-1), rel_proj_t_f16.reshape(-1),
            block_table.reshape(-1), seq_lens.reshape(-1), q_pos0.reshape(-1), kv_starts.reshape(-1),
            _u32(K, max_blocks, n_qt),
        ],
        grid=(128, B * n_qt * 8 * max_blocks, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(B, K, 8, max_blocks)],
        output_dtypes=[mx.float32],
    )[0]
