"""Hadamard + Lloyd-Max KV cache with FP16 sink token protection for Qwen3.5 / Qwen3.8 on MLX.

Stores the first `sink_tokens` positions (default 4) unrotated in exact FP16, and rotates
subsequent key/value vectors via deterministic RHT-128 or RHT-256 into Hadamard space before
quantizing against Gaussian Lloyd-Max (2, 3, 4 bit) or symmetric int8 (8 bit) codebooks with
one FP16 RMS/absmax scale per head (`bits=16` is stored unrotated in FP16).

Decode SDPA rotates the query once on entry, attends directly over packed codes in Hadamard
space with online softmax + Split-K reduction, unrotates the body accumulator once on exit,
and merges exact FP16 sink contributions. Prefill SDPA chunks keys/values into bounded tiles
using online log-sum-exp merge so long prompts never materialize `[H, L_q, L_kv]` scores or
a full dequantized FP16 body.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import ArraysCache, _BaseCache, create_attention_mask

VALID_KV_BITS = (2, 3, 4, 8, 16)
VALID_HEAD_DIMS = (128, 256)

# Exact Gaussian Lloyd-Max codebooks from opgraph/nested.py
LM2_LEVELS = (-1.5104176, -0.4527800, 0.4527800, 1.5104176)
LM2_MIDS = (-0.9815988, 0.0, 0.9815988)

LM3_LEVELS = (
    -2.1519457, -1.3439093, -0.7560053, -0.2450942,
    0.2450942, 0.7560053, 1.3439093, 2.1519457,
)
LM3_MIDS = (
    -1.7479275, -1.0499573, -0.5005497, 0.0,
    0.5005497, 1.0499573, 1.7479275,
)

LM4_LEVELS = (
    -2.7325896, -2.0690172, -1.6180464, -1.2562312,
    -0.9423405, -0.6567591, -0.3880483, -0.1283950,
    0.1283950, 0.3880483, 0.6567591, 0.9423405,
    1.2562312, 1.6180464, 2.0690172, 2.7325896,
)
LM4_MIDS = (
    -2.4008034, -1.8435318, -1.4371388, -1.0992858,
    -0.7995498, -0.5224037, -0.2582217, 0.0,
    0.2582217, 0.5224037, 0.7995498, 1.0992858,
    1.4371388, 1.8435318, 2.4008034,
)

LM_LEVELS = {2: LM2_LEVELS, 3: LM3_LEVELS, 4: LM4_LEVELS}
LM_MIDS = {2: LM2_MIDS, 3: LM3_MIDS, 4: LM4_MIDS}

# Deterministic 256-bit +/-1 sign bitmask (first 128 bits used for D=128, all 256 for D=256).
# R(x) = diag(s) * H_D * diag(s) * x is an exact orthogonal involution (R == R^T == R^-1).
RHT_SIGN_WORDS = (
    0x9C20E9F6, 0x6873B945, 0xC13521B9, 0xE0965C56,
    0x0CEF6F5C, 0x1BF1DF2F, 0x80F85CB2, 0x0A315805,
)


def rht_signs_np(head_dim: int) -> np.ndarray:
    """Return the deterministic +/-1 sign vector of length `head_dim` (128 or 256)."""
    if head_dim not in VALID_HEAD_DIMS:
        raise ValueError(f"Unsupported head_dim={head_dim}; expected one of {VALID_HEAD_DIMS}")
    bits = np.array([(RHT_SIGN_WORDS[i >> 5] >> (i & 31)) & 1 for i in range(head_dim)], dtype=np.float64)
    return 2.0 * bits - 1.0


def _packed_bytes_per_head(bits: int, head_dim: int) -> int:
    if bits == 2:
        return head_dim // 4
    if bits == 3:
        return (3 * head_dim) // 8
    if bits == 4:
        return head_dim // 2
    if bits in (8, 16):
        return head_dim
    raise ValueError(f"Unsupported KV bits={bits}; expected one of {VALID_KV_BITS}")


def _codes_dtype(bits: int) -> mx.Dtype:
    if bits in (2, 3, 4):
        return mx.uint8
    if bits == 8:
        return mx.int8
    if bits == 16:
        return mx.float16
    raise ValueError(f"Unsupported KV bits={bits}; expected one of {VALID_KV_BITS}")


def _metal_type(dtype: mx.Dtype) -> str:
    if dtype == mx.float32:
        return "float"
    if dtype == mx.float16:
        return "half"
    if dtype == mx.bfloat16:
        return "bfloat16_t"
    if dtype == mx.uint8:
        return "uchar"
    if dtype == mx.int8:
        return "char"
    raise ValueError(f"Unsupported Metal dtype: {dtype}")


# ----------------------------------------------------------------------------- Metal headers

_RHT_HEADER = """
#define LOG2E 1.4426950408889634f

constant uint RHT_SIGN_WORDS[8] = {
    0x9c20e9f6u, 0x6873b945u, 0xc13521b9u, 0xe0965c56u,
    0x0cef6f5cu, 0x1bf1df2fu, 0x80f85cb2u, 0x0a315805u
};

inline float4 rht_apply_signs4(float4 v, uint word_idx, uint shift_base) {
    const uint w = RHT_SIGN_WORDS[word_idx] >> shift_base;
    return float4(
        (w & 1u) ? v.x : -v.x,
        (w & 2u) ? v.y : -v.y,
        (w & 4u) ? v.z : -v.z,
        (w & 8u) ? v.w : -v.w
    );
}

inline float4 fwht128_unnorm(float4 v, uint lane) {
    const float a0 = v.x + v.y, b0 = v.x - v.y;
    const float a1 = v.z + v.w, b1 = v.z - v.w;
    v = float4(a0 + a1, b0 + b1, a0 - a1, b0 - b1);
    float4 o;
    o = float4(simd_shuffle_xor(v.x, 1u), simd_shuffle_xor(v.y, 1u), simd_shuffle_xor(v.z, 1u), simd_shuffle_xor(v.w, 1u));
    v = (lane & 1u) ? (o - v) : (v + o);
    o = float4(simd_shuffle_xor(v.x, 2u), simd_shuffle_xor(v.y, 2u), simd_shuffle_xor(v.z, 2u), simd_shuffle_xor(v.w, 2u));
    v = (lane & 2u) ? (o - v) : (v + o);
    o = float4(simd_shuffle_xor(v.x, 4u), simd_shuffle_xor(v.y, 4u), simd_shuffle_xor(v.z, 4u), simd_shuffle_xor(v.w, 4u));
    v = (lane & 4u) ? (o - v) : (v + o);
    o = float4(simd_shuffle_xor(v.x, 8u), simd_shuffle_xor(v.y, 8u), simd_shuffle_xor(v.z, 8u), simd_shuffle_xor(v.w, 8u));
    v = (lane & 8u) ? (o - v) : (v + o);
    o = float4(simd_shuffle_xor(v.x, 16u), simd_shuffle_xor(v.y, 16u), simd_shuffle_xor(v.z, 16u), simd_shuffle_xor(v.w, 16u));
    v = (lane & 16u) ? (o - v) : (v + o);
    return v;
}

inline float4 rht128_vec(float4 v, uint lane) {
    const uint wi = lane >> 3;
    const uint sb = (lane & 7u) * 4u;
    v = rht_apply_signs4(v, wi, sb);
    v = fwht128_unnorm(v, lane) * 0.08838834764831845f;
    return rht_apply_signs4(v, wi, sb);
}

inline half4 load_half4(const device half* p, ulong i) { return ((const device half4*)p)[i]; }
inline half4 load_half4(const constant half* p, ulong i) { return ((const constant half4*)p)[i]; }
inline float4 load_vec4(const device half* p, ulong i) { return float4(((const device half4*)p)[i]); }
inline float4 load_vec4(const constant half* p, ulong i) { return float4(((const constant half4*)p)[i]); }
inline float4 load_vec4(const device float* p, ulong i) { return ((const device float4*)p)[i]; }
inline float4 load_vec4(const constant float* p, ulong i) { return ((const constant float4*)p)[i]; }
inline uchar2 load_uchar2(const device uchar* p, ulong i) { return ((const device uchar2*)p)[i]; }
inline uchar2 load_uchar2(const constant uchar* p, ulong i) { return ((const constant uchar2*)p)[i]; }
inline char4 load_char4(const device int8_t* p, ulong i) { return char4(((const device packed_char4*)p)[i]); }
inline char4 load_char4(const constant int8_t* p, ulong i) { return char4(((const constant packed_char4*)p)[i]); }
inline uint load_uint(const device uchar* p, ulong i) { return ((const device uint*)p)[i]; }
inline uint load_uint(const constant uchar* p, ulong i) { return ((const constant uint*)p)[i]; }

inline void rht256_vec(thread float4& v0, thread float4& v1, uint lane) {
    const uint wi = lane >> 3;
    const uint sb = (lane & 7u) * 4u;
    v0 = rht_apply_signs4(v0, wi, sb);
    v1 = rht_apply_signs4(v1, wi + 4u, sb);
    const float4 h0 = fwht128_unnorm(v0, lane);
    const float4 h1 = fwht128_unnorm(v1, lane);
    v0 = rht_apply_signs4((h0 + h1) * 0.0625f, wi, sb);
    v1 = rht_apply_signs4((h0 - h1) * 0.0625f, wi + 4u, sb);
}
"""


def _format_c_array(vals: tuple[float, ...]) -> str:
    return ", ".join(f"{v:.7f}f" for v in vals)


_LM_HEADER = f"""
constant float LM2_LEVELS[4] = {{{_format_c_array(LM2_LEVELS)}}};
constant float LM2_MIDS[3] = {{{_format_c_array(LM2_MIDS)}}};

constant float LM3_LEVELS[8] = {{{_format_c_array(LM3_LEVELS)}}};
constant float LM3_MIDS[7] = {{{_format_c_array(LM3_MIDS)}}};

constant float LM4_LEVELS[16] = {{{_format_c_array(LM4_LEVELS)}}};
constant float LM4_MIDS[15] = {{{_format_c_array(LM4_MIDS)}}};

inline uint lm2_code(float z) {{
    return (uint)(z > LM2_MIDS[0]) + (uint)(z > LM2_MIDS[1]) + (uint)(z > LM2_MIDS[2]);
}}

inline uint lm3_code(float z) {{
    uint c = (z > LM3_MIDS[3]) ? 4u : 0u;
    c += (z > LM3_MIDS[c + 1u]) ? 2u : 0u;
    c += (z > LM3_MIDS[c]) ? 1u : 0u;
    return c;
}}

inline uint lm4_code(float z) {{
    uint c = (z > LM4_MIDS[7]) ? 8u : 0u;
    c += (z > LM4_MIDS[c + 3u]) ? 4u : 0u;
    c += (z > LM4_MIDS[c + 1u]) ? 2u : 0u;
    c += (z > LM4_MIDS[c]) ? 1u : 0u;
    return c;
}}
"""

_KERNEL_HEADER = _RHT_HEADER + _LM_HEADER


# ----------------------------------------------------------------------------- RHT / Quantize / Dequantize kernels

@lru_cache(maxsize=8)
def _make_rht_kernel(head_dim: int, in_t: str, out_t: str):
    if head_dim == 128:
        fn_body = f"""
    const float4 v = load_vec4(x + (ulong)row * 128u, lane);
    *((device {out_t}4*)(y + (ulong)row * 128u) + lane) = {out_t}4(rht128_vec(v, lane));
"""
    else:
        fn_body = f"""
    const auto p = x + (ulong)row * 256u;
    float4 v0 = load_vec4(p, lane), v1 = load_vec4(p, 32u + lane);
    rht256_vec(v0, v1, lane);
    device {out_t}4* o = (device {out_t}4*)(y + (ulong)row * 256u);
    o[lane] = {out_t}4(v0);
    o[32u + lane] = {out_t}4(v1);
"""
    fn_name = f"q35_rht_impl_{head_dim}_{in_t}_{out_t}"
    header = _RHT_HEADER + f"""
template <typename XPtr>
inline void {fn_name}(XPtr x, device {out_t}* y, uint row, uint lane) {{
{fn_body}
}}
"""
    return mx.fast.metal_kernel(
        name=f"q35_rht_{head_dim}_{in_t}_{out_t}",
        input_names=["x"],
        output_names=["y"],
        header=header,
        source=f"    {fn_name}(x, y, threadgroup_position_in_grid.y, thread_index_in_simdgroup);\n",
    )


def q35_rht(x: mx.array, out_dtype: mx.Dtype | None = None) -> mx.array:
    """Apply deterministic orthogonal involution RHT-128 or RHT-256 along the last axis."""
    head_dim = int(x.shape[-1])
    if head_dim not in VALID_HEAD_DIMS:
        raise ValueError(f"Unsupported head_dim={head_dim}; expected one of {VALID_HEAD_DIMS}")
    if out_dtype is None:
        out_dtype = x.dtype
    if x.dtype not in (mx.float16, mx.float32):
        x = x.astype(mx.float32)
    final_dtype = out_dtype
    if out_dtype not in (mx.float16, mx.float32):
        out_dtype = mx.float32
    if x.size == 0:
        return mx.zeros(x.shape, dtype=final_dtype)
    xc = mx.contiguous(x)
    n_rows = xc.size // head_dim
    in_t = _metal_type(xc.dtype)
    out_t = _metal_type(out_dtype)
    res = _make_rht_kernel(head_dim, in_t, out_t)(
        inputs=[xc.reshape(-1)],
        grid=(32, n_rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[xc.shape],
        output_dtypes=[out_dtype],
    )[0]
    return res if res.dtype == final_dtype else res.astype(final_dtype)


def _decode_row_expr(
    bits: int,
    head_dim: int,
    codes_ptr: str,
    scales_ptr: str,
    row_expr: str,
    v0_name: str,
    v1_name: str,
) -> str:
    """Metal snippet decoding one head vector `row_expr` into float4 `v0` (and `v1` for D=256)."""
    lines = []
    if bits == 16:
        lines.append(f"const auto _hp_{v0_name} = {codes_ptr} + (ulong)({row_expr}) * {head_dim}u;")
        lines.append(f"float4 {v0_name} = load_vec4(_hp_{v0_name}, lane);")
        if head_dim == 256:
            lines.append(f"float4 {v1_name} = load_vec4(_hp_{v0_name}, 32u + lane);")
        return "\n".join(lines)

    lines.append(f"const float _sc_{v0_name} = float({scales_ptr}[{row_expr}]);")
    if bits == 8:
        lines.append(f"const auto _cp_{v0_name} = {codes_ptr} + (ulong)({row_expr}) * {head_dim}u;")
        lines.append(f"float4 {v0_name} = float4(load_char4(_cp_{v0_name}, lane)) * _sc_{v0_name};")
        if head_dim == 256:
            lines.append(f"float4 {v1_name} = float4(load_char4(_cp_{v0_name}, 32u + lane)) * _sc_{v0_name};")
    elif bits == 4:
        stride = head_dim // 2
        lines.append(f"const auto _cp_{v0_name} = {codes_ptr} + (ulong)({row_expr}) * {stride}u;")
        lines.append(f"const uchar2 _p0_{v0_name} = load_uchar2(_cp_{v0_name}, lane);")
        lines.append(
            f"float4 {v0_name} = float4(LM4_LEVELS[_p0_{v0_name}.x & 0xFu], LM4_LEVELS[_p0_{v0_name}.x >> 4], "
            f"LM4_LEVELS[_p0_{v0_name}.y & 0xFu], LM4_LEVELS[_p0_{v0_name}.y >> 4]) * _sc_{v0_name};"
        )
        if head_dim == 256:
            lines.append(f"const uchar2 _p1_{v0_name} = load_uchar2(_cp_{v0_name}, 32u + lane);")
            lines.append(
                f"float4 {v1_name} = float4(LM4_LEVELS[_p1_{v0_name}.x & 0xFu], LM4_LEVELS[_p1_{v0_name}.x >> 4], "
                f"LM4_LEVELS[_p1_{v0_name}.y & 0xFu], LM4_LEVELS[_p1_{v0_name}.y >> 4]) * _sc_{v0_name};"
            )
    elif bits == 3:
        stride = (3 * head_dim) // 8
        pw = head_dim // 32  # uint32 words per plane: 4 for D=128, 8 for D=256
        lines.append(f"const auto _wp_{v0_name} = {codes_ptr} + (ulong)({row_expr}) * {stride}u;")
        lines.append(f"const uint _wi_{v0_name} = lane >> 3, _sh_{v0_name} = (lane & 7u) * 4u;")
        lines.append(
            f"const uint _n0_{v0_name} = (load_uint(_wp_{v0_name}, _wi_{v0_name}) >> _sh_{v0_name}) & 0xFu, "
            f"_n1_{v0_name} = (load_uint(_wp_{v0_name}, {pw}u + _wi_{v0_name}) >> _sh_{v0_name}) & 0xFu, "
            f"_n2_{v0_name} = (load_uint(_wp_{v0_name}, {2 * pw}u + _wi_{v0_name}) >> _sh_{v0_name}) & 0xFu;"
        )
        lines.append(
            f"float4 {v0_name} = float4("
            f"LM3_LEVELS[(_n0_{v0_name} & 1u) | ((_n1_{v0_name} & 1u) << 1) | ((_n2_{v0_name} & 1u) << 2)], "
            f"LM3_LEVELS[((_n0_{v0_name} >> 1) & 1u) | (((_n1_{v0_name} >> 1) & 1u) << 1) | (((_n2_{v0_name} >> 1) & 1u) << 2)], "
            f"LM3_LEVELS[((_n0_{v0_name} >> 2) & 1u) | (((_n1_{v0_name} >> 2) & 1u) << 1) | (((_n2_{v0_name} >> 2) & 1u) << 2)], "
            f"LM3_LEVELS[((_n0_{v0_name} >> 3) & 1u) | (((_n1_{v0_name} >> 3) & 1u) << 1) | (((_n2_{v0_name} >> 3) & 1u) << 2)]"
            f") * _sc_{v0_name};"
        )
        if head_dim == 256:
            lines.append(
                f"const uint _m0_{v0_name} = (load_uint(_wp_{v0_name}, 4u + _wi_{v0_name}) >> _sh_{v0_name}) & 0xFu, "
                f"_m1_{v0_name} = (load_uint(_wp_{v0_name}, 12u + _wi_{v0_name}) >> _sh_{v0_name}) & 0xFu, "
                f"_m2_{v0_name} = (load_uint(_wp_{v0_name}, 20u + _wi_{v0_name}) >> _sh_{v0_name}) & 0xFu;"
            )
            lines.append(
                f"float4 {v1_name} = float4("
                f"LM3_LEVELS[(_m0_{v0_name} & 1u) | ((_m1_{v0_name} & 1u) << 1) | ((_m2_{v0_name} & 1u) << 2)], "
                f"LM3_LEVELS[((_m0_{v0_name} >> 1) & 1u) | (((_m1_{v0_name} >> 1) & 1u) << 1) | (((_m2_{v0_name} >> 1) & 1u) << 2)], "
                f"LM3_LEVELS[((_m0_{v0_name} >> 2) & 1u) | (((_m1_{v0_name} >> 2) & 1u) << 1) | (((_m2_{v0_name} >> 2) & 1u) << 2)], "
                f"LM3_LEVELS[((_m0_{v0_name} >> 3) & 1u) | (((_m1_{v0_name} >> 3) & 1u) << 1) | (((_m2_{v0_name} >> 3) & 1u) << 2)]"
                f") * _sc_{v0_name};"
            )
    elif bits == 2:
        stride = head_dim // 4
        lines.append(f"const uchar _b0_{v0_name} = {codes_ptr}[(ulong)({row_expr}) * {stride}u + lane];")
        lines.append(
            f"float4 {v0_name} = float4(LM2_LEVELS[_b0_{v0_name} & 3u], LM2_LEVELS[(_b0_{v0_name} >> 2) & 3u], "
            f"LM2_LEVELS[(_b0_{v0_name} >> 4) & 3u], LM2_LEVELS[_b0_{v0_name} >> 6]) * _sc_{v0_name};"
        )
        if head_dim == 256:
            lines.append(f"const uchar _b1_{v0_name} = {codes_ptr}[(ulong)({row_expr}) * 64u + 32u + lane];")
            lines.append(
                f"float4 {v1_name} = float4(LM2_LEVELS[_b1_{v0_name} & 3u], LM2_LEVELS[(_b1_{v0_name} >> 2) & 3u], "
                f"LM2_LEVELS[(_b1_{v0_name} >> 4) & 3u], LM2_LEVELS[_b1_{v0_name} >> 6]) * _sc_{v0_name};"
            )
    else:
        raise ValueError(f"Unsupported KV bits={bits}")
    return "\n".join(lines)


@lru_cache(maxsize=16)
def _make_quantize_kernel(bits: int, head_dim: int):
    assert bits in (2, 3, 4, 8)
    assert head_dim in VALID_HEAD_DIMS
    if head_dim == 128:
        load_rot = """
    float4 v0 = load_vec4(x + (ulong)row * 128u, lane);
    v0 = rht128_vec(v0, lane);
"""
    else:
        load_rot = """
    const auto xp = x + (ulong)row * 256u;
    float4 v0 = load_vec4(xp, lane), v1 = load_vec4(xp, 32u + lane);
    rht256_vec(v0, v1, lane);
"""

    if bits == 8:
        scale_src = """
    float am = max(max(fabs(v0.x), fabs(v0.y)), max(fabs(v0.z), fabs(v0.w)));
""" + ("""
    am = max(am, max(max(fabs(v1.x), fabs(v1.y)), max(fabs(v1.z), fabs(v1.w))));
""" if head_dim == 256 else "") + """
    am = simd_max(am);
    const half sc_h = half(clamp(am * (1.0f / 127.0f), 1e-6f, 65504.0f));
    const float inv_sc = 1.0f / float(sc_h);
    if (lane == 0u) scales[row] = sc_h;
"""
    else:
        scale_src = """
    float ss = dot(v0, v0);
""" + ("""
    ss += dot(v1, v1);
""" if head_dim == 256 else "") + f"""
    ss = simd_sum(ss);
    const half sc_h = half(clamp(sqrt(ss * {1.0 / head_dim:.10f}f), 1e-6f, 65504.0f));
    const float inv_sc = 1.0f / float(sc_h);
    if (lane == 0u) scales[row] = sc_h;
"""

    if bits == 8:
        pack_src = f"""
    device char4* cp = (device char4*)(codes + (ulong)row * {head_dim}u);
    cp[lane] = char4(
        clamp(int(metal::rint(v0.x * inv_sc)), -127, 127),
        clamp(int(metal::rint(v0.y * inv_sc)), -127, 127),
        clamp(int(metal::rint(v0.z * inv_sc)), -127, 127),
        clamp(int(metal::rint(v0.w * inv_sc)), -127, 127)
    );
""" + ("""
    cp[32u + lane] = char4(
        clamp(int(metal::rint(v1.x * inv_sc)), -127, 127),
        clamp(int(metal::rint(v1.y * inv_sc)), -127, 127),
        clamp(int(metal::rint(v1.z * inv_sc)), -127, 127),
        clamp(int(metal::rint(v1.w * inv_sc)), -127, 127)
    );
""" if head_dim == 256 else "")
    elif bits == 4:
        stride = head_dim // 2
        pack_src = f"""
    device uchar2* cp = (device uchar2*)(codes + (ulong)row * {stride}u);
    const uint a0 = lm4_code(v0.x * inv_sc), a1 = lm4_code(v0.y * inv_sc);
    const uint a2 = lm4_code(v0.z * inv_sc), a3 = lm4_code(v0.w * inv_sc);
    cp[lane] = uchar2(uchar(a0 | (a1 << 4)), uchar(a2 | (a3 << 4)));
""" + ("""
    const uint b0 = lm4_code(v1.x * inv_sc), b1 = lm4_code(v1.y * inv_sc);
    const uint b2 = lm4_code(v1.z * inv_sc), b3 = lm4_code(v1.w * inv_sc);
    cp[32u + lane] = uchar2(uchar(b0 | (b1 << 4)), uchar(b2 | (b3 << 4)));
""" if head_dim == 256 else "")
    elif bits == 3:
        pb = head_dim // 8
        stride = 3 * pb
        pack_src = f"""
    device uchar* bp = codes + (ulong)row * {stride}u;
    const uint a0 = lm3_code(v0.x * inv_sc), a1 = lm3_code(v0.y * inv_sc);
    const uint a2 = lm3_code(v0.z * inv_sc), a3 = lm3_code(v0.w * inv_sc);
    const uint p0 = (a0 & 1u) | ((a1 & 1u) << 1) | ((a2 & 1u) << 2) | ((a3 & 1u) << 3);
    const uint p1 = ((a0 >> 1) & 1u) | (((a1 >> 1) & 1u) << 1) | (((a2 >> 1) & 1u) << 2) | (((a3 >> 1) & 1u) << 3);
    const uint p2 = ((a0 >> 2) & 1u) | (((a1 >> 2) & 1u) << 1) | (((a2 >> 2) & 1u) << 2) | (((a3 >> 2) & 1u) << 3);
    const uint q0 = simd_shuffle_xor(p0, 1u);
    const uint q1 = simd_shuffle_xor(p1, 1u);
    const uint q2 = simd_shuffle_xor(p2, 1u);
    if ((lane & 1u) == 0u) {{
        bp[lane >> 1] = uchar(p0 | (q0 << 4));
        bp[{pb}u + (lane >> 1)] = uchar(p1 | (q1 << 4));
        bp[{2 * pb}u + (lane >> 1)] = uchar(p2 | (q2 << 4));
    }}
""" + (f"""
    const uint b0 = lm3_code(v1.x * inv_sc), b1 = lm3_code(v1.y * inv_sc);
    const uint b2 = lm3_code(v1.z * inv_sc), b3 = lm3_code(v1.w * inv_sc);
    const uint r0 = (b0 & 1u) | ((b1 & 1u) << 1) | ((b2 & 1u) << 2) | ((b3 & 1u) << 3);
    const uint r1 = ((b0 >> 1) & 1u) | (((b1 >> 1) & 1u) << 1) | (((b2 >> 1) & 1u) << 2) | (((b3 >> 1) & 1u) << 3);
    const uint r2 = ((b0 >> 2) & 1u) | (((b1 >> 2) & 1u) << 1) | (((b2 >> 2) & 1u) << 2) | (((b3 >> 2) & 1u) << 3);
    const uint s0 = simd_shuffle_xor(r0, 1u);
    const uint s1 = simd_shuffle_xor(r1, 1u);
    const uint s2 = simd_shuffle_xor(r2, 1u);
    if ((lane & 1u) == 0u) {{
        bp[16u + (lane >> 1)] = uchar(r0 | (s0 << 4));
        bp[{pb + 16}u + (lane >> 1)] = uchar(r1 | (s1 << 4));
        bp[{2 * pb + 16}u + (lane >> 1)] = uchar(r2 | (s2 << 4));
    }}
""" if head_dim == 256 else "")
    else:  # bits == 2
        stride = head_dim // 4
        pack_src = f"""
    device uchar* bp = codes + (ulong)row * {stride}u;
    const uint a0 = lm2_code(v0.x * inv_sc), a1 = lm2_code(v0.y * inv_sc);
    const uint a2 = lm2_code(v0.z * inv_sc), a3 = lm2_code(v0.w * inv_sc);
    bp[lane] = uchar(a0 | (a1 << 2) | (a2 << 4) | (a3 << 6));
""" + ("""
    const uint b0 = lm2_code(v1.x * inv_sc), b1 = lm2_code(v1.y * inv_sc);
    const uint b2 = lm2_code(v1.z * inv_sc), b3 = lm2_code(v1.w * inv_sc);
    bp[32u + lane] = uchar(b0 | (b1 << 2) | (b2 << 4) | (b3 << 6));
""" if head_dim == 256 else "")

    out_code_t = "int8_t" if bits == 8 else "uchar"
    fn_name = f"q35_rht_lm_q{bits}_d{head_dim}_impl"
    header = _KERNEL_HEADER + f"""
template <typename XPtr>
inline void {fn_name}(XPtr x, device {out_code_t}* codes, device half* scales, uint row, uint lane) {{
    {load_rot}
    {scale_src}
    {pack_src}
}}
"""
    return mx.fast.metal_kernel(
        name=f"q35_rht_lm_q{bits}_d{head_dim}",
        input_names=["x"],
        output_names=["codes", "scales"],
        header=header,
        source=f"    {fn_name}(x, codes, scales, threadgroup_position_in_grid.y, thread_index_in_simdgroup);\n",
    )


def q35_rht_lm_quantize(x_f16: mx.array, bits: int) -> tuple[mx.array, mx.array]:
    """Rotate `x_f16` (*prefix, D) via RHT-D and quantize to LM{bits} (2,3,4), symmetric int8 (8), or FP16 (16)."""
    if bits not in VALID_KV_BITS:
        raise ValueError(f"Unsupported KV bits={bits}; expected one of {VALID_KV_BITS}")
    head_dim = int(x_f16.shape[-1])
    if head_dim not in VALID_HEAD_DIMS:
        raise ValueError(f"Unsupported head_dim={head_dim}; expected one of {VALID_HEAD_DIMS}")
    prefix = x_f16.shape[:-1]
    packed_dim = _packed_bytes_per_head(bits, head_dim)
    c_dtype = _codes_dtype(bits)
    if x_f16.size == 0:
        return mx.zeros((*prefix, packed_dim), dtype=c_dtype), mx.zeros((*prefix, 1), dtype=mx.float16)
    if bits == 16:
        return mx.contiguous(x_f16.astype(mx.float16)), mx.ones((*prefix, 1), dtype=mx.float16)

    xc = mx.contiguous(x_f16.astype(mx.float16))
    n_rows = xc.size // head_dim
    codes, scales = _make_quantize_kernel(bits, head_dim)(
        inputs=[xc.reshape(-1)],
        grid=(32, n_rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*prefix, packed_dim), (*prefix, 1)],
        output_dtypes=[c_dtype, mx.float16],
    )
    return codes, scales


@lru_cache(maxsize=32)
def _make_dequant_tile_kernel(bits: int, head_dim: int, unrotate: bool, out_t: str):
    assert bits in VALID_KV_BITS
    assert head_dim in VALID_HEAD_DIMS
    decode_body = _decode_row_expr(bits, head_dim, "codes", "scales", "src_row", "v0", "v1")
    unrot_src = ""
    if unrotate and bits < 16:
        unrot_src = "v0 = rht128_vec(v0, lane);" if head_dim == 128 else "rht256_vec(v0, v1, lane);"
    write_src = f"""
    device {out_t}4* op = (device {out_t}4*)(out + dst_row * {head_dim}u);
    op[lane] = {out_t}4(v0);
""" + (f"""
    op[32u + lane] = {out_t}4(v1);
""" if head_dim == 256 else "")

    fn_name = f"q35_rht_lm_deq{bits}_d{head_dim}_u{int(unrotate)}_{out_t}_impl"
    header = _KERNEL_HEADER + f"""
template <typename CPtr, typename SPtr, typename MPtr>
inline void {fn_name}(CPtr codes, SPtr scales, MPtr meta, device {out_t}* out, uint bh, uint t, uint lane) {{
    const uint start = meta[0], count = meta[1], cap = meta[2];
    const ulong src_row = (ulong)bh * cap + (start + t);
    const ulong dst_row = (ulong)bh * count + t;
    {decode_body}
    {unrot_src}
    {write_src}
}}
"""
    return mx.fast.metal_kernel(
        name=f"q35_rht_lm_deq{bits}_d{head_dim}_u{int(unrotate)}_{out_t}",
        input_names=["codes", "scales", "meta"],
        output_names=["out"],
        header=header,
        source=f"    {fn_name}(codes, scales, meta, out, threadgroup_position_in_grid.z, threadgroup_position_in_grid.y, thread_index_in_simdgroup);\n",
    )


@lru_cache(maxsize=128)
def _u32_meta(*vals: int) -> mx.array:
    return mx.array(list(vals), dtype=mx.uint32)


def _dequantize_tile(
    codes: mx.array,
    scales: mx.array | None,
    bits: int,
    head_dim: int,
    start: int,
    count: int,
    *,
    unrotate: bool = True,
    out_dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Dequantize tokens `[start : start + count]` from `codes` (*prefix_bh, cap, ...) without Python slicing."""
    if bits not in VALID_KV_BITS:
        raise ValueError(f"Unsupported KV bits={bits}")
    if head_dim not in VALID_HEAD_DIMS:
        raise ValueError(f"Unsupported head_dim={head_dim}")
    cap = int(codes.shape[-2])
    prefix_bh = codes.shape[:-2]
    if count == 0:
        return mx.zeros((*prefix_bh, 0, head_dim), dtype=out_dtype)
    if bits == 16 and start == 0 and count == cap:
        return mx.contiguous(codes.astype(out_dtype))
    n_bh = math.prod(prefix_bh) if prefix_bh else 1
    codes_c = mx.contiguous(codes)
    scales_c = (
        mx.contiguous(scales.astype(mx.float16))
        if (scales is not None and bits < 16)
        else mx.zeros((1,), dtype=mx.float16)
    )
    out_t = _metal_type(out_dtype)
    return _make_dequant_tile_kernel(bits, head_dim, bool(unrotate), out_t)(
        inputs=[codes_c.reshape(-1), scales_c.reshape(-1), _u32_meta(start, count, cap)],
        grid=(32, count, n_bh),
        threadgroup=(32, 1, 1),
        output_shapes=[(*prefix_bh, count, head_dim)],
        output_dtypes=[out_dtype],
    )[0]


def q35_rht_lm_dequantize(
    codes: mx.array,
    scales: mx.array | None,
    bits: int,
    head_dim: int,
) -> mx.array:
    """Dequantize packed RHT+LM codes back to FP16 in the original unrotated space."""
    if bits not in VALID_KV_BITS:
        raise ValueError(f"Unsupported KV bits={bits}; expected one of {VALID_KV_BITS}")
    if head_dim not in VALID_HEAD_DIMS:
        raise ValueError(f"Unsupported head_dim={head_dim}; expected one of {VALID_HEAD_DIMS}")
    prefix = codes.shape[:-1]
    if codes.size == 0:
        return mx.zeros((*prefix, head_dim), dtype=mx.float16)
    if bits == 16:
        return mx.contiguous(codes.astype(mx.float16))
    # Reshape (*prefix, ...) -> (n_rows, 1, ...) so _dequantize_tile processes every row in one launch.
    n_rows = codes.size // _packed_bytes_per_head(bits, head_dim)
    codes_3d = mx.contiguous(codes).reshape(n_rows, 1, -1)
    scales_3d = mx.contiguous(scales.astype(mx.float16)).reshape(n_rows, 1, 1)
    out = _dequantize_tile(codes_3d, scales_3d, bits, head_dim, 0, 1, unrotate=True, out_dtype=mx.float16)
    return out.reshape(*prefix, head_dim)


# ----------------------------------------------------------------------------- Fused Split-K Decode SDPA

def _choose_decode_splits(n_body: int, base_tgs: int) -> int:
    if n_body < 256:
        return 1
    want = max(1, (128 + base_tgs - 1) // base_tgs)
    max_by_len = max(1, n_body // 64)
    s = 1
    while s < want and s * 2 <= max_by_len and s < 64:
        s *= 2
    if n_body >= 4096 and s < 8 and base_tgs <= 8:
        s = min(8, max_by_len)
    if n_body >= 16384 and s < 16 and base_tgs <= 8:
        s = min(16, max_by_len)
    return s


def _online_update_snippet(head_dim: int, m_var: str, l_var: str, a0_var: str, a1_var: str, score_var: str, v0_var: str, v1_var: str) -> str:
    acc_assign_first = f"{a0_var} = {v0_var};" + (f" {a1_var} = {v1_var};" if head_dim == 256 else "")
    acc_update = f"{a0_var} = {a0_var} * _alpha + _pw * {v0_var};" + (
        f" {a1_var} = {a1_var} * _alpha + _pw * {v1_var};" if head_dim == 256 else ""
    )
    return f"""
    if ({l_var} == 0.0f) {{
        {m_var} = {score_var};
        {l_var} = 1.0f;
        {acc_assign_first}
    }} else {{
        const float _m_new = max({m_var}, {score_var});
        const float _alpha = exp2({m_var} - _m_new);
        const float _pw = exp2({score_var} - _m_new);
        {l_var} = {l_var} * _alpha + _pw;
        {acc_update}
        {m_var} = _m_new;
    }}
"""


def _merge_states_snippet(head_dim: int, m_dst: str, l_dst: str, a0_dst: str, a1_dst: str, m_src: str, l_src: str, a0_src: str, a1_src: str) -> str:
    acc_copy = f"{a0_dst} = {a0_src};" + (f" {a1_dst} = {a1_src};" if head_dim == 256 else "")
    acc_blend = f"{a0_dst} = {a0_dst} * _al + {a0_src} * _be;" + (
        f" {a1_dst} = {a1_dst} * _al + {a1_src} * _be;" if head_dim == 256 else ""
    )
    return f"""
    if ({l_src} > 0.0f) {{
        if ({l_dst} == 0.0f) {{
            {m_dst} = {m_src};
            {l_dst} = {l_src};
            {acc_copy}
        }} else {{
            const float _m_new = max({m_dst}, {m_src});
            const float _al = exp2({m_dst} - _m_new);
            const float _be = exp2({m_src} - _m_new);
            {l_dst} = {l_dst} * _al + {l_src} * _be;
            {acc_blend}
            {m_dst} = _m_new;
        }}
    }}
"""


@lru_cache(maxsize=64)
def _make_sdpa_decode_kernel(head_dim: int, k_bits: int, v_bits: int, g_cap: int, split_k: bool, q_t: str):
    assert head_dim in VALID_HEAD_DIMS
    assert k_bits in VALID_KV_BITS and v_bits in VALID_KV_BITS
    assert 1 <= g_cap <= 4

    k_decode = _decode_row_expr(k_bits, head_dim, "k_codes", "k_sc", "row_j", "kv0", "kv1")
    v_decode = _decode_row_expr(v_bits, head_dim, "v_codes", "v_sc", "row_j", "vv0", "vv1")

    q_dot_k_sink = "simd_sum(dot(q0[g], load_vec4(skp, lane)))" if head_dim == 128 else "simd_sum(dot(q0[g], load_vec4(skp, lane)) + dot(q1[g], load_vec4(skp, 32u + lane)))"
    q_dot_k_body = "simd_sum(dot(q0[g], kv0))" if head_dim == 128 else "simd_sum(dot(q0[g], kv0) + dot(q1[g], kv1))"
    rht_q_expr = "q0[g] = rht128_vec(q0[g], lane);" if head_dim == 128 else "rht256_vec(q0[g], q1[g], lane);"
    unrot_acc_expr = "ab0 = rht128_vec(ab0, lane);" if head_dim == 128 else "rht256_vec(ab0, ab1, lane);"

    sink_block = ""
    if not split_k:
        sink_reduce = _merge_states_snippet(head_dim, "ms_all", "ls_all", "as0", "as1", "tg_m[s]", "tg_l[s]", "tg_a0[s][lane]", "tg_a1[s][lane]")
        sink_update = _online_update_snippet(head_dim, "m_s[g]", "l_s[g]", "as0_loc[g]", "as1_loc[g]", "sc_s", "vs0", "vs1")
        sink_block = f"""
    if (n_sink > 0u) {{
        float m_s[{g_cap}], l_s[{g_cap}];
        float4 as0_loc[{g_cap}];
        {"float4 as1_loc[" + str(g_cap) + "];" if head_dim == 256 else ""}
        for (uint g = 0u; g < {g_cap}u; g++) {{
            m_s[g] = -1e30f; l_s[g] = 0.0f;
            as0_loc[g] = float4(0.0f);
            {"as1_loc[g] = float4(0.0f);" if head_dim == 256 else ""}
        }}
        for (uint j = simd; j < n_sink; j += 4u) {{
            const ulong row_s = (ulong)bh_kv * cap_sink + j;
            const auto skp = k_sink + row_s * {head_dim}u;
            const auto svp = v_sink + row_s * {head_dim}u;
            const float4 vs0 = load_vec4(svp, lane);
            {"const float4 vs1 = load_vec4(svp, 32u + lane);" if head_dim == 256 else ""}
            for (uint g = 0u; g < g_cnt; g++) {{
                const float sc_s = {q_dot_k_sink};
                {sink_update}
            }}
        }}
        for (uint g = 0u; g < g_cnt; g++) {{
            if (lane == 0u) {{ tg_m[simd] = m_s[g]; tg_l[simd] = l_s[g]; }}
            tg_a0[simd][lane] = as0_loc[g];
            {"tg_a1[simd][lane] = as1_loc[g];" if head_dim == 256 else ""}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd == 0u) {{
                float ms_all = -1e30f, ls_all = 0.0f;
                float4 as0 = float4(0.0f);
                {"float4 as1 = float4(0.0f);" if head_dim == 256 else ""}
                for (uint s = 0u; s < 4u; s++) {{
                    {sink_reduce}
                }}
                if (lane == 0u) {{ tg_m_sk[g] = ms_all; tg_l_sk[g] = ls_all; }}
                tg_a_sk0[g][lane] = as0;
                {"tg_a_sk1[g][lane] = as1;" if head_dim == 256 else ""}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (n_body == 0u) {{
            if (simd == 0u) {{
                for (uint g = 0u; g < g_cnt; g++) {{
                    const uint h = kvh * G + g0 + g;
                    const float inv_l = (tg_l_sk[g] > 0.0f) ? (1.0f / tg_l_sk[g]) : 0.0f;
                    device {q_t}4* op = (device {q_t}4*)(out + ((ulong)b * H + h) * {head_dim}u);
                    op[lane] = {q_t}4(tg_a_sk0[g][lane] * inv_l);
                    {"op[32u + lane] = " + q_t + "4(tg_a_sk1[g][lane] * inv_l);" if head_dim == 256 else ""}
                }}
            }}
            return;
        }}
    }}
"""

    q_rot_block = ""
    if k_bits < 16:
        q_rot_block = f"""
    for (uint g = 0u; g < g_cnt; g++) {{
        {rht_q_expr}
    }}
"""

    body_update = _online_update_snippet(head_dim, "m_loc[g]", "l_loc[g]", "ab0_loc[g]", "ab1_loc[g]", "sc_b", "vv0", "vv1")
    body_reduce = _merge_states_snippet(head_dim, "mb_all", "lb_all", "ab0", "ab1", "tg_m[s]", "tg_l[s]", "tg_a0[s][lane]", "tg_a1[s][lane]")
    sink_merge = _merge_states_snippet(head_dim, "mb_all", "lb_all", "ab0", "ab1", "tg_m_sk[g]", "tg_l_sk[g]", "tg_a_sk0[g][lane]", "tg_a_sk1[g][lane]")

    if split_k:
        epilogue = f"""
    for (uint g = 0u; g < g_cnt; g++) {{
        if (lane == 0u) {{ tg_m[simd] = m_loc[g]; tg_l[simd] = l_loc[g]; }}
        tg_a0[simd][lane] = ab0_loc[g];
        {"tg_a1[simd][lane] = ab1_loc[g];" if head_dim == 256 else ""}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {{
            float mb_all = -1e30f, lb_all = 0.0f;
            float4 ab0 = float4(0.0f);
            {"float4 ab1 = float4(0.0f);" if head_dim == 256 else ""}
            for (uint s = 0u; s < 4u; s++) {{
                {body_reduce}
            }}
            const uint h = kvh * G + g0 + g;
            const ulong out_idx = ((ulong)b * H + h) * n_splits + split_id;
            if (lane == 0u) {{
                part_m[out_idx] = mb_all;
                part_l[out_idx] = lb_all;
            }}
            device float4* pop = (device float4*)(part_o + out_idx * {head_dim}u);
            pop[lane] = ab0;
            {"pop[32u + lane] = ab1;" if head_dim == 256 else ""}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
"""
    else:
        unrot_cond = f"if (lb_all > 0.0f) {{ {unrot_acc_expr} }}" if v_bits < 16 else ""
        epilogue = f"""
    for (uint g = 0u; g < g_cnt; g++) {{
        if (lane == 0u) {{ tg_m[simd] = m_loc[g]; tg_l[simd] = l_loc[g]; }}
        tg_a0[simd][lane] = ab0_loc[g];
        {"tg_a1[simd][lane] = ab1_loc[g];" if head_dim == 256 else ""}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {{
            float mb_all = -1e30f, lb_all = 0.0f;
            float4 ab0 = float4(0.0f);
            {"float4 ab1 = float4(0.0f);" if head_dim == 256 else ""}
            for (uint s = 0u; s < 4u; s++) {{
                {body_reduce}
            }}
            {unrot_cond}
            if (n_sink > 0u) {{
                {sink_merge}
            }}
            const uint h = kvh * G + g0 + g;
            const float inv_l = (lb_all > 0.0f) ? (1.0f / lb_all) : 0.0f;
            device {q_t}4* op = (device {q_t}4*)(out + ((ulong)b * H + h) * {head_dim}u);
            op[lane] = {q_t}4(ab0 * inv_l);
            {"op[32u + lane] = " + q_t + "4(ab1 * inv_l);" if head_dim == 256 else ""}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
"""

    fn_name = f"q35_sdpa_d{head_dim}_k{k_bits}_v{v_bits}_g{g_cap}_sp{int(split_k)}_{q_t}_impl"
    out_params = (
        "device float* part_m, device float* part_l, device float* part_o"
        if split_k
        else f"device {q_t}* out"
    )
    out_call_args = "part_m, part_l, part_o" if split_k else "out"
    tg_params = (
        "threadgroup float* tg_m, threadgroup float* tg_l, "
        "threadgroup float4 (*tg_a0)[32]"
        + (", threadgroup float4 (*tg_a1)[32]" if head_dim == 256 else "")
        + (
            f", threadgroup float* tg_m_sk, threadgroup float* tg_l_sk, "
            f"threadgroup float4 (*tg_a_sk0)[32]"
            + (", threadgroup float4 (*tg_a_sk1)[32]" if head_dim == 256 else "")
            if not split_k
            else ""
        )
    )
    tg_decls = (
        "    threadgroup float tg_m[4], tg_l[4];\n"
        "    threadgroup float4 tg_a0[4][32];\n"
        + ("    threadgroup float4 tg_a1[4][32];\n" if head_dim == 256 else "")
        + (
            f"    threadgroup float tg_m_sk[{g_cap}], tg_l_sk[{g_cap}];\n"
            f"    threadgroup float4 tg_a_sk0[{g_cap}][32];\n"
            + (f"    threadgroup float4 tg_a_sk1[{g_cap}][32];\n" if head_dim == 256 else "")
            if not split_k
            else ""
        )
    )
    tg_call_args = (
        "tg_m, tg_l, tg_a0"
        + (", tg_a1" if head_dim == 256 else "")
        + (
            ", tg_m_sk, tg_l_sk, tg_a_sk0" + (", tg_a_sk1" if head_dim == 256 else "")
            if not split_k
            else ""
        )
    )
    header = _KERNEL_HEADER + f"""
template <typename QPtr, typename KSPtr, typename VSPtr, typename KCPtr, typename KScPtr, typename VCPtr, typename VScPtr, typename MPtr, typename ScArrPtr>
inline void {fn_name}(
    QPtr q, KSPtr k_sink, VSPtr v_sink, KCPtr k_codes, KScPtr k_sc, VCPtr v_codes, VScPtr v_sc,
    MPtr meta, ScArrPtr scale_arr, {out_params}, {tg_params},
    uint gid, uint simd, uint lane
) {{
    const uint H = meta[1], H_kv = meta[2], n_sink = meta[3], cap_sink = meta[4];
    const uint n_body = meta[5], cap_body = meta[6], n_splits = meta[7], n_g_chunks = meta[8];
    const uint G = H / H_kv;

    const uint split_id = gid % n_splits;
    const uint bh_gc = gid / n_splits;
    const uint gc = bh_gc % n_g_chunks;
    const uint bh_kv = bh_gc / n_g_chunks;
    const uint b = bh_kv / H_kv;
    const uint kvh = bh_kv - b * H_kv;
    const uint g0 = gc * {g_cap}u;
    const uint g_cnt = min({g_cap}u, G - g0);
    const float sc_log2 = scale_arr[0];

    float4 q0[{g_cap}];
    {"float4 q1[" + str(g_cap) + "];" if head_dim == 256 else ""}
    for (uint g = 0u; g < g_cnt; g++) {{
        const uint h = kvh * G + g0 + g;
        const auto qp = q + ((ulong)b * H + h) * {head_dim}u;
        q0[g] = load_vec4(qp, lane) * sc_log2;
        {"q1[g] = load_vec4(qp, 32u + lane) * sc_log2;" if head_dim == 256 else ""}
    }}

    {sink_block}
    {q_rot_block}

    const uint chunk = (n_body + n_splits - 1u) / n_splits;
    const uint sbeg = min(n_body, split_id * chunk);
    const uint send = min(n_body, sbeg + chunk);

    float m_loc[{g_cap}], l_loc[{g_cap}];
    float4 ab0_loc[{g_cap}];
    {"float4 ab1_loc[" + str(g_cap) + "];" if head_dim == 256 else ""}
    for (uint g = 0u; g < {g_cap}u; g++) {{
        m_loc[g] = -1e30f; l_loc[g] = 0.0f;
        ab0_loc[g] = float4(0.0f);
        {"ab1_loc[g] = float4(0.0f);" if head_dim == 256 else ""}
    }}

    for (uint j = sbeg + simd; j < send; j += 4u) {{
        const ulong row_j = (ulong)bh_kv * cap_body + j;
        {k_decode}
        {v_decode}
        for (uint g = 0u; g < g_cnt; g++) {{
            const float sc_b = {q_dot_k_body};
            {body_update}
        }}
    }}

    {epilogue}
}}
"""
    out_names = ["part_m", "part_l", "part_o"] if split_k else ["out"]
    return mx.fast.metal_kernel(
        name=f"q35_sdpa_d{head_dim}_k{k_bits}_v{v_bits}_g{g_cap}_sp{int(split_k)}_{q_t}",
        input_names=["q", "k_sink", "v_sink", "k_codes", "k_sc", "v_codes", "v_sc", "meta", "scale_arr"],
        output_names=out_names,
        header=header,
        source=(
            f"{tg_decls}"
            f"    {fn_name}(q, k_sink, v_sink, k_codes, k_sc, v_codes, v_sc, meta, scale_arr, {out_call_args}, {tg_call_args}, "
            "threadgroup_position_in_grid.y, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);\n"
        ),
    )


@lru_cache(maxsize=16)
def _make_sdpa_reduce_kernel(head_dim: int, v_bits: int, q_t: str):
    assert head_dim in VALID_HEAD_DIMS
    assert v_bits in VALID_KV_BITS

    body_reduce = _merge_states_snippet(head_dim, "m_body", "l_body", "ab0", "ab1", "tg_m[s]", "tg_l[s]", "tg_a0[s][lane]", "tg_a1[s][lane]")
    split_merge = _merge_states_snippet(head_dim, "m_loc", "l_loc", "a0_loc", "a1_loc", "part_m[idx]", "part_l[idx]", "load_vec4(pop, lane)", "load_vec4(pop, 32u + lane)")
    sink_update = _online_update_snippet(head_dim, "m_sk_loc", "l_sk_loc", "as0_loc", "as1_loc", "sc_s", "vs0", "vs1")
    sink_reduce = _merge_states_snippet(head_dim, "m_sk", "l_sk", "as0", "as1", "tg_m[s]", "tg_l[s]", "tg_a0[s][lane]", "tg_a1[s][lane]")
    final_merge = _merge_states_snippet(head_dim, "m_body", "l_body", "ab0", "ab1", "m_sk", "l_sk", "as0", "as1")

    q_dot_k_sink = "simd_sum(dot(qu0, load_vec4(skp, lane)))" if head_dim == 128 else "simd_sum(dot(qu0, load_vec4(skp, lane)) + dot(qu1, load_vec4(skp, 32u + lane)))"
    unrot_acc_expr = (
        ("ab0 = rht128_vec(ab0, lane);" if head_dim == 128 else "rht256_vec(ab0, ab1, lane);")
        if v_bits < 16
        else ""
    )

    fn_name = f"q35_sdpa_reduce_d{head_dim}_v{v_bits}_{q_t}_impl"
    tg_params = (
        "threadgroup float* tg_m, threadgroup float* tg_l, "
        "threadgroup float4 (*tg_a0)[32]"
        + (", threadgroup float4 (*tg_a1)[32]" if head_dim == 256 else "")
    )
    tg_decls = (
        "    threadgroup float tg_m[4], tg_l[4];\n"
        "    threadgroup float4 tg_a0[4][32];\n"
        + ("    threadgroup float4 tg_a1[4][32];\n" if head_dim == 256 else "")
    )
    tg_call_args = "tg_m, tg_l, tg_a0" + (", tg_a1" if head_dim == 256 else "")
    header = _RHT_HEADER + f"""
template <typename PMPtr, typename PLPtr, typename POPtr, typename QPtr, typename KSPtr, typename VSPtr, typename MPtr, typename ScArrPtr>
inline void {fn_name}(
    PMPtr part_m, PLPtr part_l, POPtr part_o, QPtr q, KSPtr k_sink, VSPtr v_sink,
    MPtr meta, ScArrPtr scale_arr, device {q_t}* out, {tg_params},
    uint bh, uint simd, uint lane
) {{
    const uint H = meta[1], H_kv = meta[2], n_sink = meta[3], cap_sink = meta[4], n_splits = meta[7];
    const uint G = H / H_kv;
    const uint b = bh / H;
    const uint h = bh - b * H;
    const uint kvh = h / G;
    const uint bh_kv = b * H_kv + kvh;
    const float sc_log2 = scale_arr[0];

    float m_loc = -1e30f, l_loc = 0.0f;
    float4 a0_loc = float4(0.0f);
    {"float4 a1_loc = float4(0.0f);" if head_dim == 256 else ""}
    for (uint s = simd; s < n_splits; s += 4u) {{
        const ulong idx = (ulong)bh * n_splits + s;
        const auto pop = part_o + idx * {head_dim}u;
        {split_merge}
    }}

    if (lane == 0u) {{ tg_m[simd] = m_loc; tg_l[simd] = l_loc; }}
    tg_a0[simd][lane] = a0_loc;
    {"tg_a1[simd][lane] = a1_loc;" if head_dim == 256 else ""}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_body = -1e30f, l_body = 0.0f;
    float4 ab0 = float4(0.0f);
    {"float4 ab1 = float4(0.0f);" if head_dim == 256 else ""}
    if (simd == 0u) {{
        for (uint s = 0u; s < 4u; s++) {{
            {body_reduce}
        }}
        if (l_body > 0.0f) {{
            {unrot_acc_expr}
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n_sink > 0u) {{
        const auto qp = q + (ulong)bh * {head_dim}u;
        const float4 qu0 = load_vec4(qp, lane) * sc_log2;
        {"const float4 qu1 = load_vec4(qp, 32u + lane) * sc_log2;" if head_dim == 256 else ""}
        float m_sk_loc = -1e30f, l_sk_loc = 0.0f;
        float4 as0_loc = float4(0.0f);
        {"float4 as1_loc = float4(0.0f);" if head_dim == 256 else ""}
        for (uint j = simd; j < n_sink; j += 4u) {{
            const ulong row_s = (ulong)bh_kv * cap_sink + j;
            const auto skp = k_sink + row_s * {head_dim}u;
            const auto svp = v_sink + row_s * {head_dim}u;
            const float4 vs0 = load_vec4(svp, lane);
            {"const float4 vs1 = load_vec4(svp, 32u + lane);" if head_dim == 256 else ""}
            const float sc_s = {q_dot_k_sink};
            {sink_update}
        }}
        if (lane == 0u) {{ tg_m[simd] = m_sk_loc; tg_l[simd] = l_sk_loc; }}
        tg_a0[simd][lane] = as0_loc;
        {"tg_a1[simd][lane] = as1_loc;" if head_dim == 256 else ""}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {{
            float m_sk = -1e30f, l_sk = 0.0f;
            float4 as0 = float4(0.0f);
            {"float4 as1 = float4(0.0f);" if head_dim == 256 else ""}
            for (uint s = 0u; s < 4u; s++) {{
                {sink_reduce}
            }}
            {final_merge}
        }}
    }}

    if (simd == 0u) {{
        const float inv_l = (l_body > 0.0f) ? (1.0f / l_body) : 0.0f;
        device {q_t}4* op = (device {q_t}4*)(out + (ulong)bh * {head_dim}u);
        op[lane] = {q_t}4(ab0 * inv_l);
        {"op[32u + lane] = " + q_t + "4(ab1 * inv_l);" if head_dim == 256 else ""}
    }}
}}
"""
    return mx.fast.metal_kernel(
        name=f"q35_sdpa_reduce_d{head_dim}_v{v_bits}_{q_t}",
        input_names=["part_m", "part_l", "part_o", "q", "k_sink", "v_sink", "meta", "scale_arr"],
        output_names=["out"],
        header=header,
        source=(
            f"{tg_decls}"
            f"    {fn_name}(part_m, part_l, part_o, q, k_sink, v_sink, meta, scale_arr, out, {tg_call_args}, "
            "threadgroup_position_in_grid.y, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);\n"
        ),
    )


def q35_rht_lm_sdpa_decode(
    q: mx.array,
    k_sink: mx.array | None,
    v_sink: mx.array | None,
    k_codes: mx.array | None,
    k_sc: mx.array | None,
    v_codes: mx.array | None,
    v_sc: mx.array | None,
    k_bits: int,
    v_bits: int,
    n_sink: int | None = None,
    n_body: int | None = None,
    scale: float | None = None,
) -> mx.array:
    """Single-token GQA decode attention over FP16 sinks + Hadamard-space Lloyd-Max body."""
    if k_bits not in VALID_KV_BITS or v_bits not in VALID_KV_BITS:
        raise ValueError(f"Unsupported KV bits: k_bits={k_bits}, v_bits={v_bits}")
    squeeze_l = q.ndim == 3
    if squeeze_l:
        q = q[:, :, None, :]
    if q.ndim != 4:
        raise ValueError(f"Expected q with ndim 3 or 4, got shape {q.shape}")
    B, H, L_q, head_dim = q.shape
    if head_dim not in VALID_HEAD_DIMS:
        raise ValueError(f"Unsupported head_dim={head_dim}; expected one of {VALID_HEAD_DIMS}")
    if L_q != 1:
        return q35_rht_lm_sdpa_prefill(
            q, k_sink, v_sink, k_codes, k_sc, v_codes, v_sc,
            k_bits=k_bits, v_bits=v_bits, n_sink=n_sink, n_body=n_body, scale=scale, mask=None,
        )

    if k_sink is not None and v_sink is not None:
        H_kv = int(k_sink.shape[1])
        cap_sink = int(k_sink.shape[2])
        n_sink_i = cap_sink if n_sink is None else int(n_sink)
    else:
        H_kv = int(k_codes.shape[1]) if k_codes is not None else H
        cap_sink = 1
        n_sink_i = 0

    if k_codes is not None and v_codes is not None:
        H_kv = int(k_codes.shape[1])
        cap_body = int(k_codes.shape[2])
        n_body_i = cap_body if n_body is None else int(n_body)
    else:
        cap_body = 1
        n_body_i = 0

    if H % H_kv != 0:
        raise ValueError(f"H={H} must be divisible by H_kv={H_kv}")
    if n_sink_i == 0 and n_body_i == 0:
        out = mx.zeros_like(q)
        return out[:, :, 0, :] if squeeze_l else out

    out_dtype = q.dtype
    if q.dtype not in (mx.float16, mx.float32):
        q = q.astype(mx.float32)
    qc = mx.contiguous(q)
    q_t = _metal_type(qc.dtype)

    G = H // H_kv
    g_cap = min(G, 4)
    n_g_chunks = (G + g_cap - 1) // g_cap
    n_splits = _choose_decode_splits(n_body_i, B * H_kv * n_g_chunks)
    split_k = n_splits > 1

    scale_val = (head_dim ** -0.5) if scale is None else float(scale)
    scale_arr = mx.array([scale_val * (1.0 / math.log(2.0))], dtype=mx.float32)

    dummy_f16 = mx.zeros((1,), dtype=mx.float16)
    ks_arr = mx.contiguous(k_sink.astype(mx.float16)).reshape(-1) if n_sink_i > 0 else dummy_f16
    vs_arr = mx.contiguous(v_sink.astype(mx.float16)).reshape(-1) if n_sink_i > 0 else dummy_f16
    if n_body_i > 0:
        kc_arr = mx.contiguous(k_codes).reshape(-1)
        vc_arr = mx.contiguous(v_codes).reshape(-1)
        ksc_arr = mx.contiguous(k_sc.astype(mx.float16)).reshape(-1) if (k_bits < 16 and k_sc is not None) else dummy_f16
        vsc_arr = mx.contiguous(v_sc.astype(mx.float16)).reshape(-1) if (v_bits < 16 and v_sc is not None) else dummy_f16
    else:
        kc_arr = mx.zeros((1,), dtype=_codes_dtype(k_bits))
        vc_arr = mx.zeros((1,), dtype=_codes_dtype(v_bits))
        ksc_arr = dummy_f16
        vsc_arr = dummy_f16

    meta = _u32_meta(B, H, H_kv, n_sink_i, max(1, cap_sink), n_body_i, max(1, cap_body), n_splits, n_g_chunks)
    kern = _make_sdpa_decode_kernel(head_dim, k_bits, v_bits, g_cap, split_k, q_t)

    if not split_k:
        out = kern(
            inputs=[qc.reshape(-1), ks_arr, vs_arr, kc_arr, ksc_arr, vc_arr, vsc_arr, meta, scale_arr],
            grid=(128, B * H_kv * n_g_chunks, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[qc.shape],
            output_dtypes=[qc.dtype],
        )[0]
    else:
        pm, pl, po = kern(
            inputs=[qc.reshape(-1), ks_arr, vs_arr, kc_arr, ksc_arr, vc_arr, vsc_arr, meta, scale_arr],
            grid=(128, B * H_kv * n_g_chunks * n_splits, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(B, H, n_splits), (B, H, n_splits), (B, H, n_splits, head_dim)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        out = _make_sdpa_reduce_kernel(head_dim, v_bits, q_t)(
            inputs=[pm.reshape(-1), pl.reshape(-1), po.reshape(-1), qc.reshape(-1), ks_arr, vs_arr, meta, scale_arr],
            grid=(128, B * H, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[qc.shape],
            output_dtypes=[qc.dtype],
        )[0]

    if out.dtype != out_dtype:
        out = out.astype(out_dtype)
    return out[:, :, 0, :] if squeeze_l else out


# ----------------------------------------------------------------------------- Memory-bounded Tiled Prefill SDPA

def _normalize_mask_5d(mask: mx.array, B: int, H_kv: int, G: int, L_q: int, total_len: int) -> mx.array:
    if mask.ndim == 2:
        m4 = mask[None, None, :, :]
    elif mask.ndim == 3:
        m4 = mask[:, None, :, :]
    elif mask.ndim == 4:
        m4 = mask
    else:
        raise ValueError(f"Unsupported mask ndim={mask.ndim}; expected 2, 3, or 4")
    if m4.shape[1] == 1:
        return m4[:, :, None, :, :]
    if m4.shape[1] == H_kv * G:
        return m4.reshape(m4.shape[0], H_kv, G, m4.shape[2], m4.shape[3])
    if m4.shape[1] == H_kv:
        return m4[:, :, None, :, :]
    raise ValueError(f"Incompatible mask head count {m4.shape[1]} for H={H_kv * G}, H_kv={H_kv}")


def _online_tile_step(
    q_chunk: mx.array,  # [B, H_kv, G, q_len, D] float32 (scaled by scale)
    k_tile: mx.array,   # [B, H_kv, t_len, D] float32
    v_tile: mx.array,   # [B, H_kv, t_len, D] float32
    k_global0: int,
    q_start: int,
    q_pos_offset: int,
    mask_mode: str | None,
    mask_5d: mx.array | None,
    m_acc: mx.array,    # [B, H_kv, G, q_len, 1] float32
    l_acc: mx.array,    # [B, H_kv, G, q_len, 1] float32
    o_acc: mx.array,    # [B, H_kv, G, q_len, D] float32
) -> tuple[mx.array, mx.array, mx.array]:
    q_len = int(q_chunk.shape[-2])
    t_len = int(k_tile.shape[-2])
    scores = mx.matmul(q_chunk, k_tile[:, :, None, :, :].swapaxes(-1, -2))  # [B, H_kv, G, q_len, t_len]

    valid: mx.array | None = None
    if mask_mode == "causal":
        q_pos_min = q_pos_offset + q_start
        q_pos_max = q_pos_min + q_len - 1
        k_pos_max = k_global0 + t_len - 1
        if k_pos_max > q_pos_min:
            q_idx = mx.arange(q_pos_min, q_pos_max + 1)[:, None]
            k_idx = mx.arange(k_global0, k_pos_max + 1)[None, :]
            valid = (q_idx >= k_idx)[None, None, None, :, :]
            scores = mx.where(valid, scores, -1e30)
    elif mask_5d is not None:
        q_sl = slice(q_start, q_start + q_len) if mask_5d.shape[3] > 1 else slice(None)
        m_sl = mask_5d[:, :, :, q_sl, k_global0 : k_global0 + t_len]
        if m_sl.dtype == mx.bool_:
            valid = m_sl
            scores = mx.where(valid, scores, -1e30)
        else:
            scores = scores + m_sl.astype(mx.float32)
            valid = scores > -1e20

    m_tile = mx.max(scores, axis=-1, keepdims=True)
    if valid is not None:
        has_valid = mx.any(valid, axis=-1, keepdims=True)
        m_safe = mx.where(has_valid, m_tile, 0.0)
        p = mx.where(valid, mx.exp(scores - m_safe), 0.0)
    else:
        has_valid = None
        p = mx.exp(scores - m_tile)
    l_tile = mx.sum(p, axis=-1, keepdims=True)
    o_tile = mx.matmul(p, v_tile[:, :, None, :, :])

    m_new = mx.maximum(m_acc, m_tile)
    has_acc = l_acc > 0.0
    m_new_safe = mx.where(has_acc | (l_tile > 0.0), m_new, 0.0)
    alpha = mx.where(has_acc, mx.exp(m_acc - m_new_safe), 0.0)
    beta = mx.where(l_tile > 0.0, mx.exp(m_tile - m_new_safe), 0.0)
    l_next = l_acc * alpha + l_tile * beta
    o_next = o_acc * alpha + o_tile * beta
    m_next = mx.where(has_acc | (l_tile > 0.0), m_new, m_acc)
    return m_next, l_next, o_next


def q35_rht_lm_sdpa_prefill(
    q: mx.array,
    k_sink: mx.array | None,
    v_sink: mx.array | None,
    k_codes: mx.array | None,
    k_sc: mx.array | None,
    v_codes: mx.array | None,
    v_sc: mx.array | None,
    k_bits: int,
    v_bits: int,
    n_sink: int | None = None,
    n_body: int | None = None,
    scale: float | None = None,
    mask: str | mx.array | None = "causal",
    tile_k: int = 512,
    tile_q: int = 512,
) -> mx.array:
    """Memory-bounded chunked prefill SDPA over FP16 sinks + Hadamard-space Lloyd-Max body."""
    if k_bits not in VALID_KV_BITS or v_bits not in VALID_KV_BITS:
        raise ValueError(f"Unsupported KV bits: k_bits={k_bits}, v_bits={v_bits}")
    if q.ndim != 4:
        raise ValueError(f"Expected 4D q [B, H, L_q, D], got {q.shape}")
    B, H, L_q, head_dim = q.shape
    if head_dim not in VALID_HEAD_DIMS:
        raise ValueError(f"Unsupported head_dim={head_dim}; expected one of {VALID_HEAD_DIMS}")

    n_sink_i = (int(k_sink.shape[2]) if k_sink is not None else 0) if n_sink is None else int(n_sink)
    n_body_i = (int(k_codes.shape[2]) if k_codes is not None else 0) if n_body is None else int(n_body)
    H_kv = int(k_codes.shape[1]) if (k_codes is not None and n_body_i > 0) else (int(k_sink.shape[1]) if k_sink is not None else H)
    if H % H_kv != 0:
        raise ValueError(f"H={H} must be divisible by H_kv={H_kv}")
    G = H // H_kv
    total_len = n_sink_i + n_body_i
    if L_q == 0 or total_len == 0:
        return mx.zeros_like(q)

    out_dtype = q.dtype
    scale_val = (head_dim ** -0.5) if scale is None else float(scale)
    q_scaled = q.astype(mx.float32) * scale_val
    q_body_scaled = q35_rht(q_scaled, out_dtype=mx.float32) if (k_bits < 16 and n_body_i > 0) else q_scaled

    mask_mode = mask if isinstance(mask, str) else None
    if mask_mode is not None and mask_mode != "causal":
        raise ValueError(f"Unsupported string mask={mask!r}; expected 'causal' or None")
    mask_5d = _normalize_mask_5d(mask, B, H_kv, G, L_q, total_len) if isinstance(mask, mx.array) else None
    q_pos_offset = total_len - L_q

    q_chunks = []
    for q0 in range(0, L_q, tile_q):
        q1 = min(L_q, q0 + tile_q)
        q_len = q1 - q0
        q_pos_max = q_pos_offset + q1 - 1

        qc_unrot = q_scaled[:, :, q0:q1, :].reshape(B, H_kv, G, q_len, head_dim)
        qc_body = q_body_scaled[:, :, q0:q1, :].reshape(B, H_kv, G, q_len, head_dim)

        # 1. Exact FP16 sink tiles
        m_sk = mx.full((B, H_kv, G, q_len, 1), -1e30, dtype=mx.float32)
        l_sk = mx.zeros((B, H_kv, G, q_len, 1), dtype=mx.float32)
        o_sk = mx.zeros((B, H_kv, G, q_len, head_dim), dtype=mx.float32)
        if n_sink_i > 0 and k_sink is not None and v_sink is not None:
            for s0 in range(0, n_sink_i, tile_k):
                s1 = min(n_sink_i, s0 + tile_k)
                if mask_mode == "causal" and s0 > q_pos_max:
                    break
                kt = k_sink[:, :, s0:s1, :].astype(mx.float32)
                vt = v_sink[:, :, s0:s1, :].astype(mx.float32)
                m_sk, l_sk, o_sk = _online_tile_step(
                    qc_unrot, kt, vt, s0, q0, q_pos_offset, mask_mode, mask_5d, m_sk, l_sk, o_sk
                )

        # 2. Compressed body tiles (K and V kept in Hadamard space when bits < 16)
        m_bd = mx.full((B, H_kv, G, q_len, 1), -1e30, dtype=mx.float32)
        l_bd = mx.zeros((B, H_kv, G, q_len, 1), dtype=mx.float32)
        o_bd = mx.zeros((B, H_kv, G, q_len, head_dim), dtype=mx.float32)
        if n_body_i > 0 and k_codes is not None and v_codes is not None:
            eval_every = 4 if n_body_i > 4 * tile_k else 0
            for tile_idx, k0 in enumerate(range(0, n_body_i, tile_k)):
                k1 = min(n_body_i, k0 + tile_k)
                k_global0 = n_sink_i + k0
                if mask_mode == "causal" and k_global0 > q_pos_max:
                    break
                t_len = k1 - k0
                kt = _dequantize_tile(k_codes, k_sc, k_bits, head_dim, k0, t_len, unrotate=False, out_dtype=mx.float32)
                vt = _dequantize_tile(v_codes, v_sc, v_bits, head_dim, k0, t_len, unrotate=False, out_dtype=mx.float32)
                m_bd, l_bd, o_bd = _online_tile_step(
                    qc_body, kt, vt, k_global0, q0, q_pos_offset, mask_mode, mask_5d, m_bd, l_bd, o_bd
                )
                if eval_every and ((tile_idx + 1) % eval_every == 0):
                    mx.eval(m_bd, l_bd, o_bd)

            if v_bits < 16:
                o_bd = q35_rht(o_bd.reshape(B, H, q_len, head_dim), out_dtype=mx.float32).reshape(
                    B, H_kv, G, q_len, head_dim
                )

        # 3. Merge sink + unrotated body
        has_sk = l_sk > 0.0
        has_bd = l_bd > 0.0
        m_all = mx.maximum(m_sk, m_bd)
        m_all_safe = mx.where(has_sk | has_bd, m_all, 0.0)
        alpha_sk = mx.where(has_sk, mx.exp(m_sk - m_all_safe), 0.0)
        beta_bd = mx.where(has_bd, mx.exp(m_bd - m_all_safe), 0.0)
        l_all = l_sk * alpha_sk + l_bd * beta_bd
        o_all = o_sk * alpha_sk + o_bd * beta_bd
        out_c = mx.where(l_all > 0.0, o_all / mx.maximum(l_all, 1e-12), 0.0).reshape(B, H, q_len, head_dim)
        if L_q > tile_q:
            mx.eval(out_c)
        q_chunks.append(out_c)

    out = q_chunks[0] if len(q_chunks) == 1 else mx.concatenate(q_chunks, axis=2)
    return out.astype(out_dtype)


# ----------------------------------------------------------------------------- HadamardLloydMaxKVCache

class HadamardLloydMaxKVCache(_BaseCache):
    """KV cache protecting `sink_tokens` in FP16 and storing the body as RHT + Lloyd-Max / int8."""

    step = 256

    def __init__(self, k_bits: int = 4, v_bits: int = 4, sink_tokens: int = 4):
        if k_bits not in VALID_KV_BITS or v_bits not in VALID_KV_BITS:
            raise ValueError(f"Unsupported KV bits: k_bits={k_bits}, v_bits={v_bits}")
        if sink_tokens < 0:
            raise ValueError(f"sink_tokens must be non-negative, got {sink_tokens}")
        self.k_bits = int(k_bits)
        self.v_bits = int(v_bits)
        self.sink_tokens = int(sink_tokens)
        self.head_dim: int | None = None
        self.offset = 0
        self.k_sink: mx.array | None = None
        self.v_sink: mx.array | None = None
        self.k_codes: mx.array | None = None
        self.k_scales: mx.array | None = None
        self.v_codes: mx.array | None = None
        self.v_scales: mx.array | None = None

    @property
    def n_sink(self) -> int:
        return min(self.offset, self.sink_tokens)

    @property
    def n_body(self) -> int:
        return max(0, self.offset - self.sink_tokens)

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.offset == 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, max(0, int(n)))
        self.offset -= n
        return n

    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None):
        return create_attention_mask(N, offset=self.offset, return_array=return_array, window_size=window_size)

    @property
    def nbytes(self) -> int:
        total = 0
        for arr in (self.k_sink, self.v_sink, self.k_codes, self.k_scales, self.v_codes, self.v_scales):
            if arr is not None:
                total += int(arr.nbytes)
        return total

    @property
    def state(self):
        return (
            self.k_sink,
            self.v_sink,
            self.k_codes,
            self.k_scales,
            self.v_codes,
            self.v_scales,
            self.offset,
            self.k_bits,
            self.v_bits,
            self.sink_tokens,
            self.head_dim,
        )

    @state.setter
    def state(self, v):
        (
            self.k_sink,
            self.v_sink,
            self.k_codes,
            self.k_scales,
            self.v_codes,
            self.v_scales,
            self.offset,
            self.k_bits,
            self.v_bits,
            self.sink_tokens,
            self.head_dim,
        ) = v

    def _ensure_body_capacity(
        self,
        B: int,
        H_kv: int,
        prev_body: int,
        need_body: int,
        head_dim: int,
    ) -> None:
        if self.k_codes is not None and need_body <= int(self.k_codes.shape[2]):
            return
        target_cap = ((need_body + self.step - 1) // self.step) * self.step
        k_packed = _packed_bytes_per_head(self.k_bits, head_dim)
        v_packed = _packed_bytes_per_head(self.v_bits, head_dim)
        k_dt = _codes_dtype(self.k_bits)
        v_dt = _codes_dtype(self.v_bits)

        def grow(buf: mx.array | None, last_dim: int, dtype: mx.Dtype) -> mx.array:
            if buf is None:
                return mx.zeros((B, H_kv, target_cap, last_dim), dtype=dtype)
            add = target_cap - prev_body
            prefix = buf if int(buf.shape[2]) == prev_body else buf[:, :, :prev_body, :]
            return mx.concatenate([prefix, mx.zeros((B, H_kv, add, last_dim), dtype=dtype)], axis=2)

        self.k_codes = grow(self.k_codes, k_packed, k_dt)
        self.v_codes = grow(self.v_codes, v_packed, v_dt)
        if self.k_bits < 16:
            self.k_scales = grow(self.k_scales, 1, mx.float16)
        if self.v_bits < 16:
            self.v_scales = grow(self.v_scales, 1, mx.float16)

    def update(self, keys: mx.array, values: mx.array) -> None:
        """Append `keys, values` [B, H_kv, L_new, D], preserving `sink_tokens` in FP16."""
        B, H_kv, L_new, head_dim = keys.shape
        if self.head_dim is None:
            if head_dim not in VALID_HEAD_DIMS:
                raise ValueError(f"Unsupported head_dim={head_dim}; expected one of {VALID_HEAD_DIMS}")
            self.head_dim = head_dim
        if L_new == 0:
            return

        prev = self.offset
        sink_free = max(0, self.sink_tokens - prev)
        take_sink = min(L_new, sink_free)

        if take_sink > 0:
            if self.k_sink is None:
                self.k_sink = mx.zeros((B, H_kv, self.sink_tokens, head_dim), dtype=mx.float16)
                self.v_sink = mx.zeros((B, H_kv, self.sink_tokens, head_dim), dtype=mx.float16)
            self.k_sink[:, :, prev : prev + take_sink, :] = keys[:, :, :take_sink, :].astype(mx.float16)
            self.v_sink[:, :, prev : prev + take_sink, :] = values[:, :, :take_sink, :].astype(mx.float16)

        rem = L_new - take_sink
        if rem > 0:
            k_rem = keys[:, :, take_sink:, :]
            v_rem = values[:, :, take_sink:, :]
            kq, ks = q35_rht_lm_quantize(k_rem, self.k_bits)
            vq, vs = q35_rht_lm_quantize(v_rem, self.v_bits)
            prev_body = max(0, prev - self.sink_tokens)
            need_body = prev_body + rem
            self._ensure_body_capacity(B, H_kv, prev_body, need_body, head_dim)
            self.k_codes[:, :, prev_body:need_body, :] = kq
            self.v_codes[:, :, prev_body:need_body, :] = vq
            if self.k_bits < 16:
                self.k_scales[:, :, prev_body:need_body, :] = ks
            if self.v_bits < 16:
                self.v_scales[:, :, prev_body:need_body, :] = vs

        self.offset = prev + L_new

    def keys_and_values(self) -> tuple[mx.array | None, mx.array | None]:
        """Return full dequantized FP16 `(K, V)` up to `self.offset`."""
        if self.offset == 0:
            return None, None
        ns = self.n_sink
        nb = self.n_body
        k_parts = []
        v_parts = []
        if ns > 0 and self.k_sink is not None and self.v_sink is not None:
            k_parts.append(self.k_sink[:, :, :ns, :])
            v_parts.append(self.v_sink[:, :, :ns, :])
        if nb > 0 and self.k_codes is not None and self.v_codes is not None and self.head_dim is not None:
            k_parts.append(_dequantize_tile(self.k_codes, self.k_scales, self.k_bits, self.head_dim, 0, nb, unrotate=True, out_dtype=mx.float16))
            v_parts.append(_dequantize_tile(self.v_codes, self.v_scales, self.v_bits, self.head_dim, 0, nb, unrotate=True, out_dtype=mx.float16))
        k_full = k_parts[0] if len(k_parts) == 1 else mx.concatenate(k_parts, axis=2)
        v_full = v_parts[0] if len(v_parts) == 1 else mx.concatenate(v_parts, axis=2)
        return k_full, v_full

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array | None, mx.array | None]:
        self.update(keys, values)
        return self.keys_and_values()


# ----------------------------------------------------------------------------- Attention Wrapper & Plan Installation

class QuantizedKVAttention(nn.Module):
    """Wraps Qwen3NextAttention (`self_attn`), routing HadamardLloydMaxKVCache to fused SDPA kernels."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "inner" or "inner" not in self:
                raise
            return getattr(self["inner"], name)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Any = None,
    ) -> mx.array:
        a = self.inner
        B, L, _ = x.shape

        q_proj_output = a.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, a.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys, values = a.k_proj(x), a.v_proj(x)

        queries = a.q_norm(queries).transpose(0, 2, 1, 3)
        keys = a.k_norm(keys.reshape(B, L, a.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, a.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if isinstance(cache, HadamardLloydMaxKVCache):
            queries = a.rope(queries, offset=cache.offset)
            keys = a.rope(keys, offset=cache.offset)
            cache.update(keys, values)
            if L == 1 and (mask is None or isinstance(mask, str)):
                output = q35_rht_lm_sdpa_decode(
                    queries,
                    cache.k_sink,
                    cache.v_sink,
                    cache.k_codes,
                    cache.k_scales,
                    cache.v_codes,
                    cache.v_scales,
                    k_bits=cache.k_bits,
                    v_bits=cache.v_bits,
                    n_sink=cache.n_sink,
                    n_body=cache.n_body,
                    scale=a.scale,
                )
            else:
                output = q35_rht_lm_sdpa_prefill(
                    queries,
                    cache.k_sink,
                    cache.v_sink,
                    cache.k_codes,
                    cache.k_scales,
                    cache.v_codes,
                    cache.v_scales,
                    k_bits=cache.k_bits,
                    v_bits=cache.v_bits,
                    n_sink=cache.n_sink,
                    n_body=cache.n_body,
                    scale=a.scale,
                    mask=mask,
                )
        else:
            if cache is not None:
                queries = a.rope(queries, offset=cache.offset)
                keys = a.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = a.rope(queries)
                keys = a.rope(keys)
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=a.scale, mask=mask
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return a.o_proj(output * mx.sigmoid(gate))


def _get_layers(model: Any) -> list[Any]:
    if hasattr(model, "layers"):
        return list(model.layers)
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "layers"):
            return list(lm.layers)
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return list(lm.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise AttributeError(f"Cannot locate decoder layers on {type(model).__name__}")


def _layer_inner_and_linear(layer: Any) -> tuple[Any, bool]:
    inner = getattr(layer, "inner", layer)
    is_linear = bool(getattr(inner, "is_linear", getattr(layer, "is_linear", False)))
    return inner, is_linear


def _parse_kv_pair(val: str) -> tuple[int, int]:
    parts = [p.strip() for p in val.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected 'k,v' bit-width pair, got {val!r}")
    kb, vb = int(parts[0]), int(parts[1])
    if kb not in VALID_KV_BITS or vb not in VALID_KV_BITS:
        raise ValueError(f"Unsupported KV bits ({kb}, {vb}); expected each in {VALID_KV_BITS}")
    return kb, vb


def build_q35_kv_plan(model: Any, spec: str) -> dict[int, tuple[int, int]]:
    """Build per-full-attention-layer `(k_bits, v_bits)` plan from a preset or override spec."""
    layers = _get_layers(model)
    full_indices = [i for i, layer in enumerate(layers) if not _layer_inner_and_linear(layer)[1]]
    s = (spec or "fp16").strip().lower()

    if s in ("fp16", "none", ""):
        return {i: (16, 16) for i in full_indices}
    if not full_indices:
        return {}

    if s == "lm4":
        return {i: (4, 4) for i in full_indices}
    if s in ("lm4_3", "q35_auto"):
        plan = {i: (4, 3) for i in full_indices}
        plan[full_indices[-1]] = (4, 4)
        return plan
    if s == "lm3_2":
        plan = {i: (3, 2) for i in full_indices}
        plan[full_indices[-1]] = (4, 4)
        return plan
    if s == "q35_compact":
        plan = {i: (3, 2) for i in full_indices}
        if len(full_indices) >= 2:
            plan[full_indices[-2]] = (4, 3)
        plan[full_indices[-1]] = (4, 4)
        return plan

    m = re.fullmatch(r"k(\d+)v(\d+)", s)
    if m:
        kb, vb = int(m.group(1)), int(m.group(2))
        if kb not in VALID_KV_BITS or vb not in VALID_KV_BITS:
            raise ValueError(f"Unsupported KV bits in {spec!r}; expected each in {VALID_KV_BITS}")
        return {i: (kb, vb) for i in full_indices}

    if ":" in s:
        plan = {i: (16, 16) for i in full_indices}
        full_set = set(full_indices)
        # Apply `default` before positional / explicit overrides regardless of token order.
        items = [tok.strip() for tok in s.split(";") if tok.strip()]
        overrides: list[tuple[str, tuple[int, int]]] = []
        for item in items:
            if ":" not in item:
                raise ValueError(f"Invalid override entry {item!r} in KV spec {spec!r}")
            key, val = item.split(":", 1)
            key = key.strip()
            pair = _parse_kv_pair(val)
            if key == "default":
                plan = {i: pair for i in full_indices}
            else:
                overrides.append((key, pair))
        for key, pair in overrides:
            if key == "first":
                plan[full_indices[0]] = pair
            elif key in ("penultimate", "penult"):
                if len(full_indices) >= 2:
                    plan[full_indices[-2]] = pair
            elif key == "last":
                plan[full_indices[-1]] = pair
            else:
                try:
                    idx = int(key)
                except ValueError as exc:
                    raise ValueError(f"Unknown KV plan key {key!r} in {spec!r}") from exc
                if idx not in full_set:
                    raise ValueError(f"Layer {idx} is not a full-attention layer (valid: {full_indices})")
                plan[idx] = pair
        return plan

    raise ValueError(f"Unknown KV quant spec: {spec!r}")


def _infer_linear_cache_size(model: Any) -> int:
    orig_mc = getattr(model, "make_cache", None)
    if callable(orig_mc):
        try:
            sample = orig_mc()
            for c in sample:
                if isinstance(c, ArraysCache):
                    return max(1, len(c.cache))
        except Exception:
            pass
    return 2


def install_kv_quant(model: Any, spec: str, sink_tokens: int = 4) -> dict[int, tuple[int, int]]:
    """Install `QuantizedKVAttention` and patch `make_cache()` on `model` according to `spec`."""
    s = (spec or "fp16").strip().lower()
    plan = build_q35_kv_plan(model, spec)
    if s in ("fp16", "none", ""):
        return plan

    linear_cache_size = _infer_linear_cache_size(model)
    layers = _get_layers(model)
    for idx in plan:
        inner, is_linear = _layer_inner_and_linear(layers[idx])
        if is_linear or not hasattr(inner, "self_attn"):
            raise ValueError(f"Expected full-attention layer with `self_attn` at index {idx}")
        if not isinstance(inner.self_attn, QuantizedKVAttention):
            inner.self_attn = QuantizedKVAttention(inner.self_attn)

    def _make_kv_cache():
        cur_layers = _get_layers(model)
        caches = []
        for i, layer in enumerate(cur_layers):
            _, is_lin = _layer_inner_and_linear(layer)
            if is_lin:
                caches.append(ArraysCache(size=linear_cache_size))
            else:
                kb, vb = plan[i]
                caches.append(HadamardLloydMaxKVCache(k_bits=kb, v_bits=vb, sink_tokens=sink_tokens))
        return caches

    model.make_cache = _make_kv_cache
    if hasattr(model, "language_model"):
        model.language_model.make_cache = _make_kv_cache
    model._lamq_kv_plan = plan
    model._lamq_kv_spec = spec
    model._lamq_kv_sink_tokens = sink_tokens
    return plan
