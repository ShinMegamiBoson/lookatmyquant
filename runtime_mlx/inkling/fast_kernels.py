"""High-bandwidth Apple Silicon Metal kernels for packed Inkling-2-Mini (decode, DSpark block verify, and prefill).

Zero extra weight formats and zero accuracy compromise:
  - L1-cached bit-exact multi-weight tables for `rtr1` (g=2 32 KB / g=3 128 KB), `rtr2` (g=2, 64 KB), `rtr3` (g=2, 128 KB), `rtr4` (8 KB)
  - 4 KB threadgroup-SRAM hashed vector tables for `ht1`/`ht2` (g=4, half4) and `ht3` (g=2, half2)
  - Single-launch threadgroup Metal RHTs (`2304` incl. transposed `sg8` layout, `1152` + fused `softcapped_swiglu`, `3072` + fused `o_gate`)
  - Fused multi-matrix `DenseStage` (all 5 attention projections & 200K LM head in 1 launch: half-warp decode, `simdgroup_half8x8` `T=2..8`, FP16 GEMM prefill)
  - Unified 1-launch multi-format `UnifiedMoESide` (T=1 direct, T=2..8 pre-planned lean half-warp grouped, T>8 balanced 16-token `simdgroup_half8x8` prefill)
  - Fused glue kernels: `block_attn_res_fused`, `block_attn_res_norm_rht_fast`, `msg_blend_norm_fused`, `msg_blend_norm_rht_fast`,
    `sigmoid_top8_norm_and_plan_fast`, `qkv_rel_prep_sliding_fused`, `kv_dyn_sconv4_norm_fused`,
    `gqa_relpos_sinks_sdpa_fast`, `short_conv4_res_fused`, `moe_wsum_sconv4_res_fused`, `embed_lookup_gpu`
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Sequence

import mlx.core as mx
import numpy as np

from .kernels import HASH_MULT_HEX, hadamard_factors_mx, nvfp4_luts_mx

FMT_ID = {
    "ht1": 0,
    "ht2": 1,
    "ht3": 2,
    "rtr1": 3,
    "rtr2": 4,
    "rtr3": 5,
    "rtr4": 6,
    "nvfp4": 7,
    "lm6": 10,
    "lm8": 11,
}


@lru_cache(maxsize=512)
def _u32(*vals: int) -> mx.array:
    return mx.array(list(vals), dtype=mx.uint32)


@lru_cache(maxsize=64)
def _f32_arr(*vals: float) -> mx.array:
    return mx.array(list(vals), dtype=mx.float32)


@lru_cache(maxsize=1)
def _is_m4() -> bool:
    if os.environ.get("LAMQ_M4_TUNE") == "0":
        return False
    arch = str(mx.device_info().get("architecture", "")).lower()
    return arch.startswith("applegpu_g16")


# ----------------------------------------------------------------------------- L1 Multi-Table LUT Preparation

def build_rtr_multitable(lut_f16: mx.array, k: int, g: int) -> mx.array:
    """Build bit-exact multi-weight L1 table for a 12-bit scalar trellis (`rtr1..rtr3`).

    Column `j` has its 12-bit state at bit `(n - 1 - j) * k`. Extracting `12 + (g - 1) * k` bits
    at `last = j + g - 1` fixes all `g` consecutive weights: column `j + t` sits at bit `(g - 1 - t) * k`.
    """
    vals = np.array(lut_f16, dtype=np.float16).reshape(4096)
    bits = 12 + (g - 1) * k
    idx = np.arange(1 << bits, dtype=np.int64)
    slots = 1 if g == 1 else (2 if g == 2 else 4)
    out = np.zeros((len(idx), slots), dtype=np.float16)
    for t in range(g):
        out[:, t] = vals[(idx >> ((g - 1 - t) * k)) & 4095]
    return mx.array(out.reshape(-1))


def build_fast_luts(luts: dict[str, mx.array]) -> dict[str, mx.array]:
    """Prepare flat half LUT buffers + L1 multi-tables once at model load (~380 KB total)."""
    dummy = mx.zeros((512,), dtype=mx.float16)
    e2m1_def, e4m3_def = nvfp4_luts_mx()
    fast = {
        "ht1": luts["ht1"].reshape(-1).astype(mx.float16) if "ht1" in luts else dummy,
        "ht2": luts["ht2"].reshape(-1).astype(mx.float16) if "ht2" in luts else dummy,
        "ht3": luts["ht3"].reshape(-1).astype(mx.float16) if "ht3" in luts else dummy,
        "rtr1_g2": build_rtr_multitable(luts["rtr1"], 1, 2) if "rtr1" in luts else dummy,
        "rtr1_g3": build_rtr_multitable(luts["rtr1"], 1, 3) if "rtr1" in luts else dummy,
        "rtr2_g2": build_rtr_multitable(luts["rtr2"], 2, 2) if "rtr2" in luts else dummy,
        "rtr3_g2": build_rtr_multitable(luts["rtr3"], 3, 2) if "rtr3" in luts else dummy,
        "rtr4": luts["rtr4"].reshape(-1).astype(mx.float16) if "rtr4" in luts else dummy,
        "nvfp4_e2m1": (luts["nvfp4_e2m1"] if "nvfp4_e2m1" in luts else e2m1_def).reshape(-1).astype(mx.float16),
        "nvfp4_e4m3": (luts["nvfp4_e4m3"] if "nvfp4_e4m3" in luts else e4m3_def).reshape(-1).astype(mx.float16),
    }
    mx.eval(fast)
    return fast


# ----------------------------------------------------------------------------- Single-Launch Threadgroup Metal RHTs

@lru_cache(maxsize=8)
def _paley_b_flat(n: int) -> mx.array:
    _, _, B, _, _ = hadamard_factors_mx(n)
    out = B.reshape(-1).astype(mx.float32)
    mx.eval(out)
    return out


@lru_cache(maxsize=8)
def _paley_bt_flat(n: int) -> mx.array:
    _, _, _, Bt, _ = hadamard_factors_mx(n)
    out = Bt.reshape(-1).astype(mx.float32)
    mx.eval(out)
    return out


@lru_cache(maxsize=16)
def _make_rht2304_in_kernel(out_half: bool, transpose_t: int = 0, pad_zero: bool = False):
    ot = "half" if out_half else "float"
    if transpose_t > 0:
        # Output layout: [n_signs, 2304, 8] (stride 8 even when 2 <= T < 8, zeroing unused columns t >= T)
        zero_tail = (
            f"if (t_out == 0u) {{ for (uint zt = {transpose_t}u; zt < 8u; zt++) y[base_col + zt] = ({ot})0.0f; }}"
            if transpose_t < 8
            else ""
        )
        store = f"""
        const uint s_out = out_idx / {transpose_t}u;
        const uint t_out = out_idx - s_out * {transpose_t}u;
        const ulong base_col = ((ulong)s_out * 2304u + (lane * 72u + c)) * 8u;
        y[base_col + t_out] = ({ot})(acc * INV_S);
        {zero_tail}
        """
    else:
        store = f"op[lane * 72u + c] = ({ot})(acc * INV_S);"
    pad_code = (
        f"""
    if (out_idx == 0u) {{
        device {ot}* zp = y + (ulong)threadgroups_per_grid.y * 2304u;
        #pragma unroll
        for (uint j = 0; j < 9u; j++) zp[lane * 72u + c0 + j] = ({ot})0.0f;
    }}
    """
        if pad_zero and transpose_t == 0
        else ""
    )
    src = f"""
    const uint out_idx = threadgroup_position_in_grid.y;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup; // a in [0..31]
    const uint x_row = x_map[out_idx];
    const uint s_row = s_map[out_idx];
    const device float* xp = x + (ulong)x_row * 2304u;
    const device float* sp = signs + (ulong)s_row * 2304u;
    device {ot}* op = y + (ulong)out_idx * 2304u;

    threadgroup float tg_z[32 * 72];
    const uint c0 = simd * 9u;
    const bool h1 = lane & 1u, h2 = lane & 2u, h4 = lane & 4u, h8 = lane & 8u, h16 = lane & 16u;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {{
        const uint c = c0 + j;
        const uint idx = lane * 72u + c;
        float v = xp[idx] * sp[idx];
        float o;
        o = simd_shuffle_xor(v, 1u);  v = h1  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 2u);  v = h2  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 4u);  v = h4  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 8u);  v = h8  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 16u); v = h16 ? (o - v) : (v + o);
        tg_z[idx] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup float* zr = tg_z + lane * 72u;
    constexpr float INV_S = 1.0f / 48.0f;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {{
        const uint c = c0 + j;
        const device float4* br = (const device float4*)(B + c * 72u);
        float acc = 0.0f;
        #pragma unroll
        for (uint k4 = 0; k4 < 18u; k4++) {{
            const float4 b4 = br[k4];
            const uint k = k4 * 4u;
            acc += zr[k] * b4.x + zr[k + 1u] * b4.y + zr[k + 2u] * b4.z + zr[k + 3u] * b4.w;
        }}
        {store}
    }}
    {pad_code}
    """
    return mx.fast.metal_kernel(
        name=f"ink_rht2304_in_{'h' if out_half else 'f'}_tr{transpose_t}_pz{int(pad_zero)}",
        input_names=["x", "signs", "B", "x_map", "s_map"],
        output_names=["y"],
        source=src,
    )


@lru_cache(maxsize=64)
def _rht_maps_shared(n_signs: int, T: int) -> tuple[mx.array, mx.array]:
    # Output layout: [n_signs, T, 2304] -> out_idx = s * T + t
    xm = mx.array(np.tile(np.arange(T, dtype=np.uint32), n_signs))
    sm = mx.array(np.repeat(np.arange(n_signs, dtype=np.uint32), T))
    mx.eval(xm, sm)
    return xm, sm


@lru_cache(maxsize=64)
def _rht_maps_qkv_split(T: int) -> tuple[mx.array, mx.array]:
    # x is stacked [q_in(T), k_in(T), v_in(T)] -> mats (0:q, 1:k, 2:v, 3:rel, 4:o_gate) read inputs (0, 1, 2, 0, 0)!
    t_arr = np.arange(T, dtype=np.uint32)
    xm = mx.array(np.concatenate([t_arr, t_arr + T, t_arr + 2 * T, t_arr, t_arr]))
    sm = mx.array(np.repeat(np.arange(5, dtype=np.uint32), T))
    mx.eval(xm, sm)
    return xm, sm


@lru_cache(maxsize=8)
def _make_rht2304_in_sg8_kernel(out_half: bool, pad_zero: bool = False):
    """Coalesced float4 + 0-bank-conflict stride-73 + simdgroup_float8x8 Paley-72 RHT-2304 for prefill (n_out >= 64)."""
    ot4 = "half4" if out_half else "float4"
    conv = "half4(tz4[i4] * INV_S)" if out_half else "(tz4[i4] * INV_S)"
    pad_code = (
        f"""
    if (out_idx == 0u) {{
        device {ot4}* zp4 = (device {ot4}*)(y + (ulong)threadgroups_per_grid.y * 2304u);
        #pragma unroll
        for (uint e = 0; e < 2u; e++) zp4[tid + e * 288u] = {ot4}(0.0f);
    }}
    """
        if pad_zero
        else ""
    )
    src = f"""
    const uint out_idx = threadgroup_position_in_grid.y;
    const uint simd = simdgroup_index_in_threadgroup; // 0..8 (9 simdgroups = 288 threads)
    const uint lane = thread_index_in_simdgroup;      // 0..31
    const uint tid = simd * 32u + lane;               // 0..287
    const uint x_row = x_map[out_idx];
    const uint s_row = s_map[out_idx];
    const device float4* xp4 = (const device float4*)(x + (ulong)x_row * 2304u);
    const device float4* sp4 = (const device float4*)(signs + (ulong)s_row * 2304u);
    device {ot4}* op4 = (device {ot4}*)(y + (ulong)out_idx * 2304u);

    threadgroup float tg_z[32 * 73];
    // 1. Coalesced float4 load into 0-bank-conflict stride-73 TG SRAM (288 threads x 2 float4 = 2304)
    #pragma unroll
    for (uint e = 0; e < 2u; e++) {{
        const uint i4 = tid + e * 288u;
        const float4 v4 = xp4[i4] * sp4[i4];
        const uint a = i4 / 18u;
        const uint c = (i4 - a * 18u) * 4u;
        threadgroup float* dst = tg_z + a * 73u + c;
        dst[0] = v4.x; dst[1] = v4.y; dst[2] = v4.z; dst[3] = v4.w;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Sylvester-32 warp shuffle (9 simdgroups x 8 cols = 72 cols)
    const uint c0 = simd * 8u;
    const bool h1 = lane & 1u, h2 = lane & 2u, h4 = lane & 4u, h8 = lane & 8u, h16 = lane & 16u;
    #pragma unroll
    for (uint j = 0; j < 8u; j++) {{
        const uint c = c0 + j;
        float v = tg_z[lane * 73u + c];
        float o;
        o = simd_shuffle_xor(v, 1u);  v = h1  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 2u);  v = h2  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 4u);  v = h4  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 8u);  v = h8  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 16u); v = h16 ? (o - v) : (v + o);
        tg_z[lane * 73u + c] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Paley-72 matmul Z[32, 72] @ B^T[72, 72] via simdgroup_float8x8
    simdgroup_float8x8 Acc[4];
    #pragma unroll
    for (uint rb = 0; rb < 4u; rb++) Acc[rb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    #pragma unroll
    for (uint kb = 0; kb < 9u; kb++) {{
        simdgroup_float8x8 Bm;
        simdgroup_load(Bm, B + (simd * 8u) * 72u + kb * 8u, 72u, ulong2(0, 0), true);
        #pragma unroll
        for (uint rb = 0; rb < 4u; rb++) {{
            simdgroup_float8x8 Zm;
            simdgroup_load(Zm, tg_z + (rb * 8u) * 73u + kb * 8u, 73u, ulong2(0, 0), false);
            simdgroup_multiply_accumulate(Acc[rb], Zm, Bm, Acc[rb]);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint rb = 0; rb < 4u; rb++) simdgroup_store(Acc[rb], tg_z + (rb * 8u) * 72u + simd * 8u, 72u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr float INV_S = 1.0f / 48.0f;
    const threadgroup float4* tz4 = (const threadgroup float4*)tg_z;
    #pragma unroll
    for (uint e = 0; e < 2u; e++) {{
        const uint i4 = tid + e * 288u;
        op4[i4] = {conv};
    }}
    {pad_code}
    """
    return mx.fast.metal_kernel(
        name=f"ink_rht2304_in_sg8_{'h' if out_half else 'f'}_pz{int(pad_zero)}",
        input_names=["x", "signs", "B", "x_map", "s_map"],
        output_names=["y"],
        header="#include <metal_simdgroup_matrix>\n",
        source=src,
    )


def rht_in_2304_fast(
    x: mx.array,
    signs: mx.array,
    n_signs: int = 1,
    T: int | None = None,
    split_qkv: bool = False,
    out_dtype=mx.float16,
    transpose_for_sg8: bool = False,
    pad_zero_row: bool = False,
) -> mx.array:
    """Single-launch threadgroup Metal RHT-2304 (1 sign or N stacked signs, shared or split Q/K/V)."""
    if T is None:
        T = int(x.size) // (2304 * (3 if split_qkv else 1))
    if split_qkv:
        xm, sm = _rht_maps_qkv_split(T)
        n_out = 5 * T
    else:
        xm, sm = _rht_maps_shared(n_signs, T)
        n_out = n_signs * T
    tr_t = T if transpose_for_sg8 else 0
    pz = pad_zero_row and tr_t == 0
    out_shape = (n_out // T, 2304, 8) if transpose_for_sg8 else (n_out + (1 if pz else 0), 2304)
    if tr_t == 0 and n_out >= 64:
        kern = _make_rht2304_in_sg8_kernel(out_dtype == mx.float16, pz)
        return kern(
            inputs=[x.reshape(-1), signs.reshape(-1), _paley_b_flat(2304), xm, sm],
            grid=(288, n_out, 1),
            threadgroup=(288, 1, 1),
            output_shapes=[out_shape],
            output_dtypes=[out_dtype],
        )[0]
    kern = _make_rht2304_in_kernel(out_dtype == mx.float16, tr_t, pz)
    out = kern(
        inputs=[x.reshape(-1), signs.reshape(-1), _paley_b_flat(2304), xm, sm],
        grid=(256, n_out, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[out_dtype],
    )[0]
    return out


@lru_cache(maxsize=2)
def _make_rht2304_out_kernel():
    # Inverse RHT for PackedEmbed: out = ((At @ y @ B) * signs_out / 48). Note A = At (Sylvester 32x32 symmetric!).
    # Row c of Bt (72x72) = B[:, c]!
    src = """
    const uint t = threadgroup_position_in_grid.y;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint s_row = s_map[t];
    const device float* yp = y + (ulong)t * 2304u;
    const device float* sp = signs + (ulong)s_row * 2304u;
    device float* op = out + (ulong)t * 2304u;

    threadgroup float tg_z[32 * 72];
    const uint c0 = simd * 9u;
    const bool h1 = lane & 1u, h2 = lane & 2u, h4 = lane & 4u, h8 = lane & 8u, h16 = lane & 16u;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {
        const uint c = c0 + j;
        const uint idx = lane * 72u + c;
        float v = yp[idx];
        float o;
        o = simd_shuffle_xor(v, 1u);  v = h1  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 2u);  v = h2  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 4u);  v = h4  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 8u);  v = h8  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 16u); v = h16 ? (o - v) : (v + o);
        tg_z[idx] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup float* zr = tg_z + lane * 72u;
    constexpr float INV_S = 1.0f / 48.0f;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {
        const uint c = c0 + j;
        const device float4* br = (const device float4*)(Bt + c * 72u);
        float acc = 0.0f;
        #pragma unroll
        for (uint k4 = 0; k4 < 18u; k4++) {
            const float4 b4 = br[k4];
            const uint k = k4 * 4u;
            acc += zr[k] * b4.x + zr[k + 1u] * b4.y + zr[k + 2u] * b4.z + zr[k + 3u] * b4.w;
        }
        const uint idx = lane * 72u + c;
        op[idx] = acc * (sp[idx] * INV_S);
    }
    """
    return mx.fast.metal_kernel(
        name="ink_rht2304_out_f",
        input_names=["y", "signs", "Bt", "s_map"],
        output_names=["out"],
        source=src,
    )


def rht_out_2304_fast(y: mx.array, signs_all: mx.array, s_map: mx.array) -> mx.array:
    T = int(s_map.shape[0])
    return _make_rht2304_out_kernel()(
        inputs=[y.reshape(-1), signs_all.reshape(-1), _paley_bt_flat(2304), s_map.astype(mx.uint32)],
        grid=(256, T, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, 2304)],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=2)
def _make_rht1024_out_kernel():
    """Single-launch 32x32 Sylvester inverse RHT-1024 for `v_proj.out_rot=True` (100% coalesced, 0 bank conflicts)."""
    src = """
    const uint t = threadgroup_position_in_grid.y;
    const uint simd = simdgroup_index_in_threadgroup; // 0..7
    const uint lane = thread_index_in_simdgroup;      // 0..31
    const uint tid = simd * 32u + lane;               // 0..255
    const device float4* yp4 = (const device float4*)(y + (ulong)t * 1024u);
    device float* op = out + (ulong)t * 1024u;

    threadgroup float tg[32 * 33];
    const float4 v4 = yp4[tid];
    const uint ch0 = tid * 4u;
    const uint a0 = ch0 >> 5, c0 = ch0 & 31u;
    tg[a0 * 33u + c0]      = v4.x;
    tg[a0 * 33u + c0 + 1u] = v4.y;
    tg[a0 * 33u + c0 + 2u] = v4.z;
    tg[a0 * 33u + c0 + 3u] = v4.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool h1 = lane & 1u, h2 = lane & 2u, h4 = lane & 4u, h8 = lane & 8u, h16 = lane & 16u;
    #pragma unroll
    for (uint j = 0; j < 4u; j++) {
        const uint c = simd * 4u + j;
        float v = tg[lane * 33u + c];
        float o;
        o = simd_shuffle_xor(v, 1u);  v = h1  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 2u);  v = h2  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 4u);  v = h4  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 8u);  v = h8  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 16u); v = h16 ? (o - v) : (v + o);
        tg[lane * 33u + c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr float INV_S = 1.0f / 32.0f;
    #pragma unroll
    for (uint j = 0; j < 4u; j++) {
        const uint a = simd * 4u + j;
        float v = tg[a * 33u + lane];
        float o;
        o = simd_shuffle_xor(v, 1u);  v = h1  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 2u);  v = h2  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 4u);  v = h4  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 8u);  v = h8  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 16u); v = h16 ? (o - v) : (v + o);
        const uint ch = a * 32u + lane;
        op[ch] = v * (signs[ch] * INV_S);
    }
    """
    return mx.fast.metal_kernel(
        name="ink_rht1024_out_f",
        input_names=["y", "signs"],
        output_names=["out"],
        source=src,
    )


def rht_out_1024_fast(y: mx.array, signs_out: mx.array) -> mx.array:
    """Fast 1-launch inverse RHT-1024 (`y @ R_out`, matching `rht_apply_out(y, signs_out)` for `m=1024`)."""
    orig = y.shape
    T = int(y.size) // 1024
    yf = y if y.dtype == mx.float32 else y.astype(mx.float32)
    sf = signs_out if signs_out.dtype == mx.float32 else signs_out.astype(mx.float32)
    out = _make_rht1024_out_kernel()(
        inputs=[yf.reshape(-1), sf.reshape(-1)],
        grid=(256, T, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, 1024)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(orig)


@lru_cache(maxsize=32)
def _make_swiglu_rht1152_kernel(out_half: bool, pad_zero: bool = False, skip_mode: int = 0, also_raw: bool = False):
    # skip_mode: 0=none, 1=eids[slot]==0xFFFFFFFFu, 2=slot>=offsets[128]
    ot = "half" if out_half else "float"
    raw_pad = (
        f"\n        device {ot}* rzp = raw_out + (ulong)threadgroups_per_grid.y * 1152u;\n"
        f"        #pragma unroll\n"
        f"        for (uint j = 0; j < 9u; j++) rzp[hl * 72u + c0 + j] = ({ot})0.0f;"
        if also_raw
        else ""
    )
    pad_code = (
        f"""
    if (slot == 0u) {{
        device {ot}* zp = out + (ulong)threadgroups_per_grid.y * 1152u;
        #pragma unroll
        for (uint j = 0; j < 9u; j++) zp[hl * 72u + c0 + j] = ({ot})0.0f;{raw_pad}
    }}
    """
        if pad_zero
        else ""
    )
    if skip_mode == 1:
        skip_cond = "eids[slot] == 0xFFFFFFFFu"
        extra_in = ["eids"]
    elif skip_mode == 2:
        skip_cond = "slot >= offsets[128]"
        extra_in = ["offsets"]
    else:
        skip_cond = ""
        extra_in = []
    skip_zero_raw = (
        f" device {ot}* rop_s = raw_out + (ulong)slot * 1152u; for (uint j = 0; j < 9u; j++) rop_s[hl * 72u + c0 + j] = ({ot})0.0f;"
        if also_raw
        else ""
    )
    skip_code = f"    if ({skip_cond}) {{{skip_zero_raw}{pad_code} return; }}\n" if skip_mode != 0 else ""
    raw_decl = f"    device {ot}* rop = raw_out + (ulong)slot * 1152u;\n" if also_raw else ""
    raw_store = f"\n        rop[idx] = ({ot})act;" if also_raw else ""
    src = f"""
    const uint slot = threadgroup_position_in_grid.y;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint hw = (simd << 1) | (lane >> 4); // 0..7
    const uint hl = lane & 15u;                 // a in 0..15
    const uint c0 = hw * 9u;
{skip_code}    const device float* gp = y_gu + (ulong)slot * 2304u;
    const device float* up = gp + 1152u;
    device {ot}* op = out + (ulong)slot * 1152u;
{raw_decl}    const float g_cap = caps[0], g_inv = caps[1], u_cap = caps[2], u_inv = caps[3];

    threadgroup float tg_z[16 * 72];
    const bool h1 = hl & 1u, h2 = hl & 2u, h4 = hl & 4u, h8 = hl & 8u;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {{
        const uint c = c0 + j;
        const uint idx = hl * 72u + c;
        const float g = gp[idx], u = up[idx];
        const float act = (metal::precise::tanh(g * g_inv) * g_cap)
                        * (1.0f / (1.0f + metal::precise::exp(-g)))
                        * (metal::precise::tanh(u * u_inv) * u_cap);{raw_store}
        float v = act * signs[idx];
        float o;
        o = simd_shuffle_xor(v, 1u); v = h1 ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 2u); v = h2 ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 4u); v = h4 ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 8u); v = h8 ? (o - v) : (v + o);
        tg_z[idx] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup float* zr = tg_z + hl * 72u;
    constexpr float INV_S = 0.02946278254943948f; // 1/sqrt(1152)
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {{
        const uint c = c0 + j;
        const device float4* br = (const device float4*)(B + c * 72u);
        float acc = 0.0f;
        #pragma unroll
        for (uint k4 = 0; k4 < 18u; k4++) {{
            const float4 b4 = br[k4];
            const uint k = k4 * 4u;
            acc += zr[k] * b4.x + zr[k + 1u] * b4.y + zr[k + 2u] * b4.z + zr[k + 3u] * b4.w;
        }}
        op[hl * 72u + c] = ({ot})(acc * INV_S);
    }}
    {pad_code}
    """
    out_names = ["out", "raw_out"] if also_raw else ["out"]
    sfx = "_ar1" if also_raw else ""
    return mx.fast.metal_kernel(
        name=f"ink_swiglu_rht1152_{'h' if out_half else 'f'}_pz{int(pad_zero)}_sk{skip_mode}{sfx}",
        input_names=["y_gu", "signs", "B", "caps"] + extra_in,
        output_names=out_names,
        source=src,
    )


@lru_cache(maxsize=32)
def _make_swiglu_rht1152_sg8_kernel(out_half: bool, pad_zero: bool = False, skip_mode: int = 0, also_raw: bool = False):
    """Coalesced float4 + stride-73 + simdgroup_float8x8 Paley-72 SwiGLU+RHT-1152 for prefill (n_slots >= 64)."""
    ot4 = "half4" if out_half else "float4"
    conv = "half4(tz4[tid] * INV_S)" if out_half else "(tz4[tid] * INV_S)"
    raw_pad = f"\n        device {ot4}* rzp4 = (device {ot4}*)(raw_out + (ulong)threadgroups_per_grid.y * 1152u);\n        rzp4[tid] = {ot4}(0.0f);" if also_raw else ""
    pad_code = (
        f"""
    if (slot == 0u) {{
        device {ot4}* zp4 = (device {ot4}*)(out + (ulong)threadgroups_per_grid.y * 1152u);
        zp4[tid] = {ot4}(0.0f);{raw_pad}
    }}
    """
        if pad_zero
        else ""
    )
    if skip_mode == 1:
        skip_cond = "eids[slot] == 0xFFFFFFFFu"
        extra_in = ["eids"]
    elif skip_mode == 2:
        skip_cond = "slot >= offsets[128]"
        extra_in = ["offsets"]
    else:
        skip_cond = ""
        extra_in = []
    skip_zero_raw = f" ((device {ot4}*)(raw_out + (ulong)slot * 1152u))[tid] = {ot4}(0.0f);" if also_raw else ""
    skip_code = f"    if ({skip_cond}) {{{skip_zero_raw}{pad_code} return; }}\n" if skip_mode != 0 else ""
    raw_decl = f"    device {ot4}* rop4 = (device {ot4}*)(raw_out + (ulong)slot * 1152u);\n" if also_raw else ""
    raw_conv = "half4(raw_act)" if out_half else "raw_act"
    raw_store = f"\n    rop4[tid] = {raw_conv};" if also_raw else ""
    src = f"""
    const uint slot = threadgroup_position_in_grid.y;
    const uint simd = simdgroup_index_in_threadgroup; // 0..8 (9 simdgroups = 288 threads)
    const uint lane = thread_index_in_simdgroup;      // 0..31
    const uint tid = simd * 32u + lane;               // 0..287
{skip_code}    const uint hw = lane >> 4, hl = lane & 15u;
    const device float4* gp4 = (const device float4*)(y_gu + (ulong)slot * 2304u);
    const device float4* up4 = gp4 + 288u;
    const device float4* sp4 = (const device float4*)signs;
    device {ot4}* op4 = (device {ot4}*)(out + (ulong)slot * 1152u);
{raw_decl}    const float g_cap = caps[0], g_inv = caps[1], u_cap = caps[2], u_inv = caps[3];

    threadgroup float tg_z[16 * 73];
    const float4 g = gp4[tid], u = up4[tid], s4 = sp4[tid];
    const float4 raw_act = (metal::precise::tanh(g * g_inv) * g_cap)
                         * (1.0f / (1.0f + metal::precise::exp(-g)))
                         * (metal::precise::tanh(u * u_inv) * u_cap);{raw_store}
    const float4 act = raw_act * s4;
    const uint a = tid / 18u, c = (tid - a * 18u) * 4u;
    threadgroup float* dst = tg_z + a * 73u + c;
    dst[0] = act.x; dst[1] = act.y; dst[2] = act.z; dst[3] = act.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c0 = simd * 8u + hw * 4u;
    const bool h1 = hl & 1u, h2 = hl & 2u, h4 = hl & 4u, h8 = hl & 8u;
    #pragma unroll
    for (uint j = 0; j < 4u; j++) {{
        const uint cc = c0 + j;
        float v = tg_z[hl * 73u + cc];
        float o;
        o = simd_shuffle_xor(v, 1u); v = h1 ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 2u); v = h2 ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 4u); v = h4 ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 8u); v = h8 ? (o - v) : (v + o);
        tg_z[hl * 73u + cc] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 Acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    #pragma unroll
    for (uint kb = 0; kb < 9u; kb++) {{
        simdgroup_float8x8 Bm, Zm0, Zm1;
        simdgroup_load(Bm, B + (simd * 8u) * 72u + kb * 8u, 72u, ulong2(0, 0), true);
        simdgroup_load(Zm0, tg_z + kb * 8u, 73u, ulong2(0, 0), false);
        simdgroup_load(Zm1, tg_z + 8u * 73u + kb * 8u, 73u, ulong2(0, 0), false);
        simdgroup_multiply_accumulate(Acc0, Zm0, Bm, Acc0);
        simdgroup_multiply_accumulate(Acc1, Zm1, Bm, Acc1);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(Acc0, tg_z + simd * 8u, 72u);
    simdgroup_store(Acc1, tg_z + 8u * 72u + simd * 8u, 72u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr float INV_S = 0.02946278254943948f;
    const threadgroup float4* tz4 = (const threadgroup float4*)tg_z;
    op4[tid] = {conv};
    {pad_code}
    """
    out_names = ["out", "raw_out"] if also_raw else ["out"]
    sfx = "_ar1" if also_raw else ""
    return mx.fast.metal_kernel(
        name=f"ink_swiglu_rht1152_sg8_{'h' if out_half else 'f'}_pz{int(pad_zero)}_sk{skip_mode}{sfx}",
        input_names=["y_gu", "signs", "B", "caps"] + extra_in,
        output_names=out_names,
        header="#include <metal_simdgroup_matrix>\n",
        source=src,
    )


def swiglu_rht_1152_fast(
    y_gu: mx.array,
    signs: mx.array,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
    out_dtype=mx.float16,
    pad_zero_row: bool = False,
    eids_flat: mx.array | None = None,
    offsets: mx.array | None = None,
    also_raw: bool = False,
):
    n_slots = int(y_gu.shape[0])
    caps = _f32_arr(gate_cap, 1.0 / gate_cap, up_cap, 1.0 / up_cap)
    ins = [y_gu.reshape(-1), signs.reshape(-1), _paley_b_flat(1152), caps]
    if eids_flat is not None:
        skip_mode = 1
        ins.append(eids_flat.astype(mx.uint32).reshape(-1))
    elif offsets is not None:
        skip_mode = 2
        ins.append(offsets.astype(mx.uint32).reshape(-1))
    else:
        skip_mode = 0
    out_shape = (n_slots + (1 if pad_zero_row else 0), 1152)
    oshapes = [out_shape, out_shape] if also_raw else [out_shape]
    odtypes = [out_dtype, out_dtype] if also_raw else [out_dtype]
    if n_slots >= 64:
        kern = _make_swiglu_rht1152_sg8_kernel(out_dtype == mx.float16, pad_zero_row, skip_mode, also_raw)
        outs = kern(
            inputs=ins,
            grid=(288, n_slots, 1),
            threadgroup=(288, 1, 1),
            output_shapes=oshapes,
            output_dtypes=odtypes,
        )
        return (outs[0], outs[1]) if also_raw else outs[0]
    kern = _make_swiglu_rht1152_kernel(out_dtype == mx.float16, pad_zero_row, skip_mode, also_raw)
    outs = kern(
        inputs=ins,
        grid=(128, n_slots, 1),
        threadgroup=(128, 1, 1),
        output_shapes=oshapes,
        output_dtypes=odtypes,
    )
    return (outs[0], outs[1]) if also_raw else outs[0]


@lru_cache(maxsize=4)
def _make_rht3072_gated_kernel(out_half: bool):
    ot = "half" if out_half else "float"
    src = f"""
    const uint t = threadgroup_position_in_grid.y;
    const uint c = simdgroup_index_in_threadgroup; // 0..11
    const uint lane = thread_index_in_simdgroup;   // 0..31
    const device float* ap = attn + (ulong)t * 3072u;
    const device float* gp = gate + (ulong)t * 3072u;
    device {ot}* op = out + (ulong)t * 3072u;

    threadgroup float tg_z[256 * 12];
    float y[8];
    #pragma unroll
    for (uint q = 0; q < 8u; q++) {{
        const uint idx = (lane + 32u * q) * 12u + c;
        const float gv = gp[idx];
        y[q] = ap[idx] * (1.0f / (1.0f + metal::precise::exp(-gv))) * signs[idx];
    }}
    for (uint h = 1u; h < 8u; h <<= 1) {{
        #pragma unroll
        for (uint q = 0; q < 8u; q++) {{
            if (!(q & h)) {{ const float u = y[q], w = y[q | h]; y[q] = u + w; y[q | h] = u - w; }}
        }}
    }}
    const bool h1 = lane & 1u, h2 = lane & 2u, h4 = lane & 4u, h8 = lane & 8u, h16 = lane & 16u;
    #pragma unroll
    for (uint q = 0; q < 8u; q++) {{
        float v = y[q], o;
        o = simd_shuffle_xor(v, 1u);  v = h1  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 2u);  v = h2  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 4u);  v = h4  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 8u);  v = h8  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 16u); v = h16 ? (o - v) : (v + o);
        tg_z[(lane + 32u * q) * 12u + c] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device float* br = B + c * 12u;
    constexpr float INV_S = 0.018042195912175808f; // 1/sqrt(3072)
    #pragma unroll
    for (uint q = 0; q < 8u; q++) {{
        const uint a = lane + 32u * q;
        const threadgroup float* zr = tg_z + a * 12u;
        float acc = 0.0f;
        #pragma unroll
        for (uint k = 0; k < 12u; k++) acc += zr[k] * br[k];
        op[a * 12u + c] = ({ot})(acc * INV_S);
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_rht3072_gated_{'h' if out_half else 'f'}",
        input_names=["attn", "gate", "signs", "B"],
        output_names=["out"],
        source=src,
    )


def rht_in_3072_gated_fast(attn_out: mx.array, gate_raw: mx.array, signs: mx.array, out_dtype=mx.float16) -> mx.array:
    T = int(attn_out.size) // 3072
    kern = _make_rht3072_gated_kernel(out_dtype == mx.float16)
    return kern(
        inputs=[attn_out.reshape(-1), gate_raw.reshape(-1), signs.reshape(-1), _paley_b_flat(3072)],
        grid=(384, T, 1),
        threadgroup=(384, 1, 1),
        output_shapes=[(T, 3072)],
        output_dtypes=[out_dtype],
    )[0]


# ----------------------------------------------------------------------------- Shared Format Decode Helpers (ch = 16)

def _emit_decode_ch16(
    a,
    fid: int,
    n: int,
    *,
    ptr_expr: str,
    wpr_expr: str,
    mode: str,
    t_or_c: int,
    x_expr_fn,
    acc_expr_fn,
    dequant_out_expr: str = "",
):
    """Emit ch=16 bit-unpack + LUT decode for ANY of the 9 formats (ht1..ht3, rtr1..rtr4, lm6, lm8).

    `mode`:
      - "dot": accumulates half dot products into `hs[b]` -> adds `(float)hs[b]` to `acc_expr_fn(b)`
      - "dequant": writes `16` scaled half weights to `dequant_out_expr` (`device half* outp`)
    """
    if fid in (0, 1):  # ht1 (k=1) or ht2 (k=2): g=4, K=512 hash in tg_ht1 / tg_ht2
        k = 1 if fid == 0 else 2
        wpr_c = (n * k) // 32
        tgn = "tg_ht1" if fid == 0 else "tg_ht2"
        a(f"            const uint B = ({n}u - 16u * (c + 1u)) * {k}u, w0 = B >> 5;")
        if k == 1:
            a("            const uint o = B & 31u;")
        a(f"            const device uint* p = {ptr_expr} + w0;")
        a(f"            uint w0v = p[0], w1v = (w0 + 1u < {wpr_c}u) ? p[1] : 0u;")
        if k == 1:
            a("            if (o != 0u) { w0v = (w0v >> 16u) | (w1v << 16u); w1v >>= 16u; }")
        if mode == "dot":
            for b in range(t_or_c):
                a(f"            half hs{b} = 0.0h;")
        for q in range(4):
            bit = (15 - (q * 4 + 3)) * k
            wi, off = bit >> 5, bit & 31
            ex = f"extract_bits(w{wi}v, {off}u, 12u)" if off + 12 <= 32 else f"insert_bits(extract_bits(w{wi}v, {off}u, {32-off}u), w{wi+1}v, {32-off}u, {12-(32-off)}u)"
            a(f"            {{ const half4 v = {tgn}[(({ex}) * {HASH_MULT_HEX}) >> 23u];")
            if mode == "dequant":
                a(f"              *(device half4*)({dequant_out_expr} + {q * 4}u) = half4(float4(v) * sc); }}")
            else:
                for b in range(t_or_c):
                    a(f"              hs{b} += dot(v, {x_expr_fn(b, q)});")
                a("            }")
        if mode == "dot":
            for b in range(t_or_c):
                a(f"            {acc_expr_fn(b)} += (float)hs{b};")

    elif fid == 2:  # ht3 (k=3, g=2, K=1024 hash in tg_ht3)
        wpr_c = (n * 3) // 32
        a(f"            const uint B = ({n}u - 16u * (c + 1u)) * 3u, w0 = B >> 5, o = B & 31u;")
        a(f"            const device uint* p = {ptr_expr} + w0;")
        a(f"            uint w0v = p[0], w1v = (w0 + 1u < {wpr_c}u) ? p[1] : 0u, w2v = (w0 + 2u < {wpr_c}u) ? p[2] : 0u;")
        a("            if (o != 0u) { w0v = (w0v >> 16u) | (w1v << 16u); w1v = (w1v >> 16u) | (w2v << 16u); w2v >>= 16u; }")
        if mode == "dot":
            for b in range(t_or_c):
                a(f"            half hs{b} = 0.0h;")
        for i in range(0, 16, 2):
            bit = (15 - (i + 1)) * 3
            wi, off = bit >> 5, bit & 31
            ex = f"extract_bits(w{wi}v, {off}u, 12u)" if off + 12 <= 32 else f"insert_bits(extract_bits(w{wi}v, {off}u, {32-off}u), w{wi+1}v, {32-off}u, {12-(32-off)}u)"
            a(f"            {{ const half2 v = tg_ht3[(({ex}) * {HASH_MULT_HEX}) >> 22u];")
            if mode == "dequant":
                a(f"              *(device half2*)({dequant_out_expr} + {i}u) = half2(float2(v) * sc); }}")
            else:
                q = i // 4
                sub = "xy" if (i % 4 == 0) else "zw"
                for b in range(t_or_c):
                    a(f"              hs{b} += dot(v, {x_expr_fn(b, q)}.{sub});")
                a("            }")
        if mode == "dot":
            for b in range(t_or_c):
                a(f"            {acc_expr_fn(b)} += (float)hs{b};")

    elif fid == 3:  # rtr1 (k=1, g=3, 14-bit multi-table in L1 tv_rtr1)
        wpr_c = n // 32
        a(f"            const uint B = ({n}u - 16u * (c + 1u)), w0 = B >> 5, o = B & 31u;")
        a(f"            const device uint* p = {ptr_expr} + w0;")
        a(f"            uint w0v = p[0], w1v = (w0 + 1u < {wpr_c}u) ? p[1] : 0u;")
        a("            if (o != 0u) { w0v = (w0v >> 16u) | (w1v << 16u); w1v >>= 16u; }")
        if mode == "dequant":
            a("            {")
            for ci, (i, gg) in enumerate([(0, 3), (3, 3), (6, 3), (9, 3), (12, 3), (15, 1)]):
                bit = 15 - (i + gg - 1)
                nb = 12 + (gg - 1)
                wi, off = bit >> 5, bit & 31
                ex = f"extract_bits(w{wi}v, {off}u, {nb}u)" if off + nb <= 32 else f"insert_bits(extract_bits(w{wi}v, {off}u, {32-off}u), w{wi+1}v, {32-off}u, {nb-(32-off)}u)"
                a(f"              const half4 v{ci} = tv_rtr1[{ex}];")
            a(f"              *(device half4*)({dequant_out_expr} + 0u)  = half4(float4(v0[0], v0[1], v0[2], v1[0]) * sc);")
            a(f"              *(device half4*)({dequant_out_expr} + 4u)  = half4(float4(v1[1], v1[2], v2[0], v2[1]) * sc);")
            a(f"              *(device half4*)({dequant_out_expr} + 8u)  = half4(float4(v2[2], v3[0], v3[1], v3[2]) * sc);")
            a(f"              *(device half4*)({dequant_out_expr} + 12u) = half4(float4(v4[0], v4[1], v4[2], v5[2]) * sc);")
            a("            }")
        else:
            for b in range(t_or_c):
                a(f"            half hs{b} = 0.0h;")
            i = 0
            while i < 16:
                gg = min(3, 16 - i)
                bit = 15 - (i + gg - 1)
                nb = 12 + (gg - 1)
                wi, off = bit >> 5, bit & 31
                ex = f"extract_bits(w{wi}v, {off}u, {nb}u)" if off + nb <= 32 else f"insert_bits(extract_bits(w{wi}v, {off}u, {32-off}u), w{wi+1}v, {32-off}u, {nb-(32-off)}u)"
                a(f"            {{ const half4 v = tv_rtr1[{ex}];")
                for b in range(t_or_c):
                    terms = " + ".join(f"v[{3 - gg + t}] * ({x_expr_fn(b, (i + t) // 4)})[{(i + t) % 4}]" for t in range(gg))
                    a(f"              hs{b} += {terms};")
                a("            }")
                i += gg
            for b in range(t_or_c):
                a(f"            {acc_expr_fn(b)} += (float)hs{b};")

    elif fid in (4, 5):  # rtr2 (k=2, g=2, 14-bit L1 tv_rtr2) or rtr3 (k=3, g=2, 15-bit L1 tv_rtr3)
        k = 2 if fid == 4 else 3
        nb = 12 + k
        wpr_c = (n * k) // 32
        tvn = "tv_rtr2" if fid == 4 else "tv_rtr3"
        a(f"            const uint B = ({n}u - 16u * (c + 1u)) * {k}u, w0 = B >> 5;")
        if k == 3:
            a("            const uint o = B & 31u;")
        a(f"            const device uint* p = {ptr_expr} + w0;")
        if k == 2:
            a(f"            const uint w0v = p[0], w1v = (w0 + 1u < {wpr_c}u) ? p[1] : 0u;")
        else:
            a(f"            uint w0v = p[0], w1v = (w0 + 1u < {wpr_c}u) ? p[1] : 0u, w2v = (w0 + 2u < {wpr_c}u) ? p[2] : 0u;")
            a("            if (o != 0u) { w0v = (w0v >> 16u) | (w1v << 16u); w1v = (w1v >> 16u) | (w2v << 16u); w2v >>= 16u; }")
        if mode == "dot":
            for b in range(t_or_c):
                a(f"            half hs{b} = 0.0h;")
        for i in range(0, 16, 2):
            bit = (15 - (i + 1)) * k
            wi, off = bit >> 5, bit & 31
            ex = f"extract_bits(w{wi}v, {off}u, {nb}u)" if off + nb <= 32 else f"insert_bits(extract_bits(w{wi}v, {off}u, {32-off}u), w{wi+1}v, {32-off}u, {nb-(32-off)}u)"
            a(f"            {{ const half2 v = {tvn}[{ex}];")
            if mode == "dequant":
                a(f"              *(device half2*)({dequant_out_expr} + {i}u) = half2(float2(v) * sc); }}")
            else:
                q = i // 4
                sub = "xy" if (i % 4 == 0) else "zw"
                for b in range(t_or_c):
                    a(f"              hs{b} += dot(v, {x_expr_fn(b, q)}.{sub});")
                a("            }")
        if mode == "dot":
            for b in range(t_or_c):
                a(f"            {acc_expr_fn(b)} += (float)hs{b};")

    elif fid == 6:  # rtr4 (k=4, g=1, 8KB L1 lut_rtr4)
        wpr_c = (n * 4) // 32
        a(f"            const uint B = ({n}u - 16u * (c + 1u)) * 4u, w0 = B >> 5;")
        a(f"            const device uint* p = {ptr_expr} + w0;")
        a(f"            const uint w0v = p[0], w1v = p[1], w2v = (w0 + 2u < {wpr_c}u) ? p[2] : 0u;")
        if mode == "dot":
            for b in range(t_or_c):
                a(f"            half hs{b} = 0.0h;")
        for q in range(4):
            a("            {")
            for u in range(4):
                i = q * 4 + u
                bit = (15 - i) * 4
                wi, off = bit >> 5, bit & 31
                ex = f"extract_bits(w{wi}v, {off}u, 12u)" if off + 12 <= 32 else f"insert_bits(extract_bits(w{wi}v, {off}u, {32-off}u), w{wi+1}v, {32-off}u, {12-(32-off)}u)"
                a(f"              const half v{u} = lut_rtr4[{ex}];")
            a("              const half4 wv = half4(v0, v1, v2, v3);")
            if mode == "dequant":
                a(f"              *(device half4*)({dequant_out_expr} + {q * 4}u) = half4(float4(wv) * sc); }}")
            else:
                for b in range(t_or_c):
                    a(f"              hs{b} += dot(wv, {x_expr_fn(b, q)});")
                a("            }")
        if mode == "dot":
            for b in range(t_or_c):
                a(f"            {acc_expr_fn(b)} += (float)hs{b};")

    elif fid == 10:  # lm6 (16 cols = 12 bytes = 3 uint32 words)
        a(f"            const device uint* p = {ptr_expr} + c * 3u;")
        a("            const uint u0 = p[0], u1 = p[1], u2 = p[2];")
        a("            const uint v0 = u0 & 0xFFFFFFu;")
        a("            const uint v1 = (u0 >> 24u) | ((u1 & 0xFFFFu) << 8u);")
        a("            const uint v2 = (u1 >> 16u) | ((u2 & 0xFFu) << 16u);")
        a("            const uint v3 = u2 >> 8u;")
        if mode == "dot":
            for b in range(t_or_c):
                a(f"            half hs{b} = 0.0h;")
        for q in range(4):
            a(f"            {{ const half4 wv = half4(lv[v{q} & 63u], lv[(v{q} >> 6u) & 63u], lv[(v{q} >> 12u) & 63u], lv[(v{q} >> 18u) & 63u]);")
            if mode == "dequant":
                a(f"              *(device half4*)({dequant_out_expr} + {q * 4}u) = half4(float4(wv) * sc); }}")
            else:
                for b in range(t_or_c):
                    a(f"              hs{b} += dot(wv, {x_expr_fn(b, q)});")
                a("            }")
        if mode == "dot":
            for b in range(t_or_c):
                a(f"            {acc_expr_fn(b)} += (float)hs{b};")

    elif fid == 11:  # lm8 (16 cols = 16 bytes = 4 uint32 words)
        a(f"            const device uint* p = {ptr_expr} + c * 4u;")
        a("            const uint u0 = p[0], u1 = p[1], u2 = p[2], u3 = p[3];")
        if mode == "dot":
            for b in range(t_or_c):
                a(f"            half hs{b} = 0.0h;")
        for q in range(4):
            a(f"            {{ const half4 wv = half4(lv[u{q} & 255u], lv[(u{q} >> 8u) & 255u], lv[(u{q} >> 16u) & 255u], lv[(u{q} >> 24u) & 255u]);")
            if mode == "dequant":
                a(f"              *(device half4*)({dequant_out_expr} + {q * 4}u) = half4(float4(wv) * sc); }}")
            else:
                for b in range(t_or_c):
                    a(f"              hs{b} += dot(wv, {x_expr_fn(b, q)});")
                a("            }")
        if mode == "dot":
            for b in range(t_or_c):
                a(f"            {acc_expr_fn(b)} += (float)hs{b};")
    else:
        raise ValueError(f"Unsupported fid={fid}")


# ----------------------------------------------------------------------------- Fused Multi-Matrix DenseStage

@lru_cache(maxsize=128)
def _make_dense_stage_kernel(n: int, fids: tuple[int, ...], t_blk: int, rps: int, sg: int):
    assert n % 256 == 0
    iters = n // 256
    has_ht1 = 0 in fids
    has_ht2 = 1 in fids
    has_ht3 = 2 in fids
    has_rtr1 = 3 in fids
    has_rtr2 = 4 in fids
    has_rtr3 = 5 in fids

    # Emit one clean inline helper per format so `switch (fid)` stays OUTSIDE the `for (it)` loop!
    H = [f"#define SG {sg}u\n#define RPS {rps}u\n#define TBLK {t_blk}u\n"]
    ha = H.append
    for fid in fids:
        ha(
            f"inline void dstage_f{fid}("
            "const device half* xin, const device uint* mcodes, const device half* mscales, const device half* lv, "
            "const threadgroup half4* tg_ht1, const threadgroup half4* tg_ht2, const threadgroup half2* tg_ht3, "
            "const device half4* tv_rtr1, const device half2* tv_rtr2, const device half2* tv_rtr3, const device half* lut_rtr4, "
            "device float* y, uint r0, uint m_rows, uint wpr, uint r_out_off, uint total_out, uint hl) {"
        )
        ha("    float acc[RPS][TBLK];")
        ha("    for (uint r = 0; r < RPS; r++) for (uint t = 0; t < TBLK; t++) acc[r][t] = 0.0f;")
        ha(f"    for (uint it = 0; it < {iters}u; it++) {{")
        ha("        const uint c = hl + 16u * it;")
        ha("        const uint col0 = c * 16u;")
        ha("        half4 xr[TBLK][4];")
        ha("        #pragma unroll")
        ha("        for (uint t = 0; t < TBLK; t++) {")
        ha(f"            const device half* xt = xin + (ulong)t * {n}u + col0;")
        for q in range(4):
            ha(f"            xr[t][{q}] = *(const device half4*)(xt + {q * 4}u);")
        ha("        }")
        ha("        #pragma unroll")
        ha("        for (uint r = 0; r < RPS; r++) {")
        ha("            const uint row = min(r0 + r, m_rows - 1u);")
        _emit_decode_ch16(
            ha,
            fid,
            n,
            ptr_expr="mcodes + (ulong)row * wpr",
            wpr_expr="wpr",
            mode="dot",
            t_or_c=t_blk,
            x_expr_fn=lambda b, q: f"xr[{b}][{q}]",
            acc_expr_fn=lambda b: f"acc[r][{b}]",
        )
        ha("        }")
        ha("    }")
        ha("    #pragma unroll")
        ha("    for (uint r = 0; r < RPS; r++) {")
        ha("        const float sc = (float)mscales[min(r0 + r, m_rows - 1u)];")
        ha("        #pragma unroll")
        ha("        for (uint t = 0; t < TBLK; t++) {")
        ha("            float s = acc[r][t];")
        ha("            s += simd_shuffle_xor(s, 8u); s += simd_shuffle_xor(s, 4u); s += simd_shuffle_xor(s, 2u); s += simd_shuffle_xor(s, 1u);")
        ha("            if (hl == 0u && r0 + r < m_rows) y[(ulong)t * total_out + r_out_off + r0 + r] = s * sc;")
        ha("        }")
        ha("    }")
        ha("}")

    L = []
    a = L.append
    a("const uint tg = threadgroup_position_in_grid.x;")
    a("const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;")
    a("const uint hw = lane >> 4, hl = lane & 15u;")
    a("const uint tid = simd * 32u + lane;")
    a("const uint total_out = meta[0], n_mats = meta[1], ntg = meta[2];")

    if has_ht1:
        a("threadgroup half4 tg_ht1[512];")
        a(f"for (uint idx = tid; idx < 512u; idx += {sg * 32}u) tg_ht1[idx] = half4(lut_ht1[4u*idx], lut_ht1[4u*idx+1u], lut_ht1[4u*idx+2u], lut_ht1[4u*idx+3u]);")
    else:
        a("threadgroup half4 tg_ht1[1];")
    if has_ht2:
        a("threadgroup half4 tg_ht2[512];")
        a(f"for (uint idx = tid; idx < 512u; idx += {sg * 32}u) tg_ht2[idx] = half4(lut_ht2[4u*idx], lut_ht2[4u*idx+1u], lut_ht2[4u*idx+2u], lut_ht2[4u*idx+3u]);")
    else:
        a("threadgroup half4 tg_ht2[1];")
    if has_ht3:
        a("threadgroup half2 tg_ht3[1024];")
        a(f"for (uint idx = tid; idx < 1024u; idx += {sg * 32}u) tg_ht3[idx] = half2(lut_ht3[2u*idx], lut_ht3[2u*idx+1u]);")
    else:
        a("threadgroup half2 tg_ht3[1];")
    if has_ht1 or has_ht2 or has_ht3:
        a("threadgroup_barrier(mem_flags::mem_threadgroup);")

    a("const device half4* tv_rtr1 = (const device half4*)lut_rtr1;")
    a("const device half2* tv_rtr2 = (const device half2*)lut_rtr2;")
    a("const device half2* tv_rtr3 = (const device half2*)lut_rtr3;")

    a("uint tot_blocks = 0u;")
    a("uint mat_blk_end[16];")
    a("for (uint mi = 0; mi < n_mats; mi++) {")
    a("    tot_blocks += (dims[mi * 8u + 1u] + RPS - 1u) / RPS;")
    a("    mat_blk_end[mi] = tot_blocks;")
    a("}")
    a("const uint blocks_per_tg = SG * 2u;")
    a("for (uint rb = tg; rb * blocks_per_tg < tot_blocks; rb += ntg) {")
    a("    const uint gb = rb * blocks_per_tg + simd * 2u + hw;")
    a("    if (gb >= tot_blocks) continue;")
    a("    uint mi = 0u, blk_base = 0u;")
    a("    while (gb >= mat_blk_end[mi]) { blk_base = mat_blk_end[mi]; mi++; }")
    a("    const device uint* d = dims + mi * 8u;")
    a("    const uint fid = d[0], m_rows = d[1], wpr = d[2], c_off = d[3], s_off = d[4], l_off = d[5], r_out_off = d[6], in_idx = d[7];")
    a("    const uint r0 = (gb - blk_base) * RPS;")
    a(f"    const device half* xin = x + (ulong)in_idx * TBLK * {n}u;")
    a("    switch (fid) {")
    for fid in fids:
        a(f"        case {fid}u: dstage_f{fid}(xin, codes + c_off, scales + s_off, levels + l_off, tg_ht1, tg_ht2, tg_ht3, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, y, r0, m_rows, wpr, r_out_off, total_out, hl); break;")
    a("    }")
    a("}")
    tag = "_".join(str(f) for f in fids)
    return mx.fast.metal_kernel(
        name=f"ink_dstage_n{n}_f{tag}_t{t_blk}_r{rps}_s{sg}",
        input_names=["x", "codes", "scales", "levels", "dims", "lut_ht1", "lut_ht2", "lut_ht3", "lut_rtr1", "lut_rtr2", "lut_rtr3", "lut_rtr4", "meta"],
        output_names=["y"],
        header="\n".join(H),
        source="\n".join(L),
    )


@lru_cache(maxsize=64)
def _make_dense_dequant_kernel(n: int, fids: tuple[int, ...], sg: int = 8):
    """Fast GPU dequantize of a DenseStage (or any subset of matrices) into float16[total_out, n] for T > 8 prefill GEMM!"""
    assert n % 256 == 0
    iters = n // 256
    has_ht1 = 0 in fids
    has_ht2 = 1 in fids
    has_ht3 = 2 in fids

    H = [f"#define SG {sg}u\n"]
    ha = H.append
    for fid in fids:
        ha(
            f"inline void ddeq_f{fid}("
            "const device uint* mcodes, const device half* mscales, const device half* lv, "
            "const threadgroup half4* tg_ht1, const threadgroup half4* tg_ht2, const threadgroup half2* tg_ht3, "
            "const device half4* tv_rtr1, const device half2* tv_rtr2, const device half2* tv_rtr3, const device half* lut_rtr4, "
            "device half* W, uint row, uint wpr, uint out_row, uint hl) {"
        )
        ha("    const float sc = (float)mscales[row];")
        ha(f"    device half* wrow = W + (ulong)out_row * {n}u;")
        ha(f"    for (uint it = 0; it < {iters}u; it++) {{")
        ha("        const uint c = hl + 16u * it;")
        _emit_decode_ch16(
            ha,
            fid,
            n,
            ptr_expr="mcodes + (ulong)row * wpr",
            wpr_expr="wpr",
            mode="dequant",
            t_or_c=0,
            x_expr_fn=None,
            acc_expr_fn=None,
            dequant_out_expr="wrow + c * 16u",
        )
        ha("    }")
        ha("}")

    L = []
    a = L.append
    a("const uint tg = threadgroup_position_in_grid.x;")
    a("const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;")
    a("const uint hw = lane >> 4, hl = lane & 15u;")
    a("const uint tid = simd * 32u + lane;")
    a("const uint total_out = meta[0], n_mats = meta[1], ntg = meta[2];")
    if has_ht1:
        a("threadgroup half4 tg_ht1[512];")
        a(f"for (uint idx = tid; idx < 512u; idx += {sg * 32}u) tg_ht1[idx] = half4(lut_ht1[4u*idx], lut_ht1[4u*idx+1u], lut_ht1[4u*idx+2u], lut_ht1[4u*idx+3u]);")
    else:
        a("threadgroup half4 tg_ht1[1];")
    if has_ht2:
        a("threadgroup half4 tg_ht2[512];")
        a(f"for (uint idx = tid; idx < 512u; idx += {sg * 32}u) tg_ht2[idx] = half4(lut_ht2[4u*idx], lut_ht2[4u*idx+1u], lut_ht2[4u*idx+2u], lut_ht2[4u*idx+3u]);")
    else:
        a("threadgroup half4 tg_ht2[1];")
    if has_ht3:
        a("threadgroup half2 tg_ht3[1024];")
        a(f"for (uint idx = tid; idx < 1024u; idx += {sg * 32}u) tg_ht3[idx] = half2(lut_ht3[2u*idx], lut_ht3[2u*idx+1u]);")
    else:
        a("threadgroup half2 tg_ht3[1];")
    if has_ht1 or has_ht2 or has_ht3:
        a("threadgroup_barrier(mem_flags::mem_threadgroup);")
    a("const device half4* tv_rtr1 = (const device half4*)lut_rtr1;")
    a("const device half2* tv_rtr2 = (const device half2*)lut_rtr2;")
    a("const device half2* tv_rtr3 = (const device half2*)lut_rtr3;")

    a("uint mat_row_end[16];")
    a("uint acc_r = 0u;")
    a("for (uint mi = 0; mi < n_mats; mi++) { acc_r += dims[mi * 8u + 1u]; mat_row_end[mi] = acc_r; }")
    a("const uint rows_per_tg = SG * 2u;")
    a("for (uint rb = tg; rb * rows_per_tg < total_out; rb += ntg) {")
    a("    const uint grow = rb * rows_per_tg + simd * 2u + hw;")
    a("    if (grow >= total_out) continue;")
    a("    uint mi = 0u, base_r = 0u;")
    a("    while (grow >= mat_row_end[mi]) { base_r = mat_row_end[mi]; mi++; }")
    a("    const device uint* d = dims + mi * 8u;")
    a("    const uint fid = d[0], wpr = d[2], c_off = d[3], s_off = d[4], l_off = d[5], r_out_off = d[6];")
    a("    const uint row = grow - base_r;")
    a("    switch (fid) {")
    for fid in fids:
        a(f"        case {fid}u: ddeq_f{fid}(codes + c_off, scales + s_off, levels + l_off, tg_ht1, tg_ht2, tg_ht3, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, W, row, wpr, r_out_off + row, hl); break;")
    a("    }")
    a("}")
    tag = "_".join(str(f) for f in fids)
    return mx.fast.metal_kernel(
        name=f"ink_ddeq_n{n}_f{tag}_s{sg}",
        input_names=["codes", "scales", "levels", "dims", "lut_ht1", "lut_ht2", "lut_ht3", "lut_rtr1", "lut_rtr2", "lut_rtr3", "lut_rtr4", "meta"],
        output_names=["W"],
        header="\n".join(H),
        source="\n".join(L),
    )


@lru_cache(maxsize=64)
def _make_sg8_dense_stage_kernel(n: int, fids: tuple[int, ...], rb: int = 2, sg: int = 32):
    """simdgroup_half8x8 DenseStage kernel for T=8 with L1 LUTs, 32 KB rtr1_g2, and sliding w_prev."""
    assert n % 32 == 0
    steps = n // 32
    H = [
        "#include <metal_simdgroup_matrix>\n",
        f"#define SG {sg}u\n#define RB {rb}u\n",
        "#define ELEMS_H2 reinterpret_cast<thread half2&>\n",
        "#define ELEMS_F2 reinterpret_cast<thread float2&>\n",
    ]
    ha = H.append
    for fid in fids:
        ha(
            f"inline void sg8_f{fid}("
            "const device half* xin, const device uint* mcodes, const device half* mscales, const device half* lv, "
            "const device half4* t_ht1, const device half4* t_ht2, const device half2* t_ht3, "
            "const device half2* tv_rtr1, const device half2* tv_rtr2, const device half2* tv_rtr3, const device half* lut_rtr4, "
            "device float* y, uint rbase, uint m_rows, uint wpr, uint r_out_off, uint total_out, uint t_out, uint lane) {"
        )
        ha("    const uint qid = lane / 4u, fm = (qid & 4u) + ((lane / 2u) % 4u), fn = (qid & 2u) * 2u + (lane % 2u) * 2u;")
        ha("    const uint q = fn / 2u, xcol = 8u * (fm / 2u) + (fm % 2u), r0 = rbase + fm;")
        ha("    const device uint* p[RB];")
        ha("    for (uint i = 0; i < RB; i++) p[i] = mcodes + (ulong)min(r0 + 8u * i, m_rows - 1u) * wpr;")
        ha("    simdgroup_float8x8 Acc[RB];")
        ha("    for (uint i = 0; i < RB; i++) Acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);")
        if fid in (0, 1, 2, 3, 4, 5, 6):
            k = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4}[fid]
            wpr_c = (n * k) // 32
            ha(f"    const uint first = ({n}u - 8u - 8u * q) * {k}u, w_first = first >> 5, off = first & 31u;")
            if fid == 3:
                ha("    uint w_prev[RB];")
                ha(f"    for (uint i = 0; i < RB; i++) w_prev[i] = (w_first + 1u < {wpr_c}u) ? p[i][w_first + 1u] : 0u;")
        elif fid == 10:
            ha("    const uint first = 8u * q * 6u, w_first = first >> 5, off = first & 31u;")
        elif fid == 11:
            ha("    const uint w_first = 2u * q;")
        ha(f"    for (uint s = 0; s < {steps}u; s++) {{")
        ha("        simdgroup_half8x8 Xm[4];")
        ha("        const device half* xp = xin + (ulong)(32u * s + xcol) * 8u + fn;")
        ha("        #pragma unroll")
        ha("        for (uint j = 0; j < 4u; j++) ELEMS_H2(Xm[j].thread_elements()) = half2(*(const device packed_half2*)(xp + j * 16u));")
        if fid in (0, 1, 2, 3, 4, 5, 6):
            ha(f"        const uint wi = w_first - s * {k}u;")
        elif fid == 10:
            ha("        const uint wi = w_first + s * 6u;")
        elif fid == 11:
            ha("        const uint wi = w_first + s * 8u;")
        ha("        #pragma unroll")
        ha("        for (uint i = 0; i < RB; i++) {")
        ha("            const device uint* pw = p[i] + wi; simdgroup_half8x8 Am;")
        if fid in (0, 1):
            tgn = "t_ht1" if fid == 0 else "t_ht2"
            if fid == 1:
                ha(f"            const uint w0 = pw[0]; const uint lo = (off == 0u) ? w0 : ((w0 >> 16u) | (((wi + 1u < {wpr_c}u) ? pw[1] : 0u) << 16u));")
            else:
                ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u; const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
            ha(f"            const half4 v0 = {tgn}[((extract_bits(lo, {4*k}u, 12u)) * {HASH_MULT_HEX}) >> 23u];")
            ha(f"            const half4 v1 = {tgn}[((extract_bits(lo, 0u, 12u)) * {HASH_MULT_HEX}) >> 23u];")
            for j, expr in [(0, "v0.xy"), (1, "v0.zw"), (2, "v1.xy"), (3, "v1.zw")]:
                ha(f"            ELEMS_H2(Am.thread_elements()) = {expr}; simdgroup_multiply_accumulate(Acc[i], Am, Xm[{j}], Acc[i]);")
        elif fid == 2:
            ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u;")
            ha("            const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
            for j, rb_b in [(0, 18), (1, 12), (2, 6), (3, 0)]:
                ha(f"            ELEMS_H2(Am.thread_elements()) = t_ht3[(extract_bits(lo, {rb_b}u, 12u) * {HASH_MULT_HEX}) >> 22u]; simdgroup_multiply_accumulate(Acc[i], Am, Xm[{j}], Acc[i]);")
        elif fid == 3:
            ha("            const uint w0 = pw[0]; const uint lo = (off == 0u) ? w0 : ((w0 >> off) | (w_prev[i] << (32u - off))); w_prev[i] = w0;")
            for j, sh in [(0, 6), (1, 4), (2, 2), (3, 0)]:
                ha(f"            ELEMS_H2(Am.thread_elements()) = tv_rtr1[extract_bits(lo, {sh}u, 13u)]; simdgroup_multiply_accumulate(Acc[i], Am, Xm[{j}], Acc[i]);")
        elif fid in (4, 5):
            tvn = "tv_rtr2" if fid == 4 else "tv_rtr3"
            ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u;")
            if k == 2:
                ha("            const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
                for j, rb_b in [(0, 12), (1, 8), (2, 4), (3, 0)]:
                    ha(f"            ELEMS_H2(Am.thread_elements()) = {tvn}[extract_bits(lo, {rb_b}u, 14u)]; simdgroup_multiply_accumulate(Acc[i], Am, Xm[{j}], Acc[i]);")
            else:
                ha(f"            const uint w2 = (wi + 2u < {wpr_c}u) ? pw[2] : 0u;")
                ha("            const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
                ha("            const uint hi = (w1 >> off) | (off == 0u ? 0u : (w2 << (32u - off)));")
                for j, rb_b in [(0, 18), (1, 12), (2, 6), (3, 0)]:
                    ex = f"extract_bits(lo, {rb_b}u, 15u)" if rb_b + 15 <= 32 else f"insert_bits(extract_bits(lo, {rb_b}u, {32-rb_b}u), hi, {32-rb_b}u, {15-(32-rb_b)}u)"
                    ha(f"            ELEMS_H2(Am.thread_elements()) = {tvn}[{ex}]; simdgroup_multiply_accumulate(Acc[i], Am, Xm[{j}], Acc[i]);")
        elif fid == 6:
            ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u;")
            for j, (b0, b1) in [(0, (28, 24)), (1, (20, 16)), (2, (12, 8)), (3, (4, 0))]:
                ex0 = f"extract_bits(w0, {b0}u, 12u)" if b0 + 12 <= 32 else f"insert_bits(extract_bits(w0, {b0}u, {32-b0}u), w1, {32-b0}u, {12-(32-b0)}u)"
                ex1 = f"extract_bits(w0, {b1}u, 12u)" if b1 + 12 <= 32 else f"insert_bits(extract_bits(w0, {b1}u, {32-b1}u), w1, {32-b1}u, {12-(32-b1)}u)"
                ha(f"            ELEMS_H2(Am.thread_elements()) = half2(lut_rtr4[{ex0}], lut_rtr4[{ex1}]); simdgroup_multiply_accumulate(Acc[i], Am, Xm[{j}], Acc[i]);")
        elif fid == 10:
            ha("            const uint w0 = pw[0], w1 = pw[1];")
            ha("            const uint lo = (off == 0u) ? w0 : ((w0 >> 16u) | (w1 << 16u)), hi = (off == 0u) ? w1 : (w1 >> 16u);")
            ha("            ELEMS_H2(Am.thread_elements()) = half2(lv[lo & 63u], lv[(lo >> 6u) & 63u]); simdgroup_multiply_accumulate(Acc[i], Am, Xm[0], Acc[i]);")
            ha("            ELEMS_H2(Am.thread_elements()) = half2(lv[(lo >> 12u) & 63u], lv[(lo >> 18u) & 63u]); simdgroup_multiply_accumulate(Acc[i], Am, Xm[1], Acc[i]);")
            ha("            ELEMS_H2(Am.thread_elements()) = half2(lv[(lo >> 24u) & 63u], lv[((lo >> 30u) | (hi << 2u)) & 63u]); simdgroup_multiply_accumulate(Acc[i], Am, Xm[2], Acc[i]);")
            ha("            ELEMS_H2(Am.thread_elements()) = half2(lv[(hi >> 4u) & 63u], lv[(hi >> 10u) & 63u]); simdgroup_multiply_accumulate(Acc[i], Am, Xm[3], Acc[i]);")
        elif fid == 11:
            ha("            const uint w0 = pw[0], w1 = pw[1];")
            ha("            ELEMS_H2(Am.thread_elements()) = half2(lv[w0 & 255u], lv[(w0 >> 8u) & 255u]); simdgroup_multiply_accumulate(Acc[i], Am, Xm[0], Acc[i]);")
            ha("            ELEMS_H2(Am.thread_elements()) = half2(lv[(w0 >> 16u) & 255u], lv[(w0 >> 24u) & 255u]); simdgroup_multiply_accumulate(Acc[i], Am, Xm[1], Acc[i]);")
            ha("            ELEMS_H2(Am.thread_elements()) = half2(lv[w1 & 255u], lv[(w1 >> 8u) & 255u]); simdgroup_multiply_accumulate(Acc[i], Am, Xm[2], Acc[i]);")
            ha("            ELEMS_H2(Am.thread_elements()) = half2(lv[(w1 >> 16u) & 255u], lv[(w1 >> 24u) & 255u]); simdgroup_multiply_accumulate(Acc[i], Am, Xm[3], Acc[i]);")
        ha("        }")
        ha("    }")
        ha("    for (uint i = 0; i < RB; i++) {")
        ha("        const uint row = r0 + 8u * i;")
        ha("        if (row >= m_rows) continue;")
        ha("        const float sc = (float)mscales[row];")
        ha("        const float2 d = ELEMS_F2(Acc[i].thread_elements()) * sc;")
        ha("        if (fn < t_out) y[(ulong)fn * total_out + r_out_off + row] = d.x;")
        ha("        if (fn + 1u < t_out) y[(ulong)(fn + 1u) * total_out + r_out_off + row] = d.y;")
        ha("    }")
        ha("}")

    L = []
    a = L.append
    a("const uint tg = threadgroup_position_in_grid.x, simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;")
    a("const uint total_out = meta[0], n_mats = meta[1], ntg = meta[2], t_out = meta[3];")
    a("const device half4* t_ht1 = (const device half4*)lut_ht1;")
    a("const device half4* t_ht2 = (const device half4*)lut_ht2;")
    a("const device half2* t_ht3 = (const device half2*)lut_ht3;")
    a("const device half2* tv_rtr1 = (const device half2*)lut_rtr1;")
    a("const device half2* tv_rtr2 = (const device half2*)lut_rtr2;")
    a("const device half2* tv_rtr3 = (const device half2*)lut_rtr3;")
    a("uint tot_blocks = 0u, mat_blk_end[16];")
    a("for (uint mi = 0; mi < n_mats; mi++) { tot_blocks += (dims[mi*8u+1u] + (8u*RB) - 1u) / (8u*RB); mat_blk_end[mi] = tot_blocks; }")
    a("for (uint rb = tg; rb * SG < tot_blocks; rb += ntg) {")
    a("    const uint gb = rb * SG + simd;")
    a("    if (gb >= tot_blocks) continue;")
    a("    uint mi = 0u, blk_base = 0u;")
    a("    while (gb >= mat_blk_end[mi]) { blk_base = mat_blk_end[mi]; mi++; }")
    a("    const device uint* d = dims + mi * 8u;")
    a("    const uint fid = d[0], m_rows = d[1], wpr = d[2], c_off = d[3], s_off = d[4], l_off = d[5], r_out_off = d[6], in_idx = d[7];")
    a("    const uint rbase = (gb - blk_base) * (8u * RB);")
    a(f"    const device half* xin = x + (ulong)in_idx * {n * 8}u;")
    a("    switch (fid) {")
    for fid in fids:
        a(f"        case {fid}u: sg8_f{fid}(xin, codes+c_off, scales+s_off, levels+l_off, t_ht1, t_ht2, t_ht3, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, y, rbase, m_rows, wpr, r_out_off, total_out, t_out, lane); break;")
    a("    }")
    a("}")
    tag = "_".join(str(f) for f in fids)
    return mx.fast.metal_kernel(
        name=f"ink_sg8_dstage_v2_n{n}_f{tag}_rb{rb}_sg{sg}",
        input_names=["x", "codes", "scales", "levels", "dims", "lut_ht1", "lut_ht2", "lut_ht3", "lut_rtr1", "lut_rtr2", "lut_rtr3", "lut_rtr4", "meta"],
        output_names=["y"],
        header="\n".join(H),
        source="\n".join(L),
    )


class DenseStage:
    """Packs one or more `PackedLinear` / `LMHead` bucket matrices with identical `n` into ONE GPU kernel launch."""

    def __init__(self, items: Sequence[dict], n: int, fast_luts: dict[str, mx.array]):
        self.n = int(n)
        self.fast_luts = fast_luts
        codes_parts, scales_parts, levels_parts, signs_parts = [], [], [], []
        dims = []
        self.slices: list[tuple[int, int]] = []
        self.out_rots: list[tuple[int, int, mx.array]] = []
        c_off = s_off = l_off = r_off = 0
        fids = set()
        for idx, it in enumerate(items):
            fmt = it["fmt"]
            fid = FMT_ID[fmt]
            fids.add(fid)
            m_rows = int(it["m"])
            if fmt in ("lm6", "lm8"):
                raw_u8 = np.ascontiguousarray(np.array(it["codes"], dtype=np.uint8)).reshape(m_rows, -1)
                assert raw_u8.shape[1] % 4 == 0
                c_u32 = raw_u32 = np.ascontiguousarray(raw_u8).view(np.uint32)
                lv = np.array(it["levels"], dtype=np.float16).reshape(-1)
            else:
                c_u32 = np.ascontiguousarray(np.array(it["codes"], dtype=np.uint32)).reshape(m_rows, -1)
                lv = np.zeros(0, dtype=np.float16)
            wpr = int(c_u32.shape[1])
            dims.append([fid, m_rows, wpr, c_off, s_off, l_off, r_off, idx])
            self.slices.append((r_off, r_off + m_rows))
            if it.get("out_rot") and it.get("signs_out") is not None:
                self.out_rots.append((r_off, r_off + m_rows, it["signs_out"]))
            codes_parts.append(c_u32.reshape(-1))
            scales_parts.append(np.array(it["scales"], dtype=np.float16).reshape(-1))
            if len(lv):
                levels_parts.append(lv)
            if "signs_in" in it and it["signs_in"] is not None:
                signs_parts.append(np.array(it["signs_in"], dtype=np.float32).reshape(1, self.n))
            c_off += c_u32.size
            s_off += m_rows
            l_off += len(lv)
            r_off += m_rows
        codes_parts.append(np.zeros(4, dtype=np.uint32))
        if not levels_parts:
            levels_parts.append(np.zeros(512, dtype=np.float16))
        self.codes = mx.array(np.concatenate(codes_parts))
        self.scales = mx.array(np.concatenate(scales_parts))
        self.levels = mx.array(np.concatenate(levels_parts))
        self.dims = mx.array(np.array(dims, dtype=np.uint32).reshape(-1))
        self.signs_in = mx.array(np.concatenate(signs_parts, axis=0)) if signs_parts else None
        self.total_out = r_off
        self.n_mats = len(items)
        self.fids = tuple(sorted(fids))
        self._lut_args = [
            fast_luts["ht1"], fast_luts["ht2"], fast_luts["ht3"],
            fast_luts["rtr1_g3"], fast_luts["rtr2_g2"], fast_luts["rtr3_g2"], fast_luts["rtr4"],
        ]
        self._lut_args_sg8 = [
            fast_luts["ht1"], fast_luts["ht2"], fast_luts["ht3"],
            fast_luts["rtr1_g2"], fast_luts["rtr2_g2"], fast_luts["rtr3_g2"], fast_luts["rtr4"],
        ]
        mx.eval(self.codes, self.scales, self.levels, self.dims)
        if self.signs_in is not None:
            mx.eval(self.signs_in)

    def dequant_f16(self, ntg: int | None = None) -> mx.array:
        if ntg is None:
            ntg = 10 if _is_m4() else 160
        sg = 8
        kern = _make_dense_dequant_kernel(self.n, self.fids, sg=sg)
        mt = _u32(self.total_out, self.n_mats, ntg)
        return kern(
            inputs=[self.codes, self.scales, self.levels, self.dims, *self._lut_args, mt],
            grid=(ntg * sg * 32, 1, 1),
            threadgroup=(sg * 32, 1, 1),
            output_shapes=[(self.total_out, self.n)],
            output_dtypes=[mx.float16],
        )[0]

    def matmul_rotated_sg8_f16(
        self,
        x_rot_tr_f16: mx.array,
        T: int = 8,
        rb: int | None = None,
        sg: int | None = None,
        ntg: int | None = None,
    ) -> mx.array:
        """Fast T=2..8 simdgroup_half8x8 path for QKV and 200K LM Head. Input layout: `[n_mats, n, 8]` half."""
        m4 = _is_m4()
        sg_limit = 32 if _has_mpp() else 8
        rb = 2 if rb is None else rb
        sg = min(sg_limit, (8 if m4 else sg_limit) if sg is None else sg)
        if ntg is None:
            ntg = (20 if self.total_out > 10000 else 30) if m4 else 80
        kern = _make_sg8_dense_stage_kernel(self.n, self.fids, rb=rb, sg=sg)
        mt = _u32(self.total_out, self.n_mats, ntg, T)
        return kern(
            inputs=[x_rot_tr_f16.reshape(-1), self.codes, self.scales, self.levels, self.dims, *self._lut_args_sg8, mt],
            grid=(ntg * sg * 32, 1, 1),
            threadgroup=(sg * 32, 1, 1),
            output_shapes=[(T, self.total_out)],
            output_dtypes=[mx.float32],
        )[0]

    def matmul_rotated_f16(self, x_rot_f16: mx.array, T: int, W_cached: mx.array | None = None) -> mx.array:
        """Execute `x_rot_f16` (shape `[n_mats, T, n]` or `[T, n]` for n_mats==1) -> `float32[T, total_out]`."""
        if T == 1:
            if _is_m4():
                if self.total_out > 10000:
                    rps, sg, ntg = 2, 4, 60
                elif self.total_out <= 3072:
                    rps, sg, ntg = 2, 4, 160
                else:
                    rps, sg, ntg = 1, 4, 160
            else:
                rps, sg, ntg = (2, 4, 160) if self.total_out <= 3072 else (2, 8, 160)
            kern = _make_dense_stage_kernel(self.n, self.fids, 1, rps=rps, sg=sg)
            mt = _u32(self.total_out, self.n_mats, ntg)
            return kern(
                inputs=[x_rot_f16.reshape(-1), self.codes, self.scales, self.levels, self.dims, *self._lut_args, mt],
                grid=(ntg * sg * 32, 1, 1),
                threadgroup=(sg * 32, 1, 1),
                output_shapes=[(1, self.total_out)],
                output_dtypes=[mx.float32],
            )[0]
        if T <= 8:
            rps, sg, ntg = (2, 4, 40) if (_is_m4() and self.total_out <= 3072) else (1, 8, 160)
            kern = _make_dense_stage_kernel(self.n, self.fids, T, rps=rps, sg=sg)
            mt = _u32(self.total_out, self.n_mats, ntg)
            return kern(
                inputs=[x_rot_f16.reshape(-1), self.codes, self.scales, self.levels, self.dims, *self._lut_args, mt],
                grid=(ntg * sg * 32, 1, 1),
                threadgroup=(sg * 32, 1, 1),
                output_shapes=[(T, self.total_out)],
                output_dtypes=[mx.float32],
            )[0]
        # T > 8 prefill: fast GPU dequant to FP16 (or reuse W_cached across loop passes!) + Hardware GEMM
        W = W_cached if W_cached is not None else self.dequant_f16()
        if self.n_mats == 1:
            return (x_rot_f16.reshape(T, self.n) @ W.T).astype(mx.float32)
        x3 = x_rot_f16.reshape(self.n_mats, T, self.n)
        parts = [x3[idx] @ W[lo:hi].T for idx, (lo, hi) in enumerate(self.slices)]
        return mx.concatenate(parts, axis=-1).astype(mx.float32)

    def matmul_rotated_f16_parts(
        self,
        x_rot_f16: mx.array,
        T: int,
        W_cached: mx.array | None = None,
        out_dtype=mx.float32,
    ) -> tuple[mx.array, ...]:
        """Return separate per-matrix outputs without `mx.concatenate` + re-slicing copies (supports `W_cached` reuse)."""
        if T <= 8:
            full = self.matmul_rotated_f16(x_rot_f16, T)
            if out_dtype != mx.float32:
                full = full.astype(out_dtype)
            return tuple(full[:, lo:hi] for lo, hi in self.slices)
        W = W_cached if W_cached is not None else self.dequant_f16()
        if self.n_mats == 1:
            y = x_rot_f16.reshape(T, self.n) @ W.T
            return (y if out_dtype == mx.float16 else y.astype(out_dtype),)
        x3 = x_rot_f16.reshape(self.n_mats, T, self.n)
        if out_dtype == mx.float16:
            return tuple(x3[idx] @ W[lo:hi].T for idx, (lo, hi) in enumerate(self.slices))
        return tuple((x3[idx] @ W[lo:hi].T).astype(out_dtype) for idx, (lo, hi) in enumerate(self.slices))


# ----------------------------------------------------------------------------- Unified 1-Launch Multi-Format MoE

def _emit_sg8_moe_mma(ha, fid: int, n: int, nb_tiles: int):
    """DRY helper emitting the 4 simdgroup_half8x8 decode + MMA steps for any MoE format `fid` in {0..6}."""
    k = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4}[fid]
    wpr_c = (n * k) // 32

    def mma_line(j: int):
        if nb_tiles == 1:
            return f"simdgroup_multiply_accumulate(Acc[i], Am, Xm[{j}], Acc[i]);"
        return (
            f"\n            #pragma unroll\n"
            f"            for (uint b = 0; b < {nb_tiles}u; b++) simdgroup_multiply_accumulate(Acc[i][b], Am, Xm[b][{j}], Acc[i][b]);"
        )

    if fid in (0, 1):
        tgn = "t_ht1" if fid == 0 else "t_ht2"
        if fid == 1:
            ha(f"            const uint w0 = pw[0]; const uint lo = (off == 0u) ? w0 : ((w0 >> 16u) | (((wi + 1u < {wpr_c}u) ? pw[1] : 0u) << 16u));")
        else:
            ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u; const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
        ha(f"            const half4 v0 = {tgn}[((extract_bits(lo, {4 * k}u, 12u)) * {HASH_MULT_HEX}) >> 23u];")
        ha(f"            const half4 v1 = {tgn}[((extract_bits(lo, 0u, 12u)) * {HASH_MULT_HEX}) >> 23u];")
        for j, expr in [(0, "v0.xy"), (1, "v0.zw"), (2, "v1.xy"), (3, "v1.zw")]:
            ha(f"            ELEMS_H2(Am.thread_elements()) = {expr}; {mma_line(j)}")
    elif fid == 2:
        ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u;")
        ha("            const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
        for j, rb_b in [(0, 18), (1, 12), (2, 6), (3, 0)]:
            ha(f"            ELEMS_H2(Am.thread_elements()) = t_ht3[(extract_bits(lo, {rb_b}u, 12u) * {HASH_MULT_HEX}) >> 22u]; {mma_line(j)}")
    elif fid == 3:
        ha("            const uint w0 = pw[0]; const uint lo = (off == 0u) ? w0 : ((w0 >> off) | (w_prev[i] << (32u - off))); w_prev[i] = w0;")
        for j, sh in [(0, 6), (1, 4), (2, 2), (3, 0)]:
            ha(f"            ELEMS_H2(Am.thread_elements()) = tv_rtr1[extract_bits(lo, {sh}u, 13u)]; {mma_line(j)}")
    elif fid == 4:
        ha(f"            const uint w0 = pw[0]; const uint lo = (off == 0u) ? w0 : ((w0 >> 16u) | (((wi + 1u < {wpr_c}u) ? pw[1] : 0u) << 16u));")
        for j, rb_b in [(0, 12), (1, 8), (2, 4), (3, 0)]:
            ha(f"            ELEMS_H2(Am.thread_elements()) = tv_rtr2[extract_bits(lo, {rb_b}u, 14u)]; {mma_line(j)}")
    elif fid == 5:
        ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u, w2 = (wi + 2u < {wpr_c}u) ? pw[2] : 0u;")
        ha("            const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
        ha("            const uint hi = (w1 >> off) | (off == 0u ? 0u : (w2 << (32u - off)));")
        for j, rb_b in [(0, 18), (1, 12), (2, 6), (3, 0)]:
            ex = f"extract_bits(lo, {rb_b}u, 15u)" if rb_b + 15 <= 32 else f"insert_bits(extract_bits(lo, {rb_b}u, {32 - rb_b}u), hi, {32 - rb_b}u, {15 - (32 - rb_b)}u)"
            ha(f"            ELEMS_H2(Am.thread_elements()) = tv_rtr3[{ex}]; {mma_line(j)}")
    elif fid == 6:
        ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u;")
        for j, (b0, b1) in [(0, (28, 24)), (1, (20, 16)), (2, (12, 8)), (3, (4, 0))]:
            ex0 = f"extract_bits(w0, {b0}u, 12u)" if b0 + 12 <= 32 else f"insert_bits(extract_bits(w0, {b0}u, {32 - b0}u), w1, {32 - b0}u, {12 - (32 - b0)}u)"
            ex1 = f"extract_bits(w0, {b1}u, 12u)" if b1 + 12 <= 32 else f"insert_bits(extract_bits(w0, {b1}u, {32 - b1}u), w1, {32 - b1}u, {12 - (32 - b1)}u)"
            ha(f"            ELEMS_H2(Am.thread_elements()) = half2(lut_rtr4[{ex0}], lut_rtr4[{ex1}]); {mma_line(j)}")


@lru_cache(maxsize=64)
def _make_unified_moe_kernel(n: int, fids: tuple[int, ...], t_mode: str, rps: int, sg: int, rb: int = 2, nb: int = 2, n_rg: int = 1, lpr: int | None = None):
    """Unified 1-launch MoE kernel supporting ALL formats {ht1,ht2,ht3,rtr1..rtr4,nvfp4} and optional row-group splitting (`n_rg`).

    Modes:
      - `t1`: T=1 (8 active slots, 1 token/expert, zero compaction, half-warp c1)
      - `grouped`: T=2..8 (up to 64 slots, pre-compacted plan, low-reg c3/c4, lpr=8 RPS=4 SG=8 -> 12.6ms!)
      - `prefill`: T>8 (arbitrary N*8 slots via GPU-sorted balanced 16-token work units, NB=2 simdgroup_half8x8!)
    """
    if lpr is None:
        lpr = 8 if (t_mode == "grouped" or n == 1152) else 16
    assert n % (lpr * 16) == 0
    iters = n // (lpr * 16)
    groups_per_simd = 32 // lpr
    shift_lane = int(round(math.log2(lpr)))
    mask_lane = lpr - 1
    has_ht1 = 0 in fids
    has_ht2 = 1 in fids
    has_ht3 = 2 in fids
    has_nv = 7 in fids

    if t_mode == "grouped":
        # Pre-planned lean half-warp mc=4 with low-register 2-pass c3/c4 and zero compaction inside MoE!
        H = [f"#define SG {sg}u\n#define RPS {rps}u\n"]
        ha = H.append
        for fid in fids:
            for c_val in (1, 2, 3, 4):
                inner_rps = (rps // 2) if (c_val >= 3 and rps >= 4) else rps
                n_passes = rps // inner_rps
                if fid == 7:
                    wpr_nv, spr_nv = n // 8, n // 16
                    ha(
                        f"inline void umoe_pp7_c{c_val}("
                        "const device half* x_raw, const device uint* ecodes, const device uchar* env_scales, float eouter, "
                        "const threadgroup half* tg_e2m1, const threadgroup half* tg_e4m3, "
                        "device float* y, uint r0, uint m_rows, uint y_stride, uint x_div, uint packed, uint hl) {"
                    )
                    for b in range(c_val):
                        ha(f"    const uint s{b} = (packed >> {b * 8}u) & 0xFFu;")
                        ha(f"    const device half* xp{b} = x_raw + (ulong)((x_div == 1u) ? s{b} : ((x_div == 8u) ? (s{b} >> 3) : (s{b} / x_div))) * {n}u;")
                    if n_passes > 1:
                        ha(f"    for (uint pp = 0; pp < {n_passes}u; pp++) {{ const uint rb0 = r0 + pp * {inner_rps}u;")
                    else:
                        ha("    { const uint rb0 = r0;")
                    for b in range(c_val):
                        ha(f"        float acc{b}[{inner_rps}]; for (uint r = 0; r < {inner_rps}u; r++) acc{b}[r] = 0.0f;")
                    ha(f"        for (uint it = 0; it < {iters}u; it++) {{")
                    ha(f"            const uint c = hl + {lpr}u * it, col0 = c * 16u;")
                    for b in range(c_val):
                        ha(f"            float4 x{b}[4];")
                        for q in range(4):
                            ha(f"            x{b}[{q}] = float4(*(const device half4*)(xp{b} + col0 + {q * 4}u));")
                    ha(f"            #pragma unroll\n            for (uint r = 0; r < {inner_rps}u; r++) {{")
                    ha("                const uint row = min(rb0 + r, m_rows - 1u);")
                    ha(f"                const device uint* p = ecodes + (ulong)row * {wpr_nv}u + 2u * c;")
                    ha("                const uint u0 = p[0], u1 = p[1];")
                    ha(f"                const float bsc = (float)tg_e4m3[env_scales[(ulong)row * {spr_nv}u + c]];")
                    ha("                const float4 wv0 = float4(tg_e2m1[u0 & 15u], tg_e2m1[(u0 >> 4u) & 15u], tg_e2m1[(u0 >> 8u) & 15u], tg_e2m1[(u0 >> 12u) & 15u]);")
                    ha("                const float4 wv1 = float4(tg_e2m1[(u0 >> 16u) & 15u], tg_e2m1[(u0 >> 20u) & 15u], tg_e2m1[(u0 >> 24u) & 15u], tg_e2m1[u0 >> 28u]);")
                    ha("                const float4 wv2 = float4(tg_e2m1[u1 & 15u], tg_e2m1[(u1 >> 4u) & 15u], tg_e2m1[(u1 >> 8u) & 15u], tg_e2m1[(u1 >> 12u) & 15u]);")
                    ha("                const float4 wv3 = float4(tg_e2m1[(u1 >> 16u) & 15u], tg_e2m1[(u1 >> 20u) & 15u], tg_e2m1[(u1 >> 24u) & 15u], tg_e2m1[u1 >> 28u]);")
                    for b in range(c_val):
                        ha(f"                acc{b}[r] += (dot(wv0, x{b}[0]) + dot(wv1, x{b}[1]) + dot(wv2, x{b}[2]) + dot(wv3, x{b}[3])) * bsc;")
                    ha("            }")
                    ha("        }")
                    ha(f"        #pragma unroll\n        for (uint r = 0; r < {inner_rps}u; r++) {{")
                    for b in range(c_val):
                        ha(f"            float sv{b} = acc{b}[r];")
                        st = lpr // 2
                        while st >= 1:
                            ha(f"            sv{b} += simd_shuffle_xor(sv{b}, {st}u);")
                            st //= 2
                        ha(f"            if (hl == 0u && rb0 + r < m_rows) y[(ulong)s{b} * y_stride + rb0 + r] = sv{b} * eouter;")
                    ha("        }")
                    ha("    }")
                    ha("}")
                    continue
                ha(
                    f"inline void umoe_pp{fid}_c{c_val}("
                    "const device half* x, const device uint* ecodes, const device half* escales, "
                    "const threadgroup half4* tg_ht1, const threadgroup half4* tg_ht2, const threadgroup half2* tg_ht3, "
                    "const device half4* tv_rtr1, const device half2* tv_rtr2, const device half2* tv_rtr3, const device half* lut_rtr4, "
                    "device float* y, uint r0, uint m_rows, uint y_stride, uint x_div, uint packed, uint hl) {"
                )
                for b in range(c_val):
                    ha(f"    const uint s{b} = (packed >> {b * 8}u) & 0xFFu;")
                    ha(f"    const device half* xp{b} = x + (ulong)((x_div == 1u) ? s{b} : (s{b} >> 3)) * {n}u;")
                if n_passes > 1:
                    ha(f"    for (uint pp = 0; pp < {n_passes}u; pp++) {{ const uint rb0 = r0 + pp * {inner_rps}u;")
                else:
                    ha("    { const uint rb0 = r0;")
                for b in range(c_val):
                    ha(f"        float acc{b}[{inner_rps}]; for (uint r = 0; r < {inner_rps}u; r++) acc{b}[r] = 0.0f;")
                ha(f"        for (uint it = 0; it < {iters}u; it++) {{")
                ha(f"            const uint c = hl + {lpr}u * it, col0 = c * 16u;")
                for b in range(c_val):
                    ha(f"            half4 x{b}[4];")
                    for q in range(4):
                        ha(f"            x{b}[{q}] = *(const device half4*)(xp{b} + col0 + {q * 4}u);")
                k_of = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4}[fid]
                wpr_c = (n * k_of) // 32
                ha(f"            #pragma unroll\n            for (uint r = 0; r < {inner_rps}u; r++) {{")
                ha("                const uint row = min(rb0 + r, m_rows - 1u);")
                _emit_decode_ch16(
                    ha,
                    fid,
                    n,
                    ptr_expr=f"ecodes + (ulong)row * {wpr_c}u",
                    wpr_expr=f"{wpr_c}u",
                    mode="dot",
                    t_or_c=c_val,
                    x_expr_fn=lambda b, q: f"x{b}[{q}]",
                    acc_expr_fn=lambda b: f"acc{b}[r]",
                )
                ha("            }")
                ha("        }")
                ha(f"        #pragma unroll\n        for (uint r = 0; r < {inner_rps}u; r++) {{")
                ha("            const float sc = (float)escales[min(rb0 + r, m_rows - 1u)];")
                for b in range(c_val):
                    ha(f"            float sv{b} = acc{b}[r];")
                    st = lpr // 2
                    while st >= 1:
                        ha(f"            sv{b} += simd_shuffle_xor(sv{b}, {st}u);")
                        st //= 2
                    ha(f"            if (hl == 0u && rb0 + r < m_rows) y[(ulong)s{b} * y_stride + rb0 + r] = sv{b} * sc;")
                ha("        }")
                ha("    }")
                ha("}")

        L = []
        a = L.append
        a("const uint tg = threadgroup_position_in_grid.x;")
        a("const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;")
        a(f"const uint hw = lane >> {shift_lane}u, hl = lane & {mask_lane}u, tid = simd * 32u + lane;")
        a("const uint m_rows = meta[0], x_div = meta[1], ntg = meta[2];")
        if n_rg > 1:
            a(f"const uint rg_sz = m_rows / {n_rg}u;")
        if has_ht1:
            a(f"threadgroup half4 tg_ht1[512]; for (uint idx = tid; idx < 512u; idx += {sg * 32}u) tg_ht1[idx] = half4(lut_ht1[4u*idx], lut_ht1[4u*idx+1u], lut_ht1[4u*idx+2u], lut_ht1[4u*idx+3u]);")
        else:
            a("threadgroup half4 tg_ht1[1];")
        if has_ht2:
            a(f"threadgroup half4 tg_ht2[512]; for (uint idx = tid; idx < 512u; idx += {sg * 32}u) tg_ht2[idx] = half4(lut_ht2[4u*idx], lut_ht2[4u*idx+1u], lut_ht2[4u*idx+2u], lut_ht2[4u*idx+3u]);")
        else:
            a("threadgroup half4 tg_ht2[1];")
        if has_ht3:
            a(f"threadgroup half2 tg_ht3[1024]; for (uint idx = tid; idx < 1024u; idx += {sg * 32}u) tg_ht3[idx] = half2(lut_ht3[2u*idx], lut_ht3[2u*idx+1u]);")
        else:
            a("threadgroup half2 tg_ht3[1];")
        if has_nv:
            a(f"threadgroup half tg_e2m1[16]; threadgroup half tg_e4m3[256]; if (tid < 16u) tg_e2m1[tid] = lut_e2m1[tid]; for (uint idx = tid; idx < 256u; idx += {sg * 32}u) tg_e4m3[idx] = lut_e4m3[idx];")
        a("const device half4* tv_rtr1 = (const device half4*)lut_rtr1;")
        a("const device half2* tv_rtr2 = (const device half2*)lut_rtr2;")
        a("const device half2* tv_rtr3 = (const device half2*)lut_rtr3;")
        if has_ht1 or has_ht2 or has_ht3 or has_nv:
            a("threadgroup_barrier(mem_flags::mem_threadgroup);")
        a("const uint n_act = plan[0];")
        a("const uint blocks_per_u = (m_rows + RPS - 1u) / RPS;")
        a("const uint total_blocks = n_act * blocks_per_u;")
        a(f"const uint blocks_per_tg = SG * {groups_per_simd}u;")
        a("for (uint rb_i = tg; rb_i * blocks_per_tg < total_blocks; rb_i += ntg) {")
        a(f"    const uint gb = rb_i * blocks_per_tg + simd * {groups_per_simd}u + hw;")
        a("    if (gb >= total_blocks) continue;")
        a("    const uint u = gb / blocks_per_u, r0 = (gb - u * blocks_per_u) * RPS;")
        a("    const uint ec = plan[1u + u], eid = ec & 0xFFu, cnt = ec >> 8u, p0 = plan[65u + u];")
        if n_rg > 1:
            a(f"    const uint rg = r0 / rg_sz, r_loc = r0 - rg * rg_sz, idx_rg = eid * {n_rg}u + rg;")
            a("    const uint fid = exp_fid[idx_rg], m_grp = rg_sz;")
            a("    const device uint* ecodes = codes + exp_code_off[idx_rg];")
            a("    const device half* escales = scales + exp_scale_off[idx_rg];")
            a("    device float* y_ptr = y + rg * rg_sz;")
        else:
            a("    const uint r_loc = r0, m_grp = m_rows;")
            a("    const uint fid = exp_fid[eid];")
            a("    const device uint* ecodes = codes + exp_code_off[eid];")
            a("    const device half* escales = scales + exp_scale_off[eid];")
            a("    device float* y_ptr = y;")
        a("    switch (fid) {")
        for fid in fids:
            a(f"        case {fid}u:")
            if fid == 7:
                a("            { const device uchar* env_s = nv_scales + exp_nv_scale_off[eid]; const float eout = nv_outer[eid];")
                a("              if (cnt == 1u) umoe_pp7_c1(x_raw, ecodes, env_s, eout, tg_e2m1, tg_e4m3, y_ptr, r_loc, m_grp, m_rows, x_div, p0, hl);")
                a("              else if (cnt == 2u) umoe_pp7_c2(x_raw, ecodes, env_s, eout, tg_e2m1, tg_e4m3, y_ptr, r_loc, m_grp, m_rows, x_div, p0, hl);")
                a("              else if (cnt == 3u) umoe_pp7_c3(x_raw, ecodes, env_s, eout, tg_e2m1, tg_e4m3, y_ptr, r_loc, m_grp, m_rows, x_div, p0, hl);")
                a("              else umoe_pp7_c4(x_raw, ecodes, env_s, eout, tg_e2m1, tg_e4m3, y_ptr, r_loc, m_grp, m_rows, x_div, p0, hl); }")
            else:
                a(f"            if (cnt == 1u) umoe_pp{fid}_c1(x, ecodes, escales, tg_ht1, tg_ht2, tg_ht3, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, y_ptr, r_loc, m_grp, m_rows, x_div, p0, hl);")
                a(f"            else if (cnt == 2u) umoe_pp{fid}_c2(x, ecodes, escales, tg_ht1, tg_ht2, tg_ht3, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, y_ptr, r_loc, m_grp, m_rows, x_div, p0, hl);")
                a(f"            else if (cnt == 3u) umoe_pp{fid}_c3(x, ecodes, escales, tg_ht1, tg_ht2, tg_ht3, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, y_ptr, r_loc, m_grp, m_rows, x_div, p0, hl);")
                a(f"            else umoe_pp{fid}_c4(x, ecodes, escales, tg_ht1, tg_ht2, tg_ht3, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, y_ptr, r_loc, m_grp, m_rows, x_div, p0, hl);")
            a("            break;")
        a("    }")
        a("}")
        in_names = ["x", "codes", "scales", "exp_fid", "exp_code_off", "exp_scale_off", "lut_ht1", "lut_ht2", "lut_ht3", "lut_rtr1", "lut_rtr2", "lut_rtr3", "lut_rtr4", "plan", "meta"]
        if has_nv:
            in_names += ["x_raw", "nv_scales", "exp_nv_scale_off", "nv_outer", "lut_e2m1", "lut_e4m3"]
        tag = "_".join(str(f) for f in fids)
        return mx.fast.metal_kernel(
            name=f"ink_umoe_pp_n{n}_f{tag}_r{rps}_s{sg}_l{lpr}_rg{n_rg}",
            input_names=in_names,
            output_names=["y"],
            header="\n".join(H),
            source="\n".join(L),
        )

    if t_mode == "prefill":
        steps = n // 32
        H = [
            "#include <metal_simdgroup_matrix>\n",
            f"#define SG {sg}u\n#define RB {rb}u\n#define NB {nb}u\n",
            "#define ELEMS_H2 reinterpret_cast<thread half2&>\n",
            "#define ELEMS_F2 reinterpret_cast<thread float2&>\n",
        ]
        ha = H.append
        for fid in fids:
            k = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4}[fid]
            wpr_c = (n * k) // 32
            ha(
                f"inline void umoe_f{fid}_prefill_bal("
                "const device half* x, const device uint* ecodes, const device half* escales, "
                "const device half4* t_ht1, const device half4* t_ht2, const device half2* t_ht3, "
                "const device half2* tv_rtr1, const device half2* tv_rtr2, const device half2* tv_rtr3, const device half* lut_rtr4, "
                "const device uint* slot_order, device float* y, uint rbase, uint m_rows, uint x_div, uint base, uint cnt, uint lane) {"
            )
            ha("    const uint qid = lane / 4u, fm = (qid & 4u) + ((lane / 2u) % 4u), fn = (qid & 2u) * 2u + (lane % 2u) * 2u;")
            ha("    const uint q = fn / 2u, xcol = 8u * (fm / 2u) + (fm % 2u), r0 = rbase + fm;")
            ha("    bool va[NB], vb[NB]; uint sa[NB], sb[NB]; const device half* xa[NB]; const device half* xb[NB];")
            ha("    #pragma unroll")
            ha("    for (uint b = 0; b < NB; b++) {")
            ha("        va[b] = (8u * b + fn) < cnt; vb[b] = (8u * b + fn + 1u) < cnt;")
            ha("        sa[b] = va[b] ? slot_order[base + 8u * b + fn] : 0u;")
            ha("        sb[b] = vb[b] ? slot_order[base + 8u * b + fn + 1u] : 0u;")
            ha(f"        xa[b] = x + (ulong)((x_div == 1u) ? sa[b] : (sa[b] >> 3)) * {n}u;")
            ha(f"        xb[b] = x + (ulong)((x_div == 1u) ? sb[b] : (sb[b] >> 3)) * {n}u;")
            ha("    }")
            ha("    const device uint* p[RB];")
            ha(f"    for (uint i = 0; i < RB; i++) p[i] = ecodes + (ulong)min(r0 + 8u * i, m_rows - 1u) * {wpr_c}u;")
            ha(f"    const uint first = ({n}u - 8u - 8u * q) * {k}u, w_first = first >> 5, off = first & 31u;")
            ha("    simdgroup_float8x8 Acc[RB][NB];")
            ha("    for (uint i = 0; i < RB; i++) for (uint b = 0; b < NB; b++) Acc[i][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);")
            if fid == 3:
                ha("    uint w_prev[RB];")
                ha(f"    for (uint i = 0; i < RB; i++) w_prev[i] = (w_first + 1u < {wpr_c}u) ? p[i][w_first + 1u] : 0u;")
            ha(f"    for (uint s = 0; s < {steps}u; s++) {{")
            ha("        simdgroup_half8x8 Xm[NB][4];")
            ha("        const uint cbase = 32u * s + xcol;")
            ha("        #pragma unroll")
            ha("        for (uint b = 0; b < NB; b++) {")
            ha("            #pragma unroll")
            ha("            for (uint j = 0; j < 4u; j++) {")
            ha("                const uint col = cbase + 2u * j;")
            ha("                ELEMS_H2(Xm[b][j].thread_elements()) = half2(va[b] ? xa[b][col] : 0.0h, vb[b] ? xb[b][col] : 0.0h);")
            ha("            }")
            ha("        }")
            ha(f"        const uint wi = w_first - s * {k}u;")
            ha("        #pragma unroll")
            ha("        for (uint i = 0; i < RB; i++) {")
            ha("            const device uint* pw = p[i] + wi; simdgroup_half8x8 Am;")
            _emit_sg8_moe_mma(ha, fid, n, nb_tiles=nb)
            ha("        }")
            ha("    }")
            ha("    for (uint i = 0; i < RB; i++) {")
            ha("        const uint row = r0 + 8u * i; if (row >= m_rows) continue;")
            ha("        const float sc = (float)escales[row];")
            ha("        #pragma unroll")
            ha("        for (uint b = 0; b < NB; b++) {")
            ha("            const float2 d = ELEMS_F2(Acc[i][b].thread_elements()) * sc;")
            ha("            if (va[b]) y[(ulong)sa[b] * m_rows + row] = d.x;")
            ha("            if (vb[b]) y[(ulong)sb[b] * m_rows + row] = d.y;")
            ha("        }")
            ha("    }")
            ha("}")

        L = []
        a = L.append
        a("const uint tg = threadgroup_position_in_grid.x, simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup, tid = simd * 32u + lane;")
        a("const uint m_rows = meta[0], n_slots = meta[1], x_div = meta[2], ntg = meta[3];")
        a("const device half4* t_ht1 = (const device half4*)lut_ht1; const device half4* t_ht2 = (const device half4*)lut_ht2; const device half2* t_ht3 = (const device half2*)lut_ht3;")
        a("const device half2* tv_rtr1 = (const device half2*)lut_rtr1; const device half2* tv_rtr2 = (const device half2*)lut_rtr2; const device half2* tv_rtr3 = (const device half2*)lut_rtr3;")
        a("threadgroup uint tg_e_order[128], tg_e_prefix[129], tg_n_units, tg_ne_tot;")
        a("if (tid == 0u) {")
        a("    uint ne = 0u, tot_u = 0u;")
        for fid in fids:
            a(f"    for (uint e = 0; e < 128u; e++) {{ if (exp_fid[e] == {fid}u) {{ uint c = expert_ids[e+1u] - expert_ids[e]; if (c > 0u) {{ tg_e_order[ne] = e; tg_e_prefix[ne] = tot_u; tot_u += (c + (8u * NB) - 1u) / (8u * NB); ne++; }} }} }}")
        a("    tg_e_prefix[ne] = tot_u; tg_n_units = tot_u; tg_ne_tot = ne;")
        a("}")
        a("threadgroup_barrier(mem_flags::mem_threadgroup);")
        a("const uint n_units = tg_n_units, ne_tot = tg_ne_tot;")
        a("const uint blocks_per_u = (m_rows + (8u * RB) - 1u) / (8u * RB), total_blocks = n_units * blocks_per_u;")
        a("for (uint rb_i = tg; rb_i * SG < total_blocks; rb_i += ntg) {")
        a("    const uint gb = rb_i * SG + simd; if (gb >= total_blocks) continue;")
        a("    const uint u = gb / blocks_per_u, rbase = (gb - u * blocks_per_u) * (8u * RB);")
        a("    uint lo = 0u, hi = ne_tot;")
        a("    while (lo + 1u < hi) { uint mid = (lo + hi) >> 1; if (tg_e_prefix[mid] <= u) lo = mid; else hi = mid; }")
        a("    const uint eid = tg_e_order[lo], tile_i = u - tg_e_prefix[lo];")
        a("    const uint off0 = expert_ids[eid], off1 = expert_ids[eid + 1u];")
        a("    const uint base = off0 + tile_i * (8u * NB), cnt = min(8u * NB, off1 - base);")
        a("    const uint fid = exp_fid[eid];")
        a("    switch (fid) {")
        for fid in fids:
            a(f"        case {fid}u: umoe_f{fid}_prefill_bal(x, codes+exp_code_off[eid], scales+exp_scale_off[eid], t_ht1, t_ht2, t_ht3, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, slot_order, y, rbase, m_rows, x_div, base, cnt, lane); break;")
        a("    }")
        a("}")
        in_names = ["x", "codes", "scales", "exp_fid", "exp_code_off", "exp_scale_off", "lut_ht1", "lut_ht2", "lut_ht3", "lut_rtr1", "lut_rtr2", "lut_rtr3", "lut_rtr4", "expert_ids", "meta", "slot_order"]
        tag = "_".join(str(f) for f in fids)
        return mx.fast.metal_kernel(
            name=f"ink_umoe_prefill_bal_n{n}_f{tag}_rb{rb}_nb{nb}_s{sg}",
            input_names=in_names,
            output_names=["y"],
            header="\n".join(H),
            source="\n".join(L),
        )

    # t1: ultra-fast half-warp c1 kernel (up to 8 active experts, sentinel-safe compaction)
    H = [f"#define SG {sg}u\n#define RPS {rps}u\n"]
    ha = H.append
    for fid in fids:
        if fid == 7:
            wpr_nv, spr_nv = n // 8, n // 16
            ha(
                "inline void umoe_f7_c1("
                "const device half* x_raw, const device uint* ecodes, const device uchar* env_scales, float eouter, "
                "const threadgroup half* tg_e2m1, const threadgroup half* tg_e4m3, "
                "device float* y, uint r0, uint m_rows, uint y_stride, uint x_div, uint s0, uint hl) {"
            )
            ha(f"    const device half* xp0 = x_raw + (ulong)((x_div == 1u) ? s0 : (s0 / x_div)) * {n}u;")
            ha("    float acc0[RPS]; for (uint r = 0; r < RPS; r++) acc0[r] = 0.0f;")
            ha(f"    for (uint it = 0; it < {iters}u; it++) {{")
            ha(f"        const uint c = hl + {lpr}u * it;")
            ha("        const uint col0 = c * 16u;")
            ha("        float4 x0[4];")
            for q in range(4):
                ha(f"        x0[{q}] = float4(*(const device half4*)(xp0 + col0 + {q * 4}u));")
            ha("        #pragma unroll")
            ha("        for (uint r = 0; r < RPS; r++) {")
            ha("            const uint row = min(r0 + r, m_rows - 1u);")
            ha(f"            const device uint* p = ecodes + (ulong)row * {wpr_nv}u + 2u * c;")
            ha("            const uint u0 = p[0], u1 = p[1];")
            ha(f"            const float bsc = (float)tg_e4m3[env_scales[(ulong)row * {spr_nv}u + c]];")
            ha("            const float4 wv0 = float4(tg_e2m1[u0 & 15u], tg_e2m1[(u0 >> 4u) & 15u], tg_e2m1[(u0 >> 8u) & 15u], tg_e2m1[(u0 >> 12u) & 15u]);")
            ha("            const float4 wv1 = float4(tg_e2m1[(u0 >> 16u) & 15u], tg_e2m1[(u0 >> 20u) & 15u], tg_e2m1[(u0 >> 24u) & 15u], tg_e2m1[u0 >> 28u]);")
            ha("            const float4 wv2 = float4(tg_e2m1[u1 & 15u], tg_e2m1[(u1 >> 4u) & 15u], tg_e2m1[(u1 >> 8u) & 15u], tg_e2m1[(u1 >> 12u) & 15u]);")
            ha("            const float4 wv3 = float4(tg_e2m1[(u1 >> 16u) & 15u], tg_e2m1[(u1 >> 20u) & 15u], tg_e2m1[(u1 >> 24u) & 15u], tg_e2m1[u1 >> 28u]);")
            ha("            acc0[r] += (dot(wv0, x0[0]) + dot(wv1, x0[1]) + dot(wv2, x0[2]) + dot(wv3, x0[3])) * bsc;")
            ha("        }")
            ha("    }")
            ha("    #pragma unroll")
            ha("    for (uint r = 0; r < RPS; r++) {")
            ha("        float sv0 = acc0[r];")
            st = lpr // 2
            while st >= 1:
                ha(f"        sv0 += simd_shuffle_xor(sv0, {st}u);")
                st //= 2
            ha("        if (hl == 0u && r0 + r < m_rows) y[(ulong)s0 * y_stride + r0 + r] = sv0 * eouter;")
            ha("    }")
            ha("}")
            continue
        ha(
            f"inline void umoe_f{fid}_c1("
            "const device half* x, const device uint* ecodes, const device half* escales, "
            "const threadgroup half4* tg_ht1, const threadgroup half4* tg_ht2, const threadgroup half2* tg_ht3, "
            "const device half4* tv_rtr1, const device half2* tv_rtr2, const device half2* tv_rtr3, const device half* lut_rtr4, "
            "device float* y, uint r0, uint m_rows, uint y_stride, uint x_div, uint s0, uint hl) {"
        )
        ha(f"    const device half* xp0 = x + (ulong)((x_div == 1u) ? s0 : (s0 / x_div)) * {n}u;")
        ha("    float acc0[RPS]; for (uint r = 0; r < RPS; r++) acc0[r] = 0.0f;")
        ha(f"    for (uint it = 0; it < {iters}u; it++) {{")
        ha(f"        const uint c = hl + {lpr}u * it;")
        ha("        const uint col0 = c * 16u;")
        ha("        half4 x0[4];")
        for q in range(4):
            ha(f"        x0[{q}] = *(const device half4*)(xp0 + col0 + {q * 4}u);")
        k_of = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4}[fid]
        wpr_c = (n * k_of) // 32
        ha("        #pragma unroll")
        ha("        for (uint r = 0; r < RPS; r++) {")
        ha("            const uint row = min(r0 + r, m_rows - 1u);")
        _emit_decode_ch16(
            ha,
            fid,
            n,
            ptr_expr=f"ecodes + (ulong)row * {wpr_c}u",
            wpr_expr=f"{wpr_c}u",
            mode="dot",
            t_or_c=1,
            x_expr_fn=lambda b, q: f"x0[{q}]",
            acc_expr_fn=lambda b: "acc0[r]",
        )
        ha("        }")
        ha("    }")
        ha("    #pragma unroll")
        ha("    for (uint r = 0; r < RPS; r++) {")
        ha("        const float sc = (float)escales[min(r0 + r, m_rows - 1u)];")
        ha("        float sv0 = acc0[r];")
        st = lpr // 2
        while st >= 1:
            ha(f"        sv0 += simd_shuffle_xor(sv0, {st}u);")
            st //= 2
        ha("        if (hl == 0u && r0 + r < m_rows) y[(ulong)s0 * y_stride + r0 + r] = sv0 * sc;")
        ha("    }")
        ha("}")

    L = []
    a = L.append
    a("const uint tg = threadgroup_position_in_grid.x;")
    a("const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;")
    a(f"const uint hw = lane >> {shift_lane}u, hl = lane & {mask_lane}u;")
    a("const uint tid = simd * 32u + lane;")
    a("const uint m_rows = meta[0], n_slots = meta[1], x_div = meta[2], ntg = meta[3];")
    if n_rg > 1:
        a(f"const uint rg_sz = m_rows / {n_rg}u;")
    if has_ht1:
        a(f"threadgroup half4 tg_ht1[512]; for (uint idx = tid; idx < 512u; idx += {sg * 32}u) tg_ht1[idx] = half4(lut_ht1[4u*idx], lut_ht1[4u*idx+1u], lut_ht1[4u*idx+2u], lut_ht1[4u*idx+3u]);")
    else:
        a("threadgroup half4 tg_ht1[1];")
    if has_ht2:
        a(f"threadgroup half4 tg_ht2[512]; for (uint idx = tid; idx < 512u; idx += {sg * 32}u) tg_ht2[idx] = half4(lut_ht2[4u*idx], lut_ht2[4u*idx+1u], lut_ht2[4u*idx+2u], lut_ht2[4u*idx+3u]);")
    else:
        a("threadgroup half4 tg_ht2[1];")
    if has_ht3:
        a(f"threadgroup half2 tg_ht3[1024]; for (uint idx = tid; idx < 1024u; idx += {sg * 32}u) tg_ht3[idx] = half2(lut_ht3[2u*idx], lut_ht3[2u*idx+1u]);")
    else:
        a("threadgroup half2 tg_ht3[1];")
    if has_nv:
        a(f"threadgroup half tg_e2m1[16]; threadgroup half tg_e4m3[256]; if (tid < 16u) tg_e2m1[tid] = lut_e2m1[tid]; for (uint idx = tid; idx < 256u; idx += {sg * 32}u) tg_e4m3[idx] = lut_e4m3[idx];")
    a("const device half4* tv_rtr1 = (const device half4*)lut_rtr1;")
    a("const device half2* tv_rtr2 = (const device half2*)lut_rtr2;")
    a("const device half2* tv_rtr3 = (const device half2*)lut_rtr3;")
    a("threadgroup uint tg_valid[8], tg_nv;")
    a("if (tid == 0u) { uint nv = 0u; for (uint s = 0u; s < n_slots; s++) { if (expert_ids[s] != 0xFFFFFFFFu) tg_valid[nv++] = s; } tg_nv = nv; }")
    a("threadgroup_barrier(mem_flags::mem_threadgroup);")
    a("const uint blocks_per_u = (m_rows + RPS - 1u) / RPS;")
    a("const uint total_blocks = tg_nv * blocks_per_u;")
    a(f"const uint blocks_per_tg = SG * {groups_per_simd}u;")
    a("for (uint rb = tg; rb * blocks_per_tg < total_blocks; rb += ntg) {")
    a(f"    const uint gb = rb * blocks_per_tg + simd * {groups_per_simd}u + hw;")
    a("    if (gb >= total_blocks) continue;")
    a("    const uint ui = gb / blocks_per_u;")
    a("    const uint u = tg_valid[ui];")
    a("    const uint r0 = (gb - ui * blocks_per_u) * RPS;")
    a("    const uint eid = expert_ids[u];")
    if n_rg > 1:
        a(f"    const uint rg = r0 / rg_sz, r_loc = r0 - rg * rg_sz, idx_rg = eid * {n_rg}u + rg;")
        a("    const uint fid = exp_fid[idx_rg], m_grp = rg_sz;")
        a("    const device uint* ecodes = codes + exp_code_off[idx_rg];")
        a("    const device half* escales = scales + exp_scale_off[idx_rg];")
        a("    device float* y_ptr = y + rg * rg_sz;")
    else:
        a("    const uint r_loc = r0, m_grp = m_rows;")
        a("    const uint fid = exp_fid[eid];")
        a("    const device uint* ecodes = codes + exp_code_off[eid];")
        a("    const device half* escales = scales + exp_scale_off[eid];")
        a("    device float* y_ptr = y;")
    a("    switch (fid) {")
    for fid in fids:
        if fid == 7:
            a("        case 7u: umoe_f7_c1(x_raw, ecodes, nv_scales + exp_nv_scale_off[eid], nv_outer[eid], tg_e2m1, tg_e4m3, y_ptr, r_loc, m_grp, m_rows, x_div, u, hl); break;")
        else:
            a(f"        case {fid}u: umoe_f{fid}_c1(x, ecodes, escales, tg_ht1, tg_ht2, tg_ht3, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, y_ptr, r_loc, m_grp, m_rows, x_div, u, hl); break;")
    a("    }")
    a("}")

    in_names = ["x", "codes", "scales", "exp_fid", "exp_code_off", "exp_scale_off", "lut_ht1", "lut_ht2", "lut_ht3", "lut_rtr1", "lut_rtr2", "lut_rtr3", "lut_rtr4", "expert_ids", "meta"]
    if has_nv:
        in_names += ["x_raw", "nv_scales", "exp_nv_scale_off", "nv_outer", "lut_e2m1", "lut_e4m3"]
    tag = "_".join(str(f) for f in fids)
    return mx.fast.metal_kernel(
        name=f"ink_umoe_t1_n{n}_f{tag}_r{rps}_s{sg}_rg{n_rg}",
        input_names=in_names,
        output_names=["y"],
        header="\n".join(H),
        source="\n".join(L),
    )


@lru_cache(maxsize=1)
def _has_mpp() -> bool:
    """Detect Apple M5+ MetalPerformancePrimitives (`mpp::tensor_ops::matmul2d`) Neural Accelerator support once."""
    if os.environ.get("LAMQ_FORCE_MPP") == "0":
        return False
    info = mx.device_info()
    arch = str(info.get("architecture", "")).lower()
    if any(arch.startswith(p) for p in ("applegpu_g13", "applegpu_g14", "applegpu_g15", "applegpu_g16")):
        return False
    try:
        k = mx.fast.metal_kernel(
            name="ink_mpp_probe",
            input_names=["a"],
            output_names=["b"],
            header="#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;",
            source="if (thread_position_in_grid.x == 0u) b[0] = a[0];",
        )
        out = k(
            inputs=[mx.zeros((1,), dtype=mx.float16)],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[(1,)],
            output_dtypes=[mx.float16],
        )[0]
        mx.eval(out)
        return True
    except Exception:
        return False


def _emit_decode_ch16_tg(a, fid: int, n: int, ptr_expr: str, out_expr: str):
    """Emit ch=16 bit-unpack directly into threadgroup `half[16]` (unscaled — row scale applied in epilogue)."""
    if fid in (0, 1):
        k = 1 if fid == 0 else 2
        wpr_c = (n * k) // 32
        a(f"    {{ const uint B = ({n}u - 16u * (c + 1u)) * {k}u, w0 = B >> 5;")
        if k == 1:
            a("      const uint o = B & 31u;")
        a(f"      const device uint* p = {ptr_expr} + w0;")
        a(f"      uint w0v = p[0], w1v = (w0 + 1u < {wpr_c}u) ? p[1] : 0u;")
        if k == 1:
            a("      if (o != 0u) { w0v = (w0v >> 16u) | (w1v << 16u); w1v >>= 16u; }")
        for q in range(4):
            bit = (15 - (q * 4 + 3)) * k
            wi, off = bit >> 5, bit & 31
            ex = f"extract_bits(w{wi}v, {off}u, 12u)" if off + 12 <= 32 else f"insert_bits(extract_bits(w{wi}v, {off}u, {32-off}u), w{wi+1}v, {32-off}u, {12-(32-off)}u)"
            a(f"      *(threadgroup half4*)({out_expr} + {q * 4}u) = tg_ht4[(({ex}) * {HASH_MULT_HEX}) >> 23u];")
        a("    }")
    elif fid == 2:
        wpr_c = (n * 3) // 32
        a(f"    {{ const uint B = ({n}u - 16u * (c + 1u)) * 3u, w0 = B >> 5, o = B & 31u;")
        a(f"      const device uint* p = {ptr_expr} + w0;")
        a(f"      uint w0v = p[0], w1v = (w0 + 1u < {wpr_c}u) ? p[1] : 0u, w2v = (w0 + 2u < {wpr_c}u) ? p[2] : 0u;")
        a("      if (o != 0u) { w0v = (w0v >> 16u) | (w1v << 16u); w1v = (w1v >> 16u) | (w2v << 16u); w2v >>= 16u; }")
        for i in range(0, 16, 2):
            bit = (15 - (i + 1)) * 3
            wi, off = bit >> 5, bit & 31
            ex = f"extract_bits(w{wi}v, {off}u, 12u)" if off + 12 <= 32 else f"insert_bits(extract_bits(w{wi}v, {off}u, {32-off}u), w{wi+1}v, {32-off}u, {12-(32-off)}u)"
            a(f"      *(threadgroup half2*)({out_expr} + {i}u) = tg_ht2[(({ex}) * {HASH_MULT_HEX}) >> 22u];")
        a("    }")
    elif fid == 3:
        wpr_c = n // 32
        a(f"    {{ const uint B = ({n}u - 16u * (c + 1u)), w0 = B >> 5, o = B & 31u;")
        a(f"      const device uint* p = {ptr_expr} + w0;")
        a(f"      uint w0v = p[0], w1v = (w0 + 1u < {wpr_c}u) ? p[1] : 0u;")
        a("      if (o != 0u) { w0v = (w0v >> 16u) | (w1v << 16u); w1v >>= 16u; }")
        for ci, (i, gg) in enumerate([(0, 3), (3, 3), (6, 3), (9, 3), (12, 3), (15, 1)]):
            bit = 15 - (i + gg - 1)
            nb = 12 + (gg - 1)
            wi, off = bit >> 5, bit & 31
            ex = f"extract_bits(w{wi}v, {off}u, {nb}u)" if off + nb <= 32 else f"insert_bits(extract_bits(w{wi}v, {off}u, {32-off}u), w{wi+1}v, {32-off}u, {nb-(32-off)}u)"
            a(f"      const half4 v{ci} = tv_rtr1[{ex}];")
        a(f"      *(threadgroup half4*)({out_expr} + 0u)  = half4(v0[0], v0[1], v0[2], v1[0]);")
        a(f"      *(threadgroup half4*)({out_expr} + 4u)  = half4(v1[1], v1[2], v2[0], v2[1]);")
        a(f"      *(threadgroup half4*)({out_expr} + 8u)  = half4(v2[2], v3[0], v3[1], v3[2]);")
        a(f"      *(threadgroup half4*)({out_expr} + 12u) = half4(v4[0], v4[1], v4[2], v5[2]);")
        a("    }")
    elif fid in (4, 5):
        k = 2 if fid == 4 else 3
        nb = 12 + k
        wpr_c = (n * k) // 32
        tvn = "tv_rtr2" if fid == 4 else "tv_rtr3"
        a(f"    {{ const uint B = ({n}u - 16u * (c + 1u)) * {k}u, w0 = B >> 5;")
        if k == 3:
            a("      const uint o = B & 31u;")
        a(f"      const device uint* p = {ptr_expr} + w0;")
        if k == 2:
            a(f"      const uint w0v = p[0], w1v = (w0 + 1u < {wpr_c}u) ? p[1] : 0u;")
        else:
            a(f"      uint w0v = p[0], w1v = (w0 + 1u < {wpr_c}u) ? p[1] : 0u, w2v = (w0 + 2u < {wpr_c}u) ? p[2] : 0u;")
            a("      if (o != 0u) { w0v = (w0v >> 16u) | (w1v << 16u); w1v = (w1v >> 16u) | (w2v << 16u); w2v >>= 16u; }")
        for i in range(0, 16, 2):
            bit = (15 - (i + 1)) * k
            wi, off = bit >> 5, bit & 31
            ex = f"extract_bits(w{wi}v, {off}u, {nb}u)" if off + nb <= 32 else f"insert_bits(extract_bits(w{wi}v, {off}u, {32-off}u), w{wi+1}v, {32-off}u, {nb-(32-off)}u)"
            a(f"      *(threadgroup half2*)({out_expr} + {i}u) = {tvn}[{ex}];")
        a("    }")
    elif fid == 6:
        wpr_c = (n * 4) // 32
        a(f"    {{ const uint B = ({n}u - 16u * (c + 1u)) * 4u, w0 = B >> 5;")
        a(f"      const device uint* p = {ptr_expr} + w0;")
        a(f"      const uint w0v = p[0], w1v = p[1], w2v = (w0 + 2u < {wpr_c}u) ? p[2] : 0u;")
        for q in range(4):
            vs = []
            for u in range(4):
                i = q * 4 + u
                bit = (15 - i) * 4
                wi, off = bit >> 5, bit & 31
                ex = f"extract_bits(w{wi}v, {off}u, 12u)" if off + 12 <= 32 else f"insert_bits(extract_bits(w{wi}v, {off}u, {32-off}u), w{wi+1}v, {32-off}u, {12-(32-off)}u)"
                vs.append(f"lut_rtr4[{ex}]")
            a(f"      *(threadgroup half4*)({out_expr} + {q * 4}u) = half4({', '.join(vs)});")
        a("    }")
    else:
        raise ValueError(f"Unsupported MoE fid={fid}")


@lru_cache(maxsize=32)
def _make_moe_prefill_unit_planner(fids: tuple[int, ...], tile_t: int = 32):
    """Single-launch GPU planner producing format-grouped `(eid, base, cnt)` prefill work units."""
    L = [
        "const uint tid = thread_position_in_threadgroup.x;",
        "if (tid != 0u) return;",
        "uint tot = 0u;",
    ]
    for fid in fids:
        L.append(
            f"for (uint e = 0u; e < 128u; e++) {{ "
            f"if (exp_fid[e] == {fid}u) {{ "
            f"uint off0 = offsets[e], off1 = offsets[e + 1u]; "
            f"for (uint b = off0; b < off1; b += {tile_t}u) {{ "
            f"uint cnt = min({tile_t}u, off1 - b); "
            f"units[tot * 3u + 0u] = e; units[tot * 3u + 1u] = b; units[tot * 3u + 2u] = cnt; tot++; "
            f"}} }} }}"
        )
    L.append("unit_count[0] = tot;")
    tag = "_".join(str(f) for f in fids)
    return mx.fast.metal_kernel(
        name=f"ink_moe_plan_prefill_f{tag}_t{tile_t}",
        input_names=["offsets", "exp_fid"],
        output_names=["units", "unit_count"],
        source="\n".join(L),
    )


def plan_moe_prefill_units(
    offsets: mx.array,
    exp_fid: mx.array,
    fids: tuple[int, ...],
    n_slots: int,
    tile_t: int = 32,
) -> tuple[mx.array, mx.array, int]:
    """Build format-grouped prefill work-unit list `(units[max_u*3], unit_count[1], max_u)` on GPU in 1 launch."""
    max_u = max(1, min(n_slots, 128 + n_slots // tile_t))
    units, ucnt = _make_moe_prefill_unit_planner(fids, tile_t)(
        inputs=[offsets.astype(mx.uint32), exp_fid.astype(mx.uint32)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(max_u * 3,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    return units, ucnt, max_u


@lru_cache(maxsize=64)
def _make_unified_moe_prefill_mpp_kernel(
    n: int,
    fids: tuple[int, ...],
    sorted_in: bool = False,
    sorted_out: bool = False,
    m_tile: int = 64,
    tile_t: int = 32,
    k_tile: int = 32,
    sg: int = 4,
    n_rg: int = 1,
):
    """M5 Neural Accelerator (`mpp::tensor_ops::matmul2d`) format-homogeneous prefill MoE kernel (10 KB SRAM, 1.50x faster)."""
    assert n % k_tile == 0 and (m_tile * k_tile) % (16 * sg * 32) == 0
    steps = n // k_tile
    n_threads = sg * 32
    ch16_per_thr = (m_tile * k_tile) // (16 * n_threads)
    has_ht = any(f in fids for f in (0, 1, 2))
    has_nv = 7 in fids
    adaptive_n = (16, 32) if tile_t == 32 else (16, 32, 64)

    H = [
        "#include <metal_tensor>",
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>",
        "using namespace mpp::tensor_ops;",
        f"#define SG {sg}u",
        f"#define MTILE {m_tile}u",
        f"#define KTILE {k_tile}u",
        f"#define TILE_T {tile_t}u",
    ]
    ha = H.append
    for fid in fids:
        if fid == 7:
            wpr_nv, spr_nv = n // 8, n // 16
            ha(
                "inline void dec_mpp_f7("
                "const device uint* ecodes, const device uchar* env_scales, threadgroup half* tg_w, "
                "const threadgroup half* tg_e2m1, const threadgroup half* tg_e4m3, "
                "uint rbase, uint m_rows, uint s_step, uint tid) {"
            )
            ha(f"    constexpr uint C_PER_ROW = {k_tile // 16}u;")
            for ci in range(ch16_per_thr):
                ha(f"    {{ const uint g_ch = tid + {ci * n_threads}u;")
                ha("      const uint r_loc = g_ch / C_PER_ROW, c_loc = g_ch - r_loc * C_PER_ROW;")
                ha("      const uint row = min(rbase + r_loc, m_rows - 1u);")
                ha("      const uint c = s_step * C_PER_ROW + c_loc;")
                ha(f"      const device uint* p = ecodes + (ulong)row * {wpr_nv}u + 2u * c;")
                ha("      const uint u0 = p[0], u1 = p[1];")
                ha(f"      const half bsc = tg_e4m3[env_scales[(ulong)row * {spr_nv}u + c]];")
                ha("      threadgroup half* dst = tg_w + r_loc * KTILE + c_loc * 16u;")
                ha("      *(threadgroup half4*)(dst + 0u)  = half4(tg_e2m1[u0 & 15u], tg_e2m1[(u0 >> 4u) & 15u], tg_e2m1[(u0 >> 8u) & 15u], tg_e2m1[(u0 >> 12u) & 15u]) * bsc;")
                ha("      *(threadgroup half4*)(dst + 4u)  = half4(tg_e2m1[(u0 >> 16u) & 15u], tg_e2m1[(u0 >> 20u) & 15u], tg_e2m1[(u0 >> 24u) & 15u], tg_e2m1[u0 >> 28u]) * bsc;")
                ha("      *(threadgroup half4*)(dst + 8u)  = half4(tg_e2m1[u1 & 15u], tg_e2m1[(u1 >> 4u) & 15u], tg_e2m1[(u1 >> 8u) & 15u], tg_e2m1[(u1 >> 12u) & 15u]) * bsc;")
                ha("      *(threadgroup half4*)(dst + 12u) = half4(tg_e2m1[(u1 >> 16u) & 15u], tg_e2m1[(u1 >> 20u) & 15u], tg_e2m1[(u1 >> 24u) & 15u], tg_e2m1[u1 >> 28u]) * bsc;")
                ha("    }")
            ha("}")
            continue
        k_bits = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4}[fid]
        wpr_c = (n * k_bits) // 32
        ha(
            f"inline void dec_mpp_f{fid}("
            "const device uint* ecodes, threadgroup half* tg_w, "
            "const threadgroup half4* tg_ht4, const threadgroup half2* tg_ht2, "
            "const device half4* tv_rtr1, const device half2* tv_rtr2, const device half2* tv_rtr3, const device half* lut_rtr4, "
            "uint rbase, uint m_rows, uint s_step, uint tid) {"
        )
        ha(f"    constexpr uint C_PER_ROW = {k_tile // 16}u;")
        for ci in range(ch16_per_thr):
            ha(f"    {{ const uint g_ch = tid + {ci * n_threads}u;")
            ha("      const uint r_loc = g_ch / C_PER_ROW, c_loc = g_ch - r_loc * C_PER_ROW;")
            ha("      const uint row = min(rbase + r_loc, m_rows - 1u);")
            ha("      const uint c = s_step * C_PER_ROW + c_loc;")
            _emit_decode_ch16_tg(ha, fid, n, ptr_expr=f"ecodes + (ulong)row * {wpr_c}u", out_expr="tg_w + r_loc * KTILE + c_loc * 16u")
            ha("    }")
        ha("}")

    nv_sig = "const device uchar* env_scales, float eouter, const threadgroup half* tg_e2m1, const threadgroup half* tg_e4m3, " if has_nv else ""
    for n_sub in adaptive_n:
        x_h4_total = (n_sub * k_tile) // 4
        x_h4_per_thr = (x_h4_total + n_threads - 1) // n_threads
        ha(
            f"inline void umoe_mpp_n{n_sub}("
            f"const device half* x, const device uint* ecodes, const device half* escales, {nv_sig}"
            "threadgroup half* tg_w, threadgroup half* tg_x, "
            "const threadgroup half4* tg_ht4, const threadgroup half2* tg_ht2, "
            "const device half4* tv_rtr1, const device half2* tv_rtr2, const device half2* tv_rtr3, const device half* lut_rtr4, "
            "const threadgroup uint* tg_xrow, const threadgroup uint* tg_out_slot, device float* y, "
            "uint fid, uint rbase, uint m_rows, uint y_stride, uint cnt, uint tid) {"
        )
        ha(f"    constexpr auto desc = matmul2d_descriptor({m_tile}, {n_sub}, {k_tile}, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);")
        ha(f"    matmul2d<desc, execution_simdgroups<{sg}>> op;")
        ha(f"    auto tW = metal::tensor<threadgroup half, metal::extents<int, {k_tile}, {m_tile}>, metal::tensor_inline>(tg_w, metal::extents<int, {k_tile}, {m_tile}>());")
        ha(f"    auto tX = metal::tensor<threadgroup half, metal::extents<int, {k_tile}, {n_sub}>, metal::tensor_inline>(tg_x, metal::extents<int, {k_tile}, {n_sub}>());")
        ha("    auto cT = op.get_destination_cooperative_tensor<decltype(tW), decltype(tX), float>();")
        ha("    #pragma unroll")
        ha("    for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0.0f; }")
        ha(f"    for (uint s = 0; s < {steps}u; s++) {{")
        ha("        switch (fid) {")
        for fid in fids:
            if fid == 7:
                ha("            case 7u: dec_mpp_f7(ecodes, env_scales, tg_w, tg_e2m1, tg_e4m3, rbase, m_rows, s, tid); break;")
            else:
                ha(f"            case {fid}u: dec_mpp_f{fid}(ecodes, tg_w, tg_ht4, tg_ht2, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, rbase, m_rows, s, tid); break;")
        ha("        }")
        ha(f"        constexpr uint H4_PER_ROW = {k_tile // 4}u;")
        for xi in range(x_h4_per_thr):
            cond = f"if (tid + {xi * n_threads}u < {x_h4_total}u) " if (xi + 1) * n_threads > x_h4_total else ""
            ha(f"        {cond}{{ const uint idx = tid + {xi * n_threads}u, t_loc = idx / H4_PER_ROW, k4 = idx - t_loc * H4_PER_ROW;")
            ha(f"          *(threadgroup half4*)(tg_x + t_loc * KTILE + k4 * 4u) = (t_loc < cnt) ? *(const device half4*)(x + (ulong)tg_xrow[t_loc] * {n}u + s * KTILE + k4 * 4u) : half4(0.0h); }}")
        ha("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        ha("        op.run(tW, tX, cT);")
        ha("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        ha("    }")
        ha("    #pragma unroll")
        ha("    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {")
        ha("        if (cT.is_valid_element(i)) {")
        ha("            auto idx = cT.get_multidimensional_index(i);")
        ha("            const uint t_loc = (uint)idx[0], row = rbase + (uint)idx[1];")
        ha("            if (t_loc < cnt && row < m_rows) {")
        if has_nv:
            ha("                const float sc = (fid == 7u) ? eouter : (float)escales[row];")
            ha("                y[(ulong)tg_out_slot[t_loc] * y_stride + row] = cT[i] * sc;")
        else:
            ha("                y[(ulong)tg_out_slot[t_loc] * y_stride + row] = cT[i] * (float)escales[row];")
        ha("            }")
        ha("        }")
        ha("    }")
        ha("}")

    L = [
        "const uint u = threadgroup_position_in_grid.x;",
        "const uint rb_tg = threadgroup_position_in_grid.y;",
        "if (u >= unit_count[0]) return;",
        "const uint tid = thread_position_in_threadgroup.x;",
        "const uint m_rows = meta[0], x_div = meta[1];",
        "const uint eid = units[u * 3u + 0u], base = units[u * 3u + 1u], cnt = units[u * 3u + 2u];",
    ]
    if n_rg > 1:
        L += [
            f"const uint rg_sz = m_rows / {n_rg}u, r_glob = rb_tg * MTILE, rg = r_glob / rg_sz;",
            f"const uint rbase = r_glob - rg * rg_sz, m_grp = rg_sz, idx_rg = eid * {n_rg}u + rg;",
            "const uint fid = exp_fid[idx_rg];",
            "device float* y_ptr = y + rg * rg_sz;",
        ]
    else:
        L += [
            "const uint fid = exp_fid[eid], rbase = rb_tg * MTILE, m_grp = m_rows;",
            "device float* y_ptr = y;",
        ]
    L += [
        "const device half4* t_ht1 = (const device half4*)lut_ht1; const device half4* t_ht2 = (const device half4*)lut_ht2;",
        "const device half4* tv_rtr1 = (const device half4*)lut_rtr1; const device half2* tv_rtr2 = (const device half2*)lut_rtr2; const device half2* tv_rtr3 = (const device half2*)lut_rtr3;",
        "threadgroup half tg_w[MTILE * KTILE];",
        "threadgroup half tg_x[TILE_T * KTILE];",
        "threadgroup half4 tg_ht4[512]; const threadgroup half2* tg_ht2 = (const threadgroup half2*)tg_ht4;" if has_ht else "threadgroup half4 tg_ht4[1]; const threadgroup half2* tg_ht2 = (const threadgroup half2*)tg_ht4;",
    ]
    if has_ht:
        L += [
            f"if (fid == 0u) {{ for (uint idx = tid; idx < 512u; idx += {n_threads}u) tg_ht4[idx] = t_ht1[idx]; }}",
            f"else if (fid == 1u) {{ for (uint idx = tid; idx < 512u; idx += {n_threads}u) tg_ht4[idx] = t_ht2[idx]; }}",
            f"else if (fid == 2u) {{ for (uint idx = tid; idx < 512u; idx += {n_threads}u) tg_ht4[idx] = ((const device half4*)lut_ht3)[idx]; }}",
        ]
    if has_nv:
        L += [
            "threadgroup half tg_e2m1[16]; threadgroup half tg_e4m3[256];",
            f"if (fid == 7u) {{ if (tid < 16u) tg_e2m1[tid] = lut_e2m1[tid]; for (uint idx = tid; idx < 256u; idx += {n_threads}u) tg_e4m3[idx] = lut_e4m3[idx]; }}",
        ]
    L += [
        "threadgroup uint tg_xrow[TILE_T], tg_out_slot[TILE_T];",
        "if (tid < cnt) {",
        "    tg_xrow[tid] = base + tid;" if sorted_in else "    const uint os = slot_order[base + tid]; tg_xrow[tid] = (x_div == 1u) ? os : (os >> 3);",
        "    tg_out_slot[tid] = base + tid;" if sorted_out else "    tg_out_slot[tid] = slot_order[base + tid];",
        "}",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        "const device uint* ec = codes + exp_code_off[" + ("idx_rg" if n_rg > 1 else "eid") + "];",
        "const device half* es = scales + exp_scale_off[" + ("idx_rg" if n_rg > 1 else "eid") + "];",
    ]
    if has_nv:
        L.append("const device half* xin = (fid == 7u) ? x_raw : x;")
        L.append("const device uchar* env_s = nv_scales + exp_nv_scale_off[eid]; const float eout = nv_outer[eid];")
        nv_call = "env_s, eout, tg_e2m1, tg_e4m3, "
        x_arg = "xin"
    else:
        nv_call = ""
        x_arg = "x"
    for idx_n, n_sub in enumerate(adaptive_n):
        cond = f"if (cnt <= {n_sub}u)" if idx_n == 0 else (f"else if (cnt <= {n_sub}u)" if idx_n < len(adaptive_n) - 1 else "else")
        L.append(f"{cond} umoe_mpp_n{n_sub}({x_arg}, ec, es, {nv_call}tg_w, tg_x, tg_ht4, tg_ht2, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, tg_xrow, tg_out_slot, y_ptr, fid, rbase, m_grp, m_rows, cnt, tid);")
    in_names = ["x", "codes", "scales", "exp_fid", "exp_code_off", "exp_scale_off", "lut_ht1", "lut_ht2", "lut_ht3", "lut_rtr1", "lut_rtr2", "lut_rtr3", "lut_rtr4", "units", "unit_count", "meta", "slot_order"]
    if has_nv:
        in_names += ["x_raw", "nv_scales", "exp_nv_scale_off", "nv_outer", "lut_e2m1", "lut_e4m3"]
    tag = "_".join(str(f) for f in fids)
    return mx.fast.metal_kernel(
        name=f"ink_umoe_prefill_mpp_n{n}_f{tag}_si{int(sorted_in)}_so{int(sorted_out)}_m{m_tile}_t{tile_t}_rg{n_rg}",
        input_names=in_names,
        output_names=["y"],
        header="\n".join(H),
        source="\n".join(L),
    )


@lru_cache(maxsize=64)
def _make_unified_moe_prefill_v2_kernel(
    n: int,
    fids: tuple[int, ...],
    sorted_in: bool = False,
    sorted_out: bool = False,
    sg: int = 8,
    rb: int = 2,
    n_rg: int = 1,
):
    """Non-MPP fallback format-homogeneous prefill MoE kernel (`SG=8, RB=2, tile_t=32`, TG SRAM X_tile + LUT)."""
    steps = n // 32
    has_ht = any(f in fids for f in (0, 1, 2))
    has_nv = 7 in fids
    H = [
        "#include <metal_simdgroup_matrix>\n",
        f"#define SG {sg}u\n#define RB {rb}u\n#define XSTR 36u\n",
        "#define ELEMS_H2 reinterpret_cast<thread half2&>\n",
        "#define ELEMS_F2 reinterpret_cast<thread float2&>\n",
    ]
    ha = H.append

    def _emit_mma_tg(fid: int, nb_val: int):
        k = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4}[fid]
        wpr_c = (n * k) // 32

        def mma_line(j: int):
            if nb_val == 1:
                return f"simdgroup_multiply_accumulate(Acc[i][0], Am, Xm[0][{j}], Acc[i][0]);"
            return f"\n            #pragma unroll\n            for (uint b = 0; b < {nb_val}u; b++) simdgroup_multiply_accumulate(Acc[i][b], Am, Xm[b][{j}], Acc[i][b]);"

        if fid in (0, 1):
            if fid == 1:
                ha(f"            const uint w0 = pw[0]; const uint lo = (off == 0u) ? w0 : ((w0 >> 16u) | (((wi + 1u < {wpr_c}u) ? pw[1] : 0u) << 16u));")
            else:
                ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u; const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
            ha(f"            const half4 v0 = tg_ht4[((extract_bits(lo, {4 * k}u, 12u)) * {HASH_MULT_HEX}) >> 23u];")
            ha(f"            const half4 v1 = tg_ht4[((extract_bits(lo, 0u, 12u)) * {HASH_MULT_HEX}) >> 23u];")
            for j, expr in [(0, "v0.xy"), (1, "v0.zw"), (2, "v1.xy"), (3, "v1.zw")]:
                ha(f"            ELEMS_H2(Am.thread_elements()) = {expr}; {mma_line(j)}")
        elif fid == 2:
            ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u;")
            ha("            const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
            for j, rb_b in [(0, 18), (1, 12), (2, 6), (3, 0)]:
                ha(f"            ELEMS_H2(Am.thread_elements()) = tg_ht2[(extract_bits(lo, {rb_b}u, 12u) * {HASH_MULT_HEX}) >> 22u]; {mma_line(j)}")
        elif fid == 3:
            ha("            const uint w0 = pw[0]; const uint lo = (off == 0u) ? w0 : ((w0 >> off) | (w_prev[i] << (32u - off))); w_prev[i] = w0;")
            for j, sh in [(0, 6), (1, 4), (2, 2), (3, 0)]:
                ha(f"            ELEMS_H2(Am.thread_elements()) = tv_rtr1[extract_bits(lo, {sh}u, 13u)]; {mma_line(j)}")
        elif fid == 4:
            ha(f"            const uint w0 = pw[0]; const uint lo = (off == 0u) ? w0 : ((w0 >> 16u) | (((wi + 1u < {wpr_c}u) ? pw[1] : 0u) << 16u));")
            for j, rb_b in [(0, 12), (1, 8), (2, 4), (3, 0)]:
                ha(f"            ELEMS_H2(Am.thread_elements()) = tv_rtr2[extract_bits(lo, {rb_b}u, 14u)]; {mma_line(j)}")
        elif fid == 5:
            ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u, w2 = (wi + 2u < {wpr_c}u) ? pw[2] : 0u;")
            ha("            const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
            ha("            const uint hi = (w1 >> off) | (off == 0u ? 0u : (w2 << (32u - off)));")
            for j, rb_b in [(0, 18), (1, 12), (2, 6), (3, 0)]:
                ex = f"extract_bits(lo, {rb_b}u, 15u)" if rb_b + 15 <= 32 else f"insert_bits(extract_bits(lo, {rb_b}u, {32 - rb_b}u), hi, {32 - rb_b}u, {15 - (32 - rb_b)}u)"
                ha(f"            ELEMS_H2(Am.thread_elements()) = tv_rtr3[{ex}]; {mma_line(j)}")
        elif fid == 6:
            ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u;")
            for j, (b0, b1) in [(0, (28, 24)), (1, (20, 16)), (2, (12, 8)), (3, (4, 0))]:
                ex0 = f"extract_bits(w0, {b0}u, 12u)" if b0 + 12 <= 32 else f"insert_bits(extract_bits(w0, {b0}u, {32 - b0}u), w1, {32 - b0}u, {12 - (32 - b0)}u)"
                ex1 = f"extract_bits(w0, {b1}u, 12u)" if b1 + 12 <= 32 else f"insert_bits(extract_bits(w0, {b1}u, {32 - b1}u), w1, {32 - b1}u, {12 - (32 - b1)}u)"
                ha(f"            ELEMS_H2(Am.thread_elements()) = half2(lut_rtr4[{ex0}], lut_rtr4[{ex1}]); {mma_line(j)}")

    for fid in fids:
        if fid == 7:
            wpr_nv, spr_nv = n // 8, n // 16
            for nb_val in (1, 2, 4):
                max_t = 8 * nb_val
                mma_fn = (
                    lambda j: f"simdgroup_multiply_accumulate(Acc[i][0], Am, Xm[0][{j}], Acc[i][0]);"
                    if nb_val == 1
                    else f"\n            #pragma unroll\n            for (uint b = 0; b < {nb_val}u; b++) simdgroup_multiply_accumulate(Acc[i][b], Am, Xm[b][{j}], Acc[i][b]);"
                )
                ha(
                    f"inline void umoe_v2_f7_nb{nb_val}("
                    "const device half* my_xp, const device uint* ecodes, const device uchar* env_scales, float eouter, "
                    "threadgroup half* tg_x, const threadgroup half* tg_e2m1, const threadgroup half* tg_e4m3, "
                    "const threadgroup uint* tg_out_slot, device float* y, "
                    "uint rbase, uint m_rows, uint y_stride, uint cnt, uint tid, uint lane) {"
                )
                ha("    const uint tok_i = tid >> 3, vec_k = tid & 7u, vk_q = vec_k >> 1, vk_h = vec_k & 1u;")
                ha("    const uint qid = lane / 4u, fm = (qid & 4u) + ((lane / 2u) % 4u), fn = (qid & 2u) * 2u + (lane % 2u) * 2u;")
                ha("    const uint q = fn / 2u, r0 = rbase + fm;")
                ha("    const device uint* p_c[RB]; const device uchar* p_s[RB];")
                ha(f"    for (uint i = 0; i < RB; i++) {{ const uint rr = min(r0 + 8u * i, m_rows - 1u); p_c[i] = ecodes + (ulong)rr * {wpr_nv}u; p_s[i] = env_scales + (ulong)rr * {spr_nv}u; }}")
                ha(f"    simdgroup_float8x8 Acc[RB][{nb_val}];")
                ha(f"    for (uint i = 0; i < RB; i++) for (uint b = 0; b < {nb_val}u; b++) Acc[i][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);")
                ha(f"    if (tok_i < {max_t}u) {{")
                ha("        const half4 xv = (tok_i < cnt) ? *(const device half4*)(my_xp) : half4(0.0h);")
                ha("        threadgroup half* rowp = tg_x + tok_i * XSTR;")
                ha("        *(threadgroup half2*)(rowp + (vk_h * 2u) * 8u + vk_q * 2u) = xv.xy;")
                ha("        *(threadgroup half2*)(rowp + (vk_h * 2u + 1u) * 8u + vk_q * 2u) = xv.zw;")
                ha("    }")
                ha("    threadgroup_barrier(mem_flags::mem_threadgroup);")
                ha(f"    for (uint s = 0; s < {steps}u; s++) {{")
                ha("        const uint cur_off = (s & 1u) * (32u * XSTR), nxt_off = ((s + 1u) & 1u) * (32u * XSTR);")
                ha(f"        if (s + 1u < {steps}u && tok_i < {max_t}u) {{")
                ha("            const half4 xv = (tok_i < cnt) ? *(const device half4*)(my_xp + 32u * (s + 1u)) : half4(0.0h);")
                ha("            threadgroup half* rowp = tg_x + nxt_off + tok_i * XSTR;")
                ha("            *(threadgroup half2*)(rowp + (vk_h * 2u) * 8u + vk_q * 2u) = xv.xy;")
                ha("            *(threadgroup half2*)(rowp + (vk_h * 2u + 1u) * 8u + vk_q * 2u) = xv.zw;")
                ha("        }")
                ha("        if (rbase < m_rows) {")
                ha(f"            simdgroup_half8x8 Xm[{nb_val}][4];")
                ha("            #pragma unroll")
                ha(f"            for (uint b = 0; b < {nb_val}u; b++) {{")
                ha("                #pragma unroll")
                ha("                for (uint j = 0; j < 4u; j++) simdgroup_load(Xm[b][j], tg_x + cur_off + (8u * b) * XSTR + 8u * j, XSTR, ulong2(0, 0), true);")
                ha("            }")
                ha("            #pragma unroll")
                ha("            for (uint i = 0; i < RB; i++) {")
                ha("                const uint w0 = p_c[i][4u * s + q];")
                ha("                const half bsc = tg_e4m3[p_s[i][2u * s + (q >> 1)]];")
                ha("                simdgroup_half8x8 Am;")
                for j in range(4):
                    ha(f"                ELEMS_H2(Am.thread_elements()) = half2(tg_e2m1[(w0 >> {8 * j}u) & 15u], tg_e2m1[(w0 >> {8 * j + 4}u) & 15u]) * bsc; {mma_fn(j)}")
                ha("            }")
                ha("        }")
                ha("        threadgroup_barrier(mem_flags::mem_threadgroup);")
                ha("    }")
                ha("    if (rbase < m_rows) {")
                ha("        for (uint i = 0; i < RB; i++) {")
                ha("            const uint row = r0 + 8u * i; if (row >= m_rows) continue;")
                ha("            #pragma unroll")
                ha(f"            for (uint b = 0; b < {nb_val}u; b++) {{")
                ha("                const uint ta = 8u * b + fn, tb = ta + 1u;")
                ha("                const float2 d = ELEMS_F2(Acc[i][b].thread_elements()) * eouter;")
                ha("                if (ta < cnt) y[(ulong)tg_out_slot[ta] * y_stride + row] = d.x;")
                ha("                if (tb < cnt) y[(ulong)tg_out_slot[tb] * y_stride + row] = d.y;")
                ha("            }")
                ha("        }")
                ha("    }")
                ha("}")
            continue
        k = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4}[fid]
        wpr_c = (n * k) // 32
        for nb_val in (1, 2, 4):
            max_t = 8 * nb_val
            ha(
                f"inline void umoe_v2_f{fid}_nb{nb_val}("
                "const device half* my_xp, const device uint* ecodes, const device half* escales, "
                "threadgroup half* tg_x, const threadgroup half4* tg_ht4, const threadgroup half2* tg_ht2, "
                "const device half2* tv_rtr1, const device half2* tv_rtr2, const device half2* tv_rtr3, const device half* lut_rtr4, "
                "const threadgroup uint* tg_out_slot, device float* y, "
                "uint rbase, uint m_rows, uint y_stride, uint cnt, uint tid, uint lane) {"
            )
            ha("    const uint tok_i = tid >> 3, vec_k = tid & 7u, vk_q = vec_k >> 1, vk_h = vec_k & 1u;")
            ha("    const uint qid = lane / 4u, fm = (qid & 4u) + ((lane / 2u) % 4u), fn = (qid & 2u) * 2u + (lane % 2u) * 2u;")
            ha("    const uint q = fn / 2u, r0 = rbase + fm;")
            ha("    const device uint* p[RB];")
            ha(f"    for (uint i = 0; i < RB; i++) p[i] = ecodes + (ulong)min(r0 + 8u * i, m_rows - 1u) * {wpr_c}u;")
            ha(f"    const uint first = ({n}u - 8u - 8u * q) * {k}u, w_first = first >> 5, off = first & 31u;")
            ha(f"    simdgroup_float8x8 Acc[RB][{nb_val}];")
            ha(f"    for (uint i = 0; i < RB; i++) for (uint b = 0; b < {nb_val}u; b++) Acc[i][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);")
            if fid == 3:
                ha("    uint w_prev[RB];")
                ha(f"    for (uint i = 0; i < RB; i++) w_prev[i] = (w_first + 1u < {wpr_c}u) ? p[i][w_first + 1u] : 0u;")
            ha(f"    if (tok_i < {max_t}u) {{")
            ha("        const half4 xv = (tok_i < cnt) ? *(const device half4*)(my_xp) : half4(0.0h);")
            ha("        threadgroup half* rowp = tg_x + tok_i * XSTR;")
            ha("        *(threadgroup half2*)(rowp + (vk_h * 2u) * 8u + vk_q * 2u) = xv.xy;")
            ha("        *(threadgroup half2*)(rowp + (vk_h * 2u + 1u) * 8u + vk_q * 2u) = xv.zw;")
            ha("    }")
            ha("    threadgroup_barrier(mem_flags::mem_threadgroup);")
            ha(f"    for (uint s = 0; s < {steps}u; s++) {{")
            ha("        const uint cur_off = (s & 1u) * (32u * XSTR), nxt_off = ((s + 1u) & 1u) * (32u * XSTR);")
            ha(f"        if (s + 1u < {steps}u && tok_i < {max_t}u) {{")
            ha("            const half4 xv = (tok_i < cnt) ? *(const device half4*)(my_xp + 32u * (s + 1u)) : half4(0.0h);")
            ha("            threadgroup half* rowp = tg_x + nxt_off + tok_i * XSTR;")
            ha("            *(threadgroup half2*)(rowp + (vk_h * 2u) * 8u + vk_q * 2u) = xv.xy;")
            ha("            *(threadgroup half2*)(rowp + (vk_h * 2u + 1u) * 8u + vk_q * 2u) = xv.zw;")
            ha("        }")
            ha("        if (rbase < m_rows) {")
            ha(f"            simdgroup_half8x8 Xm[{nb_val}][4];")
            ha("            #pragma unroll")
            ha(f"            for (uint b = 0; b < {nb_val}u; b++) {{")
            ha("                #pragma unroll")
            ha("                for (uint j = 0; j < 4u; j++) simdgroup_load(Xm[b][j], tg_x + cur_off + (8u * b) * XSTR + 8u * j, XSTR, ulong2(0, 0), true);")
            ha("            }")
            ha(f"            const uint wi = w_first - s * {k}u;")
            ha("            #pragma unroll")
            ha("            for (uint i = 0; i < RB; i++) {")
            ha("                const device uint* pw = p[i] + wi; simdgroup_half8x8 Am;")
            _emit_mma_tg(fid, nb_val)
            ha("            }")
            ha("        }")
            ha("        threadgroup_barrier(mem_flags::mem_threadgroup);")
            ha("    }")
            ha("    if (rbase < m_rows) {")
            ha("        for (uint i = 0; i < RB; i++) {")
            ha("            const uint row = r0 + 8u * i; if (row >= m_rows) continue;")
            ha("            const float sc = (float)escales[row];")
            ha("            #pragma unroll")
            ha(f"            for (uint b = 0; b < {nb_val}u; b++) {{")
            ha("                const uint ta = 8u * b + fn, tb = ta + 1u;")
            ha("                const float2 d = ELEMS_F2(Acc[i][b].thread_elements()) * sc;")
            ha("                if (ta < cnt) y[(ulong)tg_out_slot[ta] * y_stride + row] = d.x;")
            ha("                if (tb < cnt) y[(ulong)tg_out_slot[tb] * y_stride + row] = d.y;")
            ha("            }")
            ha("        }")
            ha("    }")
            ha("}")

    L = [
        "const uint u = threadgroup_position_in_grid.x, rb_tg = threadgroup_position_in_grid.y;",
        "if (u >= unit_count[0]) return;",
        "const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup, tid = simd * 32u + lane;",
        "const uint m_rows = meta[0], x_div = meta[1];",
        "const uint eid = units[u * 3u + 0u], base = units[u * 3u + 1u], cnt = units[u * 3u + 2u];",
    ]
    if n_rg > 1:
        L += [
            f"const uint rg_sz = m_rows / {n_rg}u, r_tg = rb_tg * (SG * 8u * RB), rg = r_tg / rg_sz;",
            f"const uint rbase = (r_tg - rg * rg_sz) + simd * (8u * RB), m_grp = rg_sz, idx_rg = eid * {n_rg}u + rg;",
            "const uint fid = exp_fid[idx_rg];",
            "device float* y_ptr = y + rg * rg_sz;",
        ]
    else:
        L += [
            "const uint fid = exp_fid[eid], rbase = rb_tg * (SG * 8u * RB) + simd * (8u * RB), m_grp = m_rows;",
            "device float* y_ptr = y;",
        ]
    L += [
        "const device half4* t_ht1 = (const device half4*)lut_ht1; const device half4* t_ht2 = (const device half4*)lut_ht2;",
        "const device half2* tv_rtr1 = (const device half2*)lut_rtr1; const device half2* tv_rtr2 = (const device half2*)lut_rtr2; const device half2* tv_rtr3 = (const device half2*)lut_rtr3;",
        "threadgroup half tg_x[64 * XSTR];",
        "threadgroup half4 tg_ht4[512]; const threadgroup half2* tg_ht2 = (const threadgroup half2*)tg_ht4;" if has_ht else "threadgroup half4 tg_ht4[1]; const threadgroup half2* tg_ht2 = (const threadgroup half2*)tg_ht4;",
        "threadgroup uint tg_out_slot[32];",
    ]
    if has_ht:
        L += [
            f"if (fid == 0u) {{ for (uint idx = tid; idx < 512u; idx += {sg * 32}u) tg_ht4[idx] = t_ht1[idx]; }}",
            f"else if (fid == 1u) {{ for (uint idx = tid; idx < 512u; idx += {sg * 32}u) tg_ht4[idx] = t_ht2[idx]; }}",
            f"else if (fid == 2u) {{ for (uint idx = tid; idx < 512u; idx += {sg * 32}u) tg_ht4[idx] = ((const device half4*)lut_ht3)[idx]; }}",
        ]
    if has_nv:
        L += [
            "threadgroup half tg_e2m1[16]; threadgroup half tg_e4m3[256];",
            f"if (fid == 7u) {{ if (tid < 16u) tg_e2m1[tid] = lut_e2m1[tid]; for (uint idx = tid; idx < 256u; idx += {sg * 32}u) tg_e4m3[idx] = lut_e4m3[idx]; }}",
        ]
    L.append("if (tid < cnt) tg_out_slot[tid] = " + ("base + tid;" if sorted_out else "slot_order[base + tid];"))
    L.append("threadgroup_barrier(mem_flags::mem_threadgroup);")
    L.append("const uint tok_i = tid >> 3, vec_k = tid & 7u;")
    xbase_expr = "((fid == 7u) ? x_raw : x)" if has_nv else "x"
    if sorted_in:
        L.append(f"const device half* my_xp = (tok_i < cnt) ? ({xbase_expr} + (ulong)(base + tok_i) * {n}u + 4u * vec_k) : {xbase_expr};")
    else:
        L.append("const uint my_orig = (tok_i < cnt) ? slot_order[base + tok_i] : 0u;")
        L.append("const uint my_xrow = (x_div == 1u) ? my_orig : (my_orig >> 3);")
        L.append(f"const device half* my_xp = (tok_i < cnt) ? ({xbase_expr} + (ulong)my_xrow * {n}u + 4u * vec_k) : {xbase_expr};")
    idx_expr = "idx_rg" if n_rg > 1 else "eid"
    L.append(f"const device uint* ec = codes + exp_code_off[{idx_expr}]; const device half* es = scales + exp_scale_off[{idx_expr}];")
    L.append("switch (fid) {")
    for fid in fids:
        L.append(f"    case {fid}u:")
        if fid == 7:
            L.append("        { const device uchar* env_s = nv_scales + exp_nv_scale_off[eid]; const float eout = nv_outer[eid];")
            L.append("          if (cnt <= 8u) umoe_v2_f7_nb1(my_xp, ec, env_s, eout, tg_x, tg_e2m1, tg_e4m3, tg_out_slot, y_ptr, rbase, m_grp, m_rows, cnt, tid, lane);")
            L.append("          else if (cnt <= 16u) umoe_v2_f7_nb2(my_xp, ec, env_s, eout, tg_x, tg_e2m1, tg_e4m3, tg_out_slot, y_ptr, rbase, m_grp, m_rows, cnt, tid, lane);")
            L.append("          else umoe_v2_f7_nb4(my_xp, ec, env_s, eout, tg_x, tg_e2m1, tg_e4m3, tg_out_slot, y_ptr, rbase, m_grp, m_rows, cnt, tid, lane); }")
        else:
            L.append(f"        if (cnt <= 8u) umoe_v2_f{fid}_nb1(my_xp, ec, es, tg_x, tg_ht4, tg_ht2, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, tg_out_slot, y_ptr, rbase, m_grp, m_rows, cnt, tid, lane);")
            L.append(f"        else if (cnt <= 16u) umoe_v2_f{fid}_nb2(my_xp, ec, es, tg_x, tg_ht4, tg_ht2, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, tg_out_slot, y_ptr, rbase, m_grp, m_rows, cnt, tid, lane);")
            L.append(f"        else umoe_v2_f{fid}_nb4(my_xp, ec, es, tg_x, tg_ht4, tg_ht2, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, tg_out_slot, y_ptr, rbase, m_grp, m_rows, cnt, tid, lane);")
        L.append("        break;")
    L.append("}")
    in_names = ["x", "codes", "scales", "exp_fid", "exp_code_off", "exp_scale_off", "lut_ht1", "lut_ht2", "lut_ht3", "lut_rtr1", "lut_rtr2", "lut_rtr3", "lut_rtr4", "units", "unit_count", "meta", "slot_order"]
    if has_nv:
        in_names += ["x_raw", "nv_scales", "exp_nv_scale_off", "nv_outer", "lut_e2m1", "lut_e4m3"]
    tag = "_".join(str(f) for f in fids)
    return mx.fast.metal_kernel(
        name=f"ink_umoe_prefill_v2_n{n}_f{tag}_si{int(sorted_in)}_so{int(sorted_out)}_rb{rb}_s{sg}_rg{n_rg}",
        input_names=in_names,
        output_names=["y"],
        header="\n".join(H),
        source="\n".join(L),
    )


@lru_cache(maxsize=2)
def _make_moe_offsets_kernel():
    src = """
    // Single threadgroup of 128 threads computes prefix offsets[0..128] from sorted_eids[0..n_slots-1]!
    // Sentinel-safe: offsets[128] = lower_bound(sorted_eids, 128u) excludes 0xFFFFFFFFu masked slots!
    const uint e = thread_position_in_threadgroup.x;
    const uint n_slots = meta[0];
    uint lo = 0u, hi = n_slots;
    while (lo < hi) {
        const uint mid = (lo + hi) >> 1;
        if (sorted_eids[mid] < e) lo = mid + 1u; else hi = mid;
    }
    offsets[e] = lo;
    if (e == 0u) {
        uint lo2 = 0u, hi2 = n_slots;
        while (lo2 < hi2) {
            const uint mid = (lo2 + hi2) >> 1;
            if (sorted_eids[mid] < 128u) lo2 = mid + 1u; else hi2 = mid;
        }
        offsets[128] = lo2;
    }
    """
    return mx.fast.metal_kernel(
        name="ink_moe_offsets",
        input_names=["sorted_eids", "meta"],
        output_names=["offsets"],
        source=src,
    )


def compute_moe_sorted_offsets(eids_flat: mx.array) -> tuple[mx.array, mx.array]:
    n_slots = int(eids_flat.shape[0])
    order = mx.argsort(eids_flat).astype(mx.uint32)
    sorted_eids = eids_flat[order].astype(mx.uint32)
    offsets = _make_moe_offsets_kernel()(
        inputs=[sorted_eids, _u32(n_slots)],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(129,)],
        output_dtypes=[mx.uint32],
    )[0]
    return offsets, order


@lru_cache(maxsize=32)
def _make_moe_plan_only_kernel(fids: tuple[int, ...]):
    """Single-threadgroup parallel compaction of `eids[0..n_slots-1]` -> `plan[129]` (fallback when plan not fused with router)."""
    L = [
        "const uint tid = thread_position_in_threadgroup.x;",
        "const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;",
        "const uint n_slots = meta[0];",
        "threadgroup uint tg_eids[64], tg_wsum[8][4], tg_fbase[8];",
        "if (tid < n_slots) tg_eids[tid] = expert_ids[tid];",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        "uint my_p0 = 0u, my_p1 = 0u, my_cnt = 0u, my_nu = 0u, my_fid = 99u;",
        "if (tid < 128u) {",
        "    for (uint s = 0; s < n_slots; s++) { if (tg_eids[s] == tid && my_cnt < 8u) { if (my_cnt < 4u) my_p0 |= (s << (my_cnt * 8u)); else my_p1 |= (s << ((my_cnt - 4u) * 8u)); my_cnt++; } }",
        "    my_nu = (my_cnt == 0u) ? 0u : ((my_cnt <= 4u) ? 1u : 2u);",
        "    if (my_cnt > 0u) my_fid = exp_fid[tid];",
        "}",
    ]
    a = L.append
    for fi, fid in enumerate(fids):
        a(f"uint pfx_{fi} = 0u;")
        a(f"if (tid < 128u) {{ uint u_f = (my_fid == {fid}u) ? my_nu : 0u; pfx_{fi} = simd_prefix_exclusive_sum(u_f); uint ws = simd_sum(u_f); if (lane == 0u) tg_wsum[{fi}][simd] = ws; }}")
    a("threadgroup_barrier(mem_flags::mem_threadgroup);")
    a("if (tid == 0u) { uint acc = 0u;")
    for fi in range(len(fids)):
        a(f"    tg_fbase[{fi}] = acc; acc += tg_wsum[{fi}][0] + tg_wsum[{fi}][1] + tg_wsum[{fi}][2] + tg_wsum[{fi}][3];")
    a("    plan[0] = acc; }")
    a("threadgroup_barrier(mem_flags::mem_threadgroup);")
    a("if (tid < 128u && my_cnt > 0u) { uint dst = 0u;")
    for fi, fid in enumerate(fids):
        a(f"    if (my_fid == {fid}u) {{ uint wb = tg_fbase[{fi}]; for (uint w = 0; w < simd; w++) wb += tg_wsum[{fi}][w]; dst = wb + pfx_{fi}; }}")
    a("    if (my_cnt <= 4u) { plan[1u + dst] = tid | (my_cnt << 8u); plan[65u + dst] = my_p0; }")
    a("    else { plan[1u + dst] = tid | (4u << 8u); plan[65u + dst] = my_p0; plan[2u + dst] = tid | ((my_cnt - 4u) << 8u); plan[66u + dst] = my_p1; }")
    a("}")
    tag = "_".join(str(f) for f in fids)
    return mx.fast.metal_kernel(
        name=f"ink_moe_plan_only_f{tag}",
        input_names=["expert_ids", "exp_fid", "meta"],
        output_names=["plan"],
        source="\n".join(L),
    )


class UnifiedMoESide:
    """Packs all 128 experts of one MoE side (`gu` or `dn`) across ALL formats into ONE flat GPU launch."""

    def __init__(
        self,
        slabs: list,
        map_fid: mx.array,
        map_loc: mx.array,
        m_rows: int,
        n_cols: int,
        fast_luts: dict[str, mx.array],
        nvfp4_slab: tuple[mx.array, mx.array, mx.array] | None = None,
        rg_map: np.ndarray | None = None,
    ):
        self.m_rows = int(m_rows)
        self.n_cols = int(n_cols)
        self.has_nvfp4 = nvfp4_slab is not None
        self.n_rg = int(rg_map.shape[1]) if rg_map is not None else 1
        rg_rows = self.m_rows // self.n_rg
        assert rg_rows * self.n_rg == self.m_rows, f"m_rows={m_rows} not divisible by n_rg={self.n_rg}"
        mf = np.array(map_fid, dtype=np.int32)
        ml = np.array(map_loc, dtype=np.uint32)
        slab_by_fid = {int(fid): (np.array(c, dtype=np.uint32), np.array(s, dtype=np.float16)) for fid, c, s, _, _ in slabs}
        if self.has_nvfp4:
            c_nv, s_nv_u8, out_nv_f32 = nvfp4_slab
            c_nv_np = np.ascontiguousarray(np.array(c_nv, dtype=np.uint32)).reshape(-1, rg_rows, self.n_cols // 8)
            slab_by_fid[7] = (c_nv_np, np.zeros((1, 64), dtype=np.float16))
        codes_parts, scales_parts = [], []
        fid_code_base, fid_scale_base, fid_wpr = {}, {}, {}
        c_off = s_off = 0
        for fid in sorted(slab_by_fid):
            c, s = slab_by_fid[fid]
            fid_code_base[fid] = c_off
            fid_scale_base[fid] = s_off
            fid_wpr[fid] = int(c.shape[2])
            codes_parts.append(c.reshape(-1))
            scales_parts.append(s.reshape(-1))
            c_off += c.size
            s_off += s.size
        codes_parts.append(np.zeros(4, dtype=np.uint32))
        scales_parts.append(np.zeros(64, dtype=np.float16))

        if self.n_rg > 1:
            assert rg_map is not None and rg_map.shape == (128, self.n_rg, 2)
            rg_fid_arr = np.zeros(128 * self.n_rg, dtype=np.uint32)
            exp_code_off = np.zeros(128 * self.n_rg, dtype=np.uint32)
            exp_scale_off = np.zeros(128 * self.n_rg, dtype=np.uint32)
            for e in range(128):
                for rg in range(self.n_rg):
                    f = int(rg_map[e, rg, 0])
                    loc = int(rg_map[e, rg, 1])
                    idx_rg = e * self.n_rg + rg
                    rg_fid_arr[idx_rg] = f
                    exp_code_off[idx_rg] = fid_code_base[f] + loc * rg_rows * fid_wpr[f]
                    exp_scale_off[idx_rg] = fid_scale_base[f] + loc * rg_rows
            # `self.exp_fid` stays shape (128,) for router/prefill expert planners; `self._kern_fid` is passed to MoE kernels!
            self.exp_fid = mx.array(rg_map[:, 0, 0].astype(np.uint32))
            self._kern_fid = mx.array(rg_fid_arr)
            exp_nv_scale_off = np.zeros(128, dtype=np.uint32)
            nv_outer_128 = np.zeros(128, dtype=np.float32)
        else:
            exp_code_off = np.zeros(128, dtype=np.uint32)
            exp_scale_off = np.zeros(128, dtype=np.uint32)
            exp_nv_scale_off = np.zeros(128, dtype=np.uint32)
            nv_outer_128 = np.zeros(128, dtype=np.float32)
            spr_nv = self.n_cols // 16
            out_nv_np = np.array(nvfp4_slab[2], dtype=np.float32).reshape(-1) if self.has_nvfp4 else None
            for e in range(128):
                f = int(mf[e])
                loc = int(ml[e])
                exp_code_off[e] = fid_code_base[f] + loc * self.m_rows * fid_wpr[f]
                if f == 7:
                    exp_scale_off[e] = fid_scale_base[7]
                    exp_nv_scale_off[e] = loc * self.m_rows * spr_nv
                    nv_outer_128[e] = out_nv_np[loc]
                else:
                    exp_scale_off[e] = fid_scale_base[f] + loc * self.m_rows
            self.exp_fid = mx.array(mf.astype(np.uint32))
            self._kern_fid = self.exp_fid

        self.codes = mx.array(np.concatenate(codes_parts))
        self.scales = mx.array(np.concatenate(scales_parts))
        self.exp_code_off = mx.array(exp_code_off)
        self.exp_scale_off = mx.array(exp_scale_off)
        self.fids = tuple(sorted(slab_by_fid.keys()))
        self._lut_args = [
            fast_luts["ht1"], fast_luts["ht2"], fast_luts["ht3"],
            fast_luts["rtr1_g3"], fast_luts["rtr2_g2"], fast_luts["rtr3_g2"], fast_luts["rtr4"],
        ]
        self._lut_args_sg8 = [
            fast_luts["ht1"], fast_luts["ht2"], fast_luts["ht3"],
            fast_luts["rtr1_g2"], fast_luts["rtr2_g2"], fast_luts["rtr3_g2"], fast_luts["rtr4"],
        ]
        if self.has_nvfp4:
            s_nv_flat = np.concatenate([np.array(nvfp4_slab[1], dtype=np.uint8).reshape(-1), np.zeros(64, dtype=np.uint8)])
            self.nv_scales = mx.array(s_nv_flat)
            self.exp_nv_scale_off = mx.array(exp_nv_scale_off)
            self.nv_outer = mx.array(nv_outer_128)
            self._nv_args = [
                self.nv_scales, self.exp_nv_scale_off, self.nv_outer,
                fast_luts["nvfp4_e2m1"], fast_luts["nvfp4_e4m3"],
            ]
            mx.eval(self.codes, self.scales, self.exp_fid, self._kern_fid, self.exp_code_off, self.exp_scale_off, self.nv_scales, self.exp_nv_scale_off, self.nv_outer)
        else:
            self._nv_args = []
            mx.eval(self.codes, self.scales, self.exp_fid, self._kern_fid, self.exp_code_off, self.exp_scale_off)

    def __call__(
        self,
        x_rot_f16: mx.array,
        eids_flat: mx.array,
        x_div: int,
        sorted_cache: tuple[mx.array, mx.array] | None = None,
        plan: mx.array | None = None,
        dynamic_routing: bool = False,
        sorted_in: bool = False,
        sorted_out: bool = False,
        x_raw_f16: mx.array | None = None,
    ) -> mx.array:
        n_slots = int(eids_flat.shape[0])
        nv_tail = [(x_raw_f16 if x_raw_f16 is not None else x_rot_f16).reshape(-1), *self._nv_args] if self.has_nvfp4 else []
        if n_slots <= 8 and not sorted_in and not sorted_out:
            sg, ntg = (4, 60) if _is_m4() else (8, 160)
            kern = _make_unified_moe_kernel(self.n_cols, self.fids, "t1", rps=4, sg=sg, n_rg=self.n_rg)
            mt = _u32(self.m_rows, n_slots, x_div, ntg)
            return kern(
                inputs=[x_rot_f16.reshape(-1), self.codes, self.scales, self._kern_fid, self.exp_code_off, self.exp_scale_off, *self._lut_args, eids_flat.astype(mx.uint32), mt, *nv_tail],
                grid=(ntg * sg * 32, 1, 1),
                threadgroup=(sg * 32, 1, 1),
                output_shapes=[(n_slots, self.m_rows)],
                output_dtypes=[mx.float32],
            )[0]
        if n_slots <= 64 and not sorted_in and not sorted_out:
            # T=2..8 pre-planned lean half-warp mc=4 + low-reg c3/c4 + lpr=8 (sentinel-safe, zero compaction inside MoE!)
            if plan is None:
                plan = _make_moe_plan_only_kernel(self.fids)(
                    inputs=[eids_flat.astype(mx.uint32), self.exp_fid, _u32(n_slots)],
                    grid=(128, 1, 1),
                    threadgroup=(128, 1, 1),
                    output_shapes=[(129,)],
                    output_dtypes=[mx.uint32],
                )[0]
            if _is_m4():
                sg_g, rps_g = 8, 2
                lpr_g = 16 if (self.n_cols == 2304 and n_slots > 32) else 8
                ntg_g = 120 if (self.n_cols == 2304 and 40 < n_slots <= 56) else 160
            else:
                sg_g, rps_g, ntg_g, lpr_g = 8, 4, 160, 8
            kern = _make_unified_moe_kernel(self.n_cols, self.fids, "grouped", rps=rps_g, sg=sg_g, lpr=lpr_g, n_rg=self.n_rg)
            mt = _u32(self.m_rows, x_div, ntg_g)
            return kern(
                inputs=[x_rot_f16.reshape(-1), self.codes, self.scales, self._kern_fid, self.exp_code_off, self.exp_scale_off, *self._lut_args, plan, mt, *nv_tail],
                grid=(ntg_g * sg_g * 32, 1, 1),
                threadgroup=(sg_g * 32, 1, 1),
                output_shapes=[(n_slots, self.m_rows)],
                output_dtypes=[mx.float32],
            )[0]
        # Prefill (n_slots > 64, or explicit sorted_in/sorted_out): format-homogeneous 32-token work units!
        offsets, order = sorted_cache if sorted_cache is not None else compute_moe_sorted_offsets(eids_flat)
        tile_t = 32
        units, ucnt, max_u = plan_moe_prefill_units(offsets, self.exp_fid, self.fids, n_slots, tile_t=tile_t)
        if _has_mpp():
            m_tile, sg_p = 64, 4
            rblocks = (self.m_rows + m_tile - 1) // m_tile
            kern = _make_unified_moe_prefill_mpp_kernel(
                self.n_cols, self.fids, sorted_in=sorted_in, sorted_out=sorted_out, m_tile=m_tile, tile_t=tile_t, k_tile=32, sg=sg_p, n_rg=self.n_rg,
            )
            return kern(
                inputs=[x_rot_f16.reshape(-1), self.codes, self.scales, self._kern_fid, self.exp_code_off, self.exp_scale_off, *self._lut_args, units, ucnt, _u32(self.m_rows, x_div), order, *nv_tail],
                grid=(max_u * sg_p * 32, rblocks, 1),
                threadgroup=(sg_p * 32, 1, 1),
                output_shapes=[(n_slots, self.m_rows)],
                output_dtypes=[mx.float32],
            )[0]
        sg_p, rb_p = 8, 2
        rows_per_tg = sg_p * 8 * rb_p
        rblocks = (self.m_rows + rows_per_tg - 1) // rows_per_tg
        kern = _make_unified_moe_prefill_v2_kernel(
            self.n_cols, self.fids, sorted_in=sorted_in, sorted_out=sorted_out, sg=sg_p, rb=rb_p, n_rg=self.n_rg,
        )
        return kern(
            inputs=[x_rot_f16.reshape(-1), self.codes, self.scales, self._kern_fid, self.exp_code_off, self.exp_scale_off, *self._lut_args_sg8, units, ucnt, _u32(self.m_rows, x_div), order, *nv_tail],
            grid=(max_u * sg_p * 32, rblocks, 1),
            threadgroup=(sg_p * 32, 1, 1),
            output_shapes=[(n_slots, self.m_rows)],
            output_dtypes=[mx.float32],
        )[0]


# ----------------------------------------------------------------------------- Fused Router Top-8 + MoE Plan Kernel

_DYN_TOP8_FILTER_BLOCK = """
        // Sort the 8 selected experts in descending top_s order (stable insertion sort in registers)
        for (uint i = 1u; i < 8u; i++) {
            const float ks = top_s[i];
            const uint ke = top_e[i];
            uint j = i;
            while (j > 0u && top_s[j - 1u] < ks) {
                top_s[j] = top_s[j - 1u];
                top_e[j] = top_e[j - 1u];
                j--;
            }
            top_s[j] = ks;
            top_e[j] = ke;
        }
        const float w_max = top_s[0];
        const float thr = ratio_thr * w_max;
        const bool keep = (lane < k_max) && ((lane < k_min) || (top_s[lane] >= thr));
        float denom = sum_s;
        if (do_renorm != 0u) {
            float sum_kept = 0.0f;
            #pragma unroll
            for (uint i = 0u; i < 8u; i++) {
                if ((i < k_max) && ((i < k_min) || (top_s[i] >= thr))) sum_kept += top_s[i];
            }
            denom = sum_kept;
        }
        const float inv = route_scale / (denom + 1e-20f);
        const uint out_e = keep ? top_e[lane] : 0xFFFFFFFFu;
        const float out_w = keep ? (top_s[lane] * inv) : 0.0f;
"""


@lru_cache(maxsize=4)
def _make_sigmoid_top8_norm_kernel(dynamic_routing: bool = False):
    # One warp (32 lanes) per token over 128 experts (4 experts/lane) -> exact sigmoid + bias + top-8 + norm in 1 launch!
    if not dynamic_routing:
        src = """
    const uint lane = thread_index_in_simdgroup;
    const uint t = threadgroup_position_in_grid.y;
    const float route_scale = cfg[0];
    const device half* lp = logits + (ulong)t * 128u;

    float s[4], b[4];
    #pragma unroll
    for (uint j = 0; j < 4u; j++) {
        const uint e = lane + j * 32u;
        const float sc = 1.0f / (1.0f + metal::precise::exp(-(float)lp[e]));
        s[j] = sc;
        b[j] = sc + bias[e];
    }

    uint top_e[8];
    float top_s[8];
    float sum_s = 0.0f;
    #pragma unroll
    for (uint k = 0; k < 8u; k++) {
        float loc_b = b[0];
        uint loc_j = 0u;
        if (b[1] > loc_b) { loc_b = b[1]; loc_j = 1u; }
        if (b[2] > loc_b) { loc_b = b[2]; loc_j = 2u; }
        if (b[3] > loc_b) { loc_b = b[3]; loc_j = 3u; }
        const float g_max = simd_max(loc_b);
        const uint cand_e = (loc_b == g_max) ? (lane + loc_j * 32u) : 999u;
        const uint win_e = simd_min(cand_e);
        const uint win_lane = win_e & 31u;
        const uint win_j = win_e >> 5;
        const float win_s = simd_broadcast(s[win_j], win_lane);
        if (lane == win_lane) b[win_j] = -1e30f;
        top_e[k] = win_e;
        top_s[k] = win_s;
        sum_s += win_s;
    }
    if (lane < 8u) {
        const float inv = route_scale / (sum_s + 1e-20f);
        eids_out[(ulong)t * 8u + lane] = top_e[lane];
        w_out[(ulong)t * 8u + lane] = top_s[lane] * inv;
    }
    """
        return mx.fast.metal_kernel(
            name="ink_sigmoid_top8_norm",
            input_names=["logits", "bias", "cfg"],
            output_names=["eids_out", "w_out"],
            source=src,
        )

    src = f"""
    const uint lane = thread_index_in_simdgroup;
    const uint t = threadgroup_position_in_grid.y;
    const float route_scale = cfg[0], ratio_thr = cfg[1];
    const uint k_max = meta[0], k_min = meta[1], do_renorm = meta[2];
    const device half* lp = logits + (ulong)t * 128u;

    float s[4], b[4];
    #pragma unroll
    for (uint j = 0; j < 4u; j++) {{
        const uint e = lane + j * 32u;
        const float sc = 1.0f / (1.0f + metal::precise::exp(-(float)lp[e]));
        s[j] = sc;
        b[j] = sc + bias[e];
    }}

    uint top_e[8];
    float top_s[8];
    float sum_s = 0.0f;
    #pragma unroll
    for (uint k = 0; k < 8u; k++) {{
        float loc_b = b[0];
        uint loc_j = 0u;
        if (b[1] > loc_b) {{ loc_b = b[1]; loc_j = 1u; }}
        if (b[2] > loc_b) {{ loc_b = b[2]; loc_j = 2u; }}
        if (b[3] > loc_b) {{ loc_b = b[3]; loc_j = 3u; }}
        const float g_max = simd_max(loc_b);
        const uint cand_e = (loc_b == g_max) ? (lane + loc_j * 32u) : 999u;
        const uint win_e = simd_min(cand_e);
        const uint win_lane = win_e & 31u;
        const uint win_j = win_e >> 5;
        const float win_s = simd_broadcast(s[win_j], win_lane);
        if (lane == win_lane) b[win_j] = -1e30f;
        top_e[k] = win_e;
        top_s[k] = win_s;
        sum_s += win_s;
    }}
    if (lane < 8u) {{
        {_DYN_TOP8_FILTER_BLOCK}
        eids_out[(ulong)t * 8u + lane] = out_e;
        w_out[(ulong)t * 8u + lane] = out_w;
    }}
    """
    return mx.fast.metal_kernel(
        name="ink_sigmoid_top8_norm_dyn",
        input_names=["logits", "bias", "cfg", "meta"],
        output_names=["eids_out", "w_out"],
        source=src,
    )


def sigmoid_top8_norm_fast(
    logits_f16: mx.array,
    router_bias: mx.array,
    route_scale: float,
    k_max: int = 8,
    k_min: int = 8,
    ratio_thr: float = 0.0,
    renorm: bool = False,
) -> tuple[mx.array, mx.array]:
    """Fused sigmoid + correction bias + exact Top-8 (or Dynamic Top-K with `0xFFFFFFFFu` sentinel) + scale norm."""
    T = int(logits_f16.shape[0])
    dynamic_routing = (k_max < 8) or (ratio_thr > 0.0)
    if not dynamic_routing:
        return _make_sigmoid_top8_norm_kernel(False)(
            inputs=[logits_f16.reshape(-1), router_bias.reshape(-1), _f32_arr(route_scale)],
            grid=(32, T, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(T * 8,), (T * 8,)],
            output_dtypes=[mx.uint32, mx.float32],
        )
    k_max_c = max(1, min(8, int(k_max)))
    k_min_c = max(0, min(k_max_c, int(k_min)))
    return _make_sigmoid_top8_norm_kernel(True)(
        inputs=[
            logits_f16.reshape(-1),
            router_bias.reshape(-1),
            _f32_arr(route_scale, float(ratio_thr)),
            _u32(k_max_c, k_min_c, int(renorm)),
        ],
        grid=(32, T, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(T * 8,), (T * 8,)],
        output_dtypes=[mx.uint32, mx.float32],
    )


@lru_cache(maxsize=64)
def _make_sigmoid_top8_plan_kernel(gu_fids: tuple[int, ...], dn_fids: tuple[int, ...], dynamic_routing: bool = False):
    """Single-TG (256 threads) fused Router Top-8 (+ optional Dynamic Top-K) + BOTH `gu_plan[129]` and `dn_plan[129]` for `2 <= T <= 8`!"""
    L = [
        "const uint tid = thread_position_in_threadgroup.x;",
        "const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;",
        "const uint T = meta[0], n_slots = T * 8u;",
        "const float route_scale = cfg[0];",
    ]
    if dynamic_routing:
        L.append("const float ratio_thr = cfg[1];")
        L.append("const uint k_max = meta[1], k_min = meta[2], do_renorm = meta[3];")
    L += [
        "threadgroup uint tg_eids[64], tg_wsum[8][4], tg_fbase[8];",
        "if (simd < T) {",
        "    const device half* lp = logits + (ulong)simd * 128u;",
        "    float s[4], b[4];",
        "    #pragma unroll",
        "    for (uint j = 0; j < 4u; j++) { const uint e = lane + j * 32u; const float sc = 1.0f / (1.0f + metal::precise::exp(-(float)lp[e])); s[j] = sc; b[j] = sc + bias[e]; }",
        "    uint top_e[8]; float top_s[8]; float sum_s = 0.0f;",
        "    #pragma unroll",
        "    for (uint k = 0; k < 8u; k++) {",
        "        float loc_b = b[0]; uint loc_j = 0u;",
        "        if (b[1] > loc_b) { loc_b = b[1]; loc_j = 1u; }",
        "        if (b[2] > loc_b) { loc_b = b[2]; loc_j = 2u; }",
        "        if (b[3] > loc_b) { loc_b = b[3]; loc_j = 3u; }",
        "        const float g_max = simd_max(loc_b);",
        "        const uint win_e = simd_min((loc_b == g_max) ? (lane + loc_j * 32u) : 999u);",
        "        const uint win_lane = win_e & 31u, win_j = win_e >> 5;",
        "        const float win_s = simd_broadcast(s[win_j], win_lane);",
        "        if (lane == win_lane) b[win_j] = -1e30f;",
        "        top_e[k] = win_e; top_s[k] = win_s; sum_s += win_s;",
        "    }",
        "    if (lane < 8u) {",
    ]
    if dynamic_routing:
        L += [
            _DYN_TOP8_FILTER_BLOCK,
            "        tg_eids[simd * 8u + lane] = out_e;",
            "        eids_out[simd * 8u + lane] = out_e;",
            "        w_out[simd * 8u + lane] = out_w;",
        ]
    else:
        L += [
            "        const float inv = route_scale / (sum_s + 1e-20f);",
            "        tg_eids[simd * 8u + lane] = top_e[lane];",
            "        eids_out[simd * 8u + lane] = top_e[lane];",
            "        w_out[simd * 8u + lane] = top_s[lane] * inv;",
        ]
    L += [
        "    }",
        "}",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        "uint my_p0 = 0u, my_p1 = 0u, my_cnt = 0u, my_nu = 0u;",
        "if (tid < 128u) {",
        "    for (uint s = 0; s < n_slots; s++) { if (tg_eids[s] == tid && my_cnt < 8u) { if (my_cnt < 4u) my_p0 |= (s << (my_cnt * 8u)); else my_p1 |= (s << ((my_cnt - 4u) * 8u)); my_cnt++; } }",
        "    my_nu = (my_cnt == 0u) ? 0u : ((my_cnt <= 4u) ? 1u : 2u);",
        "}",
    ]
    a = L.append

    def emit_plan(side: str, fids_t: tuple[int, ...]):
        a(f"{{ uint my_fid = (tid < 128u && my_cnt > 0u) ? {side}_exp_fid[tid] : 99u;")
        for fi, fid in enumerate(fids_t):
            a(f"    uint pfx_{fi} = 0u;")
            a(f"    if (tid < 128u) {{ uint u_f = (my_fid == {fid}u) ? my_nu : 0u; pfx_{fi} = simd_prefix_exclusive_sum(u_f); uint ws = simd_sum(u_f); if (lane == 0u) tg_wsum[{fi}][simd] = ws; }}")
        a("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        a("    if (tid == 0u) { uint acc = 0u;")
        for fi in range(len(fids_t)):
            a(f"        tg_fbase[{fi}] = acc; acc += tg_wsum[{fi}][0] + tg_wsum[{fi}][1] + tg_wsum[{fi}][2] + tg_wsum[{fi}][3];")
        a(f"        {side}_plan[0] = acc; }}")
        a("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        a("    if (tid < 128u && my_cnt > 0u) { uint dst = 0u;")
        for fi, fid in enumerate(fids_t):
            a(f"        if (my_fid == {fid}u) {{ uint wb = tg_fbase[{fi}]; for (uint w = 0; w < simd; w++) wb += tg_wsum[{fi}][w]; dst = wb + pfx_{fi}; }}")
        a(f"        if (my_cnt <= 4u) {{ {side}_plan[1u + dst] = tid | (my_cnt << 8u); {side}_plan[65u + dst] = my_p0; }}")
        a(f"        else {{ {side}_plan[1u + dst] = tid | (4u << 8u); {side}_plan[65u + dst] = my_p0; {side}_plan[2u + dst] = tid | ((my_cnt - 4u) << 8u); {side}_plan[66u + dst] = my_p1; }}")
        a("    }")
        a("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        a("}")

    emit_plan("gu", gu_fids)
    emit_plan("dn", dn_fids)
    tag = "_".join(str(f) for f in gu_fids) + "__" + "_".join(str(f) for f in dn_fids)
    return mx.fast.metal_kernel(
        name=f"ink_sig_top8_plan_{tag}{'_dyn' if dynamic_routing else ''}",
        input_names=["logits", "bias", "gu_exp_fid", "dn_exp_fid", "meta", "cfg"],
        output_names=["eids_out", "w_out", "gu_plan", "dn_plan"],
        source="\n".join(L),
    )


def sigmoid_top8_norm_and_plan_fast(
    logits_f16: mx.array,
    router_bias: mx.array,
    route_scale: float,
    gu_exp_fid: mx.array,
    gu_fids: tuple[int, ...],
    dn_exp_fid: mx.array,
    dn_fids: tuple[int, ...],
    k_max: int = 8,
    k_min: int = 8,
    ratio_thr: float = 0.0,
    renorm: bool = False,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Fused Router Top-8 (+ optional Dynamic Top-K) + BOTH pre-compacted `gu_plan` & `dn_plan` in ONE launch for `2 <= T <= 8`."""
    T = int(logits_f16.shape[0])
    dynamic_routing = (k_max < 8) or (ratio_thr > 0.0)
    if not dynamic_routing:
        return _make_sigmoid_top8_plan_kernel(gu_fids, dn_fids, False)(
            inputs=[logits_f16.reshape(-1), router_bias.reshape(-1), gu_exp_fid, dn_exp_fid, _u32(T), _f32_arr(route_scale)],
            grid=(256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(T * 8,), (T * 8,), (129,), (129,)],
            output_dtypes=[mx.uint32, mx.float32, mx.uint32, mx.uint32],
        )
    k_max_c = max(1, min(8, int(k_max)))
    k_min_c = max(0, min(k_max_c, int(k_min)))
    return _make_sigmoid_top8_plan_kernel(gu_fids, dn_fids, True)(
        inputs=[
            logits_f16.reshape(-1),
            router_bias.reshape(-1),
            gu_exp_fid,
            dn_exp_fid,
            _u32(T, k_max_c, k_min_c, int(renorm)),
            _f32_arr(route_scale, float(ratio_thr)),
        ],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T * 8,), (T * 8,), (129,), (129,)],
        output_dtypes=[mx.uint32, mx.float32, mx.uint32, mx.uint32],
    )


# ----------------------------------------------------------------------------- Fused Glue Metal Kernels

@lru_cache(maxsize=4)
def _make_block_attn_res_kernel(with_norm: bool):
    norm_inputs = ["q", "blocks", "partial", "norm_w", "meta"] if with_norm else ["q", "blocks", "partial", "meta"]
    store_code = (
        """
        float ss = 0.0f;
        #pragma unroll
        for (uint e = 0; e < 9u; e++) ss += acc[qi][e] * acc[qi][e];
        ss = simd_sum(ss);
        if (lane == 0u) red[qi][simd] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (uint i = 0; i < 8u; i++) tot += red[qi][i];
        const float ir_out = metal::precise::rsqrt(tot * (1.0f / 2304.0f) + 1e-6f);
        device float* op = out + ((ulong)qi * T + t) * 2304u;
        #pragma unroll
        for (uint e = 0; e < 9u; e++) {
            const uint ch = tid + e * 256u;
            op[ch] = acc[qi][e] * ir_out * norm_w[ch];
        }
        """
        if with_norm
        else """
        device float* op = out + ((ulong)qi * T + t) * 2304u;
        #pragma unroll
        for (uint e = 0; e < 9u; e++) op[tid + e * 256u] = acc[qi][e];
        """
    )
    src = f"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.y;
    const uint n_blocks = meta[0], num_q = meta[1], T = meta[2];
    const uint S = n_blocks + 1u;

    threadgroup float red[4][8];
    threadgroup float tg_ss[8][8];
    threadgroup float tg_dp[8][4][8];
    threadgroup float tg_w[4][8];

    float qv[4][9];
    for (uint qi = 0; qi < num_q; qi++) {{
        const device float* qp = q + (ulong)qi * 2304u;
        #pragma unroll
        for (uint e = 0; e < 9u; e++) qv[qi][e] = qp[tid + e * 256u];
    }}

    // Pass 1: accumulate all S states' norms and dot products with ZERO inner barriers!
    for (uint s = 0; s < S; s++) {{
        const device float* sp = (s < n_blocks)
            ? (blocks + ((ulong)s * T + t) * 2304u)
            : (partial + (ulong)t * 2304u);
        float sv[9];
        float ss = 0.0f;
        #pragma unroll
        for (uint e = 0; e < 9u; e++) {{
            sv[e] = sp[tid + e * 256u];
            ss += sv[e] * sv[e];
        }}
        ss = simd_sum(ss);
        if (lane == 0u) tg_ss[s][simd] = ss;
        for (uint qi = 0; qi < num_q; qi++) {{
            float dp = 0.0f;
            #pragma unroll
            for (uint e = 0; e < 9u; e++) dp += qv[qi][e] * sv[e];
            dp = simd_sum(dp);
            if (lane == 0u) tg_dp[s][qi][simd] = dp;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {{
        float inv_rms[8];
        for (uint s = 0; s < S; s++) {{
            float tot = 0.0f;
            for (uint i = 0; i < 8u; i++) tot += tg_ss[s][i];
            inv_rms[s] = metal::precise::rsqrt(tot * (1.0f / 2304.0f) + 1e-6f);
        }}
        for (uint qi = 0; qi < num_q; qi++) {{
            float sc[8];
            float mxv = -1e30f;
            for (uint s = 0; s < S; s++) {{
                float dp = 0.0f;
                for (uint i = 0; i < 8u; i++) dp += tg_dp[s][qi][i];
                sc[s] = dp * inv_rms[s] * (1.0f / 48.0f);
                mxv = max(mxv, sc[s]);
            }}
            float sum = 0.0f;
            for (uint s = 0; s < S; s++) {{ float w = metal::precise::exp(sc[s] - mxv); tg_w[qi][s] = w; sum += w; }}
            const float inv = 1.0f / sum;
            for (uint s = 0; s < S; s++) tg_w[qi][s] *= inv * inv_rms[s];
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[4][9];
    for (uint qi = 0; qi < num_q; qi++) for (uint e = 0; e < 9u; e++) acc[qi][e] = 0.0f;
    for (uint s = 0; s < S; s++) {{
        const device float* sp = (s < n_blocks)
            ? (blocks + ((ulong)s * T + t) * 2304u)
            : (partial + (ulong)t * 2304u);
        float sv[9];
        #pragma unroll
        for (uint e = 0; e < 9u; e++) sv[e] = sp[tid + e * 256u];
        for (uint qi = 0; qi < num_q; qi++) {{
            const float w = tg_w[qi][s];
            #pragma unroll
            for (uint e = 0; e < 9u; e++) acc[qi][e] += w * sv[e];
        }}
    }}
    for (uint qi = 0; qi < num_q; qi++) {{
        {store_code}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_bar_v2_n{int(with_norm)}",
        input_names=norm_inputs,
        output_names=["out"],
        source=src,
    )


_EMPTY_F32 = mx.zeros((2304,), dtype=mx.float32)


def block_attn_res_fused(
    query: mx.array,
    blocks_cat: mx.array | None,
    n_blocks: int,
    partial: mx.array,
    norm_w: mx.array | None = None,
) -> mx.array:
    """Fused BlockAttnRes (+ optional RMSNorm) for H=2304. Returns `[num_q, T, 2304]` float32."""
    T = int(partial.size) // 2304
    num_q = 1 if query.ndim == 1 else int(query.shape[0])
    b_in = blocks_cat.reshape(-1) if (blocks_cat is not None and n_blocks > 0) else _EMPTY_F32
    mt = _u32(n_blocks, num_q, T)
    if norm_w is not None:
        return _make_block_attn_res_kernel(True)(
            inputs=[query.reshape(-1), b_in, partial.reshape(-1), norm_w.reshape(-1), mt],
            grid=(256, T, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(num_q, T, 2304)],
            output_dtypes=[mx.float32],
        )[0]
    return _make_block_attn_res_kernel(False)(
        inputs=[query.reshape(-1), b_in, partial.reshape(-1), mt],
        grid=(256, T, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(num_q, T, 2304)],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=32)
def make_tree_spec(parents: tuple[int, ...]) -> dict:
    """Precompute GPU arrays & traversal tables for an arbitrary causal speculation tree (T <= 8, parents[0] == -1)."""
    T = len(parents)
    assert 1 <= T <= 8 and parents[0] == -1
    depth = [0] * T
    mask = [0] * T
    lags = [0] * (T * 4)
    paths: list[list[int]] = []
    children: list[list[int]] = [[] for _ in range(T)]
    for t in range(T):
        p = parents[t]
        if t > 0:
            assert 0 <= p < t
            depth[t] = depth[p] + 1
            mask[t] = mask[p] | (1 << t)
            children[p].append(t)
            paths.append(paths[p] + [t])
        else:
            mask[0] = 1
            paths.append([0])
        # 4-tap conv lags [oldest..newest] = [p^3, p^2, p^1, t] (stepping -1 -> -2 -> -3 past root into conv prev[2,1,0])
        cur = t
        seq4 = [0, 0, 0, 0]
        for k in (3, 2, 1, 0):
            seq4[k] = cur
            if cur > 0:
                cur = parents[cur]
            elif cur == 0:
                cur = -1
            else:
                cur -= 1
        lags[t * 4 : t * 4 + 4] = seq4
    lag_mx = mx.array(np.array(lags, dtype=np.int32))
    depth_mx = mx.array(np.array(depth, dtype=np.uint32))
    mask_mx = mx.array(np.array(mask, dtype=np.uint32))
    mx.eval(lag_mx, depth_mx, mask_mx)
    return {
        "parents": parents,
        "T": T,
        "lag_mx": lag_mx,
        "depth_mx": depth_mx,
        "mask_mx": mask_mx,
        "depth": depth,
        "paths": paths,
        "children": children,
    }


@lru_cache(maxsize=2)
def _make_bar_norm_rht_kernel():
    # Fuses BlockAttnRes(num_q=1) + RMSNorm + float16 cast + RHT-2304(gu_signs) in ONE launch (eliminates 24 RHT launches + 24 casts)!
    # Uses bijective channel layout ch = lane * 72 + (simd * 9 + j) so acc[j] feeds Sylvester-32 warp shuffle directly with ZERO transpose!
    src = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;     // 0..31 (Sylvester-32 row a)
    const uint simd = simdgroup_index_in_threadgroup; // 0..7
    const uint t = threadgroup_position_in_grid.y;
    const uint n_blocks = meta[0], T = meta[1];
    const uint S = n_blocks + 1u;
    const uint c0 = simd * 9u;

    threadgroup float red[8];
    threadgroup float tg_ss[8][8];
    threadgroup float tg_dp[8][8];
    threadgroup float tg_w[8];
    threadgroup float tg_z[32 * 72];

    float qv[9];
    #pragma unroll
    for (uint j = 0; j < 9u; j++) qv[j] = q[lane * 72u + c0 + j];

    for (uint s = 0; s < S; s++) {
        const device float* sp = (s < n_blocks)
            ? (blocks + ((ulong)s * T + t) * 2304u)
            : (partial + (ulong)t * 2304u);
        float ss = 0.0f, dp = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 9u; j++) {
            const float v = sp[lane * 72u + c0 + j];
            ss += v * v;
            dp += qv[j] * v;
        }
        ss = simd_sum(ss);
        dp = simd_sum(dp);
        if (lane == 0u) { tg_ss[s][simd] = ss; tg_dp[s][simd] = dp; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float inv_rms[8], sc[8];
        float mxv = -1e30f;
        for (uint s = 0; s < S; s++) {
            float tot_s = 0.0f, tot_d = 0.0f;
            for (uint i = 0; i < 8u; i++) { tot_s += tg_ss[s][i]; tot_d += tg_dp[s][i]; }
            inv_rms[s] = metal::precise::rsqrt(tot_s * (1.0f / 2304.0f) + 1e-6f);
            sc[s] = tot_d * inv_rms[s] * (1.0f / 48.0f);
            mxv = max(mxv, sc[s]);
        }
        float sum = 0.0f;
        for (uint s = 0; s < S; s++) { float w = metal::precise::exp(sc[s] - mxv); tg_w[s] = w; sum += w; }
        const float inv = 1.0f / sum;
        for (uint s = 0; s < S; s++) tg_w[s] *= inv * inv_rms[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[9];
    #pragma unroll
    for (uint j = 0; j < 9u; j++) acc[j] = 0.0f;
    for (uint s = 0; s < S; s++) {
        const device float* sp = (s < n_blocks)
            ? (blocks + ((ulong)s * T + t) * 2304u)
            : (partial + (ulong)t * 2304u);
        const float w = tg_w[s];
        #pragma unroll
        for (uint j = 0; j < 9u; j++) acc[j] += w * sp[lane * 72u + c0 + j];
    }
    float ss_out = 0.0f;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) ss_out += acc[j] * acc[j];
    ss_out = simd_sum(ss_out);
    if (lane == 0u) red[simd] = ss_out;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot_out = 0.0f;
    for (uint i = 0; i < 8u; i++) tot_out += red[i];
    const float ir_out = metal::precise::rsqrt(tot_out * (1.0f / 2304.0f) + 1e-6f);

    device half* hp = h_f16_out + (ulong)t * 2304u;
    device half* rp = x_rot_out + (ulong)t * 2304u;
    const bool h1 = lane & 1u, h2 = lane & 2u, h4 = lane & 4u, h8 = lane & 8u, h16 = lane & 16u;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {
        const uint c = c0 + j;
        const uint idx = lane * 72u + c;
        const float hm = acc[j] * ir_out * norm_w[idx];
        hp[idx] = (half)hm;
        float v = hm * gu_signs[idx];
        float o;
        o = simd_shuffle_xor(v, 1u);  v = h1  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 2u);  v = h2  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 4u);  v = h4  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 8u);  v = h8  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 16u); v = h16 ? (o - v) : (v + o);
        tg_z[idx] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup float* zr = tg_z + lane * 72u;
    constexpr float INV_S = 1.0f / 48.0f;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {
        const uint c = c0 + j;
        const device float4* br = (const device float4*)(B + c * 72u);
        float racc = 0.0f;
        #pragma unroll
        for (uint k4 = 0; k4 < 18u; k4++) {
            const float4 b4 = br[k4];
            const uint k = k4 * 4u;
            racc += zr[k] * b4.x + zr[k + 1u] * b4.y + zr[k + 2u] * b4.z + zr[k + 3u] * b4.w;
        }
        rp[lane * 72u + c] = (half)(racc * INV_S);
    }
    """
    return mx.fast.metal_kernel(
        name="ink_bar_norm_rht_f16",
        input_names=["q", "blocks", "partial", "norm_w", "gu_signs", "B", "meta"],
        output_names=["h_f16_out", "x_rot_out"],
        source=src,
    )


def block_attn_res_norm_rht_fast(
    query: mx.array,
    blocks_cat: mx.array | None,
    n_blocks: int,
    partial: mx.array,
    norm_w: mx.array,
    gu_signs: mx.array,
) -> tuple[mx.array, mx.array]:
    """Fused BlockAttnRes(1) + RMSNorm + RHT-2304 -> `(h_mlp_f16[T, 2304], x_rot_gu_f16[T, 2304])` in ONE launch."""
    T = int(partial.size) // 2304
    b_in = blocks_cat.reshape(-1) if (blocks_cat is not None and n_blocks > 0) else _EMPTY_F32
    return _make_bar_norm_rht_kernel()(
        inputs=[query.reshape(-1), b_in, partial.reshape(-1), norm_w.reshape(-1), gu_signs.reshape(-1), _paley_b_flat(2304), _u32(n_blocks, T)],
        grid=(256, T, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, 2304), (T, 2304)],
        output_dtypes=[mx.float16, mx.float16],
    )


@lru_cache(maxsize=2)
def _make_msg_blend_norm_rht_kernel():
    # Fuses MLP message blend + RMSNorm + float16 cast + RHT-2304(gu_signs) in ONE launch (eliminates 24 RHT launches + 24 casts)!
    src = """
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.y;
    const uint c0 = simd * 9u;
    const device float* hp = h + (ulong)t * 2304u;
    const device float* mp = msg + (ulong)t * 2304u;
    const device bfloat16_t* fgp = fg_mg + (ulong)t * 4608u;
    const device bfloat16_t* mgp = fgp + 2304u;

    float v[9];
    float ss = 0.0f;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {
        const uint idx = lane * 72u + c0 + j;
        const float fg = metal::precise::tanh((float)fgp[idx]);
        const float mg = metal::precise::tanh((float)mgp[idx]);
        v[j] = (1.0f - fg) * hp[idx] + mg * mp[idx];
        ss += v[j] * v[j];
    }
    threadgroup float red[8];
    threadgroup float tg_z[32 * 72];
    ss = simd_sum(ss);
    if (lane == 0u) red[simd] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < 8u; i++) tot += red[i];
    const float ir = metal::precise::rsqrt(tot * (1.0f / 2304.0f) + 1e-6f);

    device half* hop = h_f16_out + (ulong)t * 2304u;
    device half* rop = x_rot_out + (ulong)t * 2304u;
    const bool h1 = lane & 1u, h2 = lane & 2u, h4 = lane & 4u, h8 = lane & 8u, h16 = lane & 16u;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {
        const uint c = c0 + j;
        const uint idx = lane * 72u + c;
        const float hm = v[j] * ir * norm_w[idx];
        hop[idx] = (half)hm;
        float rv = hm * gu_signs[idx];
        float o;
        o = simd_shuffle_xor(rv, 1u);  rv = h1  ? (o - rv) : (rv + o);
        o = simd_shuffle_xor(rv, 2u);  rv = h2  ? (o - rv) : (rv + o);
        o = simd_shuffle_xor(rv, 4u);  rv = h4  ? (o - rv) : (rv + o);
        o = simd_shuffle_xor(rv, 8u);  rv = h8  ? (o - rv) : (rv + o);
        o = simd_shuffle_xor(rv, 16u); rv = h16 ? (o - rv) : (rv + o);
        tg_z[idx] = rv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup float* zr = tg_z + lane * 72u;
    constexpr float INV_S = 1.0f / 48.0f;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {
        const uint c = c0 + j;
        const device float4* br = (const device float4*)(B + c * 72u);
        float racc = 0.0f;
        #pragma unroll
        for (uint k4 = 0; k4 < 18u; k4++) {
            const float4 b4 = br[k4];
            const uint k = k4 * 4u;
            racc += zr[k] * b4.x + zr[k + 1u] * b4.y + zr[k + 2u] * b4.z + zr[k + 3u] * b4.w;
        }
        rop[lane * 72u + c] = (half)(racc * INV_S);
    }
    """
    return mx.fast.metal_kernel(
        name="ink_msg_blend_norm_rht_f16",
        input_names=["h", "msg", "fg_mg", "norm_w", "gu_signs", "B"],
        output_names=["h_f16_out", "x_rot_out"],
        source=src,
    )


def msg_blend_norm_rht_fast(
    h: mx.array,
    msg: mx.array,
    fg_mg_bf16: mx.array,
    norm_w: mx.array,
    gu_signs: mx.array,
) -> tuple[mx.array, mx.array]:
    """Fused MLP MsgBlend + RMSNorm + RHT-2304 -> `(h_mlp_f16[T, 2304], x_rot_gu_f16[T, 2304])` in ONE launch."""
    T = int(h.size) // 2304
    return _make_msg_blend_norm_rht_kernel()(
        inputs=[h.reshape(-1), msg.reshape(-1), fg_mg_bf16.reshape(-1), norm_w.reshape(-1), gu_signs.reshape(-1), _paley_b_flat(2304)],
        grid=(256, T, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, 2304), (T, 2304)],
        output_dtypes=[mx.float16, mx.float16],
    )


@lru_cache(maxsize=2)
def _make_msg_blend_norm_kernel():
    # Fuses: y = (1 - tanh(fg)) * h + tanh(mg) * msg, then RMSNorm(y, norm_w) in ONE launch!
    src = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.y;
    const device float* hp = h + (ulong)t * 2304u;
    const device float* mp = msg + (ulong)t * 2304u;
    const device bfloat16_t* fgp = fg_mg + (ulong)t * 4608u;
    const device bfloat16_t* mgp = fgp + 2304u;
    device float* op = out + (ulong)t * 2304u;

    float v[9];
    float ss = 0.0f;
    #pragma unroll
    for (uint e = 0; e < 9u; e++) {
        const uint ch = tid + e * 256u;
        const float fg = metal::precise::tanh((float)fgp[ch]);
        const float mg = metal::precise::tanh((float)mgp[ch]);
        v[e] = (1.0f - fg) * hp[ch] + mg * mp[ch];
        ss += v[e] * v[e];
    }
    threadgroup float red[8];
    ss = simd_sum(ss);
    if (lane == 0u) red[simd] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < 8u; i++) tot += red[i];
    const float ir = metal::precise::rsqrt(tot * (1.0f / 2304.0f) + 1e-6f);
    #pragma unroll
    for (uint e = 0; e < 9u; e++) {
        const uint ch = tid + e * 256u;
        op[ch] = v[e] * ir * norm_w[ch];
    }
    """
    return mx.fast.metal_kernel(
        name="ink_msg_blend_norm",
        input_names=["h", "msg", "fg_mg", "norm_w"],
        output_names=["out"],
        source=src,
    )


def msg_blend_norm_fused(h: mx.array, msg: mx.array, fg_mg_bf16: mx.array, norm_w: mx.array) -> mx.array:
    T = int(h.size) // 2304
    return _make_msg_blend_norm_kernel()(
        inputs=[h.reshape(-1), msg.reshape(-1), fg_mg_bf16.reshape(-1), norm_w.reshape(-1)],
        grid=(256, T, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, 2304)],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=2)
def _make_msg_blend_norm_3x_kernel():
    # Fuses 3x Q/K/V message blend + RMSNorm into ONE launch outputting `[3, T, 2304]` ready for split_qkv RHT!
    # meta = [T, h_shared (1 for sliding, 0 for global), norm_shared (1 for sliding, 0 for global)]
    src = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint gt = threadgroup_position_in_grid.y; // 0 .. 3*T - 1
    const uint T = meta[0], h_shared = meta[1], nw_shared = meta[2];
    const uint g = gt / T;
    const uint t = gt - g * T;
    const device float* hp = h + ((ulong)(h_shared ? 0u : g) * T + t) * 2304u;
    const device float* mp = msgs + ((ulong)g * T + t) * 2304u;
    const device bfloat16_t* fgp = fg_mg3 + ((ulong)g * T + t) * 4608u;
    const device bfloat16_t* mgp = fgp + 2304u;
    const device float* nwp = norm_w + (ulong)(nw_shared ? 0u : g) * 2304u;
    device float* op = out + ((ulong)g * T + t) * 2304u;

    float v[9];
    float ss = 0.0f;
    #pragma unroll
    for (uint e = 0; e < 9u; e++) {
        const uint ch = tid + e * 256u;
        const float fg = metal::precise::tanh((float)fgp[ch]);
        const float mg = metal::precise::tanh((float)mgp[ch]);
        v[e] = (1.0f - fg) * hp[ch] + mg * mp[ch];
        ss += v[e] * v[e];
    }
    threadgroup float red[8];
    ss = simd_sum(ss);
    if (lane == 0u) red[simd] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < 8u; i++) tot += red[i];
    const float ir = metal::precise::rsqrt(tot * (1.0f / 2304.0f) + 1e-6f);
    #pragma unroll
    for (uint e = 0; e < 9u; e++) {
        const uint ch = tid + e * 256u;
        op[ch] = v[e] * ir * nwp[ch];
    }
    """
    return mx.fast.metal_kernel(
        name="ink_msg_blend_norm_3x",
        input_names=["h", "msgs", "fg_mg3", "norm_w", "meta"],
        output_names=["out"],
        source=src,
    )


def msg_blend_norm_3x_fused(
    h: mx.array,
    msgs_qkv: mx.array,
    fg_mg3_bf16: mx.array,
    norm_w: mx.array,
    T: int,
    h_shared: bool,
    norm_shared: bool,
) -> mx.array:
    """Returns `[3, T, 2304]` float32 for `(q_in, k_in, v_in)` in one launch (zero concat!)."""
    return _make_msg_blend_norm_3x_kernel()(
        inputs=[h.reshape(-1), msgs_qkv.reshape(-1), fg_mg3_bf16.reshape(-1), norm_w.reshape(-1), _u32(T, int(h_shared), int(norm_shared))],
        grid=(256, 3 * T, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(3, T, 2304)],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=4)
def _make_q_rel_clip_kernel(tree_mode: bool = False):
    # Fuses q * q_scale + l2_clip(+log_scale) AND rel * rel_scale + l2_clip(+log_scale) in ONE launch!
    pos_expr = "(q_pos0 + tree_depth[t])" if tree_mode else "(q_pos0 + t)"
    in_names = ["q_raw", "q_scale", "rel_raw", "rel_scale", "meta", "cfg"] + (["tree_depth"] if tree_mode else [])
    src = f"""
    const uint ht = threadgroup_position_in_grid.y;
    const uint t = ht / 24u;
    const uint h = ht - t * 24u;
    const uint tid = thread_position_in_threadgroup.x; // 0..127
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint is_sliding = meta[0];
    const uint q_pos0 = meta[1];
    const float q_thresh = cfg[0], r_thresh = cfg[1], alpha = cfg[2], inv_nfloor = cfg[3];

    float tau = 1.0f;
    if (!is_sliding) {{
        const float ratio = ((float){pos_expr} + 1.0f) * inv_nfloor;
        if (ratio > 1.0f) tau = 1.0f + alpha * metal::precise::log(ratio);
    }}

    // 1. Q [T, 24, 128] (1 thread per dim)
    const ulong q_off = ((ulong)t * 24u + h) * 128u + tid;
    const float qv = q_raw[q_off] * q_scale[h * 128u + tid];
    float ss = simd_sum(qv * qv);
    threadgroup float red[4];
    if (lane == 0u) red[simd] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_norm = metal::precise::sqrt(red[0] + red[1] + red[2] + red[3]);
    float q_mul = min(q_thresh / (q_norm + 1e-6f), 1.0f) * tau;
    q_out[q_off] = qv * q_mul;

    // 2. Rel [T, 24, 16] (first 16 lanes of simd 0!)
    if (tid < 16u) {{
        const ulong r_off = ((ulong)t * 24u + h) * 16u + tid;
        const float rv = rel_raw[r_off] * rel_scale[h * 16u + tid];
        float rs = rv * rv;
        rs += simd_shuffle_xor(rs, 8u); rs += simd_shuffle_xor(rs, 4u); rs += simd_shuffle_xor(rs, 2u); rs += simd_shuffle_xor(rs, 1u);
        const float r_mul = min(r_thresh / (metal::precise::sqrt(rs) + 1e-6f), 1.0f) * tau;
        rel_out[r_off] = rv * r_mul;
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_q_rel_clip_tr{int(tree_mode)}",
        input_names=in_names,
        output_names=["q_out", "rel_out"],
        source=src,
    )


def q_rel_clip_fused(
    q_raw: mx.array,
    q_scale: mx.array,
    rel_raw: mx.array,
    rel_scale: mx.array,
    q_pos0: int,
    T: int,
    is_sliding: bool,
    qk_thresh: float,
    rel_thresh: float,
    alpha: float,
    n_floor: int,
    tree_depth: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    ins = [
        q_raw.reshape(-1), q_scale.reshape(-1), rel_raw.reshape(-1), rel_scale.reshape(-1),
        _u32(int(is_sliding), q_pos0),
        _f32_arr(qk_thresh, rel_thresh, alpha, 1.0 / float(n_floor)),
    ]
    if tree_depth is not None:
        ins.append(tree_depth)
    return _make_q_rel_clip_kernel(tree_depth is not None)(
        inputs=ins,
        grid=(128, 24 * T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(T, 24, 128), (T, 24, 16)],
        output_dtypes=[mx.float32, mx.float32],
    )


@lru_cache(maxsize=4)
def _make_kv_sconv4_norm_kernel(tree_mode: bool = False):
    # Fuses BOTH k and v static 4-tap short_conv + k l2_clip + v unweighted_rms_norm -> unified float16[2, 8, T, 128] + new conv states!
    lag_expr = "tree_lag[t * 4u + k]" if tree_mode else "((int)t + (int)k - 3)"
    in_names = ["k_prev", "k_raw", "k_w", "v_prev", "v_raw", "v_w", "meta", "cfg"] + (["tree_lag"] if tree_mode else [])
    src = f"""
    const uint ht = threadgroup_position_in_grid.y;
    const uint t = ht / 8u;
    const uint h = ht - t * 8u;
    const uint d = thread_position_in_threadgroup.x; // 0..127
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint T = meta[0];
    const uint ch = h * 128u + d;
    const float k_thresh = cfg[0];
    constexpr float INV_SQRT5 = 0.4472135954999579f;
    const ulong v_off = (ulong)8u * T * 128u;

    // 1. K static 4-tap conv + L2Clip
    float kv = k_raw[(ulong)t * 1024u + ch];
    float k_acc = kv;
    for (uint k = 0; k < 4u; k++) {{
        const int idx = {lag_expr};
        const float v = (idx >= 0) ? k_raw[(ulong)idx * 1024u + ch] : k_prev[(ulong)(idx + 3) * 1024u + ch];
        k_acc += v * k_w[(ulong)k * 1024u + ch];
    }}
    k_acc *= INV_SQRT5;
    threadgroup float red_k[4], red_v[4];
    float kss = simd_sum(k_acc * k_acc);
    if (lane == 0u) red_k[simd] = kss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float k_norm = metal::precise::sqrt(red_k[0] + red_k[1] + red_k[2] + red_k[3]);
    kv_out[((ulong)h * T + t) * 128u + d] = (half)(k_acc * min(k_thresh / (k_norm + 1e-6f), 1.0f));

    // 2. V static 4-tap conv + UnweightedRMSNorm
    float vv = v_raw[(ulong)t * 1024u + ch];
    float v_acc = vv;
    for (uint k = 0; k < 4u; k++) {{
        const int idx = {lag_expr};
        const float v = (idx >= 0) ? v_raw[(ulong)idx * 1024u + ch] : v_prev[(ulong)(idx + 3) * 1024u + ch];
        v_acc += v * v_w[(ulong)k * 1024u + ch];
    }}
    v_acc *= INV_SQRT5;
    float vss = simd_sum(v_acc * v_acc);
    if (lane == 0u) red_v[simd] = vss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float v_ir = metal::precise::rsqrt((red_v[0] + red_v[1] + red_v[2] + red_v[3]) * (1.0f / 128.0f) + 1e-6f);
    kv_out[v_off + ((ulong)h * T + t) * 128u + d] = (half)(v_acc * v_ir);

    // 3. Save new 3-step conv states
    if (t == 0u) {{
        for (uint j = 0; j < 3u; j++) {{
            const int idx = (int)T - 3 + (int)j;
            k_new_prev[(ulong)j * 1024u + ch] = (idx >= 0) ? k_raw[(ulong)idx * 1024u + ch] : k_prev[(ulong)(idx + 3) * 1024u + ch];
            v_new_prev[(ulong)j * 1024u + ch] = (idx >= 0) ? v_raw[(ulong)idx * 1024u + ch] : v_prev[(ulong)(idx + 3) * 1024u + ch];
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_kv_sconv4_norm_u2_tr{int(tree_mode)}",
        input_names=in_names,
        output_names=["kv_out", "k_new_prev", "v_new_prev"],
        source=src,
    )


def kv_sconv4_norm_fused(
    k_prev: mx.array,
    k_raw: mx.array,
    k_w: mx.array,
    v_prev: mx.array,
    v_raw: mx.array,
    v_w: mx.array,
    T: int,
    qk_thresh: float,
    tree_lag: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    ins = [k_prev.reshape(-1), k_raw.reshape(-1), k_w.reshape(-1), v_prev.reshape(-1), v_raw.reshape(-1), v_w.reshape(-1), _u32(T), _f32_arr(qk_thresh)]
    if tree_lag is not None:
        ins.append(tree_lag)
    return _make_kv_sconv4_norm_kernel(tree_lag is not None)(
        inputs=ins,
        grid=(128, 8 * T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(2, 8, T, 128), (1, 3, 1024), (1, 3, 1024)],
        output_dtypes=[mx.float16, mx.float32, mx.float32],
    )


@lru_cache(maxsize=4)
def _make_qkv_rel_prep_sliding_kernel(tree_mode: bool = False):
    # Single launch for ALL 40 sliding visits: Q+Rel clip AND K+V static 4-tap conv + norm -> unified kv_out[2, 8, T, 128]!
    lag_expr = "tree_lag[t * 4u + k]" if tree_mode else "((int)t + (int)k - 3)"
    in_names = ["y_qkv", "q_scale", "rel_scale", "k_prev", "k_w", "v_prev", "v_w", "meta", "cfg"] + (["tree_lag"] if tree_mode else [])
    src = f"""
    const uint ht = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x; // 0..127
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint T = meta[0];
    const float qk_thresh = cfg[0], r_thresh = cfg[1];
    threadgroup float red[4], red_v[4];

    if (ht < 24u * T) {{
        const uint t = ht / 24u;
        const uint h = ht - t * 24u;
        const device float* row = y_qkv + (ulong)t * 8576u;
        const float qv = row[h * 128u + tid] * q_scale[h * 128u + tid];
        float ss = simd_sum(qv * qv);
        if (lane == 0u) red[simd] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float q_norm = metal::precise::sqrt(red[0] + red[1] + red[2] + red[3]);
        q_out[((ulong)t * 24u + h) * 128u + tid] = qv * min(qk_thresh / (q_norm + 1e-6f), 1.0f);

        if (tid < 16u) {{
            const float rv = row[5120u + h * 16u + tid] * rel_scale[h * 16u + tid];
            float rs = rv * rv;
            rs += simd_shuffle_xor(rs, 8u); rs += simd_shuffle_xor(rs, 4u); rs += simd_shuffle_xor(rs, 2u); rs += simd_shuffle_xor(rs, 1u);
            rel_out[((ulong)t * 24u + h) * 16u + tid] = rv * min(r_thresh / (metal::precise::sqrt(rs) + 1e-6f), 1.0f);
        }}
    }} else {{
        const uint ht2 = ht - 24u * T;
        const uint t = ht2 / 8u;
        const uint h = ht2 - t * 8u;
        const uint ch = h * 128u + tid;
        constexpr float INV_SQRT5 = 0.4472135954999579f;
        const ulong v_off = (ulong)8u * T * 128u;

        // K static 4-tap conv + L2Clip (k_raw sits at +3072 in y_qkv[t, 8576])
        float k_acc = y_qkv[(ulong)t * 8576u + 3072u + ch];
        for (uint k = 0; k < 4u; k++) {{
            const int idx = {lag_expr};
            const float v = (idx >= 0) ? y_qkv[(ulong)idx * 8576u + 3072u + ch] : k_prev[(ulong)(idx + 3) * 1024u + ch];
            k_acc += v * k_w[(ulong)k * 1024u + ch];
        }}
        k_acc *= INV_SQRT5;
        float kss = simd_sum(k_acc * k_acc);
        if (lane == 0u) red[simd] = kss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float k_norm = metal::precise::sqrt(red[0] + red[1] + red[2] + red[3]);
        kv_out[((ulong)h * T + t) * 128u + tid] = (half)(k_acc * min(qk_thresh / (k_norm + 1e-6f), 1.0f));

        // V static 4-tap conv + UnweightedRMSNorm (v_raw sits at +4096 in y_qkv[t, 8576])
        float v_acc = y_qkv[(ulong)t * 8576u + 4096u + ch];
        for (uint k = 0; k < 4u; k++) {{
            const int idx = {lag_expr};
            const float v = (idx >= 0) ? y_qkv[(ulong)idx * 8576u + 4096u + ch] : v_prev[(ulong)(idx + 3) * 1024u + ch];
            v_acc += v * v_w[(ulong)k * 1024u + ch];
        }}
        v_acc *= INV_SQRT5;
        float vss = simd_sum(v_acc * v_acc);
        if (lane == 0u) red_v[simd] = vss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float v_ir = metal::precise::rsqrt((red_v[0] + red_v[1] + red_v[2] + red_v[3]) * (1.0f / 128.0f) + 1e-6f);
        kv_out[v_off + ((ulong)h * T + t) * 128u + tid] = (half)(v_acc * v_ir);

        if (t == 0u) {{
            for (uint j = 0; j < 3u; j++) {{
                const int idx = (int)T - 3 + (int)j;
                k_new_prev[(ulong)j * 1024u + ch] = (idx >= 0) ? y_qkv[(ulong)idx * 8576u + 3072u + ch] : k_prev[(ulong)(idx + 3) * 1024u + ch];
                v_new_prev[(ulong)j * 1024u + ch] = (idx >= 0) ? y_qkv[(ulong)idx * 8576u + 4096u + ch] : v_prev[(ulong)(idx + 3) * 1024u + ch];
            }}
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_qkv_rel_prep_sl_u2_tr{int(tree_mode)}",
        input_names=in_names,
        output_names=["q_out", "rel_out", "kv_out", "k_new_prev", "v_new_prev"],
        source=src,
    )


def qkv_rel_prep_sliding_fused(
    y_qkv: mx.array,
    q_scale: mx.array,
    rel_scale: mx.array,
    k_prev: mx.array,
    k_w: mx.array,
    v_prev: mx.array,
    v_w: mx.array,
    T: int,
    qk_thresh: float,
    rel_thresh: float,
    tree_lag: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    ins = [
        y_qkv.reshape(-1), q_scale.reshape(-1), rel_scale.reshape(-1),
        k_prev.reshape(-1), k_w.reshape(-1), v_prev.reshape(-1), v_w.reshape(-1),
        _u32(T), _f32_arr(qk_thresh, rel_thresh),
    ]
    if tree_lag is not None:
        ins.append(tree_lag)
    return _make_qkv_rel_prep_sliding_kernel(tree_lag is not None)(
        inputs=ins,
        grid=(128, 32 * T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(T, 24, 128), (T, 24, 16), (2, 8, T, 128), (1, 3, 1024), (1, 3, 1024)],
        output_dtypes=[mx.float32, mx.float32, mx.float16, mx.float32, mx.float32],
    )


@lru_cache(maxsize=4)
def _make_kv_dyn_sconv4_norm_kernel(tree_mode: bool = False):
    # Single launch for ALL 8 global visits: dynamic weight proj + 4-tap conv + K L2Clip + V RMSNorm -> unified kv_out[2, 8, T, 128]!
    lag_expr = "tree_lag[t * 4u + (3u - lag)]" if tree_mode else "((int)t - (int)lag)"
    in_names = ["k_prev", "k_raw", "k_in", "k_st", "k_pr", "v_prev", "v_raw", "v_in", "v_st", "v_pr", "meta", "cfg"] + (["tree_lag"] if tree_mode else [])
    src = f"""
    const uint ht = threadgroup_position_in_grid.y;
    const uint t = ht / 8u;
    const uint h = ht - t * 8u;
    const uint d = thread_position_in_threadgroup.x; // 0..127
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint T = meta[0];
    const uint ch = h * 128u + d;
    const float k_thresh = cfg[0];
    constexpr float INV_3 = 0.3333333333333333f;
    const ulong v_off = (ulong)8u * T * 128u;

    // 1. Compute 4 dynamic tap weights for K and 4 for V via dot(in[t], proj[h*4 + lag])
    const device float2* kin2 = (const device float2*)(k_in + (ulong)t * 2304u);
    const device float2* vin2 = (const device float2*)(v_in + (ulong)t * 2304u);
    float2 ki_reg[9], vi_reg[9];
    #pragma unroll
    for (uint i = 0; i < 9u; i++) {{
        const uint idx = d + i * 128u;
        ki_reg[i] = kin2[idx];
        vi_reg[i] = vin2[idx];
    }}
    threadgroup float tg_dk[4][4], tg_dv[4][4];
    #pragma unroll
    for (uint lag = 0; lag < 4u; lag++) {{
        const device half2* kpw = (const device half2*)(k_pr + (ulong)(h * 4u + lag) * 2304u);
        const device half2* vpw = (const device half2*)(v_pr + (ulong)(h * 4u + lag) * 2304u);
        float kd = 0.0f, vd = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 9u; i++) {{
            const uint idx = d + i * 128u;
            kd += dot(ki_reg[i], float2(kpw[idx]));
            vd += dot(vi_reg[i], float2(vpw[idx]));
        }}
        kd = simd_sum(kd);
        vd = simd_sum(vd);
        if (lane == 0u) {{ tg_dk[lag][simd] = kd; tg_dv[lag][simd] = vd; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float dk[4], dv[4];
    #pragma unroll
    for (uint lag = 0; lag < 4u; lag++) {{
        dk[lag] = tg_dk[lag][0] + tg_dk[lag][1] + tg_dk[lag][2] + tg_dk[lag][3];
        dv[lag] = tg_dv[lag][0] + tg_dv[lag][1] + tg_dv[lag][2] + tg_dv[lag][3];
    }}

    // 2. K dynamic 4-tap conv (lag 0 = newest t, lag 3 = oldest t-3) + L2Clip
    float k_acc = k_raw[(ulong)t * 1024u + ch];
    #pragma unroll
    for (uint lag = 0; lag < 4u; lag++) {{
        const int idx = {lag_expr};
        const float v = (idx >= 0) ? k_raw[(ulong)idx * 1024u + ch] : k_prev[(ulong)(idx + 3) * 1024u + ch];
        k_acc += v * (k_st[(ulong)lag * 1024u + ch] + dk[lag]);
    }}
    k_acc *= INV_3;
    threadgroup float red_k[4], red_v[4];
    float kss = simd_sum(k_acc * k_acc);
    if (lane == 0u) red_k[simd] = kss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float k_norm = metal::precise::sqrt(red_k[0] + red_k[1] + red_k[2] + red_k[3]);
    kv_out[((ulong)h * T + t) * 128u + d] = (half)(k_acc * min(k_thresh / (k_norm + 1e-6f), 1.0f));

    // 3. V dynamic 4-tap conv + UnweightedRMSNorm
    float v_acc = v_raw[(ulong)t * 1024u + ch];
    #pragma unroll
    for (uint lag = 0; lag < 4u; lag++) {{
        const int idx = {lag_expr};
        const float v = (idx >= 0) ? v_raw[(ulong)idx * 1024u + ch] : v_prev[(ulong)(idx + 3) * 1024u + ch];
        v_acc += v * (v_st[(ulong)lag * 1024u + ch] + dv[lag]);
    }}
    v_acc *= INV_3;
    float vss = simd_sum(v_acc * v_acc);
    if (lane == 0u) red_v[simd] = vss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float v_ir = metal::precise::rsqrt((red_v[0] + red_v[1] + red_v[2] + red_v[3]) * (1.0f / 128.0f) + 1e-6f);
    kv_out[v_off + ((ulong)h * T + t) * 128u + d] = (half)(v_acc * v_ir);

    // 4. Save new 3-step conv states
    if (t == 0u) {{
        for (uint j = 0; j < 3u; j++) {{
            const int idx = (int)T - 3 + (int)j;
            k_new_prev[(ulong)j * 1024u + ch] = (idx >= 0) ? k_raw[(ulong)idx * 1024u + ch] : k_prev[(ulong)(idx + 3) * 1024u + ch];
            v_new_prev[(ulong)j * 1024u + ch] = (idx >= 0) ? v_raw[(ulong)idx * 1024u + ch] : v_prev[(ulong)(idx + 3) * 1024u + ch];
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_kv_dyn_sconv4_u2_tr{int(tree_mode)}",
        input_names=in_names,
        output_names=["kv_out", "k_new_prev", "v_new_prev"],
        source=src,
    )


def kv_dyn_sconv4_norm_fused(
    k_prev: mx.array,
    k_raw: mx.array,
    k_in: mx.array,
    k_st: mx.array,
    k_pr_f16: mx.array,
    v_prev: mx.array,
    v_raw: mx.array,
    v_in: mx.array,
    v_st: mx.array,
    v_pr_f16: mx.array,
    T: int,
    qk_thresh: float,
    tree_lag: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    ins = [
        k_prev.reshape(-1), k_raw.reshape(-1), k_in.reshape(-1), k_st.reshape(-1), k_pr_f16.reshape(-1),
        v_prev.reshape(-1), v_raw.reshape(-1), v_in.reshape(-1), v_st.reshape(-1), v_pr_f16.reshape(-1),
        _u32(T), _f32_arr(qk_thresh),
    ]
    if tree_lag is not None:
        ins.append(tree_lag)
    return _make_kv_dyn_sconv4_norm_kernel(tree_lag is not None)(
        inputs=ins,
        grid=(128, 8 * T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(2, 8, T, 128), (1, 3, 1024), (1, 3, 1024)],
        output_dtypes=[mx.float16, mx.float32, mx.float32],
    )


def _build_fp8_e4m3_lut_header() -> str:
    vals = []
    for c in range(256):
        if (c & 0x7F) == 0x7F:
            v = 0.0
        else:
            s = -1.0 if (c & 0x80) else 1.0
            e = (c >> 3) & 0xF
            m = c & 0x7
            v = s * ((m / 8.0) * (2.0 ** -6) if e == 0 else (1.0 + m / 8.0) * (2.0 ** (e - 7)))
        vals.append(f"half({v:.8e}f)")
    return (
        "constant half FP8_E4M3_LUT[256] = {"
        + ",".join(vals)
        + "};\n"
        + """
inline uchar float_to_fp8_e4m3(float x) {
    uint u = as_type<uint>(x);
    uint sign = (u >> 24) & 0x80u;
    float ax = metal::fabs(x);
    if (!(ax < 448.0f)) {
        return (ax >= 448.0f) ? uchar(sign | 0x7Eu) : uchar(0u);
    }
    if (ax < 0.015625f) {
        uint q = uint(metal::rint(ax * 512.0f));
        return uchar(sign | q);
    }
    uint ua = as_type<uint>(ax);
    uint lsb = (ua >> 20) & 1u;
    uint rounded = ua + 0x7FFFFu + lsb;
    uint e = (rounded >> 23) - 120u;
    uint m = (rounded >> 20) & 0x7u;
    uint code = (e << 3) | m;
    if (code > 0x7Eu) code = 0x7Eu;
    return uchar(sign | code);
}
"""
    )


_FP8_E4M3_HEADER = _build_fp8_e4m3_lut_header()


@lru_cache(maxsize=1)
def _make_fp8_quantize_f16_kernel():
    src = """
    const uint idx = thread_position_in_grid.x;
    const uint n4 = meta[0];
    if (idx >= n4) return;
    const half4 v = *((const device half4*)(x) + idx);
    uchar4 o;
    o.x = float_to_fp8_e4m3(float(v.x));
    o.y = float_to_fp8_e4m3(float(v.y));
    o.z = float_to_fp8_e4m3(float(v.z));
    o.w = float_to_fp8_e4m3(float(v.w));
    *((device uchar4*)(out) + idx) = o;
    """
    return mx.fast.metal_kernel(
        name="ink_fp8_e4m3_quantize_f16",
        input_names=["x", "meta"],
        output_names=["out"],
        header=_FP8_E4M3_HEADER,
        source=src,
    )


_H128_HEADER = """
inline float4 h128_vec(float4 v, uint lane) {
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
    return v * 0.08838834764831845f;
}
"""


@lru_cache(maxsize=1)
def _make_h128_rows_f32_kernel():
    src = """
    const uint row = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const float4 v = *((const device float4*)(x + (ulong)row * 128u) + lane);
    *((device float4*)(out + (ulong)row * 128u) + lane) = h128_vec(v, lane);
    """
    return mx.fast.metal_kernel(
        name="ink_h128_rows_f32",
        input_names=["x"],
        output_names=["out"],
        header=_H128_HEADER,
        source=src,
    )


def h128_rows_f32(x: mx.array) -> mx.array:
    xc = mx.contiguous(x.astype(mx.float32))
    assert xc.shape[-1] == 128
    n_rows = xc.size // 128
    return _make_h128_rows_f32_kernel()(
        inputs=[xc.reshape(-1)],
        grid=(32, n_rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[xc.shape],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=1)
def _make_kv_h128_q4_quantize_kernel():
    src = """
    const uint row = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const float4 v = float4(*((const device half4*)(x + (ulong)row * 128u) + lane));
    const float4 hv = h128_vec(v, lane);
    float am = max(max(fabs(hv.x), fabs(hv.y)), max(fabs(hv.z), fabs(hv.w)));
    am = max(am, simd_shuffle_xor(am, 1u));
    am = max(am, simd_shuffle_xor(am, 2u));
    am = max(am, simd_shuffle_xor(am, 4u));
    const half sc_h = half(max(am * (1.0f / 7.0f), 1e-6f));
    const float inv_sc = 1.0f / float(sc_h);
    const int q0 = clamp(int(metal::rint(hv.x * inv_sc)), -7, 7) + 8;
    const int q1 = clamp(int(metal::rint(hv.y * inv_sc)), -7, 7) + 8;
    const int q2 = clamp(int(metal::rint(hv.z * inv_sc)), -7, 7) + 8;
    const int q3 = clamp(int(metal::rint(hv.w * inv_sc)), -7, 7) + 8;
    *((device uchar2*)(q_out + (ulong)row * 64u) + lane) = uchar2(uchar(q0 | (q1 << 4)), uchar(q2 | (q3 << 4)));
    if ((lane & 7u) == 0u) {
        sc_out[(ulong)row * 4u + (lane >> 3)] = sc_h;
    }
    """
    return mx.fast.metal_kernel(
        name="ink_kv_h128_q4_quantize",
        input_names=["x"],
        output_names=["q_out", "sc_out"],
        header=_H128_HEADER,
        source=src,
    )


def kv_h128_q4_quantize(kv_f16: mx.array) -> tuple[mx.array, mx.array]:
    xc = mx.contiguous(kv_f16.astype(mx.float16))
    assert xc.shape[-1] == 128
    n_rows = xc.size // 128
    prefix = xc.shape[:-1]
    return _make_kv_h128_q4_quantize_kernel()(
        inputs=[xc.reshape(-1)],
        grid=(32, n_rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*prefix, 64), (*prefix, 4)],
        output_dtypes=[mx.uint8, mx.float16],
    )


def fp8_e4m3_quantize_f16_to_u8(x: mx.array) -> mx.array:
    xc = mx.contiguous(x.astype(mx.float16))
    n = xc.size
    assert n % 4 == 0, f"expected size divisible by 4, got {n}"
    n4 = n // 4
    return _make_fp8_quantize_f16_kernel()(
        inputs=[xc.reshape(-1), _u32(n4)],
        grid=(((n4 + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[xc.shape],
        output_dtypes=[mx.uint8],
    )[0]


@lru_cache(maxsize=8)
def _make_gqa_relpos_sdpa_kernel(tree_mode: bool = False, kv_fp8: bool = False):
    # Fuses sink_proj directly inside SDPA threadgroup; reads unified KV[2, 8, kv_cap, 128] + optional tree mask/depth!
    in_names = ["q", "KV", "rel_st", "rel_proj_t", "q_in", "sink_w", "meta"] + (["tree_depth", "tree_mask"] if tree_mode else [])
    if tree_mode:
        qpos_decl = "const uint q_pos = meta[3] + tree_depth[t]; const uint kv_base = kv_len - T; const uint my_mask = tree_mask[t];"
        jhi_decl = "const uint j_hi = kv_len;"
        dist_block = """
        uint dist;
        if (j >= kv_base) {
            const uint u = j - kv_base;
            if (((my_mask >> u) & 1u) == 0u) continue;
            dist = tree_depth[t] - tree_depth[u];
        } else {
            dist = q_pos - (kv_start + j);
        }
        """
    else:
        qpos_decl = "const uint q_pos = meta[3] + t;"
        jhi_decl = "const uint j_hi = min(kv_len, (q_pos >= kv_start) ? (q_pos - kv_start + 1u) : 0u);"
        dist_block = "const uint dist = q_pos - (kv_start + j);"

    kv_t = "uchar" if kv_fp8 else "half"
    k_read = (
        "const uchar4 ku = *((const device uchar4*)(Kp + (ulong)j * 128u) + lane); "
        "const float4 kf = float4(half4(FP8_E4M3_LUT[ku.x], FP8_E4M3_LUT[ku.y], FP8_E4M3_LUT[ku.z], FP8_E4M3_LUT[ku.w]));"
        if kv_fp8
        else "const device half4* kp = (const device half4*)(Kp + (ulong)j * 128u); const float4 kf = float4(kp[lane]);"
    )
    v_read = (
        "const uchar4 vu = *((const device uchar4*)(Vp + (ulong)j * 128u) + lane); "
        "const float4 vf = float4(half4(FP8_E4M3_LUT[vu.x], FP8_E4M3_LUT[vu.y], FP8_E4M3_LUT[vu.z], FP8_E4M3_LUT[vu.w]));"
        if kv_fp8
        else "const device half4* vp = (const device half4*)(Vp + (ulong)j * 128u); const float4 vf = float4(vp[lane]);"
    )
    src = f"""
    const uint ht = threadgroup_position_in_grid.y;
    const uint t = ht / 24u;
    const uint h = ht - t * 24u;
    const uint kvh = h / 3u;
    const uint simd = simdgroup_index_in_threadgroup; // 0..3
    const uint lane = thread_index_in_simdgroup;      // 0..31 -> 4 dims each
    const uint tid = simd * 32u + lane;               // 0..127

    const uint T = meta[0];
    const uint kv_len = meta[1];
    const uint kv_start = meta[2];
    {qpos_decl}
    const uint kv_cap = meta[4];
    const uint is_sliding = meta[5];

    // Inline sink logit dot(q_in[t, 2304], sink_w_f16[h, 2304]): 128 threads x 9 float2 = 2304!
    const device float2* qin2 = (const device float2*)(q_in + (ulong)t * 2304u);
    const device half2* sw2 = (const device half2*)(sink_w + (ulong)h * 2304u);
    float sk_part = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 9u; i++) {{
        const uint idx = tid + i * 128u;
        sk_part += dot(qin2[idx], float2(sw2[idx]));
    }}
    sk_part = simd_sum(sk_part);

    const device float4* qp = (const device float4*)(q + ((ulong)t * 24u + h) * 128u);
    const float4 qv = qp[lane] * (1.0f / 11.313708498984761f);

    const device float4* rsp = (const device float4*)(rel_st + ((ulong)t * 24u + h) * 16u);
    const float4 rs0 = rsp[0], rs1 = rsp[1], rs2 = rsp[2], rs3 = rsp[3];

    const device {kv_t}* Kp = KV + (ulong)kvh * kv_cap * 128u;
    const device {kv_t}* Vp = KV + (ulong)(8u + kvh) * kv_cap * 128u;

    const int min_pos = is_sliding ? ((int)q_pos - 1023) : 0;
    const uint j_lo = (min_pos > (int)kv_start) ? (uint)(min_pos - (int)kv_start) : 0u;
    {jhi_decl}

    float m_loc = -1e30f;
    float l_loc = 0.0f;
    float4 acc = float4(0.0f);

    for (uint j = j_lo + simd; j < j_hi; j += 4u) {{
        {dist_block}
        {k_read}
        float score = simd_sum(dot(qv, kf));
        if (dist < 1024u) {{
            const device half4* rp = (const device half4*)(rel_proj_t + (ulong)dist * 16u);
            score += dot(rs0, float4(rp[0])) + dot(rs1, float4(rp[1])) + dot(rs2, float4(rp[2])) + dot(rs3, float4(rp[3]));
        }}
        const float m_new = max(m_loc, score);
        const float alpha = metal::precise::exp(m_loc - m_new);
        const float p = metal::precise::exp(score - m_new);
        l_loc = l_loc * alpha + p;
        {v_read}
        acc = acc * alpha + p * vf;
        m_loc = m_new;
    }}

    threadgroup float tg_m[4], tg_l[4], tg_sk[4];
    threadgroup float4 tg_a[4][32];
    if (lane == 0u) {{ tg_m[simd] = m_loc; tg_l[simd] = l_loc; tg_sk[simd] = sk_part; }}
    tg_a[simd][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd == 0u) {{
        const float sk = tg_sk[0] + tg_sk[1] + tg_sk[2] + tg_sk[3];
        float m_all = sk;
        for (uint s = 0; s < 4u; s++) m_all = max(m_all, tg_m[s]);
        float l_all = metal::precise::exp(sk - m_all);
        float4 a_all = float4(0.0f);
        for (uint s = 0; s < 4u; s++) {{
            const float scale = metal::precise::exp(tg_m[s] - m_all);
            l_all += tg_l[s] * scale;
            a_all += tg_a[s][lane] * scale;
        }}
        device float4* op = (device float4*)(out + ((ulong)t * 24u + h) * 128u);
        op[lane] = a_all * (1.0f / l_all);
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_gqa_relpos_sdpa_u2_tr{int(tree_mode)}_fp8{int(kv_fp8)}",
        input_names=in_names,
        output_names=["out"],
        header=_FP8_E4M3_HEADER if kv_fp8 else "",
        source=src,
    )


@lru_cache(maxsize=6)
def _make_gqa_relpos_flash_sdpa_kernel(kv_fp8: bool = False, kv_q4: bool = False):
    """Tiled `simdgroup_half8x8` FlashAttention-2 (`BQ=32, BK=64, SG=4`, `exp2`, O(1) memory) for prefill."""
    assert not (kv_fp8 and kv_q4)
    header = (
        """
    #include <metal_simdgroup_matrix>
    #define ELEMS_H2 reinterpret_cast<thread half2&>
    #define ELEMS_F2 reinterpret_cast<thread float2&>
    """
        + (_FP8_E4M3_HEADER if kv_fp8 else "")
    )
    in_names = ["q", "KV"] + (["KV_sc"] if kv_q4 else []) + ["rel_st", "rel_proj_t", "sinks", "meta"]
    kv_stride = 64 if kv_q4 else 128
    sc_decls = (
        f"const device half* Ksp = KV_sc + (ulong)kvh * kv_cap * 4u;\n"
        f"    const device half* Vsp = KV_sc + (ulong)(8u + kvh) * kv_cap * 4u;"
        if kv_q4
        else ""
    )
    src = f"""
    const uint bh = threadgroup_position_in_grid.y;
    const uint n_qblocks = meta[6];
    const uint h = bh / n_qblocks;
    const uint qb = bh - h * n_qblocks;
    const uint q0 = qb * 32u;
    const uint kvh = h / 3u;

    const uint simd = simdgroup_index_in_threadgroup; // 0..3
    const uint lane = thread_index_in_simdgroup;      // 0..31
    const uint tid = simd * 32u + lane;               // 0..127

    const uint T = meta[0];
    const uint kv_len = meta[1];
    const uint kv_start = meta[2];
    const uint q_pos0 = meta[3];
    const uint kv_cap = meta[4];
    const uint is_sliding = meta[5];

    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane / 2u) % 4u);
    const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
    const uint q_t = q0 + simd * 8u + fm;
    const uint q_pos = q_pos0 + q_t;

    threadgroup half tg_kv[64 * 136];
    threadgroup half tg_rp[96 * 16];

    // Pre-scale Q by LOG2E / sqrt(128) for 1-cycle hardware SFU exp2()
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

    const device __KV_T__* Kp = KV + (ulong)kvh * kv_cap * {kv_stride}u;
    const device __KV_T__* Vp = KV + (ulong)(8u + kvh) * kv_cap * {kv_stride}u;
    {sc_decls}

    const uint q_pos_min = q_pos0 + q0;
    const uint q_pos_max = q_pos0 + min(q0 + 31u, T - 1u);
    const int min_pos = is_sliding ? ((int)q_pos_min - 1023) : 0;
    const uint k_lo = (min_pos > (int)kv_start) ? (uint)(min_pos - (int)kv_start) : 0u;
    const uint k_hi = min(kv_len, (q_pos_max >= kv_start) ? (q_pos_max - kv_start + 1u) : 0u);

    for (uint k0 = k_lo; k0 < k_hi; k0 += 64u) {{
        #pragma unroll
        for (uint i = 0; i < 16u; i++) {{
            const uint idx = tid + i * 128u;
            const uint kt_l = idx >> 5, d4 = idx & 31u;
            const uint kj = k0 + kt_l;
            __LOAD_K4__
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

        const bool full_valid = (q0 + 32u <= T) && (k0 + 64u <= k_hi) && (d_min >= 0) && (!is_sliding || ((int)q_pos_max - (int)(kv_start + k0) < 1024));
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
                if ((q_t < T) && (k_a < k_hi) && (da >= 0) && (!is_sliding || da < 1024)) {{
                    if (need_rel && da < 1024) {{
                        const threadgroup half4* rp = (const threadgroup half4*)(tg_rp + (uint)(da - d_min) * 16u);
                        s2.x += dot(rs0, float4(rp[0])) + dot(rs1, float4(rp[1])) + dot(rs2, float4(rp[2])) + dot(rs3, float4(rp[3]));
                    }}
                }} else s2.x = -1e30f;
                if ((q_t < T) && (k_b < k_hi) && (db >= 0) && (!is_sliding || db < 1024)) {{
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
        #pragma unroll
        for (uint i = 0; i < 16u; i++) {{
            const uint idx = tid + i * 128u;
            const uint kt_l = idx >> 5, d4 = idx & 31u;
            const uint kj = k0 + kt_l;
            __LOAD_V4__
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
    kv_t = "uchar" if (kv_fp8 or kv_q4) else "half"
    if kv_q4:
        load_k4 = "{ half4 hk = half4(0.0h); if (kj < k_hi) { const uchar2 p = *(const device uchar2*)(Kp + (ulong)kj * 64u + d4 * 2u); const half sc = Ksp[(ulong)kj * 4u + (d4 >> 3)]; hk = half4(half(int(p.x & 0xFu) - 8) * sc, half(int(p.x >> 4) - 8) * sc, half(int(p.y & 0xFu) - 8) * sc, half(int(p.y >> 4) - 8) * sc); } *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = hk; }"
        load_v4 = "{ half4 hv = half4(0.0h); if (kj < k_hi) { const uchar2 p = *(const device uchar2*)(Vp + (ulong)kj * 64u + d4 * 2u); const half sc = Vsp[(ulong)kj * 4u + (d4 >> 3)]; hv = half4(half(int(p.x & 0xFu) - 8) * sc, half(int(p.x >> 4) - 8) * sc, half(int(p.y & 0xFu) - 8) * sc, half(int(p.y >> 4) - 8) * sc); } *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = hv; }"
    elif kv_fp8:
        load_k4 = "{ half4 hk = half4(0.0h); if (kj < k_hi) { const uchar4 u = *(const device uchar4*)(Kp + (ulong)kj * 128u + d4 * 4u); hk = half4(FP8_E4M3_LUT[u.x], FP8_E4M3_LUT[u.y], FP8_E4M3_LUT[u.z], FP8_E4M3_LUT[u.w]); } *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = hk; }"
        load_v4 = "{ half4 hv = half4(0.0h); if (kj < k_hi) { const uchar4 u = *(const device uchar4*)(Vp + (ulong)kj * 128u + d4 * 4u); hv = half4(FP8_E4M3_LUT[u.x], FP8_E4M3_LUT[u.y], FP8_E4M3_LUT[u.z], FP8_E4M3_LUT[u.w]); } *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = hv; }"
    else:
        load_k4 = "*(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = (kj < k_hi) ? *(const device half4*)(Kp + (ulong)kj * 128u + d4 * 4u) : half4(0.0h);"
        load_v4 = "*(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = (kj < k_hi) ? *(const device half4*)(Vp + (ulong)kj * 128u + d4 * 4u) : half4(0.0h);"
    src = src.replace("__KV_T__", kv_t).replace("__LOAD_K4__", load_k4).replace("__LOAD_V4__", load_v4)
    return mx.fast.metal_kernel(
        name=f"ink_gqa_relpos_flash_sdpa_bk64_fp8{int(kv_fp8)}_q4{int(kv_q4)}",
        input_names=in_names,
        output_names=["out"],
        header=header,
        source=src,
    )


@lru_cache(maxsize=48)
def _make_gqa_fused_sdpa_kernel(tree_mode: bool = False, kv_mode: str = "fp16", t_cap: int = 4, split_k: bool = False):
    """GQA+T fused Flash-Decoding kernel supporting fp16, fp8, and Hadamard-rotated 4-bit (q4)."""
    assert kv_mode in ("fp16", "fp8", "q4")
    assert t_cap in (1, 4, 8)
    in_names = ["q", "KV"] + (["KV_sc"] if kv_mode == "q4" else []) + ["rel_st", "rel_proj_t"]
    if not split_k:
        in_names.extend(["q_in", "sink_w"])
    in_names.append("meta")
    if tree_mode:
        in_names.extend(["tree_depth", "tree_mask"])

    kv_t = "uchar" if kv_mode in ("fp8", "q4") else "half"
    kv_stride = 64 if kv_mode == "q4" else 128
    sc_decls = (
        "const device half* Ksp = KV_sc + (ulong)kvh * kv_cap * 4u;\n"
        "    const device half* Vsp = KV_sc + (ulong)(8u + kvh) * kv_cap * 4u;"
        if kv_mode == "q4"
        else ""
    )
    if kv_mode == "q4":
        def load_kv_expr(j_expr: str) -> str:
            return (
                f"const uchar2 kp2 = *((const device uchar2*)(Kp + (ulong)({j_expr}) * 64u) + lane); "
                f"const half ksc = Ksp[(ulong)({j_expr}) * 4u + (lane >> 3)]; "
                f"const uchar2 vp2 = *((const device uchar2*)(Vp + (ulong)({j_expr}) * 64u) + lane); "
                f"const half vsc = Vsp[(ulong)({j_expr}) * 4u + (lane >> 3)]; "
                "tg_k[r * 32u + lane] = half4(half(int(kp2.x & 0xFu) - 8) * ksc, half(int(kp2.x >> 4) - 8) * ksc, half(int(kp2.y & 0xFu) - 8) * ksc, half(int(kp2.y >> 4) - 8) * ksc); "
                "tg_v[r * 32u + lane] = half4(half(int(vp2.x & 0xFu) - 8) * vsc, half(int(vp2.x >> 4) - 8) * vsc, half(int(vp2.y & 0xFu) - 8) * vsc, half(int(vp2.y >> 4) - 8) * vsc);"
            )
    elif kv_mode == "fp8":
        def load_kv_expr(j_expr: str) -> str:
            return (
                f"const uchar4 ku = *((const device uchar4*)(Kp + (ulong)({j_expr}) * 128u) + lane); "
                f"const uchar4 vu = *((const device uchar4*)(Vp + (ulong)({j_expr}) * 128u) + lane); "
                "tg_k[r * 32u + lane] = half4(FP8_E4M3_LUT[ku.x], FP8_E4M3_LUT[ku.y], FP8_E4M3_LUT[ku.z], FP8_E4M3_LUT[ku.w]); "
                "tg_v[r * 32u + lane] = half4(FP8_E4M3_LUT[vu.x], FP8_E4M3_LUT[vu.y], FP8_E4M3_LUT[vu.z], FP8_E4M3_LUT[vu.w]);"
            )
    else:
        def load_kv_expr(j_expr: str) -> str:
            return (
                f"tg_k[r * 32u + lane] = *((const device half4*)(Kp + (ulong)({j_expr}) * 128u) + lane); "
                f"tg_v[r * 32u + lane] = *((const device half4*)(Vp + (ulong)({j_expr}) * 128u) + lane);"
            )

    header = "#define LOG2E 1.4426950408889634f\n"
    if kv_mode == "fp8":
        header += _FP8_E4M3_HEADER
    elif kv_mode == "q4":
        header += _H128_HEADER
    out_names = ["part_m", "part_l", "part_o"] if split_k else ["out"]
    q_rot = "qv[g] = h128_vec(qv[g], lane);" if kv_mode == "q4" else ""
    q_rot_k = "qv[k_t][g] = h128_vec(qv[k_t][g], lane);" if kv_mode == "q4" else ""
    unrot_expr = "h128_vec(a_out, lane)" if kv_mode == "q4" else "a_out"

    if t_cap == 1:
        if split_k:
            sink_merge = """
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
            if (lane == 0u) {
                part_m[(ulong)split_id * 24u + h] = m_all;
                part_l[(ulong)split_id * 24u + h] = l_all;
            }
            *((device float4*)(part_o + ((ulong)split_id * 24u + h) * 128u) + lane) = a_all;
        }
    }
"""
        else:
            sink_merge = f"""
    float sk_arr[3];
    for (uint g = 0u; g < 3u; g++) {{
        const uint h = kvh * 3u + g;
        const device float2* qin2 = (const device float2*)q_in;
        const device half2* sw2 = (const device half2*)(sink_w + (ulong)h * 2304u);
        float skp = 0.0f;
        #pragma unroll
        for (uint i = 0u; i < 9u; i++) {{
            const uint idx = tid + i * 128u;
            skp += dot(qin2[idx], float2(sw2[idx]));
        }}
        sk_arr[g] = simd_sum(skp) * LOG2E;
    }}
    threadgroup float tg_sk[4][3];
    if (lane == 0u) {{
        for (uint g = 0u; g < 3u; g++) tg_sk[simd][g] = sk_arr[g];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {{
        for (uint g = 0u; g < 3u; g++) {{
            const uint h = kvh * 3u + g;
            const float sk = tg_sk[0][g] + tg_sk[1][g] + tg_sk[2][g] + tg_sk[3][g];
            float m_all = sk;
            for (uint s = 0u; s < 4u; s++) m_all = max(m_all, tg_m[s][g]);
            float l_all = exp2(sk - m_all);
            float4 a_all = float4(0.0f);
            for (uint s = 0u; s < 4u; s++) {{
                if (tg_l[s][g] > 0.0f) {{
                    const float sc = exp2(tg_m[s][g] - m_all);
                    l_all += tg_l[s][g] * sc;
                    a_all += tg_a[s][g][lane] * sc;
                }}
            }}
            const float4 a_out = a_all * (1.0f / l_all);
            *((device float4*)(out + (ulong)h * 128u) + lane) = {unrot_expr};
        }}
    }}
"""
        src = f"""
    const uint kvh_sp = threadgroup_position_in_grid.y;
    const uint n_splits = {"meta[6]" if split_k else "1u"};
    const uint kvh = kvh_sp / n_splits;
    const uint split_id = kvh_sp - kvh * n_splits;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = simd * 32u + lane;

    const uint kv_len = meta[1];
    const uint kv_start = meta[2];
    const uint q_pos = meta[3];
    const uint kv_cap = meta[4];
    const uint is_sliding = meta[5];

    const device {kv_t}* Kp = KV + (ulong)kvh * kv_cap * {kv_stride}u;
    const device {kv_t}* Vp = KV + (ulong)(8u + kvh) * kv_cap * {kv_stride}u;
    {sc_decls}

    float4 qv[3], rs0[3], rs1[3], rs2[3], rs3[3];
    for (uint g = 0u; g < 3u; g++) {{
        const uint h = kvh * 3u + g;
        qv[g] = *((const device float4*)(q + (ulong)h * 128u) + lane) * (LOG2E / 11.313708498984761f);
        {q_rot}
        const device float4* rsp = (const device float4*)(rel_st + (ulong)h * 16u);
        rs0[g] = rsp[0] * LOG2E; rs1[g] = rsp[1] * LOG2E;
        rs2[g] = rsp[2] * LOG2E; rs3[g] = rsp[3] * LOG2E;
    }}

    const int min_pos = is_sliding ? ((int)q_pos - 1023) : 0;
    const uint j_lo = (min_pos > (int)kv_start) ? (uint)(min_pos - (int)kv_start) : 0u;
    const uint j_hi = min(kv_len, (q_pos >= kv_start) ? (q_pos - kv_start + 1u) : 0u);
    const uint span = (j_hi > j_lo) ? (j_hi - j_lo) : 0u;
    const uint chunk = (span + n_splits - 1u) / n_splits;
    const uint sbeg = min(j_hi, j_lo + split_id * chunk);
    const uint send = min(j_hi, sbeg + chunk);

    float m_loc[3] = {{-1e30f, -1e30f, -1e30f}};
    float l_loc[3] = {{0.0f, 0.0f, 0.0f}};
    float4 acc[3] = {{float4(0.0f), float4(0.0f), float4(0.0f)}};

    threadgroup half4 tg_k[16u * 32u];
    threadgroup half4 tg_v[16u * 32u];

    for (uint tile0 = sbeg; tile0 < send; tile0 += 16u) {{
        const uint r_hi = min(16u, send - tile0);
        #pragma unroll
        for (uint p = 0u; p < 4u; p++) {{
            const uint r = simd + p * 4u;
            if (r < r_hi) {{
                {load_kv_expr("tile0 + r")}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = simd; r < r_hi; r += 4u) {{
            const uint j = tile0 + r;
            const uint dist = q_pos - (kv_start + j);
            const float4 kv = float4(tg_k[r * 32u + lane]);
            const float4 vv = float4(tg_v[r * 32u + lane]);
            half4 rp0 = half4(0.0h), rp1 = half4(0.0h), rp2 = half4(0.0h), rp3 = half4(0.0h);
            const bool has_rel = dist < 1024u;
            if (has_rel) {{
                const device half4* rp = (const device half4*)(rel_proj_t + (ulong)dist * 16u);
                rp0 = rp[0]; rp1 = rp[1]; rp2 = rp[2]; rp3 = rp[3];
            }}
            #pragma unroll
            for (uint g = 0u; g < 3u; g++) {{
                float score = simd_sum(dot(qv[g], kv));
                if (has_rel) {{
                    score += dot(rs0[g], float4(rp0)) + dot(rs1[g], float4(rp1)) + dot(rs2[g], float4(rp2)) + dot(rs3[g], float4(rp3));
                }}
                const float m_new = max(m_loc[g], score);
                const float alpha = exp2(m_loc[g] - m_new);
                const float p_w = exp2(score - m_new);
                l_loc[g] = l_loc[g] * alpha + p_w;
                acc[g] = acc[g] * alpha + p_w * vv;
                m_loc[g] = m_new;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    threadgroup float tg_m[4][3], tg_l[4][3];
    threadgroup float4 tg_a[4][3][32];
    if (lane == 0u) {{
        for (uint g = 0u; g < 3u; g++) {{ tg_m[simd][g] = m_loc[g]; tg_l[simd][g] = l_loc[g]; }}
    }}
    for (uint g = 0u; g < 3u; g++) tg_a[simd][g][lane] = acc[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {sink_merge}
"""
    else:
        nt_s = 1 if t_cap == 4 else 2
        if tree_mode:
            tok_setup = """
        const uint q_pos_k = meta[3] + tree_depth[t];
        const uint my_mask_k = tree_mask[t];
        const int min_pos_k = is_sliding ? ((int)q_pos_k - 1023) : 0;
        const uint j_lo_k = (min_pos_k > (int)kv_start) ? (uint)(min_pos_k - (int)kv_start) : 0u;
        const uint j_hi_k = kv_len;
"""
            valid_dist = """
                bool valid = (j >= j_lo_k) && (j < j_hi_k);
                uint dist = 0u;
                if (valid) {
                    if (j >= kv_base) {
                        const uint u = j - kv_base;
                        if (((my_mask_k >> u) & 1u) == 0u) valid = false;
                        else dist = tree_depth[t] - tree_depth[u];
                    } else {
                        dist = q_pos_k - (kv_start + j);
                    }
                }
"""
        else:
            tok_setup = """
        const uint q_pos_k = meta[3] + t;
        const int min_pos_k = is_sliding ? ((int)q_pos_k - 1023) : 0;
        const uint j_lo_k = (min_pos_k > (int)kv_start) ? (uint)(min_pos_k - (int)kv_start) : 0u;
        const uint j_hi_k = min(kv_len, (q_pos_k >= kv_start) ? (q_pos_k - kv_start + 1u) : 0u);
"""
            valid_dist = """
                const bool valid = (j >= j_lo_k) && (j < j_hi_k);
                const uint dist = valid ? (q_pos_k - (kv_start + j)) : 0u;
"""

        if split_k:
            write_out = f"""
    for (uint k_t = 0u; k_t < {nt_s}u; k_t++) {{
        const uint t = simd + k_t * 4u;
        if (t < T) {{
            for (uint g = 0u; g < 3u; g++) {{
                const uint h = kvh * 3u + g;
                const uint idx_h = ((ulong)split_id * T + t) * 24u + h;
                if (lane == 0u) {{
                    part_m[idx_h] = m_loc[k_t][g];
                    part_l[idx_h] = l_loc[k_t][g];
                }}
                *((device float4*)(part_o + idx_h * 128u) + lane) = acc[k_t][g];
            }}
        }}
    }}
"""
        else:
            write_out = f"""
    for (uint k_t = 0u; k_t < {nt_s}u; k_t++) {{
        const uint t = simd + k_t * 4u;
        if (t < T) {{
            const device float4* qin4 = (const device float4*)(q_in + (ulong)t * 2304u);
            const device half4* sw0 = (const device half4*)(sink_w + (ulong)(kvh * 3u + 0u) * 2304u);
            const device half4* sw1 = (const device half4*)(sink_w + (ulong)(kvh * 3u + 1u) * 2304u);
            const device half4* sw2 = (const device half4*)(sink_w + (ulong)(kvh * 3u + 2u) * 2304u);
            float sk0 = 0.0f, sk1 = 0.0f, sk2 = 0.0f;
            #pragma unroll
            for (uint i = 0u; i < 18u; i++) {{
                const uint idx = lane + i * 32u;
                const float4 qv4 = qin4[idx];
                sk0 += dot(qv4, float4(sw0[idx]));
                sk1 += dot(qv4, float4(sw1[idx]));
                sk2 += dot(qv4, float4(sw2[idx]));
            }}
            float skv[3] = {{simd_sum(sk0) * LOG2E, simd_sum(sk1) * LOG2E, simd_sum(sk2) * LOG2E}};
            for (uint g = 0u; g < 3u; g++) {{
                const uint h = kvh * 3u + g;
                const float m_all = max(skv[g], m_loc[k_t][g]);
                const float sc_o = (l_loc[k_t][g] > 0.0f) ? exp2(m_loc[k_t][g] - m_all) : 0.0f;
                const float l_all = exp2(skv[g] - m_all) + l_loc[k_t][g] * sc_o;
                const float4 a_out = acc[k_t][g] * (sc_o / l_all);
                *((device float4*)(out + ((ulong)t * 24u + h) * 128u) + lane) = {unrot_expr};
            }}
        }}
    }}
"""

        src = f"""
    const uint kvh_sp = threadgroup_position_in_grid.y;
    const uint n_splits = {"meta[6]" if split_k else "1u"};
    const uint kvh = kvh_sp / n_splits;
    const uint split_id = kvh_sp - kvh * n_splits;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;

    const uint T = meta[0];
    const uint kv_len = meta[1];
    const uint kv_start = meta[2];
    const uint kv_cap = meta[4];
    const uint is_sliding = meta[5];
    const uint kv_base = kv_len - T;

    const uint q_pos_min = meta[3];
    const int min_pos_all = is_sliding ? ((int)q_pos_min - 1023) : 0;
    const uint tg_lo = (min_pos_all > (int)kv_start) ? (uint)(min_pos_all - (int)kv_start) : 0u;
    const uint tg_hi = {"kv_len" if tree_mode else "min(kv_len, (meta[3] + T - 1u >= kv_start) ? (meta[3] + T - kv_start) : 0u)"};
    const uint tg_span = (tg_hi > tg_lo) ? (tg_hi - tg_lo) : 0u;
    const uint chunk = (tg_span + n_splits - 1u) / n_splits;
    const uint sbeg = min(tg_hi, tg_lo + split_id * chunk);
    const uint send = min(tg_hi, sbeg + chunk);

    const device {kv_t}* Kp = KV + (ulong)kvh * kv_cap * {kv_stride}u;
    const device {kv_t}* Vp = KV + (ulong)(8u + kvh) * kv_cap * {kv_stride}u;
    {sc_decls}

    float4 qv[{nt_s}][3], rs0[{nt_s}][3], rs1[{nt_s}][3], rs2[{nt_s}][3], rs3[{nt_s}][3];
    float m_loc[{nt_s}][3], l_loc[{nt_s}][3];
    float4 acc[{nt_s}][3];
    uint q_pos_arr[{nt_s}], j_lo_arr[{nt_s}], j_hi_arr[{nt_s}], mask_arr[{nt_s}];
    for (uint k_t = 0u; k_t < {nt_s}u; k_t++) {{
        const uint t = simd + k_t * 4u;
        j_lo_arr[k_t] = 0u; j_hi_arr[k_t] = 0u; q_pos_arr[k_t] = 0u; mask_arr[k_t] = 0u;
        for (uint g = 0u; g < 3u; g++) {{
            m_loc[k_t][g] = -1e30f; l_loc[k_t][g] = 0.0f; acc[k_t][g] = float4(0.0f);
            qv[k_t][g] = float4(0.0f); rs0[k_t][g] = float4(0.0f); rs1[k_t][g] = float4(0.0f); rs2[k_t][g] = float4(0.0f); rs3[k_t][g] = float4(0.0f);
        }}
        if (t < T) {{
            for (uint g = 0u; g < 3u; g++) {{
                const uint h = kvh * 3u + g;
                qv[k_t][g] = *((const device float4*)(q + ((ulong)t * 24u + h) * 128u) + lane) * (LOG2E / 11.313708498984761f);
                {q_rot_k}
                const device float4* rsp = (const device float4*)(rel_st + ((ulong)t * 24u + h) * 16u);
                rs0[k_t][g] = rsp[0] * LOG2E; rs1[k_t][g] = rsp[1] * LOG2E;
                rs2[k_t][g] = rsp[2] * LOG2E; rs3[k_t][g] = rsp[3] * LOG2E;
            }}
            {tok_setup}
            q_pos_arr[k_t] = q_pos_k; j_lo_arr[k_t] = j_lo_k; j_hi_arr[k_t] = j_hi_k;
            {"mask_arr[k_t] = my_mask_k;" if tree_mode else ""}
        }}
    }}

    threadgroup half4 tg_k[16u * 32u];
    threadgroup half4 tg_v[16u * 32u];

    for (uint tile0 = sbeg; tile0 < send; tile0 += 16u) {{
        const uint r_hi = min(16u, send - tile0);
        #pragma unroll
        for (uint p = 0u; p < 4u; p++) {{
            const uint r = simd + p * 4u;
            if (r < r_hi) {{
                {load_kv_expr("tile0 + r")}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0u; r < r_hi; r++) {{
            const uint j = tile0 + r;
            const float4 kv = float4(tg_k[r * 32u + lane]);
            const float4 vv = float4(tg_v[r * 32u + lane]);
            #pragma unroll
            for (uint k_t = 0u; k_t < {nt_s}u; k_t++) {{
                const uint t = simd + k_t * 4u;
                if (t < T) {{
                    const uint q_pos_k = q_pos_arr[k_t], j_lo_k = j_lo_arr[k_t], j_hi_k = j_hi_arr[k_t];
                    {"const uint my_mask_k = mask_arr[k_t];" if tree_mode else ""}
                    {valid_dist}
                    if (valid) {{
                        half4 rp0 = half4(0.0h), rp1 = half4(0.0h), rp2 = half4(0.0h), rp3 = half4(0.0h);
                        const bool has_rel = dist < 1024u;
                        if (has_rel) {{
                            const device half4* rp = (const device half4*)(rel_proj_t + (ulong)dist * 16u);
                            rp0 = rp[0]; rp1 = rp[1]; rp2 = rp[2]; rp3 = rp[3];
                        }}
                        #pragma unroll
                        for (uint g = 0u; g < 3u; g++) {{
                            float score = simd_sum(dot(qv[k_t][g], kv));
                            if (has_rel) {{
                                score += dot(rs0[k_t][g], float4(rp0)) + dot(rs1[k_t][g], float4(rp1)) + dot(rs2[k_t][g], float4(rp2)) + dot(rs3[k_t][g], float4(rp3));
                            }}
                            const float m_new = max(m_loc[k_t][g], score);
                            const float alpha = exp2(m_loc[k_t][g] - m_new);
                            const float p_w = exp2(score - m_new);
                            l_loc[k_t][g] = l_loc[k_t][g] * alpha + p_w;
                            acc[k_t][g] = acc[k_t][g] * alpha + p_w * vv;
                            m_loc[k_t][g] = m_new;
                        }}
                    }}
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    {write_out}
"""

    return mx.fast.metal_kernel(
        name=f"ink_gqa_fused_sdpa_{kv_mode}_tc{t_cap}_tr{int(tree_mode)}_sp{int(split_k)}",
        input_names=in_names,
        output_names=out_names,
        header=header,
        source=src,
    )


@lru_cache(maxsize=2)
def _make_gqa_splitk_reduce_kernel(apply_h128: bool = False):
    """Pass-2 Split-K merge + inline sink logit (+ optional H128 inverse) for fused GQA Flash-Decoding."""
    header = "#define LOG2E 1.4426950408889634f\n" + (_H128_HEADER if apply_h128 else "")
    unrot = "h128_vec(a_out, lane)" if apply_h128 else "a_out"
    src = f"""
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
    for (uint i = 0u; i < 9u; i++) {{
        const uint idx = tid + i * 128u;
        sk_part += dot(qin2[idx], float2(sw2[idx]));
    }}
    sk_part = simd_sum(sk_part) * LOG2E;

    float m_s = -1e30f;
    float l_s = 0.0f;
    float4 a_s = float4(0.0f);
    for (uint s = simd; s < n_splits; s += 4u) {{
        const uint idx_h = ((ulong)s * T + t) * 24u + h;
        const float ms = part_m[idx_h];
        const float ls = part_l[idx_h];
        if (ls > 0.0f) {{
            const float4 as = *((const device float4*)(part_o + idx_h * 128u) + lane);
            const float m_new = max(m_s, ms);
            const float al = (l_s > 0.0f) ? exp2(m_s - m_new) : 0.0f;
            const float be = exp2(ms - m_new);
            l_s = l_s * al + ls * be;
            a_s = a_s * al + as * be;
            m_s = m_new;
        }}
    }}

    threadgroup float tg_m[4], tg_l[4], tg_sk[4];
    threadgroup float4 tg_a[4][32];
    if (lane == 0u) {{ tg_m[simd] = m_s; tg_l[simd] = l_s; tg_sk[simd] = sk_part; }}
    tg_a[simd][lane] = a_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd == 0u) {{
        const float sk = tg_sk[0] + tg_sk[1] + tg_sk[2] + tg_sk[3];
        float m_all = sk;
        for (uint s = 0u; s < 4u; s++) m_all = max(m_all, tg_m[s]);
        float l_all = exp2(sk - m_all);
        float4 a_all = float4(0.0f);
        for (uint s = 0u; s < 4u; s++) {{
            if (tg_l[s] > 0.0f) {{
                const float sc = exp2(tg_m[s] - m_all);
                l_all += tg_l[s] * sc;
                a_all += tg_a[s][lane] * sc;
            }}
        }}
        const float4 a_out = a_all * (1.0f / l_all);
        *((device float4*)(out + ((ulong)t * 24u + h) * 128u) + lane) = {unrot};
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_gqa_splitk_reduce_h{int(apply_h128)}",
        input_names=["part_m", "part_l", "part_o", "q_in", "sink_w", "meta"],
        output_names=["out"],
        header=header,
        source=src,
    )


def gqa_relpos_sinks_sdpa_fast(
    q: mx.array,
    KV_buf: mx.array,
    rel_st: mx.array,
    rel_proj_t_f16: mx.array,
    q_in: mx.array,
    sink_w_f16: mx.array,
    T: int,
    kv_len: int,
    kv_start: int,
    q_pos0: int,
    kv_cap: int,
    is_sliding: bool,
    tree_depth: mx.array | None = None,
    tree_mask: mx.array | None = None,
    kv_fp8: bool = False,
    kv_q4_scales: mx.array | None = None,
) -> mx.array:
    tree_mode = tree_depth is not None and tree_mask is not None
    kv_q4 = kv_q4_scales is not None
    if kv_q4 and tree_mode and T > 8:
        raise ValueError("Q4 attention trees support at most eight tokens")
    if not tree_mode and (T >= 32 or (kv_q4 and T > 8)):
        sinks = (q_in.reshape(T, 2304).astype(mx.float16) @ sink_w_f16.reshape(24, 2304).T).astype(mx.float32)
        nqb = (T + 31) // 32
        q_in_kern = h128_rows_f32(q) if kv_q4 else q
        ins = [q_in_kern.reshape(-1), KV_buf.reshape(-1)]
        if kv_q4:
            ins.append(kv_q4_scales.reshape(-1))
        ins.extend([rel_st.reshape(-1), rel_proj_t_f16.reshape(-1), sinks.reshape(-1), _u32(T, kv_len, kv_start, q_pos0, kv_cap, int(is_sliding), nqb)])
        out = _make_gqa_relpos_flash_sdpa_kernel(kv_fp8=(kv_fp8 and not kv_q4), kv_q4=kv_q4)(
            inputs=ins,
            grid=(128, 24 * nqb, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(T, 3072)],
            output_dtypes=[mx.float32],
        )[0]
        return h128_rows_f32(out.reshape(T, 24, 128)).reshape(T, 3072) if kv_q4 else out
    span = min(kv_len, 1024 + T) if is_sliding else kv_len
    env_fused_raw = os.environ.get("LAMQ_FUSED_SDPA", "1" if _is_m4() else "0").strip().lower()
    force_fused = env_fused_raw in ("force", "2")
    env_fused = env_fused_raw != "0"
    use_fused = (
        kv_q4
        or (force_fused and 1 <= T <= 8)
        or (
            env_fused
            and (
                (T == 1 and span >= 256)
                or (2 <= T <= 4 and span >= 1024)
                or (5 <= T <= 8 and not is_sliding and span >= 2048)
            )
        )
    )
    if use_fused:
        kv_mode = "q4" if kv_q4 else ("fp8" if kv_fp8 else "fp16")
        t_cap = 1 if T == 1 else (4 if T <= 4 else 8)
        if span < 256:
            n_splits = 1
        elif span < 2048:
            n_splits = 4
        elif span < 8192:
            n_splits = 8
        else:
            n_splits = 16
        split_k = n_splits > 1
        kern = _make_gqa_fused_sdpa_kernel(tree_mode=tree_mode, kv_mode=kv_mode, t_cap=t_cap, split_k=split_k)
        ins = [q.reshape(-1), KV_buf.reshape(-1)]
        if kv_q4:
            ins.append(kv_q4_scales.reshape(-1))
        ins.extend([rel_st.reshape(-1), rel_proj_t_f16.reshape(-1)])
        if not split_k:
            ins.extend([q_in.reshape(-1), sink_w_f16.reshape(-1), _u32(T, kv_len, kv_start, q_pos0, kv_cap, int(is_sliding))])
            if tree_mode:
                ins.extend([tree_depth, tree_mask])
            return kern(
                inputs=ins,
                grid=(128, 8, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(T, 3072)],
                output_dtypes=[mx.float32],
            )[0]
        ins.append(_u32(T, kv_len, kv_start, q_pos0, kv_cap, int(is_sliding), n_splits))
        if tree_mode:
            ins.extend([tree_depth, tree_mask])
        pm, pl, po = kern(
            inputs=ins,
            grid=(128, 8 * n_splits, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n_splits, T, 24), (n_splits, T, 24), (n_splits, T, 24, 128)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        return _make_gqa_splitk_reduce_kernel(apply_h128=kv_q4)(
            inputs=[pm.reshape(-1), pl.reshape(-1), po.reshape(-1), q_in.reshape(-1), sink_w_f16.reshape(-1), _u32(T, n_splits)],
            grid=(128, 24 * T, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(T, 3072)],
            output_dtypes=[mx.float32],
        )[0]
    ins = [
        q.reshape(-1), KV_buf.reshape(-1),
        rel_st.reshape(-1), rel_proj_t_f16.reshape(-1),
        q_in.reshape(-1), sink_w_f16.reshape(-1),
        _u32(T, kv_len, kv_start, q_pos0, kv_cap, int(is_sliding)),
    ]
    if tree_mode:
        ins.extend([tree_depth, tree_mask])
    return _make_gqa_relpos_sdpa_kernel(tree_mode, kv_fp8=kv_fp8)(
        inputs=ins,
        grid=(128, 24 * T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(T, 3072)],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=8)
def _make_short_conv4_res_kernel(with_res: bool, tree_mode: bool = False):
    in_names = (["prev", "x", "w", "res", "meta"] if with_res else ["prev", "x", "w", "meta"]) + (["tree_lag"] if tree_mode else [])
    add_res = " + res[(ulong)t * C + c]" if with_res else ""
    lag_expr = "tree_lag[t * 4u + k]" if tree_mode else "((int)t + (int)k - 3)"
    src = f"""
    const uint c = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const uint C = meta[0], T = meta[1];
    if (c >= C) return;
    float acc = x[(ulong)t * C + c];
    for (uint k = 0; k < 4u; k++) {{
        const int idx = {lag_expr};
        const float v = (idx >= 0) ? x[(ulong)idx * C + c] : prev[(ulong)(idx + 3) * C + c];
        acc += v * w[(ulong)k * C + c];
    }}
    out[(ulong)t * C + c] = acc * 0.4472135954999579f{add_res};
    if (t == 0u) {{
        for (uint j = 0; j < 3u; j++) {{
            const int idx = (int)T - 3 + (int)j;
            new_prev[(ulong)j * C + c] = (idx >= 0) ? x[(ulong)idx * C + c] : prev[(ulong)(idx + 3) * C + c];
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_sconv4_r{int(with_res)}_tr{int(tree_mode)}",
        input_names=in_names,
        output_names=["out", "new_prev"],
        source=src,
    )


def short_conv4_res_fused(
    prev: mx.array,
    x: mx.array,
    w: mx.array,
    T: int,
    C: int,
    residual: mx.array | None = None,
    tree_lag: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    gx = ((C + 255) // 256) * 256
    mt = _u32(C, T)
    tree_mode = tree_lag is not None
    if residual is not None:
        ins = [prev.reshape(-1), x.reshape(-1), w.reshape(-1), residual.reshape(-1), mt]
        if tree_mode:
            ins.append(tree_lag)
        return _make_short_conv4_res_kernel(True, tree_mode)(
            inputs=ins,
            grid=(gx, T, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(T, C), (1, 3, C)],
            output_dtypes=[mx.float32, mx.float32],
        )
    ins = [prev.reshape(-1), x.reshape(-1), w.reshape(-1), mt]
    if tree_mode:
        ins.append(tree_lag)
    return _make_short_conv4_res_kernel(False, tree_mode)(
        inputs=ins,
        grid=(gx, T, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, C), (1, 3, C)],
        output_dtypes=[mx.float32, mx.float32],
    )


@lru_cache(maxsize=4)
def _make_moe_wsum_sconv4_res_kernel(tree_mode: bool = False):
    # Fuses: mlp[t, c] = sum_{k=0..7, wk!=0} y_dn[(t*8+k)*2304 + c] * w_norm[t*8+k] -> short_conv4 -> + residual!
    # Uniform `if (wk != 0.0f)` guard avoids IEEE-754 `NaN * 0.0f = NaN` on skipped dynamic-TopK slots with 0 overhead!
    if tree_mode:
        src = """
        const uint c = thread_position_in_grid.x;
        const uint T = meta[0];
        if (c >= 2304u) return;
        float hist[11];
        hist[0] = prev[c]; hist[1] = prev[2304u + c]; hist[2] = prev[4608u + c];
        const float w0 = conv_w[c], w1 = conv_w[2304u + c], w2 = conv_w[4608u + c], w3 = conv_w[6912u + c];
        constexpr float INV_SQRT5 = 0.4472135954999579f;
        for (uint t = 0; t < T; t++) {
            const ulong base_slot = (ulong)t * 8u;
            float s = 0.0f;
            #pragma unroll
            for (uint k = 0; k < 8u; k++) {
                const float wk = w_norm[base_slot + k];
                if (wk != 0.0f) s += y_dn[(base_slot + k) * 2304u + c] * wk;
            }
            hist[3u + t] = s;
            const float v0 = hist[tree_lag[t * 4u + 0u] + 3];
            const float v1 = hist[tree_lag[t * 4u + 1u] + 3];
            const float v2 = hist[tree_lag[t * 4u + 2u] + 3];
            const float conv_out = (v0 * w0 + v1 * w1 + v2 * w2 + s * w3 + s) * INV_SQRT5;
            raw_out[(ulong)t * 2304u + c] = s;
            out[(ulong)t * 2304u + c] = residual[(ulong)t * 2304u + c] + conv_out;
        }
        new_prev[c] = hist[T];
        new_prev[2304u + c] = hist[T + 1u];
        new_prev[4608u + c] = hist[T + 2u];
        """
        in_names = ["y_dn", "w_norm", "prev", "conv_w", "residual", "meta", "tree_lag"]
    else:
        src = """
        const uint c = thread_position_in_grid.x;
        const uint T = meta[0];
        if (c >= 2304u) return;
        float p0 = prev[c], p1 = prev[2304u + c], p2 = prev[4608u + c];
        const float w0 = conv_w[c], w1 = conv_w[2304u + c], w2 = conv_w[4608u + c], w3 = conv_w[6912u + c];
        constexpr float INV_SQRT5 = 0.4472135954999579f;
        for (uint t = 0; t < T; t++) {
            const ulong base_slot = (ulong)t * 8u;
            float s = 0.0f;
            #pragma unroll
            for (uint k = 0; k < 8u; k++) {
                const float wk = w_norm[base_slot + k];
                if (wk != 0.0f) s += y_dn[(base_slot + k) * 2304u + c] * wk;
            }
            const float conv_out = (p0 * w0 + p1 * w1 + p2 * w2 + s * w3 + s) * INV_SQRT5;
            raw_out[(ulong)t * 2304u + c] = s;
            out[(ulong)t * 2304u + c] = residual[(ulong)t * 2304u + c] + conv_out;
            p0 = p1; p1 = p2; p2 = s;
        }
        new_prev[c] = p0;
        new_prev[2304u + c] = p1;
        new_prev[4608u + c] = p2;
        """
        in_names = ["y_dn", "w_norm", "prev", "conv_w", "residual", "meta"]
    return mx.fast.metal_kernel(
        name=f"ink_moe_wsum_sconv4_res_tr{int(tree_mode)}",
        input_names=in_names,
        output_names=["out", "new_prev", "raw_out"],
        source=src,
    )


@lru_cache(maxsize=2)
def _make_moe_wsum_sconv4_res_tiled_kernel(chunk_t: int = 16):
    """Vectorized float4 tiled MoE weighted-sum + short_conv4 + residual for arbitrary prefill `T > 16` (1.99x faster, NaN-safe)."""
    src = f"""
    const uint c4 = thread_position_in_grid.x; // 0..575 (576 float4 = 2304 channels)
    const uint blk = threadgroup_position_in_grid.y;
    const uint T = meta[0];
    if (c4 >= 576u) return;
    const uint t0 = blk * {chunk_t}u;
    if (t0 >= T) return;
    const uint t1 = min(T, t0 + {chunk_t}u);

    const device float4* y4 = (const device float4*)y_dn;
    const device float4* pv4 = (const device float4*)prev;
    const device float4* cw4 = (const device float4*)conv_w;
    const device float4* res4 = (const device float4*)residual;
    device float4* out4 = (device float4*)out;
    device float4* np4 = (device float4*)new_prev;
    device float4* raw4 = (device float4*)raw_out;

    float4 p0, p1, p2;
    if (t0 == 0u) {{
        p0 = pv4[c4]; p1 = pv4[576u + c4]; p2 = pv4[1152u + c4];
    }} else {{
        float4 pre[3];
        #pragma unroll
        for (uint j = 0; j < 3u; j++) {{
            const int pt = (int)t0 - 3 + (int)j;
            if (pt < 0) {{
                pre[j] = pv4[(ulong)(pt + 3) * 576u + c4];
            }} else {{
                const ulong bs = (ulong)pt * 8u;
                float4 s = float4(0.0f);
                #pragma unroll
                for (uint k = 0; k < 8u; k++) {{
                    const float wk = w_norm[bs + k];
                    if (wk != 0.0f) s += y4[(bs + k) * 576u + c4] * wk;
                }}
                pre[j] = s;
            }}
        }}
        p0 = pre[0]; p1 = pre[1]; p2 = pre[2];
    }}
    const float4 w0 = cw4[c4], w1 = cw4[576u + c4], w2 = cw4[1152u + c4], w3 = cw4[1728u + c4];
    constexpr float INV_SQRT5 = 0.4472135954999579f;
    for (uint t = t0; t < t1; t++) {{
        const ulong bs = (ulong)t * 8u;
        float4 s = float4(0.0f);
        #pragma unroll
        for (uint k = 0; k < 8u; k++) {{
            const float wk = w_norm[bs + k];
            if (wk != 0.0f) s += y4[(bs + k) * 576u + c4] * wk;
        }}
        const float4 conv_out = (p0 * w0 + p1 * w1 + p2 * w2 + s * w3 + s) * INV_SQRT5;
        raw4[(ulong)t * 576u + c4] = s;
        out4[(ulong)t * 576u + c4] = res4[(ulong)t * 576u + c4] + conv_out;
        p0 = p1; p1 = p2; p2 = s;
    }}
    if (t1 == T) {{
        np4[c4] = p0;
        np4[576u + c4] = p1;
        np4[1152u + c4] = p2;
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_moe_wsum_sconv4_res_tiled_c{chunk_t}",
        input_names=["y_dn", "w_norm", "prev", "conv_w", "residual", "meta"],
        output_names=["out", "new_prev", "raw_out"],
        source=src,
    )


def moe_wsum_sconv4_res_fused(
    y_dn: mx.array,
    w_norm: mx.array,
    prev: mx.array,
    conv_w: mx.array,
    residual: mx.array,
    T: int,
    tree_lag: mx.array | None = None,
    skip_zero_w: bool = False,
) -> tuple[mx.array, mx.array, mx.array]:
    # Works for ANY T (T <= 16 single-chunk decode/tree; T > 16 float4-tiled prefill along grid.y) and skips wk == 0.0f safely!
    if T > 16 and tree_lag is None:
        chunk_t = 16
        n_blks = (T + chunk_t - 1) // chunk_t
        return _make_moe_wsum_sconv4_res_tiled_kernel(chunk_t)(
            inputs=[y_dn.reshape(-1), w_norm.reshape(-1), prev.reshape(-1), conv_w.reshape(-1), residual.reshape(-1), _u32(T)],
            grid=(576, n_blks, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(T, 2304), (1, 3, 2304), (1, T, 2304)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
    ins = [y_dn.reshape(-1), w_norm.reshape(-1), prev.reshape(-1), conv_w.reshape(-1), residual.reshape(-1), _u32(T)]
    if tree_lag is not None:
        ins.append(tree_lag)
    return _make_moe_wsum_sconv4_res_kernel(tree_lag is not None)(
        inputs=ins,
        grid=(2304, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(T, 2304), (1, 3, 2304), (1, T, 2304)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )


# ----------------------------------------------------------------------------- GPU Packed Embedding Lookup

@lru_cache(maxsize=2)
def _make_embed_lookup_kernel():
    # Decodes T token rows directly on GPU from packed uint32 codes + scales + levels/luts!
    # grid = (2304, T, 1), threadgroup = (64, 1, 1): 1 thread per (col, t)!
    src = """
    const uint j = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    if (j >= 2304u) return;
    const uint tok = (uint)ids[t];
    const uint pos = inv_order[tok];
    // Find bucket (<= 8 buckets)
    const uint nb = meta[0];
    uint bi = 0u;
    while (bi + 1u < nb && pos >= b_hi[bi]) bi++;
    const device uint* d = b_dims + bi * 6u; // [fid, k, wpr, c_off, s_off, l_off]
    const uint fid = d[0], k = d[1], wpr = d[2], c_off = d[3], s_off = d[4], l_off = d[5];
    const uint r = pos - ((bi == 0u) ? 0u : b_hi[bi - 1u]);
    const float sc = (float)scales[s_off + r];
    const device uint* p = codes + c_off + (ulong)r * wpr;
    float v = 0.0f;
    if (fid == 10u) { // lm6: 4 cols = 3 bytes -> bit offset = j * 6
        const uint bit = j * 6u, wi = bit >> 5, off = bit & 31u;
        uint s = p[wi] >> off;
        if (off + 6u > 32u) s |= p[wi + 1u] << (32u - off);
        v = (float)levels[l_off + (s & 63u)];
    } else if (fid == 11u) { // lm8
        const uint wi = j >> 2, off = (j & 3u) * 8u;
        v = (float)levels[l_off + ((p[wi] >> off) & 255u)];
    } else { // RowTrellisCode (rtr1..rtr4: g=1, L=12)
        const uint bit = (2303u - j) * k, wi = bit >> 5, off = bit & 31u;
        uint s = p[wi] >> off;
        if (off + 12u > 32u && wi + 1u < wpr) s |= p[wi + 1u] << (32u - off);
        s &= 4095u;
        // Lookup in lut_rtr_flat[fid_idx * 4096 + s] where fid_idx = fid - 3 (rtr1=0, rtr2=1, rtr3=2, rtr4=3)
        v = (float)rtr_luts[(fid - 3u) * 4096u + s];
    }
    q_out[(ulong)t * 2304u + j] = v * sc;
    if (j == 0u) b_idx_out[t] = bi;
    """
    return mx.fast.metal_kernel(
        name="ink_embed_lookup_gpu",
        input_names=["ids", "inv_order", "b_hi", "b_dims", "codes", "scales", "levels", "rtr_luts", "meta"],
        output_names=["q_out", "b_idx_out"],
        source=src,
    )


def embed_lookup_gpu(
    ids: mx.array,
    inv_order: mx.array,
    b_hi: mx.array,
    b_dims: mx.array,
    codes: mx.array,
    scales: mx.array,
    levels: mx.array,
    rtr_luts: mx.array,
    nb: int,
) -> tuple[mx.array, mx.array]:
    T = int(ids.size)
    return _make_embed_lookup_kernel()(
        inputs=[ids.reshape(-1).astype(mx.int32), inv_order, b_hi, b_dims, codes, scales, levels, rtr_luts, _u32(nb)],
        grid=(2304, T, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(T, 2304), (T,)],
        output_dtypes=[mx.float32, mx.uint32],
    )


# ----------------------------------------------------------------------------- Self-test

def _selftest():
    from .kernels import rht_apply_in, rht_apply_out, trellis_dense_matmul, lm_dense_matmul

    rng = np.random.default_rng(13)
    print(f"Testing fast_kernels on {mx.device_info().get('architecture')}...", flush=True)

    # 1. RHT-2304, SwiGLU+RHT-1152, Gate+RHT-3072, RHT-out-2304
    x = mx.array(rng.standard_normal((8, 2304)).astype(np.float32))
    s5 = mx.array(rng.choice([-1.0, 1.0], (5, 2304)).astype(np.float32))
    y_fast = rht_in_2304_fast(x, s5, n_signs=5, T=8, out_dtype=mx.float32)
    y_ref = mx.concatenate([rht_apply_in(x, s5[i]) for i in range(5)], axis=0)
    mx.eval(y_fast, y_ref)
    err = float(mx.linalg.norm(y_fast - y_ref) / mx.linalg.norm(y_ref))
    assert err < 2e-3, f"rht2304 fail: {err}"
    print(f"  PASS rht_in_2304_fast (5x8): rel_err={err:.2e}")

    y_out = rht_out_2304_fast(y_fast[:8], s5, mx.zeros((8,), dtype=mx.uint32))
    mx.eval(y_out)
    err = float(mx.linalg.norm(y_out - x) / mx.linalg.norm(x))
    assert err < 1e-4, f"rht2304_out fail: {err}"
    print(f"  PASS rht_out_2304_fast invertibility: rel_err={err:.2e}")

    # 2. DenseStage across ALL formats (ht2, ht3, rtr1, rtr2, rtr3, rtr4, lm6, lm8) for T=1, T=8, and T=16 (dequant GEMM)!
    luts = {
        "ht1": mx.array(rng.standard_normal((512, 4)).astype(np.float16)),
        "ht2": mx.array(rng.standard_normal((512, 4)).astype(np.float16)),
        "ht3": mx.array(rng.standard_normal((1024, 2)).astype(np.float16)),
        "rtr1": mx.array(rng.standard_normal((4096,)).astype(np.float16)),
        "rtr2": mx.array(rng.standard_normal((4096,)).astype(np.float16)),
        "rtr3": mx.array(rng.standard_normal((4096,)).astype(np.float16)),
        "rtr4": mx.array(rng.standard_normal((4096,)).astype(np.float16)),
    }
    fluts = build_fast_luts(luts)
    items = []
    ref_specs = []
    for fmt, k, g, K, hm in [
        ("ht2", 2, 4, 512, 0x9E3779B1),
        ("ht3", 3, 2, 1024, 0x9E3779B1),
        ("rtr1", 1, 1, 4096, 0),
        ("rtr2", 2, 1, 4096, 0),
        ("rtr3", 3, 1, 4096, 0),
        ("rtr4", 4, 1, 4096, 0),
    ]:
        m_r, n_c = 64, 2304
        codes = rng.integers(0, 2**32, (m_r, (n_c * k) // 32), dtype=np.uint32)
        scales = rng.uniform(0.01, 0.05, (m_r,)).astype(np.float16)
        items.append({"fmt": fmt, "m": m_r, "codes": codes, "scales": scales})
        ref_specs.append(("tr", fmt, m_r, n_c, k, g, K, hm, mx.array(codes), mx.array(scales), luts[fmt]))
    for fmt, k in [("lm6", 6), ("lm8", 8)]:
        m_r, n_c = 64, 2304
        levels = np.sort(rng.standard_normal(1 << k)).astype(np.float32)
        idx = rng.integers(0, 1 << k, (m_r, n_c), dtype=np.uint8)
        if k == 8:
            packed = idx.reshape(-1)
        else:
            a = idx.reshape(-1, 4).astype(np.uint32)
            v = a[:, 0] | (a[:, 1] << 6) | (a[:, 2] << 12) | (a[:, 3] << 18)
            packed = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=-1).astype(np.uint8).reshape(-1)
        scales = rng.uniform(0.01, 0.05, (m_r,)).astype(np.float16)
        items.append({"fmt": fmt, "m": m_r, "codes": packed, "scales": scales, "levels": levels})
        ref_specs.append(("lm", fmt, m_r, n_c, k, 0, 0, 0, mx.array(packed), mx.array(scales), mx.array(levels)))

    st = DenseStage(items, 2304, fluts)
    for T in (1, 4, 6, 8, 16):
        xt = mx.array(rng.standard_normal((len(items), T, 2304)).astype(np.float32))
        y_st = st.matmul_rotated_f16(xt.astype(mx.float16), T)
        refs = []
        for idx_i, sp in enumerate(ref_specs):
            if sp[0] == "tr":
                refs.append(trellis_dense_matmul(xt[idx_i], sp[8], sp[9], sp[10], sp[2], sp[3], sp[4], sp[5], sp[6], sp[7]))
            else:
                refs.append(lm_dense_matmul(xt[idx_i], sp[8], sp[9], sp[10], sp[2], sp[3], sp[4]))
        y_rf = mx.concatenate(refs, axis=-1)
        mx.eval(y_st, y_rf)
        err = float(mx.linalg.norm(y_st - y_rf) / mx.linalg.norm(y_rf))
        assert err < 2e-3, f"DenseStage T={T} fail: {err}"
        print(f"  PASS DenseStage ALL 8 formats (T={T:2d}): rel_err={err:.2e}")
        if T in (4, 6, 8):
            if T < 8:
                pad = mx.zeros((len(items), 8 - T, 2304), dtype=mx.float32)
                xt_tr = mx.concatenate([xt, pad], axis=1).transpose(0, 2, 1).astype(mx.float16)
            else:
                xt_tr = xt.transpose(0, 2, 1).astype(mx.float16)
            y_sg8 = st.matmul_rotated_sg8_f16(xt_tr, T=T)
            mx.eval(y_sg8)
            err_sg8 = float(mx.linalg.norm(y_sg8 - y_rf) / mx.linalg.norm(y_rf))
            assert err_sg8 < 2e-3, f"DenseStage sg8 T={T} fail: {err_sg8}"
            print(f"  PASS DenseStage sg8 ALL 8 formats (T={T:2d}): rel_err={err_sg8:.2e}")

    # Test transpose_for_sg8 on rht_in_2304_fast for T=8 and T=5 (zero-padded to 8)
    y_tr = rht_in_2304_fast(x, s5, n_signs=5, T=8, out_dtype=mx.float16, transpose_for_sg8=True)
    y_exp = y_fast.reshape(5, 8, 2304).transpose(0, 2, 1).astype(mx.float16)
    mx.eval(y_tr, y_exp)
    err_tr = float(mx.linalg.norm(y_tr.astype(mx.float32) - y_exp.astype(mx.float32)) / mx.linalg.norm(y_exp.astype(mx.float32)))
    assert err_tr < 2e-3, f"rht2304 transpose_for_sg8 T=8 fail: {err_tr}"
    y_tr5 = rht_in_2304_fast(x[:5], s5, n_signs=5, T=5, out_dtype=mx.float16, transpose_for_sg8=True)
    mx.eval(y_tr5)
    assert y_tr5.shape == (5, 2304, 8)
    assert float(mx.max(mx.abs(y_tr5[:, :, 5:].astype(mx.float32)))) == 0.0
    err_tr5 = float(mx.linalg.norm(y_tr5[:, :, :5].astype(mx.float32) - y_exp[:, :, :5].astype(mx.float32)) / mx.linalg.norm(y_exp[:, :, :5].astype(mx.float32)))
    assert err_tr5 < 2e-3, f"rht2304 transpose_for_sg8 T=5 fail: {err_tr5}"
    print(f"  PASS rht_in_2304_fast transpose_for_sg8 (T=8, T=5): rel_err={err_tr:.2e}, {err_tr5:.2e}")

    # 3. UnifiedMoESide across T=1, T=8 (including cnt=6, 7, 8 split!), and T=16 (prefill sg8!)
    slabs = []
    mf = np.zeros(128, dtype=np.int32)
    ml = np.zeros(128, dtype=np.uint32)
    for block_i, (fid, fmt, k, g, K, hm) in enumerate([(1, "ht2", 2, 4, 512, 0x9E3779B1), (2, "ht3", 3, 2, 1024, 0x9E3779B1), (3, "rtr1", 1, 1, 4096, 0), (6, "rtr4", 4, 1, 4096, 0)]):
        c = mx.array(rng.integers(0, 2**32, (32, 64, (2304 * k) // 32), dtype=np.uint32))
        s = mx.array(rng.uniform(0.01, 0.05, (32, 64)).astype(np.float16))
        slabs.append((fid, c, s, luts[fmt], (k, g, K, hm)))
        mf[block_i * 32 : (block_i + 1) * 32] = fid
        ml[block_i * 32 : (block_i + 1) * 32] = np.arange(32, dtype=np.uint32)
    umoe = UnifiedMoESide(slabs, mx.array(mf), mx.array(ml), 64, 2304, fluts)
    from .kernels import trellis_matvec
    for T, eids_np in [
        (1, np.array([0, 33, 66, 99, 5, 40, 70, 105], dtype=np.uint32)),
        (8, np.array([0, 32, 64, 96, 1, 33, 65, 97] * 6 + [0, 32, 64, 96, 2, 34, 66, 98] + [0, 32, 64, 96, 3, 35, 67, 99], dtype=np.uint32)),
        (16, rng.integers(0, 128, 128, dtype=np.uint32)),
    ]:
        xt = mx.array(rng.standard_normal((T, 2304)).astype(np.float32))
        ev = mx.array(eids_np)
        y_u = umoe(xt.astype(mx.float16), ev, x_div=8)
        y_r = mx.zeros((T * 8, 64), dtype=mx.float32)
        sent = mx.array(0xFFFFFFFF, dtype=mx.uint32)
        for fid, c, s, l, (k, g, K, hm) in slabs:
            lf = mx.where(mx.array(mf)[ev] == fid, mx.array(ml)[ev], sent)
            y_r = y_r + trellis_matvec(xt, c, s, l, lf, 64, 2304, k, g, K, hm, x_div=8)
        mx.eval(y_u, y_r)
        err = float(mx.linalg.norm(y_u - y_r) / mx.linalg.norm(y_r))
        assert err < 2e-3, f"UnifiedMoESide T={T} fail: {err}"
        print(f"  PASS UnifiedMoESide (ht2+ht3+rtr1+rtr4, T={T:2d}): rel_err={err:.2e}")

    # 3b. UnifiedMoESide with within-expert row-group bit splitting (`n_rg=4`, `rg_rows=128`, `m_rows=512`)
    rg_slabs = []
    rg_map = np.zeros((128, 4, 2), dtype=np.int32)
    rg_fmts = [(1, "ht2", 2, 4, 512, 0x9E3779B1), (2, "ht3", 3, 2, 1024, 0x9E3779B1), (3, "rtr1", 1, 1, 4096, 0), (6, "rtr4", 4, 1, 4096, 0)]
    fmt_locs = {fid: [] for fid, *_ in rg_fmts}
    for e in range(128):
        for rg in range(4):
            fid, *_ = rg_fmts[(e + rg) % 4]
            loc = len(fmt_locs[fid])
            fmt_locs[fid].append((e, rg))
            rg_map[e, rg] = [fid, loc]
    for fid, fmt, k, g, K, hm in rg_fmts:
        cnt_f = len(fmt_locs[fid])
        c = mx.array(rng.integers(0, 2**32, (cnt_f, 128, (2304 * k) // 32), dtype=np.uint32))
        s = mx.array(rng.uniform(0.01, 0.05, (cnt_f, 128)).astype(np.float16))
        rg_slabs.append((fid, c, s, luts[fmt], (k, g, K, hm)))
    umoe_rg = UnifiedMoESide(rg_slabs, mx.array(rg_map[:, 0, 0]), mx.zeros((128,), dtype=mx.uint32), 512, 2304, fluts, rg_map=rg_map)
    for T, eids_np in [
        (1, np.array([0, 33, 66, 99, 5, 40, 70, 105], dtype=np.uint32)),
        (8, np.array([0, 32, 64, 96, 1, 33, 65, 97] * 6 + [0, 32, 64, 96, 2, 34, 66, 98] + [0, 32, 64, 96, 3, 35, 67, 99], dtype=np.uint32)),
        (16, rng.integers(0, 128, 128, dtype=np.uint32)),
    ]:
        xt = mx.array(rng.standard_normal((T, 2304)).astype(np.float32))
        ev = mx.array(eids_np)
        y_u = umoe_rg(xt.astype(mx.float16), ev, x_div=8)
        rg_parts = []
        sent = mx.array(0xFFFFFFFF, dtype=mx.uint32)
        for rg in range(4):
            y_rg = mx.zeros((T * 8, 128), dtype=mx.float32)
            rf_mx = mx.array(rg_map[:, rg, 0])
            rl_mx = mx.array(rg_map[:, rg, 1].astype(np.uint32))
            for fid, c, s, lv, (k, g, K, hm) in rg_slabs:
                lf = mx.where(rf_mx[ev] == fid, rl_mx[ev], sent)
                y_rg = y_rg + trellis_matvec(xt, c, s, lv, lf, 128, 2304, k, g, K, hm, x_div=8)
            rg_parts.append(y_rg)
        y_r = mx.concatenate(rg_parts, axis=-1)
        mx.eval(y_u, y_r)
        err_rg = float(mx.linalg.norm(y_u - y_r) / mx.linalg.norm(y_r))
        assert err_rg < 2e-3, f"UnifiedMoESide rg_map T={T} fail: {err_rg}"
        if T == 16:
            # Also verify non-MPP fallback prefill_v2 kernel explicitly!
            offsets, order = compute_moe_sorted_offsets(ev)
            units, ucnt, max_u = plan_moe_prefill_units(offsets, umoe_rg.exp_fid, umoe_rg.fids, int(ev.shape[0]), tile_t=32)
            kv2 = _make_unified_moe_prefill_v2_kernel(2304, umoe_rg.fids, sg=8, rb=2, n_rg=4)
            y_v2 = kv2(
                inputs=[xt.astype(mx.float16).reshape(-1), umoe_rg.codes, umoe_rg.scales, umoe_rg._kern_fid, umoe_rg.exp_code_off, umoe_rg.exp_scale_off, *umoe_rg._lut_args_sg8, units, ucnt, _u32(512, 8), order],
                grid=(max_u * 256, 4, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(T * 8, 512)],
                output_dtypes=[mx.float32],
            )[0]
            mx.eval(y_v2)
            err_v2 = float(mx.linalg.norm(y_v2 - y_r) / mx.linalg.norm(y_r))
            assert err_v2 < 2e-3, f"UnifiedMoESide prefill_v2 rg_map fail: {err_v2}"
        print(f"  PASS UnifiedMoESide rg_map n_rg=4 (T={T:2d}): rel_err={err_rg:.2e}")

    # 4. Sigmoid Top-8 Norm + Pre-Compacted Plan
    logits_f16 = mx.array(rng.standard_normal((8, 128)).astype(np.float16))
    rbias = mx.array(rng.standard_normal((128,)).astype(np.float32) * 0.1)
    e_fast, w_fast = sigmoid_top8_norm_fast(logits_f16, rbias, 8.0)
    sc_ref = mx.sigmoid(logits_f16.astype(mx.float32))
    bi_ref = sc_ref + rbias[None, :]
    top_ref = mx.argsort(-bi_ref, axis=-1)[:, :8]
    wr_ref = mx.take_along_axis(sc_ref, top_ref, axis=-1)
    wn_ref = (wr_ref / (mx.sum(wr_ref, axis=-1, keepdims=True) + 1e-20)) * 8.0
    mx.eval(e_fast, w_fast, top_ref, wn_ref)
    assert np.array_equal(np.array(e_fast).reshape(8, 8), np.array(top_ref)), "sigmoid_top8_norm eids mismatch!"
    err_r = float(mx.max(mx.abs(w_fast.reshape(8, 8) - wn_ref)))
    assert err_r < 1e-5, f"sigmoid_top8_norm w fail: {err_r}"
    e_p, w_p, gp, dp = sigmoid_top8_norm_and_plan_fast(logits_f16, rbias, 8.0, umoe.exp_fid, umoe.fids, umoe.exp_fid, umoe.fids)
    y_up = umoe(xt[:8].astype(mx.float16), e_p, x_div=8, plan=gp)
    y_un = umoe(xt[:8].astype(mx.float16), e_p, x_div=8)
    mx.eval(e_p, w_p, y_up, y_un)
    assert np.array_equal(np.array(e_p), np.array(e_fast)), "sigmoid_top8_plan eids mismatch!"
    assert float(mx.max(mx.abs(y_up - y_un))) < 1e-5, "pre-planned MoE mismatch!"
    print(f"  PASS sigmoid_top8_norm_and_plan_fast: exact top-8 + bit-exact MoE plan, max_err={err_r:.2e}")

    # 4b. Fused BAR+Norm+RHT and MsgBlend+Norm+RHT
    q1 = mx.array(rng.standard_normal((2304,)).astype(np.float32))
    blks = mx.array(rng.standard_normal((2, 8, 2304)).astype(np.float32))
    part = mx.array(rng.standard_normal((8, 2304)).astype(np.float32))
    nw = mx.array(rng.uniform(0.5, 1.5, (2304,)).astype(np.float32))
    gs = mx.array(rng.choice([-1.0, 1.0], (2304,)).astype(np.float32))
    hf, xrf = block_attn_res_norm_rht_fast(q1, blks, 2, part, nw, gs)
    href = block_attn_res_fused(q1, blks, 2, part, norm_w=nw)[0]
    xrref = rht_in_2304_fast(href, gs, 1, 8, out_dtype=mx.float16)
    mx.eval(hf, xrf, href, xrref)
    eb1 = float(mx.linalg.norm(hf.astype(mx.float32) - href) / mx.linalg.norm(href))
    eb2 = float(mx.linalg.norm(xrf.astype(mx.float32) - xrref.astype(mx.float32)) / mx.linalg.norm(xrref.astype(mx.float32)))
    assert eb1 < 1e-3 and eb2 < 1e-3, f"bar_norm_rht fail: {eb1}, {eb2}"
    print(f"  PASS block_attn_res_norm_rht_fast: rel_err_h={eb1:.2e}, rel_err_rot={eb2:.2e}")

    msg1 = mx.array(rng.standard_normal((8, 2304)).astype(np.float32))
    fgmg = mx.array(rng.standard_normal((8, 4608)).astype(np.float32)).astype(mx.bfloat16)
    mf_h, mf_r = msg_blend_norm_rht_fast(part, msg1, fgmg, nw, gs)
    mref = msg_blend_norm_fused(part, msg1, fgmg, nw)
    mrref = rht_in_2304_fast(mref, gs, 1, 8, out_dtype=mx.float16)
    mx.eval(mf_h, mf_r, mref, mrref)
    em1 = float(mx.linalg.norm(mf_h.astype(mx.float32) - mref) / mx.linalg.norm(mref))
    em2 = float(mx.linalg.norm(mf_r.astype(mx.float32) - mrref.astype(mx.float32)) / mx.linalg.norm(mrref.astype(mx.float32)))
    assert em1 < 1e-3 and em2 < 1e-3, f"msg_blend_norm_rht fail: {em1}, {em2}"
    print(f"  PASS msg_blend_norm_rht_fast: rel_err_h={em1:.2e}, rel_err_rot={em2:.2e}")

    # 5. Dynamic KV sconv4 + norm fused
    T = 8
    k_prev = mx.array(rng.standard_normal((1, 3, 1024)).astype(np.float32))
    v_prev = mx.array(rng.standard_normal((1, 3, 1024)).astype(np.float32))
    k_raw = mx.array(rng.standard_normal((T, 1024)).astype(np.float32))
    v_raw = mx.array(rng.standard_normal((T, 1024)).astype(np.float32))
    k_in = mx.array(rng.standard_normal((T, 2304)).astype(np.float32))
    v_in = mx.array(rng.standard_normal((T, 2304)).astype(np.float32))
    k_st = mx.array(rng.standard_normal((4, 1024)).astype(np.float32))
    v_st = mx.array(rng.standard_normal((4, 1024)).astype(np.float32))
    k_pr = mx.array(rng.standard_normal((32, 2304)).astype(np.float32))
    v_pr = mx.array(rng.standard_normal((32, 2304)).astype(np.float32))
    kvf, knp, vnp = kv_dyn_sconv4_norm_fused(k_prev, k_raw, k_in, k_st, k_pr.astype(mx.float16), v_prev, v_raw, v_in, v_st, v_pr.astype(mx.float16), T, 22.6274)
    kf, vf = kvf[0], kvf[1]
    # Reference
    k_ctx = mx.concatenate([k_prev, k_raw[None, :, :]], axis=1)
    v_ctx = mx.concatenate([v_prev, v_raw[None, :, :]], axis=1)
    k_dw = k_st.T.reshape(1, 1, 8, 128, 4) + (k_in[None, :, :] @ k_pr.T).reshape(1, T, 8, 1, 4)
    v_dw = v_st.T.reshape(1, 1, 8, 128, 4) + (v_in[None, :, :] @ v_pr.T).reshape(1, T, 8, 1, 4)
    k_acc = (sum(k_ctx[:, (3 - lg) : (3 - lg + T)].reshape(1, T, 8, 128) * k_dw[..., lg] for lg in range(4)) + k_raw.reshape(1, T, 8, 128)) / 3.0
    v_acc = (sum(v_ctx[:, (3 - lg) : (3 - lg + T)].reshape(1, T, 8, 128) * v_dw[..., lg] for lg in range(4)) + v_raw.reshape(1, T, 8, 128)) / 3.0
    kn = mx.sqrt(mx.sum(k_acc * k_acc, axis=-1, keepdims=True))
    k_ref = (k_acc * mx.minimum(22.6274 / (kn + 1e-6), 1.0))[0].transpose(1, 0, 2)
    v_ref = mx.fast.rms_norm(v_acc, None, 1e-6)[0].transpose(1, 0, 2)
    mx.eval(kf, vf, k_ref, v_ref, knp, vnp)
    ek = float(mx.linalg.norm(kf.astype(mx.float32) - k_ref) / mx.linalg.norm(k_ref))
    ev_err = float(mx.linalg.norm(vf.astype(mx.float32) - v_ref) / mx.linalg.norm(v_ref))
    assert ek < 2e-3 and ev_err < 2e-3, f"kv_dyn_sconv4_norm fail: ek={ek}, ev={ev_err}"
    print(f"  PASS kv_dyn_sconv4_norm_fused (unified KV): rel_err_k={ek:.2e}, rel_err_v={ev_err:.2e}")

    print("ALL fast_kernels selftests PASSED!", flush=True)


if __name__ == "__main__":
    _selftest()
