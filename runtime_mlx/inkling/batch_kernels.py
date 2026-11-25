"""Batched & paged-KV Metal kernels for multi-agent Inkling-2-Mini serving on Apple Silicon MLX.

Provides:
  1. Batched 4-tap causal conv kernels (`batch_k >= 1`, `T = B * batch_k`, zero cross-sequence lag):
     - `qkv_rel_prep_sliding_batched`  (sliding visits -> paged `[T, 2, 8, 128]` KV layout)
     - `kv_sconv4_norm_batched`        (static KV -> paged `[T, 2, 8, 128]` KV layout)
     - `kv_dyn_sconv4_norm_batched`    (global dynamic KV -> paged `[T, 2, 8, 128]` KV layout)
     - `short_conv4_res_batched`       (post-attn 4-tap conv + residual)
     - `moe_wsum_sconv4_res_batched`   (fused MoE weighted sum + post-MLP conv + residual)
  2. Paged Split-K & FlashAttention-2 GQA+RelPos+Sinks SDPA (`gqa_paged_sdpa_fast`):
     - Reads 64-token physical KV blocks via `block_table[B, max_blocks]` (`fp16`, `fp8`, `q4`).
     - Decode & speculative verify (`K <= 15` or `B > 1`): Split-K paged SDPA with `(kvh, split_id)`-outer
       / `t_global`-inner grid order for hardware L2 broadcast of shared physical prefix blocks.
     - Prefill (`B == 1, K >= 16`): tiled `simdgroup_half8x8` FlashAttention-2 MMA (`BQ=32, BK=64`)
       reading directly from paged physical KV blocks with zero copy.
"""
from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

from .fast_kernels import (
    _FP8_E4M3_HEADER,
    _H128_HEADER,
    _f32_arr,
    _make_gqa_splitk_reduce_kernel,
    _u32,
    h128_rows_f32,
)

PAGED_BLOCK_SIZE = 64
_BLOCK_SHIFT = 6
_BLOCK_MASK = 63


# ----------------------------------------------------------------------------- Batched 4-Tap Conv Kernels


@lru_cache(maxsize=2)
def _make_qkv_rel_prep_sliding_batched_kernel():
    """Batched Q+Rel clip + K/V static 4-tap conv for `B` sequences of `K` tokens -> `kv_out[T, 2, 8, 128]`."""
    src = """
    const uint ht = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x; // 0..127
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint T = meta[0], K = meta[1];
    const float qk_thresh = cfg[0], r_thresh = cfg[1];
    threadgroup float red[4], red_v[4];

    if (ht < 24u * T) {
        const uint t = ht / 24u;
        const uint h = ht - t * 24u;
        const device float* row = y_qkv + (ulong)t * 8576u;
        const float qv = row[h * 128u + tid] * q_scale[h * 128u + tid];
        float ss = simd_sum(qv * qv);
        if (lane == 0u) red[simd] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float q_norm = metal::precise::sqrt(red[0] + red[1] + red[2] + red[3]);
        q_out[((ulong)t * 24u + h) * 128u + tid] = qv * min(qk_thresh / (q_norm + 1e-6f), 1.0f);

        if (tid < 16u) {
            const float rv = row[5120u + h * 16u + tid] * rel_scale[h * 16u + tid];
            float rs = rv * rv;
            rs += simd_shuffle_xor(rs, 8u); rs += simd_shuffle_xor(rs, 4u); rs += simd_shuffle_xor(rs, 2u); rs += simd_shuffle_xor(rs, 1u);
            rel_out[((ulong)t * 24u + h) * 16u + tid] = rv * min(r_thresh / (metal::precise::sqrt(rs) + 1e-6f), 1.0f);
        }
    } else {
        const uint ht2 = ht - 24u * T;
        const uint t = ht2 / 8u;
        const uint h = ht2 - t * 8u;
        const uint b = t / K;
        const uint lt = t - b * K;
        const uint b0 = b * K;
        const uint ch = h * 128u + tid;
        constexpr float INV_SQRT5 = 0.4472135954999579f;
        const ulong kv_t_base = (ulong)t * 2048u;

        // K static 4-tap conv + L2Clip -> kv_out[t, 0, h, tid]
        float k_acc = y_qkv[(ulong)t * 8576u + 3072u + ch];
        #pragma unroll
        for (uint k = 0; k < 4u; k++) {
            const int idx = (int)lt + (int)k - 3;
            const float v = (idx >= 0)
                ? y_qkv[(ulong)(b0 + (uint)idx) * 8576u + 3072u + ch]
                : k_prev[(ulong)(b * 3u + (uint)(idx + 3)) * 1024u + ch];
            k_acc += v * k_w[(ulong)k * 1024u + ch];
        }
        k_acc *= INV_SQRT5;
        float kss = simd_sum(k_acc * k_acc);
        if (lane == 0u) red[simd] = kss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float k_norm = metal::precise::sqrt(red[0] + red[1] + red[2] + red[3]);
        kv_out[kv_t_base + ch] = (half)(k_acc * min(qk_thresh / (k_norm + 1e-6f), 1.0f));

        // V static 4-tap conv + UnweightedRMSNorm -> kv_out[t, 1, h, tid]
        float v_acc = y_qkv[(ulong)t * 8576u + 4096u + ch];
        #pragma unroll
        for (uint k = 0; k < 4u; k++) {
            const int idx = (int)lt + (int)k - 3;
            const float v = (idx >= 0)
                ? y_qkv[(ulong)(b0 + (uint)idx) * 8576u + 4096u + ch]
                : v_prev[(ulong)(b * 3u + (uint)(idx + 3)) * 1024u + ch];
            v_acc += v * v_w[(ulong)k * 1024u + ch];
        }
        v_acc *= INV_SQRT5;
        float vss = simd_sum(v_acc * v_acc);
        if (lane == 0u) red_v[simd] = vss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float v_ir = metal::precise::rsqrt((red_v[0] + red_v[1] + red_v[2] + red_v[3]) * (1.0f / 128.0f) + 1e-6f);
        kv_out[kv_t_base + 1024u + ch] = (half)(v_acc * v_ir);

        if (lt == 0u) {
            #pragma unroll
            for (uint j = 0; j < 3u; j++) {
                const int idx = (int)K - 3 + (int)j;
                const ulong dst = (ulong)(b * 3u + j) * 1024u + ch;
                k_new_prev[dst] = (idx >= 0)
                    ? y_qkv[(ulong)(b0 + (uint)idx) * 8576u + 3072u + ch]
                    : k_prev[(ulong)(b * 3u + (uint)(idx + 3)) * 1024u + ch];
                v_new_prev[dst] = (idx >= 0)
                    ? y_qkv[(ulong)(b0 + (uint)idx) * 8576u + 4096u + ch]
                    : v_prev[(ulong)(b * 3u + (uint)(idx + 3)) * 1024u + ch];
            }
        }
    }
    """
    return mx.fast.metal_kernel(
        name="ink_qkv_rel_prep_sl_batched_v1",
        input_names=["y_qkv", "q_scale", "rel_scale", "k_prev", "k_w", "v_prev", "v_w", "meta", "cfg"],
        output_names=["q_out", "rel_out", "kv_out", "k_new_prev", "v_new_prev"],
        source=src,
    )


def qkv_rel_prep_sliding_batched(
    y_qkv: mx.array,
    q_scale: mx.array,
    rel_scale: mx.array,
    k_prev: mx.array,
    k_w: mx.array,
    v_prev: mx.array,
    v_w: mx.array,
    B: int,
    K: int,
    qk_thresh: float,
    rel_thresh: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    T = B * K
    return _make_qkv_rel_prep_sliding_batched_kernel()(
        inputs=[
            y_qkv.reshape(-1), q_scale.reshape(-1), rel_scale.reshape(-1),
            k_prev.reshape(-1), k_w.reshape(-1), v_prev.reshape(-1), v_w.reshape(-1),
            _u32(T, K), _f32_arr(qk_thresh, rel_thresh),
        ],
        grid=(128, 32 * T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(T, 24, 128), (T, 24, 16), (T, 2, 8, 128), (B, 3, 1024), (B, 3, 1024)],
        output_dtypes=[mx.float32, mx.float32, mx.float16, mx.float32, mx.float32],
    )


@lru_cache(maxsize=2)
def _make_kv_sconv4_norm_batched_kernel():
    """Batched static 4-tap KV conv + norm -> paged `kv_out[T, 2, 8, 128]`."""
    src = """
    const uint ht = threadgroup_position_in_grid.y;
    const uint t = ht / 8u;
    const uint h = ht - t * 8u;
    const uint d = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint T = meta[0], K = meta[1];
    const uint b = t / K;
    const uint lt = t - b * K;
    const uint b0 = b * K;
    const uint ch = h * 128u + d;
    const float k_thresh = cfg[0];
    constexpr float INV_SQRT5 = 0.4472135954999579f;
    const ulong kv_t_base = (ulong)t * 2048u;

    float k_acc = k_raw[(ulong)t * 1024u + ch];
    #pragma unroll
    for (uint k = 0; k < 4u; k++) {
        const int idx = (int)lt + (int)k - 3;
        const float v = (idx >= 0) ? k_raw[(ulong)(b0 + (uint)idx) * 1024u + ch] : k_prev[(ulong)(b * 3u + (uint)(idx + 3)) * 1024u + ch];
        k_acc += v * k_w[(ulong)k * 1024u + ch];
    }
    k_acc *= INV_SQRT5;
    threadgroup float red_k[4], red_v[4];
    float kss = simd_sum(k_acc * k_acc);
    if (lane == 0u) red_k[simd] = kss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float k_norm = metal::precise::sqrt(red_k[0] + red_k[1] + red_k[2] + red_k[3]);
    kv_out[kv_t_base + ch] = (half)(k_acc * min(k_thresh / (k_norm + 1e-6f), 1.0f));

    float v_acc = v_raw[(ulong)t * 1024u + ch];
    #pragma unroll
    for (uint k = 0; k < 4u; k++) {
        const int idx = (int)lt + (int)k - 3;
        const float v = (idx >= 0) ? v_raw[(ulong)(b0 + (uint)idx) * 1024u + ch] : v_prev[(ulong)(b * 3u + (uint)(idx + 3)) * 1024u + ch];
        v_acc += v * v_w[(ulong)k * 1024u + ch];
    }
    v_acc *= INV_SQRT5;
    float vss = simd_sum(v_acc * v_acc);
    if (lane == 0u) red_v[simd] = vss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float v_ir = metal::precise::rsqrt((red_v[0] + red_v[1] + red_v[2] + red_v[3]) * (1.0f / 128.0f) + 1e-6f);
    kv_out[kv_t_base + 1024u + ch] = (half)(v_acc * v_ir);

    if (lt == 0u) {
        #pragma unroll
        for (uint j = 0; j < 3u; j++) {
            const int idx = (int)K - 3 + (int)j;
            const ulong dst = (ulong)(b * 3u + j) * 1024u + ch;
            k_new_prev[dst] = (idx >= 0) ? k_raw[(ulong)(b0 + (uint)idx) * 1024u + ch] : k_prev[(ulong)(b * 3u + (uint)(idx + 3)) * 1024u + ch];
            v_new_prev[dst] = (idx >= 0) ? v_raw[(ulong)(b0 + (uint)idx) * 1024u + ch] : v_prev[(ulong)(b * 3u + (uint)(idx + 3)) * 1024u + ch];
        }
    }
    """
    return mx.fast.metal_kernel(
        name="ink_kv_sconv4_batched_v1",
        input_names=["k_prev", "k_raw", "k_w", "v_prev", "v_raw", "v_w", "meta", "cfg"],
        output_names=["kv_out", "k_new_prev", "v_new_prev"],
        source=src,
    )


def kv_sconv4_norm_batched(
    k_prev: mx.array,
    k_raw: mx.array,
    k_w: mx.array,
    v_prev: mx.array,
    v_raw: mx.array,
    v_w: mx.array,
    B: int,
    K: int,
    qk_thresh: float,
) -> tuple[mx.array, mx.array, mx.array]:
    T = B * K
    return _make_kv_sconv4_norm_batched_kernel()(
        inputs=[
            k_prev.reshape(-1), k_raw.reshape(-1), k_w.reshape(-1),
            v_prev.reshape(-1), v_raw.reshape(-1), v_w.reshape(-1),
            _u32(T, K), _f32_arr(qk_thresh),
        ],
        grid=(128, 8 * T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(T, 2, 8, 128), (B, 3, 1024), (B, 3, 1024)],
        output_dtypes=[mx.float16, mx.float32, mx.float32],
    )


@lru_cache(maxsize=2)
def _make_kv_dyn_sconv4_norm_batched_kernel():
    """Batched dynamic 4-tap KV conv + norm (global visits) -> paged `kv_out[T, 2, 8, 128]`."""
    src = """
    const uint ht = threadgroup_position_in_grid.y;
    const uint t = ht / 8u;
    const uint h = ht - t * 8u;
    const uint d = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint T = meta[0], K = meta[1];
    const uint b = t / K;
    const uint lt = t - b * K;
    const uint b0 = b * K;
    const uint ch = h * 128u + d;
    const float k_thresh = cfg[0];
    constexpr float INV_3 = 0.3333333333333333f;
    const ulong kv_t_base = (ulong)t * 2048u;

    const device float2* kin2 = (const device float2*)(k_in + (ulong)t * 2304u);
    const device float2* vin2 = (const device float2*)(v_in + (ulong)t * 2304u);
    float2 ki_reg[9], vi_reg[9];
    #pragma unroll
    for (uint i = 0; i < 9u; i++) {
        const uint idx = d + i * 128u;
        ki_reg[i] = kin2[idx];
        vi_reg[i] = vin2[idx];
    }
    threadgroup float tg_dk[4][4], tg_dv[4][4];
    #pragma unroll
    for (uint lag = 0; lag < 4u; lag++) {
        const device half2* kpw = (const device half2*)(k_pr + (ulong)(h * 4u + lag) * 2304u);
        const device half2* vpw = (const device half2*)(v_pr + (ulong)(h * 4u + lag) * 2304u);
        float kd = 0.0f, vd = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 9u; i++) {
            const uint idx = d + i * 128u;
            kd += dot(ki_reg[i], float2(kpw[idx]));
            vd += dot(vi_reg[i], float2(vpw[idx]));
        }
        kd = simd_sum(kd);
        vd = simd_sum(vd);
        if (lane == 0u) { tg_dk[lag][simd] = kd; tg_dv[lag][simd] = vd; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float dk[4], dv[4];
    #pragma unroll
    for (uint lag = 0; lag < 4u; lag++) {
        dk[lag] = tg_dk[lag][0] + tg_dk[lag][1] + tg_dk[lag][2] + tg_dk[lag][3];
        dv[lag] = tg_dv[lag][0] + tg_dv[lag][1] + tg_dv[lag][2] + tg_dv[lag][3];
    }

    float k_acc = k_raw[(ulong)t * 1024u + ch];
    #pragma unroll
    for (uint lag = 0; lag < 4u; lag++) {
        const int idx = (int)lt - (int)lag;
        const float v = (idx >= 0) ? k_raw[(ulong)(b0 + (uint)idx) * 1024u + ch] : k_prev[(ulong)(b * 3u + (uint)(idx + 3)) * 1024u + ch];
        k_acc += v * (k_st[(ulong)lag * 1024u + ch] + dk[lag]);
    }
    k_acc *= INV_3;
    threadgroup float red_k[4], red_v[4];
    float kss = simd_sum(k_acc * k_acc);
    if (lane == 0u) red_k[simd] = kss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float k_norm = metal::precise::sqrt(red_k[0] + red_k[1] + red_k[2] + red_k[3]);
    kv_out[kv_t_base + ch] = (half)(k_acc * min(k_thresh / (k_norm + 1e-6f), 1.0f));

    float v_acc = v_raw[(ulong)t * 1024u + ch];
    #pragma unroll
    for (uint lag = 0; lag < 4u; lag++) {
        const int idx = (int)lt - (int)lag;
        const float v = (idx >= 0) ? v_raw[(ulong)(b0 + (uint)idx) * 1024u + ch] : v_prev[(ulong)(b * 3u + (uint)(idx + 3)) * 1024u + ch];
        v_acc += v * (v_st[(ulong)lag * 1024u + ch] + dv[lag]);
    }
    v_acc *= INV_3;
    float vss = simd_sum(v_acc * v_acc);
    if (lane == 0u) red_v[simd] = vss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float v_ir = metal::precise::rsqrt((red_v[0] + red_v[1] + red_v[2] + red_v[3]) * (1.0f / 128.0f) + 1e-6f);
    kv_out[kv_t_base + 1024u + ch] = (half)(v_acc * v_ir);

    if (lt == 0u) {
        #pragma unroll
        for (uint j = 0; j < 3u; j++) {
            const int idx = (int)K - 3 + (int)j;
            const ulong dst = (ulong)(b * 3u + j) * 1024u + ch;
            k_new_prev[dst] = (idx >= 0) ? k_raw[(ulong)(b0 + (uint)idx) * 1024u + ch] : k_prev[(ulong)(b * 3u + (uint)(idx + 3)) * 1024u + ch];
            v_new_prev[dst] = (idx >= 0) ? v_raw[(ulong)(b0 + (uint)idx) * 1024u + ch] : v_prev[(ulong)(b * 3u + (uint)(idx + 3)) * 1024u + ch];
        }
    }
    """
    return mx.fast.metal_kernel(
        name="ink_kv_dyn_sconv4_batched_v1",
        input_names=["k_prev", "k_raw", "k_in", "k_st", "k_pr", "v_prev", "v_raw", "v_in", "v_st", "v_pr", "meta", "cfg"],
        output_names=["kv_out", "k_new_prev", "v_new_prev"],
        source=src,
    )


def kv_dyn_sconv4_norm_batched(
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
    B: int,
    K: int,
    qk_thresh: float,
) -> tuple[mx.array, mx.array, mx.array]:
    T = B * K
    return _make_kv_dyn_sconv4_norm_batched_kernel()(
        inputs=[
            k_prev.reshape(-1), k_raw.reshape(-1), k_in.reshape(-1), k_st.reshape(-1), k_pr_f16.reshape(-1),
            v_prev.reshape(-1), v_raw.reshape(-1), v_in.reshape(-1), v_st.reshape(-1), v_pr_f16.reshape(-1),
            _u32(T, K), _f32_arr(qk_thresh),
        ],
        grid=(128, 8 * T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(T, 2, 8, 128), (B, 3, 1024), (B, 3, 1024)],
        output_dtypes=[mx.float16, mx.float32, mx.float32],
    )


@lru_cache(maxsize=4)
def _make_short_conv4_res_batched_kernel(with_res: bool):
    in_names = ["prev", "x", "w", "res", "meta"] if with_res else ["prev", "x", "w", "meta"]
    add_res = " + res[(ulong)t * C + c]" if with_res else ""
    src = f"""
    const uint c = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const uint C = meta[0], K = meta[1];
    if (c >= C) return;
    const uint b = t / K;
    const uint lt = t - b * K;
    const uint b0 = b * K;
    float acc = x[(ulong)t * C + c];
    #pragma unroll
    for (uint k = 0; k < 4u; k++) {{
        const int idx = (int)lt + (int)k - 3;
        const float v = (idx >= 0) ? x[(ulong)(b0 + (uint)idx) * C + c] : prev[(ulong)(b * 3u + (uint)(idx + 3)) * C + c];
        acc += v * w[(ulong)k * C + c];
    }}
    out[(ulong)t * C + c] = acc * 0.4472135954999579f{add_res};
    if (lt == 0u) {{
        #pragma unroll
        for (uint j = 0; j < 3u; j++) {{
            const int idx = (int)K - 3 + (int)j;
            new_prev[(ulong)(b * 3u + j) * C + c] = (idx >= 0)
                ? x[(ulong)(b0 + (uint)idx) * C + c]
                : prev[(ulong)(b * 3u + (uint)(idx + 3)) * C + c];
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_sconv4_batched_r{int(with_res)}",
        input_names=in_names,
        output_names=["out", "new_prev"],
        source=src,
    )


def short_conv4_res_batched(
    prev: mx.array,
    x: mx.array,
    w: mx.array,
    B: int,
    K: int,
    C: int,
    residual: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    T = B * K
    gx = ((C + 255) // 256) * 256
    mt = _u32(C, K)
    if residual is not None:
        return _make_short_conv4_res_batched_kernel(True)(
            inputs=[prev.reshape(-1), x.reshape(-1), w.reshape(-1), residual.reshape(-1), mt],
            grid=(gx, T, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(T, C), (B, 3, C)],
            output_dtypes=[mx.float32, mx.float32],
        )
    return _make_short_conv4_res_batched_kernel(False)(
        inputs=[prev.reshape(-1), x.reshape(-1), w.reshape(-1), mt],
        grid=(gx, T, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, C), (B, 3, C)],
        output_dtypes=[mx.float32, mx.float32],
    )


@lru_cache(maxsize=2)
def _make_moe_wsum_sconv4_res_batched_kernel():
    """Vectorized float4 batched MoE weighted-sum + 4-tap conv + residual for `B` sequences of `K` tokens."""
    src = """
    const uint c4 = thread_position_in_grid.x; // 0..575
    const uint b = threadgroup_position_in_grid.y;
    const uint K = meta[0];
    if (c4 >= 576u) return;

    const device float4* y4 = (const device float4*)y_dn;
    const device float4* pv4 = (const device float4*)prev;
    const device float4* cw4 = (const device float4*)conv_w;
    const device float4* res4 = (const device float4*)residual;
    device float4* out4 = (device float4*)out;
    device float4* np4 = (device float4*)new_prev;
    device float4* raw4 = (device float4*)raw_out;

    const ulong pv_base = (ulong)b * 3u * 576u + c4;
    float4 p0 = pv4[pv_base];
    float4 p1 = pv4[pv_base + 576u];
    float4 p2 = pv4[pv_base + 1152u];
    const float4 w0 = cw4[c4], w1 = cw4[576u + c4], w2 = cw4[1152u + c4], w3 = cw4[1728u + c4];
    constexpr float INV_SQRT5 = 0.4472135954999579f;
    const ulong t0 = (ulong)b * K;

    for (uint lt = 0u; lt < K; lt++) {
        const ulong t = t0 + lt;
        const ulong bs = t * 8u;
        float4 s = float4(0.0f);
        #pragma unroll
        for (uint k = 0u; k < 8u; k++) {
            const float wk = w_norm[bs + k];
            if (wk != 0.0f) s += y4[(bs + k) * 576u + c4] * wk;
        }
        const float4 conv_out = (p0 * w0 + p1 * w1 + p2 * w2 + s * w3 + s) * INV_SQRT5;
        raw4[t * 576u + c4] = s;
        out4[t * 576u + c4] = res4[t * 576u + c4] + conv_out;
        p0 = p1; p1 = p2; p2 = s;
    }
    np4[pv_base] = p0;
    np4[pv_base + 576u] = p1;
    np4[pv_base + 1152u] = p2;
    """
    return mx.fast.metal_kernel(
        name="ink_moe_wsum_sconv4_res_batched_v1",
        input_names=["y_dn", "w_norm", "prev", "conv_w", "residual", "meta"],
        output_names=["out", "new_prev", "raw_out"],
        source=src,
    )


def moe_wsum_sconv4_res_batched(
    y_dn: mx.array,
    w_norm: mx.array,
    prev: mx.array,
    conv_w: mx.array,
    residual: mx.array,
    B: int,
    K: int,
) -> tuple[mx.array, mx.array, mx.array]:
    T = B * K
    return _make_moe_wsum_sconv4_res_batched_kernel()(
        inputs=[y_dn.reshape(-1), w_norm.reshape(-1), prev.reshape(-1), conv_w.reshape(-1), residual.reshape(-1), _u32(K)],
        grid=(576, B, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(T, 2304), (B, 3, 2304), (B, K, 2304)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )


# ----------------------------------------------------------------------------- Paged Split-K & Shared-Prefix SDPA


def _paged_kv_load_snippet(kv_mode: str, pt_expr: str) -> str:
    """Load K and V for physical token index `pt_expr` and KV head `kvh` into `tg_k/tg_v[r*32 + lane]`."""
    if kv_mode == "q4":
        return (
            f"const ulong pt_u = (ulong)({pt_expr}); "
            "const ulong k_off = pt_u * 1024u + (ulong)kvh * 64u; "
            "const ulong v_off = pt_u * 1024u + (ulong)(8u + kvh) * 64u; "
            "const ulong ks_off = pt_u * 64u + (ulong)kvh * 4u + (lane >> 3); "
            "const ulong vs_off = pt_u * 64u + (ulong)(8u + kvh) * 4u + (lane >> 3); "
            "const uchar2 kp2 = *((const device uchar2*)(KV + k_off) + lane); "
            "const half ksc = KV_sc[ks_off]; "
            "const uchar2 vp2 = *((const device uchar2*)(KV + v_off) + lane); "
            "const half vsc = KV_sc[vs_off]; "
            "tg_k[r * 32u + lane] = half4(half(int(kp2.x & 0xFu) - 8) * ksc, half(int(kp2.x >> 4) - 8) * ksc, half(int(kp2.y & 0xFu) - 8) * ksc, half(int(kp2.y >> 4) - 8) * ksc); "
            "tg_v[r * 32u + lane] = half4(half(int(vp2.x & 0xFu) - 8) * vsc, half(int(vp2.x >> 4) - 8) * vsc, half(int(vp2.y & 0xFu) - 8) * vsc, half(int(vp2.y >> 4) - 8) * vsc);"
        )
    if kv_mode == "fp8":
        return (
            f"const ulong pt_u = (ulong)({pt_expr}); "
            "const ulong k_off = pt_u * 2048u + (ulong)kvh * 128u; "
            "const ulong v_off = pt_u * 2048u + (ulong)(8u + kvh) * 128u; "
            "const uchar4 ku = *((const device uchar4*)(KV + k_off) + lane); "
            "const uchar4 vu = *((const device uchar4*)(KV + v_off) + lane); "
            "tg_k[r * 32u + lane] = half4(FP8_E4M3_LUT[ku.x], FP8_E4M3_LUT[ku.y], FP8_E4M3_LUT[ku.z], FP8_E4M3_LUT[ku.w]); "
            "tg_v[r * 32u + lane] = half4(FP8_E4M3_LUT[vu.x], FP8_E4M3_LUT[vu.y], FP8_E4M3_LUT[vu.z], FP8_E4M3_LUT[vu.w]);"
        )
    return (
        f"const ulong pt_u = (ulong)({pt_expr}); "
        "const ulong k_off = pt_u * 2048u + (ulong)kvh * 128u; "
        "const ulong v_off = pt_u * 2048u + (ulong)(8u + kvh) * 128u; "
        "tg_k[r * 32u + lane] = *((const device half4*)(KV + k_off) + lane); "
        "tg_v[r * 32u + lane] = *((const device half4*)(KV + v_off) + lane);"
    )


@lru_cache(maxsize=16)
def _make_gqa_paged_sdpa_kernel(kv_mode: str = "fp16", split_out: bool = True, kt_cap: int = 1):
    """Paged-KV batched GQA+RelPos+Sinks SDPA kernel supporting `B` sequences x `K` tokens (`kt_cap` = 1 or 4).

    Grid: `(128, B * n_qt * 8 * n_splits, 1)` where `n_qt = (K + kt_cap - 1) / kt_cap`.
    When `split_out=True`, writes `(part_m, part_l, part_o)` shaped `[n_splits, T, 24(, 128)]`
    compatible with `_make_gqa_splitk_reduce_kernel`.
    When `split_out=False` (`n_splits == 1`, `kt_cap == 1`), merges sinks inline into `out[T, 3072]`.
    """
    assert kt_cap in (1, 4)
    in_names = ["q", "KV"]
    if kv_mode == "q4":
        in_names.append("KV_sc")
    in_names.extend(["rel_st", "rel_proj_t"])
    if not split_out:
        in_names.extend(["q_in", "sink_w"])
    in_names.extend(["block_table", "seq_lens", "q_pos0", "kv_starts", "meta"])

    header = "#define LOG2E 1.4426950408889634f\n"
    if kv_mode == "fp8":
        header += _FP8_E4M3_HEADER
    elif kv_mode == "q4":
        header += _H128_HEADER
    q_rot = "qv[g] = h128_vec(qv[g], lane);" if kv_mode == "q4" else ""
    q_rot_k = "qv[u][g] = h128_vec(qv[u][g], lane);" if kv_mode == "q4" else ""
    unrot = "h128_vec(a_out, lane)" if kv_mode == "q4" else "a_out"
    out_names = ["part_m", "part_l", "part_o"] if split_out else ["out"]
    load_snip = _paged_kv_load_snippet(kv_mode, "block_table[bt_off + (j_cur >> 6)] * 64u + (j_cur & 63u)")

    if kt_cap == 1:
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
    }
"""
        else:
            epilogue = f"""
    float sk_arr[3];
    const device float2* qin2 = (const device float2*)(q_in + (ulong)t_global * 2304u);
    for (uint g = 0u; g < 3u; g++) {{
        const uint h = kvh * 3u + g;
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
            *((device float4*)(out + ((ulong)t_global * 24u + h) * 128u) + lane) = {unrot};
        }}
    }}
"""
        src = f"""
    // meta: [0]=T, [1]=K, [2]=max_blocks, [3]=is_sliding, [4]=n_splits, [5]=j_min_override
    // Grid order: (kvh, split_id) outer, t_global inner so B agents sharing prefix blocks hit L2 together!
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
        qv[g] = *((const device float4*)(q + ((ulong)t_global * 24u + h) * 128u) + lane) * (LOG2E / 11.313708498984761f);
        {q_rot}
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

    threadgroup half4 tg_k[16u * 32u];
    threadgroup half4 tg_v[16u * 32u];

    for (uint tile0 = sbeg; tile0 < send; tile0 += 16u) {{
        const uint r_hi = min(16u, send - tile0);
        #pragma unroll
        for (uint p = 0u; p < 4u; p++) {{
            const uint r = simd + p * 4u;
            if (r < r_hi) {{
                const uint j_cur = tile0 + r;
                {load_snip}
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
    {epilogue}
"""
    else:
        # kt_cap == 4: each threadgroup processes up to 4 contiguous query tokens of one sequence `b`
        # (reuses loaded tg_k/tg_v across all <= 4 tokens, always outputs split buffers for reduce kernel!)
        src = f"""
    // meta: [0]=T, [1]=K, [2]=max_blocks, [3]=is_sliding, [4]=n_splits, [5]=j_min_override, [6]=n_qt
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
            qv[u][g] = *((const device float4*)(q + ((ulong)tg * 24u + h) * 128u) + lane) * (LOG2E / 11.313708498984761f);
            {q_rot_k}
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

    threadgroup half4 tg_k[16u * 32u];
    threadgroup half4 tg_v[16u * 32u];

    for (uint tile0 = sbeg; tile0 < send; tile0 += 16u) {{
        const uint r_hi = min(16u, send - tile0);
        #pragma unroll
        for (uint p = 0u; p < 4u; p++) {{
            const uint r = simd + p * 4u;
            if (r < r_hi) {{
                const uint j_cur = tile0 + r;
                {load_snip}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = simd; r < r_hi; r += 4u) {{
            const uint j = tile0 + r;
            const float4 kv = float4(tg_k[r * 32u + lane]);
            const float4 vv = float4(tg_v[r * 32u + lane]);
            for (uint u = 0u; u < n_tok; u++) {{
                if (j < j_lo_t[u] || j >= j_hi_t[u]) continue;
                const uint dist = q_pos_t[u] - (kv_start + j);
                half4 rp0 = half4(0.0h), rp1 = half4(0.0h), rp2 = half4(0.0h), rp3 = half4(0.0h);
                const bool has_rel = dist < 1024u;
                if (has_rel) {{
                    const device half4* rp = (const device half4*)(rel_proj_t + (ulong)dist * 16u);
                    rp0 = rp[0]; rp1 = rp[1]; rp2 = rp[2]; rp3 = rp[3];
                }}
                #pragma unroll
                for (uint g = 0u; g < 3u; g++) {{
                    float score = simd_sum(dot(qv[u][g], kv));
                    if (has_rel) {{
                        score += dot(rs0[u][g], float4(rp0)) + dot(rs1[u][g], float4(rp1)) + dot(rs2[u][g], float4(rp2)) + dot(rs3[u][g], float4(rp3));
                    }}
                    const float m_new = max(m_loc[u][g], score);
                    const float alpha = exp2(m_loc[u][g] - m_new);
                    const float p_w = exp2(score - m_new);
                    l_loc[u][g] = l_loc[u][g] * alpha + p_w;
                    acc[u][g] = acc[u][g] * alpha + p_w * vv;
                    m_loc[u][g] = m_new;
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

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
        name=f"ink_gqa_paged_sdpa_{kv_mode}_sp{int(split_out)}_kt{kt_cap}",
        input_names=in_names,
        output_names=out_names,
        header=header,
        source=src,
    )


@lru_cache(maxsize=6)
def _make_gqa_paged_flash_sdpa_kernel(kv_mode: str = "fp16"):
    """Tiled `simdgroup_half8x8` FlashAttention-2 (`BQ=32, BK=64, SG=4`) directly over 64-token paged KV blocks (`B == 1`)."""
    assert kv_mode in ("fp16", "fp8", "q4")
    kv_fp8 = kv_mode == "fp8"
    kv_q4 = kv_mode == "q4"
    header = (
        """
    #include <metal_simdgroup_matrix>
    #define ELEMS_H2 reinterpret_cast<thread half2&>
    #define ELEMS_F2 reinterpret_cast<thread float2&>
    """
        + (_FP8_E4M3_HEADER if kv_fp8 else "")
    )
    in_names = ["q", "KV"] + (["KV_sc"] if kv_q4 else []) + ["rel_st", "rel_proj_t", "sinks", "block_table", "seq_lens", "q_pos0", "kv_starts", "meta"]

    if kv_q4:
        load_k4 = (
            "{ half4 hk = half4(0.0h); if (kj < k_hi) { "
            "const ulong pt_u = (ulong)(block_table[kj >> 6] * 64u + (kj & 63u)); "
            "const uchar2 p = *(const device uchar2*)(KV + pt_u * 1024u + (ulong)kvh * 64u + d4 * 2u); "
            "const half sc = KV_sc[pt_u * 64u + (ulong)kvh * 4u + (d4 >> 3)]; "
            "hk = half4(half(int(p.x & 0xFu) - 8) * sc, half(int(p.x >> 4) - 8) * sc, half(int(p.y & 0xFu) - 8) * sc, half(int(p.y >> 4) - 8) * sc); "
            "} *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = hk; }"
        )
        load_v4 = (
            "{ half4 hv = half4(0.0h); if (kj < k_hi) { "
            "const ulong pt_u = (ulong)(block_table[kj >> 6] * 64u + (kj & 63u)); "
            "const uchar2 p = *(const device uchar2*)(KV + pt_u * 1024u + (ulong)(8u + kvh) * 64u + d4 * 2u); "
            "const half sc = KV_sc[pt_u * 64u + (ulong)(8u + kvh) * 4u + (d4 >> 3)]; "
            "hv = half4(half(int(p.x & 0xFu) - 8) * sc, half(int(p.x >> 4) - 8) * sc, half(int(p.y & 0xFu) - 8) * sc, half(int(p.y >> 4) - 8) * sc); "
            "} *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = hv; }"
        )
    elif kv_fp8:
        load_k4 = (
            "{ half4 hk = half4(0.0h); if (kj < k_hi) { "
            "const ulong pt_u = (ulong)(block_table[kj >> 6] * 64u + (kj & 63u)); "
            "const uchar4 u = *(const device uchar4*)(KV + pt_u * 2048u + (ulong)kvh * 128u + d4 * 4u); "
            "hk = half4(FP8_E4M3_LUT[u.x], FP8_E4M3_LUT[u.y], FP8_E4M3_LUT[u.z], FP8_E4M3_LUT[u.w]); "
            "} *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = hk; }"
        )
        load_v4 = (
            "{ half4 hv = half4(0.0h); if (kj < k_hi) { "
            "const ulong pt_u = (ulong)(block_table[kj >> 6] * 64u + (kj & 63u)); "
            "const uchar4 u = *(const device uchar4*)(KV + pt_u * 2048u + (ulong)(8u + kvh) * 128u + d4 * 4u); "
            "hv = half4(FP8_E4M3_LUT[u.x], FP8_E4M3_LUT[u.y], FP8_E4M3_LUT[u.z], FP8_E4M3_LUT[u.w]); "
            "} *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = hv; }"
        )
    else:
        load_k4 = (
            "{ half4 hk = half4(0.0h); if (kj < k_hi) { "
            "const ulong pt_u = (ulong)(block_table[kj >> 6] * 64u + (kj & 63u)); "
            "hk = *(const device half4*)(KV + pt_u * 2048u + (ulong)kvh * 128u + d4 * 4u); "
            "} *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = hk; }"
        )
        load_v4 = (
            "{ half4 hv = half4(0.0h); if (kj < k_hi) { "
            "const ulong pt_u = (ulong)(block_table[kj >> 6] * 64u + (kj & 63u)); "
            "hv = *(const device half4*)(KV + pt_u * 2048u + (ulong)(8u + kvh) * 128u + d4 * 4u); "
            "} *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = hv; }"
        )

    src = f"""
    // meta: [0]=T, [1]=is_sliding, [2]=n_qblocks
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

    for (uint k0 = k_lo; k0 < k_hi; k0 += 64u) {{
        #pragma unroll
        for (uint i = 0; i < 16u; i++) {{
            const uint idx = tid + i * 128u;
            const uint kt_l = idx >> 5, d4 = idx & 31u;
            const uint kj = k0 + kt_l;
            {load_k4}
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
            {load_v4}
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
        name=f"ink_gqa_paged_flash_sdpa_{kv_mode}",
        input_names=in_names,
        output_names=["out"],
        header=header,
        source=src,
    )


def _choose_paged_splits(B: int, max_span: int, n_qt: int) -> int:
    if max_span < 256:
        return 1
    # Target >= 64 threadgroups across the GPU (each TG handles 1 KV head), capping splits so each chunk >= 128 tokens
    base_tgs = B * n_qt * 8
    want = max(1, (128 + base_tgs - 1) // base_tgs)
    # Round up to power of 2 up to 32, and cap by span / 128
    max_by_span = max(1, max_span // 128)
    s = 1
    while s < want and s * 2 <= max_by_span and s < 32:
        s *= 2
    # Also ensure very long spans get enough parallelism even at moderate B
    if max_span >= 4096 and s < 4 and B <= 8:
        s = min(4, max_by_span)
    if max_span >= 12288 and s < 8 and B <= 4:
        s = min(8, max_by_span)
    return s


def gqa_paged_sdpa_fast(
    q: mx.array,
    kv_pool: mx.array,
    rel_st: mx.array,
    rel_proj_t_f16: mx.array,
    q_in: mx.array,
    sink_w_f16: mx.array,
    block_table: mx.array,
    seq_lens: mx.array,
    q_pos0: mx.array,
    kv_starts: mx.array,
    B: int,
    K: int,
    max_span: int,
    is_sliding: bool,
    *,
    kv_mode: str = "fp16",
    kv_q4_scales: mx.array | None = None,
) -> mx.array:
    """Paged batched GQA+RelPos+Sinks SDPA (`T = B * K`) with FA2 MMA prefill (`B=1, K>=16`) and Split-K decode."""
    T = B * K
    max_blocks = int(block_table.shape[1])
    kv_q4 = kv_mode == "q4"

    # Single-sequence prefill chunk (`B == 1, K >= 16`): hardware `simdgroup_half8x8` FlashAttention-2 directly on paged KV!
    if B == 1 and K >= 16:
        sinks = (q_in.reshape(T, 2304).astype(mx.float16) @ sink_w_f16.reshape(24, 2304).T).astype(mx.float32)
        nqb = (T + 31) // 32
        q_in_kern = h128_rows_f32(q) if kv_q4 else q
        # Pass kv_pool without .reshape(-1) so pools > 2^31 elements (>1M tokens) don't hit MLX's int32 shape cap
        ins = [q_in_kern.reshape(-1), kv_pool]
        if kv_q4:
            assert kv_q4_scales is not None
            ins.append(kv_q4_scales)
        ins.extend([
            rel_st.reshape(-1), rel_proj_t_f16.reshape(-1), sinks.reshape(-1),
            block_table.reshape(-1), seq_lens.reshape(-1), q_pos0.reshape(-1), kv_starts.reshape(-1),
            _u32(T, int(is_sliding), nqb),
        ])
        out = _make_gqa_paged_flash_sdpa_kernel(kv_mode)(
            inputs=ins,
            grid=(128, 24 * nqb, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(T, 3072)],
            output_dtypes=[mx.float32],
        )[0]
        return h128_rows_f32(out.reshape(T, 24, 128)).reshape(T, 3072) if kv_q4 else out

    kt_cap = 1 if K == 1 else 4
    n_qt = (K + kt_cap - 1) // kt_cap
    eff_span = min(max_span, 1024 + K) if is_sliding else max_span
    n_splits = _choose_paged_splits(B, eff_span, n_qt)
    split_out = (n_splits > 1) or (kt_cap > 1)

    ins = [q.reshape(-1), kv_pool]
    if kv_q4:
        assert kv_q4_scales is not None
        ins.append(kv_q4_scales)
    ins.extend([rel_st.reshape(-1), rel_proj_t_f16.reshape(-1)])
    if not split_out:
        ins.extend([q_in.reshape(-1), sink_w_f16.reshape(-1)])
    meta = _u32(T, K, max_blocks, int(is_sliding), n_splits, 0) if kt_cap == 1 else _u32(T, K, max_blocks, int(is_sliding), n_splits, 0, n_qt)
    ins.extend([block_table.reshape(-1), seq_lens.reshape(-1), q_pos0.reshape(-1), kv_starts.reshape(-1), meta])

    kern = _make_gqa_paged_sdpa_kernel(kv_mode, split_out=split_out, kt_cap=kt_cap)
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
    return _make_gqa_splitk_reduce_kernel(apply_h128=kv_q4)(
        inputs=[pm.reshape(-1), pl.reshape(-1), po.reshape(-1), q_in.reshape(-1), sink_w_f16.reshape(-1), _u32(T, n_splits)],
        grid=(128, 24 * T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(T, 3072)],
        output_dtypes=[mx.float32],
    )[0]


def _selftest():
    from .fast_kernels import (
        fp8_e4m3_quantize_f16_to_u8,
        gqa_relpos_sinks_sdpa_fast,
        kv_dyn_sconv4_norm_fused,
        kv_h128_q4_quantize,
        kv_sconv4_norm_fused,
        moe_wsum_sconv4_res_fused,
        qkv_rel_prep_sliding_fused,
        short_conv4_res_fused,
    )

    rng = np.random.default_rng(123)
    print("Testing batch_kernels on", mx.device_info().get("device_name"), flush=True)

    # 1. Batched conv kernels vs single-sequence kernels (K=1 and K=4)
    for B, K in ((4, 1), (3, 4)):
        T = B * K
        y_qkv = mx.array(rng.standard_normal((T, 8576)).astype(np.float32))
        qs = mx.array(rng.uniform(0.5, 1.5, (3072,)).astype(np.float32))
        rs = mx.array(rng.uniform(0.5, 1.5, (384,)).astype(np.float32))
        kp = mx.array(rng.standard_normal((B, 3, 1024)).astype(np.float32))
        vp = mx.array(rng.standard_normal((B, 3, 1024)).astype(np.float32))
        kw = mx.array(rng.standard_normal((4, 1024)).astype(np.float32))
        vw = mx.array(rng.standard_normal((4, 1024)).astype(np.float32))
        qb, rb, kvb, knb, vnb = qkv_rel_prep_sliding_batched(y_qkv, qs, rs, kp, kw, vp, vw, B, K, 22.6274, 8.0)
        mx.eval(qb, rb, kvb, knb, vnb)
        for b in range(B):
            q1, r1, kv1, kn1, vn1 = qkv_rel_prep_sliding_fused(
                y_qkv[b * K : (b + 1) * K], qs, rs, kp[b : b + 1], kw, vp[b : b + 1], vw, K, 22.6274, 8.0
            )
            kv1_t = kv1.transpose(2, 0, 1, 3)  # [K, 2, 8, 128]
            mx.eval(q1, r1, kv1_t, kn1, vn1)
            assert float(mx.max(mx.abs(qb[b * K : (b + 1) * K] - q1))) < 1e-5
            assert float(mx.max(mx.abs(rb[b * K : (b + 1) * K] - r1))) < 1e-5
            assert float(mx.max(mx.abs(kvb[b * K : (b + 1) * K].astype(mx.float32) - kv1_t.astype(mx.float32)))) == 0.0
            assert float(mx.max(mx.abs(knb[b : b + 1] - kn1))) == 0.0
            assert float(mx.max(mx.abs(vnb[b : b + 1] - vn1))) == 0.0

        # Dynamic KV conv
        kr = mx.array(rng.standard_normal((T, 1024)).astype(np.float32))
        vr = mx.array(rng.standard_normal((T, 1024)).astype(np.float32))
        ki = mx.array(rng.standard_normal((T, 2304)).astype(np.float32))
        vi = mx.array(rng.standard_normal((T, 2304)).astype(np.float32))
        kst = mx.array(rng.standard_normal((4, 1024)).astype(np.float32))
        vst = mx.array(rng.standard_normal((4, 1024)).astype(np.float32))
        kpr = mx.array(rng.standard_normal((32, 2304)).astype(np.float16))
        vpr = mx.array(rng.standard_normal((32, 2304)).astype(np.float16))
        kvb, knb, vnb = kv_dyn_sconv4_norm_batched(kp, kr, ki, kst, kpr, vp, vr, vi, vst, vpr, B, K, 22.6274)
        mx.eval(kvb, knb, vnb)
        for b in range(B):
            kv1, kn1, vn1 = kv_dyn_sconv4_norm_fused(
                kp[b : b + 1], kr[b * K : (b + 1) * K], ki[b * K : (b + 1) * K], kst, kpr,
                vp[b : b + 1], vr[b * K : (b + 1) * K], vi[b * K : (b + 1) * K], vst, vpr, K, 22.6274,
            )
            kv1_t = kv1.transpose(2, 0, 1, 3)
            mx.eval(kv1_t, kn1, vn1)
            assert float(mx.max(mx.abs(kvb[b * K : (b + 1) * K].astype(mx.float32) - kv1_t.astype(mx.float32)))) == 0.0
            assert float(mx.max(mx.abs(knb[b : b + 1] - kn1))) == 0.0

        # Post-attn & post-MLP batched convs
        p2 = mx.array(rng.standard_normal((B, 3, 2304)).astype(np.float32))
        x2 = mx.array(rng.standard_normal((T, 2304)).astype(np.float32))
        r2 = mx.array(rng.standard_normal((T, 2304)).astype(np.float32))
        w2 = mx.array(rng.standard_normal((4, 2304)).astype(np.float32))
        ob, npb = short_conv4_res_batched(p2, x2, w2, B, K, 2304, residual=r2)
        mx.eval(ob, npb)
        for b in range(B):
            o1, np1 = short_conv4_res_fused(p2[b : b + 1], x2[b * K : (b + 1) * K], w2, K, 2304, residual=r2[b * K : (b + 1) * K])
            mx.eval(o1, np1)
            assert float(mx.max(mx.abs(ob[b * K : (b + 1) * K] - o1))) < 1e-6
            assert float(mx.max(mx.abs(npb[b : b + 1] - np1))) == 0.0

        ydn = mx.array(rng.standard_normal((T * 8, 2304)).astype(np.float32))
        wn = mx.array(rng.uniform(0.0, 2.0, (T * 8,)).astype(np.float32))
        omb, npmb, rawb = moe_wsum_sconv4_res_batched(ydn, wn, p2, w2, r2, B, K)
        mx.eval(omb, npmb, rawb)
        for b in range(B):
            o1, np1, _ = moe_wsum_sconv4_res_fused(
                ydn[b * K * 8 : (b + 1) * K * 8], wn[b * K * 8 : (b + 1) * K * 8],
                p2[b : b + 1], w2, r2[b * K : (b + 1) * K], K,
            )
            mx.eval(o1, np1)
            assert float(mx.max(mx.abs(omb[b * K : (b + 1) * K] - o1))) < 1e-5
            assert float(mx.max(mx.abs(npmb[b : b + 1] - np1))) == 0.0
    print("  PASS all 4 batched sconv4 kernels (K=1 & K=4): bit-exact parity!")

    # 2. Paged SDPA vs gqa_relpos_sinks_sdpa_fast (fp16, fp8, q4; sliding & global; K=1, K=4; shared prefix!)
    B = 5
    prefix_blks = 4  # 256 shared tokens
    prefix_len = prefix_blks * PAGED_BLOCK_SIZE
    lens = [prefix_len + 17, prefix_len + 95, prefix_len + 140, prefix_len + 3, prefix_len + 210]
    rel_p = mx.array(rng.standard_normal((16, 1025)).astype(np.float16))
    sw = mx.array(rng.standard_normal((24, 2304)).astype(np.float16))

    for kv_mode in ("fp16", "fp8", "q4"):
        for is_sl in (False, True):
            if is_sl and kv_mode != "fp16":
                continue
            for K in (1, 4):
                T = B * K
                # Build shared prefix KV + per-seq suffix KV
                pfx_kv = mx.array(rng.standard_normal((prefix_len, 2, 8, 128)).astype(np.float16))
                seq_kvs = [mx.concatenate([pfx_kv, mx.array(rng.standard_normal((ln - prefix_len, 2, 8, 128)).astype(np.float16))], axis=0) for ln in lens]
                # Allocate physical blocks in shuffled order
                max_blks = max((ln + 63) // 64 for ln in lens)
                tot_blks = prefix_blks + sum((ln - prefix_len + 63) // 64 for ln in lens) + 8
                pool_f16 = mx.zeros((tot_blks * 64, 2, 8, 128), dtype=mx.float16)
                # Put prefix in blocks 5, 2, 7, 1
                pfx_phys = [5, 2, 7, 1]
                for i, pb in enumerate(pfx_phys):
                    pool_f16[pb * 64 : (pb + 1) * 64] = pfx_kv[i * 64 : (i + 1) * 64]
                nxt = 8
                bt_np = np.zeros((B, max_blks), dtype=np.uint32)
                for b, ln in enumerate(lens):
                    for i, pb in enumerate(pfx_phys):
                        bt_np[b, i] = pb
                    rem = ln - prefix_len
                    nb_s = (rem + 63) // 64
                    for s_i in range(nb_s):
                        pb = nxt
                        nxt += 1
                        bt_np[b, prefix_blks + s_i] = pb
                        lo = prefix_len + s_i * 64
                        hi = min(ln, lo + 64)
                        pool_f16[pb * 64 : pb * 64 + (hi - lo)] = seq_kvs[b][lo:hi]
                if kv_mode == "fp16":
                    pool_use, pool_sc = pool_f16, None
                elif kv_mode == "fp8":
                    pool_use, pool_sc = fp8_e4m3_quantize_f16_to_u8(pool_f16), None
                else:
                    pool_use, pool_sc = kv_h128_q4_quantize(pool_f16)

                q = mx.array(rng.standard_normal((T, 24, 128)).astype(np.float32))
                rel_st = mx.array(rng.standard_normal((T, 24, 16)).astype(np.float32))
                q_in = mx.array(rng.standard_normal((T, 2304)).astype(np.float32))
                bt_mx = mx.array(bt_np)
                sl_mx = mx.array(np.array(lens, dtype=np.uint32))
                qp_mx = mx.array(np.array([ln - K for ln in lens], dtype=np.uint32))
                ks_mx = mx.zeros((B,), dtype=mx.uint32)

                out_p = gqa_paged_sdpa_fast(
                    q, pool_use, rel_st, rel_p, q_in, sw, bt_mx, sl_mx, qp_mx, ks_mx,
                    B, K, max(lens), is_sl, kv_mode=kv_mode, kv_q4_scales=pool_sc,
                )
                mx.eval(out_p)

                # Reference against gqa_relpos_sinks_sdpa_fast per sequence
                max_err = 0.0
                for b, ln in enumerate(lens):
                    kv_ref_f16 = seq_kvs[b].transpose(1, 2, 0, 3)  # [2, 8, ln, 128]
                    if kv_mode == "fp16":
                        kv_r, sc_r = kv_ref_f16, None
                    elif kv_mode == "fp8":
                        kv_r, sc_r = fp8_e4m3_quantize_f16_to_u8(kv_ref_f16), None
                    else:
                        kv_r, sc_r = kv_h128_q4_quantize(kv_ref_f16)
                    ref_b = gqa_relpos_sinks_sdpa_fast(
                        q[b * K : (b + 1) * K], kv_r, rel_st[b * K : (b + 1) * K], rel_p,
                        q_in[b * K : (b + 1) * K], sw, K, ln, 0, ln - K, ln, is_sl,
                        kv_fp8=(kv_mode == "fp8"), kv_q4_scales=sc_r,
                    )
                    mx.eval(ref_b)
                    err = float(mx.max(mx.abs(out_p[b * K : (b + 1) * K] - ref_b)))
                    max_err = max(max_err, err)
                assert max_err < 1e-3, f"paged SDPA mismatch mode={kv_mode} sl={is_sl} K={K}: {max_err}"

        # 3. Paged FlashAttention-2 MMA (`B=1, K=64`) parity across fp16/fp8/q4, sliding & global
        for is_sl in (False, True):
            if is_sl and kv_mode != "fp16":
                continue
            K_pf = 64
            ln_pf = prefix_len + 137
            seq_kv = mx.array(rng.standard_normal((ln_pf, 2, 8, 128)).astype(np.float16))
            nb_pf = (ln_pf + 63) // 64
            phys_pf = [((i * 3 + 2) % (nb_pf + 4)) for i in range(nb_pf)]
            pool_pf = mx.zeros(((nb_pf + 4) * 64, 2, 8, 128), dtype=mx.float16)
            for i, pb in enumerate(phys_pf):
                lo, hi = i * 64, min(ln_pf, (i + 1) * 64)
                pool_pf[pb * 64 : pb * 64 + (hi - lo)] = seq_kv[lo:hi]
            if kv_mode == "fp16":
                p_use, p_sc = pool_pf, None
                r_use, r_sc = seq_kv.transpose(1, 2, 0, 3), None
            elif kv_mode == "fp8":
                p_use, p_sc = fp8_e4m3_quantize_f16_to_u8(pool_pf), None
                r_use, r_sc = fp8_e4m3_quantize_f16_to_u8(seq_kv.transpose(1, 2, 0, 3)), None
            else:
                p_use, p_sc = kv_h128_q4_quantize(pool_pf)
                r_use, r_sc = kv_h128_q4_quantize(seq_kv.transpose(1, 2, 0, 3))
            q_pf = mx.array(rng.standard_normal((K_pf, 24, 128)).astype(np.float32))
            rel_pf = mx.array(rng.standard_normal((K_pf, 24, 16)).astype(np.float32))
            qin_pf = mx.array(rng.standard_normal((K_pf, 2304)).astype(np.float32))
            out_pf = gqa_paged_sdpa_fast(
                q_pf, p_use, rel_pf, rel_p, qin_pf, sw,
                mx.array(np.array([phys_pf], dtype=np.uint32)),
                mx.array(np.array([ln_pf], dtype=np.uint32)),
                mx.array(np.array([ln_pf - K_pf], dtype=np.uint32)),
                mx.array(np.array([0], dtype=np.uint32)),
                1, K_pf, ln_pf, is_sl, kv_mode=kv_mode, kv_q4_scales=p_sc,
            )
            ref_pf = gqa_relpos_sinks_sdpa_fast(
                q_pf, r_use, rel_pf, rel_p, qin_pf, sw, K_pf, ln_pf, 0, ln_pf - K_pf, ln_pf, is_sl,
                kv_fp8=(kv_mode == "fp8"), kv_q4_scales=r_sc,
            )
            mx.eval(out_pf, ref_pf)
            err_pf = float(mx.max(mx.abs(out_pf - ref_pf)))
            assert err_pf < 1e-3, f"paged FA2 SDPA mismatch mode={kv_mode} sl={is_sl}: {err_pf}"
    print("  PASS gqa_paged_sdpa_fast (fp16/fp8/q4, sliding/global, K=1/K=4 + FA2 K=64, shared physical prefix blocks): max_err < 1e-3!")
    print("ALL batch_kernels selftests PASSED!", flush=True)


if __name__ == "__main__":
    _selftest()
