"""Synthetic & mock-model tests for Hadamard + Lloyd-Max KV cache (`kv_q35.py` & `fast_q35.py`).

Run directly with the MLX venv Python (no pytest required, no real model weights loaded):

    .venv/bin/python runtime_mlx/test_kv_q35.py
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime_mlx"))

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import ArraysCache, KVCache

from runtime_mlx import fast_q35 as F
from runtime_mlx import kv_q35 as K


def _sylvester_np(n: int) -> np.ndarray:
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


def _rht_matrix_np(head_dim: int) -> np.ndarray:
    s = K.rht_signs_np(head_dim)
    return s[:, None] * _sylvester_np(head_dim) * s[None, :]


def _numpy_rht_lm_quantize(x_f16: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    head_dim = x_f16.shape[-1]
    prefix = x_f16.shape[:-1]
    if bits == 16:
        return x_f16.astype(np.float16), np.ones((*prefix, 1), dtype=np.float16)

    R = _rht_matrix_np(head_dim)
    x_rot = x_f16.astype(np.float64) @ R.T
    if bits == 8:
        amax = np.abs(x_rot).max(axis=-1, keepdims=True)
        scales = np.clip(amax / 127.0, 1e-6, 65504.0).astype(np.float16)
        inv_sc = 1.0 / scales.astype(np.float64)
        codes = np.clip(np.rint(x_rot * inv_sc), -127, 127).astype(np.int8)
        return codes, scales

    rms = np.sqrt(np.mean(x_rot ** 2, axis=-1, keepdims=True))
    scales = np.clip(rms, 1e-6, 65504.0).astype(np.float16)
    z = x_rot * (1.0 / scales.astype(np.float64))
    mids = np.array(K.LM_MIDS[bits], dtype=np.float64)
    idx = np.sum(z[..., None] > mids, axis=-1).astype(np.uint8)
    if bits == 4:
        packed = (idx[..., 0::2] | (idx[..., 1::2] << 4)).astype(np.uint8)
    elif bits == 3:
        planes = np.stack([(idx >> p) & 1 for p in range(3)], axis=-2)
        packed = np.packbits(planes, axis=-1, bitorder="little").reshape(*prefix, (3 * head_dim) // 8)
    elif bits == 2:
        packed = (
            idx[..., 0::4]
            | (idx[..., 1::4] << 2)
            | (idx[..., 2::4] << 4)
            | (idx[..., 3::4] << 6)
        ).astype(np.uint8)
    else:
        raise ValueError(bits)
    return packed, scales


def _numpy_rht_lm_dequantize(codes: np.ndarray, scales: np.ndarray, bits: int, head_dim: int) -> np.ndarray:
    if bits == 16:
        return codes.astype(np.float16)
    prefix = codes.shape[:-1]
    sc = scales.astype(np.float64)
    if bits == 8:
        x_rot = codes.astype(np.float64) * sc
    elif bits == 4:
        lo = codes & 0xF
        hi = codes >> 4
        idx = np.empty((*prefix, head_dim), dtype=np.uint8)
        idx[..., 0::2] = lo
        idx[..., 1::2] = hi
        levels = np.array(K.LM4_LEVELS, dtype=np.float64)
        x_rot = levels[idx] * sc
    elif bits == 3:
        pb = head_dim // 8
        planes = codes.reshape(*prefix, 3, pb)
        bits_unpacked = np.unpackbits(planes, axis=-1, bitorder="little")[..., :head_dim]
        idx = (
            bits_unpacked[..., 0, :]
            | (bits_unpacked[..., 1, :] << 1)
            | (bits_unpacked[..., 2, :] << 2)
        )
        levels = np.array(K.LM3_LEVELS, dtype=np.float64)
        x_rot = levels[idx] * sc
    elif bits == 2:
        idx = np.empty((*prefix, head_dim), dtype=np.uint8)
        idx[..., 0::4] = codes & 0x3
        idx[..., 1::4] = (codes >> 2) & 0x3
        idx[..., 2::4] = (codes >> 4) & 0x3
        idx[..., 3::4] = (codes >> 6) & 0x3
        levels = np.array(K.LM2_LEVELS, dtype=np.float64)
        x_rot = levels[idx] * sc
    else:
        raise ValueError(bits)
    R = _rht_matrix_np(head_dim)
    return (x_rot @ R.T).astype(np.float16)


def _fp32_sdpa_ref(
    q: mx.array,
    k_full: mx.array,
    v_full: mx.array,
    scale: float,
    mask: str | mx.array | None = None,
) -> np.ndarray:
    B, H, L_q, D = q.shape
    H_kv = k_full.shape[1]
    G = H // H_kv
    qf = np.array(q.astype(mx.float32)).reshape(B, H_kv, G, L_q, D).astype(np.float64)
    kf = np.array(k_full.astype(mx.float32))[:, :, None, :, :].astype(np.float64)
    vf = np.array(v_full.astype(mx.float32))[:, :, None, :, :].astype(np.float64)
    scores = np.matmul(qf * scale, np.swapaxes(kf, -1, -2))
    L_kv = k_full.shape[2]
    if mask == "causal":
        q_idx = np.arange(L_kv - L_q, L_kv)[:, None]
        k_idx = np.arange(L_kv)[None, :]
        valid = q_idx >= k_idx
        scores = np.where(valid[None, None, None, :, :], scores, -1e30)
    elif isinstance(mask, mx.array):
        m_np = np.array(mask)
        if m_np.ndim == 2:
            m_np = m_np[None, None, None, :, :]
        elif m_np.ndim == 3:
            m_np = m_np[:, None, None, :, :]
        elif m_np.ndim == 4:
            if m_np.shape[1] == 1:
                m_np = m_np[:, :, None, :, :]
            else:
                m_np = m_np.reshape(m_np.shape[0], H_kv, G, m_np.shape[2], m_np.shape[3])
        if m_np.dtype == np.bool_:
            scores = np.where(m_np, scores, -1e30)
        else:
            scores = scores + m_np.astype(np.float64)
    m = np.max(scores, axis=-1, keepdims=True)
    p = np.exp(scores - m)
    if mask == "causal":
        p = np.where(valid[None, None, None, :, :], p, 0.0)
    elif isinstance(mask, mx.array) and m_np.dtype == np.bool_:
        p = np.where(m_np, p, 0.0)
    denom = np.sum(p, axis=-1, keepdims=True)
    out = np.matmul(p / np.maximum(denom, 1e-12), vf)
    return out.reshape(B, H, L_q, D).astype(np.float32)


class RHTAndQuantTest(unittest.TestCase):
    def test_rht_orthogonality_and_involution(self):
        rng = np.random.default_rng(101)
        for D in (128, 256):
            with self.subTest(head_dim=D):
                R = _rht_matrix_np(D)
                self.assertLess(np.abs(R.T @ R - np.eye(D)).max(), 1e-14)
                self.assertLess(np.abs(R @ R - np.eye(D)).max(), 1e-14)

                x = rng.standard_normal((17, D)).astype(np.float32)
                y = K.q35_rht(mx.array(x), out_dtype=mx.float32)
                y_np = np.array(y)
                ref = x.astype(np.float64) @ R.T
                err_fwd = np.abs(y_np - ref).max()
                self.assertLess(err_fwd, 1e-5, f"D={D} fwd err={err_fwd}")

                x2 = np.array(K.q35_rht(y, out_dtype=mx.float32))
                err_inv = np.abs(x2 - x).max()
                self.assertLess(err_inv, 1e-5, f"D={D} inv err={err_inv}")

                # Dot-product preservation
                q = rng.standard_normal((8, D)).astype(np.float32)
                k = rng.standard_normal((8, D)).astype(np.float32)
                qr = np.array(K.q35_rht(mx.array(q), out_dtype=mx.float32))
                kr = np.array(K.q35_rht(mx.array(k), out_dtype=mx.float32))
                dot_err = np.abs(np.sum(q * k, axis=-1) - np.sum(qr * kr, axis=-1)).max()
                self.assertLess(dot_err, 1e-4, f"D={D} dot err={dot_err}")

    def test_quantize_dequantize_parity_vs_numpy(self):
        rng = np.random.default_rng(202)
        for D in (128, 256):
            x_f16 = (rng.standard_normal((2, 4, 9, D)) * 1.7).astype(np.float16)
            # Include an all-zero head and an outlier head
            x_f16[0, 0, 0, :] = 0.0
            x_f16[1, 3, 5, 0] = np.float16(18.5)
            for bits in (2, 3, 4, 8, 16):
                with self.subTest(head_dim=D, bits=bits):
                    codes, scales = K.q35_rht_lm_quantize(mx.array(x_f16), bits)
                    deq = K.q35_rht_lm_dequantize(codes, scales, bits, D)

                    ref_codes, ref_scales = _numpy_rht_lm_quantize(x_f16, bits)
                    ref_deq = _numpy_rht_lm_dequantize(ref_codes, ref_scales, bits, D)

                    self.assertEqual(codes.shape, ref_codes.shape)
                    self.assertEqual(scales.shape, ref_scales.shape)
                    np.testing.assert_allclose(
                        np.array(scales).astype(np.float32),
                        ref_scales.astype(np.float32),
                        rtol=1e-2,
                        atol=1e-5,
                    )
                    # Dequantized output parity (allowing 1-ULP boundary flip in fp32->fp16 RHT)
                    deq_np = np.array(deq).astype(np.float32)
                    ref_np = ref_deq.astype(np.float32)
                    max_diff = np.abs(deq_np - ref_np).max()
                    # Check exact code agreement on non-boundary entries (>99% identical)
                    code_match = np.mean(np.array(codes) == ref_codes)
                    self.assertGreater(code_match, 0.99, f"D={D} bits={bits} code_match={code_match}")
                    # Verify dequant(codes) matches numpy_dequant(same codes) to FP16 precision!
                    ref_from_same = _numpy_rht_lm_dequantize(np.array(codes), np.array(scales), bits, D).astype(np.float32)
                    deq_err = np.abs(deq_np - ref_from_same).max()
                    self.assertLess(deq_err, 5e-3, f"D={D} bits={bits} deq_err={deq_err}, full_diff={max_diff}")


class SDPADecodeTest(unittest.TestCase):
    def test_decode_parity_all_configs(self):
        rng = np.random.default_rng(303)
        pairs = [(4, 4), (4, 3), (4, 2), (3, 2), (2, 2), (8, 4), (16, 4), (4, 16), (16, 16)]
        worst_err = 0.0
        for D, H, H_kv in [(256, 16, 4), (128, 16, 4), (128, 24, 4), (256, 32, 4)]:
            scale = D ** -0.5
            lengths = (1, 4, 16, 256, 2048) if H == 16 else (1, 16, 2048)
            for L in lengths:
                n_sink = min(L, 4)
                n_body = max(0, L - n_sink)
                # Use cap_body > n_body on larger caches to prove trailing garbage is ignored
                cap_body = n_body + (37 if n_body > 0 else 0)
                q_np = (rng.standard_normal((1, H, 1, D)) * 0.8).astype(np.float32)
                k_all = (rng.standard_normal((1, H_kv, L, D)) * 0.8).astype(np.float16)
                v_all = (rng.standard_normal((1, H_kv, L, D)) * 0.8).astype(np.float16)
                k_sink = mx.array(k_all[:, :, :n_sink, :])
                v_sink = mx.array(v_all[:, :, :n_sink, :])

                test_pairs = pairs if (L in (16, 2048) and H == 16) else [(4, 3), (3, 2)]
                for k_bits, v_bits in test_pairs:
                    with self.subTest(head_dim=D, H=H, H_kv=H_kv, L=L, k_bits=k_bits, v_bits=v_bits):
                        if n_body > 0:
                            kq, ks = K.q35_rht_lm_quantize(mx.array(k_all[:, :, n_sink:, :]), k_bits)
                            vq, vs = K.q35_rht_lm_quantize(mx.array(v_all[:, :, n_sink:, :]), v_bits)
                            k_deq_body = K.q35_rht_lm_dequantize(kq, ks, k_bits, D)
                            v_deq_body = K.q35_rht_lm_dequantize(vq, vs, v_bits, D)
                            # Pad buffers with non-zero garbage in [n_body : cap_body]
                            kc_pad = mx.concatenate([kq, mx.ones((1, H_kv, cap_body - n_body, kq.shape[-1]), kq.dtype) * 7], axis=2)
                            vc_pad = mx.concatenate([vq, mx.ones((1, H_kv, cap_body - n_body, vq.shape[-1]), vq.dtype) * 7], axis=2)
                            ks_pad = mx.concatenate([ks, mx.full((1, H_kv, cap_body - n_body, 1), 99.0, mx.float16)], axis=2)
                            vs_pad = mx.concatenate([vs, mx.full((1, H_kv, cap_body - n_body, 1), 99.0, mx.float16)], axis=2)
                            k_full = mx.concatenate([k_sink, k_deq_body], axis=2)
                            v_full = mx.concatenate([v_sink, v_deq_body], axis=2)
                        else:
                            kc_pad = vc_pad = ks_pad = vs_pad = None
                            k_full, v_full = k_sink, v_sink

                        q_mx = mx.array(q_np)
                        got = K.q35_rht_lm_sdpa_decode(
                            q_mx, k_sink, v_sink, kc_pad, ks_pad, vc_pad, vs_pad,
                            k_bits=k_bits, v_bits=v_bits, n_sink=n_sink, n_body=n_body, scale=scale,
                        )
                        ref = _fp32_sdpa_ref(q_mx, k_full, v_full, scale, mask=None)
                        err = float(np.abs(np.array(got) - ref).max())
                        worst_err = max(worst_err, err)
                        self.assertLess(err, 2e-3, f"D={D} H={H} L={L} ({k_bits},{v_bits}) err={err}")
        self.assertLess(worst_err, 2e-3)

    def test_decode_batch2_and_no_sink(self):
        rng = np.random.default_rng(404)
        B, H, H_kv, D, L = 2, 16, 4, 256, 320
        q_mx = mx.array((rng.standard_normal((B, H, 1, D)) * 0.7).astype(np.float16))
        k_mx = mx.array((rng.standard_normal((B, H_kv, L, D)) * 0.7).astype(np.float16))
        v_mx = mx.array((rng.standard_normal((B, H_kv, L, D)) * 0.7).astype(np.float16))
        kq, ks = K.q35_rht_lm_quantize(k_mx, 4)
        vq, vs = K.q35_rht_lm_quantize(v_mx, 3)
        k_deq = K.q35_rht_lm_dequantize(kq, ks, 4, D)
        v_deq = K.q35_rht_lm_dequantize(vq, vs, 3, D)
        got = K.q35_rht_lm_sdpa_decode(q_mx, None, None, kq, ks, vq, vs, k_bits=4, v_bits=3, n_sink=0, n_body=L)
        ref = _fp32_sdpa_ref(q_mx, k_deq, v_deq, D ** -0.5)
        err = float(np.abs(np.array(got).astype(np.float32) - ref).max())
        self.assertLess(err, 2e-3, f"batch2 no_sink err={err}")


class SDPAPrefillTest(unittest.TestCase):
    def test_prefill_causal_and_prefix(self):
        rng = np.random.default_rng(505)
        for D in (128, 256):
            H, H_kv = 16, 4
            scale = D ** -0.5
            for L_q, prefix_len, tile_k in [(64, 0, 512), (64, 200, 128), (300, 150, 128)]:
                total_len = prefix_len + L_q
                n_sink = min(4, total_len)
                n_body = total_len - n_sink
                q_mx = mx.array((rng.standard_normal((1, H, L_q, D)) * 0.7).astype(np.float32))
                k_all = mx.array((rng.standard_normal((1, H_kv, total_len, D)) * 0.7).astype(np.float16))
                v_all = mx.array((rng.standard_normal((1, H_kv, total_len, D)) * 0.7).astype(np.float16))
                k_sink = k_all[:, :, :n_sink, :]
                v_sink = v_all[:, :, :n_sink, :]
                kq, ks = K.q35_rht_lm_quantize(k_all[:, :, n_sink:, :], 4)
                vq, vs = K.q35_rht_lm_quantize(v_all[:, :, n_sink:, :], 3)
                k_full = mx.concatenate([k_sink, K.q35_rht_lm_dequantize(kq, ks, 4, D)], axis=2)
                v_full = mx.concatenate([v_sink, K.q35_rht_lm_dequantize(vq, vs, 3, D)], axis=2)

                with self.subTest(head_dim=D, L_q=L_q, prefix=prefix_len, tile_k=tile_k):
                    got = K.q35_rht_lm_sdpa_prefill(
                        q_mx, k_sink, v_sink, kq, ks, vq, vs,
                        k_bits=4, v_bits=3, n_sink=n_sink, n_body=n_body,
                        scale=scale, mask="causal", tile_k=tile_k, tile_q=128,
                    )
                    ref = _fp32_sdpa_ref(q_mx, k_full, v_full, scale, mask="causal")
                    err = float(np.abs(np.array(got) - ref).max())
                    self.assertLess(err, 2e-3, f"prefill err={err}")

                    # Explicit boolean mask parity
                    q_idx = mx.arange(prefix_len, total_len)[:, None]
                    k_idx = mx.arange(total_len)[None, :]
                    bool_mask = q_idx >= k_idx
                    got_bool = K.q35_rht_lm_sdpa_prefill(
                        q_mx, k_sink, v_sink, kq, ks, vq, vs,
                        k_bits=4, v_bits=3, n_sink=n_sink, n_body=n_body,
                        scale=scale, mask=bool_mask, tile_k=tile_k,
                    )
                    err_bool = float(np.abs(np.array(got_bool) - ref).max())
                    self.assertLess(err_bool, 2e-3, f"prefill bool mask err={err_bool}")

    def test_prefill_then_decode_consistency(self):
        rng = np.random.default_rng(606)
        B, H, H_kv, D = 1, 16, 4, 256
        scale = D ** -0.5
        L_pre, L_dec = 70, 6
        cache = K.HadamardLloydMaxKVCache(k_bits=4, v_bits=3, sink_tokens=4)

        k_pre = mx.array((rng.standard_normal((B, H_kv, L_pre, D)) * 0.6).astype(np.float16))
        v_pre = mx.array((rng.standard_normal((B, H_kv, L_pre, D)) * 0.6).astype(np.float16))
        q_pre = mx.array((rng.standard_normal((B, H, L_pre, D)) * 0.6).astype(np.float32))
        cache.update(k_pre, v_pre)

        out_pre = K.q35_rht_lm_sdpa_prefill(
            q_pre, cache.k_sink, cache.v_sink, cache.k_codes, cache.k_scales,
            cache.v_codes, cache.v_scales, cache.k_bits, cache.v_bits,
            n_sink=cache.n_sink, n_body=cache.n_body, scale=scale, mask="causal", tile_k=32,
        )
        k_full_pre, v_full_pre = cache.keys_and_values()
        ref_pre = _fp32_sdpa_ref(q_pre, k_full_pre, v_full_pre, scale, mask="causal")
        self.assertLess(float(np.abs(np.array(out_pre) - ref_pre).max()), 2e-3)

        for step in range(L_dec):
            q_t = mx.array((rng.standard_normal((B, H, 1, D)) * 0.6).astype(np.float32))
            k_t = mx.array((rng.standard_normal((B, H_kv, 1, D)) * 0.6).astype(np.float16))
            v_t = mx.array((rng.standard_normal((B, H_kv, 1, D)) * 0.6).astype(np.float16))
            cache.update(k_t, v_t)
            out_dec = K.q35_rht_lm_sdpa_decode(
                q_t, cache.k_sink, cache.v_sink, cache.k_codes, cache.k_scales,
                cache.v_codes, cache.v_scales, cache.k_bits, cache.v_bits,
                n_sink=cache.n_sink, n_body=cache.n_body, scale=scale,
            )
            k_full, v_full = cache.keys_and_values()
            ref_dec = _fp32_sdpa_ref(q_t, k_full, v_full, scale, mask=None)
            err = float(np.abs(np.array(out_dec) - ref_dec).max())
            self.assertLess(err, 2e-3, f"step={step} decode err={err}")
        self.assertEqual(cache.offset, L_pre + L_dec)


class CacheBehaviorTest(unittest.TestCase):
    def test_sink_exact_fp16_preservation(self):
        rng = np.random.default_rng(707)
        cache = K.HadamardLloydMaxKVCache(k_bits=2, v_bits=2, sink_tokens=4)
        # Put giant outliers in positions 0..3; they must stay exact FP16!
        k = (rng.standard_normal((1, 4, 10, 256)) * 0.5).astype(np.float16)
        v = (rng.standard_normal((1, 4, 10, 256)) * 0.5).astype(np.float16)
        k[:, :, 0, 17] = np.float16(123.5)
        v[:, :, 2, 200] = np.float16(-87.25)
        k_out, v_out = cache.update_and_fetch(mx.array(k), mx.array(v))
        np.testing.assert_array_equal(np.array(k_out[:, :, :4, :]), k[:, :, :4, :])
        np.testing.assert_array_equal(np.array(v_out[:, :, :4, :]), v[:, :, :4, :])

    def test_trim_state_mask_and_exact_nbytes(self):
        rng = np.random.default_rng(808)
        for k_bits, v_bits, k_pb, v_pb in [(4, 3, 128, 96), (3, 2, 96, 64), (8, 4, 256, 128), (16, 4, 512, 128)]:
            with self.subTest(k_bits=k_bits, v_bits=v_bits):
                cache = K.HadamardLloydMaxKVCache(k_bits=k_bits, v_bits=v_bits, sink_tokens=4)
                self.assertTrue(cache.empty())
                self.assertTrue(cache.is_trimmable())
                self.assertEqual(cache.nbytes, 0)

                k1 = mx.array(rng.standard_normal((1, 4, 12, 256)).astype(np.float16))
                v1 = mx.array(rng.standard_normal((1, 4, 12, 256)).astype(np.float16))
                cache.update(k1, v1)
                self.assertFalse(cache.empty())
                self.assertEqual(cache.offset, 12)
                self.assertEqual(cache.size(), 12)
                self.assertEqual(cache.n_sink, 4)
                self.assertEqual(cache.n_body, 8)
                self.assertEqual(cache.make_mask(1), None)
                self.assertEqual(cache.make_mask(4), "causal")

                # Exact byte accounting:
                # sink: 2 * (1 * 4 * 4 * 256 * 2B) = 16384 B
                # body cap = 256: k_codes (4*256*k_pb) + v_codes (4*256*v_pb) + scales (4*256*2B per <16b side)
                sink_bytes = 2 * (1 * 4 * 4 * 256 * 2)
                k_bytes = 1 * 4 * 256 * (k_pb + (2 if k_bits < 16 else 0))
                v_bytes = 1 * 4 * 256 * (v_pb + (2 if v_bits < 16 else 0))
                self.assertEqual(cache.nbytes, sink_bytes + k_bytes + v_bytes)

                # Trim within body, then back into sink, then re-append
                trimmed = cache.trim(5)
                self.assertEqual(trimmed, 5)
                self.assertEqual(cache.offset, 7)
                self.assertEqual(cache.n_sink, 4)
                self.assertEqual(cache.n_body, 3)

                cache.trim(5)
                self.assertEqual(cache.offset, 2)
                self.assertEqual(cache.n_sink, 2)
                self.assertEqual(cache.n_body, 0)

                k2 = mx.array(rng.standard_normal((1, 4, 6, 256)).astype(np.float16))
                v2 = mx.array(rng.standard_normal((1, 4, 6, 256)).astype(np.float16))
                kf, vf = cache.update_and_fetch(k2, v2)
                self.assertEqual(cache.offset, 8)
                self.assertEqual(kf.shape, (1, 4, 8, 256))
                np.testing.assert_array_equal(np.array(kf[:, :, :2, :]), np.array(k1[:, :, :2, :]))
                np.testing.assert_array_equal(np.array(kf[:, :, 2:4, :]), np.array(k2[:, :, :2, :]))

                # State round-trip
                restored = K.HadamardLloydMaxKVCache.from_state(cache.state)
                self.assertEqual(restored.offset, cache.offset)
                self.assertEqual(restored.nbytes, cache.nbytes)
                rk, _ = restored.keys_and_values()
                np.testing.assert_array_equal(np.array(rk), np.array(kf))

    def test_zero_sink_tokens(self):
        rng = np.random.default_rng(910)
        cache = K.HadamardLloydMaxKVCache(k_bits=4, v_bits=3, sink_tokens=0)
        k = mx.array(rng.standard_normal((1, 4, 10, 256)).astype(np.float16))
        v = mx.array(rng.standard_normal((1, 4, 10, 256)).astype(np.float16))
        kf, vf = cache.update_and_fetch(k, v)
        self.assertEqual(cache.n_sink, 0)
        self.assertEqual(cache.n_body, 10)
        self.assertIsNone(cache.k_sink)
        self.assertEqual(kf.shape, (1, 4, 10, 256))
        self.assertEqual(vf.shape, (1, 4, 10, 256))

    def test_growth_beyond_step(self):
        rng = np.random.default_rng(909)
        cache = K.HadamardLloydMaxKVCache(k_bits=4, v_bits=3, sink_tokens=4)
        k1 = mx.array(rng.standard_normal((1, 4, 265, 128)).astype(np.float16))
        v1 = mx.array(rng.standard_normal((1, 4, 265, 128)).astype(np.float16))
        cache.update(k1, v1)
        self.assertEqual(cache.n_body, 261)
        self.assertEqual(cache.k_codes.shape[2], 512)
        cache.trim(50)  # n_body = 211 (not divisible by 256)
        k2 = mx.array(rng.standard_normal((1, 4, 310, 128)).astype(np.float16))
        v2 = mx.array(rng.standard_normal((1, 4, 310, 128)).astype(np.float16))
        cache.update(k2, v2)
        self.assertEqual(cache.n_body, 521)
        self.assertEqual(cache.k_codes.shape[2], 768)


class PlanAndModelIntegrationTest(unittest.TestCase):
    def _make_mock_q35_model(self, num_layers: int = 8, wrap_fast_layer: bool = False):
        cfg = qwen3_5.TextModelArgs(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=num_layers,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=128,
            vocab_size=128,
            linear_num_value_heads=8,
            linear_num_key_heads=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            full_attention_interval=4,
        )
        m_cfg = qwen3_5.ModelArgs(model_type="qwen3_5", text_config=cfg.__dict__)
        model = qwen3_5.Model(m_cfg)
        if wrap_fast_layer:
            tm = model.language_model.model
            fast = [F.FastLayer(layer, tm.norm.eps) for layer in tm.layers]
            for a, b in zip(fast, fast[1:]):
                a._next[0] = b
            tm.layers = fast
        return model

    def test_kv_plans_and_overrides(self):
        model = self._make_mock_q35_model(num_layers=12, wrap_fast_layer=True)
        # Full-attn indices: 3, 7, 11
        self.assertEqual(K.build_q35_kv_plan(model, "fp16"), {3: (16, 16), 7: (16, 16), 11: (16, 16)})
        self.assertEqual(K.build_q35_kv_plan(model, "lm4"), {3: (4, 4), 7: (4, 4), 11: (4, 4)})
        self.assertEqual(K.build_q35_kv_plan(model, "lm4_3"), {3: (4, 3), 7: (4, 3), 11: (4, 4)})
        self.assertEqual(K.build_q35_kv_plan(model, "lm3_2"), {3: (3, 2), 7: (3, 2), 11: (4, 4)})
        self.assertEqual(K.build_q35_kv_plan(model, "q35_auto"), {3: (4, 3), 7: (4, 3), 11: (4, 4)})
        self.assertEqual(K.build_q35_kv_plan(model, "q35_compact"), {3: (3, 2), 7: (4, 3), 11: (4, 4)})
        self.assertEqual(K.build_q35_kv_plan(model, "k4v3"), {3: (4, 3), 7: (4, 3), 11: (4, 3)})
        self.assertEqual(
            K.build_q35_kv_plan(model, "default:4,3;last:4,4"),
            {3: (4, 3), 7: (4, 3), 11: (4, 4)},
        )
        self.assertEqual(
            K.build_q35_kv_plan(model, "3:4,3;11:4,8"),
            {3: (4, 3), 7: (16, 16), 11: (4, 8)},
        )
        with self.assertRaises(ValueError):
            K.build_q35_kv_plan(model, "0:4,3")  # layer 0 is linear
        with self.assertRaises(ValueError):
            K.build_q35_kv_plan(model, "bad_spec")

    def test_install_forward_and_make_cache_wrapped_and_unwrapped(self):
        for wrap in (False, True):
            with self.subTest(wrap_fast_layer=wrap):
                mx.random.seed(42)
                ref_model = self._make_mock_q35_model(num_layers=8, wrap_fast_layer=wrap)
                # Copy weights cleanly before modifying
                q_model = self._make_mock_q35_model(num_layers=8, wrap_fast_layer=wrap)
                q_model.load_weights(list(tree_flatten(ref_model.parameters())))

                # fp16 must be a complete no-op
                noop_model = self._make_mock_q35_model(num_layers=8, wrap_fast_layer=wrap)
                K.install_kv_quant(noop_model, "fp16", sink_tokens=4)
                self.assertIsInstance(noop_model.make_cache()[3], KVCache)

                # 1) Single-layer QuantizedKVAttention vs Qwen3NextAttention at (16, 16)
                layer3_ref = getattr(ref_model.layers[3], "inner", ref_model.layers[3]).self_attn
                attn16 = K.QuantizedKVAttention(layer3_ref)
                x_attn = mx.array(np.random.default_rng(123).standard_normal((1, 10, 256)).astype(np.float32))
                c_kv_ref = KVCache()
                c_kv_q16 = K.HadamardLloydMaxKVCache(k_bits=16, v_bits=16, sink_tokens=4)
                y_ref = layer3_ref(x_attn, mask="causal", cache=c_kv_ref)
                y_q16 = attn16(x_attn, mask="causal", cache=c_kv_q16)
                mx.eval(y_ref, y_q16)
                self.assertLess(float(mx.abs(y_ref - y_q16).max().item()), 1e-3)

                # Also check fallback paths (cache=None and standard KVCache)
                y_nocache = attn16(x_attn, mask="causal", cache=None)
                y_nocache_ref = layer3_ref(x_attn, mask="causal", cache=None)
                mx.eval(y_nocache, y_nocache_ref)
                self.assertLess(float(mx.abs(y_nocache - y_nocache_ref).max().item()), 1e-5)

                c_std = KVCache()
                c_std_ref = KVCache()
                y_std = attn16(x_attn, mask="causal", cache=c_std)
                y_std_ref = layer3_ref(x_attn, mask="causal", cache=c_std_ref)
                mx.eval(y_std, y_std_ref)
                self.assertLess(float(mx.abs(y_std - y_std_ref).max().item()), 1e-5)

                # 2) Full 8-layer model forward at (8, 8) and q35_auto (4,3 / 4,4)
                prompt = mx.array([[1, 5, 9, 13, 17, 21, 25, 29, 33, 37]], dtype=mx.int32)
                plan = K.install_kv_quant(q_model, "k8v8", sink_tokens=4)
                self.assertEqual(plan, {3: (8, 8), 7: (8, 8)})

                # Verify exposed attributes on wrapped self_attn (required by block_q35)
                inner3 = getattr(q_model.layers[3], "inner", q_model.layers[3])
                self.assertIsInstance(inner3.self_attn, K.QuantizedKVAttention)
                for attr in (
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "q_norm", "k_norm", "rope", "scale",
                    "num_attention_heads", "num_key_value_heads", "head_dim",
                ):
                    self.assertTrue(hasattr(inner3.self_attn, attr), attr)

                # Both model.make_cache() and model.language_model.make_cache() return quantized caches
                c_q = q_model.make_cache()
                c_lm = q_model.language_model.make_cache()
                self.assertIsInstance(c_q[0], ArraysCache)
                self.assertIsInstance(c_q[3], K.HadamardLloydMaxKVCache)
                self.assertIsInstance(c_lm[7], K.HadamardLloydMaxKVCache)
                self.assertEqual(c_q[3].k_bits, 8)
                self.assertEqual(c_q[3].sink_tokens, 4)

                c_ref = ref_model.make_cache()
                logits_ref = ref_model(prompt, cache=c_ref)
                logits_q = q_model(prompt, cache=c_q)
                mx.eval(logits_ref, logits_q)
                prefill_err = float(mx.abs(logits_ref - logits_q).max().item())
                self.assertLess(prefill_err, 2e-2, f"wrap={wrap} prefill err={prefill_err}")

                # Decode 3 tokens
                for tok in (41, 45, 49):
                    step_in = mx.array([[tok]], dtype=mx.int32)
                    lr = ref_model(step_in, cache=c_ref)
                    lq = q_model(step_in, cache=c_q)
                    mx.eval(lr, lq)
                    dec_err = float(mx.abs(lr - lq).max().item())
                    self.assertLess(dec_err, 2e-2, f"wrap={wrap} tok={tok} dec err={dec_err}")
                self.assertEqual(c_q[3].offset, 13)
                self.assertEqual(c_q[7].n_sink, 4)
                self.assertEqual(c_q[7].n_body, 9)

                # 3) Smoke-test q35_auto forward (prefill + decode)
                m_auto = self._make_mock_q35_model(num_layers=8, wrap_fast_layer=wrap)
                m_auto.load_weights(list(tree_flatten(ref_model.parameters())))
                auto_plan = K.install_kv_quant(m_auto, "q35_auto", sink_tokens=4)
                self.assertEqual(auto_plan, {3: (4, 3), 7: (4, 4)})
                c_auto = m_auto.make_cache()
                la_pre = m_auto(prompt, cache=c_auto)
                la_dec = m_auto(mx.array([[41]], dtype=mx.int32), cache=c_auto)
                mx.eval(la_pre, la_dec)
                self.assertEqual(la_pre.shape, (1, 10, 128))
                self.assertEqual(la_dec.shape, (1, 1, 128))
                self.assertFalse(bool(mx.isnan(la_pre).any().item()))
                self.assertFalse(bool(mx.isnan(la_dec).any().item()))

    def test_fast_q35_load_kv_mode_dispatch(self):
        mock_model = self._make_mock_q35_model(num_layers=8, wrap_fast_layer=True)
        import tempfile, json
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
            fake_manifest = {"tensors": {}, "dense": [], "matrices": {}}

            with (
                patch("compiler_artifact.read_artifact", return_value=({}, fake_manifest)),
                patch("mlx_lm.utils._get_classes", return_value=(lambda _a: mock_model, type("A", (), {"from_dict": staticmethod(lambda d: d)}))),
                patch("mlx_lm.utils.load_tokenizer", return_value="tok"),
                patch.dict(os.environ, {"LAMQ_Q35_KV": "q35_auto", "LAMQ_Q35_KV_SINK": "6"}),
            ):
                # Explicit fp16 overrides env and does NOT install kv quant
                m_fp16, _ = F.load(p, verify=False, fuse=False, kv_mode="fp16")
                self.assertIsInstance(m_fp16.make_cache()[3], KVCache)

                # kv_mode=None reads LAMQ_Q35_KV and LAMQ_Q35_KV_SINK
                m_kv, _ = F.load(p, verify=False, fuse=False, kv_mode=None)
                caches = m_kv.make_cache()
                self.assertIsInstance(caches[3], K.HadamardLloydMaxKVCache)
                self.assertEqual((caches[3].k_bits, caches[3].v_bits, caches[3].sink_tokens), (4, 3, 6))
                self.assertEqual((caches[7].k_bits, caches[7].v_bits, caches[7].sink_tokens), (4, 4, 6))


if __name__ == "__main__":
    unittest.main()
