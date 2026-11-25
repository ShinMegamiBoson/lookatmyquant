"""Unit and integration tests for Workstream 1A: RHT-128 + Lloyd-Max + Asymmetric K/V KV-cache quantization."""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_mlx.inkling.drafter import TREE_TOPOLOGIES
from runtime_mlx.inkling.fast_kernels import (
    _RHT128_MASKS_U32,
    gqa_relpos_sinks_sdpa_fast,
    irht128_rows_f32,
    kv_rht128_lm_dequantize,
    kv_rht128_lm_quantize,
    make_tree_spec,
    rht128_rows_f32,
)
from runtime_mlx.inkling.model import (
    Inkling2Cache,
    Inkling2Config,
    Inkling2LoopPackedMLX,
    PackedDecoderLayer,
    _resolve_slot_kv_bits,
    parse_global_kv_plan,
)
from runtime_mlx.prompt_cache_disk import DiskPromptCache, _model_fingerprint
from runtime_mlx.test_inkling import build_mini_packed_dir

# Exact unit-Gaussian Lloyd-Max levels and midpoints
NESTED_LM_CONSTANTS: dict[int, tuple[np.ndarray, np.ndarray]] = {
    2: (
        np.array([-1.5104176, -0.4527800, 0.4527800, 1.5104176], dtype=np.float32),
        np.array([-0.9815988, 0.0, 0.9815988], dtype=np.float32),
    ),
    3: (
        np.array([-2.1519457, -1.3439093, -0.7560053, -0.2450942, 0.2450942, 0.7560053, 1.3439093, 2.1519457], dtype=np.float32),
        np.array([-1.7479275, -1.0499573, -0.5005497, 0.0, 0.5005497, 1.0499573, 1.7479275], dtype=np.float32),
    ),
    4: (
        np.array(
            [
                -2.7325896, -2.0690172, -1.6180464, -1.2562312,
                -0.9423405, -0.6567591, -0.3880483, -0.1283950,
                0.1283950, 0.3880483, 0.6567591, 0.9423405,
                1.2562312, 1.6180464, 2.0690172, 2.7325896,
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                -2.4008034, -1.8435318, -1.4371388, -1.0992858,
                -0.7995498, -0.5224037, -0.2582217, 0.0,
                0.2582217, 0.5224037, 0.7995498, 1.0992858,
                1.4371388, 1.8435318, 2.4008034,
            ],
            dtype=np.float32,
        ),
    ),
}

KV_BIT_PAIRS: tuple[tuple[int, int], ...] = ((4, 4), (4, 3), (4, 2), (3, 3), (3, 2), (2, 2))


def _sylvester_np(k: int) -> np.ndarray:
    h = np.ones((1, 1), dtype=np.float64)
    for _ in range(k):
        h = np.block([[h, h], [h, -h]])
    return h


def _rht128_signs_np() -> np.ndarray:
    signs = np.empty(128, dtype=np.float64)
    for lane in range(32):
        for comp in range(4):
            bit = (_RHT128_MASKS_U32[comp] >> lane) & 1
            signs[4 * lane + comp] = -1.0 if bit else 1.0
    return signs


_H128_NP = _sylvester_np(7) / math.sqrt(128.0)
_SIGNS128_NP = _rht128_signs_np()
_RHT128_MAT = (_H128_NP * _SIGNS128_NP[None, :]).astype(np.float32)


def _rht128_np(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) @ _RHT128_MAT.T


def _irht128_np(y: np.ndarray) -> np.ndarray:
    return y.astype(np.float32) @ _RHT128_MAT


def _unpack_codes_np(packed: np.ndarray, bits: int) -> np.ndarray:
    """Unpack uint8 `(..., row_bytes)` back into `(..., 128)` integer codes."""
    prefix = packed.shape[:-1]
    flat = packed.reshape(-1, packed.shape[-1])
    n_rows = flat.shape[0]
    codes = np.zeros((n_rows, 128), dtype=np.uint32)
    if bits == 4:
        b = flat.astype(np.uint32)
        codes[:, 0::2] = b & 0xF
        codes[:, 1::2] = b >> 4
    elif bits == 2:
        b = flat.astype(np.uint32)
        codes[:, 0::4] = b & 0x3
        codes[:, 1::4] = (b >> 2) & 0x3
        codes[:, 2::4] = (b >> 4) & 0x3
        codes[:, 3::4] = b >> 6
    elif bits == 3:
        for lane in range(32):
            off = (lane * 3) >> 1
            sh = (lane & 1) * 4
            w12 = (flat[:, off].astype(np.uint32) | (flat[:, off + 1].astype(np.uint32) << 8)) >> sh
            codes[:, 4 * lane + 0] = w12 & 0x7
            codes[:, 4 * lane + 1] = (w12 >> 3) & 0x7
            codes[:, 4 * lane + 2] = (w12 >> 6) & 0x7
            codes[:, 4 * lane + 3] = (w12 >> 9) & 0x7
    else:
        raise ValueError(bits)
    return codes.reshape(*prefix, 128)


def _np_quant_dequant_ref(x_f16: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    levels, mids = NESTED_LM_CONSTANTS[bits]
    hv = _rht128_np(x_f16.astype(np.float32))
    rms = np.sqrt(np.mean(hv * hv, axis=-1, keepdims=True))
    sc_f16 = np.maximum(rms, 1e-6).astype(np.float16)
    z = hv / sc_f16.astype(np.float32)
    codes = np.searchsorted(mids, z, side="right").astype(np.uint32)
    hv_hat = levels[codes] * sc_f16.astype(np.float32)
    x_hat = _irht128_np(hv_hat).astype(np.float16)
    return codes, sc_f16, x_hat


def _ref_sdpa_fp32(
    q: mx.array,
    k_u8: mx.array,
    k_sc: mx.array,
    v_u8: mx.array,
    v_sc: mx.array,
    k_bits: int,
    v_bits: int,
    rel_st: mx.array,
    rel_proj_t_f16: mx.array,
    q_in: mx.array,
    sink_w_f16: mx.array,
    T: int,
    kv_len: int,
    kv_start: int,
    q_pos0: int,
    is_sliding: bool,
    tree_depth: mx.array | None = None,
    tree_mask: mx.array | None = None,
) -> np.ndarray:
    """Explicit FP32 GQA+RelPos+Sinks attention over dequantized K/V buffers."""
    k_deq = np.array(kv_rht128_lm_dequantize(k_u8[:, :kv_len, :], k_sc[:, :kv_len, :], k_bits), dtype=np.float32)
    v_deq = np.array(kv_rht128_lm_dequantize(v_u8[:, :kv_len, :], v_sc[:, :kv_len, :], v_bits), dtype=np.float32)
    q_np = np.array(q, dtype=np.float32)  # [T, 24, 128]
    rel_np = np.array(rel_st, dtype=np.float32)  # [T, 24, 16]
    rp_np = np.array(rel_proj_t_f16, dtype=np.float32)  # [1024, 16]
    qin_np = np.array(q_in, dtype=np.float32)  # [T, 2304]
    sw_np = np.array(sink_w_f16, dtype=np.float32)  # [24, 2304]

    k_exp = np.repeat(k_deq, 3, axis=0)  # [24, kv_len, 128]
    v_exp = np.repeat(v_deq, 3, axis=0)  # [24, kv_len, 128]
    q_h = np.transpose(q_np, (1, 0, 2))  # [24, T, 128]
    scores = np.matmul(q_h, np.transpose(k_exp, (0, 2, 1))) / math.sqrt(128.0)

    rel_logits = np.matmul(np.transpose(rel_np, (1, 0, 2)), rp_np.T)  # [24, T, 1024]
    tree_mode = tree_depth is not None and tree_mask is not None
    if tree_mode:
        td = np.array(tree_depth, dtype=np.int32)
        tm = np.array(tree_mask, dtype=np.uint32)
        kv_base = kv_len - T

    allowed = np.zeros((T, kv_len), dtype=bool)
    dist_mat = np.zeros((T, kv_len), dtype=np.int32)
    for t in range(T):
        q_pos = q_pos0 + (int(td[t]) if tree_mode else t)
        for j in range(kv_len):
            if tree_mode and j >= kv_base:
                u = j - kv_base
                if ((int(tm[t]) >> u) & 1) == 0:
                    continue
                d = int(td[t]) - int(td[u])
            else:
                d = q_pos - (kv_start + j)
            if d < 0 or (is_sliding and d >= 1024):
                continue
            allowed[t, j] = True
            dist_mat[t, j] = d

    valid_rel = allowed & (dist_mat < 1024)
    clipped_d = np.clip(dist_mat, 0, 1023)
    for h in range(24):
        for t in range(T):
            scores[h, t] += np.where(valid_rel[t], rel_logits[h, t, clipped_d[t]], 0.0)
    scores = np.where(allowed[None, :, :], scores, -1e30)

    sinks = (sw_np @ qin_np.T)[:, :, None]  # [24, T, 1]
    all_logits = np.concatenate([scores, sinks], axis=-1)
    m = np.max(all_logits, axis=-1, keepdims=True)
    exp_l = np.exp(all_logits - m)
    probs = exp_l[..., :-1] / np.sum(exp_l, axis=-1, keepdims=True)
    out = np.matmul(probs, v_exp)  # [24, T, 128]
    return np.transpose(out, (1, 0, 2)).reshape(T, 3072)


class TestInklingLMKV(unittest.TestCase):
    def test_rht128_orthogonality_and_involution(self) -> None:
        eye_err = float(np.max(np.abs(_RHT128_MAT @ _RHT128_MAT.T - np.eye(128, dtype=np.float32))))
        self.assertLess(eye_err, 1e-6)

        rng = np.random.default_rng(101)
        x_np = rng.standard_normal((17, 128)).astype(np.float32)
        y_np = rng.standard_normal((17, 128)).astype(np.float32)
        x_mx = mx.array(x_np)
        y_mx = mx.array(y_np)

        rx = rht128_rows_f32(x_mx)
        ry = rht128_rows_f32(y_mx)
        ix = irht128_rows_f32(rx)
        mx.eval(rx, ry, ix)

        fwd_err = float(np.max(np.abs(np.array(rx) - _rht128_np(x_np))))
        inv_err = float(np.max(np.abs(np.array(ix) - x_np)))
        dot_ref = np.sum(x_np * y_np, axis=-1)
        dot_rot = np.sum(np.array(rx) * np.array(ry), axis=-1)
        dot_err = float(np.max(np.abs(dot_rot - dot_ref)))
        self.assertLess(fwd_err, 1e-5)
        self.assertLess(inv_err, 1e-5)
        self.assertLess(dot_err, 1e-5)

    def test_quantize_dequantize_parity_all_bits(self) -> None:
        rng = np.random.default_rng(202)
        # Unit-scale rows for < 1e-3 FP16 roundtrip check, plus an all-zero row and an outlier row
        x_np = rng.normal(0.0, 0.5, size=(8, 32, 128)).astype(np.float16)
        x_np[0, 0, :] = 0.0
        x_outlier = (rng.normal(0.0, 3.0, size=(2, 16, 128))).astype(np.float16)

        for bits in (2, 3, 4):
            codes_mx, scales_mx = kv_rht128_lm_quantize(mx.array(x_np), bits)
            deq_mx = kv_rht128_lm_dequantize(codes_mx, scales_mx, bits)
            mx.eval(codes_mx, scales_mx, deq_mx)

            expected_bytes = 16 * bits
            self.assertEqual(tuple(codes_mx.shape), (8, 32, expected_bytes))
            self.assertEqual(codes_mx.dtype, mx.uint8)
            self.assertEqual(tuple(scales_mx.shape), (8, 32, 1))
            self.assertEqual(scales_mx.dtype, mx.float16)
            self.assertEqual(tuple(deq_mx.shape), (8, 32, 128))
            self.assertEqual(deq_mx.dtype, mx.float16)

            codes_ref, sc_ref, deq_ref = _np_quant_dequant_ref(x_np, bits)
            codes_got = _unpack_codes_np(np.array(codes_mx), bits)
            sc_got = np.array(scales_mx)
            deq_got = np.array(deq_mx)

            # Non-zero rows match FP16 scale bit-for-bit; zero-row clamp (1e-6) differs by <= 1 subnormal ULP (RTZ vs RN)
            self.assertTrue(np.array_equal(sc_got.reshape(-1)[1:], sc_ref.reshape(-1)[1:]), f"scale mismatch for bits={bits}")
            self.assertLess(float(np.max(np.abs(sc_got.astype(np.float32) - sc_ref.astype(np.float32)))), 1e-6)
            self.assertTrue(np.array_equal(codes_got, codes_ref), f"code mismatch for bits={bits}")
            max_abs = float(np.max(np.abs(deq_got.astype(np.float32) - deq_ref.astype(np.float32))))
            self.assertLess(max_abs, 1e-3, f"deq max_abs={max_abs} for bits={bits}")

            # Outlier magnitude check (relative FP16 ULP tolerance)
            co_mx, so_mx = kv_rht128_lm_quantize(mx.array(x_outlier), bits)
            do_mx = kv_rht128_lm_dequantize(co_mx, so_mx, bits)
            mx.eval(co_mx, so_mx, do_mx)
            co_ref, so_ref, do_ref = _np_quant_dequant_ref(x_outlier, bits)
            self.assertTrue(np.array_equal(np.array(so_mx), so_ref))
            self.assertTrue(np.array_equal(_unpack_codes_np(np.array(co_mx), bits), co_ref))
            np.testing.assert_allclose(np.array(do_mx, dtype=np.float32), do_ref.astype(np.float32), atol=1e-3, rtol=1e-3)

    def test_sdpa_all_modes_and_pairs(self) -> None:
        rng = np.random.default_rng(303)
        max_len = 2048
        k_raw = mx.array(rng.normal(0.0, 0.35, size=(8, max_len, 128)).astype(np.float16))
        v_raw = mx.array(rng.normal(0.0, 0.35, size=(8, max_len, 128)).astype(np.float16))
        q_all = mx.array(rng.normal(0.0, 0.6, size=(64, 24, 128)).astype(np.float32))
        qin_all = mx.array(rng.normal(0.0, 0.15, size=(64, 2304)).astype(np.float32))
        sw = mx.array(rng.normal(0.0, 0.05, size=(24, 2304)).astype(np.float16))
        rel_all = mx.array(rng.normal(0.0, 0.25, size=(64, 24, 16)).astype(np.float32))
        rp = mx.array(rng.normal(0.0, 0.2, size=(1024, 16)).astype(np.float16))

        q_k = {b: kv_rht128_lm_quantize(k_raw, b) for b in (2, 3, 4)}
        q_v = {b: kv_rht128_lm_quantize(v_raw, b) for b in (2, 3, 4)}
        for b in (2, 3, 4):
            mx.eval(*q_k[b], *q_v[b])

        tree = make_tree_spec(TREE_TOPOLOGIES["4-3"])
        worst_err = 0.0

        for kb, vb in KV_BIT_PAIRS:
            ku, ks = q_k[kb]
            vu, vs = q_v[vb]
            lm_spec = (ku, ks, vu, vs, kb, vb)

            # 1. Fused decode (T=1, 4, 8) at L=64 (non-split) and T=1, 4 at L=2048 (Split-K)
            for T, L, is_sl in ((1, 64, False), (4, 64, False), (8, 64, True), (1, 2048, False), (4, 2048, False)):
                q_pos0 = L - T
                got = gqa_relpos_sinks_sdpa_fast(
                    q_all[:T], None, rel_all[:T], rp, qin_all[:T], sw,
                    T, L, 0, q_pos0, max_len, is_sl, kv_lm_spec=lm_spec,
                )
                mx.eval(got)
                ref = _ref_sdpa_fp32(
                    q_all[:T], ku, ks, vu, vs, kb, vb,
                    rel_all[:T], rp, qin_all[:T], sw, T, L, 0, q_pos0, is_sl,
                )
                err = float(np.max(np.abs(np.array(got) - ref)))
                worst_err = max(worst_err, err)
                self.assertLess(err, 2e-3, f"fused/splitk pair=({kb},{vb}) T={T} L={L} err={err}")

            # 2. Tree verify (T=8) at L=64 (single split) and L=300 (Split-K)
            for L in (64, 300):
                q_pos0 = L - 8
                got = gqa_relpos_sinks_sdpa_fast(
                    q_all[:8], None, rel_all[:8], rp, qin_all[:8], sw,
                    8, L, 0, q_pos0, max_len, False,
                    tree_depth=tree["depth_mx"], tree_mask=tree["mask_mx"], kv_lm_spec=lm_spec,
                )
                mx.eval(got)
                ref = _ref_sdpa_fp32(
                    q_all[:8], ku, ks, vu, vs, kb, vb,
                    rel_all[:8], rp, qin_all[:8], sw, 8, L, 0, q_pos0, False,
                    tree_depth=tree["depth_mx"], tree_mask=tree["mask_mx"],
                )
                err = float(np.max(np.abs(np.array(got) - ref)))
                worst_err = max(worst_err, err)
                self.assertLess(err, 2e-3, f"tree pair=({kb},{vb}) L={L} err={err}")

            # 3. FlashAttention-2 prefill (T=64) at L=64 and L=256 (with kv_start offset)
            for L, kv_start, q_pos0 in ((64, 0, 0), (256, 0, 192)):
                got = gqa_relpos_sinks_sdpa_fast(
                    q_all[:64], None, rel_all[:64], rp, qin_all[:64], sw,
                    64, L, kv_start, q_pos0, max_len, False, kv_lm_spec=lm_spec,
                )
                mx.eval(got)
                ref = _ref_sdpa_fp32(
                    q_all[:64], ku, ks, vu, vs, kb, vb,
                    rel_all[:64], rp, qin_all[:64], sw, 64, L, kv_start, q_pos0, False,
                )
                err = float(np.max(np.abs(np.array(got) - ref)))
                worst_err = max(worst_err, err)
                self.assertLess(err, 2e-3, f"FA2 prefill pair=({kb},{vb}) L={L} err={err}")

        self.assertLess(worst_err, 2e-3)

    def test_config_plan_and_slot_resolution(self) -> None:
        plan = parse_global_kv_plan("5:4,3; 11:4,2; 17:4,2; 23:4,3")
        self.assertEqual(plan, {5: (4, 3), 11: (4, 2), 17: (4, 2), 23: (4, 3)})
        self.assertIsNone(parse_global_kv_plan(""))
        self.assertIsNone(parse_global_kv_plan(None))

        for bad_mode in ("fp4", "lm5", "bogus"):
            with self.assertRaises(ValueError):
                Inkling2Config(global_kv_mode=bad_mode)
        for bad_plan in ("5", "5:5,3", "5:4,1", "-1:4,3", "5:4,3,2"):
            with self.assertRaises(ValueError):
                parse_global_kv_plan(bad_plan)

        # Accepted modes including nested
        for mode in ("fp16", "fp8", "q4", "lm4", "lm4_3", "lm4_2", "lm3", "lm3_2", "lm2", "nested"):
            cfg = Inkling2Config(global_kv_mode=mode)
            self.assertEqual(cfg.global_kv_mode, mode)

        # Physical layer vs visit slot matching
        cfg_plan = Inkling2Config(global_kv_mode="fp16", global_kv_plan={5: (4, 3), 11: (4, 2), 29: (3, 2), 23: (4, 3)})
        self.assertIsNone(_resolve_slot_kv_bits(cfg_plan, 0, 0, True))
        self.assertIsNone(_resolve_slot_kv_bits(cfg_plan, 11, 11, True))  # sliding always None
        self.assertEqual(_resolve_slot_kv_bits(cfg_plan, 5, 5, False), (4, 3))
        self.assertEqual(_resolve_slot_kv_bits(cfg_plan, 11, 11, False), (4, 2))
        self.assertEqual(_resolve_slot_kv_bits(cfg_plan, 11, 29, False), (3, 2))  # slot overrides layer_idx
        self.assertEqual(_resolve_slot_kv_bits(cfg_plan, 11, 41, False), (4, 2))  # falls back to layer_idx
        self.assertEqual(_resolve_slot_kv_bits(cfg_plan, 23, 53, False), (4, 3))  # shadow coda
        self.assertIsNone(_resolve_slot_kv_bits(cfg_plan, 17, 17, False))

        # Env parsing via from_json
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.json"
            p.write_text(json.dumps({
                "hidden_size": 2304, "num_hidden_layers": 24, "vocab_size": 201024,
                "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128,
                "d_rel": 16, "rel_extent": 1024, "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1,
                "rms_norm_eps": 1e-6, "qk_l2_clip_threshold": 22.6274, "rel_l2_clip_threshold": 8.0,
                "local_layer_ids": [0, 1, 2, 3, 4], "final_logit_softcapping": 30.0,
                "sliding_window_size": 1024, "n_layers_per_attn_res_block": 6,
                "n_routed_experts": 128, "num_experts_per_tok": 8, "intermediate_size": 1152,
                "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0,
            }))
            old_env = {k: os.environ.get(k) for k in ("LAMQ_GLOBAL_KV", "LAMQ_GLOBAL_KV_PLAN", "LAMQ_SHARED_LOOP_KV")}
            try:
                os.environ["LAMQ_GLOBAL_KV"] = "lm4_2"
                os.environ["LAMQ_GLOBAL_KV_PLAN"] = "5:4,3;11:4,2;17:4,2;23:4,3"
                os.environ["LAMQ_SHARED_LOOP_KV"] = "1"
                cfg_env = Inkling2Config.from_json(p)
                self.assertEqual(cfg_env.global_kv_mode, "lm4_2")
                self.assertEqual(cfg_env.global_kv_plan, {5: (4, 3), 11: (4, 2), 17: (4, 2), 23: (4, 3)})
                self.assertTrue(cfg_env.shared_loop_kv)
            finally:
                for k, v in old_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def test_cache_lifecycle_rollback_clone_and_kv_bytes(self) -> None:
        rng = np.random.default_rng(404)
        prefix = mx.array(rng.normal(0.0, 0.25, size=(2, 8, 250, 128)).astype(np.float16))
        p0 = mx.array(rng.normal(0.0, 0.25, size=(2, 8, 8, 128)).astype(np.float16))
        p1 = mx.array(rng.normal(0.0, 0.25, size=(2, 8, 8, 128)).astype(np.float16))
        p2 = mx.array(rng.normal(0.0, 0.25, size=(2, 8, 8, 128)).astype(np.float16))
        step1 = mx.array(rng.normal(0.0, 0.25, size=(2, 8, 4, 128)).astype(np.float16))

        mode_names = {
            (4, 4): "lm4", (4, 3): "lm4_3", (4, 2): "lm4_2",
            (3, 3): "lm3", (3, 2): "lm3_2", (2, 2): "lm2",
        }
        for (kb, vb), mode in mode_names.items():
            cfg = Inkling2Config(global_kv_mode=mode, shared_loop_kv=True)
            cache = Inkling2Cache(cfg)
            self.assertEqual(len(cache.kv_k_u8), cfg.total_visits)

            # shared_loop_kv=True: early passes use commit=False, final pass commits
            _, n0, _, _, _, _ = cache.update_kv_f16(41, prefix, False, commit=False, layer_idx=11)
            self.assertEqual(n0, 250)
            self.assertEqual(cache.kv_len[41], 0)
            _, n2, _, _, _, _ = cache.update_kv_f16(41, prefix, False, commit=True, layer_idx=11)
            self.assertEqual(n2, 250)
            self.assertEqual(cache.kv_len[41], 250)
            self.assertEqual(cache.kv_len[11], 0)
            self.assertEqual(cache.kv_len[29], 0)

            spec = cache.lm_spec(41)
            self.assertIsNotNone(spec)
            self.assertEqual(spec[4:], (kb, vb))
            # Expected bytes for cap=256, 8 heads, (k_bytes + v_bytes) + 2 * 2 bytes (k_sc + v_sc f16)
            expected_bytes = 256 * 8 * (16 * kb + 16 * vb + 4)
            self.assertEqual(cache.kv_bytes(), expected_bytes)

            # Clone isolation
            snap = cache.clone()
            snap.update_kv_f16(41, step1, False, commit=True, layer_idx=11)
            self.assertEqual(cache.kv_len[41], 250)
            self.assertEqual(snap.kv_len[41], 254)

            # Speculative linear rollback (`commit_spec_verify`)
            c_lin = cache.clone()
            c_lin.begin_spec_verify()
            c_lin.update_kv_f16(41, p0, False, commit=False, layer_idx=11)
            c_lin.update_kv_f16(41, p1, False, commit=False, layer_idx=11)
            c_lin.update_kv_f16(41, p2, False, commit=True, layer_idx=11)
            self.assertEqual(c_lin.kv_len[41], 258)
            c_lin.commit_spec_verify(3, 8)
            self.assertEqual(c_lin.kv_len[41], 253)

            ref_lin = Inkling2Cache(cfg)
            ref_lin.update_kv_f16(41, mx.concatenate([prefix, p2[:, :, :3, :]], axis=2), False, commit=True, layer_idx=11)
            for a, b in zip(c_lin.state, ref_lin.state, strict=True):
                np.testing.assert_array_equal(np.array(a[:, :253, :]), np.array(b[:, :253, :]))

            # Speculative tree rollback (`commit_tree_verify`) crossing cap 256 -> 512
            c_tree = cache.clone()
            c_tree.begin_spec_verify()
            c_tree.update_kv_f16(41, p0, False, commit=False, layer_idx=11)
            c_tree.update_kv_f16(41, p1, False, commit=False, layer_idx=11)
            c_tree.update_kv_f16(41, p2, False, commit=True, layer_idx=11)
            accepted = [0, 5, 6, 7]
            c_tree.commit_tree_verify(accepted, 8)
            self.assertEqual(c_tree.kv_len[41], 254)
            self.assertEqual(c_tree.kv_cap[41], 512)
            self.assertEqual(c_tree.kv_bytes(), 512 * 8 * (16 * kb + 16 * vb + 4))

            ref_tree = Inkling2Cache(cfg)
            ref_tree.update_kv_f16(41, mx.concatenate([prefix, p2[:, :, accepted, :]], axis=2), False, commit=True, layer_idx=11)
            for a, b in zip(c_tree.state, ref_tree.state, strict=True):
                np.testing.assert_array_equal(np.array(a[:, :254, :]), np.array(b[:, :254, :]))

        # Deduplication in kv_bytes when slots are explicitly aliased
        c_alias = Inkling2Cache(Inkling2Config(global_kv_mode="lm4_2"))
        c_alias.update_kv_f16(5, prefix[:, :, :32, :], False, layer_idx=5)
        b_single = c_alias.kv_bytes()
        c_alias.kv_k_u8[11] = c_alias.kv_k_u8[5]
        c_alias.kv_v_u8[11] = c_alias.kv_v_u8[5]
        c_alias.kv_k_sc[11] = c_alias.kv_k_sc[5]
        c_alias.kv_v_sc[11] = c_alias.kv_v_sc[5]
        self.assertEqual(c_alias.kv_bytes(), b_single)

    def test_forward_fast_layer_wiring_and_mini_model_e2e(self) -> None:
        # 1. Synthetic H=2304 PackedDecoderLayer.forward_fast test verifying LM wiring + parity vs FP16
        rng = np.random.default_rng(505)
        cfg_lm = Inkling2Config(global_kv_mode="fp16", global_kv_plan={11: (4, 2)}, shared_loop_kv=True)
        cfg_fp16 = Inkling2Config(global_kv_mode="fp16", shared_loop_kv=True)

        layer = object.__new__(PackedDecoderLayer)
        layer.layer_idx = 11
        layer.cfg = cfg_lm
        layer._fast = True
        layer.is_sliding = False
        layer.is_loop = True
        layer.is_block_boundary = False
        layer.attn_in_q = None
        layer.mlp_in_q = mx.array(rng.normal(0.0, 0.1, size=(2304,)).astype(np.float32))
        layer.q_in_ln = mx.ones((2304,), dtype=mx.float32)
        layer.k_in_ln = mx.ones((2304,), dtype=mx.float32)
        layer.v_in_ln = mx.ones((2304,), dtype=mx.float32)
        layer.qkv_in_ln = mx.stack([layer.q_in_ln, layer.k_in_ln, layer.v_in_ln], axis=0)
        layer.post_attn_ln = mx.ones((2304,), dtype=mx.float32)
        layer.q_scale = mx.ones((24, 128), dtype=mx.float32)
        layer.rel_scale = mx.ones((24, 16), dtype=mx.float32)
        layer.rel_proj = mx.array(rng.normal(0.0, 0.1, size=(16, 1024)).astype(np.float32))
        layer.rel_proj_t_f16 = layer.rel_proj.T.astype(mx.float16)
        layer.sink_w = mx.array(rng.normal(0.0, 0.02, size=(24, 2304)).astype(np.float32))
        layer.sink_w_f16 = layer.sink_w.astype(mx.float16)
        layer.k_dyn_st = mx.array(rng.normal(0.0, 0.1, size=(4, 1024)).astype(np.float32))
        layer.k_dyn_pr = mx.array(rng.normal(0.0, 0.02, size=(32, 2304)).astype(np.float32))
        layer.v_dyn_st = mx.array(rng.normal(0.0, 0.1, size=(4, 1024)).astype(np.float32))
        layer.v_dyn_pr = mx.array(rng.normal(0.0, 0.02, size=(32, 2304)).astype(np.float32))
        layer.k_dyn_pr_f16 = layer.k_dyn_pr.astype(mx.float16)
        layer.v_dyn_pr_f16 = layer.v_dyn_pr.astype(mx.float16)
        layer.post_attn_conv_w = mx.array(rng.normal(0.0, 0.1, size=(4, 2304)).astype(np.float32))
        layer.post_mlp_conv_w = mx.array(rng.normal(0.0, 0.1, size=(4, 2304)).astype(np.float32))
        layer.qkv_out_rots = [None, None, None, None, None]
        layer.has_qkv_out_rot = False
        layer.o_out_rot = None

        w_qkv = mx.array(rng.normal(0.0, 0.05, size=(8576, 2304)).astype(np.float32))
        w_o = mx.array(rng.normal(0.0, 0.05, size=(2304, 3072)).astype(np.float32))
        def _qkv_parts(x_rot: mx.array, T: int, W_cached=None):
            x3 = x_rot.reshape(5, T, 2304).astype(mx.float32)
            return (
                x3[0] @ w_qkv[:3072].T,
                x3[1] @ w_qkv[3072:4096].T,
                x3[2] @ w_qkv[4096:5120].T,
                x3[3] @ w_qkv[5120:5504].T,
                x3[4] @ w_qkv[5504:8576].T,
            )

        layer.qkv_stage = SimpleNamespace(
            signs_in=mx.ones((2304,), dtype=mx.float32),
            matmul_rotated_f16=lambda x_rot, T, W_cached=None: mx.concatenate(_qkv_parts(x_rot, T, W_cached), axis=-1),
            matmul_rotated_sg8_f16=lambda x_rot, T, **kw: mx.concatenate(_qkv_parts(x_rot, T), axis=-1),
            matmul_rotated_f16_parts=_qkv_parts,
        )
        layer.o_stage = SimpleNamespace(
            signs_in=mx.ones((3072,), dtype=mx.float32),
            matmul_rotated_f16=lambda o_rot, T, W_cached=None: o_rot.astype(mx.float32) @ w_o.T,
        )
        layer.moe = SimpleNamespace(
            gu_signs_in=mx.ones((2304,), dtype=mx.float32),
            forward_fast_fused_prerot=lambda h_mlp_f16, x_rot_gu, residual, post_mlp_conv_w, cache, cache_slot, T, tree_lag=None: residual,
        )

        c_lm = Inkling2Cache(cfg_lm)
        c_fp = Inkling2Cache(cfg_fp16)
        h_pref = mx.array(rng.normal(0.0, 0.3, size=(12, 2304)).astype(np.float32))
        h_dec = mx.array(rng.normal(0.0, 0.3, size=(1, 2304)).astype(np.float32))

        y_pref_lm = layer.forward_fast(h_pref, None, 0, 0, 12, c_lm, 11, kv_slot=41, kv_commit=False)
        y_pref_lm = layer.forward_fast(h_pref, None, 0, 0, 12, c_lm, 41, kv_slot=41, kv_commit=True)
        y_dec_lm = layer.forward_fast(h_dec, None, 0, 12, 1, c_lm, 41, kv_slot=41, kv_commit=True)

        layer.cfg = cfg_fp16
        y_pref_fp = layer.forward_fast(h_pref, None, 0, 0, 12, c_fp, 11, kv_slot=41, kv_commit=False)
        y_pref_fp = layer.forward_fast(h_pref, None, 0, 0, 12, c_fp, 41, kv_slot=41, kv_commit=True)
        y_dec_fp = layer.forward_fast(h_dec, None, 0, 12, 1, c_fp, 41, kv_slot=41, kv_commit=True)
        mx.eval(y_pref_lm, y_dec_lm, y_pref_fp, y_dec_fp)

        self.assertIsNotNone(c_lm.lm_spec(41))
        self.assertEqual(c_lm.kv_k_bits[41], 4)
        self.assertEqual(c_lm.kv_v_bits[41], 2)
        self.assertIsNone(c_lm.kv_f16[41])
        self.assertEqual(c_lm.kv_len[41], 13)
        self.assertLess(c_lm.kv_bytes(), c_fp.kv_bytes())
        rel_dec = float(mx.linalg.norm(y_dec_lm.astype(mx.float32) - y_dec_fp.astype(mx.float32)) / mx.linalg.norm(y_dec_fp.astype(mx.float32)))
        self.assertLess(rel_dec, 0.05)

        # 2. Miniature Inkling2LoopPackedMLX end-to-end with LAMQ_GLOBAL_KV + LAMQ_GLOBAL_KV_PLAN + LAMQ_SHARED_LOOP_KV
        with tempfile.TemporaryDirectory() as td:
            pdir = Path(td)
            build_mini_packed_dir(pdir)
            old_env = {k: os.environ.get(k) for k in ("LAMQ_GLOBAL_KV", "LAMQ_GLOBAL_KV_PLAN", "LAMQ_SHARED_LOOP_KV")}
            try:
                os.environ["LAMQ_GLOBAL_KV"] = "lm4_2"
                os.environ["LAMQ_GLOBAL_KV_PLAN"] = "1:4,3;3:3,2;5:4,3"
                os.environ["LAMQ_SHARED_LOOP_KV"] = "1"
                model = Inkling2LoopPackedMLX(pdir)
            finally:
                for k, v in old_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

            self.assertEqual(model.cfg.global_kv_mode, "lm4_2")
            self.assertEqual(model.cfg.global_kv_plan, {1: (4, 3), 3: (3, 2), 5: (4, 3)})
            self.assertTrue(model.cfg.shared_loop_kv)

            logits, cache = model.forward_all_logits([1, 2, 3, 4, 5, 6])
            mx.eval(logits)
            self.assertTrue(bool(mx.all(mx.isfinite(logits))))
            self.assertEqual(cache.seq_len, 6)
            self.assertGreater(cache.kv_bytes(), 0)
            self.assertIsNone(cache.keys[3])
            self.assertIsNone(cache.keys[7])
            self.assertIsNotNone(cache.keys[9])

            snap = cache.clone()
            c_spec = snap.clone()
            c_spec.begin_spec_verify()
            _, c_spec = model.forward_all_logits([7, 8, 9, 10], cache=c_spec)
            c_spec.commit_spec_verify(2, 4)
            self.assertEqual(c_spec.seq_len, 8)
            self.assertEqual(snap.seq_len, 6)

    def test_disk_prompt_cache_lm_delta_and_fingerprint(self) -> None:
        rng = np.random.default_rng(606)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdir = root / "model"
            pdir.mkdir()
            (pdir / "manifest.json").write_text('{"format":"inkling2-packed-v1"}')

            def _make_stub(mode: str, plan: dict[int, tuple[int, int]] | None):
                cfg = Inkling2Config(
                    num_hidden_layers=2,
                    local_layer_ids=(0,),
                    loop_start=1,
                    loop_end=1,
                    loop_passes=1,
                    global_kv_mode=mode,
                    global_kv_plan=plan,
                )
                return SimpleNamespace(packed_dir=pdir, cfg=cfg, make_cache=lambda: Inkling2Cache(cfg))

            m_42 = _make_stub("lm4_2", {1: (4, 3)})
            m_32 = _make_stub("lm3_2", {1: (4, 3)})
            m_p2 = _make_stub("lm4_2", {1: (3, 2)})
            m_none = _make_stub("lm4_2", None)
            fps = {
                _model_fingerprint(m_42),
                _model_fingerprint(m_32),
                _model_fingerprint(m_p2),
                _model_fingerprint(m_none),
            }
            self.assertEqual(len(fps), 4)

            slot_dir = root / "slots"
            dc = DiskPromptCache(slot_dir, max_entries=1, min_tokens=4, max_disk_gb=1.0)
            dc._owner = (m_42, None)

            d_full = mx.array(rng.normal(0.0, 0.25, size=(2, 8, 10, 128)).astype(np.float16))
            c1 = m_42.make_cache()
            c1.seq_len = 6
            c1.update_kv_f16(0, d_full[:, :, :6, :], True, layer_idx=0)
            c1.update_kv_f16(1, d_full[:, :, :6, :], False, layer_idx=1)
            mx.eval(*c1.state)
            dc._store([10, 11, 12, 13, 14, 15], c1, None)

            c2 = c1.clone()
            c2.seq_len = 10
            c2.update_kv_f16(0, d_full[:, :, 6:10, :], True, layer_idx=0)
            c2.update_kv_f16(1, d_full[:, :, 6:10, :], False, layer_idx=1)
            mx.eval(*c2.state)
            dc._store([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], c2, None)

            metas = [json.loads(p.read_text()) for p in sorted(slot_dir.glob("*.meta.json"))]
            self.assertEqual(len(metas), 2)
            for m in metas:
                self.assertEqual(m["global_kv_mode"], "lm4_2")
                self.assertEqual(m["global_kv_plan"], "1:4,3")

            dc2 = DiskPromptCache(slot_dir, max_entries=2, min_tokens=4, max_disk_gb=1.0)
            dc2._owner = (m_42, None)
            hit = dc2._match(list(range(10, 21)), drafting=False)
            self.assertIsNotNone(hit)
            rc = hit["cache"]
            self.assertEqual(rc.seq_len, 10)
            self.assertEqual(rc.kv_k_bits[1], 4)
            self.assertEqual(rc.kv_v_bits[1], 3)
            self.assertEqual(rc.kv_bytes(), c2.kv_bytes())
            for a, b in zip(rc.state, c2.state, strict=True):
                np.testing.assert_array_equal(np.array(a), np.array(b))

            # Different global_kv_plan must miss disk cache on a fresh server cache instance
            dc3 = DiskPromptCache(slot_dir, max_entries=2, min_tokens=4, max_disk_gb=1.0)
            dc3._owner = (m_p2, None)
            self.assertIsNone(dc3._match(list(range(10, 21)), drafting=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
