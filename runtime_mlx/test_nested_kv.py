"""Synthetic & kernel-level unit tests for Workstream 3: Nested 4->3->2 Bit-Plane Paged KV."""
from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path
from statistics import NormalDist

import mlx.core as mx
import numpy as np

# Ensure runtime_mlx package resolves when invoked directly (`python test_nested_kv.py`).
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from runtime_mlx.inkling.batch import BatchSequence, PagedInklingPool, _prepare_batch_meta
from runtime_mlx.inkling.model import Inkling2Config
from runtime_mlx.inkling.nested_kv_kernels import (
    NESTED2_LUT,
    NESTED3_LUT,
    NESTED4_BOUNDS,
    NESTED4_LUT,
    PAGED_BLOCK_SIZE,
    RHT128_SIGNS_NP,
    gqa_paged_nested_coarse_score,
    gqa_paged_nested_sdpa_fast,
    kv_h128_nested4_quantize,
    kv_h128_nested_dequantize,
    nested_rht128_rows_f32,
)


def _make_nested_cfg(**kwargs) -> Inkling2Config:
    """Construct Inkling2Config with global_kv_mode='nested' (supporting pre-1A model.py fallback)."""
    try:
        return Inkling2Config(global_kv_mode="nested", **kwargs)
    except ValueError:
        cfg = Inkling2Config(**kwargs)
        cfg.global_kv_mode = "nested"
        return cfg


def _ref_nested_bounds_and_luts(alpha: float = 0.4) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Exact Gaussian compander bounds and centroids."""
    nd = NormalDist()
    bounds4 = np.array([nd.inv_cdf(i / 16.0) / math.sqrt(alpha) for i in range(1, 16)], dtype=np.float64)
    luts: dict[int, np.ndarray] = {}
    for bits in (4, 3, 2):
        n = 1 << bits
        edges = [-math.inf] + [nd.inv_cdf(i / n) / math.sqrt(alpha) for i in range(1, n)] + [math.inf]
        lv = np.empty((n,), dtype=np.float64)
        for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            pa = 0.0 if math.isinf(a) else nd.pdf(a)
            pb = 0.0 if math.isinf(b) else nd.pdf(b)
            lv[j] = (pa - pb) / (nd.cdf(b) - nd.cdf(a))
        luts[bits] = lv
    return bounds4, luts


def _rht128_matrix_np() -> np.ndarray:
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < 128:
        h = np.block([[h, h], [h, -h]])
    h = h / math.sqrt(128.0)
    s = RHT128_SIGNS_NP.astype(np.float64)
    return (s[:, None] * h) * s[None, :]


def _fp32_paged_sdpa_ref(
    q: np.ndarray,           # [B * K, 24, 128]
    kv_seq_list: list[np.ndarray],  # per b: [2, 8, kv_len, 128] float32
    rel_st: np.ndarray,      # [B * K, 24, 16]
    rel_proj_t: np.ndarray,  # [>=1024, 16]
    q_in: np.ndarray,        # [B * K, 2304]
    sink_w: np.ndarray,      # [24, 2304]
    seq_lens: list[int],
    q_pos0: list[int],
    kv_starts: list[int],
    B: int,
    K: int,
    *,
    is_sliding: bool = False,
    block_mask: np.ndarray | None = None,
) -> np.ndarray:
    out = np.zeros((B * K, 24, 128), dtype=np.float32)
    scale = 1.0 / math.sqrt(128.0)
    for b in range(B):
        kv_len = seq_lens[b]
        kvs = kv_starts[b]
        kv_b = kv_seq_list[b]  # [2, 8, kv_len, 128]
        for lt in range(K):
            tg = b * K + lt
            q_pos = q_pos0[b] + lt
            min_pos = (q_pos - 1023) if is_sliding else 0
            j_lo = max(0, min_pos - kvs) if min_pos > kvs else 0
            j_hi = min(kv_len, q_pos - kvs + 1) if q_pos >= kvs else 0
            valid_js = [j for j in range(j_lo, j_hi) if block_mask is None or block_mask[b, j // 64]]
            for h in range(24):
                kvh = h // 3
                sink_logit = float(np.dot(q_in[tg].astype(np.float64), sink_w[h].astype(np.float64)))
                if not valid_js:
                    out[tg, h] = 0.0
                    continue
                js = np.asarray(valid_js, dtype=np.int64)
                k_mat = kv_b[0, kvh, js, :].astype(np.float64)
                v_mat = kv_b[1, kvh, js, :].astype(np.float64)
                logits = (k_mat @ q[tg, h].astype(np.float64)) * scale
                dists = q_pos - (kvs + js)
                rel_mask = dists < 1024
                if np.any(rel_mask):
                    logits[rel_mask] += rel_proj_t[dists[rel_mask]].astype(np.float64) @ rel_st[tg, h].astype(np.float64)
                m = max(sink_logit, float(np.max(logits)))
                exp_l = np.exp(logits - m)
                denom = math.exp(sink_logit - m) + float(np.sum(exp_l))
                out[tg, h] = ((exp_l @ v_mat) / denom).astype(np.float32)
    return out.reshape(B * K, 3072)


class TestNestedKVPaged(unittest.TestCase):
    def test_rht128_and_bit_exact_truncation_vs_ref(self) -> None:
        rng = np.random.default_rng(20261125)
        bounds4_ref, luts_ref = _ref_nested_bounds_and_luts(0.4)
        np.testing.assert_allclose(NESTED4_BOUNDS, bounds4_ref, atol=1e-6, rtol=0)
        np.testing.assert_allclose(NESTED4_LUT, luts_ref[4], atol=1e-6, rtol=0)
        np.testing.assert_allclose(NESTED3_LUT, luts_ref[3], atol=1e-6, rtol=0)
        np.testing.assert_allclose(NESTED2_LUT, luts_ref[2], atol=1e-6, rtol=0)

        R_ref = _rht128_matrix_np()
        np.testing.assert_allclose(R_ref @ R_ref, np.eye(128), atol=1e-12)

        x = mx.array(rng.standard_normal((13, 128)).astype(np.float32))
        rx = nested_rht128_rows_f32(x)
        rrx = nested_rht128_rows_f32(rx)
        mx.eval(rx, rrx)
        max_rht_err = float(np.max(np.abs(np.array(rx) - np.array(x) @ R_ref.T)))
        max_inv_err = float(np.max(np.abs(np.array(rrx) - np.array(x))))
        self.assertLess(max_rht_err, 1e-5)
        self.assertLess(max_inv_err, 1e-5)

        kv_f16 = mx.array(rng.standard_normal((19, 2, 8, 128)).astype(np.float16))
        base, sc, p2, p3 = kv_h128_nested4_quantize(kv_f16)
        self.assertEqual(base.shape, (19, 2, 8, 2, 4))
        self.assertEqual(sc.shape, (19, 2, 8))
        self.assertEqual(p2.shape, (19, 2, 8, 4))
        self.assertEqual(p3.shape, (19, 2, 8, 4))

        dq4 = kv_h128_nested_dequantize(base, sc, p2, p3)
        dq3 = kv_h128_nested_dequantize(base, sc, p2, None)
        dq2 = kv_h128_nested_dequantize(base, sc, None, None)
        mx.eval(dq4, dq3, dq2)

        # Explicit FP64 compander reference -> R_128 inverse
        kv_np = np.array(kv_f16, dtype=np.float32).astype(np.float64)
        hv_ref = kv_np @ R_ref.T
        sc_np = np.array(sc, dtype=np.float16).astype(np.float64)[..., None]
        q4_idx = np.digitize(hv_ref / sc_np, bounds4_ref, right=False)  # matches (x >= bound)
        ref4 = (luts_ref[4][q4_idx] * sc_np) @ R_ref.T
        ref3 = (luts_ref[3][q4_idx >> 1] * sc_np) @ R_ref.T
        ref2 = (luts_ref[2][q4_idx >> 2] * sc_np) @ R_ref.T

        err4 = float(np.max(np.abs(np.array(dq4) - ref4)))
        err3 = float(np.max(np.abs(np.array(dq3) - ref3)))
        err2 = float(np.max(np.abs(np.array(dq2) - ref2)))
        print(f"  [truncation vs fp64 ref] err4={err4:.2e}, err3={err3:.2e}, err2={err2:.2e}")
        self.assertLess(err4, 1e-3)
        self.assertLess(err3, 1e-3)
        self.assertLess(err2, 1e-3)

        with self.assertRaises(ValueError):
            kv_h128_nested_dequantize(base, sc, None, p3)

        # Single-row (128,) edge case (must not trigger Metal constant-vs-device address space error)
        v1 = mx.array(rng.standard_normal((128,)).astype(np.float16))
        b1, s1, p2_1, p3_1 = kv_h128_nested4_quantize(v1)
        d1_4 = kv_h128_nested_dequantize(b1, s1, p2_1, p3_1)
        d1_3 = kv_h128_nested_dequantize(b1, s1, p2_1, None)
        d1_2 = kv_h128_nested_dequantize(b1, s1, None, None)
        mx.eval(d1_4, d1_3, d1_2)
        self.assertEqual(d1_4.shape, (128,))
        self.assertTrue(bool(mx.all(mx.isfinite(d1_4))))

    def test_mixed_precision_paged_sdpa_decode_verify_and_prefill(self) -> None:
        rng = np.random.default_rng(7744)
        rel_p = mx.array(rng.standard_normal((1025, 16)).astype(np.float16) * 0.15)
        sw = mx.array(rng.standard_normal((24, 2304)).astype(np.float16) * 0.05)

        # Build 6 logical blocks with mixed [2, 3, 4, 2, 3, 4] bitwidths and shuffled physical blocks
        n_phys = 16
        base_pool = mx.zeros((n_phys * 64, 2, 8, 2, 4), dtype=mx.uint32)
        sc_pool = mx.zeros((n_phys * 64, 2, 8), dtype=mx.float16)
        p2_pool = mx.zeros((n_phys * 64, 2, 8, 4), dtype=mx.uint32)
        p3_pool = mx.zeros((n_phys * 64, 2, 8, 4), dtype=mx.uint32)
        bp2_np = np.full((n_phys,), -1, dtype=np.int32)
        bp3_np = np.full((n_phys,), -1, dtype=np.int32)

        # Shared prefix (3 blocks: 4b, 2b, 3b) + per-seq tails
        pfx_kv = mx.array(rng.standard_normal((192, 2, 8, 128)).astype(np.float16) * 0.5)
        pfx_base, pfx_sc, pfx_p2, pfx_p3 = kv_h128_nested4_quantize(pfx_kv)
        pfx_phys = [5, 2, 9]
        pfx_bits = [4, 2, 3]
        for bi, (gb, bits) in enumerate(zip(pfx_phys, pfx_bits)):
            lo, hi = bi * 64, (bi + 1) * 64
            base_pool[gb * 64 : (gb + 1) * 64] = pfx_base[lo:hi]
            sc_pool[gb * 64 : (gb + 1) * 64] = pfx_sc[lo:hi]
            if bits >= 3:
                p2_blk = gb + 1
                bp2_np[gb] = p2_blk
                p2_pool[p2_blk * 64 : (p2_blk + 1) * 64] = pfx_p2[lo:hi]
            if bits == 4:
                p3_blk = 15 - gb
                bp3_np[gb] = p3_blk
                p3_pool[p3_blk * 64 : (p3_blk + 1) * 64] = pfx_p3[lo:hi]

        # Test batched K=1 and K=4 with B=3 sequences (different lengths, mixed bit-widths, split-k triggered >256!)
        lens = [192 + 19, 192 + 95, 192 + 131]  # spans up to 323 tokens -> n_splits > 1
        seq_tails = [mx.array(rng.standard_normal((ln - 192, 2, 8, 128)).astype(np.float16) * 0.5) for ln in lens]
        tail_phys_per_seq = [[1], [3, 7], [4, 6, 11]]
        tail_bits_per_seq = [[3], [2, 4], [2, 3, 4]]

        for b, (ln, t_kv, t_phys, t_bits) in enumerate(zip(lens, seq_tails, tail_phys_per_seq, tail_bits_per_seq)):
            tb, tsc, tp2, tp3 = kv_h128_nested4_quantize(t_kv)
            for si, (gb, bits) in enumerate(zip(t_phys, t_bits)):
                lo = si * 64
                hi = min(ln - 192, lo + 64)
                cnt = hi - lo
                base_pool[gb * 64 : gb * 64 + cnt] = tb[lo:hi]
                sc_pool[gb * 64 : gb * 64 + cnt] = tsc[lo:hi]
                if bits >= 3:
                    p2_blk = gb
                    bp2_np[gb] = p2_blk
                    p2_pool[p2_blk * 64 : p2_blk * 64 + cnt] = tp2[lo:hi]
                if bits == 4:
                    p3_blk = gb
                    bp3_np[gb] = p3_blk
                    p3_pool[p3_blk * 64 : p3_blk * 64 + cnt] = tp3[lo:hi]

        mx.eval(base_pool, sc_pool, p2_pool, p3_pool)

        # Build exact dequantized FP32 KV reference per sequence matching each block's resident planes
        ref_kv_seqs: list[np.ndarray] = []
        max_blks = max((ln + 63) // 64 for ln in lens)
        bt_np = np.zeros((3, max_blks), dtype=np.uint32)
        for b, (ln, t_phys) in enumerate(zip(lens, tail_phys_per_seq)):
            seq_blks = pfx_phys + t_phys
            bt_np[b, : len(seq_blks)] = seq_blks
            chunks = []
            for bi, gb in enumerate(seq_blks):
                cnt = min(64, ln - bi * 64)
                b_sl = base_pool[gb * 64 : gb * 64 + cnt]
                s_sl = sc_pool[gb * 64 : gb * 64 + cnt]
                p2_sl = p2_pool[bp2_np[gb] * 64 : bp2_np[gb] * 64 + cnt] if bp2_np[gb] >= 0 else None
                p3_sl = p3_pool[bp3_np[gb] * 64 : bp3_np[gb] * 64 + cnt] if bp3_np[gb] >= 0 else None
                dq = kv_h128_nested_dequantize(b_sl, s_sl, p2_sl, p3_sl)
                mx.eval(dq)
                chunks.append(np.array(dq))
            ref_kv_seqs.append(np.concatenate(chunks, axis=0).transpose(1, 2, 0, 3))

        bp2_mx, bp3_mx = mx.array(bp2_np), mx.array(bp3_np)
        bt_mx = mx.array(bt_np)
        sl_mx = mx.array(np.array(lens, dtype=np.uint32))
        ks_mx = mx.zeros((3,), dtype=mx.uint32)

        for K in (1, 4):
            T = 3 * K
            q = mx.array(rng.standard_normal((T, 24, 128)).astype(np.float32))
            rel_st = mx.array(rng.standard_normal((T, 24, 16)).astype(np.float32) * 0.25)
            qin = mx.array(rng.standard_normal((T, 2304)).astype(np.float32) * 0.1)
            q_pos0 = [ln - K for ln in lens]
            qp_mx = mx.array(np.array(q_pos0, dtype=np.uint32))
            out = gqa_paged_nested_sdpa_fast(
                q, base_pool, sc_pool, p2_pool, p3_pool, rel_st, rel_p, qin, sw,
                bt_mx, bp2_mx, bp3_mx, sl_mx, qp_mx, ks_mx, 3, K, max(lens),
            )
            mx.eval(out)
            ref = _fp32_paged_sdpa_ref(
                np.array(q), ref_kv_seqs, np.array(rel_st), np.array(rel_p),
                np.array(qin), np.array(sw), lens, q_pos0, [0, 0, 0], 3, K,
            )
            err = float(np.max(np.abs(np.array(out) - ref)))
            print(f"  [mixed 2/3/4-bit SDPA B=3, K={K}] max_abs={err:.2e}")
            self.assertLess(err, 2e-3)

            # Also verify active_block_mask skipping block 1 (2-bit) and block 2 (3-bit)
            mask_np = np.ones((3, max_blks), dtype=np.uint32)
            mask_np[0, 1] = 0
            mask_np[1, 2] = 0
            out_m = gqa_paged_nested_sdpa_fast(
                q, base_pool, sc_pool, p2_pool, p3_pool, rel_st, rel_p, qin, sw,
                bt_mx, bp2_mx, bp3_mx, sl_mx, qp_mx, ks_mx, 3, K, max(lens),
                active_block_mask=mx.array(mask_np),
            )
            mx.eval(out_m)
            ref_m = _fp32_paged_sdpa_ref(
                np.array(q), ref_kv_seqs, np.array(rel_st), np.array(rel_p),
                np.array(qin), np.array(sw), lens, q_pos0, [0, 0, 0], 3, K,
                block_mask=mask_np,
            )
            err_m = float(np.max(np.abs(np.array(out_m) - ref_m)))
            print(f"  [mixed masked SDPA B=3, K={K}] max_abs={err_m:.2e}")
            self.assertLess(err_m, 2e-3)

        # Prefill B=1, K=40, 64, 128 over mixed 5-block sequence [4, 2, 3, 2, 4]
        for K_pf in (40, 64, 128):
            ln_pf = 320 - (64 - K_pf) if K_pf <= 64 else 320
            seq2_blks = pfx_phys + tail_phys_per_seq[2][:2]  # 5 blocks
            q_pf = mx.array(rng.standard_normal((K_pf, 24, 128)).astype(np.float32))
            rel_pf = mx.array(rng.standard_normal((K_pf, 24, 16)).astype(np.float32) * 0.25)
            qin_pf = mx.array(rng.standard_normal((K_pf, 2304)).astype(np.float32) * 0.1)
            out_pf = gqa_paged_nested_sdpa_fast(
                q_pf, base_pool, sc_pool, p2_pool, p3_pool, rel_pf, rel_p, qin_pf, sw,
                mx.array(np.array([seq2_blks], dtype=np.uint32)), bp2_mx, bp3_mx,
                mx.array(np.array([ln_pf], dtype=np.uint32)),
                mx.array(np.array([ln_pf - K_pf], dtype=np.uint32)),
                mx.array(np.array([0], dtype=np.uint32)),
                1, K_pf, ln_pf,
            )
            mx.eval(out_pf)
            ref_pf = _fp32_paged_sdpa_ref(
                np.array(q_pf), [ref_kv_seqs[2][:, :, :ln_pf, :]], np.array(rel_pf), np.array(rel_p),
                np.array(qin_pf), np.array(sw), [ln_pf], [ln_pf - K_pf], [0], 1, K_pf,
            )
            err_pf = float(np.max(np.abs(np.array(out_pf) - ref_pf)))
            print(f"  [mixed FA2 prefill B=1, K={K_pf}] max_abs={err_pf:.2e}")
            self.assertLess(err_pf, 2e-3)

        # Single-split decode (B=1, K=1, max_span=192 < 256 -> split_out=False epilogue)
        q_sp1 = mx.array(rng.standard_normal((1, 24, 128)).astype(np.float32))
        rel_sp1 = mx.array(rng.standard_normal((1, 24, 16)).astype(np.float32) * 0.25)
        qin_sp1 = mx.array(rng.standard_normal((1, 2304)).astype(np.float32) * 0.1)
        out_sp1 = gqa_paged_nested_sdpa_fast(
            q_sp1, base_pool, sc_pool, p2_pool, p3_pool, rel_sp1, rel_p, qin_sp1, sw,
            mx.array(np.array([pfx_phys], dtype=np.uint32)), bp2_mx, bp3_mx,
            mx.array(np.array([192], dtype=np.uint32)),
            mx.array(np.array([191], dtype=np.uint32)),
            mx.array(np.array([0], dtype=np.uint32)),
            1, 1, 192,
        )
        mx.eval(out_sp1)
        ref_sp1 = _fp32_paged_sdpa_ref(
            np.array(q_sp1), [ref_kv_seqs[0][:, :, :192, :]], np.array(rel_sp1), np.array(rel_p),
            np.array(qin_sp1), np.array(sw), [192], [191], [0], 1, 1,
        )
        err_sp1 = float(np.max(np.abs(np.array(out_sp1) - ref_sp1)))
        print(f"  [mixed single-split decode B=1, K=1, len=192] max_abs={err_sp1:.2e}")
        self.assertLess(err_sp1, 2e-3)

        # Multi-qt batched verify (B=2, K=8 -> kt_cap=4, n_qt=2)
        q_k8 = mx.array(rng.standard_normal((16, 24, 128)).astype(np.float32))
        rel_k8 = mx.array(rng.standard_normal((16, 24, 16)).astype(np.float32) * 0.25)
        qin_k8 = mx.array(rng.standard_normal((16, 2304)).astype(np.float32) * 0.1)
        lens_2 = lens[:2]
        qp_2 = [ln - 8 for ln in lens_2]
        out_k8 = gqa_paged_nested_sdpa_fast(
            q_k8, base_pool, sc_pool, p2_pool, p3_pool, rel_k8, rel_p, qin_k8, sw,
            mx.array(bt_np[:2]), bp2_mx, bp3_mx,
            mx.array(np.array(lens_2, dtype=np.uint32)),
            mx.array(np.array(qp_2, dtype=np.uint32)),
            mx.zeros((2,), dtype=mx.uint32),
            2, 8, max(lens_2),
        )
        mx.eval(out_k8)
        ref_k8 = _fp32_paged_sdpa_ref(
            np.array(q_k8), ref_kv_seqs[:2], np.array(rel_k8), np.array(rel_p),
            np.array(qin_k8), np.array(sw), lens_2, qp_2, [0, 0], 2, 8,
        )
        err_k8 = float(np.max(np.abs(np.array(out_k8) - ref_k8)))
        print(f"  [mixed batched verify B=2, K=8] max_abs={err_k8:.2e}")
        self.assertLess(err_k8, 2e-3)

        # Edge case: all blocks 2-bit with empty p2/p3 pools
        empty_u32 = mx.zeros((0, 2, 8, 4), dtype=mx.uint32)
        all_neg = mx.full((n_phys,), -1, dtype=mx.int32)
        out_2b = gqa_paged_nested_sdpa_fast(
            q_pf[:1], base_pool, sc_pool, empty_u32, empty_u32, rel_pf[:1], rel_p, qin_pf[:1], sw,
            mx.array(np.array([seq2_blks], dtype=np.uint32)), all_neg, all_neg,
            mx.array(np.array([ln_pf], dtype=np.uint32)),
            mx.array(np.array([ln_pf - 1], dtype=np.uint32)),
            mx.array(np.array([0], dtype=np.uint32)),
            1, 1, ln_pf,
        )
        mx.eval(out_2b)
        self.assertTrue(bool(mx.all(mx.isfinite(out_2b))))

    def test_aging_memory_report_refcount_and_prefix_eviction(self) -> None:
        cfg = _make_nested_cfg(
            num_hidden_layers=2, loop_passes=1, loop_start=0, loop_end=0,
            local_layer_ids=(0,), sliding_window_size=1024,
        )
        pool = PagedInklingPool(
            cfg,
            init_global_blocks=8,
            init_sliding_blocks=8,
            max_global_blocks=32,
            max_sliding_blocks=32,
            max_conv_slots=16,
            enable_prefix_cache=True,
        )
        blks = [pool.alloc_global_block() for _ in range(6)]
        self.assertEqual(len(pool.p2_free), 2)
        self.assertEqual(len(pool.p3_free), 2)

        rep0 = pool.nested_memory_report()
        self.assertEqual(rep0["blocks_4bit"], 6)
        self.assertEqual(rep0["blocks_3bit"], 0)
        self.assertEqual(rep0["blocks_2bit"], 0)
        self.assertAlmostEqual(rep0["effective_bits_per_value"], 4.125, places=6)
        self.assertEqual(rep0["base_active_bytes"], 6 * 32768)
        self.assertEqual(rep0["scale_active_bytes"], 6 * 2048)
        self.assertEqual(rep0["p2_active_bytes"], 6 * 16384)
        self.assertEqual(rep0["p3_active_bytes"], 6 * 16384)

        # Age: sink(0) + hot(5) = 4b; warm(3,4) = 3b; cold(1,2) = 2b
        stats1 = pool.age_block_planes(blks, hot_blocks=1, warm_blocks=2, preserve_sink=True)
        self.assertEqual(stats1, {"dropped_p3": 4, "dropped_p2": 2, "skipped_shared": 0})
        self.assertEqual(len(pool.p3_free), 6)
        self.assertEqual(len(pool.p2_free), 4)
        rep1 = pool.nested_memory_report()
        self.assertEqual(rep1["blocks_4bit"], 2)
        self.assertEqual(rep1["blocks_3bit"], 2)
        self.assertEqual(rep1["blocks_2bit"], 2)
        self.assertAlmostEqual(rep1["effective_bits_per_value"], 3.125, places=6)
        self.assertEqual(rep1["p2_active_bytes"], 4 * 16384)
        self.assertEqual(rep1["p3_active_bytes"], 2 * 16384)

        # Age all to 3-bit, then all to 2-bit
        pool.age_block_planes(blks, hot_blocks=0, warm_blocks=6, preserve_sink=False)
        rep3 = pool.nested_memory_report()
        self.assertEqual(rep3["blocks_3bit"], 4)
        self.assertEqual(rep3["blocks_2bit"], 2)
        self.assertEqual(rep3["p3_active_bytes"], 0)
        self.assertEqual(len(pool.p3_free), pool.n_p3_blocks)

        pool.age_block_planes(blks, hot_blocks=0, warm_blocks=0, preserve_sink=False)
        rep2 = pool.nested_memory_report()
        self.assertEqual(rep2["blocks_2bit"], 6)
        self.assertEqual(rep2["p2_active_bytes"], 0)
        self.assertEqual(rep2["p3_active_bytes"], 0)
        self.assertAlmostEqual(rep2["effective_bits_per_value"], 2.125, places=6)
        self.assertEqual(len(pool.p2_free), pool.n_p2_blocks)

        for gb in blks:
            pool.free_global_block(gb)
        self.assertEqual(len(pool.g_free), pool.n_g_blocks)
        self.assertEqual(len(pool.p2_free), pool.n_p2_blocks)
        self.assertEqual(len(pool.p3_free), pool.n_p3_blocks)

        # Prefix sharing refcount protection + automatic aging hook + eviction cleanup
        pool_auto = PagedInklingPool(
            cfg,
            init_global_blocks=8,
            init_sliding_blocks=8,
            max_global_blocks=32,
            max_sliding_blocks=32,
            max_conv_slots=16,
            enable_prefix_cache=True,
            nested_hot_blocks=1,
            nested_warm_blocks=1,
            nested_spill_to_host=True,
        )
        pfx = list(range(1000, 1000 + 4 * PAGED_BLOCK_SIZE))  # 4 full blocks
        seq1 = BatchSequence(1, pfx + [9999], pool_auto)
        seq1.ensure_capacity_for(len(pfx))
        seq1.commit_tokens(pfx)
        # 4 blocks: sink(0)=4b, cold(1)=2b, warm(2)=3b, hot(3)=4b
        r_s1 = pool_auto.nested_memory_report()
        self.assertEqual((r_s1["blocks_4bit"], r_s1["blocks_3bit"], r_s1["blocks_2bit"]), (2, 1, 1))
        self.assertGreater(r_s1["host_spilled_bytes"], 0)

        # Second sequence shares prefix blocks -> g_ref == 2 protects them from aging!
        seq2 = BatchSequence(2, pfx + [8888], pool_auto)
        self.assertEqual(seq2.reused_tokens, len(pfx))
        self.assertEqual(pool_auto.g_ref[seq2.g_blocks[3]], 2)
        res_shared = pool_auto.age_block_planes(seq2.g_blocks, hot_blocks=0, warm_blocks=0, preserve_sink=False)
        self.assertEqual(res_shared["skipped_shared"], 4)
        self.assertEqual(res_shared["dropped_p3"], 0)
        self.assertEqual(res_shared["dropped_p2"], 0)

        seq1.release()
        seq2.release()
        # Cached prefix leaf remains until evicted; evicting all 4 leaves frees all blocks & host spills
        while pool_auto._evict_one_prefix_leaf():
            pass
        rep_clean = pool_auto.nested_memory_report()
        self.assertEqual(rep_clean["active_g_blocks"], 0)
        self.assertEqual(rep_clean["host_spilled_bytes"], 0)
        self.assertEqual(len(pool_auto.g_free), pool_auto.n_g_blocks)
        self.assertEqual(len(pool_auto.p2_free), pool_auto.n_p2_blocks)
        self.assertEqual(len(pool_auto.p3_free), pool_auto.n_p3_blocks)

    def test_coarse_2bit_k_scoring_and_topk_host_promotion(self) -> None:
        rng = np.random.default_rng(9102)
        cfg = _make_nested_cfg(
            num_hidden_layers=2, loop_passes=1, loop_start=0, loop_end=0,
            local_layer_ids=(0,), sliding_window_size=1024,
        )
        # Global layer is slot 1 (pidx = 0)
        pool = PagedInklingPool(
            cfg,
            init_global_blocks=8,
            init_sliding_blocks=8,
            max_global_blocks=32,
            max_sliding_blocks=32,
            max_conv_slots=8,
            enable_prefix_cache=False,
            nested_spill_to_host=True,
            nested_coarse_topk=1,
        )
        seq = BatchSequence(1, [1], pool, use_prefix_cache=False)

        # Build 4 full blocks (256 tokens) + 1 token step at position 256 (block 4)
        # Make block 2 (tokens 128..191) strongly aligned with the query `q_step` so 2-bit K picks it!
        q_step = rng.standard_normal((1, 24, 128)).astype(np.float32) * 0.4
        # Dominant shared direction along KV head 0..7
        q_dir = np.mean(q_step[0].reshape(8, 3, 128), axis=1)  # [8, 128]
        q_dir /= np.linalg.norm(q_dir, axis=1, keepdims=True) + 1e-8

        kv_hist = rng.standard_normal((256, 2, 8, 128)).astype(np.float16) * 0.25
        # Inject strong signal into token 150 (inside block 2!)
        kv_hist[150, 0] = (q_dir * 4.5).astype(np.float16)
        kv_hist[150, 1] = rng.standard_normal((8, 128)).astype(np.float16) * 1.5

        rel_p = mx.array(rng.standard_normal((1025, 16)).astype(np.float16) * 0.05)
        sw = mx.array(rng.standard_normal((24, 2304)).astype(np.float16) * 0.05)

        # Write 4 history blocks in 64-token prefill chunks (all start 4-bit)
        for ch in range(4):
            chunk_kv = mx.array(kv_hist[ch * 64 : (ch + 1) * 64])
            meta_c = _prepare_batch_meta(pool, [seq], 64)
            q_c = mx.array(rng.standard_normal((64, 24, 128)).astype(np.float32) * 0.2)
            rel_c = mx.array(rng.standard_normal((64, 24, 16)).astype(np.float32) * 0.1)
            qin_c = mx.array(rng.standard_normal((64, 2304)).astype(np.float32) * 0.1)
            out_c = pool.write_and_attend(1, q_c, chunk_kv, rel_c, rel_p, qin_c, sw, meta_c)
            mx.eval(out_c)
            seq.commit_tokens(list(range(ch * 64, (ch + 1) * 64)), register_prefix=False)

        # Age blocks [0, 1, 2, 3]: keep sink(0) at 4-bit, age cold blocks 1, 2, 3 to 2-bit with host spill!
        age_res = pool.age_block_planes(seq.g_blocks, hot_blocks=0, warm_blocks=0, preserve_sink=True)
        self.assertEqual(age_res["dropped_p3"], 3)
        self.assertEqual(age_res["dropped_p2"], 3)
        self.assertEqual(set(pool._host_p2.keys()), {seq.g_blocks[1], seq.g_blocks[2], seq.g_blocks[3]})

        # Next token (pos 256, block 4) with `q_step`
        kv_step = mx.array(rng.standard_normal((1, 2, 8, 128)).astype(np.float16) * 0.25)
        rel_step = mx.array(rng.standard_normal((1, 24, 16)).astype(np.float32) * 0.05)
        qin_step = mx.array(rng.standard_normal((1, 2304)).astype(np.float32) * 0.05)
        meta_s = _prepare_batch_meta(pool, [seq], 1)

        # 1. Verify gqa_paged_nested_coarse_score matches explicit FP32 2-bit K logit max
        q_mx = mx.array(q_step)
        scores_per_q = gqa_paged_nested_coarse_score(
            q_mx, pool.g_base_pools[0], pool.g_nested_scales[0], rel_step, rel_p,
            meta_s.g_bt, meta_s.g_lens, meta_s.q_pos0, meta_s.g_starts, 1, 1, reduce_q=False,
        )
        mx.eval(scores_per_q)
        scores_np = np.array(scores_per_q)[0, 0]  # [8, 5]
        # Explicit 2-bit K ref for blocks 0..3
        scale = 1.0 / math.sqrt(128.0)
        rel_st_np = np.array(rel_step)[0]
        rel_p_np = np.array(rel_p)
        for bi in range(4):
            gb = seq.g_blocks[bi]
            k2_dq = kv_h128_nested_dequantize(
                pool.g_base_pools[0][gb * 64 : (gb + 1) * 64, 0:1],
                pool.g_nested_scales[0][gb * 64 : (gb + 1) * 64, 0:1],
            )
            mx.eval(k2_dq)
            k2_np = np.array(k2_dq)[:, 0]  # [64, 8, 128]
            for kvh in range(8):
                best = -1e30
                for off in range(64):
                    j = bi * 64 + off
                    dist = 256 - j
                    for g in range(3):
                        h = kvh * 3 + g
                        s = float(np.dot(q_step[0, h], k2_np[off, kvh])) * scale
                        if dist < 1024:
                            s += float(np.dot(rel_st_np[h], rel_p_np[dist]))
                        best = max(best, s)
                self.assertAlmostEqual(float(scores_np[kvh, bi]), best, places=4)

        # Also test reduce_q=True for K > 4
        q_k8 = mx.array(rng.standard_normal((8, 24, 128)).astype(np.float32) * 0.2)
        rel_k8 = mx.array(rng.standard_normal((8, 24, 16)).astype(np.float32) * 0.1)
        cs_full = gqa_paged_nested_coarse_score(
            q_k8, pool.g_base_pools[0], pool.g_nested_scales[0], rel_k8, rel_p,
            meta_s.g_bt, meta_s.g_lens, meta_s.q_pos0, meta_s.g_starts, 1, 8, reduce_q=False,
        )
        cs_red = gqa_paged_nested_coarse_score(
            q_k8, pool.g_base_pools[0], pool.g_nested_scales[0], rel_k8, rel_p,
            meta_s.g_bt, meta_s.g_lens, meta_s.q_pos0, meta_s.g_starts, 1, 8, reduce_q=True,
        )
        mx.eval(cs_full, cs_red)
        np.testing.assert_allclose(np.array(cs_red), np.max(np.array(cs_full), axis=1), atol=1e-5)

        # 2. Run write_and_attend with topk=1 host promotion -> must promote cold block 2!
        p2_free_before = len(pool.p2_free)
        p3_free_before = len(pool.p3_free)
        out_prom = pool.write_and_attend(1, q_mx, kv_step, rel_step, rel_p, qin_step, sw, meta_s)
        mx.eval(out_prom)
        self.assertEqual(pool._last_nested_promotions.get(0), [seq.g_blocks[2]])
        # Scratch slots must be immediately returned!
        self.assertEqual(len(pool.p2_free), p2_free_before)
        self.assertEqual(len(pool.p3_free), p3_free_before)

        # 3. Verify promoted output matches explicit FP32 reference where block 2 is 4-bit and blocks 1,3 are 2-bit!
        full_kv = np.concatenate([kv_hist, np.array(kv_step)], axis=0)
        b_full, sc_full, p2_full, p3_full = kv_h128_nested4_quantize(mx.array(full_kv))
        mx.eval(b_full, sc_full, p2_full, p3_full)
        oracle_chunks = []
        for bi, bits in enumerate([4, 2, 4, 2, 4]):  # block 2 promoted to exact 4-bit!
            lo = bi * 64
            hi = min(257, (bi + 1) * 64)
            dq = kv_h128_nested_dequantize(
                b_full[lo:hi], sc_full[lo:hi],
                p2_full[lo:hi] if bits >= 3 else None,
                p3_full[lo:hi] if bits == 4 else None,
            )
            mx.eval(dq)
            oracle_chunks.append(np.array(dq))
        oracle_kv = np.concatenate(oracle_chunks, axis=0).transpose(1, 2, 0, 3)
        ref_prom = _fp32_paged_sdpa_ref(
            q_step, [oracle_kv], np.array(rel_step), np.array(rel_p),
            np.array(qin_step), np.array(sw), [257], [256], [0], 1, 1,
        )
        err_prom = float(np.max(np.abs(np.array(out_prom) - ref_prom)))
        print(f"  [coarse-2bit top-k promotion -> exact 4-bit SDPA] max_abs={err_prom:.2e}")
        self.assertLess(err_prom, 1e-4)

        # Also verify promotion strictly beat un-promoted 2-bit output on accuracy vs all-4-bit
        dq_all4 = kv_h128_nested_dequantize(b_full, sc_full, p2_full, p3_full)
        mx.eval(dq_all4)
        ref_all4 = _fp32_paged_sdpa_ref(
            q_step, [np.array(dq_all4).transpose(1, 2, 0, 3)], np.array(rel_step), np.array(rel_p),
            np.array(qin_step), np.array(sw), [257], [256], [0], 1, 1,
        )
        # Unpromoted (block 2 stays 2-bit)
        pool.nested_coarse_topk = 0
        out_unprom = pool.write_and_attend(1, q_mx, kv_step, rel_step, rel_p, qin_step, sw, meta_s)
        mx.eval(out_unprom)
        err_vs_all4_prom = float(np.max(np.abs(np.array(out_prom) - ref_all4)))
        err_vs_all4_unprom = float(np.max(np.abs(np.array(out_unprom) - ref_all4)))
        print(f"  [vs all-4-bit] promoted={err_vs_all4_prom:.2e} < unpromoted={err_vs_all4_unprom:.2e}")
        self.assertLess(err_vs_all4_prom, err_vs_all4_unprom)

        seq.release()

    def test_env_fallback_config_on_pool(self) -> None:
        cfg = _make_nested_cfg(num_hidden_layers=2, loop_passes=1, loop_start=0, loop_end=0, local_layer_ids=(0,))
        old = {k: os.environ.get(k) for k in ("LAMQ_NESTED_HOT_BLOCKS", "LAMQ_NESTED_WARM_BLOCKS", "LAMQ_NESTED_SPILL_HOST", "LAMQ_NESTED_COARSE_TOPK")}
        try:
            os.environ["LAMQ_NESTED_HOT_BLOCKS"] = "4"
            os.environ["LAMQ_NESTED_WARM_BLOCKS"] = "8"
            os.environ["LAMQ_NESTED_SPILL_HOST"] = "1"
            os.environ["LAMQ_NESTED_COARSE_TOPK"] = "3"
            pool = PagedInklingPool(cfg, init_global_blocks=4, init_sliding_blocks=4, max_global_blocks=8, max_sliding_blocks=8)
            self.assertEqual(pool.nested_hot_blocks, 4)
            self.assertEqual(pool.nested_warm_blocks, 8)
            self.assertTrue(pool.nested_spill_to_host)
            self.assertEqual(pool.nested_coarse_topk, 3)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_pool_dynamic_growth_and_shared_loop_kv(self) -> None:
        cfg = _make_nested_cfg(
            num_hidden_layers=4, loop_start=1, loop_end=3, loop_passes=3,
            local_layer_ids=(0, 1), shared_loop_kv=True,
        )
        pool = PagedInklingPool(
            cfg, init_global_blocks=2, init_sliding_blocks=2,
            max_global_blocks=16, max_sliding_blocks=16,
            enable_prefix_cache=False,
        )
        # Global layers: L3 (outside loop: 1 pool) + L2 (looped 3x with shared_loop_kv=True -> 1 canonical pool) = 2 global pools
        self.assertEqual(pool.n_global_pools, 2)
        blks = [pool.alloc_global_block() for _ in range(6)]
        self.assertGreaterEqual(pool.n_g_blocks, 6)
        self.assertGreaterEqual(pool.n_p2_blocks, 6)
        self.assertGreaterEqual(pool.n_p3_blocks, 6)
        rep = pool.nested_memory_report()
        self.assertEqual(rep["n_global_pools"], 2)
        self.assertEqual(rep["active_g_blocks"], 6)
        self.assertAlmostEqual(rep["effective_bits_per_value"], 4.125, places=6)
        for gb in blks:
            pool.free_global_block(gb)
        self.assertEqual(len(pool.g_free), pool.n_g_blocks)
        self.assertEqual(len(pool.p2_free), pool.n_p2_blocks)
        self.assertEqual(len(pool.p3_free), pool.n_p3_blocks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
