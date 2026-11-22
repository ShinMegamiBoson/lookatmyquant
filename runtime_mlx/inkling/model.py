"""Standalone Apple Silicon MLX implementation of bit-packed Inkling-2-Mini (Inkling2Loop).

Matches Hugging Face PR #51050 `modeling_inkling2_loop.py` line-for-line:
  - 24 physical layers with L6..17 looped 3x (48 logical visits), 48 KV + sconv cache slots
  - Block Attention Residuals (`attention_input`, `mlp_input`, `final_input`) & cross-pass `message_input`
  - 48 Loop Message Gates (`{q,k,v,mlp}_message_gate`) on passes 1 & 2
  - Static (`ShortConvolution`) & Dynamic (`DynamicShortConvolution`) causal 4-tap convolutions
  - Scaled/LogScaled L2Clip on Q/K/relpos, UnweightedRMSNorm on V, relative position bias + per-token sinks
  - Gated attention output (`sigmoid(gate_proj(h)) * attn_out -> linear`)
  - FP32 Sigmoid Top-8 router (`e_score_correction_bias`, `route_scale=8.0`) + SoftcappedSwiGLU (4.0, 25.0)
  - Full-speed Apple Silicon Metal kernels (`fast_kernels.py`) with L1 multi-tables, unified 1-launch MoE,
    fused 5-matrix attention stages, single-launch threadgroup RHTs, pre-allocated FP16 KV cache, and
    fused Flash-Decoding GQA+RelPos+Sinks SDPA.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .fast_kernels import (
    FMT_ID,
    DenseStage,
    UnifiedMoESide,
    block_attn_res_fused,
    block_attn_res_norm_rht_fast,
    build_fast_luts,
    compute_moe_sorted_offsets,
    embed_lookup_gpu,
    gqa_relpos_sinks_sdpa_fast,
    kv_dyn_sconv4_norm_fused,
    kv_sconv4_norm_fused,
    make_tree_spec,
    moe_wsum_sconv4_res_fused,
    msg_blend_norm_3x_fused,
    msg_blend_norm_fused,
    msg_blend_norm_rht_fast,
    q_rel_clip_fused,
    qkv_rel_prep_sliding_fused,
    rht_in_2304_fast,
    rht_in_3072_gated_fast,
    rht_out_1024_fast,
    rht_out_2304_fast,
    short_conv4_res_fused,
    sigmoid_top8_norm_and_plan_fast,
    sigmoid_top8_norm_fast,
    swiglu_rht_1152_fast,
)
from .kernels import (
    HASH_MULT_U32,
    lm_dense_matmul,
    nvfp4_matvec,
    rht_apply_in,
    rht_apply_out,
    trellis_dense_matmul,
    trellis_matvec,
    trellis_moe_grouped,
)

NVFP4_FID = 7
ID_TO_FMT = {0: "ht1", 1: "ht2", 2: "ht3", 3: "rtr1", 4: "rtr2", 5: "rtr3", 6: "rtr4"}
FMT_PARAMS = {
    "ht1": (1, 4, 512, HASH_MULT_U32),
    "ht2": (2, 4, 512, HASH_MULT_U32),
    "ht3": (3, 2, 1024, HASH_MULT_U32),
    "rtr1": (1, 1, 4096, 0),
    "rtr2": (2, 1, 4096, 0),
    "rtr3": (3, 1, 4096, 0),
    "rtr4": (4, 1, 4096, 0),
}


@dataclass
class Inkling2Config:
    hidden_size: int = 2304
    num_hidden_layers: int = 24
    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: int = 8192
    log_scaling_alpha: float = 0.1
    rms_norm_eps: float = 1e-6
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    local_layer_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22)
    sconv_kernel_size: int = 4
    conv_kernel_size: int = 4
    final_logit_softcapping: float = 30.0
    sliding_window_size: int = 1024
    n_layers_per_attn_res_block: int = 6
    n_routed_experts: int = 128
    num_experts_per_tok: int = 8
    intermediate_size: int = 1152
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    route_scale: float = 8.0
    loop_start: int = 6
    loop_end: int = 18
    loop_passes: int = 3
    loop_gate_rank: int = 64
    moe_k_max: int = 8
    moe_k_min: int = 8
    moe_ratio_thr: float = 0.0
    moe_renorm: bool = False

    @classmethod
    def from_json(cls, path: Path) -> "Inkling2Config":
        raw = json.loads(path.read_text())
        tc = raw.get("text_config", raw)
        lc = tc.get("loop_config", {})
        return cls(
            hidden_size=tc["hidden_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            vocab_size=tc["vocab_size"],
            unpadded_vocab_size=tc.get("unpadded_vocab_size", tc["vocab_size"]),
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc["head_dim"],
            d_rel=tc["d_rel"],
            rel_extent=tc["rel_extent"],
            log_scaling_n_floor=tc["log_scaling_n_floor"],
            log_scaling_alpha=tc["log_scaling_alpha"],
            rms_norm_eps=tc["rms_norm_eps"],
            qk_l2_clip_threshold=tc["qk_l2_clip_threshold"],
            rel_l2_clip_threshold=tc["rel_l2_clip_threshold"],
            local_layer_ids=tuple(tc["local_layer_ids"]),
            sconv_kernel_size=tc.get("sconv_kernel_size", 4),
            conv_kernel_size=tc.get("conv_kernel_size", tc.get("sconv_kernel_size", 4)),
            final_logit_softcapping=tc["final_logit_softcapping"],
            sliding_window_size=tc["sliding_window_size"],
            n_layers_per_attn_res_block=tc["n_layers_per_attn_res_block"],
            n_routed_experts=tc["n_routed_experts"],
            num_experts_per_tok=tc["num_experts_per_tok"],
            intermediate_size=tc["intermediate_size"],
            mlp_gate_softcap=tc["mlp_gate_softcap"],
            mlp_up_softcap=tc["mlp_up_softcap"],
            route_scale=tc["route_scale"],
            loop_start=lc.get("core_start", 6),
            loop_end=lc.get("core_end", 18),
            loop_passes=lc.get("num_passes", 3),
            loop_gate_rank=lc.get("gate_rank", 64),
        )

    @property
    def total_visits(self) -> int:
        return self.num_hidden_layers + (self.loop_passes - 1) * (self.loop_end - self.loop_start)

    @property
    def use_fast_metal(self) -> bool:
        return (
            self.hidden_size == 2304
            and self.intermediate_size == 1152
            and self.num_attention_heads == 24
            and self.num_key_value_heads == 8
            and self.head_dim == 128
            and self.n_routed_experts == 128
            and self.num_experts_per_tok == 8
            and self.sconv_kernel_size == 4
            and self.d_rel == 16
            and self.rel_extent == 1024
            and self.sliding_window_size == 1024
        )


# ----------------------------------------------------------------------------- Primitives (exact FP32 matching HF)

@lru_cache(maxsize=64)
def _f32(val: float) -> mx.array:
    return mx.array(val, dtype=mx.float32)


def unweighted_rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x.astype(mx.float32), None, eps)


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x.astype(mx.float32), weight.astype(mx.float32), eps)


@mx.compile
def _l2_clip_compiled(x: mx.array, threshold: mx.array) -> mx.array:
    xf = x.astype(mx.float32)
    norm = mx.sqrt(mx.sum(xf * xf, axis=-1, keepdims=True))
    return xf * mx.minimum(threshold / (norm + 1e-6), 1.0)


def l2_clip(x: mx.array, threshold: float) -> mx.array:
    return _l2_clip_compiled(x, _f32(threshold))


def log_scale(x: mx.array, position_ids: mx.array, alpha: float, n_floor: int) -> mx.array:
    tau = 1.0 + alpha * mx.log(mx.maximum((position_ids.astype(mx.float32) + 1.0) / float(n_floor), 1.0))
    return x.astype(mx.float32) * tau[..., None, None]


@mx.compile
def _softcap_compiled(x: mx.array, cap: mx.array) -> mx.array:
    return mx.tanh(x / cap) * cap


def softcap(x: mx.array, cap: float | None) -> mx.array:
    if cap is None or cap <= 0:
        return x
    return _softcap_compiled(x, _f32(float(cap)))


def logits_to_probs(logits: mx.array, *, temp: float, top_p: float = 1.0, top_k: int = 0) -> mx.array:
    """Convert softcapped logits `[..., V]` -> exact normalized `float32[..., V]` probabilities on GPU."""
    squeeze = logits.ndim == 1
    x = logits[None, :] if squeeze else logits
    V = int(x.shape[-1])
    scaled = x.astype(mx.float32) * (1.0 / float(temp))

    if 0 < top_k < V:
        k_idx = mx.argpartition(-scaled, top_k - 1, axis=-1)[:, :top_k]
        k_vals = mx.take_along_axis(scaled, k_idx, axis=-1)
        if 0.0 < top_p < 1.0:
            ord_k = mx.argsort(-k_vals, axis=-1)
            sv = mx.take_along_axis(k_vals, ord_k, axis=-1)
            si = mx.take_along_axis(k_idx, ord_k, axis=-1)
            sp = mx.softmax(sv, axis=-1)
            sp = mx.where((mx.cumsum(sp, axis=-1) - sp) < top_p, sp, 0.0)
            sp = sp / mx.maximum(mx.sum(sp, axis=-1, keepdims=True), 1e-20)
            probs = mx.put_along_axis(mx.zeros_like(scaled), si, sp, axis=-1)
        else:
            probs = mx.put_along_axis(mx.zeros_like(scaled), k_idx, mx.softmax(k_vals, axis=-1), axis=-1)
        return probs[0] if squeeze else probs

    probs = mx.softmax(scaled, axis=-1)
    if 0.0 < top_p < 1.0:
        sorted_idx = mx.argsort(-probs, axis=-1)
        sorted_p = mx.take_along_axis(probs, sorted_idx, axis=-1)
        kept_p = mx.where((mx.cumsum(sorted_p, axis=-1) - sorted_p) < top_p, sorted_p, 0.0)
        kept_p = kept_p / mx.maximum(mx.sum(kept_p, axis=-1, keepdims=True), 1e-20)
        probs = mx.put_along_axis(mx.zeros_like(probs), sorted_idx, kept_p, axis=-1)
    return probs[0] if squeeze else probs


@mx.compile
def _softcapped_swiglu_compiled(gate_up: mx.array, gate_cap: mx.array, up_cap: mx.array) -> mx.array:
    g, u = mx.split(gate_up.astype(mx.float32), 2, axis=-1)
    return (mx.tanh(g / gate_cap) * gate_cap) * mx.sigmoid(g) * (mx.tanh(u / up_cap) * up_cap)


def softcapped_swiglu(gate_up: mx.array, gate_cap: float = 4.0, up_cap: float = 25.0) -> mx.array:
    return _softcapped_swiglu_compiled(gate_up, _f32(gate_cap), _f32(up_cap))


@mx.compile
def _route_top8_compiled(flat: mx.array, router_w: mx.array, router_bias: mx.array, route_scale: mx.array):
    logits = flat @ router_w.T
    scores = mx.sigmoid(logits)
    biased = scores + router_bias[None, :]
    idx_top = mx.argpartition(-biased, 7, axis=-1)[:, :8]
    w_raw = mx.take_along_axis(scores, idx_top, axis=-1)
    w_norm = (w_raw / (mx.sum(w_raw, axis=-1, keepdims=True) + 1e-20)) * route_scale
    return idx_top.reshape(-1), w_norm.reshape(-1)


# ----------------------------------------------------------------------------- 48-Visit KV + Conv Cache

class Inkling2Cache:
    """KV + 4-slot causal conv state cache across all 48 logical layer visits, with O(1) speculative rollback.

    Uses pre-allocated `float16[n_kv, cap, head_dim]` buffers on the fast Inkling-2-Mini path (zero per-step
    `mx.concatenate` copies) and falls back cleanly to dynamic `float32` buffers on miniature test configs.
    """

    def __init__(self, cfg: Inkling2Config):
        self.cfg = cfg
        n_slots = cfg.total_visits
        self._fast = cfg.use_fast_metal
        # Generic fallback KV
        self.keys: list[mx.array | None] = [None] * n_slots
        self.values: list[mx.array | None] = [None] * n_slots
        # Fast pre-allocated unified FP16 KV (`[2, 8, cap, 128]` -> 1 slice update per visit!)
        self.kv_f16: list[mx.array | None] = [None] * n_slots
        self.kv_len: list[int] = [0] * n_slots
        self.kv_start: list[int] = [0] * n_slots
        self.kv_cap: list[int] = [0] * n_slots
        self.kv_is_sliding: list[bool] = [False] * n_slots

        self.conv_states: list[list[mx.array | None]] = [[None, None, None, None] for _ in range(n_slots)]
        self.seq_len: int = 0
        self._spec_recording: bool = False
        self._spec_base_seq: int = 0
        self._spec_base_kv_len: list[int] = [0] * n_slots
        self._spec_kv_full: list[tuple[mx.array, mx.array, bool] | None] = [None] * n_slots
        self._spec_conv_full: list[list[mx.array | None]] = [[None, None, None, None] for _ in range(n_slots)]
        self._spec_conv_pair: list[list[tuple[mx.array, mx.array] | None]] = [[None, None, None, None] for _ in range(n_slots)]

    def clone(self) -> "Inkling2Cache":
        """Independent, evaluated copy of this cache (KV, conv state and lengths), e.g. to snapshot a prefilled prompt.

        Must not be called mid speculative verify; the copy starts with no verify in progress.
        """
        if self._spec_recording:
            raise RuntimeError("cannot clone an Inkling2Cache during speculative verification")
        new = object.__new__(Inkling2Cache)
        new.__dict__.update(self.__dict__)
        n_slots = len(self.keys)

        def cp(a):
            return None if a is None else mx.array(a)

        new.keys = [cp(a) for a in self.keys]
        new.values = [cp(a) for a in self.values]
        new.kv_f16 = [cp(a) for a in self.kv_f16]
        new.kv_len, new.kv_start, new.kv_cap = list(self.kv_len), list(self.kv_start), list(self.kv_cap)
        new.kv_is_sliding = list(self.kv_is_sliding)
        new.conv_states = [[cp(a) for a in row] for row in self.conv_states]
        new._spec_base_kv_len = list(self._spec_base_kv_len)
        new._spec_kv_full = [None] * n_slots
        new._spec_conv_full = [[None, None, None, None] for _ in range(n_slots)]
        new._spec_conv_pair = [[None, None, None, None] for _ in range(n_slots)]
        mx.eval(*new.state)
        return new

    def begin_spec_verify(self):
        self._spec_recording = True
        self._spec_base_seq = self.seq_len
        if self._fast:
            self._spec_base_kv_len = list(self.kv_len)

    def _compact_sliding_slot(self, slot: int, new_len: int):
        W = self.cfg.sliding_window_size
        if self.kv_is_sliding[slot] and new_len > W + 256 and self.kv_f16[slot] is not None:
            shift = new_len - W
            KV = self.kv_f16[slot]
            KV_new = mx.zeros((2, 8, self.kv_cap[slot], 128), dtype=mx.float16)
            KV_new[:, :, :W, :] = KV[:, :, shift:new_len, :]
            self.kv_f16[slot] = KV_new
            self.kv_start[slot] += shift
            self.kv_len[slot] = W

    def commit_spec_verify(self, k_accept: int, block_len: int):
        """Roll back KV & conv states after verifying `block_len` tokens and accepting the first `k_accept` (`1..block_len`)."""
        self._spec_recording = False
        drop = block_len - k_accept
        self.seq_len = self._spec_base_seq + k_accept
        W = self.cfg.sliding_window_size
        for slot in range(len(self.keys)):
            if self._fast:
                new_len = self._spec_base_kv_len[slot] + k_accept
                self.kv_len[slot] = new_len
                self._compact_sliding_slot(slot, new_len)
            else:
                kv = self._spec_kv_full[slot]
                if kv is not None:
                    K, V, is_sliding = kv
                    if drop > 0:
                        K = K[:, :, :-drop, :]
                        V = V[:, :, :-drop, :]
                    if is_sliding and K.shape[2] > W:
                        K = K[:, :, -W:, :]
                        V = V[:, :, -W:, :]
                    self.keys[slot] = K
                    self.values[slot] = V
                    self._spec_kv_full[slot] = None

            for ci in range(4):
                cp = self._spec_conv_pair[slot][ci]
                if cp is not None:
                    prev, x_btc = cp
                    if drop > 0:
                        if k_accept >= 3:
                            self.conv_states[slot][ci] = x_btc[:, k_accept - 3 : k_accept, :]
                        else:
                            self.conv_states[slot][ci] = mx.concatenate([prev[:, k_accept:, :], x_btc[:, :k_accept, :]], axis=1)
                    self._spec_conv_pair[slot][ci] = None
                cf = self._spec_conv_full[slot][ci]
                if cf is not None:
                    need = cf.shape[1] - block_len
                    self.conv_states[slot][ci] = cf[:, k_accept : k_accept + need, :]
                    self._spec_conv_full[slot][ci] = None

    def commit_tree_verify(self, accepted_path: list[int], block_len: int):
        """Commit an arbitrary causal tree path `accepted_path` (starting at node 0) out of `block_len` verified nodes."""
        k_accept = len(accepted_path)
        if accepted_path == list(range(k_accept)):
            self.commit_spec_verify(k_accept, block_len)
            return
        self._spec_recording = False
        self.seq_len = self._spec_base_seq + k_accept
        idx_mx = mx.array(np.array(accepted_path, dtype=np.int32))
        idx_last3 = mx.array(np.array(accepted_path[-3:], dtype=np.int32))
        W = self.cfg.sliding_window_size
        for slot in range(len(self.keys)):
            if self._fast:
                base = self._spec_base_kv_len[slot]
                KV = self.kv_f16[slot]
                if KV is not None:
                    gathered = KV[:, :, base : base + block_len, :][:, :, idx_mx, :]
                    KV[:, :, base : base + k_accept, :] = gathered
                new_len = base + k_accept
                self.kv_len[slot] = new_len
                self._compact_sliding_slot(slot, new_len)
            else:
                kv = self._spec_kv_full[slot]
                if kv is not None:
                    K, V, is_sliding = kv
                    base_k = K.shape[2] - block_len
                    K = mx.concatenate([K[:, :, :base_k, :], K[:, :, base_k:, :][:, :, idx_mx, :]], axis=2)
                    V = mx.concatenate([V[:, :, :base_k, :], V[:, :, base_k:, :][:, :, idx_mx, :]], axis=2)
                    if is_sliding and K.shape[2] > W:
                        K = K[:, :, -W:, :]
                        V = V[:, :, -W:, :]
                    self.keys[slot] = K
                    self.values[slot] = V
                    self._spec_kv_full[slot] = None

            for ci in range(4):
                cp = self._spec_conv_pair[slot][ci]
                if cp is not None:
                    prev, x_btc = cp
                    if k_accept >= 3:
                        self.conv_states[slot][ci] = x_btc[:, idx_last3, :]
                    else:
                        self.conv_states[slot][ci] = mx.concatenate([prev[:, k_accept:, :], x_btc[:, idx_mx, :]], axis=1)
                    self._spec_conv_pair[slot][ci] = None
                cf = self._spec_conv_full[slot][ci]
                if cf is not None:
                    need = cf.shape[1] - block_len
                    prev = cf[:, :need, :]
                    x_btc = cf[:, need:, :]
                    if k_accept >= need:
                        self.conv_states[slot][ci] = x_btc[:, mx.array(np.array(accepted_path[-need:], dtype=np.int32)), :]
                    else:
                        self.conv_states[slot][ci] = mx.concatenate([prev[:, k_accept:, :], x_btc[:, idx_mx, :]], axis=1)
                    self._spec_conv_full[slot][ci] = None

    def get_conv_prev(self, slot: int, conv_idx: int, C: int) -> mx.array:
        prev = self.conv_states[slot][conv_idx]
        if prev is None:
            prev = mx.zeros((1, 3, C), dtype=mx.float32)
        return prev

    def set_conv_after(self, slot: int, conv_idx: int, prev: mx.array, x_btc: mx.array, new_prev: mx.array):
        if self._spec_recording:
            self._spec_conv_pair[slot][conv_idx] = (prev, x_btc)
        self.conv_states[slot][conv_idx] = new_prev

    def update_kv_f16(self, slot: int, kv_f16: mx.array, is_sliding: bool) -> tuple[mx.array, int, int, int]:
        """Pre-allocated unified FP16 KV append. `kv_f16`: `[2, 8, T, 128]` float16. Returns `(KV_buf, kv_len, kv_start, kv_cap)`."""
        T = int(kv_f16.shape[2])
        cur = self.kv_len[slot]
        self.kv_is_sliding[slot] = is_sliding
        W = self.cfg.sliding_window_size
        if is_sliding and not self._spec_recording and cur + T > self.kv_cap[slot] and cur > W:
            shift = cur - W
            KV_old = self.kv_f16[slot]
            new_cap = max(self.kv_cap[slot], ((W + T + 255) // 256) * 256)
            KV_new = mx.zeros((2, 8, new_cap, 128), dtype=mx.float16)
            KV_new[:, :, :W, :] = KV_old[:, :, shift:cur, :]
            self.kv_f16[slot] = KV_new
            self.kv_cap[slot] = new_cap
            self.kv_start[slot] += shift
            cur = W
        need = cur + T
        if self.kv_f16[slot] is None or need > self.kv_cap[slot]:
            new_cap = ((need + 255) // 256) * 256
            KV_new = mx.zeros((2, 8, new_cap, 128), dtype=mx.float16)
            if self.kv_f16[slot] is not None and cur > 0:
                KV_new[:, :, :cur, :] = self.kv_f16[slot][:, :, :cur, :]
            self.kv_f16[slot] = KV_new
            self.kv_cap[slot] = new_cap
        KV_buf = self.kv_f16[slot]
        KV_buf[:, :, cur:need, :] = kv_f16
        self.kv_len[slot] = need
        return KV_buf, need, self.kv_start[slot], self.kv_cap[slot]

    def update_kv(self, slot: int, k: mx.array, v: mx.array, is_sliding: bool) -> tuple[mx.array, mx.array]:
        if self.keys[slot] is None:
            K, V = k, v
        else:
            K = mx.concatenate([self.keys[slot], k], axis=2)
            V = mx.concatenate([self.values[slot], v], axis=2)
        if self._spec_recording:
            self._spec_kv_full[slot] = (K, V, is_sliding)
        if is_sliding and K.shape[2] > self.cfg.sliding_window_size:
            self.keys[slot] = K[:, :, -self.cfg.sliding_window_size :, :]
            self.values[slot] = V[:, :, -self.cfg.sliding_window_size :, :]
        else:
            self.keys[slot] = K
            self.values[slot] = V
        return K, V

    def update_conv(self, slot: int, conv_idx: int, x_btc: mx.array, kernel_size: int) -> mx.array:
        B, T, C = x_btc.shape
        need = kernel_size - 1
        prev = self.conv_states[slot][conv_idx]
        if prev is None:
            prev = mx.zeros((B, need, C), dtype=mx.float32)
        full = mx.concatenate([prev, x_btc], axis=1)
        if self._spec_recording:
            self._spec_conv_full[slot][conv_idx] = full
        self.conv_states[slot][conv_idx] = full[:, -need:, :]
        return full

    @property
    def offset(self) -> int:
        return self.seq_len

    @property
    def state(self) -> list[mx.array]:
        out: list[mx.array] = []
        if self._fast:
            for kv in self.kv_f16:
                if kv is not None:
                    out.append(kv)
        else:
            for k, v in zip(self.keys, self.values):
                if k is not None:
                    out.append(k)
                if v is not None:
                    out.append(v)
        for slot_c in self.conv_states:
            for c in slot_c:
                if c is not None:
                    out.append(c)
        return out


# ----------------------------------------------------------------------------- Packed Linear & MoE Modules

class PackedLinear:
    """Executes one bit-packed attention linear (`ht*`, `rtr*`, `lm6`, or `lm8`) directly from packed bytes."""

    def __init__(self, meta: dict[str, Any], arrays: dict[str, Any], luts: dict[str, mx.array]):
        self.fmt = meta["fmt"]
        self.m = int(meta["m"])
        self.n = int(meta["n"])
        self.k = int(meta["k"])
        safe = meta["safe"]
        self.codes = mx.array(arrays[f"lin_{safe}_codes"])
        self.scales = mx.array(arrays[f"lin_{safe}_scales"])  # float16
        self.signs_in = mx.array(arrays[f"lin_{safe}_signs_in"].astype(np.float32))
        self.out_rot = bool(meta.get("out_rot", False))
        if self.out_rot:
            self.signs_out = mx.array(arrays[f"lin_{safe}_signs_out"].astype(np.float32))
        else:
            self.signs_out = None
        if self.fmt.startswith("lm"):
            self.levels = mx.array(arrays[f"lin_{safe}_levels"].astype(np.float32))
            self.kind = "lm"
        else:
            self.kind = "trellis"
            self.g = int(meta["g"])
            self.K = int(meta["K"])
            self.hash_mult = int(meta["hash_mult"])
            self.lut = luts[self.fmt]

    def __call__(self, x: mx.array) -> mx.array:
        orig = x.shape
        T = int(np.prod(orig[:-1]))
        x_rot = rht_apply_in(x.reshape(T, self.n), self.signs_in)
        if self.kind == "lm":
            y = lm_dense_matmul(x_rot, self.codes, self.scales, self.levels, self.m, self.n, self.k)
        else:
            y = trellis_dense_matmul(
                x_rot, self.codes, self.scales, self.lut,
                self.m, self.n, self.k, self.g, self.K, self.hash_mult,
            )
        if self.out_rot:
            y = rht_apply_out(y, self.signs_out)
        return y.reshape(*orig[:-1], self.m)


class PackedMoE:
    """Executes all 128 MoE experts in ONE launch per side (`gu`, `dn`) via `UnifiedMoESide` on Inkling-2-Mini."""

    def __init__(
        self,
        arrays: dict[str, Any],
        luts: dict[str, mx.array],
        fast_luts: dict[str, mx.array] | None,
        router_w: mx.array,
        router_bias: mx.array,
        cfg: Inkling2Config,
    ):
        self.cfg = cfg
        self._fast = cfg.use_fast_metal and fast_luts is not None
        self.router_w = router_w.astype(mx.float32)
        self.router_w_f16 = router_w.astype(mx.float16) if self._fast else None
        self.router_bias = router_bias.astype(mx.float32)
        self.gu_signs_in = mx.array(arrays["gu_signs_in"].astype(np.float32))
        self.dn_signs_in = mx.array(arrays["dn_signs_in"].astype(np.float32))
        gu_map = np.asarray(arrays["gu_map"], dtype=np.int32)
        dn_map = np.asarray(arrays["dn_map"], dtype=np.int32)
        gu_map_fid = mx.array(gu_map[:, 0].astype(np.int32))
        gu_map_loc = mx.array(gu_map[:, 1].astype(np.uint32))
        dn_map_fid = mx.array(dn_map[:, 0].astype(np.int32))
        dn_map_loc = mx.array(dn_map[:, 1].astype(np.uint32))

        gu_slabs = []
        dn_slabs = []
        for fid, fmt in ID_TO_FMT.items():
            gk = f"gu_{fmt}_codes"
            if gk in arrays:
                gu_slabs.append((fid, mx.array(arrays[gk]), mx.array(arrays[f"gu_{fmt}_scales"]), luts[fmt], FMT_PARAMS[fmt]))
            dk = f"dn_{fmt}_codes"
            if dk in arrays:
                dn_slabs.append((fid, mx.array(arrays[dk]), mx.array(arrays[f"dn_{fmt}_scales"]), luts[fmt], FMT_PARAMS[fmt]))

        gu_nvfp4 = None
        dn_nvfp4 = None
        if "gu_nvfp4_codes" in arrays:
            c_u32 = np.ascontiguousarray(arrays["gu_nvfp4_codes"], dtype=np.uint8).view(np.uint32)
            gu_nvfp4 = (
                mx.array(c_u32),
                mx.array(np.ascontiguousarray(arrays["gu_nvfp4_scales"], dtype=np.uint8)),
                mx.array(np.ascontiguousarray(arrays["gu_nvfp4_outer"], dtype=np.float32)),
            )
        if "dn_nvfp4_codes" in arrays:
            c_u32 = np.ascontiguousarray(arrays["dn_nvfp4_codes"], dtype=np.uint8).view(np.uint32)
            dn_nvfp4 = (
                mx.array(c_u32),
                mx.array(np.ascontiguousarray(arrays["dn_nvfp4_scales"], dtype=np.uint8)),
                mx.array(np.ascontiguousarray(arrays["dn_nvfp4_outer"], dtype=np.float32)),
            )

        if self._fast:
            self.gu_unified = UnifiedMoESide(gu_slabs, gu_map_fid, gu_map_loc, 2 * cfg.intermediate_size, cfg.hidden_size, fast_luts, nvfp4_slab=gu_nvfp4)
            self.dn_unified = UnifiedMoESide(dn_slabs, dn_map_fid, dn_map_loc, cfg.hidden_size, cfg.intermediate_size, fast_luts, nvfp4_slab=dn_nvfp4)
            self.gu_slabs = []
            self.dn_slabs = []
            self.gu_nvfp4 = None
            self.dn_nvfp4 = None
            self.e2m1_lut = None
            self.e4m3_lut = None
        else:
            self.gu_map_fid, self.gu_map_loc = gu_map_fid, gu_map_loc
            self.dn_map_fid, self.dn_map_loc = dn_map_fid, dn_map_loc
            self._sentinel = mx.array(0xFFFFFFFF, dtype=mx.uint32)
            self.gu_slabs = gu_slabs
            self.dn_slabs = dn_slabs
            self.gu_nvfp4 = gu_nvfp4
            self.dn_nvfp4 = dn_nvfp4
            self.e2m1_lut = luts.get("nvfp4_e2m1")
            self.e4m3_lut = luts.get("nvfp4_e4m3")

    def forward_fast_fused_prerot(
        self,
        h_mlp_f16: mx.array,
        x_rot_gu_f16: mx.array,
        residual: mx.array,
        post_mlp_conv_w: mx.array,
        cache: Inkling2Cache,
        cache_slot: int,
        T: int,
        tree_lag: mx.array | None = None,
    ) -> mx.array:
        """Full fast MoE taking pre-rotated `x_rot_gu_f16` + `h_mlp_f16` from fused BAR/MsgBlend+Norm+RHT."""
        logits_f16 = h_mlp_f16 @ self.router_w_f16.T
        dyn = (self.cfg.moe_k_max < 8) or (self.cfg.moe_ratio_thr > 0.0)
        if 2 <= T <= 8:
            if dyn:
                eids_flat, w_norm, gu_plan, dn_plan = sigmoid_top8_norm_and_plan_fast(
                    logits_f16, self.router_bias, self.cfg.route_scale,
                    self.gu_unified.exp_fid, self.gu_unified.fids,
                    self.dn_unified.exp_fid, self.dn_unified.fids,
                    k_max=self.cfg.moe_k_max, k_min=self.cfg.moe_k_min,
                    ratio_thr=self.cfg.moe_ratio_thr, renorm=self.cfg.moe_renorm,
                )
            else:
                eids_flat, w_norm, gu_plan, dn_plan = sigmoid_top8_norm_and_plan_fast(
                    logits_f16, self.router_bias, self.cfg.route_scale,
                    self.gu_unified.exp_fid, self.gu_unified.fids,
                    self.dn_unified.exp_fid, self.dn_unified.fids,
                )
            sc = None
        else:
            if dyn:
                eids_flat, w_norm = sigmoid_top8_norm_fast(
                    logits_f16, self.router_bias, self.cfg.route_scale,
                    k_max=self.cfg.moe_k_max, k_min=self.cfg.moe_k_min,
                    ratio_thr=self.cfg.moe_ratio_thr, renorm=self.cfg.moe_renorm,
                )
            else:
                eids_flat, w_norm = sigmoid_top8_norm_fast(logits_f16, self.router_bias, self.cfg.route_scale)
            gu_plan = dn_plan = None
            sc = compute_moe_sorted_offsets(eids_flat) if T > 8 else None
        need_dn_raw = self.dn_unified.has_nvfp4
        if dyn:
            y_gu = self.gu_unified(x_rot_gu_f16, eids_flat, x_div=8, sorted_cache=sc, plan=gu_plan, dynamic_routing=True, x_raw_f16=h_mlp_f16)
            if need_dn_raw:
                act_rot, act_raw = swiglu_rht_1152_fast(y_gu, self.dn_signs_in, self.cfg.mlp_gate_softcap, self.cfg.mlp_up_softcap, out_dtype=mx.float16, eids_flat=eids_flat, also_raw=True)
            else:
                act_rot = swiglu_rht_1152_fast(y_gu, self.dn_signs_in, self.cfg.mlp_gate_softcap, self.cfg.mlp_up_softcap, out_dtype=mx.float16, eids_flat=eids_flat)
                act_raw = None
            y_dn = self.dn_unified(act_rot, eids_flat, x_div=1, sorted_cache=sc, plan=dn_plan, dynamic_routing=True, x_raw_f16=act_raw)
        else:
            y_gu = self.gu_unified(x_rot_gu_f16, eids_flat, x_div=8, sorted_cache=sc, plan=gu_plan, x_raw_f16=h_mlp_f16)
            if need_dn_raw:
                act_rot, act_raw = swiglu_rht_1152_fast(y_gu, self.dn_signs_in, self.cfg.mlp_gate_softcap, self.cfg.mlp_up_softcap, out_dtype=mx.float16, also_raw=True)
            else:
                act_rot = swiglu_rht_1152_fast(y_gu, self.dn_signs_in, self.cfg.mlp_gate_softcap, self.cfg.mlp_up_softcap, out_dtype=mx.float16)
                act_raw = None
            y_dn = self.dn_unified(act_rot, eids_flat, x_div=1, sorted_cache=sc, plan=dn_plan, x_raw_f16=act_raw)

        prev = cache.get_conv_prev(cache_slot, 3, 2304)
        out, new_prev, raw_sum = moe_wsum_sconv4_res_fused(y_dn, w_norm, prev, post_mlp_conv_w, residual, T, tree_lag=tree_lag, skip_zero_w=dyn)
        cache.set_conv_after(cache_slot, 3, prev, raw_sum, new_prev)
        return out

    def forward_fast_fused(
        self,
        h_mlp: mx.array,
        residual: mx.array,
        post_mlp_conv_w: mx.array,
        cache: Inkling2Cache,
        cache_slot: int,
        T: int,
    ) -> mx.array:
        """Full fast MoE + fused weighted-sum + post-MLP short_conv + residual add."""
        flat = h_mlp.reshape(T, 2304)
        x_rot = rht_in_2304_fast(flat, self.gu_signs_in, n_signs=1, T=T, out_dtype=mx.float16)
        return self.forward_fast_fused_prerot(flat.astype(mx.float16), x_rot, residual, post_mlp_conv_w, cache, cache_slot, T)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B, T, H = hidden_states.shape
        N = B * T
        K_top = self.cfg.num_experts_per_tok
        flat = hidden_states.reshape(N, H).astype(mx.float32)

        if self._fast:
            flat_f16 = flat.astype(mx.float16)
            logits_f16 = flat_f16 @ self.router_w_f16.T
            eids_flat, w_norm = sigmoid_top8_norm_fast(logits_f16, self.router_bias, self.cfg.route_scale)
            x_rot = rht_in_2304_fast(flat, self.gu_signs_in, n_signs=1, T=N, out_dtype=mx.float16)
            sc = compute_moe_sorted_offsets(eids_flat) if N > 8 else None
            y_gu = self.gu_unified(x_rot, eids_flat, x_div=8, sorted_cache=sc, x_raw_f16=flat_f16)
            if self.dn_unified.has_nvfp4:
                act_rot, act_raw = swiglu_rht_1152_fast(y_gu, self.dn_signs_in, self.cfg.mlp_gate_softcap, self.cfg.mlp_up_softcap, out_dtype=mx.float16, also_raw=True)
            else:
                act_rot = swiglu_rht_1152_fast(y_gu, self.dn_signs_in, self.cfg.mlp_gate_softcap, self.cfg.mlp_up_softcap, out_dtype=mx.float16)
                act_raw = None
            y_dn = self.dn_unified(act_rot, eids_flat, x_div=1, sorted_cache=sc, x_raw_f16=act_raw)
            out = mx.sum(y_dn.reshape(N, 8, 2304) * w_norm.reshape(N, 8, 1), axis=1)
            return out.reshape(B, T, H)

        logits = flat @ self.router_w.T
        scores = mx.sigmoid(logits)
        biased = scores + self.router_bias[None, :]
        idx_top = mx.argpartition(-biased, K_top - 1, axis=-1)[:, :K_top]
        w_raw = mx.take_along_axis(scores, idx_top, axis=-1)
        w_norm = (w_raw / (mx.sum(w_raw, axis=-1, keepdims=True) + 1e-20)) * self.cfg.route_scale

        eids_flat = idx_top.reshape(-1)
        gu_fids = self.gu_map_fid[eids_flat]
        gu_locs = self.gu_map_loc[eids_flat]
        dn_fids = self.dn_map_fid[eids_flat]
        dn_locs = self.dn_map_loc[eids_flat]

        use_grouped = 2 <= N <= 8
        moe_fn = trellis_moe_grouped if use_grouped else trellis_matvec
        single_gu = (len(self.gu_slabs) == 1) and (self.gu_nvfp4 is None)
        y_gu_total = None
        if self.gu_slabs:
            x_rot = rht_apply_in(flat, self.gu_signs_in)
            for fid, codes, scales, lut, (k, g, K, hmult) in self.gu_slabs:
                locs_f = gu_locs if single_gu else mx.where(gu_fids == fid, gu_locs, self._sentinel)
                y_f = moe_fn(
                    x_rot, codes, scales, lut, locs_f,
                    2 * self.cfg.intermediate_size, self.cfg.hidden_size, k, g, K, hmult, x_div=K_top,
                )
                y_gu_total = y_f if y_gu_total is None else (y_gu_total + y_f)
        if self.gu_nvfp4 is not None:
            c_u32, s_u8, o_f32 = self.gu_nvfp4
            locs_nv = gu_locs if len(self.gu_slabs) == 0 else mx.where(gu_fids == NVFP4_FID, gu_locs, self._sentinel)
            y_nv = nvfp4_matvec(
                flat, c_u32, s_u8, o_f32, locs_nv,
                2 * self.cfg.intermediate_size, self.cfg.hidden_size, x_div=K_top,
                e2m1_lut=self.e2m1_lut, e4m3_lut=self.e4m3_lut,
            )
            y_gu_total = y_nv if y_gu_total is None else (y_gu_total + y_nv)

        act = softcapped_swiglu(y_gu_total, self.cfg.mlp_gate_softcap, self.cfg.mlp_up_softcap)

        single_dn = (len(self.dn_slabs) == 1) and (self.dn_nvfp4 is None)
        y_dn_total = None
        if self.dn_slabs:
            act_rot = rht_apply_in(act, self.dn_signs_in)
            for fid, codes, scales, lut, (k, g, K, hmult) in self.dn_slabs:
                locs_f = dn_locs if single_dn else mx.where(dn_fids == fid, dn_locs, self._sentinel)
                y_f = moe_fn(
                    act_rot, codes, scales, lut, locs_f,
                    self.cfg.hidden_size, self.cfg.intermediate_size, k, g, K, hmult, x_div=1,
                )
                y_dn_total = y_f if y_dn_total is None else (y_dn_total + y_f)
        if self.dn_nvfp4 is not None:
            c_u32, s_u8, o_f32 = self.dn_nvfp4
            locs_nv = dn_locs if len(self.dn_slabs) == 0 else mx.where(dn_fids == NVFP4_FID, dn_locs, self._sentinel)
            y_nv = nvfp4_matvec(
                act, c_u32, s_u8, o_f32, locs_nv,
                self.cfg.hidden_size, self.cfg.intermediate_size, x_div=1,
                e2m1_lut=self.e2m1_lut, e4m3_lut=self.e4m3_lut,
            )
            y_dn_total = y_nv if y_dn_total is None else (y_dn_total + y_nv)

        out = mx.sum(y_dn_total.reshape(N, K_top, H) * w_norm[:, :, None], axis=1)
        return out.reshape(B, T, H)


# ----------------------------------------------------------------------------- Packed Embed & LM Head

class PackedEmbed:
    """100% GPU-resident packed token embedding lookup (zero CPU sync on H=2304; NumPy fallback for mini tests)."""

    def __init__(self, packed_dir: Path, luts: dict[str, mx.array], embed_norm_w: mx.array, eps: float):
        self.meta = json.loads((packed_dir / "embed_meta.json").read_text())
        npz = np.load(packed_dir / "embed_packed.npz")
        order = np.array(npz["order"], dtype=np.int32)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order), dtype=np.int32)
        self.embed_norm_w = embed_norm_w.astype(mx.float32)
        self.hidden_size = int(self.embed_norm_w.shape[0])
        self.eps = eps
        self._fast = self.hidden_size == 2304

        if self._fast:
            self.inv_order_mx = mx.array(inv.astype(np.uint32))
            codes_parts, scales_parts, levels_parts, signs_parts = [], [], [], []
            b_hi, b_dims = [], []
            c_off = s_off = l_off = hi = 0
            for bm in self.meta:
                kb = bm["kb"]
                rows = int(bm["rows"])
                opt = bm["opt"]
                k = int(bm["k"])
                fid = FMT_ID[opt]
                if opt in ("lm6", "lm8"):
                    raw_u8 = np.ascontiguousarray(np.array(npz[f"b{kb}_codes"], dtype=np.uint8)).reshape(rows, -1)
                    c_u32 = raw_u8.view(np.uint32)
                    lv = np.array(npz[f"b{kb}_levels"], dtype=np.float16).reshape(-1)
                else:
                    c_u32 = np.ascontiguousarray(np.array(npz[f"b{kb}_codes"], dtype=np.uint32)).reshape(rows, -1)
                    lv = np.zeros(0, dtype=np.float16)
                wpr = int(c_u32.shape[1])
                hi += rows
                b_hi.append(hi)
                b_dims.append([fid, k, wpr, c_off, s_off, l_off])
                codes_parts.append(c_u32.reshape(-1))
                scales_parts.append(np.array(npz[f"b{kb}_scales"], dtype=np.float16).reshape(-1))
                if len(lv):
                    levels_parts.append(lv)
                signs_parts.append(np.array(npz[f"b{kb}_signs_in"], dtype=np.float32).reshape(1, 2304))
                c_off += c_u32.size
                s_off += rows
                l_off += len(lv)
            codes_parts.append(np.zeros(4, dtype=np.uint32))
            if not levels_parts:
                levels_parts.append(np.zeros(4, dtype=np.float16))
            self.gpu_codes = mx.array(np.concatenate(codes_parts))
            self.gpu_scales = mx.array(np.concatenate(scales_parts))
            self.gpu_levels = mx.array(np.concatenate(levels_parts))
            self.gpu_signs_all = mx.array(np.concatenate(signs_parts, axis=0))
            self.gpu_b_hi = mx.array(np.array(b_hi, dtype=np.uint32))
            self.gpu_b_dims = mx.array(np.array(b_dims, dtype=np.uint32).reshape(-1))
            rtr_stack = [
                (luts[name].reshape(4096).astype(mx.float16) if name in luts else mx.zeros((4096,), dtype=mx.float16))
                for name in ("rtr1", "rtr2", "rtr3", "rtr4")
            ]
            self.gpu_rtr_luts = mx.concatenate(rtr_stack, axis=0)
            self.nb = len(self.meta)
            mx.eval(self.inv_order_mx, self.gpu_codes, self.gpu_scales, self.gpu_levels, self.gpu_signs_all, self.gpu_b_hi, self.gpu_b_dims, self.gpu_rtr_luts)
        else:
            self.inv_order = inv
            self.luts_np = {k: np.array(v, dtype=np.float32) for k, v in luts.items()}
            self.buckets: list[dict[str, Any]] = []
            lo = 0
            for bm in self.meta:
                kb = bm["kb"]
                rows = bm["rows"]
                opt = bm["opt"]
                binfo: dict[str, Any] = {"lo": lo, "hi": lo + rows, "opt": opt, "k": bm["k"]}
                binfo["codes"] = np.array(npz[f"b{kb}_codes"])
                binfo["scales"] = np.array(npz[f"b{kb}_scales"], dtype=np.float32)
                binfo["signs_in"] = mx.array(np.array(npz[f"b{kb}_signs_in"], dtype=np.float32))
                if opt.startswith("lm"):
                    binfo["levels"] = np.array(npz[f"b{kb}_levels"], dtype=np.float32)
                else:
                    binfo["lut"] = self.luts_np[opt]
                self.buckets.append(binfo)
                lo += rows

    def _decode_row(self, pos: int) -> mx.array:
        n = self.hidden_size
        for b in self.buckets:
            if b["lo"] <= pos < b["hi"]:
                r = pos - b["lo"]
                k = b["k"]
                scale = float(b["scales"][r])
                if b["opt"].startswith("lm"):
                    if k == 8:
                        idx = b["codes"].reshape(-1, n)[r]
                    else:
                        bpr = (n * 6) // 8
                        raw = b["codes"][r * bpr : (r + 1) * bpr].reshape(-1, 3).astype(np.uint32)
                        v = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
                        idx = np.stack([v & 63, (v >> 6) & 63, (v >> 12) & 63, (v >> 18) & 63], axis=-1).reshape(-1)
                    q = b["levels"][idx] * scale
                else:
                    wpr = (n * k) // 32
                    row_u32 = b["codes"][r].astype(np.uint64)
                    bp = (n - 1 - np.arange(n, dtype=np.int64)) * k
                    wi = bp >> 5
                    off = (bp & 31).astype(np.uint64)
                    w0 = row_u32[wi]
                    wi1 = np.minimum(wi + 1, wpr - 1)
                    w1 = np.where(wi + 1 < wpr, row_u32[wi1], np.uint64(0))
                    hi = np.where(off > 0, w1 << (np.uint64(32) - off), np.uint64(0))
                    s = ((w0 >> off) | hi) & np.uint64(4095)
                    q = b["lut"][s] * scale
                return rht_apply_out(mx.array(q[None, :]), b["signs_in"])[0]
        raise IndexError(f"Token position {pos} out of range")

    def __call__(self, input_ids: list[int] | np.ndarray | mx.array) -> mx.array:
        if self._fast:
            ids_mx = input_ids if isinstance(input_ids, mx.array) else mx.array(np.asarray(input_ids, dtype=np.int32))
            q_out, b_idx = embed_lookup_gpu(
                ids_mx, self.inv_order_mx, self.gpu_b_hi, self.gpu_b_dims,
                self.gpu_codes, self.gpu_scales, self.gpu_levels, self.gpu_rtr_luts, self.nb,
            )
            h = rht_out_2304_fast(q_out, self.gpu_signs_all, b_idx)[None, :, :]
            return rms_norm(h, self.embed_norm_w, self.eps)
        ids = np.asarray(input_ids, dtype=np.int32).reshape(-1)
        rows = [self._decode_row(int(self.inv_order[int(t)])) for t in ids]
        h = mx.stack(rows, axis=0)[None, :, :]
        return rms_norm(h, self.embed_norm_w, self.eps)


class PackedLMHead:
    """Executes unpadded `[200058, 2304]` lm_head in ONE fused DenseStage launch + lazy 32K MLX4 draft shortlist."""

    def __init__(self, packed_dir: Path, luts: dict[str, mx.array], fast_luts: dict[str, mx.array] | None, cap: float, hidden_size: int):
        self.meta = json.loads((packed_dir / "head_meta.json").read_text())
        self.cap = cap
        self.hidden_size = int(hidden_size)
        self._fast = self.hidden_size == 2304 and fast_luts is not None
        npz = np.load(packed_dir / "head_packed.npz")
        order = np.array(npz["order"], dtype=np.int32)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order), dtype=np.int32)
        self.order_np = order
        self.order_mx = mx.array(order)
        self.inv_order = mx.array(inv)
        self._draft_cache: dict[tuple[int, int], tuple[mx.array, mx.array, mx.array, mx.array, int]] = {}

        if self._fast:
            items = []
            for bm in self.meta:
                kb = bm["kb"]
                opt = bm["opt"]
                it: dict[str, Any] = {
                    "fmt": opt,
                    "m": int(bm["rows"]),
                    "codes": npz[f"b{kb}_codes"],
                    "scales": npz[f"b{kb}_scales"],
                    "signs_in": npz[f"b{kb}_signs_in"].astype(np.float32),
                }
                if opt.startswith("lm"):
                    it["levels"] = npz[f"b{kb}_levels"].astype(np.float32)
                items.append(it)
            self.stage = DenseStage(items, 2304, fast_luts)
            self.nb = len(items)
            self._fast_luts = fast_luts
            self._packed_dir = packed_dir
            self.buckets = []
        else:
            self.stage = None
            self.buckets: list[dict[str, Any]] = []
            for bm in self.meta:
                kb = bm["kb"]
                opt = bm["opt"]
                b: dict[str, Any] = {
                    "opt": opt,
                    "rows": int(bm["rows"]),
                    "k": int(bm["k"]),
                    "codes": mx.array(npz[f"b{kb}_codes"]),
                    "scales": mx.array(npz[f"b{kb}_scales"]),
                    "signs_in": mx.array(npz[f"b{kb}_signs_in"].astype(np.float32)),
                }
                if opt.startswith("lm"):
                    b["levels"] = mx.array(npz[f"b{kb}_levels"].astype(np.float32))
                else:
                    b["g"] = int(bm["g"])
                    b["K"] = int(bm["K"])
                    b["hash_mult"] = int(bm["hash_mult"])
                    b["lut"] = luts[opt]
                self.buckets.append(b)

    def _ordered_logits(self, x2: mx.array) -> mx.array:
        T = int(x2.shape[0])
        if self._fast:
            if 2 <= T <= 8:
                x_rot_tr = rht_in_2304_fast(x2, self.stage.signs_in, n_signs=self.nb, T=T, out_dtype=mx.float16, transpose_for_sg8=True)
                return self.stage.matmul_rotated_sg8_f16(x_rot_tr, T=T)
            x_rot = rht_in_2304_fast(x2, self.stage.signs_in, n_signs=self.nb, T=T, out_dtype=mx.float16)
            return self.stage.matmul_rotated_f16(x_rot, T)
        parts = []
        for b in self.buckets:
            x_rot = rht_apply_in(x2, b["signs_in"])
            if b["opt"].startswith("lm"):
                y = lm_dense_matmul(x_rot, b["codes"], b["scales"], b["levels"], b["rows"], self.hidden_size, b["k"])
            else:
                y = trellis_dense_matmul(
                    x_rot, b["codes"], b["scales"], b["lut"],
                    b["rows"], self.hidden_size, b["k"], b["g"], b["K"], b["hash_mult"],
                )
            parts.append(y)
        return mx.concatenate(parts, axis=-1)

    def __call__(self, h: mx.array) -> mx.array:
        B, T, H = h.shape
        ordered = self._ordered_logits(h.reshape(B * T, H))
        logits = ordered[:, self.inv_order]
        return softcap(logits, self.cap).reshape(B, T, -1)

    def ordered_logits(self, h: mx.array) -> mx.array:
        """Raw pre-softcap logits `[B, T, V]` in bucket-ordered layout (`col o <-> order_mx[o]`), skipping 200K `inv_order` gather."""
        B, T, H = h.shape
        return self._ordered_logits(h.reshape(B * T, H)).reshape(B, T, -1)

    def argmax(self, h: mx.array) -> mx.array:
        """Exact greedy argmax skipping `inv_order` permutation and monotonic `softcap`. Returns `int32[B, T]`."""
        B, T, H = h.shape
        ordered = self._ordered_logits(h.reshape(B * T, H))
        best_ord = mx.argmax(ordered, axis=-1)
        return self.order_mx[best_ord].reshape(B, T)

    def _ensure_draft_head(self, max_buckets: int = 5, bits: int = 4):
        key = (max_buckets, bits)
        if key not in self._draft_cache:
            nb = min(max_buckets, self.nb)
            W_full = self.stage.dequant_f16()
            wqs, scs, bis = [], [], []
            total_rows = 0
            for bi in range(nb):
                lo, hi = self.stage.slices[bi]
                rows_b = hi - lo
                total_rows += rows_b
                Q_b = W_full[lo:hi].astype(mx.float32)  # [rows_b, 2304]
                W_b = rht_out_2304_fast(Q_b, self.stage.signs_in[bi : bi + 1], mx.zeros((rows_b,), dtype=mx.uint32))
                wq, sc, bi_q = mx.quantize(W_b.astype(mx.float16), group_size=64, bits=bits)
                mx.eval(wq, sc, bi_q)
                wqs.append(wq)
                scs.append(sc)
                bis.append(bi_q)
            del W_full
            wq_all = mx.concatenate(wqs, axis=0)
            sc_all = mx.concatenate(scs, axis=0)
            bi_all = mx.concatenate(bis, axis=0)
            order_sub = self.order_mx[:total_rows]
            mx.eval(wq_all, sc_all, bi_all, order_sub)
            mx.clear_cache()
            self._draft_cache[key] = (wq_all, sc_all, bi_all, order_sub, bits)
        return self._draft_cache[key]

    def get_draft_head_argmax(self, max_buckets: int = 5, bits: int = 4):
        """Lazily build a compact MLX 4-bit shortlist head over the top Fisher/frequency buckets (`b0..4` = 32,768 rows, ~38 MB).

        Only used by DSpark's 7-step drafter RNN; the target always verifies against the full 200,058-row packed head.
        """
        if not self._fast:
            return lambda h_1d: self.argmax(h_1d[None, None, :])[0, 0]
        wq_all, sc_all, bi_all, order_sub, qbits = self._ensure_draft_head(max_buckets, bits)

        def _draft_argmax(h_1d: mx.array) -> mx.array:
            scores = mx.quantized_matmul(
                h_1d.reshape(1, 2304).astype(mx.float16), wq_all, sc_all, bi_all,
                transpose=True, group_size=64, bits=qbits,
            )[0]
            return order_sub[mx.argmax(scores)]

        return _draft_argmax

    def get_draft_head_topk(self, max_buckets: int = 5, bits: int = 4):
        """Batched top-k shortlist scorer sharing the cached 32K MLX 4-bit head. Input `[B, 2304]` or `[2304]`, returns `int32[B, k]`."""
        if not self._fast:
            def _fallback_topk(h: mx.array, k: int = 1) -> mx.array:
                h2 = h.reshape(-1, self.hidden_size)
                logits = self(h2[None, :, :])[0]
                if k == 1:
                    return mx.argmax(logits, axis=-1, keepdims=True).astype(mx.int32)
                part = mx.argpartition(-logits, k - 1, axis=-1)[:, :k]
                vals = mx.take_along_axis(logits, part, axis=-1)
                ord_k = mx.argsort(-vals, axis=-1)
                return mx.take_along_axis(part, ord_k, axis=-1).astype(mx.int32)
            return _fallback_topk

        wq_all, sc_all, bi_all, order_sub, qbits = self._ensure_draft_head(max_buckets, bits)

        def _draft_topk(h: mx.array, k: int = 1) -> mx.array:
            scores = mx.quantized_matmul(
                h.reshape(-1, 2304).astype(mx.float16), wq_all, sc_all, bi_all,
                transpose=True, group_size=64, bits=qbits,
            )  # [B, 32768]
            if k == 1:
                return order_sub[mx.argmax(scores, axis=-1, keepdims=True)].astype(mx.int32)
            part = mx.argpartition(-scores, k - 1, axis=-1)[:, :k]
            vals = mx.take_along_axis(scores, part, axis=-1)
            ord_k = mx.argsort(-vals, axis=-1)
            top_idx = mx.take_along_axis(part, ord_k, axis=-1)
            return order_sub[top_idx].astype(mx.int32)

        return _draft_topk

    def get_draft_head_sampler(
        self,
        temp: float,
        top_p: float = 1.0,
        top_k: int = 0,
        max_buckets: int = 5,
        bits: int = 4,
    ):
        """Stochastic draft shortlist sampler sharing the cached 32K MLX 4-bit head.

        Returns `_draft_sample(h_1d) -> (vocab_tok_i32, ord_tok_i32, probs_q_f32[S])`, aligned 1-to-1 with `ordered_logits[:, :S]`.
        """
        cap = self.cap
        if not self._fast:
            order_mx = self.order_mx

            def _draft_sample_full(h_1d: mx.array) -> tuple[mx.array, mx.array, mx.array]:
                raw = self._ordered_logits(h_1d.reshape(1, self.hidden_size))[0]
                probs_q = logits_to_probs(softcap(raw, cap), temp=temp, top_p=top_p, top_k=top_k)
                logp = mx.where(probs_q > 0.0, mx.log(mx.maximum(probs_q, 1e-38)), -float("inf"))
                ord_tok = mx.random.categorical(logp).astype(mx.int32)
                return order_mx[ord_tok].astype(mx.int32), ord_tok, probs_q

            return _draft_sample_full

        wq_all, sc_all, bi_all, order_sub, qbits = self._ensure_draft_head(max_buckets, bits)

        def _draft_sample(h_1d: mx.array) -> tuple[mx.array, mx.array, mx.array]:
            raw = mx.quantized_matmul(
                h_1d.reshape(1, 2304).astype(mx.float16), wq_all, sc_all, bi_all,
                transpose=True, group_size=64, bits=qbits,
            )[0]
            probs_q = logits_to_probs(softcap(raw.astype(mx.float32), cap), temp=temp, top_p=top_p, top_k=top_k)
            logp = mx.where(probs_q > 0.0, mx.log(mx.maximum(probs_q, 1e-38)), -float("inf"))
            ord_tok = mx.random.categorical(logp).astype(mx.int32)
            return order_sub[ord_tok].astype(mx.int32), ord_tok, probs_q

        return _draft_sample


# ----------------------------------------------------------------------------- Decoder Layer & Full Looped Model

@mx.compile
def _block_attn_res_compiled(q: mx.array, stacked: mx.array, inv_sqrt_h: mx.array) -> mx.array:
    st_n = mx.fast.rms_norm(stacked, None, 1e-6)
    scores = mx.einsum("qh,btsh->btqs", q, st_n) * inv_sqrt_h
    w = mx.softmax(scores, axis=-1)
    return mx.einsum("btqs,btsh->btqh", w, st_n)


def block_attn_res(query: mx.array, block_states: list[mx.array], partial: mx.array) -> tuple[mx.array, ...]:
    stacked = mx.stack(block_states + [partial.astype(mx.float32)], axis=-2)
    H = stacked.shape[-1]
    q = query.astype(mx.float32).reshape(-1, H)
    out = _block_attn_res_compiled(q, stacked, _f32(H ** -0.5))
    return tuple(out[:, :, i, :] for i in range(q.shape[0]))


@mx.compile
def _message_gate_compiled(h: mx.array, msg: mx.array, down_w: mx.array, forget_w: mx.array, msg_w: mx.array) -> mx.array:
    cat = mx.concatenate([mx.fast.rms_norm(h, None, 1e-6), mx.fast.rms_norm(msg, None, 1e-6)], axis=-1).astype(down_w.dtype)
    g = cat @ down_w.T
    fg = mx.tanh((g @ forget_w.T).astype(mx.float32))
    mg = mx.tanh((g @ msg_w.T).astype(mx.float32))
    return (1.0 - fg) * h + mg * msg


def message_gate(h: mx.array, msg: mx.array, down_w: mx.array, forget_w: mx.array, msg_w: mx.array, eps: float) -> mx.array:
    return _message_gate_compiled(h.astype(mx.float32), msg.astype(mx.float32), down_w, forget_w, msg_w)


@mx.compile
def _short_conv4_t1(ctx: mx.array, res: mx.array, w_kc: mx.array, inv_s: mx.array) -> mx.array:
    return (mx.sum(ctx * w_kc[None, :, :], axis=1, keepdims=True) + res) * inv_s


@mx.compile
def _short_conv4_tn(ctx: mx.array, res: mx.array, w_kc: mx.array, inv_s: mx.array) -> mx.array:
    T = res.shape[1]
    out = (
        ctx[:, 0:T, :] * w_kc[None, 0:1, :]
        + ctx[:, 1 : T + 1, :] * w_kc[None, 1:2, :]
        + ctx[:, 2 : T + 2, :] * w_kc[None, 2:3, :]
        + ctx[:, 3 : T + 3, :] * w_kc[None, 3:4, :]
    )
    return (out + res) * inv_s


def short_conv(x: mx.array, w_kc: mx.array, cache: Inkling2Cache, slot: int, conv_idx: int) -> mx.array:
    K = w_kc.shape[0]
    xf = x.astype(mx.float32)
    ctx = cache.update_conv(slot, conv_idx, xf, K)
    inv_s = _f32(1.0 / math.sqrt(K + 1))
    if K == 4:
        return _short_conv4_t1(ctx, xf, w_kc, inv_s) if xf.shape[1] == 1 else _short_conv4_tn(ctx, xf, w_kc, inv_s)
    T = xf.shape[1]
    acc = mx.zeros_like(xf)
    for lag in range(K):
        acc = acc + ctx[:, lag : lag + T, :] * w_kc[None, lag : lag + 1, :]
    return (acc + xf) * inv_s


@mx.compile
def _dyn_conv4_compiled(ctx: mx.array, res: mx.array, dyn_in: mx.array, st_hk: mx.array, proj_w: mx.array, inv_s: mx.array) -> mx.array:
    B, T, C = res.shape
    num_heads = st_hk.shape[2]
    head_dim = st_hk.shape[3]
    dyn = (dyn_in.astype(mx.float32) @ proj_w.T).reshape(B, T, num_heads, 1, 4)
    w = st_hk + dyn
    c0 = ctx[:, 3 : T + 3, :].reshape(B, T, num_heads, head_dim)
    c1 = ctx[:, 2 : T + 2, :].reshape(B, T, num_heads, head_dim)
    c2 = ctx[:, 1 : T + 1, :].reshape(B, T, num_heads, head_dim)
    c3 = ctx[:, 0 : T, :].reshape(B, T, num_heads, head_dim)
    out = (c0 * w[..., 0] + c1 * w[..., 1] + c2 * w[..., 2] + c3 * w[..., 3]).reshape(B, T, C)
    return (out + res) * inv_s


def dynamic_short_conv(
    x: mx.array,
    dyn_in: mx.array,
    static_w: mx.array,
    proj_w: mx.array,
    num_heads: int,
    cache: Inkling2Cache,
    slot: int,
    conv_idx: int,
) -> mx.array:
    B, T, C = x.shape
    K = static_w.shape[0]
    head_dim = C // num_heads
    xf = x.astype(mx.float32)
    ctx = cache.update_conv(slot, conv_idx, xf, K)
    inv_s = _f32(1.0 / math.sqrt(2 * K + 1))
    st = static_w.T.reshape(1, 1, num_heads, head_dim, K)
    if K == 4:
        return _dyn_conv4_compiled(ctx, xf, dyn_in, st, proj_w, inv_s)
    dyn = (dyn_in.astype(mx.float32) @ proj_w.T).reshape(B, T, num_heads, 1, K)
    weights = st + dyn
    acc = mx.zeros((B, T, num_heads, head_dim), dtype=mx.float32)
    for lag in range(K):
        sl = ctx[:, (K - 1 - lag) : (K - 1 - lag + T), :].reshape(B, T, num_heads, head_dim)
        acc = acc + sl * weights[:, :, :, :, lag]
    return (acc.reshape(B, T, C) + xf) * inv_s


@mx.compile
def _gqa_sdpa_sinks_compiled(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    pos_bias: mx.array,
    allowed: mx.array,
    sinks: mx.array,
    inv_sqrt_d: mx.array,
) -> mx.array:
    scores = mx.matmul(q, k.transpose(0, 1, 2, 4, 3)) * inv_sqrt_d + pos_bias
    scores = mx.where(allowed, scores, -1e9)
    combined = mx.concatenate([scores, sinks], axis=-1)
    probs = mx.softmax(combined, axis=-1)[..., :-1]
    return mx.matmul(probs, v)


def _gqa_sdpa_sinks(
    q: mx.array,
    K_all: mx.array,
    V_all: mx.array,
    pos_bias: mx.array,
    allowed: mx.array,
    sinks: mx.array,
    n_kv: int,
    n_rep: int,
    head_dim: int,
) -> mx.array:
    B, _, T, D = q.shape
    kv_len = K_all.shape[2]
    qg = q.reshape(B, n_kv, n_rep, T, D)
    kg = K_all[:, :, None, :, :]
    vg = V_all[:, :, None, :, :]
    pb = pos_bias.reshape(B, n_kv, n_rep, T, kv_len)
    al = allowed[None, None, None, :, :]
    sk = sinks.reshape(B, n_kv, n_rep, T, 1)
    out = _gqa_sdpa_sinks_compiled(qg, kg, vg, pb, al, sk, _f32(head_dim ** -0.5))
    return out.reshape(B, n_kv * n_rep, T, D).transpose(0, 2, 1, 3).reshape(B, T, n_kv * n_rep * D)


@mx.compile
def _msg_3x_shared_h_compiled(h: mx.array, msgs3: mx.array, dh3: mx.array, dm3: mx.array, fm3: mx.array) -> mx.array:
    hn = mx.fast.rms_norm(h, None, 1e-6).astype(mx.bfloat16)
    mn = mx.fast.rms_norm(msgs3, None, 1e-6).astype(mx.bfloat16)
    g = mx.matmul(hn[None, :, :], dh3.swapaxes(-1, -2)) + mx.matmul(mn, dm3.swapaxes(-1, -2))
    return mx.matmul(g, fm3.swapaxes(-1, -2))


@mx.compile
def _msg_3x_split_h_compiled(h3: mx.array, msgs3: mx.array, dh3: mx.array, dm3: mx.array, fm3: mx.array) -> mx.array:
    hn = mx.fast.rms_norm(h3, None, 1e-6).astype(mx.bfloat16)
    mn = mx.fast.rms_norm(msgs3, None, 1e-6).astype(mx.bfloat16)
    g = mx.matmul(hn, dh3.swapaxes(-1, -2)) + mx.matmul(mn, dm3.swapaxes(-1, -2))
    return mx.matmul(g, fm3.swapaxes(-1, -2))


@mx.compile
def _msg_1x_compiled(h: mx.array, msg: mx.array, dh: mx.array, dm: mx.array, fm: mx.array) -> mx.array:
    hn = mx.fast.rms_norm(h, None, 1e-6).astype(mx.bfloat16)
    mn = mx.fast.rms_norm(msg, None, 1e-6).astype(mx.bfloat16)
    return (hn @ dh.T + mn @ dm.T) @ fm.T


class PackedDecoderLayer:
    def __init__(
        self,
        layer_idx: int,
        cfg: Inkling2Config,
        packed_dir: Path,
        luts: dict[str, mx.array],
        fast_luts: dict[str, mx.array] | None,
        st: dict[str, mx.array],
    ):
        self.layer_idx = layer_idx
        self.cfg = cfg
        self._fast = cfg.use_fast_metal and fast_luts is not None
        self.is_sliding = layer_idx in cfg.local_layer_ids
        self.is_loop = cfg.loop_start <= layer_idx < cfg.loop_end
        self.is_block_boundary = (layer_idx % cfg.n_layers_per_attn_res_block) == 0

        p = f"model.layers.{layer_idx}."
        meta = json.loads((packed_dir / f"layer_{layer_idx:02d}_meta.json").read_text())
        npz = np.load(packed_dir / f"layer_{layer_idx:02d}_packed.npz")
        arrays = {k: npz[k] for k in npz.files}

        lin_meta = meta["linears"]
        if self._fast:
            def _lin_item(key_name: str) -> dict[str, Any]:
                lm = lin_meta[key_name]
                safe = lm["safe"]
                it: dict[str, Any] = {
                    "fmt": lm["fmt"],
                    "m": int(lm["m"]),
                    "codes": arrays[f"lin_{safe}_codes"],
                    "scales": arrays[f"lin_{safe}_scales"],
                    "signs_in": arrays[f"lin_{safe}_signs_in"].astype(np.float32),
                    "out_rot": bool(lm.get("out_rot", False)),
                }
                if it["out_rot"]:
                    it["signs_out"] = mx.array(arrays[f"lin_{safe}_signs_out"].astype(np.float32))
                if lm["fmt"].startswith("lm"):
                    it["levels"] = arrays[f"lin_{safe}_levels"].astype(np.float32)
                return it

            qkv_items = [
                _lin_item("self_attn.q_proj.linear"),
                _lin_item("self_attn.k_proj.linear"),
                _lin_item("self_attn.v_proj.linear"),
                _lin_item("self_attn.relative_position_bias.linear"),
                _lin_item("self_attn.o_proj.gate_proj"),
            ]
            self.qkv_stage = DenseStage(qkv_items, 2304, fast_luts)
            self.qkv_out_rots = [it.get("signs_out") if it["out_rot"] else None for it in qkv_items]
            self.has_qkv_out_rot = any(r is not None for r in self.qkv_out_rots[:4])
            o_item = _lin_item("self_attn.o_proj.linear")
            self.o_stage = DenseStage([o_item], 3072, fast_luts)
            self.o_out_rot = o_item.get("signs_out") if o_item["out_rot"] else None
        else:
            self.q_lin = PackedLinear(lin_meta["self_attn.q_proj.linear"], arrays, luts)
            self.k_lin = PackedLinear(lin_meta["self_attn.k_proj.linear"], arrays, luts)
            self.v_lin = PackedLinear(lin_meta["self_attn.v_proj.linear"], arrays, luts)
            self.o_lin = PackedLinear(lin_meta["self_attn.o_proj.linear"], arrays, luts)
            self.o_gate = PackedLinear(lin_meta["self_attn.o_proj.gate_proj"], arrays, luts)
            self.rel_lin = PackedLinear(lin_meta["self_attn.relative_position_bias.linear"], arrays, luts)

        # Native steering weights
        self.attn_in_q = st[p + "attention_input.query"].astype(mx.float32) if (p + "attention_input.query") in st else None
        self.mlp_in_q = st[p + "mlp_input.query"].astype(mx.float32)
        if self.is_sliding:
            self.in_ln = st[p + "input_layernorm.weight"].astype(mx.float32)
        else:
            self.q_in_ln = st[p + "q_input_layernorm.weight"].astype(mx.float32)
            self.k_in_ln = st[p + "k_input_layernorm.weight"].astype(mx.float32)
            self.v_in_ln = st[p + "v_input_layernorm.weight"].astype(mx.float32)
            self.qkv_in_ln = mx.stack([self.q_in_ln, self.k_in_ln, self.v_in_ln], axis=0) if self._fast else None
        self.post_attn_ln = st[p + "post_attention_layernorm.weight"].astype(mx.float32)

        self.q_scale = st[p + "self_attn.q_proj.norm.scale"].astype(mx.float32)
        self.rel_scale = st[p + "self_attn.relative_position_bias.norm.scale"].astype(mx.float32)
        self.rel_proj = st[p + "self_attn.relative_position_bias.proj"].astype(mx.float32)
        self.rel_proj_t_f16 = self.rel_proj.T.astype(mx.float16) if self._fast else None
        self.sink_w = st[p + "self_attn.sink_proj.weight"].astype(mx.float32)
        self.sink_w_f16 = self.sink_w.astype(mx.float16) if self._fast else None

        if self.is_sliding:
            self.k_conv_w = st[p + "self_attn.k_proj.sconv.conv1d.weight"][:, 0, :].T.astype(mx.float32)
            self.v_conv_w = st[p + "self_attn.v_proj.sconv.conv1d.weight"][:, 0, :].T.astype(mx.float32)
        else:
            self.k_dyn_st = st[p + "self_attn.k_proj.sconv.static_weight"].astype(mx.float32)
            self.k_dyn_pr = st[p + "self_attn.k_proj.sconv.weight_proj.weight"].astype(mx.float32)
            self.v_dyn_st = st[p + "self_attn.v_proj.sconv.static_weight"].astype(mx.float32)
            self.v_dyn_pr = st[p + "self_attn.v_proj.sconv.weight_proj.weight"].astype(mx.float32)
            if self._fast:
                self.k_dyn_pr_f16 = self.k_dyn_pr.astype(mx.float16)
                self.v_dyn_pr_f16 = self.v_dyn_pr.astype(mx.float16)

        self.post_attn_conv_w = st[p + "post_attention_convolution.conv1d.weight"][:, 0, :].T.astype(mx.float32)
        self.post_mlp_conv_w = st[p + "post_mlp_convolution.conv1d.weight"][:, 0, :].T.astype(mx.float32)

        if self.is_loop:
            self.msg_gates = {}
            H = cfg.hidden_size
            qkv_dh, qkv_dm, qkv_fm = [], [], []
            for gn in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"):
                dw = st[p + f"{gn}.down_proj.weight"].astype(mx.bfloat16)
                fw = st[p + f"{gn}.forget_gate_proj.weight"].astype(mx.bfloat16)
                mw = st[p + f"{gn}.message_gate_proj.weight"].astype(mx.bfloat16)
                if self._fast:
                    fm = mx.concatenate([fw, mw], axis=0)
                    if gn == "mlp_message_gate":
                        self.mlp_dh, self.mlp_dm, self.mlp_fm = dw[:, :H], dw[:, H:], fm
                    else:
                        qkv_dh.append(dw[:, :H])
                        qkv_dm.append(dw[:, H:])
                        qkv_fm.append(fm)
                else:
                    self.msg_gates[gn] = (dw, fw, mw)
            if self._fast:
                self.qkv_dh3 = mx.stack(qkv_dh, axis=0)
                self.qkv_dm3 = mx.stack(qkv_dm, axis=0)
                self.qkv_fm3 = mx.stack(qkv_fm, axis=0)

        self.moe = PackedMoE(
            arrays, luts, fast_luts,
            st[p + "mlp.gate.weight"],
            st[p + "mlp.gate.e_score_correction_bias"],
            cfg,
        )

    def forward_fast(
        self,
        h: mx.array,
        blocks_cat: mx.array | None,
        n_blocks: int,
        q_pos0: int,
        T: int,
        cache: Inkling2Cache,
        cache_slot: int,
        messages: mx.array | None = None,
        tree_spec: dict | None = None,
        qkv_w_cached: mx.array | None = None,
        o_w_cached: mx.array | None = None,
    ) -> mx.array:
        """Ultra-fast fused Metal path for Inkling-2-Mini (B=1, h shape `[T, 2304]`)."""
        cfg = self.cfg
        block_residual = h
        tree_lag = tree_spec["lag_mx"] if tree_spec is not None else None
        tree_depth = tree_spec["depth_mx"] if tree_spec is not None else None
        tree_mask = tree_spec["mask_mx"] if tree_spec is not None else None

        # 1. Attn input AttnRes + optional batched message gates + layernorm + 5-matrix batched RHT-2304
        use_qkv_sg8 = 5 <= T <= 8
        if self.is_sliding:
            if messages is None:
                if self.attn_in_q is not None:
                    q_in = block_attn_res_fused(self.attn_in_q, blocks_cat, n_blocks, h, norm_w=self.in_ln)[0]
                else:
                    q_in = rms_norm(h, self.in_ln, cfg.rms_norm_eps)
                x_rot5 = rht_in_2304_fast(q_in, self.qkv_stage.signs_in, n_signs=5, T=T, split_qkv=False, out_dtype=mx.float16, transpose_for_sg8=use_qkv_sg8)
            else:
                h_a = block_attn_res_fused(self.attn_in_q, blocks_cat, n_blocks, h)[0]
                fg_mg3 = _msg_3x_shared_h_compiled(h_a, messages[:3], self.qkv_dh3, self.qkv_dm3, self.qkv_fm3)
                qkv_in = msg_blend_norm_3x_fused(h_a, messages[:3], fg_mg3, self.in_ln, T, h_shared=True, norm_shared=True)
                q_in = qkv_in[0]
                x_rot5 = rht_in_2304_fast(qkv_in, self.qkv_stage.signs_in, n_signs=5, T=T, split_qkv=True, out_dtype=mx.float16, transpose_for_sg8=use_qkv_sg8)
        else:
            if self.attn_in_q is None:
                qkv_h = mx.broadcast_to(h[None, :, :], (3, T, 2304))
            else:
                qkv_h = block_attn_res_fused(self.attn_in_q, blocks_cat, n_blocks, h)
            if messages is not None:
                fg_mg3 = _msg_3x_split_h_compiled(qkv_h, messages[:3], self.qkv_dh3, self.qkv_dm3, self.qkv_fm3)
                qkv_in = msg_blend_norm_3x_fused(qkv_h, messages[:3], fg_mg3, self.qkv_in_ln, T, h_shared=False, norm_shared=False)
                q_in, k_in, v_in = qkv_in[0], qkv_in[1], qkv_in[2]
            else:
                q_in = rms_norm(qkv_h[0], self.q_in_ln, cfg.rms_norm_eps)
                k_in = rms_norm(qkv_h[1], self.k_in_ln, cfg.rms_norm_eps)
                v_in = rms_norm(qkv_h[2], self.v_in_ln, cfg.rms_norm_eps)
                qkv_in = mx.stack([q_in, k_in, v_in], axis=0)
            x_rot5 = rht_in_2304_fast(qkv_in, self.qkv_stage.signs_in, n_signs=5, T=T, split_qkv=True, out_dtype=mx.float16, transpose_for_sg8=use_qkv_sg8)

        # 2. Fused 5-matrix QKV + Rel + O_Gate stage (1 launch!)
        use_split_parts = (T > 8) and not (self.is_sliding and not self.has_qkv_out_rot)
        if use_split_parts:
            q_raw, k_raw, v_raw, rel_raw, o_gate_raw = self.qkv_stage.matmul_rotated_f16_parts(x_rot5, T, W_cached=qkv_w_cached)
            y_qkv = None
        else:
            y_qkv = (
                self.qkv_stage.matmul_rotated_sg8_f16(x_rot5, T=T, rb=1, sg=8, ntg=160)
                if use_qkv_sg8
                else self.qkv_stage.matmul_rotated_f16(x_rot5, T, W_cached=qkv_w_cached)
            )  # [T, 8576]
            o_gate_raw = y_qkv[:, 5504:8576]
        if self.qkv_out_rots[4] is not None:
            o_gate_raw = rht_apply_out(o_gate_raw, self.qkv_out_rots[4])

        # 3. Fused Q/Rel clip (+log_scale) + K/V conv/norm + unified FP16 KV append
        kp = cache.get_conv_prev(cache_slot, 0, 1024)
        vp = cache.get_conv_prev(cache_slot, 1, 1024)
        if self.is_sliding and not self.has_qkv_out_rot:
            q, rel_st, kv_f16, new_kp, new_vp = qkv_rel_prep_sliding_fused(
                y_qkv, self.q_scale, self.rel_scale,
                kp, self.k_conv_w, vp, self.v_conv_w,
                T, cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold,
                tree_lag=tree_lag,
            )
            cache.set_conv_after(cache_slot, 0, kp, y_qkv[:, 3072:4096].reshape(1, T, 1024), new_kp)
            cache.set_conv_after(cache_slot, 1, vp, y_qkv[:, 4096:5120].reshape(1, T, 1024), new_vp)
        else:
            if not use_split_parts:
                q_raw = y_qkv[:, 0:3072]
                k_raw = y_qkv[:, 3072:4096]
                v_raw = y_qkv[:, 4096:5120]
                rel_raw = y_qkv[:, 5120:5504]
            if self.qkv_out_rots[0] is not None:
                q_raw = rht_apply_out(q_raw, self.qkv_out_rots[0])
            if self.qkv_out_rots[1] is not None:
                k_raw = rht_out_1024_fast(k_raw, self.qkv_out_rots[1]) if k_raw.shape[-1] == 1024 else rht_apply_out(k_raw, self.qkv_out_rots[1])
            if self.qkv_out_rots[2] is not None:
                v_raw = rht_out_1024_fast(v_raw, self.qkv_out_rots[2]) if v_raw.shape[-1] == 1024 else rht_apply_out(v_raw, self.qkv_out_rots[2])
            if self.qkv_out_rots[3] is not None:
                rel_raw = rht_apply_out(rel_raw, self.qkv_out_rots[3])
            q, rel_st = q_rel_clip_fused(
                q_raw, self.q_scale, rel_raw, self.rel_scale,
                q_pos0, T, self.is_sliding,
                cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold,
                cfg.log_scaling_alpha, cfg.log_scaling_n_floor,
                tree_depth=tree_depth,
            )
            if self.is_sliding:
                kv_f16, new_kp, new_vp = kv_sconv4_norm_fused(kp, k_raw, self.k_conv_w, vp, v_raw, self.v_conv_w, T, cfg.qk_l2_clip_threshold, tree_lag=tree_lag)
            else:
                kv_f16, new_kp, new_vp = kv_dyn_sconv4_norm_fused(
                    kp, k_raw, k_in, self.k_dyn_st, self.k_dyn_pr_f16,
                    vp, v_raw, v_in, self.v_dyn_st, self.v_dyn_pr_f16,
                    T, cfg.qk_l2_clip_threshold, tree_lag=tree_lag,
                )
            cache.set_conv_after(cache_slot, 0, kp, k_raw.reshape(1, T, 1024), new_kp)
            cache.set_conv_after(cache_slot, 1, vp, v_raw.reshape(1, T, 1024), new_vp)

        KV_buf, kv_len, kv_start, kv_cap = cache.update_kv_f16(cache_slot, kv_f16, self.is_sliding)

        # 4. Fused Sinks + GQA+RelPos Flash-Decoding / FlashAttention-2 SDPA (1 launch!)
        attn_out = gqa_relpos_sinks_sdpa_fast(
            q, KV_buf, rel_st, self.rel_proj_t_f16, q_in, self.sink_w_f16,
            T, kv_len, kv_start, q_pos0, kv_cap, self.is_sliding,
            tree_depth=tree_depth, tree_mask=tree_mask,
        )

        # 5. Fused Gate*Attn + RHT-3072 -> O projection -> Post-Attn ShortConv4 (+ Residual)
        o_rot = rht_in_3072_gated_fast(attn_out, o_gate_raw, self.o_stage.signs_in, out_dtype=mx.float16)
        o_out = self.o_stage.matmul_rotated_f16(o_rot, T, W_cached=o_w_cached)
        if self.o_out_rot is not None:
            o_out = rht_apply_out(o_out, self.o_out_rot)
        pa_prev = cache.get_conv_prev(cache_slot, 2, 2304)
        if self.is_block_boundary:
            h_post, new_pa = short_conv4_res_fused(pa_prev, o_out, self.post_attn_conv_w, T, 2304, residual=None, tree_lag=tree_lag)
            residual = h_post
            local_blocks = block_residual[None, :, :] if n_blocks == 0 else mx.concatenate([blocks_cat, block_residual[None, :, :]], axis=0)
            n_local = n_blocks + 1
        else:
            h_post, new_pa = short_conv4_res_fused(pa_prev, o_out, self.post_attn_conv_w, T, 2304, residual=block_residual, tree_lag=tree_lag)
            residual = h_post
            local_blocks = blocks_cat
            n_local = n_blocks
        cache.set_conv_after(cache_slot, 2, pa_prev, o_out.reshape(1, T, 2304), new_pa)

        # 6+7. Fused MLP input AttnRes / MsgBlend + Norm + RHT-2304(gu_signs) -> Unified MoE + post-MLP conv + residual!
        if messages is None:
            h_mlp_f16, x_rot_gu = block_attn_res_norm_rht_fast(self.mlp_in_q, local_blocks, n_local, h_post, self.post_attn_ln, self.moe.gu_signs_in)
        else:
            h_m = block_attn_res_fused(self.mlp_in_q, local_blocks, n_local, h_post)[0]
            fg_mg = _msg_1x_compiled(h_m, messages[3], self.mlp_dh, self.mlp_dm, self.mlp_fm)
            h_mlp_f16, x_rot_gu = msg_blend_norm_rht_fast(h_m, messages[3], fg_mg, self.post_attn_ln, self.moe.gu_signs_in)

        return self.moe.forward_fast_fused_prerot(h_mlp_f16, x_rot_gu, residual, self.post_mlp_conv_w, cache, cache_slot, T, tree_lag=tree_lag)

    def __call__(
        self,
        hidden_states: mx.array,
        block_states: list[mx.array],
        position_ids: mx.array,
        cache: Inkling2Cache,
        cache_slot: int,
        messages: tuple[mx.array, mx.array, mx.array, mx.array] | None = None,
    ) -> mx.array:
        cfg = self.cfg
        B, T, H = hidden_states.shape
        block_residual = hidden_states

        if self.is_sliding:
            h = block_attn_res(self.attn_in_q, block_states, hidden_states)[0] if self.attn_in_q is not None else hidden_states
            if messages is None:
                q_in = k_in = v_in = rms_norm(h, self.in_ln, cfg.rms_norm_eps)
            else:
                q_m, k_m, v_m, _ = messages
                q_in = rms_norm(message_gate(h, q_m, *self.msg_gates["q_message_gate"], cfg.rms_norm_eps), self.in_ln, cfg.rms_norm_eps)
                k_in = rms_norm(message_gate(h, k_m, *self.msg_gates["k_message_gate"], cfg.rms_norm_eps), self.in_ln, cfg.rms_norm_eps)
                v_in = rms_norm(message_gate(h, v_m, *self.msg_gates["v_message_gate"], cfg.rms_norm_eps), self.in_ln, cfg.rms_norm_eps)
        else:
            if self.attn_in_q is None:
                q_h = k_h = v_h = hidden_states
            else:
                q_h, k_h, v_h = block_attn_res(self.attn_in_q, block_states, hidden_states)
            if messages is not None:
                q_m, k_m, v_m, _ = messages
                q_h = message_gate(q_h, q_m, *self.msg_gates["q_message_gate"], cfg.rms_norm_eps)
                k_h = message_gate(k_h, k_m, *self.msg_gates["k_message_gate"], cfg.rms_norm_eps)
                v_h = message_gate(v_h, v_m, *self.msg_gates["v_message_gate"], cfg.rms_norm_eps)
            q_in = rms_norm(q_h, self.q_in_ln, cfg.rms_norm_eps)
            k_in = rms_norm(k_h, self.k_in_ln, cfg.rms_norm_eps)
            v_in = rms_norm(v_h, self.v_in_ln, cfg.rms_norm_eps)

        q = self.q_lin(q_in).reshape(B, T, cfg.num_attention_heads, cfg.head_dim)
        q = l2_clip(q * self.q_scale, cfg.qk_l2_clip_threshold)
        if not self.is_sliding:
            q = log_scale(q, position_ids, cfg.log_scaling_alpha, cfg.log_scaling_n_floor)
        q = q.transpose(0, 2, 1, 3)

        k_raw = self.k_lin(k_in)
        v_raw = self.v_lin(v_in)
        if self.is_sliding:
            k_raw = short_conv(k_raw, self.k_conv_w, cache, cache_slot, 0)
            v_raw = short_conv(v_raw, self.v_conv_w, cache, cache_slot, 1)
        else:
            k_raw = dynamic_short_conv(k_raw, k_in, self.k_dyn_st, self.k_dyn_pr, cfg.num_key_value_heads, cache, cache_slot, 0)
            v_raw = dynamic_short_conv(v_raw, v_in, self.v_dyn_st, self.v_dyn_pr, cfg.num_key_value_heads, cache, cache_slot, 1)

        k = l2_clip(k_raw.reshape(B, T, cfg.num_key_value_heads, cfg.head_dim), cfg.qk_l2_clip_threshold).transpose(0, 2, 1, 3)
        v = unweighted_rms_norm(v_raw.reshape(B, T, cfg.num_key_value_heads, cfg.head_dim), cfg.rms_norm_eps).transpose(0, 2, 1, 3)

        K_all, V_all = cache.update_kv(cache_slot, k, v, self.is_sliding)
        kv_len = K_all.shape[2]

        rel_st = self.rel_lin(q_in).reshape(B, T, cfg.num_attention_heads, cfg.d_rel)
        rel_st = l2_clip(rel_st * self.rel_scale, cfg.rel_l2_clip_threshold)
        if not self.is_sliding:
            rel_st = log_scale(rel_st, position_ids, cfg.log_scaling_alpha, cfg.log_scaling_n_floor)
        rel_logits = (rel_st @ self.rel_proj).transpose(0, 2, 1, 3)

        q_pos = position_ids[0]
        kv_end = cache.seq_len + T
        kv_start = kv_end - kv_len
        kv_pos = mx.arange(kv_start, kv_end, dtype=mx.int32)
        rel_dist = q_pos[:, None] - kv_pos[None, :]
        valid_rel = (rel_dist >= 0) & (rel_dist < cfg.rel_extent)
        g_idx = mx.clip(rel_dist, 0, cfg.rel_extent - 1)
        pos_bias = mx.where(valid_rel[None, None, :, :], rel_logits[:, :, mx.arange(T)[:, None], g_idx], 0.0)

        if self.is_sliding:
            allowed = (rel_dist >= 0) & (rel_dist < cfg.sliding_window_size)
        else:
            allowed = rel_dist >= 0
        sinks = (q_in @ self.sink_w.T).transpose(0, 2, 1)[..., None]
        n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
        attn_out = _gqa_sdpa_sinks(
            q, K_all, V_all, pos_bias, allowed, sinks,
            cfg.num_key_value_heads, n_rep, cfg.head_dim,
        )

        gate = mx.sigmoid(self.o_gate(q_in))
        attn_out = self.o_lin(attn_out * gate)
        attn_out = short_conv(attn_out, self.post_attn_conv_w, cache, cache_slot, 2)

        if self.is_block_boundary:
            local_blocks = block_states + [block_residual.astype(mx.float32)]
            residual = attn_out
            h_post = attn_out
        else:
            local_blocks = block_states
            residual = block_residual.astype(mx.float32) + attn_out
            h_post = residual

        h_mlp = block_attn_res(self.mlp_in_q, local_blocks, h_post)[0]
        if messages is not None:
            h_mlp = message_gate(h_mlp, messages[3], *self.msg_gates["mlp_message_gate"], cfg.rms_norm_eps)
        h_mlp = rms_norm(h_mlp, self.post_attn_ln, cfg.rms_norm_eps)

        mlp_out = self.moe(h_mlp)
        mlp_out = short_conv(mlp_out, self.post_mlp_conv_w, cache, cache_slot, 3)
        return residual + mlp_out


class Inkling2LoopPackedMLX:
    """Complete standalone Inkling-2-Mini text model running directly from packed artifacts on Apple Silicon."""

    def __init__(self, packed_dir: str | Path, steering_override: str | Path | None = None):
        self.packed_dir = Path(packed_dir)
        self.cfg = Inkling2Config.from_json(self.packed_dir / "config.json")
        self._fast = self.cfg.use_fast_metal
        lut_npz = np.load(self.packed_dir / "luts.npz")
        self.luts = {k: mx.array(lut_npz[k]) for k in lut_npz.files}
        self.fast_luts = build_fast_luts(self.luts) if self._fast else None

        st_raw = mx.load(str(self.packed_dir / "steering.safetensors"))
        if steering_override is not None:
            ov = mx.load(str(steering_override))
            st_raw.update(ov)
        self.st = st_raw

        self.embed = PackedEmbed(self.packed_dir, self.luts, self.st["model.embed_tokens.embed_norm.weight"], self.cfg.rms_norm_eps)
        self.final_in_q = self.st["model.final_input.query"].astype(mx.float32)
        self.final_norm_w = self.st["model.norm.weight"].astype(mx.float32)
        self.message_in_q = self.st["model.message_input.query"].astype(mx.float32)
        self.layers = [PackedDecoderLayer(i, self.cfg, self.packed_dir, self.luts, self.fast_luts, self.st) for i in range(self.cfg.num_hidden_layers)]
        self.lm_head = PackedLMHead(self.packed_dir, self.luts, self.fast_luts, self.cfg.final_logit_softcapping, self.cfg.hidden_size)

    def make_cache(self) -> Inkling2Cache:
        return Inkling2Cache(self.cfg)

    def _forward_core_fast(
        self,
        input_ids: list[int] | np.ndarray | mx.array,
        cache: Inkling2Cache,
        all_logits: bool = False,
        argmax_only: bool = False,
        ordered_only: bool = False,
        tap_visits: tuple[int, ...] = (),
        tree_spec: dict | None = None,
    ):
        cfg = self.cfg
        h = self.embed(input_ids)[0]  # [T, 2304]
        T = int(h.shape[0])
        q_pos0 = cache.seq_len
        blocks_cat: mx.array | None = None
        n_blocks = 0
        tap_set = set(tap_visits)
        captured: dict[int, mx.array] = {}
        visit = 0

        # 1. Pre-loop layers (L0..5)
        for i in range(cfg.loop_start):
            if visit in tap_set:
                captured[visit] = h
            L = self.layers[i]
            new_h = L.forward_fast(h, blocks_cat, n_blocks, q_pos0, T, cache, i, messages=None, tree_spec=tree_spec)
            if L.is_block_boundary:
                blocks_cat = h[None, :, :] if n_blocks == 0 else mx.concatenate([blocks_cat, h[None, :, :]], axis=0)
                n_blocks += 1
            h = new_h
            visit += 1
            if (visit & 3) == 0:
                mx.async_eval(h)

        pre_loop_h = h
        pre_loop_blocks = blocks_cat
        num_pre = n_blocks

        # For T >= 64 prefill, reuse dequantized FP16 QKV & O weights across the 3 loop passes of L6..17
        cache_loop_w = (T >= 64) and (cfg.loop_passes > 1)
        loop_w_cache: dict[int, tuple[mx.array, mx.array]] = {}

        # 2. Pass 0 over loop layers (L6..17)
        for i in range(cfg.loop_start, cfg.loop_end):
            if visit in tap_set:
                captured[visit] = h
            L = self.layers[i]
            if cache_loop_w:
                wq_c = L.qkv_stage.dequant_f16()
                wo_c = L.o_stage.dequant_f16()
                loop_w_cache[i] = (wq_c, wo_c)
            else:
                wq_c = wo_c = None
            new_h = L.forward_fast(h, blocks_cat, n_blocks, q_pos0, T, cache, i, messages=None, tree_spec=tree_spec, qkv_w_cached=wq_c, o_w_cached=wo_c)
            if L.is_block_boundary:
                blocks_cat = h[None, :, :] if n_blocks == 0 else mx.concatenate([blocks_cat, h[None, :, :]], axis=0)
                n_blocks += 1
            h = new_h
            visit += 1
            if (visit & 3) == 0:
                mx.async_eval(h)

        # 3. Passes 1..loop_passes-1 over loop layers with cross-pass messages
        num_loop = cfg.loop_end - cfg.loop_start
        for loop_idx in range(1, cfg.loop_passes):
            prev_n = n_blocks - (num_pre + 1)
            prev_blocks = blocks_cat[num_pre + 1 :] if prev_n > 0 else None
            prev_h = h
            h = pre_loop_h
            blocks_cat = pre_loop_blocks
            n_blocks = num_pre
            cache_offset = cfg.num_hidden_layers + (loop_idx - 1) * num_loop - cfg.loop_start
            for loop_layer_idx, i in enumerate(range(cfg.loop_start, cfg.loop_end)):
                if visit in tap_set:
                    captured[visit] = h
                L = self.layers[i]
                msgs = block_attn_res_fused(self.message_in_q[loop_layer_idx], prev_blocks, prev_n, prev_h)  # [4, T, 2304]
                wq_c, wo_c = loop_w_cache[i] if cache_loop_w else (None, None)
                new_h = L.forward_fast(h, blocks_cat, n_blocks, q_pos0, T, cache, i + cache_offset, messages=msgs, tree_spec=tree_spec, qkv_w_cached=wq_c, o_w_cached=wo_c)
                if L.is_block_boundary:
                    blocks_cat = h[None, :, :] if n_blocks == 0 else mx.concatenate([blocks_cat, h[None, :, :]], axis=0)
                    n_blocks += 1
                h = new_h
                visit += 1
                if (visit & 3) == 0:
                    mx.async_eval(h)
        if cache_loop_w:
            loop_w_cache.clear()

        # 4. Post-loop layers (L18..23)
        for i in range(cfg.loop_end, cfg.num_hidden_layers):
            if visit in tap_set:
                captured[visit] = h
            L = self.layers[i]
            new_h = L.forward_fast(h, blocks_cat, n_blocks, q_pos0, T, cache, i, messages=None, tree_spec=tree_spec)
            if L.is_block_boundary:
                blocks_cat = h[None, :, :] if n_blocks == 0 else mx.concatenate([blocks_cat, h[None, :, :]], axis=0)
                n_blocks += 1
            h = new_h
            visit += 1
            if (visit & 3) == 0 and visit < 48:
                mx.async_eval(h)

        # 5. Final AttnRes + norm + unpadded lm_head
        if not all_logits and visit not in tap_set and T > 1:
            b_last = blocks_cat[:, -1:, :] if (blocks_cat is not None and n_blocks > 0) else blocks_cat
            h_in = block_attn_res_fused(self.final_in_q, b_last, n_blocks, h[-1:, :], norm_w=self.final_norm_w)[0][None, :, :]
        else:
            h = block_attn_res_fused(self.final_in_q, blocks_cat, n_blocks, h, norm_w=self.final_norm_w)[0]  # [T, 2304]
            if visit in tap_set:
                captured[visit] = h
            h_in = h[None, :, :] if all_logits else h[-1:, :][None, :, :]
        cache.seq_len += T
        if argmax_only:
            out = self.lm_head.argmax(h_in)
        elif ordered_only:
            out = self.lm_head.ordered_logits(h_in)
        else:
            out = self.lm_head(h_in)
        if tap_visits:
            taps = mx.stack([captured[v] for v in tap_visits], axis=1)  # [T, len(taps), H]
            return out, cache, taps
        return out, cache

    def _forward_core(
        self,
        input_ids: list[int] | np.ndarray | mx.array,
        cache: Inkling2Cache | None = None,
        all_logits: bool = False,
        argmax_only: bool = False,
        ordered_only: bool = False,
        tap_visits: tuple[int, ...] = (),
        tree_spec: dict | None = None,
    ):
        if cache is None:
            cache = self.make_cache()
        if self._fast:
            return self._forward_core_fast(input_ids, cache, all_logits=all_logits, argmax_only=argmax_only, ordered_only=ordered_only, tap_visits=tap_visits, tree_spec=tree_spec)
        cfg = self.cfg
        h = self.embed(input_ids)
        T = h.shape[1]
        position_ids = mx.arange(cache.seq_len, cache.seq_len + T, dtype=mx.int32)[None, :]
        block_states: list[mx.array] = []
        tap_set = set(tap_visits)
        captured: dict[int, mx.array] = {}
        visit = 0

        for i in range(cfg.loop_start):
            if visit in tap_set:
                captured[visit] = h[0].astype(mx.float32)
            L = self.layers[i]
            new_h = L(h, block_states, position_ids, cache, i, messages=None)
            if L.is_block_boundary:
                block_states = block_states + [h.astype(mx.float32)]
            h = new_h
            visit += 1

        pre_loop_h = h
        pre_loop_blocks = list(block_states)
        num_pre = len(pre_loop_blocks)

        for i in range(cfg.loop_start, cfg.loop_end):
            if visit in tap_set:
                captured[visit] = h[0].astype(mx.float32)
            L = self.layers[i]
            new_h = L(h, block_states, position_ids, cache, i, messages=None)
            if L.is_block_boundary:
                block_states = block_states + [h.astype(mx.float32)]
            h = new_h
            visit += 1

        num_loop = cfg.loop_end - cfg.loop_start
        for loop_idx in range(1, cfg.loop_passes):
            prev_blocks = block_states[num_pre + 1 :]
            prev_h = h
            h = pre_loop_h
            block_states = list(pre_loop_blocks)
            cache_offset = cfg.num_hidden_layers + (loop_idx - 1) * num_loop - cfg.loop_start
            for loop_layer_idx, i in enumerate(range(cfg.loop_start, cfg.loop_end)):
                if visit in tap_set:
                    captured[visit] = h[0].astype(mx.float32)
                L = self.layers[i]
                msgs = block_attn_res(self.message_in_q[loop_layer_idx], prev_blocks, prev_h)
                new_h = L(h, block_states, position_ids, cache, i + cache_offset, messages=msgs)
                if L.is_block_boundary:
                    block_states = block_states + [h.astype(mx.float32)]
                h = new_h
                visit += 1

        for i in range(cfg.loop_end, cfg.num_hidden_layers):
            if visit in tap_set:
                captured[visit] = h[0].astype(mx.float32)
            L = self.layers[i]
            new_h = L(h, block_states, position_ids, cache, i, messages=None)
            if L.is_block_boundary:
                block_states = block_states + [h.astype(mx.float32)]
            h = new_h
            visit += 1

        h = block_attn_res(self.final_in_q, block_states, h)[0]
        h = rms_norm(h, self.final_norm_w, cfg.rms_norm_eps)
        if visit in tap_set:
            captured[visit] = h[0].astype(mx.float32)
        cache.seq_len += T
        h_in = h if all_logits else h[:, -1:, :]
        if argmax_only:
            out = self.lm_head.argmax(h_in)
        elif ordered_only:
            out = self.lm_head.ordered_logits(h_in)
        else:
            out = self.lm_head(h_in)
        if tap_visits:
            taps = mx.stack([captured[v] for v in tap_visits], axis=1)
            return out, cache, taps
        return out, cache

    def forward(self, input_ids: list[int] | np.ndarray | mx.array, cache: Inkling2Cache | None = None, tap_visits: tuple[int, ...] = ()):
        return self._forward_core(input_ids, cache=cache, all_logits=False, argmax_only=False, tap_visits=tap_visits)

    def forward_all_logits(self, input_ids: list[int] | np.ndarray | mx.array, cache: Inkling2Cache | None = None, tap_visits: tuple[int, ...] = (), tree_spec: dict | None = None):
        return self._forward_core(input_ids, cache=cache, all_logits=True, argmax_only=False, tap_visits=tap_visits, tree_spec=tree_spec)

    def forward_ordered_logits(self, input_ids: list[int] | np.ndarray | mx.array, cache: Inkling2Cache | None = None, tap_visits: tuple[int, ...] = ()):
        return self._forward_core(input_ids, cache=cache, all_logits=False, argmax_only=False, ordered_only=True, tap_visits=tap_visits)

    def forward_all_ordered_logits(self, input_ids: list[int] | np.ndarray | mx.array, cache: Inkling2Cache | None = None, tap_visits: tuple[int, ...] = (), tree_spec: dict | None = None):
        return self._forward_core(input_ids, cache=cache, all_logits=True, argmax_only=False, ordered_only=True, tap_visits=tap_visits, tree_spec=tree_spec)

    def forward_argmax(self, input_ids: list[int] | np.ndarray | mx.array, cache: Inkling2Cache | None = None, tap_visits: tuple[int, ...] = ()):
        return self._forward_core(input_ids, cache=cache, all_logits=False, argmax_only=True, tap_visits=tap_visits)

    def forward_all_argmax(self, input_ids: list[int] | np.ndarray | mx.array, cache: Inkling2Cache | None = None, tap_visits: tuple[int, ...] = (), tree_spec: dict | None = None):
        return self._forward_core(input_ids, cache=cache, all_logits=True, argmax_only=True, tap_visits=tap_visits, tree_spec=tree_spec)


class Inkling2MLXModel:
    """Thin `mlx_lm`-compatible adapter around `Inkling2LoopPackedMLX`."""

    def __init__(self, inner: Inkling2LoopPackedMLX):
        self.inner = inner
        self.cfg = inner.cfg
        self.layers = inner.layers

    def make_cache(self) -> list[Inkling2Cache]:
        return [self.inner.make_cache()]

    def __call__(self, inputs: mx.array, cache: list[Inkling2Cache] | Inkling2Cache | None = None) -> mx.array:
        c = cache[0] if isinstance(cache, list) and cache else cache
        logits, _ = self.inner.forward(inputs, cache=c)
        return logits


def load_inkling(packed_dir: str | Path, steering_override: str | Path | None = None):
    """Load packed Inkling-2-Mini as an `(mlx_lm-compatible model, tokenizer)` pair."""
    from mlx_lm.utils import load_tokenizer

    path = Path(packed_dir)
    inner = Inkling2LoopPackedMLX(path, steering_override=steering_override)
    tok = load_tokenizer(path)
    return Inkling2MLXModel(inner), tok
