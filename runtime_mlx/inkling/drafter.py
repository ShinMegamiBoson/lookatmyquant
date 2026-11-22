"""Official Inkling-2-Mini 3-layer DSpark MTP Drafter in Apple Silicon MLX (BF16 or 4-bit ~170 MB).

Matches SGLang's `Inkling2ForConditionalGenerationMTP` (`inkling2_mtp.py` + `inkling2_mtp_ops.py`) line-for-line:
  - Reads 4 target hidden taps (`target_layer_ids = [5, 23, 41, 48]`: inputs to logical visits 5, 23, 41 + final norm state 48).
  - Mixes the 4 taps per layer (`fuse_alpha = softmax(fuse_weight)` row `i`) via `mix_rmsnorm` with `layer.dflare_norm`.
  - Appends projected context KV (`k_proj_ctx`, `v_proj_ctx`, `k_norm_ctx`, RoPE `theta=1e6`) in `float16` to each of the 3 DSpark layers.
  - Drafts a block of 8 tokens (`[anchor_embed, mask_embed x 7]`) in parallel through the 3 bidirectional-block layers
    using native `mx.fast.scaled_dot_product_attention` over `[ctx_kv, self_block_kv]`.
  - Runs the sequential `Inkling2MTPHead` (`rnn_hidden`, rank 256) over steps `s = 1..draft_len-1` with ZERO CPU↔GPU syncs,
    optionally scoring against the target's lazy 32K MLX-4bit draft shortlist head (verified against the target's full 200K head).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .model import rms_norm


def quantize_mtp_to_4bit(src_safetensors: str | Path, dst_safetensors: str | Path, group_size: int = 64, bits: int = 4) -> dict:
    """Quantize official BF16 `mtp.safetensors` (635.8 MB) to 4-bit MLX (~169 MB) in ~1 second on M5 Max."""
    src_w = mx.load(str(src_safetensors))
    out: dict[str, mx.array] = {}
    q_keys = []
    for k, v in src_w.items():
        if v.ndim == 2 and v.shape[1] % group_size == 0 and v.nbytes >= 1_000_000:
            wq, sc, bi = mx.quantize(v, group_size=group_size, bits=bits)
            out[k] = wq
            out[k + ".scales"] = sc
            out[k + ".biases"] = bi
            q_keys.append(k)
        else:
            out[k] = v
    mx.eval(out)
    dst = Path(dst_safetensors)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst), out)
    raw_mb = sum(v.nbytes for v in src_w.values()) / 1e6
    q_mb = sum(v.nbytes for v in out.values()) / 1e6
    return {
        "src_mb": round(raw_mb, 2),
        "dst_mb": round(q_mb, 2),
        "compression": round(raw_mb / q_mb, 2),
        "quantized_matrices": len(q_keys),
        "bits": bits,
        "group_size": group_size,
    }


@mx.compile
def _apply_rope_compiled(x: mx.array, positions: mx.array, inv_freq: mx.array) -> mx.array:
    freqs = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    cos = mx.cos(freqs)[None, None, :, :]
    sin = mx.sin(freqs)[None, None, :, :]
    xf = x.astype(mx.float32)
    half = xf.shape[-1] // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


@mx.compile
def _mix_rmsnorm_compiled(targets: mx.array, alpha: mx.array, weight: mx.array) -> mx.array:
    mixed = mx.sum(targets.astype(mx.float32) * alpha[None, :, None], axis=1)
    return mx.fast.rms_norm(mixed, weight, 1e-6)


@mx.compile
def _rnn_update_compiled(state: mx.array, proj: mx.array) -> tuple[mx.array, mx.array]:
    gate_raw, cand_raw, out_raw = mx.split(proj, 3, axis=-1)
    gate = mx.sigmoid(gate_raw)
    new_state = gate * state + (1.0 - gate) * mx.tanh(cand_raw)
    latent = mx.tanh(out_raw)
    return new_state, latent


TREE_TOPOLOGIES: dict[str, tuple[int, ...]] = {
    "2-1": (-1, 0, 1, 0),                   # M4 micro-tree T=4: depth-2 A + depth-1 B (2 head calls)
    "3-1": (-1, 0, 1, 2, 0),                # M4 micro-tree T=5: depth-3 A + depth-1 B (3 head calls)
    "2-2": (-1, 0, 1, 0, 3),                # M4 micro-tree T=5: depth-2 A + depth-2 B (2 batched head calls)
    "3-2": (-1, 0, 1, 2, 0, 4),             # M4 micro-tree T=6: depth-3 A + depth-2 B (3 batched head calls)
    "3-1-1": (-1, 0, 1, 2, 0, 0),           # M4 micro-tree T=6: depth-3 A + top-2 & top-3 at depth 1 (3 head calls)
    "4-3": (-1, 0, 1, 2, 3, 0, 5, 6),       # depth-4 A + depth-3 B (4 head calls)
    "5-2": (-1, 0, 1, 2, 3, 4, 0, 6),       # depth-5 A + depth-2 B (5 head calls)
    "4-2-1": (-1, 0, 1, 2, 3, 0, 5, 0),     # depth-4 A + depth-2 B + depth-1 C (4 head calls)
    "6-1": (-1, 0, 1, 2, 3, 4, 5, 0),       # depth-6 A + depth-1 B (6 head calls)
    "5-1-1": (-1, 0, 1, 2, 3, 4, 0, 0),     # depth-5 A + top-2 & top-3 at depth 1 (5 head calls)
    "4-2-2a": (-1, 0, 1, 2, 3, 1, 5, 0),    # depth-4 A + depth-2 alt branch at a1 + depth-1 B (4 head calls)
}


@mx.compile
def _silu_mul_compiled(gate_up: mx.array) -> mx.array:
    g, u = mx.split(gate_up, 2, axis=-1)
    return nn.silu(g) * u


class _QLinear:
    def __init__(self, weights: dict[str, mx.array] | None, key: str = "", bits: int = 4, group_size: int = 64):
        if weights is None:
            return
        self.w = weights[key]
        self.quantized = (key + ".scales") in weights
        if self.quantized:
            self.scales = weights[key + ".scales"]
            self.biases = weights[key + ".biases"]
            self.bits = bits
            self.group_size = group_size
        else:
            self.w = self.w.astype(mx.float32)

    @classmethod
    def concat(cls, linears: list["_QLinear"]) -> "_QLinear":
        """Fuse multiple `_QLinear` projections sharing the same input into ONE quantized/dense matmul."""
        out = cls(None)
        out.quantized = all(lin.quantized for lin in linears)
        if out.quantized:
            out.w = mx.concatenate([lin.w for lin in linears], axis=0)
            out.scales = mx.concatenate([lin.scales for lin in linears], axis=0)
            out.biases = mx.concatenate([lin.biases for lin in linears], axis=0)
            out.bits = linears[0].bits
            out.group_size = linears[0].group_size
        else:
            out.w = mx.concatenate([lin.w.astype(mx.float32) for lin in linears], axis=0)
        return out

    def __call__(self, x: mx.array, out_dtype=mx.float32) -> mx.array:
        if self.quantized:
            y = mx.quantized_matmul(
                x.astype(mx.bfloat16) if x.dtype != mx.bfloat16 else x,
                self.w, self.scales, self.biases,
                transpose=True, group_size=self.group_size, bits=self.bits,
            )
            return y if y.dtype == out_dtype else y.astype(out_dtype)
        y = x.astype(mx.float32) @ self.w.T
        return y if y.dtype == out_dtype else y.astype(out_dtype)


class _QEmbedding:
    def __init__(self, weights: dict[str, mx.array], key: str, bits: int = 4, group_size: int = 64):
        self.w = weights[key]
        self.quantized = (key + ".scales") in weights
        if self.quantized:
            self.scales = weights[key + ".scales"]
            self.biases = weights[key + ".biases"]
            self.bits = bits
            self.group_size = group_size

    def __call__(self, idx: mx.array) -> mx.array:
        if self.quantized:
            return mx.dequantize(
                self.w[idx], self.scales[idx], self.biases[idx],
                group_size=self.group_size, bits=self.bits,
            ).astype(mx.float32)
        return self.w[idx].astype(mx.float32)


class DSparkDrafterMLX:
    """3-layer DSpark MTP block-8 drafter matching `thinkingmachines/Inkling-2-Mini` (`mtp.safetensors`)."""

    def __init__(self, mtp_path: str | Path, config_path: str | Path, bits: int = 4, group_size: int = 64):
        raw_cfg = json.loads(Path(config_path).read_text())
        mc = raw_cfg["mtp_config"]
        tc = raw_cfg["text_config"]
        self.hidden_size = int(mc["hidden_size"])
        self.num_layers = int(mc["num_hidden_layers"])
        self.num_heads = int(mc["num_attention_heads"])
        self.num_kv_heads = int(mc["num_key_value_heads"])
        self.head_dim = int(mc["head_dim"])
        self.n_rep = self.num_heads // self.num_kv_heads
        self.eps = float(mc["rms_norm_eps"])
        self.rope_theta = float(mc["rope_theta"])
        self.local_layers = set(mc["local_layer_ids"])
        self.window = int(mc["sliding_window_size"])
        self.block_size = int(mc["block_size"])
        self.markov_rank = int(mc["markov_rank"])
        self.target_layer_ids = tuple(mc["target_layer_ids"])
        self.softcap = float(tc["final_logit_softcapping"])
        self._sdpa_scale = self.head_dim ** -0.5

        half = self.head_dim // 2
        self._inv_freq = mx.exp(-math.log(self.rope_theta) * mx.arange(0, half, dtype=mx.float32) / float(half))

        w = mx.load(str(mtp_path))
        self.fuse_alpha = mx.softmax(w["fuse_weight"].astype(mx.float32), axis=-1)
        self.mask_embed = w["mask_embed"].astype(mx.bfloat16)
        self.final_norm = w["norm.weight"].astype(mx.float32)

        self.layers = []
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self._q_dim = q_dim
        self._kv_dim = kv_dim
        for i in range(self.num_layers):
            p = f"layers.{i}."
            q_lin = _QLinear(w, p + "self_attn.q_proj.weight", bits, group_size)
            k_lin = _QLinear(w, p + "self_attn.k_proj.weight", bits, group_size)
            v_lin = _QLinear(w, p + "self_attn.v_proj.weight", bits, group_size)
            k_ctx = _QLinear(w, p + "self_attn.k_proj_ctx.weight", bits, group_size)
            v_ctx = _QLinear(w, p + "self_attn.v_proj_ctx.weight", bits, group_size)
            gate_lin = _QLinear(w, p + "mlp.gate_proj.weight", bits, group_size)
            up_lin = _QLinear(w, p + "mlp.up_proj.weight", bits, group_size)
            self.layers.append({
                "dflare_norm": w[p + "dflare_norm.weight"].astype(mx.float32),
                "in_ln": w[p + "input_layernorm.weight"].astype(mx.bfloat16),
                "post_ln": w[p + "post_attention_layernorm.weight"].astype(mx.bfloat16),
                "q_norm": w[p + "self_attn.q_norm.weight"].astype(mx.bfloat16),
                "k_norm": w[p + "self_attn.k_norm.weight"].astype(mx.bfloat16),
                "k_norm_ctx": w[p + "self_attn.k_norm_ctx.weight"].astype(mx.bfloat16),
                "qkv_proj": _QLinear.concat([q_lin, k_lin, v_lin]),
                "kv_ctx_proj": _QLinear.concat([k_ctx, v_ctx]),
                "o_proj": _QLinear(w, p + "self_attn.o_proj.weight", bits, group_size),
                "gate_up_proj": _QLinear.concat([gate_lin, up_lin]),
                "down_proj": _QLinear(w, p + "mlp.down_proj.weight", bits, group_size),
            })

        self.markov_w1 = _QEmbedding(w, "markov_head.markov_w1.weight", bits, group_size)
        self.rnn_proj_w = _QLinear(w, "markov_head.rnn_proj.weight", bits, group_size)
        self.rnn_proj_b = w["markov_head.rnn_proj.bias"].astype(mx.float32)
        self.hidden_out_w = _QLinear(w, "markov_head.hidden_out.weight", bits, group_size)
        self.hidden_out_b = w["markov_head.hidden_out.bias"].astype(mx.float32)
        self.head_dflare_norm = w["markov_head.dflare_norm.weight"].astype(mx.float32)
        self._zero_state = mx.zeros((self.markov_rank,), dtype=mx.float32)
        eval_items = [self._inv_freq, self.fuse_alpha, self.mask_embed, self.final_norm, self._zero_state]
        for L in self.layers:
            for k_name in ("qkv_proj", "kv_ctx_proj", "gate_up_proj"):
                lin = L[k_name]
                eval_items.append(lin.w)
                if lin.quantized:
                    eval_items.extend([lin.scales, lin.biases])
        mx.eval(*eval_items)

        self.reset()

    def reset(self):
        self.ctx_k: list[mx.array | None] = [None] * self.num_layers
        self.ctx_v: list[mx.array | None] = [None] * self.num_layers
        self._ctx_cur: list[int] = [0] * self.num_layers
        self._ctx_cap: list[int] = [0] * self.num_layers
        self.ctx_len: int = 0
        self.f_rnn: mx.array | None = None

    def _ensure_ctx_cap(self, i: int, need: int):
        if self.ctx_k[i] is None or need > self._ctx_cap[i]:
            new_cap = ((need + 255) // 256) * 256
            K_new = mx.zeros((1, self.num_kv_heads, new_cap, self.head_dim), dtype=mx.float16)
            V_new = mx.zeros((1, self.num_kv_heads, new_cap, self.head_dim), dtype=mx.float16)
            cur = self._ctx_cur[i]
            if self.ctx_k[i] is not None and cur > 0:
                K_new[:, :, :cur, :] = self.ctx_k[i][:, :, :cur, :]
                V_new[:, :, :cur, :] = self.ctx_v[i][:, :, :cur, :]
            self.ctx_k[i] = K_new
            self.ctx_v[i] = V_new
            self._ctx_cap[i] = new_cap

    def push_context(self, target_taps: mx.array, start_pos: int):
        """Ingest committed target hidden taps `[T, 4, H]` starting at `start_pos`."""
        T = int(target_taps.shape[0])
        kv_dim = self._kv_dim
        for i, L in enumerate(self.layers):
            ctx = _mix_rmsnorm_compiled(target_taps, self.fuse_alpha[i], L["dflare_norm"])
            kv = L["kv_ctx_proj"](ctx, out_dtype=mx.bfloat16)
            k = kv[:, :kv_dim].reshape(1, T, self.num_kv_heads, self.head_dim)
            v = kv[:, kv_dim:].reshape(1, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3).astype(mx.float16)
            k = mx.fast.rms_norm(k, L["k_norm_ctx"], self.eps).transpose(0, 2, 1, 3)
            k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=start_pos).astype(mx.float16)

            cur = self._ctx_cur[i]
            need = cur + T + self.block_size
            if i in self.local_layers and cur + T > self.window + 256 and self.ctx_k[i] is not None:
                shift = cur - self.window
                K_old, V_old = self.ctx_k[i], self.ctx_v[i]
                new_cap = max(self._ctx_cap[i], ((self.window + T + self.block_size + 255) // 256) * 256)
                K_new = mx.zeros((1, self.num_kv_heads, new_cap, self.head_dim), dtype=mx.float16)
                V_new = mx.zeros((1, self.num_kv_heads, new_cap, self.head_dim), dtype=mx.float16)
                K_new[:, :, : self.window, :] = K_old[:, :, shift:cur, :]
                V_new[:, :, : self.window, :] = V_old[:, :, shift:cur, :]
                self.ctx_k[i], self.ctx_v[i], self._ctx_cap[i] = K_new, V_new, new_cap
                cur = self.window
                need = cur + T + self.block_size
            self._ensure_ctx_cap(i, need)
            self.ctx_k[i][:, :, cur : cur + T, :] = k
            self.ctx_v[i][:, :, cur : cur + T, :] = v
            self._ctx_cur[i] = cur + T
        self.ctx_len = start_pos + T
        self.f_rnn = _mix_rmsnorm_compiled(target_taps[-1:], self.fuse_alpha[self.num_layers], self.head_dflare_norm)[0]

    def _run_3layers(self, anc_emb: mx.array, n_toks: int | None = None) -> mx.array:
        B_len = self.block_size if n_toks is None else max(2, min(int(n_toks), self.block_size))
        masks = mx.broadcast_to(self.mask_embed[None, :], (B_len - 1, self.hidden_size))
        h = mx.concatenate([anc_emb.astype(mx.bfloat16)[None, :], masks], axis=0)
        q_dim = self._q_dim
        qk_dim = q_dim + self._kv_dim
        off = self.ctx_len

        for i, L in enumerate(self.layers):
            res = h
            xn = mx.fast.rms_norm(h, L["in_ln"], self.eps)
            qkv = L["qkv_proj"](xn, out_dtype=mx.bfloat16)
            q = qkv[:, :q_dim].reshape(1, B_len, self.num_heads, self.head_dim)
            k = qkv[:, q_dim:qk_dim].reshape(1, B_len, self.num_kv_heads, self.head_dim)
            v = qkv[:, qk_dim:].reshape(1, B_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3).astype(mx.float16)
            q = mx.fast.rope(mx.fast.rms_norm(q, L["q_norm"], self.eps).transpose(0, 2, 1, 3), self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=off).astype(mx.float16)
            k = mx.fast.rope(mx.fast.rms_norm(k, L["k_norm"], self.eps).transpose(0, 2, 1, 3), self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=off).astype(mx.float16)

            cur = self._ctx_cur[i]
            self._ensure_ctx_cap(i, cur + B_len)
            self.ctx_k[i][:, :, cur : cur + B_len, :] = k
            self.ctx_v[i][:, :, cur : cur + B_len, :] = v
            lo = max(0, cur - self.window) if i in self.local_layers else 0
            K_all = self.ctx_k[i][:, :, lo : cur + B_len, :]
            V_all = self.ctx_v[i][:, :, lo : cur + B_len, :]
            attn_out = mx.fast.scaled_dot_product_attention(q, K_all, V_all, scale=self._sdpa_scale)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B_len, self.num_heads * self.head_dim).astype(mx.bfloat16)
            h = res + L["o_proj"](attn_out, out_dtype=mx.bfloat16)

            res2 = h
            mn = mx.fast.rms_norm(h, L["post_ln"], self.eps)
            h = res2 + L["down_proj"](_silu_mul_compiled(L["gate_up_proj"](mn, out_dtype=mx.bfloat16)), out_dtype=mx.bfloat16)

        return rms_norm(h, self.final_norm, self.eps)

    def _markov_step(self, state: mx.array, prev_tok: mx.array, h_s: mx.array) -> tuple[mx.array, mx.array]:
        """Batched DSpark Markov RNN step: `state [B, 256]`, `prev_tok [B]` -> `(new_state [B, 256], h_pred [B, 2304])`."""
        B = int(prev_tok.shape[0])
        prev_emb = self.markov_w1(prev_tok)  # [B, 256]
        h_b = mx.broadcast_to(h_s[None, :], (B, self.hidden_size))
        f_b = mx.broadcast_to(self.f_rnn[None, :], (B, self.hidden_size))
        inp = mx.concatenate([state, prev_emb, h_b, f_b], axis=-1)
        proj = self.rnn_proj_w(inp) + self.rnn_proj_b[None, :]
        new_state, latent = _rnn_update_compiled(state, proj)
        delta_h = self.hidden_out_w(latent) + self.hidden_out_b[None, :]
        return new_state, h_b + delta_h

    def propose_gpu(self, anchor_tok: int | mx.array, embed_fn, head_argmax_fn, draft_len: int | None = None) -> mx.array:
        """Zero-sync GPU block proposal returning `int32[draft_len]` (`[anchor, d1, ..., d_{draft_len-1}]`)."""
        n_toks = self.block_size if draft_len is None else max(2, min(int(draft_len), self.block_size))
        anc_mx = anchor_tok.reshape(1).astype(mx.int32) if isinstance(anchor_tok, mx.array) else mx.array([int(anchor_tok)], dtype=mx.int32)
        anc_emb = embed_fn(anc_mx)[0, 0, :]
        h = self._run_3layers(anc_emb, n_toks=(n_toks if __import__("os").environ.get("LAMQ_DRAFTER_TRIM_BLEN") == "1" else None))

        state = self._zero_state
        prev_tok = anc_mx
        toks = [anc_mx]
        for s in range(1, n_toks):
            prev_emb = self.markov_w1(prev_tok)[0]
            inp = mx.concatenate([state, prev_emb, h[s], self.f_rnn], axis=0)[None, :]
            proj = (self.rnn_proj_w(inp) + self.rnn_proj_b)[0]
            state, latent = _rnn_update_compiled(state, proj)
            delta_h = self.hidden_out_w(latent[None, :])[0] + self.hidden_out_b
            nxt = head_argmax_fn(h[s] + delta_h).reshape(1).astype(mx.int32)
            toks.append(nxt)
            prev_tok = nxt
        return mx.concatenate(toks, axis=0)

    def propose_sample_gpu(
        self,
        anchor_tok: int | mx.array,
        embed_fn,
        head_sample_fn,
        draft_len: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Zero-sync stochastic GPU block proposal for exact speculative sampling (`temp > 0`).

        Calls `head_sample_fn(h_1d) -> (vocab_tok_i32, ord_tok_i32, probs_q_f32[S])` at each Markov RNN step
        and returns `(draft_ids: int32[n_toks], draft_ord: int32[n_toks - 1], q_probs: float32[n_toks - 1, S])`.
        """
        n_toks = self.block_size if draft_len is None else max(2, min(int(draft_len), self.block_size))
        anc_mx = (
            anchor_tok.reshape(1).astype(mx.int32)
            if isinstance(anchor_tok, mx.array)
            else mx.array([int(anchor_tok)], dtype=mx.int32)
        )
        anc_emb = embed_fn(anc_mx)[0, 0, :]
        h = self._run_3layers(anc_emb)

        state = self._zero_state
        prev_tok = anc_mx
        toks = [anc_mx]
        ords: list[mx.array] = []
        probs: list[mx.array] = []
        for s in range(1, n_toks):
            prev_emb = self.markov_w1(prev_tok)[0]
            inp = mx.concatenate([state, prev_emb, h[s], self.f_rnn], axis=0)[None, :]
            proj = (self.rnn_proj_w(inp) + self.rnn_proj_b)[0]
            state, latent = _rnn_update_compiled(state, proj)
            delta_h = self.hidden_out_w(latent[None, :])[0] + self.hidden_out_b
            vocab_tok, ord_tok, q_prob = head_sample_fn(h[s] + delta_h)
            nxt = vocab_tok.reshape(1).astype(mx.int32)
            toks.append(nxt)
            ords.append(ord_tok.reshape(1).astype(mx.int32))
            probs.append(q_prob.astype(mx.float32))
            prev_tok = nxt
        return (
            mx.concatenate(toks, axis=0),
            mx.concatenate(ords, axis=0),
            mx.stack(probs, axis=0),
        )

    def propose_tree8_gpu(
        self,
        anchor_tok: int | mx.array,
        embed_fn,
        head_topk_fn,
        topology: str = "4-3",
    ) -> mx.array:
        """Zero-sync GPU 8-node DSpark tree proposal (4–6 batched `draft_head` calls, matching `TREE_TOPOLOGIES[topology]`)."""
        if topology not in TREE_TOPOLOGIES:
            raise ValueError(f"Unknown tree topology {topology!r}; expected one of {tuple(TREE_TOPOLOGIES)}")
        anc_mx = anchor_tok.reshape(1).astype(mx.int32) if isinstance(anchor_tok, mx.array) else mx.array([int(anchor_tok)], dtype=mx.int32)
        anc_emb = embed_fn(anc_mx)[0, 0, :]
        h = self._run_3layers(anc_emb)
        z0 = self._zero_state[None, :]  # [1, 256]

        if topology == "2-1":
            st1, hp1 = self._markov_step(z0, anc_mx, h[1])
            top2 = head_topk_fn(hp1, 2)[0]  # [a1, b1]
            _, hp2 = self._markov_step(st1, top2[:1], h[2])
            a2 = head_topk_fn(hp2, 1)[0, :1]
            return mx.concatenate([anc_mx, top2[:1], a2, top2[1:2]], axis=0)

        if topology == "3-1":
            st1, hp1 = self._markov_step(z0, anc_mx, h[1])
            top2 = head_topk_fn(hp1, 2)[0]  # [a1, b1]
            st2, hp2 = self._markov_step(st1, top2[:1], h[2])
            a2 = head_topk_fn(hp2, 1)[0, :1]
            _, hp3 = self._markov_step(st2, a2, h[3])
            a3 = head_topk_fn(hp3, 1)[0, :1]
            return mx.concatenate([anc_mx, top2[:1], a2, a3, top2[1:2]], axis=0)

        if topology == "2-2":
            st1, hp1 = self._markov_step(z0, anc_mx, h[1])
            top2 = head_topk_fn(hp1, 2)[0]  # [a1, b1]
            _, hp2 = self._markov_step(mx.concatenate([st1, st1], axis=0), top2, h[2])
            t2 = head_topk_fn(hp2, 1)[:, 0]  # [a2, b2]
            return mx.concatenate([anc_mx, top2[:1], t2[:1], top2[1:2], t2[1:2]], axis=0)

        if topology == "3-2":
            st1, hp1 = self._markov_step(z0, anc_mx, h[1])
            top2 = head_topk_fn(hp1, 2)[0]  # [a1, b1]
            st2, hp2 = self._markov_step(mx.concatenate([st1, st1], axis=0), top2, h[2])
            t2 = head_topk_fn(hp2, 1)[:, 0]  # [a2, b2]
            _, hp3 = self._markov_step(st2[:1], t2[:1], h[3])
            a3 = head_topk_fn(hp3, 1)[0, :1]
            return mx.concatenate([anc_mx, top2[:1], t2[:1], a3, top2[1:2], t2[1:2]], axis=0)

        if topology == "3-1-1":
            st1, hp1 = self._markov_step(z0, anc_mx, h[1])
            top3 = head_topk_fn(hp1, 3)[0]  # [a1, b1, c1]
            st2, hp2 = self._markov_step(st1, top3[:1], h[2])
            a2 = head_topk_fn(hp2, 1)[0, :1]
            _, hp3 = self._markov_step(st2, a2, h[3])
            a3 = head_topk_fn(hp3, 1)[0, :1]
            return mx.concatenate([anc_mx, top3[:1], a2, a3, top3[1:2], top3[2:3]], axis=0)

        if topology == "4-3":
            st1, hp1 = self._markov_step(z0, anc_mx, h[1])
            top2 = head_topk_fn(hp1, 2)[0]  # [2] = [a1, b1]
            st2, hp2 = self._markov_step(mx.concatenate([st1, st1], axis=0), top2, h[2])
            t2 = head_topk_fn(hp2, 1)[:, 0]  # [2] = [a2, b2]
            st3, hp3 = self._markov_step(st2, t2, h[3])
            t3 = head_topk_fn(hp3, 1)[:, 0]  # [2] = [a3, b3]
            _, hp4 = self._markov_step(st3[:1], t3[:1], h[4])
            a4 = head_topk_fn(hp4, 1)[0, :1]
            return mx.concatenate([anc_mx, top2[:1], t2[:1], t3[:1], a4, top2[1:2], t2[1:2], t3[1:2]], axis=0)

        if topology == "5-2":
            st1, hp1 = self._markov_step(z0, anc_mx, h[1])
            top2 = head_topk_fn(hp1, 2)[0]  # [a1, b1]
            st2, hp2 = self._markov_step(mx.concatenate([st1, st1], axis=0), top2, h[2])
            t2 = head_topk_fn(hp2, 1)[:, 0]  # [a2, b2]
            st3, hp3 = self._markov_step(st2[:1], t2[:1], h[3])
            a3 = head_topk_fn(hp3, 1)[0, :1]
            st4, hp4 = self._markov_step(st3, a3, h[4])
            a4 = head_topk_fn(hp4, 1)[0, :1]
            _, hp5 = self._markov_step(st4, a4, h[5])
            a5 = head_topk_fn(hp5, 1)[0, :1]
            return mx.concatenate([anc_mx, top2[:1], t2[:1], a3, a4, a5, top2[1:2], t2[1:2]], axis=0)

        if topology == "4-2-1":
            st1, hp1 = self._markov_step(z0, anc_mx, h[1])
            top3 = head_topk_fn(hp1, 3)[0]  # [a1, b1, c1]
            st2, hp2 = self._markov_step(mx.concatenate([st1, st1], axis=0), top3[:2], h[2])
            t2 = head_topk_fn(hp2, 1)[:, 0]  # [a2, b2]
            st3, hp3 = self._markov_step(st2[:1], t2[:1], h[3])
            a3 = head_topk_fn(hp3, 1)[0, :1]
            _, hp4 = self._markov_step(st3, a3, h[4])
            a4 = head_topk_fn(hp4, 1)[0, :1]
            return mx.concatenate([anc_mx, top3[:1], t2[:1], a3, a4, top3[1:2], t2[1:2], top3[2:3]], axis=0)

        if topology == "6-1":
            st1, hp1 = self._markov_step(z0, anc_mx, h[1])
            top2 = head_topk_fn(hp1, 2)[0]  # [a1, b1]
            cur_st, cur_tok = st1, top2[:1]
            toks = [anc_mx, top2[:1]]
            for s in range(2, 7):
                cur_st, hp = self._markov_step(cur_st, cur_tok, h[s])
                cur_tok = head_topk_fn(hp, 1)[0, :1]
                toks.append(cur_tok)
            toks.append(top2[1:2])
            return mx.concatenate(toks, axis=0)

        if topology == "5-1-1":
            st1, hp1 = self._markov_step(z0, anc_mx, h[1])
            top3 = head_topk_fn(hp1, 3)[0]  # [a1, b1, c1]
            cur_st, cur_tok = st1, top3[:1]
            toks = [anc_mx, top3[:1]]
            for s in range(2, 6):
                cur_st, hp = self._markov_step(cur_st, cur_tok, h[s])
                cur_tok = head_topk_fn(hp, 1)[0, :1]
                toks.append(cur_tok)
            toks.extend([top3[1:2], top3[2:3]])
            return mx.concatenate(toks, axis=0)

        # "4-2-2a": (-1, 0, 1, 2, 3, 1, 5, 0) -> [anc, a1, a2, a3, a4, a2_alt, a3_alt, b1]
        st1, hp1 = self._markov_step(z0, anc_mx, h[1])
        top2_d1 = head_topk_fn(hp1, 2)[0]  # [a1, b1]
        st2, hp2 = self._markov_step(st1, top2_d1[:1], h[2])
        top2_d2 = head_topk_fn(hp2, 2)[0]  # [a2, a2_alt]
        st3, hp3 = self._markov_step(mx.concatenate([st2, st2], axis=0), top2_d2, h[3])
        t3 = head_topk_fn(hp3, 1)[:, 0]  # [a3, a3_alt]
        _, hp4 = self._markov_step(st3[:1], t3[:1], h[4])
        a4 = head_topk_fn(hp4, 1)[0, :1]
        return mx.concatenate([anc_mx, top2_d1[:1], top2_d2[:1], t3[:1], a4, top2_d2[1:2], t3[1:2], top2_d1[1:2]], axis=0)

    def propose(self, anchor_tok: int, embed_fn, lm_head_fn) -> list[int]:
        """Propose a block of `block_size` tokens `[anchor_tok, d1, ..., d7]`."""
        if hasattr(lm_head_fn, "get_draft_head_argmax"):
            head_argmax_fn = lm_head_fn.get_draft_head_argmax()
        elif hasattr(lm_head_fn, "argmax"):
            head_argmax_fn = lambda h1: lm_head_fn.argmax(h1[None, None, :])[0, 0]
        else:
            head_argmax_fn = lambda h1: mx.argmax(lm_head_fn(h1[None, None, :])[0, 0])
        draft_gpu = self.propose_gpu(anchor_tok, embed_fn, head_argmax_fn)
        return np.array(draft_gpu, dtype=np.int32).tolist()


def resolve_drafter_files(path: str | Path, fallback_config: str | Path | None = None) -> tuple[Path, Path]:
    """Resolve a drafter `.safetensors` file or Hugging Face / local directory into `(weights_path, config_path)`."""
    p = Path(path)
    if p.is_file():
        cfg = p.parent / "config.json"
        if not cfg.exists() and fallback_config is not None:
            cfg = Path(fallback_config)
        if not cfg.exists():
            raise FileNotFoundError(f"config.json not found beside {p}")
        return p, cfg
    if p.is_dir():
        manifest = p / "manifest.json"
        weights = None
        if manifest.exists():
            m = json.loads(manifest.read_text())
            if m.get("weights") and (p / m["weights"]).exists():
                weights = p / m["weights"]
        if weights is None:
            for cand in ("mtp_4bit.safetensors", "mtp.safetensors"):
                if (p / cand).exists():
                    weights = p / cand
                    break
        if weights is None:
            raise FileNotFoundError(f"no mtp_4bit.safetensors or mtp.safetensors found in {p}")
        cfg = p / "config.json"
        if not cfg.exists() and fallback_config is not None:
            cfg = Path(fallback_config)
        if not cfg.exists():
            raise FileNotFoundError(f"config.json not found in {p}")
        return weights, cfg
    raise FileNotFoundError(f"drafter path does not exist: {p}")
