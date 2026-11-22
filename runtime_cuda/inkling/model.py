"""Standalone PyTorch + Triton implementation of bit-packed Inkling-2-Mini (Inkling2Loop) for NVIDIA GPUs.

Follows `runtime_mlx/inkling/model.py` (which matches transformers' `modeling_inkling2_loop.py`):
  - 24 physical layers with L6..17 looped 3x (48 logical visits), one KV + 4 conv-state cache slot per visit
  - Block attention residuals (`attention_input`, `mlp_input`, `final_input`) and cross-pass `message_input`
  - Loop message gates (`{q,k,v,mlp}_message_gate`) on passes 1 and 2
  - Static and dynamic causal 4-tap convolutions, L2 clips, log-scaled queries on full-attention layers
  - Relative position bias + per-token sinks, gated attention output, sigmoid top-8 router, softcapped SwiGLU
Packed matrices run through the Triton fused decode+matmul kernels in `triton_kernels.py`.

The cache is pre-allocated and the decode step for one token touches no Python-side state that changes between
steps (position lives on the device), so `generate.py` can capture a step in a CUDA graph and replay it.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F
from safetensors.torch import load_file

from .attention import decode_attention, prefill_attention
from .conv import conv4_, dyn_conv4_, kv_conv_write_
from .fused import block_attn_res_fused, message_gates_fused
from .packed_modules import FMT_ID, FastPackedLinear, FastPackedLMHead, _Mat, _signs, _t
from .formats import FMT_PARAMS, NVFP4_FID, rht_in, rht_in_stack, rht_out, rht_out_stack
from .gemv import DenseStageGEMV, MoESideGEMV
from .schema import validate_packed_dir
from .tuning import tuning_for
from .triton_kernels import group_by_expert, packed_matmul


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

    @classmethod
    def from_json(cls, path: Path) -> "Inkling2Config":
        raw = json.loads(Path(path).read_text())
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


# ----------------------------------------------------------------------------- Primitives (float32, as transformers)
# Pure functions; `_fused` ones are compiled by torch.compile into one or two generated kernels each (set
# LAMQ_CUDA_COMPILE=0 to run them eagerly). Cache writes stay outside them as plain in-place copies.

_COMPILE = os.environ.get("LAMQ_CUDA_COMPILE", "1") != "0"
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)


def _fused(fn):
    return torch.compile(fn, fullgraph=True) if _COMPILE else fn


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return y * weight if weight is not None else y


def l2_clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    norm = x.norm(dim=-1, keepdim=True)
    return x * torch.clamp(threshold / (norm + 1e-6), max=1.0)


def log_scale(x: torch.Tensor, pos: torch.Tensor, alpha: float, n_floor: int) -> torch.Tensor:
    """x `[T, heads, d]`, pos `[T]`."""
    tau = 1.0 + alpha * torch.log(torch.clamp((pos.float() + 1.0) / float(n_floor), min=1.0))
    return x * tau[:, None, None]


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.tanh(x / cap) * cap


def softcapped_swiglu(gate_up: torch.Tensor, gate_cap: float, up_cap: float) -> torch.Tensor:
    g, u = gate_up.float().chunk(2, dim=-1)
    return softcap(g, gate_cap) * torch.sigmoid(g) * softcap(u, up_cap)


@_fused
def block_attn_res(query: torch.Tensor, block_states: list[torch.Tensor], partial: torch.Tensor) -> torch.Tensor:
    """Attention over the stacked block states of each token. query `[Q, H]` -> `[Q, T, H]`."""
    st = rms_norm(torch.stack(block_states + [partial.float()], dim=-2), None, 1e-6)  # [T, S, H]
    H = st.shape[-1]
    w = torch.softmax((st @ query.T).transpose(-1, -2) * H ** -0.5, dim=-1)  # [T, Q, S]
    return (w @ st).transpose(0, 1)


@_fused
def norm_f(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return rms_norm(x, weight, eps)


@_fused
def message_gate(h, msg, dw, fw, mw):
    """transformers' Inkling2LoopMessageGate: down_proj in bf16, the two gates in float32."""
    g = (torch.cat([rms_norm(h, None, 1e-6), rms_norm(msg, None, 1e-6)], dim=-1).to(dw.dtype) @ dw.T).float()
    return (1 - torch.tanh(g @ fw.T)) * h + torch.tanh(g @ mw.T) * msg


@_fused
def message_gate_3(h3, msgs3, dw3, fw3, mw3):
    """The q, k and v message gates in one call: h3 / msgs3 `[3, T, H]`, weights stacked `[3, ...]`."""
    g = (torch.cat([rms_norm(h3, None, 1e-6), rms_norm(msgs3, None, 1e-6)], dim=-1).to(dw3.dtype) @ dw3.transpose(1, 2)).float()
    return (1 - torch.tanh(g @ fw3.transpose(1, 2))) * h3 + torch.tanh(g @ mw3.transpose(1, 2)) * msgs3


@_fused
def q_post(q_raw, scale, pos, thr: float, alpha: float, n_floor: int, full: bool):
    """`[T, heads * d]` -> clipped, log-scaled (full-attention layers) `[T, heads, d]`."""
    T, nh, d = q_raw.shape[0], scale.shape[0], scale.shape[1]
    q = l2_clip(q_raw.reshape(T, nh, d) * scale, thr)
    return log_scale(q, pos, alpha, n_floor) if full else q


@_fused
def conv4(prev, x, w_kc):
    """Static causal conv on `ctx = [prev3, x]`. Returns (out, new prev3)."""
    T = x.shape[0]
    ctx = torch.cat([prev, x.float()], dim=0)
    out = ctx[0:T] * w_kc[0] + ctx[1 : T + 1] * w_kc[1] + ctx[2 : T + 2] * w_kc[2] + ctx[3 : T + 3] * w_kc[3]
    return (out + x.float()) * (1.0 / math.sqrt(5.0)), ctx[-3:]


@_fused
def dyn_conv4(prev, x, dyn_in, st_hdk, proj_w):
    """Dynamic conv on full-attention K/V: taps = static + per-token projection, shared within each head;
    `st_hdk` is `[heads, head_dim, 4]`, tap 0 weights the current token. Returns (out, new prev3)."""
    T, C = x.shape
    nh = st_hdk.shape[0]
    hd = C // nh
    ctx = torch.cat([prev, x.float()], dim=0)
    w = st_hdk[None] + (dyn_in.float() @ proj_w.T).reshape(T, nh, 1, 4)
    c0, c1 = ctx[3 : T + 3].reshape(T, nh, hd), ctx[2 : T + 2].reshape(T, nh, hd)
    c2, c3 = ctx[1 : T + 1].reshape(T, nh, hd), ctx[0:T].reshape(T, nh, hd)
    out = (c0 * w[..., 0] + c1 * w[..., 1] + c2 * w[..., 2] + c3 * w[..., 3]).reshape(T, C)
    return (out + x.float()) * (1.0 / math.sqrt(9.0)), ctx[-3:]


@_fused
def kv_post(k_raw, v_raw, nkv: int, thr: float, eps: float):
    T = k_raw.shape[0]
    k = l2_clip(k_raw.reshape(T, nkv, -1), thr)
    v = rms_norm(v_raw.reshape(T, nkv, -1), None, eps)
    return k.transpose(0, 1).half(), v.transpose(0, 1).half()


@_fused
def rel_post(rel_raw, scale, proj, q_in, sink_w, pos, thr: float, alpha: float, n_floor: int, full: bool):
    T, nh, dr = rel_raw.shape[0], scale.shape[0], scale.shape[1]
    rel = l2_clip(rel_raw.reshape(T, nh, dr) * scale, thr)
    if full:
        rel = log_scale(rel, pos, alpha, n_floor)
    return rel @ proj, q_in @ sink_w.T  # rel logits [T, nh, extent], sinks [T, nh]


@_fused
def rel_state(rel_raw, scale, q_in, sink_w, pos, thr: float, alpha: float, n_floor: int, full: bool):
    """Decode: the clipped (log-scaled) relative-position states `[T, nh, d_rel]` and sinks; the attention kernel
    computes the bias from them."""
    T, nh, dr = rel_raw.shape[0], scale.shape[0], scale.shape[1]
    rel = l2_clip(rel_raw.reshape(T, nh, dr) * scale, thr)
    if full:
        rel = log_scale(rel, pos, alpha, n_floor)
    return rel, q_in @ sink_w.T


@_fused
def gate_mul(attn, gate_raw):
    return attn.reshape(gate_raw.shape) * torch.sigmoid(gate_raw)


@_fused
def route_top8(h, router_w, router_bias, k: int, route_scale: float):
    scores = torch.sigmoid(h.float() @ router_w.T)
    idx = torch.topk(scores + router_bias, k, dim=-1, sorted=False).indices
    w = scores.gather(1, idx)
    return idx, w / (w.sum(-1, keepdim=True) + 1e-20) * route_scale


@_fused
def swiglu_f(gate_up, gate_cap: float, up_cap: float):
    return softcapped_swiglu(gate_up, gate_cap, up_cap)


@_fused
def moe_combine(dn, w):
    T, K = w.shape
    return (dn.reshape(T, K, -1) * w[..., None]).sum(1)


# ----------------------------------------------------------------------------- 48-visit static cache

class Inkling2Cache:
    """Pre-allocated KV (`float16 [2, n_kv, cap, d]` a visit) and causal-conv states (`float32 [3, C]`, 4 a visit).

    `pos` is the number of tokens already in the cache, kept on the device so a captured decode step can advance it.
    Sliding layers keep all positions too (attention reads only the last window), which keeps rollback trivial.
    """

    def __init__(self, cfg: Inkling2Config, capacity: int, device: torch.device):
        self.cfg, self.capacity, self.device = cfg, capacity, device
        n, kvh, d = cfg.total_visits, cfg.num_key_value_heads, cfg.head_dim
        self.kv = torch.zeros(n, 2, kvh, capacity, d, dtype=torch.float16, device=device)
        kv_dim, H = kvh * d, cfg.hidden_size
        self.conv_state = [
            [torch.zeros(3, kv_dim, device=device), torch.zeros(3, kv_dim, device=device),
             torch.zeros(3, H, device=device), torch.zeros(3, H, device=device)]
            for _ in range(n)
        ]
        self.pos = torch.zeros(1, dtype=torch.int32, device=device)
        self.seq_len = 0  # host mirror, updated by the caller after prefill/decode

    def reset(self):
        self.pos.zero_()
        self.seq_len = 0
        for slot in self.conv_state:
            for c in slot:
                c.zero_()

    def conv(self, slot: int, idx: int, x: torch.Tensor, w_kc: torch.Tensor) -> torch.Tensor:
        return conv4_(self.conv_state[slot][idx], x, w_kc)

    def dyn_conv(self, slot: int, idx: int, x, dyn_in, st_hdk, proj_w) -> torch.Tensor:
        return dyn_conv4_(self.conv_state[slot][idx], x, dyn_in, st_hdk, proj_w)


# ----------------------------------------------------------------------------- MoE

def _mat_spec(mat):
    """(fmt, m, codes, scales, table) of a packed_modules._Mat, for DenseStageGEMV."""
    return (mat.fmt, mat.m, mat.codes, mat.scales, mat.levels if mat.fmt.startswith("lm") else mat.lut)


class PackedMoE:
    """Sigmoid top-8 router + packed experts. Small batches use one block per (token, expert) pair, built on the
    device without a host sync (capturable); large batches group pairs by expert first."""

    BLOCK_T = 16

    def __init__(self, arrays, luts, router_w, router_bias, cfg: Inkling2Config, device):
        self.cfg = cfg
        self.router_w = router_w.float()
        self.router_bias = router_bias.float()
        self.gu_signs = _signs(arrays["gu_signs_in"], device)
        self.dn_signs = _signs(arrays["dn_signs_in"], device)
        E = cfg.n_routed_experts
        self.sides = {}
        self.nvfp4 = {}  # side -> (codes [E_nv, m, n/2] u8, block scales [E_nv, m, n/16] u8, outer [E_nv] f32, local [E] i32)
        for side, (m, n) in (("gu", (2 * cfg.intermediate_size, cfg.hidden_size)), ("dn", (cfg.hidden_size, cfg.intermediate_size))):
            emap = np.asarray(arrays[f"{side}_map"], dtype=np.int64)
            mats = []
            for fmt in FMT_PARAMS:
                if f"{side}_{fmt}_codes" not in arrays:
                    continue
                local = np.full(E, -1, dtype=np.int64)
                sel = emap[:, 0] == FMT_ID[fmt]
                local[sel] = emap[sel, 1]
                codes = _t(arrays[f"{side}_{fmt}_codes"], device)
                mats.append((_Mat(fmt, m, n, codes, _t(arrays[f"{side}_{fmt}_scales"], device), lut=luts[fmt]),
                             torch.from_numpy(local).to(device=device, dtype=torch.int32), codes.shape[0]))
            if f"{side}_nvfp4_codes" in arrays:
                local = np.full(E, -1, dtype=np.int64)
                sel = emap[:, 0] == NVFP4_FID
                local[sel] = emap[sel, 1]
                self.nvfp4[side] = (_t(arrays[f"{side}_nvfp4_codes"], device), _t(arrays[f"{side}_nvfp4_scales"], device),
                                    _t(arrays[f"{side}_nvfp4_outer"], device).float(),
                                    torch.from_numpy(local).to(device=device, dtype=torch.int32))
            self.sides[side] = (m, n, mats)
        # decode: all experts of a side in one launch (trellis slabs 0-3, NVFP4 slab 4)
        self.unified = {}
        for side, (m, n, mats) in self.sides.items():
            slab_of = torch.full((E,), 0, dtype=torch.int32, device=device)
            loc_of = torch.full((E,), 0, dtype=torch.int32, device=device)
            for j, (mat, local, _) in enumerate(mats):
                sel = local >= 0
                slab_of[sel] = j
                loc_of[sel] = local[sel]
            nv = self.nvfp4.get(side)
            if nv is not None:
                sel = nv[3] >= 0
                slab_of[sel] = MoESideGEMV.NV_SLAB
                loc_of[sel] = nv[3][sel]
            self.unified[side] = MoESideGEMV([(mat.fmt, mat.codes, mat.scales, mat.lut) for mat, _, _ in mats], slab_of, loc_of, m, n,
                                             nvfp4=nv[:3] if nv is not None else None)
        self._pair_blocks: dict[int, torch.Tensor] = {}

    def route(self, h: torch.Tensor):
        return route_top8(h, self.router_w, self.router_bias, self.cfg.num_experts_per_tok, self.cfg.route_scale)

    def _blocks_per_pair(self, P: int, device) -> torch.Tensor:
        if P not in self._pair_blocks:
            bp = torch.full((P, self.BLOCK_T), -1, dtype=torch.int32, device=device)
            bp[:, 0] = torch.arange(P, dtype=torch.int32, device=device)
            self._pair_blocks[P] = bp.reshape(-1)
        return self._pair_blocks[P]

    def _side(self, side: str, x: torch.Tensor | None, x_raw: torch.Tensor, eids: torch.Tensor, x_div: int, grouped: bool) -> torch.Tensor:
        """`x` the rotated input (trellis experts; None if the side has none), `x_raw` the unrotated one (NVFP4)."""
        m, n, mats = self.sides[side]
        nv = self.nvfp4.get(side)
        P = eids.numel()
        if not grouped:  # decode: one GEMV slot a pair, all formats in one launch
            return self.unified[side](x if x is not None else x_raw, eids, x_div, x_raw=x_raw if nv is not None else None)
        y = torch.zeros(P, m, dtype=torch.float32, device=x_raw.device)
        groups = [(mat, local, n_local, x) for mat, local, n_local in mats]
        if nv is not None:
            groups.append((None, nv[3], nv[0].shape[0], x_raw))
        for mat, local, n_local, xin in groups:
            loc = local[eids]  # [P], -1 where the expert uses another format
            pairs = torch.nonzero(loc >= 0).squeeze(1)
            if pairs.numel() == 0:
                continue
            be, bp = group_by_expert(loc[pairs].long(), n_local, self.BLOCK_T)
            bp = torch.where(bp >= 0, pairs.to(torch.int32)[bp.clamp(min=0).long()], bp)
            if mat is None:
                packed_matmul(xin, nv[0], nv[1], "nvfp4", m, n, levels=nv[2], block_expert=be, block_pairs=bp,
                              n_pairs=P, x_div=x_div, out=y)
            else:
                mat(xin, block_expert=be, block_pairs=bp, n_pairs=P, x_div=x_div, out=y)
        return y

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        T = h.shape[0]
        K = self.cfg.num_experts_per_tok
        idx, w = self.route(h)
        eids = idx.reshape(-1)
        grouped = T * K > 64
        rot = lambda side, a, s: rht_in(a, s) if self.sides[side][2] else None  # noqa: E731  trellis experts only
        gu = self._side("gu", rot("gu", h, self.gu_signs), h, eids, K, grouped)
        act = swiglu_f(gu, self.cfg.mlp_gate_softcap, self.cfg.mlp_up_softcap)
        dn = self._side("dn", rot("dn", act, self.dn_signs), act, eids, 1, grouped)
        return moe_combine(dn, w)


# ----------------------------------------------------------------------------- Decoder layer

class PackedDecoderLayer:
    def __init__(self, i: int, cfg: Inkling2Config, packed_dir: Path, luts, st: dict[str, torch.Tensor], device):
        self.i, self.cfg = i, cfg
        self.is_sliding = i in cfg.local_layer_ids
        self.is_loop = cfg.loop_start <= i < cfg.loop_end
        self.is_block_boundary = i % cfg.n_layers_per_attn_res_block == 0
        p = f"model.layers.{i}."
        meta = json.loads((packed_dir / f"layer_{i:02d}_meta.json").read_text())
        npz = np.load(packed_dir / f"layer_{i:02d}_packed.npz")
        arrays = {k: npz[k] for k in npz.files}
        lin = lambda name: FastPackedLinear(meta["linears"][name], arrays, luts, device)  # noqa: E731
        self.q_lin = lin("self_attn.q_proj.linear")
        self.k_lin = lin("self_attn.k_proj.linear")
        self.v_lin = lin("self_attn.v_proj.linear")
        self.o_lin = lin("self_attn.o_proj.linear")
        self.o_gate = lin("self_attn.o_proj.gate_proj")
        self.rel_lin = lin("self_attn.relative_position_bias.linear")
        # the five matrices reading the attention input share one rotation launch (one sign vector each)
        self.in_stage = [self.q_lin, self.k_lin, self.v_lin, self.rel_lin, self.o_gate]
        self.in_signs = torch.stack([l.signs_in for l in self.in_stage])
        # their output rotations, batched by output size
        self.out_groups: dict[int, list[int]] = {}
        for j, l in enumerate(self.in_stage):
            if l.signs_out is not None:
                self.out_groups.setdefault(l.out_features, []).append(j)
        self.out_signs = {m: torch.stack([self.in_stage[j].signs_out for j in js]) for m, js in self.out_groups.items()}
        self.in_dense = DenseStageGEMV([_mat_spec(l.mat) for l in self.in_stage], cfg.hidden_size, x_rows=list(range(5)))

        f32 = lambda k: st[p + k].float()  # noqa: E731
        self.attn_in_q = f32("attention_input.query").reshape(-1, cfg.hidden_size) if p + "attention_input.query" in st else None
        self.mlp_in_q = f32("mlp_input.query").reshape(1, cfg.hidden_size)
        if self.is_sliding:
            self.in_ln = f32("input_layernorm.weight")
            self.k_conv = f32("self_attn.k_proj.sconv.conv1d.weight")[:, 0, :].T.contiguous()  # [4, C]
            self.v_conv = f32("self_attn.v_proj.sconv.conv1d.weight")[:, 0, :].T.contiguous()
        else:
            self.q_in_ln, self.k_in_ln, self.v_in_ln = (f32(f"{x}_input_layernorm.weight") for x in "qkv")
            self.qkv_ln = torch.stack([self.q_in_ln, self.k_in_ln, self.v_in_ln])
            nkv, hd = cfg.num_key_value_heads, cfg.head_dim
            self.k_dyn = (f32("self_attn.k_proj.sconv.static_weight").T.reshape(nkv, hd, 4).contiguous(), f32("self_attn.k_proj.sconv.weight_proj.weight"))
            self.v_dyn = (f32("self_attn.v_proj.sconv.static_weight").T.reshape(nkv, hd, 4).contiguous(), f32("self_attn.v_proj.sconv.weight_proj.weight"))
        self.post_attn_ln = f32("post_attention_layernorm.weight")
        self.q_scale = f32("self_attn.q_proj.norm.scale")
        self.rel_scale = f32("self_attn.relative_position_bias.norm.scale")
        self.rel_proj = f32("self_attn.relative_position_bias.proj").contiguous()
        self.sink_w = f32("self_attn.sink_proj.weight")
        self.post_attn_conv = f32("post_attention_convolution.conv1d.weight")[:, 0, :].T.contiguous()
        self.post_mlp_conv = f32("post_mlp_convolution.conv1d.weight")[:, 0, :].T.contiguous()
        if self.is_loop:  # transformers' dtypes: down_proj in the model dtype (bf16), the two gate projections in f32
            self.msg_gates = {
                g: (st[p + f"{g}.down_proj.weight"].to(torch.bfloat16), f32(f"{g}.forget_gate_proj.weight"), f32(f"{g}.message_gate_proj.weight"))
                for g in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate")
            }
            self.qkv_gate = tuple(torch.stack([self.msg_gates[f"{x}_message_gate"][i] for x in "qkv"]).contiguous() for i in range(3))
            self.mlp_gate = tuple(w[None].contiguous() for w in self.msg_gates["mlp_message_gate"])
        self.moe = PackedMoE(arrays, luts, st[p + "mlp.gate.weight"], st[p + "mlp.gate.e_score_correction_bias"], cfg, device)

    def _in_stage(self, q_in, k_in, v_in):
        """q, k, v, relative-bias and gate projections: one stacked input rotation, five packed matmuls, the output
        rotations batched by size."""
        xs = torch.stack([q_in, k_in, v_in, q_in, q_in]) if k_in is not q_in else q_in[None].expand(5, -1, -1)
        x_rot = rht_in_stack(xs, self.in_signs)
        if xs.shape[1] == 1:  # decode: the five matrices in one launch
            ys = [y[None] for y in self.in_dense.split(self.in_dense(x_rot[:, 0]))]
        else:
            ys = [l.mat(x_rot[j]) for j, l in enumerate(self.in_stage)]
        for m, js in self.out_groups.items():
            if len(js) == 1:
                ys[js[0]] = rht_out(ys[js[0]], self.out_signs[m][0])
            else:
                rot = rht_out_stack(torch.stack([ys[j] for j in js]), self.out_signs[m])
                for i, j in enumerate(js):
                    ys[j] = rot[i]
        return ys

    def _msg_gate(self, name, h, msg):
        return message_gate(h, msg, *self.msg_gates[name])

    def __call__(self, h, block_states, pos, cache: Inkling2Cache, slot: int, messages=None, decode: bool = False):
        cfg, eps = self.cfg, self.cfg.rms_norm_eps
        T = h.shape[0]
        nq, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        full = not self.is_sliding
        block_residual = h

        if self.is_sliding:
            if self.attn_in_q is None:  # layer 0
                q_in = k_in = v_in = norm_f(h, self.in_ln, eps)
            elif messages is None:
                q_in = k_in = v_in = block_attn_res_fused(self.attn_in_q, block_states + [h], self.in_ln[None], eps)[0]
            else:
                ha = block_attn_res_fused(self.attn_in_q, block_states + [h])[0]
                gated = message_gates_fused(ha[None].expand(3, -1, -1), messages[:3], *self.qkv_gate)
                q_in, k_in, v_in = norm_f(gated, self.in_ln, eps).unbind(0)
        else:
            if messages is None:
                q_in, k_in, v_in = block_attn_res_fused(self.attn_in_q, block_states + [h], self.qkv_ln, eps).unbind(0)
            else:
                qkv_h = block_attn_res_fused(self.attn_in_q, block_states + [h])
                gated = message_gates_fused(qkv_h, messages[:3], *self.qkv_gate)
                q_in, k_in, v_in = norm_f(gated, self.qkv_ln[:, None, :], eps).unbind(0)

        q_raw, k_raw, v_raw, rel_raw, gate_raw = self._in_stage(q_in, k_in, v_in)
        q = q_post(q_raw, self.q_scale, pos, cfg.qk_l2_clip_threshold, cfg.log_scaling_alpha, cfg.log_scaling_n_floor, full)
        kv = cache.kv[slot]  # [2, nkv, cap, hd]
        pk, pv = cache.conv_state[slot][0], cache.conv_state[slot][1]
        if self.is_sliding:
            kv_conv_write_(k_raw, v_raw, pk, pv, kv, pos, cfg.qk_l2_clip_threshold, eps, nkv, w_k=self.k_conv, w_v=self.v_conv)
        else:
            kv_conv_write_(k_raw, v_raw, pk, pv, kv, pos, cfg.qk_l2_clip_threshold, eps, nkv, dyn=(k_in, v_in, self.k_dyn, self.v_dyn))

        window = cfg.sliding_window_size if self.is_sliding else 0
        if decode:
            rel, sinks = rel_state(rel_raw, self.rel_scale, q_in, self.sink_w, pos,
                                   cfg.rel_l2_clip_threshold, cfg.log_scaling_alpha, cfg.log_scaling_n_floor, full)
            tn = tuning_for(q.device)
            attn = decode_attention(q[0], kv, rel[0], sinks[0], pos, window, cfg.rel_extent, proj=self.rel_proj,
                                    nsplit=tn.attn_nsplit, block_k=tn.attn_block_k)[None]
        else:
            rel_logits, sinks = rel_post(rel_raw, self.rel_scale, self.rel_proj, q_in, self.sink_w, pos,
                                         cfg.rel_l2_clip_threshold, cfg.log_scaling_alpha, cfg.log_scaling_n_floor, full)
            L = int(pos[-1].item()) + 1  # prefill: host knows the length
            k_pos = torch.arange(L, device=h.device)
            attn = prefill_attention(q.transpose(0, 1), kv[0, :, :L], kv[1, :, :L], rel_logits, sinks, pos.long(), k_pos, window, cfg.rel_extent)
        attn = gate_mul(attn, gate_raw)
        attn_out = cache.conv(slot, 2, self.o_lin(attn), self.post_attn_conv)

        if self.is_block_boundary:
            local_blocks, residual, h_post = block_states + [block_residual.float()], attn_out, attn_out
        else:
            local_blocks = block_states
            residual = h_post = block_residual.float() + attn_out
        if messages is None:
            h_mlp = block_attn_res_fused(self.mlp_in_q, local_blocks + [h_post], self.post_attn_ln[None], eps)[0]
        else:
            h_mlp = block_attn_res_fused(self.mlp_in_q, local_blocks + [h_post])
            h_mlp = norm_f(message_gates_fused(h_mlp, messages[3][None], *self.mlp_gate)[0], self.post_attn_ln, eps)
        mlp_out = cache.conv(slot, 3, self.moe(h_mlp), self.post_mlp_conv)
        return residual + mlp_out


# ----------------------------------------------------------------------------- Model

class Inkling2LoopPacked:
    """The packed Inkling-2-Mini text model on one CUDA device."""

    def __init__(self, packed_dir: str | Path, device: str | torch.device = "cuda"):
        self.packed_dir = Path(packed_dir)
        self.device = torch.device(device)
        validate_packed_dir(self.packed_dir)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        dev = self.device
        self.cfg = cfg = Inkling2Config.from_json(self.packed_dir / "config.json")
        luts = {k: torch.from_numpy(v).to(dev) for k, v in np.load(self.packed_dir / "luts.npz").items()}
        st = load_file(str(self.packed_dir / "steering.safetensors"), device=str(dev))

        self.embed_norm_w = st["model.embed_tokens.embed_norm.weight"].float()
        self.embed = self._build_embedding(luts)
        self.final_in_q = st["model.final_input.query"].float().reshape(1, -1)
        self.final_norm_w = st["model.norm.weight"].float()
        self.message_in_q = st["model.message_input.query"].float()  # [loop layers, 4, H]
        with torch.cuda.device(dev):
            self.layers = [PackedDecoderLayer(i, cfg, self.packed_dir, luts, st, dev) for i in range(cfg.num_hidden_layers)]
            self.lm_head = FastPackedLMHead(self.packed_dir, luts, dev, cfg.hidden_size)
        self.order = torch.from_numpy(np.load(self.packed_dir / "head_packed.npz")["order"].astype(np.int64)).to(dev)
        self.head_signs = torch.stack(self.lm_head.signs)
        self.head_dense = DenseStageGEMV([_mat_spec(mat) for mat in self.lm_head.mats], cfg.hidden_size,
                                         x_rows=list(range(len(self.lm_head.mats))))

    def _build_embedding(self, luts) -> torch.Tensor:
        """Dense bf16 `[V, H]` table in the original basis, decoded once (0.9 GB)."""
        from .formats import decode_lm, decode_trellis, rht_out

        meta = json.loads((self.packed_dir / "embed_meta.json").read_text())
        npz = np.load(self.packed_dir / "embed_packed.npz")
        H = self.cfg.hidden_size
        rows = []
        for bm in meta:
            kb, n_rows, fmt = bm["kb"], int(bm["rows"]), bm["opt"]
            scales = _t(npz[f"b{kb}_scales"], self.device)
            for s in range(0, n_rows, 16384):  # in slices, to bound the float32 temporaries
                e = min(n_rows, s + 16384)
                if fmt.startswith("lm"):
                    bpr = H * bm["k"] // 8
                    codes = _t(npz[f"b{kb}_codes"].reshape(n_rows, bpr)[s:e], self.device)
                    Q = decode_lm(codes, scales[s:e], _t(npz[f"b{kb}_levels"], self.device), e - s, H, bm["k"])
                else:
                    codes = _t(npz[f"b{kb}_codes"].reshape(n_rows, -1)[s:e], self.device)
                    Q = decode_trellis(codes, scales[s:e], luts[fmt], H, fmt)
                rows.append(rht_out(Q, _signs(npz[f"b{kb}_signs_in"], self.device)).to(torch.bfloat16))
        table = torch.cat(rows, 0)
        order = torch.from_numpy(np.asarray(npz["order"], dtype=np.int64)).to(self.device)
        out = torch.empty_like(table)
        out[order] = table  # row p holds token order[p]
        return out

    def make_cache(self, capacity: int = 4096) -> Inkling2Cache:
        return Inkling2Cache(self.cfg, capacity, self.device)

    @torch.no_grad()
    def hidden(self, ids: torch.Tensor, cache: Inkling2Cache, pos: torch.Tensor, decode: bool = False, tap_visits=()):
        """Final hidden states `[T, H]` for token ids `[T]` at positions `pos` `[T]` (int32, on device)."""
        cfg = self.cfg
        h = norm_f(self.embed[ids.long()], self.embed_norm_w, cfg.rms_norm_eps)
        blocks: list[torch.Tensor] = []
        taps = {}
        visit = 0

        def run(L, h, blocks, slot, messages=None):
            nonlocal visit
            if visit in tap_visits:
                taps[visit] = h.clone()
            new_h = L(h, blocks, pos, cache, slot, messages, decode)
            if L.is_block_boundary:
                blocks = blocks + [h.float()]
            visit += 1
            return new_h, blocks

        for i in range(cfg.loop_start):
            h, blocks = run(self.layers[i], h, blocks, i)
        pre_h, pre_blocks = h, list(blocks)
        n_pre = len(pre_blocks)
        for i in range(cfg.loop_start, cfg.loop_end):
            h, blocks = run(self.layers[i], h, blocks, i)
        n_loop = cfg.loop_end - cfg.loop_start
        for p in range(1, cfg.loop_passes):
            prev_blocks, prev_h = blocks[n_pre + 1 :], h
            h, blocks = pre_h, list(pre_blocks)
            offset = cfg.num_hidden_layers + (p - 1) * n_loop - cfg.loop_start
            for j, i in enumerate(range(cfg.loop_start, cfg.loop_end)):
                msgs = block_attn_res_fused(self.message_in_q[j], prev_blocks + [prev_h])
                h, blocks = run(self.layers[i], h, blocks, i + offset, msgs)
        for i in range(cfg.loop_end, cfg.num_hidden_layers):
            h, blocks = run(self.layers[i], h, blocks, i)
        h = block_attn_res_fused(self.final_in_q, blocks + [h], self.final_norm_w[None], cfg.rms_norm_eps)[0]
        if visit in tap_visits:
            taps[visit] = h.clone()
        return (h, taps) if tap_visits else h

    def ordered_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Pre-softcap logits in bucket order (column o is token `order[o]`)."""
        x = h.reshape(-1, h.shape[-1])
        if x.shape[0] == 1:  # decode: the 8 buckets in one launch (their outputs are already in bucket order)
            x_rot = rht_in_stack(x[None].expand(len(self.lm_head.mats), -1, -1), self.head_signs)[:, 0]
            return self.head_dense(x_rot)[None]
        return torch.cat([mat(rht_in(x, s)) for mat, s in zip(self.lm_head.mats, self.lm_head.signs)], dim=-1)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return softcap(self.ordered_logits(h)[:, self.lm_head.inv_order], self.cfg.final_logit_softcapping)

    def argmax(self, h: torch.Tensor) -> torch.Tensor:
        """Greedy next tokens without the 200K gather (softcap is monotonic)."""
        return self.order[self.ordered_logits(h).argmax(-1)]

    @torch.no_grad()
    def prefill(self, ids: list[int] | torch.Tensor, cache: Inkling2Cache, all_logits: bool = False, chunk: int = 1024):
        """Runs the prompt through the model and the cache. Returns logits of the last token (or all tokens)."""
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(-1)
        outs = []
        for s in range(0, ids.numel(), chunk):
            part = ids[s : s + chunk]
            start = cache.seq_len
            pos = torch.arange(start, start + part.numel(), device=self.device, dtype=torch.int32)
            h = self.hidden(part, cache, pos)
            cache.seq_len += part.numel()
            cache.pos.fill_(cache.seq_len)
            outs.append(self.logits(h) if all_logits else (self.logits(h[-1:]) if s + chunk >= ids.numel() else None))
        return torch.cat([o for o in outs if o is not None], 0)

    @torch.no_grad()
    def decode_step(self, token: torch.Tensor, cache: Inkling2Cache) -> torch.Tensor:
        """One token (`[1]` int on device) at position `cache.pos`; advances `cache.pos` on the device. Returns the final
        hidden state `[1, H]`. Touches only fixed buffers, so it can be captured in a CUDA graph."""
        h = self.hidden(token, cache, cache.pos, decode=True)
        cache.pos.add_(1)
        return h
