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

from .attention import decode_attention, decode_attention_lm, prefill_attention_kv, prefill_attention_kv_lm
from .conv import (
    KV_MODE_IDS,
    conv4_,
    default_rht_signs,
    dyn_conv4_,
    kv_conv_write_,
    kv_conv_write_lm_,
    lm_bytes_per_head,
    parse_kv_mode,
)
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
# LAMQ_CUDA_COMPILE=0 to run them eagerly). Cache writes stay outside them as plain in-place copies. They compile
# with symbolic shapes from the first call: otherwise the first prompt length gets its own specialised kernels and
# later lengths recompile into symbolic ones, which round differently, so the same prompt could generate different
# text depending on what ran before it.

_COMPILE = os.environ.get("LAMQ_CUDA_COMPILE", "1") != "0"
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)


def _fused(fn):
    return torch.compile(fn, fullgraph=True, dynamic=True) if _COMPILE else fn


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

def _slot_phys_layer(cfg: Inkling2Config, slot: int) -> int:
    if slot < cfg.num_hidden_layers:
        return slot
    if slot < cfg.total_visits:
        n_loop = cfg.loop_end - cfg.loop_start
        return cfg.loop_start + ((slot - cfg.num_hidden_layers) % n_loop)
    return cfg.loop_end + (slot - cfg.total_visits)


def _slot_is_sliding(cfg: Inkling2Config, slot: int) -> bool:
    return _slot_phys_layer(cfg, slot) in cfg.local_layer_ids


def _normalize_kv_mode(mode: str) -> str:
    m = mode.strip().lower()
    if m == "q4":
        m = "q4_0"
    elif m == "q8":
        m = "q8_0"
    fam, kb, vb = parse_kv_mode(m)
    if fam == "lm":
        return f"lm{kb}" if kb == vb else f"lm{kb}_{vb}"
    return fam


def _default_global_kv_mode() -> str:
    return _normalize_kv_mode(os.environ.get("LAMQ_CUDA_GLOBAL_KV", "fp16"))


def _default_global_kv_plan() -> str | None:
    raw = os.environ.get("LAMQ_CUDA_GLOBAL_KV_PLAN", "").strip()
    return raw or None


def _default_shared_loop_kv() -> bool:
    return os.environ.get("LAMQ_CUDA_SHARED_LOOP_KV", "0").strip().lower() in ("1", "true", "yes")


def _default_kv_vram_cells() -> int | None:
    raw = os.environ.get("LAMQ_CUDA_KV_VRAM_CELLS", "").strip()
    if not raw:
        return None
    val = int(raw)
    if val < 0:
        raise ValueError("LAMQ_CUDA_KV_VRAM_CELLS must be >= 0")
    return val


def _parse_plan_entry_mode(spec) -> str:
    if isinstance(spec, (tuple, list)):
        if len(spec) != 2:
            raise ValueError(f"Expected (k_bits, v_bits) pair, got {spec!r}")
        kb, vb = int(spec[0]), int(spec[1])
        if kb == 16 and vb == 16:
            return "fp16"
        if kb == 8 and vb == 8:
            return "q8_0"
        return _normalize_kv_mode(f"lm{kb}" if kb == vb else f"lm{kb}_{vb}")
    s = str(spec).strip().lower()
    if s in ("2", "3", "4"):
        return f"lm{s}"
    if "," in s:
        p0, p1 = (x.strip() for x in s.split(",", 1))
        if p0.isdigit() and p1.isdigit():
            return _parse_plan_entry_mode((int(p0), int(p1)))
        if p0.startswith("lm") and p1.startswith("lm") and p0[2:].isdigit() and p1[2:].isdigit():
            return _parse_plan_entry_mode((int(p0[2:]), int(p1[2:])))
        raise ValueError(f"Unsupported KV plan entry {spec!r}")
    return _normalize_kv_mode(s)


def _resolve_slot_modes(
    cfg: Inkling2Config,
    n_slots: int,
    is_sliding: list[bool],
    default_mode: str,
    plan: str | dict | None,
) -> tuple[list[str], str | None]:
    fallback = _normalize_kv_mode(default_mode)
    entries: dict[int, str] = {}
    if plan is not None:
        if isinstance(plan, str):
            items = [seg.strip() for seg in plan.split(";") if seg.strip()]
            for seg in items:
                if ":" not in seg:
                    raise ValueError(f"Invalid LAMQ_CUDA_GLOBAL_KV_PLAN segment {seg!r} (expected 'layer:k,v' or 'layer:b')")
                k_raw, v_raw = (x.strip() for x in seg.split(":", 1))
                mode = _parse_plan_entry_mode(v_raw)
                if k_raw.lower() in ("*", "default"):
                    fallback = mode
                else:
                    entries[int(k_raw)] = mode
        elif isinstance(plan, dict):
            for k_raw, v_raw in plan.items():
                mode = _parse_plan_entry_mode(v_raw)
                if isinstance(k_raw, str) and k_raw.strip().lower() in ("*", "default"):
                    fallback = mode
                else:
                    entries[int(k_raw)] = mode
        else:
            raise ValueError(f"global_kv_plan must be str, dict, or None, got {type(plan).__name__}")

    for idx in entries:
        if not (0 <= idx < n_slots):
            raise ValueError(f"KV plan layer/slot {idx} out of range [0, {n_slots})")
        if is_sliding[idx]:
            raise ValueError(f"KV plan layer/slot {idx} is a sliding-window layer (must stay fp16)")

    slot_modes = []
    for s in range(n_slots):
        if is_sliding[s]:
            slot_modes.append("fp16")
        elif s in entries:
            slot_modes.append(entries[s])
        else:
            phys = _slot_phys_layer(cfg, s)
            slot_modes.append(entries[phys] if phys in entries else fallback)
    plan_repr = ";".join(f"{k}:{entries[k]}" for k in sorted(entries)) if entries else None
    return slot_modes, plan_repr


def _snapshot_pair(vram: torch.Tensor, host: torch.Tensor | None, n_save: int, vc: int | None, dim: int) -> torch.Tensor:
    if vc is not None and n_save > vc:
        assert host is not None
        part_v = vram.narrow(dim, 0, vc).detach().cpu()
        part_h = host.narrow(dim, 0, n_save - vc).detach().cpu()
        return torch.cat([part_v, part_h], dim=dim).contiguous()
    return vram.narrow(dim, 0, n_save).detach().cpu().contiguous()


def _restore_pair(vram: torch.Tensor, host: torch.Tensor | None, src: torch.Tensor, vc: int | None, dim: int) -> None:
    n_src = src.shape[dim]
    if vc is not None and n_src > vc:
        assert host is not None
        vram.narrow(dim, 0, vc).copy_(src.narrow(dim, 0, vc), non_blocking=True)
        host.narrow(dim, 0, n_src - vc).copy_(src.narrow(dim, vc, n_src - vc))
    else:
        vram.narrow(dim, 0, n_src).copy_(src, non_blocking=True)


class Inkling2Cache:
    """Per-class 48-visit KV + causal-conv cache, CUDA-graph-capturable and long-context ready.

    - The 40 sliding-window visits allocate fixed FP16 ring buffers of `min(capacity, window + prefill_chunk)`
      cells (~335 MB total at any context length up to 262k).
    - The 8 global visits support `fp16` (32 KiB/tok), `q8_0` (17 KiB/tok), `q4_0` (9 KiB/tok), and Hadamard
      Lloyd-Max (`lm4`, `lm4_3`, `lm4_2`, `lm3`, `lm3_2`, `lm2`, plus per-layer `global_kv_plan`), optional
      `shared_loop_kv` aliasing across loop passes (`L11/L17` passes 0,1 -> canonical slots `41,47`), and
      optional tiered VRAM + pinned host-RAM UVA storage (`kv_vram_cells`).
    - `pos` lives on the device so a captured CUDA decode graph replays at any sequence length without host sync.
    """

    def __init__(
        self,
        cfg: Inkling2Config,
        capacity: int,
        device: torch.device,
        *,
        global_kv_mode: str | None = None,
        kv_vram_cells: int | None = None,
        prefill_chunk: int = 1024,
        shared_loop_kv: bool | None = None,
        global_kv_plan: str | dict | None = None,
    ):
        self.cfg, self.capacity, self.device = cfg, int(capacity), device
        self.prefill_chunk = int(prefill_chunk)
        self.global_kv_mode = _default_global_kv_mode() if global_kv_mode is None else _normalize_kv_mode(global_kv_mode)
        if kv_vram_cells is None:
            kv_vram_cells = _default_kv_vram_cells()
        self.kv_vram_cells = kv_vram_cells
        self.shared_loop_kv = _default_shared_loop_kv() if shared_loop_kv is None else bool(shared_loop_kv)
        if global_kv_plan is None:
            global_kv_plan = _default_global_kv_plan()

        n, kvh, d = cfg.total_visits, cfg.num_key_value_heads, cfg.head_dim
        sw_cap = min(self.capacity, cfg.sliding_window_size + max(self.prefill_chunk, 1))
        self.is_sliding = [_slot_is_sliding(cfg, s) for s in range(n)]
        self.slot_mode, self.global_kv_plan_str = _resolve_slot_modes(
            cfg, n, self.is_sliding, self.global_kv_mode, global_kv_plan,
        )

        share_active = self.shared_loop_kv and (cfg.loop_passes > 1)
        n_loop = cfg.loop_end - cfg.loop_start
        final_loop_offset = cfg.num_hidden_layers + (cfg.loop_passes - 2) * n_loop - cfg.loop_start
        self.canonical_slot: list[int] = []
        for s in range(n):
            if share_active and (not self.is_sliding[s]):
                phys = _slot_phys_layer(cfg, s)
                if cfg.loop_start <= phys < cfg.loop_end:
                    self.canonical_slot.append(phys + final_loop_offset)
                    continue
            self.canonical_slot.append(s)

        self.slot_family: list[str] = [""] * n
        self.slot_k_bits: list[int] = [16] * n
        self.slot_v_bits: list[int] = [16] * n
        self.slot_ring: list[bool] = [False] * n
        self.kv: list[torch.Tensor | None] = [None] * n
        self.scales: list[torch.Tensor | None] = [None] * n
        self.kv_host: list[torch.Tensor | None] = [None] * n
        self.scales_host: list[torch.Tensor | None] = [None] * n
        self.k_buf: list[torch.Tensor | None] = [None] * n
        self.k_sc: list[torch.Tensor | None] = [None] * n
        self.k_host: list[torch.Tensor | None] = [None] * n
        self.k_sc_host: list[torch.Tensor | None] = [None] * n
        self.v_buf: list[torch.Tensor | None] = [None] * n
        self.v_sc: list[torch.Tensor | None] = [None] * n
        self.v_host: list[torch.Tensor | None] = [None] * n
        self.v_sc_host: list[torch.Tensor | None] = [None] * n
        self.vram_cells: list[int | None] = [None] * n

        any_lm = False
        for s in range(n):
            if self.canonical_slot[s] != s:
                continue
            sw = self.is_sliding[s]
            mode = self.slot_mode[s]
            fam, kb, vb = parse_kv_mode(mode)
            ring = sw and sw_cap < self.capacity
            cap_s = sw_cap if sw else self.capacity
            self.slot_family[s] = fam
            self.slot_k_bits[s] = kb
            self.slot_v_bits[s] = vb
            self.slot_ring[s] = ring

            tiered = (not sw) and (kv_vram_cells is not None) and (0 < kv_vram_cells < cap_s)
            v_cells = int(kv_vram_cells) if tiered else cap_s
            h_cells = cap_s - v_cells
            if fam == "lm":
                any_lm = True
                kb_bytes = lm_bytes_per_head(d, kb)
                vb_bytes = lm_bytes_per_head(d, vb)
                self.k_buf[s] = torch.zeros(kvh, v_cells, kb_bytes, dtype=torch.uint8, device=device)
                self.k_sc[s] = torch.zeros(kvh, v_cells, dtype=torch.float16, device=device)
                self.v_buf[s] = torch.zeros(kvh, v_cells, vb_bytes, dtype=torch.uint8, device=device)
                self.v_sc[s] = torch.zeros(kvh, v_cells, dtype=torch.float16, device=device)
                if tiered:
                    self.k_host[s] = torch.zeros(kvh, h_cells, kb_bytes, dtype=torch.uint8, pin_memory=True)
                    self.k_sc_host[s] = torch.zeros(kvh, h_cells, dtype=torch.float16, pin_memory=True)
                    self.v_host[s] = torch.zeros(kvh, h_cells, vb_bytes, dtype=torch.uint8, pin_memory=True)
                    self.v_sc_host[s] = torch.zeros(kvh, h_cells, dtype=torch.float16, pin_memory=True)
                    self.vram_cells[s] = v_cells
            else:
                if fam == "fp16":
                    dtype, d_store, need_sc = torch.float16, d, False
                elif fam == "q8_0":
                    dtype, d_store, need_sc = torch.int8, d, True
                else:  # q4_0
                    dtype, d_store, need_sc = torch.uint8, d // 2, True
                self.kv[s] = torch.zeros(2, kvh, v_cells, d_store, dtype=dtype, device=device)
                if need_sc:
                    self.scales[s] = torch.zeros(2, kvh, v_cells, d // 32, dtype=torch.float16, device=device)
                if tiered:
                    self.kv_host[s] = torch.zeros(2, kvh, h_cells, d_store, dtype=dtype, pin_memory=True)
                    if need_sc:
                        self.scales_host[s] = torch.zeros(2, kvh, h_cells, d // 32, dtype=torch.float16, pin_memory=True)
                    self.vram_cells[s] = v_cells

        for s in range(n):
            c = self.canonical_slot[s]
            if c == s:
                continue
            self.slot_mode[s] = self.slot_mode[c]
            self.slot_family[s] = self.slot_family[c]
            self.slot_k_bits[s] = self.slot_k_bits[c]
            self.slot_v_bits[s] = self.slot_v_bits[c]
            self.slot_ring[s] = self.slot_ring[c]
            self.kv[s] = self.kv[c]
            self.scales[s] = self.scales[c]
            self.kv_host[s] = self.kv_host[c]
            self.scales_host[s] = self.scales_host[c]
            self.k_buf[s] = self.k_buf[c]
            self.k_sc[s] = self.k_sc[c]
            self.k_host[s] = self.k_host[c]
            self.k_sc_host[s] = self.k_sc_host[c]
            self.v_buf[s] = self.v_buf[c]
            self.v_sc[s] = self.v_sc[c]
            self.v_host[s] = self.v_host[c]
            self.v_sc_host[s] = self.v_sc_host[c]
            self.vram_cells[s] = self.vram_cells[c]

        self.rht_signs = default_rht_signs(d, device) if any_lm else None

        kv_dim, H = kvh * d, cfg.hidden_size
        self.conv_state = [
            [torch.zeros(3, kv_dim, device=device), torch.zeros(3, kv_dim, device=device),
             torch.zeros(3, H, device=device), torch.zeros(3, H, device=device)]
            for _ in range(n)
        ]
        self.pos = torch.zeros(1, dtype=torch.int32, device=device)
        self.seq_len = 0  # host mirror, updated by the caller after prefill/decode

    def kv_vram_bytes(self) -> int:
        seen: set[int] = set()
        total = 0
        for seq in (self.kv, self.scales, self.k_buf, self.k_sc, self.v_buf, self.v_sc):
            for t in seq:
                if t is not None and id(t) not in seen:
                    seen.add(id(t))
                    total += t.numel() * t.element_size()
        return total

    def kv_host_bytes(self) -> int:
        seen: set[int] = set()
        total = 0
        for seq in (self.kv_host, self.scales_host, self.k_host, self.k_sc_host, self.v_host, self.v_sc_host):
            for t in seq:
                if t is not None and id(t) not in seen:
                    seen.add(id(t))
                    total += t.numel() * t.element_size()
        return total

    def reset(self):
        self.pos.zero_()
        self.seq_len = 0
        for slot in self.conv_state:
            for c in slot:
                c.zero_()

    def snapshot_cpu(self) -> dict:
        """Copy active KV + conv state to CPU tensors for `CudaPromptCache`."""
        L = int(self.seq_len)
        tensors: dict[str, torch.Tensor] = {}
        slot_lens: list[int] = []
        for s in range(len(self.kv)):
            if self.is_sliding[s]:
                cap_s = self.kv[s].shape[2]
                n_save = min(L, cap_s) if not self.slot_ring[s] else cap_s
            else:
                n_save = L
            slot_lens.append(n_save)
            if self.canonical_slot[s] == s:
                vc = self.vram_cells[s]
                if self.slot_family[s] == "lm":
                    tensors[f"k_buf.{s}"] = _snapshot_pair(self.k_buf[s], self.k_host[s], n_save, vc, dim=1)
                    tensors[f"k_sc.{s}"] = _snapshot_pair(self.k_sc[s], self.k_sc_host[s], n_save, vc, dim=1)
                    tensors[f"v_buf.{s}"] = _snapshot_pair(self.v_buf[s], self.v_host[s], n_save, vc, dim=1)
                    tensors[f"v_sc.{s}"] = _snapshot_pair(self.v_sc[s], self.v_sc_host[s], n_save, vc, dim=1)
                else:
                    tensors[f"kv.{s}"] = _snapshot_pair(self.kv[s], self.kv_host[s], n_save, vc, dim=2)
                    if self.scales[s] is not None:
                        tensors[f"sc.{s}"] = _snapshot_pair(self.scales[s], self.scales_host[s], n_save, vc, dim=2)
            for ci, c_arr in enumerate(self.conv_state[s]):
                tensors[f"conv.{s}.{ci}"] = c_arr.detach().cpu().contiguous()
        return {
            "tensors": tensors,
            "meta": {
                "seq_len": L,
                "global_kv_mode": self.global_kv_mode,
                "global_kv_plan": self.global_kv_plan_str,
                "shared_loop_kv": self.shared_loop_kv,
                "slot_mode": list(self.slot_mode),
                "canonical_slot": list(self.canonical_slot),
                "slot_lens": slot_lens,
                "slot_ring": list(self.slot_ring),
            },
        }

    def restore_from_snapshot(self, snap: dict) -> None:
        """Restore active KV + conv state from a `snapshot_cpu()` dict."""
        meta = snap["meta"]
        if meta["global_kv_mode"] != self.global_kv_mode:
            raise ValueError(f"snapshot kv mode {meta['global_kv_mode']!r} != cache {self.global_kv_mode!r}")
        if bool(meta.get("shared_loop_kv", False)) != self.shared_loop_kv:
            raise ValueError(
                f"snapshot shared_loop_kv={bool(meta.get('shared_loop_kv', False))} != cache {self.shared_loop_kv}"
            )
        if meta.get("slot_mode") is not None and list(meta["slot_mode"]) != self.slot_mode:
            raise ValueError("snapshot slot_mode does not match cache slot_mode")
        L = int(meta["seq_len"])
        if L > self.capacity:
            raise ValueError(f"snapshot length {L} exceeds cache capacity {self.capacity}")
        tensors = snap["tensors"]
        for s in range(len(self.kv)):
            if self.canonical_slot[s] == s:
                vc = self.vram_cells[s]
                if self.slot_family[s] == "lm":
                    _restore_pair(self.k_buf[s], self.k_host[s], tensors[f"k_buf.{s}"], vc, dim=1)
                    _restore_pair(self.k_sc[s], self.k_sc_host[s], tensors[f"k_sc.{s}"], vc, dim=1)
                    _restore_pair(self.v_buf[s], self.v_host[s], tensors[f"v_buf.{s}"], vc, dim=1)
                    _restore_pair(self.v_sc[s], self.v_sc_host[s], tensors[f"v_sc.{s}"], vc, dim=1)
                else:
                    _restore_pair(self.kv[s], self.kv_host[s], tensors[f"kv.{s}"], vc, dim=2)
                    if self.scales[s] is not None:
                        _restore_pair(self.scales[s], self.scales_host[s], tensors[f"sc.{s}"], vc, dim=2)
            for ci in range(4):
                self.conv_state[s][ci].copy_(tensors[f"conv.{s}.{ci}"], non_blocking=True)
        self.seq_len = L
        self.pos.fill_(L)

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
        self.row_groups = {}
        self.nvfp4 = {}  # side -> (codes [E_nv, m, n/2] u8, block scales [E_nv, m, n/16] u8, outer [E_nv] f32, local [E] i32)
        for side, (m, n) in (("gu", (2 * cfg.intermediate_size, cfg.hidden_size)), ("dn", (cfg.hidden_size, cfg.intermediate_size))):
            grouped_key = f"{side}_rg_map"
            emap = np.asarray(arrays[grouped_key if grouped_key in arrays else f"{side}_map"], dtype=np.int64)
            groups = emap.shape[1] if emap.ndim == 3 else 1
            if emap.shape[0] != E or not groups or m % groups:
                raise ValueError(f"Invalid {side} expert row-group layout")
            self.row_groups[side] = groups
            rows = m // groups
            emap = emap.reshape(-1, 2)
            mats = []
            for fmt in FMT_PARAMS:
                if f"{side}_{fmt}_codes" not in arrays:
                    continue
                local = np.full(E * groups, -1, dtype=np.int64)
                sel = emap[:, 0] == FMT_ID[fmt]
                local[sel] = emap[sel, 1]
                codes = _t(arrays[f"{side}_{fmt}_codes"], device)
                mats.append((_Mat(fmt, rows, n, codes, _t(arrays[f"{side}_{fmt}_scales"], device), lut=luts[fmt]),
                             torch.from_numpy(local).to(device=device, dtype=torch.int32), codes.shape[0]))
            if f"{side}_nvfp4_codes" in arrays:
                local = np.full(E * groups, -1, dtype=np.int64)
                sel = emap[:, 0] == NVFP4_FID
                local[sel] = emap[sel, 1]
                self.nvfp4[side] = (_t(arrays[f"{side}_nvfp4_codes"], device), _t(arrays[f"{side}_nvfp4_scales"], device),
                                    _t(arrays[f"{side}_nvfp4_outer"], device).float(),
                                    torch.from_numpy(local).to(device=device, dtype=torch.int32))
            self.sides[side] = (m, n, mats)
        # decode: all experts of a side in one launch (trellis slabs 0-3, NVFP4 slab 4)
        self.unified = {}
        for side, (m, n, mats) in self.sides.items():
            groups = self.row_groups[side]
            if len(mats) > 4:
                raise ValueError(f"Triton supports up to four trellis formats per {side} projection")
            slab_of = torch.zeros(E * groups, dtype=torch.int32, device=device)
            loc_of = torch.zeros_like(slab_of)
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
                                             nvfp4=nv[:3] if nv is not None else None, groups=groups)
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
        if not grouped or self.row_groups[side] > 1:
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
        pk, pv = cache.conv_state[slot][0], cache.conv_state[slot][1]
        window = cfg.sliding_window_size if self.is_sliding else 0
        if cache.slot_family[slot] == "lm":
            k_buf, k_sc = cache.k_buf[slot], cache.k_sc[slot]
            v_buf, v_sc = cache.v_buf[slot], cache.v_sc[slot]
            lm_kw = dict(
                k_bits=cache.slot_k_bits[slot],
                v_bits=cache.slot_v_bits[slot],
                rht_signs=cache.rht_signs,
                ring=cache.slot_ring[slot],
                k_host=cache.k_host[slot],
                k_sc_host=cache.k_sc_host[slot],
                v_host=cache.v_host[slot],
                v_sc_host=cache.v_sc_host[slot],
                vram_cells=cache.vram_cells[slot],
            )
            if self.is_sliding:
                kv_conv_write_lm_(k_raw, v_raw, pk, pv, k_buf, k_sc, v_buf, v_sc, pos,
                                  cfg.qk_l2_clip_threshold, eps, nkv,
                                  w_k=self.k_conv, w_v=self.v_conv, **lm_kw)
            else:
                kv_conv_write_lm_(k_raw, v_raw, pk, pv, k_buf, k_sc, v_buf, v_sc, pos,
                                  cfg.qk_l2_clip_threshold, eps, nkv,
                                  dyn=(k_in, v_in, self.k_dyn, self.v_dyn), **lm_kw)
            if decode:
                rel, sinks = rel_state(rel_raw, self.rel_scale, q_in, self.sink_w, pos,
                                       cfg.rel_l2_clip_threshold, cfg.log_scaling_alpha, cfg.log_scaling_n_floor, full)
                tn = tuning_for(q.device)
                attn = decode_attention_lm(
                    q[0], k_buf, k_sc, v_buf, v_sc, rel[0], sinks[0], pos, window, cfg.rel_extent,
                    proj=self.rel_proj, nsplit=tn.attn_nsplit, block_k=tn.attn_block_k, **lm_kw,
                )[None]
            else:
                rel_logits, sinks = rel_post(rel_raw, self.rel_scale, self.rel_proj, q_in, self.sink_w, pos,
                                             cfg.rel_l2_clip_threshold, cfg.log_scaling_alpha, cfg.log_scaling_n_floor, full)
                L = int(pos[-1].item()) + 1  # prefill: host knows the length
                attn = prefill_attention_kv_lm(
                    q, k_buf, k_sc, v_buf, v_sc, rel_logits, sinks, pos, L, window, cfg.rel_extent, **lm_kw,
                )
        else:
            kv = cache.kv[slot]
            kv_kw = dict(
                ring=cache.slot_ring[slot],
                kv_mode=cache.slot_mode[slot],
                scales=cache.scales[slot],
                kv_host=cache.kv_host[slot],
                scales_host=cache.scales_host[slot],
                vram_cells=cache.vram_cells[slot],
            )
            if self.is_sliding:
                kv_conv_write_(k_raw, v_raw, pk, pv, kv, pos, cfg.qk_l2_clip_threshold, eps, nkv,
                               w_k=self.k_conv, w_v=self.v_conv, **kv_kw)
            else:
                kv_conv_write_(k_raw, v_raw, pk, pv, kv, pos, cfg.qk_l2_clip_threshold, eps, nkv,
                               dyn=(k_in, v_in, self.k_dyn, self.v_dyn), **kv_kw)

            if decode:
                rel, sinks = rel_state(rel_raw, self.rel_scale, q_in, self.sink_w, pos,
                                       cfg.rel_l2_clip_threshold, cfg.log_scaling_alpha, cfg.log_scaling_n_floor, full)
                tn = tuning_for(q.device)
                attn = decode_attention(q[0], kv, rel[0], sinks[0], pos, window, cfg.rel_extent, proj=self.rel_proj,
                                        nsplit=tn.attn_nsplit, block_k=tn.attn_block_k, **kv_kw)[None]
            else:
                rel_logits, sinks = rel_post(rel_raw, self.rel_scale, self.rel_proj, q_in, self.sink_w, pos,
                                             cfg.rel_l2_clip_threshold, cfg.log_scaling_alpha, cfg.log_scaling_n_floor, full)
                L = int(pos[-1].item()) + 1  # prefill: host knows the length
                attn = prefill_attention_kv(q, kv, rel_logits, sinks, pos, L, window, cfg.rel_extent, **kv_kw)
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

    def make_cache(
        self,
        capacity: int = 4096,
        *,
        global_kv_mode: str | None = None,
        kv_vram_cells: int | None = None,
        prefill_chunk: int = 1024,
        shared_loop_kv: bool | None = None,
        global_kv_plan: str | dict | None = None,
    ) -> Inkling2Cache:
        return Inkling2Cache(
            self.cfg, capacity, self.device,
            global_kv_mode=global_kv_mode, kv_vram_cells=kv_vram_cells, prefill_chunk=prefill_chunk,
            shared_loop_kv=shared_loop_kv, global_kv_plan=global_kv_plan,
        )

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

    def blocked_columns(self, token_ids) -> torch.Tensor | None:
        """Bucket-order columns of tokens `argmax` must never pick (None if there are none)."""
        ids = sorted({int(t) for t in token_ids or ()})
        if not ids:
            return None
        return self.lm_head.inv_order[torch.tensor(ids, device=self.device)]

    def argmax(self, h: torch.Tensor, blocked: torch.Tensor | None = None) -> torch.Tensor:
        """Greedy next tokens without the 200K gather (softcap is monotonic). `blocked` comes from `blocked_columns`."""
        lg = self.ordered_logits(h)
        if blocked is not None:
            lg = lg.index_fill(-1, blocked, float("-inf"))
        return self.order[lg.argmax(-1)]

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
