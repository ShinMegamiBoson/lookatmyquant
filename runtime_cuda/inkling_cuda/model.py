"""Serial Inkling 2 Mini text inference with CUDA packed projections.

Decoder equations and tensor names follow LookAtMyQuant's MIT-licensed
runtime_mlx/inkling/model.py. See LICENSE for the upstream notice.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import struct

import numpy as np

from .packed import EXPERT_FORMATS, Linear, Vocabulary, Weights, project, rotate


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Config:
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    unpadded_vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    d_rel: int
    rel_extent: int
    log_scaling_n_floor: int
    log_scaling_alpha: float
    rms_norm_eps: float
    qk_l2_clip_threshold: float
    rel_l2_clip_threshold: float
    local_layer_ids: list
    final_logit_softcapping: float
    sliding_window_size: int
    n_layers_per_attn_res_block: int
    n_routed_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    mlp_gate_softcap: float
    mlp_up_softcap: float
    route_scale: float
    loop_start: int
    loop_end: int
    loop_passes: int

    @classmethod
    def load(cls, filename):
        raw = read_json(filename)
        text = raw.get("text_config", raw)
        if text.get("model_type") not in ("inkling2_loop", "inkling2_loop_text"):
            raise ValueError("Expected an Inkling 2 loop text configuration")
        loop = text.get("loop_config", {})
        text = {
            **text,
            "unpadded_vocab_size": text.get("unpadded_vocab_size", text["vocab_size"]),
            "loop_start": loop.get("core_start", 6),
            "loop_end": loop.get("core_end", 18),
            "loop_passes": loop.get("num_passes", 3),
        }
        return cls(**{key: text[key] for key in cls.__dataclass_fields__})

    @property
    def total_visits(self):
        return self.num_hidden_layers + (self.loop_passes - 1) * (self.loop_end - self.loop_start)


def load_steering(path):
    """Read the float tensors used by this pack, including BF16 on NumPy."""
    dtypes = {"F16": "<f2", "BF16": "<u2", "F32": "<f4"}
    with path.open("rb") as fp:
        header_size, = struct.unpack("<Q", fp.read(8))
        if not 2 <= header_size <= 100_000_000:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(fp.read(header_size))
        file_size = path.stat().st_size
        start = 8 + header_size
        tensors = {}
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            dtype = np.dtype(dtypes[entry["dtype"]])
            shape = tuple(entry["shape"])
            lo, hi = entry["data_offsets"]
            if not (0 <= lo <= hi <= file_size - start and hi - lo == math.prod(shape) * dtype.itemsize):
                raise ValueError(f"Invalid safetensors offsets for {name}")
            fp.seek(start + lo)
            values = np.frombuffer(fp.read(hi - lo), dtype=dtype).reshape(shape)
            if entry["dtype"] == "BF16":
                values = (values.astype(np.uint32) << np.uint32(16)).view(np.float32)
            tensors[name] = values.astype(np.float32, copy=False)
        return tensors


def rms(x, weight=1.0, eps=1e-6):
    return x * np.reciprocal(np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(eps))) * weight


def clip_l2(x, threshold):
    length = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return x * np.minimum(np.float32(threshold) / (length + np.float32(1e-6)), np.float32(1))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -88, 88)))


def softmax(x):
    shifted = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return shifted / np.sum(shifted, axis=-1, keepdims=True)


def softcap(x, cap):
    return np.tanh(x / cap) * cap


def block_attention(query, blocks, partial):
    states = rms(np.stack([*blocks, partial]))
    query = query.reshape(-1, states.shape[-1])
    return softmax((query @ states.T) * np.float32(states.shape[-1] ** -0.5)) @ states


def round_bf16(x):
    bits = x.astype(np.float32, copy=False).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> np.uint32(16)) & np.uint32(1)))
            & np.uint32(0xffff0000)).view(np.float32)


class ExpertSide:
    def __init__(self, archive, weights, side, width, rows, count):
        self.signs = archive[f"{side}_signs_in"].astype(np.float32)
        grouped = f"{side}_rg_map" in archive
        key = f"{side}_rg_map" if grouped else f"{side}_map"
        raw = archive[key]
        if raw.ndim != (3 if grouped else 2) or raw.shape[0] != count or raw.shape[-1] != 2:
            raise ValueError(f"Invalid {side} expert map shape")
        self.mapping = np.ascontiguousarray(raw, dtype=np.int32).reshape(count, -1, 2)
        self.groups = self.mapping.shape[1]
        if not self.groups or rows % self.groups:
            raise ValueError(f"Invalid {side} row groups")
        self.rows = rows
        self.group_rows = rows // self.groups
        self.slabs = {}
        self.width = width
        for key, fmt in enumerate(EXPERT_FORMATS):
            prefix = f"{side}_{fmt}"
            if f"{prefix}_codes" in archive:
                self.slabs[key] = weights.slab(archive, prefix, fmt, width, self.group_rows)
        for fmt, loc in self.mapping.reshape(-1, 2):
            if fmt not in self.slabs or not 0 <= loc < self.slabs[fmt].experts:
                raise ValueError(f"Invalid or unsupported {side} expert format/location ({fmt}, {loc})")

    def __call__(self, x, ids, x_div):
        x_rot = rotate(x, self.signs)
        parts = []
        for group in range(self.groups):
            formats, locations = self.mapping[ids, group].T
            routes = [np.where(formats == key, locations, -1).astype(np.int32) for key in self.slabs]
            parts.append(project(x_rot, list(self.slabs.values()), routes, x_div))
        return parts[0] if self.groups == 1 else np.concatenate(parts, axis=-1)



class VisitCache:
    def __init__(self, capacity, kv_heads, head_dim):
        self.k = np.empty((kv_heads, capacity, head_dim), dtype=np.float32)
        self.v = np.empty_like(self.k)
        self.capacity = capacity
        self.convolutions = {}

    def push_kv(self, position, k, v):
        slot = position % self.capacity
        self.k[:, slot] = k
        self.v[:, slot] = v
        first = max(0, position + 1 - self.capacity)
        if first == 0:
            return self.k[:, :position + 1], self.v[:, :position + 1], np.arange(position, -1, -1)
        # Attention is permutation equivariant once relative distance follows each slot.
        distance = (slot - np.arange(self.capacity)) % self.capacity
        return self.k, self.v, distance

    def convolution(self, slot, x, weights, dynamic=None):
        taps = weights.shape[0]
        history = self.convolutions.get(slot)
        if history is None:
            history = np.zeros((taps, x.size), dtype=np.float32)
            self.convolutions[slot] = history
        history[:-1] = history[1:]
        history[-1] = x
        if dynamic is None:
            return (x + np.sum(history * weights, axis=0)) * np.float32((taps + 1) ** -0.5)
        return (x + np.sum(history[::-1] * (weights + dynamic), axis=0)) * np.float32((2 * taps + 1) ** -0.5)


class Decoder:
    def __init__(self, folder, weights, steering, cfg, index):
        self.cfg = cfg
        self.sliding = index in cfg.local_layer_ids
        self.boundary = index % cfg.n_layers_per_attn_res_block == 0
        self.w = {name.removeprefix(f"model.layers.{index}."): value for name, value in steering.items()
                  if name.startswith(f"model.layers.{index}.")}
        for name, weight in self.w.items():
            if "_message_gate." in name:
                self.w[name] = round_bf16(weight)
        prefix = folder / f"layer_{index:02d}"
        meta = read_json(prefix.with_name(prefix.name + "_meta.json"))
        with np.load(prefix.with_name(prefix.name + "_packed.npz"), allow_pickle=False) as archive:
            self.linear = {name.removeprefix("self_attn."): Linear(weights, archive, entry)
                           for name, entry in meta["linears"].items()}
            h, d, e = cfg.hidden_size, cfg.intermediate_size, cfg.n_routed_experts
            self.gu = ExpertSide(archive, weights, "gu", h, 2 * d, e)
            self.dn = ExpertSide(archive, weights, "dn", d, h, e)

    def message_gate(self, x, message, name):
        prefix = f"{name}_message_gate."
        down = self.w[prefix + "down_proj.weight"]
        forget = self.w[prefix + "forget_gate_proj.weight"]
        add = self.w[prefix + "message_gate_proj.weight"]
        # MLX computes and rounds the two low-rank projections separately.
        h = x.size
        hidden_rank = round_bf16(down[:, :h] @ round_bf16(rms(x)))
        message_rank = round_bf16(down[:, h:] @ round_bf16(rms(message)))
        z = round_bf16(hidden_rank + message_rank)
        forget = np.tanh(round_bf16(forget @ z))
        add = np.tanh(round_bf16(add @ z))
        return (1 - forget) * x + add * message

    def kv_conv(self, x, normalized, name, cache, slot):
        prefix = f"self_attn.{name}_proj.sconv."
        if self.sliding:
            return cache.convolution(slot, x, self.w[prefix + "conv1d.weight"][:, 0, :].T)
        static = self.w[prefix + "static_weight"]
        taps = static.shape[0]
        dynamic = (self.w[prefix + "weight_proj.weight"] @ normalized).reshape(-1, taps).T
        dynamic = np.repeat(dynamic, self.cfg.head_dim, axis=1)
        return cache.convolution(slot, x, static, dynamic)

    def attention(self, q_in, k_in, v_in, position, cache):
        cfg = self.cfg
        q = self.linear["q_proj.linear"](q_in).reshape(cfg.num_attention_heads, cfg.head_dim)
        q = clip_l2(q * self.w["self_attn.q_proj.norm.scale"], cfg.qk_l2_clip_threshold)
        k = self.kv_conv(self.linear["k_proj.linear"](k_in), k_in, "k", cache, 0)
        v = self.kv_conv(self.linear["v_proj.linear"](v_in), v_in, "v", cache, 1)
        k = clip_l2(k.reshape(cfg.num_key_value_heads, cfg.head_dim), cfg.qk_l2_clip_threshold)
        v = rms(v.reshape(cfg.num_key_value_heads, cfg.head_dim), eps=cfg.rms_norm_eps)
        k, v, distances = cache.push_kv(position, k, v)
        rel = self.linear["relative_position_bias.linear"](q_in).reshape(cfg.num_attention_heads, cfg.d_rel)
        rel = clip_l2(rel * self.w["self_attn.relative_position_bias.norm.scale"], cfg.rel_l2_clip_threshold)
        if not self.sliding:
            factor = np.float32(1 + cfg.log_scaling_alpha * math.log(max((position + 1) / cfg.log_scaling_n_floor, 1)))
            q *= factor
            rel *= factor
        rel = rel @ self.w["self_attn.relative_position_bias.proj"]
        pos_bias = np.zeros((cfg.num_attention_heads, distances.size), dtype=np.float32)
        valid = distances < cfg.rel_extent
        pos_bias[:, valid] = rel[:, distances[valid]]
        repeat = cfg.num_attention_heads // cfg.num_key_value_heads
        groups = (cfg.num_key_value_heads, repeat)
        q = q.reshape(*groups, cfg.head_dim)
        scores = (q @ k.swapaxes(-1, -2)) * np.float32(cfg.head_dim ** -0.5)
        scores += pos_bias.reshape(*groups, -1)
        sinks = (self.w["self_attn.sink_proj.weight"] @ q_in).reshape(*groups, 1)
        probs = softmax(np.concatenate((scores, sinks), axis=-1))[..., :-1]
        out = (probs @ v).reshape(-1)
        gate = sigmoid(self.linear["o_proj.gate_proj"](q_in))
        output = self.linear["o_proj.linear"](out * gate)
        return cache.convolution(2, output, self.w["post_attention_convolution.conv1d.weight"][:, 0, :].T)

    def moe(self, x):
        cfg = self.cfg
        scores = sigmoid(self.w["mlp.gate.weight"] @ x)
        ranked = scores + self.w["mlp.gate.e_score_correction_bias"]
        ids = np.argsort(-ranked, kind="stable")[:cfg.num_experts_per_tok]
        weights = scores[ids]
        weights = weights / (weights.sum() + np.float32(1e-20)) * cfg.route_scale
        gate, up = np.split(self.gu(x[None, :], ids, ids.size), 2, axis=-1)
        act = softcap(gate, cfg.mlp_gate_softcap) * sigmoid(gate) * softcap(up, cfg.mlp_up_softcap)
        return weights @ self.dn(act, ids, 1)

    def __call__(self, hidden, blocks, position, cache, messages):
        eps = self.cfg.rms_norm_eps
        query = self.w.get("attention_input.query")
        inputs = (hidden, hidden, hidden)
        if query is not None:
            reads = block_attention(query, blocks, hidden)
            inputs = (reads[0],) * 3 if self.sliding else reads
        normalized = []
        for i, (name, h) in enumerate(zip("qkv", inputs, strict=True)):
            if messages is not None:
                h = self.message_gate(h, messages[i], name)
            norm = "input_layernorm.weight" if self.sliding else f"{name}_input_layernorm.weight"
            normalized.append(rms(h, self.w[norm], eps))
        a = self.attention(*normalized, position, cache)
        residual = a if self.boundary else hidden + a
        local_blocks = [*blocks, hidden] if self.boundary else blocks
        h = block_attention(self.w["mlp_input.query"], local_blocks, residual)[0]
        if messages is not None:
            h = self.message_gate(h, messages[3], "mlp")
        h = rms(h, self.w["post_attention_layernorm.weight"], eps)
        return residual + cache.convolution(3, self.moe(h), self.w["post_mlp_convolution.conv1d.weight"][:, 0, :].T)


class Model:
    def __init__(self, folder, cuda, device_embedding=False):
        folder = Path(folder)
        self.cfg = cfg = Config.load(folder / "config.json")
        self.steering = load_steering(folder / "steering.safetensors")
        weights = Weights(folder, cuda)
        self.embedding = Vocabulary(folder, weights, "embed", cfg.hidden_size, device_embedding)
        self.head = Vocabulary(folder, weights, "head", cfg.hidden_size)
        self.layers = [Decoder(folder, weights, self.steering, cfg, i) for i in range(cfg.num_hidden_layers)]
        if self.head.order.size != cfg.unpadded_vocab_size:
            raise ValueError("Packed head count differs from the configured unpadded vocabulary")

    def session(self, context):
        return Session(self, context)


class Session:
    def __init__(self, model, context):
        if context <= 0:
            raise ValueError("Context size must be positive")
        self.model = model
        self.context = context
        self.position = 0
        self.caches = {}

    def step(self, token, logits=True):
        if self.position >= self.context:
            raise ValueError("Context limit reached")
        model = self.model
        cfg, st = model.cfg, model.steering
        h = rms(model.embedding.embed(token), st["model.embed_tokens.embed_norm.weight"], cfg.rms_norm_eps)
        blocks = []

        def visit(index, slot, hidden, messages=None):
            layer = model.layers[index]
            if slot not in self.caches:
                capacity = min(self.context, cfg.sliding_window_size) if layer.sliding else self.context
                self.caches[slot] = VisitCache(capacity, cfg.num_key_value_heads, cfg.head_dim)
            new = layer(hidden, blocks, self.position, self.caches[slot], messages)
            if layer.boundary:
                blocks.append(hidden)
            return new

        for i in range(cfg.loop_start):
            h = visit(i, i, h)
        start_h, start_blocks = h, blocks.copy()
        core = range(cfg.loop_start, cfg.loop_end)
        for i in core:
            h = visit(i, i, h)
        for pass_index in range(1, cfg.loop_passes):
            previous_h = h
            previous_blocks = blocks[len(start_blocks) + 1:]
            h, blocks = start_h, start_blocks.copy()
            for j, i in enumerate(core):
                messages = block_attention(st["model.message_input.query"][j], previous_blocks, previous_h)
                slot = cfg.num_hidden_layers + (pass_index - 1) * len(core) + j
                h = visit(i, slot, h, messages)
        for i in range(cfg.loop_end, cfg.num_hidden_layers):
            h = visit(i, i, h)
        self.position += 1
        if logits:
            final = rms(block_attention(st["model.final_input.query"], blocks, h)[0], st["model.norm.weight"], cfg.rms_norm_eps)
            return softcap(model.head.logits(final), cfg.final_logit_softcapping)
        return None
