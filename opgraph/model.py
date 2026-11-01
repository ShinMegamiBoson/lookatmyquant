"""A minimal Llama-family forward pass over plain tensors: SmolLM2 by default, Qwen3 (per-head query/key RMSNorm) when
the environment variable OPGRAPH_ARCH names it, e.g. OPGRAPH_ARCH=qwen3-1.7b. The architecture's sizes are read from the
cached checkpoint's config.json when this module is imported, so every module that imports them sees the same model.

Every compressed model in this project is evaluated by this one function, so methods only ever change tensors.
The residual stream may carry shortcut matrices and a reduced width (SliceGPT); RMSNorm keeps dividing by the
original width (`mean_dim`), which is what makes slicing a truncation of the same network.
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

ARCHS = {  # name -> (Hugging Face repository, prefix of the token caches in cache/)
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", ""),
    "smollm-360m": ("HuggingFaceTB/SmolLM-360M", ""),  # same tokenizer as SmolLM2
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B-Base", "qwen3-"),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B-Base", "qwen3-"),
    "qwen3-4b": ("Qwen/Qwen3-4B-Base", "qwen3-"),
}
ARCH = os.environ.get("OPGRAPH_ARCH", "smollm2-360m")
REPO, DATA_PREFIX = ARCHS[ARCH]


def snapshot_dir() -> str:
    root = os.path.expanduser(f"~/.cache/huggingface/hub/models--{REPO.replace('/', '--')}/snapshots")
    return sorted(glob.glob(os.path.join(root, "*")))[-1]


def _config() -> dict:
    with open(os.path.join(snapshot_dir(), "config.json")) as f:
        return json.load(f)


_C = _config()
HIDDEN, HEADS, KV_HEADS, INTER, LAYERS = (_C["hidden_size"], _C["num_attention_heads"], _C["num_key_value_heads"],
                                          _C["intermediate_size"], _C["num_hidden_layers"])
HEAD_DIM = _C.get("head_dim") or HIDDEN // HEADS
EPS, THETA = _C["rms_norm_eps"], float(_C["rope_theta"])
QK_NORM = _C["model_type"] == "qwen3"
# Where models run: OPGRAPH_DEVICE, else an NVIDIA GPU, else Apple's GPU, else the CPU.
DEVICE = os.environ.get("OPGRAPH_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def empty_cache():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()
LINEARS = ("q", "k", "v", "o", "gate", "up", "down")
# Which captured signal feeds each linear layer.
INPUT_OF = {"q": "attn_in", "k": "attn_in", "v": "attn_in", "o": "o_in", "gate": "mlp_in", "up": "mlp_in", "down": "down_in"}


@dataclass
class Layer:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    attn_norm: torch.Tensor | None
    mlp_norm: torch.Tensor | None
    attn_shortcut: torch.Tensor | None = None  # h_mid = h @ attn_shortcut + attention(h)
    mlp_shortcut: torch.Tensor | None = None  # h_next = h_mid @ mlp_shortcut + mlp(h_mid)
    taps: dict | None = None  # point -> list of nodes applied to the activation there (Insert, Condition)
    skip: bool = False  # Replace(layer, identity)
    act: str = "silu"  # Replace(activation)
    kv_window: tuple | None = None  # Condition on the token index: (sink tokens, recent window)
    q_norm: torch.Tensor | None = None  # Qwen3: RMSNorm gain on each query head, before RoPE
    k_norm: torch.Tensor | None = None  # Qwen3: RMSNorm gain on each key head, before RoPE

    def copy(self) -> "Layer":
        return Layer(**{f.name: _clone(getattr(self, f.name)) for f in fields(self)})


@dataclass
class Model:
    embed: torch.Tensor
    layers: list[Layer]
    final_norm: torch.Tensor | None
    head: torch.Tensor
    mean_dim: int = HIDDEN

    def copy(self) -> "Model":
        head = self.head if self.head is self.embed else _clone(self.head)
        embed = _clone(self.embed)
        return Model(embed, [L.copy() for L in self.layers], _clone(self.final_norm), embed if head is self.embed else head, self.mean_dim)

    @property
    def device(self) -> torch.device:
        return self.embed.device


def _clone(t):
    if isinstance(t, torch.Tensor):
        return t.clone()
    if isinstance(t, dict):
        return {k: list(v) if isinstance(v, list) else v for k, v in t.items()}
    return t


def load(device: str | None = None) -> Model:
    device = device or DEVICE
    from safetensors.torch import load_file

    sd = {}
    for f in sorted(glob.glob(os.path.join(snapshot_dir(), "*.safetensors"))):
        sd.update(load_file(f))

    def g(key):
        return sd[key].to(torch.float32).to(device)

    layers = []
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        layers.append(
            Layer(
                q=g(p + "self_attn.q_proj.weight"),
                k=g(p + "self_attn.k_proj.weight"),
                v=g(p + "self_attn.v_proj.weight"),
                o=g(p + "self_attn.o_proj.weight"),
                gate=g(p + "mlp.gate_proj.weight"),
                up=g(p + "mlp.up_proj.weight"),
                down=g(p + "mlp.down_proj.weight"),
                attn_norm=g(p + "input_layernorm.weight"),
                mlp_norm=g(p + "post_attention_layernorm.weight"),
                q_norm=g(p + "self_attn.q_norm.weight") if QK_NORM else None,
                k_norm=g(p + "self_attn.k_norm.weight") if QK_NORM else None,
            )
        )
    embed = g("model.embed_tokens.weight")
    head = g("lm_head.weight") if "lm_head.weight" in sd and not _C.get("tie_word_embeddings") else embed  # tied: SmolLM2, Qwen3 ≤ 4B
    return Model(embed, layers, g("model.norm.weight"), head)


def rope(T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM))
    f = torch.outer(torch.arange(T, dtype=torch.float32, device=device), inv)
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


def _rot_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat([-b, a], -1)


def rms(x, gain, mean_dim):
    y = x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) / mean_dim + EPS)
    return y if gain is None else y * gain


def _tap(taps, point, x):
    if taps and point in taps:
        for node in taps[point]:
            x = node(x)
    return x


def window_mask(T: int, sinks: int, window: int, device) -> torch.Tensor:
    """Causal mask that keeps the first `sinks` tokens and the most recent `window` tokens (True = attend)."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    return (j <= i) & ((j < sinks) | (i - j < window))


def attention(x, q, k, v, o, cos, sin, taps=None, window=None, q_norm=None, k_norm=None):
    """Attention on already-normalised input. Returns (output, input to o_proj)."""
    B, T, _ = x.shape
    qh = (x @ q.T).view(B, T, -1, HEAD_DIM).transpose(1, 2)
    kh = (x @ k.T).view(B, T, -1, HEAD_DIM).transpose(1, 2)
    vh = (x @ v.T).view(B, T, -1, HEAD_DIM).transpose(1, 2)
    if q_norm is not None:
        qh, kh = rms(qh, q_norm, HEAD_DIM), rms(kh, k_norm, HEAD_DIM)
    qh = qh * cos + _rot_half(qh) * sin
    kh = kh * cos + _rot_half(kh) * sin
    kh, vh = _tap(taps, "k", kh), _tap(taps, "v", vh)
    rep = qh.shape[1] // kh.shape[1]
    kh, vh = kh.repeat_interleave(rep, 1), vh.repeat_interleave(rep, 1)
    if window is None:
        a = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
    else:
        a = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=window_mask(T, window[0], window[1], x.device))
    a = _tap(taps, "o_in", a.transpose(1, 2).reshape(B, T, -1))
    return a @ o.T, a


ACTS = {"silu": F.silu, "relu": F.relu, "gelu": F.gelu}


def mlp(x, gate, up, down, act="silu", taps=None):
    """SwiGLU on already-normalised input. Returns (output, input to down_proj)."""
    a = _tap(taps, "down_in", ACTS[act](x @ gate.T) * (x @ up.T))
    return a @ down.T, a


def attn_sublayer(h, L: Layer, cos, sin, mean_dim, cap=None):
    x = _tap(L.taps, "attn_in", rms(h, L.attn_norm, mean_dim))
    out, a = attention(x, L.q, L.k, L.v, L.o, cos, sin, L.taps, L.kv_window, L.q_norm, L.k_norm)
    if cap is not None:
        cap["attn_in"].append(x)
        cap["o_in"].append(a)
    return (h if L.attn_shortcut is None else h @ L.attn_shortcut) + out


def mlp_sublayer(hm, L: Layer, mean_dim, cap=None):
    x = _tap(L.taps, "mlp_in", rms(hm, L.mlp_norm, mean_dim))
    out, act = mlp(x, L.gate, L.up, L.down, L.act, L.taps)
    if cap is not None:
        cap["mlp_in"].append(x)
        cap["down_in"].append(act)
    return (hm if L.mlp_shortcut is None else hm @ L.mlp_shortcut) + out


def layer_forward(h, L: Layer, cos, sin, mean_dim, cap=None):
    if L.skip:
        return h
    return mlp_sublayer(attn_sublayer(h, L, cos, sin, mean_dim, cap), L, mean_dim, cap)


def batched(fn, h, bs: int = 16):
    return torch.cat([fn(h[b : b + bs]) for b in range(0, len(h), bs)])


@torch.no_grad()
def capture_layer(h, L: Layer, cos, sin, mean_dim, bs: int = 16):
    """Run one layer over hidden states `h`; return its output and the inputs each linear layer saw."""
    if L.skip:
        return h, {}
    cap = {key: [] for key in ("attn_in", "o_in", "mlp_in", "down_in")}
    out = torch.cat([layer_forward(h[b : b + bs], L, cos, sin, mean_dim, cap) for b in range(0, len(h), bs)])
    return out, {key: torch.cat(v) for key, v in cap.items()}


def forward_logits(model: Model, ids: torch.Tensor) -> torch.Tensor:
    """Differentiable forward pass (used by gradient Refit)."""
    cos, sin = rope(ids.shape[1], model.device)
    h = model.embed[ids]
    for L in model.layers:
        h = layer_forward(h, L, cos, sin, model.mean_dim)
    return rms(h, model.final_norm, model.mean_dim) @ model.head.T


@torch.no_grad()
def logits(model: Model, ids: torch.Tensor) -> torch.Tensor:
    cos, sin = rope(ids.shape[1], model.device)
    h = model.embed[ids]
    for L in model.layers:
        h = layer_forward(h, L, cos, sin, model.mean_dim)
    return rms(h, model.final_norm, model.mean_dim) @ model.head.T


@torch.no_grad()
def evaluate(model: Model, ids: torch.Tensor, base: Model | None = None, bs: int = 4) -> dict:
    """Perplexity on `ids`, and the mean per-token KL(base ‖ model) when a base model is given."""
    nll = kl = 0.0
    n_tok = n_pos = 0
    for b in range(0, len(ids), bs):
        x = ids[b : b + bs].to(model.device)
        lp = F.log_softmax(logits(model, x), -1)
        nll += -lp[:, :-1].gather(-1, x[:, 1:, None]).sum().item()
        n_tok += x[:, 1:].numel()
        if base is not None:
            lb = F.log_softmax(logits(base, x), -1)
            kl += (lb.exp() * (lb - lp)).sum().item()
            n_pos += x.numel()
    out = {"ppl": math.exp(nll / n_tok)}
    if base is not None:
        out["kl"] = kl / n_pos
    return out
