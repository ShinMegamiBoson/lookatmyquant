"""PyTorch modules that run LookAtMyQuant packed Inkling-2-Mini weights inside transformers' own Inkling2Loop model.

`load_reference` builds `transformers.Inkling2LoopForCausalLM` on the meta device, loads the small steering tensors
(norms, convolutions, routers, message gates, attention-residual queries), and swaps in:

- `PackedLinear` for the six attention linears of every layer,
- `PackedExperts` for each layer's 128 experts (decoded on the fly, one expert at a time),
- `PackedEmbedding` for the token embedding and `PackedLMHead` for the unembedding.

Everything else is transformers' code, so this path is a reference: correct by construction outside the decoders,
and slow. The standalone runtime (`model.py`) is checked against it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .formats import FMT_PARAMS, ID_TO_FMT, NVFP4_FID, decode_lm, decode_nvfp4, decode_trellis, rht_in, rht_out
from .schema import validate_packed_dir


def _t(a: np.ndarray, device) -> torch.Tensor:
    a = np.asarray(a)
    if a.dtype == np.uint32:
        a = a.view(np.int32)
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def _signs(a: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a).astype(np.float32)).to(device)


class PackedLinear(nn.Module):
    """`y = rht_out(rht_in(x, signs_in) @ Q.T, signs_out)`, with `Q` decoded once to float32 at load."""

    def __init__(self, meta: dict, arrays, luts: dict[str, np.ndarray], device):
        super().__init__()
        s, m, n, fmt = meta["safe"], int(meta["m"]), int(meta["n"]), meta["fmt"]
        self.fmt, self.in_features, self.out_features = fmt, n, m
        codes, scales = _t(arrays[f"lin_{s}_codes"], device), _t(arrays[f"lin_{s}_scales"], device)
        if fmt.startswith("lm"):
            Q = decode_lm(codes, scales, _t(arrays[f"lin_{s}_levels"], device), m, n, int(meta["k"]))
        else:
            Q = decode_trellis(codes.reshape(m, -1), scales, _t(luts[fmt], device), n, fmt)
        self.register_buffer("Q", Q.contiguous(), persistent=False)
        self.register_buffer("signs_in", _signs(arrays[f"lin_{s}_signs_in"], device), persistent=False)
        so = _signs(arrays[f"lin_{s}_signs_out"], device) if meta.get("out_rot") else None
        self.register_buffer("signs_out", so, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = rht_in(x, self.signs_in) @ self.Q.T
        if self.signs_out is not None:
            y = rht_out(y, self.signs_out)
        return y.to(x.dtype)

    def extra_repr(self) -> str:
        return f"{self.fmt}, in={self.in_features}, out={self.out_features}, out_rot={self.signs_out is not None}"


class PackedExperts(nn.Module):
    """Drop-in for `Inkling2LoopExperts`: same `forward(hidden_states, top_k_index, top_k_weights)`."""

    def __init__(self, arrays, luts: dict[str, np.ndarray], act_fn: nn.Module, hidden: int, inter: int, device):
        super().__init__()
        self.hidden, self.inter = hidden, inter
        self.act_fn = act_fn
        self.maps = {}
        for side in ("gu", "dn"):
            key = f"{side}_rg_map" if f"{side}_rg_map" in arrays else f"{side}_map"
            raw = np.asarray(arrays[key], dtype=np.int64)
            self.maps[side] = raw.reshape(raw.shape[0], -1, 2)
        self.num_experts = self.maps["gu"].shape[0]
        self.slabs: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {"gu": {}, "dn": {}}
        for side in ("gu", "dn"):
            for fmt in FMT_PARAMS:
                if f"{side}_{fmt}_codes" in arrays:
                    self.slabs[side][fmt] = (
                        _t(arrays[f"{side}_{fmt}_codes"], device),
                        _t(arrays[f"{side}_{fmt}_scales"], device),
                    )
            if f"{side}_nvfp4_codes" in arrays:  # NVFP4 experts read the unrotated activation
                self.slabs[side]["nvfp4"] = tuple(_t(arrays[f"{side}_nvfp4_{p}"], device) for p in ("codes", "scales", "outer"))
        self.luts = {k: _t(v, device) for k, v in luts.items()}
        self.gu_signs = _signs(arrays["gu_signs_in"], device)
        self.dn_signs = _signs(arrays["dn_signs_in"], device)

    def decode(self, side: str, e: int) -> tuple[torch.Tensor, bool]:
        """(dense weight, whether it reads the rotated input)."""
        parts = []
        for fid, loc in self.maps[side][e]:
            if int(fid) == NVFP4_FID:
                codes, scales, outer = self.slabs[side]["nvfp4"]
                return decode_nvfp4(codes[loc], scales[loc], outer[loc]), False
            fmt = ID_TO_FMT[int(fid)]
            codes, scales = self.slabs[side][fmt]
            n = self.hidden if side == "gu" else self.inter
            parts.append(decode_trellis(codes[loc], scales[loc], self.luts[fmt], n, fmt))
        return (parts[0] if len(parts) == 1 else torch.cat(parts)), True

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(hidden_states.shape[0], self.hidden, dtype=torch.float32, device=hidden_states.device)
        x_rot = rht_in(hidden_states, self.gu_signs)
        for e in torch.unique(top_k_index).tolist():
            tok, pos = torch.where(top_k_index == e)
            W, rotated = self.decode("gu", e)
            gate_up = (x_rot[tok] if rotated else hidden_states[tok].float()) @ W.T
            gate, up = gate_up.chunk(2, dim=-1)
            act = self.act_fn(gate, up)
            W, rotated = self.decode("dn", e)
            y = (rht_in(act, self.dn_signs) if rotated else act.float()) @ W.T
            out.index_add_(0, tok, y * top_k_weights[tok, pos, None].float())
        return out.to(hidden_states.dtype)


class _Buckets(nn.Module):
    """Rows of an embedding or head, stored in buckets of one format each, in `order` (row p holds token order[p])."""

    def __init__(self, packed_dir: Path, kind: str, luts, device, dense_dtype=torch.float32):
        super().__init__()
        meta = json.loads((packed_dir / f"{kind}_meta.json").read_text())
        npz = np.load(packed_dir / f"{kind}_packed.npz")
        self.hidden = int(np.asarray(npz[f"b{meta[0]['kb']}_signs_in"]).shape[-1])  # one sign per input column
        order = np.asarray(npz["order"], dtype=np.int64)
        Qs, signs, lo = [], [], 0
        self.bounds = []
        for bm in meta:
            kb, rows, fmt = bm["kb"], int(bm["rows"]), bm["opt"]
            scales = _t(npz[f"b{kb}_scales"], device)
            if fmt.startswith("lm"):
                Q = decode_lm(_t(npz[f"b{kb}_codes"], device), scales, _t(npz[f"b{kb}_levels"], device), rows, self.hidden, bm["k"])
            else:
                codes = _t(npz[f"b{kb}_codes"], device).reshape(rows, -1)
                lut = _t(luts[fmt], device)
                # in row chunks: the decoder's int64 temporaries for a 70k-row bucket alone exceed 1 GB
                Q = torch.cat([decode_trellis(codes[r : r + 8192], scales[r : r + 8192], lut, self.hidden, fmt)
                               for r in range(0, rows, 8192)])
            Qs.append(Q.to(dense_dtype))
            signs.append(_signs(npz[f"b{kb}_signs_in"], device))
            self.bounds.append((lo, lo + rows))
            lo += rows
        self.Qs = Qs
        self.signs = signs
        self.register_buffer("order", torch.from_numpy(order).to(device), persistent=False)
        inv = torch.empty_like(self.order)
        inv[self.order] = torch.arange(len(order), device=device)
        self.register_buffer("inv_order", inv, persistent=False)


class PackedEmbedding(nn.Module):
    """Drop-in for `Inkling2LoopNormedEmbedding`: decoded rows, rotated back, then transformers' `embed_norm`."""

    def __init__(self, packed_dir: Path, luts, embed_norm: nn.Module, device):
        super().__init__()
        self.b = _Buckets(packed_dir, "embed", luts, device)
        self.embed_norm = embed_norm
        # one dense table [V, H] in the original basis, bf16 (0.9 GB) - built once
        rows = []
        for Q, s in zip(self.b.Qs, self.b.signs):
            rows.append(rht_out(Q.float(), s).to(torch.bfloat16))
        table = torch.cat(rows, 0)[self.b.inv_order]  # token-indexed
        self.register_buffer("weight", table, persistent=False)
        self.b.Qs = []  # free

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(F.embedding(input_ids, self.weight))


class PackedLMHead(nn.Module):
    """Drop-in for `lm_head`: per-bucket `rht_in(x, s_b) @ Q_b.T`, concatenated, then put back in token order."""

    def __init__(self, packed_dir: Path, luts, device):
        super().__init__()
        self.b = _Buckets(packed_dir, "head", luts, device)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.reshape(-1, h.shape[-1])
        parts = [rht_in(x, s) @ Q.T for Q, s in zip(self.b.Qs, self.b.signs)]
        logits = torch.cat(parts, dim=-1)[:, self.b.inv_order]
        return logits.reshape(*h.shape[:-1], -1)


LINEAR_NAMES = (
    "self_attn.q_proj.linear",
    "self_attn.k_proj.linear",
    "self_attn.v_proj.linear",
    "self_attn.o_proj.linear",
    "self_attn.o_proj.gate_proj",
    "self_attn.relative_position_bias.linear",
)


def _set_module(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parent, _, leaf = dotted.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, leaf, new)


@torch.no_grad()
def load_reference(packed_dir: str | Path, device: str | torch.device = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Returns `(model, tokenizer)`: a transformers `Inkling2LoopForCausalLM` running from the packed file, every
    packed matrix decoded to dense float32. Slow; the standalone runtime (`model.py`) is checked against it."""
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM

    packed_dir = Path(packed_dir)
    device = torch.device(device)
    validate_packed_dir(packed_dir)
    cfg = AutoConfig.from_pretrained(packed_dir)
    tcfg = getattr(cfg, "text_config", cfg)
    tcfg._attn_implementation = "eager"
    # params declared without a dtype become `dtype`; the ones transformers pins to float32 stay float32
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = Inkling2LoopForCausalLM(tcfg)
    finally:
        torch.set_default_dtype(prev)

    luts = dict(np.load(packed_dir / "luts.npz"))
    Lin = lambda meta, arrays: PackedLinear(meta, arrays, luts, device)  # noqa: E731
    Exp = lambda arrays, act, inter: PackedExperts(arrays, luts, act, tcfg.hidden_size, inter, device)  # noqa: E731
    L = tcfg.num_hidden_layers
    replaced = set()
    for i in range(L):
        meta = json.loads((packed_dir / f"layer_{i:02d}_meta.json").read_text())
        arrays = np.load(packed_dir / f"layer_{i:02d}_packed.npz")
        arrays = {k: arrays[k] for k in arrays.files}
        layer = f"model.layers.{i}"
        for name in LINEAR_NAMES:
            _set_module(model, f"{layer}.{name}", Lin(meta["linears"][name], arrays))
            replaced.add(f"{layer}.{name}.")
        old = model.get_submodule(f"{layer}.mlp.experts")
        _set_module(model, f"{layer}.mlp.experts", Exp(arrays, old.act_fn, old.intermediate_dim))
        replaced.add(f"{layer}.mlp.experts.")

    embed_norm = model.model.embed_tokens.embed_norm
    model.model.embed_tokens = PackedEmbedding(packed_dir, luts, embed_norm, device)
    model.lm_head = PackedLMHead(packed_dir, luts, device)
    replaced.update({"lm_head.", "model.embed_tokens.weight"})

    # fill what is left on the meta device from steering.safetensors
    st = load_file(str(packed_dir / "steering.safetensors"), device=str(device))
    missing = []
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        if not p.is_meta:
            continue
        if name not in st:
            missing.append(name)
            continue
        new = st[name].to(p.dtype)
        assert new.shape == p.shape, (name, new.shape, p.shape)
        _assign(model, name, new, isinstance(p, nn.Parameter))
    if missing:
        raise KeyError(f"{len(missing)} tensors not in steering.safetensors, e.g. {missing[:8]}")
    unused = set(st) - {n for n, _ in model.named_parameters()} - {n for n, _ in model.named_buffers()}
    if unused:
        raise KeyError(f"steering tensors with no home in the model: {sorted(unused)[:8]}")
    model.eval()
    tok = AutoTokenizer.from_pretrained(packed_dir)
    return model, tok


def _assign(model: nn.Module, name: str, value: torch.Tensor, is_param: bool) -> None:
    mod_name, _, leaf = name.rpartition(".")
    mod = model.get_submodule(mod_name)
    if is_param:
        mod._parameters[leaf] = nn.Parameter(value, requires_grad=False)
    else:
        mod._buffers[leaf] = value
