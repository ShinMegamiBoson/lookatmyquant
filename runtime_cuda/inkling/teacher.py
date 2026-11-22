"""Loads the original `thinkingmachines/Inkling-2-Mini` (NVFP4 / MXFP8 experts) on one 20 GB GPU through transformers.

Dequantized to BF16 the model is ~53 GB. Here transformers loads everything as usual except the experts: its
`Inkling2LoopDequantizeExperts` converter is patched to keep the raw quantized tensors, and each expert is
dequantized to BF16 when a token routes to it, with the same math, so outputs equal the BF16 model's.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import core_model_loading as cml
from transformers.models.inkling2_loop import modeling_inkling2_loop as mi

_RAW: dict[str, dict[str, torch.Tensor]] = {}
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _stash_convert(self, input_dict, source_patterns, target_patterns, model=None, full_layer_name=None, **kwargs):
    tensors = {k: (v[0] if isinstance(v, list) else v) for k, v in input_dict.items()}
    weight, scale, amax, den = (tensors.get(p) for p in source_patterns)
    # the loader has cast every source to the param dtype (bf16); those casts are exact, so undo them to save memory
    if amax is not None:  # NVFP4: uint8 code bytes, float8 block scales
        weight, scale = weight.to(torch.uint8), scale.to(torch.float8_e4m3fn)
    else:  # MXFP8: float8 values, uint8 power-of-two exponents
        weight, scale = weight.to(torch.float8_e4m3fn), scale.to(torch.uint8)
    _RAW[full_layer_name] = {"weight": weight, "scale": scale, "amax": amax, "den": den}
    E = weight.shape[0]
    return {target_patterns[0]: torch.zeros(E, 2, 1, dtype=torch.bfloat16)}


class LazyExperts(torch.nn.Module):
    """`Inkling2LoopExperts` with placeholder params; real weights come from `_RAW`, dequantized per expert."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = torch.nn.Parameter(torch.zeros(self.num_experts, 2, 1))
        self.down_proj = torch.nn.Parameter(torch.zeros(self.num_experts, 2, 1))
        self.act_fn = mi.Inkling2LoopSoftcappedSwiGLU(config)
        self._raw = None

    def bind(self, name: str, device):
        self._raw = {}
        for side in ("gate_up_proj", "down_proj"):
            r = _RAW.pop(f"{name}.{side}")
            self._raw[side] = {k: (v.to(device) if v is not None else None) for k, v in r.items()}
        self._lut = torch.tensor(E2M1, device=device)

    def _dequant(self, side: str, e: int, dtype) -> torch.Tensor:
        r = self._raw[side]
        w, s = r["weight"][e], r["scale"][e].float()
        rows = w.shape[0]
        if r["amax"] is not None:  # NVFP4: low nibble is the even element
            codes = w.to(torch.uint8).long()
            vals = torch.stack((self._lut[codes & 0xF], self._lut[codes >> 4]), dim=-1).view(*s.shape, 16)
            amax = r["amax"].float().reshape(-1)
            den = r["den"].float().reshape(-1)
            factor = (amax[e] if amax.numel() > 1 else amax[0]) / (den[e] if den.numel() > 1 else den[0])
            W = (vals * s[..., None] * factor).reshape(rows, -1)
        else:  # MXFP8
            W = (w.float().view(*s.shape, 32) * torch.exp2(s - 127)[..., None]).reshape(rows, -1)
        W = W.to(dtype)
        if side == "gate_up_proj":  # transformers' Interleave(dim=1)
            W = W.reshape(rows // 2, 2, -1).transpose(0, 1).reshape(rows, -1)
        return W

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states, dtype=torch.float32)
        for e in torch.unique(top_k_index).tolist():
            if e >= self.num_experts:
                continue
            tok, pos = torch.where(top_k_index == e)
            h = self.act_fn(*F.linear(hidden_states[tok], self._dequant("gate_up_proj", e, hidden_states.dtype)).chunk(2, dim=-1))
            h = F.linear(h, self._dequant("down_proj", e, hidden_states.dtype))
            final.index_add_(0, tok, (h * top_k_weights[tok, pos, None]).to(torch.float32))
        return final.to(hidden_states.dtype)


@torch.no_grad()
def load_teacher(path: str, device: str = "cuda:0", dtype=torch.bfloat16):
    from transformers import AutoTokenizer

    orig_convert = cml.Inkling2LoopDequantizeExperts.convert
    orig_experts = mi.Inkling2LoopExperts
    cml.Inkling2LoopDequantizeExperts.convert = _stash_convert
    mi.Inkling2LoopExperts = LazyExperts
    try:
        model = mi.Inkling2LoopForConditionalGeneration.from_pretrained(
            path, dtype=dtype, device_map={"": device}, attn_implementation="eager"
        )
    finally:
        cml.Inkling2LoopDequantizeExperts.convert = orig_convert
        mi.Inkling2LoopExperts = orig_experts
    import re
    from pathlib import Path

    from safetensors import safe_open

    ckpt = safe_open(str(sorted(Path(path).glob("model-*.safetensors"))[0]), "pt")
    for name, mod in model.named_modules():
        if isinstance(mod, LazyExperts):
            mod.bind(name, device)
            # the loader cast the float32 per-expert NVFP4 factors to bf16; read them again at full precision
            layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
            for side, src in (("gate_up_proj", "w13"), ("down_proj", "w2")):
                key = f"model.llm.layers.{layer}.mlp.experts.{src}_weight"
                if mod._raw[side]["amax"] is not None:
                    mod._raw[side]["amax"] = ckpt.get_tensor(key + "_amax").float().to(device)
                    mod._raw[side]["den"] = ckpt.get_tensor(key + "_outer_scale_denominator").float().to(device)
    assert not _RAW, f"unbound expert tensors: {list(_RAW)[:4]}"
    model.eval()
    return model, AutoTokenizer.from_pretrained(path)
