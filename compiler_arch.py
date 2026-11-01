"""What the compression compiler needs to know about a model, discovered from the model itself rather than from names.

    decoder_layers   the ModuleList of decoder layers: the one whose length is the text config's layer count, outside
                     any vision or audio tower
    per_layer_embedding  a second lookup table, read once per token for every layer (Gemma 4's E-models), or None
    discover         each layer's stages, in execution order: one forward pass with hooks on every nn.Linear and every
                     experts module (stacked 3-D expert weights, as transformers 5 stores MoE experts). Linears that read
                     the same tensor form one stage (they share input statistics), so fused projections (Phi's qkv_proj),
                     other naming (GPT-NeoX's query_key_value) and hybrid layers (Qwen3.5's DeltaNet) need no table.
                     Linears with fewer than 64 inputs or outputs (gates onto a few heads, a shared expert's gate) and
                     MoE routers (a linear onto the experts) are kept at 16 bits and counted.
    head_and_embedding   the input embedding and the output head, and whether they are one tensor
    KVCache          a simulated quantized key/value cache for any model that calls transformers' attention interface:
                     keys and values as the cache stores them (after RoPE and any per-head norm), each token's vector per
                     head rotated by a randomized Hadamard, scaled by its RMS (16 bits per head), rounded to Gaussian
                     Lloyd–Max levels, rotated back; the first position exact
    TextMix          the text beyond English (modal_fetch.py): token frequencies with every source weighted equally,
                     calibration windows drawn evenly across sources, held-out evaluation windows per source
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import torch
import torch.nn as nn

SMALL = 64  # linears with fewer inputs or outputs than this stay at 16 bits
KNOWN_BLOCKS = {"linear_attn": 1, "self_attn": 2, "mlp": 3}  # quantize.BLOCK_SEED, so known models keep their seeds
TEXT = Path("/root/.cache/huggingface/compiler_text")


def text_config(cfg):
    return getattr(cfg, "text_config", None) or cfg


def decoder_layers(model):
    """(path, ModuleList) of the decoder layers."""
    n = text_config(model.config).num_hidden_layers
    cands = [(p, m) for p, m in model.named_modules() if isinstance(m, nn.ModuleList) and len(m) == n
             and not any(w in p for w in ("vision", "visual", "audio", "mtp", "encoder"))
             and any(isinstance(x, nn.Linear) for x in m[0].modules())]
    if not cands:
        raise ValueError("no decoder layer list found")
    return min(cands, key=lambda c: len(c[0]))


def head_and_embedding(model):
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    return emb, head, head is not None and head.weight is emb.weight


def per_layer_embedding(model):
    """Gemma 4's per-layer input table (vocabulary × layers·256), looked up per token: an nn.Embedding other than the
    input embedding, outside any vision or audio tower, with as many rows as the vocabulary."""
    emb = model.get_input_embeddings()
    V = emb.weight.shape[0]
    for p, m in model.named_modules():
        if isinstance(m, nn.Embedding) and m is not emb and m.weight.shape[0] == V and "per_layer" in p \
                and not any(w in p for w in ("vision", "visual", "audio")):
            return m
    return None


def is_experts(m):
    """A transformers 5 experts module: stacked expert weights gate_up_proj and down_proj, 3-D."""
    return (isinstance(getattr(m, "gate_up_proj", None), torch.Tensor) and getattr(m, "gate_up_proj").dim() == 3
            and isinstance(getattr(m, "down_proj", None), torch.Tensor) and getattr(m, "down_proj").dim() == 3)


def block_seed(path):
    head = path.split(".")[0]
    return KNOWN_BLOCKS.get(head, 4 + zlib.crc32(head.encode()) % 5)


@torch.no_grad()
def discover(model, sample):
    """Per layer, a list of stages in execution order. A stage is {"kind": "linear", "names": [paths in the layer],
    "keep": [small linears read with them]} or {"kind": "experts", "name": path}. `sample` is a (1, T) token tensor."""
    _, layers = decoder_layers(model)
    calls = []  # (layer, path, kind, input key) in call order
    hooks = []
    for i, L in enumerate(layers):
        for path, m in L.named_modules():
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_pre_hook(  # the tensor itself: kept alive, so identity is not a reused address
                    lambda mod, a, i=i, path=path: calls.append((i, path, "linear", a[0]))))
            elif is_experts(m):
                hooks.append(m.register_forward_pre_hook(
                    lambda mod, a, k, i=i, path=path: calls.append((i, path, "experts", None)), with_kwargs=True))
    try:
        model(sample, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    expert_paths = {(i, p) for i, p, kind, _ in calls if kind == "experts"}
    n_exp = getattr(text_config(model.config), "num_experts", None) or getattr(text_config(model.config), "num_local_experts", None)
    out = [[] for _ in layers]
    seen = set()
    for i, path, kind, key in calls:
        if (i, path) in seen:
            continue
        seen.add((i, path))
        if kind == "experts":
            out[i].append({"kind": "experts", "name": path})
            continue
        if any(path.startswith(p + ".") for j, p in expert_paths if j == i):
            continue  # a linear inside an experts module (none in transformers 5), handled with it
        mod = layers[i].get_submodule(path)
        small = min(mod.in_features, mod.out_features) < SMALL or "router" in path or (n_exp and mod.out_features == n_exp)
        st = next((s for s in out[i] if s["kind"] == "linear" and s["key"] is key), None)
        if st is None:
            st = {"kind": "linear", "names": [], "keep": [], "key": key}
            out[i].append(st)
        (st["keep"] if small else st["names"]).append(path)
    for stages in out:
        for s in stages:
            s.pop("key", None)
        stages[:] = [s for s in stages if s["kind"] == "experts" or s["names"] or s["keep"]]
    return out


def needs_bos(model):
    """Gemma models break without a beginning-of-sequence token (perplexity in the tens of thousands), so their windows
    start with one, as llama.cpp's perplexity tool does; other families keep the published convention (WikiText-2 cut into
    windows, BOS only at the start of the text), so earlier results stay comparable."""
    return "gemma" in model.config.model_type


def with_bos(windows, tok, model):
    """Windows whose first token is BOS, for models that need it (the window keeps its length)."""
    if not needs_bos(model) or tok.bos_token_id is None:
        return windows
    w = windows.clone()
    w[..., 0] = tok.bos_token_id
    return w


def stage_list_text(stages):
    return " → ".join("+".join(s["names"]) + (f" (kept {'+'.join(s['keep'])})" if s.get("keep") else "") if s["kind"] == "linear"
                      else f"[experts {s['name']}]" for s in stages)


_ACTIVE = []  # KVCache instances whose models' attention modules are rounded
_PATCHED = []


def _install():
    """Wrap transformers' attention dispatch once: every attention module calls ALL_ATTENTION_FUNCTIONS.get_interface at
    forward time (eager included, as the default), so the wrapper sees the keys and values as the cache would hold them."""
    if _PATCHED:
        return
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    orig = type(ALL_ATTENTION_FUNCTIONS).get_interface

    def get_interface(impl, default):
        fn = orig(ALL_ATTENTION_FUNCTIONS, impl, default)

        def wrapped(module, query, key, value, *a, **k):
            for c in _ACTIVE:
                l = c.layer_of.get(id(module))
                if l is not None:
                    key, value = c._round((l, "k"), key), c._round((l, "v"), value)
                    break
            return fn(module, query, key, value, *a, **k)
        return wrapped
    ALL_ATTENTION_FUNCTIONS.get_interface = get_interface
    _PATCHED.append(True)


class KVCache:
    """A simulated quantized key/value cache for one model. `widths` maps (layer, "k" | "v") to bits; absent or 16 =
    exact. Only this model's attention modules are rounded; another model in the process is left exact."""

    def __init__(self, model, widths=None, seed=0):
        from opgraph import nested as N
        from opgraph.codes import rht
        self.N = N
        self.widths, self.levels = dict(widths or {}), {}
        _, layers = decoder_layers(model)
        attn = [(i, m) for i, L in enumerate(layers) for m in L.modules() if type(m).__name__.endswith("Attention")]
        # head size and key/value heads per layer, from the modules (Gemma 4 sets them per layer)
        self.hd, self.n_kv = {}, {}
        for i, m in attn:
            hd = getattr(m, "head_dim", None)
            if hd is None:
                tc = text_config(model.config)
                hd = tc.hidden_size // tc.num_attention_heads
            self.hd[i] = hd
            k = getattr(m, "k_proj", None)
            if k is not None:
                self.n_kv[i] = k.out_features // hd
            elif hasattr(m, "query_key_value"):  # GPT-NeoX: one fused projection, as many key/value heads as heads
                self.n_kv[i] = m.query_key_value.out_features // (3 * hd)
        # KV-shared layers (Gemma 4) store no cache: they read the last unshared layer of their type, so they take its
        # widths (the same rounding, so the same cached values) and only the source layers are priced
        source = {getattr(m, "layer_type", None): i for i, m in attn if getattr(m, "store_full_length_kv", False)}
        self.layer_of = {id(m): (source[m.layer_type] if getattr(m, "is_kv_shared_layer", False) else i) for i, m in attn}
        self.layers = sorted(set(self.layer_of.values()))
        self.R = {(l, k): rht(self.hd[l], seed + 2 * l + (k == "v")).float() for l in self.layers for k in "kv"}
        _install()
        _ACTIVE.append(self)

    def _round(self, key, x):
        b = self.widths.get(key)
        if not b or b >= 16:
            return x
        if b not in self.levels:
            lv = self.N.lloyd_max(b).float().to(x.device)
            self.levels[b] = (lv, (lv[1:] + lv[:-1]) / 2)
        lv, mid = self.levels[b]
        R = self.R[key] = self.R[key].to(x.device)
        y = x.float() @ R.T  # (batch, heads, tokens, head dim)
        s = y.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-12)
        q = ((lv[torch.bucketize(y / s, mid)] * s) @ R).to(x.dtype)
        q[:, :, 0] = x[:, :, 0]
        return q

    def bits_per_token(self, b, l):
        """Cache bits per token for layer l's keys (or values) at width b: codes and a 16-bit scale per head."""
        return self.n_kv[l] * (self.hd[l] * 16 if b >= 16 else self.hd[l] * b + 16)

    def bits_per_value(self):
        keys = [(l, k) for l in self.layers for k in "kv"]
        return sum(self.bits_per_token(self.widths.get(key) or 16, key[0]) for key in keys) / sum(self.n_kv[l] * self.hd[l] for l, _ in keys)

    def remove(self):
        if self in _ACTIVE:
            _ACTIVE.remove(self)


class TextMix:
    """The text mix on the volume, tokenized once per tokenizer (cached beside it)."""

    def __init__(self, tok, tag, root=TEXT, calib_frac=0.7):
        self.root = root
        cache = root / f"tok_{tag}.pt"
        if cache.exists():
            self.ids = torch.load(cache)
        else:
            self.ids = {}
            for f in sorted(root.glob("*.json")):
                docs = json.loads(f.read_text())["docs"]
                cut = int(len(docs) * calib_frac)
                eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
                enc = lambda ds: torch.tensor([t for ids in tok(ds).input_ids for t in ids + eos])
                self.ids[f.stem] = {"calib": enc(docs[:cut]), "eval": enc(docs[cut:])}
            torch.save(self.ids, cache)
        self.sources = sorted(self.ids)

    def frequency(self, V):
        """Token frequency with every source weighted equally (each source's counts normalized to sum to 1)."""
        f = torch.zeros(V, dtype=torch.float64)
        for s in self.sources:
            c = torch.bincount(self.ids[s]["calib"].clamp(max=V - 1), minlength=V)[:V].double()
            f += c / c.sum()
        return f / len(self.sources)

    def windows(self, split, n_per_source, T, seed):
        out = {}
        g = torch.Generator().manual_seed(seed)
        for s in self.sources:
            ids = self.ids[s][split]
            starts = torch.randint(0, len(ids) - T - 1, (n_per_source,), generator=g)
            out[s] = torch.stack([ids[a : a + T] for a in starts])
        return out

    def calibration(self, n, T, seed=0):
        """n windows of T tokens drawn evenly across sources (seed 0, as the WikiText calibration)."""
        per = max(1, -(-n // len(self.sources)))
        w = self.windows("calib", per, T, seed)
        return torch.cat([w[s] for s in self.sources])[:n]
