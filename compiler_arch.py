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

import hashlib
import json
import os
import random
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
def discover(model, sample, counts_out=None):
    """Per layer, a list of stages in execution order. A stage is {"kind": "linear", "names": [paths in the layer],
    "keep": [small linears read with them]} or {"kind": "experts", "name": path}. `sample` is a (1, T) token tensor.
    When `counts_out` is a list, it is filled with each decoder layer's execution count (>1 for looped layers)."""
    _, layers = decoder_layers(model)
    calls = []  # (layer, path, kind, input key) in call order
    counts = [0] * len(layers)
    hooks = []
    for i, L in enumerate(layers):
        hooks.append(L.register_forward_pre_hook(lambda mod, a, i=i: counts.__setitem__(i, counts[i] + 1)))
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
    if counts_out is not None:
        counts_out[:] = counts
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
        return self.sample("calib", n, T, seed)[0]

    def sample(self, split, n, T, seed):
        """Stratified windows and their source/token manifest; every prefix stays balanced.

        Sources are interleaved, rather than concatenated and truncated. A seeded
        source order distributes the remainder without permanently favoring the
        first alphabetic sources. The input token streams remain unchanged.
        """
        if n < 0 or T <= 0 or not self.sources:
            raise ValueError("sampling needs sources, a nonnegative count and positive window length")
        order = list(self.sources)
        random.Random(seed).shuffle(order)
        gen = torch.Generator().manual_seed(seed)
        starts, positions = {}, {s: 0 for s in self.sources}
        per = -(-n // len(order))
        for source in self.sources:
            ids = self.ids[source][split]
            if len(ids) < T:
                raise ValueError(f"{source}/{split}: fewer than {T} tokens")
            starts[source] = torch.randint(0, len(ids) - T + 1, (per,), generator=gen).tolist()
        windows, manifest = [], []
        for i in range(n):
            source = order[i % len(order)]
            start = starts[source][positions[source]]
            positions[source] += 1
            tokens = self.ids[source][split][start:start + T]
            windows.append(tokens)
            manifest.append({"source": source, "split": split, "start": start, "length": T,
                             "tokens_sha256": hashlib.sha256(tokens.to(torch.int64).contiguous().numpy().tobytes()).hexdigest()})
        if not windows:
            return torch.empty((0, T), dtype=torch.long), manifest
        return torch.stack(windows), manifest


def is_router(m):
    """A router module with `top_k`, `num_experts` and a weight tensor, returning (logits, scores, indices)."""
    return hasattr(m, "top_k") and hasattr(m, "num_experts") and isinstance(getattr(m, "weight", None), torch.Tensor)


def router_beside(L, experts_path):
    """The router of the experts module at `experts_path` in layer L: its sibling, or None."""
    parent = L.get_submodule(experts_path.rsplit(".", 1)[0]) if "." in experts_path else L
    return next((m for m in parent.children() if is_router(m)), None)


def same_choices(a, b):
    """Mean share of each token's chosen experts in `a` that also appear in `b`."""
    return float((a.unsqueeze(-1) == b.to(a.device).unsqueeze(-2)).any(-1).float().mean())


class FixedRouting:
    """Pass-indexed fixed expert routing for sequential and looped MoEs (supports softmax and sigmoid/`score_fn` routers)."""

    def __init__(self, model):
        self.routers = [m for m in model.modules() if is_router(m)]
        self.kept, self.mode, self.key = {}, None, None
        self.differ = {}
        self.text = None
        self._pass = [0] * len(self.routers)
        self.hooks = [m.register_forward_hook(lambda mod, a, out, i=i: self._hook(i, mod, out)) for i, m in enumerate(self.routers)]

    def _reset_passes(self):
        self._pass = [0] * len(self.routers)

    def _hook(self, i, mod, out):
        if self.mode is None:
            return None
        p = self._pass[i]
        self._pass[i] = p + 1
        logits, scores_out, indices = out
        k = (i, p, self.key)
        if self.mode == "record":
            self.kept[k] = indices.detach().cpu()
            return None
        if self.mode == "watch":
            tally = self.differ.setdefault((i, p, self.text), [0.0, 0])
            tally[0] += (1 - same_choices(indices, self.kept[k])) * indices.numel()
            tally[1] += indices.numel()
            return None
        indices = self.kept[k].to(logits.device)
        if hasattr(mod, "score_fn"):
            scores = mod.score_fn(logits.float()).gather(-1, indices)
            scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
            scores = scores * getattr(mod, "routed_scaling_factor", 1.0)
        else:
            scores = torch.softmax(logits, dtype=torch.float, dim=-1).gather(-1, indices)
            if getattr(mod, "norm_topk_prob", False):
                scores = scores / scores.sum(dim=-1, keepdim=True)
        return logits, scores.to(scores_out.dtype), indices

    def record(self, key):
        self._reset_passes()
        self.mode, self.key = "record", key

    def replay(self, key):
        self._reset_passes()
        self.mode, self.key = "replay", key

    def watch(self, key, text=None):
        self._reset_passes()
        self.mode, self.key, self.text = "watch", key, text

    def differing(self):
        texts = list(dict.fromkeys(t for _, _, t in self.differ))
        out = {}
        for t in texts:
            vals = []
            for i in range(len(self.routers)):
                d_tot = sum(v[0] for (ri, _, tt), v in self.differ.items() if ri == i and tt == t)
                n_tot = sum(v[1] for (ri, _, tt), v in self.differ.items() if ri == i and tt == t)
                vals.append(d_tot / n_tot if n_tot > 0 else None)
            out[t] = vals
        self.differ = {}
        return out

    def differing_by_pass(self):
        texts = list(dict.fromkeys(t for _, _, t in self.differ))
        out = {
            t: {f"L{i}.p{p}": v[0] / v[1] for (i, p, tt), v in sorted(self.differ.items()) if tt == t and v[1] > 0}
            for t in texts
        }
        self.differ = {}
        return out

    def off(self):
        self._reset_passes()
        self.mode = None

    def remove(self):
        for h in self.hooks:
            h.remove()


def damped_hessian(H_raw: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """Damped float64 Hessian H_raw + damp * mean(diag(H_raw)) * I (identity fallback if degenerate)."""
    H = H_raw.detach().cpu().double()
    n = H.shape[0]
    if n == 0:
        return H
    mean_d = float(torch.diag(H).mean().item())
    if not (mean_d > 0.0) or not bool(torch.isfinite(H).all()):
        return torch.eye(n, dtype=torch.float64)
    return H + (damp * mean_d) * torch.eye(n, dtype=torch.float64)


@torch.no_grad()
def dense_down_moments(
    mod,
    e: int,
    X: torch.Tensor,
    transposed: bool,
    expert_hidden_fn,
    *,
    chunk_size: int = 4096,
    rmod=None,
    Xr: torch.Tensor | None = None,
    damp: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """All-token chunked input Hessian H (and optional cross-term D = E[h_ref^T h_q] - E[h_q^T h_q]) for expert `e`'s down_proj."""
    n_tok = len(X)
    dev = mod.gate_up_proj.device
    dtype = mod.gate_up_proj.dtype
    need_cross = rmod is not None and Xr is not None
    r_dev = rmod.gate_up_proj.device if need_cross else None
    r_dtype = rmod.gate_up_proj.dtype if need_cross else None
    G_qq: torch.Tensor | None = None
    G_rq: torch.Tensor | None = None
    for s in range(0, n_tok, chunk_size):
        xb = X[s : s + chunk_size].to(device=dev, dtype=dtype)
        hq = expert_hidden_fn(mod, e, xb, transposed).float()
        qq = (hq.T @ hq).double().cpu()
        G_qq = qq if G_qq is None else (G_qq + qq)
        if need_cross:
            xrb = Xr[s : s + chunk_size].to(device=r_dev, dtype=r_dtype)
            hr = expert_hidden_fn(rmod, e, xrb, transposed).float().to(hq.device)
            rq = (hr.T @ hq).double().cpu()
            G_rq = rq if G_rq is None else (G_rq + rq)
    assert G_qq is not None and n_tok > 0
    H_raw = G_qq / float(n_tok)
    H = damped_hessian(H_raw, damp=damp)
    D = (G_rq / float(n_tok) - H_raw) if G_rq is not None else None
    return H, D


DOMAIN_CLASSES = ("prose", "code", "tools", "agentic", "reasoning", "multilingual")
_MULTILINGUAL_SOURCES = (
    "arb_Arab", "cmn_Hani", "deu_Latn", "fra_Latn", "hin_Deva",
    "jpn_Jpan", "kor_Hang", "por_Latn", "rus_Cyrl", "spa_Latn",
)
_TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read lines from a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep_search", "description": "Search files for a pattern.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "glob": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace a target snippet in a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "target": {"type": "string"}, "replacement": {"type": "string"}}, "required": ["path", "target", "replacement"]}}},
    {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]


def _render_chat(tok, messages: list[dict], tools: list[dict] | None = None) -> str:
    """Render structured chat/tool/reasoning turns using `tok.apply_chat_template` when available, with clean fallbacks."""
    if getattr(tok, "chat_template", None) and hasattr(tok, "apply_chat_template"):
        if tools:
            try:
                return tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
            except Exception:
                pass
        flat_msgs = []
        if tools:
            flat_msgs.append({"role": "system", "content": "Available tools:\n" + json.dumps(tools, separators=(",", ":"))})
        for m in messages:
            parts = []
            if m.get("reasoning_content"):
                parts.append(f"<think>\n{m['reasoning_content']}\n</think>")
            if m.get("content"):
                parts.append(str(m["content"]))
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc["function"]
                    parts.append("<tool_call>" + json.dumps({"name": fn["name"], "arguments": fn.get("arguments", {})}, separators=(",", ":")) + "</tool_call>")
            role = "user" if m["role"] == "tool" else m["role"]
            prefix = f"[tool:{m.get('name', 'tool')}] " if m["role"] == "tool" else ""
            flat_msgs.append({"role": role, "content": prefix + "\n".join(parts)})
        try:
            return tok.apply_chat_template(flat_msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    chunks = []
    if tools:
        chunks.append("System:\nTools: " + json.dumps(tools, separators=(",", ":")))
    for m in messages:
        hdr = f"Tool ({m.get('name', 'tool')}):" if m["role"] == "tool" else f"{m['role'].capitalize()}:"
        body = []
        if m.get("reasoning_content"):
            body.append(f"Thought: {m['reasoning_content']}")
        if m.get("content"):
            body.append(str(m["content"]))
        for tc in m.get("tool_calls") or ():
            fn = tc["function"]
            body.append(f"Call {fn['name']}({json.dumps(fn.get('arguments', {}), separators=(',', ':'))})")
        chunks.append(hdr + "\n" + "\n".join(body))
    return "\n\n".join(chunks)


def _templated_docs_for_split(tok, raw: dict[str, list[str]], split: str, calib_frac: float) -> dict[str, list[str]]:
    def take(name: str) -> list[str]:
        docs = [d for d in raw.get(name, []) if isinstance(d, str) and len(d.strip()) > 30]
        if not docs:
            return []
        cut = max(1, int(len(docs) * calib_frac))
        return docs[:cut] if split == "calib" else (docs[cut:] or docs[-1:])

    en_docs = take("en") or next((take(k) for k in sorted(raw) if take(k)), [])
    chat_docs = take("chat") or en_docs
    code_docs = take("code") or en_docs
    rng = random.Random(101 if split == "calib" else 202)
    lim = 120 if split == "calib" else 48

    prose = []
    for i, d in enumerate(en_docs[:lim]):
        ex = d.strip()[:1600]
        if i % 3 == 0:
            prose.append(ex)
        else:
            mid = max(80, min(len(ex) // 2, 500))
            prose.append(_render_chat(tok, [
                {"role": "user", "content": f"Continue and summarize:\n\n{ex[:mid]}"},
                {"role": "assistant", "content": ex[mid:] or ex[:mid]},
            ]))
    for d in chat_docs[: lim // 2]:
        lines = [ln.strip() for ln in d.strip()[:1600].splitlines() if ln.strip()]
        if len(lines) >= 2:
            m = len(lines) // 2
            prose.append(_render_chat(tok, [
                {"role": "user", "content": "\n".join(lines[:m])},
                {"role": "assistant", "content": "\n".join(lines[m:])},
            ]))

    code = []
    for i, d in enumerate(code_docs[:lim]):
        lines = d.strip()[:1800].splitlines()
        if not lines:
            continue
        cut = max(2, min(len(lines) // 2, 24))
        a, b = "\n".join(lines[:cut]), "\n".join(lines[cut : cut + 24]) or "\n".join(lines[:cut])
        if i % 3 == 0:
            code.append(_render_chat(tok, [
                {"role": "user", "content": f"```python\n{a}\n```\nCopy these lines verbatim."},
                {"role": "assistant", "content": a},
            ]))
        else:
            code.append(_render_chat(tok, [
                {"role": "user", "content": f"Review and continue this implementation:\n\n```\n{a}\n```"},
                {"role": "assistant", "reasoning_content": f"Tracing {cut} lines for edge cases and types.", "content": f"```\n{b}\n```"},
            ]))

    tools_out = []
    paths = ("src/server.py", "src/cache.py", "src/alloc.py", "tests/test_kv.py")
    for i in range(min(lim, max(len(code_docs), 24))):
        src = code_docs[i % max(len(code_docs), 1)][:700] if code_docs else "def step(x):\n    return x + 1\n"
        fp = paths[i % len(paths)]
        sym = f"fn_{i % 23}"
        tools_out.append(_render_chat(tok, [
            {"role": "system", "content": "Use tools before answering about repository files."},
            {"role": "user", "content": f"Locate `{sym}` in `{fp}` and summarize it."},
            {"role": "assistant", "reasoning_content": f"Search `{fp}` with grep_search.",
             "tool_calls": [{"id": f"c{i}_1", "type": "function", "function": {"name": "grep_search", "arguments": {"pattern": sym, "glob": "*.py"}}}]},
            {"role": "tool", "tool_call_id": f"c{i}_1", "name": "grep_search", "content": json.dumps({"path": fp, "line": 12 + i % 40})},
            {"role": "assistant", "tool_calls": [{"id": f"c{i}_2", "type": "function", "function": {"name": "read_file", "arguments": {"path": fp, "start_line": 1, "end_line": 60}}}]},
            {"role": "tool", "tool_call_id": f"c{i}_2", "name": "read_file", "content": src},
            {"role": "assistant", "content": f"`{sym}` in `{fp}` handles the block shown above."},
        ], tools=_TOOL_SCHEMAS))

    agentic = []
    for i in range(min(lim // 2 + 12, max(len(code_docs), 20))):
        src = code_docs[(i * 3) % max(len(code_docs), 1)][:1000] if code_docs else "def solve(xs):\n    return sum(xs)\n"
        lines = [ln for ln in src.splitlines() if ln.strip()] or ["return None"]
        tgt = lines[min(len(lines) - 1, i % len(lines))]
        fp, tp = f"pkg/mod_{i % 11}.py", f"tests/test_mod_{i % 11}.py"
        agentic.append(_render_chat(tok, [
            {"role": "system", "content": "Locate the bug, edit minimally, and run tests."},
            {"role": "user", "content": f"Fix the failing check in `{fp}` and verify `{tp}`."},
            {"role": "assistant", "reasoning_content": f"Run `{tp}` first to reproduce.",
             "tool_calls": [{"id": f"a{i}_1", "type": "function", "function": {"name": "run_shell", "arguments": {"command": f"pytest -q {tp}"}}}]},
            {"role": "tool", "tool_call_id": f"a{i}_1", "name": "run_shell", "content": f"FAILED {tp}::test_case_{i % 7}\nAssertionError in {fp}"},
            {"role": "assistant", "tool_calls": [{"id": f"a{i}_2", "type": "function", "function": {"name": "read_file", "arguments": {"path": fp, "start_line": 1, "end_line": 80}}}]},
            {"role": "tool", "tool_call_id": f"a{i}_2", "name": "read_file", "content": src},
            {"role": "assistant", "tool_calls": [{"id": f"a{i}_3", "type": "function", "function": {"name": "edit_file", "arguments": {"path": fp, "target": tgt, "replacement": tgt + "  # fixed"}}}]},
            {"role": "tool", "tool_call_id": f"a{i}_3", "name": "edit_file", "content": "OK"},
            {"role": "assistant", "tool_calls": [{"id": f"a{i}_4", "type": "function", "function": {"name": "run_shell", "arguments": {"command": f"pytest -q {tp}"}}}]},
            {"role": "tool", "tool_call_id": f"a{i}_4", "name": "run_shell", "content": "4 passed in 0.21s"},
            {"role": "assistant", "content": f"Fixed `{fp}` and confirmed `{tp}` passes."},
        ], tools=_TOOL_SCHEMAS))

    reasoning = []
    for i in range(max(lim // 2, 24)):
        a = 17 + ((i * 13 + (0 if split == "calib" else 5)) % 83)
        b = 11 + ((i * 7 + (0 if split == "calib" else 3)) % 59)
        c = 5 + (i % 19)
        prod, ans = a * b, a * b - c
        reasoning.append(_render_chat(tok, [
            {"role": "user", "content": f"Compute {a} * {b} - {c} step by step."},
            {"role": "assistant",
             "reasoning_content": f"Split {a} * {b} = {a} * {(b // 10) * 10} + {a} * {b % 10} = {a * (b // 10) * 10} + {a * (b % 10)} = {prod}. Subtract {c}: {prod} - {c} = {ans}.",
             "content": f"The result is {ans}."},
        ]))

    multi = []
    m_keys = [k for k in _MULTILINGUAL_SOURCES if k in raw] or [k for k in sorted(raw) if k not in ("en", "chat", "code")]
    for k in m_keys:
        for i, d in enumerate(take(k)[: max(6, lim // max(len(m_keys), 1))]):
            ex = d.strip()[:1200]
            if i % 2 == 0:
                multi.append(ex)
            else:
                mid = max(60, len(ex) // 2)
                multi.append(_render_chat(tok, [
                    {"role": "user", "content": ex[:mid]},
                    {"role": "assistant", "content": ex[mid:] or ex[:mid]},
                ]))
    if not multi:
        multi = list(prose[: max(8, len(prose) // 2)])

    out = {"prose": prose, "code": code, "tools": tools_out, "agentic": agentic, "reasoning": reasoning, "multilingual": multi}
    for v in out.values():
        rng.shuffle(v)
    return {k: v for k, v in out.items() if v}


class TemplatedMix(TextMix):
    """6-class chat-templated calibration & evaluation mix (`prose`, `code`, `tools`, `agentic`, `reasoning`, `multilingual`)."""

    def __init__(self, tok, tag, root=TEXT, calib_frac=0.7, raw_docs: dict[str, list[str]] | None = None):
        self.root = Path(root)
        cache = self.root / f"tok_templated_{tag}.pt"
        if raw_docs is None and cache.exists():
            self.ids = torch.load(cache)
        else:
            raw = raw_docs if raw_docs is not None else {f.stem: json.loads(f.read_text())["docs"] for f in sorted(self.root.glob("*.json"))}
            if not raw:
                raise FileNotFoundError(f"no text mix JSON files found under {self.root}")
            eos = [tok.eos_token_id] if getattr(tok, "eos_token_id", None) is not None else []

            def enc(ds: list[str]) -> torch.Tensor:
                try:
                    batch_ids = tok(ds, add_special_tokens=False).input_ids
                except TypeError:
                    batch_ids = tok(ds).input_ids
                return torch.tensor([t for ids in batch_ids for t in ids + eos], dtype=torch.long)

            calib_by = _templated_docs_for_split(tok, raw, "calib", calib_frac)
            eval_by = _templated_docs_for_split(tok, raw, "eval", calib_frac)
            self.ids = {c: {"calib": enc(calib_by[c]), "eval": enc(eval_by.get(c) or calib_by[c])} for c in DOMAIN_CLASSES if c in calib_by}
            if raw_docs is None:
                self.root.mkdir(parents=True, exist_ok=True)
                torch.save(self.ids, cache)
        self.sources = [c for c in DOMAIN_CLASSES if c in self.ids] + sorted(k for k in self.ids if k not in DOMAIN_CLASSES)
        self.classes = list(self.sources)
        self.weight = {s: 1 for s in self.sources}
