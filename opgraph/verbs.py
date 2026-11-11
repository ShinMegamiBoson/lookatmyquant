"""The rest of the grammar's verbs, and the types they need.

Types (adjectives)
    SET(levels)            a finite value set: NF4, FP4 (E2M1), FP8 (E4M3) or any other
    CODEBOOK(K, d)         vectors of length d drawn from a K-entry codebook
    ONEHOT(K)              stored indices into a codebook (log2 K bits each)
    mask patterns          unstructured, n:m, channel, block (see frame.mask)

Verbs
    tie_slices, tie_cores          Tie: make slices of a tensor, or whole tensors, share one copy (lossy unless equal)
    restrict_set                   Restrict onto a finite value set
    restrict_codebook              Restrict vectors onto a codebook (lossy); split_codebook then factors exactly
    split_codebook                 Split into a one-hot assignment and a codebook (exact once vectors are codewords)
    refit_gradient                 Refit with a scope, an objective and a solver beyond least squares
    insert, rotate_at_runtime      Insert a fixed node on an activation bond; a gauge paid for at runtime
    restrict_stochastic            Randomize: stochastic rounding
    condition                      a node whose mask depends on the activations; kv_window acts on the token index
    replace_layer, replace_activation, replace_core   Replace part of the network with something else
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .frame import REAL, Core, Domain, ModelNet, Net, _real, act_on_label, check_pattern
from .model import LAYERS, LINEARS, Model, evaluate, forward_logits, rope

# ----------------------------------------------------------------------------- types


def SET(levels, name: str | None = None, bits: float | None = None) -> Domain:
    lv = tuple(sorted({float(x) for x in levels}))
    return Domain("set", float(math.ceil(math.log2(len(lv)))) if bits is None else bits, lv[0], lv[-1], levels=lv)


def ONEHOT(K: int, axis: str = "k") -> Domain:
    return Domain("onehot", math.log2(K), 0.0, float(K - 1), axis=axis)


def CODEBOOK(K: int, d: int, axis: str) -> Domain:
    return Domain("codebook", math.log2(K) / d, axis=axis)


def fp_levels(exp_bits: int, man_bits: int, bias: int, nan_top: bool = False) -> tuple:
    """Every finite value of a small floating-point format. nan_top drops the all-ones exponent-and-mantissa code (E4M3FN)."""
    vals = set()
    for e in range(2**exp_bits):
        for m in range(2**man_bits):
            if nan_top and e == 2**exp_bits - 1 and m == 2**man_bits - 1:
                continue
            mag = (m / 2**man_bits) * 2 ** (1 - bias) if e == 0 else (1 + m / 2**man_bits) * 2 ** (e - bias)
            vals |= {mag, -mag}
    return tuple(sorted(vals))


NF4_LEVELS = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
              -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
              0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0)  # QLoRA's NormalFloat4
FP4_E2M1 = fp_levels(2, 1, 1)
FP8_E4M3 = fp_levels(4, 3, 7, nan_top=True)

# ----------------------------------------------------------------------------- Tie


def tie_slices(net: Net, core: str, label: str, groups: list[list[int]], weights: torch.Tensor | None = None) -> Net:
    """Lossy (exact if the slices are equal): within each group, the slices of `core` along `label` become one shared
    slice, their weighted mean. Storage then counts one slice per group plus the untouched slices."""
    n = net.copy()
    c = n.cores[core]
    _real(c, core, "tie_slices")
    ax = c.axis(label)
    X = torch.movedim(c.data, ax, 0).clone()
    w = torch.ones(X.shape[0], dtype=X.dtype) if weights is None else weights.to(X.dtype)
    grouped = {j for g in groups for j in g}
    for g in groups:
        ww = w[g].reshape(-1, *[1] * (X.dim() - 1))
        X[g] = (X[g] * ww).sum(0, keepdim=True) / ww.sum()
    c.data = torch.movedim(X, 0, ax)
    c.tied = (len(groups) + X.shape[0] - len(grouped)) / X.shape[0]
    return n


def tie_cores(mnet: ModelNet, keys: list[tuple[int, str]], Hs: list[torch.Tensor] | None = None) -> ModelNet:
    """Lossy: several linear layers share one tensor. With input correlation matrices, the shared tensor is the one that
    minimizes the summed output error, (Σ W_i H_i)(Σ H_i)⁻¹; without them, the plain mean."""
    n = mnet.copy()
    Ws = [n.linears[k].matrix().cpu().double() for k in keys]
    if Hs is None:
        W = sum(Ws) / len(Ws)
    else:
        Hs = [H.double() for H in Hs]
        W = sum(Wi @ Hi for Wi, Hi in zip(Ws, Hs)) @ torch.linalg.inv(sum(Hs))
    from .frame import matrix_net
    n.linears[keys[0]] = matrix_net(W)
    for k in keys[1:]:
        n.ties[k] = keys[0]
    return n

# ----------------------------------------------------------------------------- Restrict onto sets and codebooks


def snap(x: torch.Tensor, levels) -> torch.Tensor:
    """The nearest value of a sorted finite set, for every entry (halfway ties go to the lower value)."""
    lv = torch.as_tensor(levels, dtype=x.dtype, device=x.device)
    return lv[torch.bucketize(x.contiguous(), (lv[1:] + lv[:-1]) / 2)]


def restrict_set(net: Net, core: str, domain: Domain) -> Net:
    """Lossy: move every entry to the nearest value of a finite set (halfway ties go to the lower value)."""
    n = net.copy()
    c = n.cores[core]
    _real(c, core, "restrict_set")
    c.data = snap(c.data, domain.levels)
    c.domain = domain
    return n


def _vectors(c: Core, label: str):
    ax = c.axis(label)
    X = torch.movedim(c.data, ax, -1)
    return X.reshape(-1, X.shape[-1]), X.shape, ax


def nearest(X: torch.Tensor, C: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    out = torch.empty(len(X), dtype=torch.long)
    for b in range(0, len(X), chunk):
        out[b : b + chunk] = torch.cdist(X[b : b + chunk].float(), C.float()).argmin(1)
    return out


def kmeans(X: torch.Tensor, K: int, iters: int = 12, seed: int = 0, sample: int = 200_000) -> torch.Tensor:
    """The codebook rule: Lloyd's algorithm on a sample of the vectors, empty clusters reseeded at random."""
    g = torch.Generator().manual_seed(seed)
    S = X[torch.randperm(len(X), generator=g)[:sample]].float()
    C = S[torch.randperm(len(S), generator=g)[:K]].clone()
    for _ in range(iters):
        a = nearest(S, C)
        sums = torch.zeros_like(C).index_add_(0, a, S)
        counts = torch.bincount(a, minlength=K).float()
        empty = counts == 0
        C = torch.where(empty[:, None], S[torch.randint(len(S), (K,), generator=g)], sums / counts.clamp(min=1)[:, None])
    return C.to(X.dtype)


def restrict_codebook(net: Net, core: str, label: str, C: torch.Tensor, search=None) -> Net:
    """Lossy: every vector along `label` moves to its nearest codeword. Storage is log2 K bits per vector.
    `search` replaces the nearest-codeword search (e.g. one that runs on the GPU)."""
    n = net.copy()
    c = n.cores[core]
    _real(c, core, "restrict_codebook")
    X, shape, ax = _vectors(c, label)
    Q = C.to(X.dtype)[(search or nearest)(X, C)]
    c.data = torch.movedim(Q.reshape(shape), -1, ax)
    c.domain = CODEBOOK(len(C), C.shape[1], label)
    return n


def split_codebook(net: Net, core: str, label: str, C: torch.Tensor, k: str = "k", search=None) -> Net:
    """Exact once every vector is a codeword: the core becomes stored indices A (one-hot over k) and the codebook C(k, label)."""
    n = net.copy()
    c = n.cores.pop(core)
    X, shape, ax = _vectors(c, label)
    idx = (search or nearest)(X, C)
    if (X - C.to(X.dtype)[idx]).abs().max() > 1e-9:
        raise ValueError(f"core {core!r} has vectors that are not codewords; Restrict onto the codebook first")
    labels_a = tuple(l for l in c.labels if l != label)
    n.cores[core + ".A"] = Core(idx.reshape(shape[:-1]), labels_a, ONEHOT(len(C), k))
    n.cores[core + ".C"] = Core(C.to(X.dtype).clone(), (k, label))
    n.terms = [(coef, sum(((core + ".A", core + ".C") if x == core else (x,) for x in names), ())) for coef, names in n.terms]
    return n

# ----------------------------------------------------------------------------- structured masks (rules)


def nm_keep(score: torch.Tensor, n: int, m: int, dim: int = -1) -> torch.Tensor:
    """Keep the n highest-scoring entries in every run of m consecutive entries along `dim`."""
    s = torch.movedim(score, dim, -1)
    g = s.reshape(*s.shape[:-1], -1, m)
    top = g.topk(n, dim=-1).indices
    keep = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, top, True)
    return torch.movedim(keep.reshape(s.shape), -1, dim)

# ----------------------------------------------------------------------------- Refit, generalized


def to_device(mnet: ModelNet, device, dtype=torch.float32) -> ModelNet:
    """Move every linear's cores to one device (indices keep their integer type)."""
    n = mnet.copy()
    for key, net in n.linears.items():
        m = net.copy()
        for c in m.cores.values():
            c.data = c.data.to(device) if c.domain.kind == "onehot" else c.data.to(device=device, dtype=dtype)
            if c.keep is not None:
                c.keep = c.keep.to(device)
        n.linears[key] = m
    return n


def refit_gradient(mnet: ModelNet, trainable: list[tuple[tuple[int, str], str]], base: Model, ids: torch.Tensor,
                   steps: int = 150, lr: float = 1e-3, bs: int = 4, seed: int = 0, log: Callable | None = None) -> ModelNet:
    """Refit with scope = the listed cores (only real-valued ones), objective = KL from the original model's next-token
    distribution on calibration text, solver = Adam. Each core's learning rate is scaled by its typical magnitude."""
    dev = base.device
    n = to_device(mnet, dev)
    params = []
    for key, name in trainable:
        c = n.linears[key].cores[name]
        if c.domain.kind != "real":
            raise ValueError(f"{key}/{name} is {c.domain.kind}-valued; gradient Refit trains real cores only")
        c.data = c.data.detach().clone().requires_grad_(True)
        params.append({"params": [c.data], "lr": lr * float(c.data.abs().mean())})
    opt = torch.optim.Adam(params)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        batch = ids[torch.randint(len(ids), (bs,), generator=g)].to(dev)
        with torch.no_grad():
            lb = F.log_softmax(forward_logits(base, batch), -1)
        lp = F.log_softmax(forward_logits(n.compile(cache=False), batch), -1)
        loss = (lb.exp() * (lb - lp)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log and (step % 25 == 0 or step == steps - 1):
            log(f"    refit step {step + 1}/{steps}: KL {loss.item():.4f}")
    for key, name in trainable:
        c = n.linears[key].cores[name]
        c.data = c.data.detach()
    for net in n.linears.values():
        net._matrix, net._conv = None, {}
    return n

# ----------------------------------------------------------------------------- Insert, Randomize, Condition, Replace


@dataclass
class Node:
    """A step placed on an activation bond. kind 'fixed' computes the same function for every input (Insert);
    kind 'data' chooses its mask from the activations themselves (Condition)."""
    name: str
    kind: str
    fn: Callable

    def __call__(self, x):
        return self.fn(x)


def act_quant(bits: int) -> Node:
    """Per-token symmetric integer quantization of an activation (simulated)."""
    qmax = 2 ** (bits - 1) - 1

    def fn(x):
        s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
        return torch.clamp(torch.round(x / s), -qmax - 1, qmax) * s
    return Node(f"int{bits} activations", "fixed", fn)


def hadamard(n: int) -> torch.Tensor:
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def block_hadamard(dim: int, block: int, seed: int = 0) -> torch.Tensor:
    """A randomized block-Hadamard rotation: random signs, then a Hadamard transform on each block."""
    signs = torch.where(torch.rand(dim, generator=torch.Generator().manual_seed(seed)) < 0.5, -1.0, 1.0)
    return torch.diag(signs) @ torch.block_diag(*[hadamard(block)] * (dim // block))


def random_orthogonal(dim: int, seed: int = 0) -> torch.Tensor:
    """The random-gauge rule: an orthogonal matrix drawn uniformly (QR of a Gaussian matrix, signs fixed)."""
    Q, R = torch.linalg.qr(torch.randn(dim, dim, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))
    return Q * torch.sign(torch.diag(R))


def insert(mnet: ModelNet, layer: int, point: str, node: Node) -> ModelNet:
    """Insert a node on an activation bond: 'attn_in', 'k', 'v', 'o_in', 'mlp_in' or 'down_in'."""
    n = mnet.copy()
    e = dict(n.extras.get(layer, {}))
    taps = {k: list(v) for k, v in (e.get("taps") or {}).items()}
    taps.setdefault(point, []).append(node)
    e["taps"] = taps
    n.extras[layer] = e
    return n


READERS = {"attn_in": ("q", "k", "v"), "o_in": ("o",), "mlp_in": ("gate", "up"), "down_in": ("down",)}


def rotate_at_runtime(mnet: ModelNet, layer: int, point: str, R: torch.Tensor) -> ModelNet:
    """Exact: a gauge on an activation bond whose producer cannot absorb it. The activation is rotated at runtime (an
    inserted node, x ↦ xR) and the cores that read it absorb the other side (W ↦ WR, since R is orthogonal)."""
    n = insert(mnet, layer, point, Node(f"rotation on {point}", "fixed", lambda x, R=R.to(mnet.embed.device, torch.float32): x @ R))
    for name in READERS[point]:
        n.linears[(layer, name)] = act_on_label(n.linears[(layer, name)], "i", R.to(n.linears[(layer, name)].cores["W"].data))
    return n


def restrict_stochastic(net: Net, core: str, domain: Domain, seed: int = 0) -> Net:
    """Randomize: round each entry up with probability equal to its fractional part, so the result is unbiased."""
    n = net.copy()
    c = n.cores[core]
    _real(c, core, "restrict_stochastic")
    g = torch.Generator().manual_seed(seed)
    lo = torch.floor(c.data)
    x = lo + (torch.rand(c.data.shape, generator=g, dtype=c.data.dtype) < (c.data - lo)).to(c.data.dtype)
    c.data = x if domain.lo is None else torch.clamp(x, domain.lo, domain.hi)
    c.domain = domain
    return n


def topk_activations(frac_zero: float) -> Node:
    """Condition: per token, zero the fraction of activations with the smallest magnitude."""
    def fn(x):
        k = int(x.shape[-1] * frac_zero)
        thresh = x.abs().kthvalue(k, dim=-1, keepdim=True).values
        return torch.where(x.abs() > thresh, x, torch.zeros_like(x))
    return Node(f"zero the smallest {int(100 * frac_zero)}% per token", "data", fn)


def condition(mnet: ModelNet, layer: int, point: str, node: Node) -> ModelNet:
    if node.kind != "data":
        raise ValueError("condition takes a data-dependent node; use insert for a fixed one")
    return insert(mnet, layer, point, node)


def kv_window(mnet: ModelNet, layer: int, sinks: int, window: int) -> ModelNet:
    """Condition on the token index: attention sees only the first `sinks` tokens and the latest `window`."""
    n = mnet.copy()
    n.extras[layer] = {**n.extras.get(layer, {}), "kv_window": (sinks, window)}
    return n


def replace_layer(mnet: ModelNet, layer: int) -> ModelNet:
    """Replace a whole layer with the identity: the residual stream passes through unchanged."""
    n = mnet.copy()
    n.extras[layer] = {**n.extras.get(layer, {}), "skip": True}
    return n


def replace_activation(mnet: ModelNet, layer: int, act: str) -> ModelNet:
    n = mnet.copy()
    n.extras[layer] = {**n.extras.get(layer, {}), "act": act}
    return n


def replace_core(mnet: ModelNet, key: tuple[int, str], net: Net) -> ModelNet:
    n = mnet.copy()
    n.linears[key] = net
    return n
