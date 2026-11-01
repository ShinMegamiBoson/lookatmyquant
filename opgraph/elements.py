"""Grammar elements that, as far as we know, no published method has as such.

Gauges
    pinned_orthogonal      a rotation of the residual stream that leaves chosen channels alone (the first token's
                           massive-activation channels), so `v` can keep its sparse 'ignore the sink' weights
Rules (adverbs)
    massive_channels       the residual channels that carry the first token's massive activations
    pin_tokens             Pin(x): weight chosen token positions in the environment, so their outputs are preserved
    lowrank_scales         scales are cores too: a scale core of rank r, factored by the same split_svd as any core
Checks
    capacity_check         compare a Refit's free parameters with the calibration it is fitted on, before it runs
Environments
    (syntax.sequential)    several calibration domains, mixed with equal trace (syntax.default_environment)
Anchors
    (syntax.sequential_cross)  `ref` names the model whose outputs a rewritten model must reproduce
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from .frame import Domain, Environment, Net, matrix_net, restrict, split_index, split_scale, split_svd, truncate
from .model import LAYERS, batched, layer_forward, rope
from .syntax import sweep
from .verbs import random_orthogonal

# ----------------------------------------------------------------------------- pinned rotation


@torch.no_grad()
def massive_channels(model, calib: torch.Tensor, k: int = 4, first_layer: int = 4) -> list[int]:
    """The residual channels where the first token's activation is largest relative to the other tokens, averaged over
    layers from `first_layer` on (where massive activations appear)."""
    cos, sin = rope(calib.shape[1], model.device)
    h = model.embed[calib[:8].to(model.device)]
    score = torch.zeros(h.shape[-1], device=h.device)
    for i, L in enumerate(model.layers):
        h = batched(lambda xb: layer_forward(xb, L, cos, sin, model.mean_dim), h)
        if i + 1 >= first_layer:
            score += h[:, 0].abs().mean(0) / (h[:, 1:].abs().mean((0, 1)) + 1e-6)
    return sorted(torch.topk(score, k).indices.tolist())


def pinned_orthogonal(n: int, pins: list[int], seed: int = 0) -> torch.Tensor:
    """An orthogonal matrix that maps every pinned channel to itself and rotates the rest by a random orthogonal matrix."""
    rest = [c for c in range(n) if c not in set(pins)]
    Q = torch.zeros(n, n, dtype=torch.float64)
    Q[pins, pins] = 1.0
    R = random_orthogonal(len(rest), seed)
    Q[torch.tensor(rest)[:, None], torch.tensor(rest)[None, :]] = R
    return Q

# ----------------------------------------------------------------------------- Pin(x)


def pin_tokens(positions=(0,), weight: float = 1.0) -> Callable:
    """Pin(x) as an environment rule: H = Σ_t x xᵀ + w · Σ_{t ∈ positions} x xᵀ, with w relative to the share those
    tokens already have, so weight 1 doubles their pull on every Refit. Works per domain, then mixes as usual."""
    from .frame import covariance
    from .syntax import default_environment

    def env(xs):
        out = []
        for x in xs:
            H, Hp = covariance(x), covariance(x[:, list(positions)])
            out.append(H + weight * Hp)
        if len(out) == 1:
            return out[0]
        scale = sum(torch.trace(H) for H in out) / len(out)
        return sum(H * (scale / torch.trace(H)) for H in out) / len(out)
    return env

# ----------------------------------------------------------------------------- scales are cores too


def SINT(bits: int) -> Domain:
    """Symmetric integers, −2^(b−1) … 2^(b−1) − 1."""
    return Domain("int", float(bits), float(-(2 ** (bits - 1))), float(2 ** (bits - 1) - 1))


def lowrank_scales(W: torch.Tensor, bits: int, gs: int, rank: int | None) -> torch.Tensor:
    """The absmax scale of every group, or its best rank-r approximation (kept positive). rank=None keeps all groups."""
    o, n = W.shape
    S = W.reshape(o, n // gs, gs).abs().amax(-1).clamp(min=1e-8) / (2 ** (bits - 1) - 1)
    if rank is None:
        return S
    U, sv, Vh = torch.linalg.svd(S, full_matrices=False)
    Sr = (U[:, :rank] * sv[:rank]) @ Vh[:rank]
    return Sr.clamp(min=0.05 * S.median())


def gptq_static_word(W: torch.Tensor, H: torch.Tensor | None, bits: int, gs: int, rank: int | None = None) -> Net:
    """Symmetric GPTQ with scales fixed before the sweep ('static groups'). The scale core is itself rewritten: with
    rank r it is factored by split_svd and truncated (exact, since the rule chose a rank-r scale), so storage counts
    two 16-bit factors, (o + groups)·r values, instead of one 16-bit scale per group.
    Word: Split(index) → Split(scale) → [Split(SVD) → Truncate(r) on the scale core] → (sweep) → Restrict(grid)."""
    W = W.cpu().double()
    o, n = W.shape
    dom = SINT(bits)
    S = lowrank_scales(W, bits, gs, rank)
    if H is not None:
        def project(j, col, T):
            s = S[:, j // gs]
            return torch.clamp(torch.round(col / s), dom.lo, dom.hi) * s
        W = sweep(W, Environment(H), project)
    net = split_index(matrix_net(W), "i", (("g", n // gs), ("w", gs)))
    net = split_scale(net, "W", S, ("o", "g"), "W.S")
    if rank is not None:
        net = truncate(split_svd(net, "W.S", ("o",), ("g",), "r"), "r", rank)
    return restrict(net, "W", dom)

# ----------------------------------------------------------------------------- capacity


def capacity_check(min_ratio: float = 50.0, log: Callable | None = print, record: list | None = None) -> Callable:
    """Before a Refit is fitted, compare its free parameters with the numbers it is fitted on. Below `min_ratio`
    samples per parameter the fit will absorb noise in the calibration text; the check reports it (and records it)."""
    def check(params: int, samples: int, what: str) -> float:
        ratio = samples / params
        if record is not None:
            record.append({"what": what, "params": params, "samples": samples, "ratio": ratio})
        if ratio < min_ratio and log:
            log(f"    capacity: {what} fits {params:,} parameters from {samples:,} numbers ({ratio:.1f} per parameter, below {min_ratio:g})")
        return ratio
    return check
