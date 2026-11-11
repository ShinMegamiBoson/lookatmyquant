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

def token_error(nets: list, Ws: list[torch.Tensor], x: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """‖(Ŵ − W) x_t‖² for every token t, summed over the matrices that read this bond."""
    x = x.reshape(-1, x.shape[-1]).float()
    e = torch.zeros(len(x), dtype=torch.float64)
    for net, W in zip(nets, Ws):
        E = (net.matrix().double().cpu() - W.cpu().double()).float().to(x.device)
        for b in range(0, len(x), chunk):
            e[b : b + chunk] += (x[b : b + chunk] @ E.T).pow(2).sum(1).cpu().double()
    return e


def token_cap(word: Callable, p: float, rounds: int = 3, floor: float = 0.1, ceiling: float = 100.0,
              record: list | None = None, errors: list | None = None) -> tuple[Callable, Callable]:
    """Pull a sweep toward the tokens it leaves worst off: reweighted least squares for the ℓp norm of the per-token
    output error. p = 2 is the ordinary sweep; a larger p moves toward capping the worst token. Each round measures every
    token's error, moves its weight toward (error ÷ mean)^(p/2 − 1) with the damped step 1/(p − 1), and sweeps again
    under H = Σ_t a_t x xᵀ. Weights stay within [floor, ceiling] times their mean, so every token keeps some pull.
    Returns (env, word) for sequential; only the stored codes change, so a capped matrix runs on the same kernels."""
    from .frame import covariance
    from .syntax import default_environment
    state = {}

    def env(xs):
        state["x"] = xs[0]
        return default_environment(xs)

    def stats(e, a, shape):
        n = max(1, len(e) // 100)
        return {"mean": float(e.mean()), "p50": float(e.quantile(0.5)), "p99": float(torch.kthvalue(e, len(e) - n).values),
                "max": float(e.max()), "top1pct_share": float(e.topk(n).values.sum() / e.sum()),
                "first_token_share": float(e.reshape(shape)[:, 0].sum() / e.sum()), "max_weight": float(a.max())}

    def capped(Ws, H, ks):
        x = state["x"]
        nets = word(Ws, H, ks)
        a = torch.ones(x.shape[0] * x.shape[1], dtype=torch.float64)
        e = token_error(nets, Ws, x)
        rec = [stats(e, a, x.shape[:2])]
        step = 1 / max(p - 1, 1)
        for _ in range(rounds if p > 2 else 0):
            goal = (e / e.mean()).clamp(min=1e-9) ** (p / 2 - 1)
            a = a ** (1 - step) * (goal / goal.mean()) ** step
            a = (a / a.mean()).clamp(floor, ceiling)
            a = a / a.mean()
            w = a.sqrt().float().to(x.device).reshape(*x.shape[:2], 1)
            nets = word(Ws, covariance(x.float() * w), ks)
            e = token_error(nets, Ws, x)
            rec.append(stats(e, a, x.shape[:2]))
        if record is not None:
            record.append({"keys": [list(k) for k in ks], "rounds": rec})
        if errors is not None:
            errors.append(e.float())
        return nets
    return env, capped

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
