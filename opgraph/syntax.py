"""The grammar's other parts of speech: where a verb applies, how its arguments are chosen, and how verbs combine.

Prepositions (placement)   linears(...) selects weight tensors by layer and name; bonds name activation points
Adverbs (argument rules)    RULES: functions that choose a verb's numbers (a grid, a mask, a codebook, a gauge)
Conjunctions (control)      seq, for_each, sweep, alternate, choose
Programs                    Program: a named sequence of steps that can be printed, stored and replayed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .frame import Environment, ModelNet, Net, canonical_gauge
from .model import LAYERS, LINEARS

# ----------------------------------------------------------------------------- prepositions


def linears(layers=None, names=None) -> list[tuple[int, str]]:
    layers = range(LAYERS) if layers is None else layers
    names = LINEARS if names is None else names
    return [(i, n) for i in layers for n in names]


BONDS = ("attn_in", "k", "v", "o_in", "mlp_in", "down_in")  # activation points a node can be placed on

# ----------------------------------------------------------------------------- adverbs: argument rules


def rule_minmax(X: torch.Tensor, maxq: int):
    """Asymmetric grid per row, zero inside the range (GPTQ's)."""
    from .methods import gptq_grid
    return gptq_grid(X, maxq)


def rule_absmax(X: torch.Tensor, top: float):
    """Symmetric scale per row: the largest magnitude maps to the set's largest level."""
    return X.abs().amax(1).clamp(min=1e-12) / top


def rule_magnitude(W: torch.Tensor, H=None):
    return W.abs()


def rule_wanda(W: torch.Tensor, H):
    return W.abs() * torch.sqrt(torch.diag(H)).to(W.dtype)


def rule_block_influence(h_in: torch.Tensor, h_out: torch.Tensor) -> float:
    """How much a layer changes the residual stream: 1 − mean cosine similarity of its input and output (ShortGPT)."""
    return 1 - torch.nn.functional.cosine_similarity(h_in.float(), h_out.float(), dim=-1).mean().item()


RULES = {
    "minmax grid": rule_minmax,
    "absmax scale": rule_absmax,
    "magnitude score": rule_magnitude,
    "wanda score": rule_wanda,
    "canonical gauge": canonical_gauge,
    "block influence": rule_block_influence,
}

# ----------------------------------------------------------------------------- conjunctions: control


def seq(*steps: Callable) -> Callable:
    """Run steps in order: seq(f, g)(x) = g(f(x))."""
    def run(x, *args, **kw):
        for f in steps:
            x = f(x, *args, **kw)
        return x
    return run


def for_each(mnet: ModelNet, keys: list[tuple[int, str]], word: Callable[[Net, tuple], Net]) -> ModelNet:
    """Apply a per-tensor word at every listed site."""
    n = mnet.copy()
    for key in keys:
        n.linears[key] = word(n.linears[key], key)
    return n


def sweep(W: torch.Tensor, env: Environment | None, project: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor],
          block: int | None = None, on_block: Callable[[int, torch.Tensor], None] | None = None, width: int = 1,
          lazy: int = 128) -> torch.Tensor:
    """Settle the columns of W in order, `width` at a time. on_block(j, T) may choose arguments for the next block from
    the running core T; project(j, columns, T) returns the settled columns (a single column when width is 1); with an
    environment, Refit then re-solves the later columns. Returns each column as it stood just before it was settled.
    GPTQ and SparseGPT are instances with width 1; with width > 1 it is block LDLQ, which settles a vector or a trellis
    sequence jointly (QuIP#, QTIP).

    Refit is applied lazily, as GPTQ's reference code does: inside a run of `lazy` columns every settled block updates
    the rest of the run at once, and the columns after the run receive the accumulated update in one matrix product.
    The result is the same Refit, in a different order of floating-point operations. Blocks chosen by on_block must not
    straddle a run (`lazy` a multiple of `block` and of `width`)."""
    T = W.clone()
    if env is not None:
        T[:, env.dead] = 0
        if lazy % width or (block and lazy % block):
            raise ValueError("lazy must be a multiple of width and of block")
    before = torch.empty_like(T)
    n = T.shape[1]
    for b0 in range(0, n, lazy):
        b1 = min(b0 + lazy, n)
        Z = torch.empty(T.shape[0], b1 - b0, dtype=T.dtype)
        for j in range(b0, b1, width):
            if on_block is not None and block and j % block == 0:
                on_block(j, T)
            cols = slice(j, j + width)
            before[:, cols] = T[:, cols]
            q = project(j, T[:, j], T) if width == 1 else project(j, T[:, cols], T)
            if env is None:
                continue
            E = (T[:, cols] - (q[:, None] if width == 1 else q))
            U = env.U[cols, cols]
            z = E / U if width == 1 else torch.linalg.solve_triangular(U, E, upper=True, left=False)  # E · U_BB⁻¹
            T[:, j + width : b1] -= z @ env.U[cols, j + width : b1]
            Z[:, j - b0 : j - b0 + width] = z
        if env is not None and b1 < n:
            T[:, b1:] -= Z @ env.U[b0:b1, b1:]
    return before


STAGES = (("attn_in",), ("o_in",), ("mlp_in",), ("down_in",))  # the order in which a layer's bonds are fed


def _rewrite(net, keys, H, word, joint, target):
    """Apply a word to the listed matrices of one bond, one at a time or jointly."""
    if joint:
        for k, o in zip(keys, word([target(k) for k in keys], H, keys)):
            net.linears[k] = o
    else:
        for k in keys:
            net.linears[k] = word(target(k), H, k)


@torch.no_grad()
def sequential(mnet: ModelNet, calib, word: Callable, names=LINEARS, log: Callable | None = None,
               joint: bool = False, true_sequential: bool = False, env: Callable | None = None) -> ModelNet:
    """Calibrate layer by layer: each listed linear's input correlation H is measured with every earlier layer already
    rewritten, then word(W, H, key) rewrites it. Inputs are captured after any runtime nodes on the bond, so a rotated
    model is calibrated in its rotated basis. With joint=True the matrices that read the same bond (q, k, v; gate, up)
    are rewritten together: word([W, ...], H, [key, ...]) returns one Net per matrix.

    With true_sequential=True a layer is calibrated in four stages (q, k, v; o; gate, up; down), and its inputs are
    captured again before each stage, so every matrix sees the inputs its own layer's rewritten matrices produce.
    Otherwise all four bonds are captured once, before any of the layer's matrices change.

    `calib` is one tensor of token windows or a list of them (one per domain); `env(xs)` turns the captured inputs of a
    bond, one tensor per domain, into H (by default the covariance, or the trace-normalized mixture of the domains)."""
    import time
    from .model import INPUT_OF, batched, capture_layer, layer_forward, rope
    t0 = time.time()
    domains = list(calib) if isinstance(calib, (list, tuple)) else [calib]
    env = env or default_environment
    cos, sin = rope(domains[0].shape[1], mnet.embed.device)
    net = mnet.copy()
    hs = [net.compile().embed[c.to(mnet.embed.device)] for c in domains]
    stages = STAGES if true_sequential else (sum(STAGES, ()),)
    for i in range(LAYERS):
        for stage in stages:
            xs = [capture_layer(h, net.compile().layers[i], cos, sin, net.mean_dim)[1] for h in hs]
            for point in stage:
                keys = [(i, n) for n in names if INPUT_OF[n] == point]
                if keys:
                    _rewrite(net, keys, env([x[point] for x in xs]), word, joint, lambda k: net.linears[k].matrix().cpu().double())
        L = net.compile().layers[i]
        hs = [batched(lambda xb: layer_forward(xb, L, cos, sin, net.mean_dim), h) for h in hs]
        if log and (i % 8 == 7 or i == LAYERS - 1):
            log(f"    layer {i + 1}/{LAYERS} ({time.time() - t0:.0f}s)")
    return net


def default_environment(xs: list) -> torch.Tensor:
    """H from the captured inputs: the covariance for one domain; for several, each domain's covariance scaled to the
    same trace and averaged, so no domain dominates because its activations happen to be larger."""
    from .frame import covariance
    Hs = [covariance(x) for x in xs]
    if len(Hs) == 1:
        return Hs[0]
    scale = sum(torch.trace(H) for H in Hs) / len(Hs)
    return sum(H * (scale / torch.trace(H)) for H in Hs) / len(Hs)


def alternate(state, steps: list[Callable], iters: int):
    """Coordinate descent: run each step on the shared state in turn, `iters` times (CALDERA's two branches)."""
    for _ in range(iters):
        for f in steps:
            state = f(state)
    return state


def choose(candidates, score: Callable) -> tuple:
    """Return (best candidate, its score) by the lowest score (AWQ's searches, mixed-precision allocation)."""
    best, best_s = None, float("inf")
    for c in candidates:
        s = score(c)
        if s < best_s:
            best, best_s = c, s
    return best, best_s

# ----------------------------------------------------------------------------- programs


@dataclass
class Step:
    verb: str
    where: str = ""
    rule: str = ""
    exact: bool = True

    def __str__(self):
        s = self.verb
        if self.rule:
            s += f"[{self.rule}]"
        if self.where:
            s += f" @ {self.where}"
        return s


@dataclass
class Program:
    """A sequence of steps, written down so it can be compared, stored and searched over. `run` applies it."""
    name: str
    steps: list[Step]
    run: Callable[[ModelNet], ModelNet] = field(repr=False)

    def __str__(self):
        return " → ".join(str(s) for s in self.steps)


def cross_covariance(x: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    """Σ over tokens of x x̃ᵀ, in float64: how the clean inputs line up with the inputs the rewritten model sees."""
    x, xt = x.reshape(-1, x.shape[-1]), xt.reshape(-1, xt.shape[-1])
    C = torch.zeros(x.shape[1], xt.shape[1], dtype=torch.float64)
    for b in range(0, len(x), 8192):
        C += (x[b : b + 8192].float().T @ xt[b : b + 8192].float()).cpu().double()
    return C


@torch.no_grad()
def sequential_cross(mnet: ModelNet, ref, calib: torch.Tensor, word: Callable, names=LINEARS, log: Callable | None = None,
                     joint: bool = False, damp: float = 0.01, points=("attn_in", "o_in", "mlp_in", "down_in"),
                     true_sequential: bool = False, env: Callable | None = None, capacity: Callable | None = None) -> ModelNet:
    """Calibrate layer by layer against an environment with a cross term. Two streams run side by side: `ref` (the
    model being approximated, in the same basis as mnet) gives each matrix its clean inputs X, the rewritten model gives
    the inputs X̃ it will actually see. Each matrix's target is first refit to W* = W (I + (X − X̃)ᵀ X̃ (X̃ᵀ X̃ + λ)⁻¹), the
    matrix that best reproduces W X from X̃ with the ridge penalty on the correction (so W* = W when X̃ = X; penalising
    W* itself shrinks every low-variance input direction, which the first token's massive activations make most of
    them), and then word(W*, H̃, key) rewrites it under H̃ = X̃ᵀ X̃. Without the penalty, ‖WX − ŴX̃‖² = ‖WX − W*X̃‖² +
    ‖(W* − Ŵ)X̃‖², so the word's usual objective is the right one (GPTAQ's idea). Only bonds in `points` get the cross
    term. true_sequential=True recaptures X̃ before each of a layer's four stages, as in `sequential`. `env` turns X̃ into
    H̃ (as in `sequential`); `capacity(params, samples, what)` is called before each cross-term fit.

    `ref` is the anchor: the model whose outputs the rewritten model should reproduce. It is the original model, or the
    original in a rotated basis when mnet has been rotated; any model with the same layer structure can serve."""
    import time
    from .frame import covariance
    from .model import INPUT_OF, batched, capture_layer, layer_forward, rope
    t0 = time.time()
    dev = mnet.embed.device
    cos, sin = rope(calib.shape[1], dev)
    net = mnet.copy()
    ids = calib.to(dev)
    h, ht = ref.embed[ids], net.compile().embed[ids]
    stages = STAGES if true_sequential else (sum(STAGES, ()),)
    for i in range(LAYERS):
        h_next, x = capture_layer(h, ref.layers[i], cos, sin, ref.mean_dim)
        for stage in stages:
            _, xt = capture_layer(ht, net.compile().layers[i], cos, sin, net.mean_dim)
            for point in stage:
                keys = [(i, n) for n in names if INPUT_OF[n] == point]
                if not keys:
                    continue
                Ht = (env or default_environment)([xt[point]])
                M = torch.eye(len(Ht), dtype=Ht.dtype)
                if point in points and capacity is not None:
                    capacity(len(Ht) ** 2, xt[point].numel(), f"cross-term target on {point}, layer {i}")
                if point in points:
                    lam = damp * torch.diag(Ht).mean() * torch.eye(len(Ht), dtype=Ht.dtype)
                    D = cross_covariance(x[point], xt[point]) - Ht  # (X − X̃)ᵀ X̃: zero when nothing upstream changed
                    M = M + torch.linalg.solve(Ht + lam, D.T).T
                _rewrite(net, keys, Ht, word, joint, lambda k, M=M: net.linears[k].matrix().cpu().double() @ M)
        L = net.compile().layers[i]
        ht = batched(lambda xb: layer_forward(xb, L, cos, sin, net.mean_dim), ht)
        h = h_next
        if log and (i % 8 == 7 or i == LAYERS - 1):
            log(f"    layer {i + 1}/{LAYERS} ({time.time() - t0:.0f}s)")
    return net
