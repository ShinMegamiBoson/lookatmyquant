"""Nested codes: one stored index per weight, read at several rates by keeping its top bits.

A compander maps a weight, in units of its row scale, to u = Φ(√α·x) in (0, 1); its b-bit index is floor(u·2^b). The
top b' bits of a b-bit index are the b'-bit index of the same compander, so a single stored tensor holds a code at
every rate up to b: a reader that fetches only the top bit-planes gets a coarser model. Under incoherence the scaled
weights are close to a unit Gaussian, so each rate decodes an index to its cell's Gaussian centroid. The level density
is ∝ p^α; α = 1/3 is the high-rate optimum for fixed-rate MSE (Panter–Dite). Storing u's top bits is storing the top
bits of the weight's rank among Gaussian samples: the rank-plus-curve code, cut into bit-planes.

Types (adjectives)
    NESTED(bits, alpha)          the b-bit compander set; its index's top bits index the coarser sets
    lloyd_max(bits)              the best fixed-rate scalar set for a unit Gaussian (not nested across rates)

Verbs
    read_top(net, core, ...)     Restrict (lossy, no new storage): a nested core onto its own coarser set, by
                                 dropping index bits

Conjunctions
    multirate_sweep              block LDLQ with one running target per rate: every column's index is chosen to
                                 minimize a weighted sum of the rates' errors, then each rate's later columns are
                                 re-solved for that rate's own error. Weight on one rate only: that rate's LDLQ.
"""

from __future__ import annotations

import math

import torch

from .codes import incoherence, incoherence_in, rht
from .frame import Environment, Net, matrix_net, split_scale
from .syntax import sweep
from .verbs import SET, restrict_set


def _phi(x):
    return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def gaussian_centroids(bounds: torch.Tensor) -> torch.Tensor:
    """The mean of a unit Gaussian on each cell [bounds[j], bounds[j+1]] (bounds may start at −inf and end at +inf)."""
    a, b = bounds[:-1], bounds[1:]
    pa, pb = torch.where(torch.isinf(a), torch.zeros_like(a), _phi(a)), torch.where(torch.isinf(b), torch.zeros_like(b), _phi(b))
    return (pa - pb) / (_Phi(b) - _Phi(a))


def gaussian_mse(bounds: torch.Tensor, levels: torch.Tensor) -> float:
    """E[(x − q(x))²] for x ~ N(0, 1), cells `bounds`, values `levels` (exact, by the truncated-moment formulas)."""
    a, b = bounds[:-1], bounds[1:]
    P = _Phi(b) - _Phi(a)
    fa = lambda t: torch.where(torch.isinf(t), torch.zeros_like(t), t * _phi(t))
    pa = lambda t: torch.where(torch.isinf(t), torch.zeros_like(t), _phi(t))
    m1 = pa(a) - pa(b)  # ∫ x φ
    m2 = P + fa(a) - fa(b)  # ∫ x² φ
    return float((m2 - 2 * levels * m1 + levels**2 * P).sum())


def compander_bounds(bits: int, alpha: float) -> torch.Tensor:
    u = torch.arange(2**bits + 1, dtype=torch.float64) / 2**bits
    return torch.special.ndtri(u) / math.sqrt(alpha)


def compander_levels(bits: int, alpha: float) -> torch.Tensor:
    return gaussian_centroids(compander_bounds(bits, alpha))


def lloyd_max(bits: int, iters: int = 500) -> torch.Tensor:
    return lloyd_max_levels(2**bits, iters)


def lloyd_max_levels(n: int, iters: int = 500) -> torch.Tensor:
    """The best set of n levels for a unit Gaussian (n need not be a power of two)."""
    lv = gaussian_centroids(torch.special.ndtri(torch.arange(n + 1, dtype=torch.float64) / n) * math.sqrt(3))
    for _ in range(iters):
        mid = (lv[1:] + lv[:-1]) / 2
        bounds = torch.cat([torch.tensor([-math.inf], dtype=lv.dtype), mid, torch.tensor([math.inf], dtype=lv.dtype)])
        lv = gaussian_centroids(bounds)
    return lv


def NESTED(bits: int, alpha: float):
    return SET(compander_levels(bits, alpha).tolist(), name=f"nested{bits}")


def read_top(net: Net, core: str, top: int, bits: int, alpha: float) -> Net:
    """Lossy, no storage: keep the top `bits` of a `top`-bit nested core's index, and decode with the coarser set."""
    n = net.copy()
    c = n.cores[core]
    fine, coarse = compander_levels(top, alpha).to(c.data), compander_levels(bits, alpha).to(c.data)
    idx = torch.bucketize(c.data.contiguous(), (fine[1:] + fine[:-1]) / 2)
    c.data = coarse[idx >> (top - bits)]
    c.domain = NESTED(bits, alpha)
    return n


GPU_LDLQ = False  # one rate, no output shaping: round on the GPU in float64 (ldlq_gpu.py)
OUTPUT_DAMP = 0.03  # the output sensitivity is noisier than the input statistics: it comes from one sampled token per position


def multirate_sweep(X: torch.Tensor, H: torch.Tensor | None, tables: torch.Tensor, weights: torch.Tensor,
                    G: torch.Tensor | None = None, block: int = 32) -> torch.Tensor:
    """Choose one index per entry of X (already in units of the row scale), for R rates at once.

    tables[r, i] is rate r's value for stored index i (so a nested code has tables[r, i] = levels_r[i >> shift_r]);
    weights[r] prices rate r's squared error. With H, each rate keeps its own running target, and once a column's
    indices are chosen each rate's later columns are re-solved for that rate's error: the R targets are swept as one
    matrix R times as tall, whose row blocks do not interact. With G (one rate only), the output sensitivity in units of the
    row scale, the error also counts under G: tr(G E H Eᵀ) = Σ_j D_jj e_jᵀ G e_j, so within each column rows are settled
    `block` at a time and later rows re-solved, as the sweep does for columns under H. Returns the indices (rows × columns)."""
    R, m = tables.shape[0], X.shape[0]
    tables, weights = tables.to(X), weights.to(X)
    if GPU_LDLQ and R == 1 and G is None and H is not None and torch.cuda.is_available():
        from .ldlq_gpu import ldlq_factor, ldlq_round
        lv = tables[0]
        Q = ldlq_round(X.cuda(), ldlq_factor(H.cuda()), lv.cuda()).cpu()
        return torch.bucketize(Q.double().contiguous(), (lv[1:] + lv[:-1]) / 2)
    idx = torch.empty(X.shape, dtype=torch.long)
    if G is not None:
        V = Environment(G, damp=OUTPUT_DAMP).U
        rows = [(a, min(a + block, m)) for a in range(0, m, block)]
        feedback = [torch.linalg.solve_triangular(V[a:b, a:b], V[a:b, b:], upper=True) for a, b in rows]

    def project(j, col, T):
        if G is not None:
            u, i, lv = col.clone(), torch.empty(m, dtype=torch.long), tables[0]
            for (a, b), M in zip(rows, feedback):
                i[a:b] = (u[a:b, None] - lv[None, :]).abs().argmin(1)
                if b < m:
                    u[b:] -= (u[a:b] - lv[i[a:b]]) @ M
            idx[:, j] = i
            return lv[i]
        t = col.view(R, m)
        cost = (weights[:, None, None] * (t[:, :, None] - tables[:, None, :]) ** 2).sum(0)  # m × K
        i = cost.argmin(1)
        idx[:, j] = i
        return tables[:, i].reshape(-1)

    if H is None:
        for j in range(X.shape[1]):
            project(j, X[:, j].repeat(R), None)
        return idx
    sweep(X.repeat(R, 1), Environment(H), project)
    return idx


def nested_tables(top: int, rates: list[int], alpha: float) -> torch.Tensor:
    i = torch.arange(2**top)
    return torch.stack([compander_levels(b, alpha)[i >> (top - b)] for b in rates])


def nested_words(Ws: list[torch.Tensor], H: torch.Tensor | None, seeds: list[int], top: int, rates: list[int], alpha: float,
                 weights: list[float], levels: torch.Tensor | None = None, in_seed: int | None = None,
                 out_rot: bool = True, G: torch.Tensor | None = None, block: int = 32) -> dict[int, list[Net]]:
    """Incoherence (shared input transform) → per-row RMS scale → multirate LDLQ onto a `top`-bit nested code → for each
    rate in `rates`, the matrices as that rate reads them. With `levels` (one rate only), the code is that set instead
    of a nested one: a dedicated code. With G, the matrix's output sensitivity (one matrix, one rate, rows not rotated), its
    rounding error is also shaped toward outputs that matter less. Returns {rate: [Net per matrix]}."""
    Ws = [W.cpu().double() for W in Ws]
    n = Ws[0].shape[1]
    R_in = rht(n, seeds[0] + 1 if in_seed is None else in_seed)  # in_seed: one input rotation shared by several calls
    nets = [incoherence(matrix_net(W), rht(W.shape[0], s), R_in) if out_rot else incoherence_in(matrix_net(W), R_in) for W, s in zip(Ws, seeds)]
    row = "u" if out_rot else "o"
    X = torch.cat([net.cores["W"].data for net in nets])
    S = X.pow(2).mean(1).sqrt().clamp(min=1e-12)
    if levels is None:
        tables = nested_tables(top, rates, alpha)
    else:
        if len(rates) != 1:
            raise ValueError("a dedicated code has one rate")
        tables = levels.double()[None]
    Hr = None if H is None else R_in @ H.double() @ R_in.T
    if G is not None and (len(Ws) > 1 or len(rates) > 1 or out_rot):
        raise ValueError("output shaping is for one matrix at one rate, with its rows not rotated")
    Gs = None if G is None else S[:, None] * G.double() * S[None, :]
    idx = multirate_sweep(X / S[:, None], Hr, tables, torch.tensor(weights, dtype=torch.float64), Gs, block)
    out = {}
    for r, b in enumerate(rates):
        dom = SET(tables[r].tolist(), bits=math.log2(tables.shape[1])) if levels is not None else NESTED(b, alpha)
        Q = tables[r][idx]
        out[b], r0 = [], 0
        for net in nets:
            r1 = r0 + net.cores["W"].data.shape[0]
            q = net.copy()
            q.cores["W"].data = (Q[r0:r1] * S[r0:r1, None]).clone()
            q = split_scale(q, "W", S[r0:r1], (row,), "W.S")
            out[b].append(restrict_set(q, "W", dom))
            r0 = r1
    return out
