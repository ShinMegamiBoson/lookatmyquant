"""Formats and codes from the current state of the art, written in the grammar.

Types (adjectives)
    E8M0                     power-of-two scales: the shared scale of the OCP microscaling (MX) formats
    FP32                     a full-precision scalar (NVFP4's per-tensor scale)
    GAUSS2                   the 4-level Lloyd–Max quantizer for a unit Gaussian
    TRELLIS(values, L, k)    sequences that are paths through a bitshift trellis with 2^L states and k bits per step
    RHT                      a randomized Hadamard transform, stored as its random signs
    ENTROPY(bits)            unbounded integers, entropy-coded: priced at the measured entropy of the stored symbols

Rules (adverbs)
    mx_scale                 OCP MX: 2^(floor(log2 amax) − 2) for each block of 32
    nv_global, nv_block_scales   NVFP4: one FP32 scale per tensor, one E4M3 scale per block of 16
    hadamard, rht            Hadamard matrices of order 2^a·(q+1) (Sylvester ⊗ Paley), with random signs
    gaussian_codebook        k-means on unit-Gaussian samples: one codebook serves every incoherent matrix
    trellis_values           a value for each trellis state: Gaussian quantiles in pseudo-random order, moved to the means of
                             the samples each attracts (Lloyd steps, or decaying means over fresh batches)

Verbs
    incoherence              Gauge (exact): both bonds of a matrix by randomized Hadamards applied at runtime
    restrict_trellis         Restrict (lossy): every sequence along a label onto its nearest trellis path (Viterbi)

Words
    fp4_word, fp4_sweep      MXFP4 and NVFP4, rounded to nearest or with GPTQ's Refit
    ldlq_word                (incoherence →) per-row scale → (block LDLQ →) a scalar set, a vector codebook or a trellis.
                             With incoherence and a vector codebook it is QuIP#-style; with a trellis, QTIP-style.
    ecsq_words               incoherence → per-row scale → LDLQ onto a uniform grid → entropy-coded integers, with the
                             step multiplier of each matrix set so its entropy hits a target rate
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from .frame import REAL, Domain, Environment, Net, _real, gauge, insert_identity, matrix_net, split_index, split_scale
from .syntax import sweep
from .verbs import FP4_E2M1, FP8_E4M3, SET, _vectors, kmeans, nearest, restrict_codebook, restrict_set, snap, split_codebook

CACHE = Path(__file__).resolve().parents[1] / "cache" / "codes"
from .model import DEVICE  # the nearest-codeword search runs where the model does

# ----------------------------------------------------------------------------- types

E8M0 = SET([2.0**e for e in range(-127, 128)])
FP32 = Domain("real", 32.0)
FP4 = SET(FP4_E2M1)
E4M3 = SET(FP8_E4M3)
GAUSS2 = SET([-1.5104, -0.4528, 0.4528, 1.5104])  # Lloyd–Max for N(0, 1) at 2 bits (Max, 1960)
RHT = Domain("rht", 0.0)


def TRELLIS(values: torch.Tensor, L: int, k: int, axis: str = "t", fixed_start: bool = False) -> Domain:
    return Domain("trellis", float(k), levels=tuple(float(v) for v in values), axis=axis, state_bits=L, fixed_start=fixed_start)

# ----------------------------------------------------------------------------- rules: microscaling scales


def mx_scale(X: torch.Tensor) -> torch.Tensor:
    """OCP MX v1.0: the shared scale of a block is 2^(floor(log2 amax) − emax), emax = 2 for E2M1. Larger elements saturate."""
    amax = X.abs().amax(-1)
    e = torch.floor(torch.log2(torch.where(amax > 0, amax, torch.ones_like(amax)))) - 2
    return torch.where(amax > 0, torch.exp2(e), torch.ones_like(amax))


def nv_global(W: torch.Tensor) -> float:
    """NVFP4's per-tensor scale, amax / (6 · 448), so that every block scale fits in E4M3; held in FP32."""
    return float(torch.tensor(float(W.abs().max()) / (6 * 448), dtype=torch.float32))


def nv_block_scales(X: torch.Tensor) -> torch.Tensor:
    """NVFP4's block scale (X already divided by the per-tensor scale): amax / 6, rounded to the nearest E4M3 value."""
    s = snap(X.abs().amax(-1) / 6, FP8_E4M3)
    return torch.where(s > 0, s, torch.full_like(s, 2.0**-9))  # the smallest positive E4M3 value, for an all-zero block

# ----------------------------------------------------------------------------- rules: Hadamard transforms and Gaussian codes


def _is_prime(q: int) -> bool:
    return q > 1 and all(q % d for d in range(2, math.isqrt(q) + 1))


def _paley(q: int) -> torch.Tensor:
    """Paley's first construction: a ±1 Hadamard matrix of order q + 1, for a prime q ≡ 3 (mod 4)."""
    squares = {(x * x) % q for x in range(1, q)}
    chi = torch.tensor([0.0] + [1.0 if a in squares else -1.0 for a in range(1, q)], dtype=torch.float64)
    Q = chi[(torch.arange(q)[None, :] - torch.arange(q)[:, None]) % q]
    S = torch.zeros(q + 1, q + 1, dtype=torch.float64)
    S[0, 1:], S[1:, 0], S[1:, 1:] = 1, -1, Q
    return S + torch.eye(q + 1, dtype=torch.float64)


def hadamard(n: int) -> torch.Tensor:
    """An orthogonal Hadamard matrix of order n = 2^a·p with p = 1, 2 or q + 1 for a prime q ≡ 3 (mod 4):
    Sylvester's construction for the power of two, Kronecker with Paley's for the rest (960 = 16·60, 2560 = 128·20)."""
    for a in range(int(math.log2(n)), -1, -1):
        if n % 2**a:
            continue
        p = n // 2**a
        if p in (1, 2) or (_is_prime(p - 1) and (p - 1) % 4 == 3):
            H = torch.ones(1, 1, dtype=torch.float64)
            while H.shape[0] < 2**a:
                H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
            P = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64) if p == 2 else torch.ones(1, 1, dtype=torch.float64) if p == 1 else _paley(p - 1)
            return torch.kron(H, P) / math.sqrt(n)
    raise ValueError(f"no Hadamard construction here for order {n}")


def rht(n: int, seed: int) -> torch.Tensor:
    """A randomized Hadamard transform H·diag(s), random signs s: orthogonal, and it spreads any weight across all n inputs."""
    s = torch.where(torch.rand(n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) < 0.5, -1.0, 1.0)
    return hadamard(n) * s[None, :]


def gaussian_codebook(K: int, d: int, seed: int = 0) -> torch.Tensor:
    """k-means on unit-Gaussian samples. After incoherence processing a weight matrix is close to i.i.d. Gaussian, so this
    one codebook serves every matrix; it is a fixed part of the format, like a lattice, and is not stored per model."""
    path = CACHE / f"gauss_K{K}_d{d}_s{seed}.pt"
    if path.exists():
        return torch.load(path)
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(max(200 * K, 400_000), d, generator=g, dtype=torch.float64)
    C = kmeans(X, K, iters=30, seed=seed, sample=len(X)).double()
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(C, path)
    return C

# ----------------------------------------------------------------------------- the trellis


def viterbi(X: torch.Tensor, values: torch.Tensor, L: int, k: int, chunk: int | None = None, start: torch.Tensor | None = None) -> torch.Tensor:
    """The path through a bitshift trellis nearest each row of X in squared error. State s_t = (s_{t−1}·2^k + b_t) mod 2^L
    emits values[s_t]; the first state is free (L bits), each later one costs k bits. With `start` (one state per row), the
    state before the first entry is given, and the first entry costs k bits too. Returns the states, (rows, T)."""
    # States s = r·2^k + b with the same r = s >> k share their predecessors r + h·2^(L−k), h < 2^k, so one minimum over
    # the cost viewed as (2^k, 2^(L−k)) serves 2^k states at once.
    nb, R = 1 << k, 1 << (L - k)
    chunk = chunk or (4096 if DEVICE == "cuda" else 512)  # rows per pass: the back-pointers are T × chunk × 2^(L−k) bytes
    C = values.to(DEVICE, torch.float32)
    out = torch.empty(X.shape, dtype=torch.long)
    for b in range(0, len(X), chunk):
        x = X[b : b + chunk].to(DEVICE, torch.float32)
        B, T = x.shape
        cost = (x[:, :1] - C[None, :]).square()
        if start is not None:
            first = ((start[b : b + chunk, None].to(DEVICE) << k) | torch.arange(nb, device=DEVICE)[None, :]) & ((1 << L) - 1)
            cost = torch.full_like(cost, torch.inf).scatter(1, first, cost.gather(1, first))
        back = torch.empty(T, B, R, dtype=torch.uint8, device=DEVICE)
        for t in range(1, T):
            best, arg = cost.view(B, nb, R).min(1)
            back[t] = arg.to(torch.uint8)
            cost = best.repeat_interleave(nb, dim=1) + (x[:, t : t + 1] - C[None, :]).square()
        s = cost.argmin(-1)
        path = torch.empty(B, T, dtype=torch.long, device=DEVICE)
        path[:, -1] = s
        for t in range(T - 1, 0, -1):
            r = s >> k
            s = r + back[t].gather(1, r[:, None]).squeeze(1).long() * R
            path[:, t - 1] = s
        out[b : b + chunk] = path.cpu()
    return out


TRELLIS_FIT = "ema"  # how trellis_values fits a table: "ema", or "lloyd" as before 2026-11-10 (--trellis-fit; a compiler plan names its own)


def trellis_values(L: int, k: int, T: int = 320, seed: int = 0, refine: int | None = None, fit: str | None = None) -> torch.Tensor:
    """A value for each of the 2^L states: unit-Gaussian quantiles in a pseudo-random order (a random code over the state
    index, in the spirit of QTIP's computed codes), then moved to the mean of the Gaussian samples Viterbi assigns to it.
    `fit` (TRELLIS_FIT unless given) says how. "lloyd": `refine` (6) Lloyd steps over one set of 4096 sequences. "ema": as
    many samples a pass, drawn new in batches of 256 sequences for `refine` (384) passes' worth; each state keeps a decaying
    sum and count, the new batch's weight falling from 0.5 to 0.01 along a cosine (e-means' update, arXiv 2611.09501). On
    held-out Gaussians under the row search "ema" leaves 4.5% less error at 2 bits, 6.2% less at 3 and 19.8% less at 6. Lloyd
    steps do not get there (48 of them gain 1.2% at 3 bits), and past 384 passes a doubling of the fit gains under a quarter
    of a point. A fit takes 3 to 5 minutes on an L40S, once per width. Like the Gaussian codebook the table is fixed by
    (L, k, seed, fit), not stored per model."""
    fit = fit or TRELLIS_FIT
    assert fit in ("lloyd", "ema"), fit
    path = CACHE / f"trellis_L{L}_k{k}_T{T}_s{seed}{'' if fit == 'lloyd' else '_' + fit}.pt"
    if path.exists():
        return torch.load(path)
    g = torch.Generator().manual_seed(seed)
    S = 1 << L
    v = torch.special.ndtri((torch.randperm(S, generator=g).double() + 0.5) / S)
    if fit == "lloyd":
        X = torch.randn(4096, T, generator=g, dtype=torch.float64)
        batches = ((X, 1.0) for _ in range(6 if refine is None else refine))
    else:
        steps = 16 * (384 if refine is None else refine)
        batches = ((torch.randn(256, T, generator=g, dtype=torch.float64), 0.01 + 0.245 * (1 + math.cos(math.pi * j / max(steps - 1, 1))))
                   for j in range(steps))
    sums, cnt = torch.zeros(S, dtype=torch.float64), torch.zeros(S, dtype=torch.float64)
    for X, a in batches:  # a: the new batch's weight in each state's sum and count (1: only this batch counts, a Lloyd step)
        st = viterbi(X, v, L, k).reshape(-1)
        sums = (1 - a) * sums + a * torch.zeros(S, dtype=torch.float64).index_add_(0, st, X.reshape(-1))
        cnt = (1 - a) * cnt + a * torch.bincount(st, minlength=S).double()
        v = torch.where(cnt > 0, sums / cnt.clamp(min=1e-300), v)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(v, path)
    return v


def restrict_trellis(net: Net, core: str, label: str, domain: Domain, given: torch.Tensor | None = None) -> Net:
    """Lossy: every sequence along `label` moves to its nearest trellis path (Viterbi). Storage is k bits per entry plus
    L − k bits per sequence for the starting state. `given` supplies the projection already computed (by a sweep, on the
    same sequences); it is checked to be a valid path through the trellis instead of being recomputed."""
    n = net.copy()
    c = n.cores[core]
    _real(c, core, "restrict_trellis")
    X, shape, ax = _vectors(c, label)
    v = torch.tensor(domain.levels, dtype=X.dtype)
    if given is None:
        Q = v[viterbi(X, v, domain.state_bits, int(domain.bits))]
    else:
        Q = given.to(X.dtype).reshape(X.shape)
        order = torch.argsort(v)
        pos = torch.searchsorted(v[order], Q.contiguous()).clamp(max=len(v) - 1)
        states = order[pos]
        if not (torch.equal(v[states], Q) and is_trellis_path(states, domain.state_bits, int(domain.bits))):
            raise ValueError("the given projection is not a path through this trellis")
    c.data = torch.movedim(Q.reshape(shape), -1, ax)
    c.domain = Domain(domain.kind, domain.bits, levels=domain.levels, axis=label, state_bits=domain.state_bits, fixed_start=domain.fixed_start)
    return n


def is_trellis_path(states: torch.Tensor, L: int, k: int) -> bool:
    """Every step keeps the previous state's low L − k bits as its high bits."""
    return bool(torch.all((states[..., 1:] >> k) == (states[..., :-1] & ((1 << (L - k)) - 1))))

# ----------------------------------------------------------------------------- the incoherence gauge


def incoherence(net: Net, R_out: torch.Tensor, R_in: torch.Tensor, core: str = "W") -> Net:
    """Exact: W = R_outᵀ (R_out W R_inᵀ) R_in. Both bonds of the matrix are gauged by orthogonal transforms that are applied
    at runtime (x ↦ R_in x before the product, y ↦ R_outᵀ y after it); the middle core is what gets quantized. The new
    bonds are u (output) and v (input); the transform cores are priced as their random signs."""
    n = insert_identity(net, "o", "u")
    n = gauge(n, "u", R_out.T, core, "I[u]", Ginv=R_out)
    n = insert_identity(n, "i", "v")
    n = gauge(n, "v", R_in.T, core, "I[v]", Ginv=R_in)
    for name in ("I[u]", "I[v]"):
        n.cores[name].domain = RHT
    return n

def incoherence_in(net: Net, R_in: torch.Tensor, core: str = "W") -> Net:
    """Exact: W = (W R_inᵀ) R_in. Only the input bond is gauged (x ↦ R_in x at runtime); with a scale per row, the rows
    need no rotation of their own. The new bond is v; the transform core is priced as its random signs."""
    n = insert_identity(net, "i", "v")
    n = gauge(n, "v", R_in.T, core, "I[v]", Ginv=R_in)
    n.cores["I[v]"].domain = RHT
    return n


# ----------------------------------------------------------------------------- words: MXFP4 and NVFP4


def fp4_word(W: torch.Tensor, fmt: str, S: torch.Tensor | None = None, g: float | None = None) -> Net:
    """MXFP4: Split(index, 32) → Split(scale) → Restrict(scale onto E8M0) → Restrict(values onto E2M1).
    NVFP4: Split(index, 16) → Split(scale, per tensor, FP32) → Split(scale) → Restrict(scale onto E4M3) → Restrict(E2M1).
    The scale rules choose representable scales, so the Restrict on a scale core is exact; it records the scale's type."""
    W = W.cpu().double()
    gs = {"mx": 32, "nv": 16}[fmt]
    net = split_index(matrix_net(W), "i", (("g", W.shape[1] // gs), ("w", gs)))
    if fmt == "nv":
        g = nv_global(W) if g is None else g
        net = split_scale(net, "W", torch.tensor(g, dtype=W.dtype), (), "W.G")
        net.cores["W.G"].domain = FP32
    X = net.cores["W"].data
    S = (mx_scale(X) if fmt == "mx" else nv_block_scales(X)) if S is None else S
    net = split_scale(net, "W", S, ("o", "g"), "W.S")
    before = net.cores["W.S"].data
    net = restrict_set(net, "W.S", E8M0 if fmt == "mx" else E4M3)
    if not torch.equal(before, net.cores["W.S"].data):
        raise ValueError("the scale rule chose scales outside the scale type")
    return restrict_set(net, "W", FP4)


def fp4_sweep(W: torch.Tensor, H: torch.Tensor, fmt: str) -> Net:
    """MXFP4 or NVFP4 with GPTQ's Refit: each block's scale is chosen from the running core, each column is rounded onto
    E2M1, and the later columns are re-solved. Then the same word as round-to-nearest, on the columns as they were fixed."""
    W = W.cpu().double()
    gs = {"mx": 32, "nv": 16}[fmt]
    g = nv_global(W) if fmt == "nv" else 1.0
    S = torch.empty(W.shape[0], W.shape[1] // gs, dtype=W.dtype)

    def on_block(j, T):
        X = T[:, j : j + gs] if fmt == "mx" else T[:, j : j + gs] / g
        S[:, j // gs] = mx_scale(X) if fmt == "mx" else nv_block_scales(X)

    def project(j, col, T):
        s = S[:, j // gs]
        if fmt == "mx":
            return snap(col / s, FP4.levels) * s
        return snap(col / g / s, FP4.levels) * s * g

    before = sweep(W, Environment(H), project, block=gs, on_block=on_block)
    return fp4_word(before, fmt, S, g if fmt == "nv" else None)

# ----------------------------------------------------------------------------- words: incoherence, block LDLQ and the three codes


class ScalarCode:
    """Each entry onto a finite set (width 1: block LDLQ is GPTQ's column sweep)."""

    def __init__(self, domain: Domain):
        self.domain, self.width = domain, 1

    def project(self, X):
        return snap(X, self.domain.levels)

    def apply(self, net, core, rows, cols, settled=None):
        return restrict_set(net, core, self.domain)


class VectorCode:
    """Each run of d consecutive inputs onto a codebook vector; split into stored indices and the shared codebook.
    The nearest-codeword search runs on the GPU when there is one, in the expanded form ‖c‖² − 2x·c."""

    def __init__(self, C: torch.Tensor):
        self.C, self.width = C.double(), C.shape[1]
        self._Cg = self.C.to(DEVICE, torch.float32)
        self._c2 = self._Cg.square().sum(1)

    def search(self, X, C=None):
        out = torch.empty(len(X), dtype=torch.long)
        for b in range(0, len(X), 1 << 18):
            x = X[b : b + (1 << 18)].to(DEVICE, torch.float32)
            out[b : b + len(x)] = (self._c2[None, :] - 2 * x @ self._Cg.T).argmin(1).cpu()
        return out

    def project(self, X):
        return self.C.to(X.dtype)[self.search(X)]

    def apply(self, net, core, rows, cols, settled=None):
        d = self.width
        net = split_index(net, cols, (("g", net.dim(cols) // d), ("w", d)))
        net = restrict_codebook(net, core, "w", self.C, search=self.search)
        net = split_codebook(net, core, "w", self.C, search=self.search)
        net.cores[core + ".C"].free = True  # the Gaussian codebook is fixed by the format, not stored per model
        return net


class TrellisCode:
    """Sequences of T consecutive outputs (one column) onto trellis paths. Block LDLQ settles `width` columns at a time."""

    def __init__(self, L: int, k: int, T: int = 320, width: int = 16, seed: int = 0):
        self.L, self.k, self.T, self.width = L, k, T, width
        self.values = trellis_values(L, k, T, seed)
        self.domain = TRELLIS(self.values, L, k)

    def _seqs(self, X):
        o, w = X.shape
        return X.reshape(o // self.T, self.T, w).permute(0, 2, 1).reshape(-1, self.T)

    def project(self, X):
        o, w = X.shape
        Q = self.values.to(X.dtype)[viterbi(self._seqs(X), self.values, self.L, self.k)]
        return Q.reshape(o // self.T, w, self.T).permute(0, 2, 1).reshape(o, w)

    def apply(self, net, core, rows, cols, settled=None):
        """With `settled` (the sweep's projection, in the scaled units of the core), the Viterbi pass is not repeated."""
        net = split_index(net, rows, (("s", net.dim(rows) // self.T), ("t", self.T)))
        return restrict_trellis(net, core, "t", self.domain, given=None if settled is None else self._seqs(settled))


class RowTrellisCode:
    """Each row as one trellis path along its columns, from state 0: k bits per entry and nothing else, and a state is 12
    consecutive bits of the row's own stream, so reading an entry touches no other row. Block LDLQ settles `width` columns
    at a time; each block's Viterbi also looks `look` columns ahead, at their targets as they stand, and keeps only the block."""

    def __init__(self, L: int, k: int, T: int = 256, width: int = 16, look: int = 8, seed: int = 0):
        self.L, self.k, self.width, self.look = L, k, width, look
        self.values = trellis_values(L, k, T, seed)
        self.domain = TRELLIS(self.values, L, k, fixed_start=True)
        self.state = None

    def step(self, j, X, ahead):
        """Column j onward, `width` columns: the block's settled values, continuing each row's path."""
        if j == 0:
            self.state = torch.zeros(len(X), dtype=torch.long)
        w = X.shape[1]
        path = viterbi(torch.cat([X, ahead], 1), self.values, self.L, self.k, start=self.state)[:, :w]
        self.state = path[:, -1]
        return self.values.to(X.dtype)[path]

    def apply(self, net, core, rows, cols, settled=None):
        if settled is None:  # no sweep: each whole row at once
            X = net.cores[core].data
            settled = self.values.to(X.dtype)[viterbi(X, self.values, self.L, self.k, start=torch.zeros(len(X), dtype=torch.long))]
        return restrict_trellis(net, core, cols, self.domain, given=settled)


def ldlq_words(Ws: list[torch.Tensor], H: torch.Tensor | None, code, seeds: list[int], incoherent: bool = True,
               in_seed: int | None = None, out_rot: bool = True) -> list[Net]:
    """(Gauge: incoherence →) Split(scale, per row, by RMS) → (block LDLQ sweep →) the code's Restrict and Split, for
    matrices that read the same input. They share one input transform (computed once at runtime), keep their own output
    transforms, and are swept together as one taller matrix: rows do not interact under Refit, so this is the same as
    sweeping each alone. With H, columns are settled `code.width` at a time and the later ones re-solved under H;
    without H, the code is applied directly (round to nearest). `in_seed` is one input rotation shared by several calls;
    without `out_rot` the rows are not rotated at all (the column-wise TrellisCode splits the rows, so it needs them rotated)."""
    Ws = [W.cpu().double() for W in Ws]
    n = Ws[0].shape[1]
    rows, cols = ("u" if out_rot else "o", "v") if incoherent else ("o", "i")
    R_in = rht(n, seeds[0] + 1 if in_seed is None else in_seed) if incoherent else None
    nets = []
    for W, s in zip(Ws, seeds):
        net = matrix_net(W)
        if incoherent:
            net = incoherence(net, rht(W.shape[0], s), R_in) if out_rot else incoherence_in(net, R_in)
        nets.append(net)
    X = torch.cat([net.cores["W"].data for net in nets])
    S = X.pow(2).mean(1).sqrt().clamp(min=1e-12)
    settled = None
    if H is not None:
        H = R_in @ H.double() @ R_in.T if incoherent else H.double()
        settled = torch.empty_like(X)  # the sweep's projection, in units of the row scale

        def project(j, x, T):
            xs = x[:, None] if code.width == 1 else x
            if isinstance(code, RowTrellisCode):
                q = code.step(j, xs / S[:, None], T[:, j + xs.shape[1] : j + xs.shape[1] + code.look] / S[:, None])
            else:
                q = code.project(xs / S[:, None])
            settled[:, j : j + code.width] = q
            q = q * S[:, None]
            return q[:, 0] if code.width == 1 else q

        X = sweep(X, Environment(H), project, width=code.width)
    out, r0 = [], 0
    for net in nets:
        r1 = r0 + net.cores["W"].data.shape[0]
        net.cores["W"].data = X[r0:r1].clone()
        net = split_scale(net, "W", S[r0:r1], (rows,), "W.S")
        out.append(code.apply(net, "W", rows, cols, None if settled is None else settled[r0:r1]))
        r0 = r1
    return out


def ldlq_word(W: torch.Tensor, H: torch.Tensor | None, code, seed: int, incoherent: bool = True) -> Net:
    """ldlq_words for a single matrix."""
    return ldlq_words([W], H, code, [seed], incoherent)[0]

# ----------------------------------------------------------------------------- words: lookup tables per row (SqueezeLLM-style) and HQQ


def lut_rows(W: torch.Tensor, weight: torch.Tensor, K: int, iters: int = 25):
    """The table rule: for every row, K values chosen by 1-D k-means weighted by each column's sensitivity (SqueezeLLM
    weights by Fisher information; here by input energy, diag H). Returns the per-row tables, sorted."""
    o, n = W.shape
    q = (torch.arange(K, dtype=W.dtype) + 0.5) / K
    C = torch.quantile(W, q, dim=1).T.contiguous()  # (o, K), start at the row's quantiles
    a = weight.to(W.dtype).expand(o, n)
    for _ in range(iters):
        A = torch.searchsorted(((C[:, 1:] + C[:, :-1]) / 2).contiguous(), W.contiguous())
        sums = torch.zeros(o, K, dtype=W.dtype).scatter_add_(1, A, W * a)
        mass = torch.zeros(o, K, dtype=W.dtype).scatter_add_(1, A, a)
        C = torch.where(mass > 0, sums / mass.clamp(min=1e-30), C).sort(1).values
    return C


def restrict_lut(net: Net, core: str, row: str, C: torch.Tensor) -> Net:
    """Lossy, then exact: every entry moves to the nearest value of its row's table; the core becomes stored indices
    (one-hot over k) and the tables C(row, k), contracted by a gather within each row."""
    from .frame import Core
    from .verbs import ONEHOT
    n = net.copy()
    c = n.cores.pop(core)
    if c.labels[0] != row or c.data.dim() != 2:
        raise ValueError("restrict_lut expects a matrix core whose first index is the row")
    A = torch.searchsorted(((C[:, 1:] + C[:, :-1]) / 2).contiguous(), c.data.contiguous())
    n.cores[core + ".A"] = Core(A, c.labels, ONEHOT(C.shape[1], "k"))
    n.cores[core + ".L"] = Core(C.to(c.data.dtype).clone(), (row, "k"))
    n.terms = [(coef, sum(((core + ".A", core + ".L") if x == core else (x,) for x in names), ())) for coef, names in n.terms]
    return n


def lut_word(W: torch.Tensor, H: torch.Tensor, bits: int) -> Net:
    """SqueezeLLM-style, dense part only: a 2^bits-entry lookup table per row, fitted by sensitivity-weighted k-means.
    Storage is `bits` per weight plus the tables (2^bits 16-bit values per row). No sparse outliers, no GPTQ."""
    W = W.cpu().double()
    return restrict_lut(matrix_net(W), "W", "o", lut_rows(W, torch.diag(H.double()), 2**bits))


def hqq_word(W: torch.Tensor, bits: int, gs: int = 64, iters: int = 20, p: float = 0.7, beta: float = 10.0, kappa: float = 1.01) -> Net:
    """HQQ: min–max scales per group, then the zero point optimized without data by half-quadratic splitting under an
    ℓp loss (p < 1, robust to outliers), stopping when the mean error stops falling. The zero point is a 16-bit real,
    as in HQQ, so storage is bits + 32/gs per weight. Word: Split(index) → Split(scale) → Split(offset, real) →
    Restrict(grid)."""
    from .frame import INT, restrict, split_offset
    W = W.cpu().double()
    o, n = W.shape
    maxq = 2**bits - 1
    X = W.reshape(o, n // gs, gs)
    lo, hi = X.amin(-1), X.amax(-1)
    s = (hi - lo).clamp(min=1e-8) / maxq
    z = -lo / s
    best, best_z = float("inf"), z.clone()
    for _ in range(iters):
        Wq = torch.clamp(torch.round(X / s[..., None] + z[..., None]), 0, maxq)
        Wr = (Wq - z[..., None]) * s[..., None]
        err = float((X - Wr).abs().mean())
        if err >= best:
            break
        best, best_z = err, z.clone()
        E = X - Wr
        We = torch.sign(E) * torch.relu(E.abs() - (1 / beta) * E.abs().clamp(min=1e-12).pow(p - 1))
        z = (Wq - (X - We) / s[..., None]).mean(-1)
        beta *= kappa
    net = split_index(matrix_net(W), "i", (("g", n // gs), ("w", gs)))
    net = split_scale(net, "W", s, ("o", "g"), "W.S")
    net = split_offset(net, "W", best_z, ("o", "g"), "W.Z")
    return restrict(net, "W", INT(bits))


# ----------------------------------------------------------------------------- words: entropy-coded uniform grids


def ENTROPY(bits: float) -> Domain:
    return Domain("int", float(bits))


def symbol_entropy(codes: torch.Tensor) -> float:
    """Zeroth-order entropy of the stored integers, in bits per symbol: what an adaptive arithmetic coder approaches."""
    _, cnt = torch.unique(codes.reshape(-1), return_counts=True)
    p = cnt.double() / cnt.sum()
    return float(-(p * torch.log2(p)).sum())


def ecsq_words(Ws: list[torch.Tensor], H: torch.Tensor, seeds: list[int], target, tol: float = 0.004, iters: int = 6,
               log=None) -> list[Net]:
    """Incoherence (shared input transform) → per-row RMS scale → LDLQ onto the integers times a step → entropy-coded.
    Each matrix has one step multiplier, adjusted until the entropy of its codes is `target` bits (entropy falls by
    about one bit per doubling of the step, so c ← c·2^(entropy − target) converges in a few sweeps). Rows do not
    interact under Refit, so the matrices that share an input are swept together with their own multipliers."""
    Ws = [W.cpu().double() for W in Ws]
    n = Ws[0].shape[1]
    R_in = rht(n, seeds[0] + 1)
    nets = [incoherence(matrix_net(W), rht(W.shape[0], s), R_in) for W, s in zip(Ws, seeds)]
    X = torch.cat([net.cores["W"].data for net in nets])
    S = X.pow(2).mean(1).sqrt().clamp(min=1e-12)
    Hr = R_in @ H.double() @ R_in.T
    bounds = [0]
    for net in nets:
        bounds.append(bounds[-1] + net.cores["W"].data.shape[0])
    target = torch.as_tensor([target] * len(nets) if isinstance(target, (int, float)) else target, dtype=torch.float64)  # one per matrix
    c = torch.pow(2.0, 0.5 * math.log2(2 * math.pi * math.e) - target)  # the step a unit Gaussian needs for that entropy
    for it in range(iters):
        step = torch.cat([c[m].expand(bounds[m + 1] - bounds[m]) for m in range(len(nets))]) * S
        before = sweep(X, Environment(Hr), lambda j, col, T: torch.round(col / step) * step)
        codes = torch.round(before / step[:, None])
        ent = torch.tensor([symbol_entropy(codes[bounds[m] : bounds[m + 1]]) for m in range(len(nets))])
        if log:
            log(f"      sweep {it}: entropy " + " ".join(f"{e:.3f}" for e in ent.tolist()))
        if (ent - target).abs().max() < tol:
            break
        c = c * torch.pow(2.0, ent - target).double()
    out = []
    for m, net in enumerate(nets):
        q = net.copy()
        q.cores["W"].data = (codes[bounds[m] : bounds[m + 1]] * step[bounds[m] : bounds[m + 1], None]).clone()
        q = split_scale(q, "W", step[bounds[m] : bounds[m + 1]].clone(), ("u",), "W.S")
        q.cores["W"].data = torch.round(q.cores["W"].data)
        q.cores["W"].domain = ENTROPY(float(ent[m]))
        out.append(q)
    return out
