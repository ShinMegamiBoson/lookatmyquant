"""Typed tensor networks, and the operators every approximation method in this project is written in.

A `Net` is a sum of tensor-network terms: value = Σ_t coef_t · contract(cores of t), indexed by `out`. A label that
several cores share is a hyperedge (contracted unless it is in `out`). Each core carries a value domain (real,
integer grid) and an optional sparsity pattern, which is what `rate_bits` prices.

Operators are either exact (the value is unchanged) or lossy:

    exact:  split_index, split_scale, split_offset, split_sum, insert_identity, gauge, split_svd, contract
            (act_on_label / scale_label apply one side of a gauge; they are exact only paired with the inverse)
    lossy:  restrict, restrict_box, truncate, mask
    refit:  Environment.refit_after re-solves the not-yet-fixed part of a core under the environment (exact
            least squares given the fixed part; it is what turns RTN into GPTQ and magnitude masks into SparseGPT)

`ModelNet` is the whole model in the same terms: one local Net per linear layer, diagonal gain cores on the norm
outputs, junction cores on the residual stream, and fixed nonlinear nodes whose symmetry group decides which gauges
a bond accepts ('O' for a residual segment whose RMSNorm gain has been folded away, 'perm' before that).
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass, field, replace

import torch

from .model import LAYERS, LINEARS, Layer, Model

# ----------------------------------------------------------------------------- domains


@dataclass(frozen=True)
class Domain:
    kind: str  # "real" | "int" | "set" (finite value set) | "codebook" (vectors from a codebook) | "onehot" (stored indices)
               # | "trellis" (sequences that are paths through a shift-register trellis) | "rht" (a randomized Hadamard transform)
    bits: float  # storage cost per stored entry
    lo: float | None = None
    hi: float | None = None
    levels: tuple | None = None  # the value set, for kind "set"; the value of each trellis state, for kind "trellis"
    axis: str | None = None  # the codebook index, for kind "onehot"; the vector index, for kind "codebook"; the sequence index, for kind "trellis"
    state_bits: int | None = None  # for kind "trellis": the trellis has 2**state_bits states


REAL = Domain("real", 16.0)
ZZ = Domain("int", 16.0)  # unbounded integers: rounding without clipping


def INT(bits: int) -> Domain:
    return Domain("int", float(bits), 0.0, float(2**bits - 1))


# ----------------------------------------------------------------------------- cores and nets


@dataclass
class Core:
    data: torch.Tensor
    labels: tuple[str, ...]
    domain: Domain = REAL
    keep: torch.Tensor | None = None  # sparsity pattern (True = stored); None = dense
    free: bool = False  # structural cores (identities) that cost nothing to store
    pattern: dict | None = None  # structure of `keep`: unstructured, n:m, channel or block
    tied: float | None = None  # after Tie: the fraction of slices actually stored

    def axis(self, label: str) -> int:
        return self.labels.index(label)


@dataclass
class Net:
    cores: dict[str, Core]
    terms: list[tuple[float, tuple[str, ...]]]
    out: tuple[str, ...]
    groups: dict[str, str] = field(default_factory=dict)  # symmetry group of each internal bond
    _matrix: torch.Tensor | None = field(default=None, repr=False, compare=False)  # nets are never mutated after use
    _conv: dict = field(default_factory=dict, repr=False, compare=False)  # the matrix on other devices, so tied layers share one tensor

    def copy(self) -> "Net":
        cores = {k: replace(c, data=c.data.clone(), keep=None if c.keep is None else c.keep.clone()) for k, c in self.cores.items()}
        return Net(cores, list(self.terms), self.out, dict(self.groups))

    def dim(self, label: str) -> int:
        for c in self.cores.values():
            if label in c.labels:
                return c.data.shape[c.axis(label)]
        raise KeyError(label)

    def value(self, device=None, dtype=None) -> torch.Tensor:
        sym: dict[str, str] = {}

        def s(label):
            return sym.setdefault(label, string.ascii_letters[len(sym)])

        def data(c):
            return c.data if device is None else c.data.to(device=device, dtype=dtype if c.domain.kind != "onehot" else c.data.dtype)

        total = None
        for coef, names in self.terms:
            cs = [replace(c, data=data(c)) for c in (self.cores[n] for n in names)]
            cs = _contract_greedy(_gather_onehot(cs), self.out)
            present = {l for c in cs for l in c.labels}
            spec = ",".join("".join(s(l) for l in c.labels) for c in cs)
            out_present = [l for l in self.out if l in present]
            t = torch.einsum(f"{spec}->{''.join(s(l) for l in out_present)}", *[c.data for c in cs])
            shape = [self.dim(l) for l in self.out]
            t = t.reshape([shape[i] if l in present else 1 for i, l in enumerate(self.out)]).expand(shape)
            total = coef * t if total is None else total + coef * t
        return total

    def matrix(self, device=None, dtype=None, cache=True) -> torch.Tensor:
        """The value as a (first out label) × (remaining out labels) matrix. cache=False keeps gradients flowing."""
        if not cache:
            v = self.value(device, dtype)
            return v.reshape(v.shape[0], -1)
        if self._matrix is None:
            c = self.cores[self.terms[0][1][0]]
            if len(self.terms) == 1 and len(self.terms[0][1]) == 1 and c.labels == self.out:
                m = c.data.reshape(c.data.shape[0], -1)
            else:
                v = self.value()
                m = v.reshape(v.shape[0], -1)
            self._matrix = m
        m = self._matrix
        if device is None:
            return m
        key = (str(device), dtype)
        if key not in self._conv:
            self._conv[key] = m.to(device=device, dtype=dtype)
        return self._conv[key]

    def rate_bits(self) -> float:
        return sum(core_bits(c) for c in self.cores.values())


def core_bits(c: Core) -> float:
    """Storage of one core: its stored values at the domain's cost, plus whatever its sparsity pattern costs to record."""
    if c.free:
        return 0.0
    if c.domain.kind == "rht":  # an n×n randomized Hadamard transform is stored as its n random signs
        return float(math.isqrt(c.data.numel()))
    share = 1.0 if c.tied is None else c.tied
    if c.domain.kind == "trellis":  # k bits per entry, plus the extra bits that pick each sequence's starting state
        seqs = c.data.numel() / c.data.shape[c.axis(c.domain.axis)]
        return share * (c.data.numel() * c.domain.bits + seqs * (c.domain.state_bits - c.domain.bits))
    if c.keep is None:
        return c.data.numel() * share * c.domain.bits
    kept = int(c.keep.sum()) * share
    pat = c.pattern or {"kind": "unstructured"}
    if pat["kind"] == "unstructured":
        return kept * c.domain.bits + c.keep.numel()  # values + a 1-bit map
    if pat["kind"] == "n:m":
        return kept * c.domain.bits + c.keep.numel() / pat["m"] * math.log2(math.comb(pat["m"], pat["n"]))
    if pat["kind"] == "channel":
        return kept * c.domain.bits + c.keep.shape[c.axis(pat["label"])]
    if pat["kind"] == "block":
        return kept * c.domain.bits + c.keep.numel() / (pat["shape"][0] * pat["shape"][1])
    raise ValueError(f"unknown pattern {pat['kind']!r}")


def _gather_onehot(cs: list) -> list:
    """A one-hot core is stored as indices; contract it with its codebook by gathering rather than by a dense product."""
    cs = list(cs)
    while True:
        a = next((c for c in cs if c.domain.kind == "onehot"), None)
        if a is None:
            return cs
        k = a.domain.axis
        cb = next(c for c in cs if c is not a and k in c.labels)
        shared = [l for l in a.labels if l in cb.labels]
        if shared:  # one codebook per slice of a shared index (a lookup table per row): gather within each slice
            if set(cb.labels) - set(shared) - {k}:
                raise NotImplementedError("a per-slice codebook may only carry the shared indices and its own index")
            others = [l for l in a.labels if l not in shared]
            book = cb.data.permute([cb.axis(l) for l in shared] + [cb.axis(k)])
            idx = a.data.permute([a.axis(l) for l in shared + others])
            S, K = math.prod(book.shape[:-1]), book.shape[-1]
            vals = torch.gather(book.reshape(S, K), 1, idx.reshape(S, -1)).reshape(idx.shape)
            cs = [c for c in cs if c is not a and c is not cb] + [Core(vals, tuple(shared + others))]
            continue
        book = torch.movedim(cb.data, cb.axis(k), 0)
        rest = tuple(l for l in cb.labels if l != k)
        cs = [c for c in cs if c is not a and c is not cb] + [Core(book[a.data], a.labels + rest)]


def _contract_greedy(cs: list, out: tuple) -> list:
    """Contract a term's cores pairwise, always taking the connected pair with the smallest result, so a chain such as
    rotation · scale · codes · rotation never forms a large outer product. Disconnected cores are left for the caller."""
    cs = list(cs)
    while len(cs) > 2:
        best = None
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                A, B = cs[a], cs[b]
                if not set(A.labels) & set(B.labels):
                    continue
                needed = set(out) | {l for k, c in enumerate(cs) if k not in (a, b) for l in c.labels}
                keep = tuple(dict.fromkeys(l for l in A.labels + B.labels if l in needed))
                size = math.prod(A.data.shape[A.axis(l)] if l in A.labels else B.data.shape[B.axis(l)] for l in keep)
                if best is None or size < best[0]:
                    best = (size, a, b, keep)
        if best is None:
            return cs
        _, a, b, keep = best
        A, B = cs[a], cs[b]
        sym = {l: string.ascii_letters[i] for i, l in enumerate(dict.fromkeys(A.labels + B.labels))}
        spec = f"{''.join(sym[l] for l in A.labels)},{''.join(sym[l] for l in B.labels)}->{''.join(sym[l] for l in keep)}"
        cs = [c for k, c in enumerate(cs) if k not in (a, b)] + [Core(torch.einsum(spec, A.data, B.data), keep)]
    return cs


def matrix_net(W: torch.Tensor, out: str = "o", inp: str = "i") -> Net:
    return Net({"W": Core(W, (out, inp))}, [(1.0, ("W",))], (out, inp))


def _bcast(t: torch.Tensor, labels: tuple[str, ...], target: tuple[str, ...]) -> torch.Tensor:
    perm = [labels.index(l) for l in target if l in labels]
    shape = [t.shape[labels.index(l)] if l in labels else 1 for l in target]
    return t.permute(perm).reshape(shape)


def _real(c: Core, name: str, op: str):
    if c.domain.kind != "real":
        raise ValueError(f"{op} would move core {name!r} out of its {c.domain.kind} domain")


# ----------------------------------------------------------------------------- exact operators


def split_index(net: Net, label: str, parts: tuple[tuple[str, int], ...]) -> Net:
    """Exact: reshape one index into several, e.g. an input index into (group, within-group)."""
    names, sizes = zip(*parts)
    n = net.copy()
    for c in n.cores.values():
        if label in c.labels:
            ax = c.axis(label)
            shape = c.data.shape[:ax] + tuple(sizes) + c.data.shape[ax + 1 :]
            c.data = c.data.reshape(shape)
            if c.keep is not None:
                c.keep = c.keep.reshape(shape)
            c.labels = c.labels[:ax] + names + c.labels[ax + 1 :]
    if label in n.out:
        ax = n.out.index(label)
        n.out = n.out[:ax] + names + n.out[ax + 1 :]
    return n


def merge_index(net: Net, labels: tuple[str, ...], new: str) -> Net:
    """Exact: the inverse of split_index. Adjacent indices, in this order, become one."""
    n = net.copy()

    def merged(lbls):
        ax = lbls.index(labels[0])
        if tuple(lbls[ax : ax + len(labels)]) != tuple(labels):
            raise ValueError(f"{labels} are not adjacent in {lbls}")
        return ax, lbls[:ax] + (new,) + lbls[ax + len(labels) :]

    for c in n.cores.values():
        if labels[0] in c.labels:
            ax, c.labels = merged(c.labels)
            shape = c.data.shape[:ax] + (-1,) + c.data.shape[ax + len(labels) :]
            c.data = c.data.reshape(shape)
            if c.keep is not None:
                c.keep = c.keep.reshape(shape)
    if labels[0] in n.out:
        _, n.out = merged(n.out)
    return n


def split_scale(net: Net, core: str, scale: torch.Tensor, labels: tuple[str, ...], name: str) -> Net:
    """Exact: X = S ⊙ (X / S), with S indexed by a subset of X's labels."""
    n = net.copy()
    c = n.cores[core]
    _real(c, core, "split_scale")
    c.data = c.data / _bcast(scale, labels, c.labels)
    n.cores[name] = Core(scale.clone(), tuple(labels))
    n.terms = [(coef, names + (name,) if core in names else names) for coef, names in n.terms]
    return n


def split_offset(net: Net, core: str, offset: torch.Tensor, labels: tuple[str, ...], name: str, domain: Domain = REAL) -> Net:
    """Exact: X = (X + Z) − Z, so every term containing X becomes two terms (the second is a low-rank branch).
    On an integer core the offset must be integral, so the core stays in its domain."""
    n = net.copy()
    c = n.cores[core]
    if c.domain.kind == "int" and not torch.equal(offset, torch.round(offset)):
        raise ValueError(f"a non-integral offset would move core {core!r} out of its int domain")
    c.data = c.data + _bcast(offset, labels, c.labels)
    n.cores[name] = Core(offset.clone(), tuple(labels), domain)
    terms = []
    for coef, names in n.terms:
        terms.append((coef, names))
        if core in names:
            terms.append((-coef, tuple(name if x == core else x for x in names)))
    n.terms = terms
    return n


def split_sum(net: Net, core: str, part: torch.Tensor, name: str, domain: Domain = REAL) -> Net:
    """Exact: X = (X − P) + P. Every term containing X becomes two terms, and the new core P carries X's labels.
    Each branch can then be rewritten on its own: one truncated to low rank, the other quantized (CALDERA, LQ-LoRA),
    or one masked to a few outliers kept at full precision (SpQR, SqueezeLLM)."""
    n = net.copy()
    c = n.cores[core]
    _real(c, core, "split_sum")
    if tuple(part.shape) != tuple(c.data.shape):
        raise ValueError(f"part has shape {tuple(part.shape)}, core {core!r} has {tuple(c.data.shape)}")
    c.data = c.data - part.to(c.data.dtype)
    n.cores[name] = Core(part.to(c.data.dtype).clone(), c.labels, domain)
    terms = []
    for coef, names in n.terms:
        terms.append((coef, names))
        if core in names:
            terms.append((coef, tuple(name if x == core else x for x in names)))
    n.terms = terms
    return n


def insert_identity(net: Net, label: str, new: str) -> Net:
    """Exact: the cores holding `label` now hold `new`, joined to `label` by an identity core. This turns a
    boundary index into an internal bond, which can then be gauged."""
    n = net.copy()
    holders = [k for k, c in n.cores.items() if label in c.labels]
    for k in holders:
        c = n.cores[k]
        c.labels = tuple(new if l == label else l for l in c.labels)
    first = n.cores[holders[0]]
    d = first.data.shape[first.axis(new)]
    name = f"I[{new}]"
    n.cores[name] = Core(torch.eye(d, dtype=first.data.dtype, device=first.data.device), (new, label), free=True)
    n.terms = [(coef, names + (name,)) if any(h in names for h in holders) else (coef, names) for coef, names in n.terms]
    n.groups[new] = "GL"
    return n


def check_group(G: torch.Tensor, group: str, what: str):
    eye = torch.eye(len(G), dtype=G.dtype, device=G.device)
    tol = 1e-6 if G.dtype == torch.float64 else 1e-4
    if group == "GL":
        return
    if group == "O" and torch.allclose(G @ G.T, eye, atol=tol):
        return
    if group == "diag" and torch.count_nonzero(G - torch.diag(torch.diag(G))) == 0:
        return
    if group == "perm" and torch.all((G == 0) | (G == 1)) and torch.all(G.sum(0) == 1) and torch.all(G.sum(1) == 1):
        return
    raise ValueError(f"{what}: this gauge is not in the bond's symmetry group {group!r}")


def act_on_label(net: Net, label: str, M: torch.Tensor) -> Net:
    """Exact only when paired with the inverse elsewhere: every core holding `label` maps x[..c..] -> Σ_c x[..c..] M[c, b]."""
    n = net.copy()
    for name, c in n.cores.items():
        if label in c.labels:
            _real(c, name, "act_on_label")
            ax = c.axis(label)
            c.data = torch.movedim(torch.tensordot(c.data, M.to(c.data.dtype), dims=([ax], [0])), -1, ax)
    return n


def scale_label(net: Net, label: str, s: torch.Tensor, divide: bool = False) -> Net:
    """The diagonal case of act_on_label, elementwise: x[..b..] -> x[..b..] · s[b] (or / s[b])."""
    n = net.copy()
    for name, c in n.cores.items():
        if label in c.labels:
            _real(c, name, "scale_label")
            d = _bcast(s.to(c.data.dtype), (label,), c.labels)
            c.data = c.data / d if divide else c.data * d
    return n


def gauge(net: Net, label: str, G: torch.Tensor, left: str, right: str, Ginv: torch.Tensor | None = None) -> Net:
    """Exact: insert G·G⁻¹ on the internal bond `label`; core `left` absorbs G, core `right` absorbs G⁻¹."""
    check_group(G, net.groups.get(label, "none"), f"bond {label!r}")
    Ginv = torch.linalg.inv(G) if Ginv is None else Ginv
    n = net.copy()
    for name, M in ((left, G), (right, Ginv.T)):
        c = n.cores[name]
        _real(c, name, "gauge")
        ax = c.axis(label)
        c.data = torch.movedim(torch.tensordot(c.data, M.to(c.data.dtype), dims=([ax], [0])), -1, ax)
        c.free = False
    return n


def split_svd(net: Net, core: str, left: tuple[str, ...], right: tuple[str, ...], new: str) -> Net:
    """Exact at full rank: X = U·V through a new bond, singular values split evenly between the two factors."""
    n = net.copy()
    c = n.cores.pop(core)
    _real(c, core, "split_svd")
    X = c.data.permute([c.axis(l) for l in left + right])
    ls, rs = X.shape[: len(left)], X.shape[len(left) :]
    U, S, Vh = torch.linalg.svd(X.reshape(math.prod(ls), math.prod(rs)), full_matrices=False)
    r = S.sqrt()
    n.cores[core + ".U"] = Core((U * r).reshape(*ls, -1), left + (new,))
    n.cores[core + ".V"] = Core((r[:, None] * Vh).reshape(-1, *rs), (new,) + right)
    n.terms = [(coef, sum(((core + ".U", core + ".V") if x == core else (x,) for x in names), ())) for coef, names in n.terms]
    n.groups[new] = "GL"
    return n


def contract(net: Net, a: str, b: str, name: str) -> Net:
    """Exact: merge two cores that always appear together, summing over the labels only they hold."""
    n = net.copy()
    A, B = n.cores.pop(a), n.cores.pop(b)
    elsewhere = set(n.out) | {l for c in n.cores.values() for l in c.labels}
    shared = set(A.labels) & set(B.labels)
    keep = tuple(dict.fromkeys(l for l in A.labels + B.labels if l not in shared or l in elsewhere))
    sym = {l: string.ascii_letters[i] for i, l in enumerate(dict.fromkeys(A.labels + B.labels))}
    spec = f"{''.join(sym[l] for l in A.labels)},{''.join(sym[l] for l in B.labels)}->{''.join(sym[l] for l in keep)}"
    n.cores[name] = Core(torch.einsum(spec, A.data, B.data), keep)
    terms = []
    for coef, names in n.terms:
        if (a in names) != (b in names):
            raise ValueError(f"{a} and {b} do not always appear together")
        terms.append((coef, tuple(name if x == a else x for x in names if x != b)))
    n.terms = terms
    return n


# ----------------------------------------------------------------------------- lossy operators


def project_int(x: torch.Tensor, domain: Domain) -> torch.Tensor:
    x = torch.round(x)  # half to even, as torch.round
    return x if domain.lo is None else torch.clamp(x, domain.lo, domain.hi)


def restrict(net: Net, core: str, domain: Domain) -> Net:
    """Lossy: project every entry of a core onto a smaller domain (rounding onto an integer grid)."""
    n = net.copy()
    c = n.cores[core]
    c.data = project_int(c.data, domain)
    c.domain = domain
    return n


def restrict_box(net: Net, core: str, lo: torch.Tensor, hi: torch.Tensor, labels: tuple[str, ...]) -> Net:
    """Lossy: project a real core onto a box (clipping); the bounds are indexed by `labels`."""
    n = net.copy()
    c = n.cores[core]
    _real(c, core, "restrict_box")
    c.data = torch.maximum(torch.minimum(c.data, _bcast(hi, labels, c.labels)), _bcast(lo, labels, c.labels))
    return n


def truncate(net: Net, label: str, keep: int) -> Net:
    """Lossy: keep the first `keep` coordinates of a bond, in its current gauge."""
    n = net.copy()
    for c in n.cores.values():
        if label in c.labels:
            ax = c.axis(label)
            c.data = c.data.narrow(ax, 0, keep).clone()
            if c.keep is not None:
                c.keep = c.keep.narrow(ax, 0, keep).clone()
    return n


def mask(net: Net, core: str, keep: torch.Tensor, pattern: dict | None = None) -> Net:
    """Lossy: zero the entries outside `keep` and store the pattern. A structured pattern (n:m along a label, whole
    channels, blocks) is checked and priced: it costs fewer bits to record than an arbitrary map."""
    n = net.copy()
    c = n.cores[core]
    if pattern is not None:
        check_pattern(keep, c.labels, pattern)
    c.data = c.data * keep.to(c.data.dtype)
    c.keep = keep.clone()
    c.pattern = pattern
    return n


def check_pattern(keep: torch.Tensor, labels: tuple, pattern: dict):
    kind = pattern["kind"]
    if kind == "unstructured":
        return
    if kind == "n:m":
        k = torch.movedim(keep, labels.index(pattern["label"]), -1)
        counts = k.reshape(*k.shape[:-1], -1, pattern["m"]).sum(-1)
        if not torch.all(counts == pattern["n"]):
            raise ValueError(f"mask is not {pattern['n']}:{pattern['m']} along {pattern['label']!r}")
    elif kind == "channel":
        k = torch.movedim(keep, labels.index(pattern["label"]), 0).reshape(keep.shape[labels.index(pattern["label"])], -1)
        if not torch.all(k.all(1) | (~k).all(1)):
            raise ValueError(f"mask does not keep or drop whole channels along {pattern['label']!r}")
    elif kind == "block":
        bo, bi = pattern["shape"]
        b = keep.reshape(keep.shape[0] // bo, bo, keep.shape[1] // bi, bi)
        if not torch.all(b.all((1, 3)) | (~b).all((1, 3))):
            raise ValueError(f"mask does not keep or drop whole {bo}×{bi} blocks")
    else:
        raise ValueError(f"unknown pattern {kind!r}")


# ----------------------------------------------------------------------------- environments


def covariance(x: torch.Tensor) -> torch.Tensor:
    """E-weighted second moment of the signal on a bond: Σ_tokens x xᵀ, in float64."""
    x = x.reshape(-1, x.shape[-1])
    H = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64)
    for b in range(0, len(x), 8192):
        chunk = x[b : b + 8192].float()
        H += (chunk.T @ chunk).cpu().double()
    return H


class Environment:
    """What the rest of the network sees through a linear core's input bond: H = Σ x xᵀ, damped.

    `U` is the upper Cholesky factor of H⁻¹. Its rows are the Schur-complement updates that re-solve the
    not-yet-fixed columns after one column is fixed, which is the Refit operator in sequential form.
    """

    def __init__(self, H: torch.Tensor, damp: float = 0.01, G: torch.Tensor | None = None):
        self.G = None if G is None else G.double()  # output-side sensitivity, for objectives that weight output rows
        H = H.clone().double()
        self.dead = torch.diag(H) == 0
        H[self.dead, self.dead] = 1
        H += damp * torch.mean(torch.diag(H)) * torch.eye(len(H), dtype=H.dtype)
        self.H = H
        self.U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    def refit_after(self, T: torch.Tensor, j: int, err: torch.Tensor):
        """Exact: column j of running core T was fixed with error `err`; re-solve columns > j in place."""
        T[:, j + 1 :].addr_(err / self.U[j, j], self.U[j, j + 1 :], alpha=-1)

    def refit_block_after(self, T: torch.Tensor, j0: int, j1: int, E: torch.Tensor):
        """Exact: columns j0..j1-1 were fixed together with errors E; re-solve the columns after them in place.
        The update is −E · U_BB⁻¹ · U_B,rest (block LDLQ); with one column it is refit_after."""
        M = torch.linalg.solve_triangular(self.U[j0:j1, j0:j1], self.U[j0:j1, j1:], upper=True)
        T[:, j1:] -= E @ M

    def error(self, D: torch.Tensor) -> float:
        """The local objective tr(G D H Dᵀ): the change D measured on real inputs (H) and, if given, by output sensitivity (G)."""
        D = D.double()
        return float(torch.trace(D @ self.H @ D.T) if self.G is None else torch.trace(self.G @ D @ self.H @ D.T))

    def refit_exact(self, W: torch.Tensor, fixed: list[int], values: torch.Tensor) -> torch.Tensor:
        """The general least-squares form of Refit: given fixed columns, the free columns minimising ‖(W−Ŵ)X‖."""
        free = [j for j in range(W.shape[1]) if j not in set(fixed)]
        H = self.H
        out = W.clone()
        out[:, fixed] = values
        out[:, free] = W[:, free] + (W[:, fixed] - values) @ H[fixed][:, free] @ torch.linalg.inv(H[free][:, free])
        return out


def canonical_gauge(H: torch.Tensor) -> torch.Tensor:
    """The gauge that diagonalises a bond's environment, columns by decreasing eigenvalue, signs fixed so the
    largest-magnitude entry of each column is positive (the only freedom eigh leaves)."""
    evals, evecs = torch.linalg.eigh(H)
    Q = evecs[:, torch.argsort(evals, descending=True)]
    idx = Q.abs().argmax(0)
    return Q * torch.sign(Q[idx, torch.arange(Q.shape[1])])


# ----------------------------------------------------------------------------- the whole model


def segments() -> list[str]:
    out = []
    for i in range(LAYERS):
        out += [f"A{i}", f"M{i}"]
    return out + [f"A{LAYERS}"]


@dataclass
class ModelNet:
    """The model as one typed network over the residual stream.

    Segment A_i enters layer i's attention, M_i enters its MLP, A_32 enters the head. Each segment is read by one
    RMSNorm (fixed node) and by the next junction; it is written by the previous junction and one projection.
    """

    embed: torch.Tensor
    head: torch.Tensor | None  # None while tied to the embedding
    linears: dict[tuple[int, str], Net]
    gains: dict  # (i, "attn") | (i, "mlp") | "final" -> diagonal gain core, or None once folded
    junctions: dict[tuple[int, str], torch.Tensor | None]  # None = identity; h_out = h_in @ J
    groups: dict[str, str]
    mean_dim: int
    ties: dict = field(default_factory=dict)  # linear key -> the key whose tensor it shares (Tie)
    extras: dict = field(default_factory=dict)  # layer -> {"taps": ..., "skip": ..., "act": ..., "kv_window": ...}

    @classmethod
    def from_model(cls, m: Model) -> "ModelNet":
        linears = {(i, n): matrix_net(getattr(L, n)) for i, L in enumerate(m.layers) for n in LINEARS}
        gains = {}
        for i, L in enumerate(m.layers):
            gains[(i, "attn")], gains[(i, "mlp")] = L.attn_norm, L.mlp_norm
        gains["final"] = m.final_norm
        junctions = {(i, s): None for i in range(LAYERS) for s in ("attn", "mlp")}
        head = None if m.head is m.embed else m.head
        extras = {}
        for i, L in enumerate(m.layers):
            e = {k: getattr(L, k) for k in ("taps", "skip", "act", "kv_window") if getattr(L, k) not in (None, False, "silu")}
            e.update({k: getattr(L, k) for k in ("q_norm", "k_norm") if getattr(L, k) is not None})  # Qwen3's per-head norms
            if e:
                extras[i] = e
        return cls(m.embed, head, linears, gains, junctions, {s: "perm" for s in segments()}, m.mean_dim, {}, extras)

    def copy(self) -> "ModelNet":
        # Shallow: operators never mutate tensors in place, they replace the entries they change.
        return ModelNet(self.embed, self.head, dict(self.linears), dict(self.gains), dict(self.junctions), dict(self.groups), self.mean_dim,
                        dict(self.ties), {i: dict(e) for i, e in self.extras.items()})

    def compile(self, cache: bool = True) -> Model:
        layers = []
        for i in range(LAYERS):
            W = {n: self.linears[self.ties.get((i, n), (i, n))].matrix(self.embed.device, self.embed.dtype, cache) for n in LINEARS}
            layers.append(Layer(**W, attn_norm=self.gains[(i, "attn")], mlp_norm=self.gains[(i, "mlp")],
                                attn_shortcut=self.junctions[(i, "attn")], mlp_shortcut=self.junctions[(i, "mlp")], **self.extras.get(i, {})))
        return Model(self.embed, layers, self.gains["final"], self.embed if self.head is None else self.head, self.mean_dim)

    def rate_bits(self) -> float:
        dropped = {i for i, e in self.extras.items() if e.get("skip")}  # a layer replaced by the identity stores nothing
        linear = sum(net.rate_bits() for key, net in self.linears.items() if key not in self.ties and key[0] not in dropped)
        other = sum(J.numel() for J in self.junctions.values() if J is not None) + sum(g.numel() for g in self.gains.values() if g is not None)
        return linear + 16.0 * other

    # The four ways a tensor touches a residual segment.
    def touching(self, seg: str):
        kind, i = seg[0], int(seg[1:])
        if kind == "A":
            producers = [("embed", None)] if i == 0 else [("junction_out", (i - 1, "mlp")), ("linear_out", (i - 1, "down"))]
            if i == LAYERS:
                consumers = [("head", None)]
            else:
                consumers = [("linear_in", (i, n)) for n in ("q", "k", "v")] + [("junction_in", (i, "attn"))]
            norm = "final" if i == LAYERS else (i, "attn")
        else:
            producers = [("junction_out", (i, "attn")), ("linear_out", (i, "o"))]
            consumers = [("linear_in", (i, n)) for n in ("gate", "up")] + [("junction_in", (i, "mlp"))]
            norm = (i, "mlp")
        return producers, consumers, norm

    def segment_dim(self, seg: str) -> int:
        producers, _, _ = self.touching(seg)
        kind, key = producers[-1]
        return self.embed.shape[1] if kind == "embed" else self.linears[key].dim("o")


def untie(net: ModelNet) -> ModelNet:
    """Exact: give the head its own copy of the embedding (the inverse of Tie)."""
    n = net.copy()
    if n.head is None:
        n.head = n.embed.clone()
    return n


def fold_gain(net: ModelNet, norm) -> ModelNet:
    """Exact: contract a norm's diagonal gain into the cores that read it. The norm then commutes with every
    rotation of its segment, so the segment's symmetry group grows from 'perm' to 'O'."""
    n = net.copy()
    g = n.gains[norm]
    if g is None:
        return n
    if norm == "final":
        if n.head is None:
            raise ValueError("the head is tied to the embedding; untie it before folding the final gain")
        n.head = n.head * g
        seg = f"A{LAYERS}"
    else:
        i, part = norm
        for name in ("q", "k", "v") if part == "attn" else ("gate", "up"):
            n.linears[(i, name)] = scale_label(n.linears[(i, name)], "i", g)
        seg = f"A{i}" if part == "attn" else f"M{i}"
    n.gains[norm] = None
    n.groups[seg] = "O"
    return n


def _identity_junction(n: ModelNet, key):
    if n.junctions[key] is None:
        i, part = key
        d = n.segment_dim(f"A{i}" if part == "attn" else f"M{i}")
        n.junctions[key] = torch.eye(d, dtype=n.embed.dtype, device=n.embed.device)


def gauge_segment(net: ModelNet, seg: str, Q: torch.Tensor) -> ModelNet:
    """Exact: change the basis of a residual segment, x = x̃ Qᵀ. Allowed only for gauges in the segment's group."""
    check_group(Q.cpu(), net.groups[seg], f"segment {seg}")
    n = net.copy()
    Q = Q.to(n.embed.dtype).to(n.embed.device)
    producers, consumers, _ = n.touching(seg)
    for kind, key in producers + consumers:
        if kind == "embed":
            n.embed = n.embed @ Q
        elif kind == "head":
            n.head = n.head @ Q
        elif kind == "linear_in":
            n.linears[key] = act_on_label(n.linears[key], "i", Q)
        elif kind == "linear_out":
            n.linears[key] = act_on_label(n.linears[key], "o", Q)
        elif kind == "junction_in":
            _identity_junction(n, key)
            n.junctions[key] = Q.T @ n.junctions[key]
        elif kind == "junction_out":
            _identity_junction(n, key)
            n.junctions[key] = n.junctions[key] @ Q
    return n


def truncate_segment(net: ModelNet, seg: str, k: int) -> ModelNet:
    """Lossy: keep the first k coordinates of a residual segment in its current gauge."""
    n = net.copy()
    producers, consumers, _ = n.touching(seg)
    for kind, key in producers + consumers:
        if kind == "embed":
            n.embed = n.embed[:, :k].clone()
        elif kind == "head":
            n.head = n.head[:, :k].clone()
        elif kind == "linear_in":
            n.linears[key] = truncate(n.linears[key], "i", k)
        elif kind == "linear_out":
            n.linears[key] = truncate(n.linears[key], "o", k)
        elif kind == "junction_in":
            _identity_junction(n, key)
            n.junctions[key] = n.junctions[key][:k].clone()
        elif kind == "junction_out":
            _identity_junction(n, key)
            n.junctions[key] = n.junctions[key][:, :k].clone()
    return n


def gauge_diag(net: ModelNet, bond: tuple[int, str], s: torch.Tensor) -> ModelNet:
    """Exact: a diagonal gauge on the bond between a producer and the linear layers that read it.
    `bond` is (i, 'attn_in' | 'mlp_in' | 'down_in'). Diagonal gauges fold into the producer, so they cost no rate."""
    i, which = bond
    n = net.copy()
    s = s.to(n.embed.dtype).to(n.embed.device)
    readers = {"attn_in": ("q", "k", "v"), "mlp_in": ("gate", "up"), "down_in": ("down",)}[which]
    if which == "down_in":  # producer: up's rows, through the SwiGLU product, which commutes with diagonal scaling
        n.linears[(i, "up")] = scale_label(n.linears[(i, "up")], "o", s, divide=True)
    else:  # producer: the RMSNorm gain
        key = (i, "attn" if which == "attn_in" else "mlp")
        if n.gains[key] is None:
            raise ValueError("no gain to absorb a diagonal gauge; it would have to be stored")
        n.gains[key] = n.gains[key] / s
    for name in readers:
        n.linears[(i, name)] = scale_label(n.linears[(i, name)], "i", s)
    return n
