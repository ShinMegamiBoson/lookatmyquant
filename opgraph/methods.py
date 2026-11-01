"""Every method as a word in the frame's operators.

    RTN        Split(index) → Split(scale) → Restrict(ℤ) → Split(offset) → Restrict(int grid)
    GPTQ       sweep over columns: [per group: choose Split(scale, offset) from the running core]
               Restrict(column) → Refit(rest); then the RTN word on the columns as they were when fixed
    AWQ        Gauge(diag, folded into the norm gain / up rows) → Restrict(box) → RTN word
    SVD        Split(identity) → Split(SVD) → Truncate → Contract
    SVD-LLM    Split(identity) → Gauge(whitening L, H = L Lᵀ) → Split(SVD) → Truncate → Contract
    SliceGPT   Untie → Contract(norm gains) → per segment: Gauge(canonical, group O) → Truncate
    CALDERA-style  Split(sum) → [branch B: Split(SVD) → Truncate(k)] + [branch A: GPTQ sweep → RTN word],
               with B chosen by alternating: A = GPTQ(W − B), B = best rank-k approximation of W − Â under H
    Magnitude  Mask(|W| rule)
    Wanda      Mask(|W|·√diag H rule)
    SparseGPT  sweep over columns: [per block: choose Mask from the running core] Mask(column) → Refit(rest)
"""

from __future__ import annotations

import torch

from .frame import (INT, ZZ, Core, split_sum, Environment, ModelNet, Net, canonical_gauge, contract, covariance, fold_gain, gauge, gauge_diag,
                    gauge_segment, insert_identity, mask, matrix_net, project_int, restrict, restrict_box, segments, split_index,
                    split_offset, split_scale, split_svd, truncate, truncate_segment, untie)
from .model import INPUT_OF, LAYERS, LINEARS, attention, attn_sublayer, batched, mlp, mlp_sublayer
from .rules import awq_search_clip, awq_search_scale

# ----------------------------------------------------------------------------- grid rules (arguments of the splits)


def gptq_grid(X: torch.Tensor, maxq: int):
    """Scale and integer offset per row of X, with zero inside the range (GPTQ's asymmetric min/max grid)."""
    z = torch.zeros(X.shape[0], dtype=X.dtype)
    lo, hi = torch.minimum(X.amin(1), z), torch.maximum(X.amax(1), z)
    both = (lo == 0) & (hi == 0)
    lo, hi = torch.where(both, -1.0, lo).to(X.dtype), torch.where(both, 1.0, hi).to(X.dtype)
    S = (hi - lo) / maxq
    return S, torch.round(-lo / S)


def awq_grid(X: torch.Tensor, maxq: int):
    """Scale and integer offset per row of X from its own min/max (AWQ's grid)."""
    hi, lo = X.amax(1), X.amin(1)
    S = (hi - lo).clamp(min=1e-5) / maxq
    return S, (-torch.round(lo / S)).clamp(0, maxq)


def quantize_core(net: Net, core: str, label: str | None, bits: int, gs: int, grid) -> Net:
    """The RTN word on one core: Split(index) → Split(scale) → Restrict(ℤ) → Split(offset) → Restrict(grid).
    Rounding comes before the offset: round-half-to-even does not commute with odd integer shifts."""
    if label is not None:
        net = split_index(net, label, (("g", net.dim(label) // gs), ("w", gs)))
    c = net.cores[core]
    X = c.data.permute(c.axis("o"), c.axis("g"), c.axis("w"))
    o, g, w = X.shape
    S, Z = grid(X.reshape(o * g, w), 2**bits - 1)
    net = split_scale(net, core, S.reshape(o, g), ("o", "g"), core + ".S")
    net = restrict(net, core, ZZ)
    net = split_offset(net, core, Z.reshape(o, g), ("o", "g"), core + ".Z", INT(bits))
    return restrict(net, core, INT(bits))


def codes(net: Net) -> torch.Tensor:
    c = net.cores["W"]
    return c.data.reshape(c.data.shape[0], -1)


# ----------------------------------------------------------------------------- per-matrix words


def rtn(W: torch.Tensor, bits: int, gs: int) -> Net:
    return quantize_core(matrix_net(W.cpu().double()), "W", "i", bits, gs, gptq_grid)


def gptq(W: torch.Tensor, H: torch.Tensor, bits: int, gs: int) -> Net:
    fixed, S, Z = gptq_sweep(W, H, bits, gs)
    return affine_word(fixed, S, Z, bits, gs)


def affine_word(fixed: torch.Tensor, S: torch.Tensor, Z: torch.Tensor, bits: int, gs: int) -> Net:
    """The RTN word with given splits: Split(index) → Split(scale) → Restrict(ℤ) → Split(offset) → Restrict(grid)."""
    dom = INT(bits)
    net = split_index(matrix_net(fixed), "i", (("g", fixed.shape[1] // gs), ("w", gs)))
    net = split_scale(net, "W", S, ("o", "g"), "W.S")
    net = restrict(net, "W", ZZ)
    net = split_offset(net, "W", Z, ("o", "g"), "W.Z", dom)
    return restrict(net, "W", dom)


def gptq_sweep(W: torch.Tensor, H: torch.Tensor, bits: int, gs: int):
    """Fix columns in order: choose each group's scale/offset from the running core, Restrict the column, Refit the rest.
    Returns the columns as they stood when fixed, and the chosen scales and offsets."""
    env = Environment(H)
    T = W.cpu().double().clone()
    T[:, env.dead] = 0
    o, n = T.shape
    maxq, dom = 2**bits - 1, INT(bits)
    fixed = torch.empty_like(T)
    S, Z = torch.empty(o, n // gs, dtype=T.dtype), torch.empty(o, n // gs, dtype=T.dtype)
    for j in range(n):
        g = j // gs
        if j % gs == 0:
            S[:, g], Z[:, g] = gptq_grid(T[:, j : j + gs], maxq)
        fixed[:, j] = T[:, j]
        p = project_int(project_int(T[:, j] / S[:, g], ZZ) + Z[:, g], dom)
        env.refit_after(T, j, T[:, j] - S[:, g] * (p - Z[:, g]))
    return fixed, S, Z


def gptq_lazy(W: torch.Tensor, H: torch.Tensor, bits: int, gs: int) -> Net:
    """GPTQ through the generic sweep, whose Refit is applied lazily (as GPTQ's reference code does): the same word as
    gptq, a different order of floating-point operations, several times faster."""
    from .syntax import sweep
    W = W.cpu().double()
    o, n = W.shape
    maxq, dom = 2**bits - 1, INT(bits)
    S, Z = torch.empty(o, n // gs, dtype=W.dtype), torch.empty(o, n // gs, dtype=W.dtype)

    def on_block(j, T):
        S[:, j // gs], Z[:, j // gs] = gptq_grid(T[:, j : j + gs], maxq)

    def project(j, col, T):
        g = j // gs
        p = project_int(project_int(col / S[:, g], ZZ) + Z[:, g], dom)
        return S[:, g] * (p - Z[:, g])

    before = sweep(W, Environment(H), project, block=gs, on_block=on_block)
    return affine_word(before, S, Z, bits, gs)


def lowrank_plus_quant(W: torch.Tensor, H: torch.Tensor, bits: int, gs: int, rank: int, iters: int = 2) -> Net:
    """CALDERA-style W ≈ Q + U·V, with 16-bit factors. The rule alternates: quantize W − B with GPTQ, then set B to the
    best rank-k approximation of W − Q̂ in the geometry of the inputs (SVD after whitening by L, H = L Lᵀ)."""
    W = W.cpu().double()
    L = _whitening(H.double())
    Linv = torch.linalg.inv(L)
    B = torch.zeros_like(W)
    for _ in range(iters):
        Qhat = affine_word(*gptq_sweep(W - B, H, bits, gs), bits, gs).matrix()
        U, S, Vh = torch.linalg.svd((W - Qhat) @ L, full_matrices=False)
        B = (U[:, :rank] * S[:rank]) @ Vh[:rank] @ Linv
    net = split_sum(matrix_net(W), "W", B, "B")                     # exact: W = (W − B) + B
    net = truncate(split_svd(net, "B", ("o",), ("i",), "r"), "r", rank)  # exact up to round-off: B has rank k
    fixed, Sc, Zc = gptq_sweep(net.cores["W"].data, H, bits, gs)    # lossy: GPTQ on branch A = W − B
    net.cores["W"].data = fixed
    net = split_index(net, "i", (("g", W.shape[1] // gs), ("w", gs)))
    net = split_scale(net, "W", Sc, ("o", "g"), "W.S")
    net = restrict(net, "W", ZZ)
    net = split_offset(net, "W", Zc, ("o", "g"), "W.Z", INT(bits))
    return restrict(net, "W", INT(bits))


def _whitening(H: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.cholesky(H)
    except torch.linalg.LinAlgError:
        eig = torch.linalg.eigvalsh(H)
        return torch.linalg.cholesky(H + (-eig[0] + 1e-6) * torch.eye(len(H), dtype=H.dtype))


def svd(W: torch.Tensor, H: torch.Tensor | None, keep: float) -> Net:
    W = W.cpu().double()
    k = int(W.shape[0] * W.shape[1] * keep / (W.shape[0] + W.shape[1]))
    net = insert_identity(matrix_net(W), "i", "b")
    if H is not None:
        net = gauge(net, "b", _whitening(H.double()), left="W", right="I[b]")
    net = truncate(split_svd(net, "W", ("o",), ("b",), "r"), "r", k)
    return contract(net, "W.V", "I[b]", "W.V")


def magnitude(W: torch.Tensor, sparsity: float) -> Net:
    W = W.cpu().double()
    score = W.abs()
    thresh = torch.kthvalue(score.flatten(), int(W.numel() * sparsity) + 1).values
    return mask(matrix_net(W), "W", score > thresh)


def wanda(W: torch.Tensor, H: torch.Tensor, sparsity: float) -> Net:
    W = W.cpu().double()
    score = W.abs() * torch.sqrt(torch.diag(H).double()).reshape(1, -1)
    kth = torch.kthvalue(score, int(W.shape[1] * sparsity), dim=1).values
    return mask(matrix_net(W), "W", score > kth[:, None])


def sparsegpt(W: torch.Tensor, H: torch.Tensor, sparsity: float, block: int = 128) -> Net:
    fixed, keep = sparsegpt_sweep(W, H, sparsity, block)
    return mask(matrix_net(fixed), "W", keep)


def sparsegpt_sweep(W: torch.Tensor, H: torch.Tensor, sparsity: float, block: int = 128):
    """Fix columns in order: choose each block's mask from the running core, Mask the column, Refit the rest."""
    env = Environment(H)
    T = W.cpu().double().clone()
    T[:, env.dead] = 0
    n = T.shape[1]
    keep = torch.ones_like(T, dtype=torch.bool)
    fixed = torch.empty_like(T)
    u = env.U.diagonal()
    for j in range(n):
        if j % block == 0:
            b = slice(j, min(j + block, n))
            score = T[:, b] ** 2 / u[b] ** 2
            thresh = torch.sort(score.flatten())[0][int(score.numel() * sparsity)]
            keep[:, b] = score > thresh
        fixed[:, j] = T[:, j]
        env.refit_after(T, j, T[:, j] * ~keep[:, j])
    return fixed, keep


# ----------------------------------------------------------------------------- AWQ


def awq_scale_candidate(bits: int, gs: int):
    """Gauge(diag s) on the input bond, then the RTN word; the inverse gauge stays on the input side."""

    def fn(weights, s):
        W = torch.cat([w.cpu().double() for w in weights])
        sd = s.cpu().double()
        net = insert_identity(matrix_net(W), "i", "b")
        net.groups["b"] = "diag"
        net = gauge(net, "b", torch.diag(sd), left="W", right="I[b]", Ginv=torch.diag(1 / sd))
        Wq = quantize_core(net, "W", "b", bits, gs, awq_grid).matrix()
        out, r = [], 0
        for w in weights:
            out.append(Wq[r : r + w.shape[0]].to(w.device, w.dtype))
            r += w.shape[0]
        return out

    return fn


def awq_clip_candidate(bits: int, gs: int):
    """Restrict(box) then the RTN word."""

    def fn(Wg, max_val):
        net = Net({"W": Core(Wg, ("o", "g", "w"))}, [(1.0, ("W",))], ("o", "g", "w"))
        net = restrict_box(net, "W", -max_val, max_val, ("o", "g"))
        return quantize_core(net, "W", None, bits, gs, awq_grid).value()

    return fn


def awq_layer(net: ModelNet, i: int, feats, bits: int, gs: int, cos, sin):
    x = feats.x
    W = {n: net.linears[(i, n)].matrix() for n in LINEARS}
    cand = awq_scale_candidate(bits, gs)
    s_attn, r_attn = awq_search_scale(x["attn_in"], lambda ws: batched(lambda xb: attention(xb, *ws, W["o"], cos, sin)[0], x["attn_in"]),
                                      [W["q"], W["k"], W["v"]], cand)
    s_mlp, r_mlp = awq_search_scale(x["mlp_in"], lambda ws: batched(lambda xb: mlp(xb, ws[0], ws[1], W["down"])[0], x["mlp_in"]),
                                    [W["gate"], W["up"]], cand)
    s_down, r_down = awq_search_scale(x["down_in"], lambda ws: x["down_in"] @ ws[0].T, [W["down"]], cand)
    for bond, s in (("attn_in", s_attn), ("mlp_in", s_mlp), ("down_in", s_down)):
        net = gauge_diag(net, (i, bond), s)
    # The environments live on the gauged bonds, so they transform with the gauge.
    xs = {"attn_in": x["attn_in"] / s_attn, "o_in": x["o_in"], "mlp_in": x["mlp_in"] / s_mlp, "down_in": x["down_in"] / s_down}
    clips = {}
    for name in ("v", "o", "gate", "up", "down"):
        lin = net.linears[(i, name)]
        Wn = lin.matrix()
        mv = awq_search_clip(Wn, xs[INPUT_OF[name]], awq_clip_candidate(bits, gs), gs)
        lin = split_index(lin, "i", (("g", Wn.shape[1] // gs), ("w", gs)))
        mvd = mv.to(Wn.device, Wn.dtype)
        net.linears[(i, name)] = restrict_box(lin, "W", -mvd, mvd, ("o", "g"))
        clips[name] = mv
    return net, {"ratios": (r_attn, r_mlp, r_down), "clips": clips}


def awq_final(net: ModelNet, bits: int, gs: int):
    art = {}
    for key, lin in list(net.linears.items()):
        c = lin.cores["W"]
        lin = Net({"W": Core(c.data.cpu().double(), c.labels)}, [(1.0, ("W",))], lin.out)
        q = quantize_core(lin, "W", "i" if "i" in lin.out else None, bits, gs, awq_grid)
        net.linears[key] = q
        art[key] = codes(q)
    return net, art


# ----------------------------------------------------------------------------- SliceGPT


@torch.no_grad()
def slicegpt(model, calib: torch.Tensor, k: int, record: dict | None = None) -> ModelNet:
    from .model import rope

    dev = model.device
    cos, sin = rope(calib.shape[1], dev)
    ids = calib.to(dev)
    net = untie(ModelNet.from_model(model))
    for norm in [(i, p) for i in range(LAYERS) for p in ("attn", "mlp")] + ["final"]:
        net = fold_gain(net, norm)

    def settle(seg, signal):
        """Gauge the segment into the basis that diagonalises its environment, then truncate it."""
        nonlocal net
        H = covariance(signal)
        if record is not None:
            ev = torch.linalg.eigvalsh(H).flip(0).clamp(min=0)
            record[seg] = {"energy_kept": (ev[:k].sum() / ev.sum()).item(), "eigenvalues": ev.tolist()}
        Q = canonical_gauge(H).to(dev, torch.float32)
        net = truncate_segment(gauge_segment(net, seg, Q), seg, k)

    settle("A0", net.embed[ids])
    h = net.embed[ids]
    for i in range(LAYERS):
        L = net.compile().layers[i]
        settle(f"M{i}", batched(lambda x: attn_sublayer(x, L, cos, sin, net.mean_dim), h))
        L = net.compile().layers[i]
        hm = batched(lambda x: attn_sublayer(x, L, cos, sin, net.mean_dim), h)
        settle(f"A{i + 1}", batched(lambda x: mlp_sublayer(x, L, net.mean_dim), hm))
        L = net.compile().layers[i]
        h = batched(lambda x: mlp_sublayer(x, L, net.mean_dim), hm)
    return net
