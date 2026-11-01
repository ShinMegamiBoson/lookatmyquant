"""Each method written directly, following its paper and the core routine of its official code, without the frame.

Ports (the algorithmic core, not the surrounding harness):
  RTN, GPTQ   IST-DASLab/gptq: quant.py Quantizer.find_params/quantize, gptq.py GPTQ.fasterquant
              (act_order off, static_groups off, damp 0.01)
  AWQ         mit-han-lab/llm-awq: auto_scale._search_module_scale, auto_clip.auto_clip_layer,
              quantizer.pseudo_quantize_tensor (zero point on)
  SVD-LLM     truncation-aware data whitening (SVD-LLM, Wang et al. 2024), without the follow-up LoRA update
  SliceGPT    computational invariance + per-segment PCA with shortcut matrices (Ashkboos et al. 2024),
              RMSNorm keeps the original width as its mean divisor
  Magnitude, Wanda   locuslab/wanda prune_magnitude / prune_wanda (unstructured, per-row for Wanda)
  SparseGPT   IST-DASLab/sparsegpt SparseGPT.fasterprune (unstructured, blocksize 128, damp 0.01)
"""

from __future__ import annotations

import torch

from .model import HIDDEN, Layer, Model, attn_sublayer, batched, mlp_sublayer, rope

# ----------------------------------------------------------------------------- RTN and GPTQ


def gptq_find_params(x: torch.Tensor, maxq: int, sym: bool = False):
    z = torch.zeros(x.shape[0], dtype=x.dtype)
    xmin = torch.minimum(x.min(1).values, z)
    xmax = torch.maximum(x.max(1).values, z)
    if sym:
        xmax = torch.maximum(xmin.abs(), xmax)
        tmp = xmin < 0
        if torch.any(tmp):
            xmin[tmp] = -xmax[tmp]
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp], xmax[tmp] = -1, +1
    scale = (xmax - xmin) / maxq
    zero = torch.full_like(scale, (maxq + 1) / 2) if sym else torch.round(-xmin / scale)
    return scale, zero


def gptq_quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero), q


def rtn(W: torch.Tensor, bits: int, gs: int):
    W = W.cpu().double()
    maxq = 2**bits - 1
    Q, codes = torch.empty_like(W), torch.empty_like(W)
    for g0 in range(0, W.shape[1], gs):
        s, z = gptq_find_params(W[:, g0 : g0 + gs], maxq)
        Q[:, g0 : g0 + gs], codes[:, g0 : g0 + gs] = gptq_quantize(W[:, g0 : g0 + gs], s[:, None], z[:, None], maxq)
    return Q, codes


def gptq(W: torch.Tensor, H: torch.Tensor, bits: int, gs: int, blocksize: int = 128, percdamp: float = 0.01):
    W = W.cpu().double().clone()
    H = H.double().clone()
    maxq = 2**bits - 1
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    columns = W.shape[1]
    Q, codes = torch.zeros_like(W), torch.zeros_like(W)
    damp = percdamp * torch.mean(torch.diag(H))
    idx = torch.arange(columns)
    H[idx, idx] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)
    scale = zero = None
    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        W1 = W[:, i1:i2].clone()
        Q1, Err1 = torch.zeros_like(W1), torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w, d = W1[:, i], Hinv1[i, i]
            if (i1 + i) % gs == 0:  # as in the official code: grid from the global W, not the block's running W1
                scale, zero = gptq_find_params(W[:, i1 + i : i1 + i + gs], maxq)
            q, c = gptq_quantize(w, scale, zero, maxq)
            Q1[:, i], codes[:, i1 + i] = q, c
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    return Q, codes


# ----------------------------------------------------------------------------- AWQ


def awq_pseudo_quantize(w: torch.Tensor, bits: int, gs: int):
    shape = w.shape
    w = w.reshape(-1, gs)
    max_val, min_val = w.amax(1, keepdim=True), w.amin(1, keepdim=True)
    max_int = 2**bits - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    q = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    return ((q - zeros) * scales).reshape(shape), q.reshape(shape)


def awq_scale_candidate(bits, gs):
    def fn(weights, s):
        sd = s.cpu().double()
        return [(awq_pseudo_quantize(W.cpu().double() * sd, bits, gs)[0] / sd).to(W.device, W.dtype) for W in weights]
    return fn


def awq_clip_candidate(bits, gs):
    def fn(Wg, max_val):  # Wg: (o, groups, gs) float64; max_val: (o, groups)
        return awq_pseudo_quantize(torch.clamp(Wg, -max_val[..., None], max_val[..., None]), bits, gs)[0]
    return fn


# ----------------------------------------------------------------------------- SVD and SVD-LLM


def _whitening(H: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.cholesky(H)
    except torch.linalg.LinAlgError:
        eig = torch.linalg.eigvalsh(H)
        return torch.linalg.cholesky(H + (-eig[0] + 1e-6) * torch.eye(len(H), dtype=H.dtype))


def svd_rank(W: torch.Tensor, keep: float) -> int:
    return int(W.shape[0] * W.shape[1] * keep / (W.shape[0] + W.shape[1]))


def svd_llm(W: torch.Tensor, H: torch.Tensor | None, keep: float):
    """Best rank-k approximation of W in the norm ‖(W − Ŵ)X‖; H = None gives plain SVD (‖W − Ŵ‖)."""
    W = W.cpu().double()
    L = torch.eye(W.shape[1], dtype=torch.float64) if H is None else _whitening(H.double())
    k = svd_rank(W, keep)
    U, S, Vh = torch.linalg.svd(W @ L, full_matrices=False)
    r = S[:k].sqrt()
    return (U[:, :k] * r) @ ((r[:, None] * Vh[:k]) @ torch.linalg.inv(L))


# ----------------------------------------------------------------------------- pruning


def magnitude(W: torch.Tensor, sparsity: float):
    W = W.cpu().double()
    metric = W.abs()
    thresh = torch.sort(metric.flatten())[0][int(W.numel() * sparsity)]
    keep = ~(metric <= thresh)
    return W * keep, keep


def wanda(W: torch.Tensor, H: torch.Tensor, sparsity: float):
    W = W.cpu().double()
    metric = W.abs() * torch.sqrt(torch.diag(H).double()).reshape(1, -1)
    idx = torch.sort(metric, dim=-1, stable=True)[1][:, : int(W.shape[1] * sparsity)]
    pruned = torch.zeros_like(metric, dtype=torch.bool).scatter_(1, idx, True)
    return W * ~pruned, ~pruned


def sparsegpt(W: torch.Tensor, H: torch.Tensor, sparsity: float, blocksize: int = 128, percdamp: float = 0.01):
    W = W.cpu().double().clone()
    H = H.double().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    columns = W.shape[1]
    damp = percdamp * torch.mean(torch.diag(H))
    idx = torch.arange(columns)
    H[idx, idx] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)
    keep_all = torch.ones_like(W, dtype=torch.bool)
    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        W1 = W[:, i1:i2].clone()
        Q1, Err1 = torch.zeros_like(W1), torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
        mask1 = tmp <= thresh
        for i in range(i2 - i1):
            w, d = W1[:, i], Hinv1[i, i]
            q = w.clone()
            q[mask1[:, i]] = 0
            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        W[:, i1:i2] = Q1
        keep_all[:, i1:i2] = ~mask1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    return W, keep_all


# ----------------------------------------------------------------------------- SliceGPT


def _pca(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1, x.shape[-1])
    H = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64)
    for b in range(0, len(x), 8192):
        c = x[b : b + 8192].float()
        H += (c.T @ c).cpu().double()
    evals, evecs = torch.linalg.eigh(H)
    Q = evecs[:, torch.argsort(evals, descending=True)]
    idx = Q.abs().argmax(0)
    return Q * torch.sign(Q[idx, torch.arange(Q.shape[1])])


@torch.no_grad()
def slicegpt(model: Model, calib: torch.Tensor, k: int) -> Model:
    m = model.copy()
    dev = m.device
    # Computational invariance: fold every RMSNorm gain into the layers that read it, untie the head.
    for L in m.layers:
        L.q, L.k, L.v = L.q * L.attn_norm, L.k * L.attn_norm, L.v * L.attn_norm
        L.gate, L.up = L.gate * L.mlp_norm, L.up * L.mlp_norm
        L.attn_norm = L.mlp_norm = None
    m.head = m.head * m.final_norm
    m.final_norm = None
    cos, sin = rope(calib.shape[1], dev)
    ids = calib.to(dev)

    def Qk(x):
        return _pca(x)[:, :k].to(dev, torch.float32)

    h = m.embed[ids]
    Q = Qk(h)
    m.embed = m.embed @ Q
    h = h @ Q
    for L in m.layers:
        Qa = Q
        L.q, L.k, L.v = L.q @ Qa, L.k @ Qa, L.v @ Qa
        L.attn_shortcut = Qa.T  # back to the original basis until the MLP segment is rotated
        Qm = Qk(batched(lambda x: attn_sublayer(x, L, cos, sin, HIDDEN), h))
        L.attn_shortcut = L.attn_shortcut @ Qm
        L.o = Qm.T @ L.o
        L.gate, L.up = L.gate @ Qm, L.up @ Qm
        hm = batched(lambda x: attn_sublayer(x, L, cos, sin, HIDDEN), h)
        L.mlp_shortcut = Qm.T
        Q = Qk(batched(lambda x: mlp_sublayer(x, L, HIDDEN), hm))
        L.mlp_shortcut = L.mlp_shortcut @ Q
        L.down = Q.T @ L.down
        h = batched(lambda x: mlp_sublayer(x, L, HIDDEN), hm)
    m.head = m.head @ Q
    return m
