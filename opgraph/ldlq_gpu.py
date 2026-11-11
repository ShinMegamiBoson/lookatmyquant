"""LDLQ on the GPU in float64: the same rounding as nested.multirate_sweep with one rate (a fixed set of levels, in units of each row's
RMS, inputs settled in order with the later ones re-solved under H), written as GPTQ's lazy-batch form. In float32 a whole build's codes
agreed with the CPU's only 80% (57% on the 9,216-wide inputs, whose H is badly conditioned) and its KL was 0.623 against 0.600, so it
works in float64.

    U = ldlq_factor(Hr)                 # once per input statistics (H in the rotated basis)
    Q = ldlq_round(X, U, levels)        # X: rows in units of their RMS; returns the chosen levels
"""

from __future__ import annotations

import torch


def ldlq_factor(Hr: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """The upper Cholesky factor of H⁻¹ (GPTQ's), in double then float, with GPTQ's damping; more if H is too thin to factor."""
    H = Hr.double()
    H = 0.5 * (H + H.T)
    eye = torch.eye(len(H), dtype=H.dtype, device=H.device)
    for d in (damp, 10 * damp, 100 * damp):
        L, info = torch.linalg.cholesky_ex(H + d * H.diagonal().mean() * eye)
        if int(info) == 0:
            break
    assert int(info) == 0, "H is not positive definite even with damping"
    return torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)


@torch.no_grad()
def ldlq_round(X: torch.Tensor, U: torch.Tensor, levels: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Each entry of X (rows × inputs) to a level, inputs in order, each one's rounding error carried into the later inputs through U."""
    X = X.clone().double()
    U = U.double()
    levels = levels.to(X)
    mids = (levels[1:] + levels[:-1]) / 2
    Q = torch.empty_like(X)
    n = X.shape[1]
    for a in range(0, n, block):
        b = min(a + block, n)
        Xb, Ub = X[:, a:b], U[a:b, a:b]
        Err = torch.empty_like(Xb)
        for j in range(b - a):
            x = Xb[:, j]
            q = levels[torch.bucketize(x.contiguous(), mids)]
            Q[:, a + j] = q
            e = (x - q) / Ub[j, j]
            Xb[:, j + 1 :] -= e[:, None] * Ub[j, j + 1 :][None, :]
            Err[:, j] = e
        X[:, b:] -= Err @ U[a:b, b:]
    return Q
