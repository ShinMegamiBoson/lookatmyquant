"""Rules that choose an operator's arguments, shared by the reference and frame versions of a method.

A method is an operator word plus rules for choosing its arguments (which scale, which clip, which mask). The
completeness check tests the words; where a rule is a search over candidates, both versions run the same search
and differ only in how they build each candidate. These are AWQ's two searches (llm-awq auto_scale / auto_clip).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def awq_search_scale(x: torch.Tensor, block_fn, weights: list[torch.Tensor], candidate_fn, n_grid: int = 20):
    """Choose per-input-channel scales s = mean|x|^α (normalised) minimising the block's output error after
    quantizing the scaled weights. Returns (s, α)."""
    x_max = x.abs().reshape(-1, x.shape[-1]).float().mean(0)
    org = block_fn(weights)
    best_loss, best_ratio, best_s = float("inf"), None, None
    for r in range(n_grid):
        ratio = r / n_grid
        s = x_max.pow(ratio).clamp(min=1e-4).reshape(-1)
        s = s / (s.max() * s.min()).sqrt()
        loss = (org - block_fn(candidate_fn(weights, s))).float().pow(2).mean().item()
        if loss < best_loss:
            best_loss, best_ratio, best_s = loss, ratio, s.clone()
    return best_s, best_ratio


@torch.no_grad()
def awq_search_clip(W: torch.Tensor, x: torch.Tensor, candidate_fn, gs: int, n_grid: int = 20, max_shrink: float = 0.5, n_sample_token: int = 512):
    """Choose a symmetric clip bound per (output row, input group) minimising the group's partial output error.
    Returns max_val of shape (rows, groups)."""
    xs = x.reshape(-1, x.shape[-1])
    xs = xs[0 :: max(1, len(xs) // n_sample_token)]
    X = xs.cpu().double().reshape(len(xs), -1, gs)
    Wg = W.cpu().double().reshape(W.shape[0], -1, gs)
    org_max = Wg.abs().amax(-1)
    org_out = torch.einsum("tgk,ogk->otg", X, Wg)
    best, min_err = org_max.clone(), torch.full_like(org_max, 1e9)
    for i_s in range(int(max_shrink * n_grid)):
        max_val = org_max * (1 - i_s / n_grid)
        q = candidate_fn(Wg, max_val)
        err = (torch.einsum("tgk,ogk->otg", X, q) - org_out).pow(2).mean(1)
        better = err < min_err
        min_err[better], best[better] = err[better], max_val[better]
    return best
