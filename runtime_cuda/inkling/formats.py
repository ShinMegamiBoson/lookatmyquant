"""PyTorch reference decoders for LookAtMyQuant packed Inkling-2-Mini weights.

A port of the format as `runtime_mlx/inkling/kernels.py` and `model.py` define it:

- Trellis codes (`ht1..ht3`, `rtr1..rtr4`): each row is a little-endian stream of uint32 words. For the group of `g`
  columns starting at column `j` (last column `j + g - 1`), the 12-bit state is the window starting at bit
  `(n - 1 - last) * k`. Bits past the last word read as zero. The table index is the state itself (`rtr*`) or
  `(state * 0x9E3779B1 mod 2^32) >> (32 - log2 K)` (`ht*`). The group's `g` values are `lut[idx]`, times the row scale.
- Lloyd-Max codes (`lm6`, `lm8`): one index per weight, row-major; `lm6` packs 4 indices into 3 bytes. The value is
  `levels[idx]` times the row scale.
- NVFP4 experts (format id 7, upstream 5106923): row-major E2M1 nibbles, low nibble first (`[E, m, n/2]` uint8), one
  E4M3 scale byte per 16 weights (`[E, m, n/16]`) and one float32 outer scale per expert. They read the activation
  WITHOUT the Hadamard rotation that trellis experts get.
- Randomized Hadamard transforms: a linear computes `y = rht_out(rht_in(x, signs_in) @ Q.T, signs_out)`, with the
  output rotation only when the meta says `out_rot`. Experts and the LM head use only the input rotation.

Everything here is plain PyTorch and meant to be exact, not fast.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

from ..format_metadata import E2M1, EXPERT_FIDS, FMT_PARAMS, HASH_MULT_U32, ID_TO_FMT, NVFP4_FID, e4m3_table_np

_MASK32 = 0xFFFFFFFF


# ----------------------------------------------------------------------------- Hadamard / RHT

def _is_prime(q: int) -> bool:
    return q > 1 and all(q % d for d in range(2, math.isqrt(q) + 1))


def _sylvester_np(a: int) -> np.ndarray:
    H = np.ones((1, 1), dtype=np.float64)
    while H.shape[0] < (1 << a):
        H = np.block([[H, H], [H, -H]])
    return H


def _paley_np(q: int) -> np.ndarray:
    squares = {(x * x) % q for x in range(1, q)}
    chi = np.array([0.0] + [1.0 if a in squares else -1.0 for a in range(1, q)], dtype=np.float64)
    idx = (np.arange(q)[None, :] - np.arange(q)[:, None]) % q
    S = np.zeros((q + 1, q + 1), dtype=np.float64)
    S[0, 1:] = 1.0
    S[1:, 0] = -1.0
    S[1:, 1:] = chi[idx]
    return S + np.eye(q + 1, dtype=np.float64)


@lru_cache(maxsize=32)
def hadamard_factors_np(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(A, B) with H = A ⊗ B, the same construction as `runtime_mlx.inkling.kernels.hadamard_factors_mx`."""
    for a in range(int(math.log2(n)), -1, -1):
        if n % (1 << a):
            continue
        p = n // (1 << a)
        if p in (1, 2) or (_is_prime(p - 1) and (p - 1) % 4 == 3):
            if p == 1 and a > 1:
                return _sylvester_np(a // 2), _sylvester_np(a - a // 2)
            A = _sylvester_np(a)
            if p == 2:
                B = np.array([[1.0, 1.0], [1.0, -1.0]])
            elif p == 1:
                B = np.ones((1, 1))
            else:
                B = _paley_np(p - 1)
            return A, B
    raise ValueError(f"No Hadamard construction for n={n}")


_FACTOR_CACHE: dict[tuple[int, str, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}


def _factors(n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    key = (n, str(dtype), device)
    if key not in _FACTOR_CACHE:
        A, B = hadamard_factors_np(n)
        _FACTOR_CACHE[key] = (torch.tensor(A, dtype=dtype, device=device), torch.tensor(B, dtype=dtype, device=device))
    return _FACTOR_CACHE[key]


def _rht_in(x, signs, A, B):
    n = x.shape[-1]
    xf = x.float() * signs.float()
    y = A @ xf.reshape(-1, A.shape[0], B.shape[0]) @ B.T
    return (y * (1.0 / math.sqrt(n))).reshape(x.shape)


def _rht_out(y, signs, A, B):
    m = y.shape[-1]
    z = (A.T @ y.float().reshape(-1, A.shape[0], B.shape[0]) @ B).reshape(y.shape)
    return z * (signs.float() * (1.0 / math.sqrt(m)))


def rht_in_stack(xs: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """`xs[i] @ R_in(signs[i]).T` for a stack `xs` `[S, T, n]` and `signs` `[S, n]`: one batched rotation, as the
    MLX runtime's multi-sign `rht_in_2304_fast` (the five attention matrices that read the same input)."""
    if xs.is_cuda:
        from .rht import rht
        return rht(xs, signs, False)
    A, B = _factors(xs.shape[-1], xs.device)
    return _rht_in(xs, signs[:, None, :], A, B)


def rht_out_stack(ys: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    if ys.is_cuda:
        from .rht import rht
        return rht(ys, signs, True)
    A, B = _factors(ys.shape[-1], ys.device)
    return _rht_out(ys, signs[:, None, :], A, B)


def rht_in(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """`x @ R_in.T` with `R_in = H diag(signs) / sqrt(n)` (MLX `rht_apply_in`). Computes in float32."""
    if x.is_cuda:  # one Triton launch, tf32x3 tensor cores (rht.py)
        from .rht import rht
        return rht(x.reshape(-1, x.shape[-1]), signs, False).reshape(x.shape)
    A, B = _factors(x.shape[-1], x.device)
    return _rht_in(x, signs, A, B)


def rht_out(y: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """`y @ R_out` (MLX `rht_apply_out`). Computes in float32."""
    if y.is_cuda:
        from .rht import rht
        return rht(y.reshape(-1, y.shape[-1]), signs, True).reshape(y.shape)
    A, B = _factors(y.shape[-1], y.device)
    return _rht_out(y, signs, A, B)


# ----------------------------------------------------------------------------- Code decoders

def trellis_states(codes: torch.Tensor, n: int, k: int, g: int) -> torch.Tensor:
    """12-bit states `[..., m, n // g]` (int64) from codes `[..., m, n*k/32]` (int32 or uint32 bit patterns)."""
    w = codes.to(torch.int64) & _MASK32
    wpr = w.shape[-1]
    assert wpr == (n * k) // 32, (wpr, n, k)
    w = torch.cat([w, torch.zeros_like(w[..., :1])], dim=-1)  # bits past the end read as zero
    last = torch.arange(g - 1, n, g, device=codes.device, dtype=torch.int64)
    bp = (n - 1 - last) * k
    wi, off = bp >> 5, bp & 31
    pair = w[..., wi] | (w[..., wi + 1] << 32)
    return (pair >> off) & 4095


def trellis_indices(states: torch.Tensor, K: int, hash_mult: int) -> torch.Tensor:
    if not hash_mult:
        return states
    shift = 32 - int(round(math.log2(K)))
    return ((states * hash_mult) & _MASK32) >> shift


def decode_trellis(codes: torch.Tensor, scales: torch.Tensor, lut: torch.Tensor, n: int, fmt: str) -> torch.Tensor:
    """Dense `Q * scale` `[..., m, n]` in float32 (still in the rotated basis)."""
    k, g, K, hmult = FMT_PARAMS[fmt]
    idx = trellis_indices(trellis_states(codes, n, k, g), K, hmult)
    vals = lut.float().reshape(K, g)[idx]  # [..., m, n/g, g]
    W = vals.reshape(*idx.shape[:-1], n)
    return W * scales.float()[..., None]


def lm_indices(codes: torch.Tensor, m: int, n: int, k: int) -> torch.Tensor:
    """Lloyd-Max indices `[m, n]` (int64) from the flat uint8 code bytes."""
    c = codes.reshape(-1).to(torch.int64) & 0xFF
    if k == 8:
        return c.reshape(m, n)
    assert k == 6, k
    t = c.reshape(-1, 3)
    v = t[:, 0] | (t[:, 1] << 8) | (t[:, 2] << 16)
    idx = torch.stack([v & 63, (v >> 6) & 63, (v >> 12) & 63, (v >> 18) & 63], dim=-1)
    return idx.reshape(m, n)


def decode_lm(codes: torch.Tensor, scales: torch.Tensor, levels: torch.Tensor, m: int, n: int, k: int) -> torch.Tensor:
    idx = lm_indices(codes, m, n, k)
    return levels.float()[idx] * scales.float()[:, None]


def effective_weight(Q: torch.Tensor, signs_in: torch.Tensor, signs_out: torch.Tensor | None = None) -> torch.Tensor:
    """The dense `W` (`y = x @ W.T`) of a packed matrix: `R_out.T @ Q @ R_in`, or `Q @ R_in` without `signs_out`."""
    n = Q.shape[-1]
    eye = torch.eye(n, device=Q.device, dtype=torch.float32)
    Wt = rht_in(eye, signs_in) @ Q.float().T  # = R_in.T Q.T
    if signs_out is not None:
        Wt = rht_out(Wt, signs_out)
    return Wt.T.contiguous()


# ----------------------------------------------------------------------------- NVFP4 (experts)

def nvfp4_tables(device) -> tuple[torch.Tensor, torch.Tensor]:
    """(e2m1 float32[16], e4m3 float32[256]) on `device`."""
    return torch.tensor(E2M1, dtype=torch.float32, device=device), torch.from_numpy(e4m3_table_np()).to(device)


def decode_nvfp4(codes: torch.Tensor, scales: torch.Tensor, outer: torch.Tensor) -> torch.Tensor:
    """Dense float32 `[..., m, n]` from codes `[..., m, n/2]` uint8, block scales `[..., m, n/16]` uint8, outer `[...]`
    (one value per leading index). No rotation: this is the weight the raw activation multiplies."""
    e2m1, e4m3 = nvfp4_tables(codes.device)
    c = codes.to(torch.int64) & 0xFF
    vals = torch.stack((e2m1[c & 0xF], e2m1[c >> 4]), dim=-1).reshape(*c.shape[:-1], -1)  # [..., m, n]
    bs = e4m3[scales.to(torch.int64) & 0xFF]  # [..., m, n/16]
    W = (vals.reshape(*bs.shape, 16) * bs[..., None]).reshape(vals.shape)
    return W * outer.float().reshape(*outer.shape, *([1] * (W.dim() - outer.dim())))
