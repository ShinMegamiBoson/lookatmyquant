"""Triton kernels: fused decode + matmul straight from LookAtMyQuant packed codes.

`packed_matmul` computes `Y[p, r] = scale[e, r] * sum_j X[row(p), j] * Q[e, r, j]` where `Q` is decoded tile by tile
from trellis codes (`ht*`, `rtr*`) or Lloyd-Max codes (`lm6`, `lm8`) inside the kernel and fed to `tl.dot`.
Rows of `X` are already rotated (`rht_in`). One launch handles either a dense matrix (one "expert") or a set of
experts with tokens grouped per expert, as in a fused MoE: `block_expert[b]` names the expert of block `b` and
`block_pairs[b * BLOCK_T + i]` the output row it fills (-1 = padding).
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .formats import FMT_PARAMS

KIND_TRELLIS, KIND_LM8, KIND_LM6, KIND_NVFP4 = 0, 1, 2, 3


@triton.jit
def _packed_mm_kernel(
    X, C, S, LUT, LEV, Y,
    block_expert, block_pairs, SB, E4M3,
    M, N, WPR, LEV_STRIDE,
    x_div, stride_x, stride_y,
    KIND: tl.constexpr, KBITS: tl.constexpr, G: tl.constexpr, LOGK: tl.constexpr, HASH: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PRECISE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    e = tl.load(block_expert + pid_b).to(tl.int64)
    if e < 0:
        return
    t_offs = tl.arange(0, BLOCK_T)
    pairs = tl.load(block_pairs + pid_b * BLOCK_T + t_offs)
    t_valid = pairs >= 0
    xrow = tl.where(t_valid, pairs // x_div, 0).to(tl.int64)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    r_valid = rows < M
    rows_c = tl.where(r_valid, rows, 0).to(tl.int64)
    acc = tl.zeros((BLOCK_T, BLOCK_M), dtype=tl.float32)
    MASK32: tl.constexpr = 0xFFFFFFFF

    for n0 in range(0, N, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        if KIND == 0:
            last = (cols // G) * G + (G - 1)
            bp = (N - 1 - last) * KBITS
            wi = bp >> 5
            off = (bp & 31).to(tl.int64)
            base = C + (e * M + rows_c[:, None]) * WPR
            w0 = tl.load(base + wi[None, :]).to(tl.int64) & MASK32
            w1 = tl.load(base + wi[None, :] + 1, mask=(wi[None, :] + 1) < WPR, other=0).to(tl.int64) & MASK32
            s = ((w0 | (w1 << 32)) >> off[None, :]) & 4095
            if HASH != 0:
                idx = ((s * HASH) & MASK32) >> (32 - LOGK)
            else:
                idx = s
            w = tl.load(LUT + idx * G + (cols % G)[None, :]).to(tl.float32)
        elif KIND == 1:
            b = tl.load(C + (e * M + rows_c[:, None]) * N + cols[None, :]).to(tl.int32) & 0xFF
            w = tl.load(LEV + e * LEV_STRIDE + b).to(tl.float32)
        elif KIND == 3:  # NVFP4: E2M1 nibble (low first) x E4M3 scale of its 16-weight block
            rbase = e * M + rows_c[:, None]
            byte = tl.load(C + rbase * (N // 2) + (cols // 2)[None, :]).to(tl.int32) & 0xFF
            nib = (byte >> ((cols % 2) * 4)[None, :]) & 15
            sb = tl.load(SB + rbase * (N // 16) + (cols // 16)[None, :]).to(tl.int32) & 0xFF
            w = tl.load(LUT + nib) * tl.load(E4M3 + sb)
        else:
            flat = (e * M + rows_c[:, None]) * N + cols[None, :]  # index of the weight in the row-major code stream
            grp = flat // 4
            sub = (flat % 4).to(tl.int32)
            b0 = tl.load(C + grp * 3).to(tl.int32) & 0xFF
            b1 = tl.load(C + grp * 3 + 1).to(tl.int32) & 0xFF
            b2 = tl.load(C + grp * 3 + 2).to(tl.int32) & 0xFF
            v = b0 | (b1 << 8) | (b2 << 16)
            q = (v >> (sub * 6)) & 63
            w = tl.load(LEV + e * LEV_STRIDE + q).to(tl.float32)
        x = tl.load(X + xrow[:, None] * stride_x + cols[None, :], mask=t_valid[:, None], other=0.0).to(tl.float32)
        xh = x.to(tl.float16)
        wh = w.to(tl.float16)
        acc = tl.dot(xh, tl.trans(wh), acc)
        if PRECISE:  # x = xh + xl and w = wh + wl in two float16 parts each: ~float32 products on tensor cores
            xl = (x - xh.to(tl.float32)).to(tl.float16)
            acc = tl.dot(xl, tl.trans(wh), acc)
            if KIND == 1 or KIND == 2:  # trellis tables and NVFP4 products are exact in float16; Lloyd-Max levels are float32
                wl = (w - wh.to(tl.float32)).to(tl.float16)
                acc = tl.dot(xh, tl.trans(wl), acc)

    if KIND == 3:  # one outer scale per expert
        acc = acc * tl.load(S + e).to(tl.float32)
    else:
        sc = tl.load(S + e * M + rows_c, mask=r_valid, other=0.0).to(tl.float32)
        acc = acc * sc[None, :]
    yp = Y + tl.where(t_valid, pairs, 0).to(tl.int64)[:, None] * stride_y + rows[None, :]
    tl.store(yp, acc, mask=t_valid[:, None] & r_valid[None, :])


def group_by_expert(expert_of_pair: torch.Tensor, num_experts: int, block_t: int):
    """Sorts pairs by expert into blocks of `block_t` (padded with -1). Returns (block_expert int32, block_pairs int32)."""
    dev = expert_of_pair.device
    order = torch.argsort(expert_of_pair, stable=True)
    counts = torch.bincount(expert_of_pair, minlength=num_experts)
    nblk = (counts + block_t - 1) // block_t
    blk_start = torch.cumsum(nblk, 0) - nblk
    total_blocks = int(nblk.sum().item())
    pair_start = torch.cumsum(counts, 0) - counts
    sorted_e = expert_of_pair[order]
    rank = torch.arange(order.numel(), device=dev) - pair_start[sorted_e]
    slot = (blk_start[sorted_e] * block_t + rank).long()
    block_pairs = torch.full((total_blocks * block_t,), -1, dtype=torch.int32, device=dev)
    block_pairs[slot] = order.to(torch.int32)
    block_expert = torch.repeat_interleave(torch.arange(num_experts, device=dev, dtype=torch.int32), nblk)
    return block_expert, block_pairs


def packed_matmul(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    fmt: str,
    m: int,
    n: int,
    lut: torch.Tensor | None = None,
    levels: torch.Tensor | None = None,
    block_expert: torch.Tensor | None = None,
    block_pairs: torch.Tensor | None = None,
    n_pairs: int | None = None,
    x_div: int = 1,
    BLOCK_T: int = 16,
    BLOCK_M: int | None = None,
    BLOCK_N: int | None = None,
    out: torch.Tensor | None = None,
    precise: bool = True,
) -> torch.Tensor:
    """See module doc. `codes` is `[E, m, wpr]` int32 (trellis) or `[E, m*n*k/8]` uint8 (lm); `scales` `[E, m]`.
    `fmt="nvfp4"`: codes `[E, m, n/2]` uint8, `scales` the block scales `[E, m, n/16]` uint8, `levels` the outer
    scales `[E]` float32; X is the UNROTATED activation."""
    if BLOCK_M is None or BLOCK_N is None:
        from .tuning import tuning_for

        bm, bn = tuning_for(x.device).prefill
        BLOCK_M, BLOCK_N = BLOCK_M or bm, BLOCK_N or bn
    if block_expert is None:  # dense: one expert, X rows = output rows
        T = x.shape[0]
        n_pairs = T
        nb = triton.cdiv(T, BLOCK_T)
        block_expert = torch.zeros(nb, dtype=torch.int32, device=x.device)
        bp = torch.arange(nb * BLOCK_T, dtype=torch.int32, device=x.device)
        block_pairs = torch.where(bp < T, bp, -1)
    y = out if out is not None else torch.empty(n_pairs, m, dtype=torch.float32, device=x.device)
    sb = e4m3 = scales  # dummies unless NVFP4
    if fmt == "nvfp4":
        from .formats import nvfp4_tables

        kind, kbits, g, K, hm, wpr, lev_stride = KIND_NVFP4, 4, 1, 2, 0, 0, 0
        lut_arg, e4m3 = nvfp4_tables(x.device)
        lev = lut_arg
        sb, scales = scales, levels  # block scales via SB; the per-expert outer scale takes S's place
    elif fmt.startswith("lm"):
        kind = KIND_LM8 if fmt == "lm8" else KIND_LM6
        kbits, g, K, hm = int(fmt[2:]), 1, 2, 0
        lev = levels.reshape(-1, levels.shape[-1]) if levels.dim() > 1 else levels[None]
        lev_stride, lut_arg, wpr = lev.shape[-1] if lev.shape[0] > 1 else 0, lev, 0
    else:
        kind = KIND_TRELLIS
        kbits, g, K, hm = FMT_PARAMS[fmt]
        lev, lev_stride, lut_arg, wpr = lut, 0, lut, (n * kbits) // 32
    grid = (triton.cdiv(m, BLOCK_M), block_expert.numel())
    with torch.cuda.device(x.device):
        _launch(grid, x, codes, scales, lut_arg, lev, y, block_expert, block_pairs, m, n, wpr, lev_stride, x_div,
                kind, kbits, g, K, hm, BLOCK_T, BLOCK_M, BLOCK_N, precise, sb, e4m3)
    return y


def _launch(grid, x, codes, scales, lut_arg, lev, y, block_expert, block_pairs, m, n, wpr, lev_stride, x_div,
            kind, kbits, g, K, hm, BLOCK_T, BLOCK_M, BLOCK_N, precise, sb, e4m3):
    _packed_mm_kernel[grid](
        x, codes, scales, lut_arg, lev, y, block_expert, block_pairs, sb, e4m3,
        m, n, wpr, lev_stride, x_div, x.stride(0), y.stride(0),
        KIND=kind, KBITS=kbits, G=g, LOGK=int(round(math.log2(K))), HASH=hm,
        BLOCK_T=BLOCK_T, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, PRECISE=precise,
    )
