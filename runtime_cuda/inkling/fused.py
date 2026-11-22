"""Fused Triton kernels for Inkling-2-Mini's small per-token operations (the decode step's glue).

`block_attn_res_fused`: transformers' Inkling2LoopBlockAttentionResidual for up to 4 queries over up to 7 stacked
states, optionally followed by a weighted RMSNorm per query, in one launch (one program a token).

`message_gates_fused`: transformers' Inkling2LoopMessageGate for 1-3 gates at once, in two launches: the rank-64 down
projection (input and output rounded to bf16 as transformers' bf16 `down_proj`), then the two float32 gates.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ld_norm(P, t, H, h, ok, eps):
    x = tl.load(P + t * H + h, mask=ok, other=0.0).to(tl.float32)
    return x * tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)


@triton.jit
def _score(q, x, inv_sqrt_h, live: tl.constexpr):
    if live:
        return tl.sum(q * x, axis=0) * inv_sqrt_h
    else:
        return -1.0e30


@triton.jit
def _bar_kernel(
    Q, S0, S1, S2, S3, S4, S5, S6, LNW, OUT,
    T, H, eps_norm, eps_out,
    NQ: tl.constexpr, NS: tl.constexpr, NORM_OUT: tl.constexpr, BH: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BH)
    ok = h < H
    eps_norm = eps_norm.to(tl.float32)
    eps_out = eps_out.to(tl.float32)
    inv_sqrt_h = 1.0 / tl.sqrt(H * 1.0)
    # normalized stacked states (unweighted RMSNorm), unrolled: Triton has no lists of tensors
    x0 = _ld_norm(S0, t, H, h, ok, eps_norm)
    x1 = _ld_norm(S1, t, H, h, ok, eps_norm) if NS > 1 else x0
    x2 = _ld_norm(S2, t, H, h, ok, eps_norm) if NS > 2 else x0
    x3 = _ld_norm(S3, t, H, h, ok, eps_norm) if NS > 3 else x0
    x4 = _ld_norm(S4, t, H, h, ok, eps_norm) if NS > 4 else x0
    x5 = _ld_norm(S5, t, H, h, ok, eps_norm) if NS > 5 else x0
    x6 = _ld_norm(S6, t, H, h, ok, eps_norm) if NS > 6 else x0
    for qi in tl.static_range(NQ):
        q = tl.load(Q + qi * H + h, mask=ok, other=0.0)
        c0 = _score(q, x0, inv_sqrt_h, True)
        c1 = _score(q, x1, inv_sqrt_h, NS > 1)
        c2 = _score(q, x2, inv_sqrt_h, NS > 2)
        c3 = _score(q, x3, inv_sqrt_h, NS > 3)
        c4 = _score(q, x4, inv_sqrt_h, NS > 4)
        c5 = _score(q, x5, inv_sqrt_h, NS > 5)
        c6 = _score(q, x6, inv_sqrt_h, NS > 6)
        mx = tl.maximum(tl.maximum(tl.maximum(c0, c1), tl.maximum(c2, c3)), tl.maximum(tl.maximum(c4, c5), c6))
        e0, e1, e2, e3 = tl.exp(c0 - mx), tl.exp(c1 - mx), tl.exp(c2 - mx), tl.exp(c3 - mx)
        e4, e5, e6 = tl.exp(c4 - mx), tl.exp(c5 - mx), tl.exp(c6 - mx)
        den = e0 + e1 + e2 + e3 + e4 + e5 + e6
        out = (e0 * x0 + e1 * x1 + e2 * x2 + e3 * x3 + e4 * x4 + e5 * x5 + e6 * x6) / den
        if NORM_OUT:
            out = out * tl.rsqrt(tl.sum(out * out, axis=0) / H + eps_out) * tl.load(LNW + qi * H + h, mask=ok, other=0.0)
        tl.store(OUT + (qi * T + t) * H + h, out, mask=ok)


def block_attn_res_fused(query: torch.Tensor, states: list[torch.Tensor], ln_w: torch.Tensor | None = None, eps: float = 1e-6):
    """query `[Q, H]` (Q <= 4), states: list of `[T, H]` (block states then the partial state, <= 7).
    Returns `[Q, T, H]` float32; with `ln_w` `[Q, H]`, each output goes through RMSNorm(ln_w[q])."""
    NQ, H = query.shape
    NS = len(states)
    assert NQ <= 4 and 1 <= NS <= 7, (NQ, NS)
    T = states[0].shape[0]
    ptrs = [s if s.is_contiguous() else s.contiguous() for s in states] + [states[0]] * (7 - NS)
    out = torch.empty(NQ, T, H, dtype=torch.float32, device=query.device)
    with torch.cuda.device(query.device):
        _bar_kernel[(T,)](query, *ptrs, ln_w if ln_w is not None else query, out, T, H, 1e-6, eps,
                          NQ=NQ, NS=NS, NORM_OUT=ln_w is not None, BH=triton.next_power_of_2(H), num_warps=8)
    return out


@triton.jit
def _gate_down_kernel(HS, MS, DW, G, T, H, R: tl.constexpr, RB: tl.constexpr, BH: tl.constexpr):
    """g[gate, t, r] = bf16(sum_j bf16(cat(norm(h), norm(m)))[j] * DW[gate, r, j])."""
    gate = tl.program_id(0)
    t = tl.program_id(1).to(tl.int64)
    rb = tl.program_id(2)
    h = tl.arange(0, BH)
    ok = h < H
    base = (gate * T + t) * H
    x = tl.load(HS + base + h, mask=ok, other=0.0)
    m = tl.load(MS + base + h, mask=ok, other=0.0)
    xn = (x * tl.rsqrt(tl.sum(x * x, axis=0) / H + 1e-6)).to(tl.bfloat16).to(tl.float32)
    mn = (m * tl.rsqrt(tl.sum(m * m, axis=0) / H + 1e-6)).to(tl.bfloat16).to(tl.float32)
    rows = rb * RB + tl.arange(0, RB)
    wbase = DW + (gate * R + rows[:, None]).to(tl.int64) * (2 * H)
    w_h = tl.load(wbase + h[None, :], mask=ok[None, :], other=0.0).to(tl.float32)
    w_m = tl.load(wbase + H + h[None, :], mask=ok[None, :], other=0.0).to(tl.float32)
    g = tl.sum(w_h * xn[None, :] + w_m * mn[None, :], axis=1)
    tl.store(G + (gate * T + t) * R + rows, g.to(tl.bfloat16).to(tl.float32))


@triton.jit
def _gate_out_kernel(HS, MS, G, FW, MW, OUT, T, H, R: tl.constexpr, BHC: tl.constexpr):
    """out = (1 - tanh(FW @ g)) * h + tanh(MW @ g) * m for a chunk of BHC hidden dims."""
    gate = tl.program_id(0)
    t = tl.program_id(1).to(tl.int64)
    hc = tl.program_id(2) * BHC + tl.arange(0, BHC)
    ok = hc < H
    r = tl.arange(0, R)
    g = tl.load(G + (gate * T + t) * R + r)
    wb = (gate * H + hc[:, None]).to(tl.int64) * R + r[None, :]
    f = tl.sum(tl.load(FW + wb, mask=ok[:, None], other=0.0) * g[None, :], axis=1)
    mm = tl.sum(tl.load(MW + wb, mask=ok[:, None], other=0.0) * g[None, :], axis=1)
    base = (gate * T + t) * H
    x = tl.load(HS + base + hc, mask=ok, other=0.0)
    msg = tl.load(MS + base + hc, mask=ok, other=0.0)
    e2f = tl.exp(2.0 * f)
    e2m = tl.exp(2.0 * mm)
    tf = 1.0 - 2.0 / (e2f + 1.0)  # tanh
    tm = 1.0 - 2.0 / (e2m + 1.0)
    tl.store(OUT + base + hc, (1.0 - tf) * x + tm * msg, mask=ok)


def message_gates_fused(h: torch.Tensor, msg: torch.Tensor, dw: torch.Tensor, fw: torch.Tensor, mw: torch.Tensor):
    """h, msg `[G, T, H]` float32; dw `[G, R, 2H]` bf16; fw, mw `[G, H, R]` float32. Returns `[G, T, H]`."""
    Gn, T, H = h.shape
    R = dw.shape[1]
    h, msg = h.float().contiguous(), msg.float().contiguous()
    g = torch.empty(Gn, T, R, dtype=torch.float32, device=h.device)
    out = torch.empty_like(h)
    RB, BHC = 8, 64
    with torch.cuda.device(h.device):
        _gate_down_kernel[(Gn, T, R // RB)](h, msg, dw, g, T, H, R=R, RB=RB, BH=triton.next_power_of_2(H), num_warps=4)
        _gate_out_kernel[(Gn, T, triton.cdiv(H, BHC))](h, msg, g, fw, mw, out, T, H, R=R, BHC=BHC, num_warps=4)
    return out
