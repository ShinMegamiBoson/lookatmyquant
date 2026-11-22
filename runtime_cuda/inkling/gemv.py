"""Triton matrix-vector kernel for packed codes, for decode (one or a few token rows a matrix).

The analogue of the half-warp kernels in `runtime_mlx/inkling/kernels.py`. Each row's columns are cut into chunks of
CH columns with CH * k a multiple of 32 bits. Every matrix width here (1152, 2304, 3072) times k is a multiple of 32,
so every chunk starts on a 32-bit word boundary and every bit position inside a chunk is a compile-time constant:
the kernel loads a chunk's NW code words once (2 for k = 1, 2, 4; 4 for k = 3) and extracts each 12-bit trellis state
(or k-bit Lloyd-Max index) with fixed shifts. Arithmetic is float32 FMA on CUDA cores, so results are exact.

Trellis rows store column groups from the high end: the state of the group ending at column `last` starts at bit
`(n - 1 - last) * k`. Lloyd-Max rows (`lm6` packs 4 indices in 3 bytes, little-endian) are a plain bitstream:
index `j` sits at bits `[k j, k j + k)`.
"""
from __future__ import annotations

import math
import weakref

import torch
import triton
import triton.language as tl

from .formats import FMT_PARAMS

from .tuning import tuning_for  # block shapes, warps, stages and two-weight tables per GPU


@triton.jit
def _word(w0, w1, w2, w3, i: tl.constexpr):
    if i == 0:
        return w0
    elif i == 1:
        return w1
    elif i == 2:
        return w2
    else:
        return w3


@triton.jit
def _bits(w0, w1, w2, w3, bit: tl.constexpr, width: tl.constexpr):
    """`width` bits starting at chunk-relative bit `bit`, from the chunk's words (uint32)."""
    wi: tl.constexpr = bit // 32
    off: tl.constexpr = bit % 32
    lo = _word(w0, w1, w2, w3, wi) >> off
    if off + width > 32:
        lo = lo | (_word(w0, w1, w2, w3, wi + 1) << (32 - off))
    return lo & ((1 << width) - 1)


@triton.jit
def _gemv_body(
    X, xrow, C, S, TAB, Y, slot, e, rows, r_ok, M, N, stride_y,
    LM: tl.constexpr, KBITS: tl.constexpr, G: tl.constexpr, LOGK: tl.constexpr, HASH: tl.constexpr,
    CH: tl.constexpr, NW: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, SW: tl.constexpr = 12,
):
    """`Y[slot, rows] = scale[e, rows] * Q[e, rows, :] @ x` for one matrix (or expert) `e` of one code format."""
    WPR = N * KBITS // 32
    row_base = C + (e * M + tl.where(r_ok, rows, 0).to(tl.int64)) * WPR  # [BM] word pointers
    NCH = N // CH
    acc = tl.zeros((BLOCK_M, BLOCK_C), tl.float32)
    for c0 in range(0, NCH, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        c_ok = c < NCH
        if LM:
            wbase = c * (CH * KBITS // 32)
        else:
            wbase = (N - CH * (c + 1)) * KBITS // 32
        ptr = row_base[:, None] + wbase[None, :]
        ok = r_ok[:, None] & c_ok[None, :]
        w0 = tl.load(ptr, mask=ok, other=0).to(tl.uint32, bitcast=True)
        w1 = tl.load(ptr + 1, mask=ok & (wbase[None, :] + 1 < WPR), other=0).to(tl.uint32, bitcast=True)
        w2 = w1
        w3 = w1
        if NW > 2:
            w2 = tl.load(ptr + 2, mask=ok & (wbase[None, :] + 2 < WPR), other=0).to(tl.uint32, bitcast=True)
            w3 = tl.load(ptr + 3, mask=ok & (wbase[None, :] + 3 < WPR), other=0).to(tl.uint32, bitcast=True)
        col0 = c * CH
        for gi in tl.static_range(CH // G):
            if LM:
                idx = _bits(w0, w1, w2, w3, gi * KBITS, KBITS).to(tl.int32)
                xv = tl.load(xrow + col0 + gi, mask=c_ok, other=0.0)
                acc += tl.load(TAB + idx) * xv[None, :]
            else:
                st = _bits(w0, w1, w2, w3, (CH - (gi + 1) * G) * KBITS, SW)
                if HASH != 0:
                    idx = (((st.to(tl.int64) * HASH) & 0xFFFFFFFF) >> (32 - LOGK)).to(tl.int32)
                else:
                    idx = st.to(tl.int32)
                if G == 1:
                    xv = tl.load(xrow + col0 + gi, mask=c_ok, other=0.0)
                    acc += tl.load(TAB + idx).to(tl.float32) * xv[None, :]
                else:
                    # one 4- or 8-byte load fetches the whole group's float16 values (TAB is viewed as int32/int64)
                    packed = tl.load(TAB + idx)
                    for t in tl.static_range(G):
                        h16 = ((packed >> (16 * t)) & 0xFFFF).to(tl.int16)
                        v = h16.to(tl.float16, bitcast=True).to(tl.float32)
                        xv = tl.load(xrow + col0 + gi * G + t, mask=c_ok, other=0.0)
                        acc += v * xv[None, :]
    y = tl.sum(acc, axis=1) * tl.load(S + e * M + rows, mask=r_ok, other=0.0).to(tl.float32)
    tl.store(Y + slot * stride_y + rows, y, mask=r_ok)


@triton.jit
def _gemv_nvfp4_body(xrow, C, SB, OUTER, E2M1, E4M3, Y, slot, e, rows, r_ok, M, N, stride_y,
                     BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    """NVFP4 expert `e`: `Y[slot, rows] = outer[e] * sum_b e4m3(SB[b]) * sum_j e2m1(nibble_j) x_j` over 16-column
    blocks b (two uint32 code words, one scale byte each). `xrow` is the UNROTATED activation."""
    WPR = N // 8
    SPR = N // 16
    r64 = (e * M + tl.where(r_ok, rows, 0).to(tl.int64))
    wbase = C + r64 * WPR
    sbase = SB + r64 * SPR
    acc = tl.zeros((BLOCK_M, BLOCK_C), tl.float32)
    for c0 in range(0, SPR, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        c_ok = c < SPR
        ok = r_ok[:, None] & c_ok[None, :]
        w0 = tl.load(wbase[:, None] + 2 * c[None, :], mask=ok, other=0).to(tl.uint32, bitcast=True)
        w1 = tl.load(wbase[:, None] + 2 * c[None, :] + 1, mask=ok, other=0).to(tl.uint32, bitcast=True)
        bs = tl.load(E4M3 + (tl.load(sbase[:, None] + c[None, :], mask=ok, other=0).to(tl.int32) & 0xFF))
        part = tl.zeros((BLOCK_M, BLOCK_C), tl.float32)
        for j in tl.static_range(16):
            if j < 8:
                nib = (w0 >> (4 * j)) & 15
            else:
                nib = (w1 >> (4 * (j - 8))) & 15
            xv = tl.load(xrow + c * 16 + j, mask=c_ok, other=0.0)
            part += tl.load(E2M1 + nib.to(tl.int32)) * xv[None, :]
        acc += part * bs
    y = tl.sum(acc, axis=1) * tl.load(OUTER + e).to(tl.float32)
    tl.store(Y + slot * stride_y + rows, y, mask=r_ok)


@triton.jit
def _nvfp4_gemv_kernel(X, C, SB, OUTER, E2M1, E4M3, Y, EIDS, M, N, x_div, stride_x, stride_y,
                       BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_m = tl.program_id(0)
    slot = tl.program_id(1)
    e = tl.load(EIDS + slot).to(tl.int64)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    r_ok = rows < M
    if e < 0:
        tl.store(Y + slot * stride_y + rows, tl.zeros((BLOCK_M,), tl.float32), mask=r_ok)
        return
    xrow = X + (slot // x_div).to(tl.int64) * stride_x
    _gemv_nvfp4_body(xrow, C, SB, OUTER, E2M1, E4M3, Y, slot, e, rows, r_ok, M, N, stride_y, BLOCK_M, BLOCK_C)


@triton.jit
def _packed_gemv_kernel(
    X, C, S, TAB, Y, EIDS,
    M, N, WPR, x_div, stride_x, stride_y,
    LM: tl.constexpr, KBITS: tl.constexpr, G: tl.constexpr, LOGK: tl.constexpr, HASH: tl.constexpr,
    CH: tl.constexpr, NW: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, ZERO_SKIPPED: tl.constexpr,
    SW: tl.constexpr,
):
    pid_m = tl.program_id(0)
    slot = tl.program_id(1)
    e = tl.load(EIDS + slot).to(tl.int64)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    r_ok = rows < M
    if e < 0:
        if ZERO_SKIPPED:
            tl.store(Y + slot * stride_y + rows, tl.zeros((BLOCK_M,), tl.float32), mask=r_ok)
        return
    xrow = X + (slot // x_div).to(tl.int64) * stride_x
    _gemv_body(X, xrow, C, S, TAB, Y, slot, e, rows, r_ok, M, N, stride_y,
               LM, KBITS, G, LOGK, HASH, CH, NW, BLOCK_M, BLOCK_C, SW)


@triton.jit
def _moe_gemv_kernel(
    X, EIDS, SLAB_OF, LOC_OF, Y,
    C0, S0, T0, C1, S1, T1, C2, S2, T2, C3, S3, T3,
    XR, CN, SN, ON, E2M1, E4M3, stride_xr,
    M, N, x_div, stride_x, stride_y,
    K0: tl.constexpr, G0: tl.constexpr, L0: tl.constexpr, H0: tl.constexpr, CH0: tl.constexpr, NW0: tl.constexpr, SW0: tl.constexpr,
    K1: tl.constexpr, G1: tl.constexpr, L1: tl.constexpr, H1: tl.constexpr, CH1: tl.constexpr, NW1: tl.constexpr, SW1: tl.constexpr,
    K2: tl.constexpr, G2: tl.constexpr, L2: tl.constexpr, H2: tl.constexpr, CH2: tl.constexpr, NW2: tl.constexpr, SW2: tl.constexpr,
    K3: tl.constexpr, G3: tl.constexpr, L3: tl.constexpr, H3: tl.constexpr, CH3: tl.constexpr, NW3: tl.constexpr, SW3: tl.constexpr,
    GROUPS: tl.constexpr, NSLAB: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, HAS_NV: tl.constexpr,
):
    """All experts of one MoE side in one launch, whatever their code formats (runtime_mlx's UnifiedMoESide): each
    (row block, token-expert pair) program looks up its expert's format slab and local index and runs that decoder.
    Slab index 4 is NVFP4, which reads the unrotated activation XR."""
    pid_m = tl.program_id(0)
    slot = tl.program_id(1)
    if GROUPS > 1:
        blocks_per_group = tl.cdiv(M, BLOCK_M)
        group = pid_m // blocks_per_group
        pid_m = pid_m % blocks_per_group
        Y += group * M
    else:
        group = 0
    eid = tl.load(EIDS + slot) * GROUPS + group
    j = tl.load(SLAB_OF + eid)
    e = tl.load(LOC_OF + eid).to(tl.int64)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    r_ok = rows < M
    xrow = X + (slot // x_div).to(tl.int64) * stride_x
    if j == 0:
        if NSLAB > 0:
            _gemv_body(X, xrow, C0, S0, T0, Y, slot, e, rows, r_ok, M, N, stride_y, False, K0, G0, L0, H0, CH0, NW0, BLOCK_M, BLOCK_C, SW0)
    elif j == 1:
        if NSLAB > 1:
            _gemv_body(X, xrow, C1, S1, T1, Y, slot, e, rows, r_ok, M, N, stride_y, False, K1, G1, L1, H1, CH1, NW1, BLOCK_M, BLOCK_C, SW1)
    elif j == 2:
        if NSLAB > 2:
            _gemv_body(X, xrow, C2, S2, T2, Y, slot, e, rows, r_ok, M, N, stride_y, False, K2, G2, L2, H2, CH2, NW2, BLOCK_M, BLOCK_C, SW2)
    elif j == 3:
        if NSLAB > 3:
            _gemv_body(X, xrow, C3, S3, T3, Y, slot, e, rows, r_ok, M, N, stride_y, False, K3, G3, L3, H3, CH3, NW3, BLOCK_M, BLOCK_C, SW3)
    else:
        if HAS_NV:
            xr = XR + (slot // x_div).to(tl.int64) * stride_xr
            _gemv_nvfp4_body(xr, CN, SN, ON, E2M1, E4M3, Y, slot, e, rows, r_ok, M, N, stride_y, BLOCK_M, 8)


def _chunk(fmt: str):
    if fmt.startswith("lm"):
        k = int(fmt[2:])
        return k, 1, 2, 0, (16 if k == 6 else 4)  # lm6: 16 indices = 96 bits = 3 words; lm8: 4 = 1 word
    k, g, K, hm = FMT_PARAMS[fmt]
    return k, g, K, hm, {1: 32, 2: 16, 3: 32, 4: 8}[k]


# Weights one lookup returns (runtime_mlx's build_rtr_multitable) are chosen per GPU: tuning.Tuning.multitable.
_MT_CACHE: dict[int, tuple] = {}  # id(lut) -> (weakref to that lut, {(k, g): table}); the ref guards against id reuse


def rtr_multitable(lut: torch.Tensor, k: int, g: int) -> torch.Tensor:
    """Bit-exact g-weight table for a 12-bit scalar trellis: index the `12 + (g - 1) k` bits at the group's last
    column; column `j + t` of the group is `lut[(idx >> (g - 1 - t) k) & 4095]`. float16 `[2^bits, g]`."""
    ent = _MT_CACHE.get(id(lut))
    if ent is None or ent[0]() is not lut:
        ent = _MT_CACHE[id(lut)] = (weakref.ref(lut), {})
    per_lut = ent[1]
    if (k, g) not in per_lut:
        vals = lut.reshape(4096)
        idx = torch.arange(1 << (12 + (g - 1) * k), device=lut.device)
        per_lut[(k, g)] = torch.stack([vals[(idx >> ((g - 1 - t) * k)) & 4095] for t in range(g)], dim=-1).contiguous()
    return per_lut[(k, g)]


def gemv_spec(fmt: str, table: torch.Tensor):
    """Kernel constants and the table as the kernel reads it: (lm, k, g, logK, hash, CH, NW, SW, table)."""
    k, g, K, hm, CH = _chunk(fmt)
    lm = fmt.startswith("lm")
    sw = k if lm else 12
    mt = tuning_for(table.device).multitable if table.is_cuda else {}
    if not lm and fmt in mt:
        g = mt[fmt]
        sw = 12 + (g - 1) * k
        table = rtr_multitable(table, k, g)
    if not lm and g > 1:  # a group's g float16 values as one int32 (g = 2) or int64 (g = 4) word
        table = table.contiguous().view(torch.int32 if g == 2 else torch.int64)
    span = (CH * k - k * (1 if lm else g)) + sw  # bits a chunk reads
    return lm, k, g, int(round(math.log2(K))), hm, CH, (4 if span > 64 else 2), sw, table


def packed_gemv(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    fmt: str,
    m: int,
    n: int,
    table: torch.Tensor,
    expert_ids: torch.Tensor | None = None,
    x_div: int = 1,
    out: torch.Tensor | None = None,
    BLOCK_M: int | None = None,
    BLOCK_C: int | None = None,
    num_warps: int | None = None,
    zero_skipped: bool = True,
    num_stages: int | None = None,
) -> torch.Tensor:
    """`out[s] = scale[e_s] * Q[e_s] @ x[s // x_div]` for each slot s. `table` is the trellis LUT or the lm levels.
    `codes`: trellis `[E, m, n*k/32]` int32, lm `[m*n*k/8]` uint8 (one expert). `expert_ids` int32 `[S]`; -1 writes a zero row, or leaves the row untouched with `zero_skipped=False` (so launches
    for different formats can fill one output)."""
    lm, k, g, logk, hm, CH, nw, sw, table = gemv_spec(fmt, table)
    fam = {"ht": "ht", "rt": "rt"}.get(fmt[:2], fmt)
    bm, bc, nwarp, nst = tuning_for(x.device).gemv_cfg(f"single:{fam}")
    BLOCK_M, BLOCK_C = BLOCK_M or bm, BLOCK_C or bc
    num_warps, num_stages = num_warps or nwarp, num_stages or nst
    assert (n * k) % 32 == 0 and n % CH == 0, (fmt, n)
    if expert_ids is None:
        expert_ids = torch.zeros(x.shape[0], dtype=torch.int32, device=x.device)
    S = expert_ids.numel()
    y = out if out is not None else torch.empty(S, m, dtype=torch.float32, device=x.device)
    cw = codes.view(torch.int32) if codes.dtype == torch.uint8 else codes
    wpr = (n * k) // 32
    with torch.cuda.device(x.device):
        _packed_gemv_kernel[(triton.cdiv(m, BLOCK_M), S)](
            x, cw, scales, table, y, expert_ids,
            m, n, wpr, x_div, x.stride(0), y.stride(0),
            LM=lm, KBITS=k, G=g, LOGK=logk, HASH=hm, SW=sw,
            CH=CH, NW=nw, BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C, ZERO_SKIPPED=zero_skipped, num_warps=num_warps, num_stages=num_stages,
        )
    return y


class MoESideGEMV:
    """One launch for all experts of an MoE side. `slabs`: list of (fmt, codes [E_f, m, wpr] int32, scales [E_f, m],
    lut) trellis slabs; `nvfp4`: optional (codes [E_nv, m, n/2] uint8, block scales [E_nv, m, n/16] uint8, outer
    [E_nv] float32). `slab_of` / `loc_of` int32 `[n_experts]` map a global expert id to its slab (0-3 trellis, 4 =
    NVFP4) and its index in it."""

    NV_SLAB = 4

    def __init__(self, slabs, slab_of: torch.Tensor, loc_of: torch.Tensor, m: int, n: int, BLOCK_M: int | None = None,
                 BLOCK_C: int | None = None, nvfp4=None, num_warps: int | None = None, num_stages: int | None = None,
                 groups: int = 1):
        assert len(slabs) <= 4 and (slabs or nvfp4 is not None)
        bm, bc, nwarp, nst = tuning_for(slab_of.device).gemv_cfg("moe")
        self.m, self.n, self.BLOCK_M, self.BLOCK_C = m, n, BLOCK_M or bm, BLOCK_C or bc
        self.groups = groups
        self.group_rows = m // groups
        self.num_warps, self.num_stages = num_warps or nwarp, num_stages or nst
        self.slab_of, self.loc_of = slab_of.to(torch.int32).contiguous(), loc_of.to(torch.int32).contiguous()
        self.ptrs, self.consts = [], {}
        self.has_nv = nvfp4 is not None
        if self.has_nv:
            from .formats import nvfp4_tables

            assert n % 16 == 0, n
            nc, nsb, nout = nvfp4
            self.nv = [nc.view(torch.int32).reshape(nc.shape[0], m, n // 8), nsb.contiguous(), nout.float().contiguous(),
                       *nvfp4_tables(nc.device)]
        for i in range(4):
            if slabs:
                fmt, codes, scales, lut = slabs[min(i, len(slabs) - 1)]
                lm, k, g, logk, hm, CH, nw, sw, table = gemv_spec(fmt, lut)
                assert (n * k) % 32 == 0 and n % CH == 0, (fmt, n)
                self.ptrs += [codes, scales, table]
            else:  # NVFP4 only: placeholders, never read
                k, g, logk, hm, CH, nw, sw = 2, 4, 9, 0, 16, 2, 12
                self.ptrs += [self.nv[0], self.nv[2], self.nv[2]]
            self.consts.update({f"K{i}": k, f"G{i}": g, f"L{i}": logk, f"H{i}": hm, f"CH{i}": CH, f"NW{i}": nw, f"SW{i}": sw})
        self.nslab = len(slabs)
        if not self.has_nv:
            self.nv = [self.ptrs[0], self.ptrs[1], self.ptrs[1], self.ptrs[1], self.ptrs[1]]

    def __call__(self, x: torch.Tensor, eids: torch.Tensor, x_div: int, out: torch.Tensor | None = None,
                 x_raw: torch.Tensor | None = None) -> torch.Tensor:
        """`x` the rotated activation (trellis experts), `x_raw` the unrotated one (NVFP4 experts)."""
        S = eids.numel()
        y = out if out is not None else torch.empty(S, self.m, dtype=torch.float32, device=x.device)
        if self.has_nv:
            assert x_raw is not None, "this side has NVFP4 experts: pass x_raw (the unrotated activation)"
        xr = x_raw if x_raw is not None else x
        with torch.cuda.device(x.device):
            _moe_gemv_kernel[(self.groups * triton.cdiv(self.group_rows, self.BLOCK_M), S)](
                x, eids, self.slab_of, self.loc_of, y, *self.ptrs, xr, *self.nv, xr.stride(0),
                self.group_rows, self.n, x_div, x.stride(0), y.stride(0),
                GROUPS=self.groups, NSLAB=self.nslab, BLOCK_M=self.BLOCK_M, BLOCK_C=self.BLOCK_C, HAS_NV=self.has_nv,
                num_warps=self.num_warps, num_stages=self.num_stages, **self.consts,
            )
        return y


def nvfp4_gemv(x, codes, block_scales, outer, m: int, n: int, expert_ids, x_div: int = 1, out=None,
               BLOCK_M: int | None = None, BLOCK_C: int | None = None, num_warps: int | None = None, num_stages: int | None = None):
    """`out[s] = W_nv[e_s] @ x[s // x_div]` for NVFP4 experts (unrotated x); -1 writes a zero row."""
    from .formats import nvfp4_tables

    S = expert_ids.numel()
    bm, bc, nwarp, nst = tuning_for(x.device).gemv_cfg("single:nvfp4")
    BLOCK_M, BLOCK_C, num_warps, num_stages = BLOCK_M or bm, BLOCK_C or bc, num_warps or nwarp, num_stages or nst
    y = out if out is not None else torch.empty(S, m, dtype=torch.float32, device=x.device)
    e2, e4 = nvfp4_tables(x.device)
    with torch.cuda.device(x.device):
        _nvfp4_gemv_kernel[(triton.cdiv(m, BLOCK_M), S)](
            x, codes.view(torch.int32), block_scales, outer.float(), e2, e4, y, expert_ids,
            m, n, x_div, x.stride(0), y.stride(0), BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C, num_warps=num_warps, num_stages=num_stages)
    return y


@triton.jit
def _dense_stage_kernel(
    X, Y, N, stride_x,
    C0, S0, T0, M0, RB0, XR0, YO0,
    C1, S1, T1, M1, RB1, XR1, YO1,
    C2, S2, T2, M2, RB2, XR2, YO2,
    C3, S3, T3, M3, RB3, XR3, YO3,
    C4, S4, T4, M4, RB4, XR4, YO4,
    C5, S5, T5, M5, RB5, XR5, YO5,
    C6, S6, T6, M6, RB6, XR6, YO6,
    C7, S7, T7, M7, RB7, XR7, YO7,
    LM0: tl.constexpr, K0: tl.constexpr, G0: tl.constexpr, L0: tl.constexpr, H0: tl.constexpr, CH0: tl.constexpr, NW0: tl.constexpr, SW0: tl.constexpr,
    LM1: tl.constexpr, K1: tl.constexpr, G1: tl.constexpr, L1: tl.constexpr, H1: tl.constexpr, CH1: tl.constexpr, NW1: tl.constexpr, SW1: tl.constexpr,
    LM2: tl.constexpr, K2: tl.constexpr, G2: tl.constexpr, L2: tl.constexpr, H2: tl.constexpr, CH2: tl.constexpr, NW2: tl.constexpr, SW2: tl.constexpr,
    LM3: tl.constexpr, K3: tl.constexpr, G3: tl.constexpr, L3: tl.constexpr, H3: tl.constexpr, CH3: tl.constexpr, NW3: tl.constexpr, SW3: tl.constexpr,
    LM4: tl.constexpr, K4: tl.constexpr, G4: tl.constexpr, L4: tl.constexpr, H4: tl.constexpr, CH4: tl.constexpr, NW4: tl.constexpr, SW4: tl.constexpr,
    LM5: tl.constexpr, K5: tl.constexpr, G5: tl.constexpr, L5: tl.constexpr, H5: tl.constexpr, CH5: tl.constexpr, NW5: tl.constexpr, SW5: tl.constexpr,
    LM6: tl.constexpr, K6: tl.constexpr, G6: tl.constexpr, L6: tl.constexpr, H6: tl.constexpr, CH6: tl.constexpr, NW6: tl.constexpr, SW6: tl.constexpr,
    LM7: tl.constexpr, K7: tl.constexpr, G7: tl.constexpr, L7: tl.constexpr, H7: tl.constexpr, CH7: tl.constexpr, NW7: tl.constexpr, SW7: tl.constexpr,
    NSEG: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Several packed matrices of any formats sharing an input width N, one launch (runtime_mlx's DenseStage): the
    grid's row blocks are split across the matrices; segment i reads row XR_i of X and writes Y[YO_i : YO_i + M_i].
    RB_i is the cumulative row-block count."""
    pid = tl.program_id(0)
    if pid < RB0:
        if NSEG > 0:
            rows = (pid - 0) * BLOCK_M + tl.arange(0, BLOCK_M)
            r_ok = rows < M0
            _gemv_body(X, X + XR0 * stride_x, C0, S0, T0, Y + YO0, 0, 0, rows, r_ok, M0, N, 0,
                       LM0, K0, G0, L0, H0, CH0, NW0, BLOCK_M, BLOCK_C, SW0)
    elif pid < RB1:
        if NSEG > 1:
            rows = (pid - RB0) * BLOCK_M + tl.arange(0, BLOCK_M)
            r_ok = rows < M1
            _gemv_body(X, X + XR1 * stride_x, C1, S1, T1, Y + YO1, 0, 0, rows, r_ok, M1, N, 0,
                       LM1, K1, G1, L1, H1, CH1, NW1, BLOCK_M, BLOCK_C, SW1)
    elif pid < RB2:
        if NSEG > 2:
            rows = (pid - RB1) * BLOCK_M + tl.arange(0, BLOCK_M)
            r_ok = rows < M2
            _gemv_body(X, X + XR2 * stride_x, C2, S2, T2, Y + YO2, 0, 0, rows, r_ok, M2, N, 0,
                       LM2, K2, G2, L2, H2, CH2, NW2, BLOCK_M, BLOCK_C, SW2)
    elif pid < RB3:
        if NSEG > 3:
            rows = (pid - RB2) * BLOCK_M + tl.arange(0, BLOCK_M)
            r_ok = rows < M3
            _gemv_body(X, X + XR3 * stride_x, C3, S3, T3, Y + YO3, 0, 0, rows, r_ok, M3, N, 0,
                       LM3, K3, G3, L3, H3, CH3, NW3, BLOCK_M, BLOCK_C, SW3)
    elif pid < RB4:
        if NSEG > 4:
            rows = (pid - RB3) * BLOCK_M + tl.arange(0, BLOCK_M)
            r_ok = rows < M4
            _gemv_body(X, X + XR4 * stride_x, C4, S4, T4, Y + YO4, 0, 0, rows, r_ok, M4, N, 0,
                       LM4, K4, G4, L4, H4, CH4, NW4, BLOCK_M, BLOCK_C, SW4)
    elif pid < RB5:
        if NSEG > 5:
            rows = (pid - RB4) * BLOCK_M + tl.arange(0, BLOCK_M)
            r_ok = rows < M5
            _gemv_body(X, X + XR5 * stride_x, C5, S5, T5, Y + YO5, 0, 0, rows, r_ok, M5, N, 0,
                       LM5, K5, G5, L5, H5, CH5, NW5, BLOCK_M, BLOCK_C, SW5)
    elif pid < RB6:
        if NSEG > 6:
            rows = (pid - RB5) * BLOCK_M + tl.arange(0, BLOCK_M)
            r_ok = rows < M6
            _gemv_body(X, X + XR6 * stride_x, C6, S6, T6, Y + YO6, 0, 0, rows, r_ok, M6, N, 0,
                       LM6, K6, G6, L6, H6, CH6, NW6, BLOCK_M, BLOCK_C, SW6)
    elif pid < RB7:
        if NSEG > 7:
            rows = (pid - RB6) * BLOCK_M + tl.arange(0, BLOCK_M)
            r_ok = rows < M7
            _gemv_body(X, X + XR7 * stride_x, C7, S7, T7, Y + YO7, 0, 0, rows, r_ok, M7, N, 0,
                       LM7, K7, G7, L7, H7, CH7, NW7, BLOCK_M, BLOCK_C, SW7)


class DenseStageGEMV:
    """One-token GEMV of up to 8 packed matrices with the same input width in one launch. `mats`: list of
    (fmt, m, codes, scales, table); `x_rows[i]`: the row of the stacked input that matrix i reads."""

    def __init__(self, mats, n: int, x_rows, BLOCK_M: int | None = None, BLOCK_C: int | None = None,
                 num_warps: int | None = None, num_stages: int | None = None):
        assert 1 <= len(mats) <= 8
        bm, bc, nwarp, nst = tuning_for(mats[0][2].device).gemv_cfg("dense")
        self.n, self.BLOCK_M, self.BLOCK_C = n, BLOCK_M or bm, BLOCK_C or bc
        self.num_warps, self.num_stages = num_warps or nwarp, num_stages or nst
        BLOCK_M = self.BLOCK_M
        self.ms = [m for _, m, _, _, _ in mats]
        self.offsets = [sum(self.ms[:i]) for i in range(len(mats))]
        self.total = sum(self.ms)
        self.args, self.consts = [], {}
        rb = 0
        for i in range(8):
            fmt, m, codes, scales, table = mats[min(i, len(mats) - 1)]
            lm, k, g, logk, hm, CH, nw, sw, table = gemv_spec(fmt, table)
            assert (n * k) % 32 == 0 and n % CH == 0, (fmt, n)
            cw = codes.view(torch.int32) if codes.dtype == torch.uint8 else codes
            if i < len(mats):
                rb += triton.cdiv(m, BLOCK_M)
            self.args += [cw, scales, table, m, rb, x_rows[min(i, len(mats) - 1)], self.offsets[min(i, len(mats) - 1)]]
            self.consts.update({f"LM{i}": lm, f"K{i}": k, f"G{i}": g, f"L{i}": logk, f"H{i}": hm,
                                f"CH{i}": CH, f"NW{i}": nw, f"SW{i}": sw})
        self.blocks = rb
        self.nseg = len(mats)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """x `[R, n]` float32 (stacked inputs); returns the concatenated outputs `[total]`."""
        y = out if out is not None else torch.empty(self.total, dtype=torch.float32, device=x.device)
        with torch.cuda.device(x.device):
            _dense_stage_kernel[(self.blocks,)](x, y, self.n, x.stride(0), *self.args,
                                                NSEG=self.nseg, BLOCK_M=self.BLOCK_M, BLOCK_C=self.BLOCK_C, num_warps=self.num_warps, num_stages=self.num_stages, **self.consts)
        return y

    def split(self, y: torch.Tensor):
        return [y[o : o + m] for o, m in zip(self.offsets, self.ms)]
