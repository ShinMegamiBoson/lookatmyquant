"""Single-token decode for Gemma 4 with each matrix stage as one kernel launch.

mlx_lm runs a decoder layer as about 30 kernels; at one token, launch latency, not memory traffic, sets the speed. Here the
matrices that read the same input run in one launch (q, k and v; gate and up), and the small steps between them run in
that launch's setup phase, where each threadgroup builds the rotated input in threadgroup memory:

    q, k, v      ← input norm (or, after a layer, that layer's last residual add, layer scale and this norm)
    o            ← attention output
    gate, up     ← residual add of the attention branch, feed-forward norm
    down         ← GELU(gate) · up
    layer gate   ← residual add of the feed-forward branch
    layer proj   ← GELU(layer gate) · per-layer input

Values are rounded to bf16 where mlx_lm rounds them, so this path and the layer-by-layer one agree to rounding.
Prompts (more than one token) use mlx_lm's layers unchanged.

    import fastpath; fastpath.install(model)
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import lamq
import model as lm

_HEADER = """
// XP: the stride of a run of 32 inputs in threadgroup memory (33 pads each run, so lanes on different runs hit different banks)
inline float lamq_bf(float v) { return (float)(bfloat16_t)v; }

inline float lamq_tg_sum(float v, threadgroup float* red, uint simd, uint lane, uint nsimd) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[simd] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint i = 0; i < nsimd; i++) t += red[i];
    return t;
}

// rows of a Lloyd–Max matrix: y = S ⊙ (L[Q] xs) · norm; the threadgroup takes row blocks tg, tg + ntg, ...
template <uint BITS, uint LPR, uint ROWS, uint N, typename BUF, uint XP>
inline void lamq_lm_rows(const device uint* codes, uint wpr, uint m, const device half* scales, const threadgroup BUF* xs,
                         const threadgroup float* lv, device bfloat16_t* y, uint tg, uint ntg, uint simd, uint lane, float norm) {
    const uint sub = lane / LPR, sl = lane % LPR;
    constexpr uint MASK = (1u << BITS) - 1u;
    constexpr uint RPT = ROWS * (32 / LPR);
    for (uint rb = tg; rb * RPT < m; rb += ntg) {
        const uint row = (rb * ROWS + simd) * (32 / LPR) + sub;
        const bool live = row < m;
        const device uint* rc = codes + (ulong)(live ? row : 0) * wpr;
        const uint ngroups = live ? N / 32 : 0;
        float acc = 0.0f;
        uint nx[BITS];
        #pragma unroll
        for (uint i = 0; i < BITS; i++) nx[i] = sl < ngroups ? rc[sl * BITS + i] : 0u;
        for (uint g = sl; g < ngroups; g += LPR) {
            uint w[BITS + 1];
            #pragma unroll
            for (uint i = 0; i < BITS; i++) w[i] = nx[i];
            w[BITS] = 0;
            const uint gn = g + LPR < ngroups ? g + LPR : g;
            #pragma unroll
            for (uint i = 0; i < BITS; i++) nx[i] = rc[gn * BITS + i];
            const threadgroup BUF* xg = xs + g * XP;
            #pragma unroll
            for (uint k = 0; k < 32; k++) {
                const uint bit = k * BITS, wi = bit >> 5, off = bit & 31u;
                uint c = w[wi] >> off;
                if (off + BITS > 32) c |= w[wi + 1] << (32 - off);
                acc += lv[c & MASK] * (float)xg[k];
            }
        }
        for (uint d = LPR / 2; d > 0; d /= 2) acc += simd_shuffle_xor(acc, d);
        if (live && sl == 0) y[row] = (bfloat16_t)(acc * (float)scales[row] * norm);
    }
}

// rows of a Lloyd–Max matrix, MLX-style: a lane holds a run of 32 inputs in registers and reuses it for RPS rows; the levels
// come from SIMD shuffles (one level per lane for up to 5 bits, two for 6) or, for wider codes, from threadgroup memory.
// Every lane runs the same number of iterations (lanes past the end use zeros), as the shuffles need.
template <uint BITS, uint LPR, uint ROWS, uint N, typename BUF, uint XP, uint RPS, bool LUT_TG>
inline void lamq_lm_rows_reg(const device uint* codes, uint wpr, uint m, const device half* scales, const threadgroup BUF* xs,
                             const threadgroup float* lv, device bfloat16_t* y, uint tg, uint ntg, uint simd, uint lane, float norm) {
    constexpr uint MASK = (1u << BITS) - 1u, G = N / 32, IT = (G + LPR - 1) / LPR;
    const uint sub = lane / LPR, sl = lane % LPR;
    const float la = lv[lane & MASK & 31u], lb = lv[(32u + lane) & MASK];
    constexpr uint RPB = ROWS * (32 / LPR) * RPS;  // rows a threadgroup takes at a time
    for (uint rb = tg; rb * RPB < m; rb += ntg) {
        const uint r0 = ((rb * ROWS + simd) * (32 / LPR) + sub) * RPS;
        float acc[RPS];
        #pragma unroll
        for (uint i = 0; i < RPS; i++) acc[i] = 0.0f;
        for (uint it = 0; it < IT; it++) {
            const uint g = sl + LPR * it;
            const bool live = g < G;
            float xr[32];
            const threadgroup BUF* xg = xs + (live ? g : 0) * XP;
            #pragma unroll
            for (uint k = 0; k < 32; k++) xr[k] = live ? (float)xg[k] : 0.0f;
            #pragma unroll
            for (uint i = 0; i < RPS; i++) {
                const device uint* rc = codes + (ulong)min(r0 + i, m - 1) * wpr + (live ? g : 0) * BITS;
                uint w[BITS + 1];
                #pragma unroll
                for (uint j = 0; j < BITS; j++) w[j] = rc[j];
                w[BITS] = 0;
                float a = 0.0f;
                #pragma unroll
                for (uint k = 0; k < 32; k++) {
                    const uint bit = k * BITS, wi = bit >> 5, off = bit & 31u;
                    uint c = w[wi] >> off;
                    if (off + BITS > 32) c |= w[wi + 1] << (32 - off);
                    c &= MASK;
                    float v;
                    if (LUT_TG) v = lv[c];
                    else if (BITS <= 5) v = simd_shuffle(la, (ushort)c);
                    else if (BITS == 6) { const float va = simd_shuffle(la, (ushort)(c & 31u)), vb = simd_shuffle(lb, (ushort)(c & 31u)); v = (c & 32u) ? vb : va; }
                    else v = lv[c];
                    a += v * xr[k];
                }
                acc[i] += a;
            }
        }
        #pragma unroll
        for (uint i = 0; i < RPS; i++) {
            float s = acc[i];
            for (uint d = LPR / 2; d > 0; d /= 2) s += simd_shuffle_xor(s, (ushort)d);
            if (sl == 0 && r0 + i < m) y[r0 + i] = (bfloat16_t)(s * (float)scales[r0 + i] * norm);
        }
    }
}

// rows of a Lloyd–Max matrix, lane k on code k of every run of 32: input reads are consecutive across lanes (no bank
// conflicts) and each is reused for the simdgroup's RPS rows; the lanes together read each run's BITS words.
template <uint BITS, uint ROWS, uint N, typename BUF, uint RPS, uint XP>
inline void lamq_lm_rows_lane(const device uint* codes, uint wpr, uint m, const device half* scales, const threadgroup BUF* xs,
                              const threadgroup float* lv, device bfloat16_t* y, uint tg, uint ntg, uint simd, uint lane, float norm) {
    constexpr uint MASK = (1u << BITS) - 1u;
    const uint bit = lane * BITS, wi = bit >> 5, off = bit & 31u;
    const bool two = off + BITS > 32;
    for (uint rb = tg; rb * ROWS * RPS < m; rb += ntg) {
        const uint r0 = (rb * ROWS + simd) * RPS;
        if (r0 >= m) continue;
        float acc[RPS];
        const device uint* rc[RPS];
        #pragma unroll
        for (uint i = 0; i < RPS; i++) { acc[i] = 0.0f; rc[i] = codes + (ulong)min(r0 + i, m - 1) * wpr + wi; }
        for (uint g = 0; g < N / 32; g++) {
            const float xv = (float)xs[g * XP + lane];
            #pragma unroll
            for (uint i = 0; i < RPS; i++) {
                const device uint* p = rc[i] + g * BITS;
                uint c = p[0] >> off;
                if (two) c |= p[1] << (32 - off);
                acc[i] += lv[c & MASK] * xv;
            }
        }
        #pragma unroll
        for (uint i = 0; i < RPS; i++) {
            const float s = simd_sum(acc[i]);
            if (lane == 0 && r0 + i < m) y[r0 + i] = (bfloat16_t)(s * (float)scales[r0 + i] * norm);
        }
    }
}

// the same, writing row r to y[map[r]] (the output head: a bucket's rows are scattered tokens)
template <uint BITS, uint LPR, uint ROWS, uint N, typename BUF, uint XP>
inline void lamq_lm_rows_map(const device uint* codes, uint wpr, uint m, const device half* scales, const threadgroup BUF* xs,
                             const threadgroup float* lv, device bfloat16_t* y, const device int* map, uint tg, uint ntg, uint simd, uint lane) {
    const uint sub = lane / LPR, sl = lane % LPR;
    constexpr uint MASK = (1u << BITS) - 1u;
    constexpr uint RPT = ROWS * (32 / LPR);
    for (uint rb = tg; rb * RPT < m; rb += ntg) {
        const uint row = (rb * ROWS + simd) * (32 / LPR) + sub;
        const bool live = row < m;
        const device uint* rc = codes + (ulong)(live ? row : 0) * wpr;
        const uint ngroups = live ? N / 32 : 0;
        float acc = 0.0f;
        for (uint g = sl; g < ngroups; g += LPR) {
            uint w[BITS + 1];
            #pragma unroll
            for (uint i = 0; i < BITS; i++) w[i] = rc[g * BITS + i];
            w[BITS] = 0;
            const threadgroup BUF* xg = xs + g * XP;
            #pragma unroll
            for (uint k = 0; k < 32; k++) {
                const uint bit = k * BITS, wi = bit >> 5, off = bit & 31u;
                uint c = w[wi] >> off;
                if (off + BITS > 32) c |= w[wi + 1] << (32 - off);
                acc += lv[c & MASK] * (float)xg[k];
            }
        }
        for (uint d = LPR / 2; d > 0; d /= 2) acc += simd_shuffle_xor(acc, d);
        if (live && sl == 0) y[map[row]] = (bfloat16_t)(acc * (float)scales[row]);
    }
}

template <uint ROWS, uint N, typename BUF, uint XP>
inline void lamq_dense_rows_map(const device bfloat16_t* W, uint m, const threadgroup BUF* xs, device bfloat16_t* y, const device int* map,
                                uint tg, uint ntg, uint simd, uint lane) {
    for (uint rb = tg; rb * ROWS < m; rb += ntg) {
        const uint row = rb * ROWS + simd;
        if (row >= m) continue;
        const device bfloat16_t* wr = W + (ulong)row * N;
        float acc = 0.0f;
        for (uint i = lane; i < N; i += 32) acc += (float)wr[i] * (float)xs[i + (XP - 32) * (i >> 5)];
        acc = simd_sum(acc);
        if (lane == 0) y[map[row]] = (bfloat16_t)acc;
    }
}

// rows of a 16-bit matrix (not rotated): one simdgroup per row
template <uint ROWS, uint N, typename BUF, uint XP>
inline void lamq_dense_rows(const device bfloat16_t* W, uint m, const threadgroup BUF* xs, device bfloat16_t* y, uint tg, uint ntg,
                            uint simd, uint lane) {
    for (uint rb = tg; rb * ROWS < m; rb += ntg) {
        const uint row = rb * ROWS + simd;
        if (row >= m) continue;
        const device bfloat16_t* wr = W + (ulong)row * N;
        float acc = 0.0f;
        for (uint i = lane; i < N; i += 32) acc += (float)wr[i] * (float)xs[i + (XP - 32) * (i >> 5)];
        acc = simd_sum(acc);
        if (lane == 0) y[row] = (bfloat16_t)acc;
    }
}
"""


def _gelu_bf16(g):
    """mlx_lm's GELU (tanh form) with every intermediate rounded to bf16, as its compiled bf16 kernel does."""
    return ("lamq_bf(lamq_bf(0.5f * {g}) * lamq_bf(1.0f + lamq_bf(metal::precise::tanh(lamq_bf(0.796875f * "
            "lamq_bf({g} + lamq_bf(0.044677734375f * lamq_bf({g} * {g} * {g}))))))))").format(g=g)


def _source(mode, kinds, second_norm, has_scale, eps, rotate="full", scatter=False, packed=False, row_kernel="group"):
    """The stage kernel. Setup: each thread holds the elements (r, c) with r = lane + 32q (q < R) of the columns
    c = simdgroup + ROWS·j (j < MAXJ, c < C) of the input viewed as an (R·32) × C matrix. It builds the input there,
    applies the signs and the Sylvester transform along r (in registers and by SIMD shuffles), then the C × C factor
    through threadgroup memory. Then each threadgroup takes its matrix's row blocks."""
    L = []
    a = L.append
    nm = len(kinds)
    if packed:  # every matrix's codes, scales, signs and row map in shared buffers, at offsets offs[2i] (words) and offs[2i + 1] (rows)
        buf = {"c": lambda i: f"(codes + offs[{2 * i}])", "s": lambda i: f"(scales + offs[{2 * i + 1}])", "g": lambda i: f"(signs + {i}u * N)",
               "r": lambda i: f"(rows + offs[{2 * i + 1}])", "d": lambda i: f"((const device bfloat16_t*)(codes + offs[{2 * i}]))"}
    else:
        buf = {"c": lambda i: f"c{i}", "s": lambda i: f"s{i}", "g": lambda i: f"g{i}", "r": lambda i: f"r{i}", "d": lambda i: f"c{i}"}
    a("const uint tg = threadgroup_position_in_grid.x, tid = thread_position_in_threadgroup.x;")
    a("const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;")
    a("constexpr uint NT = ROWS * 32;")
    a("threadgroup BUF xs[N / 32 * XP];")
    a("threadgroup float lv[LVN];")
    a("threadgroup float red[2 * ROWS];")
    a("uint mi = 0;")
    for i in range(1, nm):
        a(f"if (tg >= dims[{4 * i + 2}]) mi = {i};")
    a("const uint TGL = tg - dims[4 * mi + 2], NTG = dims[4 * mi + 3];")
    bits = [f"BITS{i}" if k == "lm" else "16u" for i, k in enumerate(kinds)]
    expr = bits[-1]
    for i in range(nm - 2, -1, -1):
        expr = f"(mi == {i} ? {bits[i]} : {expr})"
    a(f"const uint bits = {expr};")
    a("if (bits <= 8) for (uint i = tid; i < (1u << bits); i += NT) lv[i] = levels[bits * 256 + i];")
    a("threadgroup float pm[C * C];")
    a("for (uint i = tid; i < C * C; i += NT) pm[i] = Pm[i];")
    a("float v[MAXJ][R];")
    a("#define IX(j, q) ((lane + 32u * (q)) * C + simd + ROWS * (j))")
    a("#define OK(j) (simd + ROWS * (j) < C)")
    a("#define XI(i) ((i) + (XP - 32) * ((i) >> 5))")
    each = "_Pragma(\"unroll\") for (uint j = 0; j < MAXJ; j++) _Pragma(\"unroll\") for (uint q = 0; q < R; q++)"
    def rsum(var, slot):  # the threadgroup's sum of a per-thread partial `var`, into float `var`_t
        a(f"{var} = simd_sum({var});")
        a(f"if (lane == 0) red[{slot} * ROWS + simd] = {var};")
        a("threadgroup_barrier(mem_flags::mem_threadgroup);")
        a(f"{{ float t_ = 0.0f; for (uint i = 0; i < ROWS; i++) t_ += red[{slot} * ROWS + i]; {var} = t_; }}")
    rs = lambda var: f"metal::precise::rsqrt({var} / (float)N + {eps!r}f)"
    lmi = [i for i, k in enumerate(kinds) if k == "lm"] if rotate != "none" else []
    full = rotate == "full"
    if lmi and full:
        sgx = buf["g"](lmi[-1])
        for i in reversed(lmi[:-1]):
            sgx = f"(mi == {i} ? {buf['g'](i)} : {sgx})"
        a(f"const device float* sg = {sgx};")
        a("float sv[MAXJ][R];")
        a(f"{each} sv[j][q] = (bits <= 8 && OK(j)) ? sg[IX(j, q)] : 1.0f;")
    if mode == "plain":
        a(f"{each} v[j][q] = OK(j) ? (float)x0[IX(j, q)] : 0.0f;")
    elif mode == "norm":
        a("float ss = 0.0f;")
        a(f"{each} {{ v[j][q] = OK(j) ? (float)x0[IX(j, q)] : 0.0f; ss += v[j][q] * v[j][q]; }}")
        rsum("ss", 0)
        a(f"const float r1 = {rs('ss')};")
        a(f"{each} if (OK(j)) v[j][q] = lamq_bf((float)wa[IX(j, q)] * lamq_bf(v[j][q] * r1));")
    elif mode == "resid":
        a("float ss = 0.0f;")
        a(f"{each} {{ v[j][q] = OK(j) ? (float)x1[IX(j, q)] : 0.0f; ss += v[j][q] * v[j][q]; }}")
        rsum("ss", 0)
        a(f"const float r1 = {rs('ss')};")
        a("float ss2 = 0.0f;")
        a(f"{each} if (OK(j)) {{")
        a("    float h = lamq_bf((float)x0[IX(j, q)] + lamq_bf((float)wa[IX(j, q)] * lamq_bf(v[j][q] * r1)));")
        if has_scale:
            a("    h = lamq_bf(h * (float)scl[0]);")
        a("    if (tg == 0) hout[IX(j, q)] = (bfloat16_t)h;")
        a("    v[j][q] = h; ss2 += h * h;")
        a("}")
        if second_norm:
            rsum("ss2", 1)
            a(f"const float r2 = {rs('ss2')};")
            a(f"{each} if (OK(j)) v[j][q] = lamq_bf((float)wb[IX(j, q)] * lamq_bf(v[j][q] * r2));")
    elif mode == "geglu":
        a(f"{each} if (OK(j)) {{ const float g = (float)x0[IX(j, q)]; v[j][q] = lamq_bf({_gelu_bf16('g')} * (float)x1[IX(j, q)]); }} else v[j][q] = 0.0f;")
    if lmi:  # the rotation, for the threadgroups on a Lloyd–Max matrix (bits is the same across a threadgroup)
        a("if (bits <= 8) {")
        if full:
            a(f"    {each} v[j][q] *= sv[j][q];")
            a("    for (uint h = 1; h < R; h <<= 1)")
            a("        for (uint j = 0; j < MAXJ; j++) for (uint q = 0; q < R; q++) if (!(q & h)) {")
            a("            const float u = v[j][q], w = v[j][q | h]; v[j][q] = u + w; v[j][q | h] = u - w; }")
            a("    for (uint h = 1; h < 32; h <<= 1) {")
            a("        const bool hi = lane & h;")
            a(f"        {each} {{ const float o = simd_shuffle_xor(v[j][q], (ushort)h); v[j][q] = hi ? o - v[j][q] : v[j][q] + o; }}")
            a("    }")
        a(f"    const float na = {'rsqrt((float)(32 * R))' if full else '1.0f'};")
        a(f"    {each} if (OK(j)) xs[XI(IX(j, q))] = (BUF)(v[j][q] * na);")
        a("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        a("    if (C > 1) {")
        a("        const float nc = rsqrt((float)C);")
        a(f"        {each} if (OK(j)) {{")
        a("            const uint c = simd + ROWS * j, base = IX(j, q) - c;")
        a("            float acc = 0.0f;")
        a("            for (uint k = 0; k < C; k++) acc += (float)xs[XI(base + k)] * pm[c * C + k];")
        a("            v[j][q] = acc * nc; }")
        a("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        a(f"        {each} if (OK(j)) xs[XI(IX(j, q))] = (BUF)v[j][q];")
        a("    }")
        a("} else {")
        a(f"    {each} if (OK(j)) xs[XI(IX(j, q))] = (BUF)v[j][q];")
        a("}")
    else:
        a(f"{each} if (OK(j)) xs[XI(IX(j, q))] = (BUF)v[j][q];")
    a("threadgroup_barrier(mem_flags::mem_threadgroup);")
    for i, k in enumerate(kinds):
        kw = "if" if i == 0 else "else if"
        c, s, r, d = buf["c"](i), buf["s"](i), buf["r"](i), buf["d"](i)
        if scatter and k == "lm":
            a(f"{kw} (mi == {i}) lamq_lm_rows_map<BITS{i}, LPR, ROWS, N, BUF, XP>({c}, dims[{4 * i + 1}], dims[{4 * i}], {s}, xs, lv, y, {r}, TGL, NTG, simd, lane);")
        elif scatter:
            a(f"{kw} (mi == {i}) lamq_dense_rows_map<ROWS, N, BUF, XP>({d}, dims[{4 * i}], xs, y, {r}, TGL, NTG, simd, lane);")
        elif k == "lm" and row_kernel == "reg":
            a(f"{kw} (mi == {i}) lamq_lm_rows_reg<BITS{i}, LPR, ROWS, N, BUF, XP, RPS, LUT_TG>({c}, dims[{4 * i + 1}], dims[{4 * i}], {s}, xs, lv, y{i}, TGL, NTG, simd, lane, 1.0f);")
        elif k == "lm" and row_kernel == "lane":
            a(f"{kw} (mi == {i}) lamq_lm_rows_lane<BITS{i}, ROWS, N, BUF, RPS, XP>({c}, dims[{4 * i + 1}], dims[{4 * i}], {s}, xs, lv, y{i}, TGL, NTG, simd, lane, 1.0f);")
        elif k == "lm":
            a(f"{kw} (mi == {i}) lamq_lm_rows<BITS{i}, LPR, ROWS, N, BUF, XP>({c}, dims[{4 * i + 1}], dims[{4 * i}], {s}, xs, lv, y{i}, TGL, NTG, simd, lane, 1.0f);")
        else:
            a(f"{kw} (mi == {i}) lamq_dense_rows<ROWS, N, BUF, XP>({d}, dims[{4 * i}], xs, y{i}, TGL, NTG, simd, lane);")
    return "\n".join("    " + s for s in L)


def _rot_source(mode):
    """One simdgroup per column c of the input viewed as an (32R) × C matrix: the stage input (GELU computed once per
    element), times the signs, then the Sylvester transform along the rows, in registers and by SIMD shuffles. The
    product kernel applies the C × C factor, which mixes only within a row."""
    L = []
    a = L.append
    a("const uint lane = thread_index_in_simdgroup, c = threadgroup_position_in_grid.x;")
    a("float y[R];")
    a("#pragma unroll")
    a("for (uint q = 0; q < R; q++) {")
    a("    const uint i = (lane + 32u * q) * C + c;")
    if mode == "plain":
        a("    const float v = (float)x0[i];")
    else:
        a(f"    const float g = (float)x0[i]; const float v = lamq_bf({_gelu_bf16('g')} * (float)x1[i]);")
    a("    y[q] = v * s[i];")
    a("}")
    a("for (uint h = 1; h < R; h <<= 1)")
    a("    for (uint q = 0; q < R; q++) if (!(q & h)) { const float u = y[q], w = y[q | h]; y[q] = u + w; y[q | h] = u - w; }")
    a("for (uint h = 1; h < 32; h <<= 1) {")
    a("    const bool hi = lane & h;")
    a("    for (uint q = 0; q < R; q++) { const float o = simd_shuffle_xor(y[q], (ushort)h); y[q] = hi ? o - y[q] : y[q] + o; }")
    a("}")
    a("const float nrm = rsqrt((float)(32 * R));")
    a("for (uint q = 0; q < R; q++) xr[(lane + 32u * q) * C + c] = y[q] * nrm;")
    return "\n".join("    " + s for s in L)


_PF_SRC = """
    // the C × C factor of a rotation, one row of the (N / C) × C input per thread: out[r, c] = Σ_k z[r, k] Q[c, k] / √C
    const uint r = thread_position_in_grid.x;
    if (r >= N / C) return;
    float v[C];
    for (uint k = 0; k < C; k++) v[k] = z[r * C + k];
    const float nc = rsqrt((float)C);
    for (uint c = 0; c < C; c++) {
        float acc = 0.0f;
        for (uint k = 0; k < C; k++) acc += v[k] * Q[c * C + k];
        out[r * C + c] = acc * nc;
    }
"""
_pf = mx.fast.metal_kernel(name="lamq_pfactor", input_names=["z", "Q"], output_names=["out"], source=_PF_SRC)


class WideRot:
    """The first part of the rotation for an input too wide to rebuild in every threadgroup: one launch of C simdgroups
    (WIDE_ROT "kernel"; the product kernel finishes it), or MLX's compiled rotation of the stage input, whole ("mlx")."""

    def __init__(self, mode, rot, R, C, Q=None):
        self.mode, self.rot, self.R, self.C, self.n, self.Q = mode, rot, R, C, rot.n, Q
        self.fixed = [rot.s]
        key = ("rot", mode)
        if key not in _KERNELS:
            _KERNELS[key] = mx.fast.metal_kernel(name=f"lamq_widerot_{mode}", input_names=["x0"] + (["x1"] if mode == "geglu" else []) + ["s"],
                                                 output_names=["xr"], source=_rot_source(mode), header=_HEADER)
        self.kernel = _KERNELS[key]
        self.template = [("R", R), ("C", C)]

    def __call__(self, ins):
        if WIDE_ROT == "mlx":
            x = ins[0] if self.mode == "plain" else nn.gelu_approx(ins[0]) * ins[1]
            return self.rot._f(x.astype(mx.float32), self.rot.s, self.rot.P_fwd)
        z = self.kernel(inputs=ins + self.fixed, template=self.template, grid=(32 * self.C, 1, 1), threadgroup=(32, 1, 1),
                        output_shapes=[(self.n,)], output_dtypes=[mx.float32])[0]
        if WIDE_ROT == "kernel+pf":  # the C × C factor too, in a second launch over the rows
            rows = self.n // self.C
            z = _pf(inputs=[z, self.Q], template=[("N", self.n), ("C", self.C)], grid=(-(-rows // 64) * 64, 1, 1), threadgroup=(64, 1, 1),
                    output_shapes=[(self.n,)], output_dtypes=[mx.float32])[0]
        return z


PAD = False  # pad each run of 32 inputs to 33 in threadgroup memory (no bank conflicts between lanes on different runs)
LUT_TG = True  # "reg" row kernel: levels from threadgroup memory instead of SIMD shuffles
HALF_INPUT = False  # the rotated input in threadgroup memory as float16 (as the prompt path rounds it): more threadgroups per core
REG_MIN_MB = 8.0  # stages with at least this many MB of codes use the register kernel; smaller ones keep "group"
ROW_KERNEL, RPS = "group", 4  # "lane": lane k on code k of each run, RPS rows per simdgroup; "group": lanes on whole runs
PER_MATRIX_BUDGET = True  # a stage gets each matrix's threadgroups, as separate launches would run side by side
WIDE_ROT = "mlx"  # "kernel+pf": a column kernel, then the C × C factor over the rows; "kernel": the product kernel applies the C × C factor; "mlx": MLX's compiled rotation
WIDE_N = 2048  # wider inputs are rotated once by WideRot, not in every threadgroup of the product


def _layout(rot):
    """(R, C, MAXJ, Pm): the input as an (32R) × C matrix whose rows get the Sylvester transform in registers and whose
    columns get the C × C factor — the rotation's p × p factor, times a Sylvester factor split off so that every
    simdgroup has a column and no thread holds more than a few dozen values."""
    A, P = rot.A, rot.p
    Pf = np.array(rot.Pm_fwd).reshape(P, P)
    best = None
    a2 = 1
    while A // a2 >= 32:
        C, R = P * a2, A // a2 // 32
        maxj = -(-C // lamq.ROWS)
        cost = (C < lamq.ROWS or R * maxj > 48, C)  # every simdgroup with a column, at most 48 values a thread, then the shortest C × C product
        if best is None or cost < best[0]:
            best = (cost, R, C, maxj, a2)
        a2 *= 2
    _, R, C, maxj, a2 = best
    Q = np.kron(lamq._sylvester(a2) if a2 > 1 else np.ones((1, 1)), Pf)
    return R, C, maxj, mx.array(Q.astype(np.float32).reshape(-1))


_KERNELS = {}


class Stage:
    """The matrices `mats` (LMLinear or 16-bit nn.Linear, one input width) applied to one input built by `mode`:
    plain x0; norm rms(x0)·wa; resid h = (x0 + rms(x1)·wa)·scl, then rms(h)·wb if wb is given (h is also returned);
    geglu GELU(x0)·x1. Returns [h,] y0, y1, ... as 1-D bf16 arrays."""

    def __init__(self, mode, mats, wa=None, wb=None, scl=None, eps=1e-6, scatter=None, budget=None, packed=None):
        kinds = ["lm" if hasattr(m, "codes") else "dense" for m in mats]
        ns = {m.n if k == "lm" else m.weight.shape[1] for m, k in zip(mats, kinds)}
        assert len(ns) == 1, "a stage's matrices share their input"
        n = ns.pop()
        self.mode, self.second_norm, self.has_scale = mode, wb is not None, scl is not None
        lms = [m for m, k in zip(mats, kinds) if k == "lm"]
        rot = lms[0].r_in if lms else lamq.Rot(np.ones(n))
        self.wide = None
        R, C, maxj, Pm = _layout(rot)
        if n > WIDE_N and kinds == ["lm"] and mode in ("plain", "geglu"):
            self.wide = WideRot(mode, rot, R, C, Pm)
            mode = "plain"  # the product kernel takes WideRot's output as its input
        lpr = lamq.lanes_per_row(n)
        sizes = [m.codes.nbytes if k == "lm" else m.weight.nbytes for m, k in zip(mats, kinds)]
        ms = [m.m if k == "lm" else m.weight.shape[0] for m, k in zip(mats, kinds)]
        sizes0 = [m.codes.nbytes if k == "lm" else m.weight.nbytes for m, k in zip(mats, kinds)]
        self.row_kernel = ROW_KERNEL if ROW_KERNEL != "reg" or sum(sizes0) >= REG_MIN_MB * 1e6 else "group"
        rk = self.row_kernel
        rpb = lamq.ROWS * (RPS if rk == "lane" else 32 // lpr * RPS if rk == "reg" else 32 // lpr)  # rows a threadgroup takes at a time
        blocks = [-(-mm // (rpb if k == "lm" else lamq.ROWS)) for mm, k in zip(ms, kinds)]
        budget = budget or (lamq.MAX_TG_NARROW if n <= 2048 else lamq.MAX_TG) * (len(mats) if PER_MATRIX_BUDGET else 1)
        ntg = [max(1, min(b, round(budget * s / sum(sizes)))) for b, s in zip(blocks, sizes)]
        dims, tg0 = [], 0
        for m, k, mm, t in zip(mats, kinds, ms, ntg):
            dims += [mm, m.codes.shape[1] if k == "lm" else n, tg0, t]
            tg0 += t
        self.n_tg = tg0
        names = ["x0"] + (["x1"] if mode in ("resid", "geglu") else []) + (["wa"] if mode in ("norm", "resid") else []) \
            + (["wb"] if self.second_norm else []) + (["scl"] if self.has_scale else []) + ["levels", "Pm", "dims"]
        fixed = [w for w in (wa, wb, scl) if w is not None] + [_levels_table(), Pm, mx.array(dims, dtype=mx.uint32)]
        if packed is not None:  # (codes, scales, signs, rows, offsets): shared buffers, as LMTable keeps them
            names += ["codes", "scales", "signs", "rows", "offs"]
            fixed += list(packed)
        for i, (m, k) in enumerate(zip(mats, kinds)):
            if packed is not None:
                continue
            if k == "lm":
                names += [f"c{i}", f"s{i}", f"g{i}"]
                fixed += [m.codes, m.scales, m.r_in.s]
            else:
                names += [f"c{i}"]
                fixed += [m.weight.astype(mx.bfloat16)]
            if scatter is not None:
                names += [f"r{i}"]
                fixed += [scatter[0][i]]
        outs = (["hout"] if mode == "resid" else []) + (["y"] if scatter is not None else [f"y{i}" for i in range(len(mats))])
        rotate = "full" if self.wide is None else "pfactor" if WIDE_ROT == "kernel" else "none"
        key = (mode, tuple(kinds), self.second_norm, self.has_scale, eps, rotate, scatter is not None, packed is not None, self.row_kernel)
        if key not in _KERNELS:
            _KERNELS[key] = mx.fast.metal_kernel(name=f"lamq_stage{len(_KERNELS)}_{mode}_{''.join(k[0] for k in kinds)}",
                                                 input_names=names, output_names=outs,
                                                 source=_source(mode, kinds, self.second_norm, self.has_scale, eps, rotate, scatter is not None, packed is not None,
                                                                self.row_kernel),
                                                 header=_HEADER)
        self.kernel, self.fixed = _KERNELS[key], fixed
        lvn = 1 << max([m.bits for m, k in zip(mats, kinds) if k == "lm"] + [1])
        self.template = [("BUF", mx.float32 if n <= lamq.TG_FLOATS and not HALF_INPUT else mx.float16), ("LVN", lvn), ("ROWS", lamq.ROWS), ("LPR", lpr), ("R", R), ("C", C),
                         ("MAXJ", maxj), ("N", n), ("RPS", RPS), ("XP", 33 if PAD else 32), ("LUT_TG", LUT_TG)] + [(f"BITS{i}", m.bits) for i, (m, k) in enumerate(zip(mats, kinds)) if k == "lm"]
        self.shapes = ([(n,)] if mode == "resid" else []) + ([(scatter[1],)] if scatter is not None else [(mm,) for mm in ms])
        self.r_out = [None] * len(self.shapes) if scatter is not None else \
            [None] * (mode == "resid") + [m.r_out if k == "lm" else None for m, k in zip(mats, kinds)]
        mx.eval(self.fixed)

    def __call__(self, x0, x1=None):
        ins = [x0.reshape(-1).astype(mx.bfloat16)] + ([x1.reshape(-1).astype(mx.bfloat16)] if x1 is not None else [])
        if self.wide is not None:
            ins = [self.wide(ins)]
        ins = ins + self.fixed
        ys = self.kernel(inputs=ins, template=self.template, grid=(self.n_tg * lamq.ROWS * 32, 1, 1), threadgroup=(lamq.ROWS * 32, 1, 1),
                         output_shapes=self.shapes, output_dtypes=[mx.bfloat16] * len(self.shapes))
        return [y if r is None else r.bwd(y.astype(mx.float32)).astype(mx.bfloat16) for y, r in zip(ys, self.r_out)]


_LV = []


def _levels_table():
    if not _LV:
        t = np.zeros((17, 256), np.float32)
        for b in (2, 3, 4, 5, 6, 8):
            t[b, : 2 ** b] = np.array(lm._levels(b))
        _LV.append(mx.array(t.reshape(-1)))
    return _LV[0]


class _Bucket:
    """A table bucket seen as one Lloyd–Max matrix (codes, row scales, input rotation) for Stage."""

    def __init__(self, b, d):
        self.codes, self.bits, self.m, self.n = b["codes"], b["bits"], b["codes"].shape[0], d
        self.scales, self.r_in, self.r_out = b["scales"][:, 0].astype(mx.float16), b["rot"], None


class _Dense:
    def __init__(self, W):
        self.weight = W.astype(mx.bfloat16)


def head(table):
    """The tied output head as one launch: every bucket's rows, each written to its token's position. The buckets'
    codes, scales and signs are read from the table's shared buffers (a kernel takes at most 31 buffers)."""
    mats = [_Bucket(b, table.d) if b["kind"] == "lm" else _Dense(b["W"]) for b in table.buckets]
    info = np.array(table.binfo).reshape(-1, 4)  # bits, word offset, words per row, scale offset (lm rows before this bucket)
    starts = np.cumsum([0] + [len(r) for r in table.rows])[:-1]
    offs, sig = [], []
    for k, b in enumerate(table.buckets):
        offs += [int(info[k, 1]), int(info[k, 3]) if b["kind"] == "lm" else 0]
    # the row maps: bucket k's rows at the same offset as its scales (dense buckets have no scales: their map goes after)
    nlm = int(sum(len(r) for r, b in zip(table.rows, table.buckets) if b["kind"] == "lm"))
    rows_all, extra = [None] * len(table.rows), nlm
    for k, (r, b) in enumerate(zip(table.rows, table.buckets)):
        if b["kind"] == "dense":
            offs[2 * k + 1] = extra
            extra += len(r)
    flat = np.zeros(extra, np.int32)
    for k, r in enumerate(table.rows):
        flat[offs[2 * k + 1] : offs[2 * k + 1] + len(r)] = np.array(r)
    scales = mx.concatenate([table.scales_all.reshape(-1)[:nlm], mx.zeros((extra - nlm,), mx.float16)]) if extra > nlm else table.scales_all.reshape(-1)
    packed = (table.codes_all, scales, table.signs_all, mx.array(flat), mx.array(np.array(offs, np.uint32)))
    return Stage("plain", mats, scatter=(None, table.V), budget=HEAD_TG, packed=packed)


HEAD_TG = 320


def _coded(m):
    return isinstance(m, (lm.LMLinear, nn.Linear)) and not isinstance(m, nn.QuantizedLinear)


class FastLayer:
    def __init__(self, layer, prev, eps):
        at, mlp = layer.self_attn, layer.mlp
        assert not at.use_k_eq_v
        self.layer, self.has_kv = layer, at.has_kv
        qkv = [at.q_proj] + ([at.k_proj, at.v_proj] if at.has_kv else [])
        w = lambda norm: norm.weight.astype(mx.bfloat16)
        self.qkv_first = Stage("norm", qkv, wa=w(layer.input_layernorm), eps=eps)
        self.qkv_after = None if prev is None else Stage("resid", qkv, wa=w(prev.post_per_layer_input_norm), wb=w(layer.input_layernorm),
                                                         scl=prev.layer_scalar.astype(mx.bfloat16), eps=eps)
        self.o = Stage("plain", [at.o_proj], eps=eps)
        self.gu = Stage("resid", [mlp.gate_proj, mlp.up_proj], wa=w(layer.post_attention_layernorm), wb=w(layer.pre_feedforward_layernorm), eps=eps)
        self.down = Stage("geglu", [mlp.down_proj], eps=eps)
        self.plig = Stage("resid", [layer.per_layer_input_gate], wa=w(layer.post_feedforward_layernorm), eps=eps)
        self.plp = Stage("geglu", [layer.per_layer_projection], eps=eps)

    def __call__(self, x, carry, mask, cache, pli, shared_kv, offset):
        from mlx_lm.models.base import scaled_dot_product_attention
        at = self.layer.self_attn
        if carry is None:
            ys = self.qkv_first(x)
        else:
            x, *ys = self.qkv_after(*carry)
        queries = at.q_norm(ys[0].reshape(1, 1, at.n_heads, at.head_dim))
        if shared_kv is not None:
            keys, values = shared_kv
        else:
            keys = ys[1].reshape(1, 1, at.n_kv_heads, at.head_dim)
            values = ys[2].reshape(1, 1, at.n_kv_heads, at.head_dim)
            offset = mx.array(cache.offset) if cache is not None else 0
            keys = at.rope(at.k_norm(keys).transpose(0, 2, 1, 3), offset=offset)
            values = at.v_norm(values).transpose(0, 2, 1, 3)
        queries = at.rope(queries.transpose(0, 2, 1, 3), offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        out = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=at.scale, mask=mask)
        (o,) = self.o(out)
        h, gate, up = self.gu(x, o)
        (d,) = self.down(gate, up)
        h2, g = self.plig(h, d)
        (p,) = self.plp(g, pli)
        return (h2, p), (keys, values), offset


def _flush(layer, carry):
    """The last step of a fast layer, when the next layer does not take it: (h2 + rms(p)·w)·layer scale."""
    h2, p = carry
    return ((h2 + layer.post_per_layer_input_norm(p)) * layer.layer_scalar).reshape(1, 1, -1)


def _fast_forward(self, inputs, cache):
    F = self._lamq_fast
    emb = self.embed_tokens(inputs)
    h = emb * self.embed_scale
    pli = self._project_per_layer_inputs(h, self._get_per_layer_inputs(inputs, emb))
    cache = cache + [None] * (len(self.layers) - len(cache))
    masks = self._make_masks(h, cache)
    inter = [(None, None)] * len(self.layers)
    carry = None
    for idx, (layer, c, mask, prev) in enumerate(zip(self.layers, cache, masks, self.previous_kvs)):
        kvs, offset = inter[prev]
        fl = F[idx]
        if fl is None or (carry is not None and fl.qkv_after is None):
            if carry is not None:
                h, carry = _flush(self.layers[idx - 1], carry), None
        if fl is None:
            h, kvs, offset = layer(h, mask, c, per_layer_input=pli[:, :, idx, :], shared_kv=kvs, offset=offset)
        else:
            carry, kvs, offset = fl(h, carry, mask, c, pli[0, 0, idx], kvs, offset)
        inter[idx] = (kvs, offset)
    if carry is not None:
        h = _flush(self.layers[-1], carry)
    return self.norm(h)


def install(M):
    """Give the model's text decoder the single-token path. Layers with a matrix this path cannot run keep mlx_lm's."""
    tm = M.language_model.model if hasattr(M, "language_model") else M.model
    eps = tm.config.rms_norm_eps
    F, prev = [], None
    for layer in tm.layers:
        mats = [layer.self_attn.q_proj, layer.self_attn.o_proj, layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj,
                layer.per_layer_input_gate, layer.per_layer_projection] + ([layer.self_attn.k_proj, layer.self_attn.v_proj] if layer.self_attn.has_kv else [])
        ok = all(_coded(m) for m in mats) and any(isinstance(m, lm.LMLinear) for m in mats)
        fl = FastLayer(layer, prev, eps) if ok else None
        F.append(fl)
        prev = layer if ok else None
    tm._lamq_fast = F
    if isinstance(tm.embed_tokens, lm.LMTable) and tm.embed_tokens.slices == 1:
        tm.embed_tokens._head = head(tm.embed_tokens)
    cls = type(tm)
    if not getattr(cls, "_lamq_patched", False):
        orig = cls.__call__

        def __call__(self, inputs=None, cache=None, input_embeddings=None, per_layer_inputs=None):
            if getattr(self, "_lamq_fast", None) and getattr(self, "_lamq_on", True) and inputs is not None and inputs.shape[-1] == 1 \
                    and input_embeddings is None and per_layer_inputs is None and cache is not None:
                return _fast_forward(self, inputs, cache)
            return orig(self, inputs, cache=cache, input_embeddings=input_embeddings, per_layer_inputs=per_layer_inputs)

        cls.__call__ = __call__
        cls._lamq_patched = True
    return sum(f is not None for f in F)
