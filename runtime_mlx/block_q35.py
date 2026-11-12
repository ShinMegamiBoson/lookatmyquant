"""A block of 8 or 16 tokens through a packed Qwen3.5 file in one pass: what a drafter's proposals are checked with.

fast_q35 decodes one token at a time, and a block there goes through the prompt path, which writes every matrix out
dense first (68 to 87 ms for any block against 9 ms a token). Here the coded weights are multiplied by all the block's
vectors as they are decoded:

  Product   a simdgroup takes 8 RB rows and walks their columns 32 at a time, as four 8 x 8 tiles. Each thread decodes
            eight neighbouring weights of one row from one window of its bit stream (two words, shifts fixed at compile
            time, one gather for a trellis pair) straight into its two elements of each tile's simdgroup matrix, and the
            GPU's matrix multiply-accumulate adds the tile times the inputs' tile (8 columns by 8 tokens) to the
            outputs' tile (8 rows by 8 tokens). The decode is paid once for the block. A matrix of few rows and many
            columns has its columns dealt to several simdgroups, summed in threadgroup memory.
  Glue      the steps between products for the whole block in the same few kernels fast_q35 uses for one token.
  State     the linear-attention layers carry a recurrent state. A check runs the whole block from the state before
            it. After `commit(kept)` the kept tokens (the anchor and the accepted proposals) are replayed from that
            same state inside the next check's own launches, or by `flush`; the attention layers' key-value caches
            drop the rest at once.

    block = Block(model, tap_layers)             # model from fast_q35.load
    logits, taps = block.check(tokens, cache)    # tokens: 8 or 16 ids; taps: the hidden states a drafter reads
    block.commit(cache, kept)                    # ... more blocks ...
    block.flush(cache)                           # before decoding one token at a time again

Launch shapes are chosen on the whole check's time, not a product's own: launches that do
not depend on each other overlap on the GPU, so a product timed alone or beside others reads up to twice as fast as it
runs inside the model, where each waits for the one before.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

try:
    from . import fast_q35 as F
except ImportError:  # run as a script beside fast_q35.py
    import fast_q35 as F

RB = {1: 2, 2: 2}    # 8-row tiles a simdgroup carries per pass over the columns, for 8 and for 16 tokens
SG = 32              # simdgroups per threadgroup: the largest threadgroup. Measured for the whole model's products at 8 tokens,
                     # against 10.3 ms for one token: 32 simdgroups 11.2 ms, 16 18.0 ms, 8 21.7 ms; RB 1, 2, 3, 4: 13.9, 11.2, 12.7, 16.5 ms
THREADGROUPS = 80
FOLD = 8             # passes (of 32 columns) summed before they join the row's total
HEAD_THREADGROUPS = 80
D = F.D

_HEADER = "#include <metal_simdgroup_matrix>\n"
_ELEMS = "reinterpret_cast<thread float2&>"


def _block_rows_fn(k, scalar, g, kind, *, top1=False):
    """Metal source of the function that handles one threadgroup's rows of a group with k-bit symbols, for a block.

    A pass takes 32 columns as four 8 x 8 tiles. A thread holds elements (fm, fn) and (fm, fn + 1) of every tile, and
    the tiles' column slots are dealt so that its eight weights of a row are eight neighbouring columns: slot fn + e of
    tile j is column 8 (fn / 2) + 2 j + e. Their bits are one window of the row's stream, three words at most, and 32
    columns are k whole words, so the window's shifts are the same on every pass. With n a multiple of 32 the window
    starts 0, 8, 16 or 24 bits into a word and is 40 bits at most (64 for 8-bit codes, which start at 0): two words."""
    two = F.slots(g) == 2
    vec = f"{kind}2" if two else kind
    name = f"lamq_blk_{'sc' if scalar else 'tr'}{k}_g{g}"
    L = []
    a = L.append
    a(f"inline void {name}(const device float* x, const device uint* codes, uint wpr, uint rows, const device float* scales,")
    output = "thread float* best, thread uint* ids" if top1 else "device float* y"
    size = "" if top1 else "uint size, "
    a(f"                   const device int* map, const device {kind}* table, {output}, threadgroup float* red, uint n, {size}uint tgl, uint ntg, uint simd, uint lane) {{")
    a(f"    const device {vec}* tv = (const device {vec}*)table;")
    a("    const uint qid = lane / 4u, fm = (qid & 4u) + ((lane / 2u) % 4u), fn = (qid & 2u) * 2u + (lane % 2u) * 2u;")
    a("    const uint q = fn / 2u;                           // this thread decodes columns 8 q .. 8 q + 7 of every 32")
    a("    const uint xcol = 8u * (fm / 2u) + (fm % 2u);     // the column its input elements belong to, before + 2 j")
    if scalar:   # forward stream: column c's code is the k bits at bit c k
        a(f"    const uint first = 8u * q * {k}u;")
        rel = [2 * j * k for j in range(4)]
    else:        # reversed stream: column c's state is the 12 bits at bit (n - 1 - c) k; a pair's index starts at its later column's
        a(f"    const uint first = (n - 8u - 8u * q) * {k}u;")
        rel = [(6 - 2 * j) * k for j in range(4)]
    a("    const uint w_first = first >> 5, off = first & 31u;   // off is 0, 8, 16 or 24")
    a("    // a threadgroup's simdgroups: SG / CS sets of 8 RB rows, each set's columns in CS parts (a matrix of few rows and")
    a("    // many columns would otherwise be a handful of long serial passes)")
    a("    const uint cp = simd % CS, rs = simd / CS, steps = n / 32u / CS;")
    a("    for (uint rb = tgl; rb * ((SG / CS) * 8u * RB) < rows; rb += ntg) {")
    a("    const uint rbase = (rb * (SG / CS) + rs) * (8u * RB), r0 = rbase + fm;")
    a("    const bool live = rbase < rows;       // a simdgroup past the group's last row has nothing to do")
    a("    // sums in three levels, so that no chain of additions is long: a pass's four tiles in the matrix unit, the")
    a("    // passes in `part`, folded into `total` every FOLD passes (one long chain measured 2e-6 from float64, 16 times")
    a("    // the one-token kernel; near-ties between two tokens then fall the other way more often)")
    a("    float2 part[RB * TB], total[RB * TB];")
    a("    for (uint i = 0; i < RB * TB; i++) { part[i] = float2(0.0f); total[i] = float2(0.0f); }")
    a("    if (live) {")
    a("    const device uint* p[RB];")
    a("    for (uint i = 0; i < RB; i++) p[i] = codes + (ulong)min(r0 + 8u * i, rows - 1u) * wpr;")
    a("    for (uint s = cp * steps; s < (cp + 1u) * steps; s++) {")
    a("        simdgroup_float8x8 Xm[4 * TB];")
    a("        const device float* xp = x + (ulong)(32u * s + xcol) * (8u * TB) + fn;")
    a("        #pragma unroll")
    a("        for (uint j = 0; j < 4u; j++) {")
    a("            #pragma unroll")
    a(f"            for (uint t = 0; t < TB; t++) {_ELEMS}(Xm[j * TB + t].thread_elements()) = float2(*(const device packed_float2*)(xp + j * (16u * TB) + 8u * t));")
    a("        }")
    a(f"        const uint wi = w_first {'+' if scalar else '-'} s * {k}u;")
    width = 2 * k if scalar else (F.L_STATE + k)

    def weights(j):
        """The expression for tile j's two weights, from the row's window (lo: its bits 0..31, hi: from 32 on)."""
        r = rel[j]
        if r + width <= 32:
            win = f"(lo >> {r}u)" if r else "lo"
        elif r >= 32:
            win = f"(hi >> {r - 32}u)" if r > 32 else "hi"
        else:
            win = f"((lo >> {r}u) | (hi << {32 - r}u))"
        if scalar:
            m = (1 << k) - 1
            return f"float2((float)tv[{win} & {m}u], (float)tv[({win} >> {k}u) & {m}u])"
        if two:
            return f"float2(tv[{win} & {(1 << (F.L_STATE + k)) - 1}u])"
        return f"float2((float)tv[({win} >> {k}u) & 4095u], (float)tv[{win} & 4095u])"

    window = ["const uint w0 = pw[0], w1 = (wi + 1u < wpr) ? pw[1] : 0u;",
              "// the window from its first bit: lo holds bits 0..31 of it, hi the bits from 32 on (it is 40 bits at most)",
              "const uint lo = (w0 >> off) | ((w1 << 1) << (31u - off)), hi = w1 >> off;"]

    def multiply(j, w):
        a(f"            {_ELEMS}(Am.thread_elements()) = {w};")
        a("            #pragma unroll")
        if j == 0:
            a("            for (uint t = 0; t < TB; t++) simdgroup_multiply(Em[t], Am, Xm[t]);")
        else:
            a(f"            for (uint t = 0; t < TB; t++) simdgroup_multiply_accumulate(Em[t], Am, Xm[{j}u * TB + t], Em[t]);")

    # lookup and multiply pair by pair. Issuing a pass's loads first, then its lookups, then its multiplies measured the
    # same (18.22 against 18.26 ms a check), and so did sums in two levels in place of three.
    a("        #pragma unroll")
    a("        for (uint i = 0; i < RB; i++) {")
    a("            const device uint* pw = p[i] + wi;")
    a("            simdgroup_float8x8 Em[TB], Am;")
    for line in window:
        a("            " + line)
    for j in range(4):
        multiply(j, weights(j))
    a("            #pragma unroll")
    a(f"            for (uint t = 0; t < TB; t++) part[i * TB + t] += {_ELEMS}(Em[t].thread_elements());")
    a("        }")
    a("        if ((s & (FOLD - 1u)) == FOLD - 1u)")
    a("            for (uint i = 0; i < RB * TB; i++) { total[i] += part[i]; part[i] = float2(0.0f); }")
    a("    }")
    a("    }")
    a("#if CS > 1")
    a("    for (uint i = 0; i < RB * TB; i++) {")
    a("        const float2 d = total[i] + part[i];")
    a("        red[(simd * (RB * TB) + i) * 64u + fm * 8u + fn] = d.x;")
    a("        red[(simd * (RB * TB) + i) * 64u + fm * 8u + fn + 1u] = d.y;")
    a("    }")
    a("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    a("    if (live && cp == 0u) {")
    a("#else")
    a("    if (live) {")
    a("#endif")
    a("    for (uint i = 0; i < RB; i++) {")
    a("        const uint row = r0 + 8u * i;")
    a("        if (row >= rows) continue;")
    a("        const float sc = scales[row];")
    a("        const ulong m = (ulong)map[row];")
    a("        for (uint t = 0; t < TB; t++) {")
    a("            float2 d = total[i * TB + t] + part[i * TB + t];")
    a("#if CS > 1")
    a("            for (uint c = 1; c < CS; c++) {")
    a("                const uint at = ((simd + c) * (RB * TB) + i * TB + t) * 64u + fm * 8u + fn;")
    a("                d += float2(red[at], red[at + 1u]);")
    a("            }")
    a("#endif")
    if top1:
        a("            keep_best(best[2u * t], ids[2u * t], d.x * sc, (uint)m);")
        a("            keep_best(best[2u * t + 1u], ids[2u * t + 1u], d.y * sc, (uint)m);")
    else:
        a("            y[(ulong)(8u * t + fn) * size + m] = d.x * sc;")
        a("            y[(ulong)(8u * t + fn + 1u) * size + m] = d.y * sc;")
    a("        }")
    a("    }")
    a("    }")
    a("#if CS > 1")
    a("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    a("#endif")
    a("    }")
    a("}")
    return name, "\n".join(L)


_KERNELS = {}


def _block_kernel(n, kind, widths, sg, rb, tb, cs):
    key = (n, kind, widths, sg, rb, tb, cs, FOLD)
    if key not in _KERNELS:
        fns = {w: _block_rows_fn(w[0], w[1], w[2], kind) for w in widths}
        header = _HEADER + f"#define SG {sg}u\n#define RB {rb}u\n#define TB {tb}u\n#define CS {cs}\n#define FOLD {FOLD}u\n" + "\n".join(src for _, src in fns.values())
        body = ["const uint tg = threadgroup_position_in_grid.x, gi = (uint)group_of[2 * tg], tgl = (uint)group_of[2 * tg + 1];",
                f"const device uint* d = dims + gi * {D};",
                "const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;",
                "const uint size = meta[0];",
                f"threadgroup float red[{sg * rb * tb * 64 if cs > 1 else 1}];",
                f"const device float* xv = x + (ulong)d[7] * {8 * tb}u;",
                "switch (d[6]) {"]
        for (k, scalar, g), (name, _) in fns.items():
            body.append(f"    case {F.width_code(k, g, scalar)}u: {name}(xv, codes + d[3], d[2], d[1], scales + d[4], map + d[4], tables + d[5], y, red, {n}u, size, tgl, d[0], simd, lane); break;")
        body.append("}")
        _KERNELS[key] = mx.fast.metal_kernel(name=f"lamq_block_{n}_{kind}_{sg}_{rb}_{tb}_{cs}_{FOLD}_{'_'.join(f'{k}{int(s)}{g}' for k, s, g in widths)}",
                                             input_names=["x", "codes", "scales", "map", "tables", "dims", "group_of", "meta"],
                                             output_names=["y"], source="\n".join("    " + s for s in body), header=header)
    return _KERNELS[key]


class BlockProduct:
    """A fast_q35.Product's row groups times 8 or 16 vectors at once. It shares the product's codes, scales and tables.

        y = BlockProduct(product)(x, tb)     # x: [input][column][token] float32, 8 tb tokens; y: (8 tb, size) float32
    """

    def __init__(self, product, threadgroups=None, rb=None, sg=None, cs=None):
        assert product.n % 32 == 0, "the block kernel takes 32 columns a pass"
        self.p = product
        self.n, self.size, self.kind = product.n, product.size, product.kind
        self.rb, self.sg, self.cs = rb, sg, cs
        self.threadgroups = threadgroups or THREADGROUPS
        self.meta = mx.array(np.array([self.size], dtype=np.uint32))
        self.launch = {}
        self._argmax = None

    def shape(self, tb):
        """(rb, sg, cs) for 8 tb tokens: what was asked for, else by the rows and columns (see SHAPES)."""
        rb, sg, cs = shape_for(self.size, self.n, tb)
        rb, sg, cs = self.rb or rb, self.sg or sg, self.cs or cs
        assert sg % cs == 0 and (self.n // 32) % cs == 0 and sg * rb * tb * 64 * 4 <= 32768 or cs == 1, (rb, sg, cs, self.n)
        return rb, sg, cs

    def _launch(self, tb):
        """The kernel for 8 tb tokens and its threadgroups: each row group gets its share of them by its rows."""
        rb, sg, cs = self.shape(tb)
        codes, scales, rowmap, tables, dims, _ = self.p.args
        dims = np.array(dims).reshape(-1, D).copy()
        rows = dims[:, 1].astype(np.int64)
        per = 8 * rb * (sg // cs)
        group_of = []
        for i, r in enumerate(rows):
            ntg = max(1, min(-(-int(r) // per), round(self.threadgroups * int(r) / int(rows.sum()))))
            dims[i, 0] = ntg
            group_of += [(i, j) for j in range(ntg)]
        args = [codes, scales, rowmap, tables, mx.array(dims.reshape(-1).astype(np.uint32)), mx.array(np.array(group_of, dtype=np.uint16).reshape(-1)), self.meta]
        mx.eval(args)
        return _block_kernel(self.n, self.kind, self.p.widths, sg, rb, tb, cs), args, len(group_of), sg

    def __call__(self, x, tb=1):
        if tb not in self.launch:
            self.launch[tb] = self._launch(tb)
        kernel, args, n_tg, sg = self.launch[tb]
        return kernel(inputs=[x.reshape(-1), *args], grid=(n_tg * sg * 32, 1, 1), threadgroup=(sg * 32, 1, 1),
                      output_shapes=[(8 * tb, self.size)], output_dtypes=[mx.float32])[0]

    def argmax(self, x, tb=1):
        """The highest scoring row per token, breaking ties by row index."""
        if self._argmax is None:
            try:
                from .block_argmax_q35 import argmax
            except ImportError:  # run as a script beside block_q35.py
                from block_argmax_q35 import argmax
            self._argmax = argmax
        return self._argmax(self, x, tb)


# (rows, columns, tb) -> (rb, sg, cs), chosen on the whole check's time with memory wired (at
# Qwen3.5-4B's shapes): 17.3 ms a check with the defaults, 16.6 with these. The matrices of 2,560 rows and 4,096 or
# 9,216 columns gain most, from splitting their columns in 8 parts.
SHAPES = {(18432, 2560, 1): (3, 32, 2), (12288, 2560, 1): (3, 16, 1), (2560, 9216, 1): (2, 32, 8), (248320, 2560, 1): (2, 32, 2),
          (2560, 4096, 1): (2, 32, 8), (10240, 2560, 1): (1, 16, 2),
          # 16 tokens: column parts for the same two kinds; the rest at the defaults (not tuned: 8 is the block that pays)
          (2560, 9216, 2): (1, 32, 8), (2560, 4096, 2): (1, 32, 8)}


def shape_for(rows, n, tb):
    """(8-row tiles a simdgroup carries, simdgroups a threadgroup, column parts) for a product of this size."""
    return SHAPES.get((rows, n, tb), (RB[tb], SG, 1))


# ----------------------------------------------------------------------------- the steps between products, for a block

_WHT_T = F._WHT_SRC.replace("float v = x0[base + i];", "float v = x0[(ulong)thread_position_in_grid.y * STRIDE + OFF0 + i];")
assert _WHT_T != F._WHT_SRC
_wht_t = {m: mx.fast.metal_kernel(name=f"lamq_wht_block{m}", input_names=["x0", "s"], output_names=["z"],
                                  source=_WHT_T.replace("GATE", "v = v / (1.0f + metal::precise::exp(-v)) * x0[(ulong)thread_position_in_grid.y * STRIDE + OFF1 + i];" if m else ""))
          for m in (0, 1)}
_CF_T = """
    // the C x C factor along each row and the norm, written column by column: out[i][t], the block product's layout
    const uint i = thread_position_in_grid.x, t = thread_position_in_grid.y;
    if (i >= N) return;
    const uint a = i / C, c = i - a * C;
    const device float* zr = z + (ulong)t * N + a * C;
    const device float* qr = Q + c * C;
    float acc = 0.0f;
    for (uint k = 0; k < C; k++) acc += zr[k] * qr[k];
    out[(ulong)i * TT + t] = acc * rsqrt((float)N);
"""
_cfactor_t = mx.fast.metal_kernel(name="lamq_cfactor_block", input_names=["z", "Q"], output_names=["out"], source=_CF_T)


class BlockStage:
    """A fast_q35.Stage for a block: the rotation of T vectors, written in the product's layout, then the product.

        y = BlockStage(stage)(x, T)                          # x: (T, n) float32; y: (T, size)
        y = stage(x, T, stride, off0)                        # the vectors are x[t * stride + off0 :][:n]
        y = stage(x, T, stride, off0, off1)                  # the input is silu(x at off0) * (x at off1)
    """

    def __init__(self, stage):
        self.s, self.n, self.size, self.slices = stage, stage.n, stage.product.size, stage.slices
        self.product = BlockProduct(stage.product)
        self.r = stage.rot1

    def __call__(self, x, T, stride=None, off0=0, off1=None):
        r = self.r
        t = [("R", r.R), ("C", r.C), ("STRIDE", stride or self.n), ("OFF0", off0), ("OFF1", off1 or 0)]
        z = _wht_t[int(off1 is not None)](inputs=[x.reshape(-1), r.s], template=t, grid=(32 * r.C, T, 1), threadgroup=(32, 1, 1),
                                          output_shapes=[(T * self.n,)], output_dtypes=[mx.float32])[0]
        xr = _cfactor_t(inputs=[z, r.Q], template=[("N", self.n), ("C", r.C), ("TT", T)], grid=(-(-self.n // 64) * 64, T, 1),
                        threadgroup=(64, 1, 1), output_shapes=[(self.n * T,)], output_dtypes=[mx.float32])[0]
        return self.product(xr, T // 8)


_ADD_NORM_T = """
    // h = a + b and its RMS norm times w for each of the block's vectors: one threadgroup of 1,024 per vector.
    const uint tid = thread_position_in_threadgroup.x;
    const ulong base = (ulong)thread_position_in_grid.y * N;
    constexpr uint E = (N + 1023) / 1024;
    float v[E];
    float ss = 0.0f;
    for (uint e = 0; e < E; e++) {
        const uint i = tid + e * 1024u;
        v[e] = i < N ? a[base + i] + b[base + i] : 0.0f;
        ss += v[e] * v[e];
        if (i < N) h[base + i] = v[e];
    }
    threadgroup float red[32];
    ss = simd_sum(ss);
    if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = 0.0f;
    for (uint j = 0; j < 32; j++) ss += red[j];
    const float r = metal::precise::rsqrt(ss / (float)N + EPS);
    for (uint e = 0; e < E; e++) {
        const uint i = tid + e * 1024u;
        if (i < N) hn[base + i] = v[e] * r * w[i];
    }
"""
_add_norms = {}


def add_norm_t(a, b, w, eps, T):
    """(a + b, rms_norm(a + b) * w) for T vectors of n."""
    if eps not in _add_norms:
        _add_norms[eps] = mx.fast.metal_kernel(name=f"lamq_add_norm_block_{len(_add_norms)}", input_names=["a", "b", "w"], output_names=["h", "hn"],
                                               source=_ADD_NORM_T.replace("EPS", f"{eps!r}f"))
    n = a.size // T
    return _add_norms[eps](inputs=[a.reshape(-1), b.reshape(-1), w], template=[("N", n)], grid=(1024, T, 1), threadgroup=(1024, 1, 1),
                           output_shapes=[(T, n), (T, n)], output_dtypes=[mx.float32, mx.float32])


_GDN_PRE_T = """
    // The gated delta rule's inputs for each token of a block. One thread per (channel, token); a threadgroup is one
    // head of one token. The depthwise convolution reads the three projections before each token: from this block's
    // stage output (y, at OFF of every STRIDE), before that from the kept tokens of the block before (yp, its first
    // cnt tokens, which the conv state does not hold yet), before that from the conv state. cs_out: the conv state
    // with those kept tokens taken in, which is the state before this block.
    const uint c = thread_position_in_grid.x, t = thread_position_in_grid.y, head = c / DK;
    const int kept = (int)cnt[0];
    float v = 0.0f;
    for (uint j = 0; j < 4u; j++) {
        const int u = (int)t + (int)j - 3, p = kept + u;
        v += (u >= 0 ? y[(ulong)u * STRIDE + OFF + c] : (p >= 0 ? yp[(ulong)p * STRIDE + OFF + c] : cs[(uint)(3 + p) * CD + c])) * w[4 * c + j];
    }
    if (t == 0u)
        for (uint j = 0; j < 3u; j++) {
            const int p = kept + (int)j - 3;
            cs_out[j * CD + c] = p >= 0 ? yp[(ulong)p * STRIDE + OFF + c] : cs[(uint)(3 + p) * CD + c];
        }
    v = v / (1.0f + metal::precise::exp(-v));
    threadgroup float red[DK / 32];
    float ss = simd_sum(v * v);
    if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = 0.0f;
    for (uint i = 0; i < DK / 32; i++) ss += red[i];
    if (head < 2 * HK) v *= metal::precise::rsqrt(ss / (float)DK + 1e-6f) * (head < HK ? 1.0f / (float)DK : metal::precise::rsqrt((float)DK));
    out[(ulong)t * CD + c] = v;
"""
_GDN_STEP_T = """
    // The gated delta rule from state_in: first over the cnt[0] kept tokens of the block before (xp, gatesp; no output),
    // then over cnt[1] tokens of this block (x, gates; y: each one's output). A simdgroup per (value head, value row)
    // holds its row of the state throughout. state_mid: the state between the two, which is the state before this
    // block. gates: each token's g and beta per value head.
    const uint hv = thread_position_in_grid.z, hk = hv / (HV / HK);
    constexpr int n_per_t = DK / 32;
    const uint dk_idx = thread_position_in_threadgroup.x, dv_idx = thread_position_in_grid.y;
    const device float* i_state = state_in + (hv * DV + dv_idx) * DK;
    device float* o_state = state_mid + (hv * DV + dv_idx) * DK;
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) state[i] = i_state[n_per_t * dk_idx + i];
    for (uint part = 0; part < 2u; part++) {
        const device float* xs = part == 0u ? xp : x;
        const device float* gs = part == 0u ? gatesp : gates;
        if (part == 1u)
            for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = state[i];
        const uint n = (uint)cnt[part];
        for (uint t = 0; t < n; t++) {
            const device float* xt = xs + (ulong)t * CD;
            const device float* q_ = xt + hk * DK;
            const device float* k_ = xt + HK * DK + hk * DK;
            const float g = gs[t * 2 * HV + hv], beta = gs[t * 2 * HV + HV + hv];
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                const uint s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * g;
                kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);
            const float delta = (xt[2 * HK * DK + hv * DV + dv_idx] - kv_mem) * beta;
            float o = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                const uint s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * delta;
                o += state[i] * q_[s_idx];
            }
            o = simd_sum(o);
            if (part == 1u && thread_index_in_simdgroup == 0) y[(ulong)t * (HV * DV) + hv * DV + dv_idx] = o;
        }
    }
"""
_GDN_POST_T = """
    // The rule's output, normed per value head (RMS, with the weight) and gated by SiLU(z), for each token.
    const uint i = thread_position_in_grid.x, t = thread_position_in_grid.y;
    const float v = y[(ulong)t * (HV * DV) + i];
    threadgroup float red[DV / 32];
    float ss = simd_sum(v * v);
    if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = 0.0f;
    for (uint j = 0; j < DV / 32; j++) ss += red[j];
    const float zz = z[(ulong)t * STRIDE + OFF + i];
    out[(ulong)t * (HV * DV) + i] = v * metal::precise::rsqrt(ss / (float)DV + EPS) * w[i % DV] * (zz / (1.0f + metal::precise::exp(-zz)));
"""
_CONV_KEEP = """
    // The conv state after the block's first cnt tokens: the last three projections of [the state before ; those tokens].
    const uint c = thread_position_in_grid.x, j = thread_position_in_grid.y;
    const int u = (int)cnt[0] + (int)j - 3;
    cs_out[j * CD + c] = u < 0 ? cs[(uint)(3 + u) * CD + c] : y[(ulong)u * STRIDE + OFF + c];
"""
_PROJ_T = """
    // The rule's two gates for each value head and token: g = exp(-exp(A_log) softplus(a + dt_bias)) and
    // beta = sigmoid(b), with a, b = w[r] . x[t] (the two small dense projections). A simdgroup per (row, token).
    // The gates depend on the head and the token only; computed here once, not in each of the state kernel's 131,072
    // threads. The products are written out by hand because MLX's own float32 matrix-matrix product on the GPU
    // measured 5e-4 from float64 on these (its matrix-vector product, which the one-token path uses, 1.5e-7), and
    // that alone put the block's logits 3e-4 from the one-token path's.
    const uint lane = thread_index_in_simdgroup, r = thread_position_in_grid.y, t = thread_position_in_grid.z;
    const device float* wr = w + (ulong)r * N;
    const device float* xt = x + (ulong)t * N;
    float acc = 0.0f;
    for (uint c = lane; c < N; c += 32u) acc += wr[c] * xt[c];
    acc = simd_sum(acc);
    if (lane != 0u) return;
    if (r < HV) {
        const float sp = acc + dt_bias[r];
        out[t * 2u * HV + r] = metal::precise::exp(-metal::precise::exp(A_log[r]) * (sp > 20.0f ? sp : metal::precise::log(1.0f + metal::precise::exp(sp))));
    } else {
        out[t * 2u * HV + r] = 1.0f / (1.0f + metal::precise::exp(-acc));
    }
"""
_proj_t = mx.fast.metal_kernel(name="lamq_gates_block", input_names=["w", "x", "A_log", "dt_bias"], output_names=["out"], source=_PROJ_T)
_gdn_pre_t = mx.fast.metal_kernel(name="lamq_gdn_pre_block", input_names=["y", "yp", "cs", "w", "cnt"], output_names=["out", "cs_out"], source=_GDN_PRE_T)
_gdn_step_t = mx.fast.metal_kernel(name="lamq_gdn_step_block", input_names=["x", "gates", "xp", "gatesp", "state_in", "cnt"], output_names=["y", "state_mid"], source=_GDN_STEP_T)
_conv_keep = mx.fast.metal_kernel(name="lamq_conv_keep", input_names=["cs", "y", "cnt"], output_names=["cs_out"], source=_CONV_KEEP)
_gdn_posts = {}


def _gdn_post_t(eps):
    if eps not in _gdn_posts:
        _gdn_posts[eps] = mx.fast.metal_kernel(name=f"lamq_gdn_post_block_{len(_gdn_posts)}", input_names=["y", "z", "w"], output_names=["out"],
                                               source=_GDN_POST_T.replace("EPS", f"{eps!r}f"))
    return _gdn_posts[eps]


# ----------------------------------------------------------------------------- the model, a block at a time

class Block:
    """A block of 8 or 16 tokens through the model fast_q35.load returns, with the caches mlx_lm's own decoding uses.

        logits, taps = block.check(tokens, cache)   # logits (T, vocabulary); taps: {layer: (T, hidden)} of tap_layers
        block.commit(cache, kept)                   # keep the block's first `kept` tokens in the caches, drop the rest

    Between a check and its commit the caches are not to be used: the attention layers' already hold the whole block
    and the linear-attention layers' still hold the state before it."""

    def __init__(self, model, tap_layers=()):
        tm = model.language_model.model
        self.tm, self.table = tm, tm.embed_tokens
        assert isinstance(self.table, F.FastTable), "the tied embedding must be a coded table"
        self.eps = float(tm.norm.eps)
        self.taps = tuple(tap_layers)
        self.cnt = {}
        stages = {}

        def stage(s):
            if id(s) not in stages:
                stages[id(s)] = BlockStage(s)
                s.block = stages[id(s)]            # mlx_lm's attention reaches its projections through the stage
            return stages[id(s)]

        self.recs = []
        for layer in tm.layers:
            L = getattr(layer, "inner", layer)
            rec = {"L": L, "linear": L.is_linear}
            assert isinstance(L.mlp, F.FastMLP), "the feed-forward block must be fast_q35's"
            rec["m_in"], rec["m_down"] = (stage(s) for s in L.mlp._stages)
            if L.is_linear:
                g = L.linear_attn
                assert isinstance(g, F.FastGatedDelta) and g.ok, "the linear attention must be fast_q35's"
                m = g.inner
                rec.update(g=g, m=m, s_in=stage(g._stages[0]), s_out=stage(g._stages[1]), w_ab=g.w_ab.reshape(-1), A_log=m.A_log.astype(mx.float32),
                           dt_bias=m.dt_bias.astype(mx.float32), norm_w=m.norm.weight.astype(mx.float32),
                           t=[("CD", m.conv_dim), ("DK", m.head_k_dim), ("DV", m.head_v_dim), ("HK", m.num_k_heads), ("HV", m.num_v_heads)])
                mx.eval(rec["w_ab"], rec["A_log"], rec["dt_bias"], rec["norm_w"])
            else:
                a = L.self_attn
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    stage(getattr(a, name)._stage[0])
            self.recs.append(rec)
        self.head = BlockProduct(self.table.product, threadgroups=HEAD_THREADGROUPS)
        self.stages = list(stages.values())
        self._gdn_pre_kernel = _gdn_pre_t
        self._gdn_step_kernel = _gdn_step_t
        self._gdn_flush_kernel = _conv_keep
        self.pending = None
        self.carry = None      # per layer: (the last committed block, tokens kept), not yet in the linear-attention states

    def _count(self, *n):
        if n not in self.cnt:
            self.cnt[n] = mx.array(np.array(n, dtype=np.uint32))
        return self.cnt[n]

    def _linear(self, rec, xn, c, T, carry):
        """A linear-attention layer for the block. carry: (the block before, how many of its tokens were kept) when
        those tokens are not in the cache's state yet; they are taken in here, in the same launches, so that keeping
        part of a block costs no pass of its own. On return the cache holds the state before this block."""
        m, s_in = rec["m"], rec["s_in"]
        y = s_in(xn, T)                                             # (T, qkv | z)
        q_off, z_off = s_in.slices["in_proj_qkv"][0], s_in.slices["in_proj_z"][0]
        ab = _proj_t(inputs=[rec["w_ab"], xn, rec["A_log"], rec["dt_bias"]], template=[("N", s_in.n), ("HV", m.num_v_heads)], grid=(32, 2 * m.num_v_heads, T),
                     threadgroup=(32, 1, 1), output_shapes=[(T, 2 * m.num_v_heads)], output_dtypes=[mx.float32])[0]      # (T, g | beta)
        cs = (c[0] if c[0] is not None else mx.zeros((3 * m.conv_dim,), mx.float32)).reshape(-1).astype(mx.float32)
        state = (c[1] if c[1] is not None else mx.zeros((1, m.num_v_heads, m.head_v_dim, m.head_k_dim), mx.float32)).astype(mx.float32)
        before, kept = carry if carry is not None else (None, 0)
        where = rec["t"] + [("STRIDE", s_in.size)]
        pre, c[0] = self._gdn_pre_kernel(inputs=[y, before["y"] if before else y, cs, rec["g"].conv_w, self._count(kept)], template=where + [("OFF", q_off)],
                               grid=(m.conv_dim, T, 1), threadgroup=(m.head_k_dim, 1, 1),
                               output_shapes=[(T, m.conv_dim), (1, 3, m.conv_dim)], output_dtypes=[mx.float32, mx.float32])
        out, c[1] = self._steps(rec, pre, ab, before["pre"] if before else pre, before["ab"] if before else ab, state, kept, T, T)
        out = _gdn_post_t(rec["g"].eps)(inputs=[out, y, rec["norm_w"]], template=where + [("OFF", z_off)], grid=(m.value_dim, T, 1),
                                        threadgroup=(m.head_v_dim, 1, 1), output_shapes=[(T, m.value_dim)], output_dtypes=[mx.float32])[0]
        return rec["s_out"](out, T), {"y": y, "pre": pre, "ab": ab, "where": where + [("OFF", q_off)]}

    def _steps(self, rec, pre, ab, pre_before, ab_before, state, kept, n, T):
        """(each of this block's first n tokens' output, the state after the block before's `kept` tokens)."""
        m = rec["m"]
        return self._gdn_step_kernel(inputs=[pre, ab, pre_before, ab_before, state, self._count(kept, n)], template=rec["t"],
                           grid=(32, m.head_v_dim, m.num_v_heads), threadgroup=(32, 4, 1),
                           output_shapes=[(T, m.value_dim), state.shape], output_dtypes=[mx.float32, mx.float32])

    def _attention(self, a, xn, cache, T):
        """mlx_lm's Qwen3NextAttention for a block, its queries 8 at a time. MLX's attention is exact for 8 queries or
        fewer; for more it goes through its float32 matrix product, which put a 16-token block's layer outputs 3e-4
        from the one-token path's (8 queries: 2e-7)."""
        n = xn.shape[-1]
        x = xn.reshape(1, T, n)
        queries, gate = mx.split(a.q_proj(x).reshape(1, T, a.num_attention_heads, -1), 2, axis=-1)
        keys = a.k_norm(a.k_proj(x).reshape(1, T, a.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = a.v_proj(x).reshape(1, T, a.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        queries = a.q_norm(queries).transpose(0, 2, 1, 3)
        S = cache.offset
        queries, keys = a.rope(queries, offset=S), a.rope(keys, offset=S)
        keys, values = cache.update_and_fetch(keys, values)
        out = [mx.fast.scaled_dot_product_attention(queries[:, :, lo : lo + 8], keys[:, :, : S + lo + 8], values[:, :, : S + lo + 8],
                                                    scale=a.scale, mask="causal") for lo in range(0, T, 8)]
        out = (out[0] if len(out) == 1 else mx.concatenate(out, axis=2)).transpose(0, 2, 1, 3).reshape(1, T, -1)
        return a.o_proj(out * mx.sigmoid(gate.reshape(1, T, -1))).reshape(T, n).astype(mx.float32)

    def _mlp(self, rec, hn, T):
        s_in = rec["m_in"]
        y = s_in(hn, T)                                             # (T, gate | up)
        return rec["m_down"](y, T, stride=s_in.size, off0=s_in.slices["gate_proj"][0], off1=s_in.slices["up_proj"][0])

    def _head_input(self, h, T):
        tb = self.table
        xr = tb.rot.fwd(mx.broadcast_to(h[:, None, :], (T, tb.nb, tb.n)))        # every bucket's rotation of every state
        return xr.transpose(1, 2, 0)

    def logits(self, h, T):
        """The output head on T normed states (T, hidden): (T, vocabulary)."""
        return self.head(self._head_input(h, T), T // 8)

    def argmax(self, h, T):
        """The highest scoring token for each normed state."""
        return self.head.argmax(self._head_input(h, T), T // 8)

    def check(self, tokens, cache, *, top1=False):
        T = len(tokens)
        assert T in (8, 16) and self.pending is None, "a block is 8 or 16 tokens, and the last one must be committed"
        tm = self.tm
        x = self.table(mx.array(tokens)).astype(mx.float32)
        xn = mx.fast.rms_norm(x, self.recs[0]["L"].input_layernorm.weight, self.eps)
        carry, self.carry = self.carry or [None] * len(self.recs), None
        pend, taps = [], {}
        for i, rec in enumerate(self.recs):
            L = rec["L"]
            if rec["linear"]:
                r, p = self._linear(rec, xn, cache[i], T, carry[i])
            else:
                r, p = self._attention(L.self_attn, xn, cache[i], T), None
            pend.append(p)
            h, hn = add_norm_t(x, r, L.post_attention_layernorm.weight, self.eps, T)
            m = self._mlp(rec, hn, T)
            if i + 1 < len(self.recs):
                x, xn = add_norm_t(h, m, self.recs[i + 1]["L"].input_layernorm.weight, self.eps, T)
            else:
                x = h + m
            if i in self.taps:
                taps[i] = x
        self.pending = (T, pend)
        h = mx.fast.rms_norm(x, tm.norm.weight, self.eps)
        return (self.argmax(h, T) if top1 else self.logits(h, T)), taps

    def commit(self, cache, kept):
        """Keep the last checked block's first `kept` tokens. The attention layers' caches drop the rest now; the
        linear-attention layers take the kept tokens into their state in the next check's own launches (or in flush)."""
        T, pend = self.pending
        assert 1 <= kept <= T
        self.pending = None
        self.carry = []
        for rec, p, c in zip(self.recs, pend, cache):
            if rec["linear"]:
                c.advance(kept)
                self.carry.append((p, kept))
            else:
                c.trim(T - kept)
                self.carry.append(None)

    def flush(self, cache):
        """Take the kept tokens of the last block into the linear-attention layers' states now: the caches are then
        what one-token decoding expects (call it before decoding outside this class, and at the end)."""
        assert self.pending is None
        for rec, carry, c in zip(self.recs, self.carry or [], cache):
            if carry is None:
                continue
            p, kept = carry
            m = rec["m"]
            cs, state = c[0].reshape(-1), c[1]
            c[0] = self._gdn_flush_kernel(inputs=[cs, p["y"], self._count(kept)], template=p["where"], grid=(m.conv_dim, 3, 1), threadgroup=(32, 1, 1),
                              output_shapes=[(1, 3, m.conv_dim)], output_dtypes=[mx.float32])[0]
            c[1] = self._steps(rec, p["pre"], p["ab"], p["pre"], p["ab"], state, kept, 0, 8)[1]
        self.carry = None

    def discard(self):
        """Release unfinished block state when the caller will discard its caches."""
        self.pending = self.carry = None
