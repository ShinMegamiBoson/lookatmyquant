"""Fast single-token decode of a packed (opgraph-packed-1) Qwen3.5 file on Apple silicon.

Three things make it fast:

  Product   every row group of every matrix that reads one input, in one kernel launch. A lane takes CH consecutive
            columns at a time, holds their inputs in registers and reuses them for RPS rows, as MLX's own 4-bit kernel
            does. A row group keeps its own bit width and table; the launch's threadgroups are dealt out by rows.
  Tables    a trellis weight is table[state], one gather a weight. Consecutive states share 12 - k bits, so the
            12 + (g - 1) k stream bits at a column fix g states at once: a table indexed by them returns g weights
            for one gather. g is as large as keeps the table within 128 KB, past which gathers fall off the GPU's
            cache (k = 1: 3 weights, k = 2 and 3: 2 weights, k = 4: 1). The tables are built at load from the file's.
  Glue      the steps between products (norms, rotations, the activation, the gated delta rule's inputs) as a few
            kernels instead of mlx_lm's dozens of small operations.

    model, tok = load("path/to/packed")      # mlx_lm's generate and stream_generate work on it

A row is a little-endian bit stream of k-bit symbols; column j's state is the 12 bits at bit (N - 1 - j) k, zeros past
the row's end (compiler_artifact.pack_trellis). A prompt of more than one token multiplies by each matrix written out dense.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

if importlib.util.find_spec("compiler_artifact") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

L_STATE = 12
RPS = 4            # rows a lane carries per pass over its columns
SG = 16            # simdgroups per threadgroup
HEAD_THREADGROUPS = 80
THREADGROUPS = 80  # threadgroups a product launches, dealt to its row groups by their rows (the GPU has 40 cores)
WEIGHTS_PER_GATHER = {1: 2, 2: 2, 3: 2, 4: 1}   # tables of 32, 64, 128 and 8 KB as half floats (k = 1 with 3 weights, 128 KB, measured no faster)
COST = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}         # relative time a weight takes at each width: threadgroups are dealt by rows × cost
TABLE = "half"     # the table's entries: "half" (16 bits) or "float" (as the file holds them; then k = 3 gets 1 weight a gather)
FUSE_STEPS = 1     # the steps between products as this module's kernels (0: mlx_lm's own operations)
FUSE_ADD_NORM = 1  # a residual add and the norm after it in one launch
EXTRACT_BITS = 0   # 1: bit-field instructions in place of shift and mask (measured no faster)
D = 8              # values per group in a product's `dims`


def weights_per_gather(k, kind=None, gather=None):
    kind = kind or TABLE
    g = (gather or WEIGHTS_PER_GATHER)[k]
    if kind == "float":  # entries twice the size: stay within 128 KB
        g = {1: 2, 2: 2, 3: 1, 4: 1}[k]
    return g


def slots(g):
    return 1 if g == 1 else 2 if g == 2 else 4


def multi_table(values, k, g, dtype):
    """entry[idx][t] = values[state of the t-th of g consecutive columns], idx the 12 + (g - 1) k bits at the last one's place."""
    idx = np.arange(1 << (L_STATE + (g - 1) * k), dtype=np.int64)
    out = np.zeros((len(idx), slots(g)), dtype)
    for t in range(g):
        out[:, t] = values[(idx >> ((g - 1 - t) * k)) & 4095]
    return out.reshape(-1)


# ----------------------------------------------------------------------------- the product kernel

def width_code(k, g, scalar):
    """A group's kind as the kernels read it from `dims`: bits, weights per gather, scalar or trellis."""
    return k | (g << 4) | (int(scalar) << 7)


def _rows_fn(k, ch, kind, scalar=False, dense=False, g=1, space="device"):
    """Metal source of the function that handles one threadgroup's rows of a group with k-bit symbols: their products
    with the input, or (dense) the weights themselves written, scaled, into a dense matrix of OUT_T.

    Trellis groups (scalar False): reversed stream, states of 12 bits, g weights a gather. Scalar groups: forward
    stream of k-bit codes, table[code]."""
    vec = kind if slots(g) == 1 else f"{kind}{slots(g)}"
    odd = (not scalar) and bool(k & 1) and ch == 16      # a chunk starts 0 or 16 bits into a word
    sodd = scalar and (ch * k) % 32 != 0                 # the same for a forward stream
    width0 = k if scalar else L_STATE
    span = (ch - 1) * k + width0 + (16 if (odd or sodd) else 0)
    nw = (span + 31) // 32
    L = []
    a = L.append
    name = f"lamq_{'sc' if scalar else 'tr'}{k}_g{g}_ch{ch}" + ("_dense" if dense else "")
    # MLX binds short arrays in constant memory. Deduce each input's address space.
    a("template <typename Scales, typename Map, typename Table>")
    a(f"inline void {name}({'' if dense else 'const device float* x, '}const device uint* codes, uint wpr, uint rows, Scales scales,")
    a(f"                   Map map, Table table, device {'OUT_T' if dense else 'float'}* y, uint n, uint tgl, uint ntg, uint simd, uint lane) {{")
    a("    Table tv = table;" if slots(g) == 1 else f"    const {space} {vec}* tv = (const {space} {vec}*)table;")
    a(f"    const uint iters = n / {ch * 32};")
    a("    for (uint rb = tgl; rb * (SG * RPS) < rows; rb += ntg) {   // this threadgroup's row blocks: one table stays in its core's cache")
    a("    const uint r0 = (rb * SG + simd) * RPS;")
    if not dense:
        a("    float acc[RPS];")
        a("    for (uint r = 0; r < RPS; r++) acc[r] = 0.0f;")
    a("    for (uint it = 0; it < iters; it++) {")
    a("        const uint c = lane + 32 * it;")
    if not dense:
        a(f"        float xr[{ch}];")
        for i in range(ch):
            a(f"        xr[{i}] = x[c * {ch} + {i}];")
    if scalar:
        a(f"        const uint B = c * {ch * k}u, w0 = B >> 5;")
    else:
        a(f"        const uint B = (n - {ch} * (c + 1)) * {k}u, w0 = B >> 5;")
    if odd or sodd:
        a("        const uint o = B & 31u;")
    a("        #pragma unroll")
    a("        for (uint r = 0; r < RPS; r++) {")
    a("            const device uint* p = codes + (ulong)min(r0 + r, rows - 1) * wpr + w0;")
    a(f"            uint w[{nw + 1}];")
    for i in range(nw):
        a(f"            w[{i}] = (w0 + {i} < wpr) ? p[{i}] : 0u;")
    a(f"            w[{nw}] = 0u;")
    if dense:
        a("            const uint row = r0 + r;")
        a("            if (row >= rows) continue;")
        a("            const float sc = scales[row];")
        a(f"            device OUT_T* o_ = y + (ulong)map[row] * n + c * {ch};")
    else:
        a("            float a = 0.0f;")
    i = 0
    while i < ch:
        gg = min(g, ch - i)
        last = i + gg - 1
        bits = width0 + (gg - 1) * k
        bit = i * k if scalar else (ch - 1 - last) * k
        if odd or sodd:
            a(f"            {{ const uint bit = o + {bit}u, wi = bit >> 5, off = bit & 31u;")
            a(f"              const uint s = ((w[wi] >> off) | ((w[wi + 1] << 1) << (31 - off))) & {(1 << bits) - 1}u;")
        else:
            wi, off = bit >> 5, bit & 31
            if EXTRACT_BITS and off + bits <= 32:
                a(f"            {{ const uint s = extract_bits(w[{wi}], {off}u, {bits}u);")
            elif EXTRACT_BITS:
                lo = 32 - off
                a(f"            {{ const uint s = insert_bits(extract_bits(w[{wi}], {off}u, {lo}u), w[{wi + 1}], {lo}u, {bits - lo}u);")
            else:
                e = f"(w[{wi}] >> {off})" + (f" | (w[{wi + 1}] << {32 - off})" if off + bits > 32 else "")
                a(f"            {{ const uint s = ({e}) & {(1 << bits) - 1}u;")
        # a short group at the end of a chunk sits in the last slots (its index has zeros above)
        vals = ["(float)tv[s]"] if slots(g) == 1 else [f"(float)v[{g - gg + t}]" for t in range(gg)]
        if slots(g) > 1:
            a(f"              const {vec} v = tv[s];")
        if dense:
            a("              " + " ".join(f"o_[{i + t}] = (OUT_T)({v} * sc);" for t, v in enumerate(vals)) + " }")
        else:
            a("              a += " + " + ".join(f"{v} * xr[{i + t}]" for t, v in enumerate(vals)) + "; }")
        i += gg
    if not dense:
        a("            acc[r] += a;")
    a("        }")
    a("    }")
    if not dense:
        a("    #pragma unroll")
        a("    for (uint r = 0; r < RPS; r++) {")
        a("        const float s = simd_sum(acc[r]);")
        a("        if (lane == 0 && r0 + r < rows) y[map[r0 + r]] = s * scales[r0 + r];")
        a("    }")
    a("    }")
    a("}")
    return name, "\n".join(L)


def chunk(n):
    """Columns a lane takes at a time: all 32 lanes busy on every pass."""
    assert n % 512 == 0, f"{n} columns: the product kernel needs a multiple of 512"
    return 32 if n % 1024 == 0 else 16


_PRODUCTS = {}


def _product_kernel(n, kind, widths, sg, stage_tables):
    """One kernel for inputs of n columns: a threadgroup looks up its group and runs that group's width.
    widths: the (bits, scalar, weights per gather) triples that occur."""
    key = (n, kind, widths, sg, stage_tables)
    if key not in _PRODUCTS:
        ch = chunk(n)
        space = "threadgroup" if stage_tables else "device"
        fns = {w: _rows_fn(w[0], ch, kind, scalar=w[1], g=w[2], space=space) for w in widths}
        header = f"#define RPS {RPS}u\n#define SG {sg}u\n" + "\n".join(src for _, src in fns.values())
        body = [f"const uint tg = threadgroup_position_in_grid.x, gi = (uint)group_of[2 * tg], tgl = (uint)group_of[2 * tg + 1];",
                f"const device uint* d = dims + gi * {D};",
                "const uint ntg = d[0], rows = d[1], wpr = d[2];",
                "const device uint* c = codes + d[3];",
                "auto sc = scales + d[4];",
                "auto mp = map + d[4];",
                "auto tb = tables + d[5];",
                "const device float* xv = x + d[7];",
                "const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;"]
        if stage_tables:
            body += [f"threadgroup {kind} tt[{1 << L_STATE}];",
                     f"const uint tlen = (d[6] & 128u) ? (1u << (d[6] & 15u)) : {1 << L_STATE}u;",
                     "for (uint i = thread_position_in_threadgroup.x; i < tlen; i += threads_per_threadgroup.x) tt[i] = tb[i];",
                     "threadgroup_barrier(mem_flags::mem_threadgroup);"]
        table = "tt" if stage_tables else "tb"
        body.append("switch (d[6]) {")
        for (k, scalar, g), (name, _) in fns.items():
            body.append(f"    case {width_code(k, g, scalar)}u: {name}(xv, c, wpr, rows, sc, mp, {table}, y, {n}u, tgl, ntg, simd, lane); break;")
        body.append("}")
        _PRODUCTS[key] = mx.fast.metal_kernel(name=f"lamq_product_{n}_{kind}_{sg}_{int(stage_tables)}_{'_'.join(f'{k}{int(s)}{g}' for k, s, g in widths)}",
                                              input_names=["x", "codes", "scales", "map", "tables", "dims", "group_of"],
                                              output_names=["y"], source="\n".join("    " + s for s in body), header=header)
    return _PRODUCTS[key]


class Product:
    """Row groups that read inputs of one width, in one launch. Each group: codes (rows × words, uint32), scales
    (float32), the output position of each of its rows, its bit width, its table (4,096 values for a trellis group,
    2^k levels for a scalar one) and which of the launch's input vectors it reads.

        y = Product(n, groups, size)(x)      # x: (vectors, n) or (n,) float32, already rotated; y: (size,) float32
    """

    @staticmethod
    def order(group_of, interleave):
        """The launch's threadgroups as (group, index within the group) pairs: group after group, or (interleave = b)
        dealt out b at a time from each group in turn."""
        pairs = np.array(group_of, dtype=np.uint16).reshape(-1, 2)
        if interleave:
            by_group = {}
            for g, i in pairs:
                by_group.setdefault(int(g), []).append((g, i))
            out, lists = [], list(by_group.values())
            pos = [0] * len(lists)
            while any(p < len(l) for p, l in zip(pos, lists)):
                for j, l in enumerate(lists):
                    out += l[pos[j] : pos[j] + interleave]
                    pos[j] += interleave
            pairs = np.array(out, dtype=np.uint16)
        return pairs.reshape(-1)

    def __init__(self, n, groups, size, kind=None, interleave=0, gather=None, sg=None, threadgroups=None):
        kind = kind or TABLE
        family = mx.device_info().get("device_name", "").removeprefix("Apple ").partition(" ")[0]
        # M1 products exceed their pipeline limit at 512 threads. M5 runs faster with device tables.
        sg = sg or (8 if family == "M1" else SG)
        stage_tables = family in ("M1", "M4")
        threadgroups = threadgroups or THREADGROUPS
        self.n, self.size, self.kind, self.sg = n, size, kind, sg
        work = [g["codes"].shape[0] * (COST[4] if g.get("scalar") else COST[int(g["bits"])]) for g in groups]
        dtype = np.float16 if kind == "half" else np.float32
        codes, scales, rowmap, tables, dims, group_of = [], [], [], [], [], []
        w_off = r_off = t_off = tg0 = 0
        widths = set()
        for g in groups:
            k, scalar = int(g["bits"]), g.get("scalar", False)
            c = np.ascontiguousarray(g["codes"], dtype=np.uint32)
            rows, wpr = c.shape
            if wpr != n // 32 * k:
                raise ValueError("packed row length does not match its bit width")
            wg = 1 if scalar or stage_tables else weights_per_gather(k, kind, gather)
            if scalar:
                t = np.asarray(g["table"], dtype=dtype)
                if not 1 <= k <= 8 or t.ndim != 1 or not 1 <= t.size <= 1 << k:
                    raise ValueError("invalid scalar lookup table or bit width")
                # The compiler permits short codebooks. The GPU addresses the full k-bit range.
                if t.size < 1 << k:
                    t = np.pad(t, (0, (1 << k) - t.size))
            else:
                values = np.asarray(g["table"], dtype=np.float32)
                if values.shape != (1 << L_STATE,):
                    raise ValueError("invalid trellis lookup table length")
                t = multi_table(values, k, wg, dtype)
            t = np.concatenate([t, np.zeros((-len(t)) % 4, dtype)])   # each table on an 8-byte boundary at least
            # a threadgroup stays on one group (one table in its core's cache) and takes that group's row blocks in turn
            ntg = max(1, min(-(-rows // (RPS * sg)), round(threadgroups * work[len(dims) // D] / sum(work))))
            dims += [ntg, rows, wpr, w_off, r_off, t_off, width_code(k, wg, scalar), int(g.get("input", 0)) * n]
            group_of += [(len(dims) // D - 1, i) for i in range(ntg)]
            codes.append(c.reshape(-1)); scales.append(np.asarray(g["scales"], dtype=np.float32)); rowmap.append(np.asarray(g["rows"], dtype=np.int32)); tables.append(t)
            w_off, r_off, t_off, tg0 = w_off + c.size, r_off + rows, t_off + len(t), tg0 + ntg
            widths.add((k, scalar, wg))
        assert w_off < 2 ** 32
        self.n_tg = tg0
        self.table_bytes = sum(t.nbytes for t in tables)
        # two words past the end: a lane reads a word or two beyond its chunk
        self.args = [mx.array(np.concatenate(codes + [np.zeros(4, np.uint32)])), mx.array(np.concatenate(scales)), mx.array(np.concatenate(rowmap)),
                     mx.array(np.concatenate(tables)), mx.array(np.array(dims, dtype=np.uint32)), mx.array(self.order(group_of, interleave))]
        self.widths = tuple(sorted(widths))
        self.kernel = _product_kernel(n, kind, self.widths, sg, stage_tables)
        mx.eval(self.args)

    def __call__(self, x):
        return self.kernel(inputs=[x.reshape(-1), *self.args], grid=(self.n_tg * self.sg * 32, 1, 1), threadgroup=(self.sg * 32, 1, 1),
                           output_shapes=[(self.size,)], output_dtypes=[mx.float32])[0]


# ----------------------------------------------------------------------------- the same codes as a dense matrix (prompts)

_DENSE = {}


def _dense_kernel(n, kind, widths, out_t, sg):
    key = (n, kind, widths, out_t, sg)
    if key not in _DENSE:
        ch = chunk(n)
        fns = {w: _rows_fn(w[0], ch, kind, scalar=w[1], dense=True, g=w[2]) for w in widths}
        header = f"#define RPS {RPS}u\n#define SG {sg}u\n#define OUT_T {out_t}\n" + "\n".join(src for _, src in fns.values())
        body = ["const uint tg = threadgroup_position_in_grid.x, gi = (uint)group_of[2 * tg], tgl = (uint)group_of[2 * tg + 1];",
                f"const device uint* d = dims + gi * {D};",
                "const uint ntg = d[0], rows = d[1], wpr = d[2];",
                "const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;",
                "switch (d[6]) {"]
        for (k, scalar, g), (name, _) in fns.items():
            body.append(f"    case {width_code(k, g, scalar)}u: {name}(codes + d[3], wpr, rows, scales + d[4], map + d[4], tables + d[5], y, {n}u, tgl, ntg, simd, lane); break;")
        body.append("}")
        _DENSE[key] = mx.fast.metal_kernel(name=f"lamq_dense_{n}_{kind}_{out_t}_{sg}_{'_'.join(f'{k}{int(s)}{g}' for k, s, g in widths)}",
                                           input_names=["codes", "scales", "map", "tables", "dims", "group_of"],
                                           output_names=["y"], source="\n".join("    " + s for s in body), header=header)
    return _DENSE[key]


def _product_dense(self, dtype=mx.float16, rowmap=None):
    """The groups' weights as a dense (size × n) matrix, each row scaled and at its output position (or at
    rowmap's, an int32 array like the product's own map)."""
    out_t = {mx.float16: "half", mx.float32: "float", mx.bfloat16: "bfloat16_t"}[dtype]
    kernel = _dense_kernel(self.n, self.kind, self.widths, out_t, self.sg)
    args = self.args if rowmap is None else [*self.args[:2], rowmap, *self.args[3:]]
    return kernel(inputs=args, grid=(self.n_tg * self.sg * 32, 1, 1), threadgroup=(self.sg * 32, 1, 1),
                  output_shapes=[(self.size, self.n)], output_dtypes=[dtype])[0]


Product.dense = _product_dense


# ----------------------------------------------------------------------------- the model

try:
    from . import lamq
except ImportError:  # run as a script beside lamq.py
    import lamq
sys.modules.setdefault("lamq", lamq)


# ----------------------------------------------------------------------------- one vector's rotation

_WHT_SRC = """
    // R x = H (s ⊙ x), H = (S_A ⊗ Q) / √n, for one vector seen as an A × C matrix (index a C + c), A = 32 R.
    // This launch: one simdgroup per column c. Its lanes hold the column (a = lane + 32 q), take the signs and the
    // Sylvester transform along a (in registers, then by SIMD shuffles) and write it back. MODE 1: the input is
    // silu(x0) · x1 (the feed-forward's gated activation), computed here.
    const uint lane = thread_index_in_simdgroup, c = threadgroup_position_in_grid.x;
    const ulong base = (ulong)thread_position_in_grid.y * (32u * R * C);   // which vector
    float y[R];
    #pragma unroll
    for (uint q = 0; q < R; q++) {
        const uint i = (lane + 32u * q) * C + c;
        float v = x0[base + i];
        GATE
        y[q] = v * s[i];
    }
    for (uint h = 1; h < R; h <<= 1)
        for (uint q = 0; q < R; q++) if (!(q & h)) { const float u = y[q], w = y[q | h]; y[q] = u + w; y[q | h] = u - w; }
    for (uint h = 1; h < 32; h <<= 1) {
        const bool hi = lane & h;
        #pragma unroll
        for (uint q = 0; q < R; q++) { const float o = simd_shuffle_xor(y[q], (ushort)h); y[q] = hi ? o - y[q] : y[q] + o; }
    }
    #pragma unroll
    for (uint q = 0; q < R; q++) z[base + (lane + 32u * q) * C + c] = y[q];
"""
_CF_SRC = """
    // the C × C factor along each row, and the norm: out[a, c] = Σ_k z[a, k] Q[c, k] / √n
    const uint i = thread_position_in_grid.x;
    if (i >= N) return;
    const ulong base = (ulong)thread_position_in_grid.y * N;
    const uint a = i / C, c = i - a * C;
    const device float* zr = z + base + a * C;
    const device float* qr = Q + c * C;
    float acc = 0.0f;
    for (uint k = 0; k < C; k++) acc += zr[k] * qr[k];
    out[base + i] = (OUT_T)(acc * rsqrt((float)N));
"""
_wht = {m: mx.fast.metal_kernel(name=f"lamq_wht_cols{m}", input_names=["x0"] + (["x1"] if m else []) + ["s"], output_names=["z"],
                                source=_WHT_SRC.replace("GATE", "v = v / (1.0f + metal::precise::exp(-v)) * (float)x1[base + i];" if m else "")) for m in (0, 1)}
_cfactor = mx.fast.metal_kernel(name="lamq_cfactor", input_names=["z", "Q"], output_names=["out"], source=_CF_SRC)


class Rotate1:
    """R x in two launches (lamq.Rot's compiled chain of MLX operations takes about 22 µs for one vector; both steps
    in one launch, a single threadgroup with the columns in threadgroup memory, measured slower: 9.22 against 9.08 ms a token).
    rot: the lamq.Rot it must equal. __call__(x) or, for the gated activation, __call__(gate, up)."""

    def __init__(self, rot):
        n = rot.n
        A = 128 if n % 128 == 0 else 32
        assert rot.A % A == 0 and rot.s.ndim == 1, (n, rot.A)
        self.n, self.C, self.R = n, n // A, A // 32
        Q = np.kron(lamq._sylvester(rot.A // A), np.array(rot.Pm_fwd).reshape(rot.p, rot.p))   # what is left of H beside S_A
        assert Q.shape[0] == self.C
        self.Q, self.s = mx.array(Q.astype(np.float32).reshape(-1)), rot.s
        qt = (Q.T / math.sqrt(n)).astype(np.float32)
        self.Qt_f32 = mx.array(qt)
        self.Qt_f16 = mx.array(qt.astype(np.float16))
        mx.eval(self.Q, self.Qt_f32, self.Qt_f16)

    def __call__(self, x, x1=None, dtype=mx.float32):
        """x: (n,) or (vectors, n), float16 or float32. The result has x's shape, in `dtype`."""
        mode = int(x1 is not None)
        T = x.size // self.n
        ins = [x.reshape(-1)] + ([x1.reshape(-1)] if mode else []) + [self.s]
        z = _wht[mode](inputs=ins, template=[("R", self.R), ("C", self.C)], grid=(32 * self.C, T, 1), threadgroup=(32, 1, 1),
                       output_shapes=[(T * self.n,)], output_dtypes=[mx.float32])[0]
        if T > 1 and dtype == mx.float16:
            A = 32 * self.R
            return (z.reshape(T * A, self.C).astype(mx.float16) @ self.Qt_f16).reshape(x.shape)
        return _cfactor(inputs=[z, self.Q], template=[("N", self.n), ("C", self.C), ("OUT_T", dtype)], grid=(-(-self.n // 64) * 64, T, 1),
                        threadgroup=(64, 1, 1), output_shapes=[x.shape], output_dtypes=[dtype])[0]


class Stage:
    """The matrices of one module that read one input through one rotation: W_i = diag(S_i) T[state] R.
    mats: (attribute, rows, groups); a group's `rows` are its rows' numbers in its own matrix."""

    def __init__(self, n, signs, mats, kind=None):
        self.n = n
        self.rot = lamq.Rot(np.asarray(signs, dtype=np.float32))
        self.rot1 = Rotate1(self.rot)
        groups, off, self.slices = [], 0, {}
        for attr, m, gs in mats:
            groups += [{**g, "rows": np.asarray(g["rows"], dtype=np.int64) + off} for g in gs]
            self.slices[attr] = (off, off + m)
            off += m
        self.product = Product(n, groups, off, kind)
        self.readers = len(mats)
        self.block = None      # block_q35.BlockStage, once a Block is built on the model: 8 or 16 vectors in one pass
        self._x = self._y = None
        self._left = 0

    def single(self, x, x1=None):
        """One vector (n,): every matrix's output, side by side. With x1, the input is silu(x) · x1."""
        return self.product(self.rot1(x, x1))

    def prefill_f16(self, x, x1=None):
        """Multi-token projection returning float16 (T, M), avoiding giant float32 intermediates on prefill."""
        T = x.size // self.n
        xf = x.reshape(T, self.n)
        x1f = None if x1 is None else x1.reshape(T, self.n)
        return self.rot1(xf, x1f, dtype=mx.float16) @ self.product.dense(mx.float16).T

    def part(self, y, attr):
        lo, hi = self.slices[attr]
        return y[lo:hi]

    def __call__(self, x):
        """Every matrix's output for x (..., n), side by side along the last axis. The result is kept until each
        matrix has read it, so the modules that share the stage cost one launch between them."""
        if x is not self._x:
            lead = x.shape[:-1]
            T = math.prod(lead)
            xf = x.reshape(T, self.n).astype(mx.float32)
            if T == 1:
                y = self.product(self.rot1(xf))
            elif self.block is not None and T in (8, 16):
                y = self.block(xf, T)
            else:
                y = (self.rot1(xf, dtype=mx.float16) @ self.product.dense(mx.float16).T).astype(mx.float32)
            self._x, self._y, self._left = x, y.reshape(*lead, -1), self.readers
        y = self._y
        self._left -= 1
        if self._left <= 0:
            self._x = self._y = None
        return y


class StageLinear(nn.Module):
    """One matrix of a stage, as the module mlx_lm's layers call."""

    def __init__(self, stage, attr):
        super().__init__()
        self._stage = [stage]  # in a list: not a child module or parameter
        self.lo, self.hi = stage.slices[attr]

    def __call__(self, x):
        return self._stage[0](x)[..., self.lo : self.hi]


def _lookup_source(n, kind):
    return f"""
    // one thread per (column, token): the token's row of its bucket, decoded and scaled, in the bucket's rotated basis
    const uint j = thread_position_in_grid.x, t = thread_position_in_grid.y;
    const uint tok = (uint)ids[t];
    const uint gi = (uint)bucket_of[tok], loc = (uint)local_of[tok];
    const device uint* d = dims + gi * {D};
    const uint wpr = d[2], k = d[6] & 15u, g = (d[6] >> 4) & 7u;
    const device uint* p = codes + d[3] + (ulong)loc * wpr;
    float v;
    if (d[6] >> 7) {{
        const uint bit = j * k, wi = bit >> 5, off = bit & 31u;
        uint s = p[wi] >> off;
        if (off + k > 32u) s |= p[wi + 1] << (32u - off);
        v = (float)tables[d[5] + (s & ((1u << k) - 1u))];
    }} else {{
        const uint bit = ({n}u - 1u - j) * k, wi = bit >> 5, off = bit & 31u;
        uint s = p[wi] >> off;
        if (off + 12u > 32u && wi + 1u < wpr) s |= p[wi + 1] << (32u - off);
        s &= 4095u;
        // a state alone indexes the last slot of its entry (the index has zeros above it)
        v = (float)tables[d[5] + s * (g == 1u ? 1u : g == 2u ? 2u : 4u) + g - 1u];
    }}
    e[(ulong)t * {n}u + j] = v * scales[d[4] + loc];
    if (j == 0u) bucket[t] = (int)gi;
"""


_LOOKUPS = {}


class FastTable(nn.Module):
    """A token table, in buckets of rows by token frequency, each with its own code and rotation. rows looks rows up
    and as_linear is the output head. A tied table (Qwen3.5-4B) is the model's embedding and is called to look rows up;
    a model that stores its head apart (Qwen3.8-27B) has two, and the one at lm_head (head=True) is called as a layer."""

    def __init__(self, arrays, meta, kind=None, head=False):
        super().__init__()
        kind = kind or TABLE
        self.head = bool(head)
        self.V, self.n = meta["shape"]
        groups, signs, bucket_of, local_of, order = [], [], np.zeros(self.V, np.uint16), np.zeros(self.V, np.int32), []
        for i, b in enumerate(meta["buckets"]):
            assert "matrix" in b and len(b["matrix"]["groups"]) == 1, "a bucket here is one coded group"
            g = b["matrix"]["groups"][0]
            key = g["key"]
            tokens = np.asarray(arrays[b["rows"]], dtype=np.int64)[np.asarray(arrays[key + ".rows"], dtype=np.int64)]  # of each stored row
            groups.append({"codes": arrays[key + ".codes"], "scales": arrays[key + ".scales"], "rows": tokens, "bits": g["bits"],
                           "table": arrays[key + ".table"], "input": i, "scalar": g["kind"] != "trellis"})
            signs.append(np.asarray(arrays[b["matrix"]["signs"]], dtype=np.float32))
            bucket_of[tokens], local_of[tokens] = i, np.arange(len(tokens))
            order.append(tokens)
        self.nb = len(groups)
        self.product = Product(self.n, groups, self.V, kind, threadgroups=HEAD_THREADGROUPS)
        self.rot = lamq.Rot(np.stack(signs))            # one rotation per bucket: the same H, its own signs
        self.signs = mx.array(np.stack(signs))
        self.ones = mx.ones((self.n,), mx.float32)
        self.bucket_of, self.local_of = mx.array(bucket_of), mx.array(local_of)
        order = np.concatenate(order)
        self.bounds = np.cumsum([0] + [len(g["rows"]) for g in groups])
        pos = np.empty(self.V, np.int32)
        pos[order] = np.arange(self.V, dtype=np.int32)   # a token's place when the buckets sit one after another
        self.by_bucket = mx.array(pos[order])            # the map that lays the dense rows out bucket by bucket
        self.inverse = mx.array(pos)
        key = (self.n, kind)
        if key not in _LOOKUPS:
            _LOOKUPS[key] = mx.fast.metal_kernel(name=f"lamq_lookup_{self.n}_{kind}", input_names=["ids", "bucket_of", "local_of", "codes", "scales", "tables", "dims"],
                                                 output_names=["e", "bucket"], source=_lookup_source(self.n, kind))
        self.lookup = _LOOKUPS[key]
        self.freeze()

    def __call__(self, x):
        return self.as_linear(x) if self.head else self.rows(x)

    def rows(self, ids):
        shape = ids.shape
        flat = ids.reshape(-1).astype(mx.int32)
        T = flat.shape[0]
        codes, scales, _, tables, dims, _ = self.product.args
        e, bucket = self.lookup(inputs=[flat, self.bucket_of, self.local_of, codes, scales, tables, dims], grid=(self.n, T, 1),
                                threadgroup=(64, 1, 1), output_shapes=[(T, self.n), (T,)], output_dtypes=[mx.float32, mx.int32])
        out = self.rot._b(e, self.ones, self.rot.P_bwd) * self.signs[bucket]   # Rᵀ: Hᵀ, then the bucket's signs
        return out.reshape(*shape, self.n)

    def as_linear(self, h):
        lead = h.shape[:-1]
        T = math.prod(lead)
        x = h.reshape(T, self.n).astype(mx.float32)
        if T == 1:
            xr = self.rot.fwd(mx.broadcast_to(x, (self.nb, self.n)))      # every bucket's rotation of the one vector
            return self.product(xr).reshape(*lead, self.V)
        W = self.product.dense(mx.float16, self.by_bucket)
        outs = []
        for i in range(self.nb):
            lo, hi = int(self.bounds[i]), int(self.bounds[i + 1])
            xi = self.rot._f(x, self.signs[i], self.rot.P_fwd).astype(mx.float16)
            outs.append(xi @ W[lo:hi].T)
        return mx.concatenate(outs, axis=-1)[:, self.inverse].astype(mx.float32).reshape(*lead, self.V)


# ----------------------------------------------------------------------------- the steps between products, one token

_GDN_PRE = """
    // The gated delta rule's inputs for one token. One thread per channel of the depthwise convolution; a threadgroup
    // is one head (DK channels). conv over the last three tokens' projections and this one's, SiLU, then the query
    // and key heads' RMS norms with their scales (1/DK and 1/√DK). Layout in and out: [queries | keys | values].
    const uint c = thread_position_in_grid.x, head = c / DK;
    float v = cs[c] * w[4 * c] + cs[CD + c] * w[4 * c + 1] + cs[2 * CD + c] * w[4 * c + 2] + qkv[c] * w[4 * c + 3];
    v = v / (1.0f + metal::precise::exp(-v));
    cs_out[c] = cs[CD + c]; cs_out[CD + c] = cs[2 * CD + c]; cs_out[2 * CD + c] = qkv[c];
    threadgroup float red[DK / 32];
    float ss = simd_sum(v * v);
    if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = 0.0f;
    for (uint i = 0; i < DK / 32; i++) ss += red[i];
    if (head < 2 * HK) v *= metal::precise::rsqrt(ss / (float)DK + 1e-6f) * (head < HK ? 1.0f / (float)DK : metal::precise::rsqrt((float)DK));
    out[c] = v;
"""
_GDN_STEP = """
    // One step of the gated delta rule (mlx_lm's kernel for one token, with the gates computed here): a simdgroup per
    // (value head, value row), each lane holding DK / 32 of the row's state.
    const uint hv = thread_position_in_grid.z, hk = hv / (HV / HK);
    constexpr int n_per_t = DK / 32;
    const device float* q_ = x + hk * DK;
    const device float* k_ = x + HK * DK + hk * DK;
    const uint dk_idx = thread_position_in_threadgroup.x, dv_idx = thread_position_in_grid.y;
    const device float* i_state = state_in + (hv * DV + dv_idx) * DK;
    device float* o_state = state_out + (hv * DV + dv_idx) * DK;
    const float sp = ab[hv] + dt_bias[hv];
    const float g = metal::precise::exp(-metal::precise::exp(A_log[hv]) * (sp > 20.0f ? sp : metal::precise::log(1.0f + metal::precise::exp(sp))));
    const float beta = 1.0f / (1.0f + metal::precise::exp(-ab[HV + hv]));
    float state[n_per_t];
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        const uint s_idx = n_per_t * dk_idx + i;
        state[i] = i_state[s_idx] * g;
        kv_mem += state[i] * k_[s_idx];
    }
    kv_mem = simd_sum(kv_mem);
    const float delta = (x[2 * HK * DK + hv * DV + dv_idx] - kv_mem) * beta;
    float o = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        const uint s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        o += state[i] * q_[s_idx];
        o_state[s_idx] = state[i];
    }
    o = simd_sum(o);
    if (thread_index_in_simdgroup == 0) y[hv * DV + dv_idx] = o;
"""
_GDN_POST = """
    // The rule's output, normed per value head (RMS, with the weight) and gated by SiLU(z). A threadgroup is one head.
    const uint i = thread_position_in_grid.x;
    const float v = y[i];
    threadgroup float red[DV / 32];
    float ss = simd_sum(v * v);
    if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = 0.0f;
    for (uint j = 0; j < DV / 32; j++) ss += red[j];
    const float zz = z[i];
    out[i] = v * metal::precise::rsqrt(ss / (float)DV + EPS) * w[i % DV] * (zz / (1.0f + metal::precise::exp(-zz)));
"""
_gdn_pre = mx.fast.metal_kernel(name="lamq_gdn_pre", input_names=["qkv", "cs", "w"], output_names=["out", "cs_out"], source=_GDN_PRE)
_gdn_step = mx.fast.metal_kernel(name="lamq_gdn_step", input_names=["x", "ab", "A_log", "dt_bias", "state_in"], output_names=["y", "state_out"], source=_GDN_STEP)
_gdn_posts = {}


def _gdn_post(eps):  # a template takes integers and types only: the norm's epsilon is written into the source
    if eps not in _gdn_posts:
        _gdn_posts[eps] = mx.fast.metal_kernel(name=f"lamq_gdn_post_{len(_gdn_posts)}", input_names=["y", "z", "w"], output_names=["out"],
                                               source=_GDN_POST.replace("EPS", f"{eps!r}f"))
    return _gdn_posts[eps]


_GDN_PRE_SEQ = """
    // Fused prefill conv1d(k=4) + SiLU + Q/K L2-norm + conv-state update for T tokens in one launch.
    // One threadgroup = one (head, token); channels [0..HK*DK) -> q, [HK*DK..2*HK*DK) -> k, [2*HK*DK..CD) -> v.
    const uint c = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const uint T = dims[0];
    const uint head = c / DK;
    const device half* qkv_t = qkv + (ulong)t * CD;
    float s0 = (t >= 3u) ? (float)qkv[(ulong)(t - 3u) * CD + c] : cs[(ulong)t * CD + c];
    float s1 = (t >= 2u) ? (float)qkv[(ulong)(t - 2u) * CD + c] : cs[(ulong)(t + 1u) * CD + c];
    float s2 = (t >= 1u) ? (float)qkv[(ulong)(t - 1u) * CD + c] : cs[(ulong)(t + 2u) * CD + c];
    float s3 = (float)qkv_t[c];
    float v = s0 * w[4 * c] + s1 * w[4 * c + 1] + s2 * w[4 * c + 2] + s3 * w[4 * c + 3];
    v = v / (1.0f + metal::precise::exp(-v));
    if (t == T - 1u) {
        cs_out[c] = s1;
        cs_out[CD + c] = s2;
        cs_out[2 * CD + c] = s3;
    }
    if (head < 2u * HK) {
        threadgroup float red[DK / 32];
        float ss = simd_sum(v * v);
        if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ss = 0.0f;
        for (uint i = 0; i < DK / 32; i++) ss += red[i];
        const float scale = metal::precise::rsqrt(ss + 1e-6f) * (head < HK ? metal::precise::rsqrt((float)DK) : 1.0f);
        v *= scale;
        if (head < HK) {
            q_out[(ulong)t * (HK * DK) + c] = (half)v;
        } else {
            k_out[(ulong)t * (HK * DK) + (c - HK * DK)] = (half)v;
        }
    } else {
        v_out[(ulong)t * (HV * DV) + (c - 2u * HK * DK)] = (half)v;
    }
"""
_GDN_GATES_SEQ = """
    // Fused scalar GDN gates (g, beta) from ab [T, 2*HV] for all T tokens in one launch.
    const uint hv = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const ulong base = (ulong)t * (2u * HV);
    const float sp = (float)ab[base + hv] + dt_bias[hv];
    const float softp = sp > 20.0f ? sp : metal::precise::log(1.0f + metal::precise::exp(sp));
    g_out[(ulong)t * HV + hv] = metal::precise::exp(-metal::precise::exp(A_log[hv]) * softp);
    beta_out[(ulong)t * HV + hv] = 1.0f / (1.0f + metal::precise::exp(-(float)ab[base + HV + hv]));
"""
_GDN_POST_SEQ = """
    // Fused RMSNormGated across T tokens in one launch (reads half y and half z, writes half out).
    const uint i = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const ulong off = (ulong)t * (HV * DV) + i;
    const float v = (float)y[off];
    threadgroup float red[DV / 32];
    float ss = simd_sum(v * v);
    if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = 0.0f;
    for (uint j = 0; j < DV / 32; j++) ss += red[j];
    const float zz = (float)z[off];
    out[off] = (half)(v * metal::precise::rsqrt(ss / (float)DV + EPS) * w[i % DV] * (zz / (1.0f + metal::precise::exp(-zz))));
"""
_gdn_pre_seq = mx.fast.metal_kernel(
    name="lamq_gdn_pre_seq",
    input_names=["qkv", "cs", "w", "dims"],
    output_names=["q_out", "k_out", "v_out", "cs_out"],
    source=_GDN_PRE_SEQ,
)
_gdn_gates_seq = mx.fast.metal_kernel(
    name="lamq_gdn_gates_seq",
    input_names=["ab", "A_log", "dt_bias"],
    output_names=["g_out", "beta_out"],
    source=_GDN_GATES_SEQ,
)
_gdn_posts_seq = {}


def _gdn_post_seq(eps):
    if eps not in _gdn_posts_seq:
        _gdn_posts_seq[eps] = mx.fast.metal_kernel(
            name=f"lamq_gdn_post_seq_{len(_gdn_posts_seq)}",
            input_names=["y", "z", "w"],
            output_names=["out"],
            source=_GDN_POST_SEQ.replace("EPS", f"{eps!r}f"),
        )
    return _gdn_posts_seq[eps]


class FastGatedDelta(nn.Module):
    """mlx_lm's GatedDeltaNet in fused Metal launches for both single-token decode and multi-token prefill."""

    def __init__(self, inner, stage_in, stage_out):
        super().__init__()
        self.inner = inner
        self._stages = [stage_in, stage_out]
        m = inner
        self.ok = m.head_k_dim == m.head_v_dim and m.head_k_dim % 32 == 0 and m.conv_kernel_size == 4
        self.w_ab = mx.concatenate([m.in_proj_a.weight, m.in_proj_b.weight]).astype(mx.float32)
        self.w_ab_f16_t = mx.contiguous(self.w_ab.astype(mx.float16).T)
        self.conv_w = m.conv1d.weight.reshape(m.conv_dim, 4).astype(mx.float32)
        self.t = [("CD", m.conv_dim), ("DK", m.head_k_dim), ("DV", m.head_v_dim), ("HK", m.num_k_heads), ("HV", m.num_v_heads)]
        self.eps = float(m.layer_norm_epsilon)
        self._A_log_f32 = m.A_log.astype(mx.float32)
        self._dt_bias_f32 = m.dt_bias.astype(mx.float32)
        self._norm_w_f32 = m.norm.weight.astype(mx.float32)
        mx.eval(self.w_ab, self.w_ab_f16_t, self.conv_w, self._A_log_f32, self._dt_bias_f32, self._norm_w_f32)
        self.freeze()

    def __call__(self, inputs, mask=None, cache=None):
        m = self.inner
        if not self.ok or inputs.shape[0] != 1 or mask is not None or cache is None or cache.lengths is not None:
            return m(inputs, mask, cache)
        T = int(inputs.shape[1])
        s_in, s_out = self._stages
        if T == 1:
            x = inputs.reshape(-1).astype(mx.float32)
            y = s_in.single(x)
            ab = self.w_ab @ x
            cs = cache[0] if cache[0] is not None else mx.zeros((3 * m.conv_dim,), mx.float32)
            state = cache[1] if cache[1] is not None else mx.zeros((1, m.num_v_heads, m.head_v_dim, m.head_k_dim), mx.float32)
            qkv, cs = _gdn_pre(inputs=[s_in.part(y, "in_proj_qkv"), cs.reshape(-1).astype(mx.float32), self.conv_w], template=self.t,
                               grid=(m.conv_dim, 1, 1), threadgroup=(m.head_k_dim, 1, 1),
                               output_shapes=[(m.conv_dim,), (1, 3, m.conv_dim)], output_dtypes=[mx.float32, mx.float32])
            out, state = _gdn_step(inputs=[qkv, ab, self._A_log_f32, self._dt_bias_f32, state.astype(mx.float32)], template=self.t,
                                   grid=(32, m.head_v_dim, m.num_v_heads), threadgroup=(32, 4, 1),
                                   output_shapes=[(m.value_dim,), state.shape], output_dtypes=[mx.float32, mx.float32])
            out = _gdn_post(self.eps)(inputs=[out, s_in.part(y, "in_proj_z"), self._norm_w_f32], template=self.t,
                            grid=(m.value_dim, 1, 1), threadgroup=(m.head_v_dim, 1, 1), output_shapes=[(m.value_dim,)], output_dtypes=[mx.float32])[0]
            cache[0], cache[1] = cs, state
            cache.advance(1)
            return s_out.single(out).reshape(1, 1, -1)
        if s_in.block is not None and T in (8, 16):
            return m(inputs, mask, cache)
        # Fused multi-token prefill: Stage(f16) -> _gdn_pre_seq -> _gdn_gates_seq -> packed GDN kernel -> _gdn_post_seq -> Stage(f16)
        from mlx_lm.models.gated_delta import gated_delta_kernel

        y = s_in.prefill_f16(inputs)
        lo_qkv, hi_qkv = s_in.slices["in_proj_qkv"]
        lo_z, hi_z = s_in.slices["in_proj_z"]
        qkv_in = mx.contiguous(y[:, lo_qkv:hi_qkv])
        z_in = mx.contiguous(y[:, lo_z:hi_z])
        ab = inputs.reshape(T, -1).astype(mx.float16) @ self.w_ab_f16_t
        cs = cache[0].reshape(-1).astype(mx.float32) if cache[0] is not None else mx.zeros((3 * m.conv_dim,), mx.float32)
        state = cache[1].astype(mx.float32) if cache[1] is not None else mx.zeros((1, m.num_v_heads, m.head_v_dim, m.head_k_dim), mx.float32)
        q, k, v, cs_out = _gdn_pre_seq(
            inputs=[qkv_in, cs, self.conv_w, mx.array([T], dtype=mx.uint32)],
            template=self.t,
            grid=(m.conv_dim, T, 1),
            threadgroup=(m.head_k_dim, 1, 1),
            output_shapes=[(1, T, m.num_k_heads, m.head_k_dim), (1, T, m.num_k_heads, m.head_k_dim),
                           (1, T, m.num_v_heads, m.head_v_dim), (1, 3, m.conv_dim)],
            output_dtypes=[mx.float16, mx.float16, mx.float16, mx.float32],
        )
        g, beta = _gdn_gates_seq(
            inputs=[ab, self._A_log_f32, self._dt_bias_f32],
            template=[("HV", m.num_v_heads)],
            grid=(m.num_v_heads, T, 1),
            threadgroup=(min(32, m.num_v_heads), 1, 1),
            output_shapes=[(1, T, m.num_v_heads), (1, T, m.num_v_heads)],
            output_dtypes=[mx.float32, mx.float32],
        )
        out, state = gated_delta_kernel(q, k, v, g, beta, state)
        out = _gdn_post_seq(self.eps)(
            inputs=[out.reshape(-1), z_in.reshape(-1), self._norm_w_f32],
            template=self.t,
            grid=(m.value_dim, T, 1),
            threadgroup=(m.head_v_dim, 1, 1),
            output_shapes=[(T, m.value_dim)],
            output_dtypes=[mx.float16],
        )[0]
        cache[0], cache[1] = cs_out, state
        cache.advance(T)
        return s_out.prefill_f16(out).astype(mx.float32).reshape(1, T, -1)


class FastMLP(nn.Module):
    """The feed-forward block: the gated activation SiLU(gate)*up is fused inside the down-stage rotation for all T."""

    def __init__(self, inner, stage_in, stage_down):
        super().__init__()
        self.inner = inner
        self._stages = [stage_in, stage_down]
        self.freeze()

    def __call__(self, x):
        s_in, s_down = self._stages
        T = x.size // s_in.n
        if T == 1:
            y = s_in.single(x.reshape(-1).astype(mx.float32))
            return s_down.single(s_in.part(y, "gate_proj"), s_in.part(y, "up_proj")).reshape(x.shape)
        if s_in.block is not None and T in (8, 16):
            return self.inner(x)
        y = s_in.prefill_f16(x)
        glo, ghi = s_in.slices["gate_proj"]
        ulo, uhi = s_in.slices["up_proj"]
        return s_down.prefill_f16(mx.contiguous(y[:, glo:ghi]), mx.contiguous(y[:, ulo:uhi])).astype(mx.float32).reshape(x.shape)


_ADD_NORM = """
    // h = a + b and its RMS norm times w, in one launch (mlx_lm: an add, then a norm). One threadgroup of 1,024.
    const uint tid = thread_position_in_threadgroup.x;
    constexpr uint E = (N + 1023) / 1024;
    float v[E];
    float ss = 0.0f;
    for (uint e = 0; e < E; e++) {
        const uint i = tid + e * 1024u;
        v[e] = i < N ? a[i] + b[i] : 0.0f;
        ss += v[e] * v[e];
        if (i < N) h[i] = v[e];
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
        if (i < N) hn[i] = v[e] * r * w[i];
    }
"""
_add_norms = {}


def add_norm(a, b, w, eps):
    """(a + b, rms_norm(a + b) · w) for one vector."""
    if eps not in _add_norms:
        _add_norms[eps] = mx.fast.metal_kernel(name=f"lamq_add_norm_{len(_add_norms)}", input_names=["a", "b", "w"], output_names=["h", "hn"],
                                               source=_ADD_NORM.replace("EPS", f"{eps!r}f"))
    n = a.size
    return _add_norms[eps](inputs=[a.reshape(-1), b.reshape(-1), w], template=[("N", n)], grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
                           output_shapes=[(n,), (n,)], output_dtypes=[mx.float32, mx.float32])


class FastLayer(nn.Module):
    """A decoder layer for one token: each residual add shares a launch with the norm that reads its result (this
    layer's second norm, and the next layer's first, whose weight it is handed)."""

    def __init__(self, inner, eps):
        super().__init__()
        self.inner = inner
        self.is_linear = inner.is_linear
        self.eps = float(eps)
        self._next = [None]     # the next layer: it reads `out` through its own first norm
        self._normed = [None]   # (x, norm(x)) left here by the layer before
        self._tap = [None]      # a list, when a drafter reads this layer: every output is appended to it
        self.freeze()

    def __call__(self, x, mask=None, cache=None):
        out = self._layer(x, mask, cache)
        if self._tap[0] is not None:
            self._tap[0].append(out)
        return out

    def _layer(self, x, mask, cache):
        L = self.inner
        if x.shape[0] != 1 or x.shape[1] != 1:
            return L(x, mask=mask, cache=cache)
        kept, self._normed[0] = self._normed[0], None
        xn = kept[1] if kept is not None and kept[0] is x else L.input_layernorm(x)
        attn = L.linear_attn if L.is_linear else L.self_attn
        r = attn(xn, mask, cache)
        h, hn = add_norm(x, r, L.post_attention_layernorm.weight, self.eps)
        m = L.mlp(hn.reshape(x.shape))
        nxt = self._next[0]
        if nxt is None:
            return h.reshape(x.shape) + m
        out, outn = add_norm(h, m, nxt.inner.input_layernorm.weight, self.eps)
        out, outn = out.reshape(x.shape), outn.reshape(x.shape)
        nxt._normed[0] = (out, outn)
        return out


def _groups(arrays, meta):
    out = []
    for g in meta["groups"]:
        key = g["key"]
        assert g["padded_columns"] == meta["shape"][1], "padded columns are not handled here"
        out.append({"codes": arrays[key + ".codes"], "scales": arrays[key + ".scales"], "rows": arrays[key + ".rows"], "bits": g["bits"],
                    "table": arrays[key + ".table"], "scalar": g["kind"] != "trellis"})
    return out


def _set(root, path, module):
    parts = path.split(".")
    for part in parts[:-1]:
        root = root[int(part)] if part.isdigit() else getattr(root, part)
    setattr(root, parts[-1], module)


def load(path, kind=None, verify=True, fuse=None, kv_mode=None):
    """The packed file as an mlx_lm model whose coded matrices run through this module, and its tokenizer.
    fuse=False keeps mlx_lm's own steps between the products (the check that the fused steps equal them)."""
    from compiler_artifact import read_artifact
    from mlx_lm.utils import _get_classes, load_tokenizer
    path = Path(path)
    fuse = bool(FUSE_STEPS) if fuse is None else fuse
    arrays, manifest = read_artifact(path, verify=verify)
    config = json.loads((path / "config.json").read_text())
    model_class, args_class = _get_classes(config=config)
    model = model_class(args_class.from_dict(config))

    def dense_array(name):  # every uncoded tensor in float32: a single token's work on them is small
        value = mx.array(arrays[name])
        return (value.view(mx.bfloat16) if manifest["tensors"][name]["dtype"] == "BF16" else value).astype(mx.float32)

    dense = {name: dense_array(name) for name in manifest["dense"]}
    model.load_weights(list(model.sanitize(dense).items()), strict=False)
    rename = lambda n: next(iter(model.sanitize({n: mx.zeros((1,))})))  # Hugging Face name → mlx_lm name
    stages = {}
    for name, meta in manifest["matrices"].items():
        if meta["kind"] == "table":
            at = rename(name).removesuffix(".weight")
            _set(model, at, FastTable(arrays, meta, kind, head=at.endswith("lm_head")))
            continue
        owner, attr = rename(name).removesuffix(".weight").rsplit(".", 1)
        signs = np.asarray(arrays[meta["signs"]])
        stages.setdefault((owner, signs.tobytes()), (signs, []))[1].append((attr, meta["shape"], _groups(arrays, meta)))
    model.lamq_stages, stage_of = [], {}
    for (owner, _), (signs, mats) in stages.items():
        n = mats[0][1][1]
        assert all(shape[1] == n for _, shape, _ in mats)
        stage = Stage(n, signs, [(attr, shape[0], gs) for attr, shape, gs in mats], kind)
        for attr, _, _ in mats:
            _set(model, f"{owner}.{attr}", StageLinear(stage, attr))
            stage_of[f"{owner}.{attr}"] = stage
        model.lamq_stages.append(stage)
    if fuse:
        prefix = rename("model.layers.0.mlp.gate_proj.weight").split("layers.0.")[0] + "layers."
        for i, layer in enumerate(model.layers):
            at = lambda a: stage_of.get(f"{prefix}{i}.{a}")
            if at("mlp.gate_proj") is not None and at("mlp.gate_proj") is at("mlp.up_proj") and at("mlp.down_proj") is not None:
                layer.mlp = FastMLP(layer.mlp, at("mlp.gate_proj"), at("mlp.down_proj"))
            if layer.is_linear and at("linear_attn.in_proj_qkv") is not None and at("linear_attn.in_proj_qkv") is at("linear_attn.in_proj_z") \
                    and at("linear_attn.out_proj") is not None and isinstance(layer.linear_attn.in_proj_a, nn.Linear):
                layer.linear_attn = FastGatedDelta(layer.linear_attn, at("linear_attn.in_proj_qkv"), at("linear_attn.out_proj"))
        if FUSE_ADD_NORM:
            tm = model.language_model.model
            fast = [FastLayer(layer, tm.norm.eps) for layer in tm.layers]
            for a, b in zip(fast, fast[1:]):
                a._next[0] = b
            tm.layers = fast
    resolved_kv = (os.environ.get("LAMQ_Q35_KV", "fp16") if kv_mode is None else kv_mode).strip()
    if resolved_kv.lower() not in ("fp16", "none", ""):
        sink_tokens = int(os.environ.get("LAMQ_Q35_KV_SINK", "4"))
        try:
            from .kv_q35 import install_kv_quant
        except ImportError:
            from kv_q35 import install_kv_quant
        install_kv_quant(model, resolved_kv, sink_tokens=sink_tokens)
    model.eval()
    mx.eval(model.parameters())
    return model, load_tokenizer(path)
