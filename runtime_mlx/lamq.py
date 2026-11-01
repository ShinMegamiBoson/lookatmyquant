"""LookAtMyQuant runtime for Apple Silicon (MLX): Lloyd–Max and trellis codes with randomized Hadamard rotations.

A matrix W is stored as W ≈ R_outᵀ · diag(S) · L[Q] · R_in, where
  Q          b-bit codes, packed 32 to a group of b uint32 words,
  L          the 2^b Lloyd–Max levels of a unit Gaussian (fixed by b, not stored per model),
  S          one scale per row,
  R = H·diag(s)  a randomized Hadamard transform, stored as its signs s;
                 H = kron(A, P)/√n with A a Sylvester Hadamard of order 2^a and P = 1, [[1,1],[1,−1]] or a Paley matrix.
y = W x is computed as: x' = R_in x  →  y' = S ⊙ (L[Q] x')  →  y = R_outᵀ y'.

A trellis matrix replaces L[Q] with V[state]: V holds the code's 4,096 state values. A row is one stream of k-bit symbols
with later columns at lower bits, its runs of 32 stored last column first, so an entry's state is the 12 bits at its place in
the stream: reading it touches no other row, and at most one word of the run before.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import numpy as np


# ----------------------------------------------------------------------------- rotations

KERNEL_ROT = True  # one Metal kernel per rotation (False: a compiled chain of MLX ops)

def _paley(q: int) -> np.ndarray:
    """Paley's construction I of a Hadamard matrix of order q + 1, q prime ≡ 3 (mod 4) — the same matrix the quantizer uses."""
    chi = np.array([0] + [1 if pow(i, (q - 1) // 2, q) == 1 else -1 for i in range(1, q)], dtype=np.float64)
    Q = np.array([[chi[(j - i) % q] for j in range(q)] for i in range(q)])
    S = np.zeros((q + 1, q + 1))
    S[0, 1:] = 1
    S[1:, 0] = -1
    S[1:, 1:] = Q
    return S + np.eye(q + 1)


def _sylvester(n: int) -> np.ndarray:
    """The Sylvester Hadamard matrix of order n (a power of 2), unnormalized."""
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def _factor(n: int) -> tuple[int, int]:
    """n = 2^a · p, with the largest a whose p the quantizer's construction supports (1, 2 or a Paley order)."""
    def is_prime(k):
        return k > 1 and all(k % d for d in range(2, int(k ** 0.5) + 1))
    for a in range(int(math.log2(n)), -1, -1):
        if n % 2 ** a:
            continue
        p = n // 2 ** a
        if p in (1, 2) or (is_prime(p - 1) and (p - 1) % 4 == 3):
            return 2 ** a, p
    raise ValueError(f"no Hadamard construction for {n}")


@lru_cache(maxsize=None)
def _rot_fn(A: int, p: int, n: int, transpose: bool):
    """One fused, compiled rotation for a given size: H (s ⊙ x), or s ⊙ (Hᵀ y) when transpose."""
    norm = 1.0 / math.sqrt(n)

    def f(x, s, P):
        if not transpose:
            x = x * s
        shp = x.shape
        X = x.reshape(*shp[:-1], A, p)  # kron(A, P') x = vec(A X P'ᵀ), X the row-major reshape
        X = mx.swapaxes(mx.hadamard_transform(mx.swapaxes(X, -1, -2), scale=1.0), -1, -2)  # A along the 2^a axis
        if p > 1:  # a broadcast product, not a matmul: float32 matmul on the GPU is not exact enough here
            X = (X[..., None, :] * P).sum(-1)
        y = (X * norm).reshape(shp)
        return y * s if transpose else y
    return mx.compile(f)


_ROT_SRC = """
    // One threadgroup per vector: signs, a Sylvester transform along the 2^a axis, the p×p factor along the other, 1/√n.
    // TRANSPOSE = 0: y = H (s ⊙ x);  1: y = s ⊙ (Hᵀ x).  Pm is the p×p matrix applied on the right as X·Pmᵀ.
    const uint n = dims[0], A = dims[1], p = dims[2], srows = dims[3];
    const uint v = threadgroup_position_in_grid.x, tid = thread_position_in_threadgroup.x, nt = threads_per_threadgroup.x;
    threadgroup BUF buf[MAXN];
    const device float* xv = x + (ulong)v * n;
    const device float* sv = s + (ulong)(v % srows) * n;
    for (uint i = tid; i < n; i += nt) buf[i] = (BUF)(TRANSPOSE ? xv[i] : xv[i] * sv[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < A; h <<= 1) {
        for (uint j = tid; j < (A / 2) * p; j += nt) {
            const uint pair = j / p, c = j % p;
            const uint r = (pair / h) * 2 * h + (pair % h);
            const uint i0 = r * p + c, i1 = (r + h) * p + c;
            const float u = buf[i0], w = buf[i1];
            buf[i0] = (BUF)(u + w); buf[i1] = (BUF)(u - w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float norm = rsqrt((float)n);
    device float* yv = y + (ulong)v * n;
    for (uint i = tid; i < n; i += nt) {
        const uint r = i / p, c = i % p;
        float acc = 0.0f;
        for (uint k = 0; k < p; k++) acc += (float)buf[r * p + k] * Pm[c * p + k];
        acc *= norm;
        yv[i] = TRANSPOSE ? acc * sv[i] : acc;
    }
"""
_rot = mx.fast.metal_kernel(name="lamq_rot", input_names=["x", "s", "Pm", "dims"], output_names=["y"], source=_ROT_SRC)


def rot_apply(x, s, Pm, A, p, transpose):
    """The rotation as one kernel launch per call; x (..., n) float32, s (n,) or (slices, n) matching x's trailing axes."""
    n = x.shape[-1]
    shp = x.shape
    nvec = int(np.prod(shp[:-1])) if len(shp) > 1 else 1
    srows = 1 if s.ndim == 1 else s.shape[0]
    buf, maxn = mx.float32, 8192  # threadgroup memory is 32 KB; larger rotations use the compiled path
    dims = mx.array([n, A, p, srows], dtype=mx.uint32)
    y = _rot(inputs=[x.astype(mx.float32).reshape(nvec, n), s.astype(mx.float32), Pm, dims],
             template=[("BUF", buf), ("MAXN", maxn), ("TRANSPOSE", int(transpose))],
             grid=(256 * nvec, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(nvec, n)], output_dtypes=[mx.float32])[0]
    return y.reshape(shp)


_MANY_SRC = """
    // R x for many vectors, one threadgroup each. The vector is viewed as a (32·RR) × C matrix (index r·C + c); simdgroup
    // c takes column c: signs, the Sylvester transform along r in registers and by SIMD shuffles; then each thread takes
    // rows and applies the C × C factor Q (the rotation's p × p factor, times a Sylvester factor split off from A).
    const uint t = threadgroup_position_in_grid.x, tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, c = simdgroup_index_in_threadgroup;
    constexpr uint NT = 32 * C;
    threadgroup BUF buf[N];
    threadgroup float qm[C * C];
    for (uint i = tid; i < C * C; i += NT) qm[i] = Q[i];
    const device T* xv = x + (ulong)t * N;
    float y[RR];
    #pragma unroll
    for (uint q = 0; q < RR; q++) { const uint i = (lane + 32u * q) * C + c; y[q] = (float)xv[i] * s[i]; }
    for (uint h = 1; h < RR; h <<= 1)
        for (uint q = 0; q < RR; q++) if (!(q & h)) { const float u = y[q], w = y[q | h]; y[q] = u + w; y[q | h] = u - w; }
    for (uint h = 1; h < 32; h <<= 1) {
        const bool hi = lane & h;
        for (uint q = 0; q < RR; q++) { const float o = simd_shuffle_xor(y[q], (ushort)h); y[q] = hi ? o - y[q] : y[q] + o; }
    }
    const float na = rsqrt((float)(32 * RR));
    for (uint q = 0; q < RR; q++) buf[(lane + 32u * q) * C + c] = (BUF)(y[q] * na);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float nc = rsqrt((float)C);
    device half* ov = out + (ulong)t * N;
    for (uint r = tid; r < 32 * RR; r += NT) {
        float v[C];
        for (uint k = 0; k < C; k++) v[k] = buf[r * C + k];
        for (uint cc = 0; cc < C; cc++) {
            float acc = 0.0f;
            for (uint k = 0; k < C; k++) acc += v[k] * qm[cc * C + k];
            ov[r * C + cc] = (half)(acc * nc);
        }
    }
"""
_many = mx.fast.metal_kernel(name="lamq_rot_many", input_names=["x", "s", "Q"], output_names=["out"], source=_MANY_SRC)


class Rot:
    """R = H·diag(s). fwd(x) = R x, bwd(y) = Rᵀ y, along the last axis. many(x) = R x as float16 for many vectors."""

    def __init__(self, signs: np.ndarray, P: np.ndarray | None = None):
        self.n = signs.shape[-1]  # signs may be (slices, n): one rotation per slice, applied along the last axis
        self.A, self.p = _factor(self.n)
        if P is None:
            P = np.ones((1, 1)) if self.p == 1 else np.array([[1.0, 1.0], [1.0, -1.0]]) if self.p == 2 else _paley(self.p - 1)
        self.P_fwd = mx.array(P.astype(np.float32))  # X·Pᵀ as (X[..., None, :] * P).sum(-1)
        self.P_bwd = mx.array(P.T.astype(np.float32))  # X·P
        self.s = mx.array(signs.astype(np.float32))
        self.Pm_fwd = mx.array(P.astype(np.float32).reshape(-1))  # H x: X·Pᵀ, so Pm = P
        self.Pm_bwd = mx.array(P.T.astype(np.float32).reshape(-1))  # Hᵀ y: X·P, so Pm = Pᵀ
        self._f = _rot_fn(self.A, self.p, self.n, False)
        self._b = _rot_fn(self.A, self.p, self.n, True)

    def _kernel(self, x):  # the kernel runs one threadgroup per vector: faster for many vectors (prompts), slower for one
        return KERNEL_ROT and self.n <= 8192 and x.size // self.n >= 8

    def many(self, x):
        """R x for a prompt's worth of vectors, as float16 (the dtype the prompt path multiplies in): one launch."""
        if self.s.ndim != 1 or self.n % 32:
            return self.fwd(x.astype(mx.float32)).astype(mx.float16)
        if not hasattr(self, "_many"):
            a2 = 1
            while self.A // a2 // 32 > 16 and self.A // a2 >= 64:  # at most 16 values a lane
                a2 *= 2
            C, RR = self.p * a2, max(1, self.A // a2 // 32)
            if self.A // a2 < 32:
                a2, C, RR = self.A // 32 if self.A >= 32 else 1, self.n // 32, 1
            Q = np.kron(_sylvester(self.n // 32 // RR // self.p) if C > self.p else np.ones((1, 1)), np.array(self.Pm_fwd).reshape(self.p, self.p))
            assert Q.shape[0] == C and 32 * RR * C == self.n and C <= 32, (self.n, C, RR)
            self._many = (C, RR, mx.array(Q.astype(np.float32).reshape(-1)))
        C, RR, Q = self._many
        shp = x.shape
        T = int(np.prod(shp[:-1]))
        return _many(inputs=[x.reshape(T, self.n), self.s, Q], template=[("T", x.dtype), ("BUF", mx.float32 if self.n <= 7936 else mx.float16),
                     ("N", self.n), ("C", C), ("RR", RR)], grid=(32 * C * T, 1, 1), threadgroup=(32 * C, 1, 1),
                     output_shapes=[(T, self.n)], output_dtypes=[mx.float16])[0].reshape(shp)

    def fwd(self, x):
        return rot_apply(x, self.s, self.Pm_fwd, self.A, self.p, False) if self._kernel(x) else self._f(x, self.s, self.P_fwd)

    def bwd(self, y):
        return rot_apply(y, self.s, self.Pm_bwd, self.A, self.p, True) if self._kernel(y) else self._b(y, self.s, self.P_bwd)


# ----------------------------------------------------------------------------- packing

def pack_codes(Q: np.ndarray, bits: int) -> np.ndarray:
    """(m, n) codes < 2^bits → (m, n/32·bits) uint32: each run of 32 codes is a little-endian bit stream of `bits` words."""
    m, n = Q.shape
    assert n % 32 == 0 and Q.max() < 2 ** bits
    q = Q.astype(np.uint64).reshape(m, n // 32, 32)
    out = np.zeros((m, n // 32, bits), dtype=np.uint32)
    for k in range(32):
        bit = k * bits
        w, off = divmod(bit, 32)
        v = q[:, :, k] << np.uint64(off)
        out[:, :, w] |= (v & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        if off + bits > 32:
            out[:, :, w + 1] |= (v >> np.uint64(32)).astype(np.uint32)
    return out.reshape(m, -1)


# ----------------------------------------------------------------------------- kernels

TRELLIS_HEADER = """
// The state of entry k of run g, whose words start at gw. Nothing comes before a row's first run: its path starts from state 0.
inline uint lamq_tr_state(const device uint* gw, uint bits, uint g, uint k) {
    const uint bit = (31 - k) * bits, wi = bit >> 5, off = bit & 31u;
    uint s = gw[wi] >> off;
    if (off + 12 > 32 && (wi + 1 < bits || g > 0)) s |= gw[wi + 1] << (32 - off);
    return s & 4095u;
}
"""

_TR_GEMV_SRC = """
    const uint m = dims[0], n = dims[1], wpr = dims[2], G = n / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane / LPR, sl = lane % LPR;
    const uint row = (threadgroup_position_in_grid.x * ROWS + simdgroup_index_in_threadgroup) * (32 / LPR) + sub;
    const bool live = row < m;
    const device uint* rc = codes + (ulong)(live ? row : 0) * wpr;
    const uint ngroups = live ? G : 0;
    float acc = 0.0f;
    for (uint g = sl; g < ngroups; g += LPR) {
        const device uint* gw = rc + (G - 1 - g) * BITS;
        const device T* xg = x + g * 32;
        #pragma unroll
        for (uint k = 0; k < 32; k++) acc += values[lamq_tr_state(gw, BITS, g, k)] * (float)xg[k];
    }
    for (uint d = LPR / 2; d > 0; d /= 2) acc += simd_shuffle_xor(acc, d);
    if (live && sl == 0) y[row] = (T)(acc * (float)scales[row]);
"""

_TR_DEQ_SRC = """
    const uint n = dims[1], G = n / 32;
    const uint row = thread_position_in_grid.y, k = thread_position_in_grid.x & 31u, g = thread_position_in_grid.x >> 5;
    if (row >= dims[0] || g >= G) return;
    const device uint* gw = codes + (ulong)row * dims[2] + (G - 1 - g) * BITS;
    w[(ulong)row * n + g * 32 + k] = (T)(values[lamq_tr_state(gw, BITS, g, k)] * (float)scales[row]);
"""

_GEMV_SRC = """
    // LPR lanes per row: a simdgroup handles 32/LPR rows, so short rows still keep every lane busy
    const uint m = dims[0], n = dims[1], wpr = dims[2];
    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane / LPR, sl = lane % LPR;
    const uint row = (threadgroup_position_in_grid.x * ROWS + simdgroup_index_in_threadgroup) * (32 / LPR) + sub;
    threadgroup float lv[256];
    for (uint i = thread_position_in_threadgroup.x; i < (1u << BITS); i += ROWS * 32) lv[i] = levels[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool live = row < m;
    const device uint* rc = codes + (ulong)(live ? row : 0) * wpr;
    const uint ngroups = live ? n / 32 : 0;
    constexpr uint MASK = (1u << BITS) - 1u;
    float acc = 0.0f;
    for (uint g = sl; g < ngroups; g += LPR) {
        uint w[BITS + 1];
        #pragma unroll
        for (uint i = 0; i < BITS; i++) w[i] = rc[g * BITS + i];
        w[BITS] = 0;
        const device T* xg = x + g * 32;
        #pragma unroll
        for (uint k = 0; k < 32; k++) {
            const uint bit = k * BITS, wi = bit >> 5, off = bit & 31u;
            uint c = w[wi] >> off;
            if (off + BITS > 32) c |= w[wi + 1] << (32 - off);
            acc += lv[c & MASK] * (float)xg[k];
        }
    }
    for (uint s = LPR / 2; s > 0; s /= 2) acc += simd_shuffle_xor(acc, s);
    if (live && sl == 0) y[row] = (T)(acc * (float)scales[row]);
"""

_DEQ_SRC = """
    // one code per thread: lane k of a simdgroup decodes code k of a run of 32, so a simdgroup writes 32 adjacent values
    const uint n = dims[1], wpr = dims[2];
    const uint row = thread_position_in_grid.y, k = thread_position_in_grid.x & 31u, g = thread_position_in_grid.x >> 5;
    if (row >= dims[0] || g >= n / 32) return;
    const device uint* gw = codes + (ulong)row * wpr + g * BITS;
    const uint bit = k * BITS, wi = bit >> 5, off = bit & 31u;
    uint c = gw[wi] >> off;
    if (off + BITS > 32) c |= gw[wi + 1] << (32 - off);
    w[(ulong)row * n + g * 32 + k] = (T)(levels[c & ((1u << BITS) - 1u)] * (float)scales[row]);
"""

_ROT_GEMV_SRC = """
    // y = S ⊙ (L[Q] · R x) for one vector in one launch: every threadgroup rotates x into threadgroup memory (signs, a
    // Sylvester transform along the 2^a axis, the p×p factor), then runs the matrix–vector product on it. The rotation
    // is repeated per threadgroup; it is a few thousand operations against a launch saved per matrix.
    // Every second butterfly stage is halved, which keeps the values in range when BUF is half; NORM undoes it.
    const uint m = dims[0], wpr = dims[1];
    const float NORM = rsqrt((float)N) * (float)(1u << (STAGES / 2));
    const uint tid = thread_position_in_threadgroup.x;
    constexpr uint NT = ROWS * 32;
    threadgroup BUF xs[N];
    threadgroup float lv[1u << BITS];
    for (uint i = tid; i < N; i += NT) xs[i] = (BUF)((float)x[i] * s[i]);
    for (uint i = tid; i < (1u << BITS); i += NT) lv[i] = levels[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint stage = 0;
    for (uint h = 1; h < A; h <<= 1, stage++) {
        const float f = (stage & 1u) ? 0.5f : 1.0f;
        for (uint j = tid; j < (A / 2) * P; j += NT) {
            const uint pair = j / P, c = j - pair * P;
            const uint r = ((pair >> stage) << (stage + 1)) + (pair & (h - 1));
            const uint i0 = r * P + c, i1 = i0 + h * P;
            const float u = xs[i0], w = xs[i1];
            xs[i0] = (BUF)((u + w) * f); xs[i1] = (BUF)((u - w) * f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (P > 1) {
        for (uint r = tid; r < A; r += NT) {
            float v[P];
            for (uint k = 0; k < P; k++) v[k] = xs[r * P + k];
            for (uint c = 0; c < P; c++) {
                float a = 0.0f;
                for (uint k = 0; k < P; k++) a += v[k] * Pm[c * P + k];
                xs[r * P + c] = (BUF)a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // the threadgroups (fewer than the row blocks: each rotates once) take row blocks in turn
    const uint lane = thread_index_in_simdgroup;
    const uint sub = lane / LPR, sl = lane % LPR;
    constexpr uint MASK = (1u << BITS) - 1u;
    constexpr uint RPT = ROWS * (32 / LPR);
    for (uint rb = threadgroup_position_in_grid.x; rb * RPT < m; rb += threadgroups_per_grid.x) {
        const uint row = (rb * ROWS + simdgroup_index_in_threadgroup) * (32 / LPR) + sub;
        const bool live = row < m;
        const device uint* rc = codes + (ulong)(live ? row : 0) * wpr;
        const uint ngroups = live ? N / 32 : 0;
        float acc = 0.0f;
        for (uint g = sl; g < ngroups; g += LPR) {
            uint w[BITS + 1];
            #pragma unroll
            for (uint i = 0; i < BITS; i++) w[i] = rc[g * BITS + i];
            w[BITS] = 0;
            const threadgroup BUF* xg = xs + g * 32;
            #pragma unroll
            for (uint k = 0; k < 32; k++) {
                const uint bit = k * BITS, wi = bit >> 5, off = bit & 31u;
                uint c = w[wi] >> off;
                if (off + BITS > 32) c |= w[wi + 1] << (32 - off);
                acc += lv[c & MASK] * (float)xg[k];
            }
        }
        for (uint d = LPR / 2; d > 0; d /= 2) acc += simd_shuffle_xor(acc, d);
        if (live && sl == 0) y[row] = (T)(acc * (float)scales[row] * NORM);
    }
"""

_tr_gemv = mx.fast.metal_kernel(name="lamq_tr_gemv", input_names=["x", "codes", "values", "scales", "dims"], output_names=["y"], source=_TR_GEMV_SRC, header=TRELLIS_HEADER)
_tr_deq = mx.fast.metal_kernel(name="lamq_tr_dequant", input_names=["codes", "values", "scales", "dims"], output_names=["w"], source=_TR_DEQ_SRC, header=TRELLIS_HEADER)
_gemv = mx.fast.metal_kernel(name="lamq_gemv", input_names=["x", "codes", "levels", "scales", "dims"], output_names=["y"], source=_GEMV_SRC)
_rot_gemv = mx.fast.metal_kernel(name="lamq_rot_gemv", input_names=["x", "s", "Pm", "codes", "levels", "scales", "dims"], output_names=["y"],
                                 source=_ROT_GEMV_SRC)
MAX_TG, MAX_TG_NARROW = 80, 160  # threadgroups per fused launch (each rotates once): wide inputs, inputs up to 2048
TG_FLOATS = 7168  # threadgroup memory is 32 KB: the rotated vector (as float up to this size, as half above) and the levels
_deq = mx.fast.metal_kernel(name="lamq_dequant", input_names=["codes", "levels", "scales", "dims"], output_names=["w"], source=_DEQ_SRC)
ROWS = 8


def lanes_per_row(n):
    """Enough runs of 32 per lane to hide memory latency: 32 lanes on a long row, fewer on a short one."""
    groups = n // 32
    return 32 if groups >= 96 else 16 if groups >= 24 else 8 if groups >= 8 else 4


def lm_gemv(x, codes, levels, scales, m, n, bits, dtype=mx.float16, lpr=None):
    """y' = S ⊙ (L[Q] x) for one input vector x (already rotated)."""
    dims = mx.array([m, n, codes.shape[1]], dtype=mx.uint32)
    lpr = lpr or lanes_per_row(n)
    rows_per_tg = ROWS * (32 // lpr)
    return _gemv(inputs=[x, codes, levels, scales, dims], template=[("T", dtype), ("BITS", bits), ("ROWS", ROWS), ("LPR", lpr)],
                 grid=((m + rows_per_tg - 1) // rows_per_tg * ROWS * 32, 1, 1), threadgroup=(ROWS * 32, 1, 1),
                 output_shapes=[(m,)], output_dtypes=[dtype])[0]


def tr_gemv(x, codes, values, scales, m, n, k, dtype=mx.float16):
    """y' = S ⊙ (V[state] x) for one input vector x (already rotated)."""
    lpr = lanes_per_row(n)
    rows_per_tg = ROWS * (32 // lpr)
    return _tr_gemv(inputs=[x, codes, values, scales, mx.array([m, n, codes.shape[1]], dtype=mx.uint32)],
                    template=[("T", dtype), ("BITS", k), ("ROWS", ROWS), ("LPR", lpr)],
                    grid=((m + rows_per_tg - 1) // rows_per_tg * ROWS * 32, 1, 1), threadgroup=(ROWS * 32, 1, 1),
                    output_shapes=[(m,)], output_dtypes=[dtype])[0]


def tr_dequant(codes, values, scales, m, n, k, dtype=mx.float16):
    """S ⊙ V[state] as a dense (m, n) matrix (for inputs of more than one token)."""
    return _tr_deq(inputs=[codes, values, scales, mx.array([m, n, codes.shape[1]], dtype=mx.uint32)],
                   template=[("T", dtype), ("BITS", k)],
                   grid=(n, m, 1), threadgroup=(min(n, 256), 1, 1), output_shapes=[(m, n)], output_dtypes=[dtype])[0]


class RotGemv:
    """y = S ⊙ (L[Q] · R x) for one vector, rotation included, as one kernel launch. Everything but x is fixed at build."""

    def __init__(self, rot: Rot, codes, levels, scales, m, bits, dtype=mx.bfloat16):
        n = rot.n
        assert n % 32 == 0 and rot.A * rot.p == n
        lpr = lanes_per_row(n)
        stages = int(math.log2(rot.A))
        self.args = [rot.s, rot.Pm_fwd, codes, levels, scales, mx.array([m, codes.shape[1]], dtype=mx.uint32)]
        self.template = [("T", dtype), ("BUF", mx.float32 if n <= TG_FLOATS else mx.float16), ("BITS", bits), ("ROWS", ROWS),
                         ("LPR", lpr), ("A", rot.A), ("P", rot.p), ("N", n), ("STAGES", stages)]
        rows_per_tg = ROWS * (32 // lpr)
        self.grid = (min((m + rows_per_tg - 1) // rows_per_tg, MAX_TG_NARROW if n <= 2048 else MAX_TG) * ROWS * 32, 1, 1)
        self.m, self.dtype = m, dtype

    def __call__(self, x):
        return _rot_gemv(inputs=[x, *self.args], template=self.template, grid=self.grid, threadgroup=(ROWS * 32, 1, 1),
                         output_shapes=[(self.m,)], output_dtypes=[self.dtype])[0]


_GEMM_SRC = """
    // Y (TP × m) = X (TP × n) · (S ⊙ L[Q])ᵀ for a few tokens (TP = 8·TT ≤ 32), read from the codes once.
    // Each simdgroup takes 8 rows. Per run of 32 columns: the threadgroup stages X's 32 columns; each lane decodes 8 of
    // its simdgroup's 8 × 32 weights (scaled) into threadgroup memory; then 8 × 8 matrix multiply-accumulates.
    const uint m = dims[0], n = dims[1], wpr = dims[2];
    const uint tid = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    constexpr uint NT = SG * 32, TP = 8 * TT;
    threadgroup float lv[256];
    threadgroup float xs[TP * 32];
    threadgroup float wt[SG * 8 * 32];
    for (uint i = tid; i < (1u << BITS); i += NT) lv[i] = levels[i];
    const uint row0 = (threadgroup_position_in_grid.x * SG + sg) * 8;
    const uint r = lane >> 2, j0 = (lane & 3u) * 8;
    const uint row = row0 + r;
    const bool live = row < m;
    const device uint* rc = codes + (ulong)(live ? row : 0) * wpr;
    const float sc = live ? (float)scales[row] : 0.0f;
    threadgroup float* w_ = wt + sg * 256;
    simdgroup_float8x8 acc[TT];
    for (uint tt = 0; tt < TT; tt++) acc[tt] = simdgroup_float8x8(0.0f);
    const uint ks = threadgroup_position_in_grid.y, nks = threadgroups_per_grid.y, ng = n / 32;  // split over the columns: part ks of nks
    for (uint g = ks * ng / nks; g < (ks + 1) * ng / nks; g++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < TP * 32; i += NT) xs[i] = (float)x[(ulong)(i >> 5) * n + g * 32 + (i & 31u)];
        const device uint* gw = rc + g * BITS;
        #pragma unroll
        for (uint q = 0; q < 8; q++) {
            const uint bit = (j0 + q) * BITS, wi = bit >> 5, off = bit & 31u;
            uint c = gw[wi] >> off;
            if (off + BITS > 32) c |= gw[wi + 1] << (32 - off);
            w_[r * 32 + j0 + q] = lv[c & ((1u << BITS) - 1u)] * sc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint k8 = 0; k8 < 4; k8++) {
            simdgroup_float8x8 b;
            simdgroup_load(b, w_ + k8 * 8, 32, ulong2(0, 0), true);
            for (uint tt = 0; tt < TT; tt++) {
                simdgroup_float8x8 a;
                simdgroup_load(a, xs + tt * 8 * 32 + k8 * 8, 32);
                simdgroup_multiply_accumulate(acc[tt], a, b, acc[tt]);
            }
        }
    }
    if (row0 < m)
        for (uint tt = 0; tt < TT; tt++) simdgroup_store(acc[tt], y + ((ulong)ks * TP + tt * 8) * m + row0, m);
"""
_gemm = mx.fast.metal_kernel(name="lamq_gemm_few", input_names=["x", "codes", "levels", "scales", "dims"], output_names=["y"], source=_GEMM_SRC)
_TR_GEMM_SRC = _GEMM_SRC.replace("""    threadgroup float lv[256];
""", "").replace("""    for (uint i = tid; i < (1u << BITS); i += NT) lv[i] = levels[i];
""", "").replace("const device uint* gw = rc + g * BITS;", "const device uint* gw = rc + (ng - 1 - g) * BITS;").replace("""            const uint bit = (j0 + q) * BITS, wi = bit >> 5, off = bit & 31u;
            uint c = gw[wi] >> off;
            if (off + BITS > 32) c |= gw[wi + 1] << (32 - off);
            w_[r * 32 + j0 + q] = lv[c & ((1u << BITS) - 1u)] * sc;
""", """            w_[r * 32 + j0 + q] = values[lamq_tr_state(gw, BITS, g, j0 + q)] * sc;
""").replace("L[Q]", "V[state]")
_tr_gemm = mx.fast.metal_kernel(name="lamq_tr_gemm_few", input_names=["x", "codes", "values", "scales", "dims"], output_names=["y"], source=_TR_GEMM_SRC, header=TRELLIS_HEADER)
GEMM_TG = 160  # threadgroups the few-token product aims for (by splitting the columns)
GEMM_MAX_T = 32  # prompts up to this many tokens multiply from the codes; longer ones expand the codes to 16 bits first


def lm_gemm_few(x, codes, levels, scales, m, n, bits, sg=4):
    """y = x · (S ⊙ L[Q])ᵀ for x of at most GEMM_MAX_T rows (already rotated), as float32."""
    T = x.shape[0]
    tt = -(-T // 8)
    if tt * 8 != T:
        x = mx.concatenate([x, mx.zeros((tt * 8 - T, n), x.dtype)])
    dims = mx.array([m, n, codes.shape[1]], dtype=mx.uint32)
    tgs = -(-m // (8 * sg))
    ks = max(1, min(n // 32 // 8, -(-GEMM_TG // tgs)))  # enough threadgroups to fill the GPU, at least 8 runs of 32 columns each
    y = _gemm(inputs=[x, codes, levels, scales, dims], template=[("BITS", bits), ("TT", tt), ("SG", sg)],
              grid=(tgs * sg * 32, ks, 1), threadgroup=(sg * 32, 1, 1), output_shapes=[(ks, tt * 8, m)], output_dtypes=[mx.float32])[0]
    return (y.sum(0) if ks > 1 else y[0])[:T]


def tr_gemm_few(x, codes, values, scales, m, n, k, sg=4):
    """y = x · (S ⊙ V[state])ᵀ for x of at most GEMM_MAX_T rows (already rotated), as float32."""
    T = x.shape[0]
    tt = -(-T // 8)
    if tt * 8 != T:
        x = mx.concatenate([x, mx.zeros((tt * 8 - T, n), x.dtype)])
    tgs = -(-m // (8 * sg))
    ks = max(1, min(n // 32 // 8, -(-GEMM_TG // tgs)))
    y = _tr_gemm(inputs=[x, codes, values, scales, mx.array([m, n, codes.shape[1]], dtype=mx.uint32)], template=[("BITS", k), ("TT", tt), ("SG", sg)],
                 grid=(tgs * sg * 32, ks, 1), threadgroup=(sg * 32, 1, 1), output_shapes=[(ks, tt * 8, m)], output_dtypes=[mx.float32])[0]
    return (y.sum(0) if ks > 1 else y[0])[:T]


def lm_dequant(codes, levels, scales, m, n, bits, dtype=mx.float16):
    """S ⊙ L[Q] as a dense (m, n) matrix (for inputs of more than one token)."""
    dims = mx.array([m, n, codes.shape[1]], dtype=mx.uint32)
    return _deq(inputs=[codes, levels, scales, dims], template=[("T", dtype), ("BITS", bits)],
                grid=(n, m, 1), threadgroup=(min(n, 256), 1, 1), output_shapes=[(m, n)], output_dtypes=[dtype])[0]


class LMLinear:
    """y = R_outᵀ · S ⊙ L[Q] · R_in x."""

    def __init__(self, codes, scales, bits, s_in, s_out, levels, dtype=mx.float16):
        self.m, self.n = len(scales), len(s_in)
        self.bits, self.dtype = bits, dtype
        self.codes = mx.array(codes)
        self.scales = mx.array(np.asarray(scales, dtype=np.float32)).astype(dtype)
        self.levels = mx.array(np.asarray(levels, dtype=np.float32))
        self.r_in, self.r_out = Rot(s_in), Rot(s_out)

    def __call__(self, x):
        xr = self.r_in.fwd(x.astype(mx.float32)).astype(self.dtype)
        lead = x.shape[:-1]
        if math.prod(lead) == 1:
            y = lm_gemv(xr.reshape(-1), self.codes, self.levels, self.scales, self.m, self.n, self.bits, self.dtype).reshape(*lead, self.m)
        else:
            W = lm_dequant(self.codes, self.levels, self.scales, self.m, self.n, self.bits, self.dtype)
            y = xr @ W.T
        return self.r_out.bwd(y.astype(mx.float32)).astype(x.dtype)

    def nbytes(self):
        return self.codes.nbytes + self.scales.nbytes + (self.n + self.m) // 8
