"""Checks the runtime against the quantizer's reference math: rotations, packing, the matrix–vector kernel, speed."""
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import lamq

REF = Path(__file__).parent / "ref"
rng = np.random.default_rng(0)

# 1. rotations: Rot(s).fwd(x) must equal R x with the quantizer's R = rht(n, 12345)
for n in (256, 512, 1536, 2048, 4096, 6144, 12288):
    R = np.load(REF / f"rht_{n}_12345.npy").astype(np.float64)
    H = np.load(REF / f"hadamard_{n}.npy").astype(np.float64)
    s = np.rint((H.T @ R).diagonal())  # R = H diag(s), so Hᵀ R = diag(s)
    rot = lamq.Rot(s)
    x = rng.standard_normal((3, n)).astype(np.float32)
    f = np.array(rot.fwd(mx.array(x)))
    b = np.array(rot.bwd(mx.array(x)))
    ef, eb = np.abs(f - x @ R.T).max(), np.abs(b - x @ R).max()
    print(f"rot n={n:5d}  fwd err {ef:.2e}  bwd err {eb:.2e}  signs ±1: {set(np.unique(s)) <= {-1.0, 1.0}}")
    assert ef < 1e-4 and eb < 1e-4

# 2. packing + kernel: y = R_outᵀ diag(S) L[Q] R_in x
for bits in (2, 3, 4, 5, 6, 8):
    m, n = 1536, 6144
    L = np.load(REF / f"lloyd_max_{bits}.npy")
    Q = rng.integers(0, 2 ** bits, size=(m, n))
    S = rng.uniform(0.01, 0.05, size=m).astype(np.float32)
    s_in, s_out = rng.choice([-1.0, 1.0], n), rng.choice([-1.0, 1.0], m)
    lin = lamq.LMLinear(lamq.pack_codes(Q, bits), S, bits, s_in, s_out, L)
    W = np.array(lin.r_out.bwd(mx.array(np.eye(m, dtype=np.float32)))).T  # R_outᵀ as a matrix (columns)
    Rin = np.array(lin.r_in.fwd(mx.array(np.eye(n, dtype=np.float32)))).T
    Wfull = W @ (S[:, None] * L[Q]) @ Rin
    x = rng.standard_normal((1, n)).astype(np.float32)
    y = np.array(lin(mx.array(x).astype(mx.float16)).astype(mx.float32))
    ref = x @ Wfull.T
    X = rng.standard_normal((5, n)).astype(np.float32)
    Y = np.array(lin(mx.array(X).astype(mx.float16)).astype(mx.float32))
    rel = np.abs(y - ref).max() / np.abs(ref).max()
    relB = np.abs(Y - X @ Wfull.T).max() / np.abs(X @ Wfull.T).max()
    print(f"lm{bits}: gemv rel err {rel:.2e}   batched rel err {relB:.2e}   codes {lin.codes.nbytes / (m * n) * 8:.2f} bits/weight")
    assert rel < 5e-3 and relB < 5e-3

# 3. speed of the decode kernel on the largest matrix shape (12,288 × 1,536) and a wide one (1,536 × 12,288)
for (m, n) in ((12288, 1536), (1536, 12288)):
    for bits in (4, 5, 6, 8):
        Q = rng.integers(0, 2 ** bits, size=(m, n))
        codes = mx.array(lamq.pack_codes(Q, bits)); L = mx.array(np.load(REF / f"lloyd_max_{bits}.npy"))
        S = mx.array(np.ones(m, np.float16)); x = mx.array(rng.standard_normal(n).astype(np.float16))
        for _ in range(5):
            mx.eval(lamq.lm_gemv(x, codes, L, S, m, n, bits))
        N = 200
        xs = [mx.array(rng.standard_normal(n).astype(np.float16)) for _ in range(N)]
        mx.eval(xs)
        t0 = time.perf_counter()
        ys = [lamq.lm_gemv(xi, codes, L, S, m, n, bits) for xi in xs]
        mx.eval(ys)  # every call is computed; MLX is lazy, so evaluating only the last would time one call
        dt = (time.perf_counter() - t0) / N
        print(f"gemv {m}x{n} lm{bits}: {dt * 1e6:7.1f} µs   {codes.nbytes / dt / 1e9:6.1f} GB/s of codes")
print("all checks passed")
