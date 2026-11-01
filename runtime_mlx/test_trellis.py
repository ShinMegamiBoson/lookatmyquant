"""The trellis kernels against the matrices the quantizer built (test_trellis_gen.py): one token, a few, many, a fused stage; a table's lookup and head.

    python test_trellis.py <dir>
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

import fastpath as fp
import model as lm

path = Path(sys.argv[1])
W = mx.load(str(path / "model.safetensors"))
meta = json.loads((path / "lamq.json").read_text())
trellis = {int(k[len("trellis.k"):]): v.astype(mx.float32) for k, v in W.items() if k.startswith("trellis.")}
rng = np.random.default_rng(0)
TOL = 5e-3  # half a bfloat16 step at the largest output is 3.9e-3


def rel(a, b):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))


for name, m in meta["matrices"].items():
    M = lm.LMLinear(W[f"{name}.codes"], W[f"{name}.scales"], W[f"{name}.s_in"], W.get(f"{name}.s_out"), m["bits"], trellis[m["bits"]])
    R = W[f"{name}.ref"].astype(mx.float32)
    res = {}
    for T in (1, 5, 16, 40):
        x = mx.array(rng.standard_normal((T, M.n)).astype(np.float32)).astype(mx.bfloat16)
        res[f"T={T}"] = rel(M(x), x.astype(mx.float32) @ R.T)
    x = mx.array(rng.standard_normal((M.n,)).astype(np.float32)).astype(mx.bfloat16)
    (y,) = fp.Stage("plain", [M])(x)
    res["stage"] = rel(y, x.astype(mx.float32) @ R.T)
    g, u = [mx.array(rng.standard_normal((M.n,)).astype(np.float32)).astype(mx.bfloat16) for _ in "gu"]
    (y,) = fp.Stage("geglu", [M])(g, u)
    import mlx.nn as nn
    res["stage geglu"] = rel(y, (nn.gelu_approx(g) * u).astype(mx.float32) @ R.T)
    print(name, m["bits"], M.m, M.n, {k: round(v, 5) for k, v in res.items()})
    assert max(res.values()) < TOL, res
    if M.r_out is None and M.n == 1536:
        twin = lm.LMLinear(W[f"{name}.codes"], W[f"{name}.scales"], W[f"{name}.s_in"], None, m["bits"], trellis[m["bits"]])
        x = mx.array(rng.standard_normal((M.n,)).astype(np.float32)).astype(mx.bfloat16)
        y0, y1 = fp.Stage("plain", [M, twin])(x)
        assert max(rel(y0, x.astype(mx.float32) @ R.T), rel(y1, x.astype(mx.float32) @ R.T)) < TOL

for name, t in meta["tables"].items():
    T = lm.LMTable(W, name, t, trellis)
    R = W[f"{name}.ref"].astype(mx.float32)
    ids = mx.array(rng.integers(0, t["shape"][0], (1, 200)).astype(np.int32))
    res = {"lookup": rel(T(ids)[0], R[ids[0]])}
    assert res["lookup"] < TOL
    if T.slices == 1:
        for n in (1, 5, 40):
            h = mx.array(rng.standard_normal((1, n, T.d)).astype(np.float32))
            res[f"head T={n}"] = rel(T.as_linear(h)[0], h[0] @ R.T)
        T._head = fp.head(T)
        h = mx.array(rng.standard_normal((1, 1, T.d)).astype(np.float32)).astype(mx.bfloat16)
        res["head stage"] = rel(T.as_linear(h)[0], h[0].astype(mx.float32) @ R.T)
    print(name, {k: round(v, 5) for k, v in res.items()})
    assert max(res.values()) < TOL, res
print("ok")
