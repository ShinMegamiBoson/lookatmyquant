"""The fast Qwen3.5 path against float64 NumPy products of independently unpacked codes (compiler_artifact's own
unpackers). Synthetic codes only; no model is loaded.

    python runtime_mlx/test_fast_q35.py
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime_mlx"))

import mlx.core as mx
import numpy as np

import compiler_artifact as C
from runtime_mlx import fast_q35 as F


def trellis_group(rng, rows, n, k, out_rows, inp=0):
    sym = rng.integers(0, 1 << k, (rows, n), dtype=np.uint32)
    states = np.zeros((rows, n), np.uint32)
    st = np.zeros(rows, np.uint32)
    for j in range(n):
        st = ((st << k) | sym[:, j]) & 4095
        states[:, j] = st
    table = rng.standard_normal(4096).astype(np.float32)
    scales = rng.uniform(0.01, 0.03, rows).astype(np.float32)
    g = {"codes": C.pack_trellis(states, k), "scales": scales, "rows": out_rows, "bits": k, "table": table, "input": inp}
    assert (C.unpack_trellis(g["codes"], k, n) == states).all()
    return g, table[states].astype(np.float64) * scales[:, None]


def scalar_group(rng, rows, n, k, out_rows, inp=0):
    sym = rng.integers(0, 1 << k, (rows, n), dtype=np.uint32)
    levels = np.sort(rng.standard_normal(1 << k)).astype(np.float32)
    scales = rng.uniform(0.01, 0.03, rows).astype(np.float32)
    g = {"codes": C.pack_symbols(sym, k), "scales": scales, "rows": out_rows, "bits": k, "table": levels, "input": inp, "scalar": True}
    return g, levels[sym].astype(np.float64) * scales[:, None]


class ProductTest(unittest.TestCase):
    def run_case(self, n, kind, tol):
        rng = np.random.default_rng(n)
        spec = [("tr", 1, 72), ("tr", 2, 64), ("tr", 3, 40), ("tr", 4, 8), ("sc", 5, 24), ("sc", 8, 16), ("tr", 2, 3)]
        size = sum(r for _, _, r in spec)
        perm = rng.permutation(size).astype(np.int32)
        xs = rng.standard_normal((3, n)).astype(np.float32)
        groups, want, lo, full = [], np.zeros(size), 0, np.zeros((size, n))
        for i, (fam, k, rows) in enumerate(spec):
            out_rows = perm[lo : lo + rows]; lo += rows
            g, W = (trellis_group if fam == "tr" else scalar_group)(rng, rows, n, k, out_rows, inp=i % 3)
            groups.append(g)
            want[out_rows] = W @ xs[i % 3].astype(np.float64)
            full[out_rows] = W
        prod = F.Product(n, groups, size, kind)
        got = np.array(prod(mx.array(xs)))
        err = np.abs(got - want).max() / np.abs(want).max()
        self.assertLess(err, tol, (n, kind, err))
        dense = np.array(prod.dense(mx.float32)).astype(np.float64)   # every row at its output position
        err = np.abs(dense - full).max() / np.abs(full).max()
        self.assertLess(err, tol, (n, kind, "dense", err))

    def test_products(self):
        for n in (512, 1024, 2560, 4096):
            self.run_case(n, "float", 2e-6)
            self.run_case(n, "half", 1e-3)


class RotationTest(unittest.TestCase):
    def test_single_vector(self):
        """Rotate1 against the dense matrix H diag(s) in float64 (lamq's own factors), at the model's three widths."""
        from runtime_mlx import lamq
        rng = np.random.default_rng(5)
        for n in (2560, 4096, 9216):
            signs = rng.choice([-1.0, 1.0], n)
            rot = lamq.Rot(signs.astype(np.float32))
            P = np.array(rot.Pm_fwd, dtype=np.float64).reshape(rot.p, rot.p)
            H = np.kron(lamq._sylvester(rot.A), P) / np.sqrt(n)
            x, u = rng.standard_normal(n), rng.standard_normal(n)
            r1 = F.Rotate1(rot)
            got = np.array(r1(mx.array(x.astype(np.float32))))
            want = H @ (signs * x.astype(np.float32).astype(np.float64))
            self.assertLess(np.abs(got - want).max() / np.abs(want).max(), 3e-6, n)
            act = x.astype(np.float32).astype(np.float64) / (1 + np.exp(-x.astype(np.float32).astype(np.float64))) * u.astype(np.float32)
            got = np.array(r1(mx.array(x.astype(np.float32)), mx.array(u.astype(np.float32))))
            want = H @ (signs * act)
            self.assertLess(np.abs(got - want).max() / np.abs(want).max(), 3e-6, (n, "gated"))
            X = rng.standard_normal((5, n)).astype(np.float32)      # several vectors in one launch
            got = np.array(r1(mx.array(X)))
            want = (signs * X.astype(np.float64)) @ H.T
            self.assertLess(np.abs(got - want).max() / np.abs(want).max(), 3e-6, (n, "many"))


if __name__ == "__main__":
    unittest.main()
