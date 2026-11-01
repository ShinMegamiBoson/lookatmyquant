"""Tests for nested codes: the compander's sets nest, reading the top bits is the coarser code, one-rate weights reduce
the multirate sweep to that rate's LDLQ, and the rates are priced. Run: python tests/test_nested.py"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from opgraph import codes as C
from opgraph import nested as N
from opgraph.frame import Environment
from opgraph.syntax import sweep
from opgraph.verbs import snap

torch.manual_seed(0)
ok = lambda a, b, tol=1e-10: torch.allclose(a, b, atol=tol, rtol=0)


def test_lloyd_max_matches_the_known_sets():
    assert ok(N.lloyd_max(2), torch.tensor(C.GAUSS2.levels, dtype=torch.float64), 1e-4)
    inf = torch.tensor([math.inf], dtype=torch.float64)
    for b, mse in ((1, 0.3634), (2, 0.1175), (3, 0.03455), (4, 0.009497)):  # Max (1960)
        lv = N.lloyd_max(b)
        bounds = torch.cat([-inf, (lv[1:] + lv[:-1]) / 2, inf])
        assert abs(N.gaussian_mse(bounds, lv) - mse) < 2e-4, b


def test_compander_cells_nest():
    for alpha in (1 / 3, 0.4, 0.5):
        fine, coarse = N.compander_bounds(4, alpha), N.compander_bounds(2, alpha)
        assert ok(fine[::4], coarse), "every coarse boundary is a fine boundary"
        lv4, lv2 = N.compander_levels(4, alpha), N.compander_levels(2, alpha)
        p = torch.special.ndtr(fine[1:] * 1.0) - torch.special.ndtr(fine[:-1] * 1.0)
        merged = (lv4 * p).view(4, 4).sum(1) / p.view(4, 4).sum(1)
        assert ok(merged, lv2, 1e-9), "a coarse level is the centroid of its four fine cells"


def test_read_top_is_the_coarse_code():
    alpha = 0.4
    x = torch.randn(200, 64, dtype=torch.float64)
    W = torch.randn(32, 64, dtype=torch.float64)
    H = x.T @ x
    out = N.nested_words([W], H, [5], 4, [2, 3, 4], alpha, [0.2, 0.3, 1.0])
    top = out[4][0]
    for b in (2, 3):
        r = N.read_top(top, "W", 4, b, alpha)
        assert ok(r.matrix(), out[b][0].matrix(), 1e-9), b
        assert abs(r.rate_bits() - out[b][0].rate_bits()) < 1e-9
    side = top.rate_bits() - 32 * 64 * 4
    assert abs(out[2][0].rate_bits() - (32 * 64 * 2 + side)) < 1e-9, "a read is priced at its own bits"


def test_one_rate_weights_are_that_rates_ldlq():
    alpha = 0.4
    x = torch.randn(300, 48, dtype=torch.float64)
    H = x.T @ x
    X = torch.randn(16, 48, dtype=torch.float64)
    for r, b in enumerate((2, 3, 4)):
        w = [0.0, 0.0, 0.0]
        w[r] = 1.0
        if b == 2:  # ties among a 2-bit cell's children are broken by the finer rates, which do not change the 2-bit read
            w = [1.0, 1e-6, 1e-12]
        idx = N.multirate_sweep(X, H, N.nested_tables(4, [2, 3, 4], alpha), torch.tensor(w, dtype=torch.float64))
        got = N.nested_tables(4, [2, 3, 4], alpha)[r][idx]
        lv = N.compander_levels(b, alpha)
        ref = sweep(X, Environment(H), lambda j, col, T: snap(col, lv.tolist()))
        assert ok(got, snap(ref, lv.tolist()), 1e-9), b


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
