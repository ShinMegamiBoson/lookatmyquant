"""Tests for the new grammar elements: pinned rotation, Pin(x), low-rank scale cores, capacity checks and mixed
environments. Run: python tests/test_elements.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from opgraph import elements as E
from opgraph import methods as ops
from opgraph.frame import covariance
from opgraph.syntax import default_environment

torch.manual_seed(0)
ok = lambda a, b, tol=1e-10: torch.allclose(a, b, atol=tol, rtol=0)


def test_pinned_orthogonal():
    pins = [3, 17, 500]
    Q = E.pinned_orthogonal(960, pins, seed=2)
    assert ok(Q @ Q.T, torch.eye(960, dtype=Q.dtype), 1e-10)
    for p in pins:
        e = torch.zeros(960, dtype=Q.dtype)
        e[p] = 1
        assert ok(Q[p], e) and ok(Q[:, p], e), "a pinned channel must map to itself"


def test_lowrank_scale_core_is_exact_and_priced():
    W = torch.randn(64, 256, dtype=torch.float64) * torch.linspace(0.5, 2, 64)[:, None]
    for rank, gs in ((1, 16), (4, 16), (None, 64)):
        net = E.gptq_static_word(W, None, 3, gs, rank)
        S = E.lowrank_scales(W, 3, gs, rank)
        codes = torch.clamp(torch.round(W.reshape(64, -1, gs) / S[..., None]), -4, 3)
        assert ok(net.matrix(), (codes * S[..., None]).reshape(64, 256), 1e-9), "the factored scale core must be exact"
        g = 256 // gs
        side = 64 * g * 16 if rank is None else (64 + g) * rank * 16
        assert abs(net.rate_bits() - (64 * 256 * 3 + side)) < 1e-6, (rank, net.rate_bits())


def test_static_sweep_without_correlations_is_rounding():
    W = torch.randn(32, 128, dtype=torch.float64)
    assert ok(E.gptq_static_word(W, torch.eye(128, dtype=torch.float64), 4, 32).matrix(), E.gptq_static_word(W, None, 4, 32).matrix())


def test_pin_tokens_and_mixtures():
    x = torch.randn(4, 16, 8)
    H = covariance(x)
    Hp = E.pin_tokens((0,), weight=3.0)([x])
    assert ok(Hp, H + 3 * covariance(x[:, :1]), 1e-9)
    y = 10 * torch.randn(4, 16, 8)
    M = default_environment([x, y])
    assert abs(torch.trace(M) - (torch.trace(covariance(x)) + torch.trace(covariance(y))) / 2) < 1e-6 * torch.trace(M)


def test_capacity_check():
    rec = []
    check = E.capacity_check(min_ratio=50, log=None, record=rec)
    assert abs(check(2560**2, 32768 * 2560, "down") - 12.8) < 1e-9
    assert rec[0]["what"] == "down"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
