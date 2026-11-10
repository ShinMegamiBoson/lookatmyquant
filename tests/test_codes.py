"""Tests for the state-of-the-art additions: MXFP4 and NVFP4, Hadamard incoherence, block LDLQ, vector codebooks and
the trellis. Run: python tests/test_codes.py"""

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from opgraph import codes as C
from opgraph.frame import Environment, matrix_net
from opgraph.verbs import FP8_E4M3, snap

torch.manual_seed(0)
ok = lambda a, b, tol=1e-10: torch.allclose(a, b, atol=tol, rtol=0)


def test_hadamard_orders():
    for n in (64, 320, 960, 2560):
        H = C.hadamard(n)
        assert ok(H @ H.T, torch.eye(n, dtype=H.dtype), 1e-10), n
        assert ok(H.abs(), torch.full_like(H, 1 / math.sqrt(n)), 1e-12), n
    R = C.rht(960, seed=3)
    assert ok(R @ R.T, torch.eye(960, dtype=R.dtype), 1e-10)


def test_incoherence_is_exact_and_priced():
    W = torch.randn(320, 960, dtype=torch.float64)
    net = C.incoherence(matrix_net(W), C.rht(320, 1), C.rht(960, 2))
    assert ok(net.matrix(), W, 1e-9)
    assert net.cores["W"].labels == ("u", "v")
    assert net.rate_bits() == 320 * 960 * 16 + 320 + 960  # the middle core, plus one sign per row of each transform


def test_mx_word():
    W = torch.zeros(1, 32, dtype=torch.float64)
    W[0, :4] = torch.tensor([5.0, 1.0, 0.3, -2.6])
    net = C.fp4_word(W, "mx")
    assert net.cores["W.S"].data.item() == 1.0  # 2^(floor(log2 5) − 2)
    assert net.matrix()[0, :4].tolist() == [4.0, 1.0, 0.5, -3.0]  # 5 is halfway between 4 and 6 and goes to 4, as round-to-even does
    assert net.rate_bits() == 32 * 4 + 8
    W[0, 0] = 7.5  # same scale; 7.5 saturates to the largest E2M1 value
    assert C.fp4_word(W, "mx").matrix()[0, 0].item() == 6.0


def test_nv_word():
    W = torch.randn(16, 64, dtype=torch.float64)
    net = C.fp4_word(W, "nv")
    S = net.cores["W.S"].data
    assert torch.equal(S, snap(S, FP8_E4M3)) and net.cores["W.G"].domain.bits == 32
    assert net.rate_bits() == 16 * 64 * 4 + 16 * 4 * 8 + 32
    assert ((net.matrix() - W).norm() / W.norm()).item() < 0.12


def test_fp4_sweep_without_correlations_is_round_to_nearest():
    W = torch.randn(32, 64, dtype=torch.float64)
    for fmt in ("mx", "nv"):
        assert torch.equal(C.fp4_sweep(W, torch.eye(64, dtype=torch.float64), fmt).matrix(), C.fp4_word(W, fmt).matrix()), fmt


def test_block_refit():
    X = torch.randn(400, 12, dtype=torch.float64)
    H = X.T @ X
    W = torch.randn(5, 12, dtype=torch.float64)
    env = Environment(H, damp=0.0)
    T, E = W.clone(), 0.1 * torch.randn(5, 4, dtype=torch.float64)
    env.refit_block_after(T, 0, 4, E)
    assert ok(T[:, 4:], env.refit_exact(W, [0, 1, 2, 3], W[:, :4] - E)[:, 4:], 1e-9), "block LDLQ is the least-squares refit"
    a, b = W.clone(), W.clone()
    env.refit_after(a, 0, E[:, 0])
    env.refit_block_after(b, 0, 1, E[:, :1])
    assert ok(a[:, 1:], b[:, 1:], 1e-12)


def test_ldlq_word_with_codebook():
    W = torch.randn(320, 960, dtype=torch.float64)
    X = torch.randn(2000, 960, dtype=torch.float64)
    cb = C.gaussian_codebook(256, 4)
    net = C.ldlq_word(W, X.T @ X, C.VectorCode(cb), seed=5)
    assert net.rate_bits() == 320 * 960 / 4 * 8 + 320 * 16 + 320 + 960
    assert ((net.matrix() - W).norm() / W.norm()).item() < 0.4


def test_joint_sweep_equals_separate_sweeps():
    X = torch.randn(3000, 960, dtype=torch.float64)
    H = X.T @ X
    Ws = [torch.randn(320, 960, dtype=torch.float64), torch.randn(640, 960, dtype=torch.float64)]
    for code in (C.ScalarCode(C.GAUSS2), C.VectorCode(C.gaussian_codebook(256, 4))):
        together = C.ldlq_words(Ws, H, code, [3, 9])
        R_in = C.rht(960, 4)
        for W, s, net in zip(Ws, [3, 9], together):
            alone = C.ldlq_words([W], H, code, [s])[0] if s == 3 else None
            if alone is not None:
                assert ok(alone.matrix(), net.matrix(), 1e-9)
            assert ((net.matrix() - W).norm() / W.norm()).item() < 0.45


def test_viterbi_is_optimal():
    L, k, T = 4, 1, 7
    v = torch.randn(1 << L, dtype=torch.float64)
    X = torch.randn(5, T, dtype=torch.float64)
    st = C.viterbi(X, v, L, k)
    assert C.is_trellis_path(st, L, k)
    got = (X - v[st]).square().sum(1)
    for r in range(len(X)):
        best = math.inf
        for s0 in range(1 << L):
            for bits in itertools.product(range(1 << k), repeat=T - 1):
                s, cost = s0, (X[r, 0] - v[s0]).item() ** 2
                for t, b in enumerate(bits, 1):
                    s = ((s << k) | b) & ((1 << L) - 1)
                    cost += (X[r, t] - v[s]).item() ** 2
                best = min(best, cost)
        assert abs(got[r].item() - best) < 1e-4, (got[r].item(), best)


def test_trellis_restrict_and_price():
    L, k, T = 8, 2, 64
    v = C.trellis_values(L, k, T, refine=2)
    W = torch.randn(128, 8, dtype=torch.float64)
    from opgraph.frame import split_index
    net = split_index(matrix_net(W), "o", (("s", 2), ("t", T)))
    net = C.restrict_trellis(net, "W", "t", C.TRELLIS(v, L, k))
    seqs = W.reshape(2, T, 8).permute(0, 2, 1).reshape(-1, T)
    st = C.viterbi(seqs, v, L, k)
    assert C.is_trellis_path(st, L, k)
    assert ok(net.cores["W"].data.permute(0, 2, 1).reshape(-1, T), v[st])
    assert net.rate_bits() == 128 * 8 * k + 16 * (L - k)


def test_codes_beat_scalar_on_gaussians():
    X = torch.randn(512, 320, dtype=torch.float64)
    scalar = (X - snap(X, C.GAUSS2.levels)).square().mean().item()
    assert abs(scalar - 0.1175) < 0.005, scalar  # the Lloyd–Max distortion at 2 bits
    cb = C.gaussian_codebook(256, 4)
    V = X.reshape(-1, 4)
    vq = (V - C.VectorCode(cb).project(V)).square().mean().item()
    v = C.trellis_values(8, 2, 320, refine=2)
    tcq = (X - v[C.viterbi(X, v, 8, 2)]).square().mean().item()
    assert vq < scalar - 0.005 and tcq < scalar - 0.01, (scalar, vq, tcq)
    print(f"    2-bit distortion on a unit Gaussian: scalar {scalar:.4f}, 4-D codebook {vq:.4f}, 256-state trellis {tcq:.4f}")


def test_trellis_ema_fit():
    """Decaying means over fresh batches leave less error on held-out Gaussians than Lloyd steps, in a file of their own."""
    import tempfile
    cache, default, C.CACHE = C.CACHE, C.TRELLIS_FIT, Path(tempfile.mkdtemp())
    try:
        L, k, T = 8, 2, 64
        X = torch.randn(1024, T, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        err, files = {}, {"lloyd": f"trellis_L{L}_k{k}_T{T}_s0.pt", "ema": f"trellis_L{L}_k{k}_T{T}_s0_ema.pt"}
        for fit in files:
            v = C.trellis_values(L, k, T, refine=6, fit=fit)
            err[fit] = (X - v[C.viterbi(X, v, L, k)]).square().mean().item()
        assert err["ema"] < 0.99 * err["lloyd"], err
        assert sorted(p.name for p in C.CACHE.iterdir()) == sorted(files.values())
        for fit, name in files.items():  # what --trellis-fit or a plan sets
            C.TRELLIS_FIT = fit
            assert torch.equal(C.trellis_values(L, k, T), torch.load(C.CACHE / name))
        print(f"    2-bit distortion, 256-state trellis: Lloyd steps {err['lloyd']:.4f}, decaying means {err['ema']:.4f}")
    finally:
        C.CACHE, C.TRELLIS_FIT = cache, default


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
