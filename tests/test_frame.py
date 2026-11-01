"""Checks that the frame's exact operators are exact, its lossy ones do what they claim, and its whole-model gauges
leave the network's function unchanged. Run: python tests/test_frame.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from opgraph import data, model as M
from opgraph.frame import (split_sum, INT, Environment, ModelNet, canonical_gauge, check_group, contract, fold_gain, gauge, gauge_diag,
                           gauge_segment, insert_identity, matrix_net, restrict, segments, split_index, split_offset, split_scale,
                           split_svd, truncate, truncate_segment, untie)

torch.manual_seed(0)
ok = lambda a, b, tol=1e-10: torch.allclose(a, b, atol=tol, rtol=0)


def test_exact_local_operators():
    W = torch.randn(12, 16, dtype=torch.float64)
    net = matrix_net(W)
    net = split_index(net, "i", (("g", 4), ("w", 4)))
    S = W.reshape(12, 4, 4).abs().amax(-1) + 0.1
    net = split_scale(net, "W", S, ("o", "g"), "S")
    net = split_offset(net, "W", torch.round(torch.randn(12, 4)).double(), ("o", "g"), "Z")
    assert ok(net.matrix(), W), "split_index / split_scale / split_offset must be exact"
    net = insert_identity(matrix_net(W), "i", "b")
    G = torch.randn(16, 16, dtype=torch.float64) + 4 * torch.eye(16, dtype=torch.float64)
    net = gauge(net, "b", G, left="W", right="I[b]")
    assert ok(net.matrix(), W, 1e-9), "gauge must be exact"
    net = split_svd(net, "W", ("o",), ("b",), "r")
    assert ok(net.matrix(), W, 1e-9), "split_svd must be exact at full rank"
    net = contract(net, "W.V", "I[b]", "V")
    assert ok(net.matrix(), W, 1e-9), "contract must be exact"


def test_truncation_is_eckart_young():
    W = torch.randn(20, 30, dtype=torch.float64)
    k = 7
    net = truncate(split_svd(matrix_net(W), "W", ("o",), ("i",), "r"), "r", k)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    assert ok(net.matrix(), (U[:, :k] * S[:k]) @ Vh[:k], 1e-9)


def test_restrict_projects_onto_grid():
    net = restrict(matrix_net(torch.tensor([[-3.4, 0.49, 2.5, 99.0]], dtype=torch.float64)), "W", INT(3))
    assert net.cores["W"].data.tolist() == [[0.0, 0.0, 2.0, 7.0]] and net.cores["W"].domain == INT(3)


def test_sequential_refit_equals_least_squares():
    o, n = 6, 10
    W = torch.randn(o, n, dtype=torch.float64)
    X = torch.randn(n, 200, dtype=torch.float64)
    env = Environment(X @ X.T, damp=0.0)
    T, fixed_vals = W.clone(), []
    for j in range(n):
        q = torch.round(T[:, j] * 2) / 2  # any projection
        fixed_vals.append(q)
        env.refit_after(T, j, T[:, j] - q)
        exact = env.refit_exact(W, list(range(j + 1)), torch.stack(fixed_vals, 1))
        assert ok(T[:, j + 1 :], exact[:, j + 1 :], 1e-8), f"refit mismatch after fixing column {j}"


def test_groups_are_enforced():
    Q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
    check_group(Q, "O", "test")
    for G, group in ((torch.randn(8, 8, dtype=torch.float64), "O"), (Q, "diag"), (Q, "none")):
        try:
            check_group(G, group, "test")
        except ValueError:
            continue
        raise AssertionError(f"group {group} accepted a gauge outside it")
    net = restrict(insert_identity(matrix_net(torch.randn(4, 4, dtype=torch.float64)), "i", "b"), "W", INT(4))
    try:
        gauge(net, "b", torch.eye(4, dtype=torch.float64) * 2, left="W", right="I[b]")
    except ValueError:
        return
    raise AssertionError("gauge on an integer core must be refused")


def test_model_level_gauges_preserve_function():
    m = M.load()
    ids = data.evaluation(2, 128).to(M.DEVICE)
    base = M.logits(m, ids)
    net = ModelNet.from_model(m)
    assert (M.logits(net.compile(), ids) - base).abs().max() == 0, "compile must reproduce the model"
    # Diagonal gauges on the bonds AWQ uses fold into gains and rows: function unchanged.
    for i in (0, 17):
        for bond in ("attn_in", "mlp_in", "down_in"):
            net = gauge_diag(net, (i, bond), torch.rand(net.linears[(i, "down" if bond == "down_in" else "q")].dim("i"), device="mps") + 0.5)
    assert (M.logits(net.compile(), ids) - base).abs().max() < 1e-3, "diagonal gauges must preserve logits"
    # A rotation of a residual segment is refused until the norm that reads it has its gain folded away.
    Q, _ = torch.linalg.qr(torch.randn(960, 960, dtype=torch.float64))
    try:
        gauge_segment(net, "A5", Q.float())
        raise AssertionError("rotating a segment read by a gained RMSNorm must be refused")
    except ValueError:
        pass
    net = untie(net)
    for norm in [(i, p) for i in range(M.LAYERS) for p in ("attn", "mlp")] + ["final"]:
        net = fold_gain(net, norm)
    for seg in segments():
        net = gauge_segment(net, seg, torch.linalg.qr(torch.randn(960, 960, dtype=torch.float64))[0].float())
    err = (M.logits(net.compile(), ids) - base).abs().max().item()
    assert err < 2e-2, f"random rotations of every segment must preserve logits (max diff {err})"
    # Truncating every segment to full width in the canonical gauge is still exact.
    net2 = truncate_segment(net, "A3", 960)
    assert (M.logits(net2.compile(), ids) - M.logits(net.compile(), ids)).abs().max() == 0


def test_canonical_gauge():
    X = torch.randn(500, 6, dtype=torch.float64) @ torch.randn(6, 6, dtype=torch.float64)
    Q = canonical_gauge(X.T @ X)
    check_group(Q, "O", "test")
    d = torch.diag(Q.T @ (X.T @ X) @ Q)
    assert torch.all(d[:-1] >= d[1:]), "canonical gauge must order the bond by decreasing environment energy"


def test_split_sum_and_lowrank_plus_quant():
    W = torch.randn(12, 16, dtype=torch.float64)
    P = torch.randn(12, 16, dtype=torch.float64)
    net = split_sum(matrix_net(W), "W", P, "B")
    assert ok(net.matrix(), W), "split_sum must be exact"
    assert len(net.terms) == 2 and torch.equal(net.cores["B"].data, P)
    from opgraph import methods as ops
    W = torch.randn(64, 128, dtype=torch.float64)
    X = torch.randn(128, 2000, dtype=torch.float64)
    H = X @ X.T
    net = ops.lowrank_plus_quant(W, H, 3, 64, rank=8)
    assert {"W", "W.S", "W.Z", "B.U", "B.V"} <= set(net.cores), sorted(net.cores)
    assert net.cores["B.U"].data.shape[1] == 8 and net.cores["W"].domain == INT(3)
    err = lambda Wh: torch.trace((Wh - W) @ H @ (Wh - W).T).item()
    plain = ops.gptq(W, H, 3, 64).matrix()
    assert err(net.matrix()) < err(plain), "a rank-8 branch must lower the output error of 3-bit GPTQ"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
