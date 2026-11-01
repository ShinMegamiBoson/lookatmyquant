"""Tests for the extended grammar: types, Tie, finite sets and codebooks, structured masks, generalized Refit, Insert,
Randomize, Condition and Replace. Run: python tests/test_grammar.py"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from opgraph import data, methods as ops, model as M
from opgraph import verbs as V
from opgraph.frame import INT, Environment, ModelNet, mask, matrix_net, restrict, split_index, split_scale, ZZ
from opgraph.syntax import Program, Step, alternate, choose, sweep

torch.manual_seed(0)
ok = lambda a, b, tol=1e-10: torch.allclose(a, b, atol=tol, rtol=0)


def test_value_sets():
    assert len(V.FP4_E2M1) == 15 and max(V.FP4_E2M1) == 6.0
    assert len(V.FP8_E4M3) == 253 and max(V.FP8_E4M3) == 448.0
    nf4 = V.SET(V.NF4_LEVELS)
    assert nf4.bits == 4.0
    net = V.restrict_set(matrix_net(torch.tensor([[0.03, 0.05, -0.99, 0.6]], dtype=torch.float64)), "W", nf4)
    assert net.cores["W"].data.tolist() == [[0.0, 0.07958029955625534, -1.0, 0.5626170039176941]]
    assert net.rate_bits() == 4 * 4


def test_codebook_restrict_then_split_is_exact():
    W = torch.randn(32, 64, dtype=torch.float64)
    net = split_index(matrix_net(W), "i", (("g", 32), ("w", 2)))
    C = V.kmeans(net.cores["W"].data.reshape(-1, 2), 16)
    net = V.restrict_codebook(net, "W", "w", C)
    before = net.matrix().clone()
    assert net.rate_bits() == 32 * 32 * 4  # one 4-bit index per 2-vector
    net = V.split_codebook(net, "W", "w", C)
    assert ok(net.matrix(), before), "split_codebook must be exact once vectors are codewords"
    assert net.cores["W.A"].domain.kind == "onehot" and net.cores["W.A"].data.dtype == torch.long
    assert net.rate_bits() == 32 * 32 * 4 + 16 * 2 * 16
    try:
        V.split_codebook(split_index(matrix_net(W), "i", (("g", 32), ("w", 2))), "W", "w", C)
        raise AssertionError("split_codebook on non-codeword vectors must be refused")
    except ValueError:
        pass


def test_structured_masks():
    W = torch.randn(8, 16, dtype=torch.float64)
    keep = V.nm_keep(W.abs(), 2, 4)
    net = mask(matrix_net(W), "W", keep, {"kind": "n:m", "n": 2, "m": 4, "label": "i"})
    assert math.isclose(net.rate_bits(), 64 * 16 + 32 * math.log2(6))
    try:
        mask(matrix_net(W), "W", W.abs() > 0.5, {"kind": "n:m", "n": 2, "m": 4, "label": "i"})
        raise AssertionError("a mask that is not 2:4 must be refused")
    except ValueError:
        pass
    ch = torch.zeros(8, 16, dtype=torch.bool); ch[:, :6] = True
    net = mask(matrix_net(W), "W", ch, {"kind": "channel", "label": "i"})
    assert net.rate_bits() == 8 * 6 * 16 + 16


def test_tie_slices():
    W = torch.randn(6, 4, dtype=torch.float64)
    net = V.tie_slices(matrix_net(W), "W", "o", [[0, 1], [2, 3, 4]])
    X = net.cores["W"].data
    assert ok(X[0], W[:2].mean(0)) and ok(X[3], W[2:5].mean(0)) and ok(X[5], W[5])
    assert net.rate_bits() == 3 * 4 * 16
    from opgraph.frame import merge_index
    heads = merge_index(V.tie_slices(split_index(matrix_net(torch.randn(320, 8, dtype=torch.float64)), "o", (("h", 5), ("e", 64))),
                                     "W", "h", [[0, 1], [2, 3]]), ("h", "e"), "o")
    assert heads.matrix().shape == (320, 8) and heads.rate_bits() == 3 * 64 * 8 * 16


def test_generic_sweep_reproduces_gptq():
    W = torch.randn(16, 128, dtype=torch.float64)
    Xin = torch.randn(128, 1000, dtype=torch.float64)
    H = Xin @ Xin.T
    fixed, S, Z = ops.gptq_sweep(W, H, 3, 64)
    grid = {}

    def on_block(j, T):
        grid[j // 64] = ops.gptq_grid(T[:, j : j + 64], 7)

    def project(j, col, T):
        s, z = grid[j // 64]
        return s * (torch.clamp(torch.round(col / s) + z, 0, 7) - z)

    before = sweep(W, Environment(H), project, block=64, on_block=on_block)
    assert ok(before, fixed, 1e-9), "the generic sweep must reproduce GPTQ's"


def test_control_and_programs():
    assert alternate(1, [lambda x: x + 1, lambda x: x * 2], 2) == 10
    assert choose([3, 1, 2], lambda x: x) == (1, 1)
    p = Program("toy", [Step("Split", "i", "index"), Step("Restrict", "W", "grid", exact=False)], run=lambda m: m)
    assert str(p) == "Split[index] @ i → Restrict[grid] @ W"


def test_environment_error():
    D = torch.randn(4, 6, dtype=torch.float64)
    X = torch.randn(6, 50, dtype=torch.float64); H = X @ X.T
    Gm = torch.randn(4, 4, dtype=torch.float64); Gm = Gm @ Gm.T
    e = Environment(H, damp=0.0, G=Gm)
    assert math.isclose(e.error(D), torch.trace(Gm @ D @ e.H @ D.T).item(), rel_tol=1e-12)


def test_stochastic_rounding_is_unbiased():
    x = torch.full((1, 20000), 0.3, dtype=torch.float64)
    net = V.restrict_stochastic(matrix_net(x), "W", ZZ, seed=1)
    assert abs(net.cores["W"].data.mean().item() - 0.3) < 0.01


def test_model_level_verbs():
    m = M.load()
    ids = data.evaluation(2, 128).to(M.DEVICE)
    base = M.logits(m, ids)
    net = ModelNet.from_model(m)
    # Insert, exact: a runtime rotation on two activation bonds, absorbed by the cores that read them.
    rot = V.rotate_at_runtime(net, 5, "down_in", V.block_hadamard(2560, 512))
    rot = V.rotate_at_runtime(rot, 9, "attn_in", V.random_orthogonal(960).float())
    err = (M.logits(rot.compile(), ids) - base).abs().max().item()
    assert err < 2e-2, f"runtime rotations must preserve logits (max diff {err})"
    # Insert, lossy: 8-bit activations change logits a little.
    q = V.insert(net, 3, "attn_in", V.act_quant(8))
    d = (M.logits(q.compile(), ids) - base).abs().max().item()
    assert 0 < d < 5, d
    # Condition: zeroing half the MLP activations per token; a window as long as the sequence changes nothing.
    x = torch.randn(4, 2560)
    assert abs((V.topk_activations(0.5)(x) == 0).float().mean().item() - 0.5) < 0.01
    w = V.kv_window(net, 4, 4, 128)
    assert (M.logits(w.compile(), ids) - base).abs().max().item() < 1e-3
    # Replace: skipping layer 7 equals removing it.
    sk = V.replace_layer(net, 7)
    m2 = m.copy(); del m2.layers[7]
    assert (M.logits(sk.compile(), ids) - M.logits(m2, ids)).abs().max().item() < 1e-4
    # Tie whole cores: compile shares one tensor and storage counts it once.
    t = V.tie_cores(net, [(10, "up"), (11, "up")])
    cm = t.compile()
    assert cm.layers[10].up is cm.layers[11].up
    assert t.rate_bits() < net.rate_bits()
    # Refit, generalized: a few Adam steps on the scales of one quantized core reduce the KL from the original.
    qz = net.copy()
    qz.linears[(6, "gate")] = ops.rtn(m.layers[6].gate, 3, 64)
    calib = data.calibration(4, 128)
    before = M.evaluate(qz.compile(), calib, m)["kl"]
    qz = V.refit_gradient(qz, [((6, "gate"), "W.S")], m, calib, steps=30, lr=1e-3, bs=4)
    after = M.evaluate(qz.compile(), calib, m)["kl"]
    assert after < before, (before, after)  # the objective it optimizes, on the data it sees


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
