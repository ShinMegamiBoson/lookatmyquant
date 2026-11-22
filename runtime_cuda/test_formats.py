"""Checks the PyTorch decoders against the float64 numpy references in `runtime_mlx/inkling/kernels.py::_selftest`.

The numpy references below are copied from that self-test (the spec the author's Metal kernels are checked against),
so a mismatch here means our decoder disagrees with the author's format, not just with itself.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_cuda.inkling.formats import (  # noqa: E402
    HASH_MULT_U32,
    decode_lm,
    decode_trellis,
    effective_weight,
    hadamard_factors_np,
    rht_in,
    rht_out,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _t(a: np.ndarray) -> torch.Tensor:
    if a.dtype == np.uint32:
        a = a.view(np.int32)
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)


@pytest.mark.parametrize("n", [2304, 1152, 3072, 1024, 384])
def test_hadamard_is_hadamard_and_rht_matches_kron(n):
    A, B = hadamard_factors_np(n)
    H = np.kron(A, B)
    assert H.shape == (n, n)
    assert np.allclose(H @ H.T, n * np.eye(n))
    rng = np.random.default_rng(n)
    s = rng.choice([-1.0, 1.0], size=n)
    x = rng.standard_normal((3, n))
    R = H * s[None, :] / math.sqrt(n)
    want_in = x @ R.T
    got_in = rht_in(_t(x.astype(np.float32)), _t(s.astype(np.float32))).double().cpu().numpy()
    assert np.abs(got_in - want_in).max() < 1e-4
    got_out = rht_out(_t(x.astype(np.float32)), _t(s.astype(np.float32))).double().cpu().numpy()
    assert np.abs(got_out - x @ R).max() < 1e-4  # rht_out(y) = y @ R_out with R_out = H diag(s) / sqrt(n)
    back = rht_out(rht_in(_t(x.astype(np.float32)), _t(s.astype(np.float32))), _t(s.astype(np.float32)))
    assert np.abs(back.double().cpu().numpy() - x).max() < 1e-4


def _numpy_trellis_ref(x, codes, scales, lut, eids, m, n, k, g, K, hmult):
    """Verbatim logic of the `_selftest` trellis reference."""
    wpr = (n * k) // 32
    shift = np.uint32(32 - int(round(math.log2(K))))
    y_ref = np.zeros((len(eids), m), dtype=np.float64)
    for slot, eid in enumerate(eids):
        acc = np.zeros(m, dtype=np.float64)
        c_e = codes[eid]
        for j in range(0, n, g):
            last = j + g - 1
            bp = (n - 1 - last) * k
            wi, off = bp // 32, bp % 32
            w0 = c_e[:, wi].astype(np.uint64)
            w1 = c_e[:, wi + 1].astype(np.uint64) if wi + 1 < wpr else np.zeros(m, dtype=np.uint64)
            s = ((w0 >> np.uint64(off)) | (w1 << np.uint64(32 - off) if off > 0 else np.uint64(0))) & np.uint64(4095)
            idx = ((s.astype(np.uint32) * np.uint32(hmult)) >> shift).astype(np.int64) if hmult else s.astype(np.int64)
            if g == 1:
                acc += lut[idx].astype(np.float64) * float(x[slot, j])
            else:
                acc += lut[idx].astype(np.float64) @ x[slot, j : j + g].astype(np.float64)
        y_ref[slot] = acc * scales[eid].astype(np.float64)
    return y_ref


@pytest.mark.parametrize(
    "fmt,k,m,n,g,K,hmult",
    [
        ("ht2", 2, 256, 2304, 4, 512, HASH_MULT_U32),
        ("ht3", 3, 256, 1152, 2, 1024, HASH_MULT_U32),
        ("ht1", 1, 64, 2304, 4, 512, HASH_MULT_U32),
        ("rtr1", 1, 250, 2304, 1, 4096, 0),
        ("rtr2", 2, 100, 2304, 1, 4096, 0),
        ("rtr3", 3, 100, 2304, 1, 4096, 0),
        ("rtr4", 4, 128, 2304, 1, 4096, 0),
        ("rtr4", 4, 96, 1152, 1, 4096, 0),
    ],
)
def test_trellis_matches_selftest_reference(fmt, k, m, n, g, K, hmult):
    rng = np.random.default_rng(7)
    E = 3
    codes = rng.integers(0, 2**32, size=(E, m, (n * k) // 32), dtype=np.uint32)
    scales = rng.uniform(0.01, 0.05, size=(E, m)).astype(np.float16)
    lut = rng.standard_normal((K, g) if g > 1 else (K,)).astype(np.float16)
    eids = [0, 2]
    x = rng.standard_normal((len(eids), n)).astype(np.float32)
    want = _numpy_trellis_ref(x, codes, scales, lut, eids, m, n, k, g, K, hmult)
    W = decode_trellis(_t(codes), _t(scales), _t(lut), n, fmt)  # [E, m, n]
    got = torch.stack([_t(x)[i].double() @ W[e].double().T for i, e in enumerate(eids)]).cpu().numpy()
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert err < 1e-6, err


@pytest.mark.parametrize("k,m,n", [(6, 250, 2304), (8, 300, 2304), (6, 64, 3072)])
def test_lm_matches_selftest_reference(k, m, n):
    rng = np.random.default_rng(3)
    levels = np.sort(rng.standard_normal(1 << k)).astype(np.float32)
    idx = rng.integers(0, 1 << k, size=(m, n), dtype=np.uint8)
    if k == 8:
        packed = idx.reshape(-1)
    else:
        a = idx.reshape(-1, 4).astype(np.uint32)
        v = a[:, 0] | (a[:, 1] << 6) | (a[:, 2] << 12) | (a[:, 3] << 18)
        packed = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=-1).astype(np.uint8).reshape(-1)
    scales = rng.uniform(0.01, 0.05, size=(m,)).astype(np.float16)
    want = levels[idx].astype(np.float64) * scales.astype(np.float64)[:, None]
    got = decode_lm(_t(packed), _t(scales), _t(levels), m, n, k).double().cpu().numpy()
    assert np.abs(got - want).max() < 1e-6


def test_effective_weight_matches_linear_path():
    rng = np.random.default_rng(11)
    m, n = 384, 2304
    Q = _t(rng.standard_normal((m, n)).astype(np.float32))
    si = _t(rng.choice([-1.0, 1.0], size=n).astype(np.float32))
    so = _t(rng.choice([-1.0, 1.0], size=m).astype(np.float32))
    x = _t(rng.standard_normal((5, n)).astype(np.float32))
    y_path = rht_out(rht_in(x, si) @ Q.T, so)
    y_dense = x @ effective_weight(Q, si, so).T
    assert (y_path - y_dense).abs().max().item() < 1e-3 * y_path.abs().max().item()


def test_nvfp4_matches_upstream_reference():
    """runtime_mlx/test_inkling.py::test_nvfp4_unified_moe_all_modes builds its float64 reference this way."""
    from runtime_cuda.inkling.formats import decode_nvfp4, e4m3_table_np, E2M1

    rng = np.random.default_rng(4242)
    E, m, n = 3, 64, 1152
    c = rng.integers(0, 256, size=(E, m, n // 2), dtype=np.uint8)
    s = rng.integers(0, 256, size=(E, m, n // 16), dtype=np.uint8)
    o = rng.uniform(4e-5, 5e-4, size=(E,)).astype(np.float32)
    e2m1, e4m3 = np.array(E2M1, dtype=np.float64), e4m3_table_np().astype(np.float64)
    want = np.empty((E, m, n))
    for e in range(E):
        vals = np.stack([e2m1[c[e] & 0xF], e2m1[c[e] >> 4]], axis=-1).reshape(m, n // 16, 16)
        want[e] = (vals * e4m3[s[e]][..., None]).reshape(m, n) * float(o[e])
    got = decode_nvfp4(_t(c), _t(s), _t(o)).double().cpu().numpy()
    assert np.abs(got - want).max() <= 1e-6 * np.abs(want).max()
    # spot-check the E4M3 table against torch's own float8 decoding (NaN codes excepted)
    if hasattr(torch, "float8_e4m3fn"):
        b = torch.arange(256, dtype=torch.uint8)
        ref = b.view(torch.float8_e4m3fn).float().numpy()
        ok = ~np.isnan(ref)
        assert np.array_equal(ref[ok], e4m3_table_np()[ok])
