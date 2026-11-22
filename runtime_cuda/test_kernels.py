"""Triton fused decode+matmul against the PyTorch reference decoders (themselves checked against the author's numpy)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_cuda.inkling.formats import FMT_PARAMS, decode_lm, decode_trellis  # noqa: E402
from runtime_cuda.inkling.triton_kernels import group_by_expert, packed_matmul  # noqa: E402

DEV = torch.device("cuda")


def _rand_trellis(rng, E, m, n, fmt):
    k, g, K, _ = FMT_PARAMS[fmt]
    codes = torch.from_numpy(rng.integers(0, 2**32, size=(E, m, (n * k) // 32), dtype=np.uint32).view(np.int32)).to(DEV)
    scales = torch.from_numpy(rng.uniform(0.01, 0.05, size=(E, m)).astype(np.float16)).to(DEV)
    lut = torch.from_numpy(rng.standard_normal((K, g)).astype(np.float16) * 0.5).to(DEV)
    return codes, scales, lut


@pytest.mark.parametrize("fmt", list(FMT_PARAMS))
@pytest.mark.parametrize("m,n,T", [(256, 2304, 1), (200, 2304, 7), (2304, 1152, 33), (384, 2304, 16)])
def test_dense_trellis(fmt, m, n, T):
    rng = np.random.default_rng(1)
    codes, scales, lut = _rand_trellis(rng, 1, m, n, fmt)
    x = torch.randn(T, n, device=DEV)
    want = (x.double() @ decode_trellis(codes[0], scales[0], lut, n, fmt).double().T)
    got = packed_matmul(x, codes, scales, fmt, m, n, lut=lut).double()
    err = ((got - want).norm() / want.norm()).item()
    print(f"{fmt} {m}x{n} T={T}: rel err {err:.2e}")
    assert err < 5e-5, err


@pytest.mark.parametrize("k", [6, 8])
@pytest.mark.parametrize("m,n,T", [(384, 2304, 1), (1024, 2304, 20), (2304, 3072, 5)])
def test_dense_lm(k, m, n, T):
    rng = np.random.default_rng(2)
    idx = rng.integers(0, 1 << k, size=(m, n), dtype=np.uint8)
    if k == 8:
        packed = idx.reshape(-1)
    else:
        a = idx.reshape(-1, 4).astype(np.uint32)
        v = a[:, 0] | (a[:, 1] << 6) | (a[:, 2] << 12) | (a[:, 3] << 18)
        packed = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=-1).astype(np.uint8).reshape(-1)
    codes = torch.from_numpy(packed).to(DEV)
    scales = torch.from_numpy(rng.uniform(0.01, 0.05, size=(1, m)).astype(np.float16)).to(DEV)
    levels = torch.from_numpy(np.sort(rng.standard_normal(1 << k)).astype(np.float32)).to(DEV)
    x = torch.randn(T, n, device=DEV)
    want = x.double() @ decode_lm(codes, scales[0], levels, m, n, k).double().T
    got = packed_matmul(x, codes, scales, f"lm{k}", m, n, levels=levels).double()
    err = ((got - want).norm() / want.norm()).item()
    print(f"lm{k} {m}x{n} T={T}: rel err {err:.2e}")
    assert err < 5e-5, err


@pytest.mark.parametrize("fmt", ["ht2", "ht3", "rtr4"])
@pytest.mark.parametrize("T,topk,x_div", [(1, 8, 8), (5, 8, 8), (40, 8, 8), (40, 8, 1)])
def test_grouped_experts(fmt, T, topk, x_div):
    rng = np.random.default_rng(3)
    E, m, n = 12, 256, 1152
    codes, scales, lut = _rand_trellis(rng, E, m, n, fmt)
    n_pairs = T * topk
    expert = torch.from_numpy(rng.integers(0, E, size=n_pairs)).to(DEV)
    x = torch.randn(n_pairs // x_div, n, device=DEV)
    be, bp = group_by_expert(expert, E, 16)
    got = packed_matmul(x, codes, scales, fmt, m, n, lut=lut, block_expert=be, block_pairs=bp, n_pairs=n_pairs, x_div=x_div)
    W = decode_trellis(codes, scales, lut, n, fmt)  # [E, m, n]
    rows = torch.arange(n_pairs, device=DEV) // x_div
    want = torch.zeros(n_pairs, m, device=DEV)
    for e in range(E):
        sel = expert == e
        want[sel] = (x[rows[sel]].double() @ W[e].double().T).float()
    err = ((got - want).norm() / want.norm()).item()
    print(f"grouped {fmt} T={T} x_div={x_div}: rel err {err:.2e}")
    assert err < 5e-5, err


# ----------------------------------------------------------------------------- decode GEMV
from runtime_cuda.inkling.gemv import packed_gemv  # noqa: E402


@pytest.mark.parametrize("fmt", list(FMT_PARAMS))
@pytest.mark.parametrize("m,n", [(256, 2304), (200, 1152), (96, 3072)])
def test_gemv_trellis_experts(fmt, m, n):
    rng = np.random.default_rng(5)
    E = 6
    codes, scales, lut = _rand_trellis(rng, E, m, n, fmt)
    eids = torch.tensor([3, -1, 0, 5, 3], dtype=torch.int32, device=DEV)
    x = torch.randn(len(eids), n, device=DEV)
    got = packed_gemv(x, codes, scales, fmt, m, n, lut, expert_ids=eids)
    W = decode_trellis(codes, scales, lut, n, fmt).double()
    want = torch.stack([x[i].double() @ W[e].T if e >= 0 else torch.zeros(m, dtype=torch.float64, device=DEV) for i, e in enumerate(eids.tolist())])
    err = ((got.double() - want).norm() / want.norm()).item()
    print(f"gemv {fmt} {m}x{n}: rel err {err:.2e}")
    assert err < 1e-6, err
    assert got[1].abs().max().item() == 0.0


@pytest.mark.parametrize("k", [6, 8])
@pytest.mark.parametrize("m,n,T", [(384, 2304, 1), (1024, 2304, 3), (2304, 3072, 2)])
def test_gemv_lm(k, m, n, T):
    rng = np.random.default_rng(6)
    idx = rng.integers(0, 1 << k, size=(m, n), dtype=np.uint8)
    if k == 8:
        packed = idx.reshape(-1)
    else:
        a = idx.reshape(-1, 4).astype(np.uint32)
        v = a[:, 0] | (a[:, 1] << 6) | (a[:, 2] << 12) | (a[:, 3] << 18)
        packed = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=-1).astype(np.uint8).reshape(-1)
    codes = torch.from_numpy(packed).to(DEV)
    scales = torch.from_numpy(rng.uniform(0.01, 0.05, size=(1, m)).astype(np.float16)).to(DEV)
    levels = torch.from_numpy(np.sort(rng.standard_normal(1 << k)).astype(np.float32)).to(DEV)
    x = torch.randn(T, n, device=DEV)
    want = x.double() @ decode_lm(codes, scales[0], levels, m, n, k).double().T
    got = packed_gemv(x, codes, scales, f"lm{k}", m, n, levels).double()
    err = ((got - want).norm() / want.norm()).item()
    print(f"gemv lm{k} {m}x{n} T={T}: rel err {err:.2e}")
    assert err < 1e-6, err


def test_gemv_x_div():
    rng = np.random.default_rng(7)
    E, m, n = 10, 256, 2304
    codes, scales, lut = _rand_trellis(rng, E, m, n, "ht2")
    eids = torch.from_numpy(rng.integers(0, E, size=16).astype(np.int32)).to(DEV)
    x = torch.randn(2, n, device=DEV)  # 2 tokens x 8 experts, x_div = 8
    got = packed_gemv(x, codes, scales, "ht2", m, n, lut, expert_ids=eids, x_div=8)
    W = decode_trellis(codes, scales, lut, n, "ht2")
    want = torch.stack([x[i // 8] @ W[e].T for i, e in enumerate(eids.tolist())])
    assert ((got - want).norm() / want.norm()).item() < 1e-5


@pytest.mark.parametrize("groups", [1, 2, 18])
def test_moe_side_gemv_mixed_formats(groups):
    from runtime_cuda.inkling.gemv import MoESideGEMV

    rng = np.random.default_rng(9)
    group_rows, n, E = 128, 1152, 12
    m = group_rows * groups
    fmts = ["ht2", "ht3", "rtr4"]
    entries = E * groups
    owner = rng.integers(0, len(fmts), size=entries)
    slabs, weights = [], {}
    slab_of, loc_of = np.zeros(entries, np.int32), np.zeros(entries, np.int32)
    for j, fmt in enumerate(fmts):
        ids = np.nonzero(owner == j)[0]
        codes, scales, lut = _rand_trellis(rng, max(1, len(ids)), group_rows, n, fmt)
        slabs.append((fmt, codes, scales, lut))
        decoded = decode_trellis(codes, scales, lut, n, fmt)
        for loc, entry in enumerate(ids):
            slab_of[entry], loc_of[entry], weights[int(entry)] = j, loc, decoded[loc]
    W = [torch.cat([weights[e * groups + group] for group in range(groups)]) for e in range(E)]
    side = MoESideGEMV(slabs, torch.from_numpy(slab_of).to(DEV), torch.from_numpy(loc_of).to(DEV), m, n, groups=groups)
    eids = torch.from_numpy(rng.integers(0, E, size=16)).to(DEV)
    x = torch.randn(2, n, device=DEV)
    got = side(x, eids, x_div=8)
    want = torch.stack([x[i // 8] @ W[int(e)].T for i, e in enumerate(eids.tolist())])
    assert ((got - want).norm() / want.norm()).item() < 1e-5


def test_dense_stage_mixed():
    from runtime_cuda.inkling.gemv import DenseStageGEMV

    rng = np.random.default_rng(11)
    n = 2304
    specs = [("rtr4", 3072), ("lm6", 1024), ("lm8", 1024), ("ht2", 384), ("ht3", 3072)]
    mats, Ws = [], []
    for fmt, m in specs:
        if fmt.startswith("lm"):
            k = int(fmt[2:])
            idx = rng.integers(0, 1 << k, size=(m, n), dtype=np.uint8)
            if k == 8:
                packed = idx.reshape(-1)
            else:
                a = idx.reshape(-1, 4).astype(np.uint32)
                v = a[:, 0] | (a[:, 1] << 6) | (a[:, 2] << 12) | (a[:, 3] << 18)
                packed = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=-1).astype(np.uint8).reshape(-1)
            codes = torch.from_numpy(packed).to(DEV)
            scales = torch.from_numpy(rng.uniform(0.01, 0.05, size=(1, m)).astype(np.float16)).to(DEV)
            levels = torch.from_numpy(np.sort(rng.standard_normal(1 << k)).astype(np.float32)).to(DEV)
            mats.append((fmt, m, codes, scales, levels))
            Ws.append(decode_lm(codes, scales[0], levels, m, n, k))
        else:
            codes, scales, lut = _rand_trellis(rng, 1, m, n, fmt)
            mats.append((fmt, m, codes, scales, lut))
            Ws.append(decode_trellis(codes[0], scales[0], lut, n, fmt))
    x = torch.randn(5, n, device=DEV)
    st = DenseStageGEMV(mats, n, x_rows=[0, 1, 2, 3, 4])
    for i, y in enumerate(st.split(st(x))):
        want = Ws[i] @ x[i]
        assert ((y - want).norm() / want.norm()).item() < 1e-5, i


# ----------------------------------------------------------------------------- NVFP4 experts (upstream 5106923)

def _rand_nvfp4(rng, E, m, n):
    c = torch.from_numpy(rng.integers(0, 256, size=(E, m, n // 2), dtype=np.uint8)).to(DEV)
    s = torch.from_numpy(rng.integers(32, 88, size=(E, m, n // 16), dtype=np.uint8)).to(DEV)  # as upstream's test
    o = torch.from_numpy(rng.uniform(4e-5, 5e-4, size=(E,)).astype(np.float32)).to(DEV)
    return c, s, o


@pytest.mark.parametrize("m,n", [(2304, 2304), (2304, 1152), (256, 1152)])
def test_nvfp4_gemv(m, n):
    from runtime_cuda.inkling.formats import decode_nvfp4
    from runtime_cuda.inkling.gemv import nvfp4_gemv

    rng = np.random.default_rng(11)
    c, s, o = _rand_nvfp4(rng, 5, m, n)
    eids = torch.tensor([4, -1, 0, 2], dtype=torch.int32, device=DEV)
    x = torch.randn(4, n, device=DEV)
    got = nvfp4_gemv(x, c, s, o, m, n, eids).double()
    W = decode_nvfp4(c, s, o).double()
    want = torch.stack([x[i].double() @ W[e].T if e >= 0 else torch.zeros(m, dtype=torch.float64, device=DEV) for i, e in enumerate(eids.tolist())])
    err = ((got - want).norm() / want.norm()).item()
    print(f"nvfp4 gemv {m}x{n}: rel err {err:.2e}")
    assert err < 1e-6 and got[1].abs().max().item() == 0.0


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("T", [1, 6])
def test_moe_side_mixed_nvfp4(pure, T):
    """Mixed ht2 + NVFP4 (and NVFP4-only) side in one launch: trellis experts read rotated x, NVFP4 the raw x."""
    from runtime_cuda.inkling.formats import decode_nvfp4
    from runtime_cuda.inkling.gemv import MoESideGEMV

    rng = np.random.default_rng(4242)
    m, n, E = 256, 1152 if pure else 2304, 128
    x_div = 1 if pure else 8
    if pure:
        slab_of = torch.full((E,), 4, dtype=torch.int32, device=DEV)
        loc_of = torch.arange(E, dtype=torch.int32, device=DEV)
        slabs, E_nv = [], E
    else:
        slab_of = torch.tensor([4 if e % 2 == 0 else 0 for e in range(E)], dtype=torch.int32, device=DEV)
        loc_of = torch.tensor([e // 2 for e in range(E)], dtype=torch.int32, device=DEV)
        hc, hs, hl = _rand_trellis(rng, E // 2, m, n, "ht2")
        slabs, E_nv = [("ht2", hc, hs, hl)], E // 2
    nc, ns, no = _rand_nvfp4(rng, E_nv, m, n)
    side = MoESideGEMV(slabs, slab_of, loc_of, m, n, nvfp4=(nc, ns, no))
    pool = np.array([0, 1, 2, 3, 4, 5, 16, 17, 64, 65, 126, 127])
    eids = torch.from_numpy(np.concatenate([rng.choice(pool, 8, replace=False) for _ in range(T)]).astype(np.int32)).to(DEV)
    S = eids.numel()
    x_rot = torch.randn(S // x_div, n, device=DEV)
    x_raw = torch.randn(S // x_div, n, device=DEV)
    got = side(x_rot, eids, x_div, x_raw=x_raw).double()
    Wnv = decode_nvfp4(nc, ns, no).double()
    Wht = decode_trellis(hc, hs, hl, n, "ht2").double() if not pure else None
    want = torch.zeros(S, m, dtype=torch.float64, device=DEV)
    for p, e in enumerate(eids.tolist()):
        r = p // x_div
        if slab_of[e] == 4:
            want[p] = x_raw[r].double() @ Wnv[loc_of[e]].T
        else:
            want[p] = x_rot[r].double() @ Wht[loc_of[e]].T
    err = ((got - want).norm() / want.norm()).item()
    print(f"moe side pure={pure} T={T}: rel err {err:.2e}")
    assert err < 1e-6, err


def test_nvfp4_grouped_matmul():
    from runtime_cuda.inkling.formats import decode_nvfp4

    rng = np.random.default_rng(12)
    E, m, n, P = 10, 256, 1152, 200
    c, s, o = _rand_nvfp4(rng, E, m, n)
    expert = torch.from_numpy(rng.integers(0, E, size=P)).to(DEV)
    x = torch.randn(P, n, device=DEV)
    be, bp = group_by_expert(expert, E, 16)
    got = packed_matmul(x, c, s, "nvfp4", m, n, levels=o, block_expert=be, block_pairs=bp, n_pairs=P).double()
    W = decode_nvfp4(c, s, o).double()
    want = torch.stack([x[p].double() @ W[e].T for p, e in enumerate(expert.tolist())])
    err = ((got - want).norm() / want.norm()).item()
    print(f"nvfp4 grouped matmul: rel err {err:.2e}")
    assert err < 5e-5, err


def test_file_nvfp4_tables_match_ours():
    """When a packed file ships nvfp4_e2m1 / nvfp4_e4m3 tables (upstream reads them if present), they equal ours."""
    from runtime_cuda.inkling.formats import E2M1, e4m3_table_np

    p = Path(__file__).resolve().parents[1] / "models" / "Inkling-2-Mini-LookAtMyQuant-11.2GB" / "luts.npz"
    if not p.exists():
        pytest.skip("11.2 GB file not downloaded")
    l = np.load(p)
    assert np.array_equal(l["nvfp4_e2m1"].astype(np.float32), np.array(E2M1, dtype=np.float32))
    assert np.array_equal(l["nvfp4_e4m3"].astype(np.float32), e4m3_table_np())
