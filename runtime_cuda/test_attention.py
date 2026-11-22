"""Triton decode attention (split-K) against the SDPA prefill path, which builds the same scores explicitly."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_cuda.inkling.attention import decode_attention, prefill_attention  # noqa: E402



@pytest.mark.parametrize("L", [1, 5, 63, 300, 1100, 3000])
@pytest.mark.parametrize("window", [0, 1024])
def test_decode_matches_prefill(L, window):
    torch.manual_seed(L)
    H, Hkv, D, ext, cap = 24, 8, 128, 1024, 4096
    kv = (torch.randn(2, Hkv, cap, D, device="cuda") * 0.5).half()
    q = torch.randn(H, D, device="cuda") * 2
    rel = torch.randn(H, ext, device="cuda")
    sink = torch.randn(H, device="cuda")
    pos = torch.tensor([L - 1], dtype=torch.int32, device="cuda")
    got = decode_attention(q, kv, rel, sink, pos, window, ext)
    want = prefill_attention(q[:, None], kv[0, :, :L], kv[1, :, :L], rel[None], sink[None], pos.long(),
                             torch.arange(L, device="cuda"), window, ext)[0]
    err = (got - want).abs().max().item()
    assert err < 1e-4, err



@pytest.mark.parametrize("T", [1, 2, 5, 300])
def test_conv_kernels_match_reference(T):
    from runtime_cuda.inkling.conv import conv4_, dyn_conv4_
    from runtime_cuda.inkling.model import conv4, dyn_conv4

    torch.manual_seed(T)
    C, H, nh = 1024, 2304, 8
    prev = torch.randn(3, C, device="cuda")
    x = torch.randn(T, C, device="cuda")
    w = torch.randn(4, C, device="cuda")
    want, want_prev = conv4.__wrapped__(prev, x, w) if hasattr(conv4, "__wrapped__") else conv4(prev, x, w)
    p2 = prev.clone()
    got = conv4_(p2, x, w)
    assert (got - want).abs().max() < 1e-5 and (p2 - want_prev).abs().max() == 0
    st = torch.randn(nh, C // nh, 4, device="cuda")
    pw = torch.randn(nh * 4, H, device="cuda") * 0.05
    dyn_in = torch.randn(T, H, device="cuda")
    want, want_prev = dyn_conv4(prev, x, dyn_in, st, pw)
    p3 = prev.clone()
    got = dyn_conv4_(p3, x, dyn_in, st, pw)
    assert (got - want).abs().max() < 1e-4 and (p3 - want_prev).abs().max() == 0


@pytest.mark.parametrize("NQ,NS,T", [(1, 1, 1), (1, 2, 1), (3, 4, 1), (4, 5, 3), (1, 7, 2), (3, 3, 17)])
@pytest.mark.parametrize("norm", [False, True])
def test_block_attn_res_fused(NQ, NS, T, norm):
    from runtime_cuda.inkling.fused import block_attn_res_fused
    from runtime_cuda.inkling.model import block_attn_res, rms_norm

    torch.manual_seed(NQ * 10 + NS)
    H = 2304
    q = torch.randn(NQ, H, device="cuda") * 0.05
    states = [torch.randn(T, H, device="cuda") * (s + 1) for s in range(NS)]
    want = block_attn_res(q, states[:-1], states[-1])
    lnw = torch.rand(NQ, H, device="cuda") + 0.5 if norm else None
    if norm:
        want = torch.stack([rms_norm(want[i], lnw[i], 1e-6) for i in range(NQ)])
    got = block_attn_res_fused(q, states, lnw)
    assert ((got - want).norm() / want.norm()).item() < 1e-5


@pytest.mark.parametrize("G,T", [(1, 1), (3, 1), (3, 4)])
def test_message_gates_fused(G, T):
    from runtime_cuda.inkling.fused import message_gates_fused
    from runtime_cuda.inkling.model import message_gate

    torch.manual_seed(G * 7 + T)
    H, R = 2304, 64
    h, m = torch.randn(G, T, H, device="cuda"), torch.randn(G, T, H, device="cuda")
    dw = (torch.randn(G, R, 2 * H, device="cuda") * 0.02).bfloat16()
    fw, mw = torch.randn(G, H, R, device="cuda") * 0.1, torch.randn(G, H, R, device="cuda") * 0.1
    want = torch.stack([message_gate(h[g], m[g], dw[g], fw[g], mw[g]) for g in range(G)])
    got = message_gates_fused(h, m, dw, fw, mw)
    assert ((got - want).norm() / want.norm()).item() < 1e-4


@pytest.mark.parametrize("n", [2304, 1152, 3072, 384, 1024])
@pytest.mark.parametrize("S,T", [(1, 1), (5, 1), (1, 7), (3, 2)])
def test_rht_kernel_matches_reference(n, S, T):
    from runtime_cuda.inkling.formats import _factors, _rht_in, _rht_out
    from runtime_cuda.inkling.rht import rht

    torch.manual_seed(n + S + T)
    x = torch.randn(S, T, n, device="cuda")
    s = torch.randint(0, 2, (S, n), device="cuda").float() * 2 - 1
    A, B = _factors(n, x.device)
    for out, ref in ((False, _rht_in), (True, _rht_out)):
        want = ref(x, s[:, None, :], A, B).double()  # float32 matmuls (the reference casts to float32)
        got = rht(x, s, out)
        err = ((got.double() - want).norm() / want.norm()).item()
        assert err < 1e-5, (out, err)


@pytest.mark.parametrize("L", [1, 40, 1100])
@pytest.mark.parametrize("window", [0, 1024])
def test_decode_attention_inline_rel(L, window):
    torch.manual_seed(L)
    H, Hkv, D, ext, cap, dr = 24, 8, 128, 1024, 2048, 16
    kv = (torch.randn(2, Hkv, cap, D, device="cuda") * 0.5).half()
    q = torch.randn(H, D, device="cuda")
    relq = torch.randn(H, dr, device="cuda")
    proj = torch.randn(dr, ext, device="cuda") * 0.3
    sink = torch.randn(H, device="cuda")
    pos = torch.tensor([L - 1], dtype=torch.int32, device="cuda")
    want = decode_attention(q, kv, relq @ proj, sink, pos, window, ext)
    got = decode_attention(q, kv, relq, sink, pos, window, ext, proj=proj)
    assert (got - want).abs().max().item() < 1e-4
