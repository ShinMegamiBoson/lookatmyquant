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


@pytest.mark.parametrize("window", [0, 1024])
def test_prefill_flash_matches_eager_and_ring_wrap(window):
    import torch.nn.functional as F
    from runtime_cuda.inkling.attention import prefill_attention_kv
    from runtime_cuda.inkling.conv import kv_conv_write_

    torch.manual_seed(42 + window)
    H, Hkv, D, ext = 24, 8, 128, 1024
    # Test past ring wrap-around (L = 2500 > sw_cap = 2048)
    L, T = 2500, 128
    sw_cap = 2048 if window > 0 else 4096
    ring = window > 0
    kv_full = torch.zeros(2, Hkv, 4096, D, dtype=torch.float16, device="cuda")
    kv_ring = torch.zeros(2, Hkv, sw_cap, D, dtype=torch.float16, device="cuda")
    k_raw = torch.randn(L, Hkv * D, device="cuda") * 0.5
    v_raw = torch.randn(L, Hkv * D, device="cuda") * 0.5
    wk = torch.randn(4, Hkv * D, device="cuda") * 0.2
    wv = torch.randn(4, Hkv * D, device="cuda") * 0.2
    pos_all = torch.arange(L, dtype=torch.int32, device="cuda")
    pk1, pv1 = torch.zeros(3, Hkv * D, device="cuda"), torch.zeros(3, Hkv * D, device="cuda")
    pk2, pv2 = torch.zeros(3, Hkv * D, device="cuda"), torch.zeros(3, Hkv * D, device="cuda")
    kv_conv_write_(k_raw, v_raw, pk1, pv1, kv_full, pos_all, 22.6, 1e-6, Hkv, w_k=wk, w_v=wv, ring=False)
    kv_conv_write_(k_raw, v_raw, pk2, pv2, kv_ring, pos_all, 22.6, 1e-6, Hkv, w_k=wk, w_v=wv, ring=ring)

    q = torch.randn(T, H, D, device="cuda")
    rel = torch.randn(T, H, ext, device="cuda") * 0.2
    sink = torch.randn(T, H, device="cuda")
    q_pos = torch.arange(L - T, L, dtype=torch.int32, device="cuda")

    got = prefill_attention_kv(q, kv_ring, rel, sink, q_pos, L, window, ext, ring=ring)

    # Eager float32 reference
    k_pos = torch.arange(L, device="cuda")
    dist = q_pos.long()[:, None] - k_pos[None, :]
    ok = dist >= 0
    if window > 0:
        ok = ok & (dist < window)
    in_rel = ok & (dist < ext)
    bias = torch.gather(rel.permute(1, 0, 2), 2, dist.clamp(0, ext - 1)[None].expand(H, T, L))
    bias = torch.where(in_rel[None], bias, 0.0)
    bias = torch.where(ok[None], bias, float("-inf"))
    bias = torch.cat([bias, sink.T[:, :, None]], dim=-1)
    Kp = torch.cat([kv_full[0, :, :L].float(), torch.zeros(Hkv, 1, D, device="cuda")], dim=1).repeat_interleave(H // Hkv, 0)
    Vp = torch.cat([kv_full[1, :, :L].float(), torch.zeros(Hkv, 1, D, device="cuda")], dim=1).repeat_interleave(H // Hkv, 0)
    want = F.scaled_dot_product_attention(q.transpose(0, 1)[None], Kp[None], Vp[None], attn_mask=bias[None])[0].transpose(0, 1)
    assert (got - want).abs().max().item() < 2e-4

    # Also verify decode_attention on wrapped ring matches last query row
    dec = decode_attention(q[-1], kv_ring, rel[-1], sink[-1], q_pos[-1:], window, ext, ring=ring)
    assert (dec - want[-1]).abs().max().item() < 2e-4


@pytest.mark.parametrize("kv_mode,tol", [("fp16", 2e-4), ("q8_0", 2e-2), ("q4_0", 0.25)])
@pytest.mark.parametrize("tiered", [False, True])
def test_quantized_and_tiered_global_kv(kv_mode, tol, tiered):
    from runtime_cuda.inkling.attention import prefill_attention_kv
    from runtime_cuda.inkling.conv import kv_conv_write_

    torch.manual_seed(7)
    H, Hkv, D, ext = 24, 8, 128, 1024
    L, T = 384, 64
    vram_cells = 200 if tiered else L
    kv_ref = torch.zeros(2, Hkv, L, D, dtype=torch.float16, device="cuda")
    if kv_mode == "fp16":
        dtype, d_store, need_sc = torch.float16, D, False
    elif kv_mode == "q8_0":
        dtype, d_store, need_sc = torch.int8, D, True
    else:
        dtype, d_store, need_sc = torch.uint8, D // 2, True

    kv_vram = torch.zeros(2, Hkv, vram_cells, d_store, dtype=dtype, device="cuda")
    sc_vram = torch.zeros(2, Hkv, vram_cells, D // 32, dtype=torch.float16, device="cuda") if need_sc else None
    kv_host = torch.zeros(2, Hkv, L - vram_cells, d_store, dtype=dtype, pin_memory=True) if tiered else None
    sc_host = torch.zeros(2, Hkv, L - vram_cells, D // 32, dtype=torch.float16, pin_memory=True) if (tiered and need_sc) else None

    k_raw = torch.randn(L, Hkv * D, device="cuda") * 0.5
    v_raw = torch.randn(L, Hkv * D, device="cuda") * 0.5
    wk = torch.randn(4, Hkv * D, device="cuda") * 0.2
    wv = torch.randn(4, Hkv * D, device="cuda") * 0.2
    pos = torch.arange(L, dtype=torch.int32, device="cuda")
    kv_conv_write_(k_raw, v_raw, torch.zeros(3, Hkv * D, device="cuda"), torch.zeros(3, Hkv * D, device="cuda"),
                   kv_ref, pos, 22.6, 1e-6, Hkv, w_k=wk, w_v=wv)
    kw = dict(kv_mode=kv_mode, scales=sc_vram, kv_host=kv_host, scales_host=sc_host, vram_cells=vram_cells if tiered else None)
    kv_conv_write_(k_raw, v_raw, torch.zeros(3, Hkv * D, device="cuda"), torch.zeros(3, Hkv * D, device="cuda"),
                   kv_vram, pos, 22.6, 1e-6, Hkv, w_k=wk, w_v=wv, **kw)

    q = torch.randn(T, H, D, device="cuda")
    rel = torch.randn(T, H, ext, device="cuda") * 0.2
    sink = torch.randn(T, H, device="cuda")
    q_pos = torch.arange(L - T, L, dtype=torch.int32, device="cuda")
    want = prefill_attention_kv(q, kv_ref, rel, sink, q_pos, L, 0, ext)
    got = prefill_attention_kv(q, kv_vram, rel, sink, q_pos, L, 0, ext, **kw)
    dec_got = decode_attention(q[-1], kv_vram, rel[-1], sink[-1], q_pos[-1:], 0, ext, **kw)
    assert (got - want).abs().max().item() < tol
    assert (dec_got - got[-1]).abs().max().item() < 2e-4
