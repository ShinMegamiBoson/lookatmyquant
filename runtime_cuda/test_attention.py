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


def _pease_fwht_ref(y: torch.Tensor) -> torch.Tensor:
    import math

    hd = y.shape[-1]
    out = y.float()
    for _ in range(int(round(math.log2(hd)))):
        pairs = out.reshape(*out.shape[:-1], hd // 2, 2)
        a, b = pairs[..., 0], pairs[..., 1]
        out = torch.stack([a + b, a - b], dim=-1).transpose(-1, -2).reshape(*out.shape[:-1], hd)
    return out * (1.0 / math.sqrt(hd))


def _lm_quant_dequant_ref(x: torch.Tensor, bits: int, signs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from runtime_cuda.inkling.conv import lm_levels_tensor, lm_mids_tensor

    x_rot = _pease_fwht_ref(x.float() * signs)
    rms = x_rot.pow(2).mean(dim=-1).sqrt().clamp(min=1e-6).half()
    inv_sc = 1.0 / rms.float().unsqueeze(-1)
    x_norm = x_rot * inv_sc
    mids = lm_mids_tensor(bits, x.device)
    levels = lm_levels_tensor(bits, x.device)
    idx = (x_norm.unsqueeze(-1) >= mids).sum(dim=-1)
    return levels[idx] * rms.float().unsqueeze(-1), rms


@pytest.mark.parametrize("hd", [64, 128])
@pytest.mark.parametrize("nrows", [1, 4, 32])
def test_rht_fwd_inv_orthogonality(hd, nrows):
    from runtime_cuda.inkling.conv import default_rht_signs, rht_fwd, rht_inv

    torch.manual_seed(hd + nrows)
    signs = default_rht_signs(hd, "cuda")
    x = torch.randn(nrows, hd, device="cuda") if nrows > 1 else torch.randn(hd, device="cuda")
    y = rht_fwd(x, signs)
    x_rec = rht_inv(y, signs)
    assert (x_rec - x).abs().max().item() < 1e-5
    if nrows > 1:
        q = torch.randn(nrows, hd, device="cuda")
        dot_rot = rht_fwd(q) @ y.T
        dot_ref = q @ x.T
        assert ((dot_rot - dot_ref).norm() / dot_ref.norm()).item() < 1e-5
        assert (dot_rot - dot_ref).abs().max().item() < 5e-5


@pytest.mark.parametrize("k_bits,v_bits", [(4, 4), (4, 3), (4, 2), (3, 3), (3, 2), (2, 2)])
@pytest.mark.parametrize("dynamic", [False, True])
def test_lm_kv_write_decode_prefill_matches_reference(k_bits, v_bits, dynamic):
    import torch.nn.functional as F
    from runtime_cuda.inkling.attention import decode_attention_lm, prefill_attention_kv_lm
    from runtime_cuda.inkling.conv import default_rht_signs, kv_conv_write_lm_, lm_bytes_per_head
    from runtime_cuda.inkling.model import conv4, dyn_conv4, l2_clip, rms_norm

    torch.manual_seed(100 + k_bits * 10 + v_bits + int(dynamic))
    H, Hkv, D, ext = 24, 8, 128, 1024
    L, T, thr, eps = 192, 48, 22.6, 1e-6
    signs = default_rht_signs(D, "cuda")

    k_buf = torch.zeros(Hkv, L, lm_bytes_per_head(D, k_bits), dtype=torch.uint8, device="cuda")
    k_sc = torch.zeros(Hkv, L, dtype=torch.float16, device="cuda")
    v_buf = torch.zeros(Hkv, L, lm_bytes_per_head(D, v_bits), dtype=torch.uint8, device="cuda")
    v_sc = torch.zeros(Hkv, L, dtype=torch.float16, device="cuda")

    k_raw = torch.randn(L, Hkv * D, device="cuda") * 0.5
    v_raw = torch.randn(L, Hkv * D, device="cuda") * 0.5
    pk0 = torch.randn(3, Hkv * D, device="cuda") * 0.2
    pv0 = torch.randn(3, Hkv * D, device="cuda") * 0.2
    pk1, pv1 = pk0.clone(), pv0.clone()
    pos = torch.arange(L, dtype=torch.int32, device="cuda")

    if dynamic:
        H_model = 256
        k_in = torch.randn(L, H_model, device="cuda") * 0.3
        v_in = torch.randn(L, H_model, device="cuda") * 0.3
        st_k = torch.randn(Hkv, D, 4, device="cuda") * 0.2
        pr_k = torch.randn(Hkv * 4, H_model, device="cuda") * 0.05
        st_v = torch.randn(Hkv, D, 4, device="cuda") * 0.2
        pr_v = torch.randn(Hkv * 4, H_model, device="cuda") * 0.05
        kv_conv_write_lm_(
            k_raw, v_raw, pk1, pv1, k_buf, k_sc, v_buf, v_sc, pos, thr, eps, Hkv,
            dyn=(k_in, v_in, (st_k, pr_k), (st_v, pr_v)),
            k_bits=k_bits, v_bits=v_bits, rht_signs=signs,
        )
        kc_ref, _ = dyn_conv4(pk0, k_raw, k_in, st_k, pr_k)
        vc_ref, _ = dyn_conv4(pv0, v_raw, v_in, st_v, pr_v)
    else:
        wk = torch.randn(4, Hkv * D, device="cuda") * 0.2
        wv = torch.randn(4, Hkv * D, device="cuda") * 0.2
        kv_conv_write_lm_(
            k_raw, v_raw, pk1, pv1, k_buf, k_sc, v_buf, v_sc, pos, thr, eps, Hkv,
            w_k=wk, w_v=wv, k_bits=k_bits, v_bits=v_bits, rht_signs=signs,
        )
        kc_fn = conv4.__wrapped__ if hasattr(conv4, "__wrapped__") else conv4
        kc_ref, _ = kc_fn(pk0, k_raw, wk)
        vc_ref, _ = kc_fn(pv0, v_raw, wv)

    k_clip = l2_clip(kc_ref.reshape(L, Hkv, D), thr).transpose(0, 1)  # [Hkv, L, D]
    v_norm = rms_norm(vc_ref.reshape(L, Hkv, D), None, eps).transpose(0, 1)
    k_rot_q, k_sc_ref = _lm_quant_dequant_ref(k_clip, k_bits, signs)
    v_rot_q, v_sc_ref = _lm_quant_dequant_ref(v_norm, v_bits, signs)
    assert (k_sc - k_sc_ref).abs().max().item() <= 5e-4
    assert (v_sc - v_sc_ref).abs().max().item() <= 5e-4

    k_deq = _pease_fwht_ref(k_rot_q) * signs
    v_deq = _pease_fwht_ref(v_rot_q) * signs

    q = torch.randn(T, H, D, device="cuda")
    rel = torch.randn(T, H, ext, device="cuda") * 0.2
    sink = torch.randn(T, H, device="cuda")
    q_pos = torch.arange(L - T, L, dtype=torch.int32, device="cuda")

    got_pre = prefill_attention_kv_lm(
        q, k_buf, k_sc, v_buf, v_sc, rel, sink, q_pos, L, 0, ext,
        k_bits=k_bits, v_bits=v_bits, rht_signs=signs,
    )
    got_dec = decode_attention_lm(
        q[-1], k_buf, k_sc, v_buf, v_sc, rel[-1], sink[-1], q_pos[-1:], 0, ext,
        k_bits=k_bits, v_bits=v_bits, rht_signs=signs,
    )

    dr = 16
    relq = torch.randn(H, dr, device="cuda")
    proj = torch.randn(dr, ext, device="cuda") * 0.3
    got_dec_inline = decode_attention_lm(
        q[-1], k_buf, k_sc, v_buf, v_sc, relq, sink[-1], q_pos[-1:], 0, ext, proj=proj,
        k_bits=k_bits, v_bits=v_bits, rht_signs=signs,
    )
    got_dec_proj = decode_attention_lm(
        q[-1], k_buf, k_sc, v_buf, v_sc, relq @ proj, sink[-1], q_pos[-1:], 0, ext,
        k_bits=k_bits, v_bits=v_bits, rht_signs=signs,
    )
    assert (got_dec_inline - got_dec_proj).abs().max().item() < 2e-4

    # Explicit FP32 SDPA reference over dequantized K/V
    k_pos = torch.arange(L, device="cuda")
    dist = q_pos.long()[:, None] - k_pos[None, :]
    ok = dist >= 0
    in_rel = ok & (dist < ext)
    bias = torch.gather(rel.permute(1, 0, 2), 2, dist.clamp(0, ext - 1)[None].expand(H, T, L))
    bias = torch.where(in_rel[None], bias, 0.0)
    bias = torch.where(ok[None], bias, float("-inf"))
    bias = torch.cat([bias, sink.T[:, :, None]], dim=-1)
    Kp = torch.cat([k_deq, torch.zeros(Hkv, 1, D, device="cuda")], dim=1).repeat_interleave(H // Hkv, 0)
    Vp = torch.cat([v_deq, torch.zeros(Hkv, 1, D, device="cuda")], dim=1).repeat_interleave(H // Hkv, 0)
    want = F.scaled_dot_product_attention(q.transpose(0, 1)[None], Kp[None], Vp[None], attn_mask=bias[None])[0].transpose(0, 1)

    assert (got_pre - want).abs().max().item() < 2e-3
    assert (got_dec - want[-1]).abs().max().item() < 2e-3
    assert (got_dec - got_pre[-1]).abs().max().item() < 5e-4


@pytest.mark.parametrize("k_bits,v_bits", [(4, 4), (4, 3), (4, 2), (3, 3), (3, 2), (2, 2)])
def test_lm_tiered_uva_pinned_host_matches_all_vram(k_bits, v_bits):
    from runtime_cuda.inkling.attention import decode_attention_lm, prefill_attention_kv_lm
    from runtime_cuda.inkling.conv import default_rht_signs, kv_conv_write_lm_, lm_bytes_per_head

    torch.manual_seed(33 + k_bits * 7 + v_bits)
    H, Hkv, D, ext = 24, 8, 128, 1024
    vram_cells, L, T = 128, 512, 64
    h_cells = L - vram_cells
    signs = default_rht_signs(D, "cuda")
    kb, vb = lm_bytes_per_head(D, k_bits), lm_bytes_per_head(D, v_bits)

    k_vram = torch.zeros(Hkv, L, kb, dtype=torch.uint8, device="cuda")
    k_sc_vram = torch.zeros(Hkv, L, dtype=torch.float16, device="cuda")
    v_vram = torch.zeros(Hkv, L, vb, dtype=torch.uint8, device="cuda")
    v_sc_vram = torch.zeros(Hkv, L, dtype=torch.float16, device="cuda")

    k_tier = torch.zeros(Hkv, vram_cells, kb, dtype=torch.uint8, device="cuda")
    k_sc_tier = torch.zeros(Hkv, vram_cells, dtype=torch.float16, device="cuda")
    v_tier = torch.zeros(Hkv, vram_cells, vb, dtype=torch.uint8, device="cuda")
    v_sc_tier = torch.zeros(Hkv, vram_cells, dtype=torch.float16, device="cuda")
    k_host = torch.zeros(Hkv, h_cells, kb, dtype=torch.uint8, pin_memory=True)
    k_sc_host = torch.zeros(Hkv, h_cells, dtype=torch.float16, pin_memory=True)
    v_host = torch.zeros(Hkv, h_cells, vb, dtype=torch.uint8, pin_memory=True)
    v_sc_host = torch.zeros(Hkv, h_cells, dtype=torch.float16, pin_memory=True)

    k_raw = torch.randn(L, Hkv * D, device="cuda") * 0.5
    v_raw = torch.randn(L, Hkv * D, device="cuda") * 0.5
    wk = torch.randn(4, Hkv * D, device="cuda") * 0.2
    wv = torch.randn(4, Hkv * D, device="cuda") * 0.2
    pos = torch.arange(L, dtype=torch.int32, device="cuda")

    kv_conv_write_lm_(
        k_raw, v_raw, torch.zeros(3, Hkv * D, device="cuda"), torch.zeros(3, Hkv * D, device="cuda"),
        k_vram, k_sc_vram, v_vram, v_sc_vram, pos, 22.6, 1e-6, Hkv,
        w_k=wk, w_v=wv, k_bits=k_bits, v_bits=v_bits, rht_signs=signs,
    )
    tier_kw = dict(
        k_bits=k_bits, v_bits=v_bits, rht_signs=signs,
        k_host=k_host, k_sc_host=k_sc_host, v_host=v_host, v_sc_host=v_sc_host, vram_cells=vram_cells,
    )
    kv_conv_write_lm_(
        k_raw, v_raw, torch.zeros(3, Hkv * D, device="cuda"), torch.zeros(3, Hkv * D, device="cuda"),
        k_tier, k_sc_tier, v_tier, v_sc_tier, pos, 22.6, 1e-6, Hkv,
        w_k=wk, w_v=wv, **tier_kw,
    )

    q = torch.randn(T, H, D, device="cuda")
    rel = torch.randn(T, H, ext, device="cuda") * 0.2
    sink = torch.randn(T, H, device="cuda")
    q_pos = torch.arange(L - T, L, dtype=torch.int32, device="cuda")

    ref_pre = prefill_attention_kv_lm(q, k_vram, k_sc_vram, v_vram, v_sc_vram, rel, sink, q_pos, L, 0, ext,
                                      k_bits=k_bits, v_bits=v_bits, rht_signs=signs)
    got_pre = prefill_attention_kv_lm(q, k_tier, k_sc_tier, v_tier, v_sc_tier, rel, sink, q_pos, L, 0, ext, **tier_kw)
    ref_dec = decode_attention_lm(q[-1], k_vram, k_sc_vram, v_vram, v_sc_vram, rel[-1], sink[-1], q_pos[-1:], 0, ext,
                                  k_bits=k_bits, v_bits=v_bits, rht_signs=signs)
    got_dec = decode_attention_lm(q[-1], k_tier, k_sc_tier, v_tier, v_sc_tier, rel[-1], sink[-1], q_pos[-1:], 0, ext, **tier_kw)
    assert (got_pre - ref_pre).abs().max().item() < 1e-5
    assert (got_dec - ref_dec).abs().max().item() < 1e-5


@pytest.mark.parametrize("mode", ["fp16", "q4_0", "lm4", "lm4_3", "lm3_2"])
def test_cache_shared_loop_kv_vram_and_aliasing(mode, monkeypatch):
    from runtime_cuda.inkling.model import Inkling2Cache, Inkling2Config

    cfg = Inkling2Config()
    dev = torch.device("cuda")
    unshared = Inkling2Cache(cfg, 4096, dev, global_kv_mode=mode, shared_loop_kv=False)
    shared = Inkling2Cache(cfg, 4096, dev, global_kv_mode=mode, shared_loop_kv=True)

    sw_bytes = sum(unshared.kv[s].numel() * unshared.kv[s].element_size() for s in range(48) if unshared.is_sliding[s])
    assert (shared.kv_vram_bytes() - sw_bytes) * 2 == (unshared.kv_vram_bytes() - sw_bytes)

    # Aliased loop global slots (11, 29 -> 41 and 17, 35 -> 47) share exact tensor objects
    for early, canon in ((11, 41), (29, 41), (17, 47), (35, 47)):
        if mode in ("fp16", "q4_0"):
            assert shared.kv[early] is shared.kv[canon]
            assert shared.scales[early] is shared.scales[canon]
        else:
            assert shared.k_buf[early] is shared.k_buf[canon]
            assert shared.k_sc[early] is shared.k_sc[canon]
            assert shared.v_buf[early] is shared.v_buf[canon]
            assert shared.v_sc[early] is shared.v_sc[canon]
    # All 48 conv_state slots stay independent
    assert len({id(shared.conv_state[s][0]) for s in range(48)}) == 48

    # Env vars + tiered UVA 50% host reduction + per-layer KV plan
    monkeypatch.setenv("LAMQ_CUDA_SHARED_LOOP_KV", "1")
    monkeypatch.setenv("LAMQ_CUDA_GLOBAL_KV", mode)
    tiered_shared = Inkling2Cache(cfg, 4096, dev, kv_vram_cells=1024)
    tiered_unshared = Inkling2Cache(cfg, 4096, dev, kv_vram_cells=1024, shared_loop_kv=False)
    assert tiered_shared.shared_loop_kv is True
    assert tiered_shared.kv_host_bytes() * 2 == tiered_unshared.kv_host_bytes()
    assert (tiered_shared.kv_vram_bytes() - sw_bytes) * 2 == (tiered_unshared.kv_vram_bytes() - sw_bytes)

    monkeypatch.setenv("LAMQ_CUDA_GLOBAL_KV_PLAN", "5:4,4;11:4,3;17:3,2;23:4,4")
    planned = Inkling2Cache(cfg, 2048, dev)
    assert planned.slot_mode[5] == "lm4"
    assert (planned.slot_k_bits[11], planned.slot_v_bits[11]) == (4, 3)
    assert (planned.slot_k_bits[41], planned.slot_v_bits[41]) == (4, 3)
    assert (planned.slot_k_bits[17], planned.slot_v_bits[17]) == (3, 2)
    assert (planned.slot_k_bits[47], planned.slot_v_bits[47]) == (3, 2)
    assert planned.slot_mode[23] == "lm4"


def test_cache_snapshot_and_cuda_graph_shared_loop_lm(tmp_path):
    from runtime_cuda.inkling.generate import GreedyDecodeGraph, generate_serial
    from runtime_cuda.inkling.model import Inkling2LoopPacked
    from runtime_cuda.test_upstream_fixture import UPSTREAM_TEST, _add_model_type, load_upstream_fixture

    if not UPSTREAM_TEST.exists():
        pytest.skip("requires runtime_mlx/test_inkling.py fixture")
    load_upstream_fixture()(tmp_path)
    _add_model_type(tmp_path)

    m = Inkling2LoopPacked(tmp_path, "cuda")
    toks = [5, 19, 42, 128, 301, 7, 255, 12, 400, 88, 3, 77]

    # Miniature config has 6 layers, loop_start=2, loop_end=4, global layers 1,3,5 -> canonical looped slot 9 (aliases 3, 7)
    c1 = m.make_cache(128, global_kv_mode="lm4_3", kv_vram_cells=4, shared_loop_kv=True)
    m.prefill(toks[:8], c1)
    snap = c1.snapshot_cpu()
    assert snap["meta"]["shared_loop_kv"] is True
    assert snap["meta"]["global_kv_mode"] == "lm4_3"
    assert "k_buf.9" in snap["tensors"]
    assert "k_buf.3" not in snap["tensors"] and "k_buf.7" not in snap["tensors"]
    for s in range(m.cfg.total_visits):
        for ci in range(4):
            assert f"conv.{s}.{ci}" in snap["tensors"]

    c2 = m.make_cache(128, global_kv_mode="lm4_3", kv_vram_cells=4, shared_loop_kv=True)
    c2.restore_from_snapshot(snap)
    lg1 = m.prefill(toks[8:], c1, all_logits=True)
    lg2 = m.prefill(toks[8:], c2, all_logits=True)
    assert torch.equal(lg1, lg2)

    # CUDA-graph capture & replay with lm4_3 + shared_loop_kv + tiered UVA host memory
    c_graph = m.make_cache(128, global_kv_mode="lm4_3", kv_vram_cells=4, shared_loop_kv=True)
    g = GreedyDecodeGraph(m, c_graph, 16)
    gen_graph = generate_serial(m, toks, max_tokens=6, eos_ids=(), cache=c_graph, graph=g)

    c_ref = m.make_cache(128, global_kv_mode="lm4_3", kv_vram_cells=4, shared_loop_kv=True)
    nxt = int(m.prefill(toks, c_ref)[-1].argmax())
    gen_ref = [nxt]
    for _ in range(5):
        h = m.decode_step(torch.tensor([nxt], device="cuda"), c_ref)
        nxt = int(m.argmax(h).item())
        gen_ref.append(nxt)
    assert gen_graph == gen_ref
    assert int(c_graph.pos.item()) == len(toks) + 5
    assert int(c_ref.pos.item()) == len(toks) + 5
