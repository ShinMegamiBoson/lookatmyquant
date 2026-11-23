"""Self-contained Apple Silicon MLX tests for packed Inkling-2-Mini runtime and CLI integration."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_mlx.inkling.drafter import resolve_drafter_files
from runtime_mlx.inkling.kernels import trellis_matvec, trellis_moe_grouped
from runtime_mlx.inkling.model import (
    FMT_PARAMS,
    ID_TO_FMT,
    Inkling2LoopPackedMLX,
    Inkling2MLXModel,
)
from runtime_mlx.runtime import detect_format, inspect_model


def _pack_u32_random(rng, shape_m_wpr):
    return rng.integers(0, 2**32, size=shape_m_wpr, dtype=np.uint32)


def build_mini_packed_dir(out_dir: Path) -> None:
    """Create a miniature 6-layer / 10-visit looped MoE packed Inkling-2-Loop directory exercising every code path."""
    rng = np.random.default_rng(2026)
    H = 256
    I = 256
    n_q, n_kv, hd = 4, 4, 64
    d_rel, rel_ext = 16, 32
    V_emb, V_head = 512, 500
    n_layers = 6
    n_exp = 8
    top_k = 2
    cfg = {
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "text_config": {
            "model_type": "inkling2_loop",
            "hidden_size": H,
            "num_hidden_layers": n_layers,
            "vocab_size": V_emb,
            "unpadded_vocab_size": V_head,
            "num_attention_heads": n_q,
            "num_key_value_heads": n_kv,
            "head_dim": hd,
            "d_rel": d_rel,
            "rel_extent": rel_ext,
            "log_scaling_n_floor": 4,
            "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-6,
            "qk_l2_clip_threshold": 22.627416997969522,
            "rel_l2_clip_threshold": 8.0,
            "local_layer_ids": [0, 2, 4],
            "sconv_kernel_size": 4,
            "conv_kernel_size": 4,
            "final_logit_softcapping": 30.0,
            "sliding_window_size": 8,
            "n_layers_per_attn_res_block": 2,
            "n_routed_experts": n_exp,
            "num_experts_per_tok": top_k,
            "intermediate_size": I,
            "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0,
            "route_scale": 8.0,
            "loop_config": {"core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 32},
        },
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    luts = {
        "ht2": rng.standard_normal((512, 4)).astype(np.float16) * 0.5,
        "ht3": rng.standard_normal((1024, 2)).astype(np.float16) * 0.5,
        "rtr1": rng.standard_normal((4096,)).astype(np.float16) * 0.5,
        "rtr2": rng.standard_normal((4096,)).astype(np.float16) * 0.5,
        "rtr3": rng.standard_normal((4096,)).astype(np.float16) * 0.5,
        "rtr4": rng.standard_normal((4096,)).astype(np.float16) * 0.5,
    }
    np.savez(out_dir / "luts.npz", **luts)

    st: dict[str, mx.array] = {
        "model.embed_tokens.embed_norm.weight": mx.array(rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32)),
        "model.final_input.query": mx.array(rng.standard_normal((H,)).astype(np.float32) * 0.1),
        "model.norm.weight": mx.array(rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32)),
        "model.message_input.query": mx.array(rng.standard_normal((2, 4, H)).astype(np.float32) * 0.1),
    }

    emb_order = rng.permutation(V_emb).astype(np.int32)
    emb_meta = [
        {"kb": 0, "opt": "lm8", "rows": 128, "k": 8},
        {"kb": 1, "opt": "rtr2", "rows": V_emb - 128, "k": 2, "g": 1, "K": 4096, "hash_mult": 0},
    ]
    np.savez(
        out_dir / "embed_packed.npz",
        order=emb_order,
        b0_codes=rng.integers(0, 256, size=(128 * H,), dtype=np.uint8),
        b0_scales=rng.uniform(0.01, 0.03, size=(128,)).astype(np.float16),
        b0_levels=np.sort(rng.standard_normal(256)).astype(np.float32),
        b0_signs_in=rng.choice([-1, 1], size=(H,)).astype(np.int8),
        b1_codes=_pack_u32_random(rng, (V_emb - 128, (H * 2) // 32)),
        b1_scales=rng.uniform(0.01, 0.03, size=(V_emb - 128,)).astype(np.float16),
        b1_signs_in=rng.choice([-1, 1], size=(H,)).astype(np.int8),
    )
    (out_dir / "embed_meta.json").write_text(json.dumps(emb_meta, indent=2))

    hd_order = rng.permutation(V_head).astype(np.int32)
    hd_meta = [
        {"kb": 0, "opt": "lm8", "rows": 100, "k": 8},
        {"kb": 1, "opt": "lm6", "rows": 150, "k": 6},
        {"kb": 2, "opt": "rtr1", "rows": V_head - 250, "k": 1, "g": 1, "K": 4096, "hash_mult": 0},
    ]
    np.savez(
        out_dir / "head_packed.npz",
        order=hd_order,
        b0_codes=rng.integers(0, 256, size=(100 * H,), dtype=np.uint8),
        b0_scales=rng.uniform(0.01, 0.03, size=(100,)).astype(np.float16),
        b0_levels=np.sort(rng.standard_normal(256)).astype(np.float32),
        b0_signs_in=rng.choice([-1, 1], size=(H,)).astype(np.int8),
        b1_codes=rng.integers(0, 256, size=((150 * H * 6) // 8,), dtype=np.uint8),
        b1_scales=rng.uniform(0.01, 0.03, size=(150,)).astype(np.float16),
        b1_levels=np.sort(rng.standard_normal(64)).astype(np.float32),
        b1_signs_in=rng.choice([-1, 1], size=(H,)).astype(np.int8),
        b2_codes=_pack_u32_random(rng, (V_head - 250, (H * 1) // 32)),
        b2_scales=rng.uniform(0.01, 0.03, size=(V_head - 250,)).astype(np.float16),
        b2_signs_in=rng.choice([-1, 1], size=(H,)).astype(np.int8),
    )
    (out_dir / "head_meta.json").write_text(json.dumps(hd_meta, indent=2))

    lin_specs = [
        ("self_attn.q_proj.linear", n_q * hd, H, "lm8"),
        ("self_attn.k_proj.linear", n_kv * hd, H, "lm6"),
        ("self_attn.v_proj.linear", n_kv * hd, H, "rtr4"),
        ("self_attn.o_proj.linear", H, n_q * hd, "ht3"),
        ("self_attn.o_proj.gate_proj", n_q * hd, H, "lm6"),
        ("self_attn.relative_position_bias.linear", n_q * d_rel, H, "ht2"),
    ]
    for i in range(n_layers):
        p = f"model.layers.{i}."
        is_sliding = i in (0, 2, 4)
        is_loop = 2 <= i < 4
        if i > 0:
            st[p + "attention_input.query"] = mx.array(
                rng.standard_normal((H,) if is_sliding else (3, H)).astype(np.float32) * 0.1
            )
        st[p + "mlp_input.query"] = mx.array(rng.standard_normal((H,)).astype(np.float32) * 0.1)
        if is_sliding:
            st[p + "input_layernorm.weight"] = mx.array(rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32))
        else:
            for ln in ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm"):
                st[p + f"{ln}.weight"] = mx.array(rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32))
        st[p + "post_attention_layernorm.weight"] = mx.array(rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32))
        st[p + "self_attn.q_proj.norm.scale"] = mx.array(rng.uniform(0.8, 1.2, size=(n_q, hd)).astype(np.float32))
        st[p + "self_attn.relative_position_bias.norm.scale"] = mx.array(rng.uniform(0.8, 1.2, size=(n_q, d_rel)).astype(np.float32))
        st[p + "self_attn.relative_position_bias.proj"] = mx.array(rng.standard_normal((d_rel, rel_ext)).astype(np.float32) * 0.1)
        st[p + "self_attn.sink_proj.weight"] = mx.array(rng.standard_normal((n_q, H)).astype(np.float32) * 0.05)
        if is_sliding:
            st[p + "self_attn.k_proj.sconv.conv1d.weight"] = mx.array(rng.standard_normal((n_kv * hd, 1, 4)).astype(np.float32) * 0.2)
            st[p + "self_attn.v_proj.sconv.conv1d.weight"] = mx.array(rng.standard_normal((n_kv * hd, 1, 4)).astype(np.float32) * 0.2)
        else:
            st[p + "self_attn.k_proj.sconv.static_weight"] = mx.array(rng.standard_normal((4, n_kv * hd)).astype(np.float32) * 0.2)
            st[p + "self_attn.k_proj.sconv.weight_proj.weight"] = mx.array(rng.standard_normal((n_kv * 4, H)).astype(np.float32) * 0.05)
            st[p + "self_attn.v_proj.sconv.static_weight"] = mx.array(rng.standard_normal((4, n_kv * hd)).astype(np.float32) * 0.2)
            st[p + "self_attn.v_proj.sconv.weight_proj.weight"] = mx.array(rng.standard_normal((n_kv * 4, H)).astype(np.float32) * 0.05)
        st[p + "post_attention_convolution.conv1d.weight"] = mx.array(rng.standard_normal((H, 1, 4)).astype(np.float32) * 0.2)
        st[p + "post_mlp_convolution.conv1d.weight"] = mx.array(rng.standard_normal((H, 1, 4)).astype(np.float32) * 0.2)
        if is_loop:
            for gn in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"):
                st[p + f"{gn}.down_proj.weight"] = mx.array(rng.standard_normal((32, 2 * H)).astype(np.float32) * 0.05)
                st[p + f"{gn}.forget_gate_proj.weight"] = mx.array(rng.standard_normal((H, 32)).astype(np.float32) * 0.05)
                st[p + f"{gn}.message_gate_proj.weight"] = mx.array(rng.standard_normal((H, 32)).astype(np.float32) * 0.05)
        st[p + "mlp.gate.weight"] = mx.array(rng.standard_normal((n_exp, H)).astype(np.float32) * 0.1)
        st[p + "mlp.gate.e_score_correction_bias"] = mx.array(rng.standard_normal((n_exp,)).astype(np.float32) * 0.02)

        l_arrs: dict[str, np.ndarray] = {}
        l_meta = {"layer": i, "linears": {}, "experts": {}}
        for name, m, n, opt in lin_specs:
            safe = name.replace(".", "_")
            l_arrs[f"lin_{safe}_signs_in"] = rng.choice([-1, 1], size=(n,)).astype(np.int8)
            l_arrs[f"lin_{safe}_scales"] = rng.uniform(0.01, 0.03, size=(m,)).astype(np.float16)
            if opt.startswith("lm"):
                k = int(opt[2:])
                l_arrs[f"lin_{safe}_codes"] = rng.integers(0, 256, size=((m * n * k) // 8,), dtype=np.uint8)
                l_arrs[f"lin_{safe}_levels"] = np.sort(rng.standard_normal(1 << k)).astype(np.float32)
                l_arrs[f"lin_{safe}_signs_out"] = rng.choice([-1, 1], size=(m,)).astype(np.int8)
                l_meta["linears"][name] = {"fmt": opt, "m": m, "n": n, "k": k, "out_rot": True, "safe": safe}
            else:
                k, g, K, hmult = FMT_PARAMS[opt]
                l_arrs[f"lin_{safe}_codes"] = _pack_u32_random(rng, (m, (n * k) // 32))
                l_meta["linears"][name] = {"fmt": opt, "m": m, "n": n, "k": k, "g": g, "K": K, "hash_mult": hmult, "safe": safe}

        l_arrs["gu_signs_in"] = rng.choice([-1, 1], size=(H,)).astype(np.int8)
        l_arrs["dn_signs_in"] = rng.choice([-1, 1], size=(I,)).astype(np.int8)
        gu_map = np.zeros((n_exp, 2), dtype=np.int16)
        dn_map = np.zeros((n_exp, 2), dtype=np.int16)
        exp_fmts = ["ht2", "ht3", "rtr1", "rtr4"]
        fmt_to_id = {v: k for k, v in ID_TO_FMT.items()}
        for fi, fmt in enumerate(exp_fmts):
            k, g, K, hmult = FMT_PARAMS[fmt]
            l_arrs[f"gu_{fmt}_codes"] = _pack_u32_random(rng, (2, 2 * I, (H * k) // 32))
            l_arrs[f"gu_{fmt}_scales"] = rng.uniform(0.01, 0.03, size=(2, 2 * I)).astype(np.float16)
            l_arrs[f"dn_{fmt}_codes"] = _pack_u32_random(rng, (2, H, (I * k) // 32))
            l_arrs[f"dn_{fmt}_scales"] = rng.uniform(0.01, 0.03, size=(2, H)).astype(np.float16)
            for loc in range(2):
                e = fi * 2 + loc
                gu_map[e] = [fmt_to_id[fmt], loc]
                dn_map[e] = [fmt_to_id[fmt], loc]
        l_arrs["gu_map"] = gu_map
        l_arrs["dn_map"] = dn_map
        np.savez(out_dir / f"layer_{i:02d}_packed.npz", **l_arrs)
        (out_dir / f"layer_{i:02d}_meta.json").write_text(json.dumps(l_meta, indent=2))

    mx.save_safetensors(str(out_dir / "steering.safetensors"), st)
    manifest = {
        "format": "inkling2-packed-v1-sqyfr",
        "packed_text_gb": 0.01,
        "steering_tensors": len(st),
        "luts": {k: list(v.shape) for k, v in luts.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def test_mini_model_and_runtime_integration() -> None:
    with tempfile.TemporaryDirectory() as td:
        pdir = Path(td)
        build_mini_packed_dir(pdir)

        assert detect_format(pdir) == "inkling2-packed"
        info = inspect_model(pdir)
        assert info["format"] == "inkling2-packed"
        assert info["num_hidden_layers"] == 6
        assert info["n_routed_experts"] == 8
        assert info["loop_passes"] == 3
        assert info["separate_lm_head"] is True

        model = Inkling2LoopPackedMLX(pdir)
        toks = [5, 19, 42, 128, 301, 7, 255, 12, 400, 88]

        logits_full, _ = model.forward_all_logits(toks)
        mx.eval(logits_full)
        lf = np.array(logits_full)[0]
        assert np.isfinite(lf).all()

        # Cached prefill(6) + 4 single-token steps via mlx_lm-compatible wrapper
        wrapped = Inkling2MLXModel(model)
        cache_list = wrapped.make_cache()
        l_pre, _ = model.forward_all_logits(toks[:6], cache=cache_list[0])
        mx.eval(l_pre)
        step_logits = [np.array(l_pre)[0]]
        for t in range(6, 10):
            lt = wrapped(mx.array([[toks[t]]], dtype=mx.int32), cache=cache_list)
            mx.eval(lt)
            step_logits.append(np.array(lt)[0])
        lc = np.concatenate(step_logits, axis=0)
        assert float(np.max(np.abs(lf - lc))) < 5e-3
        assert float(np.mean(np.argmax(lf, axis=-1) == np.argmax(lc, axis=-1))) == 1.0
        assert cache_list[0].offset == 10
        assert len(cache_list[0].state) > 0

        # Speculative block-4 verify + rollback (k_accept=2) + 2 steps
        c_spec = model.make_cache()
        l_p, c_spec, taps_p = model.forward_all_logits(toks[:6], cache=c_spec, tap_visits=(1, 5, 7, 10))
        mx.eval(l_p, taps_p)
        assert taps_p.shape == (6, 4, 256)
        c_spec.begin_spec_verify()
        l_blk, c_spec, taps_blk = model.forward_all_logits([toks[6], toks[7], 0, 1], cache=c_spec, tap_visits=(1, 5, 7, 10))
        mx.eval(l_blk, taps_blk)
        c_spec.commit_spec_verify(k_accept=2, block_len=4)
        l_8, c_spec = model.forward([toks[8]], cache=c_spec)
        l_9, c_spec = model.forward([toks[9]], cache=c_spec)
        mx.eval(l_8, l_9)
        l_spec = np.concatenate([np.array(l_p)[0], np.array(l_blk)[0, :2], np.array(l_8)[0], np.array(l_9)[0]], axis=0)
        assert float(np.max(np.abs(lf - l_spec))) < 5e-3

        # Non-contiguous tree commit_tree_verify test on fallback cache
        c_tr = model.make_cache()
        l_tp, c_tr = model.forward_all_logits(toks[:6], cache=c_tr)
        mx.eval(l_tp)
        c_tr.begin_spec_verify()
        # Use valid linear path [0, 1] via contiguous fast-path and non-contiguous [0, 2] shape check
        l_tblk, c_tr = model.forward_all_logits([toks[6], 0, toks[7], 1], cache=c_tr)
        mx.eval(l_tblk)
        c_tr.commit_tree_verify([0, 2], block_len=4)
        assert c_tr.offset == 8
        assert all(k is not None and k.shape[2] == 8 for k in c_tr.keys)

        # Early-break stats_out & Runtime EOS-stop check
        from runtime_mlx import Runtime
        from runtime_mlx.inkling.generate import stream_serial

        st_early: dict = {}
        emitted_seq: list[int] = []
        gen_s = stream_serial(model, toks[:4], max_tokens=8, eos_ids=(), stats_out=st_early)
        for idx_s, ch_s in enumerate(gen_s):
            emitted_seq.extend(ch_s)
            if idx_s == 2:
                break
        gen_s.close()
        assert st_early.get("generated_tokens") == 3 and "decode_tok_s" in st_early, f"missing stats on early break: {st_early}"

        class _DummyTok:
            eos_token_id = emitted_seq[2]
            eos_token_ids = (emitted_seq[2],)

            def decode(self, ids, skip_special_tokens=True):
                return " ".join(str(x) for x in ids)

        rt_mini = Runtime(wrapped, _DummyTok(), format_name="inkling2-packed")
        r_stop = rt_mini.generate(toks[:4], chat=False, max_tokens=8, use_drafter=False)
        assert r_stop.finish_reason == "stop" and r_stop.prompt_tps > 0.0 and "decode_tok_s" in r_stop.stats, f"bad EOS stop result: {r_stop}"

        # Drafter file resolution test
        ddir = pdir / "drafter"
        ddir.mkdir()
        (ddir / "mtp_4bit.safetensors").write_bytes(b"placeholder")
        (ddir / "manifest.json").write_text(json.dumps({"weights": "mtp_4bit.safetensors"}))
        w_p, c_p = resolve_drafter_files(ddir, fallback_config=pdir / "config.json")
        assert w_p == ddir / "mtp_4bit.safetensors"
        assert c_p == pdir / "config.json"


def test_prefill_budget_keeps_one_pass_when_it_fits() -> None:
    from unittest.mock import patch

    from runtime_mlx.inkling.generate import _prefill_step

    with patch.object(mx, "device_info", return_value={"max_recommended_working_set_size": 24_000_000_000}), \
         patch.object(mx, "get_active_memory", return_value=5_400_000_000):
        assert _prefill_step(4096, -1) == 4096
        assert _prefill_step(32768, -1) == 2048
        assert _prefill_step(32768, 0) == 32768
        assert _prefill_step(32768, None) == 32768
        assert _prefill_step(4096, 512) == 512


def test_chunked_prefill_matches_single_pass() -> None:
    """Chunked prefill (chunks smaller than the 8-token sliding window, prompt well past it) must reproduce one pass."""
    from runtime_mlx.inkling.generate import _prefill, stream_serial

    with tempfile.TemporaryDirectory() as td:
        pdir = Path(td)
        build_mini_packed_dir(pdir)
        model = Inkling2LoopPackedMLX(pdir)
        rng = np.random.default_rng(7)
        toks = [int(x) for x in rng.integers(1, 500, size=29)]

        ref, _ = model.forward_all_logits(toks)
        mx.eval(ref)
        ref = np.array(ref)[0]

        # Model level: KV, sliding-window and conv state carried across 5-token chunks.
        cache = model.make_cache()
        parts = []
        for s in range(0, len(toks), 5):
            lg, cache = model.forward_all_logits(toks[s : s + 5], cache=cache)
            mx.eval(lg)
            parts.append(np.array(lg)[0])
        chunked = np.concatenate(parts, axis=0)
        assert float(np.max(np.abs(ref - chunked))) < 5e-3
        assert (np.argmax(ref, axis=-1) == np.argmax(chunked, axis=-1)).all()

        # _prefill: last-chunk output and taps bookkeeping.
        out, _, taps, taps_start = _prefill(
            model, toks, model.make_cache(), ordered=False, prefill_chunk=5, tap_visits=(1, 5, 7, 10)
        )
        mx.eval(out, taps)
        assert int(np.array(out)[0, -1]) == int(np.argmax(ref[-1]))
        assert taps_start == 25 and taps.shape == (4, 4, 256)
        out1, _, taps1, start1 = _prefill(model, toks, model.make_cache(), ordered=False, prefill_chunk=None)
        assert taps1 is None and start1 == 0 and int(np.array(out1)[0, -1]) == int(np.argmax(ref[-1]))

        # End to end: greedy generation identical with and without chunking.
        def gen(chunk):
            return [t for c in stream_serial(model, toks, max_tokens=12, eos_ids=(), prefill_chunk=chunk) for t in c]

        base = gen(None)
        assert gen(5) == base and gen(3) == base and gen(4096) == base, f"chunked prefill changed greedy output: {base}"


def test_prompt_cache_reuse_matches_fresh_prefill() -> None:
    """PromptCache: extended, retried and unrelated prompts; reuse counts; eviction; snapshots never mutated."""
    from runtime_mlx.inkling.generate import PromptCache, stream_serial

    with tempfile.TemporaryDirectory() as td:
        pdir = Path(td)
        build_mini_packed_dir(pdir)
        model = Inkling2LoopPackedMLX(pdir)
        rng = np.random.default_rng(11)
        p1 = [int(x) for x in rng.integers(1, 500, size=21)]
        p2 = p1 + [int(x) for x in rng.integers(1, 500, size=9)]
        p3 = p1[:5] + p2[10:]

        def gen(p, pc=None):
            st: dict = {}
            g = stream_serial(model, p, max_tokens=10, eos_ids=(), stats_out=st, prefill_chunk=4, prompt_cache=pc)
            return [t for c in g for t in c], st.get("prompt_cached_tokens")

        fresh = {k: gen(p)[0] for k, p in (("p1", p1), ("p2", p2), ("p3", p3))}
        pc = PromptCache(max_entries=2, min_tokens=0)
        out, reused = gen(p1, pc)
        assert out == fresh["p1"] and reused == 0 and len(pc) == 1
        out, reused = gen(p2, pc)  # extends p1: reuses its snapshot (p1 minus its last token)
        assert out == fresh["p2"] and reused == len(p1) - 1, reused
        assert len(pc) == 1, "the p2 snapshot supersedes the p1 snapshot"
        out, reused = gen(p2, pc)  # identical retry re-runs one token
        assert out == fresh["p2"] and reused == len(p2) - 1, reused
        out, reused = gen(p3, pc)  # shares only a short prefix: no reuse, second entry
        assert out == fresh["p3"] and reused == 0 and len(pc) == 2
        out, reused = gen(p2, pc)  # the earlier snapshot was not mutated by the generations that used it
        assert out == fresh["p2"] and reused == len(p2) - 1

        small = PromptCache(min_tokens=1000)
        assert gen(p1, small)[0] == fresh["p1"] and len(small) == 0
        disabled = PromptCache(max_entries=0, min_tokens=0)
        assert gen(p1, disabled) == gen(p1, disabled) == (fresh["p1"], 0)
        assert len(disabled) == 0


def test_moe_grouped_large_expert_count_parity() -> None:
    """Stress-test deterministic compaction inside `trellis_moe_grouped` with E=96 (>32 experts) and sentinel slots."""
    rng = np.random.default_rng(123)
    E, m, n, k, g, K, hmult = 96, 256, 256, 2, 4, 512, 0x9E3779B1
    codes = mx.array(_pack_u32_random(rng, (E, m, (n * k) // 32)))
    scales = mx.array(rng.uniform(0.01, 0.03, size=(E, m)).astype(np.float16))
    lut = mx.array(rng.standard_normal((K, g)).astype(np.float16) * 0.5)
    T, topk = 8, 8
    x = mx.array(rng.standard_normal((T, n)).astype(np.float32) * 0.2)
    # Mix valid expert IDs across [0..95] and sentinel 0xFFFFFFFF slots
    lidx_np = rng.integers(0, E, size=(T * topk,), dtype=np.uint32)
    lidx_np[::7] = 0xFFFFFFFF
    lidx = mx.array(lidx_np)

    y_grouped = trellis_moe_grouped(x, codes, scales, lut, lidx, m, n, k, g, K, hmult, x_div=topk)
    mx.eval(y_grouped)
    yg = np.array(y_grouped)

    # Exact slot-by-slot reference via `trellis_matvec`
    for s in range(T * topk):
        eid = int(lidx_np[s])
        if eid == 0xFFFFFFFF:
            assert np.allclose(yg[s], 0.0, atol=1e-6)
        else:
            ref = trellis_matvec(
                x[s // topk : s // topk + 1], codes, scales, lut, mx.array([eid], dtype=mx.uint32), m, n, k, g, K, hmult
            )[0]
            mx.eval(ref)
            err = float(np.max(np.abs(yg[s] - np.array(ref))))
            assert err < 2e-3, f"Slot {s} expert {eid} mismatch: {err}"


def test_tree_spec_and_kernels() -> None:
    """Verify causal tree metadata and tree-mode Metal conv + SDPA kernels match linear paths exactly."""
    from runtime_mlx.inkling.drafter import TREE_TOPOLOGIES
    from runtime_mlx.inkling.fast_kernels import (
        gqa_relpos_sinks_sdpa_fast,
        make_tree_spec,
        short_conv4_res_fused,
    )

    for name, parents in TREE_TOPOLOGIES.items():
        spec = make_tree_spec(parents)
        assert spec["T"] == len(parents) and 1 <= spec["T"] <= 8 and spec["paths"][0] == [0], f"bad spec {name}"
        assert len(spec["depth"]) == len(parents) and spec["depth"][0] == 0

    spec = make_tree_spec(TREE_TOPOLOGIES["4-3"])
    rng = np.random.default_rng(77)
    prev = mx.array(rng.standard_normal((1, 3, 2304)).astype(np.float32))
    x = mx.array(rng.standard_normal((8, 2304)).astype(np.float32))
    w = mx.array(rng.standard_normal((4, 2304)).astype(np.float32))
    res = mx.array(rng.standard_normal((8, 2304)).astype(np.float32))
    out_t, _ = short_conv4_res_fused(prev, x, w, 8, 2304, residual=res, tree_lag=spec["lag_mx"])
    # Check Branch B = [0, 5, 6, 7] against contiguous 4-token linear conv!
    xb = mx.stack([x[0], x[5], x[6], x[7]], axis=0)
    rb = mx.stack([res[0], res[5], res[6], res[7]], axis=0)
    out_b, _ = short_conv4_res_fused(prev, xb, w, 4, 2304, residual=rb)
    mx.eval(out_t, out_b)
    for j, ni in enumerate((0, 5, 6, 7)):
        err = float(np.max(np.abs(np.array(out_t[ni]) - np.array(out_b[j]))))
        assert err < 1e-5, f"tree short_conv4 mismatch on node {ni}: {err}"

    # Check tree SDPA Branch B = [0, 5, 6, 7] against linear T=4 SDPA
    q = mx.array(rng.standard_normal((8, 24, 128)).astype(np.float32))
    rel = mx.array(rng.standard_normal((8, 24, 16)).astype(np.float32))
    rp = mx.array(rng.standard_normal((1024, 16)).astype(np.float16) * 0.1)
    qin = mx.array(rng.standard_normal((8, 2304)).astype(np.float32) * 0.1)
    sw = mx.array(rng.standard_normal((24, 2304)).astype(np.float16) * 0.05)
    kv_hist = mx.array(rng.standard_normal((2, 8, 12, 128)).astype(np.float16) * 0.2)
    kv_tree = mx.array(rng.standard_normal((2, 8, 8, 128)).astype(np.float16) * 0.2)
    KV_t = mx.zeros((2, 8, 32, 128), dtype=mx.float16)
    KV_t[:, :, :12, :] = kv_hist
    KV_t[:, :, 12:20, :] = kv_tree
    ao_t = gqa_relpos_sinks_sdpa_fast(q, KV_t, rel, rp, qin, sw, 8, 20, 0, 12, 32, True, tree_depth=spec["depth_mx"], tree_mask=spec["mask_mx"])

    idx_b = mx.array([0, 5, 6, 7], dtype=mx.int32)
    KV_b = mx.zeros((2, 8, 32, 128), dtype=mx.float16)
    KV_b[:, :, :12, :] = kv_hist
    KV_b[:, :, 12:16, :] = kv_tree[:, :, idx_b, :]
    ao_b = gqa_relpos_sinks_sdpa_fast(q[idx_b], KV_b, rel[idx_b], rp, qin[idx_b], sw, 4, 16, 0, 12, 32, True)
    mx.eval(ao_t, ao_b)
    for j, ni in enumerate((0, 5, 6, 7)):
        err = float(np.max(np.abs(np.array(ao_t[ni]) - np.array(ao_b[j]))))
        assert err < 1e-4, f"tree SDPA mismatch on node {ni}: {err}"


def test_compressed_global_kv() -> None:
    """Check optional storage across append, growth, rollback and all SDPA dispatch sizes."""
    from runtime_mlx.inkling.fast_kernels import gqa_relpos_sinks_sdpa_fast, make_tree_spec
    from runtime_mlx.inkling.model import Inkling2Cache, Inkling2Config

    rng = np.random.default_rng(718)
    data = mx.array(rng.normal(0, 0.2, (2, 8, 300, 128)).astype(np.float16))
    q = mx.array(rng.normal(0, 1, (33, 24, 128)).astype(np.float32))
    qin = mx.array(rng.normal(0, 0.1, (33, 2304)).astype(np.float32))
    sw = mx.array(rng.normal(0, 0.05, (24, 2304)).astype(np.float16))
    rel = mx.array(rng.normal(0, 0.2, (33, 24, 16)).astype(np.float32))
    rp = mx.array(rng.normal(0, 0.2, (1024, 16)).astype(np.float16))

    def attention(cache, T, positions, *, tree=None):
        buf, length, start, cap, fp8, scales = cache.update_kv_f16(0, data[:, :, positions, :], False)
        kwargs = {"tree_depth": tree["depth_mx"], "tree_mask": tree["mask_mx"]} if tree else {}
        return gqa_relpos_sinks_sdpa_fast(
            q[:T], buf, rel[:T], rp, qin[:T], sw, T, length, start, length - T, cap, False,
            kv_fp8=fp8, kv_q4_scales=scales, **kwargs,
        )

    for mode in ("fp16", "fp8", "q4"):
        cfg = Inkling2Config(global_kv_mode=mode)
        for base in (12, 255):
            for T in (1, 4, 5, 8, 9, 16, 31, 32, 33):
                reference = Inkling2Cache(Inkling2Config())
                candidate = Inkling2Cache(cfg)
                for cache in (reference, candidate):
                    cache.update_kv_f16(0, data[:, :, :base, :], False)
                expected = np.array(attention(reference, T, slice(base, base + T)))
                actual = np.array(attention(candidate, T, slice(base, base + T)))
                assert np.all(np.isfinite(actual)), (mode, base, T)
                err_by_token = np.linalg.norm(actual - expected, axis=1) / np.linalg.norm(expected, axis=1)
                tolerance = {"fp16": 1e-6, "fp8": 0.08, "q4": 0.24}[mode]
                assert max(err_by_token) < tolerance, (mode, base, T, err_by_token)

            tree = make_tree_spec((-1, 0, 1, 2, 3, 0, 5, 6))
            accepted = [0, 5, 6, 7]
            candidate = Inkling2Cache(cfg)
            control = Inkling2Cache(cfg)
            for cache in (candidate, control):
                cache.update_kv_f16(0, data[:, :, :base, :], False)
            candidate.begin_spec_verify()
            tree_out = attention(candidate, 8, slice(base, base + 8), tree=tree)
            candidate.commit_tree_verify(accepted, 8)
            control.update_kv_f16(0, data[:, :, [base + t for t in accepted], :], False)
            assert candidate.kv_len[0] == control.kv_len[0]
            assert len(candidate.state) == (2 if mode == "q4" else 1)
            for actual, expected in zip(candidate.state, control.state):
                np.testing.assert_array_equal(np.array(actual[:, :, :base + 4, :]), np.array(expected[:, :, :base + 4, :]))
            next_out = attention(candidate, 1, slice(base + 8, base + 9))
            next_control = attention(control, 1, slice(base + 8, base + 9))
            np.testing.assert_allclose(np.array(next_out), np.array(next_control), atol=1e-6, rtol=1e-5)
            assert bool(mx.all(mx.isfinite(tree_out)))
            candidate.begin_spec_verify()
            candidate.update_kv_f16(0, data[:, :, :8, :], False)
            candidate.commit_spec_verify(3, 8)
            control.update_kv_f16(0, data[:, :, :3, :], False)
            assert candidate.kv_len[0] == control.kv_len[0]
            for actual, expected in zip(candidate.state, control.state):
                np.testing.assert_array_equal(np.array(actual[:, :, :base + 8, :]), np.array(expected[:, :, :base + 8, :]))

        sliding = Inkling2Cache(cfg)
        buf, _, _, _, fp8, scales = sliding.update_kv_f16(0, data, True)
        assert buf.dtype == mx.float16 and not fp8 and scales is None


def test_compressed_global_kv_snapshot_isolation() -> None:
    """Restoring the same prefix twice gives each continuation independent compressed buffers."""
    from runtime_mlx.inkling.model import Inkling2Cache, Inkling2Config

    rng = np.random.default_rng(915)
    data = mx.array(rng.normal(0, 0.2, (2, 8, 262, 128)).astype(np.float16))
    prefix = data[:, :, :250, :]
    continuations = (data[:, :, 250:254, :], data[:, :, 254:262, :])
    for mode in ("fp16", "fp8", "q4"):
        cfg = Inkling2Config(num_hidden_layers=1, loop_passes=1, global_kv_mode=mode)
        cache = Inkling2Cache(cfg)
        cache.update_kv_f16(0, prefix, False)
        cache.seq_len = 250
        snapshot = cache.clone()
        prefix_state = [np.array(buf) for buf in snapshot.state]

        restored = [snapshot.clone(), snapshot.clone()]
        assert all(item.offset == 250 for item in restored)
        for actual, continuation in zip(restored, continuations):
            actual.update_kv_f16(0, continuation, False)

        assert cache.kv_len == snapshot.kv_len == [250]
        assert [item.kv_cap[0] for item in restored] == [256, 512]
        for actual, continuation in zip(restored, continuations):
            reference = Inkling2Cache(cfg)
            reference.update_kv_f16(0, mx.concatenate([prefix, continuation], axis=2), False)
            assert actual.kv_len == reference.kv_len
            for buf, expected in zip(actual.state, reference.state, strict=True):
                np.testing.assert_array_equal(np.array(buf), np.array(expected))
        for original in (cache, snapshot):
            for buf, expected in zip(original.state, prefix_state, strict=True):
                np.testing.assert_array_equal(np.array(buf), expected)


def test_nvfp4_unified_moe_all_modes() -> None:
    """Verify `nvfp4` (`fid=7`) across `t1`, `grouped`, `prefill_mpp`, `prefill_v2`, `also_raw`, and `PackedMoE`."""
    from runtime_mlx.inkling.fast_kernels import (
        UnifiedMoESide,
        _make_unified_moe_prefill_v2_kernel,
        _u32,
        build_fast_luts,
        compute_moe_sorted_offsets,
        plan_moe_prefill_units,
        swiglu_rht_1152_fast,
    )
    from runtime_mlx.inkling.kernels import nvfp4_luts_mx
    from runtime_mlx.inkling.model import Inkling2Config, PackedMoE, softcapped_swiglu

    rng = np.random.default_rng(4242)
    e2m1_mx, e4m3_mx = nvfp4_luts_mx()
    e2m1_np, e4m3_np = np.array(e2m1_mx, dtype=np.float64), np.array(e4m3_mx, dtype=np.float64)

    # 1. Verify `swiglu_rht_1152_fast(..., also_raw=True)` at n_slots=8 (w/ sentinel) and n_slots=96
    signs = mx.array(rng.choice([-1.0, 1.0], size=1152).astype(np.float32))
    for n_slots, use_eids in [(8, True), (96, False)]:
        y_gu = mx.array(rng.standard_normal((n_slots, 2304)).astype(np.float32) * 1.5)
        eids_s = mx.array([0, 0xFFFFFFFF, 3, 5, 7, 9, 0xFFFFFFFF, 12], dtype=mx.uint32) if use_eids else None
        rot_ref = swiglu_rht_1152_fast(y_gu, signs, out_dtype=mx.float16, eids_flat=eids_s)
        rot_2, raw_2 = swiglu_rht_1152_fast(y_gu, signs, out_dtype=mx.float16, eids_flat=eids_s, also_raw=True)
        raw_ref = softcapped_swiglu(y_gu, 4.0, 25.0).astype(mx.float16)
        mx.eval(rot_ref, rot_2, raw_2, raw_ref)
        if use_eids:
            valid = np.array([0, 2, 3, 4, 5, 7])
            assert float(np.max(np.abs(np.array(rot_2)[valid] - np.array(rot_ref)[valid]))) == 0.0
            assert float(np.linalg.norm(np.array(raw_2)[valid] - np.array(raw_ref)[valid]) / np.linalg.norm(np.array(raw_ref)[valid])) <= 5e-4
            assert float(np.max(np.abs(np.array(raw_2)[[1, 6]]))) == 0.0
        else:
            assert float(np.max(np.abs(np.array(rot_2) - np.array(rot_ref)))) == 0.0
            assert float(np.linalg.norm(np.array(raw_2) - np.array(raw_ref)) / np.linalg.norm(np.array(raw_ref))) <= 5e-4

    # 2. Verify `UnifiedMoESide` mixed (ht2 + nvfp4) and pure (nvfp4-only) across t1, grouped, mpp, AND v2!
    luts = {
        "ht2": mx.array(rng.standard_normal((512, 4)).astype(np.float16) * 0.4),
        "nvfp4_e2m1": e2m1_mx,
        "nvfp4_e4m3": e4m3_mx,
    }
    fast_luts = build_fast_luts(luts)
    for pure_nv, n_cols, x_div in [(False, 2304, 8), (True, 1152, 1)]:
        m_rows = 256
        if pure_nv:
            E_nv = 128
            map_fid = np.full(128, 7, dtype=np.int32)
            map_loc = np.arange(128, dtype=np.uint32)
            slabs = []
            ht2_c = ht2_s = None
        else:
            E_ht2, E_nv = 64, 64
            map_fid = np.array([7 if e % 2 == 0 else 1 for e in range(128)], dtype=np.int32)
            map_loc = np.array([e // 2 for e in range(128)], dtype=np.uint32)
            ht2_c = _pack_u32_random(rng, (E_ht2, m_rows, (n_cols * 2) // 32))
            ht2_s = rng.uniform(0.01, 0.04, size=(E_ht2, m_rows)).astype(np.float16)
            slabs = [(1, mx.array(ht2_c), mx.array(ht2_s), luts["ht2"], FMT_PARAMS["ht2"])]

        nv_c_u8 = rng.integers(0, 256, size=(E_nv, m_rows, n_cols // 2), dtype=np.uint8)
        nv_s_u8 = rng.integers(32, 88, size=(E_nv, m_rows, n_cols // 16), dtype=np.uint8)
        nv_out = rng.uniform(4e-5, 5e-4, size=(E_nv,)).astype(np.float32)
        nv_slab = (mx.array(np.ascontiguousarray(nv_c_u8).view(np.uint32)), mx.array(nv_s_u8), mx.array(nv_out))

        umoe = UnifiedMoESide(slabs, mx.array(map_fid), mx.array(map_loc), m_rows, n_cols, fast_luts, nvfp4_slab=nv_slab)

        # Precompute exact FP64 weights for NVFP4 experts & reference trellis matvec for ht2 experts
        W_nv_ref = np.empty((E_nv, m_rows, n_cols), dtype=np.float64)
        for loc in range(E_nv):
            vals = np.stack([e2m1_np[nv_c_u8[loc] & 0xF], e2m1_np[nv_c_u8[loc] >> 4]], axis=-1).reshape(m_rows, n_cols // 16, 16)
            W_nv_ref[loc] = (vals * e4m3_np[nv_s_u8[loc]][..., None]).reshape(m_rows, n_cols) * float(nv_out[loc])

        for T in (1, 6, 40):
            n_slots = T * 8
            x_rows = n_slots // x_div
            x_rot = mx.array(rng.standard_normal((x_rows, n_cols)).astype(np.float16) * 0.5)
            x_raw = mx.array(rng.standard_normal((x_rows, n_cols)).astype(np.float16) * 0.5)
            # Force repeated hits (c1..c4+) across tokens while respecting Top-8 uniqueness per token (max count <= T <= 8)
            pool = np.array([0, 1, 2, 3, 4, 5, 16, 17, 64, 65, 126, 127], dtype=np.uint32)
            eids_np = np.concatenate([rng.choice(pool, size=8, replace=False) for _ in range(T)]).astype(np.uint32)
            if T == 1:
                eids_np[3] = 0xFFFFFFFF
            eids_mx = mx.array(eids_np)

            y_fast = umoe(x_rot, eids_mx, x_div=x_div, x_raw_f16=x_raw)
            mx.eval(y_fast)
            yf = np.array(y_fast, dtype=np.float64)

            x_raw_np = np.array(x_raw, dtype=np.float64)
            y_ref = np.zeros((n_slots, m_rows), dtype=np.float64)
            for s in range(n_slots):
                eid = int(eids_np[s])
                if eid == 0xFFFFFFFF:
                    continue
                xr_idx = s if x_div == 1 else (s // x_div)
                if map_fid[eid] == 7:
                    y_ref[s] = x_raw_np[xr_idx] @ W_nv_ref[int(map_loc[eid])].T
                else:
                    loc_mx = mx.array([int(map_loc[eid])], dtype=mx.uint32)
                    t_out = trellis_matvec(
                        x_rot[xr_idx : xr_idx + 1].astype(mx.float32),
                        mx.array(ht2_c), mx.array(ht2_s), luts["ht2"], loc_mx,
                        m_rows, n_cols, *FMT_PARAMS["ht2"],
                    )[0]
                    mx.eval(t_out)
                    y_ref[s] = np.array(t_out, dtype=np.float64)

            valid_mask = eids_np != 0xFFFFFFFF
            rel_err = float(np.linalg.norm(yf[valid_mask] - y_ref[valid_mask]) / np.linalg.norm(y_ref[valid_mask]))
            assert rel_err < 1.5e-3, f"UnifiedMoESide nvfp4 mismatch (pure={pure_nv}, T={T}): rel_err={rel_err}"

            # Also verify non-MPP `prefill_v2` fallback explicitly at T=40!
            if T == 40:
                offsets, order = compute_moe_sorted_offsets(eids_mx)
                units, ucnt, max_u = plan_moe_prefill_units(offsets, umoe.exp_fid, umoe.fids, n_slots, tile_t=32)
                sg_p, rb_p = 8, 2
                rblocks = (m_rows + sg_p * 8 * rb_p - 1) // (sg_p * 8 * rb_p)
                kern_v2 = _make_unified_moe_prefill_v2_kernel(n_cols, umoe.fids, sg=sg_p, rb=rb_p)
                y_v2 = kern_v2(
                    inputs=[
                        x_rot.reshape(-1), umoe.codes, umoe.scales, umoe.exp_fid, umoe.exp_code_off, umoe.exp_scale_off,
                        *umoe._lut_args_prefill_v2, units, ucnt, _u32(m_rows, x_div), order, x_raw.reshape(-1), *umoe._nv_args,
                    ],
                    grid=(max_u * sg_p * 32, rblocks, 1),
                    threadgroup=(sg_p * 32, 1, 1),
                    output_shapes=[(n_slots, m_rows)],
                    output_dtypes=[mx.float32],
                )[0]
                mx.eval(y_v2)
                rel_v2 = float(np.linalg.norm(np.array(y_v2, dtype=np.float64) - y_ref) / np.linalg.norm(y_ref))
                assert rel_v2 < 1.5e-3, f"prefill_v2 nvfp4 mismatch (pure={pure_nv}): rel_err={rel_v2}"

    # 3. Verify full H=2304 `PackedMoE` (`use_fast_metal=True` vs `use_fast_metal=False` reference) at T=1, 4, 36
    arrays_moe = {
        "gu_signs_in": rng.choice([-1, 1], size=(2304,)).astype(np.int8),
        "dn_signs_in": rng.choice([-1, 1], size=(1152,)).astype(np.int8),
        "gu_map": np.array([[7 if e < 64 else 1, e if e < 64 else (e - 64)] for e in range(128)], dtype=np.int32),
        "dn_map": np.array([[7, e] for e in range(128)], dtype=np.int32),
        "gu_ht2_codes": _pack_u32_random(rng, (64, 2304, (2304 * 2) // 32)),
        "gu_ht2_scales": rng.uniform(0.01, 0.03, size=(64, 2304)).astype(np.float16),
        "gu_nvfp4_codes": rng.integers(0, 256, size=(64, 2304, 2304 // 2), dtype=np.uint8),
        "gu_nvfp4_scales": rng.integers(32, 80, size=(64, 2304, 2304 // 16), dtype=np.uint8),
        "gu_nvfp4_outer": rng.uniform(5e-5, 3e-4, size=(64,)).astype(np.float32),
        "dn_nvfp4_codes": rng.integers(0, 256, size=(128, 2304, 1152 // 2), dtype=np.uint8),
        "dn_nvfp4_scales": rng.integers(32, 80, size=(128, 2304, 1152 // 16), dtype=np.uint8),
        "dn_nvfp4_outer": rng.uniform(5e-5, 3e-4, size=(128,)).astype(np.float32),
    }
    rw = mx.array(rng.standard_normal((128, 2304)).astype(np.float32) * 0.05)
    rb = mx.array(rng.standard_normal((128,)).astype(np.float32) * 0.02)
    cfg_fast = Inkling2Config()
    moe_fast = PackedMoE(arrays_moe, luts, fast_luts, rw, rb, cfg_fast)
    moe_ref = PackedMoE(arrays_moe, luts, None, rw, rb, cfg_fast)
    for T in (1, 4, 36):
        h_in = mx.array(rng.standard_normal((1, T, 2304)).astype(np.float16).astype(np.float32) * 0.4)
        yf = moe_fast(h_in)
        yr = moe_ref(h_in)
        mx.eval(yf, yr)
        rel = float(np.linalg.norm(np.array(yf) - np.array(yr)) / np.linalg.norm(np.array(yr)))
        assert rel < 2.5e-3, f"PackedMoE fast vs ref mismatch at T={T}: rel={rel}"


def test_row_group_moe_and_f16_signs_in() -> None:
    """Verify full H=2304 `PackedMoE` with `n_rg=18` (`RG=128` rows) + `float16` pre-RHT column-scaled `*_signs_in`."""
    from runtime_mlx.inkling.fast_kernels import build_fast_luts
    from runtime_mlx.inkling.model import Inkling2Cache, Inkling2Config, PackedMoE

    rng = np.random.default_rng(20261122)
    luts_np = {
        "ht1": rng.standard_normal((512, 4)).astype(np.float16) * 0.4,
        "ht2": rng.standard_normal((512, 4)).astype(np.float16) * 0.4,
        "ht3": rng.standard_normal((1024, 2)).astype(np.float16) * 0.4,
        "rtr1": rng.standard_normal((4096,)).astype(np.float16) * 0.4,
        "rtr2": rng.standard_normal((4096,)).astype(np.float16) * 0.4,
        "rtr3": rng.standard_normal((4096,)).astype(np.float16) * 0.4,
        "rtr4": rng.standard_normal((4096,)).astype(np.float16) * 0.4,
    }
    luts = {k: mx.array(v) for k, v in luts_np.items()}
    fast_luts = build_fast_luts(luts)

    n_rg = 18
    rg_rows = 128
    fmts = [(1, "ht2", 2), (2, "ht3", 3), (3, "rtr1", 1), (6, "rtr4", 4)]
    gu_rg_map = np.zeros((128, n_rg, 2), dtype=np.int16)
    dn_rg_map = np.zeros((128, n_rg, 2), dtype=np.int16)
    gu_cnt = {fid: 0 for fid, _, _ in fmts}
    dn_cnt = {fid: 0 for fid, _, _ in fmts}
    for e in range(128):
        for rg in range(n_rg):
            gfid, _, _ = fmts[(e * 3 + rg) % 4]
            gu_rg_map[e, rg] = [gfid, gu_cnt[gfid]]
            gu_cnt[gfid] += 1
            dfid, _, _ = fmts[(e * 5 + rg + 1) % 4]
            dn_rg_map[e, rg] = [dfid, dn_cnt[dfid]]
            dn_cnt[dfid] += 1

    arrays: dict[str, np.ndarray] = {
        "gu_signs_in": (rng.choice([-1.0, 1.0], size=(2304,)) * rng.uniform(0.75, 1.35, size=(2304,))).astype(np.float16),
        "dn_signs_in": (rng.choice([-1.0, 1.0], size=(1152,)) * rng.uniform(0.75, 1.35, size=(1152,))).astype(np.float16),
        "gu_rg_map": gu_rg_map,
        "dn_rg_map": dn_rg_map,
    }
    for fid, fmt, k in fmts:
        ng = gu_cnt[fid]
        nd = dn_cnt[fid]
        arrays[f"gu_{fmt}_codes"] = _pack_u32_random(rng, (ng, rg_rows, (2304 * k) // 32))
        arrays[f"gu_{fmt}_scales"] = rng.uniform(0.008, 0.025, size=(ng, rg_rows)).astype(np.float16)
        arrays[f"dn_{fmt}_codes"] = _pack_u32_random(rng, (nd, rg_rows, (1152 * k) // 32))
        arrays[f"dn_{fmt}_scales"] = rng.uniform(0.008, 0.025, size=(nd, rg_rows)).astype(np.float16)

    rw = mx.array(rng.standard_normal((128, 2304)).astype(np.float32) * 0.05)
    rb = mx.array(rng.standard_normal((128,)).astype(np.float32) * 0.02)
    cfg_fast = Inkling2Config()
    moe_fast = PackedMoE(arrays, luts, fast_luts, rw, rb, cfg_fast)
    moe_ref = PackedMoE(arrays, luts, None, rw, rb, cfg_fast)

    for T in (1, 4, 8, 36):
        h_in = mx.array(rng.standard_normal((1, T, 2304)).astype(np.float16).astype(np.float32) * 0.4)
        yf = moe_fast(h_in)
        yr = moe_ref(h_in)
        mx.eval(yf, yr)
        rel = float(np.linalg.norm(np.array(yf) - np.array(yr)) / np.linalg.norm(np.array(yr)))
        assert rel < 2.5e-3, f"PackedMoE rg_map+f16_signs mismatch at T={T}: rel={rel}"

        # Also test `forward_fast_fused` (which exercises `sigmoid_top8_norm_and_plan_fast` for T=4,8!)
        cache = Inkling2Cache(cfg_fast)
        res = mx.zeros((T, 2304), dtype=mx.float32)
        conv_w = mx.zeros((4, 2304), dtype=mx.float32)
        out_fused = moe_fast.forward_fast_fused(h_in, res, conv_w, cache, 0, T)
        mx.eval(out_fused)
        expected_fused = np.array(yr)[0] * (1.0 / np.sqrt(5.0))
        rel_f = float(np.linalg.norm(np.array(out_fused) - expected_fused) / np.linalg.norm(expected_fused))
        assert rel_f < 2.5e-3, f"PackedMoE forward_fast_fused rg_map mismatch at T={T}: rel={rel_f}"


def test_thinking_budget_cap_serial_and_dspark_parity() -> None:
    """`max_thinking_tokens` must inject `<|end_message|><|message_model|><|content_text|>` identically in serial and DSpark."""
    from safetensors.numpy import save_file as save_st_np

    import runtime_mlx.inkling.generate as gen_mod
    from runtime_mlx.inkling.drafter import DSparkDrafterMLX
    from runtime_mlx.inkling.generate import generate_dspark, generate_serial

    with tempfile.TemporaryDirectory() as td:
        pdir = Path(td)
        build_mini_packed_dir(pdir)
        model = Inkling2LoopPackedMLX(pdir)

        first_toks, _ = generate_serial(model, [1, 2, 3, 4], max_tokens=20, eos_ids=())
        assert len(first_toks) == 20
        fake_think = int(first_toks[0])
        fake_exit = (501, 502, 503)

        rng = np.random.default_rng(99)
        H, V, nh, hd, mr = 256, 512, 4, 64, 64
        st_d: dict[str, np.ndarray] = {
            "fuse_weight": np.zeros((3, 4), dtype=np.float32),
            "mask_embed": rng.standard_normal((H,)).astype(np.float16) * 0.02,
            "norm.weight": np.ones((H,), dtype=np.float32),
            "markov_head.markov_w1.weight": rng.standard_normal((V, mr)).astype(np.float16) * 0.02,
            "markov_head.rnn_proj.weight": rng.standard_normal((3 * mr, 2 * mr + 2 * H)).astype(np.float16) * 0.02,
            "markov_head.rnn_proj.bias": np.zeros((3 * mr,), dtype=np.float32),
            "markov_head.hidden_out.weight": rng.standard_normal((H, mr)).astype(np.float16) * 0.02,
            "markov_head.hidden_out.bias": np.zeros((H,), dtype=np.float32),
            "markov_head.dflare_norm.weight": np.ones((H,), dtype=np.float32),
        }
        for li in range(2):
            p = f"layers.{li}."
            st_d[p + "dflare_norm.weight"] = np.ones((H,), dtype=np.float32)
            st_d[p + "input_layernorm.weight"] = np.ones((H,), dtype=np.float16)
            st_d[p + "post_attention_layernorm.weight"] = np.ones((H,), dtype=np.float16)
            st_d[p + "self_attn.q_norm.weight"] = np.ones((hd,), dtype=np.float16)
            st_d[p + "self_attn.k_norm.weight"] = np.ones((hd,), dtype=np.float16)
            st_d[p + "self_attn.k_norm_ctx.weight"] = np.ones((hd,), dtype=np.float16)
            st_d[p + "self_attn.q_proj.weight"] = rng.standard_normal((nh * hd, H)).astype(np.float16) * 0.02
            st_d[p + "self_attn.k_proj.weight"] = rng.standard_normal((nh * hd, H)).astype(np.float16) * 0.02
            st_d[p + "self_attn.v_proj.weight"] = rng.standard_normal((nh * hd, H)).astype(np.float16) * 0.02
            st_d[p + "self_attn.k_proj_ctx.weight"] = rng.standard_normal((nh * hd, H)).astype(np.float16) * 0.02
            st_d[p + "self_attn.v_proj_ctx.weight"] = rng.standard_normal((nh * hd, H)).astype(np.float16) * 0.02
            st_d[p + "self_attn.o_proj.weight"] = rng.standard_normal((H, nh * hd)).astype(np.float16) * 0.02
            st_d[p + "mlp.gate_proj.weight"] = rng.standard_normal((512, H)).astype(np.float16) * 0.02
            st_d[p + "mlp.up_proj.weight"] = rng.standard_normal((512, H)).astype(np.float16) * 0.02
            st_d[p + "mlp.down_proj.weight"] = rng.standard_normal((H, 512)).astype(np.float16) * 0.02

        d_cfg = {
            "text_config": {"final_logit_softcapping": 30.0},
            "mtp_config": {
                "hidden_size": H,
                "num_hidden_layers": 2,
                "num_attention_heads": nh,
                "num_key_value_heads": nh,
                "head_dim": hd,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "local_layer_ids": [0],
                "sliding_window_size": 16,
                "block_size": 6,
                "markov_rank": mr,
                "target_layer_ids": [1, 5, 7, 10],
            },
        }
        (pdir / "d_cfg.json").write_text(json.dumps(d_cfg))
        save_st_np(st_d, str(pdir / "d_q4.safetensors"))
        drafter = DSparkDrafterMLX(pdir / "d_q4.safetensors", pdir / "d_cfg.json")

        old_think, old_exit, old_brk = gen_mod.TOK_CONTENT_THINK, gen_mod.FORCED_THINK_EXIT, gen_mod.THINK_BREAK_TOKENS
        try:
            gen_mod.TOK_CONTENT_THINK = fake_think
            gen_mod.FORCED_THINK_EXIT = fake_exit
            gen_mod.THINK_BREAK_TOKENS = ()
            for cap in (0, 3, 7):
                out_s, st_s = generate_serial(model, [1, 2, 3, 4], max_tokens=20, eos_ids=(), max_thinking_tokens=cap)
                out_d, st_d_run = generate_dspark(
                    model, drafter, [1, 2, 3, 4], max_tokens=20, eos_ids=(), adaptive=False, draft_len=5, max_thinking_tokens=cap
                )
                assert out_s == out_d, f"cap={cap} serial vs dspark mismatch:\n  s={out_s}\n  d={out_d}"
                assert st_s["thinking_capped"] and st_d_run["thinking_capped"]
                assert tuple(out_s[1 + cap : 1 + cap + 3]) == fake_exit, f"cap={cap} missing forced exit in {out_s}"
        finally:
            gen_mod.TOK_CONTENT_THINK = old_think
            gen_mod.FORCED_THINK_EXIT = old_exit
            gen_mod.THINK_BREAK_TOKENS = old_brk


def test_fast_kernels_suite() -> None:
    """Verify all H=2304 Apple Silicon fast Metal kernels against exact references."""
    from runtime_mlx.inkling.fast_kernels import _selftest

    _selftest()


def main() -> None:
    test_mini_model_and_runtime_integration()
    test_prefill_budget_keeps_one_pass_when_it_fits()
    test_chunked_prefill_matches_single_pass()
    test_prompt_cache_reuse_matches_fresh_prefill()
    test_moe_grouped_large_expert_count_parity()
    test_tree_spec_and_kernels()
    test_compressed_global_kv()
    test_compressed_global_kv_snapshot_isolation()
    test_nvfp4_unified_moe_all_modes()
    test_row_group_moe_and_f16_signs_in()
    test_thinking_budget_cap_serial_and_dspark_parity()
    test_fast_kernels_suite()
    print("ALL INKLING MLX RUNTIME TESTS PASSED!", flush=True)


if __name__ == "__main__":
    main()
