"""Synthetic pack adapted from LookAtMyQuant's MIT-licensed test_inkling.py."""
import json
from pathlib import Path
import struct

import numpy as np

from inkling_cuda.packed import EXPERT_FORMATS, FORMATS

FMT_PARAMS = FORMATS
ID_TO_FMT = dict(enumerate(EXPERT_FORMATS))


def save_steering(path, tensors):
    header = {}
    chunks = []
    offset = 0
    for name, value in tensors.items():
        data = value.astype("<f4", copy=False).tobytes()
        header[name] = {"dtype": "F32", "shape": value.shape, "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    with path.open("wb") as fp:
        fp.write(struct.pack("<Q", len(encoded)))
        fp.write(encoded)
        fp.writelines(chunks)


def _pack_u32_random(rng, shape_m_wpr):
    return rng.integers(0, 2**32, size=shape_m_wpr, dtype=np.uint32)


def build_mini_packed_dir(out_dir: Path, full_size=False) -> None:
    """Write deterministic synthetic weights at test or release dimensions."""
    rng = np.random.default_rng(2026)
    H = 256
    I = 256
    n_q, n_kv, hd = 4, 2, 64
    d_rel, rel_ext = 16, 32
    V_emb, V_head = 512, 500
    n_layers = 6
    n_exp = 8
    top_k = 2
    core_start, core_end, block_size, rank = 2, 4, 2, 32
    local_layers = [0, 2, 4]
    exp_groups = [("ht2", 2), ("ht3", 2), ("rtr1", 2), ("rtr4", 2)]
    if full_size:
        H, I = 2304, 1152
        n_q, n_kv, hd = 24, 8, 128
        d_rel, rel_ext = 16, 1024
        V_emb, V_head = 201024, 200058
        n_layers, n_exp, top_k = 24, 128, 8
        core_start, core_end, block_size, rank = 6, 18, 6, 64
        local_layers = [i for i in range(n_layers) if i % block_size != block_size - 1]
        exp_groups = [("ht2", 16), ("ht3", 16), ("rtr1", 80), ("rtr4", 16)]
    cfg = {
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "text_config": {
            "model_type": "inkling2_loop_text",
            "hidden_size": H,
            "num_hidden_layers": n_layers,
            "vocab_size": V_emb,
            "unpadded_vocab_size": V_head,
            "num_attention_heads": n_q,
            "num_key_value_heads": n_kv,
            "head_dim": hd,
            "d_rel": d_rel,
            "rel_extent": rel_ext,
            "log_scaling_n_floor": 8192 if full_size else 4,
            "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-6,
            "qk_l2_clip_threshold": 22.627416997969522,
            "rel_l2_clip_threshold": 8.0,
            "local_layer_ids": local_layers,
            "sconv_kernel_size": 4,
            "conv_kernel_size": 4,
            "final_logit_softcapping": 30.0,
            "sliding_window_size": 1024 if full_size else 8,
            "n_layers_per_attn_res_block": block_size,
            "n_routed_experts": n_exp,
            "num_experts_per_tok": top_k,
            "intermediate_size": I,
            "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0,
            "route_scale": 8.0,
            "loop_config": {"core_start": core_start, "core_end": core_end, "num_passes": 3, "gate_rank": rank},
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

    st: dict[str, np.asarray] = {
        "model.embed_tokens.embed_norm.weight": np.asarray(rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32)),
        "model.final_input.query": np.asarray(rng.standard_normal((H,)).astype(np.float32) * 0.1),
        "model.norm.weight": np.asarray(rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32)),
        "model.message_input.query": np.asarray(rng.standard_normal((core_end - core_start, 4, H)).astype(np.float32) * 0.1),
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
        is_sliding = i in local_layers
        is_loop = core_start <= i < core_end
        if i > 0:
            st[p + "attention_input.query"] = np.asarray(
                rng.standard_normal((H,) if is_sliding else (3, H)).astype(np.float32) * 0.1
            )
        st[p + "mlp_input.query"] = np.asarray(rng.standard_normal((H,)).astype(np.float32) * 0.1)
        if is_sliding:
            st[p + "input_layernorm.weight"] = np.asarray(rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32))
        else:
            for ln in ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm"):
                st[p + f"{ln}.weight"] = np.asarray(rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32))
        st[p + "post_attention_layernorm.weight"] = np.asarray(rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32))
        st[p + "self_attn.q_proj.norm.scale"] = np.asarray(rng.uniform(0.8, 1.2, size=(n_q, hd)).astype(np.float32))
        st[p + "self_attn.relative_position_bias.norm.scale"] = np.asarray(rng.uniform(0.8, 1.2, size=(n_q, d_rel)).astype(np.float32))
        st[p + "self_attn.relative_position_bias.proj"] = np.asarray(rng.standard_normal((d_rel, rel_ext)).astype(np.float32) * 0.1)
        st[p + "self_attn.sink_proj.weight"] = np.asarray(rng.standard_normal((n_q, H)).astype(np.float32) * 0.05)
        if is_sliding:
            st[p + "self_attn.k_proj.sconv.conv1d.weight"] = np.asarray(rng.standard_normal((n_kv * hd, 1, 4)).astype(np.float32) * 0.2)
            st[p + "self_attn.v_proj.sconv.conv1d.weight"] = np.asarray(rng.standard_normal((n_kv * hd, 1, 4)).astype(np.float32) * 0.2)
        else:
            st[p + "self_attn.k_proj.sconv.static_weight"] = np.asarray(rng.standard_normal((4, n_kv * hd)).astype(np.float32) * 0.2)
            st[p + "self_attn.k_proj.sconv.weight_proj.weight"] = np.asarray(rng.standard_normal((n_kv * 4, H)).astype(np.float32) * 0.05)
            st[p + "self_attn.v_proj.sconv.static_weight"] = np.asarray(rng.standard_normal((4, n_kv * hd)).astype(np.float32) * 0.2)
            st[p + "self_attn.v_proj.sconv.weight_proj.weight"] = np.asarray(rng.standard_normal((n_kv * 4, H)).astype(np.float32) * 0.05)
        st[p + "post_attention_convolution.conv1d.weight"] = np.asarray(rng.standard_normal((H, 1, 4)).astype(np.float32) * 0.2)
        st[p + "post_mlp_convolution.conv1d.weight"] = np.asarray(rng.standard_normal((H, 1, 4)).astype(np.float32) * 0.2)
        if is_loop:
            for gn in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"):
                st[p + f"{gn}.down_proj.weight"] = np.asarray(rng.standard_normal((rank, 2 * H)).astype(np.float32) * 0.05)
                st[p + f"{gn}.forget_gate_proj.weight"] = np.asarray(rng.standard_normal((H, rank)).astype(np.float32) * 0.05)
                st[p + f"{gn}.message_gate_proj.weight"] = np.asarray(rng.standard_normal((H, rank)).astype(np.float32) * 0.05)
        st[p + "mlp.gate.weight"] = np.asarray(rng.standard_normal((n_exp, H)).astype(np.float32) * 0.1)
        st[p + "mlp.gate.e_score_correction_bias"] = np.asarray(rng.standard_normal((n_exp,)).astype(np.float32) * 0.02)

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
        fmt_to_id = {v: k for k, v in ID_TO_FMT.items()}
        offset = 0
        for fmt, count in exp_groups:
            k, g, K, hmult = FMT_PARAMS[fmt]
            l_arrs[f"gu_{fmt}_codes"] = _pack_u32_random(rng, (count, 2 * I, (H * k) // 32))
            l_arrs[f"gu_{fmt}_scales"] = rng.uniform(0.01, 0.03, size=(count, 2 * I)).astype(np.float16)
            l_arrs[f"dn_{fmt}_codes"] = _pack_u32_random(rng, (count, H, (I * k) // 32))
            l_arrs[f"dn_{fmt}_scales"] = rng.uniform(0.01, 0.03, size=(count, H)).astype(np.float16)
            for loc in range(count):
                e = offset + loc
                gu_map[e] = [fmt_to_id[fmt], loc]
                dn_map[e] = [fmt_to_id[fmt], loc]
            offset += count
        l_arrs["gu_map"] = gu_map
        l_arrs["dn_map"] = dn_map
        np.savez(out_dir / f"layer_{i:02d}_packed.npz", **l_arrs)
        (out_dir / f"layer_{i:02d}_meta.json").write_text(json.dumps(l_meta, indent=2))

    save_steering(out_dir / "steering.safetensors", st)
    manifest = {
        "format": "inkling2-packed-v1-sqyfr",
        "packed_text_gb": 0.01,
        "steering_tensors": len(st),
        "luts": {k: list(v.shape) for k, v in luts.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
