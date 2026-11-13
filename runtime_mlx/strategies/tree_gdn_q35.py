"""Gated delta recurrence for an eight-node draft with one two-token branch.

Nodes 0 through 5 form the main chain. A GPU int32 scalar `fork` names the
parent of node 6 (0 through 4), and node 7 follows node 6. A value of -1
uses the normal eight-node chain. The returned projections remain in node
order so the caller can compact the accepted branch before carrying it on.
"""

import mlx.core as mx

if __package__ == "strategies":
    import block_q35 as B
else:
    from .. import block_q35 as B


def _replace(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


_pre_source = _replace(
    B._GDN_PRE_T,
    "const int u = (int)t + (int)j - 3, p = kept + u;",
    """int u = (int)t + (int)j - 3;
        // Slot 6's logical predecessor is fork[0], not slot 5. Earlier
        // ancestors move by the same offset, including negative indices
        // into the committed context. Slots 6 and 7 stay in place.
        if (fork[0] >= 0 && t >= 6u && u < 6) u += fork[0] - 5;
        const int p = kept + u;""",
)
_pre = mx.fast.metal_kernel(
    name="lamq_gdn_pre_fork8",
    input_names=["y", "yp", "cs", "w", "cnt", "fork"],
    output_names=["out", "cs_out"],
    source=_pre_source,
)


_step_source = _replace(
    B._GDN_STEP_T,
    "float state[n_per_t];",
    """float state[n_per_t], fork_state[n_per_t];
    const int parent = fork[0];""",
)
_step_source = _replace(
    _step_source,
    "const device float* xt = xs + (ulong)t * CD;",
    """if (part == 1u && t == 6u && parent >= 0)
                for (int i = 0; i < n_per_t; ++i) state[i] = fork_state[i];
            const device float* xt = xs + (ulong)t * CD;""",
)
_step_source = _replace(
    _step_source,
    "o = simd_sum(o);",
    """if (part == 1u && (int)t == parent)
                for (int i = 0; i < n_per_t; ++i) fork_state[i] = state[i];
            o = simd_sum(o);""",
)
_steps = mx.fast.metal_kernel(
    name="lamq_gdn_steps_fork8",
    input_names=["x", "gates", "xp", "gatesp", "state_in", "cnt", "fork"],
    output_names=["y", "state_mid"],
    source=_step_source,
)


def linear(block, rec, xn, cache, carry, fork):
    """Evaluate a layer and leave its cache at the state before this tree."""
    width = 8
    fork = fork.reshape(1)
    m, s_in = rec["m"], rec["s_in"]
    y = s_in(xn, width)
    q_off = s_in.slices["in_proj_qkv"][0]
    z_off = s_in.slices["in_proj_z"][0]
    ab = B._proj_t(
        inputs=[rec["w_ab"], xn, rec["A_log"], rec["dt_bias"]],
        template=[("N", s_in.n), ("HV", m.num_v_heads)],
        grid=(32, 2 * m.num_v_heads, width),
        threadgroup=(32, 1, 1),
        output_shapes=[(width, 2 * m.num_v_heads)],
        output_dtypes=[mx.float32],
    )[0]
    cs = cache[0] if cache[0] is not None else mx.zeros((3 * m.conv_dim,), mx.float32)
    state = cache[1]
    if state is None:
        state = mx.zeros((1, m.num_v_heads, m.head_v_dim, m.head_k_dim), mx.float32)
    state = state.astype(mx.float32)
    before, kept = carry if carry is not None else (None, 0)
    where = rec["t"] + [("STRIDE", s_in.size)]
    conv = where + [("OFF", q_off)]
    pre, cache[0] = _pre(
        inputs=[
            y,
            before["y"] if before else y,
            cs.reshape(-1).astype(mx.float32),
            rec["g"].conv_w,
            block._count(kept),
            fork,
        ],
        template=conv,
        grid=(m.conv_dim, width, 1),
        threadgroup=(m.head_k_dim, 1, 1),
        output_shapes=[(width, m.conv_dim), (1, 3, m.conv_dim)],
        output_dtypes=[mx.float32, mx.float32],
    )
    out, cache[1] = _steps(
        inputs=[
            pre,
            ab,
            before["pre"] if before else pre,
            before["ab"] if before else ab,
            state,
            block._count(kept, width),
            fork,
        ],
        template=rec["t"],
        grid=(32, m.head_v_dim, m.num_v_heads),
        threadgroup=(32, 4, 1),
        output_shapes=[(width, m.value_dim), state.shape],
        output_dtypes=[mx.float32, mx.float32],
    )
    out = B._gdn_post_t(rec["g"].eps)(
        inputs=[out, y, rec["norm_w"]],
        template=where + [("OFF", z_off)],
        grid=(m.value_dim, width, 1),
        threadgroup=(m.head_v_dim, 1, 1),
        output_shapes=[(width, m.value_dim)],
        output_dtypes=[mx.float32],
    )[0]
    return rec["s_out"](out, width), {"y": y, "pre": pre, "ab": ab, "where": conv}
