"""Keep the two best rows inside the packed draft head's matrix product."""

from functools import cache

import mlx.core as mx

if __package__ == "strategies":
    import block_q35 as B
else:
    from .. import block_q35 as B


_KEEP_TOP2 = """
inline void keep_top2(thread float2& best, thread uint2& ids, float value, uint id) {
    if (id == ids.x || id == ids.y) return;
    if (value > best.x || (value == best.x && id < ids.x)) {
        best = float2(value, best.x);
        ids = uint2(id, ids.x);
    } else if (value > best.y || (value == best.y && id < ids.y)) {
        best.y = value;
        ids.y = id;
    }
}

inline void reduce_top2(thread float2& best, thread uint2& ids) {
    // Each lane owns a disjoint set of rows. Only the lane that owns the
    // winner must offer its second candidate for the runner-up reduction.
    const float best_score = simd_max(best.x);
    const uint best_id = simd_min(best.x == best_score ? ids.x : 0xffffffffu);
    const float next = ids.x == best_id ? best.y : best.x;
    const uint next_id = ids.x == best_id ? ids.y : ids.x;
    const float second_score = simd_max(next);
    const uint second_id = simd_min(next == second_score ? next_id : 0xffffffffu);
    best = float2(best_score, second_score);
    ids = uint2(best_id, second_id);
}
"""


_HEAD_REDUCE = """
    threadgroup float2 simd_scores[SG * TB * 8];
    threadgroup uint2 simd_ids[SG * TB * 8];
    // As in the packed argmax, lane bits 0 and 3 encode the token pair.
    const uint pair = ((lane & 8u) >> 1) + ((lane & 1u) << 1);
    #pragma unroll
    for (uint j = 0; j < 2u * TB; j++) {
        float2 score = best[j];
        uint2 id = ids[j];
        for (uint bit = 2u; bit <= 16u; bit <<= 1) {
            if (bit == 8u) continue;
            const float2 other = simd_shuffle_xor(score, (ushort)bit);
            const uint2 other_id = simd_shuffle_xor(id, (ushort)bit);
            keep_top2(score, id, other.x, other_id.x);
            keep_top2(score, id, other.y, other_id.y);
        }
        if ((lane & 22u) == 0u) {
            const uint pos = (j / 2u) * 8u + pair + (j % 2u);
            simd_scores[pos * SG + simd] = score;
            simd_ids[pos * SG + simd] = id;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint pos = simd; pos < 8u * TB; pos += SG) {
        float2 score = lane < SG ? simd_scores[pos * SG + lane] : float2(-INFINITY);
        uint2 id = lane < SG ? simd_ids[pos * SG + lane] : uint2(0xffffffffu);
        reduce_top2(score, id);
        if (lane == 0u) {
            const uint out = pos * GROUPS + tg;
            ((device float2*)scores)[out] = score;
            ((device uint2*)tokens)[out] = id;
        }
    }
"""


def _source(n, kind, widths, sg, rb, tb, cs, fold):
    variants = {w: B._block_rows_fn(*w, kind, top1=True) for w in widths}
    rows = []
    for _, original in variants.values():
        assert original.count("thread float* best, thread uint* ids") == 1
        assert original.count("keep_best(") == 2
        rows.append(
            original.replace(
                "thread float* best, thread uint* ids",
                "thread float2* best, thread uint2* ids",
            ).replace("keep_best(", "keep_top2(")
        )
    header = (
        B._HEADER
        + f"#define SG {sg}u\n#define RB {rb}u\n#define TB {tb}u\n#define CS {cs}\n#define FOLD {fold}u\n"
        + _KEEP_TOP2
        + "\n".join(rows)
    )
    red_size = sg * rb * tb * 64 if cs > 1 else 1
    body = [
        "const uint tg = threadgroup_position_in_grid.x, gi = group_of[2 * tg], tgl = group_of[2 * tg + 1];",
        f"const device uint* d = dims + gi * {B.D};",
        "const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;",
        f"threadgroup float red[{red_size}];",
        "float2 best[2 * TB]; uint2 ids[2 * TB];",
        "for (uint j = 0; j < 2u * TB; j++) { best[j] = float2(-INFINITY); ids[j] = uint2(0xffffffffu); }",
        f"const device float* xv = x + (ulong)d[7] * {8 * tb}u;",
        "switch (d[6]) {",
    ]
    for (bits, scalar, group), (name, _) in variants.items():
        code = B.F.width_code(bits, group, scalar)
        body.append(
            f"case {code}u: {name}(xv, codes + d[3], d[2], d[1], scales + d[4], map + d[4], "
            f"tables + d[5], best, ids, red, {n}u, tgl, d[0], simd, lane); break;"
        )
    body.extend(["}", _HEAD_REDUCE])
    return header, "\n".join(body)


@cache
def _kernel(n, kind, widths, sg, rb, tb, cs, fold):
    header, body = _source(n, kind, widths, sg, rb, tb, cs, fold)
    width_key = "_".join(f"{k}{int(s)}{g}" for k, s, g in widths)
    return mx.fast.metal_kernel(
        name=f"lamq_top2_{n}_{kind}_{sg}_{rb}_{tb}_{cs}_{fold}_{width_key}",
        input_names=["x", "codes", "scales", "map", "tables", "dims", "group_of"],
        output_names=["scores", "tokens"],
        source=body,
        header=header,
    )


_FINISH = """
    const uint pos = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    float2 best = float2(-INFINITY);
    uint2 ids = uint2(0xffffffffu);
    for (uint g = lane; g < GROUPS; g += 32u) {
        const uint i = pos * GROUPS + g;
        const float2 score = ((const device float2*)scores)[i];
        const uint2 id = ((const device uint2*)tokens)[i];
        keep_top2(best, ids, score.x, id.x);
        keep_top2(best, ids, score.y, id.y);
    }
    reduce_top2(best, ids);
    if (lane == 0u) {
        ((device float2*)out_scores)[pos] = best;
        ((device uint2*)out_ids)[pos] = ids;
    }
"""


_finish = mx.fast.metal_kernel(
    name="lamq_finish_top2",
    input_names=["scores", "tokens"],
    output_names=["out_scores", "out_ids"],
    header=_KEEP_TOP2,
    source=_FINISH,
)


def _dense_top2(logits):
    first = mx.argmax(logits, axis=-1, keepdims=True)
    other = mx.put_along_axis(logits, first, mx.array(-float("inf")), axis=-1)
    second = mx.argmax(other, axis=-1, keepdims=True)
    # If all other scores are -inf, masking can leave the same ID tied for first.
    second = mx.where(second == first, (first == 0).astype(first.dtype), second)
    ids = mx.concatenate([first, second], axis=-1)
    return mx.take_along_axis(logits, ids, axis=-1), ids


def top2(head, x, tb):
    """Return FP32 scores and uint32 row IDs, both shaped (8 * tb, 2)."""
    if tb not in head.launch:
        head.launch[tb] = head._launch(tb)
    _, args, groups, sg = head.launch[tb]
    rb, _, cs = head.shape(tb)
    red_size = sg * rb * tb * 64 if cs > 1 else 1
    size = 8 * tb
    if (red_size + 4 * size * sg) * 4 > 32768:
        return _dense_top2(head(x, tb))
    kernel = _kernel(head.n, head.kind, head.p.widths, sg, rb, tb, cs, B.FOLD)
    scores, tokens = kernel(
        inputs=[x.reshape(-1), *args[:-1]],
        template=[("GROUPS", groups)],
        grid=(groups * sg * 32, 1, 1),
        threadgroup=(sg * 32, 1, 1),
        output_shapes=[(size, groups, 2), (size, groups, 2)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    return _finish(
        inputs=[scores, tokens],
        template=[("GROUPS", groups)],
        grid=(size * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(size, 2), (size, 2)],
        output_dtypes=[mx.float32, mx.uint32],
    )


def _select(scores, ids):
    first, second = ids[:7, 0], ids[:5, 1]
    eligible = scores[:5, 0] - scores[:5, 1] <= 1.0
    depth = mx.min(mx.where(eligible, mx.arange(5, dtype=mx.int32), 5))
    forked = depth < 5
    safe = mx.minimum(depth, 4)
    alternate = mx.stack([mx.take(second, safe), mx.take(first, safe + 1)])
    tokens = mx.concatenate([first[:5], mx.where(forked, alternate, first[5:])])
    return tokens, mx.where(forked, depth, -1).astype(mx.int32)


def propose(drafter, states, anchor):
    """Choose a fused top-two proposal and its fork position."""
    assert states.shape[0] == 7
    compact = drafter.proposal_head
    table = drafter.block.table
    pad = mx.concatenate(
        [states.astype(mx.float32), mx.zeros((1, table.n), mx.float32)]
    )
    rotated = table.rot.fwd(mx.broadcast_to(pad[:, None, :], (8, table.nb, table.n)))
    head = drafter.block.head if compact is None else compact.head
    tokens, fork = _select(*top2(head, rotated.transpose(1, 2, 0), 1))
    if compact is not None:
        tokens = compact.ids[tokens]
    anchor = mx.array(anchor) if isinstance(anchor, int) else anchor
    return mx.concatenate(
        [anchor.reshape(1).astype(mx.int32), tokens.astype(mx.int32)]
    ), fork
