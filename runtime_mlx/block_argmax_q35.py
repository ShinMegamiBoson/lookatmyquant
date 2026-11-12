"""Reduce the packed output head to the best token without storing full logits."""

from functools import cache

import mlx.core as mx

try:
    from . import block_q35 as B
except ImportError:
    import block_q35 as B


_KEEP_BEST = """
inline void keep_best(thread float& best, thread uint& id, float value, uint candidate) {
    if (value > best || (value == best && candidate < id)) {
        best = value;
        id = candidate;
    }
}
"""


_HEAD_REDUCE = """
    threadgroup float simd_scores[SG * TB * 8];
    threadgroup uint simd_ids[SG * TB * 8];
    // The matrix unit's output pairs encode the token position in lane bits 0 and 3.
    // Reduce across the other three bits, which encode the vocabulary row.
    const uint pair = ((lane & 8u) >> 1) + ((lane & 1u) << 1);
    #pragma unroll
    for (uint j = 0; j < 2u * TB; j++) {
        float score = best[j];
        uint id = ids[j];
        for (uint bit = 2u; bit <= 16u; bit <<= 1) {
            if (bit == 8u) continue;
            const float other = simd_shuffle_xor(score, (ushort)bit);
            const uint other_id = simd_shuffle_xor(id, (ushort)bit);
            keep_best(score, id, other, other_id);
        }
        if ((lane & 22u) == 0u) {
            const uint pos = (j / 2u) * 8u + pair + (j % 2u);
            simd_scores[pos * SG + simd] = score;
            simd_ids[pos * SG + simd] = id;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint pos = simd; pos < 8u * TB; pos += SG) {
        const float v = lane < SG ? simd_scores[pos * SG + lane] : -INFINITY;
        const uint id = lane < SG ? simd_ids[pos * SG + lane] : 0xffffffffu;
        const float top = simd_max(v);
        const uint winner = simd_min(v == top ? id : 0xffffffffu);
        if (lane == 0u) {
            scores[pos * GROUPS + tg] = top;
            tokens[pos * GROUPS + tg] = winner;
        }
    }
"""


@cache
def _top1_kernel(n, kind, widths, sg, rb, tb, cs, fold):
    variants = {w: B._block_rows_fn(*w, kind, top1=True) for w in widths}
    header = (
        B._HEADER
        + f"#define SG {sg}u\n#define RB {rb}u\n#define TB {tb}u\n#define CS {cs}\n#define FOLD {fold}u\n"
        + _KEEP_BEST
        + "\n".join(source for _, source in variants.values())
    )
    red_size = sg * rb * tb * 64 if cs > 1 else 1
    body = [
        "const uint tg = threadgroup_position_in_grid.x, gi = group_of[2 * tg], tgl = group_of[2 * tg + 1];",
        f"const device uint* d = dims + gi * {B.D};",
        "const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;",
        f"threadgroup float red[{red_size}];",
        "float best[2 * TB]; uint ids[2 * TB];",
        "for (uint j = 0; j < 2u * TB; j++) { best[j] = -INFINITY; ids[j] = 0xffffffffu; }",
        f"const device float* xv = x + (ulong)d[7] * {8 * tb}u;",
        "switch (d[6]) {",
    ]
    for (bits, scalar, group), (name, _) in variants.items():
        code = B.F.width_code(bits, group, scalar)
        body.append(
            f"case {code}u: {name}(xv, codes + d[3], d[2], d[1], scales + d[4], map + d[4], "
            f"tables + d[5], best, ids, red, {n}u, tgl, d[0], simd, lane); break;"
        )
    body += ["}", _HEAD_REDUCE]
    width_key = "_".join(f"{k}{int(s)}{g}" for k, s, g in widths)
    return mx.fast.metal_kernel(
        name=f"lamq_top1_{n}_{kind}_{sg}_{rb}_{tb}_{cs}_{fold}_{width_key}",
        input_names=["x", "codes", "scales", "map", "tables", "dims", "group_of"],
        output_names=["scores", "tokens"],
        source="\n".join(body),
        header=header,
    )


_finish = mx.fast.metal_kernel(
    name="lamq_finish_top1",
    input_names=["scores", "tokens"],
    output_names=["out"],
    header=_KEEP_BEST,
    source="""
    const uint pos = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    float best = -INFINITY;
    uint id = 0xffffffffu;
    for (uint g = lane; g < GROUPS; g += 32u) {
        const uint i = pos * GROUPS + g;
        keep_best(best, id, scores[i], tokens[i]);
    }
    const float top = simd_max(best);
    const uint winner = simd_min(best == top ? id : 0xffffffffu);
    if (lane == 0u) out[pos] = top == -INFINITY ? 0u : winner;
    """,
)


def argmax(head, x, tb):
    """Return one winning row per vector using the product’s launch schedule."""
    if tb not in head.launch:
        head.launch[tb] = head._launch(tb)
    _, args, groups, sg = head.launch[tb]
    rb, _, cs = head.shape(tb)
    red = sg * rb * tb * 64 if cs > 1 else 1
    size = 8 * tb
    if (red + 2 * size * sg) * 4 > 32768:
        return mx.argmax(head(x, tb), axis=-1)
    kernel = _top1_kernel(head.n, head.kind, head.p.widths, sg, rb, tb, cs, B.FOLD)
    scores, tokens = kernel(
        inputs=[x.reshape(-1), *args[:-1]],
        template=[("GROUPS", groups)],
        grid=(groups * sg * 32, 1, 1),
        threadgroup=(sg * 32, 1, 1),
        output_shapes=[(size, groups), (size, groups)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    return _finish(
        inputs=[scores, tokens],
        template=[("GROUPS", groups)],
        grid=(size * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(size,)],
        output_dtypes=[mx.uint32],
    )[0]
