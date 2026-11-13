"""Attention and KV cache updates for a single fork in an eight-slot check.

Slots 0 through 5 form the main path. With a fork, slot 6 follows the parent
in `fork` and slot 7 follows slot 6. A GPU scalar of -1 makes all slots linear.
"""

import mlx.core as mx

WIDTH = 8
ALT = 6


def _layout(offset, fork):
    slots = mx.arange(WIDTH, dtype=mx.int32)
    fork = fork.reshape(())
    branch = (fork >= 0) & (slots >= ALT)
    positions = offset + mx.where(branch, slots + fork - (ALT - 1), slots)

    query = slots[:, None]
    key = slots[None, :]
    linear = key <= query
    alternate = (key <= fork) | ((key >= ALT) & linear)
    ancestors = mx.where(branch[:, None], alternate, linear)
    return positions, mx.pad(ancestors, [(0, 0), (offset, 0)], constant_values=True)


def attention(block, a, xn, cache, fork):
    """Check all eight attention queries against exactly their causal ancestors."""
    offset = cache.offset
    # All attention layers have the same offset before processing this block.
    layout = getattr(block, "_tree_attention_layout", None)
    if layout is None or layout[0] is not fork or layout[1] != offset:
        layout = (fork, offset, *_layout(offset, fork))
        block._tree_attention_layout = layout
    _, _, positions, mask = layout

    n = xn.shape[-1]
    x = xn.reshape(1, WIDTH, n)
    queries, gate = mx.split(
        a.q_proj(x).reshape(1, WIDTH, a.num_attention_heads, -1), 2, axis=-1
    )
    keys = a.k_norm(a.k_proj(x).reshape(1, WIDTH, a.num_key_value_heads, -1))
    values = (
        a.v_proj(x).reshape(1, WIDTH, a.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    )
    queries = a.q_norm(queries)

    def rope_by_slot(array):
        # A separate batch entry for each slot lets the native RoPE use its
        # position without moving any of the projected keys or queries.
        _, _, heads, dim = array.shape
        array = a.rope(array.reshape(WIDTH, heads, 1, dim), offset=positions)
        return array.transpose(2, 1, 0, 3)

    queries, keys = rope_by_slot(queries), rope_by_slot(keys)
    keys, values = cache.update_and_fetch(keys, values)
    out = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=a.scale, mask=mask
    )
    out = out.transpose(0, 2, 1, 3).reshape(1, WIDTH, -1)
    return (
        a.o_proj(out * mx.sigmoid(gate.reshape(1, WIDTH, -1)))
        .reshape(WIDTH, n)
        .astype(mx.float32)
    )


def keep(cache, kept, selected):
    """Commit a prefix or gather the selected slots in chronological order."""
    assert 1 <= kept <= WIDTH and cache.offset >= WIDTH
    if selected is None:
        cache.trim(WIDTH - kept)
        return

    assert len(selected) == kept and all(0 <= slot < WIDTH for slot in selected)
    prefix = next((i for i, slot in enumerate(selected) if slot != i), kept)
    start = cache.offset - WIDTH
    if prefix == kept:
        cache.trim(WIDTH - kept)
        return

    indices = mx.array([start + slot for slot in selected[prefix:]], dtype=mx.int32)
    keys, values = (
        mx.take(cache.keys, indices, axis=2),
        mx.take(cache.values, indices, axis=2),
    )
    cache.trim(WIDTH - prefix)
    cache.update_and_fetch(keys, values)
