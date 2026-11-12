"""Score a compact proposal vocabulary using the target's packed rows.

The Qwen3.5 BPE prefix and added tokens cover nearly all tokens in the held-out
answers. This head is only for DSpine's proposals; the target verifies against
its complete vocabulary.
"""

from copy import copy

import mlx.core as mx
import numpy as np

try:
    from . import block_q35 as B
except ImportError:  # run as a script beside block_q35.py
    import block_q35 as B


PREFIX_ROWS = 96_000


class DraftVocabulary:
    """Use the target's row codes, scales and rotations for a subset of token IDs."""

    def __init__(self, block, keep):
        self.table = block.table
        original = self.table.product
        codes, scales, rowmap, tables, dims, groups = original.args
        rowmap = np.asarray(rowmap)
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != (self.table.V,) or not keep.any():
            raise ValueError("the shortlist must select at least one vocabulary ID")
        self.keep = keep

        sorted_ids = np.flatnonzero(keep).astype(np.int32)
        self.ids = mx.array(sorted_ids)
        selected_codes, selected_scales, selected_rows, selected_dims = [], [], [], []
        code_at = row_at = 0
        for row in np.array(dims).reshape(-1, B.D):
            _, n, wpr, off, pos, *_ = (int(x) for x in row)
            ids = rowmap[pos : pos + n]
            take = np.flatnonzero(keep[ids]).astype(np.int32)
            if not len(take):
                continue
            sel = mx.array(take)
            selected_codes.append(
                codes[off : off + n * wpr].reshape(n, wpr)[sel].reshape(-1)
            )
            selected_scales.append(scales[pos : pos + n][sel])
            selected_rows.append(
                np.searchsorted(sorted_ids, ids[take]).astype(np.int32)
            )
            row[1], row[3], row[4] = len(take), code_at, row_at
            selected_dims.append(row)
            code_at += len(take) * wpr
            row_at += len(take)
        product = copy(original)
        product.size = row_at
        product.args = [
            mx.concatenate(selected_codes),
            mx.concatenate(selected_scales),
            mx.array(np.concatenate(selected_rows)),
            tables,
            mx.array(np.stack(selected_dims).reshape(-1)),
            groups,
        ]
        mx.eval(*product.args)
        self.head = B.BlockProduct(product, threadgroups=B.HEAD_THREADGROUPS)

    def covers_prompt(self, prompt):
        """Use the full head when more than one in eight prompt tokens is missing."""
        present = self.keep[np.asarray(prompt)]
        return np.count_nonzero(~present) * 8 <= present.size

    def __call__(self, states, size):
        table = self.table
        pad = mx.concatenate(
            [states.astype(mx.float32), mx.zeros((1, table.n), mx.float32)]
        )
        rotated = table.rot.fwd(
            mx.broadcast_to(pad[:, None, :], (size, table.nb, table.n))
        )
        positions = self.head.argmax(rotated.transpose(1, 2, 0), size // 8)[: size - 1]
        return self.ids[positions]


def qwen_prefix(block, tokenizer):
    """Keep the frequent BPE prefix and all tokenizer-defined added tokens."""
    if block.table.V <= PREFIX_ROWS:
        return None
    keep = np.arange(block.table.V) < PREFIX_ROWS
    for i in tokenizer.get_added_vocab().values():
        if 0 <= i < len(keep):
            keep[i] = True
    return DraftVocabulary(block, keep)
