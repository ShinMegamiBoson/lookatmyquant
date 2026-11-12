"""The block path (block_q35) against float64 NumPy products of independently unpacked codes, and, when the packed
Qwen3.5 file is on this Mac, against fast_q35's one-token decoding from the same caches.

    python runtime_mlx/test_block_q35.py
    LAMQ_Q35_PACKED=path/to/packed python runtime_mlx/test_block_q35.py      # the packed file somewhere else
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime_mlx"))

import mlx.core as mx
import numpy as np

from runtime_mlx import block_q35 as B
from runtime_mlx import fast_q35 as F
from runtime_mlx.test_fast_q35 import scalar_group, trellis_group

PACKED = Path(os.environ.get("LAMQ_Q35_PACKED") or ROOT.parent / "bench_models" / "lamq" / "q35_SQYFR_1p2GB")


TOL = {"float": 2e-5, "half": 1e-3}   # tables as the file holds them, and as 16-bit floats (what the runtime keeps)


def product(rng, n, plan, table="float"):
    """A fast_q35.Product of the row groups in plan [(kind, bits, rows, input)], and its dense float64 matrix by input."""
    groups, dense, at = [], [], 0
    for kind, k, rows, inp in plan:
        g, w = (trellis_group if kind == "tr" else scalar_group)(rng, rows, n, k, np.arange(at, at + rows), inp)
        groups.append(g)
        dense.append((w, inp))
        at += rows
    return F.Product(n, groups, at, table), dense, at


class BlockProductTest(unittest.TestCase):
    def test_argmax_matches_logits_and_breaks_ties_by_token_id(self):
        rng = np.random.default_rng(4)
        plan = [("tr", 2, 79, 0), ("sc", 5, 43, 0), ("tr", 3, 57, 1)]
        p, _, _ = product(rng, 512, plan, table="half")
        head = B.BlockProduct(p, rb=2, sg=16, cs=2)
        for tb in (1, 2):
            shape = (2, 512, 8 * tb)
            for x in (mx.zeros(shape), mx.full(shape, float("nan")), mx.array(rng.standard_normal(shape).astype(np.float32))):
                np.testing.assert_array_equal(np.array(head.argmax(x, tb)), np.array(mx.argmax(head(x, tb), axis=-1)))

    def check(self, n, plan, inputs=1, **shape):
        for table, tol in TOL.items():
            rng = np.random.default_rng(0)
            p, dense, size = product(rng, n, plan, table)
            for tb in (1, 2):
                T = 8 * tb
                x = rng.standard_normal((inputs, n, T)).astype(np.float32)          # [input][column][token]
                want = np.concatenate([w @ x[inp].astype(np.float64) for w, inp in dense]).T
                got = np.array(B.BlockProduct(p, **shape)(mx.array(x), tb))
                self.assertEqual(got.shape, (T, size))
                self.assertLess(np.abs(got - want).max() / np.abs(want).max(), tol, (n, plan, tb, shape, table))

    def test_every_width(self):
        # every kind of group the packed files hold, in one launch, with row counts that fill no tile exactly
        plan = [("tr", 1, 77, 0), ("tr", 2, 130, 0), ("tr", 3, 64, 0), ("tr", 4, 9, 0), ("sc", 5, 40, 0), ("sc", 8, 33, 0)]
        for n in (512, 2560):
            self.check(n, plan)

    def test_column_parts(self):
        plan = [("tr", 2, 100, 0), ("tr", 3, 60, 0), ("sc", 5, 21, 0)]
        for cs in (2, 4, 8):
            self.check(1024, plan, rb=2, sg=16, cs=cs)
        self.check(4096, [("tr", 1, 50, 0), ("tr", 4, 30, 0)], rb=1, sg=32, cs=16)

    def test_several_inputs(self):
        self.check(512, [("tr", 2, 70, 0), ("sc", 8, 40, 1), ("tr", 3, 51, 2)], inputs=3)


@unittest.skipUnless(PACKED.exists(), f"{PACKED} is not on this machine")
class BlockModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mlx_lm.models.cache import ArraysCache, KVCache
        cls.model, cls.tok = F.load(PACKED, verify=False)
        cls.block = B.Block(cls.model, tap_layers=(1, 29))

        def snapshot(cache):
            out = []
            for c in cache:
                if isinstance(c, KVCache):
                    n = KVCache()
                    n.keys, n.values, n.offset = c.keys, c.values, c.offset
                else:
                    n = ArraysCache(size=2)
                    n[0], n[1] = c[0], c[1]
                out.append(n)
            return out

        cls.snapshot = staticmethod(snapshot)
        prompt = cls.tok.apply_chat_template([{"role": "user", "content": "A train leaves at 9:40 and arrives at 13:05. How long is the trip? Show the steps."}],
                                             add_generation_prompt=True, enable_thinking=False)
        cache = cls.model.make_cache()
        first = int(mx.argmax(cls.model(mx.array([prompt]), cache=cache)[0, -1]))
        cls.base = snapshot(cache)
        cache, cls.tokens, cls.logits = snapshot(cls.base), [first], []
        for _ in range(24):                                   # the model's own next tokens, one at a time
            cls.logits.append(np.array(cls.model(mx.array([[cls.tokens[-1]]]), cache=cache)[0, -1]))
            cls.tokens.append(int(cls.logits[-1].argmax()))

    def step(self, token, cache):
        return np.array(self.model(mx.array([[token]]), cache=cache)[0, -1])

    def test_check_equals_one_token_steps(self):
        for T in (8, 16):
            cache = self.snapshot(self.base)
            logits, taps = self.block.check(self.tokens[:T], cache)
            got, want = np.array(logits), np.stack(self.logits[:T])
            self.block.pending = None
            self.assertEqual(got.argmax(1).tolist(), want.argmax(1).tolist())
            self.assertLess(np.abs(got - want).max(), 0.05)
            self.assertEqual(taps[1].shape, (T, got.shape[0] and self.block.table.n))

    def test_greedy_check_matches_serial(self):
        for T in (8, 16):
            cache = self.snapshot(self.base)
            ids, _ = self.block.check(self.tokens[:T], cache, top1=True)
            self.assertEqual(ids.tolist(), self.tokens[1 : T + 1])
            self.block.discard()

    def test_commit_keeps_a_prefix(self):
        for T, kept in ((8, 1), (8, 5), (8, 8), (16, 3), (16, 16)):
            cache = self.snapshot(self.base)
            self.block.check(self.tokens[:T], cache)
            self.block.commit(cache, kept)
            self.block.flush(cache)
            for j in range(kept, kept + 4):                   # carrying on one token at a time gives the same next tokens
                got = self.step(self.tokens[j], cache)
                self.assertEqual(int(got.argmax()), int(self.logits[j].argmax()), (T, kept, j))
                self.assertLess(np.abs(got - self.logits[j]).max(), 0.05)

    def test_blocks_in_a_row(self):
        cache = self.snapshot(self.base)
        at = 0
        for kept in (3, 8, 1, 6):                             # check a block, keep part of it, check the next from there
            logits, _ = self.block.check(self.tokens[at : at + 8], cache)
            self.assertEqual(np.array(mx.argmax(logits, axis=-1)).tolist(), self.tokens[at + 1 : at + 9])
            self.block.commit(cache, kept)
            at += kept
        self.block.flush(cache)                               # and one token at a time from there
        for j in range(at, at + 4):
            self.assertEqual(int(self.step(self.tokens[j], cache).argmax()), int(self.logits[j].argmax()), j)


if __name__ == "__main__":
    unittest.main()
