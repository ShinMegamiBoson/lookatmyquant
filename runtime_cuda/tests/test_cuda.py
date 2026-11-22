"""Packed CUDA and end-to-end checks. No checkpoint download is required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fixture import build_mini_packed_dir
from inkling_cuda.device_model import Device, DeviceModel, PREFILL_CHUNK
from inkling_cuda.driver import Cuda
from inkling_cuda.model import Model, VisitCache
from inkling_cuda.packed import FORMATS, Slab, project, rotate


def decoded_row(words, width, fmt, table):
    """Decode using Python integer shifts over a whole row."""
    bits, group, count, multiplier = FORMATS[fmt]
    raw = int.from_bytes(words.tobytes(), "little")
    decoded = []
    for column in range(0, width, group):
        if fmt.startswith("lm"):
            index = (raw >> (column * bits)) & ((1 << bits) - 1)
        else:
            index = (raw >> ((width - column - group) * bits)) & 4095
            if multiplier:
                index = ((index * multiplier) & 0xffffffff) >> (32 - (count - 1).bit_length())
        decoded.extend(table[index])
    return np.array(decoded, dtype=np.float32)


class PackedCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cuda = Cuda()

    @classmethod
    def tearDownClass(cls):
        cls.cuda.close()

    def test_all_code_formats_and_routed_slots(self):
        rng = np.random.default_rng(511)
        width, rows = 1152, 17
        x = rng.normal(size=(3, width)).astype(np.float32)
        slabs = []
        routes = []
        expected = []
        for i, (fmt, (bits, group, count, _)) in enumerate(FORMATS.items()):
            codes = rng.integers(0, 2**32, (2, rows, width * bits // 32), dtype=np.uint32)
            scales = rng.uniform(0.005, 0.04, (2, rows)).astype(np.float16)
            table = rng.normal(size=(count, group)).astype(np.float32)
            slab = Slab(width, rows, fmt, codes, scales, table, self.cuda)
            slabs.append(slab)
            route = np.full(len(FORMATS), -1, np.int32)
            expert = i % 2
            route[i] = expert
            routes.append(route)
            matrix = np.stack([decoded_row(row, width, fmt, table) for row in codes[expert]])
            expected.append((matrix @ x[i // 3]) * scales[expert])
        result = project(x, slabs, routes, x_div=3)
        np.testing.assert_allclose(result, expected, rtol=2e-5, atol=3e-6)

    def test_hadamard_roundtrip(self):
        rng = np.random.default_rng(5)
        for width in (256, 1024, 1152, 2304, 3072):
            x = rng.normal(size=(2, width)).astype(np.float32)
            signs = rng.choice([-1, 1], size=width).astype(np.float32)
            np.testing.assert_allclose(rotate(rotate(x, signs), signs, inverse=True), x, atol=1.5e-6)

    def test_message_down_rounds_each_projection(self):
        dev = Device(self.cuda)
        try:
            x = self.cuda.upload(np.ones(4, dtype=np.float32))
            weight = dev.weight(np.array([[1, 1 / 256, -255 / 256, 0]], np.float32))
            with self.cuda.temporary(4) as (out,):
                dev.dense(x, weight, out, kernel="message_down")
                self.assertEqual(self.cuda.read(out, np.empty(1, np.float32))[0], 1 / 256)
        finally:
            dev.close()

    def test_gpu_hadamard_matches_release_dimensions(self):
        rng = np.random.default_rng(77)
        dev = Device(self.cuda)
        try:
            for width in (384, 1024, 1152, 2304, 3072):
                x = rng.normal(size=(4, width)).astype(np.float32)
                signs = rng.choice([-1, 1], size=width).astype(np.float32)
                source, device_signs = self.cuda.upload(x), dev.weight(signs)
                with self.cuda.temporary(x.nbytes) as (out,):
                    for inverse in (False, True):
                        dev.rotate(source, device_signs, out, inverse, rows=4)
                        actual = self.cuda.read(out, np.empty_like(x))
                        np.testing.assert_allclose(actual, rotate(x, signs, inverse), rtol=0, atol=2e-6)
        finally:
            dev.close()

    def test_cache_ring_keeps_relative_positions(self):
        cache = VisitCache(3, 2, 4)
        for i in range(8):
            x = np.full((2, 4), i, dtype=np.float32)
            k, v, distances = cache.push_kv(i, x, 2 * x)
            np.testing.assert_array_equal(k[0, :, 0], i - distances)
            np.testing.assert_array_equal(v[0, :, 0], 2 * (i - distances))

    def test_looped_cuda_model_against_cpu(self):
        with tempfile.TemporaryDirectory() as folder:
            build_mini_packed_dir(Path(folder))
            gpu = Model(folder, self.cuda).session(16)
            cpu = Model(folder, None).session(16)
            tokens = [5, 19, 42, 128, 301, 7, 255, 12, 400, 88]
            for index, token in enumerate(tokens):
                a = gpu.step(token, logits=index > 0)
                b = cpu.step(token, logits=index > 0)
                if index > 0:
                    # GPU and CPU dot reductions can cross a BF16 gate rounding boundary.
                    np.testing.assert_allclose(a, b, rtol=0, atol=2e-3)
                    self.assertEqual(a.argmax(), b.argmax())
                else:
                    self.assertIsNone(a)
                    self.assertIsNone(b)
            self.assertEqual(gpu.model.cfg.total_visits, 10)
            self.assertEqual(set(gpu.caches), set(range(10)))
            self.assertEqual(gpu.position, len(tokens))

    def test_device_decoder_and_greedy_readback(self):
        with tempfile.TemporaryDirectory() as folder:
            build_mini_packed_dir(Path(folder))
            reference = Model(folder, None).session(16)
            model = DeviceModel(folder, self.cuda)
            try:
                with model.session(16) as diagnostic, model.session(16) as generation:
                    # The default path must never invoke the host-assisted projections.
                    with patch.object(self.cuda, "project", side_effect=AssertionError("host projection")):
                        for i, token in enumerate((5, 19, 42, 128, 301, 7, 255, 12, 400, 88)):
                            want = reference.step(token, logits=i != 0)
                            got = diagnostic.step(token, logits=i != 0)
                            with patch.object(self.cuda, "read", wraps=self.cuda.read) as read:
                                if i == 0:
                                    generation.step(token, logits=False)
                                    self.assertIsNone(got)
                                    read.assert_not_called()
                                else:
                                    self.assertEqual(generation.predict(token), want.argmax())
                                    read.assert_called_once()
                                    self.assertEqual(read.call_args.args[1].nbytes, 8)
                                    np.testing.assert_allclose(got, want, rtol=0, atol=2e-3)
                    self.assertEqual(set(generation.caches), set(range(10)))
                    self.assertEqual(generation.position, 10)
            finally:
                model.close()

    def test_greedy_sampling_can_exclude_a_token_on_device(self):
        with tempfile.TemporaryDirectory() as folder:
            build_mini_packed_dir(Path(folder))
            model = DeviceModel(folder, self.cuda)
            try:
                with model.session(16) as reference, model.session(16) as masked:
                    scores = reference.step(42)
                    blocked = int(scores.argmax())
                    scores[blocked] = -np.inf
                    with patch.object(self.cuda, "read", wraps=self.cuda.read) as read:
                        self.assertEqual(masked.predict(42, blocked_token=blocked), int(scores.argmax()))
                        read.assert_called_once()
                    self.assertEqual(masked.position, 1)
                    self.assertEqual(masked.predict(9), int(reference.step(9).argmax()))
            finally:
                model.close()

    def test_batched_prefill_preserves_logits_and_decode_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            build_mini_packed_dir(Path(folder))
            model = DeviceModel(folder, self.cuda)
            try:
                # Cross a chunk boundary and wrap the local attention and convolution rings.
                tokens = np.random.default_rng(73).integers(0, 500, PREFILL_CHUNK + 7)
                with model.session(32) as serial, model.session(32) as batched:
                    serial.step(tokens[0], logits=False)
                    batched.step(tokens[0], logits=False)
                    with patch.object(self.cuda, "read", side_effect=AssertionError("prefill readback")):
                        batched.prefill(tokens[1:-2])
                    for token in tokens[1:-2]:
                        serial.step(token, logits=False)
                    for token in tokens[-2:]:
                        np.testing.assert_array_equal(batched.step(token), serial.step(token))
                    self.assertEqual(batched.position, len(tokens))
            finally:
                model.close()

    def test_speculative_suffix_can_be_discarded_after_cache_wrap(self):
        with tempfile.TemporaryDirectory() as folder:
            build_mini_packed_dir(Path(folder))
            model = DeviceModel(folder, self.cuda)
            try:
                with model.session(64) as serial, model.session(64) as verify:
                    prefix = tuple(range(32))
                    serial.prefill(prefix)
                    verify.prefill(prefix)
                    proposals = (300, 77, 33, 81, 55, 7, 161, 19)
                    predictions, taps = verify.predict_block(proposals, all_tokens=True, tap_visits=(2, 10))
                    self.assertEqual(set(taps), {2, 10})
                    with model.session(64) as full:
                        full.prefill(prefix)
                        np.testing.assert_array_equal(predictions, [full.predict(t) for t in proposals])
                    verify.commit_block_prefix(len(prefix), 2)
                    serial.prefill(proposals[:2])
                    for token in (62, 32, 7):
                        np.testing.assert_array_equal(verify.step(token), serial.step(token))
            finally:
                model.close()


if __name__ == "__main__":
    unittest.main()
