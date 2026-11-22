"""Shared lamq CLI routing and formatting, without loading MLX or CUDA weights."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from runtime_mlx.cli import main
from runtime_mlx.runtime import Chunk


class CLITests(unittest.TestCase):
    def invoke(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(list(args))
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_cuda_raw_and_chat(self):
        fake_cuda = ModuleType("runtime_cuda")
        fake_decoders = ModuleType("tokenizers.decoders")
        model = SimpleNamespace(context_limit=40)
        calls = []

        class Tokenizer:
            chat_template = "{{ reasoning_effort }}"
            backend_tokenizer = "backend"

            def apply_chat_template(self, messages, **kwargs):
                calls.append(("chat", messages, kwargs))
                return [4, 5]

            def encode(self, prompt, *, add_special_tokens):
                calls.append(("raw", prompt, add_special_tokens))
                return [6]

            def convert_tokens_to_ids(self, token):
                self_test.assertEqual(token, "<|content_thinking|>")
                return 88

        class DecodeStream:
            def __init__(self, *, skip_special_tokens):
                self_test.assertTrue(skip_special_tokens)

            def step(self, backend, tid):
                self_test.assertEqual(backend, "backend")
                return {13: None, 14: "☕"}[tid]

        @contextlib.contextmanager
        def open_model(path, **kwargs):
            calls.append(("open", path, kwargs))
            try:
                yield model, Tokenizer()
            finally:
                calls.append(("close",))

        def stream_tokens(target, ids, *, stats_out, **kwargs):
            self.assertIs(target, model)
            calls.append(("stream", ids, kwargs))
            try:
                yield from (13, 14, 99)
            finally:
                stats_out.update(time_to_first_token_seconds=0.25, decode_tokens_per_second=10,
                                 draft_checks=2, accepted_tokens=3)

        self_test = self
        fake_decoders.DecodeStream = DecodeStream
        fake_cuda.THINKING_TOKEN = "<|content_thinking|>"
        fake_cuda.open_model = open_model
        fake_cuda.eos_ids = lambda *_: {99}
        fake_cuda.stream_tokens = stream_tokens
        fake_cuda.load_drafter = lambda _, path, *, context: (path, context)
        with patch.dict(sys.modules, {"runtime_cuda": fake_cuda, "tokenizers.decoders": fake_decoders}):
            code, out, err = self.invoke(
                "generate", "model", "--device", "cuda:2", "-p", "hello", "--system", "system",
                "--json", "--seed", "12", "--max-tokens", "8", "--temp", "0.7",
                "--drafter", "draft", "--draft-block-size", "3", "--engine", "native", "--revision", "rev",
            )
            self.assertEqual((code, err), (0, ""))
            self.assertEqual(json.loads(out), {
                "text": "☕", "prompt_tokens": 2, "generation_tokens": 2, "prompt_tps": 8.0,
                "generation_tps": 10.0, "finish_reason": "stop", "tokens_per_check": 2.5,
            })
            self.assertEqual(calls, [
                ("open", "model", {"device": "cuda:2", "engine": "native", "revision": "rev", "drafter": "draft"}),
                ("chat", [{"role": "system", "content": "system"}, {"role": "user", "content": "hello"}],
                 {"tokenize": True, "add_generation_prompt": True, "reasoning_effort": "none"}),
                ("stream", [4, 5], {"max_tokens": 8, "eos_ids": {99}, "blocked_token": 88,
                                     "drafter": ("draft", 9), "draft_block_size": 3, "temperature": 0.7, "seed": 12}),
                ("close",),
            ])
            calls.clear()
            code, out, err = self.invoke(
                "generate", "model", "--device", "cuda", "--raw", "-p", "unaltered", "--max-tokens", "3", "--stats",
            )
            self.assertEqual(code, 0)
            self.assertEqual(out, "☕\n")
            self.assertIn("stop=stop", err)
            self.assertEqual(calls[1], ("raw", "unaltered", False))
            self.assertIsNone(calls[2][2]["blocked_token"])

    def test_backend_flag_validation(self):
        for extra in (
            ["--mode", "linear"], ["--block-size", "8"], ["--prefill-chunk", "-1"], ["--no-verify"],
            ["--full-vocab"], ["--dense-drafter"], ["--no-adaptive"], ["--top-p", "0.9"],
            ["--temp", "nan"], ["--max-tokens", "0"], ["--draft-block-size", "4"],
        ):
            with self.subTest(extra=extra):
                code, _, err = self.invoke("generate", "model", "--device", "cuda", *extra)
                self.assertEqual(code, 2)
                self.assertIn(extra[0], err)
        for args in (
            ["generate", "model", "--device", "cpu"],
            ["generate", "model", "--seed", "1"],
            ["chat", "model", "--device", "cuda"],
        ):
            with self.subTest(args=args):
                self.assertEqual(self.invoke(*args)[0], 2)

    def test_mlx_generate_preserves_defaults(self):
        class Runtime:
            def stream(self, prompt, **kwargs):
                self_test.assertEqual(prompt, "input")
                self_test.assertEqual((kwargs["mode"], kwargs["block_size"], kwargs["prefill_chunk"]), ("linear", 8, -1))
                yield Chunk(text="output", prompt_tokens=1, generation_tokens=1, finish_reason="length")

        self_test = self
        with patch("runtime_mlx.cli._load_from_args", return_value=Runtime()):
            code, out, err = self.invoke("generate", "model", "-p", "input", "--json")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(json.loads(out)["text"], "output")


if __name__ == "__main__":
    unittest.main()
