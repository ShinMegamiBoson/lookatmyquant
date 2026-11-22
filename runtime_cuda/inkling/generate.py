"""Streaming serial generation for packed Inkling-2-Mini on CUDA (counterpart of `runtime_mlx/inkling/generate.py`).

One greedy decode step - embedding, 48 layer visits, LM-head argmax, writing the token back and advancing the
position - is captured once in a CUDA graph and replayed per token. The host reads tokens back every `sync_every`
steps (the MLX runtime's one-step `async_eval` pipeline, generalised), so the GPU never waits on Python.
"""
from __future__ import annotations

import statistics
import time
from typing import Any, Iterator, Sequence

import torch

from .model import Inkling2Cache, Inkling2LoopPacked


class GreedyDecodeGraph:
    """A captured greedy decode step bound to one cache. `tok` holds the input token and receives the output."""

    def __init__(self, model: Inkling2LoopPacked, cache: Inkling2Cache, max_steps: int):
        self.model, self.cache = model, cache
        dev = model.device
        self.tok = torch.zeros(1, dtype=torch.long, device=dev)
        self.out = torch.zeros(max_steps + 1, dtype=torch.long, device=dev)
        self.step = torch.zeros(1, dtype=torch.long, device=dev)
        # warm up on a scratch cache: Triton compiles, rotation factors and block tables get created outside capture
        scratch = model.make_cache(cache.capacity)
        with torch.cuda.device(dev):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    model.argmax(model.decode_step(self.tok, scratch))
            torch.cuda.current_stream().wait_stream(s)
            del scratch
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                nxt = model.argmax(model.decode_step(self.tok, cache))
                self.tok.copy_(nxt)
                self.out.index_copy_(0, self.step, nxt)
                self.step.add_(1)

    def replay(self):
        self.graph.replay()


def stream_serial(
    model: Inkling2LoopPacked,
    prompt_ids: Sequence[int],
    *,
    max_tokens: int = 256,
    eos_ids: Sequence[int] = (200006,),
    cache: Inkling2Cache | None = None,
    graph: GreedyDecodeGraph | None = None,
    sync_every: int = 8,
    stats_out: dict[str, Any] | None = None,
) -> Iterator[list[int]]:
    """Greedy single-token generation; yields lists of new token ids. Pass `cache` and `graph` to reuse a capture."""
    eos = {int(x) for x in eos_ids}
    need = len(prompt_ids) + max_tokens + sync_every
    if cache is None:
        cache = model.make_cache(max(1024, ((need + 255) // 256) * 256))
    else:
        cache.reset()
    if need > cache.capacity:
        raise ValueError(f"prompt + max_tokens needs {need} positions, cache holds {cache.capacity}")
    if graph is None or graph.cache is not cache:
        graph = GreedyDecodeGraph(model, cache, max_tokens + sync_every)

    torch.cuda.synchronize(model.device)
    t0 = time.perf_counter()
    logits = model.prefill(list(prompt_ids), cache)
    first = int(logits[-1].argmax())
    t1 = time.perf_counter()
    yield [first]
    emitted, step_ms = 1, []
    if first not in eos and max_tokens > 1:
        graph.tok.fill_(first)
        graph.step.zero_()
        done = False
        read = 0
        ts = time.perf_counter()
        while not done and emitted < max_tokens:
            n = min(sync_every, max_tokens - emitted)
            for _ in range(n):
                graph.replay()
            new = graph.out[read : read + n].tolist()  # syncs
            now = time.perf_counter()
            step_ms += [(now - ts) * 1e3 / n] * n
            ts = now
            read += n
            out = []
            for t in new:
                out.append(t)
                emitted += 1
                if t in eos:
                    done = True
                    break
            yield out
    t2 = time.perf_counter()
    cache.seq_len = int(cache.pos.item())
    if stats_out is not None:
        decode_s = max(t2 - t1, 1e-9)
        stats_out.update({
            "mode": "serial-cuda-graph",
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": emitted,
            "prompt_s": round(t1 - t0, 4),
            "prefill_tok_s": round(len(prompt_ids) / max(t1 - t0, 1e-9), 1),
            "decode_tok_s": round((emitted - 1) / decode_s, 2) if emitted > 1 else 0.0,
            "median_step_ms": round(statistics.median(step_ms), 2) if step_ms else 0.0,
            "peak_gb": round(torch.cuda.max_memory_allocated(model.device) / 1e9, 3),
        })


def generate_serial(model: Inkling2LoopPacked, prompt_ids: Sequence[int], **kw) -> list[int]:
    return [t for chunk in stream_serial(model, prompt_ids, **kw) for t in chunk]
