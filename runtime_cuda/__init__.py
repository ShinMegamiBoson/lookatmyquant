"""Text inference for LookAtMyQuant Inkling-2-Mini on NVIDIA CUDA."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import re
import time
from pathlib import Path

from .downloads import resolve_path

THINKING_TOKEN = "<|content_thinking|>"


@dataclass
class LoadedModel:
    target: object
    path: Path
    source: str | Path
    engine: str
    ordinal: int

    @property
    def device(self):
        return f"cuda:{self.ordinal}"

    @property
    def context_limit(self):
        config = json.loads((self.path / "config.json").read_text())
        text = config.get("text_config", config)
        return int(text.get("max_position_embeddings", 16384))


def _ordinal(device):
    if device == "cuda" or device is None:
        return 0
    match = re.fullmatch(r"(?:cuda:)?([0-9]+)", str(device))
    if match is None:
        raise ValueError(f"Expected a CUDA device, got {device!r}")
    return int(match[1])


@contextmanager
def open_model(path_or_repo, *, device="cuda", engine="auto", revision=None, drafter=None):
    """Open a packed CUDA decoder and tokenizer for a sequence of requests."""
    if engine not in ("auto", "native", "triton"):
        raise ValueError(f"Unknown CUDA engine {engine!r}")
    ordinal = _ordinal(device)
    path = resolve_path(path_or_repo, revision=revision)
    from transformers import AutoTokenizer

    from .inkling.schema import validate_packed_dir

    formats = validate_packed_dir(path, deep=False)
    needs_triton = "experts:nvfp4" in formats
    if engine == "auto":
        engine = "triton" if needs_triton else "native"
    if engine == "native" and needs_triton:
        raise ValueError("NVFP4 packs require the Triton CUDA engine")
    if engine == "triton" and drafter is not None:
        raise ValueError("DSpark currently requires the native CUDA engine and a trellis-only target")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if engine == "native":
        from .inkling_cuda.device_model import DeviceModel
        from .inkling_cuda.driver import Cuda

        with Cuda(ordinal) as cuda:
            model = DeviceModel(path, cuda)
            try:
                yield LoadedModel(model, path, path_or_repo, engine, ordinal), tokenizer
            finally:
                model.close()
    else:
        import torch

        from .inkling.model import Inkling2LoopPacked

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch could not find an NVIDIA CUDA device")
        model = Inkling2LoopPacked(path, f"cuda:{ordinal}")
        yield LoadedModel(model, path, path_or_repo, engine, ordinal), tokenizer


def eos_ids(model: LoadedModel, tokenizer) -> set[int]:
    generation = model.path / "generation_config.json"
    configured = json.loads(generation.read_text()).get("eos_token_id") if generation.exists() else None
    if configured is None:
        configured = tokenizer.eos_token_id
    return {configured} if isinstance(configured, int) else set(configured or ())


def load_drafter(model, spec, *, context, revision=None):
    from .inkling_cuda.drafter import Drafter

    if model.engine != "native":
        raise ValueError("DSpark currently requires the native CUDA engine and a trellis-only target")
    path = resolve_path(spec, revision=revision, drafter=True, target=model.source)
    return Drafter(path, model.target, context, model.device)


def _native_tokens(model, ids, limit, eos, blocked, drafter, block, temperature, seed, stats):
    context = len(ids) + limit - 1
    with model.target.dev.cuda.activate(), model.target.session(context) as session:
        if temperature:
            from .inkling_cuda.drafter import TemperatureSampler

            sampler = TemperatureSampler(session, model.device, temperature, seed)
        else:
            sampler = None
        if drafter is not None:
            from .inkling_cuda.drafter import stream_speculative

            yield from stream_speculative(drafter, session, ids, limit, eos, block,
                                          blocked_token=blocked, sampler=sampler, stats=stats)
            return
        session.prefill(ids[:-1])
        token = ids[-1]
        for _ in range(limit):
            token = session.predict(token, blocked_token=blocked, sampler=sampler)
            yield token
            if token in eos:
                return


def _triton_tokens(model, ids, limit, eos, blocked, temperature, seed):
    from .inkling.generate import stream_serial

    if not temperature:
        blocked_ids = () if blocked is None else (blocked,)
        for chunk in stream_serial(model.target, ids, max_tokens=limit, eos_ids=eos, blocked=blocked_ids):
            yield from chunk
        return

    import torch

    from .inkling_cuda.drafter import sample_logits

    target = model.target
    generator = torch.Generator(device=target.device)
    generator.seed() if seed is None else generator.manual_seed(seed)
    draws = torch.rand(limit, generator=generator, device=target.device)
    cache = target.make_cache(len(ids) + limit - 1)
    logits = target.prefill(ids, cache)
    for step in range(limit):
        token, = sample_logits(logits, draws[step:step + 1], temperature, blocked)
        yield token
        if token in eos or step + 1 == limit:
            break
        ids_gpu = torch.tensor([token], device=target.device)
        logits = target.logits(target.decode_step(ids_gpu, cache))


def stream_tokens(model, ids, *, max_tokens, eos_ids=(), blocked_token=None, drafter=None, draft_block_size=2,
                  temperature=0., seed=None, stats_out=None):
    """Yield generated token IDs, including EOS when generated."""
    ids = list(ids)
    if not ids or max_tokens < 1 or len(ids) + max_tokens - 1 > model.context_limit:
        raise ValueError("Prompt and generation exceed the context limit or are empty")
    size = model.target.head.size if model.engine == "native" else model.target.order.numel()
    if any(not 0 <= token < size for token in ids) or blocked_token is not None and not 0 <= blocked_token < size:
        raise ValueError("Token ID outside the model vocabulary")
    if not 0 <= temperature <= 2:
        raise ValueError("Temperature must be in [0, 2]")
    if drafter is not None and draft_block_size < 2:
        raise ValueError("DSpark block size must be at least two")
    if drafter is not None and model.engine != "native":
        raise ValueError("DSpark currently requires the native CUDA engine")
    eos = set(eos_ids)
    stats = {} if stats_out is None else stats_out
    if model.engine == "native":
        tokens = _native_tokens(model, ids, max_tokens, eos, blocked_token, drafter, draft_block_size,
                                temperature, seed, stats)
    else:
        tokens = _triton_tokens(model, ids, max_tokens, eos, blocked_token, temperature, seed)
    start = time.perf_counter()
    first = None
    count = 0
    try:
        for token in tokens:
            if first is None:
                first = time.perf_counter()
            count += 1
            yield token
    finally:
        tokens.close()
        elapsed = time.perf_counter() - first if first is not None else 0.
        stats.update(engine=model.engine, prompt_tokens=len(ids), generated_tokens=count,
                     prefill_seconds=(first - start) if first is not None else None,
                     decode_tokens_per_second=(count - 1) / elapsed if count > 1 and elapsed else None)
