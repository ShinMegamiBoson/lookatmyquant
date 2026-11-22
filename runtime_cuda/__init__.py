"""Text inference for LookAtMyQuant Inkling-2-Mini on NVIDIA CUDA."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
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
        return int(text.get("max_position_embeddings", 32768))


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
        engine = "native" if drafter is not None and not needs_triton else "triton"
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
                loaded = LoadedModel(model, path, path_or_repo, engine, ordinal)
                yield loaded, tokenizer
            finally:
                loaded.target = None
                model.close()
    else:
        import torch

        from .inkling.model import Inkling2LoopPacked

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch could not find an NVIDIA CUDA device")
        loaded = LoadedModel(Inkling2LoopPacked(path, f"cuda:{ordinal}"), path, path_or_repo, engine, ordinal)
        try:
            yield loaded, tokenizer
        finally:
            loaded.target = None


def eos_ids(model: LoadedModel, tokenizer) -> set[int]:
    generation = model.path / "generation_config.json"
    configured = json.loads(generation.read_text()).get("eos_token_id") if generation.exists() else None
    if configured is None:
        configured = tokenizer.eos_token_id
    return {configured} if isinstance(configured, int) else set(configured or ())


def load_drafter(model, spec, *, context, revision=None):
    if model.engine != "native":
        raise ValueError("DSpark currently requires the native CUDA engine and a trellis-only target")
    from .inkling_cuda.drafter import Drafter

    if type(context) is not int or not 1 <= context <= model.context_limit:
        raise ValueError("DSpark context is outside the model context")
    path = resolve_path(spec, revision=revision, drafter=True, target=model.source)
    return Drafter(path, model.target, context, model.device)


def _native_tokens(model, ids, limit, eos, blocked, drafter, block, temperature, seed, stats):
    context = len(ids) + limit - 1
    with ExitStack() as stack:
        stack.enter_context(model.target.dev.cuda.activate())
        if temperature or drafter is not None:
            import torch

            stack.enter_context(torch.cuda.device(model.device))
            stack.enter_context(torch.cuda.stream(torch.cuda.default_stream(model.device)))
        session = stack.enter_context(model.target.session(context))
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


def _triton_tokens(model, ids, limit, eos, blocked, temperature, seed, stats):
    from .inkling.generate import stream_serial

    if not temperature:
        blocked_ids = () if blocked is None else (blocked,)
        triton_stats = {}
        for chunk in stream_serial(model.target, ids, max_tokens=limit, eos_ids=eos, blocked=blocked_ids,
                                   stats_out=triton_stats):
            yield from chunk
        if triton_stats:
            stats.update(prefill_seconds=triton_stats["prompt_s"], peak_gb=triton_stats["peak_gb"])
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
    if model.target is None:
        raise RuntimeError("The CUDA model is closed")
    ids = list(ids)
    if not ids or type(max_tokens) is not int or max_tokens < 1 or len(ids) + max_tokens - 1 > model.context_limit:
        raise ValueError("Prompt and generation exceed the context limit or are empty")
    size = model.target.head.size if model.engine == "native" else model.target.order.numel()
    if any(not 0 <= token < size for token in ids) or blocked_token is not None and not 0 <= blocked_token < size:
        raise ValueError("Token ID outside the model vocabulary")
    if type(temperature) not in (float, int) or not 0 <= temperature <= 2:
        raise ValueError("Temperature must be in [0, 2]")
    if seed is not None and (type(seed) is not int or not -(1 << 63) <= seed < (1 << 64)):
        raise ValueError("Seed must be a 64-bit integer")
    if drafter is not None and (type(draft_block_size) is not int or draft_block_size < 2):
        raise ValueError("DSpark block size must be at least two")
    if drafter is not None and model.engine != "native":
        raise ValueError("DSpark currently requires the native CUDA engine")
    eos = set(eos_ids)
    stats = {} if stats_out is None else stats_out
    if model.engine == "native":
        tokens = _native_tokens(model, ids, max_tokens, eos, blocked_token, drafter, draft_block_size,
                                temperature, seed, stats)
    else:
        tokens = _triton_tokens(model, ids, max_tokens, eos, blocked_token, temperature, seed, stats)
    start = time.perf_counter()
    first = last = None
    count = 0
    try:
        for token in tokens:
            last = time.perf_counter()
            if first is None:
                first = last
            count += 1
            yield token
    finally:
        tokens.close()
        elapsed = last - first if first is not None else 0.
        ttft = first - start if first is not None else None
        stats.setdefault("prefill_seconds", ttft)
        stats.update(engine=model.engine, prompt_tokens=len(ids), generated_tokens=count,
                     time_to_first_token_seconds=ttft,
                     decode_tokens_per_second=(count - 1) / elapsed if count > 1 and elapsed else None)
