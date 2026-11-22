"""High-level Apple Silicon MLX runtime API for LookAtMyQuant models.

Supports all packed formats (`opgraph-packed-1` via `artifact.json`, including
Qwen3.5-4B and Qwen3.8-27B with or without a DSpine drafter, `inkling2-packed`
via `manifest.json` for Inkling-2-Mini 3.9GB–12.8GB with or without a 4-bit
DSpark drafter, and `lamq.json` exports such as Gemma 4 E2B) from either a
local directory or a Hugging Face repository ID.

Drop-in `mlx_lm` integration:

    from runtime_mlx import load
    from mlx_lm import generate

    model, tokenizer = load("shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB")

Unified runtime with optional speculative decoding and streaming:

    from runtime_mlx import Runtime

    rt = Runtime.load(
        "shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB",
        drafter="shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB-drafter",
    )
    for chunk in rt.stream("What is 17 times 23? Show the steps.", max_tokens=200):
        print(chunk.text, end="", flush=True)
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    """One incremental piece of generated output."""

    text: str
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    tokens_per_check: float | None = None
    stats: dict[str, Any] | None = None


@dataclass
class GenerationResult:
    """Complete output and timing summary from a single generation call."""

    text: str
    token_ids: list[int]
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    finish_reason: str | None
    tokens_per_check: float | None = None
    stats: dict[str, Any] | None = None


def resolve_path(path_or_repo: str | Path) -> Path:
    """Return a local directory for a filesystem path or Hugging Face repo ID."""
    path = Path(path_or_repo).expanduser()
    if path.exists():
        return path
    spec = str(path_or_repo)
    if spec.count("/") == 1 and not spec.startswith((".", "/", "~")):
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=spec))
    raise FileNotFoundError(f"model path does not exist: {path_or_repo}")


def detect_format(path: str | Path) -> str:
    """Identify the quantization layout inside a local model directory."""
    directory = Path(path)
    if (directory / "artifact.json").exists():
        return "opgraph-packed-1"
    if (directory / "manifest.json").exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        if str(manifest.get("format", "")).startswith("inkling2-packed"):
            return "inkling2-packed"
    if (directory / "lamq.json").exists():
        return "lamq"
    raise FileNotFoundError(
        f"no artifact.json, manifest.json (inkling2-packed), or lamq.json found in {directory}"
    )


def inspect_model(path_or_repo: str | Path) -> dict[str, Any]:
    """Read model metadata without loading weights into GPU memory."""
    directory = resolve_path(path_or_repo)
    fmt = detect_format(directory)
    config = json.loads((directory / "config.json").read_text())
    text_cfg = config.get("text_config", config)
    info: dict[str, Any] = {
        "path": str(directory),
        "format": fmt,
        "architectures": config.get("architectures"),
        "model_type": text_cfg.get("model_type") or config.get("model_type"),
        "hidden_size": text_cfg.get("hidden_size"),
        "num_hidden_layers": text_cfg.get("num_hidden_layers"),
        "vocab_size": text_cfg.get("vocab_size"),
    }
    if fmt == "opgraph-packed-1":
        manifest = json.loads((directory / "artifact.json").read_text())
        matrices = manifest.get("matrices", {})
        info["packed_bytes"] = manifest.get("blob_bytes")
        info["matrices"] = len(matrices)
        info["separate_lm_head"] = any(
            name.endswith("lm_head.weight") for name in matrices
        )
    elif fmt == "inkling2-packed":
        manifest = json.loads((directory / "manifest.json").read_text())
        info["manifest_format"] = manifest.get("format")
        info["packed_bytes"] = sum(
            f.stat().st_size
            for f in directory.iterdir()
            if f.is_file() and f.name != "README.md"
        )
        info["packed_text_gb"] = manifest.get("packed_text_gb")
        info["n_routed_experts"] = text_cfg.get("n_routed_experts")
        info["loop_passes"] = text_cfg.get("loop_config", {}).get("num_passes")
        info["steering_tensors"] = manifest.get("steering_tensors")
        info["luts"] = manifest.get("luts")
        info["separate_lm_head"] = True
    else:
        meta = json.loads((directory / "lamq.json").read_text())
        weights = directory / "model.safetensors"
        info["packed_bytes"] = weights.stat().st_size if weights.exists() else None
        info["matrices"] = len(meta.get("matrices", {}))
        info["separate_lm_head"] = text_cfg.get("tie_word_embeddings") is False
    return info


def load(
    path_or_repo: str | Path,
    *,
    verify: bool = True,
    kind: str | None = None,
    fuse: bool | None = None,
    fastpath: bool = True,
):
    """Load a LookAtMyQuant model and tokenizer compatible with `mlx_lm`."""
    directory = resolve_path(path_or_repo)
    fmt = detect_format(directory)
    if fmt == "opgraph-packed-1":
        from . import fast_q35

        return fast_q35.load(directory, kind=kind, verify=verify, fuse=fuse)

    if fmt == "inkling2-packed":
        from .inkling import load_inkling

        return load_inkling(directory)

    from . import model as lamq_model

    model, tokenizer = lamq_model.load(directory)
    if fastpath:
        tm = (
            model.language_model.model
            if hasattr(model, "language_model")
            else getattr(model, "model", None)
        )
        layers = getattr(tm, "layers", None)
        if layers and hasattr(layers[0], "per_layer_input_gate"):
            from . import fastpath as lamq_fastpath

            lamq_fastpath.install(model)
    return model, tokenizer


def _eos_ids(tokenizer) -> set[int]:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return {int(x) for x in ids}
    single = getattr(tokenizer, "eos_token_id", None)
    return {int(single)} if single is not None else set()


def _unwrap_ids(encoded) -> list[int]:
    if isinstance(encoded, dict) and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    return [int(x) for x in encoded]


def encode_messages(
    tokenizer,
    messages: Sequence[dict[str, str]],
    *,
    thinking: bool = False,
) -> list[int]:
    """Format chat messages with the tokenizer's chat template into token IDs."""
    tpl = getattr(tokenizer, "chat_template", "") or ""
    if "reasoning_effort" in tpl:
        encoded = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            reasoning_effort="low" if thinking else "none",
        )
    else:
        try:
            encoded = tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
            )
    if isinstance(encoded, str):
        encoded = tokenizer.encode(encoded)
    return _unwrap_ids(encoded)


def encode_prompt(
    tokenizer,
    prompt: str | Sequence[int],
    *,
    chat: bool = True,
    system: str | None = None,
    thinking: bool = False,
) -> list[int]:
    """Encode a raw string, chat prompt, or pre-tokenized ID list."""
    if not isinstance(prompt, str):
        return _unwrap_ids(prompt)
    if chat and getattr(tokenizer, "chat_template", None):
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return encode_messages(tokenizer, messages, thinking=thinking)
    return _unwrap_ids(tokenizer.encode(prompt))


class _IncrementalDecoder:
    """Decode emitted token IDs incrementally using mlx_lm's detokenizer when available."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._detok = getattr(tokenizer, "detokenizer", None)
        self._tokens: list[int] = []
        self._printed = ""
        if self._detok is not None:
            self._detok.reset()

    def push(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        if self._detok is not None:
            parts = []
            for tid in token_ids:
                self._detok.add_token(int(tid))
                seg = self._detok.last_segment
                if seg:
                    parts.append(seg)
            return "".join(parts)
        self._tokens.extend(int(t) for t in token_ids)
        full = self._tokenizer.decode(self._tokens, skip_special_tokens=True)
        delta = full[len(self._printed) :]
        self._printed = full
        return delta

    def finalize(self) -> str:
        if self._detok is not None:
            self._detok.finalize()
            return self._detok.last_segment or ""
        return ""


class Runtime:
    """Stateful LookAtMyQuant inference runtime for plain or speculative generation."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        block=None,
        drafter=None,
        model_path: Path | None = None,
        drafter_path: Path | None = None,
        format_name: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.block = block
        self.drafter = drafter
        self.model_path = model_path
        self.drafter_path = drafter_path
        self.format = format_name

    @classmethod
    def load(
        cls,
        model: str | Path,
        *,
        drafter: str | Path | None = None,
        verify: bool = True,
        kind: str | None = None,
        fuse: bool | None = None,
        fastpath: bool = True,
        drafter_dtype: str = "float16",
        drafter_bits: int | None = None,
        drafter_int8: bool = True,
        select: bool = False,
        full_vocab: bool = False,
    ) -> Runtime:
        """Load a quantized model and, optionally, a paired DSpine drafter."""
        model_path = resolve_path(model)
        fmt = detect_format(model_path)
        if drafter is None:
            m, tok = load(
                model_path,
                verify=verify,
                kind=kind,
                fuse=fuse,
                fastpath=fastpath,
            )
            return cls(m, tok, model_path=model_path, format_name=fmt)

        if fmt == "inkling2-packed":
            from .inkling import DSparkDrafterMLX, load_inkling, resolve_drafter_files

            drafter_path = resolve_path(drafter)
            m, tok = load_inkling(model_path)
            w_path, cfg_path = resolve_drafter_files(
                drafter_path, fallback_config=model_path / "config.json"
            )
            draft = DSparkDrafterMLX(w_path, cfg_path)
            return cls(
                m,
                tok,
                drafter=draft,
                model_path=model_path,
                drafter_path=drafter_path,
                format_name=fmt,
            )

        if fmt != "opgraph-packed-1":
            raise ValueError(
                f"speculative drafting requires an opgraph-packed-1 or inkling2-packed model, got {fmt}"
            )
        import mlx.core as mx

        from . import block_q35, draft_vocab_q35, drafter_q35, fast_q35

        drafter_path = resolve_path(drafter)
        m, tok = fast_q35.load(model_path, kind=kind, verify=verify, fuse=fuse)
        blk = block_q35.Block(m, tap_layers=drafter_q35.drafter_taps(drafter_path))
        dtype = (
            getattr(mx, drafter_dtype)
            if isinstance(drafter_dtype, str)
            else drafter_dtype
        )
        draft = drafter_q35.Drafter(
            drafter_path,
            blk,
            dtype=dtype,
            bits=drafter_bits or None,
            select=select,
            int8=drafter_int8,
        )
        if not full_vocab and draft.path is not True:
            draft.proposal_head = draft_vocab_q35.qwen_prefix(blk, tok)
        return cls(
            m,
            tok,
            block=blk,
            drafter=draft,
            model_path=model_path,
            drafter_path=drafter_path,
            format_name=fmt,
        )

    def encode_prompt(
        self,
        prompt: str | Sequence[int],
        *,
        chat: bool = True,
        system: str | None = None,
        thinking: bool = False,
    ) -> list[int]:
        return encode_prompt(
            self.tokenizer,
            prompt,
            chat=chat,
            system=system,
            thinking=thinking,
        )

    def encode_messages(
        self,
        messages: Sequence[dict[str, str]],
        *,
        thinking: bool = False,
    ) -> list[int]:
        return encode_messages(self.tokenizer, messages, thinking=thinking)

    def stream_tokens(
        self,
        prompt_ids: Sequence[int],
        *,
        max_tokens: int = 256,
        temp: float = 0.0,
        top_p: float = 1.0,
        mode: str = "linear",
        block_size: int = 8,
        use_drafter: bool = True,
        adaptive: bool = True,
    ) -> Iterator[Chunk]:
        """Stream generated chunks for pre-encoded prompt token IDs."""
        ids = _unwrap_ids(prompt_ids)
        if not ids:
            raise ValueError("prompt must contain at least one token")
        if max_tokens <= 0:
            return

        eos = _eos_ids(self.tokenizer)
        if self.format == "inkling2-packed":
            from .inkling import stream_dspark, stream_serial

            decoder = _IncrementalDecoder(self.tokenizer)
            stats: dict[str, Any] = {}
            emitted_total = 0
            stopped = False
            t_start = time.perf_counter()
            if use_drafter and self.drafter is not None and temp >= 0.0:
                gen = stream_dspark(
                    self.model,
                    self.drafter,
                    ids,
                    max_tokens=max_tokens,
                    eos_ids=eos,
                    adaptive=adaptive,
                    temp=temp,
                    top_p=top_p,
                    stats_out=stats,
                )
            else:
                gen = stream_serial(
                    self.model,
                    ids,
                    max_tokens=max_tokens,
                    eos_ids=eos,
                    temp=temp,
                    top_p=top_p,
                    stats_out=stats,
                )
            try:
                for raw_chunk in gen:
                    kept: list[int] = []
                    for tid in raw_chunk:
                        tid = int(tid)
                        if tid in eos:
                            stopped = True
                            break
                        kept.append(tid)
                    emitted_total += len(kept)
                    text = decoder.push(kept)
                    if kept or text:
                        yield Chunk(
                            text=text,
                            token_ids=kept,
                            prompt_tokens=len(ids),
                            generation_tokens=emitted_total,
                        )
                    if stopped:
                        break
            finally:
                gen.close()
            tail = decoder.finalize()
            prompt_s = float(stats.get("prompt_s", 0.0)) + float(
                stats.get("drafter_prompt_s", 0.0)
            )
            rounds_s = float(stats.get("rounds_s", 0.0))
            if rounds_s <= 0.0:
                rounds_s = max(time.perf_counter() - t_start - prompt_s, 1e-9)
            yield Chunk(
                text=tail,
                token_ids=[],
                finish_reason="stop" if stopped else "length",
                prompt_tokens=len(ids),
                generation_tokens=emitted_total,
                prompt_tps=len(ids) / prompt_s if prompt_s > 0 else 0.0,
                generation_tps=stats.get("tokens", emitted_total) / rounds_s
                if rounds_s > 0
                else 0.0,
                tokens_per_check=stats.get("tokens_per_check"),
                stats=stats,
            )
            return

        if use_drafter and self.drafter is not None and temp == 0.0:
            from . import drafter_q35

            decoder = _IncrementalDecoder(self.tokenizer)
            stats: dict[str, Any] = {}
            emitted_total = 0
            stopped = False
            t_start = time.perf_counter()
            gen_q35 = drafter_q35.stream(
                self.model,
                self.block,
                self.drafter,
                ids,
                max_tokens,
                size=block_size,
                eos=eos,
                mode=mode,
                adaptive=adaptive,
                stats_out=stats,
            )
            try:
                for raw_chunk in gen_q35:
                    kept: list[int] = []
                    for tid in raw_chunk:
                        tid = int(tid)
                        if tid in eos:
                            stopped = True
                            break
                        kept.append(tid)
                    emitted_total += len(kept)
                    text = decoder.push(kept)
                    if kept or text:
                        yield Chunk(
                            text=text,
                            token_ids=kept,
                            prompt_tokens=len(ids),
                            generation_tokens=emitted_total,
                        )
                    if stopped:
                        break
            finally:
                gen_q35.close()
            tail = decoder.finalize()
            prompt_s = float(stats.get("prompt_s", 0.0)) + float(
                stats.get("drafter_prompt_s", 0.0)
            )
            rounds_s = float(stats.get("rounds_s", 0.0))
            if rounds_s <= 0.0:
                rounds_s = max(time.perf_counter() - t_start - prompt_s, 1e-9)
            yield Chunk(
                text=tail,
                token_ids=[],
                finish_reason="stop" if stopped else "length",
                prompt_tokens=len(ids),
                generation_tokens=emitted_total,
                prompt_tps=len(ids) / prompt_s if prompt_s > 0 else 0.0,
                generation_tps=stats.get("tokens", emitted_total) / rounds_s
                if rounds_s > 0
                else 0.0,
                tokens_per_check=stats.get("tokens_per_check"),
                stats=stats,
            )
            return

        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temp, top_p=top_p)
        for resp in mlx_stream_generate(
            self.model,
            self.tokenizer,
            ids,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            tid = int(resp.token)
            token_ids = [] if (resp.finish_reason == "stop" and tid in eos) else [tid]
            yield Chunk(
                text=resp.text,
                token_ids=token_ids,
                finish_reason=resp.finish_reason,
                prompt_tokens=resp.prompt_tokens,
                generation_tokens=resp.generation_tokens,
                prompt_tps=resp.prompt_tps,
                generation_tps=resp.generation_tps,
            )

    def stream(
        self,
        prompt: str | Sequence[int],
        *,
        max_tokens: int = 256,
        chat: bool = True,
        system: str | None = None,
        thinking: bool = False,
        temp: float = 0.0,
        top_p: float = 1.0,
        mode: str = "linear",
        block_size: int = 8,
        use_drafter: bool = True,
        adaptive: bool = True,
    ) -> Iterator[Chunk]:
        """Encode a prompt and stream output chunks."""
        ids = self.encode_prompt(
            prompt,
            chat=chat,
            system=system,
            thinking=thinking,
        )
        yield from self.stream_tokens(
            ids,
            max_tokens=max_tokens,
            temp=temp,
            top_p=top_p,
            mode=mode,
            block_size=block_size,
            use_drafter=use_drafter,
            adaptive=adaptive,
        )

    def stream_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int = 256,
        thinking: bool = False,
        temp: float = 0.0,
        top_p: float = 1.0,
        mode: str = "linear",
        block_size: int = 8,
        use_drafter: bool = True,
        adaptive: bool = True,
    ) -> Iterator[Chunk]:
        """Encode a chat message list and stream output chunks."""
        ids = self.encode_messages(messages, thinking=thinking)
        yield from self.stream_tokens(
            ids,
            max_tokens=max_tokens,
            temp=temp,
            top_p=top_p,
            mode=mode,
            block_size=block_size,
            use_drafter=use_drafter,
            adaptive=adaptive,
        )

    def _collect(self, chunks: Iterator[Chunk]) -> GenerationResult:
        parts: list[str] = []
        token_ids: list[int] = []
        last: Chunk | None = None
        for chunk in chunks:
            if chunk.text:
                parts.append(chunk.text)
            if chunk.token_ids:
                token_ids.extend(chunk.token_ids)
            last = chunk
        return GenerationResult(
            text="".join(parts),
            token_ids=token_ids,
            prompt_tokens=last.prompt_tokens if last else 0,
            generation_tokens=len(token_ids),
            prompt_tps=last.prompt_tps if last else 0.0,
            generation_tps=last.generation_tps if last else 0.0,
            finish_reason=last.finish_reason if last else None,
            tokens_per_check=last.tokens_per_check if last else None,
            stats=last.stats if last else None,
        )

    def generate(
        self,
        prompt: str | Sequence[int],
        *,
        max_tokens: int = 256,
        chat: bool = True,
        system: str | None = None,
        thinking: bool = False,
        temp: float = 0.0,
        top_p: float = 1.0,
        mode: str = "linear",
        block_size: int = 8,
        use_drafter: bool = True,
        adaptive: bool = True,
    ) -> GenerationResult:
        """Generate a complete response for a prompt string or token IDs."""
        return self._collect(
            self.stream(
                prompt,
                max_tokens=max_tokens,
                chat=chat,
                system=system,
                thinking=thinking,
                temp=temp,
                top_p=top_p,
                mode=mode,
                block_size=block_size,
                use_drafter=use_drafter,
                adaptive=adaptive,
            )
        )

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int = 256,
        thinking: bool = False,
        temp: float = 0.0,
        top_p: float = 1.0,
        mode: str = "linear",
        block_size: int = 8,
        use_drafter: bool = True,
        adaptive: bool = True,
    ) -> GenerationResult:
        """Generate a complete response for a list of chat messages."""
        return self._collect(
            self.stream_chat(
                messages,
                max_tokens=max_tokens,
                thinking=thinking,
                temp=temp,
                top_p=top_p,
                mode=mode,
                block_size=block_size,
                use_drafter=use_drafter,
                adaptive=adaptive,
            )
        )
