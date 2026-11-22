"""CUDA runtime for LookAtMyQuant packed Inkling-2-Mini models on NVIDIA GPUs (the counterpart of `runtime_mlx`).

    from runtime_cuda import load, generate

    model, tok = load("shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB")
    print(generate(model, tok, "What is 17 times 23? Show the steps.", max_tokens=200))
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

THINKING_TOKEN = "<|content_thinking|>"  # opens the model's thinking channel


def resolve_path(path_or_repo: str | Path) -> Path:
    """A local directory for a filesystem path or a Hugging Face repo ID (as `runtime_mlx.runtime.resolve_path`)."""
    path = Path(path_or_repo).expanduser()
    if path.exists():
        return path
    spec = str(path_or_repo)
    if spec.count("/") == 1 and not spec.startswith((".", "/", "~")):
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=spec))
    raise FileNotFoundError(f"model path does not exist: {path_or_repo}")


def load(path_or_repo: str | Path, device: str = "cuda"):
    """Returns `(model, tokenizer)`: the packed model on one CUDA device and its tokenizer."""
    from transformers import AutoTokenizer

    from .inkling.model import Inkling2LoopPacked

    path = resolve_path(path_or_repo)
    return Inkling2LoopPacked(path, device), AutoTokenizer.from_pretrained(path)


def encode_messages(tokenizer, messages: Sequence[dict[str, str]], *, thinking: bool = False) -> list[int]:
    """Chat messages to token ids with the model's chat template (Inkling's takes `reasoning_effort`)."""
    tpl = getattr(tokenizer, "chat_template", "") or ""
    kw = {"reasoning_effort": "low" if thinking else "none"} if "reasoning_effort" in tpl else {"enable_thinking": thinking}
    enc = tokenizer.apply_chat_template(list(messages), tokenize=True, add_generation_prompt=True, return_dict=True, **kw)
    return [int(x) for x in enc["input_ids"]]


def stream(model, tokenizer, prompt: str, *, max_tokens: int = 256, chat: bool = True, system: str | None = None,
           thinking: bool = False, no_thinking: bool = False, stats_out: dict | None = None):
    """Greedy generation; yields text pieces as tokens arrive. With thinking off the model still opens a short
    thinking channel; `no_thinking` never lets it pick the token that opens it."""
    if thinking and no_thinking:
        raise ValueError("thinking and no_thinking are exclusive")
    from .inkling.generate import stream_serial

    if chat:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        ids = encode_messages(tokenizer, msgs, thinking=thinking)
    else:
        ids = tokenizer(prompt).input_ids
    eos = getattr(tokenizer, "eos_token_id", None)
    out: list[int] = []
    shown = ""
    blocked = tokenizer.convert_tokens_to_ids([THINKING_TOKEN]) if no_thinking else ()
    for chunk in stream_serial(model, ids, max_tokens=max_tokens, eos_ids=(eos,) if eos is not None else (200006,),
                               blocked=blocked, stats_out=stats_out):
        out += [t for t in chunk if t != eos]
        text = tokenizer.decode(out, skip_special_tokens=True)
        if text.startswith(shown) and len(text) > len(shown):
            yield text[len(shown):]
            shown = text


def generate(model, tokenizer, prompt: str, **kw) -> str:
    return "".join(stream(model, tokenizer, prompt, **kw))
