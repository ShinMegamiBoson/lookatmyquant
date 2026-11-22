"""Command-line interface for running LookAtMyQuant models on Apple silicon and NVIDIA CUDA.

lamq generate shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB -p "What is 17 * 23?"
lamq generate shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB --drafter shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter -p "Write merge_intervals in Python."
lamq chat q38-27b --drafter q38-27b-drafter
lamq info shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-4.7GB
lamq serve inkling-5.4gb --port 8080
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from contextlib import closing

from .runtime import Chunk, Runtime, encode_prompt, inspect_model


def _add_runtime_args(
    parser: argparse.ArgumentParser, *, default_max_tokens: int, allow_cuda: bool = False
) -> None:
    parser.add_argument("model", help="local model directory or Hugging Face repo ID")
    parser.add_argument(
        "--drafter",
        default=None,
        help="optional DSpark or DSpine drafter directory or Hugging Face repo ID",
    )
    parser.add_argument(
        "--mode",
        choices=("linear", "chat-fork", "copy-wide"),
        default=None if allow_cuda else "linear",
        help="MLX speculative verifier strategy when --drafter is set",
    )
    parser.add_argument(
        "--block-size", type=int, default=None if allow_cuda else 8, help="MLX draft block size (8 or 16)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=default_max_tokens,
        help="maximum tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=0.0, help="sampling temperature (0.0 for greedy)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="nucleus sampling probability when --temp > 0",
    )
    parser.add_argument(
        "--prefill-chunk", type=int, default=None if allow_cuda else -1,
        help="Inkling prompt tokens per prefill chunk (-1 automatic, 0 one pass)",
    )
    parser.add_argument("--system", default=None, help="optional system message")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="enable thinking mode in the chat template",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip local checksum verification for opgraph-packed-1 models",
    )
    parser.add_argument(
        "--full-vocab",
        action="store_true",
        help="score the full vocabulary for drafter proposals",
    )
    parser.add_argument(
        "--dense-drafter",
        action="store_true",
        help="keep drafter weights dense float16 instead of 8-bit rows",
    )
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="disable adaptive serial fallback when drafter acceptance drops",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="print throughput and acceptance statistics to stderr",
    )


def _load_from_args(args: argparse.Namespace) -> Runtime:
    return Runtime.load(
        args.model,
        drafter=args.drafter,
        verify=not args.no_verify,
        drafter_int8=not args.dense_drafter,
        full_vocab=args.full_vocab,
    )


def _format_stats(last) -> str:
    parts = [
        f"prompt={last.prompt_tokens} tok ({last.prompt_tps:.1f} tok/s)",
        f"gen={last.generation_tokens} tok ({last.generation_tps:.1f} tok/s)",
    ]
    if last.tokens_per_check is not None:
        parts.append(f"tok/check={last.tokens_per_check:.2f}")
    if last.stats and last.stats.get("serial_steps"):
        parts.append(f"serial={last.stats['serial_steps']}")
    if last.finish_reason:
        parts.append(f"stop={last.finish_reason}")
    return " | ".join(parts)


def _cmd_generate(args: argparse.Namespace) -> int:
    prompt = args.prompt
    if prompt is None or prompt == "-":
        if sys.stdin.isatty() and prompt is None:
            print("error: pass -p/--prompt or pipe input on stdin", file=sys.stderr)
            return 2
        prompt = sys.stdin.read()

    if args.device != "mlx":
        try:
            with closing(_stream_cuda(args, prompt)) as chunks:
                return _write_generation(args, chunks)
        except (ValueError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    rt = _load_from_args(args)
    with closing(rt.stream(
        prompt,
        max_tokens=args.max_tokens,
        chat=not args.raw,
        system=args.system,
        thinking=args.thinking,
        temp=args.temp,
        top_p=args.top_p,
        mode=args.mode or "linear",
        block_size=8 if args.block_size is None else args.block_size,
        adaptive=not args.no_adaptive,
        prefill_chunk=-1 if args.prefill_chunk is None else args.prefill_chunk,
    )) as chunks:
        return _write_generation(args, chunks)


def _stream_cuda(args: argparse.Namespace, prompt: str) -> Iterator[Chunk]:
    from tokenizers.decoders import DecodeStream

    from runtime_cuda import eos_ids, load_drafter, open_model, stream_tokens

    with open_model(
        args.model, device=args.device, engine=args.engine or "auto", revision=args.revision,
        drafter=args.drafter,
    ) as (model, tokenizer):
        if args.raw:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
        else:
            ids = encode_prompt(tokenizer, prompt, system=args.system, thinking=args.thinking)
        context = len(ids) + args.max_tokens - 1
        if not ids or context > model.context_limit:
            raise ValueError("prompt is empty or prompt and generation exceed the model context")
        drafter = load_drafter(model, args.drafter, context=context) if args.drafter else None
        stop_ids = eos_ids(model, tokenizer)
        decoder = DecodeStream(skip_special_tokens=True)
        stats: dict = {}
        count, finish = 0, "length"
        with closing(stream_tokens(
            model, ids, max_tokens=args.max_tokens, eos_ids=stop_ids,
            drafter=drafter, draft_block_size=args.draft_block_size or 2, temperature=args.temp,
            seed=args.seed, stats_out=stats,
        )) as tokens:
            for tid in tokens:
                if tid in stop_ids:
                    finish = "stop"
                    break
                count += 1
                piece = decoder.step(tokenizer.backend_tokenizer, tid) or ""
                if piece:
                    yield Chunk(text=piece)
        prefill_s = stats.get("prefill_seconds") or stats.get("time_to_first_token_seconds")
        checks = stats.get("draft_checks")
        yield Chunk(
            text="",
            prompt_tokens=len(ids),
            generation_tokens=count,
            prompt_tps=len(ids) / prefill_s if prefill_s else 0.0,
            generation_tps=stats.get("decode_tokens_per_second") or 0.0,
            finish_reason=finish,
            tokens_per_check=(1 + stats["accepted_tokens"] / checks) if checks else None,
            stats=stats,
        )


def _write_generation(args: argparse.Namespace, chunks: Iterator[Chunk]) -> int:
    parts = []
    last = None
    for chunk in chunks:
        if chunk.text:
            if args.json:
                parts.append(chunk.text)
            else:
                print(chunk.text, end="", flush=True)
        last = chunk
    if args.json:
        result = last if last is not None else Chunk(text="")
        print(json.dumps({
            "text": "".join(parts),
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "prompt_tps": round(result.prompt_tps, 2),
            "generation_tps": round(result.generation_tps, 2),
            "finish_reason": result.finish_reason,
            "tokens_per_check": round(result.tokens_per_check, 3) if result.tokens_per_check is not None else None,
        }, ensure_ascii=False))
    else:
        print(flush=True)
        if args.stats and last is not None:
            print(f"[{_format_stats(last)}]", file=sys.stderr)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    rt = _load_from_args(args)
    base_messages: list[dict[str, str]] = []
    if args.system:
        base_messages.append({"role": "system", "content": args.system})
    messages = list(base_messages)
    print("Interactive chat ready. Commands: /reset, /exit", file=sys.stderr)
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            messages = list(base_messages)
            print("Conversation cleared.", file=sys.stderr)
            continue
        messages.append({"role": "user", "content": user})
        parts: list[str] = []
        last = None
        try:
            for chunk in rt.stream_chat(
                messages,
                max_tokens=args.max_tokens,
                thinking=args.thinking,
                temp=args.temp,
                top_p=args.top_p,
                mode=args.mode,
                block_size=args.block_size,
                adaptive=not args.no_adaptive,
                prefill_chunk=args.prefill_chunk,
            ):
                if chunk.text:
                    sys.stdout.write(chunk.text)
                    sys.stdout.flush()
                    parts.append(chunk.text)
                last = chunk
        except KeyboardInterrupt:
            print(file=sys.stderr)
        sys.stdout.write("\n")
        sys.stdout.flush()
        if args.stats and last is not None:
            print(f"[{_format_stats(last)}]", file=sys.stderr)
        messages.append({"role": "assistant", "content": "".join(parts)})
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_model(args.model), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="lamq",
        description="Run LookAtMyQuant quantized models on Apple silicon or NVIDIA CUDA.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="generate text from a prompt or stdin")
    _add_runtime_args(p_gen, default_max_tokens=256, allow_cuda=True)
    p_gen.add_argument("--device", default="mlx", help="mlx (default), cuda, or cuda:N for GPU N")
    p_gen.add_argument("--engine", choices=("auto", "native", "triton"), help="CUDA backend (default auto)")
    p_gen.add_argument("--revision", help="Hugging Face model revision for CUDA")
    p_gen.add_argument("--seed", type=int, help="random seed for CUDA sampling")
    p_gen.add_argument("--draft-block-size", type=int, help="CUDA DSpark draft block size (default 2)")
    p_gen.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="prompt text (omit or pass '-' to read stdin)",
    )
    p_gen.add_argument(
        "--raw",
        action="store_true",
        help="send the prompt verbatim without applying a chat template",
    )
    p_gen.add_argument(
        "--json", action="store_true", help="emit JSON output instead of streaming text"
    )
    p_gen.set_defaults(func=_cmd_generate)

    p_chat = sub.add_parser("chat", help="interactive multi-turn chat on Apple silicon (MLX)")
    _add_runtime_args(p_chat, default_max_tokens=512)
    p_chat.set_defaults(func=_cmd_chat)

    p_info = sub.add_parser(
        "info",
        help="inspect a packed model's format and architecture without loading weights",
    )
    p_info.add_argument("model", help="local model directory or Hugging Face repo ID")
    p_info.set_defaults(func=_cmd_info)

    from .server import add_serve_args, serve

    p_serve = sub.add_parser("serve", help="OpenAI-compatible HTTP server (packed Inkling-2-Mini models)")
    add_serve_args(p_serve)
    p_serve.set_defaults(func=serve)

    return ap


def _validate_generate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.device == "mlx":
        for name in ("engine", "revision", "seed", "draft_block_size"):
            if getattr(args, name) is not None:
                parser.error(f"--{name.replace('_', '-')} requires a CUDA device")
        return
    if re.fullmatch(r"cuda(?::[0-9]+)?", args.device) is None:
        parser.error("--device must be mlx, cuda, or cuda:N")
    for name in ("mode", "block_size", "prefill_chunk"):
        if getattr(args, name) is not None:
            parser.error(f"--{name.replace('_', '-')} is available only with MLX")
    for name in ("no_verify", "full_vocab", "dense_drafter", "no_adaptive"):
        if getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is available only with MLX")
    if args.top_p != 1:
        parser.error("CUDA sampling supports --temp and --seed. Leave --top-p at 1")
    if not 0 <= args.temp <= 2:
        parser.error("--temp must be in [0, 2] for CUDA")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive for CUDA")
    if args.seed is not None and not -(1 << 63) <= args.seed < (1 << 64):
        parser.error("--seed must be a 64-bit integer")
    if args.draft_block_size is not None and (not args.drafter or args.draft_block_size < 2):
        parser.error("--draft-block-size requires --drafter and a size of at least two")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        _validate_generate(args, parser)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
