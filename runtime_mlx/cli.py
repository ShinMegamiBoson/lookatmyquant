"""Command-line interface for running LookAtMyQuant models on Apple silicon.

lamq generate shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB -p "What is 17 * 23?"
lamq generate shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB --drafter shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter -p "Write merge_intervals in Python."
lamq chat q38-27b --drafter q38-27b-drafter
lamq info shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-4.7GB
"""

from __future__ import annotations

import argparse
import json
import sys

from .runtime import Runtime, inspect_model


def _add_runtime_args(
    parser: argparse.ArgumentParser, *, default_max_tokens: int
) -> None:
    parser.add_argument("model", help="local model directory or Hugging Face repo ID")
    parser.add_argument(
        "--drafter",
        default=None,
        help="optional DSpine drafter directory or Hugging Face repo ID",
    )
    parser.add_argument(
        "--mode",
        choices=("linear", "chat-fork", "copy-wide"),
        default="linear",
        help="speculative verifier strategy when --drafter is set",
    )
    parser.add_argument(
        "--block-size", type=int, default=8, help="draft block size (8 or 16)"
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
    parser.add_argument("--system", default=None, help="optional system message")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="enable thinking mode in the chat template",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip SHA-256 checksum verification when loading",
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

    rt = _load_from_args(args)
    if args.json:
        res = rt.generate(
            prompt,
            max_tokens=args.max_tokens,
            chat=not args.raw,
            system=args.system,
            thinking=args.thinking,
            temp=args.temp,
            top_p=args.top_p,
            mode=args.mode,
            block_size=args.block_size,
            adaptive=not args.no_adaptive,
        )
        print(
            json.dumps(
                {
                    "text": res.text,
                    "prompt_tokens": res.prompt_tokens,
                    "generation_tokens": res.generation_tokens,
                    "prompt_tps": round(res.prompt_tps, 2),
                    "generation_tps": round(res.generation_tps, 2),
                    "finish_reason": res.finish_reason,
                    "tokens_per_check": round(res.tokens_per_check, 3)
                    if res.tokens_per_check is not None
                    else None,
                },
                ensure_ascii=False,
            )
        )
        return 0

    last = None
    for chunk in rt.stream(
        prompt,
        max_tokens=args.max_tokens,
        chat=not args.raw,
        system=args.system,
        thinking=args.thinking,
        temp=args.temp,
        top_p=args.top_p,
        mode=args.mode,
        block_size=args.block_size,
        adaptive=not args.no_adaptive,
    ):
        if chunk.text:
            sys.stdout.write(chunk.text)
            sys.stdout.flush()
        last = chunk
    sys.stdout.write("\n")
    sys.stdout.flush()
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
        description="Run LookAtMyQuant quantized models on Apple silicon (MLX).",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="generate text from a prompt or stdin")
    _add_runtime_args(p_gen, default_max_tokens=256)
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

    p_chat = sub.add_parser("chat", help="interactive multi-turn chat in the terminal")
    _add_runtime_args(p_chat, default_max_tokens=512)
    p_chat.set_defaults(func=_cmd_chat)

    p_info = sub.add_parser(
        "info",
        help="inspect a packed model's format and architecture without loading weights",
    )
    p_info.add_argument("model", help="local model directory or Hugging Face repo ID")
    p_info.set_defaults(func=_cmd_info)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
