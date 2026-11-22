"""Command-line interface for running LookAtMyQuant Inkling-2-Mini models on NVIDIA GPUs.

    python -m runtime_cuda generate shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB -p "What is 17 * 23?" --stats
    python -m runtime_cuda info shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-11.2GB

The same arguments as `lamq generate` where they apply, plus `--no-thinking`; generation is greedy and serial (no drafter yet).
"""
from __future__ import annotations

import argparse
import json
import sys

from . import load, resolve_path, stream


def _cmd_generate(args: argparse.Namespace) -> int:
    prompt = args.prompt
    if prompt is None or prompt == "-":
        if sys.stdin.isatty() and prompt is None:
            print("error: pass -p/--prompt or pipe input on stdin", file=sys.stderr)
            return 2
        prompt = sys.stdin.read()
    model, tok = load(args.model, args.device)
    st: dict = {}
    pieces = []
    for piece in stream(model, tok, prompt, max_tokens=args.max_tokens, chat=not args.raw, system=args.system,
                        thinking=args.thinking, no_thinking=args.no_thinking, stats_out=st):
        if args.json:
            pieces.append(piece)
        else:
            sys.stdout.write(piece)
            sys.stdout.flush()
    if args.json:
        print(json.dumps({"text": "".join(pieces), **st}, ensure_ascii=False))
        return 0
    sys.stdout.write("\n")
    if args.stats and st:
        print(f"[prompt={st['prompt_tokens']} tok ({st['prefill_tok_s']:.1f} tok/s) | gen={st['generated_tokens']} tok "
              f"({st['decode_tok_s']:.1f} tok/s, {st['median_step_ms']:.2f} ms/step) | peak {st['peak_gb']:.2f} GB]", file=sys.stderr)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    import torch

    from .inkling.schema import validate_packed_dir
    from .inkling.tuning import describe

    path = resolve_path(args.model)
    print(json.dumps({"path": str(path), "formats": validate_packed_dir(path)}, indent=1, sort_keys=True))
    if torch.cuda.is_available():
        print(describe(torch.device(args.device)))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m runtime_cuda", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate text from a prompt")
    g.add_argument("model", help="local model directory or Hugging Face repo ID")
    g.add_argument("-p", "--prompt", default=None, help="prompt text ('-' or omitted: read stdin)")
    g.add_argument("--max-tokens", type=int, default=256)
    g.add_argument("--system", default=None, help="optional system message")
    think = g.add_mutually_exclusive_group()
    think.add_argument("--thinking", action="store_true", help="enable thinking in the chat template")
    think.add_argument("--no-thinking", action="store_true",
                       help="never open the thinking channel (without either flag the model still thinks briefly)")
    g.add_argument("--raw", action="store_true", help="feed the prompt as-is, without the chat template")
    g.add_argument("--json", action="store_true", help="print one JSON object with the text and statistics")
    g.add_argument("--stats", action="store_true", help="print throughput to stderr")
    g.add_argument("--device", default="cuda")
    g.set_defaults(fn=_cmd_generate)
    i = sub.add_parser("info", help="check a packed model's formats and show the GPU tuning in use")
    i.add_argument("model")
    i.add_argument("--device", default="cuda")
    i.set_defaults(fn=_cmd_info)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
