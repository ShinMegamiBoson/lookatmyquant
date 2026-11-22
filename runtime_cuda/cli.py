"""CUDA download, inspection and benchmark commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .downloads import DRAFTER_REPO, MODEL_REVISIONS, resolve_path


def _download(args):
    path = resolve_path(args.repo, revision=args.revision, drafter=args.drafter,
                        target=args.target, local_dir=args.directory)
    print(path)
    return 0


def _info(args):
    from .inkling.schema import validate_packed_dir

    path = resolve_path(args.model, revision=args.revision)
    print(json.dumps({"path": str(path), "formats": validate_packed_dir(path)}, indent=2, sort_keys=True))
    return 0


def benchmark_result(args):
    from . import eos_ids, load_drafter, open_model, stream_tokens

    started = time.perf_counter()
    with open_model(args.model, device=args.device, engine=args.engine, revision=args.revision, drafter=args.drafter) as (model, tokenizer):
        load_seconds = time.perf_counter() - started
        if args.tokens is not None:
            inputs = args.tokens
        else:
            inputs = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
                                                   tokenize=False, add_generation_prompt=True,
                                                   reasoning_effort=args.reasoning_effort)
            inputs = tokenizer.encode(inputs, add_special_tokens=False)
        context = len(inputs) + args.max_tokens - 1
        drafter = load_drafter(model, args.drafter, context=context) if args.drafter else None
        stats = {}
        output = list(stream_tokens(model, inputs, max_tokens=args.max_tokens, eos_ids=eos_ids(model, tokenizer),
                                    drafter=drafter, draft_block_size=args.draft_block_size, temperature=args.temperature,
                                    seed=args.seed, stats_out=stats))
        return {**stats, "prompt_tokens": inputs, "generated_tokens": output, "load_seconds": load_seconds,
                "text": tokenizer.decode(output, skip_special_tokens=True)}


def benchmark(args):
    from .inkling_cuda.telemetry import TelemetryWatch, report

    if args.check_telemetry:
        device = str(args.device).removeprefix("cuda:") if args.device != "cuda" else "0"
        with TelemetryWatch(device) as monitor:
            if monitor.samples[0]["issues"]:
                result = {}
            else:
                result = benchmark_result(args)
        result["telemetry"] = report(monitor.device, monitor.samples)
        if not result["telemetry"]["valid"]:
            result["decode_tokens_per_second"] = None
    else:
        result = benchmark_result(args)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    return 2 if result.get("telemetry", {}).get("valid") is False else 0


def telemetry(args):
    from .inkling_cuda.telemetry import read_sample, report

    samples = []
    for index in range(args.samples):
        if index:
            time.sleep(1)
        samples.append(read_sample(args.device))
    result = report(args.device, samples)
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 2


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("generate", "serve"):
        from runtime_mlx.cli import main as lamq_main

        if not any(arg == "--device" or arg.startswith("--device=") for arg in argv):
            argv[1:1] = ["--device", "cuda"]
        return lamq_main(argv)
    parser = argparse.ArgumentParser(prog="python -m runtime_cuda")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="download and verify a packed model or DSpark drafter")
    download.add_argument("repo", help=f"local release size ({', '.join(MODEL_REVISIONS)}) or Hugging Face repo ID")
    download.add_argument("--directory", type=Path)
    download.add_argument("--revision")
    download.add_argument("--drafter", action="store_true", help=f"treat the repository as a drafter, e.g. {DRAFTER_REPO}")
    download.add_argument("--target", help="target release size for choosing published DSpark weights")
    download.set_defaults(fn=_download)
    info = sub.add_parser("info", help="validate and inspect a packed model")
    info.add_argument("model")
    info.add_argument("--revision")
    info.set_defaults(fn=_info)
    bench = sub.add_parser("benchmark", help="report raw token IDs and timing, optionally gated by NVIDIA telemetry")
    bench.add_argument("model")
    prompt = bench.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--tokens", type=int, nargs="+")
    bench.add_argument("--device", default="cuda")
    bench.add_argument("--engine", choices=("auto", "native", "triton"), default="auto")
    bench.add_argument("--revision")
    bench.add_argument("--max-tokens", type=int, default=32)
    bench.add_argument("--reasoning-effort", default="none")
    bench.add_argument("--temperature", type=float, default=0.)
    bench.add_argument("--seed", type=int)
    bench.add_argument("--drafter")
    bench.add_argument("--draft-block-size", type=int, default=2)
    bench.add_argument("--check-telemetry", action="store_true")
    bench.add_argument("--json", type=Path)
    bench.set_defaults(fn=benchmark)
    monitor = sub.add_parser("check-telemetry", help="check NVIDIA telemetry for independent runs")
    monitor.add_argument("--device", default="0")
    monitor.add_argument("--samples", type=int, default=3)
    monitor.set_defaults(fn=telemetry)
    args = parser.parse_args(argv)
    if getattr(args, "samples", 1) < 1:
        parser.error("--samples must be positive")
    return args.fn(args)
