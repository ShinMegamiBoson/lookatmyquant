"""Run Inkling CUDA inference or check NVIDIA benchmark telemetry."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np

from .device_model import DeviceModel
from .driver import Cuda
from .model import Model


PACKS = {
    "5.4GB": "shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB",
    "8.9GB": "shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB",
}
PACK_REVISIONS = {
    "5.4GB": "51a8664233ff1f59b16d5ae0492cc16ee7a37ae4",
    "8.9GB": "1318b804e687f3761ced445b7a2a13ab305f82e5",
}
DRAFTER_PACK = "shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter"
DRAFTER_REVISION = "61370dcd233fdfe3b7e5e0cafe652090ca343e00"


def download(path, size, revision):
    from huggingface_hub import snapshot_download

    revision = revision or PACK_REVISIONS[size]
    snapshot_download(PACKS[size], revision=revision, local_dir=path, allow_patterns=[
        "*.json", "*.npz", "steering.safetensors", "*.jinja", "tokenizer*", "vocab.*", "merges.txt",
    ])


def download_drafter(path, revision):
    from huggingface_hub import snapshot_download

    snapshot_download(DRAFTER_PACK, revision=revision or DRAFTER_REVISION, local_dir=path,
                      allow_patterns=["config.json", "manifest.json", "mtp_4bit.safetensors"])


def greedy(session, inputs, limit, eos, on_device):
    def predict(token):
        if on_device:
            return session.predict(token)
        scores = session.step(token)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Non-finite output logits")
        return int(scores.argmax())

    started = time.perf_counter()
    if on_device:
        session.prefill(inputs[:-1])
    else:
        for token in inputs[:-1]:
            session.step(token, logits=False)
    output = [predict(inputs[-1])]
    prefill_seconds = time.perf_counter() - started
    started = time.perf_counter()
    while len(output) < limit and output[-1] not in eos:
        output.append(predict(output[-1]))
    decode_seconds = time.perf_counter() - started
    return output, prefill_seconds, (len(output) - 1) / decode_seconds if len(output) > 1 else None


def generate(args):
    tokenizer = None
    if args.tokens is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True, reasoning_effort="none",
        )
        ids = tokenizer.encode(ids, add_special_tokens=False)
    else:
        ids = args.tokens
    if len(ids) + args.max_new_tokens - 1 > args.context:
        raise ValueError("Prompt and generation exceed the context limit")
    eos = []
    generation_config = args.model / "generation_config.json"
    if generation_config.exists():
        eos = json.loads(generation_config.read_text(encoding="utf-8")).get("eos_token_id", [])
        if isinstance(eos, int):
            eos = [eos]
    monitor = None
    with ExitStack() as stack:
        cuda = stack.enter_context(Cuda(args.device))
        if args.check_telemetry:
            from .telemetry import TelemetryWatch, report

            monitor = stack.enter_context(TelemetryWatch(cuda.pci_bus_id()))
        if monitor is not None and monitor.samples[0]["issues"]:
            result = {"device": cuda.device_name(), "decode_tokens_per_second": None}
        else:
            result = generate_result(args, cuda, stack, ids, eos, tokenizer)
    if monitor is not None:
        result["telemetry"] = report(monitor.device, monitor.samples)
        if not result["telemetry"]["valid"]:
            result["decode_tokens_per_second"] = None
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if monitor is not None and not result["telemetry"]["valid"]:
        raise SystemExit(2)


def generate_result(args, cuda, stack, ids, eos, tokenizer):
    on_device = args.backend == "cuda"
    started = time.perf_counter()
    model = (DeviceModel if on_device else Model)(args.model, cuda)
    if on_device:
        stack.callback(model.close)
    load_seconds = time.perf_counter() - started
    session = model.session(args.context)
    if on_device:
        stack.enter_context(session)
    draft_stats = {}
    if args.drafter:
        from .drafter import speculative_greedy

        output, prefill_seconds, decode_rate, draft_stats = speculative_greedy(
            args.drafter, session, args.device, ids, args.max_new_tokens, set(eos), args.draft_block_size,
        )
    else:
        output, prefill_seconds, decode_rate = greedy(session, ids, args.max_new_tokens, eos, on_device)
    return {
        "device": cuda.device_name(),
        "backend": args.backend,
        "resident_weights_bytes": cuda.resident_bytes,
        "logical_visits_per_token": model.cfg.total_visits,
        "prompt_tokens": ids,
        "generated_tokens": output,
        "text": tokenizer.decode(output, skip_special_tokens=True) if tokenizer else None,
        "load_seconds": load_seconds,
        "prefill_seconds": prefill_seconds,
        "decode_tokens_per_second": decode_rate,
        **draft_stats,
    }


def check_telemetry(args):
    from .telemetry import read_sample, report

    samples = []
    for index in range(args.samples):
        if index:
            time.sleep(1)
        samples.append(read_sample(args.device))
    result = report(args.device, samples)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["valid"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True, dest="command")
    fetch = sub.add_parser("download")
    fetch.add_argument("model", type=Path)
    fetch.add_argument("--size", choices=PACKS, default="5.4GB")
    fetch.add_argument("--revision")
    draft = sub.add_parser("download-drafter")
    draft.add_argument("path", type=Path)
    draft.add_argument("--revision")
    run = sub.add_parser("generate")
    run.add_argument("model", type=Path)
    prompt = run.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--tokens", nargs="+", type=int)
    run.add_argument("--context", type=int, default=256)
    run.add_argument("--max-new-tokens", type=int, default=32)
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--backend", choices=("cuda", "hybrid"), default="cuda")
    run.add_argument("--drafter", type=Path)
    run.add_argument("--draft-block-size", type=int, choices=range(2, 9))
    run.add_argument("--check-telemetry", action="store_true", help="Withhold decode score if NVIDIA telemetry is invalid")
    run.add_argument("--json", type=Path)
    check = sub.add_parser("check-telemetry", help="Sample NVIDIA telemetry without loading a CUDA model")
    check.add_argument("--device", default="0", help="nvidia-smi GPU index, UUID or PCI bus ID")
    check.add_argument("--samples", type=int, default=3)
    server = sub.add_parser("serve")
    server.add_argument("model", type=Path)
    server.add_argument("--name")
    server.add_argument("--port", type=int, default=8098)
    server.add_argument("--context", type=int, default=16_384)
    server.add_argument("--device", type=int, default=0)
    server.add_argument("--drafter", type=Path, help="Enable DSpark using the published drafter pack")
    server.add_argument("--draft-block-size", type=int, choices=range(2, 9),
                        help="DSpark verify block size (default 2)")
    server.add_argument("--no-thinking", action="store_true",
                        help="Exclude the thinking-channel token during generation")
    args = parser.parse_args()
    if args.command == "download":
        download(args.model, args.size, args.revision)
    elif args.command == "download-drafter":
        download_drafter(args.path, args.revision)
    elif args.command == "check-telemetry":
        if args.samples < 1:
            parser.error("--samples must be positive")
        check_telemetry(args)
    elif args.command == "serve":
        if args.context < 2:
            parser.error("--context must be at least two tokens")
        if not 0 <= args.port <= 65_535:
            parser.error("--port must be between 0 and 65535")
        if args.draft_block_size and not args.drafter:
            parser.error("--draft-block-size requires --drafter")
        from .serve import serve

        serve(args)
    else:
        if args.max_new_tokens < 1:
            parser.error("--max-new-tokens must be positive")
        if args.drafter and args.backend != "cuda":
            parser.error("--drafter requires the cuda backend")
        if args.draft_block_size and not args.drafter:
            parser.error("--draft-block-size requires --drafter")
        generate(args)


if __name__ == "__main__":
    main()
