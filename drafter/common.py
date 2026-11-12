"""Shared by the stages: JSON files, the status file and progress lines, the chat template, the two targets, and the one
teacher-forced pass that gives a target's states."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import threading
import time
from pathlib import Path


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    os.replace(partial, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Status:
    """ROOT/status.json (stage, state, progress counts, seconds, last line) and the progress lines on stdout."""

    def __init__(self, root):
        self.path = Path(root) / "status.json"
        self.lock = threading.Lock()
        self.stage, self.state, self.progress, self.last = "", "running", {}, ""
        self.started = time.time()
        self.saved = 0.0

    def begin(self, stage: str) -> None:
        self.stage, self.state, self.progress, self.started = stage, "running", {}, time.time()
        self.log("start")

    def seconds(self) -> float:
        return round(time.time() - self.started, 1)

    def set(self, **progress) -> None:
        with self.lock:
            self.progress.update(progress)

    def log(self, message: str, **progress) -> None:
        self.set(**progress)
        self.echo(f"[{self.stage} {self.seconds():.1f}s] {message}", every=0)

    def echo(self, line: str, every: float = 5) -> None:
        """Print a line and write the status file, unless it was written less than `every` seconds ago."""
        print(line, flush=True)
        with self.lock:
            self.last = line
            if time.time() - self.saved < every:
                return
            self.saved = time.time()
            write_json(self.path, {
                "stage": self.stage,
                "state": self.state,
                "progress": self.progress,
                "seconds": self.seconds(),
                "last": self.last,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

    def end(self, state: str, message: str) -> None:
        self.state = state
        self.log(message)


def run_stream(command, status: Status, on_line=None) -> int:
    """Run a command, print each line it writes with the elapsed seconds, and return its exit code."""
    status.log("run: " + " ".join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                               bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    for line in process.stdout:
        line = line.rstrip()
        if on_line is not None:
            on_line(line)
        status.echo(f"[{status.stage} {status.seconds():.1f}s] | {line}")
    return process.wait()


def read_prompts(root) -> list:
    with open(Path(root) / "assets" / "prompts.jsonl", encoding="utf-8") as lines:
        return [json.loads(line) for line in lines]


def load_text(root, prefix: str) -> list:
    """Every sample of ROOT/text/{prefix}_NNNNN.pt: {id, input_ids, prompt_len}, plus its prompt's source."""
    import torch

    sources = {prompt["id"]: prompt.get("source") for prompt in read_prompts(root)}
    samples = []
    for path in sorted((Path(root) / "text").glob(f"{prefix}_[0-9][0-9][0-9][0-9][0-9].pt")):
        samples += torch.load(path)
    for sample in samples:
        sample["source"] = sources[sample["id"]]
    return samples


def tokenizer(s):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(s.original_dir)


def prompt_ids(tok, text: str) -> list:
    """One user turn under the model's chat template with thinking off, ending where the assistant starts."""
    chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return tok(chat, add_special_tokens=False)["input_ids"]


def load_model(s, which: str, status: Status, device: str = "cuda"):
    """The original Qwen3.5-4B in bfloat16, or (which="build") the same architecture with the build's weights copied in."""
    import torch
    from transformers import AutoModelForCausalLM

    if which not in ("original", "build"):
        raise ValueError(which)
    model = AutoModelForCausalLM.from_pretrained(s.original_dir, dtype=torch.bfloat16)
    if which == "build":
        params = torch.load(s.build_weights, map_location="cpu", weights_only=False)["params"]
        named = dict(model.named_parameters())
        unknown = sorted(set(params) - set(named))
        if unknown:
            raise RuntimeError(f"{s.build_weights} has {len(unknown)} names the model lacks, e.g. {unknown[:5]}")
        kept = sorted(set(named) - set(params))
        if kept:
            status.log(f"WARNING {len(kept)} parameters are not in the build file and keep the original's weights, "
                       f"e.g. {kept[:5]}")
        with torch.no_grad():
            for name, tensor in params.items():
                if tensor.shape != named[name].shape:
                    raise RuntimeError(f"{name}: build {tuple(tensor.shape)}, model {tuple(named[name].shape)}")
                named[name].copy_(tensor.to(named[name].dtype))
        del params
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise RuntimeError("the embedding and the output head are not tied")
    status.log(f"loaded the {which} model")
    return model.to(device).eval()


def release() -> None:
    """Free GPU memory after the caller dropped its model."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def target_states(model, input_ids, attention_mask, layer_ids):
    """One teacher-forced pass without the output head. Returns the drafter's context features [B, T, layers * H] (layer
    id i is transformers hidden_states[i + 1], concatenated in the order given) and the final state after the last norm
    [B, T, H], the one the output head reads."""
    import torch

    with torch.no_grad():
        out = model.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                               use_cache=False)
    return torch.cat([out.hidden_states[i + 1] for i in layer_ids], dim=-1), out.last_hidden_state


def check_target_states(model, ids, status: Status) -> None:
    """Fail unless target_states reads what it claims: one state per layer plus the embeddings, and a final state from
    which the output head picks the same tokens as the whole model does."""
    import torch

    ids = ids[:64][None].to(model.device)
    with torch.no_grad():
        out = model.base_model(input_ids=ids, output_hidden_states=True, use_cache=False)
        layers = getattr(model.config, "text_config", model.config).num_hidden_layers
        if len(out.hidden_states) != layers + 1:
            raise RuntimeError(f"{len(out.hidden_states)} hidden states for {layers} layers")
        whole = model(input_ids=ids, use_cache=False).logits.argmax(-1)
        ours = model.get_output_embeddings()(out.last_hidden_state).argmax(-1)
    same = (whole == ours).float().mean().item()
    status.log(f"target states: the head on base_model's last_hidden_state picks the model's token at {same:.1%} of positions")
    if same < 0.9:
        raise RuntimeError("base_model's last_hidden_state is not what the output head reads")
