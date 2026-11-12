"""Stage `features`: one teacher-forced pass of the build over each training sample, saved as the per-sample files
SpecForge's offline reader takes ({id}.ckpt = torch.save of input_ids [T] int64, loss_mask [T] int64 with 1 on response
tokens, hidden_states [T, 8 * H] bfloat16, target_last_hidden_states [T, H] bfloat16). They go to local disk, about
46 KB a token, and a fresh container writes them again; a file that exists is kept."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from . import common


def _save(path, record) -> None:
    partial = f"{path}.part"  # the reader takes every name that ends in .ckpt
    torch.save(record, partial)
    os.replace(partial, path)


def run(root, s, status):
    out = Path(s.features_dir)
    out.mkdir(parents=True, exist_ok=True)
    owner = out / "owner.json"
    if owner.exists() and common.read_json(owner)["root"] != str(root):
        raise RuntimeError(f"{out} holds the features of {common.read_json(owner)['root']}; give this run its own features_dir")
    common.write_json(owner, {"root": str(root)})

    samples, short = [], 0
    for sample in common.load_text(root, "train"):
        length = min(sample["input_ids"].numel(), s.max_length)
        if length - sample["prompt_len"] < 2:  # the trainer refuses a sample without two response tokens in a row
            short += 1
            continue
        samples.append((sample, length))
    todo = sorted((x for x in samples if not (out / f"{x[0]['id']}.ckpt").exists()), key=lambda x: -x[1])
    if not todo:
        return {"count": len(samples), "left_out": short, "written_now": 0, "tokens_now": 0}

    model = common.load_model(s, "build", status)
    common.check_target_states(model, samples[0][0]["input_ids"], status)
    width = model.get_input_embeddings().weight.shape[1]
    tokens = sum(length for _, length in todo)
    needed, free = tokens * (len(s.target_layer_ids) + 1) * width * 2, shutil.disk_usage(out).free
    status.log(f"{len(todo)} of {len(samples)} samples to write ({short} left out: under two response tokens within "
               f"{s.max_length}); {tokens} tokens need {needed / 1e9:.0f} GB, {free / 1e9:.0f} GB free",
               samples=len(samples) - len(todo), total=len(samples))
    if needed > free:
        raise RuntimeError(f"{out} has {free / 1e9:.0f} GB free and the features need {needed / 1e9:.0f} GB")

    written, pending = 0, []
    with ThreadPoolExecutor(s.feature_writers) as pool:
        for start in range(0, len(todo), s.feature_batch):
            batch = todo[start:start + s.feature_batch]
            ids = torch.zeros(len(batch), batch[0][1], dtype=torch.long)  # right-padded: positions stay the sample's own
            mask = torch.zeros_like(ids)
            for i, (sample, length) in enumerate(batch):
                ids[i, :length] = sample["input_ids"][:length]
                mask[i, :length] = 1
            states, last = common.target_states(model, ids.to(model.device), mask.to(model.device), s.target_layer_ids)
            if states.dtype != torch.bfloat16 or not (torch.isfinite(states).all() and torch.isfinite(last).all()):
                raise RuntimeError(f"target states are {states.dtype} or not finite in the batch at {start}")
            states, last = states.cpu(), last.cpu()
            for future in pending:  # at most two batches are in memory
                future.result()
            pending = []
            for i, (sample, length) in enumerate(batch):
                loss_mask = torch.zeros(length, dtype=torch.long)
                loss_mask[sample["prompt_len"]:] = 1
                pending.append(pool.submit(_save, out / f"{sample['id']}.ckpt", {
                    "input_ids": sample["input_ids"][:length].clone(),
                    "loss_mask": loss_mask,
                    "hidden_states": states[i, :length].clone(),
                    "target_last_hidden_states": last[i, :length].clone(),
                }))
            written += sum(length for _, length in batch)
            if start % (20 * s.feature_batch) == 0 or start + s.feature_batch >= len(todo):
                rate = written / max(status.seconds(), 1e-9)
                status.log(f"{start + len(batch)} of {len(todo)} samples, {written} of {tokens} tokens, {rate:.0f} tokens/s",
                           samples=len(samples) - len(todo) + start + len(batch), total=len(samples))
        for future in pending:
            future.result()
    del model
    common.release()
    status.log(f"features took {status.seconds():.0f} s for {len(todo)} samples")
    return {"count": len(samples), "left_out": short, "written_now": len(todo), "tokens_now": tokens}
