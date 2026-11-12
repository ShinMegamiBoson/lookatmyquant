"""Stage `train`: SpecForge's DSpine trainer on the local features, then the export.

SpecForge writes its checkpoints (about 9 GB each) to local disk. A watcher copies each new one to ROOT/train, which
holds two slots and a spare: the copy goes into slot_spare, and when it is whole the spare changes names with the older
slot, whose directory is the next spare. So one whole checkpoint is always on ROOT and nothing there is deleted. A slot is
whole when its complete.json names a step. A later start copies the newest slot back and resumes from it. The finished run
is ROOT/train/final and, in Hugging Face form, ROOT/export."""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

from . import common, settings

STATE = "training_state.pt"
FIXED_MTIME_NS = 1_700_000_000 * 10**9  # see _local_target


def _whole(directory):
    """complete.json of a copied checkpoint, or None while the copy is missing or unfinished."""
    marker = Path(directory) / "complete.json"
    if not marker.exists():
        return None
    record = common.read_json(marker)
    return record if isinstance(record.get("step"), int) else None


def _copy_checkpoint(source, target, step: int) -> None:
    """Copy a checkpoint directory file by file over whatever `target` holds; complete.json names the step last."""
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    common.write_json(target / "complete.json", {"step": None})
    files = {}
    for item in sorted(source.iterdir()):
        if item.is_file() and not item.name.endswith(".tmp"):
            partial = target / (item.name + ".part")
            shutil.copyfile(item, partial)
            os.replace(partial, target / item.name)
            files[item.name] = item.stat().st_size
    common.write_json(target / "complete.json", {"step": step, "name": source.name, "files": files})


def _slots(train):
    return [train / "slot_a", train / "slot_b"]


def _recover(train) -> None:
    """Finish a swap that a stopped container left half done."""
    spare, moved = train / "slot_spare", train / "slot_moved"
    if moved.exists():
        if spare.exists():  # stopped between the two renames: the spare is the whole new copy
            os.rename(spare, next(slot for slot in _slots(train) if not slot.exists()))
        os.rename(moved, spare)


def _to_slot(train, checkpoint, step: int, status) -> None:
    spare = train / "slot_spare"
    _copy_checkpoint(checkpoint, spare, step)
    older = min(_slots(train), key=lambda slot: (_whole(slot) or {"step": -1})["step"] if slot.exists() else -2)
    if older.exists():
        os.rename(older, train / "slot_moved")
        os.rename(spare, older)
        os.rename(train / "slot_moved", spare)
    else:
        os.rename(spare, older)
    status.log(f"checkpoint of step {step} copied to {older}", saved_step=step)


def _local_checkpoints(out, run_id: str) -> dict:
    """{step: directory} of the checkpoints SpecForge has finished writing (training_state.pt is written last)."""
    found = {}
    for item in Path(out).iterdir():
        match = re.fullmatch(re.escape(run_id) + r"-step(\d+)", item.name)
        if match and (item / STATE).is_file():
            found[int(match.group(1))] = item
    return found


def _watch(out, run_id, train, copied: int, last_step: int, since: float, stop, status) -> None:
    """Copy the newest checkpoint this run wrote to a slot, until told to stop. The final step is left to `final`."""
    while True:
        stopping = stop.wait(20)
        try:
            fresh = {step: path for step, path in _local_checkpoints(out, run_id).items()
                     if copied < step < last_step and (path / STATE).stat().st_mtime >= since}
            if fresh:
                _to_slot(train, fresh[max(fresh)], max(fresh), status)
                copied = max(fresh)
                settings.COMMIT()
        except Exception as error:  # e.g. SpecForge rotated the directory away mid-copy; training must go on
            status.log(f"checkpoint copy failed, the next checkpoint will be tried: {error!r}")
        if stopping:
            return


def _local_target(root, s) -> Path:
    """Copy ROOT/target to local disk with fixed file times. SpecForge stores each weight file's size and time in the
    checkpoint and refuses to resume when they differ, and a volume need not keep times across containers."""
    target = Path(s.target_dir)
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted((root / "target").iterdir()):
        if item.is_file() and not item.name.endswith(".tmp"):
            shutil.copyfile(item, target / item.name)
            os.utime(target / item.name, ns=(FIXED_MTIME_NS, FIXED_MTIME_NS))
    return target


def _run_config(root, s, steps: int, target) -> dict:
    return {
        "model": {
            "target_model_path": str(target),
            "draft_model_config": str(root / "init" / "drafter_config.json"),
            "embedding_key": s.embedding_key,
            "lm_head_key": s.lm_head_key,
            "torch_dtype": "bfloat16",
            "mask_token_id": s.mask_token_id,
            "use_liger_kernel": False,
        },
        "data": {
            "hidden_states_path": s.features_dir,
            "max_length": s.max_length,
            "cache_dir": str(Path(s.train_dir) / "cache"),
        },
        "training": {
            "strategy": "dspine",
            "num_epochs": s.num_epochs,
            "max_steps": steps,
            "batch_size": s.batch_size,
            "accumulation_steps": s.accumulation_steps,
            "learning_rate": s.learning_rate,
            "lr_scheduler": "cosine",
            "warmup_ratio": s.warmup_ratio,
            "max_grad_norm": s.max_grad_norm,
            "attention_backend": s.attention_backend,
            "num_anchors": s.num_anchors,
            "loss_decay_gamma": s.loss_decay_gamma,
            "objective_chunk_blocks": s.objective_chunk_blocks,
            "log_interval": s.log_interval,
            "save_interval": s.save_interval,
            "max_checkpoints": s.local_checkpoints,
            "seed": s.train_seed,
        },
        "tracking": {"report_to": "none"},
        "deployment": {"mode": "local_colocated", "trainer": {"nnodes": 1, "nproc_per_node": 1}},
        "run_id": s.run_id,
        "output_dir": s.train_dir,
    }


def _train(root, s, status, train, steps: int) -> None:
    import yaml

    out = Path(s.train_dir)
    out.mkdir(parents=True, exist_ok=True)
    _recover(train)
    config = train / "run.yaml"
    partial = train / "run.yaml.tmp"
    partial.write_text(yaml.safe_dump(_run_config(root, s, steps, _local_target(root, s)), sort_keys=False))
    os.replace(partial, config)

    whole = [(record["step"], slot, record) for slot in _slots(train) if (record := _whole(slot))]
    if whole:
        step, slot, record = max(whole, key=lambda item: item[0])
        resume = out / record["name"]
        resume.mkdir(parents=True, exist_ok=True)
        for name in record["files"]:
            shutil.copyfile(slot / name, resume / name)
        start = f"training.resume_from={resume}"
        status.log(f"resuming from step {step} ({slot})", saved_step=step)
    else:
        step = 0
        start = f"model.draft_checkpoint_path={root / 'init'}"
    status.set(step=step, steps=steps, saved_step=step)

    def on_line(line: str) -> None:
        match = re.match(r"step (\d+):", line)
        if match:
            status.set(step=int(match.group(1)))

    stop = threading.Event()
    watcher = threading.Thread(target=_watch, args=(out, s.run_id, train, step, steps, time.time(), stop, status), daemon=True)
    watcher.start()
    try:
        code = common.run_stream([sys.executable, "-m", "specforge.cli", "train", "--config", str(config), start], status, on_line)
    finally:
        stop.set()
        watcher.join()
    last = out / f"{s.run_id}-step{steps}"
    if code != 0 or not (last / STATE).is_file():
        raise RuntimeError(f"specforge train exited with {code}; the checkpoint of the final step {steps} "
                           f"{'exists' if (last / STATE).is_file() else 'is missing'}")
    _copy_checkpoint(last, train / "final.tmp", steps)
    os.rename(train / "final.tmp", train / "final")
    settings.COMMIT()
    status.log(f"final checkpoint (step {steps}) is in {train / 'final'}", saved_step=steps)


def run(root, s, status):
    features = Path(s.features_dir)
    marker = features / "done.json"
    if not marker.exists() or common.read_json(marker)["root"] != str(root):
        raise RuntimeError(f"{features} has no finished features for {root}; run --stage features in this container first")
    samples = sum(name.endswith(".ckpt") for name in os.listdir(features))
    # The trainer drops the last partial batch of each epoch and refuses a plan that ends inside an accumulation
    # window unless max_steps stops it at a whole one, so max_steps is always the whole count.
    steps = (samples // s.batch_size) * s.num_epochs // s.accumulation_steps
    if s.step_cap:
        steps = min(steps, s.step_cap)
    if steps < 1:
        raise RuntimeError(f"{samples} samples give no optimizer step")
    train = root / "train"
    train.mkdir(parents=True, exist_ok=True)
    if not _whole(train / "final"):
        _train(root, s, status, train, steps)

    export = root / "export"
    if not (export / "config.json").exists():
        staging = root / f"export.tmp-{int(time.time())}"
        code = common.run_stream([sys.executable, "-m", "specforge.cli", "export", "--to", "hf",
                                  "--checkpoint", str(train / "final"),
                                  "--draft-config", str(root / "init" / "drafter_config.json"),
                                  "--output-dir", str(staging)], status)
        if code != 0:
            raise RuntimeError(f"specforge export exited with {code}")
        os.rename(staging, export)
    return {"steps": steps, "samples": samples}
