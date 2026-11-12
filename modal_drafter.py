"""A DSpine drafter for the 1.23 GB Qwen3.5-4B build, trained from the published DFlash drafter, as one Modal job that
carries on by itself: the stages (training text, features, training, scoring; drafter/pipeline.py) each keep their
progress on the volume, the function retries when its container is preempted or fails, and a retry continues from what
is there. Nothing depends on this Mac after the launch.

    modal run modal_drafter.py::main                   # every stage at toy size, attached, on an L40S: proves the code
    modal deploy modal_drafter.py                      # then:
    modal run modal_drafter.py::launch                 # starts the real run on the deployed app and returns at once
    modal volume get opgraph-hf drafter/q35_SQYFR_dspine_v1/status.json -      # where it is

Results on the volume opgraph-hf under drafter/<run>/: scores_baseline.json (the stock DFlash drafter against the
original and against the build), scores.json (the trained drafter beside them), export/ (the drafter), status.json.
Nothing on the volume is removed.
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
HF = "/root/.cache/huggingface"
RUN = f"{HF}/drafter/q35_SQYFR_dspine_v1"
SMOKE = f"{HF}/drafter/smoke_v1"
SPECFORGE = "git+https://github.com/sgl-project/SpecForge@2dadea10e317842af21c79e2097dc05abe1e2b5a"  # PR 1038: DSpine's corrected objective

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")  # a C compiler: flex_attention compiles, and Triton builds its launcher with it
    # torch and transformers as the rest of this project runs Qwen3.5 (SpecForge's own pins at this commit are older, and it
    # pins sglang, which nothing here imports): its other needs by hand, then SpecForge itself without dependencies
    .pip_install("torch==2.14.0", "transformers==5.16.1", "accelerate", "click>=8.0", "pydantic", "pyyaml", "safetensors", "huggingface_hub",
                 "datasets", "liger-kernel>=0.8.3,<0.9", "tensorboard", "numpy", "tqdm", "psutil", "requests", "typing-extensions")
    .pip_install(SPECFORGE, extra_options="--no-deps")
    .env({"TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "HF_HOME": HF})
    .add_local_dir(ROOT / "drafter", "/root/opgraph/drafter", ignore=["**/__pycache__/**"])
)
hf = modal.Volume.from_name("opgraph-hf")
app = modal.App("lamq-drafter", image=image)


RESTART = SMOKE + "_restart2"


def _longer_toy_training():
    """For the restart test: the toy run with 80 training steps and a checkpoint every 20, so that a checkpoint reaches
    the volume while the trainer is still going. Both halves of the test must train under the same settings."""
    import dataclasses
    import sys
    sys.path.insert(0, "/root/opgraph")
    from drafter import settings
    toy = settings.Settings.smoke
    settings.Settings.smoke = lambda self: dataclasses.replace(toy(self), step_cap=80, save_interval=20)


def _pipeline(root: str, stage: str, smoke: bool):
    import sys
    sys.path.insert(0, "/root/opgraph")
    hf.reload()
    from drafter import pipeline, settings
    settings.COMMIT = hf.commit  # the volume keeps what is committed: after every checkpoint copy and every stage
    import threading

    def keep_committing():  # and every five minutes, so the status file and the text written so far are there to resume from
        import time
        while True:
            time.sleep(300)
            try:
                hf.commit()
            except Exception as e:
                print("periodic commit failed:", repr(e)[:200], flush=True)
    threading.Thread(target=keep_committing, daemon=True).start()
    try:
        pipeline.main(["--root", root, "--stage", stage] + (["--smoke"] if smoke else []))
    finally:
        hf.commit()


@app.function(gpu="B200", cpu=16.0, memory=131072, ephemeral_disk=900 * 1024, timeout=24 * 3600, volumes={HF: hf},
              retries=modal.Retries(max_retries=8, backoff_coefficient=1.0, initial_delay=30.0))
def run(stage: str = "all"):
    _pipeline(RUN, stage, False)


@app.function(gpu="L40S", cpu=8.0, memory=65536, timeout=2 * 3600, volumes={HF: hf})
def smoke(stage: str = "all"):
    _pipeline(SMOKE, stage, True)


@app.function(gpu="L40S", cpu=8.0, memory=65536, timeout=2 * 3600, volumes={HF: hf})
def interrupt_test():
    """The first half of the restart test: a fresh toy run whose trainer is killed as soon as its first checkpoint has
    reached the volume, as a preempted container would leave it."""
    import os
    import signal
    import threading
    import time
    import psutil
    root = RESTART
    _longer_toy_training()

    def kill_when_saved():
        while not any(Path(root, "train", slot, "complete.json").exists() for slot in ("slot_a", "slot_b")):
            time.sleep(1)
        for proc in psutil.process_iter(["cmdline"]):
            if "specforge.cli" in " ".join(proc.info["cmdline"] or []) and "train" in (proc.info["cmdline"] or []):
                print("killing the trainer, pid", proc.pid, flush=True)
                os.kill(proc.pid, signal.SIGKILL)
    threading.Thread(target=kill_when_saved, daemon=True).start()
    try:
        _pipeline(root, "all", True)
        print("NOT INTERRUPTED: the run finished before the kill", flush=True)
    except Exception as e:
        print("interrupted as planned:", repr(e)[:300], flush=True)
    print("left on the volume:", sorted(x.name for x in Path(root, "train").iterdir()), "| done:", sorted(x.name for x in Path(root, "done").iterdir()), flush=True)


@app.function(gpu="L40S", cpu=8.0, memory=49152, timeout=2 * 3600, volumes={HF: hf})
def resume_test():
    """The second half, in another container (so nothing local survives): the same pipeline call must carry on from the
    saved checkpoint and finish."""
    _longer_toy_training()
    _pipeline(RESTART, "all", True)


@app.function(gpu="L40S", cpu=8.0, memory=65536, timeout=3600, volumes={HF: hf})
def reference():
    """The trained drafter's inputs and outputs on a few held-out blocks, for checking a port of it (drafter/reference.py):
    RUN/reference/ref.safetensors. modal run --detach modal_drafter.py::reference"""
    import sys
    sys.path.insert(0, "/root/opgraph")
    hf.reload()
    from drafter import common, settings
    from drafter import reference as ref
    root = Path(RUN)
    (root / "reference").mkdir(exist_ok=True)
    print(ref.dump(root, settings.Settings(), common.Status(root / "reference")), flush=True)
    hf.commit()


@app.local_entrypoint()
def main(stage: str = "all", smoke_run: bool = True):
    """Attached: the toy-size run by default (--no-smoke-run for the real one, which then needs this client to stay)."""
    (smoke if smoke_run else run).remote(stage)


@app.local_entrypoint()
def launch(stage: str = "all"):
    """The real run on the DEPLOYED app (modal deploy modal_drafter.py first): it does not need this client afterwards."""
    call = modal.Function.from_name("lamq-drafter", "run").spawn(stage)
    print("started", call.object_id)
