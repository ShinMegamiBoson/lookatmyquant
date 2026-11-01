"""The compiler's named jobs. Each one is a whole script that writes its own result file in results/batch/.
modal_app.py runs them on GPUs, one job per GPU; the same command works on any machine that has the model.

    python run_batch.py --list
    OPGRAPH_ARCH=smollm2-360m python run_batch.py --job ccSensSmoke

Jobs:
    ccSens*     every SmolLM2-360M item alone at every option (compiler.py measure)
    ccQSens*    the same for Qwen3.5-4B: every matrix at tr3, spot layers at every option
    ccM<model>* any other model on the text mix: all matrices by noise, spot layers at every option, tables and cache
    cc_<plan>   one plan in results/compiler/plans/, built and scored (quantize.py --program plan)
    ccG_<plan>  the same plan with the exact gauge search, ccS4_<plan> with four Hadamard seeds per matrix
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = {}


def job(name, **spec):
    spec.setdefault("arch", "qwen3-1.7b")  # what opgraph.model reads at import; the scripts take their model from --model
    spec.setdefault("gpu", "large")
    JOBS[name] = spec


_q35p = ["--model-path", "/root/.cache/huggingface/models/Qwen3.5-4B"]
_sm = ["--model", "HuggingFaceTB/SmolLM2-360M"]

# SmolLM2-360M: matrices in eight slices of four layers; embedding buckets and cache widths together
for _a in range(0, 32, 4):
    job(f"ccSensL{_a}", arch="smollm2-360m", gpu="small",
        script=["compiler.py", "measure", "--part", "linear", "--layers", f"{_a}-{_a + 3}", "--out", f"results/batch/ccSensL{_a}.json"])
job("ccSensEmbKv", arch="smollm2-360m", gpu="small",
    script=["compiler.py", "measure", "--part", "embed,kv", "--out", "results/batch/ccSensEmbKv.json"])
job("ccSensSmoke", arch="smollm2-360m", gpu="small",
    script=["compiler.py", "measure", "--part", "linear,embed,kv", "--layers", "0-0", "--embed-buckets", "0-1", "--kv-layers", "0-1",
            "--out", "results/batch/ccSensSmoke.json"])
# SmolLM2-360M and Qwen3.5-4B plans
for _p in sorted((ROOT / "results" / "compiler" / "plans").glob("*.json")):
    if not _p.stem.startswith(("sm_", "q35_")):
        continue
    _plan = json.loads(_p.read_text())
    _q = "Qwen3.5" in (_plan.get("model") or "")
    _arch, _gpu, _m = ("qwen3-1.7b", "large", _q35p) if _q else ("smollm2-360m", "small", _sm)
    _base = ["quantize.py", "--program", "plan", *_m, "--plan", f"results/compiler/plans/{_p.name}", "--ctx", "512,2048" if "kv" in _plan else "512"]
    job(f"cc_{_p.stem}", arch=_arch, gpu=_gpu, script=[*_base, "--out", f"results/batch/cc_{_p.stem}.json"])
    job(f"ccG_{_p.stem}", arch=_arch, gpu=_gpu, script=[*_base, "--gauges", "vo,ud", "--gauge-alphas", "0,0.1", "--out", f"results/batch/ccG_{_p.stem}.json"])
    job(f"ccS4_{_p.stem}", arch=_arch, gpu=_gpu, script=[*_base, "--seeds", "4", "--out", f"results/batch/ccS4_{_p.stem}.json"])
# Qwen3.5-4B: every matrix at tr3 only (the anchor), every option on four spot layers (DeltaNet 0 and 14, attention 15
# and 31), the embedding's six small buckets at every option and its five big ones at tr2 and tr3, and the cache
_qm = ["compiler.py", "measure", "--model", "Qwen/Qwen3.5-4B", "--model-path", "/root/.cache/huggingface/models/Qwen3.5-4B", "--kl-batch", "4"]
_qopt = "tr2,tr3,tr4,lm5,lm6,lm8"
for _n, _l in (("A0", "1-7"), ("A1", "8-13,16"), ("A2", "17-23"), ("A3", "24-30")):
    job(f"ccQSens{_n}", script=[*_qm, "--part", "linear", "--layers", _l, "--options", "tr3", "--out", f"results/batch/ccQSens{_n}.json"])
for _l in (0, 14, 15, 31):
    job(f"ccQSensF{_l}", script=[*_qm, "--part", "linear", "--layers", str(_l), "--options", _qopt, "--out", f"results/batch/ccQSensF{_l}.json"])
job("ccQSensE0", script=[*_qm, "--part", "embed,kv", "--embed-buckets", "0-5", "--embed-options", _qopt, "--out", "results/batch/ccQSensE0.json"])
for _n, _o in (("E1", "tr3"), ("E2", "tr2")):
    job(f"ccQSens{_n}", script=[*_qm, "--part", "embed", "--embed-buckets", "6-10", "--embed-options", _o, "--out", f"results/batch/ccQSens{_n}.json"])

# The noise method against the full tr3 measurements, and the discovered-stage harness against the SmolLM2 baseline
job("ccVNoiseSm", arch="smollm2-360m", gpu="small", script=["compiler.py", "measure", "--part", "linear,embed", "--method", "noise", "--out", "results/batch/ccVNoiseSm.json"])
job("ccVNoiseQ35", script=[*_qm, "--part", "linear,embed", "--method", "noise", "--out", "results/batch/ccVNoiseQ35.json"])
job("ccVReg_sm_ref", arch="smollm2-360m", gpu="small",
    script=["quantize.py", "--program", "plan", *_sm, "--plan", "results/compiler/plans/sm_ref.json", "--ctx", "512,2048", "--out", "results/batch/ccVReg_sm_ref.json"])

# Any other model on the text mix: all matrices by noise, three or four spot layers at every option (they calibrate noise
# to each code), the embedding and head buckets and the cache
_CC = {  # tag: (repo, local path or "", GPU for measurement, GPU for runs, dtype, spot layers, MoE)
    "L32": ("unsloth/Llama-3.2-3B", "", "large", "large", "float32", "0,14,27", False),
    "L31": ("unsloth/Meta-Llama-3.1-8B", "", "large", "xlarge", "float32", "0,16,31", False),
    "G3": ("unsloth/gemma-3-4b-pt", "", "large", "large", "float32", "0,17,33", False),
    "Phi": ("microsoft/Phi-4-mini-instruct", "", "large", "large", "float32", "0,16,31", False),
    "Q35": ("Qwen/Qwen3.5-4B", "/root/.cache/huggingface/models/Qwen3.5-4B", "large", "large", "float32", "0,14,15,31", False),
    "Olmoe": ("allenai/OLMoE-1B-7B-0924", "", "large", "large", "float32", "0,8,15", True),
    "Q35A3": ("Qwen/Qwen3.5-35B-A3B", "", "b200", "b200", "bfloat16", "0,19,20,39", True),
    "Oss": ("openai/gpt-oss-20b", "", "xlarge", "xlarge", "bfloat16", "0,12,23", True),
    # Gemma 4, the family compared against public quantizations: spot layers cover sliding, full-attention and KV-shared layers
    "G4E2": ("google/gemma-4-E2B-it", "", "large", "large", "float32", "0,4,20,34", False),
    "G4E4": ("google/gemma-4-E4B-it", "", "large", "xlarge", "float32", "0,5,30,41", False),
    "G4_12": ("google/gemma-4-12B-it", "", "large", "xlarge", "bfloat16", "0,23,47", False),
    "G4_26": ("google/gemma-4-26B-A4B-it", "", "xlarge", "b200", "bfloat16", "0,11,29", True),
    "G4_31": ("google/gemma-4-31B-it", "", "xlarge", "b200", "bfloat16", "0,29,59", False),
}
for _t, (_repo, _path, _gm, _gr, _dt, _spot, _moe) in _CC.items():
    _m = ["compiler.py", "measure", "--model", _repo, *(["--model-path", _path] if _path else []), "--data", "mix", "--dtype", _dt, "--kl-batch", "4"]
    _lin = "linear,experts" if _moe else "linear"
    job(f"ccM{_t}Noise", gpu=_gm, script=[*_m, "--part", _lin, "--method", "noise", "--out", f"results/batch/ccM{_t}Noise.json"])
    job(f"ccM{_t}Spot", gpu=_gm, script=[*_m, "--part", "linear", "--layers", _spot, "--options", "tr2,tr3,tr4,lm5,lm6,lm8", "--out", f"results/batch/ccM{_t}Spot.json"])
    job(f"ccM{_t}EmbKv", gpu=_gm, script=[*_m, "--part", "embed,kv", "--embed-options", "tr2,tr3,lm5,lm8", "--out", f"results/batch/ccM{_t}EmbKv.json"])
    if _moe:  # experts at every option on the spot layers (fewer options: an expert bucket is many matrices)
        job(f"ccM{_t}SpotX", gpu=_gm, script=[*_m, "--part", "experts", "--layers", _spot, "--options", "tr2,tr3,lm5,lm6", "--out", f"results/batch/ccM{_t}SpotX.json"])
# Lloyd–Max at 2–6 bits for Gemma 4 E2B's embedding and per-layer tables: the options a Lloyd–Max-only plan (the menu the
# Apple Silicon runtime can run) needs for the tables
job("ccMG4E2EmbLM", gpu="large", script=["compiler.py", "measure", "--model", "google/gemma-4-E2B-it", "--data", "mix", "--dtype", "float32", "--kl-batch", "4",
            "--part", "embed", "--embed-options", "lm2,lm3,lm4,lm6", "--out", "results/batch/ccMG4E2EmbLM.json"])
# coverage plans (results/compiler/plans/x_*.json) run on their model's GPU and dtype, calibrated on and evaluated with the
# text mix (and WikiText at 512 and 2048)
for _p in sorted((ROOT / "results" / "compiler" / "plans").glob("x_*.json")):
    _plan = json.loads(_p.read_text())
    _cc = next(v for v in _CC.values() if v[0] == _plan["model"])
    job(f"cc_{_p.stem}", gpu=_cc[3], script=["quantize.py", "--program", "plan", "--model", _cc[0], *(["--model-path", _cc[1]] if _cc[1] else []),
                "--dtype", _cc[4], "--data", "mix", "--plan", f"results/compiler/plans/{_p.name}", "--ctx", "512,2048",
                "--save", f"/root/.cache/huggingface/compiler_plans/{_p.stem}.pt",  # benchmarked later by public_eval.py, with the public ones
                *(["--export", f"/root/.cache/huggingface/lamq/{_p.stem}"] if "_lm" in _p.stem else []),  # Lloyd–Max plans: the Apple Silicon runtime's files
                *(["--rot", "in-shared"] if "_lm1s_" in _p.stem else []),  # one input rotation per stage, none on the output: half the rotations at decode
                "--out", f"results/batch/cc_{_p.stem}.json"])


def run(name):
    spec = JOBS[name]
    if os.environ.get("OPGRAPH_ARCH", "smollm2-360m") != spec["arch"]:
        raise SystemExit(f"{name} needs OPGRAPH_ARCH={spec['arch']}")
    import torch
    torch.set_num_threads(int(os.environ.get("OPGRAPH_THREADS", "8")))  # the cores the job was given; os.cpu_count() reports the whole host in a container
    sys.argv = list(spec["script"])
    runpy.run_path(spec["script"][0], run_name="__main__")
    return json.loads((ROOT / spec["script"][spec["script"].index("--out") + 1]).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list or not args.job:
        for n, s in JOBS.items():
            print(f"{n:28s} {s['gpu']:8s} {' '.join(s['script'])}")
        return
    run(args.job)


if __name__ == "__main__":
    main()
