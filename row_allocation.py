"""A build with bits allocated to rows, in four commands. Every build is quantize.py with the LookAtMyQuant 2 recipe
(trellis codes along rows, calibration on worked math solutions with digit tokens weighted, QEP at α = 0.5); the two builds
differ only in which rows get which bit rate.

    build-control   B0: in each 2- and 3-bit matrix the most important eighth of the rows one bit up and the least
                    important eighth one down (row_importance.py), then packed (compiler_artifact.py)
    prepare-rows    the row plan: 1 to 4 bits per row group across every matrix, the least summed importance-weighted
                    distortion (4^-bits) inside B0's packed bytes, header and assets included (rowgroup_alloc.py)
    build-rows      R1: the same build under that row plan, packed; it must come to the plan's bytes exactly
    pack            pack any saved codes

    python row_allocation.py build-control --source-dir . --trial-dir runs/rows
    python row_allocation.py prepare-rows  --source-dir . --trial-dir runs/rows
    python row_allocation.py build-rows    --source-dir . --trial-dir runs/rows

The row plan this made for Qwen3.5-4B at 1.2 GB is results/compiler/plans/q35_rowplan_1p2GB.json.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gc
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


MATH_CACHE = "/root/.cache/huggingface/retrain/math_long_q35.pt"
IMPORTANCE = "/root/.cache/huggingface/retrain/row_importance.pt"
MODEL_PATH = "/root/.cache/huggingface/models/Qwen3.5-4B"
PLAN_RELATIVE = "results/compiler/plans/q35_trellis_1p2GB.json"
DISTORTION = {bits: 4.0 ** -bits for bits in (1, 2, 3, 4)}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_original_model(model_path, *, dtype="bfloat16", device="cpu"):
    """Load the untouched source model; dtype/device must be explicit for evaluation."""
    import torch
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=getattr(torch, dtype)).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def pack_codes(codes_path_or_payload, model_path, out_dir, *, original_model=None,
               expected_bytes=None, ceiling_bytes=None):
    """Pack saved code entries plus complete unchanged original state and assets.

    The default source load is FP32 on CPU, matching the quantizer's original
    dense state and storage aliases. Pass the same FP32 original model to reuse
    it across arms; narrowing dense values would change the selected model.
    This is for calibration-only builds; it must not be used to discard trained
    changes to uncoded parameters. Size mismatches leave a reviewable receipt.
    """
    import torch
    from compiler_artifact import export_saved_codes
    path = Path(codes_path_or_payload) if isinstance(codes_path_or_payload, (str, Path)) else None
    saved = torch.load(path, map_location="cpu", weights_only=False) if path else codes_path_or_payload
    created = original_model is None
    model = load_original_model(model_path, dtype="float32", device="cpu") if original_model is None else original_model
    started = time.perf_counter()
    try:
        receipt = export_saved_codes(saved, out_dir, dense_tensors=model.state_dict(), model_dir=model_path)
    finally:
        if created:
            del model
            gc.collect()
    receipt.update(codes_path=str(path) if path else None,
                   codes_sha256=file_sha256(path) if path else None,
                   source_model_path=str(model_path), seconds=time.perf_counter() - started,
                   expected_bytes=expected_bytes, ceiling_bytes=ceiling_bytes,
                   expected_size_matches=expected_bytes is None or receipt["total_bytes"] == expected_bytes,
                   within_ceiling=ceiling_bytes is None or receipt["total_bytes"] <= ceiling_bytes)
    receipt_path = Path(out_dir).parent / (Path(out_dir).name + ".receipt.json")
    _write_json(receipt_path, receipt)
    if not receipt["expected_size_matches"] or not receipt["within_ceiling"]:
        raise RuntimeError(f"packed bytes violate the frozen contract; see {receipt_path}")
    return receipt


def read_control_manifest(control_artifact):
    """Read and check exact file accounting without allocating the weight blob."""
    from compiler_artifact import FORMAT, artifact_serialized_bytes
    root = Path(control_artifact)
    path = root / "artifact.json"
    manifest = json.loads(path.read_text())
    if manifest.get("format") != FORMAT or manifest.get("blob") != "weights.bin":
        raise ValueError("unsupported control artifact")
    if (root / "weights.bin").stat().st_size != manifest["blob_bytes"]:
        raise ValueError("control blob size does not match its manifest")
    declared = {"artifact.json", "weights.bin"} | set(manifest.get("assets", {}))
    if {p.name for p in root.iterdir()} != declared:
        raise ValueError("control directory contains undeclared files")
    asset_bytes = 0
    for name, specification in manifest.get("assets", {}).items():
        asset = root / name
        if Path(name).name != name or asset.stat().st_size != specification["bytes"] or file_sha256(asset) != specification["sha256"]:
            raise ValueError(f"control asset mismatch: {name}")
        asset_bytes += specification["bytes"]
    specs = {name: {key: spec[key] for key in ("dtype", "shape", "nbytes")}
             for name, spec in manifest["tensors"].items()}
    semantic = {key: value for key, value in manifest.items() if key not in ("format", "blob", "blob_bytes", "tensors")}
    total = (root / "weights.bin").stat().st_size + path.stat().st_size + asset_bytes
    if artifact_serialized_bytes(specs, semantic) + asset_bytes != total:
        raise ValueError("control header is not the serializer's canonical representation")
    return manifest, specs, semantic, asset_bytes, total


def row_plan_problem(control_artifact, importance, base_plan):
    """Build candidate groups and an exact full-artifact size callback from B0."""
    from compiler_artifact import artifact_serialized_bytes, grouped_matrix_specs
    from rowgroup_alloc import make_row_groups
    manifest, all_specs, semantic, asset_bytes, ceiling = read_control_manifest(control_artifact)
    matrices = {}
    for short_name, program in sorted(base_plan["linears"].items()):
        if program not in ("tr2x", "tr3x", "tr2", "tr3"):
            continue
        names = [name for name in semantic["matrices"] if name.endswith("." + short_name + ".weight")]
        if len(names) != 1:
            raise ValueError(f"control must contain exactly one coded matrix for {short_name}")
        name = names[0]
        metadata = semantic["matrices"][name]
        if metadata["kind"] != "grouped" or any(g["kind"] != "trellis" or g.get("state_bits") != 12 or g.get("start_state") != 0 for g in metadata["groups"]):
            raise ValueError(f"{name}: not a fixed-start row-trellis control")
        scores = importance[short_name]
        scores = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        if len(scores) != metadata["shape"][0]:
            raise ValueError(f"importance row count differs for {name}")
        matrices[name] = {"importance": scores, "columns": metadata["shape"][1], "base_bits": int(program[2])}
    if not matrices:
        raise ValueError("control has no eligible row-allocation matrices")
    groups = make_row_groups(matrices, DISTORTION)
    removed = set()
    for name in matrices:
        metadata = semantic["matrices"][name]
        removed.add(metadata["signs"])
        for group in metadata["groups"]:
            removed.update(group["key"] + suffix for suffix in (".rows", ".codes", ".scales", ".table"))
    if not removed.issubset(all_specs):
        raise ValueError("control matrix references an absent tensor")
    fixed_specs = {name: spec for name, spec in all_specs.items() if name not in removed}
    fixed_bytes = sum(spec["nbytes"] for spec in fixed_specs.values()) + asset_bytes
    fixed_bytes += sum(matrix["columns"] for matrix in matrices.values())

    def layout(selected):
        by_matrix = defaultdict(list)
        for group in selected:
            by_matrix[group["matrix"]].append({"rows": len(group["rows"]), "bits": group["bits"]})
        if set(by_matrix) != set(matrices):
            raise ValueError("candidate matrix set differs from the frozen allocation problem")
        specs, candidate_semantic = dict(fixed_specs), copy.deepcopy(semantic)
        for name, descriptors in sorted(by_matrix.items()):
            shape = semantic["matrices"][name]["shape"]
            new_specs, new_meta = grouped_matrix_specs(name, shape, descriptors)
            specs.update(new_specs)
            candidate_semantic["matrices"][name] = new_meta
        return specs, candidate_semantic

    def size(selected):
        return artifact_serialized_bytes(*layout(selected)) + asset_bytes

    return {"groups": groups, "matrices": matrices, "fixed_bytes": fixed_bytes, "ceiling_bytes": ceiling,
            "artifact_size": size, "layout": layout, "control_manifest": manifest,
            "asset_bytes": asset_bytes, "fixed_tensor_bytes": sum(s["nbytes"] for s in fixed_specs.values())}


def prepare_row_plan(control_artifact, importance_path, base_plan_path, out_path):
    import torch
    from rowgroup_alloc import allocate
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(out_path)
    importance = torch.load(importance_path, map_location="cpu", weights_only=False)
    base_plan = json.loads(Path(base_plan_path).read_text())
    problem = row_plan_problem(control_artifact, importance, base_plan)
    result = allocate(problem["groups"], problem["ceiling_bytes"], fixed_bytes=problem["fixed_bytes"],
                      artifact_size=problem["artifact_size"])
    plan = result.as_dict()
    plan.update(control_artifact=str(control_artifact),
                control_manifest_sha256=file_sha256(Path(control_artifact) / "artifact.json"),
                control_total_bytes=problem["ceiling_bytes"], importance_sha256=file_sha256(importance_path),
                base_plan_sha256=file_sha256(base_plan_path), distortion_by_bits=DISTORTION,
                distortion_provenance="Preregistered relative 4^-bits proxy; not measured model accuracy")
    _write_json(out_path, plan)
    receipt = {key: value for key, value in plan.items() if key != "groups"}
    receipt.update(path=str(out_path), sha256=file_sha256(out_path), matrices=len(problem["matrices"]),
                   groups=len(result.selected), asset_bytes=problem["asset_bytes"],
                   fixed_tensor_bytes=problem["fixed_tensor_bytes"],
                   aliases_preserved=problem["control_manifest"].get("aliases", {}))
    _write_json(out_path.with_suffix(".receipt.json"), receipt)
    return receipt


def quantizer_command(arm, source_dir, model_path, trial_dir, base_plan_path, importance_path=IMPORTANCE):
    """The two commands differ only in allocation and optional B0 target capture."""
    if arm not in ("B0", "R1"):
        raise ValueError("arm must be B0 or R1")
    source_dir, trial_dir = Path(source_dir), Path(trial_dir)
    output = trial_dir / arm
    command = [sys.executable, str(source_dir / "quantize.py"),
               "--program", "plan", "--plan", str(base_plan_path), "--model-path", str(model_path),
               "--dtype", "float32", "--ctx", "512", "--skip-eval", "--rot", "in-shared", "--trellis-rows", "16,8", "--seeds", "1", "--seed-set", "1",
               "--data", "mix", "--calib", "64", "--calib-len", "2048", "--calib-math", "1.0", "--math-cache", MATH_CACHE,
               "--token-weights", "digits", "--token-floor", "0.1", "--cross-alpha", "0.5",
               "--save", str(output / "dense.pt"), "--save-codes", str(output / "codes.pt"),
               "--out", str(output / "result.json")]
    if arm == "B0":
        command += ["--row-split", str(importance_path) + ":0.125", "--capture-targets", str(output / "targets.pt")]
    else:
        command += ["--row-plan", str(trial_dir / "row-plan.json")]
    return command


def build_arm(arm, *, source_dir, model_path, trial_dir, base_plan_path, importance_path=IMPORTANCE, timeout_seconds=None):
    trial_dir, source_dir = Path(trial_dir), Path(source_dir)
    output = trial_dir / arm
    if output.exists():
        raise FileExistsError(output)
    row_plan = None
    if arm == "R1":
        row_plan = json.loads((trial_dir / "row-plan.json").read_text())
        control = trial_dir / "B0" / "packed"
        if row_plan["control_manifest_sha256"] != file_sha256(control / "artifact.json"):
            raise ValueError("row plan was prepared from a different control")
        if row_plan["base_plan_sha256"] != file_sha256(base_plan_path) or row_plan["importance_sha256"] != file_sha256(importance_path):
            raise ValueError("base plan or importance changed after row allocation")
    output.mkdir(parents=True)
    command = quantizer_command(arm, source_dir, model_path, trial_dir, base_plan_path, importance_path)
    _write_json(output / "invocation.json", {"command": command, "source_sha256": file_sha256(source_dir / "quantize.py")})
    started = time.perf_counter()
    subprocess.run(command, cwd=source_dir, check=True, timeout=timeout_seconds)
    receipt = pack_codes(output / "codes.pt", model_path, output / "packed",
                         expected_bytes=row_plan["total_bytes"] if row_plan else None,
                         ceiling_bytes=row_plan["control_total_bytes"] if row_plan else None)
    receipt.update(arm=arm, build_seconds=time.perf_counter() - started,
                   dense_sha256=file_sha256(output / "dense.pt"), result_sha256=file_sha256(output / "result.json"))
    _write_json(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-control", "prepare-rows", "build-rows", "pack"))
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--trial-dir", required=True)
    parser.add_argument("--base-plan", default="")
    parser.add_argument("--importance", default=IMPORTANCE)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--codes", default="", help="pack only")
    parser.add_argument("--output", default="", help="pack only")
    args = parser.parse_args()
    source = Path(args.source_dir).resolve()
    sys.path.insert(0, str(source))
    trial = Path(args.trial_dir)
    base_plan = Path(args.base_plan) if args.base_plan else source / PLAN_RELATIVE
    if args.command == "prepare-rows":
        receipt = prepare_row_plan(trial / "B0" / "packed", args.importance, base_plan, trial / "row-plan.json")
    elif args.command == "pack":
        if not args.codes or not args.output:
            parser.error("pack requires --codes and --output")
        receipt = pack_codes(args.codes, args.model_path, args.output)
    else:
        receipt = build_arm("B0" if args.command == "build-control" else "R1", source_dir=source,
                            model_path=args.model_path, trial_dir=trial, base_plan_path=base_plan,
                            importance_path=args.importance, timeout_seconds=args.timeout_seconds)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
