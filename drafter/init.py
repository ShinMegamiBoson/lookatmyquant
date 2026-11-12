"""Stage `init`: what SpecForge's trainer reads besides the features.

ROOT/target/   the original's config and tokenizer files, and model.safetensors holding the BUILD's embedding matrix under
               both names the trainer may open (the embedding key and lm_head.weight; equal tensors, so it ties them).
ROOT/init/     drafter_config.json (the DFlash config turned into a DSpineDraftModel config) and training_state.pt
               ({"draft_state_dict", "strategy": "dspine"}): the DFlash weights, DSpine's additions at their start
               values, and the transfer space computed from the build's output head. This is the form
               model.draft_checkpoint_path takes; the directory must never hold a model.safetensors."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch

from . import common

WEIGHT_FILES = (".safetensors", ".index.json", ".bin", ".pt", ".pth", ".gguf", ".h5", ".msgpack")
DSPINE_BUFFERS = ("transfer_codes", "transfer_ready")


def _target(root, s, status):
    """Write ROOT/target once and return the build's embedding matrix as the trainer will read it."""
    from safetensors.torch import load_file, save_file

    target = root / "target"
    weights = target / "model.safetensors"
    if not weights.exists():
        model = common.load_model(s, "build", status, device="cpu")
        weight = model.get_input_embeddings().weight.detach().clone()
        del model
        common.release()
        target.mkdir(parents=True, exist_ok=True)
        for item in sorted(Path(s.original_dir).iterdir()):
            if item.is_file() and not item.name.endswith(WEIGHT_FILES):
                shutil.copyfile(item, target / item.name)
        partial = target / "model.safetensors.tmp"
        save_file({s.embedding_key: weight, s.lm_head_key: weight.clone()}, str(partial))
        os.replace(partial, weights)
        status.log(f"wrote {weights}: embedding {tuple(weight.shape)} {weight.dtype}")
    return load_file(str(weights))[s.embedding_key]


def _drafter_config(base: dict, s, vocab: int, hidden: int) -> dict:
    """The DSpine drafter config: the downloaded DFlash config with the architecture and DSpine's settings added. Every
    field the settings state must be what the download says, because the DFlash weights only fit that."""
    method = dict(base.get("dflash_config") or {})
    found_expected = {
        "num_hidden_layers": (base.get("num_hidden_layers"), len(s.layer_types)),
        "layer_types": (list(base.get("layer_types") or ()), list(s.layer_types)),
        "use_sliding_window": (base.get("use_sliding_window"), True),
        "sliding_window": (base.get("sliding_window"), s.sliding_window),
        "intermediate_size": (base.get("intermediate_size"), s.intermediate_size),
        "vocab_size": (base.get("vocab_size", vocab), vocab),
        "hidden_size": (base.get("hidden_size", hidden), hidden),
        "block_size": (method.get("block_size", base.get("block_size")), s.block_size),
        "mask_token_id": (method.get("mask_token_id"), s.mask_token_id),
        "target_layer_ids": (list(method.get("target_layer_ids") or ()), list(s.target_layer_ids)),
    }
    wrong = {name: pair for name, pair in found_expected.items() if pair[0] != pair[1]}
    if wrong:
        raise RuntimeError(f"the DFlash config is not what the settings expect (found, expected): {wrong}")
    config = {name: value for name, value in base.items() if name != "transformers_version"}
    config.update(
        architectures=["DSpineDraftModel"],
        auto_map={"AutoModel": "dspine.DSpineDraftModel"},
        is_causal=True,  # DSpine's blocks are causal in every layer; the DFlash drafter's last layer was not
        tie_word_embeddings=False,
        vocab_size=vocab,
        hidden_size=hidden,
        block_size=s.block_size,
        dflash_config={**method, "block_size": s.block_size, "mask_token_id": s.mask_token_id,
                       "target_layer_ids": list(s.target_layer_ids)},
        dspine_config=dict(s.dspine),
    )
    return config


def run(root, s, status):
    from specforge.modeling.draft.dspine import DSpineDraftModel
    from specforge.modeling.target.checkpoint import load_checkpoint_tensors
    from specforge.training.model_loading import load_draft_config_source

    head = _target(root, s, status)
    config_path = root / "init" / "drafter_config.json"
    common.write_json(config_path, _drafter_config(common.read_json(root / "assets" / "dflash" / "config.json"), s, *head.shape))

    model = DSpineDraftModel(load_draft_config_source(str(config_path))).to(torch.bfloat16)
    state = load_checkpoint_tensors(str(root / "assets" / "dflash"))
    result = model.load_state_dict(state, strict=False)
    added = sorted(key for key in model.state_dict() if key.startswith("injection.") or key in DSPINE_BUFFERS)
    status.log(f"DFlash tensors: {len(state)}; not used by the DSpine drafter: {sorted(result.unexpected_keys)}")
    status.log(f"not in the DFlash file ({len(result.missing_keys)}): {sorted(result.missing_keys)}")
    if result.unexpected_keys or sorted(result.missing_keys) != added:
        raise RuntimeError(f"every DFlash weight must be used and only DSpine's additions may be new: {added}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).initialize_transfer_space(head.to(device))
    if not bool(model.transfer_ready):
        raise RuntimeError("the transfer space was not initialised")
    partial = root / "init" / "training_state.pt.tmp"
    torch.save({"draft_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "strategy": "dspine"}, partial)
    os.replace(partial, root / "init" / "training_state.pt")
    parameters = sum(p.numel() for p in model.parameters())
    status.log(f"wrote {root / 'init'}: {parameters / 1e6:.0f}M parameters, {len(added)} tensors new to DSpine")
    return {"parameters": parameters, "dflash_tensors": len(state), "dspine_tensors": len(added)}
