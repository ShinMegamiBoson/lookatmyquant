"""What a packed Inkling-2-Mini directory may contain, checked before loading.

The runtime refuses a file that uses anything it does not implement - an unknown format id, tensor, lookup table,
manifest format or model option - instead of silently skipping it. When upstream (`runtime_mlx/inkling/`) adds a
format, files using it fail here with a message naming what is new, which is the cue to port it (see SYNC.md).

    python -m runtime_cuda.inkling.schema models/Inkling-2-Mini-LookAtMyQuant-11.2GB
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np

from .formats import EXPERT_FIDS, FMT_PARAMS, NVFP4_FID

LM_FMTS = ("lm6", "lm8")
LINEAR_FMTS = set(FMT_PARAMS) | set(LM_FMTS)
LINEAR_NAMES = {
    "self_attn.q_proj.linear", "self_attn.k_proj.linear", "self_attn.v_proj.linear", "self_attn.o_proj.linear",
    "self_attn.o_proj.gate_proj", "self_attn.relative_position_bias.linear",
}
LUT_NAMES = set(FMT_PARAMS) | {"nvfp4_e2m1", "nvfp4_e4m3"}
MANIFEST_FORMATS = re.compile(r"^inkling2-packed-v1(-[a-z0-9]+)*$")


class UnsupportedPackedFile(ValueError):
    """The packed file uses something this runtime does not implement."""


def _npz_keys(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as z:  # names only: no arrays are read
        return [n[:-4] if n.endswith(".npy") else n for n in z.namelist()]


def _layer_key_ok(key: str) -> bool:
    m = re.fullmatch(r"lin_(\w+?)_(codes|scales|signs_in|signs_out|levels)", key)
    if m:
        return True
    if key in ("gu_signs_in", "dn_signs_in", "gu_map", "dn_map", "gu_rg_map", "dn_rg_map"):
        return True
    m = re.fullmatch(r"(gu|dn)_(\w+)_(codes|scales|outer)", key)
    if m:
        fmt, part = m.group(2), m.group(3)
        return (fmt in FMT_PARAMS and part != "outer") or fmt == "nvfp4"
    return False


def check_layer(path: Path, meta: dict, arrays=None) -> list[str]:
    """One layer: its tensors, linear formats and expert format ids. `arrays` (an open npz) enables the map check."""
    name = path.name
    keys = _npz_keys(path)
    unknown = [k for k in keys if not _layer_key_ok(k)]
    if unknown:
        raise UnsupportedPackedFile(f"{name}: unknown tensors {unknown[:6]}")
    for lname, lm in meta.get("linears", {}).items():
        if lname not in LINEAR_NAMES:
            raise UnsupportedPackedFile(f"{name}: unknown linear {lname!r}")
        if lm.get("fmt") not in LINEAR_FMTS:
            raise UnsupportedPackedFile(f"{name}: linear {lname} uses unknown format {lm.get('fmt')!r}")
    missing = sorted(LINEAR_NAMES - set(meta.get("linears", {})))
    if missing:
        raise UnsupportedPackedFile(f"{name}: missing linears {missing}")
    for side in ("gu", "dn"):
        fmts = set(meta.get("experts", {}).get(side, {}))
        bad = fmts - set(FMT_PARAMS) - {"nvfp4"}
        if bad:
            raise UnsupportedPackedFile(f"{name}: experts ({side}) use unknown formats {sorted(bad)}")
        if arrays is not None:
            legacy, grouped = f"{side}_map", f"{side}_rg_map"
            if (legacy in keys) == (grouped in keys):
                raise UnsupportedPackedFile(f"{name}: expected exactly one {side} expert map")
            key = grouped if grouped in keys else legacy
            emap = np.asarray(arrays[key])
            if emap.ndim != (3 if key == grouped else 2) or emap.shape[-1] != 2 or not emap.size:
                raise UnsupportedPackedFile(f"{name}: invalid shape for {key}: {emap.shape}")
            pairs = emap.reshape(-1, 2)
            for fid in np.unique(pairs[:, 0]).tolist():
                if fid not in EXPERT_FIDS:
                    raise UnsupportedPackedFile(f"{name}: {side} experts use unknown format id {fid}")
                if key == grouped and fid == NVFP4_FID:
                    raise UnsupportedPackedFile(f"{name}: row-group NVFP4 experts are unsupported")
                fmt = EXPERT_FIDS[fid]
                codes = f"{side}_{fmt}_codes"
                if codes not in keys:
                    raise UnsupportedPackedFile(f"{name}: {key} references missing {codes}")
                with zipfile.ZipFile(path) as archive, archive.open(f"{codes}.npy") as handle:
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0):
                        shape, _, _ = np.lib.format.read_array_header_1_0(handle)
                    elif version in ((2, 0), (3, 0)):
                        # Pack arrays use simple ASCII numeric dtype descriptors. NPY 3 changes only the encoding.
                        shape, _, _ = np.lib.format.read_array_header_2_0(handle)
                    else:
                        raise UnsupportedPackedFile(f"{name}: {codes} uses unknown NPY version {version}")
                locs = pairs[pairs[:, 0] == fid, 1]
                if locs.max() >= shape[0] or locs.min() < 0:
                    raise UnsupportedPackedFile(f"{name}: {side} {fmt} index outside the slab")
    return keys


def check_buckets(path: Path, meta: list[dict]) -> None:
    keys = set(_npz_keys(path))
    for bm in meta:
        if bm["opt"] not in LINEAR_FMTS:
            raise UnsupportedPackedFile(f"{path.name}: bucket {bm['kb']} uses unknown format {bm['opt']!r}")
    allowed = {"order"} | {f"b{bm['kb']}_{p}" for bm in meta for p in ("codes", "scales", "signs_in", "levels")}
    unknown = sorted(keys - allowed)
    if unknown:
        raise UnsupportedPackedFile(f"{path.name}: unknown tensors {unknown[:6]}")


def check_config(cfg_path: Path) -> None:
    raw = json.loads(cfg_path.read_text())
    tc = raw.get("text_config", raw)
    if tc.get("n_shared_experts", 0):
        raise UnsupportedPackedFile(f"config: n_shared_experts={tc['n_shared_experts']} is not implemented")
    if tc.get("loop_config", {}).get("is_varloop", False):
        raise UnsupportedPackedFile("config: loop_config.is_varloop is not implemented")
    n = tc.get("num_hidden_layers", 0)
    local = set(tc.get("local_layer_ids", []))
    types = tc.get("layer_types") or ["hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding" for i in range(n)]  # transformers' default
    if {i for i, t in enumerate(types) if t.endswith("sliding")} != local:  # the runtimes read local_layer_ids
        raise UnsupportedPackedFile(f"config: local_layer_ids {sorted(local)} disagree with layer_types {types} "
                                    "(transformers reads layer_types, runtime_mlx local_layer_ids)")
    if tc.get("rel_extent") != tc.get("sliding_window_size"):  # runtime_mlx uses rel_extent for every layer,
        # transformers uses sliding_window_size for sliding layers; the released files set both to 1024
        raise UnsupportedPackedFile(f"config: rel_extent={tc.get('rel_extent')} != sliding_window_size={tc.get('sliding_window_size')} "
                                    "(runtime_mlx and transformers define sliding layers' bias range differently)")
    for swa, full in (("swa_num_attention_heads", "num_attention_heads"), ("swa_num_key_value_heads", "num_key_value_heads"), ("swa_head_dim", "head_dim")):
        if swa in tc and tc[swa] != tc[full]:  # the runtime gives sliding layers the full layers' head shape
            raise UnsupportedPackedFile(f"config: {swa}={tc[swa]} differs from {full}={tc[full]}, which is not implemented")


def validate_packed_dir(packed_dir: str | Path, deep: bool = True) -> dict:
    """Raises UnsupportedPackedFile on anything unknown. `deep` also opens each layer's expert maps (cheap: two
    small arrays a layer). Returns a summary of the formats the file uses."""
    d = Path(packed_dir)
    check_config(d / "config.json")
    man = d / "manifest.json"
    if man.exists():
        fmt = json.loads(man.read_text()).get("format", "")
        if not MANIFEST_FORMATS.match(fmt):
            raise UnsupportedPackedFile(f"manifest: unknown format {fmt!r}")
        ids = json.loads(man.read_text()).get("fmt_ids", {})
        known = {v: k for k, v in EXPERT_FIDS.items()}
        for k, v in ids.items():
            if known.get(k, v) != v:
                raise UnsupportedPackedFile(f"manifest: format {k} has id {v}, this runtime expects {known[k]}")
            if k not in known:
                raise UnsupportedPackedFile(f"manifest: unknown format {k!r} (id {v})")
    luts = set(_npz_keys(d / "luts.npz"))
    if luts - LUT_NAMES:
        raise UnsupportedPackedFile(f"luts.npz: unknown tables {sorted(luts - LUT_NAMES)}")
    if luts & {"nvfp4_e2m1", "nvfp4_e4m3"}:  # upstream reads these when present; the kernels use formats.py's tables
        from .formats import E2M1, e4m3_table_np

        lz = np.load(d / "luts.npz")
        if "nvfp4_e2m1" in luts and not np.array_equal(lz["nvfp4_e2m1"].astype(np.float32).reshape(-1), np.array(E2M1, np.float32)):
            raise UnsupportedPackedFile("luts.npz: nvfp4_e2m1 differs from the standard E2M1 table")
        if "nvfp4_e4m3" in luts and not np.array_equal(lz["nvfp4_e4m3"].astype(np.float32).reshape(-1), e4m3_table_np()):
            raise UnsupportedPackedFile("luts.npz: nvfp4_e4m3 differs from the standard E4M3 table")
    used: dict[str, int] = {}
    n_layers = json.loads((d / "config.json").read_text()).get("text_config", {}).get("num_hidden_layers")
    for p in sorted(d.glob("layer_*_packed.npz")):
        meta = json.loads(p.with_name(p.name.replace("_packed.npz", "_meta.json")).read_text())
        if deep:
            with np.load(p) as arrays:
                keys = check_layer(p, meta, arrays)
        else:
            keys = check_layer(p, meta)
        for side in ("gu", "dn"):
            if f"{side}_nvfp4_codes" in keys:
                used.setdefault("experts:nvfp4", 0)
        for lm in meta["linears"].values():
            used[lm["fmt"]] = used.get(lm["fmt"], 0) + 1
        for side in ("gu", "dn"):
            for f, c in meta.get("experts", {}).get(side, {}).items():
                used[f"experts:{f}"] = used.get(f"experts:{f}", 0) + c
    if n_layers is not None and len(list(d.glob("layer_*_packed.npz"))) != n_layers:
        raise UnsupportedPackedFile(f"expected {n_layers} layer files, found {len(list(d.glob('layer_*_packed.npz')))}")
    for kind in ("embed", "head"):
        meta = json.loads((d / f"{kind}_meta.json").read_text())
        check_buckets(d / f"{kind}_packed.npz", meta)
        for bm in meta:
            used[f"{kind}:{bm['opt']}"] = used.get(f"{kind}:{bm['opt']}", 0) + 1
    return used


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        print(path, json.dumps(validate_packed_dir(path), sort_keys=True))
