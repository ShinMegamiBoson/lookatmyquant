"""Writes a quantized plan in the LookAtMyQuant runtime format (runtime_mlx/): for every matrix and table bucket the plan
codes with Lloyd–Max or trellis codes, the packed codes, the 16-bit row scales and the signs of its randomized Hadamard
transforms; every other text weight as bf16.

    model.safetensors   <name>.codes (uint32 bit streams, stored as int32), <name>.scales (fp16), <name>.s_in / .s_out (int8);
                        tables: <name>.b<k>.rows (int32 token ids), .codes, .scales (rows × slices), .signs (slices × width);
                        16-bit buckets <name>.b<k>.dense; every other weight under its own name;
                        trellis.k<k>: a trellis code's 2^12 state values (fp32)
    lamq.json           what each name holds

A trellis row is one stream of k-bit symbols, its path along the columns from state 0: later columns at lower bits, the
runs of 32 stored last column first. An entry's state is the 12 bits at its place in the stream.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from opgraph import codes as C
from opgraph import nested as N

SKIP = ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector")


def export_saved_codes(saved, out, *, dense_tensors, model_dir=None):
    """Lossless grouped export for saved mixed-width/matrix-specific code tables.

    This writes the separate opgraph-packed-1 format read by runtime_mlx.artifact.
    dense_tensors must contain the complete correctly named model state; compact
    code entries alone omit unchanged weights and kept embedding rows.
    """
    from compiler_artifact import export_saved_codes as export
    return export(saved, out, dense_tensors=dense_tensors, model_dir=model_dir)


def signs(n: int, seed: int) -> torch.Tensor:
    """The signs s of rht(n, seed) = H·diag(s)."""
    return torch.where(torch.rand(n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) < 0.5, -1.0, 1.0)


def codes_of(values: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """The index of the nearest level for each value (the values are levels, up to rounding)."""
    mids = (levels[1:] + levels[:-1]) / 2
    return torch.searchsorted(mids.contiguous(), values.contiguous())


def pack(Q: np.ndarray, bits: int) -> np.ndarray:
    """(m, n) codes → (m, ⌈n/32⌉·bits) uint32: each run of 32 codes is a little-endian bit stream of `bits` words."""
    m, n = Q.shape
    if n % 32:  # pad each row to whole runs of 32 (only tiny test models need this)
        Q = np.concatenate([Q, np.zeros((m, 32 - n % 32), dtype=Q.dtype)], 1)
        n = Q.shape[1]
    q = Q.astype(np.uint64).reshape(m, n // 32, 32)
    out = np.zeros((m, n // 32, bits), dtype=np.uint32)
    for k in range(32):
        w, off = divmod(k * bits, 32)
        v = q[:, :, k] << np.uint64(off)
        out[:, :, w] |= (v & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        if off + bits > 32:
            out[:, :, w + 1] |= (v >> np.uint64(32)).astype(np.uint32)
    return out.reshape(m, -1)


def trellis_symbols(core, k: int, L: int = 12) -> torch.Tensor:
    """A trellis-restricted core (rows × n, on the state values) as its symbols, last column first: pack() then lays out the stream."""
    v = torch.tensor(core.domain.levels, dtype=torch.float64)
    order = torch.argsort(v)
    states = order[torch.searchsorted(v[order], core.data.double().contiguous()).clamp(max=len(v) - 1)]
    assert torch.equal(v[states], core.data.double()) and C.is_trellis_path(states, L, k) and bool((states[:, 0] >> k == 0).all()), "not a path from state 0"
    return (states & ((1 << k) - 1)).flip(1)


def _u32(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a).view(np.int32))


class Exporter:
    def __init__(self):
        self.tensors: dict[str, torch.Tensor] = {}
        self.meta = {"format": "lamq-1", "matrices": {}, "tables": {}}
        self.coded: set[str] = set()

    def matrix(self, name: str, family: str, bits: int, net, seed: int, in_seed: int | None = None):
        """One matrix's word (Lloyd–Max: nested_words with a dedicated code; or trellis): W ≈ R_outᵀ diag(S) L[Q] R_in, or,
        rotated on the input side only, W ≈ diag(S) L[Q] R_in."""
        if family == "fp":
            return  # kept as stored: written with the dense weights
        if family not in ("lm", "tr"):
            raise ValueError(f"{name}: '{family}{bits}' has no runtime kernel yet")
        Wc, S = net.cores["W"].data.double(), net.cores["W.S"].data.double()
        extra = {}
        if family == "lm":
            lv = N.lloyd_max(bits).double()
            Q = codes_of(Wc, lv)
            stored, Wv = Q, lv[Q]
        else:
            assert net.cores["W"].domain.fixed_start, f"{name}: the runtime reads trellis paths along the rows (--trellis-rows)"
            stored, Wv = trellis_symbols(net.cores["W"], bits), Wc
            self.trellis(bits, net.cores["W"].domain.levels)
            extra = {"family": "tr"}
        m, n = Wv.shape
        two_sided = "I[u]" in net.cores
        s_out, s_in = signs(m, seed), signs(n, seed + 1 if in_seed is None else in_seed)
        # check: the stored pieces rebuild the matrix the harness evaluates
        Wq = (S[:, None] * Wv) @ (C.hadamard(n) * s_in[None, :])
        if two_sided:
            Wq = (C.hadamard(m) * s_out[None, :]).T @ Wq
        err = (Wq - net.matrix(dtype=torch.float64)).abs().max().item()
        assert err < 1e-6, f"{name}: export does not rebuild the matrix (max error {err:.2e})"
        self.tensors[f"{name}.codes"] = _u32(pack(stored.numpy(), bits))
        self.tensors[f"{name}.scales"] = S.to(torch.float16)
        self.tensors[f"{name}.s_in"] = s_in.to(torch.int8)
        if two_sided:
            self.tensors[f"{name}.s_out"] = s_out.to(torch.int8)
        self.meta["matrices"][name] = {"bits": bits, "shape": [m, n], "out_rot": two_sided,
                                       "in_seed": seed + 1 if in_seed is None else in_seed, **extra}
        self.coded.add(name)

    def trellis(self, k: int, values):
        self.tensors.setdefault(f"trellis.k{k}", torch.as_tensor(values, dtype=torch.float32))

    def table_bucket(self, name: str, k: int, rows: torch.Tensor, program: str, parts: list[dict], shape, block):
        """One bucket of a table's rows. `parts` has one entry per slice (one for a table read whole)."""
        t = self.meta["tables"].setdefault(name, {"shape": list(shape), "block": block, "buckets": []})
        key = f"{name}.b{k}"
        self.tensors[f"{key}.rows"] = rows.to(torch.int32).cpu()
        if "dense" in parts[0]:
            self.tensors[f"{key}.dense"] = torch.cat([p["dense"] for p in parts], 1).to(torch.bfloat16)
            t["buckets"].append({"program": program, "kind": "dense", "rows": len(rows)})
        else:
            if any("codes" not in p for p in parts):
                raise ValueError(f"{name} bucket {k}: '{program}' has no runtime kernel yet")
            bits, values = parts[0]["bits"], parts[0]["values"]
            self.tensors[f"{key}.codes"] = _u32(np.concatenate([pack(p["codes"].numpy(), bits) for p in parts], 1))
            self.tensors[f"{key}.scales"] = torch.stack([p["scales"] for p in parts], 1).to(torch.float16)
            self.tensors[f"{key}.signs"] = torch.stack([signs(p["width"], p["seed"]) for p in parts]).to(torch.int8)
            t["buckets"].append({"program": program, "kind": "lm", "bits": bits, "rows": len(rows), "slices": len(parts), **({} if values is None else {"family": "tr"})})
            if values is not None:
                self.trellis(bits, values)
        self.coded.add(name)

    def write(self, out: str, model, model_dir: str):
        from safetensors.torch import save_file
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        dense = 0
        params = dict(model.named_parameters())
        kept = model.state_dict()  # named_parameters leaves out a tied head; state_dict does not, and it holds the persistent buffers (Gemma 4's layer scales)
        for n, p in [*params.items(), *((n, b) for n, b in model.named_buffers() if n in kept)]:
            if n in self.coded or any(s in n for s in SKIP):
                continue
            self.tensors[n] = p.detach().to(torch.bfloat16).cpu().contiguous()
            dense += p.numel()
        self.meta["dense weights"] = dense
        save_file({k: v.contiguous() for k, v in self.tensors.items()}, str(out / "model.safetensors"))
        (out / "lamq.json").write_text(json.dumps(self.meta, indent=1))
        for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "tokenizer.model"):
            if (Path(model_dir) / f).exists():
                shutil.copy(Path(model_dir) / f, out / f)
        return out
