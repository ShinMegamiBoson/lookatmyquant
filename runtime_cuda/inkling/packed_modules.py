"""Packed attention linears and LM head for the standalone model (`model.py`): the codes stay on the GPU and run
through the decode GEMV (`gemv.py`, one or two token rows) or the fused decode + matmul (`triton_kernels.py`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .formats import ID_TO_FMT, rht_in, rht_out
from .gemv import packed_gemv
from .triton_kernels import packed_matmul

GEMV_MAX_ROWS = 2  # token rows per matrix up to which the decode GEMV beats the tensor-core kernel

FMT_ID = {v: k for k, v in ID_TO_FMT.items()}


def _t(a, device) -> torch.Tensor:
    a = np.asarray(a)
    if a.dtype == np.uint32:
        a = a.view(np.int32)
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def _signs(a, device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a).astype(np.float32)).to(device)


class _Mat:
    """One packed matrix (or a stack of same-format experts) and how to run it."""

    def __init__(self, fmt, m, n, codes, scales, lut=None, levels=None):
        self.fmt, self.m, self.n = fmt, m, n
        self.codes, self.scales, self.lut, self.levels = codes, scales, lut, levels

    def __call__(self, x16, **kw):
        if "block_expert" not in kw and x16.shape[0] <= GEMV_MAX_ROWS:
            return self.gemv(x16, out=kw.get("out"))
        return packed_matmul(x16, self.codes, self.scales, self.fmt, self.m, self.n, lut=self.lut, levels=self.levels, **kw)

    def gemv(self, x, expert_ids=None, x_div=1, out=None, zero_skipped=True):
        table = self.levels if self.fmt.startswith("lm") else self.lut
        return packed_gemv(x, self.codes, self.scales, self.fmt, self.m, self.n, table, expert_ids=expert_ids,
                           x_div=x_div, out=out, zero_skipped=zero_skipped)


class FastPackedLinear(nn.Module):
    def __init__(self, meta: dict, arrays, luts: dict[str, torch.Tensor], device):
        super().__init__()
        s, m, n, fmt = meta["safe"], int(meta["m"]), int(meta["n"]), meta["fmt"]
        self.in_features, self.out_features = n, m
        codes = _t(arrays[f"lin_{s}_codes"], device)
        scales = _t(arrays[f"lin_{s}_scales"], device).reshape(1, m)
        if fmt.startswith("lm"):
            self.mat = _Mat(fmt, m, n, codes, scales, levels=_t(arrays[f"lin_{s}_levels"].astype(np.float32), device))
        else:
            self.mat = _Mat(fmt, m, n, codes.reshape(1, m, -1), scales, lut=luts[fmt])
        self.signs_in = _signs(arrays[f"lin_{s}_signs_in"], device)
        self.signs_out = _signs(arrays[f"lin_{s}_signs_out"], device) if meta.get("out_rot") else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = rht_in(x.reshape(-1, shape[-1]), self.signs_in)
        y = self.mat(x2)
        if self.signs_out is not None:
            y = rht_out(y, self.signs_out)
        return y.reshape(*shape[:-1], self.out_features).to(x.dtype)


class FastPackedLMHead(nn.Module):
    def __init__(self, packed_dir: Path, luts: dict[str, torch.Tensor], device, hidden: int = 2304):
        super().__init__()
        meta = json.loads((packed_dir / "head_meta.json").read_text())
        npz = np.load(packed_dir / "head_packed.npz")
        self.mats, self.signs = [], []
        for bm in meta:
            kb, rows, fmt = bm["kb"], int(bm["rows"]), bm["opt"]
            codes = _t(npz[f"b{kb}_codes"], device)
            scales = _t(npz[f"b{kb}_scales"], device).reshape(1, rows)
            if fmt.startswith("lm"):
                mat = _Mat(fmt, rows, hidden, codes, scales, levels=_t(npz[f"b{kb}_levels"].astype(np.float32), device))
            else:
                mat = _Mat(fmt, rows, hidden, codes.reshape(1, rows, -1), scales, lut=luts[fmt])
            self.mats.append(mat)
            self.signs.append(_signs(npz[f"b{kb}_signs_in"], device))
        order = torch.from_numpy(np.asarray(npz["order"], dtype=np.int64)).to(device)
        self.inv_order = torch.empty_like(order)
        self.inv_order[order] = torch.arange(len(order), device=device)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.reshape(-1, h.shape[-1])
        parts = [mat(rht_in(x, s)) for mat, s in zip(self.mats, self.signs)]
        return torch.cat(parts, dim=-1)[:, self.inv_order].reshape(*h.shape[:-1], -1)
