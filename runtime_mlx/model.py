"""Loads a LookAtMyQuant export into an mlx_lm model: Lloyd–Max matrices, frequency-bucketed embedding and per-layer
tables, every other weight dense.

    model, tok = load("path/to/export")
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import lamq

_ROWS_SRC = """
    const uint ncodes = dims[0], wpr = dims[1];
    const uint g = thread_position_in_grid.x, t = thread_position_in_grid.y;
    if (g >= ncodes / 32 || t >= dims[2]) return;
    const device uint* gw = codes + (ulong)idx[t] * wpr + g * BITS;
    constexpr uint MASK = (1u << BITS) - 1u;
    uint w[BITS + 1];
    for (uint i = 0; i < BITS; i++) w[i] = gw[i];
    w[BITS] = 0;
    for (uint k = 0; k < 32; k++) {
        const uint bit = k * BITS, wi = bit >> 5, off = bit & 31u;
        uint c = w[wi] >> off;
        if (off + BITS > 32) c |= w[wi + 1] << (32 - off);
        out[(ulong)t * ncodes + g * 32 + k] = levels[c & MASK];
    }
"""
_rows = mx.fast.metal_kernel(name="lamq_rows", input_names=["codes", "idx", "levels", "dims"], output_names=["out"], source=_ROWS_SRC)


def decode_rows(codes, idx, levels, ncodes, bits):
    """The levels L[Q] of the selected rows: (len(idx), ncodes) float32."""
    T = idx.shape[0]
    dims = mx.array([ncodes, codes.shape[1], T], dtype=mx.uint32)
    return _rows(inputs=[codes, idx.astype(mx.int32), levels, dims], template=[("BITS", bits)], grid=(ncodes // 32, T, 1),
                 threadgroup=(min(ncodes // 32, 64), 1, 1), output_shapes=[(T, ncodes)], output_dtypes=[mx.float32])[0]


_LOOKUP_SRC = """
    // One threadgroup per (slice, token): find the token's bucket, decode its row's slice (any bit width, or 16-bit
    // dense), scale it, rotate it back (Hᵀ, then the signs) and write it in the original basis.
    const uint j = threadgroup_position_in_grid.x, t = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x, nt = threads_per_threadgroup.x;
    const uint tok = (uint)ids[t];
    const uint k = (uint)bucket_of[tok], loc = (uint)local_of[tok];
    const uint b = binfo[4 * k], soff = binfo[4 * k + 3];
    const device uint* rw = codes + (ulong)binfo[4 * k + 1] + (ulong)loc * binfo[4 * k + 2];
    device bfloat16_t* o = out + (ulong)t * D + j * WIDTH;
    if (b == 16) {  // a dense bucket: bf16 values, two to a word, not rotated
        for (uint i = tid; i < WIDTH; i += nt) {
            const uint e = j * WIDTH + i, wd = rw[e >> 1];
            o[i] = (bfloat16_t)as_type<float>(((e & 1u) ? (wd >> 16) : (wd & 0xFFFFu)) << 16);
        }
        return;
    }
    threadgroup float buf[WIDTH];
    const float sc = (float)scales[(ulong)(soff + loc) * SLICES + j];
    const device float* lv = levels + b * 256;
    const uint mask = (1u << b) - 1u;
    for (uint g = tid; g < WIDTH / 32; g += nt) {
        const device uint* gw = rw + (j * (WIDTH / 32) + g) * b;
        ulong acc = 0; uint have = 0, wi = 0;
        for (uint q = 0; q < 32; q++) {
            if (have < b) { acc |= ((ulong)gw[wi++]) << have; have += 32; }
            buf[g * 32 + q] = lv[(uint)acc & mask] * sc;
            acc >>= b; have -= b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Hᵀ = (Sylvester ⊗ P)ᵀ/√n: the Sylvester factor is symmetric, the p×p factor is applied as X·P (Pm = Pᵀ)
    uint stage = 0;
    for (uint h = 1; h < A; h <<= 1, stage++) {
        for (uint i = tid; i < (A / 2) * P; i += nt) {
            const uint pair = i / P, c = i - pair * P;
            const uint r = ((pair >> stage) << (stage + 1)) + (pair & (h - 1));
            const uint i0 = r * P + c, i1 = i0 + h * P;
            const float u = buf[i0], w = buf[i1];
            buf[i0] = u + w; buf[i1] = u - w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (P > 1) {
        for (uint r = tid; r < A; r += nt) {
            float v[P];
            for (uint q = 0; q < P; q++) v[q] = buf[r * P + q];
            for (uint c = 0; c < P; c++) {
                float a = 0.0f;
                for (uint q = 0; q < P; q++) a += v[q] * Pm[c * P + q];
                buf[r * P + c] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float norm = rsqrt((float)WIDTH);
    const device float* sg = signs + ((ulong)k * SLICES + j) * WIDTH;
    for (uint i = tid; i < WIDTH; i += nt) o[i] = (bfloat16_t)(buf[i] * norm * sg[i]);
"""
_lookup = mx.fast.metal_kernel(name="lamq_lookup", input_names=["ids", "bucket_of", "local_of", "binfo", "codes", "scales", "levels", "signs", "Pm"],
                               output_names=["out"], source=_LOOKUP_SRC)

FUSE_MAX_N = 12288  # widest input whose rotation runs inside the matrix–vector kernel


def _levels(bits):
    return mx.array(np.load(Path(__file__).parent / "ref" / f"lloyd_max_{bits}.npy"))


class LMLinear(nn.Module):
    """y = R_outᵀ · S ⊙ L[Q] · R_in x, or S ⊙ L[Q] · R_in x for a matrix rotated on the input side only (s_out None)."""

    def __init__(self, codes, scales, s_in, s_out, bits):
        super().__init__()
        self.codes = codes.view(mx.uint32)
        self.scales = scales.astype(mx.float16)
        self.levels = _levels(bits)
        self.bits, self.m, self.n = bits, scales.shape[0], s_in.shape[0]
        self.r_in = lamq.Rot(np.array(s_in.astype(mx.float32)))
        self.r_out = None if s_out is None else lamq.Rot(np.array(s_out.astype(mx.float32)))
        # one token: the input rotation and the product in one launch, for inputs narrow enough that the rotation is cheap
        self.fused = lamq.RotGemv(self.r_in, self.codes, self.levels, self.scales, self.m, bits) if self.n <= FUSE_MAX_N else None
        self.freeze()

    def __call__(self, x):
        lead = x.shape[:-1]
        if self.fused is not None and int(np.prod(lead)) == 1:
            y = self.fused(x.reshape(-1).astype(mx.bfloat16)).reshape(*lead, self.m)
            if self.r_out is None:
                return y.astype(x.dtype)
            return self.r_out.bwd(y.astype(mx.float32)).astype(x.dtype)
        T = int(np.prod(lead))
        xr = self.r_in.many(x) if T > 1 else self.r_in.fwd(x.astype(mx.float32)).astype(mx.float16)
        if T == 1:
            y = lamq.lm_gemv(xr.reshape(-1), self.codes, self.levels, self.scales, self.m, self.n, self.bits).reshape(*lead, self.m)
        elif T <= lamq.GEMM_MAX_T:  # a short prompt: from the codes, without expanding them
            y = lamq.lm_gemm_few(xr.reshape(T, self.n), self.codes, self.levels, self.scales, self.m, self.n, self.bits).reshape(*lead, self.m)
        else:
            y = xr @ lamq.lm_dequant(self.codes, self.levels, self.scales, self.m, self.n, self.bits).T
        if self.r_out is None:
            return y.astype(x.dtype)
        return self.r_out.bwd(y.astype(mx.float32)).astype(x.dtype)


class LMTable(nn.Module):
    """A table cut into buckets of rows (by token frequency), each with its own code; a bucket of a table read in
    slices (Gemma 4's per-layer inputs) has one rotation and one row scale per slice. __call__ looks rows up; as_linear
    is the tied output head (every row, each bucket in its own rotated basis)."""

    def __init__(self, W, name, meta):
        super().__init__()
        V, d = meta["shape"]
        self.V, self.d, self.block = V, d, meta["block"] or d
        self.slices = d // self.block
        self.buckets, self.rows, self._head = [], [], None
        bucket_of, local_of, order = np.zeros(V, np.int32), np.zeros(V, np.int32), []
        for k, b in enumerate(meta["buckets"]):
            key = f"{name}.b{k}"
            rows = W[f"{key}.rows"]
            r = np.array(rows)
            bucket_of[r], local_of[r] = k, np.arange(len(r))
            order.append(r)
            self.rows.append(mx.array(r.astype(np.int32)))
            if b["kind"] == "dense":
                self.buckets.append({"kind": "dense", "W": W[f"{key}.dense"]})
            else:
                width = self.block
                wpad = -(-width // 32) * 32
                self.buckets.append({"kind": "lm", "bits": b["bits"], "codes": W[f"{key}.codes"].view(mx.uint32), "scales": W[f"{key}.scales"].astype(mx.float32),
                                     "rot": lamq.Rot(np.array(W[f"{key}.signs"].astype(mx.float32))), "width": width, "wpad": wpad,
                                     "levels": _levels(b["bits"])})
        self.bucket_of, self.local_of = mx.array(bucket_of), mx.array(local_of)
        self.inv = mx.array(np.argsort(np.concatenate(order)).astype(np.int32))  # concatenated bucket outputs → token order
        self._one_kernel_lookup()
        self.freeze()

    def _one_kernel_lookup(self):
        """Every bucket's codes (or bf16 rows) in one buffer, with per-bucket bits and offsets, so that a lookup is one
        launch. The buckets' own arrays become views of it."""
        self.fast = self.block % 32 == 0
        if not self.fast:
            return
        parts, binfo, scales, signs, woff, soff = [], [], [], [], 0, 0
        ones = mx.ones((self.slices, self.block), mx.float32)
        for b in self.buckets:
            if b["kind"] == "dense":
                words = b["W"].astype(mx.bfloat16).view(mx.uint32)
                rows, bits, sg = words.shape[0], 16, ones
            else:
                words, rows, bits, sg = b["codes"], b["codes"].shape[0], b["bits"], b["rot"].s.reshape(self.slices, self.block)
                scales.append(b["scales"].astype(mx.float16).reshape(-1))
            binfo.append([bits, woff, words.shape[1], soff])
            parts.append(words.reshape(-1)); signs.append(sg)
            woff += words.size
            soff += rows if b["kind"] != "dense" else 0
        self.codes_all = mx.concatenate(parts)
        self.binfo = mx.array(np.array(binfo, dtype=np.uint32).reshape(-1))
        self.scales_all = mx.concatenate(scales) if scales else mx.zeros((1,), mx.float16)
        self.signs_all = mx.stack(signs).reshape(-1)
        lv = np.zeros((17, 256), np.float32)
        for bits in {b["bits"] for b in self.buckets if b["kind"] == "lm"}:
            lv[bits, : 2 ** bits] = np.array(_levels(bits))
        self.levels_all = mx.array(lv.reshape(-1))
        rot = next((b["rot"] for b in self.buckets if b["kind"] == "lm"), lamq.Rot(np.ones(self.block)))
        self.A, self.p, self.Pm_bwd = rot.A, rot.p, rot.Pm_bwd
        for (bits, w0, stride, _), b in zip(binfo, self.buckets):  # the per-bucket arrays (the output head reads them) share the buffer
            if b["kind"] == "lm":
                b["codes"] = self.codes_all[w0 : w0 + b["codes"].size].reshape(-1, stride)
        mx.eval(self.codes_all, self.scales_all, self.signs_all, *[b["codes"] for b in self.buckets if b["kind"] == "lm"])
        self.nt = 256 if self.block >= 512 else self.block // 2

    def _rows(self, b, local):
        """Bucket b's rows `local`, in the original basis: (T, d)."""
        if b["kind"] == "dense":
            return b["W"][local].astype(mx.float32)
        T = local.shape[0]
        q = decode_rows(b["codes"], local, b["levels"], self.slices * b["wpad"], b["bits"]).reshape(T, self.slices, b["wpad"])[..., : b["width"]]
        q = q * b["scales"][local][..., None]
        return b["rot"].bwd(q).reshape(T, self.d)

    def __call__(self, ids):
        shp = ids.shape
        flat = ids.reshape(-1)
        if self.fast:
            T = flat.shape[0]
            out = _lookup(inputs=[flat.astype(mx.int32), self.bucket_of, self.local_of, self.binfo, self.codes_all, self.scales_all, self.levels_all,
                                  self.signs_all, self.Pm_bwd],
                          template=[("WIDTH", self.block), ("SLICES", self.slices), ("D", self.d), ("A", self.A), ("P", self.p)],
                          grid=(self.slices * self.nt, T, 1), threadgroup=(self.nt, 1, 1), output_shapes=[(T, self.d)], output_dtypes=[mx.bfloat16])[0]
            return out.reshape(*shp, self.d)
        bk, loc = self.bucket_of[flat], self.local_of[flat]
        out = mx.zeros((flat.shape[0], self.d), dtype=mx.float32)
        for k, b in enumerate(self.buckets):
            sel = bk == k
            out = mx.where(sel[:, None], self._rows(b, mx.where(sel, loc, 0)), out)
        return out.reshape(*shp, self.d).astype(mx.bfloat16)

    def as_linear(self, h):
        """h @ Eᵀ for every row; each bucket: S ⊙ (L[Q] · R h) in its own basis."""
        lead = h.shape[:-1]
        if self._head is not None and h.size == self.d:  # one token: fastpath.head, one launch
            return self._head(h)[0].reshape(*lead, self.V).astype(h.dtype)
        x = h.reshape(-1, self.d).astype(mx.float32)
        outs = []
        for b in self.buckets:
            if b["kind"] == "dense":
                outs.append(x @ b["W"].astype(mx.float32).T)
                continue
            xr = b["rot"].many(x) if x.shape[0] > 1 else b["rot"].fwd(x).astype(mx.float16)
            m = b["codes"].shape[0]
            if x.shape[0] == 1:
                outs.append(lamq.lm_gemv(xr.reshape(-1), b["codes"], b["levels"], b["scales"][:, 0].astype(mx.float16), m, self.d, b["bits"])[None].astype(mx.float32))
            elif x.shape[0] <= lamq.GEMM_MAX_T:  # a short prompt: from the codes
                outs.append(lamq.lm_gemm_few(xr, b["codes"], b["levels"], b["scales"][:, 0].astype(mx.float16), m, self.d, b["bits"]))
            else:
                Wb = lamq.lm_dequant(b["codes"], b["levels"], b["scales"][:, 0].astype(mx.float16), m, self.d, b["bits"])
                outs.append((xr @ Wb.T).astype(mx.float32))
        y = mx.concatenate(outs, axis=-1)[:, self.inv]
        return y.reshape(*lead, self.V).astype(h.dtype)


def _set(root, path, module):
    parts = path.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, parts[-1], module)


def load(path):
    from mlx_lm.utils import _get_classes, load_tokenizer
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    meta = json.loads((path / "lamq.json").read_text())
    W = mx.load(str(path / "model.safetensors"))
    if (path / "buffers.safetensors").exists():  # tensors an earlier exporter left out (Gemma 4's layer scales)
        W.update(mx.load(str(path / "buffers.safetensors")))
    model_class, args_class = _get_classes(config=config)
    model = model_class(args_class.from_dict(config))
    aux = (".codes", ".scales", ".s_in", ".s_out", ".rows", ".signs", ".dense")
    coded = set(meta["matrices"]) | set(meta["tables"])
    dense = {k: v for k, v in W.items() if k not in coded and not k.endswith(aux)}
    model.load_weights(list(model.sanitize(dense).items()), strict=False)
    rename = lambda n: next(iter(model.sanitize({n: mx.zeros((1,))})))  # Hugging Face name → mlx_lm name
    for name, m in meta["matrices"].items():
        _set(model, rename(name).removesuffix(".weight"), LMLinear(W[f"{name}.codes"], W[f"{name}.scales"], W[f"{name}.s_in"], W.get(f"{name}.s_out"), m["bits"]))
    for name, t in meta["tables"].items():
        _set(model, rename(name).removesuffix(".weight"), LMTable(W, name, t))
    model.eval()
    return model, load_tokenizer(path)
