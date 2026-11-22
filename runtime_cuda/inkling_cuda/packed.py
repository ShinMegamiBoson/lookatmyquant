"""LookAtMyQuant packed matrices and their NumPy reference decoder."""
from dataclasses import dataclass
from functools import lru_cache
import json
import math

import numpy as np


from ..format_metadata import FMT_PARAMS, ID_TO_FMT

EXPERT_FORMATS = tuple(ID_TO_FMT.values())
FORMATS = {**FMT_PARAMS, "lm6": (6, 1, 64, 0), "lm8": (8, 1, 256, 0)}



@lru_cache(maxsize=16)
def hadamard_factors(n):
    """Use the same Sylvester/Paley factorization and orientation as MLX."""
    for exponent in range(n.bit_length() - 1, -1, -1):
        a = 1 << exponent
        if n % a:
            continue
        b = n // a
        prime = b - 1
        if b == 1:
            return (1 << (exponent // 2), np.ones((1 << (exponent - exponent // 2), 1), dtype=np.float32))
        if b == 2:
            return a, np.array([[1, 1], [1, -1]], dtype=np.float32)
        if prime % 4 == 3 and all(prime % d for d in range(2, math.isqrt(prime) + 1)):
            residues = {i * i % prime for i in range(1, prime)}
            chars = np.array([0] + [1 if i in residues else -1 for i in range(1, prime)])
            idx = (np.arange(prime)[None, :] - np.arange(prime)[:, None]) % prime
            matrix = np.eye(b, dtype=np.float32)
            matrix[0, 1:] = 1
            matrix[1:, 0] = -1
            matrix[1:, 1:] += chars[idx]
            return a, matrix
    raise ValueError(f"No Hadamard factorization for width {n}")


def rotate(x, signs, inverse=False):
    n = x.shape[-1]
    a, b = hadamard_factors(n)
    v = np.array(x if inverse else x * signs, dtype=np.float32, copy=True, order="C")
    if b.shape[1] == 1:
        # Both factors are Sylvester. Their Kronecker product is Sylvester.
        a = n
        b = np.ones((1, 1), dtype=np.float32)
    v = v.reshape(-1, a, b.shape[0])
    if b.shape[0] > 1:
        v = v @ (b if inverse else b.T)
    stride = 1
    while stride < a:
        blocks = v.reshape(v.shape[0], -1, 2, stride, b.shape[0])
        low = blocks[:, :, 0].copy()
        high = blocks[:, :, 1]
        blocks[:, :, 0] = low + high
        blocks[:, :, 1] = low - high
        stride *= 2
    out = v.reshape(x.shape) * np.float32(n ** -0.5)
    return out * signs if inverse else out


@dataclass
class Slab:
    width: int
    rows: int
    fmt: str
    codes: np.ndarray
    scales: np.ndarray
    table: np.ndarray
    cuda: object = None
    table_pointer: object = None

    def __post_init__(self):
        self.bits, self.group, count, self.multiplier = FORMATS[self.fmt]
        self.lloyd = self.fmt.startswith("lm")
        self.hash_shift = 32 - (count - 1).bit_length()
        self.words, remainder = divmod(self.width * self.bits, 32)
        if remainder or self.width % self.group:
            raise ValueError(f"Unsupported alignment for {self.fmt} at width {self.width}")
        self.experts, leftover = divmod(self.scales.size, self.rows)
        if leftover or self.table.size != count * self.group:
            raise ValueError(f"Invalid scale or table shape for {self.fmt}")
        if self.codes.dtype not in (np.dtype("uint8"), np.dtype("uint32")):
            raise ValueError(f"Unsupported code dtype {self.codes.dtype}")
        self.codes = np.ascontiguousarray(self.codes).view("<u4").reshape(self.experts, self.rows, self.words)
        self.scales = np.ascontiguousarray(self.scales, dtype=np.float16).reshape(self.experts, self.rows)
        self.table = np.ascontiguousarray(self.table, dtype=np.float32).reshape(count, self.group)
        if self.cuda is not None:
            self.codes = self.cuda.upload(self.codes)
            self.scales = self.cuda.upload(self.scales)
            self.table = self.table_pointer if self.table_pointer is not None else self.cuda.upload(self.table)

    def decode(self, expert, rows):
        column = np.arange(self.width, dtype=np.int64)
        bit = (column if self.lloyd else self.width - (column // self.group + 1) * self.group) * self.bits
        word, offset = np.divmod(bit, 32)
        code = self.codes[expert, rows]
        lo = code[..., word].astype(np.uint64)
        hi = code[..., np.minimum(word + 1, self.words - 1)].astype(np.uint64)
        hi[..., word + 1 == self.words] = 0
        pair = lo | (hi << np.uint64(32))
        mask = (1 << self.bits) - 1 if self.lloyd else 4095
        index = (pair >> offset.astype(np.uint64)) & np.uint64(mask)
        if self.multiplier:
            index = ((index * np.uint64(self.multiplier)) & np.uint64(0xffffffff)) >> np.uint64(self.hash_shift)
        return self.table[index, column % self.group] * self.scales[expert, rows][..., None]


def project(inputs, slabs, routes, x_div=1):
    rows = slabs[0].rows
    if slabs[0].cuda is not None:
        return slabs[0].cuda.project(inputs, slabs, routes, rows, x_div)
    out = np.empty((routes[0].size, rows), dtype=np.float32)
    for slab, experts in zip(slabs, routes, strict=True):
        for slot in np.flatnonzero(experts >= 0):
            # Chunk even the CPU oracle so a large vocabulary never expands in full.
            for lo in range(0, rows, 256):
                hi = min(lo + 256, rows)
                out[slot, lo:hi] = slab.decode(experts[slot], slice(lo, hi)) @ inputs[slot // x_div]
    return out


class Weights:
    def __init__(self, folder, cuda):
        self.cuda = cuda
        with np.load(folder / "luts.npz", allow_pickle=False) as lut:
            self.tables = {key: lut[key] for key in lut.files}
        self.device_tables = {}

    def slab(self, archive, prefix, fmt, width, rows, cpu=False):
        cuda = None if cpu else self.cuda
        pointer = None
        if fmt.startswith("lm"):
            table = archive[f"{prefix}_levels"]
        else:
            table = self.tables[fmt]
            if cuda is not None:
                if fmt not in self.device_tables:
                    self.device_tables[fmt] = cuda.upload(np.asarray(table, dtype=np.float32))
                pointer = self.device_tables[fmt]
        return Slab(width, rows, fmt, archive[f"{prefix}_codes"], archive[f"{prefix}_scales"],
                    table, cuda, pointer)


class Linear:
    def __init__(self, weights, archive, meta):
        prefix = f"lin_{meta['safe']}"
        self.slab = weights.slab(archive, prefix, meta["fmt"], int(meta["n"]), int(meta["m"]))
        self.signs = archive[f"{prefix}_signs_in"].astype(np.float32)
        self.out_signs = archive[f"{prefix}_signs_out"].astype(np.float32) if meta.get("out_rot") else None
        self.route = [np.array([0], dtype=np.int32)]

    def __call__(self, x):
        out = project(rotate(x, self.signs)[None, :], [self.slab], self.route)[0]
        return out if self.out_signs is None else rotate(out, self.out_signs, inverse=True)


class Vocabulary:
    def __init__(self, folder, weights, name, width, device_embedding=False):
        meta = json.loads((folder / f"{name}_meta.json").read_text(encoding="utf-8"))
        self.buckets = []
        self.route = [np.array([0], dtype=np.int32)]
        lo = 0
        with np.load(folder / f"{name}_packed.npz", allow_pickle=False) as archive:
            order = archive["order"].reshape(-1).astype(np.int32)
            if not np.array_equal(np.sort(order), np.arange(order.size)):
                raise ValueError(f"Invalid {name} vocabulary permutation")
            self.inverse = np.argsort(order)
            self.order = order
            for entry in meta:
                prefix = f"b{entry['kb']}"
                rows = int(entry["rows"])
                slab = weights.slab(archive, prefix, entry["opt"], width, rows,
                                    cpu=name == "embed" and not device_embedding)
                signs = archive[f"{prefix}_signs_in"].astype(np.float32)
                self.buckets.append((lo, lo + rows, slab, signs))
                lo += rows
        if lo != order.size:
            raise ValueError(f"Invalid {name} bucket rows")

    def embed(self, token):
        if not 0 <= token < self.order.size:
            raise ValueError(f"Token {token} is outside the embedding vocabulary")
        position = self.inverse[token]
        for lo, hi, slab, signs in self.buckets:
            if lo <= position < hi:
                return rotate(slab.decode(0, position - lo), signs, inverse=True)
        raise AssertionError("Invalid embedding lookup")

    def logits(self, x):
        out = np.empty(self.order.size, dtype=np.float32)
        for lo, hi, slab, signs in self.buckets:
            out[self.order[lo:hi]] = project(rotate(x, signs)[None, :], [slab], self.route)[0]
        return out
