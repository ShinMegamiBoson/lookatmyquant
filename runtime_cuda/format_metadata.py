"""Packed Inkling formats shared by the CUDA engines and pack validator."""
from functools import lru_cache

import numpy as np

HASH_MULT_U32 = 0x9E3779B1
ID_TO_FMT = {0: "ht1", 1: "ht2", 2: "ht3", 3: "rtr1", 4: "rtr2", 5: "rtr3", 6: "rtr4"}
NVFP4_FID = 7  # experts only; not a trellis format, so not in ID_TO_FMT / FMT_PARAMS
EXPERT_FIDS = {**ID_TO_FMT, NVFP4_FID: "nvfp4"}
# fmt -> (k bits per weight, g weights per state, K table entries, hash multiplier)
FMT_PARAMS = {
    "ht1": (1, 4, 512, HASH_MULT_U32),
    "ht2": (2, 4, 512, HASH_MULT_U32),
    "ht3": (3, 2, 1024, HASH_MULT_U32),
    "rtr1": (1, 1, 4096, 0),
    "rtr2": (2, 1, 4096, 0),
    "rtr3": (3, 1, 4096, 0),
    "rtr4": (4, 1, 4096, 0),
}

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


@lru_cache(maxsize=1)
def e4m3_table_np() -> np.ndarray:
    """float32[256], the same table as runtime_mlx's `nvfp4_luts_mx` (the NaN code 0x7F / 0xFF reads as 0)."""
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        exp, mant = (b >> 3) & 0xF, b & 0x7
        if exp == 15 and mant == 7:
            t[b] = 0.0
        elif exp == 0:
            t[b] = sign * (mant / 8.0) * 2.0**-6
        else:
            t[b] = sign * (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
    return t
