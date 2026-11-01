"""Writes the reference math that test_core.py checks the runtime against, from the quantizer's own code, into runtime_mlx/ref
(1.6 GB). Run it once from the repository root, in the environment that has torch:

    python runtime_mlx/make_ref.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from opgraph import codes as C
from opgraph import nested as N

REF = ROOT / "ref"
REF.mkdir(exist_ok=True)
for n in (256, 512, 1536, 2048, 4096, 6144, 12288):
    np.save(REF / f"hadamard_{n}.npy", C.hadamard(n).numpy().astype(np.float32))
    np.save(REF / f"rht_{n}_12345.npy", C.rht(n, 12345).numpy().astype(np.float32))
for bits in (2, 3, 4, 5, 6, 8):
    np.save(REF / f"lloyd_max_{bits}.npy", N.lloyd_max(bits).numpy().astype(np.float32))
print("wrote", REF)
