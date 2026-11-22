"""Per-GPU kernel tuning, and choices made from the card's limits.

Every launch parameter that was picked by measurement (GEMV block shapes, warps and pipeline stages, which trellis
formats use the two-weight lookup tables, the decode-attention split) comes from here instead of from constants
tuned on one card. Resolution, most specific first:

1. a measured table for this exact card (`tuning/<key>.json`, written by `runtime_cuda/tune.py`);
2. the measured table of the closest card of the same architecture (compute capability), with the SM-count-dependent
   values (attention split) rescaled to this card;
3. rules from the card's limits (`DEFAULTS` plus `rule_based`): a two-weight table is used only when it fits in the
   card's L1 with room to spare, judged from the measured fall-off of lookup speed with table size.

`LAMQ_TUNING=<file.json>` forces one table (benchmarking); `describe()` says which source was used and why.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import torch

TABLE_DIR = Path(__file__).with_name("tuning")

# Unified L1 + shared memory per SM, by compute capability (NVIDIA tuning guides). torch reports only the shared part.
L1_PER_SM_KB = {(7, 5): 96, (8, 0): 192, (8, 6): 128, (8, 7): 192, (8, 9): 128, (9, 0): 256, (10, 0): 256,
                (10, 3): 256, (12, 0): 128, (12, 1): 128}

# Two-weight table sizes (bytes) for the scalar trellis formats: 2^(12 + k) entries x 2 float16.
MULTITABLE_BYTES = {"rtr1": (1 << 13) * 4, "rtr2": (1 << 14) * 4, "rtr3": (1 << 15) * 4}


@dataclass(frozen=True)
class GpuInfo:
    name: str
    cc: tuple[int, int]
    sms: int
    l2_bytes: int
    smem_per_sm: int
    smem_per_block_optin: int
    l1_per_sm: int  # unified L1/shared bytes per SM
    mem_gbs: float  # theoretical DRAM bandwidth

    @property
    def key(self) -> str:
        """File-name key, e.g. `nvidia-rtx-a4500_sm86_56sm`."""
        slug = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")
        return f"{slug}_sm{self.cc[0]}{self.cc[1]}_{self.sms}sm"


def gpu_info(device: torch.device | int | str | None = None) -> GpuInfo:
    p = torch.cuda.get_device_properties(torch.device("cuda", torch.cuda.current_device()) if device is None else device)
    cc = (p.major, p.minor)
    l1 = L1_PER_SM_KB.get(cc, round(p.shared_memory_per_multiprocessor / 1024 * 1.28)) * 1024
    mem = 2 * p.memory_clock_rate * 1e3 * p.memory_bus_width / 8 / 1e9  # DDR: two transfers a clock
    return GpuInfo(p.name, cc, p.multi_processor_count, p.L2_cache_size, p.shared_memory_per_multiprocessor,
                   p.shared_memory_per_block_optin, l1, round(mem, 1))


@dataclass
class Tuning:
    # (BLOCK_M, BLOCK_C, num_warps, num_stages) per launcher; "single:<family>" for packed_gemv by format family
    gemv: dict[str, list[int]] = field(default_factory=dict)
    multitable: dict[str, int] = field(default_factory=dict)  # trellis format -> weights per lookup
    attn_nsplit: int = 16
    attn_block_k: int = 32
    prefill: list[int] = field(default_factory=lambda: [64, 64])  # packed_matmul BLOCK_M, BLOCK_N (BLOCK_T is fixed at 16)
    source: str = "default"

    def gemv_cfg(self, kind: str) -> tuple[int, int, int, int]:
        return tuple(self.gemv.get(kind, DEFAULT_GEMV[kind]))


# The values measured on an RTX A4500 before this module existed; the fallback for unknown kinds.
DEFAULT_GEMV = {
    "single:ht": [64, 8, 4, 2], "single:rt": [32, 16, 4, 2], "single:lm6": [32, 16, 4, 2], "single:lm8": [16, 32, 4, 2],
    "single:nvfp4": [64, 8, 4, 2], "moe": [64, 8, 4, 2], "dense": [32, 8, 4, 2],
}


def l1_fit_fraction(tables: list[dict] | None = None) -> tuple[float, str]:
    """The largest lookup table (as a fraction of the SM's L1) that keeps >= 60% of small-table lookup speed, from
    the measured lookup curves; the most conservative card wins. 0.5 when nothing is measured."""
    tables = _load_tables() if tables is None else tables
    fits = []
    for e in tables:
        curve = e.get("detail", {}).get("lookup_curve_kb_gps")
        if not curve:
            continue
        plateau = sorted(g for kb, g in curve if 8 <= kb <= 64)[len([1 for kb, _ in curve if 8 <= kb <= 64]) // 2]
        ok = 0
        for kb, g in sorted(curve):
            if g < 0.6 * plateau:
                break
            ok = kb
        fits.append((ok / (e["gpu"]["l1_per_sm"] / 1024), e["gpu"]["key"], ok))
    if not fits:
        return 0.5, "no lookup curves measured: half the L1"
    f, key, kb = min(fits)
    return f, f"tables up to {f:.2f} x L1 (tightest measured: {key}, fast up to {kb} KB)"


def rule_based(info: GpuInfo, tables: list[dict] | None = None) -> Tuning:
    """Choices from the card's limits alone.

    Two-weight tables: used when the table fits the fraction of the SM's L1 that the measured cards' lookup curves
    stay fast in (`l1_fit_fraction`). Attention split: the A4500's 16 splits x 8 KV heads over 56 SMs, scaled to
    the card's SM count.
    """
    frac, why = l1_fit_fraction(tables)
    t = Tuning(gemv=dict(DEFAULT_GEMV), source=f"rules for sm{info.cc[0]}{info.cc[1]}, {info.l1_per_sm // 1024} KB L1, {info.sms} SMs; {why}")
    t.multitable = {f: 2 for f, b in MULTITABLE_BYTES.items() if b <= frac * info.l1_per_sm}
    t.attn_nsplit = scaled_nsplit(16, 56, info.sms)
    return t


def scaled_nsplit(nsplit: int, from_sms: int, to_sms: int) -> int:
    """A power of two near nsplit * to_sms / from_sms, in [4, 64]."""
    want = nsplit * to_sms / from_sms
    return int(min(64, max(4, 2 ** round(__import__("math").log2(max(want, 1))))))


def _load_tables() -> list[dict]:
    if not TABLE_DIR.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(TABLE_DIR.glob("*.json"))]


def _from_table(entry: dict, info: GpuInfo, how: str) -> Tuning:
    prm = entry["params"]
    t = Tuning(gemv={**DEFAULT_GEMV, **prm.get("gemv", {})}, multitable=dict(prm.get("multitable", {})),
               attn_nsplit=prm.get("attn_nsplit", 16), attn_block_k=prm.get("attn_block_k", 32),
               prefill=prm.get("prefill", [64, 64]), source=how)
    src_sms = entry["gpu"]["sms"]
    if src_sms != info.sms:
        t.attn_nsplit = scaled_nsplit(t.attn_nsplit, src_sms, info.sms)
    src_l1 = entry["gpu"]["l1_per_sm"]
    if src_l1 != info.l1_per_sm:  # a table measured with a different L1 says nothing about which tables fit here
        t.multitable = rule_based(info).multitable
        t.source += f"; two-weight tables by the L1 rule ({info.l1_per_sm // 1024} KB L1, measured card has {src_l1 // 1024} KB)"
    return t


def resolve(info: GpuInfo, tables: list[dict] | None = None) -> Tuning:
    """The tuning for a card described by `info` (it need not be present: tuning_coverage.py uses this)."""
    tables = _load_tables() if tables is None else tables
    for e in tables:
        if e["gpu"]["key"] == info.key:
            return _from_table(e, info, f"measured on this card ({e['gpu']['key']}, {e.get('measured', '?')})")
    same_arch = [e for e in tables if tuple(e["gpu"]["cc"]) == info.cc]
    if same_arch:
        e = min(same_arch, key=lambda e: abs(e["gpu"]["sms"] - info.sms))
        return _from_table(e, info, f"nearest measured sm{info.cc[0]}{info.cc[1]} card {e['gpu']['key']} ({e['gpu']['sms']} SMs, this card {info.sms})")
    if tables:  # no measured card of this architecture: block shapes from the closest SM count, tables by rule
        e = min(tables, key=lambda e: abs(e["gpu"]["sms"] - info.sms))
        t = _from_table(e, info, f"cross-architecture guess: block shapes from {e['gpu']['key']} ({e['gpu']['sms']} SMs, this card {info.sms}, sm{info.cc[0]}{info.cc[1]})")
        t.multitable = rule_based(info).multitable
        return t
    return rule_based(info)


@lru_cache(maxsize=16)
def _tuning_for(index: int) -> Tuning:
    info = gpu_info(index)
    forced = os.environ.get("LAMQ_TUNING")
    if forced:
        e = json.loads(Path(forced).read_text())
        return _from_table(e, info, f"forced by LAMQ_TUNING={forced}")
    return resolve(info)


_OVERRIDE: Tuning | None = None


def set_override(t: Tuning | None) -> None:
    """Use `t` for every device until reset with None (tune.py tries candidates this way). Launchers built
    before the call keep the settings they were built with."""
    global _OVERRIDE
    _OVERRIDE = t


def tuning_for(device: torch.device | int | str | None = None) -> Tuning:
    if _OVERRIDE is not None:
        return _OVERRIDE
    if device is None:
        index = torch.cuda.current_device()
    else:
        d = torch.device(device) if not isinstance(device, int) else torch.device("cuda", device)
        index = d.index if d.index is not None else torch.cuda.current_device()
    return _tuning_for(index)


def describe(device=None) -> str:
    info, t = gpu_info(device), tuning_for(device)
    return f"{info.name} ({info.key}): {t.source}\n  {json.dumps({k: v for k, v in asdict(t).items() if k != 'source'})}"


if __name__ == "__main__":
    for i in range(torch.cuda.device_count()):
        print(describe(i))
