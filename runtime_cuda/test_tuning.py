"""How a card's tuning is resolved: own table, nearest same-architecture table, cross-architecture guess, rules."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_cuda.inkling import tuning as TU  # noqa: E402


def card(name, cc, sms):
    return TU.GpuInfo(name, cc, sms, 0, 0, 0, TU.L1_PER_SM_KB[cc] * 1024, 0.0)


def table(info, moe, nsplit=32, multitable=None, curve=None):
    e = {"gpu": {"key": info.key, "cc": list(info.cc), "sms": info.sms, "l1_per_sm": info.l1_per_sm},
         "params": {"gemv": {"moe": moe}, "multitable": multitable or {"rtr1": 2, "rtr2": 2}, "attn_nsplit": nsplit, "attn_block_k": 16},
         "measured": "test"}
    if curve:
        e["detail"] = {"lookup_curve_kb_gps": curve}
    return e


A4500 = card("NVIDIA RTX A4500", (8, 6), 56)
L4 = card("NVIDIA L4", (8, 9), 58)
R4090 = card("NVIDIA GeForce RTX 4090", (8, 9), 128)
CURVE_128 = [[8, 330], [16, 300], [32, 310], [64, 305], [96, 305], [128, 110], [256, 58]]
TABLES = [table(A4500, [32, 16, 2, 1], curve=CURVE_128), table(L4, [16, 32, 2, 2], nsplit=32), table(R4090, [16, 16, 2, 1], nsplit=64)]


def test_own_table_wins():
    t = TU.resolve(L4, TABLES)
    assert t.source.startswith("measured") and t.gemv_cfg("moe") == (16, 32, 2, 2)


def test_nearest_same_architecture_by_sm_count():
    t = TU.resolve(card("NVIDIA GeForce RTX 4080", (8, 9), 76), TABLES)
    assert "nvidia-l4" in t.source and t.gemv_cfg("moe") == (16, 32, 2, 2)  # 76 is nearer 58 than 128
    t = TU.resolve(card("NVIDIA L40S", (8, 9), 142), TABLES)
    assert "rtx-4090" in t.source and t.attn_nsplit == 64  # 142 vs 128 SMs: same power of two


def test_attention_split_rescaled_by_sm_count():
    t = TU.resolve(card("NVIDIA GeForce RTX 3060", (8, 6), 28), TABLES)
    assert t.attn_nsplit == 16  # A4500's 32 at 56 SMs, halved


def test_cross_architecture_uses_rules_for_tables():
    t = TU.resolve(card("NVIDIA H100 80GB HBM3", (9, 0), 132), TABLES)
    assert t.source.startswith("cross-architecture") and "rtx-4090" in t.source
    assert t.multitable == {"rtr1": 2, "rtr2": 2, "rtr3": 2}  # 128 KB table <= 0.75 x 256 KB


def test_l1_fraction_from_curve_and_rules():
    frac, _ = TU.l1_fit_fraction(TABLES)
    assert frac == 0.75  # fast through 96 KB of a 128 KB L1
    assert TU.rule_based(card("NVIDIA A100-SXM4-80GB", (8, 0), 108), TABLES).multitable == {"rtr1": 2, "rtr2": 2, "rtr3": 2}
    assert TU.rule_based(card("x", (8, 6), 56), TABLES).multitable == {"rtr1": 2, "rtr2": 2}
    assert TU.rule_based(card("x", (8, 6), 56), []).multitable == {"rtr1": 2, "rtr2": 2}  # no curves: half the L1


def test_no_tables_is_rules():
    t = TU.resolve(A4500, [])
    assert t.source.startswith("rules") and t.gemv_cfg("moe") == tuple(TU.DEFAULT_GEMV["moe"])
