"""Measures the kernel settings for the GPU it runs on and writes runtime_cuda/inkling/tuning/<card>.json.

    python runtime_cuda/tune.py                 # full sweep (~10-20 min, mostly Triton compiles)
    python runtime_cuda/tune.py --quick         # fewer candidates
    python runtime_cuda/tune.py --out x.json    # write elsewhere (e.g. on a rented card, then copy back)

Needs only PyTorch and Triton: every launcher is timed at the Inkling-2-Mini shapes it runs in decode, on random codes
in the formats and mixes the released files use, so no model download is needed. Timing is triton.testing.do_bench
(median, L2 flushed between runs, as in decode where weights come from DRAM). Beside the chosen settings the file
records the card's limits, its measured DRAM bandwidth, lookup throughput against table size (what the two-weight
table choice depends on), and each launcher's time with the old defaults, so cards that are never rented can be
extrapolated from measured ones.
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from triton.testing import do_bench

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime_cuda.inkling import tuning as TU  # noqa: E402
from runtime_cuda.inkling.formats import FMT_PARAMS  # noqa: E402
from runtime_cuda.inkling.gemv import DenseStageGEMV, MoESideGEMV, packed_gemv  # noqa: E402

DEV = torch.device("cuda")
RNG = np.random.default_rng(0)


def ms(fn) -> float:
    return float(do_bench(fn, warmup=10, rep=60, return_mode="median"))


# ----------------------------------------------------------------------------- random packed matrices

def trellis(fmt, E, m, n):
    k, g, K, _ = FMT_PARAMS[fmt]
    codes = torch.from_numpy(RNG.integers(0, 2**32, size=(E, m, n * k // 32), dtype=np.uint32).view(np.int32)).to(DEV)
    lut = torch.from_numpy(RNG.standard_normal((K, g) if g > 1 else (K,)).astype(np.float16)).to(DEV)
    return codes, torch.rand(E, m, device=DEV).half(), lut


def lm(k, m, n):
    codes = torch.from_numpy(RNG.integers(0, 256, size=(m * n * k // 8,), dtype=np.uint8)).to(DEV)
    return codes, torch.rand(1, m, device=DEV).half(), torch.randn(1 << k, device=DEV)


def nvfp4(E, m, n):
    return (torch.from_numpy(RNG.integers(0, 256, size=(E, m, n // 2), dtype=np.uint8)).to(DEV),
            torch.from_numpy(RNG.integers(32, 88, size=(E, m, n // 16), dtype=np.uint8)).to(DEV),
            torch.rand(E, device=DEV) * 1e-3)


# ----------------------------------------------------------------------------- what each launcher runs in decode

def moe_case():
    """Both sides of one MoE layer, 8 routed experts, the 11.2 GB build's mix (ht2/ht3/rtr4 + NVFP4)."""
    sides = []
    for m, n, x_div in ((2304, 2304, 8), (2304, 1152, 1)):
        slabs, slab_of, loc_of = [], np.zeros(128, np.int32), np.zeros(128, np.int32)
        for j, fmt in enumerate(("ht2", "ht3", "rtr4")):
            c, s, l = trellis(fmt, 16, m, n)
            slabs.append((fmt, c, s, l))
        nv = nvfp4(16, m, n)
        for e in range(128):
            slab_of[e] = [0, 1, 2, 4][e % 4]
            loc_of[e] = (e // 4) % 16
        sides.append((slabs, torch.from_numpy(slab_of).to(DEV), torch.from_numpy(loc_of).to(DEV), m, n, x_div, nv))
    eids = torch.tensor([0, 5, 10, 15, 21, 38, 77, 126], dtype=torch.int32, device=DEV)

    def build(cfg):
        bm, bc, nw, ns = cfg
        ls = [MoESideGEMV(sl, so, lo, m, n, BLOCK_M=bm, BLOCK_C=bc, nvfp4=nv, num_warps=nw, num_stages=ns)
              for sl, so, lo, m, n, x_div, nv in sides]
        xs = [torch.randn(8 // x_div, n, device=DEV) for _, _, _, m, n, x_div, _ in sides]
        return lambda: [l(x, eids, sd[5], x_raw=x) for l, x, sd in zip(ls, xs, sides)]
    return build


def dense_case():
    """48 visits of the five attention-input matrices + 1 LM head (8.9 GB build's formats), time weighted 48:1."""
    in_mats = [("rtr4", 3072), ("lm6", 1024), ("lm8", 1024), ("lm8", 384), ("rtr4", 3072)]
    head = [("lm8", 1024), ("lm8", 3072), ("lm8", 4096), ("lm6", 8192), ("lm6", 16384), ("rtr4", 32768), ("rtr3", 65536), ("rtr2", 69952)]

    def mats(spec):
        out = []
        for fmt, m in spec:
            if fmt.startswith("lm"):
                c, s, t = lm(int(fmt[2:]), m, 2304)
                out.append((fmt, m, c, s, t))
            else:
                c, s, l = trellis(fmt, 1, m, 2304)
                out.append((fmt, m, c, s, l))
        return out
    ins, hd = mats(in_mats), mats(head)
    x_in, x_hd = torch.randn(5, 2304, device=DEV), torch.randn(8, 2304, device=DEV)

    def build(cfg):
        bm, bc, nw, ns = cfg
        a = DenseStageGEMV(ins, 2304, list(range(5)), BLOCK_M=bm, BLOCK_C=bc, num_warps=nw, num_stages=ns)
        b = DenseStageGEMV(hd, 2304, list(range(8)), BLOCK_M=bm, BLOCK_C=bc, num_warps=nw, num_stages=ns)
        return [(lambda: a(x_in), 48.0), (lambda: b(x_hd), 1.0)]
    return build


def single_case(fam):
    """packed_gemv on the o_proj shape (2304 x 3072), one format family."""
    fmt = {"ht": "ht3", "rt": "rtr4", "lm6": "lm6", "lm8": "lm8"}[fam]
    if fmt.startswith("lm"):
        c, s, t = lm(int(fmt[2:]), 2304, 3072)
    else:
        c, s, t = trellis(fmt, 1, 2304, 3072)
    x = torch.randn(1, 3072, device=DEV)
    return lambda cfg: (lambda: packed_gemv(x, c, s, fmt, 2304, 3072, t, BLOCK_M=cfg[0], BLOCK_C=cfg[1], num_warps=cfg[2], num_stages=cfg[3]))


def timed(build, cfg):
    try:
        f = build(cfg)
        if isinstance(f, list):
            return sum(w * ms(g) for g, w in f)
        return ms(f)
    except Exception:  # an invalid combination (too few elements for the warps, out of resources)
        return float("inf")


def sweep(name, build, candidates, default):
    base = timed(build, default)
    res = sorted((timed(build, c), c) for c in candidates)
    best_t, best = res[0]
    if best_t >= base:  # never pick something slower than the default
        best_t, best = base, default
    print(f"  {name:13s} default {base * 1e3:8.1f} us  best {best_t * 1e3:8.1f} us  {list(best)}  ({base / best_t:.2f}x)", flush=True)
    return list(best), {"default_us": round(base * 1e3, 2), "best_us": round(best_t * 1e3, 2),
                        "top5": [[round(t * 1e3, 2), list(c)] for t, c in res[:5]]}


# ----------------------------------------------------------------------------- card characteristics

def dram_gbs() -> float:
    a = torch.empty(512 * 2**20 // 4, device=DEV)
    b = torch.empty_like(a)
    t = ms(lambda: b.copy_(a))
    return round(2 * a.numel() * 4 / (t * 1e-3) / 1e9, 1)


@triton.jit
def _gather_kernel(TAB, OUT, N_ENT, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    h = i.to(tl.uint32) * 0x9E3779B1
    acc = tl.zeros((BLOCK,), tl.float32)
    for _ in range(ITERS):
        h = h * 1664525 + 1013904223
        acc += tl.load(TAB + ((h >> 8) % N_ENT.to(tl.uint32)).to(tl.int32)).to(tl.float32)
    tl.store(OUT + i, acc)


def lookup_curve() -> list[list[float]]:
    """Random 4-byte lookups per second against table size: where the L1 falls off."""
    out = []
    progs, block, iters = 8192, 256, 64
    o = torch.empty(progs * block, device=DEV)
    for kb in (4, 8, 16, 32, 64, 96, 128, 192, 256, 512, 2048):
        n = kb * 1024 // 4
        tab = torch.randn(n, device=DEV)
        t = ms(lambda: _gather_kernel[(progs,)](tab, o, n, ITERS=iters, BLOCK=block, num_warps=4))
        g = progs * block * iters / (t * 1e-3) / 1e9
        out.append([kb, round(g, 1)])
        print(f"  lookups, {kb:5d} KB table: {g:7.1f} G/s", flush=True)
    return out


def multitable_choice(quick: bool):
    """Per scalar-trellis format: single vs two-weight table on a 65536 x 2304 bucket (the LM head's shape)."""
    choice, detail = {}, {}
    x = torch.randn(1, 2304, device=DEV)
    for fmt in ("rtr1", "rtr2", "rtr3"):
        c, s, l = trellis(fmt, 1, 65536 if not quick else 32768, 2304)
        r = {}
        for g in (1, 2):
            TU.set_override(TU.Tuning(gemv=dict(TU.DEFAULT_GEMV), multitable={fmt: 2} if g == 2 else {}))
            best = min(timed(lambda cfg: (lambda: packed_gemv(x, c, s, fmt, c.shape[1], 2304, l, BLOCK_M=cfg[0], BLOCK_C=cfg[1], num_warps=cfg[2])), cfg)
                       for cfg in ((32, 16, 2, 2), (16, 16, 2, 2), (32, 8, 2, 2), (64, 8, 4, 2), (32, 16, 4, 2)))
            r[g] = best
        TU.set_override(None)
        if r[2] < r[1] * 0.97:
            choice[fmt] = 2
        detail[fmt] = {"single_us": round(r[1] * 1e3, 1), "two_weight_us": round(r[2] * 1e3, 1), "table_kb": TU.MULTITABLE_BYTES[fmt] // 1024}
        print(f"  {fmt}: single {r[1] * 1e3:7.1f} us  two-weight ({TU.MULTITABLE_BYTES[fmt] // 1024} KB) {r[2] * 1e3:7.1f} us", flush=True)
    return choice, detail


def attention_choice(quick: bool):
    from runtime_cuda.inkling.attention import decode_attention

    H, Hkv, D, cap = 24, 8, 128, 4096
    kv = torch.randn(2, Hkv, cap, D, device=DEV).half()
    q, rel, sink = torch.randn(H, D, device=DEV), torch.randn(H, 16, device=DEV), torch.randn(H, device=DEV)
    proj = torch.randn(16, 1024, device=DEV)
    res = {}
    for nsplit, bk in itertools.product((4, 8, 16, 32, 64), (16, 32, 64)):
        tot = 0.0
        for L, window in ((512, 1 << 30), (2048, 1 << 30), (2048, 1024)):  # full layers at two lengths, a sliding layer
            pos = torch.tensor([L - 1], dtype=torch.int32, device=DEV)
            tot += ms(lambda: decode_attention(q, kv, rel, sink, pos, window, 1024, block_k=bk, nsplit=nsplit, proj=proj))
        res[(nsplit, bk)] = tot
    best = min(res, key=res.get)
    print(f"  attention: default (16, 32) {res[(16, 32)] * 1e3:.1f} us, best {best} {res[best] * 1e3:.1f} us", flush=True)
    return best, {f"{k[0]},{k[1]}": round(v * 1e3, 2) for k, v in sorted(res.items(), key=lambda kv: kv[1])[:5]}


def prefill_choice(quick: bool):
    from runtime_cuda.inkling.triton_kernels import group_by_expert, packed_matmul

    c, s, l = trellis("ht2", 16, 2304, 2304)
    P = 512 * 8
    expert = torch.from_numpy(RNG.integers(0, 16, size=P)).to(DEV)
    be, bp = group_by_expert(expert, 16, 16)
    x = torch.randn(512, 2304, device=DEV)
    res = {}
    for bm, bn in itertools.product((32, 64, 128), (32, 64, 128)):
        res[(bm, bn)] = timed(lambda cfg: (lambda: packed_matmul(x, c, s, "ht2", 2304, 2304, lut=l, block_expert=be, block_pairs=bp,
                                                               n_pairs=P, x_div=8, BLOCK_M=cfg[0], BLOCK_N=cfg[1])), (bm, bn))
    best = min(res, key=res.get)
    print(f"  prefill matmul: default (64, 64) {res[(64, 64)] * 1e3:.0f} us, best {best} {res[best] * 1e3:.0f} us", flush=True)
    return list(best), {f"{k[0]},{k[1]}": round(v * 1e3, 1) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    info = TU.gpu_info(0)
    print(f"{info.name}: sm{info.cc[0]}{info.cc[1]}, {info.sms} SMs, {info.l1_per_sm // 1024} KB L1/SM, "
          f"{info.l2_bytes / 2**20:.0f} MB L2, {info.mem_gbs} GB/s theoretical", flush=True)
    rec = {"gpu": {**asdict(info), "key": info.key}, "measured": datetime.date.today().isoformat(),
           "versions": {"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
                        "driver": torch.cuda.get_device_properties(0).name, "python": platform.python_version()},
           "params": {}, "detail": {}}
    print("card:", flush=True)
    rec["detail"]["dram_gbs_measured"] = dram_gbs()
    print(f"  DRAM copy {rec['detail']['dram_gbs_measured']} GB/s", flush=True)
    rec["detail"]["lookup_curve_kb_gps"] = lookup_curve()

    print("two-weight tables:", flush=True)
    mt, det = multitable_choice(args.quick)
    rec["params"]["multitable"], rec["detail"]["multitable"] = mt, det
    TU.set_override(TU.Tuning(gemv=dict(TU.DEFAULT_GEMV), multitable=mt))  # the GEMV sweeps below use the chosen tables

    BM, BC = (16, 32, 64, 128), (4, 8, 16, 32)
    W, S = ((2, 4, 8), (1, 2, 3)) if not args.quick else ((2, 4), (2,))
    cands = [c for c in itertools.product(BM, BC, W, S) if c[0] * c[1] >= 32 * c[2]]
    print(f"decode GEMVs ({len(cands)} candidates each):", flush=True)
    gemv = {}
    for name, build in [("moe", moe_case()), ("dense", dense_case())] + [(f"single:{f}", single_case(f)) for f in ("ht", "rt", "lm6", "lm8")]:
        gemv[name], rec["detail"][name] = sweep(name, build, cands, tuple(TU.DEFAULT_GEMV[name]))
    gemv["single:nvfp4"] = gemv["moe"]
    rec["params"]["gemv"] = gemv
    TU.set_override(None)

    print("decode attention:", flush=True)
    (ns, bk), rec["detail"]["attention"] = attention_choice(args.quick)
    rec["params"]["attn_nsplit"], rec["params"]["attn_block_k"] = ns, bk
    print("prefill:", flush=True)
    rec["params"]["prefill"], rec["detail"]["prefill"] = prefill_choice(args.quick)

    out = Path(args.out) if args.out else TU.TABLE_DIR / f"{info.key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
