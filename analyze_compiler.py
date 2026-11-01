"""The compression compiler's results (results/COMPILER.md): every run of a plan (results/batch/cc*_<plan>.json) next to
the uniform programs, with the knapsack's predicted KL against the measured one.

    python analyze_compiler.py                # the table
    python analyze_compiler.py --sens         # what the sensitivity measurements say (by role, layer, bucket, cache)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
B = ROOT / "results" / "batch"
UNIFORM = {"smTr2x": "uniform tr2x", "smTr3x": "uniform tr3x", "smTr4x": "uniform tr4x", "smLm4x": "uniform lm4x"}


def runs():
    rows = []
    for job, label in UNIFORM.items():
        f = B / f"{job}.json"
        if f.exists():
            d = json.loads(f.read_text())
            rows.append({"run": label, "lin": d["bits per weight (quantized linears)"], "ppl": d["ppl"], "kl": d["kl"]})
    for f in sorted(glob.glob(str(B / "cc*_*.json"))):
        d = json.loads(Path(f).read_text())
        if "ppl" not in d:
            continue
        name = Path(f).stem
        bc = d.get("by ctx", {})
        rows.append({"run": name, "lin": d["bits per weight (quantized linears)"], "emb": d.get("embedding bits per weight"),
                     "kv": d.get("kv bits per value"), "ppl": d["ppl"], "kl": d["kl"],
                     "ppl2k": bc.get("2048", {}).get("ppl"), "kl2k": bc.get("2048", {}).get("kl"),
                     "pred": (d.get("plan") or {}).get("predicted kl"), "target": (d.get("plan") or {}).get("target")})
    return rows


def table():
    f = lambda v, p=3: "" if v is None else f"{v:.{p}f}"
    print(f"{'run':34s} {'linear b':>8s} {'embed b':>7s} {'cache b':>7s} {'ppl':>7s} {'KL':>7s} {'pred KL':>7s} {'ppl 2k':>7s} {'KL 2k':>7s}")
    for r in runs():
        print(f"{r['run']:34s} {f(r['lin']):>8s} {f(r.get('emb')):>7s} {f(r.get('kv')):>7s} {f(r['ppl']):>7s} {f(r['kl'], 4):>7s}"
              f" {f(r.get('pred'), 4):>7s} {f(r.get('ppl2k')):>7s} {f(r.get('kl2k'), 4):>7s}")


def sens():
    recs = [r for f in sorted(glob.glob(str(B / "ccSens[LE]*.json"))) for r in json.loads(Path(f).read_text())["records"]]
    lin = [r for r in recs if r["kind"] == "linear" and r["option"] != "fp16"]
    print(f"{len(recs)} records")
    for o in ("tr2", "tr3", "tr4", "lm5"):
        by = {}
        for r in lin:
            if r["option"] == o:
                by.setdefault(r["role"], []).append(r["dkl"])
        print(o, "summed KL by role:", {k: round(sum(v), 4) for k, v in sorted(by.items())})
    for o in ("tr3",):
        by = {}
        for r in lin:
            if r["option"] == o:
                by[r["layer"]] = by.get(r["layer"], 0) + r["dkl"]
        print(o, "KL by layer:", {k: round(v, 4) for k, v in sorted(by.items())})
    top = sorted((r for r in lin if r["option"] == "tr3"), key=lambda r: -r["dkl"])[:10]
    print("most sensitive at tr3:", [(r["item"], round(r["dkl"], 4)) for r in top])
    for r in [r for r in recs if r["kind"] == "embed" and r["option"] in ("tr2", "tr3")]:
        print(f"  {r['item']} {r['rows']} rows, {r['frequency mass']:.1%} of tokens, {r['option']}: {r['dkl']:.2e}")
    kv = {}
    for r in recs:
        if r["kind"] == "kv" and r["option"] == 4:
            kv[r["item"]] = r["dkl"]
    if kv:
        print("cache at 4 bits:", {k: round(v, 4) for k, v in kv.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", action="store_true")
    a = ap.parse_args()
    sens() if a.sens else table()
