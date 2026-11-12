"""The trained drafter stored at 4 bits a weight, in the code and the packed format of the 1.2 GB build.

    python pack_drafter.py DRAFTER_DIR OUT_DIR [--bits 4]

Every matrix of a million weights or more is coded as quantize.py codes a row group of the build: the matrix's
input is rotated by a randomized Hadamard transform, each row gets a scale (its RMS), and each row becomes a path
through the 12-bit trellis at --bits bits a weight, read against the table fitted once for that width
(opgraph.codes.ldlq_words with RowTrellisCode). No calibration statistics are used: a row takes the path nearest it.
The drafter's token table (a row a token) is coded the same way, row by row.

OUT_DIR gets the packed file (artifact.json and weights.bin: compiler_artifact.export_saved_codes), with every other
tensor as it was and config.json beside it, and pack_report.json: each matrix's seconds and the share of its squared
weight the code leaves out. runtime_mlx/drafter_q35.py loads either form of the drafter.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

import compiler_artifact
from opgraph import codes as C
from quantize import trellis_code_entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drafter")
    ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--min-weights", type=int, default=1 << 20, help="smaller tensors are kept as they are")
    args = ap.parse_args()
    state = load_file(str(Path(args.drafter) / "model.safetensors"))
    code = C.RowTrellisCode(12, args.bits)
    entries, report, t0 = {}, [], time.time()
    for i, name in enumerate(sorted(state)):
        W = state[name]
        if W.ndim != 2 or not W.is_floating_point() or W.numel() < args.min_weights:
            continue
        t, seed, W = time.time(), 1000 + i, W.double()
        net = C.ldlq_words([W], None, code, [seed], in_seed=seed, out_rot=False)[0]
        left = float((net.matrix(dtype=torch.float64) - W).pow(2).sum() / W.pow(2).sum())
        entries[name] = trellis_code_entry(net, W.shape, args.bits, seed)
        report.append({"name": name, "shape": list(W.shape), "seconds": round(time.time() - t, 1), "squared weight left out": left})
        print(f"{name:44s} {str(tuple(W.shape)):16s} {report[-1]['seconds']:6.1f} s  left out {left:.5f}", flush=True)
    written = compiler_artifact.export_saved_codes({"matrices": entries, "model": "DSpine drafter"}, args.out, dense_tensors=state, model_dir=args.drafter)
    coded = sum(r["shape"][0] * r["shape"][1] for r in report)
    summary = {"bits": args.bits, "matrices coded": len(report), "weights coded": coded, "weights in all": sum(v.numel() for v in state.values()),
               "bytes before": (Path(args.drafter) / "model.safetensors").stat().st_size, "bytes after": written["payload_bytes"] + written["manifest_bytes"],
               "squared weight left out, over the coded matrices' weights": sum(r["squared weight left out"] * r["shape"][0] * r["shape"][1] for r in report) / coded,
               "seconds": round(time.time() - t0, 1)}
    (Path(args.out) / "pack_report.json").write_text(json.dumps({"summary": summary, "matrices": report}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
