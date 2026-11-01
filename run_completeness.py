"""Completeness check: does each method's operator word reproduce the method?

    python run_completeness.py                 # every method, all 32 layers
    python run_completeness.py --smoke         # every method on the first 2 layers
    python run_completeness.py --only GPTQ     # methods whose name contains GPTQ
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from opgraph import data
from opgraph import methods as ops
from opgraph import model as M
from opgraph import reference as ref
from opgraph.model import LAYERS, LINEARS, evaluate, logits, rope
from opgraph.pipeline import awq_ref_final, awq_ref_layer, per_matrix, run_local, summarize

RESULTS = Path(__file__).resolve().parent / "results"


def local_methods(cos, sin):
    def pm(ref_fn, ops_fn, needs_H):
        return per_matrix(ref_fn, ops_fn, needs_H)

    out = []
    for b in (4, 3):
        out.append((f"RTN w{b} g64", "Split(index) → Split(scale) → Restrict(ℤ) → Split(offset) → Restrict(grid)", "none",
                    *pm(lambda W, H, b=b: ref.rtn(W, b, 64), lambda W, H, b=b: ops.rtn(W, b, 64), False), None, None))
    for b in (4, 3):
        out.append((f"GPTQ w{b} g64", "sweep[ Split(scale, offset) per group; Restrict(column) → Refit(rest) ] → RTN word", "sequential",
                    *pm(lambda W, H, b=b: ref.gptq(W, H, b, 64, blocksize=64), lambda W, H, b=b: ops.gptq(W, H, b, 64), True), None, None))
    for b in (4, 3):
        out.append((f"AWQ w{b} g64", "Gauge(diag, folded) → Restrict(box) → RTN word", "sequential",
                    awq_ref_layer(b, 64, cos, sin), lambda net, i, f, b=b: ops.awq_layer(net, i, f, b, 64, cos, sin),
                    awq_ref_final(b, 64), lambda net, b=b: ops.awq_final(net, b, 64)))
    out.append(("SVD keep 0.8", "Split(identity) → Split(SVD) → Truncate → Contract", "none",
                *pm(lambda W, H: ref.svd_llm(W, None, 0.8), lambda W, H: ops.svd(W, None, 0.8), False), None, None))
    out.append(("SVD-LLM keep 0.8", "Split(identity) → Gauge(whitening) → Split(SVD) → Truncate → Contract", "profile",
                *pm(lambda W, H: ref.svd_llm(W, H, 0.8), lambda W, H: ops.svd(W, H, 0.8), True), None, None))
    out.append(("Magnitude 50%", "Mask(|W|)", "none", *pm(lambda W, H: ref.magnitude(W, 0.5), lambda W, H: ops.magnitude(W, 0.5), False), None, None))
    out.append(("Wanda 50%", "Mask(|W|·√diag H)", "sequential", *pm(lambda W, H: ref.wanda(W, H, 0.5), lambda W, H: ops.wanda(W, H, 0.5), True), None, None))
    out.append(("SparseGPT 50%", "sweep[ Mask per block; Mask(column) → Refit(rest) ]", "sequential",
                *pm(lambda W, H: ref.sparsegpt(W, H, 0.5), lambda W, H: ops.sparsegpt(W, H, 0.5), True), None, None))
    return out


def slicegpt_result(model, calib, eval_ids, k, log):
    t0 = time.time()
    mr = ref.slicegpt(model, calib, k)
    log(f"  [SliceGPT] reference done ({time.time() - t0:.0f}s)")
    net = ops.slicegpt(model, calib, k)
    mo = net.compile()
    log(f"  [SliceGPT] frame done ({time.time() - t0:.0f}s)")
    diffs = []
    for a, b in zip(mr.layers, mo.layers):
        for n in LINEARS + ("attn_shortcut", "mlp_shortcut"):
            A, B = getattr(a, n), getattr(b, n)
            diffs.append(((A - B).norm() / A.norm()).item())
    for A, B in ((mr.embed, mo.embed), (mr.head, mo.head)):
        diffs.append(((A - B).norm() / A.norm()).item())
    base, frame = evaluate(mr, eval_ids, model), evaluate(mo, eval_ids, model)
    x = eval_ids[:2].to(model.device)
    n_orig = sum(getattr(L, n).numel() for L in model.layers for n in LINEARS)
    decoder_bits = sum(lin.rate_bits() for lin in net.linears.values()) + 16.0 * sum(J.numel() for J in net.junctions.values() if J is not None)
    params_orig = n_orig + model.embed.numel()
    params_new = decoder_bits / 16 + net.embed.numel() + net.head.numel()
    return {"method": f"SliceGPT {round(100 * (1 - k / M.HIDDEN))}%",
            "word": "Untie → Contract(norm gains) → per segment: Gauge(canonical, group O) → Truncate",
            "ppl_ref": base["ppl"], "kl_ref": base["kl"], "ppl_frame": frame["ppl"], "kl_frame": frame["kl"],
            "max_logit_gap": (M.logits(mr, x) - M.logits(mo, x)).abs().max().item(), "max_layer_rel_weight_diff": max(diffs),
            "arg_agreement": None, "final": None, "bits_per_weight": decoder_bits / n_orig,
            "whole_model_params_ratio": params_new / params_orig}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--out", default="completeness.json")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    log_path = RESULTS / ("smoke.log" if args.smoke else Path(args.out).stem + ".log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    torch.manual_seed(0)
    model = M.load()
    calib, eval_ids = data.calibration(), data.evaluation()
    cos, sin = rope(calib.shape[1], model.device)
    n_layers = 2 if args.smoke else LAYERS
    out_path = RESULTS / ("smoke.json" if args.smoke else args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() and not args.smoke else []
    done = {r["method"] for r in results}
    original = evaluate(model, eval_ids)
    log(f"original: ppl {original['ppl']:.4f}")

    def save(r):
        results.append(r)
        out_path.write_text(json.dumps(results, indent=1))
        log(f"  => {r['method']}: ppl ref {r['ppl_ref']:.4f} | frame {r['ppl_frame']:.4f} | max logit gap {r['max_logit_gap']:.2e} | "
            f"max rel weight diff {r['max_layer_rel_weight_diff']:.2e} | arg agreement {r['arg_agreement']} | final {r['final']} | "
            f"{r['bits_per_weight']:.2f} bits/weight")

    for name, word, mode, ref_layer, ops_layer, ref_final, ops_final in local_methods(cos, sin):
        if args.only not in name or name in done or (args.skip and args.skip in name):
            continue
        log(f"{name}  [{mode}]")
        mr, net, per_layer, final = run_local(name, model, calib, ref_layer, ops_layer, mode, ref_final, ops_final, log, n_layers)
        r = summarize(name, word, model, mr, net, per_layer, final, eval_ids)
        r["per_layer"] = per_layer
        save(r)
        del mr, net
        M.empty_cache()

    if "GPTQ" in args.only or not args.only:
        name = "GPTQ w3 g64, official blocksize 128 (reference only)"
        if name not in done:
            log(name)
            mr, net, per_layer, final = run_local(name, model, calib,
                                                  *per_matrix(lambda W, H: ref.gptq(W, H, 3, 64, blocksize=128), lambda W, H: ops.gptq(W, H, 3, 64), True),
                                                  "sequential", log=log, n_layers=n_layers)
            r = summarize(name, "as GPTQ w3 g64, but the reference uses the official default blocksize 128 > group size 64", model, mr, net, per_layer, final, eval_ids)
            r["per_layer"] = per_layer
            save(r)

    if "SliceGPT" in args.only or not (args.only or args.smoke):
        if not any(r["method"].startswith("SliceGPT") for r in results):
            log("SliceGPT 25%")
            save(slicegpt_result(model, calib, eval_ids, 720, log))
    log("done")


if __name__ == "__main__":
    main()
