"""How unequal are a matrix's rows, for digits? On the original model, over calibration windows (the original's long math answers),
each linear layer's output channel i gets an importance E[g_i²]·E[y_i²]: g the gradient at that output of the log likelihood of
tokens sampled from the model's own predictions at positions that predict a digit (the Fisher information there), y the output.
A relative rounding error δ in row i costs about δ²·importance_i. With rows' widths free, the best allocation's cost is set by the
geometric mean of the importances, uniform widths' (or rotating the rows, which averages them) by the arithmetic mean: half of
log2(AM/GM) is the bits per weight that per-row widths are worth at the same cost.

    python row_importance.py --out results/row_importance.json     (run_batch.py: lm2RowImportance)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import compiler_arch as A
from token_kinds import kinds_of
from math_eval import DEV


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/.cache/huggingface/models/Qwen3.5-4B")
    ap.add_argument("--windows", default="/root/.cache/huggingface/retrain/math_long_q35.pt")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--len", type=int, default=1024)
    ap.add_argument("--skip-calib", type=int, default=0, help="leave out the windows a build calibrates on: this many from the front of its order (quantize.py --calib-math)")
    ap.add_argument("--save", default="/root/.cache/huggingface/retrain/row_importance.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    path, layers = A.decoder_layers(model)
    mods = {f"{l}.{n}": m for l, L in enumerate(layers) for n, m in L.named_modules() if isinstance(m, torch.nn.Linear)}
    gg = {k: torch.zeros(m.out_features, dtype=torch.float64, device=DEV) for k, m in mods.items()}
    yy = {k: torch.zeros(m.out_features, dtype=torch.float64, device=DEV) for k, m in mods.items()}
    w = torch.load(args.windows)
    if args.skip_calib:
        w = w[torch.randperm(len(w), generator=torch.Generator().manual_seed(0))[args.skip_calib :]]
    w = w[:, : args.len][: args.n]
    gen = torch.Generator(device=DEV).manual_seed(0)

    def keep(key):
        def grad(g):  # returns nothing, so the gradient passes on unchanged
            gg[key].add_(g.double().pow(2).sum((0, 1)))

        def hook(m_, a_, o_):
            yy[key] += o_.detach().double().pow(2).sum((0, 1))
            o_.register_hook(grad)
        return hook

    hooks = [m.register_forward_hook(keep(k)) for k, m in mods.items()]
    n_digit = 0
    for i in range(len(w)):
        x = w[i : i + 1]
        dig = torch.zeros(x.shape, dtype=torch.bool)
        dig[:, :-1] = kinds_of(x, tok) <= 1  # positions whose next token is a digit
        dig = dig.to(DEV)
        emb = model.get_input_embeddings()(x.to(DEV)).detach().requires_grad_(True)
        lg = model(inputs_embeds=emb, use_cache=False).logits.float()
        with torch.no_grad():
            y = torch.multinomial(F.softmax(lg[0], -1), 1, generator=gen)[:, 0]
        lp = F.log_softmax(lg[0], -1).gather(1, y[:, None])[:, 0]
        (lp * dig[0]).sum().backward()
        n_digit += int(dig.sum())
        del lg, lp, emb
        if i % 6 == 5:
            log(f"  {i + 1}/{len(w)} windows, {n_digit} digit positions ({time.time() - t0:.0f}s)")
    for h in hooks:
        h.remove()
    res = {"windows": len(w), "length": args.len, "digit positions": n_digit, "matrices": {}}
    saved, all_imp = {}, []
    for k in mods:
        imp = (gg[k] * yy[k]).clamp_min(1e-30)
        am, gm = float(imp.mean()), float(imp.log().mean().exp())
        top = imp.sort(descending=True).values
        res["matrices"][k] = {"rows": len(imp), "AM/GM": am / gm, "bits per weight per-row widths are worth": 0.5 * math.log2(am / gm),
                              "share of importance in the top 10% of rows": float(top[: max(1, len(top) // 10)].sum() / top.sum())}
        saved[k] = imp.float().cpu()
        all_imp.append(imp / imp.mean())
    vals = list(res["matrices"].values())
    res["median AM/GM"] = sorted(v["AM/GM"] for v in vals)[len(vals) // 2]
    res["weight-averaged bits worth"] = sum(v["bits per weight per-row widths are worth"] * mods[k].weight.numel() for k, v in res["matrices"].items()) / sum(m.weight.numel() for m in mods.values())
    by_role = {}
    for k, v in res["matrices"].items():
        by_role.setdefault(k.split(".", 1)[1], []).append(v["AM/GM"])
    res["median AM/GM by role"] = {r: sorted(x)[len(x) // 2] for r, x in by_role.items()}
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.save)
    res["seconds"] = round(time.time() - t0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    log(json.dumps({k: res[k] for k in ("median AM/GM", "weight-averaged bits worth", "median AM/GM by role")}, indent=1))
    log(f"=> rowImportance done ({res['seconds']}s)")


if __name__ == "__main__":
    main()
