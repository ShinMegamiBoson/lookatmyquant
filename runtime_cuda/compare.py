"""The float32 reference against the standalone runtime, on the fixed prompts in reference_prompts.json.

    python -m runtime_cuda.compare models/Inkling-2-Mini-LookAtMyQuant-8.9GB --save cache/oracle
    python -m runtime_cuda.compare models/Inkling-2-Mini-LookAtMyQuant-11.2GB --ref-only --save cache/oracle/11.2GB

The reference is transformers' own Inkling2LoopForCausalLM with every packed matrix decoded to dense float32
(`inkling/reference.py`); the runtime is `inkling/model.py`. Per prompt: the largest logit difference, KL(reference ‖
runtime) and top-1 agreement. Bitwise agreement is not possible: routing near-ties cascade from 1e-7 noise, so two
correct implementations differ by KL ~1e-4..1e-3 on normal text (~1e-2 on the 25-token chat prompt). `--save` writes
the reference logits and its routes (`reference_path.pt`), the oracle test_model_oracle.py checks the runtime against.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

PROMPTS = Path(__file__).with_name("reference_prompts.json")


def route_recorder(model):
    routes: list[torch.Tensor] = []

    def hook(mod, inp, out):
        routes.append(out[2].detach().sort(dim=-1).values.cpu())

    hs = [m.register_forward_hook(hook) for n, m in model.named_modules() if n.endswith("mlp.gate")]
    return routes, hs


@torch.no_grad()
def run(model, ids, device):
    routes, hs = route_recorder(model)
    logits = model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0].float().cpu()
    for h in hs:
        h.remove()
    return logits, routes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packed")
    ap.add_argument("--ref-device", default="cuda:0")
    ap.add_argument("--runtime-device", default="cuda:1")
    ap.add_argument("--save", default=None)
    ap.add_argument("--ref-only", action="store_true", help="only run the reference path and save it (for --save)")
    args = ap.parse_args()
    from .inkling.model import Inkling2LoopPacked
    from .inkling.reference import load_reference

    prompts = json.loads(PROMPTS.read_text())
    ref, _ = load_reference(args.packed, args.ref_device)
    fast = None if args.ref_only else Inkling2LoopPacked(args.packed, args.runtime_device)
    oracle = {}
    for p in prompts:
        lr, rr = run(ref, p["ids"], args.ref_device)
        if fast is not None:
            lf = fast.prefill(p["ids"], fast.make_cache(1024), all_logits=True).float().cpu()
            lpr, lpf = F.log_softmax(lr, -1), F.log_softmax(lf, -1)
            kl = (lpr.exp() * (lpr - lpf)).sum(-1)
            print(
                f"{p['name']:10s} T={len(p['ids']):3d}  max|dlogit| {(lr - lf).abs().max():.4f}  KL mean {kl.mean():.2e} max {kl.max():.2e}  "
                f"top-1 {(lr.argmax(-1) == lf.argmax(-1)).float().mean():.3f}",
                flush=True,
            )
        else:
            print(f"{p['name']:10s} T={len(p['ids']):3d}  reference top-1 tokens {lr.argmax(-1)[:8].tolist()}", flush=True)
        oracle[p["name"]] = {"ids": p["ids"], "logits": lr, "routes": torch.stack(rr)}
    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
        torch.save(oracle, Path(args.save) / "reference_path.pt")
        print("saved", Path(args.save) / "reference_path.pt")


if __name__ == "__main__":
    main()
