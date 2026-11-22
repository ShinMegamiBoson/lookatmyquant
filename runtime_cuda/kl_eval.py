"""KL(original ‖ packed) per token on English and code windows, the original on one GPU and the packed file on another.

    python -m runtime_cuda.kl_eval models/Inkling-2-Mini-LookAtMyQuant-8.9GB models/Inkling-2-Mini \
        --windows 4 --tokens 512

The LookAtMyQuant README reports KL 0.0208 for this file on its own en/chat/code text mix, which is not public;
English prose and Python code here are a stand-in, so expect the same order of magnitude, not the same number.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def windows_en(tok, n: int, T: int) -> list[torch.Tensor]:
    from datasets import load_dataset

    text = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    step = len(ids) // n
    return [ids[i * step : i * step + T] for i in range(n)]


def windows_code(tok, n: int, T: int) -> list[torch.Tensor]:
    import transformers

    root = Path(transformers.__file__).parent
    files = sorted(root.glob("models/*/modeling_*.py"))[:: max(1, len(sorted(root.glob("models/*/modeling_*.py"))) // n)]
    out = []
    for f in files:
        ids = tok(f.read_text(), return_tensors="pt").input_ids[0]
        if len(ids) >= T + 2000:
            out.append(ids[2000 : 2000 + T])  # skip license headers
        if len(out) == n:
            break
    return out


@torch.no_grad()
def logprobs_chunks(model, ids: torch.Tensor, chunk: int = 128):
    """Yields log-softmax chunks `[chunk, V]` (float32) of the full-sequence logits, without holding all of them.
    `model` is a transformers model (the original, or the float32 reference) or the standalone runtime."""
    if hasattr(model, "prefill"):  # the standalone runtime
        logits = model.prefill(ids.tolist(), model.make_cache(max(1024, len(ids))), all_logits=True)
        for s in range(0, logits.shape[0], chunk):
            yield F.log_softmax(logits[s : s + chunk].float(), dim=-1)
        return
    dev = next(p for p in model.parameters() if not p.is_meta).device
    h = model.model(input_ids=ids[None].to(dev), use_cache=False).last_hidden_state[0]
    for s in range(0, h.shape[0], chunk):
        logits = model.logit_softcap(model.lm_head(h[s : s + chunk]).float())
        yield F.log_softmax(logits.float(), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packed")
    ap.add_argument("teacher")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--packed-device", default="cuda:0")
    ap.add_argument("--teacher-device", default="cuda:1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reference", action="store_true", help="score the float32 transformers reference instead of the runtime")
    args = ap.parse_args()

    from .inkling.model import Inkling2LoopPacked
    from .inkling.reference import load_reference
    from .inkling.teacher import load_teacher

    teacher, tok = load_teacher(args.teacher, args.teacher_device)
    if hasattr(teacher.model, "vision_tower"):
        teacher.model.vision_tower = None
        torch.cuda.empty_cache()
    packed = load_reference(args.packed, args.packed_device)[0] if args.reference else Inkling2LoopPacked(args.packed, args.packed_device)

    sets = {"en": windows_en(tok, args.windows, args.tokens), "code": windows_code(tok, args.windows, args.tokens)}
    results = {}
    for name, wins in sets.items():
        kls, agree, n = [], 0, 0
        for w, ids in enumerate(wins):
            t0 = time.time()
            for lp_t, lp_p in zip(logprobs_chunks(teacher, ids), logprobs_chunks(packed, ids)):
                lp_p = lp_p.to(lp_t.device)
                kl = (lp_t.exp() * (lp_t - lp_p)).sum(-1)
                kls.append(kl.cpu())
                agree += (lp_t.argmax(-1) == lp_p.argmax(-1)).sum().item()
                n += kl.numel()
            print(f"{name} window {w}: running KL {torch.cat(kls).mean():.4f} top-1 agree {agree / n:.3f} ({time.time() - t0:.0f}s)", flush=True)
        kl_all = torch.cat(kls)
        results[name] = {"kl_mean": kl_all.mean().item(), "kl_median": kl_all.median().item(), "top1_agree": agree / n, "tokens": n}
        print(name, results[name], flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
