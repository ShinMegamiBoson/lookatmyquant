"""How far a model's next-token predictions are from the original's on text, for any model math_eval.py loads: KL(original ‖
model) over the whole vocabulary on WikiText-2's test split (64 windows of 512 tokens) and on held-out windows from other
sources: --sources mix takes the text mix's own evaluation windows, four a source (modal_fetch.py --text); a file gives
others ("windows", a tensor of token ids, and "records", each naming its window's "source"). The mean is taken per source
and then over sources.

    python text_kl.py --models orig,dense:build/dense.pt --sources mix --out results/text_kl.json     (run_batch.py: lm2TextKL)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import compiler_arch as A
from math_eval import apply_spec, log
from quantize import DEV, token_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/.cache/huggingface/models/Qwen3.5-4B")
    ap.add_argument("--models", required=True)
    ap.add_argument("--sources", default="mix")
    ap.add_argument("--wikitext-windows", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    base = lambda: AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float32).to(DEV).eval()
    ref = base()
    ids = token_ids(tok, "test")
    wins = ids[: len(ids) // 512 * 512].reshape(-1, 512)
    wins = wins[torch.randperm(len(wins), generator=torch.Generator().manual_seed(0))[: args.wikitext_windows]]
    if args.sources == "mix":  # the text mix's own evaluation windows, four a source (compiler_arch.TextMix)
        own = A.TextMix(tok, "qwen35").windows("eval", 4, 512, seed=3)
        held = {"windows": torch.cat(list(own.values())), "records": [{"source": s} for s, w in own.items() for _ in w]}
    else:
        held = torch.load(args.sources, weights_only=False)
    texts = [("wikitext", w) for w in wins] + [(r["source"], w.long()) for w, r in zip(held["windows"], held["records"])]
    res = {"device": DEV, "wikitext windows": len(wins), "source windows": len(held["windows"]), "tokens a window": 512, "models": {}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for spec in args.models.split(","):
        model, load = apply_spec(base(), spec, tok)
        by = defaultdict(list)
        with torch.no_grad():
            for source, w in texts:
                w = w.reshape(1, -1).to(DEV)
                r = ref(w, use_cache=False).logits[:, :-1].float().log_softmax(-1)
                o = model(w, use_cache=False).logits[:, :-1].float().log_softmax(-1)
                by[source].append(float((r.exp() * (r - o)).sum(-1).mean()))
                del r, o
        mean = {k: sum(v) / len(v) for k, v in by.items()}
        rest = {k: v for k, v in mean.items() if k != "wikitext"}
        res["models"][spec] = {"wikitext KL": mean["wikitext"], "13 sources KL": sum(rest.values()) / len(rest), "by source": rest, "load": load}
        log(f"{spec}: WikiText KL {mean['wikitext']:.4f}, 13 sources {res['models'][spec]['13 sources KL']:.4f} ({time.time() - t0:.0f}s)")
        res["seconds"] = round(time.time() - t0)
        out.write_text(json.dumps(res, indent=1))
        del model
        torch.cuda.empty_cache() if DEV == "cuda" else None
    log(f"=> textKL done ({res['seconds']}s)")


if __name__ == "__main__":
    main()
