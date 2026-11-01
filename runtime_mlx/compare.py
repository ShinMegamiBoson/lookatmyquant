"""Decode and prompt speed of several models in one process, runs interleaved so that background load on the Mac hits
them alike. Each entry is kind:path with kind lamq (fused single-token path), lamq-layers or mlx.

    python compare.py mlx:/path/mlx-4bit lamq:/path/export [rounds] [tokens]
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import stream_generate

import model as lm


def load(spec):
    kind, path = spec.split(":", 1)
    if kind.startswith("lamq"):
        M, tok = lm.load(path)
        if kind == "lamq":
            import fastpath
            fastpath.install(M)
    else:  # strict=False: public MLX files carry key and value weights for layers that reuse earlier ones
        from mlx_lm.utils import load_model, load_tokenizer
        M, _ = load_model(Path(path), strict=False)
        tok = load_tokenizer(Path(path))
    mx.eval(M.parameters())
    return kind, Path(path).name, M, tok


specs = [a for a in sys.argv[1:] if ":" in a]
nums = [int(a) for a in sys.argv[1:] if ":" not in a]
rounds, n_tok = (nums + [3, 256])[:2] if len(nums) < 2 else nums[:2]
models = [load(s) for s in specs]
msgs = [{"role": "user", "content": "Write a long, detailed essay on the history of the printing press, from Gutenberg to the present day."}]
res = {f"{k} {n}": {"decode": [], "prompt": []} for k, n, _, _ in models}
for k, n, M, tok in models:  # warm-up: kernels compile on first use
    for _ in stream_generate(M, tok, tok.apply_chat_template(msgs, add_generation_prompt=True), max_tokens=16):
        pass
for _ in range(rounds):
    for k, n, M, tok in models:
        for r in stream_generate(M, tok, tok.apply_chat_template(msgs, add_generation_prompt=True), max_tokens=n_tok):
            last = r
        res[f"{k} {n}"]["decode"].append(round(last.generation_tps, 1))
        res[f"{k} {n}"]["prompt"].append(round(last.prompt_tps, 1))
for name, v in res.items():
    print(f"{name:34s} decode best {max(v['decode']):6.1f} tok/s  (runs {v['decode']})   prompt best {max(v['prompt']):7.1f} tok/s")
print(json.dumps(res))
