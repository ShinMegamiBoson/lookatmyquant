"""Prompt and decode speed, and memory, on this Mac: an export (this runtime) or an mlx_lm model.

    python bench.py lamq path/to/export
    python bench.py mlx path/to/mlx-model
"""
import json
import sys

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm import stream_generate

import model as lm

kind, path = sys.argv[1], sys.argv[2]
n_tok = int(sys.argv[3]) if len(sys.argv) > 3 else 256
if kind.startswith("lamq"):
    M, tok = lm.load(path)
    if kind == "lamq":  # lamq-layers: mlx_lm's layers, without the fused single-token path
        import fastpath
        fastpath.install(M)
else:  # strict=False: some public MLX files carry key and value weights for layers that reuse earlier ones, which mlx_lm does not use
    from pathlib import Path
    from mlx_lm.utils import load_model, load_tokenizer
    M, _ = load_model(Path(path), strict=False)
    tok = load_tokenizer(Path(path))
mx.eval(M.parameters())
weights = mx.get_active_memory() / 1e9
prompt = tok.apply_chat_template([{"role": "user", "content": "Write a long, detailed essay on the history of the printing press, from Gutenberg to the present day."}],
                                 add_generation_prompt=True)
for r in stream_generate(M, tok, prompt, max_tokens=32):  # warm-up: kernels compile on first use
    pass
runs = []
for _ in range(3):
    mx.reset_peak_memory()
    text = ""
    for r in stream_generate(M, tok, prompt, max_tokens=n_tok):
        text += r.text
        last = r
    runs.append({"prompt tok/s": round(last.prompt_tps, 1), "decode tok/s": round(last.generation_tps, 1), "tokens": last.generation_tokens,
                 "peak GB": round(mx.get_peak_memory() / 1e9, 3)})
print(json.dumps({"kind": kind, "path": path, "memory after load GB": round(weights, 3), "runs": runs, "sample": text[:300]}, indent=1))
