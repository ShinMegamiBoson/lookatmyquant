"""A full-size Gemma 4 E2B with random codes at a plan's bit widths, for timing the runtime before a real export exists.
Speed does not depend on the code values, only on the shapes, the bit widths and the kernels.

    python synth.py <plan.json> <mlx model dir for config/tokenizer> [both|in] [tokens]
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import model as lm


def random_model(plan_path, cfg_dir, rot="both"):
    from mlx_lm.utils import _get_classes
    plan = json.loads(Path(plan_path).read_text())
    config = json.loads((Path(cfg_dir) / "config.json").read_text())
    config.pop("quantization", None); config.pop("quantization_config", None)
    config.get("text_config", {}).pop("quantization", None)
    mc, ac = _get_classes(config=config)
    M = mc(ac.from_dict(config))
    M.set_dtype(mx.bfloat16)  # before the swap: set_dtype would also cast the coded modules' fp16 scales and fp32 levels
    tm = M.language_model.model
    rng = np.random.default_rng(0)
    signs = lambda n: mx.array(rng.choice([-1, 1], n).astype(np.int8))
    tvals = lambda: mx.array(rng.standard_normal(4096).astype(np.float32))
    for name, prog in plan["linears"].items():
        if prog[:2] not in ("lm", "tr"):
            continue
        bits = int(prog[2:].rstrip("x"))
        i, path = name.split(".", 1)
        layer = tm.layers[int(i)]
        obj = layer
        for p in path.split("."):
            obj = getattr(obj, p)
        m, n = obj.weight.shape
        codes = mx.random.randint(0, 2 ** 31, (m, n // 32 * bits)).astype(mx.uint32)
        scales = mx.full((m,), 0.02, mx.float16)
        lm._set(layer, path, lm.LMLinear(codes, scales, signs(n), signs(m) if rot == "both" else None, bits, tvals() if prog[:2] == "tr" else None))
    for key, attr, block in (("embed", "embed_tokens", None), ("ple", "embed_tokens_per_layer", tm.config.hidden_size_per_layer_input)):
        V, d = getattr(tm, attr).weight.shape
        width = block or d
        slices = d // width
        perm = rng.permutation(V).astype(np.int32)
        W, buckets, start, trellis = {}, [], 0, {}
        for k, (size, prog) in enumerate(zip(plan[key]["sizes"], plan[key]["programs"])):
            rows = perm[start : start + size]; start += size
            W[f"t.b{k}.rows"] = mx.array(rows)
            if prog[:2] not in ("lm", "tr"):  # a 16-bit bucket
                W[f"t.b{k}.dense"] = mx.random.normal((size, d)).astype(mx.bfloat16)
                buckets.append({"kind": "dense"})
                continue
            bits = int(prog[2:])
            tr = prog[:2] == "tr"
            W[f"t.b{k}.codes"] = mx.random.randint(0, 2 ** 31, (size, slices * width // 32 * bits)).astype(mx.uint32)
            W[f"t.b{k}.scales"] = mx.full((size, slices), 0.02, mx.float16)
            W[f"t.b{k}.signs"] = mx.array(rng.choice([-1, 1], (slices, width)).astype(np.int8))
            buckets.append({"kind": "lm", "bits": bits, **({"family": "tr"} if tr else {})})
            if tr:
                trellis.setdefault(bits, tvals())
        setattr(tm, attr, lm.LMTable(W, "t", {"shape": [V, d], "block": block, "buckets": buckets}, trellis))
    M.eval()
    mx.eval(M.parameters())
    return M


def decode_speed(M, n_tok=128, prompt_len=64, reps=3):
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(M)
    ids = mx.array(np.random.default_rng(1).integers(0, 262144, (1, prompt_len)))
    mx.eval(M(ids, cache=cache))
    tok = mx.array([[5]])
    for _ in range(8):  # warm-up
        logits = M(tok, cache=cache); tok = logits[:, -1].argmax(-1)[:, None]; mx.eval(tok)
    best = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n_tok):  # a dependent chain, as in generation: each token waits for the last
            logits = M(tok, cache=cache)
            tok = logits[:, -1].argmax(-1)[:, None]
            mx.async_eval(tok)
        mx.eval(tok)
        best = max(best, n_tok / (time.perf_counter() - t0))
    return best


if __name__ == "__main__":
    plan, cfg = sys.argv[1], sys.argv[2]
    rot = sys.argv[3] if len(sys.argv) > 3 else "both"
    n_tok = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    t0 = time.perf_counter()
    M = random_model(plan, cfg, rot)
    print(f"built in {time.perf_counter() - t0:.1f}s, {mx.get_active_memory() / 1e9:.2f} GB")
    print(f"decode: {decode_speed(M, n_tok):.1f} tok/s  (rotations: {rot})")
