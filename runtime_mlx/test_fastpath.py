"""The fused single-token path must give the same logits as mlx_lm's layers (to rounding), on an export or on a random
full-size model, and this prints the decode speed of both.

    python test_fastpath.py export <dir>
    python test_fastpath.py synth <plan.json> <mlx model dir> [both|in]
"""
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

import fastpath
import model as lm
import synth

if sys.argv[1] == "export":
    M, _ = lm.load(sys.argv[2])
else:
    M = synth.random_model(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "in")
print("fast layers:", fastpath.install(M))
tm = M.language_model.model if hasattr(M, "language_model") else M.model
V = tm.embed_tokens.V
ids = mx.array(np.random.default_rng(0).integers(0, V, (1, 40)))


def run(on):
    tm._lamq_on = on
    cache = make_prompt_cache(M)
    M(ids[:, :24], cache=cache)
    return [M(ids[:, t : t + 1], cache=cache).astype(mx.float32)[0, -1] for t in range(24, 40)]


ref, fast = run(False), run(True)
for t, (a, b) in enumerate(zip(ref, fast)):
    pa, pb = mx.softmax(a), mx.softmax(b)
    kl = (pa * (mx.log(pa + 1e-12) - mx.log(pb + 1e-12))).sum().item()
    if t % 4 == 0 or t == 15:
        print(f"step {t:2d}: max |logit diff| {mx.abs(a - b).max().item():.4f} of {mx.abs(a).max().item():.2f}, KL {kl:.2e}, same top {a.argmax().item() == b.argmax().item()}")
for on in (False, True):
    tm._lamq_on = on
    print(f"{'fused' if on else 'mlx_lm layers'}: {synth.decode_speed(M, 128):.1f} tok/s")
