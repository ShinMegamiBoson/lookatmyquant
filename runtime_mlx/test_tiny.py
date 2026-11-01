"""The runtime must reproduce the model the quantizer evaluated: compare it with the same architecture loaded with the
dequantized weights, on a prompt (prefill) and on single-token steps (decode)."""
import json, sys
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx_lm.utils import _get_classes
from mlx_lm.models.cache import make_prompt_cache
import model as lm

exp, dense_file = sys.argv[1], sys.argv[2]
A, _ = lm.load(exp)
config = json.loads((Path(exp) / "config.json").read_text())
mc, ac = _get_classes(config=config)
B = mc(ac.from_dict(config))
B.load_weights(list(B.sanitize(mx.load(dense_file)).items()), strict=False)
B.eval()
ids = mx.array(np.random.default_rng(0).integers(0, 262144, size=(1, 24)))
la, lb = A(ids).astype(mx.float32), B(ids).astype(mx.float32)
pa, pb = mx.softmax(la, -1), mx.softmax(lb, -1)
kl = (pb * (mx.log(pb + 1e-12) - mx.log(pa + 1e-12))).sum(-1).mean()
print(f"prefill: max |logit diff| {mx.abs(la - lb).max().item():.4f} (logit scale {mx.abs(lb).max().item():.2f}), KL {kl.item():.2e}, same argmax {(la.argmax(-1) == lb.argmax(-1)).mean().item():.3f}")
ca, cb = make_prompt_cache(A), make_prompt_cache(B)
A(ids[:, :20], cache=ca); B(ids[:, :20], cache=cb)
diffs = []
for t in range(20, 24):
    a1, b1 = A(ids[:, t : t + 1], cache=ca).astype(mx.float32), B(ids[:, t : t + 1], cache=cb).astype(mx.float32)
    diffs.append(mx.abs(a1 - b1).max().item())
print("decode steps: max |logit diff|", [round(x, 4) for x in diffs])
