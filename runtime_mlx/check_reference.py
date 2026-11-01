"""Runs an export on its reference window and compares with the quantizer's own outputs (reference.safetensors)."""
import sys
import mlx.core as mx
import model as lm

path = sys.argv[1]
M, _ = lm.load(path)
ref = mx.load(f"{path}/reference.safetensors")
ids, want = ref["ids"].astype(mx.int32), ref["logits"].astype(mx.float32)
got = M(ids).astype(mx.float32)
lp, lq = want - mx.logsumexp(want, -1, keepdims=True), got - mx.logsumexp(got, -1, keepdims=True)
kl = (mx.exp(lp) * (lp - lq)).sum(-1)
print(f"{ids.shape[1]} tokens: KL(quantizer ‖ runtime) mean {kl.mean().item():.2e}, max {kl.max().item():.2e}; "
      f"same top token {(want.argmax(-1) == got.argmax(-1)).mean().item():.3f}")
