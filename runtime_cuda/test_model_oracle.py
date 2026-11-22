"""The standalone model against the saved reference (transformers glue + float32 decode), at the model's noise floor.

Needs `cache/oracle/reference_path.pt` (python -m runtime_cuda.compare <packed> --save cache/oracle)
and the packed 8.9 GB file in models/. Bitwise agreement is impossible (routing near-ties cascade from 1e-7 noise),
so the bounds are the floor measured between two correct paths: mean KL well under 1e-2 on normal text, top-1 >= 0.95.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# (packed file, its oracle from `compare --ref-only --save`); a pair is skipped while either is missing
BUILDS = {
    "8.9GB": (ROOT / "models" / "Inkling-2-Mini-LookAtMyQuant-8.9GB", ROOT / "cache" / "oracle" / "reference_path.pt"),
    "11.2GB": (ROOT / "models" / "Inkling-2-Mini-LookAtMyQuant-11.2GB", ROOT / "cache" / "oracle" / "11.2GB" / "reference_path.pt"),
}
AVAILABLE = [b for b, (p, o) in BUILDS.items() if p.exists() and o.exists()]

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and AVAILABLE), reason="needs CUDA, a model and its oracle")


@pytest.fixture(scope="module", params=AVAILABLE)
def build(request):
    from runtime_cuda.inkling.model import Inkling2LoopPacked

    packed, oracle = BUILDS[request.param]
    m = Inkling2LoopPacked(packed, "cuda")
    yield m, torch.load(oracle)
    del m
    torch.cuda.empty_cache()


@pytest.fixture
def model(build):
    return build[0]


@pytest.fixture
def oracle(build):
    return build[1]


def kl_top1(ref: torch.Tensor, got: torch.Tensor):
    lr, lg = F.log_softmax(ref.float(), -1), F.log_softmax(got.float().cpu(), -1)
    return (lr.exp() * (lr - lg)).sum(-1), (ref.argmax(-1) == got.cpu().argmax(-1)).float()


NAMES = ["chat_math", "en", "code", "multi"]
KL_BOUND = {"chat_math": 5e-2, "en": 5e-3, "code": 5e-3, "multi": 1e-2}


@pytest.mark.parametrize("name", NAMES)
def test_prefill_matches_reference(model, oracle, name):
    o = oracle[name]
    cache = model.make_cache(1024)
    got = model.prefill(o["ids"], cache, all_logits=True)
    kl, top1 = kl_top1(o["logits"], got)
    print(f"prefill {name}: KL {kl.mean():.2e} (max {kl.max():.2e}) top-1 {top1.mean():.3f}")
    assert kl.mean() < KL_BOUND[name] and top1.mean() >= (0.8 if name == "chat_math" else 0.95)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("chunk", [1024, 7])
def test_decode_steps_match_prefill(model, oracle, name, chunk):
    """Prefill (SDPA attention, grouped experts) and token-by-token decode (Triton attention, one block a pair) are
    different code paths over the same cache; they must agree far more tightly than either agrees with the oracle."""
    ids, n_dec = oracle[name]["ids"], 5
    cache = model.make_cache(1024)
    full = model.prefill(ids, cache, all_logits=True)[-n_dec:]
    cache = model.make_cache(1024)
    model.prefill(ids[:-n_dec], cache, chunk=chunk)
    got = torch.cat([model.logits(model.decode_step(torch.tensor([t], device="cuda"), cache)) for t in ids[-n_dec:]], 0)
    kl, top1 = kl_top1(full.cpu(), got)
    print(f"decode vs prefill {name} (prefill chunk {chunk}): KL {kl.mean():.2e} (max {kl.max():.2e}) top-1 {top1.mean():.3f}")
    # chat_math (a 25-token system prompt) moves by ~2e-2 under 1e-7 noise; the others by ~1e-4. Measured: 8.9 GB
    # 1.1e-3; 11.2 GB 1.3e-2 / 7.3e-3, while en/code on the same NVFP4 decode path agree to ~1e-7 (a routing near-tie)
    assert kl.mean() < (3e-2 if name == "chat_math" else 2e-3) and top1.mean() == 1.0 and int(cache.pos.item()) == len(ids)
