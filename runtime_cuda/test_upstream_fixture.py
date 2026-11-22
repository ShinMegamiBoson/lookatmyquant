"""The author's own synthetic packed model (`runtime_mlx/test_inkling.py::build_mini_packed_dir`), run on CUDA.

The fixture is plain numpy except for `mx.array` wrapping and one `mx.save_safetensors` call, so this test lifts the
function's source out of the upstream file and runs it against a small numpy stand-in for `mlx.core`. When upstream
extends the fixture (a new format, tensor or option), this test exercises it against our loader without anyone
copying code - and the schema check makes anything we do not implement fail loudly here.

Checks, on a 6-layer / 10-visit / 8-expert random model: the file passes the schema; the standalone CUDA model and
the transformers reference agree at the noise floor; cached decode equals prefill. A variant converts some experts
of one layer to NVFP4 (format id 7, read with the unrotated activation) to cover the mixed MoE path end to end.
"""
from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
UPSTREAM_TEST = ROOT / "runtime_mlx" / "test_inkling.py"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
try:  # the float32 reference is transformers' own Inkling2Loop, in transformers' git main for now
    import transformers.models.inkling2_loop  # noqa: F401

    HAS_INKLING2LOOP = True
except ImportError:
    HAS_INKLING2LOOP = False


def _mx_standin():
    from safetensors.numpy import save_file

    mx = types.SimpleNamespace()
    mx.array = lambda a, dtype=None: np.asarray(a)
    mx.save_safetensors = lambda path, d: save_file({k: np.ascontiguousarray(v) for k, v in d.items()}, str(path))
    return mx


def load_upstream_fixture():
    """`build_mini_packed_dir` and its helpers from the upstream test file, bound to numpy instead of MLX."""
    from runtime_cuda.inkling.formats import FMT_PARAMS, ID_TO_FMT

    tree = ast.parse(UPSTREAM_TEST.read_text())
    wanted = {"build_mini_packed_dir", "_pack_u32_random"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    missing = wanted - {n.name for n in nodes}
    assert not missing, f"upstream test no longer defines {missing}: update this test (see SYNC.md)"
    mod = ast.Module(body=nodes, type_ignores=[])
    ns = {"np": np, "json": json, "Path": Path, "mx": _mx_standin(), "FMT_PARAMS": FMT_PARAMS, "ID_TO_FMT": ID_TO_FMT}
    exec(compile(mod, str(UPSTREAM_TEST), "exec"), ns)
    return ns["build_mini_packed_dir"]


def _add_model_type(d: Path):
    """Fill in what transformers' config needs and the fixture leaves out, and give it rel_extent ==
    sliding_window_size (runtime_mlx and transformers size sliding layers' position bias differently otherwise;
    the released files set both to 1024): the window stays 8, the bias tables are cut to 8 distances."""
    from safetensors.numpy import load_file, save_file

    cfg = json.loads((d / "config.json").read_text())
    w = cfg["text_config"]["sliding_window_size"]
    cfg["text_config"]["rel_extent"] = w
    local = set(cfg["text_config"]["local_layer_ids"])  # transformers reads layer_types (default: every 6th full)
    cfg["text_config"]["layer_types"] = ["hybrid_sliding" if i in local else "hybrid" for i in range(cfg["text_config"]["num_hidden_layers"])]
    st = load_file(str(d / "steering.safetensors"))
    st = {k: (np.ascontiguousarray(v[:, :w]) if k.endswith("relative_position_bias.proj") else v) for k, v in st.items()}
    save_file(st, str(d / "steering.safetensors"))
    cfg.setdefault("model_type", "inkling2_loop")  # AutoConfig needs it; the fixture only sets text_config's
    tc = cfg["text_config"]  # transformers reads separate sliding-window head shapes; the fixture leaves them out
    for swa, full in (("swa_num_attention_heads", "num_attention_heads"), ("swa_num_key_value_heads", "num_key_value_heads"), ("swa_head_dim", "head_dim")):
        tc.setdefault(swa, tc[full])
    # transformers' config validator ties the (unused) vision tower's width to the text width
    cfg.setdefault("vision_config", {"model_type": "inkling2_vision"})["text_hidden_size"] = cfg["text_config"]["hidden_size"]
    (d / "config.json").write_text(json.dumps(cfg, indent=2))


def _convert_experts_to_nvfp4(d: Path, layer: int, side: str, experts: list[int], seed: int = 7):
    """Rewrite one layer so `experts` of `side` are NVFP4 (fid 7), the rest unchanged."""
    p = d / f"layer_{layer:02d}_packed.npz"
    arrays = dict(np.load(p))
    meta = json.loads((d / f"layer_{layer:02d}_meta.json").read_text())
    rng = np.random.default_rng(seed)
    emap = arrays[f"{side}_map"].copy()
    m, n = {"gu": (2 * 256, 256), "dn": (256, 256)}[side]
    E = len(experts)
    arrays[f"{side}_nvfp4_codes"] = rng.integers(0, 256, size=(E, m, n // 2), dtype=np.uint8)
    arrays[f"{side}_nvfp4_scales"] = rng.integers(40, 64, size=(E, m, n // 16), dtype=np.uint8)
    # per-weight RMS ~ 0.01, like the fixture's trellis experts (E2M1 RMS ~2.9 x E4M3 scales ~0.9 x outer)
    arrays[f"{side}_nvfp4_outer"] = rng.uniform(3e-3, 5e-3, size=(E,)).astype(np.float32)
    for loc, e in enumerate(experts):
        emap[e] = [7, loc]
    arrays[f"{side}_map"] = emap
    meta.setdefault("experts", {}).setdefault(side, {})["nvfp4"] = E
    np.savez(p, **arrays)
    (d / f"layer_{layer:02d}_meta.json").write_text(json.dumps(meta, indent=2))


@pytest.fixture(scope="module", params=["upstream", "upstream+nvfp4"])
def mini_dir(request, tmp_path_factory):
    d = tmp_path_factory.mktemp(request.param.replace("+", "_"))
    load_upstream_fixture()(d)
    _add_model_type(d)
    if request.param.endswith("nvfp4"):
        _convert_experts_to_nvfp4(d, 2, "gu", [0, 3, 5])  # layer 2 is a loop layer
        _convert_experts_to_nvfp4(d, 3, "dn", [1, 2, 6, 7])
    return d


TOKS = [5, 19, 42, 128, 301, 7, 255, 12, 400, 88, 3, 77]


def test_schema_accepts_fixture(mini_dir):
    from runtime_cuda.inkling.schema import validate_packed_dir

    used = validate_packed_dir(mini_dir)
    assert used, used


@pytest.mark.skipif(not HAS_INKLING2LOOP, reason="needs transformers with Inkling2Loop (git main)")
def test_standalone_matches_reference(mini_dir):
    from runtime_cuda.inkling.model import Inkling2LoopPacked

    ref = _reference_model(mini_dir)
    with torch.no_grad():
        want = ref(input_ids=torch.tensor([TOKS], device="cuda"), use_cache=False).logits[0].float().cpu()
    m = Inkling2LoopPacked(mini_dir, "cuda")
    got = m.prefill(TOKS, m.make_cache(256), all_logits=True).float().cpu()
    lw, lg = F.log_softmax(want, -1), F.log_softmax(got, -1)
    kl = (lw.exp() * (lw - lg)).sum(-1)
    top1 = (want.argmax(-1) == got.argmax(-1)).float().mean().item()
    top2 = want.topk(2, -1).values
    gaps = (top2[:, 0] - top2[:, 1])[want.argmax(-1) != got.argmax(-1)]
    print(f"{mini_dir.name}: KL {kl.mean():.2e} (max {kl.max():.2e}) top-1 {top1:.3f}; top-2 gap where they differ {gaps.tolist()}")
    assert kl.mean() < 1e-3 and top1 >= 0.9


def test_decode_matches_prefill(mini_dir):
    from runtime_cuda.inkling.model import Inkling2LoopPacked

    m = Inkling2LoopPacked(mini_dir, "cuda")
    full = m.prefill(TOKS, m.make_cache(256), all_logits=True).float()
    cache = m.make_cache(256)
    m.prefill(TOKS[:6], cache)
    steps = torch.cat([m.logits(m.decode_step(torch.tensor([t], device="cuda"), cache)) for t in TOKS[6:]]).float()
    err = (steps - full[6:]).abs().max().item() / full.abs().max().item()
    print(f"{mini_dir.name}: decode vs prefill max rel diff {err:.2e}")
    assert err < 1e-3 and int(cache.pos.item()) == len(TOKS)


def _reference_model(d: Path):
    """The transformers reference path without a tokenizer (the fixture has none)."""
    import transformers

    from runtime_cuda.inkling.reference import load_reference

    real = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: None)
    try:
        model, _ = load_reference(d, "cuda")
    finally:
        transformers.AutoTokenizer.from_pretrained = real
    return model
