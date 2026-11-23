"""Tests for the four compiler upgrades in release/lookatmyquant:
1. Dense all-token expert Hessians (`damped_hessian`, `dense_down_moments`, shared `_stage` Cholesky).
2. Chat-templated 6-class calibration & measurement mix (`TemplatedMix`, `_render_chat`).
3. Uniform-expert bucketing (`expert_buckets`) + per-class floor subtraction, prediction & minimax Lagrangian `allocate`.
4. Looped / non-sequential MoE support (`discover(..., counts_out=...)`, pass-indexed sigmoid/softmax `FixedRouting`, `price`).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compiler
import compiler_arch as A
from opgraph import codes as C


class _DummyTok:
    def __init__(self, with_template: bool = True, supports_tools: bool = True):
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.chat_template = "dummy" if with_template else None
        self._supports_tools = supports_tools

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        if tools is not None and not self._supports_tools:
            raise TypeError("template does not support tools=")
        out = []
        if tools:
            out.append("<|tools|>" + json.dumps([t["function"]["name"] for t in tools]))
        for m in messages:
            out.append(f"<|{m['role']}|>")
            if m.get("reasoning_content"):
                out.append(f"<|think|>{m['reasoning_content']}</|think|>")
            if m.get("content"):
                out.append(str(m["content"]))
            for tc in m.get("tool_calls") or ():
                out.append("<|call|>" + json.dumps(tc["function"]))
        return "\n".join(out)

    def __call__(self, docs, add_special_tokens=False):
        return SimpleNamespace(input_ids=[[10 + (b % 240) for b in s.encode("utf-8")] for s in docs])


class _ToyRouter(nn.Module):
    def __init__(self, hidden: int, num_experts: int, top_k: int, sigmoid: bool = True):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.randn(num_experts, hidden) * 0.2)
        self.norm_topk_prob = True
        if sigmoid:
            self.score_fn = torch.sigmoid
            self.routed_scaling_factor = 4.0

    def forward(self, x: torch.Tensor):
        flat = x.reshape(-1, x.shape[-1])
        logits = F.linear(flat, self.weight)
        if hasattr(self, "score_fn"):
            probs = self.score_fn(logits.float())
            top_v, top_i = torch.topk(probs, self.top_k, dim=-1)
            scores = (top_v / (top_v.sum(dim=-1, keepdim=True) + 1e-20)) * self.routed_scaling_factor
        else:
            probs = torch.softmax(logits.float(), dim=-1)
            top_v, top_i = torch.topk(probs, self.top_k, dim=-1)
            scores = top_v / top_v.sum(dim=-1, keepdim=True)
        return logits, scores.to(x.dtype), top_i


class _ToyExperts(nn.Module):
    def __init__(self, num_experts: int, hidden: int, inter: int):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * inter, hidden) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, inter) * 0.1)

    def _apply_gate(self, g: torch.Tensor) -> torch.Tensor:
        a, b = g.chunk(2, dim=-1)
        return F.silu(a) * b

    def forward(self, x: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        flat = x.reshape(-1, x.shape[-1])
        out = torch.zeros_like(flat)
        for e in range(self.gate_up_proj.shape[0]):
            mask = (top_k_index == e)
            if not mask.any():
                continue
            tok_idx, slot = mask.nonzero(as_tuple=True)
            h = self._apply_gate(F.linear(flat[tok_idx], self.gate_up_proj[e]))
            y = F.linear(h, self.down_proj[e]) * top_k_weights[tok_idx, slot].unsqueeze(-1)
            out.index_add_(0, tok_idx, y)
        return out.reshape_as(x)


class _ToyMoELayer(nn.Module):
    def __init__(self, hidden: int, num_experts: int, inter: int, sigmoid: bool = True):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Module()
        self.mlp.router = _ToyRouter(hidden, num_experts, top_k=2, sigmoid=sigmoid)
        self.mlp.experts = _ToyExperts(num_experts, hidden, inter)

    def forward(self, x: torch.Tensor):
        h = self.proj(x)
        _, scores, idx = self.mlp.router(h)
        return x + 0.1 * self.mlp.experts(h, idx, scores)


class _ToyLoopedMoEModel(nn.Module):
    def __init__(self, vocab: int = 256, hidden: int = 64, num_experts: int = 8, inter: int = 32, sigmoid: bool = True):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=4, hidden_size=hidden, num_experts=num_experts, model_type="toy_looped")
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_ToyMoELayer(hidden, num_experts, inter, sigmoid=sigmoid) for _ in range(4)])
        self.head = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        x = self.embed(input_ids)
        x = self.layers[0](x)
        for _ in range(3):
            x = self.layers[1](x)
            x = self.layers[2](x)
        x = self.layers[3](x)
        return SimpleNamespace(logits=self.head(x))


class ReleaseCompilerUpgradeTests(unittest.TestCase):
    def test_dense_down_moments_and_stage_cholesky_cache(self):
        torch.manual_seed(0)
        E, hidden, inter, N = 6, 32, 16, 100
        mod = _ToyExperts(E, hidden, inter)
        rmod = _ToyExperts(E, hidden, inter)
        X = torch.randn(N, hidden)
        Xr = X + 0.05 * torch.randn(N, hidden)

        def expert_hidden(m, e, xb, transposed):
            return m._apply_gate(F.linear(xb, m.gate_up_proj[e]))

        H_chunk, D_chunk = A.dense_down_moments(mod, 2, X, False, expert_hidden, chunk_size=19, rmod=rmod, Xr=Xr)
        hq_full = expert_hidden(mod, 2, X, False).float()
        hr_full = expert_hidden(rmod, 2, Xr, False).float()
        H_raw_full = (hq_full.T @ hq_full).double() / N
        self.assertLess((H_chunk - A.damped_hessian(H_raw_full)).abs().max().item(), 1e-6)
        self.assertLess((D_chunk - ((hr_full.T @ hq_full).double() / N - H_raw_full)).abs().max().item(), 1e-6)

        H_all = A.damped_hessian((X.T @ X) / N)
        C._STAGE.clear()
        code = C.TrellisCode(8, 2, T=16)
        orig_env = C.Environment
        env_calls = []

        def counting_env(*args, **kwargs):
            env_calls.append(1)
            return orig_env(*args, **kwargs)

        with mock.patch.object(C, "Environment", side_effect=counting_env):
            for e in range(E):
                C.ldlq_words([mod.gate_up_proj[e].detach().double()], H_all, code, [1500 + 2 * e], in_seed=1500)
        self.assertEqual(len(env_calls), 1)

    def test_templated_mix_and_minimax_allocator(self):
        raw_docs = {
            "en": [f"English text {i} about compilers and sparse mixture-of-experts. " * 6 for i in range(20)],
            "chat": [f"User question {i}\nAssistant detailed reply. " * 6 for i in range(20)],
            "code": [f"def f_{i}(a, b):\n    return a * {i + 2} + b\n" * 6 for i in range(20)],
            "fra_Latn": [f"Texte français {i} sur la quantification de modèles. " * 6 for i in range(15)],
        }
        mix = A.TemplatedMix(_DummyTok(True, True), "rel_tok", raw_docs=raw_docs)
        self.assertEqual(mix.sources, list(A.DOMAIN_CLASSES))
        win, manifest = mix.sample("calib", 12, 32, seed=3)
        self.assertEqual(win.shape, (12, 32))
        self.assertEqual({m["source"] for m in manifest[:6]}, set(A.DOMAIN_CLASSES))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            records = []
            specs = [
                ("0.mlp.gate_proj", 0, {"prose": 0.14, "code": 0.01}, {"prose": 0.03, "code": 0.002}),
                ("1.mlp.gate_proj", 1, {"prose": 0.12, "code": 0.01}, {"prose": 0.025, "code": 0.002}),
                ("2.mlp.gate_proj", 2, {"prose": 0.01, "code": 0.09}, {"prose": 0.002, "code": 0.015}),
                ("3.mlp.gate_proj", 3, {"prose": 0.01, "code": 0.08}, {"prose": 0.002, "code": 0.015}),
            ]
            for name, layer, c2, c3 in specs:
                base = {"item": name, "kind": "linear", "layer": layer, "role": "mlp.gate_proj", "weights": 1000, "shape": [40, 25], "T": 256}
                records.append(base | {"option": "tr2", "bits": 2000.0, "dkl": 0.005 + sum(c2.values()) / 2,
                                       "dkl_by_class": {k: v + 0.005 for k, v in c2.items()}})
                records.append(base | {"option": "tr3", "bits": 3000.0, "dkl": 0.005 + sum(c3.values()) / 2,
                                       "dkl_by_class": {k: v + 0.005 for k, v in c3.items()}})
                records.append(base | {"option": "lm8", "bits": 8000.0, "dkl": 0.005,
                                       "dkl_by_class": {"prose": 0.005, "code": 0.005}})
                records.append(base | {"option": "fp16", "bits": 16000.0, "dkl": 0.0,
                                       "dkl_by_class": {"prose": 0.0, "code": 0.0}})
            (root / "sens.json").write_text(json.dumps({"model": "toy", "trellis fit": "ema", "records": records}))
            out_mm = root / "plan_mm.json"
            compiler.allocate(SimpleNamespace(
                sens=str(root / "sens.json"), target="disk", target_override="", items="linear",
                bits=2.5, floor="lm8", match="", scale=1.0, budget=0.0, options="tr2,tr3",
                class_weights="", max_class_ratio=1.20, out=str(out_mm),
            ))
            plan = json.loads(out_mm.read_text())
            cls = plan["predicted kl by class"]
            ratio = max(cls.values()) / (sum(cls.values()) / len(cls))
            self.assertLessEqual(ratio, 1.25)

    def test_looped_moe_discover_and_fixed_routing(self):
        torch.manual_seed(7)
        model = _ToyLoopedMoEModel(sigmoid=True).eval()
        x = torch.randint(0, 256, (2, 8))
        counts: list[int] = []
        A.discover(model, x[:1], counts_out=counts)
        self.assertEqual(counts, [1, 3, 3, 1])

        fr = A.FixedRouting(model)
        with torch.no_grad():
            fr.record("k0")
            y0 = model(x).logits
            fr.off()
            fr.replay("k0")
            y1 = model(x).logits
            fr.off()
            self.assertLess((y0 - y1).abs().max().item(), 1e-6)
        fr.remove()

    def test_end_to_end_measure_and_allocate_on_looped_moe(self):
        torch.manual_seed(11)
        model = _ToyLoopedMoEModel(vocab=256, hidden=64, num_experts=8, inter=32, sigmoid=True).eval()
        tok = _DummyTok(with_template=True, supports_tools=True)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"model_type": "toy_looped"}))
            text_dir = root / "text"
            text_dir.mkdir()
            (text_dir / "en.json").write_text(json.dumps({"docs": [f"English note {i} on quantization and routing. " * 8 for i in range(20)]}))
            (text_dir / "chat.json").write_text(json.dumps({"docs": [f"User asks {i}\nAssistant replies {i}. " * 8 for i in range(20)]}))
            (text_dir / "code.json").write_text(json.dumps({"docs": [f"def fn_{i}(x):\n    return x + {i}\n" * 10 for i in range(20)]}))
            (text_dir / "fra_Latn.json").write_text(json.dumps({"docs": [f"Texte français {i} sur la quantification. " * 8 for i in range(15)]}))

            sens_path = root / "sens.json"
            plan_path = root / "plan.json"
            orig_templated_init = A.TemplatedMix.__init__

            def _init_with_root(self, tok_, tag_, root=text_dir, calib_frac=0.7, raw_docs=None):
                return orig_templated_init(self, tok_, tag_, root=text_dir, calib_frac=calib_frac, raw_docs=raw_docs)

            with (
                mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tok),
                mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
                mock.patch.object(A.TemplatedMix, "__init__", _init_with_root),
            ):
                compiler.measure(SimpleNamespace(
                    model="toy/looped-moe", model_path=str(model_dir), part="linear,experts",
                    options="lm4,lm8", embed_options="", embed_order="freq", ple_options="",
                    ple_metric=False, ple_center=False, embed_center=False, ple_damp=0.0,
                    kl_batch=6, floor_option="lm8", fix_routing=True, method="quant",
                    data="templated", per_class=False, dense_expert_h=True, expert_buckets=1,
                    math_windows="", digit_weight=1.0, dtype="float32", layers="0-3",
                    embed_buckets="", kv_layers="", calib=12, calib_len=16, chunk=6, windows=6,
                    trellis_fit="ema", out=str(sens_path),
                ))

            sens = json.loads(sens_path.read_text())
            self.assertEqual(sens["exec counts"], [1, 3, 3, 1])
            self.assertEqual(set(sens["classes"]), set(A.DOMAIN_CLASSES))
            self.assertTrue(sens["dense expert h"])
            self.assertEqual(sens["expert buckets"], 1)
            exp_recs = [r for r in sens["records"] if r["kind"] == "experts"]
            self.assertTrue(exp_recs and all(r["bucket"] == 0 and len(r["experts"]) == 8 for r in exp_recs))
            self.assertTrue(all(set(r["dkl_by_class"]) == set(A.DOMAIN_CLASSES) for r in sens["records"]))

            compiler.allocate(SimpleNamespace(
                sens=str(sens_path), target="disk", target_override="", items="linear,experts",
                bits=6.0, floor="lm8", match="", scale=1.0, budget=0.0, options="lm4,lm8",
                class_weights="", max_class_ratio=1.25, out=str(plan_path),
            ))
            plan = json.loads(plan_path.read_text())
            self.assertEqual(len(plan["experts"]), 4)
            self.assertTrue(all(len(v["buckets"]) == 1 and len(v["buckets"][0]) == 8 for v in plan["experts"].values()))
            self.assertEqual(set(plan["predicted kl by class"]), set(A.DOMAIN_CLASSES))


if __name__ == "__main__":
    unittest.main()
