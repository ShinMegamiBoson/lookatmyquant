"""Calibration drivers and the completeness comparison.

For each method the reference pipeline runs layer by layer. At every layer the frame version is applied to the
identical layer state and inputs, and the two results are compared (weights, integer codes, masks, chosen
arguments). Then the frame version runs again end to end with its own calibration, and both compressed models are
evaluated against the original.
"""

from __future__ import annotations

import time

import torch

from . import methods as ops
from . import reference as ref
from .frame import ModelNet, covariance
from .model import INPUT_OF, LAYERS, LINEARS, Model, attention, batched, capture_layer, evaluate, layer_forward, logits, mlp, rope
from .rules import awq_search_clip, awq_search_scale


class Feats:
    """The signals on a layer's input bonds, with their covariances computed on demand."""

    def __init__(self, x=None, H=None):
        self.x, self._H = x, dict(H or {})

    def H(self, key):
        if key not in self._H:
            self._H[key] = covariance(self.x[key])
        return self._H[key]


@torch.no_grad()
def profile(model: Model, ids, cos, sin) -> list[Feats]:
    """Input covariances of every linear layer of the uncompressed model (SVD-LLM's profiling)."""
    out, h = [], model.embed[ids]
    for L in model.layers:
        h, x = capture_layer(h, L, cos, sin, model.mean_dim)
        out.append(Feats(H={key: covariance(v) for key, v in x.items()}))
    return out


# ----------------------------------------------------------------------------- per-matrix methods, both versions


def per_matrix(ref_fn, ops_fn, needs_H: bool):
    """Lift a per-matrix method to a layer: ref mutates the Model, ops replaces nets in the ModelNet."""

    def ref_layer(m: Model, i, feats):
        L, art = m.layers[i], {}
        for n in LINEARS:
            W = getattr(L, n)
            out = ref_fn(W, feats.H(INPUT_OF[n]) if needs_H else None)
            Wn, extra = out if isinstance(out, tuple) else (out, None)
            setattr(L, n, Wn.to(W.device, W.dtype))
            art[n] = extra
        return art

    def ops_layer(net: ModelNet, i, feats):
        art = {}
        for n in LINEARS:
            lin = ops_fn(net.linears[(i, n)].matrix(), feats.H(INPUT_OF[n]) if needs_H else None)
            net.linears[(i, n)] = lin
            c = lin.cores.get("W")
            art[n] = None if c is None else (c.keep if c.keep is not None else (ops.codes(lin) if c.domain.kind == "int" else None))
        return net, art

    return ref_layer, ops_layer


def awq_ref_layer(bits, gs, cos, sin):
    def fn(m: Model, i, feats):
        L, x = m.layers[i], feats.x
        cand = ref.awq_scale_candidate(bits, gs)
        o = L.o
        s_attn, r1 = awq_search_scale(x["attn_in"], lambda ws: batched(lambda xb: attention(xb, *ws, o, cos, sin)[0], x["attn_in"]),
                                      [L.q, L.k, L.v], cand)
        down = L.down
        s_mlp, r2 = awq_search_scale(x["mlp_in"], lambda ws: batched(lambda xb: mlp(xb, ws[0], ws[1], down)[0], x["mlp_in"]), [L.gate, L.up], cand)
        s_down, r3 = awq_search_scale(x["down_in"], lambda ws: x["down_in"] @ ws[0].T, [L.down], cand)
        L.attn_norm = L.attn_norm / s_attn
        L.q, L.k, L.v = L.q * s_attn, L.k * s_attn, L.v * s_attn
        L.mlp_norm = L.mlp_norm / s_mlp
        L.gate, L.up = L.gate * s_mlp, L.up * s_mlp
        L.up = L.up / s_down.reshape(-1, 1)
        L.down = L.down * s_down
        xs = {"attn_in": x["attn_in"] / s_attn, "o_in": x["o_in"], "mlp_in": x["mlp_in"] / s_mlp, "down_in": x["down_in"] / s_down}
        clips = {}
        for name in ("v", "o", "gate", "up", "down"):
            W = getattr(L, name)
            mv = awq_search_clip(W, xs[INPUT_OF[name]], ref.awq_clip_candidate(bits, gs), gs)
            mvd = mv.to(W.device, W.dtype)[..., None]
            setattr(L, name, torch.clamp(W.reshape(W.shape[0], -1, gs), -mvd, mvd).reshape(W.shape))
            clips[name] = mv
        return {"ratios": (r1, r2, r3), "clips": clips}

    return fn


def awq_ref_final(bits, gs):
    def fn(m: Model):
        art = {}
        for i, L in enumerate(m.layers):
            for n in LINEARS:
                W = getattr(L, n)
                Wq, q = ref.awq_pseudo_quantize(W.cpu().double(), bits, gs)
                setattr(L, n, Wq.to(W.device, W.dtype))
                art[(i, n)] = q
        return art

    return fn


# ----------------------------------------------------------------------------- comparison


def _agree(a, b):
    if a is None or b is None:
        return None
    if isinstance(a, torch.Tensor):
        return (a.cpu().reshape(b.shape) == b.cpu()).double().mean().item()
    if isinstance(a, dict):
        return {k: _agree(a[k], b[k]) for k in a}
    if isinstance(a, (tuple, list)):
        return float(all(x == y for x, y in zip(a, b)))
    return float(a == b)


def compare_layer(m: Model, net: ModelNet, i: int, ref_art, ops_art) -> dict:
    L, out = m.layers[i], {"weights": {}, "args": {}}
    for n in LINEARS:
        A = getattr(L, n)
        B = net.linears[(i, n)].matrix(A.device, A.dtype)
        out["weights"][n] = ((A - B).norm() / A.norm()).item()
    for key in ("attn", "mlp"):
        g = getattr(L, f"{key}_norm")
        if g is not None:
            out["weights"][f"{key}_norm"] = ((g - net.gains[(i, key)]).norm() / g.norm()).item()
    out["args"] = {k: _agree(ref_art[k], ops_art[k]) for k in ref_art if k in ops_art}
    return out


# ----------------------------------------------------------------------------- drivers


@torch.no_grad()
def run_local(name, model: Model, calib, ref_layer, ops_layer, mode, ref_final=None, ops_final=None, log=print, n_layers=LAYERS):
    dev = model.device
    cos, sin = rope(calib.shape[1], dev)
    ids = calib.to(dev)
    prof = profile(model, ids, cos, sin) if mode == "profile" else None
    t0 = time.time()

    mr, per_layer = model.copy(), []
    h = mr.embed[ids] if mode == "sequential" else None
    for i in range(n_layers):
        feats = None
        if mode == "sequential":
            _, x = capture_layer(h, mr.layers[i], cos, sin, mr.mean_dim)
            feats = Feats(x)
        elif mode == "profile":
            feats = prof[i]
        probe, ops_art = ops_layer(ModelNet.from_model(mr), i, feats)
        ref_art = ref_layer(mr, i, feats)
        per_layer.append(compare_layer(mr, probe, i, ref_art, ops_art))
        if mode == "sequential":
            L = mr.layers[i]
            h = batched(lambda xb: layer_forward(xb, L, cos, sin, mr.mean_dim), h)
        if i % 8 == 7:
            log(f"  [{name}] reference + per-layer frame check: layer {i + 1}/{n_layers} ({time.time() - t0:.0f}s)")
    final = None
    if ref_final is not None:
        probe, ops_art = ops_final(ModelNet.from_model(mr))
        ref_art = ref_final(mr)
        final = {"codes": _flatten_agreement(_agree(ref_art, ops_art)),
                 "weights": max(compare_layer(mr, probe, i, {}, {})["weights"][n] for i in range(n_layers) for n in LINEARS)}

    net = ModelNet.from_model(model)
    h = net.embed[ids] if mode == "sequential" else None
    for i in range(n_layers):
        feats = None
        if mode == "sequential":
            _, x = capture_layer(h, net.compile().layers[i], cos, sin, net.mean_dim)
            feats = Feats(x)
        elif mode == "profile":
            feats = prof[i]
        net, _ = ops_layer(net, i, feats)
        if mode == "sequential":
            L = net.compile().layers[i]
            h = batched(lambda xb: layer_forward(xb, L, cos, sin, net.mean_dim), h)
    if ops_final is not None:
        net, _ = ops_final(net)
    log(f"  [{name}] frame end-to-end done ({time.time() - t0:.0f}s)")
    return mr, net, per_layer, final


def summarize(name, word, model: Model, mr: Model, net: ModelNet, per_layer, final, eval_ids, rate_of=None) -> dict:
    base = evaluate(mr, eval_ids, model)
    mo = net.compile()
    frame = evaluate(mo, eval_ids, model)
    x = eval_ids[:2].to(model.device)
    logit_gap = (logits(mr, x) - logits(mo, x)).abs().max().item()
    w = [v for rec in per_layer for v in rec["weights"].values()]
    args = [rec["args"] for rec in per_layer]
    n_orig = sum(getattr(L, n).numel() for L in model.layers for n in LINEARS)
    bits = (rate_of(net) if rate_of else sum(lin.rate_bits() for lin in net.linears.values())) / n_orig
    return {"method": name, "word": word, "ppl_ref": base["ppl"], "kl_ref": base["kl"], "ppl_frame": frame["ppl"], "kl_frame": frame["kl"],
            "max_logit_gap": logit_gap, "max_layer_rel_weight_diff": max(w), "arg_agreement": _flatten_agreement(args),
            "final": final, "bits_per_weight": bits}


def _flatten_agreement(args):
    vals = []

    def walk(v):
        if isinstance(v, dict):
            v = list(v.values())
        if isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, (int, float)):
            vals.append(v)

    walk(args)
    return min(vals) if vals else None
