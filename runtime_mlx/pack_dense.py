"""The packed form of a build that was saved only as dense weights.

finetune.py ends with every trained matrix on a form the format stores, W = diag(s) · T[state] · R, and saves W
alone. The base build's codes file gives each matrix's row groups, bit widths and rotation seed, and tables and scales
near the trained ones. From those this finds each row's path again (a beam search along the row: a state has 2^k
successors, so wrong paths die within a few entries), refits the scales and tables, and repeats until the stored form
reproduces W to rounding. A matrix counts as recovered only when it does; nothing is taken on trust from the search.

The embedding's codes do not move in training, so its buckets need only the scales and tables refitted.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

L = 12
NS = 1 << L
TOL = 2e-5  # a recovered matrix is within this of the dense one, relative to its largest entry


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- rotation, in float64

def factors(n):
    from opgraph import codes as C
    A, B = C.hadamard_factors(n)
    return A.double(), B.double()


def signs(n, seed):
    return torch.where(torch.rand(n, generator=torch.Generator().manual_seed(int(seed)), dtype=torch.float64) < 0.5, -1.0, 1.0)


def back(W, seed):
    """W Rᵀ (the inverse of the build's randomized Hadamard rotation), in float64."""
    m, n = W.shape
    A, B = factors(n)
    return (A @ (W.double() * signs(n, seed)).reshape(m, len(A), len(B)) @ B.T).reshape(m, n) / math.sqrt(n)


def forward(X, seed):
    """X R (FastRHT.forward), in float64."""
    m, n = X.shape
    A, B = factors(n)
    return (A.T @ X.double().reshape(m, len(A), len(B)) @ B).reshape(m, n) * signs(n, seed) / math.sqrt(n)


# ----------------------------------------------------------------------------- paths

def beam_paths(x, table, k, beam=8, cols=None):
    """For each row of x (in units of its scale) the path from state 0 with the least squared error among those a beam
    keeps: (rows, n) states. A state's successors are ((state << k) | symbol) & 4095."""
    rows, n = x.shape
    n = n if cols is None else min(n, cols)
    K = 1 << k
    sym = torch.arange(K)
    st = torch.zeros(rows, 1, dtype=torch.long)
    cost = torch.zeros(rows, 1, dtype=torch.float64)
    parent = torch.empty(n, rows, beam, dtype=torch.uint8)
    kept = torch.empty(n, rows, beam, dtype=torch.int16)
    take = 2 * beam
    tri = torch.triu(torch.ones(take, take, dtype=torch.bool), 1).T  # [i, j]: j comes before i
    for j in range(n):
        cand = (((st << k) & (NS - 1))[:, :, None] | sym).reshape(rows, -1)
        c = (cost[:, :, None] + (table[cand.reshape(rows, -1, K)] - x[:, j, None, None]) ** 2).reshape(rows, -1)
        t = min(take, c.shape[1])
        c, idx = torch.topk(c, t, dim=1, largest=False)
        s = torch.gather(cand, 1, idx)
        # two parents that differ only in the bits shifted out give the same state: keep the cheaper (the earlier here)
        dup = ((s[:, :, None] == s[:, None, :]) & tri[:t, :t]).any(2)
        c = torch.where(dup, torch.inf, c)
        b = min(beam, t)
        c, pick = torch.topk(c, b, dim=1, largest=False)
        st, cost = torch.gather(s, 1, pick), c
        parent[j, :, :b] = (torch.gather(idx, 1, pick) // K).to(torch.uint8)
        kept[j, :, :b] = st.to(torch.int16)
    out = torch.empty(rows, n, dtype=torch.int16)
    at = cost.argmin(1)
    r = torch.arange(rows)
    for j in range(n - 1, -1, -1):
        out[:, j] = kept[j, r, at]
        at = parent[j, r, at].long()
    return out


def fit(X, states, table, scale, sweeps=2):
    """Row scales by least squares and the table by the mean of the entries each state holds (in units of their rows'
    scales), as finetune.refit does, in float64. States no entry holds keep their value."""
    idx = states.long().flatten()
    count = torch.bincount(idx, minlength=NS)
    for _ in range(sweeps):
        T = table[states.long()]
        scale = ((X * T).sum(1) / T.pow(2).sum(1).clamp_min(1e-300)).clamp_min(1e-12)
        mean = torch.zeros(NS, dtype=torch.float64).index_add_(0, idx, (X / scale[:, None]).flatten()) / count.clamp_min(1)
        table = torch.where(count > 0, mean, table)
    return table, scale


def row_errors(X, st, t32, s32, top):
    return (s32[:, None] * t32[st.long()] - X.float()).abs().amax(1) / top


def recover_group(X, table0, scale0, k, beam=8, rounds=6, probe=384):
    """States, table and scales of one row group whose rotated rows X lie on a stored form near (table0, scale0).
    Returns them (float32 table and scales, as the format keeps them) and the largest error of the float32 form."""
    X = X.double()
    table, scale = table0.double().clone(), scale0.double().clone()
    # the scale first: a path over the first columns, the scale refitted on them, twice
    for _ in range(2):
        st = beam_paths(X[:, :probe] / scale[:, None], table, k, beam, cols=probe)
        T = table[st.long()]
        scale = ((X[:, :probe] * T).sum(1) / T.pow(2).sum(1).clamp_min(1e-300)).clamp_min(1e-12)
    top = float(X.abs().max())
    for r in range(rounds):
        st = beam_paths(X / scale[:, None], table, k, beam)
        table, scale = fit(X, st, table, scale)
        t32, s32 = table.float(), scale.float()
        per_row = row_errors(X, st, t32, s32, top)
        if float(per_row.max()) < TOL:
            break
        if r >= 1 and float((per_row >= TOL).float().mean()) < 0.2:
            break  # the table is settled by the rows that fit: the rest are handled one by one below
    bad = (per_row >= TOL).nonzero().flatten()
    retried = len(bad)
    if retried:
        # rows the first scale put on a wrong path: the table is fixed by the others now, so search each one's scale.
        # The right scale is the one whose path fits the first columns; then a wider beam over the whole row.
        good = per_row < TOL
        if int(good.sum()) >= 8:
            table, _ = fit(X[good], st[good], table, scale[good])
        grid = torch.logspace(-1.0, 1.0, 81, base=4.0, dtype=torch.float64)      # a quarter to four times, 3.5% apart
        base = X[bad].pow(2).mean(1).sqrt() / table.pow(2).mean().sqrt()         # the row's size over the table's
        best_s = torch.zeros(len(bad), dtype=torch.float64)
        best_e = torch.full((len(bad),), math.inf, dtype=torch.float64)
        for f in grid:
            s_try = base * f
            sp = beam_paths(X[bad][:, :probe] / s_try[:, None], table, k, beam, cols=probe)
            T = table[sp.long()]
            s_fit = ((X[bad][:, :probe] * T).sum(1) / T.pow(2).sum(1).clamp_min(1e-300)).clamp_min(1e-12)
            e = (s_fit[:, None] * T - X[bad][:, :probe]).pow(2).mean(1) / X[bad][:, :probe].pow(2).mean(1).clamp_min(1e-300)
            better = e < best_e
            best_s, best_e = torch.where(better, s_fit, best_s), torch.where(better, e, best_e)
        s_bad = best_s
        for _ in range(3):
            sb = beam_paths(X[bad] / s_bad[:, None], table, k, 4 * beam)
            T = table[sb.long()]
            s_bad = ((X[bad] * T).sum(1) / T.pow(2).sum(1).clamp_min(1e-300)).clamp_min(1e-12)
        st[bad], scale[bad] = sb, s_bad
        table, scale = fit(X, st, table, scale)
        t32, s32 = table.float(), scale.float()
        per_row = row_errors(X, st, t32, s32, top)
    return st, t32, s32, float(per_row.max()), r + 1, retried


def path_ok(states, k):
    s = states.long()
    return bool((s[:, 0] < (1 << k)).all() and ((s[:, 1:] >> k) == (s[:, :-1] & ((1 << (L - k)) - 1))).all())


def initial_scale(X, e, rows, table0):
    """A first scale for each row: its size over the size of the base's row in units of the base's scale (the trained
    row holds other states, but a row's mean square over its table values hardly depends on which)."""
    base = table0[(e["states"][rows].long() & (NS - 1))]
    return (X.pow(2).mean(1) / base.double().pow(2).mean(1).clamp_min(1e-300)).sqrt()


def recover_matrix(name, e, W, beam=8):
    """One trellis matrix: the saved entry it becomes (states with their table offsets, scales, table, row groups)."""
    t0 = time.time()
    m, n = W.shape
    X = back(W, e["in_seed"])
    states = torch.empty(m, n, dtype=torch.int16)
    scales = torch.empty(m, dtype=torch.float32)
    table = e["table"].float().clone()
    worst, most, retried = 0.0, 0, 0
    for g in e["row_groups"]:
        rows, k, ti = g["rows"].long(), int(g["bits"]), int(g["table_index"])
        t0g = e["table"][ti * NS : (ti + 1) * NS].double()
        st, t32, s32, err, rounds, again = recover_group(X[rows], t0g, initial_scale(X[rows], e, rows, t0g), k, beam)
        assert path_ok(st, k), f"{name}: a recovered row is not a path"
        states[rows], scales[rows] = st, s32
        table[ti * NS : (ti + 1) * NS] = t32
        worst, most, retried = max(worst, err), max(most, rounds), retried + again
    # the whole matrix again, from what the file will hold, against the dense one in its own basis
    off = torch.zeros(m, dtype=torch.long)
    for g in e["row_groups"]:
        off[g["rows"].long()] = int(g["table_index"]) * NS
    Q = scales[:, None] * table[states.long() + off[:, None]]
    back_err = float((forward(Q, e["in_seed"]).float() - W.float()).abs().max() / W.float().abs().max())
    changed = float((states != (e["states"].long() - off[:, None]).to(torch.int16)).float().mean())
    entry = {"states": (states.long() + off[:, None]).to(torch.int16), "scales": scales, "table": table, "in_seed": int(e["in_seed"]),
             "row_groups": [{"rows": g["rows"], "bits": int(g["bits"]), "table_index": int(g["table_index"])} for g in e["row_groups"]]}
    info = {"name": name, "shape": [m, n], "rotated error": worst, "error": back_err, "rounds": most, "rows searched again": retried,
            "states unlike the base's": changed, "seconds": round(time.time() - t0, 1)}
    return entry, info


def recover_bucket(b, E):
    """An embedding bucket: its codes as the base holds them (training does not move them), its row scales refitted to
    the rows and, for a trellis bucket, its table (which trains; a scalar bucket's levels do not)."""
    X = back(E, b["in_seed"])
    trellis = "states" in b
    st = (b["states"].long() & (NS - 1)) if trellis else b["codes"].long()
    table = (b["table"] if trellis else b["levels"]).double()
    idx = st.flatten()
    count = torch.bincount(idx, minlength=len(table))
    for _ in range(8 if trellis else 1):
        T = table[st]
        scale = ((X * T).sum(1) / T.pow(2).sum(1).clamp_min(1e-300)).clamp_min(1e-12)
        if trellis:
            mean = torch.zeros(len(table), dtype=torch.float64).index_add_(0, idx, (X / scale[:, None]).flatten()) / count.clamp_min(1)
            table = torch.where(count > 0, mean, table)
    t32, s32 = table.float(), scale.float()
    err = float((forward(s32[:, None] * t32[st], b["in_seed"]).float() - E.float()).abs().max() / E.float().abs().max())
    out = {"rows": b["rows"], "scales": s32, "in_seed": int(b["in_seed"])}
    out.update({"states": b["states"], "table": t32} if trellis else {"codes": b["codes"], "levels": t32})
    return out, err


# ----------------------------------------------------------------------------- entry points

def _describe(v, depth=0):
    pad = "  " * depth
    if torch.is_tensor(v):
        return f"tensor {tuple(v.shape)} {v.dtype}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_describe(x, depth + 1)}" for k, x in list(v.items())[:12]) + ("…" if len(v) > 12 else "") + "}"
    if isinstance(v, (list, tuple)):
        return f"[{len(v)}× " + (_describe(v[0], depth + 1) if v else "") + "]"
    return repr(v)[:80]


def inspect(codes, dense, model_dir):
    saved = torch.load(codes, map_location="cpu", mmap=True)
    log("codes file keys:", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in saved.items() if k != "matrices"})
    M = saved["matrices"]
    log(len(M), "coded tensors")
    for i, (n, e) in enumerate(M.items()):
        if i < 12 or "buckets" in e or i % 40 == 0:
            if "buckets" in e:
                log(n, "buckets:")
                for b in e["buckets"]:
                    log("   ", _describe(b))
            else:
                log(n, _describe({k: v for k, v in e.items() if k != "row_groups"}),
                    "groups:", [(len(g["rows"]), int(g["bits"]), int(g["table_index"])) for g in e.get("row_groups", [])])
    d = torch.load(dense, map_location="cpu", mmap=True)
    log("dense file keys:", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in d.items() if k != "params"})
    P = d["params"]
    log(len(P), "params; coded names missing from it:", [n for n in M if n not in P][:5])
    kinds = {}
    for n, v in P.items():
        kinds.setdefault((str(v.dtype), v.ndim), []).append(n)
    for key, names in kinds.items():
        log(key, len(names), names[:3], sum(P[n].numel() for n in names))
    uncoded = [(n, tuple(P[n].shape), str(P[n].dtype)) for n in P if n not in M and P[n].ndim >= 2]
    log("uncoded tensors of 2+ dims:", len(uncoded), uncoded[:40])
    log("model dir:", sorted(os.listdir(model_dir)))
    log(json.dumps(json.loads((Path(model_dir) / "config.json").read_text()))[:1500])


def one(codes, dense, name=""):
    torch.set_num_threads(8)
    saved = torch.load(codes, map_location="cpu", mmap=True)
    P = torch.load(dense, map_location="cpu", mmap=True)["params"]
    names = [n for n, e in saved["matrices"].items() if "states" in e]
    for n in ([name] if name else [names[0], next(x for x in names if P[x].shape[1] > 4000)]):
        entry, info = recover_matrix(n, saved["matrices"][n], P[n])
        log(json.dumps(info))


def _work(args):
    codes, dense, name = args
    torch.set_num_threads(1)
    saved = torch.load(codes, map_location="cpu", mmap=True)
    P = torch.load(dense, map_location="cpu", mmap=True)["params"]
    entry, info = recover_matrix(name, saved["matrices"][name], P[name])
    return name, entry, info


def pack(codes, dense, model_dir, out, workers=28, reference=True):
    import multiprocessing as mp
    from compiler_artifact import export_saved_codes, read_artifact, decode_rotated
    t0 = time.time()
    saved = torch.load(codes, map_location="cpu", mmap=True)
    d = torch.load(dense, map_location="cpu", mmap=True)
    P = d["params"]
    M = saved["matrices"]
    names = [n for n, e in M.items() if "states" in e]
    names.sort(key=lambda n: -P[n].numel())  # the long ones first, so the pool ends evenly
    out = Path(out)
    assert not out.exists(), f"{out} exists: write a new folder, never over one"
    out.parent.mkdir(parents=True, exist_ok=True)
    new, report = {}, []
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, (name, entry, info) in enumerate(pool.imap_unordered(_work, [(codes, dense, n) for n in names])):
            new[name] = entry
            report.append(info)
            if i % 10 == 0 or info["error"] >= TOL:
                log(f"[{time.time() - t0:5.0f}s] {i + 1}/{len(names)} {json.dumps(info)}")
    bad = [r for r in report if r["error"] >= TOL]
    assert not bad, f"{len(bad)} matrices not recovered: {bad[:5]}"
    matrices = {}
    for n, e in M.items():
        if n in new:
            matrices[n] = new[n]
        elif "buckets" in e:
            buckets = []
            for k, b in enumerate(e["buckets"]):
                if "states" in b or "codes" in b:
                    nb, err = recover_bucket(b, P[n][b["rows"].long()])
                    log(f"{n} bucket {k}: {len(b['rows'])} rows, error {err:.2e}")
                    assert err < TOL, f"{n} bucket {k} not recovered"
                    report.append({"name": f"{n}.b{k}", "error": err})
                    buckets.append(nb)
                else:
                    buckets.append({"rows": b["rows"]})
            matrices[n] = {"buckets": buckets}
        else:  # a scalar-coded matrix the training left alone
            matrices[n] = e
            log(f"{n}: kept as the base holds it ({[k for k in e]})")
    # the complete state: the original's tensors as its file holds them (16 bits), with every tensor the build changed
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    state = {n: v for n, v in model.state_dict().items() if "mtp." not in n}
    tied = [n for n in state if n not in P and state[n].shape == P["model.embed_tokens.weight"].shape]
    assert set(P) <= set(state), [n for n in P if n not in state][:5]
    state.update(P)
    for n in tied:  # the output head is the embedding
        state[n] = state["model.embed_tokens.weight"]
    rest = {n: (tuple(v.shape), str(v.dtype)) for n, v in state.items() if n not in M and n not in tied}
    log(f"{len(state)} tensors; tied to the embedding: {tied}; uncoded: {len(rest)}, {sum(state[n].numel() for n in rest)} values, "
        f"of them unchanged from the original: {sum(n not in P for n in rest)}; e.g. {list(rest.items())[:6]}")
    del model
    receipt = export_saved_codes({"matrices": matrices, "model": "Qwen/Qwen3.5-4B"}, out, dense_tensors=state, model_dir=model_dir)
    log(f"[{time.time() - t0:5.0f}s] packed: {json.dumps(receipt)[:600]}")
    # the file as written, read back and decoded, against the dense weights (rotated, where the file stores them)
    arrays, manifest = read_artifact(out)
    worst = 0.0
    for n, meta in manifest["matrices"].items():
        if meta["kind"] == "table":
            continue
        Q = torch.from_numpy(decode_rotated(arrays, meta))
        seed = M[n]["in_seed"]
        worst = max(worst, float((forward(Q, seed).float() - P[n].float()).abs().max() / P[n].float().abs().max()))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    summary = {"matrices": len(new), "worst error, file against dense": worst, "bytes": size, "seconds": round(time.time() - t0),
               "worst recovery error": max(r["error"] for r in report), "states unlike the base's": float(np.mean([r["states unlike the base's"] for r in report if "shape" in r]))}
    log(json.dumps(summary))
    (out / "recovery.json").write_text(json.dumps({"summary": summary, "matrices": report}, indent=1))
    if reference:
        write_reference(dense, model_dir, out)
    log(f"=> done ({time.time() - t0:.0f}s)")


PROMPTS = [
    "A train leaves at 9:40 and arrives at 13:05. It stops for 12 minutes at each of 4 stations. How long is it moving?",
    "Explain why the sky is blue, in three short paragraphs.",
    "Write a Python function that returns the n-th Fibonacci number, with a docstring and a loop.",
    "What is 487 times 36? Work it out step by step.",
]


def write_reference(dense, model_dir, out, new_tokens=96):
    """Logits of the dense model as it was benchmarked (transformers, float32) on real prompts and its own continuations."""
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(32)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    params = dict(model.named_parameters())
    for n, W in torch.load(dense, map_location="cpu", mmap=True)["params"].items():
        params[n].data.copy_(W.to(params[n].dtype))
    keep = {}
    with torch.no_grad():
        for i, q in enumerate(PROMPTS):
            ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
            ids = ids["input_ids"] if not torch.is_tensor(ids) else ids
            full = model.generate(ids, max_new_tokens=new_tokens, do_sample=False)
            logits = model(full).logits[0].float()
            keep[f"ids{i}"], keep[f"logits{i}"] = full[0].to(torch.int32), logits
            log(f"[reference {time.time() - t0:4.0f}s] prompt {i}: {ids.shape[1]} + {full.shape[1] - ids.shape[1]} tokens: {tok.decode(full[0, ids.shape[1]:])[:160]!r}")
    save_file(keep, str(Path(out) / "reference.safetensors"))


def compact_reference(out, top=4096):
    """reference.safetensors holds every logit (half a gigabyte). This keeps, for each position, the reference's `top`
    largest log-probabilities with their token ids: enough for KL over those tokens with the rest as one bucket."""
    from safetensors.torch import load_file, save_file
    ref = load_file(str(Path(out) / "reference.safetensors"))
    keep = {}
    for i in range(len([k for k in ref if k.startswith("ids")])):
        lp = torch.log_softmax(ref[f"logits{i}"].double(), -1)
        v, idx = lp.topk(top, dim=-1)
        keep[f"ids{i}"], keep[f"logp{i}"], keep[f"tokens{i}"] = ref[f"ids{i}"], v.float(), idx.to(torch.int32)
        log(f"prompt {i}: {lp.shape[0]} positions; mass outside the top {top}: mean {float((1 - v.exp().sum(-1)).mean()):.2e}, max {float((1 - v.exp().sum(-1)).max()):.2e}")
    save_file(keep, str(Path(out) / "reference_top.safetensors"))
