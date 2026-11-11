"""Fine-tuning a trellis-coded build so that it ends exactly on codes the format stores (LookAtMyQuant 2).

A trellis-coded weight has no small step, and quantizing a layer takes tens of seconds, so the student is a copy of the
build whose trellis matrices are real-valued weights W. They start on their quantized values P. Each step is one Adam step
on the token-weighted KL from the original model at the positions read of a fixed pool of text (finetune_lib.pool_and_held,
topk_kl), then a pull W ← W + β (P − W). Every --project-every steps one of --project-groups groups of layers is quantized
again with the build's own quantizer under its input statistics (finetune_lib.rounded), so each layer's P follows its W
once a cycle. Over the last --settle steps β rises to --beta-end and the step size falls to zero; then every matrix takes
its P, a few are quantized once more to check that nothing changes, and the model is saved in the dense format
math_eval.py reads.

The recipe's options:
  --refit-handles     at each re-quantization of a matrix its row scales and its codebooks are refitted to W: least squares
                      for the scales, a Lloyd step for the codebooks, twice over
  --train-norms       the norms' weights train
  --train-embedding   the embedding's row scales, codebooks and 16-bit rows train with its codes fixed
  --running-stats     the input statistics a re-quantization uses are a decayed running sum over the training batches'
                      own activations (a window of one --project-every), not the build's calibration statistics
  --rebuild-at t      from step t each group's next re-quantization starts once from the original model's weights, under
                      the statistics as they then stand, and its W restarts on the new P. Weights that sit on their codes
                      quantize to the same codes under any statistics, so this is what changes the codes

    python finetune.py --codes build/codes.pt --base-dense build/dense.pt --math-cache cache/math_checked.pt \
        --more-text cache/other_text.pt:science+code --refit-handles --train-norms --train-embedding \
        --running-stats --rebuild-at 1000 --save tuned/dense.pt --out results/finetune.json     (run_batch.py: lm2Finetune)

The other options are measurements and variants that the recipe does not use: --project-share (re-quantize only the rows
that drifted most; no faster), a weaker --beta (the weights drift further and the gain on math is lost), --gate-pull
(pull less where a weight's gradient is steady; no weight's is, after the first 200 steps), --stats-drift (how far the input
statistics move), --profile and --split.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import fast_delta
from opgraph import codes as C
from math_eval import load_dense
from quantize import DEV
from finetune_lib import CodedTable, FastRHT, damage_of, exact_matmul, log, owner_of_layers, pool_and_held, rotated_moment, rounded, row_code, text_args, topk_kl


H_MATH = 0.7  # of the windows a stage's input statistics are taken on, the share that are worked solutions
GATE = 0.1  # --gate-pull: the share of a weight's mean squared gradient that its mean gradient accounts for, from which it is not pulled


def layer_of(name):
    return int(name.split(".layers.")[1].split(".")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, help="the build's codes: each matrix's rotation, row groups, widths, tables, scales and states")
    ap.add_argument("--base-dense", required=True, help="the build's weights (math_eval's dense: format): the student")
    ap.add_argument("--layers", default="0-31", help="first-last: the decoder layers whose trellis matrices train")
    ap.add_argument("--model-path", default="/root/.cache/huggingface/models/Qwen3.5-4B")
    text_args(ap)
    ap.add_argument("--math-share", type=float, default=0.5, help="of each batch, drawn from the pool's worked solutions")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--split", type=int, default=1, help="a batch's windows go through in this many parts and their gradients are summed: the same step in less memory")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--eps-rms", type=float, default=1.0, help="Adam's ε for each tensor, in units of its gradient's RMS at the start (finetune_lib.py)")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--beta", type=float, default=1e-3, help="the share of the way to P(W) each matrix moves per step, until the last --settle steps")
    ap.add_argument("--beta-end", type=float, default=0.03, help="at the end, risen geometrically over the last --settle steps while the steps fall to zero")
    ap.add_argument("--settle", type=int, default=600)
    ap.add_argument("--project-every", type=int, default=250)
    ap.add_argument("--project-groups", type=int, default=4, help="layers are projected in this many groups in turn (layer mod groups)")
    ap.add_argument("--project-share", type=float, default=1.0, help="of each matrix's rows, the share rounded again at a projection: those furthest from their stored form")
    ap.add_argument("--refit-handles", action="store_true")
    ap.add_argument("--train-norms", action="store_true")
    ap.add_argument("--train-embedding", action="store_true")
    ap.add_argument("--lr-norms", type=float, default=1e-4)
    ap.add_argument("--lr-scale", type=float, default=3e-4, help="Adam step for the embedding's log row scales")
    ap.add_argument("--lr-table", type=float, default=1e-4, help="Adam step for the embedding's tables")
    ap.add_argument("--lr-rows", type=float, default=1e-5, help="Adam step for the embedding rows kept at 16 bits")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--h-windows", type=int, default=256, help="pool windows for the input statistics, 70% worked solutions")
    ap.add_argument("--running-stats", action="store_true")
    ap.add_argument("--gate-pull", action="store_true")
    ap.add_argument("--stats-drift", default="")
    ap.add_argument("--rebuild-at", type=int, default=0)
    ap.add_argument("--profile", action="store_true", help="time a step's parts on this GPU (the embedding's rebuild, the batch whole and in halves, each chunk rule, Adam, the pull) and stop")
    ap.add_argument("--fast-delta", action="store_true", help="the linear-attention layers' chunk rule with one triangular solve for its 63-step loop (fast_delta.py)")
    ap.add_argument("--save", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert not args.rebuild_at or args.running_stats, "--rebuild-at rounds for the stream as it stands: it needs --running-stats"
    if args.fast_delta:
        fast_delta.install()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    lo, hi = (int(x) for x in args.layers.split("-"))
    tok = AutoTokenizer.from_pretrained(args.model_path)
    ref = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float32).to(DEV).eval()
    student = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float32).to(DEV).eval()
    for m in (ref, student):
        for p in m.parameters():
            p.requires_grad_(False)
    loaded = load_dense(student, args.base_dense)
    sp = dict(student.named_parameters())
    saved = torch.load(args.codes, map_location="cpu", mmap=True)
    names = [n for n, e in saved["matrices"].items() if "states" in e and ".layers." in n and lo <= layer_of(n) <= hi]
    assert names, f"no trellis matrices in layers {lo}-{hi}"
    rotations, R, groups, scale, states, stage_of, by_stage = {}, {}, {}, {}, {}, {}, {}
    for n in names:
        e = saved["matrices"][n]
        assert e.get("trellis_layout", "row") == "row", f"{n}: paths along columns"
        stage_of[n] = (sp[n].shape[1], e["in_seed"])  # a stage's matrices read one input through one rotation (a block's stages share the seed)
        by_stage.setdefault(stage_of[n], []).append(n)
        R[n] = rotations.setdefault(stage_of[n], FastRHT(*stage_of[n]))
        table = lambda g: e["table"][g["table_index"] * 4096 : (g["table_index"] + 1) * 4096].to(DEV, torch.float32)
        groups[n] = [(g["rows"].to(DEV), int(g["bits"]), row_code(table(g), int(g["bits"]))) for g in e["row_groups"]]
        scale[n] = e["scales"].to(DEV, torch.float32).clamp_min(1e-12).clone()
        offset = torch.zeros(sp[n].shape[0], dtype=torch.int16)
        for g in e["row_groups"]:
            offset[g["rows"]] = g["table_index"] * 4096
        states[n] = e["states"] - offset[:, None]  # each entry's state in its own group's table, on the CPU
        sp[n].requires_grad_(True)
    n_weights = sum(sp[n].numel() for n in names)
    bits = sum(len(rows) * sp[n].shape[1] * k for n in names for rows, k, _ in groups[n]) / n_weights
    log(f"the build: {loaded['tensors']} tensors, {loaded['GB']:.3f} GB; layers {lo}-{hi}: {len(names)} trellis matrices in {len(by_stage)} stages, "
        f"{n_weights / 1e6:.0f}M weights at {bits:.3f} bits each ({time.time() - t0:.0f}s)")

    def stored(n):
        """The matrix its states, tables and row scales store."""
        with exact_matmul():
            st = states[n].to(DEV).long()
            T = torch.empty(st.shape, device=DEV)
            for g_rows, _, code in groups[n]:
                T[g_rows] = code.values.to(DEV, torch.float32)[st[g_rows]]
            return R[n].forward(scale[n][:, None] * T)

    off = max(float((stored(n) - sp[n].data).norm() / sp[n].data.norm()) for n in names[:: max(1, len(names) // 8)])
    log(f"  the codes' matrices against the build's weights, every {max(1, len(names) // 8)}th: at most {off:.2e} of their size apart")
    assert off < 1e-4, "the build's weights are not its codes' matrices"

    held, held_kinds, pool, n_math_pool, digit, sel_pos, sel_w, dropped = pool_and_held(args, tok, ref)
    n_text = len(pool) - n_math_pool

    # the original's part, once: its top predictions at the positions read
    owner_r, owner_s = owner_of_layers(ref)[0], owner_of_layers(student)[0]
    head_r, head_s = ref.get_output_embeddings(), student.get_output_embeddings()
    at = lambda hid, pos: hid.gather(1, pos[:, :, None].expand(-1, -1, hid.shape[-1]))
    top_lp = torch.empty(len(pool), args.positions, args.top)
    top_id = torch.empty(len(pool), args.positions, args.top, dtype=torch.int32)
    tail = torch.empty(len(pool), args.positions)
    with torch.no_grad():
        for b in range(0, len(pool), 32):
            hid = owner_r(input_ids=pool[b : b + 32].to(DEV), use_cache=False).last_hidden_state
            t = F.log_softmax(head_r(at(hid, sel_pos[b : b + 32].to(DEV))).float(), -1).topk(args.top, -1)
            top_lp[b : b + 32], top_id[b : b + 32] = t.values.cpu(), t.indices.to(torch.int32).cpu()
            tail[b : b + 32] = (1 - t.values.exp().sum(-1)).clamp_min(0).cpu()
            del hid, t
    ref.to("cpu")  # 17 GB the GPU needs while training: the whole build peaked at 180 of its 192 GB with the original beside it
    log(f"  the original's top {args.top} predictions at {args.positions} positions of {len(pool)} windows; mean probability outside them {float(tail.mean()):.4f} ({time.time() - t0:.0f}s)")

    # each stage's input statistics through the build's own stream, digit-weighted, rotated and factored once; the statistics
    # and their factor wait on the CPU in 32 bits (30 GB for all 32 layers)
    module = lambda n: student.get_submodule(n[: -len(".weight")])
    inputs = {}
    once = [module(n).register_forward_pre_hook(lambda m, a, n=n: inputs.__setitem__(n, a[0])) for n in names]
    with torch.no_grad():
        student(pool[:2].to(DEV), use_cache=False)
    for h_ in once:
        h_.remove()
    for s, members in by_stage.items():
        for n in members:
            assert torch.equal(inputs[n], inputs[members[0]]), f"{n} does not read its stage's input"
    inputs.clear()
    take = round(H_MATH * args.h_windows)
    windows = lambda first: torch.cat([torch.arange(n_math_pool)[first : first + take], n_math_pool + torch.arange(n_text)[first : first + args.h_windows - take]])
    rows_h = windows(0)
    weight_of = lambda rows: args.token_floor + (1 - args.token_floor) * F.pad(digit[rows], (0, 1)).float()  # a window's last position predicts nothing known
    unit = float(weight_of(rows_h).mean())
    Hs = {s: torch.zeros(s[0], s[0], device=DEV) for s in by_stage}  # each stage's Σ w x xᵀ over the tokens weighed so far
    now = {"w": None, "seen": 0.0}  # the weights of the tokens going through (None: not weighed), and the sum of all weighed

    def moment(s):
        def f(m, a):
            if now["w"] is not None:
                x = a[0].detach().reshape(-1, a[0].shape[-1]).float()
                Hs[s].addmm_(x.T, x * now["w"])
        return f
    for s, members in by_stage.items():  # kept through the run: with --running-stats the training batches are weighed too
        module(members[0]).register_forward_pre_hook(moment(s))

    def weigh(rows):
        """These windows' tokens added to every stage's statistics."""
        w = weight_of(rows) / unit
        with torch.no_grad():
            for b in range(0, len(rows), 16):
                now["w"] = w[b : b + 16].reshape(-1, 1).to(DEV)
                student(pool[rows[b : b + 16]].to(DEV), use_cache=False)
        now["w"], now["seen"] = None, now["seen"] + float(w.sum())

    weigh(rows_h)
    factors = {}  # each stage's statistics as first taken: H, and unless they run on, its factor
    for s, members in by_stage.items():
        e = C.Environment(rotated_moment(Hs[s] / now["seen"], R[members[0]]))
        factors[s] = (e.H.float().cpu(), None, None) if args.running_stats else (e.H.float().cpu(), e.U.float().cpu(), e.dead.cpu())
        del e
    if not (args.running_stats or args.stats_drift):
        Hs.clear()

    def environment(s):
        """A stage's statistics for a projection: as first taken, or with --running-stats as they stand."""
        if args.running_stats:
            return C.Environment(rotated_moment(Hs[s] / now["seen"], R[by_stage[s][0]]))
        return SimpleNamespace(H=factors[s][0].to(DEV).double(), U=factors[s][1].to(DEV).double(), dead=factors[s][2].to(DEV))
    log(f"  input statistics of {len(factors)} stages on {len(rows_h)} windows of the build's own stream, factored ({time.time() - t0:.0f}s)")

    P = {n: sp[n].data.clone() for n in names}

    def refit(n):
        """--refit-handles: the row scales by least squares and each group's table by Lloyd's step (every value to the mean of
        the real weights its entries hold, in units of their rows' scales), twice over; then P from them."""
        with exact_matmul():
            X = R[n].back(sp[n].data)
            st = states[n].to(DEV).long()
            T = torch.empty_like(X)
            for _ in range(2):
                for g_rows, _, code in groups[n]:
                    T[g_rows] = code.values.to(DEV, torch.float32)[st[g_rows]]
                scale[n].copy_(((X * T).sum(1) / T.pow(2).sum(1).clamp_min(1e-30)).clamp_min(1e-12))
                for g_rows, _, code in groups[n]:
                    idx, y = st[g_rows].flatten(), (X[g_rows] / scale[n][g_rows, None]).flatten().double()
                    count = torch.bincount(idx, minlength=4096)
                    mean = torch.zeros(4096, device=DEV, dtype=torch.float64).index_add_(0, idx, y) / count.clamp_min(1)
                    code.values = torch.where(count > 0, mean, code.values.to(DEV)).cpu()
        P[n] = stored(n)

    original = dict(ref.named_parameters())

    def project(which, share=1.0, fresh=False):
        """P of these matrices' current weights: every row, or the `share` of each matrix's rows furthest from their stored
        form. With `fresh` (--rebuild-at) P of the original's weights instead, and the real weights put on it. Returns the share of entries rounded again, the share of those whose state changed, how far the stored forms
        moved, ‖P_new − P_old‖ ÷ ‖P_old‖, and with --running-stats how far the stages' statistics are from the first taken."""
        again = changed = 0
        moved = size = 0.0
        apart = [0.0, 0.0]
        with torch.no_grad():
            for s, members in by_stage.items():
                mine = [n for n in members if n in which]
                if not mine:
                    continue
                e = environment(s)
                if args.running_stats:  # against the statistics first taken, each with trace 1
                    h0 = factors[s][0].to(DEV).double()
                    h0 /= h0.diagonal().sum()
                    apart = [apart[0] + float((e.H / e.H.diagonal().sum() - h0).pow(2).sum()), apart[1] + float(h0.pow(2).sum())]
                    del h0
                for n in mine:
                    W, old = (original[n].data.to(DEV) if fresh else sp[n].data), P[n]
                    rows = None
                    if share < 1:
                        drift = (W - old).pow(2).sum(1) / old.pow(2).sum(1).clamp_min(1e-30)
                        rows = drift.topk(max(1, round(share * len(drift)))).indices.sort().values
                    new, st = rounded(W, R[n], groups[n], scale[n], e, rows)
                    was = states[n] if rows is None else states[n][rows.cpu()]
                    again, changed = again + st.numel(), changed + int((st != was).sum())
                    if rows is None:
                        states[n], P[n] = st, new
                    else:
                        states[n][rows.cpu()] = st
                        P[n] = old.clone()
                        P[n][rows] = new
                    if fresh:
                        sp[n].data.copy_(P[n])
                    if args.refit_handles:
                        refit(n)
                    moved, size = moved + float((P[n] - old).pow(2).sum()), size + float(old.pow(2).sum())
                del e
        total = sum(sp[n].numel() for n in which)
        return again / max(1, total), changed / max(1, again), (moved / max(size, 1e-30)) ** 0.5, (apart[0] / apart[1]) ** 0.5 if apart[1] else None

    def gap():
        """‖W − P‖ ÷ ‖W‖ over the trained matrices (P as each was last projected)."""
        with torch.no_grad():
            return (sum(float((sp[n].data - P[n]).pow(2).sum()) for n in names) / sum(float(sp[n].data.pow(2).sum()) for n in names)) ** 0.5

    def against_original(swap):
        """damage_of with the original back on the GPU for it; with `swap`, every trained matrix in its stored form as last
        projected (then the real weights back)."""
        ref.to(DEV)
        kept = {n: sp[n].data for n in names}
        if swap:
            for n in names:
                sp[n].data = P[n]
        d = damage_of(ref, student, held, held_kinds, bs=2)
        for n in names:
            sp[n].data = kept[n]
        ref.to("cpu")
        torch.cuda.empty_cache()
        return d

    res = {"device": DEV, "args": vars(args), "matrices": len(names), "weights": n_weights, "bits per weight": bits,
           "pool": {"worked solutions": n_math_pool, "other text": n_text, "tokens": pool.numel(), "dropped for shared text": dropped}, "rounds": [], "projections": []}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    progress = Path(args.save + ".progress.json") if args.save else None  # on the volume, readable while the job runs

    def save():
        res["seconds"] = round(time.time() - t0)
        out.write_text(json.dumps(res, indent=1))
        if progress:
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.write_text(json.dumps(res, indent=1))

    def show(d):
        return f"math KL {d['math']['KL']:.4f}, digit KL {d['math']['digit KL']:.4f}, digit agreement {d['math']['digit agreement']:.4f}, text KL {d['text']['KL']:.4f}"

    def check(step, beta, running, same=False):
        real = against_original(False)
        rec = {"step": step, "beta": beta, "train loss": running, "gap": round(gap(), 5), "continuous": real, "rounded": real if same else against_original(True),
               "entries unlike the first projection's": None, "seconds": round(time.time() - t0)}
        res["rounds"].append(rec)
        log(f"  step {step}: gap {rec['gap']:.4f} | continuous: {show(rec['continuous'])} | stored as last projected: {show(rec['rounded'])} ({rec['seconds']}s)")
        save()

    if args.stats_drift:
        def taken(rows):
            """Each stage's statistics of the student as it stands on these windows: rotated, with trace 1, on the CPU."""
            for H in Hs.values():
                H.zero_()
            now["seen"] = 0.0
            weigh(rows)
            out_ = {}
            for s, members in by_stage.items():
                H = rotated_moment(Hs[s] / now["seen"], R[members[0]])
                out_[s] = (H / H.diagonal().sum()).float().cpu()
            return out_
        apart = lambda X, Y: float((X.to(DEV).double() - Y.to(DEV).double()).norm() / Y.to(DEV).double().norm())
        kind = lambda s: by_stage[s][0].split(".layers.")[1].split(".", 1)[1][: -len(".weight")]
        layers = sorted({lo + (hi - lo) * i // 2 for i in range(3)})
        test = [n for n in names if layer_of(n) in layers]
        with torch.no_grad():
            here, elsewhere = taken(rows_h), taken(windows(args.h_windows))
            load_dense(student, args.stats_drift)
            other = taken(rows_h)
        rows_ = [{"layer": layer_of(by_stage[s][0]), "stage": kind(s), "other windows": apart(elsewhere[s], here[s]), "other build": apart(other[s], here[s])} for s in by_stage]
        res["stats drift"] = {"other build": args.stats_drift, "stages": rows_}
        mean = lambda xs: sum(xs) / max(1, len(xs))
        for k in sorted({r["stage"] for r in rows_}):
            mine = [r for r in rows_ if r["stage"] == k]
            log(f"  statistics of {k} ({len(mine)} stages): {mean([r['other windows'] for r in mine]):.3f} apart on other windows of this build, "
                f"{mean([r['other build'] for r in mine]):.3f} apart in the other build on the same windows")
        third = max(1, (hi - lo + 1) // 3)
        for a in range(lo, hi + 1, third):
            mine = [r for r in rows_ if a <= r["layer"] < a + third]
            log(f"  layers {a}-{min(hi, a + third - 1)}: {mean([r['other windows'] for r in mine]):.3f} on other windows, {mean([r['other build'] for r in mine]):.3f} in the other build")
        # What the difference costs a rounding. (The build's own matrices round to themselves under any statistics: they are
        # codes already.) The original's matrices of three layers are rounded from scratch under each statistics, with this
        # build's row groups, tables and starting scales, and each result's error is weighed by each of the others
        sets = {"the first windows": here, "other windows": elsewhere, "the other build": other}
        err = {a_: dict.fromkeys(sets, 0.0) for a_ in sets}
        differ = dict.fromkeys(list(sets)[1:], 0)
        total = 0
        orig = dict(ref.named_parameters())
        with torch.no_grad(), exact_matmul():
            for n in test:
                W0 = orig[n].data.to(DEV)
                X0 = R[n].back(W0).double()
                env = {k: C.Environment(v[stage_of[n]].to(DEV)) for k, v in sets.items()}
                off, st = {}, {}
                for k in sets:
                    Pn, st[k] = rounded(W0, R[n], groups[n], scale[n].clone(), env[k])
                    off[k] = R[n].back(Pn).double() - X0
                for a_ in sets:
                    for b_ in sets:
                        err[a_][b_] += float(((off[a_] @ env[b_].H) * off[a_]).sum())
                for k in differ:
                    differ[k] += int((st[k] != st["the first windows"]).sum())
                total += st["the first windows"].numel()
                del env, off, st, W0, X0
        res["stats drift"]["rounding"] = {"layers": layers, "error of the row's rounding weighed by the column's": err, "codes unlike the first windows'": {k: v / total for k, v in differ.items()}}
        for judge in list(sets)[1:]:
            log(f"  layers {layers} of the original rounded under the first windows' statistics, weighed by those of {judge}: "
                f"{err['the first windows'][judge] / err[judge][judge] - 1:+.1%} error against rounding under those; {differ[judge] / total:.1%} of codes differ ({time.time() - t0:.0f}s)")
        log(f"  and rounded under other windows', weighed by the other build's: {err['other windows']['the other build'] / err['the other build']['the other build'] - 1:+.1%}")
        save()
        log("=> statistics measured")
        return
    check(0, args.beta, None, same=True)
    res["the build"] = res["rounds"][0]["continuous"]

    # what trains besides the matrices: (tensor, Adam step) with its gradient in .grad after a backward pass
    others = []
    if args.train_norms:
        norms = [(n, p) for n, p in sp.items() if p.dim() == 1 and "norm" in n]
        for _, p in norms:
            p.requires_grad_(True)
        others += [(p, args.lr_norms) for _, p in norms]
        log(f"  {len(norms)} norm vectors train, {sum(p.numel() for _, p in norms) / 1e3:.0f}K values")
    emb_name = next((n for n, e in saved["matrices"].items() if "buckets" in e), None)
    table = None
    if args.train_embedding and emb_name:
        table = CodedTable(saved["matrices"][emb_name], rotations, None, 0.08, ref.get_input_embeddings().weight.data.to(DEV))
        with torch.no_grad(), exact_matmul():
            apart = float((table.matrix() - sp[emb_name].data).norm() / sp[emb_name].data.norm())
        if apart < 1e-3:
            sp[emb_name].requires_grad_(True)  # its gradient is handed on to the handles; the matrix itself is rebuilt from them each step
            coded = [m for m in table.parts]
            others += [(m.log_s, args.lr_scale) for m in coded] + [(m.table, args.lr_table) for m in coded if hasattr(m, "table")] + [(p, args.lr_rows) for p in table.rows_kept]
            log(f"  the embedding's handles train: {len(coded)} coded buckets' row scales and tables, {sum(p.numel() for p in table.rows_kept) / 1e6:.2f}M values in rows kept at 16 bits "
                f"(rebuilt from its codes it is {apart:.1e} of its size from the build's)")
        else:
            log(f"  the embedding rebuilt from its codes is {apart:.2e} of its size from the build's: left as it is")
            table = None

    draw = torch.Generator().manual_seed(1)
    n_m = round(args.batch * args.math_share)

    def backward(weigh_too=False):
        """One batch's weighted KL from the original at the positions read, and every trained tensor's gradient. With
        weigh_too (--running-stats) its windows' inputs join each stage's statistics, in the mix the first were taken in."""
        if table is not None:
            with torch.no_grad(), exact_matmul():  # the embedding the format decodes, not TF32's 2e-4 from it
                sp[emb_name].data.copy_(table.matrix())
        rows = torch.cat([torch.randint(n_math_pool, (n_m,), generator=draw), n_math_pool + torch.randint(n_text, (args.batch - n_m,), generator=draw)])
        weight, loss = float(sel_w[rows].sum()), 0.0
        for part in rows.chunk(args.split):
            if weigh_too:
                wt = weight_of(part) / unit * torch.where(part < n_math_pool, H_MATH / args.math_share, (1 - H_MATH) / (1 - args.math_share))[:, None]
                now["w"], now["seen"] = wt.reshape(-1, 1).to(DEV), now["seen"] + float(wt.sum())
            hid = owner_s(input_ids=pool[part].to(DEV), use_cache=False).last_hidden_state
            now["w"] = None
            ls = F.log_softmax(head_s(at(hid, sel_pos[part].to(DEV))).float(), -1).flatten(0, 1)
            w = sel_w[part].to(DEV).flatten()
            piece = (topk_kl(top_lp[part].to(DEV).flatten(0, 1), top_id[part].to(DEV).long().flatten(0, 1), tail[part].to(DEV).flatten(), ls) * w).sum() / weight
            piece.backward()
            loss += piece.item()
            del hid, ls, piece
        if table is not None:
            table.take_gradient(sp[emb_name].grad)
            sp[emb_name].grad = None
            for m in table.parts:  # the codes stay: their straight-through gradient is not used
                if hasattr(m, "z"):
                    m.z.grad = None
        return loss

    trained = [(sp[n], args.lr) for n in names] + others
    square = [0.0] * len(trained)
    for _ in range(4):
        backward()
        for i, (p, _) in enumerate(trained):
            square[i] += float(p.grad.pow(2).mean()) / 4
            p.grad = None
    eps = [max(args.eps_rms * s ** 0.5, 1e-12) if args.eps_rms else 1e-8 for s in square]
    log(f"gradient RMS at the start: {min(square[: len(names)]) ** 0.5:.2e} to {max(square[: len(names)]) ** 0.5:.2e} across matrices"
        + (f", {min(square[len(names):]) ** 0.5:.2e} to {max(square[len(names):]) ** 0.5:.2e} across the other tensors" if others else "") + f" ({time.time() - t0:.0f}s)")
    opt = torch.optim.Adam([{"params": [p], "eps": e, "lr": lr, "base": lr} for (p, lr), e in zip(trained, eps)], betas=(args.momentum, 0.99))
    if args.profile:
        def timed(f, n=5):
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(n):
                f()
            torch.cuda.synchronize()
            return (time.time() - t) / n

        def one():
            opt.zero_grad(set_to_none=True)
            backward()
        if table is not None:
            with torch.no_grad():
                fast = timed(table.matrix)
                with exact_matmul():
                    exact = timed(table.matrix)
                g = torch.randn_like(sp[emb_name])
                log(f"profile: the embedding rebuilt in {fast:.3f} s with TF32, {exact:.3f} s without; its gradient handed on in {timed(lambda: table.take_gradient(g)):.3f} s")
                del g
        for split in (1, 2, 1, 2):
            args.split = split
            log(f"profile: a batch's gradient with its windows in {split}: {timed(one):.3f} s")
        args.split = 1
        opt.zero_grad(set_to_none=True)
        rules = {"transformers' chunk rule": fast_delta.original, "the one-solve rule": fast_delta.chunk_rule}
        with torch.no_grad():  # the build under each rule on 4 windows: the same model to rounding?
            lp = []
            for rule in rules.values():
                fast_delta.M.torch_chunk_gated_delta_rule = rule
                hid = owner_s(input_ids=pool[:4].to(DEV), use_cache=False).last_hidden_state
                lp.append((hid, F.log_softmax(head_s(hid[:, -128:]).float(), -1)))
            (h_a, l_a), (h_b, l_b) = lp
            log(f"profile: the build under the one-solve rule against transformers': hidden states {float((h_b - h_a).norm() / h_a.norm()):.1e} of their size apart, "
                f"KL {float((l_a.exp() * (l_a - l_b)).sum(-1).mean()):.1e} at the last 128 positions of 4 windows")
            del lp, h_a, l_a, h_b, l_b, hid
        for name, rule in list(rules.items()) * 2:
            fast_delta.M.torch_chunk_gated_delta_rule = rule
            log(f"profile: a batch's gradient under {name}: {timed(one):.3f} s")
        log(f"profile: Adam's step {timed(opt.step):.3f} s; the pull {timed(lambda: [sp[n].data.lerp_(P[n], 1e-9) for n in names]):.3f} s; "
            f"{torch.cuda.max_memory_allocated() / 1e9:.0f} GB peak")
        res["profile"] = True
        save()
        log("=> profile done")
        return
    turn = [{n for n in names if (layer_of(n) - lo) % args.project_groups == g} for g in range(args.project_groups)]
    running, t_train = None, 0.0
    forget = 1 - 1 / args.project_every  # --running-stats: what a stage's statistics keep of themselves each step
    b2 = opt.defaults["betas"][1]
    mean_g = {n: torch.zeros_like(sp[n].data) for n in names} if args.gate_pull else {}  # each weight's mean gradient over the steps Adam's mean square covers
    res["gate"] = []
    for step in range(1, args.steps + 1):
        t_s = time.time()
        late = max(0, step - (args.steps - args.settle)) / max(1, args.settle)  # 0 until the last --settle steps, then up to 1
        for g in opt.param_groups:
            g["lr"] = g["base"] * min(1.0, step / max(1, args.warmup)) * (1 - late)
        beta = args.beta * (args.beta_end / args.beta) ** late
        opt.zero_grad(set_to_none=True)
        if args.running_stats:
            for H in Hs.values():
                H.mul_(forget)
            now["seen"] *= forget
        loss = backward(args.running_stats)
        opt.step()
        with torch.no_grad():
            # --gate-pull, once both means have forgotten their start (three of their spans) and until the weights settle
            gated = args.gate_pull and late == 0 and step > 3 / (1 - b2)
            count = args.gate_pull and step % args.project_every == 0
            past = dict.fromkeys((0.02, 0.05, GATE, 0.25, 0.5), 0)
            free = 0.0
            for n in names:
                if not args.gate_pull:
                    sp[n].data.lerp_(P[n], beta)
                    continue
                mean_g[n].mul_(b2).add_(sp[n].grad, alpha=1 - b2)
                steady = mean_g[n].square().div_(opt.state[sp[n]]["exp_avg_sq"].clamp_min(1e-30)).div_(1 - b2 ** step)  # of the mean square, the mean's share
                if count:
                    for t in past:
                        past[t] += int((steady > t).sum())
                    free += float(steady.div(GATE).clamp_(max=1).sum())
                sp[n].data.lerp_(P[n], steady.div_(GATE).neg_().add_(1).clamp_(min=0).mul_(beta) if gated else beta)
                del steady
            if count:
                res["gate"].append({"step": step, "gated": bool(gated), "pull given up": free / n_weights, **{f"share past {t}": c / n_weights for t, c in past.items()}})
                log(f"  step {step}: weights pushed one way: {past[0.02] / n_weights:.3%} past 0.02, {past[0.05] / n_weights:.3%} past 0.05, {past[GATE] / n_weights:.4%} past {GATE} "
                    f"(not pulled{'' if gated else ' once the gate is on'}), {past[0.5] / n_weights:.5%} past 0.5; {free / n_weights:.2%} of the pull given up")
        running = loss if running is None else 0.98 * running + 0.02 * loss
        assert math.isfinite(running), f"the loss at step {step} is {loss}"
        t_train += time.time() - t_s
        if step in (1, 5, 25) or step % 100 == 0:
            mem = f", {torch.cuda.max_memory_allocated() / 1e9:.0f} GB peak" if DEV == "cuda" else ""
            log(f"  step {step}: train loss {running:.5f}, beta {beta:.4g}, {t_train / step:.2f} s/step ({args.batch * args.len / (t_train / step):.0f} tokens/s){mem} ({time.time() - t0:.0f}s)")
        if step < args.steps and step % args.project_every == 0:
            opt.zero_grad(set_to_none=True)
            t_p, g = time.time(), (step // args.project_every - 1) % args.project_groups
            before = torch.cuda.memory_allocated() if DEV == "cuda" else 0
            fresh = bool(args.rebuild_at) and args.rebuild_at <= step < args.rebuild_at + args.project_every * args.project_groups  # each group once
            again, changed, moved, stats_apart = project(turn[g], args.project_share, fresh)
            torch.cuda.empty_cache() if DEV == "cuda" else None
            left = (torch.cuda.memory_allocated() - before) / 1e9 if DEV == "cuda" else 0.0
            # a projection replaces tensors by ones of the same size: anything it leaves behind adds up over the run (2026-11-10:
            # each rounding left its paths on the GPU, 21 GB for 16 layers, which no 8-layer smoke test of one cycle could show)
            assert left < 1.0, f"the projection at step {step} left {left:.1f} GB more on the GPU than it found"
            res["projections"].append({"step": step, "group": g, "seconds": round(time.time() - t_p), "entries rounded again": again, "of them changed": changed,
                                       "entries changed since their last projection": again * changed, "stored forms moved": moved, "gap": gap(), "train loss": running,
                                       "input statistics from the first": stats_apart, "from the original's weights": fresh})
            how = "rounded from the original's weights" if fresh else "projected"
            log(f"  step {step}: group {g} {how} in {time.time() - t_p:.0f}s: {again:.0%} of its entries rounded again, {changed:.2%} of them changed; "
                f"its stored forms moved {moved:.4f}; gap {gap():.4f}" + (f"; its input statistics {stats_apart:.3f} from the first taken" if stats_apart is not None else ""))
            save()
        if step < args.steps and step % args.eval_every == 0:
            opt.zero_grad(set_to_none=True)
            check(step, beta, running)
    del opt, mean_g
    torch.cuda.empty_cache()

    final_gap = gap()
    with torch.no_grad():
        for n in names:
            sp[n].data = P[n]  # the stored form as last projected: a matrix the format stores, whenever that was
            sp[n].requires_grad_(False)
        if table is not None:
            with exact_matmul():
                sp[emb_name].data.copy_(table.matrix())
        some = names[:: max(1, len(names) // 4)][:4]  # projected again in full: nothing may change
        changed = moved = size = 0.0
        for n in some:
            new, st = rounded(sp[n].data, R[n], groups[n], scale[n].clone(), environment(stage_of[n]))
            changed += int((st != states[n]).sum())
            moved, size = moved + float((new - sp[n].data).pow(2).sum()), size + float(sp[n].data.pow(2).sum())
    res["projected again"] = {"gap before taking the stored forms": final_gap, "matrices": len(some), "entries that changed": changed / sum(sp[n].numel() for n in some),
                              "stored forms moved": (moved / size) ** 0.5}
    check(args.steps, args.beta_end, running, same=True)
    res["fine-tuned"] = res["rounds"][-1]["continuous"]
    for key in ("the build", "fine-tuned"):
        log(f"{key}: {show(res[key])}")
    a, b = res["fine-tuned"]["math"]["digit KL summed by window"], res["the build"]["math"]["digit KL summed by window"]
    res["held-out math windows with less digit damage than the build"] = [sum(x < y for x, y in zip(a, b)), len(a)]
    log(f"less digit damage than the build on {sum(x < y for x, y in zip(a, b))} of {len(a)} held-out math windows")
    log(f"projected again: {json.dumps(res['projected again'])}")
    if args.save:
        base = torch.load(args.base_dense, map_location="cpu", mmap=True)
        params = dict(base["params"])
        for n in names:
            params[n] = sp[n].detach().cpu()
        for p, _ in others if args.train_norms else []:
            for n in [k for k, v in sp.items() if v is p]:
                params[n] = p.detach().cpu()
        if table is not None:
            for n in [k for k in params if "embed_tokens" in k or "lm_head" in k] or [emb_name]:
                params[n] = sp[emb_name].detach().cpu()
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"params": params, "text weights": base.get("text weights"), "fine-tuned from": args.base_dense, "layers": [lo, hi]}, args.save)
        log(f"saved the model ({len(params)} tensors, {len(names)} matrices trained) to {args.save}")
    save()
    log(f"=> fine-tuning done ({res['seconds']}s)")


if __name__ == "__main__":
    main()
