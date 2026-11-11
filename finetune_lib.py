"""Parts of the fine-tuning (finetune.py): a row-trellis matrix as the build stores it and the quantizer that rounds real
weights back onto that form, the embedding with its codes fixed, the text the loss reads, and the loss itself (KL from the
original over its top tokens and one bucket for the rest)."""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

import compiler_arch as A
from opgraph import codes as C
from quantize import DEV
from token_kinds import kinds_of


class FastRHT:
    """X R and X Rᵀ for R = rht(n, seed), in about n·(a + b) operations per row instead of n², through H = (A ⊗ B) / √n."""

    def __init__(self, n, seed):
        self.n = n
        A_, B_ = C.hadamard_factors(n)
        self.A, self.B = A_.to(DEV, torch.float32), B_.to(DEV, torch.float32)
        s = torch.where(torch.rand(n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) < 0.5, -1.0, 1.0)
        self.s = s.to(DEV, torch.float32)
        self.shape = (len(A_), len(B_))

    def forward(self, X):
        return (self.A.T @ X.reshape(-1, *self.shape) @ self.B).reshape(X.shape) * self.s / math.sqrt(self.n)

    def back(self, X):
        return (self.A @ (X * self.s).reshape(-1, *self.shape) @ self.B.T).reshape(X.shape) / math.sqrt(self.n)


class Coded(torch.nn.Module):
    def __init__(self, entry, rotations, W=None, frac=0.08, col=False):
        super().__init__()
        idx = entry["codes"].to(DEV)
        levels = entry["levels"].to(DEV, torch.float32)
        self.register_buffer("first", idx)
        self.register_buffer("levels", levels)
        self.register_buffer("mid", (levels[1:] + levels[:-1]) / 2)
        self.top = 1.5 * float(levels[-1])
        n = idx.shape[1]
        self.R = rotations.setdefault((n, entry["in_seed"]), FastRHT(n, entry["in_seed"]))
        S = entry["scales"].to(DEV, torch.float32).clamp_min(1e-12)
        self.log_s = torch.nn.Parameter(S.log())
        z = levels[idx.long()]
        if W is not None:  # the original weight, inside this code's cell: frac of the cell's width from either edge
            lo = torch.cat([levels[:1] * 1.5, self.mid])[idx.long()]
            hi = torch.cat([self.mid, levels[-1:] * 1.5])[idx.long()]
            margin = (hi - lo) * frac
            z = torch.min(torch.max(self.R.back(W.to(DEV, torch.float32)) / S[:, None], lo + margin), hi - margin)
        self.z = torch.nn.Parameter(z)
        self.log_c = torch.nn.Parameter(torch.zeros(n, device=DEV)) if col else None  # a scale per input column: 2 bytes each

    def codes(self):
        return torch.bucketize(self.z.detach(), self.mid)

    def matrix(self):
        W = self.R.forward(self.log_s.exp()[:, None] * self.levels[self.codes()])
        return W if self.log_c is None else W * self.log_c.exp()[None, :]

    @torch.no_grad()  # gradients by hand: tracking them would chain every step's graph onto what keeps them
    def take_gradient(self, dW):
        """The straight-through gradient from ∂loss/∂W, by hand: W = U R with U = diag(S) z and z read at its level, so nothing about z or
        R stays in memory through the forward pass."""
        if self.log_c is not None:
            c = self.log_c.exp()
            self.log_c.grad = (dW * self.R.forward(self.log_s.exp()[:, None] * self.levels[self.codes()])).sum(0) * c
            dW = dW * c[None, :]
        g = self.R.back(dW) * self.log_s.exp()[:, None]
        self.log_s.grad = (g * self.levels[self.codes()]).sum(1)
        self.z.grad = g


L_TRELLIS = 12  # state bits of every trellis in the builds


def path_bits(states):
    """k, the bits per entry of a row trellis's paths: each state is the one before shifted k bits, from state 0."""
    s = states.long()
    for k in range(1, L_TRELLIS):
        if bool((s[:, 0] < (1 << k)).all()) and bool(((s[:, 1:] >> k) == (s[:, :-1] & ((1 << (L_TRELLIS - k)) - 1))).all()):
            return k
    raise ValueError("states that are not paths through a row trellis")


class TrellisCoded(torch.nn.Module):
    """A trellis-coded matrix: W = diag(S) T[state] R. The matrix's own table T of 4,096 values (16 bits each, 8 KB a matrix)
    trains, with the row scales; with group, also a scale for each run of that many entries along a row (16 bits each, starting
    at 1). With move, each entry also sums its gradient, and repick() moves each row along it; otherwise each entry's state stays."""

    def __init__(self, entry, rotations, move=False, group=0, kron=None):
        super().__init__()
        self.kron = [kron] if kron is not None else []  # (in a list: shared by the stage, trained and kept once, not as this module's own)
        self.register_buffer("states", entry["states"].to(DEV))
        n = self.states.shape[1]
        self.R = rotations.setdefault((n, entry["in_seed"]), FastRHT(n, entry["in_seed"]))
        self.log_s = torch.nn.Parameter(entry["scales"].to(DEV, torch.float32).clamp_min(1e-12).log())
        self.table = torch.nn.Parameter(entry["table"].to(DEV, torch.float32).clone())
        self.group = group
        self.log_g = torch.nn.Parameter(torch.zeros(len(self.states), -(-n // group), device=DEV)) if group else None
        self.k = None
        if move:
            self.k = path_bits(self.states)
            self.register_buffer("acc", torch.zeros(self.states.shape, device=DEV))
            self.first = entry["states"].clone()  # on the CPU: what moved since the start

    def scales(self):
        """Each entry's scale: its row's, times its group's when there are group scales."""
        s = self.log_s.exp()[:, None]
        return s if self.log_g is None else s * self.log_g.exp().repeat_interleave(self.group, dim=1)[:, : self.states.shape[1]]

    def matrix(self):
        Q = self.scales() * self.table[self.states.long()]
        return self.R.forward(self.kron[0].right(Q) if self.kron else Q)

    @torch.no_grad()  # gradients by hand: tracking them would chain every step's graph onto what keeps them
    def take_gradient(self, dW):
        g = self.R.back(dW)
        if self.kron:
            g = self.kron[0].back(self.scales() * self.table[self.states.long()], g)
        g = g * self.scales()
        idx = self.states.long()
        gv = g * self.table[idx]
        self.log_s.grad = gv.sum(1)
        if self.log_g is not None:
            self.log_g.grad = F.pad(gv, (0, self.log_g.shape[1] * self.group - gv.shape[1])).view(len(gv), -1, self.group).sum(-1)
        self.table.grad = torch.zeros_like(self.table).index_add_(0, idx.flatten(), g.flatten())
        if self.k is not None:
            self.acc += g

    def nearest(self, eta, back_bytes):
        """Each row's path for one proximal step on the summed gradient g: argmin g·(q − c) + |q − c|²/2η over paths q, which
        is the path nearest c − η·g (c the current values; Adam would move every entry about equally, signal or noise)."""
        T, R = self.states.shape[1], 1 << (L_TRELLIS - self.k)
        chunk = max(1, int(back_bytes // (T * R)))  # Viterbi's back-pointers are T × rows × 2^(L−k) bytes
        c = self.table.detach()[self.states.long()]
        return C.viterbi(c - eta * self.acc, self.table.detach(), L_TRELLIS, self.k, chunk=chunk, start=torch.zeros(len(c), dtype=torch.long)).to(DEV, torch.int16)

    def repick(self, eta, back_bytes):
        """Takes the step; the share of entries whose state changed."""
        new = self.nearest(eta, back_bytes)
        changed = float((new != self.states).float().mean())
        self.states.copy_(new)
        self.acc.zero_()
        return changed


class CodedTable(torch.nn.Module):
    """The embedding by frequency buckets: each bucket's rows a Coded matrix, or left real-valued if the plan keeps them at 16 bits."""

    def __init__(self, entry, rotations, W=None, frac=0.08, E=None, move=False, group=0):
        super().__init__()
        self.parts = torch.nn.ModuleList()
        self.rows_kept = torch.nn.ParameterList()
        self.order = []
        rows = []
        for b in entry["buckets"]:
            r = b["rows"].to(DEV)
            rows.append(r)
            if "codes" in b or "states" in b:
                self.parts.append(TrellisCoded(b, rotations, move, group) if "states" in b else Coded(b, rotations, None if W is None else W[r], frac))
                self.order.append(("coded", len(self.parts) - 1))
            else:
                self.rows_kept.append(torch.nn.Parameter(E[r].to(DEV, torch.float32).clone()))
                self.order.append(("kept", len(self.rows_kept) - 1))
        self.register_buffer("inverse", torch.argsort(torch.cat(rows)))

    def matrix(self):
        return torch.cat([self.parts[k].matrix() if kind == "coded" else self.rows_kept[k] for kind, k in self.order]).index_select(0, self.inverse)

    @torch.no_grad()  # gradients by hand: tracking them would chain every step's graph onto what keeps them
    def take_gradient(self, dW):
        lo = 0
        cut = dW.index_select(0, torch.argsort(self.inverse))
        for kind, k in self.order:
            size = (self.parts[k].log_s if kind == "coded" else self.rows_kept[k]).shape[0]
            if kind == "coded":
                self.parts[k].take_gradient(cut[lo : lo + size])
            else:
                self.rows_kept[k].grad = cut[lo : lo + size].clone()
            lo += size


def log(msg):
    print(msg, flush=True)


class exact_matmul:
    """TF32 off inside: with it the rotation returns a stored matrix 2e-4 off."""

    def __enter__(self):
        self.was, torch.backends.cuda.matmul.allow_tf32 = torch.backends.cuda.matmul.allow_tf32, False

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.was


class RowPath:
    """The builds' row trellis (codes.RowTrellisCode with --trellis-rows 16,8: each row one path from state 0, 16 columns
    settled at a time with 8 ahead), with the table a build stored; `paths` holds the states of the last sweep's blocks."""

    def __init__(self, table, k, width=16, look=8):
        self.values, self.k, self.width, self.look = table.double().cpu(), k, width, look

    def step(self, j, X, ahead):
        if j == 0:
            self.state, self.paths = torch.zeros(len(X), dtype=torch.long, device=X.device), []
        path = C.viterbi(torch.cat([X, ahead], 1), self.values, L_TRELLIS, self.k, start=self.state)[:, : X.shape[1]]
        self.state = path[:, -1]
        self.paths.append(path)
        return self.values.to(X.device, X.dtype)[path]


row_code = RowPath


def rotated_moment(H, R):
    """The input statistics in the rotated basis, R H Rᵀ: R.back takes a matrix's rows there (W → W Rᵀ), and its error is then
    weighed by this (as codes.ldlq_words rotates H)."""
    with exact_matmul():
        Hr = R.back(R.back(H.float()).T).double()
    return (Hr + Hr.T) / 2


def rounded(W, R, groups, scale, env, rows=None):
    """P(W) as a build rounds, and each entry's state in its group's table: block LDLQ under env (the stage's rotated input
    statistics: its .H, the factor .U and .dead as frame.Environment's), each row group through its row trellis in units of
    `scale`, which is then refitted under the same statistics in place. groups: (rows, bits per entry, RowPath) for each row
    group. With `rows` (row numbers) only those are rounded (rows do not interact), and their stored rows and states come
    back in that order."""
    with exact_matmul():
        X = R.back(W if rows is None else W[rows]).double()
        where = None
        if rows is not None:
            where = torch.full((W.shape[0],), -1, dtype=torch.long, device=W.device)
            where[rows] = torch.arange(len(rows), device=W.device)
        Q = torch.empty_like(X)
        states = torch.empty(X.shape, dtype=torch.int16)
        for g_rows, k, code in groups:
            mine = g_rows if where is None else g_rows[where[g_rows] >= 0]
            if not len(mine):
                continue
            at = mine if where is None else where[mine]
            x, S, got = X[at], scale[mine].double(), torch.empty_like(X[at])

            def project(j, cols, T):
                q = code.step(j, cols / S[:, None], T[:, j + cols.shape[1] : j + cols.shape[1] + code.look] / S[:, None])
                got[:, j : j + cols.shape[1]] = q
                return q * S[:, None]

            C.sweep(x, env, project, width=code.width)
            gH = got @ env.H
            scale[mine] = ((x * gH).sum(1) / (got * gH).sum(1).clamp_min(1e-30)).clamp_min(1e-12).float()
            Q[at] = got
            states[at.cpu()] = torch.cat(code.paths, 1).to(torch.int16).cpu()
            code.paths = []  # else each group keeps its sweep on the GPU, 12 bytes an entry: 21 GB for 16 layers
        units = scale if rows is None else scale[rows]
        return R.forward(units[:, None] * Q.float()), states


def owner_of_layers(model):
    """The module that runs the decoder layers (input_ids or inputs_embeds in, final hidden states out), and its layers."""
    path, layers = A.decoder_layers(model)
    return model.get_submodule(path.rsplit(".", 1)[0]), path.rsplit(".", 1)[1], layers


def shingles(w, n=32):
    """A hash of every run of n tokens in each row."""
    mult = torch.tensor([pow(1000003, j, 2**31 - 1) for j in range(n)])
    return (w.unfold(1, n, 1) * mult).sum(-1)


def pieces(w, T):
    """Windows cut into rows of T tokens; shorter windows are joined in order to make them."""
    L = w.shape[1]
    if L >= T:
        return torch.cat([w[:, i : i + T] for i in range(0, L - T + 1, T)])
    assert T % L == 0, f"windows of {L} do not tile {T}"
    return w[: len(w) // (T // L) * (T // L)].reshape(-1, T)


def topk_kl(lp_t, idx, tail_t, ls):
    """KL(original ‖ student) at each position with the original's distribution coarsened to its top entries (their log
    probabilities lp_t at vocabulary ids idx) and one bin for the rest (probability tail_t); ls: the student's log probabilities
    over the vocabulary. A lower bound on the full KL that meets it as the top entries take all the mass."""
    ls_k = ls.gather(1, idx)
    pt = lp_t.exp()
    tail_s = (1 - ls_k.exp().sum(1)).clamp_min(1e-9)
    return (pt * (lp_t - ls_k)).sum(1) + tail_t * (tail_t.clamp_min(1e-12).log() - tail_s.log())


def text_args(ap):
    """The held-out windows, the pool and what the loss reads of it (shared with finetune.py)."""
    ap.add_argument("--math-cache", required=True, help="windows of worked solutions (checked_solutions.py)")
    ap.add_argument("--held-math", type=int, default=64, help="windows held out for scoring: the last ones of the order builds calibrate in")
    ap.add_argument("--more-math", default="", help="file:key+key: more worked solutions for the pool, a dictionary of window tensors")
    ap.add_argument("--more-text", default="", help="file:key+key: other text for the pool, the same")
    ap.add_argument("--text-windows", type=int, default=8000, help="text-mix windows in the pool")
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--held-len", type=int, default=1024)
    ap.add_argument("--token-floor", type=float, default=0.1, help="the loss weight of positions that do not predict a digit (digits weigh 1), as the builds' calibration")
    ap.add_argument("--positions", type=int, default=128, help="positions of each pool window the loss reads")
    ap.add_argument("--digit-cap", type=int, default=96, help="of those, at most this many digit-predicting positions")
    ap.add_argument("--top", type=int, default=64, help="the original's top predictions kept at each of those positions")


def pool_and_held(args, tok, ref):
    """The held-out windows and the kind of token each position predicts; the pool, how many of its first windows are worked
    solutions, which of its positions predict a digit, the positions the loss reads and their weights; pool windows dropped."""
    # the text. Held out: the last relabeled windows of the order builds calibrate in, and text-mix evaluation windows.
    # The pool: the other relabeled windows, more worked solutions and other text, all in rows of --len
    sol = torch.load(args.math_cache)
    order = torch.randperm(len(sol), generator=torch.Generator().manual_seed(0))  # quantize.py --calib-math takes the front of this
    mix = A.TextMix(tok, "qwen35")
    held = {"math": A.with_bos(pieces(sol[order[-args.held_math :]], args.held_len), tok, ref),
            "text": A.with_bos(torch.cat(list(mix.windows("eval", 4, args.held_len, seed=3).values())), tok, ref)}
    held_kinds = {k: kinds_of(v, tok) for k, v in held.items()}

    def more(spec):
        if not spec:
            return torch.empty(0, args.len, dtype=torch.long)
        f, keys = spec.split(":")
        d = torch.load(f)
        return torch.cat([pieces(d[k], args.len) for k in keys.split("+")])

    math_pool = torch.cat([pieces(sol[order[: -args.held_math]], args.len), more(args.more_math)])
    # a held-out window's own text: runs of 32 tokens found in at most two held-out windows (the instruction is in all of them)
    # that the pool repeats: its windows with one are dropped
    runs, in_windows = torch.unique(torch.cat([torch.unique(r) for r in shingles(sol[order[-args.held_math :]])]), return_counts=True)
    pool_runs, in_pool = torch.unique(torch.cat([torch.unique(r) for r in shingles(math_pool)]), return_counts=True)
    own = runs[in_windows <= 2]
    own = own[torch.isin(own, pool_runs[in_pool <= 3])]  # and in at most three pool windows: a formulaic opening is in many
    leak = torch.isin(shingles(math_pool), own).any(1)
    n_rel = len(pieces(sol[order[: -args.held_math]], args.len))
    log(f"pool: of {n_rel} relabeled windows {int(leak[:n_rel].sum())} share text with the held-out windows, of {len(math_pool) - n_rel} more {int(leak[n_rel:].sum())}")
    math_pool = math_pool[~leak]
    text_pool = torch.cat([mix.calibration(args.text_windows, args.len, seed=1), more(args.more_text)])
    pool = A.with_bos(torch.cat([math_pool, text_pool]), tok, ref)
    n_math_pool = len(math_pool)
    log(f"pool: {n_math_pool} windows of worked solutions ({int(leak.sum())} dropped for sharing text with the held-out windows) and {len(text_pool)} of other text, "
        f"{pool.numel() / 1e6:.2f}M tokens; held out {len(held['math'])} math and {len(held['text'])} text windows of {args.held_len}")

    # the positions the loss reads in each pool window, and their weights: the sum stands for every position's weighted KL
    digit = kinds_of(pool, tok) <= 1
    gen = torch.Generator().manual_seed(0)
    sel_pos = torch.empty(len(pool), args.positions, dtype=torch.long)
    sel_w = torch.empty(len(pool), args.positions)
    for i in range(len(pool)):
        d, o = digit[i].nonzero()[:, 0], (~digit[i]).nonzero()[:, 0]
        nd = min(len(d), args.digit_cap, args.positions)
        no = min(len(o), args.positions - nd)
        nd = min(len(d), args.positions - no)  # fewer others than room: more digits
        pd, po = d[torch.randperm(len(d), generator=gen)[:nd]], o[torch.randperm(len(o), generator=gen)[:no]]
        sel_pos[i] = torch.cat([pd, po])
        sel_w[i] = torch.cat([torch.full((nd,), len(d) / max(1, nd)), torch.full((no,), args.token_floor * len(o) / max(1, no))])
    log(f"  {float(digit.float().mean()):.1%} of pool positions predict a digit; the loss reads {args.positions} positions a window, "
        f"{float((sel_w.sum(1) > 0).float().mean()):.0%} of windows with any weight")
    return held, held_kinds, pool, n_math_pool, digit, sel_pos, sel_w, int(leak.sum())


@torch.no_grad()
def damage_of(ref, student, held, held_kinds, bs=4):
    """The student against the original on the held-out windows: KL at all positions, at positions predicting a digit and at
    digits of computed results, top-1 agreement at digit positions, and each window's summed KL at its digit positions (for
    paired comparisons between sets of weights)."""
    res = {}
    for name, ids in held.items():
        kl_sum = {"KL": 0.0, "digit KL": 0.0, "result-digit KL": 0.0}
        count = dict.fromkeys(kl_sum, 0)
        agree, by_window = 0, []
        for b in range(0, len(ids), bs):
            x = ids[b : b + bs].to(DEV)
            lr_ = F.log_softmax(ref(x, use_cache=False).logits[:, :-1].float(), -1)
            lp = F.log_softmax(student(x, use_cache=False).logits[:, :-1].float(), -1)
            kl = (lr_.exp() * (lr_ - lp)).sum(-1)
            kinds = held_kinds[name][b : b + bs].to(DEV)
            for key, mask in (("KL", torch.ones_like(kinds, dtype=torch.bool)), ("digit KL", kinds <= 1), ("result-digit KL", kinds == 0)):
                kl_sum[key] += float(kl[mask].sum())
                count[key] += int(mask.sum())
            agree += int((lr_.argmax(-1) == lp.argmax(-1))[kinds <= 1].sum())
            by_window += (kl * (kinds <= 1)).sum(1).tolist()
            del lr_, lp, kl
        res[name] = {key: round(kl_sum[key] / max(1, count[key]), 5) for key in kl_sum}
        res[name] |= {"digit agreement": round(agree / max(1, count["digit KL"]), 5), "digit positions": count["digit KL"],
                      "digit KL summed by window": [round(v, 4) for v in by_window]}
    return res
