"""A DSpine drafter (SpecForge's DSpineDraftModel, as drafter/ trains it) in MLX, and generation with it.

The drafter reads the target's hidden states after a few of its layers (`taps`) for every token the target has taken in,
and proposes the tokens after the next one in a single pass: a block is the anchor (the target's own latest token) and
size - 1 mask tokens at the positions after it. Six layers attend to the context and, causally, to the block; after each
layer a position's state is nudged by a message from the position before it (the anchor's token code for the first, the
neighbour's own readout for the rest). The target's output head turns the final states into tokens.

    model, tokenizer, block, drafter = load(packed, folder)
    tokens, stats = generate(model, block, drafter, prompt_ids, max_tokens)

Each round is one evaluation: the draft, its tokens, the target's check of [anchor, proposals] and the check's own next
tokens. The leading proposals that equal the target's tokens are kept, with the token the check gives after them, so the
text is the target's own greedy text.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache


OVERLAP = 1   # start the GPU on each part of a round while Python lists the next part's launches (0: list the whole round first)


def drafter_taps(path):
    """The target layers a drafter folder's model reads (their outputs)."""
    return tuple(json.loads((Path(path) / "config.json").read_text())["dflash_config"]["target_layer_ids"])


def _rms(x, w, eps):
    return mx.fast.rms_norm(x, w, eps)


_DENSE8 = """
    // y[t][r] = scale[r] * sum_c w[r][c] x[t][c] for 8 tokens, w in 8-bit integers (rows x N, row-major).
    // The layout of block_q35's product: a simdgroup takes 8 RB rows and CS of them share a set of rows, each walking
    // its part of the columns 32 at a time as four 8 x 8 tiles; a thread reads its eight neighbouring weights of a row
    // in two loads, puts them in its elements of each tile's matrix, and the matrix unit multiplies by the inputs' tile.
    const uint lane = thread_index_in_simdgroup, simd = simdgroup_index_in_threadgroup, tg = threadgroup_position_in_grid.x;
    const uint qid = lane / 4u, fm = (qid & 4u) + ((lane / 2u) % 4u), fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
    const uint q = fn / 2u, xcol = 8u * (fm / 2u) + (fm % 2u);
    const uint cp = simd % CS, rs = simd / CS, steps = N / 32u / CS;
    const uint rbase = (tg * (SG / CS) + rs) * (8u * RB), r0 = rbase + fm;
    const bool live = rbase < ROWS;
    threadgroup float red[SG * RB * 64];
    float2 part[RB];
    for (uint i = 0; i < RB; i++) part[i] = float2(0.0f);
    if (live) {
        const device int8_t* p[RB];
        for (uint i = 0; i < RB; i++) p[i] = w + (ulong)min(r0 + 8u * i, ROWS - 1u) * N + 8u * q;
        for (uint s = cp * steps; s < (cp + 1u) * steps; s++) {
            simdgroup_float8x8 Xm[4];
            #pragma unroll
            for (uint j = 0; j < 4u; j++)                         // tokens fn and fn + 1 of this thread's input columns
                reinterpret_cast<thread float2&>(Xm[j].thread_elements()) = float2((float)x[(ulong)fn * N + 32u * s + xcol + 2u * j], (float)x[(ulong)(fn + 1u) * N + 32u * s + xcol + 2u * j]);
            #pragma unroll
            for (uint i = 0; i < RB; i++) {
                const device int8_t* pc = p[i] + 32u * s;
                const float4 lo = float4(*(const device packed_char4*)pc), hi = float4(*(const device packed_char4*)(pc + 4));
                simdgroup_float8x8 Am, Em;
                reinterpret_cast<thread float2&>(Am.thread_elements()) = lo.xy;
                simdgroup_multiply(Em, Am, Xm[0]);
                reinterpret_cast<thread float2&>(Am.thread_elements()) = lo.zw;
                simdgroup_multiply_accumulate(Em, Am, Xm[1], Em);
                reinterpret_cast<thread float2&>(Am.thread_elements()) = hi.xy;
                simdgroup_multiply_accumulate(Em, Am, Xm[2], Em);
                reinterpret_cast<thread float2&>(Am.thread_elements()) = hi.zw;
                simdgroup_multiply_accumulate(Em, Am, Xm[3], Em);
                part[i] += reinterpret_cast<thread float2&>(Em.thread_elements());
            }
        }
    }
    for (uint i = 0; i < RB; i++) {
        red[(simd * RB + i) * 64u + fm * 8u + fn] = part[i].x;
        red[(simd * RB + i) * 64u + fm * 8u + fn + 1u] = part[i].y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live || cp != 0u) return;
    for (uint i = 0; i < RB; i++) {
        const uint row = r0 + 8u * i;
        if (row >= ROWS) continue;
        float2 d = float2(0.0f);
        for (uint c = 0; c < CS; c++) {
            const uint at = ((simd + c) * RB + i) * 64u + fm * 8u + fn;
            d += float2(red[at], red[at + 1u]);
        }
        y[(ulong)fn * ROWS + row] = d.x * scale[row];
        y[(ulong)(fn + 1u) * ROWS + row] = d.y * scale[row];
    }
"""
_dense8 = mx.fast.metal_kernel(name="lamq_dense8", input_names=["x", "w", "scale"], output_names=["y"], source=_DENSE8,
                               header="#include <metal_simdgroup_matrix>\n")


class Lin:
    """x -> x @ W.T for W (out, in): dense in `dtype`; MLX's own quantized product at `bits` (groups of 64); or, with
    int8, each row as 8-bit integers and a scale, through this module's kernel, 8 vectors at a time (the draft's block).
    The dense products are bound by reading the weights (1.16 GB at 16 bits for the draft's 42); MLX's 4-bit and 8-bit
    products measured no faster than dense."""

    # (rows, columns) -> (rb, sg, cs) for the 8-bit kernel, chosen on the draft's own time:
    # the six layers take 3.18 ms dense at 16 bits and 2.3 to 2.5 ms with these
    SHAPE = {(2560, 9216): (1, 8, 8), (9216, 2560): (1, 8, 8), (2560, 4096): (1, 32, 8), (4096, 2560): (2, 8, 8)}

    def __init__(self, w, dtype, bits=None, int8=False):
        self.dtype, self.bits, self.q = dtype, bits, None
        if int8:
            w = w.astype(mx.float32)
            self.rows, self.n = w.shape
            assert self.n % 32 == 0
            self.scale = mx.maximum(mx.abs(w).max(axis=1), 1e-12) / 127.0
            self.q = mx.round(w / self.scale[:, None]).astype(mx.int8)
        elif bits:
            self.w, self.scales, self.biases = mx.quantize(w.astype(mx.float32), group_size=64, bits=bits)
        else:
            self.w = w.astype(dtype)
        mx.eval([v for v in self.__dict__.values() if isinstance(v, mx.array)])

    def shape(self):
        if (self.rows, self.n) in self.SHAPE:
            return self.SHAPE[(self.rows, self.n)]
        rb, sg, cs = 2, 32, 1      # column parts: enough that the launch has about 40 threadgroups or more
        while cs < 16 and self.rows / (8 * rb * sg / cs) < 40 and (self.n // 32) % (2 * cs) == 0:
            cs *= 2
        return rb, sg, cs

    def __call__(self, x):
        if self.q is not None:
            assert x.ndim == 2 and x.shape[0] % 8 == 0, "the 8-bit kernel takes the draft's vectors 8 at a time"
            rb, sg, cs = self.shape()
            out = [_dense8(inputs=[x[i : i + 8], self.q, self.scale], template=[("N", self.n), ("ROWS", self.rows), ("SG", sg), ("RB", rb), ("CS", cs)],
                           grid=(-(-self.rows // (8 * rb * sg // cs)) * sg * 32, 1, 1), threadgroup=(sg * 32, 1, 1),
                           output_shapes=[(8, self.rows)], output_dtypes=[self.dtype])[0] for i in range(0, x.shape[0], 8)]
            return out[0] if len(out) == 1 else mx.concatenate(out)
        if self.bits:
            return mx.quantized_matmul(x, self.w, self.scales, self.biases, transpose=True, group_size=64, bits=self.bits)
        return x @ self.w.T


def _weights(path):
    """The drafter's tensors by name. `path` holds model.safetensors as SpecForge exports it, or the packed file that
    pack_drafter.py writes (artifact.json and weights.bin): its coded matrices are written out dense here, each weight
    its table's value at the row's state times the row's scale, with the input rotation undone."""
    if not (path / "artifact.json").exists():
        return mx.load(str(path / "model.safetensors"))
    import numpy as np
    from compiler_artifact import decode_rotated, read_artifact
    import lamq                      # fast_q35, loaded before any drafter, has put it in sys.modules
    arrays, manifest = read_artifact(path, verify=False)
    w = {}
    for name in manifest["dense"]:
        value = mx.array(arrays[name])
        w[name] = value.view(mx.bfloat16) if manifest["tensors"][name]["dtype"] == "BF16" else value
    for name, meta in manifest["matrices"].items():
        rot = lamq.Rot(np.asarray(arrays[meta["signs"]], dtype=np.float32))
        coded = decode_rotated(arrays, meta)                                   # W Rᵀ, in float32
        rows = [rot.bwd(mx.array(coded[i : i + 8192])).astype(mx.float16) for i in range(0, len(coded), 8192)]   # each row times R
        w[name] = mx.concatenate(rows) if len(rows) > 1 else rows[0]
        mx.eval(w[name])
    return w


class Drafter:
    def __init__(self, path, block, dtype=mx.float16, bits=None, select=False, int8=True):
        """dtype: the drafter's arithmetic. int8: the matrices that only ever multiply the block (q, o and the
        feed-forward's three, 95% of its weights) as 8-bit rows through Lin's kernel; False keeps them dense, or at
        `bits` through MLX's quantized product. select: propose by DSpine's path selection (tokens' second form) and
        not each position's own best token.

        Measured on the M5 Max, 2026-11-11: 8-bit rows take the draft's six layers from
        3.2 to 2.4 ms and propose the reference's tokens as often as dense (176 against 174 of 180; 4.33 against 4.34
        tokens a check on 30 held-out prompts). Path selection adds 3% tokens a check and 0.9 ms a round: 0.99 times
        overall, so it is off."""
        self.path, self.torn = select, False      # torn: the last round had a rejection (generate sets it; "auto" selects only then)
        self.proposal_head = None
        self.explicit_mask = False   # check_drafter sets it: the sliding layers' mask built even where it hides nothing
        path = Path(path)
        c = json.loads((path / "config.json").read_text())
        w = _weights(path)
        self.block, self.table, self.dtype = block, block.table, dtype
        self.H, self.heads, self.kv, self.hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.eps, self.theta = float(c["rms_norm_eps"]), float(c["rope_theta"])
        self.taps = tuple(c["dflash_config"]["target_layer_ids"])
        self.mask_id, self.trained_block = int(c["dflash_config"]["mask_token_id"]), int(c["block_size"])
        self.window = [int(c["sliding_window"]) if t == "sliding_attention" else None for t in c["layer_types"]]
        self.rank, self.top_k = int(c["dspine_config"]["transfer_rank"]), int(c["dspine_config"]["top_k"])
        assert c.get("is_causal") is True and tuple(self.taps) == tuple(block.taps), "the block must tap the drafter's layers"
        g = lambda name: w[name].astype(dtype)
        big = lambda name: Lin(w[name], dtype, bits)
        small = lambda name: Lin(w[name], dtype)
        self.fc, self.hidden_norm, self.norm = big("fc.weight"), g("hidden_norm.weight"), g("norm.weight")
        self.layers = []
        for i in range(c["num_hidden_layers"]):
            p = f"layers.{i}."
            L = {k: g(p + n) for k, n in (("in_norm", "input_layernorm.weight"), ("post_norm", "post_attention_layernorm.weight"),
                                          ("q_norm", "self_attn.q_norm.weight"), ("k_norm", "self_attn.k_norm.weight"))}
            L.update({k: big(p + n) for k, n in (("k", "self_attn.k_proj.weight"), ("v", "self_attn.v_proj.weight"))})    # these also read the context
            L.update({k: Lin(w[p + n], dtype, bits, int8=int8) for k, n in (
                ("q", "self_attn.q_proj.weight"), ("o", "self_attn.o_proj.weight"), ("gate", "mlp.gate_proj.weight"), ("up", "mlp.up_proj.weight"),
                ("down", "mlp.down_proj.weight"))})
            L["readout"], L["write"] = small(f"injection.readouts.{i}.weight"), small(f"injection.writes.{i}.weight")   # the injection's own, per layer
            self.layers.append(L)
        self.message, self.receiver_gate, self.predecessor_gate = small("injection.message.weight"), small("injection.receiver_gate.weight"), small("injection.predecessor_gate.weight")
        self.bias = g("injection.bias")
        self.codes = w["transfer_codes"].astype(mx.float16)       # (vocabulary, rank): only looked up
        self.mask_rows = mx.broadcast_to(self.table(mx.array([self.mask_id])).astype(dtype), (self.trained_block - 1, self.H))
        mx.eval(self.hidden_norm, self.norm, [v for L in self.layers for v in L.values() if isinstance(v, mx.array)], self.bias, self.codes, self.mask_rows)
        self.reset()

    def reset(self):
        self.S = 0
        self.cache = [KVCache() for _ in self.layers]

    def _kv(self, L, x, offset):
        """A layer's keys (normed, rotated to positions offset ..) and values for states x (S, H): (1, kv heads, S, head dim)."""
        S = x.shape[0]
        k = _rms(L["k"](x).reshape(1, S, self.kv, self.hd), L["k_norm"], self.eps).transpose(0, 2, 1, 3)
        v = L["v"](x).reshape(1, S, self.kv, self.hd).transpose(0, 2, 1, 3)
        return mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=offset), v

    def context_features(self, taps):
        """taps: the target's states after each tapped layer, (S, H) each, in the taps' order -> the context (S, H)."""
        return _rms(self.fc(mx.concatenate([t.astype(self.dtype) for t in taps], axis=-1)), self.hidden_norm, self.eps)

    def context(self, taps):
        """Take in the target's states of the next tokens (positions self.S ..)."""
        ctx = self.context_features(taps)
        for L, c, window in zip(self.layers, self.cache, self.window):
            # The first draft cannot see the early keys of a long prompt on sliding layers.
            skip = max(0, ctx.shape[0] - window + 1) if self.S == 0 and window is not None else 0
            c.update_and_fetch(*self._kv(L, ctx[skip:] if skip else ctx, self.S + skip))
        self.S += ctx.shape[0]

    def feature(self, h, L):
        f = L["readout"](h).astype(mx.float32)
        return (math.sqrt(self.rank) * f / mx.maximum(mx.linalg.norm(f, axis=-1, keepdims=True), 1e-12)).astype(self.dtype)

    def write(self, h, receiver, predecessor, L):
        gate = mx.sigmoid(self.receiver_gate(receiver) + self.predecessor_gate(predecessor) + self.bias)
        delta = L["write"](gate * self.message(predecessor))
        scale = (mx.linalg.norm(h.astype(mx.float32), axis=-1, keepdims=True) / math.sqrt(self.H)).astype(self.dtype)
        return h + scale * delta

    def code(self, ids):
        return (math.sqrt(self.rank) * self.codes[ids]).astype(self.dtype)

    def draft(self, anchor, size):
        """anchor: the token at position self.S (an int or a 1-element array); size: the block's positions, anchor included.
        Returns the normed final states of the size - 1 proposed positions and the last layer's states before its injection."""
        anchor = mx.array([anchor]) if isinstance(anchor, int) else anchor.reshape(1)
        h = mx.concatenate([self.table(anchor).astype(self.dtype), self.mask_rows[: size - 1]])
        S, anchor_code = self.S, self.code(anchor)
        for L, c, window in zip(self.layers, self.cache, self.window):
            hn = _rms(h, L["in_norm"], self.eps)
            q = _rms(L["q"](hn).reshape(1, size, self.heads, self.hd), L["q_norm"], self.eps).transpose(0, 2, 1, 3)
            q = mx.fast.rope(q, self.hd, traditional=False, base=self.theta, scale=1.0, offset=S)
            kb, vb = self._kv(L, hn, S)
            # A key outside the first query's window is masked for every query.
            start = max(0, S - window + 1) if window is not None else 0
            # The cache may start after position zero if the initial prompt exceeded the window.
            cached_start = start - (S - c.offset)
            keys = mx.concatenate([c.keys[..., cached_start:c.offset, :], kb], axis=2)
            values = mx.concatenate([c.values[..., cached_start:c.offset, :], vb], axis=2)
            mask = "causal"
            if window is not None and S + size > window or self.explicit_mask:   # a sliding layer: the keys within `window` positions of each query
                i, j = mx.arange(size)[:, None] + S, mx.arange(start, S + size)[None, :]
                mask = (j <= i) & (j > i - (window or S + size))
            o = mx.fast.scaled_dot_product_attention(q, keys, values, scale=self.hd ** -0.5, mask=mask)
            h = h + L["o"](o.transpose(0, 2, 1, 3).reshape(size, -1))
            hn = _rms(h, L["post_norm"], self.eps)
            gate = L["gate"](hn)
            h = h + L["down"](gate * mx.sigmoid(gate) * L["up"](hn))
            pre = h
            f = self.feature(h, L)
            h = mx.concatenate([h[:1], self.write(h[1:], f[1:], mx.concatenate([anchor_code, f[1:-1]]), L)])
        return _rms(h, self.norm, self.eps)[1:], pre[1:]

    @staticmethod
    def best(logits, K, chunk=128):
        """Each row's K largest entries' places, largest first. They lie in the K chunks with the largest maxima, so
        only those chunks are sorted: 0.07 ms for 7 rows of 248,320 against 0.46 ms for MLX's partition of the rows."""
        P, V = logits.shape
        if V % chunk or V // chunk < K:
            part = mx.argpartition(-logits, K - 1, axis=-1)[:, :K]
            return mx.take_along_axis(part, mx.argsort(-mx.take_along_axis(logits, part, axis=-1), axis=-1), axis=-1)
        c = logits.reshape(P, V // chunk, chunk)
        where = mx.argpartition(-c.max(axis=-1), K - 1, axis=-1)[:, :K]                # (P, K) chunks
        values = mx.take_along_axis(c, where[:, :, None], axis=1).reshape(P, K * chunk)
        places = (where[:, :, None] * chunk + mx.arange(chunk)[None, None, :]).reshape(P, K * chunk)
        return mx.take_along_axis(places, mx.argsort(-values, axis=-1)[:, :K], axis=-1)

    def tokens(self, states, size, pre=None, anchor=None):
        """The proposals. states: the normed final states (size - 1, H). Each position's best token under the target's
        head; or, with self.path and the last layer's states before its injection (pre) and the anchor, DSpine's path
        selection: each position chooses among its top candidates given the token chosen before it."""
        greedy = not self.path or (self.path == "auto" and not self.torn)
        if greedy and self.proposal_head is not None:
            return self.proposal_head(states, size)
        pad = mx.concatenate([states.astype(mx.float32), mx.zeros((1, self.H), mx.float32)])
        if greedy:
            return self.block.argmax(pad, size)[: size - 1]
        logits = self.block.logits(pad, size)[: size - 1]
        P, K, L = size - 1, self.top_k, self.layers[-1]
        cand = self.best(logits, K)                                                   # (P, K): each position's K best tokens, best first
        before = mx.concatenate([mx.broadcast_to(anchor.reshape(1, 1).astype(cand.dtype), (1, K)), cand[:-1]])   # the tokens a position may follow
        code = self.code(before)                                                      # (P, K, rank)
        gate = mx.sigmoid(self.receiver_gate(self.feature(pre, L))[:, None, :] + self.predecessor_gate(code) + self.bias)
        scale = (mx.linalg.norm(pre.astype(mx.float32), axis=-1, keepdims=True) / math.sqrt(self.H)).astype(self.dtype)
        refined = _rms(pre[:, None, :] + scale[:, None, :] * L["write"](gate * self.message(code)), self.norm, self.eps)   # (P, K, H): the state after each token before
        scores = refined.astype(mx.float32) @ self.table(cand).astype(mx.float32).transpose(0, 2, 1)   # [position, token before, candidate]
        chosen, k = [], mx.array(0)
        for i in range(P):
            k = mx.argmax(mx.take(scores[i], k, axis=0))
            chosen.append(mx.take(cand[i], k))
        return mx.stack(chosen)


def load(packed, drafter, *, full_vocab=False, **options):
    """Load target, tokenizer, verifier, and drafter. full_vocab also uses every row for proposals."""
    try:
        from . import block_q35, draft_vocab_q35, fast_q35
    except ImportError:  # run as a script beside them
        import block_q35
        import draft_vocab_q35
        import fast_q35
    model, tok = fast_q35.load(packed)
    block = block_q35.Block(model, tap_layers=drafter_taps(drafter))
    draft = Drafter(drafter, block, **options)
    if not full_vocab and draft.path is not True:
        draft.proposal_head = draft_vocab_q35.qwen_prefix(block, tok)
    return model, tok, block, draft


def set_taps(model, taps):
    """Start collecting the outputs of the model's tapped layers (every call appends); returns {layer: list}."""
    got = {}
    for i, layer in enumerate(model.language_model.model.layers):
        layer._tap[0] = got.setdefault(i, []) if i in taps else None
    return got


def _prefill(model, block, prompt, cache, taps):
    """Match serial decoding's prompt chunks; only the last position needs logits."""
    backbone = model.language_model.model
    tokens = mx.array(prompt)
    got = set_taps(model, taps)
    try:
        for lo in range(0, len(tokens) - 1, 2048):
            backbone(tokens[None, lo : min(lo + 2048, len(tokens) - 1)], cache=cache)
            mx.eval([c.state for c in cache])
        h = backbone(tokens[None, -1:], cache=cache)
        first = mx.argmax(block.table.as_linear(h)[0, -1])
        mx.eval(first, [c.state for c in cache])
    finally:
        set_taps(model, ())
    return first, got


def generate(model, block, drafter, prompt, max_tokens, size=8, eos=(), *, mode="linear"):
    """Greedy text from the target with the drafter proposing blocks of `size`. Returns the tokens and a record of the
    run: rounds, tokens kept per round, and seconds (the prompt, the drafter's reading of it, the rounds).
    Use mode="chat-fork" for an eight-slot verifier with one alternate branch, or mode="copy-wide" for
    16-slot checks with tuned feed-forward kernels on known contiguous copies.
    Memory is wired for the run as mlx_lm's own generation wires it (a run that follows one of mlx_lm's without this
    measured 25 to 30 ms a round against 22.7)."""
    if mode not in ("linear", "chat-fork", "copy-wide"):
        raise ValueError(f"unknown drafter mode: {mode}")
    if mode != "linear" and (size != 8 or drafter.path):
        raise ValueError(f"{mode} requires size=8 and greedy proposals")
    old = mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    proposal_head = drafter.proposal_head
    try:
        if proposal_head is not None and not proposal_head.covers_prompt(prompt):
            drafter.proposal_head = None
        if mode == "linear":
            return _generate(model, block, drafter, prompt, max_tokens, size, eos)
        if mode == "chat-fork":
            if __package__:
                from .strategies.tree_verifier_q35 import generate as generate_tree
            else:
                from strategies.tree_verifier_q35 import generate as generate_tree
            return generate_tree(model, block, drafter, prompt, max_tokens, size, eos)

        if __package__:
            from .strategies.copy_wide_q35 import CopyWideGeometry
            from .strategies.ngram_source_cursor import generate as generate_copy
        else:
            from strategies.copy_wide_q35 import CopyWideGeometry
            from strategies.ngram_source_cursor import generate as generate_copy
        if not hasattr(block, "_copy_wide_geometry"):
            block._copy_wide_geometry = CopyWideGeometry(block)
        with block._copy_wide_geometry.installed():
            return generate_copy(model, block, drafter, prompt, max_tokens, size, eos, wide=True)
    finally:
        block.discard()  # generation owns its caches; no caller will resume them
        drafter.proposal_head = proposal_head
        mx.synchronize()
        mx.set_wired_limit(old)


def _generate_linear(model, block, drafter, prompt, max_tokens, size, eos, *,
                     copy_index=None, wide_index=None, deferred_context=None):
    eos, cache = set(eos), model.make_cache()
    t0 = time.perf_counter()
    first, got = _prefill(model, block, prompt, cache, drafter.taps)
    t1 = time.perf_counter()
    drafter.reset()
    first_id = int(first)
    if max_tokens > 1 and first_id not in eos:
        drafter.context([mx.concatenate(got[i], axis=1)[0] for i in drafter.taps])
        mx.eval([c.state for c in drafter.cache])
        if copy_index is not None:
            copy_index.seed(prompt, first_id)
        if wide_index is not None:
            wide_index.seed(prompt, first_id)
    t2 = time.perf_counter()
    out, anchor, rounds, times = [first_id], first, [], [t2]
    drafter.torn = False
    while len(out) < max_tokens and out[-1] not in eos:
        p = copy_index.proposals() if copy_index is not None else None
        copied = p is not None
        ordinary_wide = False
        width = size
        if copied:
            if wide_index is not None and max_tokens - len(out) >= 16:
                longer, ordinary_wide = copy_index.wide_proposals(wide_index, p)
                if longer is not None:
                    p, width = longer, 16
            tokens = mx.array([out[-1], *p], dtype=anchor.dtype)
        else:
            if deferred_context is not None:
                deferred_context.flush(drafter)
            states, pre = drafter.draft(anchor, size)
            proposed = drafter.tokens(states, size, pre, anchor)
            if OVERLAP:
                mx.async_eval(proposed)                        # the GPU starts on the draft while Python lists the check's launches
            tokens = mx.concatenate([anchor.reshape(1).astype(proposed.dtype), proposed])
        own, taps = block.check(tokens, cache, top1=True)
        if copied:
            mx.eval(own)
        else:
            mx.eval(proposed, own)
            p = proposed.tolist()
        t = own.tolist()
        a = 0
        while a < width - 1 and p[a] == t[a] and p[a] not in eos:
            a += 1
        kept = a + 1                                           # the anchor and the proposals that were right
        emitted = p[:a] + [t[a]]
        out += emitted
        rounds.append(kept)
        finished = len(out) >= max_tokens or out[-1] in eos
        if copy_index is not None:
            first_rejected = p[0] != t[0] if copied else None
            copy_index.observe(emitted, len(rounds) - 1, first_rejected, finished)
        if wide_index is not None:
            wide_index.observe(emitted, len(rounds) - 1, first_rejected if width == 16 else None,
                               finished, reject_key=ordinary_wide)
        if finished:
            times.append(time.perf_counter())
            break
        block.commit(cache, kept)
        accepted_taps = [taps[i][:kept] for i in drafter.taps]
        if deferred_context is None:
            drafter.context(accepted_taps)
            projected = True
        else:
            projected = deferred_context.push(drafter, accepted_taps, defer=copied)
        if OVERLAP and projected:                              # project context while Python lists the next draft
            mx.async_eval([x for c in drafter.cache for x in (c.keys, c.values)])   # the buffers themselves: a cache's .state is a copy
        anchor = own[a]
        drafter.torn = kept < width
        times.append(time.perf_counter())
    t3 = time.perf_counter()
    out = out[:max_tokens]
    return out, {"rounds": len(rounds), "kept": rounds, "round_ms": [round((y - x) * 1e3, 2) for x, y in zip(times, times[1:])], "prompt_s": t1 - t0, "drafter_prompt_s": t2 - t1, "rounds_s": t3 - t2,
                 "tokens": len(out), "tokens_per_check": (len(out) - 1) / max(1, len(rounds))}


_generate = _generate_linear  # Experimental benchmarks can replace this dispatch.
