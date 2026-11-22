"""Streaming serial and DSpark block-8/tree/N-gram speculative decoding for packed Inkling-2-Mini on Apple Silicon."""
from __future__ import annotations

import math
import statistics
import time
from typing import Any, Iterator, Sequence

import mlx.core as mx
import numpy as np

from .drafter import TREE_TOPOLOGIES, DSparkDrafterMLX
from .fast_kernels import _is_m4, make_tree_spec
from .model import Inkling2LoopPackedMLX, Inkling2MLXModel, logits_to_probs, softcap


def _safe_wired_limit() -> int:
    info = mx.device_info()
    rec = int(info.get("max_recommended_working_set_size", 0))
    mem = int(info.get("memory_size", 0))
    if 0 < mem <= 18_500_000_000:
        return min(rec, 10_000_000_000)
    return rec


def _unwrap_model(model: Inkling2LoopPackedMLX | Inkling2MLXModel) -> Inkling2LoopPackedMLX:
    return model.inner if isinstance(model, Inkling2MLXModel) else model


def _sample_next(logits_1d: mx.array, temp: float = 0.0, top_p: float = 1.0, top_k: int = 0) -> int:
    if temp <= 0.0:
        return int(mx.argmax(logits_1d))
    probs = logits_to_probs(logits_1d, temp=temp, top_p=top_p, top_k=top_k)
    logp = mx.where(probs > 0.0, mx.log(mx.maximum(probs, 1e-38)), -float("inf"))
    return int(mx.random.categorical(logp))


def _sample_ordered_gpu(
    ordered_logits_1d: mx.array,
    order_mx: mx.array,
    *,
    cap: float | None,
    temp: float,
    top_p: float = 1.0,
    top_k: int = 0,
) -> mx.array:
    """Sample one vocab token ID on GPU directly from bucket-ordered pre-softcap logits (no 200K `inv_order` gather)."""
    if temp <= 0.0:
        return order_mx[mx.argmax(ordered_logits_1d)].astype(mx.int32)
    probs = logits_to_probs(softcap(ordered_logits_1d, cap), temp=temp, top_p=top_p, top_k=top_k)
    logp = mx.where(probs > 0.0, mx.log(mx.maximum(probs, 1e-38)), -float("inf"))
    return order_mx[mx.random.categorical(logp)].astype(mx.int32)


class NgramDraftIndex:
    """Lossless pure-Python suffix-match prompt/history lookup drafter (`n in (5, 4, 3)`).

    Maintains `self.tokens` (the full committed prompt + generated sequence ending with the current `anchor`),
    incremental n-gram tables indexed lazily up to `_indexed_end`, an O(1) `_cursor` fast path across
    contiguous copy spans, and first-token-reject invalidation (`tables[hit_n][hit_key] = None`).
    """

    def __init__(self, initial_tokens: Sequence[int] = ()):
        self.tokens: list[int] = [int(t) for t in initial_tokens]
        self._t3: dict[tuple[int, ...], int | None] = {}
        self._t4: dict[tuple[int, ...], int | None] = {}
        self._t5: dict[tuple[int, ...], int | None] = {}
        self.tables: dict[int, dict[tuple[int, ...], int | None]] = {
            3: self._t3,
            4: self._t4,
            5: self._t5,
        }
        self._indexed_end: int = 3
        self._cursor: int | None = None
        self._last_hit: tuple[int, tuple[int, ...] | None, int] | None = None
        self._has_tombstones: bool = False

    def extend(self, toks: Sequence[int]) -> None:
        if toks:
            self.tokens.extend(int(t) for t in toks)

    def _sync_index(self, min_cont: int = 2) -> None:
        toks = self.tokens
        stop = len(toks) - max(1, min_cont) + 1
        start = self._indexed_end
        if stop <= start:
            return
        t3 = self._t3
        t4 = self._t4
        t5 = self._t5
        if not self._has_tombstones:
            if start == 3 and start < stop:
                t3[(toks[0], toks[1], toks[2])] = 3
                start = 4
            if start == 4 and start < stop:
                a, b, c, d = toks[0], toks[1], toks[2], toks[3]
                t3[(b, c, d)] = 4
                t4[(a, b, c, d)] = 4
                start = 5
            if start < stop:
                a = toks[start - 5]
                b = toks[start - 4]
                c = toks[start - 3]
                d = toks[start - 2]
                for end in range(start, stop):
                    e = toks[end - 1]
                    t3[(c, d, e)] = end
                    t4[(b, c, d, e)] = end
                    t5[(a, b, c, d, e)] = end
                    a, b, c, d = b, c, d, e
        else:
            if start == 3 and start < stop:
                k3 = (toks[0], toks[1], toks[2])
                if t3.get(k3, 0) is not None:
                    t3[k3] = 3
                start = 4
            if start == 4 and start < stop:
                a, b, c, d = toks[0], toks[1], toks[2], toks[3]
                k3 = (b, c, d)
                k4 = (a, b, c, d)
                if t3.get(k3, 0) is not None:
                    t3[k3] = 4
                if t4.get(k4, 0) is not None:
                    t4[k4] = 4
                start = 5
            if start < stop:
                a = toks[start - 5]
                b = toks[start - 4]
                c = toks[start - 3]
                d = toks[start - 2]
                for end in range(start, stop):
                    e = toks[end - 1]
                    k3 = (c, d, e)
                    k4 = (b, c, d, e)
                    k5 = (a, b, c, d, e)
                    if t3.get(k3, 0) is not None:
                        t3[k3] = end
                    if t4.get(k4, 0) is not None:
                        t4[k4] = end
                    if t5.get(k5, 0) is not None:
                        t5[k5] = end
                    a, b, c, d = b, c, d, e
        self._indexed_end = stop

    def _extract_cont(self, pos: int, max_cont: int, min_cont: int, eos: set[int] | frozenset[int]) -> list[int] | None:
        toks = self.tokens
        n_toks = len(toks)
        if pos < 0 or pos + min_cont > n_toks:
            return None
        end_lim = pos + max_cont
        if end_lim > n_toks:
            end_lim = n_toks
        if eos:
            cont: list[int] = []
            for i in range(pos, end_lim):
                t = toks[i]
                if t in eos:
                    break
                cont.append(t)
        else:
            cont = toks[pos:end_lim]
        return cont if len(cont) >= min_cont else None

    def propose(
        self,
        anchor: int,
        max_draft: int = 8,
        min_cont: int = 2,
        eos: set[int] | frozenset[int] = frozenset(),
        *,
        min_n: int = 3,
    ) -> list[int] | None:
        """Propose `[anchor, c1, ..., c_m]` (length `<= max_draft`) if a continuation of `>= min_cont` exists."""
        max_cont = max_draft - 1
        if max_cont < min_cont:
            self._last_hit = None
            return None
        toks = self.tokens
        n_toks = len(toks)
        if n_toks < 3:
            self._last_hit = None
            return None
        anc = int(anchor)

        # 1. O(1) contiguous-copy cursor fast path
        cur = self._cursor
        if cur is not None:
            if 0 < cur < n_toks and toks[cur - 1] == anc:
                cont = self._extract_cont(cur, max_cont, min_cont, eos)
                if cont is not None:
                    self._last_hit = (0, None, cur)
                    return [anc, *cont]
            self._cursor = None

        # 2. Lazily advance n-gram index watermark up to `n_toks - min_cont`
        self._sync_index(min_cont)

        # 3. Single-probe 3-gram existence check (if 3-gram absent from t3, no 4/5-gram suffix can exist)
        c = toks[n_toks - 3]
        d = toks[n_toks - 2]
        e = toks[n_toks - 1]
        key3 = (c, d, e)
        p3 = self._t3.get(key3, -1)
        if p3 == -1:
            self._last_hit = None
            return None

        # 4. Longest-match priority: n = 5 > 4 > 3
        if n_toks >= 5 and min_n <= 5:
            key5 = (toks[n_toks - 5], toks[n_toks - 4], c, d, e)
            p5 = self._t5.get(key5)
            if p5 is not None:
                cont = self._extract_cont(p5, max_cont, min_cont, eos)
                if cont is not None:
                    self._last_hit = (5, key5, p5)
                    return [anc, *cont]
        if n_toks >= 4 and min_n <= 4:
            key4 = (toks[n_toks - 4], c, d, e)
            p4 = self._t4.get(key4)
            if p4 is not None:
                cont = self._extract_cont(p4, max_cont, min_cont, eos)
                if cont is not None:
                    self._last_hit = (4, key4, p4)
                    return [anc, *cont]
        if min_n <= 3 and p3 is not None:
            cont = self._extract_cont(p3, max_cont, min_cont, eos)
            if cont is not None:
                self._last_hit = (3, key3, p3)
                return [anc, *cont]

        self._last_hit = None
        return None

    def observe(self, raw: Sequence[int], k_match: int, block_size: int, was_ngram: bool) -> None:
        """Update token history, cursor fast path, and first-token-reject invalidation after one verify round."""
        hit = self._last_hit
        self._last_hit = None
        if not raw:
            self._cursor = None
            return
        self.extend(raw)
        if not was_ngram or hit is None:
            self._cursor = None
            return
        hit_n, hit_key, hit_pos = hit
        if k_match == 0 and hit_n > 0 and hit_key is not None:
            self.tables[hit_n][hit_key] = None
            self._has_tombstones = True
            self._cursor = None
            return
        if k_match == block_size - 1 and block_size > 1 and len(raw) == k_match + 1:
            bonus_idx = hit_pos + k_match
            if bonus_idx < len(self.tokens) - 1 and self.tokens[bonus_idx] == int(raw[-1]):
                self._cursor = bonus_idx + 1
                return
        self._cursor = None


def verify_spec_sample_gpu(
    ordered_logits: mx.array,
    order_mx: mx.array,
    draft_ord: mx.array,
    q_probs: mx.array | None,
    *,
    cap: float | None,
    temp: float,
    top_p: float = 1.0,
    top_k: int = 0,
) -> tuple[mx.array, mx.array]:
    """Exact Chen/Leviathan unbiased speculative sampling verification on GPU in bucket-ordered space.

    Handles both shortlist/full `q_probs: float32[K, S]` (`S <= V`, aligned with `ordered_logits[:, :S]`)
    and one-hot N-gram drafts (`q_probs=None`, `q(x) = 1[x == draft_ord]`). Computes the residual
    `max(0, p_row - q_row)` ONLY for the single selected row `p_ord[k_match]` and returns scalar
    `(k_match_i32, final_tok_i32)` with zero CPU syncs.
    """
    if ordered_logits.ndim == 3:
        ordered_logits = ordered_logits[0]
    T, V = int(ordered_logits.shape[0]), int(ordered_logits.shape[1])
    K = T - 1

    p_ord = logits_to_probs(softcap(ordered_logits, cap), temp=temp, top_p=top_p, top_k=top_k)  # [T, V]
    rows = mx.arange(K, dtype=mx.int32)
    draft_ord_i32 = draft_ord.reshape(K).astype(mx.int32)
    p_d = p_ord[rows, draft_ord_i32]  # [K]
    q_d = q_probs[rows, draft_ord_i32] if q_probs is not None else mx.ones((K,), dtype=mx.float32)

    # Strict `<` guarantees `p_d == 0` is NEVER accepted (even if `u == 0.0`).
    u = mx.random.uniform(shape=(K,), dtype=mx.float32)
    accept = (u * q_d) < p_d
    k_match = mx.sum(mx.cumprod(accept.astype(mx.int32), axis=0)).astype(mx.int32)

    all_accepted = k_match == K
    rej_idx = mx.minimum(k_match, K - 1)
    p_row = p_ord[k_match]  # [V]

    if q_probs is not None:
        S = int(q_probs.shape[1])
        q_sel = q_probs[rej_idx]  # [S]
        if S == V:
            res_row = mx.maximum(p_row - q_sel, 0.0)
        else:
            res_row = mx.concatenate([mx.maximum(p_row[:S] - q_sel, 0.0), p_row[S:]], axis=0)
    else:
        rej_ord = draft_ord_i32[rej_idx]
        res_row = mx.where(mx.arange(V, dtype=mx.int32) == rej_ord, 0.0, p_row)

    res_sum = mx.sum(res_row)
    safe_res = mx.where(res_sum > 0.0, res_row / mx.maximum(res_sum, 1e-38), p_row)
    dist = mx.where(all_accepted, p_row, safe_res)
    log_dist = mx.where(dist > 0.0, mx.log(mx.maximum(dist, 1e-38)), -float("inf"))
    final_ord = mx.random.categorical(log_dist).astype(mx.int32)
    final_tok = order_mx[final_ord].astype(mx.int32)
    return k_match, final_tok


def verify_margin_greedy_gpu(
    ordered_logits: mx.array,
    order_mx: mx.array,
    inv_order: mx.array,
    draft_ids: mx.array,
    *,
    cap: float | None,
    margin: float,
) -> tuple[mx.array, mx.array]:
    """Tight-margin greedy verification on GPU in bucket-ordered space (zero 200K gather, zero 200K softcap)."""
    if ordered_logits.ndim == 3:
        ordered_logits = ordered_logits[0]
    ids = draft_ids.reshape(-1).astype(mx.int32)
    K = int(ordered_logits.shape[0]) - 1
    best_ord = mx.argmax(ordered_logits, axis=-1).astype(mx.int32)
    draft_ord = inv_order[ids[1:]].astype(mx.int32)
    if margin <= 0.0:
        accept = draft_ord == best_ord[:-1]
    else:
        rows = mx.arange(K, dtype=mx.int32)
        draft_raw = ordered_logits[rows, draft_ord].astype(mx.float32)
        max_raw = ordered_logits[rows, best_ord[:-1]].astype(mx.float32)
        accept = (draft_ord == best_ord[:-1]) | (softcap(draft_raw, cap) >= (softcap(max_raw, cap) - float(margin)))
    k_match = mx.sum(mx.cumprod(accept.astype(mx.int32), axis=0)).astype(mx.int32)
    bonus_tok = order_mx[best_ord[k_match]].astype(mx.int32)
    return k_match, bonus_tok


def _compile_margin(max_logit_gap: float, min_prob_ratio: float | None) -> float:
    if max_logit_gap < 0.0:
        raise ValueError(f"max_logit_gap must be >= 0, got {max_logit_gap}")
    if min_prob_ratio is not None:
        if not (0.0 < min_prob_ratio <= 1.0):
            raise ValueError(f"min_prob_ratio must be in (0, 1], got {min_prob_ratio}")
        ratio_margin = -math.log(min_prob_ratio)
        return min(float(max_logit_gap), ratio_margin) if max_logit_gap > 0.0 else ratio_margin
    return float(max_logit_gap)


def stream_serial(
    model: Inkling2LoopPackedMLX | Inkling2MLXModel,
    prompt_ids: Sequence[int],
    *,
    max_tokens: int = 256,
    eos_ids: set[int] | Sequence[int] = (200006,),
    temp: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    prefill_chunk_size: int = 256,
    prefix_state: dict[str, Any] | None = None,
    stats_out: dict[str, Any] | None = None,
) -> Iterator[list[int]]:
    """Stream single-token serial generation from a packed Inkling-2-Mini model."""
    if temp < 0.0:
        raise ValueError(f"temp must be >= 0, got {temp}")
    m = _unwrap_model(model)
    eos = set(int(x) for x in eos_ids)
    greedy = temp <= 0.0
    ids_list = list(prompt_ids)
    chunk_sz = max(1, int(prefill_chunk_size))
    old_wire = mx.set_wired_limit(_safe_wired_limit())
    t0: float | None = None
    t1: float | None = None
    emitted = 0
    step_ms: list[float] = []
    next_pending: mx.array | None = None
    try:
        cache = m.make_cache()
        t0 = time.perf_counter()
        pos = 0
        while len(ids_list) - pos > chunk_sz:
            h_chk, cache = m.forward_prefill_chunk(ids_list[pos : pos + chunk_sz], cache=cache)
            mx.eval(h_chk)
            mx.clear_cache()
            pos += chunk_sz
        tail_ids = ids_list[pos:]
        if greedy:
            nxt_mx, cache = m.forward_argmax(tail_ids, cache=cache)
            nxt_mx = nxt_mx[0, -1]
            nxt = int(nxt_mx)
        else:
            ord_logits, cache = m.forward_ordered_logits(tail_ids, cache=cache)
            nxt_mx = _sample_ordered_gpu(
                ord_logits[0, -1], m.lm_head.order_mx, cap=m.lm_head.cap, temp=temp, top_p=top_p, top_k=top_k
            )
            nxt = int(nxt_mx)
        t1 = time.perf_counter()

        emitted = 1
        yield [nxt]
        if nxt not in eos and max_tokens > 1:
            if greedy:
                # 1-step async_eval pipeline overlaps Python enqueue of step k+1 with GPU execution of step k!
                cur_tok_mx = mx.array([nxt], dtype=mx.int32)
                pending_mx, cache = m.forward_argmax(cur_tok_mx, cache=cache)
                pending_mx = pending_mx[0, -1]
                mx.async_eval(pending_mx)
                ts = time.perf_counter()
                for step_i in range(max_tokens - 1):
                    next_pending = None
                    if step_i + 1 < max_tokens - 1:
                        next_pending, cache = m.forward_argmax(pending_mx.reshape(1), cache=cache)
                        next_pending = next_pending[0, -1]
                        mx.async_eval(next_pending)
                    nxt = int(pending_mx)
                    now = time.perf_counter()
                    step_ms.append((now - ts) * 1e3)
                    ts = now
                    emitted += 1
                    yield [nxt]
                    if nxt in eos:
                        break
                    pending_mx = next_pending
                    next_pending = None
            else:
                for _ in range(max_tokens - 1):
                    ts = time.perf_counter()
                    ord_logits, cache = m.forward_ordered_logits([nxt], cache=cache)
                    nxt_mx = _sample_ordered_gpu(
                        ord_logits[0, -1], m.lm_head.order_mx, cap=m.lm_head.cap, temp=temp, top_p=top_p, top_k=top_k
                    )
                    nxt = int(nxt_mx)
                    step_ms.append((time.perf_counter() - ts) * 1e3)
                    emitted += 1
                    yield [nxt]
                    if nxt in eos:
                        break
    finally:
        if next_pending is not None:
            mx.eval(next_pending)
        if prefix_state is not None and "cache" in locals():
            prefix_state["cache"] = cache
            prefix_state["pos"] = cache.seq_len
        if stats_out is not None and t0 is not None and t1 is not None:
            t2 = time.perf_counter()
            decode_s = max(t2 - t1, 1e-9)
            decode_toks = max(emitted - 1, 0)
            stats_out.update(
                {
                    "mode": "serial",
                    "prompt_tokens": len(prompt_ids),
                    "tokens": decode_toks,
                    "generated_tokens": emitted,
                    "prompt_s": round(t1 - t0, 4),
                    "rounds_s": round(decode_s, 4),
                    "decode_tok_s": round(decode_toks / decode_s, 2) if decode_toks else 0.0,
                    "median_step_ms": round(statistics.median(step_ms), 2) if step_ms else 0.0,
                    "temp": temp,
                    "top_p": top_p,
                    "top_k": top_k,
                    "active_gb": round(mx.get_active_memory() / 1e9, 3),
                    "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
                }
            )
        mx.set_wired_limit(old_wire)


def _copy_loop_pass1_to_pass2(cache) -> None:
    """Copy Pass-1 KV/conv slots (24..35) to Pass-2 slots (36..47) after 2-loop chunk prefill."""
    for j in range(12):
        s_src, s_dst = 24 + j, 36 + j
        cache.kv_len[s_dst] = cache.kv_len[s_src]
        cache.kv_start[s_dst] = cache.kv_start[s_src]
        cache.kv_cap[s_dst] = cache.kv_cap[s_src]
        cache.kv_is_sliding[s_dst] = cache.kv_is_sliding[s_src]
        cache.kv_is_fp8[s_dst] = cache.kv_is_fp8[s_src]
        if hasattr(cache, "kv_mode"):
            cache.kv_mode[s_dst] = cache.kv_mode[s_src]
        if cache.kv_u8[s_src] is not None:
            cache.kv_u8[s_dst] = mx.array(cache.kv_u8[s_src])
        if cache.kv_f16[s_src] is not None:
            cache.kv_f16[s_dst] = mx.array(cache.kv_f16[s_src])
        if hasattr(cache, "kv_q4_sc") and cache.kv_q4_sc[s_src] is not None:
            cache.kv_q4_sc[s_dst] = mx.array(cache.kv_q4_sc[s_src])
        cache.conv_states[s_dst] = [mx.array(c) if c is not None else None for c in cache.conv_states[s_src]]


def stream_dspark(
    model: Inkling2LoopPackedMLX | Inkling2MLXModel,
    drafter: DSparkDrafterMLX,
    prompt_ids: Sequence[int],
    *,
    max_tokens: int = 256,
    eos_ids: set[int] | Sequence[int] = (200006,),
    adaptive: bool = True,
    draft_len: int | None = None,
    tree: str | None = None,
    temp: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    max_logit_gap: float = 0.0,
    min_prob_ratio: float | None = None,
    ngram: bool = False,
    ngram_max_draft: int = 8,
    ngram_min_match: int = 5,
    prefill_chunk_size: int = 256,
    draft_shortlist_buckets: int | None = None,
    prefix_state: dict[str, Any] | None = None,
    stats_out: dict[str, Any] | None = None,
) -> Iterator[list[int]]:
    """Stream hybrid N-gram + DSpark speculative decoding (exact greedy, tight-margin greedy, or exact unbiased `temp > 0`)."""
    if temp < 0.0:
        raise ValueError(f"temp must be >= 0, got {temp}")
    eff_margin = _compile_margin(max_logit_gap, min_prob_ratio)
    if tree is not None and tree not in TREE_TOPOLOGIES:
        raise ValueError(f"Unknown tree topology {tree!r}; expected one of {tuple(TREE_TOPOLOGIES)}")
    if temp > 0.0 and (eff_margin > 0.0 or tree is not None):
        raise ValueError("temp > 0 cannot be combined with max_logit_gap/min_prob_ratio or tree verification")

    m = _unwrap_model(model)
    eos = set(int(x) for x in eos_ids)
    stochastic = temp > 0.0
    tap_ids = drafter.target_layer_ids  # (5, 23, 41, 48)
    fixed_block = None if draft_len is None else max(2, min(int(draft_len), drafter.block_size))
    is_m4 = _is_m4()
    _os = __import__("os")
    low_blk = min((int(_os.environ.get("LAMQ_M4_LOW_BLK", "3")) if is_m4 else 6), drafter.block_size)
    high_blk = min((int(_os.environ.get("LAMQ_M4_HIGH_BLK", "4")) if is_m4 else drafter.block_size), drafter.block_size)
    cur_block = fixed_block if fixed_block is not None else low_blk
    ngram_cap = max(2, min(int(ngram_max_draft), 8))
    tree_spec = make_tree_spec(TREE_TOPOLOGIES[tree]) if tree is not None else None
    tree_parents_mx = mx.array(np.array(TREE_TOPOLOGIES[tree][1:], dtype=np.int32)) if tree is not None else None

    draft_head = None
    draft_head_topk = None
    draft_head_sampler = None
    if stochastic:
        draft_head_sampler = m.lm_head.get_draft_head_sampler(temp=temp, top_p=top_p, top_k=top_k)
    elif tree_spec is not None:
        nb = draft_shortlist_buckets if draft_shortlist_buckets is not None else 5
        draft_head_topk = m.lm_head.get_draft_head_topk(max_buckets=nb)
    else:
        nb = draft_shortlist_buckets if draft_shortlist_buckets is not None else 5
        draft_head = m.lm_head.get_draft_head_argmax(max_buckets=nb)

    ids_list = list(prompt_ids)
    chunk_sz = max(1, int(prefill_chunk_size))
    old_wire = mx.set_wired_limit(_safe_wired_limit())
    t0: float | None = None
    t1: float | None = None
    t2: float | None = None
    emitted_total = 0
    accepted_per_round: list[int] = []
    round_ms: list[float] = []
    serial_steps = 0
    branch_hits = 0
    ngram_rounds = 0
    ngram_accepted = 0
    ngram_first_rejects = 0
    try:
        t0 = time.perf_counter()
        if prefix_state is not None and prefix_state.get("cache") is not None and prefix_state.get("pos", 0) > 0:
            cache = prefix_state["cache"]
            pos = int(prefix_state["pos"])
        else:
            drafter.reset()
            cache = m.make_cache()
            pos = 0
        fast_pf = (__import__("os").environ.get("LAMQ_FAST_PREFILL") == "1") and (m.cfg.loop_passes == 3)
        if fast_pf and (len(ids_list) - pos > chunk_sz):
            orig_lp = m.cfg.loop_passes
            m.cfg.loop_passes = 2
            pf_taps = (tap_ids[0], tap_ids[1], 35)
            try:
                while len(ids_list) - pos > chunk_sz:
                    _, cache, c_taps = m.forward_prefill_chunk(ids_list[pos : pos + chunk_sz], cache=cache, tap_visits=pf_taps)
                    mx.eval(c_taps)
                    ct4 = mx.stack([c_taps[:, 0], c_taps[:, 1], c_taps[:, 2], c_taps[:, 2]], axis=1)
                    drafter.push_context(ct4, start_pos=pos)
                    mx.eval(drafter.f_rnn)
                    mx.clear_cache()
                    pos += chunk_sz
                _copy_loop_pass1_to_pass2(cache)
            finally:
                m.cfg.loop_passes = orig_lp
        else:
            while len(ids_list) - pos > chunk_sz:
                _, cache, c_taps = m.forward_prefill_chunk(ids_list[pos : pos + chunk_sz], cache=cache, tap_visits=tap_ids)
                mx.eval(c_taps)
                drafter.push_context(c_taps, start_pos=pos)
                mx.eval(drafter.f_rnn)
                mx.clear_cache()
                pos += chunk_sz
        tail_ids = ids_list[pos:]
        if stochastic:
            first_ord_logits, cache, taps = m.forward_ordered_logits(tail_ids, cache=cache, tap_visits=tap_ids)
            first_mx = _sample_ordered_gpu(
                first_ord_logits[0, -1], m.lm_head.order_mx, cap=m.lm_head.cap, temp=temp, top_p=top_p, top_k=top_k
            )
            mx.eval(first_mx, taps)
            first = int(first_mx)
        else:
            first_mx, cache, taps = m.forward_argmax(tail_ids, cache=cache, tap_visits=tap_ids)
            first_mx = first_mx[0, -1]
            mx.eval(first_mx, taps)
            first = int(first_mx)
        t1 = time.perf_counter()

        drafter.push_context(taps, start_pos=pos)
        ngram_idx = NgramDraftIndex(prompt_ids) if ngram else None
        if ngram_idx is not None:
            ngram_idx.extend([first])
            ngram_idx._sync_index(max(2, ngram_min_match))
        mx.eval(drafter.f_rnn)
        t2 = time.perf_counter()

        emitted_total = 1
        anchor = first
        yield [first]

        dspark_rounds = 0
        ewma_accept = 3.0
        serial_cooldown = 0
        serial_burst_len = 8

        # Pending planned next step: (source, draft_gpu, draft_list, draft_ord, q_probs, block_size)
        pending_source: str | None = None
        pending_draft: mx.array | None = None
        pending_draft_list: list[int] | None = None
        pending_draft_ord: mx.array | None = None
        pending_q_probs: mx.array | None = None
        pending_block_size: int = cur_block

        def _plan_next_draft() -> tuple[str, mx.array | None, list[int] | None, mx.array | None, mx.array | None, int]:
            in_cooldown = adaptive and serial_cooldown > 0
            if ngram_idx is not None:
                # Require high-precision suffix match + continuation so boilerplate never preempts DSpark.
                mn = max(3, min(5, int(ngram_min_match)))
                ng_list = ngram_idx.propose(anchor, max_draft=ngram_cap, min_cont=mn, eos=eos, min_n=mn)
                if ng_list is not None:
                    ng_mx = mx.array(ng_list, dtype=mx.int32)
                    ng_ord = m.lm_head.inv_order[ng_mx[1:]] if stochastic else None
                    return "ngram", ng_mx, ng_list, ng_ord, None, len(ng_list)
            if in_cooldown:
                return "serial", None, None, None, None, 1
            if stochastic:
                d_ids, d_ord, q_p = drafter.propose_sample_gpu(
                    anchor, m.embed, draft_head_sampler, draft_len=cur_block
                )
                return "dspark", d_ids, None, d_ord, q_p, cur_block
            if tree_spec is not None:
                d_ids = drafter.propose_tree8_gpu(anchor, m.embed, draft_head_topk, topology=tree)
                return "dspark", d_ids, None, None, None, tree_spec["T"]
            d_ids = drafter.propose_gpu(anchor, m.embed, draft_head, draft_len=cur_block)
            return "dspark", d_ids, None, None, None, cur_block

        def _async_eval_drafter_and_pending() -> None:
            eval_list: list[mx.array] = [drafter.f_rnn]
            eval_list.extend(x for x in drafter.ctx_k if x is not None)
            eval_list.extend(x for x in drafter.ctx_v if x is not None)
            if pending_source == "dspark":
                if pending_draft is not None:
                    eval_list.append(pending_draft)
                if pending_draft_ord is not None:
                    eval_list.append(pending_draft_ord)
                if pending_q_probs is not None:
                    eval_list.append(pending_q_probs)
            elif pending_source == "ngram" and pending_draft_ord is not None:
                eval_list.append(pending_draft_ord)
            mx.async_eval(*eval_list)

        while emitted_total < max_tokens and anchor not in eos:
            tr0 = time.perf_counter()
            if pending_source is None:
                (
                    pending_source,
                    pending_draft,
                    pending_draft_list,
                    pending_draft_ord,
                    pending_q_probs,
                    pending_block_size,
                ) = _plan_next_draft()
                if pending_source == "dspark":
                    init_eval = [x for x in (pending_draft, pending_draft_ord, pending_q_probs) if x is not None]
                    if init_eval:
                        mx.async_eval(*init_eval)

            round_source = pending_source
            draft_gpu = pending_draft
            draft_list = pending_draft_list
            draft_ord = pending_draft_ord
            q_probs = pending_q_probs
            block_size = pending_block_size
            pending_source = None
            pending_draft = None
            pending_draft_list = None
            pending_draft_ord = None
            pending_q_probs = None

            if round_source == "serial":
                if stochastic:
                    nxt_ord_logits, cache, taps = m.forward_ordered_logits([anchor], cache=cache, tap_visits=tap_ids)
                    nxt_mx = _sample_ordered_gpu(
                        nxt_ord_logits[0, -1], m.lm_head.order_mx, cap=m.lm_head.cap, temp=temp, top_p=top_p, top_k=top_k
                    )
                    mx.eval(nxt_mx, taps)
                else:
                    nxt_mx, cache, taps = m.forward_argmax([anchor], cache=cache, tap_visits=tap_ids)
                    nxt_mx = nxt_mx[0, -1]
                    mx.eval(nxt_mx, taps)
                nxt = int(nxt_mx)
                emitted_total += 1
                anchor = nxt
                serial_steps += 1
                serial_cooldown -= 1
                if ngram_idx is not None:
                    ngram_idx.observe([nxt], 0, 1, False)
                drafter.push_context(taps, start_pos=cache.seq_len - 1)
                if emitted_total < max_tokens and anchor not in eos:
                    (
                        pending_source,
                        pending_draft,
                        pending_draft_list,
                        pending_draft_ord,
                        pending_q_probs,
                        pending_block_size,
                    ) = _plan_next_draft()
                _async_eval_drafter_and_pending()
                yield [nxt]
                continue

            base_pos = cache.seq_len
            cache.begin_spec_verify()

            if stochastic:
                ord_logits, cache, ver_taps = m.forward_all_ordered_logits(
                    draft_gpu, cache=cache, tap_visits=tap_ids
                )
                k_match_mx, final_tok_mx = verify_spec_sample_gpu(
                    ord_logits[0],
                    m.lm_head.order_mx,
                    draft_ord,
                    q_probs,
                    cap=m.lm_head.cap,
                    temp=temp,
                    top_p=top_p,
                    top_k=top_k,
                )
                if draft_list is not None:
                    mx.eval(k_match_mx, final_tok_mx, ver_taps)
                    draft_block = draft_list
                else:
                    mx.eval(draft_gpu, k_match_mx, final_tok_mx, ver_taps)
                    draft_block = np.array(draft_gpu, dtype=np.int32).tolist()
                k_match = int(k_match_mx)
                final_tok = int(final_tok_mx)
                k_commit = k_match + 1
                taps_commit = ver_taps[:k_commit]
                raw = draft_block[1 : k_match + 1] + [final_tok]
                commit_fn = lambda kc=k_commit, bs=block_size: cache.commit_spec_verify(kc, bs)
            elif tree_spec is not None and round_source == "dspark":
                if eff_margin > 0.0:
                    ord_logits, cache, ver_taps = m.forward_all_ordered_logits(
                        draft_gpu, cache=cache, tap_visits=tap_ids, tree_spec=tree_spec
                    )
                    ol = ord_logits[0]
                    best_ord = mx.argmax(ol, axis=-1).astype(mx.int32)
                    ver_preds_mx = m.lm_head.order_mx[best_ord]
                    rows_t = mx.arange(block_size, dtype=mx.int32)
                    max_sc_mx = softcap(ol[rows_t, best_ord].astype(mx.float32), m.lm_head.cap)
                    child_ord = m.lm_head.inv_order[draft_gpu[1:]].astype(mx.int32)
                    child_sc_mx = softcap(ol[tree_parents_mx, child_ord].astype(mx.float32), m.lm_head.cap)
                    mx.eval(draft_gpu, ver_preds_mx, max_sc_mx, child_sc_mx, ver_taps)
                    draft_block = np.array(draft_gpu, dtype=np.int32).tolist()
                    preds = np.array(ver_preds_mx, dtype=np.int32).tolist()
                    max_sc = np.array(max_sc_mx, dtype=np.float32).tolist()
                    child_sc = np.array(child_sc_mx, dtype=np.float32).tolist()
                else:
                    ver_preds_mx, cache, ver_taps = m.forward_all_argmax(
                        draft_gpu, cache=cache, tap_visits=tap_ids, tree_spec=tree_spec
                    )
                    ver_preds_mx = ver_preds_mx[0]
                    mx.eval(draft_gpu, ver_preds_mx, ver_taps)
                    draft_block = np.array(draft_gpu, dtype=np.int32).tolist()
                    preds = np.array(ver_preds_mx, dtype=np.int32).tolist()
                    max_sc = None
                    child_sc = None

                accepted_path = [0]
                children_tbl = tree_spec["children"]
                cur = 0
                while True:
                    want = preds[cur]
                    nxt_node = -1
                    for ch in children_tbl[cur]:
                        if draft_block[ch] == want:
                            nxt_node = ch
                            break
                    if nxt_node < 0 and eff_margin > 0.0 and max_sc is not None and child_sc is not None:
                        thresh = max_sc[cur] - eff_margin
                        best_ch_sc = -float("inf")
                        for ch in children_tbl[cur]:
                            sc_v = child_sc[ch - 1]
                            if sc_v >= thresh and sc_v > best_ch_sc:
                                best_ch_sc = sc_v
                                nxt_node = ch
                    if nxt_node < 0:
                        break
                    accepted_path.append(nxt_node)
                    cur = nxt_node
                k_match = len(accepted_path) - 1
                is_contiguous = accepted_path == list(range(len(accepted_path)))
                if not is_contiguous:
                    branch_hits += 1
                taps_commit = (
                    ver_taps[: len(accepted_path)]
                    if is_contiguous
                    else ver_taps[mx.array(np.array(accepted_path, dtype=np.int32))]
                )
                raw = [draft_block[n] for n in accepted_path[1:]] + [preds[accepted_path[-1]]]
                commit_fn = lambda ap=accepted_path, bs=block_size: cache.commit_tree_verify(ap, bs)
            elif eff_margin > 0.0:
                ord_logits, cache, ver_taps = m.forward_all_ordered_logits(
                    draft_gpu, cache=cache, tap_visits=tap_ids
                )
                k_match_mx, bonus_tok_mx = verify_margin_greedy_gpu(
                    ord_logits[0],
                    m.lm_head.order_mx,
                    m.lm_head.inv_order,
                    draft_gpu,
                    cap=m.lm_head.cap,
                    margin=eff_margin,
                )
                if draft_list is not None:
                    mx.eval(k_match_mx, bonus_tok_mx, ver_taps)
                    draft_block = draft_list
                else:
                    mx.eval(draft_gpu, k_match_mx, bonus_tok_mx, ver_taps)
                    draft_block = np.array(draft_gpu, dtype=np.int32).tolist()
                k_match = int(k_match_mx)
                bonus_tok = int(bonus_tok_mx)
                k_commit = k_match + 1
                taps_commit = ver_taps[:k_commit]
                raw = draft_block[1 : k_match + 1] + [bonus_tok]
                commit_fn = lambda kc=k_commit, bs=block_size: cache.commit_spec_verify(kc, bs)
            else:
                ver_preds_mx, cache, ver_taps = m.forward_all_argmax(draft_gpu, cache=cache, tap_visits=tap_ids)
                ver_preds_mx = ver_preds_mx[0]
                if draft_list is not None:
                    mx.eval(ver_preds_mx, ver_taps)
                    draft_block = draft_list
                else:
                    mx.eval(draft_gpu, ver_preds_mx, ver_taps)
                    draft_block = np.array(draft_gpu, dtype=np.int32).tolist()
                preds = np.array(ver_preds_mx, dtype=np.int32).tolist()

                k_match = 0
                for j in range(1, block_size):
                    if preds[j - 1] == draft_block[j]:
                        k_match += 1
                    else:
                        break
                k_commit = k_match + 1
                taps_commit = ver_taps[:k_commit]
                raw = draft_block[1 : k_match + 1] + [preds[k_match]]
                commit_fn = lambda kc=k_commit, bs=block_size: cache.commit_spec_verify(kc, bs)

            chunk: list[int] = []
            for tok in raw:
                if emitted_total >= max_tokens:
                    break
                chunk.append(tok)
                emitted_total += 1
                anchor = tok
                if tok in eos:
                    break

            accepted_per_round.append(len(raw))
            if round_source == "ngram":
                ngram_rounds += 1
                ngram_accepted += k_match
                if k_match == 0:
                    ngram_first_rejects += 1
            else:
                dspark_rounds += 1
                ewma_accept = 0.7 * ewma_accept + 0.3 * len(raw)
                if adaptive:
                    if tree_spec is None and fixed_block is None:
                        # Ensure upgrade threshold is always strictly below low_blk so low_blk=3 can climb to high_blk=4
                        up_thr = (min(3.3, low_blk - 0.45) if is_m4 else 3.6)
                        cur_block = high_blk if ewma_accept >= up_thr else low_blk
                    if dspark_rounds >= 5 and ewma_accept < 1.35:
                        serial_cooldown = serial_burst_len
                        serial_burst_len = min(16, serial_burst_len * 2)
                        ewma_accept = 2.0

            # Observe committed tokens in N-gram index BEFORE planning next round so N-gram hits
            # never launch a wasted DSpark GPU proposal and all 4 DSpark<->N-gram transitions stay bubble-free.
            if ngram_idx is not None:
                ngram_idx.observe(raw, k_match, block_size, round_source == "ngram")

            # Always push committed taps into DSpark (including N-gram rounds) so its KV stays 100% synced.
            drafter.push_context(taps_commit, start_pos=base_pos)
            if emitted_total < max_tokens and anchor not in eos:
                (
                    pending_source,
                    pending_draft,
                    pending_draft_list,
                    pending_draft_ord,
                    pending_q_probs,
                    pending_block_size,
                ) = _plan_next_draft()
            _async_eval_drafter_and_pending()

            commit_fn()
            round_ms.append((time.perf_counter() - tr0) * 1e3)
            if chunk:
                yield chunk
    finally:
        if stats_out is not None and t0 is not None and t1 is not None and t2 is not None:
            t3 = time.perf_counter()
            decode_s = max(t3 - t2, 1e-9)
            decode_toks = max(emitted_total - 1, 0)
            mean_acc = sum(accepted_per_round) / max(len(accepted_per_round), 1)
            mode_name = f"dspark_tree_{tree}" if tree is not None else ("dspark_adaptive" if adaptive else "dspark")
            if stochastic:
                mode_name = f"{mode_name}_sample"
            elif eff_margin > 0.0:
                mode_name = f"{mode_name}_margin"
            stats_out.update(
                {
                    "mode": mode_name,
                    "prompt_tokens": len(prompt_ids),
                    "tokens": decode_toks,
                    "generated_tokens": emitted_total,
                    "prompt_s": round(t1 - t0, 4),
                    "drafter_prompt_s": round(t2 - t1, 4),
                    "rounds_s": round(decode_s, 4),
                    "decode_tok_s": round(decode_toks / decode_s, 2) if decode_toks else 0.0,
                    "rounds": len(accepted_per_round),
                    "tokens_per_check": round(mean_acc, 2),
                    "branch_hits": branch_hits,
                    "ngram_rounds": ngram_rounds,
                    "ngram_accepted": ngram_accepted,
                    "ngram_first_rejects": ngram_first_rejects,
                    "median_round_ms": round(statistics.median(round_ms), 2) if round_ms else 0.0,
                    "serial_steps": serial_steps,
                    "temp": temp,
                    "top_p": top_p,
                    "top_k": top_k,
                    "eff_margin": round(eff_margin, 6),
                    "active_gb": round(mx.get_active_memory() / 1e9, 3),
                    "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
                }
            )
        mx.set_wired_limit(old_wire)


def generate_serial(
    model: Inkling2LoopPackedMLX | Inkling2MLXModel,
    prompt_ids: Sequence[int],
    max_tokens: int = 128,
    eos_ids: Sequence[int] = (200006,),
    temp: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> tuple[list[int], dict[str, Any]]:
    stats: dict[str, Any] = {}
    out: list[int] = []
    for chunk in stream_serial(
        model,
        prompt_ids,
        max_tokens=max_tokens,
        eos_ids=eos_ids,
        temp=temp,
        top_p=top_p,
        top_k=top_k,
        stats_out=stats,
    ):
        out.extend(chunk)
    return out, stats


def generate_dspark(
    model: Inkling2LoopPackedMLX | Inkling2MLXModel,
    drafter: DSparkDrafterMLX,
    prompt_ids: Sequence[int],
    max_tokens: int = 128,
    eos_ids: Sequence[int] = (200006,),
    adaptive: bool = True,
    draft_len: int | None = None,
    tree: str | None = None,
    temp: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    max_logit_gap: float = 0.0,
    min_prob_ratio: float | None = None,
    ngram: bool = False,
    ngram_max_draft: int = 8,
    ngram_min_match: int = 5,
) -> tuple[list[int], dict[str, Any]]:
    stats: dict[str, Any] = {}
    out: list[int] = []
    for chunk in stream_dspark(
        model,
        drafter,
        prompt_ids,
        max_tokens=max_tokens,
        eos_ids=eos_ids,
        adaptive=adaptive,
        draft_len=draft_len,
        tree=tree,
        temp=temp,
        top_p=top_p,
        top_k=top_k,
        max_logit_gap=max_logit_gap,
        min_prob_ratio=min_prob_ratio,
        ngram=ngram,
        ngram_max_draft=ngram_max_draft,
        ngram_min_match=ngram_min_match,
        stats_out=stats,
    ):
        out.extend(chunk)
    return out, stats
