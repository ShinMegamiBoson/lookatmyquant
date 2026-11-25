"""Paged-KV continuous batching, zero-copy prefix sharing, and load-adaptive DSpark for Inkling-2-Mini MLX.

Reuses 100% of `Inkling2LoopPackedMLX`'s packed Metal dense/MoE/`lm_head` kernels while adding:
  - `PagedInklingPool`: 64-token physical block pools for both global (`fp16`/`fp8`/`q4`, optional
    `shared_loop_kv`) and sliding (`fp16`, O(1) ring eviction past 1,024 tokens) visits, plus an
    in-pool block-hash `PrefixBlockCache` for zero-copy prefix sharing across concurrent agents.
  - `forward_batched`: runs `B` independent sequences x `K` tokens each in one GPU forward pass
    (`B=1, K<=512` chunked prefill; `B=1..4, K=4..5` batched linear DSpark; `B=1..64, K=1` decode).
  - `generate_batch`: simple Python API for multi-prompt batched generation.
  - `ContinuousBatchScheduler`: thread-safe continuous batching + chunked-prefill interleaving +
    load-adaptive DSpark scheduler for `lamq serve`.
"""
from __future__ import annotations

import hashlib
import os
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import mlx.core as mx
import numpy as np

from .batch_kernels import (
    PAGED_BLOCK_SIZE,
    gqa_paged_sdpa_fast,
    kv_dyn_sconv4_norm_batched,
    kv_sconv4_norm_batched,
    moe_wsum_sconv4_res_batched,
    qkv_rel_prep_sliding_batched,
    short_conv4_res_batched,
)
from .fast_kernels import (
    _is_m4,
    block_attn_res_fused,
    block_attn_res_norm_rht_fast,
    compute_moe_sorted_offsets,
    fp8_e4m3_quantize_f16_to_u8,
    kv_dyn_sconv4_norm_fused,
    kv_h128_q4_quantize,
    kv_sconv4_norm_fused,
    moe_wsum_sconv4_res_fused,
    msg_blend_norm_3x_fused,
    msg_blend_norm_rht_fast,
    q_rel_clip_fused,
    qkv_rel_prep_sliding_fused,
    rht_in_2304_fast,
    rht_in_3072_gated_fast,
    rht_out_1024_fast,
    short_conv4_res_fused,
    sigmoid_top8_norm_and_plan_fast,
    sigmoid_top8_norm_fast,
    swiglu_rht_1152_fast,
)
from .generate import (
    FORCED_THINK_EXIT as THINK_CLOSE_IDS,
    _ThinkCapState,
    _sample_ordered_gpu,
)
from .kernels import rht_apply_out
from .model import (
    Inkling2Config,
    Inkling2LoopPackedMLX,
    _msg_1x_compiled,
    _msg_3x_shared_h_compiled,
    _msg_3x_split_h_compiled,
    rms_norm,
)
from .nested_kv_kernels import (
    gqa_paged_nested_coarse_score,
    gqa_paged_nested_sdpa_fast,
    kv_h128_nested4_quantize,
)

# Keep 18 sliding blocks (1,152 tokens) per active sequence so W=1024 + partial tail + small K fits with zero copy.
_MAX_SLIDING_BLOCKS = 18


def _block_hash(parent_hash: bytes, block_tokens: tuple[int, ...]) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(parent_hash)
    h.update(np.asarray(block_tokens, dtype=np.int32).tobytes())
    return h.digest()


@dataclass
class _CachedBlockNode:
    block_hash: bytes
    parent_hash: bytes
    g_blk: int
    s_blk: int
    conv_snap: list[list[mx.array]] | None  # [48][4] float32[3, C] at end of this block
    children: set[bytes] = field(default_factory=set)
    lru_tick: int = 0


def _resolve_nested_opt_int(val: int | None, env_name: str) -> int | None:
    if val is not None:
        return max(0, int(val))
    raw = os.environ.get(env_name, "").strip().lower()
    if not raw or raw in ("none", "off", "disabled"):
        return None
    return max(0, int(raw))


class PagedInklingPool:
    """Paged 64-token KV block allocator with zero-copy prefix sharing and contiguous conv pools for Inkling-2-Mini."""

    def __init__(
        self,
        cfg: Inkling2Config,
        *,
        init_global_blocks: int = 128,
        init_sliding_blocks: int = 96,
        max_global_blocks: int = 4096,
        max_sliding_blocks: int = 2048,
        max_conv_slots: int = 128,
        enable_prefix_cache: bool = True,
        nested_hot_blocks: int | None = None,
        nested_warm_blocks: int | None = None,
        nested_spill_to_host: bool | None = None,
        nested_coarse_topk: int | None = None,
    ):
        self.cfg = cfg
        self.block_size = PAGED_BLOCK_SIZE
        self.global_kv_mode = cfg.global_kv_mode  # "fp16" | "fp8" | "q4" | "nested"
        if self.global_kv_mode not in ("fp16", "fp8", "q4", "nested"):
            raise ValueError(f"PagedInklingPool supports fp16, fp8, q4, or nested; got {self.global_kv_mode!r}")
        self.share_loop_kv = getattr(cfg, "shared_loop_kv", False) and (cfg.loop_passes > 1)
        self.enable_prefix_cache = enable_prefix_cache
        self.max_global_blocks = max_global_blocks
        self.max_sliding_blocks = max_sliding_blocks
        self.nested_hot_blocks = _resolve_nested_opt_int(nested_hot_blocks, "LAMQ_NESTED_HOT_BLOCKS")
        self.nested_warm_blocks = _resolve_nested_opt_int(nested_warm_blocks, "LAMQ_NESTED_WARM_BLOCKS")
        self.nested_spill_to_host = (
            (os.environ.get("LAMQ_NESTED_SPILL_HOST", "0").strip().lower() in ("1", "true", "yes"))
            if nested_spill_to_host is None
            else bool(nested_spill_to_host)
        )
        self.nested_coarse_topk = (
            max(0, int(os.environ.get("LAMQ_NESTED_COARSE_TOPK", "0") or 0))
            if nested_coarse_topk is None
            else max(0, int(nested_coarse_topk))
        )

        # Contiguous GPU conv-state pools indexed by `conv_slot` (eliminates O(48*4*B) Python stack/slice overhead!)
        self.max_conv_slots = max(16, int(max_conv_slots))
        self.conv_free: list[int] = list(range(self.max_conv_slots - 1, -1, -1))
        nv = cfg.total_visits
        self.conv_pools: list[list[mx.array]] = [
            [
                mx.zeros((self.max_conv_slots, 3, 1024), dtype=mx.float32),
                mx.zeros((self.max_conv_slots, 3, 1024), dtype=mx.float32),
                mx.zeros((self.max_conv_slots, 3, 2304), dtype=mx.float32),
                mx.zeros((self.max_conv_slots, 3, 2304), dtype=mx.float32),
            ]
            for _ in range(nv)
        ]
        mx.eval(*[a for row in self.conv_pools for a in row])

        # Map each of the 48 logical visits -> (is_sliding, physical_pool_idx)
        n_loop = cfg.loop_end - cfg.loop_start
        final_loop_offset = cfg.num_hidden_layers + (cfg.loop_passes - 2) * n_loop - cfg.loop_start
        local_set = set(cfg.local_layer_ids)

        self.visit_is_sliding: list[bool] = [False] * cfg.total_visits
        self.visit_pool_idx: list[int] = [0] * cfg.total_visits
        self.visit_kv_commit: list[bool] = [True] * cfg.total_visits
        # Track canonical global visit slots (deduplicated when shared_loop_kv=True)
        g_slot_map: dict[int, int] = {}
        s_count = 0

        # Build visit table matching Inkling2LoopPackedMLX._forward_core_fast visit slot order:
        #   slots 0..5: L0..5; slots 6..17: L6..17 pass 0; slots 18..23: L18..23 (coda);
        #   slots 24..35: L6..17 pass 1; slots 36..47: L6..17 pass 2
        for slot in range(cfg.total_visits):
            if slot < cfg.num_hidden_layers:
                phys_i = slot
                is_last_loop_pass = (cfg.loop_passes == 1)
                in_loop = cfg.loop_start <= phys_i < cfg.loop_end
            else:
                rel = slot - cfg.num_hidden_layers
                loop_idx = 1 + rel // n_loop
                phys_i = cfg.loop_start + (rel % n_loop)
                is_last_loop_pass = (loop_idx == cfg.loop_passes - 1)
                in_loop = True
            is_sl = phys_i in local_set
            self.visit_is_sliding[slot] = is_sl
            if is_sl:
                self.visit_pool_idx[slot] = s_count
                s_count += 1
            else:
                sh = self.share_loop_kv and in_loop and (not is_last_loop_pass)
                canon_slot = (phys_i + final_loop_offset) if sh else slot
                self.visit_kv_commit[slot] = not sh
                if canon_slot not in g_slot_map:
                    g_slot_map[canon_slot] = len(g_slot_map)
                self.visit_pool_idx[slot] = g_slot_map[canon_slot]

        self.n_sliding_pools = s_count
        self.n_global_pools = len(g_slot_map)

        # Allocate initial pools
        self.n_g_blocks = 0
        self.g_pools: list[mx.array] = []
        self.g_scales: list[mx.array | None] = []
        self.g_base_pools: list[mx.array] = []
        self.g_nested_scales: list[mx.array] = []
        self.n_p2_blocks = 0
        self.n_p3_blocks = 0
        self.max_p2_blocks = self.max_global_blocks + max(64, self.nested_coarse_topk)
        self.max_p3_blocks = self.max_global_blocks + max(64, self.nested_coarse_topk)
        self.g_p2_pools: list[mx.array] = []
        self.g_p3_pools: list[mx.array] = []
        self.p2_free: list[int] = []
        self.p3_free: list[int] = []
        self.g_block_p2 = np.zeros((0,), dtype=np.int32)
        self.g_block_p3 = np.zeros((0,), dtype=np.int32)
        self._block_p2_mx: mx.array | None = None
        self._block_p3_mx: mx.array | None = None
        self._p2_dirty = True
        self._p3_dirty = True
        self._host_p2: dict[int, list[np.ndarray]] = {}
        self._host_p3: dict[int, list[np.ndarray]] = {}
        self._last_nested_promotions: dict[int, list[int]] = {}
        self.g_free: list[int] = []
        self.g_ref: list[int] = []
        self._grow_global(init_global_blocks)
        if self.global_kv_mode == "nested":
            self._grow_p2(init_global_blocks)
            self._grow_p3(init_global_blocks)

        self.n_s_blocks = 0
        self.s_pools: list[mx.array] = []
        self.s_free: list[int] = []
        self.s_ref: list[int] = []
        self._grow_sliding(init_sliding_blocks)

        # Prefix cache (block-hash tree)
        self._lru_clock = 0
        self._prefix_nodes: dict[bytes, _CachedBlockNode] = {}
        self._g_to_node: dict[int, bytes] = {}
        self._s_to_node: dict[int, bytes] = {}

    def eval_state(self) -> None:
        """Materialize all mutable KV and conv pools so chunked prefill never retains lazy activation graphs."""
        mx.eval(
            *self.g_pools,
            *[s for s in self.g_scales if s is not None],
            *self.g_base_pools,
            *self.g_nested_scales,
            *self.g_p2_pools,
            *self.g_p3_pools,
            *self.s_pools,
            *[a for row in self.conv_pools for a in row],
        )

    def _grow_plane_pool(
        self,
        pools: list[mx.array],
        cur_n: int,
        max_n: int,
        free_list: list[int],
        target_blocks: int,
    ) -> int:
        target = min(max_n, max(target_blocks, cur_n))
        if target <= cur_n:
            return cur_n
        new_toks = target * self.block_size
        old_toks = cur_n * self.block_size
        if cur_n == 0:
            pools[:] = [mx.zeros((new_toks, 2, 8, 4), dtype=mx.uint32) for _ in range(self.n_global_pools)]
        else:
            for i in range(self.n_global_pools):
                buf = mx.zeros((new_toks, 2, 8, 4), dtype=mx.uint32)
                buf[:old_toks] = pools[i]
                pools[i] = buf
        mx.eval(*pools)
        free_list.extend(range(cur_n, target))
        return target

    def _grow_p2(self, target_blocks: int) -> None:
        self.n_p2_blocks = self._grow_plane_pool(
            self.g_p2_pools, self.n_p2_blocks, self.max_p2_blocks, self.p2_free, target_blocks,
        )

    def _grow_p3(self, target_blocks: int) -> None:
        self.n_p3_blocks = self._grow_plane_pool(
            self.g_p3_pools, self.n_p3_blocks, self.max_p3_blocks, self.p3_free, target_blocks,
        )

    def _grow_global(self, target_blocks: int) -> None:
        target = min(self.max_global_blocks, max(target_blocks, self.n_g_blocks))
        if target <= self.n_g_blocks:
            return
        old_n = self.n_g_blocks
        new_toks = target * self.block_size
        old_toks = old_n * self.block_size
        if self.global_kv_mode == "nested":
            if old_n == 0:
                self.g_base_pools = [mx.zeros((new_toks, 2, 8, 2, 4), dtype=mx.uint32) for _ in range(self.n_global_pools)]
                self.g_nested_scales = [mx.zeros((new_toks, 2, 8), dtype=mx.float16) for _ in range(self.n_global_pools)]
            else:
                for i in range(self.n_global_pools):
                    bb = mx.zeros((new_toks, 2, 8, 2, 4), dtype=mx.uint32)
                    bb[:old_toks] = self.g_base_pools[i]
                    self.g_base_pools[i] = bb
                    sc = mx.zeros((new_toks, 2, 8), dtype=mx.float16)
                    sc[:old_toks] = self.g_nested_scales[i]
                    self.g_nested_scales[i] = sc
            mx.eval(*self.g_base_pools, *self.g_nested_scales)
            ext = np.full((target - old_n,), -1, dtype=np.int32)
            self.g_block_p2 = np.concatenate([self.g_block_p2, ext]) if old_n > 0 else ext.copy()
            self.g_block_p3 = np.concatenate([self.g_block_p3, ext]) if old_n > 0 else ext.copy()
            self._p2_dirty = True
            self._p3_dirty = True
        else:
            if self.global_kv_mode == "q4":
                dtype, dim = mx.uint8, 64
            elif self.global_kv_mode == "fp8":
                dtype, dim = mx.uint8, 128
            else:
                dtype, dim = mx.float16, 128
            if old_n == 0:
                self.g_pools = [mx.zeros((new_toks, 2, 8, dim), dtype=dtype) for _ in range(self.n_global_pools)]
                self.g_scales = (
                    [mx.zeros((new_toks, 2, 8, 4), dtype=mx.float16) for _ in range(self.n_global_pools)]
                    if self.global_kv_mode == "q4"
                    else [None] * self.n_global_pools
                )
            else:
                for i in range(self.n_global_pools):
                    buf = mx.zeros((new_toks, 2, 8, dim), dtype=dtype)
                    buf[:old_toks] = self.g_pools[i]
                    self.g_pools[i] = buf
                    if self.global_kv_mode == "q4":
                        sc = mx.zeros((new_toks, 2, 8, 4), dtype=mx.float16)
                        sc[:old_toks] = self.g_scales[i]
                        self.g_scales[i] = sc
            mx.eval(*self.g_pools, *[s for s in self.g_scales if s is not None])
        self.g_free.extend(range(old_n, target))
        self.g_ref.extend([0] * (target - old_n))
        self.n_g_blocks = target

    def _grow_sliding(self, target_blocks: int) -> None:
        target = min(self.max_sliding_blocks, max(target_blocks, self.n_s_blocks))
        if target <= self.n_s_blocks:
            return
        old_n = self.n_s_blocks
        new_toks = target * self.block_size
        old_toks = old_n * self.block_size
        if old_n == 0:
            self.s_pools = [mx.zeros((new_toks, 2, 8, 128), dtype=mx.float16) for _ in range(self.n_sliding_pools)]
        else:
            for i in range(self.n_sliding_pools):
                buf = mx.zeros((new_toks, 2, 8, 128), dtype=mx.float16)
                buf[:old_toks] = self.s_pools[i]
                self.s_pools[i] = buf
        mx.eval(*self.s_pools)
        self.s_free.extend(range(old_n, target))
        self.s_ref.extend([0] * (target - old_n))
        self.n_s_blocks = target

    def _evict_one_prefix_leaf(self, need_global: bool = False, need_sliding: bool = False) -> bool:
        """Evict the least-recently-used unreferenced leaf node from the prefix cache."""
        best_h: bytes | None = None
        best_tick = 1 << 62
        for h, node in self._prefix_nodes.items():
            if node.children:
                continue
            if self.g_ref[node.g_blk] > 0 or self.s_ref[node.s_blk] > 0:
                continue
            if node.lru_tick < best_tick:
                best_tick = node.lru_tick
                best_h = h
        if best_h is None:
            return False
        node = self._prefix_nodes.pop(best_h)
        parent = self._prefix_nodes.get(node.parent_hash)
        if parent is not None:
            parent.children.discard(best_h)
        self._g_to_node.pop(node.g_blk, None)
        self._s_to_node.pop(node.s_blk, None)
        self._free_global_block_slot(node.g_blk)
        self.s_free.append(node.s_blk)
        return True

    def _free_global_block_slot(self, blk: int) -> None:
        if self.global_kv_mode == "nested":
            p2_blk = int(self.g_block_p2[blk])
            if p2_blk >= 0:
                self.g_block_p2[blk] = -1
                self.p2_free.append(p2_blk)
                self._p2_dirty = True
            p3_blk = int(self.g_block_p3[blk])
            if p3_blk >= 0:
                self.g_block_p3[blk] = -1
                self.p3_free.append(p3_blk)
                self._p3_dirty = True
            self._host_p2.pop(blk, None)
            self._host_p3.pop(blk, None)
        self.g_free.append(blk)

    def _alloc_plane_slot(self, which: str, *, allow_evict: bool = True) -> int:
        free_list = self.p2_free if which == "p2" else self.p3_free
        while not free_list:
            cur_n = self.n_p2_blocks if which == "p2" else self.n_p3_blocks
            max_n = self.max_p2_blocks if which == "p2" else self.max_p3_blocks
            if cur_n < max_n:
                nxt = min(max_n, max(cur_n * 2, cur_n + 64, 64))
                if which == "p2":
                    self._grow_p2(nxt)
                else:
                    self._grow_p3(nxt)
            elif allow_evict and self._evict_one_prefix_leaf(need_global=True):
                continue
            else:
                raise MemoryError(f"PagedInklingPool out of nested {which} blocks")
        return free_list.pop()

    def alloc_global_block(self) -> int:
        if not self.g_free:
            if not self._evict_one_prefix_leaf(need_global=True):
                if self.n_g_blocks < self.max_global_blocks:
                    self._grow_global(min(self.max_global_blocks, max(self.n_g_blocks * 2, self.n_g_blocks + 128)))
                else:
                    raise MemoryError("PagedInklingPool out of global KV blocks")
        blk = self.g_free.pop()
        self.g_ref[blk] = 1
        if self.global_kv_mode == "nested":
            try:
                p2_blk = self._alloc_plane_slot("p2", allow_evict=True)
                p3_blk = self._alloc_plane_slot("p3", allow_evict=True)
            except Exception:
                self.g_ref[blk] = 0
                self._free_global_block_slot(blk)
                raise
            self.g_block_p2[blk] = p2_blk
            self.g_block_p3[blk] = p3_blk
            self._p2_dirty = True
            self._p3_dirty = True
            self._host_p2.pop(blk, None)
            self._host_p3.pop(blk, None)
        return blk

    def alloc_sliding_block(self) -> int:
        if not self.s_free:
            if not self._evict_one_prefix_leaf(need_sliding=True):
                if self.n_s_blocks < self.max_sliding_blocks:
                    self._grow_sliding(min(self.max_sliding_blocks, max(self.n_s_blocks * 2, self.n_s_blocks + 64)))
                else:
                    raise MemoryError("PagedInklingPool out of sliding KV blocks")
        blk = self.s_free.pop()
        self.s_ref[blk] = 1
        return blk

    def alloc_conv_slot(self, init_snap: list[list[mx.array]] | None = None) -> int:
        if not self.conv_free:
            # Grow conv slot pool dynamically
            old_n = self.max_conv_slots
            new_n = old_n * 2
            for v in range(len(self.conv_pools)):
                for ci, dim in enumerate((1024, 1024, 2304, 2304)):
                    buf = mx.zeros((new_n, 3, dim), dtype=mx.float32)
                    buf[:old_n] = self.conv_pools[v][ci]
                    self.conv_pools[v][ci] = buf
            mx.eval(*[a for row in self.conv_pools for a in row])
            self.conv_free.extend(range(new_n - 1, old_n - 1, -1))
            self.max_conv_slots = new_n
        cs = self.conv_free.pop()
        if init_snap is not None:
            for v in range(len(self.conv_pools)):
                for ci in range(4):
                    self.conv_pools[v][ci][cs] = init_snap[v][ci]
        else:
            z1 = mx.zeros((3, 1024), dtype=mx.float32)
            z2 = mx.zeros((3, 2304), dtype=mx.float32)
            for v in range(len(self.conv_pools)):
                self.conv_pools[v][0][cs] = z1
                self.conv_pools[v][1][cs] = z1
                self.conv_pools[v][2][cs] = z2
                self.conv_pools[v][3][cs] = z2
        return cs

    def snapshot_conv_slot(self, cs: int) -> list[list[mx.array]]:
        snap = [[mx.array(self.conv_pools[v][ci][cs]) for ci in range(4)] for v in range(len(self.conv_pools))]
        mx.eval(*[a for row in snap for a in row])
        return snap

    def release_conv_slot(self, cs: int) -> None:
        if cs >= 0:
            self.conv_free.append(cs)

    def release_global_block(self, blk: int) -> None:
        self.g_ref[blk] -= 1
        assert self.g_ref[blk] >= 0
        if self.g_ref[blk] == 0 and blk not in self._g_to_node:
            self._free_global_block_slot(blk)

    def free_global_block(self, blk: int) -> None:
        self.release_global_block(blk)

    def ensure_block_4bit(self, g_blk: int) -> None:
        """Ensure `g_blk` has resident `p2` and `p3` slots before writing new tokens."""
        if self.global_kv_mode != "nested":
            return
        bs = self.block_size
        if int(self.g_block_p2[g_blk]) < 0:
            p2_blk = self._alloc_plane_slot("p2", allow_evict=False)
            self.g_block_p2[g_blk] = p2_blk
            self._p2_dirty = True
            host_p2 = self._host_p2.pop(g_blk, None)
            if host_p2 is not None:
                for i, arr in enumerate(host_p2):
                    self.g_p2_pools[i][p2_blk * bs : (p2_blk + 1) * bs] = mx.array(arr)
        if int(self.g_block_p3[g_blk]) < 0:
            p3_blk = self._alloc_plane_slot("p3", allow_evict=False)
            self.g_block_p3[g_blk] = p3_blk
            self._p3_dirty = True
            host_p3 = self._host_p3.pop(g_blk, None)
            if host_p3 is not None:
                for i, arr in enumerate(host_p3):
                    self.g_p3_pools[i][p3_blk * bs : (p3_blk + 1) * bs] = mx.array(arr)

    def _sync_nested_tables(self) -> tuple[mx.array, mx.array]:
        if self._p2_dirty or self._block_p2_mx is None:
            self._block_p2_mx = mx.array(self.g_block_p2)
            self._p2_dirty = False
        if self._p3_dirty or self._block_p3_mx is None:
            self._block_p3_mx = mx.array(self.g_block_p3)
            self._p3_dirty = False
        return self._block_p2_mx, self._block_p3_mx

    def age_block_planes(
        self,
        blocks: Sequence[int],
        hot_blocks: int,
        warm_blocks: int,
        *,
        preserve_sink: bool = True,
    ) -> dict[str, int]:
        """Zero-recompute 4->3->2 aging across `blocks`: keeps sink + hot tail at 4b, warm middle at 3b, older at 2b."""
        if self.global_kv_mode != "nested" or not blocks:
            return {"dropped_p3": 0, "dropped_p2": 0, "skipped_shared": 0}
        n = len(blocks)
        hot = max(0, int(hot_blocks))
        warm = max(0, int(warm_blocks))
        keep_sink = bool(preserve_sink) and (n > hot + 1)
        hot_start = max(0, n - hot)
        warm_start = max(1 if keep_sink else 0, hot_start - warm)

        drop_p3: list[tuple[int, int]] = []
        drop_p2: list[tuple[int, int]] = []
        skipped_shared = 0
        for idx, raw_gb in enumerate(blocks):
            gb = int(raw_gb)
            if (keep_sink and idx == 0) or idx >= hot_start:
                continue
            if self.g_ref[gb] > 1:
                skipped_shared += 1
                continue
            p3_blk = int(self.g_block_p3[gb])
            if p3_blk >= 0:
                drop_p3.append((gb, p3_blk))
            if idx < warm_start:
                p2_blk = int(self.g_block_p2[gb])
                if p2_blk >= 0:
                    drop_p2.append((gb, p2_blk))

        if self.nested_spill_to_host and (drop_p3 or drop_p2):
            bs = self.block_size
            to_eval: list[mx.array] = []
            if drop_p3:
                to_eval.extend(self.g_p3_pools)
            if drop_p2:
                to_eval.extend(self.g_p2_pools)
            if to_eval:
                mx.eval(*to_eval)
            for gb, p3_blk in drop_p3:
                lo, hi = p3_blk * bs, (p3_blk + 1) * bs
                self._host_p3[gb] = [np.array(p[lo:hi], dtype=np.uint32, copy=True) for p in self.g_p3_pools]
            for gb, p2_blk in drop_p2:
                lo, hi = p2_blk * bs, (p2_blk + 1) * bs
                self._host_p2[gb] = [np.array(p[lo:hi], dtype=np.uint32, copy=True) for p in self.g_p2_pools]

        for gb, p3_blk in drop_p3:
            self.g_block_p3[gb] = -1
            self.p3_free.append(p3_blk)
            self._p3_dirty = True
        for gb, p2_blk in drop_p2:
            self.g_block_p2[gb] = -1
            self.p2_free.append(p2_blk)
            self._p2_dirty = True
        return {"dropped_p3": len(drop_p3), "dropped_p2": len(drop_p2), "skipped_shared": skipped_shared}

    def nested_memory_report(self) -> dict[str, Any]:
        """Report exact GPU/host byte counts, precision tier counts, and effective bits/value."""
        if self.global_kv_mode != "nested":
            gpu_bytes = sum(a.nbytes for a in self.g_pools) + sum(s.nbytes for s in self.g_scales if s is not None)
            bpv = 16.0 if self.global_kv_mode == "fp16" else (8.0 if self.global_kv_mode == "fp8" else 4.25)
            return {
                "mode": self.global_kv_mode,
                "n_global_pools": self.n_global_pools,
                "n_g_blocks": self.n_g_blocks,
                "active_g_blocks": self.n_g_blocks - len(self.g_free),
                "gpu_allocated_bytes": gpu_bytes,
                "effective_bits_per_value": bpv,
            }
        n_pools = self.n_global_pools
        free_set = set(self.g_free)
        active_blks = [b for b in range(self.n_g_blocks) if b not in free_set]
        n_active = len(active_blks)
        n4 = sum(1 for b in active_blks if int(self.g_block_p2[b]) >= 0 and int(self.g_block_p3[b]) >= 0)
        n3 = sum(1 for b in active_blks if int(self.g_block_p2[b]) >= 0 and int(self.g_block_p3[b]) < 0)
        n2 = sum(1 for b in active_blks if int(self.g_block_p2[b]) < 0 and int(self.g_block_p3[b]) < 0)

        # Per block per pool: base=32768 B (2.0 b/v), scale=2048 B (0.125 b/v), p2=16384 B (1.0 b/v), p3=16384 B (1.0 b/v)
        base_alloc = sum(a.nbytes for a in self.g_base_pools)
        scale_alloc = sum(a.nbytes for a in self.g_nested_scales)
        p2_alloc = sum(a.nbytes for a in self.g_p2_pools)
        p3_alloc = sum(a.nbytes for a in self.g_p3_pools)

        base_active = n_active * n_pools * 32768
        scale_active = n_active * n_pools * 2048
        p2_active = (n4 + n3) * n_pools * 16384
        p3_active = n4 * n_pools * 16384
        gpu_alloc = base_alloc + scale_alloc + p2_alloc + p3_alloc
        gpu_active = base_active + scale_active + p2_active + p3_active
        host_bytes = (
            sum(arr.nbytes for lst in self._host_p2.values() for arr in lst)
            + sum(arr.nbytes for lst in self._host_p3.values() for arr in lst)
        )
        total_vals = n_active * n_pools * self.block_size * 2 * 8 * 128
        avg_bpv = (gpu_active * 8.0 / total_vals) if total_vals > 0 else 0.0
        return {
            "mode": self.global_kv_mode,
            "n_global_pools": n_pools,
            "n_g_blocks": self.n_g_blocks,
            "active_g_blocks": n_active,
            "blocks_4bit": n4,
            "blocks_3bit": n3,
            "blocks_2bit": n2,
            "base_bytes": base_alloc,
            "base_active_bytes": base_active,
            "scale_bytes": scale_alloc,
            "scale_active_bytes": scale_active,
            "p2_allocated_bytes": p2_alloc,
            "p2_active_bytes": p2_active,
            "p3_allocated_bytes": p3_alloc,
            "p3_active_bytes": p3_active,
            "gpu_allocated_bytes": gpu_alloc,
            "gpu_active_bytes": gpu_active,
            "host_spilled_bytes": host_bytes,
            "effective_bits_per_value": avg_bpv,
            "p2_free_blocks": len(self.p2_free),
            "p3_free_blocks": len(self.p3_free),
        }

    def release_sliding_block(self, blk: int) -> None:
        self.s_ref[blk] -= 1
        assert self.s_ref[blk] >= 0
        if self.s_ref[blk] == 0 and blk not in self._s_to_node:
            self.s_free.append(blk)

    def match_prefix(self, token_ids: Sequence[int]) -> tuple[list[int], list[int], int, list[list[mx.array]] | None, bytes]:
        """Match longest full-block prefix in cache (always leaving >= 1 token to run forward).

        Returns `(g_blocks, s_blocks, sliding_start, conv_snap, last_hash)`.
        """
        if not self.enable_prefix_cache or len(token_ids) <= self.block_size:
            return [], [], 0, None, b""
        max_full = (len(token_ids) - 1) // self.block_size
        cur_hash = b""
        matched: list[_CachedBlockNode] = []
        self._lru_clock += 1
        for bi in range(max_full):
            btok = tuple(int(x) for x in token_ids[bi * self.block_size : (bi + 1) * self.block_size])
            nh = _block_hash(cur_hash, btok)
            node = self._prefix_nodes.get(nh)
            if node is None:
                break
            node.lru_tick = self._lru_clock
            matched.append(node)
            cur_hash = nh
        if not matched:
            return [], [], 0, None, b""
        # Find the deepest matched node that has a valid conv_snap
        while matched and matched[-1].conv_snap is None:
            matched.pop()
        if not matched:
            return [], [], 0, None, b""
        g_blks = [n.g_blk for n in matched]
        for gb in g_blks:
            self.g_ref[gb] += 1
        keep_s = min(len(matched), _MAX_SLIDING_BLOCKS)
        s_nodes = matched[-keep_s:]
        s_blks = [n.s_blk for n in s_nodes]
        for sb in s_blks:
            self.s_ref[sb] += 1
        sliding_start = (len(matched) - keep_s) * self.block_size
        last = matched[-1]
        conv_copy = [[mx.array(a) for a in row] for row in last.conv_snap]
        return g_blks, s_blks, sliding_start, conv_copy, last.block_hash

    def register_prefix_block(
        self,
        parent_hash: bytes,
        block_tokens: tuple[int, ...],
        g_blk: int,
        s_blk: int,
        conv_snap: list[list[mx.array]] | None,
    ) -> tuple[bytes, int, int]:
        """Register a completed 64-token block into the prefix cache (deduplicating if already cached).

        Returns `(block_hash, canonical_g_blk, canonical_s_blk)` — if deduplicated, caller should
        swap its reference to the canonical blocks (already ref-incremented here) and release its duplicates.
        """
        if not self.enable_prefix_cache or len(block_tokens) != self.block_size:
            return b"", g_blk, s_blk
        nh = _block_hash(parent_hash, block_tokens)
        self._lru_clock += 1
        existing = self._prefix_nodes.get(nh)
        if existing is not None:
            existing.lru_tick = self._lru_clock
            if existing.conv_snap is None and conv_snap is not None:
                existing.conv_snap = [[mx.array(a) for a in row] for row in conv_snap]
                mx.eval(*[a for row in existing.conv_snap for a in row])
            self.g_ref[existing.g_blk] += 1
            self.s_ref[existing.s_blk] += 1
            return nh, existing.g_blk, existing.s_blk
        # Only register if neither physical block is already mapped to another hash
        if g_blk in self._g_to_node or s_blk in self._s_to_node:
            return b"", g_blk, s_blk
        snap_copy = [[mx.array(a) for a in row] for row in conv_snap] if conv_snap is not None else None
        if snap_copy is not None:
            mx.eval(*[a for row in snap_copy for a in row])
        node = _CachedBlockNode(
            block_hash=nh,
            parent_hash=parent_hash,
            g_blk=g_blk,
            s_blk=s_blk,
            conv_snap=snap_copy,
            lru_tick=self._lru_clock,
        )
        self._prefix_nodes[nh] = node
        self._g_to_node[g_blk] = nh
        self._s_to_node[s_blk] = nh
        parent = self._prefix_nodes.get(parent_hash)
        if parent is not None:
            parent.children.add(nh)
        return nh, g_blk, s_blk

    def write_and_attend(
        self,
        visit_slot: int,
        q: mx.array,
        kv_f16: mx.array,  # [T, 2, 8, 128]
        rel_st: mx.array,
        rel_proj_t_f16: mx.array,
        q_in: mx.array,
        sink_w_f16: mx.array,
        meta: "_BatchStepMeta",
    ) -> mx.array:
        is_sl = self.visit_is_sliding[visit_slot]
        pidx = self.visit_pool_idx[visit_slot]
        if is_sl:
            self.s_pools[pidx][meta.s_write_idx] = kv_f16
            return gqa_paged_sdpa_fast(
                q, self.s_pools[pidx], rel_st, rel_proj_t_f16, q_in, sink_w_f16,
                meta.s_bt, meta.s_lens, meta.q_pos0, meta.s_starts,
                meta.B, meta.K, meta.s_max_span, True, kv_mode="fp16",
            )
        mode = self.global_kv_mode
        if mode == "nested":
            return self._write_and_attend_nested(pidx, q, kv_f16, rel_st, rel_proj_t_f16, q_in, sink_w_f16, meta)
        if mode == "fp16":
            self.g_pools[pidx][meta.g_write_idx] = kv_f16
            kv_sc = None
        elif mode == "fp8":
            self.g_pools[pidx][meta.g_write_idx] = fp8_e4m3_quantize_f16_to_u8(kv_f16)
            kv_sc = None
        else:
            q_u8, q_sc = kv_h128_q4_quantize(kv_f16)
            self.g_pools[pidx][meta.g_write_idx] = q_u8
            self.g_scales[pidx][meta.g_write_idx] = q_sc
            kv_sc = self.g_scales[pidx]
        return gqa_paged_sdpa_fast(
            q, self.g_pools[pidx], rel_st, rel_proj_t_f16, q_in, sink_w_f16,
            meta.g_bt, meta.g_lens, meta.q_pos0, meta.g_starts,
            meta.B, meta.K, meta.g_max_span, False, kv_mode=mode, kv_q4_scales=kv_sc,
        )

    def _write_and_attend_nested(
        self,
        pidx: int,
        q: mx.array,
        kv_f16: mx.array,
        rel_st: mx.array,
        rel_proj_t_f16: mx.array,
        q_in: mx.array,
        sink_w_f16: mx.array,
        meta: "_BatchStepMeta",
    ) -> mx.array:
        bs = self.block_size
        base_u32, sc_f16, p2_u32, p3_u32 = kv_h128_nested4_quantize(kv_f16)
        self.g_base_pools[pidx][meta.g_write_idx] = base_u32
        self.g_nested_scales[pidx][meta.g_write_idx] = sc_f16

        if meta.g_p2_write_idx is not None and meta.g_p3_write_idx is not None:
            p2_w, p3_w = meta.g_p2_write_idx, meta.g_p3_write_idx
        else:
            gw = np.asarray(meta.g_write_idx, dtype=np.int64)
            for gb in np.unique(gw // bs):
                self.ensure_block_4bit(int(gb))
            p2_w = mx.array(self.g_block_p2[gw // bs] * bs + (gw % bs), dtype=mx.int32)
            p3_w = mx.array(self.g_block_p3[gw // bs] * bs + (gw % bs), dtype=mx.int32)
        self.g_p2_pools[pidx][p2_w] = p2_u32
        self.g_p3_pools[pidx][p3_w] = p3_u32

        if self.nested_coarse_topk > 0 and self.nested_spill_to_host and self._host_p2:
            scores = gqa_paged_nested_coarse_score(
                q, self.g_base_pools[pidx], self.g_nested_scales[pidx],
                rel_st, rel_proj_t_f16, meta.g_bt, meta.g_lens, meta.q_pos0, meta.g_starts,
                meta.B, meta.K, reduce_q=True,
            )
            blk_scores_np = np.asarray(mx.max(scores, axis=1))
            bt_np = np.asarray(meta.g_bt)
            lens_np = np.asarray(meta.g_lens)
            promote_set: set[int] = set()
            for b in range(meta.B):
                nb = (int(lens_np[b]) + bs - 1) // bs
                cands = []
                seen_b: set[int] = set()
                for bi in range(nb):
                    gb = int(bt_np[b, bi])
                    if gb in seen_b:
                        continue
                    seen_b.add(gb)
                    if int(self.g_block_p2[gb]) < 0 and gb in self._host_p2:
                        cands.append((float(blk_scores_np[b, bi]), gb))
                if cands:
                    cands.sort(key=lambda x: x[0], reverse=True)
                    for _, gb in cands[: self.nested_coarse_topk]:
                        promote_set.add(gb)
            if promote_set:
                promote_blks = sorted(promote_set)
                scratch_p2: list[int] = []
                scratch_p3: list[int] = []
                tmp_p2 = np.array(self.g_block_p2, copy=True)
                tmp_p3 = np.array(self.g_block_p3, copy=True)
                p2_idxs: list[np.ndarray] = []
                p2_vals: list[np.ndarray] = []
                p3_idxs: list[np.ndarray] = []
                p3_vals: list[np.ndarray] = []
                try:
                    for gb in promote_blks:
                        sp2 = self._alloc_plane_slot("p2", allow_evict=False)
                        scratch_p2.append(sp2)
                        tmp_p2[gb] = sp2
                        p2_idxs.append(np.arange(sp2 * bs, (sp2 + 1) * bs, dtype=np.int32))
                        p2_vals.append(self._host_p2[gb][pidx])
                        if gb in self._host_p3:
                            sp3 = self._alloc_plane_slot("p3", allow_evict=False)
                            scratch_p3.append(sp3)
                            tmp_p3[gb] = sp3
                            p3_idxs.append(np.arange(sp3 * bs, (sp3 + 1) * bs, dtype=np.int32))
                            p3_vals.append(self._host_p3[gb][pidx])
                    self.g_p2_pools[pidx][mx.array(np.concatenate(p2_idxs))] = mx.array(np.concatenate(p2_vals, axis=0))
                    if p3_idxs:
                        self.g_p3_pools[pidx][mx.array(np.concatenate(p3_idxs))] = mx.array(np.concatenate(p3_vals, axis=0))
                    self._last_nested_promotions[pidx] = promote_blks
                    return gqa_paged_nested_sdpa_fast(
                        q, self.g_base_pools[pidx], self.g_nested_scales[pidx],
                        self.g_p2_pools[pidx], self.g_p3_pools[pidx],
                        rel_st, rel_proj_t_f16, q_in, sink_w_f16,
                        meta.g_bt, mx.array(tmp_p2), mx.array(tmp_p3),
                        meta.g_lens, meta.q_pos0, meta.g_starts, meta.B, meta.K, meta.g_max_span, False,
                    )
                finally:
                    self.p2_free.extend(scratch_p2)
                    self.p3_free.extend(scratch_p3)

        self._last_nested_promotions[pidx] = []
        bp2_mx, bp3_mx = self._sync_nested_tables()
        return gqa_paged_nested_sdpa_fast(
            q, self.g_base_pools[pidx], self.g_nested_scales[pidx],
            self.g_p2_pools[pidx], self.g_p3_pools[pidx],
            rel_st, rel_proj_t_f16, q_in, sink_w_f16,
            meta.g_bt, bp2_mx, bp3_mx,
            meta.g_lens, meta.q_pos0, meta.g_starts, meta.B, meta.K, meta.g_max_span, False,
        )


@dataclass
class _BatchStepMeta:
    B: int
    K: int
    g_write_idx: mx.array
    s_write_idx: mx.array
    g_bt: mx.array
    s_bt: mx.array
    g_lens: mx.array
    s_lens: mx.array
    q_pos0: mx.array
    q_pos_tok: mx.array
    g_starts: mx.array
    s_starts: mx.array
    g_max_span: int
    s_max_span: int
    g_p2_write_idx: mx.array | None = None
    g_p3_write_idx: mx.array | None = None


class BatchSequence:
    """Per-agent state inside `PagedInklingPool`."""

    def __init__(
        self,
        seq_id: int,
        prompt_ids: Sequence[int],
        pool: PagedInklingPool,
        *,
        max_tokens: int = 512,
        eos_ids: Sequence[int] = (200006,),
        temp: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        max_thinking_tokens: int | None = None,
        use_prefix_cache: bool = True,
    ):
        self.seq_id = seq_id
        self.prompt_ids = [int(x) for x in prompt_ids]
        self.pool = pool
        self.max_tokens = int(max_tokens)
        self.eos_ids = set(int(x) for x in eos_ids)
        self.temp = float(temp)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.use_prefix_cache = use_prefix_cache
        self.th_state = _ThinkCapState(max_thinking_tokens) if max_thinking_tokens is not None else None
        self.pending_inject: list[int] = []

        self.g_blocks: list[int] = []
        self.s_blocks: list[int] = []
        self.sliding_start: int = 0
        self.seq_len: int = 0
        self.prefill_pos: int = 0
        self.reused_tokens: int = 0
        self.all_tokens: list[int] = []
        self.generated: list[int] = []
        self.last_token: int = -1
        self.done: bool = False
        self.finish_reason: str | None = None
        self.prefix_hash: bytes = b""
        self.registered_blocks: int = 0
        self.released: bool = False

        # Optional per-sequence speculative verify rollback buffers & drafter state
        self._spec_base_len: int = 0
        self._spec_conv_prev: list[list[mx.array]] | None = None
        self._spec_conv_raw: list[list[mx.array]] | None = None
        self.drafter_state: dict[str, Any] | None = None

        # Zero-copy prefix cache match!
        conv_snap: list[list[mx.array]] | None = None
        if use_prefix_cache and pool.enable_prefix_cache:
            g_blks, s_blks, s_start, conv_snap, p_hash = pool.match_prefix(self.prompt_ids)
            if g_blks:
                self.g_blocks = g_blks
                self.s_blocks = s_blks
                self.sliding_start = s_start
                matched_toks = len(g_blks) * pool.block_size
                self.seq_len = matched_toks
                self.prefill_pos = matched_toks
                self.reused_tokens = matched_toks
                self.all_tokens = list(self.prompt_ids[:matched_toks])
                self.prefix_hash = p_hash
                self.registered_blocks = len(g_blks)

        # Allocate one contiguous GPU conv slot in `pool.conv_pools` (eliminates O(48*4*B) Python stack/slice overhead!)
        self.conv_slot: int = pool.alloc_conv_slot(conv_snap)

    def ensure_capacity_for(self, k_tokens: int) -> None:
        """Allocate new private tail blocks so `seq_len + k_tokens` fits in both global & sliding block lists."""
        bs = self.pool.block_size
        need_g = (self.seq_len + k_tokens + bs - 1) // bs
        while len(self.g_blocks) < need_g:
            self.g_blocks.append(self.pool.alloc_global_block())
        s_need_toks = (self.seq_len - self.sliding_start) + k_tokens
        need_s = (s_need_toks + bs - 1) // bs
        while len(self.s_blocks) < need_s:
            self.s_blocks.append(self.pool.alloc_sliding_block())

    def commit_tokens(self, tokens: Sequence[int], *, register_prefix: bool = True) -> None:
        """Advance `seq_len` by `len(tokens)`, register completed 64-tok blocks into prefix cache, and trim old sliding blocks."""
        bs = self.pool.block_size
        self.all_tokens.extend(int(t) for t in tokens)
        self.seq_len += len(tokens)

        # Register newly completed 64-token prefix blocks (only when a block just finished cleanly at the tip for exact conv_snap)
        full_blocks = self.seq_len // bs
        if register_prefix and self.use_prefix_cache and self.pool.enable_prefix_cache and full_blocks > self.registered_blocks:
            for bi in range(self.registered_blocks, full_blocks):
                g_blk = self.g_blocks[bi]
                s_idx = bi - (self.sliding_start // bs)
                if s_idx < 0 or s_idx >= len(self.s_blocks):
                    self.prefix_hash = b""
                    break
                s_blk = self.s_blocks[s_idx]
                btok = tuple(self.all_tokens[bi * bs : (bi + 1) * bs])
                # Exact conv_snap is valid ONLY at the exact end of the sequence!
                snap = self.pool.snapshot_conv_slot(self.conv_slot) if ((bi + 1) * bs == self.seq_len) else None
                nh, cg, cs = self.pool.register_prefix_block(self.prefix_hash, btok, g_blk, s_blk, snap)
                if not nh:
                    self.prefix_hash = b""
                    break
                if cg != g_blk:
                    self.pool.release_global_block(g_blk)
                    self.g_blocks[bi] = cg
                if cs != s_blk:
                    self.pool.release_sliding_block(s_blk)
                    self.s_blocks[s_idx] = cs
                self.prefix_hash = nh
            self.registered_blocks = full_blocks

        # O(1) sliding-window block eviction (never keeps > _MAX_SLIDING_BLOCKS = 1,152 tokens)
        while len(self.s_blocks) > _MAX_SLIDING_BLOCKS:
            old_sb = self.s_blocks.pop(0)
            self.pool.release_sliding_block(old_sb)
            self.sliding_start += bs

        if self.pool.global_kv_mode == "nested" and self.pool.nested_hot_blocks is not None and full_blocks > 0:
            warm = self.pool.nested_warm_blocks if self.pool.nested_warm_blocks is not None else 0
            self.pool.age_block_planes(self.g_blocks[:full_blocks], self.pool.nested_hot_blocks, warm)

    def rollback_spec(self, k_accept: int, K: int) -> None:
        """Roll back KV block tail and 4-tap conv states after verifying `K` tokens and accepting `k_accept` (`1..K`)."""
        drop = K - k_accept
        if drop > 0 and self._spec_conv_prev is not None and self._spec_conv_raw is not None:
            cs = self.conv_slot
            for v in range(len(self.pool.conv_pools)):
                for ci in range(4):
                    prev = self._spec_conv_prev[v][ci]  # [3, C]
                    raw = self._spec_conv_raw[v][ci]    # [K, C]
                    if k_accept >= 3:
                        self.pool.conv_pools[v][ci][cs] = raw[k_accept - 3 : k_accept]
                    else:
                        self.pool.conv_pools[v][ci][cs] = mx.concatenate([prev[k_accept:], raw[:k_accept]], axis=0)
        self._spec_conv_prev = None
        self._spec_conv_raw = None

        # Free any extra trailing blocks allocated purely for rejected tokens
        bs = self.pool.block_size
        target_len = self._spec_base_len + k_accept
        need_g = (target_len + bs - 1) // bs
        while len(self.g_blocks) > need_g:
            self.pool.release_global_block(self.g_blocks.pop())
        need_s = ((target_len - self.sliding_start) + bs - 1) // bs
        while len(self.s_blocks) > need_s:
            self.pool.release_sliding_block(self.s_blocks.pop())

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        for gb in self.g_blocks:
            self.pool.release_global_block(gb)
        for sb in self.s_blocks:
            self.pool.release_sliding_block(sb)
        self.g_blocks.clear()
        self.s_blocks.clear()
        self.pool.release_conv_slot(self.conv_slot)
        self.conv_slot = -1
        self.drafter_state = None


def _prepare_batch_meta(pool: PagedInklingPool, seqs: list[BatchSequence], K: int) -> _BatchStepMeta:
    B = len(seqs)
    bs = pool.block_size
    for s in seqs:
        s.ensure_capacity_for(K)

    g_max_b = max(len(s.g_blocks) for s in seqs)
    s_max_b = max(len(s.s_blocks) for s in seqs)
    g_bt_np = np.zeros((B, g_max_b), dtype=np.uint32)
    s_bt_np = np.zeros((B, s_max_b), dtype=np.uint32)
    g_lens_np = np.zeros((B,), dtype=np.uint32)
    s_lens_np = np.zeros((B,), dtype=np.uint32)
    q_pos0_np = np.zeros((B,), dtype=np.uint32)
    q_pos_tok_np = np.zeros((B * K,), dtype=np.uint32)
    g_starts_np = np.zeros((B,), dtype=np.uint32)
    s_starts_np = np.zeros((B,), dtype=np.uint32)
    g_write_np = np.zeros((B * K,), dtype=np.int32)
    s_write_np = np.zeros((B * K,), dtype=np.int32)
    is_nested = pool.global_kv_mode == "nested"
    g_p2_write_np = np.zeros((B * K,), dtype=np.int32) if is_nested else None
    g_p3_write_np = np.zeros((B * K,), dtype=np.int32) if is_nested else None

    g_max_span = 0
    s_max_span = 0
    for b, s in enumerate(seqs):
        g_bt_np[b, : len(s.g_blocks)] = s.g_blocks
        s_bt_np[b, : len(s.s_blocks)] = s.s_blocks
        g_len = s.seq_len + K
        s_len = (s.seq_len - s.sliding_start) + K
        g_lens_np[b] = g_len
        s_lens_np[b] = s_len
        q_pos0_np[b] = s.seq_len
        s_starts_np[b] = s.sliding_start
        g_max_span = max(g_max_span, g_len)
        s_max_span = max(s_max_span, s_len)
        for lt in range(K):
            gp = s.seq_len + lt
            sp = (s.seq_len - s.sliding_start) + lt
            t = b * K + lt
            q_pos_tok_np[t] = gp
            gb = s.g_blocks[gp // bs]
            goff = gp % bs
            g_write_np[t] = gb * bs + goff
            if is_nested:
                pool.ensure_block_4bit(gb)
                g_p2_write_np[t] = int(pool.g_block_p2[gb]) * bs + goff
                g_p3_write_np[t] = int(pool.g_block_p3[gb]) * bs + goff
            s_write_np[t] = s.s_blocks[sp // bs] * bs + (sp % bs)

    return _BatchStepMeta(
        B=B,
        K=K,
        g_write_idx=mx.array(g_write_np),
        s_write_idx=mx.array(s_write_np),
        g_bt=mx.array(g_bt_np),
        s_bt=mx.array(s_bt_np),
        g_lens=mx.array(g_lens_np),
        s_lens=mx.array(s_lens_np),
        q_pos0=mx.array(q_pos0_np),
        q_pos_tok=mx.array(q_pos_tok_np),
        g_starts=mx.array(g_starts_np),
        s_starts=mx.array(s_starts_np),
        g_max_span=g_max_span,
        s_max_span=s_max_span,
        g_p2_write_idx=mx.array(g_p2_write_np) if is_nested else None,
        g_p3_write_idx=mx.array(g_p3_write_np) if is_nested else None,
    )


def forward_batched(
    model: Inkling2LoopPackedMLX,
    token_ids: mx.array | Sequence[int],
    pool: PagedInklingPool,
    seqs: list[BatchSequence],
    K: int = 1,
    *,
    skip_logits: bool = False,
    last_only: bool = False,
    tap_visits: tuple[int, ...] = (),
    record_spec: bool = False,
) -> tuple[mx.array | None, mx.array | None]:
    """Run `B = len(seqs)` independent sequences x `K` tokens through `Inkling2LoopPackedMLX` in one pass.

    Returns `(ordered_logits, taps)` where:
      - `ordered_logits`: `float32[B, 1, V]` if `last_only`, `float32[B, K, V]` if not `last_only`, or `None` if `skip_logits`.
      - `taps`: `float32[T, len(tap_visits), H]` if `tap_visits` is non-empty, else `None`.
    """
    cfg = model.cfg
    B = len(seqs)
    T = B * K
    ids_mx = token_ids if isinstance(token_ids, mx.array) else mx.array(np.asarray(token_ids, dtype=np.int32))
    ids_mx = ids_mx.reshape(1, T)
    meta = _prepare_batch_meta(pool, seqs, K)

    slot_ids_mx = mx.array(np.asarray([s.conv_slot for s in seqs], dtype=np.int32))
    if record_spec:
        for s in seqs:
            s._spec_base_len = s.seq_len
            s._spec_conv_prev = [[mx.array(pool.conv_pools[v][ci][s.conv_slot]) for ci in range(4)] for v in range(cfg.total_visits)]
            s._spec_conv_raw = [[None, None, None, None] for _ in range(cfg.total_visits)]

    h = model.embed(ids_mx)[0]  # [T, 2304]
    blocks_cat: mx.array | None = None
    n_blocks = 0
    tap_set = set(tap_visits)
    captured: dict[int, mx.array] = {}
    visit = 0

    def _gather_conv(slot: int, ci: int) -> mx.array:
        return pool.conv_pools[slot][ci][slot_ids_mx]  # [B, 3, C] in ONE GPU op!

    def _scatter_conv(slot: int, ci: int, new_prev: mx.array, raw_tkc: mx.array | None = None) -> None:
        pool.conv_pools[slot][ci][slot_ids_mx] = new_prev
        if record_spec and raw_tkc is not None:
            for b, s in enumerate(seqs):
                if s._spec_conv_raw is not None:
                    s._spec_conv_raw[slot][ci] = raw_tkc[b]

    def _run_layer(L, h_in: mx.array, b_cat, n_b: int, c_slot: int, msgs=None, wq_c=None, wo_c=None) -> mx.array:
        block_residual = h_in
        use_qkv_sg8 = (4 <= T <= 8) if _is_m4() else (5 <= T <= 8)
        if L.is_sliding:
            if msgs is None:
                if L.attn_in_q is not None:
                    q_in = block_attn_res_fused(L.attn_in_q, b_cat, n_b, h_in, norm_w=L.in_ln)[0]
                else:
                    q_in = rms_norm(h_in, L.in_ln, cfg.rms_norm_eps)
                x_rot5 = rht_in_2304_fast(q_in, L.qkv_stage.signs_in, n_signs=5, T=T, split_qkv=False, out_dtype=mx.float16, transpose_for_sg8=use_qkv_sg8)
            else:
                h_a = block_attn_res_fused(L.attn_in_q, b_cat, n_b, h_in)[0]
                fg_mg3 = _msg_3x_shared_h_compiled(h_a, msgs[:3], L.qkv_dh3, L.qkv_dm3, L.qkv_fm3)
                qkv_in = msg_blend_norm_3x_fused(h_a, msgs[:3], fg_mg3, L.in_ln, T, h_shared=True, norm_shared=True)
                q_in = qkv_in[0]
                x_rot5 = rht_in_2304_fast(qkv_in, L.qkv_stage.signs_in, n_signs=5, T=T, split_qkv=True, out_dtype=mx.float16, transpose_for_sg8=use_qkv_sg8)
        else:
            if L.attn_in_q is None:
                qkv_h = mx.broadcast_to(h_in[None, :, :], (3, T, 2304))
            else:
                qkv_h = block_attn_res_fused(L.attn_in_q, b_cat, n_b, h_in)
            if msgs is not None:
                fg_mg3 = _msg_3x_split_h_compiled(qkv_h, msgs[:3], L.qkv_dh3, L.qkv_dm3, L.qkv_fm3)
                qkv_in = msg_blend_norm_3x_fused(qkv_h, msgs[:3], fg_mg3, L.qkv_in_ln, T, h_shared=False, norm_shared=False)
                q_in, k_in, v_in = qkv_in[0], qkv_in[1], qkv_in[2]
            else:
                q_in = rms_norm(qkv_h[0], L.q_in_ln, cfg.rms_norm_eps)
                k_in = rms_norm(qkv_h[1], L.k_in_ln, cfg.rms_norm_eps)
                v_in = rms_norm(qkv_h[2], L.v_in_ln, cfg.rms_norm_eps)
                qkv_in = mx.stack([q_in, k_in, v_in], axis=0)
            x_rot5 = rht_in_2304_fast(qkv_in, L.qkv_stage.signs_in, n_signs=5, T=T, split_qkv=True, out_dtype=mx.float16, transpose_for_sg8=use_qkv_sg8)

        # 2. Fused 5-matrix QKV + Rel + O_Gate
        use_split_parts = (T > 8) and not (L.is_sliding and not L.has_qkv_out_rot)
        if use_split_parts:
            q_raw, k_raw, v_raw, rel_raw, o_gate_raw = L.qkv_stage.matmul_rotated_f16_parts(x_rot5, T, W_cached=wq_c)
            y_qkv = None
        else:
            y_qkv = (
                (
                    L.qkv_stage.matmul_rotated_sg8_f16(x_rot5, T=T)
                    if _is_m4()
                    else L.qkv_stage.matmul_rotated_sg8_f16(x_rot5, T=T, rb=1, sg=8, ntg=160)
                )
                if use_qkv_sg8
                else L.qkv_stage.matmul_rotated_f16(x_rot5, T, W_cached=wq_c)
            )
            o_gate_raw = y_qkv[:, 5504:8576]
        if L.qkv_out_rots[4] is not None:
            o_gate_raw = rht_apply_out(o_gate_raw, L.qkv_out_rots[4])

        # 3. Batched Q/Rel clip + K/V 4-tap conv -> paged [T, 2, 8, 128]
        # When B == 1 and K > 1 (single-sequence prefill), use T-parallel fast_kernels instead of sequential-K batched convs.
        single_pf = (B == 1) and (K > 1)
        kp = _gather_conv(c_slot, 0)
        vp = _gather_conv(c_slot, 1)
        if L.is_sliding and not L.has_qkv_out_rot:
            if single_pf:
                q, rel_st, kv_28k, new_kp, new_vp = qkv_rel_prep_sliding_fused(
                    y_qkv, L.q_scale, L.rel_scale,
                    kp, L.k_conv_w, vp, L.v_conv_w,
                    K, cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold,
                )
                kv_f16 = kv_28k.transpose(2, 0, 1, 3)
            else:
                q, rel_st, kv_f16, new_kp, new_vp = qkv_rel_prep_sliding_batched(
                    y_qkv, L.q_scale, L.rel_scale,
                    kp, L.k_conv_w, vp, L.v_conv_w,
                    B, K, cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold,
                )
            _scatter_conv(c_slot, 0, new_kp, y_qkv[:, 3072:4096].reshape(B, K, 1024) if record_spec else None)
            _scatter_conv(c_slot, 1, new_vp, y_qkv[:, 4096:5120].reshape(B, K, 1024) if record_spec else None)
        else:
            if not use_split_parts:
                q_raw = y_qkv[:, 0:3072]
                k_raw = y_qkv[:, 3072:4096]
                v_raw = y_qkv[:, 4096:5120]
                rel_raw = y_qkv[:, 5120:5504]
            if L.qkv_out_rots[0] is not None:
                q_raw = rht_apply_out(q_raw, L.qkv_out_rots[0])
            if L.qkv_out_rots[1] is not None:
                k_raw = rht_out_1024_fast(k_raw, L.qkv_out_rots[1]) if k_raw.shape[-1] == 1024 else rht_apply_out(k_raw, L.qkv_out_rots[1])
            if L.qkv_out_rots[2] is not None:
                v_raw = rht_out_1024_fast(v_raw, L.qkv_out_rots[2]) if v_raw.shape[-1] == 1024 else rht_apply_out(v_raw, L.qkv_out_rots[2])
            if L.qkv_out_rots[3] is not None:
                rel_raw = rht_apply_out(rel_raw, L.qkv_out_rots[3])
            # Exact per-token position for global log_scaling via tree_depth=meta.q_pos_tok with q_pos0=0!
            q, rel_st = q_rel_clip_fused(
                q_raw, L.q_scale, rel_raw, L.rel_scale,
                0, T, L.is_sliding,
                cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold,
                cfg.log_scaling_alpha, cfg.log_scaling_n_floor,
                tree_depth=meta.q_pos_tok,
            )
            if L.is_sliding:
                if single_pf:
                    kv_28k, new_kp, new_vp = kv_sconv4_norm_fused(kp, k_raw, L.k_conv_w, vp, v_raw, L.v_conv_w, K, cfg.qk_l2_clip_threshold)
                    kv_f16 = kv_28k.transpose(2, 0, 1, 3)
                else:
                    kv_f16, new_kp, new_vp = kv_sconv4_norm_batched(kp, k_raw, L.k_conv_w, vp, v_raw, L.v_conv_w, B, K, cfg.qk_l2_clip_threshold)
            else:
                if single_pf:
                    kv_28k, new_kp, new_vp = kv_dyn_sconv4_norm_fused(
                        kp, k_raw, k_in, L.k_dyn_st, L.k_dyn_pr_f16,
                        vp, v_raw, v_in, L.v_dyn_st, L.v_dyn_pr_f16,
                        K, cfg.qk_l2_clip_threshold,
                    )
                    kv_f16 = kv_28k.transpose(2, 0, 1, 3)
                else:
                    kv_f16, new_kp, new_vp = kv_dyn_sconv4_norm_batched(
                        kp, k_raw, k_in, L.k_dyn_st, L.k_dyn_pr_f16,
                        vp, v_raw, v_in, L.v_dyn_st, L.v_dyn_pr_f16,
                        B, K, cfg.qk_l2_clip_threshold,
                    )
            _scatter_conv(c_slot, 0, new_kp, k_raw.reshape(B, K, 1024) if record_spec else None)
            _scatter_conv(c_slot, 1, new_vp, v_raw.reshape(B, K, 1024) if record_spec else None)

        # 4. Paged KV write + Batched Split-K / FA2 SDPA
        attn_out = pool.write_and_attend(c_slot, q, kv_f16, rel_st, L.rel_proj_t_f16, q_in, L.sink_w_f16, meta)

        # 5. Gated O projection + batched Post-Attn ShortConv4
        o_rot = rht_in_3072_gated_fast(attn_out, o_gate_raw, L.o_stage.signs_in, out_dtype=mx.float16)
        o_out = L.o_stage.matmul_rotated_f16(o_rot, T, W_cached=wo_c)
        if L.o_out_rot is not None:
            o_out = rht_apply_out(o_out, L.o_out_rot)
        pa_prev = _gather_conv(c_slot, 2)
        if L.is_block_boundary:
            if single_pf:
                h_post, new_pa = short_conv4_res_fused(pa_prev, o_out, L.post_attn_conv_w, K, 2304, residual=None)
            else:
                h_post, new_pa = short_conv4_res_batched(pa_prev, o_out, L.post_attn_conv_w, B, K, 2304, residual=None)
            residual = h_post
            local_blocks = block_residual[None, :, :] if n_b == 0 else mx.concatenate([b_cat, block_residual[None, :, :]], axis=0)
            n_local = n_b + 1
        else:
            if single_pf:
                h_post, new_pa = short_conv4_res_fused(pa_prev, o_out, L.post_attn_conv_w, K, 2304, residual=block_residual)
            else:
                h_post, new_pa = short_conv4_res_batched(pa_prev, o_out, L.post_attn_conv_w, B, K, 2304, residual=block_residual)
            residual = h_post
            local_blocks = b_cat
            n_local = n_b
        _scatter_conv(c_slot, 2, new_pa, o_out.reshape(B, K, 2304) if record_spec else None)

        # 6. MLP AttnRes/MsgBlend + UnifiedMoESide + batched Post-MLP ShortConv4
        if msgs is None:
            h_mlp_f16, x_rot_gu = block_attn_res_norm_rht_fast(L.mlp_in_q, local_blocks, n_local, h_post, L.post_attn_ln, L.moe.gu_signs_in)
        else:
            h_m = block_attn_res_fused(L.mlp_in_q, local_blocks, n_local, h_post)[0]
            fg_mg = _msg_1x_compiled(h_m, msgs[3], L.mlp_dh, L.mlp_dm, L.mlp_fm)
            h_mlp_f16, x_rot_gu = msg_blend_norm_rht_fast(h_m, msgs[3], fg_mg, L.post_attn_ln, L.moe.gu_signs_in)

        moe = L.moe
        logits_f16 = h_mlp_f16 @ moe.router_w_f16.T
        dyn = (cfg.moe_k_max < 8) or (cfg.moe_ratio_thr > 0.0)
        if 2 <= T <= 32:
            if dyn:
                eids_flat, w_norm, gu_plan, dn_plan = sigmoid_top8_norm_and_plan_fast(
                    logits_f16, moe.router_bias, cfg.route_scale,
                    moe.gu_unified.exp_fid, moe.gu_unified.fids,
                    moe.dn_unified.exp_fid, moe.dn_unified.fids,
                    k_max=cfg.moe_k_max, k_min=cfg.moe_k_min, ratio_thr=cfg.moe_ratio_thr, renorm=cfg.moe_renorm,
                )
            else:
                eids_flat, w_norm, gu_plan, dn_plan = sigmoid_top8_norm_and_plan_fast(
                    logits_f16, moe.router_bias, cfg.route_scale,
                    moe.gu_unified.exp_fid, moe.gu_unified.fids,
                    moe.dn_unified.exp_fid, moe.dn_unified.fids,
                )
            sc = None
        else:
            if dyn:
                eids_flat, w_norm = sigmoid_top8_norm_fast(
                    logits_f16, moe.router_bias, cfg.route_scale,
                    k_max=cfg.moe_k_max, k_min=cfg.moe_k_min, ratio_thr=cfg.moe_ratio_thr, renorm=cfg.moe_renorm,
                )
            else:
                eids_flat, w_norm = sigmoid_top8_norm_fast(logits_f16, moe.router_bias, cfg.route_scale)
            gu_plan = dn_plan = None
            sc = compute_moe_sorted_offsets(eids_flat) if T > 32 else None

        need_dn_raw = moe.dn_unified.has_nvfp4
        y_gu = moe.gu_unified(x_rot_gu, eids_flat, x_div=8, sorted_cache=sc, plan=gu_plan, dynamic_routing=dyn, x_raw_f16=h_mlp_f16)
        if need_dn_raw:
            act_rot, act_raw = swiglu_rht_1152_fast(y_gu, moe.dn_signs_in, cfg.mlp_gate_softcap, cfg.mlp_up_softcap, out_dtype=mx.float16, eids_flat=eids_flat if dyn else None, also_raw=True)
        else:
            act_rot = swiglu_rht_1152_fast(y_gu, moe.dn_signs_in, cfg.mlp_gate_softcap, cfg.mlp_up_softcap, out_dtype=mx.float16, eids_flat=eids_flat if dyn else None)
            act_raw = None
        y_dn = moe.dn_unified(act_rot, eids_flat, x_div=1, sorted_cache=sc, plan=dn_plan, dynamic_routing=dyn, x_raw_f16=act_raw)

        pm_prev = _gather_conv(c_slot, 3)
        if single_pf:
            out, new_pm, raw_sum_1 = moe_wsum_sconv4_res_fused(y_dn, w_norm, pm_prev, L.post_mlp_conv_w, residual, K)
            raw_sum = raw_sum_1.reshape(1, K, 2304) if record_spec else None
        else:
            out, new_pm, raw_sum = moe_wsum_sconv4_res_batched(y_dn, w_norm, pm_prev, L.post_mlp_conv_w, residual, B, K)
        _scatter_conv(c_slot, 3, new_pm, raw_sum if record_spec else None)
        return out

    # 1. Prelude L0..5
    for i in range(cfg.loop_start):
        if visit in tap_set:
            captured[visit] = h
        L = model.layers[i]
        new_h = _run_layer(L, h, blocks_cat, n_blocks, i)
        if L.is_block_boundary:
            blocks_cat = h[None, :, :] if n_blocks == 0 else mx.concatenate([blocks_cat, h[None, :, :]], axis=0)
            n_blocks += 1
        h = new_h
        visit += 1
        if (visit & 3) == 0:
            mx.async_eval(h)

    pre_loop_h = h
    pre_loop_blocks = blocks_cat
    num_pre = n_blocks
    cache_loop_w = (T >= 16) and (cfg.loop_passes > 1)
    loop_w_cache: dict[int, tuple[mx.array, mx.array]] = {}
    num_loop = cfg.loop_end - cfg.loop_start

    # 2. Loop Pass 0 (L6..17)
    for i in range(cfg.loop_start, cfg.loop_end):
        if visit in tap_set:
            captured[visit] = h
        L = model.layers[i]
        if cache_loop_w:
            wq_c, wo_c = L.qkv_stage.dequant_f16(), L.o_stage.dequant_f16()
            loop_w_cache[i] = (wq_c, wo_c)
        else:
            wq_c = wo_c = None
        new_h = _run_layer(L, h, blocks_cat, n_blocks, i, wq_c=wq_c, wo_c=wo_c)
        if L.is_block_boundary:
            blocks_cat = h[None, :, :] if n_blocks == 0 else mx.concatenate([blocks_cat, h[None, :, :]], axis=0)
            n_blocks += 1
        h = new_h
        visit += 1
        if (visit & 3) == 0:
            mx.async_eval(h)

    # 3. Loop Passes 1..loop_passes-1
    for loop_idx in range(1, cfg.loop_passes):
        prev_n = n_blocks - (num_pre + 1)
        prev_blocks = blocks_cat[num_pre + 1 :] if prev_n > 0 else None
        prev_h = h
        h = pre_loop_h
        blocks_cat = pre_loop_blocks
        n_blocks = num_pre
        cache_offset = cfg.num_hidden_layers + (loop_idx - 1) * num_loop - cfg.loop_start
        for loop_layer_idx, i in enumerate(range(cfg.loop_start, cfg.loop_end)):
            if visit in tap_set:
                captured[visit] = h
            L = model.layers[i]
            msgs = block_attn_res_fused(model.message_in_q[loop_layer_idx], prev_blocks, prev_n, prev_h)
            wq_c, wo_c = loop_w_cache[i] if cache_loop_w else (None, None)
            new_h = _run_layer(L, h, blocks_cat, n_blocks, i + cache_offset, msgs=msgs, wq_c=wq_c, wo_c=wo_c)
            if L.is_block_boundary:
                blocks_cat = h[None, :, :] if n_blocks == 0 else mx.concatenate([blocks_cat, h[None, :, :]], axis=0)
                n_blocks += 1
            h = new_h
            visit += 1
            if (visit & 3) == 0:
                mx.async_eval(h)
    if cache_loop_w:
        loop_w_cache.clear()

    # 4. Coda L18..23
    for i in range(cfg.loop_end, cfg.num_hidden_layers):
        if visit in tap_set:
            captured[visit] = h
        L = model.layers[i]
        new_h = _run_layer(L, h, blocks_cat, n_blocks, i)
        if L.is_block_boundary:
            blocks_cat = h[None, :, :] if n_blocks == 0 else mx.concatenate([blocks_cat, h[None, :, :]], axis=0)
            n_blocks += 1
        h = new_h
        visit += 1
        if (visit & 3) == 0 and visit < 48:
            mx.async_eval(h)

    # 5. Final AttnRes + norm + lm_head
    if skip_logits:
        if visit in tap_set:
            h_fin = block_attn_res_fused(model.final_in_q, blocks_cat, n_blocks, h, norm_w=model.final_norm_w)[0]
            captured[visit] = h_fin
        taps = mx.stack([captured[v] for v in tap_visits], axis=1) if tap_visits else None
        return None, taps

    if last_only and K > 1 and visit not in tap_set:
        idx_last = mx.array(np.arange(K - 1, T, K, dtype=np.int32))
        b_last = blocks_cat[:, idx_last, :] if (blocks_cat is not None and n_blocks > 0) else blocks_cat
        h_fin = block_attn_res_fused(model.final_in_q, b_last, n_blocks, h[idx_last], norm_w=model.final_norm_w)[0]
        ord_logits = model.lm_head.ordered_logits(h_fin[None, :, :]).reshape(B, 1, -1)
    else:
        h_fin = block_attn_res_fused(model.final_in_q, blocks_cat, n_blocks, h, norm_w=model.final_norm_w)[0]
        if visit in tap_set:
            captured[visit] = h_fin
        if last_only and K > 1:
            idx_last = mx.array(np.arange(K - 1, T, K, dtype=np.int32))
            ord_logits = model.lm_head.ordered_logits(h_fin[idx_last][None, :, :]).reshape(B, 1, -1)
        else:
            ord_logits = model.lm_head.ordered_logits(h_fin[None, :, :]).reshape(B, K, -1)
    taps = mx.stack([captured[v] for v in tap_visits], axis=1) if tap_visits else None
    return ord_logits, taps


def _sample_slot_token(model: Inkling2LoopPackedMLX, row_ord_logits: mx.array, seq: BatchSequence) -> int:
    if seq.temp <= 0.0:
        idx = int(mx.argmax(row_ord_logits, axis=-1).item())
        return int(model.lm_head.order_np[idx])
    tok_mx = _sample_ordered_gpu(
        row_ord_logits,
        model.lm_head.order_mx,
        cap=model.lm_head.cap,
        temp=seq.temp,
        top_p=seq.top_p,
        top_k=seq.top_k,
    )
    return int(tok_mx.item())


def _step_dspark_batched(
    model: Inkling2LoopPackedMLX,
    drafter,
    pool: PagedInklingPool,
    seqs: list[BatchSequence],
    *,
    draft_len: int = 4,
    emit_cb: Callable[[BatchSequence, list[int]], None] | None = None,
) -> int:
    """Run one batched linear-K DSpark verify round across `B = len(seqs)` greedy sequences (`K = 1 + draft_len`).

    Returns total accepted tokens across the batch.
    """
    B = len(seqs)
    K = 1 + draft_len
    tap_visits = drafter.target_layer_ids
    flat_tokens: list[int] = []
    proposals: list[list[int]] = []

    head_argmax_fn = model.lm_head.get_draft_head_argmax()
    for s in seqs:
        if s.drafter_state is not None:
            drafter.restore_context(s.drafter_state)
        else:
            drafter.reset()
        prop_mx = drafter.propose_gpu(s.last_token, model.embed, head_argmax_fn, draft_len=K)
        prop = [int(x) for x in np.asarray(prop_mx, dtype=np.int32).tolist()]
        proposals.append(prop)
        flat_tokens.extend(prop)

    ord_logits, taps = forward_batched(
        model, flat_tokens, pool, seqs, K=K, tap_visits=tap_visits, record_spec=True,
    )
    # Vectorized greedy argmax across all B * K positions in one GPU op!
    arg_pos = mx.argmax(ord_logits, axis=-1)  # [B, K]
    mx.eval(arg_pos, taps)
    arg_np = np.asarray(arg_pos)
    order_np = model.lm_head.order_np

    total_acc = 0
    for b, s in enumerate(seqs):
        prop = proposals[b]
        t_preds = [int(order_np[int(arg_np[b, j])]) for j in range(K)]
        accepted: list[int] = []
        hit_stop = False
        for j in range(draft_len):
            tp = t_preds[j]
            accepted.append(tp)
            if tp in s.eos_ids or len(s.generated) + len(accepted) >= s.max_tokens:
                hit_stop = True
                break
            if s.th_state is not None and s.th_state.step(tp, s.eos_ids):
                s.pending_inject.extend(THINK_CLOSE_IDS)
                hit_stop = True
                break
            if tp != prop[j + 1]:
                break
        else:
            # All `draft_len` draft tokens matched -> bonus token `t_preds[draft_len]`!
            bonus = t_preds[draft_len]
            accepted.append(bonus)
            if bonus in s.eos_ids or len(s.generated) + len(accepted) >= s.max_tokens:
                hit_stop = True
            elif s.th_state is not None and s.th_state.step(bonus, s.eos_ids):
                s.pending_inject.extend(THINK_CLOSE_IDS)
                hit_stop = True

        # k_commit = number of tokens in `prop` whose KV/taps we commit (`1..K`):
        # When `len(accepted) == K` (all drafts matched + bonus), all K input tokens in `prop` are committed!
        # Otherwise `accepted[:-1]` matched `prop[1 : len(accepted)]`, so we commit `prop[:len(accepted)]`!
        k_commit = len(accepted)
        s.rollback_spec(k_commit, K)
        committed_in = prop[:k_commit]
        pos_before = s.seq_len
        s.commit_tokens(committed_in, register_prefix=False)

        # Update drafter context with committed target taps
        if s.drafter_state is not None:
            drafter.restore_context(s.drafter_state)
        drafter.push_context(taps[b * K : b * K + k_commit], start_pos=pos_before)
        s.drafter_state = drafter.context_state()

        s.generated.extend(accepted)
        s.last_token = accepted[-1]
        total_acc += len(accepted)
        if emit_cb is not None:
            emit_cb(s, accepted)
        if hit_stop and not s.pending_inject:
            s.done = True
            s.finish_reason = "stop" if (accepted[-1] in s.eos_ids) else "length"

    return total_acc


def generate_batch(
    model: Inkling2LoopPackedMLX,
    prompts: Sequence[Sequence[int]],
    *,
    max_tokens: int = 256,
    eos_ids: Sequence[int] = (200006,),
    temp: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    max_thinking_tokens: int | None = None,
    prefill_step: int = 512,
    drafter=None,
    dspark_max_batch: int = 4,
    dspark_draft_len: int = 4,
    pool: PagedInklingPool | None = None,
    stats_out: dict[str, Any] | None = None,
) -> list[list[int]]:
    """Generate completions for `len(prompts)` prompts concurrently on one `Inkling2LoopPackedMLX`."""
    own_pool = pool is None
    if pool is None:
        pool = PagedInklingPool(model.cfg)

    t0 = time.perf_counter()
    seqs = [
        BatchSequence(
            i, p, pool, max_tokens=max_tokens, eos_ids=eos_ids,
            temp=temp, top_p=top_p, top_k=top_k,
            max_thinking_tokens=max_thinking_tokens,
        )
        for i, p in enumerate(prompts)
    ]
    use_dspark_eligible = (
        drafter is not None
        and temp <= 0.0
        and len(seqs) <= dspark_max_batch
    )
    tap_visits = drafter.target_layer_ids if use_dspark_eligible else ()

    # 1. Prefill each prompt (with automatic O(1) prefix sharing from earlier prompts in the batch!)
    t_pre0 = time.perf_counter()
    reused_total = 0
    for s in seqs:
        # Re-check prefix cache after earlier prompts in this batch populated shared blocks!
        if s.seq_len == 0 and pool.enable_prefix_cache:
            g_blks, s_blks, s_start, conv_snap, p_hash = pool.match_prefix(s.prompt_ids)
            if g_blks:
                s.g_blocks = g_blks
                s.s_blocks = s_blks
                s.sliding_start = s_start
                m_toks = len(g_blks) * pool.block_size
                s.seq_len = m_toks
                s.prefill_pos = m_toks
                s.reused_tokens = m_toks
                s.all_tokens = list(s.prompt_ids[:m_toks])
                s.prefix_hash = p_hash
                s.registered_blocks = len(g_blks)
                if conv_snap is not None:
                    for v in range(len(pool.conv_pools)):
                        for ci in range(4):
                            pool.conv_pools[v][ci][s.conv_slot] = conv_snap[v][ci]
        reused_total += s.reused_tokens
        if use_dspark_eligible and s.reused_tokens > 0:
            # DSpark drafter needs taps from token 0; disable drafter on this seq if prefix was skipped
            use_dspark_eligible = False
            tap_visits = ()

        if use_dspark_eligible:
            drafter.reset()

        n = len(s.prompt_ids)
        while s.prefill_pos < n:
            end = min(s.prefill_pos + prefill_step, n)
            is_last = end >= n
            chunk = s.prompt_ids[s.prefill_pos : end]
            ord_l, taps = forward_batched(
                model, chunk, pool, [s], K=len(chunk),
                skip_logits=not is_last, last_only=True, tap_visits=tap_visits,
            )
            pos_before = s.seq_len
            s.commit_tokens(chunk, register_prefix=True)
            s.prefill_pos = end
            if use_dspark_eligible and taps is not None:
                drafter.push_context(taps, start_pos=pos_before)
            if is_last:
                tok = _sample_slot_token(model, ord_l[0, 0], s)
                s.generated.append(tok)
                s.last_token = tok
                if use_dspark_eligible:
                    s.drafter_state = drafter.context_state()
                if tok in s.eos_ids or len(s.generated) >= s.max_tokens:
                    s.done = True
                elif s.th_state is not None and s.th_state.step(tok, s.eos_ids):
                    s.pending_inject.extend(THINK_CLOSE_IDS)
            else:
                pool.eval_state()
    t_pre1 = time.perf_counter()

    # 2. Continuous batched decode loop (with load-adaptive DSpark when 1 <= active <= dspark_max_batch!)
    decode_steps = 0
    decode_tokens = 0
    t_dec0 = time.perf_counter()
    while True:
        active = [s for s in seqs if not s.done]
        if not active:
            break

        # Handle any pending think-cap bridge tokens first (rare)
        injected = [s for s in active if s.pending_inject]
        for s in injected:
            inj = [s.last_token, *s.pending_inject]
            s.pending_inject.clear()
            ord_l, taps = forward_batched(
                model, inj, pool, [s], K=len(inj), last_only=True,
                tap_visits=drafter.target_layer_ids if s.drafter_state is not None else (),
            )
            pos_before = s.seq_len
            s.commit_tokens(inj, register_prefix=False)
            if s.drafter_state is not None and taps is not None:
                drafter.restore_context(s.drafter_state)
                drafter.push_context(taps, start_pos=pos_before)
                s.drafter_state = drafter.context_state()
            # Emit bridge tokens + next sampled token
            for bt in inj[1:]:
                s.generated.append(bt)
            nxt = _sample_slot_token(model, ord_l[0, 0], s)
            s.generated.append(nxt)
            s.last_token = nxt
            decode_tokens += len(inj)
            if nxt in s.eos_ids or len(s.generated) >= s.max_tokens:
                s.done = True
        active = [s for s in seqs if not s.done]
        if not active:
            break

        # Load-adaptive DSpark when all active slots have drafter state and B <= dspark_max_batch
        can_dspark = (
            drafter is not None
            and len(active) <= dspark_max_batch
            and all(s.drafter_state is not None and s.temp <= 0.0 for s in active)
        )
        if can_dspark:
            acc = _step_dspark_batched(model, drafter, pool, active, draft_len=dspark_draft_len)
            decode_tokens += acc
            decode_steps += 1
        else:
            step_toks = [s.last_token for s in active]
            ord_l, _ = forward_batched(model, step_toks, pool, active, K=1)
            # Vectorized greedy when all active sequences are greedy
            if all(s.temp <= 0.0 for s in active):
                idxs = np.asarray(mx.argmax(ord_l[:, 0, :], axis=-1))
                next_toks = [int(model.lm_head.order_np[int(i)]) for i in idxs]
            else:
                next_toks = [_sample_slot_token(model, ord_l[b, 0], s) for b, s in enumerate(active)]
            for s, nt in zip(active, next_toks):
                s.commit_tokens([s.last_token], register_prefix=False)
                s.generated.append(nt)
                s.last_token = nt
                s.drafter_state = None  # Switched out of DSpark mode
                decode_tokens += 1
                if nt in s.eos_ids or len(s.generated) >= s.max_tokens:
                    s.done = True
                elif s.th_state is not None and s.th_state.step(nt, s.eos_ids):
                    s.pending_inject.extend(THINK_CLOSE_IDS)
            decode_steps += 1

    t_dec1 = time.perf_counter()
    outputs = [list(s.generated) for s in seqs]
    for s in seqs:
        s.release()
    if own_pool:
        mx.clear_cache()

    if stats_out is not None:
        dec_s = max(t_dec1 - t_dec0, 1e-6)
        pre_s = max(t_pre1 - t_pre0, 1e-6)
        tot_prompt = sum(len(p) for p in prompts)
        stats_out.update({
            "batch_size": len(prompts),
            "prompt_tokens": tot_prompt,
            "reused_prefix_tokens": reused_total,
            "generated_tokens": sum(len(o) for o in outputs),
            "decode_tokens": decode_tokens,
            "decode_steps": decode_steps,
            "prefill_s": round(pre_s, 4),
            "decode_s": round(dec_s, 4),
            "decode_tok_s": round(decode_tokens / dec_s, 2),
            "total_s": round(t_dec1 - t0, 4),
            "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
        })
    return outputs


@dataclass
class _ScheduledRequest:
    req_id: int
    job: dict[str, Any]
    token_queue: "queue.Queue[tuple[str, Any]]"
    gone: Callable[[], bool]
    seq: BatchSequence | None = None
    t_submit: float = field(default_factory=time.perf_counter)
    t_prefill_done: float = 0.0
    reused_tokens: int = 0


class ContinuousBatchScheduler:
    """Background continuous-batching GPU scheduler for `lamq serve` on Inkling-2-Mini MLX.

    Features:
      - Accepts up to `max_batch` concurrent decoding requests + queued arrivals without HTTP 503.
      - Zero-copy shared-prefix KV reuse across concurrent agents via `PagedInklingPool`.
      - Chunked-prefill interleaving (`prefill_step` tokens per turn) so long prefills never stall active streams.
      - Load-adaptive 4-bit DSpark (`1 <= B <= dspark_max_batch` greedy; pure batching at higher `B`).
      - Immediate slot release on client disconnect (`gone()`).
    """

    def __init__(
        self,
        model: Inkling2LoopPackedMLX,
        *,
        drafter=None,
        eos_id: int = 200006,
        max_batch: int = 32,
        prefill_step: int = 512,
        dspark_max_batch: int = 4,
        dspark_draft_len: int = 4,
        enable_prefix_cache: bool = True,
        auto_start: bool = True,
    ):
        self.model = model
        self.drafter = drafter
        self.eos_id = int(eos_id)
        self.max_batch = max(1, int(max_batch))
        self.prefill_step = max(64, int(prefill_step))
        self.dspark_max_batch = max(0, int(dspark_max_batch)) if drafter is not None else 0
        self.dspark_draft_len = max(1, int(dspark_draft_len))
        # Size max block caps to ~55% of remaining Metal working set (allocated lazily, zero idle overhead)
        dev_lim = int(mx.device_info().get("max_recommended_working_set_size", 0) or 0)
        headroom = max(4 << 30, int((dev_lim - mx.get_active_memory()) * 0.55)) if dev_lim > 0 else (16 << 30)
        n_g_pools = 4 if (getattr(model.cfg, "shared_loop_kv", False) and model.cfg.loop_passes > 1) else 8
        b_per_tok = (
            4096 if model.cfg.global_kv_mode == "fp16"
            else (2048 if model.cfg.global_kv_mode == "fp8"
                  else (1056 if model.cfg.global_kv_mode == "nested" else 2176))
        )
        max_g_blks = max(4096, min(65536, headroom // (PAGED_BLOCK_SIZE * n_g_pools * b_per_tok)))
        max_s_blks = max(2048, self.max_batch * 24)
        self.pool = PagedInklingPool(
            model.cfg,
            max_global_blocks=max_g_blks,
            max_sliding_blocks=max_s_blks,
            max_conv_slots=max(64, self.max_batch * 2),
            enable_prefix_cache=enable_prefix_cache,
        )

        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop = threading.Event()
        self._next_id = 1
        self._pending: list[_ScheduledRequest] = []
        self._prefilling: list[_ScheduledRequest] = []
        self._decoding: list[_ScheduledRequest] = []
        self._thread: threading.Thread | None = None
        if auto_start:
            self._thread = threading.Thread(target=self.run_forever, name="lamq-batch-scheduler", daemon=True)
            self._thread.start()

    def run_forever(self) -> None:
        """Run the scheduler loop on the calling thread (use with `auto_start=False` for MLX thread affinity)."""
        self._loop()

    def stop(self) -> None:
        """Signal the scheduler loop to exit cleanly."""
        self._stop.set()
        self._wakeup.set()

    def close(self) -> None:
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def submit(self, job: dict[str, Any], gone: Callable[[], bool]) -> "queue.Queue[tuple[str, Any]]":
        """Enqueue a prepared request and return a queue yielding `('tok', int)`, `('done', meta)`, or `('err', exc)`."""
        q: queue.Queue[tuple[str, Any]] = queue.Queue()
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            self._pending.append(_ScheduledRequest(req_id=rid, job=job, token_queue=q, gone=gone))
        self._wakeup.set()
        return q

    def _finish_req(self, r: _ScheduledRequest, finish_reason: str | None = None) -> None:
        if r.seq is not None:
            meta = {
                "prompt_cached_tokens": r.reused_tokens,
                "generated_tokens": len(r.seq.generated),
                "finish_reason": finish_reason or r.seq.finish_reason or "length",
                "prefill_s": max(0.0, (r.t_prefill_done or time.perf_counter()) - r.t_submit),
            }
            r.seq.release()
            r.seq = None
        else:
            meta = {"prompt_cached_tokens": 0, "generated_tokens": 0, "finish_reason": finish_reason or "stop", "prefill_s": 0.0}
        r.token_queue.put(("done", meta))

    def _admit_pending(self) -> None:
        with self._lock:
            while self._pending and (len(self._prefilling) + len(self._decoding)) < self.max_batch:
                r = self._pending.pop(0)
                if r.gone():
                    r.token_queue.put(("done", {"aborted": True}))
                    continue
                self._prefilling.append(r)

    def _prune_disconnected(self) -> None:
        for lst in (self._prefilling, self._decoding):
            keep = []
            for r in lst:
                if r.gone():
                    if r.seq is not None:
                        r.seq.release()
                        r.seq = None
                    r.token_queue.put(("done", {"aborted": True}))
                else:
                    keep.append(r)
            lst[:] = keep

    def _run_prefill_quantum(self, r: _ScheduledRequest) -> int:
        job = r.job
        if r.seq is None:
            r.seq = BatchSequence(
                r.req_id,
                job["ids"],
                self.pool,
                max_tokens=job["max_tokens"],
                eos_ids={self.eos_id},
                temp=job["temp"],
                top_p=job["top_p"],
                top_k=job["top_k"],
                max_thinking_tokens=job.get("max_thinking_tokens"),
                use_prefix_cache=bool(job.get("use_cache", True)),
            )
            r.reused_tokens = r.seq.reused_tokens
            # Enable drafter taps only when starting from pos 0 with low load & greedy
            if (
                self.dspark_max_batch > 0
                and r.seq.reused_tokens == 0
                and r.seq.temp <= 0.0
                and (len(self._decoding) + len(self._prefilling)) <= self.dspark_max_batch
            ):
                self.drafter.reset()
                r.seq.drafter_state = self.drafter.context_state()

        s = r.seq
        n = len(s.prompt_ids)
        end = min(s.prefill_pos + self.prefill_step, n)
        is_last = end >= n
        chunk = s.prompt_ids[s.prefill_pos : end]
        tap_visits = self.drafter.target_layer_ids if s.drafter_state is not None else ()
        ord_l, taps = forward_batched(
            self.model, chunk, self.pool, [s], K=len(chunk),
            skip_logits=not is_last, last_only=True, tap_visits=tap_visits,
        )
        pos_before = s.seq_len
        s.commit_tokens(chunk, register_prefix=True)
        s.prefill_pos = end
        if s.drafter_state is not None and taps is not None:
            self.drafter.restore_context(s.drafter_state)
            self.drafter.push_context(taps, start_pos=pos_before)
            s.drafter_state = self.drafter.context_state()

        if is_last:
            r.t_prefill_done = time.perf_counter()
            r.token_queue.put((
                "meta",
                {
                    "prompt_cached_tokens": r.reused_tokens,
                    "prefill_s": max(0.0, r.t_prefill_done - r.t_submit),
                },
            ))
            tok = _sample_slot_token(self.model, ord_l[0, 0], s)
            s.generated.append(tok)
            s.last_token = tok
            r.token_queue.put(("tok", tok))
            self._prefilling.remove(r)
            if tok in s.eos_ids or len(s.generated) >= s.max_tokens:
                s.done = True
                s.finish_reason = "stop" if tok in s.eos_ids else "length"
                self._finish_req(r)
            else:
                if s.th_state is not None and s.th_state.step(tok, s.eos_ids):
                    s.pending_inject.extend(THINK_CLOSE_IDS)
                self._decoding.append(r)
        else:
            self.pool.eval_state()
        return len(chunk)

    def _run_decode_step(self) -> None:
        if not self._decoding:
            return
        # Handle any pending think-cap injections individually
        for r in list(self._decoding):
            s = r.seq
            if s is not None and s.pending_inject:
                inj = [s.last_token, *s.pending_inject]
                s.pending_inject.clear()
                tap_visits = self.drafter.target_layer_ids if s.drafter_state is not None else ()
                ord_l, taps = forward_batched(self.model, inj, self.pool, [s], K=len(inj), last_only=True, tap_visits=tap_visits)
                pos_before = s.seq_len
                s.commit_tokens(inj, register_prefix=False)
                if s.drafter_state is not None and taps is not None:
                    self.drafter.restore_context(s.drafter_state)
                    self.drafter.push_context(taps, start_pos=pos_before)
                    s.drafter_state = self.drafter.context_state()
                for bt in inj[1:]:
                    s.generated.append(bt)
                    r.token_queue.put(("tok", bt))
                nxt = _sample_slot_token(self.model, ord_l[0, 0], s)
                s.generated.append(nxt)
                s.last_token = nxt
                r.token_queue.put(("tok", nxt))
                if nxt in s.eos_ids or len(s.generated) >= s.max_tokens:
                    s.done = True
                    s.finish_reason = "stop" if nxt in s.eos_ids else "length"
                    self._decoding.remove(r)
                    self._finish_req(r)

        if not self._decoding:
            return

        reqs = list(self._decoding)
        seqs = [r.seq for r in reqs if r.seq is not None]
        if not seqs:
            return
        can_dspark = (
            self.dspark_max_batch > 0
            and len(seqs) <= self.dspark_max_batch
            and all(s.drafter_state is not None and s.temp <= 0.0 for s in seqs)
        )
        if can_dspark:
            req_by_id = {r.req_id: r for r in reqs}

            def _emit(seq: BatchSequence, toks: list[int]) -> None:
                rq = req_by_id[seq.seq_id]
                for t in toks:
                    rq.token_queue.put(("tok", t))

            _step_dspark_batched(self.model, self.drafter, self.pool, seqs, draft_len=self.dspark_draft_len, emit_cb=_emit)
            for r in reqs:
                if r.seq is not None and r.seq.done:
                    self._decoding.remove(r)
                    self._finish_req(r)
        else:
            step_toks = [s.last_token for s in seqs]
            ord_l, _ = forward_batched(self.model, step_toks, self.pool, seqs, K=1)
            if all(s.temp <= 0.0 for s in seqs):
                idxs = np.asarray(mx.argmax(ord_l[:, 0, :], axis=-1))
                next_toks = [int(self.model.lm_head.order_np[int(i)]) for i in idxs]
            else:
                next_toks = [_sample_slot_token(self.model, ord_l[b, 0], s) for b, s in enumerate(seqs)]
            for r, s, nt in zip(reqs, seqs, next_toks):
                s.commit_tokens([s.last_token], register_prefix=False)
                s.generated.append(nt)
                s.last_token = nt
                s.drafter_state = None
                r.token_queue.put(("tok", nt))
                if nt in s.eos_ids or len(s.generated) >= s.max_tokens:
                    s.done = True
                    s.finish_reason = "stop" if nt in s.eos_ids else "length"
                    self._decoding.remove(r)
                    self._finish_req(r)
                elif s.th_state is not None and s.th_state.step(nt, s.eos_ids):
                    s.pending_inject.extend(THINK_CLOSE_IDS)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._admit_pending()
            self._prune_disconnected()
            if not self._prefilling and not self._decoding:
                self._wakeup.wait(timeout=0.05)
                self._wakeup.clear()
                continue
            try:
                # Interleave up to `prefill_step` tokens of prefill work per cycle so bursts of short
                # prompts admit together while long prefills stay bounded and interleave decode steps.
                prefilled_toks = 0
                while self._prefilling and prefilled_toks < self.prefill_step:
                    self._admit_pending()
                    self._prune_disconnected()
                    if not self._prefilling:
                        break
                    r = self._prefilling[0]
                    try:
                        prefilled_toks += self._run_prefill_quantum(r)
                    except Exception as exc:
                        if r in self._prefilling:
                            self._prefilling.remove(r)
                        if r.seq is not None:
                            r.seq.release()
                            r.seq = None
                        r.token_queue.put(("err", exc))
                if self._decoding:
                    # Run 4 decode steps between long prefill cycles so streaming agents stay responsive
                    n_dec = 4 if self._prefilling else 1
                    for _ in range(n_dec):
                        self._prune_disconnected()
                        if not self._decoding:
                            break
                        self._run_decode_step()
            except Exception as exc:
                # Fail all active requests cleanly if an unexpected GPU error occurs
                for lst in (self._prefilling, self._decoding):
                    for r in lst:
                        if r.seq is not None:
                            r.seq.release()
                            r.seq = None
                        r.token_queue.put(("err", exc))
                    lst.clear()
                mx.clear_cache()
