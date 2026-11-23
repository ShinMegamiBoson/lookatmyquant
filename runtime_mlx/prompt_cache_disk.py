"""Disk-backed prompt-prefix KV cache with incremental delta saves for `lamq serve`.

Extends `runtime_mlx.inkling.generate.PromptCache` so long chats and coding-agent sessions resume
after a server restart (or when alternating between multiple conversations) without re-prefilling
the shared prefix.

Layout inside `<cache_dir>`:
    <entry_id>.meta.json     tokens, model/KV fingerprint, slot shapes, optional parent_id
    <entry_id>.safetensors   base snapshot, or delta slice (new tail of global KV + tiny SWA/conv)

Delta saves:
    When a new prompt extends an existing saved prefix (`parent_n < n` and chain depth < 4), only
    the newly appended slices of the 8 growing global caches are written, alongside the fixed-size
    sliding-window rings (<= 1,024 tokens) and causal-conv / drafter state. Restoring walks the
    short chain and assembles the exact `Inkling2Cache` bit-for-bit.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

try:
    from .inkling.generate import PromptCache
except ImportError:  # Linux CUDA host without Apple MLX installed
    class PromptCache:  # type: ignore[no-redef]
        def __init__(self, max_entries: int = 2, min_tokens: int = 256):
            self.max_entries = max_entries
            self.min_tokens = min_tokens
            self._entries: list = []

CACHE_FORMAT_VERSION = 1
MAX_DELTA_CHAIN = 4


def _model_fingerprint(model: Any, drafter: Any = None) -> str:
    """Stable key scoping saved KV snapshots to a specific model directory, KV mode, and drafter."""
    h = hashlib.sha256()
    p = getattr(model, "packed_dir", None)
    if p is not None:
        p = Path(p)
        h.update(str(p.resolve()).encode())
        manifest = p / "manifest.json"
        if manifest.exists():
            h.update(manifest.read_bytes())
    else:
        cfg = getattr(model, "cfg", None)
        h.update(repr(cfg).encode())
    kv_mode = getattr(getattr(model, "cfg", None), "global_kv_mode", "fp16")
    h.update(f"|kv={kv_mode}|draft={drafter is not None}".encode())
    return h.hexdigest()[:24]


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":")))
    os.replace(tmp, path)


def _atomic_save_mlx(path: Path, tensors: dict) -> int:
    import mlx.core as mx

    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp.safetensors")
    mx.save_safetensors(str(tmp), tensors)
    size = tmp.stat().st_size
    os.replace(tmp, path)
    return size


def _serialize_mlx_snapshot(
    cache: Any,
    drafter_state: dict | None,
    *,
    parent_meta: dict | None = None,
) -> tuple[dict, dict]:
    """Pack an `Inkling2Cache` (+ optional drafter state) into `(tensors, meta_fields)`.

    If `parent_meta` is given, global KV caches store only the appended `[prev_len:cur_len]` delta.
    """
    import mlx.core as mx

    tensors: dict[str, mx.array] = {}
    fast = bool(cache._fast)
    n_slots = len(cache.kv_len) if fast else len(cache.keys)
    parent_kv_len = parent_meta["kv_len"] if parent_meta is not None else [0] * n_slots
    delta_starts: list[int] = [0] * n_slots

    if fast:
        for s in range(n_slots):
            is_sw = bool(cache.kv_is_sliding[s])
            cur = int(cache.kv_len[s])
            prev = int(parent_kv_len[s]) if (parent_meta is not None and not is_sw) else 0
            if prev > cur or prev < 0:
                prev = 0
            delta_starts[s] = prev
            if cur > prev:
                if cache.kv_f16[s] is not None:
                    tensors[f"kv_f16.{s}"] = mx.contiguous(cache.kv_f16[s][:, :, prev:cur, :])
                if cache.kv_u8[s] is not None:
                    tensors[f"kv_u8.{s}"] = mx.contiguous(cache.kv_u8[s][:, :, prev:cur, :])
                if cache.kv_q4_sc[s] is not None:
                    tensors[f"kv_q4_sc.{s}"] = mx.contiguous(cache.kv_q4_sc[s][:, :, prev:cur, :])
    else:
        for s in range(n_slots):
            if cache.keys[s] is not None and cache.values[s] is not None:
                is_sw = bool(cache.kv_is_sliding[s]) if s < len(cache.kv_is_sliding) else False
                cur = int(cache.keys[s].shape[2])
                prev = int(parent_kv_len[s]) if (parent_meta is not None and not is_sw) else 0
                if prev > cur or prev < 0:
                    prev = 0
                delta_starts[s] = prev
                if cur > prev:
                    tensors[f"keys.{s}"] = mx.contiguous(cache.keys[s][:, :, prev:cur, :])
                    tensors[f"values.{s}"] = mx.contiguous(cache.values[s][:, :, prev:cur, :])

    for s in range(n_slots):
        for c_idx, arr in enumerate(cache.conv_states[s]):
            if arr is not None:
                tensors[f"conv.{s}.{c_idx}"] = mx.contiguous(arr)

    drafter_meta = None
    if drafter_state is not None:
        p_cur = (
            parent_meta["drafter"]["_ctx_cur"]
            if (parent_meta is not None and parent_meta.get("drafter"))
            else [0] * len(drafter_state["_ctx_cur"])
        )
        d_starts = []
        for i, (ck, cv) in enumerate(zip(drafter_state["ctx_k"], drafter_state["ctx_v"])):
            cur = int(drafter_state["_ctx_cur"][i])
            prev = int(p_cur[i]) if i < len(p_cur) else 0
            if prev > cur or prev < 0:
                prev = 0
            d_starts.append(prev)
            if ck is not None and cur > prev:
                tensors[f"d_ctx_k.{i}"] = mx.contiguous(ck[:, :, prev:cur, :])
            if cv is not None and cur > prev:
                tensors[f"d_ctx_v.{i}"] = mx.contiguous(cv[:, :, prev:cur, :])
        if drafter_state.get("f_rnn") is not None:
            tensors["d_f_rnn"] = mx.contiguous(drafter_state["f_rnn"])
        drafter_meta = {
            "n_layers": len(drafter_state["ctx_k"]),
            "_ctx_cur": [int(x) for x in drafter_state["_ctx_cur"]],
            "_ctx_cap": [int(x) for x in drafter_state["_ctx_cap"]],
            "ctx_len": int(drafter_state["ctx_len"]),
            "delta_starts": d_starts,
        }

    meta = {
        "fast": fast,
        "n_slots": n_slots,
        "seq_len": int(cache.seq_len),
        "kv_len": [int(x) for x in cache.kv_len] if fast else [int(k.shape[2]) if k is not None else 0 for k in cache.keys],
        "kv_start": [int(x) for x in cache.kv_start],
        "kv_cap": [int(x) for x in cache.kv_cap],
        "kv_is_sliding": [bool(x) for x in cache.kv_is_sliding],
        "delta_starts": delta_starts,
        "drafter": drafter_meta,
    }
    return tensors, meta


def _restore_mlx_chain(model: Any, chain: list[tuple[dict, Path]]) -> tuple[Any, dict | None]:
    """Reconstruct `(Inkling2Cache, drafter_state)` from a `[base, delta1, ...]` disk chain."""
    import mlx.core as mx

    leaf_meta = chain[-1][0]
    fast = bool(leaf_meta["fast"])
    n_slots = int(leaf_meta["n_slots"])
    cache = model.make_cache()
    cache.seq_len = int(leaf_meta["seq_len"])
    cache.kv_len = list(leaf_meta["kv_len"])
    cache.kv_start = list(leaf_meta["kv_start"])
    cache.kv_cap = list(leaf_meta["kv_cap"])
    cache.kv_is_sliding = list(leaf_meta["kv_is_sliding"])

    loaded = [mx.load(str(tpath)) for _, tpath in chain]
    leaf_tensors = loaded[-1]

    if fast:
        for s in range(n_slots):
            is_sw = bool(cache.kv_is_sliding[s])
            cur = int(cache.kv_len[s])
            if cur == 0:
                continue
            cap = max(int(cache.kv_cap[s]), ((cur + 255) // 256) * 256)
            cache.kv_cap[s] = cap
            for key_prefix, target_list in (("kv_f16", cache.kv_f16), ("kv_u8", cache.kv_u8), ("kv_q4_sc", cache.kv_q4_sc)):
                tk = f"{key_prefix}.{s}"
                if is_sw:
                    arr = leaf_tensors.get(tk)
                    if arr is not None:
                        buf = mx.zeros((*arr.shape[:2], cap, arr.shape[-1]), dtype=arr.dtype)
                        buf[:, :, : arr.shape[2], :] = arr
                        target_list[s] = buf
                else:
                    # Assemble from base + deltas
                    parts = []
                    sample = None
                    for (m_step, _), t_step in zip(chain, loaded):
                        arr = t_step.get(tk)
                        if arr is not None:
                            sample = arr
                            parts.append((int(m_step["delta_starts"][s]), arr))
                    if sample is not None:
                        buf = mx.zeros((*sample.shape[:2], cap, sample.shape[-1]), dtype=sample.dtype)
                        for start, arr in parts:
                            buf[:, :, start : start + arr.shape[2], :] = arr
                        target_list[s] = buf
    else:
        for s in range(n_slots):
            is_sw = bool(cache.kv_is_sliding[s]) if s < len(cache.kv_is_sliding) else False
            if is_sw:
                if f"keys.{s}" in leaf_tensors:
                    cache.keys[s] = leaf_tensors[f"keys.{s}"]
                    cache.values[s] = leaf_tensors[f"values.{s}"]
            else:
                ks = [t_step[f"keys.{s}"] for t_step in loaded if f"keys.{s}" in t_step]
                vs = [t_step[f"values.{s}"] for t_step in loaded if f"values.{s}" in t_step]
                if ks:
                    cache.keys[s] = mx.concatenate(ks, axis=2) if len(ks) > 1 else ks[0]
                    cache.values[s] = mx.concatenate(vs, axis=2) if len(vs) > 1 else vs[0]

    for s in range(n_slots):
        for c_idx in range(4):
            ck = f"conv.{s}.{c_idx}"
            if ck in leaf_tensors:
                cache.conv_states[s][c_idx] = leaf_tensors[ck]

    drafter_state = None
    dm = leaf_meta.get("drafter")
    if dm is not None:
        n_layers = int(dm["n_layers"])
        ctx_k: list[Any] = [None] * n_layers
        ctx_v: list[Any] = [None] * n_layers
        caps = list(dm["_ctx_cap"])
        for i in range(n_layers):
            cur = int(dm["_ctx_cur"][i])
            if cur == 0:
                continue
            cap = max(int(caps[i]), ((cur + 255) // 256) * 256)
            caps[i] = cap
            for key_prefix, target in (("d_ctx_k", ctx_k), ("d_ctx_v", ctx_v)):
                tk = f"{key_prefix}.{i}"
                parts = []
                sample = None
                for (m_step, _), t_step in zip(chain, loaded):
                    arr = t_step.get(tk)
                    d_step = m_step.get("drafter")
                    if arr is not None and d_step is not None:
                        sample = arr
                        parts.append((int(d_step["delta_starts"][i]), arr))
                if sample is not None:
                    buf = mx.zeros((*sample.shape[:2], cap, sample.shape[-1]), dtype=sample.dtype)
                    for start, arr in parts:
                        buf[:, :, start : start + arr.shape[2], :] = arr
                    target[i] = buf
        drafter_state = {
            "ctx_k": ctx_k,
            "ctx_v": ctx_v,
            "_ctx_cur": list(dm["_ctx_cur"]),
            "_ctx_cap": caps,
            "ctx_len": int(dm["ctx_len"]),
            "f_rnn": leaf_tensors.get("d_f_rnn"),
        }

    eval_list = list(cache.state)
    if drafter_state is not None:
        eval_list.extend(a for a in (*drafter_state["ctx_k"], *drafter_state["ctx_v"], drafter_state["f_rnn"]) if a is not None)
    if eval_list:
        mx.eval(*eval_list)
    return cache, drafter_state


class DiskPromptCache(PromptCache):
    """In-memory `PromptCache` backed by atomic delta-saved safetensors on disk."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        max_entries: int = 2,
        min_tokens: int = 256,
        max_disk_gb: float = 20.0,
        min_disk_grow: int = 1,
    ):
        super().__init__(max_entries=max(1, max_entries), min_tokens=min_tokens)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_disk_bytes = int(max(0.0, float(max_disk_gb)) * (1 << 30))
        self.min_disk_grow = int(max(1, min_disk_grow))
        self._disk_index: dict[str, dict] = {}
        self._load_index()

    def _meta_path(self, eid: str) -> Path:
        return self.cache_dir / f"{eid}.meta.json"

    def _tensor_path(self, eid: str) -> Path:
        return self.cache_dir / f"{eid}.safetensors"

    def _load_index(self) -> None:
        self._disk_index.clear()
        for mp in sorted(self.cache_dir.glob("*.meta.json")):
            try:
                meta = json.loads(mp.read_text())
                eid = meta["id"]
                if meta.get("version") != CACHE_FORMAT_VERSION:
                    continue
                if not self._tensor_path(eid).exists():
                    continue
                self._disk_index[eid] = meta
            except Exception:
                continue

    def _resolve_chain(self, eid: str) -> list[tuple[dict, Path]] | None:
        chain: list[tuple[dict, Path]] = []
        seen: set[str] = set()
        cur = eid
        while cur is not None:
            if cur in seen or cur not in self._disk_index:
                return None
            seen.add(cur)
            meta = self._disk_index[cur]
            tp = self._tensor_path(cur)
            if not tp.exists():
                return None
            chain.append((meta, tp))
            cur = meta.get("parent_id")
        chain.reverse()
        return chain

    def _best_disk_prefix(self, ids: list[int], model_key: str, drafting: bool, *, strict: bool = True) -> dict | None:
        best = None
        for meta in self._disk_index.values():
            if meta.get("model_key") != model_key or bool(meta.get("drafting")) != drafting:
                continue
            toks = meta["tokens"]
            n = len(toks)
            if (n < len(ids) if strict else n <= len(ids)) and (best is None or n > len(best["tokens"])):
                if ids[:n] == toks and self._resolve_chain(meta["id"]) is not None:
                    best = meta
        return best

    def _evict_disk(self) -> None:
        if self.max_disk_bytes <= 0:
            return
        total = sum(int(m.get("bytes", 0)) for m in self._disk_index.values())
        if total <= self.max_disk_bytes:
            return
        # Never evict a parent while a child still references it; delete oldest leaves first.
        while total > self.max_disk_bytes and self._disk_index:
            referenced = {m.get("parent_id") for m in self._disk_index.values() if m.get("parent_id")}
            leaves = [m for eid, m in self._disk_index.items() if eid not in referenced]
            if not leaves:
                break
            victim = min(leaves, key=lambda m: float(m.get("used_at", 0.0)))
            eid = victim["id"]
            total -= int(victim.get("bytes", 0))
            self._disk_index.pop(eid, None)
            for p in (self._meta_path(eid), self._tensor_path(eid)):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass

    def _match(self, ids: list[int], drafting: bool) -> dict | None:
        mem_hit = super()._match(ids, drafting)
        if self._owner is None:
            return mem_hit
        model, drafter = self._owner
        mkey = _model_fingerprint(model, drafter)
        disk_meta = self._best_disk_prefix(ids, mkey, drafting, strict=True)
        if disk_meta is None:
            return mem_hit
        if mem_hit is not None and len(mem_hit["tokens"]) >= len(disk_meta["tokens"]):
            return mem_hit
        chain = self._resolve_chain(disk_meta["id"])
        if chain is None:
            return mem_hit
        try:
            cache, d_state = _restore_mlx_chain(model, chain)
        except Exception as e:
            print(f"[DiskPromptCache] failed to restore {disk_meta['id']}: {e}", file=sys.stderr, flush=True)
            return mem_hit
        disk_meta["used_at"] = time.time()
        try:
            _atomic_write_json(self._meta_path(disk_meta["id"]), disk_meta)
        except OSError:
            pass
        self._clock += 1
        entry = {"tokens": list(disk_meta["tokens"]), "cache": cache, "drafter": d_state, "used": self._clock}
        self._entries.append(entry)
        self._entries.sort(key=lambda e: e["used"], reverse=True)
        del self._entries[self.max_entries :]
        return entry

    def _store(self, tokens: list[int], cache, drafter=None) -> None:
        super()._store(tokens, cache, drafter)
        if len(tokens) < self.min_tokens or self._owner is None:
            return
        model, owner_drafter = self._owner
        drafting = drafter is not None
        mkey = _model_fingerprint(model, owner_drafter)
        # Skip saving if an identical or longer matching prefix is already stored on disk
        existing = self._best_disk_prefix(tokens, mkey, drafting, strict=False)
        if existing is not None and len(existing["tokens"]) == len(tokens):
            return
        if existing is not None and (len(tokens) - len(existing["tokens"])) < self.min_disk_grow:
            return
        parent_meta = None
        chain_depth = 0
        if existing is not None:
            chain = self._resolve_chain(existing["id"])
            if chain is not None and len(chain) < MAX_DELTA_CHAIN:
                parent_meta = existing
                chain_depth = len(chain)
        try:
            d_state = drafter.context_state() if drafter is not None else None
            tensors, extra = _serialize_mlx_snapshot(cache, d_state, parent_meta=parent_meta)
            eid = f"slot_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            n_bytes = _atomic_save_mlx(self._tensor_path(eid), tensors)
            meta = {
                "version": CACHE_FORMAT_VERSION,
                "id": eid,
                "model_key": mkey,
                "drafting": drafting,
                "parent_id": parent_meta["id"] if parent_meta is not None else None,
                "chain_depth": chain_depth,
                "tokens": [int(t) for t in tokens],
                "bytes": int(n_bytes),
                "used_at": time.time(),
                **extra,
            }
            _atomic_write_json(self._meta_path(eid), meta)
            self._disk_index[eid] = meta
            self._evict_disk()
        except Exception as e:
            # Never fail a generation request because of a disk cache write error.
            print(f"[DiskPromptCache] save skipped: {e}", file=sys.stderr, flush=True)


class CudaPromptCache:
    """In-memory (+ optional disk-backed) prefix cache for CUDA `Inkling2Cache`."""

    def __init__(
        self,
        max_entries: int = 2,
        min_tokens: int = 256,
        cache_dir: str | Path | None = None,
        max_disk_gb: float = 20.0,
    ):
        self.max_entries = int(max_entries)
        self.min_tokens = int(min_tokens)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_disk_bytes = int(max(0.0, float(max_disk_gb)) * (1 << 30))
        self._entries: list[dict] = []
        self._clock = 0

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def match(self, ids: Sequence[int], model_key: str = "") -> tuple[list[int], dict] | None:
        ids_l = list(ids)
        best = None
        for e in self._entries:
            n = len(e["tokens"])
            if e["model_key"] == model_key and n < len(ids_l) and (best is None or n > len(best["tokens"])):
                if ids_l[:n] == e["tokens"]:
                    best = e
        if best is not None:
            self._clock += 1
            best["used"] = self._clock
            return best["tokens"], best["state"]
        if self.cache_dir is not None:
            disk = self._load_best_disk(ids_l, model_key)
            if disk is not None:
                return disk
        return None

    def store(self, tokens: Sequence[int], state: dict, model_key: str = "") -> None:
        toks = [int(t) for t in tokens]
        if self.max_entries <= 0 or len(toks) < self.min_tokens:
            return
        n = len(toks)
        self._entries = [
            e for e in self._entries
            if not (e["model_key"] == model_key and len(e["tokens"]) <= n and toks[: len(e["tokens"])] == e["tokens"])
        ]
        self._clock += 1
        self._entries.append({"tokens": toks, "state": state, "model_key": model_key, "used": self._clock})
        self._entries.sort(key=lambda e: e["used"], reverse=True)
        del self._entries[self.max_entries :]
        if self.cache_dir is not None:
            self._save_disk(toks, state, model_key)

    def _save_disk(self, toks: list[int], state: dict, model_key: str) -> None:
        try:
            from safetensors.torch import save_file

            eid = f"cuda_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            tpath = self.cache_dir / f"{eid}.safetensors"
            tmp = tpath.with_name(f".{tpath.name}.tmp")
            save_file(state["tensors"], str(tmp))
            os.replace(tmp, tpath)
            meta = {
                "version": CACHE_FORMAT_VERSION,
                "id": eid,
                "model_key": model_key,
                "tokens": toks,
                "meta": state["meta"],
                "bytes": tpath.stat().st_size,
                "used_at": time.time(),
            }
            _atomic_write_json(self.cache_dir / f"{eid}.meta.json", meta)
            self._evict_cuda_disk()
        except Exception as e:
            print(f"[CudaPromptCache] save skipped: {e}", file=sys.stderr, flush=True)

    def _load_best_disk(self, ids: list[int], model_key: str) -> tuple[list[int], dict] | None:
        best_meta = None
        for mp in sorted(self.cache_dir.glob("cuda_*.meta.json")):
            try:
                m = json.loads(mp.read_text())
                if m.get("version") != CACHE_FORMAT_VERSION or m.get("model_key") != model_key:
                    continue
                n = len(m["tokens"])
                if n < len(ids) and (best_meta is None or n > len(best_meta["tokens"])) and ids[:n] == m["tokens"]:
                    if (self.cache_dir / f"{m['id']}.safetensors").exists():
                        best_meta = m
            except Exception:
                continue
        if best_meta is None:
            return None
        try:
            from safetensors.torch import load_file

            tensors = load_file(str(self.cache_dir / f"{best_meta['id']}.safetensors"), device="cpu")
            return list(best_meta["tokens"]), {"tensors": tensors, "meta": best_meta["meta"]}
        except Exception:
            return None

    def _evict_cuda_disk(self) -> None:
        if self.max_disk_bytes <= 0 or self.cache_dir is None:
            return
        metas = []
        for mp in self.cache_dir.glob("cuda_*.meta.json"):
            try:
                metas.append((mp, json.loads(mp.read_text())))
            except Exception:
                continue
        total = sum(int(m.get("bytes", 0)) for _, m in metas)
        metas.sort(key=lambda pair: float(pair[1].get("used_at", 0.0)))
        for mp, m in metas:
            if total <= self.max_disk_bytes:
                break
            total -= int(m.get("bytes", 0))
            mp.unlink(missing_ok=True)
            (self.cache_dir / f"{m['id']}.safetensors").unlink(missing_ok=True)
