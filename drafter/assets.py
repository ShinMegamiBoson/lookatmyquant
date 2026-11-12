"""Stage `assets`: the published DFlash drafter into ROOT/assets/dflash, and the prompt list ROOT/assets/prompts.jsonl
(held-out prompts first, then training prompts; one line each: id, split, text, source)."""

from __future__ import annotations

import json
import os
import random

from . import common


def _first_user_turn(row):
    """The first thing the user says, for ShareGPT rows (conversations: from/value), chat rows (messages: role/content)
    and rows with a plain prompt."""
    for key in ("conversations", "messages"):
        for turn in row.get(key) or ():
            if (turn.get("from") or turn.get("role")) in ("human", "user"):
                return turn.get("value") or turn.get("content")
    return row.get("prompt")


def _source(row):
    for key in ("source", "category"):
        if row.get(key):
            return str(row[key])
    return None


def _prompts(s, status):
    from datasets import load_dataset

    data = None
    for name, split in s.datasets:
        try:
            data = load_dataset(name, split=split, cache_dir=s.datasets_cache)
            break
        except Exception as error:
            status.log(f"cannot load {name}: {error!r}")
    if data is None:
        raise RuntimeError("no prompt dataset could be loaded")
    status.log(f"prompts from {name} ({len(data)} rows)")

    tok = common.tokenizer(s)
    order = list(range(len(data)))
    random.Random(s.prompt_seed).shuffle(order)
    wanted = s.heldout_prompts + s.train_prompts
    seen, rows = set(), []
    for looked, index in enumerate(order, 1):
        row = data[index]
        text = (_first_user_turn(row) or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        if len(common.prompt_ids(tok, text)) > s.prompt_tokens:
            continue
        split = "heldout" if len(rows) < s.heldout_prompts else "train"
        rows.append({"id": f"{split[0]}{len(rows):06d}", "split": split, "text": text, "source": _source(row)})
        if len(rows) % 2000 == 0:
            status.log(f"{len(rows)} prompts kept of {looked} rows read", prompts=len(rows), wanted=wanted)
        if len(rows) == wanted:
            break
    if len(rows) < wanted:
        raise RuntimeError(f"{name} gave {len(rows)} usable prompts, {wanted} needed")
    return rows, name


def run(root, s, status):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=s.dflash_repo, local_dir=str(root / "assets" / "dflash"))
    status.log(f"{s.dflash_repo} is in {root / 'assets' / 'dflash'}")

    path = root / "assets" / "prompts.jsonl"
    if path.exists():
        return {"prompts": len(common.read_prompts(root))}
    rows, name = _prompts(s, status)
    partial = path.with_name(path.name + ".tmp")
    with open(partial, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    os.replace(partial, path)
    status.log(f"wrote {len(rows)} prompts", prompts=len(rows))
    return {"prompts": len(rows), "dataset": name}
