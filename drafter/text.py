"""Stage `text`: each target writes its own greedy response to the prompts. The build answers every prompt
(ROOT/text/train_NNNNN.pt, heldout_NNNNN.pt); the original answers the held-out ones (heldout_original_NNNNN.pt), which
the baseline needs. A shard is a list of {id, input_ids (prompt + response, with the end token when it was produced),
prompt_len}; a shard that exists is never rewritten."""

from __future__ import annotations

import os

import torch

from . import common

PLANS = (("build", "train", "train"), ("build", "heldout", "heldout"), ("original", "heldout", "heldout_original"))


def _generate(model, rows, pad_id: int, end_ids: list, s) -> list:
    """Greedy responses for one batch of prompts, left-padded to the longest."""
    width = max(len(row["ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, width - len(row["ids"]):] = torch.tensor(row["ids"])
        mask[i, width - len(row["ids"]):] = 1
    with torch.no_grad():
        out = model.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                             max_new_tokens=s.max_new_tokens, do_sample=False, num_beams=1, repetition_penalty=1.0,
                             no_repeat_ngram_size=0, eos_token_id=end_ids, pad_token_id=pad_id)
    new = out[:, width:].cpu()
    ended = torch.isin(new, torch.tensor(end_ids))
    records = []
    for i, row in enumerate(rows):
        hits = ended[i].nonzero()
        stop = int(hits[0]) + 1 if len(hits) else new.shape[1]
        records.append({"id": row["id"], "input_ids": torch.cat([torch.tensor(row["ids"]), new[i, :stop]]),
                        "prompt_len": len(row["ids"])})
    return records


def run(root, s, status):
    prompts = common.read_prompts(root)
    tok = common.tokenizer(s)
    out = root / "text"
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for which in ("build", "original"):
        todo = []
        for target, split, prefix in PLANS:
            if target != which:
                continue
            rows = [{"id": p["id"], "ids": common.prompt_ids(tok, p["text"])} for p in prompts if p["split"] == split]
            rows.sort(key=lambda row: (-len(row["ids"]), row["id"]))  # longest first: a batch that fits, fits all later
            for start in range(0, len(rows), s.shard_prompts):
                path = out / f"{prefix}_{start // s.shard_prompts:05d}.pt"
                if not path.exists():
                    todo.append((path, rows[start:start + s.shard_prompts]))
        if not todo:
            continue
        model = common.load_model(s, which, status)
        end_ids = model.generation_config.eos_token_id
        end_ids = sorted({tok.eos_token_id, *(end_ids if isinstance(end_ids, (list, tuple)) else [end_ids])} - {None})
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else end_ids[0]
        total, done, tokens, open_ended = sum(len(rows) for _, rows in todo), 0, 0, 0
        for path, rows in todo:
            records = []
            for start in range(0, len(rows), s.generate_batch):
                records += _generate(model, rows[start:start + s.generate_batch], pad_id, end_ids, s)
                done = done + min(s.generate_batch, len(rows) - start)
                status.log(f"{which}: {done} of {total} prompts answered, shard {path.name}",
                           **{f"{which}_prompts": done, f"{which}_total": total})
            partial = path.with_name(path.name + ".tmp")
            torch.save(records, partial)
            os.replace(partial, path)
            tokens += sum(r["input_ids"].numel() - r["prompt_len"] for r in records)
            open_ended += sum(int(r["input_ids"][-1]) not in end_ids for r in records)
        counts[which] = {"prompts": total, "response_tokens": tokens, "stopped_by_length": open_ended}
        status.log(f"{which}: {tokens} response tokens for {total} prompts; {open_ended} stopped by the length limit")
        del model
        common.release()
    return counts
