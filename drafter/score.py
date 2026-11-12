"""Stages `baseline` and `score`: accepted tokens per check, simulated on each target's own greedy text.

Greedy speculative decoding accepts exactly the drafted tokens that equal the target's own greedy tokens, and a causal
target's states at a position do not depend on later tokens. So one teacher-forced pass over the target's greedy text gives
every state decoding would have had, and a round can be replayed from any position without decoding again.

A round is built the way training builds a block (SpecForge's OnlineDFlashModel and OnlineDSpineModel) and the way
decoding does (z-lab's dflash_generate): the anchor is a token the target has already produced, at position p; the drafter
sees the target's features of positions before p, the anchor's embedding and block_size - 1 mask embeddings at positions
p .. p + block_size - 1, and drafts the tokens after the anchor. With a of them right the round yields a + 1 tokens, the
last being the one the check itself supplies, and the next anchor is p + a + 1. The first anchor is the first response
token, which the target's pass over the prompt supplies. The draft for every possible anchor is computed in parallel
blocks, as in training, and the rounds then walk through them.

The masks come from SpecForge's own builder with the drafter's own setting: the stock DFlash drafter leaves is_causal unset
(its sliding layers see only earlier block positions, its full layer the whole block); the DSpine drafter is causal in
every layer."""

from __future__ import annotations

from collections import defaultdict

import torch

from . import common

METHOD = ("tokens_per_check is the mean number of tokens one target check yields in greedy speculative decoding: the "
          "drafted tokens that equal the target's own next tokens, plus the one the check adds, capped at the end of the "
          "response. Rounds start at the first response token. accepted_by_position[j-1] is the share of rounds whose "
          "first j drafted tokens were all right. Block size 8 uses the first 7 drafted tokens of the same draft.")


def load_drafter(path, status):
    """SpecForge's model class for the config in `path` with the weights beside it, on the GPU in bfloat16. Every weight
    in the file must be used and none may be missing."""
    from specforge.modeling.draft.registry import DRAFT_REGISTRY
    from specforge.modeling.target.checkpoint import load_checkpoint_tensors
    from specforge.training.model_loading import load_draft_config_source

    config = load_draft_config_source(str(path))
    method = getattr(config, "dflash_config", None) or {}

    def option(name, default):  # z-lab's model reads these from either place; this scorer supports only "off"
        return method.get(name, getattr(config, name, default))

    if (float(option("input_embedding_scale", 1.0)) != 1.0 or float(option("output_multiplier", 1.0)) != 1.0
            or float(option("final_logit_softcapping", 0) or 0) > 0):
        raise RuntimeError(f"{path} scales its embeddings or logits, which this scorer does not do")
    config._attn_implementation = "sdpa"
    model = DRAFT_REGISTRY[config.architectures[0]](config)
    state = load_checkpoint_tensors(str(path))
    result = model.load_state_dict(state, strict=False)
    status.log(f"drafter {path} ({type(model).__name__}, is_causal={getattr(config, 'is_causal', None)}): {len(state)} "
               f"tensors; missing {sorted(result.missing_keys)}; unused {sorted(result.unexpected_keys)}")
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"the weights in {path} do not match {type(model).__name__}")
    return model.to("cuda", torch.bfloat16).eval()


def _masks(drafter, anchors, length: int):
    """The masks training builds (OnlineDFlashModel._attention_mask), from the dense builder."""
    from specforge.modeling.draft.dflash_mask import create_dflash_sdpa_mask

    shared = dict(anchor_positions=anchors, block_keep_mask=torch.ones_like(anchors, dtype=torch.bool), S=length,
                  block_size=drafter.block_size, device=anchors.device, is_causal=getattr(drafter.config, "is_causal", None))
    if drafter.sliding_window is None:
        return create_dflash_sdpa_mask(**shared)
    masks = {"sliding_attention": create_dflash_sdpa_mask(**shared, sliding_window=drafter.sliding_window)}
    if "full_attention" in drafter.layer_types:
        masks["full_attention"] = create_dflash_sdpa_mask(**shared)
    return masks


def _draft(drafter, embed, head, states, ids, anchors) -> dict:
    """The tokens drafted for the block_size - 1 positions after each anchor, as {decoding: [anchors, block_size - 1]}.
    states [T, layers * H] and ids [T] are one sample's; anchors [n] are positions in it."""
    count, size, length = anchors.numel(), drafter.block_size, ids.numel()
    block = torch.full((1, count, size), drafter.mask_token_id, dtype=torch.long, device=ids.device)
    block[0, :, 0] = ids[anchors]
    offsets = torch.arange(size, device=ids.device)
    positions = torch.cat([torch.arange(length, device=ids.device), (anchors[:, None] + offsets).reshape(-1)])[None]
    inputs = dict(position_ids=positions, attention_mask=_masks(drafter, anchors[None], length),
                  noise_embedding=embed(block.view(1, -1)), target_hidden=states[None])
    if not hasattr(drafter, "transition_scores"):
        hidden = drafter(**inputs).view(count, size, -1)[:, 1:]
        return {"argmax": head(hidden).float().argmax(-1)}
    out = drafter(**inputs, block_token_ids=block)  # without replacement only the anchor's id is read from the block
    logits = head(out.hidden_states[0, :, 1:]).float()
    candidates = logits.topk(drafter.dspine_config.top_k, dim=-1).indices
    scores = drafter.transition_scores(out.pre_injection[0, :, 1:], candidates, ids[anchors], head.weight)
    return {"argmax": logits.argmax(-1), "path": drafter.select_cached(scores, candidates)}


def _summary(rows: list, block: int, split: bool = True) -> dict:
    """rows: one (drafted tokens accepted, tokens produced, source) per round."""
    out = {"tokens_per_check": sum(row[1] for row in rows) / len(rows), "rounds": len(rows),
           "accepted_by_position": [sum(row[0] >= j for row in rows) / len(rows) for j in range(1, block)]}
    groups = defaultdict(list)
    for row in rows:
        if split and row[2] is not None:
            groups[row[2]].append(row)
    if groups:
        out["by_source"] = {source: _summary(group, block, split=False) for source, group in sorted(groups.items())}
    return out


@torch.no_grad()
def score_pair(target, drafter, samples: list, s, status, label: str) -> dict:
    """Replay greedy speculative decoding of every sample with this drafter and target."""
    embed, head = target.get_input_embeddings(), target.get_output_embeddings()
    reach = torch.arange(1, drafter.block_size, device="cuda")
    rounds = defaultdict(list)
    used = same = total = first_same = first_total = 0
    for number, sample in enumerate(samples, 1):
        ids, start = sample["input_ids"].to("cuda"), sample["prompt_len"]
        length = ids.numel()
        if length - start >= 2:  # a response of one token leaves nothing to draft
            used += 1
            states, last = common.target_states(target, ids[None], None, drafter.target_layer_ids)
            same += int((head(last[0, start - 1:length - 1]).float().argmax(-1) == ids[start:]).sum())
            total += length - start
            anchors = torch.arange(start, length - 1, device="cuda")
            drafted_at = anchors[:, None] + reach
            truth = torch.where(drafted_at < length, ids[drafted_at.clamp(max=length - 1)], -1)
            drafted = defaultdict(list)
            for part in anchors.split(s.score_anchors):
                for decoding, tokens in _draft(drafter, embed, head, states[0], ids, part).items():
                    drafted[decoding].append(tokens)
            drafted = {decoding: torch.cat(parts) for decoding, parts in drafted.items()}
            if "path" in drafted:  # both decodings read the first drafted position from the same state
                first_same += int((drafted["path"][:, 0] == drafted["argmax"][:, 0]).sum())
                first_total += anchors.numel()
            for decoding, tokens in drafted.items():
                right = (tokens == truth).long().cumprod(-1).sum(-1).tolist()  # leading drafted tokens that are right
                for block in s.score_blocks:
                    position = start
                    while position < length - 1:
                        accepted = min(right[position - start], block - 1)
                        produced = min(accepted + 1, length - 1 - position)
                        rounds[(decoding, block)].append((accepted, produced, sample["source"]))
                        position += produced
        if number % 25 == 0 or number == len(samples):
            now = {f"{decoding}@{block}": round(sum(row[1] for row in rows) / len(rows), 3)
                   for (decoding, block), rows in sorted(rounds.items())}
            status.log(f"{label}: {number} of {len(samples)} samples, tokens per check {now}", samples=number, total=len(samples))
    if not rounds:
        raise RuntimeError(f"{label}: no sample has two response tokens")
    result = {"samples": used, "response_tokens": total, "teacher_forced_agreement": same / total, "decoding": {}}
    for (decoding, block), rows in sorted(rounds.items()):
        result["decoding"].setdefault(decoding, {})[str(block)] = _summary(rows, block)
    if first_total:
        result["path_equals_argmax_at_first_position"] = first_same / first_total
    return result


def _headline(result: dict) -> dict:
    return {f"{decoding}@{block}": round(summary["tokens_per_check"], 4)
            for decoding, blocks in result["decoding"].items() for block, summary in blocks.items()}


def run_baseline(root, s, status):
    """The stock DFlash drafter against the original model and against the build, before any training."""
    drafter = load_drafter(root / "assets" / "dflash", status)
    targets = {}
    for which, prefix in (("original", "heldout_original"), ("build", "heldout")):
        samples = common.load_text(root, prefix)
        target = common.load_model(s, which, status)
        common.check_target_states(target, samples[0]["input_ids"], status)
        targets[which] = score_pair(target, drafter, samples, s, status, f"DFlash on the {which} model")
        del target
        common.release()
    headline = {which: _headline(result) for which, result in targets.items()}
    common.write_json(root / "scores_baseline.json", {"drafter": s.dflash_repo, "method": METHOD,
                                                      "tokens_per_check": headline, "targets": targets})
    mean = targets["original"]["decoding"]["argmax"][str(drafter.block_size)]["tokens_per_check"]
    if mean < s.suspect_below:
        status.echo(f"SCORER-SUSPECT the stock DFlash drafter gets {mean:.2f} tokens per check on the original model, "
                    f"under {s.suspect_below}: the scorer or the loading is wrong, not the drafter", every=0)
    return {"tokens_per_check": headline}


def run_score(root, s, status):
    """The trained DSpine drafter against the build: DSpine's path selection, and each position's own best token."""
    drafter = load_drafter(root / "export", status)
    samples = common.load_text(root, "heldout")
    target = common.load_model(s, "build", status)
    common.check_target_states(target, samples[0]["input_ids"], status)
    result = score_pair(target, drafter, samples, s, status, "DSpine on the build")
    baseline = common.read_json(root / "scores_baseline.json")
    headline = {"dspine_on_build": _headline(result), "dflash_on_build": baseline["tokens_per_check"]["build"],
                "dflash_on_original": baseline["tokens_per_check"]["original"]}
    common.write_json(root / "scores.json", {"method": METHOD, "tokens_per_check": headline, "dspine_on_build": result,
                                             "baseline": baseline})
    status.log(f"tokens per check: {headline}")
    return {"tokens_per_check": headline}
