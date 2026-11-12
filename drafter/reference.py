"""A reference for ports of the trained drafter: its inputs and outputs on a few held-out blocks, as SpecForge's own
model computes them (the same calls score._draft makes), saved as one safetensors file under ROOT/reference/."""

from __future__ import annotations

import torch

from . import common, score


@torch.no_grad()
def dump(root, s, status, samples: int = 2, anchors: int = 6, max_tokens: int = 700) -> dict:
    from safetensors.torch import save_file
    drafter = score.load_drafter(root / "export", status)
    target = common.load_model(s, "build", status)
    embed, head = target.get_input_embeddings(), target.get_output_embeddings()
    out, size = {}, drafter.block_size
    for n, sample in enumerate(common.load_text(root, "heldout")[:samples]):
        start = sample["prompt_len"]
        ids = sample["input_ids"][:max(start + 2 * size, max_tokens)].to("cuda")   # a causal target: a prefix's states are the same
        length = ids.numel()
        states, _ = common.target_states(target, ids[None], None, drafter.target_layer_ids)
        pick = torch.linspace(start, length - 2, anchors).long().unique().to("cuda")
        count = pick.numel()
        block = torch.full((1, count, size), drafter.mask_token_id, dtype=torch.long, device="cuda")
        block[0, :, 0] = ids[pick]
        offsets = torch.arange(size, device="cuda")
        positions = torch.cat([torch.arange(length, device="cuda"), (pick[:, None] + offsets).reshape(-1)])[None]
        noise = embed(block.view(1, -1))
        o = drafter(position_ids=positions, attention_mask=score._masks(drafter, pick[None], length), noise_embedding=noise,
                    target_hidden=states[0][None], block_token_ids=block)
        logits = head(o.hidden_states[0, :, 1:]).float()
        top = logits.topk(drafter.dspine_config.top_k, dim=-1)
        scores = drafter.transition_scores(o.pre_injection[0, :, 1:], top.indices, ids[pick], head.weight)
        path = drafter.select_cached(scores, top.indices)
        context = drafter.hidden_norm(drafter.fc(states[0][None]))[0]
        out.update({f"{n}.ids": ids.int().cpu(), f"{n}.start": torch.tensor([start], dtype=torch.int32), f"{n}.anchors": pick.int().cpu(),
                    f"{n}.states": states[0].to(torch.bfloat16).cpu().contiguous(), f"{n}.context": context.float().cpu(),
                    f"{n}.noise": noise[0].float().cpu().view(count, size, -1)[:, :2].contiguous(),
                    f"{n}.hidden": o.hidden_states[0].float().cpu(), f"{n}.pre_injection": o.pre_injection[0].float().cpu(),
                    f"{n}.layer_features": o.layer_features[:, 0].float().cpu().contiguous(),
                    f"{n}.argmax": logits.argmax(-1).int().cpu(), f"{n}.top_ids": top.indices.int().cpu(), f"{n}.top_logits": top.values.cpu(),
                    f"{n}.scores": scores.float().cpu(), f"{n}.path": path.int().cpu()})
        status.log(f"reference sample {n}: {length} tokens, response from {start}, anchors {pick.tolist()}; argmax {logits.argmax(-1)[0, :8].tolist()}")
    (root / "reference").mkdir(parents=True, exist_ok=True)
    save_file(out, str(root / "reference" / "ref.safetensors"))
    return {"tensors": len(out)}
