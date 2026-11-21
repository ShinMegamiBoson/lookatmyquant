"""CUDA DSpark for LookAtMyQuant's Inkling 2 Mini drafter pack.

The published affine 4-bit weights are expanded once on the GPU for BF16 GEMMs.
The target continues to read its own trellis-packed weights directly.
"""

from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .device_model import PREFILL_CHUNK, offset, row
from .driver import U64
from .model import read_json


def dequantize_4bit(words, scales, biases, group_size):
    rows, groups = scales.shape
    if words.shape != (rows, groups * group_size // 8) or biases.shape != scales.shape:
        raise ValueError("Invalid DSpark affine weight shape")
    shifts = torch.arange(0, 32, 4, device=words.device, dtype=torch.int32)
    output = torch.empty((rows, groups * group_size), device=words.device, dtype=torch.bfloat16)
    for start in range(0, rows, 8192):
        end = min(start + 8192, rows)
        packed = words[start:end].view(torch.int32)
        values = ((packed.unsqueeze(-1) >> shifts) & 15).reshape(end - start, groups, group_size)
        decoded = values * scales[start:end].float().unsqueeze(-1) + biases[start:end].float().unsqueeze(-1)
        output[start:end].copy_(decoded.reshape(end - start, -1))
    return output


def linear(x, weight):
    return F.linear(x.to(weight.dtype), weight)


def rms(x, weight, eps):
    return F.rms_norm(x.float(), (x.shape[-1],), weight.float(), eps).to(x.dtype)


class Drafter:
    def __init__(self, path, target, device):
        folder = path if path.is_dir() else path.parent
        config = read_json(folder / "config.json")["mtp_config"]
        manifest_path = folder / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        if manifest.get("bits", 4) != 4:
            raise ValueError("Expected a 4-bit DSpark affine pack")
        group_size = manifest.get("group_size", 64)
        if group_size < 8 or group_size % 8:
            raise ValueError("DSpark group size must be a positive multiple of eight")
        if path.is_dir():
            path = folder / manifest.get("weights", "mtp_4bit.safetensors")
        self.device, self.target = device, target
        self.block_size = config["block_size"]
        self.tap_ids = tuple(config["target_layer_ids"])
        self.hidden = config["hidden_size"]
        self.heads, self.kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
        self.dim, self.eps = config["head_dim"], config["rms_norm_eps"]
        self.rank = config["markov_rank"]
        if (self.hidden != target.model.cfg.hidden_size or not 2 <= self.block_size <= PREFILL_CHUNK
                or any(not 0 <= tap <= target.model.cfg.total_visits for tap in self.tap_ids)):
            raise ValueError("DSpark config does not match the target decoder")
        self.q_dim, self.kv_dim = self.heads * self.dim, self.kv_heads * self.dim
        self.inv_freq = config["rope_theta"] ** (
            -torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim
        )
        self.position = 0
        local_layers = set(config["local_layer_ids"])
        window = config["sliding_window_size"]
        self.weights = {}
        weights = load_file(str(path))
        for name, value in weights.items():
            if name.endswith((".scales", ".biases")):
                continue
            scales = weights.get(name + ".scales")
            if scales is not None:
                value = dequantize_4bit(value.to(device), scales.to(device),
                                        weights[name + ".biases"].to(device), group_size)
            else:
                value = value.to(device)
                if value.ndim == 2 and name.endswith(".weight"):
                    value = value.float()
            self.weights[name] = value
        self.fuse = self.weights["fuse_weight"].float().softmax(dim=-1)
        self.caches = []
        self.layers = []
        for i in range(config["num_hidden_layers"]):
            prefix = f"layers.{i}."
            layer = {k.removeprefix(prefix): v for k, v in self.weights.items() if k.startswith(prefix)}
            for key, names in (
                ("qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
                ("kv_ctx", ("self_attn.k_proj_ctx", "self_attn.v_proj_ctx")),
                ("gate_up", ("mlp.gate_proj", "mlp.up_proj")),
            ):
                layer[key] = torch.cat([layer.pop(name + ".weight") for name in names])
            self.layers.append(layer)
            capacity = min(target.context, window) if i in local_layers else target.context
            shape = (self.kv_heads, capacity, self.dim)
            self.caches.append(tuple(torch.empty(shape, device=device, dtype=torch.float16) for _ in range(2)))
        self.weights = {key: value for key, value in self.weights.items() if not key.startswith("layers.")}
        head = target.model.head
        self.shortlist = head.buckets[:5]
        shortlist_size = self.shortlist[-1][1]
        ids = np.argsort(head.inverse)[:shortlist_size]
        self.head_ids = torch.as_tensor(ids, device=device, dtype=torch.long)
        self.head_scores = torch.empty(shortlist_size, device=device, dtype=torch.float32)

    def rope(self, x, position):
        angles = position[:, None] * self.inv_freq
        cosine, sine = angles.cos(), angles.sin()
        a, b = x.float().chunk(2, dim=-1)
        return torch.cat((a * cosine - b * sine, b * cosine + a * sine), dim=-1).half()

    def fuse_taps(self, taps, index, weight):
        return rms((taps * self.fuse[index, :, None]).sum(dim=1), weight, 1e-6)

    def push(self, pointers, count):
        """Copy the committed target activations directly between CUDA allocations."""
        targets = torch.empty((count, len(self.tap_ids), self.hidden), device=self.device, dtype=torch.float32)
        cuda = self.target.cuda
        for i in range(count):
            for j, visit in enumerate(self.tap_ids):
                cuda.copy(U64(targets[i, j].data_ptr()), row(pointers[visit], i), 4 * self.hidden)
        positions = torch.arange(self.position, self.position + count, device=self.device)
        for i, (layer, (keys, values)) in enumerate(zip(self.layers, self.caches, strict=True)):
            ctx = self.fuse_taps(targets, i, layer["dflare_norm.weight"])
            k, v = linear(ctx, layer["kv_ctx"]).bfloat16().chunk(2, dim=-1)
            k = k.reshape(count, self.kv_heads, self.dim).transpose(0, 1)
            k = self.rope(rms(k, layer["self_attn.k_norm_ctx.weight"], self.eps), positions)
            v = v.reshape(count, self.kv_heads, self.dim).transpose(0, 1).half()
            capacity = keys.shape[1]
            # Only the last window of a long prefill chunk is eligible for draft attention.
            indices = positions[-capacity:] % capacity
            keys.index_copy_(1, indices, k[:, -capacity:])
            values.index_copy_(1, indices, v[:, -capacity:])
        self.condition = self.fuse_taps(
            targets[-1:], len(self.layers), self.weights["markov_head.dflare_norm.weight"],
        )[0]
        self.position += count

    def embed(self, token):
        session = self.target
        scratch = session.scratch
        scratch.mark = 0
        ptr = session.model.embedding.embed(token, scratch)
        session.dev.normalize(ptr, ptr, self.hidden, weight=session.model.embed_norm, eps=session.model.cfg.rms_norm_eps)
        out = torch.empty(self.hidden, device=self.device, dtype=torch.float32)
        session.cuda.copy(U64(out.data_ptr()), ptr, 4 * self.hidden)
        return out.bfloat16()

    def block(self, anchor):
        size = self.block_size
        masks = self.weights["mask_embed"].reshape(1, self.hidden).bfloat16().expand(size - 1, -1)
        h = torch.cat((self.embed(anchor).unsqueeze(0), masks))
        positions = torch.arange(self.position, self.position + size, device=self.device)
        for layer, (keys, values) in zip(self.layers, self.caches, strict=True):
            q, k, v = linear(rms(h, layer["input_layernorm.weight"], self.eps), layer["qkv"]).bfloat16().split(
                (self.q_dim, self.kv_dim, self.kv_dim), dim=-1,
            )
            q = q.reshape(size, self.heads, self.dim).transpose(0, 1)
            k = k.reshape(size, self.kv_heads, self.dim).transpose(0, 1)
            v = v.reshape(size, self.kv_heads, self.dim).transpose(0, 1).half()
            q = self.rope(rms(q, layer["self_attn.q_norm.weight"], self.eps), positions)
            k = self.rope(rms(k, layer["self_attn.k_norm.weight"], self.eps), positions)
            seen = min(self.position, keys.shape[1])
            k = torch.cat((keys[:, :seen], k), dim=1)
            v = torch.cat((values[:, :seen], v), dim=1)
            repeat = self.heads // self.kv_heads
            attended = F.scaled_dot_product_attention(
                q.unsqueeze(0), k.repeat_interleave(repeat, dim=0).unsqueeze(0),
                v.repeat_interleave(repeat, dim=0).unsqueeze(0),
            ).transpose(1, 2).reshape(size, self.q_dim).bfloat16()
            h = h + linear(attended, layer["self_attn.o_proj.weight"]).bfloat16()
            normed = rms(h, layer["post_attention_layernorm.weight"], self.eps)
            gate, up = linear(normed, layer["gate_up"]).bfloat16().chunk(2, dim=-1)
            h = h + linear(F.silu(gate) * up, layer["mlp.down_proj.weight"]).bfloat16()
        return rms(h.float(), self.weights["norm.weight"], self.eps)

    def draft_token(self, hidden):
        scratch = self.target.scratch
        scratch.mark = 0
        ptr, out = U64(hidden.data_ptr()), U64(self.head_scores.data_ptr())
        for lo, _, slab, signs in self.shortlist:
            self.target.model.head.project_bucket(ptr, slab, signs, offset(out, lo), scratch)
        return self.head_ids.index_select(0, self.head_scores.argmax().reshape(1))

    def propose(self, anchor, count):
        hidden = self.block(anchor)
        state = torch.zeros(self.rank, device=self.device)
        tokens = [torch.tensor([anchor], device=self.device, dtype=torch.long)]
        w = self.weights
        for h in hidden[1:count]:
            prev = w["markov_head.markov_w1.weight"].index_select(0, tokens[-1])[0]
            x = torch.cat((state, prev.float(), h, self.condition))
            gates = linear(x, w["markov_head.rnn_proj.weight"]).float() + w["markov_head.rnn_proj.bias"]
            gate, candidate, latent = gates.chunk(3)
            gate = gate.sigmoid()
            state = gate * state + (1 - gate) * candidate.tanh()
            delta = linear(latent.tanh(), w["markov_head.hidden_out.weight"]).float() + w["markov_head.hidden_out.bias"]
            tokens.append(self.draft_token(h + delta))
        return torch.cat(tokens).tolist()


@torch.inference_mode()
def speculative_greedy(path, session, ordinal, inputs, limit, eos, block_size):
    device = torch.device("cuda", ordinal)
    # The native driver launches and the PyTorch CUDA operations share this stream.
    with torch.cuda.device(device), torch.cuda.stream(torch.cuda.default_stream(device)):
        start = time.perf_counter()
        drafter = Drafter(Path(path), session, device)
        torch.cuda.current_stream().synchronize()
        load_seconds = time.perf_counter() - start
        size = min(block_size or 4, drafter.block_size)
        start = time.perf_counter()
        for i in range(0, len(inputs), PREFILL_CHUNK):
            prompt_chunk = inputs[i:i + PREFILL_CHUNK]
            if i + len(prompt_chunk) == len(inputs):
                predictions, taps = session.predict_block(prompt_chunk, tap_visits=drafter.tap_ids)
            else:
                _, taps = session.forward_block(prompt_chunk, logits=False, tap_visits=drafter.tap_ids)
            drafter.push(taps, len(prompt_chunk))
        torch.cuda.current_stream().synchronize()
        prefill_seconds = time.perf_counter() - start

        output = [int(predictions[0])]
        accepted = proposed = checks = 0
        start = time.perf_counter()
        while len(output) < limit and output[-1] not in eos:
            count = min(size, limit - len(output), session.context - session.position)
            anchor = output[-1]
            if count == 1:
                output.append(session.predict(anchor))
                break
            draft = drafter.propose(anchor, count)
            base = session.position
            predictions, taps = session.predict_block(draft, all_tokens=True, tap_visits=drafter.tap_ids)
            match = 0
            for guess, actual in zip(draft[1:], predictions):
                if guess != actual:
                    break
                match += 1
                if guess in eos:
                    break
            committed = match + 1
            session.commit_block_prefix(base, committed)
            output.extend(draft[1:committed])
            if output[-1] not in eos:
                output.append(int(predictions[match]))
            if len(output) < limit and output[-1] not in eos:
                drafter.push(taps, committed)
            checks += 1
            accepted += match
            proposed += len(draft) - 1
        decode_seconds = time.perf_counter() - start
        rate = (len(output) - 1) / decode_seconds if len(output) > 1 else None
        return output, prefill_seconds, rate, {
            "drafter_load_seconds": load_seconds,
            "draft_block_size": size,
            "draft_checks": checks,
            "proposed_tokens": proposed,
            "accepted_tokens": accepted,
        }
