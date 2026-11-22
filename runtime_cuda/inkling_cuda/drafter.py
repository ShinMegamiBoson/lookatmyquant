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


class TemperatureSampler:
    """Sample target logits with one reusable random draw per input position."""
    def __init__(self, session, device, temperature, seed):
        self.cuda = session.cuda
        self.temperature = temperature
        generator = torch.Generator(device=device)
        generator.seed() if seed is None else generator.manual_seed(seed)
        self.draws = torch.rand(session.context, generator=generator, device=device)
        self.logits = torch.empty((PREFILL_CHUNK, session.model.head.size), device=device, dtype=torch.float32)

    def __call__(self, pointers, position, blocked_token):
        count = len(pointers)
        logits = self.logits[:count]
        width = logits.shape[1]
        for index, pointer in enumerate(pointers):
            self.cuda.copy(offset(U64(logits.data_ptr()), index * width), pointer, 4 * width)
        return sample_logits(logits, self.draws[position:position + count], self.temperature, blocked_token)


def sample_logits(logits, draws, temperature, blocked_token=None):
    """Draw from target logits using a uniform value for each absolute position."""
    valid = logits.isfinite().all(dim=1)
    if blocked_token is not None:
        logits[:, blocked_token] = -torch.inf
    # Center prior to scaling to keep very low temperatures finite at the maximum.
    weights = logits.sub_(logits.amax(dim=1, keepdim=True)).div_(temperature).exp_().cumsum(dim=1)
    total = weights[:, -1]
    threshold = torch.minimum(draws * total, torch.nextafter(total, torch.zeros_like(total)))
    picks = torch.searchsorted(weights, threshold[:, None], right=True).flatten()
    indices, finite = torch.stack((picks, valid.long())).tolist()
    if not all(finite) or any(index >= logits.shape[1] for index in indices):
        raise FloatingPointError("Invalid logits for temperature sampling")
    return indices



class Drafter:
    @torch.inference_mode()
    def __init__(self, path, model, context, device):
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
        self.device = torch.device(device)
        self.context = context
        self.block_size = config["block_size"]
        self.tap_ids = tuple(config["target_layer_ids"])
        self.hidden = config["hidden_size"]
        self.heads, self.kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
        self.dim, self.eps = config["head_dim"], config["rms_norm_eps"]
        self.rank = config["markov_rank"]
        if (self.hidden != model.cfg.hidden_size or not 2 <= self.block_size <= PREFILL_CHUNK
                or any(not 0 <= tap <= model.cfg.total_visits for tap in self.tap_ids)):
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
            capacity = min(context, window) if i in local_layers else context
            shape = (self.kv_heads, capacity, self.dim)
            self.caches.append(tuple(torch.empty(shape, device=device, dtype=torch.float16) for _ in range(2)))
        self.weights = {key: value for key, value in self.weights.items() if not key.startswith("layers.")}
        head = model.head
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

    def push(self, session, pointers, count):
        """Copy the committed target activations directly between CUDA allocations."""
        targets = torch.empty((count, len(self.tap_ids), self.hidden), device=self.device, dtype=torch.float32)
        cuda = session.cuda
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

    def embed(self, session, token):
        scratch = session.scratch
        scratch.mark = 0
        ptr = session.model.embedding.embed(token, scratch)
        session.dev.normalize(ptr, ptr, self.hidden, weight=session.model.embed_norm, eps=session.model.cfg.rms_norm_eps)
        out = torch.empty(self.hidden, device=self.device, dtype=torch.float32)
        session.cuda.copy(U64(out.data_ptr()), ptr, 4 * self.hidden)
        return out.bfloat16()

    def block(self, session, anchor):
        size = self.block_size
        masks = self.weights["mask_embed"].reshape(1, self.hidden).bfloat16().expand(size - 1, -1)
        h = torch.cat((self.embed(session, anchor).unsqueeze(0), masks))
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

    def draft_token(self, session, hidden, blocked_token):
        scratch = session.scratch
        scratch.mark = 0
        ptr, out = U64(hidden.data_ptr()), U64(self.head_scores.data_ptr())
        for lo, _, slab, signs in self.shortlist:
            session.model.head.project_bucket(ptr, slab, signs, offset(out, lo), scratch)
        if blocked_token is not None:
            index = int(session.model.head.inverse[blocked_token])
            if index < self.head_scores.numel():
                self.head_scores[index] = -torch.inf
        return self.head_ids.index_select(0, self.head_scores.argmax().reshape(1))

    def propose(self, session, anchor, count, blocked_token):
        hidden = self.block(session, anchor)
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
            tokens.append(self.draft_token(session, h + delta, blocked_token))
        return torch.cat(tokens).tolist()


@torch.inference_mode()
def stream_speculative(drafter, session, inputs, limit, eos, block_size, *,
                       blocked_token=None, sampler=None, stats=None):
    """Yield only target-verified IDs and reset the drafter's request state."""
    if not len(inputs) or not 1 <= limit <= session.context - len(inputs) + 1:
        raise ValueError("Prompt and generation exceed the context limit")
    if session.position != 0 or session.context > drafter.context:
        raise ValueError("DSpark requires a fresh target session within its context capacity")
    session.validate_blocked_token(blocked_token)
    device = drafter.device
    # The native driver launches and the PyTorch CUDA operations share this stream.
    with torch.cuda.device(device), torch.cuda.stream(torch.cuda.default_stream(device)):
        drafter.position = 0
        size = min(block_size or 4, drafter.block_size)
        stats = {} if stats is None else stats
        stats.update(draft_block_size=size, draft_checks=0, proposed_tokens=0, accepted_tokens=0)
        start = time.perf_counter()
        for i in range(0, len(inputs), PREFILL_CHUNK):
            prompt_chunk = inputs[i:i + PREFILL_CHUNK]
            if i + len(prompt_chunk) == len(inputs):
                predictions, taps = session.predict_block(
                    prompt_chunk, tap_visits=drafter.tap_ids, blocked_token=blocked_token, sampler=sampler,
                )
            else:
                _, taps = session.forward_block(prompt_chunk, logits=False, tap_visits=drafter.tap_ids)
            drafter.push(session, taps, len(prompt_chunk))
        torch.cuda.current_stream().synchronize()
        stats["prefill_seconds"] = time.perf_counter() - start

        anchor = int(predictions[0])
        yield anchor
        emitted = 1
        while emitted < limit and anchor not in eos:
            count = min(size, limit - emitted, session.context - session.position)
            if count == 1:
                yield session.predict(anchor, blocked_token=blocked_token, sampler=sampler)
                return
            draft = drafter.propose(session, anchor, count, blocked_token)
            base = session.position
            predictions, taps = session.predict_block(
                draft, all_tokens=True, tap_visits=drafter.tap_ids, blocked_token=blocked_token, sampler=sampler,
            )
            match = 0
            # Accept exactly the sampled target token. A rejected suffix reuses
            # the same positional draws when it is evaluated on the corrected path.
            for guess, actual in zip(draft[1:], predictions):
                if guess != actual:
                    break
                match += 1
                if guess in eos:
                    break
            committed = match + 1
            session.commit_block_prefix(base, committed)
            verified = draft[1:committed]
            if not verified or verified[-1] not in eos:
                verified.append(int(predictions[match]))
            stats["draft_checks"] += 1
            stats["accepted_tokens"] += match
            stats["proposed_tokens"] += len(draft) - 1
            for anchor in verified:
                yield anchor
                emitted += 1
            if emitted < limit and anchor not in eos:
                drafter.push(session, taps, committed)


def speculative_greedy(path, session, ordinal, inputs, limit, eos, block_size):
    device = torch.device("cuda", ordinal)
    start = time.perf_counter()
    with torch.cuda.device(device), torch.cuda.stream(torch.cuda.default_stream(device)):
        drafter = Drafter(Path(path), session.model, session.context, device)
        torch.cuda.current_stream().synchronize()
    stats = {"drafter_load_seconds": time.perf_counter() - start}
    stream = stream_speculative(drafter, session, inputs, limit, eos, block_size, stats=stats)
    output = [next(stream)]
    start = time.perf_counter()
    output.extend(stream)
    decode_seconds = time.perf_counter() - start
    rate = (len(output) - 1) / decode_seconds if len(output) > 1 else None
    return output, stats.pop("prefill_seconds"), rate, stats
