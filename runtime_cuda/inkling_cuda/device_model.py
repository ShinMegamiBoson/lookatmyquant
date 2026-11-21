"""Device-resident serial decoder for LookAtMyQuant's Inkling text checkpoints."""
import math
from dataclasses import dataclass

import numpy as np

from .device_ptx import source
from .driver import F32, PTR, SIZE, U32, U64
from .model import Model
from .packed import EXPERT_FORMATS, hadamard_factors


NULL = U64()
THREADS = 128


def tiles(size, width=THREADS):
    return (size + width - 1) // width


def offset(pointer, floats):
    return U64(pointer.value + 4 * floats)


@dataclass(frozen=True)
class Weight:
    ptr: U64
    shape: tuple


class Device:
    def __init__(self, cuda):
        self.cuda = cuda
        self.module = cuda.load_module(source())
        self.functions = {}
        self.bases = {}
        self.zero_index = cuda.upload(np.zeros(1, dtype=np.int32))

    def close(self):
        if self.module:
            self.cuda.call("cuModuleUnload", self.module)
            self.module = PTR()

    def weight(self, value):
        return Weight(self.cuda.upload(value), value.shape)

    def launch(self, name, args, grid, shared=0):
        if name not in self.functions:
            self.functions[name] = self.cuda.module_function(self.module, name)
        self.cuda.launch(self.functions[name], grid, THREADS, *args, shared=shared)

    def vector(self, name, size, *args):
        self.launch(name, args, (tiles(size), 1, 1))

    def normalize(self, x, out, width, groups=1, weight=None, eps=1e-6, clip=None, factor=1):
        self.launch(
            "rms_norm" if clip is None else "clip_norm",
            (x, weight.ptr if weight else NULL, out, U32(width),
             F32(eps if clip is None else clip), F32(factor)),
            (groups, 1, 1),
        )

    def dense(self, x, weight, out):
        rows, width = weight.shape
        self.launch("dense_matvec", (x, weight.ptr, out, U32(rows), U32(width)), (tiles(rows, 4), 1, 1))

    def rotate(self, x, signs, out, inverse=False, rows=1):
        width, = signs.shape
        if width not in self.bases:
            _, base = hadamard_factors(width)
            self.bases[width] = (base.shape[0], self.weight(base).ptr) if base.shape[1] > 1 else (1, NULL)
        base, table = self.bases[width]
        self.launch("hadamard", (x, signs.ptr, table, out, U32(width), U32(base),
                                U32(inverse), F32(width ** -0.5)), (rows, 1, 1), shared=4 * width)


class DeviceLinear:
    def __init__(self, device, linear):
        self.device = device
        self.slab = linear.slab
        self.signs = device.weight(linear.signs)
        self.out_signs = device.weight(linear.out_signs) if linear.out_signs is not None else None

    def __call__(self, x, scratch):
        dev, slab = self.device, self.slab
        out = scratch.take(slab.rows)
        mark = scratch.mark
        rotated = scratch.take(slab.width)
        projected = scratch.take(slab.rows) if self.out_signs is not None else out
        dev.rotate(x, self.signs, rotated)
        dev.cuda.packed_project(slab, rotated, dev.zero_index, projected, 1, 1)
        if self.out_signs is not None:
            dev.rotate(projected, self.out_signs, out, inverse=True)
        scratch.mark = mark
        return out


class DeviceExperts:
    def __init__(self, device, experts):
        self.device = device
        self.signs = device.weight(experts.signs)
        self.mapping = device.weight(experts.mapping)
        first = next(iter(experts.slabs.values()))
        self.rows, self.width = first.rows, first.width
        descriptors = np.zeros((len(EXPERT_FORMATS), 6), dtype=np.uint64)
        fields = descriptors.view(np.uint32).reshape(len(EXPERT_FORMATS), 12)
        for i, slab in experts.slabs.items():
            descriptors[i, :3] = slab.codes.value, slab.scales.value, slab.table.value
            fields[i, 6:] = slab.words, slab.bits, slab.group, slab.hash_shift, slab.multiplier, slab.lloyd
        self.descriptors = device.weight(descriptors)

    def project(self, x, selected, slots, x_div, scratch):
        out = scratch.take(slots * self.rows)
        mark = scratch.mark
        rotated = scratch.take(slots // x_div * self.width)
        self.device.rotate(x, self.signs, rotated, rows=slots // x_div)
        self.device.cuda.routed_project(rotated, selected, out, slots, x_div, self.rows, self.width,
                                        self.descriptors.ptr, self.mapping.ptr)
        scratch.mark = mark
        return out


class DeviceVocabulary:
    def __init__(self, device, vocab):
        self.device = device
        self.buckets = [(lo, hi, slab, device.weight(signs)) for lo, hi, slab, signs in vocab.buckets]
        self.order = device.weight(vocab.order)
        self.inverse = vocab.inverse
        self.size = vocab.order.size

    def embed(self, token, scratch):
        if not 0 <= token < self.size:
            raise ValueError(f"Token {token} is outside the embedding vocabulary")
        position = int(self.inverse[token])
        for lo, hi, slab, signs in self.buckets:
            if lo <= position < hi:
                out = scratch.take(slab.width)
                mark = scratch.mark
                row = scratch.take(slab.width)
                self.device.vector(
                    "packed_row", slab.width, slab.codes, slab.scales, slab.table, row,
                    U32(position - lo), U32(slab.width), U32(slab.words), U32(slab.bits),
                    U32(slab.group), U32(slab.hash_shift), U32(slab.multiplier), U32(slab.lloyd),
                )
                self.device.rotate(row, signs, out, inverse=True)
                scratch.mark = mark
                return out
        raise AssertionError("Invalid embedding lookup")

    def logits(self, x, out, cap, scratch):
        dev = self.device
        mark = scratch.mark
        for lo, hi, slab, signs in self.buckets:
            rotated = scratch.take(slab.width)
            projected = scratch.take(slab.rows)
            dev.rotate(x, signs, rotated)
            dev.cuda.packed_project(slab, rotated, dev.zero_index, projected, 1, 1)
            dev.vector("head_scatter", hi - lo, projected, offset(self.order.ptr, lo), out, U32(hi - lo), F32(cap))
            scratch.mark = mark


class Scratch:
    def __init__(self, pointer, capacity):
        self.pointer = pointer
        self.capacity = capacity
        self.mark = 0

    def take(self, count):
        start = self.mark
        self.mark += (count + 31) & ~31
        if self.mark > self.capacity:
            raise RuntimeError("Decoder scratch capacity exceeded")
        return offset(self.pointer, start)


def block_attention(dev, query, blocks, partial, width, scratch):
    count = len(blocks) + 1
    reads = math.prod(query.shape) // width
    result = scratch.take(reads * width)
    mark = scratch.mark
    states = scratch.take(count * width)
    scores = scratch.take(count * reads)
    for i, hidden in enumerate([*blocks, partial]):
        dev.normalize(hidden, offset(states, i * width), width)
    dev.launch("block_scores", (states, query.ptr, scores, U32(width), U32(count), U32(reads), F32(width ** -0.5)),
               (tiles(count * reads, 4), 1, 1))
    dev.launch("block_mix", (states, scores, result, U32(width), U32(count)), (tiles(width), reads, 1))
    scratch.mark = mark
    return tuple(offset(result, i * width) for i in range(reads))


class DeviceLayer:
    def __init__(self, device, layer):
        self.dev = device
        self.cfg = layer.cfg
        self.sliding = layer.sliding
        self.boundary = layer.boundary
        self.w = {}
        for key, weight in layer.w.items():
            if key.endswith("conv1d.weight"):
                weight = weight[:, 0, :].T
            if key == "self_attn.relative_position_bias.proj":
                weight = weight.T
            if key == "self_attn.q_proj.norm.scale":
                weight = np.broadcast_to(weight, (self.cfg.num_attention_heads, self.cfg.head_dim))
            if key == "self_attn.relative_position_bias.norm.scale":
                weight = np.broadcast_to(weight, (self.cfg.num_attention_heads, self.cfg.d_rel))
            self.w[key] = device.weight(weight)
        self.linear = {name: DeviceLinear(device, linear) for name, linear in layer.linear.items()}
        self.gu = DeviceExperts(device, layer.gu)
        self.dn = DeviceExperts(device, layer.dn)

    def conv(self, x, weight, history, position, scratch, dynamic=NULL):
        taps, width = weight.shape
        out = scratch.take(width)
        divisor = 2 * taps + 1 if dynamic.value else taps + 1
        self.dev.vector("causal_conv", width, x, weight.ptr, dynamic, history, out,
                        U32(width), U32(taps), U32(position), U32(self.cfg.head_dim), F32(divisor ** -0.5))
        return out

    def kv_conv(self, x, normalized, name, history, position, scratch):
        prefix = f"self_attn.{name}_proj.sconv."
        if self.sliding:
            return self.conv(x, self.w[prefix + "conv1d.weight"], history, position, scratch)
        weight = self.w[prefix + "weight_proj.weight"]
        dynamic = scratch.take(weight.shape[0])
        self.dev.dense(normalized, weight, dynamic)
        return self.conv(x, self.w[prefix + "static_weight"], history, position, scratch, dynamic)

    def message_gate(self, x, message, name, scratch):
        dev, h = self.dev, self.cfg.hidden_size
        prefix = f"{name}_message_gate."
        down = self.w[prefix + "down_proj.weight"]
        forget = self.w[prefix + "forget_gate_proj.weight"]
        add = self.w[prefix + "message_gate_proj.weight"]
        out = scratch.take(h)
        mark = scratch.mark
        pair = scratch.take(2 * h)
        rank = scratch.take(down.shape[0])
        f, a = scratch.take(h), scratch.take(h)
        dev.normalize(x, pair, h)
        dev.normalize(message, offset(pair, h), h)
        dev.vector("vec_bf16", 2 * h, pair, pair, U32(2 * h))
        dev.dense(pair, down, rank)
        dev.vector("vec_bf16", down.shape[0], rank, rank, U32(down.shape[0]))
        dev.dense(rank, forget, f)
        dev.dense(rank, add, a)
        dev.vector("message_mix", h, x, message, f, a, out, U32(h))
        scratch.mark = mark
        return out

    def attention(self, q_in, k_in, v_in, position, cache, scores, scratch):
        cfg, dev, w = self.cfg, self.dev, self.w
        q = self.linear["q_proj.linear"](q_in, scratch)
        k = self.kv_conv(self.linear["k_proj.linear"](k_in, scratch), k_in, "k", cache.conv[0], position, scratch)
        v = self.kv_conv(self.linear["v_proj.linear"](v_in, scratch), v_in, "v", cache.conv[1], position, scratch)
        rel = self.linear["relative_position_bias.linear"](q_in, scratch)
        factor = 1.0
        if not self.sliding:
            factor += cfg.log_scaling_alpha * math.log(max((position + 1) / cfg.log_scaling_n_floor, 1))
        q_heads, kv_heads, dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        dev.normalize(q, q, dim, q_heads, w["self_attn.q_proj.norm.scale"],
                      clip=cfg.qk_l2_clip_threshold, factor=factor)
        dev.normalize(k, k, dim, kv_heads, clip=cfg.qk_l2_clip_threshold)
        dev.normalize(v, v, dim, kv_heads, eps=cfg.rms_norm_eps)
        dev.normalize(rel, rel, cfg.d_rel, q_heads, w["self_attn.relative_position_bias.norm.scale"],
                      clip=cfg.rel_l2_clip_threshold, factor=factor)
        kv_width = kv_heads * dim
        dev.vector("kv_store", kv_width, k, v, cache.k, cache.v, U32(kv_width), U32(dim), U32(cache.capacity),
                   U32(position % cache.capacity))
        count = min(position + 1, cache.capacity)
        repeat = q_heads // kv_heads
        dev.launch("attention_scores", (q, cache.k, rel, w["self_attn.relative_position_bias.proj"].ptr,
                                       scores, U32(count), U32(dim), U32(repeat), U32(position),
                                       U32(cache.capacity), U32(cfg.d_rel), U32(cfg.rel_extent), F32(dim ** -0.5)),
                   (tiles(count, 4), q_heads, 1))
        sink = scratch.take(q_heads)
        dev.dense(q_in, w["self_attn.sink_proj.weight"], sink)
        attended = scratch.take(q_heads * dim)
        dev.launch("attention_mix", (scores, cache.v, sink, attended, U32(count), U32(dim),
                                     U32(repeat), U32(position), U32(cache.capacity)),
                   (q_heads, 1, 1))
        gate = self.linear["o_proj.gate_proj"](q_in, scratch)
        dev.vector("attention_gate", q_heads * dim, attended, gate, attended, U32(q_heads * dim))
        projected = self.linear["o_proj.linear"](attended, scratch)
        return self.conv(projected, w["post_attention_convolution.conv1d.weight"], cache.conv[2], position, scratch)

    def moe(self, x, scratch):
        dev, cfg = self.dev, self.cfg
        h, d, e, top = cfg.hidden_size, cfg.intermediate_size, cfg.n_routed_experts, cfg.num_experts_per_tok
        logits = scratch.take(e)
        probs = scratch.take(top)
        selected = scratch.take(top)
        dev.dense(x, self.w["mlp.gate.weight"], logits)
        dev.launch("select_experts", (logits, self.w["mlp.gate.e_score_correction_bias"].ptr,
                                      probs, selected, U32(e), U32(top), F32(cfg.route_scale)),
                   (1, 1, 1), shared=8 * e)
        gu = self.gu.project(x, selected, top, top, scratch)
        act = scratch.take(top * d)
        dev.launch("softcapped_swiglu", (gu, act, U32(d), F32(cfg.mlp_gate_softcap), F32(cfg.mlp_up_softcap)),
                   (tiles(d), top, 1))
        dn = self.dn.project(act, selected, top, 1, scratch)
        out = scratch.take(h)
        dev.vector("mix_experts", h, dn, probs, out, U32(h), U32(top))
        return out

    def __call__(self, x, blocks, position, cache, messages, scores, scratch, out):
        cfg, dev, w = self.cfg, self.dev, self.w
        h = cfg.hidden_size
        query = w.get("attention_input.query")
        inputs = (x,) * 3
        if query is not None:
            reads = block_attention(dev, query, blocks, x, h, scratch)
            inputs = (reads[0],) * 3 if self.sliding else reads
        normalized = []
        for i, (name, hidden) in enumerate(zip("qkv", inputs, strict=True)):
            if messages is not None:
                hidden = self.message_gate(hidden, messages[i], name, scratch)
            norm = "input_layernorm.weight" if self.sliding else f"{name}_input_layernorm.weight"
            y = scratch.take(h)
            dev.normalize(hidden, y, h, weight=w[norm], eps=cfg.rms_norm_eps)
            normalized.append(y)
        attn = self.attention(*normalized, position, cache, scores, scratch)
        residual, local_blocks = attn, blocks
        if self.boundary:
            local_blocks = [*blocks, x]
        else:
            dev.vector("vec_add", h, attn, x, residual, U32(h))
        hidden, = block_attention(dev, w["mlp_input.query"], local_blocks, residual, h, scratch)
        if messages is not None:
            hidden = self.message_gate(hidden, messages[3], "mlp", scratch)
        dev.normalize(hidden, hidden, h, weight=w["post_attention_layernorm.weight"], eps=cfg.rms_norm_eps)
        mixed = self.moe(hidden, scratch)
        conv = self.conv(mixed, w["post_mlp_convolution.conv1d.weight"], cache.conv[3], position, scratch)
        dev.vector("vec_add", h, residual, conv, out, U32(h))


class DeviceModel:
    def __init__(self, folder, cuda):
        self.dev = Device(cuda)
        try:
            base = Model(folder, cuda, device_embedding=True)
            self.cfg = base.cfg
            self.embedding = DeviceVocabulary(self.dev, base.embedding)
            self.head = DeviceVocabulary(self.dev, base.head)
            self.layers = [DeviceLayer(self.dev, layer) for layer in base.layers]
            self.embed_norm = self.dev.weight(base.steering["model.embed_tokens.embed_norm.weight"])
            self.final_query = self.dev.weight(base.steering["model.final_input.query"])
            self.final_norm = self.dev.weight(base.steering["model.norm.weight"])
            self.message_queries = [self.dev.weight(value) for value in base.steering["model.message_input.query"]]
        except Exception:
            self.close()
            raise

    def close(self):
        self.dev.close()

    def session(self, context):
        return DeviceSession(self, context)


@dataclass
class DeviceCache:
    capacity: int
    k: U64
    v: U64
    conv: list


class DeviceSession:
    def __init__(self, model, context):
        if context <= 0:
            raise ValueError("Context size must be positive")
        self.model, self.context = model, context
        self.dev, self.cuda = model.dev, model.dev.cuda
        self.position = 0
        self.buffers, self.caches = [], {}
        cfg = model.cfg
        h, e, d, top = cfg.hidden_size, cfg.n_routed_experts, cfg.intermediate_size, cfg.num_experts_per_tok
        q_width = cfg.num_attention_heads * cfg.head_dim
        kv_width = cfg.num_key_value_heads * cfg.head_dim
        biggest_bucket = max(slab.rows for _, _, slab, _ in model.head.buckets)
        scratch_size = (biggest_bucket + 128 * (h + d + q_width + kv_width) +
                        4 * top * (h + d) + e)
        try:
            self.scratch = Scratch(self.alloc(scratch_size), scratch_size)
            self.scores = self.alloc(cfg.num_attention_heads * context)
            self.residuals = self.alloc((cfg.total_visits + 1) * h)
            self.logits = self.alloc(model.head.size)
            self.argmax_groups = min(128, tiles(model.head.size, 2048))
            self.argmax_values = self.alloc(self.argmax_groups + 1)
            self.argmax_ids = self.alloc(self.argmax_groups)
            self.best = self.alloc(2)
        except Exception:
            self.close()
            raise

    def alloc(self, floats):
        buffer = self.cuda.allocate(4 * floats)
        self.buffers.append(buffer)
        return buffer

    def cache(self, layer, slot):
        if slot not in self.caches:
            cfg = self.model.cfg
            capacity = min(self.context, cfg.sliding_window_size) if layer.sliding else self.context
            suffix = "conv1d.weight" if layer.sliding else "static_weight"
            taps = [
                layer.w[f"self_attn.{key}_proj.sconv.{suffix}"].shape
                for key in ("k", "v")
            ]
            taps += [layer.w[f"post_{name}_convolution.conv1d.weight"].shape for name in ("attention", "mlp")]
            size = capacity * cfg.num_key_value_heads * cfg.head_dim
            self.caches[slot] = DeviceCache(capacity, self.alloc(size), self.alloc(size),
                                            [self.alloc(math.prod(shape)) for shape in taps])
        return self.caches[slot]

    def forward(self, token, logits):
        if self.position >= self.context:
            raise ValueError("Context limit reached")
        model, scratch = self.model, self.scratch
        cfg, dev = model.cfg, model.dev
        width = cfg.hidden_size
        scratch.mark = 0
        h = offset(self.residuals, cfg.total_visits * width)
        embedding = model.embedding.embed(token, scratch)
        dev.normalize(embedding, h, width, weight=model.embed_norm, eps=cfg.rms_norm_eps)
        blocks = []

        def visit(index, slot, hidden, messages=None):
            layer = model.layers[index]
            out = offset(self.residuals, slot * width)
            layer(hidden, blocks, self.position, self.cache(layer, slot), messages, self.scores, scratch, out)
            if layer.boundary:
                blocks.append(hidden)
            scratch.mark = 0
            return out

        for i in range(cfg.loop_start):
            h = visit(i, i, h)
        start_h, start_blocks = h, blocks.copy()
        core = range(cfg.loop_start, cfg.loop_end)
        for i in core:
            h = visit(i, i, h)
        for pass_index in range(1, cfg.loop_passes):
            previous_h = h
            previous_blocks = blocks[len(start_blocks) + 1:]
            h, blocks = start_h, start_blocks.copy()
            for j, i in enumerate(core):
                messages = block_attention(dev, model.message_queries[j], previous_blocks, previous_h, width, scratch)
                slot = cfg.num_hidden_layers + (pass_index - 1) * len(core) + j
                h = visit(i, slot, h, messages)
        for i in range(cfg.loop_end, cfg.num_hidden_layers):
            h = visit(i, i, h)
        if logits:
            final, = block_attention(dev, model.final_query, blocks, h, width, scratch)
            dev.normalize(final, final, width, weight=model.final_norm, eps=cfg.rms_norm_eps)
            model.head.logits(final, self.logits, cfg.final_logit_softcapping, scratch)
        self.position += 1

    def step(self, token, logits=True):
        """Diagnostic full-logit readback. Use predict for ordinary generation."""
        self.forward(token, logits)
        return self.cuda.read(self.logits, np.empty(self.model.head.size, np.float32)) if logits else None

    def predict(self, token):
        self.forward(token, logits=True)
        groups = self.argmax_groups
        invalid = offset(self.best, 1)
        self.cuda.call("cuMemsetD32Async", invalid, U32(0), SIZE(1), PTR())
        self.dev.launch("argmax", (self.logits, NULL, self.argmax_values, self.argmax_ids, invalid,
                                   U32(self.model.head.size), U32(groups)), (groups, 1, 1))
        self.dev.launch("argmax", (self.argmax_values, self.argmax_ids, offset(self.argmax_values, groups),
                                   self.best, invalid, U32(groups), U32(1)), (1, 1, 1))
        next_token, invalid = self.cuda.read(self.best, np.empty(2, np.int32))
        if invalid or next_token < 0:
            raise FloatingPointError("Non-finite output logits")
        return int(next_token)

    def close(self):
        for buffer in reversed(self.buffers):
            self.cuda.call("cuMemFree_v2", buffer)
        self.buffers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
