"""Device-resident decoder for LookAtMyQuant's Inkling text checkpoints."""
import math
from dataclasses import dataclass

import numpy as np

from .device_ptx import source
from .driver import F32, PTR, SIZE, U32, U64
from .model import Model
from .packed import EXPERT_FORMATS, hadamard_factors


NULL = U64()
THREADS = 128
PREFILL_CHUNK = 16


def tiles(size, width=THREADS):
    return (size + width - 1) // width


@dataclass(frozen=True)
class Batch:
    ptr: U64
    stride: int
    count: int

    def row(self, index):
        return U64(self.ptr.value + index * self.stride)


def offset(pointer, floats):
    if isinstance(pointer, Batch):
        return Batch(offset(pointer.ptr, floats), pointer.stride, pointer.count)
    return U64(pointer.value + 4 * floats)


def row(pointer, index):
    return pointer.row(index) if isinstance(pointer, Batch) else pointer


def batch_params(pointer):
    return {"batch": pointer.count, "stride": pointer.stride} if isinstance(pointer, Batch) else {}


@dataclass(frozen=True)
class Weight:
    ptr: U64
    shape: tuple


class Device:
    def __init__(self, cuda):
        self.cuda = cuda
        self.module = cuda.load_module(source(cuda.compute_capability()))
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
        batch = args[0]
        stride = 0
        if isinstance(batch, Batch):
            args = tuple(row(arg, 0) for arg in args)
            grid = (*grid[:2], batch.count)
            stride = batch.stride
        if name not in self.functions:
            self.functions[name] = self.cuda.module_function(self.module, name)
        self.cuda.launch(self.functions[name], grid, THREADS, *args, U32(stride), shared=shared)

    def vector(self, name, size, *args):
        self.launch(name, args, (tiles(size), 1, 1))

    def normalize(self, x, out, width, groups=1, weight=None, eps=1e-6, clip=None, factor=1):
        self.launch(
            "rms_norm" if clip is None else "clip_norm",
            (x, weight.ptr if weight else NULL, out, U32(width),
             F32(eps if clip is None else clip), F32(factor)),
            (groups, 1, 1),
        )

    def dense(self, x, weight, out, kernel="dense_matvec"):
        rows, width = weight.shape
        self.launch(kernel, (x, weight.ptr, out, U32(rows), U32(width)), (tiles(rows, 4), 1, 1))

    def rotate(self, x, signs, out, inverse=False, rows=1):
        width, = signs.shape
        if width not in self.bases:
            _, base = hadamard_factors(width)
            self.bases[width] = (base.shape[0], self.weight(base).ptr) if base.shape[1] > 1 else (1, NULL)
        base, table = self.bases[width]
        if base == 1:
            self.launch("hadamard", (x, signs.ptr, table, out, U32(width), U32(base),
                                    U32(inverse), F32(width ** -0.5)), (rows, 1, 1), shared=4 * width)
            return
        # Spread the columns over blocks of THREADS threads: each block mixes `cols` columns, then transforms them.
        sylvester = width // base
        cols = max(1, THREADS // sylvester)
        self.launch("hadamard_paley", (x, signs.ptr, table, out, U32(width), U32(base), U32(inverse),
                                       U32(sylvester), U32(cols), F32(width ** -0.5)),
                    (tiles(base, cols), rows, 1), shared=4 * sylvester * cols)


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
        dev.cuda.packed_project(slab, row(rotated, 0), dev.zero_index, row(projected, 0), 1, 1,
                                **batch_params(rotated))
        if self.out_signs is not None:
            dev.rotate(projected, self.out_signs, out, inverse=True)
        scratch.mark = mark
        return out


class DeviceExperts:
    def __init__(self, device, experts):
        self.device = device
        self.signs = device.weight(experts.signs)
        self.mapping = device.weight(experts.mapping)
        self.rows, self.width = experts.rows, experts.width
        self.groups, self.group_rows = experts.groups, experts.group_rows
        self.expert_count = experts.mapping.shape[0]
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
        self.device.cuda.routed_project(row(rotated, 0), row(selected, 0), row(out, 0), slots, x_div,
                                        self.rows, self.width, self.descriptors.ptr, self.mapping.ptr,
                                        groups=self.groups, group_rows=self.group_rows,
                                        expert_count=self.expert_count, **batch_params(rotated))
        scratch.mark = mark
        return out


class DeviceVocabulary:
    def __init__(self, device, vocab):
        self.device = device
        self.buckets = [(lo, hi, slab, device.weight(signs)) for lo, hi, slab, signs in vocab.buckets]
        self.order = device.weight(vocab.order)
        self.inverse = vocab.inverse
        self.size = vocab.order.size

    def embed(self, token, scratch, out=None):
        if not 0 <= token < self.size:
            raise ValueError(f"Token {token} is outside the embedding vocabulary")
        position = int(self.inverse[token])
        for lo, hi, slab, signs in self.buckets:
            if lo <= position < hi:
                if out is None:
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
            projected = scratch.take(slab.rows)
            self.project_bucket(x, slab, signs, projected, scratch)
            dev.vector("head_scatter", hi - lo, projected, offset(self.order.ptr, lo), out, U32(hi - lo), F32(cap))
            scratch.mark = mark

    def project_bucket(self, x, slab, signs, out, scratch):
        dev = self.device
        mark = scratch.mark
        rotated = scratch.take(slab.width)
        dev.rotate(x, signs, rotated)
        dev.cuda.packed_project(slab, row(rotated, 0), dev.zero_index, row(out, 0), 1, 1,
                                **batch_params(rotated))
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

    def row(self, index):
        scratch = Scratch(row(self.pointer, index), self.capacity)
        scratch.mark = self.mark
        return scratch


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
        divisor = 2 * taps + 1 if row(dynamic, 0).value else taps + 1
        self.dev.vector("causal_conv", width, x, weight.ptr, dynamic, history, out,
                        U32(width), U32(taps), U32(position), U32(self.cfg.head_dim),
                        U32(taps + PREFILL_CHUNK - 1), F32(divisor ** -0.5))
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
        dev.dense(pair, down, rank, kernel="message_down")
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
        q_heads, kv_heads, dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        dev.normalize(k, k, dim, kv_heads, clip=cfg.qk_l2_clip_threshold)
        dev.normalize(v, v, dim, kv_heads, eps=cfg.rms_norm_eps)
        sink = scratch.take(q_heads)
        dev.dense(q_in, w["self_attn.sink_proj.weight"], sink)
        attended = scratch.take(q_heads * dim)
        count = q.count if isinstance(q, Batch) else 1
        if self.sliding or position + count <= cfg.log_scaling_n_floor:
            self.clip_queries(q, rel, 1.0)
        else:
            for i in range(count):
                factor = 1 + cfg.log_scaling_alpha * math.log(max((position + i + 1) / cfg.log_scaling_n_floor, 1))
                self.clip_queries(row(q, i), row(rel, i), factor)
        self.attend(q, k, v, rel, sink, attended, position, count, cache, scores)
        gate = self.linear["o_proj.gate_proj"](q_in, scratch)
        dev.vector("attention_gate", q_heads * dim, attended, gate, attended, U32(q_heads * dim))
        projected = self.linear["o_proj.linear"](attended, scratch)
        return self.conv(projected, w["post_attention_convolution.conv1d.weight"], cache.conv[2], position, scratch)

    def clip_queries(self, q, rel, factor):
        cfg, dev, w = self.cfg, self.dev, self.w
        heads = cfg.num_attention_heads
        dev.normalize(q, q, cfg.head_dim, heads, w["self_attn.q_proj.norm.scale"],
                      clip=cfg.qk_l2_clip_threshold, factor=factor)
        dev.normalize(rel, rel, cfg.d_rel, heads, w["self_attn.relative_position_bias.norm.scale"],
                      clip=cfg.rel_l2_clip_threshold, factor=factor)

    def attend(self, q, k, v, rel, sink, out, position, batch_count, cache, scores):
        cfg, dev = self.cfg, self.dev
        q_heads, kv_heads, dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        kv_width = kv_heads * dim
        # The ring holds the live window plus the rest of this chunk. Every
        # position can read its causal window after the batched cache write.
        dev.vector("kv_store", kv_width, k, v, cache.k, cache.v, U32(kv_width), U32(dim),
                   U32(cache.capacity), U32(position))
        max_count = min(position + batch_count, cache.window)
        repeat = q_heads // kv_heads
        dev.launch("attention_scores", (q, cache.k, rel, self.w["self_attn.relative_position_bias.proj"].ptr,
                                       scores, U32(cache.window), U32(dim), U32(repeat), U32(position),
                                       U32(cache.capacity), U32(cfg.d_rel), U32(cfg.rel_extent), F32(dim ** -0.5)),
                   (tiles(max_count, 4), q_heads, 1))
        dev.launch("attention_mix", (scores, cache.v, sink, out, U32(cache.window), U32(dim),
                                     U32(repeat), U32(position), U32(cache.capacity)),
                   (q_heads, 1, 1))

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
    window: int
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
        scratch_size = (biggest_bucket + model.head.size + 128 * (h + d + q_width + kv_width) +
                        4 * top * (h + d) + e)
        self.residual_size = (cfg.total_visits + 1) * h
        self.score_size = cfg.num_attention_heads * context
        self.prefill_buffer = None
        try:
            self.scratch = Scratch(self.alloc(scratch_size), scratch_size)
            self.scores = self.alloc(self.score_size)
            self.residuals = self.alloc(self.residual_size)
            self.logits = self.alloc(model.head.size)
            self.argmax_groups = min(128, tiles(model.head.size, 2048))
            self.argmax_values = self.alloc(self.argmax_groups + 1)
            self.argmax_ids = self.alloc(self.argmax_groups)
            self.best = self.alloc(2 * PREFILL_CHUNK)
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
            window = min(self.context, cfg.sliding_window_size) if layer.sliding else self.context
            # A speculative block can overwrite at most this many expired ring slots
            # before it commits a prefix. Keep those slots readable on rollback.
            capacity = min(self.context, window + PREFILL_CHUNK - 1)
            suffix = "conv1d.weight" if layer.sliding else "static_weight"
            taps = [
                layer.w[f"self_attn.{key}_proj.sconv.{suffix}"].shape
                for key in ("k", "v")
            ]
            taps += [layer.w[f"post_{name}_convolution.conv1d.weight"].shape for name in ("attention", "mlp")]
            size = capacity * cfg.num_key_value_heads * cfg.head_dim
            self.caches[slot] = DeviceCache(capacity, window, self.alloc(size), self.alloc(size),
                                            [self.alloc((count + PREFILL_CHUNK - 1) * width)
                                             for count, width in taps])
        return self.caches[slot]

    def forward(self, token, logits):
        self.forward_block((token,), logits)

    def prefill(self, tokens):
        tokens = tuple(tokens)
        self.validate_tokens(tokens)
        for i in range(0, len(tokens), PREFILL_CHUNK):
            self.forward_block(tokens[i:i + PREFILL_CHUNK], logits=False)

    def validate_tokens(self, tokens):
        if self.position + len(tokens) > self.context:
            raise ValueError("Context limit reached")
        for token in tokens:
            if not 0 <= token < self.model.embedding.size:
                raise ValueError(f"Token {token} is outside the embedding vocabulary")

    def forward_block(self, tokens, logits, *, all_logits=False, tap_visits=()):
        """Process consecutive positions layer by layer and optionally score the last."""
        self.validate_tokens(tokens)
        count = len(tokens)
        if not 1 <= count <= PREFILL_CHUNK:
            raise ValueError(f"Expected 1 to {PREFILL_CHUNK} tokens")
        model, scratch = self.model, self.scratch
        cfg, dev = model.cfg, model.dev
        width = cfg.hidden_size
        residuals = self.residuals
        attention_scores = self.scores
        if count > 1:
            # All activation pointers in a batched launch use the same stride.
            residual_end = tiles(self.residual_size, 32) * 32
            score_start = residual_end + tiles(scratch.capacity, 32) * 32
            stride = tiles(score_start + self.score_size, 32) * 32
            if self.prefill_buffer is None:
                self.prefill_buffer = self.alloc(PREFILL_CHUNK * stride)
            residuals = Batch(self.prefill_buffer, 4 * stride, count)
            scratch = Scratch(offset(residuals, residual_end), scratch.capacity)
            attention_scores = offset(residuals, score_start)
        scratch.mark = 0
        h = offset(residuals, cfg.total_visits * width)
        embedding = scratch.take(width)
        for i, token in enumerate(tokens):
            model.embedding.embed(token, scratch.row(i), out=row(embedding, i))
        dev.normalize(embedding, h, width, weight=model.embed_norm, eps=cfg.rms_norm_eps)
        blocks = []
        scratch.mark = 0
        logical_visit = 0
        taps = {}

        def visit(index, slot, hidden, messages=None):
            nonlocal logical_visit
            if logical_visit in tap_visits:
                taps[logical_visit] = hidden
            logical_visit += 1
            layer = model.layers[index]
            out = offset(residuals, slot * width)
            layer(hidden, blocks, self.position, self.cache(layer, slot), messages, attention_scores, scratch, out)
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
        scores = self.logits
        if logits or all_logits or logical_visit in tap_visits:
            if all_logits or logical_visit in tap_visits:
                final, = block_attention(dev, model.final_query, blocks, h, width, scratch)
                if logical_visit in tap_visits:
                    taps[logical_visit] = final
            else:
                last = count - 1
                scratch = scratch.row(last)
                final, = block_attention(dev, model.final_query, [row(block, last) for block in blocks],
                                         row(h, last), width, scratch)
            dev.normalize(final, final, width, weight=model.final_norm, eps=cfg.rms_norm_eps)
            if all_logits:
                scores = scratch.take(model.head.size)
                model.head.logits(final, scores, cfg.final_logit_softcapping, scratch)
            elif logits:
                last = count - 1 if isinstance(final, Batch) else 0
                model.head.logits(row(final, last), scores, cfg.final_logit_softcapping, scratch.row(last))
        self.position += count
        return scores, taps

    def step(self, token, logits=True):
        """Diagnostic full-logit readback. Use predict for ordinary generation."""
        self.forward(token, logits)
        return self.cuda.read(self.logits, np.empty(self.model.head.size, np.float32)) if logits else None

    def predict(self, token, *, blocked_token=None, sampler=None):
        predictions, _ = self.predict_block((token,), blocked_token=blocked_token, sampler=sampler)
        return int(predictions[0])

    def validate_blocked_token(self, blocked_token):
        if blocked_token is not None and not 0 <= blocked_token < self.model.head.size:
            raise ValueError("Blocked token is outside the output vocabulary")

    def predict_block(self, tokens, *, all_tokens=False, tap_visits=(), blocked_token=None, sampler=None):
        self.validate_blocked_token(blocked_token)
        scores, taps = self.forward_block(tokens, logits=True, all_logits=all_tokens, tap_visits=tap_visits)
        count = len(tokens) if all_tokens else 1
        if sampler is not None:
            start = self.position - count
            return sampler([row(scores, i) for i in range(count)], start, blocked_token), taps
        for i in range(count):
            self.sample(row(scores, i), offset(self.best, 2 * i), blocked_token)
        return self.read_predictions(count), taps

    def sample(self, logits, result, blocked_token=None):
        if blocked_token is not None:
            # The argmax kernel treats non-finite logits as errors.
            self.cuda.call("cuMemsetD32Async", offset(logits, blocked_token),
                           U32(0xff7fffff), SIZE(1), PTR())
        groups = self.argmax_groups
        invalid = offset(result, 1)
        self.cuda.call("cuMemsetD32Async", invalid, U32(0), SIZE(1), PTR())
        self.dev.launch("argmax", (logits, NULL, self.argmax_values, self.argmax_ids, invalid,
                                   U32(self.model.head.size), U32(groups)), (groups, 1, 1))
        self.dev.launch("argmax", (self.argmax_values, self.argmax_ids, offset(self.argmax_values, groups),
                                   result, invalid, U32(groups), U32(1)), (1, 1, 1))

    def read_predictions(self, count):
        predictions = self.cuda.read(self.best, np.empty((count, 2), np.int32))
        tokens, invalid = predictions.T
        if invalid.any() or (tokens < 0).any():
            raise FloatingPointError("Non-finite output logits")
        return tokens

    def commit_block_prefix(self, start, count):
        """Discard a suffix of the most recently verified block."""
        if not (0 <= start < self.position and 1 <= count <= self.position - start <= PREFILL_CHUNK):
            raise ValueError("Invalid speculative block prefix")
        self.position = start + count

    def close(self):
        for buffer in reversed(self.buffers):
            self.cuda.call("cuMemFree_v2", buffer)
        self.buffers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
