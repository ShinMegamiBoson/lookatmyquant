"""Experimental eight-slot verifier with one alternative of two tokens."""

from itertools import pairwise
from time import perf_counter

import mlx.core as mx

if __package__ == "strategies":
    import drafter_q35 as D
else:
    from .. import drafter_q35 as D

from . import tree_attention_q35, tree_gdn_q35
from .tree_argmax2 import propose


def accepted(ids, own, fork, eos):
    """Walk verified edges and return the cache slots and emitted tokens."""
    slots, emitted = [0], []
    while True:
        slot = slots[-1]
        token = own[slot]
        emitted.append(token)
        if token in eos:
            break
        if fork >= 0 and slot == fork and token == ids[6]:
            child = 6
        elif slot == 7 or (fork >= 0 and slot == 5):
            break
        else:
            child = slot + 1
        if ids[child] != token:
            break
        slots.append(child)
    return slots, emitted


def check(block, ids, cache, fork):
    linear, attention = block._linear, block._attention
    block._linear = lambda rec, xn, c, _, carry: tree_gdn_q35.linear(
        block, rec, xn, c, carry, fork
    )
    block._attention = lambda a, xn, c, _: tree_attention_q35.attention(
        block, a, xn, c, fork
    )
    try:
        return block.check(ids, cache, top1=True)
    finally:
        block._linear, block._attention = linear, attention


def commit(block, cache, slots):
    kept = len(slots)
    if slots[-1] == kept - 1:
        block.commit(cache, kept)
        return slice(0, kept)

    _, pending = block.pending
    block.pending, block.carry = None, []
    select = mx.array(slots)
    for rec, p, c in zip(block.recs, pending, cache):
        if rec["linear"]:
            c.advance(kept)
            carry = {key: p[key][select] for key in ("y", "pre", "ab")}
            carry["where"] = p["where"]
            block.carry.append((carry, kept))
        else:
            tree_attention_q35.keep(c, kept, slots)
            block.carry.append(None)
    return select


def generate(
    model,
    block,
    drafter,
    prompt,
    max_tokens,
    size,
    eos,
):
    if size != 8 or drafter.path:
        raise ValueError("the tree experiment uses a greedy eight-slot drafter")
    eos, cache = set(eos), model.make_cache()
    start = perf_counter()
    first, got = D._prefill(model, block, prompt, cache, drafter.taps)
    prefilling = perf_counter()
    drafter.reset()
    first_id = int(first)
    if max_tokens > 1 and first_id not in eos:
        drafter.context([mx.concatenate(got[i], axis=1)[0] for i in drafter.taps])
        mx.eval([c.state for c in drafter.cache])
    drafted = perf_counter()
    out, anchor, rounds, times = [first_id], first, [], [drafted]
    forks = used_forks = 0
    drafter.torn = False
    while len(out) < max_tokens and out[-1] not in eos:
        states, _ = drafter.draft(anchor, size)
        ids, fork = propose(drafter, states, anchor)
        if D.OVERLAP:
            mx.async_eval(ids, fork)
        own, taps = check(block, ids, cache, fork)
        mx.eval(ids, own, fork)
        parent = int(fork)
        checked_ids, predictions = ids.tolist(), own.tolist()
        slots, emitted = accepted(checked_ids, predictions, parent, eos)
        forks += parent >= 0
        used_forks += parent >= 0 and slots[-1] >= 6
        out.extend(emitted)
        rounds.append(len(slots))
        if len(out) >= max_tokens or out[-1] in eos:
            times.append(perf_counter())
            break
        selected = commit(block, cache, slots)
        drafter.context([taps[i][selected] for i in drafter.taps])
        if D.OVERLAP:
            mx.async_eval([x for c in drafter.cache for x in (c.keys, c.values)])
        anchor = own[slots[-1]]
        drafter.torn = len(slots) < size
        times.append(perf_counter())
    elapsed = perf_counter() - drafted
    out = out[:max_tokens]
    return out, {
        "rounds": len(rounds),
        "kept": rounds,
        "round_ms": [round((y - x) * 1e3, 2) for x, y in pairwise(times)],
        "prompt_s": prefilling - start,
        "drafter_prompt_s": drafted - prefilling,
        "rounds_s": elapsed,
        "tokens": len(out),
        "tokens_per_check": (len(out) - 1) / max(1, len(rounds)),
        "forks": forks,
        "used_forks": used_forks,
    }
