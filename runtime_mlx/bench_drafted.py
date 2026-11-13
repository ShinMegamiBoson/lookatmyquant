"""Generation with and without the drafter, on the same prompts in the same process: speed, and whether the text is the same.

    python bench_drafted.py PACKED_DIR DRAFTER_DIR [--mlx 4-BIT_DIR] [--size 8] [--tokens 256] [--rounds 3] [--out file.json]

Plain is mlx_lm's own stream_generate on fast_q35's model. Drafted is
drafter_q35.generate: the drafter's time is inside it. Both are greedy with thinking off, and both speeds are mlx_lm's
generation_tps: the tokens generated over the time after the first one. --mlx adds MLX's own 4-bit model on the first prompt each round: the
check that the machine is quiet (about 165 tok/s on the M5 Max).
"""
import argparse
import json
import statistics
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

import drafter_q35

PROMPTS = {
    "essay": "Write a long, detailed essay on the history of the printing press, from Gutenberg to the present day.",
    "word problem": "A train leaves at 9:40 and arrives at 13:05. It stops for 12 minutes on the way and covers 246 km. What is its average speed while moving, in km/h? Show each step.",
    "algebra": "Solve for x and y: 3x + 4y = 31 and 5x - 2y = 17. Show every step, then check the answer.",
    "code": "Write a Python function that merges overlapping intervals in a list of (start, end) pairs, with a docstring and three test cases.",
    "explain": "Explain how a refrigerator works, step by step, to someone who has never studied physics.",
    "chat": "I have three days in Lisbon in November. What should I see, and what should I skip?",
    "summary": "Summarize the causes of the First World War in about 200 words, then list the three most important ones.",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
}


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)


def plain(model, tok, prompt, n):
    tokens, last = [], None
    for r in stream_generate(model, tok, prompt, max_tokens=n, sampler=make_sampler(temp=0.0)):
        tokens.append(r.token)
        last = r
    return tokens, last.generation_tps


def gap_at(model, prompt, tokens, i):
    """Plain decoding's two best logits for token i (after the prompt and tokens[:i]): how close the choice was."""
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)[0, -1]
    for t in tokens[:i]:
        logits = model(mx.array([[t]]), cache=cache)[0, -1]
        mx.eval(logits)
    top = mx.sort(logits)[-2:].tolist()
    return round(top[1] - top[0], 6), round(top[1], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packed")
    ap.add_argument("drafter")
    ap.add_argument("--mlx", default="")
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--mode", choices=("linear", "chat-fork", "copy-wide"), default="linear",
                    help="verifier strategy; use copy-wide for known contiguous copies")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--prompts", default="")
    ap.add_argument("--heldout", default="", help="a JSON list of {source, prompt_len, input_ids}: its prompts in place of PROMPTS")
    ap.add_argument("--take", type=int, default=24, help="how many of --heldout's prompts, spread evenly over the file")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--bits", type=int, default=0, help="MLX quantization of the drafter's large matrices (0: dense)")
    ap.add_argument("--reverse", action="store_true", help="drafted before plain on each prompt")
    ap.add_argument("--select", action="store_true", help="the drafter proposes by path selection")
    ap.add_argument("--dense", action="store_true", help="the drafter's matrices dense at 16 bits (default: its block-only ones as 8-bit rows)")
    ap.add_argument("--full-vocab", action="store_true", help="score the whole vocabulary for proposals as well as verification")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.mode != "linear" and (args.size != 8 or args.select):
        ap.error(f"--mode {args.mode} requires --size 8 and greedy proposals")
    generate = partial(drafter_q35.generate, mode=args.mode)
    model, tok, blk, drafter = drafter_q35.load(
        args.packed, args.drafter, dtype=getattr(mx, args.dtype), bits=args.bits or None,
        select=args.select, int8=not args.dense, full_vocab=args.full_vocab,
    )
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    ref = None
    if args.mlx:
        from mlx_lm.utils import load_model, load_tokenizer
        ref = (load_model(Path(args.mlx), strict=False)[0], load_tokenizer(Path(args.mlx)))
    names = [p for p in (args.prompts.split(",") if args.prompts else PROMPTS)]
    prompts = {name: chat(tok, PROMPTS[name]) for name in names}
    if args.heldout:
        held = json.loads(Path(args.heldout).read_text())
        step = max(1, len(held) // args.take)
        prompts = {f"{h['source'].split('/')[-1][:18]} {i}": h["input_ids"][: h["prompt_len"]] for i, h in list(enumerate(held))[::step][: args.take]}
    first = next(iter(prompts.values()))
    plain(model, tok, first, 16)                                  # kernels compile on first use
    generate(model, blk, drafter, first, 24, size=args.size, eos=eos)
    if ref:
        plain(*ref, chat(ref[1], PROMPTS['essay']), 16)
    res = {name: {"plain tok/s": [], "drafted tok/s": [], "tokens per check": [], "same": [], "tokens": []} for name in prompts}
    quiet = []
    for rnd in range(args.rounds):
        if ref:
            quiet.append(round(plain(*ref, chat(ref[1], PROMPTS['essay']), args.tokens)[1], 1))
        for name, prompt in prompts.items():
            if args.reverse:
                got, st = generate(model, blk, drafter, prompt, args.tokens, size=args.size, eos=eos)
                want, tps = plain(model, tok, prompt, args.tokens)
            else:
                want, tps = plain(model, tok, prompt, args.tokens)
                got, st = generate(model, blk, drafter, prompt, args.tokens, size=args.size, eos=eos)
            n = min(len(want), len(got))
            diff = next((i for i in range(n) if want[i] != got[i]), None if len(want) == len(got) else n)
            r = res[name]
            r["plain tok/s"].append(round(tps, 1))
            r["drafted tok/s"].append(round(st["tokens"] / st["rounds_s"], 1))   # mlx_lm's own count: every token over the time after the first
            r["tokens per check"].append(round(st["tokens_per_check"], 2))
            r.setdefault("ratio", []).append(round(r["drafted tok/s"][-1] / r["plain tok/s"][-1], 3))
            r["same"].append(diff is None)
            r["tokens"].append(len(got))
            r["first difference"] = diff
            r["tokens kept by round"] = st["kept"]
            r["round ms, median"] = round(statistics.median(st["round_ms"]), 2) if st["round_ms"] else None
            if diff is not None and "gap at the first difference" not in r:
                r["gap at the first difference"], r["logit there"] = gap_at(model, prompt, want, diff)
            r["drafter prompt s"] = round(st["drafter_prompt_s"], 4)
            r["prompt tokens"] = len(prompt)
        line = f"round {rnd + 1}:" + (f" MLX 4-bit {quiet[-1]}" if ref else "")
        print(line + " | " + "  ".join(f"{name} {r['plain tok/s'][-1]:.0f}->{r['drafted tok/s'][-1]:.0f}" for name, r in res.items()), flush=True)
    med = statistics.median
    width = max(14, max(len(n) for n in res))
    print(f"\n{'prompt':{width}s} {'plain':>7s} {'drafted':>8s} {'ratio':>6s} {'tokens/check':>13s} {'tokens':>7s}  same text")
    for name, r in res.items():
        r["median plain"], r["median drafted"] = med(r["plain tok/s"]), med(r["drafted tok/s"])
        print(f"{name:{width}s} {r['median plain']:7.1f} {r['median drafted']:8.1f} {r['median drafted'] / r['median plain']:6.2f} {med(r['tokens per check']):13.2f} {r['tokens'][-1]:7d}  "
              f"{'yes' if all(r['same']) else 'NO, from token ' + str(r['first difference']) + ': plain decoding had its two best logits ' + str(r.get('gap at the first difference')) + ' apart at ' + str(r.get('logit there'))}")
    allp, alld = [r["median plain"] for r in res.values()], [r["median drafted"] for r in res.values()]
    total = sum(r["tokens"][-1] for r in res.values())
    plain_s, drafted_s = sum(r["tokens"][-1] / r["median plain"] for r in res.values()), sum(r["tokens"][-1] / r["median drafted"] for r in res.values())
    print(f"{'all prompts':{width}s} {total / plain_s:7.1f} {total / drafted_s:8.1f} {plain_s / drafted_s:6.2f}   (tokens over summed time)" + (f"   MLX 4-bit by round: {quiet}" if ref else ""))
    out = {"device": mx.device_info(), "size": args.size, "mode": args.mode, "tokens": args.tokens, "rounds": args.rounds, "drafter dtype": args.dtype, "drafter bits": args.bits, "heldout": args.heldout, "reverse": args.reverse, "path selection": args.select, "drafter 8-bit rows": not args.dense, "proposal vocabulary": blk.table.V if drafter.proposal_head is None else int(drafter.proposal_head.ids.size),
           "MLX 4-bit tok/s by round": quiet, "results": res, "all prompts": {"plain tok/s": round(total / plain_s, 1), "drafted tok/s": round(total / drafted_s, 1),
                                                                             "tokens per check": round(sum(r["tokens"][-1] - 1 for r in res.values()) / sum((r["tokens"][-1] - 1) / med(r["tokens per check"]) for r in res.values()), 2),
                                                                             "same text": sum(all(r["same"]) for r in res.values()), "prompts": len(res)}}
    print(f"tokens per check over all prompts {out['all prompts']['tokens per check']}; same text on {out['all prompts']['same text']} of {len(res)} prompts")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
