"""The MLX drafter against SpecForge's own model on the reference blocks (drafter/reference.py wrote them on
Modal): the context features, the final states, the states before the last injection, and the proposed tokens.

    python check_drafter.py PACKED_DIR DRAFTER_DIR [float32|float16] [int8 (default) | dense | bits]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
import numpy as np

import block_q35 as B
import drafter_q35 as Dq
import fast_q35 as F


def main():
    packed, folder = sys.argv[1], Path(sys.argv[2])
    dtype = {"float32": mx.float32, "float16": mx.float16}[sys.argv[3] if len(sys.argv) > 3 else "float16"]
    bits = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] not in ("int8", "dense") else None
    int8 = len(sys.argv) <= 4 or sys.argv[4] == "int8"
    model, tok = F.load(packed, verify=False)
    blk = B.Block(model, tap_layers=Dq.drafter_taps(folder))
    d = Dq.Drafter(folder, blk, dtype=dtype, bits=bits, int8=int8)
    ref = mx.load(str(folder / "reference" / "ref.safetensors"))
    f32 = lambda a: np.array(a.astype(mx.float32))
    rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
    same = total = same_first = blocks = path_same = path_lead = differs = 0
    n = 0
    while f"{n}.ids" in ref:
        ids, anchors = np.array(ref[f"{n}.ids"]), np.array(ref[f"{n}.anchors"])
        states = ref[f"{n}.states"].astype(mx.float32)
        H = d.H
        taps = [states[:, i * H:(i + 1) * H] for i in range(len(d.taps))]
        ctx = d.context_features(taps)
        print(f"sample {n}: {len(ids)} tokens; context features apart {rel(f32(ctx), f32(ref[f'{n}.context'])):.1e}")
        noise = f32(ref[f"{n}.noise"])                                  # (anchors, 2, H): the anchor's and the mask's embedding
        emb = f32(d.table(mx.array([int(ids[anchors[0]]), d.mask_id])))
        print(f"  embeddings (this file's against the trainer's) apart {rel(emb, noise[0]):.1e}")
        for j, a in enumerate(anchors.tolist()):
            d.reset()
            d.context([t[:a] for t in taps])
            out, pre = d.draft(int(ids[a]), d.trained_block)
            toks = np.array(d.tokens(out, d.trained_block))
            d.path = True
            chosen = np.array(d.tokens(out, d.trained_block, pre, mx.array(int(ids[a]))))
            d.path = False
            r_path = np.array(ref[f"{n}.path"])[j]
            path_same += int((chosen == r_path).sum()); path_lead += int(np.cumprod(chosen == r_path).sum())
            differs += int((chosen != toks).sum())
            r_hidden, r_pre, r_arg = f32(ref[f"{n}.hidden"])[j, 1:], f32(ref[f"{n}.pre_injection"])[j, 1:], np.array(ref[f"{n}.argmax"])[j]
            agree = toks == r_arg
            lead = int(np.cumprod(agree).sum())
            same += int(agree.sum()); total += agree.size; same_first += lead; blocks += 1
            print(f"  anchor {a}: final states apart {rel(f32(out), r_hidden):.1e}, before the last injection {rel(f32(pre), r_pre):.1e}; "
                  f"proposals equal {int(agree.sum())}/{agree.size} (the first {lead})")
        n += 1
    # the mask a sliding layer builds past its window, where it hides nothing: the same states as the built-in causal mask
    a = int(anchors[-1])
    d.reset()
    d.context([t[:a] for t in taps])
    plain, _ = d.draft(int(ids[a]), d.trained_block)
    d.explicit_mask = True
    masked, _ = d.draft(int(ids[a]), d.trained_block)
    d.explicit_mask = False
    print(f"explicit mask against the built-in causal one: states apart {rel(f32(masked), f32(plain)):.1e}")
    print(f"path selection: {path_same}/{total} tokens equal the reference's path; leading equal {path_lead / blocks:.1f} a block; it differs from each position's own best at {differs} places")
    print(f"{dtype}, {'8-bit rows' if int8 else 'bits ' + str(bits)}: {same}/{total} proposed tokens equal the reference's; leading equal {same_first / blocks:.1f} of {d.trained_block - 1} a block")


if __name__ == "__main__":
    main()
