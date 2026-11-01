"""Writes small trellis matrices and tables through the exporter, with the matrices they must rebuild, for test_trellis.py (run with the quantizer's Python).

    python runtime_mlx/test_trellis_gen.py <out dir>
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent.parent))
import lamq_export
import quantize as Q
from opgraph import codes as C

Q.DEV = "cpu"  # the tables round in float64
Q.TRELLIS_ROWS = (16, 8)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
g = torch.Generator().manual_seed(0)
E = lamq_export.Exporter()
refs = {}
for i, (m, n, k) in enumerate(((1536, 1536, 6), (1536, 6144, 5), (6144, 1536, 6), (512, 1536, 4), (256, 12288, 3), (1536, 1536, 2))):
    W = torch.randn(m, n, generator=g, dtype=torch.float64) * (0.01 + 0.05 * torch.rand(m, 1, generator=g, dtype=torch.float64))
    H = None if n > 2048 else torch.eye(n, dtype=torch.float64) + 0.1
    for side in ("one", "two"):
        name = f"m{i}.{side}"
        net = C.ldlq_words([W], H, C.RowTrellisCode(12, k), [100 + i], in_seed=1000 + i if side == "one" else None, out_rot=side == "two")[0]
        Wq = net.matrix(dtype=torch.float64)
        assert Wq.shape == W.shape, Wq.shape
        E.matrix(name, "tr", k, net, 100 + i, 1000 + i if side == "one" else None)
        refs[name + ".ref"] = Wq.float()
V, d = 1024, 1536
for name, block, buckets in (("table", None, [(256, "fp16"), (256, "lm6"), (256, "tr6"), (256, "tr5")]), ("slices", 256, [(256, "lm4"), (256, "tr5"), (512, "tr3")])):
    dd = 256 * 3 if block else d
    W = torch.randn(V, dd, generator=g, dtype=torch.float64) * 0.03
    order = torch.randperm(V, generator=g)
    lo, Wq = 0, torch.zeros(V, dd)
    Q.EXPORT = E
    for k, (size, p) in enumerate(buckets):
        rows = order[lo : lo + size]
        rec = []
        net = Q.embed_word(W[rows], None, p, 7000 + k, block, rec=rec)
        Wq[rows] = net.matrix(dtype=torch.float64).float()
        E.table_bucket(name, k, rows, p, rec, (V, dd), block)
        lo += size
    refs[name + ".ref"] = Wq
save_file({k: v.contiguous() for k, v in {**E.tensors, **refs}.items()}, str(out / "model.safetensors"))
(out / "lamq.json").write_text(json.dumps(E.meta, indent=1))
print("wrote", len(E.tensors), "tensors")
