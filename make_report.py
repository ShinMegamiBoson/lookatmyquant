"""Build results/COMPLETENESS.md from the completeness JSON files."""

import json
import re
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
ORDER = ["RTN w4", "RTN w3", "GPTQ w4", "GPTQ w3 g64", "AWQ w4", "AWQ w3", "SVD keep", "SVD-LLM", "Magnitude", "Wanda", "SparseGPT", "SliceGPT", "GPTQ w3 g64, official"]


def load():
    rows = []
    for f in ("completeness_rest.json", "completeness_awq.json"):
        p = R / f
        if p.exists():
            rows += json.loads(p.read_text())
    key = lambda r: next((i for i, o in enumerate(ORDER) if r["method"].startswith(o)), 99) + (0.5 if "official" in r["method"] else 0)
    return sorted(rows, key=key)


def original_ppl():
    for f in ("completeness_rest.log", "completeness_awq.log"):
        p = R / f
        if p.exists():
            m = re.search(r"original: ppl ([\d.]+)", p.read_text())
            if m:
                return float(m.group(1))
    return None


def agreement(r):
    a, fin = r.get("arg_agreement"), r.get("final")
    parts = []
    if a is not None:
        parts.append(f"{100 * a:.3f}%")
    if fin:
        parts.append(f"final codes {100 * fin['codes']:.3f}%")
    return ", ".join(parts) if parts else "n/a (no discrete choices)"


def main():
    rows = load()
    lines = [
        "# Completeness check",
        "",
        f"SmolLM2-360M in fp32. Calibration: 64 random 512-token windows of WikiText-2 train. Evaluation: the first 64 "
        f"512-token windows of WikiText-2 test (32,768 tokens). Original perplexity: **{original_ppl():.4f}**.",
        "",
        "For each method, the reference implementation runs layer by layer; at every layer the operator-word version is "
        "applied to the identical layer state and inputs and compared. Then the operator-word version runs end to end "
        "with its own calibration. *Agreement* is the minimum over layers and matrices of the fraction of identical "
        "integer codes, mask entries, or chosen arguments (AWQ's scale ratios and clip bounds). *Rel. weight diff* is the "
        "largest relative Frobenius difference of any weight matrix between the two versions at any layer.",
        "",
        "| Method | Operator word | Agreement | Rel. weight diff | PPL reference | PPL operator word | KL vs original (ref / word) | Max logit gap | Bits / weight |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        bits = f"{r['bits_per_weight']:.2f}"
        if "whole_model_params_ratio" in r:
            bits += f" (whole model: {100 * r['whole_model_params_ratio']:.1f}% of params)"
        lines.append(f"| {r['method']} | {r['word']} | {agreement(r)} | {r['max_layer_rel_weight_diff']:.1e} | {r['ppl_ref']:.4f} | "
                     f"{r['ppl_frame']:.4f} | {r['kl_ref']:.4f} / {r['kl_frame']:.4f} | {r['max_logit_gap']:.1e} | {bits} |")
    (R / "COMPLETENESS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
