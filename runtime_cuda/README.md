# Inkling 2 Mini CUDA runtime

This runtime reads the original LookAtMyQuant Inkling 2 Mini text packs. The CUDA path keeps packed weights, activations, residuals and all 48 visit caches on the device. It includes packed embedding decode, Hadamard rotations, clustered residual reads, cross-pass message gates, causal convolution, grouped-query attention with relative bias and sinks, and top-k expert routing. Each routed projection dispatches all selected experts in one kernel regardless of which packed formats they use.

The NVIDIA driver JIT-compiles the PTX. Python loads the checkpoint, handles tokenization and schedules kernels. Prefill processes up to 16 consecutive positions per layer. Each greedy decode step copies an eight-byte token and status pair to the host. The `hybrid` backend provides the NumPy decoder with CUDA projections for comparisons.

## Run

Use Python 3.10 or newer and a recent NVIDIA driver on Ampere or newer. From `runtime_cuda` in the checkout:

```powershell
python -m pip install numpy huggingface-hub transformers jinja2
python -m inkling_cuda download .\models\inky-5.4
python -m inkling_cuda generate .\models\inky-5.4 --prompt "What is the capital of France?" --max-new-tokens 64
```

`generate` accepts an existing pack directory, `--tokens` for raw IDs, `--json` for timing and token records, and `--backend hybrid` for comparison. Downloads pin the selected release. `--size 8.9GB` selects the larger pack.

### DSpark speculative decoding

Install a [CUDA-enabled PyTorch build](https://pytorch.org/get-started/locally/) and `safetensors` to use the optional drafter.

~~~powershell
python -m pip install safetensors
python -m inkling_cuda download-drafter .\models\inky-drafter
python -m inkling_cuda generate .\models\inky-5.4 --drafter .\models\inky-drafter --draft-block-size 4 --prompt "What is the capital of France?" --max-new-tokens 64
~~~

The drafter opens the published 4-bit pack and expands its matrices on the GPU for BF16 GEMMs. It uses four target activations per position and scores its proposals over the first five head buckets. The target verifies the entire block against the full packed head and commits only the matching input prefix. DSpark reports draft acceptance with the usual timing and token data.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The tests cover all nine packed code formats, release-width Hadamard transforms, BF16 message-gate rounding, and GPU logits against the NumPy reference. Batched prefill and speculative verification are checked against serial decoding through local attention and convolution cache wraparound.

`synthetic-smoke.json` records the original serial decoder benchmark. In that fixture, small differences at BF16 message-gate boundaries changed the experts selected later in the loop.

The implemented scope is single-sequence greedy text generation. Both published packs were loaded and run on an RTX 5080 Laptop GPU. End-to-end comparison with the Apple Silicon Metal runtime remains open.

Decoder equations and test fixtures follow the existing MLX runtime under the upstream MIT license.
