# Inkling 2 Mini CUDA runtime

This runtime reads the original LookAtMyQuant Inkling 2 Mini text packs. The CUDA path keeps packed weights, activations, residuals and all 48 visit caches on the device. It includes packed embedding decode, Hadamard rotations, clustered residual reads, cross-pass message gates, causal convolution, grouped-query attention with relative bias and sinks, and top-k expert routing. Each routed projection dispatches all selected experts in one kernel regardless of which packed formats they use.

The NVIDIA driver JIT-compiles the PTX. Python loads the checkpoint, handles tokenization and schedules kernels. For text generation, each sampled step copies an eight-byte token and status pair to the host. The `hybrid` backend provides the earlier NumPy decoder with CUDA projections for comparisons.

## Run

Use Python 3.10 or newer and a recent NVIDIA driver on Ampere or newer. From `runtime_cuda` in the checkout:

```powershell
python -m pip install numpy huggingface-hub transformers
python -m inkling_cuda download .\models\inky-5.4
python -m inkling_cuda generate .\models\inky-5.4 --prompt "What is 17 times 23? Give just the answer." --max-new-tokens 32
```

`generate` accepts an existing pack directory, `--tokens` for raw IDs, `--json` for timing and token records, and `--backend hybrid` for comparison. The default download pins the 5.4 GB release. `--size 8.9GB` selects the larger release.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The tests exercise all nine packed code formats, CUDA Hadamard transforms at release widths, cache rotation, and GPU logits and greedy decisions against the NumPy reference on a six-layer looped fixture. The tests also check that prefill reads no GPU data back and ordinary generation reads only the eight-byte result.

On an RTX 5080 Laptop GPU, a synthetic fixture with 24 physical layers, 48 visits, 128 experts per layer, and a 200,058-token head decoded at 13.2–13.3 tokens/s across two runs. The previous hybrid path decoded at 8.0 tokens/s on the same inputs. Both CUDA runs produced the same token IDs. The hybrid path produced different IDs on that fixture. Its first 18 layer visits for token ID 5 stayed within `1.3e-5` absolute error of CUDA, and the paths diverged during the message-gated visits. See `synthetic-smoke.json` for the benchmark records.

The implemented scope is single-sequence greedy text generation with serial prefill. Batched prefill and DSpark speculative decoding remain open. The released 5.4 GB and 8.9 GB weights still need an end-to-end MLX comparison. The development host could run CUDA but its managed shell could not resolve Hugging Face.

Decoder equations and test fixtures follow the existing MLX runtime under the upstream MIT license.
