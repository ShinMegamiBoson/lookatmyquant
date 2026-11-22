# Inkling-2-Mini on NVIDIA CUDA

Install with `python -m pip install -e '.[cuda]'` and use the same `lamq` command as on Apple Silicon.

```sh
lamq generate 3.9GB --device cuda --prompt "Explain how a thermos works."
lamq generate 5.4GB --device cuda --drafter auto --prompt "What is 17 times 23?"
lamq serve 8.9GB --device cuda:0 --drafter auto --port 8098
lamq generate 11.2GB --device cuda --prompt "Why is the sky blue?"
```

CUDA supports single-sequence text generation. The loopback server provides OpenAI chat completions, including streaming, structured tool calls, and separate reasoning text. Set `reasoning_effort` in a request or choose the server default at startup. Both engines support greedy decoding and temperature sampling with an optional seed.

| Pack | Engine selected by default | DSpark |
| --- | --- | --- |
| 3.9 GB and 4.7 GB SQYFR v3 | PyTorch and Triton | Native PTX with the generic published drafter or an explicit local file |
| 5.4 GB SQYFR v2 and 8.9 GB SQYFR | PyTorch and Triton | Native PTX with a target-specific published drafter or an explicit local file |
| 11.2 GB and 12.8 GB with NVFP4 | PyTorch and Triton | Requires the native engine and a trellis target |

The native engine requires an NVIDIA driver for Ampere (SM 80) or newer. The Triton engine uses tuned GEMV for decode, packed tensor-core kernels where applicable for prefill, and a CUDA graph for greedy decode. See [GPU tuning coverage](inkling/tuning/COVERAGE.md) and [the upstream sync guide](SYNC.md). `python -m runtime_cuda.tune` runs the tuner on the current GPU.

The six short size names resolve to pinned public Hugging Face revisions. Repository IDs and local folders also work. Downloaded LFS files are checked against SHA-256 hashes for the resolved commit. `--drafter auto` uses the public DSpark repository. A local drafter file selects that exact weight file.

```sh
python -m runtime_cuda download 4.7GB
python -m runtime_cuda download auto --drafter --target 5.4GB
python -m runtime_cuda info 4.7GB
python -m runtime_cuda benchmark 8.9GB --engine triton --tokens 4 12 88 --max-tokens 32 --check-telemetry
```

Benchmark JSON includes raw token IDs, load time, time to first token, and decode throughput. With `--check-telemetry`, invalid NVIDIA telemetry withholds the throughput score and exits with status 2. A fresh Triton request can compile or capture a graph before it emits the first token.

Run `python -m pytest runtime_cuda` for synthetic GPU kernels, CPU-reference comparisons, downloads, and server framing. Released-pack oracle tests are optional and require the relevant weights and reference files.
