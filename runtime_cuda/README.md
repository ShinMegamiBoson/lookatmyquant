# Inkling 2 Mini CUDA runtime

This runtime reads the original LookAtMyQuant Inkling 2 Mini text packs. The CUDA path keeps packed weights, activations, residuals and all 48 visit caches on the device. It includes packed embedding decode, Hadamard rotations, clustered residual reads, cross-pass message gates, causal convolution, grouped-query attention with relative bias and sinks, and top-k expert routing. Each routed projection dispatches all selected experts in one kernel regardless of which packed formats they use.

The NVIDIA driver JIT-compiles the PTX. Python loads the checkpoint, handles tokenization and schedules kernels. Prefill processes up to 16 consecutive positions per layer, including packed projections, causal convolutions and attention. Each greedy decode step copies an eight-byte token and status pair to the host. The `hybrid` backend provides the NumPy decoder with CUDA projections for comparisons.

## Run

Use Python 3.10 or newer and a recent NVIDIA driver on Ampere or newer. From `runtime_cuda` in the checkout:

```powershell
python -m pip install numpy huggingface-hub transformers jinja2
python -m inkling_cuda download .\models\inky-5.4
python -m inkling_cuda generate .\models\inky-5.4 --prompt "What is the capital of France?" --max-new-tokens 64
```

`generate` accepts an existing pack directory, `--tokens` for raw IDs, `--json` for timing and token records, and `--backend hybrid` for comparison. Downloads pin the selected release. `--size 8.9GB` selects the larger pack.

### Goose and other local chat clients

Use the cached pack directory and start the loopback provider.

~~~powershell
python -m inkling_cuda serve .\models\inky-5.4 --no-thinking
~~~

Set the client's OpenAI base URL to `http://127.0.0.1:8098/v1/` and its model ID to `inky-5.4`. The provider supports streaming and non-streaming chat completions and defaults to a 16,384-token context. It streams reasoning separately from answer text and returns completed, declared Inkling tool calls in OpenAI format. The template and tokenizer endpoints remain available for Goose preflight.

Short requests and requests with tools default to low reasoning effort. Other prompts of at least 512 tokens default to direct answers to leave more of the output budget for the answer. For direct answers the server masks the thinking-channel marker at target sampling. Callers can select an explicit `reasoning_effort` at any prompt length. The process-wide `--no-thinking` option forces direct answers for every request.

### DSpark speculative decoding

Install a [CUDA-enabled PyTorch build](https://pytorch.org/get-started/locally/) and `safetensors` to use the optional drafter.

~~~powershell
python -m pip install safetensors
python -m inkling_cuda download-drafter .\models\inky-drafter
python -m inkling_cuda serve .\models\inky-5.4 --drafter .\models\inky-drafter
~~~

The server keeps the drafter resident and uses block size 2 by default. Use `--draft-block-size` to override it. The drafter uses four target activations per position and scores proposals over the first five head buckets. The target verifies against the full packed head before each token can reach the client. A positive `temperature` uses CUDA-enabled PyTorch to sample the target distribution, with a positional random draw reused if a draft is rejected. The standalone `generate` command retains greedy decoding and reports draft acceptance with its timing and token data.

### Checking telemetry before benchmarking

Use `check-telemetry` for a read-only NVIDIA diagnostic without loading a model. Use `generate --check-telemetry` to sample the actual CUDA device before, during, and after a timing run.

```powershell
python -m inkling_cuda check-telemetry
python -m inkling_cuda generate .\models\inky-5.4 --prompt "Explain how a thermos works." --max-new-tokens 64 --check-telemetry --json timing.json
```

The check records the enforced power limit, clocks, and GPU utilization. It returns exit code 2 on query errors, unavailable power draw, or implausible readings. A checked run skips model loading if the first sample fails. If telemetry fails during generation, the report retains the tokens and samples with a null decode score. Low clocks by themselves can be normal at idle. Passing the check establishes that the sampled fields were plausible, and a separate CUDA workload can help isolate a clock or performance problem.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The tests cover all nine packed code formats, release-width Hadamard transforms, BF16 message-gate rounding, and GPU logits against the NumPy reference. Batched prefill and speculative verification are checked against serial decoding through local attention and convolution cache wraparound.

`synthetic-smoke.json` records the original serial decoder benchmark. In that fixture, small differences at BF16 message-gate boundaries changed the experts selected later in the loop.

On an RTX 5080 Laptop GPU, the two published packs produced identical token IDs with CUDA greedy decoding and DSpark on “What is the capital of France? Reply with just the city name.” The test used a 48-token limit and a four-position draft block.

| Pack | Matched output IDs | Greedy decode | DSpark decode |
| --- | ---: | ---: | ---: |
| 5.4 GB | 48 | 14.1 tokens/s | 17.6 tokens/s |
| 8.9 GB | 47 (EOS) | 12.7 tokens/s | 1.7 tokens/s with another inference process active |

For the 5.4 GB run, the hybrid decoder matched the same 48 IDs. The 28-token prompt took 0.56 s to prefill with CUDA and 2.97 s with the hybrid backend. On the separate arithmetic prompt “What is 17 times 23? Give just the answer.” the 5.4 GB pack repeated the expression in both generation modes.

The implemented scope is single-sequence text generation. End-to-end comparison with the Apple Silicon Metal runtime remains open. To replay the same prompt tokens on MLX, save a greedy CUDA run with `--json cuda-run.json` and run this from the repository root on Apple Silicon with the same checkpoint revision.

~~~python
import json
from pathlib import Path

from runtime_mlx.inkling import Inkling2LoopPackedMLX, generate_serial

run = json.loads(Path("cuda-run.json").read_text())
model = Inkling2LoopPackedMLX("/path/to/same/packed/checkpoint")
ids, _ = generate_serial(model, run["prompt_tokens"], max_tokens=len(run["generated_tokens"]))
assert ids == run["generated_tokens"]
~~~

Decoder equations and test fixtures follow the existing MLX runtime under the upstream MIT license.
