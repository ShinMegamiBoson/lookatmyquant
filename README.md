# LookAtMyQuant

LookAtMyQuant is a quantization compiler and runtime. The compiler measures how much compressing each stored piece of a model hurts its output, then picks a code for each piece so the whole model fits a size budget with the least damage.

The papers, methods, and full results are at [lookatmyquant.com](https://lookatmyquant.com/) ([compiler paper](https://lookatmyquant.com/compiler/), [LookAtMyQuant 2](https://lookatmyquant.com/2/)), and all published checkpoints are on [Hugging Face](https://huggingface.co/shinmegamiboson). The code is under the MIT license.

## Published models

### Inkling-2-Mini (3.9 GB to 12.8 GB)

Six Apple Silicon MLX and NVIDIA CUDA builds of [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26.4B parameters, 5.4B active, 18.89 GB of native text weights) spanning 8 GB devices up to teacher parity. They combine 4 KB SRAM hashed block-vector trellis codes, row trellis codes, per-row-group Fisher outlier allocation, exact native `NVFP4` passthrough on the highest-sensitivity expert buckets at 8.9–12.8 GB, and chat-templated proximal polish (v5).

All six builds were evaluated directly on the shipped packed artifacts against the BF16 teacher (`max_new_tokens = 1,536`, greedy, `reasoning_effort="none"`) on `openai/gsm8k` (`test[:200]`), all 164 `openai/openai_humaneval` problems (strict `<|content_text|>` execution), and 13-language / 6-class chat-templated KL windows. At low bit rates, most remaining HumanEval misses happen when long `<|content_thinking|>` traces exhaust the 1,536-token limit before answering; passing `--max-thinking-tokens 960` closes thinking cleanly at a sentence or newline boundary so the model finishes its answer. Below, unmodified → cap 960 shows scores without and with that cap:

| Build | Text GB | bpp | Raw KL (en / 13-lang) | 6-class KL | GSM8K (200) | HumanEval (164) |
|---|---:|---:|---:|---:|---:|---:|
| **Original BF16 teacher** | 18.89 | 5.80 | 0.0000 / 0.0000 | 0.0000 | 94.0% → 94.5% | 89.6% → 97.0% |
| **[12.8 GB](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-12.8GB)** | 12.74 | 3.92 | 0.0084 / 0.0209 | 0.0465 | 94.5% → 96.0% | 90.9% → 95.7% |
| **[11.2 GB](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-11.2GB)** | 11.16 | 3.43 | 0.0131 / 0.0315 | 0.0461 | 95.0% → 97.5% | 88.4% → 95.1% |
| **[8.9 GB](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB)** | 8.85 | 2.72 | 0.0278 / 0.0628 | 0.0530 | 95.0% → 96.5% | 84.8% → 94.5% |
| **[5.4 GB](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB)** | 5.38 | 1.65 | 0.0859 / 0.2659 | 0.1440 | 92.5% → 93.5% | 76.8% → 85.4% |
| **[4.7 GB](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-4.7GB)** | 4.72 | 1.45 | 0.1179 / 0.3731 | 0.1668 | 90.5% → 94.5% | 67.1% → 78.0% |
| **[3.9 GB](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-3.9GB)** | 3.85 | 1.18 | 0.2189 / 0.6058 | 0.2329 | 86.5% → 88.5% | 59.2% → 70.7% |

All six keep the untouched 637 MB BF16 vision tower (4/4 Vision VQA) and run through `lamq` with unified Metal kernels and optional lossless speculative decoding using the 179 MB [4-bit DSpark drafters](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter). Measured on an Apple M5 Max with the drafter loaded:

| Build | Active / peak RAM | Prefill (T=512) | Serial | Drafted (math / code) | Notes |
|---|---:|---:|---:|---:|---|
| **12.8 GB** | 12.73 / 13.51 GB | 1,556 tok/s | 53.8 tok/s | 125.9 / 75.5 tok/s | Teacher parity (16/16 16k recall) |
| **11.2 GB** | 11.14 / 11.97 GB | 1,540 tok/s | 53.1 tok/s | 116.9 / 87.7 tok/s | Highest cap-960 mean (96.3%) |
| **8.9 GB** | 8.84 / 9.68 GB | 1,784 tok/s | 56.0 tok/s | 139.7 / 96.2 tok/s | 15/16 16k recall |
| **5.4 GB** | 5.34 / 6.41 GB | 1,552 tok/s | 66.3 tok/s | 115.5 / 105.0 tok/s | Best sub-2 bpp quality |
| **4.7 GB** | 4.67 / 5.75 GB | 1,497 tok/s | 68.3 tok/s | 126.3 / 100.3 tok/s | Fits 12 GB iPhones with drafter |
| **3.9 GB** | 3.81 / 4.91 GB | 1,353 tok/s | 75.2 tok/s | 118.3 / 96.3 tok/s | Fits 8 GB devices with drafter |

Each model's Hugging Face card has per-class KL and 16k recall breakdowns.

### Qwen3.8-27B in 7.2 GB

The LookAtMyQuant 2 recipe takes Qwen3.8-27B from 53.7 GB to 7.2 GB (2.1 bits per weight), calibrated and distilled on English prose, chat, code, and math. [The packed file](https://huggingface.co/shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB) solves 401 of 540 exact-answer math problems — matching the original — while Unsloth's same-size `UD-IQ2_XXS` GGUF solves 348. It also scores above that GGUF on HumanEval (73.8% vs. 67.1%), MMLU (78.8% vs. 75.0%), and GSM8K (94.5% vs. 92.7%).

### LookAtMyQuant 2: Qwen3.5-4B in 1.2 GB

[LookAtMyQuant 2](https://lookatmyquant.com/2/) compresses Qwen3.5-4B to 1.23 GB (2.3 bits per weight). It solves 330 of 540 exact-answer math problems — against 328 for Prism ML's 1.20 GB Bonsai 2 Ternary 4B and 359 for the original — and lowers held-out text KL from 0.59 to 0.45 nats.

The recipe is trellis post-training quantization calibrated on the model's own worked math solutions (with digit tokens weighted 10× and bits allocated per row by Fisher information), followed by 4,000 steps of distillation in which every layer is re-quantized once from the original weights (5.4 GPU-hours; see [LookAtMyQuant 2 recipe on Qwen3.5-4B](#lookatmyquant-2-recipe-on-qwen35-4b)).

The 1.23 GB file is on [Hugging Face](https://huggingface.co/shinmegamiboson/Qwen3.5-4B-LookAtMyQuant-1.2GB) and in the [`lm2-qwen35-4b-1.2gb` release](https://github.com/ShinMegamiBoson/lookatmyquant/releases/tag/lm2-qwen35-4b-1.2gb). Its 464 MB 4-bit DSpine drafter ([Hugging Face](https://huggingface.co/shinmegamiboson/Qwen3.5-4B-LookAtMyQuant-1.2GB-drafter) / [`lm2-drafter-4bit` release](https://github.com/ShinMegamiBoson/lookatmyquant/releases/tag/lm2-drafter-4bit)) runs 1.5–2.0× faster on Apple silicon with identical greedy output (176 vs. 88 tok/s on an M5 Max across 30 held-out prompts).

## Quick start

### Apple Silicon (MLX)

Install the runtime package directly (installs `mlx`, `mlx-lm`, `numpy`, and `huggingface-hub` without the PyTorch compiler dependencies):

```sh
pip install git+https://github.com/ShinMegamiBoson/lookatmyquant.git
```

From a local clone, `pip install -e .` installs the `lamq` command, or run `python -m runtime_mlx` without installing.

**CLI.** Pass a local model folder or a Hugging Face repo ID:

```sh
lamq generate shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB \
    --drafter shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter \
    --max-thinking-tokens 960 \
    -p "What is 17 times 23? Show the steps." --stats

lamq chat shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB \
    --drafter shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter

lamq generate shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB \
    -p "Write a Python function that merges overlapping intervals." --stats

lamq generate shinmegamiboson/Qwen3.5-4B-LookAtMyQuant-1.2GB \
    --drafter shinmegamiboson/Qwen3.5-4B-LookAtMyQuant-1.2GB-drafter \
    -p "What is 17 times 23? Show the steps." --stats
```

`lamq generate` reads stdin when `-p` is omitted and supports `--raw`, `--system`, `--thinking`, `--max-thinking-tokens`, `--temp`, `--top-p`, `--mode linear|chat-fork|copy-wide`, `--no-adaptive`, `--prefill-chunk N` (`-1` auto, `0` single pass), and `--json`. `lamq info <model>` inspects a packed model's format and dimensions without loading weights.

Inkling prefill runs in one pass when the prompt fits the estimated MLX memory budget and chunks automatically under pressure. To shrink global attention cache memory on long prompts, set `LAMQ_GLOBAL_KV=fp8` or `LAMQ_GLOBAL_KV=q4` (default `fp16`; sliding-window attention stays FP16; either option can change generated tokens).

**Python API.** `runtime_mlx.load` returns a `(model, tokenizer)` pair compatible with `mlx_lm.generate` and `mlx_lm.stream_generate` (auto-detecting `opgraph-packed-1`, `inkling2-packed`, and `lamq.json` exports):

```python
from mlx_lm import generate
from runtime_mlx import load

model, tok = load("shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB")
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "What is 17 times 23? Show the steps."}],
    add_generation_prompt=True,
    enable_thinking=False,
)
print(generate(model, tok, prompt, max_tokens=200))
```

For streaming, multi-turn chat, or speculative decoding through one interface, use `Runtime`:

```python
from runtime_mlx import Runtime

rt = Runtime.load(
    "shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB",
    drafter="shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter",
)
for chunk in rt.stream(
    "Solve 3x + 4y = 31 and 5x - 2y = 17.",
    max_tokens=1536,
    max_thinking_tokens=960,
):
    print(chunk.text, end="", flush=True)
```

### NVIDIA GPUs (CUDA)

Install the CUDA extra on a machine with a supported NVIDIA GPU. The model IDs and chat templates are shared with the Apple Silicon runtime:

```sh
pip install -e ".[cuda]"
lamq generate shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB --device cuda -p "What is 17 times 23?"
lamq serve shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB --device cuda
```

Triton handles ordinary generation and `NVFP4` experts; the native CUDA engine adds DSpark speculative decoding on trellis packs. Set `LAMQ_CUDA_GLOBAL_KV=q8_0` or `q4_0` to compress global KV caches, and `LAMQ_CUDA_KV_VRAM_CELLS` to spill older cache cells to pinned host memory. See [`runtime_cuda/README.md`](runtime_cuda/README.md) for model support, short size aliases, revision selection, and tuning commands.

### HTTP server and browser chat (`lamq serve`)

`lamq serve` serves a packed Inkling-2-Mini model over HTTP on Apple Silicon or CUDA (`--device cuda`):

```sh
lamq serve shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB \
    --drafter shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter --port 8080 --ui

curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "What is 17 times 23?"}], "reasoning_effort": "low"}'
```

- **Endpoints:** OpenAI `/v1/chat/completions` and `/v1/responses` (Codex CLI), Anthropic `/v1/messages` (Claude Code; `developer` messages map to `system`), `/v1/models`, `/health`, plus `POST /apply-template` and `POST /tokenize`. Supports streaming, `reasoning_effort` (`reasoning_content`), and `tools` (`"tool_choice": "auto"` or `"none"`; forced tool calls return HTTP 400).
- **Browser chat (`--ui`):** Serves a streaming chat UI at `http://127.0.0.1:8080/` that keeps the conversation and draft in session storage, with settings for reasoning and output length and expandable timing/token usage. Browser-origin API calls are disabled by default; for remote access behind an HTTPS proxy, preserve the browser's `Host` header (including any non-default port) and disable response buffering. The page itself is public and prompts for the API key when auth is enabled.
- **Batching and prompt caching:** On Apple Silicon MLX, `lamq serve` runs continuous batching by default (`--max-batch 32`) with paged KV pools, automatic zero-copy shared-prefix reuse on loopback (`--no-prefix-cache` disables it), interleaved chunked prefill (`--prefill-step 512`), and load-adaptive DSpark (`--dspark-max-batch 4`). Pass `--max-batch 1` for single-stream serving, where `--prompt-cache N` keeps up to `N` in-memory KV snapshots and `--prompt-cache-dir PATH` persists incremental deltas to disk across restarts (capped by `--prompt-cache-gb`, default `20`; passing `--prompt-cache-dir` selects single-stream mode automatically). Prefix and prompt caching are disabled when auth or non-loopback binding is active. In Python, pass `prompt_cache=PromptCache()` from `runtime_mlx.inkling` to `stream_dspark` or `stream_serial`.
- **Limits, guard, and auth:** `--context` (default `32768`) caps prompt plus completion with llama.cpp's context-overflow error for agent compaction; requests cap at 1 MiB JSON and return HTTP 503 when all admission slots are full. A stateless tool-loop repetition guard is enabled by default (`--no-repeat-guard` or `LAMQ_REPEAT_GUARD=0` disables it). The server binds to `127.0.0.1` by default; set `LAMQ_API_KEY` to require bearer / `x-api-key` auth (required for network listeners; put a TLS-terminating proxy in front on untrusted networks).

### Exporting custom plans, drafters, and benchmarks

- **Lloyd–Max and trellis exports:** `quantize.py --export <folder>` writes a `lamq.json` export for `runtime_mlx/` (add `--trellis-rows 16,8` for trellis plans). Benchmark with `python runtime_mlx/bench.py lamq path/to/export`.
- **Packing Qwen3.5 / Qwen3.8 and training drafters:** `runtime_mlx/pack_dense.py` (or `runtime_mlx/modal_pack.py` on Modal) writes the `opgraph-packed-1` format. `modal_drafter.py` trains a DSpine drafter with [SpecForge](https://github.com/sgl-project/SpecForge) on a packed model's greedy answers, and `pack_drafter.py` compresses a trained drafter to 4 bits with no calibration data (464 MB for Qwen3.5-4B).
- **Speculative decoding modes:** `drafter_q35.py` verifies 8-token linear blocks with a compact 96,033-token proposal vocabulary, fused packed argmax, and self-calibrating adaptive serial fallback (`--no-adaptive` disables fallback) so low-acceptance prose does not slow down while greedy output stays bit-exact. Pass `--mode chat-fork` to branch on ambiguous early tokens or `--mode copy-wide` for 16-slot contiguous copy verification:

```sh
python runtime_mlx/bench_drafted.py path/to/packed path/to/drafter
python runtime_mlx/bench_drafted.py path/to/packed path/to/drafter --mode chat-fork --prompts chat
python runtime_mlx/bench_drafted.py path/to/packed path/to/drafter \
    --mode copy-wide --heldout path/to/copy-prompts.json
python runtime_mlx/check_drafter.py path/to/packed path/to/drafter
python pack_drafter.py path/to/drafter path/to/drafter-4bit
```

## Running the compiler

### Quick run on SmolLM2-360M

Install the PyTorch compiler dependencies and download SmolLM2-360M and WikiText-2 first (the compiler reads the model's config at startup):

```sh
pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download as d; d('HuggingFaceTB/SmolLM2-360M')"
python -c "from datasets import load_dataset as l; l('Salesforce/wikitext', 'wikitext-2-raw-v1')"
```

**1. Measure.** Compresses one piece at a time, leaves the rest exact, and records the KL to the original model. The full 2,312 measurements for SmolLM2-360M took about 50 minutes on an L40S:

```sh
python compiler.py measure --model HuggingFaceTB/SmolLM2-360M \
    --part linear --layers 0-3 --out results/batch/my_measurements.json
```

**2. Allocate.** Picks one code per piece with the least summed KL inside the budget (CPU only). Using the SmolLM2-360M measurements already in this repository at 3.044 bits per weight reproduces the widths in `results/compiler/plans/sm_lin_3044.json`:

```sh
python compiler.py allocate --sens 'results/batch/ccSens*.json' \
    --target disk --items linear --bits 3.044 --out results/compiler/plans/my_plan.json
```

**3. Build and score.** Quantizes the model to the plan one matrix at a time and reports WikiText-2 test perplexity and KL (our run in `results/batch/cc_sm_lin_3044.json` reached perplexity 16.64 and KL 0.0926 against original perplexity 15.31; add `--layers 1 --calib 8 --eval 2` for a quick smoke check):

```sh
python quantize.py --program plan --model HuggingFaceTB/SmolLM2-360M \
    --plan results/compiler/plans/my_plan.json --out results/my_run.json
```

### LookAtMyQuant 2 recipe on Qwen3.5-4B

Each step is a named job in `run_batch.py` (`python run_batch.py --list` prints its command). The jobs expect Qwen3.5-4B in `models/Qwen3.5-4B` on the Modal volume and the text mix (`modal run --detach modal_fetch.py --text`), and write outputs under `lm2/` on the volume:

```sh
modal run --detach modal_app.py --jobs lm2Text
modal run --detach modal_app.py --jobs lm2Calibration,lm2Solutions
modal run --detach modal_app.py --jobs lm2RowImportance
modal run --detach modal_app.py --jobs lm2Build
modal run --detach modal_app.py --jobs lm2Finetune
modal run --detach modal_app.py --jobs lm2Math,lm2Math216,lm2TextKL
```

| Job | Script | What it does | Our run |
|---|---|---|---|
| `lm2Text` | `other_text.py` | Original answers science, code, instruction, tool, MATH, and GSM8K prompts (downloads MATH) | — |
| `lm2Calibration` | `math_calibration.py` | Greedy solutions to 384 MATH training problems in 2,048-token windows | 15 min, H100 |
| `lm2Solutions` | `checked_solutions.py` | Math distillation text from MATH and `math_bench.py`, with wrong steps corrected and only right answers kept | 33 min, H100 |
| `lm2RowImportance` | `row_importance.py` | Per-row Fisher information at digit-predicting positions | 1 min, H100 |
| `lm2Build` | `quantize.py` | Row trellis PTQ with 10× digit-weighted Hessians, QEP (α = 0.5), row plan, and e-means codebooks | 37 min, H100 |
| `lm2Finetune` | `finetune.py` | 4,000 steps of proximal distillation, re-quantizing every layer once from the original weights | 2.7 hr, B200 |
| `lm2Math`, `lm2Math216` | `math_eval.py` | 540-problem and 216-problem exact-answer math benchmarks across original, Bonsai, and builds | 20 min / model, H100 |
| `lm2TextKL` | `text_kl.py` | KL from original on WikiText-2 and the 13-source text mix | 1 min / model, H100 |

`lm2BuildSmoke`, `lm2FinetuneSmoke`, and `lm2MathSmoke` run the same code on two layers, 60 steps, and 36 problems in a few minutes each.

- **Row plan and codebooks:** `lm2Build` reads `results/compiler/rows/q35_rowplan_1p2GB.json` (1 to 4 bits across 1,520 row groups, generated by `row_allocation.py` and `rowgroup_alloc.py`) and decodes with the four e-means codebooks in `cache/codes/` (`--trellis-fit lloyd` selects the earlier codebooks).
- **Equal-byte recipe options:** `--embed-order sfisher-mix` (`compiler.py measure`, `quantize.py`) ranks tied token buckets by a 50/50 blend of Softmax-Fisher and corpus frequency, and `--realloc-at 1000 --rebuild-at 1000 --realloc-max-delta 1` (`finetune.py`) shifts bits mid-training by each layer's continuous-vs-stored proximal KL gap at exact byte parity.
- **Differences from our published run:** Our GSM8K solutions were written earlier with thinking off, and our 13-source KL used 52 separately sampled windows (`--sources mix` uses the mix's own evaluation windows, scoring 0.44 where the page reports 0.45).

### Large models and GGUF on Modal

`run_batch.py` defines all named jobs, `modal_app.py` runs them (one GPU per job, 8 CPU cores), and `modal_fetch.py` downloads models and the text mix:

```sh
modal run modal_app.py::setup
modal run --detach modal_fetch.py --text
modal run --detach modal_fetch.py --models <hf-repo-id>
python run_batch.py --list
```

`gguf_compiler.py` compiles plans into llama.cpp's native tensor types so the output runs directly in llama.cpp, and `public_eval.py` evaluates public quantizations under the same harness. Both run in the CUDA llama.cpp image defined by `modal_gguf.py`.

## Repository layout

| Path | What it holds |
|---|---|
| `compiler.py`, `compiler_arch.py` | Sensitivity measurement (`measure`), knapsack allocation (`allocate`), architecture discovery, and text mixes |
| `quantize.py` | Builds a program or plan and evaluates perplexity and KL |
| `opgraph/` | Model-as-tensor-network grammar, rewrites, LDLQ, and codes |
| `row_importance.py`, `rowgroup_alloc.py`, `row_allocation.py`, `compiler_artifact.py` | Per-row Fisher importance, row-group allocation, and the `opgraph-packed-1` format |
| `math_calibration.py`, `checked_solutions.py`, `other_text.py`, `calc_check.py`, `token_kinds.py` | Calibration/distillation text generation, arithmetic checker, and token-kind tagging |
| `finetune.py`, `finetune_lib.py`, `fast_delta.py` | Proximal code-targeted distillation and fast Qwen3.5 linear-attention kernel |
| `math_bench.py`, `math_eval.py`, `text_kl.py` | Exact-answer math benchmark, runner, and text KL evaluation |
| `gguf_compiler.py`, `public_eval.py`, `eval_tasks.py` | llama.cpp GGUF compiler, public quantization scorer, and task benchmarks |
| `pyproject.toml`, `runtime_mlx/runtime.py`, `runtime_mlx/cli.py` | Installable runtime package, unified `load` / `Runtime` Python API, and `lamq` CLI |
| `runtime_mlx/server.py`, `runtime_mlx/prompt_cache_disk.py`, `runtime_mlx/chat.html` | `lamq serve` HTTP server (OpenAI + Anthropic endpoints), disk prompt cache, and `--ui` chat page |
| `lamq_export.py`, `runtime_mlx/lamq.py`, `runtime_mlx/model.py` | `lamq.json` exporter and Apple Silicon MLX runner for Lloyd–Max and trellis plans |
| `runtime_mlx/fast_q35.py`, `runtime_mlx/block_q35.py`, `runtime_mlx/block_argmax_q35.py`, `runtime_mlx/pack_dense.py`, `runtime_mlx/modal_pack.py` | Packed Qwen3.5-4B / Qwen3.8-27B MLX runtime (serial and block-8/16) and dense packer |
| `runtime_mlx/inkling/` | Apple Silicon MLX Inkling-2-Mini runtime, 4 KB SRAM trellis / `NVFP4` Metal kernels, paged-KV continuous batching, and 4-bit DSpark drafter |
| `runtime_mlx/drafter_q35.py`, `runtime_mlx/draft_vocab_q35.py`, `runtime_mlx/convert_dspine.py`, `runtime_mlx/strategies/`, `drafter/`, `modal_drafter.py`, `pack_drafter.py` | DSpine speculative decoding, fork/copy verifiers, DSpine format conversion, drafter training, and 4-bit packing |
| `runtime_cuda/inkling/`, `runtime_cuda/inkling_cuda/` | Triton and native PTX CUDA engines for packed Inkling-2-Mini models, plus float32 Transformers reference |
| `runtime_cuda/tune.py`, `runtime_cuda/tuning_coverage.py`, `runtime_cuda/upstream_sync.py` | Per-GPU kernel tuning and MLX/CUDA coverage checks |
| `run_completeness.py`, `make_report.py`, `analyze_compiler.py` | Nine-method reproduction check and result tables |
| `run_batch.py`, `modal_app.py`, `modal_gguf.py`, `modal_fetch.py` | Named batch jobs and Modal runners |
| `cache/codes/` | E-means trellis codebooks at 1 to 4 bits |
| `results/` | Compiler plans (`compiler/plans/`, `compiler/rows/`), measurements (`batch/`, `gguf/`), protocols (`protocols/`), and write-ups (`COMPILER.md`, `COMPLETENESS.md`) |
| `tests/` | Compiler, grammar, code, allocation, and reallocation unit tests |

## Tests

The eight compiler and grammar test suites run from the repository root in under a minute (most need SmolLM2-360M and WikiText-2 cached locally):

```sh
OPGRAPH_DEVICE=cpu python -m unittest discover -s tests
```

Or run any file directly: `tests/test_frame.py`, `tests/test_grammar.py`, `tests/test_codes.py`, `tests/test_elements.py`, `tests/test_nested.py`, `tests/test_rowgroup_alloc.py`, `tests/test_reallocation.py`, `tests/test_compiler_upgrades.py`.

**Apple Silicon (MLX) tests:**

- `runtime_mlx/test_fast_q35.py` and `runtime_mlx/test_block_q35.py` check Qwen3.5 kernels against float64 NumPy products with no model required. Setting `LAMQ_Q35_PACKED` or `LAMQ_Q38_PACKED` also compares block decoding against one-token decoding on a real packed checkpoint.
- `runtime_mlx/test_inkling.py` builds a miniature packed Inkling-2-Mini model to verify Metal kernels, chunked prefill, `--max-thinking-tokens`, and `PromptCache` reuse.
- `runtime_mlx/test_server.py` and `runtime_mlx/test_cli_cuda.py` test `lamq serve` (parsing, OpenAI/Anthropic/Responses translation, HTTP layer) and CLI routing with no model weights.
- `runtime_mlx/test_core.py` (after `python runtime_mlx/make_ref.py`, which writes 1.6 GB of reference tensors) and `runtime_mlx/test_trellis_gen.py` + `runtime_mlx/test_trellis.py` check `lamq.json` runtime math against the quantizer. `runtime_mlx/test_chat_browser.py` runs optional Playwright browser checks for `--ui`.

**CUDA tests:**

```sh
python -m pytest runtime_cuda/
```

GPU kernel checks use synthetic inputs (and skip on machines without CUDA); checkpoint oracle checks require the packed weights and reference outputs.

## What this repository leaves out

- **Model weights.** Published checkpoints download from [Hugging Face](https://huggingface.co/shinmegamiboson); base models download from their upstream Hugging Face repos (the Qwen3.5-4B Modal jobs expect `models/Qwen3.5-4B` on the volume).
- **Benchmark copies.** `eval_tasks.py` downloads benchmarks automatically, or reads offline exports from `data/bench/` (cache them on the Modal volume before offline runs).
- **The website.** The papers live at [lookatmyquant.com](https://lookatmyquant.com/). The one-off scripts that assembled the site's JSON data are omitted; the underlying compiler measurements are in `results/`.

## Credits and acknowledgments

### Ideas from Cary Palmer's Kintsugi

Cary Palmer's [Kintsugi method](https://github.com/professorpalmer/kintsugi-method) paper measured our earlier 8.9 GB Inkling-2-Mini build against [Kintsugi for Inkling 2 Mini](https://github.com/professorpalmer/Kintsugi-Inkling-2-Mini) by text category. Both builds tied on overall KL and ours was 25% lower on prose, while higher on code (+29%), tool calls (+35%), agentic text (+83%), and reasoning (+74%), because our calibration had not been rendered in Inkling's chat template and our expert Hessians saw only tokens routed to each expert. We adopted eight ideas from the two Kintsugi repositories:

| Idea from Kintsugi | Where it is here |
|---|---|
| Each expert's Hessian comes from every calibration token, not only tokens routed to it | `--dense-expert-h` in `compiler.py measure` and `quantize.py` |
| Calibration text is rendered in the model's chat template, and KL is scored separately for prose, code, tools, agentic text, reasoning, and multilingual text | `--data templated` and `--per-class` in `compiler.py measure` |
| A plan is judged by its worst text class as well as by its mean KL | `--max-class-ratio` and `--class-weights` in `compiler.py allocate` |
| Long-context recall of a fact or code copy is scored as a teacher-forced margin against the same item placed nearby | The recall check run on new Inkling-2-Mini builds |
| Sliding-window caches stay small FP16 rings, global caches use `q8_0` or `q4_0`, and cells past a VRAM limit move to pinned host memory | `LAMQ_CUDA_GLOBAL_KV` and `LAMQ_CUDA_KV_VRAM_CELLS` in the CUDA runtime |
| Long chats persist to disk so they resume across restarts without re-prefilling | `--prompt-cache-dir` and `--prompt-cache-gb` in `lamq serve` |
| When an agent repeats nearly the same tool call, the server adds a short nudge to the request | On by default in `lamq serve`; `--no-repeat-guard` disables it |
| Claude Code and Codex connect directly via Anthropic's `/v1/messages` and OpenAI's `/v1/responses`, with `developer` messages read as `system` | `lamq serve` |

### Runtime speedups from Actual's aci-cpp

Justin Bennington's post [In Search of Perfection: aci-cpp vs. llama.cpp](https://actual.inc/blog/in-search-of-perfection) describes aci-cpp, Actual's fork of llama.cpp. We implemented two of its ideas from that description:

| Idea from aci-cpp | Where it is here |
|---|---|
| Gated DeltaNet prefill runs as fused, chunked kernels | `runtime_mlx/fast_q35.py` (16–37% faster Qwen3.8-27B prefill on M5 Max at 256–2,048 tokens) |
| Long-context attention decoding splits the span across parallel reductions | `runtime_mlx/inkling/fast_kernels.py` (32- or 64-way splits on long spans) |

### Contributors

| Contributor | Contribution |
|---|---|
| [Patrick Dugan](https://github.com/patrickdugan) | Native CUDA engine for packed Inkling-2-Mini files, with DSpark verification and chat server (`runtime_cuda/inkling_cuda/`) |
| [Ian B](https://github.com/ifkb99) | Triton CUDA runtime for packed Inkling-2-Mini models (`runtime_cuda/inkling/`) |
| [Mark Pesce](https://github.com/mpesce) | `lamq serve` with prompt caching, and chunked Inkling prefill so long prompts fit in memory |
| [Noah Lindner](https://github.com/Octember) | Trellis tables staged in threadgroup memory for Qwen and Inkling decoding, after reporting slow M1 Pro decode |
| [Dmitry (Dee) Kargaev](https://github.com/deeflect) | Apple M4 Metal tuning, `fp8` and `q4` global KV cache, and the fix for the DSpark threadgroup crash on M1–M3 |
| [Dung Duc Huynh](https://github.com/jellydn) | Report of the Metal threadgroup crash on M1 Max |
| [Max Ng](https://github.com/maxwindiff) | Inkling kernels that launch within M1 threadgroup limits |
| [Amlandeep Bhadra](https://github.com/koko1123) | Browser chat page for `lamq serve` (`--ui`) |
