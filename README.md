# LookAtMyQuant

LookAtMyQuant is a quantization compiler. It measures how much each stored piece of a model hurts the output when compressed. It then picks a code for each piece so the whole model fits a size budget with the least damage.

On Gemma 4 E2B, its GGUF has 12× lower KL than Q4_K_M at the same file size. The method and every result are on [the site](https://lookatmyquant.pages.dev).

The code is under the MIT license.

## LookAtMyQuant 2: Qwen3.5-4B in 1.2 GB

[LookAtMyQuant 2](https://lookatmyquant2.pages.dev) takes Qwen3.5-4B to 1.23 GB, 2.3 bits per weight. The file solves 330 of 540 exact-answer math problems. Prism ML's Bonsai 2 Ternary 4B solves 328 at 1.20 GB, and the original solves 359. Its KL from the original on held-out text is 0.45 nats, against 0.59 for Bonsai.

The recipe is post-training quantization with trellis codes, calibrated on the model's own worked math solutions with digit tokens weighted 10×, and with bits allocated per row by Fisher information. 4,000 steps of distillation follow, in which every layer is re-quantized once from the original weights. The page has the findings, the full recipe and its cost, about $34 of GPU time.

Of that recipe, this repository has the trellis codes and the e-means fit of their codebooks. New measurements and plans use that fit by default; `--trellis-fit lloyd` gives the earlier codebooks, and older plans keep them.

## Run the compiler on SmolLM2-360M in three commands

Install the dependencies and download the model and WikiText-2. The code reads SmolLM2-360M's config when it starts, so do this first.

```sh
pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download as d; d('HuggingFaceTB/SmolLM2-360M')"
python -c "from datasets import load_dataset as l; l('Salesforce/wikitext', 'wikitext-2-raw-v1')"
```

**Measure.** This compresses one piece at a time, leaves the rest exact, and records the KL to the original model. The full set for this model, 2,312 measurements, took about 50 minutes on an L40S.

```sh
python compiler.py measure --model HuggingFaceTB/SmolLM2-360M --part linear --layers 0-3 --out results/batch/my_measurements.json
```

**Allocate.** This picks one code per piece, with the least summed KL inside the budget. It needs no GPU. This command uses the SmolLM2-360M measurements in this repository, with a budget of 3.044 bits per weight.

```sh
python compiler.py allocate --sens 'results/batch/ccSens*.json' --target disk --items linear --bits 3.044 --out results/compiler/plans/my_plan.json
```

The plan it writes has the same widths as `results/compiler/plans/sm_lin_3044.json`.

**Build and score.** This quantizes the model to the plan, one matrix at a time. It then reports perplexity and KL to the original on WikiText-2 test.

```sh
python quantize.py --program plan --model HuggingFaceTB/SmolLM2-360M --plan results/compiler/plans/my_plan.json --out results/my_run.json
```

Our run of that plan is in `results/batch/cc_sm_lin_3044.json`. It reached perplexity 16.64 and KL 0.0926, against an original perplexity of 15.31. Add `--layers 1 --calib 8 --eval 2` for a quick check that everything works.

## Large models run on Modal, one GPU per job

`run_batch.py` names every job, and `modal_app.py` runs them. `modal_fetch.py` downloads models and the text mix used for calibration and evaluation.

```sh
modal run modal_app.py::setup
modal run --detach modal_fetch.py --text
modal run --detach modal_fetch.py --models google/gemma-4-E2B-it
modal run --detach modal_app.py --jobs ccMG4E2Noise,ccMG4E2Spot,ccMG4E2EmbKv
```

The runs behind the site used these machines, each with 8 CPU cores:

| Model | GPU | Time |
|---|---|---|
| SmolLM2-360M | L40S | About 50 minutes to measure, in 9 parallel jobs |
| Gemma 4 E2B, measuring | H100 | 7–26 minutes per job, in 4 jobs |
| Gemma 4 E2B, building a plan | H100 | 28–58 minutes per plan |

`gguf_compiler.py` makes plans out of llama.cpp's own tensor types, so the result runs in llama.cpp as it is. `public_eval.py` scores public quantizations the same way as our plans. Both run inside the image in `modal_gguf.py`, which builds llama.cpp with CUDA.

## The Apple Silicon runtime reads Lloyd–Max and trellis plans

`quantize.py --export <folder>` writes a plan in the format `runtime_mlx/` reads. A plan with trellis codes also needs `--trellis-rows 16,8`, which runs each trellis path along a row of its matrix, so that decoding an entry reads only that row. The runtime needs `mlx` and `mlx_lm`, which are not in `requirements.txt`.

On Gemma 4 E2B at 2.5 GB, the trellis plan has 26% lower KL than the Lloyd–Max plan and decodes about 22% slower. `runtime_mlx/test_trellis_gen.py` and `runtime_mlx/test_trellis.py` check the trellis kernels against the matrices the quantizer built.

```sh
python runtime_mlx/bench.py lamq path/to/export
```

## Where everything is

| Path | What it holds |
|---|---|
| `compiler.py` | Measure and allocate |
| `compiler_arch.py` | How the compiler finds a model's layers, tables and stages, and the text mix |
| `quantize.py` | Builds a program or a plan, then scores it |
| `gguf_compiler.py`, `public_eval.py`, `eval_tasks.py` | The llama.cpp route, public quantizations and task benchmarks |
| `lamq_export.py`, `runtime_mlx/` | The export format and the Apple Silicon runtime |
| `opgraph/` | The grammar: a model as a tensor network, and the rewrites and codes that act on it |
| `run_completeness.py`, `make_report.py` | The check that nine published methods are reproduced exactly |
| `analyze_compiler.py` | The tables of plan results |
| `run_batch.py`, `modal_app.py`, `modal_gguf.py`, `modal_fetch.py` | Named jobs and the Modal runners |
| `results/compiler/plans/` | Every plan |
| `results/batch/`, `results/gguf/` | The measurements and plan runs behind the site |
| `results/COMPILER.md`, `results/COMPLETENESS.md` | Write-ups of the compiler on SmolLM2-360M and Qwen3.5-4B, and of the reproduction check |
| `site/` | The website. Run `python3 -m http.server 8000` inside it, then visit http://localhost:8000 |
| `site2/` | The LookAtMyQuant 2 page. It runs the same way |
| `tests/` | Tests of the grammar and the codes |

## The tests need SmolLM2-360M and WikiText-2

Each test file runs on its own from the repository root, and all five finish in under a minute.

```sh
python tests/test_frame.py
python tests/test_grammar.py
python tests/test_codes.py
python tests/test_elements.py
python tests/test_nested.py
```

`runtime_mlx/test_core.py` checks the runtime against the quantizer's math. Run `python runtime_mlx/make_ref.py` first, which writes 1.6 GB of reference files.

## What this repository leaves out

- **Model weights.** Everything downloads from Hugging Face. The Qwen3.5-4B jobs expect that model in `models/Qwen3.5-4B` on the volume.
- **Benchmark copies.** `eval_tasks.py` downloads its benchmarks. On an offline machine it reads exported copies from `data/bench/`, which are not here. The Modal images run offline, so cache the benchmarks on the volume first.
- **The script that assembled `site/data.json`.** It was a set of one-off steps. The source numbers are all in `results/`.
- **The rest of the LookAtMyQuant 2 recipe.** Token-weighted calibration, the QEP strength α, the per-row bit allocation, the fine-tuning and the math benchmark are not here. Neither are the result files behind `site2/data.json`, or the script that writes the page's numbers from them.
