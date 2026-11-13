# LookAtMyQuant

LookAtMyQuant is a quantization compiler. It measures how much each stored piece of a model hurts the output when compressed. It then picks a code for each piece so the whole model fits a size budget with the least damage.

On Gemma 4 E2B, its GGUF has 12× lower KL than Q4_K_M at the same file size. The method and every result are on [the site](https://lookatmyquant.pages.dev).

The code is under the MIT license.

## LookAtMyQuant 2: Qwen3.5-4B in 1.2 GB

[LookAtMyQuant 2](https://lookatmyquant2.pages.dev) takes Qwen3.5-4B to 1.23 GB, 2.3 bits per weight. The file solves 330 of 540 exact-answer math problems. Prism ML's Bonsai 2 Ternary 4B solves 328 at 1.20 GB, and the original solves 359. Its KL from the original on held-out text is 0.45 nats, against 0.59 for Bonsai.

The recipe is post-training quantization with trellis codes, calibrated on the model's own worked math solutions with digit tokens weighted 10×, and with bits allocated per row by Fisher information. 4,000 steps of distillation follow, in which every layer is re-quantized once from the original weights. The page has the findings, the full recipe and its cost, about $34 of GPU time.

The recipe's code is here, and the next section runs it. The 1.23 GB file itself is the [`lm2-qwen35-4b-1.2gb` release](https://github.com/ShinMegamiBoson/lookatmyquant/releases/tag/lm2-qwen35-4b-1.2gb), packed for the Apple Silicon runtime.

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

## Run the LookAtMyQuant 2 recipe on Qwen3.5-4B

Each step is a named job; `python run_batch.py --list` prints its command. The jobs need Qwen3.5-4B in `models/Qwen3.5-4B` on the volume and the text mix (`modal run --detach modal_fetch.py --text`). Every file they make goes to `lm2/` on the volume.

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
| `lm2Text` | `other_text.py` | The original answers science, code, instruction and tool prompts, and MATH and GSM8K training problems. This also downloads MATH | |
| `lm2Calibration` | `math_calibration.py` | The calibration set: the original's greedy solutions to 384 MATH training problems, in 2,048-token windows | 15 minutes, H100 |
| `lm2Solutions` | `checked_solutions.py` | The math text the fine-tuning distils on: solutions to MATH problems and to `math_bench.py` problems from another seed, with wrong calculations corrected and only right answers kept | 33 minutes, H100 |
| `lm2RowImportance` | `row_importance.py` | Each output row's Fisher information at digit-predicting positions | 1 minute, H100 |
| `lm2Build` | `quantize.py` | Post-training quantization: trellis codes along rows, digit tokens weighted 10× in the Hessians, QEP at α = 0.5, bits per row from the row plan, e-means codebooks | 37 minutes, H100 |
| `lm2Finetune` | `finetune.py` | 4,000 steps of distillation on latent weights pulled toward their codes, with every layer re-quantized once from the original weights | 2.7 hours, B200 |
| `lm2Math`, `lm2Math216` | `math_eval.py` | The 540-problem benchmark and a second set of 216, for the original, Bonsai, the build and the fine-tuned build | 20 minutes a model, H100 |
| `lm2TextKL` | `text_kl.py` | KL from the original on WikiText-2 and on the text mix's 13 sources | 1 minute a model, H100 |

`lm2BuildSmoke`, `lm2FinetuneSmoke` and `lm2MathSmoke` run the same code on two layers, 60 steps and 36 problems, in a few minutes each.

**The row plan.** `lm2Build` reads `results/compiler/rows/q35_rowplan_1p2GB.json`, which gives 1 to 4 bits to each of 1,520 row groups. `row_allocation.py` made it. Its `build-control` command builds and packs a control with a fixed split of rows, `prepare-rows` allocates rows inside the control's packed bytes with `rowgroup_alloc.py`, and `build-rows` builds under the plan and checks the packed size against it.

**The codebooks.** The four e-means codebooks the build decodes with are in `cache/codes/`. `--trellis-fit lloyd` gives the earlier ones, and a plan made before the change keeps them unless the flag says otherwise.

**Where these jobs differ from our run.** Our GSM8K solutions were written earlier, with thinking always off. Our text KL on 13 sources used 52 windows picked from the text mix by a separate tool; `--sources mix` takes the mix's own evaluation windows, on which our file scores 0.44 where the page reports 0.45.

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

## A drafter makes the 1.2 GB file generate 1.5 to 2 times faster on Apple silicon

`runtime_mlx/fast_q35.py` decodes the packed 1.2 GB Qwen3.5-4B file one token at a time, straight from its trellis codes. `runtime_mlx/drafter_q35.py` adds speculative decoding. A small drafter proposes the seven tokens after the model's latest one in a single pass. The model then checks all eight positions in one pass over its coded weights (`runtime_mlx/block_q35.py`) and keeps the proposals that match its own choice, so the text is exactly its own greedy text.

On an M5 Max, over 30 held-out prompts of up to 256 tokens, generation with the drafter was 2.0 times as fast: 176 tokens a second against 88, timed in alternation in one process on 2026-11-11. A check kept 4.3 tokens on average. Over eight everyday prompts it was 1.5 times as fast. The gain follows how predictable the text is: 2.5 times on algebra, 2.0 on code, 1.2 on chat and 1.05 on a story, and one of the 30 held-out prompts ran slower with the drafter. The text matched plain greedy decoding on every prompt. These are speeds after the first token; reading the prompt takes about 0.1 s either way.

Two later changes are in this code. Each was measured against the version before it on 40 paired generations: proposing from the 96,033 most common tokens was 6.0% faster, and picking the winning token inside the packed product was 5.3% faster.

The drafter is a DSpine model, trained with [SpecForge](https://github.com/sgl-project/SpecForge) from the published DFlash drafter for Qwen3.5-4B on the 1.2 GB file's own greedy answers. `modal_drafter.py` runs the training as one Modal job, and it cost $42. On 300 held-out prompts the trained drafter gets 4.65 tokens per check, against 4.30 for the stock DFlash drafter.

`pack_drafter.py` stores the drafter at 4 bits a weight in the build's own code, with no calibration data: 464 MB against 1.83 GB at 16 bits. On 60 held-out prompts the 4-bit drafter kept 4.35 tokens per check against 4.38, and the text was the same. `drafter_q35.py` loads either form and writes the packed one out dense as it loads. The 4-bit drafter is the [`lm2-drafter-4bit` release](https://github.com/ShinMegamiBoson/lookatmyquant/releases/tag/lm2-drafter-4bit). It runs beside the packed 1.2 GB file, the [`lm2-qwen35-4b-1.2gb` release](https://github.com/ShinMegamiBoson/lookatmyquant/releases/tag/lm2-qwen35-4b-1.2gb).

```sh
gh release download lm2-qwen35-4b-1.2gb --repo ShinMegamiBoson/lookatmyquant -D q35-1.2gb
gh release download lm2-drafter-4bit --repo ShinMegamiBoson/lookatmyquant -D drafter-4bit
python runtime_mlx/bench_drafted.py q35-1.2gb drafter-4bit
```

The benchmark defaults to linear verification. Use `--mode chat-fork` to try a second candidate at an ambiguous early position, or `--mode copy-wide` for known contiguous copies. Copy mode uses the source cursor and tunes the gate/up and down projections for 16-slot checks. Both modes verify all emitted tokens with the packed target. To benchmark the chat fork, run:

```sh
python runtime_mlx/bench_drafted.py q35-1.2gb drafter-4bit --mode chat-fork --prompts chat
```

For a copy benchmark, pass `--mode copy-wide --heldout path/to/copy-prompts.json`. The JSON entries carry `source`, `prompt_len`, and `input_ids` from the packed model's tokenizer.

The checks, and the packer for a drafter of your own:

```sh
python runtime_mlx/test_fast_q35.py
LAMQ_Q35_PACKED=path/to/packed python runtime_mlx/test_block_q35.py
python runtime_mlx/check_drafter.py path/to/packed path/to/drafter
python runtime_mlx/bench_drafted.py path/to/packed path/to/drafter --heldout path/to/drafter/heldout/heldout.json --take 30
python pack_drafter.py path/to/drafter path/to/drafter-4bit
```

`runtime_mlx/pack_dense.py` writes the packed file from the fine-tuned build's dense weights, and `runtime_mlx/modal_pack.py` runs it on Modal. `check_drafter.py` compares the drafter in MLX with SpecForge's own model on reference blocks that `drafter/reference.py` writes.

## Where everything is

| Path | What it holds |
|---|---|
| `compiler.py` | Measure and allocate |
| `compiler_arch.py` | How the compiler finds a model's layers, tables and stages, and the text mix |
| `quantize.py` | Builds a program or a plan, then scores it |
| `row_importance.py`, `rowgroup_alloc.py`, `row_allocation.py`, `compiler_artifact.py` | Bits per row: each row's importance, the allocation, the builds it compares, and the packed format that prices them |
| `math_calibration.py`, `checked_solutions.py`, `other_text.py`, `calc_check.py`, `token_kinds.py` | The text the recipe calibrates and fine-tunes on, the checker for stated calculations, and the kind of token each position predicts |
| `finetune.py`, `finetune_lib.py`, `fast_delta.py` | Fine-tuning that ends on the codes, and a faster linear-attention rule for Qwen3.5 |
| `math_bench.py`, `math_eval.py`, `text_kl.py` | The exact-answer math benchmark, its runner for the Qwen3.5 family and Bonsai, and KL from the original on text |
| `gguf_compiler.py`, `public_eval.py`, `eval_tasks.py` | The llama.cpp route, public quantizations and task benchmarks |
| `lamq_export.py`, `runtime_mlx/` | The export format and the Apple Silicon runtime |
| `runtime_mlx/fast_q35.py`, `block_q35.py`, `block_argmax_q35.py`, `pack_dense.py`, `modal_pack.py` | The runtime for the packed 1.2 GB file: one token at a time, a block of 8 or 16 tokens in one pass, and the packer |
| `runtime_mlx/drafter_q35.py`, `draft_vocab_q35.py`, `check_drafter.py`, `bench_drafted.py`, `drafter/`, `modal_drafter.py`, `pack_drafter.py` | Speculative decoding with a DSpine drafter, the drafter's training, and the drafter stored at 4 bits |
| `opgraph/` | The grammar: a model as a tensor network, and the rewrites and codes that act on it |
| `run_completeness.py`, `make_report.py` | The check that nine published methods are reproduced exactly |
| `analyze_compiler.py` | The tables of plan results |
| `run_batch.py`, `modal_app.py`, `modal_gguf.py`, `modal_fetch.py` | Named jobs and the Modal runners |
| `results/compiler/plans/`, `results/compiler/rows/` | Every plan, and the row plan for Qwen3.5-4B at 1.2 GB |
| `results/protocols/` | The second math set of 216 problems, and the hashes of every question kept out of training text |
| `cache/codes/` | The e-means trellis codebooks at 1 to 4 bits |
| `results/batch/`, `results/gguf/` | The measurements and plan runs behind the site |
| `results/COMPILER.md`, `results/COMPLETENESS.md` | Write-ups of the compiler on SmolLM2-360M and Qwen3.5-4B, and of the reproduction check |
| `site/` | The website. Run `python3 -m http.server 8000` inside it, then visit http://localhost:8000 |
| `site2/` | The LookAtMyQuant 2 page. It runs the same way |
| `tests/` | Tests of the grammar and the codes |

## The tests need SmolLM2-360M and WikiText-2

Each test file runs on its own from the repository root, and all six finish in under a minute.

```sh
python tests/test_frame.py
python tests/test_grammar.py
python tests/test_codes.py
python tests/test_elements.py
python tests/test_nested.py
python tests/test_rowgroup_alloc.py
```

`runtime_mlx/test_core.py` checks the runtime against the quantizer's math. Run `python runtime_mlx/make_ref.py` first, which writes 1.6 GB of reference files.

`runtime_mlx/test_fast_q35.py` and `runtime_mlx/test_block_q35.py` check the Qwen3.5 kernels against NumPy products of independently unpacked codes, with no model. When `LAMQ_Q35_PACKED` names a packed file, the second also checks a block of tokens against decoding them one at a time.

## What this repository leaves out

- **Model weights.** Everything downloads from Hugging Face. The Qwen3.5-4B jobs expect that model in `models/Qwen3.5-4B` on the volume.
- **Benchmark copies.** `eval_tasks.py` downloads its benchmarks. On an offline machine it reads exported copies from `data/bench/`, which are not here. The Modal images run offline, so cache the benchmarks on the volume first.
- **The script that assembled `site/data.json`.** It was a set of one-off steps. The source numbers are all in `results/`.
- **LookAtMyQuant 2's dense weights and result files.** The 1.2 GB file is a release in the packed format. The dense weights the benchmarks ran on are not published, and neither are the per-problem results behind `site2/data.json` or the script that writes the page's numbers from them.
- **The drafter at 16 bits.** The drafter as trained is 1.8 GB and is not published. The 4-bit one is a release file, and `modal_drafter.py` trains it.
