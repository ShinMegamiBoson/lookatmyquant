# The compression compiler, first version

A plan gives every stored piece of a model its own program. The compiler finds the plan that loses least under a hardware target. Everything here is SmolLM2-360M; perplexity is WikiText-2 test (original 15.31) and KL is to the original model, both on 64 test windows of 512 tokens unless a column says 2k.

Code:
- `compiler.py`: `measure`, `allocate`, `uniform`, `shuffle`.
- `quantize.py --program plan --plan FILE`: runs a plan; `--gauges`, `--seeds`.
- `analyze_compiler.py`: the tables.
- Jobs `ccSens*`, `cc_*`, `ccG_*`, `ccS4_*` in `run_batch.py`.
- Plans in `results/compiler/plans/`.

## How it works

1. **Items.** The knapsack's rows:
   - each of the 224 matrices, with options tr2, tr3, tr4 (trellis), lm4, lm5, lm6, lm8 (Lloyd–Max) and fp16. Every code uses incoherence and LDLQ, as the uniform programs do;
   - each of 9 buckets of the tied embedding / output table, rows ranked by token frequency in the training split (256, 256, 512, …, 16384, 16384 rows), with options tr1–tr4, lm5, lm6, lm8 and fp16. A bucket gets a randomized Hadamard over the hidden dimension only, so a lookup still decodes one row;
   - each layer's cached keys and values, at 2, 3, 4, 5, 6, 8 or 16 bits per value. The cache is simulated: each token's vector per head is rotated, scaled by its RMS (16 bits per head), and rounded to Gaussian Lloyd–Max levels. The first position stays exact.
2. **Measure.** Each item alone at each option: quantize only that item, measure the KL to the original model on 16 held-out training windows (never the test split). This is 2,312 measurements, about 50 L40S-minutes in 9 parallel jobs.
3. **Price.** A target prices an option from three numbers: stored bits, cached bits per token of context, and decode operations. The decode operations are assumed (1 per weight for 16-bit, 2 for a scalar code, 3 for the trellis), not measured with kernels. The price is stored/batch + decode ops·price/batch + context·cache bits.

   | Target | Batch | Context | Decode price |
   |---|---|---|---|
   | disk | – | – | 0 |
   | laptop | 1 | 2,048 | 0 |
   | server | 32 | 16,384 | 0 |
   | npu | 1 | 2,048 | 2 bits per operation |

4. **Allocate.** Dynamic programming over the items (a multiple-choice knapsack) finds the options with the least summed KL whose price is within the budget. Summing assumes damage adds up; the runs test that.
5. **Run.** The harness quantizes sequentially to the plan, with cross-term targets so later matrices absorb earlier error, then evaluates on the test split.

## What the measurements say

- **Per weight, `v_proj` is the most sensitive matrix.** Summed KL per million weights at 3-bit trellis:

  | v | k | down | up | gate | o | q |
  |---|---|---|---|---|---|---|
  | 0.0016 | 0.0005 | 0.0004 | 0.0004 | 0.0002 | 0.0002 | 0.0002 |

- **By layer, the last layer stands out.** Layer 31 costs 0.0123 at 3 bits, against 0.0018–0.0023 for layers 14–21 and about 0.004 for layers 0–3. `31.mlp.up_proj` alone costs 0.0066.
- **Each bit removed multiplies a matrix's KL by about 3.9** (median ratio of tr2 to tr3 3.99, tr3 to tr4 3.77, lm5 to lm6 3.89), what Gaussian rate–distortion predicts. At 4 bits, Lloyd–Max costs 1.8× the trellis.
- **KL is proportional to the matrix's weighted error.** For each matrix, KL ÷ tr(ΔW H ΔWᵀ)/tr(W H Wᵀ) varies by only 1.1× between tr2 and lm5 (median; 1.37× at the 90th percentile). This is HIGGS's linearity theorem, matrix by matrix. One width per matrix, plus each code's error, would predict the rest: about a seventh of the measurement.
- **The embedding is expensive at 3 bits.** The whole table at 3 bits costs 0.040 KL, against 0.108 for all 224 matrices. The 256 most frequent rows (58% of tokens) cost 0.0165 alone. From the most frequent bucket to the rarest, KL per row falls 900×, while frequency per row falls 34,000×. Rare rows are cheap but not free: the output head reads every row at every position.
- **In the cache, keys are 7× more sensitive than values.**

  | Cache width | All keys | All values |
  |---|---|---|
  | 4 bits | 0.0255 | 0.0037 |
  | 8 bits | 0.0001 | under 0.0001 |

  At 8 bits the cache is near lossless.

## Step 1: a width per matrix

Linears only; the embedding and the cache stay at 16 bits. Each cell is perplexity, with KL in brackets.

| Bits per weight | Uniform trellis | Compiler | Same widths, shuffled within each role across layers |
|---|---|---|---|
| 2.05 | 23.46 (0.439) | – | – |
| 2.25 | – | 20.50 (0.308) | – |
| 2.50 | – | 18.63 (0.213) | 19.82 (0.267) |
| 3.04 | 16.72 (0.0959) | 16.64 (0.0926) | 16.98 (0.1163) |
| 3.50 | – | 15.95 (0.0484) | – |
| 4.04 | 15.64 (0.0237) | 15.61 (0.0226) | – |

At 3.04 bits with only trellis codes on the menu: 16.74 (0.0927).

- **At the uniform programs' own rates, per-matrix widths cut KL by only 3–5%.** Incoherence, LDLQ and the cross term already even out most of the damage. The outliers left are the last layer and `v`.
- **Between rates, the compiler fills in the frontier but does not beat it.** At 2.5 bits its KL of 0.213 is what the log-KL line between uniform 2 and 3 bits gives (0.220).
- **Which layer matters more than which role.** The shuffled plan keeps the per-role widths and moves them to random layers. It is worse than uniform (+21% KL at 3.04 bits).
- **Code family makes no difference at 3 bits.** The trellis-only menu gets the same KL (0.0927 against 0.0926).

## Steps 2 and 3: every item, under a target

Each compiler plan costs what the reference costs under that target's price. Cells are perplexity / KL; the laptop, server and npu rows are measured on 64 windows of 2,048 tokens.

| Target | Reference | Reference | Compiler | Predicted KL | What the compiler moved |
|---|---|---|---|---|---|
| disk | tr3 matrices, 16-bit embedding | 16.72 / 0.0959 | **15.48 / 0.0128** | 0.0133 | embedding 16 → 4.8 bits (lm8 on the 2k most frequent rows, tr4 on the 33k rarest); matrices 3.0 → 4.7 bits |
| disk | tr3 matrices, tr3 embedding | 17.40 / 0.1376 | **16.81 / 0.1068** | 0.1005 | embedding by frequency: lm8, lm6, lm5, lm5, tr4, tr4, tr3, tr3, tr2 |
| laptop | tr3, 16-bit embedding and cache | 12.25 / 0.0979 | **11.22 / 0.0050** | 0.0049 | cache keys 8 bits, values 5–6; matrices 5.8 bits; embedding 5.6 bits |
| server | tr3, 16-bit embedding, 4-bit cache | 12.70 / 0.1323 | **11.44 / 0.0216** | 0.0204 | matrices 7.4 bits, since 32 sequences share each read; cache keys 4–5 bits, values 3 |
| npu | tr3, 16-bit embedding and cache | 12.25 / 0.0979 | **11.20 / 0.0012** | 0.0011 | no trellis: Lloyd–Max at 6–8 bits; cache 6–8 bits |

- **The large gains come from items a uniform format treats as fixed:** the 16-bit embedding (44% of a 3-bit model's bytes) and the 16-bit cache (at 2k context, 0.67 Gbit read per token, against 1.7 Gbit for all the weights).
- **Against a 3-bit embedding the gain is 22% of KL,** all from giving frequent rows more bits.
- **The same model gets four different plans,** one per target.

### Does the damage add up?

For every all-item plan, the measured KL is within 10% of the sum. For the linear-only plans:

| Plan | Predicted | Measured |
|---|---|---|
| compiler, 2.25 bits | 0.262 | 0.308 |
| compiler, 3.04 bits | 0.0883 | 0.0926 |
| compiler, 4.04 bits | 0.0240 | 0.0226 |
| uniform tr3 | 0.108 | 0.096 |
| uniform tr4 | 0.028 | 0.024 |

The error is systematic. The cross term repairs uniform plans better than predicted and mixed plans at low rates worse. Searching around the knapsack's plan with real evaluations, as EvoPress does, would target that gap.

## Step 4: exact rewrites as search moves

- **Gauges** (`--gauges vo,ud`, α ∈ {0, 0.1}, on uniform tr3x): 63 of 64 blocks keep α = 0, and the result is 16.735 (0.0987) against 16.721 (0.0959). In the smoke test, α = 0.25 raised layer 0's attention error 60× (0.0053 → 0.31), though each rewrite is exact before quantizing (5e-7). The randomized Hadamard on v's outputs spreads the quantization error evenly over channels the gauge made unequal, and o's compensating 1/d amplifies it in the small ones. On top of incoherence, diagonal gauges across a bond can only hurt.
- **Seeds** (`--seeds 4`: per matrix, the randomized Hadamard signs with the least H-weighted error out of 4, also exact): uniform tr2x 23.69 (0.449) against 23.46 (0.439); tr3x 16.76 (0.0960) against 16.72 (0.0959). Nothing. The per-matrix proxy picks among seeds whose errors differ by noise, and the choice does not carry to the model.
- **Neither move helps on top of incoherence.** The exact rewrites worth searching are those the code does not already do, such as the residual-stream rotation for entropy-coded codes.

## Qwen3.5-4B

Original perplexity 12.283 (64 test windows of 512). Qwen3.5-4B is a hybrid: three of every four layers are Gated DeltaNet, every fourth (3, 7, …, 31) is full attention, so only 8 layers keep a key/value cache (4 heads of 256). Its tied embedding is 248,320 rows × 2,560: 636M weights. At 16 bits that is about as many bytes as all the matrices at 3 bits. The DeltaNet's recurrent state stays at 16 bits and is not priced.

**Measurement at one width.** Every matrix was measured at tr3 only, except four spot layers measured at every option: DeltaNet layers 0 and 14 and attention layers 15 and 31. The big embedding buckets were measured at tr2 and tr3. Every other option, 890 in all, was predicted from the spot layers' ratios. 11 jobs took 1.6 H100-hours.

On SmolLM2, the same shortcut (spot layers 0, 15 and 30) finds a plan whose true summed KL is 0.01333, against 0.0133 from full measurement. The ratios belong to the codes, not the model:

| Model | tr2 / tr3 | tr4 / tr3 | lm5 / tr3 | lm6 / tr3 | lm8 / tr3 |
|---|---|---|---|---|---|
| SmolLM2 | 3.99 | 0.265 | 0.125 | 0.032 | 0.002 |
| Qwen3.5 | 4.09 | 0.262 | 0.126 | 0.033 | 0.002 |

**What the measurements say:**
- Sensitivity by layer is U-shaped: layers 0–4 and 31 cost most, layers 13–18 least.
- `down_proj` is the costliest role in total (0.019 of 0.068 at tr3). Per weight, `v_proj` is the most sensitive matrix again, about 7× the query or DeltaNet projections.
- The whole embedding at tr3 costs 0.017. The rarest 183k rows, never seen in WikiText's training split, cost 0.0007 together at 2 bits.
- In the cache, keys and values are about equally sensitive (at 4 bits, 0.0018 for keys and 0.0024 for values), unlike SmolLM2. In layer 31, values cost 6× keys.

**Results.** Cells are perplexity / KL; the 2k column is 64 windows of 2,048.

| Plan | Bits (matrices / embedding / cache) | Perplexity / KL | Predicted KL | 2k |
|---|---|---|---|---|
| uniform tr3x, 16-bit embedding (reference) | 3.06 / 16 / 16 | 12.884 / 0.0614 | 0.068 | 9.593 / 0.0702 |
| compiler, matrices only, same bits | 3.05 / 16 / – | 12.848 / 0.0592 | 0.060 | |
| **compiler, matrices + embedding, same bytes as the reference** | 5.35 / 3.10 / – | **12.299 / 0.0050** | 0.0048 | |
| uniform tr3x, tr3 embedding | 3.06 / 3.04 / – | 13.008 / 0.0735 | 0.085 | |
| **compiler, same bytes as that** | 3.17 / 2.41 / – | **12.801 / 0.0542** | 0.055 | |
| **compiler, laptop at 32k context, same bits read per token as the reference** | 6.62 / 4.23 / 6.19 | **12.298 / 0.0015** | 0.0014 | 9.140 / 0.0022 |

- **The embedding lever reproduces, and it is larger here.** At the bytes of "3-bit matrices + 16-bit embedding", the compiler's plan costs +0.13% perplexity instead of +4.9%: 12× less KL. Its embedding plan:

  | Rows (most frequent first) | Code |
  |---|---|
  | 256 | 16-bit |
  | next ~8k | 8-bit Lloyd–Max |
  | ~25k | 5–6-bit |
  | 33k | tr4 |
  | 66k | tr3 |
  | rarest 117k | tr2 |

  The matrices go to 5.35 bits.
- **It also beats uniform 5-bit at fewer bytes.** Uniform lm5x with a 16-bit embedding (KL 0.0083, `q35Lm5x`) uses 33% more bytes for more KL.
- **Against everything at 3 bits, KL falls 26%.** The predicted fall was 36%; the uniform plan beats its own prediction again, the cross term.
- **Matrices alone gain 3.6%** of KL, as on SmolLM2.
- **The prediction held:** the summed KL is within 4% of measured for every compiler plan.

Cost: 19 jobs (smoke tests, 11 measurement jobs, 6 runs), 6.9 H100-hours, about $33.

## Next

1. **Longer contexts and real caches.** The Qwen3.5 laptop plan is priced at 32k context but evaluated at 2k; evaluate at the priced context, and price the DeltaNet state.
2. **The codes the plans ask for.** Frequent embedding rows want 8 bits and the cache wants keys above values: kernels for mixed-width rows and asymmetric key/value widths.
3. **The additivity gap.** Use the knapsack plan as the start of a search with real evaluations (EvoPress), for the low-rate plans where the sum is 15–18% off.
4. **Decode work, measured.** Replace the assumed operation counts with kernel timings per format and device.

Cost: 31 jobs, 6.8 L40S-hours, about $18.
