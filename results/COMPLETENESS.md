# Completeness check

SmolLM2-360M in fp32. Calibration: 64 random 512-token windows of WikiText-2 train. Evaluation: the first 64 512-token windows of WikiText-2 test (32,768 tokens). Original perplexity: **15.3147**.

For each method, the reference implementation runs layer by layer; at every layer the operator-word version is applied to the identical layer state and inputs and compared. Then the operator-word version runs end to end with its own calibration. *Agreement* is the minimum over layers and matrices of the fraction of identical integer codes, mask entries, or chosen arguments (AWQ's scale ratios and clip bounds). *Rel. weight diff* is the largest relative Frobenius difference of any weight matrix between the two versions at any layer.

| Method | Operator word | Agreement | Rel. weight diff | PPL reference | PPL operator word | KL vs original (ref / word) | Max logit gap | Bits / weight |
|---|---|---|---|---|---|---|---|---|
| RTN w4 g64 | Split(index) → Split(scale) → Restrict(ℤ) → Split(offset) → Restrict(grid) | 100.000% | 0.0e+00 | 17.9178 | 17.9178 | 0.1647 / 0.1647 | 0.0e+00 | 4.31 |
| RTN w3 g64 | Split(index) → Split(scale) → Restrict(ℤ) → Split(offset) → Restrict(grid) | 100.000% | 0.0e+00 | 45.4496 | 45.4496 | 1.1126 / 1.1126 | 0.0e+00 | 3.30 |
| GPTQ w4 g64 | sweep[ Split(scale, offset) per group; Restrict(column) → Refit(rest) ] → RTN word | 100.000% | 0.0e+00 | 16.6588 | 16.6588 | 0.0892 / 0.0892 | 0.0e+00 | 4.31 |
| GPTQ w3 g64 | sweep[ Split(scale, offset) per group; Restrict(column) → Refit(rest) ] → RTN word | 100.000% | 0.0e+00 | 21.1618 | 21.1618 | 0.3180 / 0.3180 | 0.0e+00 | 3.30 |
| GPTQ w3 g64, official blocksize 128 (reference only) | as GPTQ w3 g64, but the reference uses the official default blocksize 128 > group size 64 | 77.265% | 3.1e-01 | 20.8067 | 21.1618 | 0.3141 / 0.3180 | 1.6e+01 | 3.30 |
| AWQ w4 g64 | Gauge(diag, folded) → Restrict(box) → RTN word | 100.000%, final codes 100.000% | 0.0e+00 | 16.7094 | 16.7094 | 0.0963 / 0.0963 | 1.2e-04 | 4.31 |
| AWQ w3 g64 | Gauge(diag, folded) → Restrict(box) → RTN word | 100.000%, final codes 100.000% | 0.0e+00 | 25.1789 | 25.1789 | 0.4805 / 0.4805 | 0.0e+00 | 3.30 |
| SVD keep 0.8 | Split(identity) → Split(SVD) → Truncate → Contract | n/a (no discrete choices) | 0.0e+00 | 890636.8443 | 890636.8443 | 11.0472 / 11.0472 | 0.0e+00 | 12.79 |
| SVD-LLM keep 0.8 | Split(identity) → Gauge(whitening) → Split(SVD) → Truncate → Contract | n/a (no discrete choices) | 0.0e+00 | 40.2549 | 40.2549 | 1.0075 / 1.0075 | 0.0e+00 | 12.79 |
| Magnitude 50% | Mask(|W|) | 100.000% | 0.0e+00 | 2078.2701 | 2078.2701 | 4.8643 / 4.8643 | 0.0e+00 | 8.98 |
| Wanda 50% | Mask(|W|·√diag H) | 100.000% | 0.0e+00 | 25.5225 | 25.5225 | 0.5091 / 0.5091 | 0.0e+00 | 9.00 |
| SparseGPT 50% | sweep[ Mask per block; Mask(column) → Refit(rest) ] | 100.000% | 1.1e-10 | 21.9255 | 21.9255 | 0.3868 / 0.3868 | 1.9e-06 | 9.00 |
| SliceGPT 25% | Untie → Contract(norm gains) → per segment: Gauge(canonical, group O) → Truncate | n/a (no discrete choices) | 0.0e+00 | 78.8600 | 78.8600 | 1.6906 / 1.6906 | 0.0e+00 | 13.69 (whole model: 94.0% of params) |
