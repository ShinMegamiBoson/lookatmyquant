# Keeping `runtime_cuda` in sync with `runtime_mlx`

`runtime_cuda/inkling/` ports `runtime_mlx/inkling/` (the author's Apple Silicon runtime) to NVIDIA GPUs. We port
**behaviour** (formats, model semantics, decoding algorithms), not Metal speed work: correctness here is anchored to
numpy/PyTorch references and transformers, so upstream kernel tuning never forces work on this side.

**Synced through upstream commit:** `f9f8008` (2026-11-22, SQYFR v3 trellis row groups and FP16 input column scales).

When you sync, update the commit above, then run `python runtime_cuda/upstream_sync.py` again: it should list
nothing new.

## How to sync

1. `python runtime_cuda/upstream_sync.py --fetch` lists upstream commits since the commit above, each changed
   file sorted into *behaviour* (port it) or *Metal speed* (read it, usually nothing to do), with the file(s) here
   that correspond.
2. Merge upstream into the branch (`git merge origin/main`).
3. Run `python -m runtime_cuda.inkling.schema <packed dir>` on any new released file. It refuses unknown format ids,
   tensors, tables, manifest formats and config options, and names what is new.
4. `pytest runtime_cuda/`. `test_upstream_fixture.py` runs the author's own synthetic model
   (`runtime_mlx/test_inkling.py::build_mini_packed_dir`) through this runtime, so fixture additions upstream show up
   here as failures without copying code.
5. For a new released file: `python -m runtime_cuda.compare <dir> --ref-only --save cache/oracle/<size>`,
   add it to `BUILDS` in `test_model_oracle.py`, and run `python -m runtime_cuda.kl_eval` against the original model.

## File map

| Here (`runtime_cuda/inkling/` unless noted) | Upstream (`runtime_mlx/inkling/` unless noted) | What |
|---|---|---|
| `formats.py` | `kernels.py` (numpy references in `_selftest`, Hadamard factors, `nvfp4_luts_mx`), `model.py` (`ID_TO_FMT`, `FMT_PARAMS`, `NVFP4_FID`) | code formats, reference decoders |
| `schema.py` | `model.py` loaders (`PackedMoE`, `PackedDecoderLayer`, `PackedEmbed`, `PackedLMHead`) | what a packed file may contain |
| `gemv.py` | `kernels.py` half-warp kernels, `fast_kernels.py` `UnifiedMoESide`, `DenseStage`, `build_fast_luts` / `build_rtr_multitable` | one-token decode matmuls |
| `triton_kernels.py` | `kernels.py` `trellis_dense_matmul`, `fast_kernels.py` MoE prefill kernels | prefill / grouped matmuls |
| `model.py` | `model.py` (`Inkling2Config`, `Inkling2Cache`, `PackedMoE`, `PackedDecoderLayer`, `Inkling2LoopPackedMLX`) | the model |
| `packed_modules.py` | `model.py` `PackedLinear`, `PackedLMHead` | packed attention linears and LM head |
| `attention.py` | `fast_kernels.py` `gqa_relpos_sinks_sdpa_fast`, `model.py` `_gqa_sdpa_sinks` | attention + relative bias + sinks |
| `conv.py` | `fast_kernels.py` `short_conv4_res_fused`, `kv_sconv4_norm_fused`, `kv_dyn_sconv4_norm_fused`; `model.py` `short_conv`, `dynamic_short_conv` | causal 4-tap convolutions |
| `fused.py` | `fast_kernels.py` `block_attn_res_fused`, `msg_blend_norm*` | attention residuals, message gates |
| `rht.py` | `fast_kernels.py` `rht_in_2304_fast`, `rht_out_*_fast` | Hadamard rotations |
| `generate.py` | `generate.py` `stream_serial`, `generate_serial` | serial greedy decoding |
| `../test_upstream_fixture.py` | `../test_inkling.py` `build_mini_packed_dir` | the author's synthetic model, reused |
| `reference.py`, `teacher.py`; `../compare.py`, `../kl_eval.py`, `../check_weights.py` | none (`../compare.py`, `../check_reference.py` play this part upstream) | transformers-based float32 reference, the original model, evaluation |
| `tuning.py`, `tuning/`; `../tune.py`, `../tuning_coverage.py` | none | per-GPU kernel settings |

## Upstream behaviour not ported yet

| Upstream | Where | Notes |
|---|---|---|
| Tree, chat-fork and copy-wide speculative decoding | `generate.py` `stream_dspark`, `fast_kernels.py` `make_tree_spec` | The native CUDA path implements linear DSpark verification |
| Top-p and top-k sampling | `generate.py` `_sample_*` | CUDA supports temperature sampling and positional seeded draws for native DSpark |
| margin verify, N-gram prompt lookup | `generate.py` `verify_margin_greedy_gpu`, `NgramDraftIndex` | opt-in upstream |
| dynamic MoE top-k (`moe_k_max`, `moe_k_min`, `moe_ratio_thr`, `moe_renorm`) | `model.py` `Inkling2Config`, `fast_kernels.py` `sigmoid_top8_norm_fast` | opt-in upstream; default (top-8) is what we run |

## Known differences between upstream and transformers (guarded by `schema.py`)

Found by running the author's fixture against transformers. The released files avoid all three; a file that does not
is refused rather than run with one definition or the other.

- Sliding layers' position-bias range: `runtime_mlx` uses `rel_extent`, transformers `sliding_window_size`.
- Which layers slide: `runtime_mlx` reads `local_layer_ids`, transformers `layer_types` (default: every 6th layer full).
- Sliding layers' head shape: transformers reads `swa_num_attention_heads` / `swa_num_key_value_heads` / `swa_head_dim`.
