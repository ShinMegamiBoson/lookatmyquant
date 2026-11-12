"""Every setting of the drafter pipeline. `Settings()` is the real run and `Settings().smoke()` the same run at toy size."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


def COMMIT() -> None:
    """Persist ROOT's volume. Does nothing here; a launcher replaces it (`drafter.settings.COMMIT = volume.commit`). The
    pipeline calls it after every checkpoint copy and at the end of every stage."""


@dataclass(frozen=True)
class Settings:
    # The two targets: the original model directory, and the build's weights for the same architecture.
    original_dir: str = "/root/.cache/huggingface/models/Qwen3.5-4B"
    build_weights: str = "/root/.cache/huggingface/lm2/tuned/dense.pt"
    dflash_repo: str = "z-lab/Qwen3.5-4B-DFlash"

    # Prompts: the first that loads of (dataset, split); its download cache stays off ROOT.
    datasets: tuple = (("mlabonne/open-perfectblend", "train"), ("HuggingFaceH4/ultrachat_200k", "train_sft"))
    datasets_cache: str = "/tmp/drafter_datasets"
    prompt_seed: int = 20261111
    train_prompts: int = 20000
    heldout_prompts: int = 300
    prompt_tokens: int = 512

    # Text the targets write.
    generate_batch: int = 64
    max_new_tokens: int = 512
    shard_prompts: int = 512

    # Features (local disk, rebuilt in a fresh container) and the sample length the trainer reads.
    features_dir: str = "/tmp/drafter_features"
    feature_batch: int = 32
    feature_writers: int = 8
    max_length: int = 1024

    # The drafter. These must equal what the downloaded DFlash config says; `init` checks each one.
    block_size: int = 16
    mask_token_id: int = 248077
    target_layer_ids: tuple = (1, 5, 9, 13, 17, 21, 25, 29)
    layer_types: tuple = ("sliding_attention",) * 5 + ("full_attention",)
    sliding_window: int = 4096
    intermediate_size: int = 9216
    dspine: dict = field(default_factory=lambda: {  # SpecForge PR 1038, configs/qwen3-8b-dspine.json
        "transfer_rank": 1024,
        "message_size": 512,
        "top_k": 16,
        "ce_weight": 0.1,
        "alignment_weight": 0.5,
        "alignment_warmup_steps": None,
        "replacement_layers": 3,
        "replacement_probability": 0.5,
        "replacement_hold_ratio": 1 / 6,
        "replacement_end_ratio": 1 / 3,
        "candidate_mass_threshold": 1e-4,
    })

    # Training. SpecForge writes to train_dir and reads the target's embedding from target_dir, both local.
    train_dir: str = "/tmp/drafter_train"
    target_dir: str = "/tmp/drafter_target"
    run_id: str = "q35-dspine"
    embedding_key: str = "model.embed_tokens.weight"
    lm_head_key: str = "lm_head.weight"
    num_epochs: int = 3
    batch_size: int = 8
    accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.04
    max_grad_norm: float = 1.0
    # Draft blocks per sequence a step. The paper's 512 is for sequences up to 3,072 tokens; the build's answers average 388
    # tokens, so 512 made every answer position an anchor every step: 24 s a step on a B200, 12 hours for the run (2026-11-11).
    # 160 is about two in five positions a visit, the paper's coverage over its epochs
    num_anchors: int = 160
    loss_decay_gamma: float = 7.0
    objective_chunk_blocks: int = 16
    attention_backend: str = "flex_attention"
    log_interval: int = 10
    save_interval: int = 100
    local_checkpoints: int = 3
    train_seed: int = 42
    step_cap: int | None = None  # stop after this many optimizer steps; None trains every epoch

    # Scoring.
    score_blocks: tuple = (16, 8)
    score_anchors: int = 128  # draft blocks per drafter forward
    suspect_below: float = 2.5  # the stock drafter on the original model must accept more than this per check

    def smoke(self) -> "Settings":
        return replace(
            self,
            train_prompts=64,
            heldout_prompts=16,
            generate_batch=8,
            max_new_tokens=64,
            max_length=256,
            batch_size=2,
            accumulation_steps=1,
            num_anchors=64,
            save_interval=4,
            step_cap=12,
            features_dir=self.features_dir + "_smoke",
            train_dir=self.train_dir + "_smoke",
            target_dir=self.target_dir + "_smoke",
        )
