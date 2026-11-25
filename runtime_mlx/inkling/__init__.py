"""Apple Silicon MLX runtime for packed Inkling-2-Mini models and 4-bit DSpark speculative decoding."""

from .batch import BatchSequence, ContinuousBatchScheduler, PagedInklingPool, forward_batched, generate_batch
from .drafter import DSparkDrafterMLX, quantize_mtp_to_4bit, resolve_drafter_files
from .generate import PromptCache, generate_dspark, generate_serial, stream_dspark, stream_serial
from .model import Inkling2Cache, Inkling2Config, Inkling2LoopPackedMLX, Inkling2MLXModel, load_inkling

__all__ = [
    "BatchSequence",
    "ContinuousBatchScheduler",
    "DSparkDrafterMLX",
    "Inkling2Cache",
    "Inkling2Config",
    "Inkling2LoopPackedMLX",
    "Inkling2MLXModel",
    "PagedInklingPool",
    "PromptCache",
    "forward_batched",
    "generate_batch",
    "generate_dspark",
    "generate_serial",
    "load_inkling",
    "quantize_mtp_to_4bit",
    "resolve_drafter_files",
    "stream_dspark",
    "stream_serial",
]
